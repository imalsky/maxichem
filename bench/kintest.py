"""Fast test runner for the GPU chemistry solver against VULCAN truth on NCHO_photo; misses first.

The tiers and what they score are in docs/testing.md. Exit status 1 when a path or box has a miss, a call is unfinished
or leaves a non-finite or negative state, or a cell still fails after the recovery.

    python bench/kintest.py quick|full [--save out.npz] [--ref ref.npz]

--save writes per-path summaries, the 12 gases of every cell at intervals and at the end, the per-box results, timings
and the run's commit and settings (it refuses to overwrite). --ref compares with a saved run of the same tier and the
same truth.
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from kintest_truth import FLOOR, HOLD, TINY, check_jobs, load_p, load_paths
from run_record import run_record

from maxichem import kincuda, kinvulcan, paths
from maxichem._checks import GASES

MISS, LARGE = 0.1, 1.0
DAY = 86400.0
READ = 1000  # ticks between read-backs of the counters
CHUNK = 65536  # most boxes per chunk
CAP_H = 1.5  # full: stop above this estimate (hours): no test or check run may take longer
ESTIMATE_TICK = 12 * READ  # full: make the estimate here: the QC family and the holds, the dearest ticks, are over
START_STEP = 1e-10  # s: the first call's h
QUICK_BOXES = 32768  # the boxes of the quick tier
META = ("id", "panel", "kind", "stratum", "start", "comp")
BOX_KEYS = ("box", "X", "T", "P", "J", "lit", "kind")
GUARD_KEYS = ("bad_input", "nonfinite", "unfinished", "drift", "recomputed", "recovered", "still_failed")
NEVER = 1 << 62


@dataclass(frozen=True)
class ScoredGases:
    """The scored gases in a state: their columns, on the host and the device, and their log10 floors on the device."""

    columns: list[int]
    index: torch.Tensor
    floor: torch.Tensor

    def floored_log(self, fractions):
        """Return the floored log10 of mole fractions of these gases alone: [cells, gases]."""
        return torch.log10(fractions.clamp(min=TINY)).maximum(self.floor)

    def snapshot(self, x):
        """Return the floored log10 of these gases in a whole state, as float32 on the host."""
        return self.floored_log(x[:, self.index]).float().cpu().numpy()


class ErrorTally:
    """Per cell, over the states scored so far: the worst floored error, its call and gas, and the counts."""

    def __init__(self, cells, device):
        self.worst = torch.zeros(cells, dtype=torch.float64, device=device)
        self.worst_call = torch.full((cells,), -1, device=device)
        self.worst_gas = torch.full((cells,), -1, device=device)
        self.counts = torch.zeros(cells, 3, dtype=torch.float64, device=device)  # stored states, misses, large misses

    def add(self, x, cells, truth, call, gases):
        """Score the states x[cells] after this call against truth rows; bad truth counts as a failure."""
        log_x = gases.floored_log(x[cells][:, gases.index])
        worst, gas = (log_x - truth).abs().max(1)
        worst = torch.where(torch.isfinite(log_x).all(1) & torch.isfinite(truth).all(1), worst, torch.inf)
        before = self.worst[cells]
        up = worst > before
        self.worst[cells] = torch.where(up, worst, before)
        self.worst_call[cells] = torch.where(up, call, self.worst_call[cells])
        self.worst_gas[cells] = torch.where(up, gas, self.worst_gas[cells])
        hits = torch.stack((torch.ones_like(worst), (worst > MISS).double(), (worst > LARGE).double()), 1)
        self.counts.index_add_(0, cells, hits)

    def numpy(self):
        """Return the worst error, its call and gas, as host arrays."""
        return tuple(a.cpu().numpy() for a in (self.worst, self.worst_call, self.worst_gas))


class Block:
    """One panel on one schedule.

    Call j runs on tick j * stride and uses forcing row (j // calls_per_row) % cycle_rows of each path (calls_per_row
    calls per 1500-s hold; cycle_rows = the row count of a cycled panel); a path takes part in calls first_call <= j <
    end_call; a mix keyed to row r comes before call calls_per_row * r, a stored state of row r is the state after call
    calls_per_row * r + calls_per_row - 1.
    """

    def __init__(self, name, cells, stride, calls, calls_per_row=1, dt=None, cycle_rows=None):
        self.name, self.cells, self.stride, self.calls, self.dt = name, cells, stride, calls, dt
        self.calls_per_row = calls_per_row
        self.cycle_rows = cycle_rows or NEVER
        self.n_paths = len(cells)
        self.truth = cells[0]["truth"] is not None
        lengths = np.array([len(c["T"]) for c in cells])
        assert cycle_rows is None or (cycle_rows == lengths).all()
        self.first_call = np.array([calls_per_row * c["first_row"] for c in cells])
        self.end_call = np.full(self.n_paths, calls) if cycle_rows else np.minimum(calls_per_row * lengths, calls)
        self.rows_used = lengths if cycle_rows else np.minimum(lengths, -(-calls // calls_per_row))
        self.end = int(self.end_call.max())
        self.events = np.unique(np.concatenate([self.first_call, self.end_call]))
        self.next_event, self.n_live = -1, 0

    def setup(self, first_cell, forcing_start, device):
        """Set the global cell numbers and forcing offsets; put the mixes by row and stored truth by call on the GPU."""
        self.first_cell, self.forcing_start = first_cell, forcing_start
        mix_row, mix_cell, mix_fraction, mix_gas = [], [], [], []
        stored_call, stored_cell, stored_truth = [], [], []
        self.mixed = np.zeros(self.n_paths, bool)
        for i, path in enumerate(self.cells):
            for row, fraction, gas in path["mix"]:
                if self.cycle_rows < NEVER or self.calls_per_row * row < self.end_call[i]:
                    mix_row.append(row), mix_cell.append(first_cell + i), mix_fraction.append(fraction)
                    mix_gas.append(gas)
                    self.mixed[i] = True
            if self.truth:
                calls = self.calls_per_row * path["rows"] + self.calls_per_row - 1
                keep = calls < self.end_call[i]
                stored_call.append(calls[keep])
                stored_cell.append(np.full(int(keep.sum()), first_cell + i))
                stored_truth.append(path["truth"][keep])
        mix_row = np.asarray(mix_row, np.int64)
        order = np.argsort(mix_row, kind="stable")
        self.mix_start = np.searchsorted(mix_row[order], np.arange(int(self.rows_used.max()) + 1))
        self.mix_cell = torch.as_tensor(np.asarray(mix_cell, np.int64)[order], device=device)
        self.mix_fraction = torch.as_tensor(np.asarray(mix_fraction, np.float64)[order], device=device)[:, None]
        species = len(self.cells[0]["x0"])
        self.mix_gas = torch.as_tensor(np.asarray(mix_gas, np.float64).reshape(-1, species)[order], device=device)
        if self.truth:
            stored_call = np.concatenate(stored_call)
            order = np.argsort(stored_call, kind="stable")
            self.stored_start = np.searchsorted(stored_call[order], np.arange(self.calls + 1))
            self.stored_cell = torch.as_tensor(np.concatenate(stored_cell)[order], device=device)
            self.stored_truth = torch.as_tensor(np.concatenate(stored_truth)[order], device=device)

    def live(self, j, device):
        """Refresh the scheduled cells (global numbers, forcing offsets) when a path starts or ends at call j."""
        if j < self.next_event:
            return
        on = np.flatnonzero((self.first_call <= j) & (j < self.end_call))
        self.live_cells = torch.as_tensor(self.first_cell + on, device=device)
        self.live_forcing = torch.as_tensor(self.forcing_start[on], device=device)
        self.n_live = len(on)
        later = self.events[self.events > j]
        self.next_event = int(later[0]) if len(later) else NEVER

    def mix(self, row, x):
        """Apply the mixes keyed to this forcing row, before the row's first call."""
        start, stop = self.mix_start[row], self.mix_start[row + 1]
        if stop > start:
            cells, fraction = self.mix_cell[start:stop], self.mix_fraction[start:stop]
            x[cells] = (1.0 - fraction) * x[cells] + fraction * self.mix_gas[start:stop]


def counts_since(solver, start):
    """Return the guard's counts since start, an earlier copy of solver.total (one small copy from the device)."""
    return dict(zip(kincuda.COUNTS, (solver.total - start)[1:-1].tolist(), strict=True))


def _complete_box_set(species):
    """Read the boxes' manifest and refuse a set that is not complete: (manifest, shards, first shard)."""
    manifest = json.loads((paths.boxes() / "manifest.json").read_text())
    shards = [paths.boxes() / f for f in manifest.get("shards", [])]
    bad = {k: len(manifest.get(k, [])) for k in ("errors", "bad_states", "drift_over_1e-6")}
    ids = np.concatenate([np.load(f)["box"] for f in shards]) if shards else np.zeros(0, int)
    cover = np.zeros(manifest["boxes"], int)
    np.add.at(cover, np.concatenate([ids, np.asarray(manifest.get("quarantine", []), int)]), 1)
    if manifest.get("pending", 1) or any(bad.values()) or len(ids) != manifest["accepted"] or (cover != 1).any():
        raise SystemExit(
            f"{paths.boxes()}: not complete (manifest: {manifest.get('pending')} boxes not run, {bad}, "
            f"{manifest['accepted']} accepted; shards hold {len(ids)} boxes, design boxes not held exactly once "
            f"(accepted or quarantined) {int((cover != 1).sum())})"
        )
    print(
        f"boxes: {manifest['accepted']:,} of {manifest['boxes']:,} accepted, "
        f"{len(manifest['quarantine'])} quarantined (left out)"
    )
    first = np.load(shards[0])
    for f in shards:
        z = np.load(f)
        if (
            [str(q) for q in z["species"]] != species
            or not np.array_equal(z["stops"], first["stops"])
            or not np.array_equal(z["lines"], first["lines"])
        ):
            raise SystemExit(f"{f}: species, stops or photolysis lines differ from the paths' and the other shards'")
    return manifest, shards, first


def _load_boxes(shards, cap):
    """Return the first cap boxes (None: all) of the shards, by field."""
    box_set = {k: [] for k in BOX_KEYS}
    loaded = 0
    for f in shards:
        if cap is not None and loaded >= cap:
            break
        z = np.load(f)
        for k in BOX_KEYS:
            box_set[k].append(z[k])
        loaded += len(z["box"])
    return {k: np.concatenate(v)[:cap] for k, v in box_set.items()}


def boxes(solver, net, species, gases, cap):
    """Run the VULCAN truth boxes.

    The boxes (paths.boxes(), made by the truth generator's sobol.py: constant T, P and photolysis rates,
    stored at fixed stops from 0 to 1e8 s): the first cap boxes (None: all), each from its stored start in chained
    calls whose lengths are the stop intervals, every stored state scored. Refuses a set whose manifest lists boxes not
    run, failed runs, bad states or element drift, or whose shards do not hold the accepted count; reports the
    quarantined boxes (a step forced at the generator's dt_min, left out). b_set: a digest of the scored truth and
    inputs. Returns the per-box results (b_*).
    """
    device = solver.dev
    manifest, shards, first = _complete_box_set(species)
    stops, lines = first["stops"], [str(q) for q in first["lines"]]
    branch_column = [lines.index(f"{q['absorber']},{q['branch']}") for q in net.photo]
    box_set = _load_boxes(shards, cap)
    n_boxes, calls = len(box_set["box"]), len(stops) - 1
    results = {"b_box": box_set["box"], "b_calls": np.full(n_boxes, calls)}
    results |= {
        f"b_{k}": np.zeros(n_boxes) for k in ("worst", "saved", "nmiss", "nlarge", "unfinished", "nonfinite", "steps")
    }
    results |= {"b_kworst": np.full(n_boxes, -1), "b_gworst": np.full(n_boxes, -1)}
    results["b_lx"] = np.zeros((n_boxes, len(GASES)), np.float32)
    start = solver.total.clone()
    started = time.time()

    def tensor(a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64, device=device)

    for first_box in range(0, n_boxes, CHUNK):
        chunk = slice(first_box, min(n_boxes, first_box + CHUNK))
        x, T, P = tensor(box_set["X"][chunk, 0]), tensor(box_set["T"][chunk]), tensor(box_set["P"][chunk])
        J = tensor(box_set["J"][chunk][:, branch_column])
        h = torch.full_like(T, START_STEP)
        outcome = torch.zeros(len(T), 3, dtype=torch.float64, device=device)  # unfinished, non-finite, steps
        tally = ErrorTally(len(T), device)
        cells = torch.arange(len(T), device=device)
        for k in range(1, calls + 1):
            x, h, done, steps = solver(x, T, P, torch.full_like(T, stops[k] - stops[k - 1]), h, J=J)
            bad = torch.logical_not(torch.isfinite(x).all(1))
            outcome += torch.stack((torch.logical_not(done).double(), bad.double(), steps), 1)
            stored = tensor(box_set["X"][chunk, k][:, gases.columns])
            truth = torch.where(torch.isfinite(stored), gases.floored_log(stored), torch.nan)
            tally.add(x, cells, truth, k, gases)
        results["b_worst"][chunk], results["b_kworst"][chunk], results["b_gworst"][chunk] = tally.numpy()
        results["b_saved"][chunk], results["b_nmiss"][chunk], results["b_nlarge"][chunk] = tally.counts.cpu().numpy().T
        results["b_unfinished"][chunk], results["b_nonfinite"][chunk], results["b_steps"][chunk] = (
            outcome.cpu().numpy().T
        )
        results["b_lx"][chunk] = gases.snapshot(x)
    torch.cuda.synchronize()
    results["b_s"], results["b_cellcalls"] = time.time() - started, n_boxes * calls
    digest = hashlib.sha256(json.dumps(manifest["stamp"], sort_keys=True).encode())
    for a in [*(box_set[k] for k in BOX_KEYS), stops, np.array(lines), np.array(species)]:
        digest.update(np.ascontiguousarray(a).tobytes())
    results["b_set"] = np.array(
        f"{manifest['network']} rtol {manifest.get('rtol')} {n_boxes} boxes sha256 {digest.hexdigest()[:16]}"
    )
    guard = counts_since(solver, start)
    results["fallback_boxes"] = np.array([guard[q] for q in GUARD_KEYS])
    box_report(results, box_set, stops)
    return results


def box_report(results, box_set, stops):
    """Misses-first lines of the box pass."""
    worst = results["b_worst"]
    print(
        f"boxes: {len(worst):,} boxes x {len(stops) - 1} calls in {results['b_s']:.0f} s "
        f"({1e6 * results['b_s'] / results['b_cellcalls']:.3f} us per cell-call), steps per call "
        f"{results['b_steps'].sum() / results['b_calls'].sum():.3f}; guard "
        f"{dict(zip(GUARD_KEYS, results['fallback_boxes'].tolist(), strict=True))}"
    )

    def box_line(name, s):
        if s.any():
            print(
                f"  {name:20s} {int(s.sum()):7,d} boxes: miss {100 * (worst[s] > MISS).mean():6.3f} %  large "
                f"{100 * (worst[s] > LARGE).mean():6.3f} %  worst {worst[s].max():.3f} | stored states "
                f"{int(results['b_saved'][s].sum()):,}: miss {int(results['b_nmiss'][s].sum()):,} large "
                f"{int(results['b_nlarge'][s].sum()):,} | boxes with an unfinished call "
                f"{int((results['b_unfinished'][s] > 0).sum())}"
                f", with a non-finite state {int((results['b_nonfinite'][s] > 0).sum())}"
            )

    box_line("all boxes", np.ones(len(worst), bool))
    box_line("dark", box_set["lit"] == 0)
    box_line("lit", box_set["lit"] == 1)
    for k, name in enumerate(("unreacted", "reheat", "mixed", "scrambled", "hot-H")):
        box_line(f"start {name}", box_set["kind"] == k)
    for lo, hi in ((300, 400), (400, 700), (700, 1200), (1200, 2000), (2000, 3001)):
        box_line(f"T {lo}-{hi} K", (box_set["T"] >= lo) & (box_set["T"] < hi))
    print("  worst boxes:")
    for i in np.argsort(-worst)[:5]:
        print(
            f"    box {box_set['box'][i]} T {box_set['T'][i]:.0f} K P {box_set['P'][i] / 1e6:.1e} bar "
            f"{'lit' if box_set['lit'][i] else 'dark'} "
            f"start {int(box_set['kind'][i])}: {worst[i]:.3f} dex in {GASES[results['b_gworst'][i]]} at stop "
            f"{stops[results['b_kworst'][i]]:.3g} s"
        )


def make_blocks(tier, species, gas_columns):
    """Return the tier's blocks (see the module docstring)."""
    started = time.time()
    qc = load_paths(("QC", "QCs", "twins/QC"), species, gas_columns, hold=False)
    qh = load_paths(("QH", "QHs", "twins/QH"), species, gas_columns, hold=True)
    ql = load_paths(("QL",), species, gas_columns, hold=True)
    qlc = load_paths(("QLc",), species, gas_columns, hold=False)
    panel = load_p(species)
    n_qc, n_qh, n_ql, n_qlc, n_panel = (max(len(c["T"]) for c in v) for v in (qc, qh, ql, qlc, panel))
    print(f"truth and forcing loaded in {time.time() - started:.0f} s", flush=True)
    if tier == "quick":
        return [
            Block("QC family, per call", qc, 1, n_qc),
            Block("QH family, 1500-s holds", qh, 1, n_qh, dt=HOLD),
            Block("QL, dt 500 s", ql, 1, 10000, calls_per_row=3, dt=500.0),
            Block("QLc, dt 500 s", qlc, 1, 10000),
            *[Block(f"P, dt {d:g} s", panel, 1, n_panel, dt=d, cycle_rows=n_panel) for d in (25.0, 50.0, 500.0)],
        ]
    return [
        Block("QL, dt 25 s", ql, 1, 60 * n_ql, calls_per_row=60, dt=25.0),
        Block("QL, dt 500 s", ql, 20, 3 * n_ql, calls_per_row=3, dt=500.0),
        Block("QLc, dt 500 s", qlc, 20, n_qlc),
        Block("QH family, dt 50 s", qh, 2, 30 * n_qh, calls_per_row=30, dt=50.0),
        Block("QC family, per call", qc, 1, n_qc),
        Block("QH family, 1500-s holds", qh, 1, n_qh, dt=HOLD),
        *[
            Block(f"P, dt {d:g} s", panel, stride, int(days * DAY / d), dt=d, cycle_rows=n_panel)
            for d, stride, days in ((25.0, 1, 200), (50.0, 1, 400), (500.0, 2, 1000))
        ],
    ]


@dataclass
class PathRun:
    """All blocks' cells together: their state, flat forcing per row and per-cell counters."""

    x: torch.Tensor
    h: torch.Tensor
    T: torch.Tensor
    P: torch.Tensor  # dyn cm^-2
    dt: torch.Tensor
    host_forcing: tuple  # T, P and dt on the host
    atoms: torch.Tensor
    h_column: int  # the element H's column in atoms
    start_ratio: torch.Tensor  # each element per H at the start
    meta: dict
    outcome: torch.Tensor  # calls, accepted steps, unfinished, non-finite, negative
    tally: ErrorTally
    drift: torch.Tensor
    ticks: int

    @property
    def n_cells(self):
        """Return the number of cells."""
        return len(self.x)


def assemble(blocks, net, device):
    """Return the run's state, with each block set up on it."""
    x_start, t_forcing, p_forcing, dt_forcing, forcing_rows = [], [], [], [], 0
    meta = {q: [] for q in META}
    first_cell = 0
    for b in blocks:
        forcing_start = np.zeros(b.n_paths, np.int64)
        for i, path in enumerate(b.cells):
            rows = int(b.rows_used[i])
            forcing_start[i] = forcing_rows
            forcing_rows += rows
            t_forcing.append(path["T"][:rows]), p_forcing.append(path["P"][:rows] * 1e6)
            dt_forcing.append(path["dt"][:rows] if b.dt is None else np.full(rows, b.dt))
            x_start.append(path["x0"] / path["x0"].sum())
            for q in META:
                meta[q].append(path["meta"][q])
        b.setup(first_cell, forcing_start, device)
        first_cell += b.n_paths

    def flat(a):
        return torch.as_tensor(np.concatenate(a), dtype=torch.float64, device=device)

    x = torch.as_tensor(np.stack(x_start), device=device)
    atoms = torch.as_tensor(net.atoms, device=device)
    h_column = net.elements.index("H")
    elements = x @ atoms
    cells = len(x_start)
    return PathRun(
        x=x,
        h=torch.full((cells,), START_STEP, dtype=torch.float64, device=device),
        T=flat(t_forcing),
        P=flat(p_forcing),
        dt=flat(dt_forcing),
        host_forcing=(np.concatenate(t_forcing), np.concatenate(p_forcing), np.concatenate(dt_forcing)),
        atoms=atoms,
        h_column=h_column,
        start_ratio=elements / elements[:, h_column : h_column + 1],
        meta={q: np.array(v) for q, v in meta.items()},
        outcome=torch.zeros(cells, 5, dtype=torch.float64, device=device),
        tally=ErrorTally(cells, device),
        drift=torch.zeros(cells, dtype=torch.float64, device=device),
        ticks=max(b.stride * (b.end - 1) + 1 for b in blocks),
    )


def _stop_if_too_long(ticks_left, so_far, read_s):
    """Print the full tier's estimate, from the cost of the last READ ticks; stop above CAP_H hours."""
    estimate = so_far + ticks_left * read_s / READ
    print(
        f"estimate: {estimate / 3600:.2f} h in all ({so_far / 60:.1f} min so far, then {ticks_left:,} ticks x "
        f"{1e3 * read_s / READ:.2f} ms, the last {READ:,} ticks' rate)",
        flush=True,
    )
    if estimate > CAP_H * 3600:
        print(f"stopped: the estimate exceeds {CAP_H:g} h", flush=True)
        sys.exit(2)


def _schedule(blocks, tick, x, device):
    """Return the cells, forcing rows and blocks of this tick, after the mixes of its forcing rows."""
    cells, forcing, scheduled = [], [], []
    for b in blocks:
        if tick % b.stride:
            continue
        j = tick // b.stride
        if j >= b.end:
            continue
        b.live(j, device)
        if not b.n_live:
            continue
        row = (j // b.calls_per_row) % b.cycle_rows
        if j % b.calls_per_row == 0:
            b.mix(row, x)
        cells.append(b.live_cells), forcing.append(b.live_forcing + row), scheduled.append((b, j))
    return cells, forcing, scheduled


def _advance(state, solver, cells, forcing):
    """Make one dark call of these cells (the paths run without light), and add its counts."""
    dark = torch.zeros(forcing.shape[0], solver.nj, dtype=torch.float64, device=solver.dev)
    x_out, h_out, done, steps = solver(
        state.x[cells], state.T[forcing], state.P[forcing], state.dt[forcing], state.h[cells], J=dark
    )
    state.x[cells], state.h[cells] = x_out, h_out
    finite = torch.isfinite(x_out).all(1)
    negative = x_out.min(1).values < 0
    outcome = (torch.ones_like(steps), steps, (~done).double(), (~finite).double(), negative.double())
    state.outcome.index_add_(0, cells, torch.stack(outcome, 1))
    elements = x_out @ state.atoms
    per_h = elements / elements[:, state.h_column : state.h_column + 1]
    change = torch.log10(per_h / state.start_ratio[cells]).abs().nan_to_num(nan=0.0).max(1).values
    state.drift[cells] = torch.maximum(state.drift[cells], change)


def run(blocks, state, solver, gases, full, started):
    """Run the tick loop of a tier; return timings, snapshots and the guard's counts.

    full: the full tier, which reads back less often and can stop at its estimate; started: the run's start time.
    """
    device, ticks = solver.dev, state.ticks
    snapshot_every = 10 * READ if full else READ
    active, snapshots, snapshot_ticks, cell_calls = list(blocks), [], [], 0
    start = solver.total.clone()
    loop_start = last_read = time.time()
    for tick in range(ticks):
        cells, forcing, scheduled = _schedule(active, tick, state.x, device)
        if cells:
            cells = cells[0] if len(cells) == 1 else torch.cat(cells)
            forcing = forcing[0] if len(forcing) == 1 else torch.cat(forcing)
            _advance(state, solver, cells, forcing)
            cell_calls += cells.numel()
            for b, j in scheduled:
                if b.truth:
                    first, stop = b.stored_start[j], b.stored_start[j + 1]
                    if stop > first:
                        state.tally.add(state.x, b.stored_cell[first:stop], b.stored_truth[first:stop], j, gases)
        if (tick + 1) % READ == 0 or tick + 1 == ticks:
            worst = state.tally.worst
            now = time.time()
            if not full or (tick + 1) % (100 * READ) == 0 or tick + 1 in (READ, ticks):
                totals = state.outcome.sum(0).tolist()
                since_read = (tick % READ) + 1
                print(
                    f"tick {tick + 1:,} of {ticks:,}: {now - loop_start:.0f} s, "
                    f"{1e3 * (now - last_read) / since_read:.2f} ms per tick (last {since_read:,}), "
                    f"{sum(b.n_live for b, _ in scheduled)} cells this tick, paths with a "
                    f"miss {int((worst > MISS).sum())}, worst {float(worst.max()):.3f} dex, "
                    f"unfinished {totals[2]:.0f}, non-finite {totals[3]:.0f}, negative {totals[4]:.0f}, "
                    f"recomputed by the guard {counts_since(solver, start)['recomputed']}",
                    flush=True,
                )
            if full and tick + 1 == ESTIMATE_TICK:
                _stop_if_too_long(ticks - tick - 1, now - started, now - last_read)
            if full and now - started > CAP_H * 3600:  # a stuck cell or another job on the GPU, after the estimate
                print(f"stopped at tick {tick + 1:,}: the run has taken {CAP_H:g} h", flush=True)
                sys.exit(2)
            last_read = now
            if (tick + 1) % snapshot_every == 0:
                snapshots.append(gases.snapshot(state.x)), snapshot_ticks.append(tick + 1)
            active = [b for b in active if (b.end - 1) * b.stride > tick]
    torch.cuda.synchronize()
    guard = counts_since(solver, start)
    return {
        "loop_s": time.time() - loop_start,
        "ticks": ticks,
        "cellcalls": cell_calls,
        "snap_tick": np.array(snapshot_ticks),
        "snap_lx": np.stack(snapshots) if snapshots else np.zeros((0, state.n_cells, len(GASES)), np.float32),
        "final_lx": gases.snapshot(state.x),
        "fallback_loop": np.array([guard[q] for q in GUARD_KEYS]),
    }


def path_results(blocks, state):
    """Per-cell counters and metadata as numpy arrays (the saved per-path summaries)."""
    results = {
        "block_names": np.array([b.name for b in blocks]),
        "truth_block": np.array([b.truth for b in blocks]),
        "block_of": np.concatenate([np.full(b.n_paths, i) for i, b in enumerate(blocks)]),
        "mixed": np.concatenate([b.mixed for b in blocks]),
        **state.meta,
    }
    results["worst"], results["kworst"], results["gworst"] = state.tally.numpy()
    results["stored"], results["nmiss"], results["nlarge"] = state.tally.counts.cpu().numpy().T
    results["calls"], results["steps"], results["unfinished"], results["nonfinite"], results["negative"] = (
        state.outcome.cpu().numpy().T
    )
    results["drift"] = state.drift.cpu().numpy()
    return results


def path_line(name, s, results, truth):
    """One misses-first line for the paths s."""
    worst, calls = results["worst"][s], results["calls"][s].sum()
    unmixed = s & ~results["mixed"]
    drift = f"{results['drift'][unmixed].max():.1e} dex" if unmixed.any() else "-"
    head = f"  {name:10s} {int(s.sum()):5d} paths"
    if truth:
        head += (
            f": miss {100 * (worst > MISS).mean():5.1f} %  large {100 * (worst > LARGE).mean():5.1f} %  "
            f"worst {worst.max():.3f} | stored states {int(results['stored'][s].sum()):,}: "
            f"miss {int(results['nmiss'][s].sum())} large {int(results['nlarge'][s].sum())}"
        )
    print(
        head + f" | calls {int(calls):,}: unfinished {int(results['unfinished'][s].sum())} non-finite "
        f"{int(results['nonfinite'][s].sum())} negative {int(results['negative'][s].sum())} | "
        f"element drift after the correction (unmixed) {drift} | "
        f"steps per call {results['steps'][s].sum() / max(calls, 1):.3f}"
    )


def report(blocks, results, loop_results, state):
    """Print the report, misses first.

    Totals over the truth paths, then per block (split by panel and kind; breakdowns by stratum,
    start and composition; the worst paths), then timings and the guard's counts.
    """
    has_truth = results["truth_block"][results["block_of"]]
    worst = np.where(has_truth, results["worst"], -1.0)
    c = int(np.argmax(worst))
    print(
        f"\nMISSES FIRST. qual truth paths {int(has_truth.sum()):,}: with a miss {int((worst > MISS).sum())}, "
        f"with a large miss {int((worst > LARGE).sum())}; stored states {int(results['stored'].sum()):,}: with a miss "
        f"{int(results['nmiss'].sum())}, large "
        f"{int(results['nlarge'].sum())}; worst {worst[c]:.3f} dex "
        f"({results['block_names'][results['block_of'][c]]}, path {results['id'][c]}); "
        f"all blocks: unfinished calls {int(results['unfinished'].sum())}, "
        f"non-finite states {int(results['nonfinite'].sum())}, "
        f"negative {int(results['negative'].sum())}"
    )
    t_forcing, p_forcing, dt_forcing = state.host_forcing
    for block_number, b in enumerate(blocks):
        s = results["block_of"] == block_number
        every = f"every {b.stride} tick{'s' if b.stride > 1 else ''}"
        print(f"{b.name}: {b.n_paths} paths, up to {b.calls:,} calls ({every})")
        labels = np.char.add(np.char.add(results["panel"], " "), results["kind"])
        for label in dict.fromkeys(labels[s]):
            path_line(label, s & (labels == label), results, b.truth)
        if not b.truth:
            continue
        for key in ("stratum", "start", "comp"):
            parts = []
            for value in np.unique(results[key][s]):
                q = s & (results[key] == value)
                parts.append(
                    f"{value} {int(q.sum())} (miss {int((results['worst'][q] > MISS).sum())}, "
                    f"worst {results['worst'][q].max():.3f})"
                )
            print(f"    by {key}: " + "; ".join(parts))
        print("    worst paths:")
        for c in b.first_cell + np.argsort(-results["worst"][s])[:5]:
            i, j = c - b.first_cell, int(results["kworst"][c])
            if j < 0:
                continue
            row = b.forcing_start[i] + (j // b.calls_per_row) % b.cycle_rows
            segments = b.cells[i]["seg"]
            where = ""
            if segments:
                where = f" seg {segments[1][int(np.searchsorted(segments[0], j // b.calls_per_row, side='right')) - 1]}"
            print(
                f"      {results['id'][c]} {results['panel'][c]} {results['kind'][c]} {results['stratum'][c]} "
                f"{results['start'][c]} {results['comp'][c]}: "
                f"{results['worst'][c]:.3f} dex in {GASES[results['gworst'][c]]} at call {j}{where} "
                f"(T {t_forcing[row]:.0f} K, P {p_forcing[row] / 1e6:.2e} bar, dt {dt_forcing[row]:.1f} s); "
                f"stored states with a miss {int(results['nmiss'][c])}"
            )
    print(
        f"loop: {loop_results['ticks']:,} ticks in {loop_results['loop_s']:.0f} s = "
        f"{1e3 * loop_results['loop_s'] / loop_results['ticks']:.3f} ms per tick; "
        f"{loop_results['cellcalls']:,} cell-calls = "
        f"{1e6 * loop_results['loop_s'] / loop_results['cellcalls']:.3f} us per cell-call; guard "
        f"{dict(zip(GUARD_KEYS, loop_results['fallback_loop'].tolist(), strict=True))}"
    )


def _paired_line(name, s, ids, worst, state_change, steps, unfinished):
    """One line of a paired run: worst is (new, ref) per path, None without truth; steps, unfinished (new, ref)."""
    text = f"  {name:24s} {int(s.sum()):7,d} paths"
    if worst is not None:
        rise = np.nan_to_num(worst[0][s] - worst[1][s], nan=np.inf, posinf=np.inf, neginf=-np.inf)
        most, least = int(np.argmax(rise)), int(np.argmin(rise))
        text += (
            f" | worst error new - ref: mean {rise[np.isfinite(rise)].mean() if np.isfinite(rise).any() else 0:+.5f}, "
            f"largest rise {rise[most]:+.4f} ({ids[s][most]}), largest fall {rise[least]:+.4f} ({ids[s][least]}); "
            f"|change| > 0.01: {int((np.abs(rise) > 0.01).sum())}, > 0.03: {int((np.abs(rise) > 0.03).sum())}"
        )
    print(
        text + f" | state change > 0.01: {int((state_change[s] > 0.01).sum())}, "
        f"> 0.03: {int((state_change[s] > 0.03).sum())} (largest {state_change[s].max():.2e}) | "
        f"steps per call {steps[0]:.3f} vs {steps[1]:.3f} | unfinished {int(unfinished[0])} vs {int(unfinished[1])}"
    )


def compare(new, ref):
    """Paired comparison of two runs of the same tier (mappings as written by --save)."""
    for k in ("tier", "id", "block_of", "b_box", "b_set", "q_set"):
        assert np.array_equal(np.asarray(new[k]), np.asarray(ref[k])), f"runs differ in {k}: not comparable"
    assert np.array_equal(new["snap_tick"], ref["snap_tick"])

    def latency(z):
        return (1e3 * float(z["loop_s"]) / int(z["ticks"]), 1e6 * float(z["b_s"]) / int(z["b_cellcalls"]))

    (tick_new, box_new), (tick_ref, box_ref) = latency(new), latency(ref)
    print(
        f"\nPAIRED with the reference (tier {new['tier']}): latency loop {tick_new:.3f} vs {tick_ref:.3f} ms per tick "
        f"({tick_new / tick_ref:.2f}x), boxes {box_new:.3f} vs {box_ref:.3f} us per cell-call "
        f"({box_new / box_ref:.2f}x)"
    )

    def steps_per_call(z, s):
        return z["steps"][s].sum() / max(z["calls"][s].sum(), 1)

    def difference(a, b):
        return np.nan_to_num(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)), nan=np.inf)

    change = np.maximum(
        difference(new["final_lx"], ref["final_lx"]).max(1),
        difference(new["snap_lx"], ref["snap_lx"]).max((0, 2)) if len(new["snap_tick"]) else 0.0,
    )
    for block_number, name in enumerate(new["block_names"]):
        s = new["block_of"] == block_number
        _paired_line(
            str(name),
            s,
            new["id"],
            (new["worst"], ref["worst"]) if new["truth_block"][block_number] else None,
            change,
            (steps_per_call(new, s), steps_per_call(ref, s)),
            (new["unfinished"][s].sum(), ref["unfinished"][s].sum()),
        )
    _paired_line(
        "boxes",
        np.ones(len(new["b_box"]), bool),
        new["b_box"],
        (new["b_worst"], ref["b_worst"]),
        difference(new["b_lx"], ref["b_lx"]).max(1),
        (new["b_steps"].sum() / new["b_calls"].sum(), ref["b_steps"].sum() / ref["b_calls"].sum()),
        (new["b_unfinished"].sum(), ref["b_unfinished"].sum()),
    )


def _paths_digest(blocks):
    """Return a digest of the paths' truth and forcing, for paired runs."""
    digest = hashlib.sha256()
    for b in blocks:
        for path in b.cells:
            for v in (path["T"], path["P"], path["dt"], path["x0"], path["rows"], path["truth"]):
                if v is not None:
                    digest.update(np.ascontiguousarray(v).tobytes())
            for row, fraction, gas in path["mix"]:
                digest.update(np.array([row, fraction]).tobytes() + np.ascontiguousarray(gas).tobytes())
    return digest.hexdigest()[:16]


def main():
    """Run the tier, print the report and exit 1 on any miss or failed cell."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tier", choices=("quick", "full"))
    ap.add_argument("--save", help="write this npz (refuses to overwrite)")
    ap.add_argument("--ref", help="compare with this saved run of the same tier")
    a = ap.parse_args()
    if a.save and Path(a.save).exists():
        sys.exit(f"{a.save} exists; not overwriting")
    paths.need(paths.qual() / "QC", paths.designs() / "decision/P", paths.boxes(), *([a.ref] if a.ref else []))
    check_jobs()
    ref = np.load(a.ref) if a.ref else None
    started = time.time()
    device = torch.device("cuda")
    quick = a.tier == "quick"
    species = [str(q) for q in np.load(min((paths.qual() / "QC").glob("*.npz")))["species"]]
    columns = [species.index(g) for g in GASES]
    thermo = paths.vulcan_thermo()
    net = kinvulcan.VulcanNetwork(f"{thermo}/NCHO_photo_network.txt", f"{thermo}/NASA9", species=species)
    solver = kincuda.CudaSolver(net, device, photo=True)
    gases = ScoredGases(columns, torch.tensor(columns, device=device), torch.tensor(FLOOR, device=device))
    print(
        f"{a.tier}: {torch.cuda.get_device_name(device)}, kernel ready ({solver.wpb} warps per block), "
        f"{time.time() - started:.0f} s",
        flush=True,
    )
    box_results = boxes(solver, net, species, gases, QUICK_BOXES if quick else None)
    blocks = make_blocks(a.tier, species, columns)
    paths_digest = _paths_digest(blocks)
    state = assemble(blocks, net, device)
    print(
        f"{len(blocks)} blocks, {state.n_cells:,} cells, {state.ticks:,} ticks; "
        f"set up in {time.time() - started:.0f} s",
        flush=True,
    )
    loop_results = run(blocks, state, solver, gases, not quick, started)
    solver.check()  # raises if any cell was below 300 K
    results = path_results(blocks, state)
    report(blocks, results, loop_results, state)
    record = run_record(thermo / "NCHO_photo_network.txt", thermo / "NASA9", species, solver.drift, a)
    results |= loop_results | box_results
    results |= {"tier": np.array(a.tier), "q_set": np.array(paths_digest), "run": np.array(json.dumps(record))}
    print(
        f"wall time {time.time() - started:.0f} s "
        f"(boxes {box_results['b_s']:.0f} s, loop {loop_results['loop_s']:.0f} s)",
        flush=True,
    )
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.save, **results)
        print(f"saved {a.save}")
    if ref is not None:
        compare(results, ref)
    has_truth = results["truth_block"][results["block_of"]]
    fail = int(
        (np.where(has_truth, results["worst"], 0.0) > MISS).sum()
        + (box_results["b_worst"] > MISS).sum()
        + results["unfinished"].sum()
        + results["nonfinite"].sum()
        + results["negative"].sum()
        + box_results["b_unfinished"].sum()
        + box_results["b_nonfinite"].sum()
        + box_results["fallback_boxes"][GUARD_KEYS.index("still_failed")]
        + results["fallback_loop"][GUARD_KEYS.index("still_failed")]
    )
    if fail:
        sys.exit(
            f"FAIL: {fail} paths or boxes with a miss, unfinished calls, non-finite or negative states, or cell-calls "
            "still failed after the recovery"
        )


if __name__ == "__main__":
    main()
