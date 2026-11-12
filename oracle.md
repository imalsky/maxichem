# oracle.md — what maxichem must achieve

This file states the goal of maxichem and what counts as success or failure. It holds targets and decisions, not
measurements: the evidence is in notes.md. Only the maintainer changes this file.

## Goal: VULCAN's 0-D chemistry, in CUDA

maxichem is a fast CUDA version of VULCAN's 0-D chemistry. Given the same network, temperature, pressure, photolysis
rates and start state, it gives VULCAN's answer. It does this for many cells at once, as the chemistry step of a GCM,
and it gives the derivatives of each call. The GCM solves transport, diffusion and radiation. maxichem does not solve
them, but it stays correct on the gas they hand it.

"VULCAN" is VULCAN-0D as the truth generator runs it, with the corrections listed in notes "Truth generator". The
reference will move to vulcan3 (todo.md).

### Success

- **Accuracy.** Each scored gas is within 0.1 dex of VULCAN at every scored time, over the whole domain. The scored
  gases and their floors are in `maxichem/_checks.py`.
- **Every miss is investigated.** A miss with a recorded cause outside maxichem (a VULCAN error, unsound network data)
  is listed in notes "VULCAN reference errors" and is not a failure.
- **Domain.** Temperature 300 to 3000 K and pressure 1e-5 to 100 bar. Metallicity 0.1 to 300 times solar and C/O 0.1 to
  2. Dark and lit, with J from 1e-3 to 10 times a 1-D model layer's. Call lengths 1 to 500 s. Input outside the domain
  is refused with an error.
- **Stability.** Over at least 1,000 consecutive calls on its own output, there are no misses and no failed cells,
  including on gas mixed between calls.
- **Speed.** Under 10 us per cell-call for every network, sulfur included. This holds at a GCM-sized batch (2,457,600
  cells), at the default tolerances, on gas as in a coupled run.
- **Networks.** Every VULCAN-JAX network with sound data loads and runs. Accuracy is scored on the two defaults: the
  newest NCHO network (`NCHO_photo_network.txt`) and the newest sulfur network (`SNCHO_photo_network_2025.txt`). A
  network with unsound data is refused with an error naming the file.
- **Conservation.** In one call, no element's share changes by more than 1e-6, relative.
- **With transport.** The coupled box test passes: its passive-tracer convergence check, element totals and guard
  counts.
- **Derivatives.** The tangent and adjoint of a whole call exist for both default networks, with respect to the state,
  T, P, J and the rate multipliers.
  - Their primal result is the call's, bit for bit.
  - Tangents match central finite differences: 90 % of cells within 1e-4 and 99 % within 1e-2, relative.
  - Tangent and adjoint match each other: 99 % of cells within 1e-6.
  - A failed cell gives NaN derivatives, never a finite wrong number, and it does not change other cells.
- **C++.** The C++ call equals the Python call bit for bit, counts included.
- **Honest failures.** A failed cell is flagged and counted.

### Failure

- A miss with no recorded cause.
- A failed or non-finite cell counted as a pass.
- A feature that works only for NCHO.
- Speed bought with accuracy.
- "Close to VULCAN" claimed from an averaged score.
- A finite derivative from a failed cell.

### Not goals

Solving transport or radiation; a neural emulator; reduced tracers; the TiSNCHO network.
