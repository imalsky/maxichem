#include "maxichem.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace maxichem {

namespace {

// Copies of maxichem/_constants.py, held equal by tests/test_constants_source.py.
constexpr int kExportFormat = 6;                    // EXPORT_FORMAT: any other export is refused
constexpr int kGuardBlockThreads = 128;             // GUARD_BLOCK_THREADS: any size works; this one packs cells densely
constexpr int kWarpLanes = 32;                      // WARP_LANES: threads per warp on every CUDA GPU
constexpr char const* kPhotolysisArgument = "Jin";  // PHOTOLYSIS_ARGUMENT
constexpr size_t kCountSlotsAroundNames = 2;        // the two cell queues around the export's "counts" names
// The kernel parameters a call supplies besides the export's tables and J; operator() fills exactly these.
constexpr std::array<char const*, 20> kCallArguments{"x0",    "Ta",   "Pa",        "dta",     "hin",    "hio",   "xout",
                                                     "doneo", "noko", "status",    "redo",    "list",   "count", "B",
                                                     "rtol",  "atol", "max_steps", "scratch", "sround", "jround"};
// Each kernel's parameters in order, as KERNEL_ARGUMENTS in maxichem/_constants.py, so that an export with another
// contract is refused before anything runs; call and recover take J last with photolysis. Kept here as well because
// the caller must know what it can supply; every test_export_cpp case exports from Python and loads it here, so a
// difference between the two lists fails them all.
struct KernelArguments {
  char const* kernel;
  std::vector<char const*> names;
  bool photolysis;
};
std::array<KernelArguments, 3> const kKernelArguments{{
    {"call",
     {"x0",    "Ta",      "Pa",   "dta",  "hio",       "xout", "doneo",  "noko",  "status",
      "count", "B",       "rtol", "atol", "max_steps", "perm", "blob",   "rc",    "rf",
      "nasa",  "scratch", "sst",  "jt",   "lu",        "optr", "sround", "jround"},
     true},
    {"guard", {"x0", "xout", "doneo", "noko", "status", "atoms", "redo", "list", "count", "B", "atol"}, false},
    {"recover",
     {"x0",   "Ta",   "Pa",      "dta",  "hin",  "xout",      "hio",   "doneo",  "noko",
      "redo", "list", "count",   "rtol", "atol", "max_steps", "perm",  "blob",   "rc",
      "rf",   "nasa", "scratch", "sst",  "jt",   "didx",      "atoms", "sround", "jround"},
     true},
}};

void check(CUresult r, char const* what) {
  if (r != CUDA_SUCCESS) {
    char const* s = nullptr;
    cuGetErrorString(r, &s);
    throw std::runtime_error(std::string("maxichem: CUDA driver, ") + what + ": " + (s ? s : "unknown error"));
  }
}

std::string read_file(std::string const& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("maxichem: cannot read " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

torch::Dtype dtype_of(std::string const& name) {
  if (name == "float64") return torch::kFloat64;
  if (name == "int64") return torch::kInt64;
  if (name == "int32") return torch::kInt32;
  if (name == "int16") return torch::kInt16;
  throw std::runtime_error("maxichem: table dtype " + name);
}

// meta.txt: "key value..." per line, with the tables, the kernels' arguments and the name lists apart.
void read_meta(std::string const& path, std::map<std::string, std::string>& meta,
               std::vector<std::vector<std::string>>& tables,
               std::map<std::string, std::vector<std::string>>& arguments,
               std::map<std::string, std::vector<std::string>>& lists) {
  std::istringstream lines(read_file(path));
  for (std::string line; std::getline(lines, line);) {
    std::istringstream ws(line);
    std::vector<std::string> tok;
    for (std::string t; ws >> t;) tok.push_back(t);
    if (tok.empty()) continue;
    if (tok[0] == "table") {
      tables.push_back(tok);
    } else if (tok[0] == "args") {
      arguments[tok[1]].assign(tok.begin() + 2, tok.end());
    } else if (tok[0] == "species" || tok[0] == "elements" || tok[0] == "counts") {
      lists[tok[0]].assign(tok.begin() + 1, tok.end());
    } else if (tok.size() >= 2) {
      meta[tok[0]] = tok[1];
    }
  }
}

// The device's multiprocessors, after checking that the export's cubin is for its architecture.
int multiprocessors_on(int index, std::string const& arch) {
  CUdevice dev;
  check(cuDeviceGet(&dev, index), "device");
  int major = 0, minor = 0, sms = 0;
  check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "attribute");
  check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "attribute");
  check(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), "attribute");
  if (std::to_string(major * 10 + minor) != arch)
    throw std::runtime_error("maxichem: the export is for sm_" + arch + ", this device is sm_" +
                             std::to_string(major * 10 + minor) + "; export on this machine");
  return sms;
}

// As CudaSolver: opt the kernel in to its dynamic shared memory, then the blocks one multiprocessor holds.
int blocks_per_multiprocessor(CUfunction fn, int threads, int smem) {
  int blocks = 0;
  check(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem), "shared memory");
  check(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, threads, smem), "occupancy");
  if (blocks < 1)
    throw std::runtime_error("maxichem: a kernel of " + std::to_string(threads) + " threads and " +
                             std::to_string(smem) + " B of shared memory does not fit on this device");
  return blocks;
}

// One table from its line of meta.txt: table name dtype dims...
torch::Tensor read_table(std::string const& folder, std::vector<std::string> const& line) {
  std::vector<int64_t> shape;
  for (size_t i = 3; i < line.size(); ++i) shape.push_back(std::stoll(line[i]));
  auto cpu = torch::empty(shape, torch::TensorOptions().dtype(dtype_of(line[2])));
  std::string bytes = read_file(folder + "/tables/" + line[1] + ".bin");
  if (bytes.size() != static_cast<size_t>(cpu.nbytes()))
    throw std::runtime_error("maxichem: table " + line[1] + " has " + std::to_string(bytes.size()) +
                             " bytes, its shape " + "and dtype in meta.txt need " + std::to_string(cpu.nbytes()));
  std::memcpy(cpu.data_ptr(), bytes.data(), bytes.size());
  return cpu;
}

// Refuse an export whose kernel argument lists differ from the ones this caller supplies (each name once, in order,
// J only where a kernel takes it, every table listed), naming the kernel and the argument.
void check_arguments(std::map<std::string, std::vector<std::string>> const& arguments,
                     std::vector<std::vector<std::string>> const& tables, bool photolysis) {
  auto refuse = [](std::string const& kernel, std::string const& what) {
    throw std::runtime_error("maxichem: kernel " + kernel + " " + what + "; export again with this version");
  };
  for (auto const& [kernel, names, takes_j] : kKernelArguments) {
    std::vector<std::string> want(names.begin(), names.end());
    if (takes_j && photolysis) want.emplace_back(kPhotolysisArgument);
    auto it = arguments.find(kernel);
    if (it == arguments.end()) refuse(kernel, "has no args line in meta.txt");
    auto const& got = it->second;
    for (auto const& name : got) {
      if (std::find(want.begin(), want.end(), name) == want.end())
        refuse(kernel, "takes argument " + name + ", which this caller does not supply");
      if (std::count(got.begin(), got.end(), name) > 1) refuse(kernel, "lists argument " + name + " twice");
    }
    for (auto const& name : want)
      if (std::find(got.begin(), got.end(), name) == got.end()) refuse(kernel, "lacks argument " + name);
    for (size_t i = 0; i < want.size(); ++i)
      if (got[i] != want[i])
        refuse(kernel, "has argument " + got[i] + " at position " + std::to_string(i) + ", where this caller passes " +
                           want[i]);
    for (auto const& name : want) {
      bool const supplied =
          name == kPhotolysisArgument ||
          std::find(kCallArguments.begin(), kCallArguments.end(), name) != kCallArguments.end() ||
          std::any_of(tables.begin(), tables.end(), [&](auto const& t) { return t.size() > 1 && t[1] == name; });
      if (!supplied) refuse(kernel, "takes table " + name + ", which meta.txt does not list");
    }
  }
}

}  // namespace

Solver::Solver(std::string const& folder, torch::Device device) : device_(device) {
  TORCH_CHECK(device.is_cuda(), "maxichem::Solver: a CUDA device");
  // every check of meta.txt comes before any CUDA work
  std::map<std::string, std::string> meta;
  std::vector<std::vector<std::string>> tables;
  std::map<std::string, std::vector<std::string>> lists;
  read_meta(folder + "/meta.txt", meta, tables, arguments_, lists);
  species_ = lists["species"];
  elements_ = lists["elements"];
  auto geti = [&](char const* k) {
    auto it = meta.find(k);
    if (it == meta.end()) throw std::runtime_error(std::string("maxichem: meta.txt lacks ") + k);
    return std::stoll(it->second);
  };
  if (geti("export_format") != kExportFormat)
    throw std::runtime_error("maxichem: export format " + meta.at("export_format") + ", this code reads " +
                             std::to_string(kExportFormat) + "; export again with this version");
  ns_ = static_cast<int>(geti("ns"));
  nj_ = static_cast<int>(geti("nj"));
  w_ = static_cast<int>(geti("w"));
  cpb_ = static_cast<int>(geti("cpb"));
  smem_ = static_cast<int>(geti("smem"));
  smem_rec_ = static_cast<int>(geti("smem_rec"));
  stride_ = geti("stride");
  stride_rec_ = geti("stride_rec");
  sround_ = static_cast<int>(geti("sround"));
  jround_ = static_cast<int>(geti("jround"));
  rec_steps_ = static_cast<int>(geti("rec_steps"));
  default_atol_ = std::stod(meta.at("atol"));
  t_min_ = std::stod(meta.at("t_min"));
  t_max_ = std::stod(meta.at("t_max"));
  x_sum_tol_ = std::stod(meta.at("x_sum_tol"));
  auto slot = [&](char const* name) {
    auto const& names = lists["counts"];
    auto it = std::find(names.begin(), names.end(), name);
    if (it == names.end()) throw std::runtime_error(std::string("maxichem: meta.txt counts lack ") + name);
    return static_cast<int64_t>(it - names.begin()) + 1;  // after the queue's slot
  };
  bad_input_slot_ = slot("bad_input");
  cold_slot_ = slot("cold");
  hot_slot_ = slot("hot");
  if (static_cast<int>(species_.size()) != ns_)
    throw std::runtime_error("maxichem: meta.txt lists " + std::to_string(species_.size()) + " species, ns is " +
                             std::to_string(ns_));
  check_arguments(arguments_, tables, nj_ > 0);

  c10::cuda::CUDAGuard guard(device);
  device_ = torch::Device(torch::kCUDA, guard.current_device().index());  // "cuda": the current one
  device = device_;
  torch::zeros({1}, torch::TensorOptions().device(device));  // torch's context is current
  int sms = multiprocessors_on(device_.index(), meta.at("arch"));

  std::string cubin = read_file(folder + "/kernel.cubin");
  CUmodule mod;
  check(cuModuleLoadData(&mod, cubin.data()), "load");
  module_ = mod;
  try {  // unload the module if anything below fails
    CUfunction f[3];
    char const* names[3] = {"call", "guard", "recover"};
    for (int i = 0; i < 3; ++i) check(cuModuleGetFunction(&f[i], mod, names[i]), "function");
    call_ = f[0];
    guard_ = f[1];
    recover_ = f[2];
    int threads = kWarpLanes * w_;
    max_blocks_ = sms * blocks_per_multiprocessor(f[0], threads * cpb_, smem_);
    rec_blocks_ = sms * blocks_per_multiprocessor(f[2], threads, smem_rec_);
    for (auto const& t : tables) tab_[t[1]] = read_table(folder, t).to(device);
    auto f64 = torch::TensorOptions().dtype(torch::kFloat64).device(device);
    scratch_ = torch::empty({max_blocks_ * cpb_ * stride_}, f64);
    scratch_rec_ = torch::empty({rec_blocks_ * stride_rec_}, f64);
    const int64_t slots = static_cast<int64_t>(lists["counts"].size() + kCountSlotsAroundNames);
    count_ = torch::zeros({slots}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    total_ = torch::zeros({slots}, torch::TensorOptions().dtype(torch::kInt64).device(device));
  } catch (...) {
    cuModuleUnload(mod);
    module_ = nullptr;
    throw;
  }
}

Solver::~Solver() {
  if (!module_) return;
  c10::cuda::CUDAGuard guard(device_);  // the module's context must be current
  cuModuleUnload(static_cast<CUmodule>(module_));
}

std::array<torch::Tensor, 4> Solver::operator()(torch::Tensor x, torch::Tensor T, torch::Tensor P, torch::Tensor dt,
                                                torch::Tensor h, torch::Tensor J, double rtol, double atol,
                                                int max_steps) {
  std::array<std::pair<char const*, torch::Tensor const*>, 5> given{
      {{"x", &x}, {"T", &T}, {"P", &P}, {"dt", &dt}, {"h", &h}}};
  for (auto const& [name, a] : given)
    TORCH_CHECK(a->device() == device_ && a->scalar_type() == torch::kFloat64, "maxichem: ", name, ": float64 on ",
                device_, " expected, got ", a->scalar_type(), " on ", a->device());
  TORCH_CHECK(x.dim() == 2 && x.size(1) == ns_, "maxichem: x: [B, ", ns_, "] expected, got ", x.sizes());
  TORCH_CHECK(x.size(0) <= std::numeric_limits<int>::max(), "maxichem: at most 2^31 - 1 cells per call, got ",
              x.size(0));
  for (auto const& [name, a] : given)
    TORCH_CHECK(a == &x || (a->dim() == 1 && a->size(0) == x.size(0)), "maxichem: ", name, ": [", x.size(0),
                "] expected, got ", a->sizes());
  if (atol < 0) atol = default_atol_;
  TORCH_CHECK(std::isfinite(rtol) && rtol > 0 && std::isfinite(atol) && atol > 0,
              "maxichem: rtol and atol finite and positive");
  TORCH_CHECK(max_steps >= 1 && max_steps <= std::numeric_limits<int>::max() / rec_steps_,
              "maxichem: max_steps from 1 to ", std::numeric_limits<int>::max() / rec_steps_);
  c10::cuda::CUDAGuard guard(device_);
  int B = static_cast<int>(x.size(0));
  x = x.contiguous();
  T = T.contiguous();
  P = P.contiguous();
  dt = dt.contiguous();
  h = h.contiguous();
  // the rule and the messages of check_photolysis in maxichem/_solver.py: without photolysis no J or an empty [B, 0]
  // one, with it J [B, nj]
  if (J.defined()) {
    TORCH_CHECK(nj_ || (J.dim() == 2 && J.size(1) == 0), "maxichem: J: none or an empty [", B,
                ", 0] one expected, since this solver has no photolysis; got ", J.sizes());
    TORCH_CHECK(J.device() == device_ && J.scalar_type() == torch::kFloat64 && J.dim() == 2 && J.size(0) == B &&
                    J.size(1) == nj_,
                "maxichem: J: float64 [", B, ", ", nj_, "] on ", device_, " expected, got ", J.scalar_type(), " ",
                J.sizes(), " on ", J.device());
    J = J.contiguous();
  } else {
    TORCH_CHECK(nj_ == 0, "maxichem: J [B, ", nj_, "] is required with photolysis: pass zeros for dark cells");
  }
  auto xo = torch::empty_like(x);
  auto ho = h.clone();
  auto i32 = x.options().dtype(torch::kInt32);
  auto done = torch::empty({B}, i32), redo = torch::empty({B}, i32), lst = torch::empty({B}, i32),
       status = torch::empty({B}, i32);
  auto nok = torch::empty({B}, x.options());
  count_.zero_();
  CUstream stream = reinterpret_cast<CUstream>(at::cuda::getCurrentCUDAStream().stream());

  // the address of each kernel parameter's value, by name; meta.txt gives each kernel's order
  std::deque<void*> pointers;
  std::map<std::string, void*> values;
  auto tensor = [&](std::string const& name, torch::Tensor const& t) {
    pointers.push_back(t.data_ptr());
    values[name] = &pointers.back();
  };
  for (auto const& [name, t] : tab_) tensor(name, t);
  std::array<std::pair<char const*, torch::Tensor const*>, 13> inputs{{{"x0", &x},
                                                                       {"Ta", &T},
                                                                       {"Pa", &P},
                                                                       {"dta", &dt},
                                                                       {"hin", &h},
                                                                       {"hio", &ho},
                                                                       {"xout", &xo},
                                                                       {"doneo", &done},
                                                                       {"noko", &nok},
                                                                       {"status", &status},
                                                                       {"redo", &redo},
                                                                       {"list", &lst},
                                                                       {"count", &count_}}};
  for (auto const& [name, t] : inputs) tensor(name, *t);
  if (nj_) tensor(kPhotolysisArgument, J);
  values["B"] = &B;
  values["rtol"] = &rtol;
  values["atol"] = &atol;
  values["sround"] = &sround_;
  values["jround"] = &jround_;
  auto launch = [&](void* fn, std::string const& kernel, unsigned grid, unsigned block, unsigned smem) {
    std::vector<void*> args;
    for (auto const& name : arguments_.at(kernel)) args.push_back(values.at(name));
    check(cuLaunchKernel(static_cast<CUfunction>(fn), grid, 1, 1, block, 1, 1, smem, stream, args.data(), nullptr),
          kernel.c_str());
  };

  values["max_steps"] = &max_steps;
  tensor("scratch", scratch_);
  launch(call_, "call", std::max(1, std::min((B + cpb_ - 1) / cpb_, max_blocks_)), kWarpLanes * w_ * cpb_, smem_);
  launch(guard_, "guard", std::max(1, (B + kGuardBlockThreads - 1) / kGuardBlockThreads), kGuardBlockThreads, 0);

  int recsteps = rec_steps_ * max_steps;
  values["max_steps"] = &recsteps;
  tensor("scratch", scratch_rec_);
  launch(recover_, "recover", std::max(1, std::min(B, rec_blocks_)), kWarpLanes * w_, smem_rec_);

  total_ += count_;
  refuse(x, T, P, dt, J, status);
  return {xo, ho, done.to(torch::kBool), nok};
}

void Solver::refuse(torch::Tensor const& x, torch::Tensor const& T, torch::Tensor const& P, torch::Tensor const& dt,
                    torch::Tensor const& J, torch::Tensor const& status) const {
  auto counts = count_.cpu();  // the call's one host sync
  int const bad = counts[bad_input_slot_].item<int>(), cold = counts[cold_slot_].item<int>(),
            hot = counts[hot_slot_].item<int>();
  if (bad == 0 && cold == 0 && hot == 0) return;
  // as CudaSolver: the call kernel screened each cell once (status, 0 when stepped); the host names the first refused
  // cell of each kind from it
  auto refused = status.ne(0);
  auto is_cold = refused & T.lt(t_min_), is_hot = refused & T.gt(t_max_);
  auto first = [](torch::Tensor const& mask) { return torch::nonzero(mask)[0][0].item<int64_t>(); };
  std::ostringstream m;
  m << "maxichem: the call refused cells: ";
  bool any = false;
  auto range = [&](int n, torch::Tensor const& cells, char const* side, double limit) {
    if (!n) return;
    if (any) m << "; ";
    any = true;
    int64_t const i = first(cells);
    m << n << " cells " << side << " " << limit << " K, outside the solver's range " << t_min_ << "-" << t_max_
      << " K (first: cell " << i << ", T = " << T[i].item<double>() << " K); leave them out of the call";
  };
  range(cold, is_cold, "below", t_min_);
  range(hot, is_hot, "above", t_max_);
  if (bad) {
    if (any) m << "; ";
    int64_t const i = first(refused & (is_cold | is_hot).logical_not());
    m << bad << " cells of bad input: x negative or non-finite, |sum(x) - 1| above " << x_sum_tol_
      << " or no x above 0; T non-finite, P non-finite, not above 0 or so large that P / (k_B T) overflows, dt negative"
      << " or non-finite, or J negative or non-finite "
      << "(first: cell " << i << ": sum(x) - 1 = " << (x[i].sum() - 1.0).item<double>()
      << " (added in another order than the kernel's, so it can differ in the last bits), min(x) = "
      << x[i].min().item<double>() << ", T = " << T[i].item<double>() << " K, P = " << P[i].item<double>()
      << " dyn cm^-2, dt = " << dt[i].item<double>() << " s";
    if (nj_) m << ", min(J) = " << J[i].min().item<double>() << " s^-1";
    m << "); pass x normalized to sum 1 in float64, clamped at 0";
  }
  throw std::invalid_argument(m.str());
}

}  // namespace maxichem
