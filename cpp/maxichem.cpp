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

constexpr int kExportFormat = 3;              // EXPORT_FORMAT in maxichem/_constants.py: any other export is refused
constexpr int kGuardBlockThreads = 128;       // any size works; this one packs cells densely
constexpr size_t kCountSlotsAroundNames = 2;  // the two cell queues around the export's "counts" names

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

}  // namespace

Solver::Solver(std::string const& folder, torch::Device device) : device_(device) {
  TORCH_CHECK(device.is_cuda(), "maxichem::Solver: a CUDA device");
  c10::cuda::CUDAGuard guard(device);
  device_ = torch::Device(torch::kCUDA, guard.current_device().index());  // "cuda": the current one
  device = device_;
  torch::zeros({1}, torch::TensorOptions().device(device));  // torch's context is current

  std::map<std::string, std::string> meta;
  std::vector<std::vector<std::string>> tables;
  std::map<std::string, std::vector<std::string>> lists;
  read_meta(folder + "/meta.txt", meta, tables, arguments_, lists);
  species = lists["species"];
  elements = lists["elements"];
  auto geti = [&](char const* k) {
    auto it = meta.find(k);
    if (it == meta.end()) throw std::runtime_error(std::string("maxichem: meta.txt lacks ") + k);
    return std::stoll(it->second);
  };
  if (geti("export_format") != kExportFormat)
    throw std::runtime_error("maxichem: export format " + meta.at("export_format") + ", this code reads " +
                             std::to_string(kExportFormat) + "; export again with this version");
  ns = static_cast<int>(geti("ns"));
  nj = static_cast<int>(geti("nj"));
  w_ = static_cast<int>(geti("w"));
  cpb_ = static_cast<int>(geti("cpb"));
  smem_ = static_cast<int>(geti("smem"));
  smem_rec_ = static_cast<int>(geti("smem_rec"));
  stride_ = geti("stride");
  stride_rec_ = geti("stride_rec");
  sround_ = static_cast<int>(geti("sround"));
  jround_ = static_cast<int>(geti("jround"));
  rec_steps_ = static_cast<int>(geti("rec_steps"));
  default_atol = std::stod(meta.at("atol"));
  if (static_cast<int>(species.size()) != ns)
    throw std::runtime_error("maxichem: meta.txt lists " + std::to_string(species.size()) + " species, ns is " +
                             std::to_string(ns));

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
    int threads = 32 * w_;
    max_blocks_ = sms * blocks_per_multiprocessor(f[0], threads * cpb_, smem_);
    rec_blocks_ = sms * blocks_per_multiprocessor(f[2], threads, smem_rec_);
    for (auto const& t : tables) tab_[t[1]] = read_table(folder, t).to(device);
    auto f64 = torch::TensorOptions().dtype(torch::kFloat64).device(device);
    scratch_ = torch::empty({max_blocks_ * cpb_ * stride_}, f64);
    scratch_rec_ = torch::empty({rec_blocks_ * stride_rec_}, f64);
    const int64_t slots = static_cast<int64_t>(lists["counts"].size() + kCountSlotsAroundNames);
    count = torch::zeros({slots}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    total = torch::zeros({slots}, torch::TensorOptions().dtype(torch::kInt64).device(device));
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
  TORCH_CHECK(x.dim() == 2 && x.size(1) == ns, "maxichem: x: [B, ", ns, "] expected, got ", x.sizes());
  TORCH_CHECK(x.size(0) <= std::numeric_limits<int>::max(), "maxichem: at most 2^31 - 1 cells per call, got ",
              x.size(0));
  for (auto const& [name, a] : given)
    TORCH_CHECK(a == &x || (a->dim() == 1 && a->size(0) == x.size(0)), "maxichem: ", name, ": [", x.size(0),
                "] expected, got ", a->sizes());
  if (atol < 0) atol = default_atol;
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
  if (nj == 0) {
    TORCH_CHECK(!J.defined(), "maxichem: this solver has no photolysis");
  } else {
    TORCH_CHECK(J.defined(), "maxichem: J [B, ", nj, "] is required for a network exported with photo=True");
    TORCH_CHECK(J.device() == device_ && J.scalar_type() == torch::kFloat64 && J.dim() == 2 && J.size(0) == B &&
                    J.size(1) == nj,
                "maxichem: J: float64 [", B, ", ", nj, "] on ", device_, " expected, got ", J.scalar_type(), " ",
                J.sizes(), " on ", J.device());
    J = J.contiguous();
  }
  auto xo = torch::empty_like(x);
  auto ho = h.clone();
  auto i32 = x.options().dtype(torch::kInt32);
  auto done = torch::empty({B}, i32), redo = torch::empty({B}, i32), lst = torch::empty({B}, i32);
  auto nok = torch::empty({B}, x.options());
  count.zero_();
  CUstream stream = reinterpret_cast<CUstream>(at::cuda::getCurrentCUDAStream().stream());

  // the address of each kernel parameter's value, by name; meta.txt gives each kernel's order
  std::deque<void*> pointers;
  std::map<std::string, void*> values;
  auto tensor = [&](std::string const& name, torch::Tensor const& t) {
    pointers.push_back(t.data_ptr());
    values[name] = &pointers.back();
  };
  for (auto const& [name, t] : tab_) tensor(name, t);
  std::array<std::pair<char const*, torch::Tensor const*>, 12> inputs{{{"x0", &x},
                                                                       {"Ta", &T},
                                                                       {"Pa", &P},
                                                                       {"dta", &dt},
                                                                       {"hin", &h},
                                                                       {"hio", &ho},
                                                                       {"xout", &xo},
                                                                       {"doneo", &done},
                                                                       {"noko", &nok},
                                                                       {"redo", &redo},
                                                                       {"list", &lst},
                                                                       {"count", &count}}};
  for (auto const& [name, t] : inputs) tensor(name, *t);
  if (nj) tensor("Jin", J);
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
  launch(call_, "call", std::max(1, std::min((B + cpb_ - 1) / cpb_, max_blocks_)), 32 * w_ * cpb_, smem_);
  launch(guard_, "guard", std::max(1, (B + kGuardBlockThreads - 1) / kGuardBlockThreads), kGuardBlockThreads, 0);

  int recsteps = rec_steps_ * max_steps;
  values["max_steps"] = &recsteps;
  tensor("scratch", scratch_rec_);
  launch(recover_, "recover", std::max(1, std::min(B, rec_blocks_)), 32 * w_, smem_rec_);

  total += count;
  return {xo, ho, done.to(torch::kBool), nok};
}

}  // namespace maxichem
