#include "maxichem.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace maxichem {

namespace {

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

}  // namespace

Solver::Solver(std::string const& folder, torch::Device device) : device_(device) {
  TORCH_CHECK(device.is_cuda(), "maxichem::Solver: a CUDA device");
  c10::cuda::CUDAGuard guard(device);
  device_ = torch::Device(torch::kCUDA, guard.current_device().index());  // "cuda": the current one
  device = device_;
  torch::zeros({1}, torch::TensorOptions().device(device));  // torch's context is current

  // meta.txt: "key value..." per line
  std::map<std::string, std::string> meta;
  std::vector<std::vector<std::string>> tables;
  std::istringstream lines(read_file(folder + "/meta.txt"));
  for (std::string line; std::getline(lines, line);) {
    std::istringstream ws(line);
    std::vector<std::string> tok;
    for (std::string t; ws >> t;) tok.push_back(t);
    if (tok.empty()) continue;
    if (tok[0] == "table") {
      tables.push_back(tok);
    } else if (tok[0] == "species" || tok[0] == "elements") {
      (tok[0] == "species" ? species : elements).assign(tok.begin() + 1, tok.end());
    } else if (tok.size() >= 2) {
      meta[tok[0]] = tok[1];
    }
  }
  auto geti = [&](char const* k) {
    auto it = meta.find(k);
    if (it == meta.end()) throw std::runtime_error(std::string("maxichem: meta.txt lacks ") + k);
    return std::stoll(it->second);
  };
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

  CUdevice dev;
  check(cuDeviceGet(&dev, device_.index()), "device");
  int major = 0, minor = 0, sms = 0;
  check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev), "attribute");
  check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev), "attribute");
  check(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev), "attribute");
  if (std::to_string(major * 10 + minor) != meta.at("arch"))
    throw std::runtime_error("maxichem: the export is for sm_" + meta.at("arch") + ", this device is sm_" +
                             std::to_string(major * 10 + minor) + "; export on this machine");

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
    // as CudaSolver: the dynamic shared memory opt-in, then blocks from the occupancy
    int nl = 32 * w_;
    int occ[2];
    CUfunction fn2[2] = {f[0], f[2]};
    int threads[2] = {nl * cpb_, nl};
    int smem[2] = {smem_, smem_rec_};
    for (int i = 0; i < 2; ++i) {
      check(cuFuncSetAttribute(fn2[i], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem[i]), "shared memory");
      check(cuOccupancyMaxActiveBlocksPerMultiprocessor(&occ[i], fn2[i], threads[i], smem[i]), "occupancy");
      if (occ[i] < 1) throw std::runtime_error("maxichem: the kernel does not fit on this device");
    }
    max_blocks_ = sms * occ[0];
    rec_blocks_ = sms * occ[1];

    for (auto const& t : tables) {  // table name dtype dims...
      std::vector<int64_t> shape;
      for (size_t i = 3; i < t.size(); ++i) shape.push_back(std::stoll(t[i]));
      auto dt = dtype_of(t[2]);
      auto cpu = torch::empty(shape, torch::TensorOptions().dtype(dt));
      std::string bytes = read_file(folder + "/tables/" + t[1] + ".bin");
      if (bytes.size() != static_cast<size_t>(cpu.nbytes()))
        throw std::runtime_error("maxichem: table " + t[1] + " has the wrong size");
      std::memcpy(cpu.data_ptr(), bytes.data(), bytes.size());
      tab_[t[1]] = cpu.to(device);
    }
    auto f64 = torch::TensorOptions().dtype(torch::kFloat64).device(device);
    scratch_ = torch::empty({max_blocks_ * cpb_ * stride_}, f64);
    scratch_rec_ = torch::empty({rec_blocks_ * stride_rec_}, f64);
    count = torch::zeros({10}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    total = torch::zeros({10}, torch::TensorOptions().dtype(torch::kInt64).device(device));
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
  for (auto const* a : {&x, &T, &P, &dt, &h})
    TORCH_CHECK(a->device() == device_ && a->scalar_type() == torch::kFloat64,
                "maxichem: x, T, P, dt, h: float64 on the solver's device");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == ns, "maxichem: x [B, ", ns, "]");
  if (atol < 0) atol = default_atol;
  c10::cuda::CUDAGuard guard(device_);
  int B = static_cast<int>(x.size(0));
  x = x.contiguous();
  T = T.contiguous();
  P = P.contiguous();
  dt = dt.contiguous();
  h = h.contiguous();
  if (nj == 0) {
    TORCH_CHECK(!J.defined(), "maxichem: this solver has no photolysis");
  } else if (!J.defined()) {
    J = torch::zeros({B, nj}, x.options());
  } else {
    TORCH_CHECK(J.device() == device_ && J.scalar_type() == torch::kFloat64 && J.dim() == 2 && J.size(0) == B &&
                    J.size(1) == nj,
                "maxichem: J [B, ", nj, "] float64 on the solver's device");
    J = J.contiguous();
  }
  auto xo = torch::empty_like(x);
  auto ho = h.clone();
  auto i32 = x.options().dtype(torch::kInt32);
  auto done = torch::empty({B}, i32), redo = torch::empty({B}, i32), lst = torch::empty({B}, i32);
  auto nok = torch::empty({B}, x.options());
  count.zero_();
  CUstream stream = reinterpret_cast<CUstream>(at::cuda::getCurrentCUDAStream().stream());

  // pointers and scalars, in CudaSolver.launch's order
  std::vector<void*> ptrs;
  ptrs.reserve(64);
  std::vector<void*> args;
  auto P_ = [&](torch::Tensor const& t) {
    ptrs.push_back(t.data_ptr());
    args.push_back(&ptrs.back());
  };
  int recsteps = rec_steps_ * max_steps;
  auto launch = [&](void* fn, unsigned grid, unsigned block, unsigned smem, char const* what) {
    check(cuLaunchKernel(static_cast<CUfunction>(fn), grid, 1, 1, block, 1, 1, smem, stream, args.data(), nullptr),
          what);
    args.clear();
    ptrs.clear();
  };

  // call
  for (auto const* t : {&x, &T, &P, &dt, &ho, &xo, &done, &nok, &count}) P_(*t);
  args.push_back(&B);
  args.push_back(&rtol);
  args.push_back(&atol);
  args.push_back(&max_steps);
  for (char const* n : {"perm", "blob", "rc", "rf", "nasa"}) P_(tab_.at(n));
  P_(scratch_);
  for (char const* n : {"sst", "jt", "lu", "optr"}) P_(tab_.at(n));
  args.push_back(&sround_);
  args.push_back(&jround_);
  if (nj) P_(J);
  int nb = std::max(1, std::min((B + cpb_ - 1) / cpb_, max_blocks_));
  launch(call_, nb, 32 * w_ * cpb_, smem_, "launch call");

  // guard
  for (auto const* t : {&x, &T, &P, &dt, &xo, &done, &nok}) P_(*t);
  P_(tab_.at("atoms"));
  for (auto const* t : {&redo, &lst, &count}) P_(*t);
  args.push_back(&B);
  args.push_back(&atol);
  if (nj) P_(J);
  launch(guard_, std::max(1, (B + 127) / 128), 128, 0, "launch guard");

  // recover
  for (auto const* t : {&x, &T, &P, &dt, &h, &xo, &ho, &done, &nok, &redo, &lst, &count}) P_(*t);
  args.push_back(&rtol);
  args.push_back(&atol);
  args.push_back(&recsteps);
  for (char const* n : {"perm", "blob", "rc", "rf", "nasa"}) P_(tab_.at(n));
  P_(scratch_rec_);
  for (char const* n : {"sst", "jt", "didx", "atoms"}) P_(tab_.at(n));
  args.push_back(&sround_);
  args.push_back(&jround_);
  if (nj) P_(J);
  launch(recover_, std::max(1, std::min(B, rec_blocks_)), 32 * w_, smem_rec_, "launch recover");

  total += count;
  return {xo, ho, done.to(torch::kBool), nok};
}

}  // namespace maxichem
