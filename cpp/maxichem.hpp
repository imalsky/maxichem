#pragma once

// maxichem from C++: the kernels a Python CudaSolver exported with CudaSolver.export(folder), launched on
// torch's current CUDA stream with the arguments CudaSolver.launch passes, so a call gives the same bits.
//
//   maxichem::Solver chem("export/NCHO_photo", torch::Device(torch::kCUDA, 0));
//   auto [x1, h1, done, steps] = chem(x, T, P, dt, h, J);
//
// x [B, ns] mole fractions in the network's species order, T [B] K, P [B] dyn cm^-2, dt [B] s, h [B] the
// starting step sizes (all float64 on the solver's device); J [B, nj] s^-1, required for a network exported with
// photo=True (zeros for dark cells). Returns the new mole fractions, the next step sizes, done (bool) and the accepted
// steps per cell (float64). `count` holds the call's counts on the device (int32: queue, then bad_input, cold,
// nonfinite, unfinished, drift, recomputed, recovered, still_failed, nonfinite_h, then the recovery's queue) and
// `total` their sum since construction (int64). The call reads the counts back (its one host sync) and throws
// std::invalid_argument if it refused a cell: bad input (x negative or non-finite, |sum(x) - 1| above the export's
// x_sum_tol, no x above 0, T non-finite, P <= 0, dt negative or non-finite, J negative or non-finite) or T below
// t_min. The message names the counts, the limits and the first such cell; the counts stay readable.
//
// The cubin is for one GPU architecture (meta.txt "arch"): export on the machine that runs it.
// Calls share the scratch and the counts, so they run one at a time: on one stream, or synchronize before
// calling from another (as in Python).

#include <torch/torch.h>

#include <array>
#include <map>
#include <string>
#include <vector>

namespace maxichem {

// DEFAULT_RTOL and DEFAULT_MAX_STEPS in maxichem/_constants.py; tests/test_export_cpp.py calls with them.
inline constexpr double kDefaultRtol = 1e-3;
inline constexpr int kDefaultMaxSteps = 4000;

class Solver {
 public:
  Solver(std::string const& folder, torch::Device device);
  ~Solver();
  Solver(Solver const&) = delete;
  Solver& operator=(Solver const&) = delete;

  std::array<torch::Tensor, 4> operator()(torch::Tensor x, torch::Tensor T, torch::Tensor P, torch::Tensor dt,
                                          torch::Tensor h, torch::Tensor J = torch::Tensor(),
                                          double rtol = kDefaultRtol, double atol = -1.0,
                                          int max_steps = kDefaultMaxSteps);

  int ns = 0;                                  // species
  std::vector<std::string> species, elements;  // names; the atoms table is species x elements
  int nj = 0;                                  // photolysis branches (0: thermal network)
  double default_atol = 0.0;
  //! species x elements (float64, on the device): atoms of each element in each species
  torch::Tensor atoms() const { return tab_.at("atoms"); }
  torch::Tensor count, total;

 private:
  torch::Device device_;
  int w_ = 0, cpb_ = 0, smem_ = 0, smem_rec_ = 0, sround_ = 0, jround_ = 0;
  int rec_steps_ = 0;
  double t_min_ = 0.0, x_sum_tol_ = 0.0;        // K; the refusal's limits, from meta.txt
  int64_t bad_input_slot_ = 0, cold_slot_ = 0;  // their counts' slots
  int64_t stride_ = 0, stride_rec_ = 0;
  int max_blocks_ = 0, rec_blocks_ = 0;
  void* module_ = nullptr;  // CUmodule
  void* call_ = nullptr;    // CUfunction
  void* guard_ = nullptr;
  void* recover_ = nullptr;
  std::map<std::string, torch::Tensor> tab_;
  std::map<std::string, std::vector<std::string>> arguments_;  // each kernel's parameters, in order
  torch::Tensor scratch_, scratch_rec_;
  // throws std::invalid_argument if the call just launched refused a cell (bad input or below t_min)
  void refuse(torch::Tensor const& x, torch::Tensor const& T, torch::Tensor const& P, torch::Tensor const& dt,
              torch::Tensor const& J, torch::Tensor const& done, torch::Tensor const& redo) const;
};

}  // namespace maxichem
