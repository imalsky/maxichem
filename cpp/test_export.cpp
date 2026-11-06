// Runs maxichem::Solver on inputs the Python solver saved and compares the results bit for bit.
//   test_export <export folder> <case folder>
// The case folder holds raw little-endian files written by tests/cpp_export_case.py: the inputs x.bin [B, ns],
// T.bin, P.bin, dt.bin, h.bin [B], J.bin [B, nj] (photolysis networks; all float64), the Python call's results
// x_out.bin, h_out.bin, steps_out.bin (float64), done_out.bin (int8) and counts_out.bin (int32 [11]), and B.txt and
// max_steps.txt. The call runs twice, and both are compared in full (counts included); the second is timed.
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "maxichem.hpp"

static torch::Tensor load(std::string const& path, std::vector<int64_t> shape, torch::Dtype dt) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  std::string b = ss.str();
  auto t = torch::empty(shape, torch::TensorOptions().dtype(dt));
  if (b.size() != static_cast<size_t>(t.nbytes())) throw std::runtime_error("wrong size: " + path);
  std::memcpy(t.data_ptr(), b.data(), b.size());
  return t;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: test_export <export folder> <case folder>\n";
    return 2;
  }
  std::string ex = argv[1], cs = argv[2];
  torch::Device dev(torch::kCUDA, 0);
  maxichem::Solver chem(ex, dev);
  int64_t B = 0;
  if (!(std::ifstream(cs + "/B.txt") >> B) || B < 1) throw std::runtime_error("cannot read " + cs + "/B.txt");
  auto f64 = torch::kFloat64;
  auto x = load(cs + "/x.bin", {B, chem.ns}, f64).to(dev);
  auto T = load(cs + "/T.bin", {B}, f64).to(dev);
  auto P = load(cs + "/P.bin", {B}, f64).to(dev);
  auto dt = load(cs + "/dt.bin", {B}, f64).to(dev);
  auto h = load(cs + "/h.bin", {B}, f64).to(dev);
  torch::Tensor J;
  if (chem.nj) J = load(cs + "/J.bin", {B, chem.nj}, f64).to(dev);
  int max_steps = 4000;
  {
    std::ifstream ms(cs + "/max_steps.txt");
    if (ms) ms >> max_steps;
  }
  auto r = chem(x, T, P, dt, h, J, 1e-3, -1.0, max_steps);
  auto counts = chem.count.cpu().clone();
  torch::cuda::synchronize();
  auto t0 = std::chrono::steady_clock::now();
  auto r2 = chem(x, T, P, dt, h, J, 1e-3, -1.0, max_steps);
  auto counts2 = chem.count.cpu().clone();
  torch::cuda::synchronize();
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  auto xo = load(cs + "/x_out.bin", {B, chem.ns}, f64);
  auto ho = load(cs + "/h_out.bin", {B}, f64);
  auto so = load(cs + "/steps_out.bin", {B}, f64);
  auto dn = load(cs + "/done_out.bin", {B}, torch::kInt8).to(torch::kBool);
  auto co = load(cs + "/counts_out.bin", {11}, torch::kInt32);
  auto same = [](torch::Tensor a, torch::Tensor b) {
    return torch::equal(a.cpu().contiguous().view(torch::kInt64), b.contiguous().view(torch::kInt64));
  };
  auto all_same = [&](std::array<torch::Tensor, 4> const& q, torch::Tensor const& c) {
    return same(q[0], xo) && same(q[1], ho) && same(q[3], so) && torch::equal(q[2].cpu(), dn) && torch::equal(c, co);
  };
  bool ok = all_same(r, counts) && all_same(r2, counts2);
  std::cout << "cells " << B << ", species " << chem.ns << ", branches " << chem.nj << ": C++ call "
            << (ok ? "bitwise identical to the Python call" : "DIFFERENT") << "; second call " << ms << " ms ("
            << 1e3 * ms / B << " us per cell)"
            << "; max_steps " << max_steps << "; counts " << counts.view({1, 11}) << std::endl;
  return ok ? 0 : 1;
}
