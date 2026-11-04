# maxichem's C++ caller: a library (maxichem.hpp) and a test that compares it with the Python solver.
#   cmake -S cpp -B cpp/build -DPython3_EXECUTABLE=<python with torch> -DCUDAToolkit_ROOT=<CUDA toolkit>
cmake_minimum_required(VERSION 3.18)
project(maxichem_cpp LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(CUDAToolkit REQUIRED)
execute_process(COMMAND ${Python3_EXECUTABLE} -c
  "import torch, os; print(os.path.dirname(torch.__file__)); print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE _torch OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REPLACE "\n" ";" _torch "${_torch}")
list(GET _torch 0 TORCH_ROOT)
list(GET _torch 1 TORCH_ABI)
add_library(maxichem_cpp SHARED maxichem.cpp)
target_include_directories(maxichem_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
  ${TORCH_ROOT}/include ${TORCH_ROOT}/include/torch/csrc/api/include)
target_compile_definitions(maxichem_cpp PUBLIC _GLIBCXX_USE_CXX11_ABI=${TORCH_ABI} USE_CUDA)
target_link_directories(maxichem_cpp PUBLIC ${TORCH_ROOT}/lib)
target_link_libraries(maxichem_cpp PUBLIC torch torch_cpu torch_cuda c10 c10_cuda CUDA::cuda_driver CUDA::cudart)
set_target_properties(maxichem_cpp PROPERTIES BUILD_RPATH "${TORCH_ROOT}/lib")
add_executable(test_export test_export.cpp)
target_link_libraries(test_export PRIVATE maxichem_cpp)
set_target_properties(test_export PROPERTIES BUILD_RPATH "${TORCH_ROOT}/lib;${CMAKE_CURRENT_BINARY_DIR}")
