# ============================================================================
# CUDA toolchain configuration
#
# The CUDA language itself is enabled by the top-level project() call; the
# toolkit is located by find_package(CUDAToolkit) there. This module pins the
# one thing the build must NOT take from the build machine: the target
# architecture list.
#
# It used to be produced by CUDA_DETECT_INSTALLED_GPUS() — i.e. by the GPUs
# physically present on the build host — so a release built on an A100 embedded
# sm_80 only and refused to load ("no kernel image is available for execution on
# the device") on any other GPU, while a GPU-less build machine (CI) could not
# pick an architecture at all. The list below is fixed at configure time and
# always contains sm_80; override it for a specific deployment with
#   -DCMAKE_CUDA_ARCHITECTURES="80;90"
# ============================================================================

if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR
    "No CUDA compiler was found. Install the CUDA toolkit (nvcc) and make sure "
    "it is on PATH, or point CMake at it with -DCMAKE_CUDA_COMPILER=/path/to/nvcc.")
endif()

# sm_80 (A100) is mandatory — the runtime's reference target. The rest cover the
# Ampere / Ada / Hopper parts the runtime is deployed on. `native` is deliberately
# NOT used: it re-introduces the build-machine dependency this module removes.
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90" CACHE STRING
      "CUDA architectures to embed in the fatbin (semicolon-separated, e.g. \"80;90\")")
endif()

message(STATUS "Found CUDA Toolkit v${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
