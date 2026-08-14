cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the CUDA compiler path or name
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(llama_infer CXX CUDA)
include(cmake/cuda.cmake)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

option(LLAMA3_SUPPORT OFF)
if (LLAMA3_SUPPORT)
  message(STATUS "LLAMA3 SUPPORT")
  add_definitions(-DLLAMA3_SUPPORT)
endif()
option(QWEN2_SUPPORT OFF)
if (QWEN2_SUPPORT)
  message(STATUS "QWEN2 SUPPORT")
  add_definitions(-DQWEN2_SUPPORT)
endif()

option(QWEN3_SUPPORT OFF)
if (QWEN3_SUPPORT)
  message(STATUS "QWEN3 SUPPORT")
  add_definitions(-DQWEN3_SUPPORT)
endif()
# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info
option(USE_CPM "Use CPM for dependency management" OFF)

if(USE_CPM)
  # Use CPM to manage dependencies
  include(cmake/CPM.cmake)

  # 定义本地依赖目录（项目根目录下的 deps 文件夹）
  set(DEPS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps")

  CPMAddPackage(
    NAME glog
    URL "file://${DEPS_DIR}/v0.7.1.tar.gz"
    OPTIONS "BUILD_TESTING Off"
  )

  CPMAddPackage(
    NAME Armadillo
    URL "file://${DEPS_DIR}/armadillo-code-14.0.1.tar.gz"
  )

  CPMAddPackage(
    NAME sentencepiece
    URL "file://${DEPS_DIR}/v0.2.0.tar.gz"
  )
  find_package(sentencepiece REQUIRED)

  CPMAddPackage(
      NAME absl
      URL "file://${DEPS_DIR}/20240722.0.tar.gz"
      OPTIONS "BUILD_TESTING Off" "ABSL_PROPAGATE_CXX_STD ON" "ABSL_ENABLE_INSTALL ON"
  )
  CPMAddPackage(
      NAME re2
      URL "file://${DEPS_DIR}/2024-07-02.tar.gz"
  )
  CPMAddPackage(
      NAME nlohmann_json
      URL "file://${DEPS_DIR}/v3.11.3.tar.gz"
  )
endif()

# ---------------------------------
find_package(glog REQUIRED)
find_package(Armadillo REQUIRED)

# OpenMP: the CPU continuous-batching path (P0) parallelizes batched GEMM,
# RoPE and attention over sequences/heads.
find_package(OpenMP QUIET)

aux_source_directory(src/source/tensor/ DIR_TENSOR)
aux_source_directory(src/source/base/ DIR_BASE)
aux_source_directory(src/source/op/ DIR_OP)
aux_source_directory(src/source/model/ DIR_MODEL)
aux_source_directory(src/source/scheduler/ DIR_SCHEDULER)
aux_source_directory(src/source/op/kernels/cpu DIR_KERNEL_CPU)
aux_source_directory(src/source/op/kernels/cuda DIR_KERNEL_CUDA)
aux_source_directory(src/source/op/kernels/ DIR_KERNEL)
aux_source_directory(src/source/sampler DIR_SAMPLE)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(llama SHARED ${DIR_TENSOR} ${DIR_BASE} ${DIR_OP} ${DIR_KERNEL} ${DIR_MODEL} ${DIR_KERNEL_CPU} ${DIR_KERNEL_CUDA} ${DIR_KERNEL} ${DIR_SAMPLE} ${DIR_SCHEDULER})
target_link_libraries(llama sentencepiece glog::glog pthread cudart cublas armadillo)
if(OpenMP_CXX_FOUND)
  target_link_libraries(llama OpenMP::OpenMP_CXX)
endif()
target_link_directories(llama PUBLIC ${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}/lib64)

target_include_directories(llama PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${PROJECT_SOURCE_DIR}/src/include)
target_include_directories(llama PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(llama PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(USE_CPM)
  # Add sentencepiece include directory
  target_include_directories(llama PUBLIC ${sentencepiece_SOURCE_DIR}/src)
  # encode.cpp uses absl, re2, and nlohmann_json — expose their include dirs
  target_include_directories(llama PUBLIC ${absl_SOURCE_DIR})
  target_include_directories(llama PUBLIC ${re2_SOURCE_DIR})
  target_include_directories(llama PUBLIC ${nlohmann_json_SOURCE_DIR}/include)
endif()

set_target_properties(llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_subdirectory(benchmark)