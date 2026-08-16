# ============================================================================
# 统一 Benchmark 构建脚本(双模式)
#
# 包含目标:
#   benchmark_llama / benchmark_qwen2 / benchmark_qwen3   单模型快速压测
#   offline_batch_benchmark                               离线批推理全指标压测
#
# 模式 1 — 根项目子目录(推荐):根 CMakeLists.txt 中 add_subdirectory 引入,
#   直接复用根项目的 llama target 与 CPM 依赖。
# 模式 2 — 独立构建:在本目录内 `cmake .. && make`,只引用根项目已构建的产物
#   - ${ROOT}/build/lib/libllama.so 或 ${ROOT}/lib/libllama.so   引擎动态库
#   - ${ROOT}/build/_deps/*             已构建的第三方库
#   - ${ROOT}/src/include               引擎头文件
#   (使用前请确认根项目已完整构建过一次)
# ============================================================================

cmake_minimum_required(VERSION 3.16)

# 判断构建模式:顶层 cmake 时两者相等;被 add_subdirectory 引入时不相等
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(BENCH_STANDALONE ON)
else()
  set(BENCH_STANDALONE OFF)
endif()

# 选项控制(默认开启;作为根项目子目录时沿用根项目 cache 中的同名选项,
# 即引擎编译时启用的模型集合与压测程序自动保持一致)
option(LLAMA3_SUPPORT        "Build LLaMA3 benchmark"            ON)
option(QWEN2_SUPPORT         "Build Qwen2 benchmark"             ON)
option(QWEN3_SUPPORT         "Build Qwen3 benchmark"             ON)
option(OFFLINE_BATCH_BENCHMARK "Build offline batch benchmark"   ON)
option(ONLINE_SERVING_BENCHMARK "Build online serving benchmark (vLLM-style)" ON)

# ----------------------------------------------
# 公共依赖:两种模式统一收敛到一个 INTERFACE 目标 bench_deps,
# 各可执行目标只需链接 bench_deps。
# ----------------------------------------------
if(BENCH_STANDALONE)
  # ---------- 独立模式:建立子项目并定位根项目构建产物 ----------
  project(benchmark CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
  get_filename_component(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)
  set(DEPS ${ROOT}/build/_deps)
  if(NOT EXISTS ${DEPS})
    message(FATAL_ERROR
      "未找到 ${DEPS} — 请先在项目根目录完成一次完整构建")
  endif()

  add_library(bench_deps INTERFACE)

  # ---------- 编译选项(与根项目 benchmark 目标保持一致) ----------
  # 引擎无条件编译全部三个模型,这里同时定义三个 SUPPORT 宏,与引擎产物对齐。
  target_compile_definitions(bench_deps INTERFACE
    GLOG_USE_GLOG_EXPORT LLAMA3_SUPPORT QWEN2_SUPPORT QWEN3_SUPPORT)

  # ---------- 头文件 ----------
  target_include_directories(bench_deps INTERFACE
    ${ROOT}/src/include
    ${DEPS}/sentencepiece-src/src
    ${DEPS}/absl-src
    ${DEPS}/re2-src
    ${DEPS}/nlohmann_json-src/include
    ${DEPS}/glog-build
    ${DEPS}/glog-src/src
    ${DEPS}/armadillo-build/tmp/include)

  # ---------- CUDA ----------
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    target_include_directories(bench_deps INTERFACE
      ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(bench_deps INTERFACE
      CUDA::cudart CUDA::cublas)
    list(APPEND RPATH_DIRS ${CUDAToolkit_LIBRARY_DIR})
  else()
    message(WARNING "未找到 CUDAToolkit,回退到 /usr/local/cuda 硬编码路径")
    target_include_directories(bench_deps INTERFACE
      /usr/local/cuda/include)
    target_link_libraries(bench_deps INTERFACE cudart cublas)
    list(APPEND RPATH_DIRS /usr/local/cuda/lib64)
  endif()

  # ---------- 引擎动态库 ----------
  if(EXISTS ${ROOT}/build/lib/libllama.so)
    set(LLAMA_LIB ${ROOT}/build/lib/libllama.so)
  elseif(EXISTS ${ROOT}/lib/libllama.so)
    set(LLAMA_LIB ${ROOT}/lib/libllama.so)
  else()
    message(FATAL_ERROR "未找到 libllama.so (查找了 ${ROOT}/build/lib 与 ${ROOT}/lib)")
  endif()
  get_filename_component(LLAMA_LIB_DIR ${LLAMA_LIB} DIRECTORY)
  target_link_libraries(bench_deps INTERFACE ${LLAMA_LIB})
  list(APPEND RPATH_DIRS ${LLAMA_LIB_DIR})

  # ---------- 第三方动态库(链接 libllama.so 未直接依赖的 absl/re2 等) ----------
  # 注意顺序:libllama.so 在前,其未定义的 absl 符号由后面的库满足。
  file(GLOB ABSL_LIBS ${DEPS}/absl-build/absl/*/libabsl_*.so)
  foreach(lib IN LISTS ABSL_LIBS)
    target_link_libraries(bench_deps INTERFACE ${lib})
    get_filename_component(d ${lib} DIRECTORY)
    list(APPEND RPATH_DIRS ${d})
  endforeach()

  foreach(lib
      ${DEPS}/re2-build/libre2.so
      ${DEPS}/sentencepiece-build/src/libsentencepiece.so
      ${DEPS}/glog-build/libglog.so
      ${DEPS}/armadillo-build/libarmadillo.so)
    if(EXISTS ${lib})
      target_link_libraries(bench_deps INTERFACE ${lib})
      get_filename_component(d ${lib} DIRECTORY)
      list(APPEND RPATH_DIRS ${d})
    endif()
  endforeach()

  # ---------- OpenMP ----------
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(bench_deps INTERFACE OpenMP::OpenMP_CXX)
  else()
    target_compile_options(bench_deps INTERFACE -fopenmp)
    target_link_options(bench_deps INTERFACE -fopenmp)
  endif()

else()
  # ---------- 根项目子目录模式 ----------
  # 头文件/编译选项/CUDA/引擎依赖均从 llama target 的 PUBLIC 接口继承,
  # 这里只需补上 libllama 未直接链接的 absl/re2/nlohmann_json。
  find_package(absl REQUIRED)
  find_package(re2 REQUIRED)
  find_package(nlohmann_json REQUIRED)
  add_library(bench_deps INTERFACE)
  target_link_libraries(bench_deps INTERFACE
    llama
    absl::base
    re2::re2
    nlohmann_json::nlohmann_json)
endif()

# ---------- 公共依赖(两种模式都需要):Threads + NVML ----------
# NVML 供 offline/online 压测的 GPU 利用率后台采样线程使用;
# 找不到则自动降级(仅缺失该指标),通过 BENCH_NO_NVML 宏关闭对应代码路径。
find_package(Threads REQUIRED)
target_link_libraries(bench_deps INTERFACE Threads::Threads)
find_library(NVML_LIB nvidia-ml
  PATHS /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64)
if(NVML_LIB)
  target_link_libraries(bench_deps INTERFACE ${NVML_LIB})
else()
  message(WARNING "未找到 libnvidia-ml,GPU 利用率指标将不可用")
  target_compile_definitions(bench_deps INTERFACE BENCH_NO_NVML)
endif()

# ----------------------------------------------
# 各可执行目标:统一链接 bench_deps
# ----------------------------------------------
# LLaMA3 基准 (支持 --batch 多批次 continuous batching 模式)
if (LLAMA3_SUPPORT)
    message(STATUS "Building LLaMA3 benchmark")
    add_executable(benchmark_llama benchmark_llama.cpp)
    target_link_libraries(benchmark_llama PRIVATE bench_deps)
    set_target_properties(benchmark_llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Qwen2 基准
if (QWEN2_SUPPORT)
    message(STATUS "Building Qwen2 benchmark")
    add_executable(benchmark_qwen2 benchmark_qwen2.cpp)
    target_link_libraries(benchmark_qwen2 PRIVATE bench_deps)
    set_target_properties(benchmark_qwen2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Qwen3 基准
if (QWEN3_SUPPORT)
    message(STATUS "Building Qwen3 benchmark")
    add_executable(benchmark_qwen3 benchmark_qwen3.cpp)
    target_link_libraries(benchmark_qwen3 PRIVATE bench_deps)
    set_target_properties(benchmark_qwen3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# 离线批推理基准 (离线压测全指标:吞吐/延迟/KV 碎片/硬件利用率)
if (OFFLINE_BATCH_BENCHMARK)
    message(STATUS "Building offline batch benchmark")
    add_executable(offline_batch_benchmark offline_batch_benchmark.cpp)
    target_link_libraries(offline_batch_benchmark PRIVATE bench_deps)
    set_target_properties(offline_batch_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# 在线 Serving 基准 (vLLM benchmark_serving 风格:泊松到达/排队/Goodput)
if (ONLINE_SERVING_BENCHMARK)
    message(STATUS "Building online serving benchmark")
    add_executable(online_serving_benchmark online_serving_benchmark.cpp)
    target_link_libraries(online_serving_benchmark PRIVATE bench_deps)
    set_target_properties(online_serving_benchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# Token 一致性 A/B 验证驱动 (分页布局 vs 连续布局门禁)
option(VERIFY_TOKENS "Build the token-consistency verify driver" ON)
if (VERIFY_TOKENS)
    message(STATUS "Building verify_tokens driver")
    add_executable(verify_tokens verify_tokens.cpp)
    target_link_libraries(verify_tokens PRIVATE bench_deps)
    set_target_properties(verify_tokens PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# ---------- 独立模式:运行时库搜索路径(作用于全部目标) ----------
if(BENCH_STANDALONE)
  list(REMOVE_DUPLICATES RPATH_DIRS)
  foreach(tgt benchmark_llama benchmark_qwen2 benchmark_qwen3 offline_batch_benchmark online_serving_benchmark verify_tokens)
    if(TARGET ${tgt})
      set_target_properties(${tgt} PROPERTIES
        BUILD_RPATH "${RPATH_DIRS}"
        INSTALL_RPATH_USE_LINK_PATH TRUE)
    endif()
  endforeach()
endif()
