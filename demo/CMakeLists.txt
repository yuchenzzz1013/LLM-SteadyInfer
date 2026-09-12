# ============================================================================
# Demo 构建脚本(双模式,与 benchmark/CMakeLists.txt 同款约定)
#
# 包含目标:
#   chat_demo        连续对话(多轮 Chat)最小示例
#
# 模式 1 — 根项目子目录(推荐):根 CMakeLists.txt 中 add_subdirectory 引入,
#   直接复用根项目的 llama target 与依赖。
# 模式 2 — 独立构建:在本目录内 `cmake .. && make`,只引用根项目已构建的产物
#   - ${ROOT}/build/lib/libllama.so 或 ${ROOT}/lib/libllama.so   引擎动态库
#   - ${ROOT}/build/_deps/*             已构建的第三方库
#   - ${ROOT}/src/include               引擎头文件
#   (使用前请确认根项目已完整构建过一次)
# ============================================================================

cmake_minimum_required(VERSION 3.16)

# 判断构建模式:顶层 cmake 时两者相等;被 add_subdirectory 引入时不相等
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(DEMO_STANDALONE ON)
else()
  set(DEMO_STANDALONE OFF)
endif()

if(DEMO_STANDALONE)
  # ---------- 独立模式:建立子项目并定位根项目构建产物 ----------
  project(llama_demo CXX)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
  endif()
  get_filename_component(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)
  set(DEPS ${ROOT}/build/_deps)
  if(NOT EXISTS ${DEPS})
    message(FATAL_ERROR "未找到 ${DEPS} — 请先在项目根目录完成一次完整构建")
  endif()

  add_library(demo_deps INTERFACE)

  target_compile_definitions(demo_deps INTERFACE GLOG_USE_GLOG_EXPORT)

  # ---------- 头文件 ----------
  target_include_directories(demo_deps INTERFACE
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
    target_include_directories(demo_deps INTERFACE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(demo_deps INTERFACE CUDA::cudart CUDA::cublas)
    list(APPEND RPATH_DIRS ${CUDAToolkit_LIBRARY_DIR})
  else()
    message(WARNING "未找到 CUDAToolkit,回退到 /usr/local/cuda 硬编码路径")
    target_include_directories(demo_deps INTERFACE /usr/local/cuda/include)
    target_link_libraries(demo_deps INTERFACE cudart cublas)
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
  target_link_libraries(demo_deps INTERFACE ${LLAMA_LIB})
  list(APPEND RPATH_DIRS ${LLAMA_LIB_DIR})

  # ---------- 第三方动态库(libllama.so 未直接链接的 absl / re2 等) ----------
  # 注意顺序:libllama.so 在前,其未定义的 absl 符号由后面的库满足。
  file(GLOB ABSL_LIBS ${DEPS}/absl-build/absl/*/libabsl_*.so)
  foreach(lib IN LISTS ABSL_LIBS)
    target_link_libraries(demo_deps INTERFACE ${lib})
    get_filename_component(d ${lib} DIRECTORY)
    list(APPEND RPATH_DIRS ${d})
  endforeach()

  foreach(lib
      ${DEPS}/re2-build/libre2.so
      ${DEPS}/sentencepiece-build/src/libsentencepiece.so
      ${DEPS}/glog-build/libglog.so
      ${DEPS}/armadillo-build/libarmadillo.so)
    if(EXISTS ${lib})
      target_link_libraries(demo_deps INTERFACE ${lib})
      get_filename_component(d ${lib} DIRECTORY)
      list(APPEND RPATH_DIRS ${d})
    endif()
  endforeach()

  # ---------- OpenMP ----------
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(demo_deps INTERFACE OpenMP::OpenMP_CXX)
  else()
    target_compile_options(demo_deps INTERFACE -fopenmp)
    target_link_options(demo_deps INTERFACE -fopenmp)
  endif()
else()
  # ---------- 根项目子目录模式 ----------
  # 头文件 / 编译选项 / CUDA / 引擎依赖均从 llama target 的 PUBLIC 接口继承,
  # 这里只需补上 libllama 未直接链接的 absl / re2 / nlohmann_json。
  find_package(absl REQUIRED)
  find_package(re2 REQUIRED)
  find_package(nlohmann_json REQUIRED)
  add_library(demo_deps INTERFACE)
  target_link_libraries(demo_deps INTERFACE
    llama
    absl::base
    re2::re2
    nlohmann_json::nlohmann_json)
endif()

# ---------- 公共依赖(两种模式都需要):Threads ----------
find_package(Threads REQUIRED)
target_link_libraries(demo_deps INTERFACE Threads::Threads)

# ----------------------------------------------
# 目标:连续对话 Demo
# ----------------------------------------------
message(STATUS "Building chat_demo (连续对话示例)")
add_executable(chat_demo chat_demo.cpp)
target_link_libraries(chat_demo PRIVATE demo_deps)
set_target_properties(chat_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# ---------- 独立模式:运行时库搜索路径 ----------
if(DEMO_STANDALONE)
  list(REMOVE_DUPLICATES RPATH_DIRS)
  set_target_properties(chat_demo PROPERTIES
    BUILD_RPATH "${RPATH_DIRS}"
    INSTALL_RPATH_USE_LINK_PATH TRUE)
endif()
