# 基准测试程序 CMakeLists.txt，与 demo/CMakeLists.txt 风格一致

# 选项控制（默认开启，便于测试；您可根据需要关闭）
option(LLAMA3_SUPPORT  "Build LLaMA3 benchmark" ON)
option(QWEN2_SUPPORT   "Build Qwen2 benchmark" ON)
option(QWEN3_SUPPORT   "Build Qwen3 benchmark" ON)

# 公共依赖（放在外面，避免重复查找）
find_package(absl REQUIRED)
find_package(re2 REQUIRED)
find_package(nlohmann_json REQUIRED)

# ----------------------------------------------
# LLaMA3 基准
if (LLAMA3_SUPPORT)
    message(STATUS "Building LLaMA3 benchmark")
    add_executable(benchmark_llama benchmark_llama.cpp)
    target_link_directories(benchmark_llama PUBLIC ${PROJECT_SOURCE_DIR}/../lib)
    target_link_libraries(benchmark_llama
        llama
        absl::base
        re2::re2
        nlohmann_json::nlohmann_json
    )
    set_target_properties(benchmark_llama PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# ----------------------------------------------
# Qwen2 基准
if (QWEN2_SUPPORT)
    message(STATUS "Building Qwen2 benchmark")
    add_executable(benchmark_qwen2 benchmark_qwen2.cpp)
    target_link_directories(benchmark_qwen2 PUBLIC ${PROJECT_SOURCE_DIR}/../lib)
    target_link_libraries(benchmark_qwen2
        llama
        absl::base
        re2::re2
        nlohmann_json::nlohmann_json
    )
    set_target_properties(benchmark_qwen2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# ----------------------------------------------
# Qwen3 基准
if (QWEN3_SUPPORT)
    message(STATUS "Building Qwen3 benchmark")
    add_executable(benchmark_qwen3 benchmark_qwen3.cpp)
    target_link_directories(benchmark_qwen3 PUBLIC ${PROJECT_SOURCE_DIR}/../lib)
    target_link_libraries(benchmark_qwen3
        llama
        absl::base
        re2::re2
        nlohmann_json::nlohmann_json
    )
    set_target_properties(benchmark_qwen3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()