// ============================================================================
// Benchmark 公共工具(offline / serving 压测共享,header-only)
//   参数解析 / 默认路径推断 / ShareGPT 数据集加载 / 统计工具 /
//   NVML GPU 利用率后台采样 / MFU 硬件算力估算
// 编译宏 BENCH_NO_NVML:未找到 libnvidia-ml 时由 CMake 定义,自动降级。
// ============================================================================
#pragma once

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#ifndef BENCH_NO_NVML
#include <nvml.h>
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace bench {

using json = nlohmann::json;

// ============================== 参数解析 ==============================

inline std::string get_arg(int argc, char** argv, const std::string& name,
                           const std::string& def = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == name) return argv[i + 1];
  }
  return def;
}

inline bool has_arg(int argc, char** argv, const std::string& name) {
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == name) return true;
  }
  return false;
}

// std::stoi / std::stod throw on bad input and nothing above catches them (the
// process just aborts): report the offending value and exit instead.
inline int to_int(const std::string& s) {
  try {
    return std::stoi(s);
  } catch (const std::exception&) {
    std::cerr << "[bench] 非法整数参数: \"" << s << "\"\n";
    std::exit(2);
  }
  return 0;  // unreachable
}

inline double to_double(const std::string& s) {
  try {
    return std::stod(s);
  } catch (const std::exception&) {
    std::cerr << "[bench] 非法浮点参数: \"" << s << "\"\n";
    std::exit(2);
  }
  return 0.0;  // unreachable
}

// 默认路径解析:优先按 CWD 解释,失败则相对可执行文件推断出的项目根目录
// (build/benchmark/<bin> 与 benchmark/build/<bin> 两种布局均适用)
inline std::string resolve_default(const std::string& path,
                                   const std::string& exe_root) {
  if (exe_root.empty()) return path;
  std::ifstream probe(path);
  if (probe.good()) return path;
  std::string alt = exe_root + "/" + path;
  probe.open(alt);
  if (probe.good()) return alt;
  return path;  // 保持原值,后续读文件时会给出明确报错
}

// ============================== 数据集 ==============================

struct Request {
  std::string id;
  std::string prompt;
};

// 读取 ShareGPT_prompts.jsonl,每行 {"id":..., "prompt":...}
inline std::vector<Request> load_dataset(const std::string& path) {
  std::vector<Request> reqs;
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "[bench] 无法打开数据集: " << path << "\n";
    return reqs;
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    try {
      auto j = json::parse(line);
      std::string prompt = j.value("prompt", "");
      if (prompt.empty()) continue;
      Request r;
      r.id = j.value("id", "");
      r.prompt = prompt;
      reqs.push_back(std::move(r));
    } catch (const json::exception& e) {
      std::cerr << "[bench] 跳过无法解析的行: " << e.what() << "\n";
    }
  }
  return reqs;
}

// ============================== 统计工具 ==============================

// Nearest-rank percentile. Takes the samples by value: they must be sorted,
// and callers reuse their vectors (latency lists) after the call.
inline double percentile(const std::vector<double>& data, double p) {
  if (data.empty()) return 0.0;
  std::vector<double> sorted = data;
  std::sort(sorted.begin(), sorted.end());
  // ceil(p/100 * n) - 1 clamped to [0, n-1]: p == 0 (and any p below the first
  // rank) must return the minimum, not the maximum.
  const double rank = std::ceil(p / 100.0 * static_cast<double>(sorted.size())) - 1.0;
  size_t idx = rank < 0.0 ? 0 : static_cast<size_t>(rank);
  if (idx >= sorted.size()) idx = sorted.size() - 1;
  return sorted[idx];
}

inline double mean(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

// ============================== GPU 利用率采样(NVML) ==============================

class GpuUtilSampler {
 public:
  // 启动后台采样线程(每 50ms 一次),失败返回 false(GPU 利用率指标将不可用)
  bool start() {
#ifndef BENCH_NO_NVML
    if (nvmlInit_v2() != NVML_SUCCESS) return false;
    if (nvmlDeviceGetHandleByIndex_v2(0, &dev_) != NVML_SUCCESS) return false;
    stop_.store(false);
    th_ = std::thread([this] { run(); });
    return true;
#else
    return false;
#endif
  }

  void stop() {
#ifndef BENCH_NO_NVML
    stop_.store(true);
    if (th_.joinable()) th_.join();
    nvmlShutdown();
#endif
  }

  double sm_util_pct() const { return samples_ > 0 ? sm_sum_ / samples_ : 0.0; }
  double mem_util_pct() const { return samples_ > 0 ? mem_sum_ / samples_ : 0.0; }
  double mem_used_mb() const { return samples_ > 0 ? mem_used_mb_sum_ / samples_ : 0.0; }
  int samples() const { return samples_; }

 private:
#ifndef BENCH_NO_NVML
  void run() {
    while (!stop_.load()) {
      nvmlUtilization_t u{};
      nvmlMemory_t m{};
      bool got = false;
      if (nvmlDeviceGetUtilizationRates(dev_, &u) == NVML_SUCCESS) {
        sm_sum_ += u.gpu;
        mem_sum_ += u.memory;
        got = true;
      }
      if (nvmlDeviceGetMemoryInfo(dev_, &m) == NVML_SUCCESS) {
        mem_used_mb_sum_ += static_cast<double>(m.used) / (1024.0 * 1024.0);
        got = true;
      }
      // Only count ticks that actually produced a sample — a failed query must
      // not dilute the averages.
      if (got) {
        samples_++;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
  nvmlDevice_t dev_{};
#endif
  std::atomic<bool> stop_{true};
  std::thread th_;
  double sm_sum_ = 0.0, mem_sum_ = 0.0, mem_used_mb_sum_ = 0.0;
  std::atomic<int> samples_{0};
};

// ============================== 硬件峰值算力 / MFU ==============================

// 每 SM 的 FP32 core 数(按计算能力查表)
inline int cores_per_sm(int major, int minor) {
  int cc = major * 10 + minor;
  if (cc <= 21) return 8;
  if (cc <= 30) return 192;
  if (cc <= 35) return 192;
  if (cc <= 50) return 128;
  if (cc <= 52) return 128;
  if (cc <= 60) return 64;
  if (cc <= 61) return 128;
  if (cc <= 70) return 64;
  if (cc <= 75) return 64;
  if (cc <= 80) return 64;
  return 128;  // 8.6 及之后
}

// 理论峰值 FP32 FLOPS(cores x 2 FMA x 时钟)
inline double peak_fp32_tflops() {
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 0.0;
  double cores = static_cast<double>(prop.multiProcessorCount) *
                 cores_per_sm(prop.major, prop.minor);
  double clock_ghz = prop.clockRate / 1e6;  // kHz -> GHz
  return cores * 2.0 * clock_ghz / 1000.0;  // TFLOPS
}

// 每 SM 每周期 BF16 Tensor Core 稠密 FLOPs(查表,按 NVIDIA 官方表格折算;
// 均为带 FP32 累加的 dense 值,不含结构化稀疏的 2 倍):
//   A100(cc 8.0) 2048:108 SM x 1.41GHz x 2048 ≈ 312 TFLOPS dense;
//   消费级 Ampere(8.6)/Ada(8.9) 512:RTX 4090 约 128 SM x 2.52GHz x 512 ≈ 165;
//   Hopper(9.0)及之后 4096:H100 SXM 132 SM x 1.83GHz x 4096 ≈ 990 TFLOPS dense。
// 数据中心 Ampere 的 Tensor Core 吞吐是消费级的 4 倍,二者不能共用一个常数;
// 前代(Volta/Turing)无 BF16 Tensor Core,返回 0。
inline double bf16_flops_per_sm(int major, int minor) {
  int cc = major * 10 + minor;
  if (cc >= 90) return 4096;              // Hopper(90)及之后按 4096 起步
  if (cc == 80) return 2048;              // A100 / A30(数据中心 Ampere)
  if (cc == 86 || cc == 89) return 512;   // 消费级 Ampere / Ada
  return 0;                               // 前代无 BF16 Tensor Core
}

// 理论峰值 BF16 TFLOPS(仅 BF16 模型 MFU 用;架构不支持时返回 0)
inline double peak_bf16_tflops() {
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 0.0;
  const double per_sm = bf16_flops_per_sm(prop.major, prop.minor);
  if (per_sm <= 0) return 0.0;
  return per_sm * prop.multiProcessorCount * prop.clockRate / 1e9;  // TFLOPS
}

struct ModelDims {
  double d = 0, ffn = 0, head_size = 0;
  int layers = 0, heads = 0, kv_heads = 0;
};

// 从 HF 模型目录读取 config.json(亦兼容直接传入 config.json 文件路径)
inline bool load_model_dims(const std::string& path, ModelDims& m) {
  std::filesystem::path p(path);
  std::ifstream f;
  if (std::filesystem::is_directory(p)) {
    f.open(p / "config.json");
  } else {
    f.open(p);
  }
  if (!f.is_open()) return false;
  try {
    auto j = json::parse(f);
    m.d = j.at("hidden_size");
    m.ffn = j.at("intermediate_size");
    m.layers = j.at("num_hidden_layers");
    m.heads = j.at("num_attention_heads");
    m.kv_heads = j.at("num_key_value_heads");
    m.head_size = j.contains("head_dim") ? static_cast<double>(j["head_dim"])
                                         : m.d / m.heads;
    return m.d > 0 && m.layers > 0;
  } catch (const json::exception& e) {
    std::cerr << "[bench] 解析 model config 失败: " << e.what() << "\n";
    return false;
  }
}

// 每个 token 的 FLOPs 近似(decoder 前向,含 QKV/out proj + FFN;
// 与 "2 x 参数量" 的经典 MFU 估算等价,attention 部分对短序列可忽略)
inline double flops_per_token(const ModelDims& m) {
  double attn = 4.0 * m.d * m.head_size * (m.heads + m.kv_heads);
  double ffn = 6.0 * m.d * m.ffn;
  return m.layers * (attn + ffn);
}

}  // namespace bench
