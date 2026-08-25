# LLM-SteadyInfer

**轻量级 LLM Serving Runtime**，面向单 GPU 场景，支持 LLaMA / Qwen 系列模型的高效推理。

> **项目状态**：本项目**仍在持续开发中**。
> 当前已完成核心 Runtime 架构、CPU / CUDA 双后端、FP32 / INT8（LLaMA & Qwen2）推理、
> Paged KV Cache、PagedAttention、Continuous Batching 调度器以及 Offline / Online Benchmark 系统。
> 后续将持续扩展模型支持、优化推理性能并完善 Serving 能力。

---

## 项目简介

LLM-SteadyInfer 是一个基于 **C++ / CUDA C++** 实现的轻量级大语言模型推理 Runtime，
面向单 GPU 环境提供完整的 LLM Serving 链路。

框架支持 **CPU / CUDA 双后端**，
覆盖模型加载、算子执行、KV Cache 管理、请求调度与性能评估等核心模块。

目前支持：

- FP32 推理
- INT8 Group-wise Quantization 推理（LLaMA / Qwen2）
- LLaMA3 / Qwen2 / Qwen3 模型架构

---

## 核心特性

- **Paged KV Cache**
  - 基于 Block Memory Pool 管理 KV Cache
  - 通过 BlockTable 实现逻辑 Token Block 到物理 KV Block 的间接映射
  - 支持动态 Block Allocation、Block Reuse、Prefix KV Sharing
  - 根据请求负载动态调整 Block Size，降低 KV Cache 内存碎片

- **PagedAttention Kernel**
  - 实现基于 CUDA 的 PagedAttention 推理内核
  - 支持 GQA（Grouped Query Attention）
  - 采用 Flash Decoding 风格 Split-KV 并行与 Online Softmax
  - 支持 Partial Attention Merge 优化长序列 Decode

- **Continuous Batching Scheduler**
  - 状态机驱动请求调度
  - 支持 Prefill / Decode 混合执行
  - 支持 Chunked Prefill 与 Dynamic Token Budget
  - 根据 KV Cache 资源状态进行动态调度

- **Preemption 调度**
  - 支持 WAITING / RUNNING / PREEMPTED / FINISHED 请求状态管理
  - 基于 recompute 的抢占机制释放 KV Cache 资源
  - 支持被抢占序列重新调度恢复执行

- **Prefix Caching**
  - 基于内容 Hash 匹配公共 Prompt 前缀
  - 复用已有 KV Block，减少重复 Prefill 计算

- **CUDA Graph Optimization**
  - Decode 路径支持 CUDA Graph Capture / Replay
  - 基于 Batch Size 管理 Graph Cache
  - 降低 Decode 阶段 Kernel Launch Overhead

- **Kernel Optimization**
  - 支持 Fused QKV Projection
  - 基于 cuBLAS TF32 GEMM 与 CUDA Kernel 实现核心算子
  - 覆盖 Attention、RoPE、RMSNorm、SwiGLU 等推理算子

---

## 支持模型

当前支持：

- LLaMA 系列
- Qwen 系列
  - Qwen2
  - Qwen3

量化支持：

- FP32 推理
- INT8 Group-wise Quantization
  - LLaMA
  - Qwen2

---

## 环境要求与编译

### 依赖

- Linux
- g++ 9+
- C++17
- CUDA 11.8+
- CMake 3.16+

---

### 构建

```bash
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CPM=ON \
  -DUSE_PAGED_ATTENTION=ON \
  -DLLAMA3_SUPPORT=ON \
  -DQWEN2_SUPPORT=ON \
  -DQWEN3_SUPPORT=ON

make -j$(nproc)
````

---

### 主要选项说明

| 选项                    | 作用                               |
| --------------------- | -------------------------------- |
| `USE_CPM`             | 自动管理第三方依赖                        |
| `USE_PAGED_ATTENTION` | 开启 Paged KV Cache 与 Attention 优化 |
| `LLAMA3_SUPPORT`      | 开启 LLaMA3 模型支持                   |
| `QWEN2_SUPPORT`       | 开启 Qwen2 模型支持                    |
| `QWEN3_SUPPORT`       | 开启 Qwen3 模型支持                    |

---

## Benchmark

提供 Offline Batch 与 Online Serving 两类性能测试：

### Offline Batch

用于测试最大吞吐能力：

* Output Token Throughput
* Total Token Throughput
* GPU Utilization
* MFU
* KV Cache Fragmentation

### Online Serving

模拟真实请求到达：

* TTFT
* TPOT
* E2E Latency
* Goodput
* SLA 达标率

---

## 项目结构

```
LLM-SteadyInfer/
├── benchmark/                       # 性能基准测试与验证
│   ├── bench_common.h
│   ├── offline_batch_benchmark.cpp  # 离线吞吐测试
│   ├── online_serving_benchmark.cpp # 在线 Serving 测试
│   └── verify_tokens.cpp            # Token 一致性验证
├── cmake/                           # CMake 构建配置
├── src/
│   ├── include/
│   │   ├── base/                    # Runtime 基础组件
│   │   ├── model/                   # 模型实现
│   │   ├── op/                      # 算子接口
│   │   ├── scheduler/               # Serving Scheduler
│   │   └── tensor/                  # Tensor 抽象
│   └── source/
│       └── op/kernels/cuda/         # CUDA Kernel 实现
├── tools/                           # 数据处理工具
├── CMakeLists.txt
└── README.md
```

---

## 致谢

> 感谢 [vLLM](https://github.com/vllm-project/vllm) 与
> [KuiperLLama](https://github.com/zjhellofss/KuiperLLama)
> 项目在 LLM Serving 架构、KV Cache 管理以及轻量级 Runtime 设计方面提供的参考与启发。

```