# LLM-SteadyInfer

轻量级大模型推理框架，支持 LLaMA / Qwen 系列模型的高效推理。

> **项目状态**：本项目**仍在持续开发中**。当前已完成核心框架、双后端算子、量化推理、分页注意力（PagedAttention）、Continuous Batching 调度器及离线/在线 Benchmark，后续将持续扩展功能、优化性能、支持更多模型。

## 项目简介

LLM-SteadyInfer 是一个轻量级大语言模型推理框架，采用 **C++ / CUDA C++** 实现，支持 **CPU / CUDA 双后端**运行。框架支持 FP32 及 INT8 量化推理（LLaMA && Qwen2），并提供了完整的推理流水线及性能基准测试工具。

核心特性：

- **分页 KV Cache**：块式显存池 + BlockTable 间接寻址，消除连续 KV 布局的显存碎片；block size 按工作负载自适应
- **PagedAttention 内核**：支持 GQA 与 Flash Decoding 风格的分裂规约
- **Continuous Batching**：Prefill / Decode 混合调度，Chunked Prefill + 自适应 token 预算
- **抢占调度**：无空闲块时驱逐低优序列尾块，序列支持 WAITING / RUNNING / PREEMPTED / FINISHED 状态机
- **Prefix Caching**：按内容哈希复用公共前缀的物理块，跳过重复 Prefill
- **CUDA Graph**：Decode 路径图捕获与重放，多 batch size 图池化
- **Fused QKV**：Q / K / V 投影合并为单次 GEMM

## 支持模型

- LLaMA 系列
- Qwen 系列（Qwen2、Qwen3）

## 项目结构

```
LLM-SteadyInfer/
├── benchmark/                      # 性能基准测试与验证驱动
│   ├── bench_common.h              # 公共工具
│   ├── offline_batch_benchmark.cpp # 离线批推理全指标压测
│   ├── online_serving_benchmark.cpp # 在线 Serving 压测
│   └── verify_tokens.cpp           # Token 一致性 A/B 验证
├── cmake/                          # CMake 构建配置
|
├── src/
│   ├── include/                    # 头文件（base / model / op / sampler / scheduler / tensor）
│   └── source/                     # 源文件（含 op/kernels/cuda 下的 CUDA 内核）
├── tools/                          # 数据集预处理脚本
├── CMakeLists.txt
└── README.md
```

## 致谢

本项目在设计与实现过程中参考了 [KuiperLLama](https://github.com/zjhellofss/KuiperLLama)，特此致谢！

---
