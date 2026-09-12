# LLM-SteadyInfer

**轻量级 LLM Serving Runtime**，面向单 GPU 场景，支持 LLaMA / Qwen 系列模型的高效推理。

> **项目状态**：本项目**仍在持续开发中**。
> 当前已完成核心 Runtime 架构、CPU / CUDA 双后端、BF16 全链路推理（HF safetensors
> 权重,CUDA 全链路 BF16 存储 / Tensor Core 计算,CPU 设备 FP32 兼容后备）、Paged KV Cache、
> PagedAttention、Continuous Batching 调度器以及 Offline / Online Benchmark 系统。
> 后续将持续扩展模型支持、优化推理性能并完善 Serving 能力。

---

## 项目简介

LLM-SteadyInfer 是一个基于 **C++ / CUDA C++** 实现的轻量级大语言模型推理 Runtime，
面向单 GPU 环境提供完整的 LLM Serving 链路。

框架支持 **CPU / CUDA 双后端**，
覆盖模型加载、算子执行、KV Cache 管理、请求调度与性能评估等核心模块。

目前支持：

- BF16 推理（HF safetensors 模型目录加载;CUDA 全链路 BF16——权重 / 激活 /
  KV Cache / logits 均 BF16,Tensor Core 数学运算;CPU 设备自动回退 FP32,
  权重加载时转换）
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

精度支持：

- BF16（CUDA:全链路权重 / 激活 / KV Cache / RoPE sin-cos 表均 BF16）
- FP32（后备:仅 CPU 设备,权重加载时转换;CUDA 侧不保留任何 FP32 算子）
- INT8 Group-wise Quantization（LLaMA / Qwen2,未随 BF16 迁移,已禁用）

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
  -DUSE_PAGED_ATTENTION=ON

make -j$(nproc)
````

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

## Demo

### 连续对话（命令行实时 Chat）

`demo/chat_demo.cpp` 是一个最小可跑的连续对话示例：ChatML 模板拼 prompt →
`model->encode` → `Scheduler::add_request` → `step()` → 逐 token 流式打印。
每轮携带全部历史（超出上下文预算时从最旧一轮整轮丢弃），并支持对话中热切换模型。

```bash
./build/demo/chat_demo                      # 实时对话,exit / quit 退出
./build/demo/chat_demo --max-new-tokens 256 --max-seq-len 1024
./build/demo/chat_demo --questions "你好|讲个笑话"   # 非交互,依次提问
./build/demo/chat_demo --help               # 全部参数(模型类型 / 目录 / 设备 / 思考模式…)
```

常用参数：

* `--model-type qwen3|qwen2|llama`、`--model-dir <目录>`、`--tokenizer <文件>`：选择模型
* `--max-seq-len`（默认 0 = 自适应）：上下文容量，同时容纳历史与生成长度。
  自适应取「模型窗口」与「显存放得下」的较小值，启动时会打印 KV 池大小与整卡占用
* `--max-batch`（默认 1）：调度器槽位数；KV 池按 `max_batch × max_seq_len` 整块预分配，
  想多占显存就调大（例如 40G 卡上 `--max-batch 8` 会把 KV 池撑到 ~26 GB，整卡占用约 93%）
* `--gpu-mem-fraction`（默认 0.9）：KV 池最多占空闲显存的比例
* `--device cuda|cpu`：推理设备

> 长上下文依赖模型 config 的 `max_position_embeddings`：本仓库 `Qwen3-4B/config.json`
> 已设为官方值 40960，所以上下文可以开到 40960；更小的窗口模型会自动按窗口截断。
> 注意单流对话是延迟受限的，显存可以吃满，但 GPU 算力吃不满（要算力得上并发 batching）。

对话中的命令：`/model`（查看或热切换模型）、`/history`、`/clear`、`exit`。

---

## 项目结构

```
LLM-SteadyInfer/
├── benchmark/                       # 性能基准测试与验证
│   ├── bench_common.h
│   ├── offline_batch_benchmark.cpp  # 离线吞吐测试
│   ├── online_serving_benchmark.cpp # 在线 Serving 测试
│   └── verify_tokens.cpp            # Token 一致性验证
├── demo/                            # 示例
│   └── chat_demo.cpp                # 命令行实时连续对话
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