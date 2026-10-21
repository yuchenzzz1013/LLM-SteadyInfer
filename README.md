# LLM-SteadyInfer

**轻量级 LLM Serving Runtime**，面向单 GPU 场景，支持 LLaMA / Qwen 系列模型的高效推理。

> **项目状态**：本项目**仍在持续开发中**。
> 当前已完成核心 Runtime 架构、CPU / CUDA 双后端、BF16 全链路推理（HF safetensors
> 权重,CUDA 全链路 BF16 存储 / Tensor Core 计算,CPU 设备 FP32 兼容后备）、Paged KV Cache、
> PagedAttention、Continuous Batching 调度器以及 Offline / Online / Prefix Caching
> Benchmark 系统。
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

- **KV 量化（S4：K int4 分组 + BF16 残差窗口, V int4 非对称）**
  - 只有一个量化形态，开关是 `LLAMA_KV_QUANT=0`（off）或 `6`（S4）。
  - K、V 都以 INT4 存储（每字节两个值，行宽 `kv_dim/2`）。
  - **K**：非对称量化，scale/zero-point 粒度 per-(token group, channel group)。
    最近 `group_size` 个 token 的 K 以 BF16 保存在残差池；一组写满后整组量化为
    INT4 并回写。`channel_group` 默认 8，`group_size` 默认等于 block size。
  - **V**：非对称量化，scale/zero-point 粒度 per-(token, kv head)，写出即量化，
    没有残差窗口。
  - 分组可由 `LLAMA_KV_GROUP_SIZE` / `LLAMA_KV_CHANNEL_GROUP` 调整；旧取值
    `1/2/3/4/5` 现在直接 FATAL（旧的 K int8 + V int4 形态已整体删除）。
  - 解量化统一为 `x = (q - zero) * scale`，`q ∈ [0,15]`。
  - 实测（Qwen3-4B / ShareGPT / 32 并发 × 4096 上下文 / block size 16 / A100,
    单遍 PPL 共 131040 token;PASS 为 4096 步 decode 的累计秒数;
    S4 用默认 `group_size=16, channel_group=8`):

    | kv_quant | KV 池 | 压缩比 | PASS (s) | PPL | ΔPPL |
    | --- | --- | --- | --- | --- | --- |
    | 0（BF16） | 18.00 GB | 1.00× | 79.6 | 14.614163 | 基准 |
    | 6（S4） | 5.10 GB | 3.53× | 190.8 | 14.707136 | +0.636% |

  - 注意：S4 的 decode 比 BF16 慢（INT4 没有 dp4a 可用，K/V 需要逐值展开）。
    收益在显存/容量，不在延迟。
  - 贪心逐位一致性验证（T7）：Qwen3-4B 上 5 条 prompt × 48 token 当前
    1/5 完全一致；其余从第 6–27 个 token 开始分叉。门禁**未达标**。



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
  - 配套 A/B Benchmark 量化 TTFT 与吞吐收益

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

### Offline Batch
* Output Token Throughput
* Total Token Throughput
* GPU Utilization
* MFU
* KV Cache Fragmentation
* TTFT / TPOT / ITL / E2E

### Online Serving
* TTFT
* TPOT
* E2E Latency
* Queue Wait / Queue Length
* Goodput
* SLA 达标率

### Prefix Caching
* Prefix Cache 命中率（lookups / hits / matched_blocks / inserts / evictions）
* TTFT / TPOT / ITL / E2E
* Output Token Throughput / Total Token Throughput / Completed RPS
* `skipped_prefill_ratio`

---

## Demo

### 连续对话

每轮携带全部历史（超出上下文预算时从最旧一轮整轮丢弃），并支持对话中热切换模型。

```bash
./build/demo/chat_demo                      # 实时对话,exit / quit 退出
./build/demo/chat_demo --max-new-tokens 256 --max-seq-len 1024
./build/demo/chat_demo --questions "你好|讲个笑话"   # 非交互,依次提问
./build/demo/chat_demo --help               # 全部参数(模型类型 / 目录 / 设备 / 思考模式…)
```
---

## 项目结构

```
LLM-SteadyInfer/
├── benchmark/                       # 性能基准测试与验证
│   ├── bench_common.h               # 数据集加载 / 统计工具 / NVML 采样
│   ├── offline_batch_benchmark.cpp  # 离线吞吐测试
│   ├── online_serving_benchmark.cpp # 在线 Serving 测试
│   ├── prefix_caching_benchmark.cpp # Prefix Caching A/B 收益测试
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