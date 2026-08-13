# LLM-SteadyInfer

轻量级大模型推理框架，支持 LLaMA / Qwen 系列模型的高效推理。

> **项目状态**：本项目**仍在持续开发中**。当前已完成核心框架、双后端算子、量化推理及基础 Benchmark，后续将持续扩展功能、优化性能、支持更多模型,支持 Continuous Batching。

## 项目简介

LLM-SteadyInfer 是一个轻量级大语言模型推理框架，采用 **C++ / CUDA C++** 实现，支持 **CPU / CUDA 双后端**运行。框架支持 FP32 及 INT8 量化推理(LLaMA && Qwen2)，并提供了完整的推理流水线及性能基准测试工具。

## 支持模型

- LLaMA 系列
- Qwen 系列（Qwen2、Qwen3）

## 项目结构

```
LLM-SteadyInfer/
├── benchmark/          # 性能基准测试
│   ├── benchmark_llama.cpp
│   ├── benchmark_qwen2.cpp
│   └── benchmark_qwen3.cpp
├── cmake/              # CMake 构建配置
|
├── src/
│   ├── include/        # 头文件
│   └── source/         # 源文件
|
├── CMakeLists.txt
└── .gitignore
```


## 致谢

本项目在设计与实现过程中参考了 [KuiperLLama](https://github.com/zjhellofss/KuiperLLama)，特此致谢！

---
