# LLM-SteadyInfer

轻量级大模型推理框架，支持 LLaMA / Qwen 系列模型的高效推理。

> **项目状态**：本项目**仍在持续开发中，远未结束**。当前已完成核心框架、双后端算子、量化推理及基础 Benchmark，后续将持续扩展功能、优化性能、支持更多模型。

## 项目简介

LLM-SteadyInfer 是一个轻量级大语言模型推理框架，采用 **C++ / CUDA C++** 实现，支持 **CPU / CUDA 双后端**运行。框架支持 FP32 及 INT8 量化推理，并提供了完整的推理流水线及性能基准测试工具。

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
├── demo/               # 示例程序
│   ├── chat_qwen.cpp
│   ├── main_llama.cpp
│   ├── main_qwen2.cpp
│   └── main_qwen3.cpp
├── src/
│   ├── include/        # 头文件
│   └── source/         # 源文件
├── test/               # 单元测试
│   ├── optimized/
│   ├── test_cu/
│   ├── test_op/
│   └── test_tensor/
├── CMakeLists.txt
└── .gitignore
```


## 致谢

本项目在设计与实现过程中参考了 [KuiperLLama](https://github.com/zjhellofss/KuiperLLama)，特此致谢！

---

如果觉得“后续计划”过于琐碎，也可只保留项目状态中的“远未结束”一句，其他部分不动。当前版本已清楚表明项目是 ongoing 状态。
