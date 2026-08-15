#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理 ShareGPT_V3_unfiltered_cleaned_split.json:
  1. 删除 conversations 中 "from": "gpt" 的条目及其 value,只保留 human 部分
  2. 将 human 部分改造成 LLM 推理框架测试数据集(每行一个 prompt)

输出文件(默认与输入同目录):
  - ShareGPT_prompts.jsonl : 测试数据集,每行一个 {"id": ..., "prompt": ...}
    (仅保留长度在 10-1024 字符之间的 prompt)

用法示例:
  python process_sharegpt.py
  python process_sharegpt.py -i ShareGPT_V3_unfiltered_cleaned_split.json
  python process_sharegpt.py --min-len 32 --max-len 2048 --limit 5000 --seed 42
"""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="清洗 ShareGPT 数据集并生成推理测试数据集")
    p.add_argument("-i", "--input", default="ShareGPT_V3_unfiltered_cleaned_split.json",
                   help="输入 ShareGPT JSON 文件路径")
    p.add_argument("-o", "--out-dir", default=None,
                   help="输出目录(默认与输入文件同目录)")
    p.add_argument("--min-len", type=int, default=10,
                   help="prompt 最小字符数,短于该值的样本丢弃(默认 10)")
    p.add_argument("--max-len", type=int, default=1024,
                   help="prompt 最大字符数,长于该值的样本丢弃(默认 1024)")
    p.add_argument("--limit", type=int, default=0,
                   help="最多保留 N 条样本(0 表示全部保留)")
    p.add_argument("--seed", type=int, default=42,
                   help="--limit 随机抽样时使用的随机种子")
    return p.parse_args()


def clean_conversations(sample):
    """只保留 human 的 value,返回 [value, ...];无有效 human 则返回空列表。"""
    convs = sample.get("conversations")
    if not isinstance(convs, list):
        return []
    humans = []
    for c in convs:
        if not isinstance(c, dict) or c.get("from") != "human":
            continue  # 删除 gpt / system 等所有非 human 条目
        v = c.get("value")
        if isinstance(v, str) and v.strip():
            humans.append(v)
    return humans


def main():
    args = parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"找不到输入文件: {src}")

    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取: {src}")
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("格式错误: 顶层应为 JSON 数组")

    total = len(data)
    prompts = []

    for sample in data:
        humans = clean_conversations(sample)
        if not humans:
            continue

        prompt = "\n\n".join(humans)  # 多条 human 消息合并为一个 prompt
        plen = len(prompt)
        if args.min_len > 0 and plen < args.min_len:
            continue
        if args.max_len > 0 and plen > args.max_len:
            continue

        prompts.append({"id": sample.get("id", ""), "prompt": prompt})

    # 可选随机抽样
    if args.limit > 0 and len(prompts) > args.limit:
        rng = random.Random(args.seed)
        idx = sorted(rng.sample(range(len(prompts)), args.limit))
        prompts = [prompts[i] for i in idx]

    # 唯一输出: 推理测试数据集(每行一个 prompt 的 JSONL)
    jsonl_out = out_dir / "ShareGPT_prompts.jsonl"
    with open(jsonl_out, "w", encoding="utf-8") as f:
        for item in prompts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"已写出: {jsonl_out} ({len(prompts)} 条)")

    # 统计信息
    if prompts:
        lens = [len(item["prompt"]) for item in prompts]
        print("\n===== 统计 =====")
        print(f"原始样本数      : {total}")
        print(f"保留样本数      : {len(prompts)}")
        print(f"丢弃样本数      : {total - len(prompts)}")
        print(f"prompt 平均长度 : {sum(lens) / len(lens):.1f} 字符")
        print(f"prompt 最短/最长: {min(lens)} / {max(lens)} 字符")


if __name__ == "__main__":
    main()
