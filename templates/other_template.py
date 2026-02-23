#!/usr/bin/env python3
"""
{description}

实验: {exp_id}
输出: {other_name}

自动生成 by labshelf.py add-other
"""
import sys
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.shared.loaders import load_data
import yaml


def main():
    exp_dir = Path(__file__).resolve().parents[1]
    metadata = yaml.safe_load((exp_dir / "metadata.yaml").read_text())

    # --- 加载数据 ---
    # 通过 load_data(exp_dir, metadata, "逻辑名") 加载已注册的数据
    # 查看已注册数据: metadata["data"].keys()

    # --- 处理逻辑（请编辑此处）---
    result = None  # TODO: 在此添加处理代码

    # --- 保存 ---
    output_path = exp_dir / "others" / "{output_file}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # TODO: 将 result 写入 output_path
    print(f"已生成: {{output_path}}")


if __name__ == "__main__":
    main()
