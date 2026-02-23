#!/usr/bin/env python3
"""
{description}

实验: {exp_id}
图表: {fig_name}
数据输入: {data_inputs}

自动生成 by labshelf.py add-script
"""
import sys
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.shared.loaders import load_data
from scripts.shared.plot_utils import setup_style, save_figure
import matplotlib.pyplot as plt
import yaml


def main():
    exp_dir = Path(__file__).resolve().parents[1]
    metadata = yaml.safe_load((exp_dir / "metadata.yaml").read_text())

    setup_style()

    # --- 加载数据 ---
{load_lines}

    # --- 绘图逻辑（请编辑此处）---
    fig, ax = plt.subplots()
    ax.set_title("{fig_name}")
    # TODO: 在此添加绘图代码

    # --- 保存 ---
    save_figure(fig, exp_dir / "figures" / "{output_file}")


if __name__ == "__main__":
    main()
