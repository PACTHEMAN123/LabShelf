#!/usr/bin/env python3
"""{description}

自动生成 by labshelf.py add-script
"""
import json
import sys
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.shared.loaders import load_data


def main():
    args = json.loads(sys.argv[1])
    exp_dir = Path(args["exp_dir"])
    output_dir = Path(args["output_dir"])
    inputs = args["inputs"]  # {{"logical_name": "path/to/file", ...}}

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 处理逻辑（请编辑此处）---
    # inputs 中包含所有输入数据的路径
    # 使用 load_data() 加载数据:
    #   import yaml
    #   metadata = yaml.safe_load((exp_dir / "metadata.yaml").read_text())
    #   data = load_data(exp_dir, metadata, "逻辑名")

    # TODO: 在此添加处理/绘图代码

    # --- 保存输出 ---
    # 将结果保存到 output_dir 下
    # 例如:
    #   (output_dir / "result.json").write_text(json.dumps(result))
    #   fig.savefig(output_dir / "plot.png")
    print(f"完成: {{output_dir}}")


if __name__ == "__main__":
    main()
