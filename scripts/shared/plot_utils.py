"""通用绘图工具模块"""
from pathlib import Path


def setup_style():
    """设置统一绘图风格。"""
    import matplotlib.pyplot as plt
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass  # 使用默认风格

    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


def save_figure(fig, path, dpi=150, fmt=None):
    """保存图表并打印路径。

    Args:
        fig: matplotlib Figure 对象
        path: 输出文件路径
        dpi: 分辨率
        fmt: 格式（默认从文件扩展名推断）
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, format=fmt)
    print(f"  图表已保存: {path}")
