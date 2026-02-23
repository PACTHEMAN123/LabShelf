"""统一数据加载模块

支持格式: json, sqlite, nsys-rep, txt, csv
"""
import json
import sqlite3
from pathlib import Path


def load_data(exp_dir, metadata, logical_name):
    """根据 metadata 中的 format 字段自动加载数据。

    Args:
        exp_dir: 实验目录路径 (Path 或 str)
        metadata: 解析后的 metadata.yaml 字典
        logical_name: data 区块中注册的逻辑名

    Returns:
        加载后的数据对象（类型取决于格式）
    """
    exp_dir = Path(exp_dir)
    entry = metadata["data"].get(logical_name)
    if entry is None:
        raise KeyError(f"数据 '{logical_name}' 未在 metadata.data 中注册")

    file_path = exp_dir / entry["file"]
    fmt = entry.get("format", "").lower()

    if not file_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    loader = _LOADERS.get(fmt)
    if loader is None:
        raise ValueError(f"不支持的格式: {fmt}（支持: {', '.join(_LOADERS)}）")

    return loader(file_path)


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_csv(path):
    import csv as csv_mod
    with open(path, "r", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        return list(reader)


def _load_sqlite(path):
    """返回 sqlite3 连接对象，由调用者负责关闭。"""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _load_nsys_rep(path):
    """nsys-rep 文件返回路径字符串，需要用 nsys 工具链处理。"""
    if not path.exists():
        raise FileNotFoundError(f"nsys-rep 文件不存在: {path}")
    return str(path)


_LOADERS = {
    "json": _load_json,
    "txt": _load_txt,
    "csv": _load_csv,
    "sqlite": _load_sqlite,
    "nsys-rep": _load_nsys_rep,
}
