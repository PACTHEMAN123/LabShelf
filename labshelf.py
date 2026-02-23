#!/usr/bin/env python3
"""LabShelf — 实验数据管理框架 CLI

用法: python labshelf.py <command> [options]

命令:
  new <slug>                         创建新实验目录 + 骨架 metadata
  add-data <exp> <file>              拷贝数据文件并注册到 metadata
  add-script <exp> <fig-name>        生成可视化脚本骨架并注册溯源关系
  plot <exp> [fig-name]              运行可视化脚本生成图表
  list [--tag TAG] [--status STATUS] 列出实验
  show <exp>                         显示实验详情
  info <exp>                         显示 数据→脚本→图表 溯源图
  rebuild-catalog                    重建全局索引 catalog.yaml
  validate [exp]                     检查文件完整性
"""
import argparse
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# ── 全局路径 ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
CATALOG_FILE = ROOT / "catalog.yaml"
CONFIG_FILE = ROOT / "config.yaml"
TEMPLATE_DIR = ROOT / "templates"


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _now_iso():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _today():
    return datetime.date.today().strftime("%Y-%m-%d")


def _load_config():
    if CONFIG_FILE.exists():
        return yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8")) or {}
    return {}


def _load_metadata(exp_dir):
    meta_path = Path(exp_dir) / "metadata.yaml"
    if not meta_path.exists():
        sys.exit(f"错误: 未找到 {meta_path}")
    return yaml.safe_load(meta_path.read_text(encoding="utf-8"))


def _save_metadata(exp_dir, metadata):
    meta_path = Path(exp_dir) / "metadata.yaml"
    metadata["updated"] = _now_iso()
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.dump(metadata, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _resolve_experiment(query):
    """模糊匹配实验 ID，返回 (exp_id, exp_dir)。"""
    if not EXPERIMENTS_DIR.exists():
        sys.exit("错误: experiments/ 目录不存在")

    candidates = sorted([
        d.name for d in EXPERIMENTS_DIR.iterdir()
        if d.is_dir() and (d / "metadata.yaml").exists()
    ])

    # 精确匹配
    if query in candidates:
        return query, EXPERIMENTS_DIR / query

    # 模糊匹配：query 是 candidate 的子串
    matches = [c for c in candidates if query in c]
    if len(matches) == 1:
        return matches[0], EXPERIMENTS_DIR / matches[0]
    elif len(matches) > 1:
        sys.exit(f"错误: '{query}' 匹配到多个实验:\n  " + "\n  ".join(matches))
    else:
        sys.exit(f"错误: 未找到匹配 '{query}' 的实验")


def _detect_format(file_path):
    """根据文件扩展名推断格式。"""
    suffix = Path(file_path).suffix.lower()
    mapping = {
        ".json": "json",
        ".txt": "txt",
        ".csv": "csv",
        ".sqlite": "sqlite",
        ".db": "sqlite",
        ".nsys-rep": "nsys-rep",
    }
    # 处理复合后缀如 .nsys-rep
    name = Path(file_path).name
    if name.endswith(".nsys-rep"):
        return "nsys-rep"
    return mapping.get(suffix, "txt")


def _rebuild_catalog_data():
    """扫描所有实验，构建 catalog 数据。"""
    catalog = {"generated": _now_iso(), "experiments": {}}

    if not EXPERIMENTS_DIR.exists():
        return catalog

    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        meta_path = exp_dir / "metadata.yaml"
        if not exp_dir.is_dir() or not meta_path.exists():
            continue
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        catalog["experiments"][meta["id"]] = {
            "purpose": meta.get("purpose", ""),
            "status": meta.get("status", ""),
            "tags": meta.get("tags", []),
            "created": meta.get("created", ""),
            "data_count": len(meta.get("data", {}) or {}),
            "figure_count": len(meta.get("figures", {}) or {}),
        }

    return catalog


def _save_catalog(catalog):
    with open(CATALOG_FILE, "w", encoding="utf-8") as f:
        f.write("# 自动生成 — 勿手动编辑\n")
        f.write(f"# 最后更新: {catalog['generated']}\n\n")
        yaml.dump(catalog, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ── 命令实现 ──────────────────────────────────────────────────────────────────

def cmd_new(args):
    """创建新实验目录 + 骨架 metadata。"""
    slug = args.slug
    exp_id = f"{_today()}_{slug}"
    exp_dir = EXPERIMENTS_DIR / exp_id

    if exp_dir.exists():
        sys.exit(f"错误: 实验目录已存在: {exp_dir}")

    # 创建目录结构
    (exp_dir / "data").mkdir(parents=True)
    (exp_dir / "scripts").mkdir(parents=True)
    (exp_dir / "figures").mkdir(parents=True)

    # 生成 metadata
    now = _now_iso()
    tags = args.tags if args.tags else []
    metadata = {
        "id": exp_id,
        "created": now,
        "updated": now,
        "purpose": args.purpose or "",
        "tags": tags,
        "status": "active",
        "environment": {
            "machine": args.machine or "",
            "gpu": args.gpu or "",
            "notes": "",
        },
        "config": {"custom": {}},
        "provenance": {
            "code_repo": "",
            "code_branch": args.code_branch or "",
            "code_commit": args.code_commit or "",
            "notes": "",
        },
        "data": {},
        "figures": {},
    }

    with open(exp_dir / "metadata.yaml", "w", encoding="utf-8") as f:
        yaml.dump(metadata, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # 更新 catalog
    catalog = _rebuild_catalog_data()
    _save_catalog(catalog)

    print(f"已创建实验: {exp_id}")
    print(f"  目录: {exp_dir}")
    print(f"  metadata: {exp_dir / 'metadata.yaml'}")


def cmd_add_data(args):
    """拷贝数据文件并注册到 metadata。"""
    exp_id, exp_dir = _resolve_experiment(args.exp)
    src = Path(args.file).expanduser().resolve()

    if not src.exists():
        sys.exit(f"错误: 源文件不存在: {src}")

    # 确定逻辑名和目标路径
    logical_name = args.name if args.name else src.stem
    dest = exp_dir / "data" / src.name

    # 拷贝文件
    shutil.copy2(str(src), str(dest))

    # 推断格式
    fmt = args.format if args.format else _detect_format(src.name)

    # 更新 metadata
    metadata = _load_metadata(exp_dir)
    if metadata.get("data") is None:
        metadata["data"] = {}
    metadata["data"][logical_name] = {
        "file": f"data/{src.name}",
        "format": fmt,
        "description": args.desc or "",
    }
    _save_metadata(exp_dir, metadata)

    # 更新 catalog
    catalog = _rebuild_catalog_data()
    _save_catalog(catalog)

    print(f"已添加数据: {logical_name}")
    print(f"  文件: {dest}")
    print(f"  格式: {fmt}")
    print(f"  实验: {exp_id}")


def cmd_add_script(args):
    """生成可视化脚本骨架并注册溯源关系。"""
    exp_id, exp_dir = _resolve_experiment(args.exp)
    fig_name = args.fig_name
    inputs = args.inputs if args.inputs else []
    description = args.desc or fig_name

    metadata = _load_metadata(exp_dir)

    # 验证 data_inputs 引用的逻辑名是否存在
    data_section = metadata.get("data") or {}
    for inp in inputs:
        if inp not in data_section:
            sys.exit(f"错误: 数据 '{inp}' 未注册，请先使用 add-data 注册")

    # 读取脚本模板
    template_path = TEMPLATE_DIR / "plot_template.py"
    template = template_path.read_text(encoding="utf-8")

    # 生成 load_data 行
    load_lines = []
    for inp in inputs:
        load_lines.append(f"    {inp} = load_data(exp_dir, metadata, \"{inp}\")")
    if not load_lines:
        load_lines.append("    pass  # 无数据输入")
    load_block = "\n".join(load_lines)

    # 确定输出文件名
    config = _load_config()
    plot_fmt = config.get("plot", {}).get("format", "png")
    output_file = f"{fig_name}.{plot_fmt}"

    # 填充模板
    script_content = template.format(
        description=description,
        exp_id=exp_id,
        fig_name=fig_name,
        data_inputs=", ".join(inputs) if inputs else "无",
        load_lines=load_block,
        output_file=output_file,
    )

    # 写入脚本
    script_path = exp_dir / "scripts" / f"plot_{fig_name}.py"
    script_path.write_text(script_content, encoding="utf-8")
    script_path.chmod(0o755)

    # 更新 metadata
    if metadata.get("figures") is None:
        metadata["figures"] = {}
    metadata["figures"][fig_name] = {
        "file": f"figures/{output_file}",
        "script": f"scripts/plot_{fig_name}.py",
        "data_inputs": inputs,
        "description": description,
    }
    _save_metadata(exp_dir, metadata)

    # 更新 catalog
    catalog = _rebuild_catalog_data()
    _save_catalog(catalog)

    print(f"已创建脚本: {script_path}")
    print(f"  图表名: {fig_name}")
    print(f"  数据输入: {', '.join(inputs) if inputs else '无'}")
    print(f"  输出: figures/{output_file}")
    print(f"  请编辑脚本添加绘图逻辑")


def cmd_plot(args):
    """运行可视化脚本生成图表。"""
    exp_id, exp_dir = _resolve_experiment(args.exp)
    metadata = _load_metadata(exp_dir)
    figures = metadata.get("figures") or {}

    if not figures:
        sys.exit(f"错误: 实验 '{exp_id}' 没有注册的图表")

    # 确定要生成哪些图表
    if args.fig_name:
        if args.fig_name not in figures:
            sys.exit(f"错误: 图表 '{args.fig_name}' 未注册")
        targets = {args.fig_name: figures[args.fig_name]}
    else:
        targets = figures

    for fig_name, fig_info in targets.items():
        script_path = exp_dir / fig_info["script"]
        if not script_path.exists():
            print(f"  跳过 {fig_name}: 脚本不存在 ({script_path})")
            continue

        print(f"运行: {fig_name} ...")
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(exp_dir),
            capture_output=True,
            text=True,
        )

        if result.stdout:
            print(result.stdout.rstrip())
        if result.returncode != 0:
            print(f"  错误 (exit {result.returncode}):")
            if result.stderr:
                print(f"  {result.stderr.rstrip()}")
        else:
            # 更新 metadata 时间戳
            metadata = _load_metadata(exp_dir)
            _save_metadata(exp_dir, metadata)
            print(f"  完成: {fig_name}")


def cmd_list(args):
    """列出实验（支持过滤）。"""
    if not EXPERIMENTS_DIR.exists():
        print("暂无实验")
        return

    experiments = []
    for exp_dir in sorted(EXPERIMENTS_DIR.iterdir()):
        meta_path = exp_dir / "metadata.yaml"
        if not exp_dir.is_dir() or not meta_path.exists():
            continue
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        experiments.append(meta)

    # 过滤
    if args.tag:
        experiments = [e for e in experiments if args.tag in (e.get("tags") or [])]
    if args.status:
        experiments = [e for e in experiments if e.get("status") == args.status]

    if not experiments:
        print("无匹配的实验")
        return

    # 显示
    for exp in experiments:
        tags_str = ", ".join(exp.get("tags") or [])
        data_count = len(exp.get("data") or {})
        fig_count = len(exp.get("figures") or {})
        status = exp.get("status", "?")
        print(f"  [{status:^9s}] {exp['id']}")
        print(f"             {exp.get('purpose', '')}")
        print(f"             tags: [{tags_str}]  data: {data_count}  figures: {fig_count}")
        print()


def cmd_show(args):
    """显示实验详情。"""
    exp_id, exp_dir = _resolve_experiment(args.exp)
    metadata = _load_metadata(exp_dir)

    print(f"实验: {metadata['id']}")
    print(f"  状态:   {metadata.get('status', '?')}")
    print(f"  目的:   {metadata.get('purpose', '')}")
    print(f"  标签:   {', '.join(metadata.get('tags') or [])}")
    print(f"  创建:   {metadata.get('created', '?')}")
    print(f"  更新:   {metadata.get('updated', '?')}")

    env = metadata.get("environment") or {}
    if env.get("machine"):
        print(f"  机器:   {env['machine']}")
    if env.get("gpu"):
        print(f"  GPU:    {env['gpu']}")
    if env.get("notes"):
        print(f"  环境备注: {env['notes']}")

    prov = metadata.get("provenance") or {}
    if prov.get("code_branch"):
        print(f"  分支:   {prov['code_branch']}")
    if prov.get("code_commit"):
        print(f"  Commit: {prov['code_commit']}")

    config = metadata.get("config") or {}
    config_items = {k: v for k, v in config.items() if k != "custom" or v}
    if config_items:
        print(f"  配置:")
        for k, v in config_items.items():
            print(f"    {k}: {v}")

    data = metadata.get("data") or {}
    if data:
        print(f"  数据 ({len(data)}):")
        for name, info in data.items():
            exists = "✓" if (exp_dir / info["file"]).exists() else "✗"
            print(f"    [{exists}] {name}: {info['file']} ({info.get('format', '?')})")

    figures = metadata.get("figures") or {}
    if figures:
        print(f"  图表 ({len(figures)}):")
        for name, info in figures.items():
            exists = "✓" if (exp_dir / info["file"]).exists() else "✗"
            print(f"    [{exists}] {name}: {info['file']}")


def cmd_info(args):
    """显示 数据→脚本→图表 溯源图。"""
    exp_id, exp_dir = _resolve_experiment(args.exp)
    metadata = _load_metadata(exp_dir)

    print(f"溯源图: {exp_id}")
    print()

    figures = metadata.get("figures") or {}
    data_section = metadata.get("data") or {}

    if not figures:
        print("  暂无注册的图表")
        return

    for fig_name, fig_info in figures.items():
        fig_exists = "✓" if (exp_dir / fig_info["file"]).exists() else "✗"
        script_exists = "✓" if (exp_dir / fig_info["script"]).exists() else "✗"

        print(f"  [{fig_exists}] {fig_info['file']}")
        print(f"    └── [{script_exists}] {fig_info['script']}")

        inputs = fig_info.get("data_inputs") or []
        for i, inp in enumerate(inputs):
            connector = "└" if i == len(inputs) - 1 else "├"
            data_info = data_section.get(inp, {})
            data_file = data_info.get("file", f"(未注册: {inp})")
            data_exists = "✓" if data_info and (exp_dir / data_file).exists() else "✗"
            print(f"        {connector}── [{data_exists}] {data_file}  ({inp})")

        if fig_info.get("description"):
            print(f"        描述: {fig_info['description']}")
        print()


def cmd_rebuild_catalog(args):
    """重建全局索引。"""
    catalog = _rebuild_catalog_data()
    _save_catalog(catalog)
    count = len(catalog.get("experiments", {}))
    print(f"已重建 catalog.yaml ({count} 个实验)")


def cmd_validate(args):
    """检查文件完整性。"""
    if args.exp:
        exp_id, exp_dir = _resolve_experiment(args.exp)
        experiments = [(exp_id, exp_dir)]
    else:
        if not EXPERIMENTS_DIR.exists():
            print("暂无实验")
            return
        experiments = []
        for d in sorted(EXPERIMENTS_DIR.iterdir()):
            if d.is_dir() and (d / "metadata.yaml").exists():
                experiments.append((d.name, d))

    total_issues = 0

    for exp_id, exp_dir in experiments:
        metadata = _load_metadata(exp_dir)
        issues = []

        # 检查 data 文件
        for name, info in (metadata.get("data") or {}).items():
            file_path = exp_dir / info["file"]
            if not file_path.exists():
                issues.append(f"数据文件缺失: {info['file']} ({name})")

        # 检查 figures 文件和脚本
        for name, info in (metadata.get("figures") or {}).items():
            script_path = exp_dir / info["script"]
            if not script_path.exists():
                issues.append(f"脚本缺失: {info['script']} ({name})")

            # 检查 data_inputs 引用
            for inp in (info.get("data_inputs") or []):
                if inp not in (metadata.get("data") or {}):
                    issues.append(f"图表 '{name}' 引用了未注册的数据: {inp}")

        # 检查必填字段
        if not metadata.get("id"):
            issues.append("缺少 id 字段")
        if not metadata.get("purpose"):
            issues.append("缺少 purpose 字段")

        if issues:
            print(f"  {exp_id}: {len(issues)} 个问题")
            for issue in issues:
                print(f"    - {issue}")
            total_issues += len(issues)
        else:
            print(f"  {exp_id}: 通过 ✓")

    if total_issues == 0:
        print(f"\n全部通过 ({len(experiments)} 个实验)")
    else:
        print(f"\n发现 {total_issues} 个问题")


# ── CLI 入口 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="labshelf",
        description="LabShelf — 实验数据管理框架",
    )
    sub = parser.add_subparsers(dest="command", help="可用命令")

    # new
    p_new = sub.add_parser("new", help="创建新实验")
    p_new.add_argument("slug", help="实验名称 slug")
    p_new.add_argument("--purpose", help="实验目的")
    p_new.add_argument("--tags", nargs="*", help="标签列表")
    p_new.add_argument("--code-branch", help="推理框架分支")
    p_new.add_argument("--code-commit", help="推理框架 commit hash")
    p_new.add_argument("--machine", help="实验机器名称")
    p_new.add_argument("--gpu", help="GPU 型号")

    # add-data
    p_add_data = sub.add_parser("add-data", help="添加数据文件")
    p_add_data.add_argument("exp", help="实验 ID（支持模糊匹配）")
    p_add_data.add_argument("file", help="数据文件路径")
    p_add_data.add_argument("--name", help="逻辑名（默认取文件名）")
    p_add_data.add_argument("--format", help="数据格式（默认自动检测）")
    p_add_data.add_argument("--desc", help="数据描述")

    # add-script
    p_add_script = sub.add_parser("add-script", help="创建可视化脚本")
    p_add_script.add_argument("exp", help="实验 ID（支持模糊匹配）")
    p_add_script.add_argument("fig_name", help="图表名称")
    p_add_script.add_argument("--inputs", nargs="*", help="数据输入（逻辑名列表）")
    p_add_script.add_argument("--desc", help="图表描述")

    # plot
    p_plot = sub.add_parser("plot", help="运行可视化脚本")
    p_plot.add_argument("exp", help="实验 ID")
    p_plot.add_argument("fig_name", nargs="?", help="图表名称（省略则运行所有）")

    # list
    p_list = sub.add_parser("list", help="列出实验")
    p_list.add_argument("--tag", help="按标签过滤")
    p_list.add_argument("--status", help="按状态过滤")

    # show
    p_show = sub.add_parser("show", help="显示实验详情")
    p_show.add_argument("exp", help="实验 ID")

    # info
    p_info = sub.add_parser("info", help="显示溯源图")
    p_info.add_argument("exp", help="实验 ID")

    # rebuild-catalog
    sub.add_parser("rebuild-catalog", help="重建全局索引")

    # validate
    p_validate = sub.add_parser("validate", help="检查完整性")
    p_validate.add_argument("exp", nargs="?", help="实验 ID（省略则检查所有）")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "new": cmd_new,
        "add-data": cmd_add_data,
        "add-script": cmd_add_script,
        "plot": cmd_plot,
        "list": cmd_list,
        "show": cmd_show,
        "info": cmd_info,
        "rebuild-catalog": cmd_rebuild_catalog,
        "validate": cmd_validate,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
