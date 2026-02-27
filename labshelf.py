#!/usr/bin/env python3
"""LabShelf — 实验数据管理框架 CLI

用法: python labshelf.py <command> [options]

命令:
  new <slug>                         创建新实验目录 + 骨架 metadata
  add-data <exp> <file>              拷贝数据文件/文件夹并注册到 metadata
  add-script <name>                  在 scripts/ 下创建脚本
  run <script> <exp>                 运行脚本处理实验数据
  list [--tag TAG] [--status STATUS] 列出实验
  show <exp>                         显示实验详情
  info <exp>                         显示溯源图
  rebuild-catalog                    重建全局索引 catalog.yaml
  validate [exp]                     检查文件完整性
"""
import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# ── 全局路径 ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = ROOT / "experiments"
SCRIPTS_DIR = ROOT / "scripts"
CATALOG_FILE = ROOT / "catalog.yaml"
CONFIG_FILE = ROOT / "config.yaml"
TEMPLATE_DIR = ROOT / "templates"


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _now_iso():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


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
            "output_count": len(meta.get("outputs", {}) or {}),
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
    exp_dir = EXPERIMENTS_DIR / slug

    if exp_dir.exists():
        sys.exit(f"错误: 实验目录已存在: {exp_dir}")

    # 创建目录结构
    (exp_dir / "data").mkdir(parents=True)
    (exp_dir / "output").mkdir(parents=True)

    # 生成 metadata
    now = _now_iso()
    tags = args.tags if args.tags else []
    metadata = {
        "id": slug,
        "created": now,
        "updated": now,
        "purpose": args.purpose or "",
        "tags": tags,
        "status": "active",
        "data": {},
        "outputs": {},
    }

    with open(exp_dir / "metadata.yaml", "w", encoding="utf-8") as f:
        yaml.dump(metadata, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    # 更新 catalog
    catalog = _rebuild_catalog_data()
    _save_catalog(catalog)

    print(f"已创建实验: {slug}")
    print(f"  目录: {exp_dir}")
    print(f"  metadata: {exp_dir / 'metadata.yaml'}")


def cmd_add_data(args):
    """拷贝数据文件或文件夹并注册到 metadata（per-entry 环境和溯源）。"""
    exp_id, exp_dir = _resolve_experiment(args.exp)
    src = Path(args.file).expanduser().resolve()

    if not src.exists():
        sys.exit(f"错误: 源路径不存在: {src}")

    # 确定逻辑名和目标路径
    logical_name = args.name if args.name else src.stem
    dest = exp_dir / "data" / src.name

    # 拷贝文件或文件夹
    if src.is_dir():
        if dest.exists():
            shutil.rmtree(str(dest))
        shutil.copytree(str(src), str(dest))
    else:
        shutil.copy2(str(src), str(dest))

    # 更新 metadata
    metadata = _load_metadata(exp_dir)
    if metadata.get("data") is None:
        metadata["data"] = {}

    entry = {
        "file": f"data/{src.name}",
        "description": args.desc or "",
        "added": _now_iso(),
    }

    # 环境描述（自由文本）
    if args.env:
        entry["environment"] = args.env

    # 推理框架溯源（有值才写入）
    prov = {}
    if args.branch:
        prov["code_branch"] = args.branch
    if args.commit:
        prov["code_commit"] = args.commit
    if prov:
        entry["provenance"] = prov

    metadata["data"][logical_name] = entry
    _save_metadata(exp_dir, metadata)

    # 更新 catalog
    catalog = _rebuild_catalog_data()
    _save_catalog(catalog)

    print(f"已添加数据: {logical_name}")
    print(f"  路径: {dest}")
    print(f"  实验: {exp_id}")
    if args.env:
        print(f"  环境: {args.env}")
    if args.branch:
        print(f"  分支: {args.branch}")
    if args.commit:
        print(f"  Commit: {args.commit[:12]}")


def cmd_add_script(args):
    """在 scripts/ 下创建脚本（不绑定实验）。"""
    script_name = args.name
    description = args.desc or script_name

    # 读取脚本模板
    template_path = TEMPLATE_DIR / "script_template.py"
    template = template_path.read_text(encoding="utf-8")

    # 填充模板
    script_content = template.format(
        description=description,
    )

    # 写入脚本
    script_path = SCRIPTS_DIR / f"{script_name}.py"
    if script_path.exists():
        sys.exit(f"错误: 脚本已存在: {script_path}")

    script_path.write_text(script_content, encoding="utf-8")
    script_path.chmod(0o755)

    print(f"已创建脚本: {script_path}")
    print(f"  请编辑脚本添加处理逻辑")


def cmd_run(args):
    """运行脚本处理实验数据，输出到 output/，自动记录溯源。"""
    script_name = args.script
    exp_id, exp_dir = _resolve_experiment(args.exp)
    metadata = _load_metadata(exp_dir)

    # 定位脚本
    script_path = SCRIPTS_DIR / f"{script_name}.py"
    if not script_path.exists():
        # 尝试加 .py
        script_path = SCRIPTS_DIR / script_name
        if not script_path.exists():
            sys.exit(f"错误: 脚本不存在: {SCRIPTS_DIR / f'{script_name}.py'}")

    # 确定输出名
    output_name = args.name if args.name else script_path.stem

    # 确定输入数据
    data = metadata.get("data") or {}
    if not data:
        sys.exit(f"错误: 实验 '{exp_id}' 没有注册的数据")

    if args.inputs:
        # 指定输入
        inputs = {}
        for name in args.inputs:
            if name not in data:
                sys.exit(f"错误: 数据 '{name}' 未在实验 '{exp_id}' 中注册")
            inputs[name] = str(exp_dir / data[name]["file"])
    else:
        # 默认使用全部数据
        inputs = {name: str(exp_dir / info["file"]) for name, info in data.items()}

    # 创建输出目录
    output_dir = exp_dir / "output" / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 构建 JSON 参数
    run_args = json.dumps({
        "exp_dir": str(exp_dir),
        "output_dir": str(output_dir),
        "inputs": inputs,
    }, ensure_ascii=False)

    print(f"运行: {script_path.name} → {exp_id}/output/{output_name}")

    result = subprocess.run(
        [sys.executable, str(script_path), run_args],
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode != 0:
        print(f"  错误 (exit {result.returncode}):")
        if result.stderr:
            print(f"  {result.stderr.rstrip()}")
        return

    # 记录输出文件
    output_files = []
    if output_dir.exists():
        for f in sorted(output_dir.iterdir()):
            if f.is_file():
                output_files.append(f"output/{output_name}/{f.name}")

    # 更新 metadata
    metadata = _load_metadata(exp_dir)
    if metadata.get("outputs") is None:
        metadata["outputs"] = {}
    metadata["outputs"][output_name] = {
        "files": output_files,
        "script": script_path.name,
        "inputs": list(inputs.keys()),
        "description": args.desc or "",
        "created": _now_iso(),
    }
    _save_metadata(exp_dir, metadata)

    # 更新 catalog
    catalog = _rebuild_catalog_data()
    _save_catalog(catalog)

    print(f"  完成: {output_name}")
    if output_files:
        for f in output_files:
            print(f"    {f}")


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
        output_count = len(exp.get("outputs") or {})
        status = exp.get("status", "?")
        print(f"  [{status:^9s}] {exp['id']}")
        print(f"             {exp.get('purpose', '')}")
        print(f"             tags: [{tags_str}]  data: {data_count}  outputs: {output_count}")
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

    data = metadata.get("data") or {}
    if data:
        print(f"  数据 ({len(data)}):")
        for name, info in data.items():
            exists = "✓" if (exp_dir / info["file"]).exists() else "✗"
            print(f"    [{exists}] {name}: {info['file']}")
            if info.get("description"):
                print(f"        描述: {info['description']}")
            if info.get("environment"):
                print(f"        环境: {info['environment']}")
            prov = info.get("provenance") or {}
            if prov:
                parts = []
                if prov.get("code_branch"):
                    parts.append(f"分支={prov['code_branch']}")
                if prov.get("code_commit"):
                    parts.append(f"commit={prov['code_commit'][:12]}")
                print(f"        溯源: {', '.join(parts)}")

    outputs = metadata.get("outputs") or {}
    if outputs:
        print(f"  输出 ({len(outputs)}):")
        for name, info in outputs.items():
            files = info.get("files") or []
            print(f"    {name}:")
            print(f"        脚本: {info.get('script', '?')}")
            print(f"        输入: {', '.join(info.get('inputs') or [])}")
            if info.get("description"):
                print(f"        描述: {info['description']}")
            for f in files:
                exists = "✓" if (exp_dir / f).exists() else "✗"
                print(f"        [{exists}] {f}")


def cmd_info(args):
    """显示溯源图: inputs → script → output files。"""
    exp_id, exp_dir = _resolve_experiment(args.exp)
    metadata = _load_metadata(exp_dir)

    print(f"溯源图: {exp_id}")
    print()

    outputs = metadata.get("outputs") or {}

    if not outputs:
        print("  暂无注册的输出")
        return

    for output_name, output_info in outputs.items():
        script = output_info.get("script", "?")
        input_names = output_info.get("inputs") or []
        files = output_info.get("files") or []

        # 输入数据
        for inp in input_names:
            data_entry = (metadata.get("data") or {}).get(inp)
            if data_entry:
                exists = "✓" if (exp_dir / data_entry["file"]).exists() else "✗"
                print(f"  [{exists}] {data_entry['file']}  ({inp})")
            else:
                print(f"  [?] {inp}")

        # 脚本
        script_path = SCRIPTS_DIR / script
        script_exists = "✓" if script_path.exists() else "✗"
        print(f"    └── [{script_exists}] scripts/{script}")

        # 输出文件
        for f in files:
            f_exists = "✓" if (exp_dir / f).exists() else "✗"
            print(f"          └── [{f_exists}] {f}")

        if output_info.get("description"):
            print(f"        描述: {output_info['description']}")
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

        # 检查 outputs
        for name, info in (metadata.get("outputs") or {}).items():
            # 检查脚本
            script = info.get("script", "")
            if script:
                script_path = SCRIPTS_DIR / script
                if not script_path.exists():
                    issues.append(f"脚本缺失: scripts/{script} ({name})")
            # 检查输出文件
            for f in (info.get("files") or []):
                if not (exp_dir / f).exists():
                    issues.append(f"输出文件缺失: {f} ({name})")

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

    # add-data
    p_add_data = sub.add_parser("add-data", help="添加数据文件或文件夹")
    p_add_data.add_argument("exp", help="实验 ID（支持模糊匹配）")
    p_add_data.add_argument("file", help="数据文件或文件夹路径")
    p_add_data.add_argument("--name", help="逻辑名（默认取文件名）")
    p_add_data.add_argument("--desc", help="数据描述")
    p_add_data.add_argument("--env", help="实验环境描述（自由文本）")
    p_add_data.add_argument("--branch", help="推理框架代码分支")
    p_add_data.add_argument("--commit", help="推理框架代码 commit")

    # add-script
    p_add_script = sub.add_parser("add-script", help="创建脚本")
    p_add_script.add_argument("name", help="脚本名称")
    p_add_script.add_argument("--desc", help="脚本描述")

    # run
    p_run = sub.add_parser("run", help="运行脚本处理实验数据")
    p_run.add_argument("script", help="脚本名称")
    p_run.add_argument("exp", help="实验 ID（支持模糊匹配）")
    p_run.add_argument("--inputs", nargs="*", help="输入数据逻辑名（默认全部）")
    p_run.add_argument("--name", help="输出名称（默认用脚本名）")
    p_run.add_argument("--desc", help="输出描述")

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
        "run": cmd_run,
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
