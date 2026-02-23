# AlloScale — 开发者指南

## 项目概述

AlloScale 是一个 **纯数据管理框架**，为 Attention MoE 分离式推理实验提供元数据管理和数据溯源。它不运行实验本身，只负责：

- 接收人工输入的元数据（目的、配置、代码版本等）
- 管理多样化的实验数据文件（json / nsys-rep / txt / sqlite / csv）
- 维护 **数据 → 脚本 → 图表** 的清晰溯源链

## 架构

```
alloscale.py          单文件 CLI，所有命令入口（无需 pip install）
config.yaml           全局配置（绘图参数、支持格式等）
catalog.yaml          自动生成的全局索引（由 _rebuild_catalog_data() 维护，勿手动编辑）
templates/            模板文件
  metadata.yaml       新实验的 metadata 模板（Python str.format 占位符）
  plot_template.py    可视化脚本骨架（Python str.format 占位符）
scripts/shared/       共享工具库
  loaders.py          统一数据加载：load_data(exp_dir, metadata, logical_name)
  plot_utils.py       绘图工具：setup_style() + save_figure()
experiments/          实验数据根目录
  YYYY-MM-DD_<slug>/  每个实验一个目录，自包含
    metadata.yaml     唯一真相源
    data/             原始数据文件
    scripts/          该实验的可视化脚本
    figures/          生成的图表
```

## 唯一外部依赖

- **PyYAML**（`import yaml`）— 必须安装：`pip install pyyaml`
- **matplotlib**（仅绘图脚本需要）— `pip install matplotlib`

## 核心设计决策

1. **metadata.yaml 是唯一真相源** — 所有数据注册、图表溯源都写在这里。catalog.yaml 是从各实验 metadata 自动汇总的派生物。
2. **扁平实验列表 + 标签过滤** — 不做层级分类，避免分类争议。
3. **实验 ID = `YYYY-MM-DD_<slug>`** — 日期前缀保证按时间排序。
4. **模糊匹配** — `_resolve_experiment(query)` 支持子串匹配，输入 `moe-routing` 即可匹配 `2026-02-23_moe-routing-latency`。
5. **每个实验自包含** — 可直接打包分享给他人。

## CLI 命令一览

| 命令 | 入口函数 | 作用 |
|---|---|---|
| `new <slug>` | `cmd_new` | 创建实验目录 + 骨架 metadata |
| `add-data <exp> <file>` | `cmd_add_data` | 拷贝数据文件并注册到 metadata.data |
| `add-script <exp> <fig-name>` | `cmd_add_script` | 从模板生成脚本并注册到 metadata.figures |
| `plot <exp> [fig-name]` | `cmd_plot` | 运行可视化脚本（subprocess 调用） |
| `list [--tag/--status]` | `cmd_list` | 列出并过滤实验 |
| `show <exp>` | `cmd_show` | 显示实验详情 |
| `info <exp>` | `cmd_info` | 显示 数据→脚本→图表 溯源树 |
| `rebuild-catalog` | `cmd_rebuild_catalog` | 重建 catalog.yaml |
| `validate [exp]` | `cmd_validate` | 检查文件完整性和引用一致性 |

## alloscale.py 内部结构

```
全局路径常量: ROOT, EXPERIMENTS_DIR, CATALOG_FILE, CONFIG_FILE, TEMPLATE_DIR

工具函数:
  _now_iso()              → ISO 时间戳字符串
  _today()                → YYYY-MM-DD 日期字符串
  _load_config()          → 读取 config.yaml
  _load_metadata(exp_dir) → 读取实验 metadata.yaml
  _save_metadata(exp_dir, metadata) → 写入 metadata.yaml（自动更新 updated 时间戳）
  _resolve_experiment(query) → 模糊匹配实验 ID，返回 (exp_id, exp_dir)
  _detect_format(file_path)  → 根据扩展名推断数据格式
  _rebuild_catalog_data()    → 扫描所有实验构建 catalog 字典
  _save_catalog(catalog)     → 写入 catalog.yaml

命令函数: cmd_new, cmd_add_data, cmd_add_script, cmd_plot, cmd_list, cmd_show, cmd_info, cmd_rebuild_catalog, cmd_validate

CLI 入口: main() → argparse 路由到命令函数
```

## metadata.yaml 关键字段

- `environment`: 实验环境，含 `machine`（机器名）、`gpu`（GPU 型号）、`notes`
- `provenance`: 代码溯源，含 `code_repo`、`code_branch`（框架分支）、`code_commit`（commit hash）、`notes`
- `data`: 字典，key 是逻辑名，value 含 `file`、`format`、`description`
- `figures`: 字典，key 是图表名，value 含 `file`、`script`、`data_inputs`（引用 data 中的逻辑名）、`description`
- `status`: `active` / `complete` / `abandoned`

## 脚本模板机制

`templates/plot_template.py` 使用 Python `str.format()` 占位符：`{description}`, `{exp_id}`, `{fig_name}`, `{data_inputs}`, `{load_lines}`, `{output_file}`。生成脚本时通过 `PROJECT_ROOT = Path(__file__).resolve().parents[3]` 定位项目根目录以导入 `scripts.shared`。

## 修改注意事项

- 添加新数据格式：在 `loaders.py` 的 `_LOADERS` 字典中添加加载函数，在 `alloscale.py` 的 `_detect_format` 中添加扩展名映射，在 `config.yaml` 的 `supported_formats` 中添加格式名。
- 添加新 CLI 命令：在 `main()` 中添加 argparse 子命令，实现 `cmd_xxx` 函数，添加到 `commands` 字典。
- 每个修改 metadata 的命令都应调用 `_rebuild_catalog_data()` + `_save_catalog()` 保持 catalog 同步。
