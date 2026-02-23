# LabShelf

Attention MoE 分离式推理实验的数据管理框架。

LabShelf 不运行实验本身，只负责管理实验元数据、数据文件和可视化脚本，维护清晰的 **数据 → 脚本 → 图表** 溯源链。

## 安装

唯一依赖：

```bash
pip install pyyaml matplotlib
```

无需安装 LabShelf 本身，直接运行 `python3 labshelf.py`。

## 快速开始

```bash
# 1. 创建新实验
python3 labshelf.py new moe-routing-latency \
  --purpose "测量 MoE 层在不同 expert 数量下的 routing 延迟" \
  --tags routing latency moe \
  --code-branch feature/moe-split \
  --code-commit abc1234 \
  --machine node-01 \
  --gpu A100-80G

# 2. 添加数据文件
python3 labshelf.py add-data moe-routing ~/results/latency_log.json \
  --name latency_log \
  --desc "Per-token routing latency"

# 3. 创建可视化脚本
python3 labshelf.py add-script moe-routing latency_cdf \
  --inputs latency_log \
  --desc "延迟 CDF 分布图"

# 4. 编辑生成的脚本，填入绘图逻辑
# vim experiments/2026-02-23_moe-routing-latency/scripts/plot_latency_cdf.py

# 5. 运行绘图
python3 labshelf.py plot moe-routing latency_cdf

# 6. 查看实验
python3 labshelf.py list --tag routing
python3 labshelf.py show moe-routing
python3 labshelf.py info moe-routing
```

## 命令参考

| 命令 | 作用 |
|---|---|
| `new <slug>` | 创建新实验目录 + 骨架 metadata |
| `add-data <exp> <file>` | 拷贝数据文件并注册到 metadata |
| `add-script <exp> <fig-name>` | 生成可视化脚本骨架并注册溯源关系 |
| `plot <exp> [fig-name]` | 运行可视化脚本生成图表 |
| `list [--tag TAG] [--status STATUS]` | 列出实验（支持过滤） |
| `show <exp>` | 显示实验详情 |
| `info <exp>` | 显示 数据→脚本→图表 溯源图 |
| `rebuild-catalog` | 重建全局索引 catalog.yaml |
| `validate [exp]` | 检查文件完整性 |

所有接受 `<exp>` 参数的命令都支持模糊匹配——输入 `moe-routing` 即可匹配 `2026-02-23_moe-routing-latency`。

## 目录结构

```
LabShelf/
├── labshelf.py                          # CLI 入口（单文件）
├── config.yaml                           # 全局配置
├── catalog.yaml                          # 自动生成的索引（勿手动编辑）
├── templates/
│   ├── metadata.yaml                     # 新实验元数据模板
│   └── plot_template.py                  # 可视化脚本骨架
├── scripts/
│   └── shared/
│       ├── loaders.py                    # 统一数据加载
│       └── plot_utils.py                 # 通用绘图工具
└── experiments/
    └── YYYY-MM-DD_<slug>/                # 每个实验一个目录
        ├── metadata.yaml                 # 元数据（唯一真相源）
        ├── data/                         # 原始数据
        ├── scripts/                      # 可视化脚本
        └── figures/                      # 生成的图表
```

## 元数据结构

每个实验的 `metadata.yaml` 包含：

```yaml
id: "2026-02-23_moe-routing-latency"
created: "2026-02-23T14:30:00"
updated: "2026-02-23T16:45:00"
purpose: "测量 MoE 层在不同 expert 数量下的 routing 延迟"
tags: [routing, latency, moe]
status: "active"          # active / complete / abandoned

environment:              # 实验环境
  machine: "node-01"
  gpu: "A100-80G"

config:                   # 实验配置（自由填写）
  num_experts: [8, 16, 32]
  num_gpus: 4

provenance:               # 代码溯源
  code_branch: "feature/moe-split"
  code_commit: "abc1234"

data:                     # 数据注册表（逻辑名 → 文件）
  latency_log:
    file: "data/latency_log.json"
    format: "json"
    description: "Per-token routing latency"

figures:                  # 数据→脚本→图表 溯源链
  latency_cdf:
    file: "figures/latency_cdf.png"
    script: "scripts/plot_latency_cdf.py"
    data_inputs: [latency_log]
    description: "延迟 CDF 分布图"
```

## 支持的数据格式

| 格式 | 扩展名 | 加载方式 |
|---|---|---|
| JSON | `.json` | `json.load()` → dict/list |
| CSV | `.csv` | `csv.DictReader()` → list[dict] |
| TXT | `.txt` | 读取为字符串 |
| SQLite | `.sqlite`, `.db` | 返回 `sqlite3.Connection` |
| Nsight | `.nsys-rep` | 返回文件路径（需 nsys 工具链处理） |

在可视化脚本中使用 `load_data(exp_dir, metadata, "逻辑名")` 自动加载。

## 设计原则

- **单文件 CLI**：无需安装，`python3 labshelf.py` 即用
- **YAML 优先**：支持注释和多行文本，适合手动编辑
- **扁平列表 + 标签**：不做层级分类，用标签过滤
- **每个实验自包含**：可直接打包分享
- **catalog 自动同步**：从各实验 metadata 汇总，永远不会与实际数据脱节
