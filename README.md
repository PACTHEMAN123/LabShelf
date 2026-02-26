# LabShelf

Attention MoE 分离式推理实验的数据管理框架。

LabShelf 不运行实验本身，只负责管理实验元数据、数据文件和可视化脚本，维护清晰的 **数据 → 脚本 → 图表** 溯源链。

## 安装

唯一依赖：

```bash
pip install pyyaml matplotlib
```

无需安装 LabShelf 本身，项目根目录下的 `lab` 是指向 `labshelf.py` 的符号链接，**所有命令请在项目根目录下运行**：

```bash
./lab <command> [options]
```

## 快速开始

```bash
# 1. 创建新实验
./lab new <slug> \
  --purpose "<实验目的>" \
  --tags <tag1> <tag2> \
  --code-branch <branch> \
  --code-commit <commit> \
  --machine <machine> \
  --gpu <gpu>

# 2. 添加数据文件
./lab add-data <exp> <file> \
  --name <logical-name> \
  --desc "<数据描述>"

# 3. 创建可视化脚本
./lab add-script <exp> <fig-name> \
  --desc "<图表描述>"

# 4. 编辑生成的脚本，填入绘图逻辑
#    脚本中通过 load_data(exp_dir, metadata, "逻辑名") 加载所需数据

# 5. 运行绘图
./lab plot <exp> [<fig-name>]

# 6. 创建处理脚本
./lab add-other <exp> <other-name> --ext <ext> \
  --desc "<输出描述>"

# 7. 运行处理脚本
./lab run-other <exp> [<other-name>]

# 8. 查看实验
./lab list [--tag <tag>] [--status <status>]
./lab show <exp>
./lab info <exp>
```

## 命令参考

| 命令 | 作用 |
|---|---|
| `new <slug>` | 创建新实验目录 + 骨架 metadata |
| `add-data <exp> <file>` | 拷贝数据文件并注册到 metadata |
| `add-script <exp> <fig-name>` | 生成可视化脚本骨架并注册溯源关系 |
| `plot <exp> [fig-name]` | 运行可视化脚本生成图表 |
| `add-other <exp> <other-name>` | 生成处理脚本骨架并注册溯源关系 |
| `run-other <exp> [other-name]` | 运行处理脚本生成输出 |
| `list [--tag TAG] [--status STATUS]` | 列出实验（支持过滤） |
| `show <exp>` | 显示实验详情 |
| `info <exp>` | 显示 脚本→图表 溯源图 |
| `rebuild-catalog` | 重建全局索引 catalog.yaml |
| `validate [exp]` | 检查文件完整性 |

所有接受 `<exp>` 参数的命令都支持模糊匹配——输入子串即可匹配完整实验 ID。

## 目录结构

```
LabShelf/
├── labshelf.py                          # CLI 入口（单文件）
├── config.yaml                           # 全局配置
├── catalog.yaml                          # 自动生成的索引（勿手动编辑）
├── templates/
│   ├── metadata.yaml                     # 新实验元数据模板
│   ├── plot_template.py                  # 可视化脚本骨架
│   └── other_template.py                 # 处理脚本骨架
├── scripts/
│   └── shared/
│       ├── loaders.py                    # 统一数据加载
│       └── plot_utils.py                 # 通用绘图工具
└── experiments/
    └── YYYY-MM-DD_<slug>/                # 每个实验一个目录
        ├── metadata.yaml                 # 元数据（唯一真相源）
        ├── data/                         # 原始数据
        ├── scripts/                      # 可视化/处理脚本
        ├── figures/                      # 生成的图表
        └── others/                       # 其他输出
```

## 元数据结构

每个实验的 `metadata.yaml` 包含：

```yaml
id: "YYYY-MM-DD_<slug>"
created: "YYYY-MM-DDTHH:MM:SS"
updated: "YYYY-MM-DDTHH:MM:SS"
purpose: "<实验目的>"
tags: [<tag1>, <tag2>]
status: "active"          # active / complete / abandoned

environment:              # 实验环境
  machine: "<machine>"
  gpu: "<gpu>"

config:                   # 实验配置（自由填写）
  custom: {}

provenance:               # 代码溯源
  code_branch: "<branch>"
  code_commit: "<commit>"

data:                     # 数据注册表（逻辑名 → 文件）
  <logical-name>:
    file: "data/<filename>"
    format: "<format>"
    description: "<描述>"

figures:                  # 脚本→图表 溯源链
  <fig-name>:
    file: "figures/<fig-name>.<fmt>"
    script: "scripts/plot_<fig-name>.py"
    description: "<描述>"

others:                   # 脚本→其他输出 溯源链
  <other-name>:
    file: "others/<other-name>.<ext>"
    script: "scripts/gen_<other-name>.py"
    description: "<描述>"
```

## 支持的数据格式

| 格式 | 扩展名 | 加载方式 |
|---|---|---|
| JSON | `.json` | `json.load()` → dict/list |
| CSV | `.csv` | `csv.DictReader()` → list[dict] |
| TXT | `.txt` | 读取为字符串 |
| SQLite | `.sqlite`, `.db` | 返回 `sqlite3.Connection` |
| Nsight | `.nsys-rep` | 返回文件路径（需 nsys 工具链处理） |

在脚本中使用 `load_data(exp_dir, metadata, "逻辑名")` 自动加载。

## 设计原则

- **单文件 CLI**：无需安装，`./lab` 即用（需在项目根目录运行）
- **YAML 优先**：支持注释和多行文本，适合手动编辑
- **扁平列表 + 标签**：不做层级分类，用标签过滤
- **每个实验自包含**：可直接打包分享
- **脚本绑定实验而非数据**：脚本创建时不绑定特定数据输入，运行时通过 `load_data()` 动态加载所需数据
- **catalog 自动同步**：从各实验 metadata 汇总，永远不会与实际数据脱节
