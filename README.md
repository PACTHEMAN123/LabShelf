# LabShelf

轻量级实验数据管理框架。

LabShelf 不运行实验本身，只负责管理实验元数据、数据文件和处理脚本，维护清晰的 **数据 → 脚本 → 输出** 溯源链。适用于任何需要追踪实验数据来源与处理过程的研究场景。

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
  --tags <tag1> <tag2>

# 2. 添加数据文件（环境/溯源 per-entry 记录）
./lab add-data <exp> <file> \
  --name <logical-name> \
  --desc "<数据描述>" \
  --env "<环境描述>" \
  --branch <代码分支> \
  --commit <代码commit>

# 3. 创建脚本（不绑定实验，在 scripts/ 下）
./lab add-script <name> \
  --desc "<脚本描述>"

# 4. 编辑生成的脚本，填入处理/绘图逻辑

# 5. 运行脚本处理实验数据
./lab run <script> <exp> \
  --inputs <name1> <name2> \
  --name <output-name> \
  --desc "<输出描述>"

# 6. 查看实验
./lab list [--tag <tag>] [--status <status>]
./lab show <exp>
./lab info <exp>
```

## 命令参考

| 命令 | 作用 |
|---|---|
| `new <slug>` | 创建新实验目录 + 骨架 metadata |
| `add-data <exp> <file>` | 拷贝数据文件并注册到 metadata（per-entry 环境/溯源） |
| `add-script <name>` | 在 `scripts/` 下创建脚本（不绑定实验） |
| `run <script> <exp>` | 运行脚本，输入从实验 data/ 读取，输出到 output/ |
| `list [--tag TAG] [--status STATUS]` | 列出实验（支持过滤） |
| `show <exp>` | 显示实验详情 |
| `info <exp>` | 显示 输入→脚本→输出 溯源图 |
| `rebuild-catalog` | 重建全局索引 catalog.yaml |
| `validate [exp]` | 检查文件完整性 |

所有接受 `<exp>` 参数的命令都支持模糊匹配——输入子串即可匹配完整实验 ID。

## 目录结构

```
LabShelf/
├── lab → labshelf.py                        # CLI 入口（符号链接）
├── config.yaml                               # 全局配置
├── catalog.yaml                              # 自动生成的索引（勿手动编辑）
├── templates/
│   ├── metadata.yaml                         # 新实验元数据模板
│   └── script_template.py                    # 统一脚本骨架
├── scripts/
│   ├── shared/
│   │   ├── loaders.py                        # 统一数据加载
│   │   └── plot_utils.py                     # 通用绘图工具
│   └── <用户脚本>.py                          # 脚本在此创建
└── experiments/
    └── <slug>/                               # 每个实验一个目录（无日期前缀）
        ├── metadata.yaml                     # 元数据（唯一真相源）
        ├── data/                             # 原始数据
        └── output/                           # 脚本输出（替代 figures/ + others/）
```

## 元数据结构

每个实验的 `metadata.yaml` 包含：

```yaml
id: "<slug>"
created: "YYYY-MM-DDTHH:MM:SS"
updated: "YYYY-MM-DDTHH:MM:SS"
purpose: "<实验目的>"
tags: [<tag1>, <tag2>]
status: "active"          # active / complete / abandoned

data:                     # 数据注册表（逻辑名 → 文件 + 环境/溯源）
  <logical-name>:
    file: "data/<filename>"
    description: "<描述>"
    added: "YYYY-MM-DDTHH:MM:SS"
    environment: "<环境描述>"         # 自由文本，如 "4×A100-80G, CUDA 12.1"
    provenance:                       # 产生该数据的代码溯源
      code_branch: "<branch>"
      code_commit: "<commit>"

outputs:                  # 输出注册表
  <output-name>:
    files: ["output/<output-name>/file1", ...]
    script: "<script>.py"
    inputs: [<逻辑名1>, ...]
    description: "<描述>"
    created: "YYYY-MM-DDTHH:MM:SS"
```

## 脚本运行机制

`./lab run` 通过 `sys.argv[1]` 传 JSON 给脚本：

```json
{"exp_dir": "...", "output_dir": "...", "inputs": {"name": "path", ...}}
```

- `--inputs` 不指定时默认使用实验的全部数据
- `--name` 不指定时默认用脚本名

在脚本中使用 `load_data(exp_dir, metadata, "逻辑名")` 加载数据，格式由文件扩展名自动推断。

## 设计原则

- **单文件 CLI**：无需安装，`./lab` 即用
- **YAML 优先**：支持注释和多行文本，适合手动编辑
- **扁平列表 + 标签**：不做层级分类，用标签过滤
- **实验 = 研究方向**：无日期前缀，每个实验代表一个研究方向而非单次运行
- **环境/溯源 per-entry**：每条数据独立记录实验环境和产生该数据的代码版本
- **脚本与实验解绑**：脚本统一在 `scripts/` 下，可复用于多个实验
- **统一输出**：`output/` 不限制输出格式
- **catalog 自动同步**：从各实验 metadata 汇总，永远不会与实际数据脱节
