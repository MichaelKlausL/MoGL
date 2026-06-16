# MolFrag 数据处理脚本

本目录用于将包含分子信息的表格数据（`CID / SMILES / Description`）转换为用于训练/评估的片段增强数据。

核心处理包括：
- 使用 **RDKit BRICS** 对分子进行片段化（`fragments`）
- 使用 **ChemDataExtractor** 从描述文本中抽取化学实体关键词（`keywords`）
- 生成以 JSON 对象为单位的输出文件，支持断点续跑

## 目录说明

- `build_dataset.py`：主脚本，读取输入数据并生成输出文件
- `count_records.py`：统计输出文件中的样本条数
- `frag.py`：BRICS 片段化最小示例
- `chem.py`：ChemDataExtractor 抽取最小示例
- `frag_train.jsonl / frag_valid.jsonl / frag_test.jsonl`：示例输出

## 环境依赖

建议 Python 3.9+。

安装依赖（示例）：

```bash
pip install rdkit-pypi chemdataextractor tqdm
```

> 说明：不同环境下 RDKit 安装方式可能不同（例如 conda 安装更稳定）。

## 输入数据格式

`build_dataset.py` 目前通过脚本顶部常量配置输入/输出路径：

```python
INPUT_PATH = "/path/to/your/input.txt"
OUTPUT_JSONL_PATH = "frag_test.jsonl"
```

输入文件可为：
- `.txt` / `.tsv`（按制表符 `\t` 分隔）
- `.csv`（按逗号 `,` 分隔）

需要包含以下列名（大小写按脚本内逻辑匹配）：
- `CID` 或 `cid`
- `SMILES` 或 `smiles`
- `Description` 或 `description`

缺失上述任一字段的样本会被跳过。

## 运行方式

### 1) 生成数据

先修改 `build_dataset.py` 中的 `INPUT_PATH` 与 `OUTPUT_JSONL_PATH`，再执行：

```bash
python build_dataset.py
```

脚本会显示进度条并写入输出文件。

### 2) 统计样本数

修改 `count_records.py` 中的 `INPUT_PATH` 后执行：

```bash
python count_records.py
```

## 输出格式说明

输出文件中的每个样本结构如下：

```json
{
  "cid": "12345",
  "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
  "description": "...",
  "fragments": ["...", "..."],
  "keywords": ["...", "..."]
}
```

文件写出方式为：
- 每条样本使用带缩进的 JSON（`indent=2`）
- 样本之间额外写入一个空行

因此该文件更准确是“**空行分隔的 JSON 记录**”（而非严格一行一条的 JSONL）。

## 断点续跑机制

`build_dataset.py` 会先读取已存在输出文件中的 `cid`，再次运行时自动跳过已处理样本。

适合：
- 长时间任务中断后继续处理
- 分批追加构建数据

## 关键实现细节

- BRICS 分解设置了超时（默认 5 秒），超时或异常时该样本 `fragments` 置为空列表，避免整体卡死。
- `keywords` 去重并保持原顺序。

## 常见问题

1. **RDKit 安装失败**
	- 建议优先使用 conda 环境安装 RDKit。

2. **输出文件不是一行一个 JSON**
	- 这是当前脚本设计（可读性优先）。
	- 配套计数脚本 `count_records.py` 已按该格式解析。

3. **为什么有些样本 fragments 为空**
	- 可能是 SMILES 无法解析，或 BRICS 分解超时/异常。


