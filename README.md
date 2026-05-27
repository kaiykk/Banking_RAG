# 银行业 RAG 问答实验项目

面向银行业务问答场景的 RAG 与模型微调实验项目。仓库目前聚焦三件事：

- 用配置文件统一管理数据、模型、LoRA、DPO、RAG 和日志参数
- 提供 LoRA SFT 与 DPO 偏好优化训练入口
- 为后续银行领域数据处理、向量检索、重排序和推理服务预留 CLI 流程

> 当前代码仍处于实验/搭建阶段。`train-lora` 和 `train-dpo` 已有实现；`setup-rag` 与 `inference` 是预留命令；`process-data` 入口已声明，但依赖的 `src.data.processor` 模块尚未提交到仓库。

## 功能概览

### 已实现

- `ConfigManager`: 支持 YAML / JSON 配置加载、必填项校验、默认值补齐和点号路径读取。
- `Logger`: 统一控制台与文件日志输出。
- `LoRATrainer`: 基于 `transformers`、`datasets`、`peft` 的因果语言模型 LoRA 微调流程。
- `DPOOptimizer`: 基于 `trl` 的 DPO 偏好优化流程，兼容不同版本 `DPOTrainer` 的 `tokenizer` / `processing_class` 参数。
- 命令行入口：
  - `train-lora`
  - `train-dpo`
  - `process-data`（入口存在，数据处理模块待补齐）
  - `setup-rag`（预留）
  - `inference`（预留）

### 规划中

- 银行业数据抽取、清洗、过滤与 SFT / DPO 数据构造
- BGE-M3 embedding 与 reranker 微调
- FAISS 向量库构建
- 自适应检索策略
- 银行业问答推理链路
- 评估指标与回归测试

## 项目结构

```text
Banking_RAG/
├── config.yaml              # 默认运行配置
├── config.run312.yaml       # 备用/实验运行配置
├── requirements.txt         # Python 依赖
├── src/
│   ├── config_manager.py    # 配置加载、校验与默认值
│   ├── logger.py            # 日志封装
│   ├── cli/
│   │   └── main.py          # 命令行入口
│   └── training/
│       ├── lora_trainer.py  # LoRA SFT 训练
│       └── dpo_optimizer.py # DPO 偏好优化
└── README.md
```

运行过程中会按配置使用或生成以下本地目录，这些目录不包含在仓库中：

```text
data/
  processed/
  vector_db/
models/
  Baichuan2-7B-Base/
  bge-m3/
  lora_adapter/
  dpo_model/
logs/
```

## 环境要求

- 推荐 Python 3.10+
- PyTorch 2.x
- `transformers`
- `datasets`
- `peft`
- `trl`
- `accelerate`
- `faiss-cpu`
- `sentence-transformers`

安装依赖：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果需要在本地训练大模型，请准备可用的 CUDA 环境，并将基础模型文件放到 `config.yaml` 中配置的路径下。

## 配置说明

默认配置文件是 [config.yaml](config.yaml)。主要配置项如下：

| 配置段 | 作用 |
| --- | --- |
| `data` | 数据集 ID、银行数据过滤开关、SFT 与 DPO 数据输出路径 |
| `models` | 基础模型、嵌入模型、重排序模型、LoRA 适配器与 DPO 模型输出路径 |
| `lora` | LoRA rank、alpha、dropout、学习率、批次大小和训练步数 |
| `dpo` | DPO beta、学习率、批次大小和训练步数 |
| `rag` | 分块大小、检索 top-k、重排序 top-n 和向量库路径 |
| `inference` | 生成温度、最大 token 数、top-p 和重复惩罚 |
| `logging` | 日志级别、控制台/文件输出开关和日志文件路径 |
| `evaluation` | 测试集路径和基线分数占位配置 |

训练前至少需要确认这些路径：

```yaml
models:
  base_model_path: "./models/Baichuan2-7B-Base"
  lora_adapter_path: "./models/lora_adapter"
  dpo_model_path: "./models/dpo_model"

data:
  lora_output_path: "./data/processed/lora_data.json"
  dpo_output_path: "./data/processed/dpo_data.json"
```

## 命令行用法

查看可用命令：

```bash
python -m src.cli.main --help
```

运行 LoRA 微调：

```bash
python -m src.cli.main train-lora \
  --config config.yaml \
  --data-path ./data/processed/lora_data.json
```

SFT 数据格式示例：

```json
[
  {
    "instruction": "请解释企业流动资金贷款的适用场景。",
    "input": "",
    "output": "企业流动资金贷款主要用于..."
  }
]
```

运行 DPO 偏好优化：

```bash
python -m src.cli.main train-dpo \
  --config config.yaml \
  --data-path ./data/processed/dpo_data.json
```

DPO 数据格式示例：

```json
[
  {
    "prompt": "客户想了解企业贷款准入条件，应如何回答？",
    "chosen": "可以从企业资质、经营流水、征信和担保方式等方面说明...",
    "rejected": "企业贷款就是给企业的钱，满足条件就能申请。"
  }
]
```

以下命令已声明，但尚未完整实现：

```bash
python -m src.cli.main process-data --config config.yaml
python -m src.cli.main setup-rag
python -m src.cli.main inference
```

`setup-rag` 和 `inference` 当前会抛出 `NotImplementedError`。`process-data` 需要先补充 `src/data/processor.py`。

## 开发说明

- 大型数据集、模型权重、向量数据库和日志文件不要提交到 Git。
- API key 不要写入 `config.yaml`；接入外部服务时建议使用环境变量或本地忽略的配置文件。
- 当某个 CLI 命令从预留状态变为已实现状态时，同步更新 README。
- 运行完整模型训练前，优先用小型 JSON 样例做本地冒烟测试。

## 许可证

本项目使用 Apache License 2.0，详见 [LICENSE](LICENSE)。
