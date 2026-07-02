# 银行业 RAG 问答实验项目

面向银行业务问答场景的 RAG 与模型微调实验项目。仓库目前聚焦三件事：

- 用配置文件统一管理数据、模型、LoRA、DPO、RAG 和日志参数
- 提供 LoRA SFT 与 DPO 偏好优化训练入口
- 为后续银行领域数据处理、向量检索、重排序和推理服务预留 CLI 流程

> 当前代码仍处于实验/搭建阶段。`train-lora` 和 `train-dpo` 已有实现；`setup-rag` 与 `inference` 是预留命令；`process-data` 入口已声明，但依赖的 `src.data.processor` 模块尚未提交到仓库。

## 功能概览

### 已实现

- `ConfigManager`: 支持 YAML / JSON 配置加载、必填项校验、默认值补齐、点号路径读取、`config.local.yaml` 本地覆盖和 `BANKING_RAG__...` 环境变量覆盖。
- `Logger`: 统一控制台与文件日志输出。
- `LoRATrainer`: 基于 `transformers`、`datasets`、`peft` 的因果语言模型 LoRA 微调流程。
- `DPOOptimizer`: 基于 `trl` 的 DPO 偏好优化流程，兼容不同版本 `DPOTrainer` 的 `tokenizer` / `processing_class` 参数。
- `DataProcessor`: 支持从 JSON、JSONL、CSV、TSV 或目录读取初始问答数据，输出 LoRA、DPO 和 RAG 知识源文件。
- `ConfigValidator`: 检查核心配置、检索参数和关键路径，并输出结构化错误/提醒。
- `RAGIndexer`: 支持从 txt、md、json、jsonl 或目录读取知识源，按段落/句子边界切块，完成 embedding 和 FAISS 建库。
- `RAGRetriever`: 加载本地 FAISS 索引并返回相关文本块，支持相似度检索、MMR 去重检索、可选 rerank 和索引状态查看。
- `InferenceEngine`: 执行检索、上下文拼接，可控制 prompt / sources 输出，并可选调用本地生成模型。
- `RetrievalEvaluator`: 基于标注 query 和相关文本计算 Hit Rate、Recall@k 和 MRR。
- 命令行入口：
  - `validate-config`
  - `train-lora`
  - `train-dpo`
  - `process-data`
  - `setup-rag`
  - `query-rag`
  - `evaluate-retrieval`
  - `inference`

### 规划中

- 更细粒度的银行字段映射和数据质量规则
- BGE-M3 embedding 与 reranker 微调
- rerank 训练与评估
- 自适应检索策略
- 生成模型加载策略优化
- 生成答案评估与回归测试

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
│   ├── inference/
│   │   └── engine.py        # 检索增强推理编排
│   ├── rag/
│   │   └── pipeline.py      # 文档读取、切块、建库与检索
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
| `rag` | 知识源路径、分块大小、检索 top-k、MMR 参数、rerank 开关、索引文件和向量库路径 |
| `inference` | 是否启用生成模型、模型路径、输出字段、生成温度、最大 token 数、top-p 和重复惩罚 |
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

rag:
  source_paths:
    - "./data/processed/knowledge.jsonl"
```

如果本地路径、密钥或实验参数不希望提交到 Git，可以新建 `config.local.yaml`，它会自动覆盖 `config.yaml` 中的同名字段：

```yaml
data:
  input_paths:
    - "./data/raw/local_banking_qa.jsonl"
  field_mapping:
    question: ["用户问题", "question"]
    answer: ["标准答案", "answer"]
    rejected: ["错误答案"]
    source: ["来源"]
    category: ["业务类型"]
```

也可以用环境变量临时覆盖配置，格式是 `BANKING_RAG__配置段__配置项`：

```bash
export BANKING_RAG__RAG__RETRIEVAL_TOP_K=5
export BANKING_RAG__INFERENCE__USE_GENERATION=false
```

## 命令行用法

查看可用命令：

```bash
python -m src.cli.main --help
```

校验配置：

```bash
python -m src.cli.main validate-config --config config.yaml
```

处理本地问答数据：

```bash
python -m src.cli.main process-data \
  --config config.yaml \
  --input-paths ./data/raw/banking_qa.jsonl \
  --max-samples 100
```

`process-data` 会读取 `data.input_paths` 中的 JSON、JSONL、CSV、TSV 文件或目录，自动识别常见问题/答案字段，并输出：

- `data.lora_output_path`
- `data.dpo_output_path`
- `data.knowledge_output_path`

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

构建 RAG 向量索引：

```bash
python -m src.cli.main setup-rag \
  --config config.yaml \
  --documents ./data/processed/knowledge.jsonl \
  --reset
```

知识源支持 `.txt`、`.md`、`.json`、`.jsonl` 文件，也支持目录。JSON / JSONL 会优先抽取 `text`、`content`、`question`、`answer`、`instruction`、`output`、`prompt`、`chosen` 等字段。

直接查询本地 RAG 索引：

```bash
python -m src.cli.main query-rag \
  --config config.yaml \
  --query "企业流动资金贷款适合什么场景？" \
  --top-k 5 \
  --status
```

`query-rag` 会返回命中的文本块、相似度分数、来源文件和索引状态，适合在接入生成模型前先检查检索质量。

评估检索效果：

```bash
python -m src.cli.main evaluate-retrieval \
  --config config.yaml \
  --data-path ./data/retrieval_eval.jsonl \
  --top-k 10 \
  --output ./data/reports/retrieval_eval_report.json \
  --markdown-output ./data/reports/retrieval_eval_report.md
```

评估数据支持 JSON / JSONL，每条样本至少包含问题和相关文本：

```json
{"query": "企业贷款需要什么条件？", "relevant_texts": ["企业贷款通常需要营业执照、经营流水、征信和担保材料。"]}
```

执行检索增强推理：

```bash
python -m src.cli.main inference \
  --config config.yaml \
  --query "企业流动资金贷款适合什么场景？" \
  --top-k 5
```

默认情况下，`inference` 只返回检索上下文和拼好的 prompt，不调用本地大模型。若已经准备好生成模型，可在配置中设置 `inference.model_path`，或使用默认 `models.dpo_model_path`，然后添加 `--generate`：

```bash
python -m src.cli.main inference \
  --config config.yaml \
  --query "企业流动资金贷款适合什么场景？" \
  --top-k 5 \
  --generate
```

如果只想拿最终回答和来源数量，可以隐藏 prompt 或来源文本：

```bash
python -m src.cli.main inference \
  --config config.yaml \
  --query "企业流动资金贷款适合什么场景？" \
  --top-k 5 \
  --no-prompt \
  --no-sources
```

如果要启用重排序，在 `config.yaml` 中设置：

```yaml
rag:
  enable_rerank: true
  rerank_top_n: 5

models:
  rerank_model_path: "./models/bge-m3-reranker"
```

数据清洗、样例数据和测试会在拿到真实数据后继续完善。

## 开发说明

- 大型数据集、模型权重、向量数据库和日志文件不要提交到 Git。
- API key 不要写入 `config.yaml`；接入外部服务时建议使用环境变量或本地忽略的配置文件。
- 当某个 CLI 命令从预留状态变为已实现状态时，同步更新 README。
- 运行完整模型训练前，优先用小型 JSON 样例做本地冒烟测试。

## 许可证

本项目使用 Apache License 2.0，详见 [LICENSE](LICENSE)。
