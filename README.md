# Banking RAG

面向银行业务问答场景的 RAG 与模型微调实验项目。仓库目前聚焦三件事：

- 用配置文件统一管理数据、模型、LoRA、DPO、RAG 和日志参数
- 提供 LoRA SFT 与 DPO 偏好优化训练入口
- 为后续银行领域数据处理、向量检索、重排序和推理服务预留 CLI 流程

> 当前代码仍处于实验/搭建阶段。`train-lora` 和 `train-dpo` 已有实现；`setup-rag` 与 `inference` 是预留命令；`process-data` 入口已声明，但依赖的 `src.data.processor` 模块尚未提交到仓库。

## Features

### Implemented

- `ConfigManager`: 支持 YAML / JSON 配置加载、必填项校验、默认值补齐和点号路径读取。
- `Logger`: 统一控制台与文件日志输出。
- `LoRATrainer`: 基于 `transformers`、`datasets`、`peft` 的因果语言模型 LoRA 微调流程。
- `DPOOptimizer`: 基于 `trl` 的 DPO 偏好优化流程，兼容不同版本 `DPOTrainer` 的 `tokenizer` / `processing_class` 参数。
- CLI:
  - `train-lora`
  - `train-dpo`
  - `process-data`（入口存在，数据处理模块待补齐）
  - `setup-rag`（预留）
  - `inference`（预留）

### Planned

- 银行业数据抽取、清洗、过滤与 SFT / DPO 数据构造
- BGE-M3 embedding 与 reranker 微调
- FAISS 向量库构建
- 自适应检索策略
- 银行业问答推理链路
- 评估指标与回归测试

## Project Structure

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

## Requirements

- Python 3.10+ recommended
- PyTorch 2.x
- `transformers`
- `datasets`
- `peft`
- `trl`
- `accelerate`
- `faiss-cpu`
- `sentence-transformers`

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you train large language models locally, prepare a CUDA-capable environment and place the base model files under the paths configured in `config.yaml`.

## Configuration

The default config is [config.yaml](config.yaml). Key sections:

| Section | Purpose |
| --- | --- |
| `data` | Dataset id, banking filter switch, output paths for SFT and DPO data |
| `models` | Base model, embedding model, reranker, LoRA adapter and DPO output paths |
| `lora` | LoRA rank, alpha, dropout, learning rate, batch size and training steps |
| `dpo` | DPO beta, learning rate, batch size and training steps |
| `rag` | Chunk size, retrieval top-k, rerank top-n and vector DB path |
| `inference` | Generation temperature, max tokens, top-p and repetition penalty |
| `logging` | Log level, console/file output and log file path |
| `evaluation` | Test data path and baseline score placeholder |

Before training, update at least these paths:

```yaml
models:
  base_model_path: "./models/Baichuan2-7B-Base"
  lora_adapter_path: "./models/lora_adapter"
  dpo_model_path: "./models/dpo_model"

data:
  lora_output_path: "./data/processed/lora_data.json"
  dpo_output_path: "./data/processed/dpo_data.json"
```

## CLI Usage

Show available commands:

```bash
python -m src.cli.main --help
```

Run LoRA fine-tuning:

```bash
python -m src.cli.main train-lora \
  --config config.yaml \
  --data-path ./data/processed/lora_data.json
```

Expected SFT data format:

```json
[
  {
    "instruction": "请解释企业流动资金贷款的适用场景。",
    "input": "",
    "output": "企业流动资金贷款主要用于..."
  }
]
```

Run DPO optimization:

```bash
python -m src.cli.main train-dpo \
  --config config.yaml \
  --data-path ./data/processed/dpo_data.json
```

Expected DPO data format:

```json
[
  {
    "prompt": "客户想了解企业贷款准入条件，应如何回答？",
    "chosen": "可以从企业资质、经营流水、征信和担保方式等方面说明...",
    "rejected": "企业贷款就是给企业的钱，满足条件就能申请。"
  }
]
```

Declared but not fully implemented yet:

```bash
python -m src.cli.main process-data --config config.yaml
python -m src.cli.main setup-rag
python -m src.cli.main inference
```

`setup-rag` and `inference` currently raise `NotImplementedError`. `process-data` requires adding `src/data/processor.py`.

## Development Notes

- Keep large datasets, model weights, vector databases and logs out of Git.
- Keep API keys out of `config.yaml`; use environment variables or a local ignored config when integrating external services.
- Add or update README commands whenever a CLI command moves from placeholder to implemented.
- Prefer small sample JSON files for local smoke tests before running full model training.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).
