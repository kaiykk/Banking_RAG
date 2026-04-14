# 银行业RAG问答系统

基于Baichuan2和BGE-M3的银行领域智能问答系统，结合LoRA微调、DPO优化和自适应RAG技术。

## 系统概述

本系统是一个企业级银行问答解决方案，主要特点包括：

- **领域微调模型**: 使用LoRA和DPO技术微调Baichuan2-7B模型
- **高精度检索**: BGE-M3双重微调（嵌入+重排序）
- **自适应RAG**: 受PageIndex启发的智能检索策略
- **向量数据库**: 基于FAISS的高效向量检索
- **金融数据集**: 使用DISC-FIN-SFT标注的银行金融数据

## 项目结构

```
banking-rag-qa-system/
├── config.yaml              # 系统配置文件
├── requirements.txt         # Python依赖项
├── README.md               # 项目说明文档
├── src/                    # 源代码目录
│   ├── __init__.py
│   ├── config_manager.py   # 配置管理器
│   ├── logger.py          # 日志记录器
│   ├── data/              # 数据处理模块
│   ├── training/          # 训练流程模块
│   ├── rag/               # RAG系统模块
│   ├── inference/         # 推理引擎模块
│   └── cli/               # 命令行界面
├── data/                   # 数据目录
│   ├── processed/         # 处理后的数据
│   └── vector_db/         # 向量数据库
├── models/                 # 模型目录
│   ├── Baichuan2-7B-Base/ # 基础模型
│   ├── bge-m3/            # 嵌入模型
│   ├── lora_adapter/      # LoRA适配器
│   └── dpo_model/         # DPO优化模型
└── logs/                   # 日志目录
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置系统

编辑 `config.yaml` 文件，配置以下内容：

- 数据集ID（默认使用DISC-FIN-SFT）
- 模型路径
- 训练超参数
- RAG参数
- GPT-4 API密钥（用于风格优化）

### 3. 数据处理

```bash
# 加载和处理数据集
python -m src.cli.main process-data
```

### 4. 模型训练

```bash
# LoRA微调
python -m src.cli.main train-lora

# DPO优化
python -m src.cli.main train-dpo
```

### 5. RAG设置

```bash
# 构建向量数据库
python -m src.cli.main setup-rag

# 微调嵌入模型
python -m src.cli.main finetune-embedding

# 微调重排序模型
python -m src.cli.main finetune-rerank
```

### 6. 推理测试

```bash
# 启动问答系统
python -m src.cli.main inference --query "什么是企业贷款？"
```

## 主要功能

### 数据处理流程

1. **数据加载**: 从Hugging Face加载DISC-FIN-SFT数据集
2. **质量过滤**: 使用困惑度分数过滤低质量数据
3. **提示转换**: 应用模板转换问题格式
4. **风格优化**: 使用GPT-4生成产品经理风格答案

### 训练流程

1. **LoRA微调**: 在银行问答数据上微调Baichuan2模型
2. **DPO优化**: 使用成对数据优化模型输出风格

### RAG系统

1. **知识分块**: 将问答对分块为可检索片段
2. **向量嵌入**: 使用BGE-M3生成文本嵌入
3. **向量检索**: FAISS快速相似度搜索
4. **结果重排序**: BGE-M3重排序模型优化结果
5. **自适应检索**: 根据查询复杂度动态调整检索策略

### 推理引擎

1. **查询处理**: 接收用户查询
2. **上下文检索**: RAG系统检索相关知识
3. **答案生成**: 微调模型生成专业答案
4. **日志记录**: 记录查询和响应用于评估

## 配置说明

### 数据配置

- `dataset_id`: Hugging Face数据集ID
- `filter_banking`: 是否过滤银行相关数据
- `perplexity_threshold`: 困惑度过滤阈值

### 模型配置

- `base_model_path`: Baichuan2基础模型路径
- `embedding_model_path`: BGE-M3嵌入模型路径
- `rerank_model_path`: BGE-M3重排序模型路径

### 训练配置

- **LoRA**: rank, alpha, dropout, learning_rate, batch_size, epochs
- **DPO**: beta, learning_rate, batch_size, epochs

### RAG配置

- `chunk_max_tokens`: 每个块的最大令牌数
- `retrieval_top_k`: 检索的top-k块数
- `rerank_top_n`: 重排序后保留的top-n块数

## 性能指标

- **TruLen分数**: 企业产品知识准确性评估
- **检索延迟**: 目标 <500ms
- **基线对比**: 目标超越67.2的基线分数


