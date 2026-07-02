"""Retrieval evaluation metrics for RAG."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config_manager import ConfigManager
from src.logger import Logger
from src.rag import RAGRetriever


class RetrievalEvaluator:
    """Evaluate retrieval quality with simple labeled data."""

    QUERY_FIELDS = ["query", "question", "prompt", "问题"]
    RELEVANT_FIELDS = ["relevant", "relevant_text", "answer", "expected", "答案"]

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = ConfigManager(config_path).get_all()
        self.eval_cfg = self.config.get("evaluation", {})
        self.logger = Logger.get_logger("RetrievalEvaluator", self.config.get("logging")).logger
        self.retriever = RAGRetriever(config_path=config_path)

    def evaluate(
        self,
        data_path: Optional[str] = None,
        top_k: Optional[int] = None,
        output_path: Optional[str] = None,
        markdown_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = data_path or self.eval_cfg.get("retrieval_test_data_path") or self.eval_cfg.get(
            "test_data_path"
        )
        if not path:
            raise ValueError("未配置评估数据路径，请设置 evaluation.retrieval_test_data_path。")

        rows = self._load_rows(Path(path).expanduser())
        examples = [example for row in rows if (example := self._normalize_row(row))]
        if not examples:
            raise ValueError("评估数据中没有可用样本。")

        k = int(top_k or self.eval_cfg.get("retrieval_top_k", self.config["rag"]["retrieval_top_k"]))
        per_item: List[Dict[str, Any]] = []
        hit_count = 0
        recall_sum = 0.0
        reciprocal_rank_sum = 0.0

        for example in examples:
            results = self.retriever.retrieve(query=example["query"], top_k=k)
            matched_ranks = self._matched_ranks(results, example["relevant"])
            hit = bool(matched_ranks)
            hit_count += int(hit)
            recall = min(len(set(matched_ranks)) / len(example["relevant"]), 1.0)
            recall_sum += recall
            reciprocal_rank_sum += 1.0 / matched_ranks[0] if matched_ranks else 0.0
            per_item.append(
                {
                    "query": example["query"],
                    "relevant_count": len(example["relevant"]),
                    "hit": hit,
                    "recall": recall,
                    "first_hit_rank": matched_ranks[0] if matched_ranks else None,
                    "retrieved": [
                        {
                            "rank": index,
                            "score": item.score,
                            "id": item.chunk.id,
                            "source": item.chunk.source,
                        }
                        for index, item in enumerate(results, start=1)
                    ],
                }
            )

        total = len(examples)
        summary = {
            "examples": total,
            "top_k": k,
            "hit_rate": hit_count / total,
            "recall_at_k": recall_sum / total,
            "mrr": reciprocal_rank_sum / total,
            "items": per_item,
        }
        resolved_output_path = output_path or self.eval_cfg.get("retrieval_report_path")
        if resolved_output_path:
            self._write_report(summary, resolved_output_path)
            summary["report_path"] = str(Path(resolved_output_path))
        resolved_markdown_path = markdown_path or self.eval_cfg.get("retrieval_markdown_report_path")
        if resolved_markdown_path:
            self._write_markdown_report(summary, resolved_markdown_path)
            summary["markdown_report_path"] = str(Path(resolved_markdown_path))
        self.logger.info("Retrieval evaluation summary: %s", summary)
        return summary

    def _matched_ranks(self, results, relevant_texts: List[str]) -> List[int]:
        ranks: List[int] = []
        normalized_relevant = [self._normalize_text(text) for text in relevant_texts]
        for rank, item in enumerate(results, start=1):
            chunk_text = self._normalize_text(item.chunk.text)
            for relevant in normalized_relevant:
                if relevant and (relevant in chunk_text or chunk_text in relevant):
                    ranks.append(rank)
                    break
        return ranks

    def _load_rows(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"评估数据路径不存在: {path}")
        if path.suffix.lower() == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            rows.append(obj)
            return rows
        if path.suffix.lower() == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            rows = obj if isinstance(obj, list) else [obj]
            return [row for row in rows if isinstance(row, dict)]
        raise ValueError(f"暂不支持的评估数据类型: {path}")

    def _normalize_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        query = self._first_value(row, self.QUERY_FIELDS)
        relevant_value = row.get("relevant_texts") or row.get("relevant_answers")
        if relevant_value is None:
            relevant_value = self._first_value(row, self.RELEVANT_FIELDS)
        relevant = self._normalize_relevant(relevant_value)
        if not query or not relevant:
            return None
        return {"query": query, "relevant": relevant}

    @staticmethod
    def _first_value(row: Dict[str, Any], fields: List[str]) -> str:
        for field in fields:
            value = row.get(field)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    @staticmethod
    def _normalize_relevant(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text).split())

    @staticmethod
    def _write_report(summary: Dict[str, Any], path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _write_markdown_report(summary: Dict[str, Any], path: str) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# 检索评估报告",
            "",
            f"- 样本数: {summary['examples']}",
            f"- Top-K: {summary['top_k']}",
            f"- Hit Rate: {summary['hit_rate']:.4f}",
            f"- Recall@K: {summary['recall_at_k']:.4f}",
            f"- MRR: {summary['mrr']:.4f}",
            "",
            "## 未命中样本",
            "",
        ]
        misses = [item for item in summary["items"] if not item["hit"]]
        if not misses:
            lines.append("无。")
        else:
            for index, item in enumerate(misses, start=1):
                lines.append(f"{index}. {item['query']}")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
