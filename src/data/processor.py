"""Initial data processing pipeline for banking QA data."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.config_manager import ConfigManager
from src.logger import Logger


class DataProcessor:
    """Load local raw data and export SFT, DPO and knowledge files."""

    QUESTION_FIELDS = ["question", "query", "instruction", "prompt", "title", "问题"]
    ANSWER_FIELDS = ["answer", "output", "response", "chosen", "content", "答案"]
    REJECTED_FIELDS = ["rejected", "bad_answer", "negative", "错误答案"]

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = ConfigManager(config_path).get_all()
        self.data_cfg = self.config.get("data", {})
        self.field_mapping = self.data_cfg.get("field_mapping", {})
        self.logger = Logger.get_logger("DataProcessor", self.config.get("logging")).logger

    def run(
        self,
        split: str = "train",
        max_samples: Optional[int] = None,
        input_paths: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        resolved_input_paths = self._resolve_input_paths(split, input_paths=input_paths)
        rows = self._load_rows(resolved_input_paths)
        if max_samples is not None:
            rows = rows[:max_samples]

        records = [record for row in rows if (record := self._normalize_row(row))]
        if self.data_cfg.get("filter_banking", True):
            records = [record for record in records if self._is_banking_related(record)]

        records = self._deduplicate(records)
        lora_rows = [self._to_lora_row(record) for record in records]
        dpo_rows = [
            self._to_dpo_row(record)
            for record in records
            if record.get("rejected")
        ]
        knowledge_rows = [self._to_knowledge_row(record) for record in records]

        lora_path = self._write_json(lora_rows, self.data_cfg["lora_output_path"])
        dpo_path = self._write_json(dpo_rows, self.data_cfg["dpo_output_path"])
        knowledge_path = self._write_jsonl(
            knowledge_rows,
            self.data_cfg.get("knowledge_output_path", "./data/processed/knowledge.jsonl"),
        )

        summary = {
            "input_paths": [str(path) for path in resolved_input_paths],
            "loaded_rows": len(rows),
            "processed_records": len(records),
            "lora_rows": len(lora_rows),
            "dpo_rows": len(dpo_rows),
            "knowledge_rows": len(knowledge_rows),
            "lora_output_path": str(lora_path),
            "dpo_output_path": str(dpo_path),
            "knowledge_output_path": str(knowledge_path),
        }
        self.logger.info("Data processing summary: %s", summary)
        return summary

    def _resolve_input_paths(
        self,
        split: str,
        input_paths: Optional[List[str]] = None,
    ) -> List[Path]:
        configured = (
            input_paths
            or self.data_cfg.get("input_paths")
            or self.data_cfg.get(f"{split}_input_paths")
            or self.data_cfg.get("raw_data_paths")
            or []
        )
        paths = [Path(path).expanduser() for path in configured]
        if not paths:
            raise ValueError(
                "未配置原始数据路径。请在 data.input_paths 中配置 JSON/JSONL/CSV/TSV 文件或目录。"
            )
        return paths

    def _load_rows(self, paths: Iterable[Path]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for path in paths:
            if path.is_dir():
                for file_path in sorted(path.rglob("*")):
                    if file_path.is_file() and file_path.suffix.lower() in {
                        ".json",
                        ".jsonl",
                        ".csv",
                        ".tsv",
                    }:
                        rows.extend(self._load_file(file_path))
            elif path.is_file():
                rows.extend(self._load_file(path))
            else:
                raise FileNotFoundError(f"原始数据路径不存在: {path}")
        return rows

    def _load_file(self, path: Path) -> List[Dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            rows = obj if isinstance(obj, list) else [obj]
            return [row for row in rows if isinstance(row, dict)]
        if suffix == ".jsonl":
            rows = []
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()
                    if line:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            rows.append(obj)
            return rows
        if suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                return list(csv.DictReader(file, delimiter=delimiter))
        raise ValueError(f"暂不支持的数据文件类型: {path}")

    def _normalize_row(self, row: Dict[str, Any]) -> Optional[Dict[str, str]]:
        question = self._first_value(row, self._fields_for("question", self.QUESTION_FIELDS))
        answer = self._first_value(row, self._fields_for("answer", self.ANSWER_FIELDS))
        rejected = self._first_value(row, self._fields_for("rejected", self.REJECTED_FIELDS))
        if not question or not answer:
            return None
        return {
            "question": question,
            "answer": answer,
            "rejected": rejected,
            "source": self._first_value(row, self._fields_for("source", ["source", "来源"])),
            "category": self._first_value(row, self._fields_for("category", ["category", "业务类型"])),
        }

    def _fields_for(self, logical_name: str, defaults: List[str]) -> List[str]:
        configured = self.field_mapping.get(logical_name)
        if not configured:
            return defaults
        if isinstance(configured, str):
            configured_fields = [configured]
        else:
            configured_fields = [str(field) for field in configured]
        return configured_fields + [field for field in defaults if field not in configured_fields]

    @staticmethod
    def _first_value(row: Dict[str, Any], fields: List[str]) -> str:
        for field in fields:
            value = row.get(field)
            if value is not None and str(value).strip():
                return str(value).strip()
        return ""

    @staticmethod
    def _is_banking_related(record: Dict[str, str]) -> bool:
        text = f"{record.get('question', '')} {record.get('answer', '')}"
        keywords = [
            "银行",
            "贷款",
            "存款",
            "信用卡",
            "账户",
            "利率",
            "还款",
            "授信",
            "征信",
            "抵押",
            "理财",
            "支付",
            "结算",
            "票据",
        ]
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _deduplicate(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        unique: List[Dict[str, str]] = []
        for record in records:
            key = (record["question"], record["answer"])
            if key not in seen:
                seen.add(key)
                unique.append(record)
        return unique

    def _to_lora_row(self, record: Dict[str, str]) -> Dict[str, str]:
        return {
            "instruction": record["question"],
            "input": "",
            "output": record["answer"],
        }

    @staticmethod
    def _to_dpo_row(record: Dict[str, str]) -> Dict[str, str]:
        return {
            "prompt": record["question"],
            "chosen": record["answer"],
            "rejected": record["rejected"],
        }

    @staticmethod
    def _to_knowledge_row(record: Dict[str, str]) -> Dict[str, str]:
        return {
            "question": record["question"],
            "answer": record["answer"],
            "text": f"问题：{record['question']}\n答案：{record['answer']}",
            "source": record.get("source", ""),
            "category": record.get("category", ""),
        }

    @staticmethod
    def _write_json(rows: List[Dict[str, str]], path: str) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    @staticmethod
    def _write_jsonl(rows: List[Dict[str, str]], path: str) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        return output_path
