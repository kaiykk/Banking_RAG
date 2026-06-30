"""Project configuration validation helpers."""

from pathlib import Path
from typing import Any, Dict, List

from src.config_manager import ConfigManager


class ConfigValidator:
    """Validate configuration and return structured warnings."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = ConfigManager(config_path).get_all()

    def validate(self) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []

        self._check_positive_int("rag.chunk_max_chars", errors)
        self._check_positive_int("rag.retrieval_top_k", errors)
        self._check_positive_int("rag.retrieval_fetch_k", errors)
        self._check_positive_int("rag.rerank_top_n", errors)
        self._check_positive_int("rag.rerank_batch_size", errors)
        self._check_positive_int("evaluation.retrieval_top_k", errors)
        self._check_float_range("rag.mmr_lambda", 0.0, 1.0, errors)

        strategy = self._get("rag.retrieval_strategy")
        if strategy not in {"similarity", "mmr"}:
            errors.append("rag.retrieval_strategy 仅支持 similarity 或 mmr。")

        input_paths = self._get("data.input_paths", [])
        field_mapping = self._get("data.field_mapping", {})
        if field_mapping and not isinstance(field_mapping, dict):
            errors.append("data.field_mapping 必须是字典。")
        if not input_paths:
            warnings.append("data.input_paths 为空，process-data 暂无原始数据入口。")
        else:
            for path in input_paths:
                if not Path(path).expanduser().exists():
                    warnings.append(f"data.input_paths 中的路径不存在: {path}")

        source_paths = self._get("rag.source_paths", [])
        if not source_paths:
            warnings.append("rag.source_paths 为空，setup-rag 需要通过配置或 --documents 指定知识源。")

        embedding_path = self._get("models.embedding_model_path")
        if embedding_path and not Path(embedding_path).expanduser().exists():
            warnings.append(
                f"models.embedding_model_path 不存在，将依赖 Hugging Face 名称或运行时下载: {embedding_path}"
            )

        if self._get("rag.enable_rerank", False):
            rerank_path = self._get("models.rerank_model_path")
            if not rerank_path:
                errors.append("启用 rag.enable_rerank 后必须配置 models.rerank_model_path。")
            elif not Path(rerank_path).expanduser().exists():
                warnings.append(
                    f"models.rerank_model_path 不存在，将依赖 Hugging Face 名称或运行时下载: {rerank_path}"
                )

        retrieval_eval_path = self._get("evaluation.retrieval_test_data_path")
        if retrieval_eval_path and not Path(retrieval_eval_path).expanduser().exists():
            warnings.append(f"evaluation.retrieval_test_data_path 不存在: {retrieval_eval_path}")

        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
        }

    def _get(self, key_path: str, default: Any = None) -> Any:
        value: Any = self.config
        for key in key_path.split("."):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def _check_positive_int(self, key_path: str, errors: List[str]) -> None:
        value = self._get(key_path)
        try:
            if int(value) <= 0:
                errors.append(f"{key_path} 必须大于 0。")
        except (TypeError, ValueError):
            errors.append(f"{key_path} 必须是整数。")

    def _check_float_range(
        self,
        key_path: str,
        lower: float,
        upper: float,
        errors: List[str],
    ) -> None:
        value = self._get(key_path)
        try:
            number = float(value)
        except (TypeError, ValueError):
            errors.append(f"{key_path} 必须是数字。")
            return
        if not lower <= number <= upper:
            errors.append(f"{key_path} 必须在 {lower} 到 {upper} 之间。")
