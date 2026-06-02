"""Inference orchestration for retrieval-augmented answers."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config_manager import ConfigManager
from src.logger import Logger
from src.rag.pipeline import RetrievalResult
from src.rag import RAGRetriever


class InferenceEngine:
    """Retrieve context and optionally generate an answer with a local model."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = ConfigManager(config_path).get_all()
        self.logger = Logger.get_logger("InferenceEngine", self.config.get("logging")).logger
        self.rag_cfg = self.config.get("rag", {})
        self.inference_cfg = self.config.get("inference", {})
        self.models_cfg = self.config.get("models", {})
        self.retriever = RAGRetriever(config_path=config_path)
        self._model = None
        self._tokenizer = None

    def answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        generate: Optional[bool] = None,
        include_prompt: Optional[bool] = None,
        include_sources: Optional[bool] = None,
    ) -> Dict[str, Any]:
        if not query.strip():
            raise ValueError("query 不能为空。")

        results = self.retriever.retrieve(query=query, top_k=top_k)
        context = self._format_context(results)
        prompt = self._build_prompt(query=query, context=context)

        should_generate = (
            bool(self.inference_cfg.get("use_generation", False))
            if generate is None
            else generate
        )
        if should_generate:
            answer = self._generate(prompt)
            mode = "generated"
        else:
            answer = self._build_retrieval_only_answer(results)
            mode = "retrieval_only"

        payload: Dict[str, Any] = {
            "query": query,
            "mode": mode,
            "answer": answer,
            "source_count": len(results),
        }
        if self._resolve_bool(include_prompt, "include_prompt", True):
            payload["prompt"] = prompt
        if self._resolve_bool(include_sources, "include_sources", True):
            payload["sources"] = [self._source_to_dict(item) for item in results]
        return payload

    def _build_prompt(self, query: str, context: str) -> str:
        template = self.rag_cfg.get(
            "context_template",
            "参考信息：\n{context}\n\n问题：{question}\n答案：",
        )
        return template.format(context=context, question=query)

    @staticmethod
    def _format_context(results: List[RetrievalResult]) -> str:
        blocks = []
        for index, item in enumerate(results, start=1):
            blocks.append(
                f"[{index}] 来源: {item.chunk.source}\n"
                f"相关度: {item.score:.4f}\n"
                f"{item.chunk.text}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _source_to_dict(item: RetrievalResult) -> Dict[str, Any]:
        return {
            "score": item.score,
            "id": item.chunk.id,
            "source": item.chunk.source,
            "text": item.chunk.text,
            "metadata": item.chunk.metadata,
        }

    @staticmethod
    def _build_retrieval_only_answer(results: List[RetrievalResult]) -> str:
        if not results:
            return "未检索到可用参考信息。"
        return "已完成检索，当前未启用本地生成模型。请根据 sources 中的上下文生成最终回答。"

    def _resolve_bool(
        self,
        override: Optional[bool],
        config_key: str,
        default: bool,
    ) -> bool:
        if override is not None:
            return override
        return bool(self.inference_cfg.get(config_key, default))

    def _generate(self, prompt: str) -> str:
        self._ensure_model_loaded()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[-1]
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=int(self.inference_cfg.get("max_tokens", 512)),
            temperature=float(self.inference_cfg.get("temperature", 0.7)),
            top_p=float(self.inference_cfg.get("top_p", 0.9)),
            repetition_penalty=float(self.inference_cfg.get("repetition_penalty", 1.1)),
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        answer_ids = output_ids[0][input_length:]
        return self._tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    def _ensure_model_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("缺少 transformers。请先执行: pip install transformers") from exc

        model_path = self.inference_cfg.get("model_path") or self.models_cfg.get(
            "dpo_model_path"
        )
        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(
                f"未找到可用于生成的模型路径: {model_path}。"
                "如只需检索上下文，请不要传 --generate。"
            )

        self.logger.info("Loading inference model from %s", model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        torch_dtype = self.inference_cfg.get("torch_dtype")
        device_map = self.inference_cfg.get("device_map")
        if torch_dtype:
            try:
                import torch
            except ImportError as exc:
                raise ImportError("设置 torch_dtype 需要安装 torch。") from exc
            dtype = getattr(torch, str(torch_dtype), None)
            if dtype is None:
                raise ValueError(f"不支持的 torch_dtype: {torch_dtype}")
            model_kwargs["torch_dtype"] = dtype
        if device_map:
            model_kwargs["device_map"] = device_map

        self._model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self._model.eval()
