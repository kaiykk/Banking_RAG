"""Inference orchestration for retrieval-augmented answers."""

from pathlib import Path
from typing import Any, Dict, Optional

from src.config_manager import ConfigManager
from src.logger import Logger
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
    ) -> Dict[str, Any]:
        results = self.retriever.retrieve(query=query, top_k=top_k)
        context = self._format_context(results)
        prompt = self.rag_cfg.get(
            "context_template",
            "参考信息：\n{context}\n\n问题：{question}\n答案：",
        ).format(context=context, question=query)

        should_generate = (
            bool(self.inference_cfg.get("use_generation", False))
            if generate is None
            else generate
        )
        if should_generate:
            answer = self._generate(prompt)
            mode = "generated"
        else:
            answer = "已完成检索，当前未启用本地生成模型。请根据 sources 中的上下文生成最终回答。"
            mode = "retrieval_only"

        return {
            "query": query,
            "mode": mode,
            "answer": answer,
            "prompt": prompt,
            "sources": [
                {
                    "score": item.score,
                    "id": item.chunk.id,
                    "source": item.chunk.source,
                    "text": item.chunk.text,
                    "metadata": item.chunk.metadata,
                }
                for item in results
            ],
        }

    @staticmethod
    def _format_context(results) -> str:
        blocks = []
        for index, item in enumerate(results, start=1):
            blocks.append(
                f"[{index}] 来源: {item.chunk.source}\n"
                f"相关度: {item.score:.4f}\n"
                f"{item.chunk.text}"
            )
        return "\n\n".join(blocks)

    def _generate(self, prompt: str) -> str:
        self._ensure_model_loaded()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        model_device = getattr(self._model, "device", None)
        if model_device is not None:
            inputs = {key: value.to(model_device) for key, value in inputs.items()}
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=int(self.inference_cfg.get("max_tokens", 512)),
            temperature=float(self.inference_cfg.get("temperature", 0.7)),
            top_p=float(self.inference_cfg.get("top_p", 0.9)),
            repetition_penalty=float(self.inference_cfg.get("repetition_penalty", 1.1)),
            do_sample=True,
        )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

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
        self._model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
