"""DPO optimization trainer."""

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config_manager import ConfigManager
from src.logger import Logger


class DPOOptimizer:
    """DPO optimizer for pairwise preference training."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: str = "config.yaml",
    ) -> None:
        self.config = config or ConfigManager(config_path).get_all()
        self.logger = Logger.get_logger("DPOOptimizer", self.config.get("logging")).logger

        self.models_cfg = self.config.get("models", {})
        self.dpo_cfg = self.config.get("dpo", {})
        self.data_cfg = self.config.get("data", {})

    def load_lora_model(self, model_path: Optional[str] = None):
        """Load base model and apply LoRA adapter weights."""
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Missing dependencies for DPO training. Install with: "
                "pip install torch transformers peft trl datasets"
            ) from exc

        base_model_path = self.models_cfg.get("base_model_path")
        adapter_path = model_path or self.models_cfg.get(
            "lora_adapter_path", "./models/lora_adapter"
        )
        self.logger.info(
            "Loading base model from %s and LoRA adapter from %s",
            base_model_path,
            adapter_path,
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True)
        model = PeftModel.from_pretrained(base_model, adapter_path)
        return model, tokenizer

    def train_with_preferences(self, pairwise_data_path: Optional[str] = None) -> Dict[str, Any]:
        """Run DPO training using pairwise preference data."""
        try:
            import torch
            from datasets import Dataset
            from trl import DPOConfig, DPOTrainer
        except ImportError as exc:
            raise ImportError(
                "Missing dependencies for DPO training. Install with: "
                "pip install torch transformers peft trl datasets"
            ) from exc

        data_path = pairwise_data_path or self.data_cfg.get(
            "dpo_output_path", "./data/processed/dpo_data.json"
        )
        output_path = self.models_cfg.get("dpo_model_path", "./models/dpo_model")
        rows = self._load_pairwise_data(data_path)
        if not rows:
            raise ValueError("No DPO pairwise rows found.")
        data_summary = self._summarize_pairwise_data(rows)
        self.logger.info("DPO data summary: %s", data_summary)

        model, tokenizer = self.load_lora_model()
        dataset = Dataset.from_list(rows)

        dpo_args = DPOConfig(
            output_dir=output_path,
            beta=float(self.dpo_cfg.get("beta", 0.1)),
            per_device_train_batch_size=int(self.dpo_cfg.get("batch_size", 2)),
            gradient_accumulation_steps=int(
                self.dpo_cfg.get("gradient_accumulation_steps", 8)
            ),
            learning_rate=float(self.dpo_cfg.get("learning_rate", 5e-5)),
            num_train_epochs=float(self.dpo_cfg.get("epochs", 2)),
            warmup_steps=int(self.dpo_cfg.get("warmup_steps", 50)),
            logging_steps=int(self.dpo_cfg.get("logging_steps", 10)),
            save_steps=int(self.dpo_cfg.get("save_steps", 100)),
            save_total_limit=2,
            max_length=int(self.dpo_cfg.get("max_length", 512)),
            max_prompt_length=int(self.dpo_cfg.get("max_prompt_length", 256)),
            fp16=bool(torch.cuda.is_available()),
            report_to=[],
        )

        trainer_kwargs = {
            "model": model,
            "ref_model": None,
            "args": dpo_args,
            "train_dataset": dataset,
        }

        signature = inspect.signature(DPOTrainer.__init__).parameters
        if "tokenizer" in signature:
            trainer_kwargs["tokenizer"] = tokenizer
        elif "processing_class" in signature:
            trainer_kwargs["processing_class"] = tokenizer

        trainer = DPOTrainer(**trainer_kwargs)

        self.logger.info("Starting DPO training with %d pairs.", len(rows))
        trainer.train()

        Path(output_path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        self.logger.info("DPO model saved to %s", output_path)

        summary = {
            "train_pairs": len(rows),
            "data_summary": data_summary,
            "output_path": output_path,
            "data_path": data_path,
        }
        self._write_summary(summary, output_path)
        return summary

    @staticmethod
    def _load_pairwise_data(path: str) -> List[Dict[str, str]]:
        with open(path, "r", encoding="utf-8") as file:
            obj = json.load(file)
        if not isinstance(obj, list):
            raise ValueError("DPO data must be a list of JSON objects.")

        rows: List[Dict[str, str]] = []
        for item in obj:
            if not isinstance(item, dict):
                continue
            prompt = str(item.get("prompt", "")).strip()
            chosen = str(item.get("chosen", "")).strip()
            rejected = str(item.get("rejected", "")).strip()
            if prompt and chosen and rejected:
                rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        return rows

    @staticmethod
    def _summarize_pairwise_data(rows: List[Dict[str, str]]) -> Dict[str, Any]:
        prompt_lengths = [len(row["prompt"]) for row in rows]
        chosen_lengths = [len(row["chosen"]) for row in rows]
        rejected_lengths = [len(row["rejected"]) for row in rows]
        same_answer_pairs = sum(1 for row in rows if row["chosen"] == row["rejected"])
        return {
            "pairs": len(rows),
            "avg_prompt_chars": sum(prompt_lengths) / len(prompt_lengths),
            "avg_chosen_chars": sum(chosen_lengths) / len(chosen_lengths),
            "avg_rejected_chars": sum(rejected_lengths) / len(rejected_lengths),
            "max_prompt_chars": max(prompt_lengths),
            "max_chosen_chars": max(chosen_lengths),
            "max_rejected_chars": max(rejected_lengths),
            "same_answer_pairs": same_answer_pairs,
        }

    @staticmethod
    def _write_summary(summary: Dict[str, Any], output_path: str) -> None:
        summary_path = Path(output_path) / "training_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
