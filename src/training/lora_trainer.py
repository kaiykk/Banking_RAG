"""LoRA fine-tuning trainer."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config_manager import ConfigManager
from src.logger import Logger


class LoRATrainer:
    """LoRA trainer for banking QA SFT data."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: str = "config.yaml",
    ) -> None:
        self.config = config or ConfigManager(config_path).get_all()
        self.logger = Logger.get_logger("LoRATrainer", self.config.get("logging")).logger

        self.models_cfg = self.config.get("models", {})
        self.lora_cfg = self.config.get("lora", {})
        self.data_cfg = self.config.get("data", {})

    def load_base_model(self):
        """Load base model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Missing dependencies for LoRA training. Install with: "
                "pip install torch transformers peft datasets"
            ) from exc

        model_path = self.models_cfg.get("base_model_path")
        self.logger.info("Loading base model from %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer

    def apply_lora_adapters(self, model):
        """Apply LoRA adapters to base model."""
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise ImportError(
                "Missing dependency 'peft'. Install with: pip install peft"
            ) from exc

        lora_config = LoraConfig(
            r=self.lora_cfg.get("rank", 8),
            lora_alpha=self.lora_cfg.get("alpha", 16),
            lora_dropout=self.lora_cfg.get("dropout", 0.05),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=self.lora_cfg.get(
                "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()
        return peft_model

    def train(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """Run LoRA fine-tuning."""
        try:
            import torch
            from datasets import Dataset
            from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
        except ImportError as exc:
            raise ImportError(
                "Missing dependencies for LoRA training. Install with: "
                "pip install torch transformers peft datasets"
            ) from exc

        train_data_path = data_path or self.data_cfg.get(
            "lora_output_path", "./data/processed/lora_data.json"
        )
        output_path = self.models_cfg.get("lora_adapter_path", "./models/lora_adapter")

        records = self._load_sft_data(train_data_path)
        if not records:
            raise ValueError("No training records found in SFT data file.")
        data_summary = self._summarize_sft_data(records)
        self.logger.info("LoRA data summary: %s", data_summary)

        model, tokenizer = self.load_base_model()
        model = self.apply_lora_adapters(model)

        dataset = Dataset.from_dict({"text": [self._build_sft_text(item) for item in records]})

        max_length = int(self.lora_cfg.get("max_length", 512))

        def tokenize_fn(batch):
            return tokenizer(batch["text"], truncation=True, max_length=max_length)

        tokenized_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        tokenized_ds = tokenized_ds.map(lambda batch: {"labels": batch["input_ids"]}, batched=True)

        training_args = TrainingArguments(
            output_dir=output_path,
            per_device_train_batch_size=int(self.lora_cfg.get("batch_size", 4)),
            gradient_accumulation_steps=int(
                self.lora_cfg.get("gradient_accumulation_steps", 4)
            ),
            learning_rate=float(self.lora_cfg.get("learning_rate", 1e-4)),
            num_train_epochs=float(self.lora_cfg.get("epochs", 3)),
            warmup_steps=int(self.lora_cfg.get("warmup_steps", 100)),
            logging_steps=int(self.lora_cfg.get("logging_steps", 10)),
            save_steps=int(self.lora_cfg.get("save_steps", 200)),
            save_total_limit=2,
            fp16=bool(torch.cuda.is_available()),
            report_to=[],
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        self.logger.info("Starting LoRA training with %d samples.", len(records))
        trainer.train()

        Path(output_path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        self.logger.info("LoRA adapters saved to %s", output_path)

        summary = {
            "train_samples": len(records),
            "data_summary": data_summary,
            "output_path": output_path,
            "data_path": train_data_path,
        }
        self._write_summary(summary, output_path)
        return summary

    def preview_data(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """Validate and summarize LoRA data without loading a model."""
        train_data_path = data_path or self.data_cfg.get(
            "lora_output_path", "./data/processed/lora_data.json"
        )
        records = self._load_sft_data(train_data_path)
        if not records:
            raise ValueError("No training records found in SFT data file.")
        return {
            "mode": "dry_run",
            "data_path": train_data_path,
            "data_summary": self._summarize_sft_data(records),
        }

    @staticmethod
    def _build_sft_text(item: Dict[str, Any]) -> str:
        instruction = item.get("instruction", "").strip()
        user_input = item.get("input", "").strip()
        output = item.get("output", "").strip()
        prompt = instruction if not user_input else "%s\n%s" % (instruction, user_input)
        return "%s\n%s" % (prompt, output)

    @staticmethod
    def _load_sft_data(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as file:
            obj = json.load(file)
        if not isinstance(obj, list):
            raise ValueError("SFT data must be a list of JSON objects.")
        rows = [x for x in obj if isinstance(x, dict)]
        valid_rows = []
        for row in rows:
            instruction = str(row.get("instruction", "")).strip()
            output = str(row.get("output", "")).strip()
            if instruction and output:
                valid_rows.append(row)
        return valid_rows

    @staticmethod
    def _summarize_sft_data(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt_lengths = []
        output_lengths = []
        for record in records:
            instruction = str(record.get("instruction", ""))
            user_input = str(record.get("input", ""))
            output = str(record.get("output", ""))
            prompt_lengths.append(len(instruction) + len(user_input))
            output_lengths.append(len(output))
        return {
            "samples": len(records),
            "avg_prompt_chars": sum(prompt_lengths) / len(prompt_lengths),
            "avg_output_chars": sum(output_lengths) / len(output_lengths),
            "max_prompt_chars": max(prompt_lengths),
            "max_output_chars": max(output_lengths),
        }

    @staticmethod
    def _write_summary(summary: Dict[str, Any], output_path: str) -> None:
        summary_path = Path(output_path) / "training_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
