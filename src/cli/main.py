"""Command line interface for banking RAG QA system."""

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Banking RAG QA System CLI")
    subparsers = parser.add_subparsers(dest="command")

    process_parser = subparsers.add_parser("process-data", help="Run data processing pipeline")
    process_parser.add_argument("--config", default="config.yaml", help="Path to config file")
    process_parser.add_argument("--split", default="train", help="Dataset split name")
    process_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max number of samples to process",
    )

    lora_parser = subparsers.add_parser("train-lora", help="Run LoRA training")
    lora_parser.add_argument("--config", default="config.yaml", help="Path to config file")
    lora_parser.add_argument(
        "--data-path",
        default=None,
        help="Override LoRA training data path (JSON)",
    )

    dpo_parser = subparsers.add_parser("train-dpo", help="Run DPO training")
    dpo_parser.add_argument("--config", default="config.yaml", help="Path to config file")
    dpo_parser.add_argument(
        "--data-path",
        default=None,
        help="Override DPO pairwise data path (JSON)",
    )

    subparsers.add_parser("setup-rag", help="Placeholder for RAG setup")
    subparsers.add_parser("inference", help="Placeholder for inference")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "process-data":
        from src.data.processor import DataProcessor

        processor = DataProcessor(config_path=args.config)
        summary = processor.run(split=args.split, max_samples=args.max_samples)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "train-lora":
        from src.training.lora_trainer import LoRATrainer

        trainer = LoRATrainer(config_path=args.config)
        summary = trainer.train(data_path=args.data_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "train-dpo":
        from src.training.dpo_optimizer import DPOOptimizer

        optimizer = DPOOptimizer(config_path=args.config)
        summary = optimizer.train_with_preferences(pairwise_data_path=args.data_path)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    raise NotImplementedError(
        f"Command '{args.command}' is reserved for next milestones and not implemented yet."
    )


if __name__ == "__main__":
    main()
