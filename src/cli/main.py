"""Command line interface for banking RAG QA system."""

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Banking RAG QA System CLI")
    subparsers = parser.add_subparsers(dest="command")

    validate_parser = subparsers.add_parser("validate-config", help="Validate project config")
    validate_parser.add_argument("--config", default="config.yaml", help="Path to config file")

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

    rag_parser = subparsers.add_parser("setup-rag", help="Build local RAG vector index")
    rag_parser.add_argument("--config", default="config.yaml", help="Path to config file")
    rag_parser.add_argument(
        "--documents",
        nargs="+",
        default=None,
        help="Override knowledge source paths configured in rag.source_paths",
    )
    rag_parser.add_argument(
        "--reset",
        action="store_true",
        help="Remove existing vector index files before rebuilding",
    )

    query_rag_parser = subparsers.add_parser("query-rag", help="Query the local RAG index")
    query_rag_parser.add_argument("--config", default="config.yaml", help="Path to config file")
    query_rag_parser.add_argument("--query", required=True, help="Search query")
    query_rag_parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k")
    query_rag_parser.add_argument(
        "--status",
        action="store_true",
        help="Print index status before running retrieval",
    )

    eval_parser = subparsers.add_parser(
        "evaluate-retrieval",
        help="Evaluate RAG retrieval quality",
    )
    eval_parser.add_argument("--config", default="config.yaml", help="Path to config file")
    eval_parser.add_argument(
        "--data-path",
        default=None,
        help="Override evaluation.retrieval_test_data_path",
    )
    eval_parser.add_argument("--top-k", type=int, default=None, help="Override evaluation top-k")

    inference_parser = subparsers.add_parser("inference", help="Run RAG retrieval/inference")
    inference_parser.add_argument("--config", default="config.yaml", help="Path to config file")
    inference_parser.add_argument("--query", required=True, help="User question")
    inference_parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k")
    inference_parser.add_argument(
        "--generate",
        action="store_true",
        default=None,
        help="Use the configured local generation model after retrieval",
    )
    inference_parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Do not include the assembled prompt in JSON output",
    )
    inference_parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Do not include retrieved source chunks in JSON output",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "validate-config":
        from src.validation import ConfigValidator

        summary = ConfigValidator(config_path=args.config).validate()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
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

    if args.command == "setup-rag":
        from src.rag import RAGIndexer

        indexer = RAGIndexer(config_path=args.config)
        summary = indexer.build(source_paths=args.documents, reset=args.reset)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "query-rag":
        from src.rag import RAGRetriever

        retriever = RAGRetriever(config_path=args.config)
        payload = {"results": retriever.retrieve_as_dicts(query=args.query, top_k=args.top_k)}
        if args.status:
            payload["status"] = retriever.status()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate-retrieval":
        from src.evaluation import RetrievalEvaluator

        evaluator = RetrievalEvaluator(config_path=args.config)
        summary = evaluator.evaluate(data_path=args.data_path, top_k=args.top_k)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "inference":
        from src.inference import InferenceEngine

        engine = InferenceEngine(config_path=args.config)
        summary = engine.answer(
            query=args.query,
            top_k=args.top_k,
            generate=args.generate,
            include_prompt=False if args.no_prompt else None,
            include_sources=False if args.no_sources else None,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    raise NotImplementedError(
        f"Command '{args.command}' is reserved for next milestones and not implemented yet."
    )


if __name__ == "__main__":
    main()
