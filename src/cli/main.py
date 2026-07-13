"""Command line interface for banking RAG QA system."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path to write the command JSON summary",
    )


def emit_json(payload: Dict[str, Any], output_path: str | None = None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if output_path:
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Banking RAG QA System CLI")
    subparsers = parser.add_subparsers(dest="command")

    validate_parser = subparsers.add_parser("validate-config", help="Validate project config")
    add_common_options(validate_parser)

    process_parser = subparsers.add_parser("process-data", help="Run data processing pipeline")
    add_common_options(process_parser)
    process_parser.add_argument("--split", default="train", help="Dataset split name")
    process_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max number of samples to process",
    )
    process_parser.add_argument(
        "--input-paths",
        nargs="+",
        default=None,
        help="Override data.input_paths with JSON/JSONL/CSV/TSV files or directories",
    )

    lora_parser = subparsers.add_parser("train-lora", help="Run LoRA training")
    add_common_options(lora_parser)
    lora_parser.add_argument(
        "--data-path",
        default=None,
        help="Override LoRA training data path (JSON)",
    )
    lora_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize LoRA data without loading a model",
    )

    dpo_parser = subparsers.add_parser("train-dpo", help="Run DPO training")
    add_common_options(dpo_parser)
    dpo_parser.add_argument(
        "--data-path",
        default=None,
        help="Override DPO pairwise data path (JSON)",
    )
    dpo_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize DPO data without loading a model",
    )

    rag_parser = subparsers.add_parser("setup-rag", help="Build local RAG vector index")
    add_common_options(rag_parser)
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
    add_common_options(query_rag_parser)
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
    add_common_options(eval_parser)
    eval_parser.add_argument(
        "--data-path",
        default=None,
        help="Override evaluation.retrieval_test_data_path",
    )
    eval_parser.add_argument("--top-k", type=int, default=None, help="Override evaluation top-k")
    eval_parser.add_argument(
        "--output",
        default=None,
        help="Write evaluation report JSON to this path",
    )
    eval_parser.add_argument(
        "--markdown-output",
        default=None,
        help="Write evaluation summary Markdown to this path",
    )

    inference_parser = subparsers.add_parser("inference", help="Run RAG retrieval/inference")
    add_common_options(inference_parser)
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
        emit_json(summary, args.json_output)
        return

    if args.command == "process-data":
        from src.data.processor import DataProcessor

        processor = DataProcessor(config_path=args.config)
        summary = processor.run(
            split=args.split,
            max_samples=args.max_samples,
            input_paths=args.input_paths,
        )
        emit_json(summary, args.json_output)
        return

    if args.command == "train-lora":
        from src.training.lora_trainer import LoRATrainer

        trainer = LoRATrainer(config_path=args.config)
        summary = (
            trainer.preview_data(data_path=args.data_path)
            if args.dry_run
            else trainer.train(data_path=args.data_path)
        )
        emit_json(summary, args.json_output)
        return

    if args.command == "train-dpo":
        from src.training.dpo_optimizer import DPOOptimizer

        optimizer = DPOOptimizer(config_path=args.config)
        summary = (
            optimizer.preview_data(pairwise_data_path=args.data_path)
            if args.dry_run
            else optimizer.train_with_preferences(pairwise_data_path=args.data_path)
        )
        emit_json(summary, args.json_output)
        return

    if args.command == "setup-rag":
        from src.rag import RAGIndexer

        indexer = RAGIndexer(config_path=args.config)
        summary = indexer.build(source_paths=args.documents, reset=args.reset)
        emit_json(summary, args.json_output)
        return

    if args.command == "query-rag":
        from src.rag import RAGRetriever

        retriever = RAGRetriever(config_path=args.config)
        payload = {"results": retriever.retrieve_as_dicts(query=args.query, top_k=args.top_k)}
        if args.status:
            payload["status"] = retriever.status()
        emit_json(payload, args.json_output)
        return

    if args.command == "evaluate-retrieval":
        from src.evaluation import RetrievalEvaluator

        evaluator = RetrievalEvaluator(config_path=args.config)
        summary = evaluator.evaluate(
            data_path=args.data_path,
            top_k=args.top_k,
            output_path=args.output,
            markdown_path=args.markdown_output,
        )
        emit_json(summary, args.json_output)
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
        emit_json(summary, args.json_output)
        return

    raise NotImplementedError(
        f"Command '{args.command}' is reserved for next milestones and not implemented yet."
    )


if __name__ == "__main__":
    main()
