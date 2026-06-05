"""Optional reranking for retrieved RAG chunks."""

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from src.rag.pipeline import RetrievalResult


class Reranker:
    """Cross-encoder style reranker wrapper."""

    def __init__(self, model_path: str, batch_size: int = 16) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "缺少 sentence-transformers。请先执行: pip install sentence-transformers"
            ) from exc

        if not model_path:
            raise ValueError("rerank_model_path 不能为空。")
        self.model_path = model_path
        self.batch_size = batch_size
        self.model = CrossEncoder(model_path)

    def rerank(
        self,
        query: str,
        results: List["RetrievalResult"],
        top_n: Optional[int] = None,
    ) -> List["RetrievalResult"]:
        if not results:
            return []
        pairs = [(query, item.chunk.text) for item in results]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        reranked: List[RetrievalResult] = []
        for item, score in zip(results, scores):
            reranked.append(RetrievalResult(chunk=item.chunk, score=float(score)))
        reranked.sort(key=lambda item: item.score, reverse=True)
        if top_n is not None:
            return reranked[:top_n]
        return reranked
