"""Minimal RAG indexing and retrieval pipeline."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.config_manager import ConfigManager
from src.logger import Logger


@dataclass
class Document:
    """A source document before chunking."""

    text: str
    source: str
    metadata: Dict[str, Any]


@dataclass
class Chunk:
    """A retrievable text chunk."""

    id: str
    text: str
    source: str
    metadata: Dict[str, Any]


@dataclass
class RetrievalResult:
    """A ranked retrieval result."""

    chunk: Chunk
    score: float


class DocumentLoader:
    """Load local text, JSON and JSONL files into documents."""

    TEXT_EXTENSIONS = {".txt", ".md"}
    JSON_EXTENSIONS = {".json", ".jsonl"}

    def load_many(self, paths: Sequence[str]) -> List[Document]:
        documents: List[Document] = []
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            if path.is_dir():
                documents.extend(self._load_dir(path))
            elif path.is_file():
                documents.extend(self._load_file(path))
            else:
                raise FileNotFoundError(f"知识源路径不存在: {path}")
        return [doc for doc in documents if doc.text.strip()]

    def _load_dir(self, path: Path) -> List[Document]:
        documents: List[Document] = []
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in (
                self.TEXT_EXTENSIONS | self.JSON_EXTENSIONS
            ):
                documents.extend(self._load_file(file_path))
        return documents

    def _load_file(self, path: Path) -> List[Document]:
        suffix = path.suffix.lower()
        if suffix in self.TEXT_EXTENSIONS:
            return [
                Document(
                    text=path.read_text(encoding="utf-8"),
                    source=str(path),
                    metadata={"file_name": path.name},
                )
            ]
        if suffix == ".jsonl":
            return self._load_jsonl(path)
        if suffix == ".json":
            return self._load_json(path)
        raise ValueError(f"暂不支持的知识源文件类型: {path}")

    def _load_jsonl(self, path: Path) -> List[Document]:
        documents: List[Document] = []
        with path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = self._extract_text(obj)
                documents.append(
                    Document(
                        text=text,
                        source=str(path),
                        metadata={"file_name": path.name, "line": line_no},
                    )
                )
        return documents

    def _load_json(self, path: Path) -> List[Document]:
        obj = json.loads(path.read_text(encoding="utf-8"))
        rows = obj if isinstance(obj, list) else [obj]
        documents: List[Document] = []
        for index, row in enumerate(rows):
            text = self._extract_text(row)
            documents.append(
                Document(
                    text=text,
                    source=str(path),
                    metadata={"file_name": path.name, "record_index": index},
                )
            )
        return documents

    @staticmethod
    def _extract_text(obj: Any) -> str:
        if isinstance(obj, str):
            return obj.strip()
        if not isinstance(obj, dict):
            return str(obj).strip()

        preferred_fields = [
            "text",
            "content",
            "document",
            "answer",
            "output",
            "chosen",
            "question",
            "prompt",
            "instruction",
            "input",
        ]
        parts: List[str] = []
        for field in preferred_fields:
            value = obj.get(field)
            if value is not None and str(value).strip():
                parts.append(f"{field}: {str(value).strip()}")
        if parts:
            return "\n".join(parts)
        return json.dumps(obj, ensure_ascii=False)


class TextChunker:
    """Split documents into overlapping character chunks."""

    def __init__(self, max_chars: int = 1200, overlap_chars: int = 120) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if overlap_chars < 0:
            raise ValueError("overlap_chars must be non-negative")
        if overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be smaller than max_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def split(self, documents: Sequence[Document]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc_index, document in enumerate(documents):
            text = self._normalize_text(document.text)
            start = 0
            chunk_index = 0
            while start < len(text):
                end = min(start + self.max_chars, len(text))
                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(
                        Chunk(
                            id=f"doc{doc_index}-chunk{chunk_index}",
                            text=chunk_text,
                            source=document.source,
                            metadata={
                                **document.metadata,
                                "doc_index": doc_index,
                                "chunk_index": chunk_index,
                            },
                        )
                    )
                if end == len(text):
                    break
                start = max(0, end - self.overlap_chars)
                chunk_index += 1
        return chunks

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "\n".join(line.strip() for line in text.splitlines() if line.strip())


class EmbeddingModel:
    """SentenceTransformer wrapper with normalized embeddings."""

    def __init__(self, model_path: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "缺少 sentence-transformers。请先执行: pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(model_path)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")


class RAGIndexer:
    """Build and persist a FAISS index from local documents."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = ConfigManager(config_path).get_all()
        self.logger = Logger.get_logger("RAGIndexer", self.config.get("logging")).logger
        self.rag_cfg = self.config.get("rag", {})
        self.models_cfg = self.config.get("models", {})

    def build(
        self,
        source_paths: Optional[Sequence[str]] = None,
        reset: bool = False,
    ) -> Dict[str, Any]:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("缺少 faiss。请先执行: pip install faiss-cpu") from exc

        paths = list(source_paths or self.rag_cfg.get("source_paths", []))
        if not paths:
            raise ValueError("没有配置知识源路径，请设置 rag.source_paths 或传入 --documents。")

        vector_db_path = Path(self.rag_cfg.get("vector_db_path", "./data/vector_db"))
        index_path = vector_db_path / self.rag_cfg.get("index_file", "index.faiss")
        chunks_path = vector_db_path / self.rag_cfg.get("chunks_file", "chunks.json")
        if reset and vector_db_path.exists():
            for child in vector_db_path.iterdir():
                if child.is_file():
                    child.unlink()
        vector_db_path.mkdir(parents=True, exist_ok=True)

        documents = DocumentLoader().load_many(paths)
        chunker = TextChunker(
            max_chars=int(self.rag_cfg.get("chunk_max_chars", 1200)),
            overlap_chars=int(self.rag_cfg.get("chunk_overlap_chars", 120)),
        )
        chunks = chunker.split(documents)
        if not chunks:
            raise ValueError("没有可索引的文本块。")

        embedding_model = EmbeddingModel(self.models_cfg.get("embedding_model_path", "BAAI/bge-m3"))
        vectors = embedding_model.encode([chunk.text for chunk in chunks])

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(index_path))
        chunks_path.write_text(
            json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.logger.info("RAG index saved to %s with %d chunks.", index_path, len(chunks))
        return {
            "documents": len(documents),
            "chunks": len(chunks),
            "index_path": str(index_path),
            "chunks_path": str(chunks_path),
        }


class RAGRetriever:
    """Load a FAISS index and retrieve relevant chunks."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.config = ConfigManager(config_path).get_all()
        self.logger = Logger.get_logger("RAGRetriever", self.config.get("logging")).logger
        self.rag_cfg = self.config.get("rag", {})
        self.models_cfg = self.config.get("models", {})
        self._index = None
        self._chunks: List[Chunk] = []
        self._embedding_model: Optional[EmbeddingModel] = None

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        if not query.strip():
            raise ValueError("query 不能为空。")
        self._ensure_loaded()

        k = int(top_k or self.rag_cfg.get("retrieval_top_k", 10))
        k = min(k, len(self._chunks))
        query_vector = self._embedding_model.encode([query])
        scores, indices = self._index.search(query_vector, k)
        results: List[RetrievalResult] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append(RetrievalResult(chunk=self._chunks[int(index)], score=float(score)))
        return results

    def _ensure_loaded(self) -> None:
        if self._index is not None:
            return
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("缺少 faiss。请先执行: pip install faiss-cpu") from exc

        vector_db_path = Path(self.rag_cfg.get("vector_db_path", "./data/vector_db"))
        index_path = vector_db_path / self.rag_cfg.get("index_file", "index.faiss")
        chunks_path = vector_db_path / self.rag_cfg.get("chunks_file", "chunks.json")
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"未找到 RAG 索引，请先运行 setup-rag。缺失: {index_path} 或 {chunks_path}"
            )

        self._index = faiss.read_index(str(index_path))
        raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self._chunks = [Chunk(**item) for item in raw_chunks]
        self._embedding_model = EmbeddingModel(
            self.models_cfg.get("embedding_model_path", "BAAI/bge-m3")
        )
