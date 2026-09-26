"""
storage/vector_store.py — Local ChromaDB Vector Store for Security Knowledge Retrieval.

Provides a persistent, local vector database wrapper using ChromaDB and
sentence-transformers ('all-MiniLM-L6-v2') embeddings.

Capabilities:
  • Adding & indexing security documents
  • Local sentence-transformers embedding generation
  • Semantic search & top-k context retrieval
  • Automatic initialization & preloading of standard vulnerability dataset
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import settings
from sample_data.security_knowledge import SECURITY_KNOWLEDGE_DOCUMENTS

logger = logging.getLogger(__name__)

# Lazy import helpers
_CHROMADB_AVAILABLE = False
_SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None


@dataclass
class RetrievedDocument:
    """Represents a retrieved security knowledge document with relevance score."""
    doc_id: str
    title: str
    content: str
    cwe: str
    severity: str
    category: str
    distance: float
    relevance_score: float  # Normalized 0.0 to 1.0 (higher = more relevant)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecurityVectorStore:
    """
    Local ChromaDB vector store backed by sentence-transformers embeddings.
    """

    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: str = "security_knowledge",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        auto_preload: bool = True,
    ):
        """
        Initialize the local vector store.

        Args:
            persist_directory: Path to ChromaDB storage directory. Defaults to settings.chroma_persist_dir.
            collection_name: Name of the ChromaDB collection.
            embedding_model_name: HuggingFace model tag for sentence-transformers.
            auto_preload: Whether to automatically seed with curated security knowledge if empty.
        """
        self.persist_dir = Path(persist_directory or settings.chroma_persist_dir)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        self._embedder: Optional[Any] = None
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None

        # Ensure directory exists
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        if auto_preload:
            self.ensure_initialized()

    def _get_embedder(self) -> Any:
        """Lazy-load sentence-transformers model."""
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is not installed. Run `pip install sentence-transformers`."
            )
        if self._embedder is None:
            logger.info(f"Loading embedding model '{self.embedding_model_name}'...")
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _get_collection(self) -> Any:
        """Lazy-initialize ChromaDB persistent client and collection."""
        if not _CHROMADB_AVAILABLE:
            raise RuntimeError(
                "chromadb is not installed. Run `pip install chromadb`."
            )
        if self._collection is None:
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Compute normalized vector embeddings for a list of strings."""
        embedder = self._get_embedder()
        embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def add_document(
        self,
        doc_id: str,
        content: str,
        title: str,
        cwe: str = "",
        severity: str = "MEDIUM",
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single security document into the vector store."""
        self.add_documents([
            {
                "id": doc_id,
                "content": content,
                "title": title,
                "cwe": cwe,
                "severity": severity,
                "category": category,
                **(metadata or {}),
            }
        ])

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Batch-embed and upsert documents into ChromaDB.

        Returns:
            Number of documents added/updated.
        """
        if not documents:
            return 0

        collection = self._get_collection()

        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for doc in documents:
            doc_id = str(doc.get("id"))
            content = doc.get("content", "")
            title = doc.get("title", "")
            
            # Combine title + content + category for richer embedding representation
            text_to_embed = f"{title}\nCategory: {doc.get('category', '')}\nCWE: {doc.get('cwe', '')}\n{content}"

            ids.append(doc_id)
            texts.append(content)
            metadatas.append({
                "title": title,
                "cwe": doc.get("cwe", ""),
                "severity": doc.get("severity", "MEDIUM"),
                "category": doc.get("category", "general"),
                "indexed_text": text_to_embed[:1000],
            })

        # Generate embeddings locally
        embeddings = self.embed_texts(texts)

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(f"Indexed {len(documents)} document(s) in collection '{self.collection_name}'.")
        return len(documents)

    def search(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None,
    ) -> List[RetrievedDocument]:
        """
        Perform semantic similarity search for a code snippet or vulnerability description.

        Args:
            query: Query text or code snippet to search for.
            top_k: Number of relevant context documents to return.
            category: Optional category filter.

        Returns:
            List of RetrievedDocument objects sorted by relevance.
        """
        if not query or not query.strip():
            return []

        collection = self._get_collection()

        query_embedding = self.embed_texts([query])[0]

        where_filter = {"category": category} if category else None

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[RetrievedDocument] = []

        if not results or not results.get("ids") or not results["ids"][0]:
            return retrieved

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
            # In cosine space in ChromaDB: distance = 1 - cosine_similarity
            # Relevance score bounded between 0.0 and 1.0
            relevance = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

            retrieved.append(
                RetrievedDocument(
                    doc_id=doc_id,
                    title=meta.get("title", ""),
                    content=doc_text,
                    cwe=meta.get("cwe", ""),
                    severity=meta.get("severity", "MEDIUM"),
                    category=meta.get("category", "general"),
                    distance=dist,
                    relevance_score=round(relevance, 4),
                    metadata=meta,
                )
            )

        return retrieved

    def ensure_initialized(self) -> int:
        """
        Preload curated security knowledge if collection is currently empty.

        Returns:
            Current count of indexed security documents.
        """
        try:
            collection = self._get_collection()
            count = collection.count()
            if count == 0:
                logger.info("Initializing vector store with curated security knowledge base...")
                return self.add_documents(SECURITY_KNOWLEDGE_DOCUMENTS)
            return count
        except Exception as e:
            logger.warning(f"Could not auto-initialize vector store: {e}")
            return 0

    def count(self) -> int:
        """Return total number of indexed documents in the collection."""
        try:
            collection = self._get_collection()
            return collection.count()
        except Exception:
            return 0
