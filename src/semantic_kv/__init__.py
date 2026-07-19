"""semantic-kv: semantic-diff KV cache for RAG."""

from .cache import Assembled, SemanticKVCache
from .chunking import Chunk, text_hash
from .embeddings import HashingEmbedder
from .kv_store import KVSegment, KVStore
from .metrics import AssemblyStats
from .pipeline import CachedRAGPipeline, GenerationResult
from .vectorstore import InMemoryVectorStore, Point, QdrantVectorStore

__all__ = [
    "Assembled",
    "AssemblyStats",
    "CachedRAGPipeline",
    "Chunk",
    "GenerationResult",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "KVSegment",
    "KVStore",
    "Point",
    "QdrantVectorStore",
    "SemanticKVCache",
    "text_hash",
]

__version__ = "0.1.0"
