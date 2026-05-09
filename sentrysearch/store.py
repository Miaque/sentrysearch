"""ChromaDB 向量存储。"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import chromadb


DEFAULT_DB_PATH = Path.home() / ".sentrysearch" / "db"


class BackendMismatchError(RuntimeError):
    """搜索后端/模型与已索引的后端/模型不匹配时抛出。"""


def _collection_name(backend: str, model: str | None = None) -> str:
    """返回指定后端和可选模型对应的 ChromaDB 集合名称。"""
    if backend == "gemini":
        return "dashcam_chunks"
    if backend == "remote":
        if model:
            return f"dashcam_chunks_remote_{model}"
        return "dashcam_chunks_remote"
    if model:
        return f"dashcam_chunks_local_{model}"
    # 旧版：local 后端，未区分模型
    return "dashcam_chunks_local"


def detect_index(db_path: str | Path | None = None) -> tuple[str | None, str | None]:
    """返回第一个有数据的索引的 ``(backend, model)``。

    如果没有索引包含数据则返回 ``(None, None)``。
    优先检查 gemini，然后是带模型后缀的 local 集合，最后是
    旧版 ``dashcam_chunks_local`` 集合（视为 qwen8b）。
    """
    db_path = str(db_path or DEFAULT_DB_PATH)
    if not Path(db_path).exists():
        return None, None
    client = chromadb.PersistentClient(path=db_path)
    existing = {c.name for c in client.list_collections()}

    # Gemini 优先（默认 / 旧版）
    if "dashcam_chunks" in existing:
        col = client.get_collection("dashcam_chunks")
        if col.count() > 0:
            return "gemini", None

    # 带模型后缀的 local 集合 (dashcam_chunks_local_<model>)
    for name in sorted(existing):
        if name.startswith("dashcam_chunks_local_"):
            col = client.get_collection(name)
            if col.count() > 0:
                meta = col.metadata or {}
                model = meta.get("embedding_model")
                if model is None:
                    model = name.removeprefix("dashcam_chunks_local_")
                return "local", model

    # 旧版 local 集合（无模型后缀）— 视为 qwen8b
    if "dashcam_chunks_local" in existing:
        col = client.get_collection("dashcam_chunks_local")
        if col.count() > 0:
            meta = col.metadata or {}
            return "local", meta.get("embedding_model", "qwen8b")

    # Remote 集合 (dashcam_chunks_remote_<model>)
    for name in sorted(existing):
        if name.startswith("dashcam_chunks_remote_"):
            col = client.get_collection(name)
            if col.count() > 0:
                meta = col.metadata or {}
                model = meta.get("embedding_model")
                if model is None:
                    model = name.removeprefix("dashcam_chunks_remote_")
                return "remote", model

    # 旧版 remote 集合（无模型后缀）
    if "dashcam_chunks_remote" in existing:
        col = client.get_collection("dashcam_chunks_remote")
        if col.count() > 0:
            meta = col.metadata or {}
            return "remote", meta.get("embedding_model", "Qwen/Qwen3-VL-Embedding-8B")

    return None, None


def detect_backend(db_path: str | Path | None = None) -> str | None:
    """返回已有索引数据的后端名称，如果为空则返回 None。"""
    backend, _ = detect_index(db_path)
    return backend


def _make_chunk_id(source_file: str, start_time: float) -> str:
    """根据源文件路径和起始时间生成确定性分块 ID。"""
    raw = f"{source_file}:{start_time}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class SentryStore:
    """基于 ChromaDB 的持久化向量存储。"""

    def __init__(self, db_path: str | Path | None = None, backend: str = "gemini",
                 model: str | None = None):
        db_path = str(db_path or DEFAULT_DB_PATH)
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=db_path)
        self._backend = backend
        self._model = model
        # 按后端+模型分离集合，确保不兼容的向量不会混在一起。
        col_name = _collection_name(backend, model)
        metadata = {"hnsw:space": "cosine", "embedding_backend": backend}
        if model:
            metadata["embedding_model"] = model
        self._collection = self._client.get_or_create_collection(
            name=col_name,
            metadata=metadata,
        )

    @property
    def collection(self) -> chromadb.Collection:
        return self._collection

    def get_backend(self) -> str:
        """返回构建此索引所使用的后端名称。"""
        meta = self._collection.metadata or {}
        return meta.get("embedding_backend", "gemini")

    def get_model(self) -> str | None:
        """返回构建此索引所使用的模型名称，如果没有则返回 None。"""
        meta = self._collection.metadata or {}
        return meta.get("embedding_model")

    def check_backend(self, backend: str) -> None:
        """如果 *backend* 与索引不匹配则抛出 BackendMismatchError。"""
        indexed_backend = self.get_backend()
        if indexed_backend != backend:
            raise BackendMismatchError(
                f"此索引是使用 {indexed_backend} 后端构建的。"
                f"请使用 --backend {indexed_backend} 进行搜索，"
                f"或使用 --backend {backend} 重新索引。"
            )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------

    def add_chunk(
        self,
        chunk_id: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        """存储单个分块的 Embedding 和元数据。

        必需的元数据键: source_file, start_time, end_time。
        indexed_at ISO 时间戳会自动添加。
        """
        meta = {
            "source_file": metadata["source_file"],
            "start_time": float(metadata["start_time"]),
            "end_time": float(metadata["end_time"]),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        # 保留调用方提供的额外元数据
        for key in metadata:
            if key not in meta and key != "embedding":
                meta[key] = metadata[key]

        self._collection.upsert(
            ids=[chunk_id],
            embeddings=[embedding],
            metadatas=[meta],
        )

    def add_chunks(self, chunks: list[dict]) -> None:
        """批量存储分块。每个字典必须包含 'embedding' 和元数据键。"""
        now = datetime.now(timezone.utc).isoformat()
        ids = []
        embeddings = []
        metadatas = []

        for chunk in chunks:
            chunk_id = _make_chunk_id(chunk["source_file"], chunk["start_time"])
            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            metadatas.append({
                "source_file": chunk["source_file"],
                "start_time": float(chunk["start_time"]),
                "end_time": float(chunk["end_time"]),
                "indexed_at": now,
            })

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
    ) -> list[dict]:
        """返回前 N 个结果，包含距离和元数据。"""
        count = self._collection.count()
        if count == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
        )

        hits = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            hits.append({
                "source_file": meta["source_file"],
                "start_time": meta["start_time"],
                "end_time": meta["end_time"],
                "score": 1.0 - distance,  # 余弦距离 → 相似度
                "distance": distance,
            })
        return hits

    def is_indexed(self, source_file: str) -> bool:
        """检查 source_file 是否已有分块被存储。"""
        results = self._collection.get(
            where={"source_file": source_file},
            limit=1,
        )
        return len(results["ids"]) > 0

    def has_chunk(self, chunk_id: str) -> bool:
        """检查指定分块 ID 是否已存储。"""
        results = self._collection.get(ids=[chunk_id], limit=1)
        return len(results["ids"]) > 0

    def make_chunk_id(self, source_file: str, start_time: float) -> str:
        """返回此存储使用的确定性分块 ID。"""
        return _make_chunk_id(source_file, start_time)

    def remove_file(self, source_file: str) -> int:
        """删除指定源文件的所有分块。返回已删除数量。"""
        results = self._collection.get(where={"source_file": source_file})
        ids = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def get_stats(self) -> dict:
        """返回存储统计信息。"""
        total = self._collection.count()
        if total == 0:
            return {"total_chunks": 0, "unique_source_files": 0, "source_files": []}

        # 获取所有元数据（仅需要的字段）
        all_meta = self._collection.get(include=["metadatas"])
        source_files = sorted({m["source_file"] for m in all_meta["metadatas"]})
        return {
            "total_chunks": total,
            "unique_source_files": len(source_files),
            "source_files": source_files,
        }
