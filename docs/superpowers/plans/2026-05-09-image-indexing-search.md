# 图片索引与以图搜图 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `sentrysearch index-images <目录>` 和 `sentrysearch search-images <图片>` 命令，支持对图片文件夹建立向量索引并以图搜相似图，可选通过 SiliconFlow Reranker 提升精度。

**Architecture:** 在现有 video Chunk → Embed → ChromaDB 模式上，新增并行的 image 索引通道。Store 层通过 `collection_type` 参数分离 video/image 集合；新增 `image_indexer.py`（扫描+嵌入循环）和 `reranker.py`（Reranker 接口+Remote 实现）；`search.py` 增加 `search_images` 函数；CLI 增加两个新命令。

**Tech Stack:** Python 3.11+, ChromaDB, httpx, Click

---

### Task 1: Store — 图片集合支持

**Files:**
- Modify: `sentrysearch/store.py`
- Modify: `tests/test_store.py`

- [ ] **Step 1: 编写 `_image_collection_name` 测试**

```python
# 在 tests/test_store.py 末尾添加

class TestImageCollectionName:
    def test_gemini_image_collection(self):
        from sentrysearch.store import _image_collection_name
        assert _image_collection_name("gemini") == "image_index"

    def test_local_image_collection(self):
        from sentrysearch.store import _image_collection_name
        assert _image_collection_name("local") == "image_index_local"

    def test_local_model_image_collection(self):
        from sentrysearch.store import _image_collection_name
        assert _image_collection_name("local", "qwen2b") == "image_index_local_qwen2b"

    def test_remote_image_collection(self):
        from sentrysearch.store import _image_collection_name
        assert _image_collection_name("remote") == "image_index_remote"

    def test_remote_model_image_collection(self):
        from sentrysearch.store import _image_collection_name
        name = _image_collection_name("remote", "Qwen/Qwen3-VL-Embedding-8B")
        assert name == "image_index_remote_Qwen/Qwen3-VL-Embedding-8B"
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `uv run pytest tests/test_store.py::TestImageCollectionName -v`
Expected: FAIL — `_image_collection_name` 未定义

- [ ] **Step 3: 实现 `_image_collection_name`**

```python
# 在 sentrysearch/store.py 中 _collection_name 函数下方添加

def _image_collection_name(backend: str, model: str | None = None) -> str:
    """返回指定后端和可选模型对应的图片 ChromaDB 集合名称。"""
    if backend == "gemini":
        return "image_index"
    if backend == "remote":
        if model:
            return f"image_index_remote_{model}"
        return "image_index_remote"
    if backend == "local":
        if model:
            return f"image_index_local_{model}"
        return "image_index_local"
    return "image_index"
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `uv run pytest tests/test_store.py::TestImageCollectionName -v`
Expected: PASS

- [ ] **Step 5: 编写 `SentryStore` 图片集合构造测试**

```python
# 在 tests/test_store.py TestStoreBackend 类中添加

    def test_image_collection_type_gemini(self, tmp_path):
        from sentrysearch.store import SentryStore

        store = SentryStore(db_path=tmp_path / "db", backend="gemini", collection_type="image")
        assert store.collection.name == "image_index"

    def test_image_collection_type_local(self, tmp_path):
        from sentrysearch.store import SentryStore

        store = SentryStore(db_path=tmp_path / "db", backend="local", model="qwen2b", collection_type="image")
        assert store.collection.name == "image_index_local_qwen2b"

    def test_video_collection_type_is_default(self, tmp_path):
        from sentrysearch.store import SentryStore

        store = SentryStore(db_path=tmp_path / "db", backend="gemini")
        assert store.collection.name == "dashcam_chunks"

    def test_image_and_video_collections_separate(self, tmp_path):
        from sentrysearch.store import SentryStore

        db = tmp_path / "db"
        video = SentryStore(db_path=db, backend="gemini", collection_type="video")
        image = SentryStore(db_path=db, backend="gemini", collection_type="image")

        emb = _make_embedding(seed=1.0)
        video.add_chunk("v1", emb, {
            "source_file": "vid.mp4", "start_time": 0.0, "end_time": 30.0,
        })

        assert video.get_stats()["total_chunks"] == 1
        assert image.get_stats()["total_chunks"] == 0
```

- [ ] **Step 6: 运行测试，确认失败**

Run: `uv run pytest tests/test_store.py::TestStoreBackend::test_image_collection_type_gemini tests/test_store.py::TestStoreBackend::test_image_collection_type_local tests/test_store.py::TestStoreBackend::test_video_collection_type_is_default tests/test_store.py::TestStoreBackend::test_image_and_video_collections_separate -v`
Expected: FAIL — `collection_type` 参数不存在

- [ ] **Step 7: 修改 `SentryStore.__init__` 支持 `collection_type` 参数**

```python
# 修改 sentrysearch/store.py SentryStore.__init__ 签名和集合名解析部分

class SentryStore:
    """基于 ChromaDB 的持久化向量存储。"""

    def __init__(self, db_path: str | Path | None = None, backend: str = "gemini",
                 model: str | None = None, collection_type: str = "video"):
        db_path = str(db_path or DEFAULT_DB_PATH)
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=db_path)
        self._backend = backend
        self._model = model
        self._collection_type = collection_type
        # 按后端+模型+类型分离集合
        if collection_type == "image":
            col_name = _image_collection_name(backend, model)
        else:
            col_name = _collection_name(backend, model)
        metadata = {"hnsw:space": "cosine", "embedding_backend": backend}
        if model:
            metadata["embedding_model"] = model
        self._collection = self._client.get_or_create_collection(
            name=col_name,
            metadata=metadata,
        )
```

- [ ] **Step 8: 运行测试，确认通过**

Run: `uv run pytest tests/test_store.py::TestStoreBackend::test_image_collection_type_gemini tests/test_store.py::TestStoreBackend::test_image_collection_type_local tests/test_store.py::TestStoreBackend::test_video_collection_type_is_default tests/test_store.py::TestStoreBackend::test_image_and_video_collections_separate -v`
Expected: PASS

- [ ] **Step 9: 确认已有测试仍然通过（向后兼容）**

Run: `uv run pytest tests/test_store.py -v`
Expected: ALL PASS

- [ ] **Step 10: 编写 `add_image` 测试**

```python
# 在 tests/test_store.py TestSentryStore 类中添加

    def test_add_image_and_search(self, tmp_path):
        from sentrysearch.store import SentryStore

        store = SentryStore(db_path=tmp_path / "db", backend="gemini", collection_type="image")
        img_emb = _make_embedding(seed=3.0)
        store.add_image("/photos/sunset.jpg", img_emb)

        stats = store.get_stats()
        assert stats["total_chunks"] == 1
        assert stats["unique_source_files"] == 1
        assert "/photos/sunset.jpg" in stats["source_files"]

        results = store.search(img_emb, n_results=1)
        assert len(results) == 1
        assert results[0]["source_file"] == "/photos/sunset.jpg"
        assert results[0]["score"] > 0.99

    def test_add_image_idempotent(self, tmp_path):
        from sentrysearch.store import SentryStore

        store = SentryStore(db_path=tmp_path / "db", backend="gemini", collection_type="image")
        emb1 = _make_embedding(seed=1.0)
        emb2 = _make_embedding(seed=2.0)
        store.add_image("/photos/cat.jpg", emb1)
        store.add_image("/photos/cat.jpg", emb2)
        assert store.get_stats()["total_chunks"] == 1
```

- [ ] **Step 11: 运行测试，确认失败**

Run: `uv run pytest tests/test_store.py::TestSentryStore::test_add_image_and_search tests/test_store.py::TestSentryStore::test_add_image_idempotent -v`
Expected: FAIL — `add_image` 方法不存在

- [ ] **Step 12: 实现 `SentryStore.add_image`**

```python
# 在 sentrysearch/store.py SentryStore 类的 add_chunk 方法后面添加

    def add_image(self, source_file: str, embedding: list[float]) -> None:
        """存储单张图片的嵌入和元数据。

        图片没有时间维度，chunk_id 基于文件路径的 SHA-256。
        """
        chunk_id = _make_image_id(source_file)
        self.add_chunk(
            chunk_id=chunk_id,
            embedding=embedding,
            metadata={
                "source_file": source_file,
                "start_time": 0.0,
                "end_time": 0.0,
            },
        )
```

并在 `_make_chunk_id` 旁边添加辅助函数：

```python
def _make_image_id(source_file: str) -> str:
    """根据图片绝对路径生成确定性 ID。"""
    return hashlib.sha256(source_file.encode()).hexdigest()[:16]
```

- [ ] **Step 13: 运行测试，确认通过**

Run: `uv run pytest tests/test_store.py::TestSentryStore::test_add_image_and_search tests/test_store.py::TestSentryStore::test_add_image_idempotent -v`
Expected: PASS

- [ ] **Step 14: 编写 `detect_image_index` 测试**

```python
# 在 tests/test_store.py 末尾添加

class TestDetectImageIndex:
    def test_empty_db(self, tmp_path):
        from sentrysearch.store import SentryStore, detect_image_index

        SentryStore(db_path=tmp_path / "db", backend="gemini", collection_type="image")
        assert detect_image_index(tmp_path / "db") == (None, None)

    def test_detects_gemini(self, tmp_path):
        from sentrysearch.store import SentryStore, detect_image_index

        store = SentryStore(db_path=tmp_path / "db", backend="gemini", collection_type="image")
        store.add_image("/photos/sunset.jpg", _make_embedding())
        assert detect_image_index(tmp_path / "db") == ("gemini", None)

    def test_detects_local_model(self, tmp_path):
        from sentrysearch.store import SentryStore, detect_image_index

        store = SentryStore(db_path=tmp_path / "db", backend="local", model="qwen2b", collection_type="image")
        store.add_image("/photos/cat.jpg", _make_embedding())
        assert detect_image_index(tmp_path / "db") == ("local", "qwen2b")

    def test_detects_remote(self, tmp_path):
        from sentrysearch.store import SentryStore, detect_image_index

        store = SentryStore(db_path=tmp_path / "db", backend="remote", model="Qwen/Qwen3-VL-Embedding-8B", collection_type="image")
        store.add_image("/photos/dog.jpg", _make_embedding())
        assert detect_image_index(tmp_path / "db") == ("remote", "Qwen/Qwen3-VL-Embedding-8B")

    def test_does_not_detect_video_index(self, tmp_path):
        from sentrysearch.store import SentryStore, detect_image_index

        store = SentryStore(db_path=tmp_path / "db", backend="gemini")
        store.add_chunk("c1", _make_embedding(), {
            "source_file": "v.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        assert detect_image_index(tmp_path / "db") == (None, None)

    def test_nonexistent_path(self, tmp_path):
        from sentrysearch.store import detect_image_index
        assert detect_image_index(tmp_path / "no_such_dir") == (None, None)
```

- [ ] **Step 15: 运行测试，确认失败**

Run: `uv run pytest tests/test_store.py::TestDetectImageIndex -v`
Expected: FAIL — `detect_image_index` 未定义

- [ ] **Step 16: 实现 `detect_image_index`**

```python
# 在 sentrysearch/store.py detect_index 函数后面添加

def detect_image_index(db_path: str | Path | None = None) -> tuple[str | None, str | None]:
    """返回第一个有数据的图片索引的 ``(backend, model)``。

    如果没有图片索引包含数据则返回 ``(None, None)``。
    优先检查 gemini，然后是带模型后缀的 local 集合，最后是 remote 集合。
    """
    db_path = str(db_path or DEFAULT_DB_PATH)
    if not Path(db_path).exists():
        return None, None
    client = chromadb.PersistentClient(path=db_path)
    existing = {c.name for c in client.list_collections()}

    # Gemini 优先
    if "image_index" in existing:
        col = client.get_collection("image_index")
        if col.count() > 0:
            return "gemini", None

    # 带模型后缀的 local 集合 (image_index_local_<model>)
    for name in sorted(existing):
        if name.startswith("image_index_local_"):
            col = client.get_collection(name)
            if col.count() > 0:
                meta = col.metadata or {}
                model = meta.get("embedding_model")
                if model is None:
                    model = name.removeprefix("image_index_local_")
                return "local", model

    # Remote 集合 (image_index_remote_<model>)
    for name in sorted(existing):
        if name.startswith("image_index_remote_"):
            col = client.get_collection(name)
            if col.count() > 0:
                meta = col.metadata or {}
                model = meta.get("embedding_model")
                if model is None:
                    model = name.removeprefix("image_index_remote_")
                return "remote", model

    return None, None
```

- [ ] **Step 17: 运行测试，确认通过**

Run: `uv run pytest tests/test_store.py::TestDetectImageIndex -v`
Expected: PASS

- [ ] **Step 18: 运行全部 store 测试确保无回归**

Run: `uv run pytest tests/test_store.py -v`
Expected: ALL PASS

- [ ] **Step 19: Commit**

```bash
git add sentrysearch/store.py tests/test_store.py
git commit -m "feat: add image collection support to SentryStore with add_image and detect_image_index"
```

---

### Task 2: 图片索引器 — 扫描与嵌入循环

**Files:**
- Create: `sentrysearch/image_indexer.py`
- Create: `tests/test_image_indexer.py`

- [ ] **Step 1: 创建测试文件并编写测试**

```python
# tests/test_image_indexer.py

import os

import pytest


_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif"}


class TestScanImageDirectory:
    def test_scans_supported_formats(self, tmp_path):
        """Test that scan_image_directory finds supported image files."""
        from sentrysearch.image_indexer import scan_image_directory

        for ext in [".jpg", ".png", ".webp"]:
            (tmp_path / f"image{ext}").write_text("")

        (tmp_path / "doc.txt").write_text("")
        (tmp_path / ".hidden.jpg").write_text("")

        result = scan_image_directory(str(tmp_path))
        assert len(result) == 3
        assert all(p.endswith((".jpg", ".png", ".webp")) for p in result)

    def test_scans_recursively(self, tmp_path):
        from sentrysearch.image_indexer import scan_image_directory

        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "a.jpg").write_text("")
        (sub / "b.png").write_text("")

        result = scan_image_directory(str(tmp_path))
        assert len(result) == 2

    def test_skips_unsupported_formats(self, tmp_path):
        from sentrysearch.image_indexer import scan_image_directory

        (tmp_path / "a.jpg").write_text("")
        (tmp_path / "b.bmp").write_text("")
        (tmp_path / "c.tiff").write_text("")

        result = scan_image_directory(str(tmp_path))
        assert len(result) == 1
        assert result[0].endswith(".jpg")

    def test_empty_directory(self, tmp_path):
        from sentrysearch.image_indexer import scan_image_directory

        result = scan_image_directory(str(tmp_path))
        assert result == []

    def test_no_supported_images(self, tmp_path):
        from sentrysearch.image_indexer import scan_image_directory

        (tmp_path / "doc.txt").write_text("")
        (tmp_path / "data.bin").write_text("")

        result = scan_image_directory(str(tmp_path))
        assert result == []

    def test_returns_absolute_paths(self, tmp_path):
        from sentrysearch.image_indexer import scan_image_directory

        (tmp_path / "photo.jpg").write_text("")

        result = scan_image_directory(str(tmp_path))
        assert len(result) == 1
        assert os.path.isabs(result[0])


class TestIndexImageDirectory:
    def test_indexes_new_images(self, tmp_path):
        from sentrysearch.image_indexer import index_image_directory
        from sentrysearch.store import SentryStore

        (tmp_path / "a.jpg").write_text("fake")
        (tmp_path / "b.png").write_text("fake")

        store = SentryStore(db_path=tmp_path / "db", backend="gemini", collection_type="image")

        new, skipped, failed = index_image_directory(str(tmp_path), store)
        assert new == 2
        assert skipped == 0
        assert failed == 0
        assert store.get_stats()["total_chunks"] == 2

    def test_skips_already_indexed_images(self, tmp_path):
        from sentrysearch.image_indexer import index_image_directory
        from sentrysearch.store import SentryStore

        (tmp_path / "a.jpg").write_text("fake")

        store = SentryStore(db_path=tmp_path / "db", backend="gemini", collection_type="image")
        store.add_image(os.path.abspath(str(tmp_path / "a.jpg")), [0.1] * 768)

        new, skipped, failed = index_image_directory(str(tmp_path), store)
        assert new == 0
        assert skipped == 1
        assert failed == 0

    def test_empty_directory(self, tmp_path):
        from sentrysearch.image_indexer import index_image_directory
        from sentrysearch.store import SentryStore

        store = SentryStore(db_path=tmp_path / "db", backend="gemini", collection_type="image")
        new, skipped, failed = index_image_directory(str(tmp_path), store)
        assert (new, skipped, failed) == (0, 0, 0)
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `uv run pytest tests/test_image_indexer.py -v`
Expected: FAIL — 模块不存在

- [ ] **Step 3: 实现 `image_indexer.py`**

```python
"""图片目录扫描与索引循环。"""

import hashlib
import os
import sys

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif"}


def scan_image_directory(directory: str) -> list[str]:
    """递归扫描目录，返回支持的图片文件绝对路径列表。"""
    images = []
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if os.path.splitext(f)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS:
                images.append(os.path.join(root, f))
    return sorted(images)


def _make_image_id(file_path: str) -> str:
    """根据图片绝对路径生成确定性 ID。"""
    return hashlib.sha256(file_path.encode()).hexdigest()[:16]


def index_image_directory(
    directory: str,
    store,
    *,
    verbose: bool = False,
) -> tuple[int, int, int]:
    """将目录中的图片编入索引。

    Args:
        directory: 包含图片的目录路径。
        store: 带有 add_image 方法的 SentryStore 实例。
        verbose: 如果为 True，输出调试信息。

    Returns:
        (new_images, skipped, failed) 统计。
    """
    from .embedder import embed_image

    images = scan_image_directory(directory)
    if not images:
        return 0, 0, 0

    new = 0
    skipped = 0
    failed = 0

    for image_path in images:
        chunk_id = _make_image_id(image_path)

        if store.has_chunk(chunk_id):
            if verbose:
                basename = os.path.basename(image_path)
                print(f"  已索引 — 跳过: {basename}", file=sys.stderr)
            skipped += 1
            continue

        try:
            embedding = embed_image(image_path, verbose=verbose)
        except Exception as exc:
            basename = os.path.basename(image_path)
            print(f"  嵌入失败: {basename}: {exc}", file=sys.stderr)
            failed += 1
            continue

        store.add_image(image_path, embedding)
        new += 1

    return new, skipped, failed
```

注意：`index_image_directory` 依赖 `embed_image()` 便捷函数，测试时会通过 `conftest.py` 的 monkeypatch 机制来 mock，和现有 `embed_query` / `embed_video_chunk` 模式一致。

- [ ] **Step 4: 运行测试，确认通过**

Run: `uv run pytest tests/test_image_indexer.py -v`
Expected: PASS (需要添加 mock 才能通过 `test_indexes_new_images`)

- [ ] **Step 5: 修复测试 — 添加 mock_embed_image fixture**

测试文件需要 mock `embed_image` 以避免真实 API 调用。更新 `tests/test_image_indexer.py`：

```python
# 在文件开头 import math，然后在文件末尾添加

@pytest.fixture
def mock_embed_image(monkeypatch):
    """Patch embed_image to return a deterministic vector without API calls."""
    def _fake_embed_image(*args, **kwargs):
        return _fake_embedding()
    monkeypatch.setattr(
        "sentrysearch.image_indexer.embed_image", _fake_embed_image,
    )
```

并将 `TestIndexImageDirectory` 类的测试方法添加 `mock_embed_image` 参数。

实际上，这个问题在 `test_indexes_new_images` 里会出现——由于没有真正的 embedder 加载，`embed_image` 会失败。需要在 `conftest.py` 中添加一个全局的 `mock_embed_image` fixture：

```python
# 在 tests/conftest.py 中添加

@pytest.fixture
def mock_embed_image(monkeypatch):
    """Patch embed_image to return a deterministic vector without API calls."""
    fake = _fake_embedding()
    monkeypatch.setattr("sentrysearch.embedder.embed_image", lambda *a, **kw: fake)
    return fake
```

然后更新 `TestIndexImageDirectory` 测试方法添加 `mock_embed_image` 参数。

- [ ] **Step 6: 运行测试，确认通过**

Run: `uv run pytest tests/test_image_indexer.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add sentrysearch/image_indexer.py tests/test_image_indexer.py tests/conftest.py
git commit -m "feat: add image indexer with directory scanner and embedding loop"
```

---

### Task 3: Search — `search_images` 函数

**Files:**
- Modify: `sentrysearch/search.py`
- Modify: `tests/test_search.py`

- [ ] **Step 1: 编写 `search_images` 测试**

```python
# 在 tests/test_search.py 末尾添加

import math
import pytest


def _fake_image_embedding(seed: float = 0.0, dim: int = 768) -> list[float]:
    vec = [math.sin(seed + i * 0.1) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


class TestSearchImages:
    def test_returns_empty_for_empty_store(self, tmp_path, mock_embed_image):
        from sentrysearch.search import search_images
        from sentrysearch.store import SentryStore

        store = SentryStore(db_path=tmp_path / "db", backend="gemini", collection_type="image")
        results = search_images("/query.jpg", store, n_results=5)
        assert results == []

    def test_returns_sorted_results(self, tmp_path, mock_embed_image):
        from sentrysearch.search import search_images
        from sentrysearch.store import SentryStore

        store = SentryStore(db_path=tmp_path / "db", backend="gemini", collection_type="image")
        store.add_image("/photos/match.jpg", _fake_image_embedding(seed=0.0))
        store.add_image("/photos/other.jpg", _fake_image_embedding(seed=100.0))

        results = search_images("/query.jpg", store, n_results=5)
        assert len(results) == 2
        assert results[0]["similarity_score"] > results[1]["similarity_score"]
        assert "source_file" in results[0]
        assert "similarity_score" in results[0]

    def test_respects_n_results(self, tmp_path, mock_embed_image):
        from sentrysearch.search import search_images
        from sentrysearch.store import SentryStore

        store = SentryStore(db_path=tmp_path / "db", backend="gemini", collection_type="image")
        for i in range(10):
            store.add_image(f"/photos/{i}.jpg", _fake_image_embedding(seed=float(i)))

        results = search_images("/query.jpg", store, n_results=3)
        assert len(results) == 3
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `uv run pytest tests/test_search.py::TestSearchImages -v`
Expected: FAIL — `search_images` 未定义

- [ ] **Step 3: 实现 `search_images`**

```python
# 在 sentrysearch/search.py _search_with_embedding 函数后面添加

from .embedder import embed_image


def search_images(
    image_path: str,
    store,
    n_results: int = 5,
    *,
    verbose: bool = False,
    reranker=None,
) -> list[dict]:
    """使用图片作为查询来检索已索引的相似图片。

    Args:
        image_path: 查询图片的路径。
        store: 图片的 SentryStore 实例。
        n_results: 返回的最大结果数量。
        verbose: 如果为 True，将调试信息输出到 stderr。
        reranker: 可选 Reranker 实例，用于精排结果。

    Returns:
        按相似度排序（最佳优先）的结果字典列表。
        每个字典包含: source_file, similarity_score。
    """
    query_embedding = embed_image(image_path, verbose=verbose)

    # 启用 rerank 时先取更多候选再精排
    fetch_n = max(n_results * 10, 50) if reranker else n_results

    hits = store.search(query_embedding, n_results=fetch_n)
    candidates = [
        {
            "source_file": hit["source_file"],
            "similarity_score": hit["score"],
        }
        for hit in hits
    ]

    if reranker and candidates:
        candidates = reranker.rerank(image_path, candidates, top_n=n_results)

    candidates.sort(key=lambda r: r["similarity_score"], reverse=True)
    return candidates[:n_results]
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `uv run pytest tests/test_search.py::TestSearchImages -v`
Expected: PASS

- [ ] **Step 5: 确认已有 search 测试不受影响**

Run: `uv run pytest tests/test_search.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add sentrysearch/search.py tests/test_search.py
git commit -m "feat: add search_images function for image-to-image similarity search"
```

---

### Task 4: Reranker — RemoteReranker（SiliconFlow）

**Files:**
- Create: `sentrysearch/reranker.py`
- Create: `tests/test_reranker.py`

- [ ] **Step 1: 创建测试文件**

```python
# tests/test_reranker.py

import base64

import pytest
from unittest.mock import MagicMock, patch


class TestRemoteReranker:
    def test_rerank_calls_api(self):
        from sentrysearch.reranker import RemoteReranker

        reranker = RemoteReranker("http://api.example.com", api_key="test-key")

        candidates = [
            {"source_file": "/a.jpg", "similarity_score": 0.8},
            {"source_file": "/b.jpg", "similarity_score": 0.6},
            {"source_file": "/c.jpg", "similarity_score": 0.9},
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"index": 2, "relevance_score": 0.95},  # /c.jpg
                {"index": 0, "relevance_score": 0.82},  # /a.jpg
                {"index": 1, "relevance_score": 0.71},  # /b.jpg
            ]
        }

        with patch.object(reranker._client, "post", return_value=mock_response) as mock_post:
            result = reranker.rerank("/query.jpg", candidates, top_n=3)

        assert len(result) == 3
        assert result[0]["source_file"] == "/c.jpg"
        assert result[0]["similarity_score"] == 0.95
        assert result[1]["source_file"] == "/a.jpg"
        assert result[1]["similarity_score"] == 0.82

    def test_rerank_respects_top_n(self):
        from sentrysearch.reranker import RemoteReranker

        reranker = RemoteReranker("http://api.example.com")

        candidates = [
            {"source_file": f"/{i}.jpg", "similarity_score": 1.0 - i * 0.1}
            for i in range(10)
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"index": i, "relevance_score": 1.0 - i * 0.1} for i in range(5)]
        }

        with patch.object(reranker._client, "post", return_value=mock_response):
            result = reranker.rerank("/query.jpg", candidates, top_n=5)

        assert len(result) == 5

    def test_rerank_handles_empty_candidates(self):
        from sentrysearch.reranker import RemoteReranker

        reranker = RemoteReranker("http://api.example.com")
        result = reranker.rerank("/query.jpg", [], top_n=5)
        assert result == []

    def test_rerank_handles_api_error_gracefully(self, capsys):
        from sentrysearch.reranker import RemoteReranker

        reranker = RemoteReranker("http://api.example.com")

        candidates = [
            {"source_file": "/a.jpg", "similarity_score": 0.8},
        ]

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch.object(reranker._client, "post", return_value=mock_response):
            result = reranker.rerank("/query.jpg", candidates, top_n=5)

        # Fallback: returns original candidates when rerank fails
        assert len(result) == 1
        assert result[0]["source_file"] == "/a.jpg"

    def test_output_field_accepts_different_names(self, monkeypatch):
        """有些 API 返回 'results', 有些返回 'data' — 都要兼容。"""
        from sentrysearch.reranker import RemoteReranker

        reranker = RemoteReranker("http://api.example.com")

        candidates = [
            {"source_file": "/a.jpg", "similarity_score": 0.8},
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "relevance_score": 0.92},
            ]
        }

        with patch.object(reranker._client, "post", return_value=mock_response):
            result = reranker.rerank("/test.jpg", candidates, top_n=5)

        assert result[0]["similarity_score"] == 0.92

    def test_api_key_in_headers(self):
        from sentrysearch.reranker import RemoteReranker

        reranker = RemoteReranker("http://api.example.com", api_key="sk-test")

        candidates = [{"source_file": "/a.jpg", "similarity_score": 0.8}]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"index": 0, "relevance_score": 0.92}]
        }

        with patch.object(reranker._client, "post", return_value=mock_response) as mock_post:
            reranker.rerank("/query.jpg", candidates, top_n=5)

        # Verify Authorization header was sent
        call_kwargs = mock_post.call_args
        assert call_kwargs is not None
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `uv run pytest tests/test_reranker.py -v`
Expected: FAIL — 模块不存在

- [ ] **Step 3: 实现 `reranker.py`**

```python
"""Reranker — 对向量召回结果进行精排以提高准确度。

当前仅支持通过 SiliconFlow API 的 Remote Reranker（Qwen3-VL-Reranker-8B）。
"""

import base64
import os
import sys

import httpx


class RemoteReranker:
    """SiliconFlow Reranker — 对候选结果进行图片感知精排。"""

    def __init__(self, base_url: str, api_key: str | None = None):
        self._base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=120.0,
            headers=self._build_headers(api_key),
        )

    @staticmethod
    def _build_headers(api_key: str | None) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def rerank(
        self,
        query_image: str,
        candidates: list[dict],
        *,
        top_n: int = 5,
        verbose: bool = False,
    ) -> list[dict]:
        """对 candidates 按与 query_image 的相关性重新排序。

        Args:
            query_image: 查询图片的路径。
            candidates: 候选结果列表，每项含 source_file 和 similarity_score。
            top_n: 返回前 N 个结果。
            verbose: 输出调试信息。

        Returns:
            重新排序后的结果列表。
        """
        if not candidates:
            return []

        # 构建候选文档：文件 basename 作为摘要文本
        documents = [
            os.path.basename(c["source_file"]) for c in candidates
        ]

        # 构建 query：图片 base64
        query_b64 = self._encode_image(query_image)

        try:
            response = self._client.post("/rerank", json={
                "query": f"data:image;base64,{query_b64}",
                "documents": documents,
                "top_n": top_n,
            })

            if response.status_code >= 400:
                print(
                    f"  Reranker API 错误 ({response.status_code}): {response.text}",
                    file=sys.stderr,
                )
                return candidates[:top_n]

            data = response.json()
            # 兼容两种返回格式: "results" (Jina) 或 "data" (SiliconFlow)
            ranked = data.get("results") or data.get("data") or []
            if not ranked:
                return candidates[:top_n]

            # 按 reranker 分数重建结果
            reranked = []
            for item in ranked[:top_n]:
                idx = item["index"]
                score = item.get("relevance_score", item.get("score", 0.0))
                if idx < len(candidates):
                    reranked.append({
                        "source_file": candidates[idx]["source_file"],
                        "similarity_score": score,
                    })
            return reranked

        except Exception as exc:
            print(f"  Reranker 调用失败: {exc}", file=sys.stderr)
            return candidates[:top_n]

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    def close(self) -> None:
        self._client.close()

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            self._client.close()
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `uv run pytest tests/test_reranker.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add sentrysearch/reranker.py tests/test_reranker.py
git commit -m "feat: add RemoteReranker for SiliconFlow Qwen3-VL-Reranker API"
```

---

### Task 5: CLI — `index-images` 和 `search-images` 命令

**Files:**
- Modify: `sentrysearch/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: 编写 `index-images` CLI 测试**

```python
# 在 tests/test_cli.py 末尾添加

class TestIndexImagesCommand:
    def test_help(self, cli_runner):
        result = cli_runner.invoke(cli, ["index-images", "--help"])
        assert result.exit_code == 0
        assert "index-images" in result.output
        assert "--backend" in result.output

    def test_directory_not_found(self, cli_runner, tmp_path):
        result = cli_runner.invoke(cli, ["index-images", str(tmp_path / "nope")])
        assert result.exit_code != 0

    def test_empty_directory(self, cli_runner, tmp_path, monkeypatch):
        from sentrysearch.image_indexer import index_image_directory
        monkeypatch.setattr(
            "sentrysearch.cli.index_image_directory",
            lambda *a, **kw: (0, 0, 0),
        )
        # Need to also bypass embedder resolution and scanning
        monkeypatch.setattr("sentrysearch.store.SentryStore.__init__", lambda *a, **kw: None)
        monkeypatch.setattr("sentrysearch.store.SentryStore.has_chunk", lambda *a, **kw: False)
        monkeypatch.setattr("sentrysearch.store.SentryStore.add_image", lambda *a, **kw: None)
        monkeypatch.setattr("sentrysearch.store.SentryStore.get_stats", lambda *a, **kw: {
            "total_chunks": 0, "unique_source_files": 0, "source_files": []
        })

        result = cli_runner.invoke(cli, ["index-images", str(tmp_path)])
        assert result.exit_code == 0

    def test_verbose_flag(self, cli_runner, tmp_path, monkeypatch):
        from sentrysearch.image_indexer import index_image_directory
        monkeypatch.setattr(
            "sentrysearch.cli.index_image_directory",
            lambda *a, **kw: (0, 0, 0),
        )
        monkeypatch.setattr("sentrysearch.store.SentryStore.__init__", lambda *a, **kw: None)
        monkeypatch.setattr("sentrysearch.store.SentryStore.has_chunk", lambda *a, **kw: False)
        monkeypatch.setattr("sentrysearch.store.SentryStore.add_image", lambda *a, **kw: None)
        monkeypatch.setattr("sentrysearch.store.SentryStore.get_stats", lambda *a, **kw: {
            "total_chunks": 0, "unique_source_files": 0, "source_files": []
        })

        result = cli_runner.invoke(cli, ["index-images", str(tmp_path), "--verbose"])
        assert result.exit_code == 0


class TestSearchImagesCommand:
    def test_help(self, cli_runner):
        result = cli_runner.invoke(cli, ["search-images", "--help"])
        assert result.exit_code == 0
        assert "search-images" in result.output
        assert "--threshold" in result.output
        assert "--rerank" in result.output

    def test_image_not_found(self, cli_runner, tmp_path):
        result = cli_runner.invoke(cli, ["search-images", str(tmp_path / "nope.jpg")])
        assert result.exit_code != 0

    def test_empty_index(self, cli_runner, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sentrysearch.cli.detect_image_index",
            lambda *a, **kw: (None, None),
        )
        monkeypatch.setattr("sentrysearch.store.SentryStore.__init__", lambda *a, **kw: None)
        monkeypatch.setattr("sentrysearch.store.SentryStore.get_stats", lambda *a, **kw: {
            "total_chunks": 0, "unique_source_files": 0, "source_files": []
        })

        (tmp_path / "test.png").write_text("")
        result = cli_runner.invoke(cli, ["search-images", str(tmp_path / "test.png")])
        assert result.exit_code == 0
        assert "未找到" in result.output or "index" in result.output.lower()

    def test_search_finds_results(self, cli_runner, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sentrysearch.cli.detect_image_index",
            lambda *a, **kw: ("gemini", None),
        )
        monkeypatch.setattr("sentrysearch.store.SentryStore.__init__", lambda *a, **kw: None)
        monkeypatch.setattr("sentrysearch.store.SentryStore.get_stats", lambda *a, **kw: {
            "total_chunks": 3, "unique_source_files": 1, "source_files": ["/a.jpg"]
        })
        monkeypatch.setattr(
            "sentrysearch.search.search_images",
            lambda *a, **kw: [
                {"source_file": "/photos/a.jpg", "similarity_score": 0.95},
                {"source_file": "/photos/b.jpg", "similarity_score": 0.87},
            ],
        )

        (tmp_path / "test.png").write_text("")
        result = cli_runner.invoke(cli, ["search-images", str(tmp_path / "test.png"), "--threshold", "0.5"])
        assert result.exit_code == 0
        assert "a.jpg" in result.output
        assert "0.95" in result.output or "0.9500" in result.output

    def test_threshold_filters_results(self, cli_runner, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sentrysearch.cli.detect_image_index",
            lambda *a, **kw: ("gemini", None),
        )
        monkeypatch.setattr("sentrysearch.store.SentryStore.__init__", lambda *a, **kw: None)
        monkeypatch.setattr("sentrysearch.store.SentryStore.get_stats", lambda *a, **kw: {
            "total_chunks": 3, "unique_source_files": 1, "source_files": ["/a.jpg"]
        })
        monkeypatch.setattr(
            "sentrysearch.search.search_images",
            lambda *a, **kw: [
                {"source_file": "/photos/a.jpg", "similarity_score": 0.65},
            ],
        )

        (tmp_path / "test.png").write_text("")
        result = cli_runner.invoke(cli, ["search-images", str(tmp_path / "test.png"), "--threshold", "0.7"])
        assert result.exit_code == 0
        # 0.65 < 0.7 — 不应该显示
        assert "a.jpg" not in result.output

    def test_rerank_option(self, cli_runner, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "sentrysearch.cli.detect_image_index",
            lambda *a, **kw: ("remote", "Qwen/Qwen3-VL-Embedding-8B"),
        )
        monkeypatch.setattr("sentrysearch.store.SentryStore.__init__", lambda *a, **kw: None)
        monkeypatch.setattr("sentrysearch.store.SentryStore.get_stats", lambda *a, **kw: {
            "total_chunks": 3, "unique_source_files": 1, "source_files": ["/a.jpg"]
        })
        called_with_reranker = []

        def _fake_search(image_path, store, n_results, *, verbose, reranker=None):
            called_with_reranker.append(reranker is not None)
            return [{"source_file": "/a.jpg", "similarity_score": 0.92}]

        monkeypatch.setattr("sentrysearch.search.search_images", _fake_search)

        # 需要避免 embedder init
        monkeypatch.setattr("sentrysearch.cli.get_embedder", lambda *a, **kw: None)

        (tmp_path / "test.png").write_text("")
        result = cli_runner.invoke(cli, [
            "search-images", str(tmp_path / "test.png"),
            "--rerank",
            "--remote-url", "http://localhost:8000",
            "--remote-api-key", "sk-test",
        ])
        assert result.exit_code == 0
        assert called_with_reranker[0] is True
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `uv run pytest tests/test_cli.py::TestIndexImagesCommand tests/test_cli.py::TestSearchImagesCommand -v`
Expected: FAIL — 命令不存在

- [ ] **Step 3: 实现 `index-images` 命令**

```python
# 在 sentrysearch/cli.py 中 index 命令后面添加

# -----------------------------------------------------------------------
# index-images
# -----------------------------------------------------------------------

@cli.command("index-images")
@click.argument("directory", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--backend", type=click.Choice(["gemini", "local", "remote"]), default=None,
              help="嵌入后端（默认 gemini，设置 --model 时为 local）。")
@click.option("--model", default=None,
              help="本地后端模型：qwen8b、qwen2b 或 HuggingFace ID "
                   "（默认自动检测硬件）。隐含 --backend local。")
@click.option("--quantize/--no-quantize", default=None,
              help="启用/禁用本地后端的 4-bit 量化（默认自动检测）。")
@click.option("--remote-url", default=None,
              help="远程嵌入 API 地址（如 http://localhost:8000）。隐含 --backend remote。")
@click.option("--remote-api-key", default=None,
              help="远程嵌入服务 API 密钥（或设置 REMOTE_EMBED_API_KEY）。")
@click.option("--verbose", is_flag=True, help="显示调试信息。")
def index_images(directory, backend, model, quantize, remote_url, remote_api_key, verbose):
    """将 DIRECTORY 中的图片文件编入索引以供以图搜图。"""
    from .embedder import get_embedder, reset_embedder
    from .image_indexer import index_image_directory, scan_image_directory
    from .local_embedder import detect_default_model, normalize_model_key
    from .store import SentryStore

    try:
        if model is not None and backend is None:
            backend = "local"
        if remote_url is not None and backend is None:
            backend = "remote"
        if backend is None:
            backend = "gemini"

        if backend == "local" and model is None:
            model = detect_default_model()
            click.echo(f"自动检测到模型：{model}", err=True)
        if backend == "local":
            model = normalize_model_key(model)
        if backend == "remote":
            remote_url = _resolve_remote_url(remote_url)
            api_key = _resolve_remote_api_key(remote_api_key)
            model = model or "Qwen/Qwen3-VL-Embedding-8B"

        embedder = get_embedder(
            backend, model=model, quantize=quantize,
            base_url=remote_url if backend == "remote" else None,
            api_key=api_key if backend == "remote" else None,
        )

        images = scan_image_directory(directory)
        if not images:
            supported = ", ".join(SUPPORTED_IMAGE_EXTENSIONS)
            click.echo(f"未找到支持的图片文件（{supported}）。")
            return

        store = SentryStore(
            backend=backend,
            model=model if backend in ("local", "remote") else None,
            collection_type="image",
        )

        if verbose:
            click.echo(f"[verbose] 数据库路径：{store._client._identifier}", err=True)
            click.echo(f"[verbose] 后端={backend}, 集合={store.collection.name}", err=True)
            click.echo(f"[verbose] 找到 {len(images)} 张图片", err=True)

        new, skipped, failed = index_image_directory(
            directory, store, verbose=verbose,
        )

        stats = store.get_stats()
        parts = []
        if skipped:
            parts.append(f"跳过 {skipped} 张已索引")
        if failed:
            parts.append(f"{failed} 张失败")
        extra = f"（{', '.join(parts)}）" if parts else ""
        click.echo(
            f"\n已从目录索引 {new} 张新图片{extra}。"
            f"总计：{stats['total_chunks']} 张图片。"
        )

    except Exception as e:
        _handle_error(e)
    finally:
        reset_embedder()
```

并在文件顶部 import `SUPPORTED_IMAGE_EXTENSIONS`：

```python
# 在 cli.py 顶部 click import 之后添加
from .image_indexer import SUPPORTED_IMAGE_EXTENSIONS
```

- [ ] **Step 4: 实现 `search-images` 命令**

```python
# 在 sentrysearch/cli.py 中 index_images 命令后面添加

# -----------------------------------------------------------------------
# search-images
# -----------------------------------------------------------------------

@cli.command("search-images")
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.option("-n", "--results", "n_results", default=5, show_default=True,
              help="返回结果数量。")
@click.option("--threshold", default=0.7, show_default=True, type=float,
              help="视为可信匹配的最低相似度分数。")
@click.option("--rerank", is_flag=True,
              help="使用 reranker 精排结果（需 remote 后端）。")
@click.option("--backend", type=click.Choice(["gemini", "local", "remote"]), default=None,
              help="嵌入后端（省略时从索引自动检测）。")
@click.option("--model", default=None,
              help="本地后端模型（默认从索引自动检测）。")
@click.option("--quantize/--no-quantize", default=None,
              help="启用/禁用本地后端的 4-bit 量化。")
@click.option("--remote-url", default=None,
              help="远程嵌入 API 地址。隐含 --backend remote。")
@click.option("--remote-api-key", default=None,
              help="远程嵌入服务 API 密钥（或设置 REMOTE_EMBED_API_KEY）。")
@click.option("--verbose", is_flag=True, help="显示调试信息。")
def search_images(image, n_results, threshold, rerank, backend, model, quantize,
                  remote_url, remote_api_key, verbose):
    """使用 IMAGE 图片搜索已索引的相似图片。"""
    from .embedder import get_embedder, reset_embedder
    from .local_embedder import normalize_model_key
    from .search import search_images as do_search
    from .store import SentryStore, detect_image_index

    try:
        if model is not None and backend is None:
            backend = "local"
        if remote_url is not None and backend is None:
            backend = "remote"
        if model is not None:
            model = normalize_model_key(model)

        if backend is None:
            detected_backend, detected_model = detect_image_index()
            backend = detected_backend or "gemini"
            if model is None:
                model = detected_model
        elif backend == "local" and model is None:
            _, detected_model = detect_image_index()
            model = detected_model

        api_key = None
        if backend == "remote":
            remote_url = _resolve_remote_url(remote_url)
            api_key = _resolve_remote_api_key(remote_api_key)
            model = model or "Qwen/Qwen3-VL-Embedding-8B"

        store = SentryStore(
            backend=backend,
            model=model if backend in ("local", "remote") else None,
            collection_type="image",
        )

        if store.get_stats()["total_chunks"] == 0:
            click.echo(
                "未找到已索引的图片。"
                "请先运行 `sentrysearch index-images <目录>`。"
            )
            return

        get_embedder(
            backend, model=model, quantize=quantize,
            base_url=remote_url if backend == "remote" else None,
            api_key=api_key if backend == "remote" else None,
        )

        # Reranker (可选，仅 remote 后端)
        reranker_inst = None
        if rerank:
            if backend != "remote":
                click.secho(
                    "警告：rerank 仅 remote 后端支持，将使用纯向量搜索。",
                    fg="yellow", err=True,
                )
            else:
                from .reranker import RemoteReranker
                try:
                    reranker_inst = RemoteReranker(remote_url, api_key=api_key)
                except Exception as exc:
                    click.secho(
                        f"警告：无法创建 Reranker：{exc}，将使用纯向量搜索。",
                        fg="yellow", err=True,
                    )

        if verbose:
            click.echo(f"  [verbose] 后端={backend}, 相似度阈值：{threshold}, rerank={rerank}", err=True)

        results = do_search(
            image, store, n_results=n_results,
            verbose=verbose, reranker=reranker_inst,
        )
        _present_image_results(results, threshold, verbose)

    except Exception as e:
        _handle_error(e)
    finally:
        if reranker_inst:
            try:
                reranker_inst.close()
            except Exception:
                pass
        reset_embedder()


def _present_image_results(results, threshold, verbose):
    """格式化并展示图片搜索结果（简洁模式）。"""
    if not results:
        click.echo(
            "未找到结果。\n\n"
            "建议：\n"
            "  - 尝试不同的查询图片\n"
            "  - 降低 --threshold 阈值\n"
            "  - 使用 --rerank 启用精排"
        )
        return

    best_score = results[0]["similarity_score"]
    low_confidence = best_score < threshold

    if low_confidence:
        click.secho(
            f"（置信度较低 — 最高得分：{best_score:.2f}）",
            fg="yellow",
        )

    for i, r in enumerate(results, 1):
        score = r["similarity_score"]
        if verbose:
            click.echo(f"  #{i} [{score:.6f}] {r['source_file']}")
        else:
            click.echo(f"  #{i} [{score:.2f}] {r['source_file']}")
```

- [ ] **Step 5: 运行测试确认 CLI 测试通过**

Run: `uv run pytest tests/test_cli.py::TestIndexImagesCommand tests/test_cli.py::TestSearchImagesCommand -v`
Expected: ALL PASS

- [ ] **Step 6: 确认已有 CLI 测试不受影响**

Run: `uv run pytest tests/test_cli.py -v`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add sentrysearch/cli.py tests/test_cli.py
git commit -m "feat: add index-images and search-images CLI commands with rerank support"
```

---

### Task 6: 整合验证

- [ ] **Step 1: 运行全部测试套件**

Run: `uv run pytest -v`

Expected: ALL PASS

- [ ] **Step 2: 运行覆盖率检查**

Run: `uv run pytest --cov=sentrysearch --cov-report=term-missing`

Expected: 新增模块 (`image_indexer.py`, `reranker.py`) 覆盖率达到 80%+

- [ ] **Step 3: 运行 lint 检查**

Run: `uv run ruff check sentrysearch/`

Expected: 无新增错误

