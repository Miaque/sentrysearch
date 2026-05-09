"""远程 Embedding 后端，通过 OpenAI 兼容的 REST API 调用。

通过 /v1/embeddings 端点调用远程 Qwen3-VL-Embedding 服务。
"""

import base64
import mimetypes
import os
import sys
import time
from pathlib import Path

import httpx

from .base_embedder import BaseEmbedder

DEFAULT_MODEL = "Qwen/Qwen3-VL-Embedding-8B"
DEFAULT_DIMENSIONS = 4096
DEFAULT_RPM = 55
DEFAULT_TIMEOUT = 120.0


class _RateLimiter:
    """基于请求时间戳的简单滑动窗口速率限制器。"""

    def __init__(self, max_per_minute: int = DEFAULT_RPM):
        from collections import deque
        self._max = max_per_minute
        self._timestamps: deque[float] = deque()

    def wait(self) -> None:
        import time as _time
        now = _time.monotonic()
        while self._timestamps and now - self._timestamps[0] >= 60:
            self._timestamps.popleft()
        if len(self._timestamps) >= self._max:
            sleep_for = 60.0 - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


class RemoteAPIError(RuntimeError):
    """远程 API 返回错误时抛出。"""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class RemoteEmbedder(BaseEmbedder):
    """远程 Embedding 后端，通过 OpenAI 兼容的 REST API 调用。"""

    def __init__(
        self,
        base_url: str,
        model: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        api_key: str | None = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimensions = dimensions
        self._api_key = api_key
        self._limiter = _RateLimiter()
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=DEFAULT_TIMEOUT,
            headers=self._build_headers(),
        )

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def embed_video_chunk(self, chunk_path: str, verbose: bool = False) -> list[float]:
        chunk = Path(chunk_path).resolve()
        if not chunk.is_file():
            raise FileNotFoundError(f"视频分块未找到: {chunk_path}")

        with open(chunk, "rb") as f:
            video_bytes = f.read()
        b64 = base64.b64encode(video_bytes).decode("ascii")
        mime = mimetypes.guess_type(chunk)[0] or "video/mp4"

        self._limiter.wait()
        t0 = time.monotonic()
        embedding = self._embed_with_retry(
            content={"video": f"data:{mime};base64,{b64}"},
        )
        elapsed = time.monotonic() - t0

        if verbose:
            size_kb = len(video_bytes) / 1024
            print(
                f"    [verbose] 维度数={len(embedding)}, "
                f"分块大小={size_kb:.0f}KB, "
                f"API耗时={elapsed:.2f}s",
                file=sys.stderr,
            )

        return embedding

    def embed_query(self, query_text: str, verbose: bool = False) -> list[float]:
        self._limiter.wait()
        t0 = time.monotonic()
        embedding = self._embed_with_retry(content=query_text)
        elapsed = time.monotonic() - t0

        if verbose:
            print(
                f"  [verbose] 查询 Embedding: 维度数={len(embedding)}, "
                f"API耗时={elapsed:.2f}s",
                file=sys.stderr,
            )

        return embedding

    def embed_image(self, image_path: str, verbose: bool = False) -> list[float]:
        image = Path(image_path).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"图片文件未找到: {image_path}")

        ext = image.suffix.lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp",
            ".gif": "image/gif",
            ".heic": "image/heic", ".heif": "image/heif",
        }
        mime = mime_map.get(ext) or mimetypes.guess_type(image)[0] or "image/jpeg"

        with open(image, "rb") as f:
            img_bytes = f.read()
        b64 = base64.b64encode(img_bytes).decode("ascii")

        self._limiter.wait()
        t0 = time.monotonic()
        embedding = self._embed_with_retry(
            content={"image": f"data:{mime};base64,{b64}"},
        )
        elapsed = time.monotonic() - t0

        if verbose:
            size_kb = len(img_bytes) / 1024
            print(
                f"  [verbose] 图片 Embedding: 维度数={len(embedding)}, "
                f"大小={size_kb:.0f}KB, API耗时={elapsed:.2f}s",
                file=sys.stderr,
            )

        return embedding

    def dimensions(self) -> int:
        return self._dimensions

    def _embed_with_retry(
        self,
        content: str | dict,
        *,
        max_retries: int = 5,
        initial_delay: float = 2.0,
        max_delay: float = 60.0,
    ) -> list[float]:
        delay = initial_delay
        for attempt in range(max_retries + 1):
            try:
                return self._do_embed(content)
            except RemoteAPIError as exc:
                if exc.retryable and attempt < max_retries:
                    wait = min(delay, max_delay)
                    print(
                        f"  可重试错误 (第 {attempt + 1}/{max_retries} 次), "
                        f"等待 {wait:.0f}s: {exc}",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    delay *= 2
                else:
                    raise

    def _do_embed(self, content: str | dict) -> list[float]:
        body: dict = {
            "model": self._model,
            "input": content,
        }
        response = self._client.post("/embeddings", json=body)
        if response.status_code == 429:
            raise RemoteAPIError(
                f"远程 API 速率限制超出 (429): {response.text}",
                retryable=True,
            )
        if response.status_code == 503:
            raise RemoteAPIError(
                f"远程 API 不可用 (503): {response.text}",
                retryable=True,
            )
        if response.status_code >= 400:
            raise RemoteAPIError(
                f"远程 API 错误 {response.status_code}: {response.text}",
                retryable=False,
            )
        data = response.json()
        if not data.get("data"):
            raise RemoteAPIError("远程 API 未返回 Embedding 数据", retryable=False)
        return data["data"][0]["embedding"]

    def close(self) -> None:
        self._client.close()

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            self._client.close()
