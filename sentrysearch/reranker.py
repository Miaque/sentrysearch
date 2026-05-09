"""远程图片重排序后端。"""

import base64
import os
import sys

import httpx


class RemoteReranker:
    """通过远程 API 对图片检索候选结果重排序。"""

    def __init__(self, base_url: str, api_key: str | None = None):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=120.0,
            headers=self._build_headers(),
        )

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def rerank(
        self,
        query_image: str,
        candidates: list[dict],
        *,
        top_n=5,
        verbose=False,
    ) -> list[dict]:
        if not candidates:
            return []

        fallback = candidates[:top_n]
        try:
            documents = [
                os.path.basename(candidate["source_file"])
                for candidate in candidates
            ]
            body = {
                "query": f"data:image;base64,{self._encode_image(query_image)}",
                "documents": documents,
                "top_n": top_n,
            }
            response = self._client.post("/rerank", json=body)
            if response.status_code >= 400:
                print(
                    f"远程重排序 API 错误 {response.status_code}: {response.text}",
                    file=sys.stderr,
                )
                return fallback

            data = response.json()
            results = data.get("results") or data.get("data") or []
            if not results:
                return fallback

            reranked = []
            for item in results[:top_n]:
                index = item["index"]
                if not 0 <= index < len(candidates):
                    continue
                score = item.get("relevance_score", item.get("score", 0.0))
                reranked.append({
                    "source_file": candidates[index]["source_file"],
                    "similarity_score": score,
                })
            return reranked
        except Exception as exc:
            print(f"远程重排序失败: {exc}", file=sys.stderr)
            return fallback

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    def close(self) -> None:
        self._client.close()

    def __del__(self) -> None:
        if hasattr(self, "_client"):
            self._client.close()
