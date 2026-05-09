"""查询与检索逻辑。"""

from .embedder import embed_image, embed_query
from .store import SentryStore


def _search_with_embedding(
    embedding: list[float],
    store: SentryStore,
    n_results: int,
) -> list[dict]:
    hits = store.search(embedding, n_results=n_results)
    results = [
        {
            "source_file": hit["source_file"],
            "start_time": hit["start_time"],
            "end_time": hit["end_time"],
            "similarity_score": hit["score"],
        }
        for hit in hits
    ]
    results.sort(key=lambda r: r["similarity_score"], reverse=True)
    return results


def search_footage(
    query: str,
    store: SentryStore,
    n_results: int = 5,
    verbose: bool = False,
) -> list[dict]:
    """使用自然语言查询检索已索引的视频片段。

    Args:
        query: 自然语言搜索字符串。
        store: 用于搜索的 SentryStore 实例。
        n_results: 返回的最大结果数量。
        verbose: 如果为 True，将调试信息输出到 stderr。

    Returns:
        按相关性排序（最佳优先）的结果字典列表。
        每个字典包含: source_file, start_time, end_time, similarity_score。
    """
    return _search_with_embedding(
        embed_query(query, verbose=verbose), store, n_results,
    )


def search_footage_by_image(
    image_path: str,
    store: SentryStore,
    n_results: int = 5,
    verbose: bool = False,
) -> list[dict]:
    """使用图片作为查询来检索已索引的视频片段。"""
    return _search_with_embedding(
        embed_image(image_path, verbose=verbose), store, n_results,
    )
