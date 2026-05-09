"""Tests for sentrysearch.search."""

import math

import pytest

from sentrysearch.search import search_footage, search_images
from sentrysearch.store import SentryStore


def _make_embedding(seed: float, dim: int = 768) -> list[float]:
    vec = [math.sin(seed + i * 0.1) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


class TestSearchFootage:
    def test_empty_store(self, tmp_store, mock_embed_query):
        results = search_footage("a red car", tmp_store)
        assert results == []

    def test_returns_results(self, tmp_store, mock_embed_query):
        # mock_embed_query returns _fake_embedding(), store a chunk with same vector
        tmp_store.add_chunk("c1", mock_embed_query, {
            "source_file": "video.mp4",
            "start_time": 0.0,
            "end_time": 30.0,
        })
        results = search_footage("anything", tmp_store, n_results=5)
        assert len(results) == 1
        assert results[0]["source_file"] == "video.mp4"
        assert results[0]["similarity_score"] > 0.99

    def test_sorted_by_score(self, tmp_store, mock_embed_query):
        tmp_store.add_chunk("match", mock_embed_query, {
            "source_file": "match.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        tmp_store.add_chunk("diff", _make_embedding(seed=999.0), {
            "source_file": "diff.mp4", "start_time": 0.0, "end_time": 30.0,
        })
        results = search_footage("query", tmp_store, n_results=5)
        assert len(results) == 2
        assert results[0]["source_file"] == "match.mp4"
        assert results[0]["similarity_score"] > results[1]["similarity_score"]

    def test_n_results_limits_output(self, tmp_store, mock_embed_query):
        for i in range(10):
            tmp_store.add_chunk(f"c{i}", _make_embedding(seed=float(i)), {
                "source_file": f"v{i}.mp4",
                "start_time": 0.0,
                "end_time": 30.0,
            })
        results = search_footage("q", tmp_store, n_results=3)
        assert len(results) == 3


class TestSearchImages:
    def test_empty_image_store(self, tmp_path, tiny_png, mock_embed_image):
        store = SentryStore(db_path=tmp_path / "image_db", collection_type="image")

        results = search_images(tiny_png, store)

        assert results == []

    def test_returns_sorted_image_results(self, tmp_path, tiny_png, mock_embed_image):
        store = SentryStore(db_path=tmp_path / "image_db", collection_type="image")
        store.add_image("match.jpg", mock_embed_image)
        store.add_image("diff.jpg", _make_embedding(seed=999.0))

        results = search_images(tiny_png, store, n_results=5)

        assert len(results) == 2
        assert set(results[0]) == {"source_file", "similarity_score"}
        assert results[0]["source_file"] == "match.jpg"
        assert results[0]["similarity_score"] > results[1]["similarity_score"]

    def test_n_results_limits_output(self, tmp_path, tiny_png, mock_embed_image):
        store = SentryStore(db_path=tmp_path / "image_db", collection_type="image")
        for i in range(10):
            store.add_image(f"image{i}.jpg", _make_embedding(seed=float(i)))

        results = search_images(tiny_png, store, n_results=3)

        assert len(results) == 3

    def test_reranker_uses_expanded_candidates(self, tmp_path, tiny_png, mock_embed_image):
        store = SentryStore(db_path=tmp_path / "image_db", collection_type="image")
        for i in range(60):
            store.add_image(f"image{i}.jpg", _make_embedding(seed=float(i)))

        class RecordingReranker:
            def rerank(self, image_path, candidates, top_n):
                self.image_path = image_path
                self.candidate_count = len(candidates)
                self.top_n = top_n
                return candidates[:top_n]

        reranker = RecordingReranker()

        results = search_images(tiny_png, store, n_results=2, reranker=reranker)

        assert len(results) == 2
        assert reranker.image_path == tiny_png
        assert reranker.candidate_count == 50
        assert reranker.top_n == 2

    def test_reranker_order_is_preserved(self, tmp_path, tiny_png, mock_embed_image):
        store = SentryStore(db_path=tmp_path / "image_db", collection_type="image")
        store.add_image("match.jpg", mock_embed_image)
        store.add_image("diff.jpg", _make_embedding(seed=999.0))

        class ReverseReranker:
            def rerank(self, image_path, candidates, top_n):
                return list(reversed(candidates))[:top_n]

        results = search_images(
            tiny_png,
            store,
            n_results=2,
            reranker=ReverseReranker(),
        )

        assert [result["source_file"] for result in results] == [
            "diff.jpg",
            "match.jpg",
        ]
