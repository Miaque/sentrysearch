"""Tests for sentrysearch.reranker."""

from unittest.mock import MagicMock, patch

from sentrysearch.reranker import RemoteReranker


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def _candidates():
    return [
        {"source_file": "/clips/alpha.jpg", "similarity_score": 0.1},
        {"source_file": "/clips/bravo.jpg", "similarity_score": 0.2},
        {"source_file": "/clips/charlie.jpg", "similarity_score": 0.3},
    ]


def test_rerank_calls_api_and_maps_ranked_indices_and_scores(monkeypatch):
    reranker = RemoteReranker("https://rerank.example/")
    reranker._client = MagicMock()
    reranker._client.post.return_value = FakeResponse(
        payload={
            "results": [
                {"index": 2, "relevance_score": 0.91},
                {"index": 0, "relevance_score": 0.73},
            ],
        },
    )
    monkeypatch.setattr(reranker, "_encode_image", lambda path: "encoded-image")

    result = reranker.rerank("query.png", _candidates(), top_n=2)

    reranker._client.post.assert_called_once_with(
        "/rerank",
        json={
            "query": "data:image;base64,encoded-image",
            "documents": ["alpha.jpg", "bravo.jpg", "charlie.jpg"],
            "top_n": 2,
        },
    )
    assert result == [
        {"source_file": "/clips/charlie.jpg", "similarity_score": 0.91},
        {"source_file": "/clips/alpha.jpg", "similarity_score": 0.73},
    ]


def test_rerank_respects_top_n(monkeypatch):
    reranker = RemoteReranker("https://rerank.example")
    reranker._client = MagicMock()
    reranker._client.post.return_value = FakeResponse(
        payload={
            "results": [
                {"index": 2, "score": 0.9},
                {"index": 1, "score": 0.8},
                {"index": 0, "score": 0.7},
            ],
        },
    )
    monkeypatch.setattr(reranker, "_encode_image", lambda path: "encoded-image")

    result = reranker.rerank("query.png", _candidates(), top_n=2)

    assert len(result) == 2
    assert [item["source_file"] for item in result] == [
        "/clips/charlie.jpg",
        "/clips/bravo.jpg",
    ]


def test_rerank_empty_candidates_returns_empty_list():
    reranker = RemoteReranker("https://rerank.example")
    reranker._client = MagicMock()

    assert reranker.rerank("query.png", []) == []
    reranker._client.post.assert_not_called()


def test_rerank_api_error_falls_back_to_original_candidates(monkeypatch, capsys):
    reranker = RemoteReranker("https://rerank.example")
    reranker._client = MagicMock()
    reranker._client.post.return_value = FakeResponse(
        status_code=500,
        text="server exploded",
    )
    monkeypatch.setattr(reranker, "_encode_image", lambda path: "encoded-image")

    result = reranker.rerank("query.png", _candidates(), top_n=2)

    assert result == _candidates()[:2]
    assert "500" in capsys.readouterr().err


def test_rerank_exception_falls_back_to_original_candidates(monkeypatch, capsys):
    reranker = RemoteReranker("https://rerank.example")
    reranker._client = MagicMock()
    reranker._client.post.side_effect = RuntimeError("network down")
    monkeypatch.setattr(reranker, "_encode_image", lambda path: "encoded-image")

    result = reranker.rerank("query.png", _candidates(), top_n=2)

    assert result == _candidates()[:2]
    stderr = capsys.readouterr().err
    assert "远程重排序失败" in stderr
    assert "network down" in stderr


def test_rerank_accepts_data_response_field(monkeypatch):
    reranker = RemoteReranker("https://rerank.example")
    reranker._client = MagicMock()
    reranker._client.post.return_value = FakeResponse(
        payload={
            "data": [
                {"index": 1, "score": 0.66},
                {"index": 0, "score": 0.55},
            ],
        },
    )
    monkeypatch.setattr(reranker, "_encode_image", lambda path: "encoded-image")

    result = reranker.rerank("query.png", _candidates(), top_n=2)

    assert result == [
        {"source_file": "/clips/bravo.jpg", "similarity_score": 0.66},
        {"source_file": "/clips/alpha.jpg", "similarity_score": 0.55},
    ]


def test_rerank_empty_results_falls_back_to_original_candidates(monkeypatch):
    reranker = RemoteReranker("https://rerank.example")
    reranker._client = MagicMock()
    reranker._client.post.return_value = FakeResponse(payload={"results": []})
    monkeypatch.setattr(reranker, "_encode_image", lambda path: "encoded-image")

    result = reranker.rerank("query.png", _candidates(), top_n=2)

    assert result == _candidates()[:2]


def test_rerank_skips_out_of_range_indices_and_keeps_valid_items(monkeypatch):
    reranker = RemoteReranker("https://rerank.example")
    reranker._client = MagicMock()
    reranker._client.post.return_value = FakeResponse(
        payload={
            "results": [
                {"index": 99, "score": 0.99},
                {"index": 1, "score": 0.66},
                {"index": -1, "score": 0.55},
            ],
        },
    )
    monkeypatch.setattr(reranker, "_encode_image", lambda path: "encoded-image")

    result = reranker.rerank("query.png", _candidates(), top_n=3)

    assert result == [
        {"source_file": "/clips/bravo.jpg", "similarity_score": 0.66},
    ]


def test_rerank_missing_score_defaults_to_zero(monkeypatch):
    reranker = RemoteReranker("https://rerank.example")
    reranker._client = MagicMock()
    reranker._client.post.return_value = FakeResponse(
        payload={"results": [{"index": 0}]},
    )
    monkeypatch.setattr(reranker, "_encode_image", lambda path: "encoded-image")

    result = reranker.rerank("query.png", _candidates(), top_n=1)

    assert result == [
        {"source_file": "/clips/alpha.jpg", "similarity_score": 0.0},
    ]


def test_api_key_headers_are_configured():
    with patch("sentrysearch.reranker.httpx.Client") as client_cls:
        RemoteReranker("https://rerank.example/", api_key="secret-key")

    client_cls.assert_called_once_with(
        base_url="https://rerank.example",
        timeout=120.0,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer secret-key",
        },
    )


def test_no_api_key_omits_authorization_header():
    with patch("sentrysearch.reranker.httpx.Client") as client_cls:
        RemoteReranker("https://rerank.example/", api_key=None)

    headers = client_cls.call_args.kwargs["headers"]
    assert headers == {"Content-Type": "application/json"}
    assert "Authorization" not in headers


def test_close_closes_underlying_client():
    reranker = RemoteReranker("https://rerank.example")
    reranker._client = MagicMock()

    reranker.close()

    reranker._client.close.assert_called_once_with()


def test_encode_image_returns_base64_ascii(tmp_path):
    image = tmp_path / "query.bin"
    image.write_bytes(b"abc123")
    reranker = RemoteReranker("https://rerank.example")

    assert reranker._encode_image(str(image)) == "YWJjMTIz"
