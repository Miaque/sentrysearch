"""Integration tests for RemoteEmbedder against the real SiliconFlow API.

These tests make actual HTTP calls to the embedding service. They are skipped
if the REMOTE_EMBED_API_KEY environment variable is not set.
"""

import os
import pathlib

import pytest

from sentrysearch.remote_embedder import RemoteEmbedder

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not os.environ.get("REMOTE_EMBED_API_KEY"),
    reason="REMOTE_EMBED_API_KEY not set",
)


@pytest.fixture(scope="module")
def embedder():
    """Create a RemoteEmbedder connected to the real SiliconFlow API."""
    emb = RemoteEmbedder(
        base_url="https://api.siliconflow.cn/v1",
        model="Qwen/Qwen3-VL-Embedding-8B",
        dimensions=4096,
        api_key=os.environ["REMOTE_EMBED_API_KEY"],
    )
    yield emb
    emb.close()


@pytest.fixture(scope="module")
def tiny_mp4(ffmpeg_exe, tmp_path_factory):
    """Generate a 1-second synthetic MP4 video."""
    import subprocess

    video_dir = tmp_path_factory.mktemp("videos")
    video_path = video_dir / "test_1s.mp4"
    subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x64:rate=10:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        capture_output=True,
        check=True,
    )
    assert video_path.exists() and video_path.stat().st_size > 0
    return str(video_path)


@pytest.fixture(scope="module")
def test_image():
    image_dir = pathlib.Path(__file__).parent / "images"

    return str(image_dir / "source.png")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmbedQuery:
    def test_text_query_returns_4096_dim_vector(self, embedder):
        result = embedder.embed_query("a car driving on a highway")
        assert isinstance(result, list)
        assert len(result) == 4096
        assert all(isinstance(v, float) for v in result)

    def test_different_queries_produce_different_embeddings(self, embedder):
        emb1 = embedder.embed_query("a red car on a highway")
        emb2 = embedder.embed_query("a cat sitting on a sofa")
        assert emb1 != emb2

    def test_chinese_query(self, embedder):
        result = embedder.embed_query("一辆汽车在高速公路上行驶")
        assert len(result) == 4096

    def test_empty_query(self, embedder):
        result = embedder.embed_query("")
        assert len(result) == 4096


class TestEmbedImage:
    def test_image_embedding(self, embedder, test_image):
        result = embedder.embed_image(str(test_image))
        assert isinstance(result, list)
        assert len(result) == 4096

    def test_image_not_found(self, embedder):
        with pytest.raises(FileNotFoundError):
            embedder.embed_image("/nonexistent/path/image.png")


class TestEmbedVideoChunk:
    def test_video_embedding(self, embedder, tiny_mp4):
        result = embedder.embed_video_chunk(tiny_mp4)
        assert isinstance(result, list)
        assert len(result) == 4096

    def test_video_not_found(self, embedder):
        with pytest.raises(FileNotFoundError):
            embedder.embed_video_chunk("/nonexistent/path/chunk.mp4")


class TestDimensions:
    def test_returns_configured_dimensions(self, embedder):
        assert embedder.dimensions() == 4096


class TestRateLimitRetry:
    def test_verbose_output(self, embedder, capsys):
        embedder.embed_query("test verbose", verbose=True)
        captured = capsys.readouterr()
        output = (captured.out + captured.err).lower()
        assert "verbose" in output
