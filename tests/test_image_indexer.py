"""Tests for sentrysearch.image_indexer."""

import os


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")


class TestScanImageDirectory:
    def test_finds_supported_formats_and_skips_unsupported_and_hidden(self, tmp_path):
        from sentrysearch.image_indexer import scan_image_directory

        jpg = tmp_path / "image.jpg"
        png = tmp_path / "image.png"
        webp = tmp_path / "image.webp"
        _touch(jpg)
        _touch(png)
        _touch(webp)
        _touch(tmp_path / "notes.txt")
        _touch(tmp_path / ".hidden.jpg")

        assert scan_image_directory(str(tmp_path)) == sorted(
            [str(jpg.resolve()), str(png.resolve()), str(webp.resolve())]
        )

    def test_recurses_into_subdirectories(self, tmp_path):
        from sentrysearch.image_indexer import scan_image_directory

        root_image = tmp_path / "root.jpg"
        nested_image = tmp_path / "nested" / "child.png"
        _touch(root_image)
        _touch(nested_image)

        assert scan_image_directory(str(tmp_path)) == sorted(
            [str(root_image.resolve()), str(nested_image.resolve())]
        )

    def test_skips_bmp_and_tiff(self, tmp_path):
        from sentrysearch.image_indexer import scan_image_directory

        _touch(tmp_path / "image.bmp")
        _touch(tmp_path / "image.tiff")

        assert scan_image_directory(str(tmp_path)) == []

    def test_empty_directory_returns_empty_list(self, tmp_path):
        from sentrysearch.image_indexer import scan_image_directory

        assert scan_image_directory(str(tmp_path)) == []

    def test_directory_with_no_supported_files_returns_empty_list(self, tmp_path):
        from sentrysearch.image_indexer import scan_image_directory

        _touch(tmp_path / "notes.txt")

        assert scan_image_directory(str(tmp_path)) == []

    def test_returns_absolute_paths(self, tmp_path, monkeypatch):
        from sentrysearch.image_indexer import scan_image_directory

        image_dir = tmp_path / "images"
        image = image_dir / "image.jpg"
        _touch(image)
        monkeypatch.chdir(tmp_path)

        results = scan_image_directory("images")

        assert results == [str(image.resolve())]
        assert os.path.isabs(results[0])


class TestIndexImageDirectory:
    def test_indexes_new_images_into_image_store(self, tmp_path, mock_embed_image):
        from sentrysearch.image_indexer import index_image_directory
        from sentrysearch.store import SentryStore

        image_a = tmp_path / "a.jpg"
        image_b = tmp_path / "b.png"
        _touch(image_a)
        _touch(image_b)
        store = SentryStore(
            db_path=tmp_path / "db",
            backend="gemini",
            collection_type="image",
        )

        assert index_image_directory(str(tmp_path), store) == (2, 0, 0)
        assert store.get_stats()["total_chunks"] == 2

    def test_skips_already_indexed_image(self, tmp_path, mock_embed_image):
        from sentrysearch.image_indexer import index_image_directory
        from sentrysearch.store import SentryStore

        image = tmp_path / "a.jpg"
        _touch(image)
        store = SentryStore(
            db_path=tmp_path / "db",
            backend="gemini",
            collection_type="image",
        )
        store.add_image(str(image.resolve()), mock_embed_image)

        assert index_image_directory(str(tmp_path), store) == (0, 1, 0)
        assert store.get_stats()["total_chunks"] == 1

    def test_empty_directory_counts_zero(self, tmp_path, mock_embed_image):
        from sentrysearch.image_indexer import index_image_directory
        from sentrysearch.store import SentryStore

        store = SentryStore(
            db_path=tmp_path / "db",
            backend="gemini",
            collection_type="image",
        )

        assert index_image_directory(str(tmp_path), store) == (0, 0, 0)
