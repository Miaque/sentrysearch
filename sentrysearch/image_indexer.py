"""Image directory scanning and indexing."""

import hashlib
import os
import sys


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".gif",
    ".heic",
    ".heif",
}


def scan_image_directory(directory: str) -> list[str]:
    """Return sorted absolute paths for supported images under *directory*."""
    image_paths = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.startswith("."):
                continue
            ext = os.path.splitext(filename)[1].lower()
            if ext not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            image_paths.append(os.path.abspath(os.path.join(root, filename)))
    return sorted(image_paths)


def _make_image_id(file_path: str) -> str:
    """Return the deterministic image ID used by SentryStore.add_image."""
    return hashlib.sha256(file_path.encode()).hexdigest()[:16]


def index_image_directory(directory: str, store, *, verbose=False) -> tuple[int, int, int]:
    """Index supported images in *directory* into *store*."""
    from sentrysearch.embedder import embed_image

    new = 0
    skipped = 0
    failed = 0
    for image_path in scan_image_directory(directory):
        image_id = _make_image_id(image_path)
        if store.has_chunk(image_id):
            skipped += 1
            continue
        try:
            embedding = embed_image(image_path, verbose=verbose)
            store.add_image(image_path, embedding)
        except Exception as exc:
            print(f"Failed to index image {image_path}: {exc}", file=sys.stderr)
            failed += 1
            continue
        new += 1
    return new, skipped, failed
