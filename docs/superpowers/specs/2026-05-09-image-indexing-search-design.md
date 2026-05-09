# Image Folder Indexing & Search-to-Image Design

## Summary

Add image folder indexing and image-to-image similarity search to SentrySearch. Users can index a directory of images and search for visually similar images within the index using an image as query. Optional reranking via SiliconFlow Qwen3-VL-Reranker API for improved accuracy.

## Commands

### `sentrysearch index-images <directory>`

Index all supported images from a directory into ChromaDB.

Options:
- `--backend` (gemini|local|remote) — embedding backend, default gemini
- `--model` — local model ID (implies `--backend local`)
- `--quantize / --no-quantize` — 4-bit quantization for local backend
- `--remote-url` — remote embed API URL (implies `--backend remote`)
- `--remote-api-key` — remote embed API key
- `--verbose` — show debug output

Supported formats: jpg/jpeg/png/webp/gif/heic/heif. Unsupported formats are skipped with a warning.

### `sentrysearch search-images <image>`

Search the image index for visually similar images using an image query.

Options:
- `-n / --results` — number of results (default 5)
- `--threshold` — minimum similarity score (default 0.7)
- `--rerank` — use reranker for improved ranking (off by default)
- `--backend`, `--model`, `--quantize`, `--remote-url`, `--remote-api-key`, `--verbose`

Output: sorted list of matching file paths with similarity scores. No trim/save/open operations (pure search).

## ChromaDB Collections

Image collections are separate from video collections:

| Backend | Collection Name |
|---------|----------------|
| gemini | `image_index` |
| local + model | `image_index_local_{model}` |
| remote + model | `image_index_remote_{model}` |

## Module Design

### `sentrysearch/image_indexer.py` (new)

Image indexing loop:

```
index_image_directory(directory, store, embedder, *, verbose)
  - Scan directory for supported image extensions
  - For each image:
    1. chunk_id = SHA-256(absolute_file_path)[:16]
    2. Skip if already indexed (store.has_chunk)
    3. embedder.embed_image(path) → embedding
    4. store.add_image(id, embedding, metadata)
  - Failed images go to DLQ
  - Return (new_images, skipped, failed) stats
```

### `sentrysearch/reranker.py` (new)

Reranker interface and SiliconFlow implementation:

- `RemoteReranker(api_url)` — calls `/rerank` endpoint with query image and candidate list
- Model: `Qwen/Qwen3-VL-Reranker-8B`
- API auth via `REMOTE_EMBED_API_KEY` (same key as embedding)
- `--rerank` flag triggers: vector search top-50 → rerank → return top-N

### `sentrysearch/store.py` (modify)

Add `add_image()` method to `SentryStore`:

- chunk_id = SHA-256 of absolute file path (no start_time needed)
- Metadata stored: source_file
- Existing `search()`, `get_stats()`, `has_chunk()` reusable as-is

Add collection naming for image collections:

```python
def _image_collection_name(backend, model=None):
    if backend == "gemini": return "image_index"
    if backend == "local": return f"image_index_local_{model}"
    if backend == "remote": return f"image_index_remote_{model}"
```

`detect_index()` extended to detect image indexes alongside video indexes.

### `sentrysearch/search.py` (modify)

Add `search_images()` function:

```
search_images(image_path, store, n_results, *, verbose, reranker=None)
  - embed_image(image_path) → embedding
  - vector search → candidates (top-50 if rerank, else top-n_results)
  - if reranker: reranker.rerank(image_path, candidates) → top-n_results
  - return results list
```

### `sentrysearch/cli.py` (modify)

Two new Click commands:
- `index_images` — mirrors `index` command structure (without video-specific options)
- `search_images` — simplified search output (paths + scores only)

## Rerank Flow

```
User: search-images query.jpg --rerank
  ├── embed_image(query.jpg) → embedding
  ├── ChromaDB.query(embedding, n_results=50) → candidates
  ├── RemoteReranker.rerank(query.jpg, candidates, top_n=n_results)
  │     POST /rerank {query_image: base64, documents: [...], top_n: N}
  └── Display reranked results
```

## Data Flow

**Indexing:**
```
scan_directory → filter image formats → embedder.embed_image → SentryStore.add_image → ChromaDB
```

**Searching:**
```
query image → embedder.embed_image → SentryStore.search → [optional: Reranker.rerank] → display results
```

## Error Handling

- Unsupported image formats: skip with warning
- Embedding failures: record to DLQ, continue
- Missing remote URL: raise UsageError (same as video flow)
- Empty index: display helpful message
- Search with no results: display suggestions
