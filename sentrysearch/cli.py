"""基于 Click 的 CLI 入口。"""

import os
import platform
import shutil
import subprocess

import click
from dotenv import load_dotenv

_ENV_PATH = os.path.join(os.path.expanduser("~"), ".sentrysearch", ".env")

# Load from stable config location first, then cwd as fallback
load_dotenv(_ENV_PATH)
load_dotenv()  # cwd .env can override


def _resolve_remote_url(cli_url: str | None = None) -> str:
    """从 CLI 参数或环境变量解析远程 API 地址。"""
    url = cli_url or os.environ.get("REMOTE_EMBED_URL")
    if not url:
        raise click.UsageError(
            "Remote backend requires --remote-url or REMOTE_EMBED_URL environment variable.\n"
            "Example: sentrysearch index <dir> --backend remote --remote-url http://localhost:8000"
        )
    return url


def _resolve_remote_api_key(cli_key: str | None = None) -> str | None:
    """从 CLI 参数或环境变量解析远程 API 密钥。"""
    return cli_key or os.environ.get("REMOTE_EMBED_API_KEY")


def _fmt_time(seconds: float) -> str:
    """将秒数格式化为 MM:SS 格式。"""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _cache_last_clip(path: str) -> None:
    """记录最近保存的 clip 路径，供 sentryblur --last 跨工具集成使用。

    缓存失败不致命 — 这是体验优化，非正确性要求。
    """
    from pathlib import Path

    from . import _toolkit_cache

    try:
        _toolkit_cache.write_last_clip(Path(os.path.abspath(path)))
        click.echo("已为 sentryblur --last 缓存 clip 路径", err=True)
    except Exception as e:
        click.secho(
            f"（警告：无法写入 last-clip 缓存：{e}）",
            fg="yellow", err=True,
        )


def _open_file(path: str) -> None:
    """用系统默认应用打开文件。"""
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", path])
        elif system == "Windows":
            os.startfile(path)
        else:
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass  # non-critical — clip is already saved


def _overlay_output_path(path: str) -> str:
    """返回源视频的默认叠加层输出路径。"""
    base, _ext = os.path.splitext(path)
    return f"{base}_overlay.mp4"


def _is_permanent_failure(exc: Exception) -> bool:
    """判断错误是否为不可恢复的永久性错误（重试同一片段无济于事）。"""
    msg = str(exc).lower()
    if isinstance(exc, FileNotFoundError):
        return True
    # OOM — same chunk at same settings will OOM again
    if "out of memory" in msg or "cuda out of memory" in msg:
        return True
    # Decoder failures on specific files
    if "invalid data" in msg or "could not decode" in msg:
        return True
    return False


def _embed_with_retry(
    embedder,
    embed_path: str,
    chunk: dict,
    dlq,
    *,
    max_attempts: int = 3,
    verbose: bool = False,
) -> list[float] | None:
    """对片段进行嵌入并支持重试。永久性或耗尽失败时记录到 DLQ
    并返回 None，以便调用方继续处理。

    配额错误会向上抛出 — 用户需要停止并等待。
    """
    import time as _time
    from .gemini_embedder import GeminiAPIKeyError, GeminiQuotaError

    chunk_id = chunk["chunk_id"]
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return embedder.embed_video_chunk(embed_path, verbose=verbose)
        except (GeminiQuotaError, GeminiAPIKeyError):
            raise  # user-facing, stop the run
        except Exception as exc:
            last_exc = exc
            if _is_permanent_failure(exc) or attempt == max_attempts:
                dlq.record(
                    chunk_id,
                    source_file=chunk["source_file"],
                    start_time=chunk["start_time"],
                    end_time=chunk["end_time"],
                    error=repr(exc),
                    attempts=attempt,
                )
                click.secho(
                    f"  Failed after {attempt} attempt(s), recorded to DLQ: {exc}",
                    fg="yellow",
                    err=True,
                )
                return None
            wait = 2 ** attempt
            click.secho(
                f"  Embed error (attempt {attempt}/{max_attempts}), "
                f"retrying in {wait}s: {exc}",
                fg="yellow",
                err=True,
            )
            _time.sleep(wait)
    # unreachable — loop always returns or records
    if last_exc is not None:
        raise last_exc
    return None


def _handle_error(e: Exception) -> None:
    """打印用户友好的错误信息并退出。"""
    from .gemini_embedder import GeminiAPIKeyError, GeminiQuotaError
    from .local_embedder import LocalModelError
    from .store import BackendMismatchError

    if isinstance(e, GeminiAPIKeyError):
        click.secho("错误：" + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, GeminiQuotaError):
        click.secho("错误：" + str(e), fg="yellow", err=True)
        raise SystemExit(1)
    if isinstance(e, LocalModelError):
        click.secho("错误：" + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, BackendMismatchError):
        click.secho("错误：" + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, PermissionError):
        click.secho("错误：" + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, FileNotFoundError):
        click.secho("错误：" + str(e), fg="red", err=True)
        raise SystemExit(1)
    if isinstance(e, RuntimeError) and "ffmpeg not found" in str(e).lower():
        click.secho(
            "错误：ffmpeg 不可用。\n\n"
            "请使用以下方式之一安装：\n"
            "  Ubuntu/Debian:  sudo apt install ffmpeg\n"
            "  macOS:          brew install ffmpeg\n"
            "  pip 备选方案:   uv add imageio-ffmpeg",
            fg="red",
            err=True,
        )
        raise SystemExit(1)
    raise e


def _apply_overlay_to_clip(
    clip_path: str,
    source_file: str,
    start_time: float,
    end_time: float,
    *,
    replace: bool = True,
) -> bool:
    """将 Tesla 遥测叠加层烧录到 clip 上。成功返回 True。

    当 *replace* 为 True 时，叠加层会原地写入 *clip_path*。
    """
    from .overlay import apply_overlay, get_metadata_samples, reverse_geocode

    samples = get_metadata_samples(source_file, start_time, end_time)
    if samples is None:
        click.secho(
            "未找到 Tesla SEI 元数据 — 跳过叠加层。",
            fg="yellow", err=True,
        )
        return False

    location = None
    mid = samples[len(samples) // 2]
    lat = mid.get("latitude_deg", 0.0)
    lon = mid.get("longitude_deg", 0.0)
    if lat and lon:
        click.echo("正在反查地理位置...")
        location = reverse_geocode(lat, lon)
        if location is None:
            click.secho(
                "地理编码失败 — 继续不带位置信息。"
                "安装依赖：uv tool install \".[tesla]\"",
                fg="yellow", err=True,
            )

    overlay_path = _overlay_output_path(clip_path)
    result_path = apply_overlay(
        clip_path, overlay_path, samples, location,
        source_file=source_file,
        start_time=start_time,
    )
    if result_path == overlay_path and os.path.isfile(overlay_path):
        if replace:
            os.replace(overlay_path, clip_path)
        click.echo("已应用 Tesla 元数据叠加层")
        return True

    click.secho("叠加层应用失败。", fg="yellow", err=True)
    return False


@click.group()
def cli():
    """使用自然语言查询搜索行车记录仪视频。"""


# -----------------------------------------------------------------------
# init
# -----------------------------------------------------------------------

@cli.command()
def init():
    """配置 sentrysearch 的 Gemini API 密钥。"""
    env_path = _ENV_PATH
    os.makedirs(os.path.dirname(env_path), exist_ok=True)

    # Check for existing key
    if os.path.exists(env_path):
        with open(env_path) as f:
            contents = f.read()
        if "GEMINI_API_KEY=" in contents:
            if not click.confirm("API 密钥已配置，是否覆盖？", default=False):
                return

    api_key = click.prompt(
        "请输入你的 Gemini API 密钥\n"
        "  在 https://aistudio.google.com/apikey 获取\n"
        "  （输入内容已隐藏）",
        hide_input=True,
    )

    # Write/update .env
    if os.path.exists(env_path):
        with open(env_path) as f:
            lines = f.readlines()
        with open(env_path, "w") as f:
            found = False
            for line in lines:
                if line.startswith("GEMINI_API_KEY="):
                    f.write(f"GEMINI_API_KEY={api_key}\n")
                    found = True
                else:
                    f.write(line)
            if not found:
                f.write(f"GEMINI_API_KEY={api_key}\n")
    else:
        with open(env_path, "w") as f:
            f.write(f"GEMINI_API_KEY={api_key}\n")

    # Validate by embedding a test string
    os.environ["GEMINI_API_KEY"] = api_key
    click.echo("正在验证 API 密钥...")
    try:
        from .embedder import get_embedder

        embedder = get_embedder("gemini")
        vec = embedder.embed_query("test")
        if len(vec) != 768:
            click.secho(
                f"嵌入维度异常：{len(vec)}（预期 768）。"
                "密钥可能有效，但存在其他问题。",
                fg="yellow",
                err=True,
            )
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        click.secho(f"验证失败：{e}", fg="red", err=True)
        click.secho("请检查你的 API 密钥并重试。", fg="red", err=True)
        raise SystemExit(1)

    click.secho(
        "配置完成。你可以运行 "
        "`sentrysearch index <目录>` 开始使用。",
        fg="green",
    )
    click.secho(
        "\n提示：在 https://aistudio.google.com/billing 设置消费限额，"
        "以防止意外超支。",
        fg="yellow",
    )


# -----------------------------------------------------------------------
# index
# -----------------------------------------------------------------------

@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=True, dir_okay=True))
@click.option("--chunk-duration", default=30, show_default=True,
              help="片段时长（秒）。")
@click.option("--overlap", default=5, show_default=True,
              help="片段间重叠时长（秒）。")
@click.option("--preprocess/--no-preprocess", default=True, show_default=True,
              help="编入索引前降低分辨率和帧率。")
@click.option("--target-resolution", default=480, show_default=True,
              help="预处理目标视频高度（像素）。")
@click.option("--target-fps", default=5, show_default=True,
              help="预处理目标帧率。")
@click.option("--skip-still/--no-skip-still", default=True, show_default=True,
              help="跳过无显著视觉变化的片段。")
@click.option("--backend", type=click.Choice(["gemini", "local", "remote"]), default=None,
              help="嵌入后端（默认 gemini，设置 --model 时为 local）。")
@click.option("--model", default=None, show_default=False,
              help="本地后端模型：qwen8b、qwen2b 或 HuggingFace ID "
                   "（默认自动检测硬件）。隐含 --backend local。")
@click.option("--quantize/--no-quantize", default=None,
              help="启用/禁用本地后端的 4-bit 量化（默认自动检测）。")
@click.option("--remote-url", default=None,
              help="远程嵌入 API 地址（如 http://localhost:8000）。隐含 --backend remote。")
@click.option("--remote-api-key", default=None,
              help="远程嵌入服务 API 密钥（或设置 REMOTE_EMBED_API_KEY）。")
@click.option("--retry-failed", is_flag=True,
              help="重试之前失败并被路由到 DLQ 的片段。")
@click.option("--verbose", is_flag=True, help="显示调试信息。")
def index(directory, chunk_duration, overlap, preprocess, target_resolution,
          target_fps, skip_still, backend, model, quantize, remote_url, remote_api_key, retry_failed, verbose):
    """将 DIRECTORY 中的视频文件编入索引以供搜索。"""
    from .chunker import (
        SUPPORTED_VIDEO_EXTENSIONS,
        _get_video_duration,
        chunk_video,
        expected_chunk_spans,
        is_still_frame_chunk,
        preprocess_chunk,
        scan_directory,
    )
    from .dlq import DeadLetterQueue
    from .embedder import get_embedder, reset_embedder
    from .local_embedder import detect_default_model, normalize_model_key
    from .store import SentryStore

    try:
        if overlap >= chunk_duration:
            raise click.BadParameter(
                f"overlap ({overlap}s) must be less than chunk_duration ({chunk_duration}s).",
                param_hint="'--overlap'",
            )

        # --model implies --backend local, --remote-url implies --backend remote
        if model is not None and backend is None:
            backend = "local"
        if remote_url is not None and backend is None:
            backend = "remote"
        if backend is None:
            backend = "gemini"

        # Auto-detect model from hardware when using local backend
        if backend == "local" and model is None:
            model = detect_default_model()
            click.echo(f"自动检测到模型：{model}", err=True)

        # Normalize model key for consistent collection naming
        if backend == "local":
            model = normalize_model_key(model)

        # Resolve remote URL and API key
        if backend == "remote":
            remote_url = _resolve_remote_url(remote_url)
            api_key = _resolve_remote_api_key(remote_api_key)
            model = model or "Qwen/Qwen3-VL-Embedding-8B"

        embedder = get_embedder(
            backend, model=model, quantize=quantize,
            base_url=remote_url if backend == "remote" else None,
            api_key=api_key if backend == "remote" else None,
        )

        if os.path.isfile(directory):
            videos = [os.path.abspath(directory)]
        else:
            videos = scan_directory(directory)

        if not videos:
            supported = ", ".join(SUPPORTED_VIDEO_EXTENSIONS)
            click.echo(f"未找到支持的视频文件（{supported}）。")
            return

        store = SentryStore(backend=backend, model=model if backend in ("local", "remote") else None)
        dlq = DeadLetterQueue()
        total_files = len(videos)
        new_files = 0
        new_chunks = 0
        skipped_chunks = 0
        dlq_chunks = 0

        if verbose:
            click.echo(f"[verbose] 数据库路径：{store._client._identifier}", err=True)
            click.echo(f"[verbose] 后端={backend}, 片段时长={chunk_duration}s, 重叠={overlap}s", err=True)

        for file_idx, video_path in enumerate(videos, 1):
            abs_path = os.path.abspath(video_path)
            basename = os.path.basename(video_path)

            # Fast path: if every expected chunk ID is already in the store,
            # skip ffmpeg splitting entirely. A mismatch (e.g. due to
            # still-frame chunks that were skipped rather than stored) falls
            # through to the normal path.
            try:
                duration = _get_video_duration(abs_path)
                expected_spans = expected_chunk_spans(
                    duration, chunk_duration=chunk_duration, overlap=overlap,
                )
                if expected_spans and all(
                    store.has_chunk(store.make_chunk_id(abs_path, s))
                    for s, _ in expected_spans
                ):
                    click.echo(
                        f"跳过 ({file_idx}/{total_files})：{basename} "
                        f"（已编入索引）"
                    )
                    continue
            except Exception:
                # Duration probe failed — let chunk_video surface the error
                pass

            chunks = chunk_video(abs_path, chunk_duration=chunk_duration, overlap=overlap)
            num_chunks = len(chunks)
            file_new_chunks = 0

            if verbose:
                click.echo(f"  [verbose] {basename}：时长拆分为 {num_chunks} 个片段", err=True)

            # Track files to clean up after processing
            files_to_cleanup = []

            for chunk_idx, chunk in enumerate(chunks, 1):
                chunk_id = store.make_chunk_id(abs_path, chunk["start_time"])

                if store.has_chunk(chunk_id):
                    if verbose:
                        click.echo(
                            f"  [verbose] chunk {chunk_idx}/{num_chunks} already indexed — resuming",
                            err=True,
                        )
                    files_to_cleanup.append(chunk["chunk_path"])
                    continue

                if dlq.contains(chunk_id):
                    if retry_failed:
                        dlq.remove(chunk_id)
                        if verbose:
                            click.echo(
                                f"  [verbose] retrying DLQ'd chunk {chunk_idx}/{num_chunks}",
                                err=True,
                            )
                    else:
                        click.echo(
                            f"跳过片段 {chunk_idx}/{num_chunks}（在 DLQ 中 — "
                            f"使用 --retry-failed 重试）"
                        )
                        files_to_cleanup.append(chunk["chunk_path"])
                        continue

                if skip_still and is_still_frame_chunk(
                    chunk["chunk_path"], verbose=verbose,
                ):
                    click.echo(
                        f"跳过片段 {chunk_idx}/{num_chunks}（静态帧）"
                    )
                    skipped_chunks += 1
                    files_to_cleanup.append(chunk["chunk_path"])
                    continue

                click.echo(
                    f"正在索引文件 {file_idx}/{total_files}：{basename} "
                    f"[片段 {chunk_idx}/{num_chunks}]"
                )

                embed_path = chunk["chunk_path"]
                if preprocess:
                    original_size = os.path.getsize(embed_path)
                    embed_path = preprocess_chunk(
                        embed_path,
                        target_resolution=target_resolution,
                        target_fps=target_fps,
                    )
                    if verbose:
                        new_size = os.path.getsize(embed_path)
                        click.echo(
                            f"    [verbose] 预处理：{original_size / 1024:.0f}KB -> "
                            f"{new_size / 1024:.0f}KB "
                            f"（缩减 {100 * (1 - new_size / original_size):.0f}%）",
                            err=True,
                        )
                    if embed_path != chunk["chunk_path"]:
                        files_to_cleanup.append(embed_path)

                embedding = _embed_with_retry(
                    embedder, embed_path,
                    {
                        "chunk_id": chunk_id,
                        "source_file": abs_path,
                        "start_time": chunk["start_time"],
                        "end_time": chunk["end_time"],
                    },
                    dlq, verbose=verbose,
                )
                files_to_cleanup.append(chunk["chunk_path"])
                if embedding is None:
                    dlq_chunks += 1
                    continue
                store.add_chunk(chunk_id, embedding, {
                    "source_file": abs_path,
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                })
                file_new_chunks += 1

            for f in files_to_cleanup:
                try:
                    os.unlink(f)
                except OSError:
                    pass

            if chunks:
                tmp_dir = os.path.dirname(chunks[0]["chunk_path"])
                shutil.rmtree(tmp_dir, ignore_errors=True)

            if file_new_chunks:
                new_files += 1
                new_chunks += file_new_chunks

        stats = store.get_stats()
        parts = []
        if skipped_chunks:
            parts.append(f"跳过 {skipped_chunks} 个静态片段")
        if dlq_chunks:
            parts.append(f"{dlq_chunks} 个失败 → DLQ")
        extra = f"（{', '.join(parts)}）" if parts else ""
        click.echo(
            f"\n已从 {new_files} 个文件索引 {new_chunks} 个新片段{extra}。"
            f"总计：{stats['total_chunks']} 个片段，"
            f"{stats['unique_source_files']} 个源文件。"
        )
        if dlq_chunks:
            click.secho(
                f"详情见 `sentrysearch dlq list`。"
                f"使用 `sentrysearch index <目录> --retry-failed` 重试。",
                fg="yellow",
            )

    except Exception as e:
        _handle_error(e)
    finally:
        reset_embedder()


# -----------------------------------------------------------------------
# search
# -----------------------------------------------------------------------

@cli.command()
@click.argument("query")
@click.option("-n", "--results", "n_results", default=5, show_default=True,
              help="返回结果数量。")
@click.option("-o", "--output-dir", default="~/sentrysearch_clips", show_default=True,
              help="保存剪辑视频的目录。")
@click.option("--trim/--no-trim", default=True, show_default=True,
              help="自动截取排名最高的结果。")
@click.option("--save-top", default=None, type=click.IntRange(min=1),
              help="保存前 N 个匹配的剪辑视频（如 --save-top 3）。")
@click.option("--threshold", default=0.41, show_default=True, type=float,
              help="视为可信匹配的最低相似度分数。")
@click.option("--overlay/--no-overlay", default=False, show_default=True,
              help="将 Tesla 遥测叠加层（速度、GPS、转向灯）烧录到剪辑视频。")
@click.option("--backend", type=click.Choice(["gemini", "local", "remote"]), default=None,
              help="嵌入后端（省略时从索引自动检测）。")
@click.option("--model", default=None, show_default=False,
              help="本地后端模型：qwen8b、qwen2b 或 HuggingFace ID "
                   "（默认从索引自动检测）。隐含 --backend local。")
@click.option("--quantize/--no-quantize", default=None,
              help="启用/禁用本地后端的 4-bit 量化（默认自动检测）。")
@click.option("--remote-url", default=None,
              help="远程嵌入 API 地址。隐含 --backend remote。")
@click.option("--remote-api-key", default=None,
              help="远程嵌入服务 API 密钥（或设置 REMOTE_EMBED_API_KEY）。")
@click.option("--verbose", is_flag=True, help="显示调试信息。")
def search(query, n_results, output_dir, trim, save_top, threshold, overlay, backend, model, quantize, remote_url, remote_api_key, verbose):
    """使用自然语言 QUERY 搜索已索引的视频。"""
    from .embedder import get_embedder, reset_embedder
    from .local_embedder import normalize_model_key
    from .search import search_footage
    from .store import SentryStore, detect_index

    output_dir = os.path.expanduser(output_dir)

    try:
        # --model implies --backend local, --remote-url implies --backend remote
        if model is not None and backend is None:
            backend = "local"
        if remote_url is not None and backend is None:
            backend = "remote"

        # Normalize model key for consistent collection naming
        if model is not None:
            model = normalize_model_key(model)

        # Auto-detect backend and model from whichever collection has data
        if backend is None:
            detected_backend, detected_model = detect_index()
            backend = detected_backend or "gemini"
            if model is None:
                model = detected_model
        elif backend == "local" and model is None:
            _, detected_model = detect_index()
            model = detected_model

        # Resolve remote URL and API key
        api_key = None
        if backend == "remote":
            remote_url = _resolve_remote_url(remote_url)
            api_key = _resolve_remote_api_key(remote_api_key)
            model = model or "Qwen/Qwen3-VL-Embedding-8B"

        store = SentryStore(backend=backend, model=model if backend in ("local", "remote") else None)

        if store.get_stats()["total_chunks"] == 0:
            # Check if data exists under a different model
            det_backend, det_model = detect_index()
            if det_backend == backend and det_model and det_model != model:
                click.echo(
                    f"未使用 {model} 模型索引视频。"
                    f"你的索引使用的是 {det_model}。\n\n"
                    f"尝试：sentrysearch search \"{query}\" --model {det_model}"
                )
            elif det_backend and det_backend != backend:
                click.echo(
                    f"未使用 {backend} 后端索引视频。"
                    f"你的索引使用的是 {det_backend}。"
                )
            else:
                click.echo(
                    "未找到已索引的视频。"
                    "请先运行 `sentrysearch index <目录>`。"
                )
            return

        if backend == "local":
            click.secho(
                "提示：`sentrysearch shell` 可在多次查询间保持模型加载状态。",
                fg="yellow", err=True,
            )

        get_embedder(
            backend, model=model, quantize=quantize,
            base_url=remote_url if backend == "remote" else None,
            api_key=api_key if backend == "remote" else None,
        )

        # Ensure we fetch enough results for --save-top
        if save_top is not None and save_top > n_results:
            n_results = save_top

        if verbose:
            click.echo(f"  [verbose] 后端={backend}, 相似度阈值：{threshold}", err=True)

        results = search_footage(query, store, n_results=n_results, verbose=verbose)
        _present_results(results, threshold, trim, save_top, output_dir, overlay, verbose)

    except Exception as e:
        _handle_error(e)
    finally:
        reset_embedder()


def _present_results(results, threshold, trim, save_top, output_dir, overlay, verbose):
    """格式化并展示搜索结果。"""
    if not results:
        click.echo(
            "未找到结果。\n\n"
            "建议：\n"
            "  - 尝试更广泛或不同的查询\n"
            "  - 使用更小的 --chunk-duration 重新索引以获得更细粒度\n"
            "  - 运行 `sentrysearch stats` 查看已索引内容"
        )
        return

    best_score = results[0]["similarity_score"]
    low_confidence = best_score < threshold

    if low_confidence and not trim:
        click.secho(
            f"（置信度较低 — 最高得分：{best_score:.2f}）",
            fg="yellow",
            err=True,
        )

    for i, r in enumerate(results, 1):
        basename = os.path.basename(r["source_file"])
        start_str = _fmt_time(r["start_time"])
        end_str = _fmt_time(r["end_time"])
        score = r["similarity_score"]
        if verbose:
            click.echo(f"  #{i} [{score:.6f}] {basename} @ {start_str}-{end_str}")
        else:
            click.echo(f"  #{i} [{score:.2f}] {basename} @ {start_str}-{end_str}")

    should_trim = trim or save_top is not None
    if should_trim:
        if low_confidence:
            if not click.confirm(
                f"未找到可信匹配（最高得分：{best_score:.2f}）。"
                "仍要显示结果吗？",
                default=False,
            ):
                return

        from .trimmer import trim_top_results
        count = save_top if save_top is not None else 1
        clip_paths = trim_top_results(results, output_dir, count=count)

        for i, clip_path in enumerate(clip_paths):
            if overlay:
                r = results[i]
                _apply_overlay_to_clip(
                    clip_path, r["source_file"],
                    r["start_time"], r["end_time"],
                )
            click.echo(f"\n已保存剪辑：{clip_path}")

        if clip_paths:
            _cache_last_clip(clip_paths[0])
            _open_file(clip_paths[0])


@cli.command()
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.option("-n", "--results", "n_results", default=5, show_default=True,
              help="返回结果数量。")
@click.option("-o", "--output-dir", default="~/sentrysearch_clips", show_default=True,
              help="保存剪辑视频的目录。")
@click.option("--trim/--no-trim", default=True, show_default=True,
              help="截取并保存排名最高的结果为剪辑视频。")
@click.option("--save-top", default=None, type=click.IntRange(min=1),
              help="保存前 N 个匹配的剪辑视频。")
@click.option("--threshold", default=0.41, show_default=True, type=float,
              help="视为可信匹配的最低相似度分数。")
@click.option("--overlay/--no-overlay", default=False, show_default=True,
              help="对保存的剪辑视频应用 Tesla 遥测叠加层。")
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
def img(image, n_results, output_dir, trim, save_top, threshold, overlay,
        backend, model, quantize, remote_url, remote_api_key, verbose):
    """使用 IMAGE 图片作为查询搜索已索引的视频。"""
    from .embedder import get_embedder, reset_embedder
    from .local_embedder import normalize_model_key
    from .search import search_footage_by_image
    from .store import SentryStore, detect_index

    output_dir = os.path.expanduser(output_dir)

    try:
        if model is not None and backend is None:
            backend = "local"
        if remote_url is not None and backend is None:
            backend = "remote"
        if model is not None:
            model = normalize_model_key(model)
        if backend is None:
            detected_backend, detected_model = detect_index()
            backend = detected_backend or "gemini"
            if model is None:
                model = detected_model
        elif backend == "local" and model is None:
            _, model = detect_index()

        api_key = None
        if backend == "remote":
            remote_url = _resolve_remote_url(remote_url)
            api_key = _resolve_remote_api_key(remote_api_key)
            model = model or "Qwen/Qwen3-VL-Embedding-8B"

        store = SentryStore(backend=backend, model=model if backend in ("local", "remote") else None)

        if store.get_stats()["total_chunks"] == 0:
            click.echo(
                "未找到已索引的视频。"
                "请先运行 `sentrysearch index <目录>`。"
            )
            return

        get_embedder(
            backend, model=model, quantize=quantize,
            base_url=remote_url if backend == "remote" else None,
            api_key=api_key if backend == "remote" else None,
        )

        if save_top is not None and save_top > n_results:
            n_results = save_top

        if verbose:
            click.echo(
                f"  [verbose] 后端={backend}, 图片={image}, "
                f"相似度阈值：{threshold}", err=True,
            )

        results = search_footage_by_image(
            image, store, n_results=n_results, verbose=verbose,
        )
        _present_results(results, threshold, trim, save_top, output_dir, overlay, verbose)

    except Exception as e:
        _handle_error(e)
    finally:
        reset_embedder()


# -----------------------------------------------------------------------
# shell
# -----------------------------------------------------------------------

_HISTORY_PATH = os.path.join(os.path.expanduser("~"), ".sentrysearch", "history")


def _print_shell_results(results, threshold):
    """在 shell 模式下打印搜索结果。"""
    if not results:
        click.echo("  （无结果）")
        return
    best = results[0]["similarity_score"]
    if best < threshold:
        click.secho(f"  （置信度较低 — 最高得分：{best:.2f}）", fg="yellow")
    for i, r in enumerate(results, 1):
        basename = os.path.basename(r["source_file"])
        click.echo(
            f"  #{i} [{r['similarity_score']:.2f}] {basename} "
            f"@ {_fmt_time(r['start_time'])}-{_fmt_time(r['end_time'])}"
        )


@cli.command()
@click.option("--backend", type=click.Choice(["gemini", "local", "remote"]), default=None,
              help="Embedding backend (auto-detected from index if omitted).")
@click.option("--model", default=None,
              help="Model for local backend (default: auto-detect from index).")
@click.option("--quantize/--no-quantize", default=None,
              help="Enable/disable 4-bit quantization for local backend.")
@click.option("--remote-url", default=None,
              help="URL of remote embedding API. Implies --backend remote.")
@click.option("--remote-api-key", default=None,
              help="API key for remote embedding service (or set REMOTE_EMBED_API_KEY).")
@click.option("-n", "--results", "n_results", default=5, show_default=True,
              help="每次查询的结果数量。")
@click.option("--threshold", default=0.41, show_default=True, type=float,
              help="视为可信匹配的最低相似度分数。")
@click.option("--verbose", is_flag=True, help="显示调试信息。")
def shell(backend, model, quantize, remote_url, remote_api_key, n_results, threshold, verbose):
    """启动交互式搜索会话，保持模型加载状态。

    适用于使用本地后端连续执行多次查询的场景，
    否则每次 `search` 调用都会重新加载模型。

    元命令：
      :n <整数>  更改结果数量
      :help      显示帮助
      :quit      退出（Ctrl-D 也可用）
    """
    from .embedder import get_embedder, reset_embedder
    from .local_embedder import normalize_model_key
    from .search import search_footage
    from .store import SentryStore, detect_index

    try:
        # Resolve backend/model (mirrors `search`)
        if model is not None and backend is None:
            backend = "local"
        if remote_url is not None and backend is None:
            backend = "remote"
        if model is not None:
            model = normalize_model_key(model)
        if backend is None:
            detected_backend, detected_model = detect_index()
            backend = detected_backend or "gemini"
            if model is None:
                model = detected_model
        elif backend == "local" and model is None:
            _, model = detect_index()

        api_key = None
        if backend == "remote":
            remote_url = _resolve_remote_url(remote_url)
            api_key = _resolve_remote_api_key(remote_api_key)
            model = model or "Qwen/Qwen3-VL-Embedding-8B"

        store = SentryStore(backend=backend, model=model)
        stats = store.get_stats()
        if stats["total_chunks"] == 0:
            click.echo("未找到已索引的视频。请先运行 `sentrysearch index <目录>`。")
            return

        label = backend + (f" ({model})" if model else "")
        click.echo(f"正在加载 {label}...")
        get_embedder(
            backend, model=model, quantize=quantize,
            base_url=remote_url if backend == "remote" else None,
            api_key=api_key if backend == "remote" else None,
        )

        # Readline for arrow-key history and persistent history file
        try:
            import readline
            os.makedirs(os.path.dirname(_HISTORY_PATH), exist_ok=True)
            if os.path.exists(_HISTORY_PATH):
                try:
                    readline.read_history_file(_HISTORY_PATH)
                except OSError:
                    pass
            readline.set_history_length(1000)
        except ImportError:
            readline = None

        click.secho(
            f"就绪。已索引 {stats['total_chunks']} 个片段。"
            "输入查询，:help 查看命令，:quit 退出。",
            fg="green",
        )

        while True:
            try:
                query = input("search> ").strip()
            except EOFError:
                click.echo()
                break
            except KeyboardInterrupt:
                click.echo()
                continue
            if not query:
                continue

            if query.startswith(":"):
                cmd, _, arg = query[1:].partition(" ")
                cmd = cmd.strip().lower()
                arg = arg.strip()
                if cmd in ("q", "quit", "exit"):
                    break
                if cmd == "help":
                    click.echo(
                        ":n <整数>  设置结果数量（当前："
                        f"{n_results}）\n"
                        ":help      显示此帮助\n"
                        ":quit      退出"
                    )
                    continue
                if cmd == "n":
                    try:
                        new_n = int(arg)
                        if new_n < 1:
                            raise ValueError
                        n_results = new_n
                        click.echo(f"n_results = {n_results}")
                    except ValueError:
                        click.secho("用法：:n <正整数>", fg="yellow")
                    continue
                click.secho(f"未知命令：:{cmd}", fg="yellow")
                continue

            try:
                results = search_footage(
                    query, store, n_results=n_results, verbose=verbose,
                )
            except Exception as e:
                click.secho(f"错误：{e}", fg="red")
                continue

            _print_shell_results(results, threshold)

        if readline is not None:
            try:
                readline.write_history_file(_HISTORY_PATH)
            except OSError:
                pass

    except Exception as e:
        _handle_error(e)
    finally:
        reset_embedder()


# -----------------------------------------------------------------------
# overlay
# -----------------------------------------------------------------------

@cli.command()
@click.argument("video", type=click.Path(exists=True, dir_okay=False))
@click.option("-o", "--output", default=None,
              help="输出路径（默认 <视频>_overlay.mp4）。")
def overlay(video, output):
    """将 Tesla 遥测叠加层应用到 VIDEO 文件（用于测试）。"""
    from .chunker import _get_video_duration

    video = os.path.abspath(video)
    if output is None:
        output = _overlay_output_path(video)

    try:
        duration = _get_video_duration(video)
    except Exception as e:
        _handle_error(e)

    success = _apply_overlay_to_clip(
        video, video, 0.0, duration, replace=False,
    )
    if success:
        overlay_path = _overlay_output_path(video)
        if output != overlay_path and os.path.isfile(overlay_path):
            os.replace(overlay_path, output)
        click.secho(f"已保存：{output}", fg="green")
        _cache_last_clip(output)
        _open_file(output)
    else:
        raise SystemExit(1)


# -----------------------------------------------------------------------
# stats
# -----------------------------------------------------------------------

@cli.command()
def stats():
    """打印索引统计信息。"""
    from .store import SentryStore, detect_index

    backend, model = detect_index()
    if backend is None:
        backend = "gemini"
    store = SentryStore(backend=backend, model=model)
    s = store.get_stats()

    if s["total_chunks"] == 0:
        click.echo("索引为空。请先运行 `sentrysearch index <目录>`。")
        return

    click.echo(f"总片段数：    {s['total_chunks']}")
    click.echo(f"源文件数：    {s['unique_source_files']}")
    backend_label = store.get_backend()
    if model:
        backend_label += f" ({model})"
    click.echo(f"后端：        {backend_label}")
    click.echo("\n已索引的文件：")
    for f in s["source_files"]:
        exists = os.path.exists(f)
        label = "" if exists else "  [缺失]"
        click.echo(f"  {f}{label}")


# -----------------------------------------------------------------------
# reset
# -----------------------------------------------------------------------

@cli.command()
@click.option("--backend", type=click.Choice(["gemini", "local", "remote"]), default=None,
              help="Backend to reset (auto-detected if omitted).")
@click.option("--model", default=None,
              help="Model to reset (auto-detected if omitted). Implies --backend local.")
@click.confirmation_option(prompt="这将删除所有已索引的数据。是否继续？")
def reset(backend, model):
    """删除所有已索引的数据。"""
    from .store import SentryStore, detect_index

    if model is not None and backend is None:
        backend = "local"
    if backend is None:
        backend, detected_model = detect_index()
        backend = backend or "gemini"
        if model is None:
            model = detected_model

    store = SentryStore(backend=backend, model=model)
    s = store.get_stats()

    if s["total_chunks"] == 0:
        click.echo("索引已经为空。")
        return

    for f in s["source_files"]:
        store.remove_file(f)

    click.echo(f"已从 {s['unique_source_files']} 个文件中删除 {s['total_chunks']} 个片段。")


# -----------------------------------------------------------------------
# remove
# -----------------------------------------------------------------------

@cli.command()
@click.argument("files", nargs=-1, required=True)
@click.option("--backend", type=click.Choice(["gemini", "local", "remote"]), default=None,
              help="要从中删除的后端（省略时自动检测）。")
@click.option("--model", default=None,
              help="要从中删除的模型（省略时自动检测）。隐含 --backend local。")
def remove(files, backend, model):
    """从索引中删除指定文件。

    接受完整路径或匹配已索引文件路径的子字符串。
    """
    from .store import SentryStore, detect_index

    if model is not None and backend is None:
        backend = "local"
    if backend is None:
        backend, detected_model = detect_index()
        backend = backend or "gemini"
        if model is None:
            model = detected_model

    store = SentryStore(backend=backend, model=model)
    s = store.get_stats()

    if s["total_chunks"] == 0:
        click.echo("索引为空。")
        return

    total_removed = 0
    for pattern in files:
        # 匹配已索引的源文件（子字符串匹配）
        matches = [f for f in s["source_files"] if pattern in f]
        if not matches:
            click.echo(f"没有匹配 '{pattern}' 的已索引文件")
            continue
        for source_file in matches:
            removed = store.remove_file(source_file)
            click.echo(f"已从 {source_file} 删除 {removed} 个片段")
            total_removed += removed

    if total_removed:
        click.echo(f"\n总计：已删除 {total_removed} 个片段。")


# -----------------------------------------------------------------------
# dlq
# -----------------------------------------------------------------------

@cli.group()
def dlq():
    """查看或清空失败片段的死信队列。"""


@dlq.command("list")
def dlq_list():
    """显示嵌入失败的片段。"""
    from datetime import datetime

    from .dlq import DeadLetterQueue

    q = DeadLetterQueue()
    entries = q.entries()
    if not entries:
        click.echo("DLQ 为空。")
        return

    click.echo(f"DLQ 中有 {len(entries)} 个片段：\n")
    for chunk_id, info in sorted(
        entries.items(), key=lambda kv: kv[1]["last_attempt"]
    ):
        ts = datetime.fromtimestamp(info["last_attempt"]).strftime("%Y-%m-%d %H:%M:%S")
        basename = os.path.basename(info["source_file"])
        click.echo(
            f"  {chunk_id}  {basename} "
            f"@ {_fmt_time(info['start_time'])}-{_fmt_time(info['end_time'])}  "
            f"(attempts={info['attempts']}, last={ts})"
        )
        click.echo(f"    错误：{info['error']}")
    click.echo(
        "\n使用以下命令重试：sentrysearch index <目录> --retry-failed"
    )


@dlq.command("clear")
@click.confirmation_option(prompt="清除所有 DLQ 条目？")
def dlq_clear():
    """从死信队列中删除所有条目。"""
    from .dlq import DeadLetterQueue

    q = DeadLetterQueue()
    count = q.clear()
    click.echo(f"已清除 {count} 个 DLQ 条目。")
