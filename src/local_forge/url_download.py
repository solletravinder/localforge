from __future__ import annotations

import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

USER_AGENT = "LocalForge/0.5.0"
ALLOWED_SUFFIXES = {".gguf", ".litertlm"}


def normalize_model_url(url: str) -> str:
    url = url.strip()
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Enter a direct http(s) model file URL")
    if parsed.netloc.lower().endswith("huggingface.co") and "/blob/" in parsed.path:
        path = parsed.path.replace("/blob/", "/resolve/", 1)
        parsed = parsed._replace(path=path)
    return urllib.parse.urlunsplit(parsed)


def safe_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def filename_from_url(url: str, provided: str | None = None) -> str:
    if provided:
        name = Path(provided).name
    else:
        parsed = urllib.parse.urlsplit(url)
        name = Path(urllib.parse.unquote(parsed.path)).name
    name = re.sub(r"[\x00-\x1f]", "", name).strip()
    if not name or name in {".", ".."}:
        raise ValueError(
            "Could not determine a filename from the URL. "
            "Use a direct link ending in .gguf or .litertlm."
        )
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(
            "URL downloads currently support direct .gguf or .litertlm files. "
            "For Hugging Face, use the file's /resolve/ download link."
        )
    return name


def engine_for_file(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".gguf":
        return "llama.cpp"
    if suffix == ".litertlm":
        return "litert"
    raise ValueError("Only GGUF and LiteRT files can be downloaded by URL")


def download_url(
    url: str,
    target_dir: Path,
    filename: str | None = None,
) -> Path:
    normalized = normalize_model_url(url)
    target_dir.mkdir(parents=True, exist_ok=True)
    name = filename_from_url(normalized, filename)
    target = target_dir / name
    part = target.with_name(name + ".part")
    resume_at = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": USER_AGENT, "Accept": "application/octet-stream,*/*"}
    if resume_at:
        headers["Range"] = f"bytes={resume_at}-"
    request = urllib.request.Request(normalized, headers=headers, method="GET")
    print(f"Downloading {name} from {urllib.parse.urlsplit(normalized).netloc} ...")
    try:
        response = urllib.request.urlopen(request, timeout=90)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and part.exists():
            part.unlink(missing_ok=True)
            return download_url(normalized, target_dir, name)
        raise RuntimeError(
            f"Download failed for {safe_url(normalized)}: HTTP {exc.code} {exc.reason}"
        ) from exc
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"Download failed for {safe_url(normalized)}: {type(exc).__name__}"
        ) from exc

    status = getattr(response, "status", 200)
    content_length = response.headers.get("Content-Length")
    total = int(content_length) if content_length and content_length.isdigit() else None
    append = status == 206 and resume_at > 0
    if append and total is not None:
        total += resume_at
    if not append and resume_at:
        part.unlink(missing_ok=True)
        resume_at = 0
    mode = "ab" if append else "wb"
    downloaded = resume_at
    last_report = 0.0
    try:
        with response, part.open(mode) as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 2.0:
                    last_report = now
                    if total:
                        percent = min(100, downloaded * 100 / total)
                        print(
                            f"[url] {downloaded / 1_000_000:.1f} MB / "
                            f"{total / 1_000_000:.1f} MB ({percent:.0f}%)"
                        )
                    else:
                        print(f"[url] {downloaded / 1_000_000:.1f} MB")
    except (OSError, urllib.error.URLError) as exc:
        raise RuntimeError(
            f"Download interrupted for {safe_url(normalized)}: {type(exc).__name__}"
        ) from exc

    if total is not None and downloaded < total:
        raise RuntimeError(
            f"Download incomplete: {downloaded} of {total} bytes. Retry to resume."
        )
    if part.stat().st_size == 0:
        part.unlink(missing_ok=True)
        raise RuntimeError("The server returned an empty file.")
    part.replace(target)
    print(f"Saved {name} ({target.stat().st_size / 1_000_000:.1f} MB)")
    return target


def disk_free_gb(path: Path) -> float:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free / 1e9
