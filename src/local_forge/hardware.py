from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from pathlib import Path

from .config import AppConfig


def _powershell(script: str, timeout: int = 8) -> str:
    if os.name != "nt":
        return ""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=flags,
        )
        return (result.stdout or result.stderr or "").strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _ram_gb() -> float:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        try:
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / (1024**3), 1)
        except (AttributeError, OSError):
            return 0.0
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return round(int(line.split()[1]) / (1024**2), 1)
    except OSError:
        pass
    return 0.0


def find_llama_server(configured: str = "") -> str | None:
    if configured:
        path = Path(configured).expanduser()
        if path.exists():
            return str(path)
    found = shutil.which("llama-server")
    if found:
        return found
    candidates: list[Path] = []
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        winget = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        candidates.extend(winget.glob("ggml.llamacpp*/llama-server.exe"))
    home = Path.home()
    candidates.extend(
        [
            home / "llama.cpp" / "build" / "bin" / "llama-server.exe",
            home / "llama.cpp" / "llama-server.exe",
            Path("C:/llama.cpp/build/bin/llama-server.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def llama_server_version(path: str | None) -> str:
    if not path:
        return "not found"
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        text = (result.stdout or "") + (result.stderr or "")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[0] if lines else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def openvino_devices() -> list[str]:
    try:
        import openvino as ov

        core = ov.Core()
        return list(core.get_available_devices())
    except Exception:
        return []


def detect_hardware(config: AppConfig) -> dict:
    cpu = platform.processor() or _powershell(
        "Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name"
    )
    gpu_text = _powershell(
        "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"
    )
    gpus = [line.strip() for line in gpu_text.splitlines() if line.strip()]
    server = find_llama_server(config.llama_server)
    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu or "unknown",
        "cpu_cores": os.cpu_count() or 0,
        "ram_gb": _ram_gb(),
        "gpus": gpus,
        "openvino_devices": openvino_devices(),
        "llama_server": server,
        "llama_server_version": llama_server_version(server),
        "python": platform.python_version(),
    }


def format_gb(value: float) -> str:
    return f"{value:.2f} GB"


def directory_size(path: Path) -> float:
    total = 0
    if path.is_file():
        return path.stat().st_size / 1e9
    if not path.exists():
        return 0.0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total / 1e9
