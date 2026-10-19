from __future__ import annotations

import ctypes
import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

APP_TITLE = "LocalForge"
APP_VERSION = "0.4.1"
BG = "#0b0f14"
PANEL = "#141a23"
PANEL_2 = "#10151d"
BORDER = "#232b38"
TEXT = "#eef2f7"
MUTED = "#8b98aa"
DIM = "#5f6b7c"
AMBER = "#f6a82f"
AMBER_HOVER = "#ffbf57"
GREEN = "#4ade80"
BLUE = "#7cb7ff"
RED = "#f87171"
MONO = "Consolas"
UI = "Segoe UI"

FALLBACK_MODELS = [
    {
        "id": "qwen3.5-4b-abliterated",
        "name": "Qwen3.5 4B Abliterated (GGUF Q4_K_M)",
        "engine": "llama.cpp",
        "format": "gguf",
        "installed": True,
        "size_gb": 3.07,
        "expected_size_gb": 3.3,
        "modalities": ["text", "image"],
        "license": "Apache-2.0",
        "local": False,
        "path": "",
        "notes": "",
    },
    {
        "id": "gemma4-e2b-uncensored-ov",
        "name": "Gemma 4 E2B Uncensored (OpenVINO INT4)",
        "engine": "openvino",
        "format": "openvino",
        "installed": True,
        "size_gb": 4.28,
        "expected_size_gb": 4.28,
        "modalities": ["text", "image"],
        "license": "Apache-2.0",
        "local": False,
        "path": "",
        "notes": "",
    },
    {
        "id": "qwen3-0.6b-abliterated",
        "name": "Qwen3 0.6B Abliterated (GGUF Q4_K_M, sanity check)",
        "engine": "llama.cpp",
        "format": "gguf",
        "installed": False,
        "size_gb": 0.4,
        "expected_size_gb": 0.4,
        "modalities": ["text"],
        "license": "Apache-2.0",
        "local": False,
        "path": "",
        "notes": "",
        "context_length": 8192,
    },
]
KNOWN_SHORT_NAMES = {
    "qwen3.5-4b-abliterated": "Qwen3.5-4B",
    "gemma4-e2b-uncensored-ov": "Gemma 4 E2B",
    "lfm2.5-2.6b-abliterated": "LFM2.5 2.6B",
    "qwen3-0.6b-abliterated": "Qwen3 0.6B Sanity",
}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def model_short_name(record: dict) -> str:
    model_id = str(record.get("id") or "model")
    short = KNOWN_SHORT_NAMES.get(model_id)
    if short:
        return short
    name = str(record.get("name") or model_id)
    return name if len(name) <= 34 else name[:33] + "…"


def model_display_label(record: dict) -> str:
    status = "installed" if record.get("installed") else "not installed"
    return f"{model_short_name(record)} · {status}"


def engine_label(engine: str) -> str:
    return {"llama.cpp": "llama.cpp", "openvino": "OpenVINO", "litert": "LiteRT"}.get(
        engine, engine
    )


def format_model_size(record: dict) -> str:
    installed = bool(record.get("installed"))
    value = record.get("size_gb") if installed else record.get("expected_size_gb")
    try:
        size = float(value or 0)
    except (TypeError, ValueError):
        size = 0.0
    if size <= 0:
        return "—"
    return f"{size:.2f} GB" if installed else f"~{size:.2f} GB"


def models_json(app_dir: Path) -> list[dict]:
    python = runtime_python(app_dir)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [str(python), "-u", "-m", "local_forge", "models", "list", "--json"],
        cwd=str(app_dir),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=90,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    output = (result.stdout or "").strip()
    start = output.find("[")
    if start < 0:
        detail = (result.stderr or output or "no output").strip()
        raise RuntimeError(f"Model list failed: {detail[-500:]}")
    data = json.loads(output[start:])
    if not isinstance(data, list):
        raise RuntimeError("Model list returned an unexpected response.")
    return [item for item in data if isinstance(item, dict)]


def resource_path(name: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / "assets" / name


def find_app_dir() -> Path | None:
    starts = [Path(sys.executable).resolve().parent]
    if not getattr(sys, "frozen", False):
        starts.append(Path(__file__).resolve().parent)
    seen: set[Path] = set()
    for start in starts:
        for candidate in (start, start.parent, start.parent.parent):
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            has_venv = (candidate / ".venv" / "Scripts" / "python.exe").is_file()
            has_portable = (
                (candidate / "portable.flag").is_file()
                and (candidate / "runtime" / "python" / "python.exe").is_file()
            )
            if has_venv or has_portable:
                return candidate
    return None


def runtime_python(app_dir: Path) -> Path:
    portable = app_dir / "runtime" / "python" / "python.exe"
    if portable.is_file():
        return portable
    return app_dir / ".venv" / "Scripts" / "python.exe"


def configure_portable_environment() -> Path | None:
    starts = [Path(sys.executable).resolve().parent]
    if not getattr(sys, "frozen", False):
        starts.append(Path(__file__).resolve().parent)
    seen: set[Path] = set()
    for start in starts:
        for candidate in (start, start.parent, start.parent.parent):
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if not (
                (candidate / "portable.flag").is_file()
                and (candidate / "runtime" / "python" / "python.exe").is_file()
            ):
                continue
            data = candidate / "data"
            models = candidate / "models"
            app_dir = candidate / "runtime" / "app"
            data.mkdir(parents=True, exist_ok=True)
            models.mkdir(parents=True, exist_ok=True)
            os.environ["LOCALFORGE_HOME"] = str(data)
            os.environ["LOCALFORGE_MODELS_DIR"] = str(models)
            existing_pythonpath = os.environ.get("PYTHONPATH", "")
            os.environ["PYTHONPATH"] = (
                str(app_dir)
                if not existing_pythonpath
                else str(app_dir) + os.pathsep + existing_pythonpath
            )
            os.environ["PYTHONNOUSERSITE"] = "1"
            return candidate
    return None


def find_jan() -> Path | None:
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    candidates = [
        local / "Programs" / "Jan" / "Jan-Desktop.exe",
        local / "Programs" / "Jan" / "Jan.exe",
        program_files / "Jan" / "Jan.exe",
        program_files / "Jan" / "Jan-Desktop.exe",
    ]
    return next((path for path in candidates if path.is_file()), None)


def config_path() -> Path:
    home = os.environ.get("LOCALFORGE_HOME")
    if home:
        return Path(home).expanduser() / "config.json"
    appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    return appdata / "LocalForge" / "config.json"


def load_config() -> dict:
    try:
        data = json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    models_dir = os.environ.get("LOCALFORGE_MODELS_DIR")
    if models_dir:
        data["models_dir"] = models_dir
    return data


def free_ram_gb() -> float | None:
    class MEMORYSTATUSEX(ctypes.Structure):
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

    try:
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys / (1024**3)
    except Exception:
        pass
    return None


def server_health(port: int, timeout: float = 1.0) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                return data if isinstance(data, dict) else None
    except (OSError, ValueError, urllib.error.URLError):
        return None
    return None


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.35)
        return sock.connect_ex(("127.0.0.1", port)) == 0


class ServerController:
    def __init__(self, app_dir: Path, port: int):
        self.app_dir = app_dir
        self.port = port
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()
        self.proc: subprocess.Popen[str] | None = None
        self._stop_requested = threading.Event()
        self._lock = threading.Lock()

    def _put(self, kind: str, value: str) -> None:
        self.events.put((kind, value))

    def _state(self, state: str, detail: str) -> None:
        self._put("state", f"{state}\x1f{detail}")

    def is_active(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, model_id: str) -> None:
        with self._lock:
            if self.is_active():
                return
            self._stop_requested.clear()
            health = server_health(self.port)
            if health:
                self._put("log", "An existing LocalForge server answered on the configured port.")
                self._state("external", f"Existing LocalForge server on 127.0.0.1:{self.port}")
                return
            if port_in_use(self.port):
                self._state("error", f"Port {self.port} is already in use by another program.")
                self._put("log", f"Cannot start: 127.0.0.1:{self.port} is occupied.")
                return

            python = runtime_python(self.app_dir)
            command = [
                str(python),
                "-u",
                "-m",
                "local_forge",
                "serve",
                "--preload",
                "--model",
                model_id,
            ]
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONIOENCODING"] = "utf-8"
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._put("log", f"Starting {model_id} with the saved LocalForge config...")
            try:
                self.proc = subprocess.Popen(
                    command,
                    cwd=str(self.app_dir),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=creationflags,
                )
            except OSError as exc:
                self._state("error", f"Could not start the local server: {exc}")
                self._put("log", f"Start failed: {exc}")
                return
            self._state("starting", "Loading the model and warming up. This usually takes 5–15 seconds.")
            threading.Thread(target=self._read_output, daemon=True).start()
            threading.Thread(target=self._watch_startup, daemon=True).start()

    def _read_output(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for raw_line in proc.stdout:
            line = ANSI_RE.sub("", raw_line).rstrip()
            if line:
                self._put("log", line)
        self._put("log", "Server output stream closed.")

    def _watch_startup(self) -> None:
        deadline = time.time() + 300
        while time.time() < deadline:
            if self._stop_requested.is_set():
                return
            proc = self.proc
            if proc is None:
                return
            code = proc.poll()
            if code is not None:
                if not self._stop_requested.is_set():
                    self._state("error", f"The server exited before it was ready (code {code}).")
                return
            health = server_health(self.port)
            if health and health.get("app") == "localforge":
                self._state("running", "Model loaded and warm. Ready for Jan and the local API.")
                return
            time.sleep(0.4)
        if not self._stop_requested.is_set():
            self._state("error", "The server did not become ready within 5 minutes.")

    def stop(self, wait: bool = False) -> None:
        if self.proc is None:
            self._state("stopped", "Server is not running.")
            return
        self._stop_requested.set()
        self._state("stopping", "Unloading the model and freeing memory...")

        def work() -> None:
            proc = self.proc
            if proc is None:
                self._state("stopped", "Server is not running.")
                return
            if proc.poll() is None:
                try:
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{self.port}/admin/unload",
                        data=b"",
                        method="POST",
                    )
                    with urllib.request.urlopen(request, timeout=10):
                        pass
                except (OSError, ValueError, urllib.error.URLError):
                    pass
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        subprocess.run(
                            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        )
            else:
                proc.wait(timeout=1)
            self._state("stopped", "Server stopped. Model memory released.")

        if wait:
            work()
        else:
            threading.Thread(target=work, daemon=True).start()


class LocalForgeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.cfg = load_config()
        self.port = int(self.cfg.get("api_port", 8090) or 8090)
        self.app_dir = find_app_dir()
        self.controller = ServerController(self.app_dir, self.port) if self.app_dir else None
        self.state = "error" if self.app_dir is None else "stopped"
        self.detail = (
            "The LocalForge runtime was not found. Keep the EXE inside the local-forge or portable folder."
            if self.app_dir is None
            else "Server is not running. Start it to use Jan or the local API."
        )
        self.external = False
        self._closing = False
        self._header_icon: tk.PhotoImage | None = None
        self.model_records: list[dict] = [dict(item) for item in FALLBACK_MODELS]
        self.model_by_id: dict[str, dict] = {
            str(item.get("id")): item for item in self.model_records
        }
        self.model_display_by_id: dict[str, str] = {}
        self.models_loading = False
        self._models_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.models_window: ModelsWindow | None = None

        self._configure_window()
        self._build_ui()
        self._set_state(self.state, self.detail)
        self._update_metrics()
        self._refresh_ram()
        self._poll_events()

        if self.app_dir is None:
            self.append_log("ERROR: .venv\\Scripts\\python.exe was not found near this EXE.")
            self.append_log(f"EXE location: {Path(sys.executable).resolve()}")
        else:
            self.append_log("LocalForge desktop ready.")
            self.append_log(
                f"Config: ctx {int(self.cfg.get('llama_ctx', 131072)):,} · "
                f"{self.cfg.get('llama_ngl', 10)}/32 GPU layers · "
                f"K {self.cfg.get('llama_cache_k', 'q8_0')} / "
                f"V {self.cfg.get('llama_cache_v', 'q4_0')}"
            )
            health = server_health(self.port)
            if health and health.get("app") == "localforge":
                self.external = True
                self.controller = None
                self._set_state("external", f"Existing LocalForge server on 127.0.0.1:{self.port}")
                self.append_log("Attached to an existing LocalForge server on the configured port.")
            self._start_models_refresh()

    def _configure_window(self) -> None:
        self.root.title(f"{APP_TITLE} · local AI control")
        width, height = 920, 720
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2 - 20)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(820, 640)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        try:
            ico = resource_path("localforge.ico")
            if ico.is_file():
                self.root.iconbitmap(default=str(ico))
        except tk.TclError:
            pass

    def _build_ui(self) -> None:
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)

        self._build_header()
        self._build_status_card()
        self._build_log_card()

        footer = tk.Frame(self.root, bg=BG)
        footer.grid(row=3, sticky="ew", padx=24, pady=(4, 14))
        footer.grid_columnconfigure(0, weight=1)
        tk.Label(
            footer,
            text="Server is on demand: closing this window stops the API and frees the model.",
            bg=BG,
            fg=DIM,
            font=(UI, 9),
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            footer,
            text=f"{APP_TITLE} {APP_VERSION}",
            bg=BG,
            fg=DIM,
            font=(UI, 9),
        ).grid(row=0, column=1, sticky="e")

    def _build_header(self) -> None:
        header = tk.Frame(self.root, bg=BG)
        header.grid(row=0, sticky="ew", padx=24, pady=(18, 12))
        header.grid_columnconfigure(1, weight=1)

        try:
            png = resource_path("localforge.png")
            if png.is_file():
                self._header_icon = tk.PhotoImage(file=str(png)).subsample(8, 8)
                tk.Label(header, image=self._header_icon, bg=BG, bd=0).grid(
                    row=0, column=0, rowspan=2, sticky="w", padx=(0, 13)
                )
        except tk.TclError:
            pass

        tk.Label(
            header,
            text=APP_TITLE,
            bg=BG,
            fg=TEXT,
            font=(UI, 20, "bold"),
            anchor="w",
        ).grid(row=0, column=1, sticky="sw")
        tk.Label(
            header,
            text="Uncensored local AI · 131k context · on demand",
            bg=BG,
            fg=MUTED,
            font=(UI, 9),
            anchor="w",
        ).grid(row=1, column=1, sticky="nw", pady=(1, 0))

        self.header_pill = tk.Label(
            header,
            text="STOPPED",
            bg=PANEL_2,
            fg=MUTED,
            padx=12,
            pady=6,
            font=(UI, 8, "bold"),
        )
        self.header_pill.grid(row=0, column=2, rowspan=2, sticky="e")

    def _build_status_card(self) -> None:
        card = tk.Frame(
            self.root,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightcolor=BORDER,
            highlightthickness=1,
            bd=0,
        )
        card.grid(row=1, sticky="ew", padx=24, pady=(0, 12))
        card.grid_columnconfigure(0, weight=1)
        inner = tk.Frame(card, bg=PANEL)
        inner.grid(row=0, column=0, sticky="ew", padx=18, pady=16)
        inner.grid_columnconfigure(1, weight=1)

        self.dot = tk.Canvas(inner, width=12, height=12, bg=PANEL, highlightthickness=0)
        self.dot.grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._dot_id = self.dot.create_oval(2, 2, 10, 10, fill=DIM, outline="")

        self.state_label = tk.Label(
            inner,
            text="Stopped",
            bg=PANEL,
            fg=TEXT,
            font=(UI, 16, "bold"),
            anchor="w",
        )
        self.state_label.grid(row=0, column=1, sticky="w")

        self.detail_label = tk.Label(
            inner,
            text=self.detail,
            bg=PANEL,
            fg=MUTED,
            font=(UI, 9),
            anchor="w",
            justify="left",
            wraplength=810,
        )
        self.detail_label.grid(row=1, column=1, sticky="w", pady=(2, 8))

        url_row = tk.Frame(inner, bg=PANEL)
        url_row.grid(row=2, column=1, sticky="w")
        self.url_label = tk.Label(
            url_row,
            text=f"http://127.0.0.1:{self.port}/v1",
            bg=PANEL,
            fg=BLUE,
            font=(MONO, 9),
            cursor="hand2",
        )
        self.url_label.pack(side="left")
        self.url_label.bind("<Button-1>", lambda _event: self.copy_url())
        copy_button = tk.Button(
            url_row,
            text="copy",
            command=self.copy_url,
            bg=PANEL,
            fg=DIM,
            activebackground=PANEL,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            padx=6,
            font=(UI, 8),
            cursor="hand2",
        )
        copy_button.pack(side="left", padx=(4, 0))

        sep = tk.Frame(inner, bg=BORDER, height=1)
        sep.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 12))

        metrics = tk.Frame(inner, bg=PANEL)
        metrics.grid(row=4, column=0, columnspan=2, sticky="ew")
        for index in range(4):
            metrics.grid_columnconfigure(index, weight=1)
        self.metric_vars = {
            "model": tk.StringVar(),
            "context": tk.StringVar(),
            "offload": tk.StringVar(),
            "ram": tk.StringVar(),
        }
        labels = [("MODEL", "model"), ("CONTEXT", "context"), ("ACCELERATION", "offload"), ("RAM FREE", "ram")]
        for index, (heading, key) in enumerate(labels):
            cell = tk.Frame(metrics, bg=PANEL)
            cell.grid(row=0, column=index, sticky="w", padx=(0, 12))
            tk.Label(
                cell,
                text=heading,
                bg=PANEL,
                fg=DIM,
                font=(UI, 7, "bold"),
                anchor="w",
            ).pack(anchor="w")
            tk.Label(
                cell,
                textvariable=self.metric_vars[key],
                bg=PANEL,
                fg=TEXT,
                font=(UI, 10, "bold"),
                anchor="w",
            ).pack(anchor="w", pady=(3, 0))

        sep2 = tk.Frame(inner, bg=BORDER, height=1)
        sep2.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(14, 14))

        controls = tk.Frame(inner, bg=PANEL)
        controls.grid(row=6, column=0, columnspan=2, sticky="ew")
        controls.grid_columnconfigure(1, weight=1)

        tk.Label(
            controls,
            text="Model",
            bg=PANEL,
            fg=MUTED,
            font=(UI, 9),
        ).grid(row=0, column=0, sticky="w", padx=(0, 8))

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Forge.TCombobox",
            fieldbackground=PANEL_2,
            background=PANEL_2,
            foreground=TEXT,
            arrowcolor=MUTED,
            bordercolor=BORDER,
            lightcolor=PANEL_2,
            darkcolor=PANEL_2,
            padding=6,
        )
        style.map(
            "Forge.TCombobox",
            fieldbackground=[("readonly", PANEL_2), ("disabled", PANEL_2)],
            foreground=[("readonly", TEXT), ("disabled", DIM)],
            selectbackground=[("readonly", PANEL_2)],
            selectforeground=[("readonly", TEXT)],
            arrowcolor=[("disabled", DIM)],
        )
        self.model_combo = ttk.Combobox(
            controls,
            values=[model_display_label(item) for item in self.model_records],
            state="readonly",
            style="Forge.TCombobox",
            font=(UI, 9),
            width=42,
        )
        self.model_display_by_id = {
            model_display_label(item): str(item.get("id")) for item in self.model_records
        }
        self.model_combo.current(0)
        self.model_combo.grid(row=0, column=1, sticky="w")
        self.model_combo.bind("<<ComboboxSelected>>", lambda _event: self._update_metrics())

        self.btn_start = self._button(
            controls,
            "Start Server",
            self.start,
            primary=True,
            width=13,
        )
        self.btn_start.grid(row=0, column=2, padx=(12, 8))
        self.btn_stop = self._button(
            controls,
            "Stop",
            self.stop,
            primary=False,
            width=8,
            disabled=True,
        )
        self.btn_stop.grid(row=0, column=3)

        actions = tk.Frame(inner, bg=PANEL)
        actions.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        self.btn_jan = self._button(actions, "Open Jan", self.open_jan, secondary=True, width=12, disabled=True)
        self.btn_jan.pack(side="left")
        self.btn_docs = self._button(actions, "API Docs", self.open_docs, secondary=True, width=11, disabled=True)
        self.btn_docs.pack(side="left", padx=(8, 0))
        self.btn_logs = self._button(actions, "Open Logs", self.open_logs, ghost=True, width=11)
        self.btn_logs.pack(side="left", padx=(8, 0))
        self.btn_models = self._button(actions, "Models", self.open_models, ghost=True, width=9)
        self.btn_models.pack(side="left", padx=(8, 0))
        self.btn_settings = self._button(actions, "Settings", self.open_settings, ghost=True, width=10)
        self.btn_settings.pack(side="left", padx=(8, 0))
        self.engine_hint = tk.Label(
            actions,
            text="Jan: qwen3.5-4b-abliterated · context 131000",
            bg=PANEL,
            fg=DIM,
            font=(UI, 8),
        )
        self.engine_hint.pack(side="right")

    def _build_log_card(self) -> None:
        card = tk.Frame(
            self.root,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightcolor=BORDER,
            highlightthickness=1,
            bd=0,
        )
        card.grid(row=2, sticky="nsew", padx=24, pady=(0, 8))
        card.grid_rowconfigure(1, weight=1)
        card.grid_columnconfigure(0, weight=1)

        header = tk.Frame(card, bg=PANEL)
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 6))
        header.grid_columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="ACTIVITY",
            bg=PANEL,
            fg=MUTED,
            font=(UI, 7, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        clear = tk.Button(
            header,
            text="clear",
            command=self.clear_log,
            bg=PANEL,
            fg=DIM,
            activebackground=PANEL,
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            font=(UI, 8),
            cursor="hand2",
        )
        clear.grid(row=0, column=1, sticky="e")

        text_frame = tk.Frame(card, bg=PANEL_2)
        text_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)

        self.log = tk.Text(
            text_frame,
            bg="#090d13",
            fg="#aeb9c9",
            insertbackground=TEXT,
            selectbackground="#2b3b52",
            relief="flat",
            bd=0,
            wrap="word",
            font=(MONO, 9),
            padx=10,
            pady=8,
            height=10,
            state="disabled",
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(
            text_frame,
            orient="vertical",
            command=self.log.yview,
            style="Forge.Vertical.TScrollbar",
        )
        style = ttk.Style(self.root)
        style.configure(
            "Forge.Vertical.TScrollbar",
            background=PANEL_2,
            troughcolor="#090d13",
            bordercolor="#090d13",
            arrowcolor=MUTED,
        )
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)

    def _button(
        self,
        parent: tk.Misc,
        text: str,
        command,
        *,
        primary: bool = False,
        secondary: bool = False,
        ghost: bool = False,
        width: int = 12,
        disabled: bool = False,
    ) -> tk.Button:
        if primary:
            bg, fg, active_bg, active_fg = AMBER, "#181006", AMBER_HOVER, "#181006"
        elif secondary:
            bg, fg, active_bg, active_fg = "#22304a", "#d5e5ff", "#2c3f60", TEXT
        elif ghost:
            bg, fg, active_bg, active_fg = PANEL_2, MUTED, BORDER, TEXT
        else:
            bg, fg, active_bg, active_fg = "#232b38", TEXT, "#2f3a4c", TEXT
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=active_bg,
            activeforeground=active_fg,
            disabledforeground=DIM,
            relief="flat",
            bd=0,
            highlightthickness=0,
            padx=10,
            pady=7,
            width=width,
            font=(UI, 9, "bold" if primary else "normal"),
            cursor="hand2",
            state="disabled" if disabled else "normal",
        )
        return button

    def _set_state(self, state: str, detail: str) -> None:
        if state == "external":
            self.external = True
        self.state = state
        self.detail = detail
        labels = {
            "stopped": ("Stopped", MUTED, "STOPPED", MUTED),
            "starting": ("Starting", AMBER, "STARTING", AMBER),
            "running": ("Running", GREEN, "RUNNING", GREEN),
            "stopping": ("Stopping", AMBER, "STOPPING", AMBER),
            "error": ("Attention", RED, "ERROR", RED),
            "external": ("Running", GREEN, "EXTERNAL", GREEN),
        }
        title, color, pill_text, pill_fg = labels.get(state, labels["error"])
        self.state_label.configure(text=title, fg=color)
        self.detail_label.configure(text=detail)
        self.header_pill.configure(text=pill_text, fg=pill_fg)
        self.dot.itemconfigure(self._dot_id, fill=color)

        proc_active = self.controller is not None and self.controller.is_active()
        busy = state in {"starting", "running", "stopping"} or proc_active
        start_enabled = not busy and not self.external
        self.btn_start.configure(
            state="normal" if start_enabled else "disabled",
            bg=AMBER if start_enabled else "#2a303b",
            fg="#181006" if start_enabled else DIM,
            activebackground=AMBER_HOVER if start_enabled else "#2a303b",
            cursor="hand2" if start_enabled else "arrow",
        )
        can_stop = proc_active and not self.external and state != "stopping"
        self.btn_stop.configure(
            state="normal" if can_stop else "disabled",
            bg="#22304a" if can_stop else "#1b2029",
            fg="#d5e5ff" if can_stop else DIM,
            activebackground="#2c3f60" if can_stop else "#1b2029",
            cursor="hand2" if can_stop else "arrow",
        )
        self.model_combo.configure(state="disabled" if busy or self.external else "readonly")
        live = state in {"running", "external"}
        self.btn_jan.configure(state="normal" if live else "disabled")
        self.btn_docs.configure(state="normal" if live else "disabled")
        if state == "external":
            self.engine_hint.configure(text="Existing server · close its original window to stop it")
        elif state == "error":
            self.engine_hint.configure(text="Check ACTIVITY for the failure details")
        else:
            self.engine_hint.configure(text="Jan: qwen3.5-4b-abliterated · context 131000")
        if (
            self.models_window is not None
            and self.models_window.winfo_exists()
        ):
            self.models_window._update_actions()

    def _update_metrics(self) -> None:
        if not hasattr(self, "metric_vars"):
            return
        selected = self.selected_model()
        record = self.model_by_id.get(selected, {})
        self.metric_vars["model"].set(model_short_name(record) if record else selected)
        engine = str(record.get("engine") or "llama.cpp")
        if engine == "llama.cpp":
            ctx = int(record.get("context_length") or self.cfg.get("llama_ctx", 131072) or 131072)
            ngl = self.cfg.get("llama_ngl", 10)
            cache_k = str(self.cfg.get("llama_cache_k", "q8_0")).split("_")[0].upper()
            cache_v = str(self.cfg.get("llama_cache_v", "q4_0")).split("_")[0].upper()
            self.metric_vars["context"].set(f"{ctx:,}")
            self.metric_vars["offload"].set(f"{ngl} / 32 GPU")
            self.engine_hint.configure(text=f"KV cache {cache_k} K · {cache_v} V · one slot")
        elif engine == "litert":
            ctx = int(self.cfg.get("litert_ctx", 4096) or 4096)
            backend = str(self.cfg.get("litert_backend", "GPU")).upper()
            self.metric_vars["context"].set(f"{ctx:,}")
            self.metric_vars["offload"].set(f"LiteRT {backend}")
            self.engine_hint.configure(text="LiteRT native runtime · validated bundles only")
        else:
            device = str(self.cfg.get("openvino_device", "CPU")).upper()
            self.metric_vars["context"].set("model default")
            self.metric_vars["offload"].set(f"OpenVINO {device}")
            self.engine_hint.configure(text=f"OpenVINO {device} · memory depends on model")

    def _refresh_ram(self) -> None:
        ram = free_ram_gb()
        if hasattr(self, "metric_vars"):
            self.metric_vars["ram"].set(f"{ram:.1f} GB" if ram is not None else "—")
        self.root.after(2000, self._refresh_ram)

    def _start_models_refresh(self) -> None:
        if self.app_dir is None or self.models_loading:
            return
        self.models_loading = True
        threading.Thread(target=self._models_fetch_worker, daemon=True).start()

    def _models_fetch_worker(self) -> None:
        try:
            records = models_json(self.app_dir)
        except Exception as exc:
            self._models_queue.put(("error", str(exc)))
            return
        self._models_queue.put(("models", records))

    def _apply_models(self, records: list[dict], notify: bool = True) -> None:
        if not records:
            self.models_loading = False
            return
        current = self.selected_model()
        self.model_records = [dict(item) for item in records]
        self.model_by_id = {str(item.get("id")): item for item in self.model_records}
        self.model_display_by_id = {}
        values: list[str] = []
        for record in self.model_records:
            label = model_display_label(record)
            if label in self.model_display_by_id:
                label = f"{label} · {record.get('id')}"
            self.model_display_by_id[label] = str(record.get("id"))
            values.append(label)
        self.model_combo.configure(values=values)
        selected_display = next(
            (label for label, model_id in self.model_display_by_id.items() if model_id == current),
            None,
        )
        current_record = self.model_by_id.get(current, {})
        installed = [item for item in self.model_records if item.get("installed")]
        if selected_display is None or (installed and not current_record.get("installed")):
            preferred = next(
                (item for item in installed if item.get("id") == "qwen3.5-4b-abliterated"),
                installed[0] if installed else self.model_records[0],
            )
            selected_display = next(
                (
                    label
                    for label, model_id in self.model_display_by_id.items()
                    if model_id == preferred.get("id")
                ),
                values[0],
            )
        self.model_combo.set(selected_display)
        self.models_loading = False
        self._update_metrics()
        if notify:
            installed_count = sum(1 for item in self.model_records if item.get("installed"))
            self.append_log(
                f"Models refreshed: {installed_count} installed of {len(self.model_records)} available."
            )

    def _poll_events(self) -> None:
        while True:
            try:
                kind, value = self._models_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "models":
                self._apply_models(value)  # type: ignore[arg-type]
            else:
                self.models_loading = False
                self.append_log(f"Model refresh failed: {value}")
        if self.controller is not None:
            for _ in range(60):
                try:
                    kind, value = self.controller.events.get_nowait()
                except queue.Empty:
                    break
                if kind == "log":
                    self.append_log(value)
                elif kind == "state":
                    state, _, detail = value.partition("\x1f")
                    self._set_state(state, detail)
                    if state == "stopped" and self._closing:
                        self.root.after(50, self.root.destroy)
                        return
        if self._closing and (self.controller is None or not self.controller.is_active()):
            self.root.after(50, self.root.destroy)
            return
        self.root.after(100, self._poll_events)

    def append_log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.configure(state="normal")
        self.log.insert("end", f"{stamp}  {message}\n")
        lines = int(self.log.index("end-1c").split(".")[0])
        if lines > 600:
            self.log.delete("1.0", f"{lines - 500}.0")
        self.log.configure(state="disabled")
        self.log.see("end")

    def clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def selected_model(self) -> str:
        selected = self.model_display_by_id.get(self.model_combo.get())
        if selected:
            return selected
        if self.model_records:
            return str(self.model_records[0].get("id") or "")
        return ""

    def start(self) -> None:
        if self.controller is None:
            messagebox.showerror(
                APP_TITLE,
                "The LocalForge environment was not found.\n"
                "Keep LocalForge.exe inside the local-forge folder.",
            )
            return
        model_id = self.selected_model()
        record = self.model_by_id.get(model_id)
        if record is not None and not record.get("installed"):
            messagebox.showerror(
                APP_TITLE,
                f"{model_short_name(record)} is not installed.\n\n"
                "Open Models, select it, and click Download first.",
            )
            return
        self.append_log(f"--- start requested: {model_id} ---")
        self._set_state("starting", "Loading the model and warming up. This usually takes 5–15 seconds.")
        self.controller.start(model_id)

    def stop(self) -> None:
        if self.controller is None or self.state == "stopping":
            return
        self.append_log("--- stop requested ---")
        self._set_state("stopping", "Unloading the model and freeing memory...")
        self.controller.stop()

    def copy_url(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(f"http://127.0.0.1:{self.port}/v1")
        self.append_log(f"Copied API base URL: http://127.0.0.1:{self.port}/v1")

    def open_docs(self) -> None:
        webbrowser.open(f"http://127.0.0.1:{self.port}/docs")

    def open_jan(self) -> None:
        jan = find_jan()
        if jan is None:
            messagebox.showerror(
                APP_TITLE,
                "Jan was not found. Install Jan or open it manually and use:\n"
                f"http://127.0.0.1:{self.port}/v1",
            )
            return
        try:
            os.startfile(str(jan))
            self.append_log(f"Opened Jan: {jan}")
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"Could not open Jan:\n{exc}")

    def open_logs(self) -> None:
        home = os.environ.get("LOCALFORGE_HOME")
        if home:
            log_dir = Path(home).expanduser() / "logs"
        else:
            appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
            log_dir = appdata / "LocalForge" / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            os.startfile(str(log_dir))
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"Could not open the log folder:\n{exc}")

    def open_settings(self) -> None:
        SettingsDialog(self)

    def open_models(self) -> None:
        if self.app_dir is None:
            messagebox.showerror(
                APP_TITLE,
                "The LocalForge environment was not found.\n"
                "Keep LocalForge.exe inside the local-forge folder.",
            )
            return
        if self.models_window is not None and self.models_window.winfo_exists():
            self.models_window.lift()
            self.models_window.focus_force()
            return
        self.models_window = ModelsWindow(self)

    def set_selected_model(self, model_id: str) -> bool:
        display = next(
            (
                label
                for label, selected_id in self.model_display_by_id.items()
                if selected_id == model_id
            ),
            None,
        )
        if display is None:
            return False
        self.model_combo.set(display)
        self._update_metrics()
        self.append_log(f"Selected model: {model_id}")
        return True

    def on_close(self) -> None:
        if self._closing:
            return
        if (
            self.models_window is not None
            and self.models_window.winfo_exists()
            and self.models_window.has_active_operation()
        ):
            if not messagebox.askyesno(
                APP_TITLE,
                "A model download or change is still running.\n\n"
                "Cancel it and exit?",
            ):
                return
            self.models_window.cancel_operation(silent=True)
        if self.external:
            self.root.destroy()
            return
        active = self.controller is not None and self.controller.is_active()
        if active:
            if not messagebox.askyesno(
                APP_TITLE,
                "The local server is running.\n\nStop it and free the model memory before exiting?",
            ):
                return
            self._closing = True
            self._set_state("stopping", "Unloading the model and freeing memory...")
            self.controller.stop()
            return
        self.root.destroy()


class SettingsDialog(tk.Toplevel):
    def __init__(self, app: LocalForgeApp):
        super().__init__(app.root)
        self.app = app
        self.title("LocalForge · settings")
        self.configure(bg=PANEL)
        self.resizable(False, False)
        self.transient(app.root)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Escape>", lambda _event: self.destroy())
        self._build()

    def _build(self) -> None:
        outer = tk.Frame(self, bg=PANEL, padx=22, pady=18)
        outer.pack(fill="both", expand=True)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(2, weight=1)

        tk.Label(
            outer,
            text="Settings",
            bg=PANEL,
            fg=TEXT,
            font=(UI, 15, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            outer,
            text="Saved to the LocalForge config. Changes apply the next time the server starts.",
            bg=PANEL,
            fg=MUTED,
            font=(UI, 9),
            anchor="w",
            wraplength=580,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        style = ttk.Style(self)
        style.configure(
            "Forge.TNotebook",
            background=PANEL,
            borderwidth=0,
            lightcolor=PANEL,
            darkcolor=PANEL,
            bordercolor=PANEL,
        )
        style.configure(
            "Forge.TNotebook.Tab",
            background=PANEL_2,
            foreground=MUTED,
            padding=(14, 7),
            font=(UI, 9, "bold"),
            borderwidth=0,
            lightcolor=PANEL_2,
            darkcolor=PANEL_2,
            bordercolor=BORDER,
        )
        style.map(
            "Forge.TNotebook.Tab",
            background=[("selected", PANEL), ("active", BORDER)],
            foreground=[("selected", AMBER), ("active", TEXT)],
        )

        notebook = ttk.Notebook(outer, style="Forge.TNotebook")
        notebook.grid(row=2, column=0, sticky="nsew", pady=(12, 10))
        generation = tk.Frame(notebook, bg=PANEL, padx=14, pady=10)
        runtime = tk.Frame(notebook, bg=PANEL, padx=14, pady=10)
        cache = tk.Frame(notebook, bg=PANEL, padx=14, pady=10)
        for page in (generation, runtime, cache):
            page.grid_columnconfigure(1, weight=1)
        notebook.add(generation, text="Generation")
        notebook.add(runtime, text="llama.cpp")
        notebook.add(cache, text="KV cache")

        self.max_tokens_var = tk.StringVar(value=str(self.app.cfg.get("max_new_tokens", 512)))
        self.temperature_var = tk.StringVar(value=str(self.app.cfg.get("temperature", 0.7)))
        self.top_p_var = tk.StringVar(value=str(self.app.cfg.get("top_p", 0.9)))
        self.disable_thinking_var = tk.BooleanVar(
            value=bool(self.app.cfg.get("disable_thinking", True))
        )
        self._row(
            generation,
            0,
            "Max output tokens",
            self.max_tokens_var,
            "Default used when the client does not set max_tokens.",
        )
        self._row(
            generation,
            1,
            "Temperature",
            self.temperature_var,
            "0.0 is deterministic; 2.0 is the most random.",
        )
        self._row(
            generation,
            2,
            "Top-p",
            self.top_p_var,
            "Nucleus sampling cutoff; 0.9 is the default.",
        )
        self._check(
            generation,
            3,
            "Disable model thinking",
            self.disable_thinking_var,
            "Stops Qwen reasoning tokens from consuming the output budget.",
        )

        self.context_var = tk.StringVar(value=str(self.app.cfg.get("llama_ctx", 131072)))
        self.layers_var = tk.StringVar(value=str(self.app.cfg.get("llama_ngl", 10)))
        self.batch_var = tk.StringVar(value=str(self.app.cfg.get("llama_batch", 2048)))
        self.ubatch_var = tk.StringVar(value=str(self.app.cfg.get("llama_ubatch", 1024)))
        self.threads_var = tk.StringVar(value=str(self.app.cfg.get("llama_threads", 0)))
        self.parallel_var = tk.StringVar(value=str(self.app.cfg.get("llama_parallel", 1)))
        self.flash_var = tk.StringVar(value=str(self.app.cfg.get("llama_flash_attn", "on")))
        self.warmup_var = tk.BooleanVar(value=bool(self.app.cfg.get("warmup", True)))
        self._row(
            runtime,
            0,
            "Context length",
            self.context_var,
            "Tested: 131,072 tokens. The KV cache is allocated at model load.",
        )
        self._row(
            runtime,
            1,
            "GPU layers",
            self.layers_var,
            "10 of 32 is the tested 70/30 CPU/iGPU split.",
        )
        self._row(runtime, 2, "Batch size", self.batch_var, "Logical prompt batch; 2048 default.")
        self._row(
            runtime,
            3,
            "Ubatch size",
            self.ubatch_var,
            "Must be <= batch. 1024 keeps the 131k cache lighter.",
        )
        self._row(
            runtime,
            4,
            "CPU threads",
            self.threads_var,
            "0 = automatic. On this i5, 8 is better than 12 for decode.",
        )
        self._row(
            runtime,
            5,
            "Parallel slots",
            self.parallel_var,
            "Keep at 1 on 16 GB; each extra slot adds overhead.",
        )
        self._combo_row(
            runtime,
            6,
            "Flash attention",
            self.flash_var,
            ["on", "off", "auto"],
            "on is recommended for Vulkan with quantized KV.",
        )
        self._check(
            runtime,
            7,
            "Warm up after load",
            self.warmup_var,
            "Absorbs Qwen's slow first request penalty.",
        )

        self.k_var = tk.StringVar(value=str(self.app.cfg.get("llama_cache_k", "q8_0")))
        self.v_var = tk.StringVar(value=str(self.app.cfg.get("llama_cache_v", "q4_0")))
        self._combo_row(
            cache,
            0,
            "K cache",
            self.k_var,
            ["q8_0", "f16", "q4_0"],
            "Q8_0 recommended. Q4 K reduces long-context and tool accuracy.",
        )
        self._combo_row(
            cache,
            1,
            "V cache",
            self.v_var,
            ["q4_0", "q8_0", "f16"],
            "Q4_0 saves memory; Q8_0 is higher fidelity.",
        )
        tk.Label(
            cache,
            text="At 131k tokens the KV cache is roughly 2.1 GB with Q8/Q8, 1.6 GB with "
            "Q8/Q4, and 4.0 GB at f16. A completely cold 131k prompt can take hours; "
            "later turns reuse the cached prefix.",
            bg=PANEL,
            fg=AMBER,
            font=(UI, 8),
            wraplength=560,
            justify="left",
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(14, 0))

        running = self.app.controller is not None and self.app.controller.is_active()
        if running:
            tk.Label(
                outer,
                text="The server is running. Use Stop and Start after saving to apply changes.",
                bg=PANEL,
                fg=BLUE,
                font=(UI, 9),
                wraplength=580,
                justify="left",
            ).grid(row=3, column=0, sticky="w", pady=(0, 4))

        buttons = tk.Frame(outer, bg=PANEL)
        buttons.grid(row=4, column=0, sticky="e", pady=(10, 0))
        self.app._button(buttons, "Cancel", self.destroy, ghost=True, width=9).pack(side="left")
        self.app._button(buttons, "Save", self.save, primary=True, width=9).pack(side="left", padx=(8, 0))

        self.update_idletasks()
        width = 640
        height = max(570, outer.winfo_reqheight() + 20)
        x = (self.winfo_screenwidth() - width) // 2
        y = (self.winfo_screenheight() - height) // 2
        self.geometry(f"{width}x{height}+{max(0, x)}+{max(0, y)}")

    def _row(
        self,
        page: tk.Misc,
        index: int,
        label: str,
        variable: tk.StringVar,
        hint: str,
    ) -> None:
        row = index * 2
        tk.Label(page, text=label, bg=PANEL, fg=MUTED, font=(UI, 9)).grid(
            row=row, column=0, sticky="w", pady=(5, 0)
        )
        entry = self._entry(page, variable)
        entry.grid(row=row, column=1, sticky="w", padx=(18, 0), pady=(5, 0))
        tk.Label(
            page,
            text=hint,
            bg=PANEL,
            fg=DIM,
            font=(UI, 8),
            wraplength=410,
            justify="left",
        ).grid(row=row + 1, column=1, sticky="w", padx=(18, 0))

    def _combo_row(
        self,
        page: tk.Misc,
        index: int,
        label: str,
        variable: tk.StringVar,
        values: list[str],
        hint: str,
    ) -> None:
        row = index * 2
        tk.Label(page, text=label, bg=PANEL, fg=MUTED, font=(UI, 9)).grid(
            row=row, column=0, sticky="w", pady=(5, 0)
        )
        combo = ttk.Combobox(
            page,
            textvariable=variable,
            values=values,
            state="readonly",
            style="Forge.TCombobox",
            font=(MONO, 9),
            width=14,
        )
        combo.grid(row=row, column=1, sticky="w", padx=(18, 0), pady=(5, 0))
        tk.Label(
            page,
            text=hint,
            bg=PANEL,
            fg=DIM,
            font=(UI, 8),
            wraplength=410,
            justify="left",
        ).grid(row=row + 1, column=1, sticky="w", padx=(18, 0))

    def _check(
        self,
        page: tk.Misc,
        index: int,
        text: str,
        variable: tk.BooleanVar,
        hint: str,
    ) -> None:
        row = index * 2
        tk.Checkbutton(
            page,
            text=text,
            variable=variable,
            bg=PANEL,
            fg=TEXT,
            selectcolor=AMBER,
            activebackground=PANEL,
            activeforeground=TEXT,
            highlightthickness=0,
            bd=0,
            anchor="w",
            font=(UI, 9),
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        tk.Label(
            page,
            text=hint,
            bg=PANEL,
            fg=DIM,
            font=(UI, 8),
            wraplength=560,
            justify="left",
        ).grid(row=row + 1, column=1, sticky="w", padx=(18, 0))

    @staticmethod
    def _entry(parent: tk.Misc, variable: tk.StringVar) -> tk.Entry:
        return tk.Entry(
            parent,
            textvariable=variable,
            bg=PANEL_2,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=AMBER,
            font=(MONO, 10),
            width=16,
        )

    def save(self) -> None:
        def as_int(variable: tk.StringVar, label: str, low: int, high: int) -> int:
            try:
                value = int(variable.get().strip())
            except ValueError as exc:
                raise ValueError(f"{label} must be a whole number.") from exc
            if not low <= value <= high:
                raise ValueError(f"{label} must be between {low:,} and {high:,}.")
            return value

        def as_float(variable: tk.StringVar, label: str, low: float, high: float) -> float:
            try:
                value = float(variable.get().strip())
            except ValueError as exc:
                raise ValueError(f"{label} must be a number.") from exc
            if not low <= value <= high:
                raise ValueError(f"{label} must be between {low:g} and {high:g}.")
            return value

        try:
            max_tokens = as_int(self.max_tokens_var, "Max output tokens", 1, 32768)
            temperature = as_float(self.temperature_var, "Temperature", 0.0, 2.0)
            top_p = as_float(self.top_p_var, "Top-p", 0.01, 1.0)
            context = as_int(self.context_var, "Context length", 512, 262144)
            layers = as_int(self.layers_var, "GPU layers", 0, 32)
            batch = as_int(self.batch_var, "Batch size", 1, 8192)
            ubatch = as_int(self.ubatch_var, "Ubatch size", 1, 4096)
            threads = as_int(self.threads_var, "CPU threads", 0, 64)
            parallel = as_int(self.parallel_var, "Parallel slots", 1, 4)
            if ubatch > batch:
                raise ValueError("Ubatch size cannot be larger than the batch size.")
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc), parent=self)
            return

        flash = self.flash_var.get()
        cache_k = self.k_var.get()
        cache_v = self.v_var.get()
        if flash not in {"on", "off", "auto"}:
            messagebox.showerror(APP_TITLE, "Flash attention must be on, off, or auto.", parent=self)
            return
        if cache_k not in {"q8_0", "f16", "q4_0"} or cache_v not in {"q4_0", "q8_0", "f16"}:
            messagebox.showerror(APP_TITLE, "Choose a valid KV cache type.", parent=self)
            return
        if flash == "off" and (cache_k != "f16" or cache_v != "f16"):
            messagebox.showerror(
                APP_TITLE,
                "Quantized KV cache requires flash attention. Set flash attention to on/auto, "
                "or set both caches to f16.",
                parent=self,
            )
            return

        cfg = dict(self.app.cfg)
        try:
            cfg = json.loads(config_path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        cfg.update(
            {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "disable_thinking": bool(self.disable_thinking_var.get()),
                "llama_ctx": context,
                "llama_ngl": layers,
                "llama_batch": batch,
                "llama_ubatch": ubatch,
                "llama_threads": threads,
                "llama_parallel": parallel,
                "llama_flash_attn": flash,
                "llama_cache_k": cache_k,
                "llama_cache_v": cache_v,
                "warmup": bool(self.warmup_var.get()),
            }
        )
        try:
            config_path().parent.mkdir(parents=True, exist_ok=True)
            config_path().write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"Could not save settings:\n{exc}", parent=self)
            return

        self.app.cfg = cfg
        self.app._update_metrics()
        self.app.append_log(
            f"Settings saved: ctx {context:,} · {layers}/32 GPU · ubatch {ubatch} · "
            f"temp {temperature:g} · KV {cache_k}/{cache_v}"
        )
        running = self.app.controller is not None and self.app.controller.is_active()
        messagebox.showinfo(
            APP_TITLE,
            "Settings saved."
            + ("\n\nStop and start the server to apply them." if running else "\nThey apply on the next start."),
            parent=self,
        )
        self.destroy()


def path_size_bytes(path_value: str) -> int:
    if not path_value:
        return 0
    path = Path(path_value)
    try:
        if path.is_file():
            return path.stat().st_size
        if not path.exists():
            return 0
        total = 0
        for item in path.rglob("*"):
            try:
                if item.is_file():
                    total += item.stat().st_size
            except OSError:
                continue
        return total
    except OSError:
        return 0


class ModelsWindow(tk.Toplevel):
    def __init__(self, app: LocalForgeApp):
        super().__init__(app.root)
        self.app = app
        self.title("LocalForge · models")
        self.configure(bg=PANEL)
        self.transient(app.root)
        self.minsize(850, 650)
        self.records: list[dict] = []
        self.records_by_id: dict[str, dict] = {}
        self._queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._loading = False
        self._closed = False
        self.proc: subprocess.Popen[str] | None = None
        self._op_kind: str | None = None
        self._op_record: dict | None = None
        self._op_focus_id: str | None = None
        self._op_cancelled = False
        self._finishing = False
        self._expected_bytes = 0
        self._last_progress_at = 0.0
        self._last_lines: list[str] = []
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(120, self._poll)
        self.refresh_models()

    def _build(self) -> None:
        outer = tk.Frame(self, bg=BG, padx=20, pady=16)
        outer.pack(fill="both", expand=True)
        outer.grid_columnconfigure(0, weight=1)
        outer.grid_rowconfigure(1, weight=3)
        outer.grid_rowconfigure(5, weight=1)

        header = tk.Frame(outer, bg=BG)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.grid_columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="Models",
            bg=BG,
            fg=TEXT,
            font=(UI, 18, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            header,
            text="Download, register, pick, or remove local models.",
            bg=BG,
            fg=MUTED,
            font=(UI, 9),
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        style = ttk.Style(self)
        style.configure(
            "Forge.Treeview",
            background=PANEL_2,
            fieldbackground=PANEL_2,
            foreground=TEXT,
            rowheight=27,
            borderwidth=0,
            font=(UI, 9),
        )
        style.configure(
            "Forge.Treeview.Heading",
            background=PANEL,
            foreground=MUTED,
            relief="flat",
            font=(UI, 8, "bold"),
        )
        style.map(
            "Forge.Treeview",
            background=[("selected", "#2b3b52")],
            foreground=[("selected", TEXT)],
        )
        style.configure(
            "Forge.Horizontal.TProgressbar",
            troughcolor="#090d13",
            background=AMBER,
            bordercolor=PANEL_2,
            lightcolor=AMBER,
            darkcolor=AMBER,
        )

        table_card = tk.Frame(
            outer,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
            bd=0,
        )
        table_card.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        table_card.grid_rowconfigure(0, weight=1)
        table_card.grid_columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(
            table_card,
            columns=("model", "engine", "size", "status", "modalities"),
            show="headings",
            selectmode="browse",
            style="Forge.Treeview",
            height=7,
        )
        headings = {
            "model": ("MODEL", 370),
            "engine": ("ENGINE", 105),
            "size": ("SIZE", 95),
            "status": ("STATUS", 125),
            "modalities": ("MODALITIES", 130),
        }
        for key, (label, width) in headings.items():
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, minwidth=70, anchor="w", stretch=key == "model")
        self.tree.tag_configure("missing", foreground=DIM)
        self.tree.tag_configure("local", foreground=BLUE)
        self.tree.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        scroll = ttk.Scrollbar(
            table_card,
            orient="vertical",
            command=self.tree.yview,
            style="Forge.Vertical.TScrollbar",
        )
        scroll.grid(row=0, column=1, sticky="ns", pady=10, padx=(0, 8))
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind("<<TreeviewSelect>>", lambda _event: self._selection_changed())
        self.tree.bind("<Double-1>", lambda _event: self.use_selected())

        detail_card = tk.Frame(
            outer,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
            bd=0,
        )
        detail_card.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        detail_card.grid_columnconfigure(0, weight=1)
        self.detail_title = tk.Label(
            detail_card,
            text="Select a model",
            bg=PANEL,
            fg=TEXT,
            font=(UI, 11, "bold"),
            anchor="w",
        )
        self.detail_title.grid(row=0, column=0, sticky="w", padx=14, pady=(10, 0))
        self.detail_meta = tk.Label(
            detail_card,
            text="",
            bg=PANEL,
            fg=MUTED,
            font=(UI, 9),
            anchor="w",
        )
        self.detail_meta.grid(row=1, column=0, sticky="w", padx=14, pady=(2, 0))
        self.detail_notes = tk.Label(
            detail_card,
            text="",
            bg=PANEL,
            fg=MUTED,
            font=(UI, 8),
            anchor="w",
            justify="left",
            wraplength=820,
        )
        self.detail_notes.grid(row=2, column=0, sticky="w", padx=14, pady=(5, 0))
        self.detail_path = tk.Label(
            detail_card,
            text="",
            bg=PANEL,
            fg=DIM,
            font=(MONO, 8),
            anchor="w",
            justify="left",
            wraplength=820,
        )
        self.detail_path.grid(row=3, column=0, sticky="w", padx=14, pady=(4, 10))

        actions = tk.Frame(outer, bg=BG)
        actions.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        self.btn_download = self.app._button(
            actions, "Download", self.download_selected, primary=True, width=11, disabled=True
        )
        self.btn_download.pack(side="left")
        self.btn_use = self.app._button(
            actions, "Use Selected", self.use_selected, secondary=True, width=12, disabled=True
        )
        self.btn_use.pack(side="left", padx=(7, 0))
        self.btn_add = self.app._button(
            actions, "Add Local", self.open_add_local, ghost=True, width=10
        )
        self.btn_add.pack(side="left", padx=(7, 0))
        self.btn_remove = self.app._button(
            actions, "Remove", self.remove_selected, ghost=True, width=9, disabled=True
        )
        self.btn_remove.pack(side="left", padx=(7, 0))
        self.btn_refresh = self.app._button(
            actions, "Refresh", self.refresh_models, ghost=True, width=9
        )
        self.btn_refresh.pack(side="left", padx=(7, 0))
        self.btn_folder = self.app._button(
            actions, "Open Folder", self.open_models_folder, ghost=True, width=12
        )
        self.btn_folder.pack(side="left", padx=(7, 0))

        status_row = tk.Frame(outer, bg=BG)
        status_row.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        status_row.grid_columnconfigure(1, weight=1)
        self.status_label = tk.Label(
            status_row,
            text="Ready.",
            bg=BG,
            fg=MUTED,
            font=(UI, 9),
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.progress = ttk.Progressbar(
            status_row,
            style="Forge.Horizontal.TProgressbar",
            mode="determinate",
            length=260,
        )
        self.progress.grid(row=0, column=1, sticky="w")
        self.progress.grid_remove()
        self.btn_cancel = self.app._button(
            status_row, "Cancel", self.cancel_selected_operation, ghost=True, width=8, disabled=True
        )
        self.btn_cancel.grid(row=0, column=2, sticky="e")

        log_card = tk.Frame(
            outer,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
            bd=0,
        )
        log_card.grid(row=5, column=0, sticky="nsew")
        log_card.grid_rowconfigure(1, weight=1)
        log_card.grid_columnconfigure(0, weight=1)
        tk.Label(
            log_card,
            text="OPERATION LOG",
            bg=PANEL,
            fg=DIM,
            font=(UI, 7, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 4))
        text_frame = tk.Frame(log_card, bg=PANEL_2)
        text_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        text_frame.grid_rowconfigure(0, weight=1)
        text_frame.grid_columnconfigure(0, weight=1)
        self.log = tk.Text(
            text_frame,
            bg="#090d13",
            fg="#aeb9c9",
            insertbackground=TEXT,
            selectbackground="#2b3b52",
            relief="flat",
            bd=0,
            wrap="word",
            font=(MONO, 8),
            padx=9,
            pady=7,
            height=7,
            state="disabled",
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        log_scroll = ttk.Scrollbar(
            text_frame,
            orient="vertical",
            command=self.log.yview,
            style="Forge.Vertical.TScrollbar",
        )
        log_scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=log_scroll.set)

        self.update_idletasks()
        width, height = 920, 780
        x = (self.winfo_screenwidth() - width) // 2
        y = max(20, (self.winfo_screenheight() - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _set_status(self, text: str, color: str = MUTED) -> None:
        self.status_label.configure(text=text, fg=color)

    def _append_log(self, message: str) -> None:
        line = ANSI_RE.sub("", message).strip()
        if not line:
            return
        self._last_lines.append(line)
        self._last_lines = self._last_lines[-8:]
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        count = int(self.log.index("end-1c").split(".")[0])
        if count > 250:
            self.log.delete("1.0", f"{count - 200}.0")
        self.log.configure(state="disabled")
        self.log.see("end")

    def refresh_models(self) -> None:
        if self.proc is not None or self._loading:
            return
        self._loading = True
        self._set_status("Loading model catalog…", BLUE)
        self._update_actions()
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self) -> None:
        try:
            records = models_json(self.app.app_dir)
        except Exception as exc:
            self._queue.put(("error", str(exc)))
            return
        self._queue.put(("models", records))

    def _apply_records(self, records: list[dict]) -> None:
        self._loading = False
        self.records = records
        self.records_by_id = {str(item.get("id")): item for item in records}
        self.app._apply_models(records, notify=False)
        selected_id = self.selected_id() or self.app.selected_model()
        self.tree.delete(*self.tree.get_children())
        for record in records:
            model_id = str(record.get("id"))
            installed = bool(record.get("installed"))
            local = bool(record.get("local"))
            status = "Installed" if installed else "Not installed"
            if local:
                status = "Local" if installed else "Local · missing"
            tags = ("missing",) if not installed else (("local",) if local else ())
            self.tree.insert(
                "",
                "end",
                iid=model_id,
                values=(
                    record.get("name") or model_id,
                    engine_label(str(record.get("engine") or "")),
                    format_model_size(record),
                    status,
                    ",".join(record.get("modalities") or []),
                ),
                tags=tags,
            )
        target = self._op_focus_id or selected_id
        if target not in self.records_by_id:
            installed = [item for item in records if item.get("installed")]
            target = str((installed[0] if installed else records[0]).get("id")) if records else ""
        if target:
            self.tree.selection_set(target)
            self.tree.focus(target)
            self.tree.see(target)
        self._op_focus_id = None
        self._set_status(f"Ready · {len(records)} models.", MUTED)
        self._selection_changed()

    def _selection_changed(self) -> None:
        record = self.selected_record()
        if record is None:
            self.detail_title.configure(text="Select a model")
            self.detail_meta.configure(text="")
            self.detail_notes.configure(text="")
            self.detail_path.configure(text="")
        else:
            installed = bool(record.get("installed"))
            local = bool(record.get("local"))
            size = format_model_size(record)
            state = "local registration" if local else "catalog"
            self.detail_title.configure(text=str(record.get("name") or record.get("id")))
            ctx = record.get("context_length")
            context_note = f" · {int(ctx):,} ctx" if ctx else ""
            self.detail_meta.configure(
                text=f"{engine_label(str(record.get('engine') or ''))} · "
                f"{record.get('license') or 'unknown license'} · {size} · {state}{context_note}"
            )
            notes = str(record.get("notes") or "")
            if not installed and record.get("repo"):
                notes = f"{notes}\nDownloads from {record.get('repo')}."
            self.detail_notes.configure(text=notes)
            self.detail_path.configure(text=str(record.get("path") or ""))
        self._update_actions()

    def selected_record(self) -> dict | None:
        selection = self.tree.selection()
        if not selection:
            return None
        return self.records_by_id.get(str(selection[0]))

    def selected_id(self) -> str | None:
        record = self.selected_record()
        return str(record.get("id")) if record else None

    def _server_busy(self) -> bool:
        return (
            self.app.external
            or self.app.state in {"starting", "running", "stopping"}
            or (self.app.controller is not None and self.app.controller.is_active())
        )

    def _update_actions(self) -> None:
        busy = self._loading or self.proc is not None
        record = self.selected_record()
        installed = bool(record and record.get("installed"))
        local = bool(record and record.get("local"))
        can_download = bool(record) and not installed and not local and not busy
        can_use = installed and not busy and not self._server_busy()
        can_remove = bool(record) and (installed or local) and not busy and not self._server_busy()
        self.btn_download.configure(
            state="normal" if can_download else "disabled",
            bg=AMBER if can_download else "#2a303b",
            fg="#181006" if can_download else DIM,
            activebackground=AMBER_HOVER if can_download else "#2a303b",
            cursor="hand2" if can_download else "arrow",
        )
        self.btn_use.configure(
            state="normal" if can_use else "disabled",
            bg="#22304a" if can_use else "#1b2029",
            fg="#d5e5ff" if can_use else DIM,
            activebackground="#2c3f60" if can_use else "#1b2029",
            cursor="hand2" if can_use else "arrow",
        )
        for button, enabled in (
            (self.btn_add, not busy),
            (self.btn_remove, can_remove),
            (self.btn_refresh, not busy),
            (self.btn_folder, not busy),
            (self.btn_cancel, self.proc is not None),
        ):
            button.configure(
                state="normal" if enabled else "disabled",
                fg=MUTED if enabled else DIM,
                cursor="hand2" if enabled else "arrow",
            )
        self.btn_remove.configure(text="Unregister" if local else "Remove")

    def use_selected(self) -> None:
        record = self.selected_record()
        if record is None:
            return
        if not record.get("installed"):
            messagebox.showinfo(APP_TITLE, "Download this model before selecting it.", parent=self)
            return
        if self._server_busy():
            messagebox.showinfo(
                APP_TITLE,
                "Stop the local server before switching models.",
                parent=self,
            )
            return
        model_id = str(record.get("id"))
        if self.app.set_selected_model(model_id):
            self._set_status(f"Selected {model_short_name(record)}. Start the server to load it.", GREEN)
            self._append_log(f"Selected main model: {model_id}")
        else:
            self.refresh_models()

    def download_selected(self) -> None:
        record = self.selected_record()
        if record is None or record.get("installed") or record.get("local"):
            return
        size = format_model_size(record)
        repo = record.get("repo") or "Hugging Face"
        if not messagebox.askyesno(
            "Download model",
            f"{record.get('name')}\n\nSize: {size}\nSource: {repo}\n\n"
            "Download now? Partial files resume if interrupted.",
            parent=self,
        ):
            return
        command = [
            str(runtime_python(self.app.app_dir)),
            "-u",
            "-m",
            "local_forge",
            "models",
            "download",
            str(record.get("id")),
        ]
        self._start_operation("download", record, command)

    def remove_selected(self) -> None:
        record = self.selected_record()
        if record is None:
            return
        if self._server_busy():
            messagebox.showinfo(
                APP_TITLE,
                "Stop the local server before removing or unregistering a model.",
                parent=self,
            )
            return
        model_id = str(record.get("id"))
        if record.get("local"):
            message = (
                f"Unregister local model {model_short_name(record)}?\n\n"
                f"{record.get('path')}\n\nThe files will not be deleted."
            )
        else:
            message = (
                f"Delete {model_short_name(record)} and its files?\n\n"
                f"{record.get('path')}\nSize: {format_model_size(record)}\n\n"
                "This cannot be undone."
            )
        if not messagebox.askyesno("Remove model", message, parent=self):
            return
        command = [
            str(runtime_python(self.app.app_dir)),
            "-u",
            "-m",
            "local_forge",
            "models",
            "remove",
            model_id,
            "--yes",
        ]
        self._start_operation("remove", record, command)

    def open_add_local(self) -> None:
        AddLocalDialog(self)

    def start_add_local(
        self,
        model_id: str,
        path: str,
        name: str,
        engine: str,
        modalities: str,
    ) -> None:
        command = [
            str(runtime_python(self.app.app_dir)),
            "-u",
            "-m",
            "local_forge",
            "models",
            "add-local",
            model_id,
            path,
            "--engine",
            engine,
            "--name",
            name,
            "--modalities",
            modalities,
        ]
        record = {
            "id": model_id,
            "name": name,
            "engine": engine,
            "modalities": modalities.split(","),
            "installed": True,
            "local": True,
            "path": path,
            "size_gb": 0,
            "expected_size_gb": 0,
            "notes": "",
            "license": "local",
            "repo": "",
        }
        self._op_focus_id = model_id
        self._start_operation("add", record, command)

    def open_models_folder(self) -> None:
        path = Path(str(self.app.cfg.get("models_dir") or ""))
        if not str(path):
            messagebox.showerror(APP_TITLE, "Models directory is not configured.", parent=self)
            return
        try:
            path.mkdir(parents=True, exist_ok=True)
            os.startfile(str(path))
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"Could not open the models folder:\n{exc}", parent=self)

    def _start_operation(self, kind: str, record: dict, command: list[str]) -> None:
        if self.proc is not None:
            return
        self._op_kind = kind
        self._op_record = record
        self._op_cancelled = False
        self._finishing = False
        self._last_lines = []
        self._expected_bytes = int(float(record.get("expected_size_gb") or 0) * 1_000_000_000)
        self._last_progress_at = 0.0
        self._append_log("")
        self._append_log(f"--- {kind}: {record.get('id')} ---")
        self._set_status(f"{kind.title()} in progress…", AMBER)
        if kind == "download":
            self.progress.configure(mode="determinate", maximum=max(self._expected_bytes, 1), value=0)
        else:
            self.progress.configure(mode="indeterminate")
        self.progress.grid()
        if kind != "download":
            self.progress.start(14)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            self.proc = subprocess.Popen(
                command,
                cwd=str(self.app.app_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            self.proc = None
            self.progress.stop()
            self.progress.grid_remove()
            self._set_status(f"Could not start: {exc}", RED)
            self._append_log(f"Start failed: {exc}")
            self._update_actions()
            return
        threading.Thread(target=self._read_operation_output, daemon=True).start()
        self._update_actions()

    def _read_operation_output(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for raw_line in proc.stdout:
            for part in raw_line.replace("\r", "\n").splitlines():
                if part.strip():
                    self._queue.put(("log", part.strip()))
        self._queue.put(("op_end", None))

    def _poll(self) -> None:
        if self._closed:
            return
        while True:
            try:
                kind, value = self._queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_log(str(value))
            elif kind == "models":
                self._apply_records(value)  # type: ignore[arg-type]
            elif kind == "error":
                self._loading = False
                self._set_status(f"Model list failed: {value}", RED)
                self._append_log(f"Model list failed: {value}")
                self._update_actions()
            elif kind == "op_end":
                self._finish_operation()
        if (
            self.proc is not None
            and self._op_kind == "download"
            and time.monotonic() - self._last_progress_at >= 2.0
        ):
            self._last_progress_at = time.monotonic()
            current = path_size_bytes(str(self._op_record.get("path") or "")) if self._op_record else 0
            expected = self._expected_bytes
            if expected > 0:
                value = min(current, expected)
                self.progress.stop()
                self.progress.configure(mode="determinate", maximum=expected, value=value)
                percent = min(99, int(value * 100 / expected))
                self._set_status(
                    f"Downloading {value / 1_000_000_000:.2f} / "
                    f"{expected / 1_000_000_000:.2f} GB · {percent}%",
                    AMBER,
                )
        self.after(150, self._poll)

    def _finish_operation(self) -> None:
        if self._finishing:
            return
        proc = self.proc
        if proc is None:
            return
        self._finishing = True
        try:
            code = proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            code = -1
        kind = self._op_kind or "operation"
        record = self._op_record or {}
        cancelled = self._op_cancelled
        self.progress.stop()
        self.progress.grid_remove()
        self.proc = None
        self._op_kind = None
        self._op_record = None
        self._finishing = False
        if cancelled:
            self._set_status("Cancelled.", MUTED)
            self._append_log("Operation cancelled.")
        elif code == 0:
            self._set_status(f"{kind.title()} complete.", GREEN)
            self._append_log(f"{kind.title()} complete.")
            if kind in {"download", "remove", "add"}:
                self._op_focus_id = str(record.get("id")) if kind == "add" else self._op_focus_id
                self.refresh_models()
        else:
            self._set_status(f"{kind.title()} failed (code {code}). See the log.", RED)
            self._append_log(f"{kind.title()} failed with exit code {code}.")
            detail = "\n".join(self._last_lines[-5:])
            messagebox.showerror(
                APP_TITLE,
                f"{kind.title()} failed.\n\n{detail[-1500:]}",
                parent=self,
            )
        self._update_actions()

    def cancel_selected_operation(self) -> None:
        self.cancel_operation()

    def cancel_operation(self, silent: bool = False) -> bool:
        proc = self.proc
        if proc is None:
            return True
        if not silent and not messagebox.askyesno(
            APP_TITLE,
            "Cancel the running model operation?",
            parent=self,
        ):
            return False
        self._op_cancelled = True
        self._set_status("Cancelling…", AMBER)
        if proc.poll() is None:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self._finish_operation()
        return True

    def has_active_operation(self) -> bool:
        return self.proc is not None

    def on_close(self) -> None:
        if self.proc is not None and not self.cancel_operation():
            return
        self.destroy()

    def destroy(self) -> None:
        if self.proc is not None:
            proc = self.proc
            self._op_cancelled = True
            if proc.poll() is None:
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            self.proc = None
        self._closed = True
        if self.app.models_window is self:
            self.app.models_window = None
        super().destroy()


class AddLocalDialog(tk.Toplevel):
    def __init__(self, manager: ModelsWindow):
        super().__init__(manager)
        self.manager = manager
        self.app = manager.app
        self.title("LocalForge · add local model")
        self.configure(bg=PANEL)
        self.resizable(False, False)
        self.transient(manager)
        self.grab_set()
        self.bind("<Escape>", lambda _event: self.destroy())
        self._build()

    def _build(self) -> None:
        outer = tk.Frame(self, bg=PANEL, padx=22, pady=18)
        outer.pack(fill="both", expand=True)
        outer.grid_columnconfigure(1, weight=1)

        tk.Label(
            outer,
            text="Add a local model",
            bg=PANEL,
            fg=TEXT,
            font=(UI, 15, "bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        tk.Label(
            outer,
            text="Registers an existing file or folder in place. Nothing is copied or deleted.",
            bg=PANEL,
            fg=MUTED,
            font=(UI, 9),
            anchor="w",
            wraplength=540,
            justify="left",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 14))

        self.engine_var = tk.StringVar(value="llama.cpp")
        self.path_var = tk.StringVar()
        self.id_var = tk.StringVar()
        self.name_var = tk.StringVar()
        self.text_var = tk.BooleanVar(value=True)
        self.image_var = tk.BooleanVar(value=False)

        self._field(outer, 2, "Engine", self.engine_var, combo=["llama.cpp", "openvino", "litert"])
        self._field(outer, 3, "Path", self.path_var)
        buttons = tk.Frame(outer, bg=PANEL)
        buttons.grid(row=3, column=2, sticky="w", padx=(8, 0))
        self.btn_file = self.app._button(buttons, "File…", self.browse_file, ghost=True, width=6)
        self.btn_file.pack(side="left")
        self.btn_dir = self.app._button(buttons, "Folder…", self.browse_folder, ghost=True, width=7)
        self.btn_dir.pack(side="left", padx=(4, 0))
        self._field(outer, 4, "Model ID", self.id_var)
        self._field(outer, 5, "Display name", self.name_var)

        tk.Label(outer, text="Modalities", bg=PANEL, fg=MUTED, font=(UI, 9)).grid(
            row=6, column=0, sticky="w", pady=(10, 0)
        )
        checks = tk.Frame(outer, bg=PANEL)
        checks.grid(row=6, column=1, columnspan=2, sticky="w", pady=(10, 0))
        for text, var, state in (
            ("Text", self.text_var, "disabled"),
            ("Image", self.image_var, "normal"),
        ):
            tk.Checkbutton(
                checks,
                text=text,
                variable=var,
                state=state,
                disabledforeground=MUTED,
                bg=PANEL,
                fg=TEXT,
                selectcolor=AMBER,
                activebackground=PANEL,
                activeforeground=TEXT,
                highlightthickness=0,
                bd=0,
                font=(UI, 9),
            ).pack(side="left", padx=(0, 14))

        tk.Label(
            outer,
            text="GGUF accepts a file or a folder containing the .gguf and optional mmproj. "
            "OpenVINO requires a model folder. LiteRT requires a .litertlm file.",
            bg=PANEL,
            fg=DIM,
            font=(UI, 8),
            wraplength=540,
            justify="left",
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(14, 0))

        footer = tk.Frame(outer, bg=PANEL)
        footer.grid(row=8, column=0, columnspan=3, sticky="e", pady=(16, 0))
        self.app._button(footer, "Cancel", self.destroy, ghost=True, width=9).pack(side="left")
        self.app._button(footer, "Add Model", self.add, primary=True, width=11).pack(
            side="left", padx=(8, 0)
        )

        self.update_idletasks()
        width = 640
        height = outer.winfo_reqheight() + 24
        x = (self.winfo_screenwidth() - width) // 2
        y = (self.winfo_screenheight() - height) // 2
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _field(
        self,
        parent: tk.Misc,
        row: int,
        label: str,
        variable: tk.StringVar,
        combo: list[str] | None = None,
    ) -> None:
        tk.Label(parent, text=label, bg=PANEL, fg=MUTED, font=(UI, 9)).grid(
            row=row, column=0, sticky="w", pady=5
        )
        if combo:
            widget: tk.Widget = ttk.Combobox(
                parent,
                textvariable=variable,
                values=combo,
                state="readonly",
                style="Forge.TCombobox",
                font=(MONO, 9),
                width=16,
            )
        else:
            widget = tk.Entry(
                parent,
                textvariable=variable,
                bg=PANEL_2,
                fg=TEXT,
                insertbackground=TEXT,
                relief="flat",
                highlightthickness=1,
                highlightbackground=BORDER,
                highlightcolor=AMBER,
                font=(MONO, 9),
                width=38,
            )
        widget.grid(row=row, column=1, sticky="ew", padx=(14, 0), pady=5)

    def browse_file(self) -> None:
        engine = self.engine_var.get()
        if engine == "openvino":
            messagebox.showinfo(
                APP_TITLE,
                "OpenVINO models are folders. Use Folder… instead.",
                parent=self,
            )
            return
        filetypes = (
            [("GGUF model", "*.gguf"), ("All files", "*.*")]
            if engine == "llama.cpp"
            else [("LiteRT bundle", "*.litertlm"), ("All files", "*.*")]
        )
        path = filedialog.askopenfilename(parent=self, filetypes=filetypes)
        if path:
            self.path_var.set(path)
            self._suggest(path)

    def browse_folder(self) -> None:
        path = filedialog.askdirectory(parent=self)
        if path:
            self.path_var.set(path)
            self._suggest(path)

    def _suggest(self, path: str) -> None:
        candidate = Path(path)
        stem = candidate.stem if candidate.is_file() else candidate.name
        clean = re.sub(r"[^a-zA-Z0-9._-]+", "-", stem).strip("-").lower()
        if not self.id_var.get().strip():
            self.id_var.set(clean)
        if not self.name_var.get().strip():
            self.name_var.set(stem)

    def add(self) -> None:
        path = self.path_var.get().strip()
        model_id = self.id_var.get().strip()
        name = self.name_var.get().strip() or model_id
        engine = self.engine_var.get()
        if not path or not Path(path).exists():
            messagebox.showerror(APP_TITLE, "Choose an existing model file or folder.", parent=self)
            return
        if not model_id:
            messagebox.showerror(APP_TITLE, "Enter a model ID.", parent=self)
            return
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model_id):
            messagebox.showerror(
                APP_TITLE,
                "Model ID may contain letters, numbers, dots, underscores, and hyphens.",
                parent=self,
            )
            return
        modalities = "text"
        if self.image_var.get():
            modalities = "text,image"
        if not self.text_var.get() and not self.image_var.get():
            messagebox.showerror(APP_TITLE, "Select at least one modality.", parent=self)
            return
        self.manager.start_add_local(model_id, path, name, engine, modalities)
        self.destroy()


def main() -> None:
    configure_portable_environment()
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("LocalForge.Desktop")
    except Exception:
        pass
    root = tk.Tk()
    LocalForgeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
