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
from tkinter import messagebox, ttk

APP_TITLE = "LocalForge"
APP_VERSION = "0.2.0"
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

MODELS = [
    ("Qwen3.5-4B · 131k · recommended", "qwen3.5-4b-abliterated", "Qwen3.5-4B"),
    ("Gemma 4 E2B · CPU vision fallback", "gemma4-e2b-uncensored-ov", "Gemma 4 E2B"),
]
MODEL_IDS = {label: model_id for label, model_id, _ in MODELS}
MODEL_SHORT = {model_id: short for _, model_id, short in MODELS}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


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
            if (candidate / ".venv" / "Scripts" / "python.exe").is_file():
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
    appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    return appdata / "LocalForge" / "config.json"


def load_config() -> dict:
    try:
        return json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


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

            python = self.app_dir / ".venv" / "Scripts" / "python.exe"
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
            "The LocalForge environment was not found. Keep the EXE inside the local-forge folder."
            if self.app_dir is None
            else "Server is not running. Start it to use Jan or the local API."
        )
        self.external = False
        self._closing = False
        self._header_icon: tk.PhotoImage | None = None

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
            values=[label for label, _, _ in MODELS],
            state="readonly",
            style="Forge.TCombobox",
            font=(UI, 9),
            width=42,
        )
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

    def _update_metrics(self) -> None:
        if not hasattr(self, "metric_vars"):
            return
        selected = MODEL_IDS.get(self.model_combo.get(), MODELS[0][1])
        self.metric_vars["model"].set(MODEL_SHORT.get(selected, selected))
        if selected.startswith("qwen"):
            ctx = int(self.cfg.get("llama_ctx", 131072) or 131072)
            ngl = self.cfg.get("llama_ngl", 10)
            cache_k = str(self.cfg.get("llama_cache_k", "q8_0")).split("_")[0].upper()
            cache_v = str(self.cfg.get("llama_cache_v", "q4_0")).split("_")[0].upper()
            self.metric_vars["context"].set(f"{ctx:,}")
            self.metric_vars["offload"].set(f"{ngl} / 32 GPU")
            self.engine_hint.configure(text=f"KV cache {cache_k} K · {cache_v} V · one slot")
        else:
            self.metric_vars["context"].set("model default")
            self.metric_vars["offload"].set("OpenVINO CPU")
            self.engine_hint.configure(text="Gemma fallback · CPU stays stable on 16 GB")

    def _refresh_ram(self) -> None:
        ram = free_ram_gb()
        if hasattr(self, "metric_vars"):
            self.metric_vars["ram"].set(f"{ram:.1f} GB" if ram is not None else "—")
        self.root.after(2000, self._refresh_ram)

    def _poll_events(self) -> None:
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
        return MODEL_IDS.get(self.model_combo.get(), MODELS[0][1])

    def start(self) -> None:
        if self.controller is None:
            messagebox.showerror(
                APP_TITLE,
                "The LocalForge environment was not found.\n"
                "Keep LocalForge.exe inside the local-forge folder.",
            )
            return
        model_id = self.selected_model()
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
        appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        log_dir = appdata / "LocalForge" / "logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            os.startfile(str(log_dir))
        except OSError as exc:
            messagebox.showerror(APP_TITLE, f"Could not open the log folder:\n{exc}")

    def open_settings(self) -> None:
        SettingsDialog(self)

    def on_close(self) -> None:
        if self._closing:
            return
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


def main() -> None:
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
