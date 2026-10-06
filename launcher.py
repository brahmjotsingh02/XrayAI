import os
import sys
import socket
import time
import threading
from pathlib import Path

import webview
import uvicorn

# ── Base directory (works for a normal script AND a PyInstaller build) ────
# Spawning a subprocess with sys.executable (the old approach) breaks on
# Windows once this is packaged into a .exe: sys.executable then points at
# the launcher.exe itself, not a python.exe, so "python -m uvicorn ..."
# either fails outright or re-launches the whole app recursively. Running
# uvicorn in-process inside a thread avoids that entirely and works
# identically on macOS, Linux, and Windows/PyInstaller.
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).resolve().parent
else:
    BASE_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BASE_DIR / "src"))
os.environ.setdefault("XRAY_BASE_DIR", str(BASE_DIR))

HOST = "127.0.0.1"
PORT = 8000


def start_server():
    from app.backend import app  # imported here so BASE_DIR/sys.path is set first
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


def wait_for_server(host, port, timeout=20):
    """Poll the port instead of guessing a fixed sleep duration, since
    startup time varies a lot more across Windows machines than on macOS."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


threading.Thread(target=start_server, daemon=True).start()

if not wait_for_server(HOST, PORT):
    print(
        f"Server did not start within the timeout. Check that all "
        f"dependencies are installed and port {PORT} is free."
    )

try:
    window = webview.create_window(
        "XrayAI — Disease Detection",
        f"http://{HOST}:{PORT}",
        width=1100,
        height=750,
        resizable=True,
    )
    # Force the Chromium-based Edge WebView2 backend on Windows. Without
    # this, pywebview can silently fall back to the legacy IE/Trident
    # engine on machines without WebView2 installed, which breaks modern
    # JS (arrow functions, template literals) and CSS (custom properties,
    # backdrop-filter) used by this UI — the app "loads" but the page is
    # blank or broken.
    gui = "edgechromium" if sys.platform.startswith("win") else None
    webview.start(gui=gui)
except Exception as exc:
    print(f"Failed to start the app window: {exc}")
    if sys.platform.startswith("win"):
        print(
            "On Windows this usually means the Microsoft Edge WebView2 "
            "Runtime is missing. Install it from: "
            "https://developer.microsoft.com/microsoft-edge/webview2/"
        )
    raise