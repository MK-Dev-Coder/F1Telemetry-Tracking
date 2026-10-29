"""The desktop app: the F1 Strategy Predictor in a window of its own, from a shortcut on the desktop.

When it starts, it fetches any newer race data from the cloud (if this PC is set up for AWS, see aws_sync.py; without
it, or without the internet, it uses the data already here), starts the app on this PC only
(http://127.0.0.1:8000) and opens it in a window of Microsoft Edge (or Chrome) with no address bar. Closing the
window stops the app. If the app is already running (python api.py), it only opens a window on it. The game's
telemetry still comes straight from the console to this PC, as before.

    python desktop.py              start it (the shortcut runs this without a console window)
    python desktop.py --shortcut   put the shortcut on the desktop (Windows)

What it did each time is in desktop.log, in the app's folder in your user folder (AppData\\Local).
"""
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from config import ROOT

NAME = "F1 Strategy Predictor"
PORT = int(os.environ.get("F1_APP_PORT", "8000"))                   # another port if 8000 is taken
URL = f"http://127.0.0.1:{PORT}/"
HOME = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / NAME     # the window's browser profile and the log
BROWSERS = [r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe"]
ICON = ROOT / "web" / "app.ico"


def log(text):
    print(time.strftime("%H:%M:%S"), text, flush=True)


def running(url=URL, timeout=1.0):
    """Is the app already answering on this PC?"""
    try:
        with urllib.request.urlopen(url + "game/status", timeout=timeout) as answer:
            return answer.status == 200
    except OSError:
        return False


def browser():
    """Edge or Chrome on this PC: both can show a page as an app window."""
    places = [os.environ.get(v) for v in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
    for exe in BROWSERS:
        for place in places:
            if place and (Path(place) / exe).exists():
                return str(Path(place) / exe)
    return None


def window(exe, url=URL, profile=HOME / "window"):
    """The command for an app window: no address bar or tabs, and a browser profile of its own, so the window is a
    program of its own (it closes when you close it) and your own browser is left as it is."""
    return [exe, f"--app={url}", f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
            "--window-size=1440,900"]


def from_the_cloud():
    """Newer data from the cloud, if this PC is set up for it. It never stops the app from starting."""
    try:
        import aws_sync
        cfg = aws_sync.settings()
        if cfg:
            aws_sync.download(aws_sync.client(cfg), cfg["bucket"], log=log)
        else:
            log("The cloud: this PC isn't set up for AWS, so the data here is used")
    except BaseException as e:                   # offline, no key yet, or no permission
        log(f"The cloud: skipped ({e}); the data here is used")


def start_app():
    """The app, in this program: the same as python api.py, on this PC only."""
    os.environ.setdefault("F1_APP_LAN", "0")
    import uvicorn
    import api                                   # after the download: api reads the data as it loads
    server = uvicorn.Server(uvicorn.Config(api.app, host="127.0.0.1", port=PORT, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(1200):                        # the first start builds the replay file (a few seconds)
        if running():
            return server, thread
        if not thread.is_alive():
            break
        time.sleep(0.1)
    raise SystemExit("The app didn't start: see desktop.log")


def shortcut():
    """A shortcut on the desktop that starts the app without a console window."""
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    made = ROOT / f"{NAME}.lnk"                  # made here, then moved: Windows' shortcut maker can't save into a
    script = ("[Console]::OutputEncoding = [Text.Encoding]::UTF8; "                 # folder with a Greek name
              f"$t = '{made}'; $s = (New-Object -ComObject WScript.Shell).CreateShortcut($t); "
              f"$s.TargetPath = '{pythonw}'; $s.Arguments = '\"{ROOT / 'desktop.py'}\"'; "
              f"$s.WorkingDirectory = '{ROOT}'; $s.IconLocation = '{ICON}'; $s.Description = '{NAME}'; $s.Save(); "
              "$p = Join-Path ([Environment]::GetFolderPath('Desktop')) '" + NAME + ".lnk'; "
              "Move-Item -LiteralPath $t -Destination $p -Force; "
              "if (Test-Path -LiteralPath $p) { 'Made: ' + $p } else { 'The shortcut could not be made' }")
    done = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, encoding="utf-8")
    print((done.stdout + done.stderr).strip())


def main():
    if "--shortcut" in sys.argv:
        return shortcut()
    HOME.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None:                       # started from the shortcut (no console): write to desktop.log
        sys.stdout = sys.stderr = open(HOME / "desktop.log", "a", encoding="utf-8", buffering=1)
    log(f"{NAME}: starting")
    ours = None
    if running():
        log("The app is already running: opening a window on it")
    else:
        from_the_cloud()
        ours = start_app()
        log(f"The app is running at {URL}")
    exe = browser()
    if exe:
        subprocess.run(window(exe))              # until the window is closed
        log("The window was closed")
    else:
        webbrowser.open(URL)                     # no Edge or Chrome: the usual browser, until Ctrl+C
        try:
            while ours and ours[1].is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    if ours:
        ours[0].should_exit = True               # stop the app this program started
        ours[1].join(timeout=10)
        log("The app stopped")


if __name__ == "__main__":
    main()
