"""Desktop app: the F1 Strategy Predictor in its own window, started from a desktop shortcut.

At start it downloads any newer race data from the cloud (if AWS is set up, see aws_sync.py; otherwise, or with no
internet, it uses the data already here), starts the app on this PC only (http://127.0.0.1:8000) and opens it in a
Microsoft Edge (or Chrome) app window without an address bar. Closing the window stops the app. If the app is
already running (python src/api.py) it just opens a window. While running it builds any new race the cloud fetched
(newraces.py) and a message says when one is ready (it shows the next time the app opens). The game telemetry still
goes straight from the console to this PC.

python src/desktop.py              start it (the shortcut runs this without a console window)
python src/desktop.py --shortcut   create the desktop shortcut (Windows)

A log of each run is in desktop.log, in the app's folder in LOCALAPPDATA."""
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from config import ROOT

NAME = "F1 Strategy Predictor"
PORT = int(os.environ.get("F1_APP_PORT", "8000"))                   # set another port if 8000 is taken
URL = f"http://127.0.0.1:{PORT}/"
HOME = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / NAME     # browser profile for the window, and the log
BROWSERS = [r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe"]
ICON = ROOT / "web" / "app.ico"


def log(text):
    print(time.strftime("%H:%M:%S"), text, flush=True)


def running(url=URL, timeout=1.0):
    """Is the app already running on this PC?"""
    try:
        with urllib.request.urlopen(url + "game/status", timeout=timeout) as answer:
            return answer.status == 200
    except OSError:
        return False


def browser():
    """Find Edge or Chrome on this PC (both can open a page as an app window)."""
    places = [os.environ.get(v) for v in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA")]
    for exe in BROWSERS:
        for place in places:
            if place and (Path(place) / exe).exists():
                return str(Path(place) / exe)
    return None


def window(exe, url=URL, profile=HOME / "window"):
    """Command for the app window: no address bar or tabs and its own browser profile, so it runs as a separate
    program (it ends when the window is closed) and the normal browser isn't touched."""
    return [exe, f"--app={url}", f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
            "--window-size=1440,900"]


def online(cfg, timeout=2.0):
    """Can this PC reach the bucket's region within `timeout` seconds (DNS lookup included)? Checked first so that
    with no internet the app opens straight away instead of waiting for the cloud connection to time out."""
    host, reached = f"s3.{cfg.get('region', 'eu-north-1')}.amazonaws.com", []

    def reach():
        try:
            socket.create_connection((host, 443), timeout=timeout).close()
            reached.append(True)
        except OSError:
            pass

    worker = threading.Thread(target=reach, daemon=True)        # the DNS lookup has no timeout of its own
    worker.start()
    worker.join(timeout)
    return bool(reached)


def from_the_cloud():
    """Download newer data from the cloud if AWS is set up and we're online. It never stops the app from starting.
    Returns True if it downloaded (so new_races doesn't download again)."""
    try:
        import aws_sync
        cfg = aws_sync.settings()
        if not cfg:
            log("The cloud: this PC isn't set up for AWS, so the data here is used")
        elif not online(cfg):
            log("The cloud: no internet, so the data here is used")
        else:
            aws_sync.download(aws_sync.client(cfg), cfg["bucket"], log=log)
            return True
    except BaseException as e:                   # no key yet, or no permission
        log(f"The cloud: skipped ({e}); the data here is used")
    return False


def new_races(fetched=False):
    """Build the races the cloud fetched since the last start (newraces.py) while the app runs; a message says when
    one is ready (it shows the next time the app opens). Only one build at a time, whichever copy of the app starts
    it. fetched = True if the data was just downloaded (otherwise it downloads first)."""
    lock = HOME / "newraces.lock"
    try:
        if lock.exists() and time.time() - lock.stat().st_mtime > 3600:
            lock.unlink()                        # left by a build that was stopped
        with open(lock, "x"):
            pass
    except OSError:
        return
    try:
        import aws_sync
        import newraces
        cfg = aws_sync.settings()
        added = []
        if cfg and online(cfg):
            s3 = aws_sync.client(cfg)
            if not fetched:                                     # another PC may have built a race already
                aws_sync.download(s3, cfg["bucket"], log=log)
            added = newraces.add(s3, cfg["bucket"], log=log)
        if added:
            log("New races added: " + ", ".join(added))
            tell("New race added: " + ", ".join(r.split(" ", 1)[1] for r in added) +
                 ".\nClose the app and open it again to see it.")
    except BaseException as e:
        log(f"New races: skipped ({e})")
    finally:
        lock.unlink(missing_ok=True)


def tell(text):
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, text, NAME, 0x40)       # info message box
    else:
        log(text)


def start_app():
    """Run the app inside this program, same as python src/api.py, on this PC only."""
    os.environ.setdefault("F1_APP_LAN", "0")
    import uvicorn
    import api                                   # import after the download, api reads the data when it loads
    server = uvicorn.Server(uvicorn.Config(api.app, host="127.0.0.1", port=PORT, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(1200):                        # the first start builds the replay file (takes a few seconds)
        if running():
            return server, thread
        if not thread.is_alive():
            break
        time.sleep(0.1)
    raise SystemExit("The app didn't start: see desktop.log")


def shortcut():
    """Create a desktop shortcut that starts the app without a console window."""
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    made = ROOT / f"{NAME}.lnk"                  # created here and then moved: the shortcut maker can't save to a
    script = ("[Console]::OutputEncoding = [Text.Encoding]::UTF8; "                 # folder with a Greek name
              f"$t = '{made}'; $s = (New-Object -ComObject WScript.Shell).CreateShortcut($t); "
              f"$s.TargetPath = '{pythonw}'; $s.Arguments = '\"{Path(__file__).resolve()}\"'; "
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
    if sys.stdout is None:                       # started from the shortcut (no console), so log to desktop.log
        sys.stdout = sys.stderr = open(HOME / "desktop.log", "a", encoding="utf-8", buffering=1)
    log(f"{NAME}: starting")
    ours, fetched = None, False
    if running():
        log("The app is already running: opening a window on it")
    else:
        fetched = from_the_cloud()
        ours = start_app()
        log(f"The app is running at {URL}")
    builder = threading.Thread(target=new_races, args=(fetched,), name="new-races")
    builder.start()
    exe = browser()
    if exe:
        subprocess.run(window(exe))              # until the window is closed
        log("The window was closed")
    else:
        webbrowser.open(URL)                     # no Edge or Chrome: use the default browser until Ctrl+C
        try:
            while ours and ours[1].is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    if ours:
        ours[0].should_exit = True               # stop the app this program started
        ours[1].join(timeout=10)
        log("The app stopped")
    builder.join()                               # let a running build finish first


if __name__ == "__main__":
    main()
