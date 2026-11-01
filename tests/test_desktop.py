"""desktop.py: the app in a window of its own. The tests don't open a window or start the app."""
import socket
import time

import aws_sync
import desktop
import newraces


def test_the_window_is_an_app_of_its_own():
    command = desktop.window("msedge.exe", "http://127.0.0.1:8000/", profile="C:/profile")
    assert command[0] == "msedge.exe"
    assert "--app=http://127.0.0.1:8000/" in command           # no address bar or tabs
    assert "--user-data-dir=C:/profile" in command             # its own profile: closing it ends it


def test_it_knows_when_the_app_is_not_running():
    with socket.socket() as s:                                 # a port nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert desktop.running(f"http://127.0.0.1:{port}/", timeout=0.5) is False


def cloud(monkeypatch, tmp_path, online):
    """A PC set up for AWS, with or without the internet; returns what the cloud was asked, in order."""
    asked = []
    monkeypatch.setattr(desktop, "HOME", tmp_path)
    monkeypatch.setattr(desktop, "log", asked.append)
    monkeypatch.setattr(desktop, "online", lambda cfg: online)
    monkeypatch.setattr(aws_sync, "settings", lambda: {"bucket": "b", "region": "eu-north-1"})
    monkeypatch.setattr(aws_sync, "client", lambda cfg: "s3")
    monkeypatch.setattr(aws_sync, "download", lambda *a, **k: asked.append("download"))
    monkeypatch.setattr(newraces, "add", lambda *a, **k: asked.append("add") or [])
    return asked


def test_without_the_internet_the_app_opens_at_once(monkeypatch, tmp_path):
    asked = cloud(monkeypatch, tmp_path, online=False)
    desktop.from_the_cloud()
    desktop.new_races()
    assert asked == ["The cloud: no internet, so the data here is used"]     # no waiting on the cloud
    assert not (tmp_path / "newraces.lock").exists()


def test_new_races_fetch_the_latest_data_first(monkeypatch, tmp_path):
    asked = cloud(monkeypatch, tmp_path, online=True)
    desktop.new_races()
    assert asked == ["download", "add"]                         # another PC may have built the race already
    asked.clear()
    desktop.new_races(fetched=True)                             # fetched as the app started: not again
    assert asked == ["add"]


def test_the_internet_check_has_a_time_limit(monkeypatch):
    """Looking up the cloud's address has no time limit of its own: the check gives up after its own."""
    def no_answer(*args, **kwargs):
        time.sleep(3)
        raise OSError("the name lookup failed")
    monkeypatch.setattr(desktop.socket, "create_connection", no_answer)
    start = time.monotonic()
    assert desktop.online({"region": "eu-north-1"}, timeout=0.2) is False
    assert time.monotonic() - start < 1

    class Connection:
        def close(self):
            pass
    monkeypatch.setattr(desktop.socket, "create_connection", lambda *a, **k: Connection())
    assert desktop.online({"region": "eu-north-1"}, timeout=0.5) is True
