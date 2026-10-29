"""desktop.py: the app in a window of its own. The tests don't open a window or start the app."""
import socket

import desktop


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
