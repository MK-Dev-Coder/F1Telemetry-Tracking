"""F1 25 / F1 26: reading the game's packets, its laps, and keeping online players' names out."""
import math

import pytest

import game_strategy as gs
import game_telemetry as gt


def packet(fmt=2025, pid=0, uid=123, size=None):
    """A packet from the game: its header, then zeros up to the size the packet id should have."""
    head = gt.HEADER.pack(fmt, fmt % 100, 1, 0, 1, pid, uid, 12.5, 100, 100, 0, 255)
    return head + bytes((size if size is not None else gt.SIZES[fmt][pid]) - len(head))


def test_a_new_session_starts_with_its_first_packet():
    r = gt.Recorder(resume=False)
    r.handle(packet(pid=3), arrived=0.0, record=False)                       # an event packet
    assert list(r.sessions) == [123] and r.live_uid == 123


def test_bad_packets_are_skipped():
    r = gt.Recorder(resume=False)
    r.handle(b"too short", 0.0, record=False)
    r.handle(packet(fmt=2019, size=60), 0.0, record=False)                   # a game this app doesn't read
    r.handle(packet(uid=0), 0.0, record=False)                               # menus: no session
    assert r.sessions == {}
    r.handle(packet(pid=0, size=500), 0.0, record=False)                     # motion, but the wrong size
    assert r.bad_size == {(2019, 0, 60): 1, (2025, 0, 500): 1}
    assert r.error is None


def test_online_players_names_are_never_kept():
    """The project has no human participants: in an online session the other people are renamed."""
    s = gt.Session(1, 2025)
    s.info, s.player = {"online": True}, 0
    s.drivers = {0: {"name": "Me", "code": "MEE", "ai": False}, 1: {"name": "SomeoneOnline", "code": "SOM", "ai": False},
                 2: {"name": "Max Verstappen", "code": "VER", "ai": True}}
    s.hide_other_players()
    assert s.drivers[1]["name"] == "Driver 2" and s.drivers[1]["code"] == "D02"
    assert s.drivers[0]["name"] == "Me"                                      # the player
    assert s.drivers[2]["name"] == "Max Verstappen"                          # the game's own AI drivers


def test_offline_names_stay():
    s = gt.Session(1, 2025)
    s.info, s.drivers = {"online": False}, {1: {"name": "Friend", "code": "FRI", "ai": False}}
    s.hide_other_players()
    assert s.drivers[1]["name"] == "Friend"


def test_the_games_first_pit_window_is_its_plan():
    """The game's strategy: its first pit window is the plan (it moves on after a stop, and is 0 after the flag)."""
    r = gt.Recorder(resume=False)

    def session(ideal, latest, rejoin):
        p = bytearray(packet(pid=1))
        p[682:685] = bytes([ideal, latest, rejoin])
        r.handle(bytes(p), 0.0, record=False)
        return r.sessions[123].info
    assert session(0, 0, 0)["pit_plan"] is None
    assert session(12, 16, 5)["pit_plan"] == [12, 16]
    info = session(30, 34, 7)
    assert info["pit_plan"] == [12, 16] and info["pit_window"] == [30, 34] and info["pit_rejoin"] == 7
    assert session(0, 0, 0)["pit_plan"] == [12, 16]


def test_lap_times():
    assert gt.seconds(25_123, 1) == pytest.approx(85.123)                    # 1 min 25.123 s


def test_track_status_codes():
    assert gs.status_name("1") == "GREEN"
    assert gs.status_name("4") == "SAFETY CAR"
    assert gs.status_name("6") == gs.status_name("67") == "VSC"
    assert gs.status_name("45") == "RED FLAG"
    periods = [(150.0, 300.0, "4")]
    assert gs.track_status(100.0, 190.0, periods, []) == "4"                 # out during the lap
    assert gs.track_status(300.0, 390.0, periods, []) == "1"                 # back in when the lap starts
    assert gs.track_status(100.0, 190.0, periods, [120.0]) == "45"


def test_safety_car_periods_from_the_games_messages():
    events = [{"code": "SCAR", "text": "Formation lap deployed", "t": 10.0},
              {"code": "SCAR", "text": "Safety car deployed", "t": 100.0},
              {"code": "SCAR", "text": "Safety car returning", "t": 180.0},
              {"code": "SCAR", "text": "Safety car returned", "t": 200.0},
              {"code": "FTLP", "text": "Fastest lap", "t": 250.0},
              {"code": "SCAR", "text": "Virtual safety car deployed", "t": 300.0}]
    assert gs.neutral_periods(events) == [(100.0, 200.0, "4"), (300.0, math.inf, "6")]    # the VSC is still out


def test_the_last_laps_end_times_are_put_back():
    """At the flag the game gives every car the same end time; it comes back from the lap before + the lap time."""
    rows = [{"Driver": d, "LapNumber": 4, "Time": t, "LapTime": 90.0} for d, t in [("A", 400.0), ("B", 401.0), ("C", 403.0)]]
    rows += [{"Driver": d, "LapNumber": 5, "Time": 500.0, "LapTime": lt} for d, lt in [("A", 98.0), ("B", 99.0), ("C", 97.5)]]
    out = gs.last_lap_times(rows)
    assert [r["Time"] for r in out if r["LapNumber"] == 5] == [498.0, 500.0, 500.5]
    assert [r["Time"] for r in out if r["LapNumber"] == 4] == [400.0, 401.0, 403.0]     # other laps untouched
