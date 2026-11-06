"""Tests for F1 25 / F1 26: reading the game's packets and laps, and keeping online players' names out."""
import math
import time

import pytest

import game_strategy as gs
import game_telemetry as gt


def packet(fmt=2025, pid=0, uid=123, size=None):
    """Fake game packet: the header, then zeros up to the right size for the packet id."""
    head = gt.HEADER.pack(fmt, fmt % 100, 1, 0, 1, pid, uid, 12.5, 100, 100, 0, 255)
    return head + bytes((size if size is not None else gt.SIZES[fmt][pid]) - len(head))


def test_a_new_session_starts_with_its_first_packet():
    r = gt.Recorder(resume=False)
    r.handle(packet(pid=3), arrived=0.0, record=False)                       # an event packet
    assert list(r.sessions) == [123] and r.live_uid == 123


def event(code, fmt=2026):
    p = bytearray(packet(fmt=fmt, pid=3))
    p[gt.HEADER.size:gt.HEADER.size + 4] = code.encode()
    return bytes(p)


def test_a_paused_game_is_paused_not_over():
    """While paused the game sends nothing; when a session is over (the flag, or quitting to the menu) it sends
    "Session ended" first. So a quiet session that hasn't ended stays live, as paused."""
    now = time.time()
    r = gt.Recorder(resume=False)
    r.handle(packet(fmt=2026, pid=1), now - 61, record=False)                # the session packet
    r.handle(event("SSTA"), now - 60, record=False)                         # driving, then a minute without packets
    assert r.is_live(123) and r.is_paused(123)
    r.handle(packet(fmt=2026, pid=0), now, record=False)                     # you carry on
    assert r.is_live(123) and not r.is_paused(123)
    r.handle(event("SEND"), now - 60, record=False)                          # the session ended a minute ago
    assert not r.is_live(123) and not r.is_paused(123)
    r = gt.Recorder(resume=False)
    r.handle(packet(fmt=2026, pid=1), now - gt.PAUSE_MAX - 2, record=False)
    r.handle(event("SSTA"), now - gt.PAUSE_MAX - 1, record=False)            # the console was switched off
    assert not r.is_live(123)


def test_stray_packets_are_not_a_paused_session():
    """After a race the game can send a few packets under a new session number with no session packet; once they
    stop, that's not a paused session."""
    r = gt.Recorder(resume=False)
    r.handle(event("SSTA"), time.time() - 60, record=False)
    assert not r.is_paused(123) and not r.is_live(123)


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
    """The project has no human participants, so other people in an online session are renamed."""
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


def test_nothing_is_saved_before_the_game_says_if_the_session_is_online(tmp_path, monkeypatch):
    """The names arrive before the session packet that says the session is online, and are only hidden then, so
    nothing is saved before it, not even for a moment."""
    monkeypatch.setattr(gt, "GAME_DIR", tmp_path)
    r = gt.Recorder(resume=False)
    s = r.sessions[7] = gt.Session(7, 2025)
    s.last_seen, s.drivers = time.time(), {1: {"name": "RealGamertag", "code": "REA", "ai": False}}
    r.save_due(force=True)
    assert not (tmp_path / "7").exists()
    s.info["online"] = True                                                  # the session packet
    s.hide_other_players()
    r.save_due(force=True)
    saved = (tmp_path / "7" / "session.json").read_text(encoding="utf-8")
    assert "RealGamertag" not in saved and "Driver 2" in saved


def lap_data(lap, last_ms, status=2, pit=0):
    """One car in a lap data packet (only the fields the laps need)."""
    return dict(result_status=status, last_lap_ms=last_ms, lap=lap, sector=0, s1_ms=0, s1_min=0, s2_ms=0, s2_min=0,
                position=1, leader_ms=0, leader_min=0, pit_stops=0, pit_status=pit, lap_distance=10.0, driver_status=4)


def test_a_flashback_takes_away_what_came_after_it():
    """A flashback to before the pit lane removes the stop if the car then stays out, and a finish it undoes is
    recorded again."""
    s = gt.Session(1, 2025)
    s.on_lap_data(90.0, [lap_data(10, 90_000)])
    s.on_lap_data(100.0, [lap_data(10, 90_000, pit=1)])                      # into the pit lane on lap 10
    assert s.laps[0][10]["pitted"] and s.laps[0][10]["pit_entry"]
    s.on_flashback(95.0)                                                     # back to before it...
    s.on_lap_data(95.1, [lap_data(10, 90_000)])                              # ...and the car stays out
    assert not any(k in s.laps[0][10] for k in ("pitted", "pit_entry", "pit_exit"))
    f = gt.Session(2, 2025)
    f.on_lap_data(100.0, [lap_data(5, 90_000)])
    f.on_lap_data(190.0, [lap_data(5, 91_500, status=3)])                    # took the flag...
    f.on_flashback(150.0)                                                    # ...undone
    assert 5 not in f.laps[0]
    f.on_lap_data(150.1, [lap_data(5, 90_000)])
    f.on_lap_data(192.0, [lap_data(5, 91_200, status=3)])                    # and taken again
    assert f.laps[0][5]["time"] == 91.2 and f.laps[0][5]["t"] == 192.0


def test_the_last_lap_at_the_flag_is_kept():
    """At the flag the game stops counting laps, and it often sends the last lap's time a frame before it says the
    car has finished (in real races the player's last lap was being lost)."""
    s = gt.Session(1, 2025)
    s.on_lap_data(100.0, [lap_data(5, 90_000)])                              # on the last lap
    s.on_lap_data(190.0, [lap_data(5, 91_500)])                              # the time first, still racing...
    s.on_lap_data(190.1, [lap_data(5, 91_500, status=3)])                    # ...then "finished"
    assert s.laps[0][5]["time"] == 91.5 and s.laps[0][5]["t"] == 190.1
    garage = gt.Session(2, 2025)
    garage.on_lap_data(100.0, [lap_data(5, 90_000)])
    garage.on_lap_data(500.0, [lap_data(5, 90_000, status=3)])                # in the garage at the end: no new time
    assert 5 not in garage.laps.get(0, {})                                    # so no lap is made up


def test_the_games_first_pit_window_is_its_plan():
    """The game's strategy: its first pit window is the plan (the window moves after a stop, and is 0 after the
    flag)."""
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
    """At the flag the game gives every car the same end time; it's rebuilt from the lap before + the lap time."""
    rows = [{"Driver": d, "LapNumber": 4, "Time": t, "LapTime": 90.0} for d, t in [("A", 400.0), ("B", 401.0), ("C", 403.0)]]
    rows += [{"Driver": d, "LapNumber": 5, "Time": 500.0, "LapTime": lt} for d, lt in [("A", 98.0), ("B", 99.0), ("C", 97.5)]]
    out = gs.last_lap_times(rows)
    assert [r["Time"] for r in out if r["LapNumber"] == 5] == [498.0, 500.0, 500.5]
    assert [r["Time"] for r in out if r["LapNumber"] == 4] == [400.0, 401.0, 403.0]     # other laps untouched
