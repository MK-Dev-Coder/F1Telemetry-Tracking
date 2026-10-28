"""F1 25 telemetry: listen to the game over the network, decode its packets, and keep every session.

The game (PS5, Xbox or PC) sends UDP packets to this PC while you drive. Turn it on in F1 25:
Settings > Telemetry Settings > UDP Telemetry: On, UDP IP Address: this PC's address, UDP Port: 20777,
UDP Format: 2025 (2024 also works). With the 2026 Season Pack choose 2026: it adds active aero, overtake mode
and the 2026 teams (those sessions show in the app's F1 26 section).

Run on its own to record without the app:   python game_telemetry.py
Re-decode every saved session:               python game_telemetry.py rebuild
The app (python api.py) also records while it runs.
"""
import atexit
import gzip
import itertools
import json
import os
import socket
import struct
import sys
import threading
import time
import zlib

from collections import OrderedDict

import pandas as pd

from config import DATA_DIR

GAME_DIR = DATA_DIR / "f1_25"                 # one folder per session
PORT = 20777
KEEP_RAW = True                               # also keep the raw packets (compressed), to re-decode later
SAVE_EVERY = 30                               # seconds between saves of a live session
RUN = time.strftime("%Y%m%d-%H%M%S")          # each run of the app writes its own raw packets file
EARLY_MAX = 5000                              # packets held before the session packet arrives (about 10 s' worth)

# ---------------------------------------------------------------------------------------------------
# Packet layouts, from EA's F1 25 UDP specification and its 2026 Season Pack version (all little-endian).
# The sizes are checked below, so a layout that doesn't add up stops the program instead of decoding nonsense.
# The 2026 format has room for 24 cars, sends g-forces in thousandths, the engine temperature in one byte and
# driver and team numbers in two, and adds a packet (16) with every car's active aero and overtake mode.
CARS = {2024: 22, 2025: 22, 2026: 24}         # car slots in each packet
HEADER = struct.Struct("<HBBBBBQfIIBB")       # format, year, major, minor, packet version, id, session uid,
                                              # session time, frame, overall frame, player car, 2nd player car
_MOTION = struct.Struct("<6f6h6f")            # world position x/y/z, velocity x/y/z, directions, g-forces, angles
MOTION_CAR = {2024: _MOTION, 2025: _MOTION, 2026: struct.Struct("<6f6h3h3f")}
LAP = struct.Struct("<IIHBHBHBHBfff15BHHBfB")
LAP_FIELDS = ["last_lap_ms", "current_lap_ms", "s1_ms", "s1_min", "s2_ms", "s2_min", "front_ms", "front_min",
              "leader_ms", "leader_min", "lap_distance", "total_distance", "sc_delta", "position", "lap", "pit_status",
              "pit_stops", "sector", "invalid", "penalties", "warnings", "cut_warnings", "drive_throughs", "stop_gos",
              "grid", "driver_status", "result_status", "pit_lane_timer", "pit_lane_ms", "pit_stop_ms",
              "serve_penalty", "speed_trap", "speed_trap_lap"]
_TELEMETRY = struct.Struct("<HfffBbHBBH4H4B4BH4f4B")
TELEMETRY = {2024: _TELEMETRY, 2025: _TELEMETRY, 2026: struct.Struct("<HfffBbHBBH4H4B4BB4f4B")}
TELEMETRY_FIELDS = (["speed", "throttle", "steer", "brake", "clutch", "gear", "rpm", "drs", "rev_pct", "rev_bits"]
                    + [f"brake_temp_{w}" for w in "RL RR FL FR".split()]
                    + [f"tyre_surface_{w}" for w in "RL RR FL FR".split()]
                    + [f"tyre_inner_{w}" for w in "RL RR FL FR".split()] + ["engine_temp"]
                    + [f"tyre_pressure_{w}" for w in "RL RR FL FR".split()] + [f"surface_{w}" for w in "RL RR FL FR".split()])
_STATUS = struct.Struct("<5B3f2H2BH3Bb2ffB3fB")
STATUS = {2024: _STATUS, 2025: _STATUS, 2026: struct.Struct("<5B3f2H2BH3Bb2ffB4fB")}   # 2026: + harvest limit
_STATUS_FIELDS = ["traction_control", "abs", "fuel_mix", "brake_bias", "pit_limiter", "fuel", "fuel_capacity",
                  "fuel_laps", "max_rpm", "idle_rpm", "max_gears", "drs_allowed", "drs_distance", "actual_compound",
                  "visual_compound", "tyre_age", "fia_flag", "power_ice", "power_mguk", "ers_store", "ers_mode",
                  "ers_harvest_mguk", "ers_harvest_mguh", "ers_deployed", "network_paused"]
STATUS_FIELDS = {2024: _STATUS_FIELDS, 2025: _STATUS_FIELDS,
                 2026: _STATUS_FIELDS[:-2] + ["ers_harvest_limit"] + _STATUS_FIELDS[-2:]}   # energy allowed per lap
DAMAGE = {2025: struct.Struct("<4f4B4B4B18B"), 2024: struct.Struct("<4f4B4B18B")}
DAMAGE[2026] = DAMAGE[2025]
DAMAGE_PARTS = ["front_left_wing", "front_right_wing", "rear_wing", "floor", "diffuser", "sidepod", "drs_fault",
                "ers_fault", "gearbox", "engine", "mguh_wear", "es_wear", "ce_wear", "ice_wear", "mguk_wear",
                "tc_wear", "engine_blown", "engine_seized"]
PARTICIPANT = {2026: struct.Struct("<BHHHBBB32sBBHBB12B"), 2025: struct.Struct("<7B32sBBHBB12B"),
               2024: struct.Struct("<7B48sBBHB")}
FINAL = {2025: struct.Struct("<7BId3B8B8B8B"), 2024: struct.Struct("<6BId3B8B8B8B")}
FINAL[2026] = FINAL[2025]
# 2026 only: active aero (0 = corner mode, 1 = straight mode), overtake mode, metres until each can be used
# (0 = not this lap), whether the car is built to the 2026 rules, and driving the wrong way
TELEMETRY2 = struct.Struct("<BBHBBHBB")
TELEMETRY2_FIELDS = ["aero_mode", "aero_available", "aero_distance", "overtake_available", "overtake_active",
                     "overtake_distance", "regs_2026", "wrong_way"]
LAP_HISTORY = struct.Struct("<IHBHBHBB")
SETUP = struct.Struct("<4B4f9B4fBf")           # wings, differential, camber, toe, suspension, brakes, pressures, fuel
SETUP_FIELDS = ["front_wing", "rear_wing", "diff_on_throttle", "diff_off_throttle", "front_camber", "rear_camber",
                "front_toe", "rear_toe", "front_suspension", "rear_suspension", "front_anti_roll_bar",
                "rear_anti_roll_bar", "front_ride_height", "rear_ride_height", "brake_pressure", "brake_bias",
                "engine_braking", "pressure_RL", "pressure_RR", "pressure_FL", "pressure_FR", "ballast", "fuel_load"]
TYRE_SET = struct.Struct("<7BhB")             # compound, wear, available, best session, life, delta, fitted
TIME_TRIAL = struct.Struct("<BBIIIIBBBBBB")   # car, team, lap and sector times (ms), assists, equal cars, setup, valid
FORECAST = struct.Struct("<BBBbbbbB")         # session type, minutes ahead, weather, track temp (and its change),
                                              # air temp (and its change), chance of rain (%)
SIZES = {2025: {0: 1349, 1: 753, 2: 1285, 3: 45, 4: 1284, 5: 1133, 6: 1352, 7: 1239, 8: 1042, 10: 1041, 11: 1460,
                12: 231, 13: 273, 14: 101},
         2024: {0: 1349, 1: 753, 2: 1285, 3: 45, 4: 1350, 5: 1133, 6: 1352, 7: 1239, 8: 1020, 10: 953, 11: 1460,
                12: 231, 13: 237, 14: 101},
         2026: {0: 1325, 1: 926, 2: 1399, 3: 45, 4: 1470, 5: 1233, 6: 1448, 7: 1445, 8: 1134, 10: 1133, 11: 1460,
                12: 231, 13: 273, 14: 101, 16: 269}}
assert HEADER.size == 29 and MOTION_CAR[2025].size == 60 and LAP.size == 57 and TELEMETRY[2025].size == 60
assert STATUS[2025].size == 55 and DAMAGE[2025].size == 46 and PARTICIPANT[2025].size == 57 and FINAL[2025].size == 46
assert TIME_TRIAL.size == 24 and 29 + 3 * TIME_TRIAL.size == SIZES[2025][14]
assert LAP_HISTORY.size == 14 and 29 + 22 * LAP.size + 2 == SIZES[2025][2] and FORECAST.size == 8
assert 29 + 127 + 64 * FORECAST.size + 2 + 4 == 674          # the forecast ends where the weekend number begins
assert 29 + 22 * TELEMETRY[2025].size + 3 == SIZES[2025][6] and 29 + 1 + 22 * PARTICIPANT[2025].size == SIZES[2025][4]
assert 29 + 22 * DAMAGE[2025].size == SIZES[2025][10] and 29 + 1 + 22 * FINAL[2025].size == SIZES[2025][8]
assert 29 + 22 * DAMAGE[2024].size == SIZES[2024][10] and 29 + 1 + 22 * PARTICIPANT[2024].size == SIZES[2024][4]
assert 29 + 22 * SETUP.size + 4 == SIZES[2025][5] and 29 + 1 + 20 * TYRE_SET.size + 1 == SIZES[2025][12]
assert 29 + 61 * 4 == SIZES[2025][13] and 29 + 52 * 4 == SIZES[2024][13]   # motion ex: 61 floats (F1 24: 52)
for _fmt, _n in CARS.items():                 # every format: each per-car packet is the header plus its car slots
    _s = SIZES[_fmt]
    assert 29 + _n * MOTION_CAR[_fmt].size == _s[0] and 29 + _n * LAP.size + 2 == _s[2]
    assert 29 + _n * TELEMETRY[_fmt].size + 3 == _s[6] and 29 + _n * STATUS[_fmt].size == _s[7]
    assert 29 + _n * DAMAGE[_fmt].size == _s[10] and 29 + 1 + _n * PARTICIPANT[_fmt].size == _s[4]
    assert 29 + _n * SETUP.size + 4 == _s[5] and len(STATUS_FIELDS[_fmt]) == len(STATUS[_fmt].unpack(bytes(STATUS[_fmt].size)))
assert 29 + 1 + 24 * FINAL[2026].size == SIZES[2026][8] and 29 + 24 * TELEMETRY2.size == SIZES[2026][16]
assert 753 + 2 + 2 * 8 * 8 + 1 + 1 + 4 * 8 + 4 + 5 == SIZES[2026][1]      # 2026 session: + aero and DRS zones

# What the numbers mean (EA's appendices)
TRACKS = {0: "Melbourne", 1: "Paul Ricard", 2: "Shanghai", 3: "Sakhir", 4: "Catalunya", 5: "Monaco", 6: "Montreal",
          7: "Silverstone", 8: "Hockenheim", 9: "Hungaroring", 10: "Spa-Francorchamps", 11: "Monza", 12: "Singapore",
          13: "Suzuka", 14: "Yas Marina", 15: "Austin", 16: "Interlagos", 17: "Spielberg", 18: "Sochi",
          19: "Mexico City", 20: "Baku", 21: "Sakhir Short", 22: "Silverstone Short", 23: "Austin Short",
          24: "Suzuka Short", 25: "Hanoi", 26: "Zandvoort", 27: "Imola", 28: "Portimao", 29: "Jeddah", 30: "Miami",
          31: "Las Vegas", 32: "Lusail", 39: "Silverstone Reverse", 40: "Spielberg Reverse", 41: "Zandvoort Reverse",
          42: "Madrid"}                       # 2026 Season Pack
SESSION_TYPES = {0: "Unknown", 1: "Practice 1", 2: "Practice 2", 3: "Practice 3", 4: "Short practice",
                 5: "Qualifying 1", 6: "Qualifying 2", 7: "Qualifying 3", 8: "Short qualifying", 9: "One-shot qualifying",
                 10: "Sprint shootout 1", 11: "Sprint shootout 2", 12: "Sprint shootout 3", 13: "Short sprint shootout",
                 14: "One-shot sprint shootout", 15: "Race", 16: "Race 2", 17: "Race 3", 18: "Time trial"}
WEATHER = {0: "Clear", 1: "Light cloud", 2: "Overcast", 3: "Light rain", 4: "Heavy rain", 5: "Storm"}
RAIN = {"Light rain", "Heavy rain", "Storm"}
TREND = {0: "up", 1: "down", 2: "same"}       # the forecast's temperature changes
VISUAL_TYRES = {16: "SOFT", 17: "MEDIUM", 18: "HARD", 7: "INTERMEDIATE", 8: "WET", 9: "SOFT", 10: "WET",
                11: "SOFT", 12: "SOFT", 13: "MEDIUM", 14: "HARD", 15: "WET"}
ACTUAL_TYRES = {16: "C5", 17: "C4", 18: "C3", 19: "C2", 20: "C1", 21: "C0", 22: "C6", 7: "Inter", 8: "Wet"}
TEAMS = {0: ("Mercedes", "27F4D2"), 1: ("Ferrari", "E8002D"), 2: ("Red Bull Racing", "3671C6"), 3: ("Williams", "64C4FF"),
         4: ("Aston Martin", "229971"), 5: ("Alpine", "0093CC"), 6: ("Racing Bulls", "6692FF"), 7: ("Haas F1 Team", "B6BABD"),
         8: ("McLaren", "FF8000"), 9: ("Sauber", "52E252"), 41: ("F1 Generic", "888888"), 104: ("My Team", "C9CFDB"),
         # The 2026 Season Pack's teams (the colours are only used when the game sends no livery colour)
         476: ("Mercedes", "27F4D2"), 477: ("Ferrari", "E8002D"), 478: ("Red Bull Racing", "3671C6"),
         479: ("Williams", "64C4FF"), 480: ("Aston Martin", "229971"), 481: ("Alpine", "0093CC"),
         482: ("Racing Bulls", "6692FF"), 483: ("Haas F1 Team", "B6BABD"), 484: ("McLaren", "FF8000"),
         485: ("Audi", "F50537"), 486: ("Cadillac", "E6E6E6")}
SHOOTOUTS = {10, 11, 12, 13, 14}              # the sprint shootout session types
SAFETY_CAR = {0: "GREEN", 1: "SAFETY CAR", 2: "VSC", 3: "FORMATION LAP"}
RESULT = {0: "INVALID", 1: "INACTIVE", 2: "ACTIVE", 3: "FINISHED", 4: "DNF", 5: "DSQ", 6: "NOT CLASSIFIED", 7: "RETIRED"}
EVENTS = {"SSTA": "Session started", "SEND": "Session ended", "FTLP": "Fastest lap", "RTMT": "Retirement",
          "DRSE": "DRS enabled", "DRSD": "DRS disabled", "TMPT": "Team mate in the pits", "CHQF": "Chequered flag",
          "RCWN": "Race winner", "PENA": "Penalty", "SPTP": "Speed trap", "STLG": "Start lights", "LGOT": "Lights out",
          "DTSV": "Drive-through served", "SGSV": "Stop-go served", "FLBK": "Flashback", "RDFL": "Red flag",
          "OVTK": "Overtake", "SCAR": "Safety car", "COLL": "Collision",
          # 2026 Season Pack: race control turns partial mode (less active aero) and overtake mode on and off
          "PMEN": "Partial mode on", "PMDI": "Partial mode off", "OVEN": "Overtake mode enabled",
          "OVDI": "Overtake mode disabled"}
PARTIAL_REASONS = {0: "wet track", 1: "safety car", 2: "red flag"}
SEVERITY = {0: "light", 1: "medium", 2: "heavy"}
QUIET_EVENTS = {"BUTN", "SPTP", "STLG", "FLBK"}   # button presses and the like: not worth listing
# The game's "penalty" event is also sent for warnings, laps taken away and retirements (in the recorded
# sessions 16 of the player's 21 were warnings: 13 for track limits, 3 for small collisions); only the real
# penalties are called penalties
PENALTY_TYPES = {0: "Drive-through penalty", 1: "Stop-go penalty", 2: "Grid penalty", 3: "Penalty reminder",
                 4: "Time penalty", 5: "Warning", 6: "Disqualified", 7: "Removed from formation lap",
                 8: "Parked too long", 9: "Tyre regulations", 10: "Lap invalidated", 11: "Lap invalidated",
                 12: "Lap invalidated", 13: "Lap invalidated", 14: "Lap invalidated", 15: "Lap invalidated",
                 16: "Retired", 17: "Black flag timer"}
REAL_PENALTIES = {0, 1, 2, 4, 6}
_TOKENS = itertools.count(1)                     # a number for every Session object made (see Session.version)


def seconds(ms_part, minutes_part=0):
    return (ms_part + minutes_part * 60000) / 1000


TT_TC, TT_GEARBOX = {0: "off", 1: "medium", 2: "full"}, {1: "manual", 2: "manual, suggested gear", 3: "automatic"}


def tt_lap(v):
    """One lap of the Time Trial packet: its time and sectors (seconds), the team, and the assists it was done with."""
    car, team, lap, s1, s2, s3, tc, gearbox, abs_, equal, custom, valid = v
    return {"car": car, "team": TEAMS.get(team, (f"Team {team}",))[0], "time": lap / 1000 or None,
            "s1": s1 / 1000 or None, "s2": s2 / 1000 or None, "s3": s3 / 1000 or None,
            "traction_control": TT_TC.get(tc, str(tc)), "gearbox": TT_GEARBOX.get(gearbox, str(gearbox)),
            "abs": bool(abs_), "equal_cars": bool(equal), "custom_setup": bool(custom), "valid": bool(valid)}


def short_name(name, index):
    words = [w for w in name.replace(".", " ").split() if w.isalpha()]
    return (words[-1][:3] if words else f"C{index:02d}").upper()


def _write(path, write):
    """Write a file in one go: to a temporary name first, then swap it in, so a crash never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _rows(file, columns):
    """A saved table as plain lists, in today's column order (missing columns and gaps become None)."""
    df = pd.read_parquet(file).reindex(columns=columns)
    return df.astype(object).where(df.notna(), None).values.tolist()


# ---------------------------------------------------------------------------------------------------
class Session:
    """Everything known about one game session (practice, qualifying, race or time trial)."""

    def __init__(self, uid, packet_format, test=False):
        self.uid, self.format, self.test = uid, packet_format, test
        self.started = self.last_seen = time.time()
        self.info = {}
        self.player = 0
        self.num_cars = 0
        self.drivers = {}                     # car index -> name, code, team, colour, ai
        self.lap_now = {}                     # car index -> latest lap data
        self.status_now = {}                  # car index -> latest car status
        self.damage_now = {}                  # car index -> latest damage
        self.modes_now = {}                   # car index -> latest active aero and overtake mode (2026 format)
        self.laps = {}                        # car index -> {lap number -> lap record}
        self.events, self.classification = [], []
        self.samples = []                     # the player's telemetry, every frame
        self.car_samples = []                 # the player's tyres, fuel and ERS, twice a second
        self.positions = []                   # every car's position, twice a second
        self.outline = {}                     # lap number -> the player's positions on that lap
        self.setup = {}                       # the player's car setup
        self.tyre_sets = {}                   # the player's tyre sets for the weekend
        self._g = self._slip = None           # the player's latest g-forces and wheel slip
        self._sector = {}                     # car index -> sector times of the lap in progress
        self._finished = set()                # cars whose last lap (at the flag) is already recorded
        self._overtake = {}                   # car index -> {lap: when overtake mode was first used on it} (2026)
        self._last_pos_t = self._last_car_t = -1.0
        self.packets = 0
        # (token, version) changes whenever the laps or the events change, so work done on them can be kept
        self.token, self.version = next(_TOKENS), 0

    # -- one decoded packet -----------------------------------------------------------------------
    def lap_record(self, car, lap):
        return self.laps.setdefault(car, {}).setdefault(lap, {"lap": lap})

    def on_lap_data(self, t, cars):
        # Every slot: when a car retires the game lowers its count of active cars, but the other cars
        # keep their slots (in a real Jeddah race the player, in slot 19, was lost after the first retirement)
        for i, d in enumerate(cars):
            if d["result_status"] in (0, 1):                     # an empty or inactive slot
                continue
            before = self.lap_now.get(i)
            new_time = bool(before) and d["last_lap_ms"] != before["last_lap_ms"]   # the game just set a lap time
            crossed = bool(before) and d["lap"] > before["lap"]  # crossed the line: the lap before is complete
            # At the flag the game stops counting laps: the last lap ends when the finished car gets its time.
            # A car already in the garage when the session ends gets no new time, so no lap is made up for it.
            flag = d["result_status"] == 3 and new_time and i not in self._finished
            if flag:
                self._finished.add(i)
            if crossed or flag:
                done = self.lap_record(i, before["lap"])
                s1, s2 = self._sector.get(i, (None, None))
                lap_time = d["last_lap_ms"] / 1000 if new_time and d["last_lap_ms"] else None
                if lap_time:
                    done.setdefault("time", lap_time)
                if s1 and s2 and lap_time and not done.get("s3"):
                    done.setdefault("s1", s1), done.setdefault("s2", s2)
                    done["s3"] = round(lap_time - done["s1"] - done["s2"], 3)
                done["t"] = round(t, 2)                          # when it ended (undone by a flashback to before)
                done["position"] = d["position"]
                done["gap_to_leader"] = seconds(d["leader_ms"], d["leader_min"])
                done["invalid"] = bool(before.get("invalid"))
                status = self.status_now.get(i, {})
                done["compound"] = VISUAL_TYRES.get(status.get("visual_compound"), "UNKNOWN")
                done["tyre_age"] = status.get("tyre_age")
                done["pit_stops"] = d["pit_stops"]
                done["pitted"] = done.get("pitted", False) or d["pit_stops"] > before["pit_stops"]
                if before["lap"] in self._overtake.get(i, ()):
                    done["overtake"] = True
                wear = self.damage_now.get(i, {}).get("wear")
                if wear:
                    done["tyre_wear"] = [round(w, 1) for w in wear]
                for k in ("weather", "air_temp", "track_temp"):  # the conditions on this lap, not at the end
                    done[k] = self.info.get(k)
                if "aero" in self.info:                          # 2026 format: active aero full or partial
                    done["aero"] = self.info["aero"]
                if i == self.player and self.tyre_sets.get("sets"):
                    done["set_deltas"] = self.set_deltas()
                self.version += 1
            if d["sector"] >= 1 and d["s1_ms"]:
                self._sector[i] = (seconds(d["s1_ms"], d["s1_min"]), self._sector.get(i, (None, None))[1])
            if d["sector"] >= 2 and d["s2_ms"]:
                self._sector[i] = (self._sector.get(i, (None, None))[0], seconds(d["s2_ms"], d["s2_min"]))
            if d["sector"] == 0 and before and d["lap"] > before["lap"]:
                self._sector[i] = (None, None)
            # In the pit lane. Not counted while leaving the garage before the line: in practice and
            # qualifying that out-lap is part of the next lap, which would then show as a pit lap.
            if d["pit_status"] and d["lap_distance"] >= 0 and d["driver_status"] != 0:
                rec = self.lap_record(i, d["lap"])
                rec["pitted"] = True
                # The lap the car came into the pit lane on (the in-lap), and the lap it came out on
                if before and not before["pit_status"]:
                    rec["pit_entry"] = True
                    self.version += 1
            elif before and before["pit_status"] and not d["pit_status"] and d["lap_distance"] >= 0:
                self.lap_record(i, d["lap"])["pit_exit"] = True
                self.version += 1
            self.lap_now[i] = d

    def on_history(self, car, laps, best_lap):
        for n, (lap_ms, s1, s1m, s2, s2m, s3, s3m, valid) in enumerate(laps, start=1):
            if not lap_ms:
                continue
            rec = self.lap_record(car, n)
            new = dict(time=lap_ms / 1000, s1=seconds(s1, s1m), s2=seconds(s2, s2m), s3=seconds(s3, s3m),
                       invalid=not (valid & 0x01))
            if any(rec.get(k) != v for k, v in new.items()):
                rec.update(new)
                self.version += 1
        self.drivers.setdefault(car, {})["best_lap_number"] = best_lap

    def on_telemetry(self, t, cars):
        p = cars[self.player] if self.player < len(cars) else None
        lap, m = self.lap_now.get(self.player), self.modes_now.get(self.player)
        if p is None or lap is None:
            return
        if self.on_a_lap(lap):
            last = self.samples[-1] if self.samples else None
            if last and last[1] == lap["lap"] and lap["current_lap_ms"] / 1000 < (last[10] or 0) - 1:
                # The lap timer went back without a new lap number: the game started this lap again.
                self.samples = [x for x in self.samples if x[1] != lap["lap"]]      # keep only the new attempt
                self.outline.pop(lap["lap"], None)
            self.samples.append([round(t, 3), lap["lap"], round(lap["lap_distance"], 1), p["speed"], round(p["throttle"], 3),
                                 round(p["brake"], 3), round(p["steer"], 3), p["gear"], p["rpm"], p["drs"],
                                 lap["current_lap_ms"] / 1000,
                                 *([round(v, 2) for v in self._g[:2]] if self._g else [None] * 2),       # g-forces
                                 *([round(v, 3) for v in self._slip] if self._slip else [None] * 4),     # wheel slip
                                 *((m["aero_mode"], m["overtake_active"]) if m else (None, None))])      # 2026 only
        if t - self._last_car_t >= 0.5:
            self._last_car_t = t
            st, dm = self.status_now.get(self.player, {}), self.damage_now.get(self.player, {})
            self.car_samples.append([round(t, 3), lap["lap"], st.get("fuel"), st.get("fuel_laps"), st.get("ers_store"),
                                     *(dm.get("wear") or [None] * 4),
                                     *[p[f"tyre_surface_{w}"] for w in "RL RR FL FR".split()],
                                     *[p[f"tyre_inner_{w}"] for w in "RL RR FL FR".split()],
                                     *[p[f"brake_temp_{w}"] for w in "RL RR FL FR".split()],
                                     round(lap["lap_distance"], 1),
                                     *[round(p[f"tyre_pressure_{w}"], 2) for w in "RL RR FL FR".split()],
                                     (st["ers_harvest_mguk"] + st["ers_harvest_mguh"]) if "ers_harvest_mguk" in st else None,
                                     st.get("ers_deployed")])

    def on_a_lap(self, lap):
        """Is the player driving a timed lap? The game's lap timer runs (not on the grid, nor driving around
        with no timing as seen after a trip to the garage), not in the garage, not on the out-lap before the
        line (practice and qualifying), not on the formation lap and not after the flag."""
        return (lap["current_lap_ms"] > 0 and lap["driver_status"] != 0 and lap["lap_distance"] >= 0
                and lap["result_status"] != 3 and self.info.get("safety_car") != "FORMATION LAP")

    def on_modes(self, t, cars):
        """2026 format: every car's active aero (straight or corner mode) and overtake mode. The laps a car
        used overtake mode on are marked when they end, and a car built to the 2026 rules puts the session in F1 26.
        The 2026 format sends this packet for the older cars too (with DRS, no overtake mode): those are skipped."""
        for i, m in enumerate(cars):
            lap = self.lap_now.get(i)
            if lap is None or not m["regs_2026"]:          # an empty slot, or a car built to the older rules
                continue
            self.modes_now[i] = m
            if m["overtake_active"] and lap["result_status"] == 2:     # racing (not after the flag)
                self._overtake.setdefault(i, {}).setdefault(lap["lap"], t)
        me = self.modes_now.get(self.player)
        if me and me["regs_2026"]:
            self.info["regs_2026"] = True

    def game(self):
        """Which part of the app the session belongs in: F1 26 (the 2026 Season Pack's cars) or F1 25. None until
        the first session packet: the first packets of a session come before it (it's sent twice a second)."""
        if "session_type_id" not in self.info:
            return None
        return "F1 26" if self.info.get("formula") == 13 or self.info.get("regs_2026") else "F1 25"

    def race_control(self):
        """2026 format: race control's latest word on overtake mode (OVEN on, OVDI off), kept in the session's info,
        so the page has it however many events came after it."""
        word = next((e["code"] for e in reversed(self.events) if e["code"] in ("OVEN", "OVDI")), None)
        if word:
            self.info["overtake_mode"] = word == "OVEN"
        else:
            self.info.pop("overtake_mode", None)

    def on_motion(self, t, cars):
        me, lap = cars[self.player], self.lap_now.get(self.player)
        self._g = me[12:15]                   # g-forces: sideways (cornering), forward (+ accelerating, - braking), up
        if lap is not None and self.on_a_lap(lap):
            self.outline.setdefault(lap["lap"], []).append([round(me[0], 1), round(me[2], 1), round(lap["lap_distance"], 1)])
        if t - self._last_pos_t >= 0.5:
            self._last_pos_t = t
            for i, c in enumerate(cars):                          # cars racing or finished (not retired ones)
                if self.lap_now.get(i, {}).get("result_status") in (2, 3):
                    self.positions.append([round(t, 2), i, round(c[0], 1), round(c[2], 1)])

    def on_flashback(self, t):
        """The player used a flashback: forget what was recorded after the moment the game went back to."""
        lap_again = next((x[1] for x in self.samples if x[0] > t), self.lap_now.get(self.player, {}).get("lap"))
        self.samples = [x for x in self.samples if x[0] <= t]
        self.car_samples = [x for x in self.car_samples if x[0] <= t]
        self.positions = [x for x in self.positions if x[0] <= t]
        self.events = [e for e in self.events if e["t"] <= t]
        for laps in self.laps.values():                           # laps finished after that moment
            for n in [n for n, r in laps.items() if r.get("t", 0) > t]:
                del laps[n]
        if lap_again is not None:                                 # the track outline of the laps driven again
            for n in [n for n in self.outline if n != "saved" and n >= lap_again]:
                del self.outline[n]
        self._sector.clear()
        for used in self._overtake.values():
            for n in [n for n, when in used.items() if when > t]:
                del used[n]
        self.race_control()
        self._last_pos_t = self._last_car_t = -1.0
        self.version += 1

    def lap_chart(self):
        """Every car's position at the end of each lap (for a lap chart)."""
        out = []
        for car, laps in self.laps.items():
            pts = [[n, r["position"]] for n, r in sorted(laps.items()) if r.get("t") is not None and r.get("position")]
            if pts:
                d = self.drivers.get(car, {})
                out.append({"car": car, "driver": d.get("code", f"C{car:02d}"), "colour": "#" + d.get("colour", "888888"),
                            "player": car == self.player, "positions": pts})
        return out

    def tyre_set_list(self):
        """The player's tyre sets for the weekend: compound, wear, whether it can still be used, and its pace.
        The game's "life span" is the laps left in that set; its "usable life" is the most laps the compound lasts."""
        sets, fitted = self.tyre_sets.get("sets", []), self.tyre_sets.get("fitted")
        return [{"set": k + 1, "compound": VISUAL_TYRES.get(v, "UNKNOWN"), "actual": ACTUAL_TYRES.get(a, ""), "wear": wear,
                 "available": bool(ok), "fitted": k == fitted, "laps_left": laps_left, "compound_life": most,
                 "best_for": SESSION_TYPES.get(session, ""), "delta": round(delta / 1000, 3)}
                for k, (a, v, wear, ok, session, laps_left, most, delta, _) in enumerate(sets) if v]

    def set_deltas(self):
        """The game's own word on each kind of tyre now: how much faster (-) or slower (+) a lap your best set of it
        would be than the set fitted (from your sets that can still be used), and which is fitted."""
        out = {}
        for t in self.tyre_set_list():
            if t["fitted"]:
                out["fitted"] = t["compound"]
            elif t["available"] and (t["compound"] not in out or t["delta"] < out[t["compound"]]):
                out[t["compound"]] = t["delta"]
        return out

    def hide_other_players(self):
        """Online sessions: other people's names are never kept (the project has no human participants)."""
        if self.info.get("online"):
            for i, d in self.drivers.items():
                if i != self.player and not d.get("ai", True) and d.get("name"):
                    d.update(name=f"Driver {i + 1}", code=f"D{i + 1:02d}")

    def is_race(self):
        return self.info.get("session_type_id") in (15, 16, 17)

    # -- views for the app ------------------------------------------------------------------------
    def best_times(self):
        """Fastest lap and sectors of anyone in the session (for purple)."""
        best = {"time": None, "s1": None, "s2": None, "s3": None}
        for laps in self.laps.values():
            for rec in laps.values():
                if rec.get("invalid"):
                    continue
                for k in best:
                    v = rec.get(k)
                    if v and (best[k] is None or v < best[k]):
                        best[k] = v
        return best

    def summary(self):
        me = self.drivers.get(self.player, {})
        mine = [r for r in self.laps.get(self.player, {}).values() if r.get("time") and not r.get("invalid")]
        return {"id": str(self.uid), "track": self.info.get("track", "Unknown track"),
                "type": self.info.get("session_type", "Session"), "started": self.started, "last_seen": self.last_seen,
                "player": me.get("name"), "cars": len(self.drivers),
                "laps": max(self.laps.get(self.player, {0: None}).keys(), default=0),
                "best_lap": min((r["time"] for r in mine), default=None), "test": self.test, "format": self.format,
                "type_id": self.info.get("session_type_id"), "weekend": self.info.get("weekend"), "game": self.game(),
                "weather": self.info.get("weather")}

    def leaderboard(self):
        best = self.best_times()
        rows = []
        for i, d in self.lap_now.items():
            drv, st, m = self.drivers.get(i, {}), self.status_now.get(i, {}), self.modes_now.get(i)
            laps = self.laps.get(i, {})
            finished = d["result_status"] == 3                   # at the flag the lap number stops going up
            done = [r for n, r in sorted(laps.items()) if (n < d["lap"] or finished) and r.get("time")]
            last = done[-1] if done else {}
            valid = [r for r in done if not r.get("invalid")]
            personal = {k: min((r[k] for r in valid if r.get(k)), default=None) for k in ("time", "s1", "s2", "s3")}

            def colour(k, v):
                if not v or last.get("invalid"):
                    return ""
                return "p" if best[k] and v <= best[k] + 1e-6 else "g" if personal[k] and v <= personal[k] + 1e-6 else "y"

            result = RESULT.get(d["result_status"], "")
            # A retired or finished car parked in the pit lane is still retired / finished, not pitting
            status = (result if result in ("DNF", "DSQ", "RETIRED", "FINISHED", "NOT CLASSIFIED") else "PIT"
                      if d["pit_status"] else "OUT LAP" if d["driver_status"] == 3 else "ON TRACK")
            rows.append({"car": i, "driver": drv.get("code", f"C{i:02d}"), "name": drv.get("name", ""),
                         "team": drv.get("team", ""), "colour": "#" + drv.get("colour", "888888"),
                         "player": i == self.player, "ai": drv.get("ai", True), "position": d["position"],
                         "lap": d["lap"], "status": status, "pit_stops": d["pit_stops"], "penalties": d["penalties"],
                         "gap": seconds(d["leader_ms"], d["leader_min"]) if d["position"] > 1 else None,
                         "interval": seconds(d["front_ms"], d["front_min"]) if d["position"] > 1 else None,
                         "last_lap": last.get("time"), "last_col": colour("time", last.get("time")),
                         "best_lap": personal["time"], "best_is_fastest": bool(personal["time"] and best["time"]
                                                                                and personal["time"] <= best["time"] + 1e-6),
                         "sectors": [{"time": last.get(k), "col": colour(k, last.get(k)), "best": personal[k]}
                                     for k in ("s1", "s2", "s3")],
                         "compound": VISUAL_TYRES.get(st.get("visual_compound"), "UNKNOWN"),
                         "actual_compound": ACTUAL_TYRES.get(st.get("actual_compound"), ""), "tyre_age": st.get("tyre_age"),
                         "invalid_now": bool(d["invalid"]), "lap_distance": d["lap_distance"],
                         "stints": self.stints(i),
                         "overtake": bool(m["overtake_active"]) if m else None})     # 2026 format only
        if self.is_race():
            return sorted(rows, key=lambda r: r["position"] or 99)
        # Practice, qualifying and time trial go by best lap: the game's own positions can stop updating
        # (they did in a real Q1), and its time gaps are only meant for races.
        rows.sort(key=lambda r: (r["best_lap"] is None, r["best_lap"] or 0, r["position"] or 99))
        first = rows[0]["best_lap"] if rows else None
        for k, r in enumerate(rows):
            ahead = rows[k - 1]["best_lap"] if k else None
            r["position"] = k + 1
            r["gap"] = round(r["best_lap"] - first, 3) if k and r["best_lap"] and first else None
            r["interval"] = round(r["best_lap"] - ahead, 3) if k and r["best_lap"] and ahead else None
        return rows

    def stints(self, car):
        """Tyre stints: a new one starts when the compound changes or the tyre age goes back down (new set)."""
        out, age_before = [], None
        for n, r in sorted(self.laps.get(car, {}).items()):
            c, age = r.get("compound"), r.get("tyre_age")
            if not c:
                continue
            new_set = age is not None and age_before is not None and age < age_before
            if out and out[-1]["compound"] == c and not new_set:
                out[-1]["to"] = n
            else:
                out.append({"compound": c, "from": n, "to": n})
            age_before = age
        return out

    def player_laps(self):
        best = self.best_times()
        race = self.is_race()                 # outside races the game's position at the line isn't reliable
        out = []
        for n, r in sorted(self.laps.get(self.player, {}).items()):
            if r.get("time"):
                out.append({"lap": n, "time": r["time"], "s1": r.get("s1"), "s2": r.get("s2"), "s3": r.get("s3"),
                            "invalid": bool(r.get("invalid")), "compound": r.get("compound"), "tyre_age": r.get("tyre_age"),
                            "position": r.get("position") if race else None, "tyre_wear": r.get("tyre_wear"),
                            "pitted": bool(r.get("pitted")), "overtake": bool(r.get("overtake")), "aero": r.get("aero"),
                            "weather": r.get("weather"), "track_temp": r.get("track_temp"), "air_temp": r.get("air_temp"),
                            "fastest": bool(best["time"] and r["time"] <= best["time"] + 1e-6 and not r.get("invalid"))})
        return out

    def car_now(self):
        st, dm = self.status_now.get(self.player, {}), self.damage_now.get(self.player, {})
        last = self.car_samples[-1] if self.car_samples else None
        return {"fuel": st.get("fuel"), "fuel_laps": st.get("fuel_laps"), "ers": st.get("ers_store"),
                "ers_mode": st.get("ers_mode"), "compound": VISUAL_TYRES.get(st.get("visual_compound"), "UNKNOWN"),
                "actual_compound": ACTUAL_TYRES.get(st.get("actual_compound"), ""), "tyre_age": st.get("tyre_age"),
                "wear": dm.get("wear"), "damage": {k: dm.get(k) for k in DAMAGE_PARTS[:6]},
                "surface_temp": last[9:13] if last else None, "inner_temp": last[13:17] if last else None,
                "brake_temp": last[17:21] if last else None, "pressure": last[22:26] if last else None,
                # energy this lap (joules): harvested, used, and the most the rules allow to harvest (2026 format)
                "ers_harvested": (st["ers_harvest_mguk"] + st["ers_harvest_mguh"]) if "ers_harvest_mguk" in st else None,
                "ers_deployed": st.get("ers_deployed"), "ers_limit": st.get("ers_harvest_limit"),
                "engine_wear": {k: dm.get(k) for k in ("ice_wear", "mguk_wear", "mguh_wear", "es_wear", "ce_wear", "tc_wear")},
                "modes": self.modes_now.get(self.player), "format": self.format,
                "drs_allowed": st.get("drs_allowed"), "drs_distance": st.get("drs_distance")}

    # -- saving and loading -----------------------------------------------------------------------
    def folder(self):
        return GAME_DIR / str(self.uid)

    def save(self):
        """Save the session. Each file is swapped in whole, so a crash or a full disk never leaves half a file."""
        folder = self.folder()
        folder.mkdir(parents=True, exist_ok=True)
        for name, rows, columns in [("telemetry", self.samples, SAMPLE_COLUMNS), ("car", self.car_samples, CAR_COLUMNS),
                                    ("positions", self.positions, POSITION_COLUMNS)]:
            table = pd.DataFrame(rows, columns=columns)
            _write(folder / f"{name}.parquet", lambda path: table.to_parquet(path, index=False))
        state = {"uid": str(self.uid), "format": self.format, "test": self.test, "started": self.started,
                 "last_seen": self.last_seen, "info": self.info, "player": self.player, "num_cars": self.num_cars,
                 "drivers": self.drivers, "lap_now": self.lap_now, "status_now": self.status_now,
                 "damage_now": self.damage_now, "modes_now": self.modes_now, "laps": self.laps, "events": self.events,
                 "classification": self.classification, "packets": self.packets,
                 "outline": self.best_outline(), "setup": self.setup, "tyre_sets": self.tyre_sets,
                 "overtake_laps": self._overtake}
        _write(folder / "session.json", lambda path: path.write_text(json.dumps(state), encoding="utf-8"))

    def best_outline(self):
        """The player's path on one complete lap: the shape of the track for the map."""
        full = {n: pts for n, pts in self.outline.items() if len(pts) > 50 and pts[0][2] < 200}
        if not full:                          # no lap started cleanly at the line yet (e.g. lap 1 after the
            some = [p for p in self.outline.values() if len(p) > 50]      # formation lap): use the longest one
            return max(some, key=len) if some else []
        lap = max(full, key=lambda n: full[n][-1][2] - full[n][0][2])       # the one covering the most track
        return full[lap]

    @classmethod
    def load(cls, folder, light=False):
        """A saved session. light=True skips the telemetry files (enough for the session list)."""
        state = json.loads((folder / "session.json").read_text(encoding="utf-8"))
        s = cls(int(state["uid"]), state["format"], state.get("test", False))
        s.started, s.last_seen, s.info = state["started"], state["last_seen"], state["info"]
        s.player, s.num_cars, s.packets = state["player"], state["num_cars"], state.get("packets", 0)
        intkeys = lambda d: {int(k): v for k, v in d.items()}
        s.drivers, s.lap_now = intkeys(state["drivers"]), intkeys(state["lap_now"])
        s.status_now, s.damage_now = intkeys(state["status_now"]), intkeys(state["damage_now"])
        s.modes_now = intkeys(state.get("modes_now", {}))
        s._overtake = {int(c): intkeys(used) for c, used in state.get("overtake_laps", {}).items()}
        s.laps = {int(c): intkeys(laps) for c, laps in state["laps"].items()}
        s.events, s.classification = state["events"], state["classification"]
        s.outline = {"saved": state.get("outline", [])}
        s.setup, s.tyre_sets = state.get("setup", {}), state.get("tyre_sets", {})
        s._finished = {i for i, d in s.lap_now.items() if d["result_status"] == 3}
        if light:
            return s
        for name, attr, columns in [("telemetry", "samples", SAMPLE_COLUMNS), ("car", "car_samples", CAR_COLUMNS),
                                    ("positions", "positions", POSITION_COLUMNS)]:
            file = folder / f"{name}.parquet"
            if file.exists():
                setattr(s, attr, _rows(file, columns))
        return s


SAMPLE_COLUMNS = (["t", "lap", "distance", "speed", "throttle", "brake", "steer", "gear", "rpm", "drs",
                   "lap_time",                # seconds into the lap, from the game's own lap timer
                   "g_lat", "g_long"]         # g-forces: cornering, and accelerating (+) / braking (-)
                  + [f"slip_{w}" for w in "RL RR FL FR".split()]    # wheel slip: - locking up, + spinning
                  + ["aero", "overtake"])     # 2026 format: straight mode (1) or corner mode (0), overtake mode on
POSITION_COLUMNS = ["t", "car", "x", "z"]
CAR_COLUMNS = (["t", "lap", "fuel", "fuel_laps", "ers"] + [f"wear_{w}" for w in "RL RR FL FR".split()]
               + [f"surface_{w}" for w in "RL RR FL FR".split()] + [f"inner_{w}" for w in "RL RR FL FR".split()]
               + [f"brake_{w}" for w in "RL RR FL FR".split()] + ["distance"]
               + [f"pressure_{w}" for w in "RL RR FL FR".split()]
               + ["ers_harvested", "ers_deployed"])   # the energy counters of the lap (joules)


# ---------------------------------------------------------------------------------------------------
class Recorder:
    """Receives the game's packets and keeps every session up to date (thread-safe)."""

    def __init__(self, resume=True):
        self.lock = threading.RLock()
        self.resume = resume                  # carry on with a saved session after a restart (not when rebuilding)
        self.sessions = {}                    # uid -> Session (the ones seen since the program started)
        self.live_uid = None
        self.packets = 0
        self.bad_size = {}                    # (format, packet id, size) -> count, when a layout doesn't match
        self.last_packet = None
        self.listening, self.error, self.save_error = False, None, None
        self._raw = {}
        self._early = {}                      # uid -> packets that came before the session packet (see handle)
        self._last_save = {}                  # uid -> when a save was last tried
        self._saved_seen = {}                 # uid -> the session's last packet time in its last good save
        self._failed = set()                  # sessions whose last save failed (tried again sooner)
        self._loaded, self._loaded_lock = OrderedDict(), threading.Lock()   # saved sessions recently opened
        self._sprint = {}                     # uid -> is this race a sprint (see is_sprint)
        self._started = False

    # -- network ----------------------------------------------------------------------------------
    def start(self, port=PORT):
        if self._started:                     # only one listener, however often it's asked
            return
        self._started = True
        atexit.register(self.close)
        threading.Thread(target=self._listen, args=(port,), daemon=True, name="f1-25-recorder").start()

    def _listen(self, port):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("0.0.0.0", port))      # every network card, so a PS5 or Xbox on the network can reach it
            sock.settimeout(1.0)
        except OSError as err:
            self.error = f"Can't listen on port {port}: {err}"
            return
        self.listening = True
        while True:
            try:
                try:
                    data, _ = sock.recvfrom(4096)
                except socket.timeout:
                    self.save_due()
                    continue
                self.handle(data, time.time())
            except Exception as err:          # one bad packet or a failed save must never stop the recording
                self.error = f"{type(err).__name__}: {err}"

    # -- decoding ---------------------------------------------------------------------------------
    def handle(self, data, arrived, record=True):
        if len(data) < HEADER.size:
            return
        fmt, year, major, minor, _, pid, uid, t, _, _, player, _ = HEADER.unpack_from(data)
        if fmt not in SIZES:
            self.bad_size[(fmt, pid, len(data))] = self.bad_size.get((fmt, pid, len(data)), 0) + 1
            return
        with self.lock:
            self.packets += 1
            if uid == 0:                         # menus and result screens send some packets without a session:
                return                           # button presses, and the race's penalties sent again
            self.last_packet = arrived
            s = self.sessions.get(uid)
            if s is None:                        # a session we haven't seen since the app started
                s = self._resume(uid) if self.resume else None
                if s is None:
                    s = Session(uid, fmt, test=(major == 0 and minor == 0))
                    s.started = arrived
                self.sessions[uid] = s
            s.packets += 1
            s.format = max(s.format, fmt)        # the menu's button presses come in the 2025 format even with 2026 set
            s.last_seen, s.player = arrived, player if player < CARS[fmt] else s.player
            self.live_uid = uid
            # Raw packets are kept once the session is known to be offline: online ones hold other people's names.
            # Until the session packet says which (within a second), they wait in memory, so the first ones
            # ("Session started", the first session packet) are in the raw file too and a rebuild has them
            if record and KEEP_RAW:
                online = s.info.get("online")
                if online is False:
                    self._write_raw(s, data, arrived)
                elif online is None and len(self._early.setdefault(uid, [])) < EARLY_MAX:
                    self._early[uid].append((arrived, data))
            expected = SIZES[fmt].get(pid)
            if expected is None:                 # packets this app doesn't use (setups, lobby, tyre sets...)
                return
            if len(data) != expected:
                self.bad_size[(fmt, pid, len(data))] = self.bad_size.get((fmt, pid, len(data)), 0) + 1
                return
            try:
                self._decode(s, fmt, pid, t, data)
            except (struct.error, IndexError, KeyError, ValueError) as err:
                self.error = f"Packet {pid} could not be decoded: {err}"
            if uid in self._early and s.info.get("online") is not None:      # now known: keep them, or forget
                for when, early in self._early.pop(uid):                     # them if the session is online
                    if s.info["online"] is False:
                        self._write_raw(s, early, when)
            if record:
                self.save_due()

    @staticmethod
    def _resume(uid):
        """If the app was restarted in the middle of a session, carry on from what was saved,
        instead of starting again and saving over it."""
        folder = GAME_DIR / str(uid)
        if not (folder / "session.json").exists():
            return None
        try:
            return Session.load(folder)
        except (OSError, ValueError, KeyError):
            return None

    def _decode(self, s, fmt, pid, t, data):
        o, n = HEADER.size, CARS[fmt]
        if pid == 0:                                             # motion: positions on track
            layout = MOTION_CAR[fmt]
            cars = [layout.unpack_from(data, o + i * layout.size) for i in range(n)]
            if fmt >= 2026:                                      # g-forces in thousandths
                cars = [c[:12] + tuple(g / 1000 for g in c[12:15]) + c[15:] for c in cars]
            s.on_motion(t, cars)
        elif pid == 1:                                           # session: track, type, weather
            (weather, track_temp, air_temp, total_laps, length, stype, track, formula, time_left, duration, pit_limit,
             paused, *_rest) = struct.unpack_from("<BbbBHBbBHHBBBBBB", data, o)
            sc, network = struct.unpack_from("<BB", data, o + 19 + 105)
            ideal, latest, rejoin = struct.unpack_from("<BBB", data, 682)     # the game's pit window and rejoin place
            plan = s.info.get("pit_plan") or ([ideal, latest] if ideal or latest else None)   # its first window: the plan
            s2_start, s3_start = struct.unpack_from("<ff", data, 745)
            weekend = struct.unpack_from("<I", data, 674)[0]     # the same for every session of one weekend
            # The weather forecast: this session's samples (the list also holds the weekend's other sessions)
            forecast = {}
            for k in range(min(data[o + 126], 64)):
                for_type, ahead, w, track_t, track_ch, air_t, air_ch, rain = FORECAST.unpack_from(data, o + 127 + k * FORECAST.size)
                if for_type == stype:
                    forecast.setdefault(ahead, {"in": ahead, "weather": WEATHER.get(w, ""), "rain": rain,
                                                "track_temp": track_t, "air_temp": air_t,
                                                "track_trend": TREND.get(track_ch), "air_trend": TREND.get(air_ch)})
            s.info.update(track=TRACKS.get(track, f"Track {track}"), track_id=track, session_type=SESSION_TYPES.get(stype, "Session"),
                          session_type_id=stype, total_laps=total_laps, track_length=length, weather=WEATHER.get(weather, ""),
                          track_temp=track_temp, air_temp=air_temp, safety_car=SAFETY_CAR.get(sc, "GREEN"),
                          online=bool(network), time_left=time_left, pit_window=[ideal, latest], pit_rejoin=rejoin, pit_plan=plan,
                          sector_starts=[0.0, round(s2_start, 1), round(s3_start, 1)], paused=bool(paused),
                          weekend=weekend, formula=formula, forecast=[forecast[k] for k in sorted(forecast)],
                          forecast_accuracy="approximate" if data[o + 639] == 1 else "perfect")
            if fmt >= 2026:                                      # active aero zones, DRS zones, start reaction
                aero, n_full = struct.unpack_from("<BB", data, 753)
                zones = lambda at, count, most: [[round(a, 4), round(b, 4)] for a, b in
                                                 itertools.batched(struct.unpack_from(f"<{2 * most}f", data, at), 2)][:count]
                reaction = struct.unpack_from("<f", data, 917)[0]
                s.info.update(aero="PARTIAL" if aero == 1 else "FULL", aero_zones=zones(755, n_full, 8),
                              aero_zones_partial=zones(820, data[819], 8), drs_zones=zones(885, data[884], 4),
                              start_reaction=round(reaction, 3) if reaction > 0 else None)
            s.hide_other_players()
        elif pid == 2:                                           # lap data: positions, lap times, pit status
            cars = [dict(zip(LAP_FIELDS, LAP.unpack_from(data, o + i * LAP.size))) for i in range(n)]
            s.on_lap_data(t, cars)
        elif pid == 3:                                           # events
            code = data[o:o + 4].decode("ascii", "replace")
            if code == "FLBK":                                   # flashback: frame number, then the session time
                s.on_flashback(struct.unpack_from("<If", data, o + 4)[1])     # the game went back to
            if code in QUIET_EVENTS:
                return
            event = {"t": round(t, 2), "code": code, "text": EVENTS.get(code, code),
                     "lap": s.lap_now.get(s.player, {}).get("lap")}
            name = lambda i: s.drivers.get(i, {}).get("code", f"Car {i}")
            if code == "FTLP":
                i, lap_time = struct.unpack_from("<Bf", data, o + 4)
                event["text"] = f"Fastest lap: {name(i)} {lap_time // 60:.0f}:{lap_time % 60:06.3f}"
            elif code in ("RTMT", "RCWN", "TMPT", "DTSV", "SGSV"):
                event["text"] = f"{EVENTS[code]}: {name(data[o + 4])}"
            elif code in ("OVTK", "COLL"):
                a, b, severity = struct.unpack_from("<BBB", data, o + 4)
                event["text"] = f"{name(a)} passes {name(b)}" if code == "OVTK" else f"Collision: {name(a)} and {name(b)}"
                event["cars"] = [a, b]
                if code == "COLL" and fmt >= 2026 and severity in SEVERITY:         # 2026: how hard they hit
                    event["text"] = f"{SEVERITY[severity].capitalize()} collision: {name(a)} and {name(b)}"
                    event["severity"] = severity
            elif code == "PMEN" and (data[o + 4] in (1, 2) or data[o + 4] == 0 and s.info.get("weather") in RAIN):
                # 0 (wet track) is also what a game without the reason byte (spec 1.1) sends: believed in the rain only
                event["text"] = f"Partial mode on ({PARTIAL_REASONS[data[o + 4]]})"
                event["reason"] = PARTIAL_REASONS[data[o + 4]]
            elif code == "PENA":
                ptype, infringement, i, other, secs, lap, _ = struct.unpack_from("<7B", data, o + 4)
                event["text"] = f"{PENALTY_TYPES.get(ptype, 'Penalty')}: {name(i)}" + (f", {secs} s" if 0 < secs < 255 else "")
                event.update(car=i, penalty=ptype, seconds=secs if 0 < secs < 255 else None)
            elif code == "SCAR":
                kind, phase = struct.unpack_from("<BB", data, o + 4)
                what = {1: "Safety car", 2: "Virtual safety car", 3: "Formation lap"}.get(kind, "Safety car")
                event["text"] = f"{what} " + {0: "deployed", 1: "returning", 2: "returned", 3: "resume race"}.get(phase, "")
            s.events.append(event)
            s.version += 1
            if code in ("OVEN", "OVDI"):
                s.race_control()
        elif pid == 4:                                           # participants: names and teams
            layout = PARTICIPANT[fmt]
            s.num_cars = data[o]
            for i in range(n):
                p = layout.unpack_from(data, o + 1 + i * layout.size)
                if i >= s.num_cars and i not in s.drivers:        # slots keep their car after a retirement
                    continue
                ai, driver_id, _, team_id, _, number, _, raw_name, *more = p
                name = raw_name.split(b"\0")[0].decode("utf-8", "replace").strip() or f"Car {i}"
                team, colour = TEAMS.get(team_id, (f"Team {team_id}", "888888"))
                if fmt >= 2025 and more[4] > 0:                  # the car's own livery colour
                    colour = "{:02X}{:02X}{:02X}".format(*more[5:8])
                s.drivers.setdefault(i, {}).update(name=name, code=short_name(name, i), team=team, colour=colour,
                                                   ai=bool(ai), number=number)
            s.hide_other_players()
        elif pid == 6:                                           # car telemetry: speed, throttle, brake, gear
            layout = TELEMETRY[fmt]
            s.on_telemetry(t, [dict(zip(TELEMETRY_FIELDS, layout.unpack_from(data, o + i * layout.size))) for i in range(n)])
        elif pid == 7:                                           # car status: tyres, fuel, ERS
            layout, fields = STATUS[fmt], STATUS_FIELDS[fmt]
            for i in range(n):
                s.status_now[i] = dict(zip(fields, layout.unpack_from(data, o + i * layout.size)))
        elif pid == 8:                                           # final classification
            layout = FINAL[fmt]
            count = data[o]
            s.classification = []
            for i in range(count):
                f = layout.unpack_from(data, o + 1 + i * layout.size)
                pos, laps, grid, points, stops, result = f[:6]
                best_ms, total = f[6:8] if fmt == 2024 else f[7:9]
                s.classification.append({"car": i, "driver": s.drivers.get(i, {}).get("code", f"C{i:02d}"),
                                         "position": pos, "laps": laps, "grid": grid, "points": points, "pit_stops": stops,
                                         "result": RESULT.get(result, ""), "best_lap": best_ms / 1000 or None,
                                         "race_time": round(total, 3)})
            s.classification.sort(key=lambda c: c["position"] or 99)
        elif pid == 10:                                          # damage: tyre wear, wings
            layout = DAMAGE[fmt]
            for i in range(n):
                v = layout.unpack_from(data, o + i * layout.size)
                parts = v[-18:]
                s.damage_now[i] = {"wear": [round(w, 1) for w in v[:4]], **dict(zip(DAMAGE_PARTS, parts))}
        elif pid == 5:                                           # car setups (only the player's is kept)
            s.setup = {k: round(v, 3) if isinstance(v, float) else v
                       for k, v in zip(SETUP_FIELDS, SETUP.unpack_from(data, o + s.player * SETUP.size))}
        elif pid == 12:                                          # one car's tyre sets for the weekend
            if data[o] == s.player:
                s.tyre_sets = {"sets": [TYRE_SET.unpack_from(data, o + 1 + k * TYRE_SET.size) for k in range(20)],
                               "fitted": data[o + 1 + 20 * TYRE_SET.size]}
        elif pid == 13:                                          # the player's car in detail: wheel slip ratio
            s._slip = struct.unpack_from("<4f", data, o + 64)    # after suspension position, speed, accel., wheel speed
        elif pid == 14:                                          # Time Trial: your best, your personal best, your rival
            s.info["time_trial"] = {name: tt_lap(TIME_TRIAL.unpack_from(data, o + k * TIME_TRIAL.size))
                                    for k, name in enumerate(("session_best", "personal_best", "rival"))}
        elif pid == 11:                                          # session history: every lap time of one car
            car, num_laps, _, best_lap, *_ = struct.unpack_from("<7B", data, o)
            laps = [LAP_HISTORY.unpack_from(data, o + 7 + k * 14) for k in range(min(num_laps, 100))]
            if car < n:
                s.on_history(car, laps, best_lap)
        elif pid == 16:                                          # 2026: active aero and overtake mode
            s.on_modes(t, [dict(zip(TELEMETRY2_FIELDS, TELEMETRY2.unpack_from(data, o + i * TELEMETRY2.size)))
                           for i in range(n)])

    # -- saving -----------------------------------------------------------------------------------
    def _write_raw(self, s, data, arrived):
        f = self._raw.get(s.uid)
        if f is None:
            s.folder().mkdir(parents=True, exist_ok=True)
            f = self._raw[s.uid] = gzip.open(s.folder() / f"packets-{RUN}.bin.gz", "ab", compresslevel=3)
        f.write(struct.pack("<dH", arrived, len(data)) + data)

    def close(self):
        """Save everything and close the raw files properly (when the app stops)."""
        self.save_due(force=True)
        with self.lock:
            for f in self._raw.values():
                try:
                    f.close()
                except OSError:
                    pass
            self._raw.clear()

    def save_due(self, force=False):
        """Save the sessions that changed: every SAVE_EVERY seconds while live, and once when they go quiet.
        A save that fails (e.g. OneDrive or antivirus holding a file) is tried again 5 seconds later."""
        with self.lock:
            now = time.time()
            for uid, s in self.sessions.items():
                if s.last_seen <= self._saved_seen.get(uid, 0):
                    continue                     # nothing new since the last good save
                since = now - self._last_save.get(uid, 0)
                wait = 5 if uid in self._failed else SAVE_EVERY
                if force or since >= wait or (now - s.last_seen > 5 and since >= 5):
                    self._last_save[uid], seen = now, s.last_seen
                    try:
                        s.save()
                        if uid in self._raw:
                            self._raw[uid].flush()
                    except OSError as err:
                        self.save_error = f"session {uid}: {err}"
                        self._failed.add(uid)
                        continue
                    self._failed.discard(uid)
                    self._saved_seen[uid], self.save_error = seen, None

    # -- reading sessions for the app ------------------------------------------------------------------
    def get(self, uid):
        """A session: the live one from memory, or a saved one. The last two saved sessions opened are kept,
        so a page asking for its leaderboard, telemetry, map and strategy reads the files once, not each time."""
        with self.lock:
            if int(uid) in self.sessions:
                return self.sessions[int(uid)]
        file = GAME_DIR / str(uid) / "session.json"
        if not file.exists():
            return None
        key = (str(uid), file.stat().st_mtime_ns)            # a new save or a rebuild changes the file time
        with self._loaded_lock:
            s = self._loaded.get(key)
            if s is not None:
                self._loaded.move_to_end(key)
                return s
        s = Session.load(file.parent)                          # outside the locks: this takes a moment
        with self._loaded_lock:
            self._loaded[key] = s
            while len(self._loaded) > 2:
                self._loaded.popitem(last=False)
        return s

    def all_sessions(self):
        with self.lock:
            out = {str(uid): s.summary() for uid, s in self.sessions.items()}
        if GAME_DIR.exists():
            for folder in GAME_DIR.iterdir():         # "0" held menu packets (older versions): not a session
                if folder.name not in out and folder.name != "0" and (folder / "session.json").exists():
                    try:
                        out[folder.name] = Session.load(folder, light=True).summary()
                    except (OSError, ValueError, KeyError):
                        continue
        return sorted(out.values(), key=lambda s: s["started"], reverse=True)

    def setups(self, track=None):
        """Every session with your setup (at one track, or all), with its best lap: for comparing setups."""
        with self.lock:
            out = {str(uid): s for uid, s in self.sessions.items()}
        if GAME_DIR.exists():
            for folder in GAME_DIR.iterdir():
                if folder.name not in out and folder.name != "0" and (folder / "session.json").exists():
                    try:
                        out[folder.name] = Session.load(folder, light=True)
                    except (OSError, ValueError, KeyError):
                        continue
        rows = []
        for s in out.values():
            with self.lock:
                x, setup = s.summary(), dict(s.setup)
            if setup and not x["test"] and (track is None or x["track"] == track):
                rows.append({**x, "setup": setup})
        return sorted(rows, key=lambda r: r["started"], reverse=True)

    def is_sprint(self, s):
        """Is this race a sprint? In F1 25 the sprint is a "Race" session too; it's the one in a weekend that
        also has a sprint shootout (the sessions of one weekend share the game's weekend number)."""
        if s.uid in self._sprint:
            return self._sprint[s.uid]
        weekend, sprint = s.info.get("weekend"), False
        if s.info.get("session_type_id") == 15 and weekend:
            sprint = any(x.get("weekend") == weekend and x.get("type_id") in SHOOTOUTS for x in self.all_sessions())
        if "session_type_id" in s.info:                      # known once the session packet has arrived
            self._sprint[s.uid] = sprint
        return sprint

    def is_live(self, uid):
        return self.last_packet is not None and str(self.live_uid) == str(uid) and time.time() - self.last_packet < 5

    def status(self):
        return {"listening": self.listening, "port": PORT, "error": self.error, "save_error": self.save_error,
                "packets": self.packets,
                "seconds_since_last_packet": None if self.last_packet is None else round(time.time() - self.last_packet, 1),
                "live_session": str(self.live_uid) if self.live_uid and self.is_live(self.live_uid) else None,
                "pc_addresses": local_addresses(), "suggested_address": suggested_address(),
                "wrong_size_packets": [{"format": f, "packet": p, "size": n, "count": c}
                                       for (f, p, n), c in sorted(self.bad_size.items())][:10]}


def local_addresses():
    """This PC's addresses on the home network: what to type into the game's UDP IP Address setting."""
    found = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.168.0.1", 9))     # nothing is sent: this only picks the network card
            found.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        found.update(a for a in socket.gethostbyname_ex(socket.gethostname())[2] if not a.startswith("127."))
    except OSError:
        pass
    return sorted(found)


def suggested_address():
    """The address to type into the game: a home-network one (192.168.x.x, 10.x.x.x or 172.16-31.x.x),
    not a VPN address."""
    home = [a for a in local_addresses() if a.startswith(("192.168.", "10."))
            or (a.startswith("172.") and 16 <= int(a.split(".")[1]) <= 31)]
    return (home or local_addresses() or [None])[0]


def raw_files(folder):
    """A session's raw packet files, oldest first (one per run of the app; packets.bin.gz is the old name)."""
    return sorted(folder.glob("packets*.bin.gz"), key=lambda f: (f.name != "packets.bin.gz", f.name))


def _gzip_parts(blob):
    """The data in every gzip part of a file. A part cut off because the app was closed while writing
    gives what was written before the cut, and reading carries on with the next part."""
    i = 0
    while (start := blob.find(b"\x1f\x8b\x08", i)) >= 0:
        part, out, pos = zlib.decompressobj(16 + zlib.MAX_WBITS), [], start
        try:
            while pos < len(blob) and not part.eof:
                out.append(part.decompress(blob[pos:pos + 65536]))
                pos = min(pos + 65536, len(blob))
        except zlib.error:
            pass
        yield b"".join(out)
        i = pos - len(part.unused_data) if part.eof else start + 3


def _records(data):
    o = 0
    while o + 10 <= len(data):
        arrived, n = struct.unpack_from("<dH", data, o)
        if not (HEADER.size <= n <= 2048 and 1.5e9 < arrived < 5e9) or o + 10 + n > len(data):
            return                            # the end of what was written (or a damaged spot)
        yield arrived, data[o + 10:o + 10 + n]
        o += 10 + n


def read_raw(path):
    """Every packet saved for a session (a session folder, or one raw file)."""
    for f in raw_files(path) if path.is_dir() else [path]:
        for data in _gzip_parts(f.read_bytes()):
            yield from _records(data)


def rebuild():
    """Decode every saved session again from its raw packets (after a fix to the decoding)."""
    for folder in sorted(GAME_DIR.iterdir()) if GAME_DIR.exists() else []:
        if not raw_files(folder):
            continue
        rec = Recorder(resume=False)          # from the raw packets alone, not on top of what was saved
        for arrived, data in read_raw(folder):
            rec.handle(data, arrived, record=False)
        for s in rec.sessions.values():
            s.save()
            print(f"{folder.name}: {s.summary()['track']} {s.summary()['type']}, {s.packets} packets")


RECORDER = Recorder()

if __name__ == "__main__":
    if sys.argv[1:] == ["rebuild"]:
        rebuild()
        sys.exit()
    RECORDER.start()
    time.sleep(1)
    print(RECORDER.error or f"Listening for F1 25 on port {PORT}. Set the game's UDP IP Address to: "
          + (", ".join(local_addresses()) or "this PC's address"))
    try:
        while True:
            time.sleep(5)
            st = RECORDER.status()
            if st["live_session"]:
                s = RECORDER.sessions[int(st["live_session"])]
                print(f"  {s.info.get('track', '?')} {s.info.get('session_type', '')}: {s.packets} packets")
    except KeyboardInterrupt:
        RECORDER.save_due(force=True)
        print("Saved.")
