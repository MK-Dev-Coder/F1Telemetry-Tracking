"""A pretend F1 25: sends correctly formatted telemetry packets for a short race, to test the app without
a console. Its sessions are marked TEST in the app (it sends game version 0.0).

    python game_simulator.py                 # a 5-lap race, 10x faster than real time
    python game_simulator.py --laps 3 --speed 20
"""
import argparse
import math
import random
import socket
import struct
import time

from game_telemetry import (DAMAGE, FINAL, HEADER, LAP, LAP_HISTORY, MOTION_CAR, PARTICIPANT, PORT, SIZES, STATUS,
                            TELEMETRY)

MOTION_CAR, STATUS, TELEMETRY = MOTION_CAR[2025], STATUS[2025], TELEMETRY[2025]   # it sends the 2025 format

FPS = 20                                         # frames per second of game time
CARS = 20
LENGTH = 5000.0                                  # metres per lap


def track_point(s):
    """A made-up closed circuit: an oval with wiggles. s = metres along the lap."""
    a = 2 * math.pi * s / LENGTH
    r = 1 + 0.25 * math.sin(2 * a) + 0.12 * math.sin(5 * a + 1)
    return 900 * r * math.cos(a), 600 * r * math.sin(a)


def corner_speed(s):
    """Faster on straights, slower in corners (m/s)."""
    a = 2 * math.pi * s / LENGTH
    bend = abs(math.sin(5 * a + 1)) * 0.6 + abs(math.sin(2 * a)) * 0.4
    return 85 - 45 * bend ** 2


class Car:
    def __init__(self, i):
        self.i, self.pace = i, 1 - 0.004 * i - random.random() * 0.006
        self.distance = -12.0 * i                # grid slots behind the line
        self.lap_times, self.sectors = [], []
        self.lap_start = self.sector_start = 0.0
        self.current = [None, None]
        self.compound, self.age, self.stops = (16 if i % 3 else 17), 0, 0
        self.pit_lap = 3 if i % 2 == 0 else 2   # the player (car 0) stops at the end of lap 3
        self.in_lane, self.box_time, self.fuel = False, 0.0, 40.0
        self.finished, self.finish_t = False, None

    @property
    def lap(self):
        return max(1, int(self.distance // LENGTH) + 1)

    @property
    def lap_distance(self):
        return self.distance % LENGTH if self.distance >= 0 else LENGTH + self.distance


def header(pid, uid, t, frame):
    return HEADER.pack(2025, 25, 0, 0, 1, pid, uid, t, frame, frame, 0, 255)


def pad(rows, size):
    return b"".join(rows) + b"\0" * (22 - len(rows)) * size


def run(laps, speed, host):
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(pid, payload):
        assert len(payload) == SIZES[2025][pid], f"packet {pid}: {len(payload)} bytes"
        out.sendto(payload, (host, PORT))

    uid = random.getrandbits(63)
    cars = [Car(i) for i in range(CARS)]
    names = ["PLAYER", "ALVAREZ", "BECK", "CARTER", "DUBOIS", "EKSTROM", "FERRO", "GALLO", "HAYES", "INGRAM", "JANSEN",
             "KOVAC", "LAURENT", "MORENO", "NAKAMURA", "OKAFOR", "PETROV", "QUINN", "ROSSI", "SILVA"][:CARS]
    best, t, frame, leader_done = math.inf, 0.0, 0, None
    print(f"Sending a {laps}-lap test race to {host}:{PORT} ({speed}x real time)...")

    def event(code, details=b""):
        send(3, header(3, uid, t, frame) + code.encode() + details.ljust(12, b"\0"))

    def history(c):
        rows = [LAP_HISTORY.pack(round(lt * 1000), round(s[0] * 1000), 0, round(s[1] * 1000), 0, round(s[2] * 1000), 0, 0x0F)
                for lt, s in zip(c.lap_times, c.sectors)]
        best_n = c.lap_times.index(min(c.lap_times)) + 1 if c.lap_times else 0
        head = struct.pack("<7B", c.i, len(rows), 1, best_n, best_n, best_n, best_n)
        send(11, header(11, uid, t, frame) + head + b"".join(rows) + b"\0" * (100 - len(rows)) * LAP_HISTORY.size + b"\0" * 24)

    event("SSTA")
    while True:
        t += 1 / FPS
        frame += 1
        for c in cars:                                            # move every car
            if c.finished:
                continue
            v = corner_speed(c.lap_distance) * c.pace * (1 - 0.002 * c.age)
            if c.stops == 0 and c.lap == c.pit_lap and c.lap_distance > LENGTH - 300:
                c.in_lane = True                                  # into the pit lane at the end of the lap
            if c.in_lane:
                v = min(v, 22.0)
                if c.lap > c.pit_lap:
                    c.box_time += 1 / FPS
                if c.box_time > 6:                                # stop done: new tyres
                    c.in_lane, c.box_time, c.stops, c.age = False, 0.0, c.stops + 1, 0
                    c.compound = 17 if c.compound == 16 else 18
            before_lap, before_sector = c.lap, int(c.lap_distance // (LENGTH / 3))
            c.distance += v / FPS
            c.fuel -= 0.0016
            sector = int(c.lap_distance // (LENGTH / 3))
            if c.distance > 0 and sector != before_sector and sector in (1, 2):
                c.current[sector - 1] = round(t - c.sector_start, 3)
                c.sector_start = t
            if c.lap > before_lap and c.distance > LENGTH:          # crossed the line
                lap_time = t - c.lap_start
                s1 = c.current[0] or lap_time / 3
                s2 = c.current[1] or lap_time / 3
                c.lap_times.append(lap_time)
                c.sectors.append((s1, s2, lap_time - s1 - s2))
                c.lap_start, c.sector_start, c.current, c.age = t, t, [None, None], c.age + 1
                if lap_time < best:
                    best = lap_time
                    event("FTLP", struct.pack("<Bf", c.i, lap_time))
                if c.lap > laps:                                  # took the chequered flag
                    c.finished, c.finish_t = True, t
                    leader_done = leader_done or t
        order = sorted(cars, key=lambda c: (0, c.finish_t) if c.finished else (1, -c.distance))   # finishers first
        leader = order[0]
        rows = []
        for c in cars:                                            # lap data
            pos = order.index(c) + 1
            ahead = order[pos - 2] if pos > 1 else c
            gap_lead, gap_front = (leader.distance - c.distance) / 60, (ahead.distance - c.distance) / 60
            if c.finished and leader.finished:                    # at the flag: the gap is the finishing time
                gap_lead, gap_front = c.finish_t - leader.finish_t, c.finish_t - (ahead.finish_t or c.finish_t)
            last = round(c.lap_times[-1] * 1000) if c.lap_times else 0
            s1, s2 = round((c.current[0] or 0) * 1000), round((c.current[1] or 0) * 1000)
            ms = lambda g: (round(g * 1000) % 60000, round(g * 1000) // 60000)
            rows.append(LAP.pack(last, round((t - c.lap_start) * 1000), s1 % 60000, s1 // 60000, s2 % 60000, s2 // 60000,
                                 *ms(gap_front), *ms(gap_lead), c.lap_distance, max(c.distance, 0), 0.0, pos,
                                 min(c.lap, laps), 1 if c.in_lane else 0, c.stops, int(c.lap_distance // (LENGTH / 3)),
                                 0, 0, 0, 0, 0, 0, c.i + 1, 4, 3 if c.finished else 2, 0, 0, 0, 0, 300.0, 1))
        send(2, header(2, uid, t, frame) + pad(rows, LAP.size) + bytes([255, 255]))
        motion = [MOTION_CAR.pack(*track_point(c.lap_distance)[:1], 0.0, track_point(c.lap_distance)[1], *[0] * 15)
                  for c in cars]
        send(0, header(0, uid, t, frame) + pad(motion, MOTION_CAR.size))
        tele, status, damage = [], [], []
        for c in cars:                                            # telemetry, status, damage
            v = corner_speed(c.lap_distance) * c.pace
            nxt = corner_speed((c.lap_distance + 40) % LENGTH) * c.pace
            throttle, brake = (1.0, 0.0) if nxt >= v else (0.0, min(1.0, (v - nxt) / 15))
            gear = max(1, min(8, int(v * 3.6 / 40)))
            tele.append(TELEMETRY.pack(int(v * 3.6), throttle, math.sin(t), brake, 0, gear, 9000 + int(v * 30), 0, 50, 0,
                                       *[500 + int(brake * 300)] * 4, *[90 + c.age] * 4, *[100 + c.age] * 4, 105,
                                       *[23.0] * 4, *[0] * 4))
            status.append(STATUS.pack(0, 0, 1, 56, 1 if c.in_lane else 0, c.fuel, 110.0, c.fuel / 1.5, 13000, 4000, 8, 0, 0,
                                      18 if c.compound == 16 else 19, c.compound, c.age, 0, 0.0, 0.0,
                                      3e6 * (0.5 + 0.4 * math.sin(t / 10)), 1, 0.0, 0.0, 0.0, 0))
            wear = min(99.0, c.age * (3.0 if c.compound == 16 else 2.0) + c.lap_distance / LENGTH)
            damage.append(DAMAGE[2025].pack(*[wear] * 4, *[0] * 30))
        send(6, header(6, uid, t, frame) + pad(tele, TELEMETRY.size) + bytes([255, 255, 0]))
        send(7, header(7, uid, t, frame) + pad(status, STATUS.size))
        if frame % 10 == 1:
            send(10, header(10, uid, t, frame) + pad(damage, DAMAGE[2025].size))
            session = bytearray(SIZES[2025][1])
            session[:29] = header(1, uid, t, frame)
            struct.pack_into("<BbbBHBbBHHBBBBBB", session, 29, 0, 32, 24, laps, int(LENGTH), 15, 3, 0, 0, 3600, 80,
                             0, 0, 255, 0, 0)
            struct.pack_into("<ff", session, 745, LENGTH / 3, 2 * LENGTH / 3)
            send(1, bytes(session))
        if frame % 20 == 1:
            people = [PARTICIPANT[2025].pack(int(i > 0), i, 0, i % 10, 0, i + 1, 0, n.encode().ljust(32, b"\0"),
                                             1, 1, 0, 1, 0, *[0] * 12) for i, n in enumerate(names)]
            send(4, header(4, uid, t, frame) + bytes([CARS]) + pad(people, PARTICIPANT[2025].size))
        history(cars[frame % CARS])                               # lap history, one car per frame
        if all(c.finished for c in cars) or (leader_done and t - leader_done > 120):
            break
        time.sleep(1 / FPS / speed)
    event("CHQF")
    for c in cars:
        history(c)
    final = [FINAL[2025].pack(order.index(c) + 1, len(c.lap_times), c.i + 1, max(0, 26 - 2 * (order.index(c) + 1)),
                              c.stops, 3, 0, round(min(c.lap_times) * 1000) if c.lap_times else 0, sum(c.lap_times),
                              0, 0, 1, *[0] * 24) for c in cars]
    send(8, header(8, uid, t, frame) + bytes([CARS]) + pad(final, FINAL[2025].size))
    event("SEND")
    print(f"Done: {frame} frames, {laps} laps, fastest lap {best // 60:.0f}:{best % 60:06.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laps", type=int, default=5)
    ap.add_argument("--speed", type=float, default=10)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    run(a.laps, a.speed, a.host)
