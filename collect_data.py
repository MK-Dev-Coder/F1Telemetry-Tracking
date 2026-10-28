import time

import fastf1
import pandas as pd
import requests
from fastf1.exceptions import RateLimitExceededError

from config import CACHE_DIR, RAW_DIR, SEASONS

# Never wait forever for a server. If it says nothing for 30 seconds, give up on that file.
_original_request = requests.Session.request


def _request_with_timeout(self, method, url, *args, **kwargs):
    kwargs.setdefault("timeout", 30)
    return _original_request(self, method, url, *args, **kwargs)


requests.Session.request = _request_with_timeout

# The columns we keep for every lap
LAP_COLUMNS = ["Driver", "Team", "LapNumber", "LapTime", "Time", "Stint", "Compound",
               "TyreLife", "FreshTyre", "PitInTime", "PitOutTime", "TrackStatus", "Position"]
WEATHER_COLUMNS = ["AirTemp", "TrackTemp", "Humidity", "Rainfall"]
TIME_COLUMNS = ["LapTime", "Time", "PitInTime", "PitOutTime"]      # we save these as seconds

F1_TEST_URL = "https://livetiming.formula1.com/static/2024/Index.json"   # NEW: a small file to test F1's server


# NEW - Recipe 0: wait until F1's server answers us (it blocks us for a while after many downloads)
def wait_for_f1():
    while True:
        try:
            status = requests.get(F1_TEST_URL, headers={"User-Agent": "BestHTTP"}, timeout=20).status_code
        except requests.RequestException:
            status = "no answer"
        if status == 200:
            return
        print("   F1's server says", status, "- waiting 15 minutes...")
        time.sleep(15 * 60)


# Recipe 1: load one race (if the download limit is reached, wait 15 minutes and try again)
def load_race(year, rnd):
    while True:
        try:
            session = fastf1.get_session(year, rnd, "R")
            session.load(laps=True, telemetry=False, weather=True, messages=False)
            return session
        except RateLimitExceededError:
            print("   download limit reached - waiting 15 minutes...")
            time.sleep(15 * 60)


# Recipe 2: turn one race into one clean table (laps + the weather on each lap)
def race_table(session, year, rnd, event_name):
    laps = session.laps.reset_index(drop=True)
    weather = laps.get_weather_data().reset_index(drop=True)      # weather on each lap
    table = pd.concat([laps[LAP_COLUMNS], weather[WEATHER_COLUMNS]], axis=1)
    for col in TIME_COLUMNS:
        table[col] = table[col].dt.total_seconds()                # 0 days 00:01:37.28 -> 97.28
    table["Year"] = year
    table["Round"] = rnd
    table["EventName"] = event_name
    try:
        table["TotalLaps"] = session.total_laps                   # laps planned for this race
    except Exception:
        table["TotalLaps"] = table["LapNumber"].max()             # unknown? use the laps raced
    return table


# Recipe 3: go through every race of every season, and save the ones you don't have yet
def main():
    CACHE_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    fastf1.set_log_level("WARNING")                               # hide FastF1's INFO lines
    today = pd.Timestamp.now()

    for year in SEASONS:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event in schedule.iterrows():                        # one race of the calendar at a time
            rnd = int(event["RoundNumber"])
            name = event["EventName"]
            out_file = RAW_DIR / f"{year}_{rnd:02d}.parquet"        # e.g. 2024_05.parquet
            if out_file.exists() or event["EventDate"] > today:
                continue                                            # already saved, or not raced yet
            print(year, "round", rnd, "-", name)
            wait_for_f1()                                           # NEW: wait here while F1 blocks us
            try:
                session = load_race(year, rnd)
                table = race_table(session, year, rnd, name)
            except Exception as err:
                print("   skipped:", err)                           # e.g. no data for this race
                time.sleep(60)                                      # give the servers a minute
                continue
            table.to_parquet(out_file, index=False)
            print("   saved", len(table), "laps")
            time.sleep(30)                                          # CHANGED: a 30-second break between races

    print("Done:", len(list(RAW_DIR.glob("*.parquet"))), "races saved")


if __name__ == "__main__":
    main()