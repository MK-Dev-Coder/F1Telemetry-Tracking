import json
import time

import pandas as pd
import requests

from config import DATA_DIR, RAW_DIR

OPENF1 = "https://api.openf1.org/v1"
ENDPOINTS = ["drivers", "laps", "stints", "pit", "position", "weather", "race_control"]


# Recipe 1: ask OpenF1 for one thing (if it says "too many requests", wait and try again)
def get(endpoint, **params):
    for attempt in range(5):
        answer = requests.get(f"{OPENF1}/{endpoint}", params=params, timeout=60)
        if answer.status_code == 429:
            print("   too many requests - waiting 20 seconds...")
            time.sleep(20)
            continue
        answer.raise_for_status()
        return answer.json()
    raise RuntimeError(f"OpenF1 kept saying 429 for {endpoint}")


# Recipe 2: find the 2026 Singapore race and download everything about it
def main():
    races = get("sessions", year=2026, session_name="Race")
    race = next(r for r in races if r["country_name"] == "Singapore")
    key = race["session_key"]
    print("Singapore 2026 race, session_key", key)

    sample = {"session": race}
    for name in ENDPOINTS:
        sample[name] = get(name, session_key=key)
        print("  ", name, len(sample[name]), "rows")
        time.sleep(1)                                   # a short break between requests

    out = DATA_DIR / "openf1_sample.json"
    out.write_text(json.dumps(sample, indent=1), encoding="utf-8")
    print("saved", out)

    # The FastF1 version of the same race, as a CSV you can open in Excel
    fastf1_csv = DATA_DIR / "singapore_fastf1.csv"
    pd.read_parquet(RAW_DIR / "2026_17.parquet").to_csv(fastf1_csv, index=False)
    print("saved", fastf1_csv)


if __name__ == "__main__":
    main()
