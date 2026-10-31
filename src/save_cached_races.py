import fastf1

from collect_data import race_table
from config import CACHE_DIR, RAW_DIR, SEASONS


# Save every race that is fully in fastf1_cache but not in data/raw yet.
# It never uses the internet, so F1's block doesn't matter here.
def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    fastf1.Cache.offline_mode(True)
    fastf1.set_log_level("CRITICAL")                         # hide the errors for races not in the cache

    for year in SEASONS:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, event in schedule.iterrows():
            rnd = int(event["RoundNumber"])
            out_file = RAW_DIR / f"{year}_{rnd:02d}.parquet"
            if out_file.exists():
                continue
            try:
                session = fastf1.get_session(year, rnd, "R")
                session.load(laps=True, telemetry=False, weather=True, messages=False)
                table = race_table(session, year, rnd, event["EventName"])
            except Exception:
                continue                                    # not in the cache
            if len(table) == 0:
                continue
            table.to_parquet(out_file, index=False)
            print(year, "round", rnd, "-", event["EventName"], "- saved", len(table), "laps")

    print("Done:", len(list(RAW_DIR.glob("*.parquet"))), "FastF1 races saved")


if __name__ == "__main__":
    main()
