import fastf1
import matplotlib.pyplot as plt                 # for the chart

from config import CACHE_DIR

CACHE_DIR.mkdir(exist_ok=True)                  # make a folder for downloads
fastf1.Cache.enable_cache(CACHE_DIR)            # save every download, never fetch twice

session = fastf1.get_session(2026, "Singapore", "R")   # year, race, "R" = the race
session.load(laps=True, telemetry=False, weather=True, messages=False)
print(session.event["EventName"], "-", session.total_laps, "laps")

laps = session.laps
print(len(laps), "rows")
print(laps.columns.tolist())
print(laps["Driver"].unique())                  # all driver codes in this race

ver = laps.pick_drivers("VER")                  # keep only Verstappen's laps
print(ver[["Driver", "LapNumber", "LapTime", "Compound", "TyreLife", "Stint"]].head(10))

stops = ver[ver["PitInTime"].notna()]           # the laps where he drove into the pits
print(stops[["LapNumber", "Compound", "TyreLife"]])

# Chart of his lap times, coloured by tyre, with a dashed line at each pit stop
colours = {"SOFT": "red", "MEDIUM": "gold", "HARD": "grey", "INTERMEDIATE": "green", "WET": "blue"}
clean = ver.pick_quicklaps()                    # remove very slow laps (safety car, pit laps)
x = clean["LapNumber"]                          # x axis: lap number
y = clean["LapTime"].dt.total_seconds()         # y axis: lap time in seconds
dot_colours = clean["Compound"].map(colours).fillna("black")   # colour of each dot = its tyre
plt.scatter(x, y, c=dot_colours)                # one dot per lap
for lap in stops["LapNumber"]:
    plt.axvline(lap, linestyle="--", color="black")   # dashed line at each pit stop
plt.xlabel("Lap")
plt.ylabel("Lap time (seconds)")
plt.title("Verstappen - Singapore 2026")
plt.show()                                      # open the chart window