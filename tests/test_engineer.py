"""engineer.py: the track briefing - how long each tyre lasts on a day this hot, in the game's shorter races, and
from only the races before one (a replay: no spoilers)."""
import pytest

import engineer


def race(temp, winner, plan_stints, sc=False, wet=False, rain=False):
    return {"laps": 56, "track": temp, "air": 28.0, "rain": rain, "wet": wet, "sc": sc, "vsc": False, "winner": winner,
            "winner_grid": 1, "winner_stints": plan_stints}


# Two races at a made-up Austin: 2024 (38 °C) and 2025 (42 °C). Medium stints of 20-30 laps, a few softs.
STINTS = ([["2024_19", "M", n] for n in (20, 22, 24, 26, 28)] + [["2025_19", "M", n] for n in (22, 24, 26, 28, 30)]
          + [["2024_19", "S", n] for n in (12, 14, 16)] + [["2025_19", "S", n] for n in (14, 16)])
WEAR = [[r, c, w] for r in ("2024_19", "2025_19") for c, w in (("M", 0.06), ("S", 0.08), ("H", 0.05)) for _ in range(3)]
FIN = [["2024_19", "M-H", [25], 2, 1], ["2024_19", "M-H", [27], -1, 4], ["2024_19", "S-M-H", [12, 30], 3, 6],
       ["2025_19", "M-H", [26], 0, 2], ["2025_19", "S-H", [15], 1, 1]]
TABLE = {"tracks": {"Austin": {"races": ["2024_19", "2025_19"], "pit_loss": 22.0, "stints": STINTS, "wear": WEAR, "finishers": FIN,
                               "race_info": {"2024_19": race(38.0, "VER", [{"compound": "MEDIUM", "from": 1, "to": 25}, {"compound": "HARD", "from": 26, "to": 56}], sc=True),
                                             "2025_19": race(42.0, "NOR", [{"compound": "SOFT", "from": 1, "to": 15}, {"compound": "HARD", "from": 16, "to": 56}])}}},
         "temperature_effect": {"SOFT": 0.001, "MEDIUM": 0.002, "HARD": 0.0014}, "fuel": 0.06,
         "pit_share": {"green": 1.0, "SC": 0.43, "VSC": 0.61}, "all_wear": {"SOFT": 0.09, "MEDIUM": 0.07, "HARD": 0.06},
         "wet_stints": [["2024_03", "I", n] for n in (10, 15, 20)] + [["2025_10", "W", n] for n in (4, 6)],
         "wet_races": ["2024_03", "2025_10"]}


def test_a_usual_day_is_what_happened_here():
    b = engineer.brief(TABLE, "Austin")
    m = b["tyres"]["MEDIUM"]
    assert b["usual_temp"] == 40.0 and m["life"] == 25.0 and m["life_today"] == 25.0 and m["change"] == 0
    assert m["range"] == [22.5, 27.5] and m["stints"] == 10


def test_a_hotter_day_wears_the_tyres_faster():
    m = engineer.brief(TABLE, "Austin", track_temp=50)["tyres"]["MEDIUM"]
    assert m["wear_today"] == pytest.approx(0.06 + 0.002 * 10)               # 10 °C hotter
    assert m["life_today"] == pytest.approx(25 * 0.06 / 0.08, abs=0.1)        # wears a third faster: lasts a quarter less
    assert engineer.brief(TABLE, "Austin", track_temp=30)["tyres"]["MEDIUM"]["life_today"] > 25


def test_the_change_is_at_most_40_percent_either_way():
    assert engineer.brief(TABLE, "Austin", track_temp=90)["tyres"]["MEDIUM"]["change"] == pytest.approx(-0.4)
    assert engineer.brief(TABLE, "Austin", track_temp=-10)["tyres"]["MEDIUM"]["change"] == pytest.approx(0.4)


def test_a_shorter_game_race_wears_the_tyres_as_much_faster():
    b = engineer.brief(TABLE, "Austin", game_laps=14)                         # a quarter of 56 laps
    assert (b["game_laps"], b["distance"]) == (14, 25)
    assert b["tyres"]["MEDIUM"]["game"]["life"] == pytest.approx(25 / 4, abs=0.05)
    assert engineer.brief(TABLE, "Austin", distance=25)["game_laps"] == 14
    assert engineer.brief(TABLE, "Austin")["tyres"]["MEDIUM"]["game"] is None


def test_too_few_stints_say_so():
    b = engineer.brief(TABLE, "Austin")
    assert b["tyres"]["HARD"]["life_today"] is None and b["tyres"]["SOFT"]["life"] == 14.0


def test_only_the_races_before_one():
    """A replay of the 2025 race: only 2024 is known (no spoilers of 2025's winner or strategies)."""
    b = engineer.brief(TABLE, "Austin", before="2025_19")
    assert b["races"] == ["2024_19"] and [w["driver"] for w in b["winners"]] == ["VER"]
    assert b["usual_temp"] == 38.0 and b["tyres"]["MEDIUM"]["life"] == 24.0     # the 2024 stints only
    assert {p["plan"] for p in b["plans"]} == {"M-H", "S-M-H"} and b["safety_car"]["either"] == 1
    assert engineer.brief(TABLE, "Austin", before="2024_19")["empty"]


def test_the_strategies_here():
    plans = {p["plan"]: p for p in engineer.brief(TABLE, "Austin")["plans"]}
    assert plans["M-H"]["drivers"] == 3 and plans["M-H"]["stop_laps"] == [26] and plans["M-H"]["wins"] == 1
    assert plans["S-M-H"]["stop_laps"] == [12, 30] and plans["S-H"]["top5"] == 1


def test_a_stop_costs_less_under_the_safety_car():
    p = engineer.brief(TABLE, "Austin")["pit_loss"]
    assert (p["green"], p["sc"], p["vsc"]) == (22.0, round(22 * 0.43, 1), round(22 * 0.61, 1))


def test_each_race_here_with_its_safety_car():
    h = engineer.brief(TABLE, "Austin")["history"]
    assert [(x["year"], x["sc"], x["track"]) for x in h] == [(2024, True, 38.0), (2025, False, 42.0)]


def test_rain_adds_the_wet_tyres():
    assert engineer.brief(TABLE, "Austin")["wet"] is None
    w = engineer.brief(TABLE, "Austin", rain=True)["wet"]
    assert w["INTERMEDIATE"]["life"] == 15.0 and w["WET"]["stints"] == 2 and w["races"] == 2


def test_an_unknown_track():
    assert engineer.brief(TABLE, "Nowhere") is None
