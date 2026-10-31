"""New races by themselves: the Lambda (aws_fetch.py) fetches each finished race from OpenF1 into the bucket, and the
PC (newraces.py) builds it with the usual steps. A made-up bucket and made-up OpenF1 answers: no internet or AWS."""
import gzip
import json
from datetime import datetime, timezone

import pytest

import aws_fetch
import newraces
from conftest import FakeS3

NOW = datetime(2026, 11, 2, 6, 0, tzinfo=timezone.utc)


def QUIET(*_):
    pass


def race(key, end, cancelled=False):
    return {"session_key": key, "year": 2026, "country_name": f"Country {key}", "date_start": end,
            "date_end": end, "is_cancelled": cancelled, "circuit_short_name": "Somewhere"}


SESSIONS = [race(11396, "2026-10-25T22:00:00+00:00"),            # Austin: built already
            race(11420, "2026-11-01T22:00:00+00:00"),            # Mexico City: new, finished
            race(11450, "2026-11-08T19:00:00+00:00"),            # Sao Paulo: not raced yet
            race(11300, "2026-06-01T15:00:00+00:00", cancelled=True)]


def openf1(laps=3):
    """Made-up OpenF1 answers: the sessions, and a few rows from every other endpoint."""
    def fetch(endpoint, **params):
        if endpoint == "sessions":
            return SESSIONS
        return [{"session_key": params["session_key"], "lap_number": n} for n in range(1, laps + 1)]
    return fetch


def bucket_with_index(built):
    s3 = FakeS3()
    s3.put_object(Bucket="b", Key=aws_fetch.INDEX, Body=json.dumps({str(k): v for k, v in built.items()}).encode())
    return s3


def test_the_cloud_fetches_only_the_new_finished_race():
    s3 = bucket_with_index({11396: "2026_18"})
    done = aws_fetch.lambda_handler({"bucket": "b"}, None, s3=s3, fetch=openf1(), now=NOW)
    assert [f["session_key"] for f in done["fetched"]] == [11420]           # not Austin, Sao Paulo or the cancelled one
    raw = json.loads(gzip.decompress(s3.objects["data/openf1/incoming/11420.json.gz"][0]))
    assert list(raw) == ["session"] + aws_fetch.ENDPOINTS                    # as collect_openf1.py saves a race
    assert raw["session"]["session_key"] == 11420 and len(raw["laps"]) == 3
    again = aws_fetch.lambda_handler({"bucket": "b"}, None, s3=s3, fetch=openf1(), now=NOW)
    assert again["fetched"] == [] and again["already"] == 2                  # waiting in the bucket: not twice


def test_a_race_without_laps_yet_waits_for_tomorrow():
    s3 = bucket_with_index({11396: "2026_18"})
    assert aws_fetch.lambda_handler({"bucket": "b"}, None, s3=s3, fetch=openf1(laps=0), now=NOW)["fetched"] == []
    assert not any(k.startswith(aws_fetch.INCOMING) for k in s3.objects)


def test_a_race_that_just_ended_waits_for_openf1_to_fill_it_in():
    s3 = bucket_with_index({11396: "2026_18"})
    soon = datetime(2026, 11, 2, 1, 0, tzinfo=timezone.utc)                  # 3 hours after Mexico City
    assert aws_fetch.lambda_handler({"bucket": "b"}, None, s3=s3, fetch=openf1(), now=soon)["fetched"] == []


def test_one_race_can_be_fetched_again():
    s3 = bucket_with_index({11396: "2026_18"})
    done = aws_fetch.lambda_handler({"bucket": "b", "session_key": 11396}, None, s3=s3, fetch=openf1(), now=NOW)
    assert [f["session_key"] for f in done["fetched"]] == [11396]


def project(root):
    """Austin built already: its OpenF1 answers and its table."""
    folder = root / "data" / "openf1"
    (folder / "json").mkdir(parents=True)
    austin = {"session": race(11396, "2026-10-25T22:00:00+00:00"), "laps": [{"lap_number": 1}]}
    (folder / "json" / "2026_18.json.gz").write_bytes(gzip.compress(json.dumps(austin).encode()))
    (folder / "2026_18.parquet").write_bytes(b"Austin's laps")
    return root


def steps(root, ran, collects=True, fails=None):
    """Made-up steps: each one is noted, collect_openf1.py makes the race's table (unless it skips the race), and
    the step named in fails stops with an error."""
    def step(script, *args, **kwargs):
        ran.append((script, *args))
        if script == "collect_openf1.py" and collects:
            year, rnd = args
            (root / "data" / "openf1" / f"{year}_{rnd:02d}.parquet").write_bytes(b"the race's laps")
        if script == fails:
            raise RuntimeError(f"{script} failed")
    return step


def mexico(raw):
    return 2026, 19, "Mexico City Grand Prix"


def test_the_pc_builds_the_new_race(tmp_path):
    root = project(tmp_path)
    s3 = bucket_with_index({11396: "2026_18"})
    aws_fetch.lambda_handler({"bucket": "b"}, None, s3=s3, fetch=openf1(), now=NOW)
    ran = []
    added = newraces.add(s3, "b", root, log=QUIET, step=steps(root, ran), number=mexico, send=False)
    assert added == ["2026_19 Mexico City Grand Prix"]
    assert (root / "data" / "openf1" / "json" / "2026_19.json.gz").read_bytes() == s3.objects["data/openf1/incoming/11420.json.gz"][0]
    assert ran == [("collect_openf1.py", 2026, 19)] + [(s,) for s in newraces.STEPS]   # the README's order
    index = json.loads((root / newraces.INDEX).read_text(encoding="utf-8"))
    assert index == {"11396": "2026_18", "11420": "2026_19"}                # the Lambda skips both from now on
    assert not (root / newraces.NOTE).exists()
    assert newraces.add(s3, "b", root, log=QUIET, step=steps(root, ran), number=mexico, send=False) == []


def test_a_failed_step_runs_again_next_time(tmp_path):
    root = project(tmp_path)
    s3 = bucket_with_index({11396: "2026_18"})
    aws_fetch.lambda_handler({"bucket": "b"}, None, s3=s3, fetch=openf1(), now=NOW)
    with pytest.raises(RuntimeError):
        newraces.add(s3, "b", root, log=QUIET, step=steps(root, [], fails="features.py"), number=mexico, send=False)
    assert json.loads((root / newraces.NOTE).read_text(encoding="utf-8")) == ["2026_19 Mexico City Grand Prix"]
    ran = []
    assert newraces.add(s3, "b", root, log=QUIET, step=steps(root, ran), number=mexico,
                        send=False) == ["2026_19 Mexico City Grand Prix"]
    assert ran == [(s,) for s in newraces.STEPS]                             # the race isn't collected twice
    assert not (root / newraces.NOTE).exists()


@pytest.mark.parametrize("collect", ["skips the race", "fails"])
def test_a_race_collect_openf1_didnt_make_is_tried_again(tmp_path, collect):
    root = project(tmp_path)
    s3 = bucket_with_index({11396: "2026_18"})
    aws_fetch.lambda_handler({"bucket": "b"}, None, s3=s3, fetch=openf1(), now=NOW)
    ran = []
    broken = steps(root, ran, collects=False, fails="collect_openf1.py" if collect == "fails" else None)
    assert newraces.add(s3, "b", root, log=QUIET, step=broken, number=mexico, send=False) == []
    assert ran == [("collect_openf1.py", 2026, 19)]                          # the other steps wait for a race
    assert "11420" not in newraces.sessions(root) and not (root / newraces.NOTE).exists()
    ran.clear()
    assert newraces.add(s3, "b", root, log=QUIET, step=steps(root, ran), number=mexico,
                        send=False) == ["2026_19 Mexico City Grand Prix"]
    assert ran[0] == ("collect_openf1.py", 2026, 19)


def test_a_race_that_cant_be_placed_doesnt_hold_up_the_others(tmp_path):
    root = project(tmp_path)
    s3 = bucket_with_index({11396: "2026_18"})
    for key in (11420, 11440):                                              # Mexico City, then Las Vegas
        raw = {"session": race(key, "2026-11-01T22:00:00+00:00"), "laps": [{"lap_number": 1}]}
        s3.put_object(Bucket="b", Key=f"{aws_fetch.INCOMING}{key}.json.gz", Body=gzip.compress(json.dumps(raw).encode()))
    calendar = {11440: (2026, 20, "Las Vegas Grand Prix")}                  # Mexico City isn't in FastF1's yet

    def number(raw):
        key = raw["session"]["session_key"]
        if key not in calendar:
            raise LookupError(f"OpenF1 session {key} isn't in FastF1's 2026 calendar yet")
        return calendar[key]

    assert newraces.add(s3, "b", root, log=QUIET, step=steps(root, []), number=number,
                        send=False) == ["2026_20 Las Vegas Grand Prix"]
    calendar[11420] = mexico(None)                                          # FastF1's calendar has it now
    assert newraces.add(s3, "b", root, log=QUIET, step=steps(root, []), number=number,
                        send=False) == ["2026_19 Mexico City Grand Prix"]
    assert newraces.sessions(root) == {"11396": "2026_18", "11420": "2026_19", "11440": "2026_20"}


def test_the_steps_are_where_newraces_runs_them(tmp_path):
    for script in ["collect_openf1.py"] + newraces.STEPS:
        assert (newraces.HERE / script).exists(), script
    assert newraces.run("config.py", root=tmp_path) == ""                    # a real run, in the project folder
    with pytest.raises(RuntimeError, match="no_such_step.py failed"):
        newraces.run("no_such_step.py", root=tmp_path)


def test_the_note_never_goes_to_the_cloud(tmp_path):
    import aws_sync
    (tmp_path / "data" / "openf1").mkdir(parents=True)
    (tmp_path / "data" / "openf1" / "building.json").write_text("[]", encoding="utf-8")
    (tmp_path / "data" / "openf1" / "sessions.json").write_text("{}", encoding="utf-8")
    assert aws_sync.local_files(tmp_path) == ["data/openf1/sessions.json"]
