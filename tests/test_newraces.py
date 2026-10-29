"""New races by themselves: the Lambda (aws_fetch.py) fetches each finished race from OpenF1 into the bucket, and the
PC (newraces.py) builds it with the usual steps. A made-up bucket and made-up OpenF1 answers: no internet or AWS."""
import gzip
import io
import json
from datetime import datetime, timezone

import pytest

import aws_fetch
import newraces

NOW = datetime(2026, 11, 2, 6, 0, tzinfo=timezone.utc)


class NoSuchKey(Exception):
    pass


class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise NoSuchKey(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def list_objects_v2(self, Bucket, Prefix, **_):
        return {"Contents": [{"Key": k} for k in sorted(self.objects) if k.startswith(Prefix)], "IsTruncated": False}


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
    raw = json.loads(gzip.decompress(s3.objects["data/openf1/incoming/11420.json.gz"]))
    assert list(raw) == ["session"] + aws_fetch.ENDPOINTS                    # as collect_openf1.py saves a race
    assert raw["session"]["session_key"] == 11420 and len(raw["laps"]) == 3
    again = aws_fetch.lambda_handler({"bucket": "b"}, None, s3=s3, fetch=openf1(), now=NOW)
    assert again["fetched"] == [] and again["already"] == 2                  # waiting in the bucket: not twice


def test_a_race_without_laps_yet_waits_for_tomorrow():
    s3 = bucket_with_index({11396: "2026_18"})
    assert aws_fetch.lambda_handler({"bucket": "b"}, None, s3=s3, fetch=openf1(laps=0), now=NOW)["fetched"] == []
    assert not any(k.startswith(aws_fetch.INCOMING) for k in s3.objects)


def test_one_race_can_be_fetched_again():
    s3 = bucket_with_index({11396: "2026_18"})
    done = aws_fetch.lambda_handler({"bucket": "b", "session_key": 11396}, None, s3=s3, fetch=openf1(), now=NOW)
    assert [f["session_key"] for f in done["fetched"]] == [11396]


def project(root):
    json_dir = root / "data" / "openf1" / "json"
    json_dir.mkdir(parents=True)
    austin = {"session": race(11396, "2026-10-25T22:00:00+00:00"), "laps": [{"lap_number": 1}]}
    (json_dir / "2026_18.json.gz").write_bytes(gzip.compress(json.dumps(austin).encode()))
    return root


def test_the_pc_builds_the_new_race(tmp_path):
    root = project(tmp_path)
    s3 = bucket_with_index({11396: "2026_18"})
    aws_fetch.lambda_handler({"bucket": "b"}, None, s3=s3, fetch=openf1(), now=NOW)
    ran = []
    added = newraces.add(s3, "b", root, log=lambda *_: None, step=lambda *a, **k: ran.append(a),
                         number=lambda raw: (2026, 19, "Mexico City Grand Prix"), send=False)
    assert added == ["2026_19 Mexico City Grand Prix"]
    assert (root / "data" / "openf1" / "json" / "2026_19.json.gz").read_bytes() == s3.objects["data/openf1/incoming/11420.json.gz"]
    assert ran == [("collect_openf1.py", 2026, 19)] + [(s,) for s in newraces.STEPS]   # the README's order
    index = json.loads((root / newraces.INDEX).read_text(encoding="utf-8"))
    assert index == {"11396": "2026_18", "11420": "2026_19"}                # the Lambda skips both from now on
    assert not (root / newraces.NOTE).exists()
    assert newraces.add(s3, "b", root, log=lambda *_: None, step=lambda *a, **k: ran.append(a),
                        number=lambda raw: (2026, 19, "Mexico City Grand Prix"), send=False) == []


def test_a_failed_step_runs_again_next_time(tmp_path):
    root = project(tmp_path)
    s3 = bucket_with_index({11396: "2026_18"})
    aws_fetch.lambda_handler({"bucket": "b"}, None, s3=s3, fetch=openf1(), now=NOW)
    number = lambda raw: (2026, 19, "Mexico City Grand Prix")

    def broken(script, *args, **kwargs):
        if script == "features.py":
            raise RuntimeError("features.py failed")

    with pytest.raises(RuntimeError):
        newraces.add(s3, "b", root, log=lambda *_: None, step=broken, number=number, send=False)
    assert json.loads((root / newraces.NOTE).read_text(encoding="utf-8")) == ["2026_19 Mexico City Grand Prix"]
    ran = []
    assert newraces.add(s3, "b", root, log=lambda *_: None, step=lambda *a, **k: ran.append(a),
                        number=number, send=False) == ["2026_19 Mexico City Grand Prix"]
    assert ran == [(s,) for s in newraces.STEPS]                             # the race isn't collected twice
    assert not (root / newraces.NOTE).exists()


def test_the_note_never_goes_to_the_cloud(tmp_path):
    import aws_sync
    (tmp_path / "data" / "openf1").mkdir(parents=True)
    (tmp_path / "data" / "openf1" / "building.json").write_text("[]", encoding="utf-8")
    (tmp_path / "data" / "openf1" / "sessions.json").write_text("{}", encoding="utf-8")
    assert aws_sync.local_files(tmp_path) == ["data/openf1/sessions.json"]
