"""The cloud data layer (aws_sync.py), its check in the cloud (aws_check.py) and the Athena tables (aws_athena.py),
against a made-up bucket in memory: no AWS account, internet or keys are needed, and the real data isn't touched."""
import hashlib
import io
import json

import pandas as pd
import pytest

import aws_athena
import aws_check
import aws_sync
from conftest import make_race


class NoSuchKey(Exception):
    pass


class FakeS3:
    """A bucket in memory, with the calls aws_sync.py and aws_check.py make."""

    def __init__(self):
        self.objects = {}                        # key -> (bytes, metadata)

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        with open(filename, "rb") as f:
            self.objects[key] = (f.read(), (ExtraArgs or {}).get("Metadata", {}))

    def download_file(self, bucket, key, filename):
        with open(filename, "wb") as f:
            f.write(self.objects[key][0])

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = (Body, {})

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise NoSuchKey(Key)
        body, meta = self.objects[Key]
        return {"Body": io.BytesIO(body), "Metadata": meta}


def project(root, laps=None):
    """A made-up project folder: the race data, a FastF1 race, a model, and the game's own data."""
    for folder in ("data/raw", "data/f1_25/2026-10-28", "data/driver", "models"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (laps if laps is not None else make_race()).to_parquet(root / "data" / "all_laps.parquet", index=False)
    make_race().head(20).to_parquet(root / "data" / "raw" / "2022_01.parquet", index=False)
    (root / "data" / "team_colours.json").write_text('{"2025_01": {}}', encoding="utf-8")
    (root / "data" / "replay_laps.parquet").write_bytes(b"built by the app")
    (root / "data" / "f1_25" / "2026-10-28" / "session.json").write_text("{}", encoding="utf-8")
    (root / "data" / "driver" / "corners.parquet").write_bytes(b"made from the recordings")
    (root / "models" / "pit_model.json").write_text('{"features": []}', encoding="utf-8")
    return root


def edit(path, text):
    path.write_text(text, encoding="utf-8")


def test_what_goes_to_the_cloud(tmp_path):
    root = project(tmp_path)
    assert aws_sync.local_files(root) == ["data/all_laps.parquet", "data/raw/2022_01.parquet",
                                          "data/team_colours.json", "models/pit_model.json"]
    # the game's recordings and the driver report made from them never leave the PC; the app rebuilds the replay
    assert aws_sync.key_for("data/all_laps.parquet") == "data/all_laps/all_laps.parquet"     # a folder per table
    assert aws_sync.key_for("data/raw/2022_01.parquet") == "data/raw/2022_01.parquet"
    assert aws_sync.key_for("models/pit_model.json") == "models/pit_model.json"


def test_a_new_pc_gets_the_same_files(tmp_path):
    here, new = project(tmp_path / "here"), tmp_path / "new"
    new.mkdir()
    s3 = FakeS3()
    sent, _, _ = aws_sync.upload(s3, "b", here, log=lambda *_: None)
    got, _, _ = aws_sync.download(s3, "b", new, log=lambda *_: None)
    assert sent == got == 4
    for path in aws_sync.local_files(here):
        assert (new / path).read_bytes() == (here / path).read_bytes()
    assert not (new / "data" / "f1_25").exists() and not (new / "data" / "driver").exists()
    assert aws_sync.upload(s3, "b", here, log=lambda *_: None)[0] == 0          # nothing changed: nothing sent
    assert aws_sync.download(s3, "b", new, log=lambda *_: None)[0] == 0
    assert aws_sync.upload(s3, "b", here, everything=True, log=lambda *_: None)[0] == 4    # --all: every file again


def test_a_change_is_never_lost(tmp_path):
    """The laptop fetched the data; then the PC sends a new race (colours) while the laptop changes the model.
    The laptop's download takes the new colours but keeps its own model, and its upload then sends the model."""
    pc, laptop = project(tmp_path / "pc"), tmp_path / "laptop"
    laptop.mkdir()
    s3, quiet = FakeS3(), (lambda *_: None)
    aws_sync.upload(s3, "b", pc, log=quiet)
    aws_sync.download(s3, "b", laptop, log=quiet)
    edit(pc / "data" / "team_colours.json", '{"2025_01": {}, "2025_02": {}}')
    aws_sync.upload(s3, "b", pc, log=quiet)
    edit(laptop / "models" / "pit_model.json", '{"features": ["TyreLife"]}')
    assert aws_sync.download(s3, "b", laptop, log=quiet)[0] == 1
    assert "2025_02" in (laptop / "data" / "team_colours.json").read_text(encoding="utf-8")
    assert "TyreLife" in (laptop / "models" / "pit_model.json").read_text(encoding="utf-8")      # kept
    assert aws_sync.upload(s3, "b", laptop, log=quiet)[0] == 1
    assert aws_sync.download(s3, "b", pc, log=quiet)[0] == 1                       # the PC gets the model
    # the same file changed on both sides is left alone, unless one side is chosen
    edit(pc / "models" / "pit_model.json", '{"features": ["A"]}')
    aws_sync.upload(s3, "b", pc, log=quiet)
    edit(laptop / "models" / "pit_model.json", '{"features": ["B"]}')
    assert aws_sync.download(s3, "b", laptop, log=quiet)[0] == 0
    assert aws_sync.upload(s3, "b", laptop, log=quiet)[0] == 0
    assert aws_sync.download(s3, "b", laptop, force=True, log=quiet)[0] == 1
    assert '"A"' in (laptop / "models" / "pit_model.json").read_text(encoding="utf-8")


def test_a_broken_download_is_refused(tmp_path):
    here, new = project(tmp_path / "here"), tmp_path / "new"
    new.mkdir()
    s3 = FakeS3()
    aws_sync.upload(s3, "b", here, log=lambda *_: None)
    key = "data/all_laps/all_laps.parquet"
    s3.objects[key] = (b"not the file that was sent", s3.objects[key][1])
    with pytest.raises(IOError, match="doesn't match"):
        aws_sync.download(s3, "b", new, log=lambda *_: None)
    assert not (new / "data" / "all_laps.parquet").exists()
    assert not list(new.rglob("*.part"))


def test_without_aws_nothing_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(aws_sync, "SETTINGS_FILE", tmp_path / "aws_settings.json")
    assert aws_sync.settings() is None                       # the desktop app then uses the data on the PC


def test_the_key_can_use_only_the_bucket():
    rules = aws_sync.policy("my-bucket")["Statement"]
    s3 = [r for r in rules if any(a.startswith("s3:") for a in r["Action"])]
    assert {r["Resource"] for r in s3} == {"arn:aws:s3:::my-bucket", "arn:aws:s3:::my-bucket/*"}
    assert not any("Delete" in a for r in s3 for a in r["Action"])    # with versioning on, nothing is ever lost


def test_the_cloud_check():
    laps = make_race()
    good = aws_check.check(laps, "all_laps")
    assert good["status"] == "ok" and good["seasons"] == [2025] and good["races"] == 1 and good["rows"] == len(laps)
    bad = aws_check.check(pd.concat([laps, laps.head(1)]).drop(columns="Compound"), "all_laps")
    assert bad["status"] == "problems found"
    assert "1 laps listed twice" in bad["problems"] and any("Compound" in p for p in bad["problems"])
    odd = laps.assign(Compound=laps["Compound"].where(laps["LapNumber"] != 3, "UNKNOWN"))
    assert aws_check.check(odd, "all_laps")["status"] == "ok"            # an unnamed tyre is a note, not a problem
    assert aws_check.kind("data/raw/2022_01.parquet") == "race"
    assert aws_check.kind("data/all_laps/all_laps.parquet") == "all_laps"


def test_the_check_runs_on_each_upload(tmp_path):
    root = project(tmp_path)
    s3 = FakeS3()
    aws_sync.upload(s3, "b", root, log=lambda *_: None)
    event = {"Records": [{"s3": {"bucket": {"name": "b"}, "object": {"key": "data/all_laps/all_laps.parquet"}}}]}
    assert aws_check.lambda_handler(event, None, s3=s3) == {"checked": 1, "problems": 0}
    report = json.loads(s3.objects["checks/data/all_laps/all_laps.json"][0])
    assert report["status"] == "ok" and report["arrived_whole"] is True
    body, meta = s3.objects["data/all_laps/all_laps.parquet"]
    tampered = make_race().head(5).to_parquet(index=False)
    s3.objects["data/all_laps/all_laps.parquet"] = (tampered, meta)               # not what was sent
    assert aws_check.lambda_handler(event, None, s3=s3)["problems"] == 1
    report = json.loads(s3.objects["checks/data/all_laps/all_laps.json"][0])
    assert report["arrived_whole"] is False and "SHA-256" in report["problems"][0]
    assert meta["sha256"] == hashlib.sha256(body).hexdigest()


def test_the_athena_tables(tmp_path):
    table = pd.DataFrame({"RaceId": ["2025_01"], "LapNumber": [1.0], "Year": [2025], "Rainfall": [False],
                          "Seg1": [["2049", "2051"]], "Compound": pd.Categorical(["SOFT"])})
    table.to_parquet(tmp_path / "all_laps.parquet", index=False)
    sql = aws_athena.create_table("laps", "all_laps", "my-bucket", data=tmp_path)
    for column in ("`raceid` string", "`lapnumber` double", "`year` bigint", "`rainfall` boolean",
                   "`seg1` array<string>", "`compound` string"):
        assert column in sql
    assert "LOCATION 's3://my-bucket/data/all_laps/'" in sql                   # the folder aws_sync.py sends it to
    assert all(" FROM laps" in q or "FROM laps " in q for _, q in aws_athena.QUESTIONS.values())
