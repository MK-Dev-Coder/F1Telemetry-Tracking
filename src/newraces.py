"""Builds new races fetched by the cloud. Every morning AWS fetches any new race from OpenF1 into the bucket
(aws_fetch.py, a Lambda on a daily schedule). This script finds the waiting races and builds them here with the
usual steps, like the README's "add a new race" does by hand: collect_openf1.py for the race (it reads the fetched
data, so nothing is downloaded twice), then build_dataset.py, features.py, build_timing.py, race_outcome.py (final
results), race_preview.py and engineer.py. Then it uploads the results (aws_sync.py) so every PC gets the race. The
desktop app runs it at start.

python src/newraces.py           fetch the latest data, then build any new race
python src/newraces.py --index   write data/openf1/sessions.json (races built here, for the Lambda to skip)

If a step or the upload fails, the steps run again next time (data/openf1/building.json lists those races). OpenF1
isn't asked again about a race the cloud fetched, and while OpenF1 is closed (during a live F1 session) a build is
retried next time instead of waiting. The study doesn't change: its test races stay 2026 rounds 1-17
(TEST_LAST_ROUND in features.py)."""
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import aws_fetch
import aws_sync
from config import CACHE_DIR, ROOT

HERE = Path(__file__).resolve().parent  # src/ folder with the steps
STEPS = ["build_dataset.py", "features.py", "build_timing.py", "race_outcome.py", "race_preview.py", "engineer.py"]
INDEX = "data/openf1/sessions.json"      # OpenF1 session key -> race id, so the Lambda skips built races
NOTE = "data/openf1/building.json"       # races not built and uploaded yet (this file never goes to the cloud)


def sessions(root=ROOT):
    """OpenF1 session key -> race id for every OpenF1 race built here (it has saved OpenF1 data and a table; a race
    without a table isn't built yet, so it's tried again)."""
    out, folder = {}, root / "data" / "openf1"
    for f in sorted((folder / "json").glob("*.json.gz")):
        race_id = f.name.split(".")[0]
        if (folder / f"{race_id}.parquet").exists():
            out[str(json.loads(gzip.decompress(f.read_bytes()))["session"]["session_key"])] = race_id
    return out


def write_index(root=ROOT):
    index = sessions(root)
    (root / INDEX).write_text(json.dumps(index, indent=1, sort_keys=True), encoding="utf-8")
    return index


def round_of(raw):
    """Year, round and name of the race from FastF1's calendar, numbered the same way as collect_openf1.py (OpenF1
    isn't asked again)."""
    import fastf1
    import collect_openf1
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    year, key = raw["session"]["year"], raw["session"]["session_key"]
    found = collect_openf1.round_for(raw["session"], collect_openf1.rounds_by_date(year))
    if found is None:
        raise LookupError(f"OpenF1 session {key} isn't in FastF1's {year} calendar yet")
    return year, *found


def run(script, *args, root=ROOT):
    """Run one step like it would run by hand (but never wait for hours while OpenF1 is closed). A failure stops
    the rest."""
    done = subprocess.run([sys.executable, str(HERE / script), *map(str, args)], cwd=root, capture_output=True,
                          text=True, env={**os.environ, "F1_OPENF1_QUICK": "1"})
    if done.returncode:
        last = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no message"]
        raise RuntimeError(f"{script} failed: {last[0]}")
    return done.stdout


def add(s3, bucket, root=ROOT, log=print, step=run, number=round_of, send=True):
    """Build every race waiting in the bucket that isn't here yet, and finish any build that failed before. Returns
    the races added (race id and name)."""
    note = root / NOTE
    pending = json.loads(note.read_text(encoding="utf-8")) if note.exists() else []
    here = set(sessions(root))
    names = {k: k[len(aws_fetch.INCOMING):].split(".")[0] for k in aws_fetch.keys_in(s3, bucket, aws_fetch.INCOMING)}
    waiting = {name: k for k, name in names.items() if name.isdigit()}     # only session keys, not hand-made folders
    for key in sorted((k for k in waiting if k not in here), key=int):
        try:
            body = s3.get_object(Bucket=bucket, Key=waiting[key])["Body"].read()
            year, rnd, name = number(json.loads(gzip.decompress(body)))
        except Exception as e:               # a race that can't be read or isn't in the calendar waits
            log(f"Race {key} from the cloud waits for next time: {e}")
            continue
        race_id, folder = f"{year}_{rnd:02d}", root / "data" / "openf1"
        (folder / "json").mkdir(parents=True, exist_ok=True)
        (folder / "json" / f"{race_id}.json.gz").write_bytes(body)
        log(f"New race from the cloud: {year} round {rnd}, {name}")
        try:
            step("collect_openf1.py", year, rnd, root=root)
            if not (folder / f"{race_id}.parquet").exists():      # collect_openf1.py prints "skipped" and carries on
                raise RuntimeError("collect_openf1.py made no table for it")
        except Exception as e:               # not built, so not in sessions() and it's tried again next time
            log(f"   not built ({e}): it is tried again next time")
            continue
        pending = sorted(set(pending) | {f"{race_id} {name}"})
        note.write_text(json.dumps(pending), encoding="utf-8")                # in case a later step fails
    if not pending:
        return []
    for script in STEPS:
        log(f"   {script}")
        step(script, root=root)
    write_index(root)
    if send:
        aws_sync.upload(s3, bucket, root, log=log)
    note.unlink()                            # built and uploaded (until then the steps and upload run again next time)
    return pending


def main():
    if "--index" in sys.argv:
        print(f"{len(write_index())} races in {INDEX}")
        return
    cfg = aws_sync.settings()
    if not cfg:
        raise SystemExit("This PC isn't set up for AWS: see docs/AWS_SETUP.md")
    s3 = aws_sync.client(cfg)
    aws_sync.download(s3, cfg["bucket"])                                     # another PC may have built it already
    added = add(s3, cfg["bucket"])
    print("Added:", ", ".join(added) if added else "no new race")


if __name__ == "__main__":
    main()
