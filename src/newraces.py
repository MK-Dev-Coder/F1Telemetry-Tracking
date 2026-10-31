"""New races from the cloud. Each morning AWS fetches any new race from OpenF1 into the bucket (aws_fetch.py, a Lambda
function on a daily schedule). This finds the ones waiting there and builds them here with the usual steps, as the
README's "add a new race" does by hand: collect_openf1.py for the race (it reads the fetched answers, so nothing is
downloaded twice), then build_dataset.py, features.py, build_timing.py, race_preview.py and engineer.py. Then it sends
the results back (aws_sync.py), so every PC gets the race. The desktop app runs it after it starts. By hand:

    python src/newraces.py           fetch the latest data, then build any new race
    python src/newraces.py --index   write data/openf1/sessions.json (the races built here, for the Lambda to skip)

If a step fails, the steps run again next time (data/openf1/building.json lists the races they're for). The study
doesn't change: its test races stay 2026 rounds 1-17 (TEST_LAST_ROUND in features.py).
"""
import gzip
import json
import subprocess
import sys
from pathlib import Path

import aws_fetch
import aws_sync
from config import CACHE_DIR, ROOT

HERE = Path(__file__).resolve().parent  # src/, where the steps are
STEPS = ["build_dataset.py", "features.py", "build_timing.py", "race_preview.py", "engineer.py"]
INDEX = "data/openf1/sessions.json"      # OpenF1 session key -> race id, for the Lambda to skip the races built
NOTE = "data/openf1/building.json"       # races whose steps haven't all run yet (never sent to the cloud)


def sessions(root=ROOT):
    """OpenF1 session key -> race id, for every OpenF1 race built here: its saved OpenF1 answers and its table (a race
    whose table wasn't made isn't built, so it is tried again)."""
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
    """The race's year, round and name in FastF1's calendar, numbered as collect_openf1.py numbers them."""
    import fastf1
    import collect_openf1
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)
    year, key = raw["session"]["year"], raw["session"]["session_key"]
    for rnd, name, race in collect_openf1.race_list(year):
        if race["session_key"] == key:
            return year, rnd, name
    raise LookupError(f"OpenF1 session {key} isn't in FastF1's {year} calendar yet")


def run(script, *args, root=ROOT):
    """One step, as it would run by hand; a failure stops the rest."""
    done = subprocess.run([sys.executable, str(HERE / script), *map(str, args)], cwd=root, capture_output=True,
                          text=True)
    if done.returncode:
        last = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no message"]
        raise RuntimeError(f"{script} failed: {last[0]}")
    return done.stdout


def add(s3, bucket, root=ROOT, log=print, step=run, number=round_of, send=True):
    """Build every race waiting in the bucket that isn't here yet, and finish any build a failure left. Returns the
    races added (race id and name)."""
    note = root / NOTE
    pending = json.loads(note.read_text(encoding="utf-8")) if note.exists() else []
    here = set(sessions(root))
    waiting = {k[len(aws_fetch.INCOMING):].split(".")[0]: k for k in aws_fetch.keys_in(s3, bucket, aws_fetch.INCOMING)}
    for key in sorted((k for k in waiting if k not in here), key=int):
        try:
            body = s3.get_object(Bucket=bucket, Key=waiting[key])["Body"].read()
            year, rnd, name = number(json.loads(gzip.decompress(body)))
        except Exception as e:               # one race that can't be read or found in the calendar waits; others go on
            log(f"Race {key} from the cloud waits for next time: {e}")
            continue
        race_id, folder = f"{year}_{rnd:02d}", root / "data" / "openf1"
        (folder / "json").mkdir(parents=True, exist_ok=True)
        (folder / "json" / f"{race_id}.json.gz").write_bytes(body)
        log(f"New race from the cloud: {year} round {rnd}, {name}")
        try:
            step("collect_openf1.py", year, rnd, root=root)
            if not (folder / f"{race_id}.parquet").exists():      # collect_openf1.py says "skipped" and carries on
                raise RuntimeError("collect_openf1.py made no table for it")
        except Exception as e:               # not built, so not in sessions(): it is tried again next time
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
    note.unlink()
    if send:
        aws_sync.upload(s3, bucket, root, log=log)
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
