"""The cloud data layer: a safe copy of the race data and the models in Amazon S3, which any of your computers can
fetch in seconds (the desktop app, desktop.py, does it by itself when it starts).

Everything in data/ goes up except the game's folders (data/f1_25: your recordings, data/driver: the driver report
made from them), which never leave this PC, and the replay file, which the app builds again by itself. A table
straight in data/ gets a folder of its own in the bucket (data/all_laps.parquet -> data/all_laps/all_laps.parquet),
because Athena reads a folder as one table (aws_athena.py). Each file is sent with its SHA-256 and listed in the
bucket's manifest.json, so every download is checked against what was sent, and with the bucket's versioning on, S3
keeps every older copy too. A download never replaces a file changed on this PC since the last sync, and an upload
never replaces a newer copy in the cloud with an older one (aws_state.json remembers the last sync).

Once (the steps are in AWS_SETUP.md):
    python aws_sync.py setup BUCKET [REGION]   the bucket (kept in aws_settings.json, on this PC only)
    python aws_sync.py keys                    type your access key (kept in your user folder's .aws, not here)
    python aws_sync.py policy                  the permissions for that key, ready to paste into AWS
    python aws_sync.py policy check            the permissions for the Lambda check (aws_check.py)
Then:
    python aws_sync.py upload                  send what changed (after a new race; --all: every file again)
    python aws_sync.py download                fetch what changed
    python aws_sync.py status                  what's here and in the cloud, and how many versions S3 keeps
    python aws_sync.py check                   fetch everything into an empty folder, as a new PC would, and time it
Add --force to upload or download to take that side's copy of a file changed on both sides.
"""
import configparser
import getpass
import hashlib
import json
import re
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from config import ROOT

SETTINGS_FILE = ROOT / "aws_settings.json"
STATE = "aws_state.json"                         # in the project folder: each file's SHA-256 at the last sync
MANIFEST = "manifest.json"                       # in the bucket: every file, its size and SHA-256
PROFILE = "f1-strategy"                          # the key's name in your user folder's .aws/credentials
REGION = "eu-central-1"                          # Frankfurt, the closest to Greece
FOLDERS = ["data", "models"]
STAY_HERE = {"f1_25", "driver"}                  # the game's data never leaves this PC
REBUILT = {"replay_laps.parquet"}                # the app builds it again from all_laps.parquet
THREADS = 8


def settings():
    """This PC's bucket, region and key name, or None when it isn't set up for AWS (the app then works as before)."""
    if not SETTINGS_FILE.exists():
        return None
    return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))


def client(cfg, service="s3"):
    import boto3                                 # only here: the app and the tests run without AWS
    from botocore.config import Config
    known = configparser.ConfigParser()
    known.read(Path.home() / ".aws" / "credentials", encoding="utf-8")
    profile = cfg.get("profile") if cfg.get("profile") in known.sections() else None   # else AWS's usual places
    session = boto3.Session(profile_name=profile, region_name=cfg.get("region", REGION))
    return session.client(service, config=Config(connect_timeout=5, read_timeout=60, retries={"max_attempts": 2}))


def local_files(root=ROOT):
    """Every file that goes to the cloud, as paths from the project folder (data/..., models/...)."""
    out = []
    for folder in FOLDERS:
        for p in sorted((root / folder).rglob("*")):
            first = p.relative_to(root / folder).parts[0]
            if p.is_file() and first not in STAY_HERE and p.name not in REBUILT and p.suffix != ".part":
                out.append(p.relative_to(root).as_posix())
    return out


def key_for(path):
    """Where a file goes in the bucket. A table straight in data/ gets a folder of its own, as Athena reads every file
    in a folder as one table: data/all_laps.parquet -> data/all_laps/all_laps.parquet."""
    folder, *rest = path.split("/")
    if folder == "data" and len(rest) == 1:
        return f"data/{Path(rest[0]).stem}/{rest[0]}"
    return path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(s3, bucket):
    try:
        return json.loads(s3.get_object(Bucket=bucket, Key=MANIFEST)["Body"].read())
    except Exception as e:
        if "NoSuchKey" in type(e).__name__ or "NoSuchKey" in str(e):     # nothing sent yet
            return {"files": []}
        raise


def read_state(root):
    f = root / STATE
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def save_state(root, state):
    (root / STATE).write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")


def compare(root, manifest, state):
    """Each file's side: "same"; "up" (new or changed here); "down" (new or changed in the cloud); or "both" (changed
    here and in the cloud since the last sync, or no record of it on this PC: left alone, as either copy could be
    the one you want). Returns {path: (side, this PC's SHA-256, the cloud's entry)}."""
    cloud = {e["path"]: e for e in manifest.get("files", [])}
    here = {p: sha256(root / p) for p in local_files(root)}
    out = {}
    for path in sorted(set(cloud) | set(here)):
        mine, theirs, last = here.get(path), cloud.get(path, {}).get("sha256"), state.get(path)
        if mine == theirs:
            side = "same"
        elif theirs is None:
            side = "up"
        elif mine is None:
            side = "down"
        elif last == mine:                       # unchanged here since the last sync: the cloud has a newer copy
            side = "down"
        elif last == theirs:                     # unchanged in the cloud since the last sync: this copy is newer
            side = "up"
        else:
            side = "both"
        out[path] = (side, mine, cloud.get(path))
    return out


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def upload(s3, bucket, root=ROOT, force=False, everything=False, log=print):
    """Send what is new or changed here, then the new manifest (everything=True sends every file again: the cloud's
    check runs on each, and S3 keeps the copies before as older versions). Returns the files and bytes sent, and the
    seconds it took."""
    manifest, state = read_manifest(s3, bucket), read_state(root)
    sides = compare(root, manifest, state)
    send = [p for p, (side, _, _) in sides.items()
            if side == "up" or (force and side == "both") or (everything and side == "same")]
    entries = {e["path"]: e for e in manifest.get("files", [])}
    start = time.perf_counter()

    def one(path):
        sha = sides[path][1]
        s3.upload_file(str(root / path), bucket, key_for(path), ExtraArgs={"Metadata": {"sha256": sha}})
        return {"path": path, "key": key_for(path), "size": (root / path).stat().st_size, "sha256": sha}

    with ThreadPoolExecutor(THREADS) as pool:
        for entry in pool.map(one, send):
            entries[entry["path"]] = entry
    if send:
        new = {"updated": now(), "files": sorted(entries.values(), key=lambda e: e["path"])}
        s3.put_object(Bucket=bucket, Key=MANIFEST, Body=json.dumps(new, indent=1).encode("utf-8"),
                      ContentType="application/json")
    for path, (side, mine, _) in sides.items():
        if side == "same" or path in send:
            state[path] = mine
    save_state(root, state)
    seconds, size = time.perf_counter() - start, sum((root / p).stat().st_size for p in send)
    log(f"Sent {len(send)} files ({size / 1e6:.1f} MB) in {seconds:.1f} s; "
        f"{Counter(s for s, _, _ in sides.values())['same']} were already the same in the cloud")
    left(sides, "down", "Newer in the cloud, so not sent (python aws_sync.py download)", log)
    left(sides, "both" if not force else None, "Changed here and in the cloud, so left alone (add --force to send yours)", log)
    return len(send), size, seconds


def download(s3, bucket, root=ROOT, force=False, log=print):
    """Fetch what is new or changed in the cloud. Each file is checked against its SHA-256 before it replaces anything
    here, and a file changed here since the last sync is kept. Returns the files and bytes fetched, and the seconds."""
    manifest, state = read_manifest(s3, bucket), read_state(root)
    if not manifest.get("files"):
        raise SystemExit("The cloud has nothing yet: run python aws_sync.py upload on the PC with the data")
    sides = compare(root, manifest, state)
    fetch = [p for p, (side, _, _) in sides.items() if side == "down" or (force and side == "both")]
    start = time.perf_counter()

    def one(path):
        entry, target = sides[path][2], root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        s3.download_file(bucket, entry["key"], str(part))
        if sha256(part) != entry["sha256"]:
            part.unlink()
            raise IOError(f"{path}: the cloud's copy doesn't match what was sent, so it wasn't used")
        part.replace(target)
        return path

    with ThreadPoolExecutor(THREADS) as pool:
        done = set(pool.map(one, fetch))
    for path, (side, mine, entry) in sides.items():
        if side == "same" or path in done:
            state[path] = entry["sha256"]
    save_state(root, state)
    seconds, size = time.perf_counter() - start, sum(sides[p][2]["size"] for p in done)
    log(f"Fetched {len(done)} files ({size / 1e6:.1f} MB) in {seconds:.1f} s; "
        f"{Counter(s for s, _, _ in sides.values())['same']} were already the same here")
    left(sides, "up", "Changed here, so not replaced (python aws_sync.py upload)", log)
    left(sides, "both" if not force else None, "Changed here and in the cloud, so left alone (add --force to take the cloud's)", log)
    return len(done), size, seconds


def left(sides, side, why, log):
    paths = [p for p, (s, _, _) in sides.items() if s == side]
    if paths:
        log(f"{why}: {', '.join(paths[:6])}{' and more' if len(paths) > 6 else ''}")


def versions(s3, bucket):
    """How many copies S3 keeps of each file (more than one once versioning is on and a file was sent again)."""
    counts = Counter()
    for page in s3.get_paginator("list_object_versions").paginate(Bucket=bucket):
        for v in page.get("Versions", []):
            counts[v["Key"]] += 1
    return counts


def status(s3, bucket, root=ROOT, log=print):
    manifest = read_manifest(s3, bucket)
    sides = compare(root, manifest, read_state(root))
    counts = Counter(s for s, _, _ in sides.values())
    size = sum(e["size"] for e in manifest.get("files", []))
    log(f"Bucket {bucket}: {len(manifest.get('files', []))} files ({size / 1e6:.1f} MB), last sent "
        f"{manifest.get('updated', 'never')}")
    log(f"Here: {counts['same']} the same, {counts['up']} to send, {counts['down']} to fetch, {counts['both']} changed on both sides")
    kept = versions(s3, bucket)
    many = sorted(((n, k) for k, n in kept.items() if n > 1), reverse=True)
    log(f"Versions: {sum(kept.values())} copies of {len(kept)} files" +
        (f"; most: {', '.join(f'{k} ({n})' for n, k in many[:5])}" if many else " (one each so far)"))


def check(s3, bucket, root=ROOT, into=None, log=print):
    """As a new PC would: fetch everything into an empty folder, time it, and compare every file with this PC's.
    Saves the numbers in results/cloud.md."""
    with tempfile.TemporaryDirectory(dir=into) as tmp:
        files, size, seconds = download(s3, bucket, Path(tmp), log=lambda *_: None)
        fetched = local_files(Path(tmp))
        here = [p for p in fetched if (root / p).exists()]
        same = sum(sha256(Path(tmp) / p) == sha256(root / p) for p in here)
    lines = ["# The cloud data layer: a new PC's download", "",
             f"Fetched from the S3 bucket into an empty folder on {now()[:10]}, as a new PC would, then every file "
             "compared (SHA-256) with this PC's copy. The game's recordings are never in the cloud.", "",
             "| Files | Size | Time | Same as this PC's copy |", "|---:|---:|---:|---:|",
             f"| {files} | {size / 1e6:.1f} MB | {seconds:.1f} s | {same} of {len(here)} |", ""]
    (root / "results").mkdir(exist_ok=True)
    (root / "results" / "cloud.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"A new PC: {files} files ({size / 1e6:.1f} MB) in {seconds:.1f} s; {same} of {len(here)} the same as here "
        "(saved in results/cloud.md)")
    return files, size, seconds, same, len(here)


def policy(bucket, region=REGION):
    """The permissions for the app's key: this one bucket (without deleting: with versioning on, nothing is ever lost)
    and Athena on its own database."""
    arn, glue = f"arn:aws:s3:::{bucket}", f"arn:aws:glue:{region}:*:"
    return {"Version": "2012-10-17", "Statement": [
        {"Sid": "TheBucket", "Effect": "Allow", "Resource": arn,
         "Action": ["s3:ListBucket", "s3:ListBucketVersions", "s3:GetBucketLocation"]},
        {"Sid": "ItsFiles", "Effect": "Allow", "Resource": f"{arn}/*",
         "Action": ["s3:GetObject", "s3:GetObjectVersion", "s3:PutObject"]},
        {"Sid": "Athena", "Effect": "Allow", "Resource": f"arn:aws:athena:{region}:*:workgroup/primary",
         "Action": ["athena:StartQueryExecution", "athena:GetQueryExecution", "athena:GetQueryResults",
                    "athena:GetWorkGroup"]},
        {"Sid": "AthenaTables", "Effect": "Allow",
         "Resource": [glue + "catalog", glue + "database/f1_strategy", glue + "table/f1_strategy/*"],
         "Action": ["glue:GetDatabase", "glue:GetDatabases", "glue:CreateDatabase", "glue:GetTable", "glue:GetTables",
                    "glue:CreateTable", "glue:UpdateTable", "glue:DeleteTable", "glue:GetPartitions"]},
    ]}


def check_policy(bucket):
    """The permissions for the Lambda check (aws_check.py): read the tables, write the reports."""
    arn = f"arn:aws:s3:::{bucket}"
    return {"Version": "2012-10-17", "Statement": [
        {"Sid": "ReadTheTables", "Effect": "Allow", "Action": "s3:GetObject", "Resource": f"{arn}/data/*"},
        {"Sid": "WriteTheReports", "Effect": "Allow", "Action": "s3:PutObject", "Resource": f"{arn}/checks/*"},
    ]}


def save_keys():
    """Ask for the access key (the secret part isn't shown as you type) and keep it in your user folder's
    .aws/credentials, as AWS's own tools do. It is never kept in the project folder."""
    path = Path.home() / ".aws" / "credentials"
    print("Paste your access key (AWS: IAM > Users > f1-strategy-app > Security credentials).")
    key_id = input("Access key ID: ").strip()
    secret = getpass.getpass("Secret access key (paste it with a right-click; it won't show): ").strip()
    if not re.fullmatch(r"AKIA[A-Z0-9]{16}", key_id):
        raise SystemExit("That isn't an access key ID (20 letters and digits, starting AKIA). Nothing saved.")
    if not re.fullmatch(r"[A-Za-z0-9/+]{40}", secret):
        raise SystemExit(f"That isn't a secret access key: it should be 40 characters, and {len(secret)} came in. "
                         "Nothing saved. Paste it with a right-click (Ctrl+V doesn't paste into a hidden prompt "
                         "in every window).")
    creds = configparser.ConfigParser()
    creds.read(path, encoding="utf-8")
    creds[PROFILE] = {"aws_access_key_id": key_id, "aws_secret_access_key": secret}
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        creds.write(f)
    print(f"Saved in {path} as '{PROFILE}' (on this PC only)")
    cfg = settings()
    if cfg:
        try:
            client(cfg).head_bucket(Bucket=cfg["bucket"])
            print(f"The key works: it can open the bucket {cfg['bucket']}")
        except Exception as e:
            print(f"Saved, but AWS refused it ({e}). Check that both parts come from the same key, then run this again.")


def main():
    words = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    command = words[0] if words else "help"
    if command == "setup" and len(words) >= 2:
        cfg = {"bucket": words[1], "region": words[2] if len(words) > 2 else REGION, "profile": PROFILE}
        SETTINGS_FILE.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        print(f"Saved in {SETTINGS_FILE.name}: bucket {cfg['bucket']} in {cfg['region']}")
    elif command == "keys":
        save_keys()
    elif command in ("policy", "upload", "download", "status", "check"):
        cfg = settings()
        if not cfg:
            raise SystemExit("First: python aws_sync.py setup YOUR-BUCKET-NAME  (see AWS_SETUP.md)")
        if command == "policy":
            rules = check_policy(cfg["bucket"]) if words[1:2] == ["check"] else policy(cfg["bucket"], cfg["region"])
            print(json.dumps(rules, indent=2))
            return
        s3 = client(cfg)
        if command == "upload":
            upload(s3, cfg["bucket"], force=force, everything="--all" in sys.argv)
        elif command == "download":
            download(s3, cfg["bucket"], force=force)
        elif command == "status":
            status(s3, cfg["bucket"])
        else:
            check(s3, cfg["bucket"])
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
