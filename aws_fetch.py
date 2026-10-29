"""The cloud's daily job: an AWS Lambda function that an EventBridge schedule runs every morning. It asks OpenF1 for
this season's races that have finished and fetches each one the bucket doesn't have yet, exactly as
collect_openf1.py saves it (the session and OpenF1's drivers, laps, stints, pit, position, weather and race control
answers, gzipped JSON), into data/openf1/incoming/<session key>.json.gz. The PC then builds it with the usual steps
(newraces.py; the desktop app runs it when it starts), so a new Grand Prix reaches the app without typing anything.

It knows which races are built already from data/openf1/sessions.json (OpenF1 session key -> race id, kept by
newraces.py), and skips the ones still waiting in incoming/. Only the standard library and boto3: no layer.

In AWS (AWS_SETUP.md): a Python 3.12 function, this file as lambda_function.py (handler
lambda_function.lambda_handler), 256 MB, a 5-minute timeout, the environment variable BUCKET, and permission to read
sessions.json, list and write incoming/. To fetch one race again, run it with {"session_key": 11396}.
"""
import gzip
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

OPENF1 = "https://api.openf1.org/v1"
ENDPOINTS = ["drivers", "laps", "stints", "pit", "position", "weather", "race_control"]   # as collect_openf1.py
PAUSE = 2.1                              # seconds between requests (free users: 30 requests a minute)
INCOMING = "data/openf1/incoming/"
INDEX = "data/openf1/sessions.json"
SETTLE = timedelta(hours=1)              # OpenF1's history opens 30 minutes after a session ends


def get(endpoint, **params):
    """One OpenF1 answer. A live session closes OpenF1 to free users, so that day is skipped (tomorrow works)."""
    url = f"{OPENF1}/{endpoint}?{urllib.parse.urlencode(params)}"
    for _ in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as answer:
                data = json.loads(answer.read())
            time.sleep(PAUSE)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:                # "nothing found"
                return []
            if e.code == 401:
                raise RuntimeError("OpenF1 is closed while a session is live: the next run tries again")
            time.sleep(30 if e.code == 429 else 10)
        except OSError:
            time.sleep(10)
    raise RuntimeError(f"OpenF1 didn't answer for {endpoint} {params}")


def finished(races, now):
    """The races that took place and ended long enough ago for OpenF1's history to have them."""
    return [r for r in races if not r.get("is_cancelled")
            and datetime.fromisoformat(r["date_end"].replace("Z", "+00:00")) < now - SETTLE]


def keys_in(s3, bucket, prefix):
    keys, token = [], None
    while True:
        page = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, **({"ContinuationToken": token} if token else {}))
        keys += [o["Key"] for o in page.get("Contents", [])]
        if not page.get("IsTruncated"):
            return keys
        token = page["NextContinuationToken"]


def built(s3, bucket):
    try:
        return set(json.loads(s3.get_object(Bucket=bucket, Key=INDEX)["Body"].read()))
    except Exception as e:
        if "NoSuchKey" in str(e) or "NoSuchKey" in type(e).__qualname__:
            return set()
        raise


def lambda_handler(event, context, s3=None, fetch=get, now=None):
    """AWS calls this every morning (and with {"session_key": ...} to fetch one race again)."""
    if s3 is None:
        import boto3
        s3 = boto3.client("s3")
    event = event or {}
    bucket = event.get("bucket") or os.environ["BUCKET"]
    now = now or datetime.now(timezone.utc)
    have = built(s3, bucket)
    waiting = {k[len(INCOMING):].split(".")[0] for k in keys_in(s3, bucket, INCOMING)}
    only = str(event["session_key"]) if event.get("session_key") else None
    years = sorted({now.year, (now - timedelta(days=45)).year})            # early in a season, last season too
    fetched, skipped = [], 0
    for year in years:
        for race in finished(fetch("sessions", year=year, session_name="Race"), now):
            key = str(race["session_key"])
            if only and key != only:
                continue
            if not only and (key in have or key in waiting):
                skipped += 1
                continue
            raw = {"session": race}
            for name in ENDPOINTS:
                raw[name] = fetch(name, session_key=race["session_key"])
            if not raw["laps"]:                                            # not in OpenF1 yet: tomorrow
                continue
            s3.put_object(Bucket=bucket, Key=f"{INCOMING}{key}.json.gz", ContentType="application/gzip",
                          Body=gzip.compress(json.dumps(raw).encode("utf-8")))
            fetched.append({"session_key": race["session_key"], "race": f"{race.get('year')} {race.get('country_name')}",
                            "laps": len(raw["laps"])})
    print(json.dumps({"fetched": fetched, "already": skipped}))            # to CloudWatch
    return {"fetched": fetched, "already": skipped}
