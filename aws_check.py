"""The cloud's processing step: an AWS Lambda function that checks every table sent to the bucket's data/ folder and
writes a short report next to it, in checks/ (data/all_laps/all_laps.parquet -> checks/data/all_laps/all_laps.json).

It runs by itself on each upload (an S3 trigger on data/, for files ending .parquet). It checks that the file
arrived whole (its SHA-256 against the one aws_sync.py sent with it), counts the rows, seasons and races, and, for
the lap tables, looks for missing columns, laps listed twice, lap times missing and tyres it doesn't know. A report
says "ok" or "problems found", with the problems.

In AWS (AWS_SETUP.md has every step): a Python 3.12 function, this file pasted as lambda_function.py (the handler is
lambda_function.lambda_handler, AWS's default), the layer AWSSDKPandas-Python312 (AWS's own, with pandas), 512 MB and
a 1-minute timeout, and permission to read data/* and write checks/* in the bucket.
"""
import hashlib
import io
import json
import urllib.parse
from datetime import datetime, timezone

import pandas as pd

TYRES = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
LAP_COLUMNS = ["Driver", "LapNumber", "LapTime", "Stint", "Compound", "TyreLife", "Position", "Year", "Round"]
EXPECTED = {                                     # each table's columns that the app can't do without
    "all_laps": LAP_COLUMNS + ["RaceId", "Circuit"],
    "features": ["RaceId", "Driver", "LapNumber", "TyreLife", "Compound", "LapsLeft", "PitSoon"],
    "timing": ["RaceId", "Driver", "LapNumber", "S1", "S2", "S3"],
    "race_control": ["RaceId", "Lap", "Category", "Message"],
    "race_results": ["RaceId", "Driver", "Final", "Grid"],
}


def kind(key):
    """Which table a file in the bucket is: the name of a table in data/, or "race" for one race's laps (the
    downloads from FastF1 in data/raw/ and from OpenF1 in data/openf1/)."""
    parts = key.split("/")
    if len(parts) == 3 and parts[1] in ("raw", "openf1"):
        return "race"
    return parts[1] if len(parts) == 3 else None


def check(table, what=None):
    """What is in one table and anything wrong with it."""
    report = {"rows": len(table), "columns": len(table.columns), "problems": [], "notes": []}
    if len(table) == 0:
        report["problems"].append("no rows")
    expected = LAP_COLUMNS if what == "race" else EXPECTED.get(what, [])
    missing = [c for c in expected if c not in table]
    if missing:
        report["problems"].append("missing columns: " + ", ".join(missing))
    if "Year" in table:
        report["seasons"] = sorted(int(y) for y in table["Year"].dropna().unique())
    elif "RaceId" in table:
        report["seasons"] = sorted({int(r[:4]) for r in table["RaceId"].dropna().astype(str)})
    race = ["RaceId"] if "RaceId" in table else [c for c in ("Year", "Round") if c in table]
    if race:
        report["races"] = int(len(table.drop_duplicates(race)))
    if {"Driver", "LapNumber"} <= set(table.columns):                   # a table with a row per driver per lap
        twice = int(table.duplicated(race + ["Driver", "LapNumber"]).sum())
        if twice:
            report["problems"].append(f"{twice} laps listed twice")
        if "LapTime" in table:
            report["lap_times_missing"] = f"{table['LapTime'].isna().mean():.1%}"
        if "Compound" in table:
            tyres = table["Compound"].dropna().astype(str)
            odd = tyres[~tyres.isin(TYRES)]
            if len(odd):
                report["notes"].append(f"{len(odd)} laps on a tyre the timing didn't name ({', '.join(sorted(set(odd)))})")
    report["status"] = "problems found" if report["problems"] else "ok"
    return report


def lambda_handler(event, context, s3=None):
    """AWS calls this with the files just sent (an S3 event)."""
    if s3 is None:
        import boto3
        s3 = boto3.client("s3")
    reports = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        got = s3.get_object(Bucket=bucket, Key=key)
        body = got["Body"].read()
        sent = got.get("Metadata", {}).get("sha256")
        try:
            report = check(pd.read_parquet(io.BytesIO(body)), kind(key))
        except Exception as e:                   # not a table at all
            report = {"status": "problems found", "problems": [f"can't be read as a table: {e}"], "notes": []}
        whole = None if sent is None else hashlib.sha256(body).hexdigest() == sent    # None: sent without one
        if whole is False:
            report["problems"].insert(0, "the file isn't the one that was sent (its SHA-256 differs)")
            report["status"] = "problems found"
        report.update(file=key, bytes=len(body), arrived_whole=whole,
                      checked=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        name = "checks/" + key.rsplit(".", 1)[0] + ".json"
        s3.put_object(Bucket=bucket, Key=name, Body=json.dumps(report, indent=1).encode("utf-8"),
                      ContentType="application/json")
        reports.append(report)
        print(json.dumps({"file": key, "status": report["status"], "problems": report["problems"]}))   # to CloudWatch
    return {"checked": len(reports), "problems": sum(r["status"] != "ok" for r in reports)}
