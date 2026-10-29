"""Athena: ask questions of the race data in the cloud, in SQL, straight from the files in S3. There is no server or
database to run: Athena reads the Parquet files where they are and charges for the data it reads (5 dollars a
terabyte; a question here reads a few megabytes).

    python aws_athena.py setup             the database f1_strategy, with a table for each of the main files
    python aws_athena.py ask               every saved question (seasons, stints, tyres, sets)
    python aws_athena.py ask tyres         one of them
    python aws_athena.py ask "SELECT ..."  your own question

Each answer says how long Athena took and how much data it read. The tables take their columns from the files on
this PC (data/), so send them first (python aws_sync.py upload). In the AWS console the same tables are in Athena's
query editor (database f1_strategy), for the video.
"""
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

import aws_sync
from config import DATA_DIR

DATABASE = "f1_strategy"
TABLES = {"laps": "all_laps", "features": "features", "timing": "timing", "race_control": "race_control",
          "race_results": "race_results"}           # Athena's name: the file in data/
DRY = "compound IN ('SOFT', 'MEDIUM', 'HARD')"
QUESTIONS = {
    "seasons": ("Races and laps in each season",
                "SELECT year, COUNT(DISTINCT raceid) AS races, COUNT(*) AS laps FROM laps GROUP BY year ORDER BY year"),
    "stints": ("Tyre stints per driver per race (each stop adds one)",
               "SELECT year, ROUND(AVG(stints), 2) AS stints_per_driver FROM (SELECT year, raceid, driver, "
               "MAX(stint) AS stints FROM laps GROUP BY year, raceid, driver) GROUP BY year ORDER BY year"),
    "tyres": ("The share of the dry laps on each tyre",
              f"SELECT year, compound, COUNT(*) AS laps, ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER "
              f"(PARTITION BY year), 1) AS share FROM laps WHERE {DRY} GROUP BY year, compound ORDER BY year, compound"),
    "sets": ("How many laps a set of tyres lasts",
             f"SELECT year, compound, ROUND(AVG(laps), 1) AS laps_per_set FROM (SELECT year, raceid, driver, stint, "
             f"compound, COUNT(*) AS laps FROM laps WHERE {DRY} GROUP BY year, raceid, driver, stint, compound) "
             f"GROUP BY year, compound ORDER BY year, compound"),
}


def athena_type(t):
    if pa.types.is_dictionary(t):
        return athena_type(t.value_type)
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "string"
    if pa.types.is_floating(t):
        return "double"
    if pa.types.is_integer(t):
        return "bigint"
    if pa.types.is_boolean(t):
        return "boolean"
    if pa.types.is_list(t):
        return f"array<{athena_type(t.value_type)}>"
    raise ValueError(f"no Athena type for {t}")


def create_table(table, stem, bucket, data=DATA_DIR):
    """The SQL for one table, with the columns of the file on this PC and the folder the file is in, in the bucket."""
    schema = pq.read_schema(data / f"{stem}.parquet")
    columns = ",\n  ".join(f"`{n.lower()}` {athena_type(t)}" for n, t in zip(schema.names, schema.types)
                           if not n.startswith("__"))           # (an index pandas may have saved)
    folder = aws_sync.key_for(f"data/{stem}.parquet").rsplit("/", 1)[0]
    return (f"CREATE EXTERNAL TABLE IF NOT EXISTS {DATABASE}.{table} (\n  {columns}\n)\nSTORED AS PARQUET\n"
            f"LOCATION 's3://{bucket}/{folder}/'")


def run(athena, sql, bucket, database=DATABASE):
    """Ask Athena and wait for the answer: (rows with the header first, seconds Athena took, bytes it read)."""
    context = {"QueryExecutionContext": {"Database": database}} if database else {}
    qid = athena.start_query_execution(QueryString=sql, WorkGroup="primary", **context,
                                       ResultConfiguration={"OutputLocation": f"s3://{bucket}/athena-results/"})["QueryExecutionId"]
    while True:
        e = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
        if e["Status"]["State"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(0.4)
    if e["Status"]["State"] != "SUCCEEDED":
        raise SystemExit(f"Athena: {e['Status']['State']}: {e['Status'].get('StateChangeReason', '')}")
    rows = []
    for page in athena.get_paginator("get_query_results").paginate(QueryExecutionId=qid):
        rows += [[c.get("VarCharValue", "") for c in r["Data"]] for r in page["ResultSet"]["Rows"]]
    stats = e.get("Statistics", {})
    return rows, stats.get("EngineExecutionTimeInMillis", 0) / 1000, stats.get("DataScannedInBytes", 0)


def show(rows):
    if not rows:
        return
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for n, r in enumerate(rows):
        print("   " + "  ".join(v.rjust(w) if n and v.replace(".", "", 1).isdigit() else v.ljust(w)
                                for v, w in zip(r, widths)))


def main():
    cfg = aws_sync.settings()
    if not cfg:
        raise SystemExit("First: python aws_sync.py setup YOUR-BUCKET-NAME  (see AWS_SETUP.md)")
    athena, bucket = aws_sync.client(cfg, "athena"), cfg["bucket"]
    words = sys.argv[1:]
    if words[:1] == ["setup"]:
        run(athena, f"CREATE DATABASE IF NOT EXISTS {DATABASE}", bucket, database=None)
        for table, stem in TABLES.items():
            run(athena, f"DROP TABLE IF EXISTS {DATABASE}.{table}", bucket)      # so new columns are picked up
            run(athena, create_table(table, stem, bucket), bucket)
            print(f"Table {DATABASE}.{table}: data/{stem}.parquet")
    elif words[:1] == ["ask"]:
        asked = words[1:] or list(QUESTIONS)
        for q in asked:
            title, sql = QUESTIONS.get(q, (q, q))
            rows, seconds, read = run(athena, sql, bucket)
            print(f"\n{title}  ({seconds:.1f} s in Athena, {read / 1e6:.1f} MB read)")
            show(rows)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
