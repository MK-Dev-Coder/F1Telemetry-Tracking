# The cloud data layer on AWS: setting it up

The app works without any of this. With it, the race data and the models have a safe copy in Amazon S3 (with every
older version kept), a Lambda function checks each table that arrives, Athena answers questions about the data in
SQL, and the desktop app fetches the newest data when it starts. The game's recordings never go to AWS.

About an hour, once. Do every step in one region: **Europe (Frankfurt) eu-central-1**, or **Europe (Stockholm)
eu-north-1** on AWS's free plan, which allows only that one (the menu at the top right of the AWS console). On the
free plan each project is an AWS account of its own: open the console from the project that holds the bucket (the
account ID is at the top right). Keep the keys to yourself: never put them in the project folder, an email or a chat.

## 1. An account and a budget alarm (10 minutes)

1. Create an AWS account at <https://aws.amazon.com> (an email, a card and a phone), or sign in to yours. Turn on
   MFA for the account (IAM → the security recommendations at the top).
2. **Billing and Cost Management → Budgets → Create budget → Use a template → Zero spend budget** (or a monthly
   budget of $1), with your email. AWS then emails you if anything starts to cost money.

## 2. The bucket (3 minutes)

**S3 → Create bucket:**

- Bucket name: for example `f1-strategy-yourname-2026` (lowercase; it must be unique in the whole of AWS).
- Block all public access: **on** (the default).
- Bucket Versioning: **Enable** (S3 then keeps every older copy of a file).
- Default encryption: leave it as it is (SSE-S3). **Create bucket.**

On this PC, in the project folder:

```bat
python aws_sync.py setup f1-strategy-yourname-2026
```

In Stockholm, add the region: `python aws_sync.py setup f1-strategy-yourname-2026 eu-north-1`.

## 3. A key for the app (5 minutes)

1. **IAM → Users → Create user**: name `f1-strategy-app`, no console access → Next → Next → **Create user**.
2. Open the user → **Permissions → Add permissions → Create inline policy → JSON**. Paste what this prints (your
   bucket only, and no deleting), then Next, name it `f1-strategy-bucket`, **Create policy**:

   ```bat
   python aws_sync.py policy
   ```

3. **Security credentials → Create access key → Local code** (or "Application running outside AWS") → Next →
   **Create access key**. Keep the page open, and on this PC paste both parts (paste the secret with a right-click:
   it doesn't show, and Ctrl+V doesn't paste into a hidden prompt in every window). It checks them and tests the key
   on the bucket:

   ```bat
   python aws_sync.py keys
   ```

   It keeps them in `C:\Users\<you>\.aws\credentials`, where AWS's own tools keep them, not in the project.

## 4. The Lambda check (10 minutes)

**Lambda → Create function → Author from scratch:** name `f1-strategy-check`, runtime **Python 3.12** → **Create
function**. Then:

1. **Code**: open `lambda_function.py`, replace everything in it with the contents of `aws_check.py`, **Deploy**.
2. **Layers** (under the code): **Add a layer → AWS layers → AWSSDKPandas-Python312** (AWS's own, with pandas),
   the newest version → **Add**.
3. **Configuration → General configuration → Edit**: memory **512 MB**, timeout **1 min** → Save.
4. **Configuration → Permissions**: click the role's name (it opens IAM) → **Add permissions → Create inline policy
   → JSON**, paste what this prints (read the tables, write the reports), name it `f1-strategy-check-bucket`:

   ```bat
   python aws_sync.py policy check
   ```

5. **Add trigger → S3**: your bucket, **All object create events**, prefix `data/`, suffix `.parquet`, tick the box
   about recursive invocation (the reports go to `checks/`, outside `data/`, so it can't call itself) → **Add**.

## 5. The first upload (1 minute)

```bat
python aws_sync.py upload
python aws_sync.py status
```

About 240 files (25 MB) go up. Each table gets its check: in S3, open the bucket's `checks/` folder (for example
`checks/data/all_laps/all_laps.json`), or see **CloudWatch → Log groups → /aws/lambda/f1-strategy-check**.

After a new race (the usual steps in the README), run `python aws_sync.py upload` again: only what changed goes up,
and S3 keeps the copies before as older versions (`python aws_sync.py upload --all` sends every file again).

## 6. Athena (5 minutes)

```bat
python aws_athena.py setup
python aws_athena.py ask
```

`setup` makes the database `f1_strategy` with the tables `laps`, `features`, `timing`, `race_control` and
`race_results`; `ask` runs the saved questions (races and laps per season, tyre stints per race, the share of laps on
each tyre, how long a set lasts), each with the time Athena took and the data it read. Your own question:
`python aws_athena.py ask "SELECT circuit, COUNT(*) AS laps FROM laps GROUP BY circuit ORDER BY laps DESC"`.

In the console (for the video): **Athena → Query editor → Settings → Manage**, query result location
`s3://f1-strategy-yourname-2026/athena-results/` → Save; then in the editor pick the database `f1_strategy`.

## 7. The desktop app (1 minute)

```bat
python desktop.py --shortcut
```

Double-click **F1 Strategy Predictor** on the desktop. It fetches any newer data from the bucket, starts the app on
this PC and opens it in its own window; closing the window stops it. On another PC: install the project (README), do
steps 2 to 3 there (`setup` and `keys`, with the same key or a new one), and the first start fetches everything.

## 8. The numbers for the report

```bat
python aws_sync.py check
```

It fetches everything into an empty folder, as a new PC would, times it and compares every file with this PC's
(saved in `results/cloud.md`). `python aws_sync.py status` counts the versions S3 keeps.

## 9. New races by themselves (optional, about 10 minutes)

A second Lambda fetches each new Grand Prix from OpenF1 into the bucket the morning after it, exactly as
`collect_openf1.py` saves a race; the desktop app then builds it on this PC with the usual steps (`newraces.py`),
sends the results back and says when it's in.

1. On this PC, the races you have already (so they aren't fetched again): `python newraces.py --index`, then
   `python aws_sync.py upload`.
2. **Lambda → Create function**: name `f1-strategy-fetch`, runtime **Python 3.12** → paste `aws_fetch.py` into
   `lambda_function.py` → **Deploy**. **Configuration → General configuration**: 256 MB, timeout 5 min.
   **Configuration → Environment variables**: `BUCKET` = your bucket's name. **Configuration → Permissions**: the
   role → inline policy with what `python aws_sync.py policy fetch` prints.
3. **EventBridge → Scheduler → Create schedule**: name `f1-strategy-daily-fetch`, recurring, cron `0 6 * * ? *`
   (every day at 06:00 UTC), flexible time window off → target **AWS Lambda Invoke**, the function
   `f1-strategy-fetch` → let it create its role → **Create schedule**.
4. To try it: in the Lambda, **Test** with the event `{"session_key": 11396}` (the 2026 United States GP fetched
   again); it appears in the bucket's `data/openf1/incoming/` in about 20 seconds. `python newraces.py` builds any
   race that is new to this PC (the desktop app does it by itself when it starts).

## What it costs, and turning it off

- **S3**: about 25 MB and a few older versions: a few cents a month at most.
- **Lambda**: a few hundred checks a month, inside AWS's free allowance.
- **Athena**: $5 per terabyte read; a question here reads a few megabytes.
- Check the prices on AWS's own pages; the budget alarm emails you before anything adds up.

When you no longer need it: delete the Lambda function, empty the bucket (with **Show versions** on) and delete it,
delete the access key (IAM → Users → f1-strategy-app → Security credentials), and the budget.
