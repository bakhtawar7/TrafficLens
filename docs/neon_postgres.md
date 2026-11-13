# Neon PostgreSQL history setup

TrafficLens keeps SQLite as its default local database. When `DATABASE_URL` is set,
the Streamlit History tab uses Neon PostgreSQL. Do not commit `.env`; it is ignored.
Copy `.env.example` to `.env` and add the Neon pooled connection URL as
`DATABASE_URL`. Keep `DIRECT_DATABASE_URL` for administration; runtime code uses
the pooled URL.

## Install and migrate existing local history

From the repository root in the activated virtual environment:

```powershell
python -m pip install -r requirements.txt
python scripts/migrate_history_to_neon.py
```

The script creates a consistent timestamped SQLite backup beside
`data/history.db`, initializes the Neon schema, and copies runs, model results,
reliability rows, and all correction records. It is repeatable: existing conflicting
rows are skipped. It never deletes or modifies the source SQLite database. Compare
the printed source and Neon totals before switching deployment traffic.

To migrate another SQLite file:

```powershell
python scripts/migrate_history_to_neon.py --source "C:/path/to/history.db"
```

## Deployment

Set `DATABASE_URL` in the hosting provider's environment-variable settings (not in
GitHub). Use the Neon pooled connection string for the app. Install dependencies from
`requirements.txt`. On startup the History module creates the PostgreSQL tables if
they do not exist. No local SQLite file is required for deployed history.

`DIRECT_DATABASE_URL` is retained for administrative tooling; application queries
use `DATABASE_URL`. Uploaded images and videos are not stored in the history database.
They require separate object/file storage if persistence across deployments is needed.

## Safety and verification

- Keep `data/history.db` and the timestamped pre-migration backup until counts and
  representative run details/corrections have been checked in Neon.
- Never paste database URLs into source code, commits, screenshots, or public issues.
- Rotate any Neon password that has been exposed and update local and hosting secrets.
- Explicit `path=... ` arguments continue to use SQLite, preserving the existing tests
  and local database workflows.
