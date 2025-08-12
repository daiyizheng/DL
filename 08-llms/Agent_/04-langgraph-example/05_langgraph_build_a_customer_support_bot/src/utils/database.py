import os
import shutil
import sqlite3

import pandas as pd
import requests

DB_URL = "https://storage.googleapis.com/benchmarks-artifacts/travel-db/travel2.sqlite"
LOCAL_FILE = "data/travel2.sqlite"
BACKUP_FILE = "data/travel2.backup.sqlite"


def download_database(overwrite=False):
    if overwrite or not os.path.exists(LOCAL_FILE):
        print("Downloading database...")
        response = requests.get(DB_URL)
        response.raise_for_status()
        os.makedirs(os.path.dirname(LOCAL_FILE), exist_ok=True)

        with open(LOCAL_FILE, "wb") as f:
            f.write(response.content)
        print(f"Database downloaded to {LOCAL_FILE}")

        shutil.copy(LOCAL_FILE, BACKUP_FILE)
        print(f"Backup created at {BACKUP_FILE}")
    else:
        print(f"Database already exists at {LOCAL_FILE}")


def update_dates():
    shutil.copy(BACKUP_FILE, LOCAL_FILE)
    conn = sqlite3.connect(LOCAL_FILE)

    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table';", conn
    ).name.tolist()
    tdf = {}
    for t in tables:
        tdf[t] = pd.read_sql(f"SELECT * from {t}", conn)

    example_time = pd.to_datetime(
        tdf["flights"]["actual_departure"].replace("\\N", pd.NaT)
    ).max()
    current_time = pd.to_datetime("now").tz_localize(example_time.tz)
    time_diff = current_time - example_time

    tdf["bookings"]["book_date"] = (
        pd.to_datetime(tdf["bookings"]["book_date"].replace("\\N", pd.NaT), utc=True)
        + time_diff
    )

    datetime_columns = [
        "scheduled_departure",
        "scheduled_arrival",
        "actual_departure",
        "actual_arrival",
    ]
    for column in datetime_columns:
        tdf["flights"][column] = (
            pd.to_datetime(tdf["flights"][column].replace("\\N", pd.NaT)) + time_diff
        )

    for table_name, df in tdf.items():
        df.to_sql(table_name, conn, if_exists="replace", index=False)
    del df
    del tdf
    conn.commit()
    conn.close()