"""
csv_to_sqlite.py

Converts the three Kaggle medical CSVs into properly-typed SQLite databases:

    data/heart.csv     -> db/heart_disease.db   (table: heart_disease)
    data/cancer.csv     -> db/cancer.db          (table: cancer_patients)
    data/diabetes.csv   -> db/diabetes.db        (table: diabetes_patients)

Usage:
    python data_prep/csv_to_sqlite.py

Before running, download the three CSVs from Kaggle and place them in
./data/ with these exact names (rename after download):

    data/heart.csv     <- johnsmith88/heart-disease-dataset
    data/cancer.csv     <- rabieelkharoua/cancer-prediction-dataset
    data/diabetes.csv   <- iammustafatz/diabetes-prediction-dataset

Column types are inferred by pandas and then explicitly mapped to sensible
SQLite types (INTEGER / REAL / TEXT) rather than leaving everything as
SQLAlchemy's default guess, so the SQL agents get clean schemas to reason
about.
"""

import os
import sqlite3
import sys

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "db")

DATASETS = [
    {
        "csv": os.path.join(DATA_DIR, "heart.csv"),
        "db": os.path.join(DB_DIR, "heart_disease.db"),
        "table": "heart_disease",
    },
    {
        "csv": os.path.join(DATA_DIR, "cancer.csv"),
        "db": os.path.join(DB_DIR, "cancer.db"),
        "table": "cancer_patients",
    },
    {
        "csv": os.path.join(DATA_DIR, "diabetes.csv"),
        "db": os.path.join(DB_DIR, "diabetes.db"),
        "table": "diabetes_patients",
    },
]


def infer_sqlite_type(series: pd.Series) -> str:
    """Map a pandas dtype to a SQLite column type."""
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    if pd.api.types.is_bool_dtype(series):
        return "INTEGER"  # SQLite has no native bool; store 0/1
    return "TEXT"


def clean_column_name(col: str) -> str:
    """Normalize column names to snake_case, SQL-safe identifiers."""
    return (
        col.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def convert(csv_path: str, db_path: str, table_name: str) -> None:
    if not os.path.exists(csv_path):
        print(f"  [skip] {csv_path} not found — download it from Kaggle first.")
        return

    df = pd.read_csv(csv_path)
    df.columns = [clean_column_name(c) for c in df.columns]

    # Build an explicit CREATE TABLE statement so types aren't left to chance
    col_defs = ", ".join(f'"{c}" {infer_sqlite_type(df[c])}' for c in df.columns)
    create_stmt = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({col_defs});'

    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        cur.execute(create_stmt)
        conn.commit()

        df.to_sql(table_name, conn, if_exists="append", index=False)

        # Helpful indexes: primary-key-like id column if present
        if "id" in df.columns:
            cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{table_name}_id ON "{table_name}"(id)')
            conn.commit()

        row_count = cur.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        print(f"  [ok] {db_path} -> table '{table_name}' ({row_count} rows, {len(df.columns)} cols)")
    finally:
        conn.close()


def main():
    print("Converting CSVs to SQLite databases...\n")
    for spec in DATASETS:
        convert(spec["csv"], spec["db"], spec["table"])
    print("\nDone. Check the db/ directory for the resulting .db files.")


if __name__ == "__main__":
    sys.exit(main())
