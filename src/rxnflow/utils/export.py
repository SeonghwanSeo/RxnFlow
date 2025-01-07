from pathlib import Path
import sqlite3
import csv


def export_db(db_path: str | Path, out_csv_path: str | Path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = "SELECT * FROM results"
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]
    with open(out_csv_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        # Write column headers
        writer.writerow(columns)
        # Write data rows
        writer.writerows(rows)
    conn.close()
