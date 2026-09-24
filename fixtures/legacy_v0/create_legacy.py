"""Create the in-repo legacy v0 fixture database (tests use this file only)."""

from __future__ import annotations

import os
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE ping_rows (
    ping_id TEXT PRIMARY KEY,
    t_s REAL,
    heading_deg REAL, pitch_deg REAL, roll_deg REAL,
    declination_deg REAL, mount_deg REAL,
    beam1_cmps REAL, beam2_cmps REAL, beam3_cmps REAL, beam4_cmps REAL,
    corr1 INTEGER, corr2 INTEGER, corr3 INTEGER, corr4 INTEGER,
    amp1 REAL, amp2 REAL, amp3 REAL, amp4 REAL,
    badflag1 INTEGER, badflag2 INTEGER, badflag3 INTEGER, badflag4 INTEGER,
    bottom_m REAL
);
"""

PINGS = [
    ("p01", 0.0, 340.0, 2.1, -1.3, 7.0, 0.0,
     10.3, -9.0, 6.9, -5.5, 110, 105, 108, 102,
     90.0, 91.0, 89.5, 90.2, 0, 0, 0, 0, 20.0),
    ("p02", 4.0, 348.0, 2.2, -1.4, 7.0, 0.0,
     10.1, -8.9, 6.4, -5.2, 112, 100, 99, 96,
     87.0, 88.1, 86.4, 87.2, 0, 0, 0, 0, 20.0),
    ("p03", 8.0, 356.0, 2.0, -1.2, 7.0, 0.0,
     9.8, -8.8, 5.9, -4.9, 108, 103, 101, 0,
     84.0, 85.1, 83.2, 0.0, 0, 0, 0, 1, 20.0),
]


def main() -> str:
    path = os.path.join(HERE, "legacy_adcp_v0.db")
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO meta VALUES('schema_version','0')")
    conn.executemany(
        "INSERT INTO ping_rows VALUES(" + ",".join(["?"] * 24) + ")",
        PINGS,
    )
    conn.commit()
    conn.close()
    return path


if __name__ == "__main__":
    print(main())
