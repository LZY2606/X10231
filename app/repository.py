"""Read/write the separated raw source records and version pointers."""
from __future__ import annotations

from .ident import make_id, source_id

BEAM_COLS = 8


def add_ping(con, ping_id, ord_, t, case_id, raw_ref=None):
    con.execute(
        "INSERT OR IGNORE INTO pings(ping_id,ord,t,case_id,raw_ref) "
        "VALUES(?,?,?,?,?)", (ping_id, ord_, t, case_id, raw_ref))


def write_beam_vel(con, sver, ping_id, channel, cell_idx, beam, raw, unit, geom_id):
    con.execute("INSERT OR REPLACE INTO beam_vel VALUES(?,?,?,?,?,?,?,?)",
                (source_id("beam_vel", ping_id, channel, cell_idx, beam),
                 sver, ping_id, channel, cell_idx, beam,
                 None if raw is None else str(raw), unit, geom_id))


def write_correlation(con, sver, ping_id, channel, cell_idx, beam, raw, unit):
    con.execute("INSERT OR REPLACE INTO correlation VALUES(?,?,?,?,?,?,?,?)",
                (source_id("correlation", ping_id, channel, cell_idx, beam),
                 sver, ping_id, channel, cell_idx, beam,
                 None if raw is None else str(raw), unit))


def write_amplitude(con, sver, ping_id, channel, cell_idx, beam, raw, unit):
    con.execute("INSERT OR REPLACE INTO amplitude VALUES(?,?,?,?,?,?,?,?)",
                (source_id("amplitude", ping_id, channel, cell_idx, beam),
                 sver, ping_id, channel, cell_idx, beam,
                 None if raw is None else str(raw), unit))


def write_badbeam(con, sver, ping_id, channel, cell_idx, beam, flag):
    con.execute("INSERT OR REPLACE INTO badbeam VALUES(?,?,?,?,?,?,?,?)",
                (source_id("badbeam", ping_id, channel, cell_idx, beam),
                 sver, ping_id, channel, cell_idx, beam,
                 1 if flag else 0, str(flag)))


def write_attitude(con, sver, ping_id, heading, pitch, roll,
                   h_unit, p_unit, r_unit, heading_basis):
    con.execute("INSERT OR REPLACE INTO attitude VALUES(?,?,?,?,?,?,?,?,?)",
                (source_id("attitude", ping_id), sver, ping_id,
                 None if heading is None else str(heading),
                 None if pitch is None else str(pitch),
                 None if roll is None else str(roll),
                 h_unit, p_unit, r_unit, heading_basis))


def write_declination(con, sver, ord_, value, unit, version, label):
    con.execute("INSERT OR REPLACE INTO declination VALUES(?,?,?,?,?,?)",
                (make_id("src", "declination", version), sver, ord_,
                 None if value is None else str(value), unit, version, label))


def write_installation(con, sver, ord_, heading_mount, pitch_offset,
                       roll_offset, unit, version, label):
    con.execute("INSERT OR REPLACE INTO installation VALUES(?,?,?,?,?,?,?,?,?)",
                (make_id("src", "installation", version), sver, ord_,
                 None if heading_mount is None else str(heading_mount),
                 None if pitch_offset is None else str(pitch_offset),
                 None if roll_offset is None else str(roll_offset),
                 unit, version, label))


def write_geom(con, sver, ord_, beam_angle, beam_angle_unit, n_beams,
               version, convention, label):
    con.execute("INSERT OR REPLACE INTO geom VALUES(?,?,?,?,?,?,?,?,?)",
                (make_id("src", "geom", version), sver, ord_,
                 None if beam_angle is None else str(beam_angle),
                 beam_angle_unit, n_beams, version, convention, label))


def write_cell(con, sver, ping_id, cell_idx, depth_raw, depth_unit, configured):
    con.execute("INSERT OR REPLACE INTO cells VALUES(?,?,?,?,?,?,?)",
                (source_id("cells", ping_id, cell_idx), sver, ping_id, cell_idx,
                 None if depth_raw is None else str(depth_raw),
                 depth_unit, 1 if configured else 0))


def write_bottom_range(con, sver, ping_id, beam, raw, unit):
    con.execute("INSERT OR REPLACE INTO bottom_range VALUES(?,?,?,?,?,?)",
                (source_id("bottom_range", ping_id, beam), sver, ping_id, beam,
                 None if raw is None else str(raw), unit))


def _rows(con, table, sver):
    return [dict(r) for r in con.execute(
        f"SELECT * FROM {table} WHERE sver=?", (sver,))]


def load_snapshot(con, sver):
    return {"pings": [dict(r) for r in con.execute("SELECT * FROM pings ORDER BY ord")],
            "beam_vel": _rows(con, "beam_vel", sver),
            "correlation": _rows(con, "correlation", sver),
            "amplitude": _rows(con, "amplitude", sver),
            "badbeam": _rows(con, "badbeam", sver),
            "attitude": _rows(con, "attitude", sver),
            "declination": _rows(con, "declination", sver),
            "installation": _rows(con, "installation", sver),
            "geom": _rows(con, "geom", sver),
            "cells": _rows(con, "cells", sver),
            "bottom_range": _rows(con, "bottom_range", sver)}
