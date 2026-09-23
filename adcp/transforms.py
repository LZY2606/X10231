"""ADCP coordinate transforms.

Conventions
-----------
- Angles are degrees, velocities are m/s, depths are metres.
- Instrument frame: x forward, y starboard, z up (beam radials mapped via
  the Janus beam matrix below).
- Ship frame: instrument frame rotated by the mounting (alignment) angle
  about the vertical axis.
- Earth frame: ship frame rotated by roll, pitch, then yaw
  (heading + magnetic declination), i.e. ENU.
"""
from __future__ import annotations

import math

import numpy as np

# Transform-order identifiers.  The order string is part of every derived
# fingerprint so a change of order always yields a different identity.
ORDER_MOUNTING_FIRST = "instr>ship(mounting)>earth(yaw,pitch,roll,decl)"
ORDER_ATTITUDE_FIRST = "instr>earth(yaw,pitch,roll,decl)>ship(mounting)"


def unwrap_degrees(headings):
    """Unwrap a heading sequence so that 359.5 -> 0.5 steps +1 deg."""
    if not headings:
        return []
    out = [float(headings[0])]
    for raw in headings[1:]:
        h = float(raw)
        prev = out[-1]
        while h - prev > 180.0:
            h -= 360.0
        while h - prev < -180.0:
            h += 360.0
        out.append(h)
    return out


def interpolate_attitude(samples, t, max_gap_s):
    """Interpolate heading/pitch/roll at time ``t``.

    ``samples`` is a list of ``(time, heading, pitch, roll)`` sorted by time.
    Returns ``(result, diagnostic)`` where ``result`` is a dict or ``None``.
    Interpolation across a time gap larger than ``max_gap_s`` is refused:
    the value is left in place with a diagnostic instead of being invented.
    """
    if not samples:
        return None, "no attitude samples available"
    times = [s[0] for s in samples]
    heads = unwrap_degrees([s[1] for s in samples])
    prev_i = None
    next_i = None
    for i, st in enumerate(times):
        if st <= t:
            prev_i = i
        if st >= t and next_i is None:
            next_i = i
    if prev_i is not None and times[prev_i] == t:
        i = prev_i
        return ({"heading": heads[i] % 360.0, "pitch": float(samples[i][2]),
                 "roll": float(samples[i][3]), "t": t, "method": "exact"},
                None)
    if prev_i is not None and next_i is not None and prev_i != next_i:
        gap = times[next_i] - times[prev_i]
        if gap > max_gap_s:
            return None, (
                f"attitude gap {gap:.2f}s around t={t} exceeds max_gap "
                f"{max_gap_s}s; interpolation across gap refused")
        f = (t - times[prev_i]) / gap
        h = heads[prev_i] + f * (heads[next_i] - heads[prev_i])
        p = samples[prev_i][2] + f * (samples[next_i][2] - samples[prev_i][2])
        r = samples[prev_i][3] + f * (samples[next_i][3] - samples[prev_i][3])
        return ({"heading": h % 360.0, "pitch": float(p), "roll": float(r),
                 "t": t, "method": "interpolated"}, None)
    # Only one side available: nearest-sample hold within tolerance.
    cand = prev_i if prev_i is not None else next_i
    if cand is not None and abs(times[cand] - t) <= max_gap_s:
        return ({"heading": heads[cand] % 360.0,
                 "pitch": float(samples[cand][2]),
                 "roll": float(samples[cand][3]), "t": t,
                 "method": "nearest-hold"}, None)
    return None, (f"no attitude sample within max_gap {max_gap_s}s of t={t}; "
                  "value left in instrument frame")


def beam_matrix(theta_deg):
    """Janus 4-beam matrix mapping instrument (u, v, w) to beam radials."""
    s = math.sin(math.radians(theta_deg))
    c = math.cos(math.radians(theta_deg))
    return np.array([
        [s, 0.0, c],    # beam 1
        [-s, 0.0, c],   # beam 2
        [0.0, s, c],    # beam 3
        [0.0, -s, c],   # beam 4
    ])


def instrument_from_beams(beam_vels, theta_deg):
    """Least-squares instrument-frame velocity from 3 or 4 beam radials.

    Returns ``(vector, error_velocity)``.  With 4 beams the residual norm is
    the error velocity; with 3 beams the system is exactly determined and the
    residual is ~0 -- callers must keep 3-beam and 4-beam solutions as
    separate candidates and never merge them into one value.
    """
    b = np.asarray([v for v in beam_vels if v is not None], dtype=float)
    if b.size < 3:
        raise ValueError("need at least 3 beams for an instrument solution")
    m = beam_matrix(theta_deg)
    m = m[[i for i, v in enumerate(beam_vels) if v is not None]]
    x, *_ = np.linalg.lstsq(m, b, rcond=None)
    err = float(np.linalg.norm(b - m @ x))
    return x, err


def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0],
                     [0, math.cos(a), -math.sin(a)],
                     [0, math.sin(a), math.cos(a)]])


def rot_y(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)],
                     [0, 1, 0],
                     [-math.sin(a), 0, math.cos(a)]])


def rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0],
                     [math.sin(a), math.cos(a), 0],
                     [0, 0, 1]])


def instrument_to_ship(v_instr, mounting_angle_deg):
    """Rotate instrument-frame vector into the ship frame (yaw alignment)."""
    return rot_z(mounting_angle_deg) @ np.asarray(v_instr, dtype=float)


def ship_to_earth(v_ship, heading, pitch, roll, declination_deg):
    """Rotate ship-frame vector into ENU earth frame.

    Magnetic declination is applied to the heading (versioned parameter).
    """
    yaw = heading + declination_deg
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll) @ np.asarray(v_ship, float)


def instr_to_earth(v_instr, attitude, mounting_angle_deg, declination_deg,
                   order=ORDER_MOUNTING_FIRST):
    """Full instrument->earth transform honouring the recorded order.

    The order string is returned alongside so callers can embed it in the
    derived fingerprint.
    """
    v = np.asarray(v_instr, dtype=float)
    att = rot_z(attitude["heading"] + declination_deg) \
        @ rot_y(attitude["pitch"]) @ rot_x(attitude["roll"])
    if order == ORDER_MOUNTING_FIRST:
        ship = rot_z(mounting_angle_deg) @ v
        earth = att @ ship
    elif order == ORDER_ATTITUDE_FIRST:
        earth = rot_z(mounting_angle_deg) @ (att @ v)
        ship = None
    else:
        raise ValueError(f"unknown transform order: {order}")
    return earth, (None if ship is None else ship.tolist()), order
