from __future__ import annotations


def shortest_angle_delta(target: float, reference: float) -> float:
    return (target - reference + 180.0) % 360.0 - 180.0


def interpolate_heading(left: float, right: float, fraction: float) -> float:
    return (left + shortest_angle_delta(right, left) * fraction) % 360.0


def interpolate_linear(left: float, right: float, fraction: float) -> float:
    return left + (right - left) * fraction


def interpolate_attitude(
    target_time: float,
    samples: list[dict],
    max_gap_s: float,
) -> dict:
    ordered = sorted(samples, key=lambda item: item["time_epoch_s"])
    for sample in ordered:
        if float(sample["time_epoch_s"]) == float(target_time):
            return {
                "status": "EXACT",
                "heading_deg": float(sample["heading_deg"]),
                "pitch_deg": float(sample["pitch_deg"]),
                "roll_deg": float(sample["roll_deg"]),
                "left_id": sample["id"],
                "right_id": sample["id"],
                "fraction": 0.0,
                "gap_s": 0.0,
            }
    left = None
    right = None
    for sample in ordered:
        if float(sample["time_epoch_s"]) < target_time:
            left = sample
        if float(sample["time_epoch_s"]) > target_time and right is None:
            right = sample
            break
    if left is None or right is None:
        return {"status": "MISSING_EXTERNAL_BOUND", "diagnostic": "ATTITUDE_NOT_BOUNDED"}
    gap = float(right["time_epoch_s"]) - float(left["time_epoch_s"])
    if gap > max_gap_s:
        return {
            "status": "MISSING_GAP",
            "diagnostic": "ATTITUDE_GAP_NOT_BRIDGED",
            "left_id": left["id"],
            "right_id": right["id"],
            "gap_s": gap,
        }
    fraction = (target_time - float(left["time_epoch_s"])) / gap
    return {
        "status": "INTERPOLATED",
        "heading_deg": interpolate_heading(float(left["heading_deg"]), float(right["heading_deg"]), fraction),
        "pitch_deg": interpolate_linear(float(left["pitch_deg"]), float(right["pitch_deg"]), fraction),
        "roll_deg": interpolate_linear(float(left["roll_deg"]), float(right["roll_deg"]), fraction),
        "left_id": left["id"],
        "right_id": right["id"],
        "fraction": fraction,
        "gap_s": gap,
    }
