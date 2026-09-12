"""Leg kinematics.

Extracted verbatim from the duplicated copies that lived in main-ik-step.py,
main-ik-linear.py and ik-visualizer.py. Lengths are in millimetres and angles in
radians; `is_left_stance` already selects the correct yaw branch for either leg.
"""

import math


def calculate_leg_ik(x: float, y: float, z: float,
                     is_left_stance: bool = True,
                     knee_forward: bool = True) -> tuple[float, float, float]:
    """Joint angles (q1, q2, q3) for a foot target relative to joint 1."""
    # --- Joint 1 (yaw) ---
    xz_dist_sq = x ** 2 + z ** 2
    if xz_dist_sq < 6.25:
        xz_dist_sq = 6.25

    yaw_offset_angle = math.acos(-2.5 / math.sqrt(xz_dist_sq))
    if is_left_stance:
        q1 = math.atan2(z, x) + yaw_offset_angle
    else:
        q1 = math.atan2(z, x) - yaw_offset_angle

    # --- Translate into the 2D leg plane ---
    y_prime = y - 88.5
    z_prime = z * math.cos(q1) - x * math.sin(q1)
    L_diag_sq = y_prime ** 2 + z_prime ** 2

    # --- Joint 3 (knee) ---
    D = (L_diag_sq - 102500.0) / 100000.0
    D = max(-1.0, min(1.0, D))
    knee_inner_angle = math.acos(D)

    calf_offset = 3.0 * math.pi / 4.0
    if knee_forward:
        q3 = knee_inner_angle - calf_offset
    else:
        q3 = -knee_inner_angle - calf_offset

    # --- Joint 2 (hip pitch) ---
    phi = -q3 - calf_offset
    k1 = 200.0 + 250.0 * math.cos(phi)
    k2 = 250.0 * math.sin(phi)
    q2 = math.atan2(z_prime, y_prime) - math.atan2(k2, k1)

    return q1, q2, q3
