import math

def deg_to_quat(yaw_deg: float) -> tuple[float, float]:
    yaw_rad = math.radians(yaw_deg)
    orientation_z = math.sin(yaw_rad / 2)
    orientation_w = math.cos(yaw_rad / 2)
    return orientation_z, orientation_w

def yaw_to_quat(yaw_rad: float) -> tuple[float, float]:
    orientation_z = math.sin(yaw_rad / 2)
    orientation_w = math.cos(yaw_rad / 2)
    return orientation_z, orientation_w