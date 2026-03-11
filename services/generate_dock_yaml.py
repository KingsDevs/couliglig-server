import math
import yaml
from schema import Dock
from typing import List
import io

def flip_theta(theta: float) -> float:
    angle = theta + math.pi
    angle = (angle + math.pi) % (2 * math.pi) - math.pi

    if abs(angle) < 1e-6:
        angle = 0.0

    if abs(angle + math.pi) < 1e-6:
        angle = math.pi

    return round(angle, 2)


def clean_num(x: float) -> float:
    if x is None:
        return 0.0
    x = float(x)
    if abs(x) < 1e-6:
        x = 0.0
    return round(x, 2)

def clean_angle(angle: float) -> float:
    angle = round(angle, 2)

    # remove negative zero
    if abs(angle) < 1e-6:
        angle = 0.0

    return angle


class InlineList(list):
    pass


class InlineListDumper(yaml.SafeDumper):
    pass


def represent_inline_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


InlineListDumper.add_representer(InlineList, represent_inline_list)


def generate_dock_yaml(docks: List[Dock], output_path: str):

    yaml_data = {"docks": {}}

    for dock in docks:
        x = clean_num(dock.x)
        y = clean_num(dock.y)
        theta = clean_num(dock.theta or 0.0)
        aruco_theta = clean_angle(flip_theta(theta))

        yaml_data["docks"][dock.dock_id] = {
            "pose": InlineList([x, y, theta]),
            "aruco_markers": {
                "pos": InlineList([x, y, aruco_theta]),
                "center": dock.aruco_id,
            },
        }

    yaml_str = yaml.dump(
        yaml_data,
        Dumper=InlineListDumper,
        sort_keys=False,
        default_flow_style=False,
    )

    buffer = io.BytesIO()
    buffer.write(yaml_str.encode("utf-8"))
    buffer.seek(0)

    return buffer