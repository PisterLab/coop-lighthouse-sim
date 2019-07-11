import enum

class DroneType(enum.Enum):
    lighthouse_robot = enum.auto()      # localizing itself and localizing anchor robots
    measurement_robot = enum.auto()     # only taking measurements
    anchor_robot = enum.auto()      # in place, acts as an anchor point

def __init__(self, x=5, y=5, theta=0, vx=0, vy=0, drone_type=DroneType.measurement_robot):
