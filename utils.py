import enum

class DroneType(enum.Enum):
    lighthouse_robot = enum.auto()      # localizing itself and localizing anchor robots
    measurement_robot = enum.auto()     # only taking measurements
    anchor_robot = enum.auto()      # in place, acts as an anchor point
