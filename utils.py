import enum

class DroneType(enum.Enum):
    lighthouse_robot = enum.auto()      # localizing itself and localizing anchor robots
    measurement_robot = enum.auto()     # only taking measurements
    anchor_robot = enum.auto()      # in place, acts as an anchor point

def step_dynamics(pos, vel, att, acc, omega, dt):
    x = pos[0] + vel[0] * dt
    y = pos[1] + vel[1] * dt
    theta = (att + dt * omega + 3.14159) % (2 * 3.14159) - 3.14159
    vx = vel[0] + (math.cos(att) * acc[0] - math.sin(att) * acc[1]) * dt
    vy = vel[1] + (math.sin(att) * acc[0] + math.cos(att) * acc[1]) * dt
    return [x, y], [vx, vy], theta
