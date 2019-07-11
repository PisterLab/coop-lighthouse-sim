import enum
import numpy as np
import State
import Vehicle

class DroneType(enum.Enum):
    lighthous_dronee = enum.auto()      # localizing itself and localizing anchor robots
    robot_drone = enum.auto()     # only taking measurements
    anchor_drone = enum.auto()      # in place, acts as an anchor point

class Drone:
    def __init__(self, init_state=State(), drone_type=DroneType.robot_drone):
        self.drone_type = drone_type

        self.ax = np.full(timesteps, .1)
        self.ax[0] = 1
        self.ay = np.zeros(timesteps)
        self.omega = np.full(timesteps, 2)

        self.omega_m = []
        self.ax_m = []
        self.ay_m = []
        self.vehicle = Vehicle(init_state)

        self.sig_x0 = 0.05  # initial uncertainty of x
        self.sig_y0 = 0.05  # initial uncertainty of y
        self.sig_th0 = 0.01  # initial uncertainty of theta
        self.sig_vx0 = 0.001  # initial uncertainty of vx
        self.sig_vy0 = 0.001  # initial uncertainty of vy

        # covariance of measurements
        self.Pm = [np.diag([self.sig_x0, self.sig_y0, self.sig_th0, self.sig_vx0, self.sig_vy0])]
