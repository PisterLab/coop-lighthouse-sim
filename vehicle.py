# (c) 2019 Mark Mueller

from __future__ import division, print_function
import numpy as np
from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
from motor import Motor
from imu import IMU
from Estimator import Estimator6Dof

class Vehicle:
    def __init__(self, mass, inertiaMatrix, omegaSqrToDragTorque, disturbanceTorqueStdDev,estimator = "6dof",drone_type=DroneType.robot_drone):
        self.drone_type = drone_type
        self._inertia = inertiaMatrix
        self._mass = mass

        self._pos = Vec3(0,0,0)
        self._vel = Vec3(0,0,0)
        self._att = Rotation.identity()
        self._omega = Vec3(0,0,0)

        #imu measurements
        self._accImu = Vec3(0,0,0) #body axis acceleration including gravity acc
        self._omegaImu = Vec3(0,0,0) #body axis angular vel
        self._magImu = Vec3(0,0,0) #body axis magnetometer

        #imu object
        accStd = Vec3(0.01,0.01,0.01)
        gyroStd = Vec3(0.01,0.01,0.01)
        magStd = Vec3(0.05,0.05,0.05)
        testImu = False
        self._imu = IMU(accStd,gyroStd,magStd,testImu)



        self._motors = []

        self._omegaSqrToDragTorque = omegaSqrToDragTorque

        self._disturbanceTorqueStdDev = disturbanceTorqueStdDev

        #6dof kalman estimator
        if estimator == "6dof":
            self._estimator = Estimator6Dof(self._mass,self._inertia,self._omegaSqrToDragTorque, self._disturbanceTorqueStdDev, self._pos, self._vel, self._att)
        else:
            self._estimator = None

        #state history
        self.posHist = []
        self.stateHist = []
        self.attHist = []

        return


    def add_motor(self, motorPosition, spinDir, minSpeed, maxSpeed, speedSqrToThrust, speedSqrToTorque, timeConst, inertia):
        self._motors.append(Motor(motorPosition, spinDir, minSpeed, maxSpeed, speedSqrToThrust, speedSqrToTorque, timeConst, inertia))
        return


    def run(self, dt, motorThrustCommands):


        totalForce_b  = Vec3(0,0,0)
        totalTorque_b = Vec3(0,0,0)
        for (mot,thrustCmd) in zip(self._motors, motorThrustCommands):
            mot.run(dt, thrustCmd)

            totalForce_b  += mot._thrust
            totalTorque_b += mot._torque

        totalTorque_b += (- self._omega.norm2()*self._omegaSqrToDragTorque*self._omega)

        #add noise:
        totalTorque_b += Vec3(np.random.normal(), np.random.normal(), np.random.normal())*self._disturbanceTorqueStdDev


        angMomentum = self._inertia*self._omega
        for mot in self._motors:
            angMomentum += mot._angularMomentum

        angAcc = np.linalg.inv(self._inertia)*(totalTorque_b - self._omega.cross(angMomentum))

        #translational acceleration:
        acc = Vec3(0,0,-9.81)  # gravity
        acc += self._att*totalForce_b/self._mass

        vel = self._vel
        att = self._att
        omega = self._omega

        #generate imu measurements
        (self._accImu, self._omegaImu, self._magImu) = self._imu.get_imu_measurements(acc = acc, att = att, omega = omega)
        #euler integration
        self._pos += vel*dt
        self._vel += acc*dt
        self._att  = att*Rotation.from_rotation_vector(omega*dt)
        self._omega += angAcc*dt

        #record state
        self.posHist.append(self._pos)
        self.velHist.append(self._vel)
        self.attHist.append(self._att)

    def kalman_predict(self, dt):

        if self._estimator == None:
            return
        else:
            #print("kalman predicting")
            self._estimator.kalmanPredict(self._accImu,self._omegaImu,dt)

    def kalman_update(self, dt):

        if self._estimator == None:
            return
        else:
            #print("kalman updating")
            self._estimator.kalmanUpdate(dt)
    def set_position(self, pos):
        self._pos = pos


    def set_velocity(self, velocity):
        self._vel = velocity


    def set_attitude(self, att):
        self._att = att


    def get_num_motors(self):
        return len(self._motors)


    def get_motor_speeds(self):
        out = np.zeros(len(self._motors,))
        for i in range(len(self._motors)):
            out[i] = self._motors[i]._speed
        return out


    def get_motor_forces(self):
        out = np.zeros(len(self._motors,))
        for i in range(len(self._motors)):
            out[i] = self._motors[i]._thrust.z
        return out

    def get_total_power_consumption(self):
        pwr = 0
        for m in self._motors:
            pwr += m._powerConsumptionInstantaneous


        return pwr

class 2DVehicle:
    def __init__(self, drone_type=DroneType.robot_drone):
        self.drone_type = drone_type

        # TODO: Fix this and the names
        self._pos = [0, 0]
        self._vel = [0, 0]
        self._accel = [0, 0]
        self._att = 0

        self.posHist = []
        self.velHist = []
        self.attHist = []

        self._estimator =  ""

        # Should run be used as an overarching function with three changing functions or the previous implementation?

    def run(self, dt):

        self.posHist.append(self._pos)
        self.velHist.append(self._vel)
        self.attHist.append(self._att)

        if self.drone_type == DroneType.lighthouse_drone:
            self.run_lighthouse(dt)
        elif self.drone_type == DroneType.robot_drone:
            self.run_robot(dt)
        elif self.drone_type == DroneType.anchor_drone: # I know that you can use else for this but I'm doing this for code clarity
            self.run_anchor(dt)



    def run_lighthouse(self, dt):
        self.step_dynamics(dt)

    def run_robot(self, dt):
        self.step_dynamics(dt)

    def run_anchor(self, dt):
        print("No movement.")

    # Use States?
    # where do i get ax, ay, and omega
    # Should I move this to a different method?
    # Should I save the states in this method?
    def step_dynamics(self, dt):
        x = self._pos[0] + self._vel[0] * dt
        y = self._pos[1] + self._vel[1] * dt
        theta = (state_truth_prev.theta + dt * omega + 3.14159) % (2 * 3.14159) - 3.14159
        vx = state_truth_prev.vx + (math.cos(state_truth_prev.theta) * ax - math.sin(state_truth_prev.theta) * ay) * dt
        vy = state_truth_prev.vy + (math.sin(state_truth_prev.theta) * ax + math.cos(state_truth_prev.theta) * ay) * dt


class DroneType(enum.Enum):
    lighthouse_drone = enum.auto()      # localizing itself and localizing anchor robots
    robot_drone = enum.auto()     # only receiving measurements
    anchor_drone = enum.auto()      # in place, acts as an anchor point
