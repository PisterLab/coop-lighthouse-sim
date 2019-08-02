# (c) 2019 Mark Mueller

from __future__ import division, print_function
import numpy as np
from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
from motor import Motor
from imu import IMU, IMU2D
from Estimator import Estimator6Dof #, Estimator3Dof
import enum

class DroneType(enum.Enum):
    lighthouse_drone = enum.auto()      # localizing itself and localizing anchor robots
    robot_drone = enum.auto()     # only receiving measurements
    anchor_drone = enum.auto()      # in place, acts as an anchor point

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
        self.velHist = []
        self.attHist = []
        self.inputHistory = []
        self.angleVelHistory = []
        self.motForcesHistory = []
        self.estPosHistory = []
        self.estVelHistory = []
        self.estAttHistory = []
        self.times = []

        return

    def get_pos_hist(self):
        return np.array([row.to_list() for row in self.posHist])

    def get_vel_hist(self):
        return np.array([row.to_list() for row in self.velHist])

    def get_att_hist(self):
        return np.array([row.to_euler_YPR() for row in self.attHist])
    def get_est_pos_hist(self):
        return np.array([row.to_list() for row in self.estPosHistory])
    def get_est_vel_hist(self):
        return np.array([row.to_list() for row in self.estVelHistory])
    def get_est_att_hist(self): 
        return np.array([row.to_euler_YPR() for row in self.estAttHistory])
    def get_input_history(self):
        return np.array(self.inputHistory)
    def get_angle_vel_history(self):
        return np.array([row.to_list() for row in self.angleVelHistory])

    def get_mot_force_history(self):
        return np.array(self.motForcesHistory)
    def get_times(self):
        return np.array(self.times)

    def add_motor(self, motorPosition, spinDir, minSpeed, maxSpeed, speedSqrToThrust, speedSqrToTorque, timeConst, inertia):
        self._motors.append(Motor(motorPosition, spinDir, minSpeed, maxSpeed, speedSqrToThrust, speedSqrToTorque, timeConst, inertia))
        return


    def run(self, dt, motorThrustCommands, t):

        if self.drone_type == DroneType.anchor_drone:
            self.run_anchor()
        else:
            self.run_robot(dt, motorThrustCommands)

        #store history
        self.angleVelHistory.append(self._omega)
        self.estPosHistory.append(self._estimator._state_p.pos)
        self.estVelHistory.append(self._estimator._state_p.vel)
        self.estAttHistory.append(self._estimator._state_p.att)
        self.inputHistory.append(motorThrustCommands)
        self.motForcesHistory.append(self.get_motor_forces())
        self.times.append(t)
    def run_anchor(self):

        #robot is stationary since it is an anchor
        self._pos = self._pos
        self._vel = Vec3(0,0,0)
        self._acc = Vec3(0,0,0)

        #record state
        self.posHist.append(self._pos)
        self.velHist.append(self._vel)
        self.attHist.append(self._att)   

    def run_robot(self, dt, motorThrustCommands):

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

class Vehicle2D:
    def __init__(self, drone_type=DroneType.robot_drone):
        self.drone_type = drone_type

        # TODO: Fix this and the names
        self._pos = [5, 5]
        self._vel = [0, 0]
        self._att = 0

        self._accel = [0, 0]
        self._omega = 0

        self.posHist = []
        self.velHist = []
        self.attHist = []

        self._estimator =  Estimator3Dof(self.Vec3(self._pos[0], self._pos[1], 0), Vec3(self._vel[0], self._vel[1], 0), Rotation.from_euler_YPR([self._att, 0, 0]), drone_type)

        self._acc = [.1, 0] #body axis acceleration including gravity acc
        self._omega = 2 #body axis angular vel

        #Keep IMUs in 3D then just grab 2D measurements?
        #imu measurements
        self._accImu = self._acc
        self._omegaImu = self._omega
        self._magImu = self._att

        #imu object
        accStd = [0.01,0.01]
        gyroStd = 0.01
        magStd = 0.05
        testImu = False
        self._imu = IMU2D(accStd,gyroStd,magStd,testImu)

        # Should run be used as an overarching function with three changing functions or the previous implementation?

    def run(self, dt):

        self.posHist.append(self._pos)
        self.velHist.append(self._vel)
        self.attHist.append(self._att)

        acc = self._accImu
        omega = self._omegaImu
        att = self._magImu
        self._accImu, self._omegaImu, self._magImu = self._imu.get_imu_measurements(acc = acc, att = att, omega = omega)

        if self.drone_type == DroneType.lighthouse_drone:
            self.run_lighthouse(dt)
        elif self.drone_type == DroneType.robot_drone:
            self.run_robot(dt)
        elif self.drone_type == DroneType.anchor_drone: # I know that you can use else for this but I'm doing this for code clarity
            self.run_anchor(dt)

    def run_lighthouse(self, dt):
        self._pos, self._vel, self._att = self.step_dynamics(self._pos, self._vel, self._att, self._acc, self._omega, dt)

    def run_robot(self, dt):
        self._pos, self._vel, self._att = self.step_dynamics(self._pos, self._vel, self._att, self._acc, self._omega, dt)

    def run_anchor(self, dt):
        print("No movement.")

    # Use States?
    # where do i get ax, ay, and omega
    # Should I move this to a different method?
    # Should I save the states in this method?
    def step_dynamics(self, pos, vel, att, acc, omega, dt):
        x = pos[0] + vel[0] * dt
        y = pos[1] + vel[1] * dt
        theta = (att + dt * omega + 3.14159) % (2 * 3.14159) - 3.14159
        vx = vel[0] + (math.cos(att) * acc[0] - math.sin(att) * acc[1]) * dt
        vy = vel[1] + (math.sin(att) * acc[0] + math.cos(att) * acc[1]) * dt
        return [x, y], [vx, vy], theta



