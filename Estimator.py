#(c) 2019 Brian Kilberg
from __future__ import division, print_function
import numpy as np
from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
import copy

class Estimator3Dof:
	def __init__(self,mass, inertiaMatrix, omegaSqrToDragTorque, disturbanceTorqueStdDev):
		
		self._mass = mass
		self._inertia = inertiaMatrix
		self._omegaSqrToDragTorque = omegaSqrToDragTorque
		self._disturbanceTorqueStdDev = disturbanceTorqueStdDev


class Estimator6Dof:

	#maps state to A matrix
	stateDict = {"x": 0,
			"y": 1,
			"z": 2,
			"vx": 3,
			"vy": 4,
			"vz": 5,
			"d0": 6,
			"d1": 7,
			"d2": 8}

	def __init__(self,mass, inertiaMatrix,omegaSqrToDragTorque, disturbanceTorqueStdDev,pos = Vec3(0,0,0), vel = Vec3(0,0,0), att=Rotation.identity(),d = Vec3(0,0,0)):

		self._mass = mass
		self._inertia = inertiaMatrix
		self._omegaSqrToDragTorque = omegaSqrToDragTorque
		self._disturbanceTorqueStdDev = disturbanceTorqueStdDev
		self._state_m= State(pos,vel,d,att)
		self._state_p = State(pos,vel,d,att)
		self._Pp = None
		self._Pm = None
		self._stateHistP = []
		#state vars

	def kalmanPredict(self, accImu, gyroImu, dt):

		#create linearized A matrix

		A = np.zeros((9,9))

		A_pos = np.eye(3)
		A_d = np.eye(3)
			
		#position to velocity model
		A_posvel = self._state_m.att.to_rotation_matrix()*dt
		#print(self._state_m.att.to_rotation_matrix)
		#print(A_posvel)

		#position to attitude model
		d0 = Vec3(1,0,0).to_cross_product_matrix()
		d1 = Vec3(0,1,0).to_cross_product_matrix()
		d2 = Vec3(0,0,1).to_cross_product_matrix()

		#dynamics are R*(I + |_del_|) * vel
		grad_d0 = self._state_m.att.to_rotation_matrix()*d0*self._state_m.vel.to_array() * dt 
		grad_d1 = self._state_m.att.to_rotation_matrix()*d1*self._state_m.vel.to_array() * dt 
		grad_d2 = self._state_m.att.to_rotation_matrix()*d2*self._state_m.vel.to_array() * dt


		A_pos_att = np.column_stack((grad_d0,grad_d1,grad_d2))

		#velocity to velocity = Rot(omega)_T
		A_velvel = gyroImu.to_cross_product_matrix().T

		#velocity from attitude error = -g*(-[del x])*R_T * e3
		grad_d0 = 9.81*d0*self._state_m.att.to_rotation_matrix().T*Vec3(0,0,1).to_array() * dt
		grad_d1 = 9.81*d1*self._state_m.att.to_rotation_matrix().T*Vec3(0,0,1).to_array() * dt 
		grad_d2 = 9.81*d2*self._state_m.att.to_rotation_matrix().T*Vec3(0,0,1).to_array() * dt 
		A_velatt = np.column_stack((grad_d0,grad_d1,grad_d2))

		#att error from att error. this is from the covariance update paper
		gyroCrossRod = gyroImu.to_cross_product_matrix()/2 * dt
		mat = np.eye(3)+gyroCrossRod.T + gyroCrossRod.T*gyroCrossRod.T/2

		A_attatt = mat

		#create A matrix 
		A = np.block([[A_pos, A_posvel, A_pos_att],
					  [np.zeros((3,3)), A_velvel, A_velatt],
					  [np.zeros((3,3)), np.zeros((3,3)), A_attatt]])
		#TODO: covariance update

		#position prediction
		dposBody = self._state_m.vel * dt + Vec3(0,0,accImu[2]*dt*dt / 2.0) 
		#self._state_p.pos = self._state_m.att * dposBody - Vec3(0,0,9.81*dt*dt / 2.0)
		self._state_p.pos = self._state_m.pos + self._state_m.att * dposBody - Vec3(0,0,9.81*dt*dt / 2.0)
		
		#velocity prediction: zacc - gyro x vel - g_body 
		gyroCross = gyroImu.to_cross_product_matrix()
		print("velocity cross")
		print(self._state_m.att.inverse() * Vec3(0,0,9.81))
		print(Vec3(0,0,accImu[2])-gyroCross*self._state_m.vel - self._state_m.att.inverse() * Vec3(0,0,9.81))
		print(self._state_m.vel)
		print(gyroCross*self._state_m.vel)
		self._state_p.vel +=  dt*(Vec3(0,0,accImu[2])-gyroCross*self._state_m.vel - self._state_m.att.inverse() * Vec3(0,0,9.81))

		#attitude update
		gyroRot = Rotation.from_rotation_vector(gyroImu*dt)
		self._state_p.att = self._state_m.att*gyroRot
		self._stateHistP.append(self._state_p)

	def kalmanUpdate(self, dt):
		self._state_m = copy.deepcopy(self._state_p)


class State:
	def __init__(self, pos, vel, d, att):
		self.pos = pos #position as Vec3
		self.vel = vel #velocity as Vec3
		self.d = d #attitude error as Vec3 (rodrigues params)
		self.att = att #reference attitude

	def getVector(self):
		return np.array([self.pos[0], self.pos[1], self.pos[2], self.vel[0], self.vel[1], self.vel[2], self.d[0], self.d[1], self.d[2]])

	def devectorize(state_vector,att=None):

		self.pos = Vec3(state_vector[0], state_vector[1], state_vector[2])
		self.vel = Vec3(state_vector[3], state_vector[4],state_vector[5])
		self.d = Vec3(state_vector[6], state_vector[7], state_vector[8])

		if att != None:
			self.att = att



