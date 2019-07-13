#(c) 2019 Brian Kilberg
from __future__ import division, print_function
import numpy as np
from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath

class Estimator3Dof:
	def __init__(self,mass, inertiaMatrix, omegaSqrToDragTorque, disturbanceTorqueStdDev,initialState):
		
		self._mass = mass
		self._inertia = inertiaMatrix
		self._omegaSqrToDragTorque = omegaSqrToDragTorque
        self._disturbanceTorqueStdDev = disturbanceTorqueStdDev
        self._initalState = initalState


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

	def __init__(self,mass, inertiaMatrix,omegaSqrToDragTorque, disturbanceTorqueStdDev,initialPos)

		self._mass = mass
		self._inertia = inertiaMatrix
		self._omegaSqrToDragTorque = omegaSqrToDragTorque
        self._disturbanceTorqueStdDev = disturbanceTorqueStdDev

        #state vars
        self.Pos = initialPos

    def kalmanPredict(self, accImu, gyroImu, dt):

    	#prevState is dictionary of states 
    	pos = prevState["pos"]
    	vel = prevState["vel"]

    	#create linearized A matrix

    	A = np.zeros((9,9))

    	A_pos = np.eye(3)
    	A_d = np.eye(3)
   		
   		A_posvel = att.to_rotation_matrix*dt
   		print(att.to_rotation_matrix)
   		print(A_posvel)

   		A_pos_att = np.asarray([[],
   								[],
   							    []])

class State:
    def __init__(self, pos, vel, d, att):
	    self.pos = pos #position as Vec3
		self.vel = vel #velocity as Vec3
		self.d = d #attitude error as Vec3 (rodrigues params)
		self.att = att #reference attitude

    def vectorize(self):
        return np.array([self.pos[0], self.pos[1], self.pos[2], self.vel[0], self.vel[1], self.vel[2], self.d[0], self.d[1], self.d[2]])

    def devectorize(state_vector,att):
    	self.pos = Vec3(state_vector[0], state_vector[1], state_vector[2])
    	self.vel = Vec3(state_vector[3], state_vector[4],state_vector[5])
    	self.d = Vec3(state_vector[6], state_vector[7], state_vector[8])
    	self.att = att



