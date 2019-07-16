from __future__ import division, print_function
import numpy as np
from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath

class IMU:

	#earth's magnetic field vector in earth coordinates, assumed to be pointing in y
	_refMag = Vec3(0,1,0)

	def __init__(self, accStd, gyroStd, magStd):

		#assert accStd.shape == (3,1); assert gyroStd.shape == (3,1); assert magStd.shape == (3,1) 
		#print(accStd)

		self._accStd = accStd
		self._gyroStd = gyroStd
		self._magStd= magStd

		#print(self._ref_mag)

		
	def get_imu_measurements(self, acc, att, omega): 

		#generate noise based on sensor noise model
		accNoise = Vec3(np.random.normal()*self._accStd[0],np.random.normal()*self._accStd[1],np.random.normal()*self._accStd[2])
		gyroNoise = Vec3(np.random.normal()*self._gyroStd[0],np.random.normal()*self._gyroStd[1],np.random.normal()*self._gyroStd[2])
		magNoise = Vec3(np.random.normal()*self._magStd[0],np.random.normal()*self._magStd[1],np.random.normal()*self._magStd[2])
		print("in imu")
		print(acc)
		#add gravity acceleration, which is measured by IMU, and rotate earth frame acceleration to body frame  
		acc = att.inverse()*(acc + Vec3(0,0,-9.81))
		print(acc)
		#add noise to true state to obtain measurement
		acc += accNoise
		omega += gyroNoise
		mag = att.inverse()*self._refMag + magNoise

		#return measurement tuples 
		return acc, omega, mag
		
		
