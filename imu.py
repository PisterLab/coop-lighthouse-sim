from __future__ import division, print_function
import numpy as np
from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath

class IMU:

	#earth's magnetic field vector in earth coordinates, assumed to be pointing in y
	_ref_mag = Vec3(0,1,0)

	def __init__(self, acc_std, gyro_std, mag_std):

		#assert acc_std.shape == (3,1); assert gyro_std.shape == (3,1); assert mag_std.shape == (3,1) 
		#print(acc_std)

		self._acc_std = acc_std
		self._gyro_std = gyro_std
		self._mag_std= mag_std

		#print(self._ref_mag)

		
	def get_imu_measurements(self, acc, att, omega): 

		#generate noise based on sensor noise model
		acc_noise = Vec3(np.random.normal()*self._acc_std[0],np.random.normal()*self._acc_std[1],np.random.normal()*self._acc_std[2])
		gyro_noise = Vec3(np.random.normal()*self._gyro_std[0],np.random.normal()*self._gyro_std[1],np.random.normal()*self._gyro_std[2])
		mag_noise = Vec3(np.random.normal()*self._mag_std[0],np.random.normal()*self._mag_std[1],np.random.normal()*self._mag_std[2])

		#add gravity acceleration, which is measured by IMU, and rotate earth frame acceleration to body frame  
		acc = att.inverse()*(acc + Vec3(0,0,-9.81))
		
		#add noise to true state to obtain measurement
		acc += acc_noise
		omega += gyro_noise
		mag = att.inverse()*self._ref_mag + mag_noise

		#return measurement tuples 
		return acc, omega, mag
		
		
