#(c) 2019 Brian Kilberg
from __future__ import division, print_function
import numpy as np
from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
import copy
from vehicle import Vehicle, DroneType

class MeasurementHandler:
	def __init__(self, vehicles):
		self._vehicles = vehicles

		self._robot_drones = [drone for drone in self._vehicles if drone.drone_type == DroneType.robot_drone]
		self._lighthouse_drones = [drone for drone in self._vehicles if drone.drone_type == DroneType.lighthouse_drone]
		self._anchor_drones = [drone for drone in self._vehicles if drone.drone_type == DroneType.anchor_drone]
	
		print([drone.drone_type for drone in self._vehicles])
	
		#global list of current anchor, robot, and robot lighthouse locations
		# self._currRobotLocations = []
		# self._currLighthouseLocations =[]
		# self._currAnchorLocations = []

		#global list of previous robot and lighthouse locations 
		# self._prevRobotLocations = []
		# self._prevLighthouseLocations = []

	# def computeGlobalLighthouseMeas(self,quadList):

	# 	#shift current states to previous states
	# 	self._prevRobotLocations = copy.deepcopy(self._currRobotLocations)
	# 	self._prevLighthouseLocations = copy.deepcopy(self._currLighthouseLocations)

	# 	#update current states from array of robots passed in 
	# 	self._currLighthouseLocations = []

	def get_measurement(self, drone):
		if drone.drone_type == DroneType.lighthouse_drone or drone.drone_type == DroneType.robot_drone:
			measurement_avail, phi_final, measurer_pos_p = self.compute_anchor_meas(drone)



		elif drone.drone_type == DroneType.anchor_drone:
			measurement_avail, phi_final, measurer_pos_p = self.compute_lighthouse_meas(drone)

		if measurement_avail:
			#add noise
			noise = np.random.randn() * np.sqrt(drone._estimator._Pp[6,6])  # lighthouse noise
			z = ((phi_final + noise + np.pi) % (2 * np.pi)) - np.pi
			return (z, phi_final,measurer_pos_p, drone._att.to_euler_YPR(),drone._pos)
		else:
			return None
		
	def compute_lighthouse_meas(self, unknown_anchor):


		x_column = np.array([l._pos[0] for l in self._lighthouse_drones])[:,None]
		y_column = np.array([l._pos[1] for l in self._lighthouse_drones])[:,None]

		if len(self._lighthouse_drones[0].posHist) < 2:
			return False,0,0
		x_column_prev = np.array([l.posHist[-2][0] for l in self._lighthouse_drones])[:,None]
		y_column_prev = np.array([l.posHist[-2][1] for l in self._lighthouse_drones])[:,None]

		# calculate headings from all lighthouses to unknown anchor
		# try switching y_column and state truth to fix bug
		phis_k = np.arctan2(unknown_anchor._pos[1] - y_column, unknown_anchor._pos[0] - x_column)

		# calculate headings from all lighthouses to unknown anchor from previous state
		phis_prev = np.arctan2(unknown_anchor.posHist[-2][1] - y_column_prev , unknown_anchor.posHist[-2][0] - x_column_prev)

		# calc anchor distances from robot
		# This is unused so I don't know why it's here
		# ds = np.linalg.norm(X_a - np.tile([state_truth.x,state_truth.y], (num_anchors, 1)), 2, 1)

		# find a phi that matches current
		phi_robot_vec_k = np.array([(l._att.to_euler_YPR()[0] + np.pi) % (2 * np.pi) - np.pi for l in self._lighthouse_drones])[:,None]

		phi_robot_vec_prev = np.array([(l.attHist[-2].to_euler_YPR()[0]  + np.pi) % (2 * np.pi) - np.pi for l in self._lighthouse_drones])[:,None]

		phi_product = np.multiply((phis_k - phi_robot_vec_k + np.pi) % (2 * np.pi) - np.pi,
		                      (phis_prev - phi_robot_vec_prev + np.pi) % (2 * np.pi) - np.pi)

		match_idx = phi_product <= 0
		# if len(np.shape(match_idx)) == 1:
		#     match_idx = match_idx[:, None]

		# match_idx = abs(phis_k - repmat(phi_robot_k,num_anchors,1)) < MATCH_THRESH
		phi_matches = []
		lighthouse_pos_p = []

		# Add the location of the anchor it matched with
		for i in range(len(match_idx)):
			if match_idx[i][0]:
			    phi_matches.append(phis_k[i][0])
			    lighthouse_pos_p.append([lighthouse_drones[i]._estimator._state_p[0], lighthouse_drones[i]._estimator._state_p[1]])


		# If the robot didn't cross an anchor
		if len(phi_matches) == 0:
			measurement_avail = False
			phi_final = 100

		# If the robot did cross an anchor
		else:
			measurement_avail = True

			phi_final = phi_matches[0]

		return measurement_avail, phi_final, lighthouse_pos_p

	def compute_anchor_meas(self, lighthouse_drone):
	    num_anchors = len(self._anchor_drones)

	    x_column = np.array([a._pos[0] for a in self._anchor_drones])
	    y_column = np.array([a._pos[1] for a in self._anchor_drones])

	    # calculate headings to all anchor points
	    phis_k = np.arctan2(lighthouse_drone._pos[1] - y_column, lighthouse_drone._pos[0] - x_column)

	    # calculate headings to all anchor points from previous state
	    if len(lighthouse_drone.posHist) < 2:
	    	return False,0,0

	    phis_prev = np.arctan2(lighthouse_drone.posHist[-2][1] - y_column, lighthouse_drone.posHist[-2][0] - x_column)

	    # calc anchor distances from robot
	    # This is unused so I don't know why it's here
	    # ds = np.linalg.norm(X_a - np.tile([state_truth.x,state_truth.y], (num_anchors, 1)), 2, 1)

	    # find a phi that matches current
	    phi_robot_k = (lighthouse_drone._att.to_euler_YPR()[0]  + np.pi) % (2 * np.pi) - np.pi
	    phi_robot_vec_k = np.tile(phi_robot_k, (num_anchors, 1))  # stacked vector of robot orientation

	    phi_robot_prev = (lighthouse_drone.attHist[-2].to_euler_YPR()[0]  + np.pi) % (2 * np.pi) - np.pi
	    phi_robot_vec_prev = np.tile(phi_robot_prev, (num_anchors, 1))  # stacked vector of robot orientation

	    if len(np.shape(phis_k)) == 1:
	        phis_k = np.array([phis_k]).T
	        phis_prev = np.array([phis_prev]).T

	    phi_product = np.multiply((phis_k - phi_robot_vec_k + np.pi) % (2 * np.pi) - np.pi,
	                              (phis_prev - phi_robot_vec_prev + np.pi) % (2 * np.pi) - np.pi)
	    match_idx = phi_product <= 0

	    # match_idx = abs(phis_k - repmat(phi_robot_k,num_anchors,1)) < MATCH_THRESH
	    phi_matches = []
	    match_locs = []

	    # Add the location of the anchor it matched with
	    for i in range(len(match_idx)):
	        if match_idx[i][0]:
	            phi_matches.append(phi_robot_k)
	            match_locs.append([x_column[i],y_column[i]])

	    # If the robot didn't cross an anchor
	    if len(phi_matches) == 0:
	        measurement_avail = False
	        phi_final = 100

	    # If the robot did cross an anchor
	    else:
	        measurement_avail = True
	        phi_final = phi_matches[0]


	    return measurement_avail, phi_final, match_locs
