#(c) 2019 Brian Kilberg
from __future__ import division, print_function
import numpy as np
from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
import copy

class MeasurementHandler:
	def __init__(self):

		#gobal list of current anchor, robot, and robot lighthouse locations
		self._currRobotLocations = []
		self._currLighthouseLocations =[]
		self._currAnchorLocations = []

		#global list of previous robot and lighthouse locations 
		self._prevRobotLocations = []
		self._prevLighthouseLocations = []

	def computeGlobalLighthouseMeas(self,quadList):

		#shift current states to previous states
		self._prevRobotLocations = copy.deepcopy(self._currRobotLocations)
		self._prevLighthouseLocations = copy.deepcopy(self._currLighthouseLocations)

		#update current states from array of robots passed in 
		self._currLighthouseLocations = []
		
	def compute_lighthouse_meas(state_truth, state_truth_prev, meas_record, state_estimate):


		x_column = np.array([l.state_truth_arr[-1].x for l in lighthouse_drones])[:,None]
		y_column = np.array([l.state_truth_arr[-1].y for l in lighthouse_drones])[:,None]

		x_column_prev = np.array([l.state_truth_arr[-2].x for l in lighthouse_drones])[:,None]
		y_column_prev = np.array([l.state_truth_arr[-2].y for l in lighthouse_drones])[:,None]

		# calculate headings from all lighthouses to unknown anchor
		# try switching y_column and state truth to fix bug
		phis_k = np.arctan2(state_truth.y - y_column, state_truth.x - x_column)

		# calculate headings from all lighthouses to unknown anchor from previous state
		phis_prev = np.arctan2(state_truth_prev.y - y_column_prev , state_truth_prev.x - x_column_prev)

		# calc anchor distances from robot
		# This is unused so I don't know why it's here
		# ds = np.linalg.norm(X_a - np.tile([state_truth.x,state_truth.y], (num_anchors, 1)), 2, 1)

		# find a phi that matches current
		phi_robot_vec_k = np.array([(l.state_truth_arr[-1].theta + PI) % (2 * PI) - PI for l in lighthouse_drones])[:,None]

		phi_robot_vec_prev = np.array([(l.state_truth_arr[-2].theta + PI) % (2 * PI) - PI for l in lighthouse_drones])[:,None]

		phi_product = np.multiply((phis_k - phi_robot_vec_k + PI) % (2 * PI) - PI,
		                      (phis_prev - phi_robot_vec_prev + PI) % (2 * PI) - PI)

		match_idx = phi_product <= 0
		# if len(np.shape(match_idx)) == 1:
		#     match_idx = match_idx[:, None]

		# match_idx = abs(phis_k - repmat(phi_robot_k,num_anchors,1)) < MATCH_THRESH
		phi_matches = []
		match_locs = []

		# Add the location of the anchor it matched with
		for i in range(len(match_idx)):
			if match_idx[i][0]:
			    phi_matches.append(phis_k[i][0])
			    match_locs.append([lighthouse_drones[i].xp_obj.x, lighthouse_drones[i].xp_obj.y])


		# If the robot didn't cross an anchor
		if len(phi_matches) == 0:
			lighthouse = False
			phi_final = 100

		# If the robot did cross an anchor
		else:
			lighthouse = True

		# Store where we think the robot is and which lighthouse crossed it
		meas_record.append([state_estimate.x, state_estimate.y,
		                    match_locs[0][0], match_locs[0][1]])  # store measurement vector

		# TODO: figure out noise integration
		phi_final = phi_matches[0]

		return lighthouse, phi_final, meas_record