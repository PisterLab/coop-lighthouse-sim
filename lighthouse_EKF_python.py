import numpy as np
import math
import matplotlib.pyplot as plt
import enum

PI = 3.1415927      # constant


class DroneType(enum.Enum):
    lighthouse_robot = enum.auto()      # localizing itself and localizing anchor robots
    measurement_robot = enum.auto()     # only taking measurements
    anchor_robot = enum.auto()      # in place, acts as an anchor point


class Drone:
    def __init__(self, x=5, y=5, theta=0, vx=0, vy=0, drone_type=DroneType.measurement_robot):
        self.drone_type = drone_type
        self.t = np.linspace(0, timesteps * dt, timesteps)

        # initialize real ax, ay, omega
        self.ax = np.full(timesteps, .1)
        self.ax[0] = 1
        self.ay = np.zeros(timesteps)
        self.omega = np.full(timesteps, 2)

        self.omega_m = []
        self.ax_m = []
        self.ay_m = []

        # actual starting position of drone: x, y, theta, vx, vy
        self.state_truth_arr = [StateTruth(x, y, theta, vx, vy)]
        self.state_truth_vec = self.state_truth_arr[0].vectorize()[:, None]

        self.sig_x0 = 0.05  # initial uncertainty of x
        self.sig_y0 = 0.05  # initial uncertainty of y
        self.sig_th0 = 0.01  # initial uncertainty of theta
        self.sig_vx0 = 0.001  # initial uncertainty of vx
        self.sig_vy0 = 0.001  # initial uncertainty of vy

        # covariance of measurements
        self.Pm = [np.diag([self.sig_x0, self.sig_y0, self.sig_th0, self.sig_vx0, self.sig_vy0])]

        # initial measured starting position
        # TODO: Figure out dimensions of xm_vec (whether it should be 2D or 1D)
        self.xm_vec = self.state_truth_vec + np.dot(np.random.rand(1, len(self.state_truth_vec)), self.Pm[0]).T
        self.xm_obj = [StateTruth(self.xm_vec[0][0], self.xm_vec[1][0], self.xm_vec[2][0], self.xm_vec[3][0],
                                  self.xm_vec[4][0])]

        # lighthouse_available = False      # default variable
        self.anchor_counter = 0     # TODO: Is this necessary?
        self.meas_record = [[0, 0, 0, 0]]
        self.Pp = [np.zeros((5, 5))]

        # TODO: these are directly copied from anchor_sim but not sure if this is best place for them
        self.K_rx = self.K_ry = self.K_lx = self.K_ly = [0]
        self.D, self.V = np.linalg.eig(self.Pm[0])
        self.D = np.diag(self.D)[:, :, None]
        self.V = self.V[:, :, None]
        self.r_diffx = []
        self.r_diffy = []
        self.last_direction = np.array([0, 0])[:, None]

        self.error = []

    def default_lighthouse_move(self):
        assert self.drone_type == DroneType.lighthouse_robot

        self.state_truth_arr.append(StateTruth.step_dynamics_ekf(self.state_truth_arr[k - 1], self.omega[k - 1],
                                                                 self.ax[k - 1], self.ay[k - 1], dt))
        self.state_truth_vec = np.hstack((self.state_truth_vec,
                                          StateTruth.vectorize(self.state_truth_arr[k])[:, None]))

        # Prior update/Prediction step

        # Calculate Xp using previous measured x (aka xm)
        xp_obj = StateTruth.step_dynamics_ekf(self.xm_obj[k - 1], self.omega_m[k - 1], self.ax_m[k - 1],
                                              self.ay_m[k - 1], dt)
        xp_vec = StateTruth.vectorize(xp_obj)
        return xp_obj, xp_vec

    def lighthouse_with_anchor_drone_move(self, anchor_drone):
        d = np.linalg.norm(anchor_drone.xm_vec[0:2, k - 1] - self.state_truth_vec[0:2, k - 1])
        angle = np.arctan2(anchor_drone.xm_vec[1, k - 1] - self.state_truth_vec[1, k - 1], anchor_drone.xm_vec[0, k - 1]
                           - self.state_truth_vec[0, k - 1])
        Hp = (1 / d) * np.array([[np.sin(angle), -np.cos(angle)]])
        # -10*(x_m(1,i-1)-(x_l(i-1)))/(log(10)* d), -10*(x_m(2,i-1)-(y_l(i-1)))/(log(10)* d)];
        Rp = np.diag([np.power(sig1, 2)])
        fim = np.matmul(Hp.T / np.linalg.inv(Rp), Hp)
        lam, v = np.linalg.eig(fim)
        lam = np.diag(lam)
        if lam[0, 0] >= lam[1, 1]:
            direction = v[:, 0]
        else:
            direction = v[:, 1]

        # dot = np.matmul(np.transpose(direction), self.last_direction)

        if (np.matmul(np.transpose(direction), self.last_direction)) < 0:
            direction = -direction

        self.last_direction = direction

        # x_l(i) = x_l(i-1)+ u_l(1,max_idx);
        # y_l(i) = y_l(i-1)+ u_l(2,max_idx);
        self.state_truth_arr.append(StateTruth(self.state_truth_vec[0, k - 1] + direction[0],
                                               self.state_truth_vec[1, k - 1] + direction[1]))
        self.state_truth_vec = np.hstack((self.state_truth_vec, StateTruth.vectorize(self.state_truth_arr[k])[:, None]))
        return self.state_truth_arr[k], self.state_truth_vec[:, k]

    def change_to_lighthouse(self):
        self.drone_type = DroneType.lighthouse_robot
        self.Pm[k - 1][2][2] = self.sig_th0
        self.Pm[k - 1][3][3] = self.sig_vx0
        self.Pm[k - 1][4][4] = self.sig_vy0

    def run_lighthouse(self, k):
        if self.drone_type != DroneType.lighthouse_robot:
            self.change_to_lighthouse()

        assert k >= 1

        # corrupt IMU inputs with noise (aka sensor measurements will have some noise)
        self.omega_m.append(self.omega[k - 1] + np.random.randn() * omega_n)
        self.ax_m.append(self.ax[k - 1] + np.random.randn() * ax_n)
        self.ay_m.append(self.ay[k - 1] + np.random.randn() * ay_n)

        # step true state and save its vector
        # TODO: Fill the if part in, replacing the default lighthouse move
        if len(anchor_drones) > 0:
            # anchor = anchor_drones[0]       # TODO: fix this
            # xp_obj, xp_vec = self.lighthouse_with_anchor_drone_move(anchor)
            self.xp_obj, xp_vec = self.default_lighthouse_move()
        else:
            self.xp_obj, xp_vec = self.default_lighthouse_move()

        # sig_l = 0.0001     # Default variable

        # Calculate A(k-1)
        A = [[1, 0, 0, dt, 0],
             [0, 1, 0, 0, dt],
             [0, 0, 1, 0, 0],
             [0, 0, (-math.sin(self.xm_obj[k - 1].theta) * self.ax_m[k - 1] - math.cos(self.xm_obj[k - 1].theta) *
                     self.ay_m[k - 1]) * dt, 1, 0],
             [0, 0, (math.cos(self.xm_obj[k - 1].theta) * self.ax_m[k - 1] - math.sin(self.xm_obj[k - 1].theta) *
                     self.ay_m[k - 1]) * dt, 0, 1]]

        # linearized prediction for debugging
        # x_lin = A * xm_vec(:,k-1) + xm_vec(:,k-1)

        # Calculate L(k-1)
        L = [[dt, 0, 0, 0, 0],
             [0, dt, 0, 0, 0],
             [0, 0, dt, 0, 0],
             [0, 0, 0, -dt * math.sin(self.xm_obj[k - 1].theta), -dt * math.cos(self.xm_obj[k - 1].theta)],
             [0, 0, 0, dt * math.cos(self.xm_obj[k - 1].theta), -dt * math.sin(self.xm_obj[k - 1].theta)]]

        # calculate Pp(k)
        self.Pp.append(np.dot(np.dot(A, self.Pm[k - 1]), np.transpose(A)) + np.dot(np.dot(L, Q),
                                                                                             np.transpose(L)))
        # decide whether the lighthouse robot is crossing an anchor

        lighthouse_available, phi, self.meas_record = compute_anchor_meas(self.state_truth_arr[k],
                                                                            self.state_truth_arr[k - 1],
                                                                            self.meas_record, self.xp_obj)

        # decide if lighthouse measurement is available
        # if mod(k*dt, lighthouse_dt)==0
        if lighthouse_available:
            # lighthouse measurement is available

            # choose anchor
            # currently useless, I guess we can just keep track of which anchor?
            self.anchor_counter = (self.anchor_counter + 1) % n_anchors

            x_a = self.meas_record[len(self.meas_record) - 1][2]
            y_a = self.meas_record[len(self.meas_record) - 1][3]

            # calculate noise corrupted measurement
            c_noise = np.random.randn() * compass_n  # compass noise
            l_noise = np.random.randn() * math.sqrt(self.Pp[k - 1][2][2])  # lighthouse noise
            sig_l = math.sqrt(self.Pp[k - 1][2][2])
            # l_noise = randn * compass_n
            # l_noise = xp_obj.theta - phi     # this is the actual error of lighthouse i believe

            z = np.array([[((phi + l_noise + PI) % (2 * PI)) - PI],
                               [((self.state_truth_arr[k].theta + c_noise + PI) % (2 * PI)) - PI]])

            # calculate H
            # TODO: Switch to transposing function
            r = np.linalg.norm(np.array([xp_vec[0:2]]).T - [[x_a], [y_a]])
            angle = np.arctan2(self.xp_obj.y - y_a, self.xp_obj.x - x_a)
            H_mat = np.array([[-np.sin(angle) / r, np.cos(angle) / r, 0, 0, 0],
                               [0, 0, 1, 0, 0]])

            # calculate M. I could potentially add two more entries: var(x_a), var(y_a)
            M = np.array([[1, 0], [0, 1]])

            # calculate zhat
            zhat = np.array([[angle], [self.xp_obj.theta]])

            # calculate R(noise covariance matrix)
            compass_w = 0.001 * 0.001
            R_mat = [[sig_l * sig_l, 0], [0, compass_w]]

            # Kalman gain
            # TODO: Switch to transposing function
            kalman_gain = np.dot(np.dot(self.Pp[k], H_mat.T), np.linalg.inv(
                np.dot(np.dot(H_mat, self.Pp[k]), H_mat.T) + np.dot(np.dot(M, R_mat), M.T)))
        else:
            # calc noise corrupted measurement
            c_noise = np.random.rand() * compass_n

            z = [((self.state_truth_arr[k].theta + c_noise + PI) % (2 * PI)) - PI]

            # calc zhat
            zhat = [self.xp_obj.theta]

            # calculate H
            H_mat = np.array([[0, 0, 1, 0, 0]])

            # calc M
            M = 1

            # calc R
            compass_w = 0.001 * 0.001
            R_mat = compass_w

            # Kalman gain
            # TODO: Switch to transposing function
            kalman_gain = np.dot(np.dot(self.Pp[k], H_mat.T),
                                 np.linalg.inv([np.dot(np.dot(H_mat, self.Pp[k])[0], H_mat.T) + M * R_mat * M]))

        # xm
        # calculate measurement diff in order to wrap angles
        diff = []

        if lighthouse_available:
            diff.append(z[0] - zhat[0])
            diff.append(z[1] - zhat[1])

            diff[0] = ((diff[0] + PI) % (2 * PI)) - PI
            diff[1] = ((diff[1] + PI) % (2 * PI)) - PI
        else:
            diff.append(z[0] - zhat[0])
            diff[0] = ((diff[0] + PI) % (2 * PI)) - PI

        # TODO: Switch to transposing function
        xp_vec_trans = np.array([xp_vec[:]]).T

        k_diff_trans = np.dot(kalman_gain, diff)

        # TODO: Switch to transposing function
        if len(np.shape(k_diff_trans)) == 1:
            k_diff_trans = np.array([k_diff_trans]).T
        add_xm_vec = np.array(xp_vec_trans + k_diff_trans)

        self.xm_vec = np.hstack((self.xm_vec, add_xm_vec))
        self.xm_obj.append(StateTruth.devectorize(self.xm_vec[:, k]))

        # Pm
        self.Pm.append(np.dot(np.eye(5) - np.dot(kalman_gain, H_mat), self.Pp[k]))

        # calculates anchor measurements based on the true theta state of the
        # robot. X_a is the set of anchor point locations.

    def run_anchor(self, k):
        assert k >= 1
        if self.drone_type != DroneType.anchor_robot:
            self.change_to_anchor()

        self.state_truth_arr.append(self.state_truth_arr[k-1])
        self.state_truth_vec = np.hstack((self.state_truth_vec,
                                          StateTruth.vectorize(self.state_truth_arr[k])[:, None]))

        xp_vec = self.xm_vec[0:2, k-1]
        self.xp_obj = StateTruth.devectorize(np.hstack((xp_vec, [0,0,0])))
        self.Pp.append(self.Pm[k-1])

        lighthouse_available, z, self.meas_record = compute_lighthouse_meas(self.state_truth_arr[k], self.state_truth_arr[k-1], self.meas_record, self.xp_obj)

        if lighthouse_available:
            h = np.arctan2(xp_vec[1] - self.meas_record[-1][3], xp_vec[0] - self.meas_record[-1][2])

            lighthouse_xy = self.meas_record[-1][2:]

            r = np.linalg.norm(xp_vec - lighthouse_xy)
            angle = ((np.arctan2(xp_vec[1] - lighthouse_xy[1], xp_vec[0] - lighthouse_xy[0]) + PI) % (2*PI)) - PI
            H = (1/r) * np.array([-np.sin(angle), np.cos(angle)])
            # H = [-(x_p(2,i)-y_l(i))/norm(x_p(:,i)-X_l(:,i))^2 , (x_p(1,i)-x_l(i))/norm(x_p(:,i)-X_l(:,i))^2;
            #      -10*(x_p(1,i)-x_l(i))/(log(10)* norm(x_p(:,i)-X_l(:,i))^2), -10*(x_p(2,i)-y_l(i))/(log(10)* norm(x_p(:,i)-X_l(:,i))^2)];


            W = np.array([[(xp_vec[1] - lighthouse_xy[1]) / np.power(np.linalg.norm(xp_vec - lighthouse_xy), 2), -(xp_vec[0] - lighthouse_xy[0]) / np.power(np.linalg.norm(xp_vec- lighthouse_xy), 2), 1, 0],
                            [10 * (xp_vec[0] - lighthouse_xy[0]) / (np.log(10) * np.power(np.linalg.norm(xp_vec- lighthouse_xy), 2)), 10*(xp_vec[1]-lighthouse_xy[1]) / (np.log(10) * np.power(np.linalg.norm(xp_vec- lighthouse_xy), 2)), 0 ,1]])

            R = np.array([np.append(P_l[0,:],[0]),
                        np.append(P_l[1,:], [0]),
                        np.append(P_l[2,:], [0]),
                        [0,0,0,sig4**2]])

            K = self.Pp[k][0:2, 0:2] @ H.T @ np.linalg.inv(H @ self.Pp[k][0:2, 0:2] @ H.T + W @ R @ W.T)
            K = K[:,None]

            # is the kalman gain helpful?



            z_h_diff = ((z-h + PI) % (2*PI)) - PI


            new_xm_xy = np.array(xp_vec[:,None] + K * z_h_diff)
            new_xm = np.append(new_xm_xy, np.zeros((3,1)), axis=0)

            self.xm_vec = np.append(self.xm_vec, new_xm, axis=1)
            self.xm_obj.append(StateTruth(self.xm_vec[0,k], self.xm_vec[1,k], self.xm_vec[2,k], self.xm_vec[3,k], self.xm_vec[4,k]))
            
            new_Pm = np.zeros((5,5))
            new_Pm[0:2, 0:2] = np.array((np.identity(2) - K @ np.array([H])) @ self.Pp[k][0:2, 0:2])
            self.Pm.append(new_Pm)

        else:
            
            self.xm_vec = np.append(self.xm_vec, self.xm_vec[:, -1][:,None], axis=1)
            self.xm_obj.append(self.xm_obj[-1])
            self.Pm.append(self.Pm[-1])


        # self.K_rx.append(K[0,1])
        # self.K_ry.append(K[1,1])
        # self.K_lx.append(K[0,0])
        # self.K_ly.append(K[1,0])

        # self.r_diffx.append(K[0,1]*(z[1]-h[1]))
        # self.r_diffy.append(K[1,1]*(z[1]-h[1]))
        # tempD, tempV = np.linalg.eig(self.Pm[k])
        # self.V = np.append(self.V, tempV)
        # self.D = np.append(self.D, np.diag(tempD))

    def change_to_anchor(self):
        self.drone_type = DroneType.anchor_robot
        self.Pm[k - 1][2][2] = 0
        self.Pm[k - 1][3][3] = 0
        self.Pm[k - 1][4][4] = 0
        self.xm_vec[2] = 0
        self.xm_vec[3] = 0
        self.xm_vec[4] = 0
        self.xm_obj[k - 1] = StateTruth(self.xm_vec[0], self.xm_vec[1], self.xm_vec[2], self.xm_vec[3], self.xm_vec[4])

    def measurement_move(self):
        assert self.drone_type == DroneType.measurement_robot

        self.state_truth_arr.append(StateTruth.step_dynamics_ekf(self.state_truth_arr[k - 1], self.omega[k - 1],
                                                                 self.ax[k - 1], self.ay[k - 1], dt))
        self.state_truth_vec = np.hstack((self.state_truth_vec,
                                          StateTruth.vectorize(self.state_truth_arr[k])[:, None]))

        # Prior update/Prediction step

        # Calculate Xp using previous measured x (aka xm)
        xp_obj = StateTruth.step_dynamics_ekf(self.xm_obj[k - 1], self.omega_m[k - 1], self.ax_m[k - 1],
                                              self.ay_m[k - 1], dt)
        xp_vec = StateTruth.vectorize(xp_obj)
        return xp_obj, xp_vec

    def change_to_measurement(self):
        self.drone_type = DroneType.measurement_robot

    def run_measurement(self, k):
        if self.drone_type != DroneType.measurement_robot:
            self.change_to_measurement()

        assert k >= 1
        # corrupt IMU inputs with noise (aka sensor measurements will have some noise)
        self.omega_m.append(self.omega[k - 1] + np.random.randn() * omega_n)
        self.ax_m.append(self.ax[k - 1] + np.random.randn() * ax_n)
        self.ay_m.append(self.ay[k - 1] + np.random.randn() * ay_n)

        # step true state and save its vector
        self.xp_obj, xp_vec = self.measurement_move()

        # sig_l = 0.0001     # Default variable

        # Calculate A(k-1)
        A = [[1, 0, 0, dt, 0],
             [0, 1, 0, 0, dt],
             [0, 0, 1, 0, 0],
             [0, 0, (-math.sin(self.xm_obj[k - 1].theta) * self.ax_m[k - 1] - math.cos(self.xm_obj[k - 1].theta) *
                     self.ay_m[k - 1]) * dt, 1, 0],
             [0, 0, (math.cos(self.xm_obj[k - 1].theta) * self.ax_m[k - 1] - math.sin(self.xm_obj[k - 1].theta) *
                     self.ay_m[k - 1]) * dt, 0, 1]]

        # linearized prediction for debugging
        # x_lin = A * xm_vec(:,k-1) + xm_vec(:,k-1)

        # Calculate L(k-1)
        L = [[dt, 0, 0, 0, 0],
             [0, dt, 0, 0, 0],
             [0, 0, dt, 0, 0],
             [0, 0, 0, -dt * math.sin(self.xm_obj[k - 1].theta), -dt * math.cos(self.xm_obj[k - 1].theta)],
             [0, 0, 0, dt * math.cos(self.xm_obj[k - 1].theta), -dt * math.sin(self.xm_obj[k - 1].theta)]]

        # calculate Pp(k)
        self.Pp.append(np.dot(np.dot(A, self.Pm[k - 1]), np.transpose(A)) + np.dot(np.dot(L, Q),
                                                                                   np.transpose(L)))

        # decide whether the lighthouse robot is crossing an anchor
        lighthouse_available, phi, self.meas_record = compute_lighthouse_meas(self.state_truth_arr[k],
                                                                              self.state_truth_arr[k - 1],
                                                                              self.meas_record, self.xp_obj)
        # decide if lighthouse measurement is available
        # if mod(k*dt, lighthouse_dt)==0
        if lighthouse_available:
            # lighthouse measurement is available
            # choose lighthouse location
            x_a = self.meas_record[len(self.meas_record) - 1][2]
            y_a = self.meas_record[len(self.meas_record) - 1][3]

            # calculate noise corrupted measurement
            c_noise = np.random.randn() * compass_n  # compass noise
            l_noise = np.random.randn() * math.sqrt(self.Pp[k - 1][2][2])  # lighthouse noise
            sig_l = math.sqrt(self.Pp[k - 1][2][2])
            # l_noise = randn * compass_n
            # l_noise = xp_obj.theta - phi     # this is the actual error of lighthouse i believe

            z = np.array([[((phi + l_noise + PI) % (2 * PI)) - PI],
                          [((self.state_truth_arr[k].theta + c_noise + PI) % (2 * PI)) - PI]])
            # calculate H
            # TODO: Switch to transposing function
            r = np.linalg.norm(np.array([xp_vec[0:2]]).T - [[x_a], [y_a]])
            angle = np.arctan2(self.xp_obj.y - y_a, self.xp_obj.x - x_a)

            H_mat = np.array([[-np.sin(angle) / r, np.cos(angle) / r, 0, 0, 0],
                              [0, 0, 1, 0, 0]])

            # calculate M. I could potentially add two more entries: var(x_a), var(y_a)
            M = np.array([[1, 0], [0, 1]])

            # calculate zhat
            zhat = np.array([[angle], [self.xp_obj.theta]])

            # calculate R(noise covariance matrix)
            compass_w = 0.001 * 0.001
            R_mat = [[sig_l * sig_l, 0], [0, compass_w]]

            # Kalman gain
            # TODO: Switch to transposing function
            kalman_gain = np.dot(np.dot(self.Pp[k], H_mat.T), np.linalg.inv(
                np.dot(np.dot(H_mat, self.Pp[k]), H_mat.T) + np.dot(np.dot(M, R_mat), M.T)))

        else:
            # calc noise corrupted measurement
            c_noise = np.random.rand() * compass_n

            z = [((self.state_truth_arr[k].theta + c_noise + PI) % (2 * PI)) - PI]

            # calc zhat
            zhat = [self.xp_obj.theta]

            # calculate H
            H_mat = np.array([[0, 0, 1, 0, 0]])

            # calc M
            M = 1

            # calc R
            compass_w = 0.001 * 0.001
            R_mat = compass_w

            # Kalman gain
            # TODO: Switch to transposing function
            kalman_gain = np.dot(np.dot(self.Pp[k], H_mat.T),
                                 np.linalg.inv([np.dot(np.dot(H_mat, self.Pp[k])[0], H_mat.T) + M * R_mat * M]))

        # xm
        # calculate measurement diff in order to wrap angles
        diff = []

        if lighthouse_available:
            diff.append(z[0] - zhat[0])
            diff.append(z[1] - zhat[1])

            diff[0] = ((diff[0] + PI) % (2 * PI)) - PI
            diff[1] = ((diff[1] + PI) % (2 * PI)) - PI
        else:
            diff.append(z[0] - zhat[0])
            diff[0] = ((diff[0] + PI) % (2 * PI)) - PI

        # TODO: Switch to transposing function
        xp_vec_trans = np.array([xp_vec[:]]).T

        k_diff_trans = np.dot(kalman_gain, diff)

        # TODO: Switch to transposing function
        if len(np.shape(k_diff_trans)) == 1:
            k_diff_trans = np.array([k_diff_trans]).T
        add_xm_vec = np.array(xp_vec_trans + k_diff_trans)

        self.xm_vec = np.hstack((self.xm_vec, add_xm_vec))
        self.xm_obj.append(StateTruth.devectorize(self.xm_vec[:, k]))

        # Pm
        self.Pm.append(np.dot(np.eye(5) - np.dot(kalman_gain, H_mat), self.Pp[k]))

        # calculates anchor measurements based on the true theta state of the
        # robot. X_a is the set of anchor point locations.

# State Truth class
class StateTruth:
    def __init__(self, x=0, y=0, theta=0, vx=0, vy=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.vx = vx
        self.vy = vy

    def vectorize(self):
        return np.array([self.x, self.y, self.theta, self.vx, self.vy])

    def devectorize(state_vector):
        return StateTruth(state_vector[0], state_vector[1], state_vector[2], state_vector[3], state_vector[4])

    def step_dynamics_ekf(state_truth_prev, omega, ax, ay, dt):
        # dynamic evolution function. Used for truth dynamics and for EKF prior
        # updates
        x = state_truth_prev.x + state_truth_prev.vx * dt
        y = state_truth_prev.y + state_truth_prev.vy * dt
        theta = (state_truth_prev.theta + dt * omega + 3.14159) % (2 * 3.14159) - 3.14159
        vx = state_truth_prev.vx + (math.cos(state_truth_prev.theta) * ax - math.sin(state_truth_prev.theta) * ay) * dt
        vy = state_truth_prev.vy + (math.sin(state_truth_prev.theta) * ax + math.cos(state_truth_prev.theta) * ay) * dt
        return StateTruth(x, y, theta, vx, vy)


def compute_anchor_meas(state_truth, state_truth_prev, meas_record, state_estimate):
    num_anchors = len(X_a)

    x_column = X_a[:, 0]
    y_column = X_a[:, 1]

    # calculate headings to all anchor points
    phis_k = np.arctan2(state_truth.y - y_column, state_truth.x - x_column)
    # calculate headings to all anchor points from previous state
    phis_prev = np.arctan2(state_truth_prev.y - y_column, state_truth_prev.x - x_column)

    # calc anchor distances from robot
    # This is unused so I don't know why it's here
    # ds = np.linalg.norm(X_a - np.tile([state_truth.x,state_truth.y], (num_anchors, 1)), 2, 1)

    # find a phi that matches current
    phi_robot_k = (state_truth.theta + PI) % (2 * PI) - PI
    phi_robot_vec_k = np.tile(phi_robot_k, (num_anchors, 1))  # stacked vector of robot orientation

    phi_robot_prev = (state_truth_prev.theta + PI) % (2 * PI) - PI
    phi_robot_vec_prev = np.tile(phi_robot_prev, (num_anchors, 1))  # stacked vector of robot orientation

    # TODO: Switch to transposing function
    if len(np.shape(phis_k)) == 1:
        phis_k = np.array([phis_k]).T
        phis_prev = np.array([phis_prev]).T

    phi_product = np.multiply((phis_k - phi_robot_vec_k + PI) % (2 * PI) - PI,
                              (phis_prev - phi_robot_vec_prev + PI) % (2 * PI) - PI)
    match_idx = phi_product <= 0

    # match_idx = abs(phis_k - repmat(phi_robot_k,num_anchors,1)) < MATCH_THRESH
    phi_matches = []
    match_locs = []

    # Add the location of the anchor it matched with
    for i in range(len(match_idx)):
        if match_idx[i][0]:
            phi_matches.append(phis_k[i][0])
            match_locs.append(X_a[i])

    # If the robot didn't cross an anchor
    if len(phi_matches) == 0:
        lighthouse = False
        phi_final = 100

    # If the robot did cross an anchor
    else:
        lighthouse = True

        # Store where we think the robot is and which anchor it crossed
        meas_record.append([state_estimate.x, state_estimate.y,
                            match_locs[0][0], match_locs[0][1]])  # store measurement vector
        phi_final = phi_matches[0]

    return lighthouse, phi_final, meas_record


def compute_lighthouse_meas(state_truth, state_truth_prev, meas_record, state_estimate):
    num_lighthouses = len(lighthouse_drones)

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
        # Adding PI makes it return the angle from robot to lighthouse instead of vice versa
        phi_final = phi_matches[0]

    return lighthouse, phi_final, meas_record


iterations = 100
errors = []

plot_run = True
for i in range(iterations):

    timesteps = 5000
    lighthouse_dt = .3      # UNUSED
    dt = 0.01
    P_l = np.diag([.05**2, .05**2, (1.5 * 3.1415 / 180)**2])    # covariance of lighthouse states
    x_l_0 = 1   # UNUSED
    y_l_0 = 1   # UNUSED

    # anchor locations
    n_anchors = 3
    area_size = 10      # length of one side of potential experiment area in meters
    X_a = np.random.rand(n_anchors, 2) * area_size      # matrix of anchors in 2D
    # X_a = np.array([[0.5773, 0.31849]])
    # X_a(1,:) = [0.234076005284792,6.43920174599930]
    # X_a = np.array([[0.5773, 0.31849], [0.234076005284792, 6.43920174599930], [4, 5]])

    # noise standard deviations
    ax_n = 0.08
    ay_n = 0.08
    omega_n = 0.01
    light_n = 0.04
    compass_n = 0.001

    # Don't quite understand why this is * 10 instead of just inputting the number desired
    g_x = 0.001 * 10        # standard deviations
    g_y = 0.001 * 10
    g_w = 0.001 * 10
    g_ax = 0.08
    g_ay = 0.08

    sig1 = .05
    sig2 = .05
    sig3 = 1.5 * 3.1415 / 180
    sig4 = 10

    g = [g_x, g_y, g_w, g_ax, g_ay]
    Q = np.diag(np.multiply(g, g))


    meas_diff = [0]     # UNUSED
    light_noise = [0]   # UNUSED

    d1 = Drone()
    d2 = Drone(x=7.5, y=7.5)
    d3 = Drone(np.random.rand() * area_size, np.random.rand() * area_size)
    d4 = Drone(x=7, y=3)

    drones = [d1, d2, d3, d4]
    lighthouse_drones = [d1, d4]
    anchor_drones = [d3]
    measurement_drones = [d2]

    for k in range(1, timesteps):
        for d in lighthouse_drones:
            d.run_lighthouse(k)
        for d in anchor_drones:
            d.run_anchor(k)
        for d in measurement_drones:
            d.run_measurement(k)

    if plot_run:
        drone_errors = []
        for d in drones:
            drone_errors.append(np.linalg.norm(d.state_truth_vec[0:2, -1] - d.xm_vec[0:2, -1]))
            plt.figure(1)
            plt.scatter(X_a[:, 0], X_a[:, 1], color='black')        # anchor points

            plt.scatter(d.state_truth_vec[0, ::100], d.state_truth_vec[1, ::100],   # state truths
                        linewidths=0.001, marker=".", color='m')
            plt.plot(d.state_truth_vec[0, :], d.state_truth_vec[1, :], color='m')

            plt.scatter(d.xm_vec[0, ::100], d.xm_vec[1, ::100],     # measured paths
                        linewidths=0.001, marker=".", color='b')
            plt.plot(d.xm_vec[0, :], d.xm_vec[1, :], color='b')

            plt.scatter(d.state_truth_vec[0, 0], d.state_truth_vec[1, 0], color='g')      # actual start points
            plt.scatter(d.xm_vec[0, -1], d.xm_vec[1, -1], color='r')     # measured endpoints

        errors.append(drone_errors)    
            
        plt.savefig('plots/plot_%s' % i)

if plot_run:
    for j in range(len(drones)):
        error = [drone_error[j] for drone_error in errors]
        plt.figure()
        plt.hist(error, 100)
        plt.title('Error After 5000 Timesteps', fontsize = 20)
        plt.xlabel('L2 Norm Error (m)', fontsize = 16)
        plt.ylabel('Count', fontsize=16)
        plt.savefig('plots/error_%s' % j)
