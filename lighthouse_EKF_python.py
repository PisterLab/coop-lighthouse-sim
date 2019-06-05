import numpy as np
import math
import matplotlib.pyplot as plt

# TODO: Make a Lighthouse Robot class
# TODO: Make a Drone class

# TODO: Figure out what actually needs to be instance vs. static vs. temp
# TODO: In other words, what is constant between the drones? What isn't? Out of everything, what needs to be recorded and kept?

# TODO: Figure out what to do with StateTruth class


class Drone:
    def __init__(self):
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
        self.state_truth_arr = [StateTruth(5, 5, 0, 0, 0)]      # TODO: Change this to something you can initialize
        self.state_truth_vec = self.state_truth_arr[0].vectorize()[:, None]

        # initial measured starting position
        # initial measurement is going to be the same as state truth
        self.xm_vec = np.copy(self.state_truth_vec)
        self.xm_obj = [self.state_truth_arr[0]]

        self.sig_x0 = 0.05  # initial uncertainty of x      # TODO: Figure out if this is drone specific or overall
        self.sig_y0 = 0.05  # initial uncertainty of y
        self.sig_th0 = 0.01  # initial uncertainty of z
        self.sig_vx0 = 0.001  # initial uncertainty of vx
        self.sig_vy0 = 0.001  # initial uncertainty of vy

        # covariance of measurements
        self.Pm = [np.diag([self.sig_x0, self.sig_y0, self.sig_th0, self.sig_vx0, self.sig_vy0])]

        self.lighthouse_available = False
        self.anchor_counter = 0
        self.anchor_record = [[0, 0, 0, 0]]
        self.Pp = [np.zeros((5, 5))]
        self.sim_loop()

    def sim_loop(self):
        for k in range(1, timesteps):
            # step true state and save its vector
            self.state_truth_arr.append(StateTruth.step_dynamics_ekf(self.state_truth_arr[k - 1], self.omega[k - 1],
                                                                     self.ax[k - 1], self.ay[k - 1], dt))
            self.state_truth_vec = np.hstack((self.state_truth_vec,
                                              StateTruth.vectorize(self.state_truth_arr[k])[:, None]))

            # corrupt IMU inputs with noise (aka sensor measurements will have some noise)
            self.omega_m.append(self.omega[k - 1] + np.random.randn() * omega_n)
            self.ax_m.append(self.ax[k - 1] + np.random.randn() * ax_n)
            self.ay_m.append(self.ay[k - 1] + np.random.randn() * ay_n)

            # Prior update/Prediction step

            # Calculate Xp using previous measured x (aka xm)
            self.xp_obj = StateTruth.step_dynamics_ekf(self.xm_obj[k - 1], self.omega_m[k - 1], self.ax_m[k - 1],
                                                       self.ay_m[k - 1], dt)
            self.xp_vec = StateTruth.vectorize(self.xp_obj)

            self.sig_l = 0.0001     # TODO: Figure out whether this is actually instance variable

            # Calculate A(k-1)
            self.A = [[1, 0, 0, dt, 0],
                 [0, 1, 0, 0, dt],
                 [0, 0, 1, 0, 0],
                 [0, 0, (-math.sin(self.xm_obj[k - 1].theta) * self.ax_m[k - 1] - math.cos(self.xm_obj[k - 1].theta) *
                         self.ay_m[k - 1]) * dt, 1, 0],
                 [0, 0,(math.cos(self.xm_obj[k - 1].theta) * self.ax_m[k - 1] - math.sin(self.xm_obj[k - 1].theta) *
                        self.ay_m[k - 1]) * dt, 0, 1]]

            # linearized prediction for debugging
            # x_lin = A * xm_vec(:,k-1) + xm_vec(:,k-1)

            # Calculate L(k-1)
            self.L = [[dt, 0, 0, 0, 0],
                 [0, dt, 0, 0, 0],
                 [0, 0, dt, 0, 0],
                 [0, 0, 0, -dt * math.sin(self.xm_obj[k - 1].theta), -dt * math.cos(self.xm_obj[k - 1].theta)],
                 [0, 0, 0, dt * math.cos(self.xm_obj[k - 1].theta), -dt * math.sin(self.xm_obj[k - 1].theta)]]

            # calculate Pp(k)
            self.Pp.append(np.dot(np.dot(self.A, self.Pm[k - 1]), np.transpose(self.A)) + np.dot(np.dot(self.L, Q),
                                                                                                 np.transpose(self.L)))
            # decide whether the beacon robot is crossing a lighthouse
            self.lighthouse_available, self.phi, self.anchor_record = StateTruth.compute_anchor_meas(
                X_a, self.state_truth_arr[k], self.state_truth_arr[k - 1], self.anchor_record, self.xp_obj)

            # decide if lighthouse measurement is available
            # if mod(k*dt, lighthouse_dt)==0
            if self.lighthouse_available:
                # lighthouse measurement is available

                # choose anchor
                # currently useless, I guess we can just keep track of which anchor?
                self.anchor_counter = (self.anchor_counter + 1) % n_anchors

                self.x_a = self.anchor_record[len(self.anchor_record) - 1][2]
                self.y_a = self.anchor_record[len(self.anchor_record) - 1][3]

                # calculate noise corrupted measurement
                self.c_noise = np.random.randn() * compass_n  # compass noise
                self.l_noise = np.random.randn() * math.sqrt(self.Pp[k - 1][2][2])  # lighthouse noise
                self.sig_l = math.sqrt(self.Pp[k - 1][2][2])
                # l_noise = randn * compass_n
                # l_noise = xp_obj.theta - phi     # this is the actual error of lighthouse i believe

                self.z = np.array([[((self.phi + self.l_noise + PI) % (2 * PI)) - PI],
                                   [((self.state_truth_arr[k].theta + self.c_noise + PI) % (2 * PI)) - PI]])

                # calculate H
                # TODO: Switch to transposing function
                self.r = np.linalg.norm(np.array([self.xp_vec[0:2]]).T - [[self.x_a], [self.y_a]])
                self.angle = np.arctan2(self.xp_obj.y - self.y_a, self.xp_obj.x - self.x_a)
                self.H = np.array([[-np.sin(self.angle) / self.r, np.cos(self.angle) / self.r, 0, 0, 0],
                                   [0, 0, 1, 0, 0]])

                # calculate M. I could potentially add two more entries: var(x_a), var(y_a)
                self.M = [[1, 0], [0, 1]]

                # calculate zhat
                self.zhat = np.array([[self.angle], [self.xp_obj.theta]])

                # calculate R(noise covariance matrix)
                self.compass_w = 0.001 * 0.001
                self.R = [[self.sig_l * self.sig_l, 0], [0, self.compass_w]]

                # Kalman gain
                # TODO: Switch to transposing function
                self.K = np.dot(np.dot(self.Pp[k], self.H.T), np.linalg.inv(
                    np.dot(np.dot(self.H, self.Pp[k]), self.H.T) + np.dot(np.dot(self.M, self.R), np.array(self.M).T)))
            else:
                # calc noise corrupted measurement
                self.c_noise = np.random.rand() * compass_n

                self.z = [((self.state_truth_arr[k].theta + self.c_noise + PI) % (2 * PI)) - PI]

                # calc zhat
                self.zhat = [self.xp_obj.theta]

                # calculate H
                self.H = np.array([[0, 0, 1, 0, 0]])

                # calc M
                self.M = 1

                # calc R
                self.compass_w = 0.001 * 0.001
                self.R = self.compass_w

                # Kalman gain
                # TODO: Switch to transposing function
                self.K = np.dot(np.dot(self.Pp[k], self.H.T),
                           np.linalg.inv([np.dot(np.dot(self.H, self.Pp[k])[0], self.H.T) + self.M * self.R * self.M]))

            # xm
            # calculate measurement diff in order to wrap angles
            self.diff = []

            if self.lighthouse_available:
                self.diff.append(self.z[0] - self.zhat[0])
                self.diff.append(self.z[1] - self.zhat[1])

                self.diff[0] = ((self.diff[0] + PI) % (2 * PI)) - PI
                self.diff[1] = ((self.diff[1] + PI) % (2 * PI)) - PI
            else:
                self.diff.append(self.z[0] - self.zhat[0])
                self.diff[0] = ((self.diff[0] + PI) % (2 * PI)) - PI

            # TODO: Switch to transposing function
            self.xp_vec_trans = np.array([self.xp_vec[:]]).T

            self.k_diff_trans = np.dot(self.K, self.diff)

            # TODO: Switch to transposing function
            if len(np.shape(self.k_diff_trans)) == 1:
                self.k_diff_trans = np.array([self.k_diff_trans]).T
            self.add_xm_vec = np.array(self.xp_vec_trans + self.k_diff_trans)

            self.xm_vec = np.hstack((self.xm_vec, self.add_xm_vec))
            self.xm_obj.append(StateTruth.devectorize(self.xm_vec[:, k]))

            # Pm
            if len(np.shape(self.H)) == 1:
                self.H = np.array([self.H])

            self.Pm.append(np.dot(np.eye(5) - np.dot(self.K, self.H), self.Pp[k]))


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

    # calculates anchor measurements based on the true theta state of the
    # robot. X_a is the set of anchor point locations.
    def compute_anchor_meas(X_a, state_truth, state_truth_prev, meas_record, state_estimate):

        MATCH_THRESH = 0
        PI = 3.1415927
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
        phi_robot_k = (state_truth.theta + 2 * PI) % (2 * PI) - PI
        phi_robot_vec_k = np.tile(phi_robot_k, (num_anchors, 1))       # stacked vector of robot orientation

        phi_robot_prev = (state_truth_prev.theta + 2 * PI) % (2 * PI) - PI
        phi_robot_vec_prev = np.tile(phi_robot_prev, (num_anchors, 1))     # stacked vector of robot orientation

        # TODO: Switch to transposing function
        if len(np.shape(phis_k)) == 1:
            phis_k = np.array([phis_k]).T
            phis_prev = np.array([phis_prev]).T

        phi_product = np.multiply((phis_k - phi_robot_vec_k + PI) % (2 * PI) - PI,
                                  (phis_prev - phi_robot_vec_prev + PI) % (2 * PI) - PI)
        match_idx = phi_product <= MATCH_THRESH
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
                                match_locs[0][0], match_locs[0][1]])       # store measurement vector
            phi_final = phi_matches[0]
        return lighthouse, phi_final, meas_record


timesteps = 5000
lighthouse_dt = .3      # UNUSED
dt = 0.01
P_l = np.array([[1, 0, 0],      # UNUSED
                [0, 1, 0],
                [0, 0, 1]])   # covariance of lighthouse states
x_l_0 = 1   # UNUSED
y_l_0 = 1   # UNUSED
PI = 3.1415927

iterations = 1000
plot_run = True

# anchor locations
n_anchors = 3
area_size = 10      # length of one side of potential experiment area in meters
X_a = np.random.rand(n_anchors, 2) * area_size      # matrix of anchors in 2D
# X_a = np.array([[0.5773, 0.31849]])
# X_a(1,:) = [0.234076005284792,6.43920174599930]
# X_a = np.array([[0.5773, 0.31849], [0.234076005284792, 6.43920174599930], [4, 5]])

# noise standard deviations     # TODO: Figure out if this is drone specific or overall
ax_n = 0.08
ay_n = 0.08
omega_n = 0.01
light_n = 0.04
compass_n = 0.001

# Don't quite understand why this is * 10 instead of just inputting the number desired
g_x = 0.001 * 10        # standard deviations       # TODO: Figure out if this is drone specific or overall
g_y = 0.001 * 10
g_w = 0.001 * 10
g_ax = 0.08
g_ay = 0.08

g = [g_x, g_y, g_w, g_ax, g_ay]
Q = np.diag(np.multiply(g, g))


meas_diff = [0]     # UNUSED
light_noise = [0]   # UNUSED

d = Drone()

print("Debugging statement")


if plot_run:
    plt.figure(1)

    plt.plot(d.state_truth_vec[0, :], d.state_truth_vec[1, :])
    plt.scatter(X_a[:, 0], X_a[:, 1])
    plt.plot(d.xm_vec[0, :], d.xm_vec[1, :])
    # n_measures = size(anchor_record);
    for i in range(1, len(d.anchor_record)):
        x = d.anchor_record[i][0]
        y = d.anchor_record[i][1]
        x_a = d.anchor_record[i][2]
        y_a = d.anchor_record[i][3]
        delta = np.array([x_a, y_a]) - np.array([x, y])

        # TODO: Figure out dashed lines
        plt.quiver(x, y, delta[0], delta[1], angles='xy', scale_units='xy', scale=1, width=0.001,
                   linestyle='dashed', color=['r', 'b', 'y', 'g', 'm'][i % 5])

    plt.figure(2)

    # pm plots
    Pm = np.array(d.Pm)
    Pp = np.array(d.Pp)

    plt.subplot(2, 3, 1)
    plt.semilogy(Pm[:, 2, 2])
    plt.subplot(2, 3, 2)
    plt.semilogy(Pp[:, 2, 2])

    plt.subplot(2, 3, 3)
    plt.semilogy(Pm[:, 0, 0])
    plt.subplot(2, 3, 4)
    plt.semilogy(Pm[:, 1, 1])
    plt.subplot(2, 3, 5)
    plt.semilogy(Pm[:, 3, 3])
    plt.subplot(2, 3, 6)
    plt.semilogy(Pm[:, 4, 4])

    plt.figure(3)

    # state plots
    plt.subplot(2, 3, 1)
    plt.plot(d.t, d.state_truth_vec[0, :], d.t, d.xm_vec[0, :])

    plt.subplot(2, 3, 2)
    plt.plot(d.t / dt, d.state_truth_vec[1, :], d.t / dt, d.xm_vec[1, :])

    plt.subplot(2, 3, 3)
    plt.plot(d.t, d.state_truth_vec[2, :], d.t, d.xm_vec[2, :])

    plt.subplot(2, 3, 4)
    plt.plot(d.t, d.state_truth_vec[3, :], d.t, d.xm_vec[3, :])

    plt.subplot(2, 3, 5)
    plt.plot(d.t, d.state_truth_vec[4, :], d.t, d.xm_vec[4, :])

    plt.show()
