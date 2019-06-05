import numpy as np
import math
import matplotlib.pyplot as plt

# TODO: Make a Lighthouse Robot class
# TODO: Make a Drone class


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
lighthouse_dt = .3
dt = 0.01
P_l = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])   # covariance of lighthouse states
x_l_0 = 1
y_l_0 = 1
PI = 3.1415927

iterations = 1000
plot_run = True

# setup input trajectory
t = np.linspace(0, timesteps * dt, timesteps)

"""
ax = np.sin(t)
ax[0] = 1
ax[1:timesteps] = .1 * np.ones(timesteps - 1);
"""

# real ax, ay, omega
ax = np.full(timesteps, .1)
ax[0] = 1
ay = np.zeros(timesteps)
omega = np.full(timesteps, 2)

# measured ax, ay, omega
omega_m = []
ax_m = []
ay_m = []

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

# actual starting position of lighthouse/beacon robot
state_truth_arr = [StateTruth(5, 5, 0, 0, 0)]
state_truth_vec = state_truth_arr[0].vectorize()[:, None]

# initial measured starting position
# initial measurement is going to be the same as state truth
xm_vec = np.copy(state_truth_vec)
xm_obj = [state_truth_arr[0]]

sig_x0 = 0.05       # initial uncertainty of x
sig_y0 = 0.05       # initial uncertainty of y
sig_th0 = 0.01      # initial uncertainty of z
sig_vx0 = 0.001     # initial uncertainty of vx
sig_vy0 = 0.001     # initial uncertainty of vy

# covariance of measurements
Pm = [np.diag([sig_x0, sig_y0, sig_th0, sig_vx0, sig_vy0])]

# Don't quite understand why this is * 10 instead of just inputting the number desired
g_x = 0.001 * 10        # standard deviations
g_y = 0.001 * 10
g_w = 0.001 * 10
g_ax = 0.08
g_ay = 0.08

g = [g_x, g_y, g_w, g_ax, g_ay]
Q = np.diag(np.multiply(g, g))

# measurement w
sig_l = 0.0001

# initialize helper variables
lighthouse_available = False
anchor_counter = 0
Pp = [np.zeros((5, 5))]
meas_diff = [0]
light_noise = [0]
anchor_record = [[0, 0, 0, 0]]      # list of anchor measurements in 2D

# start sim loop with estimation
for k in range(1, timesteps):
    # step true state and save its vector
    state_truth_arr.append(StateTruth.step_dynamics_ekf(state_truth_arr[k - 1], omega[k - 1], ax[k - 1], ay[k - 1], dt))
    state_truth_vec = np.hstack((state_truth_vec, StateTruth.vectorize(state_truth_arr[k])[:, None]))

    # corrupt IMU inputs with noise (aka sensor measurements will have some noise)
    omega_m.append(omega[k - 1] + np.random.randn() * omega_n)
    ax_m.append(ax[k - 1] + np.random.randn() * ax_n)
    ay_m.append(ay[k - 1] + np.random.randn() * ay_n)

    # Prior update/Prediction step

    # Calculate Xp using previous measured x (aka xm)
    xp_obj = StateTruth.step_dynamics_ekf(xm_obj[k - 1], omega_m[k - 1], ax_m[k - 1], ay_m[k - 1], dt)
    xp_vec = StateTruth.vectorize(xp_obj)

    # Calculate A(k-1)
    A = [[1, 0, 0, dt, 0],
         [0, 1, 0, 0, dt],
         [0, 0, 1, 0, 0],
         [0, 0, (-math.sin(xm_obj[k-1].theta) * ax_m[k-1] - math.cos(xm_obj[k-1].theta) * ay_m[k-1])*dt, 1, 0],
         [0, 0, (math.cos(xm_obj[k-1].theta) * ax_m[k-1] - math.sin(xm_obj[k-1].theta) * ay_m[k-1])*dt, 0, 1]]

    # linearized prediction for debugging
    # x_lin = A * xm_vec(:,k-1) + xm_vec(:,k-1)

    # Calculate L(k-1)
    L = [[dt, 0, 0, 0, 0],
         [0, dt, 0, 0, 0],
         [0, 0, dt, 0, 0],
         [0, 0, 0, -dt * math.sin(xm_obj[k-1].theta), -dt * math.cos(xm_obj[k-1].theta)],
         [0, 0, 0, dt * math.cos(xm_obj[k-1].theta), -dt * math.sin(xm_obj[k-1].theta)]]

    # calculate Pp(k)
    Pp.append(np.dot(np.dot(A, Pm[k - 1]), np.transpose(A)) + np.dot(np.dot(L, Q), np.transpose(L)))
    # decide whether the beacon robot is crossing a lighthouse
    lighthouse_available, phi, anchor_record = StateTruth.compute_anchor_meas(X_a, state_truth_arr[k],
                                                                              state_truth_arr[k-1], anchor_record,
                                                                              xp_obj)

    # decide if lighthouse measurement is available
    # if mod(k*dt, lighthouse_dt)==0
    if lighthouse_available:
        # lighthouse measurement is available

        # choose anchor
        # currently useless, I guess we can just keep track of which anchor?
        anchor_counter = (anchor_counter + 1) % n_anchors

        x_a = anchor_record[len(anchor_record) - 1][2]
        y_a = anchor_record[len(anchor_record) - 1][3]

        # calculate noise corrupted measurement
        c_noise = np.random.randn() * compass_n     # compass noise
        l_noise = np.random.randn() * math.sqrt(Pp[k - 1][2][2])        # lighthouse noise
        sig_l = math.sqrt(Pp[k - 1][2][2])
        # l_noise = randn * compass_n
        # l_noise = xp_obj.theta - phi     # this is the actual error of lighthouse i believe

        z = np.array([[((phi + l_noise + PI) % (2 * PI)) - PI],
                      [((state_truth_arr[k].theta + c_noise + PI) % (2 * PI)) - PI]])

        # calculate H
        # TODO: Switch to transposing function
        r = np.linalg.norm(np.array([xp_vec[0:2]]).T - [[x_a], [y_a]])
        angle = np.arctan2(xp_obj.y - y_a, xp_obj.x - x_a)
        H = np.array([[-np.sin(angle) / r, np.cos(angle) / r, 0, 0, 0], [0, 0, 1, 0, 0]])

        # calculate M. I could potentially add two more entries: var(x_a), var(y_a)
        M = [[1, 0], [0, 1]]

        # calculate zhat
        zhat = np.array([[angle], [xp_obj.theta]])

        # calculate R(noise covariance matrix)
        compass_w = 0.001 * 0.001
        R = [[sig_l * sig_l, 0], [0, compass_w]]

        # Kalman gain
        # TODO: Switch to transposing function
        K = np.dot(np.dot(Pp[k], H.T),
                   np.linalg.inv(np.dot(np.dot(H, Pp[k]), H.T) + np.dot(np.dot(M, R), np.array(M).T)))
    else:
        # calc noise corrupted measurement
        c_noise = np.random.rand() * compass_n

        z = [((state_truth_arr[k].theta + c_noise + PI) % (2 * PI)) - PI]

        # calc zhat
        zhat = [xp_obj.theta]

        # calculate H
        H = np.array([[0, 0, 1, 0, 0]])

        # calc M
        M = 1

        # calc R
        compass_w = 0.001 * 0.001
        R = compass_w

        # Kalman gain
        # TODO: Switch to transposing function
        K = np.dot(np.dot(Pp[k], H.T),
                   np.linalg.inv([np.dot(np.dot(H, Pp[k])[0], H.T) + M * R * M]))

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

    k_diff_trans = np.dot(K, diff)

    # TODO: Switch to transposing function
    if len(np.shape(k_diff_trans)) == 1:
        k_diff_trans = np.array([k_diff_trans]).T
    add_xm_vec = np.array(xp_vec_trans + k_diff_trans)

    xm_vec = np.hstack((xm_vec, add_xm_vec))
    xm_obj.append(StateTruth.devectorize(xm_vec[:, k]))

    # Pm
    if len(np.shape(H)) == 1:
        H = np.array([H])

    Pm.append(np.dot(np.eye(5) - np.dot(K, H), Pp[k]))

print("Debugging statement")


if plot_run:
    plt.figure(1)

    plt.plot(state_truth_vec[0, :], state_truth_vec[1, :])
    plt.scatter(X_a[:, 0], X_a[:, 1])
    plt.plot(xm_vec[0, :], xm_vec[1, :])
    # n_measures = size(anchor_record);
    for i in range(1, len(anchor_record)):
        x = anchor_record[i][0]
        y = anchor_record[i][1]
        x_a = anchor_record[i][2]
        y_a = anchor_record[i][3]
        delta = np.array([x_a, y_a]) - np.array([x, y])

        # TODO: Figure out dashed lines
        plt.quiver(x, y, delta[0], delta[1], angles='xy', scale_units='xy', scale=1, width=0.001,
                   linestyle='dashed', color=['r', 'b', 'y', 'g', 'm'][i % 5])

    plt.figure(2)

    # pm plots
    Pm = np.array(Pm)
    Pp = np.array(Pp)

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
    plt.plot(t, state_truth_vec[0, :], t, xm_vec[0, :])

    plt.subplot(2, 3, 2)
    plt.plot(t / dt, state_truth_vec[1, :], t / dt, xm_vec[1, :])

    plt.subplot(2, 3, 3)
    plt.plot(t, state_truth_vec[2, :], t, xm_vec[2, :])

    plt.subplot(2, 3, 4)
    plt.plot(t, state_truth_vec[3, :], t, xm_vec[3, :])

    plt.subplot(2, 3, 5)
    plt.plot(t, state_truth_vec[4, :], t, xm_vec[4, :])

    plt.show()
