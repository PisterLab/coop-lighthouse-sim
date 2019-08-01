# (c) 2019 Mark Mueller

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
from vehicle import Vehicle, 2DVehicle, DroneType

from positioncontroller import PositionController
from attitudecontroller import QuadcopterAttitudeControllerNested
from mixer import QuadcopterMixer
from measurement_handler import MeasurementHandler

np.random.seed(0)


#==============================================================================
# Define the simulation
#==============================================================================
dt = 0.01  # s
endTime = 20

#==============================================================================
# Define the vehicles
#==============================================================================


quadcopter1 = Vehicle2D(DroneType.lighthouse_drone)
quadcopter2 = Vehicle2D(DroneType.robot_drone)
quadcopter3 = Vehicle2D(DroneType.anchor_drone)

vehicle_list = [quadcopter1, quadcopter2, quadcopter3]


#==============================================================================
# Define the measurement handler
#==============================================================================
measurement_handler = MeasurementHandler(vehicle_list)

#==============================================================================
# Run the simulation
#==============================================================================

numSteps = np.int((endTime)/dt)
index = 0
times = np.zeros([numSteps,1])

t = 0

while index < numSteps:
    #define commands:

    #mass-normalised thrust:

    #run the simulator
    for drone in vehicle_list:
        drone.run(dt)

    #######################
    # get measurement here
    #######################

    for drone in vehicle_list:
        measurement = measurement_handler.get_measurement(drone)

    #run the estimator
    quadrocopter.kalman_predict(dt, measurement)
    quadrocopter.kalman_update(dt)
    #print(quadrocopter._estimator._state_p.pos)
    #for plotting
    times[index] = t

    t += dt
    index += 1

#==============================================================================
# Make the plots
#==============================================================================

fig, ax = plt.subplots(6,1, sharex=True)

ax[0].plot(times, posHistory[:,0], label='x')
ax[0].plot(times, posHistory[:,1], label='y')
ax[0].plot(times, posHistory[:,2], label='z')
ax[0].plot([0, endTime], [desPos.to_list(), desPos.to_list()],':')
ax[1].plot(times, velHistory)
ax[2].plot(times, attHistory[:,0]*180/np.pi, label='Y')
ax[2].plot(times, attHistory[:,1]*180/np.pi, label='P')
ax[2].plot(times, attHistory[:,2]*180/np.pi, label='R')
ax[3].plot(times, angVelHistory[:,0], label='p')
ax[3].plot(times, angVelHistory[:,1], label='q')
ax[3].plot(times, angVelHistory[:,2], label='r')
ax[4].plot(times, inputHistory)
ax[5].plot(times, estPosHistory, label = 'est x')
#ax[5].plot(times, velHistory, label = 'x')
#ax[4].plot(times, motForcesHistory,':')

ax[-1].set_xlabel('Time [s]')

ax[0].set_ylabel('Pos')
ax[1].set_ylabel('Vel')
ax[2].set_ylabel('Att [deg]')
ax[3].set_ylabel('AngVel (in B)')
ax[4].set_ylabel('MotForces')

ax[0].set_xlim([0, endTime])
ax[0].legend()
ax[2].legend()
ax[3].legend()

print('Ang vel: ',angVelHistory[-1,:])
print('Motor speeds: ',quadrocopter.get_motor_speeds())

fig, ax = plt.subplots(6,1 ,sharex=True)
ax[0].plot(times,estVelHistory)
ax[1].plot(times,velHistory)
ax[2].plot(times,attHistory)
ax[3].plot(times,estAttHistory)
ax[4].plot(times,estAttHistory-attHistory)
ax[0].set_ylabel('estVel')
ax[1].set_ylabel('Vel')
ax[2].set_ylabel('Att')
ax[3].set_ylabel('EstAtt')


plt.show()
