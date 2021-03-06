# (c) 2019 Mark Mueller

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
from vehicle import Vehicle
from lh_utils import DroneType

from positioncontroller import PositionController
from attitudecontroller import QuadcopterAttitudeControllerNested
from mixer import QuadcopterMixer

np.random.seed(0)

useNestedAttControl = False

#==============================================================================
# Define the simulation
#==============================================================================
dt = 0.01  # s
endTime = 20

#==============================================================================
# Define the vehicle
#==============================================================================
mass = 0.5  # kg
Ixx = 2.7e-3
Iyy = 2.7e-3
Izz = 5.2e-3
Ixy = 0
Ixz = 0
Iyz = 0
omegaSqrToDragTorque = np.matrix(np.diag([0, 0, 0.00014]))  # N.m/(rad/s)**2
armLength = 0.17  # m

##MOTORS##
motSpeedSqrToThrust = 6.4e-6  # propeller coefficient
motSpeedSqrToTorque = 1.1e-7  # propeller coefficient
motInertia   = 15e-6  #inertia of all rotating parts (motor + prop) [kg.m**2]

motTimeConst = 0.015  # time constant with which motor's speed responds [s]
motMinSpeed  = 0  #[rad/s]
motMaxSpeed  = 800  #[rad/s]

#==============================================================================
# Define the disturbance
#==============================================================================
stdDevTorqueDisturbance = 0e-3  # [N.m]

#==============================================================================
# Define the attitude controller
#==============================================================================
#time constants for the angle components:
timeConstAngleRP = 0.1  # [s]
timeConstAngleY  = 1.0  # [s]

#gain from angular velocities
timeConstRatesRP = 0.05  # [s]
timeConstRatesY  = 0.5   # [s]

#==============================================================================
# Define the position controller
#==============================================================================
disablePositionControl = False
posCtrlNatFreq = 1.5  # rad/s
posCtrlDampingRatio = 0.7  # -

#==============================================================================
#Create empty dict for storing quadcopter objects
#==============================================================================
robotDict = {}
nLighthouses = 1
nAnchors = 1
nRobots = 1

#initial locations
lhLocations = [Vec3(0,0,0)]
anchorLocations = [Vec3(0.5,0.5,0)]
robotLocations = [Vec3(0,1,0)]
#==============================================================================
# initialize all quadcopters:
#==============================================================================
for i in range(0,nLighthouses + nAnchors + nRobots):

    #determine quadcopter type
    if nLighthouses > 0:
        droneType = DroneType.lighthouse_drone
        nLighthouses-=1
        initPos = lhLocations[nLighthouses] #traverses the initial pos list backwards

    elif nAnchors > 0:
        droneType = DroneType.anchor_drone
        nAnchors-=1
        initPos = anchorLocations[nLighthouses]

    #execution should never go past this
    elif nRobots > 0:
        droneType = DroneType.robot_drone
        nRobots-=1
        initPos = robotLocations[nLighthouses]

    inertiaMatrix = np.matrix([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    quadrocopter = Vehicle(mass, inertiaMatrix, omegaSqrToDragTorque, stdDevTorqueDisturbance)

    quadrocopter.add_motor(Vec3( armLength, 0, 0), Vec3(0,0,-1), motMinSpeed, motMaxSpeed, motSpeedSqrToThrust, motSpeedSqrToTorque, motTimeConst, motInertia)
    quadrocopter.add_motor(Vec3( 0, armLength, 0), Vec3(0,0, 1), motMinSpeed, motMaxSpeed, motSpeedSqrToThrust, motSpeedSqrToTorque, motTimeConst, motInertia)
    quadrocopter.add_motor(Vec3(-armLength, 0, 0), Vec3(0,0,-1), motMinSpeed, motMaxSpeed, motSpeedSqrToThrust, motSpeedSqrToTorque, motTimeConst, motInertia)
    quadrocopter.add_motor(Vec3( 0,-armLength, 0), Vec3(0,0, 1), motMinSpeed, motMaxSpeed, motSpeedSqrToThrust, motSpeedSqrToTorque, motTimeConst, motInertia)

    posControl = PositionController(posCtrlNatFreq, posCtrlDampingRatio)
    attController = QuadcopterAttitudeControllerNested(timeConstAngleRP, timeConstAngleY, timeConstRatesRP, timeConstRatesY)
    mixer = QuadcopterMixer(mass, inertiaMatrix, armLength, motSpeedSqrToTorque/motSpeedSqrToThrust)

    desPos = Vec3(1, 0, 0)

    quadrocopter.set_position(initPos)
    quadrocopter.set_velocity(Vec3(0, 0, 0))

    quadrocopter.set_attitude(Rotation.identity())


    #start at equilibrium rates:
    quadrocopter._omega = Vec3(0,0,0)

    #add quad to dictionary
    robotDict[i] = quadrocopter

print(robotDict)

#==============================================================================
# Run the simulation
#==============================================================================

numSteps = np.int((endTime)/dt)
index = 0

t = 0

posHistory       = np.zeros([numSteps,3])
velHistory       = np.zeros([numSteps,3])
angVelHistory    = np.zeros([numSteps,3])
attHistory       = np.zeros([numSteps,3])
motForcesHistory = np.zeros([numSteps,quadrocopter.get_num_motors()])
inputHistory     = np.zeros([numSteps,quadrocopter.get_num_motors()])
times            = np.zeros([numSteps,1])

#estimator history
estPosHistory = np.zeros([numSteps,3])
estVelHistory = np.zeros([numSteps,3])
estAttHistory = np.zeros([numSteps,3])

while index < numSteps:

    #traverse dict of robots
    for quadrocopter in robotDict.values():
        #define commands:
        accDes = posControl.get_acceleration_command(desPos, quadrocopter._pos, quadrocopter._vel)
        if disablePositionControl:
            accDes *= 0 #disable position control

        #mass-normalised thrust:
        thrustNormDes = accDes + Vec3(0, 0, 9.81)
        angAccDes = attController.get_angular_acceleration(thrustNormDes, quadrocopter._att, quadrocopter._omega)
        motForceCmds = mixer.get_motor_force_cmd(thrustNormDes, angAccDes)

        #run the simulator
        quadrocopter.run(dt, motForceCmds, t)

        #check for lighthouse measurements that have occured


        #run the estimator
        quadrocopter.kalman_predict(dt)
        quadrocopter.kalman_update(dt)
        #print(quadrocopter._estimator._state_p.pos)
        #for plotting
        times[index] = t
        inputHistory[index,:]     = motForceCmds
        posHistory[index,:]       = quadrocopter._pos.to_list()
        velHistory[index,:]       = quadrocopter._vel.to_list()
        attHistory[index,:]       = quadrocopter._att.to_euler_YPR()
        angVelHistory[index,:]    = quadrocopter._omega.to_list()
        motForcesHistory[index,:] = quadrocopter.get_motor_forces()
        estPosHistory[index, :] = quadrocopter._estimator._state_p.pos.to_list()
        estVelHistory[index, :] = quadrocopter._estimator._state_p.vel.to_list()
        estAttHistory[index, :] =quadrocopter._estimator._state_p.att.to_euler_YPR()
    t += dt
    index += 1

#==============================================================================
# Make the plots
#==============================================================================

fig, ax = plt.subplots(6,1, sharex=True)

#TODO: get histories from vehicle objects and turn this into a function that plots a specific robot
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

#plot trajectories of all robots
fig= plt.figure()
ax = fig.add_subplot(111)
for quad in robotDict.values():
    position = quad.get_pos_hist()
    print("state truth pos:", position)
    print("vel hist:")
    for i in range(0, len(quad.velHist)):
        print(quad.velHist[i].to_list())
    ax.plot(position[:,0], position[:,1])

for quad in robotDict.values():
    fig= plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    position = quad.get_pos_hist()
    ax.plot(position[:,0], position[:,1], position[:,2], color='b')
    est_pos = quad.get_est_pos_hist()
    ax.plot(est_pos[:,0], est_pos[:,1], est_pos[:,2], color='r')

plt.show()
