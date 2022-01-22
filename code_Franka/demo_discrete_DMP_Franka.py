# -*- coding: utf-8 -*-
# 把via的位置设成DMP的轨迹位置，vrep里面再用规划器，把target的目标位置射程via的位置，然后tip跟着target
# 为什么需要via？不直接把target的位置设成DMP的轨迹位置？
# 一次moveToPose以每一时刻的via point位置为目标，到先前设定的via point了就停下，然后再进行下一次via point的规划。所以会有停顿的感觉。
# 一次moveToPose调用21次callback（调用次数是由tip的maxVel，maxAcc控制的，越大调用的次数就越小），每次callback改变的是target的位置，然后tip跟上，tip跟上的过程用maxVel，maxAcc约束了。

# 这里via就是我实物实验要记录的轨迹，我写的力矩控制起到的作用相当于这里target point（也就是callback）的作用。
# 但是我真正要的是关节空间的DMP，所以这个版本的代码还不能直接用，只能练练手。

#%% import package
import time
import numpy as np
import pandas as pd

import sys
sys.path.append('./Franka/VREP_RemoteAPIs')
import sim as vrep_sim

sys.path.append('./Franka')
from FrankaSimModel import FrankaSimModel

sys.path.append('./DMP')
from dmp_discrete import dmp_discrete

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%% program
print ('Program started')

# ------------------------------- Connect to VREP (CoppeliaSim) ------------------------------- 
vrep_sim.simxFinish(-1) # just in case, close all opened connections
while True:
    client_ID = vrep_sim.simxStart('127.0.0.1', 19999, True, False, 5000, 5) # Connect to CoppeliaSim
    if client_ID > -1: # connected
        print('Connect to remote API server.')
        break
    else:
        print('Failed connecting to remote API server! Try it again ...')

# Pause the simulation
# res = vrep_sim.simxPauseSimulation(client_ID, vrep_sim.simx_opmode_blocking)

delta_t = 0.01 # simulation time step
# Set the simulation step size for VREP
vrep_sim.simxSetFloatingParameter(client_ID, vrep_sim.sim_floatparam_simulation_time_step, delta_t, vrep_sim.simx_opmode_oneshot)
# Open synchronous mode
vrep_sim.simxSynchronous(client_ID, True) 
# Start simulation
vrep_sim.simxStartSimulation(client_ID, vrep_sim.simx_opmode_oneshot)

# ------------------------------- Initialize simulation model ------------------------------- 
Franka_sim_model = FrankaSimModel()
Franka_sim_model.initializeSimModel(client_ID)


return_code, initial_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'initial', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get initial dummy handle ok.')

return_code, goal_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'goal', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get goal dummy handle ok.')

return_code, Franka_target_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'Franka_target', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get Franka target dummy handle ok.')

return_code, via_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'via', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get via dummy handle ok.')

time.sleep(0.1)

#%% DMP learning
# get demonstrated trajectory from file
df = pd.read_csv('./demo_trajectory/demo_trajectory_for_discrete_dmp.csv', header=None)
reference_trajectory = np.array(df)
data_dim = reference_trajectory.shape[0]
data_len = reference_trajectory.shape[1]

dmp = dmp_discrete(n_dmps=data_dim, n_bfs=1000, dt=1.0/data_len)# 建立离散dmp
dmp.learning(reference_trajectory)

reproduced_trajectory, _, _ = dmp.reproduce()

fig = plt.figure()
ax=Axes3D(fig)
plt.plot(reference_trajectory[0,:], reference_trajectory[1,:], reference_trajectory[2,:], 'g', label='reference position')
plt.plot(reproduced_trajectory[:,0], reproduced_trajectory[:,1], reproduced_trajectory[:,2], 'r--', label='reproduce position')
plt.legend()

fig = plt.figure()
plt.subplot(311)
plt.plot(reference_trajectory[0,:], 'g', label='reference x')
plt.plot(reproduced_trajectory[:,0], 'r--', label='reproduce x')
plt.legend()

plt.subplot(312)
plt.plot(reference_trajectory[1,:], 'g', label='reference y')
plt.plot(reproduced_trajectory[:,1], 'r--', label='reproduce y')
plt.legend()

plt.subplot(313)
plt.plot(reference_trajectory[2,:], 'g', label='reference z')
plt.plot(reproduced_trajectory[:,2], 'r--', label='reproduce z')
plt.legend()

plt.draw()

#%% the main loop
print("Main loop is begining ...")
max_loop = 2

reproduced_trajectory_record_x = np.zeros((data_len, max_loop))
reproduced_trajectory_record_y = np.zeros((data_len, max_loop))
reproduced_trajectory_record_z = np.zeros((data_len, max_loop))

for loop in range(max_loop):
    if loop == 0:
        # DMP reproduce the reference trajectory
        reproduced_trajectory, _, _ = dmp.reproduce()
    elif loop <= 5:
        # randomly add offset to the trajectory initial and goal positions
        goal_pos = [reference_trajectory[0,-1] + np.random.uniform(-0.1,0.2), reference_trajectory[1,-1]  + np.random.uniform(-0.2,0.05), reference_trajectory[2,-1] + np.random.uniform(-0.1, 0.3), 
            reference_trajectory[3,-1], reference_trajectory[4,-1], reference_trajectory[5,-1], reference_trajectory[6,-1]]
        vrep_sim.simxSetObjectPosition(client_ID, goal_dummy_handle, -1, goal_pos[:3], vrep_sim.simx_opmode_oneshot)

        # DMP reproduce with new goal positions
        reproduced_trajectory, _, _ = dmp.reproduce(goal=goal_pos)
    else:
        # randomly add offset to the trajectory initial and goal positions
        initial_pos = [reference_trajectory[0,0] + np.random.uniform(-0.05,0.05), reference_trajectory[1,0]  + np.random.uniform(-0.05,0.05), reference_trajectory[2,0] + np.random.uniform(-0.05, 0.05)]
        vrep_sim.simxSetObjectPosition(client_ID, initial_dummy_handle, -1, initial_pos, vrep_sim.simx_opmode_oneshot)

        goal_pos = [reference_trajectory[0,-1] + np.random.uniform(-0.1,0.2), reference_trajectory[1,-1]  + np.random.uniform(-0.2,0.05), reference_trajectory[2,-1] + np.random.uniform(-0.5, 0.1)]
        vrep_sim.simxSetObjectPosition(client_ID, goal_dummy_handle, -1, goal_pos, vrep_sim.simx_opmode_oneshot)

        # DMP reproduce with new initial and goal positions
        reproduced_trajectory, _, _ = dmp.reproduce(initial=initial_pos, goal=goal_pos)

    data_len = reproduced_trajectory.shape[0]
    reproduced_trajectory_record_x[:,loop] = reproduced_trajectory[:,0]
    reproduced_trajectory_record_y[:,loop] = reproduced_trajectory[:,1]
    reproduced_trajectory_record_z[:,loop] = reproduced_trajectory[:,2]

    print(reproduced_trajectory)

    # go to the goal position
    for i in range(data_len):
        Franka_target_pos = reproduced_trajectory[i,:3]
        Franka_target_ort = reproduced_trajectory[i,3:]
        vrep_sim.simxSetObjectPosition(client_ID, via_dummy_handle, -1, Franka_target_pos, vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSetObjectQuaternion(client_ID, via_dummy_handle, -1, Franka_target_ort, vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSynchronousTrigger(client_ID)  # trigger one simulation step

    # go back to the initial position
    for i in range(data_len-1, 0, -1):
        Franka_target_pos = reproduced_trajectory[i,:3]
        Franka_target_ort = reproduced_trajectory[i,3:]
        vrep_sim.simxSetObjectPosition(client_ID, via_dummy_handle, -1, Franka_target_pos, vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSetObjectQuaternion(client_ID, via_dummy_handle, -1, Franka_target_ort, vrep_sim.simx_opmode_oneshot)
        vrep_sim.simxSynchronousTrigger(client_ID)  # trigger one simulation step

vrep_sim.simxStopSimulation(client_ID, vrep_sim.simx_opmode_blocking) # stop the simulation
vrep_sim.simxFinish(-1)  # Close the connection
print('Program terminated')

#%% Plot
fig=plt.figure()
ax=Axes3D(fig)
plt.plot(reference_trajectory[0,:], reference_trajectory[1,:], reference_trajectory[2,:], 'g', label='reference')
for i in range(max_loop):
    plt.plot(reproduced_trajectory_record_x[:,i], reproduced_trajectory_record_y[:,i], reproduced_trajectory_record_z[:,i], '--')

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.show()
