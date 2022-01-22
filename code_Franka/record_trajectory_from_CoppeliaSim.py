# -*- coding: utf-8 -*-
# 这就只是个记录的python文件，真正轨迹写在vrep里面
# 在vrep里用IKGroup，让tip追着target运动，而target在两个定点initial和goal之间来回运动。vrep里面用
# moveToPose这么一个自动内插，自带planning的函数，通过限定max速度来控制target运动的速度。记录的也是target的位置，不是机械臂此刻实际的位置。
# 原先版本记录的轨迹只有position，我将方向也加入轨迹记录
#%% import package
import time
import numpy as np

import sys
sys.path.append('./Franka/VREP_RemoteAPIs')
import sim as vrep_sim

sys.path.append('./Franka')
from FrankaSimModel import FrankaSimModel

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd

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

delta_t = 0.005 # simulation time step
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

time.sleep(0.1)

# get initial and goal position from CoppeliaSim 
_, initial_pos = vrep_sim.simxGetObjectPosition(client_ID, initial_dummy_handle, -1, vrep_sim.simx_opmode_blocking)
_, initial_quat = vrep_sim.simxGetObjectQuaternion(client_ID, initial_dummy_handle, -1, vrep_sim.simx_opmode_blocking)

_, goal_pos = vrep_sim.simxGetObjectPosition(client_ID, goal_dummy_handle, -1, vrep_sim.simx_opmode_blocking)
_, goal_quat = vrep_sim.simxGetObjectQuaternion(client_ID, goal_dummy_handle, -1, vrep_sim.simx_opmode_blocking)

pos_record_x = list()
pos_record_y = list()
pos_record_z = list()
ort_record_x = list()
ort_record_y = list()
ort_record_z = list()
ort_record_w = list()

record_enable = False

while True:
    _, Franka_target_pos = vrep_sim.simxGetObjectPosition(client_ID, Franka_target_dummy_handle, -1, vrep_sim.simx_opmode_oneshot)# 得到Franka_target的位置
    _, Franka_target_quat = vrep_sim.simxGetObjectQuaternion(client_ID, Franka_target_dummy_handle, -1, vrep_sim.simx_opmode_oneshot)# 得到Franka_target的方向四元数

    if (record_enable == False) and (np.sqrt((Franka_target_pos[0] - initial_pos[0])**2 + (Franka_target_pos[1] - initial_pos[1])**2 + (Franka_target_pos[2] - initial_pos[2])**2) < 0.005):# 如果距离初始点很近，并且flag=false就开始record
        record_enable = True
    if (np.sqrt((Franka_target_pos[0] - goal_pos[0])**2 + (Franka_target_pos[1] - goal_pos[1])**2 + (Franka_target_pos[2] - goal_pos[2])**2) < 0.005):# 如果距离目标点很近，并且flag=true就结束record，并break
        record_enable = False
        break

    if record_enable == True:
        pos_record_x.append(Franka_target_pos[0])
        pos_record_y.append(Franka_target_pos[1])
        pos_record_z.append(Franka_target_pos[2])
        ort_record_x.append(Franka_target_quat[0])
        ort_record_y.append(Franka_target_quat[1])
        ort_record_z.append(Franka_target_quat[2])
        ort_record_w.append(Franka_target_quat[3])
    vrep_sim.simxSynchronousTrigger(client_ID)  # trigger one simulation step

vrep_sim.simxStopSimulation(client_ID, vrep_sim.simx_opmode_blocking) # stop the simulation
vrep_sim.simxFinish(-1)  # Close the connection
print('Program terminated')

print(len(pos_record_x))

fig = plt.figure()
ax=Axes3D(fig)
plt.plot(pos_record_x, pos_record_y, pos_record_z)
plt.show()

#%% save the recorded data to files
data = np.vstack((pos_record_x, pos_record_y, pos_record_z,ort_record_x,ort_record_y,ort_record_z,ort_record_w))
print(data)

df = pd.DataFrame(data)
df.to_csv('./demo_trajectory/demo_trajectory_for_discrete_dmp.csv', index=False, header=None)
