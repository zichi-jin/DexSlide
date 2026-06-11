import os
import sys
import ctypes
import keyboard

# 获取当前脚本所在目录，并根据情况调整路径
base_dir = "/home/jzq/MyJob/DexSlide/robot_manipulation/JAKA_control/JAKA_dependecies/x86_64-linux-gnu"

# 1. 加载 libjakaAPI.so
lib_path = os.path.join(base_dir, 'libjakaAPI.so')
ctypes.CDLL(lib_path)
sys.path.append(base_dir)

# 3. 导入 jkrc 模块
import jkrc

# -*- coding: utf-8 -*-   
import time        
PI = 3.1415926  

robot = jkrc.RC("192.168.99.31")#返回一个机器人对象  
# robot = jkrc.RC("192.168.99.44")#返回一个机器人对象  
ret = robot.login()#登录  
ret = robot.power_on()  
ret = robot.enable_robot()  
robot.set_torsenosr_brand(2)  
robot.set_torque_sensor_mode(1)  
robot.set_compliant_type(1, 1)  
print("inint sensor comple")  
print("ready to run")  
ret = robot.get_joint_position()  
joint_pos_origin = ret[1]  
joint_pos = ret[1]  
print(joint_pos)  
joint_pos[3] += PI / 4  
if (joint_pos[3] > 265 * PI / 180):  
    joint_pos[3] -= 90  
joint_pos[4] += PI / 4  
if (joint_pos[4] > 320 * PI / 180):  
    joint_pos[4] -= 90  
joint_pos[5] += PI / 4  
if (joint_pos[5] > 265 * PI / 180):  
    joint_pos[5] -= PI  
print(joint_pos)  
ret = robot.start_torq_sensor_payload_identify(joint_pos)  
time.sleep(1)  
flag = 1  
while (1 == flag):  
    ret = robot.get_torq_sensor_identify_staus()  
    print(ret)  
    time.sleep(1)  
    flag = ret[1]  
print("identy_finish")  
ret = robot.get_torq_sensor_payload_identify_result()  
print(ret)  
ret = robot.get_payload()  
print(ret)  
ret = robot.set_payload(mass=ret[1][0], centroid=ret[1][1:])  
print(ret)  
ret = robot.get_torq_sensor_payload_identify_result()  
print(ret)  
ret = robot.get_payload()  
print(ret)  
robot.joint_move(joint_pos_origin,0,1,10)  
print("back")  
robot.logout()  #登出
