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

# 4. 测试连接
print("尝试连接机械臂...")
robot = jkrc.RC('192.168.99.44')
robot.login()    
robot.power_on()    
robot.enable_robot()
robot.drag_mode_enable(True)

while True:
    ret = robot.is_in_drag_mode()    
    print(ret)
    if keyboard.is_pressed('space'):
        break
    
robot.drag_mode_enable(False)
ret = robot.is_in_drag_mode()    
print(ret)    
robot.logout()