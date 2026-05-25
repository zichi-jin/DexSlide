import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jzq/MyJob/DexSlide/robot_manipulation/orca_control/orca_hand_ros/install/orca_hand_ros'
