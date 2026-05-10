采集数据：
		
	
	3. 使用roslaunch开启视频流与IMU
		a. roslaunch realsense2_camera rs_camera.launch
		b. 使用rostopic echo /camera/color/metadata 查看actual_fps
		c. 使用rostopic echo /camera/accel/sample + rostopic echo /camera/gyro/sample 测试IMU可读取到
		d. 使用rqt_image_view   可视化视频流check …
		e. 录制数据
			i. rosbag record -o data /camera/color/image_raw /camera/accel/sample  /camera/gyro/sample 
			ii. rqt_bag <name>.bag 查看topic输出
	4. 先录制一个mapping的视频，大约60s
	5. 录制一个用于检测aruco码的视频，并把名字改称aruco.bag
	6. 针对上面两个结果进行一次建图 ... conda activate umi
	num_worker = 8
	rsync -avz /local/*.bag username@remote_host:/remote/
	
ros两台电脑通信
1. 下面两行分别写进.bashrc
export ROS_MASTER_URI=http://<master_ip>:11311
export ROS_IP=<local_ip_of_this_computer>

数据预处理
1. vis the path with script "visualize_action.py" , note to change the "dataset path", vis will be saved in dataset/vis
2. delete the noisy path with "delete_noisypath.py", note to change the "deleted episode number list" and "dataset path"
3. run vis_generate_mask to generate side view mask, note to change the "dataset path"
    ** TODO - also generate wrist view mask ..
4. change the settings and training the model -- check "backbone" set in imitate_episodes, "dataset_dir" and "camera names" in constants, and parameters in default_config.yaml
    # note: check the sample range in utils.py!!!
    
训练模型
...

模型预处理
1. 修改ckpt路径参数并运行export_onnx生成对应ckpt的onnx文件
2. 修改GetTrajectory.py使用的onnx和norm和denorm用的dataset_states.pkl的位置


实体实验
1.关闭两边电脑的wifi后开启roscore以及action 推理结点
roscore
rosrun inference_control GetTrajectory.py

2. 读取摄像头
使用realsense-viewer关闭自动曝光，调整亮度/增益（pickduck: 300/64, picksocket: 100/64
roslaunch realsense2_camera rs_multiple_devices.launch serial_no_camera1:=238222075705  serial_no_camera2:=147322072111

3. 获取/joint_states topic && 机械臂控制
roslaunch jaka_planner moveit_server.launch ip:=192.168.2.199 model:=zu3

4. 夹爪控制
rosrun robotiq_2f_gripper_control robotiq_2f_action_server.py

5. 启动moveit
roslaunch jaka_zu3_moveit_config demo.launch

