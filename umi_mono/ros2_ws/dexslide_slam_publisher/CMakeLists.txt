cmake_minimum_required(VERSION 3.8)
project(dexslide_slam_publisher CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(Python3_EXECUTABLE /usr/bin/python3 CACHE FILEPATH "System Python for ROS2 tooling" FORCE)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(LIBZMQ REQUIRED libzmq)

add_executable(pose_publisher_node src/pose_publisher_node.cpp)
target_include_directories(pose_publisher_node PRIVATE
  /home/jzq/MyJob/DexSlide/umi_mono/external/cppzmq
  ${LIBZMQ_INCLUDE_DIRS}
)
target_link_directories(pose_publisher_node PRIVATE ${LIBZMQ_LIBRARY_DIRS})
target_link_libraries(pose_publisher_node ${LIBZMQ_LIBRARIES})
ament_target_dependencies(pose_publisher_node rclcpp geometry_msgs tf2_ros)

# ----- new realsense_topic_slam_node -----
set(ORB_SLAM3_FORK_ROOT /home/jzq/MyJob/DexSlide/umi_mono/external/ORB_SLAM3_fork)
list(APPEND CMAKE_PREFIX_PATH /home/jzq/MyJob/DexSlide/umi_mono/external/Pangolin/build)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(realsense_topic_slam_node src/realsense_topic_slam_node.cpp)
target_include_directories(realsense_topic_slam_node PRIVATE
  ${ORB_SLAM3_FORK_ROOT}
  ${ORB_SLAM3_FORK_ROOT}/include
  ${ORB_SLAM3_FORK_ROOT}/include/CameraModels
  ${ORB_SLAM3_FORK_ROOT}/Thirdparty/Sophus
  ${ORB_SLAM3_FORK_ROOT}/Examples/Monocular-Inertial
  ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(realsense_topic_slam_node
  ${ORB_SLAM3_FORK_ROOT}/lib/libORB_SLAM3.so
  ${OpenCV_LIBS}
  pango_core
  pango_display
  pango_geometry
  pango_glgeometry
  pango_image
  pango_opengl
  pango_packetstream
  pango_plot
  pango_scene
  pango_tools
  pango_vars
  pango_video
  pango_windowing
)
target_link_options(realsense_topic_slam_node PRIVATE
  "-Wl,--allow-shlib-undefined"
  "-Wl,-rpath,${ORB_SLAM3_FORK_ROOT}/lib"
  "-Wl,-rpath,${ORB_SLAM3_FORK_ROOT}/Thirdparty/DBoW2/lib"
  "-Wl,-rpath,${ORB_SLAM3_FORK_ROOT}/Thirdparty/g2o/lib"
  "-Wl,-rpath,/home/jzq/MyJob/DexSlide/umi_mono/external/Pangolin/build"
)
ament_target_dependencies(realsense_topic_slam_node
  rclcpp
  geometry_msgs
  tf2_ros
  cv_bridge
  sensor_msgs
)

install(TARGETS pose_publisher_node realsense_topic_slam_node
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/aruco_world_pose_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}
)

ament_package()
