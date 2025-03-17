
# ROS AprilTag SLAM

This repo contains a ROS implementation of a real-time landmark SLAM based on [AprilTag](https://github.com/AprilRobotics/apriltag) and [GTSAM](https://gtsam.org/). We use AprilTag as the front end, which alleviates the challengings in data association. Both CPU and CUDA versions of the AprilTag front end are included. We provide iSAM2 and batch fixed-lag smoother for SLAM and visual Inertia Odometry.

The package is initially built for a [ZED2](https://www.stereolabs.com/zed-2/) camera running on [Nvidia Jetson Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-xavier-nx/). However, it can be easily adapted for other sensors and other platforms. We have tested this on JetPack 5.0.2 and regular PC with Ubuntu 20.04, ROS Noetic and Nvidia GTX 1080Ti.

## Dependency (Must)
* [Apriltag 3](https://github.com/AprilRobotics/apriltag)
* [ROS Humble](https://docs.ros.org/en/humble/index.html)
* A slightly modified version of [GTSAM 4.1.1](https://github.com/SafeRoboticsLab/gtsam/tree/release-4.1.1) built with *gtsam_unstable*.
## Dependency (Highly Recommended)

* **[ZED SDK 3.8](https://www.stereolabs.com/developers/release/)**:

    If ZED SDK is not installed, you will not able to run SLAM that directly interface with ZED SDK. However, you can still run SLAM that subscribes to ROS topics. **If you do not have ZED SDK, You will have to to add `-DUSE_ZED=OFF` flag to the catkin.**

* **CUDA 11+**:

    If CUDA is not enabled, we will only do SLAM with CPU based CUDA detector. Interface with ZED SDK will be turned off as well. **If you do not have CUDA, You will have to to add `-DUSE_CUDA=OFF` flag to the catkin.**

    *Note*: older JetPack with CUDA 10 should work as well, but we have not tested it.

* **OpenCV 4.6.0 with CUDA support**:


## Install
First, complete an environment set up on your Ubuntu 20.04 machine like [here](https://github.com/SafeRoboticsLab/PrincetonRaceCar/tree/SP2025/Jetson_Setup). This will install 
- [x] [ros-noetic ](http://wiki.ros.org/noetic/Installation/Ubuntu)
- [x] [zed-sdk](https://www.stereolabs.com/developers/release/)
- [x] [GTSAM Release 4.1.1](https://gtsam.org/build/)
- [x] RoboStack (conda ROS environment) using a PySpline compiled .whl file

In the last step `source set_startup_package` we build a `~/StartUp` directory similar to below. However, this also includes the [`Maestro-Controller-ROS`](https://github.com/SafeRoboticsLab/Maestro-Controller-ROS) repository. To build `AprilTagSLAM_ROS` on
its own, follow these instructions:
<!-- note: may need to include PrincetonRaceCar_msgs -->

```
mkdir tagslam_ws
cd tagslam_ws
mkdir src
cd src

git clone https://github.com/SafeRoboticsLab/AprilTagSLAM_ROS.git 
cd AprilTagSLAM_ROS
git checkout ros2_updates
cd ..

git clone https://github.com/AprilRobotics/apriltag.git
cd apriltag
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install
cd ..

# We need compile cv_bridge from source files to avoid mixing OpenCV versions
git clone https://github.com/ros-perception/vision_opencv.git
cd vision_opencv
git checkout humble
cd ../..

# build
source /opt/ros/humble/setup.bash
colcon build --cmake-args -DUSE_CUDA=OFF
```
 
## Run with ZED SDK
Checkout the settings in [config/config.yaml](config/config.yaml).

```
source devel_isolated/setup.bash
ros2 launch tagslam_ros zed_sdk.launch
```
