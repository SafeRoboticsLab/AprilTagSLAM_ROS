
# ROS AprilTag SLAM

This repo contains a ROS implementation of a real-time landmark SLAM based on [AprilTag](https://github.com/AprilRobotics/apriltag) and [GTSAM](https://gtsam.org/). We use AprilTag as the front end, which alleviates the challengings in data association. Both CPU and CUDA versions of the AprilTag front end are included. We provide iSAM2 and batch fixed-lag smoother for SLAM and visual Inertia Odometry.

The package is initially built for a [ZED2](https://www.stereolabs.com/zed-2/) camera running on [Nvidia Jetson Xavier](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-xavier-nx/). However, it can be easily adapted for other sensors and other platforms. We have tested this on JetPack 5.0.2 and regular PC with Ubuntu 20.04, ROS Noetic and Nvidia GTX 1080Ti.

## Dependency (Must)
* [Apriltag 3](https://github.com/AprilRobotics/apriltag)
* [ROS Noetic](http://wiki.ros.org/noetic)
* A slightly modified version of [GTSAM 4.1.1](https://github.com/SafeRoboticsLab/gtsam/tree/release-4.1.1) built with *gtsam_unstable*.
## Dependency (Highly Recommanded)

* **[ZED SDK 3.8](https://www.stereolabs.com/developers/release/)**:

    If ZED SDK is not installed, you will not able to run SLAM that directly interface with ZED SDK. However, you can still run SLAM that subscribes to ROS topics. **If you do not have ZED SDK, You will have to to add `-DUSE_ZED=OFF` flag to the catkin.**

* **CUDA 11+**:

    If CUDA is not enabled, we will only do SLAM with CPU based CUDA detector. Interface with ZED SDK will be turned off as well. **If you do not have CUDA, You will have to to add `-DUSE_CUDA=OFF` flag to the catkin.**

    *Note*: older JetPack with CUDA 10 should work as well, but we have not tested it.

* **OpenCV 4.6.0 with CUDA support**:

    ZED's color images have BGRA8 format but CUDA Apriltag Detector requires RGBA8 format. When you have OpenCV compiled with CUDA, ZED SDK can directly send images to the CUDA device (GPU), and we can do conversion and detections without copy data between host (RAM) and device (GPU). This slightly improve the tag detection performance (15ms vs 20ms) on Jetson Xavier AGX in MAXN mode. **If you do not have CUDA enabled OpenCV, You will have to to add `-DUSE_CUDA_OPENCV=OFF` flag to the catkin.**

## Install
```
mkdir tagslam_ws
cd tagslam_ws
mkdir src
cd src

git clone https://github.com/SafeRoboticsLab/AprilTagSLAM_ROS.git 


# We need compile cv_bridge from source files to avoid mixing OpenCV versions
git clone https://github.com/ros-perception/vision_opencv.git

cd vision_opencv
git checkout noetic

cd ../..

# build
catkin_make_isolated
```
 
## Run with ZED SDK
Checkout the setting in [config/config.yaml](config/config.yaml).

```
source devel_isolated/setup.bash
roslaunch tagslam_ros zed_sdk.launch
```
