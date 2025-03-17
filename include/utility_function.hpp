/**
* BSD 3-Clause License
* Copyright (c) 2021, The Trustees of Princeton University. All rights reserved.
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are
* met:
*     1. Redistributions of source code must retain the above copyrigh
*        notice, this list of conditions and the following disclaimer.
*     2. Redistributions in binary form must reproduce the above
*        copyright notice, this list of conditions and the following
*        disclaimer in the documentation and/or other materials provided
*        with the distribution
*     3. Neither the name of the copyright holder nor the names of its
*        contributors may be used to endorse or promote products derived
*        from this software without specific prior written permission.
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, TH
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE US
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

* Please contact the author(s) of this library if you have any questions.
* Authors:    Zixu Zhang       ( zixuz@princeton.edu )
              Jarod Wille      ( jwille@princeton.edu )

 **/

 /***************************** utility_function.h ************************************
 *
 * Header file of utility functions.
 *
 ******************************************************************************/

#ifndef UTIL_H
#define UTIL_H

// system includes
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <map>
#include <memory>
#include <unordered_map>
#include <cmath>
#include <ctime>
#include <chrono>

// front end includes
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

// rclcpp (ros2) includes
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rclcpp/logging.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/header.hpp>
#include <std_msgs/msg/float32.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "tagslam_ros/msg/april_tag_detection.hpp"
#include "tagslam_ros/msg/april_tag_detection_array.hpp"

namespace tagslam_ros{

  using TagDetectionArrayPtr = msg::AprilTagDetectionArray::SharedPtr;
  using EigenPose = Eigen::Matrix4d;
  using EigenPoseSigma = Eigen::Matrix<double, 6, 1>; // rotx, roty, rotz, x, y, z
  using EigenPoseCov = Eigen::Matrix<double, 6, 6>; // rotx, roty, rotz, x, y, z
  using EigenPoseWithSigma = std::pair<EigenPose, EigenPoseSigma>;
  
  using EigenPoseMap = std::unordered_map<int, EigenPose>;
  using EigenPoseMapPtr = std::shared_ptr<EigenPoseMap>;

  // symbol used in backend
  static constexpr unsigned char kPoseSymbol = 'x';
  static constexpr unsigned char kLandmarkSymbol = 'l';
  static constexpr unsigned char kVelSymbol = 'V'; // (xdot,ydot,zdot)
  static constexpr unsigned char kBiasSymbol = 'B'; //(ax,ay,az,gx,gy,gz)

  template<typename T>
  T get_ros_option(std::shared_ptr<rclcpp::Node> node,
                      const std::string &param_name, const T &default_val)
  {
    if(!node->has_parameter(param_name))
    {
      node->declare_parameter(param_name, default_val);
      std::ostringstream default_val_stream;
      default_val_stream << default_val;
      RCLCPP_WARN(node->get_logger(), "Parameter %s does not exist, setting to default: %s",
                    param_name.c_str(), default_val_stream.str().c_str());
    }
    T param_val;
    node->get_parameter(param_name, param_val);
    return param_val;
  }

  static inline EigenPose get_transform(const geometry_msgs::msg::Pose& pose_msg)
  {
    Eigen::Affine3d transform;
    transform.translation() << pose_msg.position.x, pose_msg.position.y, pose_msg.position.z;
    transform.linear() = Eigen::Quaterniond(pose_msg.orientation.w, pose_msg.orientation.x,
                                  pose_msg.orientation.y, pose_msg.orientation.z).toRotationMatrix();
    return transform.matrix();
  }

  static inline EigenPoseSigma get_transform_sigma(const geometry_msgs::msg::PoseWithCovariance & pose_cov_msg)
  {
    // gtsam have order rotx, roty, rotz, x, y, z
    // ros have order x, y, z, rotx, roty, rotz
    EigenPoseSigma sigma;
    sigma << std::sqrt(pose_cov_msg.covariance[21]), std::sqrt(pose_cov_msg.covariance[28]),
            std::sqrt(pose_cov_msg.covariance[35]), std::sqrt(pose_cov_msg.covariance[0]),
            std::sqrt(pose_cov_msg.covariance[7]), std::sqrt(pose_cov_msg.covariance[14]);

    return sigma;
  }

  static inline EigenPoseCov get_transform_cov(const geometry_msgs::msg::PoseWithCovariance & pose_cov_msg)
  {
    // gtsam have order rotx, roty, rotz, x, y, z
    // ros have order x, y, z, rotx, roty, rotz

    // [TT, TR;
    //  RT, RR]
    EigenPoseCov cov_ros;
    for(int i=0; i<6; i++){
      for(int j=0; j<6; j++)
      {
        int k = i*6+j;
        cov_ros(i,j) = pose_cov_msg.covariance[k];
      }
    }


    Eigen::Matrix3d TT = cov_ros.block<3,3>(0,0);
    Eigen::Matrix3d TR = cov_ros.block<3,3>(0,3);
    Eigen::Matrix3d RT = cov_ros.block<3,3>(3,0);
    Eigen::Matrix3d RR = cov_ros.block<3,3>(3,3);

    // [RR, RT;
    // [TR, TT]
    EigenPoseCov cov_gtsam;
    cov_gtsam.block<3,3>(0,0) = RR;
    cov_gtsam.block<3,3>(0,3) = RT;
    cov_gtsam.block<3,3>(3,0) = TR;
    cov_gtsam.block<3,3>(3,3) = TT;

    return cov_gtsam;
  }

  static inline EigenPoseWithSigma getTransformWithSigma(const geometry_msgs::msg::PoseWithCovariance & pose_cov_msg)
  {
    EigenPose transform = get_transform(pose_cov_msg.pose);
    EigenPoseSigma sigma = get_transform_sigma(pose_cov_msg);
    return std::make_pair(transform, sigma);
  }

  static inline geometry_msgs::msg::Pose createPoseMsg(const EigenPose &transform)
  {
    geometry_msgs::msg::Pose pose;
    // translation
    pose.position.x = transform(0, 3);
    pose.position.y = transform(1, 3);
    pose.position.z = transform(2, 3);

    // rotation
    Eigen::Matrix3d rotation = transform.block<3, 3>(0, 0);
    Eigen::Quaternion<double> rot_quaternion(rotation);
    pose.orientation.x = rot_quaternion.x();
    pose.orientation.y = rot_quaternion.y();
    pose.orientation.z = rot_quaternion.z();
    pose.orientation.w = rot_quaternion.w();
    return pose;
  }
} // namespace tagslam_ros
#endif /* util_h */