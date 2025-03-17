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

 /***************************** tag_slam.h ************************************
 *
 * Header file of TagSlam class which estimate the camera's pose once 
 * received camera image and odometry information through ROS subscriber.
 *
 ******************************************************************************/

#ifndef TAG_SLAM_H
#define TAG_SLAM_H

#include "frontend/tag_detector.h"
#include "frontend/tag_detector_cpu.h"

#ifndef NO_CUDA
  #include "frontend/tag_detector_cuda.h"
#endif

#include "backend/backend.h"
#include "backend/fixed_lag_backend.h"
#include "backend/isam2_backend.h"

#include <cv_bridge/cv_bridge.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>

#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

namespace tagslam_ros
{

  class TagSlam : public nodelet::Nodelet
  {
  public:
    TagSlam() = default;
    ~TagSlam();

    void onInit();

    // private function
  private:
    void DetectionOnlyCallback(const sensor_msgs::msg::Image::ConstSharedPtr &image,
                                const sensor_msgs::msg::Image::ConstSharedPtr &camera_info);

    void imageOdomCallback(
        const sensor_msgs::msg::Image::ConstSharedPtr &image,
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info,
        const nav_msgs::msg::Odometry::ConstSharedPtr &odom);

    // front end
  private:
    std::unique_ptr<TagDetector> tag_detector_;

    // parameters
    bool if_pub_tag_det_;
    bool if_pub_tag_det_image_;
    bool use_approx_sync_;
    std::string image_topic_;
    std::string camera_info_topic_;
    std::string odom_topic_;
    std::string slam_pose_topic_;
    cv_bridge::CvImagePtr cv_image_;

    bool detection_only_;

    // runtime parameters
    int num_frames_;
    EigenPose prev_pose_;
    std::unordered_map<int, int> tag_frequency_;

    // subscribers
    image_transport::ImageTransport::SharedPtr it_;
    image_transport::CameraSubscriber camera_image_subscriber_;

    image_transport::SubscriberFilter image_subscriber_;
    message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_subscriber_;
    message_filters::Subscriber<nav_msgs::msg::Odometry> odom_subscriber_;

    // sync policy
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                            sensor_msgs::msg::CameraInfo,
                                                            nav_msgs::msg::Odometry>
        ApproxSyncPolicy;

    typedef message_filters::sync_policies::ExactTime<sensor_msgs::msg::Image,
                                                      sensor_msgs::msg::CameraInfo,
                                                      nav_msgs::msg::Odometry>
        ExactSyncPolicy;

    typedef message_filters::Synchronizer<ApproxSyncPolicy> ApproxSync;
    typedef message_filters::Synchronizer<ExactSyncPolicy> ExactSync;

    // synchronizer
    ApproxSync::SharedPtr approx_sync_;
    ExactSyn::SharedPtr exact_sync_;

    // publishers
    image_transport::Publisher tag_detections_image_publisher_;
    rclcpp::Publisher static_tag_det_pub_;
    rclcpp::Publisher dyn_tag_det_pub_;
    rclcpp::Publisher slam_pose_publisher_;

    // back end
  private:
    std::unique_ptr<Backend> slam_backend_;
    EigenPose prev_vio_pose_;
  };

} // namespace apriltag_ros

#endif // TAG_SLAM_H
