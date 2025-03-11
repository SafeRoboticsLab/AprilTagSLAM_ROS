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

#include "tag_slam.h"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

RCLCPP_COMPONENTS_REGISTER_NODE(tagslam_ros::TagSlam)

namespace tagslam_ros
{
  TagSlam::~TagSlam()
  {
    std::cout<<"Shutting down TagSlam nodelet..."<<std::endl;
    std::cout<<"Total "<<tag_frequency_.size()<<" tags detected."<<std::endl;
    for (auto it = tag_frequency_.begin(); it != tag_frequency_.end(); ++it)
      std::cout<<"Tag " << it->first << " detected " << it->second << " times."<<std::endl;
    std::cout<<"TagSlam nodelet shut down."<<std::endl;
  }

  void TagSlam::onInit ()
  {
    ros::NodeHandle& nh = getNodeHandle();
    ros::NodeHandle& pnh = getPrivateNodeHandle();

    num_frames_ = 0;
    tag_frequency_.clear();

    // Load parameters    
    if_pub_tag_det_ = get_ros_option<bool>(pnh, "publish/publish_tags", false);
    if_pub_tag_det_image_ = get_ros_option<bool>(pnh, "publish/publish_image_with_tags", false);
    use_approx_sync_ = get_ros_option<bool>(pnh, "publish/use_approx_sync", false);

    image_topic_ = get_ros_option<std::string>(pnh, "image_topic", "/camera/image_raw");
    camera_info_topic_ = get_ros_option<std::string>(pnh, "camera_info_topic", "/camera/camera_info");
    odom_topic_ = get_ros_option<std::string>(pnh, "odom_topic", "/odom");
    slam_pose_topic_ = get_ros_option<std::string>(pnh, "slam_pose_topic", "/slam_pose");
    
    std::string transport_hint;
    pnh.param<std::string>("transport_hint", transport_hint, "raw");

    // Initialize AprilTag detector
    std::string detector_type_ = pnh.param<std::string>("frontend/type", "CPU");
    if(detector_type_ == "CPU"){
        tag_detector_ = std::make_unique<TagDetectorCPU>(pnh);
    }else if(detector_type_ == "GPU"){
#ifndef NO_CUDA
        tag_detector_ = std::make_unique<TagDetectorCUDA>(pnh);
#else
        RCLCPP_ERROR(this->get_logger(), "CUDA AprilTag detector is not built. Add '-DUSE_CUDA=ON' flag to catkin");
#endif
    }else{ 
        RCLCPP_ERROR(this->get_logger(), "Invalid detector type: %s", detector_type_.c_str());
    }

    // Initialize SLAM backend
    std::string backend_type_ = pnh.param<std::string>("backend/smoother", "isam2");
    if(backend_type_=="isam2"){
        slam_backend_ = std::make_unique<iSAM2Backend>(pnh);
        RCLCPP_INFO(this->get_logger(), "Using iSAM2 backend.");
    }else if(backend_type_=="fixed_lag"){
        slam_backend_ = std::make_unique<FixedLagBackend>(pnh);
        RCLCPP_INFO(this->get_logger(), "Using fixed-lag backend.");
    }else if(backend_type_ == "none")
    {
        slam_backend_ = nullptr;
        detection_only_ = true;
        RCLCPP_INFO(this->get_logger(), "AprilTag Detector Mode.");
    }

    prev_vio_pose_ = EigenPose::Identity();

    // set up ros publishers and subscribers
    it_ = std::shared_ptr<image_transport::ImageTransport>(new image_transport::ImageTransport(nh));    

    if(detection_only_)
    {
      // We only use images to find april tags as landmarks
      camera_image_subscriber_ =
          it_->subscribeCamera(image_topic_, 1,
                              &TagSlam::DetectionOnlyCallback, this,
                              image_transport::TransportHints(transport_hint));
    }else{
      // do slam 
      // we want to use camera VIO as the odometry factor between frames
      RCLCPP_INFO_STREAM(this->get_logger(), "Subscribe to camera: "<< image_topic_);  
      image_subscriber_.subscribe(*it_, image_topic_, 1,
                              image_transport::TransportHints(transport_hint));
      camera_info_subscriber_.subscribe(nh, camera_info_topic_, 1);
      odom_subscriber_.subscribe(nh, odom_topic_, 1);

      // set up synchronizer
      if (use_approx_sync_)
      {
        approx_sync_.reset(new ApproxSync(ApproxSyncPolicy(10),
                                                image_subscriber_,
                                                camera_info_subscriber_,
                                                odom_subscriber_));
        approx_sync_->registerCallback(std::bind(&TagSlam::imageOdomCallback, this, _1, _2, _3));
      }else{
        exact_sync_.reset(new ExactSync(ExactSyncPolicy(10),
                                        image_subscriber_,
                                        camera_info_subscriber_,
                                        odom_subscriber_));
        exact_sync_->registerCallback(std::bind(&TagSlam::imageOdomCallback, this, _1, _2, _3));
      }
    }

    // set up publishers for tag detections and visualization
    if (if_pub_tag_det_){
      static_tag_det_pub_ = nh.advertise<AprilTagDetectionArray>("tag_detections", 1);
      dyn_tag_det_pub_ = nh.advertise<AprilTagDetectionArray>("tag_detections_dynamic", 1);
    }
    
    if (if_pub_tag_det_image_){
      tag_detections_image_publisher_ = it_->advertise("tag_detections_image", 1);
    }
    
    if(!detection_only_){
      slam_pose_publisher_ = nh.advertise<nav_msgs::msg::Odometry>("slam_pose", 1);
    }
  }

  void TagSlam::DetectionOnlyCallback (
      const sensor_msgs::msg::Image::ConstSharedPtr& image,
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info)
  {
    auto start = std::chrono::system_clock::now();

    auto static_tag_array_ptr = std::make_shared<AprilTagDetectionArray>();
    auto dyn_tag_array_ptr = std::make_shared<AprilTagDetectionArray>();
    tag_detector_->detectTags(image, camera_info,
                        static_tag_array_ptr, dyn_tag_array_ptr);

    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    ROS_INFO_STREAM("detection "<<elapsed.count() << " ms");
    
    // publish tag detections
    if (static_tag_det_pub_.get_subscription_count() > 0 && if_pub_tag_det_){
      // Publish detected tags in the image by AprilTag 2
      static_tag_det_pub_.publish(*static_tag_array_ptr);
    }

    if(dyn_tag_det_pub_.get_subscription_count() > 0 && if_pub_tag_det_){
      // Publish detected tags in the image by AprilTag 2
      dyn_tag_det_pub_.publish(*dyn_tag_array_ptr);
    }

    // Publish the camera image overlaid by outlines of the detected tags and their ids

    if (tag_detections_image_publisher_.get_subscription_count() > 0 && 
        if_pub_tag_det_image_)
    {
      std::shared_ptr<cv_bridge::CvImage> cv_ptr = cv_bridge::toCvCopy(image, "bgr8");
      tag_detector_->drawDetections(cv_ptr, static_tag_array_ptr);
      tag_detections_image_publisher_.publish(cv_ptr->toImageMsg());
    }
    num_frames_++;
  }

  void TagSlam::imageOdomCallback (
      const sensor_msgs::msg::Image::ConstSharedPtr& image,
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr& camera_info,
      const nav_msgs::msg::Odometry::ConstSharedPtr& odom)
  {
    auto static_tag_array_ptr = std::make_shared<AprilTagDetectionArray>();
    auto dyn_tag_array_ptr = std::make_shared<AprilTagDetectionArray>();
    tag_detector_->detectTags(image, camera_info,
                        static_tag_array_ptr, dyn_tag_array_ptr);

    // get the camera pose from VIO
    EigenPose vio_pose = get_transform(odom->pose.pose);
    EigenPoseCov vio_cov = get_transform_cov(odom->pose);

    // get relative vio pose between this frame and the previous frame
    EigenPose vio_pose_delta = prev_vio_pose_.inverse() * vio_pose;

    // do one step of slam
    auto slam_pose_msg = slam_backend_->updateSLAM(static_tag_array_ptr, vio_pose_delta, vio_cov);
    
    //publish the pose message
    if(slam_pose_msg){
      slam_pose_publisher_.publish(slam_pose_msg);
    }
    
    // publish tag detections
    if (static_tag_det_pub_.get_subscription_count() > 0 && if_pub_tag_det_){
      // Publish detected tags in the image by AprilTag 2
      static_tag_det_pub_.publish(*static_tag_array_ptr);
    }

    if(dyn_tag_det_pub_.get_subscription_count() > 0 && if_pub_tag_det_){
      // Publish detected tags in the image by AprilTag 2
      dyn_tag_det_pub_.publish(*dyn_tag_array_ptr);
    }

    // Publish the camera image overlaid by outlines of the detected tags and their ids
    if (tag_detections_image_publisher_.get_subscription_count() > 0 && 
        if_pub_tag_det_image_)
    {
      std::shared_ptr<cv_bridge::CvImage> cv_ptr = cv_bridge::toCvCopy(image, "bgr8");
      tag_detector_->drawDetections(cv_ptr, static_tag_array_ptr);
      tag_detector_->drawDetections(cv_ptr, dyn_tag_array_ptr);
      tag_detections_image_publisher_.publish(cv_ptr->toImageMsg());
    }

    // record the current vio pose for the next frame
    prev_vio_pose_ = vio_pose;
    num_frames_++;

  }

} // namespace tagslam_ros
