/*
Code is adapted from the AprilTag_ROS package by Danylo Malyuta, JPL
*/

/**
 * Copyright (c) 2017, California Institute of Technology.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those of the authors and should not be interpreted as representing official
 * policies, either expressed or implied, of the California Institute of
 * Technology.
 */
#include "frontend/tag_detector_cpu.hpp"

#include "common/homography.h"
#include "tagStandard52h13.h"
#include "tagStandard41h12.h"
#include "tag36h11.h"
#include "tag25h9.h"
#include "tag16h5.h"
#include "tagCustom48h12.h"
#include "tagCircle21h7.h"
#include "tagCircle49h12.h"

namespace tagslam_ros
{

  TagDetectorCPU::TagDetectorCPU(std::shared_ptr<rclcpp::Node> node) : 
    TagDetector(node),
    node_(node),
    family_(get_ros_option<std::string>(node_, "frontend/tag_family", "tag36h11")),
    threads_(get_ros_option<int>(node_, "frontend/tag_threads", 4)),
    decimate_(get_ros_option<double>(node_, "frontend/tag_decimate", 1.0)),
    blur_(get_ros_option<double>(node_, "frontend/tag_blur", 0.0)),
    refine_edges_(get_ros_option<int>(node_, "frontend/tag_refine_edges", 1)),
    debug_(get_ros_option<int>(node_, "frontend/tag_debug", 0)),
    max_hamming_distance_(get_ros_option<int>(node_, "frontend/max_hamming_dist", 2)),
    tag_size_(get_ros_option<double>(node_, "frontend/tag_size", 1.0))
  {
    //
    RCLCPP_INFO(node_->get_logger(),"Initializing cpu AprilTag detector with family %s", family_.c_str());
    RCLCPP_INFO(node_->get_logger(), "Tag Size: %f", tag_size_);
    RCLCPP_INFO(node_->get_logger(), "Threads: %d", threads_);
    RCLCPP_INFO(node_->get_logger(), "Decimate: %f", decimate_);
    RCLCPP_INFO(node_->get_logger(), "Blur: %f", blur_);
    RCLCPP_INFO(node_->get_logger(), "Refine edges: %d", refine_edges_);
    RCLCPP_INFO(node_->get_logger(), "Debug: %i", debug_);
    RCLCPP_INFO(node_->get_logger(), "Max hamming distance: %i", max_hamming_distance_);
    
    // Define the tag family whose tags should be searched for in the camera
    // images
    if (family_ == "tagStandard52h13") {
      tf_ = tagStandard52h13_create();
    } else if (family_ == "tagStandard41h12") {
      tf_ = tagStandard41h12_create();
    } else if (family_ == "tag36h11") {
      tf_ = tag36h11_create();
    } else if (family_ == "tag25h9") {
      tf_ = tag25h9_create();
    } else if (family_ == "tag16h5") {
      tf_ = tag16h5_create();
    } else if (family_ == "tagCustom48h12") {
      tf_ = tagCustom48h12_create();
    } else if (family_ == "tagCircle21h7") {
      tf_ = tagCircle21h7_create();
    } else if (family_ == "tagCircle49h12") {
      tf_ = tagCircle49h12_create();
    } else {
      RCLCPP_ERROR(node_->get_logger(), "Invalid tag family specified! Aborting");
      rclcpp::shutdown();
    }

    // Create the AprilTag 2 detector
    td_ = apriltag_detector_create();
    apriltag_detector_add_family_bits(td_, tf_, max_hamming_distance_);
    td_->quad_decimate = (float)decimate_;
    td_->quad_sigma = (float)blur_;
    td_->nthreads = threads_;
    td_->debug = debug_;
    td_->refine_edges = refine_edges_;

    detections_ = nullptr;
  }

  // destructor
  TagDetectorCPU::~TagDetectorCPU()
  {
    // free memory associated with tag detector
    apriltag_detector_destroy(td_);

    // free memory associated with the array of tag detections
    if (detections_) {
      apriltag_detections_destroy(detections_);
    }

    // free memory associated with tag family
    if (tf_) {
      if (family_ == "tagStandard52h13") tagStandard52h13_destroy(tf_);
      else if (family_ == "tagStandard41h12") tagStandard41h12_destroy(tf_);
      else if (family_ == "tag36h11") tag36h11_destroy(tf_);
      else if (family_ == "tag25h9") tag25h9_destroy(tf_);
      else if (family_ == "tag16h5") tag16h5_destroy(tf_);
      else if (family_ == "tagCustom48h12") tagCustom48h12_destroy(tf_);
      else if (family_ == "tagCircle21h7") tagCircle21h7_destroy(tf_);
      else if (family_ == "tagCircle49h12") tagCircle49h12_destroy(tf_);
    }
  }

  // detect april tag from image
  void TagDetectorCPU::detectTags(const sensor_msgs::msg::Image::ConstSharedPtr msg_img, 
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info,
      TagDetectionArrayPtr static_tag_array_ptr, TagDetectionArrayPtr dyn_tag_array_ptr)
  {
    // Convert image to AprilTag code's format
    cv::Mat gray_image;
    try {
      gray_image = cv_bridge::toCvCopy(msg_img, "mono8")->image;
    } catch (cv_bridge::Exception& e){
      RCLCPP_ERROR(node_->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    detectTags(gray_image, camera_info, msg_img->header, static_tag_array_ptr, dyn_tag_array_ptr);
  }
  
  void TagDetectorCPU::detectTags(cv::Mat& cv_mat_cpu,
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg_cam_info, std_msgs::msg::Header header,
        TagDetectionArrayPtr static_tag_array_ptr, TagDetectionArrayPtr dyn_tag_array_ptr)
  {
    image_u8_t apriltag_image = {.width = cv_mat_cpu.cols,
                                  .height = cv_mat_cpu.rows,
                                  .stride = cv_mat_cpu.cols,
                                  .buf = cv_mat_cpu.data};

    image_geometry::PinholeCameraModel camera_model;
    camera_model.fromCameraInfo(msg_cam_info);

    // Get camera intrinsic properties for rectified image.
    double fx = camera_model.fx(); // focal length in camera x-direction [px]
    double fy = camera_model.fy(); // focal length in camera y-direction [px]
    double cx = camera_model.cx(); // optical center x-coordinate [px]
    double cy = camera_model.cy(); // optical center y-coordinate [px]

    cv::Matx33d cameraMatrix(fx, 0, cx,
                              0, fy, cy,
                              0, 0, 1);

    cv::Mat distCoeffs = camera_model.distortionCoeffs();

    // std::cout<<"camera matrix "<<cameraMatrix<<std::endl;
    // std::cout<<"distCoeffs matrix "<<distCoeffs<<std::endl;
    
    // Run AprilTag 2 algorithm on the image
    if (detections_)
    {
      apriltag_detections_destroy(detections_);
      detections_ = nullptr;
    }
    detections_ = apriltag_detector_detect(td_, &apriltag_image);

    // Compute the estimated translation and rotation individually for each
    // detected tag
    static_tag_array_ptr->header = header;
    dyn_tag_array_ptr->header = header;

    for (int i = 0; i < zarray_size(detections_); i++)
    {
      // Get the i-th detected tag
      apriltag_detection_t *detection;
      zarray_get(detections_, i, &detection);

      // Get the tag ID
      int tagID = detection->id;
      
      double cur_tag_size = tag_size_;
      bool cur_tag_static = true;
      // try to see if the tag is in the list of tags to be detected
      if(tag_size_list_.find(tagID) != tag_size_list_.end())
      {
        cur_tag_size = tag_size_list_[tagID].first;
        cur_tag_static = tag_size_list_[tagID].second;
      }

      // Get estimated tag pose in the camera frame.
      //
      // Note on frames:
      // we want:
      //   - camera frame: looking from behind the camera (like a
      //     photographer), x is right, y is down and z is straight
      //     ahead
      //   - tag frame: looking straight at the tag (oriented correctly),
      //     x is right, y is up and z is towards you (out of the tag).
      
      std::vector<cv::Point2d> TagImagePoints;
      for (int corner_idx = 0; corner_idx < 4; corner_idx++) {
        TagImagePoints.push_back(cv::Point2d(detection->p[corner_idx][0],
                                    detection->p[corner_idx][1]));
      }

      std::vector<cv::Point3d> TagObjectPoints;
      // from bottom left to bottom left and going counter clockwise
      TagObjectPoints.push_back(cv::Point3d(-cur_tag_size / 2, -cur_tag_size / 2, 0));
      TagObjectPoints.push_back(cv::Point3d(cur_tag_size / 2, -cur_tag_size / 2, 0));
      TagObjectPoints.push_back(cv::Point3d(cur_tag_size / 2, cur_tag_size / 2, 0));
      TagObjectPoints.push_back(cv::Point3d(-cur_tag_size / 2, cur_tag_size / 2, 0));

      EigenPose T_tag_to_cam = getRelativeTransform(TagObjectPoints,TagImagePoints,cameraMatrix, distCoeffs);
      EigenPose T_tag_to_ros = T_cam_to_ros_ * T_tag_to_cam;

      geometry_msgs::msg::Pose tag_pose = createPoseMsg(T_tag_to_ros);

      // Add the detection to the back of the tag detection array
      tagslam_ros::msg::AprilTagDetection tag_detection;
      tag_detection.pose = tag_pose;
      tag_detection.id = tagID;
      tag_detection.static_tag = cur_tag_static;
      tag_detection.size = cur_tag_size;

      // corners
      for (int corner_idx = 0; corner_idx < 4; corner_idx++) {
        tag_detection.corners.data()[corner_idx].x = detection->p[corner_idx][0];
        tag_detection.corners.data()[corner_idx].y = detection->p[corner_idx][1];
      }

      // center
      tag_detection.center.x = detection->c[0];
      tag_detection.center.y = detection->c[1];
      
      if(cur_tag_static)
      {
        static_tag_array_ptr->detections.push_back(tag_detection);
      }
      else
      {
        dyn_tag_array_ptr->detections.push_back(tag_detection);
      }
    }
  }

} // namespace tagslam_ros