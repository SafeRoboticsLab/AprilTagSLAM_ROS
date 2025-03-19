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

 /***************************** tag_detector.h ************************************
 *
 * Header file of TagDetector class. It is the base class of TagDetectorGPU and TagDetectorCPU. 
 *
 ******************************************************************************/

#ifndef TAG_DETECTOR_ABS_H
#define TAG_DETECTOR_ABS_H

#include "utility_function.hpp"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include "image_geometry/pinhole_camera_model.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>

#ifndef NO_CUDA_OPENCV
  #include <opencv2/cvconfig.h>
  #include <opencv2/core/cuda.hpp>
  #include <opencv2/cudaimgproc.hpp>
#endif


namespace tagslam_ros{
  using SizeStaticPair = std::pair<double, bool>;
  class TagDetector
  {
    public:
      TagDetector(std::shared_ptr<rclcpp::Node> node);

      ~TagDetector() = default;

      // Detect tags in an image
      virtual void detectTags(const sensor_msgs::msg::Image::ConstSharedPtr msg_img,
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg_cam_info,
        TagDetectionArrayPtr static_tag_array_ptr,
        TagDetectionArrayPtr dyn_tag_array_ptr) = 0;

#ifndef NO_CUDA_OPENCV
      virtual void detectTags(cv::cuda::GpuMat &cv_mat_gpu,
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg_cam_info, std_msgs::msg::Header,
        TagDetectionArrayPtr static_tag_array_ptr, TagDetectionArrayPtr dyn_tag_array_ptr) = 0;
#endif

      virtual void detectTags(cv::Mat &cv_mat_cpu,
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg_cam_info, std_msgs::msg::Header, 
        TagDetectionArrayPtr static_tag_array_ptr, TagDetectionArrayPtr dyn_tag_array_ptr) = 0;

      void drawDetections(cv_bridge::CvImagePtr image, 
                  TagDetectionArrayPtr tag_detection);

      void drawDetections(cv::Mat &image,
            TagDetectionArrayPtr tag_detection);

    protected:
    void parseTagGroup(std::map<int, SizeStaticPair> &tag_group_map, 
      const std::vector<std::string> &tag_group, bool static_tag);

      // Get the pose of the tag in the camera frame
      // Returns homogeneous transformation matrix [R,t;[0 0 0 1]] which
      // takes a point expressed in the tag frame to the same point
      // expressed in the camera frame. As usual, R is the (passive)
      // rotation from the tag frame to the camera frame and t is the
      // vector from the camera frame origin to the tag frame origin,
      // expressed in the camera frame.
      EigenPose getRelativeTransform(
          std::vector<cv::Point3d > objectPoints,
          std::vector<cv::Point2d > imagePoints,
          double fx, double fy, double cx, double cy) const;

      EigenPose getRelativeTransform(
          std::vector<cv::Point3d > objectPoints,
          std::vector<cv::Point2d > imagePoints,
          cv::Matx33d cameraMatrix, cv::Mat distCoeffs) const;

    protected:
      std::map<int, SizeStaticPair> tag_size_list_;

      EigenPose T_cam_to_ros_;

    private:
      // declare node_ to keep track for logging and params
      std::shared_ptr<rclcpp::Node> node_; 
  };
} // namespace tagslam_ros
#endif // TAG_DETECTOR_ABS_H
