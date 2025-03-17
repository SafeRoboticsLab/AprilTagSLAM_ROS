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

 /***************************** tag_detector_cuda.h ************************************
 *
 * Header file of TagDetectorGPU class which interface with nvapriltags, detect the tags
 * and estimate their relative poses wrt the camera. 
 *
 ******************************************************************************/

#ifndef TAG_DETECTOR_CUDA_H_
#define TAG_DETECTOR_CUDA_H_

#include "utility_function.hpp"
#include "frontend/tag_detector.hpp"

#include "cuda.h"  // NOLINT - include .h without directory
#include "cuda_runtime.h"  // NOLINT - include .h without directory
#include "nvAprilTags.hpp"


namespace tagslam_ros
{

class TagDetectorCUDA : public TagDetector
{
public:
  TagDetectorCUDA(std::shared_ptr<rclcpp::Node> node);

  ~TagDetectorCUDA();

  void detectTags(const sensor_msgs::msg::Image::ConstSharedPtr,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg_cam_info,
    TagDetectionArrayPtr static_tag_array_ptr, TagDetectionArrayPtr dyn_tag_array_ptr);

#ifndef NO_CUDA_OPENCV
  // takes in RGBA8 cv::cuda::GpuMat
  void detectTags(cv::cuda::GpuMat& cv_mat_gpu,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg_cam_info, std_msgs::msg::Header header,
    TagDetectionArrayPtr static_tag_array_ptr, TagDetectionArrayPtr dyn_tag_array_ptr);
#endif

  void detectTags(cv::Mat& cv_mat_cpu,
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg_cam_info, std_msgs::msg::Header header,
        TagDetectionArrayPtr static_tag_array_ptr, TagDetectionArrayPtr dyn_tag_array_ptr);
    
private:
  geometry_msgs::msg::Pose DetectionToPose(const nvAprilTagsID_t & detection);

  void runDetection(TagDetectionArrayPtr static_tag_array_ptr, TagDetectionArrayPtr dyn_tag_array_ptr);

  // declare node_ to keep track for logging and params
  std::shared_ptr<rclcpp::Node> node_;  

  const std::string tag_family_ = "36h11"; // cuda version only support this family
  const double tag_size_;
  const int max_tags_;

  struct AprilTagsImpl;
  std::unique_ptr<AprilTagsImpl> impl_;

  cv::Matx33d cameraMatrix_;
  cv::Mat distCoeffs_;
};

}  // namespace tagslam_ros

#endif  // TAG_DETECTOR_CUDA_H_
