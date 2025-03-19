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

 /***************************** tag_detector_cpu.h ************************************
 *
 * Header file of TagDetectorCPU class which interface with AprilTag3, detect the tags
 * and estimate their relative poses wrt the camera. 
 *
 ******************************************************************************/

#ifndef TAG_DETECTOR_CPU_H
#define TAG_DETECTOR_CPU_H

#include "utility_function.hpp"
#include "frontend/tag_detector.hpp"

#include <apriltag.h>

namespace tagslam_ros{
  class TagDetectorCPU: public TagDetector
  {
    public:
      TagDetectorCPU(std::shared_ptr<rclcpp::Node> node);

      ~TagDetectorCPU();

      // Detect tags in an image
      void detectTags(const sensor_msgs::msg::Image::ConstSharedPtr,
          const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg_cam_info,
          TagDetectionArrayPtr static_tag_array_ptr, TagDetectionArrayPtr dyn_tag_array_ptr);

#ifndef NO_CUDA_OPENCV
      void detectTags(cv::cuda::GpuMat& cv_mat_gpu,
        const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg_cam_info, std_msgs::msg::Header header,
        TagDetectionArrayPtr static_tag_array_ptr, TagDetectionArrayPtr dyn_tag_array_ptr)
        {
          throw std::logic_error("CPU based Apriltag only supports cv::Mat");
        }
#endif
      
      void detectTags(cv::Mat& cv_mat_cpu, const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg_cam_info,
          std_msgs::msg::Header header,
          TagDetectionArrayPtr static_tag_array_ptr,
          TagDetectionArrayPtr dyn_tag_array_ptr);

      // Draw the detected tags' outlines and payload values on the image
      void drawDetections(cv_bridge::CvImagePtr image);

    private:
      // declare node_ to keep track for logging and params
      std::shared_ptr<rclcpp::Node> node_; 
       
      // AprilTag 2 code's attributes
      std::string family_;
      int threads_;
      double decimate_;
      double blur_;
      int refine_edges_;
      int debug_;
      int max_hamming_distance_ = 2;  // Tunable, but really, 2 is a good choice. Values of >=3
                                      // consume prohibitively large amounts of memory, and otherwise
                                      // you want the largest value possible.
      double tag_size_ = 1.0;

      // AprilTag 2 objects
      apriltag_family_t *tf_;
      apriltag_detector_t *td_;
      zarray_t *detections_;

  };
} // namespace tagslam_ros
#endif // APRILTAG_ROS_COMMON_FUNCTIONS_H
