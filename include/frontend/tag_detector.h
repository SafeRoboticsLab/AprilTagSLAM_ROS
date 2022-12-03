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

 **/

 /***************************** tag_detector.h ************************************
 *
 * Header file of TagDetector class. It is the base class of TagDetectorGPU and TagDetectorCPU. 
 *
 ******************************************************************************/

#ifndef TAG_DETECTOR_ABS_H
#define TAG_DETECTOR_ABS_H

#include "utility_function.h"
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>

#ifndef NO_CUDA_OPENCV
  #include <opencv2/cvconfig.h>
  #include <opencv2/core/cuda.hpp>
  #include <opencv2/cudaimgproc.hpp>
#endif


namespace tagslam_ros{
  class TagDetector
  {
    public:
      TagDetector() = default;

      ~TagDetector() = default;

      // Detect tags in an image
      virtual TagDetectionArrayPtr detectTags(const sensor_msgs::ImageConstPtr& msg_img,
        const sensor_msgs::CameraInfoConstPtr& msg_cam_info) = 0;

#ifndef NO_CUDA_OPENCV
      virtual TagDetectionArrayPtr detectTags(cv::cuda::GpuMat& cv_mat_gpu,
        const sensor_msgs::CameraInfoConstPtr& msg_cam_info, std_msgs::Header) = 0;
#endif

      virtual TagDetectionArrayPtr detectTags(cv::Mat& cv_mat_cpu,
        const sensor_msgs::CameraInfoConstPtr& msg_cam_info, std_msgs::Header) = 0;

      void drawDetections(cv_bridge::CvImagePtr image, 
                  TagDetectionArrayPtr tag_detection);

      void drawDetections(cv::Mat & image,
            TagDetectionArrayPtr tag_detection);
  };
} // namespace tagslam_ros
#endif // APRILTAG_ROS_COMMON_FUNCTIONS_H