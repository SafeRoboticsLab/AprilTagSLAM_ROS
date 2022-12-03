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

 /***************************** tag_detector_cpu.h ************************************
 *
 * Header file of TagDetectorCPU class which interface with AprilTag3, detect the tags
 * and estimate their relative poses wrt the camera. 
 *
 ******************************************************************************/

#ifndef TAG_DETECTOR_CPU_H
#define TAG_DETECTOR_CPU_H

#include "utility_function.h"
#include "frontend/tag_detector.h"

#include <apriltag.h>

namespace tagslam_ros{
  class TagDetectorCPU: public TagDetector
  {
    public:
      TagDetectorCPU(ros::NodeHandle pnh);

      ~TagDetectorCPU();

      // Detect tags in an image
      TagDetectionArrayPtr detectTags(const sensor_msgs::ImageConstPtr&,
          const sensor_msgs::CameraInfoConstPtr& msg_cam_info);

#ifndef NO_CUDA_OPENCV
      TagDetectionArrayPtr detectTags(cv::cuda::GpuMat& cv_mat_gpu,
        const sensor_msgs::CameraInfoConstPtr& msg_cam_info, std_msgs::Header header)
        {
          throw std::logic_error("CPU based Apriltag only support cv::Mat");
          return nullptr;
        }
#endif
      
      TagDetectionArrayPtr detectTags(cv::Mat& cv_mat_cpu,
        const sensor_msgs::CameraInfoConstPtr& msg_cam_info, std_msgs::Header header);

      // Draw the detected tags' outlines and payload values on the image
      void drawDetections(cv_bridge::CvImagePtr image);

    private:
      
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
      
      void addImagePoints(apriltag_detection_t *detection,
                            std::vector<cv::Point2d >& imagePoints) const;
      void addObjectPoints(double s, cv::Matx44d T_oi,
                            std::vector<cv::Point3d >& objectPoints) const;

    private:
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
