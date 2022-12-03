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
#include "frontend/tag_detector_cpu.h"

#include "image_geometry/pinhole_camera_model.h"
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

  TagDetectorCPU::TagDetectorCPU(ros::NodeHandle pnh) : 
    TagDetector(),
    family_(getRosOption<std::string>(pnh, "frontend/tag_family", "tag36h11")),
    threads_(getRosOption<int>(pnh, "frontend/tag_threads", 4)),
    decimate_(getRosOption<double>(pnh, "frontend/tag_decimate", 1.0)),
    blur_(getRosOption<double>(pnh, "frontend/tag_blur", 0.0)),
    refine_edges_(getRosOption<int>(pnh, "frontend/tag_refine_edges", 1)),
    debug_(getRosOption<int>(pnh, "frontend/tag_debug", 0)),
    max_hamming_distance_(getRosOption<int>(pnh, "frontend/max_hamming_dist", 2)),
    tag_size_(getRosOption<double>(pnh, "frontend/tag_size", 1.0))
  {
    //
    ROS_INFO_STREAM("Initializing cpu AprilTag detector with family " << family_);
    ROS_INFO_STREAM("Tag Size: " << tag_size_);
    ROS_INFO_STREAM("Threads: " << threads_);
    ROS_INFO_STREAM("Decimate: " << decimate_);
    ROS_INFO_STREAM("Blur: " << blur_);
    ROS_INFO_STREAM("Refine edges: " << refine_edges_);
    ROS_INFO_STREAM("Debug: " << debug_);
    ROS_INFO_STREAM("Max hamming distance: " << max_hamming_distance_);
    
    // Define the tag family whose tags should be searched for in the camera
    // images
    if (family_ == "tagStandard52h13")
    {
      tf_ = tagStandard52h13_create();
    }
    else if (family_ == "tagStandard41h12")
    {
      tf_ = tagStandard41h12_create();
    }
    else if (family_ == "tag36h11")
    {
      tf_ = tag36h11_create();
    }
    else if (family_ == "tag25h9")
    {
      tf_ = tag25h9_create();
    }
    else if (family_ == "tag16h5")
    {
      tf_ = tag16h5_create();
    }
    else if (family_ == "tagCustom48h12")
    {
      tf_ = tagCustom48h12_create();
    }
    else if (family_ == "tagCircle21h7")
    {
      tf_ = tagCircle21h7_create();
    }
    else if (family_ == "tagCircle49h12")
    {
      tf_ = tagCircle49h12_create();
    }
    else
    {
      ROS_WARN("Invalid tag family specified! Aborting");
      exit(1);
    }

    // Create the AprilTag 2 detector
    td_ = apriltag_detector_create();
    apriltag_detector_add_family_bits(td_, tf_, max_hamming_distance_);
    td_->quad_decimate = (float)decimate_;
    td_->quad_sigma = (float)blur_;
    td_->nthreads = threads_;
    td_->debug = debug_;
    td_->refine_edges = refine_edges_;
    detections_ = NULL;
  }

  // destructor
  TagDetectorCPU::~TagDetectorCPU()
  {
    // free memory associated with tag detector
    apriltag_detector_destroy(td_);

    // Free memory associated with the array of tag detections
    if (detections_)
    {
      apriltag_detections_destroy(detections_);
    }

    // free memory associated with tag family
    if (family_ == "tagStandard52h13")
    {
      tagStandard52h13_destroy(tf_);
    }
    else if (family_ == "tagStandard41h12")
    {
      tagStandard41h12_destroy(tf_);
    }
    else if (family_ == "tag36h11")
    {
      tag36h11_destroy(tf_);
    }
    else if (family_ == "tag25h9")
    {
      tag25h9_destroy(tf_);
    }
    else if (family_ == "tag16h5")
    {
      tag16h5_destroy(tf_);
    }
    else if (family_ == "tagCustom48h12")
    {
      tagCustom48h12_destroy(tf_);
    }
    else if (family_ == "tagCircle21h7")
    {
      tagCircle21h7_destroy(tf_);
    }
    else if (family_ == "tagCircle49h12")
    {
      tagCircle49h12_destroy(tf_);
    }
  }

  // detect april tag from image
  TagDetectionArrayPtr TagDetectorCPU::detectTags(
      const sensor_msgs::ImageConstPtr& msg_img,
      const sensor_msgs::CameraInfoConstPtr &camera_info)
  {
    // Convert image to AprilTag code's format
    cv::Mat gray_image;
    try{
      gray_image = cv_bridge::toCvShare(msg_img, "mono8")->image;
    }catch (cv_bridge::Exception& e){
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return nullptr;
    }

    return detectTags(gray_image, camera_info, msg_img->header);
  }
  
  TagDetectionArrayPtr TagDetectorCPU::detectTags(cv::Mat& cv_mat_cpu,
        const sensor_msgs::CameraInfoConstPtr& msg_cam_info, std_msgs::Header header)
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
      detections_ = NULL;
    }
    detections_ = apriltag_detector_detect(td_, &apriltag_image);

    // Compute the estimated translation and rotation individually for each
    // detected tag
    auto tag_detection_array = std::make_shared<AprilTagDetectionArray>();
    tag_detection_array->header = header;

    for (int i = 0; i < zarray_size(detections_); i++)
    {
      // Get the i-th detected tag
      apriltag_detection_t *detection;
      zarray_get(detections_, i, &detection);

      // Get the tag ID
      int tagID = detection->id;

      // Get estimated tag pose in the camera frame.
      //
      // Note on frames:
      // The raw AprilTag 2 uses the following frames:
      //   - camera frame: looking from behind the camera (like a
      //     photographer), x is right, y is up and z is towards you
      //     (i.e. the back of camera)
      //   - tag frame: looking straight at the tag (oriented correctly),
      //     x is right, y is down and z is away from you (into the tag).
      // But we want:
      //   - camera frame: looking from behind the camera (like a
      //     photographer), x is right, y is down and z is straight
      //     ahead
      //   - tag frame: looking straight at the tag (oriented correctly),
      //     x is right, y is up and z is towards you (out of the tag).
      // Using these frames together with cv::solvePnP directly avoids
      // AprilTag 2's frames altogether.
      // TODO solvePnP[Ransac] better?
      std::vector<cv::Point3d> standaloneTagObjectPoints;
      std::vector<cv::Point2d> standaloneTagImagePoints;
      addObjectPoints(tag_size_ / 2, cv::Matx44d::eye(), standaloneTagObjectPoints);
      addImagePoints(detection, standaloneTagImagePoints);
      EigenPose transform = getRelativeTransform(standaloneTagObjectPoints,
                                                        standaloneTagImagePoints,
                                                        cameraMatrix, distCoeffs);

      geometry_msgs::Pose tag_pose = makePoseMsg(transform);

      // Add the detection to the back of the tag detection array
      AprilTagDetection tag_detection;
      tag_detection.pose = tag_pose;
      tag_detection.id = detection->id;
      tag_detection.size = tag_size_;

      // corners
      for (int corner_idx = 0; corner_idx < 4; corner_idx++) {
        tag_detection.corners.data()[corner_idx].x = detection->p[corner_idx][0];
        tag_detection.corners.data()[corner_idx].y = detection->p[corner_idx][1];
      }

      // center
      tag_detection.center.x = detection->c[0];
      tag_detection.center.y = detection->c[1];
      tag_detection_array->detections.push_back(tag_detection);
    }
    return tag_detection_array;
  }
  
  void TagDetectorCPU::addObjectPoints(
      double s, cv::Matx44d T_oi, std::vector<cv::Point3d> &objectPoints) const
  {
    // Add to object point vector the tag corner coordinates in the bundle frame
    // Going counterclockwise starting from the bottom left corner
    objectPoints.push_back(T_oi.get_minor<3, 4>(0, 0) * cv::Vec4d(-s, -s, 0, 1));
    objectPoints.push_back(T_oi.get_minor<3, 4>(0, 0) * cv::Vec4d(s, -s, 0, 1));
    objectPoints.push_back(T_oi.get_minor<3, 4>(0, 0) * cv::Vec4d(s, s, 0, 1));
    objectPoints.push_back(T_oi.get_minor<3, 4>(0, 0) * cv::Vec4d(-s, s, 0, 1));
  }

  void TagDetectorCPU::addImagePoints(
      apriltag_detection_t *detection,
      std::vector<cv::Point2d> &imagePoints) const
  {
    // Add to image point vector the tag corners in the image frame
    // Going counterclockwise starting from the bottom left corner
    double tag_x[4] = {-1, 1, 1, -1};
    double tag_y[4] = {1, 1, -1, -1}; // Negated because AprilTag tag local
                                      // frame has y-axis pointing DOWN
                                      // while we use the tag local frame
                                      // with y-axis pointing UP
    for (int i = 0; i < 4; i++)
    {
      // Homography projection taking tag local frame coordinates to image pixels
      double im_x, im_y;
      homography_project(detection->H, tag_x[i], tag_y[i], &im_x, &im_y);
      imagePoints.push_back(cv::Point2d(im_x, im_y));
    }
  }

  EigenPose TagDetectorCPU::getRelativeTransform(
      std::vector<cv::Point3d> objectPoints,
      std::vector<cv::Point2d> imagePoints,
      double fx, double fy, double cx, double cy) const
  {
    // perform Perspective-n-Point camera pose estimation using the
    // above 3D-2D point correspondences
    cv::Mat rvec, tvec;
    cv::Matx33d cameraMatrix(fx, 0, cx,
                              0, fy, cy,
                              0, 0, 1);
    cv::Vec4f distCoeffs(0, 0, 0, 0); // distortion coefficients
    // TODO Perhaps something like SOLVEPNP_EPNP would be faster? Would
    // need to first check WHAT is a bottleneck in this code, and only
    // do this if PnP solution is the bottleneck.
    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);
    cv::Matx33d R;
    cv::Rodrigues(rvec, R);
    Eigen::Matrix3d wRo;
    wRo << R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1), R(1, 2), R(2, 0), R(2, 1), R(2, 2);

    EigenPose T; // homogeneous transformation matrix
    T.topLeftCorner(3, 3) = wRo;
    T.col(3).head(3) << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);
    T.row(3) << 0, 0, 0, 1;
    return T;
  }

  EigenPose TagDetectorCPU::getRelativeTransform(
      std::vector<cv::Point3d> objectPoints,
      std::vector<cv::Point2d> imagePoints,
      cv::Matx33d cameraMatrix, cv::Mat distCoeffs) const
  {
    // perform Perspective-n-Point camera pose estimation using the
    // above 3D-2D point correspondences
    cv::Mat rvec, tvec;

    // TODO Perhaps something like SOLVEPNP_EPNP would be faster? Would
    // need to first check WHAT is a bottleneck in this code, and only
    // do this if PnP solution is the bottleneck.
    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, false, 6);
    cv::Matx33d R;
    cv::Rodrigues(rvec, R);
    Eigen::Matrix3d wRo;
    wRo << R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1), R(1, 2), R(2, 0), R(2, 1), R(2, 2);

    EigenPose T; // homogeneous transformation matrix
    T.topLeftCorner(3, 3) = wRo;
    T.col(3).head(3) << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);
    T.row(3) << 0, 0, 0, 1;
    return T;
  }
} // namespace tagslam_ros