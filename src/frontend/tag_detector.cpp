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

#include "frontend/tag_detector.hpp"
#include <stdexcept>


namespace tagslam_ros
{   
    TagDetector::TagDetector(std::shared_ptr<rclcpp::Node> node) : 
     node_(node)
    {
        // Transform from camera frame to ROS frame
        T_cam_to_ros_ << 0, 0, 1, 0,
                        -1, 0, 0, 0,
                        0, -1, 0, 0,
                        0, 0, 0, 1;

        // Declare parameters with default values
        // each tag is a std::string with comma (,) separated attributes: id_start, id_end, tag_size
        node_->declare_parameter<std::vector<std::string>>("landmark_tags", {}, rcl_interfaces::msg::ParameterDescriptor());
        node_->declare_parameter<std::vector<std::string>>("ignore_tags", {}, rcl_interfaces::msg::ParameterDescriptor());

        // parse landmark tag group
        
        if (node_->has_parameter("landmark_tags")) {
            std::vector<std::string> landmark_tags;
            node_->get_parameter("landmark_tags", landmark_tags);
            try {
                parseTagGroup(tag_size_list_, landmark_tags, true); 
            } catch(const std::exception &e) {
                RCLCPP_ERROR(node_->get_logger(), "Error loading landmark_tags descriptions: %s",
                            e.what());
            }
        } else {
            RCLCPP_WARN(node_->get_logger(), "Failed to get landmark_tags");
        }
        

        if (node_->has_parameter("ignore_tags")) {
            std::vector<std::string> ignore_tags;
            node_->get_parameter("ignore_tags", ignore_tags);
            try {
                parseTagGroup(tag_size_list_, ignore_tags, false); 
            } catch(const std::exception &e) {
                RCLCPP_ERROR(node_->get_logger(), "Error loading ignore_tags descriptions: %s",
                            e.what());
            }
        } else {
            RCLCPP_WARN(node_->get_logger(), "Failed to get ignore_tags");
        }
    }

    void TagDetector::drawDetections(std::shared_ptr<cv_bridge::CvImage> image,
            TagDetectionArrayPtr tag_detection)
    {
        drawDetections(image->image, tag_detection);
    }

    void TagDetector::drawDetections(cv::Mat & image,
            TagDetectionArrayPtr tag_detection)
    {
        for (auto &det : tag_detection->detections)
        {

        // Draw tag outline with edge colors green, blue, blue, red
        // (going counter-clockwise, starting from lower-left corner in
        // tag coords). cv::Scalar(Blue, Green, Red) format for the edge
        // colors! 

        cv::Point p1((int)det.corners[0].x, (int)det.corners[0].y);
        cv::Point p2((int)det.corners[1].x, (int)det.corners[1].y);
        cv::Point p3((int)det.corners[2].x, (int)det.corners[2].y);
        cv::Point p4((int)det.corners[3].x, (int)det.corners[3].y);

        line(image, p1, p2, cv::Scalar(0,0xff,0)); // green
        line(image, p2, p3, cv::Scalar(0,0,0xff));
        line(image, p3, p4, cv::Scalar(0xff,0,0)); // green
        line(image, p4, p1, cv::Scalar(0xff,0,0)); // green
        
        // Print tag ID in the middle of the tag
        std::stringstream ss;
        ss << det.id;
        cv::String text = ss.str();
        int fontface = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
        double fontscale = 0.5;
        int baseline;
        cv::Size textsize = cv::getTextSize(text, fontface,
                                            fontscale, 2, &baseline);
        cv::Point center((int)det.center.x, (int)det.center.y);
        cv::putText(image, text, center,
                    fontface, fontscale, cv::Scalar(0xff, 0x99, 0), 2);
        }
    }

    void TagDetector::parseTagGroup(std::map<int, SizeStaticPair> &tag_group_map, 
                        const std::vector<std::string> &tag_group, bool static_tag)
    {   
        for (const auto &tag_str : tag_group) {
            std::istringstream tag_stream(tag_str);
            std::string id_start_str, id_end_str, tag_size_str;
            if (!std::getline(tag_stream, id_start_str, ',') ||
                !std::getline(tag_stream, id_end_str, ',') ||
                !std::getline(tag_stream, tag_size_str, ','))
            {
                RCLCPP_ERROR(node_->get_logger(), "Invalid tag format: %s", tag_str.c_str());
                continue;
            }

            int id_start = std::stoi(id_start_str);
            int id_end = std::stoi(id_end_str);
            double tag_size = std::stod(tag_size_str);
            
            RCLCPP_INFO(node_->get_logger(), "Tag group from %d to %d has size %f", id_start, id_end, tag_size);
            if (id_end < id_start) {
                RCLCPP_ERROR(node_->get_logger(), "id_start %d should be less than id_end %d", id_start, id_end);
                continue;
            }
            for (int id = id_start; id <= id_end; id++) {
                if (tag_group_map.find(id) != tag_group_map.end()) {
                    RCLCPP_WARN(node_->get_logger(), "Tag id %d is already in tag group, will be overwritten", id);
                }
                tag_group_map[id] = std::make_pair(tag_size, static_tag);
            }
        }
    }

    EigenPose TagDetector::getRelativeTransform(
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

    EigenPose TagDetector::getRelativeTransform(
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
}