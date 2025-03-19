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

/***************************** backend.h ************************************
 *
 * Header file of Backend base class.
 *
 ******************************************************************************/

#ifndef backend_h
#define backend_h

#include "utility_function.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>

#include <tbb/concurrent_queue.h>

#include <iosfwd>
#include <fstream>
#include <iostream>
#include <ctime>
#include <mutex>
#include <thread>

using namespace std;
using namespace gtsam;
#define LINESIZE 81920

/*
This class is the base class for all landmark slam backends.

We assume that the front end will provide the following information:
    1. The landmark detection result including its id, relateive pose wrt current camera pose.
    2. The odometry measurement between current and previous camera pose (either from IMU or a seperate VIO module).

Using April Tag as landmark alleviates the uncertainty in data association.
*/
namespace tagslam_ros
{
    class Backend
    {

    public:
        Backend(std::shared_ptr<rclcpp::Node> node);

        ~Backend();
        /*
        Update the smoother with landmark detections and odometry measurement.
        */
        virtual nav_msgs::msg::Odometry::SharedPtr updateSLAM(TagDetectionArrayPtr landmark_ptr, 
                                    EigenPose odom, EigenPoseCov odom_cov) = 0;

        /*
        Update the smoother with landmark detections, preintegrated IMU factor and odometry measurement.
        */
        virtual nav_msgs::msg::Odometry::SharedPtr updateVIO(TagDetectionArrayPtr landmark_ptr,
                                    EigenPose odom, EigenPoseCov odom_cov, bool use_odom) = 0;

        void setupIMU(double accel_noise_sigma, double accel_bias_rw_sigma,
                    double gyro_noise_sigma, double gyro_bias_rw_sigma, EigenPose T_sensor2cam);

        /*
        Add IMU message to the queue
        */
        void updateIMU(sensor_msgs::msg::Imu::SharedPtr imu_msg_ptr);

        /*
        Retrive the smoothed poses from the smoother as Eigen::Matrix4d, and save it to the given container.
        */
        virtual void getPoses(EigenPoseMap &container, const unsigned char filter_char) = 0;

        virtual void reset() = 0;

        Values getLandmarks()
        {
            return landmark_values_;
        }

        /*
        create visualization_msgs/msg/marker_array message from landmark_values_
        */
        visualization_msgs::msg::MarkerArray::SharedPtr createMarkerArray(std_msgs::msg::Header header);

    protected:
        /*
        read Values from a file
        */
        Values::shared_ptr read_from_file(const string &filename);

        /*
        write Values to a file
        */
        void write_to_file(const Values &estimate, const string &filename);

        /*
        initialize the slam
        */
        Pose3 initSLAM(double cur_img_t);

        /*
        add combined imu factor to connect previous NavState to current NavState
        */
        Pose3 addImuFactor(double cur_img_t);

        /*
        add odometry factor to connect previous Pose3 to current Pose3
        */
        Pose3 addOdomFactor(EigenPose odom, EigenPoseCov odom_cov);

        /*
        add odometry factor to connect previous Pose3 to current Pose3 along with IMU preintegration
        */
        Pose3 addImuFactor(EigenPose odom, EigenPoseCov odom_cov, double cur_img_t, bool use_odom);

        /*
        create nav_msg/odometry message from Eigen::Matrix4d
        */
        nav_msgs::msg::Odometry::SharedPtr createOdomMsg(Pose3 pose, EigenPoseCov pose_cov, 
                                            Vector3 linear_v, Vector3 angular_w, 
                                            double time, int seq);
    
        /*
        Template member function to add landmark factors into factor graph
        */
        template <typename T>
        void addLandmarkFactor(T &smoother,
                            TagDetectionArrayPtr landmark_ptr,
                            Pose3 &cur_pose_init)
        {
            Key cur_pose_key = Symbol(kPoseSymbol, pose_count_);

            // add prior factor for landmarks
            for (auto &landmark : landmark_ptr->detections)
            {
                if (!landmark.static_tag)
                {
                    // skip non-static tags
                    RCLCPP_WARN(this->node_->get_logger(), "Found no static tag! This should not happen! Check the tag detection result.");
                    continue;
                }
                Key landmark_key = Symbol(kLandmarkSymbol, landmark.id);
                // landmark.pose is geometry_msgs::msg::Pose
                Pose3 landmark_factor = Pose3(get_transform(landmark.pose));
                auto landmark_noise = noiseModel::Gaussian::Covariance(landmark_factor_cov_);

                // add prior factor to the local graph
                factor_graph_.emplace_shared<BetweenFactor<Pose3>>(
                    cur_pose_key, landmark_key, landmark_factor, landmark_noise);

                if (!smoother.valueExists(landmark_key))
                {
                    // check if the landmark is in the prior map
                    if (landmark_values_.exists(landmark_key))
                    {
                        // if the landmark is in the prior map, we add it to the initial estimate
                        Pose3 landmark_prior = landmark_values_.at<Pose3>(landmark_key);
                        initial_estimate_.insert(landmark_key, landmark_prior);
                        // insert prior factor to the graph
                        auto landmark_prior_noise = noiseModel::Gaussian::Covariance(landmark_cov_[landmark_key]);
                        factor_graph_.add(PriorFactor<Pose3>(landmark_key, landmark_prior, landmark_prior_noise));

                        // Equality Constraints can only use QR solver It is too slow
                        // factor_graph_.add(NonlinearEquality1<Pose3>(landmark_prior, landmark_key));
                    }
                    else
                    {
                        // if this a new landmark, we add it by reprojecting the landmark from the current predicted pose
                        Pose3 landmark_init = cur_pose_init * landmark_factor;
                        initial_estimate_.insert(landmark_key, landmark_init);
                    }
                }
            }
        }

    private:
        // declare node_ to keep track for logging and params
        std::shared_ptr<rclcpp::Node> node_; 
        
        // set the gravity vector for preint_param_ using the imu message
        void setGravity(sensor_msgs::msg::Imu::SharedPtr imu_msg_ptr);

        // load map from load_map_path_
        void loadMap();

    protected:
        // system parameters
        bool initialized_;
        bool prior_map_;  // if true, the system will use prior factor to initialize the map
        bool save_graph_; // if true, the system will save the factor graph to a file

        // file path to the map file
        // if the map file is not empty, the system will load the map from the file
        // the robot will localize or do SLAM based using the loaded map as prior
        std::string load_map_path_;
        std::string save_map_path_;

        // Factor graph and inital value of each variables
        NonlinearFactorGraph factor_graph_;
        Values initial_estimate_;

        bool fix_prior_;  // if true, the system will fix the prior factor for fixed lag smoother
        Values fixed_landmark_;
        Values landmark_values_;
        std::unordered_map<Key, EigenPoseCov> landmark_cov_;

        // keep track of the number of poses
        int pose_count_;

        // noise parameters
        double landmark_factor_sigma_trans_;
        double landmark_factor_sigma_rot_;
        EigenPoseCov landmark_factor_cov_;

        double landmark_prior_sigma_trans_;
        double landmark_prior_sigma_rot_;
        EigenPoseCov landmark_prior_cov_;

        double pose_prior_sigma_trans_;
        double pose_prior_sigma_rot_;
        double vel_prior_sigma_;
        double bias_prior_sigma_;

        // concurrent queue for imu data
        tbb::concurrent_queue<sensor_msgs::msg::Imu::SharedPtr> imu_queue_;
        boost::shared_ptr<PreintegrationCombinedParams> preint_param_;
        boost::shared_ptr<PreintegratedCombinedMeasurements> preint_meas_ = nullptr;

        // values to track of previous states
        Pose3 prev_pose_;
        Vector3 prev_vel_;
        NavState prev_state_;
        imuBias::ConstantBias prev_bias_; // assume zero initial bias

        bool need_pose_offset_ = false;
        EigenPose pose_offset_ = EigenPose::Identity();
        EigenPose pose_offset_inv_ = EigenPose::Identity();

        // correct the acceleration and angular velocity using the bias
        Vector3 correct_acc_ = Vector3::Zero();
        Vector3 correct_gyro_ = Vector3::Zero();

        double prev_img_t_ = 0.0;

        // mutex for reset
        std::mutex reset_mutex_;
    };

}

#endif