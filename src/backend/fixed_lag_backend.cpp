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
#include "backend/fixed_lag_backend.hpp"

using namespace gtsam;

namespace tagslam_ros
{
    FixedLagBackend::FixedLagBackend(std::shared_ptr<rclcpp::Node> node) : 
        Backend(node),
        node_(node)
    {
        
        // load backend parameters
        lm_params_.lambdaInitial = get_ros_option<double>(node_, "backend/lambda_initial", 1e-5);
        lm_params_.lambdaUpperBound = get_ros_option<double>(node_, "backend/lambda_upper_bound", 1e5);
        lm_params_.lambdaLowerBound = get_ros_option<double>(node_, "backend/lambda_lower_bound", 0);
        lm_params_.lambdaFactor = get_ros_option<double>(node_, "backend/lambda_factor", 10.0);
        lm_params_.maxIterations = get_ros_option<int>(node_, "backend/max_iterations", 100);
        lm_params_.errorTol = get_ros_option<double>(node_, "backend/error_tol", 1e-5);
        lm_params_.relativeErrorTol = get_ros_option<double>(node_, "backend/relative_error_tol", 1e-4);
        lm_params_.absoluteErrorTol = get_ros_option<double>(node_, "backend/absolute_error_tol", 1e-5);
        local_optimal_ = get_ros_option<bool>(node_, "backend/local_optimal", false);

        
        lag_ = get_ros_option<double>(node_, "backend/lag", 1.0);
        smoother_ = BatchFixedLagSmoother(lag_, lm_params_, true, local_optimal_);
    }

    void FixedLagBackend::reset()
    {
        reset_mutex_.lock();
        initialized_ = false;
        pose_count_ = 0;
        prev_img_t_ = 0.0;
        imu_queue_.clear();
        landmark_values_.clear();
        landmark_cov_.clear();
        if (prior_map_){
            loadMap();
        }
        smoother_ = BatchFixedLagSmoother(lag_, lm_params_, true, local_optimal_);
        RCLCPP_INFO(node_->get_logger(), "Reset iSAM2");
        reset_mutex_.unlock();
    }

    nav_msgs::msg::Odometry::SharedPtr FixedLagBackend::updateSLAM(TagDetectionArrayPtr landmark_ptr, EigenPose odom, EigenPoseCov odom_cov)
    {
        // reset local graph and values
        factor_graph_.resize(0);
        initial_estimate_.clear();
        newTimestamps_.clear();

        if(reset_mutex_.try_lock()){
            int num_landmarks_detected = landmark_ptr->detections.size();
            Key cur_pose_key = Symbol(kPoseSymbol, pose_count_);
            Pose3 cur_pose_init;
            double cur_img_t = rclcpp::Time(landmark_ptr->header.stamp).seconds();

            if (initialized_)
            {
                cur_pose_init = addOdomFactor(odom, odom_cov);
            }
            else if (num_landmarks_detected > 0)
            {
                cur_pose_init = initSLAM(cur_img_t);
            }
            else
            {
                reset_mutex_.unlock();
                static bool warned1 = false;
                if (!warned1) {
                    RCLCPP_WARN(node_->get_logger(), "System not initialized, waiting for landmarks");
                    warned1 = true;
                }
                return nullptr;
            }

            // create new timesampe map for the current pose
            newTimestamps_[cur_pose_key] = cur_img_t;

            // add prior factor for landmarks
            addLandmarkFactor<BatchFixedLagSmoother>(smoother_, landmark_ptr, cur_pose_init);

            // update landmark timestamp
            for (auto &landmark : landmark_ptr->detections)
            {
                Key landmark_key = Symbol(kLandmarkSymbol, landmark.id);
                newTimestamps_[landmark_key] = cur_img_t;
            }

            // do a batch optimization
            try{
                smoother_.update(factor_graph_, initial_estimate_, newTimestamps_);
            }
            catch(gtsam::IndeterminantLinearSystemException)
            {
                reset_mutex_.unlock();
                static bool warned2 = false;
                if (!warned2) {
                    RCLCPP_WARN(node_->get_logger(), "SLAM Update Failed. Re-try next time step.");
                    warned2 = true;
                }
                return nullptr;
            }

            // get the current pose and save it for next iteration
            Values estimated_vals = smoother_.calculateEstimate();

            // get the marginals
            Marginals marginals(smoother_.getFactors(), estimated_vals);

            prev_pose_ = estimated_vals.at<Pose3>(cur_pose_key);
            EigenPoseCov pose_cov = marginals.marginalCovariance(cur_pose_key);

            updateLandmarkValues(estimated_vals, marginals);

            // make message
            auto odom_msg = createOdomMsg(prev_pose_, pose_cov, Vector3::Zero(), Vector3::Zero(), cur_img_t, pose_count_);

            pose_count_++;

            // reset local graph and values
            reset_mutex_.unlock();
            return odom_msg;
        }
        else{
            static bool warned3 = false;
                if (!warned3) {
                    RCLCPP_WARN(node_->get_logger(), "Resetting, waiting for reset to finish");
                    warned3 = true;
                }
            return nullptr;
        }
    }

    nav_msgs::msg::Odometry::SharedPtr FixedLagBackend::updateVIO(TagDetectionArrayPtr landmark_ptr, 
                                    EigenPose odom, EigenPoseCov odom_cov, bool use_odom)
    {
        // reset local graph and values
        factor_graph_.resize(0);
        initial_estimate_.clear();
        newTimestamps_.clear();

        // Reset the preintegration object.
        if(preint_meas_)
            preint_meas_->resetIntegrationAndSetBias(prev_bias_);
        
        if(reset_mutex_.try_lock()){
                
            int num_landmarks_detected = landmark_ptr->detections.size();

            Key cur_pose_key = Symbol(kPoseSymbol, pose_count_);
            Key cur_vel_key = Symbol(kVelSymbol, pose_count_);
            Key cur_bias_key = Symbol(kBiasSymbol, pose_count_);

            Pose3 cur_pose_init;

            Vector3 corrected_acc = Vector3::Zero();
            Vector3 corrected_gyro = Vector3::Zero();

            double cur_img_t = rclcpp::Time(landmark_ptr->header.stamp).seconds();

            if (initialized_)
            {
                cur_pose_init = addImuFactor(odom, odom_cov, cur_img_t, use_odom);
            }
            else if (num_landmarks_detected > 0)
            {
                cur_pose_init = initSLAM(cur_img_t);
            }
            else
            {   
                static bool warned4 = false;
                if (!warned4) {
                    RCLCPP_WARN(node_->get_logger(), "System not initialized, waiting for landmarks");
                    warned4 = true;
                }
                
                // dump all previous inserted imu measurement
                while (!imu_queue_.empty())
                {
                    sensor_msgs::msg::Imu::SharedPtr imu_msg_ptr;
                    while (!imu_queue_.try_pop(imu_msg_ptr))
                    {
                    }
                    if (rclcpp::Time(imu_msg_ptr->header.stamp).seconds() >= cur_img_t)
                    {
                        break;
                    }
                }
                reset_mutex_.unlock();
                return nullptr;
            }

            // create new timesampe map for the current pose
            newTimestamps_[cur_pose_key] = cur_img_t;
            newTimestamps_[cur_vel_key] = cur_img_t;
            newTimestamps_[cur_bias_key] = cur_img_t;

            // add prior factor for landmarks
            addLandmarkFactor<BatchFixedLagSmoother>(smoother_, landmark_ptr, cur_pose_init);

            // update landmark timestamp
            for (auto &landmark : landmark_ptr->detections)
            {
                Key landmark_key = Symbol(kLandmarkSymbol, landmark.id);
                newTimestamps_[landmark_key] = cur_img_t;
            }

            // do a batch optimization
            try{
                smoother_.update(factor_graph_, initial_estimate_, newTimestamps_);
            }
            catch(gtsam::IndeterminantLinearSystemException)
            {
                reset_mutex_.unlock();
                RCLCPP_WARN(node_->get_logger(), "SLAM Update Failed. Re-try next time step.");
                return nullptr;
            }

            // get the current pose and save it for next iteration
            Values estimated_vals = smoother_.calculateEstimate();

            // calculate marginals based on the current estimate
            Marginals marginals(smoother_.getFactors(), estimated_vals);

            prev_pose_ = estimated_vals.at<Pose3>(cur_pose_key);
            prev_vel_ = estimated_vals.at<Vector3>(cur_vel_key);
            prev_bias_ = estimated_vals.at<imuBias::ConstantBias>(cur_bias_key);

            EigenPoseCov pose_cov = marginals.marginalCovariance(cur_pose_key);

            // RCLCPP_INFO(node_->get_logger(), "Pose covariance: %s", sl::toString(pose_cov).c_str();

            prev_state_ = NavState(prev_pose_, prev_vel_);

            Vector3 body_vel = prev_state_.bodyVelocity();

            updateLandmarkValues(estimated_vals, marginals);

            // make message
            auto odom_msg = createOdomMsg(prev_pose_, pose_cov, body_vel, correct_gyro_, cur_img_t, pose_count_);

            prev_img_t_ = cur_img_t;

            pose_count_++;

            // reset local graph and values
            reset_mutex_.unlock();
            return odom_msg;
        }else{
            static bool warned5 = false;
                if (!warned5) {
                    RCLCPP_WARN(node_->get_logger(), "Resetting, waiting for reset to finish");
                    warned5 = true;
                }
            return nullptr;
        }
    }

    void FixedLagBackend::getPoses(EigenPoseMap &container, const unsigned char filter_char)
    {
        for (const auto &key_value : landmark_values_)
        {
            Key landmark_key = key_value.key;
            int landmark_id = Symbol(landmark_key).index();
            container[landmark_id] = key_value.value.cast<Pose3>().matrix();
        }
    }

    void FixedLagBackend::updateLandmarkValues(Values &estimated_vals, Marginals &marginals)
    {

        Values estimated_landmarks = estimated_vals.filter(Symbol::ChrTest(kLandmarkSymbol));

        // iterate through landmarks, and update them to priors
        for (const auto &key_value : estimated_landmarks)
        {
            Key temp_key = key_value.key;
            EigenPoseCov cov = marginals.marginalCovariance(temp_key);
            if(fixed_landmark_.exists(temp_key)){
                continue;
            }
            else if (landmark_values_.exists(temp_key))
            {
                landmark_values_.update(temp_key, key_value.value);
            }
            else
            {
                landmark_values_.insert(temp_key, key_value.value);
            }
            landmark_cov_[temp_key] = cov;
        }
    }

} // namespace tagslam_ros
