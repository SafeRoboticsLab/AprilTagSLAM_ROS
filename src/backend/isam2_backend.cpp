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

#include "backend/isam2_backend.hpp"

using namespace gtsam;

namespace tagslam_ros
{
    iSAM2Backend::iSAM2Backend(std::shared_ptr<rclcpp::Node> node) :
        Backend(node),
        node_(node)
    {
        // initialize ISAM2
        if(get_ros_option<std::string>(node_, "backend/optimizer", "GaussNewton") == "Dogleg"){
            isam_params_.optimizationParams = ISAM2DoglegParams();
        }
        isam_params_.relinearizeSkip = std::max(get_ros_option<int>(node_, "backend/relinearize_skip", 10), 1);
        isam_params_.cacheLinearizedFactors=get_ros_option<bool>(node_, "backend/cacheLinearizedFactors", true);
        isam_params_.relinearizeThreshold = get_ros_option<double>(node_, "backend/relinearize_threshold", 0.1);
        
        isam_ = ISAM2(isam_params_);
    }

    void iSAM2Backend::reset()
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
        isam_ = ISAM2(isam_params_);
        reset_mutex_.unlock();
        RCLCPP_INFO(node_->get_logger(), "Reset iSAM2");
    }

    nav_msgs::msg::Odometry::SharedPtr iSAM2Backend::updateSLAM(TagDetectionArrayPtr landmark_ptr, EigenPose odom, EigenPoseCov odom_cov)
    {
        if(reset_mutex_.try_lock()){
            int num_landmarks_detected = landmark_ptr->detections.size();
            Key cur_pose_key = Symbol(kPoseSymbol, pose_count_);
            Pose3 cur_pose_init;
            double cur_img_t = rclcpp::Time(landmark_ptr->header.stamp).seconds();
            
            if(initialized_){
                cur_pose_init = addOdomFactor(odom, odom_cov);
            }else if(num_landmarks_detected>0){
                cur_pose_init = initSLAM(cur_img_t);
            }else{
                static bool warned1 = false;
                if (!warned1) {
                    RCLCPP_WARN(node_->get_logger(), "System not initialized, waiting for landmarks");
                    warned1 = true;
                }
                return nullptr;
            }

            addLandmarkFactor<ISAM2>(isam_, landmark_ptr, cur_pose_init);

            // do a isam update
            isam_.update(factor_graph_, initial_estimate_);
            isam_.update();

            // get the current pose and save it for next iteration
            Values current_estimate = isam_.calculateEstimate();
            prev_pose_ = current_estimate.at<Pose3>(cur_pose_key);
            
            // landmark_values_ = current_estimate.filter(Symbol::ChrTest(kLandmarkSymbol));
            updateLandmarkValues(current_estimate);

            // prev_pose_ = isam_.calculateEstimate<Pose3>(cur_pose_key);
            EigenPoseCov pose_cov = isam_.marginalCovariance(cur_pose_key);

            auto odom_msg = createOdomMsg(prev_pose_, pose_cov, Vector3::Zero(), Vector3::Zero(), cur_img_t, pose_count_);
            
            pose_count_++;

            // reset local graph and values
            factor_graph_.resize(0);
            initial_estimate_.clear();
            reset_mutex_.unlock();
            return odom_msg;
        }else{
            static bool warned2 = false;
            if (!warned2) {
                RCLCPP_WARN(node_->get_logger(), "Resetting, waiting for reset to finish");
                warned2 = true;
            }
            return nullptr;
        }
    }

    nav_msgs::msg::Odometry::SharedPtr iSAM2Backend::updateVIO(TagDetectionArrayPtr landmark_ptr, 
                                EigenPose odom, EigenPoseCov odom_cov, bool use_odom)
    {
        if(reset_mutex_.try_lock()){
            int num_landmarks_detected = landmark_ptr->detections.size();

            Key cur_pose_key = Symbol(kPoseSymbol, pose_count_);
            Key cur_vel_key = Symbol(kVelSymbol, pose_count_);
            Key cur_bias_key = Symbol(kBiasSymbol, pose_count_);
            
            Pose3 cur_pose_init;
        
            double cur_img_t = rclcpp::Time(landmark_ptr->header.stamp).seconds();
            
            if(initialized_){
                cur_pose_init = addImuFactor(odom, odom_cov, cur_img_t, use_odom);
            }else if(num_landmarks_detected>0){
                // if the system is not initialized, initialize the slam
                cur_pose_init = initSLAM(cur_img_t);
            }else{
                static bool warned3 = false;
                if (!warned3) {
                    RCLCPP_WARN(node_->get_logger(), "System not initialized, waiting for landmarks");
                    warned3 = true;
                }
                // dump all previous inserted imu measurement
                while(!imu_queue_.empty())
                {
                    sensor_msgs::msg::Imu::SharedPtr imu_msg_ptr;
                    while(!imu_queue_.try_pop(imu_msg_ptr)){}
                    if(rclcpp::Time(imu_msg_ptr->header.stamp).seconds()>=cur_img_t)
                    {
                        break;
                    }
                }
                return nullptr;
            }

            addLandmarkFactor<ISAM2>(isam_, landmark_ptr, cur_pose_init);

            // do a isam update
            isam_.update(factor_graph_, initial_estimate_);
            isam_.update();

            // get the current pose and save it for next iteration
            Values current_estimate = isam_.calculateEstimate();
            prev_pose_ = current_estimate.at<Pose3>(cur_pose_key);
            prev_vel_ = current_estimate.at<Vector3>(cur_vel_key);
            prev_bias_ = current_estimate.at<imuBias::ConstantBias>(cur_bias_key);

            updateLandmarkValues(current_estimate);
            // landmark_values_ = current_estimate.filter(Symbol::ChrTest(kLandmarkSymbol));

            prev_state_ = NavState(prev_pose_, prev_vel_);

            Vector3 body_vel = prev_state_.bodyVelocity();

            EigenPoseCov pose_cov = isam_.marginalCovariance(cur_pose_key);

            auto odom_msg = createOdomMsg(prev_pose_, pose_cov, body_vel, correct_gyro_, cur_img_t, pose_count_);
            
            prev_img_t_ = cur_img_t;
            
            pose_count_++;

            // Reset the preintegration object.
            preint_meas_->resetIntegrationAndSetBias(prev_bias_);

            // reset local graph and values
            factor_graph_.resize(0);
            initial_estimate_.clear();
            reset_mutex_.unlock();
            return odom_msg;
        }else{
            static bool warned4 = false;
            if (!warned4) {
                RCLCPP_WARN(node_->get_logger(), "Resetting, waiting for reset to finish");
                warned4 = true;
            }
            return nullptr;
        }
    }

    
    void iSAM2Backend::getPoses(EigenPoseMap & container, const unsigned char filter_char)
    {   
        Values landmark_values = isam_.calculateEstimate().filter(Symbol::ChrTest(filter_char));
        for(const auto &key_value: landmark_values) {
            Key landmark_key = key_value.key;
            int landmark_id = Symbol(landmark_key).index();
            Pose3 landmark_pose = landmark_values.at<Pose3>(landmark_key);   
            container[landmark_id] = landmark_pose.matrix();
        }
    }

    void iSAM2Backend::updateLandmarkValues(Values &estimated_vals)
    {

        Values estimated_landmarks = estimated_vals.filter(Symbol::ChrTest(kLandmarkSymbol));

        // iterate through landmarks, and update them to priors
        for (const auto &key_value : estimated_landmarks)
        {
            Key temp_key = key_value.key;
            if (landmark_values_.exists(temp_key))
            {
                landmark_values_.update(temp_key, key_value.value);
            }
            else
            {
                landmark_values_.insert(temp_key, key_value.value);
            }
        }
    }


} // namespace backend

