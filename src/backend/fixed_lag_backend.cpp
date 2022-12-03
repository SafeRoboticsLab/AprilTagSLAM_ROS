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
#include "backend/fixed_lag_backend.h"

using namespace gtsam;

namespace tagslam_ros
{
    FixedLagBackend::FixedLagBackend(ros::NodeHandle pnh) : 
        Backend(pnh)
    {
        LevenbergMarquardtParams lm_params;
        lm_params.lambdaInitial = getRosOption<double>(pnh, "backend/lambda_initial", 1e-5);
        lm_params.lambdaUpperBound = getRosOption<double>(pnh, "backend/lambda_upper_bound", 1e5);
        lm_params.lambdaLowerBound = getRosOption<double>(pnh, "backend/lambda_lower_bound", 0);
        lm_params.lambdaFactor = getRosOption<double>(pnh, "backend/lambda_factor", 10.0);
        lm_params.maxIterations = getRosOption<int>(pnh, "backend/max_iterations", 100);
        lm_params.errorTol = getRosOption<double>(pnh, "backend/error_tol", 1e-5);
        lm_params.relativeErrorTol = getRosOption<double>(pnh, "backend/relative_error_tol", 1e-4);
        lm_params.absoluteErrorTol = getRosOption<double>(pnh, "backend/absolute_error_tol", 1e-5);
        bool local_optimal = getRosOption<bool>(pnh, "backend/local_optimal", false);
        
        double lag = getRosOption<double>(pnh, "backend/lag", 1.0);
        smoother_ = BatchFixedLagSmoother(lag, lm_params, true, local_optimal);
    }

    FixedLagBackend::~FixedLagBackend()
    {
        // write map and pose to file
        if (save_graph_){
            Values smoother_solution = smoother_.calculateEstimate();
            write_to_file(smoother_solution, save_map_path_);
            std::cout << "Graph saved to " << save_map_path_ << std::endl;
        }
    }

    EigenPose FixedLagBackend::updateSLAM(TagDetectionArrayPtr landmark_ptr, EigenPose odom, EigenPoseCov odom_cov)
    {
        int num_landmarks_detected = landmark_ptr->detections.size();
        Key cur_pose_key = Symbol(kPoseSymbol, pose_count_);
        Pose3 cur_pose_init;
        double cur_img_t = landmark_ptr->header.stamp.toSec();

        if (initialized_){
            cur_pose_init = addOdomFactor(odom, odom_cov);
        }else if (num_landmarks_detected > 0){
            cur_pose_init = initSLAM(cur_img_t);
        }else{
            ROS_WARN_ONCE("System not initialized, waiting for landmarks");
            return EigenPose::Zero();
        }

        // create new timesampe map for the current pose
        newTimestamps_[cur_pose_key] = cur_img_t;

        // add prior factor for landmarks
        addLandmarkFactor<BatchFixedLagSmoother>(smoother_, landmark_ptr, cur_pose_init);

        // update landmark timestamp
        for (auto &landmark : landmark_ptr->detections){
            Key landmark_key = Symbol(kLandmarkSymbol, landmark.id);
            newTimestamps_[landmark_key] = cur_img_t;
        }

        // do a batch optimization
        smoother_.update(factor_graph_, initial_estimate_, newTimestamps_);

        // get the current pose and save it for next iteration
        Values estimated_vals = smoother_.calculateEstimate();
        prev_pose_ = estimated_vals.at<Pose3>(cur_pose_key);

        updateLandmarkValues(estimated_vals);

        pose_count_++;

        // reset local graph and values
        factor_graph_.resize(0);
        initial_estimate_.clear();
        newTimestamps_.clear();

        return prev_pose_.matrix();
    }

    EigenPose FixedLagBackend::updateVIO(TagDetectionArrayPtr landmark_ptr)
    {
        int num_landmarks_detected = landmark_ptr->detections.size();
        
        Key cur_pose_key = Symbol(kPoseSymbol, pose_count_);
        Key cur_vel_key = Symbol(kVelSymbol, pose_count_);
        Key cur_bias_key = Symbol(kBiasSymbol, pose_count_);

        Pose3 cur_pose_init;
        
        double cur_img_t = landmark_ptr->header.stamp.toSec();

        if (initialized_){
            cur_pose_init = addImuFactor(cur_img_t);
        }else if (num_landmarks_detected > 0){
            cur_pose_init = initSLAM(cur_img_t);
        }else{
            ROS_WARN_ONCE("System not initialized, waiting for landmarks");
            // dump all previous inserted imu measurement
            while(!imu_queue_.empty())
            {
                sensor_msgs::ImuPtr imu_msg_ptr;
                while(!imu_queue_.try_pop(imu_msg_ptr)){}
                if(imu_msg_ptr->header.stamp.toSec()>=cur_img_t)
                {
                    break;
                }
            }
            return EigenPose::Zero ();
        }

        // create new timesampe map for the current pose
        newTimestamps_[cur_pose_key] = cur_img_t;
        newTimestamps_[cur_vel_key] = cur_img_t;
        newTimestamps_[cur_bias_key] = cur_img_t;

        // add prior factor for landmarks
        addLandmarkFactor<BatchFixedLagSmoother>(smoother_, landmark_ptr, cur_pose_init);

        // update landmark timestamp
        for (auto &landmark : landmark_ptr->detections){
            Key landmark_key = Symbol(kLandmarkSymbol, landmark.id);
            newTimestamps_[landmark_key] = cur_img_t;
        }

        // do a batch optimization
        smoother_.update(factor_graph_, initial_estimate_, newTimestamps_);
        
        // get the current pose and save it for next iteration
        Values estimated_vals = smoother_.calculateEstimate();
        prev_pose_ = estimated_vals.at<Pose3>(cur_pose_key);
        prev_vel_ = estimated_vals.at<Vector3>(cur_vel_key);
        prev_bias_ = estimated_vals.at<imuBias::ConstantBias>(cur_bias_key);

        prev_state_ = NavState(prev_pose_, prev_vel_);

        updateLandmarkValues(estimated_vals);

        prev_img_t_ = cur_img_t;

        pose_count_++;

        // Reset the preintegration object.
        preint_meas_->resetIntegrationAndSetBias(prev_bias_);

        // reset local graph and values
        factor_graph_.resize(0);
        initial_estimate_.clear();
        newTimestamps_.clear();

        return prev_pose_.matrix();
    }

    void FixedLagBackend::getPoses(EigenPoseMap &container, const unsigned char filter_char){
        for (const auto key_value : landmark_values_){
            Key landmark_key = key_value.key;
            int landmark_id = Symbol(landmark_key).index();
            container[landmark_id] = key_value.value.cast<Pose3>().matrix();
        }
    }

    void FixedLagBackend::updateLandmarkValues(Values& estimated_vals)
    {
        Values estimated_landmarks = estimated_vals.filter(Symbol::ChrTest(kLandmarkSymbol));
        // iterate through landmarks, and update them to priors
        for(const auto key_value: estimated_landmarks) {
            Key temp_key = key_value.key;
            // EigenPoseCov cov = smoother_.marginalCovariance(temp_key);
            if(landmark_values_.exists(temp_key))
            {
                landmark_values_.update(temp_key, key_value.value);                
            }else{
                landmark_values_.insert(temp_key, key_value.value);
            }
            // landmark_cov_[temp_key] = cov;
        }
    }

} // namespace backend
