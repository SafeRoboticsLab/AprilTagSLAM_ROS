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

#include "backend/isam2_backend.h"

using namespace gtsam;

namespace tagslam_ros
{
    iSAM2Backend::iSAM2Backend(ros::NodeHandle pnh) :
        Backend(pnh)
    {
        // initialize ISAM2
        ISAM2Params isam_params;
        if(getRosOption<std::string>(pnh, "backend/optimizer", "GaussNewton") == "Dogleg"){
            isam_params.optimizationParams = ISAM2DoglegParams();
        }
        isam_params.relinearizeSkip = std::max(getRosOption<int>(pnh, "backend/relinearize_skip", 10), 1);
        isam_params.cacheLinearizedFactors=getRosOption<bool>(pnh, "backend/cacheLinearizedFactors", true);
        isam_params.relinearizeThreshold = getRosOption<double>(pnh, "backend/relinearize_threshold", 0.1);
        
        isam_ = ISAM2(isam_params);
    }

    iSAM2Backend::~iSAM2Backend()
    {
        // write map and pose to file
        if(save_graph_)
        {
            Values isam_solution = isam_.calculateEstimate();
            write_to_file(isam_solution, save_map_path_);
            std::cout<<"Graph saved to "<<save_map_path_<<std::endl;
        }
    }

    nav_msgs::OdometryPtr iSAM2Backend::updateSLAM(TagDetectionArrayPtr landmark_ptr, EigenPose odom, EigenPoseCov odom_cov)
    {
        int num_landmarks_detected = landmark_ptr->detections.size();
        Key cur_pose_key = Symbol(kPoseSymbol, pose_count_);
        Pose3 cur_pose_init;
        double cur_img_t = landmark_ptr->header.stamp.toSec();
        
        if(initialized_){
            cur_pose_init = addOdomFactor(odom, odom_cov);
        }else if(num_landmarks_detected>0){
            cur_pose_init = initSLAM(cur_img_t);
        }else{
            ROS_WARN_ONCE("System not initialized, waiting for landmarks");
            return nullptr;
        }

        addLandmarkFactor<ISAM2>(isam_, landmark_ptr, cur_pose_init);

        // do a isam update
        isam_.update(factor_graph_, initial_estimate_);
        isam_.update();

        // get the current pose and save it for next iteration
        prev_pose_ = isam_.calculateEstimate<Pose3>(cur_pose_key);
        EigenPoseCov pose_cov = isam_.marginalCovariance(cur_pose_key);

        auto odom_msg = createOdomMsg(prev_pose_, pose_cov, Vector3::Zero(), Vector3::Zero(), cur_img_t, pose_count_);
        
        pose_count_++;

        // reset local graph and values
        factor_graph_.resize(0);
        initial_estimate_.clear();

        return odom_msg;
    }

    nav_msgs::OdometryPtr iSAM2Backend::updateVIO(TagDetectionArrayPtr landmark_ptr, 
                                EigenPose odom, EigenPoseCov odom_cov, bool use_odom)
    {
        int num_landmarks_detected = landmark_ptr->detections.size();

        Key cur_pose_key = Symbol(kPoseSymbol, pose_count_);
        Key cur_vel_key = Symbol(kVelSymbol, pose_count_);
        Key cur_bias_key = Symbol(kBiasSymbol, pose_count_);
        
        Pose3 cur_pose_init;
    
        double cur_img_t = landmark_ptr->header.stamp.toSec();
        
        if(initialized_){
            cur_pose_init = addImuFactor(odom, odom_cov, cur_img_t, use_odom);
        }else if(num_landmarks_detected>0){
            // if the system is not initialized, initialize the slam
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
            return nullptr;
        }

        addLandmarkFactor<ISAM2>(isam_, landmark_ptr, cur_pose_init);

        // do a isam update
        isam_.update(factor_graph_, initial_estimate_);
        isam_.update();

        // get the current pose and save it for next iteration
        prev_pose_ = isam_.calculateEstimate<Pose3>(cur_pose_key);
        prev_vel_ = isam_.calculateEstimate<Vector3>(cur_vel_key);
        prev_bias_ = isam_.calculateEstimate<imuBias::ConstantBias>(cur_bias_key);
        prev_state_ = NavState(prev_pose_, prev_vel_);

        EigenPoseCov pose_cov = isam_.marginalCovariance(cur_pose_key);

        auto odom_msg = createOdomMsg(prev_pose_, pose_cov, prev_vel_, correct_gyro_, cur_img_t, pose_count_);
        
        prev_img_t_ = cur_img_t;
        
        pose_count_++;

        // Reset the preintegration object.
        preint_meas_->resetIntegrationAndSetBias(prev_bias_);

        // reset local graph and values
        factor_graph_.resize(0);
        initial_estimate_.clear();

        return odom_msg;
    }

    
    void iSAM2Backend::getPoses(EigenPoseMap & container, const unsigned char filter_char)
    {   
        Values landmark_values = isam_.calculateEstimate().filter(Symbol::ChrTest(filter_char));
        for(const auto key_value: landmark_values) {
            Key landmark_key = key_value.key;
            int landmark_id = Symbol(landmark_key).index();
            Pose3 landmark_pose = landmark_values.at<Pose3>(landmark_key);   
            container[landmark_id] = landmark_pose.matrix();
        }
    }


} // namespace backend

