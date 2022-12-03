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

 /***************************** backend.h ************************************
 *
 * Header file of Backend base class.
 *
 ******************************************************************************/

#ifndef backend_h
#define backend_h

#include "utility_function.h"

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
    struct FullState
    {
        EigenPose pose;
        EigenPoseCov pose_cov;
    };
    

    class Backend
    {
        
        public:
            Backend(ros::NodeHandle pnh);
            
            virtual ~Backend()
            {
                std::cout<<"Default deconstructor of Backend is called."<<std::endl;
            };

            /*
            Update the smoother with landmark detections and odometry measurement.
            */
            virtual EigenPose updateSLAM(TagDetectionArrayPtr landmark_ptr, EigenPose odom, EigenPoseCov odom_cov) = 0;

            /*
            Update the smoother with landmark detections and preintegrated IMU factor
            */
            virtual EigenPose updateVIO(TagDetectionArrayPtr landmark_ptr) = 0;

            void setupIMU(double accel_noise_sigma, double accel_bias_rw_sigma,
                            double gyro_noise_sigma, double gyro_bias_rw_sigma, EigenPose T_sensor2cam);

            /*
            Add IMU message to the queue
            */
            void updateIMU(sensor_msgs::ImuPtr imu_msg_ptr);

            /*
            Retrive the smoothed poses from the smoother as Eigen::Matrix4d, and save it to the given container.
            */
            virtual void getPoses(EigenPoseMap & container, const unsigned char filter_char) = 0;
        
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
            Template member function to add landmark factors into factor graph
            */
            template <typename T>
            void addLandmarkFactor(T & smoother, 
                                    TagDetectionArrayPtr landmark_ptr,
                                    Pose3 & cur_pose_init)
            {
                // auto t0 = std::chrono::system_clock::now();
                Key cur_pose_key = Symbol(kPoseSymbol, pose_count_);
                
                // add prior factor for landmarks
                for (auto &landmark : landmark_ptr->detections){
                    Key landmark_key = Symbol(kLandmarkSymbol, landmark.id);
                    // landmark.pose is geometry_msgs::Pose
                    Pose3 landmark_factor = Pose3(getTransform(landmark.pose));
                    auto landmark_noise = noiseModel::Gaussian::Covariance(landmark_factor_cov_);

                    // add prior factor to the local graph
                    factor_graph_.emplace_shared<BetweenFactor<Pose3>>(
                            cur_pose_key, landmark_key, landmark_factor, landmark_noise);

                    if (!smoother.valueExists(landmark_key)){
                        // check if the landmark is in the prior map
                        if (landmark_values_.exists(landmark_key)){
                            // if the landmark is in the prior map, we add it to the initial estimate
                            Pose3 landmark_prior = landmark_values_.at<Pose3>(landmark_key);
                            initial_estimate_.insert(landmark_key, landmark_prior);

                            // insert prior factor to the graph
                            auto landmark_prior_noise = noiseModel::Gaussian::Covariance(landmark_prior_cov_);
                            factor_graph_.add(PriorFactor<Pose3>(landmark_key, landmark_prior, landmark_prior_noise));

                            // Equality Constraints can only use QR solver It is too slow
                            //factor_graph_.add(NonlinearEquality1<Pose3>(landmark_prior, landmark_key));
                        }else{
                            // if this a new landmark, we add it by reprojecting the landmark from the current predicted pose
                            Pose3 landmark_init = cur_pose_init * landmark_factor;
                            initial_estimate_.insert(landmark_key, landmark_init);
                        }
                    }
                }

                // auto t1 = std::chrono::system_clock::now();
                // auto d0 = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
                // ROS_INFO_STREAM("addOdomFactor takes "<<d0.count());
            }
        
        private:
            /*
            set the gravity vector for preint_param_ using the imu message
            */
            void setGravity(sensor_msgs::ImuPtr imu_msg_ptr);

            /*
            loda map from load_map_path_
            */
            void loadMap();

        protected:
            // system parameters
            bool initialized_;
            bool prior_map_; // if true, the system will use prior factor to initialize the map
            bool save_graph_; // if true, the system will save the factor graph to a file

            // file path to the map file
            // if the map file is not empty, the system will load the map from the file
            // the robot will localize or do SLAM based using the loaded map as prior
            std::string load_map_path_;
            std::string save_map_path_;

            // Factor graph and inital value of each variables
            NonlinearFactorGraph factor_graph_;
            Values initial_estimate_;

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
            tbb::concurrent_queue <sensor_msgs::ImuPtr> imu_queue_;
            boost::shared_ptr<PreintegrationCombinedParams> preint_param_;
            std::shared_ptr<PreintegratedCombinedMeasurements> preint_meas_ = nullptr;
            
            // values to track of previous states
            Pose3 prev_pose_;
            Vector3 prev_vel_;
            NavState prev_state_;
            imuBias::ConstantBias prev_bias_; // assume zero initial bias
            double prev_img_t_ = 0.0; 
            
    };
    
}



#endif