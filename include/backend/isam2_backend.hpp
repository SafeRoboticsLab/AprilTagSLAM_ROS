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

 /***************************** isam2_backend.h ************************************
 *
 * Header file of iSAM2Backend base class based on GTSAM::ISAM2.
 * 
 * This class implement a standard iSAM2 backend for landmark slam.
 *  https://www.cs.cmu.edu/~kaess/pub/Kaess12ijrr.pdf

 *
 ******************************************************************************/

#ifndef isam2_backend_h
#define isam2_backend_h

#include <backend/backend.hpp>
#include <gtsam/nonlinear/ISAM2.h>

using namespace gtsam;
namespace tagslam_ros
{

    class iSAM2Backend : public Backend
    {
        public:
            iSAM2Backend(std::shared_ptr<rclcpp::Node> node);
            
            ~iSAM2Backend();

            nav_msgs::msg::Odometry::SharedPtr updateSLAM(TagDetectionArrayPtr landmark_ptr, 
                                    EigenPose odom, EigenPoseCov odom_cov); 

            // EigenPose updateVIO(TagDetectionArrayPtr landmark_ptr);

            nav_msgs::msg::Odometry::SharedPtr updateVIO(TagDetectionArrayPtr landmark_ptr, 
                                    EigenPose odom, EigenPoseCov odom_cov, bool use_odom);

            void getPoses(EigenPoseMap & container, const unsigned char filter_char);

            void setupIMU(double accel_noise_sigma, double accel_bias_rw_sigma,
                        double gyro_noise_sigma, double gyro_bias_rw_sigma, 
                        EigenPose T_sensor2cam);

            void updateIMU(sensor_msgs::msg::Imu::SharedPtr imu_msg_ptr);

            void reset();

        private:
            void loadMap();

            void updateLandmarkValues(Values& estimated_vals);

        private:
            // declare node_ to keep track for logging and params
            std::shared_ptr<rclcpp::Node> node_; 
            
            // The ISAM2 smoother
            ISAM2Params isam_params_;
            ISAM2 isam_;
            
    };  // class FactorGraph    
} // namespace backend


#endif