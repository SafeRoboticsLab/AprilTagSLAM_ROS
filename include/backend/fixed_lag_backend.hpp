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

 /***************************** fixed_lag_backend.h ************************************
 *
 * Header file of FixedLagBackend base class based on GTSAM::BatchFixedLagSmoother.
 *
 ******************************************************************************/

#ifndef FIXED_LAG_BACKEND_H
#define FIXED_LAG_BACKEND_H

#include <backend/backend.hpp>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam_unstable/nonlinear/BatchFixedLagSmoother.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/linear/linearExceptions.h>

/*
This class implement a FixedLagBackend using the gtsam "IncrementalFixedLagSmoother".

Basically, it is a ISAM2 with a moving window of fixed time. This allows us to 
have bounded runtime and memory usage, even with growing number of poses and landmark. 
Technically, if we do not update the "KeyTimestampMap", this behaves like a normal ISAM2.
However, since this is a module from gtsam_unstable, we implement a separate class for normal isam2.

*/
using namespace gtsam;
namespace tagslam_ros
{

    class FixedLagBackend : public Backend
    {
        public:
            FixedLagBackend(std::shared_ptr<rclcpp::Node> node);
            
            ~FixedLagBackend();

            nav_msgs::msg::Odometry::SharedPtr updateSLAM(TagDetectionArrayPtr landmark_ptr,
                                    EigenPose odom, EigenPoseCov odom_cov);

            nav_msgs::msg::Odometry::SharedPtr updateVIO(TagDetectionArrayPtr landmark_ptr, 
                                    EigenPose odom, EigenPoseCov odom_cov, bool use_odom);

            void getPoses(EigenPoseMap & container, const unsigned char filter_char);

            void reset();

        private:
            void loadMap();

            void updateLandmarkValues(Values& estimated_vals, Marginals & marginals);

        private:
            // declare node_ to keep track for logging and params
            std::shared_ptr<rclcpp::Node> node_;

            // Fixed lag smoother
            // IncrementalFixedLagSmoother isam_;
            bool local_optimal_;
            
            double lag_;
            LevenbergMarquardtParams lm_params_;
            BatchFixedLagSmoother smoother_;
            FixedLagSmoother::KeyTimestampMap newTimestamps_;
            double t0_;
            

    };  // class FactorGraph    
} // namespace backend


#endif