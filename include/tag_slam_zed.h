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

 /***************************** tag_slam_zed.h ************************************
 *
 * Header file of TagSlamZED class which estimate the camera's pose once 
 * received camera image and odometry information through ZED SDK.
 *
 ******************************************************************************/


#ifndef TAG_SLAM_ZED_H
#define TAG_SLAM_ZED_H



// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/cvconfig.h>

#include "utility_function.h"
#include "frontend/tag_detector.h"
#include "frontend/tag_detector_cpu.h"



#include "backend/backend.h"
#include "backend/fixed_lag_backend.h"
#include "backend/isam2_backend.h"

#include <image_transport/image_transport.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/distortion_models.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include <pluginlib/class_list_macros.h>

#ifndef NO_CUDA_OPENCV
    #include <opencv2/core/cuda.hpp>
    #include <opencv2/cudaimgproc.hpp>
#endif

#ifndef NO_ZED
    // zed include
    #include <sl/Camera.hpp>
    
    // Note: ZED SDK depends on CUDA,
    // therefore if we can have zed sdk, then cuda detector will run
    #include "frontend/tag_detector_cuda.h"// Topics

    namespace tagslam_ros {
    class TagSlamZED : public nodelet::Nodelet {

    public:
        /*! \brief Default constructor
    */
        TagSlamZED();

        /*! \brief \ref destructor
    */
        ~TagSlamZED();

    protected:
        /*! \brief Initialization function called by the Nodelet base class
    */
        void onInit();

        /*! \brief Reads parameters from the param server
    */
        void readParameters();

        /*! \brief Set up ros publisher
        */
        void setup_publisher();

        /*! \brief Set up ros dynamic reconfigure
        */
        void setup_dynmaic_reconfig();

        /*! \brief Turn on the zed camera
        */
        void turn_on_zed();

        /*! \brief ZED camera polling thread function
        */
        void gpu_image_thread_func();

        /*! \brief ZED camera polling thread function
        */
        void cpu_image_thread_func();

        /*! \brief Sensors data publishing function
        */
        void sensors_thread_func();

        /*! \brief publish the ros image
        */
        void publishImages(TagDetectionArrayPtr static_tag_array_ptr, TagDetectionArrayPtr dyn_tag_array_ptr);

    private: 
        void estimateState(TagDetectionArrayPtr tag_array_ptr);

        void checkResolFps();

        void slMatToROSmsg(sensor_msgs::ImagePtr imgMsgPtr, sl::Mat img, std_msgs::Header header);

        void fillCameraInfo(sensor_msgs::CameraInfoPtr CamInfoMsg, sl::CalibrationParameters zedParam);

        ros::Time slTime2Ros(sl::Timestamp t);

        EigenPose slTrans2Eigen(sl::Transform& pose);

        EigenPose slPose2Eigen(sl::Pose& pose);

        EigenPoseSigma slPose2Sigma(sl::Pose& pose);

        EigenPoseCov slPose2Cov(sl::Pose& pose);

        cv::Mat slMat2cvMat(sl::Mat& input);

#ifndef NO_CUDA_OPENCV
        cv::cuda::GpuMat slMat2cvMatGPU(sl::Mat& input);
#endif

        int getOCVtype(sl::MAT_TYPE type);
    private:
        uint64_t frame_count_ = 0;

        /*
        *** ZED Parameters ****
        */
        // Launch file parameters
        sl::RESOLUTION zed_resol_;
        int zed_frame_rate_;
        sl::DEPTH_MODE zed_depth_mode_;
        sl::SENSING_MODE zed_sensing_mode_;
        // double zed_sensor_rate_ = 400.0;
        double zed_min_depth_;
        double zed_max_depth_;

        sl::InitParameters zed_init_param_;
        sl::MODEL zed_user_model_;
        sl::MODEL zed_real_model_;
        sl::Camera zed_camera_;

        // Positional tracking
        bool zed_pos_tracking_enabled_ = false;

        sl::RuntimeParameters zed_runtime_param_;
        
        /*
        *** ROS Parameters ****
        */
        ros::NodeHandle nh_;
        ros::NodeHandle pnh_;
        std::thread cam_thread_; // camera data thread
        std::thread sens_thread_; // Sensors data thread

        // Publishers
        bool if_pub_image_;
        bool if_pub_tag_det_;
        bool if_pub_tag_det_image_;
        image_transport::CameraPublisher img_pub_; //
        image_transport::Publisher det_img_pub_; //

        ros::Publisher static_tag_det_pub_;
        ros::Publisher dyn_tag_det_pub_;
        // ros::Publisher zed_pose_pub_;
        ros::Publisher slam_pose_pub_;
        ros::Publisher imu_pub_;
        // ros::Publisher imu_pub_Raw;

        // Services
        ros::ServiceServer mSrvSetInitPose;
        ros::ServiceServer mSrvResetOdometry;
        ros::ServiceServer mSrvResetTracking;

        bool mStopNode = false;

        /*
        *** SLAM Parameters ****
        */
        // front end
        bool use_gpu_detector_;
        std::unique_ptr<TagDetector> tag_detector_;
        sl::VIEW zed_imge_type_;

        // back end
        std::string backend_type_;
        std::unique_ptr<Backend> slam_backend_;
        bool detection_only_ = false;
        bool use_imu_odom_ = false;
        EigenPose pose_prev_ = EigenPose::Identity();
        
        // Camera info
        sensor_msgs::CameraInfoPtr cam_info_msg_;

        // Thread Sync
        std::mutex mCloseZedMutex;
        std::mutex mCamDataMutex;
        std::mutex mPcMutex;
        std::mutex mRecMutex;
        std::mutex mPosTrkMutex;
        std::mutex mDynParMutex;
        std::mutex mMappingMutex;
        std::mutex mObjDetMutex;
        std::condition_variable mPcDataReadyCondVar;
        bool mPcDataReady;

    };
    } 
#else
    namespace tagslam_ros {
    class TagSlamZED : public nodelet::Nodelet {

    public:
        
        TagSlamZED() = default;

        ~TagSlamZED() = default;

        void onInit()
        {
            NODELET_ERROR("TagSlamZED not built. Use '-DUSE_ZED=ON' with catkin");
            ros::shutdown();
        }
    };
    }
#endif // NO_ZED
#endif // TAG_SLAM_ZED_H
