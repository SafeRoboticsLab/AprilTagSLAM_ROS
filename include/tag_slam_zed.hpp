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

#include "utility_function.hpp"
#include "frontend/tag_detector.hpp"
#include "frontend/tag_detector_cpu.hpp"

#include "backend/backend.hpp"
#include "backend/fixed_lag_backend.hpp"
#include "backend/isam2_backend.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/distortion_models.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <string>

#ifndef NO_CUDA_OPENCV
    #include <opencv2/core/cuda.hpp>
    #include <opencv2/cudaimgproc.hpp>
#endif

using Trigger = std_srvs::srv::trigger; 

#ifndef NO_ZED
    // zed include
    #include <sl/Camera.hpp>
    
    // Note: ZED SDK depends on CUDA,
    // therefore if we can have zed sdk, then cuda detector will run
    #include "frontend/tag_detector_cuda.hpp"// Topics

    namespace tagslam_ros {
    class TagSlamZED : public rclcpp::Node {

    public:
        /*! \brief Default constructor
    */
        explicit TagSlamZED(std::string name = "tag_slam_zed_node", const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

        /*! \brief \ref destructor
    */
        ~TagSlamZED();

    protected:

        /*! \brief Reads parameters from the param server
    */
        void read_parameters();

        /*! \brief Set up ros publisher
        */
        void setup_publisher();

        /*! \brief Set up ros publisher
        */
        void setup_service();
        
        /*! \brief Set up ros dynamic reconfigure
        */
        void setup_dynamic_reconfig();

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
        void publish_images(TagDetectionArrayPtr static_tag_array_ptr, TagDetectionArrayPtr dyn_tag_array_ptr);

        /*! \brief publish the ros detection array
        */
        void publish_detection_array(TagDetectionArrayPtr static_tag_array_ptr, TagDetectionArrayPtr dyn_tag_array_ptr);

    private: 
        void estimate_state(TagDetectionArrayPtr tag_array_ptr);

        void check_resol_fps();

        void sl_mat_to_ros_msg(sensor_msgs::msg::Image::SharedPtr img_msg_ptr, sl::Mat img, std_msgs::msg::Header header);

        void fill_camera_info(sensor_msgs::msg::CameraInfo::SharedPtr CamInfoMsg, sl::CalibrationParameters zedParam);

        rclcpp::Time sl_time_to_ros(sl::Timestamp t);

        EigenPose sl_trans_to_eigen(sl::Transform& pose);

        EigenPose sl_pose_to_eigen(sl::Pose& pose);

        EigenPoseSigma sl_pose_to_sigma(sl::Pose& pose);

        EigenPoseCov sl_pose_to_cov(sl::Pose& pose);

        cv::Mat sl_mat_to_cv_mat(sl::Mat& input);

        void reset_callback(Trigger::Request::ConstSharedPtr request, Trigger::Response::SharedPtr response)
        {
            (void)request; // Avoid unused variable warning
            run_slam_ = false;
            // wait for one second to allow the slam thread to finish current computation
            slam_backend_->reset();
            response->success = true;
            response->message = "Reset slam.";
        }

        void start_callback(Trigger::Request::ConstSharedPtr request, Trigger::Response::SharedPtr response)
        {
            run_slam_ = true;
            response->success = true;
            response->message = "Start slam.";
            RCLCPP_INFO(this->get_logger(), "SLAM Started.");
        }

        void stop_callback(Trigger::Request::ConstSharedPtr request, Trigger::Response::SharedPtr response)
        {
            run_slam_ = false;
            response->success = true;
            response->message = "Stop slam.";
            RCLCPP_INFO(this->get_logger(), "SLAM Stopped.");
        }

#ifndef NO_CUDA_OPENCV
        cv::cuda::GpuMat sl_mat_to_cv_mat_gpu(sl::Mat& input);
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
        sl::DEPTH_MODE zed_sensing_mode_;
        // double zed_sensor_rate_ = 400.0;
        double zed_min_depth_;
        double zed_max_depth_;
        int zed_exposure_;

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
        std::thread cam_thread_; // camera data thread
        std::thread sens_thread_; // Sensors data thread

        // Publishers
        bool if_pub_image_;
        bool if_pub_tag_det_;
        bool if_pub_tag_det_image_;
        bool if_pub_landmark_;
        bool if_pub_latency_;
        image_transport::CameraPublisher img_pub_; //
        image_transport::Publisher det_img_pub_; //

        rclcpp::Publisher<AprilTagDetectionArray>::SharedPtr static_tag_det_pub_;
        rclcpp::Publisher<AprilTagDetectionArray>::SharedPtr dyn_tag_det_pub_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr slam_pose_pub_;
        rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr landmark_pub_;

        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr debug_convert_pub_;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr debug_det_pub_;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr debug_opt_pub_;
        rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr debug_total_pub_;

        // Services
        rclcpp::Service<Trigger>::SharedPtr srv_start_slam_;
        rclcpp::Service<Trigger>::SharedPtr srv_stop_slam_;
        rclcpp::Service<Trigger>::SharedPtr srv_reset_slam_;

        /*
        *** SLAM Parameters ****
        */
        // front end
        bool use_gpu_detector_;
        std::unique_ptr<TagDetector> tag_detector_;
        sl::VIEW zed_image_type_;

        // back end
        std::string backend_type_;
        std::unique_ptr<Backend> slam_backend_;
        bool detection_only_ = false;
        bool use_imu_odom_ = false;
        EigenPose pose_prev_ = EigenPose::Identity();
        
        // Camera info
        sensor_msgs::msg::CameraInfo::SharedPtr cam_info_msg_ptr_;

        // ROS services parameters
        std::atomic<bool> run_slam_ = false;

    };
    } 
#else
    namespace tagslam_ros {
    class TagSlamZED : public rclcpp::Node {

    public:
        
        TagSlamZED() = default;

        ~TagSlamZED() = default;

        void onInit()
        {
            RCLCPP_ERROR(this->get_logger(), "TagSlamZED not built. Use '-DUSE_ZED=ON' with catkin");
            rclcpp::shutdown();
        }
    };
    }
#endif // NO_ZED
#endif // TAG_SLAM_ZED_H
