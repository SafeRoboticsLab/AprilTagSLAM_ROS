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

#include "tag_slam_zed.h"

RCLCPP_COMPONENTS_REGISTER_NODE(tagslam_ros::TagSlamZED)

#ifndef NO_ZED
namespace tagslam_ros {
    #ifndef DEG2RAD
    #define DEG2RAD 0.017453293
    #define RAD2DEG 57.295777937
    #endif

    #define MAG_FREQ 50.
    #define BARO_FREQ 25.

    // Basic structure to compare timestamps of a sensor. Determines if a specific sensor data has been updated or not.
    struct TimestampHandler {

        // Compare the new timestamp to the last valid one. If it is higher, save it as new reference.
        inline bool isNew(sl::Timestamp& ts_curr, sl::Timestamp& ts_ref) {
            bool new_ = ts_curr > ts_ref;
            if (new_) ts_ref = ts_curr;
            return new_;
        }
        // Specific function for IMUData.
        inline bool isNew(sl::SensorsData::IMUData& imu_data) {
            return isNew(imu_data.timestamp, ts_imu);
        }
        // Specific function for MagnetometerData.
        inline bool isNew(sl::SensorsData::MagnetometerData& mag_data) {
            return isNew(mag_data.timestamp, ts_mag);
        }
        // Specific function for BarometerData.
        inline bool isNew(sl::SensorsData::BarometerData& baro_data) {
            return isNew(baro_data.timestamp, ts_baro);
        }

        sl::Timestamp ts_imu = 0, ts_baro = 0, ts_mag = 0; // Initial values
    };

    TagSlamZED::TagSlamZED(std::string name, const rclcpp::NodeOptions &options)
        : rclcpp::Node(name, options)
    {   
        RCLCPP_INFO(
            this->get_logger(), 
            "********** Starting node '%s' **********", 
            this->get_name());

        read_parameters();
        turn_on_zed();
        setup_publisher();
        setup_service();
        
        // Start pool thread
        if(use_gpu_detector_){
            cam_thread_ = std::thread(&TagSlamZED::gpu_image_thread_func, this);
        }else{
            cam_thread_ = std::thread(&TagSlamZED::cpu_image_thread_func, this);
        }

        // Start Sensors thread
        if(use_imu_odom_)
            sens_thread_ = std::thread(&TagSlamZED::sensors_thread_func, this);

        RCLCPP_INFO(this->get_logger(), "TagSlamZED node initialized.");
    }

    TagSlamZED::~TagSlamZED()
    {
        if (cam_thread_.joinable()) {
            cam_thread_.join();
        }

        if (sens_thread_.joinable()) {
            sens_thread_.join();
        }

        // close the camera
        zed_camera_.close();

        RCLCPP_INFO(this->get_logger(), "Tag Slam ZED Node destroyed");
    }


    void TagSlamZED::setup_service()
    {
        // Set the service to reset the map
        srv_reset_slam_ = this->create_service<Trigger>(
            "reset_slam", std::bind(&TagSlamZED::reset_callback, this, std::placeholders::_1, std::placeholders::_2));
        
        srv_start_slam_ = this->create_service<Trigger>(
            "start_slam", std::bind(&TagSlamZED::start_callback, this, std::placeholders::_1, std::placeholders::_2));

        srv_stop_slam_ = this->create_service<Trigger>(
            "stop_slam", std::bind(&TagSlamZED::stop_callback, this, std::placeholders::_1, std::placeholders::_2));
    }

    void TagSlamZED::setup_publisher()
    {
        // Create all the publishers

        // Image publishers
        if (if_pub_image_)
            img_pub_ = image_transport::create_camera_publisher(this, "Image");

        if (if_pub_tag_det_image_)
            det_img_pub_ = image_transport::create_publisher(this, "Tag_Detection_Image");

        if (if_pub_tag_det_) {
            static_tag_det_pub_ = this->create_publisher<AprilTagDetectionArray>("Tag_Detections", rclcpp::QoS(1));
            dyn_tag_det_pub_ = this->create_publisher<AprilTagDetectionArray>("Tag_Detections_Dynamic", rclcpp::QoS(1));
        }

        if (!detection_only_)
            slam_pose_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("Pose", rclcpp::QoS(1));

        // IMU Publishers
        if (use_imu_odom_)
            imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>("IMU/Data", rclcpp::QoS(1));

        // Landmark publisher
        if (if_pub_landmark_)
            landmark_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("Landmarks", rclcpp::QoS(1));

        // Debug latency publishers
        if (if_pub_latency_)
        {
            debug_convert_pub_ = this->create_publisher<std_msgs::msg::Float32>("Debug/Convert", rclcpp::QoS(1));
            debug_det_pub_ = this->create_publisher<std_msgs::msg::Float32>("Debug/Detect", rclcpp::QoS(1));
            debug_opt_pub_ = this->create_publisher<std_msgs::msg::Float32>("Debug/Optimize", rclcpp::QoS(1));
            debug_total_pub_ = this->create_publisher<std_msgs::msg::Float32>("Debug/Total", rclcpp::QoS(1));
        }
    }

    void TagSlamZED::read_parameters(){

        RCLCPP_INFO(this->get_logger(), "*** GENERAL PARAMETERS ***");

        /*
        ************** Zed Camera Parameters **************
        */
       // Declare parameters with defaults
        this->declare_parameter<std::string>("camera/camera_model", "zed2");
        this->declare_parameter<int>("camera/exposure", 20);
        this->declare_parameter<int>("camera/resolution", 3);
        this->declare_parameter<int>("camera/frame_rate", 30); // default to 30fps
        this->declare_parameter<std::string>("frontend/type", "CPU");
        this->declare_parameter<bool>("backend/use_imu", true);
        this->declare_parameter<bool>("backend/use_odom", true);
        this->declare_parameter<std::string>("backend/smoother", "isam2");
        this->declare_parameter<int>("depth/quality", 1);
        this->declare_parameter<int>("depth/sensing_mode", 0);
        this->declare_parameter<double>("depth/min_depth", 0.5);
        this->declare_parameter<double>("depth/max_depth", 15.0);
        this->declare_parameter<bool>("publish/publish_tags", true);
        this->declare_parameter<bool>("publish/publish_image_with_tags", true);
        this->declare_parameter<bool>("publish/publish_image", true);
        this->declare_parameter<bool>("publish/publish_landmarks", true);
        this->declare_parameter<bool>("publish/publish_latency", true);

        // Get parameters from param files
        std::string camera_model;
        this->get_parameter("camera/camera_model", camera_model);

        if (camera_model == "zed") {
        zed_user_model_ = sl::MODEL::ZED;
        } else if (camera_model == "zedm") {
            zed_user_model_ = sl::MODEL::ZED_M;
        } else if (camera_model == "zed2") {
            zed_user_model_ = sl::MODEL::ZED2;
        } else if (camera_model == "zed2i") {
            zed_user_model_ = sl::MODEL::ZED2i;
        } else {
            RCLCPP_ERROR(this->get_logger(), "Camera model not valid: %s", camera_model.c_str());
        }
        RCLCPP_INFO(this->get_logger(), " * Camera Model by param -> %s", camera_model.c_str());


        this->get_parameter("camera/exposure", zed_exposure_);

        int resol;
        this->get_parameter("camera/resolution", resol);
        zed_resol_ = static_cast<sl::RESOLUTION>(resol);
        RCLCPP_INFO(this->get_logger(), " * Camera Resolution        -> %s", sl::toString(zed_resol_).c_str());

        this->get_parameter("camera/frame_rate", zed_frame_rate_);
        check_resol_fps();
        RCLCPP_INFO(this->get_logger(), " * Camera Grab Framerate    -> %d", zed_frame_rate_);

        /*
        ************** Frontend Setup **************
        */
        std::string frontend_type;
        this->get_parameter("frontend/type", frontend_type);
        use_gpu_detector_ = (frontend_type == "GPU");

        // GPU detector take a recitified RGBA8 image
            // ZED returns a BGRA8 image, we need to convert it later
        zed_image_type_ = use_gpu_detector_ ? sl::VIEW::LEFT : sl::VIEW::LEFT_GRAY;
        tag_detector_ = use_gpu_detector_ ? std::make_unique<TagDetectorCUDA>(this) : std::make_unique<TagDetectorCPU>(this);
        RCLCPP_INFO(this->get_logger(), " * %s", use_gpu_detector_ ? "Use GPU tag detector" : "Use CPU tag detector");

        /*
        *********** Backend Setup **************
        */
        // std::string odom_type
        // this->get_parameter("backend/odom", "vision");

        this->get_parameter("backend/use_imu", use_imu_odom_);
        this->get_parameter("backend/use_odom", zed_pos_tracking_enabled_);
        
        if(!use_imu_odom_ && !zed_pos_tracking_enabled_){
            RCLCPP_WARN(this->get_logger(), "No odometry source is enabled, please enable at least one of them. Running in detection only mode.");
            detection_only_ = true;
        }

        this->get_parameter("backend/smoother", backend_type_);
        
        if (backend_type_ =="isam2") {
            slam_backend_ = std::make_unique<iSAM2Backend>(this);
            RCLCPP_INFO(this->get_logger(), "Using iSAM2 backend.");
        } else if (backend_type_ == "fixed_lag"){
            slam_backend_ = std::make_unique<FixedLagBackend>(this);
            RCLCPP_INFO(this->get_logger(), "Using fixed-lag backend.");
        } else if (backend_type_ == "none"){
            slam_backend_ = nullptr;
            detection_only_ = true;
            use_imu_odom_ = false;
            zed_pos_tracking_enabled_ = false;
            RCLCPP_INFO(this->get_logger(), "Apriltag Detector Mode.");
        } else{
            RCLCPP_ERROR(this->get_logger(), "Not supported backend type: %s", backend_type_.c_str());
        }

        if(zed_pos_tracking_enabled_)
        {
            // -----> Depth
            RCLCPP_INFO(this->get_logger(), "*** DEPTH PARAMETERS ***");

            int depth_mode;
            this->get_parameter("depth/quality", depth_mode);
            zed_depth_mode_ = static_cast<sl::DEPTH_MODE>(depth_mode);
            RCLCPP_INFO(this->get_logger(), " * Depth quality        -> %s", sl::toString(zed_depth_mode_).c_str());

            int sensing_mode;
            this->get_parameter("depth/sensing_mode", sensing_mode);
            // assuming "this" is TagSlamZED node
            zed_sensing_mode_ = static_cast<sl::DE = get_ros_option<int>(this, "depth/sensing_mode", 0);PTH_MODE>(sensing_mode);
            RCLCPP_INFO(this->get_logger(), " * Depth Sensing mode       -> %s", sl::toString(zed_sensing_mode_).c_str());

            this->get_parameter("depth/min_depth", zed_min_depth_);
            RCLCPP_INFO(this->get_logger(), " * Minimum depth        -> %f m", zed_min_depth_);

            this->get_parameter("depth/max_depth", zed_max_depth_);
            RCLCPP_INFO(this->get_logger(), " * Maximum depth        -> %f m", zed_max_depth_);
        } else{
            zed_depth_mode_ = sl::DEPTH_MODE::NONE;
        }
            
        // ROS publication parameters
        this->get_parameter("publish/publish_tags", if_pub_tag_det_);
        this->get_parameter("publish/publish_image_with_tags", if_pub_tag_det_image_);
        this->get_parameter("publish/publish_image", if_pub_image_);
        this->get_parameter("publish/publish_landmarks", if_pub_landmark_);
        this->get_parameter("publish/publish_latency", if_pub_latency_);
    }

    void TagSlamZED::turn_on_zed()
    {
        // Try to initialize the ZED
        
        zed_init_param_.camera_fps = zed_frame_rate_;
        zed_init_param_.camera_resolution = static_cast<sl::RESOLUTION>(zed_resol_);
        
        // Set default coordinate system
        zed_init_param_.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
        RCLCPP_INFO(this->get_logger(), " * Camera coordinate system     -> %s", sl::toString(zed_init_param_.coordinate_system).c_str());

        // set up camera parameters
        zed_init_param_.coordinate_units = sl::UNIT::METER;
        zed_init_param_.depth_mode = static_cast<sl::DEPTH_MODE>(zed_depth_mode_);
        zed_init_param_.depth_stabilization = 0; //disable the depth stabilization
        zed_init_param_.depth_minimum_distance = static_cast<float>(zed_min_depth_);
        zed_init_param_.depth_maximum_distance = static_cast<float>(zed_max_depth_);
        zed_init_param_.enable_image_enhancement = true; // Always active
        zed_init_param_.camera_image_flip = sl::FLIP_MODE::OFF;

        sl::ERROR_CODE conn_status = sl::ERROR_CODE::CAMERA_NOT_DETECTED;

        RCLCPP_INFO(this->get_logger(), " *** Opening %s ...", sl::toString(zed_user_model_).c_str());

        while (conn_status != sl::ERROR_CODE::SUCCESS) {
            conn_status = zed_camera_.open(zed_init_param_);
            RCLCPP_INFO(this->get_logger(), "ZED connection -> %s", sl::toString(conn_status).c_str());
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));

            if (!rclcpp::ok()) {
                RCLCPP_DEBUG(this->get_logger(), "Closing ZED");
                zed_camera_.close();
                RCLCPP_DEBUG(this->get_logger(), "ZED pool thread finished");
                return;
            }
        }
        RCLCPP_INFO(this->get_logger(), " ... %s ready", sl::toString(zed_real_model_).c_str());

        // Disable AEC_AGC and Auto Whitebalance to trigger it if use set to automatic
        // zed_camera_.setCameraSettings(sl::VIDEO_SETTINGS::AEC_AGC, 0);
        // zed_camera_.setCameraSettings(sl::VIDEO_SETTINGS::WHITEBALANCE_AUTO, 0);
        zed_camera_.setCameraSettings(sl::VIDEO_SETTINGS::EXPOSURE, zed_exposure_);
        zed_real_model_ = zed_camera_.getCameraInformation().camera_model;

        if (zed_real_model_ == sl::MODEL::ZED) {
            RCLCPP_ERROR(this->get_logger(), "ZED not supported, please use ZED2");
        }

        if (zed_user_model_ != zed_real_model_) {
            RCLCPP_WARN(this->get_logger(), "Camera model does not match user parameter. Please"
                                            " modify the value of the parameter 'camera_model' to 'zed2'");
        }

        // Enable positional tracking if configured
        if(zed_pos_tracking_enabled_){
            sl::PositionalTrackingParameters tracking_parameters;
            tracking_parameters.enable_area_memory = false;
            tracking_parameters.enable_imu_fusion = true;
            tracking_parameters.set_gravity_as_origin = false;
            zed_camera_.enablePositionalTracking(tracking_parameters);
            RCLCPP_INFO(this->get_logger(), "Positional tracking enabled");
        }

        // Initialize the camera runtime parameters
        // zed_runtime_param_.sensing_mode = zed_sensing_mode_;
        zed_runtime_param_.enable_depth = zed_pos_tracking_enabled_; // pose tracking require depth

        // Get camera intrinsics and generate camera info
        cam_info_msg_ptr_ = std::make_shared<sensor_msgs::msg::CameraInfo>(); // safer than cam_info_msg_.reset(new sensor_msgs::msg::CameraInfo());

        sl::CameraConfiguration zed_cam_config = zed_camera_.getCameraInformation().camera_configuration;
        cam_info_msg_ptr_->width = zed_cam_config.resolution.width;
        cam_info_msg_ptr_->height = zed_cam_config.resolution.height;
        fill_camera_info(cam_info_msg_ptr_, zed_cam_config.calibration_parameters);

        if(use_imu_odom_){
            // Set up IMU
            sl::SensorsConfiguration sensor_config = zed_camera_.getCameraInformation().sensors_configuration;
            sl::SensorParameters accel_param = sensor_config.accelerometer_parameters;
            sl::SensorParameters gyro_param = sensor_config.gyroscope_parameters;

            double accel_noise_sigma = accel_param.noise_density;
            double accel_bias_rw_sigma = accel_param.random_walk;
            double gyro_noise_sigma = gyro_param.noise_density;
            double gyro_bias_rw_sigma = gyro_param.random_walk;

            // Get gravity estimate from IMU
            sl::SensorsData sensor_data;
            zed_camera_.getSensorsData(sensor_data, sl::TIME_REFERENCE::CURRENT);
            double accl_x =  sensor_data.imu.linear_acceleration[0];
            double accl_y = sensor_data.imu.linear_acceleration[1];
            double accl_z = sensor_data.imu.linear_acceleration[2];
            
            EigenPose T_sensor2cam = sl_trans_to_eigen(sensor_config.camera_imu_transform);

            slam_backend_->setupIMU(accel_noise_sigma, accel_bias_rw_sigma,
                        gyro_noise_sigma, gyro_bias_rw_sigma, T_sensor2cam); 
                        // accl_x, accl_y, accl_z);
        }
    }

    void TagSlamZED::gpu_image_thread_func(){
        
        // Initialize the image container
        sl::Mat sl_mat;
        std_msgs::msg::Header msg_header;
        msg_header.frame_id = "left_rect";

        while(rclcpp::ok()) // ros is still runing
        {
            msg_header.stamp = this->get_clock()->now();
            
            sl::ERROR_CODE zed_grab_status = zed_camera_.grab(zed_runtime_param_);
            if(zed_grab_status == sl::ERROR_CODE::SUCCESS) {
                auto t0 = std::chrono::system_clock::now();

#ifndef NO_CUDA_OPENCV
                // retrieve left image from ZED
                zed_camera_.retrieveImage(sl_mat, zed_image_type_, sl::MEM::GPU);
                // store the image as a cv_mat
                cv::cuda::GpuMat cv_mat = sl_mat_to_cv_mat_gpu(sl_mat);
                // change from BGRA to RGBA
                cv::cuda::cvtColor(cv_mat, cv_mat, cv::COLOR_BGRA2RGBA);
#else           
                static bool warned = false;
                if (!warned) {
                    RCLCPP_WARN(this->get_logger(), "Use CUDA enabled OpenCV will reduce memory copy overhead");
                    warned = true;
                }
                // retrieve left image without CUDA
                zed_camera_.retrieveImage(sl_mat, zed_image_type_, sl::MEM::CPU);
                cv::Mat cv_mat = sl_mat_to_cv_mat(sl_mat);
                cv::cvtColor(cv_mat, cv_mat, cv::COLOR_BGRA2RGBA);
#endif

                // msg_header.stamp = sl_time_to_ros(zed_camera_.getTimestamp(sl::TIME_REFERENCE::IMAGE));

                auto t1 = std::chrono::system_clock::now();

                // Run detection
                auto static_tag_array_ptr = std::make_shared<AprilTagDetectionArray>();
                auto dyn_tag_array_ptr = std::make_shared<AprilTagDetectionArray>();
                tag_detector_->detectTags(cv_mat, cam_info_msg_ptr_, msg_header,
                                    static_tag_array_ptr, dyn_tag_array_ptr);

                auto t2 = std::chrono::system_clock::now();

                if(!detection_only_ && run_slam_)
                {
                    // Do a SLAM update to estimate current pose and publish the message
                    estimate_state(static_tag_array_ptr);    
                }

                auto t3 = std::chrono::system_clock::now();
                
                if(if_pub_latency_)
                {
                    float d0 = std::chrono::duration<float, std::milli>(t1 - t0).count();
                    float d1 = std::chrono::duration<float, std::milli>(t2 - t1).count();
                    float d2 = std::chrono::duration<float, std::milli>(t3 - t2).count();
                    float d = std::chrono::duration<float, std::milli>(t3 - t0).count();

                    // publishing latency debug messages
                    auto temp = std_msgs::msg::Float32();
                    temp.data = d0;
                    debug_convert_pub_->publish(temp);
                    temp.data = d1;
                    debug_det_pub_->publish(temp);
                    temp.data = d2;
                    debug_opt_pub_->publish(temp);
                    temp.data = d;
                    debug_total_pub_->publish(temp);
                }

                publish_images(static_tag_array_ptr, dyn_tag_array_ptr);

                publish_detection_array(static_tag_array_ptr, dyn_tag_array_ptr);

                frame_count_++;
            }
        }
    }

    void TagSlamZED::cpu_image_thread_func(){
        // initialize the image container
        sl::Mat sl_mat;

        std_msgs::msg::Header msg_header;
        msg_header.frame_id = "left_rect";

        while(rclcpp::ok()) // ros is still runing
        {
            msg_header.stamp = this->get_clock()->now();

            sl::ERROR_CODE zed_grab_status = zed_camera_.grab(zed_runtime_param_);
            if(zed_grab_status == sl::ERROR_CODE::SUCCESS)
            {
                auto t0 = std::chrono::system_clock::now();

#ifndef NO_CUDA_OPENCV
                // Retrieve left image
                zed_camera_.retrieveImage(sl_mat, zed_image_type_, sl::MEM::GPU);
                // store the image as a cv_mat
                cv::cuda::GpuMat cv_mat_gpu = sl_mat_to_cv_mat_gpu(sl_mat);
                cv::Mat cv_mat_cpu;
                cv_mat_gpu.download(cv_mat_cpu);
                // // change from BGRA to RGBA
                // cv::cuda::cvtColor(cv_mat, cv_mat, cv::COLOR_BGRA2RGBA);
#else 
                // Retrieve left image
                zed_camera_.retrieveImage(sl_mat, zed_image_type_, sl::MEM::CPU);
            
                // store the image as a cv_mat
                // this is a gray scale image
                cv::Mat cv_mat_cpu = sl_mat_to_cv_mat(sl_mat);
#endif
                // msg_header.stamp = sl_time_to_ros(zed_camera_.getTimestamp(sl::TIME_REFERENCE::IMAGE));

                auto t1 = std::chrono::system_clock::now();

                // Run detection
                auto static_tag_array_ptr = std::make_shared<AprilTagDetectionArray>();
                auto dyn_tag_array_ptr = std::make_shared<AprilTagDetectionArray>();
                tag_detector_->detectTags(cv_mat_cpu, cam_info_msg_ptr_, msg_header,
                                    static_tag_array_ptr, dyn_tag_array_ptr);
                
                auto t2 = std::chrono::system_clock::now();

                if(!detection_only_ && run_slam_)
                {
                    // Do a SLAM update to estimate current pose and publish the message
                    estimate_state(static_tag_array_ptr);    
                }

                auto t3 = std::chrono::system_clock::now();

                if(if_pub_latency_)
                {
                    float d0 = std::chrono::duration<float, std::milli>(t1 - t0).count();
                    float d1 = std::chrono::duration<float, std::milli>(t2 - t1).count();
                    float d2 = std::chrono::duration<float, std::milli>(t3 - t2).count();
                    float d = std::chrono::duration<float, std::milli>(t3 - t0).count();
                    
                    // Publishing latency debug messages
                    auto temp = std_msgs::msg::Float32();
                    temp.data = d0;
                    debug_convert_pub_->publish(temp);
                    temp.data = d1;
                    debug_det_pub_->publish(temp);
                    temp.data = d2;
                    debug_opt_pub_->publish(temp);
                    temp.data = d;
                    debug_total_pub_->publish(temp);
                }

                publish_images(static_tag_array_ptr, dyn_tag_array_ptr);
                publish_detection_array(static_tag_array_ptr, dyn_tag_array_ptr);

                frame_count_++;
            }
        }
    }

    void TagSlamZED::sensors_thread_func()
    {
        sl::SensorsData sensor_data;
        TimestampHandler ts_handler;

        while(rclcpp::ok()) // ros is still runing
        {
            // try to retrive the sensor data
            zed_camera_.getSensorsData(sensor_data, sl::TIME_REFERENCE::CURRENT);
            if(ts_handler.isNew(sensor_data.imu))
            {
                auto imu_msg_ptr = boost::make_shared<sensor_msgs::msg::Imu>();

                imu_msg_ptr->header.stamp = sl_time_to_ros(sensor_data.imu.timestamp);

                imu_msg_ptr->orientation.x = sensor_data.imu.pose.getOrientation()[0];
                imu_msg_ptr->orientation.y = sensor_data.imu.pose.getOrientation()[1];
                imu_msg_ptr->orientation.z = sensor_data.imu.pose.getOrientation()[2];
                imu_msg_ptr->orientation.w = sensor_data.imu.pose.getOrientation()[3];

                imu_msg_ptr->angular_velocity.x = sensor_data.imu.angular_velocity[0] * DEG2RAD;
                imu_msg_ptr->angular_velocity.y = sensor_data.imu.angular_velocity[1] * DEG2RAD;
                imu_msg_ptr->angular_velocity.z = sensor_data.imu.angular_velocity[2] * DEG2RAD;

                imu_msg_ptr->linear_acceleration.x = sensor_data.imu.linear_acceleration[0];
                imu_msg_ptr->linear_acceleration.y = sensor_data.imu.linear_acceleration[1];
                imu_msg_ptr->linear_acceleration.z = sensor_data.imu.linear_acceleration[2];

                for (int i = 0; i < 3; ++i)
                {
                    int r = i;

                    imu_msg_ptr->orientation_covariance[i * 3 + 0] = 
                        sensor_data.imu.pose_covariance.r[r * 3 + 0] * DEG2RAD * DEG2RAD;
                    imu_msg_ptr->orientation_covariance[i * 3 + 1] = 
                        sensor_data.imu.pose_covariance.r[r * 3 + 1] * DEG2RAD * DEG2RAD;
                    imu_msg_ptr->orientation_covariance[i * 3 + 2] = 
                        sensor_data.imu.pose_covariance.r[r * 3 + 2] * DEG2RAD * DEG2RAD;

                    imu_msg_ptr->linear_acceleration_covariance[i * 3 + 0] = 
                        sensor_data.imu.linear_acceleration_covariance.r[r * 3 + 0];
                    imu_msg_ptr->linear_acceleration_covariance[i * 3 + 1] = 
                        sensor_data.imu.linear_acceleration_covariance.r[r * 3 + 1];
                    imu_msg_ptr->linear_acceleration_covariance[i * 3 + 2] = 
                        sensor_data.imu.linear_acceleration_covariance.r[r * 3 + 2];

                    imu_msg_ptr->angular_velocity_covariance[i * 3 + 0] =
                        sensor_data.imu.angular_velocity_covariance.r[r * 3 + 0] * DEG2RAD * DEG2RAD;
                    imu_msg_ptr->angular_velocity_covariance[i * 3 + 1] =
                        sensor_data.imu.angular_velocity_covariance.r[r * 3 + 1] * DEG2RAD * DEG2RAD;
                    imu_msg_ptr->angular_velocity_covariance[i * 3 + 2] =
                        sensor_data.imu.angular_velocity_covariance.r[r * 3 + 2] * DEG2RAD * DEG2RAD;
                }
                if(use_imu_odom_){
                    slam_backend_->updateIMU(imu_msg_ptr);
                }

                imu_pub_->publish(*imu_msg_ptr);
            }
        }
    }

    void TagSlamZED::estimate_state(TagDetectionArrayPtr tag_array_ptr)
    {
        nav_msgs::msg::Odometry::SharedPtr slam_pose_msg_ptr;

        EigenPose relative_pose;
        EigenPoseCov pose_cur_cov;
                
        // Retrieve pose
        if (zed_pos_tracking_enabled_)
        {
            // this is accumlated camera odometry pose with loop closure
            sl::Pose sl_pose_current;
            zed_camera_.getPosition(sl_pose_current, sl::REFERENCE_FRAME::WORLD);   
            EigenPose pose_cur = sl_pose_to_eigen(sl_pose_current);

            pose_cur_cov = sl_pose_to_cov(sl_pose_current);
            relative_pose = pose_prev_.inverse() * pose_cur;
            
            pose_prev_ = pose_cur; // only needed if we use ZED's visual odomtery 
        }

        
        if (use_imu_odom_) {
            slam_pose_msg_ptr = slam_backend_->updateVIO(tag_array_ptr, relative_pose, pose_cur_cov, zed_pos_tracking_enabled_);
        } else {
            slam_pose_msg_ptr = slam_backend_->updateSLAM(tag_array_ptr, relative_pose, pose_cur_cov);
        }

        // publish the message
        if (slam_pose_msg_ptr) {
            slam_pose_pub_->publish(*slam_pose_msg_ptr);
        }

        if (if_pub_landmark_) {
            // publish landmark
            visualization_msgs::msg::MarkerArray::SharedPtr landmark_msg_ptr = slam_backend_->createMarkerArray(tag_array_ptr->header);
            landmark_pub_->publish(*landmark_msg_ptr);
        }
        
    }

    void TagSlamZED::publish_images(TagDetectionArrayPtr static_tag_array_ptr, 
                                TagDetectionArrayPtr dyn_tag_array_ptr)
    {
        // Publish ros messages    
        int num_image_subscriber = img_pub_->get_subscription_count();
        int num_detection_subscriber = det_img_pub_->get_subscription_count();
        
        std_msgs::msg::Header header = static_tag_array_ptr->header;

        if (num_image_subscriber > 0 || num_detection_subscriber > 0) {
            // Download the image to cpu
            sl::Mat sl_mat_cpu;
            zed_camera_.retrieveImage(sl_mat_cpu, sl::VIEW::LEFT, sl::MEM::CPU); 
            
            // create raw image message
            if(num_image_subscriber > 0 && if_pub_image_){
                auto raw_img_msg_ptr = boost::make_shared<sensor_msgs::msg::Image>();
                sl_mat_to_ros_msg(raw_img_msg_ptr, sl_mat_cpu, header);
                cam_info_msg_ptr_->header = header;
                img_pub_->publish(*raw_img_msg_ptr, *cam_info_msg_ptr_);
            }

            // create detection image
            if(num_detection_subscriber > 0 && if_pub_tag_det_image_){

                cv::Mat detectionMat = sl_mat_to_cv_mat(sl_mat_cpu).clone();
                tag_detector_->drawDetections(detectionMat, static_tag_array_ptr);
                tag_detector_->drawDetections(detectionMat, dyn_tag_array_ptr);

                cv_bridge::CvImage detectionImgMsg;
                detectionImgMsg.header = header;
                detectionImgMsg.encoding = sensor_msgs::msg::image_encodings::BGRA8;
                detectionImgMsg.image = detectionMat;

                det_img_pub_->publish(*detectionImgMsg.toImageMsg()); // .toImageMsg() returns ImagePtr
            }
        }

    }

    void TagSlamZED::publish_detection_array(TagDetectionArrayPtr static_tag_array_ptr,
                                TagDetectionArrayPtr dyn_tag_array_ptr)
    {
        if(static_tag_det_pub_->get_subscription_count() > 0 && if_pub_tag_det_){
            static_tag_det_pub_->publish(*static_tag_array_ptr);
        }

        if(dyn_tag_det_pub_->get_subscription_count() > 0 && if_pub_tag_det_){
            dyn_tag_det_pub_->publish(*dyn_tag_array_ptr);
        }
    }

    void TagSlamZED::check_resol_fps()
    {
        switch (zed_resol_) {
        case sl::RESOLUTION::HD2K:
            if (zed_frame_rate_ != 15) {
                RCLCPP_WARN(this->get_logger(),
                        "Wrong FrameRate (%d) for the resolution HD2K. Set to 15 FPS.",
                        zed_frame_rate_);
                zed_frame_rate_ = 15;
            }

            break;

        case sl::RESOLUTION::HD1080:
            if (zed_frame_rate_ == 15 || zed_frame_rate_ == 30) {
                break;
            }

            if (zed_frame_rate_ > 15 && zed_frame_rate_ < 30) {
                RCLCPP_WARN(this->get_logger(), 
                        "Wrong FrameRate (%d) for the resolution HD1080. Set to 15 FPS.",
                        zed_frame_rate_);
                zed_frame_rate_ = 15;
            } else if (zed_frame_rate_ > 30) {
                RCLCPP_WARN(this->get_logger(), "Wrong FrameRate (%d}) for the resolution HD1080. Set to 30 FPS.",
                zed_frame_rate_);
                zed_frame_rate_ = 30;
            } else {
                RCLCPP_WARN(this->get_logger(), "Wrong FrameRate (%d) for the resolution HD1080. Set to 15 FPS.",
                zed_frame_rate_);
                zed_frame_rate_ = 15;
            }

            break;

        case sl::RESOLUTION::HD720:
            if (zed_frame_rate_ == 15 || zed_frame_rate_ == 30 || zed_frame_rate_ == 60) {
                break;
            }

            if (zed_frame_rate_ > 15 && zed_frame_rate_ < 30) {
                RCLCPP_WARN(this->get_logger(), "Wrong FrameRate (%d) for the resolution HD720. Set to 15 FPS.",
                zed_frame_rate_);
                zed_frame_rate_ = 15;
            } else if (zed_frame_rate_ > 30 && zed_frame_rate_ < 60) {
                RCLCPP_WARN(this->get_logger(), "Wrong FrameRate (%d) for the resolution HD720. Set to 30 FPS.",
                zed_frame_rate_);
                zed_frame_rate_ = 30;
            } else if (zed_frame_rate_ > 60) {
                RCLCPP_WARN(this->get_logger(), "Wrong FrameRate (%d) for the resolution HD720. Set to 60 FPS.",
                zed_frame_rate_);
                zed_frame_rate_ = 60;
            } else {
                RCLCPP_WARN(this->get_logger(), "Wrong FrameRate (%d) for the resolution HD720. Set to 15 FPS.",
                zed_frame_rate_);
                zed_frame_rate_ = 15;
            }

            break;

        case sl::RESOLUTION::VGA:
            if (zed_frame_rate_ == 15 || zed_frame_rate_ == 30 || zed_frame_rate_ == 60 || zed_frame_rate_ == 100) {
                break;
            }

            if (zed_frame_rate_ > 15 && zed_frame_rate_ < 30) {
                RCLCPP_WARN(this->get_logger(), "Wrong FrameRate (%d) for the resolution VGA. Set to 15 FPS.",
                zed_frame_rate_);
                zed_frame_rate_ = 15;
            } else if (zed_frame_rate_ > 30 && zed_frame_rate_ < 60) {
                RCLCPP_WARN(this->get_logger(), "Wrong FrameRate (%d) for the resolution VGA. Set to 30 FPS.",
                zed_frame_rate_);
                zed_frame_rate_ = 30;
            } else if (zed_frame_rate_ > 60 && zed_frame_rate_ < 100) {
                RCLCPP_WARN(this->get_logger(), "Wrong FrameRate (%d) for the resolution VGA. Set to 60 FPS.",
                zed_frame_rate_);
                zed_frame_rate_ = 60;
            } else if (zed_frame_rate_ > 100) {
                RCLCPP_WARN(this->get_logger(), "Wrong FrameRate (%d) for the resolution VGA. Set to 100 FPS.",
                zed_frame_rate_);
                zed_frame_rate_ = 100;
            } else {
                RCLCPP_WARN(this->get_logger(), "Wrong FrameRate (%d) for the resolution VGA. Set to 15 FPS.",
                zed_frame_rate_);
                zed_frame_rate_ = 15;
            }

            break;

        default:
            RCLCPP_WARN(this->get_logger(), "Invalid resolution. Set to HD720 @ 30 FPS");
            zed_resol_ = sl::RESOLUTION::HD720;
            zed_frame_rate_ = 30;
        }
    }

    void TagSlamZED::sl_mat_to_ros_msg(sensor_msgs::msg::Image::SharedPtr img_msg_ptr, sl::Mat img, std_msgs::msg::Header header)
    {
        if (!img_msg_ptr)
        {
            return;
        }

        img_msg_ptr->header = header;
        img_msg_ptr->height = img.getHeight();
        img_msg_ptr->width = img.getWidth();

        int num = 1;  // for endianness detection
        img_msg_ptr->is_bigendian = !(*(char*)&num == 1);

        img_msg_ptr->step = img.getStepBytes();

        size_t size = img_msg_ptr->step * img_msg_ptr->height;
        img_msg_ptr->data.resize(size);

        sl::MAT_TYPE dataType = img.getDataType();

        switch (dataType)
        {
            case sl::MAT_TYPE::F32_C1: /**< float 1 channel.*/
            img_msg_ptr->encoding = sensor_msgs::msg::image_encodings::TYPE_32FC1;
            memcpy((char*)(&img_msg_ptr->data[0]), img.getPtr<sl::float1>(), size);
            break;

            case sl::MAT_TYPE::F32_C2: /**< float 2 channels.*/
            img_msg_ptr->encoding = sensor_msgs::msg::image_encodings::TYPE_32FC2;
            memcpy((char*)(&img_msg_ptr->data[0]), img.getPtr<sl::float2>(), size);
            break;

            case sl::MAT_TYPE::F32_C3: /**< float 3 channels.*/
            img_msg_ptr->encoding = sensor_msgs::msg::image_encodings::TYPE_32FC3;
            memcpy((char*)(&img_msg_ptr->data[0]), img.getPtr<sl::float3>(), size);
            break;

            case sl::MAT_TYPE::F32_C4: /**< float 4 channels.*/
            img_msg_ptr->encoding = sensor_msgs::msg::image_encodings::TYPE_32FC4;
            memcpy((char*)(&img_msg_ptr->data[0]), img.getPtr<sl::float4>(), size);
            break;

            case sl::MAT_TYPE::U8_C1: /**< unsigned char 1 channel.*/
            img_msg_ptr->encoding = sensor_msgs::msg::image_encodings::MONO8;
            memcpy((char*)(&img_msg_ptr->data[0]), img.getPtr<sl::uchar1>(), size);
            break;

            case sl::MAT_TYPE::U8_C2: /**< unsigned char 2 channels.*/
            img_msg_ptr->encoding = sensor_msgs::msg::image_encodings::TYPE_8UC2;
            memcpy((char*)(&img_msg_ptr->data[0]), img.getPtr<sl::uchar2>(), size);
            break;

            case sl::MAT_TYPE::U8_C3: /**< unsigned char 3 channels.*/
            img_msg_ptr->encoding = sensor_msgs::msg::image_encodings::BGR8;
            memcpy((char*)(&img_msg_ptr->data[0]), img.getPtr<sl::uchar3>(), size);
            break;

            case sl::MAT_TYPE::U8_C4: /**< unsigned char 4 channels.*/
            img_msg_ptr->encoding = sensor_msgs::msg::image_encodings::BGRA8;
            memcpy((char*)(&img_msg_ptr->data[0]), img.getPtr<sl::uchar4>(), size);
            break;

            case sl::MAT_TYPE::U16_C1: /**< unsigned short 1 channel.*/
            img_msg_ptr->encoding = sensor_msgs::msg::image_encodings::TYPE_16UC1;
            memcpy((uint16_t*)(&img_msg_ptr->data[0]), img.getPtr<sl::ushort1>(), size);
            break;
        }
    }

    rclcpp::Time TagSlamZED::sl_time_to_ros(sl::Timestamp t)
    {
        uint32_t sec = static_cast<uint32_t>(t.getNanoseconds() / 1000000000);
        uint32_t nsec = static_cast<uint32_t>(t.getNanoseconds() % 1000000000);
        return rclcpp::Time(sec, nsec);
    }

    EigenPose TagSlamZED::sl_trans_to_eigen(sl::Transform& pose){
        sl::Translation t = pose.getTranslation();
        sl::Orientation quat = pose.getOrientation();
        Eigen::Affine3d eigenPose;
        eigenPose.translation() = Eigen::Vector3d(t(0), t(1), t(2));
        eigenPose.linear() = Eigen::Quaterniond(quat(3), quat(0), quat(1), quat(2)).toRotationMatrix();
        return eigenPose.matrix();
    }

    EigenPose TagSlamZED::sl_pose_to_eigen(sl::Pose& pose){
        sl::Translation t = pose.getTranslation();
        sl::Orientation quat = pose.getOrientation();
        Eigen::Affine3d eigenPose;
        eigenPose.translation() = Eigen::Vector3d(t(0), t(1), t(2));
        eigenPose.linear() = Eigen::Quaterniond(quat(3), quat(0), quat(1), quat(2)).toRotationMatrix();
        return eigenPose.matrix();
    }

    EigenPoseSigma TagSlamZED::sl_pose_to_sigma(sl::Pose& pose){
        EigenPoseSigma sigma;
        sigma << std::sqrt(pose.pose_covariance[21]), std::sqrt(pose.pose_covariance[28]),
                std::sqrt(pose.pose_covariance[35]), std::sqrt(pose.pose_covariance[0]),
                std::sqrt(pose.pose_covariance[7]), std::sqrt(pose.pose_covariance[14]);
        return sigma;
    }

    EigenPoseCov TagSlamZED::sl_pose_to_cov(sl::Pose& pose)
    {
        // gtsam have order rotx, roty, rotz, x, y, z
        // ros have order x, y, z, rotx, roty, rotz

        // [TT, TR;
        //  RT, RR]

        EigenPoseCov cov_ros;
        for(int i=0; i<6; i++){
        for(int j=0; j<6; j++)
        {
            int k = i*6+j;
            cov_ros(i,j) = pose.pose_covariance[k];
        }
        }

        Eigen::Matrix3d TT = cov_ros.block<3,3>(0,0);
        Eigen::Matrix3d TR = cov_ros.block<3,3>(0,3);
        Eigen::Matrix3d RT = cov_ros.block<3,3>(3,0);
        Eigen::Matrix3d RR = cov_ros.block<3,3>(3,3);

        // [RR, RT;
        // [TR, TT]
        EigenPoseCov cov_gtsam;
        cov_gtsam.block<3,3>(0,0) = RR;
        cov_gtsam.block<3,3>(0,3) = RT;
        cov_gtsam.block<3,3>(3,0) = TR;
        cov_gtsam.block<3,3>(3,3) = TT;

        return cov_gtsam;
    }

    cv::Mat TagSlamZED::sl_mat_to_cv_mat(sl::Mat& input) {
        // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
        // cv::Mat and sl::Mat will share a single memory structure
        size_t height = input.getHeight();
        size_t width = input.getWidth();
        sl::MAT_TYPE dataType = input.getDataType();
        int cv_type = getOCVtype(dataType);
        size_t step_bytes = input.getStepBytes(sl::MEM::CPU);
        cv::Mat output;
        
        switch(dataType)
        {
            case sl::MAT_TYPE::F32_C1: /**< float 1 channel.*/
            output = cv::Mat(height, width, cv_type, input.getPtr<sl::float1>(sl::MEM::CPU), step_bytes);
            break;

            case sl::MAT_TYPE::F32_C2: /**< float 2 channels.*/
            output = cv::Mat(height, width, cv_type, input.getPtr<sl::float2>(sl::MEM::CPU), step_bytes);
            break;

            case sl::MAT_TYPE::F32_C3: /**< float 3 channels.*/
            output = cv::Mat(height, width, cv_type, input.getPtr<sl::float3>(sl::MEM::CPU), step_bytes);
            break;

            case sl::MAT_TYPE::F32_C4: /**< float 4 channels.*/
            output = cv::Mat(height, width, cv_type, input.getPtr<sl::float4>(sl::MEM::CPU), step_bytes);
            break;

            case sl::MAT_TYPE::U8_C1: /**< unsigned char 1 channel.*/
            output = cv::Mat(height, width, cv_type, input.getPtr<sl::uchar1>(sl::MEM::CPU), step_bytes);
            break;

            case sl::MAT_TYPE::U8_C2: /**< unsigned char 2 channels.*/
            output = cv::Mat(height, width, cv_type, input.getPtr<sl::uchar2>(sl::MEM::CPU), step_bytes);
            break;

            case sl::MAT_TYPE::U8_C3: /**< unsigned char 3 channels.*/
            output = cv::Mat(height, width, cv_type, input.getPtr<sl::uchar3>(sl::MEM::CPU), step_bytes);
            break;

            case sl::MAT_TYPE::U8_C4: /**< unsigned char 4 channels.*/
            output = cv::Mat(height, width, cv_type, input.getPtr<sl::uchar4>(sl::MEM::CPU), step_bytes);
            break;

            case sl::MAT_TYPE::U16_C1: /**< unsigned short 1 channel.*/
            output = cv::Mat(height, width, cv_type, input.getPtr<sl::ushort1>(sl::MEM::CPU), step_bytes);
            break;
        }

        return output;
    }

#ifndef NO_CUDA_OPENCV
    /**
    * Conversion function between sl::Mat and cv::cuda::GpuMat 
    **/
    cv::cuda::GpuMat TagSlamZED::sl_mat_to_cv_mat_gpu(sl::Mat& input) {
        // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
        // cv::Mat and sl::Mat will share a single memory structure
        size_t height = input.getHeight();
        size_t width = input.getWidth();
        sl::MAT_TYPE dataType = input.getDataType();
        int cv_type = getOCVtype(dataType);
        size_t step_bytes = input.getStepBytes(sl::MEM::GPU);
        cv::cuda::GpuMat output;
        
        switch(dataType)
        {
            case sl::MAT_TYPE::F32_C1: /**< float 1 channel.*/
            output = cv::cuda::GpuMat (height, width, cv_type, input.getPtr<sl::float1>(sl::MEM::GPU), step_bytes);
            break;

            case sl::MAT_TYPE::F32_C2: /**< float 2 channels.*/
            output = cv::cuda::GpuMat (height, width, cv_type, input.getPtr<sl::float2>(sl::MEM::GPU), step_bytes);
            break;

            case sl::MAT_TYPE::F32_C3: /**< float 3 channels.*/
            output = cv::cuda::GpuMat (height, width, cv_type, input.getPtr<sl::float3>(sl::MEM::GPU), step_bytes);
            break;

            case sl::MAT_TYPE::F32_C4: /**< float 4 channels.*/
            output = cv::cuda::GpuMat (height, width, cv_type, input.getPtr<sl::float4>(sl::MEM::GPU), step_bytes);
            break;

            case sl::MAT_TYPE::U8_C1: /**< unsigned char 1 channel.*/
            output = cv::cuda::GpuMat (height, width, cv_type, input.getPtr<sl::uchar1>(sl::MEM::GPU), step_bytes);
            break;

            case sl::MAT_TYPE::U8_C2: /**< unsigned char 2 channels.*/
            output = cv::cuda::GpuMat (height, width, cv_type, input.getPtr<sl::uchar2>(sl::MEM::GPU), step_bytes);
            break;

            case sl::MAT_TYPE::U8_C3: /**< unsigned char 3 channels.*/
            output = cv::cuda::GpuMat (height, width, cv_type, input.getPtr<sl::uchar3>(sl::MEM::GPU), step_bytes);
            break;

            case sl::MAT_TYPE::U8_C4: /**< unsigned char 4 channels.*/
            output = cv::cuda::GpuMat (height, width, cv_type, input.getPtr<sl::uchar4>(sl::MEM::GPU), step_bytes);
            break;

            case sl::MAT_TYPE::U16_C1: /**< unsigned short 1 channel.*/
            output = cv::cuda::GpuMat (height, width, cv_type, input.getPtr<sl::ushort1>(sl::MEM::GPU), step_bytes);
            break;
        }
        return output;
    }
#endif

    // Mapping between MAT_TYPE and CV_TYPE
    int TagSlamZED::getOCVtype(sl::MAT_TYPE type) {
        int cv_type = -1;
        switch (type) {
            case sl::MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
            case sl::MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
            case sl::MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
            case sl::MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
            case sl::MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
            case sl::MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
            case sl::MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
            case sl::MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
            default: break;
        }
        return cv_type;
    }

    void TagSlamZED::fill_camera_info(sensor_msgs::msg::CameraInfo::SharedPtr cam_info_msg_ptr, sl::CalibrationParameters zedParam)
    {

        // distortion
        cam_info_msg_ptr->distortion_model = sensor_msgs::msg::distortion_models::PLUMB_BOB;
        cam_info_msg_ptr->D.resize(5);
        cam_info_msg_ptr->D[0] = zedParam.left_cam.disto[0]; // k1
        cam_info_msg_ptr->D[1] = zedParam.left_cam.disto[1]; // k2
        cam_info_msg_ptr->D[2] = zedParam.left_cam.disto[4]; // k3
        cam_info_msg_ptr->D[3] = zedParam.left_cam.disto[2]; // p1
        cam_info_msg_ptr->D[4] = zedParam.left_cam.disto[3]; // p2

        // intrinsic
        cam_info_msg_ptr->K.fill(0.0);
        cam_info_msg_ptr->K[0] = static_cast<double>(zedParam.left_cam.fx);
        cam_info_msg_ptr->K[2] = static_cast<double>(zedParam.left_cam.cx);
        cam_info_msg_ptr->K[4] = static_cast<double>(zedParam.left_cam.fy);
        cam_info_msg_ptr->K[5] = static_cast<double>(zedParam.left_cam.cy);
        cam_info_msg_ptr->K[8] = 1.0;

        // Set Rectification matrix to identity

        for (size_t i = 0; i < 3; i++) {
            // identity
            cam_info_msg_ptr->R[i + i * 3] = 1;
        }

        // Projection/camera matrix
        cam_info_msg_ptr->P.fill(0.0);
        cam_info_msg_ptr->P[0] = static_cast<double>(zedParam.left_cam.fx);
        cam_info_msg_ptr->P[2] = static_cast<double>(zedParam.left_cam.cx);
        cam_info_msg_ptr->P[5] = static_cast<double>(zedParam.left_cam.fy);
        cam_info_msg_ptr->P[6] = static_cast<double>(zedParam.left_cam.cy);
        cam_info_msg_ptr->P[10] = 1.0;

    }

} // namespace tag_slam
#endif //NO_ZED