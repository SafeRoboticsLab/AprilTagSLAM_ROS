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

#include "tag_slam_zed.h"

PLUGINLIB_EXPORT_CLASS(tagslam_ros::TagSlamZED, nodelet::Nodelet);

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

    TagSlamZED::TagSlamZED()
        : Nodelet()
    {
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

        std::cerr << "Tag Slam ZED Nodelet destroyed" << std::endl;
    }

    void TagSlamZED::onInit()
    {
        // Node handlers
        nh_ = getMTNodeHandle();
        pnh_ = getMTPrivateNodeHandle();

        NODELET_INFO("********** Starting nodelet '%s' **********", getName().c_str());

        readParameters();

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
    }

    void TagSlamZED::setup_service()
    {
        // Set the service to reset the map
        srv_reset_slam_ = pnh_.advertiseService("reset_slam", &TagSlamZED::resetCallback, this);
        srv_start_slam_ = pnh_.advertiseService("start_slam", &TagSlamZED::startCallback, this);
        srv_stop_slam_ = pnh_.advertiseService("stop_slam", &TagSlamZED::stopCallback, this);
    }

    void TagSlamZED::setup_publisher()
    {
        // Create all the publishers
        // Image publishers
        image_transport::ImageTransport it_zed(pnh_);

        if(if_pub_image_)
            img_pub_ = it_zed.advertiseCamera("Image", 1); // raw image
        
        if(if_pub_tag_det_image_)
            det_img_pub_ = it_zed.advertise("Tag_Detection_Image", 1); // image with tag detection
        
        if(if_pub_tag_det_)
            static_tag_det_pub_ = pnh_.advertise<AprilTagDetectionArray>("Tag_Detections", 1);
            dyn_tag_det_pub_ = pnh_.advertise<AprilTagDetectionArray>("Tag_Detections_Dynamic", 1);

        if(!detection_only_)
            slam_pose_pub_ = pnh_.advertise<nav_msgs::msg::Odometry>("Pose", 1);

        // // IMU Publishers
        if(use_imu_odom_)
            imu_pub_ = pnh_.advertise<sensor_msgs::msg::Imu>("IMU/Data", 1);

        // landmark publisher
        if(if_pub_landmark_)
            landmark_pub_ = pnh_.advertise<visualization_msgs::msg::MarkerArray>("Landmarks", 1);

        if (if_pub_latency_)
        {
            debug_convert_pub_ = pnh_.advertise<std_msgs::msg::Float32>("Debug/Convert", 1);
            debug_det_pub_ = pnh_.advertise<std_msgs::msg::Float32>("Debug/Detect", 1);
            debug_opt_pub_ = pnh_.advertise<std_msgs::msg::Float32>("Debug/Optimize", 1);
            debug_total_pub_ = pnh_.advertise<std_msgs::msg::Float32>("Debug/Total", 1);
        }
    }

    void TagSlamZED::readParameters(){

        NODELET_INFO_STREAM("*** GENERAL PARAMETERS ***");

        /*
        ************** Zed Camera Parameters **************
        */
        // Get parameters from param files
        std::string camera_model = getRosOption<std::string>(pnh_, "camera/camera_model", "zed2");
        if (camera_model == "zed") {
            zed_user_model_ = sl::MODEL::ZED;
            NODELET_INFO_STREAM(" * Camera Model by param\t-> " << camera_model);
        } else if (camera_model == "zedm") {
            zed_user_model_ = sl::MODEL::ZED_M;
            NODELET_INFO_STREAM(" * Camera Model by param\t-> " << camera_model);
        } else if (camera_model == "zed2") {
            zed_user_model_ = sl::MODEL::ZED2;
            NODELET_INFO_STREAM(" * Camera Model by param\t-> " << camera_model);
        } else if (camera_model == "zed2i") {
            zed_user_model_ = sl::MODEL::ZED2i;
            NODELET_INFO_STREAM(" * Camera Model by param\t-> " << camera_model);
        } else {
            NODELET_ERROR_STREAM("Camera model not valid: " << camera_model);
        }

        zed_exposure_ = getRosOption<int>(pnh_, "camera/exposure", 20);

        int resol = getRosOption<int>(pnh_, "camera/resolution", 3); // defulat to HD720
        zed_resol_ = static_cast<sl::RESOLUTION>(resol);
        NODELET_INFO_STREAM(" * Camera Resolution\t\t-> " << sl::toString(zed_resol_).c_str());

        zed_frame_rate_ = getRosOption<int>(pnh_, "camera/frame_rate", 30); // default to 30 fps
        checkResolFps();
        NODELET_INFO_STREAM(" * Camera Grab Framerate\t-> " << zed_frame_rate_);

        /*
        ************** Setup Front End **************
        */
        std::string frontend_type = getRosOption<std::string>(pnh_, "frontend/type", "CPU");
        if (frontend_type == "GPU") {
            // GPU detecotr take a recitified RGBA8 image
            // ZED return a BGRA8 image, we need to convert it later
            use_gpu_detector_ = true;
            zed_imge_type_ = sl::VIEW::LEFT; 
            tag_detector_ = std::make_unique<TagDetectorCUDA>(pnh_);
        }else{
            // CPU detector take a rectified gray image
            use_gpu_detector_ = false;
            zed_imge_type_ = sl::VIEW::LEFT_GRAY;
            tag_detector_ = std::make_unique<TagDetectorCPU>(pnh_);
        }
        NODELET_INFO_STREAM(" * " <<(use_gpu_detector_ ? "Use GPU tag detector" : "Use CPU tag detector"));


        /*
        *********** Setup Backend **************
        */
        // std::string odom_type = getRosOption<std::string>(pnh_, "backend/odom", "vision");

        use_imu_odom_ = getRosOption<bool>(pnh_, "backend/use_imu", true);
        zed_pos_tracking_enabled_ = getRosOption<bool>(pnh_, "backend/use_odom", true);

        if(!use_imu_odom_ && !zed_pos_tracking_enabled_){
            NODELET_WARN("No odometry source is enabled, please enable at least one of them. Running in detection only mode.");
            detection_only_ = true;
        }

        backend_type_ = getRosOption<std::string>(pnh_, "backend/smoother", "isam2");
        if (backend_type_ =="isam2") {
            slam_backend_ = std::make_unique<iSAM2Backend>(pnh_);
            NODELET_INFO("Using iSAM2 backend.");
        }else if (backend_type_ == "fixed_lag"){
            slam_backend_ = std::make_unique<FixedLagBackend>(pnh_);
            NODELET_INFO("Using fixed-lag backend.");
        }else if (backend_type_ == "none"){
            slam_backend_ = nullptr;
            detection_only_ = true;
            use_imu_odom_ = false;
            zed_pos_tracking_enabled_ = false;
            NODELET_INFO("Apriltag Detector Mode.");
        }else{
            NODELET_ERROR("Not supported backend type: %s", backend_type_.c_str());
        }

        if(zed_pos_tracking_enabled_)
        {
            // -----> Depth
            NODELET_INFO_STREAM("*** DEPTH PARAMETERS ***");
            int depth_mode = getRosOption<int>(pnh_, "depth/quality", 1);
            zed_depth_mode_ = static_cast<sl::DEPTH_MODE>(depth_mode);
            NODELET_INFO_STREAM(" * Depth quality\t\t-> " << sl::toString(zed_depth_mode_).c_str());

            int sensing_mode = getRosOption<int>(pnh_, "depth/sensing_mode", 0);
            zed_sensing_mode_ = static_cast<sl::DEPTH_MODE>(sensing_mode);
            NODELET_INFO_STREAM(" * Depth Sensing mode\t\t-> " << sl::toString(zed_sensing_mode_).c_str());

            zed_min_depth_ = getRosOption<double>(pnh_, "depth/min_depth", 0.5);
            NODELET_INFO_STREAM(" * Minimum depth\t\t-> " << zed_min_depth_ << " m");

            zed_max_depth_ = getRosOption<double>(pnh_, "depth/max_depth", 15.0);
            NODELET_INFO_STREAM(" * Maximum depth\t\t-> " << zed_max_depth_ << " m");
        }else{
            zed_depth_mode_ = sl::DEPTH_MODE::NONE;
        }
            
        // ros publication parameters
        if_pub_tag_det_ = getRosOption<bool>(pnh_, "publish/publish_tags", true);
        if_pub_tag_det_image_ = getRosOption<bool>(pnh_, "publish/publish_image_with_tags", true);
        if_pub_image_ = getRosOption<bool>(pnh_, "publish/publish_image", true);
        if_pub_landmark_ = getRosOption<bool>(pnh_, "publish/publish_landmarks", true);
        if_pub_latency_ = getRosOption<bool>(pnh_, "publish/publish_latency", true);
    }

    void TagSlamZED::turn_on_zed()
    {
        // Try to initialize the ZED
        
        zed_init_param_.camera_fps = zed_frame_rate_;
        zed_init_param_.camera_resolution = static_cast<sl::RESOLUTION>(zed_resol_);
        
        // Set default coordinate system
        zed_init_param_.coordinate_system = sl::COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD;
        NODELET_INFO_STREAM(" * Camera coordinate system\t-> " << sl::toString(zed_init_param_.coordinate_system));

        // set up camera parameters
        zed_init_param_.coordinate_units = sl::UNIT::METER;
        zed_init_param_.depth_mode = static_cast<sl::DEPTH_MODE>(zed_depth_mode_);
        zed_init_param_.depth_stabilization = 0; //disable the depth stabilization
        zed_init_param_.depth_minimum_distance = static_cast<float>(zed_min_depth_);
        zed_init_param_.depth_maximum_distance = static_cast<float>(zed_max_depth_);
        zed_init_param_.enable_image_enhancement = true; // Always active
        zed_init_param_.camera_image_flip = sl::FLIP_MODE::OFF;

        sl::ERROR_CODE conn_status = sl::ERROR_CODE::CAMERA_NOT_DETECTED;

        NODELET_INFO_STREAM(" *** Opening " << sl::toString(zed_user_model_) << "...");
        while (conn_status != sl::ERROR_CODE::SUCCESS) {
            conn_status = zed_camera_.open(zed_init_param_);
            NODELET_INFO_STREAM("ZED connection -> " << sl::toString(conn_status));
            std::this_thread::sleep_for(std::chrono::milliseconds(2000));

            if (!pnh_.ok()) {
                NODELET_DEBUG("Closing ZED");
                zed_camera_.close();
                NODELET_DEBUG("ZED pool thread finished");
                return;
            }
        }
        NODELET_INFO_STREAM(" ...  " << sl::toString(zed_real_model_) << " ready");

        // Disable AEC_AGC and Auto Whitebalance to trigger it if use set to automatic
        // zed_camera_.setCameraSettings(sl::VIDEO_SETTINGS::AEC_AGC, 0);
        // zed_camera_.setCameraSettings(sl::VIDEO_SETTINGS::WHITEBALANCE_AUTO, 0);
        zed_camera_.setCameraSettings(sl::VIDEO_SETTINGS::EXPOSURE, zed_exposure_);
        zed_real_model_ = zed_camera_.getCameraInformation().camera_model;
        if (zed_real_model_ == sl::MODEL::ZED)
        {
            NODELET_ERROR("ZED not supported, please use ZED2");
        }

        if (zed_user_model_ != zed_real_model_) {
            NODELET_WARN("Camera model does not match user parameter. Please modify "
                            "the value of the parameter 'camera_model' to 'zed2'");
        }

        // enable positional tracking
        if(zed_pos_tracking_enabled_){
            sl::PositionalTrackingParameters tracking_parameters;
            tracking_parameters.enable_area_memory = false;
            tracking_parameters.enable_imu_fusion = true;
            tracking_parameters.set_gravity_as_origin = false;
            zed_camera_.enablePositionalTracking(tracking_parameters);
            NODELET_INFO("Positional tracking enabled");
        }

        // Initialize the camera runtime parameters
        // zed_runtime_param_.sensing_mode = zed_sensing_mode_;
        zed_runtime_param_.enable_depth = zed_pos_tracking_enabled_; // pose tracking require depth

        //get camera intrinsics and generate camera info
        cam_info_msg_.reset(new sensor_msgs::msg::CameraInfo());

        sl::CameraConfiguration zed_cam_config = zed_camera_.getCameraInformation().camera_configuration;
        cam_info_msg_->width = zed_cam_config.resolution.width;
        cam_info_msg_->height = zed_cam_config.resolution.height;
        fillCameraInfo(cam_info_msg_, zed_cam_config.calibration_parameters);

        if(use_imu_odom_){
            // set up imu
            sl::SensorsConfiguration sensor_config = zed_camera_.getCameraInformation().sensors_configuration;
            sl::SensorParameters accel_param = sensor_config.accelerometer_parameters;
            sl::SensorParameters gyro_param = sensor_config.gyroscope_parameters;

            double accel_noise_sigma = accel_param.noise_density;
            double accel_bias_rw_sigma = accel_param.random_walk;
            double gyro_noise_sigma = gyro_param.noise_density;
            double gyro_bias_rw_sigma = gyro_param.random_walk;

            // estimate the grivaty
            sl::SensorsData sensor_data;
            zed_camera_.getSensorsData(sensor_data, sl::TIME_REFERENCE::CURRENT);
            double accl_x =  sensor_data.imu.linear_acceleration[0];
            double accl_y = sensor_data.imu.linear_acceleration[1];
            double accl_z = sensor_data.imu.linear_acceleration[2];
            
            EigenPose T_sensor2cam = slTrans2Eigen(sensor_config.camera_imu_transform);

            slam_backend_->setupIMU(accel_noise_sigma, accel_bias_rw_sigma,
                        gyro_noise_sigma, gyro_bias_rw_sigma, T_sensor2cam); 
                        // accl_x, accl_y, accl_z);
        }
    }

    void TagSlamZED::gpu_image_thread_func(){
        // initialize the image container
        
        sl::Mat sl_mat;
            
        std_msgs::msg::Header msg_header;
        msg_header.frame_id = "left_rect";

        while(pnh_.ok()) // ros is still runing
        {
            msg_header.stamp = rclcpp::Time::now();
            sl::ERROR_CODE zed_grab_status = zed_camera_.grab(zed_runtime_param_);
            if(zed_grab_status == sl::ERROR_CODE::SUCCESS)
            {
                auto t0 = std::chrono::system_clock::now();

#ifndef NO_CUDA_OPENCV
                // Retrieve left image
                zed_camera_.retrieveImage(sl_mat, zed_imge_type_, sl::MEM::GPU);
                // store the image as a cv_mat
                cv::cuda::GpuMat cv_mat = slMat2cvMatGPU(sl_mat);
                // change from BGRA to RGBA
                cv::cuda::cvtColor(cv_mat, cv_mat, cv::COLOR_BGRA2RGBA);
#else 
                NODELET_WARN_ONCE("Use CUDA enabled OpenCV will reduce memory copy overhead");
                zed_camera_.retrieveImage(sl_mat, zed_imge_type_, sl::MEM::CPU);
            
                cv::Mat cv_mat = slMat2cvMat(sl_mat);

                cv::cvtColor(cv_mat, cv_mat, cv::COLOR_BGRA2RGBA);
#endif

                // msg_header.stamp = slTime2Ros(zed_camera_.getTimestamp(sl::TIME_REFERENCE::IMAGE));
                msg_header.seq = frame_count_;

                auto t1 = std::chrono::system_clock::now();
                // Run detection
                auto static_tag_array_ptr = std::make_shared<AprilTagDetectionArray>();
                auto dyn_tag_array_ptr = std::make_shared<AprilTagDetectionArray>();
                tag_detector_->detectTags(cv_mat, cam_info_msg_, msg_header,
                                    static_tag_array_ptr, dyn_tag_array_ptr);

                auto t2 = std::chrono::system_clock::now();

                if(!detection_only_ && run_slam_)
                {
                    // Do a SLAM update to estimate current pose and publish the message
                    estimateState(static_tag_array_ptr);    
                }

                auto t3 = std::chrono::system_clock::now();
                
                if(if_pub_latency_)
                {
                    float d0 = std::chrono::duration<float, std::milli>(t1 - t0).count();
                    float d1 = std::chrono::duration<float, std::milli>(t2 - t1).count();
                    float d2 = std::chrono::duration<float, std::milli>(t3 - t2).count();
                    float d = std::chrono::duration<float, std::milli>(t3 - t0).count();
                    std_msgs::msg::Float32 temp;
                    temp.data = d0;
                    debug_convert_pub_.publish(temp);
                    temp.data = d1;
                    debug_det_pub_.publish(temp);
                    temp.data = d2;
                    debug_opt_pub_.publish(temp);
                    temp.data = d;
                    debug_total_pub_.publish(temp);
                }

                publishImages(static_tag_array_ptr, dyn_tag_array_ptr);

                publishDetectionArray(static_tag_array_ptr, dyn_tag_array_ptr);

                frame_count_++;
            }
        }
    }

    void TagSlamZED::cpu_image_thread_func(){
        // initialize the image container
        sl::Mat sl_mat;

        std_msgs::msg::Header msg_header;
        msg_header.frame_id = "left_rect";

        while(pnh_.ok()) // ros is still runing
        {
            msg_header.stamp = rclcpp::Time::now();
            sl::ERROR_CODE zed_grab_status = zed_camera_.grab(zed_runtime_param_);
            if(zed_grab_status == sl::ERROR_CODE::SUCCESS)
            {
                auto t0 = std::chrono::system_clock::now();

#ifndef NO_CUDA_OPENCV
                // Retrieve left image
                zed_camera_.retrieveImage(sl_mat, zed_imge_type_, sl::MEM::GPU);
                // store the image as a cv_mat
                cv::cuda::GpuMat cv_mat_gpu = slMat2cvMatGPU(sl_mat);
                cv::Mat cv_mat_cpu;
                cv_mat_gpu.download(cv_mat_cpu);
                // // change from BGRA to RGBA
                // cv::cuda::cvtColor(cv_mat, cv_mat, cv::COLOR_BGRA2RGBA);
#else 
                // Retrieve left image
                zed_camera_.retrieveImage(sl_mat, zed_imge_type_, sl::MEM::CPU);
            
                // store the image as a cv_mat
                // this is a gray scale image
                cv::Mat cv_mat_cpu = slMat2cvMat(sl_mat);
#endif
                // msg_header.stamp = slTime2Ros(zed_camera_.getTimestamp(sl::TIME_REFERENCE::IMAGE));
                msg_header.seq = frame_count_;

                auto t1 = std::chrono::system_clock::now();

                // Run detection
                auto static_tag_array_ptr = std::make_shared<AprilTagDetectionArray>();
                auto dyn_tag_array_ptr = std::make_shared<AprilTagDetectionArray>();
                tag_detector_->detectTags(cv_mat_cpu, cam_info_msg_, msg_header,
                                    static_tag_array_ptr, dyn_tag_array_ptr);
                
                auto t2 = std::chrono::system_clock::now();

                if(!detection_only_ && run_slam_)
                {
                    // Do a SLAM update to estimate current pose and publish the message
                    estimateState(static_tag_array_ptr);    
                }

                auto t3 = std::chrono::system_clock::now();

                if(if_pub_latency_)
                {
                    float d0 = std::chrono::duration<float, std::milli>(t1 - t0).count();
                    float d1 = std::chrono::duration<float, std::milli>(t2 - t1).count();
                    float d2 = std::chrono::duration<float, std::milli>(t3 - t2).count();
                    float d = std::chrono::duration<float, std::milli>(t3 - t0).count();
                    std_msgs::msg::Float32 temp;
                    temp.data = d0;
                    debug_convert_pub_.publish(temp);
                    temp.data = d1;
                    debug_det_pub_.publish(temp);
                    temp.data = d2;
                    debug_opt_pub_.publish(temp);
                    temp.data = d;
                    debug_total_pub_.publish(temp);
                }

                publishImages(static_tag_array_ptr, dyn_tag_array_ptr);

                publishDetectionArray(static_tag_array_ptr, dyn_tag_array_ptr);

                frame_count_++;
            }
        }
    }

    void TagSlamZED::sensors_thread_func()
    {
        sl::SensorsData sensor_data;
        TimestampHandler ts_handler;
        while(pnh_.ok()) // ros is still runing
        {
            // try to retrive the sensor data
            zed_camera_.getSensorsData(sensor_data, sl::TIME_REFERENCE::CURRENT);
            if(ts_handler.isNew(sensor_data.imu))
            {
                std::shared_ptr<sensor_msgs::msg::Imu> imu_msg_ptr = std::make_shared<sensor_msgs::msg::Imu>();
                imu_msg_ptr->header.stamp = slTime2Ros(sensor_data.imu.timestamp);

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
                    int r = 0;

                    if (i == 0)
                    {
                        r = 0;
                    }
                    else if (i == 1)
                    {
                        r = 1;
                    }
                    else
                    {
                        r = 2;
                    }

                    imu_msg_ptr->orientation_covariance[i * 3 + 0] = sensor_data.imu.pose_covariance.r[r * 3 + 0] * DEG2RAD * DEG2RAD;
                    imu_msg_ptr->orientation_covariance[i * 3 + 1] = sensor_data.imu.pose_covariance.r[r * 3 + 1] * DEG2RAD * DEG2RAD;
                    imu_msg_ptr->orientation_covariance[i * 3 + 2] = sensor_data.imu.pose_covariance.r[r * 3 + 2] * DEG2RAD * DEG2RAD;

                    imu_msg_ptr->linear_acceleration_covariance[i * 3 + 0] = sensor_data.imu.linear_acceleration_covariance.r[r * 3 + 0];
                    imu_msg_ptr->linear_acceleration_covariance[i * 3 + 1] = sensor_data.imu.linear_acceleration_covariance.r[r * 3 + 1];
                    imu_msg_ptr->linear_acceleration_covariance[i * 3 + 2] = sensor_data.imu.linear_acceleration_covariance.r[r * 3 + 2];

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

                imu_pub_.publish(imu_msg_ptr);
            }

        }
    }

    void TagSlamZED::estimateState(TagDetectionArrayPtr tag_array_ptr)
    {
        std::shared_ptr<nav_msgs::msg::Odometry> slam_pose_msg;

        EigenPose relative_pose;
        EigenPoseCov pose_cur_cov;
                
        // Retrieve pose
        if(zed_pos_tracking_enabled_)
        {
            // this is accumlated camera odometry pose with loop closure
            sl::Pose sl_pose_current;
            zed_camera_.getPosition(sl_pose_current, sl::REFERENCE_FRAME::WORLD);   
            EigenPose pose_cur = slPose2Eigen(sl_pose_current);

            pose_cur_cov = slPose2Cov(sl_pose_current);
            relative_pose = pose_prev_.inverse()*pose_cur;
            
            pose_prev_ = pose_cur; // only needed if we use ZED's visual odomtery 
        }

        
        if(use_imu_odom_){
            slam_pose_msg = slam_backend_->updateVIO(tag_array_ptr, relative_pose, pose_cur_cov, zed_pos_tracking_enabled_);
        }else{
            slam_pose_msg = slam_backend_->updateSLAM(tag_array_ptr, relative_pose, pose_cur_cov);
        }

        // publish the message
        if(slam_pose_msg){
            slam_pose_pub_.publish(slam_pose_msg);
        }

        if(if_pub_landmark_)
        {
            // publish landmark
            std::shared_ptr<visualization_msgs::msg::MarkerArray> landmark_msg_ptr = slam_backend_->createMarkerArray(tag_array_ptr->header);
            landmark_pub_.publish(landmark_msg_ptr);
        }
        
    }

    void TagSlamZED::publishImages(TagDetectionArrayPtr static_tag_array_ptr, 
                                TagDetectionArrayPtr dyn_tag_array_ptr)
    {
        // Publish ros messages    
        int num_image_subscriber = img_pub_.getNumSubscribers();
        int num_detection_subscriber = det_img_pub_.getNumSubscribers();
        
        std_msgs::msg::Header header = static_tag_array_ptr->header;

        if(num_image_subscriber||num_detection_subscriber){
            // Download the image to cpu
            sl::Mat sl_mat_cpu;
            zed_camera_.retrieveImage(sl_mat_cpu, sl::VIEW::LEFT, sl::MEM::CPU); 
            
            // create raw image message
            if(num_image_subscriber > 0 && if_pub_image_){
                std::shared_ptr<sensor_msgs::msg::Image> rawImgMsg = std::make_shared<sensor_msgs::msg::Image>();
                slMatToROSmsg(rawImgMsg, sl_mat_cpu, header);
                cam_info_msg_->header = header;
                img_pub_.publish(rawImgMsg, cam_info_msg_);
            }

            // create detection image
            if(num_detection_subscriber > 0 && if_pub_tag_det_image_){

                cv::Mat detectionMat = slMat2cvMat(sl_mat_cpu).clone();
                tag_detector_->drawDetections(detectionMat, static_tag_array_ptr);
                tag_detector_->drawDetections(detectionMat, dyn_tag_array_ptr);

                cv_bridge::CvImage detectionImgMsg;
                detectionImgMsg.header = header;
                detectionImgMsg.encoding = sensor_msgs::msg::image_encodings::BGRA8;
                detectionImgMsg.image = detectionMat;
                det_img_pub_.publish(detectionImgMsg.toImageMsg());
            }
        }

    }

    void TagSlamZED::publishDetectionArray(TagDetectionArrayPtr static_tag_array_ptr,
                                TagDetectionArrayPtr dyn_tag_array_ptr)
    {
        if(static_tag_det_pub_.getNumSubscribers() > 0 && if_pub_tag_det_){
            static_tag_det_pub_.publish(*static_tag_array_ptr);
        }

        if(dyn_tag_det_pub_.getNumSubscribers() > 0 && if_pub_tag_det_){
            dyn_tag_det_pub_.publish(*dyn_tag_array_ptr);
        }
    }

    void TagSlamZED::checkResolFps()
    {
        switch (zed_resol_) {
        case sl::RESOLUTION::HD2K:
            if (zed_frame_rate_ != 15) {
                NODELET_WARN_STREAM("Wrong FrameRate (" << zed_frame_rate_ << ") for the resolution HD2K. Set to 15 FPS.");
                zed_frame_rate_ = 15;
            }

            break;

        case sl::RESOLUTION::HD1080:
            if (zed_frame_rate_ == 15 || zed_frame_rate_ == 30) {
                break;
            }

            if (zed_frame_rate_ > 15 && zed_frame_rate_ < 30) {
                NODELET_WARN_STREAM("Wrong FrameRate (" << zed_frame_rate_ << ") for the resolution HD1080. Set to 15 FPS.");
                zed_frame_rate_ = 15;
            } else if (zed_frame_rate_ > 30) {
                NODELET_WARN_STREAM("Wrong FrameRate (" << zed_frame_rate_ << ") for the resolution HD1080. Set to 30 FPS.");
                zed_frame_rate_ = 30;
            } else {
                NODELET_WARN_STREAM("Wrong FrameRate (" << zed_frame_rate_ << ") for the resolution HD1080. Set to 15 FPS.");
                zed_frame_rate_ = 15;
            }

            break;

        case sl::RESOLUTION::HD720:
            if (zed_frame_rate_ == 15 || zed_frame_rate_ == 30 || zed_frame_rate_ == 60) {
                break;
            }

            if (zed_frame_rate_ > 15 && zed_frame_rate_ < 30) {
                NODELET_WARN_STREAM("Wrong FrameRate (" << zed_frame_rate_ << ") for the resolution HD720. Set to 15 FPS.");
                zed_frame_rate_ = 15;
            } else if (zed_frame_rate_ > 30 && zed_frame_rate_ < 60) {
                NODELET_WARN_STREAM("Wrong FrameRate (" << zed_frame_rate_ << ") for the resolution HD720. Set to 30 FPS.");
                zed_frame_rate_ = 30;
            } else if (zed_frame_rate_ > 60) {
                NODELET_WARN_STREAM("Wrong FrameRate (" << zed_frame_rate_ << ") for the resolution HD720. Set to 60 FPS.");
                zed_frame_rate_ = 60;
            } else {
                NODELET_WARN_STREAM("Wrong FrameRate (" << zed_frame_rate_ << ") for the resolution HD720. Set to 15 FPS.");
                zed_frame_rate_ = 15;
            }

            break;

        case sl::RESOLUTION::VGA:
            if (zed_frame_rate_ == 15 || zed_frame_rate_ == 30 || zed_frame_rate_ == 60 || zed_frame_rate_ == 100) {
                break;
            }

            if (zed_frame_rate_ > 15 && zed_frame_rate_ < 30) {
                NODELET_WARN_STREAM("Wrong FrameRate (" << zed_frame_rate_ << ") for the resolution VGA. Set to 15 FPS.");
                zed_frame_rate_ = 15;
            } else if (zed_frame_rate_ > 30 && zed_frame_rate_ < 60) {
                NODELET_WARN_STREAM("Wrong FrameRate (" << zed_frame_rate_ << ") for the resolution VGA. Set to 30 FPS.");
                zed_frame_rate_ = 30;
            } else if (zed_frame_rate_ > 60 && zed_frame_rate_ < 100) {
                NODELET_WARN_STREAM("Wrong FrameRate (" << zed_frame_rate_ << ") for the resolution VGA. Set to 60 FPS.");
                zed_frame_rate_ = 60;
            } else if (zed_frame_rate_ > 100) {
                NODELET_WARN_STREAM("Wrong FrameRate (" << zed_frame_rate_ << ") for the resolution VGA. Set to 100 FPS.");
                zed_frame_rate_ = 100;
            } else {
                NODELET_WARN_STREAM("Wrong FrameRate (" << zed_frame_rate_ << ") for the resolution VGA. Set to 15 FPS.");
                zed_frame_rate_ = 15;
            }

            break;

        default:
            NODELET_WARN_STREAM("Invalid resolution. Set to HD720 @ 30 FPS");
            zed_resol_ = sl::RESOLUTION::HD720;
            zed_frame_rate_ = 30;
        }
    }

    void TagSlamZED::slMatToROSmsg(std::shared_ptr<sensor_msgs::msg::Image> imgMsgPtr, sl::Mat img, std_msgs::msg::Header header)
    {
        if (!imgMsgPtr)
        {
            return;
        }

        imgMsgPtr->header = header;
        imgMsgPtr->height = img.getHeight();
        imgMsgPtr->width = img.getWidth();

        int num = 1;  // for endianness detection
        imgMsgPtr->is_bigendian = !(*(char*)&num == 1);

        imgMsgPtr->step = img.getStepBytes();

        size_t size = imgMsgPtr->step * imgMsgPtr->height;
        imgMsgPtr->data.resize(size);

        sl::MAT_TYPE dataType = img.getDataType();

        switch (dataType)
        {
            case sl::MAT_TYPE::F32_C1: /**< float 1 channel.*/
            imgMsgPtr->encoding = sensor_msgs::msg::image_encodings::TYPE_32FC1;
            memcpy((char*)(&imgMsgPtr->data[0]), img.getPtr<sl::float1>(), size);
            break;

            case sl::MAT_TYPE::F32_C2: /**< float 2 channels.*/
            imgMsgPtr->encoding = sensor_msgs::msg::image_encodings::TYPE_32FC2;
            memcpy((char*)(&imgMsgPtr->data[0]), img.getPtr<sl::float2>(), size);
            break;

            case sl::MAT_TYPE::F32_C3: /**< float 3 channels.*/
            imgMsgPtr->encoding = sensor_msgs::msg::image_encodings::TYPE_32FC3;
            memcpy((char*)(&imgMsgPtr->data[0]), img.getPtr<sl::float3>(), size);
            break;

            case sl::MAT_TYPE::F32_C4: /**< float 4 channels.*/
            imgMsgPtr->encoding = sensor_msgs::msg::image_encodings::TYPE_32FC4;
            memcpy((char*)(&imgMsgPtr->data[0]), img.getPtr<sl::float4>(), size);
            break;

            case sl::MAT_TYPE::U8_C1: /**< unsigned char 1 channel.*/
            imgMsgPtr->encoding = sensor_msgs::msg::image_encodings::MONO8;
            memcpy((char*)(&imgMsgPtr->data[0]), img.getPtr<sl::uchar1>(), size);
            break;

            case sl::MAT_TYPE::U8_C2: /**< unsigned char 2 channels.*/
            imgMsgPtr->encoding = sensor_msgs::msg::image_encodings::TYPE_8UC2;
            memcpy((char*)(&imgMsgPtr->data[0]), img.getPtr<sl::uchar2>(), size);
            break;

            case sl::MAT_TYPE::U8_C3: /**< unsigned char 3 channels.*/
            imgMsgPtr->encoding = sensor_msgs::msg::image_encodings::BGR8;
            memcpy((char*)(&imgMsgPtr->data[0]), img.getPtr<sl::uchar3>(), size);
            break;

            case sl::MAT_TYPE::U8_C4: /**< unsigned char 4 channels.*/
            imgMsgPtr->encoding = sensor_msgs::msg::image_encodings::BGRA8;
            memcpy((char*)(&imgMsgPtr->data[0]), img.getPtr<sl::uchar4>(), size);
            break;

            case sl::MAT_TYPE::U16_C1: /**< unsigned short 1 channel.*/
            imgMsgPtr->encoding = sensor_msgs::msg::image_encodings::TYPE_16UC1;
            memcpy((uint16_t*)(&imgMsgPtr->data[0]), img.getPtr<sl::ushort1>(), size);
            break;
        }
    }

    rclcpp::Time TagSlamZED::slTime2Ros(sl::Timestamp t)
    {
        uint32_t sec = static_cast<uint32_t>(t.getNanoseconds() / 1000000000);
        uint32_t nsec = static_cast<uint32_t>(t.getNanoseconds() % 1000000000);
        return rclcpp::Time(sec, nsec);
    }

    EigenPose TagSlamZED::slTrans2Eigen(sl::Transform& pose){
        sl::Translation t = pose.getTranslation();
        sl::Orientation quat = pose.getOrientation();
        Eigen::Affine3d eigenPose;
        eigenPose.translation() = Eigen::Vector3d(t(0), t(1), t(2));
        eigenPose.linear() = Eigen::Quaterniond(quat(3), quat(0), quat(1), quat(2)).toRotationMatrix();
        return eigenPose.matrix();
    }

    EigenPose TagSlamZED::slPose2Eigen(sl::Pose& pose){
        sl::Translation t = pose.getTranslation();
        sl::Orientation quat = pose.getOrientation();
        Eigen::Affine3d eigenPose;
        eigenPose.translation() = Eigen::Vector3d(t(0), t(1), t(2));
        eigenPose.linear() = Eigen::Quaterniond(quat(3), quat(0), quat(1), quat(2)).toRotationMatrix();
        return eigenPose.matrix();
    }

    EigenPoseSigma TagSlamZED::slPose2Sigma(sl::Pose& pose){
        EigenPoseSigma sigma;
        sigma << std::sqrt(pose.pose_covariance[21]), std::sqrt(pose.pose_covariance[28]),
                std::sqrt(pose.pose_covariance[35]), std::sqrt(pose.pose_covariance[0]),
                std::sqrt(pose.pose_covariance[7]), std::sqrt(pose.pose_covariance[14]);
        return sigma;
    }

    EigenPoseCov TagSlamZED::slPose2Cov(sl::Pose& pose)
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

    cv::Mat TagSlamZED::slMat2cvMat(sl::Mat& input) {
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
    cv::cuda::GpuMat TagSlamZED::slMat2cvMatGPU(sl::Mat& input) {
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

    void TagSlamZED::fillCameraInfo(std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_msg_ptr, sl::CalibrationParameters zedParam)
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