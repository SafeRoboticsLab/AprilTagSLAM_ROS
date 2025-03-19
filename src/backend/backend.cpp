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

#include "backend/backend.hpp"

using namespace gtsam;

namespace tagslam_ros
{    
    /*
    * Helper function
    */
   // parse quaternion in x,y,z,w order, and normalize to unit length
    istream &operator>>(istream &is, Quaternion &q) {
        double x, y, z, w;
        is >> x >> y >> z >> w;
        const double norm = sqrt(w * w + x * x + y * y + z * z), f = 1.0 / norm;
        q = Quaternion(f * w, f * x, f * y, f * z);
        return is;
    }

    /* ************************************************************************* */
    // parse Rot3 from roll, pitch, yaw
    istream &operator>>(istream &is, Rot3 &R) {
        double yaw, pitch, roll;
        is >> roll >> pitch >> yaw; // noticenode_(node), order !
        R = Rot3::Ypr(yaw, pitch, roll);
        return is;
    }

    Backend::Backend(std::shared_ptr<rclcpp::Node> node):
        node_(node),
        prior_map_(get_ros_option<bool>(node_, "backend/prior_map", false)),
        save_graph_(get_ros_option<bool>(node_, "backend/save_graph", false)),
        load_map_path_(get_ros_option<std::string>(node_, "backend/load_path", "")),
        save_map_path_(get_ros_option<std::string>(node_, "backend/save_path", "")),
        landmark_factor_sigma_trans_(get_ros_option<double>(node_, "backend/landmark_sigma_trans", 0.1)),
        landmark_factor_sigma_rot_(get_ros_option<double>(node_, "backend/landmark_sigma_rot", 0.3)),
        landmark_prior_sigma_trans_(get_ros_option<double>(node_, "backend/landmark_prior_sigma_trans", 0.1)),
        landmark_prior_sigma_rot_(get_ros_option<double>(node_, "backend/landmark_prior_sigma_rot", 0.3)),
        pose_prior_sigma_trans_(get_ros_option<double>(node_, "backend/pose_prior_sigma_trans", 0.1)),
        pose_prior_sigma_rot_(get_ros_option<double>(node_, "backend/pose_prior_sigma_rot", 0.3)),
        vel_prior_sigma_(get_ros_option<double>(node_, "backend/vel_prior_sigma", 0.1)),
        bias_prior_sigma_(get_ros_option<double>(node_, "backend/bias_prior_sigma", 0.1)),
        fix_prior_(get_ros_option<bool>(node_, "backend/fix_prior", true)),
        initialized_(false), pose_count_(0)
    {
        // get pose_offset
        if (node_->has_parameter("backend/pose_offset")) {
            std::vector<double> pose_offset_vec;
            node_->get_parameter("backend/pose_offset", pose_offset_vec);
            need_pose_offset_ = true;
            pose_offset_ = EigenPose(pose_offset_vec.data()).transpose();
            pose_offset_inv_ = pose_offset_.inverse();
        }
        
        // reset graph and values
        factor_graph_.resize(0);
        initial_estimate_.clear();

        //calculate landmark covariance matrix
        EigenPoseSigma landmark_prior_sigma;
        landmark_prior_sigma<< Vector3::Constant(landmark_prior_sigma_rot_),Vector3::Constant(landmark_prior_sigma_trans_);
        landmark_prior_cov_ = landmark_prior_sigma.array().pow(2).matrix().asDiagonal();

        EigenPoseSigma landmark_factor_sigma;
        landmark_factor_sigma<< Vector3::Constant(landmark_factor_sigma_rot_),Vector3::Constant(landmark_factor_sigma_trans_);
        landmark_factor_cov_ = landmark_factor_sigma.array().pow(2).matrix().asDiagonal();


        // if no path is provided, the system will not save the map
        if(save_graph_ && save_map_path_.empty())
        {
            time_t t = time(0);   // get time now
            struct tm * now = localtime( & t );

            char buffer [80];
            strftime (buffer,80,"graph-%d-%m-%Y-%H-%M-%S.g2o",now);
            save_map_path_ = std::string(buffer);
            RCLCPP_WARN(node_->get_logger(), "No graph path provided. By default, system will save graph to %s",
                save_map_path_.c_str());
        }
        
        // load the map from the file
        if(prior_map_ && !load_map_path_.empty()){
            loadMap();
        }else{
            prior_map_ = false; // in case the map file is empty
            // if there is no prior map, the system will not do localization
            RCLCPP_WARN(node_->get_logger(), "No prior map is provided, the system will operate in SLAM mode");
        }
    }

    Backend::~Backend()
    {
        // write map and pose to file
        if (save_graph_)
        {
            write_to_file(landmark_values_, save_map_path_);
            RCLCPP_INFO(node_->get_logger(), "Graph saved to %s", save_map_path_.c_str());
        }
    }
    
    Values::shared_ptr Backend::read_from_file(const std::string &filename)
    {  
        // Pointer to GTSAM Values
        auto values = boost::make_shared<gtsam::Values>();

        // Parse the file
        std::ifstream is(filename.c_str());

        if (!is){
            RCLCPP_ERROR(node_->get_logger(), "Failed to open %s", filename.c_str());
            return values;
        }
        
        std::string tag;
        while (is >> tag) {
            if (tag == "VERTEX3") {
                unsigned char key_char;
                size_t id;
                double x, y, z;
                gtsam::Rot3 R;
                is >> key_char >> id >> x >> y >> z >> R;
                gtsam::Symbol key(key_char, id);
                values->insert(key, Pose3(R, {x, y, z}));
            } else if (tag == "VERTEX_SE3:QUAT") {
                unsigned char key_char;
                size_t id;
                double x, y, z;
                gtsam::Quaternion q;
                is >> key_char >> id >> x >> y >> z >> q;
                gtsam::Symbol key(key_char, id);
                values->insert(key, Pose3(q, {x, y, z}));
            } 
            is.ignore(LINESIZE, '\n');
        }
        return values;
    }

    Pose3 Backend::initSLAM(double cur_img_t)
    {
        Key cur_pose_key = Symbol(kPoseSymbol, pose_count_);
        Key cur_vel_key = Symbol(kVelSymbol, pose_count_);
        Key cur_bias_key = Symbol(kBiasSymbol, pose_count_);

        // if the system is not initialized, the first pose will be the origin
        // initialize the first pose as the origin with small covariance
        Pose3 cur_pose_init = Pose3(pose_offset_inv_); //Pose3(Rot3::RzRyRx(0, 0, 0), Point3(0, 0, 0));

        // in gtsam, covariacen order are rotx, roty, rotz, x, y, z
        auto pose_prior_noise = noiseModel::Diagonal::Sigmas((Vector(6) << Vector3::Constant(pose_prior_sigma_rot_),
                                                    Vector3::Constant(pose_prior_sigma_trans_)).finished());    

        // if there is no prior map, we add the prior factor to the first pose
        if(!prior_map_){                
            factor_graph_.addPrior<Pose3>(cur_pose_key, cur_pose_init, pose_prior_noise);
            RCLCPP_INFO(node_->get_logger(), "Set the first frame as the origin");
        }

        initial_estimate_.insert(cur_pose_key, cur_pose_init);

        // Prior of velocity, bias and imu only needed for VIO
        Point3 cur_vel_init = Point3(0,0,0);
        // set large sigma for velocity since we do not know
        auto vel_prior_noise = noiseModel::Isotropic::Sigma(3, vel_prior_sigma_);  // m/s 

        auto bias_prior_noise = noiseModel::Isotropic::Sigma(6, bias_prior_sigma_);

        // dump all previous inserted imu measurement
        while(!imu_queue_.empty())
        {
            sensor_msgs::msg::Imu::SharedPtr imu_msg_ptr;
            while(!imu_queue_.try_pop(imu_msg_ptr)){}
            if(rclcpp::Time(imu_msg_ptr->header.stamp).seconds()>=cur_img_t)
            {
                // use this imu measurement to initialize the gravity
                setGravity(imu_msg_ptr);
                // initialize the IMU pre-integrator
                preint_meas_ = boost::make_shared<PreintegratedCombinedMeasurements>(preint_param_, prev_bias_);
                break;
            }
        }
        
        // add prior factor to velocity and bias only if imu is initialized
        if(preint_meas_){
            factor_graph_.addPrior<Vector3>(cur_vel_key, cur_vel_init, vel_prior_noise);
            factor_graph_.addPrior<imuBias::ConstantBias>(cur_bias_key, prev_bias_, bias_prior_noise);
            initial_estimate_.insert(cur_vel_key, cur_vel_init);
            initial_estimate_.insert(cur_bias_key, prev_bias_);
        }

        initialized_ = true;
        RCLCPP_INFO(node_->get_logger(), "SLAM initialized");
        return cur_pose_init;
    }

    Pose3 Backend::addOdomFactor(EigenPose odom, EigenPoseCov odom_cov)
    {
        Key prev_pose_key = Symbol(kPoseSymbol,pose_count_-1);
        Key cur_pose_key = Symbol(kPoseSymbol, pose_count_);

        Pose3 odom_factor = Pose3(odom);
        auto odom_noise = noiseModel::Gaussian::Covariance(odom_cov);
        factor_graph_.emplace_shared<BetweenFactor<Pose3>>(
                    prev_pose_key, cur_pose_key, odom_factor, odom_noise);
        
        // calculate the initial pose of the current state based on the previous state
        // Pose3 prev_pose = isam_.calculateEstimate<pose3>(prev_pose_key);
        Pose3 cur_pose_init = prev_pose_ * odom_factor;
        initial_estimate_.insert(cur_pose_key, cur_pose_init);
        
        return cur_pose_init;
    }

    Pose3 Backend::addImuFactor(EigenPose odom, EigenPoseCov odom_cov, double cur_img_t, bool use_odom)
    {
        Key prev_pose_key = Symbol(kPoseSymbol,pose_count_-1);
        Key prev_vel_key = Symbol(kVelSymbol, pose_count_-1);
        Key prev_bias_key = Symbol(kBiasSymbol, pose_count_-1);

        Key cur_pose_key = Symbol(kPoseSymbol, pose_count_);
        Key cur_vel_key = Symbol(kVelSymbol, pose_count_);
        Key cur_bias_key = Symbol(kBiasSymbol, pose_count_);

        Pose3 cur_pose_init;
        Point3 cur_vel_init;

        // If the slam has been initialized
        // Integrate the imu factor
        double last_msg_t = prev_img_t_;
        int imu_count = 0;
        while(!imu_queue_.empty())
        {
            sensor_msgs::msg::Imu::SharedPtr imu_msg_ptr;
            while(!imu_queue_.try_pop(imu_msg_ptr)){}
            double msg_t = rclcpp::Time(imu_msg_ptr->header.stamp).seconds();
            // We will have imu messages newer than the image due to the latency in tag detection
            if(msg_t > last_msg_t)
            {
                double dt = msg_t-last_msg_t;
                last_msg_t = msg_t;
                Vector3 accel(imu_msg_ptr->linear_acceleration.x,
                                imu_msg_ptr->linear_acceleration.y,
                                imu_msg_ptr->linear_acceleration.z);

                Vector3 omega(imu_msg_ptr->angular_velocity.x,
                                imu_msg_ptr->angular_velocity.y,
                                imu_msg_ptr->angular_velocity.z);
                imu_count++;
                preint_meas_->integrateMeasurement(accel, omega, dt);
                // check if we should break the loop
                if(msg_t>=cur_img_t)
                {
                    auto corrected_imu = preint_meas_->correctMeasurementsBySensorPose(accel, omega);
                    correct_acc_ = corrected_imu.first;
                    correct_gyro_ = corrected_imu.second;
                    break;
                }
            }
        }
        
        // create IMU factor
        CombinedImuFactor imu_factor(prev_pose_key, prev_vel_key,
                                    cur_pose_key, cur_vel_key,
                                    prev_bias_key, cur_bias_key,
                                    *preint_meas_);
        
        factor_graph_.add(imu_factor);

        // from imu, estimate the next pose, velocity and bias
        NavState state_init = preint_meas_->predict(prev_state_, prev_bias_);

        //create odom factor
        if(use_odom){
            Pose3 odom_factor = Pose3(odom);
            auto odom_noise = noiseModel::Gaussian::Covariance(odom_cov);
            factor_graph_.emplace_shared<BetweenFactor<Pose3>>(
                        prev_pose_key, cur_pose_key, odom_factor, odom_noise);
            
            // calculate the initial pose of the current state based on the previous state
            cur_pose_init = prev_pose_ * odom_factor;
        }
        else{
            cur_pose_init = state_init.pose();
        }

        // cur_pose_init = state_init.pose();
        cur_vel_init = state_init.v();
        initial_estimate_.insert(cur_pose_key, cur_pose_init);
        initial_estimate_.insert(cur_vel_key, cur_vel_init);
        initial_estimate_.insert(cur_bias_key, prev_bias_);
        return cur_pose_init;
    }

    void Backend::updateIMU(sensor_msgs::msg::Imu::SharedPtr imu_msg_ptr)
    {
        // this queue is thread safe
        imu_queue_.push(imu_msg_ptr);
        reset_mutex_.unlock();
    }

    /*
    Initialize IMU
        `accel_noise_sigma` → `accelerometer_noise_density` m/s²/√Hz` (Accelerometer white noise)
        `gyro_noise_sigma` → `gyroscope_noise_density`  `rad/s/√Hz` (Gyro white noise)
        `accel_bias_rw_sigma` → `accelerometer_random_walk` m √Hz/s² (Accelerometer bias random walk)
        `gyro_bias_rw_sigma` → `gyroscope_random_walk`  rad √Hz/s (Gyro bias random walk)
    
    * notes regarding imu noise
    * https://groups.google.com/g/gtsam-users/c/o9brW87ZjZ0/m/dWOigRcjBAAJl
    * https://github.com/borglab/gtsam/issues/213
    */ 
    void Backend::setupIMU(double accel_noise_sigma, double accel_bias_rw_sigma,
                            double gyro_noise_sigma, double gyro_bias_rw_sigma, 
                            EigenPose T_sensor2cam)
    {
        RCLCPP_INFO(node_->get_logger(), "Imu Preintegration use following parameters:");
        RCLCPP_INFO(node_->get_logger(), " - accel_noise_sigma: %f", accel_noise_sigma);
        RCLCPP_INFO(node_->get_logger(), " - accel_bias_rw_sigma: %f", accel_bias_rw_sigma);
        RCLCPP_INFO(node_->get_logger(), " - gyro_noise_sigma: %f", gyro_noise_sigma);
        RCLCPP_INFO(node_->get_logger(), " - gyro_bias_rw_sigma: %f", gyro_bias_rw_sigma);

        std::stringstream T_sensor2cam_stream;
        T_sensor2cam_stream << T_sensor2cam;
        RCLCPP_INFO(node_->get_logger(), "T_sensor2cam: \n%s", T_sensor2cam_stream.str().c_str());

        Matrix33 I_33 = Matrix33::Identity();

        Matrix33 measured_acc_cov = I_33 * pow(accel_noise_sigma, 2);
        Matrix33 measured_omega_cov = I_33 * pow(gyro_noise_sigma, 2);
        Matrix33 integration_error_cov = I_33 * 1e-8;  // error committed in integrating position from velocities
        Matrix33 bias_acc_cov = I_33 * pow(accel_bias_rw_sigma, 2);
        Matrix33 bias_omega_cov = I_33 * pow(gyro_bias_rw_sigma, 2);
        // Eigen::Matrix<double,6,6> bias_acc_omega_init = Eigen::Matrix<double,6,6>::Identity() * 1e-5;  // error in the bias used for preintegration
        Matrix6 bias_acc_omega_init = Matrix6::Identity() * 1e-3;

        // the imu is in camera frame, which have y point down, z point forward
        preint_param_ = boost::make_shared<PreintegrationCombinedParams>(Vector3(0.0, 0.0, 0.0));
        // PreintegrationBase params:
        preint_param_->accelerometerCovariance =
            measured_acc_cov;  // acc white noise in continuous
        preint_param_->integrationCovariance =
            integration_error_cov;  // integration uncertainty continuous
        // should be using 2nd order integration
        // PreintegratedRotation params:
        preint_param_->gyroscopeCovariance =
            measured_omega_cov;  // gyro white noise in continuous
        // PreintegrationCombinedMeasurements params:
        preint_param_->biasAccCovariance = bias_acc_cov;      // acc bias in continuous
        preint_param_->biasOmegaCovariance = bias_omega_cov;  // gyro bias in continuous
        preint_param_->biasAccOmegaInt = bias_acc_omega_init;
        
        //set The pose of the sensor in the body frame.
        Pose3 body2sensor = Pose3(T_sensor2cam);
        preint_param_->setBodyPSensor(body2sensor);
    }

    void Backend::setGravity(sensor_msgs::msg::Imu::SharedPtr imu_msg_ptr)
    {
        Eigen::Vector3d gravity(0, 0, -9.81);
        Eigen::Matrix3d orientation = Eigen::Quaterniond(imu_msg_ptr->orientation.w,
                                                    imu_msg_ptr->orientation.x,
                                                    imu_msg_ptr->orientation.y,
                                                    imu_msg_ptr->orientation.z).toRotationMatrix();
        Eigen::Vector3d gravity_trans = orientation.inverse()*gravity;
        preint_param_->n_gravity = Vector3(gravity_trans[0], gravity_trans[1], gravity_trans[2]);

        std::stringstream gravity_trans_stream;
        gravity_trans_stream << gravity_trans;
        RCLCPP_INFO(node_->get_logger(), "Initialize the gravity for IMU with: %s", gravity_trans_stream.str().c_str());
    }

    void Backend::write_to_file(const Values &estimate, const std::string &filename)
    {
        /*
        * 
        * This function writes the estimated poses and tags to a file
        */
        std::fstream stream(filename.c_str(), std::fstream::out);

        // Use a lambda here to more easily modify behavior in future.
        auto index = [](gtsam::Key key) { return Symbol(key).index(); };
        auto key_char = [](gtsam::Key key) { return Symbol(key).chr(); };
        // save 3D poses
        for (const auto key_value : estimate) {
            auto p = dynamic_cast<const GenericValue<Pose3> *>(&key_value.value);
            if (!p)
                continue;
            const gtsam::Key key = key_value.key;
            const gtsam::Pose3 &pose = p->value();
            const gtsam::Point3 t = pose.translation();
            const auto q = pose.rotation().toQuaternion();
            stream << "VERTEX_SE3:QUAT " << key_char(key) <<" " << index(key)
                << " " << t.x() << " "<< t.y() << " " << t.z() << " "
                << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << endl;
        }
        stream.close();
    }

    void Backend::loadMap()
    {
        Values::shared_ptr prior_values = read_from_file(load_map_path_);

        // prior factor graph contains both landmark, poses and their between factors
        // we only need the landmark factors and add them to the factor graph
        landmark_values_ = prior_values->filter(Symbol::ChrTest(kLandmarkSymbol));
        if (fix_prior_)
        {
            fixed_landmark_ = landmark_values_;
        }

        // check if the landmark are loaded correctly
        int num_landmarks = landmark_values_.size();

        for(const auto key_value: landmark_values_) {
            Key landmark_key = key_value.key;
            landmark_cov_[landmark_key] = landmark_prior_cov_;
        }

        if (num_landmarks < 1)
        {
            RCLCPP_WARN(node_->get_logger(), "No landmarks are loaded from the map file, SLAM will initialize from first frame");
            prior_map_ = false;
            return;
        }

        // initialized_ = true;
        RCLCPP_INFO(node_->get_logger(), "Load %d landmarks from %s", num_landmarks, load_map_path_.c_str());
    }

    nav_msgs::msg::Odometry::SharedPtr Backend::createOdomMsg(Pose3 pose, EigenPoseCov pose_cov, 
                                            Vector3 linear_v, Vector3 angular_w, 
                                            double time, int seq)
    {

        if(need_pose_offset_)
        {
            pose = Pose3(pose.matrix() * pose_offset_);

            Eigen::Matrix4d twist_cam;
            twist_cam << 0, -angular_w[2], angular_w[1], linear_v[0], 
                    angular_w[2], 0, -angular_w[0], linear_v[1],
                    -angular_w[1], angular_w[0], 0, linear_v[2],
                    0,0,0,0;

            Eigen::Matrix4d twist = pose_offset_inv_ * twist_cam * pose_offset_;
            linear_v << twist(0,3), twist(1,3), twist(2,3);
            angular_w << twist(2,1), twist(0,2), twist(1,0);
        }

        nav_msgs::msg::Odometry::SharedPtr odom_msg = std::make_shared<nav_msgs::msg::Odometry>();
        odom_msg->header.stamp = rclcpp::Time(time);
        odom_msg->header.frame_id = "map";

        // Pose information in the message
        odom_msg->pose.pose.position.x = pose.x();
        odom_msg->pose.pose.position.y = pose.y();
        odom_msg->pose.pose.position.z = pose.z();

        Eigen::Quaterniond q(pose.rotation().toQuaternion());
        odom_msg->pose.pose.orientation.x = q.x();
        odom_msg->pose.pose.orientation.y = q.y();
        odom_msg->pose.pose.orientation.z = q.z();
        odom_msg->pose.pose.orientation.w = q.w();

        // covariance information in the message
        for (int i = 0; i < 6; i++)
        {
            for (int j = 0; j < 6; j++)
            {
                odom_msg->pose.covariance[6 * i + j] = pose_cov(i, j);
            }
        }

        // NOTE: twist is in body frame
        // twist information in the message
        odom_msg->twist.twist.linear.x = linear_v(0);
        odom_msg->twist.twist.linear.y = linear_v(1);
        odom_msg->twist.twist.linear.z = linear_v(2);

        odom_msg->twist.twist.angular.x = angular_w(0);
        odom_msg->twist.twist.angular.y = angular_w(1);
        odom_msg->twist.twist.angular.z = angular_w(2);
        return odom_msg;
    }

    visualization_msgs::msg::MarkerArray::SharedPtr Backend::createMarkerArray(std_msgs::msg::Header header)
    {
        // initialize the marker array
        visualization_msgs::msg::MarkerArray::SharedPtr marker_array_ptr = std::make_shared<visualization_msgs::msg::MarkerArray>();
        if(reset_mutex_.try_lock())
        {
            // iterate through landmarks, and update them to priors
            for(const auto &key_value: landmark_values_) {
                Key temp_key = key_value.key;
                Pose3 temp_pose = landmark_values_.at<Pose3>(temp_key);
                
                visualization_msgs::msg::Marker marker;
                visualization_msgs::msg::Marker id;
                marker.header = header;
                marker.header.frame_id = "map";

                id.header = header;
                id.header.frame_id = "map";

                Eigen::Quaterniond q(temp_pose.rotation().toQuaternion());

                marker.ns = "landmarks";
                marker.id = static_cast<int>(temp_key);
                marker.type = 1; //Cubic
                marker.action = 0; //add/modify
                marker.lifetime = rclcpp::Duration(0, 0); // explicit in ros2
                marker.scale.x = 0.2;
                marker.scale.y = 0.2;
                marker.scale.z = 0.01;
                // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
                marker.pose.position.x = temp_pose.x();
                marker.pose.position.y = temp_pose.y();
                marker.pose.position.z = temp_pose.z();

                marker.pose.orientation.x = q.x();
                marker.pose.orientation.y = q.y();
                marker.pose.orientation.z = q.z();
                marker.pose.orientation.w = q.w();

                marker.color.r = 1.0f;
                marker.color.g = 1.0f;
                marker.color.b = 1.0f;
                marker.color.a = 0.5f;

                id.ns = "landmark_id";
                id.id = static_cast<int>(temp_key);
                id.type = 9; //Text
                id.action = 0; //add/modify
                id.lifetime = rclcpp::Duration(0, 0); // explicit in ros2
                id.scale.z = 0.04;
                // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
                id.pose.position.x = temp_pose.x();
                id.pose.position.y = temp_pose.y();
                id.pose.position.z = temp_pose.z();

                id.pose.orientation.x = q.x();
                id.pose.orientation.y = q.y();
                id.pose.orientation.z = q.z();
                id.pose.orientation.w = q.w();

                id.color.r = 0.0f;  
                id.color.g = 0.0f;
                id.color.b = 0.0f;
                id.color.a = 1.0f;

                id.text = "Tag " + std::to_string(static_cast<int>(temp_key));

                marker_array_ptr->markers.push_back(id);
                marker_array_ptr->markers.push_back(marker);
            }
            reset_mutex_.unlock();
        }

        return marker_array_ptr;
    }
}