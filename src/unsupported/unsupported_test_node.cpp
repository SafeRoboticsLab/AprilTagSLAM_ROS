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

#include "backend/isam2_backend.h"
#include "backend/fixed_lag_backend.h"
#include <sstream>
#include <unordered_map>
#include <random>
#include <matplot/matplot.h>
#include <cmath>        // std::abs

using namespace tagslam_ros;
using namespace gtsam;

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
    is >> roll >> pitch >> yaw; // notice order !
    R = Rot3::Ypr(yaw, pitch, roll);
    return is;
}


EigenPose genNormalRandomPose(double sigma_trans, double sigma_rot){
  std::random_device mch;
  std::default_random_engine generator(mch());
  std::normal_distribution<double> trans_distribution(0, sigma_trans);
  std::normal_distribution<double> rot_distribution(0, sigma_rot);

  double x = trans_distribution(generator);
  double y = trans_distribution(generator);
  double z = trans_distribution(generator);

  double rot_x = rot_distribution(generator);
  double rot_y = rot_distribution(generator);
  double rot_z = rot_distribution(generator);

  Eigen::Affine3d pose;

  pose.translation() << x, y, z;
  Eigen::Quaterniond rot;
  rot = Eigen::AngleAxisd(rot_x, Eigen::Vector3d::UnitX())
      * Eigen::AngleAxisd(rot_y, Eigen::Vector3d::UnitY())
      * Eigen::AngleAxisd(rot_z, Eigen::Vector3d::UnitZ());
  pose.linear() = rot.toRotationMatrix();
  return pose.matrix();
}

struct Motion{
  int step = 0;
  EigenPose odom = EigenPose::Identity();
  TagDetectionArrayPtr obs = nullptr;
};

class Env
{
  public:
    Env(int num_landmarks, double space_lim, double p_obs = 0.8){
      generator_ = std::default_random_engine(mch_());
      // initialize random number generator
      obs_distribution_ = std::bernoulli_distribution(p_obs);
      trans_distribution_ = std::uniform_real_distribution<double>(0.0, space_lim);
      rot_distribution_ = std::uniform_real_distribution<double>(-1*M_PI, M_PI);

      // initialize landmarks
      num_landmarks_ = num_landmarks;
      for(int i=0; i<num_landmarks; i++)
      {
        landmarks[i] = genUniformRandomPose();
        // ROS_INFO_STREAM("landmark " << i << " : " << landmarks[i]);
      }
    }

    Env(std::string & filename, double p_obs=0.8)
    {
      generator_ = std::default_random_engine(mch_());
      // initialize random number generator
      obs_distribution_ = std::bernoulli_distribution(p_obs);

    // Parse the file
      std::ifstream is(filename.c_str());
            
      if (!is){
          std::cout<<"\033[1;31mbold parse: can not find file " <<filename<<"\033[0m\n"<<std::endl;
          return;
      }
      
      std::string tag;
      int new_id = 0;
      while (is >> tag) {
          if (tag == "VERTEX3") {
              unsigned char key_char;
              size_t id;
              double x, y, z;
              gtsam::Rot3 R;
              is >> key_char >> id >> x >> y >> z >> R;
              if(key_char == kLandmarkSymbol){
                landmarks[new_id] = gtsam::Pose3(R, gtsam::Point3(x, y, z)).matrix();
                new_id++;
              }
          } else if (tag == "VERTEX_SE3:QUAT") {
              unsigned char key_char;
              size_t id;
              double x, y, z;
              gtsam::Quaternion q;
              is >> key_char >> id >> x >> y >> z >> q;
              gtsam::Symbol key(key_char, id);
              if(key_char == kLandmarkSymbol){
                landmarks[new_id] = gtsam::Pose3(q, gtsam::Point3(x, y, z)).matrix();
                new_id++;
              }
          } 
          is.ignore(LINESIZE, '\n');
          
      }
      ROS_INFO_STREAM("Loaded " << new_id << " landmarks from file " << filename);

      num_landmarks_ = new_id;
    }

    TagDetectionArrayPtr getObs(const EigenPose& pose, double sigma_trans, double sigma_rot){
      TagDetectionArrayPtr observations(new AprilTagDetectionArray);
      for (int i=0; i<num_landmarks_; i++){
        if (obs_distribution_(generator_)){
          EigenPose landmark_pose = landmarks[i];
          EigenPose obs_pose = pose.inverse() * landmark_pose;
          EigenPose noised_obs_pose = obs_pose * genNormalRandomPose(sigma_trans, sigma_rot);
          tagslam_ros::msg::AprilTagDetection detection;
          detection.id = i;
          detection.pose = createPoseMsg(noised_obs_pose);
          observations->detections.push_back(detection);
        }
      }
      
      return observations;
    }

    void visualize_landmark(EigenPoseMap & estimated_landmarks){
      using namespace matplot;
      std::vector<double> pos_error;
      std::vector<double> rot_error;
      std::vector<double> index;
      for(int i = 0; i<num_landmarks_; i++){
        if(estimated_landmarks.find(i) != estimated_landmarks.end()){
          EigenPose est_landmark = estimated_landmarks[i];
          EigenPose gt_landmark = landmarks[i];
          EigenPose error = gt_landmark.inverse() * est_landmark;
          pos_error.push_back((est_landmark.block<3,1>(0,3) - gt_landmark.block<3,1>(0,3)).norm());
          Eigen::Quaterniond q_est(est_landmark.block<3,3>(0,0));
          Eigen::Quaterniond q_gt(gt_landmark.block<3,3>(0,0));
          rot_error.push_back(1 - std::abs(q_est.dot(q_gt)));
          index.push_back(i);
        }
      }
      auto f = figure(true);
      f->size(1800, 1000);
      subplot(2,1,1);
      plot(index, pos_error, "k*");
      title("pos error");
      subplot(2,1,2);
      plot(index, rot_error, "k*");
      title("rot error");
      
      show();
    }


  private:
    EigenPose genUniformRandomPose(){
      double x = trans_distribution_(generator_);
      double y = trans_distribution_(generator_);
      double z = trans_distribution_(generator_);

      double rot_x = rot_distribution_(generator_);
      double rot_y = rot_distribution_(generator_);
      double rot_z = rot_distribution_(generator_);

      Eigen::Affine3d pose;

      pose.translation() << x, y, z;
      
      Eigen::Quaterniond rot;
      rot = Eigen::AngleAxisd(rot_x, Eigen::Vector3d::UnitX())
          * Eigen::AngleAxisd(rot_y, Eigen::Vector3d::UnitY())
          * Eigen::AngleAxisd(rot_z, Eigen::Vector3d::UnitZ());
      pose.linear() = rot.toRotationMatrix();
      return pose.matrix();
    }

  private:
    int num_landmarks_;
    EigenPoseMap landmarks;

    // random generator
    std::random_device mch_;
    std::default_random_engine generator_;
    std::uniform_real_distribution<double> trans_distribution_;
    std::uniform_real_distribution<double> rot_distribution_; 
    std::bernoulli_distribution obs_distribution_;
};

class Agent{
  public:
    Agent(double sigma_trans, double sigma_rot,
      double odom_noise_trans, double odom_noise_rot,
      double landmark_noise_trans, double landmark_noise_rot):
            sigma_trans_(sigma_trans),
            sigma_rot_(sigma_rot),
            odom_noise_trans_(odom_noise_trans),
            odom_noise_rot_(odom_noise_rot),
            landmark_noise_trans_(landmark_noise_trans),
            landmark_noise_rot_(landmark_noise_rot)
    {
        // initialize pose at origin
        pose = EigenPose::Identity();
        step_ = 0;
    }

    Motion one_step(Env & env){
      EigenPose odom = genNormalRandomPose(sigma_trans_, sigma_rot_);
      //ROS_INFO_STREAM("odom: " << odom);
      // slam treat first step as origin
      if (step_ > 0)
        pose = pose * odom;

      TagDetectionArrayPtr obs = env.getObs(pose, landmark_noise_trans_, landmark_noise_rot_);

      Motion motion;
      motion.step = step_;
      motion.odom = odom*genNormalRandomPose(odom_noise_trans_, odom_noise_rot_);
      motion.obs = obs;

      trajectory[step_] = pose;
      step_++;

      return motion;
    }

    void visualize_trajectory(EigenPoseMap & estimated_trajectory){
      using namespace matplot;
      std::vector<double> x_est, y_est, z_est, x_gt, y_gt, z_gt;
      std::vector<double> rot_error;
      std::vector<double> steps;
      for(int i =0; i<step_; i++){
        steps.push_back((double)i);
        x_est.push_back(estimated_trajectory[i](0,3));
        y_est.push_back(estimated_trajectory[i](1,3));
        z_est.push_back(estimated_trajectory[i](2,3));

        x_gt.push_back(trajectory[i](0,3));
        y_gt.push_back(trajectory[i](1,3));
        z_gt.push_back(trajectory[i](2,3));

        // https://www.cs.cmu.edu/~cga/dynopt/readings/Rmetric.pdf
        Eigen::Quaterniond q_est(estimated_trajectory[i].block<3,3>(0,0));
        Eigen::Quaterniond q_gt(trajectory[i].block<3,3>(0,0));
        rot_error.push_back(1 - std::abs(q_est.dot(q_gt)));
      }
      auto f = figure(true);
      f->size(1800, 1000);
      subplot(2,2,1);
      plot(steps, x_est, "r--", steps, x_gt, "k-");
      title("x");
      subplot(2,2,2);
      plot(steps, y_est, "r--", steps, y_gt, "k-");
      title("y");
      subplot(2,2,3);
      plot(steps, z_est, "r--", steps, z_gt, "k-");
      title("z");
      subplot(2,2,4);
      plot(steps, rot_error, "k-");
      title("rot error");
      

      show();
    }
    
  private:
    int step_ = 0;

    double sigma_trans_;
    double sigma_rot_;

    double odom_noise_trans_;
    double odom_noise_rot_;

    double landmark_noise_trans_;
    double landmark_noise_rot_;

    EigenPoseMap trajectory;

    EigenPose pose;
};

int main(int argc, char **argv)
{
  
  ros::init(argc, argv, "test_node");
  ros::NodeHandle pnh("~");
  
  rclcpp::Rate loop_rate(10);
  
  double dt = 0.05;

  // Read parameters
  int num_steps = get_ros_option<int>(pnh, "num_steps", 100);
  // landmark parameters
  int num_landmarks = get_ros_option<int>(pnh, "num_landmarks", 50);
  double space_lim = get_ros_option<double>(pnh, "space_lim", 10.0);

  // pose parameters
  double sigma_trans = get_ros_option<double>(pnh, "sigma_trans", 1);
  double sigma_rot = get_ros_option<double>(pnh, "sigma_rot", 0.5);

  double odom_noise_trans = get_ros_option<double>(pnh, "odom_noise_trans", 0.2);
  double odom_noise_rot = get_ros_option<double>(pnh, "odom_noise_rot", 0.3);

  double p_obs = get_ros_option<double>(pnh, "p_obs", 0.5);
  double landmark_noise_trans = get_ros_option<double>(pnh, "landmark_noise_trans", 0.2);
  double landmark_noise_rot = get_ros_option<double>(pnh, "landmark_noise_rot", 0.3);

  std::string backend_type = get_ros_option<std::string>(pnh, "backend/type", "isam2");
  std::string load_path = get_ros_option<std::string>(pnh, "backend/load_path", "");

  bool plot_figure = get_ros_option<bool>(pnh, "plot_figure", false);
  
  // Create environment
  std::shared_ptr<Env> env;
  if (load_path.empty())
    env = std::make_shared<Env>(num_landmarks, space_lim, p_obs);
  else
    env = std::make_shared<Env>(load_path, p_obs);
    
  Agent agent(sigma_trans, sigma_rot, odom_noise_trans, odom_noise_rot,
              landmark_noise_trans, landmark_noise_rot);
  

  // SLAM backend
  std::shared_ptr<Backend> slam_backend;
  if(backend_type=="isam2")
    slam_backend = std::make_shared<iSAM2Backend>(pnh);
  else
    slam_backend = std::make_shared<FixedLagBackend>(pnh);

  int step = 0;

  EigenPoseSigma sigma;
  sigma << Vector3::Constant(odom_noise_rot),Vector3::Constant(odom_noise_trans);

  EigenPoseCov cov = sigma.array().pow(2).matrix().asDiagonal();
  
  std::vector<double> time_duration;
  EigenPoseMap estimated_poses;
  while (step < num_steps)
  {
    Motion motion = agent.one_step(*env);
    step = motion.step;
    TagDetectionArrayPtr obs = motion.obs;
    obs->header.stamp = rclcpp::Time(step*dt);
    EigenPose odom = motion.odom;
    clock_t t0 = clock();
    estimated_poses[step] = slam_backend->updateSLAM(obs, odom, cov);
    double dur = ((double)clock() - t0)/CLOCKS_PER_SEC;
    time_duration.push_back(dur);
  }

  double sum = std::accumulate(time_duration.begin(), time_duration.end(), 0.0);
  double mean = sum / time_duration.size();

  std::vector<double> diff(time_duration.size());
  std::transform(time_duration.begin(), time_duration.end(), diff.begin(),
                std::bind2nd(std::minus<double>(), mean));
  double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  double stdev = std::sqrt(sq_sum / time_duration.size());

  std::cout << "mean: " << mean << " [s] with std: "<< stdev<<std:: endl;
  if(plot_figure){
    using namespace matplot;
    auto f = figure(true);
    f->size(1800, 1000);
    plot(time_duration);
    title("time duration");
    show();

    // get landmark poses
    EigenPoseMap estimated_landmarks;
    slam_backend->getPoses(estimated_landmarks, kLandmarkSymbol);
    agent.visualize_trajectory(estimated_poses);
    env->visualize_landmark(estimated_landmarks);
  }

  rclcpp::shutdown();
  return 0;
}