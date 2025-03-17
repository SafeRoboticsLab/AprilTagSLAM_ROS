/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "frontend/tag_detector_cuda.hpp"

namespace tagslam_ros
{
  struct TagDetectorCUDA::AprilTagsImpl
  {
    // Handle used to interface with the stereo library.
    nvAprilTagsHandle april_tags_handle = nullptr;

    // Camera intrinsics
    nvAprilTagsCameraIntrinsics_t cam_intrinsics;

    // Output vector of detected Tags
    std::vector<nvAprilTagsID_t> tags;

    // CUDA stream
    cudaStream_t main_stream = {};

    // CUDA buffers to store the input image.
    nvAprilTagsImageInput_t input_image;

    // CUDA memory buffer container for RGBA images.
    char * input_image_buffer = nullptr;

    // Size of image buffer
    size_t input_image_buffer_size = 0;

    // boolean to indicate if buffer has been created
    bool buffer_created = false;

    void initialize(
      const double tag_edge_size,
      const int max_tag,
      const size_t image_buffer_size,
      const size_t pitch_bytes,
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info,
      bool create_buffer = true)
    {
      assert(april_tags_handle == nullptr && "Already initialized.");

      input_image_buffer_size = image_buffer_size;

      // Get camera intrinsics
      
      // const double * k = camera_info->K; 
      const float fx = static_cast<float>(camera_info->K[0]);
      const float fy = static_cast<float>(camera_info->K[4]);
      const float cx = static_cast<float>(camera_info->K[2]);
      const float cy = static_cast<float>(camera_info->K[5]);
      cam_intrinsics = {fx, fy, cx, cy};

      const uint32_t height = camera_info->height;
      const uint32_t width = camera_info->width;

      // Create AprilTags detector instance and get handle
      const int error = nvCreateAprilTagsDetector(
        &april_tags_handle, width, height, nvAprilTagsFamily::NVAT_TAG36H11,
        &cam_intrinsics, (float) tag_edge_size);
      if (error != 0) {
        throw std::runtime_error(
                "Failed to create NV April Tags detector (error code " +
                std::to_string(error) + ")");
      }

      // Create stream for detection
      cudaStreamCreate(&main_stream);

      // Allocate the output vector to contain detected AprilTags.
      tags.resize(max_tag);

      if(create_buffer)
      {
        setup_image_buffer();
      }    

      // Setup input image.
      input_image.width = width;
      input_image.height = height;
      if(create_buffer){
        
      }
      input_image.pitch = pitch_bytes;
    }

    void setup_image_buffer()
    {
      // Setup input image CUDA buffer.
      const cudaError_t cuda_error = cudaMalloc(&input_image_buffer, input_image_buffer_size);
      if (cuda_error != cudaSuccess) {
        throw std::runtime_error("Could not allocate CUDA memory (error code " +
                std::to_string(cuda_error) + ")");
      }
      // assigne pointer to the image buffer
      input_image.dev_ptr = reinterpret_cast<uchar4 *>(input_image_buffer);
      buffer_created = true;
    }

    void copy_to_buffer(cv::Mat & cv_mat_cpu)
    {
      if(!buffer_created)
      {
        setup_image_buffer();
      }
      // copy cv mat from host to iamge buffer on device
      const cudaError_t cuda_error = cudaMemcpy(input_image_buffer, cv_mat_cpu.ptr(), 
                                input_image_buffer_size, cudaMemcpyHostToDevice);
      if (cuda_error != cudaSuccess) {
        RCLCPP_ERROR(node_->get_logger(), "Could not memcpy to device CUDA memory (error code %s)", std::to_string(cuda_error).c_str());
      }
    }

    void assign_image(cv::cuda::GpuMat& cv_mat_gpu)
    {
      // assign a image that already on the device as input image
      if(buffer_created)
      {
        cudaFree(input_image_buffer);
        input_image_buffer = nullptr;
      }
      input_image.dev_ptr = reinterpret_cast<uchar4 *>(cv_mat_gpu.ptr());
    }

    ~AprilTagsImpl()
    {
      if (april_tags_handle != nullptr) {
        cudaStreamDestroy(main_stream);
        nvAprilTagsDestroy(april_tags_handle);
        if(buffer_created)
          cudaFree(input_image_buffer);
      }
    }
  };

  TagDetectorCUDA::TagDetectorCUDA(std::shared_ptr<rclcpp::Node> node):
    TagDetector(node),
    node_(node), // Instance variable of node to be used by other functions
    // parameters
    tag_size_(get_ros_option<double>(node_, "frontend/tag_size", 1.0)),
    max_tags_(get_ros_option<int>(node_, "frontend/max_tags", 20))
  {
    // Create the AprilTags detector instance.
    impl_ = std::make_unique<AprilTagsImpl>();
  }

  void TagDetectorCUDA::detectTags(const sensor_msgs::msg::Image::ConstSharedPtr msg_img,
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg_cam_info, 
      TagDetectionArrayPtr static_tag_array_ptr, TagDetectionArrayPtr dyn_tag_array_ptr)
  {
    /*
    Perform Detection from a sensor_msgs::msg::Image::ConstSharedPtr on CPU
    */
    // Convert frame to 8-bit RGBA image
    cv::Mat img_rgba8;

    try{
      img_rgba8 = cv_bridge::toCvCopy(msg_img, "rgba8")->image;
    }catch (cv_bridge::Exception& e){
      RCLCPP_ERROR(node_->get_logger(), "cv_bridge exception: %s", e.what());
    }

    detectTags(img_rgba8, msg_cam_info, msg_img->header,
            static_tag_array_ptr, dyn_tag_array_ptr);
  }

#ifndef NO_CUDA_OPENCV
  void TagDetectorCUDA::detectTags(cv::cuda::GpuMat& cv_mat_gpu,
      const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg_cam_info, std_msgs::msg::Header header, 
      TagDetectionArrayPtr static_tag_array_ptr, TagDetectionArrayPtr dyn_tag_array_ptr)
  {
    image_geometry::PinholeCameraModel camera_model;
    camera_model.fromCameraInfo(msg_cam_info);

    // Get camera intrinsic properties for rectified image.
    double fx = camera_model.fx(); // focal length in camera x-direction [px]
    double fy = camera_model.fy(); // focal length in camera y-direction [px]
    double cx = camera_model.cx(); // optical center x-coordinate [px]
    double cy = camera_model.cy(); // optical center y-coordinate [px]

    cameraMatrix_ = cv::Matx33d(fx, 0, cx,
                              0, fy, cy,
                              0, 0, 1);

    distCoeffs_ = camera_model.distortionCoeffs();
    
    // Setup detector on first frame
    if (impl_->april_tags_handle == nullptr) {
      size_t buffer_size = cv_mat_gpu.rows*cv_mat_gpu.cols*cv_mat_gpu.elemSize();
      impl_->initialize(
          tag_size_,
          max_tags_,
          buffer_size, 
          cv_mat_gpu.step,
          msg_cam_info);
      RCLCPP_INFO(node_->get_logger(), "CUDA Apriltag Detector Initialized.");
    }
    
    // assign the pointer to gpu mat
    impl_->assign_image(cv_mat_gpu);
    
    runDetection(static_tag_array_ptr, dyn_tag_array_ptr);

    static_tag_array_ptr->header = header;
    dyn_tag_array_ptr->header = header;

  }
#endif

  void TagDetectorCUDA::detectTags(cv::Mat& cv_mat_cpu,
          const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg_cam_info, std_msgs::msg::Header header,
          TagDetectionArrayPtr static_tag_array_ptr, TagDetectionArrayPtr dyn_tag_array_ptr)
  {
    image_geometry::PinholeCameraModel camera_model;
    camera_model.fromCameraInfo(msg_cam_info);

    // Get camera intrinsic properties for rectified image.
    double fx = camera_model.fx(); // focal length in camera x-direction [px]
    double fy = camera_model.fy(); // focal length in camera y-direction [px]
    double cx = camera_model.cx(); // optical center x-coordinate [px]
    double cy = camera_model.cy(); // optical center y-coordinate [px]

    cameraMatrix_ = cv::Matx33d(fx, 0, cx,
                              0, fy, cy,
                              0, 0, 1);

    distCoeffs_ = camera_model.distortionCoeffs();
    
    // Setup detector on first frame
    if (impl_->april_tags_handle == nullptr) {
      impl_->initialize(
          tag_size_,
          max_tags_,
          cv_mat_cpu.total() * cv_mat_cpu.elemSize(), 
          cv_mat_cpu.step,
          msg_cam_info);

      RCLCPP_INFO(node_->get_logger(), "CUDA Apriltag Detector Initialized.");
    }

    impl_->copy_to_buffer(cv_mat_cpu);
    
    runDetection(static_tag_array_ptr, dyn_tag_array_ptr);
    
    static_tag_array_ptr->header = header;
    dyn_tag_array_ptr->header = header;

  }

  void TagDetectorCUDA::runDetection(TagDetectionArrayPtr static_tag_array_ptr,
                              TagDetectionArrayPtr dyn_tag_array_ptr){

    // Perform detection
    uint32_t num_detections;
    const int error = nvAprilTagsDetect(
      impl_->april_tags_handle, &(impl_->input_image), impl_->tags.data(),
      &num_detections, max_tags_, impl_->main_stream);
    if (error != 0) {
      RCLCPP_ERROR(node_->get_logger(), "Failed to run AprilTags detector (error code %d)", error);
      return;
    }
    
    // Parse detections into published protos
    
    for (uint32_t i = 0; i < num_detections; i++) {
      // detection
      const nvAprilTagsID_t & detection = impl_->tags[i];

      int tagID = detection.id;
      double cur_tag_size = tag_size_;
      bool cur_tag_static = true;
      // try to see if the tag is in the list of tags to be detected
      if(tag_size_list_.find(tagID) != tag_size_list_.end())
      {
        cur_tag_size = tag_size_list_[tagID].first;
        cur_tag_static = tag_size_list_[tagID].second;
      }

      //make pose
      // geometry_msgs::msg::Pose tag_pose = DetectionToPose(detection);
      // instead of pose generated from detection, we solve them us PnP due to various tag size

      // Get estimated tag pose in the camera frame.
      //
      // Note on frames:
      // we want:
      //   - camera frame: looking from behind the camera (like a
      //     photographer), x is right, y is down and z is straight
      //     ahead
      //   - tag frame: looking straight at the tag (oriented correctly),
      //     x is right, y is up and z is towards you (out of the tag).
      
      //from top left to bottom left and going clockwise
      std::vector<cv::Point2d> TagImagePoints;
      for (int corner_idx = 0; corner_idx < 4; corner_idx++) {
        TagImagePoints.push_back(cv::Point2d(detection.corners[corner_idx].x,
                                    detection.corners[corner_idx].y));
      }

      std::vector<cv::Point3d> TagObjectPoints;

       //from top left to bottom left and going clockwise
      TagObjectPoints.push_back(cv::Point3d(-cur_tag_size / 2, cur_tag_size / 2, 0));
      TagObjectPoints.push_back(cv::Point3d(cur_tag_size / 2, cur_tag_size / 2, 0));
      TagObjectPoints.push_back(cv::Point3d(cur_tag_size / 2, -cur_tag_size / 2, 0));
      TagObjectPoints.push_back(cv::Point3d(-cur_tag_size / 2, -cur_tag_size / 2, 0));

      EigenPose T_tag_to_cam = getRelativeTransform(TagObjectPoints,TagImagePoints, cameraMatrix_, distCoeffs_);
      EigenPose T_tag_to_ros = T_cam_to_ros_ * T_tag_to_cam;
      
      geometry_msgs::msg::Pose tag_pose = createPoseMsg(T_tag_to_ros);

      // create message
      tagslam_ros::msg::AprilTagDetection tag_detection;
      tag_detection.id = detection.id;
      tag_detection.size = cur_tag_size;
      tag_detection.static_tag = cur_tag_static;
      tag_detection.pose = tag_pose;
      
      // corners
      for (int corner_idx = 0; corner_idx < 4; corner_idx++) {
        tag_detection.corners.data()[corner_idx].x =
          detection.corners[corner_idx].x;
        tag_detection.corners.data()[corner_idx].y =
          detection.corners[corner_idx].y;
      }

      // center
      const float slope_1 = (detection.corners[2].y - detection.corners[0].y) /
        (detection.corners[2].x - detection.corners[0].x);
      const float slope_2 = (detection.corners[3].y - detection.corners[1].y) /
        (detection.corners[3].x - detection.corners[1].x);
      const float intercept_1 = detection.corners[0].y -
        (slope_1 * detection.corners[0].x);
      const float intercept_2 = detection.corners[3].y -
        (slope_2 * detection.corners[3].x);
      tag_detection.center.x = (intercept_2 - intercept_1) / (slope_1 - slope_2);
      tag_detection.center.y = (slope_2 * intercept_1 - slope_1 * intercept_2) /
        (slope_2 - slope_1);

      if(cur_tag_static){
        static_tag_array_ptr->detections.push_back(tag_detection);
      }else{
        dyn_tag_array_ptr->detections.push_back(tag_detection);
      }
    }
  }

  geometry_msgs::msg::Pose TagDetectorCUDA::DetectionToPose(const nvAprilTagsID_t & detection)
  {
    geometry_msgs::msg::Pose pose;
    pose.position.x = detection.translation[0];
    pose.position.y = detection.translation[1];
    pose.position.z = detection.translation[2];

    // Rotation matrix from nvAprilTags is column major
    const Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::ColMajor>>
    orientation(detection.orientation);
    const Eigen::Quaternion<float> q(orientation);

    pose.orientation.w = q.w();
    pose.orientation.x = q.x();
    pose.orientation.y = q.y();
    pose.orientation.z = q.z();

    return pose;
  }

  TagDetectorCUDA::~TagDetectorCUDA() = default;



}  // namespace tagslam_ros

