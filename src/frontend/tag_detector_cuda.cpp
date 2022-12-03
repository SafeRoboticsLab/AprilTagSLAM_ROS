/**
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "frontend/tag_detector_cuda.h"

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
      const sensor_msgs::CameraInfoConstPtr & camera_info,
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
        ROS_ERROR("Could not memcpy to device CUDA memory (error code %s)", std::to_string(cuda_error).c_str());
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

  TagDetectorCUDA::TagDetectorCUDA(ros::NodeHandle pnh):
    TagDetector(),
    // parameter
    tag_size_(getRosOption<double>(pnh, "frontend/tag_size", 1.0)),
    max_tags_(getRosOption<int>(pnh, "frontend/max_tags", 20))
  {
    // Create the AprilTags detector instance.
    impl_ = std::make_unique<AprilTagsImpl>();
  }

  TagDetectionArrayPtr TagDetectorCUDA::detectTags(const sensor_msgs::ImageConstPtr& msg_img,
      const sensor_msgs::CameraInfoConstPtr& msg_cam_info)
  {
    /*
    Perform Detection from a sensor_msgs::ImageConstPtr on CPU
    */
    // Convert frame to 8-bit RGBA image
    cv::Mat img_rgba8;

    try{
      img_rgba8 = cv_bridge::toCvShare(msg_img, "rgba8")->image;
    }catch (cv_bridge::Exception& e){
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return nullptr;
    }

    return detectTags(img_rgba8, msg_cam_info, msg_img->header);
  }

#ifndef NO_CUDA_OPENCV
  TagDetectionArrayPtr TagDetectorCUDA::detectTags(cv::cuda::GpuMat& cv_mat_gpu,
      const sensor_msgs::CameraInfoConstPtr& msg_cam_info, std_msgs::Header header)
  {
    // Setup detector on first frame
    if (impl_->april_tags_handle == nullptr) {
      size_t buffer_size = cv_mat_gpu.rows*cv_mat_gpu.cols*cv_mat_gpu.elemSize();
      impl_->initialize(
          tag_size_,
          max_tags_,
          buffer_size, 
          cv_mat_gpu.step,
          msg_cam_info);
      ROS_INFO("CUDA Apriltag Detector Initialized.");
    }
    
    // assign the pointer to gpu mat
    impl_->assign_image(cv_mat_gpu);
    
    auto tag_detection_array = runDetection();

    if(tag_detection_array)
      tag_detection_array->header = header;

    return tag_detection_array;
  }
#endif

  TagDetectionArrayPtr TagDetectorCUDA::detectTags(cv::Mat& cv_mat_cpu,
          const sensor_msgs::CameraInfoConstPtr& msg_cam_info, std_msgs::Header header)
  {
    // Setup detector on first frame
    if (impl_->april_tags_handle == nullptr) {
      impl_->initialize(
          tag_size_,
          max_tags_,
          cv_mat_cpu.total() * cv_mat_cpu.elemSize(), 
          cv_mat_cpu.step,
          msg_cam_info);
      ROS_INFO("CUDA Apriltag Detector Initialized.");
    }

    impl_->copy_to_buffer(cv_mat_cpu);
    
    auto tag_detection_array = runDetection();
    if(tag_detection_array)
      tag_detection_array->header = header;

    return tag_detection_array;

  }

  TagDetectionArrayPtr TagDetectorCUDA::runDetection(){
    auto start = std::chrono::system_clock::now();
    // Perform detection
    uint32_t num_detections;
    const int error = nvAprilTagsDetect(
      impl_->april_tags_handle, &(impl_->input_image), impl_->tags.data(),
      &num_detections, max_tags_, impl_->main_stream);
    if (error != 0) {
      ROS_ERROR("Failed to run AprilTags detector (error code %d)", error);
      return nullptr;
    }
    
    // Parse detections into published protos
    auto tag_detection_array = std::make_shared<AprilTagDetectionArray>();
    
    
    for (uint32_t i = 0; i < num_detections; i++) {
      // detection
      const nvAprilTagsID_t & detection = impl_->tags[i];

      //make pose
      geometry_msgs::Pose tag_pose = DetectionToPose(detection);

      // create message
      AprilTagDetection tag_detection;
      tag_detection.id = detection.id;
      tag_detection.size = tag_size_;
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

          
      tag_detection_array->detections.push_back(tag_detection);
    }
    return tag_detection_array;
  }

  geometry_msgs::Pose TagDetectorCUDA::DetectionToPose(const nvAprilTagsID_t & detection)
  {
    geometry_msgs::Pose pose;
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

