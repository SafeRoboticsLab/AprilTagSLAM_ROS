cmake_minimum_required(VERSION 3.8)
project(tagslam_ros)
# set(CMAKE_PROJECT_NAME tagslam_ros) # can probably remove

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
# We default to 'Release' if none was specified
IF(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  MESSAGE(STATUS "Setting build type to 'Release' as none was specified.")
  SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the build type" FORCE)
  # SET(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the build type" FORCE)
  SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Coverage" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
ENDIF()
message( STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}" )

###########################################
###### Set build parameter ################
###########################################
option(USE_CUDA "Build with CUDA" ON)
option(USE_ZED "Build with ZED SDK" ON)
option(USE_CUDA_OPENCV "Build with OpenCV CUDA" ON)

if(NOT USE_CUDA)
  # these packages depend on CUDA
  SET(USE_ZED  OFF)
  SET(USE_CUDA_OPENCV OFF)
endif()

MESSAGE("Use CUDA: " ${USE_CUDA})
MESSAGE("Use ZED SDK: " ${USE_ZED})
MESSAGE("Use OpenCV CUDA: " ${USE_CUDA_OPENCV})

set(BUILD_SHARED_LIBS ON)

# add_compile_options("-O3" "-funsafe-loop-optimizations" "-fsee" "-funroll-loops" "-fno-math-errno" "-funsafe-math-optimizations" "-ffinite-math-only" "-fno-signed-zeros")

# force use -march=native
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(image_geometry REQUIRED)
find_package(image_transport REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)  
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)  
find_package(rosidl_cmake)

find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(apriltag REQUIRED) # front end

# needed for backend with GTSAM
find_package(Eigen3 3.3.4 REQUIRED)
find_package(Boost 1.58 REQUIRED COMPONENTS system serialization filesystem thread date_time regex timer chrono)
find_package(GTSAM REQUIRED)
find_package(GTSAM_UNSTABLE REQUIRED)
find_package(TBB REQUIRED)

# TODO: add linting in future
# find_package(ament_lint_auto REQUIRED)
# ament_lint_auto_find_test_dependencies() 

# CUDA
if(USE_CUDA)
  set(CUDA_MIN_VERSION "10.2")
  find_package(CUDA ${CUDA_MIN_VERSION} REQUIRED)
  include_directories(include ${CUDA_INCLUDE_DIRS})
else()
  add_compile_definitions(NO_CUDA)
endif()

# ZED
if(USE_ZED)
  find_package(ZED 3)
else()
  add_compile_definitions(NO_ZED)
endif()

if(NOT USE_CUDA_OPENCV)
  add_compile_definitions(NO_CUDA_OPENCV)
endif()


# custom messages
set(msg_files
  "msg/AprilTagDetection.msg"
  "msg/AprilTagDetectionArray.msg"
)

# rosidl_generate_interfaces(${PROJECT_NAME}
#   ${msg_files}
#   DEPENDENCIES geometry_msgs std_msgs
# )

# Extract the include directories and libraries from apriltag::apriltag as colcon does not support modern cmake? perhaps
get_target_property(apriltag_INCLUDE_DIRS apriltag::apriltag INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(apriltag_LIBRARIES apriltag::apriltag INTERFACE_LINK_LIBRARIES)

# ament_package
ament_export_dependencies(
  cv_bridge REQUIRED
  geometry_msgs REQUIRED
  image_transport REQUIRED
  pluginlib REQUIRED
  rclcpp REQUIRED
  rclcpp_components REQUIRED
  sensor_msgs REQUIRED
  std_msgs REQUIRED
  visualization_msgs REQUIRED
)
ament_export_dependencies(rosidl_default_runtime) # required for custom messages

###########
## Build ##
###########

include_directories(include
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${GTSAM_INCLUDE_DIR}
  ${GTSAM_UNSTABLE_INCLUDE_DIR}
)

# ################################
# ## Add library for the front-end
# ################################

# NVAprilTags
if(USE_CUDA)
  execute_process(COMMAND uname -m COMMAND tr -d '\n'
    OUTPUT_VARIABLE ARCHITECTURE
  )
  include_directories(nvapriltags/nvapriltags)
  add_library(nvapriltags STATIC IMPORTED)
  if( ${ARCHITECTURE} STREQUAL "x86_64" )
    set_property(TARGET nvapriltags PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/nvapriltags/lib_x86_64_cuda_11_4/libapril_tagging.a)
  elseif( ${ARCHITECTURE} STREQUAL "aarch64" )
    set_property(TARGET nvapriltags PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/nvapriltags/lib_aarch64_jetpack44/libapril_tagging.a)
  endif()
endif()

rosidl_get_typesupport_target(cpp_typesupport_target
  "${PROJECT_NAME}" "rosidl_typesupport_cpp")

################################
## Add library for the front-end
################################

if(USE_CUDA)
  add_library(${PROJECT_NAME}_frontend 
            src/frontend/tag_detector.cpp
            src/frontend/tag_detector_cpu.cpp
            src/frontend/tag_detector_cuda.cpp)
  target_include_directories(${PROJECT_NAME}_frontend PUBLIC
            ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

  target_link_libraries(${PROJECT_NAME}_frontend
            ${OpenCV_LIBRARIES}
            ${CUDA_LIBRARIES}
            apriltag::apriltag
            nvapriltags
            "${cpp_typesupport_target}")

  ament_target_dependencies(${PROJECT_NAME}_frontend
            cv_bridge
            geometry_msgs
            image_geometry
            image_transport
            nav_msgs
            pluginlib
            rclcpp
            rclcpp_components
            rosidl_default_generators
            sensor_msgs
            std_msgs
            visualization_msgs
            ) 
else()

  add_library(${PROJECT_NAME}_frontend 
            src/frontend/tag_detector.cpp
            src/frontend/tag_detector_cpu.cpp)

  ament_target_dependencies(${PROJECT_NAME}_frontend
            cv_bridge
            geometry_msgs
            image_geometry
            image_transport
            nav_msgs
            pluginlib
            rclcpp
            rclcpp_components
            rosidl_default_generators
            sensor_msgs
            std_msgs
            visualization_msgs
            )
  target_link_libraries(${PROJECT_NAME}_frontend
            ${OpenCV_LIBRARIES}
            apriltag::apriltag
            "${cpp_typesupport_target}"
            )
  # to use custom messages
  # rosidl_target_interfaces(${PROJECT_NAME}_frontend
  # ${PROJECT_NAME} "rosidl_typesupport_cpp")
endif()



################################
## Add library for the back-end
################################
add_library(${PROJECT_NAME}_backend 
            src/backend/backend.cpp
            src/backend/isam2_backend.cpp
            src/backend/fixed_lag_backend.cpp)

ament_target_dependencies(${PROJECT_NAME}_backend
            cv_bridge
            geometry_msgs
            image_geometry
            image_transport
            nav_msgs
            pluginlib
            rclcpp
            rclcpp_components
            rosidl_default_generators
            sensor_msgs
            std_msgs
            visualization_msgs
            )

target_link_libraries(${PROJECT_NAME}_backend
            ${Boost_LIBRARIES}
            gtsam
            gtsam_unstable
            tbb
            "${cpp_typesupport_target}"
            )
  # to use custom messages
  # rosidl_target_interfaces(${PROJECT_NAME}_backend
  # ${PROJECT_NAME} "rosidl_typesupport_cpp")

# ################################
# ## Add library for the SLAM
# ################################

# no longer supported
# add_library(${PROJECT_NAME}_slam src/tag_slam.cpp)
# add_dependencies(${PROJECT_NAME}_slam ${PROJECT_NAME}_generate_messages_cpp)
# target_link_libraries(${PROJECT_NAME}_slam
#                       ${PROJECT_NAME}_frontend 
#                       ${PROJECT_NAME}_backend
#                       ${catkin_LIBRARIES} )
                      

################################
## SLAM ZED
################################
add_library(${PROJECT_NAME}_slam_zed src/tag_slam_zed.cpp)
# add_dependencies(${PROJECT_NAME}_slam_zed ${PROJECT_NAME}_generate_messages_cpp)
target_include_directories(${PROJECT_NAME}_slam_zed PUBLIC
                      ${ZED_INCLUDE_DIRS})

ament_target_dependencies(${PROJECT_NAME}_slam_zed
                      cv_bridge
                      geometry_msgs
                      image_geometry
                      image_transport
                      nav_msgs
                      pluginlib
                      rclcpp
                      rclcpp_components
                      rosidl_default_generators
                      sensor_msgs
                      std_msgs
                      visualization_msgs
                      )

target_link_libraries(${PROJECT_NAME}_slam_zed
                      ${PROJECT_NAME}_frontend 
                      ${PROJECT_NAME}_backend
                      ${ZED_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      "${cpp_typesupport_target}")
# to use custom messages
# rosidl_target_interfaces(${PROJECT_NAME}_slam_zed
# ${PROJECT_NAME} "rosidl_typesupport_cpp")
################################
## TAG SLAM Node
################################
# no longer supported
# add_executable(${PROJECT_NAME}_node src/${PROJECT_NAME}_node.cpp)
# target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES})


################################
## TAG SLAM ZED Node
################################
add_executable(${PROJECT_NAME}_zed_node src/${PROJECT_NAME}_zed_node.cpp)
ament_target_dependencies(${PROJECT_NAME}_zed_node
  cv_bridge
  geometry_msgs
  image_geometry
  image_transport
  nav_msgs
  pluginlib
  rclcpp
  rclcpp_components
  rosidl_default_generators
  sensor_msgs
  std_msgs
  visualization_msgs
)

#############
## Install ##
#############

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
  # ${PROJECT_NAME}_slam
  ${PROJECT_NAME}_slam_zed
  # ${PROJECT_NAME}_node
  ${PROJECT_NAME}_zed_node
  RUNTIME DESTINATION ${bin}
  ARCHIVE DESTINATION ${lib}
  LIBRARY DESTINATION ${lib}
)

ament_package() # EOF

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES geometry_msgs std_msgs
)