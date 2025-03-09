cmake_minimum_required(VERSION 3.5)
project(tagslam_ros)

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
find_package(cmake_modules REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(image_geometry REQUIRED)
find_package(image_transport REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(rcl_cpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package( REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(ament_cmake REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  pluginlib
  roscpp
  sensor_msgs
  std_msgs
)

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(image_geometry REQUIRED)
find_package(image_transport REQUIRED)
find_package(nodelet REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)  
find_package(std_msgs REQUIRED)  

find_package(rosidl_default_generators REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(apriltag REQUIRED)
find_package(TBB REQUIRED)
# find_package(Boost 1.58 REQUIRED COMPONENTS system serialization filesystem thread date_time regex timer chrono)
find_package(GTSAM REQUIRED)
find_package(GTSAM_UNSTABLE REQUIRED)

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


rosidl_generate_interfaces(
  DEPENDENCIES
  geometry_msgs
  sensor_msgs
  std_msgs
)

# Extract the include directories and libraries from apriltag::apriltag as catkin_package() does not support modern cmake.
get_target_property(apriltag_INCLUDE_DIRS apriltag::apriltag INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(apriltag_LIBRARIES apriltag::apriltag INTERFACE_LINK_LIBRARIES)

# catkin_package
ament_export_dependencies(
  cv_bridge REQUIRED
  geometry_msgs REQUIRED
  image_transport REQUIRED
  nodelet REQUIRED
  pluginlib REQUIRED
  roscpp REQUIRED
  sensor_msgs REQUIRED
  std_msgs REQUIRED
)
ament_package() # double check

###########
## Build ##
###########

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  # ${Boost_INCLUDE_DIR}
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
            ${catkin_LIBRARIES} 
            ${OpenCV_LIBRARIES}
            ${CUDA_LIBRARIES}
            apriltag::apriltag
            nvapriltags)

else()
  add_library(${PROJECT_NAME}_frontend 
            src/frontend/tag_detector.cpp
            src/frontend/tag_detector_cpu.cpp)
  target_link_libraries(${PROJECT_NAME}_frontend
            ${catkin_LIBRARIES} 
            ${OpenCV_LIBRARIES}
            apriltag::apriltag)
endif()

# TODO: check generate messages, rosidl_generate_interfaces
add_dependencies(${PROJECT_NAME}_frontend ${PROJECT_NAME}_generate_messages_cpp)

################################
## Add library for the back-end
################################
add_library(${PROJECT_NAME}_backend 
            src/backend/backend.cpp
            src/backend/isam2_backend.cpp
            src/backend/fixed_lag_backend.cpp)
add_dependencies(${PROJECT_NAME}_backend ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_backend 
                      ${catkin_LIBRARIES}
                      gtsam
                      gtsam_unstable
                      tbb  
                      # ${Boost_LIBRARIES} 
                      )

# ################################
# ## Add library for the SLAM
# ################################

add_library(${PROJECT_NAME}_slam src/tag_slam.cpp)
add_dependencies(${PROJECT_NAME}_slam ${PROJECT_NAME}_generate_messages_cpp)
target_link_libraries(${PROJECT_NAME}_slam
                      ${PROJECT_NAME}_frontend 
                      ${PROJECT_NAME}_backend
                      ${catkin_LIBRARIES} )
                      

################################
## ZED Node
################################
add_library(${PROJECT_NAME}_slam_zed src/tag_slam_zed.cpp)
add_dependencies(${PROJECT_NAME}_slam_zed ${PROJECT_NAME}_generate_messages_cpp)
target_include_directories(${PROJECT_NAME}_slam_zed PUBLIC
                      ${ZED_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}_slam_zed
                      ${PROJECT_NAME}_frontend 
                      ${PROJECT_NAME}_backend
                      ${ZED_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${catkin_LIBRARIES} )

################################
## TAG SLAM Node
################################
add_executable(${PROJECT_NAME}_node src/${PROJECT_NAME}_node.cpp)
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES})


################################
## TAG SLAM ZED Node
################################
add_executable(${PROJECT_NAME}_zed_node src/${PROJECT_NAME}_zed_node.cpp)
target_link_libraries(${PROJECT_NAME}_zed_node ${catkin_LIBRARIES})

#############
## Install ##
#############

install(DIRECTORY config launch
  DESTINATION ${share}
)
install(FILES nodelet_plugins.xml
  DESTINATION ${share}
)

install(TARGETS
  ${PROJECT_NAME}_slam
  ${PROJECT_NAME}_slam_zed
  ${PROJECT_NAME}_node
  ${PROJECT_NAME}_zed_node
  RUNTIME DESTINATION ${bin}
  ARCHIVE DESTINATION ${lib}
  LIBRARY DESTINATION ${lib}
)

