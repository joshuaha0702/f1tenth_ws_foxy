cmake_minimum_required(VERSION 3.8)
project(f1tenth_fgm_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANG)
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 의존성 패키지 찾아오기
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(Eigen3 REQUIRED) # Eigen은 별도 include 경로 필요

add_executable(fgm_node src/fgm_disparities.cpp) # 파일명이 맞는지 확인하세요!

# 헤더 경로 설정 (Eigen 포함)
target_include_directories(fgm_node PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

# 라이브러리 연결
ament_target_dependencies(fgm_node
  rclcpp
  sensor_msgs
  nav_msgs
  visualization_msgs
  ackermann_msgs
)

install(TARGETS
  fgm_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()