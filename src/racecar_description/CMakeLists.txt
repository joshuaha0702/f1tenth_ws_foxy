cmake_minimum_required(VERSION 3.8)
project(racecar_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_COMPILER_IS_CLANG)
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# 데이터 폴더들을 share/패키지명 경로로 복사
install(DIRECTORY
  launch
  urdf
  meshes
  models
  worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()