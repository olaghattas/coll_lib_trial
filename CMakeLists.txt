cmake_minimum_required(VERSION 3.8)
project(coll_lib_trial)
enable_language(CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(collision_lib REQUIRED)
find_package(CUDA REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(ros2-tiny-cuda-nn REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

set(dependencies
        rclcpp
        ament_index_cpp
        collision_lib
        visualization_msgs
        sensor_msgs
)

# Add executable
add_executable(try_pkg_coll_lib src/try_pkg_coll_lib.cpp)
ament_target_dependencies(try_pkg_coll_lib ${dependencies})
#target_link_libraries(particle_filter  collision_lib)

ament_target_dependencies(try_pkg_coll_lib ros2-tiny-cuda-nn)
target_link_libraries(try_pkg_coll_lib ${ros2-tiny-cuda-nn_LIBRARIES})

target_include_directories(try_pkg_coll_lib PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${ros2-tiny-cuda-nn_INCLUDE_DIR}
)

# Add executable
add_executable(try_pkg_coll_lib_inf src/try_pkg_coll_lib_inf.cpp)
ament_target_dependencies(try_pkg_coll_lib_inf ${dependencies})
#target_link_libraries(particle_filter  collision_lib)

ament_target_dependencies(try_pkg_coll_lib_inf ros2-tiny-cuda-nn)
target_link_libraries(try_pkg_coll_lib_inf ${ros2-tiny-cuda-nn_LIBRARIES})

target_include_directories(try_pkg_coll_lib_inf PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${ros2-tiny-cuda-nn_INCLUDE_DIR}
)

# Add executable
add_executable(try_pkg_coll_lib_inf_only src/try_pkg_coll_lib_inf_only.cpp)
ament_target_dependencies(try_pkg_coll_lib_inf_only ${dependencies})
#target_link_libraries(particle_filter  collision_lib)

ament_target_dependencies(try_pkg_coll_lib_inf_only ros2-tiny-cuda-nn)
target_link_libraries(try_pkg_coll_lib_inf_only ${ros2-tiny-cuda-nn_LIBRARIES})

target_include_directories(try_pkg_coll_lib_inf_only PUBLIC
        ${CUDA_INCLUDE_DIRS}
        ${ros2-tiny-cuda-nn_INCLUDE_DIR}
)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
