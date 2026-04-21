cmake_minimum_required(VERSION 3.12)
project(octomap_bridge)

find_package(ament_cmake REQUIRED)
find_package(octomap REQUIRED)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(octomap_bridge src/bridge.cpp)

# Link OctoMap library
target_link_libraries(octomap_bridge PRIVATE ${OCTOMAP_LIBRARIES})
target_include_directories(octomap_bridge PRIVATE ${OCTOMAP_INCLUDE_DIRS})

# Install the compiled Python module to the ROS 2 Jazzy Python directory
install(TARGETS octomap_bridge
  DESTINATION "lib/python3.12/site-packages"
)

ament_package()