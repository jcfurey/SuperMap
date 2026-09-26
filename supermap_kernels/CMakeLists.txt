cmake_minimum_required(VERSION 3.18)
project(supermap_kernels LANGUAGES CXX)

# Two builds from one file: colcon (ament_cmake, installs into the workspace's
# Python path) and pip (scikit-build-core sets SKBUILD, installs into the wheel).

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
# OpenMP is optional: without it every kernel runs on one thread.
find_package(OpenMP COMPONENTS CXX)

# The Python package supermap_kernels re-exports this compiled module.
pybind11_add_module(_native src/native.cpp)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_native PRIVATE OpenMP::OpenMP_CXX)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # No fused multiply-add contraction: arithmetic rounds step by step, as
  # NumPy's separate ufuncs do, on every target (FMA is baseline on aarch64).
  target_compile_options(_native PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

if(SKBUILD)
  install(TARGETS _native LIBRARY DESTINATION ${PROJECT_NAME})
else()
  find_package(ament_cmake REQUIRED)
  find_package(ament_cmake_python REQUIRED)
  ament_python_install_package(${PROJECT_NAME})
  install(TARGETS _native LIBRARY DESTINATION "${PYTHON_INSTALL_DIR}/${PROJECT_NAME}")
  ament_package()
endif()
