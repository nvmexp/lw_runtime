cmake_minimum_required(VERSION 3.4.0)
if(POLICY CMP0046)
  cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
  cmake_policy(SET CMP0054 NEW)
endif()

if(USE_MPI)
  set(CMAKE_C_COMPILER "mpicc")
  set(CMAKE_CXX_COMPILER "mpic++")
endif()

# ---[ Caffe project
project(Caffe C CXX)

# ---[ Caffe version
set(CAFFE_TARGET_VERSION "0.17.3")
set(CAFFE_TARGET_SOVERSION "0.17")
add_definitions(-DCAFFE_VERSION=${CAFFE_TARGET_VERSION})

# Skip `typedef __half half;`
add_definitions(-DLWDA_NO_HALF=1)

# ---[ Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

include(ExternalProject)

include(cmake/Utils.cmake)
include(cmake/Targets.cmake)
include(cmake/Misc.cmake)
include(cmake/Summary.cmake)
include(cmake/ConfigGen.cmake)

# ---[ Options
caffe_option(USE_LWDNN "Build Caffe with lwDNN library support" ON)

# USE_NCCL: Build Caffe with LWCL Library support
# Regular ON/OFF option doesn't work here because we need to recognize 3 states:
# 1. User didn't set USE_NCCL option =>
#   1.1 We *quietly* try to find it and use if found; do nothing otherwise.
# 2. User explicitly set USE_NCCL=ON option =>
#   1.1 We try to find it with *required* option, thus CMake fails if not found.
# 3. User explicitly set USE_NCCL=OFF option => we do nothing.
set(USE_NCCL "NONE" CACHE STRING "Link Caffe with LWCL Library for Multi-GPU support")
if(USE_NCCL STREQUAL "NONE")
  set(USE_NCCL_SET OFF)
else()
  set(USE_NCCL_SET ON)
endif()

caffe_option(BUILD_SHARED_LIBS "Build shared libraries" ON)
caffe_option(BUILD_python "Build Python wrapper" ON)
set(python_version "2" CACHE STRING "Specify which Python version to use")
caffe_option(BUILD_matlab "Build Matlab wrapper" OFF IF UNIX OR APPLE)
caffe_option(BUILD_docs   "Build documentation" ON IF UNIX OR APPLE)
caffe_option(BUILD_python_layer "Build the Caffe Python layer" ON)
caffe_option(USE_LEVELDB "Build with levelDB" ON)
caffe_option(USE_LMDB "Build with lmdb" ON)
caffe_option(TEST_FP16 "Build Caffe Tests with 16 bit mode included" OFF)
caffe_option(NO_LWML "Build Caffe Tests without LWML (i.e. no CPU affinity)" OFF)
caffe_option(USE_TRT "Build with TensorRT support" OFF)
caffe_option(USE_MPI "Build with MPI support" OFF)

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^aarch")
  # CheetAh
  set(NO_LWML ON)
endif()

# ---[ Dependencies
include(cmake/Dependencies.cmake)

# ---[ Flags
if(UNIX OR APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall -std=c++11")
endif()

caffe_set_caffe_link()

if(USE_libstdcpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
  message("-- Warning: forcing libstdc++ (controlled by USE_libstdcpp option in cmake)")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG")

add_definitions(-DGTEST_USE_OWN_TR1_TUPLE)

# ---[ Warnings
caffe_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized)

# ---[ Config generation
configure_file(cmake/Templates/caffe_config.h.in "${PROJECT_BINARY_DIR}/caffe_config.h")

# ---[ Includes
set(Caffe_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(THIRDPARTY_DIR ${PROJECT_SOURCE_DIR}/3rdparty)
include_directories(${Caffe_INCLUDE_DIR} ${PROJECT_BINARY_DIR} ${THIRDPARTY_DIR})
include_directories(BEFORE src) # This is needed for gtest.

# ---[ Subdirectories
add_subdirectory(src/gtest)
add_subdirectory(src/caffe)
add_subdirectory(tools)
add_subdirectory(examples)
add_subdirectory(python)
add_subdirectory(matlab)
add_subdirectory(docs)

# ---[ Linter target
add_lwstom_target(lint COMMAND ${CMAKE_COMMAND} -P ${PROJECT_SOURCE_DIR}/cmake/lint.cmake)

# ---[ pytest target
if(BUILD_python)
  add_lwstom_target(pytest COMMAND python${python_version} -m unittest discover -s caffe/test WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/python )
  add_dependencies(pytest pycaffe)
endif()

# ---[ Configuration summary
caffe_print_configuration_summary()

# ---[ Export configs generation
caffe_generate_export_configs()
