# Copyright (c) 2017-2020, LWPU CORPORATION.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted
# provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice, this list of
#       conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of the LWPU CORPORATION nor the names of its contributors may be used
#       to endorse or promote products derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LWPU CORPORATION BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Minimum CMake version required.
# Using LWCA support in CMake 3.8+
# Using SHELL escape support for argument de-duplication added in CMake 3.12+
# https://devblogs.lwpu.com/building-lwca-applications/cmake/
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# ----------------------------------------------------------------------
# Build options
option(USE_MATLAB "Build MEX libraries to allow use from MATLAB" OFF)
option(BUILD_DOCS "Generate Doxygen documentation" OFF)
option(BUILD_ALL_LDPC_KERNELS "Build all LDPC decoder kernels - even those that would not be chosen for a pipeline" OFF)
option(BUILD_ALL_LDPC_LIFTING_SIZES "Build LDPC decoder kernels to support all lifting sizes" OFF)
option(LWPHY_BUILD_PTXAS_VERBOSE "Enable verbose mode for the PTX assembler to show local memory usage" OFF)

# ----------------------------------------------------------------------
# Set the version number for this project
set(lwPHY_VERSION_MAJOR 0)
set(lwPHY_VERSION_MINOR 7)

# ----------------------------------------------------------------------
# Enable testing support
enable_testing()

# ----------------------------------------------------------------------
# Global C++ options
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------------------------------------------------
# lwPHY LWCA Options
#     Generate a string to specify LWCA target architectures
set(LWPHY_GENCODE_ARCH_LIST "70,72,75" CACHE STRING "List of target LWCA architectures for lwPHY")
message("LWPHY_GENCODE_ARCH_LIST is ${LWPHY_GENCODE_ARCH_LIST}")

# Colwert comma-delimited string to CMake list (semicolon-delimited).
# When using target_compile_options, CMake will perform "de-duplication"
# merging different '-gencode' options together, unless we use the
# SHELL: prefix to escape that feature. See:
# https://cmake.org/cmake/help/latest/command/target_compile_options.html
set(SPACE " ")
string(REPLACE "," ";" LWPHY_GENCODE_ARCH_LIST_ ${LWPHY_GENCODE_ARCH_LIST})
foreach(LWDA_ARCH_ ${LWPHY_GENCODE_ARCH_LIST_})
  list(APPEND LWPHY_GENCODE_STRING "SHELL:-gencode arch=compute_${LWDA_ARCH_},code=sm_${LWDA_ARCH_}${SPACE}")
  #message("LWDA_ARCH_=${LWDA_ARCH_}")
endforeach(LWDA_ARCH_)

#string(REPLACE ";" " " LWPHY_GENCODE_STRING "${LWPHY_GENCODE_STRING}")
#message("LWPHY_GENCODE_STRING is ${LWPHY_GENCODE_STRING}")
#message(${LWPHY_GENCODE_STRING})

# Old way of setting global CMAKE_LWDA_FLAGS variable...
#set(CMAKE_LWDA_FLAGS " --lwdart static -lineinfo ${LWDA_GENCODE_STRING}" CACHE STRING "LWCA Global Flags")
#set(CMAKE_LWDA_FLAGS " --lwdart static -lineinfo --keep ${LWDA_GENCODE_STRING}" CACHE STRING "LWCA Global Flags")
# LWCA flags for device debugging with lwca-gdb
# string(APPEND CMAKE_LWDA_FLAGS " --lwdart static -lineinfo -g -G")


set(CMAKE_LWDA_STANDARD 14)
project(lwPHY LANGUAGES C CXX LWCA)

# ----------------------------------------------------------------------
# Additional packages
set(THREADS_PREFER_PTHREAD_FLAG OFF)
find_package(Threads REQUIRED)
set(HDF5_USE_STATIC_LIBRARIES ON)
find_package(HDF5 1.10 REQUIRED COMPONENTS C)
if (USE_MATLAB)
    find_package(Matlab REQUIRED)
endif (USE_MATLAB)

# ----------------------------------------------------------------------
# Generation of documentation
if (BUILD_DOCS)
    find_package(Doxygen)
    if (DOXYGEN_FOUND)
        set(DOXYGEN_IN  ${CMAKE_LWRRENT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_LWRRENT_BINARY_DIR}/Doxyfile)
        message("Configuring ${DOXYGEN_OUT}")
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        add_lwstom_target(docs_doxygen ALL
          COMMAND ${DOXYGEN_EXELWTABLE} ${DOXYGEN_OUT}
          WORKING_DIRECTORY ${CMAKE_LWRRENT_BINARY_DIR}
          COMMENT "Generating API documentation (doxygen)"
          VERBATIM)
    else (DOXYGEN_FOUND)
        message("ERROR: Documentation generation requested, but Doxygen package not found")
    endif (DOXYGEN_FOUND)
endif (BUILD_DOCS)

project(lwPHY)
# ----------------------------------------------------------------------
# Subdirectories
add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(test)

if (USE_MATLAB)
    add_subdirectory(util/matlab/mex)
endif (USE_MATLAB)

# ----------------------------------------------------------------------
# Installation
install(DIRECTORY examples/common DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/ch_est DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/channel_eq DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/pusch_rx_multi_pipe DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/error_correction DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/ldpc_encode DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/pdsch_tx DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/pucch_receiver DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(DIRECTORY examples/polar_encoder DESTINATION examples FILES_MATCHING PATTERN "*" PATTERN "CMakeLists.*" EXCLUDE)
install(FILES CMakeLists.release.txt DESTINATION . RENAME CMakeLists.txt)
if (BUILD_DOCS)
    install(DIRECTORY ${CMAKE_LWRRENT_BINARY_DIR}/docs DESTINATION . FILES_MATCHING PATTERN "*")
endif (BUILD_DOCS)

