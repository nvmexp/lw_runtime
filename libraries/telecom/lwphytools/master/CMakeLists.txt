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
# Using LWCA support inn CMake 3.8+
# https://devblogs.lwpu.com/building-lwca-applications/cmake/

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# ----------------------------------------------------------------------
# Set the version number for this project
set(lwPHYTools_VERSION_MAJOR 0)
set(lwPHYTools_VERSION_MINOR 1)

# ----------------------------------------------------------------------
# Global C++ options
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DLWPHY_NO_AUTO_LOG_INIT")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLWPHY_NO_AUTO_LOG_INIT")

# ----------------------------------------------------------------------
# LWCA Options
#     Generate a string to specify LWCA target architectures
set(LWDA_GENCODE_STRING "" )
#     List of target architectures
set(LWDA_GENCODE_ARCH_LIST "70,72,75" CACHE STRING "List of target LWCA architectures")
message("LWDA_GENCODE_ARCH_LIST is ${LWDA_GENCODE_ARCH_LIST}")

#     Colwert comma-delimited string to CMake list (semicolon-delimited)
string(REPLACE "," ";" LWDA_GENCODE_ARCH_LIST_ ${LWDA_GENCODE_ARCH_LIST})
foreach(LWDA_ARCH_ ${LWDA_GENCODE_ARCH_LIST_})
  string(APPEND LWDA_GENCODE_STRING "-gencode arch=compute_${LWDA_ARCH_},code=sm_${LWDA_ARCH_} ")
  #message("LWDA_ARCH_=${LWDA_ARCH_}")
endforeach(LWDA_ARCH_)

set(CMAKE_LWDA_FLAGS "${LWDA_GENCODE_STRING}")
#set(CMAKE_LWDA_FLAGS " --lwdart static -lineinfo --keep ${LWDA_GENCODE_STRING}" CACHE STRING "LWCA Global Flags")

#     LWCA flags for device debugging with lwca-gdb
# string(APPEND CMAKE_LWDA_FLAGS " --lwdart static -lineinfo -g -G")
set(CMAKE_LWDA_STANDARD 14)
project(lwPHYTools LANGUAGES C CXX LWCA)

message("CMAKE_LWDA_FLAGS is ${CMAKE_LWDA_FLAGS}")

# ----------------------------------------------------------------------
# Add DPDK Libraries


# Read $ELW{RTE_SDK}/$ELW{RTE_TARGET}/.config and set all the config variables
# adopted from https://cmake.org/pipermail/cmake/2007-May/014222.html
file(READ $ELW{RTE_SDK}/$ELW{RTE_TARGET}/.config DPDK_CONFIG_FILE)

STRING(REGEX REPLACE ";" "\\\\;" DPDK_CONFIG_FILE "${DPDK_CONFIG_FILE}")
STRING(REGEX REPLACE "\n" ";" DPDK_CONFIG_FILE "${DPDK_CONFIG_FILE}")

# message("${DPDK_CONFIG_FILE}")

foreach(line ${DPDK_CONFIG_FILE})
	string(REGEX MATCH "^[#| #]" output ${line})
	if ((NOT (NOT ${output})))
		STRING(REGEX REPLACE "=" ";" assignment ${line})
		list(GET assignment 0 rhs)
		list(GET assignment 1 lhs)
		set(${rhs} ${lhs})
	endif()
endforeach()


list(APPEND DPDK_REQ_LIBS
	rte_flow_classify
	rte_pipeline
	rte_table
	rte_port
	rte_pdump
	rte_distributor
	rte_ip_frag
	rte_meter
	rte_lpm
	rte_acl
	rte_jobstats
	rte_metrics
	rte_bitratestats
	rte_latencystats
	rte_power
	rte_efd
	rte_bpf
	rte_ipsec
	rte_cfgfile
	rte_gro
	rte_gso
	rte_hash
	rte_member
	rte_vhost
	rte_kvargs
	rte_mbuf
	rte_net
	rte_ethdev
	rte_bbdev
	rte_cryptodev
	rte_selwrity
	rte_compressdev
	rte_eventdev
	rte_rawdev
	rte_timer
	rte_mempool
	rte_stack
	rte_mempool_ring
	rte_ring
	rte_pci
	rte_eal
	rte_cmdline
	rte_reorder
	rte_sched
	rte_rlw
	rte_kni
	rte_bus_pci
	)

list(APPEND MLX5_REQ_LIBS
	rte_pmd_mlx5
	mnl
	ibverbs
	mlx5
	)

list(APPEND LWPHYTOOLS_REQ_LIBS
	${DPDK_REQ_LIBS}
	${MLX5_REQ_LIBS}
	)

list(APPEND DPDK_LW_LIBS
	rte_lw
	)

set(MISC_REQ_LIBS
	jsoncpp 
	pthread 
	dl 
	rt 
	m 
	numa 
	mlx5 
	mnl 
	ibverbs
	)

# ----------------------------------------------------------------------
# DPDK C FLAGS

# Extract architecture from RTE_TARGET
string(REPLACE "-" ";" RTE_TARGET_LIST $ELW{RTE_TARGET})
list(GET RTE_TARGET_LIST 1 MACHINE_ARCH)

string(FIND "${MACHINE_ARCH}" "armv8" FOUND)
if (NOT(${FOUND} STREQUAL "-1"))
	set(MACHINE_ARCH "armv8-a+crc+crypto") 
	set(DPDK_RTE_CFLAGS "-march=${MACHINE_ARCH}")
else()
	#CheetAh make command does not include -m64 flag
	set(DPDK_RTE_CFLAGS "-m64 -march=${MACHINE_ARCH}")
endif()

# Colwerting rte.cpuflags.mk to cmake
exelwte_process(COMMAND bash "-c" "echo | g++ -dM -E - -march=${MACHINE_ARCH}" OUTPUT_VARIABLE AUTO_CPUFLAGS)

string(FIND "${AUTO_CPUFLAGS}" "__SSE__" FOUND)
if (NOT(${FOUND} STREQUAL "-1"))
	set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_SSE") 
endif()

string(FIND "${AUTO_CPUFLAGS}" "__SSE2__" FOUND)
if (NOT(${FOUND} STREQUAL "-1"))
	set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_SSE2") 
endif()

string(FIND "${AUTO_CPUFLAGS}" "__SSE3__" FOUND)
if (NOT(${FOUND} STREQUAL "-1"))
	set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_SSE3") 
endif()

string(FIND "${AUTO_CPUFLAGS}" "__SSSE3__" FOUND)
if (NOT(${FOUND} STREQUAL "-1"))
	set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_SSSE3") 
endif()

string(FIND "${AUTO_CPUFLAGS}" "__SSE4_1__" FOUND)
if (NOT(${FOUND} STREQUAL "-1"))
	set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_SSE4_1") 
endif()

string(FIND "${AUTO_CPUFLAGS}" "__SSE4_2__" FOUND)
if (NOT(${FOUND} STREQUAL "-1"))
	set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_SSE4_2") 
endif()

string(FIND "${AUTO_CPUFLAGS}" "__AES__" FOUND)
if (NOT(${FOUND} STREQUAL "-1"))
	set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_AES") 
endif()

string(FIND "${AUTO_CPUFLAGS}" "__PXLMUL__" FOUND)
if (NOT(${FOUND} STREQUAL "-1"))
	set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_PCLMULQDQ") 
endif()

string(FIND "${AUTO_CPUFLAGS}" "__AVX__" FOUND)
if (NOT(${FOUND} STREQUAL "-1"))
	if(${CONFIG_RTE_ENABLE_AVX} STREQUAL "y")
		set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_AVX") 
	endif()
endif()

string(FIND "${AUTO_CPUFLAGS}" "__RDRND__" FOUND)
if (NOT(${FOUND} STREQUAL "-1"))
	set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_RDRAND") 
endif()

string(FIND "${AUTO_CPUFLAGS}" "__FSGSBASE__" FOUND)
if (NOT(${FOUND} STREQUAL "-1"))
	set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_FSGSBASE") 
endif()

string(FIND "${AUTO_CPUFLAGS}" "__F16C__" FOUND)
if (NOT(${FOUND} STREQUAL "-1"))
	set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_F16C") 
endif()

string(FIND "${AUTO_CPUFLAGS}" "__AVX2__" FOUND)
if (NOT(${FOUND} STREQUAL "-1"))
	if(${CONFIG_RTE_ENABLE_AVX} STREQUAL "y")
		set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_AVX2") 
	endif()
endif()

# CheetAh flags for armv8
string(FIND "${AUTO_CPUFLAGS}" "__ARM_NEON" FOUND1)
string(FIND "${AUTO_CPUFLAGS}" "__aarch64__" FOUND2)
if ((NOT(${FOUND1} STREQUAL "-1")) OR (NOT(${FOUND2} STREQUAL "-1")))
	set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_CRC32") 
endif()

string(FIND "${AUTO_CPUFLAGS}" "__ARM_FEATURE_CRC32" FOUND)
if (NOT(${FOUND} STREQUAL "-1"))
	set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_CRC32") 
endif()

string(FIND "${AUTO_CPUFLAGS}" "__ARM_FEATURE_CRYPTO" FOUND)
if (NOT(${FOUND} STREQUAL "-1"))
	set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_AES") 
	set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_PMULL")
	set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_SHA1")
	set(DPDK_RTE_CFLAGS "${DPDK_RTE_CFLAGS} -DRTE_MACHINE_CPUFLAG_SHA2")
endif()

# ----------------------------------------------------------------------
# Additional packages
set(HDF5_USE_STATIC_LIBRARIES ON)
find_package(HDF5 1.10 REQUIRED COMPONENTS C)

find_package(jsoncpp REQUIRED)

# ----------------------------------------------------------------------
# Subdirectories
if (LWPHYCONTROLLER)
	message(STATUS "Building with lwPHY Controller")
	add_subdirectory($ELW{LWPHYTOOLS_PATH}/lwphycontroller)
else()
	message(STATUS "Building without lwPHY Controller")
endif()

add_subdirectory(receiver)
add_subdirectory(generator)

# ----------------------------------------------------------------------
# Installation
install(PROGRAMS scripts/receiver.sh DESTINATION . )
install(PROGRAMS scripts/generator.sh DESTINATION . )
install(PROGRAMS scripts/setup_system.sh DESTINATION . )
install(PROGRAMS ${CMAKE_LWRRENT_BINARY_DIR}/generator/lwPHYTools_generator 
		DESTINATION . 
		)
install(PROGRAMS ${CMAKE_LWRRENT_BINARY_DIR}/receiver/lwPHYTools_receiver 
		DESTINATION . 
		)
