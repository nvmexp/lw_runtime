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

cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

set(TARGET lwPHYTools_receiver)

set(CMAKE_COMMON_FLAGS "-DALLOW_EXPERIMENTAL_API -DLWDA_ENABLED")
#Activate timers in the persistent kernel
#set(CMAKE_COMMON_FLAGS "${CMAKE_COMMON_FLAGS} -DORDER_KERNEL_TIMERS")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GNU_SOURCE -O3 -g3 -std=c++11  ${CMAKE_COMMON_FLAGS} ${DPDK_RTE_CFLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DORAN_STRICT_COMPLIANT")
#When using lwvp to profile the app
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DPROFILE_LWTX_RANGES")

set(CMAKE_C_FLAGS ${CMAKE_CXX_FLAGS})

set(CMAKE_LWDA_FLAGS "${CMAKE_LWDA_FLAGS} ${CMAKE_COMMON_FLAGS}")

find_package(LWCA 10.1 REQUIRED)

if (LWPHYCONTROLLER)
	set(SOURCES receiver.cpp
		src/lwda_kernels.lw
		$ELW{LWPHY_PATH}/examples/pdsch_tx/pdsch_tx.cpp
		src/dpdk_manager.cpp
		src/lwphy_manager.cpp
		src/utils.cpp
		src/uplink_utils.cpp
		src/downlink_utils.cpp
		src/configuration.cpp 
		src/uplink_cores.cpp
		src/downlink_cores.cpp
		src/downlink_cores.cpp
		src/standalone_core.cpp
		$ELW{LWPHYTOOLS_PATH}/common/src/oran.cpp
		$ELW{LWPHYTOOLS_PATH}/common/src/tv_parser.cpp
		src/controller/controller.cpp
		src/controller/lw_altran_phy.cpp
		src/controller/lw_altran_stack.cpp
	)
else()
	set(SOURCES receiver.cpp
		src/lwda_kernels.lw
		$ELW{LWPHY_PATH}/examples/pdsch_tx/pdsch_tx.cpp
		src/dpdk_manager.cpp
		src/lwphy_manager.cpp
		src/utils.cpp
		src/uplink_utils.cpp
		src/downlink_utils.cpp
		src/configuration.cpp 
		src/uplink_cores.cpp
		src/downlink_cores.cpp
		src/standalone_core.cpp
		$ELW{LWPHYTOOLS_PATH}/common/src/oran.cpp
		$ELW{LWPHYTOOLS_PATH}/common/src/tv_parser.cpp
	)

endif()

add_exelwtable(${TARGET} ${SOURCES})

# ----------------------------------------------------------------------
#  Include directories
target_include_directories(${TARGET} PRIVATE /usr/local/include)
target_include_directories(${TARGET} PRIVATE $ELW{LWPHYTOOLS_PATH}/common/include)
target_include_directories(${TARGET} PRIVATE $ELW{LWPHYTOOLS_PATH}/receiver/include)

target_include_directories(${TARGET} PRIVATE $ELW{LWPHY_PATH}/include)
target_include_directories(${TARGET} PRIVATE ${LWDA_INCLUDE_DIRS})

target_include_directories(${TARGET} PRIVATE $ELW{LWPHY_PATH}/examples/common)
target_include_directories(${TARGET} PRIVATE $ELW{RTE_SDK}/$ELW{RTE_TARGET}/include)
target_include_directories(${TARGET} PRIVATE $ELW{LWPHY_PATH}/examples/pdsch_tx)

if (NO_PRINTS)
    target_compile_definitions(${TARGET} PRIVATE NO_PRINTS=1)
endif()

# lwphycontroller
if (LWPHYCONTROLLER)
	target_compile_definitions(${TARGET} PRIVATE LWPHYCONTROLLER=1)

	target_include_directories(${TARGET} PRIVATE $ELW{LWPHYTOOLS_PATH}/lwphycontroller/altran/include)
	target_include_directories(${TARGET} PRIVATE $ELW{LWPHYTOOLS_PATH}/lwphycontroller/lib/altran_fapi)
	target_include_directories(${TARGET} PRIVATE $ELW{LWPHYTOOLS_PATH}/lwphycontroller/lib/lwIPC/include)
	target_include_directories(${TARGET} PRIVATE $ELW{LWPHYTOOLS_PATH}/lwphycontroller/lib/lwPHY)
	target_include_directories(${TARGET} PRIVATE $ELW{LWPHYTOOLS_PATH}/lwphycontroller/external/g3log/src)
	target_include_directories(${TARGET} PRIVATE $ELW{LWPHYTOOLS_PATH}/lwphycontroller/build/external/g3log/include)
	target_include_directories(${TARGET} PRIVATE $ELW{LWPHYTOOLS_PATH}/lwphycontroller/build/lib)

	add_dependencies(${TARGET} lwipc altran_fapi lwphy yaml)
	target_link_libraries(${TARGET} lwipc altran_fapi lwphy yaml)
endif()

# ----------------------------------------------------------------------
#  Library directories
target_link_directories(${TARGET} PRIVATE $ELW{RTE_SDK}/$ELW{RTE_TARGET}/lib)
target_link_directories(${TARGET} PRIVATE $ELW{LWPHY_PATH}/lib)

# ----------------------------------------------------------------------
#  Libraries 
foreach(lib ${LWPHYTOOLS_REQ_LIBS})
	target_link_libraries(${TARGET} -Wl,--whole-archive ${lib} -Wl,--no-whole-archive)
endforeach()

target_link_libraries(${TARGET} ${MISC_REQ_LIBS})
target_link_libraries(${TARGET} lwca lwdart lwphy_hdf5 lwphy lwphy_channels)

#When using lwvp to profile the app
#target_link_libraries(${TARGET} lwToolsExt)

target_link_libraries(${TARGET} ${HDF5_C_LIBRARIES})

foreach(lib ${DPDK_LW_LIBS})
	target_link_libraries(${TARGET} ${lib})
endforeach()
