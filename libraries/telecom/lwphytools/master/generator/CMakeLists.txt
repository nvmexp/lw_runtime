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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GNU_SOURCE -O3 -g3 -std=c++11 \
					 -Werror=return-type -DALLOW_EXPERIMENTAL_API \
					 ${DPDK_RTE_CFLAGS}")
set(CMAKE_C_FLAGS ${CMAKE_CXX_FLAGS})

set(SOURCES generator.cpp
			generator_impl.cpp
			$ELW{LWPHYTOOLS_PATH}/common/src/oran.cpp
			$ELW{LWPHYTOOLS_PATH}/common/src/tv_parser.cpp
			)

set(TARGET lwPHYTools_generator)

add_exelwtable(${TARGET} ${SOURCES})


# ----------------------------------------------------------------------
#  Include directories
target_include_directories(${TARGET} PRIVATE /usr/local/include)
target_include_directories(${TARGET} PRIVATE $ELW{LWPHYTOOLS_PATH}/common/include)
target_include_directories(${TARGET} PRIVATE $ELW{LWPHY_PATH}/examples/common)
target_include_directories(${TARGET} PRIVATE $ELW{RTE_SDK}/$ELW{RTE_TARGET}/include)

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
target_link_libraries(${TARGET} lwphy_hdf5)
target_link_libraries(${TARGET} ${HDF5_C_LIBRARIES})
