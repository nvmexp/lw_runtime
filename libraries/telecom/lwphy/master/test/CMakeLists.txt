# Copyright (c) 2017-2018, LWPU CORPORATION.  All rights reserved.
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
include_directories("${PROJECT_SOURCE_DIR}/src/lwphy/crc")
include_directories("${PROJECT_SOURCE_DIR}/src/lwphy/descrambling")
include_directories("${PROJECT_SOURCE_DIR}/src/lwphy/rate_matching")
include_directories("${PROJECT_SOURCE_DIR}/src/lwphy/dl_rate_matching")
include_directories("${PROJECT_SOURCE_DIR}/src/lwphy/modulation_mapper")
include_directories("${PROJECT_BINARY_DIR}/src/lwphy/LUTS")
include_directories("${PROJECT_SOURCE_DIR}/src/lwphy_hdf5")
include_directories("${PROJECT_SOURCE_DIR}/src/lwphy/error_correction")
include_directories("${PROJECT_SOURCE_DIR}/examples/common")
include_directories("${PROJECT_SOURCE_DIR}/examples/pusch_rx_multi_pipe")
add_subdirectory(crc)
add_subdirectory(descrambling)
add_subdirectory(rate_matching)
add_subdirectory(dl_rate_matching)
add_subdirectory(hdf5)
add_subdirectory(modulation_mapper)
add_subdirectory(error_correction)
