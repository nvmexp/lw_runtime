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

add_exelwtable(test_ldpc_internal test_ldpc_internal.lw)
target_include_directories(test_ldpc_internal PRIVATE ${PROJECT_SOURCE_DIR}/examples/common)
target_link_libraries(test_ldpc_internal lwphy)
target_link_libraries(test_ldpc_internal gtest gtest_main)
target_compile_options(test_ldpc_internal PRIVATE $<$<COMPILE_LANGUAGE:LWCA>:
                       ${LWPHY_GENCODE_STRING}
                       -lineinfo
                       >)
#                        --keep
#                       --lwdart static
#set_source_files_properties(error_correction/ldpc2_split_dynamic_96KB_BG1_Z384.lw PROPERTIES COMPILE_FLAGS --keep)

#add_test(NAME test_ldpc_internal COMMAND test_ldpc_internal WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test/error_correction)
add_test(NAME test_ldpc_internal COMMAND test_ldpc_internal)
