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

# Generation of compile time LUTs for CRC

set (LUTS_DIR "${CMAKE_LWRRENT_BINARY_DIR}/LUTS")

file(MAKE_DIRECTORY ${LUTS_DIR})

add_exelwtable(genLUT crc/gen_crc_LUTs.cpp descrambling/descrambling.hpp)

target_include_directories(genLUT PUBLIC ${CMAKE_LWDA_TOOLKIT_INCLUDE_DIRECTORIES} $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}>)

add_lwstom_command( OUTPUT
            ${LUTS_DIR}/G_CRC_24_A_COALESCED_P_LUT.h
            ${LUTS_DIR}/G_CRC_24_B_COALESCED_P_LUT.h
            ${LUTS_DIR}/G_CRC_16_COALESCED_P_LUT.h
            ${LUTS_DIR}/GOLD_1_SEQ_LUT.h
            ${LUTS_DIR}/GOLD_2_COALESCED_P_LUT.h
            DEPENDS crc/crc.hpp crc/gen_crc_LUTs.cpp descrambling/descrambling.hpp
            COMMAND genLUT ${LUTS_DIR}
            COMMENT "Generating LUTs for CRC and Descrambling")

add_lwstom_target(CRCLUTGEN DEPENDS
            ${LUTS_DIR}/G_CRC_24_A_COALESCED_P_LUT.h
            ${LUTS_DIR}/G_CRC_24_B_COALESCED_P_LUT.h
            ${LUTS_DIR}/G_CRC_16_COALESCED_P_LUT.h
            ${LUTS_DIR}/GOLD_1_SEQ_LUT.h
            ${LUTS_DIR}/GOLD_2_COALESCED_P_LUT.h
            )

# Generation of compile time LUTs for PUCCH receiver Format 1

add_exelwtable(genPucchLUT pucch_receiver/gen_pucch_receiver_LUTs.cpp)
target_include_directories(genPucchLUT PUBLIC ${CMAKE_LWDA_TOOLKIT_INCLUDE_DIRECTORIES} $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}>)

add_lwstom_command( OUTPUT
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_TOCC_VALUES_LUT.h
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_TIME_SHIFT_SEQ_VALUES_LUT.h
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_PAPR_SEQ_VALUES_LUT.h
            DEPENDS pucch_receiver/gen_pucch_receiver_LUTs.cpp
            COMMAND genPucchLUT ${LUTS_DIR}
            COMMENT "Generating LUTs for PUCCH receiver Format 1")

add_lwstom_target(PUCCHLUTGEN DEPENDS
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_TOCC_VALUES_LUT.h
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_TIME_SHIFT_SEQ_VALUES_LUT.h
            ${LUTS_DIR}/PUCCH_RECEIVER_F1_PAPR_SEQ_VALUES_LUT.h
            )

if(BUILD_ALL_LDPC_KERNELS)
    add_definitions(-DLWPHY_LDPC_INCLUDE_ALL_ALGOS=1)
endif()
  
if(BUILD_ALL_LDPC_LIFTING_SIZES)
    add_definitions(-DLWPHY_LDPC_INCLUDE_ALL_LIFTING=1)
endif()
  

if(LWPHY_BUILD_PTXAS_VERBOSE)
    #set(LWPHY_PTXAS_OPTIONS "--ptxas-options=-v")
    set(LWPHY_PTXAS_OPTIONS "--resource-usage")
endif()

add_library(lwphy SHARED lwphy.h
                         lwphy.cpp
                         lwphy_internal.h
                         lwphy_context.hpp
                         lwphy_context.cpp
                         tensor_desc.cpp
                         device.hpp
                         device.cpp
                         type_colwert.hpp
                         colwert_tensor.lwh
                         colwert_tensor.lw
                         lwphy_kernel_util.lwh
                         channel_est/channel_est.lw
                         ch_est/ch_est.lw
                         channel_eq/channel_eq.lw
                         error_correction/nrLDPC.lwh
                         error_correction/nrLDPC_flat.lwh
                         error_correction/ldpc.lwh
                         error_correction/ldpc.hpp
                         error_correction/ldpc_load_store.lwh
                         error_correction/ldpc.cpp
                         error_correction/ldpc_ms_small_flooding.hpp
                         error_correction/ldpc_ms_small_flooding.lw
                         error_correction/ldpc_ms_mk_flooding.hpp
                         error_correction/ldpc_ms_mk_flooding.lw
                         error_correction/ldpc_ms_mka_flooding.hpp
                         error_correction/ldpc_ms_mka_flooding.lw
                         error_correction/ldpc_ms_cta_layered.hpp
                         error_correction/ldpc_ms_cta_layered.lw
                         rate_matching/rate_matching.lw
                         dl_rate_matching/dl_rate_matching.lwh
                         dl_rate_matching/dl_rate_matching.lw
                         error_correction/ldpc_ms_cta_flooding.hpp
                         error_correction/ldpc_ms_cta_flooding.lw
                         error_correction/ldpc_ms_cta_simd_flooding.hpp
                         error_correction/ldpc_ms_cta_simd_flooding.lw
                         error_correction/ldpc_ms_cta_shmem_flooding.hpp
                         error_correction/ldpc_ms_cta_shmem_flooding.lw
                         error_correction/ldpc_ms_cta_shmem_layered.hpp
                         error_correction/ldpc_ms_cta_shmem_layered.lw
                         error_correction/ldpc_ms_mka_flooding_flat.hpp
                         error_correction/ldpc_ms_mka_flooding_flat.lw
                         error_correction/ldpc_fast_layered.lw
                         error_correction/ldpc_fast.h
                         error_correction/ldpc_fast_layered.h
                         error_correction/ldpc_ms_cta_shmem_layered_unroll.hpp
                         error_correction/ldpc_ms_cta_shmem_layered_unroll.lw
                         error_correction/ldpc2.hpp
                         error_correction/ldpc2_c2v_cache_register.lwh
                         error_correction/ldpc2_c2v_cache_global.lwh
                         error_correction/ldpc2_c2v_cache_shared.lwh
                         error_correction/ldpc2_c2v_cache_split.lwh
                         error_correction/ldpc2_schedule_fixed.lwh
                         error_correction/ldpc2_schedule_dynamic.lwh
                         error_correction/ldpc2_schedule_cluster.lwh
                         error_correction/ldpc2_app_address.lwh
                         error_correction/ldpc2_app_address_fp.lwh
                         error_correction/ldpc2_reg.hpp
                         error_correction/ldpc2_reg.cpp
                         error_correction/ldpc2_reg.lwh
                         error_correction/ldpc2_reg_index_BG1_Z36.lw
                         error_correction/ldpc2_reg_index_BG1_Z40.lw
                         error_correction/ldpc2_reg_index_BG1_Z44.lw
                         error_correction/ldpc2_reg_index_BG1_Z48.lw
                         error_correction/ldpc2_reg_index_BG1_Z52.lw
                         error_correction/ldpc2_reg_index_BG1_Z56.lw
                         error_correction/ldpc2_reg_index_BG1_Z60.lw
                         error_correction/ldpc2_reg_index_BG1_Z64.lw
                         error_correction/ldpc2_reg_index_BG1_Z72.lw
                         error_correction/ldpc2_reg_index_BG1_Z80.lw
                         error_correction/ldpc2_reg_index_BG1_Z88.lw
                         error_correction/ldpc2_reg_index_BG1_Z96.lw
                         error_correction/ldpc2_reg_index_BG1_Z104.lw
                         error_correction/ldpc2_reg_index_BG1_Z112.lw
                         error_correction/ldpc2_reg_index_BG1_Z120.lw
                         error_correction/ldpc2_reg_index_BG1_Z128.lw
                         error_correction/ldpc2_reg_index_BG1_Z144.lw
                         error_correction/ldpc2_reg_index_BG1_Z160.lw
                         error_correction/ldpc2_reg_index_BG1_Z176.lw
                         error_correction/ldpc2_reg_index_BG1_Z192.lw
                         error_correction/ldpc2_reg_index_BG1_Z208.lw
                         error_correction/ldpc2_reg_index_BG1_Z224.lw
                         error_correction/ldpc2_reg_index_BG1_Z240.lw
                         error_correction/ldpc2_reg_index_BG1_Z256.lw
                         error_correction/ldpc2_reg_index_BG1_Z288.lw
                         error_correction/ldpc2_reg_index_BG1_Z320.lw
                         error_correction/ldpc2_reg_index_BG1_Z352.lw
                         error_correction/ldpc2_reg_index_BG1_Z384.lw
                         error_correction/ldpc2_reg_index_BG2_Z36.lw
                         error_correction/ldpc2_reg_index_BG2_Z40.lw
                         error_correction/ldpc2_reg_index_BG2_Z44.lw
                         error_correction/ldpc2_reg_index_BG2_Z48.lw
                         error_correction/ldpc2_reg_index_BG2_Z52.lw
                         error_correction/ldpc2_reg_index_BG2_Z56.lw
                         error_correction/ldpc2_reg_index_BG2_Z60.lw
                         error_correction/ldpc2_reg_index_BG2_Z64.lw
                         error_correction/ldpc2_reg_index_BG2_Z72.lw
                         error_correction/ldpc2_reg_index_BG2_Z80.lw
                         error_correction/ldpc2_reg_index_BG2_Z88.lw
                         error_correction/ldpc2_reg_index_BG2_Z96.lw
                         error_correction/ldpc2_reg_index_BG2_Z104.lw
                         error_correction/ldpc2_reg_index_BG2_Z112.lw
                         error_correction/ldpc2_reg_index_BG2_Z120.lw
                         error_correction/ldpc2_reg_index_BG2_Z128.lw
                         error_correction/ldpc2_reg_index_BG2_Z144.lw
                         error_correction/ldpc2_reg_index_BG2_Z160.lw
                         error_correction/ldpc2_reg_index_BG2_Z176.lw
                         error_correction/ldpc2_reg_index_BG2_Z192.lw
                         error_correction/ldpc2_reg_index_BG2_Z208.lw
                         error_correction/ldpc2_reg_index_BG2_Z224.lw
                         error_correction/ldpc2_reg_index_BG2_Z240.lw
                         error_correction/ldpc2_reg_index_BG2_Z256.lw
                         error_correction/ldpc2_reg_index_BG2_Z288.lw
                         error_correction/ldpc2_reg_index_BG2_Z320.lw
                         error_correction/ldpc2_reg_index_BG2_Z352.lw
                         error_correction/ldpc2_reg_index_BG2_Z384.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z36.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z40.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z44.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z48.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z52.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z56.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z60.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z64.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z72.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z80.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z88.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z96.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z104.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z112.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z120.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z128.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z144.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z160.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z176.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z192.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z208.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z224.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z240.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z256.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z288.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z320.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z352.lw
                         error_correction/ldpc2_reg_index_fp_BG1_Z384.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z36.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z40.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z44.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z48.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z52.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z56.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z60.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z64.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z72.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z80.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z88.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z96.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z104.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z112.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z120.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z128.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z144.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z160.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z176.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z192.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z208.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z224.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z240.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z256.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z288.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z320.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z352.lw
                         error_correction/ldpc2_reg_index_fp_BG2_Z384.lw
                         error_correction/ldpc2_c2v_x2.lwh
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z36.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z40.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z44.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z48.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z52.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z56.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z60.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z64.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z72.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z80.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z88.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z96.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z104.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z112.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z120.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z128.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z144.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z160.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z176.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z192.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z208.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z224.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z240.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z256.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z288.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z320.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z352.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG1_Z384.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z36.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z40.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z44.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z48.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z52.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z56.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z60.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z64.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z72.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z80.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z88.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z96.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z104.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z112.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z120.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z128.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z144.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z160.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z176.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z192.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z208.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z224.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z240.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z256.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z288.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z320.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z352.lw
                         error_correction/ldpc2_reg_index_fp_x2_BG2_Z384.lw
                         error_correction/ldpc2_reg_address_BG1_Z36.lw
                         error_correction/ldpc2_reg_address_BG1_Z40.lw
                         error_correction/ldpc2_reg_address_BG1_Z44.lw
                         error_correction/ldpc2_reg_address_BG1_Z48.lw
                         error_correction/ldpc2_reg_address_BG1_Z52.lw
                         error_correction/ldpc2_reg_address_BG1_Z56.lw
                         error_correction/ldpc2_reg_address_BG1_Z60.lw
                         error_correction/ldpc2_reg_address_BG1_Z64.lw
                         error_correction/ldpc2_reg_address_BG1_Z72.lw
                         error_correction/ldpc2_reg_address_BG1_Z80.lw
                         error_correction/ldpc2_reg_address_BG1_Z88.lw
                         error_correction/ldpc2_reg_address_BG1_Z96.lw
                         error_correction/ldpc2_reg_address_BG1_Z104.lw
                         error_correction/ldpc2_reg_address_BG1_Z112.lw
                         error_correction/ldpc2_reg_address_BG1_Z120.lw
                         error_correction/ldpc2_reg_address_BG1_Z128.lw
                         error_correction/ldpc2_reg_address_BG1_Z144.lw
                         error_correction/ldpc2_reg_address_BG1_Z160.lw
                         error_correction/ldpc2_reg_address_BG1_Z176.lw
                         error_correction/ldpc2_reg_address_BG1_Z192.lw
                         error_correction/ldpc2_reg_address_BG1_Z208.lw
                         error_correction/ldpc2_reg_address_BG1_Z224.lw
                         error_correction/ldpc2_reg_address_BG1_Z240.lw
                         error_correction/ldpc2_reg_address_BG1_Z256.lw
                         error_correction/ldpc2_reg_address_BG1_Z288.lw
                         error_correction/ldpc2_reg_address_BG1_Z320.lw
                         error_correction/ldpc2_reg_address_BG1_Z352.lw
                         error_correction/ldpc2_reg_address_BG1_Z384.lw
                         error_correction/ldpc2_reg_address_BG2_Z36.lw
                         error_correction/ldpc2_reg_address_BG2_Z40.lw
                         error_correction/ldpc2_reg_address_BG2_Z44.lw
                         error_correction/ldpc2_reg_address_BG2_Z48.lw
                         error_correction/ldpc2_reg_address_BG2_Z52.lw
                         error_correction/ldpc2_reg_address_BG2_Z56.lw
                         error_correction/ldpc2_reg_address_BG2_Z60.lw
                         error_correction/ldpc2_reg_address_BG2_Z64.lw
                         error_correction/ldpc2_reg_address_BG2_Z72.lw
                         error_correction/ldpc2_reg_address_BG2_Z80.lw
                         error_correction/ldpc2_reg_address_BG2_Z88.lw
                         error_correction/ldpc2_reg_address_BG2_Z96.lw
                         error_correction/ldpc2_reg_address_BG2_Z104.lw
                         error_correction/ldpc2_reg_address_BG2_Z112.lw
                         error_correction/ldpc2_reg_address_BG2_Z120.lw
                         error_correction/ldpc2_reg_address_BG2_Z128.lw
                         error_correction/ldpc2_reg_address_BG2_Z144.lw
                         error_correction/ldpc2_reg_address_BG2_Z160.lw
                         error_correction/ldpc2_reg_address_BG2_Z176.lw
                         error_correction/ldpc2_reg_address_BG2_Z192.lw
                         error_correction/ldpc2_reg_address_BG2_Z208.lw
                         error_correction/ldpc2_reg_address_BG2_Z224.lw
                         error_correction/ldpc2_reg_address_BG2_Z240.lw
                         error_correction/ldpc2_reg_address_BG2_Z256.lw
                         error_correction/ldpc2_reg_address_BG2_Z288.lw
                         error_correction/ldpc2_reg_address_BG2_Z320.lw
                         error_correction/ldpc2_reg_address_BG2_Z352.lw
                         error_correction/ldpc2_reg_address_BG2_Z384.lw
                         error_correction/ldpc2_global.hpp
                         error_correction/ldpc2_global.cpp
                         error_correction/ldpc2_global.lwh
                         error_correction/ldpc2_global_index_BG1_Z36.lw
                         error_correction/ldpc2_global_index_BG1_Z40.lw
                         error_correction/ldpc2_global_index_BG1_Z44.lw
                         error_correction/ldpc2_global_index_BG1_Z48.lw
                         error_correction/ldpc2_global_index_BG1_Z52.lw
                         error_correction/ldpc2_global_index_BG1_Z56.lw
                         error_correction/ldpc2_global_index_BG1_Z60.lw
                         error_correction/ldpc2_global_index_BG1_Z64.lw
                         error_correction/ldpc2_global_index_BG1_Z72.lw
                         error_correction/ldpc2_global_index_BG1_Z80.lw
                         error_correction/ldpc2_global_index_BG1_Z88.lw
                         error_correction/ldpc2_global_index_BG1_Z96.lw
                         error_correction/ldpc2_global_index_BG1_Z104.lw
                         error_correction/ldpc2_global_index_BG1_Z112.lw
                         error_correction/ldpc2_global_index_BG1_Z120.lw
                         error_correction/ldpc2_global_index_BG1_Z128.lw
                         error_correction/ldpc2_global_index_BG1_Z144.lw
                         error_correction/ldpc2_global_index_BG1_Z160.lw
                         error_correction/ldpc2_global_index_BG1_Z176.lw
                         error_correction/ldpc2_global_index_BG1_Z192.lw
                         error_correction/ldpc2_global_index_BG1_Z208.lw
                         error_correction/ldpc2_global_index_BG1_Z224.lw
                         error_correction/ldpc2_global_index_BG1_Z240.lw
                         error_correction/ldpc2_global_index_BG1_Z256.lw
                         error_correction/ldpc2_global_index_BG1_Z288.lw
                         error_correction/ldpc2_global_index_BG1_Z320.lw
                         error_correction/ldpc2_global_index_BG1_Z352.lw
                         error_correction/ldpc2_global_index_BG1_Z384.lw
                         error_correction/ldpc2_global_index_BG2_Z36.lw
                         error_correction/ldpc2_global_index_BG2_Z40.lw
                         error_correction/ldpc2_global_index_BG2_Z44.lw
                         error_correction/ldpc2_global_index_BG2_Z48.lw
                         error_correction/ldpc2_global_index_BG2_Z52.lw
                         error_correction/ldpc2_global_index_BG2_Z56.lw
                         error_correction/ldpc2_global_index_BG2_Z60.lw
                         error_correction/ldpc2_global_index_BG2_Z64.lw
                         error_correction/ldpc2_global_index_BG2_Z72.lw
                         error_correction/ldpc2_global_index_BG2_Z80.lw
                         error_correction/ldpc2_global_index_BG2_Z88.lw
                         error_correction/ldpc2_global_index_BG2_Z96.lw
                         error_correction/ldpc2_global_index_BG2_Z104.lw
                         error_correction/ldpc2_global_index_BG2_Z112.lw
                         error_correction/ldpc2_global_index_BG2_Z120.lw
                         error_correction/ldpc2_global_index_BG2_Z128.lw
                         error_correction/ldpc2_global_index_BG2_Z144.lw
                         error_correction/ldpc2_global_index_BG2_Z160.lw
                         error_correction/ldpc2_global_index_BG2_Z176.lw
                         error_correction/ldpc2_global_index_BG2_Z192.lw
                         error_correction/ldpc2_global_index_BG2_Z208.lw
                         error_correction/ldpc2_global_index_BG2_Z224.lw
                         error_correction/ldpc2_global_index_BG2_Z240.lw
                         error_correction/ldpc2_global_index_BG2_Z256.lw
                         error_correction/ldpc2_global_index_BG2_Z288.lw
                         error_correction/ldpc2_global_index_BG2_Z320.lw
                         error_correction/ldpc2_global_index_BG2_Z352.lw
                         error_correction/ldpc2_global_index_BG2_Z384.lw
                         error_correction/ldpc2_shared.hpp
                         error_correction/ldpc2_shared.cpp
                         error_correction/ldpc2_shared_BG1_Z64.lw
                         error_correction/ldpc2_shared_BG1_Z96.lw
                         error_correction/ldpc2_shared_BG1_Z128.lw
                         error_correction/ldpc2_shared_BG1_Z160.lw
                         error_correction/ldpc2_shared_BG1_Z192.lw
                         error_correction/ldpc2_shared_BG1_Z224.lw
                         error_correction/ldpc2_shared_BG1_Z256.lw
                         error_correction/ldpc2_shared_BG1_Z288.lw
                         error_correction/ldpc2_shared_BG1_Z320.lw
                         error_correction/ldpc2_shared_BG1_Z352.lw
                         error_correction/ldpc2_shared_BG1_Z384.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z36.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z40.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z44.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z48.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z52.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z56.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z60.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z64.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z72.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z80.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z88.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z96.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z104.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z112.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z120.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z128.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z144.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z160.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z176.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z192.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z208.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z224.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z240.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z256.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z288.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z320.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z352.lw
                         error_correction/ldpc2_shared_index_fp_x2_BG1_Z384.lw
                         error_correction/ldpc2_shared_dynamic.hpp
                         error_correction/ldpc2_shared_dynamic.cpp
                         error_correction/ldpc2_shared_dynamic_BG1_Z384.lw
                         error_correction/ldpc2_shared_cluster.lwh
                         error_correction/ldpc2_shared_cluster.hpp
                         error_correction/ldpc2_shared_cluster.cpp
                         error_correction/ldpc2_shared_cluster_BG1_Z64.lw
                         error_correction/ldpc2_shared_cluster_BG1_Z96.lw
                         error_correction/ldpc2_shared_cluster_BG1_Z128.lw
                         error_correction/ldpc2_shared_cluster_BG1_Z160.lw
                         error_correction/ldpc2_shared_cluster_BG1_Z192.lw
                         error_correction/ldpc2_shared_cluster_BG1_Z224.lw
                         error_correction/ldpc2_shared_cluster_BG1_Z256.lw
                         error_correction/ldpc2_shared_cluster_BG1_Z288.lw
                         error_correction/ldpc2_shared_cluster_BG1_Z320.lw
                         error_correction/ldpc2_shared_cluster_BG1_Z352.lw
                         error_correction/ldpc2_shared_cluster_BG1_Z384.lw
                         error_correction/ldpc2_shared_cluster_BG2_Z64.lw
                         error_correction/ldpc2_shared_cluster_BG2_Z96.lw
                         error_correction/ldpc2_shared_cluster_BG2_Z128.lw
                         error_correction/ldpc2_shared_cluster_BG2_Z160.lw
                         error_correction/ldpc2_shared_cluster_BG2_Z192.lw
                         error_correction/ldpc2_shared_cluster_BG2_Z224.lw
                         error_correction/ldpc2_shared_cluster_BG2_Z256.lw
                         error_correction/ldpc2_shared_cluster_BG2_Z288.lw
                         error_correction/ldpc2_shared_cluster_BG2_Z320.lw
                         error_correction/ldpc2_shared_cluster_BG2_Z352.lw
                         error_correction/ldpc2_shared_cluster_BG2_Z384.lw
                         error_correction/ldpc2_split.lwh
                         error_correction/ldpc2_split.hpp
                         error_correction/ldpc2_split.cpp
                         error_correction/ldpc2_split_96KB.cpp
                         error_correction/ldpc2_split_96KB_BG1_Z64.lw
                         error_correction/ldpc2_split_96KB_BG1_Z96.lw
                         error_correction/ldpc2_split_96KB_BG1_Z128.lw
                         error_correction/ldpc2_split_96KB_BG1_Z160.lw
                         error_correction/ldpc2_split_96KB_BG1_Z192.lw
                         error_correction/ldpc2_split_96KB_BG1_Z224.lw
                         error_correction/ldpc2_split_96KB_BG1_Z256.lw
                         error_correction/ldpc2_split_96KB_BG1_Z288.lw
                         error_correction/ldpc2_split_96KB_BG1_Z320.lw
                         error_correction/ldpc2_split_96KB_BG1_Z352.lw
                         error_correction/ldpc2_split_96KB_BG1_Z384.lw
                         error_correction/ldpc2_split_96KB_BG2_Z64.lw
                         error_correction/ldpc2_split_96KB_BG2_Z96.lw
                         error_correction/ldpc2_split_96KB_BG2_Z128.lw
                         error_correction/ldpc2_split_96KB_BG2_Z160.lw
                         error_correction/ldpc2_split_96KB_BG2_Z192.lw
                         error_correction/ldpc2_split_96KB_BG2_Z224.lw
                         error_correction/ldpc2_split_96KB_BG2_Z256.lw
                         error_correction/ldpc2_split_96KB_BG2_Z288.lw
                         error_correction/ldpc2_split_96KB_BG2_Z320.lw
                         error_correction/ldpc2_split_96KB_BG2_Z352.lw
                         error_correction/ldpc2_split_96KB_BG2_Z384.lw
                         error_correction/ldpc2_split_dynamic.lwh
                         error_correction/ldpc2_split_dynamic.hpp
                         error_correction/ldpc2_split_dynamic.cpp
                         error_correction/ldpc2_split_dynamic_96KB.cpp
                         error_correction/ldpc2_split_dynamic_96KB_BG1_Z288.lw
                         error_correction/ldpc2_split_dynamic_96KB_BG1_Z320.lw
                         error_correction/ldpc2_split_dynamic_96KB_BG1_Z352.lw
                         error_correction/ldpc2_split_dynamic_96KB_BG1_Z384.lw
                         error_correction/ldpc2_split_cluster.lwh
                         error_correction/ldpc2_split_cluster.hpp
                         error_correction/ldpc2_split_cluster.cpp
                         error_correction/ldpc2_split_cluster_96KB.cpp
                         error_correction/ldpc2_split_cluster_96KB_BG1_Z384.lw
                         error_correction/ldpc2.lwh
                         error_correction/ldpc2.cpp
                         error_correction/ldpc_encode.lw
                         error_correction/nrLDPC_templates.lwh
                         crc/crc.lw
                         descrambling/descrambling.lw
                         pucch_receiver/pucch_receiver.lw
                         modulation_mapper/modulation_mapper.lw
                         pdcch/embed_pdcch_tf_signal.lw
                         pdsch_dmrs/pdsch_dmrs.lw
                         ss/ss.lw
                         polar_encoder/polar_encoder.lw
                         )

# CMake LINK_FLAGS_<CONFIG> was superseded by LINK_OPTIONS around version
# v3.13, but we still may be running on older versions.
set_property(TARGET lwphy APPEND PROPERTY LINK_FLAGS_RELEASE "-Wl,--version-script=${CMAKE_LWRRENT_SOURCE_DIR}/lwphy.map -s")

add_dependencies(lwphy CRCLUTGEN)
add_dependencies(lwphy PUCCHLUTGEN)
target_include_directories(lwphy PUBLIC ${CMAKE_LWDA_TOOLKIT_INCLUDE_DIRECTORIES} $<BUILD_INTERFACE:${CMAKE_LWRRENT_SOURCE_DIR}>)
target_include_directories(lwphy PRIVATE crc
                                         channel_est
                                         channel_eq
                                         ch_est
                                         descrambling
                                         dl_rate_matching
                                         rate_matching 
                                         modulation_mapper
                                         "${CMAKE_LWRRENT_BINARY_DIR}/LUTS"
                                         error_correction
                                         pucch_receiver
                                         pdsch_dmrs
                                         ss
                                         polar_encoder
                                         )
# Use target_compile_options() to set LWCA-specific flags:
#target_compile_options(<my_target> PRIVATE $<$<COMPILE_LANGUAGE:LWCA>:my_lwda_option>)
#target_compile_options(lwphy PRIVATE $<$<COMPILE_LANGUAGE:LWCA>:${LWPHY_OPTS}>)

#set(LWPHY_OPTS --ptxas-options=-v --resource-usage --lwdart static ${LWPHY_GENCODE_STRING} -lineinfo)
#set(LWPHY_OPTS --ptxas-options=-v --resource-usage --lwdart static "SHELL:-gencode arch=compute_70,code=sm_70" "SHELL:-gencode arch=compute_75,code=sm_75" -lineinfo)
#message(${LWPHY_OPTS})

target_compile_options(lwphy PRIVATE $<$<COMPILE_LANGUAGE:LWCA>:
                       ${LWPHY_PTXAS_OPTIONS}
                       --lwdart static
                       ${LWPHY_GENCODE_STRING}
                       -lineinfo
                       >)

# Add the line below to the list above to keep intermediate lwcc files
#                      --keep
# Add the line below to the list above to compile GPU kernelw with debugging info:
#                      -g -G
#set_source_files_properties(error_correction/ldpc2_split_cluster_96KB_BG1_Z384.lw PROPERTIES COMPILE_FLAGS --keep)
#set_source_files_properties(error_correction/ldpc2_split_dynamic_96KB_BG1_Z384.lw PROPERTIES COMPILE_FLAGS --keep)
#set_source_files_properties(error_correction/ldpc2_reg_index_BG1_Z384.lw PROPERTIES COMPILE_FLAGS --keep)
#set_source_files_properties(error_correction/ldpc2_reg_index_fp_BG1_Z384.lw PROPERTIES COMPILE_FLAGS --keep)
#set_source_files_properties(error_correction/ldpc2_reg_index_fp_x2_BG1_Z384.lw PROPERTIES COMPILE_FLAGS --keep)
set_source_files_properties(error_correction/ldpc_ms_cta_shmem_layered.lw PROPERTIES COMPILE_FLAGS --use_fast_math)
set_source_files_properties(pucch_receiver/pucch_receiver.lw PROPERTIES COMPILE_FLAGS --use_fast_math)

install(TARGETS lwphy DESTINATION lib)
install(FILES lwphy.h utils.lwh DESTINATION include)
