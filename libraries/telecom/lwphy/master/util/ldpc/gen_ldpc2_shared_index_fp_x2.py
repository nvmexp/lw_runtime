#!/usr/bin/elw python

import math

# Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
#
# LWPU CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from LWPU CORPORATION is strictly prohibited.

lw_file_bg1_hdr="""/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

//#define LWPHY_DEBUG 1

#include "ldpc2_c2v_x2.lwh"
#include "ldpc2_app_address_fp.lwh"
#include "ldpc2_shared.lwh"

namespace ldpc2
{{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index_fp_x2_half_BG1_Z{Z}()
lwphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG1_Z{Z}(ldpc::decoder&            dec,
                                                            const LDPC_config&        cfg,
                                                            const LDPC_kernel_params& params,
                                                            const dim3&               grdDim,
                                                            const dim3&               blkDim,
                                                            lwdaStream_t              strm)
{{
    lwphyStatus_t  s = LWPHY_STATUS_NOT_SUPPORTED;
{guard_begin}    constexpr int  BG = 1;
    constexpr int  Z  = {Z};
    constexpr int  Kb = 22;
    
    typedef __half2 T;
    typedef cC2V_index<T, BG, unused, sign_mgr_pair_src> cC2V_t;

    // Maximum parity node based on 96 KiB shared mem configuration
    
    switch(cfg.mb)
    {{
"""

lw_file_ftr="""    default:                                                                                                                  break;
    }}
{guard_end}    return s;
}}

}} // namespace ldpc2
"""

lw_file_bg2_hdr="""/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

//#define LWPHY_DEBUG 1

#include "ldpc2_c2v_x2.lwh"
#include "ldpc2_app_address_fp.lwh"
#include "ldpc2_shared.lwh"

namespace ldpc2
{{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_shared_index_fp_x2_half_BG2_Z{Z}()
lwphyStatus_t decode_ldpc2_shared_index_fp_x2_half_BG2_Z{Z}(ldpc::decoder&            dec,
                                                            const LDPC_config&        cfg,
                                                            const LDPC_kernel_params& params,
                                                            const dim3&               grdDim,
                                                            const dim3&               blkDim,
                                                            lwdaStream_t              strm)
{{
    lwphyStatus_t  s  = LWPHY_STATUS_NOT_SUPPORTED;
    constexpr int  BG = 2;
    constexpr int  Z  = {Z};
    constexpr int  Kb = {Kb};
    
    typedef __half2 T;
    typedef cC2V_index<T, BG, unused, sign_mgr_pair_src> cC2V_t;

    // Maximum parity node based on 96 KiB shared mem configuration

    switch(cfg.mb)
    {{
"""

case_line = """    case  %2i:  s = launch_all_shared_strided<T, BG, Kb, Z, %2i, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
"""
# gen_launch_cases()
# Return a string with case statements from 4 to max_parity_nodes
def gen_launch_cases(max_parity_nodes):
    str = ""
    for mb in range(4, max_parity_nodes + 1):
        str = str + case_line % (mb, mb)
    return str

# Assuming 2 bytes APP values, with 2 codewords at a time
# Assuming first 4 parity nodes store 4 words for cC2V (for 2 codewords),
# and the remaining parity nodes can store cC2V for both codewords in
# 3 32-bit words. See:
# src/lwphy/error_correction/ldpc2_c2v_cache_shared_strided.lwh
def get_max_parity_nodes(bg, Z, shmem_kb):
    bytes_per_APP        = 2
    cw_per_sm            = 2
    bytes_per_word       = 4
    words_per_cC2V_base  = 4
    words_per_cC2V_other = 3
    base_parity_nodes    = 4
    mem_bytes = shmem_kb * 1024
    if 1 == bg:
        Kinfo = 22
        max_bg_nodes = 46
    else:
        Kinfo = 10
        max_bg_nodes = 42
    # The "base" shared memory requirement is the amount of
    # shared memory used for the 4 parity node case.
    base_shmem = Z * ((Kinfo + base_parity_nodes) * bytes_per_APP * cw_per_sm + (bytes_per_word * words_per_cC2V_base * base_parity_nodes))
    if base_shmem > mem_bytes:
        return 0
    # Each additional parity node (after the "base" case) will add Z
    # pairs of APP values and Z cC2V entries.
    bytes_per_node = Z * (bytes_per_APP * cw_per_sm + words_per_cC2V_other * bytes_per_word)
    max_nodes = base_parity_nodes + math.floor((mem_bytes - base_shmem) / bytes_per_node)
    #print('base_shmem = %i, bytes_per_node = %i, max_nodes = %i' % (base_shmem, bytes_per_node, max_nodes))
    if max_nodes > max_bg_nodes:
        max_nodes = max_bg_nodes
    return int(max_nodes)
    

Z_full = [2,  4,  8,  16, 32,   64, 128, 256, # 0
          3,  6, 12,  24, 48,   96, 192, 384, # 1
          5, 10, 20,  40, 80,  160, 320,      # 2
          7, 14, 28,  56, 112, 224,           # 3
          9, 18, 36,  72, 144, 288,           # 4
         11, 22, 44,  88, 176, 352,           # 5
         13, 26, 52, 104, 208,                # 6
         15, 30, 60, 120, 240]                # 7
#Z_lwrrent = [ 64,  96, 128, 160, 192, 224, 240, 256, 288, 320, 352, 384]
#Z_lwrrent = [384]
#Z_lwrrent = [ 64,  96, 128, 160, 192, 224, 256, 288, 320, 352]
Z_lwrrent = [ 36,
              40,
              44,
              48,
              52,
              56,
              60,
              64,
              72,
              80,
              88,
              96,
              104,
              112,
              120,
              128,
              144,
              160,
              176,
              192,
              208,
              224,
              240,
              256,
              288,
              320,
              352,
              384]

# Compilation of these lifting sizes will be guarded by a #define to
# reduce compile time.
Z_optional = [36,
              40,
              44,
              48,
              52,
              56,
              60,
              72,
              80,
              88,
              104,
              112,
              120,
              144,
              176,
              208,
              240]

# We will use a preprocessor conditional around some kernel template
# instantiations.
# We lwrrently have two reasons that we might not want to compile
# an LDPC implementation:
# 1.) The algorithm would not be chosen by the decoder algorithm selection
# 2.) The lifting size is not a common lifting size. (This would be a
#     temporary condition, until the number of test vectors increases to
#     exercise all lifting sizes.)
algo_guard_begin = '#if LWPHY_LDPC_INCLUDE_ALL_ALGOS\n'
algo_guard_end   = '#endif // if LWPHY_LDPC_INCLUDE_ALL_ALGOS\n'
Z_guard_begin    = '#if LWPHY_LDPC_INCLUDE_ALL_LIFTING\n'
Z_guard_end      = '#endif // if LWPHY_LDPC_INCLUDE_ALL_LIFTING\n'
algo_Z_guard_begin = '#if LWPHY_LDPC_INCLUDE_ALL_ALGOS && LWPHY_LDPC_INCLUDE_ALL_LIFTING\n'
algo_Z_guard_end   = '#endif // if LWPHY_LDPC_INCLUDE_ALL_ALGOS && LWPHY_LDPC_INCLUDE_ALL_LIFTING\n'

#                    ALGO   Z
guard_dict_begin = {(False, False): '',
                    (False, True):  Z_guard_begin,
                    (True,  False): algo_guard_begin,
                    (True,  True):  algo_Z_guard_begin}
#                  ALGO   Z
guard_dict_end = {(False, False): '',
                  (False, True):  Z_guard_end,
                  (True,  False): algo_guard_end,
                  (True,  True):  algo_Z_guard_end}

for Z_val in Z_lwrrent:
    max_nodes = get_max_parity_nodes(1, Z_val, 96)
    sub_dict = {'Z':                 str(Z_val),
                #                                 ALGO   Z
                'guard_begin':  guard_dict_begin[(False, Z_val in Z_optional)],
                'guard_end':    guard_dict_end  [(False, Z_val in Z_optional)]}
    formatted_file = lw_file_bg1_hdr.format(**sub_dict)
    formatted_file = formatted_file + gen_launch_cases(max_nodes) + lw_file_ftr.format(**sub_dict)
    fname = 'ldpc2_shared_index_fp_x2_BG1_Z%d.lw' % Z_val
    fOut = open(fname, 'w')

    print('Writing %s (max nodes = %i)' % (fname, max_nodes))
    fOut.write(formatted_file)

# Note: With a smaller maximum row degree, BG2 implementatiions use
# fewer registers, and shared memory 2x codewords at a time
# implementations are not faster. This may change for post-Volta.
#for Z_val in Z_lwrrent:
#    # TODO: Fix for other Kb values
#    Kb_val = 10 if (Z_val > 64) else 9
#    max_nodes = get_max_parity_nodes(2, Z_val, 96)
#    formatted_file = lw_file_bg2_hdr.format(Z=str(Z_val), Kb=str(Kb_val))
#    formatted_file = formatted_file + gen_launch_cases(max_nodes) + lw_file_ftr
#    fname = 'ldpc2_shared_index_fp_x2_BG2_Z%d.lw' % Z_val
#    fOut = open(fname, 'w')
#    print('Writing %s (max nodes = %i)' % (fname, max_nodes))
#    fOut.write(formatted_file)
