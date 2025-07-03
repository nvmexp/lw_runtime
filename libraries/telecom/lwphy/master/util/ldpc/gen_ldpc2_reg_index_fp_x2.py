#!/usr/bin/elw python

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

#include "ldpc2_reg.lwh"
#include "ldpc2_app_address_fp.lwh"
#include "ldpc2_c2v_x2.lwh"

namespace ldpc2
{{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_fp_x2_half_BG1_Z{Z}()
lwphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG1_Z{Z}(ldpc::decoder&            dec,
                                                         const LDPC_config&        cfg,
                                                         const LDPC_kernel_params& params,
                                                         const dim3&               grdDim,
                                                         const dim3&               blkDim,
                                                         lwdaStream_t              strm)
{{
    lwphyStatus_t s = LWPHY_STATUS_NOT_SUPPORTED;

{guard_begin}    constexpr int  BG = 1;
    constexpr int  Z  = {Z};
    constexpr int  Kb = 22;

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    typedef __half2 T;
    // Check to Variable (C2V) message type
    typedef cC2V_index<T, BG, unused, sign_mgr_pair_src> cC2V_t;
    
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

#include "ldpc2_reg.lwh"
#include "ldpc2_app_address_fp.lwh"
#include "ldpc2_c2v_x2.lwh"

namespace ldpc2
{{

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_reg_index_fp_x2_half_BG2_Z{Z}()
lwphyStatus_t decode_ldpc2_reg_index_fp_x2_half_BG2_Z{Z}(ldpc::decoder&            dec,
                                                         const LDPC_config&        cfg,
                                                         const LDPC_kernel_params& params,
                                                         const dim3&               grdDim,
                                                         const dim3&               blkDim,
                                                         lwdaStream_t              strm)
{{
    lwphyStatus_t s = LWPHY_STATUS_NOT_SUPPORTED;

{guard_begin}    constexpr int  BG = 2;
    constexpr int  Z  = {Z};
    constexpr int  Kb = {Kb};

    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    typedef __half2 T;
    // Check to Variable (C2V) message type
    typedef cC2V_index<T, BG, unused, sign_mgr_pair_src> cC2V_t;

    switch(cfg.mb)
    {{
"""

case_line = """    case  %2i:  s = launch_register_kernel<T, BG, Kb, Z, %2i, cC2V_t, app_loc_address_fp_imad, 1>(dec, params, grdDim, blkDim, strm);         break;
"""

def gen_launch_cases(max_parity_nodes):
    str = ""
    for mb in range(4, max_parity_nodes + 1):
        str = str + case_line % (mb, mb)
    return str

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


# Dictionary to determine the maximum number of parity nodes that we
# should generate a switch case for. (In some cases, large parity node
# counts result in local memory spills that kill performance, and the
# resulting kernel would not be used.) Keeping the number of generated
# kernels as small as possible minimizes compile time.
# TODO: Automate the process of determining these values.
max_num_parity_BG1 = { 2   : 12,
                       3   : 12,
                       4   : 12,
                       5   : 12,
                       6   : 12,
                       7   : 12,
                       8   : 12,
                       9   : 12,
                       10  : 12,
                       11  : 12,
                       12  : 12,
                       13  : 12,
                       14  : 12,
                       15  : 12,
                       16  : 12,
                       18  : 12,
                       20  : 12,
                       22  : 12,
                       24  : 12,
                       26  : 12,
                       28  : 12,
                       30  : 12,
                       32  : 12,
                       36  : 12,
                       40  : 12,
                       44  : 12,
                       48  : 12,
                       52  : 12,
                       56  : 12,
                       60  : 12,
                       64  : 24,
                       72  : 24,
                       80  : 24,
                       88  : 24,
                       96  : 24,
                       104 : 24,
                       112 : 24,
                       120 : 24,
                       128 : 24,
                       144 : 24,
                       160 : 24,
                       176 : 24,
                       192 : 24,
                       208 : 24,
                       224 : 24,
                       240 : 24,
                       256 : 24,
                       288 : 16,
                       320 : 16,
                       352 : 16,
                       384 : 16 }
max_num_parity_BG2 = { 2   : 20,
                       3   : 20,
                       4   : 20,
                       5   : 20,
                       6   : 20,
                       7   : 20,
                       8   : 20,
                       9   : 20,
                       10  : 20,
                       11  : 20,
                       12  : 20,
                       13  : 20,
                       14  : 20,
                       15  : 20,
                       16  : 20,
                       18  : 20,
                       20  : 20,
                       22  : 20,
                       24  : 20,
                       26  : 20,
                       28  : 20,
                       30  : 20,
                       32  : 20,
                       36  : 20,
                       40  : 20,
                       44  : 20,
                       48  : 20,
                       52  : 20,
                       56  : 20,
                       60  : 20,
                       64  : 42,
                       72  : 42,
                       80  : 42,
                       88  : 42,
                       96  : 42,
                       104 : 42,
                       112 : 42,
                       120 : 42,
                       128 : 42,
                       144 : 42,
                       160 : 42,
                       176 : 42,
                       192 : 36,
                       208 : 36,
                       224 : 36,
                       240 : 36,
                       256 : 40,
                       288 : 24,
                       320 : 24,
                       352 : 24,
                       384 : 24 }

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

# Max num parity nodes determined by experimentation. Local memory
# spills occur at around 12 for BG1, Z = 384.
for Z_val in Z_lwrrent:
    sub_dict = {'Z':                 str(Z_val),
                #                                 ALGO   Z
                'guard_begin':  guard_dict_begin[(False, Z_val in Z_optional)],
                'guard_end':    guard_dict_end  [(False, Z_val in Z_optional)]}
    formatted_file = lw_file_bg1_hdr.format(**sub_dict)
    formatted_file = formatted_file + gen_launch_cases(max_num_parity_BG1[Z_val]) + lw_file_ftr.format(**sub_dict)
    fname = 'ldpc2_reg_index_fp_x2_BG1_Z%d.lw' % Z_val
    fOut = open(fname, 'w')
    print('Writing %s (max nodes = %i)' % (fname, max_num_parity_BG1[Z_val]))
    fOut.write(formatted_file)

# Max num parity nodes determined by experimentation. Local memory
# spills occur at around 20 for BG2, Z = 384.
for Z_val in Z_lwrrent:
    # TODO: Fix for other Kb values
    Kb_val = 10 if (Z_val > 64) else 9
    sub_dict = {'Z':                 str(Z_val),
                'Kb':                str(Kb_val),
                #                                 ALGO   Z
                'guard_begin':  guard_dict_begin[(False, Z_val in Z_optional)],
                'guard_end':    guard_dict_end  [(False, Z_val in Z_optional)]}
    formatted_file = lw_file_bg2_hdr.format(**sub_dict)
    formatted_file = formatted_file + gen_launch_cases(max_num_parity_BG2[Z_val]) + lw_file_ftr.format(**sub_dict)
    fname = 'ldpc2_reg_index_fp_x2_BG2_Z%d.lw' % Z_val
    fOut = open(fname, 'w')
    print('Writing %s (max nodes = %i)' % (fname, max_num_parity_BG2[Z_val]))
    fOut.write(formatted_file)
