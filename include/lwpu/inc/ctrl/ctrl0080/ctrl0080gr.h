/*
 * SPDX-FileCopyrightText: Copyright (c) 2004-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl0080/ctrl0080gr.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl0080/ctrl0080base.h"

typedef struct LW0080_CTRL_GR_ROUTE_INFO {
    LwU32 flags;
    LW_DECLARE_ALIGNED(LwU64 route, 8);
} LW0080_CTRL_GR_ROUTE_INFO;

/* LW01_DEVICE_XX/LW03_DEVICE gr engine control commands and parameters */

/**
 * LW0080_CTRL_CMD_GR_GET_CAPS
 *
 * This command returns the set of graphics capabilities for the device
 * in the form of an array of unsigned bytes.  Graphics capabilities
 * include supported features and required workarounds for the graphics
 * engine(s) within the device, each represented by a byte offset into the
 * table and a bit position within that byte.
 *
 *   capsTblSize
 *     This parameter specifies the size in bytes of the caps table.
 *     This value should be set to LW0080_CTRL_GR_CAPS_TBL_SIZE.
 *   capsTbl
 *     This parameter specifies a pointer to the client's caps table buffer
 *     into which the graphics caps bits will be transferred by the RM.
 *     The caps table is an array of unsigned bytes.
 */
#define LW0080_CTRL_CMD_GR_GET_CAPS (0x801102) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GR_INTERFACE_ID << 8) | LW0080_CTRL_GR_GET_CAPS_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GR_GET_CAPS_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0080_CTRL_GR_GET_CAPS_PARAMS {
    LwU32 capsTblSize;
    LW_DECLARE_ALIGNED(LwP64 capsTbl, 8);
} LW0080_CTRL_GR_GET_CAPS_PARAMS;

/* extract cap bit setting from tbl */
#define LW0080_CTRL_GR_GET_CAP(tbl,c)              (((LwU8)tbl[(1?c)]) & (0?c))

/* caps format is byte_index:bit_mask */
#define LW0080_CTRL_GR_CAPS_FF                                     0:0x04
#define LW0080_CTRL_GR_CAPS_AA_LINES                               0:0x10
#define LW0080_CTRL_GR_CAPS_AA_POLYS                               0:0x20
#define LW0080_CTRL_GR_CAPS_LOGIC_OPS                              0:0x80
#define LW0080_CTRL_GR_CAPS_2SIDED_LIGHTING                        1:0x02
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#define LW0080_CTRL_GR_CAPS_QUADRO_GENERIC                         1:0x04
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


#define LW0080_CTRL_GR_CAPS_UBB                                    1:0x08
#define LW0080_CTRL_GR_CAPS_3D_TEXTURES                            1:0x20
#define LW0080_CTRL_GR_CAPS_ANISOTROPIC                            1:0x40
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#define LW0080_CTRL_GR_CAPS_VPC_NAN_CLIPS_BUG_237942               1:0x80 // Deprecated
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


#define LW0080_CTRL_GR_CAPS_SET_SHADER_PACKER_SUPPORTED            6:0x40
#define LW0080_CTRL_GR_CAPS_SET_SHADER_SAMPLE_MASK_SUPPORTED       7:0x01
#define LW0080_CTRL_GR_CAPS_FP16_TEXTURE_BLENDING_SUPPORTED        7:0x40
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#define LW0080_CTRL_GR_CAPS_BUG_210168_VPC_LWLLS_SMOOTH_POLYGONS_BETWEEN_CENTERS 7:0x80 // Deprecated
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


#define LW0080_CTRL_GR_CAPS_SET_REDUCE_DST_COLOR_SUPPORTED         8:0x08
#define LW0080_CTRL_GR_CAPS_SET_NO_PARANOID_TEXTURE_FETCHES_SUPPORTED   8:0x10
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#define LW0080_CTRL_GR_CAPS_VIEWPORT_INDEX_BUG_287160             10:0x01 // Deprecated
#define LW0080_CTRL_GR_CAPS_CLIP_ID_BUG_236807                    11:0x04 // Deprecated
#define LW0080_CTRL_GR_CAPS_M2M_LINE_COUNT_BUG_232480             11:0x08 // Deprecated
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


#define LW0080_CTRL_GR_CAPS_ENABLE_SECTOR_PROMOTION               11:0x10
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#define LW0080_CTRL_GR_CAPS_FAST_POLYMODE_BUG_203347              11:0x20 // Deprecated
#define LW0080_CTRL_GR_CAPS_CNULL_BUG_234953                      11:0x40 // Deprecated
#define LW0080_CTRL_GR_CAPS_TICKS_BUG_269088                      11:0x80 // Deprecated
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


#define LW0080_CTRL_GR_CAPS_TURBOCIPHER_SUPPORTED                 12:0x01
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#define LW0080_CTRL_GR_CAPS_DUMMY_READ_BUG_252892                 12:0x02 // Deprecated
#define LW0080_CTRL_GR_CAPS_ZLWLL_CORRUPTION_BUG_262498           12:0x04 // Deprecated
#define LW0080_CTRL_GR_CAPS_SHADOW_MAPPING_BUG_280195             12:0x08 // Deprecated
#define LW0080_CTRL_GR_CAPS_DST_REDUCE_BUG_277021                 12:0x10 // Deprecated
#define LW0080_CTRL_GR_CAPS_SMC_RAU_THICK_BUG_282580              12:0x20 // Deprecated
#define LW0080_CTRL_GR_CAPS_G8X_BE_LEVEL_HANG_BUG_278895          12:0x40 // Deprecated (Bug 283159)
#define LW0080_CTRL_GR_CAPS_PFM_SAFE_OVERLAP_BUG_247451           12:0x80 // Deprecated (Bug 285635)
#define LW0080_CTRL_GR_CAPS_PS_TEXTURE_BUG_294656                 13:0x01 // Deprecated (Bug 294656)
#define LW0080_CTRL_GR_CAPS_GS_ENABLE_HANG_BUG_293973             13:0x02 // Deprecated (Bug 293973)
#define LW0080_CTRL_GR_CAPS_CROP_BLEND_FETCH_BUG_288726           13:0x04 // Deprecated (Bug 288726)
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


#define LW0080_CTRL_GR_CAPS_G80_SFR_PANNING_BUG_290103            13:0x08
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#define LW0080_CTRL_GR_CAPS_EARLY_Z_MULTISAMPLE_BUG_306999        13:0x10 // Deprecated
#define LW0080_CTRL_GR_CAPS_GS_HANG_BUG_322785                    13:0x20 // Deprecated
#define LW0080_CTRL_GR_CAPS_SM_1_05_BUG_232540                    13:0x40 // Deprecated (Bug 335308)
#define LW0080_CTRL_GR_CAPS_LWS_GENERIC                           13:0x80  //Lwdqro LWS detection
#define LW0080_CTRL_GR_CAPS_SET_SHADER_NAN_SAT_RETURNS_ZERO_BUG_339955    14:0x01 // Deprecated
#define LW0080_CTRL_GR_CAPS_BBOX_LWLL_MSDISABLE_BUG_310947        14:0x02 // Deprecated (Bug 310151)
#define LW0080_CTRL_GR_CAPS_DST_REDUCE_BUG_345079                 14:0x04 // Deprecated
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


#define LW0080_CTRL_GR_CAPS_PS_TEXTURE_BUG_386623                 14:0x08
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#define LW0080_CTRL_GR_CAPS_PREFER_64K_PAGES_BUG_369795           14:0x10 // Deprecated
#define LW0080_CTRL_GR_CAPS_G80_RENDER_CLIP_BUG_364207            14:0x20 // Deprecated
#define LW0080_CTRL_GR_CAPS_Z_COMPRESSION_BUG_263524              14:0x40 // Deprecated
#define LW0080_CTRL_GR_CAPS_TEX_UTLB_BUG_368516                   14:0x80 // Deprecated
#define LW0080_CTRL_GR_CAPS_ZROP_RNDR_TGT_IDX_BUG_282650          15:0x01 // Deprecated (Bug 286312)
#define LW0080_CTRL_GR_CAPS_ZLWLL_BUG_395571                      15:0x02 // Deprecated
#define LW0080_CTRL_GR_CAPS_BRX_BUG_392743                        15:0x04 // Deprecated (Bug 398076)
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


#define LW0080_CTRL_GR_CAPS_ROP_REQUIRES_CACHED_SYSMEM            15:0x10
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#define LW0080_CTRL_GR_CAPS_Z16_TRANS_BUG_436112                  15:0x20 // Deprecated (Bug 445181)
#define LW0080_CTRL_GR_CAPS_ZLWLL_CORRUPTION_BUG_417310           15:0x40 // Deprecated (Bug 412399)
#define LW0080_CTRL_GR_CAPS_VTX_OVERFETCH_BUG_282292              15:0x80 // Deprecated
#define LW0080_CTRL_GR_CAPS_FLAT_COLOR_UCLIP_BUG_250824           16:0x01 // Deprecated (Bug 306875)
#define LW0080_CTRL_GR_CAPS_QUADRO_AD                             16:0x02  //Lwdqro AD detection
#define LW0080_CTRL_GR_CAPS_STP_CHECKSUM_BUG_434787               16:0x04 // Deprecated
#define LW0080_CTRL_GR_CAPS_SMC_STRI_DEADLOCK_BUG_434797          16:0x08 // Deprecated
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


#define LW0080_CTRL_GR_CAPS_2DBLITTER_ONE_TPC_BUG_267720          16:0x10
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#define LW0080_CTRL_GR_CAPS_ZLWLL_REALLOC_SUPPORTED               16:0x20 // Deprecated
#define LW0080_CTRL_GR_CAPS_DISABLE_BLENDOPT_BUG_559108           16:0x40 // Deprecated
#define LW0080_CTRL_GR_CAPS_CROP_FAST_CLEAR_BUG_566530            16:0x80 // Deprecated
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


#define LW0080_CTRL_GR_CAPS_PSHADER_COMPILE_TXL_BUG_614058        17:0x02   // Bug 628824
#define LW0080_CTRL_GR_CAPS_SET_SINGLE_ROP_CONTROL_BUG_619383     17:0x04
#define LW0080_CTRL_GR_CAPS_VAB_RECANT_BUG_661471                 17:0x08
#define LW0080_CTRL_GR_CAPS_CBE_THROTTLING_AND_ALPHA_WORKS_COLLISION_BUG_635350 17:0x10  // Bug 648679, bug 635350
#define LW0080_CTRL_GR_CAPS_GS_ONLY_TO_TES_TRANSITION_BUG_601948  17:0x20   // Bug 605850, bug 601948
#define LW0080_CTRL_GR_CAPS_VCAA_16_BIT_FORMAT_BUG_651500         17:0x40   // Bug 574008, bug 651500
#define LW0080_CTRL_GR_CAPS_UNCOMP_ZCLEAR_WITH_SHADERZ_BUG_557758 17:0x80   // Bug 557588, bug 557758
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#define LW0080_CTRL_GR_CAPS_LWIDIA_LWS                            18:0x01   // LWPU LWS detection
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


#define LW0080_CTRL_GR_CAPS_2X_TEX_HANG_BUG_686487                18:0x02   // Bug 695237
#define LW0080_CTRL_GR_CAPS_L1_CONFIG_ILWALIDATE_SHADER_CACHE_BUG_790694 18:0x04   // Bug 790694, bug 785100
#define LW0080_CTRL_GR_CAPS_TS_GS_STREAMOUT_BUG_775275            18:0x08   // Bug 775275
#define LW0080_CTRL_GR_CAPS_CROP_HANG_CLEARS_BUG_870748           18:0x10   // Bug 870478
#define LW0080_CTRL_GR_CAPS_CROP_HANG_ATOMICS_BUG_870748          18:0x20   // Bug 870478
#define LW0080_CTRL_GR_CAPS_NEEDS_LMEM_THROTTLE_WFI_BUG_903136    18:0x40   // Bug 903136
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#define LW0080_CTRL_GR_CAPS_VGX                                   18:0x80   // VGX detection
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


#define LW0080_CTRL_GR_CAPS_ZPASS_ACLWM_TIMEOUT_COUNTER_HANG_BUG_925697    19:0x01   // Bug 925697
#define LW0080_CTRL_GR_CAPS_SET_OBJ_CTXSW_HANG_BUG_969839         19:0x02   // Bug 969839
#define LW0080_CTRL_GR_CAPS_SCC_MAX_VALID_PAGES_BUG_972630        19:0x04   // Bug 972630
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#define LW0080_CTRL_GR_CAPS_GEFORCE_SMB                           19:0x10   // VdChip SKUs specially targeted for the M&E Small to Medium Business market
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


#define LW0080_CTRL_GR_CAPS_L1_CACHE_BROKEN_BUG_986473            19:0x20   // Bug 986473
#define LW0080_CTRL_GR_CAPS_MEMBAR_BUG_980428                     19:0x40   // Bug 980428
#define LW0080_CTRL_GR_CAPS_MEMBAR_WAR_BUG_1047433                19:0x80   // Bug 1047433
#define LW0080_CTRL_GR_CAPS_FGS_ATTR_LOAD_BUG_1514369             20:0x01   // Bug 1514369
#define LW0080_CTRL_GR_CAPS_SM_WAR_BUG_1318757                    20:0x02   // Bug 1318757
#define LW0080_CTRL_GR_CAPS_UCODE_SUPPORTS_PRIV_ACCESS_MAP        20:0x04   // Whether engine validates register accesses from userspace.
#define LW0080_CTRL_GR_CAPS_SULD_CACHING_BUG_1355386              20:0x08   // Bug 1355386
#define LW0080_CTRL_GR_CAPS_IID_OPT_BUG_1405412                   20:0x10   // Bug 1405412
#define LW0080_CTRL_GR_CAPS_FORCE_64_PS_WATERMARKS_BUG_1642392    20:0x20   // Bug 1642392
#define LW0080_CTRL_GR_CAPS_TESSELATION_PERF_LOAD_BUG_1665952     20:0x40   // Bug 1665952
#define LW0080_CTRL_GR_CAPS_IQ2M_NO_RENDER_ENABLE_BUG_1666637     20:0x80   // Bug 1666637
#define LW0080_CTRL_GR_CAPS_INSTANCED_DRAW_PERF_BUG_1722438       21:0x01   // Bug 1722438
#define LW0080_CTRL_GR_CAPS_PLA_BUG_1746518                       21:0x02   // Bug 1746518
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#define LW0080_CTRL_GR_CAPS_TC_ILWALIDATE_BUG_200264501           21:0x04   // Deprecated
#define LW0080_CTRL_GR_CAPS_TITAN                                 21:0x08   // Titan detection
/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


#define LW0080_CTRL_GR_CAPS_L2_ADDRESSED_ILWALIDATE_BUG_2040262   21:0x10   // Bug 2040262
#define LW0080_CTRL_GR_CAPS_SM_TWOD_FIFO_STALL_BUG_2187760        21:0x20   // Bug 2187760
#define LW0080_CTRL_GR_CAPS_ZBC_MASK_BUG_2920456                  21:0x40   // Bug 2920456
#define LW0080_CTRL_GR_CAPS_CONSTANT_UPDATE_BUG_2548370           21:0x80   // Bug 2548370
#define LW0080_CTRL_GR_CAPS_PARTIAL_GCC_ILWALIDATE_BUG_200648621  22:0x01   // Bug 200648621
#define LW0080_CTRL_GR_CAPS_SMSCG_HANG_BUG_2464528                22:0x02   // Bug 2464528
#define LW0080_CTRL_GR_CAPS_MPC_DEADLOCK_BUG_3123003              22:0x04   // Bug 3123003
#define LW0080_CTRL_GR_CAPS_PE_CACHE_ILWALIDATE_BUG_3333882       22:0x08   // Bug 3333882

/*
 * Size in bytes of gr caps table.  This value should be one greater
 * than the largest byte_index value above.
 */
#define LW0080_CTRL_GR_CAPS_TBL_SIZE            23

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW0080_CTRL_CMD_GR_SET_CONTEXT_OVERRIDE
 *
 * This command allows the client to override the GR context
 * switched register value which is loaded at channel switch
 * time.
 *
 *   regAddr;
 *     This field specifies register to override in gr context
 *   andMask
 *     This field is negated and ANDed to the original register value
 *   orMask
 *     This field is ORed to the original value
 *   grRouteInfo
 *     This parameter specifies the routing information used to
 *     disambiguate the target GR engine. When MIG is enabled, this
 *     is a mandatory parameter
 */
#define LW0080_CTRL_CMD_GR_SET_CONTEXT_OVERRIDE (0x801103) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GR_INTERFACE_ID << 8) | 0x3" */

typedef struct LW0080_CTRL_CMD_GR_SET_CONTEXT_OVERRIDE_PARAMS {
    LwU32 regAddr;
    LwU32 andMask;
    LwU32 orMask;
    LW_DECLARE_ALIGNED(LW0080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW0080_CTRL_CMD_GR_SET_CONTEXT_OVERRIDE_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW0080_CTRL_CMD_GR_INFO
 *
 * This structure represents a single 32bit graphics engine value.  Clients
 * request a particular graphics engine value by specifying a unique bus
 * information index.
 *
 * Legal graphics information index values are:
 *   LW0080_CTRL_GR_INFO_INDEX_MAXCLIPS
 *     This index is used to request the number of clip IDs supported by
 *     the device.
 *   LW0080_CTRL_GR_INFO_INDEX_MIN_ATTRS_BUG_261894
 *     This index is used to request the minimum number of attributes that
 *     need to be enabled to avoid bug 261894.  A return value of 0
 *     indicates that there is no minimum and the bug is not present on this
 *     system.
 */
typedef struct LW0080_CTRL_GR_INFO {
    LwU32 index;
    LwU32 data;
} LW0080_CTRL_GR_INFO;

/* valid graphics info index values */
#define LW0080_CTRL_GR_INFO_INDEX_MAXCLIPS                          (0x00000000)
#define LW0080_CTRL_GR_INFO_INDEX_MIN_ATTRS_BUG_261894              (0x00000001)
#define LW0080_CTRL_GR_INFO_XBUF_MAX_PSETS_PER_BANK                 (0x00000002)
#define LW0080_CTRL_GR_INFO_INDEX_BUFFER_ALIGNMENT                  (0x00000003)
#define LW0080_CTRL_GR_INFO_INDEX_SWIZZLE_ALIGNMENT                 (0x00000004)
#define LW0080_CTRL_GR_INFO_INDEX_VERTEX_CACHE_SIZE                 (0x00000005)
#define LW0080_CTRL_GR_INFO_INDEX_VPE_COUNT                         (0x00000006)
#define LW0080_CTRL_GR_INFO_INDEX_SHADER_PIPE_COUNT                 (0x00000007)
#define LW0080_CTRL_GR_INFO_INDEX_THREAD_STACK_SCALING_FACTOR       (0x00000008)
#define LW0080_CTRL_GR_INFO_INDEX_SHADER_PIPE_SUB_COUNT             (0x00000009)
#define LW0080_CTRL_GR_INFO_INDEX_SM_REG_BANK_COUNT                 (0x0000000A)
#define LW0080_CTRL_GR_INFO_INDEX_SM_REG_BANK_REG_COUNT             (0x0000000B)
#define LW0080_CTRL_GR_INFO_INDEX_SM_VERSION                        (0x0000000C)
#define LW0080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM                  (0x0000000D)
#define LW0080_CTRL_GR_INFO_INDEX_MAX_THREADS_PER_WARP              (0x0000000E)
#define LW0080_CTRL_GR_INFO_INDEX_GEOM_GS_OBUF_ENTRIES              (0x0000000F)
#define LW0080_CTRL_GR_INFO_INDEX_GEOM_XBUF_ENTRIES                 (0x00000010)
#define LW0080_CTRL_GR_INFO_INDEX_FB_MEMORY_REQUEST_GRANULARITY     (0x00000011)
#define LW0080_CTRL_GR_INFO_INDEX_HOST_MEMORY_REQUEST_GRANULARITY   (0x00000012)
#define LW0080_CTRL_GR_INFO_INDEX_MAX_SP_PER_SM                     (0x00000013)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS                   (0x00000014)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_FBPS                   (0x00000015)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_ZLWLL_BANKS            (0x00000016)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC            (0x00000017)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_MIN_FBPS               (0x00000018)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_MXBAR_FBP_PORTS        (0x00000019)
#define LW0080_CTRL_GR_INFO_INDEX_TIMESLICE_ENABLED                 (0x0000001A)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_FBPAS                  (0x0000001B)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_PES_PER_GPC            (0x0000001C)
#define LW0080_CTRL_GR_INFO_INDEX_GPU_CORE_COUNT                    (0x0000001D)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPCS_PER_PES           (0x0000001E)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_MXBAR_HUB_PORTS        (0x0000001F)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC             (0x00000020)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_HSHUB_FBP_PORTS        (0x00000021)
#define LW0080_CTRL_GR_INFO_INDEX_RT_CORE_COUNT                     (0x00000022)
#define LW0080_CTRL_GR_INFO_INDEX_TENSOR_CORE_COUNT                 (0x00000023)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_GRS                    (0x00000024)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_LTCS                   (0x00000025)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_LTC_SLICES             (0x00000026)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCMMU_PER_GPC         (0x00000027)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_LTC_PER_FBP            (0x00000028)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_ROP_PER_GPC            (0x00000029)
#define LW0080_CTRL_GR_INFO_INDEX_FAMILY_MAX_TPC_PER_GPC            (0x0000002A)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_FBPA_PER_FBP           (0x0000002B)
#define LW0080_CTRL_GR_INFO_INDEX_MAX_SUBCONTEXT_COUNT              (0x0000002C)
#define LW0080_CTRL_GR_INFO_INDEX_MAX_LEGACY_SUBCONTEXT_COUNT       (0x0000002D)
#define LW0080_CTRL_GR_INFO_INDEX_MAX_PER_ENGINE_SUBCONTEXT_COUNT   (0x0000002E)
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_SINGLETON_GPCS         (0x0000002F)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_GFXC_GPCS              (0x00000030)
#define LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_GFXC_TPCS_PER_GFXC_GPC (0x00000031)
/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/* When adding a new INDEX, please update MAX_SIZE accordingly
 * NOTE: 0080 functionality is merged with 2080 functionality, so this max size
 * reflects that.
 */
#define LW0080_CTRL_GR_INFO_INDEX_MAX                               (0x00000031)
#define LW0080_CTRL_GR_INFO_MAX_SIZE                                (0x32) /* finn: Evaluated from "(LW0080_CTRL_GR_INFO_INDEX_MAX + 1)" */

/*
 * LW0080_CTRL_CMD_GR_GET_INFO
 *
 * This command returns graphics engine information for the associate GPU.
 * Request to retrieve graphics information use a list of one or more
 * LW0080_CTRL_GR_INFO structures.
 *
 *   grInfoListSize
 *     This field specifies the number of entries on the caller's
 *     grInfoList.
 *   grInfoList
 *     This field specifies a pointer in the caller's address space
 *     to the buffer into which the bus information is to be returned.
 *     This buffer must be at least as big as grInfoListSize multiplied
 *     by the size of the LW0080_CTRL_GR_INFO structure.
 */
#define LW0080_CTRL_CMD_GR_GET_INFO                                 (0x801104) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GR_INTERFACE_ID << 8) | LW0080_CTRL_GR_GET_INFO_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GR_GET_INFO_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW0080_CTRL_GR_GET_INFO_PARAMS {
    LwU32 grInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 grInfoList, 8);
} LW0080_CTRL_GR_GET_INFO_PARAMS;

/*
 * LW0080_CTRL_CMD_GR_GET_TPC_PARTITION_MODE
 *     This command gets the current partition mode of a TSG context.
 *
 * LW0080_CTRL_CMD_GR_SET_TPC_PARTITION_MODE
 *     This command sets the partition mode of a TSG context.
 *
 * LW0080_CTRL_GR_TPC_PARTITION_MODE_PARAMS
 *     This structure defines the parameters used for TPC partitioning mode SET/GET commands
 *
 *     hChannelGroup [IN]
 *         RM Handle to the TSG
 *
 *     mode [IN/OUT]
 *         Partitioning mode enum value
 *             For the SET cmd, this is an input parameter
 *             For the GET cmd, this is an output parameter
 *
 *     bEnableAllTpcs [IN]
 *         Flag to enable all TPCs by default
 *
 *     grRouteInfo[IN]
 *         This parameter specifies the routing information used to
 *         disambiguate the target GR engine.
 *
 */
#define LW0080_CTRL_CMD_GR_GET_TPC_PARTITION_MODE (0x801107) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GR_INTERFACE_ID << 8) | 0x7" */

#define LW0080_CTRL_CMD_GR_SET_TPC_PARTITION_MODE (0x801108) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GR_INTERFACE_ID << 8) | 0x8" */

/* Enum for listing TPC partitioning modes */
typedef enum LW0080_CTRL_GR_TPC_PARTITION_MODE {
    LW0080_CTRL_GR_TPC_PARTITION_MODE_NONE = 0,
    LW0080_CTRL_GR_TPC_PARTITION_MODE_STATIC = 1,
    LW0080_CTRL_GR_TPC_PARTITION_MODE_DYNAMIC = 2,
} LW0080_CTRL_GR_TPC_PARTITION_MODE;

typedef struct LW0080_CTRL_GR_TPC_PARTITION_MODE_PARAMS {
    LwHandle                          hChannelGroup;   // [in]
    LW0080_CTRL_GR_TPC_PARTITION_MODE mode;            // [in/out]
    LwBool                            bEnableAllTpcs;  // [in/out]
    LW_DECLARE_ALIGNED(LW0080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);     // [in]
} LW0080_CTRL_GR_TPC_PARTITION_MODE_PARAMS;

/**
 * LW0080_CTRL_CMD_GR_GET_CAPS_V2
 *
 * This command returns the same set of graphics capabilities for the device
 * as @ref LW0080_CTRL_CMD_GR_GET_CAPS. The difference is in the structure
 * LW0080_CTRL_GR_GET_INFO_V2_PARAMS, which contains a statically sized array,
 * rather than a caps table pointer and a caps table size in
 * LW0080_CTRL_GR_GET_INFO_PARAMS. Additionally,
 * LW0080_CTRL_GR_GET_INFO_V2_PARAMS contains a parameter for specifying routing
 * information, used for MIG.
 *
 *   capsTbl
 *     This parameter specifies a pointer to the client's caps table buffer
 *     into which the graphics caps bits will be written by the RM.
 *     The caps table is an array of unsigned bytes.
 *
 *   grRouteInfo
 *     This parameter specifies the routing information used to
 *     disambiguate the target GR engine.
 *
 *   bCapsPopulated
 *     This parameter indicates that the capsTbl has been partially populated by
 *     previous calls to LW0080_CTRL_CMD_GR_GET_CAPS_V2 on other subdevices.
 */
#define LW0080_CTRL_CMD_GR_GET_CAPS_V2 (0x801109) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GR_INTERFACE_ID << 8) | LW0080_CTRL_GR_GET_CAPS_V2_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GR_GET_CAPS_V2_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW0080_CTRL_GR_GET_CAPS_V2_PARAMS {
    LwU8   capsTbl[LW0080_CTRL_GR_CAPS_TBL_SIZE];
    LW_DECLARE_ALIGNED(LW0080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
    LwBool bCapsPopulated;
} LW0080_CTRL_GR_GET_CAPS_V2_PARAMS;

#define LW0080_CTRL_CMD_GR_GET_INFO_V2 (0x801110) /* finn: Evaluated from "(FINN_LW01_DEVICE_0_GR_INTERFACE_ID << 8) | LW0080_CTRL_GR_GET_INFO_V2_PARAMS_MESSAGE_ID" */

#define LW0080_CTRL_GR_GET_INFO_V2_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW0080_CTRL_GR_GET_INFO_V2_PARAMS {
    LwU32               grInfoListSize;
    LW0080_CTRL_GR_INFO grInfoList[LW0080_CTRL_GR_INFO_MAX_SIZE];
    LW_DECLARE_ALIGNED(LW0080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW0080_CTRL_GR_GET_INFO_V2_PARAMS;

/* _ctrl0080gr_h_ */
