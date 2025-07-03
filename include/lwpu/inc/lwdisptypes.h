/*
 * SPDX-FileCopyrightText: Copyright (c) 2014-2016 LWPU CORPORATION & AFFILIATES. All rights reserved.
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

 /***************************************************************************\
|*                                                                           *|
|*                        LW Display Common Types                            *|
|*                                                                           *|
|*  <lwdisptypes.h>  defines the common display types.                       *|
|*                                                                           *|
 \***************************************************************************/

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: lwdisptypes.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "lwtypes.h"

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


typedef enum LW_PIXEL_FORMAT {
    LW_PIXEL_FORMAT_I8 = 0,
    LW_PIXEL_FORMAT_VOID16 = 1,
    LW_PIXEL_FORMAT_VOID32 = 2,
    LW_PIXEL_FORMAT_RF16_GF16_BF16_AF16 = 3,
    LW_PIXEL_FORMAT_A8_R8_G8_B8 = 4,
    LW_PIXEL_FORMAT_A2_B10_G10_R10 = 5,
    LW_PIXEL_FORMAT_A8_B8_G8_R8 = 6,
    LW_PIXEL_FORMAT_R5_G6_B5 = 7,
    LW_PIXEL_FORMAT_A1_R5_G5_B5 = 8,
    LW_PIXEL_FORMAT_X2BL10GL10RL10_XRBIAS = 9,
    LW_PIXEL_FORMAT_R16_G16_B16_A16 = 10,
    LW_PIXEL_FORMAT_R16_G16_B16_A16_LWBIAS = 11,
    LW_PIXEL_FORMAT_X2BL10GL10RL10_XVYCC = 12,
} LW_PIXEL_FORMAT;

typedef enum LW_RASTER_STRUCTURE {
    LW_RASTER_STRUCTURE_PROGRESSIVE = 0,
    LW_RASTER_STRUCTURE_INTERLACED = 1,
} LW_RASTER_STRUCTURE;

typedef enum LW_VERTICAL_TAPS {
    LW_VERTICAL_TAPS_1 = 0,
    LW_VERTICAL_TAPS_2 = 1,
    LW_VERTICAL_TAPS_3 = 2,
    LW_VERTICAL_TAPS_ADAPTIVE_3 = 3,
    LW_VERTICAL_TAPS_5 = 4,
} LW_VERTICAL_TAPS;

typedef enum LW_HORIZONTAL_TAPS {
    LW_HORIZONTAL_TAPS_1 = 0,
    LW_HORIZONTAL_TAPS_2 = 1,
    LW_HORIZONTAL_TAPS_8 = 2,
} LW_HORIZONTAL_TAPS;


typedef enum LW_OUTPUT_SCALER_FORCE422_MODE {
    LW_OUTPUT_SCALER_FORCE422_MODE_DISABLE = 0,
    LW_OUTPUT_SCALER_FORCE422_MODE_ENABLE = 1,
} LW_OUTPUT_SCALER_FORCE422_MODE;

typedef enum LW_PIXEL_DEPTH {
    LW_PIXEL_DEPTH_8 = 0,
    LW_PIXEL_DEPTH_16 = 1,
    LW_PIXEL_DEPTH_32 = 2,
    LW_PIXEL_DEPTH_64 = 3,
} LW_PIXEL_DEPTH;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



typedef enum LW_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP {
    LW_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_DEFAULT = 0,
    LW_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_16_422 = 1,
    LW_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_18_444 = 2,
    LW_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_20_422 = 3,
    LW_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_24_422 = 4,
    LW_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_24_444 = 5,
    LW_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_30_444 = 6,
    LW_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_32_422 = 7,
    LW_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_36_444 = 8,
    LW_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP_48_444 = 9,
} LW_OUTPUT_RESOURCE_PIXEL_DEPTH_BPP;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


typedef enum LW_PIXEL_REPLICATE_MODE {
    LW_PIXEL_REPLICATE_MODE_OFF = 0,
    LW_PIXEL_REPLICATE_MODE_X2 = 1,
    LW_PIXEL_REPLICATE_MODEP_X4 = 2,
} LW_PIXEL_REPLICATE_MODE;

typedef enum LW_MEMORY_LAYOUT {
    LW_MEMORY_LAYOUT_PITCH = 0,
    LW_MEMORY_LAYOUT_BLOCK_LINEAR = 1,
    LW_MEMORY_LAYOUT_SUPPORTED = 2,
} LW_MEMORY_LAYOUT;

typedef enum LW_SUPER_SAMPLE {
    LW_SUPER_SAMPLE_X1AA = 0,
    LW_SUPER_SAMPLE_X4AA = 1,
} LW_SUPER_SAMPLE;

typedef enum LW_OR_OWNER {
    LW_OR_OWNER_HEAD0 = 0,
    LW_OR_OWNER_HEAD1 = 1,
    LW_OR_OWNER_HEAD2 = 2,
    LW_OR_OWNER_HEAD3 = 3,
    orOwner_None = 2147483647,
} LW_OR_OWNER;
#define LW_OR_OWNER_HEAD(i) (LW_OR_OWNER_HEAD0 + i)

typedef enum LW_DAC_PROTOCOL {
    LW_DAC_PROTOCOL_RGB_CRT = 0,
} LW_DAC_PROTOCOL;

typedef enum LW_SOR_PROTOCOL {
    LW_SOR_PROTOCOL_SINGLE_TMDS_A = 0,
    LW_SOR_PROTOCOL_SINGLE_TMDS_B = 1,
    LW_SOR_PROTOCOL_DUAL_TMDS = 2,
    LW_SOR_PROTOCOL_LVDS_LWSTOM = 3,
    LW_SOR_PROTOCOL_DP_A = 4,
    LW_SOR_PROTOCOL_DP_B = 5,
    LW_SOR_PROTOCOL_SUPPORTED = 6,
} LW_SOR_PROTOCOL;

typedef enum LW_WBOR_PROTOCOL {
    LW_WBOR_PROTOCOL_WRBK = 0,
} LW_WBOR_PROTOCOL;

typedef enum LW_PIOR_PROTOCOL {
    LW_PIOR_PROTOCOL_EXT_TMDS_ENC = 0,
    LW_PIOR_PROTOCOL_EXT_SDI_SD_ENC = 2,
    LW_PIOR_PROTOCOL_EXT_SDI_HD_ENC = 3,
    LW_PIOR_PROTOCOL_DIST_RENDER_OUT = 4,
    LW_PIOR_PROTOCOL_DIST_RENDER_IN = 5,
    LW_PIOR_PROTOCOL_DIST_RENDER_INOUT = 6,
} LW_PIOR_PROTOCOL;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



typedef LwU32 LW_DISP_LOCK_PIN;

#define LW_DISP_LOCK_PIN_0           0x0
#define LW_DISP_LOCK_PIN_1           0x1
#define LW_DISP_LOCK_PIN_2           0x2
#define LW_DISP_LOCK_PIN_3           0x3
#define LW_DISP_LOCK_PIN_4           0x4
#define LW_DISP_LOCK_PIN_5           0x5
#define LW_DISP_LOCK_PIN_6           0x6
#define LW_DISP_LOCK_PIN_7           0x7
#define LW_DISP_LOCK_PIN_8           0x8
#define LW_DISP_LOCK_PIN_9           0x9
#define LW_DISP_LOCK_PIN_A           0xA
#define LW_DISP_LOCK_PIN_B           0xB
#define LW_DISP_LOCK_PIN_C           0xC
#define LW_DISP_LOCK_PIN_D           0xD
#define LW_DISP_LOCK_PIN_E           0xE
#define LW_DISP_LOCK_PIN_F           0xF

// Value used solely for HW initialization
#define LW_DISP_LOCK_PIN_UNSPECIFIED 0x10

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

//
// Specify an internal scan lock or stereo pin.
// In the case of a SlaveLockPin, this is which head's output to listen to.
// In the case of a MasterLockPin, this indicates the head's internal lock
// signal output.
// In the case of a StereoPin, this indicates the head's internal stereo
// signal output.
//
#define LW_DISP_INTERNAL_SCAN_LOCK_0 0x18
#define LW_DISP_INTERNAL_SCAN_LOCK_1 0x19
#define LW_DISP_INTERNAL_SCAN_LOCK_2 0x1A
#define LW_DISP_INTERNAL_SCAN_LOCK_3 0x1B

//
// Select which internal flip lock pin to use
// With up to three heads, only one internal fliplock pin is ever needed.
// In a four head system however,
// Two internal fliplock pins would be required for full flexibility.
//
#define LW_DISP_INTERNAL_FLIP_LOCK_0 0x1E
#define LW_DISP_INTERNAL_FLIP_LOCK_1 0x1F

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



typedef LwU32 LW_DISP_LOCK_MODE;

#define LW_DISP_LOCK_MODE_NO_LOCK     0x0
#define LW_DISP_LOCK_MODE_FRAME_LOCK  0x1
#define LW_DISP_LOCK_MODE_RASTER_LOCK 0x3

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


typedef enum LW_LUT_USAGE {
    LW_LUT_USAGE_NONE = 0,
    LW_LUT_USAGE_257 = 1,
    LW_LUT_USAGE_1025 = 2,
} LW_LUT_USAGE;

typedef enum LW_LUT_LO_MODE {
    LW_LUT_LO_MODE_LORES = 0,
    LW_LUT_LO_MODE_HIRES = 1,
    LW_LUT_LO_MODE_INDEX_1025_UNITY_RANGE = 2,
    LW_LUT_LO_MODE_INTERPOLATE_1025_UNITY_RANGE = 3,
    LW_LUT_LO_MODE_INTERPOLATE_1025_XRBIAS_RANGE = 4,
    LW_LUT_LO_MODE_INTERPOLATE_1025_XVYCC_RANGE = 5,
    LW_LUT_LO_MODE_INTERPOLATE_257_UNITY_RANGE = 6,
    LW_LUT_LO_MODE_INTERPOLATE_257_LEGACY_RANGE = 7,
} LW_LUT_LO_MODE;

typedef enum LW_LUT_LO {
    LW_LUT_LO_DISABLE = 0,
    LW_LUT_LO_ENABLE = 1,
} LW_LUT_LO;

// LwDisplay IMP usage bound formats
#define LW_IMP_FORMAT_RGB_PACKED_1_BPP         (0x00000001)
#define LW_IMP_FORMAT_RGB_PACKED_2_BPP         (0x00000002)
#define LW_IMP_FORMAT_RGB_PACKED_4_BPP         (0x00000004)
#define LW_IMP_FORMAT_RGB_PACKED_8_BPP         (0x00000008)
#define LW_IMP_FORMAT_YUV_PACKED_422           (0x00000010)
#define LW_IMP_FORMAT_YUV_PLANAR_420           (0x00000020)
#define LW_IMP_FORMAT_YUV_PLANAR_444           (0x00000040)
#define LW_IMP_FORMAT_YUV_SEMI_PLANAR_420      (0x00000080)
#define LW_IMP_FORMAT_YUV_SEMI_PLANAR_422      (0x00000100)
#define LW_IMP_FORMAT_YUV_SEMI_PLANAR_422R     (0x00000200)
#define LW_IMP_FORMAT_YUV_SEMI_PLANAR_444      (0x00000400)
#define LW_IMP_FORMAT_EXT_YUV_PLANAR_420       (0x00000800)
#define LW_IMP_FORMAT_EXT_YUV_PLANAR_444       (0x00001000)
#define LW_IMP_FORMAT_EXT_YUV_SEMI_PLANAR_420  (0x00002000)
#define LW_IMP_FORMAT_EXT_YUV_SEMI_PLANAR_422  (0x00004000)
#define LW_IMP_FORMAT_EXT_YUV_SEMI_PLANAR_422R (0x00008000)
#define LW_IMP_FORMAT_EXT_YUV_SEMI_PLANAR_444  (0x00010000)
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

