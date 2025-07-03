/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */
#ifndef _cl_e2b7_h_
#define _cl_e2b7_h_

#include "lwtypes.h"
#define LWE2_MPE                   (0xE2B7)

// ############# Internal packet definitions between MPEA/B/C ####################
//********************************************************
// Packet definition of interface between MPEA and MPEB
//********************************************************

// Packet MPEA2B_LWRR
#define MPEA2B_LWRR_SIZE 128

//16 pixels at 8 bits each
#define MPEA2B_LWRR_DATA_SHIFT                  (0)
#define MPEA2B_LWRR_DATA_FIELD                  ((0xffffffff) << MPEA2B_LWRR_DATA_SHIFT)
#define MPEA2B_LWRR_DATA_RANGE                  (127):(0)
#define MPEA2B_LWRR_DATA_ROW                    0


// Packet MPEA2B_PRED_FWD
#define MPEA2B_PRED_FWD_SIZE 64

//8 pixels at 8 bits each
#define MPEA2B_PRED_FWD_DATA_SHIFT                      (0)
#define MPEA2B_PRED_FWD_DATA_FIELD                      ((0xffffffff) << MPEA2B_PRED_FWD_DATA_SHIFT)
#define MPEA2B_PRED_FWD_DATA_RANGE                      (63):(0)
#define MPEA2B_PRED_FWD_DATA_ROW                        0


// Packet MPEA2B_PRED_RCN
#define MPEA2B_PRED_RCN_SIZE 64

//8 pixels at 8 bits each
#define MPEA2B_PRED_RCN_DATA_SHIFT                      (0)
#define MPEA2B_PRED_RCN_DATA_FIELD                      ((0xffffffff) << MPEA2B_PRED_RCN_DATA_SHIFT)
#define MPEA2B_PRED_RCN_DATA_RANGE                      (63):(0)
#define MPEA2B_PRED_RCN_DATA_ROW                        0

////////////////////////////////
///  MPEG4/H264 BYPASS MODE (A)
///
///   +-----+-----+
///   | A0  | A1  |
///   +-----+-----+
///   | A2  | A3  |
///   +-----+-----+
///
///  H264 NEW MODE THAT INCLUDES SUB8x8 MODES (B)
///
///  +-----+-----+-----+-----+
///  | B0  | B1  | B4  | B5  |
///  +-----+-----+-----+-----+
///  | B2  | B3  | B6  | B7  |
///  +-----+-----+-----+-----+
///  | B8  | B9  | B12 | B13 |
///  +-----+-----+-----+-----+
///  | B10 | B11 | B14 | B15 |
///  +-----+-----+-----+-----+

// Packet MPEA2B_MB_START_PARAM
#define MPEA2B_MB_START_PARAM_SIZE 514

//1bit,h264
#define MPEA2B_MB_START_PARAM_LEFTMB_NB_MAP_SHIFT                       (0)
#define MPEA2B_MB_START_PARAM_LEFTMB_NB_MAP_FIELD                       ((0x1) << MPEA2B_MB_START_PARAM_LEFTMB_NB_MAP_SHIFT)
#define MPEA2B_MB_START_PARAM_LEFTMB_NB_MAP_RANGE                       (0):(0)
#define MPEA2B_MB_START_PARAM_LEFTMB_NB_MAP_ROW                 0
#define MPEA2B_MB_START_PARAM_LEFTMB_NB_MAP_NOT_AVAILABLE                       (0)
#define MPEA2B_MB_START_PARAM_LEFTMB_NB_MAP_AVAILABLE                   (1)

//1bit,h264
#define MPEA2B_MB_START_PARAM_TOPLEFTMB_NB_MAP_SHIFT                    (1)
#define MPEA2B_MB_START_PARAM_TOPLEFTMB_NB_MAP_FIELD                    ((0x1) << MPEA2B_MB_START_PARAM_TOPLEFTMB_NB_MAP_SHIFT)
#define MPEA2B_MB_START_PARAM_TOPLEFTMB_NB_MAP_RANGE                    (1):(1)
#define MPEA2B_MB_START_PARAM_TOPLEFTMB_NB_MAP_ROW                      0
#define MPEA2B_MB_START_PARAM_TOPLEFTMB_NB_MAP_NOT_AVAILABLE                    (0)
#define MPEA2B_MB_START_PARAM_TOPLEFTMB_NB_MAP_AVAILABLE                        (1)

//1bit,h264
#define MPEA2B_MB_START_PARAM_TOPMB_NB_MAP_SHIFT                        (2)
#define MPEA2B_MB_START_PARAM_TOPMB_NB_MAP_FIELD                        ((0x1) << MPEA2B_MB_START_PARAM_TOPMB_NB_MAP_SHIFT)
#define MPEA2B_MB_START_PARAM_TOPMB_NB_MAP_RANGE                        (2):(2)
#define MPEA2B_MB_START_PARAM_TOPMB_NB_MAP_ROW                  0
#define MPEA2B_MB_START_PARAM_TOPMB_NB_MAP_NOT_AVAILABLE                        (0)
#define MPEA2B_MB_START_PARAM_TOPMB_NB_MAP_AVAILABLE                    (1)

//1bit,h264
#define MPEA2B_MB_START_PARAM_TOPRIGHT_NB_MAP_SHIFT                     (3)
#define MPEA2B_MB_START_PARAM_TOPRIGHT_NB_MAP_FIELD                     ((0x1) << MPEA2B_MB_START_PARAM_TOPRIGHT_NB_MAP_SHIFT)
#define MPEA2B_MB_START_PARAM_TOPRIGHT_NB_MAP_RANGE                     (3):(3)
#define MPEA2B_MB_START_PARAM_TOPRIGHT_NB_MAP_ROW                       0
#define MPEA2B_MB_START_PARAM_TOPRIGHT_NB_MAP_NOT_AVAILABLE                     (0)
#define MPEA2B_MB_START_PARAM_TOPRIGHT_NB_MAP_AVAILABLE                 (1)

//3bit,h264 
#define MPEA2B_MB_START_PARAM_SLICE_GROUPID_SHIFT                       (4)
#define MPEA2B_MB_START_PARAM_SLICE_GROUPID_FIELD                       ((0x7) << MPEA2B_MB_START_PARAM_SLICE_GROUPID_SHIFT)
#define MPEA2B_MB_START_PARAM_SLICE_GROUPID_RANGE                       (6):(4)
#define MPEA2B_MB_START_PARAM_SLICE_GROUPID_ROW                 0

//1bit,h264
#define MPEA2B_MB_START_PARAM_FIRSTMBINSLICE_SHIFT                      (7)
#define MPEA2B_MB_START_PARAM_FIRSTMBINSLICE_FIELD                      ((0x1) << MPEA2B_MB_START_PARAM_FIRSTMBINSLICE_SHIFT)
#define MPEA2B_MB_START_PARAM_FIRSTMBINSLICE_RANGE                      (7):(7)
#define MPEA2B_MB_START_PARAM_FIRSTMBINSLICE_ROW                        0

//2bit,h264
#define MPEA2B_MB_START_PARAM_REFID_0_SHIFT                     (8)
#define MPEA2B_MB_START_PARAM_REFID_0_FIELD                     ((0x3) << MPEA2B_MB_START_PARAM_REFID_0_SHIFT)
#define MPEA2B_MB_START_PARAM_REFID_0_RANGE                     (9):(8)
#define MPEA2B_MB_START_PARAM_REFID_0_ROW                       0
#define MPEA2B_MB_START_PARAM_REFID_0_N_MINUS_1                 (0)
#define MPEA2B_MB_START_PARAM_REFID_0_N_MINUS_2                 (1)
#define MPEA2B_MB_START_PARAM_REFID_0_LONGTERM                  (2)

//2bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_REFID_1_SHIFT                     (10)
#define MPEA2B_MB_START_PARAM_REFID_1_FIELD                     ((0x3) << MPEA2B_MB_START_PARAM_REFID_1_SHIFT)
#define MPEA2B_MB_START_PARAM_REFID_1_RANGE                     (11):(10)
#define MPEA2B_MB_START_PARAM_REFID_1_ROW                       0
#define MPEA2B_MB_START_PARAM_REFID_1_N_MINUS_1                 (0)
#define MPEA2B_MB_START_PARAM_REFID_1_N_MINUS_2                 (1)
#define MPEA2B_MB_START_PARAM_REFID_1_LONGTERM                  (2)

//2bit,h264/mpeg4 
#define MPEA2B_MB_START_PARAM_REFID_2_SHIFT                     (12)
#define MPEA2B_MB_START_PARAM_REFID_2_FIELD                     ((0x3) << MPEA2B_MB_START_PARAM_REFID_2_SHIFT)
#define MPEA2B_MB_START_PARAM_REFID_2_RANGE                     (13):(12)
#define MPEA2B_MB_START_PARAM_REFID_2_ROW                       0
#define MPEA2B_MB_START_PARAM_REFID_2_N_MINUS_1                 (0)
#define MPEA2B_MB_START_PARAM_REFID_2_N_MINUS_2                 (1)
#define MPEA2B_MB_START_PARAM_REFID_2_LONGTERM                  (2)

//2bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_REFID_3_SHIFT                     (14)
#define MPEA2B_MB_START_PARAM_REFID_3_FIELD                     ((0x3) << MPEA2B_MB_START_PARAM_REFID_3_SHIFT)
#define MPEA2B_MB_START_PARAM_REFID_3_RANGE                     (15):(14)
#define MPEA2B_MB_START_PARAM_REFID_3_ROW                       0
#define MPEA2B_MB_START_PARAM_REFID_3_N_MINUS_1                 (0)
#define MPEA2B_MB_START_PARAM_REFID_3_N_MINUS_2                 (1)
#define MPEA2B_MB_START_PARAM_REFID_3_LONGTERM                  (2)

//2bit. Increase to 2 bits to support B-slice
#define MPEA2B_MB_START_PARAM_SLICETYPE_SHIFT                   (16)
#define MPEA2B_MB_START_PARAM_SLICETYPE_FIELD                   ((0x3) << MPEA2B_MB_START_PARAM_SLICETYPE_SHIFT)
#define MPEA2B_MB_START_PARAM_SLICETYPE_RANGE                   (17):(16)
#define MPEA2B_MB_START_PARAM_SLICETYPE_ROW                     0
#define MPEA2B_MB_START_PARAM_SLICETYPE_IType                   (0)
#define MPEA2B_MB_START_PARAM_SLICETYPE_PType                   (1)

//1bit,h264
#define MPEA2B_MB_START_PARAM_LASTMBINSLICEGRP_SHIFT                    (18)
#define MPEA2B_MB_START_PARAM_LASTMBINSLICEGRP_FIELD                    ((0x1) << MPEA2B_MB_START_PARAM_LASTMBINSLICEGRP_SHIFT)
#define MPEA2B_MB_START_PARAM_LASTMBINSLICEGRP_RANGE                    (18):(18)
#define MPEA2B_MB_START_PARAM_LASTMBINSLICEGRP_ROW                      0

//2bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_MVMODE_SHIFT                      (19)
#define MPEA2B_MB_START_PARAM_MVMODE_FIELD                      ((0x3) << MPEA2B_MB_START_PARAM_MVMODE_SHIFT)
#define MPEA2B_MB_START_PARAM_MVMODE_RANGE                      (20):(19)
#define MPEA2B_MB_START_PARAM_MVMODE_ROW                        0
#define MPEA2B_MB_START_PARAM_MVMODE_m16x16                     (0)
#define MPEA2B_MB_START_PARAM_MVMODE_m8x8                       (1)
#define MPEA2B_MB_START_PARAM_MVMODE_m16x8                      (2)
#define MPEA2B_MB_START_PARAM_MVMODE_m8x16                      (3)

//8bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_MBY_SHIFT                 (21)
#define MPEA2B_MB_START_PARAM_MBY_FIELD                 ((0xff) << MPEA2B_MB_START_PARAM_MBY_SHIFT)
#define MPEA2B_MB_START_PARAM_MBY_RANGE                 (28):(21)
#define MPEA2B_MB_START_PARAM_MBY_ROW                   0

//8bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_MBX_SHIFT                 (29)
#define MPEA2B_MB_START_PARAM_MBX_FIELD                 ((0xff) << MPEA2B_MB_START_PARAM_MBX_SHIFT)
#define MPEA2B_MB_START_PARAM_MBX_RANGE                 (36):(29)
#define MPEA2B_MB_START_PARAM_MBX_ROW                   0

//2bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_MBTYPE_SHIFT                      (37)
#define MPEA2B_MB_START_PARAM_MBTYPE_FIELD                      ((0x3) << MPEA2B_MB_START_PARAM_MBTYPE_SHIFT)
#define MPEA2B_MB_START_PARAM_MBTYPE_RANGE                      (38):(37)
#define MPEA2B_MB_START_PARAM_MBTYPE_ROW                        0
#define MPEA2B_MB_START_PARAM_MBTYPE_I                  (0)
#define MPEA2B_MB_START_PARAM_MBTYPE_P                  (1)
#define MPEA2B_MB_START_PARAM_MBTYPE_IPCM                       (2)
#define MPEA2B_MB_START_PARAM_MBTYPE_SKIPSUG                    (3)

//17bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_MSC_MB_COST_SHIFT                 (39)
#define MPEA2B_MB_START_PARAM_MSC_MB_COST_FIELD                 ((0x1ffff) << MPEA2B_MB_START_PARAM_MSC_MB_COST_SHIFT)
#define MPEA2B_MB_START_PARAM_MSC_MB_COST_RANGE                 (55):(39)
#define MPEA2B_MB_START_PARAM_MSC_MB_COST_ROW                   0

//10bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_MVY_0_SHIFT                       (56)
#define MPEA2B_MB_START_PARAM_MVY_0_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVY_0_SHIFT)
#define MPEA2B_MB_START_PARAM_MVY_0_RANGE                       (65):(56)
#define MPEA2B_MB_START_PARAM_MVY_0_ROW                 0

//10bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_MVY_1_SHIFT                       (66)
#define MPEA2B_MB_START_PARAM_MVY_1_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVY_1_SHIFT)
#define MPEA2B_MB_START_PARAM_MVY_1_RANGE                       (75):(66)
#define MPEA2B_MB_START_PARAM_MVY_1_ROW                 0

//10bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_MVY_2_SHIFT                       (76)
#define MPEA2B_MB_START_PARAM_MVY_2_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVY_2_SHIFT)
#define MPEA2B_MB_START_PARAM_MVY_2_RANGE                       (85):(76)
#define MPEA2B_MB_START_PARAM_MVY_2_ROW                 0

//10bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_MVY_3_SHIFT                       (86)
#define MPEA2B_MB_START_PARAM_MVY_3_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVY_3_SHIFT)
#define MPEA2B_MB_START_PARAM_MVY_3_RANGE                       (95):(86)
#define MPEA2B_MB_START_PARAM_MVY_3_ROW                 0

//10bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_MVX_0_SHIFT                       (96)
#define MPEA2B_MB_START_PARAM_MVX_0_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVX_0_SHIFT)
#define MPEA2B_MB_START_PARAM_MVX_0_RANGE                       (105):(96)
#define MPEA2B_MB_START_PARAM_MVX_0_ROW                 0

//10bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_MVX_1_SHIFT                       (106)
#define MPEA2B_MB_START_PARAM_MVX_1_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVX_1_SHIFT)
#define MPEA2B_MB_START_PARAM_MVX_1_RANGE                       (115):(106)
#define MPEA2B_MB_START_PARAM_MVX_1_ROW                 0

//10bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_MVX_2_SHIFT                       (116)
#define MPEA2B_MB_START_PARAM_MVX_2_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVX_2_SHIFT)
#define MPEA2B_MB_START_PARAM_MVX_2_RANGE                       (125):(116)
#define MPEA2B_MB_START_PARAM_MVX_2_ROW                 0

//10bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_MVX_3_SHIFT                       (126)
#define MPEA2B_MB_START_PARAM_MVX_3_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVX_3_SHIFT)
#define MPEA2B_MB_START_PARAM_MVX_3_RANGE                       (135):(126)
#define MPEA2B_MB_START_PARAM_MVX_3_ROW                 0

//10bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_NB_MVY_0_SHIFT                    (136)
#define MPEA2B_MB_START_PARAM_NB_MVY_0_FIELD                    ((0x3ff) << MPEA2B_MB_START_PARAM_NB_MVY_0_SHIFT)
#define MPEA2B_MB_START_PARAM_NB_MVY_0_RANGE                    (145):(136)
#define MPEA2B_MB_START_PARAM_NB_MVY_0_ROW                      0

//10bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_NB_MVY_1_SHIFT                    (146)
#define MPEA2B_MB_START_PARAM_NB_MVY_1_FIELD                    ((0x3ff) << MPEA2B_MB_START_PARAM_NB_MVY_1_SHIFT)
#define MPEA2B_MB_START_PARAM_NB_MVY_1_RANGE                    (155):(146)
#define MPEA2B_MB_START_PARAM_NB_MVY_1_ROW                      0

//10bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_NB_MVY_2_SHIFT                    (156)
#define MPEA2B_MB_START_PARAM_NB_MVY_2_FIELD                    ((0x3ff) << MPEA2B_MB_START_PARAM_NB_MVY_2_SHIFT)
#define MPEA2B_MB_START_PARAM_NB_MVY_2_RANGE                    (165):(156)
#define MPEA2B_MB_START_PARAM_NB_MVY_2_ROW                      0

//10bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_NB_MVY_3_SHIFT                    (166)
#define MPEA2B_MB_START_PARAM_NB_MVY_3_FIELD                    ((0x3ff) << MPEA2B_MB_START_PARAM_NB_MVY_3_SHIFT)
#define MPEA2B_MB_START_PARAM_NB_MVY_3_RANGE                    (175):(166)
#define MPEA2B_MB_START_PARAM_NB_MVY_3_ROW                      0

//10bit,mpeg4
#define MPEA2B_MB_START_PARAM_NB_MVY_4_SHIFT                    (176)
#define MPEA2B_MB_START_PARAM_NB_MVY_4_FIELD                    ((0x3ff) << MPEA2B_MB_START_PARAM_NB_MVY_4_SHIFT)
#define MPEA2B_MB_START_PARAM_NB_MVY_4_RANGE                    (185):(176)
#define MPEA2B_MB_START_PARAM_NB_MVY_4_ROW                      0

//10bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_NB_MVX_0_SHIFT                    (186)
#define MPEA2B_MB_START_PARAM_NB_MVX_0_FIELD                    ((0x3ff) << MPEA2B_MB_START_PARAM_NB_MVX_0_SHIFT)
#define MPEA2B_MB_START_PARAM_NB_MVX_0_RANGE                    (195):(186)
#define MPEA2B_MB_START_PARAM_NB_MVX_0_ROW                      0

//10bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_NB_MVX_1_SHIFT                    (196)
#define MPEA2B_MB_START_PARAM_NB_MVX_1_FIELD                    ((0x3ff) << MPEA2B_MB_START_PARAM_NB_MVX_1_SHIFT)
#define MPEA2B_MB_START_PARAM_NB_MVX_1_RANGE                    (205):(196)
#define MPEA2B_MB_START_PARAM_NB_MVX_1_ROW                      0

//10bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_NB_MVX_2_SHIFT                    (206)
#define MPEA2B_MB_START_PARAM_NB_MVX_2_FIELD                    ((0x3ff) << MPEA2B_MB_START_PARAM_NB_MVX_2_SHIFT)
#define MPEA2B_MB_START_PARAM_NB_MVX_2_RANGE                    (215):(206)
#define MPEA2B_MB_START_PARAM_NB_MVX_2_ROW                      0

//10bit,h264/mpeg4
#define MPEA2B_MB_START_PARAM_NB_MVX_3_SHIFT                    (216)
#define MPEA2B_MB_START_PARAM_NB_MVX_3_FIELD                    ((0x3ff) << MPEA2B_MB_START_PARAM_NB_MVX_3_SHIFT)
#define MPEA2B_MB_START_PARAM_NB_MVX_3_RANGE                    (225):(216)
#define MPEA2B_MB_START_PARAM_NB_MVX_3_ROW                      0

//10bit,mpeg4
#define MPEA2B_MB_START_PARAM_NB_MVX_4_SHIFT                    (226)
#define MPEA2B_MB_START_PARAM_NB_MVX_4_FIELD                    ((0x3ff) << MPEA2B_MB_START_PARAM_NB_MVX_4_SHIFT)
#define MPEA2B_MB_START_PARAM_NB_MVX_4_RANGE                    (235):(226)
#define MPEA2B_MB_START_PARAM_NB_MVX_4_ROW                      0

// 10 bit,h264
#define MPEA2B_MB_START_PARAM_NB_MVX_5_SHIFT                    (236)
#define MPEA2B_MB_START_PARAM_NB_MVX_5_FIELD                    ((0x3ff) << MPEA2B_MB_START_PARAM_NB_MVX_5_SHIFT)
#define MPEA2B_MB_START_PARAM_NB_MVX_5_RANGE                    (245):(236)
#define MPEA2B_MB_START_PARAM_NB_MVX_5_ROW                      0

// 10 bit,h264
#define MPEA2B_MB_START_PARAM_NB_MVY_5_SHIFT                    (246)
#define MPEA2B_MB_START_PARAM_NB_MVY_5_FIELD                    ((0x3ff) << MPEA2B_MB_START_PARAM_NB_MVY_5_SHIFT)
#define MPEA2B_MB_START_PARAM_NB_MVY_5_RANGE                    (255):(246)
#define MPEA2B_MB_START_PARAM_NB_MVY_5_ROW                      0

// 2 bits, h264
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_0_SHIFT                    (256)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_0_FIELD                    ((0x3) << MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_0_SHIFT)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_0_RANGE                    (257):(256)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_0_ROW                      0
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_0_s8x8                     (0)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_0_s8x4                     (1)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_0_s4x8                     (2)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_0_s4x4                     (3)

// 2 bits, h264
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_1_SHIFT                    (258)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_1_FIELD                    ((0x3) << MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_1_SHIFT)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_1_RANGE                    (259):(258)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_1_ROW                      0
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_1_s8x8                     (0)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_1_s8x4                     (1)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_1_s4x8                     (2)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_1_s4x4                     (3)

// 2 bits, h264
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_2_SHIFT                    (260)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_2_FIELD                    ((0x3) << MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_2_SHIFT)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_2_RANGE                    (261):(260)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_2_ROW                      0
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_2_s8x8                     (0)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_2_s8x4                     (1)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_2_s4x8                     (2)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_2_s4x4                     (3)

// 2 bits, h264
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_3_SHIFT                    (262)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_3_FIELD                    ((0x3) << MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_3_SHIFT)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_3_RANGE                    (263):(262)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_3_ROW                      0
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_3_s8x8                     (0)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_3_s8x4                     (1)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_3_s4x8                     (2)
#define MPEA2B_MB_START_PARAM_MVMODE_SUBTYPE_3_s4x4                     (3)

// 10bit,h264/mpeg4 (mv pos B4 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVX_4_SHIFT                       (264)
#define MPEA2B_MB_START_PARAM_MVX_4_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVX_4_SHIFT)
#define MPEA2B_MB_START_PARAM_MVX_4_RANGE                       (273):(264)
#define MPEA2B_MB_START_PARAM_MVX_4_ROW                 0

// 10bit,h264/mpeg4 (mv pos B5 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVX_5_SHIFT                       (274)
#define MPEA2B_MB_START_PARAM_MVX_5_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVX_5_SHIFT)
#define MPEA2B_MB_START_PARAM_MVX_5_RANGE                       (283):(274)
#define MPEA2B_MB_START_PARAM_MVX_5_ROW                 0

// 10bit,h264/mpeg4 (mv pos B6 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVX_6_SHIFT                       (284)
#define MPEA2B_MB_START_PARAM_MVX_6_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVX_6_SHIFT)
#define MPEA2B_MB_START_PARAM_MVX_6_RANGE                       (293):(284)
#define MPEA2B_MB_START_PARAM_MVX_6_ROW                 0

// 10bit,h264/mpeg4 (mv pos B7 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVX_7_SHIFT                       (294)
#define MPEA2B_MB_START_PARAM_MVX_7_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVX_7_SHIFT)
#define MPEA2B_MB_START_PARAM_MVX_7_RANGE                       (303):(294)
#define MPEA2B_MB_START_PARAM_MVX_7_ROW                 0

// 10bit,h264/mpeg4 (mv pos B8 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVX_8_SHIFT                       (304)
#define MPEA2B_MB_START_PARAM_MVX_8_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVX_8_SHIFT)
#define MPEA2B_MB_START_PARAM_MVX_8_RANGE                       (313):(304)
#define MPEA2B_MB_START_PARAM_MVX_8_ROW                 0

// 10bit,h264/mpeg4 (mv pos B9 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVX_9_SHIFT                       (314)
#define MPEA2B_MB_START_PARAM_MVX_9_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVX_9_SHIFT)
#define MPEA2B_MB_START_PARAM_MVX_9_RANGE                       (323):(314)
#define MPEA2B_MB_START_PARAM_MVX_9_ROW                 0

// 10bit,h264/mpeg4 (mv pos B10 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVX_10_SHIFT                      (324)
#define MPEA2B_MB_START_PARAM_MVX_10_FIELD                      ((0x3ff) << MPEA2B_MB_START_PARAM_MVX_10_SHIFT)
#define MPEA2B_MB_START_PARAM_MVX_10_RANGE                      (333):(324)
#define MPEA2B_MB_START_PARAM_MVX_10_ROW                        0

// 10bit,h264/mpeg4 (mv pos B11 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVX_11_SHIFT                      (334)
#define MPEA2B_MB_START_PARAM_MVX_11_FIELD                      ((0x3ff) << MPEA2B_MB_START_PARAM_MVX_11_SHIFT)
#define MPEA2B_MB_START_PARAM_MVX_11_RANGE                      (343):(334)
#define MPEA2B_MB_START_PARAM_MVX_11_ROW                        0

// 10bit,h264/mpeg4 (mv pos B12 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVX_12_SHIFT                      (344)
#define MPEA2B_MB_START_PARAM_MVX_12_FIELD                      ((0x3ff) << MPEA2B_MB_START_PARAM_MVX_12_SHIFT)
#define MPEA2B_MB_START_PARAM_MVX_12_RANGE                      (353):(344)
#define MPEA2B_MB_START_PARAM_MVX_12_ROW                        0

// 10bit,h264/mpeg4 (mv pos B13 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVX_13_SHIFT                      (354)
#define MPEA2B_MB_START_PARAM_MVX_13_FIELD                      ((0x3ff) << MPEA2B_MB_START_PARAM_MVX_13_SHIFT)
#define MPEA2B_MB_START_PARAM_MVX_13_RANGE                      (363):(354)
#define MPEA2B_MB_START_PARAM_MVX_13_ROW                        0

// 10bit,h264/mpeg4 (mv pos B14 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVX_14_SHIFT                      (364)
#define MPEA2B_MB_START_PARAM_MVX_14_FIELD                      ((0x3ff) << MPEA2B_MB_START_PARAM_MVX_14_SHIFT)
#define MPEA2B_MB_START_PARAM_MVX_14_RANGE                      (373):(364)
#define MPEA2B_MB_START_PARAM_MVX_14_ROW                        0

// 10bit,h264/mpeg4 (mv pos B15 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVX_15_SHIFT                      (374)
#define MPEA2B_MB_START_PARAM_MVX_15_FIELD                      ((0x3ff) << MPEA2B_MB_START_PARAM_MVX_15_SHIFT)
#define MPEA2B_MB_START_PARAM_MVX_15_RANGE                      (383):(374)
#define MPEA2B_MB_START_PARAM_MVX_15_ROW                        0

// 10bit,h264/mpeg4 (mv pos B4 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVY_4_SHIFT                       (384)
#define MPEA2B_MB_START_PARAM_MVY_4_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVY_4_SHIFT)
#define MPEA2B_MB_START_PARAM_MVY_4_RANGE                       (393):(384)
#define MPEA2B_MB_START_PARAM_MVY_4_ROW                 0

// 10bit,h264/mpeg4 (mv pos B5 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVY_5_SHIFT                       (394)
#define MPEA2B_MB_START_PARAM_MVY_5_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVY_5_SHIFT)
#define MPEA2B_MB_START_PARAM_MVY_5_RANGE                       (403):(394)
#define MPEA2B_MB_START_PARAM_MVY_5_ROW                 0

// 10bit,h264/mpeg4 (mv pos B6 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVY_6_SHIFT                       (404)
#define MPEA2B_MB_START_PARAM_MVY_6_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVY_6_SHIFT)
#define MPEA2B_MB_START_PARAM_MVY_6_RANGE                       (413):(404)
#define MPEA2B_MB_START_PARAM_MVY_6_ROW                 0

// 10bit,h264/mpeg4 (mv pos B7 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVY_7_SHIFT                       (414)
#define MPEA2B_MB_START_PARAM_MVY_7_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVY_7_SHIFT)
#define MPEA2B_MB_START_PARAM_MVY_7_RANGE                       (423):(414)
#define MPEA2B_MB_START_PARAM_MVY_7_ROW                 0

// 10bit,h264/mpeg4 (mv pos B8 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVY_8_SHIFT                       (424)
#define MPEA2B_MB_START_PARAM_MVY_8_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVY_8_SHIFT)
#define MPEA2B_MB_START_PARAM_MVY_8_RANGE                       (433):(424)
#define MPEA2B_MB_START_PARAM_MVY_8_ROW                 0

// 10bit,h264/mpeg4 (mv pos B9 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVY_9_SHIFT                       (434)
#define MPEA2B_MB_START_PARAM_MVY_9_FIELD                       ((0x3ff) << MPEA2B_MB_START_PARAM_MVY_9_SHIFT)
#define MPEA2B_MB_START_PARAM_MVY_9_RANGE                       (443):(434)
#define MPEA2B_MB_START_PARAM_MVY_9_ROW                 0

// 10bit,h264/mpeg4 (mv pos B10 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVY_10_SHIFT                      (444)
#define MPEA2B_MB_START_PARAM_MVY_10_FIELD                      ((0x3ff) << MPEA2B_MB_START_PARAM_MVY_10_SHIFT)
#define MPEA2B_MB_START_PARAM_MVY_10_RANGE                      (453):(444)
#define MPEA2B_MB_START_PARAM_MVY_10_ROW                        0

// 10bit,h264/mpeg4 (mv pos B11 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVY_11_SHIFT                      (454)
#define MPEA2B_MB_START_PARAM_MVY_11_FIELD                      ((0x3ff) << MPEA2B_MB_START_PARAM_MVY_11_SHIFT)
#define MPEA2B_MB_START_PARAM_MVY_11_RANGE                      (463):(454)
#define MPEA2B_MB_START_PARAM_MVY_11_ROW                        0

// 10bit,h264/mpeg4 (mv pos B12 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVY_12_SHIFT                      (464)
#define MPEA2B_MB_START_PARAM_MVY_12_FIELD                      ((0x3ff) << MPEA2B_MB_START_PARAM_MVY_12_SHIFT)
#define MPEA2B_MB_START_PARAM_MVY_12_RANGE                      (473):(464)
#define MPEA2B_MB_START_PARAM_MVY_12_ROW                        0

// 10bit,h264/mpeg4 (mv pos B13 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVY_13_SHIFT                      (474)
#define MPEA2B_MB_START_PARAM_MVY_13_FIELD                      ((0x3ff) << MPEA2B_MB_START_PARAM_MVY_13_SHIFT)
#define MPEA2B_MB_START_PARAM_MVY_13_RANGE                      (483):(474)
#define MPEA2B_MB_START_PARAM_MVY_13_ROW                        0

// 10bit,h264/mpeg4 (mv pos B14 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVY_14_SHIFT                      (484)
#define MPEA2B_MB_START_PARAM_MVY_14_FIELD                      ((0x3ff) << MPEA2B_MB_START_PARAM_MVY_14_SHIFT)
#define MPEA2B_MB_START_PARAM_MVY_14_RANGE                      (493):(484)
#define MPEA2B_MB_START_PARAM_MVY_14_ROW                        0

// 10bit,h264/mpeg4 (mv pos B15 refer to diagram above)
#define MPEA2B_MB_START_PARAM_MVY_15_SHIFT                      (494)
#define MPEA2B_MB_START_PARAM_MVY_15_FIELD                      ((0x3ff) << MPEA2B_MB_START_PARAM_MVY_15_SHIFT)
#define MPEA2B_MB_START_PARAM_MVY_15_RANGE                      (503):(494)
#define MPEA2B_MB_START_PARAM_MVY_15_ROW                        0

// 2 bit, used in APPROX mode, 00=V, 01=H, 10=DC, 11=Planar
#define MPEA2B_MB_START_PARAM_IMODE_SHIFT                       (504)
#define MPEA2B_MB_START_PARAM_IMODE_FIELD                       ((0x3) << MPEA2B_MB_START_PARAM_IMODE_SHIFT)
#define MPEA2B_MB_START_PARAM_IMODE_RANGE                       (505):(504)
#define MPEA2B_MB_START_PARAM_IMODE_ROW                 0
#define MPEA2B_MB_START_PARAM_IMODE_VERT                        (0)
#define MPEA2B_MB_START_PARAM_IMODE_HORZ                        (1)
#define MPEA2B_MB_START_PARAM_IMODE_DC                  (2)
#define MPEA2B_MB_START_PARAM_IMODE_PLANAR                      (3)

// 4bit,h264
#define MPEA2B_MB_START_PARAM_SLICE_ALPHA_C0_OFFSET_DIV2_SHIFT                  (506)
#define MPEA2B_MB_START_PARAM_SLICE_ALPHA_C0_OFFSET_DIV2_FIELD                  ((0xf) << MPEA2B_MB_START_PARAM_SLICE_ALPHA_C0_OFFSET_DIV2_SHIFT)
#define MPEA2B_MB_START_PARAM_SLICE_ALPHA_C0_OFFSET_DIV2_RANGE                  (509):(506)
#define MPEA2B_MB_START_PARAM_SLICE_ALPHA_C0_OFFSET_DIV2_ROW                    0

// 4bit,h264
#define MPEA2B_MB_START_PARAM_SLICE_BETA_OFFSET_DIV2_SHIFT                      (510)
#define MPEA2B_MB_START_PARAM_SLICE_BETA_OFFSET_DIV2_FIELD                      ((0xf) << MPEA2B_MB_START_PARAM_SLICE_BETA_OFFSET_DIV2_SHIFT)
#define MPEA2B_MB_START_PARAM_SLICE_BETA_OFFSET_DIV2_RANGE                      (513):(510)
#define MPEA2B_MB_START_PARAM_SLICE_BETA_OFFSET_DIV2_ROW                        0


// Packet MPEA2B_FRAME_START_PARAM
#define MPEA2B_FRAME_START_PARAM_SIZE 114

//2bit,h264/mpeg4, new field
#define MPEA2B_FRAME_START_PARAM_CONTEXT_ID_SHIFT                       (0)
#define MPEA2B_FRAME_START_PARAM_CONTEXT_ID_FIELD                       ((0x3) << MPEA2B_FRAME_START_PARAM_CONTEXT_ID_SHIFT)
#define MPEA2B_FRAME_START_PARAM_CONTEXT_ID_RANGE                       (1):(0)
#define MPEA2B_FRAME_START_PARAM_CONTEXT_ID_ROW                 0

//3bit,h264. Increased to 3 bits
#define MPEA2B_FRAME_START_PARAM_NUM_SLICE_GROUPS_SHIFT                 (2)
#define MPEA2B_FRAME_START_PARAM_NUM_SLICE_GROUPS_FIELD                 ((0x7) << MPEA2B_FRAME_START_PARAM_NUM_SLICE_GROUPS_SHIFT)
#define MPEA2B_FRAME_START_PARAM_NUM_SLICE_GROUPS_RANGE                 (4):(2)
#define MPEA2B_FRAME_START_PARAM_NUM_SLICE_GROUPS_ROW                   0
#define MPEA2B_FRAME_START_PARAM_NUM_SLICE_GROUPS_ZERO                  (0)
#define MPEA2B_FRAME_START_PARAM_NUM_SLICE_GROUPS_ONE                   (1)
#define MPEA2B_FRAME_START_PARAM_NUM_SLICE_GROUPS_TWO                   (2)

//1bit,h264
#define MPEA2B_FRAME_START_PARAM_LWRRFRAMEREFTYPE_SHIFT                 (5)
#define MPEA2B_FRAME_START_PARAM_LWRRFRAMEREFTYPE_FIELD                 ((0x1) << MPEA2B_FRAME_START_PARAM_LWRRFRAMEREFTYPE_SHIFT)
#define MPEA2B_FRAME_START_PARAM_LWRRFRAMEREFTYPE_RANGE                 (5):(5)
#define MPEA2B_FRAME_START_PARAM_LWRRFRAMEREFTYPE_ROW                   0
#define MPEA2B_FRAME_START_PARAM_LWRRFRAMEREFTYPE_SHORTTERM                     (0)
#define MPEA2B_FRAME_START_PARAM_LWRRFRAMEREFTYPE_LONGTERM                      (1)

//2bit,h264/mpeg4
//reserve 2'b3 for BFrame      
#define MPEA2B_FRAME_START_PARAM_FRAMETYPE_SHIFT                        (6)
#define MPEA2B_FRAME_START_PARAM_FRAMETYPE_FIELD                        ((0x3) << MPEA2B_FRAME_START_PARAM_FRAMETYPE_SHIFT)
#define MPEA2B_FRAME_START_PARAM_FRAMETYPE_RANGE                        (7):(6)
#define MPEA2B_FRAME_START_PARAM_FRAMETYPE_ROW                  0
#define MPEA2B_FRAME_START_PARAM_FRAMETYPE_IDR                  (0)
#define MPEA2B_FRAME_START_PARAM_FRAMETYPE_NONIDR                       (1)
#define MPEA2B_FRAME_START_PARAM_FRAMETYPE_SKIP                 (2)

//32bit,h264/mpeg4
#define MPEA2B_FRAME_START_PARAM_FRAMENUM_SHIFT                 (8)
#define MPEA2B_FRAME_START_PARAM_FRAMENUM_FIELD                 ((0xffffffff) << MPEA2B_FRAME_START_PARAM_FRAMENUM_SHIFT)
#define MPEA2B_FRAME_START_PARAM_FRAMENUM_RANGE                 (39):(8)
#define MPEA2B_FRAME_START_PARAM_FRAMENUM_ROW                   0

//16bit,h264/mpeg4
#define MPEA2B_FRAME_START_PARAM_IMAGESIZEY_SHIFT                       (40)
#define MPEA2B_FRAME_START_PARAM_IMAGESIZEY_FIELD                       ((0xffff) << MPEA2B_FRAME_START_PARAM_IMAGESIZEY_SHIFT)
#define MPEA2B_FRAME_START_PARAM_IMAGESIZEY_RANGE                       (55):(40)
#define MPEA2B_FRAME_START_PARAM_IMAGESIZEY_ROW                 0

//16bit,h264/mpeg4
#define MPEA2B_FRAME_START_PARAM_IMAGESIZEX_SHIFT                       (56)
#define MPEA2B_FRAME_START_PARAM_IMAGESIZEX_FIELD                       ((0xffff) << MPEA2B_FRAME_START_PARAM_IMAGESIZEX_SHIFT)
#define MPEA2B_FRAME_START_PARAM_IMAGESIZEX_RANGE                       (71):(56)
#define MPEA2B_FRAME_START_PARAM_IMAGESIZEX_ROW                 0

//32bit,h264/mpeg4
#define MPEA2B_FRAME_START_PARAM_FRAMEBITBUDGET_SHIFT                   (72)
#define MPEA2B_FRAME_START_PARAM_FRAMEBITBUDGET_FIELD                   ((0xffffffff) << MPEA2B_FRAME_START_PARAM_FRAMEBITBUDGET_SHIFT)
#define MPEA2B_FRAME_START_PARAM_FRAMEBITBUDGET_RANGE                   (103):(72)
#define MPEA2B_FRAME_START_PARAM_FRAMEBITBUDGET_ROW                     0

// 1bit,h264
#define MPEA2B_FRAME_START_PARAM_CONSTRAINED_INTRA_PRED_FLAG_SHIFT                      (104)
#define MPEA2B_FRAME_START_PARAM_CONSTRAINED_INTRA_PRED_FLAG_FIELD                      ((0x1) << MPEA2B_FRAME_START_PARAM_CONSTRAINED_INTRA_PRED_FLAG_SHIFT)
#define MPEA2B_FRAME_START_PARAM_CONSTRAINED_INTRA_PRED_FLAG_RANGE                      (104):(104)
#define MPEA2B_FRAME_START_PARAM_CONSTRAINED_INTRA_PRED_FLAG_ROW                        0

// 2bit,h264
// All the following fields are new for AP20
#define MPEA2B_FRAME_START_PARAM_DISABLE_DEBLOCKING_FILTER_IDC_SHIFT                    (105)
#define MPEA2B_FRAME_START_PARAM_DISABLE_DEBLOCKING_FILTER_IDC_FIELD                    ((0x3) << MPEA2B_FRAME_START_PARAM_DISABLE_DEBLOCKING_FILTER_IDC_SHIFT)
#define MPEA2B_FRAME_START_PARAM_DISABLE_DEBLOCKING_FILTER_IDC_RANGE                    (106):(105)
#define MPEA2B_FRAME_START_PARAM_DISABLE_DEBLOCKING_FILTER_IDC_ROW                      0

// This parameter is applicable for APxx H.264
//  encoding only and is not used for
//  MPEG4/H.263 encoding.
// Each bit of this parameter specifies which
//  of the inter modes are enabled for encoding.
// 0 means "mode not present (disabled)".
// 1 means "mode is present (enabled)".
// Bit 0 - 16x16.
// Bit 1 - 16x8 (H264 only).
// Bit 2 - 8x16 (H264 only).
// Bit 3 - 8x8.
// Bit 4 - 8x4 (H264 only).
// Bit 5 - 4x8 (H264 only).
// Bit 6 - 4x4 (H264 only).
#define MPEA2B_FRAME_START_PARAM_MV_MODE_SHIFT                  (107)
#define MPEA2B_FRAME_START_PARAM_MV_MODE_FIELD                  ((0x7f) << MPEA2B_FRAME_START_PARAM_MV_MODE_SHIFT)
#define MPEA2B_FRAME_START_PARAM_MV_MODE_RANGE                  (113):(107)
#define MPEA2B_FRAME_START_PARAM_MV_MODE_ROW                    0


// Packet MPEA2B_MB_SUGQP
#define MPEA2B_MB_SUGQP_SIZE 25

//1-bit, indicates vlc frame start
#define MPEA2B_MB_SUGQP_VLC_FRAME_START_SHIFT                   (0)
#define MPEA2B_MB_SUGQP_VLC_FRAME_START_FIELD                   ((0x1) << MPEA2B_MB_SUGQP_VLC_FRAME_START_SHIFT)
#define MPEA2B_MB_SUGQP_VLC_FRAME_START_RANGE                   (0):(0)
#define MPEA2B_MB_SUGQP_VLC_FRAME_START_ROW                     0

//6bit,h264/mpeg4
#define MPEA2B_MB_SUGQP_SUGQP_SHIFT                     (1)
#define MPEA2B_MB_SUGQP_SUGQP_FIELD                     ((0x3f) << MPEA2B_MB_SUGQP_SUGQP_SHIFT)
#define MPEA2B_MB_SUGQP_SUGQP_RANGE                     (6):(1)
#define MPEA2B_MB_SUGQP_SUGQP_ROW                       0

// current slice group id, h264
#define MPEA2B_MB_SUGQP_LWRRMB_SLICEGRPID_SHIFT                 (7)
#define MPEA2B_MB_SUGQP_LWRRMB_SLICEGRPID_FIELD                 ((0x7) << MPEA2B_MB_SUGQP_LWRRMB_SLICEGRPID_SHIFT)
#define MPEA2B_MB_SUGQP_LWRRMB_SLICEGRPID_RANGE                 (9):(7)
#define MPEA2B_MB_SUGQP_LWRRMB_SLICEGRPID_ROW                   0

// left neighbor's slice group id, h264
#define MPEA2B_MB_SUGQP_LEFTMB_SLICEGRPID_SHIFT                 (10)
#define MPEA2B_MB_SUGQP_LEFTMB_SLICEGRPID_FIELD                 ((0x7) << MPEA2B_MB_SUGQP_LEFTMB_SLICEGRPID_SHIFT)
#define MPEA2B_MB_SUGQP_LEFTMB_SLICEGRPID_RANGE                 (12):(10)
#define MPEA2B_MB_SUGQP_LEFTMB_SLICEGRPID_ROW                   0

// top left neighbor's slice group id, h264
#define MPEA2B_MB_SUGQP_TOPLEFTMB_SLICEGRPID_SHIFT                      (13)
#define MPEA2B_MB_SUGQP_TOPLEFTMB_SLICEGRPID_FIELD                      ((0x7) << MPEA2B_MB_SUGQP_TOPLEFTMB_SLICEGRPID_SHIFT)
#define MPEA2B_MB_SUGQP_TOPLEFTMB_SLICEGRPID_RANGE                      (15):(13)
#define MPEA2B_MB_SUGQP_TOPLEFTMB_SLICEGRPID_ROW                        0

// top neighbor's slice group id, h264
#define MPEA2B_MB_SUGQP_TOPMB_SLICEGRPID_SHIFT                  (16)
#define MPEA2B_MB_SUGQP_TOPMB_SLICEGRPID_FIELD                  ((0x7) << MPEA2B_MB_SUGQP_TOPMB_SLICEGRPID_SHIFT)
#define MPEA2B_MB_SUGQP_TOPMB_SLICEGRPID_RANGE                  (18):(16)
#define MPEA2B_MB_SUGQP_TOPMB_SLICEGRPID_ROW                    0

// top right neighbor's slice group id, h264
#define MPEA2B_MB_SUGQP_TOPRIGHTMB_SLICEGRPID_SHIFT                     (19)
#define MPEA2B_MB_SUGQP_TOPRIGHTMB_SLICEGRPID_FIELD                     ((0x7) << MPEA2B_MB_SUGQP_TOPRIGHTMB_SLICEGRPID_SHIFT)
#define MPEA2B_MB_SUGQP_TOPRIGHTMB_SLICEGRPID_RANGE                     (21):(19)
#define MPEA2B_MB_SUGQP_TOPRIGHTMB_SLICEGRPID_ROW                       0

// 2bit,increased to 2 bits to support B-slice
#define MPEA2B_MB_SUGQP_SLICE_TYPE_SHIFT                        (22)
#define MPEA2B_MB_SUGQP_SLICE_TYPE_FIELD                        ((0x3) << MPEA2B_MB_SUGQP_SLICE_TYPE_SHIFT)
#define MPEA2B_MB_SUGQP_SLICE_TYPE_RANGE                        (23):(22)
#define MPEA2B_MB_SUGQP_SLICE_TYPE_ROW                  0
#define MPEA2B_MB_SUGQP_SLICE_TYPE_IType                        (0)
#define MPEA2B_MB_SUGQP_SLICE_TYPE_PType                        (1)

//1bit,h264, needed in MB-packetization mode
#define MPEA2B_MB_SUGQP_FIRSTMBINSLICE_SHIFT                    (24)
#define MPEA2B_MB_SUGQP_FIRSTMBINSLICE_FIELD                    ((0x1) << MPEA2B_MB_SUGQP_FIRSTMBINSLICE_SHIFT)
#define MPEA2B_MB_SUGQP_FIRSTMBINSLICE_RANGE                    (24):(24)
#define MPEA2B_MB_SUGQP_FIRSTMBINSLICE_ROW                      0


// Packet MPEB2A_I_SATD
#define MPEB2A_I_SATD_SIZE 17

// 17 bits, indicates SATD/SAD value of the I-MB in the best mode
#define MPEB2A_I_SATD_SATD_SHIFT                        (0)
#define MPEB2A_I_SATD_SATD_FIELD                        ((0x1ffff) << MPEB2A_I_SATD_SATD_SHIFT)
#define MPEB2A_I_SATD_SATD_RANGE                        (16):(0)
#define MPEB2A_I_SATD_SATD_ROW                  0

//*********************************
// packet MPEB2A_PRED_DONE
//
// This packet is sent in H264 hw mode (not valid in  mpeg4 or h264-bypass) for P MB only. 
// It is sent by MPEB, when it is done reading the pred buffer in MPEA, or when it wants to read
// the pred buffer again. The packet contents indicate these two possibilities as follows:
//
// 1. If all the bits in the packet are zero, the packet indicates MB  done.
// 2. If any bit is "1", pred buffer sends the corresponding 8x8 blocks to MPEB, and considers 
// the MB to be done at the end of last block transfer.
//*********************************

// Packet MPEB2A_PRED_DONE
#define MPEB2A_PRED_DONE_SIZE 6

// indicates request for Y0 pred data
#define MPEB2A_PRED_DONE_Y0_PRED_SHIFT                  (0)
#define MPEB2A_PRED_DONE_Y0_PRED_FIELD                  ((0x1) << MPEB2A_PRED_DONE_Y0_PRED_SHIFT)
#define MPEB2A_PRED_DONE_Y0_PRED_RANGE                  (0):(0)
#define MPEB2A_PRED_DONE_Y0_PRED_ROW                    0

// indicates request for Y1 pred data   
#define MPEB2A_PRED_DONE_Y1_PRED_SHIFT                  (1)
#define MPEB2A_PRED_DONE_Y1_PRED_FIELD                  ((0x1) << MPEB2A_PRED_DONE_Y1_PRED_SHIFT)
#define MPEB2A_PRED_DONE_Y1_PRED_RANGE                  (1):(1)
#define MPEB2A_PRED_DONE_Y1_PRED_ROW                    0

// indicates request for Y2 pred data
#define MPEB2A_PRED_DONE_Y2_PRED_SHIFT                  (2)
#define MPEB2A_PRED_DONE_Y2_PRED_FIELD                  ((0x1) << MPEB2A_PRED_DONE_Y2_PRED_SHIFT)
#define MPEB2A_PRED_DONE_Y2_PRED_RANGE                  (2):(2)
#define MPEB2A_PRED_DONE_Y2_PRED_ROW                    0

// indicates request for Y3 pred data
#define MPEB2A_PRED_DONE_Y3_PRED_SHIFT                  (3)
#define MPEB2A_PRED_DONE_Y3_PRED_FIELD                  ((0x1) << MPEB2A_PRED_DONE_Y3_PRED_SHIFT)
#define MPEB2A_PRED_DONE_Y3_PRED_RANGE                  (3):(3)
#define MPEB2A_PRED_DONE_Y3_PRED_ROW                    0

// indicates request for U pred data
#define MPEB2A_PRED_DONE_U_PRED_SHIFT                   (4)
#define MPEB2A_PRED_DONE_U_PRED_FIELD                   ((0x1) << MPEB2A_PRED_DONE_U_PRED_SHIFT)
#define MPEB2A_PRED_DONE_U_PRED_RANGE                   (4):(4)
#define MPEB2A_PRED_DONE_U_PRED_ROW                     0

// indicates request for V pred data
#define MPEB2A_PRED_DONE_V_PRED_SHIFT                   (5)
#define MPEB2A_PRED_DONE_V_PRED_FIELD                   ((0x1) << MPEB2A_PRED_DONE_V_PRED_SHIFT)
#define MPEB2A_PRED_DONE_V_PRED_RANGE                   (5):(5)
#define MPEB2A_PRED_DONE_V_PRED_ROW                     0

//*********************************
//
// MPEA and MPEC packets
//
//*********************************

// Packet MPEC2A_VLC_FRAME_DONE
#define MPEC2A_VLC_FRAME_DONE_SIZE 76

//[31:0]
#define MPEC2A_VLC_FRAME_DONE_FRBITLEN_SHIFT                    (0)
#define MPEC2A_VLC_FRAME_DONE_FRBITLEN_FIELD                    ((0xffffffff) << MPEC2A_VLC_FRAME_DONE_FRBITLEN_SHIFT)
#define MPEC2A_VLC_FRAME_DONE_FRBITLEN_RANGE                    (31):(0)
#define MPEC2A_VLC_FRAME_DONE_FRBITLEN_ROW                      0

//[19:0]
#define MPEC2A_VLC_FRAME_DONE_VLDMQPSUM_SHIFT                   (32)
#define MPEC2A_VLC_FRAME_DONE_VLDMQPSUM_FIELD                   ((0xfffff) << MPEC2A_VLC_FRAME_DONE_VLDMQPSUM_SHIFT)
#define MPEC2A_VLC_FRAME_DONE_VLDMQPSUM_RANGE                   (51):(32)
#define MPEC2A_VLC_FRAME_DONE_VLDMQPSUM_ROW                     0

//[23:0]
#define MPEC2A_VLC_FRAME_DONE_MBMODLENOUT_SHIFT                 (52)
#define MPEC2A_VLC_FRAME_DONE_MBMODLENOUT_FIELD                 ((0xffffff) << MPEC2A_VLC_FRAME_DONE_MBMODLENOUT_SHIFT)
#define MPEC2A_VLC_FRAME_DONE_MBMODLENOUT_RANGE                 (75):(52)
#define MPEC2A_VLC_FRAME_DONE_MBMODLENOUT_ROW                   0


// Packet MPEC2A_MB_DONE_PARAM
#define MPEC2A_MB_DONE_PARAM_SIZE 59

//[15:0]
#define MPEC2A_MB_DONE_PARAM_MBTEXLENOUT_SHIFT                  (0)
#define MPEC2A_MB_DONE_PARAM_MBTEXLENOUT_FIELD                  ((0xffff) << MPEC2A_MB_DONE_PARAM_MBTEXLENOUT_SHIFT)
#define MPEC2A_MB_DONE_PARAM_MBTEXLENOUT_RANGE                  (15):(0)
#define MPEC2A_MB_DONE_PARAM_MBTEXLENOUT_ROW                    0

//[7:0]
#define MPEC2A_MB_DONE_PARAM_LWRXRC_SHIFT                       (16)
#define MPEC2A_MB_DONE_PARAM_LWRXRC_FIELD                       ((0xff) << MPEC2A_MB_DONE_PARAM_LWRXRC_SHIFT)
#define MPEC2A_MB_DONE_PARAM_LWRXRC_RANGE                       (23):(16)
#define MPEC2A_MB_DONE_PARAM_LWRXRC_ROW                 0

//[7:0]
#define MPEC2A_MB_DONE_PARAM_LWRYRC_SHIFT                       (24)
#define MPEC2A_MB_DONE_PARAM_LWRYRC_FIELD                       ((0xff) << MPEC2A_MB_DONE_PARAM_LWRYRC_SHIFT)
#define MPEC2A_MB_DONE_PARAM_LWRYRC_RANGE                       (31):(24)
#define MPEC2A_MB_DONE_PARAM_LWRYRC_ROW                 0

//[5:0]
#define MPEC2A_MB_DONE_PARAM_LWRQPRC_SHIFT                      (32)
#define MPEC2A_MB_DONE_PARAM_LWRQPRC_FIELD                      ((0x3f) << MPEC2A_MB_DONE_PARAM_LWRQPRC_SHIFT)
#define MPEC2A_MB_DONE_PARAM_LWRQPRC_RANGE                      (37):(32)
#define MPEC2A_MB_DONE_PARAM_LWRQPRC_ROW                        0

#define MPEC2A_MB_DONE_PARAM_IPCM_FLAG_SHIFT                    (38)
#define MPEC2A_MB_DONE_PARAM_IPCM_FLAG_FIELD                    ((0x1) << MPEC2A_MB_DONE_PARAM_IPCM_FLAG_SHIFT)
#define MPEC2A_MB_DONE_PARAM_IPCM_FLAG_RANGE                    (38):(38)
#define MPEC2A_MB_DONE_PARAM_IPCM_FLAG_ROW                      0

#define MPEC2A_MB_DONE_PARAM_NEW_PKT_SHIFT                      (39)
#define MPEC2A_MB_DONE_PARAM_NEW_PKT_FIELD                      ((0x1) << MPEC2A_MB_DONE_PARAM_NEW_PKT_SHIFT)
#define MPEC2A_MB_DONE_PARAM_NEW_PKT_RANGE                      (39):(39)
#define MPEC2A_MB_DONE_PARAM_NEW_PKT_ROW                        0

#define MPEC2A_MB_DONE_PARAM_SLICE_GRPID_SHIFT                  (40)
#define MPEC2A_MB_DONE_PARAM_SLICE_GRPID_FIELD                  ((0x7) << MPEC2A_MB_DONE_PARAM_SLICE_GRPID_SHIFT)
#define MPEC2A_MB_DONE_PARAM_SLICE_GRPID_RANGE                  (42):(40)
#define MPEC2A_MB_DONE_PARAM_SLICE_GRPID_ROW                    0

#define MPEC2A_MB_DONE_PARAM_LWR_MB_MOD_LEN_SHIFT                       (43)
#define MPEC2A_MB_DONE_PARAM_LWR_MB_MOD_LEN_FIELD                       ((0xffff) << MPEC2A_MB_DONE_PARAM_LWR_MB_MOD_LEN_SHIFT)
#define MPEC2A_MB_DONE_PARAM_LWR_MB_MOD_LEN_RANGE                       (58):(43)
#define MPEC2A_MB_DONE_PARAM_LWR_MB_MOD_LEN_ROW                 0


// Packet MPEA2C_FRAME_START_PARAM
#define MPEA2C_FRAME_START_PARAM_SIZE 49

//2bit,h264/mpeg4
//reserve 2'b3 for BFrame      
#define MPEA2C_FRAME_START_PARAM_FRAMETYPE_SHIFT                        (0)
#define MPEA2C_FRAME_START_PARAM_FRAMETYPE_FIELD                        ((0x3) << MPEA2C_FRAME_START_PARAM_FRAMETYPE_SHIFT)
#define MPEA2C_FRAME_START_PARAM_FRAMETYPE_RANGE                        (1):(0)
#define MPEA2C_FRAME_START_PARAM_FRAMETYPE_ROW                  0
#define MPEA2C_FRAME_START_PARAM_FRAMETYPE_IDR                  (0)
#define MPEA2C_FRAME_START_PARAM_FRAMETYPE_NONIDR                       (1)
#define MPEA2C_FRAME_START_PARAM_FRAMETYPE_SKIP                 (2)

//32bit,h264/mpeg4 
#define MPEA2C_FRAME_START_PARAM_FRAMENUM_SHIFT                 (2)
#define MPEA2C_FRAME_START_PARAM_FRAMENUM_FIELD                 ((0xffff) << MPEA2C_FRAME_START_PARAM_FRAMENUM_SHIFT)
#define MPEA2C_FRAME_START_PARAM_FRAMENUM_RANGE                 (17):(2)
#define MPEA2C_FRAME_START_PARAM_FRAMENUM_ROW                   0

//32bit,h264/mpeg4
#define MPEA2C_FRAME_START_PARAM_FRAMEINITQP_SHIFT                      (18)
#define MPEA2C_FRAME_START_PARAM_FRAMEINITQP_FIELD                      ((0x3f) << MPEA2C_FRAME_START_PARAM_FRAMEINITQP_SHIFT)
#define MPEA2C_FRAME_START_PARAM_FRAMEINITQP_RANGE                      (23):(18)
#define MPEA2C_FRAME_START_PARAM_FRAMEINITQP_ROW                        0

// 1bit,h264
#define MPEA2C_FRAME_START_PARAM_CONSTRAINED_INTRA_PRED_FLAG_SHIFT                      (24)
#define MPEA2C_FRAME_START_PARAM_CONSTRAINED_INTRA_PRED_FLAG_FIELD                      ((0x1) << MPEA2C_FRAME_START_PARAM_CONSTRAINED_INTRA_PRED_FLAG_SHIFT)
#define MPEA2C_FRAME_START_PARAM_CONSTRAINED_INTRA_PRED_FLAG_RANGE                      (24):(24)
#define MPEA2C_FRAME_START_PARAM_CONSTRAINED_INTRA_PRED_FLAG_ROW                        0

#define MPEA2C_FRAME_START_PARAM_DISABLE_DEBLOCKING_FILTER_IDC_SHIFT                    (25)
#define MPEA2C_FRAME_START_PARAM_DISABLE_DEBLOCKING_FILTER_IDC_FIELD                    ((0x3) << MPEA2C_FRAME_START_PARAM_DISABLE_DEBLOCKING_FILTER_IDC_SHIFT)
#define MPEA2C_FRAME_START_PARAM_DISABLE_DEBLOCKING_FILTER_IDC_RANGE                    (26):(25)
#define MPEA2C_FRAME_START_PARAM_DISABLE_DEBLOCKING_FILTER_IDC_ROW                      0

// 1bit,h264
#define MPEA2C_FRAME_START_PARAM_SEQ_PARAM_CHANGED_SHIFT                        (27)
#define MPEA2C_FRAME_START_PARAM_SEQ_PARAM_CHANGED_FIELD                        ((0x1) << MPEA2C_FRAME_START_PARAM_SEQ_PARAM_CHANGED_SHIFT)
#define MPEA2C_FRAME_START_PARAM_SEQ_PARAM_CHANGED_RANGE                        (27):(27)
#define MPEA2C_FRAME_START_PARAM_SEQ_PARAM_CHANGED_ROW                  0

// 1bit,h264
#define MPEA2C_FRAME_START_PARAM_PIC_PARAM_CHANGED_SHIFT                        (28)
#define MPEA2C_FRAME_START_PARAM_PIC_PARAM_CHANGED_FIELD                        ((0x1) << MPEA2C_FRAME_START_PARAM_PIC_PARAM_CHANGED_SHIFT)
#define MPEA2C_FRAME_START_PARAM_PIC_PARAM_CHANGED_RANGE                        (28):(28)
#define MPEA2C_FRAME_START_PARAM_PIC_PARAM_CHANGED_ROW                  0

// 4bit,h264
#define MPEA2C_FRAME_START_PARAM_SLICE_GROUP_CHANGE_CYCLE_WIDTH_SHIFT                   (29)
#define MPEA2C_FRAME_START_PARAM_SLICE_GROUP_CHANGE_CYCLE_WIDTH_FIELD                   ((0xf) << MPEA2C_FRAME_START_PARAM_SLICE_GROUP_CHANGE_CYCLE_WIDTH_SHIFT)
#define MPEA2C_FRAME_START_PARAM_SLICE_GROUP_CHANGE_CYCLE_WIDTH_RANGE                   (32):(29)
#define MPEA2C_FRAME_START_PARAM_SLICE_GROUP_CHANGE_CYCLE_WIDTH_ROW                     0

//32bit,h264/mpeg4 
#define MPEA2C_FRAME_START_PARAM_MPEC_FRAMENUM_SHIFT                    (33)
#define MPEA2C_FRAME_START_PARAM_MPEC_FRAMENUM_FIELD                    ((0xffff) << MPEA2C_FRAME_START_PARAM_MPEC_FRAMENUM_SHIFT)
#define MPEA2C_FRAME_START_PARAM_MPEC_FRAMENUM_RANGE                    (48):(33)
#define MPEA2C_FRAME_START_PARAM_MPEC_FRAMENUM_ROW                      0

//*********************************
//
// MPEB and MPEC packets
//
//*********************************

// Packet MPEB2C_QUANT_DATA
#define MPEB2C_QUANT_DATA_SIZE 104

#define MPEB2C_QUANT_DATA_COEFF_0_SHIFT                 (0)
#define MPEB2C_QUANT_DATA_COEFF_0_FIELD                 ((0x1fff) << MPEB2C_QUANT_DATA_COEFF_0_SHIFT)
#define MPEB2C_QUANT_DATA_COEFF_0_RANGE                 (12):(0)
#define MPEB2C_QUANT_DATA_COEFF_0_ROW                   0

#define MPEB2C_QUANT_DATA_COEFF_1_SHIFT                 (13)
#define MPEB2C_QUANT_DATA_COEFF_1_FIELD                 ((0x1fff) << MPEB2C_QUANT_DATA_COEFF_1_SHIFT)
#define MPEB2C_QUANT_DATA_COEFF_1_RANGE                 (25):(13)
#define MPEB2C_QUANT_DATA_COEFF_1_ROW                   0

#define MPEB2C_QUANT_DATA_COEFF_2_SHIFT                 (26)
#define MPEB2C_QUANT_DATA_COEFF_2_FIELD                 ((0x1fff) << MPEB2C_QUANT_DATA_COEFF_2_SHIFT)
#define MPEB2C_QUANT_DATA_COEFF_2_RANGE                 (38):(26)
#define MPEB2C_QUANT_DATA_COEFF_2_ROW                   0

#define MPEB2C_QUANT_DATA_COEFF_3_SHIFT                 (39)
#define MPEB2C_QUANT_DATA_COEFF_3_FIELD                 ((0x1fff) << MPEB2C_QUANT_DATA_COEFF_3_SHIFT)
#define MPEB2C_QUANT_DATA_COEFF_3_RANGE                 (51):(39)
#define MPEB2C_QUANT_DATA_COEFF_3_ROW                   0

#define MPEB2C_QUANT_DATA_COEFF_4_SHIFT                 (52)
#define MPEB2C_QUANT_DATA_COEFF_4_FIELD                 ((0x1fff) << MPEB2C_QUANT_DATA_COEFF_4_SHIFT)
#define MPEB2C_QUANT_DATA_COEFF_4_RANGE                 (64):(52)
#define MPEB2C_QUANT_DATA_COEFF_4_ROW                   0

#define MPEB2C_QUANT_DATA_COEFF_5_SHIFT                 (65)
#define MPEB2C_QUANT_DATA_COEFF_5_FIELD                 ((0x1fff) << MPEB2C_QUANT_DATA_COEFF_5_SHIFT)
#define MPEB2C_QUANT_DATA_COEFF_5_RANGE                 (77):(65)
#define MPEB2C_QUANT_DATA_COEFF_5_ROW                   0

#define MPEB2C_QUANT_DATA_COEFF_6_SHIFT                 (78)
#define MPEB2C_QUANT_DATA_COEFF_6_FIELD                 ((0x1fff) << MPEB2C_QUANT_DATA_COEFF_6_SHIFT)
#define MPEB2C_QUANT_DATA_COEFF_6_RANGE                 (90):(78)
#define MPEB2C_QUANT_DATA_COEFF_6_ROW                   0

#define MPEB2C_QUANT_DATA_COEFF_7_SHIFT                 (91)
#define MPEB2C_QUANT_DATA_COEFF_7_FIELD                 ((0x1fff) << MPEB2C_QUANT_DATA_COEFF_7_SHIFT)
#define MPEB2C_QUANT_DATA_COEFF_7_RANGE                 (103):(91)
#define MPEB2C_QUANT_DATA_COEFF_7_ROW                   0

#define MPEB2C_QUANT_DATA_DATA_SHIFT                    (0)
#define MPEB2C_QUANT_DATA_DATA_FIELD                    ((0xff) << MPEB2C_QUANT_DATA_DATA_SHIFT)
#define MPEB2C_QUANT_DATA_DATA_RANGE                    (103):(0)
#define MPEB2C_QUANT_DATA_DATA_ROW                      0


// Packet MPEC2B_QUANT_ADDR
#define MPEC2B_QUANT_ADDR_SIZE 6

#define MPEC2B_QUANT_ADDR_ADDR_SHIFT                    (0)
#define MPEC2B_QUANT_ADDR_ADDR_FIELD                    ((0x3f) << MPEC2B_QUANT_ADDR_ADDR_SHIFT)
#define MPEC2B_QUANT_ADDR_ADDR_RANGE                    (5):(0)
#define MPEC2B_QUANT_ADDR_ADDR_ROW                      0


// Packet MPEB2C_H264_PKT
#define MPEB2C_H264_PKT_SIZE 76

#define MPEB2C_H264_PKT_VLC_DATA_SHIFT                  (0)
#define MPEB2C_H264_PKT_VLC_DATA_FIELD                  ((0xffffffff) << MPEB2C_H264_PKT_VLC_DATA_SHIFT)
#define MPEB2C_H264_PKT_VLC_DATA_RANGE                  (63):(0)
#define MPEB2C_H264_PKT_VLC_DATA_ROW                    0

#define MPEB2C_H264_PKT_VLC_BLK_NUM_SHIFT                       (64)
#define MPEB2C_H264_PKT_VLC_BLK_NUM_FIELD                       ((0x7) << MPEB2C_H264_PKT_VLC_BLK_NUM_SHIFT)
#define MPEB2C_H264_PKT_VLC_BLK_NUM_RANGE                       (66):(64)
#define MPEB2C_H264_PKT_VLC_BLK_NUM_ROW                 0

#define MPEB2C_H264_PKT_VLC_WORD_TYPE_SHIFT                     (67)
#define MPEB2C_H264_PKT_VLC_WORD_TYPE_FIELD                     ((0x7) << MPEB2C_H264_PKT_VLC_WORD_TYPE_SHIFT)
#define MPEB2C_H264_PKT_VLC_WORD_TYPE_RANGE                     (69):(67)
#define MPEB2C_H264_PKT_VLC_WORD_TYPE_ROW                       0

#define MPEB2C_H264_PKT_VLC_FR_START_SHIFT                      (70)
#define MPEB2C_H264_PKT_VLC_FR_START_FIELD                      ((0x1) << MPEB2C_H264_PKT_VLC_FR_START_SHIFT)
#define MPEB2C_H264_PKT_VLC_FR_START_RANGE                      (70):(70)
#define MPEB2C_H264_PKT_VLC_FR_START_ROW                        0

#define MPEB2C_H264_PKT_VLC_LAST_MB_SHIFT                       (71)
#define MPEB2C_H264_PKT_VLC_LAST_MB_FIELD                       ((0x1) << MPEB2C_H264_PKT_VLC_LAST_MB_SHIFT)
#define MPEB2C_H264_PKT_VLC_LAST_MB_RANGE                       (71):(71)
#define MPEB2C_H264_PKT_VLC_LAST_MB_ROW                 0

#define MPEB2C_H264_PKT_VLC_MB_START_SHIFT                      (72)
#define MPEB2C_H264_PKT_VLC_MB_START_FIELD                      ((0x1) << MPEB2C_H264_PKT_VLC_MB_START_SHIFT)
#define MPEB2C_H264_PKT_VLC_MB_START_RANGE                      (72):(72)
#define MPEB2C_H264_PKT_VLC_MB_START_ROW                        0

#define MPEB2C_H264_PKT_VLC_MB_END_SHIFT                        (73)
#define MPEB2C_H264_PKT_VLC_MB_END_FIELD                        ((0x1) << MPEB2C_H264_PKT_VLC_MB_END_SHIFT)
#define MPEB2C_H264_PKT_VLC_MB_END_RANGE                        (73):(73)
#define MPEB2C_H264_PKT_VLC_MB_END_ROW                  0

#define MPEB2C_H264_PKT_VLC_BLK_START_SHIFT                     (74)
#define MPEB2C_H264_PKT_VLC_BLK_START_FIELD                     ((0x1) << MPEB2C_H264_PKT_VLC_BLK_START_SHIFT)
#define MPEB2C_H264_PKT_VLC_BLK_START_RANGE                     (74):(74)
#define MPEB2C_H264_PKT_VLC_BLK_START_ROW                       0

#define MPEB2C_H264_PKT_VLC_BLK_END_SHIFT                       (75)
#define MPEB2C_H264_PKT_VLC_BLK_END_FIELD                       ((0x1) << MPEB2C_H264_PKT_VLC_BLK_END_SHIFT)
#define MPEB2C_H264_PKT_VLC_BLK_END_RANGE                       (75):(75)
#define MPEB2C_H264_PKT_VLC_BLK_END_ROW                 0


// Packet MPEC2B_QBUF_ADDR
#define MPEC2B_QBUF_ADDR_SIZE 6

#define MPEC2B_QBUF_ADDR_ADDRESS_SHIFT                  (0)
#define MPEC2B_QBUF_ADDR_ADDRESS_FIELD                  ((0x3f) << MPEC2B_QBUF_ADDR_ADDRESS_SHIFT)
#define MPEC2B_QBUF_ADDR_ADDRESS_RANGE                  (5):(0)
#define MPEC2B_QBUF_ADDR_ADDRESS_ROW                    0


// Packet MPEC2B_MB_DONE_PARAM
#define MPEC2B_MB_DONE_PARAM_SIZE 30

#define MPEC2B_MB_DONE_PARAM_REWIND_TYPE_SHIFT                  (0)
#define MPEC2B_MB_DONE_PARAM_REWIND_TYPE_FIELD                  ((0x3) << MPEC2B_MB_DONE_PARAM_REWIND_TYPE_SHIFT)
#define MPEC2B_MB_DONE_PARAM_REWIND_TYPE_RANGE                  (1):(0)
#define MPEC2B_MB_DONE_PARAM_REWIND_TYPE_ROW                    0
#define MPEC2B_MB_DONE_PARAM_REWIND_TYPE_NOREWIND                       (0)
#define MPEC2B_MB_DONE_PARAM_REWIND_TYPE_IPCM                   (1)
#define MPEC2B_MB_DONE_PARAM_REWIND_TYPE_PACKETBASE                     (2)
#define MPEC2B_MB_DONE_PARAM_REWIND_TYPE_BOTH                   (3)

#define MPEC2B_MB_DONE_PARAM_LWRXRC_SHIFT                       (2)
#define MPEC2B_MB_DONE_PARAM_LWRXRC_FIELD                       ((0xff) << MPEC2B_MB_DONE_PARAM_LWRXRC_SHIFT)
#define MPEC2B_MB_DONE_PARAM_LWRXRC_RANGE                       (9):(2)
#define MPEC2B_MB_DONE_PARAM_LWRXRC_ROW                 0

#define MPEC2B_MB_DONE_PARAM_LWRYRC_SHIFT                       (10)
#define MPEC2B_MB_DONE_PARAM_LWRYRC_FIELD                       ((0xff) << MPEC2B_MB_DONE_PARAM_LWRYRC_SHIFT)
#define MPEC2B_MB_DONE_PARAM_LWRYRC_RANGE                       (17):(10)
#define MPEC2B_MB_DONE_PARAM_LWRYRC_ROW                 0

#define MPEC2B_MB_DONE_PARAM_SLICE_GRPID_SHIFT                  (18)
#define MPEC2B_MB_DONE_PARAM_SLICE_GRPID_FIELD                  ((0x7) << MPEC2B_MB_DONE_PARAM_SLICE_GRPID_SHIFT)
#define MPEC2B_MB_DONE_PARAM_SLICE_GRPID_RANGE                  (20):(18)
#define MPEC2B_MB_DONE_PARAM_SLICE_GRPID_ROW                    0

#define MPEC2B_MB_DONE_PARAM_LWRQPRC_SHIFT                      (21)
#define MPEC2B_MB_DONE_PARAM_LWRQPRC_FIELD                      ((0x3f) << MPEC2B_MB_DONE_PARAM_LWRQPRC_SHIFT)
#define MPEC2B_MB_DONE_PARAM_LWRQPRC_RANGE                      (26):(21)
#define MPEC2B_MB_DONE_PARAM_LWRQPRC_ROW                        0

#define MPEC2B_MB_DONE_PARAM_NEW_PKT_SHIFT                      (27)
#define MPEC2B_MB_DONE_PARAM_NEW_PKT_FIELD                      ((0x1) << MPEC2B_MB_DONE_PARAM_NEW_PKT_SHIFT)
#define MPEC2B_MB_DONE_PARAM_NEW_PKT_RANGE                      (27):(27)
#define MPEC2B_MB_DONE_PARAM_NEW_PKT_ROW                        0

#define MPEC2B_MB_DONE_PARAM_LEFT_NB_AVAIL_SHIFT                        (28)
#define MPEC2B_MB_DONE_PARAM_LEFT_NB_AVAIL_FIELD                        ((0x1) << MPEC2B_MB_DONE_PARAM_LEFT_NB_AVAIL_SHIFT)
#define MPEC2B_MB_DONE_PARAM_LEFT_NB_AVAIL_RANGE                        (28):(28)
#define MPEC2B_MB_DONE_PARAM_LEFT_NB_AVAIL_ROW                  0

#define MPEC2B_MB_DONE_PARAM_TOP_NB_AVAIL_SHIFT                 (29)
#define MPEC2B_MB_DONE_PARAM_TOP_NB_AVAIL_FIELD                 ((0x1) << MPEC2B_MB_DONE_PARAM_TOP_NB_AVAIL_SHIFT)
#define MPEC2B_MB_DONE_PARAM_TOP_NB_AVAIL_RANGE                 (29):(29)
#define MPEC2B_MB_DONE_PARAM_TOP_NB_AVAIL_ROW                   0


// Packet MPEB2C_MB_START_PARAM
#define MPEB2C_MB_START_PARAM_SIZE 572

#define MPEB2C_MB_START_PARAM_MBTYPE_SHIFT                      (0)
#define MPEB2C_MB_START_PARAM_MBTYPE_FIELD                      ((0x3) << MPEB2C_MB_START_PARAM_MBTYPE_SHIFT)
#define MPEB2C_MB_START_PARAM_MBTYPE_RANGE                      (1):(0)
#define MPEB2C_MB_START_PARAM_MBTYPE_ROW                        0
#define MPEB2C_MB_START_PARAM_MBTYPE_I                  (0)
#define MPEB2C_MB_START_PARAM_MBTYPE_P                  (1)
#define MPEB2C_MB_START_PARAM_MBTYPE_IPCM                       (2)
#define MPEB2C_MB_START_PARAM_MBTYPE_SKIPSUG                    (3)

#define MPEB2C_MB_START_PARAM_MVMODE_SHIFT                      (2)
#define MPEB2C_MB_START_PARAM_MVMODE_FIELD                      ((0x3) << MPEB2C_MB_START_PARAM_MVMODE_SHIFT)
#define MPEB2C_MB_START_PARAM_MVMODE_RANGE                      (3):(2)
#define MPEB2C_MB_START_PARAM_MVMODE_ROW                        0
#define MPEB2C_MB_START_PARAM_MVMODE_m16x16                     (0)
#define MPEB2C_MB_START_PARAM_MVMODE_m8x8                       (1)
#define MPEB2C_MB_START_PARAM_MVMODE_m16x8                      (2)
#define MPEB2C_MB_START_PARAM_MVMODE_m8x16                      (3)

#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_0_SHIFT                    (4)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_0_FIELD                    ((0x3) << MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_0_SHIFT)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_0_RANGE                    (5):(4)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_0_ROW                      0
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_0_s8x8                     (0)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_0_s8x4                     (1)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_0_s4x8                     (2)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_0_s4x4                     (3)

#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_1_SHIFT                    (6)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_1_FIELD                    ((0x3) << MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_1_SHIFT)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_1_RANGE                    (7):(6)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_1_ROW                      0
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_1_s8x8                     (0)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_1_s8x4                     (1)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_1_s4x8                     (2)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_1_s4x4                     (3)

#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_2_SHIFT                    (8)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_2_FIELD                    ((0x3) << MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_2_SHIFT)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_2_RANGE                    (9):(8)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_2_ROW                      0
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_2_s8x8                     (0)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_2_s8x4                     (1)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_2_s4x8                     (2)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_2_s4x4                     (3)

#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_3_SHIFT                    (10)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_3_FIELD                    ((0x3) << MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_3_SHIFT)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_3_RANGE                    (11):(10)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_3_ROW                      0
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_3_s8x8                     (0)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_3_s8x4                     (1)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_3_s4x8                     (2)
#define MPEB2C_MB_START_PARAM_MVMODE_SUBTYPE_3_s4x4                     (3)

#define MPEB2C_MB_START_PARAM_LWRQP_SHIFT                       (12)
#define MPEB2C_MB_START_PARAM_LWRQP_FIELD                       ((0x3f) << MPEB2C_MB_START_PARAM_LWRQP_SHIFT)
#define MPEB2C_MB_START_PARAM_LWRQP_RANGE                       (17):(12)
#define MPEB2C_MB_START_PARAM_LWRQP_ROW                 0

#define MPEB2C_MB_START_PARAM_MBX_SHIFT                 (18)
#define MPEB2C_MB_START_PARAM_MBX_FIELD                 ((0xff) << MPEB2C_MB_START_PARAM_MBX_SHIFT)
#define MPEB2C_MB_START_PARAM_MBX_RANGE                 (25):(18)
#define MPEB2C_MB_START_PARAM_MBX_ROW                   0

#define MPEB2C_MB_START_PARAM_MBY_SHIFT                 (26)
#define MPEB2C_MB_START_PARAM_MBY_FIELD                 ((0xff) << MPEB2C_MB_START_PARAM_MBY_SHIFT)
#define MPEB2C_MB_START_PARAM_MBY_RANGE                 (33):(26)
#define MPEB2C_MB_START_PARAM_MBY_ROW                   0

#define MPEB2C_MB_START_PARAM_TOPLEFT_MVX_SHIFT                 (34)
#define MPEB2C_MB_START_PARAM_TOPLEFT_MVX_FIELD                 ((0x3ff) << MPEB2C_MB_START_PARAM_TOPLEFT_MVX_SHIFT)
#define MPEB2C_MB_START_PARAM_TOPLEFT_MVX_RANGE                 (43):(34)
#define MPEB2C_MB_START_PARAM_TOPLEFT_MVX_ROW                   0

#define MPEB2C_MB_START_PARAM_TOPLEFT_MVY_SHIFT                 (44)
#define MPEB2C_MB_START_PARAM_TOPLEFT_MVY_FIELD                 ((0x3ff) << MPEB2C_MB_START_PARAM_TOPLEFT_MVY_SHIFT)
#define MPEB2C_MB_START_PARAM_TOPLEFT_MVY_RANGE                 (53):(44)
#define MPEB2C_MB_START_PARAM_TOPLEFT_MVY_ROW                   0

#define MPEB2C_MB_START_PARAM_TOP0_MVX_SHIFT                    (54)
#define MPEB2C_MB_START_PARAM_TOP0_MVX_FIELD                    ((0x3ff) << MPEB2C_MB_START_PARAM_TOP0_MVX_SHIFT)
#define MPEB2C_MB_START_PARAM_TOP0_MVX_RANGE                    (63):(54)
#define MPEB2C_MB_START_PARAM_TOP0_MVX_ROW                      0

#define MPEB2C_MB_START_PARAM_TOP0_MVY_SHIFT                    (64)
#define MPEB2C_MB_START_PARAM_TOP0_MVY_FIELD                    ((0x3ff) << MPEB2C_MB_START_PARAM_TOP0_MVY_SHIFT)
#define MPEB2C_MB_START_PARAM_TOP0_MVY_RANGE                    (73):(64)
#define MPEB2C_MB_START_PARAM_TOP0_MVY_ROW                      0

#define MPEB2C_MB_START_PARAM_TOP1_MVX_SHIFT                    (74)
#define MPEB2C_MB_START_PARAM_TOP1_MVX_FIELD                    ((0x3ff) << MPEB2C_MB_START_PARAM_TOP1_MVX_SHIFT)
#define MPEB2C_MB_START_PARAM_TOP1_MVX_RANGE                    (83):(74)
#define MPEB2C_MB_START_PARAM_TOP1_MVX_ROW                      0

#define MPEB2C_MB_START_PARAM_TOP1_MVY_SHIFT                    (84)
#define MPEB2C_MB_START_PARAM_TOP1_MVY_FIELD                    ((0x3ff) << MPEB2C_MB_START_PARAM_TOP1_MVY_SHIFT)
#define MPEB2C_MB_START_PARAM_TOP1_MVY_RANGE                    (93):(84)
#define MPEB2C_MB_START_PARAM_TOP1_MVY_ROW                      0

#define MPEB2C_MB_START_PARAM_TOP2_MVX_SHIFT                    (94)
#define MPEB2C_MB_START_PARAM_TOP2_MVX_FIELD                    ((0x3ff) << MPEB2C_MB_START_PARAM_TOP2_MVX_SHIFT)
#define MPEB2C_MB_START_PARAM_TOP2_MVX_RANGE                    (103):(94)
#define MPEB2C_MB_START_PARAM_TOP2_MVX_ROW                      0

#define MPEB2C_MB_START_PARAM_TOP2_MVY_SHIFT                    (104)
#define MPEB2C_MB_START_PARAM_TOP2_MVY_FIELD                    ((0x3ff) << MPEB2C_MB_START_PARAM_TOP2_MVY_SHIFT)
#define MPEB2C_MB_START_PARAM_TOP2_MVY_RANGE                    (113):(104)
#define MPEB2C_MB_START_PARAM_TOP2_MVY_ROW                      0

#define MPEB2C_MB_START_PARAM_TOP3_MVX_SHIFT                    (114)
#define MPEB2C_MB_START_PARAM_TOP3_MVX_FIELD                    ((0x3ff) << MPEB2C_MB_START_PARAM_TOP3_MVX_SHIFT)
#define MPEB2C_MB_START_PARAM_TOP3_MVX_RANGE                    (123):(114)
#define MPEB2C_MB_START_PARAM_TOP3_MVX_ROW                      0

#define MPEB2C_MB_START_PARAM_TOP3_MVY_SHIFT                    (124)
#define MPEB2C_MB_START_PARAM_TOP3_MVY_FIELD                    ((0x3ff) << MPEB2C_MB_START_PARAM_TOP3_MVY_SHIFT)
#define MPEB2C_MB_START_PARAM_TOP3_MVY_RANGE                    (133):(124)
#define MPEB2C_MB_START_PARAM_TOP3_MVY_ROW                      0

#define MPEB2C_MB_START_PARAM_TOPRIGHT_MVX_SHIFT                        (134)
#define MPEB2C_MB_START_PARAM_TOPRIGHT_MVX_FIELD                        ((0x3ff) << MPEB2C_MB_START_PARAM_TOPRIGHT_MVX_SHIFT)
#define MPEB2C_MB_START_PARAM_TOPRIGHT_MVX_RANGE                        (143):(134)
#define MPEB2C_MB_START_PARAM_TOPRIGHT_MVX_ROW                  0

#define MPEB2C_MB_START_PARAM_TOPRIGHT_MVY_SHIFT                        (144)
#define MPEB2C_MB_START_PARAM_TOPRIGHT_MVY_FIELD                        ((0x3ff) << MPEB2C_MB_START_PARAM_TOPRIGHT_MVY_SHIFT)
#define MPEB2C_MB_START_PARAM_TOPRIGHT_MVY_RANGE                        (153):(144)
#define MPEB2C_MB_START_PARAM_TOPRIGHT_MVY_ROW                  0

#define MPEB2C_MB_START_PARAM_LEFT_SLICE_GRPID_SHIFT                    (154)
#define MPEB2C_MB_START_PARAM_LEFT_SLICE_GRPID_FIELD                    ((0x7) << MPEB2C_MB_START_PARAM_LEFT_SLICE_GRPID_SHIFT)
#define MPEB2C_MB_START_PARAM_LEFT_SLICE_GRPID_RANGE                    (156):(154)
#define MPEB2C_MB_START_PARAM_LEFT_SLICE_GRPID_ROW                      0

#define MPEB2C_MB_START_PARAM_TOPLEFT_SLICE_GRPID_SHIFT                 (157)
#define MPEB2C_MB_START_PARAM_TOPLEFT_SLICE_GRPID_FIELD                 ((0x7) << MPEB2C_MB_START_PARAM_TOPLEFT_SLICE_GRPID_SHIFT)
#define MPEB2C_MB_START_PARAM_TOPLEFT_SLICE_GRPID_RANGE                 (159):(157)
#define MPEB2C_MB_START_PARAM_TOPLEFT_SLICE_GRPID_ROW                   0

#define MPEB2C_MB_START_PARAM_TOP_SLICE_GRPID_SHIFT                     (160)
#define MPEB2C_MB_START_PARAM_TOP_SLICE_GRPID_FIELD                     ((0x7) << MPEB2C_MB_START_PARAM_TOP_SLICE_GRPID_SHIFT)
#define MPEB2C_MB_START_PARAM_TOP_SLICE_GRPID_RANGE                     (162):(160)
#define MPEB2C_MB_START_PARAM_TOP_SLICE_GRPID_ROW                       0

#define MPEB2C_MB_START_PARAM_TOPRIGHT_SLICE_GRPID_SHIFT                        (163)
#define MPEB2C_MB_START_PARAM_TOPRIGHT_SLICE_GRPID_FIELD                        ((0x7) << MPEB2C_MB_START_PARAM_TOPRIGHT_SLICE_GRPID_SHIFT)
#define MPEB2C_MB_START_PARAM_TOPRIGHT_SLICE_GRPID_RANGE                        (165):(163)
#define MPEB2C_MB_START_PARAM_TOPRIGHT_SLICE_GRPID_ROW                  0

#define MPEB2C_MB_START_PARAM_LWR_SLICE_GRPID_SHIFT                     (166)
#define MPEB2C_MB_START_PARAM_LWR_SLICE_GRPID_FIELD                     ((0x7) << MPEB2C_MB_START_PARAM_LWR_SLICE_GRPID_SHIFT)
#define MPEB2C_MB_START_PARAM_LWR_SLICE_GRPID_RANGE                     (168):(166)
#define MPEB2C_MB_START_PARAM_LWR_SLICE_GRPID_ROW                       0

#define MPEB2C_MB_START_PARAM_INTRA_PART_PRED_MODE_SHIFT                        (169)
#define MPEB2C_MB_START_PARAM_INTRA_PART_PRED_MODE_FIELD                        ((0x1) << MPEB2C_MB_START_PARAM_INTRA_PART_PRED_MODE_SHIFT)
#define MPEB2C_MB_START_PARAM_INTRA_PART_PRED_MODE_RANGE                        (169):(169)
#define MPEB2C_MB_START_PARAM_INTRA_PART_PRED_MODE_ROW                  0
#define MPEB2C_MB_START_PARAM_INTRA_PART_PRED_MODE_I4x4                 (0)
#define MPEB2C_MB_START_PARAM_INTRA_PART_PRED_MODE_I16x16                       (1)

//block 0, sub-block 0
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_0_SHIFT                     (170)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_0_FIELD                     ((0x1) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_0_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_0_RANGE                     (170):(170)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_0_ROW                       0

//block 0, sub-block 1
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_1_SHIFT                     (171)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_1_FIELD                     ((0x1) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_1_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_1_RANGE                     (171):(171)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_1_ROW                       0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_2_SHIFT                     (172)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_2_FIELD                     ((0x1) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_2_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_2_RANGE                     (172):(172)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_2_ROW                       0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_3_SHIFT                     (173)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_3_FIELD                     ((0x1) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_3_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_3_RANGE                     (173):(173)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_0_3_ROW                       0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_0_SHIFT                     (174)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_0_FIELD                     ((0x1) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_0_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_0_RANGE                     (174):(174)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_0_ROW                       0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_1_SHIFT                     (175)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_1_FIELD                     ((0x1) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_1_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_1_RANGE                     (175):(175)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_1_ROW                       0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_2_SHIFT                     (176)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_2_FIELD                     ((0x1) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_2_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_2_RANGE                     (176):(176)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_2_ROW                       0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_3_SHIFT                     (177)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_3_FIELD                     ((0x1) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_3_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_3_RANGE                     (177):(177)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_1_3_ROW                       0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_0_SHIFT                     (178)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_0_FIELD                     ((0x1) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_0_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_0_RANGE                     (178):(178)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_0_ROW                       0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_1_SHIFT                     (179)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_1_FIELD                     ((0x1) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_1_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_1_RANGE                     (179):(179)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_1_ROW                       0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_2_SHIFT                     (180)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_2_FIELD                     ((0x1) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_2_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_2_RANGE                     (180):(180)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_2_ROW                       0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_3_SHIFT                     (181)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_3_FIELD                     ((0x1) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_3_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_3_RANGE                     (181):(181)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_2_3_ROW                       0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_0_SHIFT                     (182)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_0_FIELD                     ((0x1) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_0_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_0_RANGE                     (182):(182)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_0_ROW                       0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_1_SHIFT                     (183)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_1_FIELD                     ((0x1) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_1_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_1_RANGE                     (183):(183)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_1_ROW                       0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_2_SHIFT                     (184)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_2_FIELD                     ((0x1) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_2_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_2_RANGE                     (184):(184)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_2_ROW                       0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_3_SHIFT                     (185)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_3_FIELD                     ((0x1) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_3_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_3_RANGE                     (185):(185)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_FLAG_3_3_ROW                       0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_0_SHIFT                      (186)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_0_FIELD                      ((0x7) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_0_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_0_RANGE                      (188):(186)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_0_ROW                        0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_1_SHIFT                      (189)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_1_FIELD                      ((0x7) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_1_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_1_RANGE                      (191):(189)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_1_ROW                        0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_2_SHIFT                      (192)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_2_FIELD                      ((0x7) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_2_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_2_RANGE                      (194):(192)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_2_ROW                        0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_3_SHIFT                      (195)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_3_FIELD                      ((0x7) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_3_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_3_RANGE                      (197):(195)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_0_3_ROW                        0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_0_SHIFT                      (198)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_0_FIELD                      ((0x7) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_0_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_0_RANGE                      (200):(198)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_0_ROW                        0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_1_SHIFT                      (201)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_1_FIELD                      ((0x7) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_1_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_1_RANGE                      (203):(201)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_1_ROW                        0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_2_SHIFT                      (204)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_2_FIELD                      ((0x7) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_2_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_2_RANGE                      (206):(204)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_2_ROW                        0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_3_SHIFT                      (207)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_3_FIELD                      ((0x7) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_3_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_3_RANGE                      (209):(207)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_1_3_ROW                        0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_0_SHIFT                      (210)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_0_FIELD                      ((0x7) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_0_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_0_RANGE                      (212):(210)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_0_ROW                        0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_1_SHIFT                      (213)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_1_FIELD                      ((0x7) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_1_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_1_RANGE                      (215):(213)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_1_ROW                        0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_2_SHIFT                      (216)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_2_FIELD                      ((0x7) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_2_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_2_RANGE                      (218):(216)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_2_ROW                        0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_3_SHIFT                      (219)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_3_FIELD                      ((0x7) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_3_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_3_RANGE                      (221):(219)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_2_3_ROW                        0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_0_SHIFT                      (222)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_0_FIELD                      ((0x7) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_0_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_0_RANGE                      (224):(222)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_0_ROW                        0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_1_SHIFT                      (225)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_1_FIELD                      ((0x7) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_1_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_1_RANGE                      (227):(225)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_1_ROW                        0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_2_SHIFT                      (228)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_2_FIELD                      ((0x7) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_2_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_2_RANGE                      (230):(228)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_2_ROW                        0

#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_3_SHIFT                      (231)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_3_FIELD                      ((0x7) << MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_3_SHIFT)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_3_RANGE                      (233):(231)
#define MPEB2C_MB_START_PARAM_I4x4_PRED_MODE_REM_3_3_ROW                        0

// 2 bit, 00=V, 01=H, 10=DC, 11=Plane
#define MPEB2C_MB_START_PARAM_I16x16_PRED_MODE_TYPE_SHIFT                       (234)
#define MPEB2C_MB_START_PARAM_I16x16_PRED_MODE_TYPE_FIELD                       ((0x3) << MPEB2C_MB_START_PARAM_I16x16_PRED_MODE_TYPE_SHIFT)
#define MPEB2C_MB_START_PARAM_I16x16_PRED_MODE_TYPE_RANGE                       (235):(234)
#define MPEB2C_MB_START_PARAM_I16x16_PRED_MODE_TYPE_ROW                 0
#define MPEB2C_MB_START_PARAM_I16x16_PRED_MODE_TYPE_V                   (0)
#define MPEB2C_MB_START_PARAM_I16x16_PRED_MODE_TYPE_H                   (1)
#define MPEB2C_MB_START_PARAM_I16x16_PRED_MODE_TYPE_DC                  (2)
#define MPEB2C_MB_START_PARAM_I16x16_PRED_MODE_TYPE_P                   (3)

// 2 bit, 00=DC, 01=H, 10=V, 11=Plane 
#define MPEB2C_MB_START_PARAM_CHROMA_PRED_MODE_TYPE_SHIFT                       (236)
#define MPEB2C_MB_START_PARAM_CHROMA_PRED_MODE_TYPE_FIELD                       ((0x3) << MPEB2C_MB_START_PARAM_CHROMA_PRED_MODE_TYPE_SHIFT)
#define MPEB2C_MB_START_PARAM_CHROMA_PRED_MODE_TYPE_RANGE                       (237):(236)
#define MPEB2C_MB_START_PARAM_CHROMA_PRED_MODE_TYPE_ROW                 0
#define MPEB2C_MB_START_PARAM_CHROMA_PRED_MODE_TYPE_DC                  (0)
#define MPEB2C_MB_START_PARAM_CHROMA_PRED_MODE_TYPE_H                   (1)
#define MPEB2C_MB_START_PARAM_CHROMA_PRED_MODE_TYPE_V                   (2)
#define MPEB2C_MB_START_PARAM_CHROMA_PRED_MODE_TYPE_P                   (3)

#define MPEB2C_MB_START_PARAM_SLICE_TYPE_SHIFT                  (238)
#define MPEB2C_MB_START_PARAM_SLICE_TYPE_FIELD                  ((0x1) << MPEB2C_MB_START_PARAM_SLICE_TYPE_SHIFT)
#define MPEB2C_MB_START_PARAM_SLICE_TYPE_RANGE                  (238):(238)
#define MPEB2C_MB_START_PARAM_SLICE_TYPE_ROW                    0
#define MPEB2C_MB_START_PARAM_SLICE_TYPE_I                      (0)
#define MPEB2C_MB_START_PARAM_SLICE_TYPE_P                      (1)

// block = 0, sub-block = 0
#define MPEB2C_MB_START_PARAM_MVX_0_0_SHIFT                     (239)
#define MPEB2C_MB_START_PARAM_MVX_0_0_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVX_0_0_SHIFT)
#define MPEB2C_MB_START_PARAM_MVX_0_0_RANGE                     (248):(239)
#define MPEB2C_MB_START_PARAM_MVX_0_0_ROW                       0

#define MPEB2C_MB_START_PARAM_MVY_0_0_SHIFT                     (249)
#define MPEB2C_MB_START_PARAM_MVY_0_0_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVY_0_0_SHIFT)
#define MPEB2C_MB_START_PARAM_MVY_0_0_RANGE                     (258):(249)
#define MPEB2C_MB_START_PARAM_MVY_0_0_ROW                       0

// block = 0, sub-block = 1
#define MPEB2C_MB_START_PARAM_MVX_0_1_SHIFT                     (259)
#define MPEB2C_MB_START_PARAM_MVX_0_1_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVX_0_1_SHIFT)
#define MPEB2C_MB_START_PARAM_MVX_0_1_RANGE                     (268):(259)
#define MPEB2C_MB_START_PARAM_MVX_0_1_ROW                       0

#define MPEB2C_MB_START_PARAM_MVY_0_1_SHIFT                     (269)
#define MPEB2C_MB_START_PARAM_MVY_0_1_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVY_0_1_SHIFT)
#define MPEB2C_MB_START_PARAM_MVY_0_1_RANGE                     (278):(269)
#define MPEB2C_MB_START_PARAM_MVY_0_1_ROW                       0

#define MPEB2C_MB_START_PARAM_MVX_0_2_SHIFT                     (279)
#define MPEB2C_MB_START_PARAM_MVX_0_2_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVX_0_2_SHIFT)
#define MPEB2C_MB_START_PARAM_MVX_0_2_RANGE                     (288):(279)
#define MPEB2C_MB_START_PARAM_MVX_0_2_ROW                       0

#define MPEB2C_MB_START_PARAM_MVY_0_2_SHIFT                     (289)
#define MPEB2C_MB_START_PARAM_MVY_0_2_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVY_0_2_SHIFT)
#define MPEB2C_MB_START_PARAM_MVY_0_2_RANGE                     (298):(289)
#define MPEB2C_MB_START_PARAM_MVY_0_2_ROW                       0

#define MPEB2C_MB_START_PARAM_MVX_0_3_SHIFT                     (299)
#define MPEB2C_MB_START_PARAM_MVX_0_3_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVX_0_3_SHIFT)
#define MPEB2C_MB_START_PARAM_MVX_0_3_RANGE                     (308):(299)
#define MPEB2C_MB_START_PARAM_MVX_0_3_ROW                       0

#define MPEB2C_MB_START_PARAM_MVY_0_3_SHIFT                     (309)
#define MPEB2C_MB_START_PARAM_MVY_0_3_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVY_0_3_SHIFT)
#define MPEB2C_MB_START_PARAM_MVY_0_3_RANGE                     (318):(309)
#define MPEB2C_MB_START_PARAM_MVY_0_3_ROW                       0

// block = 1, sub-block = 0
#define MPEB2C_MB_START_PARAM_MVX_1_0_SHIFT                     (319)
#define MPEB2C_MB_START_PARAM_MVX_1_0_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVX_1_0_SHIFT)
#define MPEB2C_MB_START_PARAM_MVX_1_0_RANGE                     (328):(319)
#define MPEB2C_MB_START_PARAM_MVX_1_0_ROW                       0

#define MPEB2C_MB_START_PARAM_MVY_1_0_SHIFT                     (329)
#define MPEB2C_MB_START_PARAM_MVY_1_0_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVY_1_0_SHIFT)
#define MPEB2C_MB_START_PARAM_MVY_1_0_RANGE                     (338):(329)
#define MPEB2C_MB_START_PARAM_MVY_1_0_ROW                       0

// block = 1, sub-block = 1
#define MPEB2C_MB_START_PARAM_MVX_1_1_SHIFT                     (339)
#define MPEB2C_MB_START_PARAM_MVX_1_1_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVX_1_1_SHIFT)
#define MPEB2C_MB_START_PARAM_MVX_1_1_RANGE                     (348):(339)
#define MPEB2C_MB_START_PARAM_MVX_1_1_ROW                       0

#define MPEB2C_MB_START_PARAM_MVY_1_1_SHIFT                     (349)
#define MPEB2C_MB_START_PARAM_MVY_1_1_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVY_1_1_SHIFT)
#define MPEB2C_MB_START_PARAM_MVY_1_1_RANGE                     (358):(349)
#define MPEB2C_MB_START_PARAM_MVY_1_1_ROW                       0

#define MPEB2C_MB_START_PARAM_MVX_1_2_SHIFT                     (359)
#define MPEB2C_MB_START_PARAM_MVX_1_2_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVX_1_2_SHIFT)
#define MPEB2C_MB_START_PARAM_MVX_1_2_RANGE                     (368):(359)
#define MPEB2C_MB_START_PARAM_MVX_1_2_ROW                       0

#define MPEB2C_MB_START_PARAM_MVY_1_2_SHIFT                     (369)
#define MPEB2C_MB_START_PARAM_MVY_1_2_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVY_1_2_SHIFT)
#define MPEB2C_MB_START_PARAM_MVY_1_2_RANGE                     (378):(369)
#define MPEB2C_MB_START_PARAM_MVY_1_2_ROW                       0

#define MPEB2C_MB_START_PARAM_MVX_1_3_SHIFT                     (379)
#define MPEB2C_MB_START_PARAM_MVX_1_3_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVX_1_3_SHIFT)
#define MPEB2C_MB_START_PARAM_MVX_1_3_RANGE                     (388):(379)
#define MPEB2C_MB_START_PARAM_MVX_1_3_ROW                       0

#define MPEB2C_MB_START_PARAM_MVY_1_3_SHIFT                     (389)
#define MPEB2C_MB_START_PARAM_MVY_1_3_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVY_1_3_SHIFT)
#define MPEB2C_MB_START_PARAM_MVY_1_3_RANGE                     (398):(389)
#define MPEB2C_MB_START_PARAM_MVY_1_3_ROW                       0

// block = 2, sub-block = 0
#define MPEB2C_MB_START_PARAM_MVX_2_0_SHIFT                     (399)
#define MPEB2C_MB_START_PARAM_MVX_2_0_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVX_2_0_SHIFT)
#define MPEB2C_MB_START_PARAM_MVX_2_0_RANGE                     (408):(399)
#define MPEB2C_MB_START_PARAM_MVX_2_0_ROW                       0

#define MPEB2C_MB_START_PARAM_MVY_2_0_SHIFT                     (409)
#define MPEB2C_MB_START_PARAM_MVY_2_0_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVY_2_0_SHIFT)
#define MPEB2C_MB_START_PARAM_MVY_2_0_RANGE                     (418):(409)
#define MPEB2C_MB_START_PARAM_MVY_2_0_ROW                       0

// block = 2, sub-block = 1
#define MPEB2C_MB_START_PARAM_MVX_2_1_SHIFT                     (419)
#define MPEB2C_MB_START_PARAM_MVX_2_1_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVX_2_1_SHIFT)
#define MPEB2C_MB_START_PARAM_MVX_2_1_RANGE                     (428):(419)
#define MPEB2C_MB_START_PARAM_MVX_2_1_ROW                       0

#define MPEB2C_MB_START_PARAM_MVY_2_1_SHIFT                     (429)
#define MPEB2C_MB_START_PARAM_MVY_2_1_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVY_2_1_SHIFT)
#define MPEB2C_MB_START_PARAM_MVY_2_1_RANGE                     (438):(429)
#define MPEB2C_MB_START_PARAM_MVY_2_1_ROW                       0

#define MPEB2C_MB_START_PARAM_MVX_2_2_SHIFT                     (439)
#define MPEB2C_MB_START_PARAM_MVX_2_2_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVX_2_2_SHIFT)
#define MPEB2C_MB_START_PARAM_MVX_2_2_RANGE                     (448):(439)
#define MPEB2C_MB_START_PARAM_MVX_2_2_ROW                       0

#define MPEB2C_MB_START_PARAM_MVY_2_2_SHIFT                     (449)
#define MPEB2C_MB_START_PARAM_MVY_2_2_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVY_2_2_SHIFT)
#define MPEB2C_MB_START_PARAM_MVY_2_2_RANGE                     (458):(449)
#define MPEB2C_MB_START_PARAM_MVY_2_2_ROW                       0

#define MPEB2C_MB_START_PARAM_MVX_2_3_SHIFT                     (459)
#define MPEB2C_MB_START_PARAM_MVX_2_3_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVX_2_3_SHIFT)
#define MPEB2C_MB_START_PARAM_MVX_2_3_RANGE                     (468):(459)
#define MPEB2C_MB_START_PARAM_MVX_2_3_ROW                       0

#define MPEB2C_MB_START_PARAM_MVY_2_3_SHIFT                     (469)
#define MPEB2C_MB_START_PARAM_MVY_2_3_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVY_2_3_SHIFT)
#define MPEB2C_MB_START_PARAM_MVY_2_3_RANGE                     (478):(469)
#define MPEB2C_MB_START_PARAM_MVY_2_3_ROW                       0

// block = 3, sub-block = 0
#define MPEB2C_MB_START_PARAM_MVX_3_0_SHIFT                     (479)
#define MPEB2C_MB_START_PARAM_MVX_3_0_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVX_3_0_SHIFT)
#define MPEB2C_MB_START_PARAM_MVX_3_0_RANGE                     (488):(479)
#define MPEB2C_MB_START_PARAM_MVX_3_0_ROW                       0

#define MPEB2C_MB_START_PARAM_MVY_3_0_SHIFT                     (489)
#define MPEB2C_MB_START_PARAM_MVY_3_0_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVY_3_0_SHIFT)
#define MPEB2C_MB_START_PARAM_MVY_3_0_RANGE                     (498):(489)
#define MPEB2C_MB_START_PARAM_MVY_3_0_ROW                       0

// block = 3, sub-block = 1
#define MPEB2C_MB_START_PARAM_MVX_3_1_SHIFT                     (499)
#define MPEB2C_MB_START_PARAM_MVX_3_1_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVX_3_1_SHIFT)
#define MPEB2C_MB_START_PARAM_MVX_3_1_RANGE                     (508):(499)
#define MPEB2C_MB_START_PARAM_MVX_3_1_ROW                       0

#define MPEB2C_MB_START_PARAM_MVY_3_1_SHIFT                     (509)
#define MPEB2C_MB_START_PARAM_MVY_3_1_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVY_3_1_SHIFT)
#define MPEB2C_MB_START_PARAM_MVY_3_1_RANGE                     (518):(509)
#define MPEB2C_MB_START_PARAM_MVY_3_1_ROW                       0

#define MPEB2C_MB_START_PARAM_MVX_3_2_SHIFT                     (519)
#define MPEB2C_MB_START_PARAM_MVX_3_2_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVX_3_2_SHIFT)
#define MPEB2C_MB_START_PARAM_MVX_3_2_RANGE                     (528):(519)
#define MPEB2C_MB_START_PARAM_MVX_3_2_ROW                       0

#define MPEB2C_MB_START_PARAM_MVY_3_2_SHIFT                     (529)
#define MPEB2C_MB_START_PARAM_MVY_3_2_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVY_3_2_SHIFT)
#define MPEB2C_MB_START_PARAM_MVY_3_2_RANGE                     (538):(529)
#define MPEB2C_MB_START_PARAM_MVY_3_2_ROW                       0

#define MPEB2C_MB_START_PARAM_MVX_3_3_SHIFT                     (539)
#define MPEB2C_MB_START_PARAM_MVX_3_3_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVX_3_3_SHIFT)
#define MPEB2C_MB_START_PARAM_MVX_3_3_RANGE                     (548):(539)
#define MPEB2C_MB_START_PARAM_MVX_3_3_ROW                       0

#define MPEB2C_MB_START_PARAM_MVY_3_3_SHIFT                     (549)
#define MPEB2C_MB_START_PARAM_MVY_3_3_FIELD                     ((0x3ff) << MPEB2C_MB_START_PARAM_MVY_3_3_SHIFT)
#define MPEB2C_MB_START_PARAM_MVY_3_3_RANGE                     (558):(549)
#define MPEB2C_MB_START_PARAM_MVY_3_3_ROW                       0

#define MPEB2C_MB_START_PARAM_DEQUAN_SHIFT                      (559)
#define MPEB2C_MB_START_PARAM_DEQUAN_FIELD                      ((0x7) << MPEB2C_MB_START_PARAM_DEQUAN_SHIFT)
#define MPEB2C_MB_START_PARAM_DEQUAN_RANGE                      (561):(559)
#define MPEB2C_MB_START_PARAM_DEQUAN_ROW                        0

// First MB in slice-valid in macro block based packetization
#define MPEB2C_MB_START_PARAM_FIRSTMBINSLICE_SHIFT                      (562)
#define MPEB2C_MB_START_PARAM_FIRSTMBINSLICE_FIELD                      ((0x1) << MPEB2C_MB_START_PARAM_FIRSTMBINSLICE_SHIFT)
#define MPEB2C_MB_START_PARAM_FIRSTMBINSLICE_RANGE                      (562):(562)
#define MPEB2C_MB_START_PARAM_FIRSTMBINSLICE_ROW                        0

// Last MB in slice grp-valid in macro block based packetization
#define MPEB2C_MB_START_PARAM_LASTMBINSLICEGRP_SHIFT                    (563)
#define MPEB2C_MB_START_PARAM_LASTMBINSLICEGRP_FIELD                    ((0x1) << MPEB2C_MB_START_PARAM_LASTMBINSLICEGRP_SHIFT)
#define MPEB2C_MB_START_PARAM_LASTMBINSLICEGRP_RANGE                    (563):(563)
#define MPEB2C_MB_START_PARAM_LASTMBINSLICEGRP_ROW                      0

// 4bit,h264
#define MPEB2C_MB_START_PARAM_SLICE_ALPHA_C0_OFFSET_DIV2_SHIFT                  (564)
#define MPEB2C_MB_START_PARAM_SLICE_ALPHA_C0_OFFSET_DIV2_FIELD                  ((0xf) << MPEB2C_MB_START_PARAM_SLICE_ALPHA_C0_OFFSET_DIV2_SHIFT)
#define MPEB2C_MB_START_PARAM_SLICE_ALPHA_C0_OFFSET_DIV2_RANGE                  (567):(564)
#define MPEB2C_MB_START_PARAM_SLICE_ALPHA_C0_OFFSET_DIV2_ROW                    0

// 4bit,h264
#define MPEB2C_MB_START_PARAM_SLICE_BETA_OFFSET_DIV2_SHIFT                      (568)
#define MPEB2C_MB_START_PARAM_SLICE_BETA_OFFSET_DIV2_FIELD                      ((0xf) << MPEB2C_MB_START_PARAM_SLICE_BETA_OFFSET_DIV2_SHIFT)
#define MPEB2C_MB_START_PARAM_SLICE_BETA_OFFSET_DIV2_RANGE                      (571):(568)
#define MPEB2C_MB_START_PARAM_SLICE_BETA_OFFSET_DIV2_ROW                        0

// --------------------------------------------------------------------------
// 
// Copyright (c) 2004, LWPU Corp.
// All Rights Reserved.
// 
// This is UNPUBLISHED PROPRIETARY SOURCE CODE of LWPU Corp.;
// the contents of this file may not be disclosed to third parties, copied or
// duplicated in any form, in whole or in part, without the prior written
// permission of LWPU Corp.
// 
// RESTRICTED RIGHTS LEGEND:
// Use, duplication or disclosure by the Government is subject to restrictions
// as set forth in subdivision (c)(1)(ii) of the Rights in Technical Data
// and Computer Software clause at DFARS 252.227-7013, and/or in similar or
// successor clauses in the FAR, DOD or NASA FAR Supplement. Unpublished -
// rights reserved under the Copyright Laws of the United States.
// 
// --------------------------------------------------------------------------
// 
// -------------------------------------------------------------------
// FIFO parameters (ME legacy)
// -------------------------------------------------------------------
// Host IF FIFOs
#define LW_MPEA_HOST_RDFIFO_DEPTH       5
#define LW_MPEA_HOST_RDFIFO_WIDTH       38
#define LW_MPEA_HOST_WRFIFO_DEPTH       5
#define LW_MPEA_HOST_WRFIFO_WIDTH       55
//this is the depth/width of the fifo used to queue up "new buffer" commands
#define LW_MPEA_IB_FIFO_DEPTH   8
#define LW_MPEA_IB_FIFO_WIDTH   10
#define LW_MPEA_IB_NEW_BUFFER_SIZE      9
#define LW_MPEA_FETCH_SM_STATE_VECTOR_SZ        3
#define LW_MPEA_XFR_SM_STATE_VECTOR_SZ  2
#define LW_MPEA_NGBR_RAM_W      82
// CBR RC Input Bit Width
#define BITS_PER_FRAME_SZ       28
#define BUPFMAD_DO_SZ   24
#define BU_NUM_SZ       10
#define LWRRENT_BUFFER_FULLNESS_SZ      28
#define LWRRENT_FRAME_AVG_MAD_SZ        30
// define LWRRENT_FRAME_BITS_I_SZ                  32;
#define LWRRENT_FRAME_BITS_SZ   25
#define LWRRENT_FRAME_MAD_SZ    26
#define LWRRENT_FRAME_TEXT_BITS_MPEC_SZ 28
#define DDQUANT_SZ      6
#define GOP_OVERDUE_SZ  1
#define GOV_PERIOD_SZ   16
#define INITIAL_DELAY_OFFSET_SZ 28
#define INITIAL_QP_SZ   6
#define INTRA_INTER_SAD_SZ      17
#define LOWER_BOUND_SZ  33
#define MAD_PIC_1_SZ    21
#define MAX_DQP_FROM_PREV_FRM_SZ        6
#define MAX_QP_SZ       6
#define MIN_QP_SZ       6
#define MX1_SZ  21
#define MY_INITIAL_QP_SZ        6
#define NUMBER_OF_BASIC_UNIT_HEADER_BITS_SZ     19
#define NUMBER_OF_BASIC_UNIT_SZ 10
#define NUMBER_OF_BASIC_UNIT_MPEG4_SZ   13
#define NUMBER_OF_BASIC_UNIT_TEXTURE_BITS_SZ    22
#define NUMBER_OF_GOP_SZ        16
#define NUMBER_OF_HEADER_BITS_SZ        25
#define NUM_CODED_FRAMES_I_SZ   26
#define NUM_CODED_FRAMES_SZ     26
#define NUM_CODED_P_FRAMES_SZ   26
#define NUM_P_PICTURE_SZ        16
#define PF_BU_MB_MAD_SZ 17
#define PIC_SIZE_IN_MBS_SZ      13
#define PIC_WIDTH_IN_MBS_SZ     7
#define PREV_FRAME_MAD_SZ       17
#define P_AVE_FRAME_QP_SZ       6
#define P_AVE_HEADER_BITS_1_SZ  25
#define P_AVE_HEADER_BITS_2_SZ  25
#define P_AVE_HEADER_BITS_3_SZ  25
#define QC_SZ   6
// define QP.*_SZ                                  6;
#define QP_SZ   6
#define QP_TOTAL_SZ     19
#define RC_BASIC_UNIT_SIZE_SZ   7
#define REMAINING_BITS_SZ       33
#define REMAINING_P_FRAMES_SZ   16
#define SLICE_TYPE_SZ   1
#define TARGET_BITS_SZ  33
#define TARGET_BUFFER_LEVEL_FLAG_SZ     1
#define TOTAL_FRAME_MAD_SZ      30
#define TOTAL_FRAME_QP_SZ       19
#define TOTAL_MAD_BASIC_UNIT_SZ 24
#define TOTAL_NUM_OF_BASIC_UNIT_SZ      10
#define TOTAL_QP_FOR_P_PICTURE_SZ       22
#define UPM_CNT_SZ      13
#define UPPER_BOUND1_SZ 33
#define UPPER_BOUND2_SZ 33
// PACE_RC WIDTH defines for internal variables
// GetQP
#define TARGET_BITS_DELTA_WIDTH 22
#define BUFFER_FULLNESS_WIDTH   29
#define P_AVERAGE_QP_WIDTH      6
#define INT_QP_WIDTH    8
#define LWR_BU_MAD_WIDTH        24
#define TARGETBITS_TEMP_WIDTH   22
#define TOTALBUMAD_TEMP_WIDTH   12
// InitGOP
#define QP_LAST_FRAME_WIDTH     6
// other funcs
#define QP_STEP_SZ      13
#define M_RP_SZ 32
#define WINDOW_SIZE_SZ  5
#define REAL_WINDOW_SIZE_SZ     5
#define ERROR_SZ        23
#define A00_SZ  11
#define A01_SZ  22
#define A11_SZ  32
#define B0_SZ   36
#define B1_SZ   32
#define AB_VALUE_SZ     32
// --------------------------------------------------------------------------
// 
// Copyright (c) 2004, LWPU Corp.
// All Rights Reserved.
// 
// This is UNPUBLISHED PROPRIETARY SOURCE CODE of LWPU Corp.;
// the contents of this file may not be disclosed to third parties, copied or
// duplicated in any form, in whole or in part, without the prior written
// permission of LWPU Corp.
// 
// RESTRICTED RIGHTS LEGEND:
// Use, duplication or disclosure by the Government is subject to restrictions
// as set forth in subdivision (c)(1)(ii) of the Rights in Technical Data
// and Computer Software clause at DFARS 252.227-7013, and/or in similar or
// successor clauses in the FAR, DOD or NASA FAR Supplement. Unpublished -
// rights reserved under the Copyright Laws of the United States.
// 
// --------------------------------------------------------------------------
// 

// Packet DBLK_PARAM_BUF
#define DBLK_PARAM_BUF_SIZE 128

#define DBLK_PARAM_BUF_MBTYPE_SHIFT                     (0)
#define DBLK_PARAM_BUF_MBTYPE_FIELD                     ((0x3) << DBLK_PARAM_BUF_MBTYPE_SHIFT)
#define DBLK_PARAM_BUF_MBTYPE_RANGE                     (1):(0)
#define DBLK_PARAM_BUF_MBTYPE_ROW                       0
#define DBLK_PARAM_BUF_MBTYPE_I                 (0)
#define DBLK_PARAM_BUF_MBTYPE_P                 (1)
#define DBLK_PARAM_BUF_MBTYPE_IPCM                      (2)
#define DBLK_PARAM_BUF_MBTYPE_SKIPSUG                   (3)

#define DBLK_PARAM_BUF_BLKTYPE_15_SHIFT                 (2)
#define DBLK_PARAM_BUF_BLKTYPE_15_FIELD                 ((0x1) << DBLK_PARAM_BUF_BLKTYPE_15_SHIFT)
#define DBLK_PARAM_BUF_BLKTYPE_15_RANGE                 (2):(2)
#define DBLK_PARAM_BUF_BLKTYPE_15_ROW                   0
#define DBLK_PARAM_BUF_BLKTYPE_15_INTER                 (0)
#define DBLK_PARAM_BUF_BLKTYPE_15_INTRA                 (1)

#define DBLK_PARAM_BUF_BLKTYPE_14_SHIFT                 (3)
#define DBLK_PARAM_BUF_BLKTYPE_14_FIELD                 ((0x1) << DBLK_PARAM_BUF_BLKTYPE_14_SHIFT)
#define DBLK_PARAM_BUF_BLKTYPE_14_RANGE                 (3):(3)
#define DBLK_PARAM_BUF_BLKTYPE_14_ROW                   0
#define DBLK_PARAM_BUF_BLKTYPE_14_INTER                 (0)
#define DBLK_PARAM_BUF_BLKTYPE_14_INTRA                 (1)

#define DBLK_PARAM_BUF_BLKTYPE_11_SHIFT                 (4)
#define DBLK_PARAM_BUF_BLKTYPE_11_FIELD                 ((0x1) << DBLK_PARAM_BUF_BLKTYPE_11_SHIFT)
#define DBLK_PARAM_BUF_BLKTYPE_11_RANGE                 (4):(4)
#define DBLK_PARAM_BUF_BLKTYPE_11_ROW                   0
#define DBLK_PARAM_BUF_BLKTYPE_11_INTER                 (0)
#define DBLK_PARAM_BUF_BLKTYPE_11_INTRA                 (1)

#define DBLK_PARAM_BUF_BLKTYPE_10_SHIFT                 (5)
#define DBLK_PARAM_BUF_BLKTYPE_10_FIELD                 ((0x1) << DBLK_PARAM_BUF_BLKTYPE_10_SHIFT)
#define DBLK_PARAM_BUF_BLKTYPE_10_RANGE                 (5):(5)
#define DBLK_PARAM_BUF_BLKTYPE_10_ROW                   0
#define DBLK_PARAM_BUF_BLKTYPE_10_INTER                 (0)
#define DBLK_PARAM_BUF_BLKTYPE_10_INTRA                 (1)

#define DBLK_PARAM_BUF_QPC_SHIFT                        (6)
#define DBLK_PARAM_BUF_QPC_FIELD                        ((0x3f) << DBLK_PARAM_BUF_QPC_SHIFT)
#define DBLK_PARAM_BUF_QPC_RANGE                        (11):(6)
#define DBLK_PARAM_BUF_QPC_ROW                  0

#define DBLK_PARAM_BUF_QPY_SHIFT                        (12)
#define DBLK_PARAM_BUF_QPY_FIELD                        ((0x3f) << DBLK_PARAM_BUF_QPY_SHIFT)
#define DBLK_PARAM_BUF_QPY_RANGE                        (17):(12)
#define DBLK_PARAM_BUF_QPY_ROW                  0

// 1 if non-zero coefficients exist in block 15
#define DBLK_PARAM_BUF_NON_ZERO_COEF_15_SHIFT                   (18)
#define DBLK_PARAM_BUF_NON_ZERO_COEF_15_FIELD                   ((0x1) << DBLK_PARAM_BUF_NON_ZERO_COEF_15_SHIFT)
#define DBLK_PARAM_BUF_NON_ZERO_COEF_15_RANGE                   (18):(18)
#define DBLK_PARAM_BUF_NON_ZERO_COEF_15_ROW                     0

// 1 if non-zero coefficients exist in block 14
#define DBLK_PARAM_BUF_NON_ZERO_COEF_14_SHIFT                   (19)
#define DBLK_PARAM_BUF_NON_ZERO_COEF_14_FIELD                   ((0x1) << DBLK_PARAM_BUF_NON_ZERO_COEF_14_SHIFT)
#define DBLK_PARAM_BUF_NON_ZERO_COEF_14_RANGE                   (19):(19)
#define DBLK_PARAM_BUF_NON_ZERO_COEF_14_ROW                     0

// 1 if non-zero coefficients exist in block 11
#define DBLK_PARAM_BUF_NON_ZERO_COEF_11_SHIFT                   (20)
#define DBLK_PARAM_BUF_NON_ZERO_COEF_11_FIELD                   ((0x1) << DBLK_PARAM_BUF_NON_ZERO_COEF_11_SHIFT)
#define DBLK_PARAM_BUF_NON_ZERO_COEF_11_RANGE                   (20):(20)
#define DBLK_PARAM_BUF_NON_ZERO_COEF_11_ROW                     0

// 1 if non-zero coefficients exist in block 10
#define DBLK_PARAM_BUF_NON_ZERO_COEF_10_SHIFT                   (21)
#define DBLK_PARAM_BUF_NON_ZERO_COEF_10_FIELD                   ((0x1) << DBLK_PARAM_BUF_NON_ZERO_COEF_10_SHIFT)
#define DBLK_PARAM_BUF_NON_ZERO_COEF_10_RANGE                   (21):(21)
#define DBLK_PARAM_BUF_NON_ZERO_COEF_10_ROW                     0

#define DBLK_PARAM_BUF_REFID_Y2_L0_SHIFT                        (22)
#define DBLK_PARAM_BUF_REFID_Y2_L0_FIELD                        ((0x3) << DBLK_PARAM_BUF_REFID_Y2_L0_SHIFT)
#define DBLK_PARAM_BUF_REFID_Y2_L0_RANGE                        (23):(22)
#define DBLK_PARAM_BUF_REFID_Y2_L0_ROW                  0

#define DBLK_PARAM_BUF_REFID_Y3_L0_SHIFT                        (24)
#define DBLK_PARAM_BUF_REFID_Y3_L0_FIELD                        ((0x3) << DBLK_PARAM_BUF_REFID_Y3_L0_SHIFT)
#define DBLK_PARAM_BUF_REFID_Y3_L0_RANGE                        (25):(24)
#define DBLK_PARAM_BUF_REFID_Y3_L0_ROW                  0

#define DBLK_PARAM_BUF_MVX_10_SHIFT                     (26)
#define DBLK_PARAM_BUF_MVX_10_FIELD                     ((0x3ff) << DBLK_PARAM_BUF_MVX_10_SHIFT)
#define DBLK_PARAM_BUF_MVX_10_RANGE                     (35):(26)
#define DBLK_PARAM_BUF_MVX_10_ROW                       0

#define DBLK_PARAM_BUF_MVY_10_SHIFT                     (36)
#define DBLK_PARAM_BUF_MVY_10_FIELD                     ((0x3ff) << DBLK_PARAM_BUF_MVY_10_SHIFT)
#define DBLK_PARAM_BUF_MVY_10_RANGE                     (45):(36)
#define DBLK_PARAM_BUF_MVY_10_ROW                       0

#define DBLK_PARAM_BUF_MVX_11_SHIFT                     (46)
#define DBLK_PARAM_BUF_MVX_11_FIELD                     ((0x3ff) << DBLK_PARAM_BUF_MVX_11_SHIFT)
#define DBLK_PARAM_BUF_MVX_11_RANGE                     (55):(46)
#define DBLK_PARAM_BUF_MVX_11_ROW                       0

#define DBLK_PARAM_BUF_MVY_11_SHIFT                     (56)
#define DBLK_PARAM_BUF_MVY_11_FIELD                     ((0x3ff) << DBLK_PARAM_BUF_MVY_11_SHIFT)
#define DBLK_PARAM_BUF_MVY_11_RANGE                     (65):(56)
#define DBLK_PARAM_BUF_MVY_11_ROW                       0

#define DBLK_PARAM_BUF_MVX_14_SHIFT                     (66)
#define DBLK_PARAM_BUF_MVX_14_FIELD                     ((0x3ff) << DBLK_PARAM_BUF_MVX_14_SHIFT)
#define DBLK_PARAM_BUF_MVX_14_RANGE                     (75):(66)
#define DBLK_PARAM_BUF_MVX_14_ROW                       0

#define DBLK_PARAM_BUF_MVY_14_SHIFT                     (76)
#define DBLK_PARAM_BUF_MVY_14_FIELD                     ((0x3ff) << DBLK_PARAM_BUF_MVY_14_SHIFT)
#define DBLK_PARAM_BUF_MVY_14_RANGE                     (85):(76)
#define DBLK_PARAM_BUF_MVY_14_ROW                       0

#define DBLK_PARAM_BUF_MVX_15_SHIFT                     (86)
#define DBLK_PARAM_BUF_MVX_15_FIELD                     ((0x3ff) << DBLK_PARAM_BUF_MVX_15_SHIFT)
#define DBLK_PARAM_BUF_MVX_15_RANGE                     (95):(86)
#define DBLK_PARAM_BUF_MVX_15_ROW                       0

#define DBLK_PARAM_BUF_MVY_15_SHIFT                     (96)
#define DBLK_PARAM_BUF_MVY_15_FIELD                     ((0x3ff) << DBLK_PARAM_BUF_MVY_15_SHIFT)
#define DBLK_PARAM_BUF_MVY_15_RANGE                     (105):(96)
#define DBLK_PARAM_BUF_MVY_15_ROW                       0

#define DBLK_PARAM_BUF_RESERVED_SHIFT                   (106)
#define DBLK_PARAM_BUF_RESERVED_FIELD                   ((0x3fffff) << DBLK_PARAM_BUF_RESERVED_SHIFT)
#define DBLK_PARAM_BUF_RESERVED_RANGE                   (127):(106)
#define DBLK_PARAM_BUF_RESERVED_ROW                     0

// --------------------------------------------------------------------------
// 
// Copyright (c) 2004, LWPU Corp.
// All Rights Reserved.
// 
// This is UNPUBLISHED PROPRIETARY SOURCE CODE of LWPU Corp.;
// the contents of this file may not be disclosed to third parties, copied or
// duplicated in any form, in whole or in part, without the prior written
// permission of LWPU Corp.
// 
// RESTRICTED RIGHTS LEGEND:
// Use, duplication or disclosure by the Government is subject to restrictions
// as set forth in subdivision (c)(1)(ii) of the Rights in Technical Data
// and Computer Software clause at DFARS 252.227-7013, and/or in similar or
// successor clauses in the FAR, DOD or NASA FAR Supplement. Unpublished -
// rights reserved under the Copyright Laws of the United States.
// 
// --------------------------------------------------------------------------
// 
// -------------------------------------------------------------------
// FIFO parameters (ME legacy)
// -------------------------------------------------------------------
// increased to hold 1MB of data (1 chunk + 1 frame + 3 MB) header +   
// (24 lwrr + 24 pred) data = 53
#define LW_MPEC_DMA2EBM_FIFO_DEPTH      8
#define LW_MPEC_DMA2EBM_FIFO_WIDTH      128

// Packet LL_MPEG4_VLC
#define LL_MPEG4_VLC_SIZE 128

#define LL_MPEG4_VLC_HDR1PKTBITCNT_SHIFT                        (0)
#define LL_MPEG4_VLC_HDR1PKTBITCNT_FIELD                        ((0x7fffff) << LL_MPEG4_VLC_HDR1PKTBITCNT_SHIFT)
#define LL_MPEG4_VLC_HDR1PKTBITCNT_RANGE                        (22):(0)
#define LL_MPEG4_VLC_HDR1PKTBITCNT_ROW                  0

#define LL_MPEG4_VLC_FRAME_NUM_SHIFT                    (23)
#define LL_MPEG4_VLC_FRAME_NUM_FIELD                    ((0xffff) << LL_MPEG4_VLC_FRAME_NUM_SHIFT)
#define LL_MPEG4_VLC_FRAME_NUM_RANGE                    (38):(23)
#define LL_MPEG4_VLC_FRAME_NUM_ROW                      0

#define LL_MPEG4_VLC_SAVEDTSTAMP_SHIFT                  (64)
#define LL_MPEG4_VLC_SAVEDTSTAMP_FIELD                  ((0xffffffff) << LL_MPEG4_VLC_SAVEDTSTAMP_SHIFT)
#define LL_MPEG4_VLC_SAVEDTSTAMP_RANGE                  (95):(64)
#define LL_MPEG4_VLC_SAVEDTSTAMP_ROW                    0

#define LL_MPEG4_VLC_STREAM_TYPE_SHIFT                  (111)
#define LL_MPEG4_VLC_STREAM_TYPE_FIELD                  ((0x1) << LL_MPEG4_VLC_STREAM_TYPE_SHIFT)
#define LL_MPEG4_VLC_STREAM_TYPE_RANGE                  (111):(111)
#define LL_MPEG4_VLC_STREAM_TYPE_ROW                    0
#define LL_MPEG4_VLC_STREAM_TYPE_NAL                    (0)
#define LL_MPEG4_VLC_STREAM_TYPE_BYTE                   (1)

#define LL_MPEG4_VLC_STREAM_ID_SHIFT                    (112)
#define LL_MPEG4_VLC_STREAM_ID_FIELD                    ((0xff) << LL_MPEG4_VLC_STREAM_ID_SHIFT)
#define LL_MPEG4_VLC_STREAM_ID_RANGE                    (119):(112)
#define LL_MPEG4_VLC_STREAM_ID_ROW                      0

#define LL_MPEG4_VLC_VIDEO_MODE_SHIFT                   (120)
#define LL_MPEG4_VLC_VIDEO_MODE_FIELD                   ((0xf) << LL_MPEG4_VLC_VIDEO_MODE_SHIFT)
#define LL_MPEG4_VLC_VIDEO_MODE_RANGE                   (123):(120)
#define LL_MPEG4_VLC_VIDEO_MODE_ROW                     0
#define LL_MPEG4_VLC_VIDEO_MODE_MPEG4_VLC                       (0)
#define LL_MPEG4_VLC_VIDEO_MODE_MPEG4_DP                        (1)
#define LL_MPEG4_VLC_VIDEO_MODE_MPEG4_VLC_BYPASS                        (2)
#define LL_MPEG4_VLC_VIDEO_MODE_MPEG4_ME_BYPASS                 (3)
#define LL_MPEG4_VLC_VIDEO_MODE_H264                    (4)
#define LL_MPEG4_VLC_VIDEO_MODE_H264_CAVLC_BYPASS                       (5)
#define LL_MPEG4_VLC_VIDEO_MODE_H264_ME_BYPASS                  (6)

#define LL_MPEG4_VLC_NEW_FRAME_SHIFT                    (124)
#define LL_MPEG4_VLC_NEW_FRAME_FIELD                    ((0x1) << LL_MPEG4_VLC_NEW_FRAME_SHIFT)
#define LL_MPEG4_VLC_NEW_FRAME_RANGE                    (124):(124)
#define LL_MPEG4_VLC_NEW_FRAME_ROW                      0

#define LL_MPEG4_VLC_FRAME_TYPE_SHIFT                   (125)
#define LL_MPEG4_VLC_FRAME_TYPE_FIELD                   ((0x3) << LL_MPEG4_VLC_FRAME_TYPE_SHIFT)
#define LL_MPEG4_VLC_FRAME_TYPE_RANGE                   (126):(125)
#define LL_MPEG4_VLC_FRAME_TYPE_ROW                     0
#define LL_MPEG4_VLC_FRAME_TYPE_I                       (0)
#define LL_MPEG4_VLC_FRAME_TYPE_P                       (1)

#define LL_MPEG4_VLC_END_FRAME_SHIFT                    (127)
#define LL_MPEG4_VLC_END_FRAME_FIELD                    ((0x1) << LL_MPEG4_VLC_END_FRAME_SHIFT)
#define LL_MPEG4_VLC_END_FRAME_RANGE                    (127):(127)
#define LL_MPEG4_VLC_END_FRAME_ROW                      0

#define LL_MPEG4_VLC_NEW_PACKET_SHIFT                   (125)
#define LL_MPEG4_VLC_NEW_PACKET_FIELD                   ((0x1) << LL_MPEG4_VLC_NEW_PACKET_SHIFT)
#define LL_MPEG4_VLC_NEW_PACKET_RANGE                   (125):(125)
#define LL_MPEG4_VLC_NEW_PACKET_ROW                     0

#define LL_MPEG4_VLC_FRAME_BIT_LENGTH_SHIFT                     (64)
#define LL_MPEG4_VLC_FRAME_BIT_LENGTH_FIELD                     ((0xffffffff) << LL_MPEG4_VLC_FRAME_BIT_LENGTH_SHIFT)
#define LL_MPEG4_VLC_FRAME_BIT_LENGTH_RANGE                     (95):(64)
#define LL_MPEG4_VLC_FRAME_BIT_LENGTH_ROW                       0

#define LL_MPEG4_VLC_BUFFER_OVERFLOW_SHIFT                      (126)
#define LL_MPEG4_VLC_BUFFER_OVERFLOW_FIELD                      ((0x1) << LL_MPEG4_VLC_BUFFER_OVERFLOW_SHIFT)
#define LL_MPEG4_VLC_BUFFER_OVERFLOW_RANGE                      (126):(126)
#define LL_MPEG4_VLC_BUFFER_OVERFLOW_ROW                        0


// Packet LL_MPEG4_DP
#define LL_MPEG4_DP_SIZE 128

#define LL_MPEG4_DP_HDR1PKTBITCNT_SHIFT                 (0)
#define LL_MPEG4_DP_HDR1PKTBITCNT_FIELD                 ((0x7fff) << LL_MPEG4_DP_HDR1PKTBITCNT_SHIFT)
#define LL_MPEG4_DP_HDR1PKTBITCNT_RANGE                 (14):(0)
#define LL_MPEG4_DP_HDR1PKTBITCNT_ROW                   0

#define LL_MPEG4_DP_HDR2PKTBITCNT_SHIFT                 (15)
#define LL_MPEG4_DP_HDR2PKTBITCNT_FIELD                 ((0x7fff) << LL_MPEG4_DP_HDR2PKTBITCNT_SHIFT)
#define LL_MPEG4_DP_HDR2PKTBITCNT_RANGE                 (29):(15)
#define LL_MPEG4_DP_HDR2PKTBITCNT_ROW                   0

#define LL_MPEG4_DP_TEXTPKTBITCNT_SHIFT                 (30)
#define LL_MPEG4_DP_TEXTPKTBITCNT_FIELD                 ((0x7fff) << LL_MPEG4_DP_TEXTPKTBITCNT_SHIFT)
#define LL_MPEG4_DP_TEXTPKTBITCNT_RANGE                 (44):(30)
#define LL_MPEG4_DP_TEXTPKTBITCNT_ROW                   0

#define LL_MPEG4_DP_FRAME_NUM_SHIFT                     (45)
#define LL_MPEG4_DP_FRAME_NUM_FIELD                     ((0xffff) << LL_MPEG4_DP_FRAME_NUM_SHIFT)
#define LL_MPEG4_DP_FRAME_NUM_RANGE                     (60):(45)
#define LL_MPEG4_DP_FRAME_NUM_ROW                       0

#define LL_MPEG4_DP_SAVEDTSTAMP_SHIFT                   (64)
#define LL_MPEG4_DP_SAVEDTSTAMP_FIELD                   ((0xffffffff) << LL_MPEG4_DP_SAVEDTSTAMP_SHIFT)
#define LL_MPEG4_DP_SAVEDTSTAMP_RANGE                   (95):(64)
#define LL_MPEG4_DP_SAVEDTSTAMP_ROW                     0

#define LL_MPEG4_DP_STREAM_TYPE_SHIFT                   (111)
#define LL_MPEG4_DP_STREAM_TYPE_FIELD                   ((0x1) << LL_MPEG4_DP_STREAM_TYPE_SHIFT)
#define LL_MPEG4_DP_STREAM_TYPE_RANGE                   (111):(111)
#define LL_MPEG4_DP_STREAM_TYPE_ROW                     0
#define LL_MPEG4_DP_STREAM_TYPE_NAL                     (0)
#define LL_MPEG4_DP_STREAM_TYPE_BYTE                    (1)

#define LL_MPEG4_DP_STREAM_ID_SHIFT                     (112)
#define LL_MPEG4_DP_STREAM_ID_FIELD                     ((0xff) << LL_MPEG4_DP_STREAM_ID_SHIFT)
#define LL_MPEG4_DP_STREAM_ID_RANGE                     (119):(112)
#define LL_MPEG4_DP_STREAM_ID_ROW                       0

#define LL_MPEG4_DP_VIDEO_MODE_SHIFT                    (120)
#define LL_MPEG4_DP_VIDEO_MODE_FIELD                    ((0xf) << LL_MPEG4_DP_VIDEO_MODE_SHIFT)
#define LL_MPEG4_DP_VIDEO_MODE_RANGE                    (123):(120)
#define LL_MPEG4_DP_VIDEO_MODE_ROW                      0
#define LL_MPEG4_DP_VIDEO_MODE_MPEG4_VLC                        (0)
#define LL_MPEG4_DP_VIDEO_MODE_MPEG4_DP                 (1)
#define LL_MPEG4_DP_VIDEO_MODE_MPEG4_VLC_BYPASS                 (2)
#define LL_MPEG4_DP_VIDEO_MODE_MPEG4_ME_BYPASS                  (3)
#define LL_MPEG4_DP_VIDEO_MODE_H264                     (4)
#define LL_MPEG4_DP_VIDEO_MODE_H264_CAVLC_BYPASS                        (5)
#define LL_MPEG4_DP_VIDEO_MODE_H264_ME_BYPASS                   (6)

#define LL_MPEG4_DP_NEW_FRAME_SHIFT                     (124)
#define LL_MPEG4_DP_NEW_FRAME_FIELD                     ((0x1) << LL_MPEG4_DP_NEW_FRAME_SHIFT)
#define LL_MPEG4_DP_NEW_FRAME_RANGE                     (124):(124)
#define LL_MPEG4_DP_NEW_FRAME_ROW                       0

#define LL_MPEG4_DP_FRAME_TYPE_SHIFT                    (125)
#define LL_MPEG4_DP_FRAME_TYPE_FIELD                    ((0x3) << LL_MPEG4_DP_FRAME_TYPE_SHIFT)
#define LL_MPEG4_DP_FRAME_TYPE_RANGE                    (126):(125)
#define LL_MPEG4_DP_FRAME_TYPE_ROW                      0
#define LL_MPEG4_DP_FRAME_TYPE_I                        (0)
#define LL_MPEG4_DP_FRAME_TYPE_P                        (1)

#define LL_MPEG4_DP_END_FRAME_SHIFT                     (127)
#define LL_MPEG4_DP_END_FRAME_FIELD                     ((0x1) << LL_MPEG4_DP_END_FRAME_SHIFT)
#define LL_MPEG4_DP_END_FRAME_RANGE                     (127):(127)
#define LL_MPEG4_DP_END_FRAME_ROW                       0

#define LL_MPEG4_DP_NEW_PACKET_SHIFT                    (125)
#define LL_MPEG4_DP_NEW_PACKET_FIELD                    ((0x1) << LL_MPEG4_DP_NEW_PACKET_SHIFT)
#define LL_MPEG4_DP_NEW_PACKET_RANGE                    (125):(125)
#define LL_MPEG4_DP_NEW_PACKET_ROW                      0

#define LL_MPEG4_DP_BUFFER_OVERFLOW_SHIFT                       (126)
#define LL_MPEG4_DP_BUFFER_OVERFLOW_FIELD                       ((0x1) << LL_MPEG4_DP_BUFFER_OVERFLOW_SHIFT)
#define LL_MPEG4_DP_BUFFER_OVERFLOW_RANGE                       (126):(126)
#define LL_MPEG4_DP_BUFFER_OVERFLOW_ROW                 0

#define LL_MPEG4_DP_FRAME_BIT_LENGTH_SHIFT                      (64)
#define LL_MPEG4_DP_FRAME_BIT_LENGTH_FIELD                      ((0xffffffff) << LL_MPEG4_DP_FRAME_BIT_LENGTH_SHIFT)
#define LL_MPEG4_DP_FRAME_BIT_LENGTH_RANGE                      (95):(64)
#define LL_MPEG4_DP_FRAME_BIT_LENGTH_ROW                        0


// Packet LL_H264
#define LL_H264_SIZE 128

#define LL_H264_HDR1PKTBITCNT_SHIFT                     (0)
#define LL_H264_HDR1PKTBITCNT_FIELD                     ((0x7fffff) << LL_H264_HDR1PKTBITCNT_SHIFT)
#define LL_H264_HDR1PKTBITCNT_RANGE                     (22):(0)
#define LL_H264_HDR1PKTBITCNT_ROW                       0

#define LL_H264_FRAME_NUM_SHIFT                 (23)
#define LL_H264_FRAME_NUM_FIELD                 ((0xffff) << LL_H264_FRAME_NUM_SHIFT)
#define LL_H264_FRAME_NUM_RANGE                 (38):(23)
#define LL_H264_FRAME_NUM_ROW                   0

#define LL_H264_LAST_SLICE_IN_SG_SHIFT                  (39)
#define LL_H264_LAST_SLICE_IN_SG_FIELD                  ((0x1) << LL_H264_LAST_SLICE_IN_SG_SHIFT)
#define LL_H264_LAST_SLICE_IN_SG_RANGE                  (39):(39)
#define LL_H264_LAST_SLICE_IN_SG_ROW                    0

#define LL_H264_FIRST_MBNUM_SHIFT                       (40)
#define LL_H264_FIRST_MBNUM_FIELD                       ((0x1fff) << LL_H264_FIRST_MBNUM_SHIFT)
#define LL_H264_FIRST_MBNUM_RANGE                       (52):(40)
#define LL_H264_FIRST_MBNUM_ROW                 0

#define LL_H264_SLICE_GRP_ACT_SHIFT                     (57)
#define LL_H264_SLICE_GRP_ACT_FIELD                     ((0x7) << LL_H264_SLICE_GRP_ACT_SHIFT)
#define LL_H264_SLICE_GRP_ACT_RANGE                     (59):(57)
#define LL_H264_SLICE_GRP_ACT_ROW                       0

#define LL_H264_NAL_TYPE_SHIFT                  (60)
#define LL_H264_NAL_TYPE_FIELD                  ((0xf) << LL_H264_NAL_TYPE_SHIFT)
#define LL_H264_NAL_TYPE_RANGE                  (63):(60)
#define LL_H264_NAL_TYPE_ROW                    0
#define LL_H264_NAL_TYPE_SEQ                    (7)
#define LL_H264_NAL_TYPE_PIC                    (8)
#define LL_H264_NAL_TYPE_NON_IDR                        (1)
#define LL_H264_NAL_TYPE_IDR                    (5)

#define LL_H264_SAVEDTSTAMP_SHIFT                       (64)
#define LL_H264_SAVEDTSTAMP_FIELD                       ((0xffffffff) << LL_H264_SAVEDTSTAMP_SHIFT)
#define LL_H264_SAVEDTSTAMP_RANGE                       (95):(64)
#define LL_H264_SAVEDTSTAMP_ROW                 0

#define LL_H264_STREAM_TYPE_SHIFT                       (111)
#define LL_H264_STREAM_TYPE_FIELD                       ((0x1) << LL_H264_STREAM_TYPE_SHIFT)
#define LL_H264_STREAM_TYPE_RANGE                       (111):(111)
#define LL_H264_STREAM_TYPE_ROW                 0
#define LL_H264_STREAM_TYPE_NAL                 (0)
#define LL_H264_STREAM_TYPE_BYTE                        (1)

#define LL_H264_STREAM_ID_SHIFT                 (112)
#define LL_H264_STREAM_ID_FIELD                 ((0xff) << LL_H264_STREAM_ID_SHIFT)
#define LL_H264_STREAM_ID_RANGE                 (119):(112)
#define LL_H264_STREAM_ID_ROW                   0

#define LL_H264_VIDEO_MODE_SHIFT                        (120)
#define LL_H264_VIDEO_MODE_FIELD                        ((0xf) << LL_H264_VIDEO_MODE_SHIFT)
#define LL_H264_VIDEO_MODE_RANGE                        (123):(120)
#define LL_H264_VIDEO_MODE_ROW                  0
#define LL_H264_VIDEO_MODE_MPEG4_VLC                    (0)
#define LL_H264_VIDEO_MODE_MPEG4_DP                     (1)
#define LL_H264_VIDEO_MODE_MPEG4_VLC_BYPASS                     (2)
#define LL_H264_VIDEO_MODE_MPEG4_ME_BYPASS                      (3)
#define LL_H264_VIDEO_MODE_H264                 (4)
#define LL_H264_VIDEO_MODE_H264_CAVLC_BYPASS                    (5)
#define LL_H264_VIDEO_MODE_H264_ME_BYPASS                       (6)

#define LL_H264_NEW_FRAME_SHIFT                 (124)
#define LL_H264_NEW_FRAME_FIELD                 ((0x1) << LL_H264_NEW_FRAME_SHIFT)
#define LL_H264_NEW_FRAME_RANGE                 (124):(124)
#define LL_H264_NEW_FRAME_ROW                   0

#define LL_H264_FRAME_TYPE_SHIFT                        (125)
#define LL_H264_FRAME_TYPE_FIELD                        ((0x3) << LL_H264_FRAME_TYPE_SHIFT)
#define LL_H264_FRAME_TYPE_RANGE                        (126):(125)
#define LL_H264_FRAME_TYPE_ROW                  0
#define LL_H264_FRAME_TYPE_I                    (0)
#define LL_H264_FRAME_TYPE_P                    (1)

#define LL_H264_END_FRAME_SHIFT                 (127)
#define LL_H264_END_FRAME_FIELD                 ((0x1) << LL_H264_END_FRAME_SHIFT)
#define LL_H264_END_FRAME_RANGE                 (127):(127)
#define LL_H264_END_FRAME_ROW                   0

#define LL_H264_NEW_PACKET_SHIFT                        (125)
#define LL_H264_NEW_PACKET_FIELD                        ((0x1) << LL_H264_NEW_PACKET_SHIFT)
#define LL_H264_NEW_PACKET_RANGE                        (125):(125)
#define LL_H264_NEW_PACKET_ROW                  0

#define LL_H264_BUFFER_OVERFLOW_SHIFT                   (126)
#define LL_H264_BUFFER_OVERFLOW_FIELD                   ((0x1) << LL_H264_BUFFER_OVERFLOW_SHIFT)
#define LL_H264_BUFFER_OVERFLOW_RANGE                   (126):(126)
#define LL_H264_BUFFER_OVERFLOW_ROW                     0

#define LL_H264_FRAME_BIT_LENGTH_SHIFT                  (64)
#define LL_H264_FRAME_BIT_LENGTH_FIELD                  ((0xffffffff) << LL_H264_FRAME_BIT_LENGTH_SHIFT)
#define LL_H264_FRAME_BIT_LENGTH_RANGE                  (95):(64)
#define LL_H264_FRAME_BIT_LENGTH_ROW                    0


// Packet EBM_MPEG4_FIRST_0
#define EBM_MPEG4_FIRST_0_SIZE 96

#define EBM_MPEG4_FIRST_0_CHUNK_LENGTH_SHIFT                    (0)
#define EBM_MPEG4_FIRST_0_CHUNK_LENGTH_FIELD                    ((0xfffff) << EBM_MPEG4_FIRST_0_CHUNK_LENGTH_SHIFT)
#define EBM_MPEG4_FIRST_0_CHUNK_LENGTH_RANGE                    (19):(0)
#define EBM_MPEG4_FIRST_0_CHUNK_LENGTH_ROW                      0

#define EBM_MPEG4_FIRST_0_FRAME_TYPE_SHIFT                      (29)
#define EBM_MPEG4_FIRST_0_FRAME_TYPE_FIELD                      ((0x3) << EBM_MPEG4_FIRST_0_FRAME_TYPE_SHIFT)
#define EBM_MPEG4_FIRST_0_FRAME_TYPE_RANGE                      (30):(29)
#define EBM_MPEG4_FIRST_0_FRAME_TYPE_ROW                        0
#define EBM_MPEG4_FIRST_0_FRAME_TYPE_I                  (0)
#define EBM_MPEG4_FIRST_0_FRAME_TYPE_P                  (1)

#define EBM_MPEG4_FIRST_0_END_FRAME_SHIFT                       (31)
#define EBM_MPEG4_FIRST_0_END_FRAME_FIELD                       ((0x1) << EBM_MPEG4_FIRST_0_END_FRAME_SHIFT)
#define EBM_MPEG4_FIRST_0_END_FRAME_RANGE                       (31):(31)
#define EBM_MPEG4_FIRST_0_END_FRAME_ROW                 0

#define EBM_MPEG4_FIRST_0_FRAME_NUM_SHIFT                       (32)
#define EBM_MPEG4_FIRST_0_FRAME_NUM_FIELD                       ((0xffff) << EBM_MPEG4_FIRST_0_FRAME_NUM_SHIFT)
#define EBM_MPEG4_FIRST_0_FRAME_NUM_RANGE                       (47):(32)
#define EBM_MPEG4_FIRST_0_FRAME_NUM_ROW                 0

#define EBM_MPEG4_FIRST_0_STREAM_ID_SHIFT                       (48)
#define EBM_MPEG4_FIRST_0_STREAM_ID_FIELD                       ((0xff) << EBM_MPEG4_FIRST_0_STREAM_ID_SHIFT)
#define EBM_MPEG4_FIRST_0_STREAM_ID_RANGE                       (55):(48)
#define EBM_MPEG4_FIRST_0_STREAM_ID_ROW                 0

#define EBM_MPEG4_FIRST_0_TIMESTAMP_SHIFT                       (64)
#define EBM_MPEG4_FIRST_0_TIMESTAMP_FIELD                       ((0xffffffff) << EBM_MPEG4_FIRST_0_TIMESTAMP_SHIFT)
#define EBM_MPEG4_FIRST_0_TIMESTAMP_RANGE                       (95):(64)
#define EBM_MPEG4_FIRST_0_TIMESTAMP_ROW                 0

// packet OLD_EBM_MPEG4_FIRST
//     10:0        0       CHUNK_LENGTH
//     26:11       0       FRAME_NUM
//     30:30       0       FRAME_TYPE
//     31:31       0       END_FRAME
//     63:32       0       TIMESTAMP
// ;

// Packet EBM_MPEG4_MIDDLE_0
#define EBM_MPEG4_MIDDLE_0_SIZE 32

#define EBM_MPEG4_MIDDLE_0_CHUNK_LENGTH_SHIFT                   (0)
#define EBM_MPEG4_MIDDLE_0_CHUNK_LENGTH_FIELD                   ((0xfffff) << EBM_MPEG4_MIDDLE_0_CHUNK_LENGTH_SHIFT)
#define EBM_MPEG4_MIDDLE_0_CHUNK_LENGTH_RANGE                   (19):(0)
#define EBM_MPEG4_MIDDLE_0_CHUNK_LENGTH_ROW                     0

#define EBM_MPEG4_MIDDLE_0_NEW_PACKET_SHIFT                     (29)
#define EBM_MPEG4_MIDDLE_0_NEW_PACKET_FIELD                     ((0x1) << EBM_MPEG4_MIDDLE_0_NEW_PACKET_SHIFT)
#define EBM_MPEG4_MIDDLE_0_NEW_PACKET_RANGE                     (29):(29)
#define EBM_MPEG4_MIDDLE_0_NEW_PACKET_ROW                       0

#define EBM_MPEG4_MIDDLE_0_END_FRAME_SHIFT                      (31)
#define EBM_MPEG4_MIDDLE_0_END_FRAME_FIELD                      ((0x1) << EBM_MPEG4_MIDDLE_0_END_FRAME_SHIFT)
#define EBM_MPEG4_MIDDLE_0_END_FRAME_RANGE                      (31):(31)
#define EBM_MPEG4_MIDDLE_0_END_FRAME_ROW                        0


// Packet EBM_MPEG4_LAST_0
#define EBM_MPEG4_LAST_0_SIZE 64

#define EBM_MPEG4_LAST_0_CHUNK_LENGTH_SHIFT                     (0)
#define EBM_MPEG4_LAST_0_CHUNK_LENGTH_FIELD                     ((0xfffff) << EBM_MPEG4_LAST_0_CHUNK_LENGTH_SHIFT)
#define EBM_MPEG4_LAST_0_CHUNK_LENGTH_RANGE                     (19):(0)
#define EBM_MPEG4_LAST_0_CHUNK_LENGTH_ROW                       0

#define EBM_MPEG4_LAST_0_NEW_PACKET_SHIFT                       (29)
#define EBM_MPEG4_LAST_0_NEW_PACKET_FIELD                       ((0x1) << EBM_MPEG4_LAST_0_NEW_PACKET_SHIFT)
#define EBM_MPEG4_LAST_0_NEW_PACKET_RANGE                       (29):(29)
#define EBM_MPEG4_LAST_0_NEW_PACKET_ROW                 0

#define EBM_MPEG4_LAST_0_OVER_FLOW_SHIFT                        (30)
#define EBM_MPEG4_LAST_0_OVER_FLOW_FIELD                        ((0x1) << EBM_MPEG4_LAST_0_OVER_FLOW_SHIFT)
#define EBM_MPEG4_LAST_0_OVER_FLOW_RANGE                        (30):(30)
#define EBM_MPEG4_LAST_0_OVER_FLOW_ROW                  0

#define EBM_MPEG4_LAST_0_END_FRAME_SHIFT                        (31)
#define EBM_MPEG4_LAST_0_END_FRAME_FIELD                        ((0x1) << EBM_MPEG4_LAST_0_END_FRAME_SHIFT)
#define EBM_MPEG4_LAST_0_END_FRAME_RANGE                        (31):(31)
#define EBM_MPEG4_LAST_0_END_FRAME_ROW                  0

#define EBM_MPEG4_LAST_0_FRAME_LENGTH_SHIFT                     (32)
#define EBM_MPEG4_LAST_0_FRAME_LENGTH_FIELD                     ((0xffffffff) << EBM_MPEG4_LAST_0_FRAME_LENGTH_SHIFT)
#define EBM_MPEG4_LAST_0_FRAME_LENGTH_RANGE                     (63):(32)
#define EBM_MPEG4_LAST_0_FRAME_LENGTH_ROW                       0


// Packet EBM_H264_NAL_FIRST_0
#define EBM_H264_NAL_FIRST_0_SIZE 96

#define EBM_H264_NAL_FIRST_0_CHUNK_LENGTH_SHIFT                 (0)
#define EBM_H264_NAL_FIRST_0_CHUNK_LENGTH_FIELD                 ((0xfffff) << EBM_H264_NAL_FIRST_0_CHUNK_LENGTH_SHIFT)
#define EBM_H264_NAL_FIRST_0_CHUNK_LENGTH_RANGE                 (19):(0)
#define EBM_H264_NAL_FIRST_0_CHUNK_LENGTH_ROW                   0

#define EBM_H264_NAL_FIRST_0_FRAME_TYPE_SHIFT                   (29)
#define EBM_H264_NAL_FIRST_0_FRAME_TYPE_FIELD                   ((0x3) << EBM_H264_NAL_FIRST_0_FRAME_TYPE_SHIFT)
#define EBM_H264_NAL_FIRST_0_FRAME_TYPE_RANGE                   (30):(29)
#define EBM_H264_NAL_FIRST_0_FRAME_TYPE_ROW                     0
#define EBM_H264_NAL_FIRST_0_FRAME_TYPE_I                       (0)
#define EBM_H264_NAL_FIRST_0_FRAME_TYPE_P                       (1)

#define EBM_H264_NAL_FIRST_0_END_FRAME_SHIFT                    (31)
#define EBM_H264_NAL_FIRST_0_END_FRAME_FIELD                    ((0x1) << EBM_H264_NAL_FIRST_0_END_FRAME_SHIFT)
#define EBM_H264_NAL_FIRST_0_END_FRAME_RANGE                    (31):(31)
#define EBM_H264_NAL_FIRST_0_END_FRAME_ROW                      0

#define EBM_H264_NAL_FIRST_0_FRAME_NUM_SHIFT                    (32)
#define EBM_H264_NAL_FIRST_0_FRAME_NUM_FIELD                    ((0xffff) << EBM_H264_NAL_FIRST_0_FRAME_NUM_SHIFT)
#define EBM_H264_NAL_FIRST_0_FRAME_NUM_RANGE                    (47):(32)
#define EBM_H264_NAL_FIRST_0_FRAME_NUM_ROW                      0

#define EBM_H264_NAL_FIRST_0_STREAM_ID_SHIFT                    (48)
#define EBM_H264_NAL_FIRST_0_STREAM_ID_FIELD                    ((0xff) << EBM_H264_NAL_FIRST_0_STREAM_ID_SHIFT)
#define EBM_H264_NAL_FIRST_0_STREAM_ID_RANGE                    (55):(48)
#define EBM_H264_NAL_FIRST_0_STREAM_ID_ROW                      0

#define EBM_H264_NAL_FIRST_0_NAL_TYPE_SHIFT                     (60)
#define EBM_H264_NAL_FIRST_0_NAL_TYPE_FIELD                     ((0xf) << EBM_H264_NAL_FIRST_0_NAL_TYPE_SHIFT)
#define EBM_H264_NAL_FIRST_0_NAL_TYPE_RANGE                     (63):(60)
#define EBM_H264_NAL_FIRST_0_NAL_TYPE_ROW                       0
#define EBM_H264_NAL_FIRST_0_NAL_TYPE_SEQ                       (7)
#define EBM_H264_NAL_FIRST_0_NAL_TYPE_PIC                       (8)
#define EBM_H264_NAL_FIRST_0_NAL_TYPE_NON_IDR                   (1)
#define EBM_H264_NAL_FIRST_0_NAL_TYPE_IDR                       (5)

#define EBM_H264_NAL_FIRST_0_TIMESTAMP_SHIFT                    (64)
#define EBM_H264_NAL_FIRST_0_TIMESTAMP_FIELD                    ((0xffffffff) << EBM_H264_NAL_FIRST_0_TIMESTAMP_SHIFT)
#define EBM_H264_NAL_FIRST_0_TIMESTAMP_RANGE                    (95):(64)
#define EBM_H264_NAL_FIRST_0_TIMESTAMP_ROW                      0


// Packet EBM_H264_NAL_MIDDLE_0
#define EBM_H264_NAL_MIDDLE_0_SIZE 32

#define EBM_H264_NAL_MIDDLE_0_CHUNK_LENGTH_SHIFT                        (0)
#define EBM_H264_NAL_MIDDLE_0_CHUNK_LENGTH_FIELD                        ((0xfffff) << EBM_H264_NAL_MIDDLE_0_CHUNK_LENGTH_SHIFT)
#define EBM_H264_NAL_MIDDLE_0_CHUNK_LENGTH_RANGE                        (19):(0)
#define EBM_H264_NAL_MIDDLE_0_CHUNK_LENGTH_ROW                  0

#define EBM_H264_NAL_MIDDLE_0_NAL_TYPE_SHIFT                    (20)
#define EBM_H264_NAL_MIDDLE_0_NAL_TYPE_FIELD                    ((0xf) << EBM_H264_NAL_MIDDLE_0_NAL_TYPE_SHIFT)
#define EBM_H264_NAL_MIDDLE_0_NAL_TYPE_RANGE                    (23):(20)
#define EBM_H264_NAL_MIDDLE_0_NAL_TYPE_ROW                      0
#define EBM_H264_NAL_MIDDLE_0_NAL_TYPE_SEQ                      (7)
#define EBM_H264_NAL_MIDDLE_0_NAL_TYPE_PIC                      (8)
#define EBM_H264_NAL_MIDDLE_0_NAL_TYPE_NON_IDR                  (1)
#define EBM_H264_NAL_MIDDLE_0_NAL_TYPE_IDR                      (5)

#define EBM_H264_NAL_MIDDLE_0_NEW_PACKET_SHIFT                  (29)
#define EBM_H264_NAL_MIDDLE_0_NEW_PACKET_FIELD                  ((0x1) << EBM_H264_NAL_MIDDLE_0_NEW_PACKET_SHIFT)
#define EBM_H264_NAL_MIDDLE_0_NEW_PACKET_RANGE                  (29):(29)
#define EBM_H264_NAL_MIDDLE_0_NEW_PACKET_ROW                    0

#define EBM_H264_NAL_MIDDLE_0_END_FRAME_SHIFT                   (31)
#define EBM_H264_NAL_MIDDLE_0_END_FRAME_FIELD                   ((0x1) << EBM_H264_NAL_MIDDLE_0_END_FRAME_SHIFT)
#define EBM_H264_NAL_MIDDLE_0_END_FRAME_RANGE                   (31):(31)
#define EBM_H264_NAL_MIDDLE_0_END_FRAME_ROW                     0


// Packet EBM_H264_NAL_LAST_0
#define EBM_H264_NAL_LAST_0_SIZE 64

#define EBM_H264_NAL_LAST_0_CHUNK_LENGTH_SHIFT                  (0)
#define EBM_H264_NAL_LAST_0_CHUNK_LENGTH_FIELD                  ((0xfffff) << EBM_H264_NAL_LAST_0_CHUNK_LENGTH_SHIFT)
#define EBM_H264_NAL_LAST_0_CHUNK_LENGTH_RANGE                  (19):(0)
#define EBM_H264_NAL_LAST_0_CHUNK_LENGTH_ROW                    0

#define EBM_H264_NAL_LAST_0_NAL_TYPE_SHIFT                      (20)
#define EBM_H264_NAL_LAST_0_NAL_TYPE_FIELD                      ((0xf) << EBM_H264_NAL_LAST_0_NAL_TYPE_SHIFT)
#define EBM_H264_NAL_LAST_0_NAL_TYPE_RANGE                      (23):(20)
#define EBM_H264_NAL_LAST_0_NAL_TYPE_ROW                        0
#define EBM_H264_NAL_LAST_0_NAL_TYPE_SEQ                        (7)
#define EBM_H264_NAL_LAST_0_NAL_TYPE_PIC                        (8)
#define EBM_H264_NAL_LAST_0_NAL_TYPE_NON_IDR                    (1)
#define EBM_H264_NAL_LAST_0_NAL_TYPE_IDR                        (5)

#define EBM_H264_NAL_LAST_0_NEW_PACKET_SHIFT                    (29)
#define EBM_H264_NAL_LAST_0_NEW_PACKET_FIELD                    ((0x1) << EBM_H264_NAL_LAST_0_NEW_PACKET_SHIFT)
#define EBM_H264_NAL_LAST_0_NEW_PACKET_RANGE                    (29):(29)
#define EBM_H264_NAL_LAST_0_NEW_PACKET_ROW                      0

#define EBM_H264_NAL_LAST_0_OVER_FLOW_SHIFT                     (30)
#define EBM_H264_NAL_LAST_0_OVER_FLOW_FIELD                     ((0x1) << EBM_H264_NAL_LAST_0_OVER_FLOW_SHIFT)
#define EBM_H264_NAL_LAST_0_OVER_FLOW_RANGE                     (30):(30)
#define EBM_H264_NAL_LAST_0_OVER_FLOW_ROW                       0

#define EBM_H264_NAL_LAST_0_END_FRAME_SHIFT                     (31)
#define EBM_H264_NAL_LAST_0_END_FRAME_FIELD                     ((0x1) << EBM_H264_NAL_LAST_0_END_FRAME_SHIFT)
#define EBM_H264_NAL_LAST_0_END_FRAME_RANGE                     (31):(31)
#define EBM_H264_NAL_LAST_0_END_FRAME_ROW                       0

#define EBM_H264_NAL_LAST_0_FRAME_LENGTH_SHIFT                  (32)
#define EBM_H264_NAL_LAST_0_FRAME_LENGTH_FIELD                  ((0xffffffff) << EBM_H264_NAL_LAST_0_FRAME_LENGTH_SHIFT)
#define EBM_H264_NAL_LAST_0_FRAME_LENGTH_RANGE                  (63):(32)
#define EBM_H264_NAL_LAST_0_FRAME_LENGTH_ROW                    0


// Packet EBM_H264_BYTE_0
#define EBM_H264_BYTE_0_SIZE 128

#define EBM_H264_BYTE_0_FRAME_LENGTH_SHIFT                      (0)
#define EBM_H264_BYTE_0_FRAME_LENGTH_FIELD                      ((0x3fffff) << EBM_H264_BYTE_0_FRAME_LENGTH_SHIFT)
#define EBM_H264_BYTE_0_FRAME_LENGTH_RANGE                      (21):(0)
#define EBM_H264_BYTE_0_FRAME_LENGTH_ROW                        0

#define EBM_H264_BYTE_0_OVER_FLOW_SHIFT                 (29)
#define EBM_H264_BYTE_0_OVER_FLOW_FIELD                 ((0x1) << EBM_H264_BYTE_0_OVER_FLOW_SHIFT)
#define EBM_H264_BYTE_0_OVER_FLOW_RANGE                 (29):(29)
#define EBM_H264_BYTE_0_OVER_FLOW_ROW                   0

#define EBM_H264_BYTE_0_FRAME_TYPE_SHIFT                        (30)
#define EBM_H264_BYTE_0_FRAME_TYPE_FIELD                        ((0x3) << EBM_H264_BYTE_0_FRAME_TYPE_SHIFT)
#define EBM_H264_BYTE_0_FRAME_TYPE_RANGE                        (31):(30)
#define EBM_H264_BYTE_0_FRAME_TYPE_ROW                  0
#define EBM_H264_BYTE_0_FRAME_TYPE_I                    (0)
#define EBM_H264_BYTE_0_FRAME_TYPE_P                    (1)

#define EBM_H264_BYTE_0_FRAME_NUM_SHIFT                 (32)
#define EBM_H264_BYTE_0_FRAME_NUM_FIELD                 ((0xffff) << EBM_H264_BYTE_0_FRAME_NUM_SHIFT)
#define EBM_H264_BYTE_0_FRAME_NUM_RANGE                 (47):(32)
#define EBM_H264_BYTE_0_FRAME_NUM_ROW                   0

#define EBM_H264_BYTE_0_STREAM_ID_SHIFT                 (48)
#define EBM_H264_BYTE_0_STREAM_ID_FIELD                 ((0xff) << EBM_H264_BYTE_0_STREAM_ID_SHIFT)
#define EBM_H264_BYTE_0_STREAM_ID_RANGE                 (55):(48)
#define EBM_H264_BYTE_0_STREAM_ID_ROW                   0

#define EBM_H264_BYTE_0_BYTE_STREAM_SHIFT                       (59)
#define EBM_H264_BYTE_0_BYTE_STREAM_FIELD                       ((0x1) << EBM_H264_BYTE_0_BYTE_STREAM_SHIFT)
#define EBM_H264_BYTE_0_BYTE_STREAM_RANGE                       (59):(59)
#define EBM_H264_BYTE_0_BYTE_STREAM_ROW                 0

#define EBM_H264_BYTE_0_NAL_TYPE_SHIFT                  (60)
#define EBM_H264_BYTE_0_NAL_TYPE_FIELD                  ((0xf) << EBM_H264_BYTE_0_NAL_TYPE_SHIFT)
#define EBM_H264_BYTE_0_NAL_TYPE_RANGE                  (63):(60)
#define EBM_H264_BYTE_0_NAL_TYPE_ROW                    0
#define EBM_H264_BYTE_0_NAL_TYPE_SEQ                    (7)
#define EBM_H264_BYTE_0_NAL_TYPE_PIC                    (8)
#define EBM_H264_BYTE_0_NAL_TYPE_NON_IDR                        (1)
#define EBM_H264_BYTE_0_NAL_TYPE_IDR                    (5)

#define EBM_H264_BYTE_0_TIMESTAMP_SHIFT                 (64)
#define EBM_H264_BYTE_0_TIMESTAMP_FIELD                 ((0xffffffff) << EBM_H264_BYTE_0_TIMESTAMP_SHIFT)
#define EBM_H264_BYTE_0_TIMESTAMP_RANGE                 (95):(64)
#define EBM_H264_BYTE_0_TIMESTAMP_ROW                   0

#define EBM_H264_BYTE_0_ZERO_SHIFT                      (96)
#define EBM_H264_BYTE_0_ZERO_FIELD                      ((0xffffffff) << EBM_H264_BYTE_0_ZERO_SHIFT)
#define EBM_H264_BYTE_0_ZERO_RANGE                      (127):(96)
#define EBM_H264_BYTE_0_ZERO_ROW                        0

// --------------------------------------------------------------------------
// 
// Copyright (c) 2004, LWPU Corp.
// All Rights Reserved.
// 
// This is UNPUBLISHED PROPRIETARY SOURCE CODE of LWPU Corp.;
// the contents of this file may not be disclosed to third parties, copied or
// duplicated in any form, in whole or in part, without the prior written
// permission of LWPU Corp.
// 
// RESTRICTED RIGHTS LEGEND:
// Use, duplication or disclosure by the Government is subject to restrictions
// as set forth in subdivision (c)(1)(ii) of the Rights in Technical Data
// and Computer Software clause at DFARS 252.227-7013, and/or in similar or
// successor clauses in the FAR, DOD or NASA FAR Supplement. Unpublished -
// rights reserved under the Copyright Laws of the United States.
// 
// MPEG4 Encoder register definition
//
// The MPEG4 Encoder can perform simple profile DCT-based coding of a input data frames
// Motion Estimation/Compensation is support for P-frames (Predicted frames)
// Only YUV420 input data format is supported.
//
// The input frame data are assumed to be stored in input buffer(s) in the memory in planar
// YUV420 format. The input buffers can be organized as two sets of ring buffer. Each set can
// consists of a number of buffers of same size. Each input buffer must store the data in the
// planar YUV420 format therefore each buffer consists of a Y plane, U plane, and V plane where
// the size of the Y plane is 4 times the size of the U or V plane. The Y, U, and V planes can
// be organized contiguously or scattered in separate area of the memory.
// Each frame can be stored as multiple buffers in the memory but each frame must start on a
// buffer boundary. Y plane must be stored as multiple of 16x16 macro blocks both horizontally
// and vertically. The input memory is restricted such that the macro blocks are aligned on
// 128-bit memory boundary so that blocks of data can be read efficiently.
// Similarly, U and V planes must be stored as multiple of 8x8 blocks both horizontally and
// and vertically. The input memory is restricted such that the blocks are aligned on 64-bit
// memory boundary so that the blocks of data can be read efficiently.
// When an input buffer is ready to be encoded, the MPEG4 Encoder can be triggered to start
// or resume encoding. MPEG4 encoder can be triggered to start/resume encoding by Video Input
// (VI) module or by Encoder Pre-Processor (EPP) module or by host.
//
// When a "new buffer" trigger is sent either by EPP or by host, a start of frame
// indicator should be sent also to indicate whether the new input buffer contains the
// beginning of a a frame.
// 
// The MPEG4 Encoder can be enabled by the host with the following sequence:
//  a. Enable MPEG4 Encoder clock
//  b. Program parameters for MPEG4 encoding in the MPEG4 Encoder registers.
//  c. Enable MPEG4 Encoder module by setting MPEG4_ENABLE register bit to START.
// When MPEG4 Encoder module is enabled it will wait for "New Buffer" commands to be issued
// either by the host (by writing into NEW_BUFFER register) or by VI or EPP module
// (by detecting the "New Buffer" pulse). At any time, only one source can trigger MPEG4
// Encoder.
// After enabling MPEG4 Encoder, the encoding process starts when "New Buffer" command is
// received with "Start of Frame" flag enabled.
// The ME_ENABLE register bit should be kept at START state when encoding multiple frames
// and then set to STOP when encoding the last frame. When encoding a single frame, ME_ENABLE
// register bit can be reset to STOP state immediately after setting ME_ENABLE to START.
// After ME_ENABLE is set to STOP, the MPEG4 Encoder will continue encoding until the end of
// frame. After ME_ENABLE is set to START, if there is no "New Buffer" command with "Start
// of Frame" received, the MPEG4 Encoder will continue to be enabled even after ME_ENABLE is
// set back to STOP. In this scenario, the MPEG4 Encoder can be reset/disabled by disabling the
// MPEG4 Encoder clock.
// A raise can be sent by the host or by VI or EPP module.
// The processing stages are:
//  a. CKG - Clock Generator module that monitors the state of MPEG4_ENABLE and generates
//     the enables for second-level clocks.
//  b. IMI - Input Memory Interface which consists of a 128-bit read/write. An arbiter to handle
//     several clients reading/writing
//  c. MEST - Motion Estimation module that processes on 16x16 blocks. It requests memory access
//     for Input Video Y component, it also caches the reference frame (3x3 windows for the 16x16
//     of the previous frame). SAD and MAD comparison makes decision of macroblock type and motion
//     vectors.
//  d. MCMP - Motion Compensation, take away the residue of the prediction. Both Y and UV
//  e. FDCT - Forward DCT module that processes 8x8 block of input data and generates 64 DCT
//     coefficients in 32 clock cycles. 
//  f. QRZ - Quantization, Rounding and Zig-zag that processes one DCT coefficient per clock
//     cycle in a pre-defined zig-zag ordering.
//  g. ACPRD - AC/DC prediction for INTRA macroblock only.
//  h. ZRLE - Zigzag scan and Run-Length to get zeros run and level for non-zero quantized 
//      coefficients 
//  i. RECN - Reconstructed the frame for next frame reference
//  j. VLC - VLC coding the run length result
//  k. VLCDMA - Generate Link List in the Memory for VLC output
//  l. DMAOUT - Pull out DMA data from host
//  f. RATECN - Rate Control, suggesting Qp for next Macroblock
// Buffer controls and other parameters that controls the MPEG4 encoder operation are programmed
// in 32-bit registers that can be written/read from host register interface.
// 
// When input source comes from host, data must be stored in input buffer set in memory and
// this module supports one buffer set: input buffer set 0. In the future more input buffer
// sets may be supported.
// Each input buffer set consists of a programmable number of buffers (from 1 to 255 for input
// buffer set 0 and 0 to 255 for subsequent input buffer set if there is any). Each input buffer
// consists of a programmable number of macro block rows (from 1 to 256)) and a programmable
// number of macro blocks per macro block row.
// Data in the input buffer must be stored in YUV420 planar format therefore the each input
// buffer set consists of a set for Y, U, and V planes.
// Also in this format, input data must be arranged in multiple of macro blocks horizontally
// and in multiple of macro block rows vertically.
// Each macro block consists of four (2x2) Y blocks, one U block, and one V block and each
// Y/U/V block consists of 8x8 (64) bytes.
// Y buffer must be aligned to macro block (256-byte) boundary in the memory. U and V buffers
// must be aligned to to block (64-byte) boundary in the memory.
// Buffers in the same input buffer set must have the same size horizontally and vertically.
// Input buffer horizontal size is determined by the horizontal scan size (H_SCAN_SIZE).
// Input buffer chroma line stride determines the distance between vertically adjacent pixels
// in the U and V buffers and this is restricted to multiple of 8-bytes so that U and V block
// words are aligned in 8-byte boundary in the memory. An option is provided to set the input
// buffer luma line stride for Y buffer to be either the same as chroma line stride or twice
// the chroma line stride. If luma line stride is set to be the same as the chroma line stride
// then the chroma line stride must be programmed to multiple of 16-bytes so that luma macro
// block words are aligned to 16-byte boundary in the memory.
// Input buffer vertical size determines the number of macro block rows in each buffer in the
// buffer set.
// Input buffer chroma buffer stride determines the distance from start address of one chroma
// buffer to the start address of the next chroma buffer and this is restricted to multiple
// of 64-bytes so that U and V blocks are aligned to 64-byte boundary in the memory.
// An option is provided to set the input buffer luma buffer stride for Y buffer to be either
// the same as chroma buffer stride or four times the chroma buffer stride. If luma buffer
// stride is set to be the same as the chroma buffer stride then the chroma buffer stride
// must be programmed to multiple of 256-bytes so that luma macro blocks are aligned to
// 256-byte boundary in the memory.
//
// When input source comes from VI/EPP, the MPEG4 Encoder expects the respective module to
// send a "New Buffer" pulse. This signal is pulse synchronized by the MPEG4 Encoder module.
// A buffer ID and the "Start of Frame" flag are sent together with the "New Buffer" pulse and
// they are sampled when the pulse synchronized "New Buffer" pulse is received.
// 
// The output buffer consists of linked lists. Each linked-list corresponds to a DMA chunk.
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// Include internal defines.
// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
// 
// Copyright (c) 2004, LWPU Corp.
// All Rights Reserved.
// 
// This is UNPUBLISHED PROPRIETARY SOURCE CODE of LWPU Corp.;
// the contents of this file may not be disclosed to third parties, copied or
// duplicated in any form, in whole or in part, without the prior written
// permission of LWPU Corp.
// 
// RESTRICTED RIGHTS LEGEND:
// Use, duplication or disclosure by the Government is subject to restrictions
// as set forth in subdivision (c)(1)(ii) of the Rights in Technical Data
// and Computer Software clause at DFARS 252.227-7013, and/or in similar or
// successor clauses in the FAR, DOD or NASA FAR Supplement. Unpublished -
// rights reserved under the Copyright Laws of the United States.
// 
// --------------------------------------------------------------------------
// 
// -------------------------------------------------------------------
// FIFO parameters (ME legacy)
// -------------------------------------------------------------------
// Host IF FIFOs
#define LW_MPEA_HOST_RDFIFO_DEPTH       5
#define LW_MPEA_HOST_RDFIFO_WIDTH       38
#define LW_MPEA_HOST_WRFIFO_DEPTH       5
#define LW_MPEA_HOST_WRFIFO_WIDTH       55
//this is the depth/width of the fifo used to queue up "new buffer" commands
#define LW_MPEA_IB_FIFO_DEPTH   8
#define LW_MPEA_IB_FIFO_WIDTH   10
#define LW_MPEA_IB_NEW_BUFFER_SIZE      9
#define LW_MPEA_FETCH_SM_STATE_VECTOR_SZ        3
#define LW_MPEA_XFR_SM_STATE_VECTOR_SZ  2
#define LW_MPEA_NGBR_RAM_W      82
// CBR RC Input Bit Width
#define BITS_PER_FRAME_SZ       28
#define BUPFMAD_DO_SZ   24
#define BU_NUM_SZ       10
#define LWRRENT_BUFFER_FULLNESS_SZ      28
#define LWRRENT_FRAME_AVG_MAD_SZ        30
// define LWRRENT_FRAME_BITS_I_SZ                  32;
#define LWRRENT_FRAME_BITS_SZ   25
#define LWRRENT_FRAME_MAD_SZ    26
#define LWRRENT_FRAME_TEXT_BITS_MPEC_SZ 28
#define DDQUANT_SZ      6
#define GOP_OVERDUE_SZ  1
#define GOV_PERIOD_SZ   16
#define INITIAL_DELAY_OFFSET_SZ 28
#define INITIAL_QP_SZ   6
#define INTRA_INTER_SAD_SZ      17
#define LOWER_BOUND_SZ  33
#define MAD_PIC_1_SZ    21
#define MAX_DQP_FROM_PREV_FRM_SZ        6
#define MAX_QP_SZ       6
#define MIN_QP_SZ       6
#define MX1_SZ  21
#define MY_INITIAL_QP_SZ        6
#define NUMBER_OF_BASIC_UNIT_HEADER_BITS_SZ     19
#define NUMBER_OF_BASIC_UNIT_SZ 10
#define NUMBER_OF_BASIC_UNIT_MPEG4_SZ   13
#define NUMBER_OF_BASIC_UNIT_TEXTURE_BITS_SZ    22
#define NUMBER_OF_GOP_SZ        16
#define NUMBER_OF_HEADER_BITS_SZ        25
#define NUM_CODED_FRAMES_I_SZ   26
#define NUM_CODED_FRAMES_SZ     26
#define NUM_CODED_P_FRAMES_SZ   26
#define NUM_P_PICTURE_SZ        16
#define PF_BU_MB_MAD_SZ 17
#define PIC_SIZE_IN_MBS_SZ      13
#define PIC_WIDTH_IN_MBS_SZ     7
#define PREV_FRAME_MAD_SZ       17
#define P_AVE_FRAME_QP_SZ       6
#define P_AVE_HEADER_BITS_1_SZ  25
#define P_AVE_HEADER_BITS_2_SZ  25
#define P_AVE_HEADER_BITS_3_SZ  25
#define QC_SZ   6
// define QP.*_SZ                                  6;
#define QP_SZ   6
#define QP_TOTAL_SZ     19
#define RC_BASIC_UNIT_SIZE_SZ   7
#define REMAINING_BITS_SZ       33
#define REMAINING_P_FRAMES_SZ   16
#define SLICE_TYPE_SZ   1
#define TARGET_BITS_SZ  33
#define TARGET_BUFFER_LEVEL_FLAG_SZ     1
#define TOTAL_FRAME_MAD_SZ      30
#define TOTAL_FRAME_QP_SZ       19
#define TOTAL_MAD_BASIC_UNIT_SZ 24
#define TOTAL_NUM_OF_BASIC_UNIT_SZ      10
#define TOTAL_QP_FOR_P_PICTURE_SZ       22
#define UPM_CNT_SZ      13
#define UPPER_BOUND1_SZ 33
#define UPPER_BOUND2_SZ 33
// PACE_RC WIDTH defines for internal variables
// GetQP
#define TARGET_BITS_DELTA_WIDTH 22
#define BUFFER_FULLNESS_WIDTH   29
#define P_AVERAGE_QP_WIDTH      6
#define INT_QP_WIDTH    8
#define LWR_BU_MAD_WIDTH        24
#define TARGETBITS_TEMP_WIDTH   22
#define TOTALBUMAD_TEMP_WIDTH   12
// InitGOP
#define QP_LAST_FRAME_WIDTH     6
// other funcs
#define QP_STEP_SZ      13
#define M_RP_SZ 32
#define WINDOW_SIZE_SZ  5
#define REAL_WINDOW_SIZE_SZ     5
#define ERROR_SZ        23
#define A00_SZ  11
#define A01_SZ  22
#define A11_SZ  32
#define B0_SZ   36
#define B1_SZ   32
#define AB_VALUE_SZ     32
// --------------------------------------------------------------------------
// Include internal mpe module definition
// --------------------------------------------------------------------------
// move this to mpe when top level wrapper is available
//  #include "armpe_internal.spec"
// --------------------------------------------------------------------------
// MPEA parameters
// --------------------------------------------------------------------------
// Max horizontal/vertical resolution
#define LW_MPEA_MAX_H   1920
#define LW_MPEA_MAX_V   1088
#define LW_MPEA_H_LOG2  12
#define LW_MPEA_V_LOG2  12
#define LW_MPEA_H_WDLOG2        4
#define LW_MPEA_V_WDLOG2        4
// --------------------------------------------------------------------------
// MPEA packets
// --------------------------------------------------------------------------
// Reserved size of 256 bytes for frame parameter section in normal mode 
// no MPEA bypass, in bits
#define MPE_FRAME_PARAM_NORMAL_FSIZE    2048
// Reserved size of 8k bytes for frame parameter section in MPEA bypass mode 
// in bits
#define MPE_FRAME_PARAM_MPEA_BYPASS_FSIZE       65536

// Packet STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2
#define STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_SIZE 10

// FRAME_START is set for the buffer contain
// the beginning of a frame.
#define STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_FRAME_START_SHIFT                   (0)
#define STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_FRAME_START_FIELD                   ((0x1) << STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_FRAME_START_SHIFT)
#define STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_FRAME_START_RANGE                   (0):(0)
#define STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_FRAME_START_ROW                     0

// FRAME_END is set when a buffer contain the end of a frame.
#define STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_FRAME_END_SHIFT                     (1)
#define STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_FRAME_END_FIELD                     ((0x1) << STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_FRAME_END_SHIFT)
#define STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_FRAME_END_RANGE                     (1):(1)
#define STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_FRAME_END_ROW                       0

// Current buffer index number. Y/U/V should have the same index number.
#define STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_LWR_BUFFER_INDEX_SHIFT                      (2)
#define STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_LWR_BUFFER_INDEX_FIELD                      ((0x3f) << STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_LWR_BUFFER_INDEX_SHIFT)
#define STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_LWR_BUFFER_INDEX_RANGE                      (7):(2)
#define STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_LWR_BUFFER_INDEX_ROW                        0

// MPE parameter buffer index number. 
#define STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_MPE_BUFFER_INDEX_SHIFT                      (8)
#define STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_MPE_BUFFER_INDEX_FIELD                      ((0x3) << STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_MPE_BUFFER_INDEX_SHIFT)
#define STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_MPE_BUFFER_INDEX_RANGE                      (9):(8)
#define STREAM_VIDEO_ASYNC_CONTROL_MPEA2VP2_MPE_BUFFER_INDEX_ROW                        0


// Packet MPE_MB_PARAM
#define MPE_MB_PARAM_SIZE 512

// 0=I, 1=P 
#define MPE_MB_PARAM_MB_TYPE_SHIFT                      (0)
#define MPE_MB_PARAM_MB_TYPE_FIELD                      ((0x1) << MPE_MB_PARAM_MB_TYPE_SHIFT)
#define MPE_MB_PARAM_MB_TYPE_RANGE                      (0):(0)
#define MPE_MB_PARAM_MB_TYPE_ROW                        0
#define MPE_MB_PARAM_MB_TYPE_INTRA                      (0)
#define MPE_MB_PARAM_MB_TYPE_INTER                      (1)

// H264 mode
// 01 = 16x16
// 00 = 16x8 
// 11 = 8x16
// 10 = 8x8
// MPEG4/H263 mode
// 01 = 16x16
// 10 = 8x8 
#define MPE_MB_PARAM_MV_TYPE_SHIFT                      (1)
#define MPE_MB_PARAM_MV_TYPE_FIELD                      ((0x3) << MPE_MB_PARAM_MV_TYPE_SHIFT)
#define MPE_MB_PARAM_MV_TYPE_RANGE                      (2):(1)
#define MPE_MB_PARAM_MV_TYPE_ROW                        0
#define MPE_MB_PARAM_MV_TYPE_P16x16                     (0)
#define MPE_MB_PARAM_MV_TYPE_P16x8                      (1)
#define MPE_MB_PARAM_MV_TYPE_P8x16                      (2)
#define MPE_MB_PARAM_MV_TYPE_P8x8                       (3)

// Frame type: 0=I, 1=P 
#define MPE_MB_PARAM_FRAME_TYPE_SHIFT                   (3)
#define MPE_MB_PARAM_FRAME_TYPE_FIELD                   ((0x1) << MPE_MB_PARAM_FRAME_TYPE_SHIFT)
#define MPE_MB_PARAM_FRAME_TYPE_RANGE                   (3):(3)
#define MPE_MB_PARAM_FRAME_TYPE_ROW                     0
#define MPE_MB_PARAM_FRAME_TYPE_IFRAME                  (0)
#define MPE_MB_PARAM_FRAME_TYPE_PFRAME                  (1)

// Neighbor Availability Map - based on frame boundary only
// Bit 0 - Left MB
// Bit 1 - Top Left MB
// Bit 2 - Top MB
// Bit 3 - Top Right MB
// 0 = MB not available
// 1 = MB available
#define MPE_MB_PARAM_NB_MAP_SHIFT                       (4)
#define MPE_MB_PARAM_NB_MAP_FIELD                       ((0xf) << MPE_MB_PARAM_NB_MAP_SHIFT)
#define MPE_MB_PARAM_NB_MAP_RANGE                       (7):(4)
#define MPE_MB_PARAM_NB_MAP_ROW                 0

#define MPE_MB_PARAM_SLICE_GROUP_ID_SHIFT                       (8)
#define MPE_MB_PARAM_SLICE_GROUP_ID_FIELD                       ((0x7) << MPE_MB_PARAM_SLICE_GROUP_ID_SHIFT)
#define MPE_MB_PARAM_SLICE_GROUP_ID_RANGE                       (10):(8)
#define MPE_MB_PARAM_SLICE_GROUP_ID_ROW                 0

#define MPE_MB_PARAM_FIRST_MB_IN_SLICE_GROUP_SHIFT                      (11)
#define MPE_MB_PARAM_FIRST_MB_IN_SLICE_GROUP_FIELD                      ((0x1) << MPE_MB_PARAM_FIRST_MB_IN_SLICE_GROUP_SHIFT)
#define MPE_MB_PARAM_FIRST_MB_IN_SLICE_GROUP_RANGE                      (11):(11)
#define MPE_MB_PARAM_FIRST_MB_IN_SLICE_GROUP_ROW                        0
#define MPE_MB_PARAM_FIRST_MB_IN_SLICE_GROUP_FALSE                      (0)
#define MPE_MB_PARAM_FIRST_MB_IN_SLICE_GROUP_TRUE                       (1)

#define MPE_MB_PARAM_SLICE_TYPE_SHIFT                   (12)
#define MPE_MB_PARAM_SLICE_TYPE_FIELD                   ((0x1) << MPE_MB_PARAM_SLICE_TYPE_SHIFT)
#define MPE_MB_PARAM_SLICE_TYPE_RANGE                   (12):(12)
#define MPE_MB_PARAM_SLICE_TYPE_ROW                     0
#define MPE_MB_PARAM_SLICE_TYPE_ISLICE                  (0)
#define MPE_MB_PARAM_SLICE_TYPE_PSLICE                  (1)

// SAD - MPEG4/H263 P MB
// MAD - MPEG4/H263 I MB
// SATD - H264
#define MPE_MB_PARAM_SATD_SHIFT                 (16)
#define MPE_MB_PARAM_SATD_FIELD                 ((0xffff) << MPE_MB_PARAM_SATD_SHIFT)
#define MPE_MB_PARAM_SATD_RANGE                 (31):(16)
#define MPE_MB_PARAM_SATD_ROW                   0

// Frame Number
#define MPE_MB_PARAM_FRAME_NUM_SHIFT                    (32)
#define MPE_MB_PARAM_FRAME_NUM_FIELD                    ((0xffff) << MPE_MB_PARAM_FRAME_NUM_SHIFT)
#define MPE_MB_PARAM_FRAME_NUM_RANGE                    (47):(32)
#define MPE_MB_PARAM_FRAME_NUM_ROW                      0

// Valid at framestart 
#define MPE_MB_PARAM_FRAME_BIT_BUDGET_SHIFT                     (48)
#define MPE_MB_PARAM_FRAME_BIT_BUDGET_FIELD                     ((0xffff) << MPE_MB_PARAM_FRAME_BIT_BUDGET_SHIFT)
#define MPE_MB_PARAM_FRAME_BIT_BUDGET_RANGE                     (63):(48)
#define MPE_MB_PARAM_FRAME_BIT_BUDGET_ROW                       0

// {11,10,9,8}-Ref ID {Y3,Y2,Y1,Y0}
// Ref ID Values
// 0 - Short Term Ref 0
// 1 - Short Term Ref 1 or Long Term Ref Frame
#define MPE_MB_PARAM_REF_ID_SHIFT                       (64)
#define MPE_MB_PARAM_REF_ID_FIELD                       ((0xf) << MPE_MB_PARAM_REF_ID_SHIFT)
#define MPE_MB_PARAM_REF_ID_RANGE                       (67):(64)
#define MPE_MB_PARAM_REF_ID_ROW                 0

// Previous Frame type: 0 = Short Term, 1 = Long Term 
#define MPE_MB_PARAM_PREV_REF_FRAME_TYPE_SHIFT                  (68)
#define MPE_MB_PARAM_PREV_REF_FRAME_TYPE_FIELD                  ((0x1) << MPE_MB_PARAM_PREV_REF_FRAME_TYPE_SHIFT)
#define MPE_MB_PARAM_PREV_REF_FRAME_TYPE_RANGE                  (68):(68)
#define MPE_MB_PARAM_PREV_REF_FRAME_TYPE_ROW                    0
#define MPE_MB_PARAM_PREV_REF_FRAME_TYPE_SHORT                  (0)
#define MPE_MB_PARAM_PREV_REF_FRAME_TYPE_LONG                   (1)

// MV0X_REF0 
#define MPE_MB_PARAM_MV0X_REF0_SHIFT                    (256)
#define MPE_MB_PARAM_MV0X_REF0_FIELD                    ((0xffff) << MPE_MB_PARAM_MV0X_REF0_SHIFT)
#define MPE_MB_PARAM_MV0X_REF0_RANGE                    (271):(256)
#define MPE_MB_PARAM_MV0X_REF0_ROW                      0

// MV0Y_REF0 
#define MPE_MB_PARAM_MV0Y_REF0_SHIFT                    (272)
#define MPE_MB_PARAM_MV0Y_REF0_FIELD                    ((0xffff) << MPE_MB_PARAM_MV0Y_REF0_SHIFT)
#define MPE_MB_PARAM_MV0Y_REF0_RANGE                    (287):(272)
#define MPE_MB_PARAM_MV0Y_REF0_ROW                      0

// MV1X_REF0 
#define MPE_MB_PARAM_MV1X_REF0_SHIFT                    (288)
#define MPE_MB_PARAM_MV1X_REF0_FIELD                    ((0xffff) << MPE_MB_PARAM_MV1X_REF0_SHIFT)
#define MPE_MB_PARAM_MV1X_REF0_RANGE                    (303):(288)
#define MPE_MB_PARAM_MV1X_REF0_ROW                      0

// MV1Y_REF0 
#define MPE_MB_PARAM_MV1Y_REF0_SHIFT                    (304)
#define MPE_MB_PARAM_MV1Y_REF0_FIELD                    ((0xffff) << MPE_MB_PARAM_MV1Y_REF0_SHIFT)
#define MPE_MB_PARAM_MV1Y_REF0_RANGE                    (319):(304)
#define MPE_MB_PARAM_MV1Y_REF0_ROW                      0

// MV2X_REF0 
#define MPE_MB_PARAM_MV2X_REF0_SHIFT                    (320)
#define MPE_MB_PARAM_MV2X_REF0_FIELD                    ((0xffff) << MPE_MB_PARAM_MV2X_REF0_SHIFT)
#define MPE_MB_PARAM_MV2X_REF0_RANGE                    (335):(320)
#define MPE_MB_PARAM_MV2X_REF0_ROW                      0

// MV2Y_REF0 
#define MPE_MB_PARAM_MV2Y_REF0_SHIFT                    (336)
#define MPE_MB_PARAM_MV2Y_REF0_FIELD                    ((0xffff) << MPE_MB_PARAM_MV2Y_REF0_SHIFT)
#define MPE_MB_PARAM_MV2Y_REF0_RANGE                    (351):(336)
#define MPE_MB_PARAM_MV2Y_REF0_ROW                      0

// MV3X_REF0 
#define MPE_MB_PARAM_MV3X_REF0_SHIFT                    (352)
#define MPE_MB_PARAM_MV3X_REF0_FIELD                    ((0xffff) << MPE_MB_PARAM_MV3X_REF0_SHIFT)
#define MPE_MB_PARAM_MV3X_REF0_RANGE                    (367):(352)
#define MPE_MB_PARAM_MV3X_REF0_ROW                      0

// MV3Y_REF0 
#define MPE_MB_PARAM_MV3Y_REF0_SHIFT                    (368)
#define MPE_MB_PARAM_MV3Y_REF0_FIELD                    ((0xffff) << MPE_MB_PARAM_MV3Y_REF0_SHIFT)
#define MPE_MB_PARAM_MV3Y_REF0_RANGE                    (383):(368)
#define MPE_MB_PARAM_MV3Y_REF0_ROW                      0

// MV0X_REF1 
#define MPE_MB_PARAM_MV0X_REF1_SHIFT                    (384)
#define MPE_MB_PARAM_MV0X_REF1_FIELD                    ((0xffff) << MPE_MB_PARAM_MV0X_REF1_SHIFT)
#define MPE_MB_PARAM_MV0X_REF1_RANGE                    (399):(384)
#define MPE_MB_PARAM_MV0X_REF1_ROW                      0

// MV0Y_REF1 
#define MPE_MB_PARAM_MV0Y_REF1_SHIFT                    (400)
#define MPE_MB_PARAM_MV0Y_REF1_FIELD                    ((0xffff) << MPE_MB_PARAM_MV0Y_REF1_SHIFT)
#define MPE_MB_PARAM_MV0Y_REF1_RANGE                    (415):(400)
#define MPE_MB_PARAM_MV0Y_REF1_ROW                      0

// MV1X_REF1 
#define MPE_MB_PARAM_MV1X_REF1_SHIFT                    (416)
#define MPE_MB_PARAM_MV1X_REF1_FIELD                    ((0xffff) << MPE_MB_PARAM_MV1X_REF1_SHIFT)
#define MPE_MB_PARAM_MV1X_REF1_RANGE                    (431):(416)
#define MPE_MB_PARAM_MV1X_REF1_ROW                      0

// MV1Y_REF1 
#define MPE_MB_PARAM_MV1Y_REF1_SHIFT                    (432)
#define MPE_MB_PARAM_MV1Y_REF1_FIELD                    ((0xffff) << MPE_MB_PARAM_MV1Y_REF1_SHIFT)
#define MPE_MB_PARAM_MV1Y_REF1_RANGE                    (447):(432)
#define MPE_MB_PARAM_MV1Y_REF1_ROW                      0

// MV2X_REF1 
#define MPE_MB_PARAM_MV2X_REF1_SHIFT                    (448)
#define MPE_MB_PARAM_MV2X_REF1_FIELD                    ((0xffff) << MPE_MB_PARAM_MV2X_REF1_SHIFT)
#define MPE_MB_PARAM_MV2X_REF1_RANGE                    (463):(448)
#define MPE_MB_PARAM_MV2X_REF1_ROW                      0

// MV2Y_REF1 
#define MPE_MB_PARAM_MV2Y_REF1_SHIFT                    (464)
#define MPE_MB_PARAM_MV2Y_REF1_FIELD                    ((0xffff) << MPE_MB_PARAM_MV2Y_REF1_SHIFT)
#define MPE_MB_PARAM_MV2Y_REF1_RANGE                    (479):(464)
#define MPE_MB_PARAM_MV2Y_REF1_ROW                      0

// MV3X_REF1 
#define MPE_MB_PARAM_MV3X_REF1_SHIFT                    (480)
#define MPE_MB_PARAM_MV3X_REF1_FIELD                    ((0xffff) << MPE_MB_PARAM_MV3X_REF1_SHIFT)
#define MPE_MB_PARAM_MV3X_REF1_RANGE                    (495):(480)
#define MPE_MB_PARAM_MV3X_REF1_ROW                      0

// MV3Y_REF1 
#define MPE_MB_PARAM_MV3Y_REF1_SHIFT                    (496)
#define MPE_MB_PARAM_MV3Y_REF1_FIELD                    ((0xffff) << MPE_MB_PARAM_MV3Y_REF1_SHIFT)
#define MPE_MB_PARAM_MV3Y_REF1_RANGE                    (511):(496)
#define MPE_MB_PARAM_MV3Y_REF1_ROW                      0


// Packet MPE_PRED_PARAM
#define MPE_PRED_PARAM_SIZE 2048

// Predicted Luma Pixels
// {127:120,119:112,111:104,103:96,95:88,87:80,79:72,71:64,63:56,55:48,47:40,39:32,31:24,23:16,15:8,7:0}
// Row0 Pixels {15, 14, 13, ... 0} 
// {127:120,119:112,111:104,103:96,95:88,87:80,79:72,71:64,63:56,55:48,47:40,39:32,31:24,23:16,15:8,7:0}
// Row1 Pixels {15, 14, 13, ... 0} 
// {127:120,119:112,111:104,103:96,95:88,87:80,79:72,71:64,63:56,55:48,47:40,39:32,31:24,23:16,15:8,7:0}
// Row2 Pixels {15, 14, 13, ... 0} 
// {127:120,119:112,111:104,103:96,95:88,87:80,79:72,71:64,63:56,55:48,47:40,39:32,31:24,23:16,15:8,7:0}
// Row3 Pixels {15, 14, 13, ... 0} 
// {127:120,119:112,111:104,103:96,95:88,87:80,79:72,71:64,63:56,55:48,47:40,39:32,31:24,23:16,15:8,7:0}
// Row4 Pixels {15, 14, 13, ... 0} 
// {127:120,119:112,111:104,103:96,95:88,87:80,79:72,71:64,63:56,55:48,47:40,39:32,31:24,23:16,15:8,7:0}
// Row5 Pixels {15, 14, 13, ... 0} 
// {127:120,119:112,111:104,103:96,95:88,87:80,79:72,71:64,63:56,55:48,47:40,39:32,31:24,23:16,15:8,7:0}
// Row6 Pixels {15, 14, 13, ... 0} 
// {127:120,119:112,111:104,103:96,95:88,87:80,79:72,71:64,63:56,55:48,47:40,39:32,31:24,23:16,15:8,7:0}
// Row7 Pixels {15, 14, 13, ... 0} 
// {127:120,119:112,111:104,103:96,95:88,87:80,79:72,71:64,63:56,55:48,47:40,39:32,31:24,23:16,15:8,7:0}
// Row8 Pixels {15, 14, 13, ... 0} 
// {127:120,119:112,111:104,103:96,95:88,87:80,79:72,71:64,63:56,55:48,47:40,39:32,31:24,23:16,15:8,7:0}
// Row9 Pixels {15, 14, 13, ... 0} 
// {127:120,119:112,111:104,103:96,95:88,87:80,79:72,71:64,63:56,55:48,47:40,39:32,31:24,23:16,15:8,7:0}
// Row10 Pixels {15, 14, 13, ... 0} 
// {127:120,119:112,111:104,103:96,95:88,87:80,79:72,71:64,63:56,55:48,47:40,39:32,31:24,23:16,15:8,7:0}
// Row11 Pixels {15, 14, 13, ... 0} 
// {127:120,119:112,111:104,103:96,95:88,87:80,79:72,71:64,63:56,55:48,47:40,39:32,31:24,23:16,15:8,7:0}
// Row12 Pixels {15, 14, 13, ... 0} 
// {127:120,119:112,111:104,103:96,95:88,87:80,79:72,71:64,63:56,55:48,47:40,39:32,31:24,23:16,15:8,7:0}
// Row13 Pixels {15, 14, 13, ... 0} 
// {127:120,119:112,111:104,103:96,95:88,87:80,79:72,71:64,63:56,55:48,47:40,39:32,31:24,23:16,15:8,7:0}
// Row14 Pixels {15, 14, 13, ... 0} 
// {127:120,119:112,111:104,103:96,95:88,87:80,79:72,71:64,63:56,55:48,47:40,39:32,31:24,23:16,15:8,7:0}
// Row15 Pixels {15, 14, 13, ... 0} 
// {127:120,119:112,111:104,103:96,95:88,87:80,79:72,71:64,63:56,55:48,47:40,39:32,31:24,23:16,15:8,7:0}
#define MPE_PRED_PARAM_LUMA_SHIFT                       (0)
#define MPE_PRED_PARAM_LUMA_FIELD                       ((0xffffffff) << MPE_PRED_PARAM_LUMA_SHIFT)
#define MPE_PRED_PARAM_LUMA_RANGE                       (2047):(0)
#define MPE_PRED_PARAM_LUMA_ROW                 0


// Packet MPEC2MPEA_PARAM
#define MPEC2MPEA_PARAM_SIZE 28

// Frame bit length in bits 
#define MPEC2MPEA_PARAM_FRAME_BIT_LEN_SHIFT                     (0)
#define MPEC2MPEA_PARAM_FRAME_BIT_LEN_FIELD                     ((0x3ffff) << MPEC2MPEA_PARAM_FRAME_BIT_LEN_SHIFT)
#define MPEC2MPEA_PARAM_FRAME_BIT_LEN_RANGE                     (17):(0)
#define MPEC2MPEA_PARAM_FRAME_BIT_LEN_ROW                       0

// DMA overflow 
#define MPEC2MPEA_PARAM_DMA_OVERFLOW_SHIFT                      (18)
#define MPEC2MPEA_PARAM_DMA_OVERFLOW_FIELD                      ((0x1) << MPEC2MPEA_PARAM_DMA_OVERFLOW_SHIFT)
#define MPEC2MPEA_PARAM_DMA_OVERFLOW_RANGE                      (18):(18)
#define MPEC2MPEA_PARAM_DMA_OVERFLOW_ROW                        0

// Skip next frame if the DMA buffers exceed predefined thresholds 
#define MPEC2MPEA_PARAM_SKIP_NEXT_FRAME_SHIFT                   (19)
#define MPEC2MPEA_PARAM_SKIP_NEXT_FRAME_FIELD                   ((0x1) << MPEC2MPEA_PARAM_SKIP_NEXT_FRAME_SHIFT)
#define MPEC2MPEA_PARAM_SKIP_NEXT_FRAME_RANGE                   (19):(19)
#define MPEC2MPEA_PARAM_SKIP_NEXT_FRAME_ROW                     0

// Qp of the last MB of the frame
#define MPEC2MPEA_PARAM_QP_SHIFT                        (20)
#define MPEC2MPEA_PARAM_QP_FIELD                        ((0x3f) << MPEC2MPEA_PARAM_QP_SHIFT)
#define MPEC2MPEA_PARAM_QP_RANGE                        (25):(20)
#define MPEC2MPEA_PARAM_QP_ROW                  0

// Frame is done
#define MPEC2MPEA_PARAM_FRAME_DONE_SHIFT                        (26)
#define MPEC2MPEA_PARAM_FRAME_DONE_FIELD                        ((0x1) << MPEC2MPEA_PARAM_FRAME_DONE_SHIFT)
#define MPEC2MPEA_PARAM_FRAME_DONE_RANGE                        (26):(26)
#define MPEC2MPEA_PARAM_FRAME_DONE_ROW                  0

// Row Done
#define MPEC2MPEA_PARAM_ROW_DONE_SHIFT                  (27)
#define MPEC2MPEA_PARAM_ROW_DONE_FIELD                  ((0x1) << MPEC2MPEA_PARAM_ROW_DONE_SHIFT)
#define MPEC2MPEA_PARAM_ROW_DONE_RANGE                  (27):(27)
#define MPEC2MPEA_PARAM_ROW_DONE_ROW                    0

// Packet definition for frame parameter section in normal mode (no MPEA bypass)

// Packet MPE_FRAME_PARAM_NORMAL
#define MPE_FRAME_PARAM_NORMAL_SIZE 1152

// 
#define MPE_FRAME_PARAM_NORMAL_VOL_CTRL_SHIFT                   (0)
#define MPE_FRAME_PARAM_NORMAL_VOL_CTRL_FIELD                   ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_VOL_CTRL_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_VOL_CTRL_RANGE                   (31):(0)
#define MPE_FRAME_PARAM_NORMAL_VOL_CTRL_ROW                     0

// 
#define MPE_FRAME_PARAM_NORMAL_WIDTH_HEIGHT_SHIFT                       (32)
#define MPE_FRAME_PARAM_NORMAL_WIDTH_HEIGHT_FIELD                       ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_WIDTH_HEIGHT_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_WIDTH_HEIGHT_RANGE                       (63):(32)
#define MPE_FRAME_PARAM_NORMAL_WIDTH_HEIGHT_ROW                 0

// 
#define MPE_FRAME_PARAM_NORMAL_REF_Y_ADDR_SHIFT                 (64)
#define MPE_FRAME_PARAM_NORMAL_REF_Y_ADDR_FIELD                 ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_REF_Y_ADDR_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_REF_Y_ADDR_RANGE                 (95):(64)
#define MPE_FRAME_PARAM_NORMAL_REF_Y_ADDR_ROW                   0

// 
#define MPE_FRAME_PARAM_NORMAL_REF_U_ADDR_SHIFT                 (96)
#define MPE_FRAME_PARAM_NORMAL_REF_U_ADDR_FIELD                 ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_REF_U_ADDR_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_REF_U_ADDR_RANGE                 (127):(96)
#define MPE_FRAME_PARAM_NORMAL_REF_U_ADDR_ROW                   0

// 
#define MPE_FRAME_PARAM_NORMAL_REF_V_ADDR_SHIFT                 (128)
#define MPE_FRAME_PARAM_NORMAL_REF_V_ADDR_FIELD                 ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_REF_V_ADDR_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_REF_V_ADDR_RANGE                 (159):(128)
#define MPE_FRAME_PARAM_NORMAL_REF_V_ADDR_ROW                   0

// 
#define MPE_FRAME_PARAM_NORMAL_REF_STRIDE_SHIFT                 (160)
#define MPE_FRAME_PARAM_NORMAL_REF_STRIDE_FIELD                 ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_REF_STRIDE_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_REF_STRIDE_RANGE                 (191):(160)
#define MPE_FRAME_PARAM_NORMAL_REF_STRIDE_ROW                   0

// 
#define MPE_FRAME_PARAM_NORMAL_REF_BUFFER_LEN_SHIFT                     (192)
#define MPE_FRAME_PARAM_NORMAL_REF_BUFFER_LEN_FIELD                     ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_REF_BUFFER_LEN_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_REF_BUFFER_LEN_RANGE                     (223):(192)
#define MPE_FRAME_PARAM_NORMAL_REF_BUFFER_LEN_ROW                       0

// 
#define MPE_FRAME_PARAM_NORMAL_IB_OFFSET_CHROMA_SHIFT                   (224)
#define MPE_FRAME_PARAM_NORMAL_IB_OFFSET_CHROMA_FIELD                   ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_IB_OFFSET_CHROMA_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_IB_OFFSET_CHROMA_RANGE                   (255):(224)
#define MPE_FRAME_PARAM_NORMAL_IB_OFFSET_CHROMA_ROW                     0

// 
#define MPE_FRAME_PARAM_NORMAL_IB_OFFSET_LUMA_SHIFT                     (256)
#define MPE_FRAME_PARAM_NORMAL_IB_OFFSET_LUMA_FIELD                     ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_IB_OFFSET_LUMA_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_IB_OFFSET_LUMA_RANGE                     (287):(256)
#define MPE_FRAME_PARAM_NORMAL_IB_OFFSET_LUMA_ROW                       0

// 
#define MPE_FRAME_PARAM_NORMAL_FIRST_IB_OFFSET_CHROMA_SHIFT                     (288)
#define MPE_FRAME_PARAM_NORMAL_FIRST_IB_OFFSET_CHROMA_FIELD                     ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_FIRST_IB_OFFSET_CHROMA_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_FIRST_IB_OFFSET_CHROMA_RANGE                     (319):(288)
#define MPE_FRAME_PARAM_NORMAL_FIRST_IB_OFFSET_CHROMA_ROW                       0

// 
#define MPE_FRAME_PARAM_NORMAL_FIRST_IB_OFFSET_LUMA_SHIFT                       (320)
#define MPE_FRAME_PARAM_NORMAL_FIRST_IB_OFFSET_LUMA_FIELD                       ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_FIRST_IB_OFFSET_LUMA_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_FIRST_IB_OFFSET_LUMA_RANGE                       (351):(320)
#define MPE_FRAME_PARAM_NORMAL_FIRST_IB_OFFSET_LUMA_ROW                 0

// 
#define MPE_FRAME_PARAM_NORMAL_FIRST_IB_V_SIZE_SHIFT                    (352)
#define MPE_FRAME_PARAM_NORMAL_FIRST_IB_V_SIZE_FIELD                    ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_FIRST_IB_V_SIZE_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_FIRST_IB_V_SIZE_RANGE                    (383):(352)
#define MPE_FRAME_PARAM_NORMAL_FIRST_IB_V_SIZE_ROW                      0

// 
#define MPE_FRAME_PARAM_NORMAL_IB0_START_ADDR_Y_SHIFT                   (384)
#define MPE_FRAME_PARAM_NORMAL_IB0_START_ADDR_Y_FIELD                   ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_IB0_START_ADDR_Y_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_IB0_START_ADDR_Y_RANGE                   (415):(384)
#define MPE_FRAME_PARAM_NORMAL_IB0_START_ADDR_Y_ROW                     0

// 
#define MPE_FRAME_PARAM_NORMAL_IB0_START_ADDR_U_SHIFT                   (416)
#define MPE_FRAME_PARAM_NORMAL_IB0_START_ADDR_U_FIELD                   ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_IB0_START_ADDR_U_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_IB0_START_ADDR_U_RANGE                   (447):(416)
#define MPE_FRAME_PARAM_NORMAL_IB0_START_ADDR_U_ROW                     0

// 
#define MPE_FRAME_PARAM_NORMAL_IB0_START_ADDR_V_SHIFT                   (448)
#define MPE_FRAME_PARAM_NORMAL_IB0_START_ADDR_V_FIELD                   ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_IB0_START_ADDR_V_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_IB0_START_ADDR_V_RANGE                   (479):(448)
#define MPE_FRAME_PARAM_NORMAL_IB0_START_ADDR_V_ROW                     0

// 
#define MPE_FRAME_PARAM_NORMAL_IB0_SIZE_SHIFT                   (480)
#define MPE_FRAME_PARAM_NORMAL_IB0_SIZE_FIELD                   ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_IB0_SIZE_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_IB0_SIZE_RANGE                   (511):(480)
#define MPE_FRAME_PARAM_NORMAL_IB0_SIZE_ROW                     0

// 
#define MPE_FRAME_PARAM_NORMAL_IB0_LINE_STRIDE_SHIFT                    (512)
#define MPE_FRAME_PARAM_NORMAL_IB0_LINE_STRIDE_FIELD                    ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_IB0_LINE_STRIDE_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_IB0_LINE_STRIDE_RANGE                    (543):(512)
#define MPE_FRAME_PARAM_NORMAL_IB0_LINE_STRIDE_ROW                      0

// 
#define MPE_FRAME_PARAM_NORMAL_IB0_BUFFER_STRIDE_LUMA_SHIFT                     (544)
#define MPE_FRAME_PARAM_NORMAL_IB0_BUFFER_STRIDE_LUMA_FIELD                     ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_IB0_BUFFER_STRIDE_LUMA_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_IB0_BUFFER_STRIDE_LUMA_RANGE                     (575):(544)
#define MPE_FRAME_PARAM_NORMAL_IB0_BUFFER_STRIDE_LUMA_ROW                       0

// Grey scale encode
#define MPE_FRAME_PARAM_NORMAL_FRAME_CTRL_SHIFT                 (576)
#define MPE_FRAME_PARAM_NORMAL_FRAME_CTRL_FIELD                 ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_FRAME_CTRL_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_FRAME_CTRL_RANGE                 (607):(576)
#define MPE_FRAME_PARAM_NORMAL_FRAME_CTRL_ROW                   0

// Rate Control 
#define MPE_FRAME_PARAM_NORMAL_I_RATE_CTRL_SHIFT                        (608)
#define MPE_FRAME_PARAM_NORMAL_I_RATE_CTRL_FIELD                        ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_I_RATE_CTRL_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_I_RATE_CTRL_RANGE                        (639):(608)
#define MPE_FRAME_PARAM_NORMAL_I_RATE_CTRL_ROW                  0

//      
#define MPE_FRAME_PARAM_NORMAL_P_RATE_CTRL_SHIFT                        (640)
#define MPE_FRAME_PARAM_NORMAL_P_RATE_CTRL_FIELD                        ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_P_RATE_CTRL_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_P_RATE_CTRL_RANGE                        (671):(640)
#define MPE_FRAME_PARAM_NORMAL_P_RATE_CTRL_ROW                  0

//      
#define MPE_FRAME_PARAM_NORMAL_OUTPUT_BUFFER_INFO_SHIFT                 (672)
#define MPE_FRAME_PARAM_NORMAL_OUTPUT_BUFFER_INFO_FIELD                 ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_OUTPUT_BUFFER_INFO_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_OUTPUT_BUFFER_INFO_RANGE                 (703):(672)
#define MPE_FRAME_PARAM_NORMAL_OUTPUT_BUFFER_INFO_ROW                   0

//      
#define MPE_FRAME_PARAM_NORMAL_MIN_FRAME_SIZE_SHIFT                     (704)
#define MPE_FRAME_PARAM_NORMAL_MIN_FRAME_SIZE_FIELD                     ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_MIN_FRAME_SIZE_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_MIN_FRAME_SIZE_RANGE                     (735):(704)
#define MPE_FRAME_PARAM_NORMAL_MIN_FRAME_SIZE_ROW                       0

//      
#define MPE_FRAME_PARAM_NORMAL_SUGGESTED_FRAME_SIZE_SHIFT                       (736)
#define MPE_FRAME_PARAM_NORMAL_SUGGESTED_FRAME_SIZE_FIELD                       ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_SUGGESTED_FRAME_SIZE_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_SUGGESTED_FRAME_SIZE_RANGE                       (767):(736)
#define MPE_FRAME_PARAM_NORMAL_SUGGESTED_FRAME_SIZE_ROW                 0

//      
#define MPE_FRAME_PARAM_NORMAL_TARGET_BUFFER_SIZE_SHIFT                 (768)
#define MPE_FRAME_PARAM_NORMAL_TARGET_BUFFER_SIZE_FIELD                 ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_TARGET_BUFFER_SIZE_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_TARGET_BUFFER_SIZE_RANGE                 (799):(768)
#define MPE_FRAME_PARAM_NORMAL_TARGET_BUFFER_SIZE_ROW                   0

//      
#define MPE_FRAME_PARAM_NORMAL_SKIP_THRESHOLD_SHIFT                     (800)
#define MPE_FRAME_PARAM_NORMAL_SKIP_THRESHOLD_FIELD                     ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_SKIP_THRESHOLD_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_SKIP_THRESHOLD_RANGE                     (831):(800)
#define MPE_FRAME_PARAM_NORMAL_SKIP_THRESHOLD_ROW                       0

//      
#define MPE_FRAME_PARAM_NORMAL_OVERFLOW_THRESHOLD_SHIFT                 (832)
#define MPE_FRAME_PARAM_NORMAL_OVERFLOW_THRESHOLD_FIELD                 ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_OVERFLOW_THRESHOLD_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_OVERFLOW_THRESHOLD_RANGE                 (863):(832)
#define MPE_FRAME_PARAM_NORMAL_OVERFLOW_THRESHOLD_ROW                   0

// A to B Buffer        
#define MPE_FRAME_PARAM_NORMAL_MPE_PRED_BUF_AB_ADDR_SHIFT                       (864)
#define MPE_FRAME_PARAM_NORMAL_MPE_PRED_BUF_AB_ADDR_FIELD                       ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_MPE_PRED_BUF_AB_ADDR_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_MPE_PRED_BUF_AB_ADDR_RANGE                       (895):(864)
#define MPE_FRAME_PARAM_NORMAL_MPE_PRED_BUF_AB_ADDR_ROW                 0

// B to C Buffer        
#define MPE_FRAME_PARAM_NORMAL_MPE_PARAM_BUF_BC_ADDR_SHIFT                      (896)
#define MPE_FRAME_PARAM_NORMAL_MPE_PARAM_BUF_BC_ADDR_FIELD                      ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_MPE_PARAM_BUF_BC_ADDR_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_MPE_PARAM_BUF_BC_ADDR_RANGE                      (927):(896)
#define MPE_FRAME_PARAM_NORMAL_MPE_PARAM_BUF_BC_ADDR_ROW                        0

//      
#define MPE_FRAME_PARAM_NORMAL_MPE_BUF_AB_SIZE_SHIFT                    (928)
#define MPE_FRAME_PARAM_NORMAL_MPE_BUF_AB_SIZE_FIELD                    ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_MPE_BUF_AB_SIZE_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_MPE_BUF_AB_SIZE_RANGE                    (959):(928)
#define MPE_FRAME_PARAM_NORMAL_MPE_BUF_AB_SIZE_ROW                      0

//      
#define MPE_FRAME_PARAM_NORMAL_MPE_BUF_BC_SIZE_SHIFT                    (960)
#define MPE_FRAME_PARAM_NORMAL_MPE_BUF_BC_SIZE_FIELD                    ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_MPE_BUF_BC_SIZE_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_MPE_BUF_BC_SIZE_RANGE                    (991):(960)
#define MPE_FRAME_PARAM_NORMAL_MPE_BUF_BC_SIZE_ROW                      0

//      
#define MPE_FRAME_PARAM_NORMAL_SLICE_PARAMS_SHIFT                       (992)
#define MPE_FRAME_PARAM_NORMAL_SLICE_PARAMS_FIELD                       ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_SLICE_PARAMS_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_SLICE_PARAMS_RANGE                       (1023):(992)
#define MPE_FRAME_PARAM_NORMAL_SLICE_PARAMS_ROW                 0

//      
#define MPE_FRAME_PARAM_NORMAL_SLICE_MAP_OFFSET_A_SHIFT                 (1024)
#define MPE_FRAME_PARAM_NORMAL_SLICE_MAP_OFFSET_A_FIELD                 ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_SLICE_MAP_OFFSET_A_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_SLICE_MAP_OFFSET_A_RANGE                 (1055):(1024)
#define MPE_FRAME_PARAM_NORMAL_SLICE_MAP_OFFSET_A_ROW                   0

//      
#define MPE_FRAME_PARAM_NORMAL_SLICE_MAP_OFFSET_B_SHIFT                 (1056)
#define MPE_FRAME_PARAM_NORMAL_SLICE_MAP_OFFSET_B_FIELD                 ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_SLICE_MAP_OFFSET_B_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_SLICE_MAP_OFFSET_B_RANGE                 (1087):(1056)
#define MPE_FRAME_PARAM_NORMAL_SLICE_MAP_OFFSET_B_ROW                   0

// Intra Pred Ctrl, SAD/SATD    
#define MPE_FRAME_PARAM_NORMAL_MOT_SEARCH_CTRL_SHIFT                    (1088)
#define MPE_FRAME_PARAM_NORMAL_MOT_SEARCH_CTRL_FIELD                    ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_MOT_SEARCH_CTRL_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_MOT_SEARCH_CTRL_RANGE                    (1119):(1088)
#define MPE_FRAME_PARAM_NORMAL_MOT_SEARCH_CTRL_ROW                      0

//      
#define MPE_FRAME_PARAM_NORMAL_PIC_PARAMETERS_SHIFT                     (1120)
#define MPE_FRAME_PARAM_NORMAL_PIC_PARAMETERS_FIELD                     ((0xffffffff) << MPE_FRAME_PARAM_NORMAL_PIC_PARAMETERS_SHIFT)
#define MPE_FRAME_PARAM_NORMAL_PIC_PARAMETERS_RANGE                     (1151):(1120)
#define MPE_FRAME_PARAM_NORMAL_PIC_PARAMETERS_ROW                       0

// Packet definition for frame parameter section in MPEA bypass mode 

// Packet MPE_FRAME_PARAM_MPEA_BYPASS
#define MPE_FRAME_PARAM_MPEA_BYPASS_SIZE 1152

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_VOL_CTRL_SHIFT                      (0)
#define MPE_FRAME_PARAM_MPEA_BYPASS_VOL_CTRL_FIELD                      ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_VOL_CTRL_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_VOL_CTRL_RANGE                      (31):(0)
#define MPE_FRAME_PARAM_MPEA_BYPASS_VOL_CTRL_ROW                        0

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_WIDTH_HEIGHT_SHIFT                  (32)
#define MPE_FRAME_PARAM_MPEA_BYPASS_WIDTH_HEIGHT_FIELD                  ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_WIDTH_HEIGHT_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_WIDTH_HEIGHT_RANGE                  (63):(32)
#define MPE_FRAME_PARAM_MPEA_BYPASS_WIDTH_HEIGHT_ROW                    0

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_Y_ADDR_SHIFT                    (64)
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_Y_ADDR_FIELD                    ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_REF_Y_ADDR_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_Y_ADDR_RANGE                    (95):(64)
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_Y_ADDR_ROW                      0

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_U_ADDR_SHIFT                    (96)
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_U_ADDR_FIELD                    ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_REF_U_ADDR_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_U_ADDR_RANGE                    (127):(96)
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_U_ADDR_ROW                      0

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_V_ADDR_SHIFT                    (128)
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_V_ADDR_FIELD                    ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_REF_V_ADDR_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_V_ADDR_RANGE                    (159):(128)
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_V_ADDR_ROW                      0

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_STRIDE_SHIFT                    (160)
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_STRIDE_FIELD                    ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_REF_STRIDE_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_STRIDE_RANGE                    (191):(160)
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_STRIDE_ROW                      0

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_BUFFER_LEN_SHIFT                        (192)
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_BUFFER_LEN_FIELD                        ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_REF_BUFFER_LEN_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_BUFFER_LEN_RANGE                        (223):(192)
#define MPE_FRAME_PARAM_MPEA_BYPASS_REF_BUFFER_LEN_ROW                  0

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB_OFFSET_CHROMA_SHIFT                      (224)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB_OFFSET_CHROMA_FIELD                      ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_IB_OFFSET_CHROMA_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB_OFFSET_CHROMA_RANGE                      (255):(224)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB_OFFSET_CHROMA_ROW                        0

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB_OFFSET_LUMA_SHIFT                        (256)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB_OFFSET_LUMA_FIELD                        ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_IB_OFFSET_LUMA_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB_OFFSET_LUMA_RANGE                        (287):(256)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB_OFFSET_LUMA_ROW                  0

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_FIRST_IB_OFFSET_CHROMA_SHIFT                        (288)
#define MPE_FRAME_PARAM_MPEA_BYPASS_FIRST_IB_OFFSET_CHROMA_FIELD                        ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_FIRST_IB_OFFSET_CHROMA_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_FIRST_IB_OFFSET_CHROMA_RANGE                        (319):(288)
#define MPE_FRAME_PARAM_MPEA_BYPASS_FIRST_IB_OFFSET_CHROMA_ROW                  0

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_FIRST_IB_OFFSET_LUMA_SHIFT                  (320)
#define MPE_FRAME_PARAM_MPEA_BYPASS_FIRST_IB_OFFSET_LUMA_FIELD                  ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_FIRST_IB_OFFSET_LUMA_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_FIRST_IB_OFFSET_LUMA_RANGE                  (351):(320)
#define MPE_FRAME_PARAM_MPEA_BYPASS_FIRST_IB_OFFSET_LUMA_ROW                    0

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_FIRST_IB_V_SIZE_SHIFT                       (352)
#define MPE_FRAME_PARAM_MPEA_BYPASS_FIRST_IB_V_SIZE_FIELD                       ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_FIRST_IB_V_SIZE_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_FIRST_IB_V_SIZE_RANGE                       (383):(352)
#define MPE_FRAME_PARAM_MPEA_BYPASS_FIRST_IB_V_SIZE_ROW                 0

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_START_ADDR_Y_SHIFT                      (384)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_START_ADDR_Y_FIELD                      ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_IB0_START_ADDR_Y_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_START_ADDR_Y_RANGE                      (415):(384)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_START_ADDR_Y_ROW                        0

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_START_ADDR_U_SHIFT                      (416)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_START_ADDR_U_FIELD                      ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_IB0_START_ADDR_U_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_START_ADDR_U_RANGE                      (447):(416)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_START_ADDR_U_ROW                        0

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_START_ADDR_V_SHIFT                      (448)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_START_ADDR_V_FIELD                      ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_IB0_START_ADDR_V_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_START_ADDR_V_RANGE                      (479):(448)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_START_ADDR_V_ROW                        0

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_SIZE_SHIFT                      (480)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_SIZE_FIELD                      ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_IB0_SIZE_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_SIZE_RANGE                      (511):(480)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_SIZE_ROW                        0

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_LINE_STRIDE_SHIFT                       (512)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_LINE_STRIDE_FIELD                       ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_IB0_LINE_STRIDE_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_LINE_STRIDE_RANGE                       (543):(512)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_LINE_STRIDE_ROW                 0

// 
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_BUFFER_STRIDE_LUMA_SHIFT                        (544)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_BUFFER_STRIDE_LUMA_FIELD                        ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_IB0_BUFFER_STRIDE_LUMA_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_BUFFER_STRIDE_LUMA_RANGE                        (575):(544)
#define MPE_FRAME_PARAM_MPEA_BYPASS_IB0_BUFFER_STRIDE_LUMA_ROW                  0

// Grey scale encode
#define MPE_FRAME_PARAM_MPEA_BYPASS_FRAME_CTRL_SHIFT                    (576)
#define MPE_FRAME_PARAM_MPEA_BYPASS_FRAME_CTRL_FIELD                    ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_FRAME_CTRL_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_FRAME_CTRL_RANGE                    (607):(576)
#define MPE_FRAME_PARAM_MPEA_BYPASS_FRAME_CTRL_ROW                      0

// Rate Control 
#define MPE_FRAME_PARAM_MPEA_BYPASS_I_RATE_CTRL_SHIFT                   (608)
#define MPE_FRAME_PARAM_MPEA_BYPASS_I_RATE_CTRL_FIELD                   ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_I_RATE_CTRL_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_I_RATE_CTRL_RANGE                   (639):(608)
#define MPE_FRAME_PARAM_MPEA_BYPASS_I_RATE_CTRL_ROW                     0

//      
#define MPE_FRAME_PARAM_MPEA_BYPASS_P_RATE_CTRL_SHIFT                   (640)
#define MPE_FRAME_PARAM_MPEA_BYPASS_P_RATE_CTRL_FIELD                   ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_P_RATE_CTRL_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_P_RATE_CTRL_RANGE                   (671):(640)
#define MPE_FRAME_PARAM_MPEA_BYPASS_P_RATE_CTRL_ROW                     0

//      
#define MPE_FRAME_PARAM_MPEA_BYPASS_OUTPUT_BUFFER_INFO_SHIFT                    (672)
#define MPE_FRAME_PARAM_MPEA_BYPASS_OUTPUT_BUFFER_INFO_FIELD                    ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_OUTPUT_BUFFER_INFO_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_OUTPUT_BUFFER_INFO_RANGE                    (703):(672)
#define MPE_FRAME_PARAM_MPEA_BYPASS_OUTPUT_BUFFER_INFO_ROW                      0

//      
#define MPE_FRAME_PARAM_MPEA_BYPASS_MIN_FRAME_SIZE_SHIFT                        (704)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MIN_FRAME_SIZE_FIELD                        ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_MIN_FRAME_SIZE_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MIN_FRAME_SIZE_RANGE                        (735):(704)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MIN_FRAME_SIZE_ROW                  0

//      
#define MPE_FRAME_PARAM_MPEA_BYPASS_SUGGESTED_FRAME_SIZE_SHIFT                  (736)
#define MPE_FRAME_PARAM_MPEA_BYPASS_SUGGESTED_FRAME_SIZE_FIELD                  ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_SUGGESTED_FRAME_SIZE_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_SUGGESTED_FRAME_SIZE_RANGE                  (767):(736)
#define MPE_FRAME_PARAM_MPEA_BYPASS_SUGGESTED_FRAME_SIZE_ROW                    0

//      
#define MPE_FRAME_PARAM_MPEA_BYPASS_TARGET_BUFFER_SIZE_SHIFT                    (768)
#define MPE_FRAME_PARAM_MPEA_BYPASS_TARGET_BUFFER_SIZE_FIELD                    ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_TARGET_BUFFER_SIZE_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_TARGET_BUFFER_SIZE_RANGE                    (799):(768)
#define MPE_FRAME_PARAM_MPEA_BYPASS_TARGET_BUFFER_SIZE_ROW                      0

//      
#define MPE_FRAME_PARAM_MPEA_BYPASS_SKIP_THRESHOLD_SHIFT                        (800)
#define MPE_FRAME_PARAM_MPEA_BYPASS_SKIP_THRESHOLD_FIELD                        ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_SKIP_THRESHOLD_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_SKIP_THRESHOLD_RANGE                        (831):(800)
#define MPE_FRAME_PARAM_MPEA_BYPASS_SKIP_THRESHOLD_ROW                  0

//      
#define MPE_FRAME_PARAM_MPEA_BYPASS_OVERFLOW_THRESHOLD_SHIFT                    (832)
#define MPE_FRAME_PARAM_MPEA_BYPASS_OVERFLOW_THRESHOLD_FIELD                    ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_OVERFLOW_THRESHOLD_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_OVERFLOW_THRESHOLD_RANGE                    (863):(832)
#define MPE_FRAME_PARAM_MPEA_BYPASS_OVERFLOW_THRESHOLD_ROW                      0

// A to B Buffer        
#define MPE_FRAME_PARAM_MPEA_BYPASS_MPE_PRED_BUF_AB_ADDR_SHIFT                  (864)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MPE_PRED_BUF_AB_ADDR_FIELD                  ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_MPE_PRED_BUF_AB_ADDR_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MPE_PRED_BUF_AB_ADDR_RANGE                  (895):(864)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MPE_PRED_BUF_AB_ADDR_ROW                    0

// B to C Buffer        
#define MPE_FRAME_PARAM_MPEA_BYPASS_MPE_PARAM_BUF_BC_ADDR_SHIFT                 (896)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MPE_PARAM_BUF_BC_ADDR_FIELD                 ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_MPE_PARAM_BUF_BC_ADDR_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MPE_PARAM_BUF_BC_ADDR_RANGE                 (927):(896)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MPE_PARAM_BUF_BC_ADDR_ROW                   0

//      
#define MPE_FRAME_PARAM_MPEA_BYPASS_MPE_BUF_AB_SIZE_SHIFT                       (928)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MPE_BUF_AB_SIZE_FIELD                       ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_MPE_BUF_AB_SIZE_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MPE_BUF_AB_SIZE_RANGE                       (959):(928)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MPE_BUF_AB_SIZE_ROW                 0

//      
#define MPE_FRAME_PARAM_MPEA_BYPASS_MPE_BUF_BC_SIZE_SHIFT                       (960)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MPE_BUF_BC_SIZE_FIELD                       ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_MPE_BUF_BC_SIZE_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MPE_BUF_BC_SIZE_RANGE                       (991):(960)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MPE_BUF_BC_SIZE_ROW                 0

//      
#define MPE_FRAME_PARAM_MPEA_BYPASS_SLICE_PARAMS_SHIFT                  (992)
#define MPE_FRAME_PARAM_MPEA_BYPASS_SLICE_PARAMS_FIELD                  ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_SLICE_PARAMS_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_SLICE_PARAMS_RANGE                  (1023):(992)
#define MPE_FRAME_PARAM_MPEA_BYPASS_SLICE_PARAMS_ROW                    0

//      
#define MPE_FRAME_PARAM_MPEA_BYPASS_SLICE_MAP_OFFSET_A_SHIFT                    (1024)
#define MPE_FRAME_PARAM_MPEA_BYPASS_SLICE_MAP_OFFSET_A_FIELD                    ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_SLICE_MAP_OFFSET_A_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_SLICE_MAP_OFFSET_A_RANGE                    (1055):(1024)
#define MPE_FRAME_PARAM_MPEA_BYPASS_SLICE_MAP_OFFSET_A_ROW                      0

//      
#define MPE_FRAME_PARAM_MPEA_BYPASS_SLICE_MAP_OFFSET_B_SHIFT                    (1056)
#define MPE_FRAME_PARAM_MPEA_BYPASS_SLICE_MAP_OFFSET_B_FIELD                    ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_SLICE_MAP_OFFSET_B_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_SLICE_MAP_OFFSET_B_RANGE                    (1087):(1056)
#define MPE_FRAME_PARAM_MPEA_BYPASS_SLICE_MAP_OFFSET_B_ROW                      0

// Intra Pred Ctrl, SAD/SATD    
#define MPE_FRAME_PARAM_MPEA_BYPASS_MOT_SEARCH_CTRL_SHIFT                       (1088)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MOT_SEARCH_CTRL_FIELD                       ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_MOT_SEARCH_CTRL_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MOT_SEARCH_CTRL_RANGE                       (1119):(1088)
#define MPE_FRAME_PARAM_MPEA_BYPASS_MOT_SEARCH_CTRL_ROW                 0

//      
#define MPE_FRAME_PARAM_MPEA_BYPASS_PIC_PARAMETERS_SHIFT                        (1120)
#define MPE_FRAME_PARAM_MPEA_BYPASS_PIC_PARAMETERS_FIELD                        ((0xffffffff) << MPE_FRAME_PARAM_MPEA_BYPASS_PIC_PARAMETERS_SHIFT)
#define MPE_FRAME_PARAM_MPEA_BYPASS_PIC_PARAMETERS_RANGE                        (1151):(1120)
#define MPE_FRAME_PARAM_MPEA_BYPASS_PIC_PARAMETERS_ROW                  0

// --------------------------------------------------------------------------
// Special Packets
// --------------------------------------------------------------------------
// Registers to be used by both MPEA and MPEC

// Packet MPEA2MPEC_SHARED_REGS
#define MPEA2MPEC_SHARED_REGS_SIZE 32

#define MPEA2MPEC_SHARED_REGS_WIDTH_HEIGHT_SHIFT                        (0)
#define MPEA2MPEC_SHARED_REGS_WIDTH_HEIGHT_FIELD                        ((0xffffffff) << MPEA2MPEC_SHARED_REGS_WIDTH_HEIGHT_SHIFT)
#define MPEA2MPEC_SHARED_REGS_WIDTH_HEIGHT_RANGE                        (31):(0)
#define MPEA2MPEC_SHARED_REGS_WIDTH_HEIGHT_ROW                  0

// Registers to be used by MPEC exclusively

// Packet MPEA2MPEC_EXCLUSIVE_REGS
#define MPEA2MPEC_EXCLUSIVE_REGS_SIZE 32

#define MPEA2MPEC_EXCLUSIVE_REGS_ACDC_ADDR_SHIFT                        (0)
#define MPEA2MPEC_EXCLUSIVE_REGS_ACDC_ADDR_FIELD                        ((0xffffffff) << MPEA2MPEC_EXCLUSIVE_REGS_ACDC_ADDR_SHIFT)
#define MPEA2MPEC_EXCLUSIVE_REGS_ACDC_ADDR_RANGE                        (31):(0)
#define MPEA2MPEC_EXCLUSIVE_REGS_ACDC_ADDR_ROW                  0

/*
 * MPEG4/H264 Encoder register definition
 * The MPEG4/H264 Encoder can perform Simple Profile MPEG4/H.263 and Baseline Profile H.264
 * DCT-based coding of a input data frames and Motion Estimation/Compensation for P-frames
 * (Predicted frames) as well as H.264 Intra-Prediction for H.264 and CA-VLC (Content Adaptive
 * Variable Length Coding).
 * Only YUV420 input data format with 8-bit per color component is supported.
 *
 * The input frame data are assumed to be stored in input buffer(s) in the memory in planar
 * or semi-planar YUV420 format. The input buffers can be organized as two sets of ring buffer.
 * Each set can consists of a number of buffers of same size. Each input buffer must store the
 * data in YUV420 format therefore each buffer consists of a Y plane, and either separate
 * U plane, and V plane (planar) or combined UV plane (semi-planar) where the size of the
 * Y plane is 4 times the size of the U or V plane.
 * The Y, U, V, or combined UV planes can be organized contiguously or scattered in separate
 * area of the memory.
 * Each frame can be stored as multiple buffers in the memory but each frame must start on a
 * buffer boundary. Y plane must be stored as multiple of 16x16 macro blocks both horizontally
 * and vertically. The input memory is restricted such that the macro blocks are aligned on
 * 128-bit memory boundary so that blocks of data can be read efficiently.
 * Similarly, U and V planes must be stored as multiple of 8x8 blocks both horizontally and
 * and vertically. The input memory is restricted such that the blocks are aligned on 64-bit
 * memory boundary so that the blocks of data can be read efficiently.
 * When an input buffer is ready to be encoded, the MPEG4/H264 Encoder can be triggered to start
 * or resume encoding. MPEG4/H264 encoder can be triggered to start/resume encoding by
 * host.
 *
 * When a "new buffer" trigger is sent either by host, a start of frame
 * indicator should be sent also to indicate whether the new input buffer contains the
 * beginning of a a frame.
 * 
 * The MPEG4/H264 Encoder can be enabled by the host with the following sequence:
 *  a. Enable MPEG4/H264 Encoder clock
 *  b. Program parameters for MPEG4/H264 encoding in the MPEG4/H264 Encoder registers.
 *  c. Enable MPEG4/H264 Encoder module by setting ME_ENABLE register bit to START.
 * When MPEG4/H264 Encoder module is enabled it will wait for "New Buffer" commands to be
 * issued either by the host (by writing into NEW_BUFFER register) 
 * (by detecting the "New Buffer" pulse). At any time, only one source can trigger MPEG4/H264
 * Encoder.
 * After enabling MPEG4/H264 Encoder, the encoding process starts when "New Buffer" command is
 * received with "Start of Frame" flag enabled.
 * The ME_ENABLE register bit should be kept at START state when encoding multiple frames
 * and then set to STOP when encoding the last frame. When encoding a single frame, ME_ENABLE
 * register bit can be reset to STOP state immediately after setting ME_ENABLE to START.
 * After ME_ENABLE is set to STOP, the MPEG4/H264 Encoder will continue encoding until the end
 * of frame. After ME_ENABLE is set to START, if there is no "New Buffer" command with "Start
 * of Frame" received, the MPEG4/H264 Encoder will continue to be enabled even after ME_ENABLE
 * is set back to STOP. In this scenario, the MPEG4/H264 Encoder can be reset/disabled by
 * disabling the MPEG4/H264 Encoder clock.
 * A raise can be sent by the host .
 * The processing stages are:
 *  a. CKG - Clock Generator module that monitors the state of ME_ENABLE and generates
 *     the enables for second-level clocks.
 *  b. IMI - Input Memory Interface which consists of a 128-bit read/write. An arbiter to handle
 *     several clients reading/writing
 *  c. MEST - Motion Estimation module that processes on 16x16 blocks. It requests memory access
 *     for Input Video Y component, it also caches the reference frame (3x3 windows for the
 *     16x16 of the previous frame). SAD and MAD comparison makes decision of macroblock type
 *     and motion vectors.
 *  d. MCMP - Motion Compensation, take away the residue of the prediction. Both Y and UV
 *     motion compensation.
 *  e. FDCT - Forward DCT module that processes 8x8 block of input data and generates 64 DCT
 *     coefficients in 32 clock cycles.
 *  f. QRZ - Quantization, Rounding and Zig-zag that processes one DCT coefficient per clock
 *     cycle in a pre-defined zig-zag ordering.
 *  g. IPRED - Intra Prediction for INTRA macroblock only (for H264).
 *  h. ACPRD - AC/DC prediction for INTRA macroblock only (for MPEG4).
 *  i. ZRLE - Zigzag scan and Run-Length to get zeros run and level for non-zero quantized 
 *     coefficients 
 *  j. RECN - Reconstructed the frame for next frame reference
 *  k. VLC - VLC coding the run length result
 *  l. VLC DMA - Generate Link List in the Memory for VLC output
 *  m. VLC DMAOUT - Pull out encoded packets data from memory and send to EBM.
 *  n. RATECN - Rate Control, suggesting Qp for next Macroblock
 *  o. EBM - Encoded Bitstream Manager that takes encoded packets from DMAOUT and write them
 *     to memory.
 * Buffer controls and other parameters that controls the MPEG encoder operation are programmed
 * in 32-bit registers that can be written/read from host register interface.
 * 
 * When input source comes from host, data must be stored in input buffer set in memory and
 * this module supports one buffer set: input buffer set 0. In the future more input buffer
 * sets may be supported.
 * Each input buffer set consists of a programmable number of buffers (from 1 to 255 for input
 * buffer set 0 and 0 to 255 for subsequent input buffer set if there is any). Each input buffer
 * consists of a programmable number of macro block rows (from 1 to 256)) and a programmable
 * number of macro blocks per macro block row.
 * Data in the input buffer must be stored in YUV420 planar/semi-planar format therefore
 * each input buffer set consists of a set for Y, U, V planes or Y, UV planes.
 * Also in this format, input data must be arranged in multiple of macro blocks horizontally
 * and in multiple of macro block rows vertically.
 * Each macro block consists of four (2x2) Y blocks, one U block, and one V block and each
 * Y/U/V block consists of 8x8 (64) bytes.
 * Y buffer must be aligned to macro block (16-byte) boundary in the memory. U and V buffers
 * must be aligned to to block (16-byte) boundary in the memory. If memory tiling is enabled
 * all blocks must be aligned to 64-byte boundary in memory.
 * Buffers in the same input buffer set must have the same size horizontally and vertically.
 * Input buffer horizontal size is determined by the horizontal scan size (WIDTH).
 * Input buffer chroma line stride determines the distance between vertically adjacent pixels
 * in the U and V buffers and this is restricted to multiple of 16-bytes so that U and V block
 * words are aligned in 16-byte boundary in the memory.
 * Similarly, input buffer luma line stride determines the distance between verticall adjacent
 * pixels in the Y buffers and this is restricted to multiple of 16-bytes so that Y block words
 * are aligned in 16-byte boundary in the memory.
 * Input buffer vertical size determines the number of macro block rows in each buffer in the
 * buffer set.
 * Input buffer chroma buffer stride determines the distance from start address of one chroma
 * buffer to the start address of the next chroma buffer and this is restricted to multiple
 * of 16-bytes so that U and V blocks are aligned to 16-byte boundary in the memory.
 * Input buffer luma buffer stride determines the distance from start address of one luma
 * buffer to the start address of the next luma buffer and this is restricted to multiple
 * of 16-bytes so that Y blocks are aligned to 16-byte boundary in the memory.
 *
 * The VLC buffer consists of linked lists. Each linked-list corresponds to a DMA chunk.
 * This buffer will be transferred and output via an Encoded Bitstream Manager logic and
 *  written to memory.
 * 
 * ---------------------------------------------------------------------------------------------
 * Context save and restore.
 * Note that this is P1 feature in AP15 so it is likely that the implementation is broken :-(
 *
 * Context of encoder can be saved at the end of the frame encoding after REG_WR_SAFE condition
 *  is reached. Note however that at REG_WR_SAFE condition, part of VLC DMA that reads the
 *  encoded data packets from VLC DMA buffer and transfer it to EBM may still have activities.
 *  However, it is safe to change VLC DMA buffer context during REG_WR_SAFE condition. But it
 *  is not safe to change EBM active buffer contexts until ENCODER_IDLE condition oclwrs.
 *  Typically it should be possible to share EBM buffers with the new/next context so in this
 *  case it is not necessary to flush EBM before saving encoder context and programming new
 *  context. 
 *  If it is necessary to reset EBM active context or to change EBM buffer size for the next
 *  context then software must also wait for ENCODER_IDLE condition, which indicates that all
 *  encoded data to be written by EBM.
 * 
 * For context saving, the following internal encoder state must also be saved:
 * 1. Intra refresh RAM must be saved if intra refresh is enabled.
 * 2. REF_BUFFER_IDX register.
 * 3. REF_RD_MBROW  register.
 * 4. REF_WR_MBROW register.
 * 5. FRAME_INDEX register.
 * 6. ENC_FRAME_NUM parameter in ENC_FRAME_NUM register.
 * 7. FRAME_NUM_GOP and FRAME_NUM parameter in FRAME_NUM register.
 * 8. IDR_PIC_ID register.
 * 9. Rate Control RAM.
 * 10. LOWER_BOUND register.
 * 11. UPPER_BOUND register.
 * 12. REMAINING_BITS register.
 * 13. NUM_CODED_BU register.
 * 14. PREVIOUS_QP register.
 * 15. NUM_P_PICTURE register.
 * 16. QP_SUM register.
 * 17. TOTAL_ENERGY register.
 * 18. A1_VALUE register.
 * 19. LENGTH_OF_STREAM register.
 * 20. BUFFER_FULL_READ register.
 * 21. CODED_FRAMES register.
 * 22. P_AVE_HEADER_BITS_A register.
 * 23. P_AVE_HEADER_BITS_B register.
 * 24. PREV_FRAME_MAD register.
 * 25. TOTAL_QP_FOR_P_PICTURE register.
 * 26. CONTEXT_SAVE_MISC register.
 * 27. LENGTH_OF_MOTION_MODE register.
 * 28. TARGET_BUFFER_LEVEL register.
 * 29. DELTA_P register.
 * 30. LENGTH_OF_STREAM_CBR register.
 * ---------------------------------------------------------------------------------------------
 * base MPE 0x000000;
 * The following include is to add registers to control IB and REF surfaces address tiling.
 * The registers are not inserted here but instead inserted at the point where 
 *  ADD_TILE_MODE_REG_SPEC is specified.
 * --------------------------------------------------------------------------

 *
 * Memory Controller Tiling definitions
 *
 *
 *  To enable tiling for a buffer in your module you'll want to include
 *  this spec file and then make use of either the ADD_TILE_MODE_REG_SPEC
 *  or ADD_TILE_MODE_REG_FIELD_SPEC macro.
 *
 *  For the ADD_TILE_MODE_REG_SPEC macro, the regp arg is added to the
 *  register name as a prefix to match the names of the other registers
 *  for this buffer. The fldp is the field name prefix to make the name
 *  unique so it works with arreggen generated reg blocks (e.g.):
 *
 *       * specify how addressing should occur for IB0 buffer
 *      ADD_TILE_MODE_REG_SPEC(IB0, IB0);
 *
 *  There's also a REG_RW_SPEC version, if you need to specify a special
 *  flag (e.g. rws for shadow, or rwt for trigger).
 *  
 *  For the ADD_TILE_MODE_REG_FIELD_SPEC macro, the fldp is the field
 *  name prefix and bitpos arg describes the starting bit position for
 *  this field within another register.
 *
 *  Like the register version, there's a REG_RW_FIELD_SPEC version if
 *  you need to set explicit bits other than "rw".
 *
 *  Note: this requires having at least LW_MC_TILE_MODEWIDTH bits of
 *  space available after bitpos (e.g.) in the register:
 *
 *      ADD_TILE_MODE_REG_FIELD_SPEC(REF, 16)    * This parameter specifies how addressing
 *                                               * for the REF buffer should occur
 *
 * ---------------------------------------------------------------------------------------------
 * Sync Point condition definition.
 * Note that host interface supports sync point counters that may be incremented when certain
 *  sync point conditions are met. Software would send command to MPE to increment the sync
 *  point counter at the proper sync point condition(s) and with corresponding sync point
 *  counter index for each condition. When the condition event oclwrs, the sync point counter
 *  index is returned to host interface logic.
 *  The host interface logic would then increments the appropriate sync point counter based
 *  on the sync point counter index.  Software can use these sync point counters value to
 *  to issue a "WAIT" command or to program such that interrupt is generated when sync point
 *  counter is incremented.
 * Software enables a sync point condition by writing into INCR_SYNCPT register and providing
 *  the condition and the corresponding sync point counter index to be incremented when the
 *  condition event oclwrs.
 * MPE can deal with multiple sync point conditions in parallel. However for each unique sync
 *  point condition to be monitored, only one pending sync point can be programmed.
 *  If software issues more than one INCR_SYNCPT commands for the same sync point condition
 *  without waiting for the sync point counters to be incremented then less number of sync
 *  point index will be returned. This condition will also be recorded in INCR_SYNCPT_ERROR
 *  register. One exception is OP_DONE sync point. The encoder hardware can queue up to 128
 *  requests to increment at OP_DONE.
 *
 * Primary mode of operation for MPE will be with INPUT_SRC_SEL set to HOST.
 * In this case, input buffer should be set to 1 frame per buffer to withstand host latencies.
 * Three sync points should be used by host to control encoding operation:
 * a. RD_DONE (End of Input Buffer) can be used to indicate to host that input buffer has been
 *    consumed by encoder so that the module that generate the encoding source clip can refill
 *    this buffer with the next input frame. This sync point condition is generated even if
 *    encoder decides to drop/skip the corresponding input buffer for any reason. Frame
 *    dropping may occur due to decision by rate control or frame pattern logic, or due to
 *    input buffer arriving early and IB_FRAME_SKIP_MODE is set to SKIP.
 * b. OP_DONE (Operation Done) can be used to keep tracks of exelwtion of NEW_BUFFER commands.
 *    This may be checked prior to each frame encoding to make sure that increment OP_DONE
 *    requests do not overlow the internal hardware queue. More importantly this may be used
 *    to keep track of end of encoding to know when EBM can be reset.
 * c. REG_WR_SAFE (Safe to Write Registers) can be used by host to indicate when it is safe
 *    to reprogram encoder context (especially non-shadow registers) except for EBM active
 *    buffer context. This sync point condition is generated after previous frame encoding is
 *    completed and previous encoded frame data has been written to VLC DMA buffer and its
 *    reconstructed frame has been written to reference buffer and before next frame encoding
 *    starts. Internally this is a level signal which may not be generated if the next frame
 *    NEW_BUFFER command is issued while the current frame encoding is in progress.
 *    If there is no change in encoder context for the next frame to be encoded, software may
 *    choose not to check for REG_WR_SAFE sync point before issuing NEW_BUFFER command for the
 *    next frame. This will allow encoder hardware to do frame skipping for early input frame
 *    (IB_FRAME_SKIP_MODE set to SKIP).
 *    This sync point MUST not be checked if the next NEW_BUFFER command does not start a new
 *    frame.
 *
 * The encoded output of encoder must typically be processed by host to be either transmitted
 *  or stored. Host must read the chunk headers in memory in order to process the encoded
 *  bitstream, therefore an interrupt should be used.  EBM End of Frame interrupt should be
 *  used typically. However there maybe cases where a large encoded frame is generated that
 *  spans over multiple EBM buffers. In this case, EBM End of Buffer interrupt can also be
 *  used in addition to EBM End of Frame to distribute load on host and prevent VLC DMA buffer
 *  overflow due to unavailable EBM buffer. Continuous sync points for EBM End of Frame and
 *  EBM End of Buffer are provided since the use of sync point to generate interrupts is
 *  recommended for a more uniform interrupt handling in the Resource Manager (RM) software.
 *  Note however that using sync point to generate interrupts will tie up sync point counter.
 *  Note that EBM has no knowledge of skipped buffer/frame so if a buffer is skipped/frame,
 *  there will be no corresponding EBM End of Frame interrupt generated. Host must not expect
 *  a 1-to-1 correspondence between NEW_BUFFER commands issued and EBM End of Interrupts.
 * 
 * At the end of stream encoding if encoder is to be shut down, due to the fact that the last
 *  frame(s) may be skipped, it maybe necessary to know whether encoding pipeline and VLC DMA
 *  buffers have been completely flushed and written to EBM buffers. When encoding from host,
 *  OP_DONE sync point counters may be used to get this status. Typically the command to 
 *  increment at ENCODER_IDLE sync point should be issued
 *  after setting ME_ENABLE to STOP.
 *
 * During encoding, host need to periodically send feedback to the encoder rate control on
 *  the status of virtual output buffer fullness. The virtual output buffer may be the EBM
 *  buffer or another buffer which is used for bitstream transmission or storage.
 *  This is done by writing into BUFFER_FULL and REPORTED_FRAME registers.
 *  There may also be other rate control specific direction that host may want to do
 *  (such as forcing I_frame) in CYA mode if internal rate control needs extra help from host.
 *  One way is to synchronize this feedback mechanism with issuance of the next NEW_BUFFER
 *  command. However if host is "looking ahead", i.e. encoder commands are prepared by host
 *  not for the next frame but for the one after, this may introduce additional frame delay
 *  on encoder receiving this feedback and may affect rate control performance even though
 *  the rate control is designed to withstand multiple frame latencies.
 *  A better way to send this information is to use the shadow register mechanism. Note that
 *  not all encoder registers are shadowed.
 *  
 * Typical programming sequence during normal encoding from host may look like:
 * a. Enable MPE clock.
 * b. Set ME_ENABLE=START and program static registers such as memory client interface
 *    parameters and set up interrupts for EBM end of frame/buffer.
 * c. If starting a new frame, send increment at REG_WR_SAFE and wait for the sync point
 *    counter to be incremented, and then program new context for the next frame to be encoded.
 *    This step may be skipped if there is no encoder context change for the next frame.
 *    This step MUST not performed if not starting a new frame.
 * d. Check sync point counter for OP_DONE and make sure there are not more than 128 pending
 *    requests for increment at OP_DONE.
 * e. Send new buffer (NEW_BUFFER) command.
 * f. Send increment at RD_DONE (end of input buffer).
 * g. Send increment at OP_DONE sync point.
 * h. Wait for RD_DONE sync point counter increment (end of input buffer) to let encoding
 *    source to re-use this buffer (this maybe in a separate s/w thread that controls the
 *    encoding source?). For real-time camera encoding, this sync point should also be used for
 *    host to decide if an input frame encoding should be skipped if encoder cannot encode the
 *    current frame fast enough and to prevent overflow in source module (VI).
 * i. Repeat step c to h if there is more input buffer to encode
 * j. When there is no more input buffer to encode and encoder needs to be shut down or for
 *    full context switch to a different sequence that requires a new set of EBM buffer,
 *    wait for OP_DONE sync point counter to be incremented for the last request to increment
 *    at OP_DONE.
 * k. Set ME_ENABLE to STOP.
 * l. Save encoder context if encoding is to be resumed at a later point. Note that
 *    implementation of context save/restore may be incomplete in AP15 (this is P1 feature).
 * m. Disable MPE clock if MPE is to be shut down or re-program MPE with a new context if
 *    a new/different sequence encoding is to be started/resumed.
 *
 * The above sequence does not "look ahead" 1 frame but can be modified to look ahead for
 *  better pipelining.
 *
 * ---------------------------------------------------------------------------------------------
 * Number of sync point conditions.
 * Sync point IMMEDIATE : No condition
 * Immediate return of the sync point index.
 * Sync point OP_DONE : Operation Done.
 * This sync point condition is generated when all previously issued NEW_BUFFER commands
 *  exelwtion have been completed.
 *  This indicates that the input buffers corresponding to the NEW_BUFFER commands have been
 *  either skipped or encoded and encoded bitstream have been transferred from VLC DMA buffers
 *  to EBM buffers.
 *  In case that a frame is composed of multiple buffers and the number of buffers per frame
 *  then requests for increment at OP_DONE sync point is queued internally for all buffers and
 *  the corresponding sync point indices returned at the end of frames.
 *  Also in case that a frame is skipped for any reason while previous frame is still being
 *  encoded or before the previous frame output bitstream is completely written to EBM memory,
 *  the requests for increment at OP_DONE is queued internally until previous frame encoding is
 *  completed before the sync point indices are returned.
 *  Note that input frames/buffers may be dropped for various reasons. Input frames may be
 *  dropped by the encoder rate control or encoder frame pattern programming. Input frames may
 *  also be dropped because input buffer comes too early and IB_FRAME_SKIP_MODE is set to SKIP.
 *  Input buffers may also be dropped if they're beyond the current encoded frame size.
 *  The encoder hardware is capable of queuing up to 128 requests for increment at OP_DONE.
 *  The host software MUST not issue anymore NEW_BUFFER command if there are 128 requests
 *  for OP_DONE pending. If there are 128 requests for OP_DONE pending then REG_WR_SAFE
 *  condition will be deactivated.
 *  So one way to ensure that NEW_BUFFER is not issued while there are 128 OP_DONE pending is
 * to check for REG_WR_SAFE condition before starting a new frame encoding. However in this
 *  way, early frame skip cannot be done by encoder hardware and must be done by software.
 *  A better way is to check for OP_DONE sync point counter state.
 * When encoding input source is host, this sync point should be issued after every
 *  NEW_BUFFER command and the corresponding sync point counter value may be used by host to
 *  know when all NEW_BUFFER commands have been completed at the end of encoding so that
 *  host knows when to reset EBM to release all unused active EBM buffers.
 * Sync point RD_DONE : Read Done (End of Input Buffer).
 * This sync point condition indicates that current input buffer has been read by encoder
 *  hardware. This allow host or input source to re-use/overwrite this current input buffer.
 *  Note that this does not mean that the current input buffer has been fully encoded; it just
 *  means that encoder hardware no longer needs this input buffer. The condition is also met
 *  and the sync point index returned to host if the input buffer is dropped by the encoder
 *  rate control or encoder frame pattern. The sync point index is returned to host also when
 *  an input buffer is skipped because it comes too early and IB_FRAME_SKIP_MODE is set to SKIP.
 * When encoding input source is host, this sync point may be used by host to re-use the input
 *  buffer and start generation of next input buffer from encoding source or reclaim this buffer
 *  for other use. This sync point condition may also be used for host to decide on frame
 *  skipping.
 * Sync point REG_WR_SAFE : Safe to Write Registers.
 * This sync point condition is generated when encoding of current frame encoding is done
 *  and the next frame encoding has not started.  Note that this indicates that the
 *  previous encoded frame data has been written to VLC DMA buffer and its reconstructed
 *  frame data written to reference buffer. But this does not indicate that content of
 *  VLC DMA buffer has been transferred to EBM buffer for the previous encoded frame.
 *  If after the the current frame encoding, the next NEW_BUFFER command is already pending,
 *  then this condition will NOT be met in between the two buffer/frame encoding.
 *  If input frame is composed of more multiple input buffers then this condition will NOT
 *  be generated in between input buffers in the middle of the frame encoding. This condition
 *  can only be generated in between frames so this MUST NOT be checked to issue input buffers
 *  in the middle of a frame in case the frame consists of multiple buffers.
 * When encoding input source is host, this sync point may be used by host to re-program encoder
 *  context (both shadowed and non-shadowed registers) and issue the next NEW_BUFFER command
 *  to encode the next frame. However active EBM buffers programming MUST be preserved until
 *  data transfer from VLC DMA buffer to EBM is completed.
 *  If encoder non-shadowed registers do not need to be reprogrammed (encoder context remain the
 *  same) for the next frame to be encoded, it is possible for host to issue the next NEW_BUFFER
 *  command for the next frame without checking for this condition. This allow the use of
 *  hardware early frame skip feature (IB_FRAME_SKIP_MODE is set to SKIP) to drop an input frame
 *  if the input frame comes before the current frame encoding is completed. However software
 *  must check that there are not too many pending increment at OP_DONE requests.
 * Sync point ENCODER_IDLE : Encoder Idle.
 * This sync point condition indicates that no encoding is in progress AND encoded data transfer
 *  from VLC DMA buffer to EBM buffer is complete (no pending VLC DMA read and no pending EBM
 *  buffer write). This is a point where it is safe to reprogram any encoder context including
 *  EBM context.
 * If encoding input source is host, this sync point is not needed and OP_DONE should be used
 *  instead.
 * Sync point XFER_DONE: Shadow Register Transfer completed.
 * This should be used just before XFER in SHADOW_REG_EN is set to ENABLE to provide response to
 *  host when the shadow registers have been transferred to working registers.
 * Note that if this is issued after XFER in SHADOW_REG_EN is set to ENABLE, there is a slight
 *  possibility that condition might be missed.
 * This sync point is needed only if host is concern about updating shadow register more than
 *  once in the same frame. The register shadowing will only retain the last programming if
 *  shadow register is updated more than once in the same frame. This shadow register behavior
 *  is acceptable for updating virtual output buffer fullness for rate control operation but
 *  could be a problem for other shadow register programming that must take effect for at least
 *  one frame.
 * Sync point IB_SOF: Input Start of Frame.
 * This sync point condition indicates that encoder starts encoding a new input frame.
 *
 * VOL and VOP Short Header Insertion
 *
 * In MPEG4 and H263 mode, MPE does not generate the complete decodable bitstream.  
 *
 * In MPEG4, the encode encodes VOP level and below.  Syntax above VOP, such as VOL are inserted
 *  by the software at the beginning of the bitstream.
 * In H263 mode, MPE provides all the encoded bits except the first 5 bytes (40bits) of 
 *  video_palen_with_short_header.  That means software needs to insert the followings at the
 *  beginning of each frame:
 *      short_video_start_marker
 *      temporal_reference
 *      marker_bit
 *      zero_bit
 *      split_screen_indicator
 *      dolwment_camera_indicator
 *      full_picture_freeze_release
 *      source_format
 *      picture_coding_type
 *      four_reserved_zero_bits (only 1 bit is needed to insert by sw)
 *  The rest of the video_plane_with_short_header are provided in the encoded bitsteam.
 *      four_reserved_zero_bits  (remaining 3 bits)
 *      vop_quant 
 *      zero_bit
 *      .....
 *
 *
 *
 * MinQp/MaxQp/InitQp/MaxQpRelieve for VBR (MPEG4 Rate Control)
 * 
 * MinQp/MaxQp/InitQp/MaxQpRelieve can be looked up based on BITS_PER_MB:
 *
 * If I-Frame is not inserted
 *           BITS_PER_MB = bitrate/(framerate * MB width * MB height);
 * else if I-Frame is inserted periodically
 *           I_Frame_size_factor = 3 (I-Frame size factor compared to P-Frame)
 *           IFPS = framerate/period_of_I_frame    ( No. of Periodic I_Frame per sec. ) 
 *           effective_fps = framerate + (I_Frame_size_factor-1)*IFPS;
 * 
 *           BITS_PER_MB = bitrate/(effective_fps * MB width * MB height);
 *
 *  Depending on the frame resolution, look up and linear interpolate the MinQp/MaxQp from 
 *    one of the following three sets of (QCIF, CIF/QVGA, VGA) tables. 
 *   
 *    *** QCIF ***
 *    MinQp,  BITS_PER_MB
 *        1, 351
 *        2, 103
 *        3,  70
 *        4,  48
 *        5,  39
 *        6,  31
 *        7,  26
 *        8,  22
 *        9,  20
 *       10,  17
 *       12,  14
 *       14,  12
 *       16,  10
 *       18,   9
 *       20,   8
 *       24,   7
 *       29,   6
 *
 * 
 *    *** QCIF ***
 *    MaxQp,  BITS_PER_MB
 *        4, 338
 *        6, 218
 *        8, 159
 *       10, 124
 *       12, 102
 *       14, 86
 *       16, 76
 *       18, 67
 *       20, 61
 *       22, 56
 *       24, 52
 *       26, 48
 *       28, 45
 *       29, 44
 *       30, 43
 *
 *
 *    ** CIF/QVGA ***
 *    MinQp,  BITS_PER_MB
 *        1, 327
 *        2, 66
 *        3, 44
 *        4, 29
 *        5, 24
 *        6, 19
 *        7, 16
 *        8, 13
 *        9, 12
 *       10, 11
 *       12, 9
 *       14, 8
 *       16, 7
 *       19, 6
 *       24, 5
 *       31, 4
 *
 *
 *    *** CIF/QVGA ***
 *    MaxQp,  BITS_PER_MB
 *        4, 239
 *        6, 153
 *        8, 112
 *       10, 88
 *       12, 73
 *       14, 63
 *       16, 56
 *       18, 50
 *       20, 46
 *       22, 43
 *       24, 40
 *       26, 38
 *       28, 36
 *       30, 34
 *
 *
 *    *** VGA ***
 *    MinQp,  BITS_PER_MB
 *        1, 114
 *        2,  57
 *        3,  28
 *        4,  22
 *
 *
 *    *** VGA ***
 *    MaxQp,  BITS_PER_MB
 *       12, 114
 *       14,  57
 *       16,  28
 *       20,  22
 *
 *    *** 720P P-Frame ***
 *    MinQp,  BITS_PER_MB
 *        1, 9999
 *        2,  91
 *        3,  31
 *        4,  17
 *        5,  11
 *        6,   7
 *        7,   5
 *        8,   4
 *        9,   3
 *       12,   2
 *       24,   1
 *
 *    *** 720P P-Frame ***
 *    MaxQp,  BITS_PER_MB
 *       17,   5
 *       20,   4
 *       24,   3
 *       31,   2
 *
 *    *** 720P I-Frame ***
 *    MinQp   BITS_PER_MB
 *        3,  62
 *        4,  47
 *        5,  37
 *        6,  30
 *        7,  25
 *        8,  22
 *        9,  19
 *       10,  17
 *       11,  16
 *       13,  13
 *       14,  12
 *       17,  10
 *       19,   9
 *       22,   8
 *       25,   7
 *       31,   5
 *
 *    *** 720P I-Frame ***
 *    MaxQp   BITS_PER_MB
 *       17,  25
 *       19,  22
 *       25,  16
 *       28,  14
 *       31,  12
 *
 *    **** 1080P P-Frame ***
 *    MinQp,  BITS_PER_MB
 *        1, 9999
 *        2, 167
 *        3,  88
 *        4,  51
 *        5,  41
 *        6,  32 
 *        7,  28
 *        8,  24
 *        9,  22
 *       10,  20
 *       1l,  18
 *       12,  17
 *       13,  16
 *       15,  15
 *       17,  14
 *       19,  13
 *       23,  12
 *       31,  11
 * 
 *    **** 1080P P-Frame ***
 *    MaxQp,  BITS_PER_MB
 *       15,  57
 *       16,  53
 *       17,  51
 *       18,  48
 *       19,  46
 *       20,  44
 *       21,  43
 *       22,  41
 *       23,  40
 *       24,  38
 *       25,  37
 *       26,  36
 *       27,  35
 *       28,  34
 *       31,  32
 *
 *    **** 1080P I-Frame ***
 *    MinQp,  BITS_PER_MB
 *        1,  9999 
 *        2,  281
 *        3,  158
 *        4,  113
 *        5,   90
 *        6,   76
 *        7,   67
 *        8,   60
 *        9,   55
 *       10,   51
 *       11,   47
 *       12,   45
 *       13,   42
 *       14,   40
 *       15,   39
 *       16,   37
 *       17,   36
 *       18,   35
 *       19,   34
 *       20,   33
 *       22,   32
 *       23,   31
 *       25,   30
 *       30,   28
 *
 *    **** 1080P I-Frame ***
 *    MaxQp,  BITS_PER_MB
 *       15,   58
 *       16,   55
 *       17,   53
 *       18,   51
 *       19,   48
 *       20,   47
 *       21,   45
 *       22,   43
 *       23,   42
 *       24,   41
 *       25,   40
 *       26,   38
 *       27,   37
 *       28,   36
 *       29,   35
 *       31,   34
 *
 *  Recommended InitQp/MaxQpRelieve value for VBR:
 *     if (Num of MBs <= 396) { # Up to CIF
 *         InitQP = 12;
 *         MaxQpRelieve = 12;
 *     } else if (Num of MBs <= 1650)  {
 *         if (BITS_PER_MB >= 100)  {
 *             InitQP = 10;
 *             MaxQpRelieve = 6;
 *         } else {
 *             InitQp = 12;
 *             MaxQpRelieve = 12;
 *         }
 *     }  else {     * 720P or 1080P
 *         if (BITS_PER_MB >= 100)  {
 *             MaxQpRelieve = 6;
 *         } else {
 *             MaxQpRelieve = 12;
 *         }
 *         InitQpI = (MaxQpI + MinQpI)/2;
 *         InitQpP = (MaxQpP + MinQpP)/2;
 *     }     
 *
 *
 *
 *  MinQp/InitQp recommendation for H264 Rate Control
 *
 *  double  L1,L2,L3,bpp;
 *  double FrameRate;
 *  int min_i_qp,max_i_qp,min_p_qp,max_p_qp;
 *  FrameRate = (double)frameRate / (double)(skip + 1);
 *  bpp = 1.0 * bitRate / (FrameRate * width*height);
 *  if(width == 176) {
 *            L1 = 0.1;
 *            L2 = 0.3;
 *            L3 = 0.6;
 *  }
 *  else if(width == 352) {
 *            L1 = 0.2;
 *            L2 = 0.6;
 *            L3 = 1.2;
 *  }
 *  else {
 *            L1 = 0.1*sqrt((1.0 * width*height)/(256 * 99.0));
 *            L2 = 0.3*sqrt((1.0 * width*height)/(256 * 99.0));
 *            L3 = 0.6*sqrt((1.0 * width*height)/(256 * 99.0));
 *  }
 *  if(bpp <= L1) {
 *                  
 *                    init_qp = 35;        
 *                    min_i_qp = 25;
 *  }
 *  else if(bpp <=L2)
 *  {
 *                    init_qp = 25;        
 *                    min_i_qp = 20;
 *  }
 *  else if(bpp <=L3) {
 *                    init_qp = 20;        
 *                    min_i_qp = 15;
 *  }
 *  else {
 *                    init_qp = 10;
 *                    min_i_qp = 5;
 *  }
 *  
 *  max_i_qp = 51;
 *  min_p_qp = 0;
 *  max_p_qp = 51;
 * The following include file defines the host registers including context switch and sync
 *  point registers.
 * Total of 8 registers are reserved although only 3 are lwrrently defined. 
 * 
 */

#define LWE2B7_INCR_SYNCPT_NB_CONDS                       7

#define LWE2B7_INCR_SYNCPT_0                              (0x0)
// Condition mapped from raise/wait
#define LWE2B7_INCR_SYNCPT_0_COND                         15:8
#define LWE2B7_INCR_SYNCPT_0_COND_IMMEDIATE               (0)
#define LWE2B7_INCR_SYNCPT_0_COND_OP_DONE                 (1)
#define LWE2B7_INCR_SYNCPT_0_COND_RD_DONE                 (2)
#define LWE2B7_INCR_SYNCPT_0_COND_REG_WR_SAFE             (3)
#define LWE2B7_INCR_SYNCPT_0_COND_ENCODER_IDLE            (4)
#define LWE2B7_INCR_SYNCPT_0_COND_XFER_DONE               (5)
#define LWE2B7_INCR_SYNCPT_0_COND_IB_SOF                  (6)
#define LWE2B7_INCR_SYNCPT_0_COND_COND_7                  (7)
#define LWE2B7_INCR_SYNCPT_0_COND_COND_8                  (8)
#define LWE2B7_INCR_SYNCPT_0_COND_COND_9                  (9)
#define LWE2B7_INCR_SYNCPT_0_COND_COND_10                 (10)
#define LWE2B7_INCR_SYNCPT_0_COND_COND_11                 (11)
#define LWE2B7_INCR_SYNCPT_0_COND_COND_12                 (12)
#define LWE2B7_INCR_SYNCPT_0_COND_COND_13                 (13)
#define LWE2B7_INCR_SYNCPT_0_COND_COND_14                 (14)
#define LWE2B7_INCR_SYNCPT_0_COND_COND_15                 (15)
#define LWE2B7_INCR_SYNCPT_0_COND_DEFAULT                          (0x00000000)
// syncpt index value
#define LWE2B7_INCR_SYNCPT_0_INDX                         7:0
#define LWE2B7_INCR_SYNCPT_0_INDX_DEFAULT                          (0x00000000)

#define LWE2B7_INCR_SYNCPT_CNTRL_0                        (0x1)
// If NO_STALL is 1, then when fifos are full,
// INCR_SYNCPT methods will be dropped and the
// INCR_SYNCPT_ERROR[COND] bit will be set.
// If NO_STALL is 0, then when fifos are full,
// the client host interface will be stalled.
#define LWE2B7_INCR_SYNCPT_CNTRL_0_INCR_SYNCPT_NO_STALL   8:8
#define LWE2B7_INCR_SYNCPT_CNTRL_0_INCR_SYNCPT_NO_STALL_DEFAULT    (0x00000000)

// If SOFT_RESET is set, then all internal state
// of the client syncpt block will be reset.
// To do soft reset, first set SOFT_RESET of
// all host1x clients affected, then clear all
// SOFT_RESETs.
#define LWE2B7_INCR_SYNCPT_CNTRL_0_INCR_SYNCPT_SOFT_RESET 0:0
#define LWE2B7_INCR_SYNCPT_CNTRL_0_INCR_SYNCPT_SOFT_RESET_DEFAULT  (0x00000000)

#define LWE2B7_INCR_SYNCPT_ERROR_0                        (0x2)
// COND_STATUS[COND] is set if the fifo for COND overflows.
// This bit is sticky and will remain set until cleared.
// Cleared by writing 1.
#define LWE2B7_INCR_SYNCPT_ERROR_0_COND_STATUS            31:0
#define LWE2B7_INCR_SYNCPT_ERROR_0_COND_STATUS_DEFAULT             (0x00000000)

// just in case names were redefined using macros
// Context switch reg is defined in this include file which takes one 4-byte register space.
// Note that at some point this maybe absorbed inside "hcif_syncpt.spec".
// Context switch register.  Should be common to all modules.  Includes the
// current channel/class (which is writable by SW) and the next channel/class
// (which the hardware sets when it receives a context switch).
// Context switch works like this:
// Any context switch request triggers an interrupt to the host and causes the
// new channel/class to be stored in NEXT_CHANNEL/NEXT_CLASS (see
// vmod/chexample).  SW sees that there is a context switch interrupt and does
// the necessary operations to make the module ready to receive traffic from
// the new context.  It clears the context switch interrupt and writes
// LWRR_CHANNEL/CLASS to the same value as NEXT_CHANNEL/CLASS, which causes a
// context switch acknowledge packet to be sent to the host.  This completes
// the context switch and allows the host to continue sending data to the
// module.
// Context switches can also be pre-loaded.  If LWRR_CLASS/CHANNEL are written
// and updated to the next CLASS/CHANNEL before the context switch request
// oclwrs, an acknowledge will be generated by the module and no interrupt will
// be triggered.  This is one way for software to avoid dealing with context
// switch interrupts.
// Another way to avoid context switch interrupts is to set the AUTO_ACK bit.
// This bit tells the module to automatically acknowledge any incoming context
// switch requests without triggering an interrupt.  LWRR_* and NEXT_* will be
// updated by the module so they will always be current.

#define LWE2B7_CTXSW_0                                    (0x8)
// Current working class
#define LWE2B7_CTXSW_0_LWRR_CLASS                         9:0
#define LWE2B7_CTXSW_0_LWRR_CLASS_DEFAULT                          (0x00000000)

// Automatically acknowledge any incoming context switch requests
#define LWE2B7_CTXSW_0_AUTO_ACK                           11:11
#define LWE2B7_CTXSW_0_AUTO_ACK_MANUAL                    (0)
#define LWE2B7_CTXSW_0_AUTO_ACK_AUTOACK                   (1)
#define LWE2B7_CTXSW_0_AUTO_ACK_DEFAULT                            (0x00000000)

// Current working channel, reset to 'invalid'
#define LWE2B7_CTXSW_0_LWRR_CHANNEL                       15:12
#define LWE2B7_CTXSW_0_LWRR_CHANNEL_DEFAULT                        (0x00000000)

// Next requested class
#define LWE2B7_CTXSW_0_NEXT_CLASS                         25:16
#define LWE2B7_CTXSW_0_NEXT_CLASS_DEFAULT                          (0x00000000)

// Next requested channel
#define LWE2B7_CTXSW_0_NEXT_CHANNEL                       31:28
#define LWE2B7_CTXSW_0_NEXT_CHANNEL_DEFAULT                        (0x00000000)

// Continuous sync points registers.

#define LWE2B7_CONT_SYNCPT_EBM_EOF_0                      (0xa)
// Sync Point Counter Index.
// This parameter specifies the index of the
//  sync point counter that will be returned to
//  host when continuous sync point is enabled
//  (COND = ENABLE) and whenever an EBM
//  frame is written.
#define LWE2B7_CONT_SYNCPT_EBM_EOF_0_EBM_EOF_INDX         7:0
#define LWE2B7_CONT_SYNCPT_EBM_EOF_0_EBM_EOF_INDX_DEFAULT          (0x00000000)

// Sync Point Condition Control.
// This bit can be used to enable/disable
//  generation of continuous sync point
//  increment.
#define LWE2B7_CONT_SYNCPT_EBM_EOF_0_EBM_EOF_COND             8:8
#define LWE2B7_CONT_SYNCPT_EBM_EOF_0_EBM_EOF_COND_DISABLE     (0)    // // EBM End of Frame continuous sync point
//  is disabled.

#define LWE2B7_CONT_SYNCPT_EBM_EOF_0_EBM_EOF_COND_ENABLE      (1)    // // EBM End of Frame continuous sync point
//  is enabled.
#define LWE2B7_CONT_SYNCPT_EBM_EOF_0_EBM_EOF_COND_DEFAULT          (0x00000000)


#define LWE2B7_CONT_SYNCPT_EBM_EOB_0                          (0xb)
// Sync Point Counter Index.
// This parameter specifies the index of the
//  sync point counter that will be returned to
//  host when continuous sync point is enabled
//  (COND = ENABLE) and whenever an EBM
//  buffer is filled.
#define LWE2B7_CONT_SYNCPT_EBM_EOB_0_EBM_EOB_INDX             7:0
#define LWE2B7_CONT_SYNCPT_EBM_EOB_0_EBM_EOB_INDX_DEFAULT          (0x00000000)

// Sync Point Condition Control.
// This bit can be used to enable/disable
//  generation of continuous sync point
//  increment.
#define LWE2B7_CONT_SYNCPT_EBM_EOB_0_EBM_EOB_COND             8:8
#define LWE2B7_CONT_SYNCPT_EBM_EOB_0_EBM_EOB_COND_DISABLE     (0)    // // EBM End of Buffer continuous sync point
//  is disabled.

#define LWE2B7_CONT_SYNCPT_EBM_EOB_0_EBM_EOB_COND_ENABLE      (1)    // // EBM End of Buffer continuous sync point
//  is enabled.
#define LWE2B7_CONT_SYNCPT_EBM_EOB_0_EBM_EOB_COND_DEFAULT          (0x00000000)


#define LWE2B7_CONT_SYNCPT_XFER_DONE_0                        (0xc)
// Sync Point Counter Index.
// This parameter specifies the index of the
//  sync point counter that will be returned to
//  host when continuous sync point is enabled
//  (XFER_DONE_COND = ENABLE) and whenever
//  shadow register transfer command (XFER =
//  ENABLE in SHADOW_REG_EN register) is
//  completed.
// This sync point is provided for colwenience
//  so that host need not issue XFER_DONE
//  sync point increment everytime XFER command
//  is issued.
#define LWE2B7_CONT_SYNCPT_XFER_DONE_0_XFER_DONE_INDX                      7:0
#define LWE2B7_CONT_SYNCPT_XFER_DONE_0_XFER_DONE_INDX_DEFAULT      (0x00000000)

// Sync Point Condition Control.
// This bit can be used to enable/disable
//  generation of continuous sync point
//  increment.
#define LWE2B7_CONT_SYNCPT_XFER_DONE_0_XFER_DONE_COND                      8:8
#define LWE2B7_CONT_SYNCPT_XFER_DONE_0_XFER_DONE_COND_DISABLE                     (0)    // // Shadow Register Transfer Done continuous
//  sync point is disabled.

#define LWE2B7_CONT_SYNCPT_XFER_DONE_0_XFER_DONE_COND_ENABLE                      (1)    // // Shadow Register Transfer Done continuous
//  sync point is enabled.
#define LWE2B7_CONT_SYNCPT_XFER_DONE_0_XFER_DONE_COND_DEFAULT      (0x00000000)

//  Sync Point.
#define LWE2B7_CONT_SYNCPT_RD_DONE_0                      (0xd)
// Sync Point Counter Index.
// This parameter specifies the index of the
//  sync point counter that will be returned to
//  host when continuous sync point is enabled
//  (RD_DONE_COND = ENABLE) and whenever an
//  input buffer is completely read by encoder
//  hardware or when the buffer is dropped or
//  skipped for any reason. Note that this does
//  not mean that the input buffer has been
//  fully encoded; it just means that encoder
//  hardware no longer needs this input buffer.
#define LWE2B7_CONT_SYNCPT_RD_DONE_0_RD_DONE_INDX                  7:0
#define LWE2B7_CONT_SYNCPT_RD_DONE_0_RD_DONE_INDX_DEFAULT          (0x00000000)

// Sync Point Condition Control.
// This bit can be used to enable/disable
//  generation of continuous sync point
//  increment.
#define LWE2B7_CONT_SYNCPT_RD_DONE_0_RD_DONE_COND                  8:8
#define LWE2B7_CONT_SYNCPT_RD_DONE_0_RD_DONE_COND_DISABLE                 (0)    // // Read Done (End of Input Buffer) continuous
//  sync point is disabled.

#define LWE2B7_CONT_SYNCPT_RD_DONE_0_RD_DONE_COND_ENABLE                  (1)    // // Read Done (End of Input Buffer) continuous
//  sync point is enabled.
#define LWE2B7_CONT_SYNCPT_RD_DONE_0_RD_DONE_COND_DEFAULT          (0x00000000)

//
// Raise Events
//

// Writing to this register issues raise for
//  input buffer consumed event. When the
//  event oclwrs, the raise vector signal and
//  the raise channel ID are returned to host.
//  This raise is valid only when the input
//  source is host.
#define LWE2B7_RAISE_BUFFER_0                     (0xe)
// Input Buffer Consumed Raise Vector.
// This parameter is the raise vector signal
//  which is returned to host together with
//  the raise channel ID (BUFFER_SIGNAL_CHANNEL)
//  when raise buffer is issued and all the
//  input buffers prior to the raise is
//  consumed (have been encoded).
#define LWE2B7_RAISE_BUFFER_0_BUFFER_SIGNAL                4:0
#define LWE2B7_RAISE_BUFFER_0_BUFFER_SIGNAL_DEFAULT                (0x00000000)

// Input Buffer Consumed Raise Channel ID.
// This parameter is the raise channel ID
//  signal which is returned to host together
//  with the raise vector (BUFFER_SIGNAL)
//  when raise buffer is issued and all the
//  input buffers prior to the raise are
//  consumed (have been encoded).
#define LWE2B7_RAISE_BUFFER_0_BUFFER_SIGNAL_CHANNEL                19:16
#define LWE2B7_RAISE_BUFFER_0_BUFFER_SIGNAL_CHANNEL_DEFAULT        (0x00000000)

// Input Buffer Consumed Event Pending.
// This status bit is set by the hardware
//  when host issued input buffer consumed
//  raise by writing to this register.
// This status bit is then cleared by the
//  hardware when all the previous input
//  buffers prior to the raise buffer command
//  are consumed.
#define LWE2B7_RAISE_BUFFER_0_BUFFER_PENDING                       31:31
#define LWE2B7_RAISE_BUFFER_0_BUFFER_PENDING_DEFAULT               (0x00000000)

// Writing to this register issues raise for
//  frame encoding done event. When the event
//  oclwrs, the raise vector signal and the
//  raise channel ID are returned to host.
#define LWE2B7_RAISE_FRAME_0                      (0xf)
// Frame Encoding Done Raise Vector.
// This parameter is the raise vector signal
//  which is returned to host together with
//  the raise channel ID (FRAME_SIGNAL_CHANNEL)
//  when raise frame is issued and the
//  encoder finished encoding a frame.
#define LWE2B7_RAISE_FRAME_0_FRAME_SIGNAL                  4:0
#define LWE2B7_RAISE_FRAME_0_FRAME_SIGNAL_DEFAULT                  (0x00000000)

// Frame Encoding Done Raise Channel ID.
// This parameter is the raise channel ID
//  signal which is returned to host together
//  with the raise vector (FRAME_SIGNAL)
//  when raise frame is issued and the
//  encoder finished encoding a frame.
#define LWE2B7_RAISE_FRAME_0_FRAME_SIGNAL_CHANNEL                  19:16
#define LWE2B7_RAISE_FRAME_0_FRAME_SIGNAL_CHANNEL_DEFAULT          (0x00000000)

// Frame Encoded Event Pending
// This status bit is set by the hardware
//  when host issued frame encoding done
//  raise by writing to this register.
// This status bit is then cleared by the
//  hardware when a frame encoding is done.
#define LWE2B7_RAISE_FRAME_0_FRAME_PENDING                 31:31
#define LWE2B7_RAISE_FRAME_0_FRAME_PENDING_DEFAULT                 (0x00000000)

// Offset 0x010
//
// Encoder Command Registers.
//

// Writing this register will issue either
//  a start or stop encoding. The MPEG encoder
//  module clock should already be enabled
//  when this register is written.
//  When ME_ENABLE is set to START, the
//  encoder will wait for new buffer trigger
//  from the selected input source as defined
//  by INPUT_SRC_SEL.
//  When ME_ENABLE is set to STOP, the
//  encoder will stop encoding at the next
//  frame boundary, and if INPUT_SRC_SEL is
//  HOST.
//  Multiple frames can be encoded by setting
//  ME_ENABLE to START and then setting it back
//  to STOP when receiving the last frame.
//  If only one frame is to be encoded,
//  the ME_ENABLE bit can be toggled once.
#define LWE2B7_COMMAND_0                  (0x10)
// MPEG Encoder Enable.
// This bit can be used to start/stop video
//  encoding command.
#define LWE2B7_COMMAND_0_ME_ENABLE                 0:0
#define LWE2B7_COMMAND_0_ME_ENABLE_STOP                   (0)    // // Stop encoding at next frame boundary. This
#define LWE2B7_COMMAND_0_ME_ENABLE_DEFAULT                         (0x00000000)
//  takes effect only if encoder is enabled
//  (ME_ENABLE was set to START previously).
//  REG_WR_SAFE sync point may be used after
//  setting ME_ENABLE to STOP  to indicate when
//  it is safe to reprogram encoder registers
//  except for EBM active registers.

#define LWE2B7_COMMAND_0_ME_ENABLE_START                  (1)    // // Start encoding at next frame boundary. This
//  takes effect only if encoder is disabled
//  (after reset or ME_ENABLE was set to STOP
//  previously).
//  Before re-enabling ME_ENABLE from a STOP,
//  make sure all the frames have been processed
//  and all encoded bitstream has been written to 
//  EBM buffer. This can be done by counting the 
//  OPDONE to make sure OPDONE for all the frames 
//  input to MPE are completely processed.
#define LWE2B7_COMMAND_0_ME_ENABLE_DEFAULT                         (0x00000000)

// ---------------------------------------------------------------------------------------------
//
// The following is a list of shadow registers that are effective at the next frame start after
//  XFER bit in SHADOW_REG_EN is set to ENABLE:
// A. SLICE_GRP_MAP_TYPE and CHROMA_QP_INDEX parameters in PIC_PARAMETERS register
// B. NUM_SLICE_GROUPS
// C. PIC_INIT_Q (for H.264 only)
// D. MAX_MIN_QP_I (for H.264 only)
// E. MAX_MIN_QP_P (for H.264 only)
// F. SLICE_PARAMS (for H.264 only)
// G. NUM_OF_UNITS (for H.264 only)
// H. TOP_LEFT (for H.264 only)
// I. BOTTOM_RIGHT (for H.264 only)
// J. CHANGE_RATE (for H.264 only)
//
// K. I_RATE_CTRL
// L. P_RATE_CTRL
// M. RC_ADJUSTMENT
// N. BUFFER_DEPLETION_RATE
// O. BUFFER_SIZE
// P. INITIAL_DELAY_OFFSET
// Q. BUFFER_FULL
// R. REPORTED_FRAME
// 
// ---------------------------------------------------------------------------------------------

// This register controls the writing of
//  shadowed registers into the corresponding
//  shadow and working registers.
//  Shadowed registers are triple-registered.
//  The first stage is written when software
//  writes to the register. The second stage
//  (shadow register stage) is loaded from
//  first stage when XFER is set to ENABLE.
//  The third stage (working register stage)
//  is loaded from the second stage in the
//  next frame start after the XFER is set to
//  ENABLE.
//  When read, shadow registers will return
//  the content of the first stage register.
#define LWE2B7_SHADOW_REG_EN_0                    (0x11)
// Transfer shadow registers.
// A write to this bit (register) will initiate
//  transfer of all shadowed registers
//  first into their shadow register stage and
//  then into their working register stage.
//  If S/W writes XFER=ENABLE twice within the
//  same frame time then the first set of
//  transfer will not take effect. To prevent
//  this, raise should be used or XFER_PENDING
//  bit should be read prior to issuing
//  XFER=ENABLE.
#define LWE2B7_SHADOW_REG_EN_0_XFER                0:0
#define LWE2B7_SHADOW_REG_EN_0_XFER_NOP                   (0)    // // No operation.

#define LWE2B7_SHADOW_REG_EN_0_XFER_ENABLE                        (1)    // // Transfer all shadowed register to the shadow
//  register and then transfer them to the
//  working register at the start of the next
//  frame.
#define LWE2B7_SHADOW_REG_EN_0_XFER_DEFAULT                        (0x00000000)

// Enable Raise on Shadow Register Transfer.
#define LWE2B7_SHADOW_REG_EN_0_EN_RAISE_XFER                       1:1
#define LWE2B7_SHADOW_REG_EN_0_EN_RAISE_XFER_DISABLE                      (0)    // // Raise is disabled when XFER is written and
//  set to ENABLE.

#define LWE2B7_SHADOW_REG_EN_0_EN_RAISE_XFER_ENABLE                       (1)    // // Raise is enabled when XFER is written and
//  set to ENABLE. The raise vector
//  (XFER_RAISE_VECTOR) and the corresponding
//  raise channel ID (XFER_RAISE_CHANNEL) are
//  returned when the transfer is completed.
#define LWE2B7_SHADOW_REG_EN_0_EN_RAISE_XFER_DEFAULT               (0x00000000)

// Shadow Register Transfer Raise Vector.
// This parameter specifies the raise vector
//  to be returned when shadow registers takes
//  effect (transferred to the working register
//  set when XFER and EN_RAISE_XFER are both
//  set to ENABLE.
#define LWE2B7_SHADOW_REG_EN_0_XFER_RAISE_VECTOR                   6:2
#define LWE2B7_SHADOW_REG_EN_0_XFER_RAISE_VECTOR_DEFAULT           (0x00000000)

// Shadow Transfer Raise Channel.
// This parameter specifies the raise channel ID
//  to be returned when shadow registers takes
//  effect (transferred to the working register
//  set when XFER and EN_RAISE_XFER are both
//  set to ENABLE.
#define LWE2B7_SHADOW_REG_EN_0_XFER_RAISE_CHANNEL                  19:16
#define LWE2B7_SHADOW_REG_EN_0_XFER_RAISE_CHANNEL_DEFAULT          (0x00000000)

// Shadow Transfer Event Pending.
// This status bit can be polled by S/W if
//  raise is not used to find out if shadow
//  register transfer is completed or not.
//  This bit is set by hardware when XFER is
//  written and set to ENABLE. Hardware will
//  clear this bit when the shadowed register
//  transfer is completed at the next frame
//  start.
#define LWE2B7_SHADOW_REG_EN_0_XFER_PENDING                31:31
#define LWE2B7_SHADOW_REG_EN_0_XFER_PENDING_DEFAULT                (0x00000000)

// This register is used to send command to
//  trigger the encoder when INPUT_SRC_SEL
//  is HOST.  A write to this register when
//  INPUT_SRC_SEL is HOST indicates that the
//  next input buffer is available to be
//  encoded. This register can be written
//  when current buffer is being encoded.
// After writing this register to encode all
//  buffers for current frame, this register
//  should not be written again to send buffers
/// for next current frame until the encoder
//  finished the current frame encoding.
#define LWE2B7_NEW_BUFFER_0                       (0x12)
// Start of Frame buffer.
// This bit indicates whether the written buffer
//  starts a new frame or not.
#define LWE2B7_NEW_BUFFER_0_START_OF_FRAME                 0:0
#define LWE2B7_NEW_BUFFER_0_START_OF_FRAME_NO_SOF                 (0)    // // This buffer does not start a new frame.

#define LWE2B7_NEW_BUFFER_0_START_OF_FRAME_SOF                    (1)    // // This buffer starts a new frame.
#define LWE2B7_NEW_BUFFER_0_START_OF_FRAME_DEFAULT                 (0x00000000)

// Enable Raise on Buffer
// This bit indicates whether a raise buffer
//  is issued for the written current buffer.
#define LWE2B7_NEW_BUFFER_0_EN_RAISE_BUFFER                1:1
#define LWE2B7_NEW_BUFFER_0_EN_RAISE_BUFFER_DISABLE                       (0)    // // Raise buffer is disabled.

#define LWE2B7_NEW_BUFFER_0_EN_RAISE_BUFFER_ENABLE                        (1)    // // Raise buffer is enabled. The raise vector
//  is returned whenever the encoder finished
//  consuming the input buffer. The raise
//  vector is returned early (as soon as encoder
//  no longer need the current buffer even
//  though the encoding of current buffer may
//  not be fully completed.
#define LWE2B7_NEW_BUFFER_0_EN_RAISE_BUFFER_DEFAULT                (0x00000000)

// VLC DMA Buffer context control.
// This controls VLC DMA context when a new
//  buffer command is issued by writing to
//  this register.
#define LWE2B7_NEW_BUFFER_0_VLCDMA_CONTEXT                 7:7
#define LWE2B7_NEW_BUFFER_0_VLCDMA_CONTEXT_RESUME                 (0)    // // Resume previous VLC DMA context. In this
//  case VLC DMA buffer addresses and sizes
//  are resumed from the internal state at
//  the end of the previous frame encoding.
//  This setting should be used when the
//  new buffer contains a frame which is to
//  be encoded and stored as continuation of
//  previous frame in the same VLC DMA buffer.
//  This can be used even if the rest of
//  encoder context is changed as long as the
//  same VLC DMA buffer context is continued
//  from previous frame. Note that it is not
//  always possible to maintain/continue the
//  same VLC DMA buffer context in an encoder
//  context switch because the new encoder
//  context may require an increase in VLC DMA
//  buffer sizes.
//  This setting is ignored when ME_ENABLE is
//  set to START since VLC DMA context is
//  always reloaded in this case.

#define LWE2B7_NEW_BUFFER_0_VLCDMA_CONTEXT_RELOAD                 (1)    // // Reload previous VLC DMA context as if a
//  ME_ENABLE is set to START. In this case
//  VLC DMA buffer addresses and sizes are
//  re-loaded from the programmed register
//  values. This setting should be used only
//  when START_OF_FRAME is set to SOF.
//  This setting should be used when a new
//  set of VLC DMA buffers are allocated with
//  a new encoding context or when resuming
//  from a previously saved context. Note that
//  if the new set of VLC DMA buffers are to
//  override part or all of the the previous
//  set of VLC DMA buffers, the new buffer
//  command must not be issued until the
//  previous VLC DMA buffers content are
//  completely transferred to EBM memory.
#define LWE2B7_NEW_BUFFER_0_VLCDMA_CONTEXT_DEFAULT                 (0x00000000)

// New Buffer Raise Vector
// This field specifies the raise vector to be
//  returned when EN_RAISE_BUFFER is set.
#define LWE2B7_NEW_BUFFER_0_NEWBUFFER_SIGNAL                       6:2
#define LWE2B7_NEW_BUFFER_0_NEWBUFFER_SIGNAL_DEFAULT               (0x00000000)

// Input Buffer Index
//  This specifies the index of the new input
//  (current) buffer.
#define LWE2B7_NEW_BUFFER_0_IB_INDEX                       15:8
#define LWE2B7_NEW_BUFFER_0_IB_INDEX_DEFAULT                       (0x00000000)

// New Buffer Raise Vector
//  This field specifies the channel ID of the 
//  raise vector to be returned when
//  EN_RAISE_BUFFER is set.
#define LWE2B7_NEW_BUFFER_0_NEWBUFFER_SIGNAL_CHANNEL                       19:16
#define LWE2B7_NEW_BUFFER_0_NEWBUFFER_SIGNAL_CHANNEL_DEFAULT       (0x00000000)

// This register is applicable for H.264
//  encoding only.
// This register specifies sequence picture
//  parameter control.
#define LWE2B7_SEQ_PIC_CTRL_0                     (0x13)
// Sequence Header Parameter Change  
// 
#define LWE2B7_SEQ_PIC_CTRL_0_SEQ_PARAM_CHANGE                     0:0
#define LWE2B7_SEQ_PIC_CTRL_0_SEQ_PARAM_CHANGE_DISABLE                    (0)    // // No operation.

#define LWE2B7_SEQ_PIC_CTRL_0_SEQ_PARAM_CHANGE_ENABLE                     (1)    // // This indicates that sequence header content
//  has changed so sequence header NAL will
//  be sent at start of next frame. This bit
//  will be self cleared by H/W when the next
//  frame is encoded. This register bit must be 
//  enabled for the first frame of a sequence (context)
#define LWE2B7_SEQ_PIC_CTRL_0_SEQ_PARAM_CHANGE_DEFAULT             (0x00000000)

// Picture Header Parameter Change
#define LWE2B7_SEQ_PIC_CTRL_0_PIC_PARAM_CHANGE                     1:1
#define LWE2B7_SEQ_PIC_CTRL_0_PIC_PARAM_CHANGE_DISABLE                    (0)    // // No operation.

#define LWE2B7_SEQ_PIC_CTRL_0_PIC_PARAM_CHANGE_ENABLE                     (1)    // // This indicates that picture header content
//  has changed so picture header NAL will
//  be sent at start of next frame. This bit
//  will be self cleared by H/W when the next
//  frame is encoded.This register bit must be 
//  enabled for the first frame of a sequence (context)
#define LWE2B7_SEQ_PIC_CTRL_0_PIC_PARAM_CHANGE_DEFAULT             (0x00000000)

// This register allows software to force the
//  next frame to be encoded as an I frame.
#define LWE2B7_FORCE_I_FRAME_0                    (0x14)
// Force I frame (self-clearing).
// Writing a "1" to this bit will force the
//  next frame to be encoded as I frame.
//  Hardware will clear this bit to 0 after
//  encoding the I frame.
#define LWE2B7_FORCE_I_FRAME_0_FORCE_I_FRAME                       0:0
#define LWE2B7_FORCE_I_FRAME_0_FORCE_I_FRAME_NOP                  (0)    // // No operation.

#define LWE2B7_FORCE_I_FRAME_0_FORCE_I_FRAME_I_FRAME                      (1)    // // Next frame is encoded as I frame.
// 0x015-0x01D
#define LWE2B7_FORCE_I_FRAME_0_FORCE_I_FRAME_DEFAULT               (0x00000000)

//
// Interrupt Registers
//

// This reflects status of all pending
//  interrupts which is valid as long as
//  the interrupt is not cleared even if the
//  interrupt is masked. Hardware sets bits in
//  this register when interrupt events
//  occur. Software can clear a pending
//  interrupt bit by writing a '1' to the
//  corresponding interrupt status bit
//  in this register.
//  Writing a zero has no effect.
//  These bits are set by hardware regardless
//  of the value of the interrupt enable bits.
#define LWE2B7_INTSTATUS_0                        (0x1e)
// Context Switch Interrupt Status.
// This bit is set to 1 when hardware detects a
//  context swap.
#define LWE2B7_INTSTATUS_0_CTXSW_INT                       0:0
#define LWE2B7_INTSTATUS_0_CTXSW_INT_NOTPENDING                   (0)    // //  interrupt not pending

#define LWE2B7_INTSTATUS_0_CTXSW_INT_PENDING                      (1)    // //  interrupt pending
#define LWE2B7_INTSTATUS_0_CTXSW_INT_DEFAULT                       (0x00000000)

// Frame Encoded Interrupt Status.
// This bit is set to 1 when encoding of a
//  frame is completed. This reflects VLC DMA
//  write status; i.e. the frame has been
//  written to the VLC DMA buffer however it
//  may not be transferred to the encoded
//  bitstream buffer yet.
#define LWE2B7_INTSTATUS_0_FRAME_INT                       1:1
#define LWE2B7_INTSTATUS_0_FRAME_INT_NOTPENDING                   (0)    // //  interrupt not pending

#define LWE2B7_INTSTATUS_0_FRAME_INT_PENDING                      (1)    // //  interrupt pending
#define LWE2B7_INTSTATUS_0_FRAME_INT_DEFAULT                       (0x00000000)

// Buffer Consumed Interrupt Status.
// This bit is set to 1 when an input buffer
//  supplied by the host has been consumed.
#define LWE2B7_INTSTATUS_0_BUFFER_INT                      2:2
#define LWE2B7_INTSTATUS_0_BUFFER_INT_NOTPENDING                  (0)    // //  interrupt not pending

#define LWE2B7_INTSTATUS_0_BUFFER_INT_PENDING                     (1)    // //  interrupt pending
#define LWE2B7_INTSTATUS_0_BUFFER_INT_DEFAULT                      (0x00000000)

// Frame Start Interrupt Status.
// This bit is set to 1 when encoding of a
//  frame has started.
#define LWE2B7_INTSTATUS_0_FRAME_START_INT                 3:3
#define LWE2B7_INTSTATUS_0_FRAME_START_INT_NOTPENDING                     (0)    // //  interrupt not pending

#define LWE2B7_INTSTATUS_0_FRAME_START_INT_PENDING                        (1)    // //  interrupt pending
#define LWE2B7_INTSTATUS_0_FRAME_START_INT_DEFAULT                 (0x00000000)

// Buffer Rejected Interrupt Status.
// This bit is set to 1 when the MPEG encoder
//  could not accept the last NEW_BUFFER
//  command. In this case, host should retry
//  the last NEW_BUFFER command.
#define LWE2B7_INTSTATUS_0_BUFFER_REJ_INT                  4:4
#define LWE2B7_INTSTATUS_0_BUFFER_REJ_INT_NOTPENDING                      (0)    // //  interrupt not pending

#define LWE2B7_INTSTATUS_0_BUFFER_REJ_INT_PENDING                 (1)    // //  interrupt pending
#define LWE2B7_INTSTATUS_0_BUFFER_REJ_INT_DEFAULT                  (0x00000000)

// VLC DMA Overflow Interrupt Status.
// This bit is set to 1 when the MPEG encoder
//  VLC DMA buffer overflows.
#define LWE2B7_INTSTATUS_0_VLC_DMA_OVFL_INT                5:5
#define LWE2B7_INTSTATUS_0_VLC_DMA_OVFL_INT_NOTPENDING                    (0)    // //  interrupt not pending

#define LWE2B7_INTSTATUS_0_VLC_DMA_OVFL_INT_PENDING                       (1)    // //  interrupt pending
#define LWE2B7_INTSTATUS_0_VLC_DMA_OVFL_INT_DEFAULT                (0x00000000)

// Encoded Bitstream (EB) Buffer End Interrupt
//  Status.
// This bit is set to 1 every time an EB buffer
//  is completely written / filled.
#define LWE2B7_INTSTATUS_0_EB_BUFFER_END_INT                       6:6
#define LWE2B7_INTSTATUS_0_EB_BUFFER_END_INT_NOTPENDING                   (0)    // //  interrupt not pending

#define LWE2B7_INTSTATUS_0_EB_BUFFER_END_INT_PENDING                      (1)    // //  interrupt pending
#define LWE2B7_INTSTATUS_0_EB_BUFFER_END_INT_DEFAULT               (0x00000000)

// Encoded Bitstream (EB) Frame End Interrupt
//  Status.
// This bit is set to 1 every time an encoded
//  frame is completely written to EB buffer.
#define LWE2B7_INTSTATUS_0_EB_FRAME_END_INT                7:7
#define LWE2B7_INTSTATUS_0_EB_FRAME_END_INT_NOTPENDING                    (0)    // //  interrupt not pending

#define LWE2B7_INTSTATUS_0_EB_FRAME_END_INT_PENDING                       (1)    // //  interrupt pending
#define LWE2B7_INTSTATUS_0_EB_FRAME_END_INT_DEFAULT                (0x00000000)

// Encoder Idle Interrupt Status.
// This bit is set to 1 when there is no
//  encoding in progress AND encoded data
//  transfer from VLC DMA buffer to EBM buffer
//  is complete. This correspond to the
//  ENCODER_IDLE sync point condition.
#define LWE2B7_INTSTATUS_0_ENCODER_IDLE_INT                8:8
#define LWE2B7_INTSTATUS_0_ENCODER_IDLE_INT_NOTPENDING                    (0)    // //  interrupt not pending

#define LWE2B7_INTSTATUS_0_ENCODER_IDLE_INT_PENDING                       (1)    // //  interrupt pending
#define LWE2B7_INTSTATUS_0_ENCODER_IDLE_INT_DEFAULT                (0x00000000)


// Setting bits in this register enable
//  the corresponding interrrupt event to
//  generate a pending interrupt. Interrupt
//  output signal will be activated only if
//  the corresponding interrupt is not masked
//  (enabled).
// Masking (disabling) an interrupt will not
//  clear a corresponding pending interrupt -
//  it only prevent a new interrupt event to
//  generate a pending interrupt.
//  Note that there is no enable for the
//  context switch interrupt; it is always
//  enabled.
#define LWE2B7_INTMASK_0                  (0x1f)
// Context Switch Interrupt Mask.
// If the auto ack bit in the CTXSW register 
//  is disabled, then software must enable 
//  interrupt and unmask interrupt mask 
//  to get context switch acknowledge
//  This bit is always forced to '1'.
#define LWE2B7_INTMASK_0_CTXSW_INT_ENABLE                  0:0
#define LWE2B7_INTMASK_0_CTXSW_INT_ENABLE_MASKED                  (0)    // // Interrupt masked (disabled).

#define LWE2B7_INTMASK_0_CTXSW_INT_ENABLE_NOTMASKED                       (1)    // // Interrupt not masked (enabled). Note that
//   this interrupt is enabled upon reset.
#define LWE2B7_INTMASK_0_CTXSW_INT_ENABLE_DEFAULT                  (0x00000000)

// Frame Encoded Interrupt Mask.
// When set, this bit enables generation of
//  frame encoding completion interrupts.
//  This reflects VLC DMA write status;
//  i.e. the frame has been written to the
//  VLC DMA buffer however it may not be
//  transferred to the encoded bitstream
//  buffer yet.
#define LWE2B7_INTMASK_0_FRAME_INT_ENABLE                  1:1
#define LWE2B7_INTMASK_0_FRAME_INT_ENABLE_MASKED                  (0)    // // Interrupt masked (disabled).

#define LWE2B7_INTMASK_0_FRAME_INT_ENABLE_NOTMASKED                       (1)    // // Interrupt not masked (enabled).
#define LWE2B7_INTMASK_0_FRAME_INT_ENABLE_DEFAULT                  (0x00000000)

// Buffer Consumed Interrupt Mask.
// When set, this bit enables generation of
//  input buffer consumed interrupts.
#define LWE2B7_INTMASK_0_BUFFER_INT_ENABLE                 2:2
#define LWE2B7_INTMASK_0_BUFFER_INT_ENABLE_MASKED                 (0)    // // Interrupt masked (disabled).

#define LWE2B7_INTMASK_0_BUFFER_INT_ENABLE_NOTMASKED                      (1)    // // Interrupt not masked (enabled).
#define LWE2B7_INTMASK_0_BUFFER_INT_ENABLE_DEFAULT                 (0x00000000)

// Frame Start Interrupt Mask.
// When set, this bit enables generation of
//  frame encoding start interrupts.
#define LWE2B7_INTMASK_0_FRAME_START_INT_ENABLE                    3:3
#define LWE2B7_INTMASK_0_FRAME_START_INT_ENABLE_MASKED                    (0)    // // Interrupt masked (disabled).

#define LWE2B7_INTMASK_0_FRAME_START_INT_ENABLE_NOTMASKED                 (1)    // // Interrupt not masked (enabled).
#define LWE2B7_INTMASK_0_FRAME_START_INT_ENABLE_DEFAULT            (0x00000000)

// Buffer Rejected Interrupt Mask.
// When set, this bit enables generation of
//  buffer rejected interrupts.
#define LWE2B7_INTMASK_0_BUFFER_REJ_INT_ENABLE                     4:4
#define LWE2B7_INTMASK_0_BUFFER_REJ_INT_ENABLE_MASKED                     (0)    // // Interrupt masked (disabled).

#define LWE2B7_INTMASK_0_BUFFER_REJ_INT_ENABLE_NOTMASKED                  (1)    // // Interrupt not masked (enabled).
#define LWE2B7_INTMASK_0_BUFFER_REJ_INT_ENABLE_DEFAULT             (0x00000000)

// VLC DMA Overflow Interrupt Mask.
// When set, this bit enables generation of
//  VLC DMA overflow interrupts.
#define LWE2B7_INTMASK_0_VLC_DMA_OVFL_INT_ENABLE                   5:5
#define LWE2B7_INTMASK_0_VLC_DMA_OVFL_INT_ENABLE_MASKED                   (0)    // // Interrupt masked (disabled).

#define LWE2B7_INTMASK_0_VLC_DMA_OVFL_INT_ENABLE_NOTMASKED                        (1)    // // Interrupt not masked (enabled).
#define LWE2B7_INTMASK_0_VLC_DMA_OVFL_INT_ENABLE_DEFAULT           (0x00000000)

// Encoded Bitstream (EB) Buffer End Interrupt
//  Mask.
// When set, this bit enables generation of
//  EB buffer end interrupts.
#define LWE2B7_INTMASK_0_EB_BUFFER_END_INT_ENABLE                  6:6
#define LWE2B7_INTMASK_0_EB_BUFFER_END_INT_ENABLE_MASKED                  (0)    // // Interrupt masked (disabled).

#define LWE2B7_INTMASK_0_EB_BUFFER_END_INT_ENABLE_NOTMASKED                       (1)    // // Interrupt not masked (enabled).
#define LWE2B7_INTMASK_0_EB_BUFFER_END_INT_ENABLE_DEFAULT          (0x00000000)

// Encoded Bitstream (EB) Frame End Interrupt
//  Mask.
// When set, this bit enables generation of
//  encoded frame end interrupts.
#define LWE2B7_INTMASK_0_EB_FRAME_END_INT_ENABLE                   7:7
#define LWE2B7_INTMASK_0_EB_FRAME_END_INT_ENABLE_MASKED                   (0)    // // Interrupt masked (disabled).

#define LWE2B7_INTMASK_0_EB_FRAME_END_INT_ENABLE_NOTMASKED                        (1)    // // Interrupt not masked (enabled).
#define LWE2B7_INTMASK_0_EB_FRAME_END_INT_ENABLE_DEFAULT           (0x00000000)

// Encoder Idle Interrupt Mask.
// When set, this bit enables generation of
//  encoder idle interrupts.
#define LWE2B7_INTMASK_0_ENCODER_IDLE_INT_ENABLE                   8:8
#define LWE2B7_INTMASK_0_ENCODER_IDLE_INT_ENABLE_MASKED                   (0)    // // Interrupt masked (disabled).

#define LWE2B7_INTMASK_0_ENCODER_IDLE_INT_ENABLE_NOTMASKED                        (1)    // // Interrupt not masked (enabled).
#define LWE2B7_INTMASK_0_ENCODER_IDLE_INT_ENABLE_DEFAULT           (0x00000000)

// Offset 0x020
//
// Video Sequence Registers
//
// Bit 29 (PKT_MODE) of VOL_CTRL register is deleted.
// Bit 30 (HW_DBLK_CTRL) of VOL_CTRL register is deleted.
// Bit 31 (DBLK_EN) of VOL_CTRL register is deleted.

#define LWE2B7_VOL_CTRL_0                 (0x20)
// Short Video Header.
// This bit is effective only for MPEG4/H.263
//  encoding when H264_ENABLE is set to DISABLE.
#define LWE2B7_VOL_CTRL_0_SHORT_VIDEO_HEADER                       0:0
#define LWE2B7_VOL_CTRL_0_SHORT_VIDEO_HEADER_MPEG4                        (0)    // // MPEG4 encoding.

#define LWE2B7_VOL_CTRL_0_SHORT_VIDEO_HEADER_H263                 (1)    // // H.263 (short video header) encoding.
#define LWE2B7_VOL_CTRL_0_SHORT_VIDEO_HEADER_DEFAULT               (0x00000000)

// Bypass VLC Control.
// This bit is used to disable/enable bypassing
//  of VLC hardware.
// This bit is lwrrently available only for
//  MPEG4/H.263 encoding.
#define LWE2B7_VOL_CTRL_0_BYPASS_VLC                       1:1
#define LWE2B7_VOL_CTRL_0_BYPASS_VLC_DISABLE                      (0)    // // Hardware VLC is not bypassed.

#define LWE2B7_VOL_CTRL_0_BYPASS_VLC_ENABLE                       (1)    // // Hardware VLC is bypassed. 
#define LWE2B7_VOL_CTRL_0_BYPASS_VLC_DEFAULT                       (0x00000000)

// Resync Marker Control.
// This bit is used to disable/enable the
//  resync marker insertion in the VLC.
// This bit is effective only for MPEG4 encoding
//  and is not used for H.264 encoding.
#define LWE2B7_VOL_CTRL_0_RESYNC_ENABLE                    2:2
#define LWE2B7_VOL_CTRL_0_RESYNC_ENABLE_DISABLE                   (0)    // // Resync marker insertion is disabled.

#define LWE2B7_VOL_CTRL_0_RESYNC_ENABLE_ENABLE                    (1)    // // Resync marker insertion is enabled.
#define LWE2B7_VOL_CTRL_0_RESYNC_ENABLE_DEFAULT                    (0x00000000)

// Data Partition Control.
// This bit is used to enable/disable the data
//  partition in the VLC.
// This bit is effective only for MPEG4 encoding
//  and is not used for H.264 encoding.
#define LWE2B7_VOL_CTRL_0_DATAPART_ENABLE                  3:3
#define LWE2B7_VOL_CTRL_0_DATAPART_ENABLE_DISABLE                 (0)    // // Data partitioning is disabled.

#define LWE2B7_VOL_CTRL_0_DATAPART_ENABLE_ENABLE                  (1)    // // Data partitioning is enabled.
#define LWE2B7_VOL_CTRL_0_DATAPART_ENABLE_DEFAULT                  (0x00000000)

// Reversible VLC Control.
// This bit is used to enable/disable the
//  Reversible VLC.
// This bit is effective only for MPEG4 encoding
//  and is not used for H.264 encoding.
#define LWE2B7_VOL_CTRL_0_RVLC_ENABLE                      4:4
#define LWE2B7_VOL_CTRL_0_RVLC_ENABLE_DISABLE                     (0)    // // Reversible VLC is disabled.

#define LWE2B7_VOL_CTRL_0_RVLC_ENABLE_ENABLE                      (1)    // // Reversible VLC is enabled.
#define LWE2B7_VOL_CTRL_0_RVLC_ENABLE_DEFAULT                      (0x00000000)

// Input Buffer Frame Skip.
// This bit defines whether to skip (drop) or
//  not skip incoming frames when the current
//  frame encoding is not finished and input
//  buffer for next frame arrives.
// It is recommended that for real-time camera
//  encoding from host
//  that this bit be set to "SKIP" to prevent
//  buffer overflow in host especially
//  if input buffers are processed in frame
//  buffers. When next frame is skipped due to
//  this condition, interrupt (BUFFER_REJ_INT)
//  can be set if enabled.
//  This condition should not occur normally
//  for real-time encoding because it is
//  expected that the system is designed
//  such that it does not exceed the encoder
//  encoding capability even in worst case.
//  Note that if input frames are processed in
//  mb-rows, no skipping is performed on input
//  buffers until the buffer with start of next
//  frame appears at the top of the input
//  buffer queue (FIFO) therefore if video
//  source is faster than encoder, overflow
//  may still occur at the source of video.
// For non real-time host-controlled encoding
//  this bit should be set to "NO_SKIP" to
//  prevent unnecessary frame skipping.
// Setting this bit to NO_SKIP will cause the
//  encoding source to be stalled if encoder
//  cannot keep up with the source.
#define LWE2B7_VOL_CTRL_0_IB_FRAME_SKIP_MODE                       5:5
#define LWE2B7_VOL_CTRL_0_IB_FRAME_SKIP_MODE_NOSKIP                       (0)    // // Do not skip the next frame at input buffer
//  interface, until the current frame encoding
//  is done. 

#define LWE2B7_VOL_CTRL_0_IB_FRAME_SKIP_MODE_SKIP                 (1)    // // Skip input buffers - Skip the next frame at
//  the input buffer interface, if the current
//  frame encoding is not finished. 
#define LWE2B7_VOL_CTRL_0_IB_FRAME_SKIP_MODE_DEFAULT               (0x00000000)

// Stream Mode.
// This bit specifies the mode of output
//  bitstream.
#define LWE2B7_VOL_CTRL_0_STREAM_MODE                      6:6
#define LWE2B7_VOL_CTRL_0_STREAM_MODE_BYTESTREAM                  (0)    // // This enum is applicable to H264 only.
//  H264 bitstream will be generated in
//  byte stream mode. In this mode, EB Buffer
//  End Interrupt should be masked and only
//  EB Frame End Interrupt should be used.

#define LWE2B7_VOL_CTRL_0_STREAM_MODE_NAL                 (1)    // // This enum is applicable to H264 only.
//  H264 bitstream will be generated in NAL
//  mode. In this mode, EB Buffer End Interrupt
//  may be used in addition to EB Frame End
//  Interrupt.

#define LWE2B7_VOL_CTRL_0_STREAM_MODE_FRAME                       (0)    // // This enum is applicable to MPEG4 only.
//  MPEG4 bitstream will be returned in 
//  one complete frame.  In this mode, EB 
//  Buffer End Interrupt should be masked 
//  and only EB Frame End Interrupt should be 
//  used.  

#define LWE2B7_VOL_CTRL_0_STREAM_MODE_PACKET                      (1)    // // This enum is applicable to MPEG4 only.
//  MPEG4 bitstream will be return in packets,
//  each with its own header. In this 
//  mode, EB Buffer End Interrupt may be used 
//  in addition to EB Frame End Interrupt.
#define LWE2B7_VOL_CTRL_0_STREAM_MODE_DEFAULT                      (0x00000000)

// Input Source Select.
// This parameter specifies input source.
#define LWE2B7_VOL_CTRL_0_INPUT_SRC_SEL                    9:8
#define LWE2B7_VOL_CTRL_0_INPUT_SRC_SEL_HOST                      (2)    // // Host-controlled encoding.
// 0 and 1 are Reserved
#define LWE2B7_VOL_CTRL_0_INPUT_SRC_SEL_DEFAULT                    (0x00000000)

// H.264 Encoding Enable.
//  This bit enables H.264 encoding.
#define LWE2B7_VOL_CTRL_0_H264_ENABLE                      11:11
#define LWE2B7_VOL_CTRL_0_H264_ENABLE_DISABLE                     (0)    // // MPEG4 or H.263 encoding as selected by
//  SHORT_VIDEO_HEADER bit.

#define LWE2B7_VOL_CTRL_0_H264_ENABLE_ENABLE                      (1)    // // H.264 encoding.
#define LWE2B7_VOL_CTRL_0_H264_ENABLE_DEFAULT                      (0x00000000)

// VLC DMA Stall control.
// This bit controls VLC DMA behavior in case
//  any of the VLC DMA buffer(s) is full.
//  This bit is valid regardless of the setting
//  of BYPASS_VLCDMA.
#define LWE2B7_VOL_CTRL_0_VLCDMA_STALL                     14:14
#define LWE2B7_VOL_CTRL_0_VLCDMA_STALL_ENABLE                     (0)    // // VLC DMA will stall encoding pipeline if
//  any of the VLC DMA buffer(s) is full.
//  In this case VLC DMA will never overflow
//  however this may potentially cause input
//  buffer overflow. This setting should be used
//  when using input frame buffers. Pending
//  input buffer overflow should be detected
//  (in S/W or in H/W) and input frame(s)
// dropped.

#define LWE2B7_VOL_CTRL_0_VLCDMA_STALL_DISABLE                    (1)    // // VLC DMA will not stall encoding pipeline
//  if any of the VLC DMA buffer(s) is full.
//  This will create a VLC DMA overflow
//  condition that will trigger VLC_DMA_OVFL_INT
//  interrupt if enabled. This setting should
//  be used when using input mb-row buffers to
//  minimize the risk of input buffer being
//  dropped.
#define LWE2B7_VOL_CTRL_0_VLCDMA_STALL_DEFAULT                     (0x00000000)

// Bypass VLC DMA.
// This bit controls bypassing of VLC DMA
//  buffer.
// This feature is not implemented yet !!
#define LWE2B7_VOL_CTRL_0_BYPASS_VLCDMA                    15:15
#define LWE2B7_VOL_CTRL_0_BYPASS_VLCDMA_DISABLE                   (0)    // // VLC DMA is not bypassed.
// This setting must be used for:
// a) MPEG4/H.263 encoding with either
//    DATAPART_ENABLE or RVLC_ENABLE set to
//    ENABLE and both BYPASS_VLC and ME_BYPASS
//    set to DISABLE.
// b) H.264 encoding with multiple slice and
//    both BYPASS_VLC and ME_BYPASS set to
//    DISABLE.

#define LWE2B7_VOL_CTRL_0_BYPASS_VLCDMA_ENABLE                    (1)    // // VLC DMA is bypassed.
// This setting can be used for:
// a) MPEG4/H.263 encoding with DATAPART_ENABLE
//    and RVLC_ENABLE both set to DISABLE.
// b) H.264 encoding with single slice
// c) MPEG4/H.263/H.264 encoding with
//    BYPASS_VLC set to ENABLE.
// d) MPEG4/H.263/H.264 encoding with
//    ME_BYPASS set to ENABLE.
#define LWE2B7_VOL_CTRL_0_BYPASS_VLCDMA_DEFAULT                    (0x00000000)

// H.263+ Annex I support.
// This bit is effective only for H.263 encoding
//  and is not used for MPEG4 and H.264
//  encoding.
// This functionality is not implemented.
#define LWE2B7_VOL_CTRL_0_H263PI                   16:16
#define LWE2B7_VOL_CTRL_0_H263PI_DISABLE                  (0)    // // H.263+ Annex I disabled.

#define LWE2B7_VOL_CTRL_0_H263PI_ENABLE                   (1)    // // H.263+ Annex I enabled.
#define LWE2B7_VOL_CTRL_0_H263PI_DEFAULT                           (0x00000000)

// H.263+ Annex J support.
// This bit is effective only for H.263 encoding
//  and is not used for MPEG4 and H.264
//  encoding.
// This functionality is not implemented.
#define LWE2B7_VOL_CTRL_0_H263PJ                   17:17
#define LWE2B7_VOL_CTRL_0_H263PJ_DISABLE                  (0)    // // H.263+ Annex J disabled.

#define LWE2B7_VOL_CTRL_0_H263PJ_ENABLE                   (1)    // // H.263+ Annex J enabled.
#define LWE2B7_VOL_CTRL_0_H263PJ_DEFAULT                           (0x00000000)

// H.263+ Annex K support.
// This bit is effective only for H.263 encoding
//  and is not used for MPEG4 and H.264
//  encoding.
// This functionality is not implemented.
#define LWE2B7_VOL_CTRL_0_H263PK                   18:18
#define LWE2B7_VOL_CTRL_0_H263PK_DISABLE                  (0)    // // H.263+ Annex K disabled.

#define LWE2B7_VOL_CTRL_0_H263PK_ENABLE                   (1)    // // H.263+ Annex K enabled.
#define LWE2B7_VOL_CTRL_0_H263PK_DEFAULT                           (0x00000000)

// H.263+ Annex T support.
// This bit is effective only for H.263 encoding
//  and is not used for MPEG4 and H.264
//  encoding.
// This functionality is not implemented.
#define LWE2B7_VOL_CTRL_0_H263PT                   19:19
#define LWE2B7_VOL_CTRL_0_H263PT_DISABLE                  (0)    // // H.263+ Annex T disabled.

#define LWE2B7_VOL_CTRL_0_H263PT_ENABLE                   (1)    // // H.263+ Annex T enabled.
#define LWE2B7_VOL_CTRL_0_H263PT_DEFAULT                           (0x00000000)

// Number of Reference Frames (H264 only)
// This field specifies the number of reference
//  frames for encoding (from 1 to 16).
// Programmed value = actual value - 1.
// This bit is effective only for H.264 encoding
//  and is not used for MPEG4/H.263 encoding.
// This functionality is not implemented.
#define LWE2B7_VOL_CTRL_0_NUM_REF_FRAME                    24:21
#define LWE2B7_VOL_CTRL_0_NUM_REF_FRAME_DEFAULT                    (0x00000000)

// Motion Estimation Bypass.
// This controls whether encoding process is
//  performed or bypassed after motion
//  estimation process. 
#define LWE2B7_VOL_CTRL_0_ME_BYPASS                25:25
#define LWE2B7_VOL_CTRL_0_ME_BYPASS_DISABLE                       (0)    // // The rest of encoding process is performed
//  after motion estimation.

#define LWE2B7_VOL_CTRL_0_ME_BYPASS_ENABLE                        (1)    // // The rest of encoding process is bypassed
//  after motion estimation.
//  Result of motion estimation is stored in
//  memory in packets aligned to 128-bit
//  words.
// This is supported only for H.264 encoding?
// Design note: internally this will force
//  VLC module to code with MB  packetization
//  packet count to 1.
#define LWE2B7_VOL_CTRL_0_ME_BYPASS_DEFAULT                        (0x00000000)

// Motion Estimation Bypass Mode.
// This controls motion estimation bypass mode
//  when ME_BYPASS is set to ENABLE. This is not
//  effective if ME_BYPASS is set to DISABLE.
// This functionality is not yet implemented.
#define LWE2B7_VOL_CTRL_0_ME_BYPASS_MODE                   26:26
#define LWE2B7_VOL_CTRL_0_ME_BYPASS_MODE_FULL                     (0)    // // Full Bypass Mode.
// Motion vectors and predicted luma/chroma
//  as well as recon data are written to output
//  (EBM) memory when ME_BYPASS is set to
//  ENABLE.

#define LWE2B7_VOL_CTRL_0_ME_BYPASS_MODE_MV_ONLY                  (1)    // // Motion Vector Only.
// Only motion vectors will be writtent to
//  output (EBM) memory when ME_BYPASS is set to
//  ENABLE.
#define LWE2B7_VOL_CTRL_0_ME_BYPASS_MODE_DEFAULT                   (0x00000000)

// Rate Control Mode
// This bit must be set to VBR for MPEG4/H.263
//  encoding and to CBR for H.264 encoding.
#define LWE2B7_VOL_CTRL_0_RATE_CTRL_MODE                   27:27
#define LWE2B7_VOL_CTRL_0_RATE_CTRL_MODE_VBR                      (0)    // // Encode with a more Variable Bit Rate
//  characteristic.
//  This is lwrrently supported for MPEG4/H.263
//  encoding only.

#define LWE2B7_VOL_CTRL_0_RATE_CTRL_MODE_CBR                      (1)    // // Encode with a more Constant Bit Rate
//  characteristic.
//  This is lwrrently supported for H.264
//  encoding only.
#define LWE2B7_VOL_CTRL_0_RATE_CTRL_MODE_DEFAULT                   (0x00000000)

// Current Surface memory organization.
#define LWE2B7_VOL_CTRL_0_LWR_SURF                 28:28
#define LWE2B7_VOL_CTRL_0_LWR_SURF_THREE_PLANE                    (0)    // // Y,U,V are in three separate planes (planar).
//  This setting is not recommended because
//  it is less efficient from memory bandwidth
//  utilization point of view.
//  However, this setting must be used if the
//  video source module does not support
//  semi-planar mode.

#define LWE2B7_VOL_CTRL_0_LWR_SURF_TWO_PLANE                      (1)    // // Y and UV are in two separate planes
//  (semi-planar). Chroma (UV) is stored
//  byte-interleaved in the same plane with
//  U bytes in even addresses and V bytes in
//  odd addresses. This setting is recommended
//  for better memory bandwidth utilization
//  however this may not be possible if the
//  video input source module does not support
//  semi-planar mode.
#define LWE2B7_VOL_CTRL_0_LWR_SURF_DEFAULT                         (0x00000000)

// Reference Surface memory organization.
#define LWE2B7_VOL_CTRL_0_REF_SURF                 29:29
#define LWE2B7_VOL_CTRL_0_REF_SURF_THREE_PLANE                    (0)    // // Y,U,V are in three separate planes (planar).
//  This setting is not recommended because
//  it is less efficient from memory bandwidth
//  utilization point of view.
//  However, for AP15, this setting must be used
//  if reference frame is used for encoding
//  preview because display controller does not
//  support semi-planar mode.

#define LWE2B7_VOL_CTRL_0_REF_SURF_TWO_PLANE                      (1)    // // Y and UV are in two separate planes
//  (semi-planar). Chroma (UV) is stored
//  byte-interleaved in the same plane with
//  U bytes in even addresses and V bytes in
//  odd addresses. This setting is recommended
//  for better memory bandwidth utilization.
#define LWE2B7_VOL_CTRL_0_REF_SURF_DEFAULT                         (0x00000000)


// This register specifies the width and height
//  of the VOP. The same register is used to
//  define the Y, U & V plane width and heights.
// The width and height for the U & V plane
//  will be half of the Y plane for planar YUV.
// The height for the UV plane will be half of
//  Y plane for semi-planar YUV.
#define LWE2B7_WIDTH_HEIGHT_0                     (0x21)
// VOP width in pixel units modulo 16.
// AP15 doesn't support width to be non-multiple of 16.
// In AP15, this field must be set to 0.
#define LWE2B7_WIDTH_HEIGHT_0_WIDTH_4LSB                   3:0
#define LWE2B7_WIDTH_HEIGHT_0_WIDTH_4LSB_DEFAULT                   (0x00000000)

// VOP Width in macro block unit.
// This field specifies the width of the VOP in
//  terms of complete macroblocks.
// This is equal to VOP width in pixel units
//  divided by 16  and fraction truncated.
// Input image width = WIDTH * 16 + WIDTH_4LSB.
// If input image width is not a multiple of 16,
//  the remaining pixels to make width multiple
//  of 16 will be padded internally inside MPE.
// Recon buffers must be programmed to hold the
//  nearest multiple of 16 image size.
// For example if the input image size is
//  (150,128), recon buffer must be large
//  enough to hold (160,128).
// If Y image width is not multiple of 2, U/V
//  image width is assumed to be Y width
//  divided by 2 and rounded up.
#define LWE2B7_WIDTH_HEIGHT_0_WIDTH                11:4
#define LWE2B7_WIDTH_HEIGHT_0_WIDTH_DEFAULT                        (0x00000000)

// VOP height in pixel units modulo 16.
// AP15 doesn't support height to be non-multiple of 16
// In AP15, this field must be set to 0.
#define LWE2B7_WIDTH_HEIGHT_0_HEIGHT_4LSB                  19:16
#define LWE2B7_WIDTH_HEIGHT_0_HEIGHT_4LSB_DEFAULT                  (0x00000000)

// VOP Height in macro block unit.
// This field specifies the height of the VOP in
//  terms of complete macroblocks.
// This is equal to VOP height in pixel units
//  divided by 16 and fraction truncated.
// Input image height = HEIGHT*16 + HEIGHT_4LSB.
// If input image height is not a multiple of
//  16, the remaining lines to make height
//  multiple of 16 will be padded internally
//  inside MPE.
// Recon buffers must be programmed to hold the
//  nearest multiple of 16 image size.
// For example if the input image size is
//  (160,118), recon buffer must be large
//  enough to hold (160,128).
// If Y image height is not multiple of 2, U/V
//  image height is assumed to be Y height
//   divided by 2 and rounded up.
#define LWE2B7_WIDTH_HEIGHT_0_HEIGHT                       27:20
#define LWE2B7_WIDTH_HEIGHT_0_HEIGHT_DEFAULT                       (0x00000000)

// Pixel Extension Mode.
// This is used to select the pixel extension
//  mode when the VOP width in pixel units is
//  not a multiple of 16. This is possible for
//  MPEG4/H.263 encoding since width and height
//  are encoded in term of number of pixels
//  and number of lines. This should not
//  occur in H.264 encoding since width and
//  height are encoded in term of number of
//  of macroblocks.
#define LWE2B7_WIDTH_HEIGHT_0_PIX_EXTEND_MODE                      31:31
#define LWE2B7_WIDTH_HEIGHT_0_PIX_EXTEND_MODE_BLACK                       (0)    // // The VOP is padded with pixel value 0 for
//  luma and pixel value 128 for chroma till
//  the closest macroblock boundary.

#define LWE2B7_WIDTH_HEIGHT_0_PIX_EXTEND_MODE_REPLICATE                   (1)    // // The VOP is padded with the last pixel either
//  on the right or the bottom till the closest
//  macroblock boundary. This setting should
//  be used for MPEG4 encoding.
#define LWE2B7_WIDTH_HEIGHT_0_PIX_EXTEND_MODE_DEFAULT              (0x00000000)


// This register is applicable for H.264
//  encoding only.
// This register specifies various H.264
//  sequence picture parameters.
#define LWE2B7_SEQ_PARAMETERS_0                   (0x22)
// 
// 
#define LWE2B7_SEQ_PARAMETERS_0_IDC_PROFILE                7:0
#define LWE2B7_SEQ_PARAMETERS_0_IDC_PROFILE_DEFAULT                (0x00000042)

// 
// 
#define LWE2B7_SEQ_PARAMETERS_0_CONSTRAINT0                8:8
#define LWE2B7_SEQ_PARAMETERS_0_CONSTRAINT0_DEFAULT                (0x00000001)

// 
// 
#define LWE2B7_SEQ_PARAMETERS_0_CONSTRAINT1                9:9
#define LWE2B7_SEQ_PARAMETERS_0_CONSTRAINT1_DEFAULT                (0x00000000)

// 
// 
#define LWE2B7_SEQ_PARAMETERS_0_CONSTRAINT2                10:10
#define LWE2B7_SEQ_PARAMETERS_0_CONSTRAINT2_DEFAULT                (0x00000000)

// 
// 
#define LWE2B7_SEQ_PARAMETERS_0_CONSTRAINT3                11:11
#define LWE2B7_SEQ_PARAMETERS_0_CONSTRAINT3_DEFAULT                (0x00000000)

// This parameter specifies H.264 Level number.
// When frame size > 1620 macroblocks, Level 
//  should be set >= 31
// When frame size > 3600 macroblocks, Level 
//  should be set >= 32
// When frame size > 5120 macroblocks, Level
//  should be set >= 40
// 
#define LWE2B7_SEQ_PARAMETERS_0_LEVEL_IDC                  19:12
#define LWE2B7_SEQ_PARAMETERS_0_LEVEL_IDC_DEFAULT                  (0x0000001e)

// 
// 
#define LWE2B7_SEQ_PARAMETERS_0_PIC_ORDER_CNT_TYPE                 23:20
#define LWE2B7_SEQ_PARAMETERS_0_PIC_ORDER_CNT_TYPE_DEFAULT         (0x00000002)

// 
#define LWE2B7_SEQ_PARAMETERS_0_GAPS_IN_FRAME_NUM_VALUE_ALLOWED                    24:24
#define LWE2B7_SEQ_PARAMETERS_0_GAPS_IN_FRAME_NUM_VALUE_ALLOWED_DEFAULT    (0x00000001)

// 
// 
#define LWE2B7_SEQ_PARAMETERS_0_FRAME_MBS_ONLY                     25:25
#define LWE2B7_SEQ_PARAMETERS_0_FRAME_MBS_ONLY_DEFAULT             (0x00000001)

// 
#define LWE2B7_SEQ_PARAMETERS_0_MB_ADAPTIVE_FRAME_FIELD                    26:26
#define LWE2B7_SEQ_PARAMETERS_0_MB_ADAPTIVE_FRAME_FIELD_DEFAULT    (0x00000000)

// 
#define LWE2B7_SEQ_PARAMETERS_0_DIRECT_8X8_INFERENCE                       27:27
#define LWE2B7_SEQ_PARAMETERS_0_DIRECT_8X8_INFERENCE_DEFAULT       (0x00000000)

// 
// 
#define LWE2B7_SEQ_PARAMETERS_0_FRAME_CROP                 28:28
#define LWE2B7_SEQ_PARAMETERS_0_FRAME_CROP_DEFAULT                 (0x00000000)

// 
// 
#define LWE2B7_SEQ_PARAMETERS_0_DELTA_PIC_ORDER                    29:29
#define LWE2B7_SEQ_PARAMETERS_0_DELTA_PIC_ORDER_DEFAULT            (0x00000001)

// This register is applicable for H.264
//  encoding only.
// This register is needed as part of H.264
//  sequence parameters.
#define LWE2B7_LOG2_MAX_FRAME_NUM_MINUS4_0                        (0x23)
// Log2(Max Frame Number + 1) - 4.
// This parameter specifies the number of bits
//  - 4 for encoded frame number (FRAME_NUM).
//  Note that this also specifies how encoded
//  frame number counter should wrap around in
//  H.264 encoding.
// Valid value is from 0 to 12.
// Note that Max Frame Number can be callwlated
//  from 2^^(LOG2_MAX_FRAME_NUM_MINUS4 + 4) - 1.
//  And, absolute max frame number is 16'hffff.
#define LWE2B7_LOG2_MAX_FRAME_NUM_MINUS4_0_LOG2_MAX_FRAME_NUM_MINUS4                       3:0
#define LWE2B7_LOG2_MAX_FRAME_NUM_MINUS4_0_LOG2_MAX_FRAME_NUM_MINUS4_DEFAULT   (0x00000004)

// This register is applicable for H.264
//  encoding only.
// This register is needed as part of H.264
//  sequence parameters.
#define LWE2B7_CROP_LR_OFFSET_0                   (0x24)
// Left Offset
// 
#define LWE2B7_CROP_LR_OFFSET_0_CROP_LEFT_OFFSET                   15:0
#define LWE2B7_CROP_LR_OFFSET_0_CROP_LEFT_OFFSET_DEFAULT           (0x00000000)

// Right Offset
// 
#define LWE2B7_CROP_LR_OFFSET_0_CROP_RIGHT_OFFSET                  31:16
#define LWE2B7_CROP_LR_OFFSET_0_CROP_RIGHT_OFFSET_DEFAULT          (0x00000000)

// This register is applicable for H.264
//  encoding only.
// This register is needed as part of H.264
//  sequence parameters.
#define LWE2B7_CROP_TB_OFFSET_0                   (0x25)
// Top Offset
// 
#define LWE2B7_CROP_TB_OFFSET_0_CROP_TOP_OFFSET                    15:0
#define LWE2B7_CROP_TB_OFFSET_0_CROP_TOP_OFFSET_DEFAULT            (0x00000000)

// Bottom Offset
// 
#define LWE2B7_CROP_TB_OFFSET_0_CROP_BOT_OFFSET                    31:16
#define LWE2B7_CROP_TB_OFFSET_0_CROP_BOT_OFFSET_DEFAULT            (0x00000000)

// Offset 0x030
// Bit 5 (DEBLK_FILTER_CTRL) of PIC_PARAMETERS is deleted.

// This register is applicable for both H.264 and MPEG4
// There is one register field (STREAM_ID) of this register which is
// applicable to both MPEG4/H264.
// Remaining register fields are applicable to H264 only and specify various
// picture paramters syntax elements.
#define LWE2B7_PIC_PARAMETERS_0                   (0x30)
// Applicable to H264 only
#define LWE2B7_PIC_PARAMETERS_0_ENTROPY_CODING_MODE                0:0
#define LWE2B7_PIC_PARAMETERS_0_ENTROPY_CODING_MODE_DEFAULT        (0x00000000)

// Applicable to H264 only
#define LWE2B7_PIC_PARAMETERS_0_PIC_ORDER_PRESENT                  1:1
#define LWE2B7_PIC_PARAMETERS_0_PIC_ORDER_PRESENT_DEFAULT          (0x00000000)

// Applicable to H264 only
#define LWE2B7_PIC_PARAMETERS_0_WEIGHTED_PRED_FLAG                 2:2
#define LWE2B7_PIC_PARAMETERS_0_WEIGHTED_PRED_FLAG_DEFAULT         (0x00000000)

// Applicable to H264 only
// 
#define LWE2B7_PIC_PARAMETERS_0_WEIGHTED_BIPRED_IDC                4:3
#define LWE2B7_PIC_PARAMETERS_0_WEIGHTED_BIPRED_IDC_DEFAULT        (0x00000000)

// Applicable to H264 only
// 
#define LWE2B7_PIC_PARAMETERS_0_CONSTRAINED_INTRA_PRED                     6:6
#define LWE2B7_PIC_PARAMETERS_0_CONSTRAINED_INTRA_PRED_DEFAULT     (0x00000000)

// Applicable to H264 only
// 
#define LWE2B7_PIC_PARAMETERS_0_REDUNDANT_PIC_CNT                  7:7
#define LWE2B7_PIC_PARAMETERS_0_REDUNDANT_PIC_CNT_DEFAULT          (0x00000000)

// Slice Group Map Type (applicable to H264 only)
// This parameter defines slice group map type
//  when there are multiple slice groups.
//  This parameter is not used when there is
//  only 1 slice group. 
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
#define LWE2B7_PIC_PARAMETERS_0_SLICE_GRP_MAP_TYPE                 10:8
#define LWE2B7_PIC_PARAMETERS_0_SLICE_GRP_MAP_TYPE_INTERLEAVED                    (0)    // // Interleaved slice group map.

#define LWE2B7_PIC_PARAMETERS_0_SLICE_GRP_MAP_TYPE_DISPERSED                      (1)    // // Dispersed slice group map.

#define LWE2B7_PIC_PARAMETERS_0_SLICE_GRP_MAP_TYPE_FOREGROUND_BACKGROUND                  (2)    // // Foreground & Background slice group map.

#define LWE2B7_PIC_PARAMETERS_0_SLICE_GRP_MAP_TYPE_BOX_OUT                        (3)    // // Box out slice group map.

#define LWE2B7_PIC_PARAMETERS_0_SLICE_GRP_MAP_TYPE_RASTER                 (4)    // // Raster slice group map.

#define LWE2B7_PIC_PARAMETERS_0_SLICE_GRP_MAP_TYPE_WIPE                   (5)    // // Wipe slice group map.

#define LWE2B7_PIC_PARAMETERS_0_SLICE_GRP_MAP_TYPE_EXPLICIT                       (6)    // // Explicit slice group map.
// This is lwrrently not supported.
#define LWE2B7_PIC_PARAMETERS_0_SLICE_GRP_MAP_TYPE_DEFAULT         (0x00000000)

// Chroma Qp Index.
// This field is applicable for H.264
//  encoding only.
// This is 2's complement value ranging from
//  -12 to +12 used to callwlate chroma Qp
//  from luma Qp as specified in H.264 standard.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
#define LWE2B7_PIC_PARAMETERS_0_CHROMA_QP_INDEX                    20:16
#define LWE2B7_PIC_PARAMETERS_0_CHROMA_QP_INDEX_DEFAULT            (0x00000000)

// Stream ID
// This register field is applicable to both H264 and MPEG4
// This parameter may be used to differentiate
//  between multiple streams in EBM.
// This parameter must be used for multiple
//  stream encoding.
// Note that multiple stream encoding rely on
//  context save/restore to work in AP15 which
//  is a P1 feature so likely not to work.
#define LWE2B7_PIC_PARAMETERS_0_STREAM_ID                  31:24
#define LWE2B7_PIC_PARAMETERS_0_STREAM_ID_DEFAULT                  (0x00000000)

// This register is applicable for H.264
//  encoding only.
// This register is needed as part of H.264
//  picture parameters.
#define LWE2B7_PIC_PARAM_SET_ID_0                 (0x31)
// Picture Parameter Set ID.
// Valid value is from 0 to 255.
#define LWE2B7_PIC_PARAM_SET_ID_0_PIC_PARAM_SET_ID                 7:0
#define LWE2B7_PIC_PARAM_SET_ID_0_PIC_PARAM_SET_ID_DEFAULT         (0x00000000)

// This register is applicable for H.264
//  encoding only.
// This register is lwrrently not used.
#define LWE2B7_SEQ_PARAM_SET_ID_0                 (0x32)
// Sequence Parameter Set ID.
// Valid value is from 0 to 31. This parameter
//  is encoded into PIC NAL and it identifies
//  which SEQ NAL it is referring to.
#define LWE2B7_SEQ_PARAM_SET_ID_0_SEQ_PARAM_SET_ID                 4:0
#define LWE2B7_SEQ_PARAM_SET_ID_0_SEQ_PARAM_SET_ID_DEFAULT         (0x00000000)

// This register is applicable for H.264
//  encoding only.
// This register is needed as part of H.264
//  picture parameters.
#define LWE2B7_NUM_REF_IDX_ACTIVE_0                       (0x33)
// Number of reference frame - 1.
// Valid value is from 0 to 31.
#define LWE2B7_NUM_REF_IDX_ACTIVE_0_NUM_REF_IDX_I0_ACTIVE_MINUS1                   4:0
#define LWE2B7_NUM_REF_IDX_ACTIVE_0_NUM_REF_IDX_I0_ACTIVE_MINUS1_DEFAULT   (0x00000000)

// Number of reference frame - 1.
// Valid value is from 0 to 31.
#define LWE2B7_NUM_REF_IDX_ACTIVE_0_NUM_REF_IDX_I1_ACTIVE_MINUS1                   12:8
#define LWE2B7_NUM_REF_IDX_ACTIVE_0_NUM_REF_IDX_I1_ACTIVE_MINUS1_DEFAULT   (0x00000000)

// This register is applicable for H.264
//  encoding only.
// This register may be written when it is
//  safe to write registers to specify
//  the IDR_PIC_ID in the slice header,
//  when resuming from context switch.
//  It is not necessary to initialize this
//  register at the beginning of encoding.
// The value of this register is updated
//  automatically by the encoder hardware
//  so when it is read at the end of a frame
//  encoding, it will point to the last IDR
//  frame's IDR_PIC_ID.
// Encoder hardware toggles this value between 
//  "0" and "1" at the start of every IDR frame,
//  because the H264 standard specified that if
//  2 conselwtive frames are IDR frames then
//  this value should differ for them.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_IDR_PIC_ID_0                       (0x34)
// IDR_PIC_ID of the last IDR frame.
// This parameter specifies the IDR_PIC_ID
//  of the last IDR frame.
#define LWE2B7_IDR_PIC_ID_0_IDR_PIC_ID_VALUE                       0:0
#define LWE2B7_IDR_PIC_ID_0_IDR_PIC_ID_VALUE_DEFAULT               (0x00000000)

// Offset 0x040
//
// Input Buffer Registers
//
// Add 1 register to control tiling mode for Input Buffer memory.
// Note that TILE_BUFFER_STRIDE is added because when tiling is enabled and input buffer size is
//  less than 1 frame/buffer then it is not easy to generate x,y screen coordinate if the frame
//  is stored non-conselwtively in multiple buffers.
// This register can be used to program the output frame orientation as follows:
//   XY_SWAP | VERT_DIR | HORI_DIR | Output frame orientation
// ----------|----------|----------|------------------------------------------------------
//      0    |    0     |    0     | Normal (same as input frame orientation)
//      0    |    0     |    1     | H flip (mirror on vertical axis)
//      0    |    1     |    0     | V flip (mirror on horizontal axis)
//      0    |    1     |    1     | 180-degree rotation
//      1    |    0     |    0     | XY swap (mirror on 315-degree diagonal)
//      1    |    0     |    1     | 270-degree rotation
//      1    |    1     |    0     | 90-degree rotation
//      1    |    1     |    1     | XY swap and H,V flips (mirror on 45-degree diagonal)
// ---------------------------------------------------------------------------------------
// Notes:
// a. XY swap, if enabled, is performed first prior to H-flip and V-flip.

#define LWE2B7_IB_BUFFER_ADDR_MODE_0                      (0x40)
// Tile Buffer Stride control.
// This bit is used only when tiling is enabled.
#define LWE2B7_IB_BUFFER_ADDR_MODE_0_TILE_BUFFER_STRIDE                    0:0
#define LWE2B7_IB_BUFFER_ADDR_MODE_0_TILE_BUFFER_STRIDE_ENABLE                    (0)    // // The tiling base address will be determined
//  based on the IB0_START_ADDR and
//  IB0_BUFFER_STRIDE * buffer index.
//  This setting may be used when each input
//  buffer size is equal to frame size or more.

#define LWE2B7_IB_BUFFER_ADDR_MODE_0_TILE_BUFFER_STRIDE_DISABLE                   (1)    // // The tiling base address is IB0_START_ADDR,
//  and the Y is callwlated assuming a
//  contiguous IB surface. IB0_BUFFER_STRIDE
//  parameter is not used.
//  This setting must be used when the buffer
//  size is less than the frame size when
//  memory address tiling is enabled.
#define LWE2B7_IB_BUFFER_ADDR_MODE_0_TILE_BUFFER_STRIDE_DEFAULT    (0x00000000)

#define LWE2B7_IB_BUFFER_ADDR_MODE_0_IB_TILE_MODE                  1:1
#define LWE2B7_IB_BUFFER_ADDR_MODE_0_IB_TILE_MODE_LINEAR                  (0)
#define LWE2B7_IB_BUFFER_ADDR_MODE_0_IB_TILE_MODE_TILED                   (1)
#define LWE2B7_IB_BUFFER_ADDR_MODE_0_IB_TILE_MODE_DEFAULT          (0x00000000)

// Horizontal scan direction.
#define LWE2B7_IB_BUFFER_ADDR_MODE_0_HORI_DIR                      2:2
#define LWE2B7_IB_BUFFER_ADDR_MODE_0_HORI_DIR_INCREASE                    (0)    // // Increasing address.

#define LWE2B7_IB_BUFFER_ADDR_MODE_0_HORI_DIR_DECREASE                    (1)    // // Decreasing address.
#define LWE2B7_IB_BUFFER_ADDR_MODE_0_HORI_DIR_DEFAULT              (0x00000000)

// Vertical scan direction.
#define LWE2B7_IB_BUFFER_ADDR_MODE_0_VERT_DIR                      3:3
#define LWE2B7_IB_BUFFER_ADDR_MODE_0_VERT_DIR_INCREASE                    (0)    // // Increasing address.

#define LWE2B7_IB_BUFFER_ADDR_MODE_0_VERT_DIR_DECREASE                    (1)    // // Decreasing address.
#define LWE2B7_IB_BUFFER_ADDR_MODE_0_VERT_DIR_DEFAULT              (0x00000000)

// X and Y swap control.
#define LWE2B7_IB_BUFFER_ADDR_MODE_0_XY_SWAP                       4:4
#define LWE2B7_IB_BUFFER_ADDR_MODE_0_XY_SWAP_DISABLE                      (0)    // // XY swap disabled.

#define LWE2B7_IB_BUFFER_ADDR_MODE_0_XY_SWAP_ENABLE                       (1)    // // XY swap enabled.
#define LWE2B7_IB_BUFFER_ADDR_MODE_0_XY_SWAP_DEFAULT               (0x00000000)


// This is the offset for input buffers that do
//  not start at beginning of the buffers.
//  This is used to specify beginning of the
//  buffer when there is H/V flip.
#define LWE2B7_IB_OFFSET_LUMA_0                   (0x41)
// Input Luma Buffer Offset.
// This specifies the number of bytes to be
//  added to the luma input buffer start
//  address for each input buffer except for
//  the first input buffer of each frame which
//  has offset defined in FIRST_LUMA_OFFSET.
// The sum of the start address and the offset
//  must be 16-byte aligned.  
// For 4:2:0 input format, IB_OFFSET_LUMA 
//  must be multiple of 2.
#define LWE2B7_IB_OFFSET_LUMA_0_LUMA_OFFSET                23:0
#define LWE2B7_IB_OFFSET_LUMA_0_LUMA_OFFSET_DEFAULT                (0x00000000)

// This is the offset for input buffers that do
//  not start at beginning of the buffers.
//  This is used to specify beginning of the
//  buffer when there is H/V flip.
#define LWE2B7_IB_OFFSET_CHROMA_0                 (0x42)
// Input Chroma Buffer Offset.
// This specifies the number of bytes to be
//  added to the chroma input buffer start
//  address for each input buffer except for
//  the first input buffer of each frame which
//  has offset defined in FIRST_CHROMA_OFFSET.
// The sum of the start address and the offset
//  must be 8-byte aligned for planar YUV input
//  data and must be 16-byte aligned for
//  semi-planar YUV input data.
#define LWE2B7_IB_OFFSET_CHROMA_0_CHROMA_OFFSET                    23:0
#define LWE2B7_IB_OFFSET_CHROMA_0_CHROMA_OFFSET_DEFAULT            (0x00000000)

// This is the offset for the buffers that do
//  not start at beginning of input frame.
//  This is used to specify top/left cropping
//  of input frame. This offset must reside
//  within the first input buffer of each frame.
#define LWE2B7_FIRST_IB_OFFSET_LUMA_0                     (0x43)
// Input Luma Buffer Offset.
// This specifies the number of lines to be
//  cropped from the first buffer of each input
//  frame.
// The offset from the luma input buffer start
//  address to the start of the cropped frame is
//  FIRST_LUMA_OFFSET*IB0_LINE_STRIDE_L*16 +
//  LUMA_OFFSET bytes.
// The sum of the start address and the offset
//  must be 16-byte aligned if memory tiling is
//  disabled or 256-byte aligned if memory
//  tiling is enabled.
// For 4:2:0 input format, IB_OFFSET_LUMA 
//  must be multiple of 2.
#define LWE2B7_FIRST_IB_OFFSET_LUMA_0_FIRST_LUMA_OFFSET                    11:0
#define LWE2B7_FIRST_IB_OFFSET_LUMA_0_FIRST_LUMA_OFFSET_DEFAULT    (0x00000000)

// This is the offset for the buffers that do
//  not start at beginning of input frame.
//  This is used to specify top/left cropping
//  of input frame. This offset must reside
//  within the first input buffer of each frame.
#define LWE2B7_FIRST_IB_OFFSET_CHROMA_0                   (0x44)
// Input Chroma Buffer Offset.
// This specifies the number of lines to be
//  cropped from the first buffer of each input
//  frame.
// The offset from the chroma input buffer start
//  address to the start of the cropped frame is
//  FIRST_CHROMA_OFFSET*IB0_LINE_STRIDE_C*8 +
//  CHROMA_OFFSET bytes.
// The sum of the start address and the offset
//  must be 8-byte aligned for planar YUV input
//  data and must be 16-byte aligned for
//  semi-planar YUV input data if memory tiling
//  is disabled or 256-byte aligned if memory
//  tiling is enabled.
#define LWE2B7_FIRST_IB_OFFSET_CHROMA_0_FIRST_CHROMA_OFFSET                11:0
#define LWE2B7_FIRST_IB_OFFSET_CHROMA_0_FIRST_CHROMA_OFFSET_DEFAULT        (0x00000000)

#define LWE2B7_FIRST_IB_V_SIZE_0                  (0x45)
// First Input Buffer Vertical Size.
// This parameter specifies the height of the
//  first input buffer in multiple number of
//  macroblocks.
// This is used instead of IB0_V_SIZE for the
//  first buffer in a frame and should reflect
//  the value in FIRST_OFFSET.
// First input buffer should contains at least
// 3 encoding MBs unless the frame has single input
// buffer.
// FIRST_IB_V_SIZE * Encoding WIDTH >= 3 MB size
#define LWE2B7_FIRST_IB_V_SIZE_0_FIRST_V_SIZE                      8:1
#define LWE2B7_FIRST_IB_V_SIZE_0_FIRST_V_SIZE_DEFAULT              (0x00000000)

#define LWE2B7_IB0_START_ADDR_Y_0                 (0x46)
// This parameter specifies the start address
//  of input buffer set 0 Y plane.
// The sum of this address and the offset
//  must be 16-byte aligned if memory tiling
//  is disabled or 256-byte aligned if memory
//  tiling is enabled.
#define LWE2B7_IB0_START_ADDR_Y_0_IB0_START_ADDR_Y                 31:0
#define LWE2B7_IB0_START_ADDR_Y_0_IB0_START_ADDR_Y_DEFAULT         (0x00000000)

#define LWE2B7_IB0_START_ADDR_U_0                 (0x47)
// Input Buffer Set 0 U Start Address.
// This parameter specifies the start address
//  of input buffer set 0 U plane for planar
//  YUV current data or the start address of
//  input buffer set 0 UV plane for semi-planar
//  YUV current data.
// The sum of this address and the offset
//  must be 8-byte aligned for planar YUV
//  current data or must be 16-byte aligned for
//  semi-planar YUV current data if memory
//  tiling is disabled or 256-byte aligned if
//  memory tiling is enabled.
#define LWE2B7_IB0_START_ADDR_U_0_IB0_START_ADDR_U                 31:0
#define LWE2B7_IB0_START_ADDR_U_0_IB0_START_ADDR_U_DEFAULT         (0x00000000)

#define LWE2B7_IB0_START_ADDR_V_0                 (0x48)
// Input Buffer Set 0 V Start Address.
//  This parameter specifies the start address
//  of input buffer set 0 V plane for planar
//  YUV current data. This parameter is not
//  used for semi-planar YUV current data.
// The sum of this address and the offset
//  must be 8-byte aligned if memory tiling is
//  disabled or 256-byte aligned if memory
//  tiling is enabled.
#define LWE2B7_IB0_START_ADDR_V_0_IB0_START_ADDR_V                 31:0
#define LWE2B7_IB0_START_ADDR_V_0_IB0_START_ADDR_V_DEFAULT         (0x00000000)

// This register specifies the number of buffers
//  in input buffer set 0 count and the size of
//  each buffer.
#define LWE2B7_IB0_SIZE_0                 (0x49)
// Input Buffer Set 0 Count.
//  This parameter specifies the the number of
//  buffers in the input buffer set.
//  The encoder has 8-deep input buffer FIFO to
//  receive up to 8 input buffers so if MPE
//  input buffer FIFO becomes full, it will
//  stall its input FIFO or assert a Buffer
//  Rejected interrupt.
//  If the programmed value is zero,
//  zero means to use the natural FIFO size.
// Note: a buffer is not popped from the input
//  buffer until the encoder has consumed the
//  buffer and it is safe for something
//  to overwrite the buffer's memory location.
#define LWE2B7_IB0_SIZE_0_IB0_COUNT                7:0
#define LWE2B7_IB0_SIZE_0_IB0_COUNT_DEFAULT                        (0x00000000)

// Input Buffer Set 0 Vertical Size.
// This parameter specifies the height of the
//  each input buffer in multiple number of
//  macroblocks except for the first input
//  buffer of each frame which is specified
//  by FIRST_IB_V_SIZE.
#define LWE2B7_IB0_SIZE_0_IB0_V_SIZE                       24:17
#define LWE2B7_IB0_SIZE_0_IB0_V_SIZE_DEFAULT                       (0x00000000)

// This register specifies both luma and chroma
//  line stride for input buffers.
// This register specifies the input buffers
//  line stride in planar (THREE_PLANE), or
//  semi-planar (TWO_PLANE) modes. The input
//  frame mode type, THREE_PLANE (Y/U/V) or
//  TWO_PLANE (Y/UV), is determined by LWR_SURF
//  parameter in VOL_CTRL register.
#define LWE2B7_IB0_LINE_STRIDE_0                  (0x4a)
// Input Buffer Set 0 Luma Line Stride.
// This parameter specifies line stride for
//  Y input plane in multiple of 16-bytes,
//  in both planar and semi-planar modes.
//  If memory tiling is enabled, the line stride
//  is specified in multiple of 64-bytes, and
//  bits [2:1] must be programmed to 2'b00.
//  The actual line stride is callwlated by
//  multiplying this field by 16.
#define LWE2B7_IB0_LINE_STRIDE_0_IB0_LINE_STRIDE_L                 12:1
#define LWE2B7_IB0_LINE_STRIDE_0_IB0_LINE_STRIDE_L_DEFAULT         (0x00000000)

// Input Buffer Set 0 Chroma Line Stride.
// In planar mode, this parameter specifies the
//  line stride for the U and V input planes,
//  in multiple of 8-bytes.
//  In semi-planar  mode, this parameter
//  specifies the line stride for the UV
//  input planes, in multiple of 16-bytes,
//  and bit [16] must be programmed to 1'b0.
//  If memory tiling is enabled, the line stride
//  is specified in multiple of 64-bytes, and
//  bits [28:16] must be programmed to 3'b000.
//  The actual line stride is callwlated by
//  multiplying this field by 8.
#define LWE2B7_IB0_LINE_STRIDE_0_IB0_LINE_STRIDE_C                 28:16
#define LWE2B7_IB0_LINE_STRIDE_0_IB0_LINE_STRIDE_C_DEFAULT         (0x00000000)

// This register specifies luma buffer stride
//  for input buffers.
#define LWE2B7_IB0_BUFFER_STRIDE_LUMA_0                   (0x4b)
// Input Buffer Set 0 Luma Buffer Stride.
// This parameter specifies luma buffer stride
//  in multiple of 16-byte words.
//  If memory tiling is enabled, this needs to
//  be multiple of 64-byte words.  To take full 
//  advantage of memory tiling, this field is 
//  recommended to be multiple of 256-byte words.
#define LWE2B7_IB0_BUFFER_STRIDE_LUMA_0_IB0_BUFFER_STRIDE_L                31:4
#define LWE2B7_IB0_BUFFER_STRIDE_LUMA_0_IB0_BUFFER_STRIDE_L_DEFAULT        (0x00000000)

// This register specifies chroma buffer stride
//  for input buffers.
#define LWE2B7_IB0_BUFFER_STRIDE_CHROMA_0                 (0x4c)
// Input Buffer Set 0 Chroma Buffer Stride.
// This parameter specifies chroma buffer stride
//  in multiple of 8-byte words for planar YUV
//  current data or in multiple of 16-byte words
//  for semi-planar YUV current data.
//  If memory tiling is enabled, this needs to
//  be multiple of 64-byte words.  To take full 
//  advantage of memory tiling, this field is 
//  recommended to be multiple of 256-byte words.
#define LWE2B7_IB0_BUFFER_STRIDE_CHROMA_0_IB0_BUFFER_STRIDE_C                      31:3
#define LWE2B7_IB0_BUFFER_STRIDE_CHROMA_0_IB0_BUFFER_STRIDE_C_DEFAULT      (0x00000000)

// Offset 0x050
//
// Reference/reconstructed buffer management registers.
//
// Add 1 register to control tiling mode for Reference memory.

#define LWE2B7_REF_BUFFER_ADDR_MODE_0                     (0x50)
#define LWE2B7_REF_BUFFER_ADDR_MODE_0_REF_TILE_MODE                0:0
#define LWE2B7_REF_BUFFER_ADDR_MODE_0_REF_TILE_MODE_LINEAR                        (0)
#define LWE2B7_REF_BUFFER_ADDR_MODE_0_REF_TILE_MODE_TILED                 (1)
#define LWE2B7_REF_BUFFER_ADDR_MODE_0_REF_TILE_MODE_DEFAULT        (0x00000000)

// This register specifies the starting
//  address in the frame buffer to store/fetch
//  the reference/reconstructed Y data.
// This address is used to compute reference
//  frame address to the Motion Compensation
//  (differential) modules which is also the
//  reconstructed frame address to the Motion
//  Compensation (decoder) module.
// This buffer memory is used as a loop memory.
#define LWE2B7_REF_Y_START_ADDR_0                 (0x51)
// Start Address for Reference Y Buffer.
// This is the 16-byte aligned start address
//  of the reference Y buffer.
//  If memory tiling is enabled, this needs to
//  be 256-byte aligned.
#define LWE2B7_REF_Y_START_ADDR_0_REF_Y_START_ADDR                 31:4
#define LWE2B7_REF_Y_START_ADDR_0_REF_Y_START_ADDR_DEFAULT         (0x00000000)

// This register specifies the starting
//  address in the frame buffer to store/fetch
//  the reference/reconstructed U data for
//  3-plane (planar) YUV reference data OR
//  UV data for 2-plane (semi-planar) YUV
//  reference data.
// This address is used to compute reference
//  frame address to the Motion Compensation
//  (differential) modules which is also the
//  reconstructed frame address to the Motion
//  Compensation (decoder) module.
// This buffer memory is used as a loop memory.
#define LWE2B7_REF_U_START_ADDR_0                 (0x52)
// Start Address for Reference U or UV Buffer.
// This is the 16-byte aligned start address
//  of the reference U buffer for planar YUV
//  reference data OR the reference UV buffer
//  for semi-planar YUV reference data.
//  If memory tiling is enabled, this needs to
//  be 256-byte aligned.
#define LWE2B7_REF_U_START_ADDR_0_REF_U_START_ADDR                 31:4
#define LWE2B7_REF_U_START_ADDR_0_REF_U_START_ADDR_DEFAULT         (0x00000000)

// This register specifies the starting
//  address in the frame buffer to store/fetch
//  the reference/reconstructed V data for
//  3-plane (planar) YUV reference data. This
//  register is not used for 2-plane
//   (semi-planar) YUV reference data.
// This address is used to compute reference
//  frame address to the Motion Compensation
//  (differential) modules which is also the
//  reconstructed frame address to the Motion
//  Compensation (decoder) module.
// This buffer memory is used as a loop memory.
#define LWE2B7_REF_V_START_ADDR_0                 (0x53)
// Start Address for Reference V Buffer.
// This is the 16-byte aligned start address
//  of the reference V buffer for planar YUV
//  reference frame. This is not used for
//  2-plane (semi-planar) YUV reference data.
//  If memory tiling is enabled, this needs to
//  be 256-byte aligned.
#define LWE2B7_REF_V_START_ADDR_0_REF_V_START_ADDR                 31:4
#define LWE2B7_REF_V_START_ADDR_0_REF_V_START_ADDR_DEFAULT         (0x00000000)

// This register specifies the reference/
//  reconstructed frame line stride in planar
//  (THREE_PLANE), or semi-planar (TWO_PLANE)
//  modes. The reference frame mode type,
//  THREE_PLANE (Y/U/V) or TWO_PLANE (Y/UV), is
//  determined by REF_SURF parameter in VOL_CTRL
//  register.
#define LWE2B7_REF_STRIDE_0                       (0x54)
// Y Line Stride.
// This parameter specifies the line stride for
//  the Y reference plane in multiple of
//  16-bytes, in both planar and semi-planar
//  modes.
//  If memory tiling is enabled, the line stride
//  is specified in multiple of 64-bytes, and
//  bits [2:1] must be programmed to 2'b00.
//  The actual line stride is callwlated by
//  multiplying this field by 16.
#define LWE2B7_REF_STRIDE_0_REF_Y_STRIDE                   12:1
#define LWE2B7_REF_STRIDE_0_REF_Y_STRIDE_DEFAULT                   (0x00000000)

// U, V Line Stride.
// In both planar and semi-planar modes, this
//  parameter specifies the line stride for the
//  U and V or UV reference planes, in multiple
//  of 16-bytes.
//  If memory tiling is enabled, the line stride
//  is specified in multiple of 64-bytes, and
//  bits [18:17] must be programmed to 2'b00.
//  The actual line stride is callwlated by
//  multiplying this field by 16.
#define LWE2B7_REF_STRIDE_0_REF_UV_STRIDE                  28:17
#define LWE2B7_REF_STRIDE_0_REF_UV_STRIDE_DEFAULT                  (0x00000000)

// This register specifies the number of macro-
//  blocks in the Y-direction (height) that are
//  reserved for the reference/reconstructed
//  frame from the starting address defined
//  in the reference buffer start address
//  registers. Hardware manages this memory
//  between the reference and the next
//  reconstructed frames.
//  This length is used by the hardware to
//  loop to the starting address. This length
//  must be at least equal to the height of the
//  frame in macroblocks + 6.
//  This length must be multiple of frames
//  (min 2 frames) if reconstructed frames
//  are used for display preview.
#define LWE2B7_REF_BUFFER_LEN_0                   (0x55)
// Length of Y, U and V Buffers
//  This is the length of the Y, U & V buffers
//  in multiple of macroblock rows.
#define LWE2B7_REF_BUFFER_LEN_0_REF_BUFFER_LEN                     8:0
#define LWE2B7_REF_BUFFER_LEN_0_REF_BUFFER_LEN_DEFAULT             (0x00000000)

// This register may be written when it is
//  safe to write registers to specify
//  initial index value for reconstructed frame
//  buffer.
//  This buffer index is sent to display in
//  case reconstructed frame is used for
//  encoding preview (RECON_PREVIEW_ENABLE
//  is set to ENABLE).
// The value of this register is updated
//  automatically by the encoder hardware
//  so when it is read at the end of a frame
//  encoding, it will point to the next frame
//  buffer index.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_REF_BUFFER_IDX_0                   (0x56)
// Reconstructed Buffer Index.
// This parameter specifies the buffer
//  index for reconstructed frame buffer.
#define LWE2B7_REF_BUFFER_IDX_0_REF_BUFFER_IDX                     1:0
#define LWE2B7_REF_BUFFER_IDX_0_REF_BUFFER_IDX_DEFAULT             (0x00000000)

// This register may be written when it is
//  safe to write registers to specify
//  the address to write the first reconstructed
//  frame in term of mb-row position with
//  respect to reference memory buffer start
//  addresses.
// The value of this register is updated
//  automatically by the encoder hardware
//  so when it is read at the end of a frame
//  encoding, it will point to the next frame
//  buffer position.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_REF_WR_MBROW_0                     (0x57)
// Reconstructed Frame Write MB row position.
// This is the mb-row position to write the
//  first reference frame with respect to the
//  reconstructed buffer start addresses.
// This position MUST be within the reference/
//  reconstructed buffer.
//  This position must also be exactly 1 frame
//  later than the REF_RD_ADDR considering
//  that the reference/reconstructed buffer
//  is a loop memory.
// Subsequent reconstructed frame position is
//  callwlated by the hardware.
#define LWE2B7_REF_WR_MBROW_0_REF_WR_MBROW                 8:0
#define LWE2B7_REF_WR_MBROW_0_REF_WR_MBROW_DEFAULT                 (0x00000000)

// This register may be written when it is
//  safe to write registers to specify
//  the address to read the first reference
//  frame in term of mb-row position with
//  respect to reference memory buffer start
//  addresses.
//  This is needed only if the first frame
//  to be encoded is a P-Frame.
// The value of this register is updated
//  automatically by the encoder hardware
//  so when it is read at the end of a frame
//  encoding, it will point to the next frame
//  buffer position.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_REF_RD_MBROW_0                     (0x58)
// Reference Frame Read MB row position.
// This is the mb-row position to read the
//  first reference frame with respect to the
//  reference buffer start addresses.
// This position MUST be within the reference/
//  reconstructed buffer.
//  This position must also be exactly 1 frame
//  ahead from the REF_WR_ADDR considering
//  that the reference/reconstructed buffer
//  is a loop memory.
// Subsequent reference frame position is
//  callwlated by the hardware.
#define LWE2B7_REF_RD_MBROW_0_REF_RD_MBROW                 8:0
#define LWE2B7_REF_RD_MBROW_0_REF_RD_MBROW_DEFAULT                 (0x00000000)

// Offset 0x060








// 0x060-0x067
//
// Other row buffers.
//
// New register for H.264 Intra Prediction.

// This register specifies the starting address 
//  of the intra pred pixel memory to store the
//  last pixel row of a MB, for use in the intra
//  prediction of the next row. 
// Software must allocate 32-byte per
//  macroblock for one macroblock row for this
//  buffer. The 32 bytes for each MB are
//  allocated as 16 lower bytes for Y pixels,
//  followed by 8 bytes for U pixels,
//  followed by 8 bytes for V pixels.  
// This register is programmed only once.
#define LWE2B7_IPRED_ROW_ADDR_0                   (0x68)
// Start Address of I-Pred MB Row Buffer. 
// This is the 16-byte aligned address to
//  store the start address of I-pred row
//  buffer.
//  Subsequent addresses will be callwlated by
//  the hardware.
#define LWE2B7_IPRED_ROW_ADDR_0_IPRED_ROW_ADDR                     31:4
#define LWE2B7_IPRED_ROW_ADDR_0_IPRED_ROW_ADDR_DEFAULT             (0x00000000)

// New register for H264 Deblocker parameter buffer.

// This register specifies the starting address 
//  of the deblocker parameter memory to store
//  16 bytes per MB for one MB row.
// Software must allocate 16-byte per
//  macroblock for one macroblock row for this
//  buffer. 
// This register is programmed only once.
#define LWE2B7_DBLK_PARAM_ADDR_0                  (0x69)
// Start Address of Deblocker Parameter Buffer. 
// This is the 16-byte aligned address to
//  store the start address of deblocker
//  parameter row buffer.
// Subsequent addresses will be callwlated by
// the hardware.
#define LWE2B7_DBLK_PARAM_ADDR_0_DBLK_PARAM_ADDR                   31:4
#define LWE2B7_DBLK_PARAM_ADDR_0_DBLK_PARAM_ADDR_DEFAULT           (0x00000000)

// This bit is effective only for MPEG4/H.263
//  encoding and is not used for H.264 encoding.
// This register specifies the starting address
//  of the frame buffer memory to store the
//  AC/DC coefficients for the Y, U and V data.
//  The AC/DC prediction needs to store one row
//  of coefficients in the memory. This memory
//  is used to fetch and store the AC/DC
//  coefficients for the AC/DC prediction.
//  Software must allocate 64-byte per
//  macroblock for 1 macroblock row for this
//  buffer. E.g. for CIF resolution, there are
//  22 macroblocks per row, so 22 * 64 = 1408
//  bytes need to be allocated.
// This register is programmed only once.
#define LWE2B7_ACDC_ADDR_0                        (0x6a)
// Start Address of AC/DC Coefficient Buffer.
//  This parameter specifies the 16-byte aligned
//  address to store the first AC/DC
//  coefficient. Subsequent addresses will be
//  callwlated by the hardware.
#define LWE2B7_ACDC_ADDR_0_ACDC_ADDR                       31:4
#define LWE2B7_ACDC_ADDR_0_ACDC_ADDR_DEFAULT                       (0x00000000)

// Offset 0x070
//
// Video Frame Registers.
//

// This register specifies control bits
//  needed for encoding a frame.
#define LWE2B7_FRAME_CTRL_0                       (0x70)
// Four Motion Vector Enable.
// This bit is applicable for MPEG4/H.263
//  encoding.
// This bit is not applicable for APxx H.264
//  encoding which uses MV_MODE field of
//  MOT_SEARCH_CTRL_REG instead.
// This bit specifies either 1 or 4 motion
//  vectors encoding per macroblock. This
//  dictates the search algorithm in the motion
//  estimation engine. 
#define LWE2B7_FRAME_CTRL_0_MV4ENABLE                      0:0
#define LWE2B7_FRAME_CTRL_0_MV4ENABLE_MV1                 (0)    // // One motion vector (16x16) per macroblock.

#define LWE2B7_FRAME_CTRL_0_MV4ENABLE_MV4                 (1)    // // Allow 4 motion vectors (16x16 or 4x4) per
//  macroblock. This setting is recommended
//  for better video encoding quality.
#define LWE2B7_FRAME_CTRL_0_MV4ENABLE_DEFAULT                      (0x00000001)

// Motion Estimation Start Vector Control.
// This parameter specifies how the start
//  vector is determined for the motion
//  estimation engine.
#define LWE2B7_FRAME_CTRL_0_MOT_SEARCH_START_VEC                   2:1
#define LWE2B7_FRAME_CTRL_0_MOT_SEARCH_START_VEC_START1                   (1)    // // 1 prediction point is used as the start
//  vector in the motion estimation engine.

#define LWE2B7_FRAME_CTRL_0_MOT_SEARCH_START_VEC_START2                   (2)    // // 2 prediction points are used as the start
//  vectors in the motion estimation engine.
//  This setting is recommended for better
//  video encoding quality.
#define LWE2B7_FRAME_CTRL_0_MOT_SEARCH_START_VEC_DEFAULT           (0x00000002)

// Half Pel Search Control.
// This bit enables/disables the half-pel
//  search in the motion estimation engine.
#define LWE2B7_FRAME_CTRL_0_HALF_PEL_MV_ENABLE                     3:3
#define LWE2B7_FRAME_CTRL_0_HALF_PEL_MV_ENABLE_DISABLE                    (0)    // // Half-pel search is disabled.

#define LWE2B7_FRAME_CTRL_0_HALF_PEL_MV_ENABLE_ENABLE                     (1)    // // Half-pel search is enabled.
//  This setting is recommended for better
//  video encoding quality.
#define LWE2B7_FRAME_CTRL_0_HALF_PEL_MV_ENABLE_DEFAULT             (0x00000001)

// Packet Count select
// This bit is applicable for MPEG4/H.263
//  encoding only.
//  For H.264 encoding, the control for actual
//  packetization is specified in
//  PACKET_CTRL_H264 register.
// In APxx, if ME_BYPASS is set to ENABLE,
//  this bit is ignored.
// This bit specifies how the size of the
//  encoded bitstream packet is determined.
#define LWE2B7_FRAME_CTRL_0_PACKET_CNT_SRC                 4:4
#define LWE2B7_FRAME_CTRL_0_PACKET_CNT_SRC_BITS                   (0)    // // The size of encoded bitstream packet is
//  determined by the number of bits in the
//  bitstream as specified by PACKET_COUNT
//  field in PACKET_HEC register.

#define LWE2B7_FRAME_CTRL_0_PACKET_CNT_SRC_MBLK                   (1)    // // The size of encoded bitstream packet is
//  determined by the number of macroblocks
//  specified by PACKET_COUNT field in
//  PACKET_HEC register.
#define LWE2B7_FRAME_CTRL_0_PACKET_CNT_SRC_DEFAULT                 (0x00000000)

// AC Prediction Control.
// This bit is applicable for MPEG4/H.263
//  encoding only and is not applicable for
//  H.264 encoding.
// This parameter controls the AC prediction
//  processing for MPEG4/H.263 encoding.
#define LWE2B7_FRAME_CTRL_0_ACPRED                 6:5
#define LWE2B7_FRAME_CTRL_0_ACPRED_OFF                    (0)    // // AC prediction is off (disabled).

#define LWE2B7_FRAME_CTRL_0_ACPRED_ON                     (1)    // // AC prediction is always on (enabled).

#define LWE2B7_FRAME_CTRL_0_ACPRED_DYN_ON                 (2)    // // AC prediction is dynamically triggered.
#define LWE2B7_FRAME_CTRL_0_ACPRED_DEFAULT                         (0x00000000)

// Macroblock Reprocess Control
// This bit is applicable for MPEG4/H.263
//  encoding only and is not applicable for
//  H.264 encoding.
// This bit specifies whether the macroblock at
//  a packet boundary needs to be reprocessed.
//  If the Data Partition and the Resync Marker
//  are enabled, the max packet size should be
//  strictly followed. Note that hardware does
//  not know whether the size exceeds the max
//  packet size (PACKET_COUNT of PACKET_HEC
//  register) until it fully process the
//  macroblock. So, if the size exceeds max
//  packet it will start the new packet with
//  the current macroblock. This will affect the
//  AC/DC prediction and motion vectors since
//  they are differentially coded with their
//  neighbors in the packet. The macroblock
//  needs to be reprocessed in such a case. 
#define LWE2B7_FRAME_CTRL_0_REPROCESS_ENABLE                       7:7
#define LWE2B7_FRAME_CTRL_0_REPROCESS_ENABLE_DISABLE                      (0)    // // If max packet size is exceeded, switch off
//  AC/DC prediction and force Intra DC
//  Threshold value to "0".

#define LWE2B7_FRAME_CTRL_0_REPROCESS_ENABLE_ENABLE                       (1)    // // If max packet size is exceeded, reprocess
//  the AC/DC prediction.
#define LWE2B7_FRAME_CTRL_0_REPROCESS_ENABLE_DEFAULT               (0x00000000)

// Motion Estimation Cache Optimization Disable
// This bit controls optimization in motion
//  estimation reference cache read when the
//  first set of macroblocks are forced to be
//  intra by the intra refresh module.
#define LWE2B7_FRAME_CTRL_0_MOT_EST_CACHE_OPT_DIS                  8:8
#define LWE2B7_FRAME_CTRL_0_MOT_EST_CACHE_OPT_DIS_OPT_ENABLE                      (0)    // // Skip fetching of reference data if first
//  macroblocks are forced to be intra by intra
//  refresh module to reduce bandwidth/power.
//  This is recommended for normal operation.

#define LWE2B7_FRAME_CTRL_0_MOT_EST_CACHE_OPT_DIS_OPT_DISABLE                     (1)    // // Force motion estimation cache to fetch
//  reference data even if the first macroblocks
//  are forced to be intra by the intra refresh
//  module.
#define LWE2B7_FRAME_CTRL_0_MOT_EST_CACHE_OPT_DIS_DEFAULT          (0x00000000)

// Quarter Pel Search Control
// This bit is applicable for H.264 encoding
//  only.
// This bit must be set to DISABLE for
//  MPEG4/H.263 encoding.
// This bit controls quarter-pel search in
//  the motion estimation engine.
#define LWE2B7_FRAME_CTRL_0_QUARTER_PEL_MV_ENABLE                  9:9
#define LWE2B7_FRAME_CTRL_0_QUARTER_PEL_MV_ENABLE_DISABLE                 (0)    // // Quarter-pel search is disabled.

#define LWE2B7_FRAME_CTRL_0_QUARTER_PEL_MV_ENABLE_ENABLE                  (1)    // // Quarter-pel search is enabled.
//  This setting is recommended for better
//  H.264 video encoding quality.
#define LWE2B7_FRAME_CTRL_0_QUARTER_PEL_MV_ENABLE_DEFAULT          (0x00000000)

// Enable Frame Preview from Recon Buffer
// This bit enables/disables display preview
//  of encoded frames from the reconstructed
//  (reference) frame buffer. For this to work
//  reconstructed buffer must be exact multiple
//  of frames. Software must also make sure
//  that reconstructed (reference) data format
//  is stored in a format that is supported
//  by display. In AP15, display controller
//  does not support YUV semi-planar format.
#define LWE2B7_FRAME_CTRL_0_RECON_PREVIEW_ENABLE                   10:10
#define LWE2B7_FRAME_CTRL_0_RECON_PREVIEW_ENABLE_DISABLE                  (0)    // // Display preview from reconstructed frame
//  buffer is disabled.

#define LWE2B7_FRAME_CTRL_0_RECON_PREVIEW_ENABLE_ENABLE                   (1)    // // Display preview from reconstructed frame
//  buffer is enabled.
#define LWE2B7_FRAME_CTRL_0_RECON_PREVIEW_ENABLE_DEFAULT           (0x00000000)

// Enable Gray Color Encoding.
// This bit controls grey color encoding by
//  encoding luma plane only and forcing chroma
//  plane data to 128.
// This feature is lwrrently NOT implemented.
#define LWE2B7_FRAME_CTRL_0_GRAY_ENCODING                  11:11
#define LWE2B7_FRAME_CTRL_0_GRAY_ENCODING_DISABLE                 (0)    // // Gray color encoding is disabled.

#define LWE2B7_FRAME_CTRL_0_GRAY_ENCODING_ENABLE                  (1)    // // Gray color encoding is enabled.
#define LWE2B7_FRAME_CTRL_0_GRAY_ENCODING_DEFAULT                  (0x00000000)

// Multiple Reference Frame Mode (H264 only),
// This parameter is applicable for H.264
//  encoding only and not used for MPEG4/H.263
//  encoding.
// This feature is lwrrently NOT implemented.
// This parameter controls the selection of
//  reference frame for multiple reference
//  frame encoding.
// If NUM_REF_FRAME is programmed as N frames
//  then the N reference frames will be
//  determined as follows:
#define LWE2B7_FRAME_CTRL_0_MULT_REF_MODE                  17:13
#define LWE2B7_FRAME_CTRL_0_MULT_REF_MODE_SHORT                   (0)    // // N short term conselwtive last reference
//  frames are used for encoding.

#define LWE2B7_FRAME_CTRL_0_MULT_REF_MODE_LONG_ONE                        (1)    // // One long term reference frame is used plus
//  N-1 short term conselwtive last reference
//  frames are used for encoding.

#define LWE2B7_FRAME_CTRL_0_MULT_REF_MODE_LONG_AUTO                       (2)    // // One auto detect long term reference frame is
//  used plus N-1 short term conselwtive last
//  reference frames are used for encoding.
#define LWE2B7_FRAME_CTRL_0_MULT_REF_MODE_DEFAULT                  (0x00000000)

// Deblocking control.
// This parameter is applicable for H.264
//  encoding only and not used for MPEG4/H.263
//  encoding.
// This parameter controls deblocking process
//  for H.264 encoding.
#define LWE2B7_FRAME_CTRL_0_DISABLE_DEBLOCKING_FILTER_IDC                  19:18
#define LWE2B7_FRAME_CTRL_0_DISABLE_DEBLOCKING_FILTER_IDC_ENABLE_ACROSS_SLICE                     (0)    // // Enable deblocking and filter across slice
//  boundaries.

#define LWE2B7_FRAME_CTRL_0_DISABLE_DEBLOCKING_FILTER_IDC_DISABLE                 (1)    // // Disable deblocking

#define LWE2B7_FRAME_CTRL_0_DISABLE_DEBLOCKING_FILTER_IDC_ENABLE_NOT_ACROSS_SLICE                 (2)    // // Enable deblocking but don't filter across
//  slice boundaries.
#define LWE2B7_FRAME_CTRL_0_DISABLE_DEBLOCKING_FILTER_IDC_DEFAULT  (0x00000000)

// Logic is added in AP15 to do power saving for all I Frames encoding by not writing
// reconstructed/reference frame to memory.

// This register suggests the I-to-P frame
//  ratio.
#define LWE2B7_FRAME_TYPE_0                       (0x71)
// P-Frame to I-Frame Ratio.
// This parameter specifies the number of
//  P frames between any two I frames when
//  P_FRAME_INTERVAL_DIS is set to ENABLE.
//  Setting this parameter to 0 with
//  P_FRAME_INTERVAL_DIS = ENABLE will result
//  in all frames encoded as I frames 
// 
#define LWE2B7_FRAME_TYPE_0_P_FRAME_INTERVAL                       15:0
#define LWE2B7_FRAME_TYPE_0_P_FRAME_INTERVAL_DEFAULT               (0x00000000)

// Pattern Length.
// In the future projects, pattern based frame skipping inside 
// MPE will be deprecated. So to maintain compatibility 
// with future chips, SW must not do pattern based frame
// skipping in MPE. Instead, SW must do pattern based 
// frame skipping in driver. So PATTERN_LEN register
// field (also FRAME_PATTERN register) must be 
// programmed in AP20 so as not to do frame 
// skipping in MPE.
// This parameter sets the length of the pattern
//  in the FRAME_PATTERN register below.
//  Valid value is 1 to 32.
#define LWE2B7_FRAME_TYPE_0_PATTERN_LEN                    21:16
#define LWE2B7_FRAME_TYPE_0_PATTERN_LEN_DEFAULT                    (0x0000001e)

// P-Frame to I-Frame Ratio Disable.
// This bit is used to enable/disable the I
//  frame insertion based on the P frame to
//  I frame ratio.
#define LWE2B7_FRAME_TYPE_0_P_FRAME_INTERVAL_DIS                   30:30
#define LWE2B7_FRAME_TYPE_0_P_FRAME_INTERVAL_DIS_ENABLE                   (0)    // // I frame is inserted once the P frame to I
//  frame ratio is met irrespective of whether
//  the Intra Refresh is enabled or disabled.

#define LWE2B7_FRAME_TYPE_0_P_FRAME_INTERVAL_DIS_DISABLE                  (1)    // // I frame insertion is disabled. Only the
//  first frame is encoded as an I frame. The
//  rest of the frame encoding (macroblocks) is
//  decided by the Intra Refresh programming. A
//  frame can still be forced to be encoded as
//  intra by using the FORCE_I_FRAME bit.
#define LWE2B7_FRAME_TYPE_0_P_FRAME_INTERVAL_DIS_DEFAULT           (0x00000000)


// In the future projects, pattern based frame skipping inside 
// MPE will be deprecated. So to maintain compatibility 
// with future chips, SW must not do pattern based frame
// skipping in MPE. Instead, SW must do pattern based 
// frame skipping in driver. So FRAME_PATTERN register
// must be programmed in AP20 so as not to do frame 
// skipping in MPE.
// This register is used to decrease the frame
//  rate of the encoder. The incoming data
//  stream and the encoder frame rates can be
//  different. This register specifies the
//  frames that should be dropped in the
//  incoming data stream.
#define LWE2B7_FRAME_PATTERN_0                    (0x72)
// Frame Drop Pattern.
//  This parameter specifies the frames to be
//  dropped in the incoming data stream.
//  Each bit in the register determines if a
//  frame is dropped (if 0) or encoded (if 1).
//  Bit 0 correspond to the first input frame
//  in the pattern. The number of frame in the
//  pattern is determined by PATTERN_LEN field
//  in FRAME_TYPE register.
#define LWE2B7_FRAME_PATTERN_0_PATTERN                     31:0
#define LWE2B7_FRAME_PATTERN_0_PATTERN_DEFAULT                     (0xffffffff)

// This register may be written when it is
//  safe to write reigsters to specify
//  P-Frame count after last I-frame and to
//  specify initial value of the position
//  index of FRAME_PATTERN register for the
//  first incoming stream.
// The value of this register is updated
//  automatically by the encoder hardware
//  so when it is read at the end of a frame
//  encoding, it will point to the updated
//  P-Frame count and the next frame pattern
//  index.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_FRAME_INDEX_0                      (0x73)
// P-Frame Index.
// This parameter specifies the number of P
//  frame after the last I frame.
#define LWE2B7_FRAME_INDEX_0_P_FRAME_INDEX                 15:0
#define LWE2B7_FRAME_INDEX_0_P_FRAME_INDEX_DEFAULT                 (0x00000000)

// Frame Pattern Index.
// This parameter specifies the index
//  (position) of the FRAME_PATTERN.
// Valid value is from 0 to PATTERN_LEN-1.
#define LWE2B7_FRAME_INDEX_0_PATTERN_INDEX                 28:24
#define LWE2B7_FRAME_INDEX_0_PATTERN_INDEX_DEFAULT                 (0x00000000)

// ENC_FRAME_NUM is reduced from 31 bits to 15 bits.

// This register may be written when it is
//  safe to write registers to specify
//  initial value of the internal frame number
//  counter. This frame number is what will
//  be reported in the chunk header of the
//  output bitstream. This header is not part
//  of the encoded bitstream and it contains
//  information that is used by CPU to be able
//  to process the real encoded bitstream.
// The value of this register is updated
//  automatically by the encoder hardware
//  at beginning of each frame encoding.
//  This parameter is incremented even
//  for frames skipped by MPE hardware. It is
//  not incremented for frames skipped by host
//  therefore host should do the incrementation
//  in this case.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
//  When context is resumed, the value read
//  from this register may also be used to
#define LWE2B7_ENC_FRAME_NUM_0                    (0x74)
// Encoder Frame Number.
// This parameter specifies the frame number of
//  the frames to be encoded.
#define LWE2B7_ENC_FRAME_NUM_0_ENC_FRAME_NUM                       15:0
#define LWE2B7_ENC_FRAME_NUM_0_ENC_FRAME_NUM_DEFAULT               (0x00000000)

// This bit indicates if the Link List Buffers
//  are full. If the Link List Buffers are full
//  encoder will skip frames until there is
//  enough space.
#define LWE2B7_ENC_FRAME_NUM_0_LINK_BUF_FULL                       30:30
#define LWE2B7_ENC_FRAME_NUM_0_LINK_BUF_FULL_DEFAULT               (0x00000000)

// Frame Status.
// This bit specifies whether the last frame
//  was skipped or encoded.
#define LWE2B7_ENC_FRAME_NUM_0_FRAME_SKIPPED                       31:31
#define LWE2B7_ENC_FRAME_NUM_0_FRAME_SKIPPED_ENCODED                      (0)    // // Last frame was encoded by the hardware.

#define LWE2B7_ENC_FRAME_NUM_0_FRAME_SKIPPED_SKIPPED                      (1)    // // Last frame was skipped by the hardware.
#define LWE2B7_ENC_FRAME_NUM_0_FRAME_SKIPPED_DEFAULT               (0x00000000)

// This register is added for H.264 encoding.

// This register is applicable only for H.264
//  encoding and not used for MPEG4/H.263
//  encoding.
// This register may be written when it is
//  safe to write registers to specify
//  initial value of the internal frame number
//  counter. This frame number is the value
//  that will be encoded and will be part of
//  the encoded bitstream.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_FRAME_NUM_0                        (0x75)
// Frame Number of the encoded bitstream.
// This parameter specifies the frame number of
//  the next encoded frame.
// The value of this register is updated
//  automatically by the encoder hardware
//  at the beginning of each frame encoding.
//  This parameter is incremented even
//  for frames skipped by MPE hardware if
//  GAPS_IN_FRAME_NUM_VALUE_ALLOWED is set to 1.
//  It is however reset for I-frame and it
//  wraps around based on value of
//  LOG2_MAX_FRAME_NUM_MINUS4.
#define LWE2B7_FRAME_NUM_0_FRAME_NUM                       15:0
#define LWE2B7_FRAME_NUM_0_FRAME_NUM_DEFAULT                       (0x00000000)

// Frame Number of encoded bitstream for previous frame.
// This parameter specifies the frame number that goes into
// bitstream for the previous encoded frame
#define LWE2B7_FRAME_NUM_0_PREV_FRAME_NUM                  31:16
#define LWE2B7_FRAME_NUM_0_PREV_FRAME_NUM_DEFAULT                  (0x00000000)

// This register is applicable only for H.264
//  encoding and not used for MPEG4/H.263
//  encoding.
// This register may be written when it is
//  safe to write registers to specify
//  initial value of the internal frame number
//  counter. This frame number is the value
//  that will be encoded and will be part of
//  the encoded bitstream.
// The value of this register is updated
//  automatically by the encoder hardware
//  at the end of each frame encoding.
#define LWE2B7_FRAME_NUM_GOP_0                    (0x76)
// Frame Number of the CBR RC.
// This parameter specifies the frame number of
//  the current encoded frame in
//  in the CBR RC, that is used to 
//  track the ptr in the GOP.
// For doing context switching, for the very first frame of a context,
// this register must be programmed to P_FRAME_INTERVAL
// when GOP_GLAG is CLOSED or to GOP_LENGTH - 1 when GOP_FLAG is OPEN.
#define LWE2B7_FRAME_NUM_GOP_0_FRAME_NUM_GOP                       15:0
#define LWE2B7_FRAME_NUM_GOP_0_FRAME_NUM_GOP_DEFAULT               (0x00000000)








// 0x077-0x07E
//
// Slice Group Registers.
//

// This register specifies number of data
//  partitions for MPEG4 encoding and controls
//  FMO (Flexible Macroblock Order) encoding
//  for H.264 encoding.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
#define LWE2B7_NUM_SLICE_GROUPS_0                 (0x7f)
// Number of Slice Group - 1.
// This parameter is applicable for BOTH
//  MPEG4/H.263 and for H.264 encoding.
// This parameter specifies the number of data
//  partition - 1 for MPEG4/H.264 encoding and
//  the number of slice group - 1 in a frame
//  for H.264 encoding.
// If this value is programmed to 0, this
//  specifies MPEG4/H.263 encoding without
//  data partitioning. If data partitioning
//  is enabled, this value must be set to 2.
// If this value is programmed to 0, only 1
//  slice group is present which means that
//  FMO encoding is disabled for H.264.
// Lwrrently maximum of 3 slice groups are
//  supported in H/W so valid value for this
//  parameter is 0 to 2.
// This parameter must be programmed first to
//  write/read other slice groups registers.
// For H264 case, the number of slice groups
//  must be less or equal to the number of
//  macroblocks in a frame.
#define LWE2B7_NUM_SLICE_GROUPS_0_NUM_SLICE_GROUPS_MINUS1                  2:0
#define LWE2B7_NUM_SLICE_GROUPS_0_NUM_SLICE_GROUPS_MINUS1_DEFAULT  (0x00000000)

// Slice Group Slice Type.
// This parameter is applicable for H.264
//  encoding only and is not used for
//  MPEG4/H.263 encoding.
// This parameter is applied only for P-Frame
//  and is not used for I-Frame because in
//  I-Frame, all slices are I_TYPE.
// Each bit in this field specifies slice type
//  of each corresponding slice group in a
//  frame.
// SLICEGROUP_SLICETYPE(0) specifies slice type
//  for slice group # 0.
// SLICEGROUP_SLICETYPE(1) specifies slice type
//  for slice group # 1. Etc.
// For each slice group, 0 means P_TYPE, 1
//  means I_TYPE.
// If slice group slice type is I_TYPE, all the
//  slices in that group are I-slices.
// If slice group slice type is P_TYPE, all the
//  slices in that group are P-slices.
// Typically, all bits of this parameter should
//  be set to zero such that all slices of a
//  P-Frame are P-slices. Setting slice group
//  slice type to I_TYPE will generate I-slices
//  which may cause high bitrate encoding which
//  may not be desirable.
#define LWE2B7_NUM_SLICE_GROUPS_0_SLICEGROUP_SLICETYPE                     11:4
#define LWE2B7_NUM_SLICE_GROUPS_0_SLICEGROUP_SLICETYPE_DEFAULT     (0x00000000)

// Slice Group Refresh Enable.
// This parameter is applicable for H.264
//  encoding only and is not used for
//  MPEG4/H.263 encoding.
// This parameter is applied only for P-Frame
//  and is not used for I-Frame because in
//  I-Frame, all slices are I_TYPE.
// Each bit in this field specifies slice group
//  refresh enable for each corresponding slice
//  group in a frame.
// SLICE_GROUP_REFRESH_ENABLE(0) specifies
//  refresh enable for slice group # 0.
// SLICE_GROUP_REFRESH_ENABLE(1) specifies
//  refresh enable for slice group # 1. Etc.
// Setting a bit in this parameter to 1 will
//  cause all the slices in the corresponding
//  slicegroup to be I-slices for P-frame.
// This feature is provided to generate intra
//  refresh of all the slices (hence all
//  macroblocks) inside a slice group of a
//  P-frame. If Intra Refres is enabled,
//  intra refresh counts in the intra
//  refresh pattern SRAM will also be updated
//  when I-slices are generated.
// These bits are self cleared by H/W after
//  they take effect in a P-Frame.
// This parameter should not be used to force
//  I-frame because then it could cause sudden
//  increase in bitrate which is not desirable.
#define LWE2B7_NUM_SLICE_GROUPS_0_SLICE_GROUP_REFRESH_ENABLE                       19:12
#define LWE2B7_NUM_SLICE_GROUPS_0_SLICE_GROUP_REFRESH_ENABLE_DEFAULT       (0x00000000)

// ---------------------------------------------------------------------------------------------
//
// IMPORTANT !!! 
//
// Per-slice register group.
// This set of registers are mainly used for H.264 encoding to specify the slice groups.
// However some of these registers that pertains to VLC DMA operations are also applicable
//  for MPEG4/H.263 encoding with and without data partitioning. Note that MPEG4 encoding
//  with data partitioning enabled requires 3 data partitions.
// The encoder hardware maintain a multiple set of slice group registers for the maximum
//  supported number of slice groups. Lwrrently maximum of 3 slice groups are supported out
//  of the 8 slice groups allowed by H.264 standard. This may be expanded in the future.
//
// For each data partition or slice group, there are a number of registers that need to be
//  programmed:
// A. DMA_BUFFER_ADDR (for MPEG4/H.264)
// B. DMA_LIST_ADDR (for MPEG4/H.264)
// C. DMA_BUFFER_SIZE (for MPEG4/H.264)
// D. DMA_LIST_SIZE (for MPEG4/H.264)
// E. PIC_INIT_Q (for H.264 only)
// F. MAX_MIN_QP_I (for H.264 only)
// G. MAX_MIN_QP_P (for H.264 only)
// H. SLICE_PARAMS (for H.264 only)
// I. NUM_OF_UNITS (for H.264 only)
// J. TOP_LEFT (for H.264 only)
// K. BOTTOM_RIGHT (for H.264 only)
// L. CHANGE_RATE (for H.264 only)
// M. DMA_BUFFER_STATUS (for MPEG4/H.264) (read-only)
// N. DMA_LIST_STATUS (for MPEG4/H.264) (read-only)
//
// To program the slice group related registers:
// 1. First the number of slice groups (NUM_SLICE_GROUPS_MINUS1 field in NUM_SLICE_GROUP
//    register must be programmed.
// 2. DMA_BUFFER_ADDR must then be written as the first register for slice group 0.
// 3. Write other registers for slice group 0.
// 4. Repeat step 2 and 3 for slice group 1, etc. Note that DMA_BUFFER_ADDR register in step 2
//    MUST be written repeatedly for the exact number of slice groups.
// The same procedure must be followed for reading slice group registers.
// 
// ---------------------------------------------------------------------------------------------
// DMA_BUFFER_OFFSET is renamed to DMA_BUFFER_ADDR.

// This register is effective in the next frame
//  start if VLCDMA_CONTEXT in set to RELOAD.
// The hardware maintains multiple copies of
//  this register for each data partition in
//  MPEG4/H.263 encoding and each slice group
//  in H.264 encoding for as many slice groups
//  as the number of supported slice groups.
// Writing or reading this register will advance
//  the internal slice group pointer which
//  determine the set of slice group registers
//  to be written or read. This internal slice
//  group pointer is initialized to point to
//  slice group 0 registers by writing
//  or reading this register immediately after
//  writing to NUM_SLICE_GROUPS register.
//  So, this register must be written or read
//  first after NUM_SLICE_GROUPS register is
//  written. And for each slice group, this
//  register must also be the first register
//  written or read before the rest of the
//  slice group registers are written or read.
// Consequently, this register must be written
//  or read for as many times as the number of
//  slice groups programmed in NUM_SLICE_GROUPS
//  register to fully write or read all the
//  slice group registers.
#define LWE2B7_DMA_BUFFER_ADDR_0                  (0x80)
// VLC DMA Buffer Start Address.
// This parameter specifies the 16-byte aligned
//  start address for VLC DMA buffer for each
//  for each data partition in MPEG4/H.263
//  encoding and each slice group in H.264
//  encoding.
// For MPEG4 data partition mode:
//  VLC DMA Buffer 0 holds header 1 data,
//  VLC DMA Buffer 1 holds header 2 data,
//  VLC DMA Buffer 2 holds texture data.
#define LWE2B7_DMA_BUFFER_ADDR_0_DMA_BUFFER_ADDR                   31:4
#define LWE2B7_DMA_BUFFER_ADDR_0_DMA_BUFFER_ADDR_DEFAULT           (0x00000000)

// DMA_LINK_LIST_OFFSET is renamed to DMA_LIST_ADDR.

// This register is effective in the next frame
//  start if VLCDMA_CONTEXT in set to RELOAD.
// The hardware maintains multiple copies of
//  this register for each data partition in
//  MPEG4/H.263 encoding and each slice group
//  in H.264 encoding for as many slice groups
//  as the number of supported slice groups.
// This register must be written repeatedly
//  for each slice group to fully program all
//  the slice groups specified in
//  NUM_SLICE_GROUPS register.
#define LWE2B7_DMA_LIST_ADDR_0                    (0x81)
// VLC DMA Linked List Start Address.
// This parameter specifies the 16-byte aligned
//  start address for the linked lists that
//  keep track of saved packets in the
//  VLC DMA buffer for each data partition in
//  MPEG4/H.263 encoding and each slice group
//  in H.264 encoding.
// For MPEG4 data partition mode:
//  VLC DMA Buffer 0 holds header 1 data,
//  VLC DMA Buffer 1 holds header 2 data,
//  VLC DMA Buffer 2 holds texture data.
#define LWE2B7_DMA_LIST_ADDR_0_DMA_LIST_ADDR                       31:4
#define LWE2B7_DMA_LIST_ADDR_0_DMA_LIST_ADDR_DEFAULT               (0x00000000)

// DMA_BUF_SIZE is renamed to DMA_BUFFER_SIZE.

// This register is effective in the next frame
//  start if VLCDMA_CONTEXT in set to RELOAD.
// The hardware maintains multiple copies of
//  this register for each data partition in
//  MPEG4/H.263 encoding and each slice group
//  in H.264 encoding for as many slice groups
//  as the number of supported slice groups.
// This register must be written repeatedly
//  for each slice group to fully program all
//  the slice groups specified in
//  NUM_SLICE_GROUPS register.
#define LWE2B7_DMA_BUFFER_SIZE_0                  (0x82)
// VLC DMA Buffer Size.
// This parameter specifies the size - 1 for
//  VLC DMA buffer in multiple of 16-byte words
//  for each data partition in MPEG4/H.263
//  encoding and each slice group in H.264
// Programming of this parameter depends on
//  video encoding mode. Assuming that enough
//  memory is allocated for encoded bitstream
//  buffers, the following is recommendation
//  on this parameter programming. In all cases,
//  the programmed value should not be less
//  than 2KB for each VLC DMA buffer.
// For non data-partition  MPEG4/H.263 encoding,
//  only one VLC DMA Buffer is needed and this
//  parameter should be set such that it is
//  enough to hold up to 4 worst case packets.
//  In bit-based packetization, the packet size
//  is known (PACKET_COUNT). In MB-based
//  packetization, estimated worst case packet
//  size should be programmed.
// For single slice group H.264 encoding,
//  only one VLC DMA buffer is needed and this
//  parameter should be set such that it is
//  enough to store 4 worst case NALs.
//  In bit-based packetization, the packet size
//  is known (PACKET_COUNT_H264). In MB-based
//  packetization, estimated worst case packet
//  size should be programmed.
// For data-partition MPEG4 encoding,
//  three VLC DMA buffers are needed and this
//  parameter should be set such that it is
//  enough to hold up to 4 worst case packets.
//  Maximum texture packet size is 2KB so up to
//  8KB should be allocated (maybe less for
//  smaller packet size) for VLC DMA buffer 2.
//  Maximum header packet size is 1KB so up to
//  4KB should be allocated (maybe less for
//  smaller packet size) for each DMA buffer 0
//  and 1.
//  Note that for MPEG4 data partition mode:
//  VLC DMA Buffer 0 holds header 1 data,
//  VLC DMA Buffer 1 holds header 2 data,
//  VLC DMA Buffer 2 holds texture data.
// For multi slice groups H.264 4 encoding,
//  one VLC DMA buffer is needed per slice
//  group and this parameter should be set such
//  that total VLC DMA buffer is enough to hold
//  1.5 to 2 worst-case encoded frames.
//  This should be distributed among the VLC
//  DMA buffers according to the ratio of
//  macroblocks in the slice groups (slice map).
#define LWE2B7_DMA_BUFFER_SIZE_0_DMA_BUFFER_SIZE                   24:4
#define LWE2B7_DMA_BUFFER_SIZE_0_DMA_BUFFER_SIZE_DEFAULT           (0x00000000)

// DMA_LINK_LIST_SIZE_H264 is renamed to DMA_LIST_SIZE.

// This register is effective in the next frame
//  start if VLCDMA_CONTEXT in set to RELOAD.
// The hardware maintains multiple copies of
//  this register for each data partition in
//  MPEG4/H.263 encoding and each slice group
//  in H.264 encoding for as many slice groups
//  as the number of supported slice groups.
// This register must be written repeatedly
//  for each slice group to fully program all
//  the slice groups specified in
//  NUM_SLICE_GROUPS register.
#define LWE2B7_DMA_LIST_SIZE_0                    (0x83)
// VLC DMA Linked List Buffer Size.
// This parameter specifies the size - 1 for
//  the linked lists that keep track of saved
//  packets in the VLC DMA buffer for each
//  data partition in MPEG4/H.263 encoding
//  and each slice group in H.264 encoding.
// For MPEG4 with and without data partition,
//  this parameter may be set to 2KB for each
//  VLC DMA buffer.
// For single slice group H.264 encoding, this
//  parameter may also be set to 2KB.
// For multi slice groups H.264 encoding, this
//  parameter may be set to 2 * 16-bytes *
//  number of NALs per frame.
#define LWE2B7_DMA_LIST_SIZE_0_DMA_LIST_SIZE                       24:4
#define LWE2B7_DMA_LIST_SIZE_0_DMA_LIST_SIZE_DEFAULT               (0x00000000)

// Bits 20-16 (CHROMA_QP_INDEX) of PIC_INIT_Q register is moved to PIC_PARAMETERS.
// Bit 24 (INIT_QP_I_ENABLE_H264) of PIC_INIT_Q register is deleted.
// Bit 25 (INIT_QP_P_ENABLE_H264) of PIC_INIT_Q register is deleted.
// PIC_INIT_Q is made to be per slice parameters.

// This register is applicable for H.264
//  encoding only. For MPEG4/H.263 encoding,
//  I_INIT_QP parameter in I_RATE_CTRL register
//  and P_INIT_QP parameter in P_RATE_CTRL
//  register are used instead.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
// The hardware maintain multiple copies of this
//  register for each slice group for as many
//  slice groups as the number of supported
//  slice groups.
// This register must be written repeatedly
//  for each slice group to fully program all
//  the slice groups specified in
//  NUM_SLICE_GROUPS register.
// For AP15, the implementation can only use
//  the same value for all slice groups so
//  only the value corresponding to slice group
//  0 is used although software should program
//  the same value for slice group 1 and 2 for
//  future compatibility.
#define LWE2B7_PIC_INIT_Q_0                       (0x84)
// Initial Qp for I-Frame.
// This register specifies initial quantization
//  scale for I frames for H.264 encoding.
// Recommendation  on how to select initial QP
//  based on bpp (bits per pixel) is given at
//  the begining of this document.
//  Search for "MinQp/InitQp recommendation for
//  H264 Rate Control"
#define LWE2B7_PIC_INIT_Q_0_I_INIT_QP_H264                 7:0
#define LWE2B7_PIC_INIT_Q_0_I_INIT_QP_H264_DEFAULT                 (0x00000000)

// Initial Qp for P-Frame.
// This register specifies initial quantization
//  scale for P frames for H.264 encoding.
// Design note: in HW simulations, RMC generator
//  callwlates internally I_INIT_QP_H264,
//  P_INIT_QP_H264, MIN_QP_I, MAX_QP_I,
// MIN_QP_P,/MAX_QP_P from -bitrate option. This
//  can be overriden by passing the csv options
//  for all these variables. When csv options
//  are passed extrenally all the parameters
//  should be specified together (not some) as
//  this may conflict with internal callwlation.
//  Software driver should have similar input 
//  scheme for matching A-model. 
#define LWE2B7_PIC_INIT_Q_0_P_INIT_QP_H264                 15:8
#define LWE2B7_PIC_INIT_Q_0_P_INIT_QP_H264_DEFAULT                 (0x00000000)

// This register is applicable for H.264
//  encoding only and is not used for
//  MPEG4/H.263 encoding.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
// The hardware maintain multiple copies of this
//  register for each slice group for as many
//  slice groups as the number of supported
//  slice groups.
// This register must be written repeatedly
//  for each slice group to fully program all
//  the slice groups specified in
//  NUM_SLICE_GROUPS register.
// For AP15, the implementation can only use
//  the same value for all slice groups so
//  only the value corresponding to slice group
//  0 is used although software should program
//  the same value for slice group 1 and 2 for
//  future compatibility.
#define LWE2B7_MAX_MIN_QP_I_0                     (0x85)
// Minimum Qp for I-Slice.
// For H.264 encoding, this parameter specifies
//  the minimum Qp for I-slice.
// Recommendation  on how to select min QP based
//  on bpp (bits per pixel) is given at the
//  begining of this document.
//  Search for "MinQp/InitQp recommendation for
//  H264 Rate Control"
#define LWE2B7_MAX_MIN_QP_I_0_MIN_QP_I                     7:0
#define LWE2B7_MAX_MIN_QP_I_0_MIN_QP_I_DEFAULT                     (0x00000000)

// Maximum Qp for I-Slice.
// For H.264 encoding, this parameter specifies
//  the maximum Qp for I-slice.
#define LWE2B7_MAX_MIN_QP_I_0_MAX_QP_I                     15:8
#define LWE2B7_MAX_MIN_QP_I_0_MAX_QP_I_DEFAULT                     (0x00000000)

// Minimum Qp for Redundant I-Slice.
// For H.264 encoding, this parameter specifies
//  the minimum Qp for redundant I-slice.
// This parameter is lwrrently not used.
#define LWE2B7_MAX_MIN_QP_I_0_RED_MIN_QP_I                 23:16
#define LWE2B7_MAX_MIN_QP_I_0_RED_MIN_QP_I_DEFAULT                 (0x00000000)

// Maximum Qp for Redundant I-Slice.
// For H.264 encoding, this parameter specifies
//  the maximum Qp for redundant I-slice.
// This parameter is lwrrently not used.
#define LWE2B7_MAX_MIN_QP_I_0_RED_MAX_QP_I                 31:24
#define LWE2B7_MAX_MIN_QP_I_0_RED_MAX_QP_I_DEFAULT                 (0x00000000)

// This register is applicable for H.264
//  encoding only and is not used for
//  MPEG4/H.263 encoding.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
// The hardware maintain multiple copies of this
//  register for each slice group for as many
//  slice groups as the number of supported
//  slice groups.
// This register must be written repeatedly
//  for each slice group to fully program all
//  the slice groups specified in
//  NUM_SLICE_GROUPS register.
// For AP15, the implementation can only use
//  the same value for all slice groups so
//  only the value corresponding to slice group
//  0 is used although software should program
//  the same value for slice group 1 and 2 for
//  future compatibility.
#define LWE2B7_MAX_MIN_QP_P_0                     (0x86)
// Minimum Qp for P-Slice.
// For H.264 encoding, this parameter specifies
//  the minimum Qp for P-slice.
#define LWE2B7_MAX_MIN_QP_P_0_MIN_QP_P                     7:0
#define LWE2B7_MAX_MIN_QP_P_0_MIN_QP_P_DEFAULT                     (0x00000000)

// Maximum Qp for P-Slice.
// For H.264 encoding, this parameter specifies
//  the maximum Qp for P-slice.
#define LWE2B7_MAX_MIN_QP_P_0_MAX_QP_P                     15:8
#define LWE2B7_MAX_MIN_QP_P_0_MAX_QP_P_DEFAULT                     (0x00000000)

// Minimum Qp for Redundant P-Slice.
// For H.264 encoding, this parameter specifies
//  the minimum Qp for redundant P-slice.
// This parameter is lwrrently not used.
#define LWE2B7_MAX_MIN_QP_P_0_RED_MIN_QP_P                 23:16
#define LWE2B7_MAX_MIN_QP_P_0_RED_MIN_QP_P_DEFAULT                 (0x00000000)

// Maximum Qp for Redundant P-Slice.
// For H.264 encoding, this parameter specifies
//  the maximum Qp for redundant P-slice.
// This parameter is lwrrently not used.
#define LWE2B7_MAX_MIN_QP_P_0_RED_MAX_QP_P                 31:24
#define LWE2B7_MAX_MIN_QP_P_0_RED_MAX_QP_P_DEFAULT                 (0x00000000)

// This register is applicable for H.264
//  encoding only and is not used for
//  MPEG4/H.263 encoding.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
// The hardware maintain multiple copies of this
//  register for each slice group for as many
//  slice groups as the number of supported
//  slice groups.
// This register must be written repeatedly
//  for each slice group to fully program all
//  the slice groups specified in
//  NUM_SLICE_GROUPS register.
// For AP15, the implementation can only use
//  the same value for all slice groups so
//  only the value corresponding to slice group
//  0 is used although software should program
//  the same value for slice group 1 and 2 for
//  future compatibility.
#define LWE2B7_SLICE_PARAMS_0                     (0x87)
// Slice Alpha C0 (slice_alpha_c0_offset_div2)
// Valid value is from -6 to 6.
// Default may be set to 0.
#define LWE2B7_SLICE_PARAMS_0_SLICE_ALPHA_C0                       3:0
#define LWE2B7_SLICE_PARAMS_0_SLICE_ALPHA_C0_DEFAULT               (0x00000000)

// Slice Beta (slice_beta_offset_div2)
// Valid value is from -6 to 6.
// Default may be set to 0.
#define LWE2B7_SLICE_PARAMS_0_SLICE_BETA                   19:16
#define LWE2B7_SLICE_PARAMS_0_SLICE_BETA_DEFAULT                   (0x00000000)

//  Map.
// This register is applicable for H.264
//  encoding only.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
// The hardware maintain multiple copies of this
//  register for each slice group for as many
//  slice groups as the number of supported
//  slice groups.
// This register must be written repeatedly
//  for each slice group to fully program all
//  the slice groups specified in
//  NUM_SLICE_GROUPS register.
// This register specifies the slice group map
//  when SLICE_GRP_MAP_TYPE is set to
//  INTERLEAVED.
#define LWE2B7_NUM_OF_UNITS_0                     (0x88)
// Number of Units for Interleaved Slice Group
//  Map.
// This parameter specifies number of units
//  in macroblocks when SLICE_GRP_MAP_TYPE is
//  set to INTERLEAVED.
// This parameter is not used for other slice
/// group map type.
#define LWE2B7_NUM_OF_UNITS_0_NUM_OF_UNITS                 12:0
#define LWE2B7_NUM_OF_UNITS_0_NUM_OF_UNITS_DEFAULT                 (0x00000000)

//  Slice Group Map.
// This register is applicable for H.264
//  encoding only.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
// The hardware maintain multiple copies of this
//  register for each slice group for as many
//  slice groups as the number of supported
//  slice groups.
// This register should be written repeatedly
//  for each slice group to fully program all
//  the slice groups specified in
//  NUM_SLICE_GROUPS register.
//  However parameters in this register is
//  not needed for the last slice group
//  therefore software may optionally
//  skip programming in the last slice group.
// This register specifies the slice group map
//  when SLICE_GRP_MAP_TYPE is set to
//  FOREGROUND_BACKGROUND.
#define LWE2B7_TOP_LEFT_0                 (0x89)
// Top Left corner Y position for Foreground
//  Background Slice Group Map.
// This parameter specifies top position
//  in macroblocks when SLICE_GRP_MAP_TYPE is
//  set to FOREGROUND_BACKGROUND.
// This parameter is not used for other slice
//  group map type.
#define LWE2B7_TOP_LEFT_0_TOP_LEFT_Y                       7:0
#define LWE2B7_TOP_LEFT_0_TOP_LEFT_Y_DEFAULT                       (0x00000000)

// Top Left corner X position for Foreground
//  Background Slice Group Map.
// This parameter specifies left position
//  in macroblocks when SLICE_GRP_MAP_TYPE is
//  set to FOREGROUND_BACKGROUND.
// This parameter is not used for other slice
//  group map type.
#define LWE2B7_TOP_LEFT_0_TOP_LEFT_X                       23:16
#define LWE2B7_TOP_LEFT_0_TOP_LEFT_X_DEFAULT                       (0x00000000)

//  Background Slice Group Map.
// This register is applicable for H.264
//  encoding only.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
// The hardware maintain multiple copies of this
//  register for each slice group for as many
//  slice groups as the number of supported
//  slice groups.
// This register should be written repeatedly
//  for each slice group to fully program all
//  the slice groups specified in
//  NUM_SLICE_GROUPS register.
//  However parameters in this register is
//  not needed for the last slice group
//  therefore software may optionally
//  skip programming in the last slice group.
// This register specifies the slice group map
//  when SLICE_GRP_MAP_TYPE is set to
//  FOREGROUND_BACKGROUND.
#define LWE2B7_BOTTOM_RIGHT_0                     (0x8a)
// Bottom Right corner Y position for
//  Foreground Background Slice Group Map.
// This parameter specifies bottom position
//  in macroblocks when SLICE_GRP_MAP_TYPE is
//  set to FOREGROUND_BACKGROUND.
// This parameter is not used for other slice
//  group map type.
#define LWE2B7_BOTTOM_RIGHT_0_BOTTOM_RIGHT_Y                       7:0
#define LWE2B7_BOTTOM_RIGHT_0_BOTTOM_RIGHT_Y_DEFAULT               (0x00000000)

// Bottom Right corner X position for
//  Foreground Background Slice Group Map.
// This parameter specifies right position
//  in macroblocks when SLICE_GRP_MAP_TYPE is
//  set to FOREGROUND_BACKGROUND.
// This parameter is not used for other slice
//  group map type.
#define LWE2B7_BOTTOM_RIGHT_0_BOTTOM_RIGHT_X                       23:16
#define LWE2B7_BOTTOM_RIGHT_0_BOTTOM_RIGHT_X_DEFAULT               (0x00000000)

// This register is applicable for H.264
//  encoding only.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
// The hardware maintain multiple copies of this
//  register for each slice group for as many
//  slice groups as the number of supported
//  slice groups.
// This register must be written repeatedly
//  for each slice group to fully program all
//  the slice groups specified in
//  NUM_SLICE_GROUPS register.
// This register specifies the slice group map
//  when SLICE_GRP_MAP_TYPE is set to
//  BOX_OUT, RASTER, and WIPE.
#define LWE2B7_CHANGE_RATE_0                      (0x8b)
// Slice Group Change Rate - 1 for
//  Box Out, Raster, and Wipe Slice Group Maps.
// This parameter specifies ?
//  when SLICE_GRP_MAP_TYPE is set to
//  BOX_OUT, RASTER, and WIPE.
// This parameter is not used for other slice
//  group map type.
#define LWE2B7_CHANGE_RATE_0_SLICE_GROUP_CHANGE_RATE_MINUS1                12:0
#define LWE2B7_CHANGE_RATE_0_SLICE_GROUP_CHANGE_RATE_MINUS1_DEFAULT        (0x00000000)

// Slice Group Change Direction for
//  Box Out, Raster, and Wipe Slice Group Maps.
// This parameter specifies slice map direction
//  when SLICE_GRP_MAP_TYPE is set to
//  BOX_OUT, RASTER, and WIPE.
// This parameter is not used for other slice
//  group map type.
// Only the value programmed for slice group 0
//  is used because the H.264 standard specifies
//  that this value must be programmed the same
//  for all slice groups. The values programmed
//  for slice group 1 and 2 will be ignored.
#define LWE2B7_CHANGE_RATE_0_SLICE_GROUP_CHANGE_DIRECTION                  16:16
#define LWE2B7_CHANGE_RATE_0_SLICE_GROUP_CHANGE_DIRECTION_NORMAL                  (0)    // // For Box Out slice group map type, this
//  specifies clockwise direction.
// For Raster slice group map type, this
//  specifies normal raster direction.
// For Wipe slice group map type, this
//  specifies wipe right direction.

#define LWE2B7_CHANGE_RATE_0_SLICE_GROUP_CHANGE_DIRECTION_REVERSE                 (1)    // // For Box Out slice group map type, this
//  specifies counter clockwise direction.
// For Raster slice group map type, this
//  specifies reverse raster direction.
// For Wipe slice group map type, this
//  specifies wipe left direction.
#define LWE2B7_CHANGE_RATE_0_SLICE_GROUP_CHANGE_DIRECTION_DEFAULT  (0x00000000)

// Slice Group Change Cycle Width for
//  Box Out, Raster, and Wipe Slice Group Maps.
// This parameter must be programmed with
//  Ceil(log2(PicSizeInMapUnits /
//       SliceGroupChangeRate+1))
// PicSizeInMapUnits is the number of
//  of macroblocks in a frame that can be
//  callwlated as WIDTH * HEIGHT as defined
//  in WIDTH_HEIGHT register.
// Maximum value for this register is
//  log2(PicSizeInMapUnits) = 13.
// This parameter is not used for other slice
//  group map type.
#define LWE2B7_CHANGE_RATE_0_SLICE_GROUP_CHANGE_CYCLE_WIDTH                27:24
#define LWE2B7_CHANGE_RATE_0_SLICE_GROUP_CHANGE_CYCLE_WIDTH_DEFAULT        (0x00000000)

// DMA_BUF_OVFL_THRESH is renamed to DMA_BUFFER_STATUS.

// The hardware maintains multiple copies of
//  this register for each data partition in
//  MPEG4/H.263 encoding and each slice group
//  in H.264 encoding for as many slice groups
//  as the number of supported slice groups.
// This register must be read repeatedly for
//  each slice group to fully read all the
//  slice groups specified in NUM_SLICE_GROUPS
//  register.
#define LWE2B7_DMA_BUFFER_STATUS_0                        (0x8c)
// VLC DMA Buffer Status.
// This parameter returns VLC DMA buffer
//  status in terms of used slots for
//  each data partition in MPEG4/H.263 encoding
//  and each slice group in H.264 encoding.
#define LWE2B7_DMA_BUFFER_STATUS_0_DMA_BUFFER_STATUS                       24:4
#define LWE2B7_DMA_BUFFER_STATUS_0_DMA_BUFFER_STATUS_DEFAULT       (0x00000000)

// DMA_LINK_LIST_OVFL_THRESH is renamed to DMA_LIST_STATUS.

// The hardware maintains multiple copies of
//  this register for each data partition in
//  MPEG4/H.263 encoding and each slice group
//  in H.264 encoding for as many slice groups
//  as the number of supported slice groups.
// This register must be read repeatedly for
//  each slice group to fully read all the
//  slice groups specified in NUM_SLICE_GROUPS
//  register.
#define LWE2B7_DMA_LIST_STATUS_0                  (0x8d)
// VLC DMA Link List Status.
// This parameter returns the status in terms
//  of used slots for the linked lists that
//  keep track of saved packets in the VLC DMA
//  buffer for each data partition in
//  MPEG4/H.263 encoding and each slice group
//  in H.264 encoding.
#define LWE2B7_DMA_LIST_STATUS_0_DMA_LIST_STATUS                   24:4
#define LWE2B7_DMA_LIST_STATUS_0_DMA_LIST_STATUS_DEFAULT           (0x00000000)

// End of per-slice registers.
//----------------------------------------------------------------------------------------------

// This register is applicable for H.264
//  encoding only.
// This register specifies address of slice
//   map A to support arbitrary slice map.
//  (SLICE_GRP_MAP_TYPE set to EXPLICIT).
// Two slice map buffers (A and B) are
//  supported so that software can use them
//  as ping-pong buffers.
#define LWE2B7_SLICE_MAP_OFFSET_A_0                       (0x8e)
// Slice Map A Start Address.
// This parameter specifies 16-byte aligned
//  address of slice map A to support
//  arbitrary  explicit) slice map
//  (SLICE_GRP_MAP_TYPE set to EXPLICIT).
// This feature is lwrrently not supported.
#define LWE2B7_SLICE_MAP_OFFSET_A_0_SLICE_MAP_OFFSET_A                     31:4
#define LWE2B7_SLICE_MAP_OFFSET_A_0_SLICE_MAP_OFFSET_A_DEFAULT     (0x00000000)

// This register is applicable for H.264
//  encoding only.
// This register specifies address of slice
//   map A to support arbitrary slice map.
//  (SLICE_GRP_MAP_TYPE set to EXPLICIT).
// Two slice map buffers (A and B) are
//  supported so that software can use them
//  as ping-pong buffers.
#define LWE2B7_SLICE_MAP_OFFSET_B_0                       (0x8f)
// Slice Map B Start Address.
// This parameter specifies 16-byte aligned
//  address of slice map A to support
//  arbitrary  explicit) slice map
//  (SLICE_GRP_MAP_TYPE set to EXPLICIT).
// This feature is lwrrently not supported.
#define LWE2B7_SLICE_MAP_OFFSET_B_0_SLICE_MAP_OFFSET_B                     31:4
#define LWE2B7_SLICE_MAP_OFFSET_B_0_SLICE_MAP_OFFSET_B_DEFAULT     (0x00000000)

// Offset 0x0A0
//
// Motion Search Control
//
// Bit 28 (INTRA_SUBMODE_COST_FUNCTION) of MOT_SEARCH_CTRL is deleted.
// Bit 30 (IFRAME_INTRAMODE_APPROX_PRED) of MOT_SEARCH_CTRL is deleted.
// Bit 31 (PFRAME_INTRAMODE_APPROX_PRED) of MOT_SEARCH_CTRL is deleted.

// This register specifies various control for
//  mode decision.
// This register is programmed only once.
#define LWE2B7_MOT_SEARCH_CTRL_0                  (0xa0)
// Inter Motion Vector (MV) modes.
// This parameter is applicable for APxx H.264
//  encoding only and is not used for
//  MPEG4/H.263 encoding.
// For MPEG4/H.263 encoding motion vector type
//  is decided by MV4ENABLE in FRAME_CTRL
//  register.
// Each bit of this parameter specifies which
//  of the inter modes are enabled for encoding.
// 0 means "mode not present (disabled)".
// 1 means "mode is present (enabled)".
// Bit 0 - 16x16.
// Bit 1 - 16x8 (H264 only).
// Bit 2 - 8x16 (H264 only).
// Bit 3 - 8x8.
// Bit 4 - 8x4 (H264 only).
// Bit 5 - 4x8 (H264 only).
// Bit 6 - 4x4 (H264 only).
// When LEVEL_IDC >= 31, in order to meet level 
//  limit (max number of motion vectors per two
//  conselwtive MBs), 4x4 motion vector should 
//  be disabled.
#define LWE2B7_MOT_SEARCH_CTRL_0_MV_MODE                   6:0
#define LWE2B7_MOT_SEARCH_CTRL_0_MV_MODE_DEFAULT                   (0x00000000)

// Intra Spatial Prediction Modes (H264 only)
// This parameter is applicable for H.264
//  encoding only and is not used for
//  MPEG4/H.263 encoding.
// Each bit of this parameter specifies which
//  of the intra modes are enabled for encoding.
// 0 means "mode not present (disabled)"
// 1 means "mode is present (enabled)"
// Bit 7 -  Intra luma 4x4 Vertical.
// Bit 8 -  Intra luma 4x4 Horizontal.
// Bit 9 -  Intra luma 4x4 DC.
// Bit 10 - Intra luma 4x4 Diagonal down left.
// Bit 11 - Intra luma 4x4 Diagonal down right.
// Bit 12 - Intra luma 4x4 Vertical right.
// Bit 13 - Intra luma 4x4 Horizontal down.
// Bit 14 - Intra luma 4x4 Vertical left.
// Bit 15 - Intra luma 4x4 Horizontal up.
// Bit 16 - Intra luma 16x16 Verical.
// Bit 17 - Intra luma 16x16 Horizontal.
// Bit 18 - Intra luma 16x16 DC.
// Bit 19 - Intra luma 16x16 Plane.
// Bit 20 - Intra chroma DC.
// Bit 21 - Intra chroma Horizontal.
// Bit 22 - Intra chroma Vertical.
// Bit 23 - Intra chroma Plane.
// The following restrictions apply when
//   programming this field.
// If IP_DECISION_CTRL is set to USE_16X16_4X4,
//  either one or both of bit 9 (4x4 DC) and
//  bit 18 (16x16 DC) must be set.
// If IP_DECISION_CTRL is set to USE_16X16,
//  bit 18 (16x16 DC) must be set.
// If IP_DECISION_MODE is set to APPROX,
//  bit 9 (4x4 DC) must be set.
// Bit 20 (chroma DC) must always be set.
#define LWE2B7_MOT_SEARCH_CTRL_0_INTRA_PRED_MODE                   23:7
#define LWE2B7_MOT_SEARCH_CTRL_0_INTRA_PRED_MODE_DEFAULT           (0x00000000)

// Mode Decision.
// This parameter specifies how mode decision
//  is made in the motion search engine.
// Recommended value is BEST. Other settings
//  may be selected to reduce cycle count
//  with trade-off in quality.
#define LWE2B7_MOT_SEARCH_CTRL_0_MODE_DECISION                     25:24
#define LWE2B7_MOT_SEARCH_CTRL_0_MODE_DECISION_BEST                       (0)    // // Best mode decision possible. Decision is
//  made after half-pel search for MPEG4/H.263
//  and after quarter-pel search for H264).

#define LWE2B7_MOT_SEARCH_CTRL_0_MODE_DECISION_AFTER_FP                   (1)    // // Best mode decision made after full/integer
//  pel search.

#define LWE2B7_MOT_SEARCH_CTRL_0_MODE_DECISION_AFTER_HP                   (2)    // // Best mode decision after half-pel search.

#define LWE2B7_MOT_SEARCH_CTRL_0_MODE_DECISION_AFTER_QP                   (3)    // // Best mode decision after quarter-pel search.
//  This is valid for H.264 encoding only and
//  must not be programmed for MPEG4.
#define LWE2B7_MOT_SEARCH_CTRL_0_MODE_DECISION_DEFAULT             (0x00000000)

// Cost function for Motion Estimation/Search.
// This bit specifies the cost function used
//  for motion estimation/search.
#define LWE2B7_MOT_SEARCH_CTRL_0_DIFF_MODE                 26:26
#define LWE2B7_MOT_SEARCH_CTRL_0_DIFF_MODE_SAD                    (0)    // // SAD. This setting is recommended for
//  MPEG4/H.263 encoding.

#define LWE2B7_MOT_SEARCH_CTRL_0_DIFF_MODE_SATD                   (1)    // // SATD. This setting is recommended for
//  H.264 encoding.
#define LWE2B7_MOT_SEARCH_CTRL_0_DIFF_MODE_DEFAULT                 (0x00000000)

// Cost function for Intra mode.
// This bit is applicable for H.264 encoding
//  only and is not used for MPEG4/H.263
//  encoding.
// This bit specifies the cost function used
//  to decide between Intra 4x4/16x16 modes.
#define LWE2B7_MOT_SEARCH_CTRL_0_INTRA_MODE_COST_FUNCTION                  27:27
#define LWE2B7_MOT_SEARCH_CTRL_0_INTRA_MODE_COST_FUNCTION_SATD                    (0)    // // SATD.

#define LWE2B7_MOT_SEARCH_CTRL_0_INTRA_MODE_COST_FUNCTION_SAD                     (1)    // // SAD. This setting is not implemented yet.
#define LWE2B7_MOT_SEARCH_CTRL_0_INTRA_MODE_COST_FUNCTION_DEFAULT  (0x00000000)

// Internal Motion Vector Cost control.
// This bit is applicable for H.264 encoding
//  only and is not used for MPEG4/H.263
//  encoding.
// This bit controls internal motion cost
//  callwlation.
// Recommended value for this bit is ENABLE.
#define LWE2B7_MOT_SEARCH_CTRL_0_INTERNAL_MV_COST                  29:29
#define LWE2B7_MOT_SEARCH_CTRL_0_INTERNAL_MV_COST_ENABLE                  (0)    // // Internal motion vector cost callwlation
//  is enabled. Other motion search biases
//  should be set to 0 with this setting.

#define LWE2B7_MOT_SEARCH_CTRL_0_INTERNAL_MV_COST_DISABLE                 (1)    // // Internal motion vector cost callwlation
//  is disabled.
#define LWE2B7_MOT_SEARCH_CTRL_0_INTERNAL_MV_COST_DEFAULT          (0x00000000)


// This register specifies the motion search
//  range.
#define LWE2B7_MOT_SEARCH_RANGE_0                 (0xa1)
// Maximum motion vector in X direction.
// This parameter specifies motion search range
//  is specified in quarter pel resolution for
//  both MPEG4/H.263 and H.264.
// Though MPEG4/H.263 doesn't have quarter pel,
//  MV is still specified in 1/4 pel resolution
//  (for uniformity) and the corresponding MV
//  range in half pel resolution for MPEG4/H.263
//  is internally callwlated.
// Min search range in X direction permitted is
//  8 in quarter pel units.
// For MPEG4/H.263, maximum MV search range
//  supported in H/W is 31 in half pel resolution. 
//  If larger search range is
//  programmed, MV search range is internally
//  clipped to 31 in half pel resolution
//  for MPEG4/H.263.
// Max search range in X direction supported by
//  H/W for H.264 is 120 in quarter pel units.
#define LWE2B7_MOT_SEARCH_RANGE_0_MOT_SEARCH_X_RANGE                       15:0
#define LWE2B7_MOT_SEARCH_RANGE_0_MOT_SEARCH_X_RANGE_DEFAULT       (0x00000000)

// Maximum motion vector in Y direction.
// This parameter specifies motion search range
//  is specified in quarter pel resolution for
//  both MPEG4/H.263 and H.264.
// Though MPEG4/H.263 doesn't have quarter pel,
//  MV is still specified in 1/4 pel resolution
//  (for uniformity) and the corresponding MV
//  range in half pel resolution for MPEG4/H.263
//  is internally callwlated.
// Min search range in Y direction  permitted is
//  8 in quarter pel units.
// For MPEG4/H.263, maximum MV search range
//  supported in H/W is 31 in half pel resolution. 
//  If larger search range is
//  programmed, MV search range is internally
//  clipped to 31 in half pel resolution
//  for MPEG4/H.263.
// Max search range in Y direction supported by
//  H/W for H.264 is 72 in quarter pel units.
#define LWE2B7_MOT_SEARCH_RANGE_0_MOT_SEARCH_Y_RANGE                       31:16
#define LWE2B7_MOT_SEARCH_RANGE_0_MOT_SEARCH_Y_RANGE_DEFAULT       (0x00000000)

// This register specifies the threshold and
//  the enable bits for the motion search
//  engine to exit a search based on MAD values.
#define LWE2B7_MOTSEARCH_EXIT_0                   (0xa2)
// MAD-based Exit Enable.
// This bit is used to enable/disable the
//  MAD-based search termination.
#define LWE2B7_MOTSEARCH_EXIT_0_MAD_BASE_EXIT_ENABLE                       0:0
#define LWE2B7_MOTSEARCH_EXIT_0_MAD_BASE_EXIT_ENABLE_DISABLE                      (0)    // // Motion Search engine will not exit a search
//  based on the MAD comparison with the MAD
//  threshold.

#define LWE2B7_MOTSEARCH_EXIT_0_MAD_BASE_EXIT_ENABLE_ENABLE                       (1)    // // Motion Search engine will exit a search if
//  the MAD is less than the MAD threshold
//  and set the macroblock as Intra MB.
#define LWE2B7_MOTSEARCH_EXIT_0_MAD_BASE_EXIT_ENABLE_DEFAULT       (0x00000000)

// MAD Threshold.
// This parameter specifies the MAD threshold
// when MAD_BASED_EXIT_ENABLE is set to ENABLE.
#define LWE2B7_MOTSEARCH_EXIT_0_MAD_THRESHOLD                      16:1
#define LWE2B7_MOTSEARCH_EXIT_0_MAD_THRESHOLD_DEFAULT              (0x00000000)

// Motion Search Exit Mode.
// This bit controls exit mode of motion search
//  process.
#define LWE2B7_MOTSEARCH_EXIT_0_MOTSEARCH_EXIT_MODE                31:31
#define LWE2B7_MOTSEARCH_EXIT_0_MOTSEARCH_EXIT_MODE_NORMAL                        (0)    // // Motion Search engine will exit a search
//  when an optimum solution is found.
//  This setting MUST be used for normal
//  operation.

#define LWE2B7_MOTSEARCH_EXIT_0_MOTSEARCH_EXIT_MODE_FORCED                        (1)    // // Motion Search engine will exit a search
//  after traversing the complete search path.
//  This can be used for debug purpose and to
//  determine worst case performance
#define LWE2B7_MOTSEARCH_EXIT_0_MOTSEARCH_EXIT_MODE_DEFAULT        (0x00000000)


// This register specifies the Favor-Inter and
//  the Favor-16x16 biases.
// This register is programmed only once.
#define LWE2B7_MOTSEARCH_BIAS1_0                  (0xa3)
// Favor-Inter Bias.
// This parameter sets the bias towards inter
//  based on the MAD/SAD/SATD comparison.
// Recommended value is 128 for QCIF/CIF and
// 0 for VGA and larger resolution sequences.
#define LWE2B7_MOTSEARCH_BIAS1_0_FAVOR_INTER_BIAS                  15:0
#define LWE2B7_MOTSEARCH_BIAS1_0_FAVOR_INTER_BIAS_DEFAULT          (0x00000000)

// Favor-16x16 Bias.
// This parameter sets the bias towards inter
//  one motion vector instead of the four motion
//  vectors.
// Recommended value for MPEG4/H.263 is 129.
// Recommended value for H264 is 0.
#define LWE2B7_MOTSEARCH_BIAS1_0_FAVOR_16X16_BIAS                  31:16
#define LWE2B7_MOTSEARCH_BIAS1_0_FAVOR_16X16_BIAS_DEFAULT          (0x00000000)

// This register specifies the Favor-Zero bias.
// This register is programmed only once.
#define LWE2B7_MOTSEARCH_BIAS2_0                  (0xa4)
// Favor-Zero Bias.
// This value sets the bias towards (0,0)
//  motion vector.
// For APxx, recommended value for MPEG4/H.263
//  encoding is 129.
// For APxx, FAVOR_ZERO is not applicable for
//  H.264 encoding but MUST be programmed to 0
//  for good quality encoding?
#define LWE2B7_MOTSEARCH_BIAS2_0_FAVOR_ZERO_BIAS                   15:0
#define LWE2B7_MOTSEARCH_BIAS2_0_FAVOR_ZERO_BIAS_DEFAULT           (0x00000000)

// This register specifies the Favor 8x8 and
//  the Favor 16x8/8x16 biases.
// This register is programmed only once.
#define LWE2B7_MOTSEARCH_BIAS3_0                  (0xa5)
// Favor 16x8/8x16 Bias.
// This parameter is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
// This parameter sets the bias towards
//  16x8/8x16  motion vector beyond the internal
//  bias.
// Recommended value for H264 encoding is 0.
#define LWE2B7_MOTSEARCH_BIAS3_0_FAVOR_16X8_BIAS                   15:0
#define LWE2B7_MOTSEARCH_BIAS3_0_FAVOR_16X8_BIAS_DEFAULT           (0x00000000)

// Favor 8x8 Bias.
// This parameter sets the bias towards 8x8
//  beyond the internal bias based on the
//  SAD/SATD comparison.
// Recommended value for both MPEG4/H.263 and
//  H264 encoding is 0.
#define LWE2B7_MOTSEARCH_BIAS3_0_FAVOR_8X8_BIAS                    31:16
#define LWE2B7_MOTSEARCH_BIAS3_0_FAVOR_8X8_BIAS_DEFAULT            (0x00000000)

// Bit 31 (FAVOR_INTRA_INTERNAL_BIAS) of MOTSEARCH_BIAS4 is deleted.

// This register specifies the skip bias &
//  favor intra bias for Intra (I) vs Inter (P)
//  decision. 
// This register is programmed only once.
#define LWE2B7_MOTSEARCH_BIAS4_0                  (0xa6)
// Favor Skip Mode Bias.
// This parameter is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
// This parameter specifies the bias toward
//  skip motion vector beyond the internal bias.
// Recommended value for H264 encoding is 0.
#define LWE2B7_MOTSEARCH_BIAS4_0_FAVOR_SKIP_BIAS                   13:0
#define LWE2B7_MOTSEARCH_BIAS4_0_FAVOR_SKIP_BIAS_DEFAULT           (0x00000000)

// Internal Skip Bias Control.
// This parameter is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
// Recommended value for H264 encoding is
//  ENABLE.
#define LWE2B7_MOTSEARCH_BIAS4_0_FAVOR_SKIP_INTERNAL_BIAS                  14:14
#define LWE2B7_MOTSEARCH_BIAS4_0_FAVOR_SKIP_INTERNAL_BIAS_ENABLE                  (0)    // // Internal skip bias enabled.

#define LWE2B7_MOTSEARCH_BIAS4_0_FAVOR_SKIP_INTERNAL_BIAS_DISABLE                 (1)    // // Internal skip bias disabled.
#define LWE2B7_MOTSEARCH_BIAS4_0_FAVOR_SKIP_INTERNAL_BIAS_DEFAULT  (0x00000000)

// Favor Intra Bias.
// This register specifies the bias toward intra
//  macro block beyond the internal bias.
// Recommended value for MPEG4/H.263 and H264
//  encoding is 0.
#define LWE2B7_MOTSEARCH_BIAS4_0_FAVOR_INTRA_BIAS                  30:15
#define LWE2B7_MOTSEARCH_BIAS4_0_FAVOR_INTRA_BIAS_DEFAULT          (0x00000000)

// This register specifies the Favor 4x4 and
//  the Favor 4x8/8x4 sub mode biases during
//  sub-mode selection.
// This register is programmed only once.
#define LWE2B7_MOTSEARCH_BIAS5_0                  (0xa7)
// Favor 4x4 Sub-mode Bias.
// This parameter is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
// This parameter specifies the bias toward
//  an 8x8 sub-block which is broken into
//  4x4 sub-mode.
// Recommended value for H264 encoding is 0.
#define LWE2B7_MOTSEARCH_BIAS5_0_FAVOR_SUBMODE_4X4_BIAS                    15:0
#define LWE2B7_MOTSEARCH_BIAS5_0_FAVOR_SUBMODE_4X4_BIAS_DEFAULT    (0x00000000)

// Favor 4x8/8x4 Sub-mode Bias.
// This parameter is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
// This parameter specifies the bias toward
//  an 8x8 motion vector which is broken into
//  8x4 or 4x8 motion vectors.
#define LWE2B7_MOTSEARCH_BIAS5_0_FAVOR_SUBMODE_8X4_BIAS                    31:16
#define LWE2B7_MOTSEARCH_BIAS5_0_FAVOR_SUBMODE_8X4_BIAS_DEFAULT    (0x00000000)

// This register specifies the Favor 8x8
//  sub-mode biases during sub-mode selection.
// This register is programmed only once.
#define LWE2B7_MOTSEARCH_BIAS6_0                  (0xa8)
// Favor 8x8 Sub-mode Bias.
// This value sets the bias towards an 8x8
//  sub-block which is broken into 8x8 sub-mode.
//  during sub-mode selection.
// This bias is not applicable if none of
//  8x8 sub-block of the macroblock is broken
//  further. In this case macroblock level bias
//  (FAVOR_8X8_BIAS) is used instead.
// Recommended value for H264 encoding is 0.
#define LWE2B7_MOTSEARCH_BIAS6_0_FAVOR_SUBMODE_8X8_BIAS                    15:0
#define LWE2B7_MOTSEARCH_BIAS6_0_FAVOR_SUBMODE_8X8_BIAS_DEFAULT    (0x00000000)

// Offset 0x0C0
//
// Intra-Refresh Registers.
//

#define LWE2B7_INTRA_REF_CTRL_0                   (0xc0)
// Intra Refresh Enable.
// This bit controls the use of Intra Refresh
//  to control the intra macroblock insertion.
//  Note that intra macroblock may also be
//  inserted by I/P decision logic in the
//  motion estimation/search engine.
#define LWE2B7_INTRA_REF_CTRL_0_INTRA_REF_ENABLE                   0:0
#define LWE2B7_INTRA_REF_CTRL_0_INTRA_REF_ENABLE_DISABLE                  (0)    // // Intra Refresh disabled.

#define LWE2B7_INTRA_REF_CTRL_0_INTRA_REF_ENABLE_ENABLE                   (1)    // // Intra Refresh enabled.
#define LWE2B7_INTRA_REF_CTRL_0_INTRA_REF_ENABLE_DEFAULT           (0x00000000)

// Intra Refresh Mode.
// This bit specifies Intra Refresh modes.
//  Note that Intra Refresh logic maintains
//  a set of intra-refresh count in the
//  Intra Refresh pattern SRAM which is normally
//  decremented each frame. Each entry in the
//  Intra Refresh RAM correspond to 1 or more
//  (up to 16) macroblocks.
#define LWE2B7_INTRA_REF_CTRL_0_INTRA_REF_MODE                     2:1
#define LWE2B7_INTRA_REF_CTRL_0_INTRA_REF_MODE_DEFAULT_MODE                       (0)    // // Intra Refresh Default mode.
// In this mode, every time the intra refresh
//  counter is decremented to zero for a
//  macroblock or if an intra is decided by
//  the motion estimation engine, the
//  macroblock intra refresh count gets
//  reinitialized with a pseudo random value.
//  Every time an intra refresh count is
//  decremented to 0, the corresponding
//  macroblock(s) is coded as intra macroblock.

#define LWE2B7_INTRA_REF_CTRL_0_INTRA_REF_MODE_PATTERN_MODE                       (1)    // // Intra Refresh Pattern mode.
// In this mode, software will initialize the
//  intra refresh counts at the beginning.
//  When an intra refresh count is decremented
//  to 0, the corresponding macroblock(s) is
//  coded as intra macroblock and the intra
//  refresh count is reinitialized with a fixed
//  programmed value (MIN_FOR_IFRAME or
//  MIN_FOR_PFRAME).
//  Note that if a macroblock is decided by
//  motion estimation to be coded as an intra
//  macroblock, then the corresponding intra
//  refresh count is not reinitialized.

#define LWE2B7_INTRA_REF_CTRL_0_INTRA_REF_MODE_AIR_MODE                   (2)    // // Auto Intra Refresh mode.
// In this mode, intra refresh counts stored
//  in Intra Refresh SRAM indicates how many
//  times the corresponding macroblock will
//  be refreshed in the subsequent P-frames.
//  The intra-refresh started in the following
//  P-frame after the frame where the Intra
//  Refresh SRAM entry is loaded with non-zero
//  value. The encoder hardware will load an
//  Intra Refresh SRAM with the value programmed
//  in AIR_COUNTER when the SAD of the
//  corresponding macroblock exceed the limit
//  specified in AIR_SAD_THRESHOLD.
//  Intra Refresh SRAM entries are loaded with
//  0 for macroblocks in Intra frame. 
//  Some options are provided whether to
//  count natural intra MB in a P-frame
//  and whether to 
// This is a P1 that may be broken in AP15.
#define LWE2B7_INTRA_REF_CTRL_0_INTRA_REF_MODE_DEFAULT             (0x00000000)

// Number of MBs per Intra Refresh SRAM entry.
// This parameter specifies the number of
//  macroblocks - 1 per Intra Refresh SRAM
//  entry. The internal Intra Refresh SRAM
//  that is used to store intra refresh count
//  pattern can store maximum of 1620
//  entries/counts. If the total number of
//  macroblocks per frame exceeds 1620, then
//  multiple macroblocks must be mapped to each
//  entry in the Intra Refresh SRAM by
//  programming this parameter. This parameter
//  must be programmed such that the total
//  number of macroblocks in a frame does not
//  exceed 1620 * (MB_PER_ENTRY+1).
// E.g. if MB_PER_ENTRY = 1 then: 
//  entry 0 is for mb 0, 1;
//  entry 1 is for mb 2, 3;
//  entry 2 is for mb 4, 5;
//  ...
//  entry 1619 is for mb 3238, 3239.
#define LWE2B7_INTRA_REF_CTRL_0_MB_PER_ENTRY                       10:8
#define LWE2B7_INTRA_REF_CTRL_0_MB_PER_ENTRY_DEFAULT               (0x00000000)

// Making INTRA_REF_AIR and INTRA_REF_AIR_REFRESH_LIMIT private for now since this is P1
//  feature which may be broken.

// This register is applicable when
//  INTRA_REF_MODE = AIR_MODE.
#define LWE2B7_INTRA_REF_AIR_0                    (0xc1)
// AIR Count.
// This parameter specifies Intra-refresh
//  counter value to load when macroblock
//  SAD > INTRA_REF_AIR_SAD_THRESHOLD.
#define LWE2B7_INTRA_REF_AIR_0_AIR_COUNT                   7:0
#define LWE2B7_INTRA_REF_AIR_0_AIR_COUNT_DEFAULT                   (0x00000000)

// AIR SAD Threshold.
// This parameter specifies the threshold for
//  the SAD value. If the macroblock SAD value
//  exceeds this threshold, then the intra
//  refresh counter value for that MB will be
//  set to REF_AIR_COUNTER_VAL register.
#define LWE2B7_INTRA_REF_AIR_0_AIR_SAD_THRESHOLD                   24:8
#define LWE2B7_INTRA_REF_AIR_0_AIR_SAD_THRESHOLD_DEFAULT           (0x00000000)

// AIR Natural Intra MB control.
#define LWE2B7_INTRA_REF_AIR_0_AIR_NATURAL_I                       31:31
#define LWE2B7_INTRA_REF_AIR_0_AIR_NATURAL_I_IGNORE                       (0)    // // Ignore natural I MBs for updating the intra
//  refresh counters. 

#define LWE2B7_INTRA_REF_AIR_0_AIR_NATURAL_I_VALID                        (1)    // // Count natural I MBs for updating the
//  intra refresh counters.  The intra refresh
//  counter will be decremented by 1 if natural
//  I MB is encountered and clipped to 0.
#define LWE2B7_INTRA_REF_AIR_0_AIR_NATURAL_I_DEFAULT               (0x00000000)


// This register is applicable only when
//  INTRA_REF_MODE = AIR_MODE.
//  Limits on the maximum number of MBs that
//  are forced to I per frame.
#define LWE2B7_INTRA_REF_AIR_REFRESH_LIMIT_0                      (0xc2)
// This value is the maximum number of I MBs
//  that are allowed per I frame. This
//  does not limit the naturally oclwring I MBs,
//  it only limits the I MBs that are 
//  forced by intra-refresh.
#define LWE2B7_INTRA_REF_AIR_REFRESH_LIMIT_0_REFRESH_LIMIT_MAX                     12:0
#define LWE2B7_INTRA_REF_AIR_REFRESH_LIMIT_0_REFRESH_LIMIT_MAX_DEFAULT     (0x00000000)

// AIR Refresh Limit for Natural Intra MB.
#define LWE2B7_INTRA_REF_AIR_REFRESH_LIMIT_0_REFRESH_LIMIT_NATURAL_I                       31:31
#define LWE2B7_INTRA_REF_AIR_REFRESH_LIMIT_0_REFRESH_LIMIT_NATURAL_I_IGNORE                       (0)    // // Ignore natural I MBs when counting the
//  I MBs which will be compared to the
//  REFRESH_LIMIT_MAX limit.

#define LWE2B7_INTRA_REF_AIR_REFRESH_LIMIT_0_REFRESH_LIMIT_NATURAL_I_VALID                        (1)    // // Count natural I MBs when counting the
//  I MBs which will be compared to the
//  REFRESH_LIMT_MAX limit.
#define LWE2B7_INTRA_REF_AIR_REFRESH_LIMIT_0_REFRESH_LIMIT_NATURAL_I_DEFAULT   (0x00000000)


// This register specifies the minimum intra
//  refresh count re-initialization values for
//  I and P frames in the Intra Refresh Default
//  mode. This register also specifies the
//  and re-initialization values for I and P
//  frames in the Intra Refresh Pattern mode.
// This register is programmed only once. 
#define LWE2B7_INTRA_REF_MIN_COUNTER_0                    (0xc3)
// Minimum for I-Frame.
// This parameter specifies the minimum intra
//  refresh count re-initialization values for
//  I-frame in the Intra Refresh Default mode.
// This parameter also specifies the refresh
//  count re-initialization values for I-frame
//  in the Intra Refresh Pattern mode.
#define LWE2B7_INTRA_REF_MIN_COUNTER_0_MIN_FOR_IFRAME                      7:0
#define LWE2B7_INTRA_REF_MIN_COUNTER_0_MIN_FOR_IFRAME_DEFAULT      (0x00000000)

// Minimum for P-frame.
// This parameter specifies the minimum intra
//  refresh count re-initialization values for
//  P-frame in the Intra Refresh Default mode. 
// This parameter also specifies the refresh
//  count re-initialization values for I-frame
//  in the Intra Refresh Pattern mode.
#define LWE2B7_INTRA_REF_MIN_COUNTER_0_MIN_FOR_PFRAME                      15:8
#define LWE2B7_INTRA_REF_MIN_COUNTER_0_MIN_FOR_PFRAME_DEFAULT      (0x00000000)

// This register specifies the delta values
//  between maximum and minimum intra refresh
//  count re-initialization values for I and P
//  frames in the Intra Refresh Default mode.
// This register is not used for Intra Refresh
//  Pattern mode.
#define LWE2B7_INTRA_REF_DELTA_COUNTER_0                  (0xc4)
// Delta for I-frame.
// This register specifies the delta values
//  between maximum and minimum intra refresh
//  count re-initialization values for I Frame
//  in the Intra Refresh Default mode.
//  The value programmed should be >= 4
#define LWE2B7_INTRA_REF_DELTA_COUNTER_0_DELTA_FOR_IFRAME                  7:0
#define LWE2B7_INTRA_REF_DELTA_COUNTER_0_DELTA_FOR_IFRAME_DEFAULT  (0x00000000)

// Delta for P-frame.
// This register specifies the delta values
//  between maximum and minimum intra refresh
//  count re-initialization values for P Frame
//  in the Intra Refresh Default mode.
//  The value programmed should be >= 4
#define LWE2B7_INTRA_REF_DELTA_COUNTER_0_DELTA_FOR_PFRAME                  15:8
#define LWE2B7_INTRA_REF_DELTA_COUNTER_0_DELTA_FOR_PFRAME_DEFAULT  (0x00000000)

// This register specifies the address and
//  count of 4-byte words to be loaded into
//  the Intra Refresh pattern SRAM. Software
//  writes to this register then follows it
//  by a number of writes to the counter
//  register as specified by the
//  NUM_DWORDS_TO_WRITE field.
//  Size of this RAM is 408, 32-bit entries.
#define LWE2B7_INTRA_REF_LOAD_CMD_0                       (0xc5)
// Number of 4-byte words of Data to Write
//  to Intra Refresh SRAM.
// This parameter specifies the number of
//  4-byte words of Intra Refresh count data
//  to be written to the Intra Refresh pattern
//  SRAM.
#define LWE2B7_INTRA_REF_LOAD_CMD_0_NUM_DWORDS_TO_WRITE                    8:0
#define LWE2B7_INTRA_REF_LOAD_CMD_0_NUM_DWORDS_TO_WRITE_DEFAULT    (0x00000000)

// Intra Refresh SRAM 4-byte Write Offset.
//  This value specifies the 4-byte aligned
//  address where the 4-byte Intra Refresh
//  count data will be written to the pattern
//  SRAM.  This address offset is in multiple
//  of 4-byte words.
#define LWE2B7_INTRA_REF_LOAD_CMD_0_SRAM_DWORD_WRITE_OFFS                  26:18
#define LWE2B7_INTRA_REF_LOAD_CMD_0_SRAM_DWORD_WRITE_OFFS_DEFAULT  (0x00000000)


// This register contains the Intra Refresh
//  count data to be written to the Intra
//  Refresh pattern SRAM.  It must be written
//  after a write to the INTRA_REF_LOAD_CMD
//  register a number of times equal to the
//  NUM_DWORDS_TO_WRITE field.
#define LWE2B7_INTRA_REF_LOAD_DATA_0                      (0xc6)
// Intra Refresh SRAM Load Data.
// This is the 4-byte data word to be written
//  to Intra Refresh SRAM.
#define LWE2B7_INTRA_REF_LOAD_DATA_0_SRAM_LOAD_DATA_DWORD                  31:0
#define LWE2B7_INTRA_REF_LOAD_DATA_0_SRAM_LOAD_DATA_DWORD_DEFAULT  (0x00000000)

// This register specifies the address and data
//  for a single update of the Intra Refresh
//  pattern SRAM.
#define LWE2B7_INTRA_REF_LOAD_ONE_CMD_0                   (0xc7)
// Intra Refresh SRAM Load Data Byte.
// This parameter specifies the data byte
//  value to be written into Intra Refresh
//  pattern SRAM at the address specified by
//  SRAM_BYTE_WRITE_OFFS.
#define LWE2B7_INTRA_REF_LOAD_ONE_CMD_0_SRAM_LOAD_DATA_BYTE                7:0
#define LWE2B7_INTRA_REF_LOAD_ONE_CMD_0_SRAM_LOAD_DATA_BYTE_DEFAULT        (0x00000000)

// Intra Refresh SRAM Byte Write Offset
// This parameter specifies the address of the
//  Intra Refresh SRAM where the write data
//  byte SRAM_LOAD_DATA_BYTE will be written to.
#define LWE2B7_INTRA_REF_LOAD_ONE_CMD_0_SRAM_BYTE_WRITE_OFFS                       26:16
#define LWE2B7_INTRA_REF_LOAD_ONE_CMD_0_SRAM_BYTE_WRITE_OFFS_DEFAULT       (0x00000000)





// 0x0C8-0x0CC
//
// Intra Refresh SRAM Read - this may be used for context saving or for diagnostics.
//

// This register specifies the address and
//  count of 4-byte words of Intra Refresh
//  count data  to be read from the Intra
//  Refresh pattern SRAM. Software writes to
//  this register then follows it by a number
//  of reads from INTRA_REF_READ_DATA
//  register as specified by the
//  NUM_DWORDS_TO_READ field.
// If intra refresh is enabled then intra
//  refresh RAM content must be saved during
//  context save in order to be able to resume
//  the context at a later time.
// WARNING: After writing this register SW needs
//  to insert 1 cmd (possibly a read of this register)
//  before doing the actual reads by reading the 
//  INTRA_REF_READ_DATA register
#define LWE2B7_INTRA_REF_READ_CMD_0                       (0xcd)
// Number of words to read from Intra Refresh
//  SRAM.
// This value specifies the number of 4-byte
//  words of Intra Refresh count data to be
//  read from the Intra Refresh pattern SRAM.
#define LWE2B7_INTRA_REF_READ_CMD_0_NUM_DWORDS_TO_READ                     8:0
#define LWE2B7_INTRA_REF_READ_CMD_0_NUM_DWORDS_TO_READ_DEFAULT     (0x00000000)

// Intra Refresh SRAM Read Offset.
// This value specifies the 4-byte aligned
//  address where the 4-byte Intra Refresh
//  count data words will be read from the
//  Intra Refresh pattern SRAM.
#define LWE2B7_INTRA_REF_READ_CMD_0_SRAM_DWORD_READ_OFFS                   26:18
#define LWE2B7_INTRA_REF_READ_CMD_0_SRAM_DWORD_READ_OFFS_DEFAULT   (0x00000000)

// This register contains the 4-byte Intra
//  Refresh count data which is read from the
//  Intra Refresh pattern SRAM.  This register
//  must be read after a write to the
//  INTRA_REF_READ_CMD register a number of
//  times equal to the value written to
//  NUM_DWORDS_TO_READ field.
#define LWE2B7_INTRA_REF_READ_DATA_0                      (0xce)
// Intra Refresh SRAM Read Data.
// This is the 4-byte read data from Intra
//  Refresh SRAM.
#define LWE2B7_INTRA_REF_READ_DATA_0_SRAM_READ_DATA_DWORD                  31:0
#define LWE2B7_INTRA_REF_READ_DATA_0_SRAM_READ_DATA_DWORD_DEFAULT  (0x00000000)

// This register specifies the number of Intra
//  macroblocks in the last encoded P-frame.
//  This register is not updated when an I-frame
//  is encoded.
//  This information may be used to assist in
//  dynamic intra-refresh decision.
#define LWE2B7_NUM_INTRAMB_0                      (0xcf)
// Number of Natural Intra Macroblocks.
// This parameter specifies the number of Intra
//  macroblocks that are naturally oclwrring
//  in the last encoded P-frame due to intra/
//  inter macroblock decision.
#define LWE2B7_NUM_INTRAMB_0_NUM_INTRA_NATURAL                     12:0
#define LWE2B7_NUM_INTRAMB_0_NUM_INTRA_NATURAL_DEFAULT             (0x00000000)

// Total Number of Intra Macroblocks.
// This parameter specifies the total number of
//  Intra macroblocks (both naturally oclwrring
//  intra and intra macroblocks due to intra
//  refresh) in the last encoded P-frame.
#define LWE2B7_NUM_INTRAMB_0_TOTAL_INTRA                   28:16
#define LWE2B7_NUM_INTRAMB_0_TOTAL_INTRA_DEFAULT                   (0x00000000)

// Offset 0x0D0
//
// I-prediction, reconstruction loop, deblock registers.
//

// This register is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
// This register specifies intra prediction
//  4x4 and 16x16 biases.
// This register is programmed only once.
#define LWE2B7_INTRA_PRED_BIAS_0                  (0xd0)
// Favor 4x4 bias.
// This parameter is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
// This parameter specifies intra prediction
//  4x4 biases.
#define LWE2B7_INTRA_PRED_BIAS_0_FAVOR_INTRA_4X4                   14:0
#define LWE2B7_INTRA_PRED_BIAS_0_FAVOR_INTRA_4X4_DEFAULT           (0x00000000)

// Favor 16x16 bias.
// This parameter is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
// This parameter specifies intra prediction
//  16x16 biases.
#define LWE2B7_INTRA_PRED_BIAS_0_FAVOR_INTRA_16X16                 30:16
#define LWE2B7_INTRA_PRED_BIAS_0_FAVOR_INTRA_16X16_DEFAULT         (0x00000000)

// Disable Intra Prediction Bias Control.
// This parameter is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
// Recommended value for H264 encoding is
//  ENABLE.
#define LWE2B7_INTRA_PRED_BIAS_0_INTRA_4X4_16X16_INTERNAL_BIAS                     31:31
#define LWE2B7_INTRA_PRED_BIAS_0_INTRA_4X4_16X16_INTERNAL_BIAS_ENABLE                     (0)    // // Internal intra prediction bias enabled.

#define LWE2B7_INTRA_PRED_BIAS_0_INTRA_4X4_16X16_INTERNAL_BIAS_DISABLE                    (1)    // // Internal intra prediction bias disabled.
#define LWE2B7_INTRA_PRED_BIAS_0_INTRA_4X4_16X16_INTERNAL_BIAS_DEFAULT     (0x00000000)


// This register specifies the most probable
//  bias for intra mode and skip biases and
//  Intra prediction control.
// This register is programmed only once.
#define LWE2B7_MISC_MODE_BIAS_0                   (0xd1)
// Favor Most Probable Intra Mode.
// This parameter is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
// This parameter specifies the bias toward
//  most probable intra mode.
// Recommended value for H264 encoding is 0.
#define LWE2B7_MISC_MODE_BIAS_0_FAVOR_MOST_PROB_INTRAMODE                  14:0
#define LWE2B7_MISC_MODE_BIAS_0_FAVOR_MOST_PROB_INTRAMODE_DEFAULT  (0x00000000)

// Internal Most Probable Intra Mode
//  Bias Control.
// This parameter is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
// Recommended value for H264 encoding is
//  ENABLE.
#define LWE2B7_MISC_MODE_BIAS_0_MOST_PROB_INTRAMODE_INTERNAL_BIAS                  15:15
#define LWE2B7_MISC_MODE_BIAS_0_MOST_PROB_INTRAMODE_INTERNAL_BIAS_ENABLE                  (0)    // // Internal most probable intra mode bias
//  enabled.

#define LWE2B7_MISC_MODE_BIAS_0_MOST_PROB_INTRAMODE_INTERNAL_BIAS_DISABLE                 (1)    // // Internal most probable intra mode bias
//  disabled.
#define LWE2B7_MISC_MODE_BIAS_0_MOST_PROB_INTRAMODE_INTERNAL_BIAS_DEFAULT  (0x00000000)

// Intra Pred UV Decision Mode.
// This parameter is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
// This parameter specifies chroma mode
//  intra prediction options.
#define LWE2B7_MISC_MODE_BIAS_0_IPRED_UV_DECISION_MODE                     16:16
#define LWE2B7_MISC_MODE_BIAS_0_IPRED_UV_DECISION_MODE_UANDV                      (0)    // // Use minimum cost (I-8x8U + I-8x8V) mode for
//  chroma mode.

#define LWE2B7_MISC_MODE_BIAS_0_IPRED_UV_DECISION_MODE_UONLY                      (1)    // // Use minimum cost I-8x8 U mode for chroma
//  mode.
#define LWE2B7_MISC_MODE_BIAS_0_IPRED_UV_DECISION_MODE_DEFAULT     (0x00000000)

// I vs P Decision Mode.
// This parameter is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
#define LWE2B7_MISC_MODE_BIAS_0_IP_DECISION_MODE                   20:20
#define LWE2B7_MISC_MODE_BIAS_0_IP_DECISION_MODE_ACTUAL                   (0)    // // Use actual intra mode cost for I vs P
//  decision.

#define LWE2B7_MISC_MODE_BIAS_0_IP_DECISION_MODE_APPROX                   (1)    // // Use approximate intra mode cost for I vs P
//  decision.
// When in ME_BYPASS mode, approximate intra
//  mode cost is always used for I vs P
//  decision regardless of IP_DECISION_MODE.
#define LWE2B7_MISC_MODE_BIAS_0_IP_DECISION_MODE_DEFAULT           (0x00000000)

// I vs P Decision Control.
// This parameter is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
#define LWE2B7_MISC_MODE_BIAS_0_IP_DECISION_CTRL                   21:21
#define LWE2B7_MISC_MODE_BIAS_0_IP_DECISION_CTRL_USE_16X16                        (0)    // // Use 16x16 modes only for best I mode cost
//  callwlation for I/P decision.

#define LWE2B7_MISC_MODE_BIAS_0_IP_DECISION_CTRL_USE_16X16_4X4                    (1)    // // Use both 16x16 and 4x4 modes for best I mode
//  cost callwlation for I/P decision.
//  In this case, IPRED_4X4_SRCH_CTRL determine
//  the number of 4x4 modes searched.
#define LWE2B7_MISC_MODE_BIAS_0_IP_DECISION_CTRL_DEFAULT           (0x00000000)

// I/P Decision Intra 4x4 Search Mode.
// This parameter is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
// This parameter defines the search order used
//  for best intra 4x4 mode for I/P decision
//  when IP_DECISION_CTRL is set to
//  USE_16X16_4X4. The actual 4x4
//  intra mode depends on the availability of
//  that mode, based on INTRA_PRED_MODE field in
//  MOT_SEARCH_CTRL method, and
//  IPRED_4X4_SRCH_CTRL field. 
#define LWE2B7_MISC_MODE_BIAS_0_IP_4X4_SRCH_MODE                   22:22
#define LWE2B7_MISC_MODE_BIAS_0_IP_4X4_SRCH_MODE_LINEAR                   (0)    // // Linear search. Intra 4x4 modes are searched
//  in linear order from mode 0.

#define LWE2B7_MISC_MODE_BIAS_0_IP_4X4_SRCH_MODE_OPTIMAL                  (1)    // // Optimized search. Intra 4x4 modes are
//  searched in a pre-determined order.
#define LWE2B7_MISC_MODE_BIAS_0_IP_4X4_SRCH_MODE_DEFAULT           (0x00000000)

// I/P Decision Intra 4x4 Search Control.
// This parameter is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
// This field defines the number of 4x4 modes
//  searched for I/P decision when
//  IP_DECISION_CTRL is set to USE_16X16_4X4.
// Setting this parameter to 0 is the same as
//  setting IP_DECISION_CTRL to USE_16X16.
#define LWE2B7_MISC_MODE_BIAS_0_IP_4X4_SRCH_CTRL                   26:23
#define LWE2B7_MISC_MODE_BIAS_0_IP_4X4_SRCH_CTRL_DEFAULT           (0x00000000)

// This register is used to control quantized
//  coefficient post-processing.
// Lwrrently this is applicable only for H.264
//  encoding.
#define LWE2B7_QUANTIZATION_CONTROL_0                     (0xd2)
// Quantized Coefficient Saturation limit.
// This bit is effective only for H.264 encoding
//  and is not used for MPEG4/H.263 encoding.
#define LWE2B7_QUANTIZATION_CONTROL_0_COEF_SATURATION_LIMIT                0:0
#define LWE2B7_QUANTIZATION_CONTROL_0_COEF_SATURATION_LIMIT_MIN_LIMIT                     (0)    // // Coefficients are saturated at min limit
//  regardless of their position.

#define LWE2B7_QUANTIZATION_CONTROL_0_COEF_SATURATION_LIMIT_LIMIT                 (1)    // // Coefficients are saturated at proper limit
//  depending on their position as defined by
//  the H.264 standard.
// This setting is lwrrently not implemented.
#define LWE2B7_QUANTIZATION_CONTROL_0_COEF_SATURATION_LIMIT_DEFAULT        (0x00000000)

// Quantized Coefficient Saturation rewind.
// This bit is effective only for H.264 encoding
//  and is not used for MPEG4/H.263 encoding.
// This functionality is not implemented.
#define LWE2B7_QUANTIZATION_CONTROL_0_COEF_SATURATION_REWIND                       3:3
#define LWE2B7_QUANTIZATION_CONTROL_0_COEF_SATURATION_REWIND_NO_REWIND                    (0)    // // Quantized coefficients are saturated as
//  specified by COEF_SATURATION_LIMIT when
//  they exceed the limits.

#define LWE2B7_QUANTIZATION_CONTROL_0_COEF_SATURATION_REWIND_REWIND                       (1)    // // When quantized coefficients exceed the limit
//  specified by COEF_SATURATION_LIMIT,
//  quantization process is rewound with higher
//  quant values. If there are still quantized
//  coefficients that exceed the limit specified
//  by COEF_SATURATION_LIMIT, then they will
//  be saturated to the limit values.
#define LWE2B7_QUANTIZATION_CONTROL_0_COEF_SATURATION_REWIND_DEFAULT       (0x00000000)

// Coefficient Saturation Quant Rewind Delta.
// This field is effective only for H.264
//  encoding and is not used for MPEG4/H.263
//  encoding.
// This field is effective only when
//  COEF_SAT_REWIND is set to REWIND.
// This field specifies the increase of quant
//  values when quantized coefficients exceed
//  their limit values and quantization needs
//  to be rewound. Valid value is between 1
//  and 16.
// Programmed value = actual value - 1.
// This functionality is not implemented.
#define LWE2B7_QUANTIZATION_CONTROL_0_COEF_SAT_QP_DELTA                    7:4
#define LWE2B7_QUANTIZATION_CONTROL_0_COEF_SAT_QP_DELTA_DEFAULT    (0x00000000)

// New registers to control H264 Quantization Post Processor (QPP) for APxx products.
// The following defines Quantization Post Processor (QPP) registers which controls the output
// of the quantization process to increase compression efficiency. 
//
// In a typical case, post processing is done only on a 8X8 block basis by controlling the
// QPP_MODE bit. For maximum compression efficiency, QPP_MODE should be set to QPP_16X16. In
// QPP_16X16 mode the LUMA_16X16_COST should be set greater than or equal to LUMA_8X8_COST to
// gain any performance benefit over QPP_8X8 mode.
// For best results, turn QPP ON only when rate control is ON.
// Recommended settings when QPP is ON:
// (a) Frame Size <= QCIF: qpp_mode 0 qpp_cost 1,1,X
// (b) Frame Size >  QCIF: qpp_mode 1 qpp_cost 2,2,3 qpp_run_vect 1407

// This register is effective only for H.264
//  encoding and is not used for MPEG4/H.263
//  encoding.
#define LWE2B7_QPP_CTRL_0                 (0xd3)
// Quantization post proc enable.
// Quantization post processing module post 
//  processes quantized coefficients in inter
//  mode and discards small and relatively
//  expensive luma or chroma coefficients to
//  increase compression performance.
#define LWE2B7_QPP_CTRL_0_QPP_ENABLE                       0:0
#define LWE2B7_QPP_CTRL_0_QPP_ENABLE_DISABLE                      (0)    // // Disable quantization post processing

#define LWE2B7_QPP_CTRL_0_QPP_ENABLE_ENABLE                       (1)    // // Enable  quantization post processing
#define LWE2B7_QPP_CTRL_0_QPP_ENABLE_DEFAULT                       (0x00000000)

// QPP Mode.
#define LWE2B7_QPP_CTRL_0_QPP_MODE                 1:1
#define LWE2B7_QPP_CTRL_0_QPP_MODE_QPP_8X8                        (0)    // // Discard coefficients at 8X8 block level only.

#define LWE2B7_QPP_CTRL_0_QPP_MODE_QPP_16X16                      (1)    // // Discard coefficients at both 16x16 block
//  level and at 8X8 block level.
#define LWE2B7_QPP_CTRL_0_QPP_MODE_DEFAULT                         (0x00000000)

// Luma Coefficient Cost for 8X8 block.
// This field sets up the threshold for
//  discarding luma coefficients in a 8x8 block.
//  The maximum coefficient level of a candidate 
//  block for quantization post processing must  
//  be <= 1. If the callwlated coefficient cost 
//  of the candidate block is more than 
//  LUMA_8X8_COST, then the coefficients are not 
//  discarded.
// 4'b0000 = All the coefficients are discarded.
// 4'b0001 = Threshold is 1.
// 4'b0010 = Threshold is 2.
// ....... 
// 4'b1111 = Threshold is 15.
#define LWE2B7_QPP_CTRL_0_LUMA_8X8_COST                    7:4
#define LWE2B7_QPP_CTRL_0_LUMA_8X8_COST_DEFAULT                    (0x0000000f)

// Chroma Coefficient Cost for 8X8 block.
// This field sets up the threshold for
//  discarding chroma coefficients in a 8x8 block.
//  The maximum coefficient level of a candidate
//  block for quantization post processing must
//  be <= 1. If the callwlated coefficient cost
//  of the candidate block is more than 
//  CHROMA_8X8_COST, then the coefficients are not 
//  discarded.
// 4'b0000 = All the coefficients are discarded.
// 4'b0001 = Threshold is 1.
// 4'b0010 = Threshold is 2.
// ....... 
// 4'b1111 = Threshold is 15.
#define LWE2B7_QPP_CTRL_0_CHROMA_8X8_COST                  11:8
#define LWE2B7_QPP_CTRL_0_CHROMA_8X8_COST_DEFAULT                  (0x0000000f)

// Luma Coeff Cost for 16X16 block.
// This is the threshold for discarding luma
//  coefficients in a 16X16 block. If the total
//  coeff cost of a 16X16 MB is less than or 
//  equal to threshold, then the entire 16X16
//  macro-block is discarded.
//  The maximum coefficient level of a candidate 
//  16x16 block for quantization post processing 
//  must be <= 1. If the callwlated coefficient 
//  cost of the candidate block is more than 
//  LUMA_16X16_COST, then the coefficients are not 
//  discarded.
// The total coefficient cost for a 16X16 luma
//  MB is the sum of the LUMA_8X8_COST of
//  individual 8X8 blocks, where those 8X8
//  blocks coefficient cost is more than the
//  LUMA_8X8_COST.
// LUMA_16X16_COST should always be more 
//  than LUMA_8X8_COST.
// 4'b0000 = All the coefficients are discarded.
// 4'b0001 = Threshold is 1.
// 4'b0010 = Threshold is 2.
// .
// 4'b1111 = Threshold is 15.
#define LWE2B7_QPP_CTRL_0_LUMA_16X16_COST                  15:12
#define LWE2B7_QPP_CTRL_0_LUMA_16X16_COST_DEFAULT                  (0x0000000f)

// RUN vector for  8X8 block.
// Reset value 16'b0000_0101_0110_1011=1387.
// RUN = 3,2,2,1,1,1,0,0,0,.....(0 for RUN >9)
// These bits set up the RUN vector for
//  callwlating coefficient cost when
//  coefficient level =1.
// Bits 17:16 = coefficient cost when RUN=0.
// Bits 19:18 = coefficient cost when RUN=1.
// Bits 21:20 = coefficient cost when RUN=2.
// Bits 23:22 = coefficient cost when RUN=3.
// Bits 25:24 = coefficient cost when RUN=4.
// Bits 27:26 = coefficient cost when RUN=5.
// Bits 28 = coefficient cost when RUN=6.
// Bits 29 = coefficient cost when RUN=7.
// Bits 30 = coefficient cost when RUN=8.
// Bits 31 = coefficient cost when RUN=9.
#define LWE2B7_QPP_CTRL_0_RUN_VECTOR                       31:16
#define LWE2B7_QPP_CTRL_0_RUN_VECTOR_DEFAULT                       (0x0000056b)

// Reserve 4 registers for future expansion of QPP.

// 0x0D4-0x0D7

// This register specifies control for IPCM
//  macroblock for H.264 encoding.
#define LWE2B7_IPCM_CTRL_0                        (0xd8)
// IPCM Rewind Control.
// This parameter is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
// This register specifies control for IPCM
//  rewind process.
#define LWE2B7_IPCM_CTRL_0_IPCM_REWIND                     0:0
#define LWE2B7_IPCM_CTRL_0_IPCM_REWIND_ENABLE                     (0)    // // IPCM rewind enabled. This setting is
//  recommended for compliance to H.264
//  standard.

#define LWE2B7_IPCM_CTRL_0_IPCM_REWIND_DISABLE                    (1)    // // IPCM rewind disabled. This setting may be
//  be used for diagnostics/debugging.
// When IPCM rewind is disabled MB data processing could 
// potentially overflow MPE's internal macroblock 
// assembly buffer.The limit of this buffer is 1k byte, 
// while IPCM limit is 384 bytes. This overflow can only happen for many very large coeff 
// (low QP setting) and in debug mode only. The symptom is header corruption. 
// Therefore, when IPCM rewind is disabled for debug,
// it is recommanded to not to use use low QP values.
#define LWE2B7_IPCM_CTRL_0_IPCM_REWIND_DEFAULT                     (0x00000000)

// IPCM 0x00 Coefficient Mapping.
// This parameter is applicable for H.264
//  encoding only and is not applicable for
//  MPEG4/H.263 encoding.
// This parameter specifies mapping for zero
//  coefficients. This should be set to 0x01
//  for H.264 standard compliance.
#define LWE2B7_IPCM_CTRL_0_IPCM_ZERO_COEF_MAP                      15:8
#define LWE2B7_IPCM_CTRL_0_IPCM_ZERO_COEF_MAP_DEFAULT              (0x00000000)

// Offset 0x0E0
//
// VLC Control Registers.
//

#define LWE2B7_PACKET_HEC_0                       (0xe0)
// HEC Marker Frequency.
// This parameter is applicable for MPEG4/H.263
//  encoding only and not used for H.264
//  encoding.
// These bits specify how often the HEC Marker
//  has to be inserted in the header.
#define LWE2B7_PACKET_HEC_0_HEC_FREQ                       2:0
#define LWE2B7_PACKET_HEC_0_HEC_FREQ_DISABLE                      (0)    // // HEC marker is not inserted.

#define LWE2B7_PACKET_HEC_0_HEC_FREQ_EVERY1                       (1)    // // HEC marker is inserted after every
//  resynchronization marker.

#define LWE2B7_PACKET_HEC_0_HEC_FREQ_EVERY2                       (2)    // // HEC marker is inserted after every second
//  resynchronization marker.

#define LWE2B7_PACKET_HEC_0_HEC_FREQ_EVERY3                       (3)    // // HEC marker is inserted after every third
//  resynchronization marker.

#define LWE2B7_PACKET_HEC_0_HEC_FREQ_EVERY4                       (4)    // // HEC marker is inserted after every fourth
//  resynchronization marker.

#define LWE2B7_PACKET_HEC_0_HEC_FREQ_EVERY5                       (5)    // // HEC marker is inserted after every fifth
//  resynchronization marker.

#define LWE2B7_PACKET_HEC_0_HEC_FREQ_EVERY6                       (6)    // // HEC marker is inserted after every sixth
//  resynchronization marker.

#define LWE2B7_PACKET_HEC_0_HEC_FREQ_EVERY7                       (7)    // // HEC marker is inserted after every seventh
//  resynchronization marker.
#define LWE2B7_PACKET_HEC_0_HEC_FREQ_DEFAULT                       (0x00000000)

// Packet length.
// This bit is applicable for MPEG4/H.263
//  encoding only.
//  For H.264 encoding, packet length for
//  packetization is specified in
//  PACKET_COUNT_H264 field of PACKET_CTRL_H264
//  register.
// In APxx, if ME_BYPASS is set to ENABLE,
//  this parameter is ignored.
// This parameter specifies the maximum number
//  of bytes per packet or the maximum
//  number of macroblocks per packet.
//  The PACKET_CNT_SRC field of the FRAME_CTRL
//  register defines the actual interpretation.
#define LWE2B7_PACKET_HEC_0_PACKET_COUNT                   25:8
#define LWE2B7_PACKET_HEC_0_PACKET_COUNT_DEFAULT                   (0x00000000)

// This bit is applicable for H.264 encoding
//  only and is not used for MPEG4/H.263
//  encoding.
#define LWE2B7_PACKET_CTRL_H264_0                 (0xe1)
// Packet Count Select.
// This bit is applicable for H.264 encoding
//  only.
//  For MPEG4/H.263 encoding, the control for
//  actual packetization is specified in
//  FRAME_CTRL register.
// In APxx, if ME_BYPASS is set to ENABLE,
//  this bit is ignored.
// This bit specifies how the size of the
//  encoded bitstream packet is determined.
#define LWE2B7_PACKET_CTRL_H264_0_PACKET_CNT_SRC_H264                      0:0
#define LWE2B7_PACKET_CTRL_H264_0_PACKET_CNT_SRC_H264_BITS                        (0)    // // The size of encoded bitstream packet is
//  determined by the number of bits in the
//  bitstream as specified by PACKET_COUNT_H264
//  field in this register.

#define LWE2B7_PACKET_CTRL_H264_0_PACKET_CNT_SRC_H264_MBLK                        (1)    // // The size of encoded bitstream packet is
//  determined by the number of macroblocks
//  specified by PACKET_COUNT_H264 field in
//  this register.
#define LWE2B7_PACKET_CTRL_H264_0_PACKET_CNT_SRC_H264_DEFAULT      (0x00000000)

// Packet Length (for H264)
// This bit is applicable for H.264 encoding
//  only.
//  For MPEG4/H.263 encoding, the control for
//  actual packetization is specified in
//  PACKET_COUNT field of PACKET_HEC register.
// In APxx, if ME_BYPASS is set to ENABLE,
//  this parameter is ignored.
// This parameter specifies the maximum number
//  of bytes per packet if or the maximum
//  number of macroblocks per packet for each
//  slice group.
//  The PACKET_CNT_SRC_264 field of this
//  register defines the actual interpretation.
// Programming this parameter to 0 signifies
//  no packetization inside slice group.
#define LWE2B7_PACKET_CTRL_H264_0_PACKET_COUNT_H264                25:8
#define LWE2B7_PACKET_CTRL_H264_0_PACKET_COUNT_H264_DEFAULT        (0x00000000)

// Bit 16 (DMA_DEST) of DMA_SWAP_CTRL register is deleted.

#define LWE2B7_DMA_SWAP_CTRL_0                    (0xe2)
// VLC DMA Bitstream Byte Swap.
// This parameter control the type of byte swap
//  that will be done for VLC DMA bitstream
//  data. Byte swapping, if specified,  oclwrs
//  after VLC DMA bitstream data is read from
//  the VLC DMA buffer.
#define LWE2B7_DMA_SWAP_CTRL_0_BODY_SWAP                   1:0
#define LWE2B7_DMA_SWAP_CTRL_0_BODY_SWAP_NOSWAP                   (0)    // // No Swap.

#define LWE2B7_DMA_SWAP_CTRL_0_BODY_SWAP_SWAP01_23                        (1)    // // 16-bit Byte Swap (Byte 0 swapped with Byte 1,
//  Byte 2 swapped with Byte 3)

#define LWE2B7_DMA_SWAP_CTRL_0_BODY_SWAP_SWAP03_12                        (2)    // // 4-byte Byte Swap (Byte 0 swapped with Byte 3,
//  Byte 1 swapped with Byte 2)

#define LWE2B7_DMA_SWAP_CTRL_0_BODY_SWAP_SWAP02_13                        (3)    // // Word Swap (Byte 0 swapped with Byte 2,
//  Byte 1 swapped with Byte 3)
#define LWE2B7_DMA_SWAP_CTRL_0_BODY_SWAP_DEFAULT                   (0x00000000)

// VLC DMA Chunk Header Byte Swap.
// This parameter control the type of byte swap
//  that will be done for VLC DMA chunk header
//  data. Byte swapping, if specified,  oclwrs
//  after VLC DMA chunnk header is read from
//  the VLC DMA buffer.
#define LWE2B7_DMA_SWAP_CTRL_0_HEADER_SWAP                 9:8
#define LWE2B7_DMA_SWAP_CTRL_0_HEADER_SWAP_NOSWAP                 (0)    // // No Swap.

#define LWE2B7_DMA_SWAP_CTRL_0_HEADER_SWAP_SWAP01_23                      (1)    // // 16-bit Byte Swap (Byte 0 swapped with Byte 1,
//  Byte 2 swapped with Byte 3)

#define LWE2B7_DMA_SWAP_CTRL_0_HEADER_SWAP_SWAP03_12                      (2)    // // 4-byte Byte Swap (Byte 0 swapped with Byte 3,
//  Byte 1 swapped with Byte 2)

#define LWE2B7_DMA_SWAP_CTRL_0_HEADER_SWAP_SWAP02_13                      (3)    // // Word Swap (Byte 0 swapped with Byte 2,
//  Byte 1 swapped with Byte 3)
#define LWE2B7_DMA_SWAP_CTRL_0_HEADER_SWAP_DEFAULT                 (0x00000000)

//
// Time stamp registers.
//

//  This register is used by the host to write
//  or force initialization of timestamp.
#define LWE2B7_CPU_TIMESTAMP_0                    (0xe3)
// VOP Time Increment.
#define LWE2B7_CPU_TIMESTAMP_0_VOP_TIME_INC                15:0
#define LWE2B7_CPU_TIMESTAMP_0_VOP_TIME_INC_DEFAULT                (0x00000000)

// Modulo Time Base.
#define LWE2B7_CPU_TIMESTAMP_0_MODULO_TIME_BASE                    19:16
#define LWE2B7_CPU_TIMESTAMP_0_MODULO_TIME_BASE_DEFAULT            (0x00000000)

// Timestamp Write Enable.
// This bit controls writing of CPU timestamp.
// Note that this should be set before
//  VOP_TIME_INC and MODULO_TIME_BASE are set
//  because this is being registered with
//  MPEG encoder clock.
#define LWE2B7_CPU_TIMESTAMP_0_TIMESTAMP_WRITE_ENABLE                      31:31
#define LWE2B7_CPU_TIMESTAMP_0_TIMESTAMP_WRITE_ENABLE_DISABLE                     (0)    // // No operation.

#define LWE2B7_CPU_TIMESTAMP_0_TIMESTAMP_WRITE_ENABLE_ENABLE                      (1)    // // New timestamp value provided by CPU is
//  written. This setting should be used only
//  when TIMESTAMP_MODE bit is set to CPU_MODEL.
#define LWE2B7_CPU_TIMESTAMP_0_TIMESTAMP_WRITE_ENABLE_DEFAULT      (0x00000000)

// Deleted TIMESTAMP_INT related field
// Removed the audioclk based timestamp logic from h/w
//  since S/W always programs the correct time stamp (from AP20 and beyond)

// This register is used to 
//  control whether timestamp is written by
//  CPU or generated internally.
// Note that this register exists only to 
//  maintain backward compatibility with AP15.
// Internal mode (VI_MODE) is disabled in h/w
#define LWE2B7_TIMESTAMP_INT_0                    (0xe4)
// Timestamp Mode.
// This bit must be programmed before any other
//  timestamp register is programmed.
#define LWE2B7_TIMESTAMP_INT_0_TIMESTAMP_MODE                      31:31
#define LWE2B7_TIMESTAMP_INT_0_TIMESTAMP_MODE_VI_MODEL                    (0)    // // Internal timestamp logic is enabled and
//  timestamp is sampled when a frame encoding
//  starts.

#define LWE2B7_TIMESTAMP_INT_0_TIMESTAMP_MODE_CPU_MODEL                   (1)    // // Internal timestamp logic is disabled and
//  therefore timestamp must be provided by CPU.
#define LWE2B7_TIMESTAMP_INT_0_TIMESTAMP_MODE_DEFAULT              (0x00000000)

// Deleted audio timestamp related registers
// Reserve 1 registers for backward compatibility


// This register is used to program the
//  timestamp resolution which is used when
//  internal timestamp generation is enabled
//  (TIMESTAMP_MODE is set to VI_MODEL).
#define LWE2B7_TIMESTAMP_RES_0                    (0xe6)
// Timestamp Resolution Count.
// Timestamp resolution in multiple of timestamp
//  source clocks. Timestamp source clock is
//  typically set to the same clock that drives
//  audio sub-system so that video timestamp
//  can be synchronized to audio timestamp.
//  This resolution count will effectively
//  divide the timestamp source clock frequency
//  before it is being used to increment the
//  timestamp speed counter.
#define LWE2B7_TIMESTAMP_RES_0_RESOLUTION_CNT                      15:0
#define LWE2B7_TIMESTAMP_RES_0_RESOLUTION_CNT_DEFAULT              (0x00000000)

// This register specifies the time stamp
//  for the last encoded frame.
#define LWE2B7_TIMESTAMP_LAST_FRAME_0                     (0xe7)
// Timestamp of the Last Frame.
// This field returns the time stamp in
//  milliseconds for the last encoded frame.
#define LWE2B7_TIMESTAMP_LAST_FRAME_0_TIMESTAMP                    31:0
#define LWE2B7_TIMESTAMP_LAST_FRAME_0_TIMESTAMP_DEFAULT            (0x00000000)

//
// VLC status Registers
//

// This register returns the length of the
//  motion and mode of the last frame.
// In AP15, this register is meaningful in to mpeg4/h263 mode only.
// In H264 mode, HW updating this register with 
// appropriate value is not implemented for AP15.
// This register may be written when it is
//  safe to write registers to specify
//  the last pframe's header legnth
//  size for VBR rate control when resuming
//  from context switch. It is not necessary to
//  initialize this register at the beginning
//  of encoding.
// The value of this register is updated
//  automatically by the encoder hardware.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_LENGTH_OF_MOTION_MODE_0                    (0xe8)
//  Motion Vector and Mode Bit Length
//  This is the size of the motion vector and
//  mode for the last frame in bits
#define LWE2B7_LENGTH_OF_MOTION_MODE_0_FRAME_MOD_LEN                       23:0
#define LWE2B7_LENGTH_OF_MOTION_MODE_0_FRAME_MOD_LEN_DEFAULT       (0x00000000)

// Last Frame Type.
// This bit specifies whether the last frame
//  was encoded either an I or a P frame.
#define LWE2B7_LENGTH_OF_MOTION_MODE_0_PREV_FRAME_TYPE                     27:27
#define LWE2B7_LENGTH_OF_MOTION_MODE_0_PREV_FRAME_TYPE_INTRA                      (0)    // // Last frame was an I frame.

#define LWE2B7_LENGTH_OF_MOTION_MODE_0_PREV_FRAME_TYPE_INTER                      (1)    // // Last frame was a P frame.
#define LWE2B7_LENGTH_OF_MOTION_MODE_0_PREV_FRAME_TYPE_DEFAULT     (0x00000000)


// This register returns the status of the
//  encoded frame and VLC.
#define LWE2B7_FRAME_VLC_STATUS_0                 (0xe9)
// Macroblock Texture Bit Length.
// This returns the size of the texture for the
//  current macroblock in bits.
// In AP15, this register field is meaningful in to mpeg4/h263 mode only.
// In H264 mode, HW updating this register field with 
// appropriate value is not implemented for AP15.
#define LWE2B7_FRAME_VLC_STATUS_0_MB_TEX_LEN                       15:0
#define LWE2B7_FRAME_VLC_STATUS_0_MB_TEX_LEN_DEFAULT               (0x00000000)

// VLC Total Packets.
// This returns the total number of packets
//  produced by the VLC for last encoded frame.
// In AP15, this register field is meaningful in to mpeg4/h263 mode only.
// In H264 mode, HW updating this register field with 
// appropriate value is not implemented for AP15.
#define LWE2B7_FRAME_VLC_STATUS_0_VLC_TOTAL_PACKETS                24:16
#define LWE2B7_FRAME_VLC_STATUS_0_VLC_TOTAL_PACKETS_DEFAULT        (0x00000000)

// Macroblock IDCT Mismatch.
// This bit is set when the encoder detects
//  that the output of the IDCT stage is all
//  zeroes.  It changes every macroblock.
// This register field is applicable to mpeg4/h263 mode only.
// In H264 mode, this register field value has no meaning.
#define LWE2B7_FRAME_VLC_STATUS_0_IDCT_MISMATCH_STAT                       25:25
#define LWE2B7_FRAME_VLC_STATUS_0_IDCT_MISMATCH_STAT_DEFAULT       (0x00000000)

// Last Frame Type.
// This bit specifies whether the last frame
//  was encoded either an I or a P frame.
#define LWE2B7_FRAME_VLC_STATUS_0_FRAME_TYPE                       26:26
#define LWE2B7_FRAME_VLC_STATUS_0_FRAME_TYPE_INTRA                        (0)    // // Last frame was an I frame.

#define LWE2B7_FRAME_VLC_STATUS_0_FRAME_TYPE_INTER                        (1)    // // Last frame was a P frame.
#define LWE2B7_FRAME_VLC_STATUS_0_FRAME_TYPE_DEFAULT               (0x00000000)

// Offset 0x100
//
// Rate Control Registers
//
// Bit 16 (INIT_QP_I_ENABLE) of I_RATE_CTRL register is deleted.

// This register specifies all the control bits
//  needed for processing the I-frame in the
//  Rate Control module.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
#define LWE2B7_I_RATE_CTRL_0                      (0x100)
// Rate Control Enable for I-Frame
// This bit is used to enable/disable the rate
//  control for I-frame.
// This is applicable for VBR rate control only
//  and therefore affect only MPEG4/H.263
//  encoding only.
#define LWE2B7_I_RATE_CTRL_0_RATE_CTRL_I_ENABLE                    0:0
#define LWE2B7_I_RATE_CTRL_0_RATE_CTRL_I_ENABLE_DISABLE                   (0)    // // Rate Control is disabled for I-frame.

#define LWE2B7_I_RATE_CTRL_0_RATE_CTRL_I_ENABLE_ENABLE                    (1)    // // Rate Control is enabled for I-frame.
#define LWE2B7_I_RATE_CTRL_0_RATE_CTRL_I_ENABLE_DEFAULT            (0x00000000)

// Initial Quantization Scale (Qp) for I-Frame.
// This field is applicable for MPEG4/H.263
//  encoding only. For H.264 encoding,
//  I_INIT_QP_H264 parameter in PIC_INIT_Q
//  register is used instead.
// For MPEG4/H.263 encoding, this parameter
//  specifies the Qp for the whole first
//  I-frame.
#define LWE2B7_I_RATE_CTRL_0_I_INIT_QP                     5:1
#define LWE2B7_I_RATE_CTRL_0_I_INIT_QP_DEFAULT                     (0x00000000)

// Minimum Quantization Scale for I-Frame.
// This field is applicable for MPEG4/H.263
//  encoding only. For H.264 encoding,
//  MAX_MIN_QP_I register is used instead.
// For MPEG4/H.263 encoding, this parameter
//  specifies the minimum Qp for I-frame.
#define LWE2B7_I_RATE_CTRL_0_I_MIN_QP                      10:6
#define LWE2B7_I_RATE_CTRL_0_I_MIN_QP_DEFAULT                      (0x00000000)

// Maximum Quantization Scale for I-Frame.
// This field is applicable for MPEG4/H.263
//  encoding only. For H.264 encoding,
//  MAX_MIN_QP_I register is used instead.
// For MPEG4/H.263 encoding, this parameter
//  specifies the maximum Qp for I-frame.
#define LWE2B7_I_RATE_CTRL_0_I_MAX_QP                      15:11
#define LWE2B7_I_RATE_CTRL_0_I_MAX_QP_DEFAULT                      (0x00000000)

// I-Frame Skip Control.
// This bit specifies whether I-frames
//  should be skipped when output buffer
//  overflows.
//  This bit is effective for both VBR and
//  CBR rate control.
//  Frame skipping shouldn't be enabled when
//  rate control is turned off
#define LWE2B7_I_RATE_CTRL_0_I_SKIP_ENABLE                 17:17
#define LWE2B7_I_RATE_CTRL_0_I_SKIP_ENABLE_DISABLE                        (0)    // // Encode I-frame even when there is VLC DMA
//  buffer overflows. This setting can be
//  used for variable bitrate encoding.

#define LWE2B7_I_RATE_CTRL_0_I_SKIP_ENABLE_ENABLE                 (1)    // // Skip I-frame encoding when there is VLC DMA
//  buffer overflows. This setting is
//  recommended for constant bitrate encoding.
#define LWE2B7_I_RATE_CTRL_0_I_SKIP_ENABLE_DEFAULT                 (0x00000000)

// Average Qp I-Frame Enable.
// This bit determines whether I-frame min Qp
//  is callwlated from the average Qp of
//  previous frame or whether it is set to
//  the software programmed values: I_MIN_QP
//  for MPEG4/H.263 encoding or I-frame min Qp
//  values in MAX_MIN_QP_I register for H.264
//  encoding.
//  This register bit must be turned off 
//  when rate control is turned off (constant qp encoding)
#define LWE2B7_I_RATE_CTRL_0_AVE_QPI_ENABLE                28:28
#define LWE2B7_I_RATE_CTRL_0_AVE_QPI_ENABLE_DISABLE                       (0)    // // Minimum Qp for I-frame is set to the I-frame
//  minimum Qp programmed value.

#define LWE2B7_I_RATE_CTRL_0_AVE_QPI_ENABLE_ENABLE                        (1)    // // Average Qp of previous frame is used as
//  minimum Qp for I-frame encoding.
#define LWE2B7_I_RATE_CTRL_0_AVE_QPI_ENABLE_DEFAULT                (0x00000000)

// Set First I-Frame Qp Constant.
// This bit provides the option to set Qp of
//  the first I-frame to be constant for whole
//  frame (for preview purposes).
#define LWE2B7_I_RATE_CTRL_0_SET_FIRST_IFRAME_QP_CONSTANT                  29:29
#define LWE2B7_I_RATE_CTRL_0_SET_FIRST_IFRAME_QP_CONSTANT_DISABLE                 (0)    // // QP is not forced to be constant on first
//  I-frame. If rate control is on, Qp is
//  allowed to vary across the first I-frame.
//  This setting should be used for normal
//  encoding.

#define LWE2B7_I_RATE_CTRL_0_SET_FIRST_IFRAME_QP_CONSTANT_ENABLE                  (1)    // // QP is forced to be constant on the first
//  I-frame even if rate control is on for
//  first I-frame, Qp is forced to be constant
//  for all macroblocks in the first I-frame.
#define LWE2B7_I_RATE_CTRL_0_SET_FIRST_IFRAME_QP_CONSTANT_DEFAULT  (0x00000000)


// This register specifies all the control bits
//  needed for processing the P-frame in the
//  Rate Control module.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
#define LWE2B7_P_RATE_CTRL_0                      (0x101)
// Rate Control enable for the P-frame.
// This bit is used to enable/disable the Rate
//  Control for the P-frame for both CBR and
//  and VBR rate control.
#define LWE2B7_P_RATE_CTRL_0_RATE_CTRL_P_ENABLE                    0:0
#define LWE2B7_P_RATE_CTRL_0_RATE_CTRL_P_ENABLE_DISABLE                   (0)    // // Rate Control is disabled for P-frame.

#define LWE2B7_P_RATE_CTRL_0_RATE_CTRL_P_ENABLE_ENABLE                    (1)    // // Rate Control is enabled for P-frame.
#define LWE2B7_P_RATE_CTRL_0_RATE_CTRL_P_ENABLE_DEFAULT            (0x00000000)

// Initial Quantization Scale (Qp) for P-Frame.
// This field is applicable for MPEG4/H.263
//  encoding only. For H.264 encoding,
//  P_INIT_QP_H264 parameter in PIC_INIT_Q
//  register is used instead.
// For MPEG4/H.263 encoding, this parameter
//  specifies the Qp for the first P-frame.
#define LWE2B7_P_RATE_CTRL_0_P_INIT_QP                     5:1
#define LWE2B7_P_RATE_CTRL_0_P_INIT_QP_DEFAULT                     (0x00000000)

// Minimum Quantization Scale for P-Frame.
// This field is applicable for MPEG4/H.263
//  encoding only. For H.264 encoding,
//  MAX_MIN_QP_P register is used instead.
// For MPEG4/H.263 encoding, this parameter
//  specifies the minimum Qp for P-frame.
#define LWE2B7_P_RATE_CTRL_0_P_MIN_QP                      10:6
#define LWE2B7_P_RATE_CTRL_0_P_MIN_QP_DEFAULT                      (0x00000000)

// Maximum Quantization Scale for P-Frame.
// This field is applicable for MPEG4/H.263
//  encoding only. For H.264 encoding,
//  MAX_MIN_QP_P register is used instead.
// For MPEG4/H.263 encoding, this parameter
//  specifies the maximum Qp for P-frame.
#define LWE2B7_P_RATE_CTRL_0_P_MAX_QP                      15:11
#define LWE2B7_P_RATE_CTRL_0_P_MAX_QP_DEFAULT                      (0x00000000)

// P Frame Skip Control.
// This bit specifies whether P-frames
//  should be skipped when output buffer
//  overflows.
//  This bit is effective for both CBR and VBR
//  rate controls.
//  Frame skipping shouldn't be enabled when
//  rate control is turned off.
#define LWE2B7_P_RATE_CTRL_0_P_SKIP_ENABLE                 17:17
#define LWE2B7_P_RATE_CTRL_0_P_SKIP_ENABLE_DISABLE                        (0)    // // Encode P-frame even when there is output
//  buffer overflows. This setting can be
//  used for variable bitrate encoding.

#define LWE2B7_P_RATE_CTRL_0_P_SKIP_ENABLE_ENABLE                 (1)    // // Skip P-frame encoding when there is output
//  buffer overflows. This setting is
//  recommended for constant bitrate encoding.
#define LWE2B7_P_RATE_CTRL_0_P_SKIP_ENABLE_DEFAULT                 (0x00000001)

// A1 Parameter Control.
// This parameter controls how A1 is used in the
//  Rate Control equation.
#define LWE2B7_P_RATE_CTRL_0_A1CTRL                19:18
#define LWE2B7_P_RATE_CTRL_0_A1CTRL_OLD_UPDATE                    (0)    // // Use the previously registered A1 for both I
//  and P frames and update the A1 value block
//  by block.

#define LWE2B7_P_RATE_CTRL_0_A1CTRL_FIXED                 (1)    // // Use the new fixed A1 for both I and P
//  frames (for all blocks).

#define LWE2B7_P_RATE_CTRL_0_A1CTRL_NEW_UPDATE                    (2)    // // Use new A1 for both I and P frames and
//  update the A1 value block by block.
#define LWE2B7_P_RATE_CTRL_0_A1CTRL_DEFAULT                        (0x00000000)

// Fixed A1 value.
// This parameter specifies the quarter A1 for
//  I and P frames.
#define LWE2B7_P_RATE_CTRL_0_FIXED_A1_VALUE                27:20
#define LWE2B7_P_RATE_CTRL_0_FIXED_A1_VALUE_DEFAULT                (0x00000000)

// Max Qp Relieve Level.
//  This is a dynamic adjustment for MaxQp.
#define LWE2B7_P_RATE_CTRL_0_REL_MAX_QP                    31:28
#define LWE2B7_P_RATE_CTRL_0_REL_MAX_QP_DEFAULT                    (0x00000000)

// The following is used for VBR rate control. Note that the rate control logic typically make
// adjustment on Max Qp for P-frame based on the following equation:
//   newMaxQp = Clip( (prevNewMaxQp+MaxQp)*prevFrameSize*relieveLevel/
//                    (16*lwrrentFrameBudget),MaxQp, 31);
//
// However, for P-frame following an I-frame, this may result in newMaxQp to be too large;
// therefore Max Qp Factor, if programmed >0, is used to make adjustment as follows:
//   newMaxQp = Clip( (prevNewMaxQp+MaxQp)*prevFrameSize*MAX_QP_FACTOR/64*relieveLevel/
//                    (16*lwrrentFrameBudget),MaxQp, 31);
// where: relieveLevel  is REL_MAX_QP
//        prevFrameSize is STREAM_BYTE_LEN
//
// And if Max Qp Factor is programmed to 0, newMaxQp for P-frame will be set to P_MAX_QP.
// 

// This register is applicable for VBR rate
//  control.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
#define LWE2B7_RC_ADJUSTMENT_0                    (0x102)
// Max Qp Factor.
// This scale factor is used to adjust internal
//  Max Qp callwlation. It is programmed in
//  units of 1/64. Valid value is from 0 to 64.
// Setting this parameter to 64 will result
//  in maximum Max Qp adjustment.
// Setting this parameter to 0 will disable
//  the adjustment.
#define LWE2B7_RC_ADJUSTMENT_0_MAX_QP_FACTOR                       6:0
#define LWE2B7_RC_ADJUSTMENT_0_MAX_QP_FACTOR_DEFAULT               (0x00000040)

// BUFFER_DRAIN is changed from 15:0 to (BFS-1):2

// This register is applicable for both CBR and
//  VBR rate control.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
#define LWE2B7_BUFFER_DEPLETION_RATE_0                    (0x103)
// Buffer Drain/Depletion Rate.
// This parameter specifies the number of 4-byte
//  words the buffer availability will go down
//  after each frame.
// It is recommended that BUFFER_DRAIN be set to
//  bitrate / captured frames per second for both
//  CBR and VBR rate control.
#define LWE2B7_BUFFER_DEPLETION_RATE_0_BUFFER_DRAIN                24:2
#define LWE2B7_BUFFER_DEPLETION_RATE_0_BUFFER_DRAIN_DEFAULT        (0x00000000)

// This register specifies the virtual output
//  buffer size to be used by CBR Rate Control.
// The encoded bitstream (EB) buffers that
//  is managed by EBM may be used as virtual
//  output buffer if there is no other virtual
//  output buffer used by the application.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
#define LWE2B7_BUFFER_SIZE_0                      (0x104)
// Virtual Output Buffer Size.
// This parameter specifies the number of 4-byte
//  words in the virtual output buffer.
#define LWE2B7_BUFFER_SIZE_0_BUFFER_SIZE                   24:2
#define LWE2B7_BUFFER_SIZE_0_BUFFER_SIZE_DEFAULT                   (0x00000000)

// This register specifies the initial delay
//  offset internal parameter used by CBR Rate
//  Control.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
#define LWE2B7_INITIAL_DELAY_OFFSET_0                     (0x105)
// Initial delay offset
// This parameter is specified in 4-byte words
// This parameter is used by CBR RC only
// Recommended value: (0.8)* BUFFER_SIZE 
#define LWE2B7_INITIAL_DELAY_OFFSET_0_INITIAL_DELAY_OFFSET                 24:2
#define LWE2B7_INITIAL_DELAY_OFFSET_0_INITIAL_DELAY_OFFSET_DEFAULT (0x00000000)

// BUFFER_FULL is changed from 15:0 to (BFS-1):2
// REPORTED_FRAME is moved to REPORTED_FRAME register.
// Note that this register is full since BFS is lwrrently defined to 24, so REPORTED_FRAME
//  ideally should be relocated.

// This register specifies the virtual output
//  buffer fullness to be used by Rate Control.
// The encoded bitstream (EB) buffers that
//  is managed by EBM may be used as virtual
//  output buffer if there is no other virtual
//  output buffer used by the application.
// This register and the corresponding
//  REPORTED_FRAME register must be updated
//  periodically by software as software
//  processes the virtual output buffer for
//  better rate control. This register may be
//  updated at any time during encoding.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
#define LWE2B7_BUFFER_FULL_0                      (0x106)
// Virtual Buffer Fullness.
// This parameter specifies the number of 4-byte
//  words in the output buffer yet to be
//  emptied by the application.
#define LWE2B7_BUFFER_FULL_0_BUFFER_FULL                   24:0
#define LWE2B7_BUFFER_FULL_0_BUFFER_FULL_DEFAULT                   (0x00000000)

// This register specifies the frame number
//  corresponding to the written BUFFER_FULL
//  register to be used by Rate Control.
//  The encoder hardware maintains a 16-bit
//  frame number counter that is reported back
//  to host via header information in EBM  or
//  in ENC_FRAME_NUM register. Only lower 8-bit
//  of the frame number needs to be written in
//  this register.
// This register and the corresponding
//  BUFFER_FULL register must be updated
//  periodically by software as software
//  processes the virtual output buffer for
//  better rate control. This register may be
//  updated at any time during encoding.
// This parameter is shadowed and effective in
//  the next frame start after XFER bit in
//  SHADOW_REG_EN is set to  ENABLE.
#define LWE2B7_REPORTED_FRAME_0                   (0x107)
// Frame Number for written buffer fullness.
// This parameter specifies the frame number at
//  which the programmed BUFFER_FULL size was
//  observed.
#define LWE2B7_REPORTED_FRAME_0_REPORTED_FRAME                     7:0
#define LWE2B7_REPORTED_FRAME_0_REPORTED_FRAME_DEFAULT             (0x00000000)

// MIN_FRAME_SIZE is split into MIN_IFRAME_SIZE and MAX_PFRAME_SIZE because number of bits for
//  MIN_IFRAME and MIN_PFRAME parameters increase.

// This register specifies the minimum size of
//  the encoded I-frame. These parameters will
//  be used in the Rate Control module.
#define LWE2B7_MIN_IFRAME_SIZE_0                  (0x108)
// Minimum Frame Size Control.
// This bit specifies whether the minimum size
//  of the frame should be taken from this
//  register (MIN_IFRAME and MIN_PFRAME fields)
//  or should be callwlated internally based
//  on the buffer fullness.
#define LWE2B7_MIN_IFRAME_SIZE_0_MIN_CTRL                  0:0
#define LWE2B7_MIN_IFRAME_SIZE_0_MIN_CTRL_FRAME                   (0)    // // VLC uses the minimum size programmed in this
//  register. If the bit length of the frame is
//  less than this programmed value, it will
//  stuff the frame to meet this minimum size.

#define LWE2B7_MIN_IFRAME_SIZE_0_MIN_CTRL_BUFFER                  (1)    // // The minimum size is computed in the Rate
//  Control module by subtracting the buffer
//  fullness from the underflow threshold.
//  VLC uses this value as the target minimum  
//  size of the frame.
#define LWE2B7_MIN_IFRAME_SIZE_0_MIN_CTRL_DEFAULT                  (0x00000000)

// Minimum Size of an Encoded I-Frame.
// For VBR rate control this parameter specifies
//  the minimum number of 4-byte words that need
//  to be in the encoded I-frame. Recommended
//  value for VBR rate control is half of the
//  average bits/frame: bitrate/framerate/2.
// For CBR rate control this parameter is
//  lwrrently not used but it is reserved to
//  specify (in the future) the minimum encoded
//  size of a Basic Unit (BU) in a I-Frame in
//  bits; i.e. the minimum number of encoded
//  bits that needs to be in a BU.
#define LWE2B7_MIN_IFRAME_SIZE_0_MIN_IFRAME                24:2
#define LWE2B7_MIN_IFRAME_SIZE_0_MIN_IFRAME_DEFAULT                (0x00000000)

// This register specifies the minimum size of
//  the encoded P-frame and frame stuffing
//  control. These parameters will be used
//  in the Rate Control module.
#define LWE2B7_MIN_PFRAME_SIZE_0                  (0x109)
// Stuffing Enable.
// This bit specifies whether the encoded frame
//  is stuffed at the end or not if it does not
//  meet the minimum frame size requirement.
#define LWE2B7_MIN_PFRAME_SIZE_0_STUFFING_ENABLE                   0:0
#define LWE2B7_MIN_PFRAME_SIZE_0_STUFFING_ENABLE_DISABLE                  (0)    // // Encoded frame stuffing is disabled.

#define LWE2B7_MIN_PFRAME_SIZE_0_STUFFING_ENABLE_ENABLE                   (1)    // // Encoded frame stuffing is enabled.
#define LWE2B7_MIN_PFRAME_SIZE_0_STUFFING_ENABLE_DEFAULT           (0x00000000)

// Stuffing Data Control.
// This bit specifies the stuffing data to meet
//  the minimum frame size requirement.
//  The stuffing data is repeated after the last
//  macroblock of a frame until the bit length
//  matches the minimum frame size.
#define LWE2B7_MIN_PFRAME_SIZE_0_STUFFING_TYPE                     1:1
#define LWE2B7_MIN_PFRAME_SIZE_0_STUFFING_TYPE_NORMAL                     (0)    // // Stuffing data is 9'b1 for I-frames and 10'b1
//  for P-frames.

#define LWE2B7_MIN_PFRAME_SIZE_0_STUFFING_TYPE_GARBAGE                    (1)    // // Stuffing data is 0xff.
#define LWE2B7_MIN_PFRAME_SIZE_0_STUFFING_TYPE_DEFAULT             (0x00000000)

// Minimum Size of an Encoded P-Frame.
// For VBR rate control this parameter specifies
//  the minimum number of 4-byte words that need
//  to be in the encoded P-frame. Recommended
//  value for VBR rate control is half of the
//  average bits/frame: bitrate/framerate/2.
// For CBR rate control this parameter is
//  lwrrently not used but it is reserved to
//  specify (in the future) the minimum encoded
//  size of a Basic Unit (BU) in a P-Frame in
//  bits; i.e. the minimum number of encoded
//  bits that needs to be in a BU. 
#define LWE2B7_MIN_PFRAME_SIZE_0_MIN_PFRAME                24:2
#define LWE2B7_MIN_PFRAME_SIZE_0_MIN_PFRAME_DEFAULT                (0x00000000)

// SUGGESTED_I_SIZE is changed from 15:0 to (BFS-1):2.
// SUGGESTED_FRAME_SIZE reg is split into two new registers SUGGESTED_IFRAME_SIZE and
//  SUGGESTED_PFRAME_SIZE to create more space.

// This register specifies the suggested size
//  for I frames. These parameters will
//  be used in the Rate Control module.
#define LWE2B7_SUGGESTED_IFRAME_SIZE_0                    (0x10a)
// Suggested Size for Encoded I-Frame.
// This parameter is applicable only for VBR
//  rate control and is not used (ignored) for
//  CBR rate control.
// This parameter specifies the suggested
//  number of 4-byte words in any encoded
//  I-frame.
// It is recommended that SUGGESTED_I_SIZE be
//  set to 4 * SUGGESTED_P_SIZE.
#define LWE2B7_SUGGESTED_IFRAME_SIZE_0_SUGGESTED_I_SIZE                    24:2
#define LWE2B7_SUGGESTED_IFRAME_SIZE_0_SUGGESTED_I_SIZE_DEFAULT    (0x00000000)

// SUGGESTED_P_SIZE is changed from 31:16 to (BFS-1):2.
// SUGGESTED_FRAME_SIZE reg is split into two new registers SUGGESTED_IFRAME_SIZE and
//  SUGGESTED_PFRAME_SIZE to create more space.

// This register specifies the suggested size
//  for P frames. These parameters will
//  be used in the Rate Control module.
#define LWE2B7_SUGGESTED_PFRAME_SIZE_0                    (0x10b)
// Suggested Size for Encoded P-Frame.
// This parameter is applicable to both VBR
//  and CBR rate control 
// This parameter specifies the suggested
//  number of 4-byte words in any encoded
//  P-frame.
// It is recommended that SUGGESTED_P_SIZE be
//  set to the bitrate / encoded frames per
//  second for both VBR and CBR
#define LWE2B7_SUGGESTED_PFRAME_SIZE_0_SUGGESTED_P_SIZE                    24:2
#define LWE2B7_SUGGESTED_PFRAME_SIZE_0_SUGGESTED_P_SIZE_DEFAULT    (0x00000000)

// TARGET_BUFFER_I is changed from 15:0 to (BFS-1):2.
// TARGET_BUFFER_SIZE reg is split into two new registers TARGET_BUFFER_I_SIZE and
//  TARGET_BUFFER_P_SIZE to create more space.

// This register specifies the size for the
//  target output buffer for I frames.
//  These parameters will be used in
//  the Rate Control module.
#define LWE2B7_TARGET_BUFFER_I_SIZE_0                     (0x10c)
// Target Buffer Size for I-Frame.
// This parameter specifies the number of
//  4-byte words in an I-frame that will make
//  the target buffer full.
// For VBR rate control, it is recommended that
//  TARGET_BUFFER_I be set to 1/2 of buffer
//  full size.
// This register is not applicable to CBR
#define LWE2B7_TARGET_BUFFER_I_SIZE_0_TARGET_BUFFER_I                      24:2
#define LWE2B7_TARGET_BUFFER_I_SIZE_0_TARGET_BUFFER_I_DEFAULT      (0x00000000)

// TARGET_BUFFER_P is changed from 31:16 to (BFS-1):2.
// TARGET_BUFFER_SIZE reg is renamed/moved to a new register TARGET_BUFFER_P_SIZE to create
//  more space.

// This register specifies the size for the
//  target output buffer for P frames.
//  These parameters will be used in
//  the Rate Control module.
#define LWE2B7_TARGET_BUFFER_P_SIZE_0                     (0x10d)
// Target Buffer Size for P-Frame.
// This parameter specifies the number of
//  4-byte words in a P-frame that will make
//  the target buffer full.
// For VBR rate control, it is recommended that
//  TARGET_BUFFER_P be set to 1/3 of buffer
//  full size.
// This register is not applicable to CBR
#define LWE2B7_TARGET_BUFFER_P_SIZE_0_TARGET_BUFFER_P                      24:2
#define LWE2B7_TARGET_BUFFER_P_SIZE_0_TARGET_BUFFER_P_DEFAULT      (0x00000000)

// SKIP_THRESHOLD is changed from 15:0 to (BFS-1):2

// This register specifies the output virtual
//  buffer threshold to make the frame skip
//  decision in Rate Control module.
//  This decision is made at the beginning of
//  frame encoding whenever a new input frame
//  is received.
#define LWE2B7_SKIP_THRESHOLD_0                   (0x10e)
// Buffer Threshold for frame Skip.
// This specifies the number of 4-byte words
//  buffer threshold for frame skip.
// The Rate Control module will skip the next
//  frame if the buffer size goes over this
//  threshold.
// For VBR rate control, the recommended value
//  is 0.9 * buffer size.
// For CBR rate control, the recommended value
//  is 0.8 * buffer size.
#define LWE2B7_SKIP_THRESHOLD_0_SKIP_THRESHOLD                     24:2
#define LWE2B7_SKIP_THRESHOLD_0_SKIP_THRESHOLD_DEFAULT             (0x00000000)

// UNDER_GUARD is changed from 15:0 to (BFS-1):2 and moved from OVERFLOW_THRESHOLD to
//  UNDERFLOW_THRESHOLD.

// This register is applicable only for VBR
//  rate control and is not used (ignored) for
//  CBR rate control.
// This register specifies the output virtual
//  buffer thresholds.
// These parameters will be used in the Rate
//  Control module.
#define LWE2B7_UNDERFLOW_THRESHOLD_0                      (0x10f)
// Underflow Threshold.
// This parameter specifies the number of
//  4-byte words buffer underflow threshold.
// It is recommended that UNDER_GUARD be set to
//  10% of the buffer full size.
#define LWE2B7_UNDERFLOW_THRESHOLD_0_UNDER_GUARD                   24:2
#define LWE2B7_UNDERFLOW_THRESHOLD_0_UNDER_GUARD_DEFAULT           (0x00000000)

// OVER_GUARD is changed from 31:16 to (BFS-1):2.
// UNDER_GUARD is moved to a new register UNDERFLOW_THRESHOLD to create more space.

// This register is applicable only for VBR
//  rate control and is not used (ignored) for
//  CBR rate control.
// This register specifies the output virtual
//  buffer thresholds.
// These parameters will be used in the Rate
//  Control module.
#define LWE2B7_OVERFLOW_THRESHOLD_0                       (0x110)
// Overflow Threshold.
// This parameter specifies the number of
//  4-byte words buffer overflow threshold.
// In beginning of frame encoding, current
//  target frame size is added to the
//  aclwmulated buffer fullness and if the
//  value exceed the programmed OVER_GUARD,
//  then the rate control target bitrate is
//  adjusted.
// It is recommended that OVER_GUARD be set to
//  80% of the buffer full size.
#define LWE2B7_OVERFLOW_THRESHOLD_0_OVER_GUARD                     24:2
#define LWE2B7_OVERFLOW_THRESHOLD_0_OVER_GUARD_DEFAULT             (0x00000000)

// Bit 31 (RC_SELECT) of AVE_BIT_LEN is deleted.

// This register specifies the average bit
//  length of a frame after the VLC coding.
//  This register can be used to do the rate
//  control in Bypass VLC mode.
//  In the Bypass VLC mode the frame length
//  (after VLC coding) is not known to the
//  hardware. In such a case this average
//  value can be used to do the rate control.
//  This register is programmed only once.
#define LWE2B7_AVE_BIT_LEN_0                      (0x111)
// Average Bit Length.
// This value specifies the average frame
//  length in bytes to be used by the rate
//  control in Bypass VLC mode.
#define LWE2B7_AVE_BIT_LEN_0_AVE_BIT_LEN                   22:0
#define LWE2B7_AVE_BIT_LEN_0_AVE_BIT_LEN_DEFAULT                   (0x00000000)

// Bits 21:16 (GOP_DQUANT) of RC_QP_DQUANT is deleted but reserved for future use in case it is
//  needed because of clipping on average frame Qp when crossing GOP boundary).

// This register is applicable for CBR rate
//  control only.
// This register specifies the maximum Qp
//  difference between Basic Units for CBR rate
//  control.
// This register is programmed only once.
#define LWE2B7_RC_QP_DQUANT_0                     (0x112)
// Max Frame Qp Delta.
// This parameter is applicable for CBR rate
//  control only.
// This parameter specifies the maximum quant
//  (Qp) difference between Qp of any Basic
//  Unit (BU) from the average Qp of past frame.
#define LWE2B7_RC_QP_DQUANT_0_FRAME_DQUANT                 5:0
#define LWE2B7_RC_QP_DQUANT_0_FRAME_DQUANT_DEFAULT                 (0x00000000)

// Max BU Qp Delta.
// This parameter is applicable for CBR rate
//  control only.
// This parameter specifies the maximum quant
//  (Qp) difference between Qp of a Basic Unit
//  (BU) from the previous BU in the same frame.
#define LWE2B7_RC_QP_DQUANT_0_BU_DQUANT                    13:8
#define LWE2B7_RC_QP_DQUANT_0_BU_DQUANT_DEFAULT                    (0x00000000)

// Note that there is a restriction on the total number of Basic Units in a frame. This
//  restriction is due to the fact that rate control uses the rate control RAM (204x96-bit)
//  to store SATD data for each BU for CBR rate control, or to store SAD data for each BU for
//  VBR rate control.
// 
// For VBR rate control:
//  If BU_SIZE=1, each entry in RC RAM is 12-bit (SAD) and therefore maximum number of MB per
//  frame is 1632 MBs.
//  If BU_SIZE>1, each entry in RC RAM is 16-bit (16 mb SAD) and therefore maximum number of
//  MB per frame is 1224 BUs = 16x1224 MBs = 19584 MBs.
//
// For CBR rate control:
//  Each entry in RC RAM is 24-bit (upto 127 17-bit SATD) so max BU_SIZE is 127 MBs and maximum
//  number of MB per frame is 408 BUs = 127 x 408 MBs = 51816 MBs
//  Total number of entries in RAM is 4x204=816, but there are two tables one for storing current frame
// MAD and another for past frame MAD. So number of BU is 408.

// This register is applicable for CBR rate
//  control only.
// This register specifies the Basic Unit
//  size for CBR rate control.
// This register is programmed only once.
#define LWE2B7_RC_BU_SIZE_0                       (0x113)
// Basic Unit Size.
// This parameter is applicable for both CBR
//  and VBR rate control.
// This parameter specifies the number of
//  macroblocks in a basic unit (BU).
// For VBR rate control, this parameter maybe
//  programmed from 1 to 16 and maximum of 1632
//  MBs/frame if BU_SIZE=1 and maximum 1224
//  BUs/frame if BU_SIZE>1.
// For CBR rate control, this parameter maybe
//  programmed from 1 to 127 for maximum of 408
//  BUs/frame. 
// FOR CBR, it is recommended that this is
//  programmed with the the number of macroblocks
//  in a macroblock row if image width is smaller than that of VGA.
// When image width is equal to or greater than that of VGA, 
//  it is recommended to program this value as 1/2 the number of macoblocks   
//  a mbrow in the CBR case.
// RESTRICTION: Total number of macroblocks in
//  a frame must be divisible by the number of
//  macroblocks in a BU.
#define LWE2B7_RC_BU_SIZE_0_BU_SIZE                6:0
#define LWE2B7_RC_BU_SIZE_0_BU_SIZE_DEFAULT                        (0x00000000)

// Total Number of Basic Units.
// This parameter is applicable for CBR rate
//  control only.
// This should be programmed by the driver to
//  WIDTH * HEIGHT / BU_SIZE and the result
//  must be an whole integer.
#define LWE2B7_RC_BU_SIZE_0_NUMBER_OF_BU                   28:16
#define LWE2B7_RC_BU_SIZE_0_NUMBER_OF_BU_DEFAULT                   (0x00000000)

// This register is applicable for CBR rate
//  control only.
// This register specifies the GOP parameters
//  for CBR rate control.
// This register is programmed only once.
#define LWE2B7_GOP_PARAM_0                        (0x114)
// GOP Length.
// This parameter is applicable for CBR rate
//  control only.
// This parameter specifies the number of frames
//  in a GOP to be used by the rate control.
// This parameter should be set to 75 when
//  GOP_FLAG is set to OPEN. When GOP_FLAG is
//  set to close then this parameter should be
//  programmed to the I-frame interval desired
//  by application.
#define LWE2B7_GOP_PARAM_0_GOP_LENGTH                      15:0
#define LWE2B7_GOP_PARAM_0_GOP_LENGTH_DEFAULT                      (0x0000004b)

// GOP Flag.
// This parameter is applicable for CBR rate
//  control only.
// This parameter specifies if the GOP starts
//  with an I-Frame or not.
#define LWE2B7_GOP_PARAM_0_GOP_FLAG                31:31
#define LWE2B7_GOP_PARAM_0_GOP_FLAG_OPEN                  (0)    // // GOP does not need to start with an I-frame.
// This setting is recommended for video
//  telephony application where I-frame is
//  rarely encoded and intra refresh is used
//  instead.

#define LWE2B7_GOP_PARAM_0_GOP_FLAG_CLOSED                        (1)    // // GOP starts with I-Frame.
// I-frame should be inserted using
//  P_FRAME_INTERVAL parameter and setting
//  P_FRAME_INTERVAL_DIS to ENABLE in
//  FRAME_TYPE register.
#define LWE2B7_GOP_PARAM_0_GOP_FLAG_DEFAULT                        (0x00000000)


// This register specifies the number of
//  4-byte words to be loaded into the Rate
//  Control RAM. This RAM needs to be loaded
//  only to resume a saved context.
//  Software writes to this register then
//  follows it by a number of writes to the
//  counter register as specified by the
//  RC_NUM_DWORDS_TO_WRITE field.
// Size of Rate Control RAM is 692 4-byte words.
// The first 612 locations are used for storing
// SAD/MAD information, in both VBR and CBR modes.
// The last 80 locations are used only in the CBR mode.
// These locations are used to store the past BU
// history (last 20 BUs) for computing mx1 and
// MadPic1. In this mode, each entry of the BU Filter RAM 
// is 48-bits. So when Host writes the first 
// entry, 32-bits of host data are mapped to the 
// lower 32-bits of the RAM data and the lower 16-bits 
// of the next host write will map to the upper 16-bits 
// of the RAM data. Upper 16-bits of host-data during the 
// second host write is not used by the h/w.
// 
#define LWE2B7_RC_RAM_LOAD_CMD_0                  (0x115)
// Number of 4-byte words of Data to Write
//  to Rate Control RAM.
// This parameter specifies the number of
//  4-byte words to be written to the Rate
//  Control RAM.
#define LWE2B7_RC_RAM_LOAD_CMD_0_RC_NUM_DWORDS_TO_WRITE                    9:0
#define LWE2B7_RC_RAM_LOAD_CMD_0_RC_NUM_DWORDS_TO_WRITE_DEFAULT    (0x00000000)

// This register contains the data to be
//  written to the Rate Control RAM. It must be
//  written after a write to the RC_RAMLOAD_CMD
//  register a number of times equal to the
//  RC_NUM_DWORDS_TO_WRITE field.
#define LWE2B7_RC_RAM_LOAD_DATA_0                 (0x116)
// Rate Control SRAM Load Data.
// This is the 4-byte data word to be written
//  to Rate Control RAM.
#define LWE2B7_RC_RAM_LOAD_DATA_0_LOAD_DATA                31:0
#define LWE2B7_RC_RAM_LOAD_DATA_0_LOAD_DATA_DEFAULT                (0x00000000)

// Reserve 3 registers for future expansion of rate control context save/restore registers.




// This register may be written when it is
//  safe to write reigsters to specify
//  lower bound for next frame target encoded
//  size for CBR rate control when resuming
//  from context switch. It is not necessary to
//  initialize this register at the beginning
//  of encoding.
// The value of this register is updated
//  automatically by the encoder hardware.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_LOWER_BOUND_0                      (0x11a)
// Lower Bound of target frame size (LSB)
// Internal register for Lower bound is 33 bits.
// This register field specfies the lower 32
//  bits. The most siginificant bit is in
//  CONTEXT_SAVE_MISC register.
// This parameter specifies the lower bound for
//  next frame target encoded size for CBR rate
//  control.
#define LWE2B7_LOWER_BOUND_0_LOWER_BOUND_LSB                       31:0
#define LWE2B7_LOWER_BOUND_0_LOWER_BOUND_LSB_DEFAULT               (0x00000000)

// This register may be written when it is
//  safe to write registers to specify
//  upper bound for next frame target encoded
//  size for CBR rate control when resuming
//  from context switch. It is not necessary to
//  initialize this register at the beginning
//  of encoding.
// The value of this register is updated
//  automatically by the encoder hardware.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_UPPER_BOUND_0                      (0x11b)
// Upper Bound of target frame size (LSB).
// Internal register for Upper bound is 33 bits.
// This register field specfies the lower 32
//  bits. The most significant bit is in
//  CONTEXT_SAVE_MISC register.
// This parameter specifies the upper bound for
//  next frame target encoded size for CBR rate
//  control.
#define LWE2B7_UPPER_BOUND_0_UPPER_BOUND_LSB                       31:0
#define LWE2B7_UPPER_BOUND_0_UPPER_BOUND_LSB_DEFAULT               (0x00000000)

// This register may be written when it is
//  safe to write registers to specify
//  remaining encoded bits allocated to a GOP
//  for CBR rate control when resuming
//  from context switch. It is not necessary to
//  initialize this register at the beginning
//  of encoding.
// The value of this register is updated
//  automatically by the encoder hardware.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_REMAINING_BITS_0                   (0x11c)
// Remaining Bits allocated to a GOP.
// Internal register for Remaining bits is 33
//  bits. This parameter specifies the lower 32
//  bits. The most significant bit is in
//  CONTEXT_SAVE_MISC register.
#define LWE2B7_REMAINING_BITS_0_REMAINING_BITS_LSB                 31:0
#define LWE2B7_REMAINING_BITS_0_REMAINING_BITS_LSB_DEFAULT         (0x00000000)

// This register may be written when it is
//  safe to write registers to specify
//  number of encoded BU so far in a GOP
//  for CBR rate control when resuming
//  from context switch. It is not necessary to
//  initialize this register at the beginning
//  of encoding.
// The value of this register is updated
//  automatically by the encoder hardware.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_NUM_CODED_BU_0                     (0x11d)
// Number of coded BU so far in a GOP.
// This parameter specifies the lower 32 bits
//  of number of encoded BU so far in a GOP
//  for CBR rate control.
#define LWE2B7_NUM_CODED_BU_0_NUM_CODED_BU                 25:0
#define LWE2B7_NUM_CODED_BU_0_NUM_CODED_BU_DEFAULT                 (0x00000000)

// This register may be written when it is
//  safe to write registers to specify
//  previous Qp context for CBR rate control
//  and LFSR (DEFAULT INTRA_REF mode) 
//  when resuming from context switch.
//  It is not necessary to initialize this
//  register at the beginning of encoding.
// The value of this register is updated
//  automatically by the encoder hardware.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_PREVIOUS_QP_0                      (0x11e)
// Qp of Last GOP 
// This parameter specifies Qp of last GOP
// for CBR rate control.
#define LWE2B7_PREVIOUS_QP_0_QP_LAST_GOP                   5:0
#define LWE2B7_PREVIOUS_QP_0_QP_LAST_GOP_DEFAULT                   (0x00000000)

// This parameter specifies the QP that is
//  assigned to an I-frame
#define LWE2B7_PREVIOUS_QP_0_INITIAL_QP                    13:8
#define LWE2B7_PREVIOUS_QP_0_INITIAL_QP_DEFAULT                    (0x00000000)

// This parameter specifies the average QP of
//  the last frame.             
#define LWE2B7_PREVIOUS_QP_0_PAVERAGE_FRAME_QP                     21:16
#define LWE2B7_PREVIOUS_QP_0_PAVERAGE_FRAME_QP_DEFAULT             (0x00000000)

// This parameter specifies the LFSR counter
//  used in the DEFAULT_MODE
#define LWE2B7_PREVIOUS_QP_0_LFSR_COUNT                    31:24
#define LWE2B7_PREVIOUS_QP_0_LFSR_COUNT_DEFAULT                    (0x00000000)

// This register may be written when it is
//  safe to write registers to specify
//  number of encoded P frames so far in a GOP
//  and the GOP number for CBR rate control when
//  resuming from context switch. It is not
//  necessary to initialize this register at the
//  beginning of encoding.
// The value of this register is updated
//  automatically by the encoder hardware.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_NUM_P_PICTURE_0                    (0x11f)
// Number of coded P frames so far in a GOP.
// This parameter specifies the number of
//  encoded P frames so far in a GOP for CBR
//  rate control.
#define LWE2B7_NUM_P_PICTURE_0_NUM_P_PICTURE                       15:0
#define LWE2B7_NUM_P_PICTURE_0_NUM_P_PICTURE_DEFAULT               (0x00000000)

// Number of GOP.
// This parameter specifies the number of
//  encoded GOP.
#define LWE2B7_NUM_P_PICTURE_0_NUMBER_OF_GOP                       31:16
#define LWE2B7_NUM_P_PICTURE_0_NUMBER_OF_GOP_DEFAULT               (0x00000000)

// This register may be written when it is
//  safe to write registers to specify
//  starting Qp Sum  and HEC Count (MPEG4) 
//  for both CBR and VBR rate
//  control when resuming from context switch.
// The value of this register is updated
//  automatically by the encoder hardware
//  so when it is read at the end of a frame
//  encoding, it will point to the last frame
//  Qp Sum value and HEC Count.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_QP_SUM_0                   (0x120)
// Qp Sum for the last frame.
// This parameter specifies the aclwmulated
//  sum of the Qp for VBR and CBR rate control.
#define LWE2B7_QP_SUM_0_QP_SUM                     18:0
#define LWE2B7_QP_SUM_0_QP_SUM_DEFAULT                             (0x00000000)

// VLC mode and Data Partition mode (MPEG4) 
//  HEC Count for the last frame
#define LWE2B7_QP_SUM_0_HEC_CNT                    31:24
#define LWE2B7_QP_SUM_0_HEC_CNT_DEFAULT                            (0x00000000)

// This register may be written when it is
//  safe to write registers to specify
//  total energy (SAD) for both VBR and CBR
//  rate control
//  when resuming from context switch.
// The value of this register is updated
//  automatically by the encoder hardware
//  so when it is read at the end of a frame
//  encoding, it will point to the last frame
//  sum of SAD value.
// In the VBR mode, lower 24-bits are valid.
// In the CBR mode, lower 26-bits are valid.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_TOTAL_ENERGY_0                     (0x121)
// Sum of SAD for the last frame.
// This parameter specifies the aclwmulated
//  sum of SAD for VBR and CBR rate control.
#define LWE2B7_TOTAL_ENERGY_0_SUM_OF_SAD                   28:0
#define LWE2B7_TOTAL_ENERGY_0_SUM_OF_SAD_DEFAULT                   (0x00000000)

// This register may be written when it is
//  safe to write registers to specify
//  A1 value for VBR rate control,
//  GOP_OVERFUE and Total Frame Qp for CBR RC
//  when resuming from context switch.
//  It is not necessary to initialize this
//  register at the beginning of encoding.
// The value of this register is updated
//  automatically by the encoder hardware
//  so when it is read at the end of a frame
//  encoding, it will point to the last frame
//  A1 value.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_A1_VALUE_0                 (0x122)
// A1 Value for the last frame.
// This parameter specifies the A1 value
//  for VBR rate control.
#define LWE2B7_A1_VALUE_0_A1_VALUE                 8:0
#define LWE2B7_A1_VALUE_0_A1_VALUE_DEFAULT                         (0x00000000)

// Sum of Qp over all the BUs for the last frame.
// This parameter specifies the aclwmulated
//  sum of the Qp for CBR rate control over all BUs.
#define LWE2B7_A1_VALUE_0_TOTAL_FRAME_QP                   28:10
#define LWE2B7_A1_VALUE_0_TOTAL_FRAME_QP_DEFAULT                   (0x00000000)

// This parameter represents the gop_overdue flag
//  used in CBR RC
#define LWE2B7_A1_VALUE_0_GOP_OVERDUE                      29:29
#define LWE2B7_A1_VALUE_0_GOP_OVERDUE_DEFAULT                      (0x00000000)

// This register may be written when it is safe
//  to write registers to specify the length
//  of frame bitstream when resuming from
//  context switch. If virtual output buffer
//  fullness state is not known, this saved
//  value may also be added to BUFFER_FULL_READ
//  saved value and used to initialize
//  BUFFER_FULL register.
// The value of this register is updated
//  automatically by the encoder hardware
//  so when it is read at the end of a frame
//  encoding, it will return the size of the
//  last frame encoded bitstream.
// In the VBR mode, stream length is returned in
//  bytes.
// In the CBR mode, buffer fullness is returned in
//  bits.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_LENGTH_OF_STREAM_0                 (0x123)
// Output Bitstream Byte Length.
// This parameter returns the size of the
//  last frame encoded bitstream.
#define LWE2B7_LENGTH_OF_STREAM_0_STREAM_BYTE_LEN                  31:0
#define LWE2B7_LENGTH_OF_STREAM_0_STREAM_BYTE_LEN_DEFAULT          (0x00000000)

// This register must be read when it is safe
//  to write registers and saved as part of
//  context save. When the context is resumed
//  at a later time, if virtual output buffer
//  fullness state is not known, this saved
//  value may be added to STREAM_BYTE_LEN
//  saved value and used to initialize
//  BUFFER_FULL register.
// The value of this register is updated
//  automatically by the encoder hardware
//  so when it is read at the end of a frame
//  encoding, it will return the internally
//  callwlated virtual buffer size (fullness).
#define LWE2B7_BUFFER_FULL_READ_0                 (0x124)
// Internal Buffer Full Read.
// This parameter returns the fullness of
//  the stream buffer after encoding the
//  current frame in bytes.
#define LWE2B7_BUFFER_FULL_READ_0_BUFFER_FULL_READ                 24:0
#define LWE2B7_BUFFER_FULL_READ_0_BUFFER_FULL_READ_DEFAULT         (0x00000000)

// This register may be written when it is
//  safe to write reigsters to specify average
//  past frame header bits for CBR rate control
//  when resuming from context switch. It is not
//  necessary to initialize this register at the
//  beginning of encoding.
// The value of this register is updated
//  automatically by the encoder hardware.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_TARGET_BUFFER_LEVEL_0                      (0x125)
// Target buffer level
#define LWE2B7_TARGET_BUFFER_LEVEL_0_TARGET_BUFFER_LEVEL                   27:0
#define LWE2B7_TARGET_BUFFER_LEVEL_0_TARGET_BUFFER_LEVEL_DEFAULT   (0x00000000)

// This register may be written when it is
//  safe to write reigsters to specify average
//  past frame header bits for CBR rate control
//  when resuming from context switch. It is not
//  necessary to initialize this register at the
//  beginning of encoding.
// The value of this register is updated
//  automatically by the encoder hardware.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_DELTA_P_0                  (0x126)
// Delta_P used in AUP
#define LWE2B7_DELTA_P_0_DELTA_P                   27:0
#define LWE2B7_DELTA_P_0_DELTA_P_DEFAULT                           (0x00000000)

// This register may be written when it is safe
//  to write registers to specify the length
//  of frame bitstream when resuming from
//  context switch. 
// The value of this register is updated
//  automatically by the encoder hardware
//  so when it is read at the end of a frame
//  encoding, it will return the size of the
//  last frame encoded bitstream.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_LENGTH_OF_STREAM_CBR_0                     (0x127)
// Output Bitstream Byte Length in CBR mode.
// This parameter returns the size of the
//  last frame encoded bitstream in bits.
#define LWE2B7_LENGTH_OF_STREAM_CBR_0_STREAM_BYTE_LEN_CBR                  31:0
#define LWE2B7_LENGTH_OF_STREAM_CBR_0_STREAM_BYTE_LEN_CBR_DEFAULT  (0x00000000)

// Reserve 1 registers for future expansion of rate control context save/restore registers.
// reserve[1] incr1;
//
// Rate Control SRAM Read - this may be used for context saving or for diagnostics.
// Size of Rate Control RAM is 692 4-byte words.
// 

// This register specifies the address and
//  count of 4-byte words data  to be read
//  from the Rate Control RAM.
// Software writes to this register then
//  follows it by a number of reads from
//  RC_RAM_READ_DATA register as specified
//  by the RC_NUM_DWORDS_TO_READ field.
// If Rate Control is enabled then Rate Control
//  RAM content must be saved during context
//  save in order to be able to resume the
//  context at a later time.
// In VBR mode, it is enough to save/restore the 
//  first 612 locations.
// In CBR mode, all the 692 locations must be
//  saved and restored.
// The last 80 locations are mapped to the BU 
//  Filter RAM, in which each entry is 48-bits wide.
// In the CBR mode, when the last 80 loations are 
//  read out, the first host read will return the 
//  lower 32-bits of the Filter RAM data and the 
//  next host read will return the upper 
//  16-bits of the RAM data in the lower 
//  16-bits of the host-data and the upper 
//  16-bits of host-data are returned as 0's
#define LWE2B7_RC_RAM_READ_CMD_0                  (0x128)
// Number of words to read from Rate Control
//  RAM.
// This value specifies the number of 4-byte
//  words to be read from the Rate Control RAM.
#define LWE2B7_RC_RAM_READ_CMD_0_RC_NUM_DWORDS_TO_READ                     9:0
#define LWE2B7_RC_RAM_READ_CMD_0_RC_NUM_DWORDS_TO_READ_DEFAULT     (0x00000000)

// This register contains the 4-byte Rate
//  control data which is read from the
//  RC SRAM.  This register
//  must be read after a write to the
//  RC_RAM_READ_CMD register a number of
//  times equal to the value written to
//  RC_NUM_DWORDS_TO_READ field.
#define LWE2B7_RC_RAM_READ_DATA_0                 (0x129)
// Rate Control SRAM Read Data.
// This is the 4-byte read data from Rate
//  Control RAM.
#define LWE2B7_RC_RAM_READ_DATA_0_READ_DATA                31:0
#define LWE2B7_RC_RAM_READ_DATA_0_READ_DATA_DEFAULT                (0x00000000)

// This register may be written when it is
//  safe to write reigsters to specify coded
//  frame information for CBR rate control when
//  resuming from context switch. It is not
//  necessary to initialize this register at the
//  beginning of encoding.
// The value of this register is updated
//  automatically by the encoder hardware.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_CODED_FRAMES_0                     (0x12a)
// Number of coded P frame in GOP.
#define LWE2B7_CODED_FRAMES_0_NUM_CODED_P_FRAMES                   15:0
#define LWE2B7_CODED_FRAMES_0_NUM_CODED_P_FRAMES_DEFAULT           (0x00000000)

// Remaining P frames in GOP.
#define LWE2B7_CODED_FRAMES_0_REMAINING_P_FRAMES                   31:16
#define LWE2B7_CODED_FRAMES_0_REMAINING_P_FRAMES_DEFAULT           (0x00000000)

// This register may be written when it is
//  safe to write reigsters to specify average
//  past frame header bits for CBR rate control
//  when resuming from context switch. It is not
//  necessary to initialize this register at the
//  beginning of encoding.
// The value of this register is updated
//  automatically by the encoder hardware.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_P_AVE_HEADER_BITS_A_0                      (0x12b)
// Average header bits for past frame.
#define LWE2B7_P_AVE_HEADER_BITS_A_0_P_AVE_HEADER_BITS_A                   24:0
#define LWE2B7_P_AVE_HEADER_BITS_A_0_P_AVE_HEADER_BITS_A_DEFAULT   (0x00000000)

// This register may be written when it is
//  safe to write reigsters to specify average
//  past frame header bits for CBR rate control
//  when resuming from context switch. It is not
//  necessary to initialize this register at the
//  beginning of encoding.
// The value of this register is updated
//  automatically by the encoder hardware.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_P_AVE_HEADER_BITS_B_0                      (0x12c)
// Average header bits for past frame.
#define LWE2B7_P_AVE_HEADER_BITS_B_0_P_AVE_HEADER_BITS_B                   24:0
#define LWE2B7_P_AVE_HEADER_BITS_B_0_P_AVE_HEADER_BITS_B_DEFAULT   (0x00000000)

// This register may be written when it is
//  safe to write reigsters to specify average
//  past frame header bits for CBR rate control
//  when resuming from context switch. It is not
//  necessary to initialize this register at the
//  beginning of encoding.
// The value of this register is updated
//  automatically by the encoder hardware.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_PREV_FRAME_MAD_0                   (0x12d)
// Previous frame MAD.
#define LWE2B7_PREV_FRAME_MAD_0_PREV_FRAME_MAD                     16:0
#define LWE2B7_PREV_FRAME_MAD_0_PREV_FRAME_MAD_DEFAULT             (0x00000000)

// Current WindowSize for the RC model estimator
//
#define LWE2B7_PREV_FRAME_MAD_0_RC_QP_WINDOW_SIZE                  24:20
#define LWE2B7_PREV_FRAME_MAD_0_RC_QP_WINDOW_SIZE_DEFAULT          (0x00000000)

// Current Window Size for the MAD model estimator
#define LWE2B7_PREV_FRAME_MAD_0_RC_MAD_WINDOW_SIZE                 30:26
#define LWE2B7_PREV_FRAME_MAD_0_RC_MAD_WINDOW_SIZE_DEFAULT         (0x00000000)

// This register may be written when it is
//  safe to write reigsters to specify total
//  average qp for P picture in GOP for CBR rate
//  control when resuming from context switch.
//  It is not necessary to initialize this
//  register at the beginning of encoding.
// The value of this register is updated
//  automatically by the encoder hardware.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_TOTAL_QP_FOR_P_PICTURE_0                   (0x12e)
// Total average qp for GOP.
#define LWE2B7_TOTAL_QP_FOR_P_PICTURE_0_TOTAL_QP_FOR_P_PICTURE                     21:0
#define LWE2B7_TOTAL_QP_FOR_P_PICTURE_0_TOTAL_QP_FOR_P_PICTURE_DEFAULT     (0x00000000)

//  for CBR RC and VBR RC
// This register may be written when it is
//  safe to write reigsters to specify various
//  other context save information for CBR rate
//  control when resuming from context switch.
//  It is not necessary to initialize this
//  register at the beginning of encoding.
// The value of this register is updated
//  automatically by the encoder hardware.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_CONTEXT_SAVE_MISC_0                        (0x12f)
// MSB of the lower bound (refer to LOWER_BOUND
//  register)
#define LWE2B7_CONTEXT_SAVE_MISC_0_LOWER_BOUND_MSB                 0:0
#define LWE2B7_CONTEXT_SAVE_MISC_0_LOWER_BOUND_MSB_DEFAULT         (0x00000000)

// MSB of the Upper bound (refer to UPPER_BOUND
//  register)
#define LWE2B7_CONTEXT_SAVE_MISC_0_UPPER_BOUND_MSB                 3:3
#define LWE2B7_CONTEXT_SAVE_MISC_0_UPPER_BOUND_MSB_DEFAULT         (0x00000000)

// MSB of the Remaining bits (refer to
//  REMAINING_BITS register)
#define LWE2B7_CONTEXT_SAVE_MISC_0_REMAINING_BITS_MSB                      6:6
#define LWE2B7_CONTEXT_SAVE_MISC_0_REMAINING_BITS_MSB_DEFAULT      (0x00000000)

// Target buffer level flag
#define LWE2B7_CONTEXT_SAVE_MISC_0_TARGET_BUFFER_LEVEL_FLAG                9:9
#define LWE2B7_CONTEXT_SAVE_MISC_0_TARGET_BUFFER_LEVEL_FLAG_DEFAULT        (0x00000000)

// Current Frame Pointer.
// For CBR RC, the RC RAM holds two tables one
//  for current frame MAD and one for past frame
//  MAD. This parameter selects which one is
//  current and which one is past.
#define LWE2B7_CONTEXT_SAVE_MISC_0_LWRRENT_FRAME_POINTER                   10:10
#define LWE2B7_CONTEXT_SAVE_MISC_0_LWRRENT_FRAME_POINTER_DEFAULT   (0x00000000)

// This should be set to "1", when
// a new context starts for the first time
// (not being restored, but starting for the first time)
// HW self-clears this bit on the next frame whenever 
// it is set
#define LWE2B7_CONTEXT_SAVE_MISC_0_FIRST_FRAME                     11:11
#define LWE2B7_CONTEXT_SAVE_MISC_0_FIRST_FRAME_DEFAULT             (0x00000000)

// The pointer points to the internal array where the
//  frame bit lengths are stored. This is an array of 16.
//  This is used by both CBR and VBR RC.
#define LWE2B7_CONTEXT_SAVE_MISC_0_FRAME_BITS_PTR                  15:12
#define LWE2B7_CONTEXT_SAVE_MISC_0_FRAME_BITS_PTR_DEFAULT          (0x00000000)

// UPT filter RAM pointers.
// This parameter specifies internal UPT filter
//  RAM array pointers.
#define LWE2B7_CONTEXT_SAVE_MISC_0_UPT_FILT_PTRS                   25:16
#define LWE2B7_CONTEXT_SAVE_MISC_0_UPT_FILT_PTRS_DEFAULT           (0x00000000)

// Adjusted Max Qp of the Prev Frame
// This parameter specifies the prev frame's
//  adjusted Max Qp, used by VBR RC.
#define LWE2B7_CONTEXT_SAVE_MISC_0_PREV_MAX_QP                     31:27
#define LWE2B7_CONTEXT_SAVE_MISC_0_PREV_MAX_QP_DEFAULT             (0x00000000)

// This register specifies the max packet size.
// In AP15, this register is meaningful in to mpeg4/h263 mode only.
// In H264 mode, HW updating this register with 
// appropriate value is not implemented for AP15.
#define LWE2B7_MAX_PACKET_0                       (0x130)
// Maximum Packet Size.
//  This parameter returns the size of the
//  biggest packet in 4-byte words.
#define LWE2B7_MAX_PACKET_0_MAX_PACKET                     17:0
#define LWE2B7_MAX_PACKET_0_MAX_PACKET_DEFAULT                     (0x00000000)

// This register specifies the number of clock
//  cycles to encode the last frame.
// This register may be read for diagnostics or
//  for power management.
#define LWE2B7_FRAME_CYCLE_COUNT_0                        (0x131)
// Frame Cycle Count.
// This parameter specifies the number of clock
//  cycles to encode the last frame.
//  This will return 0 if the last frame is
//  skipped/dropped.
#define LWE2B7_FRAME_CYCLE_COUNT_0_FRAME_CYCLE_COUNT                       31:0
#define LWE2B7_FRAME_CYCLE_COUNT_0_FRAME_CYCLE_COUNT_DEFAULT       (0x00000000)

// This register specifies the number of clock
//  cycles from the beginning of frame encoding
//  to the last encoded macroblock.
//  Content of this register is updated every
//  time a macroblock is encoded and is reset
//  at the start of frame encoding. However this
//  update is generated few mb earlier than
//  actual encoding.
// This register may be read for diagnostics or
//  for power management.
#define LWE2B7_MB_CYCLE_COUNT_0                   (0x132)
// MB Cycle Count.
// This parameter specifies the number of clock
//  cycles to encode from beginning of current
//  frame to last encoded macroblock in multiple
//  of 256 encoder clocks. The last encoded
//  macroblock count is specified in MB_COUNT
//  parameter.
#define LWE2B7_MB_CYCLE_COUNT_0_MB_CYCLE_COUNT                     17:0
#define LWE2B7_MB_CYCLE_COUNT_0_MB_CYCLE_COUNT_DEFAULT             (0x00000000)

// Macroblock Count.
// This parameter specifies the count of the
//  last encoded macroblock in the current
//  frame.
#define LWE2B7_MB_CYCLE_COUNT_0_MB_COUNT                   31:18
#define LWE2B7_MB_CYCLE_COUNT_0_MB_COUNT_DEFAULT                   (0x00000000)

// This register may be written when it is safe
//  to write registers to specify the length
//  of frame bitstream when resuming from
//  context switch. 
// The value of this register is updated
//  automatically by the encoder hardware
//  so when it is read at the end of a frame
//  encoding, it will return the size of the
//  last frame encoded bitstream.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_MAD_PIC_1_CBR_0                    (0x133)
// MAD PIC 1 used in GetQp
// This parameter returns the mad_pic_1
#define LWE2B7_MAD_PIC_1_CBR_0_MAD_PIC_1_CBR                       20:0
#define LWE2B7_MAD_PIC_1_CBR_0_MAD_PIC_1_CBR_DEFAULT               (0x00000000)

// This register may be written when it is safe
//  to write registers to specify the length
//  of frame bitstream when resuming from
//  context switch. 
// The value of this register is updated
//  automatically by the encoder hardware
//  so when it is read at the end of a frame
//  encoding, it will return the size of the
//  last frame encoded bitstream.
// This register must be read and saved as part
//  of context save and its state restored when
//  the context is resumed at a later time.
#define LWE2B7_MX1_CBR_0                  (0x134)
// MX1 used in GetQp
// This parameter returns the mx1
#define LWE2B7_MX1_CBR_0_MX1_CBR                   20:0
#define LWE2B7_MX1_CBR_0_MX1_CBR_DEFAULT                           (0x00000000)

// Offset 0x140

// Offset 0x160
//
// ME_BYPASS registers.
//

// For APxx this register is applicable for
//  both H.264 and MPEG4/H.263 encoding. Some
//  Parameters in this register is applicable
//  only when ME_BYPASS is set to ENABLE.
#define LWE2B7_FRAME_BIT_LEN_0                    (0x160)
// Frame Bit Length (bytes).
// This parameter must be written by CPU after
//  encoding a frame only when ME_BYPASS is
//  set to ENABLE. If ME_BYPASS is DISABLE,
//  then this parameter is ignored.
// Can this field be re-written multiple times
//  to change FRAME_SKIP_INFO ??
#define LWE2B7_FRAME_BIT_LEN_0_FRAME_BIT_LEN                       21:0
#define LWE2B7_FRAME_BIT_LEN_0_FRAME_BIT_LEN_DEFAULT               (0x00000000)

// Frame Skip Information.
// This parameter provides ability for software
//  to force frame skipping. This is useful
//  for frame-level rate control when ME_BYPASS
//  is set to ENABLE. However this is effective
//  when ME_BYPASS is set to DISABLE also.
#define LWE2B7_FRAME_BIT_LEN_0_FRAME_SKIP_INFO                     24:22
#define LWE2B7_FRAME_BIT_LEN_0_FRAME_SKIP_INFO_NO_SKIP                    (0)    // // Don't skip next frame.
// This takes precedence over other settings
//  when written.

#define LWE2B7_FRAME_BIT_LEN_0_FRAME_SKIP_INFO_SKIP1                      (1)    // // Skip next 1 frame, and then hardware will
//  reset this field to 000.

#define LWE2B7_FRAME_BIT_LEN_0_FRAME_SKIP_INFO_SKIP2                      (2)    // // Skip next 2 frames, and then hardware will
//  reset this field to 000.

#define LWE2B7_FRAME_BIT_LEN_0_FRAME_SKIP_INFO_SKIP3                      (3)    // // Skip next 3 frames, and then hardware will
//  reset this field to 000.

#define LWE2B7_FRAME_BIT_LEN_0_FRAME_SKIP_INFO_SKIP4                      (4)    // // Skip next 4 frames, and then hardware will
//  reset this field to 000.

#define LWE2B7_FRAME_BIT_LEN_0_FRAME_SKIP_INFO_SKIP5                      (5)    // // Skip next 5 frames, and then hardware will
//  reset this field to 000.

#define LWE2B7_FRAME_BIT_LEN_0_FRAME_SKIP_INFO_SKIP6                      (6)    // // Skip next 6 frames, and then hardware will
//  reset this field to 000.

#define LWE2B7_FRAME_BIT_LEN_0_FRAME_SKIP_INFO_SKIP_FRAME                 (7)    // // Skip frames continously until software
//  reset this field to 000. In this mode,
//  hardware will not automatically clear
//  this field. Software must keep track of
//  time or virtual buffer fullness to skip
//  the appropriate number of frames.
// This takes precedence over other settings
//  when written. So it is always possible to
//  force the clearing of the frame skip
//  condition by setting this field to NO_SKIP,
//  and it is possible to force frame skipping
//  by setting this field to SKIP_FRAME.
#define LWE2B7_FRAME_BIT_LEN_0_FRAME_SKIP_INFO_DEFAULT             (0x00000000)

// Last frame Qp.
// This parameter must be written by CPU after
//  encoding a frame only when ME_BYPASS is
//  set to ENABLE. If ME_BYPASS is DISABLE,
//  then this parameter is ignored.
// When ME_BYPASS=ENABLE, this parameter is
//  written by software to pass the last frame
//  average Qp which is used for motion cost
//  callwlation of next frame.
#define LWE2B7_FRAME_BIT_LEN_0_LAST_FRAME_QP                       30:25
#define LWE2B7_FRAME_BIT_LEN_0_LAST_FRAME_QP_DEFAULT               (0x00000000)

// Frame Encode Done status.
// This bit is set when FRAME_BIT_LEN is
//  written after encoding a frame in mode.
// This bit is cleared by HW after reading?
// Is Raise/Wait needed instead?
#define LWE2B7_FRAME_BIT_LEN_0_FRAME_ENC_DONE                      31:31
#define LWE2B7_FRAME_BIT_LEN_0_FRAME_ENC_DONE_DEFAULT              (0x00000000)

// New registers to control encoded bitstream manager (EBM) for APxx products.
// Offset 0x180
// The following defines EBM registers which controls the process of writing final encoded
// bitstream to the encoded bitstream (EB) buffers.
//
// Up to 16 EB buffers can be allocated (EB00 to EB15) by software and they must be
// allocated sequentially from EB00 to EBxx where xx is the last buffer index (xx = 3 to 15).
// Each EB buffer is defined with unique start address for each buffer and a common buffer
// size for all EB buffers.
// Total size of the EB buffers must be able to store two conselwtive largest encoded frames
// which is approximately 45x average bits/frame. For 30fps encoding, this is equivalent to
// about 1.5 second of target bitrate.
// The number of buffers allocated depends on how fast the host can consume these buffers.
// Interrupt is used to notify host of availability of EB buffer or encoded frame to be
// processed, so the interrupt latency must also be taken into account in determining how
// fast the host can consume the EB buffers.
// It is recommended that 8 buffers be used typically. In a "fast" system, the number of EB
// buffers can be reduced but should not be less than 4 buffers. In a "slow" system, the number
// of EB buffers can be increased up to 16 buffers.
//
// S/W activates EB buffers and H/W de-activates the EB buffers as buffers are written/filled.
// S/W processes encoded bitstreams in frame basis or buffer basis and keep track of EB
// buffers that have been processed. S/W then reactivate processed EB buffers or activate
// new EB buffers before H/W consumes all the EB buffers.  Note that H/W will not consume
// a previously deactivated EB buffer without S/W reactivation of this buffer.
//
// H/W can issue interrupt to host either when a frame is completely written to memory or
// when a buffer is completely written to memory. Note that encoded frames are written
// back to back into the EB buffers therefore a frame may start and end in the middle of
// EB buffer.
//
// It is possible to dynamically change the number of EBM buffers. Note that EBM buffers must
// still be allocated/activated sequentially. To reduce the number of EBM buffers, software
// can simply not activate the last EBM buffers after they have been deactivated by EBM.
// To increase the number of EBM buffers, software must activate new buffers instead of
// wrapping around to buffer 0. This MUST be done while buffer 0 is inactive or while EBM is
// filling buffer 0 to prevent unexpected buffer wraparound just prior to new buffers being
// allocated and therefore causing non-sequential discontinuity in the active buffers.
//

#define LWE2B7_EB00_START_ADDRESS_0                       (0x180)
// EB00 start address aligned to 16-byte
//  boundary.
#define LWE2B7_EB00_START_ADDRESS_0_EB00_START_ADDRESS                     31:4
#define LWE2B7_EB00_START_ADDRESS_0_EB00_START_ADDRESS_DEFAULT     (0x00000000)

#define LWE2B7_EB01_START_ADDRESS_0                       (0x181)
// EB01 start address aligned to 16-byte
//  boundary.
#define LWE2B7_EB01_START_ADDRESS_0_EB01_START_ADDRESS                     31:4
#define LWE2B7_EB01_START_ADDRESS_0_EB01_START_ADDRESS_DEFAULT     (0x00000000)

#define LWE2B7_EB02_START_ADDRESS_0                       (0x182)
// EB02 start address aligned to 16-byte
//  boundary.
#define LWE2B7_EB02_START_ADDRESS_0_EB02_START_ADDRESS                     31:4
#define LWE2B7_EB02_START_ADDRESS_0_EB02_START_ADDRESS_DEFAULT     (0x00000000)

#define LWE2B7_EB03_START_ADDRESS_0                       (0x183)
// EB03 start address aligned to 16-byte
//  boundary.
#define LWE2B7_EB03_START_ADDRESS_0_EB03_START_ADDRESS                     31:4
#define LWE2B7_EB03_START_ADDRESS_0_EB03_START_ADDRESS_DEFAULT     (0x00000000)

#define LWE2B7_EB04_START_ADDRESS_0                       (0x184)
// EB04 start address aligned to 16-byte
//  boundary.
#define LWE2B7_EB04_START_ADDRESS_0_EB04_START_ADDRESS                     31:4
#define LWE2B7_EB04_START_ADDRESS_0_EB04_START_ADDRESS_DEFAULT     (0x00000000)

#define LWE2B7_EB05_START_ADDRESS_0                       (0x185)
// EB05 start address aligned to 16-byte
//  boundary.
#define LWE2B7_EB05_START_ADDRESS_0_EB05_START_ADDRESS                     31:4
#define LWE2B7_EB05_START_ADDRESS_0_EB05_START_ADDRESS_DEFAULT     (0x00000000)

#define LWE2B7_EB06_START_ADDRESS_0                       (0x186)
// EB06 start address aligned to 16-byte
//  boundary.
#define LWE2B7_EB06_START_ADDRESS_0_EB06_START_ADDRESS                     31:4
#define LWE2B7_EB06_START_ADDRESS_0_EB06_START_ADDRESS_DEFAULT     (0x00000000)

#define LWE2B7_EB07_START_ADDRESS_0                       (0x187)
// EB07 start address aligned to 16-byte
//  boundary.
#define LWE2B7_EB07_START_ADDRESS_0_EB07_START_ADDRESS                     31:4
#define LWE2B7_EB07_START_ADDRESS_0_EB07_START_ADDRESS_DEFAULT     (0x00000000)

#define LWE2B7_EB08_START_ADDRESS_0                       (0x188)
// EB08 start address aligned to 16-byte
//  boundary.
#define LWE2B7_EB08_START_ADDRESS_0_EB08_START_ADDRESS                     31:4
#define LWE2B7_EB08_START_ADDRESS_0_EB08_START_ADDRESS_DEFAULT     (0x00000000)

#define LWE2B7_EB09_START_ADDRESS_0                       (0x189)
// EB09 start address aligned to 16-byte
//  boundary.
#define LWE2B7_EB09_START_ADDRESS_0_EB09_START_ADDRESS                     31:4
#define LWE2B7_EB09_START_ADDRESS_0_EB09_START_ADDRESS_DEFAULT     (0x00000000)

#define LWE2B7_EB10_START_ADDRESS_0                       (0x18a)
// EB10 start address aligned to 16-byte
//  boundary.
#define LWE2B7_EB10_START_ADDRESS_0_EB10_START_ADDRESS                     31:4
#define LWE2B7_EB10_START_ADDRESS_0_EB10_START_ADDRESS_DEFAULT     (0x00000000)

#define LWE2B7_EB11_START_ADDRESS_0                       (0x18b)
// EB11 start address aligned to 16-byte
//  boundary.
#define LWE2B7_EB11_START_ADDRESS_0_EB11_START_ADDRESS                     31:4
#define LWE2B7_EB11_START_ADDRESS_0_EB11_START_ADDRESS_DEFAULT     (0x00000000)

#define LWE2B7_EB12_START_ADDRESS_0                       (0x18c)
// EB12 start address aligned to 16-byte
//  boundary.
#define LWE2B7_EB12_START_ADDRESS_0_EB12_START_ADDRESS                     31:4
#define LWE2B7_EB12_START_ADDRESS_0_EB12_START_ADDRESS_DEFAULT     (0x00000000)

#define LWE2B7_EB13_START_ADDRESS_0                       (0x18d)
// EB13 start address aligned to 16-byte
//  boundary.
#define LWE2B7_EB13_START_ADDRESS_0_EB13_START_ADDRESS                     31:4
#define LWE2B7_EB13_START_ADDRESS_0_EB13_START_ADDRESS_DEFAULT     (0x00000000)

#define LWE2B7_EB14_START_ADDRESS_0                       (0x18e)
// EB14 start address aligned to 16-byte
//  boundary.
#define LWE2B7_EB14_START_ADDRESS_0_EB14_START_ADDRESS                     31:4
#define LWE2B7_EB14_START_ADDRESS_0_EB14_START_ADDRESS_DEFAULT     (0x00000000)

#define LWE2B7_EB15_START_ADDRESS_0                       (0x18f)
// EB15 start address aligned to 16-byte
//  boundary.
#define LWE2B7_EB15_START_ADDRESS_0_EB15_START_ADDRESS                     31:4
#define LWE2B7_EB15_START_ADDRESS_0_EB15_START_ADDRESS_DEFAULT     (0x00000000)

// Align rest of EBM regs to next 32-reg boundary to reserve 16 regs for future expansion
// if more encoded bitstream buffers are needed.
// Offset 0x1A0

#define LWE2B7_EB_SIZE_0                  (0x1a0)
// Buffer size for all EB buffers.
// This is in multiple of 16-byte words.
#define LWE2B7_EB_SIZE_0_EB_SIZE                   31:4
#define LWE2B7_EB_SIZE_0_EB_SIZE_DEFAULT                           (0x00000000)

// Bits in this register is set-only by S/W
//  and will be reset by H/W or by setting
//  EBM_INIT.
// Writing a "1" to bit xx in this register
//  will activate the corresponding encoded
//  bitstream buffer (EBxx) and set the bit.
//  Bits which are set in this registers
//  will be deactivated by the EBM logic when
//  the corresponding encoded bitstream buffer
//  has been fully written/filled by EBM logic.
//  Once EBM logic deactivates an encoded
//  bitstream buffer and reset the coresponding
//  bit in this register, EBM logic will not
//  re-use or write new data to this EB buffer
//  unless S/W reactivates the EB buffer by
//  writing a "1" to the corresponding bit in
//  this register.
// Writing a "0" to bit xx in this register
//  will have no effect at all.
// This register can be read to find out the
//  status of all active encoded bitstream
//  (EB) buffers.
#define LWE2B7_EB_ACTIVATE_0                      (0x1a1)
// Encoded bitstream buffer 00 (EB00) activate.
#define LWE2B7_EB_ACTIVATE_0_EB00_ACTIVATE                 0:0
#define LWE2B7_EB_ACTIVATE_0_EB00_ACTIVATE_NOP                    (0)    // // No operation.

#define LWE2B7_EB_ACTIVATE_0_EB00_ACTIVATE_ACTIVATE                       (1)    // // Activate EB00.
#define LWE2B7_EB_ACTIVATE_0_EB00_ACTIVATE_DEFAULT                 (0x00000000)

// Encoded bitstream buffer 01 (EB01) activate.
#define LWE2B7_EB_ACTIVATE_0_EB01_ACTIVATE                 1:1
#define LWE2B7_EB_ACTIVATE_0_EB01_ACTIVATE_NOP                    (0)    // // No operation.

#define LWE2B7_EB_ACTIVATE_0_EB01_ACTIVATE_ACTIVATE                       (1)    // // Activate EB01.
#define LWE2B7_EB_ACTIVATE_0_EB01_ACTIVATE_DEFAULT                 (0x00000000)

// Encoded bitstream buffer 02 (EB02) activate.
#define LWE2B7_EB_ACTIVATE_0_EB02_ACTIVATE                 2:2
#define LWE2B7_EB_ACTIVATE_0_EB02_ACTIVATE_NOP                    (0)    // // No operation.

#define LWE2B7_EB_ACTIVATE_0_EB02_ACTIVATE_ACTIVATE                       (1)    // // Activate EB02.
#define LWE2B7_EB_ACTIVATE_0_EB02_ACTIVATE_DEFAULT                 (0x00000000)

// Encoded bitstream buffer 03 (EB03) activate.
#define LWE2B7_EB_ACTIVATE_0_EB03_ACTIVATE                 3:3
#define LWE2B7_EB_ACTIVATE_0_EB03_ACTIVATE_NOP                    (0)    // // No operation.

#define LWE2B7_EB_ACTIVATE_0_EB03_ACTIVATE_ACTIVATE                       (1)    // // Activate EB03.
#define LWE2B7_EB_ACTIVATE_0_EB03_ACTIVATE_DEFAULT                 (0x00000000)

// Encoded bitstream buffer 04 (EB04) activate.
#define LWE2B7_EB_ACTIVATE_0_EB04_ACTIVATE                 4:4
#define LWE2B7_EB_ACTIVATE_0_EB04_ACTIVATE_NOP                    (0)    // // No operation.

#define LWE2B7_EB_ACTIVATE_0_EB04_ACTIVATE_ACTIVATE                       (1)    // // Activate EB04.
#define LWE2B7_EB_ACTIVATE_0_EB04_ACTIVATE_DEFAULT                 (0x00000000)

// Encoded bitstream buffer 05 (EB05) activate.
#define LWE2B7_EB_ACTIVATE_0_EB05_ACTIVATE                 5:5
#define LWE2B7_EB_ACTIVATE_0_EB05_ACTIVATE_NOP                    (0)    // // No operation.

#define LWE2B7_EB_ACTIVATE_0_EB05_ACTIVATE_ACTIVATE                       (1)    // // Activate EB05.
#define LWE2B7_EB_ACTIVATE_0_EB05_ACTIVATE_DEFAULT                 (0x00000000)

// Encoded bitstream buffer 06 (EB06) activate.
#define LWE2B7_EB_ACTIVATE_0_EB06_ACTIVATE                 6:6
#define LWE2B7_EB_ACTIVATE_0_EB06_ACTIVATE_NOP                    (0)    // // No operation.

#define LWE2B7_EB_ACTIVATE_0_EB06_ACTIVATE_ACTIVATE                       (1)    // // Activate EB06.
#define LWE2B7_EB_ACTIVATE_0_EB06_ACTIVATE_DEFAULT                 (0x00000000)

// Encoded bitstream buffer 07 (EB07) activate.
#define LWE2B7_EB_ACTIVATE_0_EB07_ACTIVATE                 7:7
#define LWE2B7_EB_ACTIVATE_0_EB07_ACTIVATE_NOP                    (0)    // // No operation.

#define LWE2B7_EB_ACTIVATE_0_EB07_ACTIVATE_ACTIVATE                       (1)    // // Activate EB07.
#define LWE2B7_EB_ACTIVATE_0_EB07_ACTIVATE_DEFAULT                 (0x00000000)

// Encoded bitstream buffer 08 (EB08) activate.
#define LWE2B7_EB_ACTIVATE_0_EB08_ACTIVATE                 8:8
#define LWE2B7_EB_ACTIVATE_0_EB08_ACTIVATE_NOP                    (0)    // // No operation.

#define LWE2B7_EB_ACTIVATE_0_EB08_ACTIVATE_ACTIVATE                       (1)    // // Activate EB08.
#define LWE2B7_EB_ACTIVATE_0_EB08_ACTIVATE_DEFAULT                 (0x00000000)

// Encoded bitstream buffer 09 (EB09) activate.
#define LWE2B7_EB_ACTIVATE_0_EB09_ACTIVATE                 9:9
#define LWE2B7_EB_ACTIVATE_0_EB09_ACTIVATE_NOP                    (0)    // // No operation.

#define LWE2B7_EB_ACTIVATE_0_EB09_ACTIVATE_ACTIVATE                       (1)    // // Activate EB09.
#define LWE2B7_EB_ACTIVATE_0_EB09_ACTIVATE_DEFAULT                 (0x00000000)

// Encoded bitstream buffer 10 (EB10) activate.
#define LWE2B7_EB_ACTIVATE_0_EB10_ACTIVATE                 10:10
#define LWE2B7_EB_ACTIVATE_0_EB10_ACTIVATE_NOP                    (0)    // // No operation.

#define LWE2B7_EB_ACTIVATE_0_EB10_ACTIVATE_ACTIVATE                       (1)    // // Activate EB10.
#define LWE2B7_EB_ACTIVATE_0_EB10_ACTIVATE_DEFAULT                 (0x00000000)

// Encoded bitstream buffer 11 (EB11) activate.
#define LWE2B7_EB_ACTIVATE_0_EB11_ACTIVATE                 11:11
#define LWE2B7_EB_ACTIVATE_0_EB11_ACTIVATE_NOP                    (0)    // // No operation.

#define LWE2B7_EB_ACTIVATE_0_EB11_ACTIVATE_ACTIVATE                       (1)    // // Activate EB11.
#define LWE2B7_EB_ACTIVATE_0_EB11_ACTIVATE_DEFAULT                 (0x00000000)

// Encoded bitstream buffer 12 (EB12) activate.
#define LWE2B7_EB_ACTIVATE_0_EB12_ACTIVATE                 12:12
#define LWE2B7_EB_ACTIVATE_0_EB12_ACTIVATE_NOP                    (0)    // // No operation.

#define LWE2B7_EB_ACTIVATE_0_EB12_ACTIVATE_ACTIVATE                       (1)    // // Activate EB12.
#define LWE2B7_EB_ACTIVATE_0_EB12_ACTIVATE_DEFAULT                 (0x00000000)

// Encoded bitstream buffer 13 (EB13) activate.
#define LWE2B7_EB_ACTIVATE_0_EB13_ACTIVATE                 13:13
#define LWE2B7_EB_ACTIVATE_0_EB13_ACTIVATE_NOP                    (0)    // // No operation.

#define LWE2B7_EB_ACTIVATE_0_EB13_ACTIVATE_ACTIVATE                       (1)    // // Activate EB13.
#define LWE2B7_EB_ACTIVATE_0_EB13_ACTIVATE_DEFAULT                 (0x00000000)

// Encoded bitstream buffer 14 (EB14) activate.
#define LWE2B7_EB_ACTIVATE_0_EB14_ACTIVATE                 14:14
#define LWE2B7_EB_ACTIVATE_0_EB14_ACTIVATE_NOP                    (0)    // // No operation.

#define LWE2B7_EB_ACTIVATE_0_EB14_ACTIVATE_ACTIVATE                       (1)    // // Activate EB14.
#define LWE2B7_EB_ACTIVATE_0_EB14_ACTIVATE_DEFAULT                 (0x00000000)

// Encoded bitstream buffer 15 (EB15) activate.
#define LWE2B7_EB_ACTIVATE_0_EB15_ACTIVATE                 15:15
#define LWE2B7_EB_ACTIVATE_0_EB15_ACTIVATE_NOP                    (0)    // // No operation.

#define LWE2B7_EB_ACTIVATE_0_EB15_ACTIVATE_ACTIVATE                       (1)    // // Activate EB15.
#define LWE2B7_EB_ACTIVATE_0_EB15_ACTIVATE_DEFAULT                 (0x00000000)

// Reserve 2 registers for future expansion.

#define LWE2B7_EBM_CONTROL_0                      (0x1a4)
// EBM Initialization.
// This bit can be used to do software reset
// of EBM logic. Note that EBM is also reset
// when MPE hardware reset is active.
#define LWE2B7_EBM_CONTROL_0_EBM_INIT                      0:0
#define LWE2B7_EBM_CONTROL_0_EBM_INIT_EBM_ACTIVE                  (0)    // // EBM is active (enabled).

#define LWE2B7_EBM_CONTROL_0_EBM_INIT_EBM_RESET                   (1)    // // EBM is reset (disabled).
// This will reset all bits in EB_ACTIVATE
// register (deactivate all EBxx buffers),
// reset all fields in LAST_EB_BUFFER_STATUS
// register to 31 and finally reset 
// LAST_EB_FRAME_BUFFER field of 
// LAST_EB_FRAME_STATUS register to 31 as well.
#define LWE2B7_EBM_CONTROL_0_EBM_INIT_DEFAULT                      (0x00000000)

// This register is read-only.
#define LWE2B7_LAST_EB_BUFFER_STATUS_0                    (0x1a5)
// Last EB buffer status.
// This field can be read by software to
//  indicate the last EB buffer index written
//  or filled by EBM logic.
#define LWE2B7_LAST_EB_BUFFER_STATUS_0_LAST_EB_BUFFER_STATUS                       4:0
#define LWE2B7_LAST_EB_BUFFER_STATUS_0_LAST_EB_BUFFER_STATUS_DEFAULT       (0x00000000)

// Last EB buffer wraparound status.
// This field can be read by software to
//  indicate the last EB buffer index where
//  EBM wraparound while writing by EB buffers.
#define LWE2B7_LAST_EB_BUFFER_STATUS_0_LAST_EB_BUFFER_WRAP                 12:8
#define LWE2B7_LAST_EB_BUFFER_STATUS_0_LAST_EB_BUFFER_WRAP_DEFAULT (0x00000000)

// This register is read-only.
#define LWE2B7_LAST_EB_FRAME_STATUS_0                     (0x1a6)
// Last EB frame status.
// This field can be read by software to
//  indicate the last EB frame count written
//  or filled by EBM logic.
#define LWE2B7_LAST_EB_FRAME_STATUS_0_LAST_EB_FRAME_STATUS                 15:0
#define LWE2B7_LAST_EB_FRAME_STATUS_0_LAST_EB_FRAME_STATUS_DEFAULT (0x00000000)

// Last EB buffer of the last EB frame.
// This field can be read by software to
//  indicate the last EB buffer index where
//  the corresponding last EB Frame is written.
#define LWE2B7_LAST_EB_FRAME_STATUS_0_LAST_EB_FRAME_BUFFER                 28:24
#define LWE2B7_LAST_EB_FRAME_STATUS_0_LAST_EB_FRAME_BUFFER_DEFAULT (0x00000000)

// Offset 0x200;
// CYA Registers.

// Content of this register should not be
//  changed (always at its reset value).
#define LWE2B7_INTERNAL_BIAS_MULTIPLIER_0                 (0x200)
// Bias multiplier for I4x4.
#define LWE2B7_INTERNAL_BIAS_MULTIPLIER_0_I4x4                     7:0
#define LWE2B7_INTERNAL_BIAS_MULTIPLIER_0_I4x4_DEFAULT             (0x00000000)

// Bias multiplier for most probable I4x4.
#define LWE2B7_INTERNAL_BIAS_MULTIPLIER_0_MOST_PROBABLE                    14:10
#define LWE2B7_INTERNAL_BIAS_MULTIPLIER_0_MOST_PROBABLE_DEFAULT    (0x00000000)

// Bias multiplier for skip.
// Recommended value is 0.
#define LWE2B7_INTERNAL_BIAS_MULTIPLIER_0_SKIP                     24:20
#define LWE2B7_INTERNAL_BIAS_MULTIPLIER_0_SKIP_DEFAULT             (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0                   (0x201)
#define LWE2B7_CLK_OVERRIDE_A_0_mperc_clk_ovr                      0:0
#define LWE2B7_CLK_OVERRIDE_A_0_mperc_clk_ovr_OFF                 (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_mperc_clk_ovr_ON                  (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_mperc_clk_ovr_DEFAULT              (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_mpercsr_clk_ovr                    1:1
#define LWE2B7_CLK_OVERRIDE_A_0_mpercsr_clk_ovr_OFF                       (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_mpercsr_clk_ovr_ON                        (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_mpercsr_clk_ovr_DEFAULT            (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_memrd_arb_clk_ovr                  2:2
#define LWE2B7_CLK_OVERRIDE_A_0_memrd_arb_clk_ovr_OFF                     (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_memrd_arb_clk_ovr_ON                      (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_memrd_arb_clk_ovr_DEFAULT          (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_intraref_clk_ovr                   3:3
#define LWE2B7_CLK_OVERRIDE_A_0_intraref_clk_ovr_OFF                      (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_intraref_clk_ovr_ON                       (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_intraref_clk_ovr_DEFAULT           (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_ccache_clk_ovr                     4:4
#define LWE2B7_CLK_OVERRIDE_A_0_ccache_clk_ovr_OFF                        (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_ccache_clk_ovr_ON                 (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_ccache_clk_ovr_DEFAULT             (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_fclk_ovr                    5:5
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_fclk_ovr_OFF                       (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_fclk_ovr_ON                        (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_fclk_ovr_DEFAULT            (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_wclk_ovr                    6:6
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_wclk_ovr_OFF                       (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_wclk_ovr_ON                        (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_wclk_ovr_DEFAULT            (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_rclk_ovr                    7:7
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_rclk_ovr_OFF                       (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_rclk_ovr_ON                        (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_rclk_ovr_DEFAULT            (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram0_clk_ovr                8:8
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram0_clk_ovr_OFF                   (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram0_clk_ovr_ON                    (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram0_clk_ovr_DEFAULT        (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram1_clk_ovr                9:9
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram1_clk_ovr_OFF                   (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram1_clk_ovr_ON                    (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram1_clk_ovr_DEFAULT        (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram2_clk_ovr                10:10
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram2_clk_ovr_OFF                   (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram2_clk_ovr_ON                    (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram2_clk_ovr_DEFAULT        (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram3_clk_ovr                11:11
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram3_clk_ovr_OFF                   (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram3_clk_ovr_ON                    (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram3_clk_ovr_DEFAULT        (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram4_clk_ovr                12:12
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram4_clk_ovr_OFF                   (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram4_clk_ovr_ON                    (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_ram4_clk_ovr_DEFAULT        (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_predbuf_ram_clk_ovr                13:13
#define LWE2B7_CLK_OVERRIDE_A_0_predbuf_ram_clk_ovr_OFF                   (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_predbuf_ram_clk_ovr_ON                    (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_predbuf_ram_clk_ovr_DEFAULT        (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_predbuf_clk_ovr                    14:14
#define LWE2B7_CLK_OVERRIDE_A_0_predbuf_clk_ovr_OFF                       (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_predbuf_clk_ovr_ON                        (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_predbuf_clk_ovr_DEFAULT            (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_msc_clk_ovr                15:15
#define LWE2B7_CLK_OVERRIDE_A_0_msc_clk_ovr_OFF                   (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_msc_clk_ovr_ON                    (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_msc_clk_ovr_DEFAULT                (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_ib_clk_ovr                 16:16
#define LWE2B7_CLK_OVERRIDE_A_0_ib_clk_ovr_OFF                    (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_ib_clk_ovr_ON                     (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_ib_clk_ovr_DEFAULT                 (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_pmem_clk_ovr                       17:17
#define LWE2B7_CLK_OVERRIDE_A_0_pmem_clk_ovr_OFF                  (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_pmem_clk_ovr_ON                   (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_pmem_clk_ovr_DEFAULT               (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_fphpgen_clk_ovr                    18:18
#define LWE2B7_CLK_OVERRIDE_A_0_fphpgen_clk_ovr_OFF                       (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_fphpgen_clk_ovr_ON                        (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_fphpgen_clk_ovr_DEFAULT            (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_satd_clk_ovr                       19:19
#define LWE2B7_CLK_OVERRIDE_A_0_satd_clk_ovr_OFF                  (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_satd_clk_ovr_ON                   (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_satd_clk_ovr_DEFAULT               (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_predbuf_clk_ovr                     20:20
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_predbuf_clk_ovr_OFF                        (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_rcache_predbuf_clk_ovr_ON                 (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_rcache_predbuf_clk_ovr_DEFAULT     (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_regs_clk_ovr                       21:21
#define LWE2B7_CLK_OVERRIDE_A_0_regs_clk_ovr_OFF                  (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_regs_clk_ovr_ON                   (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_regs_clk_ovr_DEFAULT               (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_fract_clk_ovr                      22:22
#define LWE2B7_CLK_OVERRIDE_A_0_fract_clk_ovr_OFF                 (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_fract_clk_ovr_ON                  (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_fract_clk_ovr_DEFAULT              (0x00000000)

#define LWE2B7_CLK_OVERRIDE_A_0_act_clk_ovr                23:23
#define LWE2B7_CLK_OVERRIDE_A_0_act_clk_ovr_OFF                   (0)    // // Do not override clock

#define LWE2B7_CLK_OVERRIDE_A_0_act_clk_ovr_ON                    (1)    // // Override clock
#define LWE2B7_CLK_OVERRIDE_A_0_act_clk_ovr_DEFAULT                (0x00000000)

// Offset 0x300

// Offset 0x380

// Offset 0x3C0;
// Memory Client Interface Fifo Control Register.
// The registers below allow to optimize the synchronization timing in
// the memory client asynchronous fifos. When they can be used depend on
// the client and memory controller clock ratio.
// Additionally, the RDMC_RDFAST/RDCL_RDFAST fields can increase power
// consumption if the asynchronous fifo is implemented as a real ram.
// There is no power impact on latch-based fifos. Flipflop-based fifos
// do not use these fields.
// See recommended settings below.
//
// !! IMPORTANT !!
// The register fields can only be changed when the memory client async
// fifos are empty.
//
// The register field ending with WRCL_MCLE2X (if any) can be set to improve
// async fifo synchronization on the write side by one client clock cycle if
// the memory controller clock frequency is less or equal to twice the client
// clock frequency:
//
//      mcclk_freq <= 2 * clientclk_freq
//
// The register field ending with WRMC_CLLE2X (if any) can be set to improve
// async fifo synchronization on the write side by one memory controller clock
// cycle if the client clock frequency is less or equal to twice the memory
// controller clock frequency:
//
//      clientclk_freq <= 2 * mcclk_freq
//
// The register field ending with RDMC_RDFAST (if any) can be set to improve async
// fifo synchronization on the read side by one memory controller clock cycle.
//
// !! WARNING !!
// RDMC_RDFAST can be used along with WRCL_MCLE2X only when:
//
//       mcclk_freq <= clientclk_freq
//
// The register field ending with RDCL_RDFAST (if any) can be set to improve async
// fifo synchronization on the read side by one client clock cycle.
//
// !! WARNING !!
// RDCL_RDFAST can be used along with WRMC_CLLE2X only when:
//
//       clientclk_freq <= mcclk_freq
//
// RECOMMENDED SETTINGS
// # Client writing to fifo, memory controller reading from fifo
// - mcclk_freq <= clientclk_freq
//     You can enable both RDMC_RDFAST and WRCL_CLLE2X. If one of the fifos is
//     a real ram and power is a concern, you should avoid enabling RDMC_RDFAST.
// - clientclk_freq < mcclk_freq <= 2 * clientclk_freq
//     You can enable RDMC_RDFAST or WRCL_MCLE2X, but because the client clock
//     is slower, you should enable only WRCL_MCLE2X.
// - 2 * clientclk_freq < mcclk_freq
//     You can only enable RDMC_RDFAST. If one of the fifos is a real ram and
//     power is a concern, you should avoid enabling RDMC_RDFAST.
//
// # Memory controller writing to fifo, client reading from fifo
// - clientclk_freq <= mcclk_freq
//     You can enable both RDCL_RDFAST and WRMC_CLLE2X. If one of the fifos is
//     a real ram and power is a concern, you should avoid enabling RDCL_RDFAST.
// - mcclk_freq < clientclk_freq <= 2 * mcclk_freq
//     You can enable RDCL_RDFAST or WRMC_CLLE2X, but because the memory controller
//     clock is slower, you should enable only WRMC_CLLE2X.
// - 2 * mcclk_freq < clientclk_freq
//     You can only enable RDCL_RDFAST. If one of the fifos is a real ram and
//     power is a concern, you should avoid enabling RDCL_RDFAST.
//
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0                      (0x3c0)
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_WRCL_MCLE2X                0:0
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_WRCL_MCLE2X_INIT_ENUM                     DISABLE
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_WRCL_MCLE2X_DISABLE                       (0)
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_WRCL_MCLE2X_ENABLE                        (1)
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_WRCL_MCLE2X_DEFAULT    (0x00000000)

#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_RDMC_RDFAST                1:1
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_RDMC_RDFAST_INIT_ENUM                     DISABLE
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_RDMC_RDFAST_DISABLE                       (0)
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_RDMC_RDFAST_ENABLE                        (1)
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_RDMC_RDFAST_DEFAULT    (0x00000000)

#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_WRMC_CLLE2X                2:2
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_WRMC_CLLE2X_INIT_ENUM                     DISABLE
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_WRMC_CLLE2X_DISABLE                       (0)
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_WRMC_CLLE2X_ENABLE                        (1)
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_WRMC_CLLE2X_DEFAULT    (0x00000000)

#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_RDCL_RDFAST                3:3
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_RDCL_RDFAST_INIT_ENUM                     DISABLE
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_RDCL_RDFAST_DISABLE                       (0)
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_RDCL_RDFAST_ENABLE                        (1)
#define LWE2B7_MPEA_MCCIF_FIFOCTRL_0_MPEA_MCCIF_RDCL_RDFAST_DEFAULT        (0x00000000)

#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0                      (0x3c4)
#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_WRCL_MCLE2X                0:0
#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_WRCL_MCLE2X_INIT_ENUM                     DISABLE
#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_WRCL_MCLE2X_DISABLE                       (0)
#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_WRCL_MCLE2X_ENABLE                        (1)
#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_WRCL_MCLE2X_DEFAULT        (0x00000000)

#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_RDMC_RDFAST                1:1
#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_RDMC_RDFAST_INIT_ENUM                     DISABLE
#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_RDMC_RDFAST_DISABLE                       (0)
#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_RDMC_RDFAST_ENABLE                        (1)
#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_RDMC_RDFAST_DEFAULT        (0x00000000)

#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_WRMC_CLLE2X                2:2
#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_WRMC_CLLE2X_INIT_ENUM                     DISABLE
#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_WRMC_CLLE2X_DISABLE                       (0)
#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_WRMC_CLLE2X_ENABLE                        (1)
#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_WRMC_CLLE2X_DEFAULT        (0x00000000)

#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_RDCL_RDFAST                3:3
#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_RDCL_RDFAST_INIT_ENUM                     DISABLE
#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_RDCL_RDFAST_DISABLE                       (0)
#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_RDCL_RDFAST_ENABLE                        (1)
#define LWE2B7_MPEB_MCCIF_FIFOCTRL_0_MPEB_MCCIF_RDCL_RDFAST_DEFAULT        (0x00000000)

#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0                      (0x3c8)
#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_WRCL_MCLE2X                0:0
#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_WRCL_MCLE2X_INIT_ENUM                     DISABLE
#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_WRCL_MCLE2X_DISABLE                       (0)
#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_WRCL_MCLE2X_ENABLE                        (1)
#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_WRCL_MCLE2X_DEFAULT        (0x00000000)

#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_RDMC_RDFAST                1:1
#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_RDMC_RDFAST_INIT_ENUM                     DISABLE
#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_RDMC_RDFAST_DISABLE                       (0)
#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_RDMC_RDFAST_ENABLE                        (1)
#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_RDMC_RDFAST_DEFAULT        (0x00000000)

#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_WRMC_CLLE2X                2:2
#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_WRMC_CLLE2X_INIT_ENUM                     DISABLE
#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_WRMC_CLLE2X_DISABLE                       (0)
#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_WRMC_CLLE2X_ENABLE                        (1)
#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_WRMC_CLLE2X_DEFAULT        (0x00000000)

#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_RDCL_RDFAST                3:3
#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_RDCL_RDFAST_INIT_ENUM                     DISABLE
#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_RDCL_RDFAST_DISABLE                       (0)
#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_RDCL_RDFAST_ENABLE                        (1)
#define LWE2B7_MPEC_MCCIF_FIFOCTRL_0_MPEC_MCCIF_RDCL_RDFAST_DEFAULT        (0x00000000)

// Write Coalescing Time-Out Register
// This register exists only for write clients. Reset value defaults to 
// to 50 for most clients, but may be different for certain clients.
// Write coalescing happens inside the memory client.
// Coalescing means two (LW_MC_MW/2)-bit requests are grouped together in one LW_MC_MW-bit request.
// The register value indicates how many cycles a first write request is going to wait
// for a subsequent one for possible coalescing. The coalescing can only happen
// if the request addresses are compatible. A value of zero means that coalescing is off
// and requests are sent right away to the memory controller.
// Write coalescing can have a very significant impact performance when accessing the internal memory,
// because its memory word is LW_MC_WM-bit wide. Grouping two half-word accesses is
// much more efficient, because the two accesses would actually have taken three cycles,
// due to a stall when accessing the same memory bank. It also reduces the number of
// accessing (one instead of two), freeing up internal memory bandwidth for other accesses.
// The impact on external memory accesses is not as significant as the burst access is for
// LW_MC_MW/2 bits. But a coalesced write guarantees two conselwtive same page accesses
// which is good for external memory bandwidth utilization.
// The write coalescing time-out should be programmed depending on the client behavior.
// The first write is obviously delayed by an amount of client cycles equal to the time-out value.
// Note that writes tagged by the client (i.e. the client expects a write response, usually
// for coherency), and the last write of a block transfer are not delayed.
// They only have a one-cycle opportunity to get coalesced.
//
#define LWE2B7_TIMEOUT_WCOAL_MPEB_0                       (0x3d0)
#define LWE2B7_TIMEOUT_WCOAL_MPEB_0_MPEUNIFBW_WCOAL_TMVAL                  7:0
#define LWE2B7_TIMEOUT_WCOAL_MPEB_0_MPEUNIFBW_WCOAL_TMVAL_DEFAULT          (0x00000000)

// Write Coalescing Time-Out Register
// This register exists only for write clients. Reset value defaults to 
// to 50 for most clients, but may be different for certain clients.
// Write coalescing happens inside the memory client.
// Coalescing means two (LW_MC_MW/2)-bit requests are grouped together in one LW_MC_MW-bit request.
// The register value indicates how many cycles a first write request is going to wait
// for a subsequent one for possible coalescing. The coalescing can only happen
// if the request addresses are compatible. A value of zero means that coalescing is off
// and requests are sent right away to the memory controller.
// Write coalescing can have a very significant impact performance when accessing the internal memory,
// because its memory word is LW_MC_WM-bit wide. Grouping two half-word accesses is
// much more efficient, because the two accesses would actually have taken three cycles,
// due to a stall when accessing the same memory bank. It also reduces the number of
// accessing (one instead of two), freeing up internal memory bandwidth for other accesses.
// The impact on external memory accesses is not as significant as the burst access is for
// LW_MC_MW/2 bits. But a coalesced write guarantees two conselwtive same page accesses
// which is good for external memory bandwidth utilization.
// The write coalescing time-out should be programmed depending on the client behavior.
// The first write is obviously delayed by an amount of client cycles equal to the time-out value.
// Note that writes tagged by the client (i.e. the client expects a write response, usually
// for coherency), and the last write of a block transfer are not delayed.
// They only have a one-cycle opportunity to get coalesced.
//

#define LWE2B7_TIMEOUT_WCOAL_MPEC_0                       (0x3d4)
#define LWE2B7_TIMEOUT_WCOAL_MPEC_0_MPECSWR_WCOAL_TMVAL                    7:0
#define LWE2B7_TIMEOUT_WCOAL_MPEC_0_MPECSWR_WCOAL_TMVAL_DEFAULT            (0x00000000)

// Memory Client High-Priority Control Register
// This register exists only for clients with high-priority. Reset values are 0 (disabled).
// The high-priority should be enabled for hard real-time clients only. The values to program
// depend on the client bandwidth requirement and the client versus memory controllers clolck ratio.
// The high-priority is set if the number of entries in the return data fifo is under the threshold.
// The high-priority assertion can be delayed by a number of memory clock cycles indicated by the timer.
// This creates an hysteresis effect, avoiding setting the high-priority for very short periods of time,
// which may or may not be desirable.

#define LWE2B7_MCCIF_MPECSRD_HP_0                 (0x3fc)
#define LWE2B7_MCCIF_MPECSRD_HP_0_CSR_MPECSRD2MC_HPTH                      3:0
#define LWE2B7_MCCIF_MPECSRD_HP_0_CSR_MPECSRD2MC_HPTH_DEFAULT      (0x00000000)
#define LWE2B7_MCCIF_MPECSRD_HP_0_CSR_MPECSRD2MC_HPTM                      21:16
#define LWE2B7_MCCIF_MPECSRD_HP_0_CSR_MPECSRD2MC_HPTM_DEFAULT      (0x00000000)

// Memory Client Hysteresis Control Register
// This register exists only for clients with hysteresis.
// BUG 505006: Hysteresis configuration can only be updated when memory traffic is idle.
// HYST_EN can be used to turn on or off the hysteresis logic.
// HYST_REQ_TH is the threshold of pending requests required
//   before allowing them to pass through
//   (overriden after HYST_REQ_TM cycles).
// Hysteresis logic will stop holding request after (1<<HYST_TM) cycles
//   (this should not have to be used and is only a WAR for
//   unexpected hangs).
// Deep hysteresis is a second level of hysteresis on a longer time-frame.
//   DHYST_TH is the size of the read burst (requests are held until there
//   is space for the entire burst in the return data fifo).
//   During a burst period, if there are no new requests after
//   DHYST_TM cycles, then the burst is terminated early.

#define LWE2B7_MCCIF_MPEUNIFBR_HYST_0                     (0x43c)
#define LWE2B7_MCCIF_MPEUNIFBR_HYST_0_CBR_MPEUNIFBR2MC_HYST_REQ_TM                 7:0
#define LWE2B7_MCCIF_MPEUNIFBR_HYST_0_CBR_MPEUNIFBR2MC_HYST_REQ_TM_DEFAULT (0x00000000)
#define LWE2B7_MCCIF_MPEUNIFBR_HYST_0_CBR_MPEUNIFBR2MC_DHYST_TM                    15:8
#define LWE2B7_MCCIF_MPEUNIFBR_HYST_0_CBR_MPEUNIFBR2MC_DHYST_TM_DEFAULT    (0x00000000)
#define LWE2B7_MCCIF_MPEUNIFBR_HYST_0_CBR_MPEUNIFBR2MC_DHYST_TH                    23:16
#define LWE2B7_MCCIF_MPEUNIFBR_HYST_0_CBR_MPEUNIFBR2MC_DHYST_TH_DEFAULT    (0x00000000)
#define LWE2B7_MCCIF_MPEUNIFBR_HYST_0_CBR_MPEUNIFBR2MC_HYST_TM                     27:24
#define LWE2B7_MCCIF_MPEUNIFBR_HYST_0_CBR_MPEUNIFBR2MC_HYST_TM_DEFAULT     (0x00000000)
#define LWE2B7_MCCIF_MPEUNIFBR_HYST_0_CBR_MPEUNIFBR2MC_HYST_REQ_TH                 30:28
#define LWE2B7_MCCIF_MPEUNIFBR_HYST_0_CBR_MPEUNIFBR2MC_HYST_REQ_TH_DEFAULT (0x00000000)
#define LWE2B7_MCCIF_MPEUNIFBR_HYST_0_CBR_MPEUNIFBR2MC_HYST_EN                     31:31
#define LWE2B7_MCCIF_MPEUNIFBR_HYST_0_CBR_MPEUNIFBR2MC_HYST_EN_INIT_ENUM                  ENABLE
#define LWE2B7_MCCIF_MPEUNIFBR_HYST_0_CBR_MPEUNIFBR2MC_HYST_EN_ENABLE                     (1)
#define LWE2B7_MCCIF_MPEUNIFBR_HYST_0_CBR_MPEUNIFBR2MC_HYST_EN_DISABLE                    (0)
#define LWE2B7_MCCIF_MPEUNIFBR_HYST_0_CBR_MPEUNIFBR2MC_HYST_EN_DEFAULT     (0x00000000)

// Memory Client Hysteresis Control Register
// This register exists only for clients with hysteresis.
// BUG 505006: Hysteresis configuration can only be updated when memory traffic is idle.
// HYST_EN can be used to turn on or off the hysteresis logic.
// HYST_REQ_TH is the threshold of pending requests required
//   before allowing them to pass through
//   (overriden after HYST_REQ_TM cycles).

#define LWE2B7_MCCIF_MPEUNIFBW_HYST_0                     (0x43d)
#define LWE2B7_MCCIF_MPEUNIFBW_HYST_0_CBW_MPEUNIFBW2MC_HYST_REQ_TM                 11:0
#define LWE2B7_MCCIF_MPEUNIFBW_HYST_0_CBW_MPEUNIFBW2MC_HYST_REQ_TM_DEFAULT (0x00000000)
#define LWE2B7_MCCIF_MPEUNIFBW_HYST_0_CBW_MPEUNIFBW2MC_HYST_REQ_TH                 30:28
#define LWE2B7_MCCIF_MPEUNIFBW_HYST_0_CBW_MPEUNIFBW2MC_HYST_REQ_TH_DEFAULT (0x00000000)
#define LWE2B7_MCCIF_MPEUNIFBW_HYST_0_CBW_MPEUNIFBW2MC_HYST_EN                     31:31
#define LWE2B7_MCCIF_MPEUNIFBW_HYST_0_CBW_MPEUNIFBW2MC_HYST_EN_INIT_ENUM                  ENABLE
#define LWE2B7_MCCIF_MPEUNIFBW_HYST_0_CBW_MPEUNIFBW2MC_HYST_EN_ENABLE                     (1)
#define LWE2B7_MCCIF_MPEUNIFBW_HYST_0_CBW_MPEUNIFBW2MC_HYST_EN_DISABLE                    (0)
#define LWE2B7_MCCIF_MPEUNIFBW_HYST_0_CBW_MPEUNIFBW2MC_HYST_EN_DEFAULT     (0x00000000)
// Stream 1
//base MPEA1X 0x1000;
//#define LW_MPE_CONTEXT_1
//#include "armpea_class.spec"
//#undef LW_MPE_CONTEXT_1
//align 128;

//
// REGISTER LIST
//
#define LIST_ARMPE_REGS(_op_) \
_op_(LWE2B7_INCR_SYNCPT_0) \
_op_(LWE2B7_INCR_SYNCPT_CNTRL_0) \
_op_(LWE2B7_INCR_SYNCPT_ERROR_0) \
_op_(LWE2B7_CTXSW_0) \
_op_(LWE2B7_CONT_SYNCPT_EBM_EOF_0) \
_op_(LWE2B7_CONT_SYNCPT_EBM_EOB_0) \
_op_(LWE2B7_CONT_SYNCPT_XFER_DONE_0) \
_op_(LWE2B7_CONT_SYNCPT_RD_DONE_0) \
_op_(LWE2B7_RAISE_BUFFER_0) \
_op_(LWE2B7_RAISE_FRAME_0) \
_op_(LWE2B7_COMMAND_0) \
_op_(LWE2B7_SHADOW_REG_EN_0) \
_op_(LWE2B7_NEW_BUFFER_0) \
_op_(LWE2B7_SEQ_PIC_CTRL_0) \
_op_(LWE2B7_FORCE_I_FRAME_0) \
_op_(LWE2B7_INTSTATUS_0) \
_op_(LWE2B7_INTMASK_0) \
_op_(LWE2B7_VOL_CTRL_0) \
_op_(LWE2B7_WIDTH_HEIGHT_0) \
_op_(LWE2B7_SEQ_PARAMETERS_0) \
_op_(LWE2B7_LOG2_MAX_FRAME_NUM_MINUS4_0) \
_op_(LWE2B7_CROP_LR_OFFSET_0) \
_op_(LWE2B7_CROP_TB_OFFSET_0) \
_op_(LWE2B7_PIC_PARAMETERS_0) \
_op_(LWE2B7_PIC_PARAM_SET_ID_0) \
_op_(LWE2B7_SEQ_PARAM_SET_ID_0) \
_op_(LWE2B7_NUM_REF_IDX_ACTIVE_0) \
_op_(LWE2B7_IDR_PIC_ID_0) \
_op_(LWE2B7_IB_BUFFER_ADDR_MODE_0) \
_op_(LWE2B7_IB_OFFSET_LUMA_0) \
_op_(LWE2B7_IB_OFFSET_CHROMA_0) \
_op_(LWE2B7_FIRST_IB_OFFSET_LUMA_0) \
_op_(LWE2B7_FIRST_IB_OFFSET_CHROMA_0) \
_op_(LWE2B7_FIRST_IB_V_SIZE_0) \
_op_(LWE2B7_IB0_START_ADDR_Y_0) \
_op_(LWE2B7_IB0_START_ADDR_U_0) \
_op_(LWE2B7_IB0_START_ADDR_V_0) \
_op_(LWE2B7_IB0_SIZE_0) \
_op_(LWE2B7_IB0_LINE_STRIDE_0) \
_op_(LWE2B7_IB0_BUFFER_STRIDE_LUMA_0) \
_op_(LWE2B7_IB0_BUFFER_STRIDE_CHROMA_0) \
_op_(LWE2B7_REF_BUFFER_ADDR_MODE_0) \
_op_(LWE2B7_REF_Y_START_ADDR_0) \
_op_(LWE2B7_REF_U_START_ADDR_0) \
_op_(LWE2B7_REF_V_START_ADDR_0) \
_op_(LWE2B7_REF_STRIDE_0) \
_op_(LWE2B7_REF_BUFFER_LEN_0) \
_op_(LWE2B7_REF_BUFFER_IDX_0) \
_op_(LWE2B7_REF_WR_MBROW_0) \
_op_(LWE2B7_REF_RD_MBROW_0) \
_op_(LWE2B7_IPRED_ROW_ADDR_0) \
_op_(LWE2B7_DBLK_PARAM_ADDR_0) \
_op_(LWE2B7_ACDC_ADDR_0) \
_op_(LWE2B7_FRAME_CTRL_0) \
_op_(LWE2B7_FRAME_TYPE_0) \
_op_(LWE2B7_FRAME_PATTERN_0) \
_op_(LWE2B7_FRAME_INDEX_0) \
_op_(LWE2B7_ENC_FRAME_NUM_0) \
_op_(LWE2B7_FRAME_NUM_0) \
_op_(LWE2B7_FRAME_NUM_GOP_0) \
_op_(LWE2B7_NUM_SLICE_GROUPS_0) \
_op_(LWE2B7_DMA_BUFFER_ADDR_0) \
_op_(LWE2B7_DMA_LIST_ADDR_0) \
_op_(LWE2B7_DMA_BUFFER_SIZE_0) \
_op_(LWE2B7_DMA_LIST_SIZE_0) \
_op_(LWE2B7_PIC_INIT_Q_0) \
_op_(LWE2B7_MAX_MIN_QP_I_0) \
_op_(LWE2B7_MAX_MIN_QP_P_0) \
_op_(LWE2B7_SLICE_PARAMS_0) \
_op_(LWE2B7_NUM_OF_UNITS_0) \
_op_(LWE2B7_TOP_LEFT_0) \
_op_(LWE2B7_BOTTOM_RIGHT_0) \
_op_(LWE2B7_CHANGE_RATE_0) \
_op_(LWE2B7_DMA_BUFFER_STATUS_0) \
_op_(LWE2B7_DMA_LIST_STATUS_0) \
_op_(LWE2B7_SLICE_MAP_OFFSET_A_0) \
_op_(LWE2B7_SLICE_MAP_OFFSET_B_0) \
_op_(LWE2B7_MOT_SEARCH_CTRL_0) \
_op_(LWE2B7_MOT_SEARCH_RANGE_0) \
_op_(LWE2B7_MOTSEARCH_EXIT_0) \
_op_(LWE2B7_MOTSEARCH_BIAS1_0) \
_op_(LWE2B7_MOTSEARCH_BIAS2_0) \
_op_(LWE2B7_MOTSEARCH_BIAS3_0) \
_op_(LWE2B7_MOTSEARCH_BIAS4_0) \
_op_(LWE2B7_MOTSEARCH_BIAS5_0) \
_op_(LWE2B7_MOTSEARCH_BIAS6_0) \
_op_(LWE2B7_INTRA_REF_CTRL_0) \
_op_(LWE2B7_INTRA_REF_AIR_0) \
_op_(LWE2B7_INTRA_REF_AIR_REFRESH_LIMIT_0) \
_op_(LWE2B7_INTRA_REF_MIN_COUNTER_0) \
_op_(LWE2B7_INTRA_REF_DELTA_COUNTER_0) \
_op_(LWE2B7_INTRA_REF_LOAD_CMD_0) \
_op_(LWE2B7_INTRA_REF_LOAD_DATA_0) \
_op_(LWE2B7_INTRA_REF_LOAD_ONE_CMD_0) \
_op_(LWE2B7_INTRA_REF_READ_CMD_0) \
_op_(LWE2B7_INTRA_REF_READ_DATA_0) \
_op_(LWE2B7_NUM_INTRAMB_0) \
_op_(LWE2B7_INTRA_PRED_BIAS_0) \
_op_(LWE2B7_MISC_MODE_BIAS_0) \
_op_(LWE2B7_QUANTIZATION_CONTROL_0) \
_op_(LWE2B7_QPP_CTRL_0) \
_op_(LWE2B7_IPCM_CTRL_0) \
_op_(LWE2B7_PACKET_HEC_0) \
_op_(LWE2B7_PACKET_CTRL_H264_0) \
_op_(LWE2B7_DMA_SWAP_CTRL_0) \
_op_(LWE2B7_CPU_TIMESTAMP_0) \
_op_(LWE2B7_TIMESTAMP_INT_0) \
_op_(LWE2B7_TIMESTAMP_RES_0) \
_op_(LWE2B7_TIMESTAMP_LAST_FRAME_0) \
_op_(LWE2B7_LENGTH_OF_MOTION_MODE_0) \
_op_(LWE2B7_FRAME_VLC_STATUS_0) \
_op_(LWE2B7_I_RATE_CTRL_0) \
_op_(LWE2B7_P_RATE_CTRL_0) \
_op_(LWE2B7_RC_ADJUSTMENT_0) \
_op_(LWE2B7_BUFFER_DEPLETION_RATE_0) \
_op_(LWE2B7_BUFFER_SIZE_0) \
_op_(LWE2B7_INITIAL_DELAY_OFFSET_0) \
_op_(LWE2B7_BUFFER_FULL_0) \
_op_(LWE2B7_REPORTED_FRAME_0) \
_op_(LWE2B7_MIN_IFRAME_SIZE_0) \
_op_(LWE2B7_MIN_PFRAME_SIZE_0) \
_op_(LWE2B7_SUGGESTED_IFRAME_SIZE_0) \
_op_(LWE2B7_SUGGESTED_PFRAME_SIZE_0) \
_op_(LWE2B7_TARGET_BUFFER_I_SIZE_0) \
_op_(LWE2B7_TARGET_BUFFER_P_SIZE_0) \
_op_(LWE2B7_SKIP_THRESHOLD_0) \
_op_(LWE2B7_UNDERFLOW_THRESHOLD_0) \
_op_(LWE2B7_OVERFLOW_THRESHOLD_0) \
_op_(LWE2B7_AVE_BIT_LEN_0) \
_op_(LWE2B7_RC_QP_DQUANT_0) \
_op_(LWE2B7_RC_BU_SIZE_0) \
_op_(LWE2B7_GOP_PARAM_0) \
_op_(LWE2B7_RC_RAM_LOAD_CMD_0) \
_op_(LWE2B7_RC_RAM_LOAD_DATA_0) \
_op_(LWE2B7_LOWER_BOUND_0) \
_op_(LWE2B7_UPPER_BOUND_0) \
_op_(LWE2B7_REMAINING_BITS_0) \
_op_(LWE2B7_NUM_CODED_BU_0) \
_op_(LWE2B7_PREVIOUS_QP_0) \
_op_(LWE2B7_NUM_P_PICTURE_0) \
_op_(LWE2B7_QP_SUM_0) \
_op_(LWE2B7_TOTAL_ENERGY_0) \
_op_(LWE2B7_A1_VALUE_0) \
_op_(LWE2B7_LENGTH_OF_STREAM_0) \
_op_(LWE2B7_BUFFER_FULL_READ_0) \
_op_(LWE2B7_TARGET_BUFFER_LEVEL_0) \
_op_(LWE2B7_DELTA_P_0) \
_op_(LWE2B7_LENGTH_OF_STREAM_CBR_0) \
_op_(LWE2B7_RC_RAM_READ_CMD_0) \
_op_(LWE2B7_RC_RAM_READ_DATA_0) \
_op_(LWE2B7_CODED_FRAMES_0) \
_op_(LWE2B7_P_AVE_HEADER_BITS_A_0) \
_op_(LWE2B7_P_AVE_HEADER_BITS_B_0) \
_op_(LWE2B7_PREV_FRAME_MAD_0) \
_op_(LWE2B7_TOTAL_QP_FOR_P_PICTURE_0) \
_op_(LWE2B7_CONTEXT_SAVE_MISC_0) \
_op_(LWE2B7_MAX_PACKET_0) \
_op_(LWE2B7_FRAME_CYCLE_COUNT_0) \
_op_(LWE2B7_MB_CYCLE_COUNT_0) \
_op_(LWE2B7_MAD_PIC_1_CBR_0) \
_op_(LWE2B7_MX1_CBR_0) \
_op_(LWE2B7_FRAME_BIT_LEN_0) \
_op_(LWE2B7_EB00_START_ADDRESS_0) \
_op_(LWE2B7_EB01_START_ADDRESS_0) \
_op_(LWE2B7_EB02_START_ADDRESS_0) \
_op_(LWE2B7_EB03_START_ADDRESS_0) \
_op_(LWE2B7_EB04_START_ADDRESS_0) \
_op_(LWE2B7_EB05_START_ADDRESS_0) \
_op_(LWE2B7_EB06_START_ADDRESS_0) \
_op_(LWE2B7_EB07_START_ADDRESS_0) \
_op_(LWE2B7_EB08_START_ADDRESS_0) \
_op_(LWE2B7_EB09_START_ADDRESS_0) \
_op_(LWE2B7_EB10_START_ADDRESS_0) \
_op_(LWE2B7_EB11_START_ADDRESS_0) \
_op_(LWE2B7_EB12_START_ADDRESS_0) \
_op_(LWE2B7_EB13_START_ADDRESS_0) \
_op_(LWE2B7_EB14_START_ADDRESS_0) \
_op_(LWE2B7_EB15_START_ADDRESS_0) \
_op_(LWE2B7_EB_SIZE_0) \
_op_(LWE2B7_EB_ACTIVATE_0) \
_op_(LWE2B7_EBM_CONTROL_0) \
_op_(LWE2B7_LAST_EB_BUFFER_STATUS_0) \
_op_(LWE2B7_LAST_EB_FRAME_STATUS_0) \
_op_(LWE2B7_INTERNAL_BIAS_MULTIPLIER_0) \
_op_(LWE2B7_CLK_OVERRIDE_A_0) \
_op_(LWE2B7_MPEA_MCCIF_FIFOCTRL_0) \
_op_(LWE2B7_MPEB_MCCIF_FIFOCTRL_0) \
_op_(LWE2B7_MPEC_MCCIF_FIFOCTRL_0) \
_op_(LWE2B7_TIMEOUT_WCOAL_MPEB_0) \
_op_(LWE2B7_TIMEOUT_WCOAL_MPEC_0) \
_op_(LWE2B7_MCCIF_MPECSRD_HP_0) \
_op_(LWE2B7_MCCIF_MPEUNIFBR_HYST_0) \
_op_(LWE2B7_MCCIF_MPEUNIFBW_HYST_0)

//
// ADDRESS SPACES
//
#define BASE_ADDRESS_MPEA       0x00000000

//
// ARMPE REGISTER BANKS
//
#define MPEA0_FIRST_REG 0x0000 // MPEA_INCR_SYNCPT_0
#define MPEA0_LAST_REG 0x0002 // MPEA_INCR_SYNCPT_ERROR_0
#define MPEA1_FIRST_REG 0x0008 // MPEA_CTXSW_0
#define MPEA1_LAST_REG 0x0008 // MPEA_CTXSW_0
#define MPEA2_FIRST_REG 0x000a // MPEA_CONT_SYNCPT_EBM_EOF_0
#define MPEA2_LAST_REG 0x0014 // MPEA_FORCE_I_FRAME_0
#define MPEA3_FIRST_REG 0x001e // MPEA_INTSTATUS_0
#define MPEA3_LAST_REG 0x0025 // MPEA_CROP_TB_OFFSET_0
#define MPEA4_FIRST_REG 0x0030 // MPEA_PIC_PARAMETERS_0
#define MPEA4_LAST_REG 0x0034 // MPEA_IDR_PIC_ID_0
#define MPEA5_FIRST_REG 0x0040 // MPEA_IB_BUFFER_ADDR_MODE_0
#define MPEA5_LAST_REG 0x004c // MPEA_IB0_BUFFER_STRIDE_CHROMA_0
#define MPEA6_FIRST_REG 0x0050 // MPEA_REF_BUFFER_ADDR_MODE_0
#define MPEA6_LAST_REG 0x0058 // MPEA_REF_RD_MBROW_0
#define MPEA7_FIRST_REG 0x0068 // MPEA_IPRED_ROW_ADDR_0
#define MPEA7_LAST_REG 0x006a // MPEA_ACDC_ADDR_0
#define MPEA8_FIRST_REG 0x0070 // MPEA_FRAME_CTRL_0
#define MPEA8_LAST_REG 0x0076 // MPEA_FRAME_NUM_GOP_0
#define MPEA9_FIRST_REG 0x007f // MPEA_NUM_SLICE_GROUPS_0
#define MPEA9_LAST_REG 0x008f // MPEA_SLICE_MAP_OFFSET_B_0
#define MPEA10_FIRST_REG 0x00a0 // MPEA_MOT_SEARCH_CTRL_0
#define MPEA10_LAST_REG 0x00a8 // MPEA_MOTSEARCH_BIAS6_0
#define MPEA11_FIRST_REG 0x00c0 // MPEA_INTRA_REF_CTRL_0
#define MPEA11_LAST_REG 0x00c7 // MPEA_INTRA_REF_LOAD_ONE_CMD_0
#define MPEA12_FIRST_REG 0x00cd // MPEA_INTRA_REF_READ_CMD_0
#define MPEA12_LAST_REG 0x00d3 // MPEA_QPP_CTRL_0
#define MPEA13_FIRST_REG 0x00d8 // MPEA_IPCM_CTRL_0
#define MPEA13_LAST_REG 0x00d8 // MPEA_IPCM_CTRL_0
#define MPEA14_FIRST_REG 0x00e0 // MPEA_PACKET_HEC_0
#define MPEA14_LAST_REG 0x00e4 // MPEA_TIMESTAMP_INT_0
#define MPEA15_FIRST_REG 0x00e6 // MPEA_TIMESTAMP_RES_0
#define MPEA15_LAST_REG 0x00e9 // MPEA_FRAME_VLC_STATUS_0
#define MPEA16_FIRST_REG 0x0100 // MPEA_I_RATE_CTRL_0
#define MPEA16_LAST_REG 0x0116 // MPEA_RC_RAM_LOAD_DATA_0
#define MPEA17_FIRST_REG 0x011a // MPEA_LOWER_BOUND_0
#define MPEA17_LAST_REG 0x0134 // MPEA_MX1_CBR_0
#define MPEA18_FIRST_REG 0x0160 // MPEA_FRAME_BIT_LEN_0
#define MPEA18_LAST_REG 0x0160 // MPEA_FRAME_BIT_LEN_0
#define MPEA19_FIRST_REG 0x0180 // MPEA_EB00_START_ADDRESS_0
#define MPEA19_LAST_REG 0x018f // MPEA_EB15_START_ADDRESS_0
#define MPEA20_FIRST_REG 0x01a0 // MPEA_EB_SIZE_0
#define MPEA20_LAST_REG 0x01a1 // MPEA_EB_ACTIVATE_0
#define MPEA21_FIRST_REG 0x01a4 // MPEA_EBM_CONTROL_0
#define MPEA21_LAST_REG 0x01a6 // MPEA_LAST_EB_FRAME_STATUS_0
#define MPEA22_FIRST_REG 0x0200 // MPEA_INTERNAL_BIAS_MULTIPLIER_0
#define MPEA22_LAST_REG 0x0201 // MPEA_CLK_OVERRIDE_A_0
#define MPEA23_FIRST_REG 0x03c0 // MPEA_MPEA_MCCIF_FIFOCTRL_0
#define MPEA23_LAST_REG 0x03c0 // MPEA_MPEA_MCCIF_FIFOCTRL_0
#define MPEA24_FIRST_REG 0x03c4 // MPEA_MPEB_MCCIF_FIFOCTRL_0
#define MPEA24_LAST_REG 0x03c4 // MPEA_MPEB_MCCIF_FIFOCTRL_0
#define MPEA25_FIRST_REG 0x03c8 // MPEA_MPEC_MCCIF_FIFOCTRL_0
#define MPEA25_LAST_REG 0x03c8 // MPEA_MPEC_MCCIF_FIFOCTRL_0
#define MPEA26_FIRST_REG 0x03d0 // MPEA_TIMEOUT_WCOAL_MPEB_0
#define MPEA26_LAST_REG 0x03d0 // MPEA_TIMEOUT_WCOAL_MPEB_0
#define MPEA27_FIRST_REG 0x03d4 // MPEA_TIMEOUT_WCOAL_MPEC_0
#define MPEA27_LAST_REG 0x03d4 // MPEA_TIMEOUT_WCOAL_MPEC_0
#define MPEA28_FIRST_REG 0x03fc // MPEA_MCCIF_MPECSRD_HP_0
#define MPEA28_LAST_REG 0x03fc // MPEA_MCCIF_MPECSRD_HP_0
#define MPEA29_FIRST_REG 0x043c // MPEA_MCCIF_MPEUNIFBR_HYST_0
#define MPEA29_LAST_REG 0x043d // MPEA_MCCIF_MPEUNIFBW_HYST_0

#endif // ifndef _cl_e2b7_h_


