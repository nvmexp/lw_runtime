/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
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


#ifndef clc6b0_h_
#define clc6b0_h_

#include "lwtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LWC6B0_VIDEO_DECODER                                                       (0x0000C6B0)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
typedef volatile struct _clc6b0_tag0 {
    LwV32 Reserved00[0x40];
    LwV32 Nop;                                                                  // 0x00000100 - 0x00000103
    LwV32 Reserved01[0xF];
    LwV32 PmTrigger;                                                            // 0x00000140 - 0x00000143
    LwV32 Reserved02[0x2F];
    LwV32 SetApplicationID;                                                     // 0x00000200 - 0x00000203
    LwV32 SetWatchdogTimer;                                                     // 0x00000204 - 0x00000207
    LwV32 Reserved03[0xE];
    LwV32 SemaphoreA;                                                           // 0x00000240 - 0x00000243
    LwV32 SemaphoreB;                                                           // 0x00000244 - 0x00000247
    LwV32 SemaphoreC;                                                           // 0x00000248 - 0x0000024B
    LwV32 Reserved04[0x2D];
    LwV32 Execute;                                                              // 0x00000300 - 0x00000303
    LwV32 SemaphoreD;                                                           // 0x00000304 - 0x00000307
    LwV32 SetPredicationOffsetUpper;                                            // 0x00000308 - 0x0000030B
    LwV32 SetPredicationOffsetLower;                                            // 0x0000030C - 0x0000030F
    LwV32 Reserved05[0x3C];
    LwV32 SetControlParams;                                                     // 0x00000400 - 0x00000403
    LwV32 SetDrvPicSetupOffset;                                                 // 0x00000404 - 0x00000407
    LwV32 SetInBufBaseOffset;                                                   // 0x00000408 - 0x0000040B
    LwV32 SetPictureIndex;                                                      // 0x0000040C - 0x0000040F
    LwV32 SetSliceOffsetsBufOffset;                                             // 0x00000410 - 0x00000413
    LwV32 SetColocDataOffset;                                                   // 0x00000414 - 0x00000417
    LwV32 SetHistoryOffset;                                                     // 0x00000418 - 0x0000041B
    LwV32 SetDisplayBufSize;                                                    // 0x0000041C - 0x0000041F
    LwV32 SetHistogramOffset;                                                   // 0x00000420 - 0x00000423
    LwV32 SetLwdecStatusOffset;                                                 // 0x00000424 - 0x00000427
    LwV32 SetDisplayBufLumaOffset;                                              // 0x00000428 - 0x0000042B
    LwV32 SetDisplayBufChromaOffset;                                            // 0x0000042C - 0x0000042F
    LwV32 SetPictureLumaOffset0;                                                // 0x00000430 - 0x00000433
    LwV32 SetPictureLumaOffset1;                                                // 0x00000434 - 0x00000437
    LwV32 SetPictureLumaOffset2;                                                // 0x00000438 - 0x0000043B
    LwV32 SetPictureLumaOffset3;                                                // 0x0000043C - 0x0000043F
    LwV32 SetPictureLumaOffset4;                                                // 0x00000440 - 0x00000443
    LwV32 SetPictureLumaOffset5;                                                // 0x00000444 - 0x00000447
    LwV32 SetPictureLumaOffset6;                                                // 0x00000448 - 0x0000044B
    LwV32 SetPictureLumaOffset7;                                                // 0x0000044C - 0x0000044F
    LwV32 SetPictureLumaOffset8;                                                // 0x00000450 - 0x00000453
    LwV32 SetPictureLumaOffset9;                                                // 0x00000454 - 0x00000457
    LwV32 SetPictureLumaOffset10;                                               // 0x00000458 - 0x0000045B
    LwV32 SetPictureLumaOffset11;                                               // 0x0000045C - 0x0000045F
    LwV32 SetPictureLumaOffset12;                                               // 0x00000460 - 0x00000463
    LwV32 SetPictureLumaOffset13;                                               // 0x00000464 - 0x00000467
    LwV32 SetPictureLumaOffset14;                                               // 0x00000468 - 0x0000046B
    LwV32 SetPictureLumaOffset15;                                               // 0x0000046C - 0x0000046F
    LwV32 SetPictureLumaOffset16;                                               // 0x00000470 - 0x00000473
    LwV32 SetPictureChromaOffset0;                                              // 0x00000474 - 0x00000477
    LwV32 SetPictureChromaOffset1;                                              // 0x00000478 - 0x0000047B
    LwV32 SetPictureChromaOffset2;                                              // 0x0000047C - 0x0000047F
    LwV32 SetPictureChromaOffset3;                                              // 0x00000480 - 0x00000483
    LwV32 SetPictureChromaOffset4;                                              // 0x00000484 - 0x00000487
    LwV32 SetPictureChromaOffset5;                                              // 0x00000488 - 0x0000048B
    LwV32 SetPictureChromaOffset6;                                              // 0x0000048C - 0x0000048F
    LwV32 SetPictureChromaOffset7;                                              // 0x00000490 - 0x00000493
    LwV32 SetPictureChromaOffset8;                                              // 0x00000494 - 0x00000497
    LwV32 SetPictureChromaOffset9;                                              // 0x00000498 - 0x0000049B
    LwV32 SetPictureChromaOffset10;                                             // 0x0000049C - 0x0000049F
    LwV32 SetPictureChromaOffset11;                                             // 0x000004A0 - 0x000004A3
    LwV32 SetPictureChromaOffset12;                                             // 0x000004A4 - 0x000004A7
    LwV32 SetPictureChromaOffset13;                                             // 0x000004A8 - 0x000004AB
    LwV32 SetPictureChromaOffset14;                                             // 0x000004AC - 0x000004AF
    LwV32 SetPictureChromaOffset15;                                             // 0x000004B0 - 0x000004B3
    LwV32 SetPictureChromaOffset16;                                             // 0x000004B4 - 0x000004B7
    LwV32 SetPicScratchBufOffset;                                               // 0x000004B8 - 0x000004BB
    LwV32 SetExternalMVBufferOffset;                                            // 0x000004BC - 0x000004BF
    LwV32 Reserved06[0x10];
    LwV32 H264SetMBHistBufOffset;                                               // 0x00000500 - 0x00000503
    LwV32 Reserved07[0xF];
    LwV32 VP8SetProbDataOffset;                                                 // 0x00000540 - 0x00000543
    LwV32 VP8SetHeaderPartitionBufBaseOffset;                                   // 0x00000544 - 0x00000547
    LwV32 Reserved08[0xE];
    LwV32 HevcSetScalingListOffset;                                             // 0x00000580 - 0x00000583
    LwV32 HevcSetTileSizesOffset;                                               // 0x00000584 - 0x00000587
    LwV32 HevcSetFilterBufferOffset;                                            // 0x00000588 - 0x0000058B
    LwV32 HevcSetSaoBufferOffset;                                               // 0x0000058C - 0x0000058F
    LwV32 HevcSetSliceInfoBufferOffset;                                         // 0x00000590 - 0x00000593
    LwV32 HevcSetSliceGroupIndex;                                               // 0x00000594 - 0x00000597
    LwV32 Reserved09[0xA];
    LwV32 VP9SetProbTabBufOffset;                                               // 0x000005C0 - 0x000005C3
    LwV32 VP9SetCtxCounterBufOffset;                                            // 0x000005C4 - 0x000005C7
    LwV32 VP9SetSegmentReadBufOffset;                                           // 0x000005C8 - 0x000005CB
    LwV32 VP9SetSegmentWriteBufOffset;                                          // 0x000005CC - 0x000005CF
    LwV32 VP9SetTileSizeBufOffset;                                              // 0x000005D0 - 0x000005D3
    LwV32 VP9SetColMVWriteBufOffset;                                            // 0x000005D4 - 0x000005D7
    LwV32 VP9SetColMVReadBufOffset;                                             // 0x000005D8 - 0x000005DB
    LwV32 VP9SetFilterBufferOffset;                                             // 0x000005DC - 0x000005DF
    LwV32 VP9ParserSetPicSetupOffset;                                           // 0x000005E0 - 0x000005E3
    LwV32 VP9ParserSetPrevPicSetupOffset;                                       // 0x000005E4 - 0x000005E7
    LwV32 VP9ParserSetProbTabBufOffset;                                         // 0x000005E8 - 0x000005EB
    LwV32 Reserved10[0x185];
    LwV32 Reserved11[0x40];
    LwV32 Reserved12[0x40];
    LwV32 Reserved13[0x40];
    LwV32 Reserved14[0x85];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved15[0x3BA];
} LWC6B0_VIDEO_DECODERControlPio;

#define LWC6B0_NOP                                                              (0x00000100)
#define LWC6B0_NOP_PARAMETER                                                    31:0
#define LWC6B0_PM_TRIGGER                                                       (0x00000140)
#define LWC6B0_PM_TRIGGER_V                                                     31:0
#define LWC6B0_SET_APPLICATION_ID                                               (0x00000200)
#define LWC6B0_SET_APPLICATION_ID_ID                                            31:0
#define LWC6B0_SET_APPLICATION_ID_ID_MPEG12                                     (0x00000001)
#define LWC6B0_SET_APPLICATION_ID_ID_VC1                                        (0x00000002)
#define LWC6B0_SET_APPLICATION_ID_ID_H264                                       (0x00000003)
#define LWC6B0_SET_APPLICATION_ID_ID_MPEG4                                      (0x00000004)
#define LWC6B0_SET_APPLICATION_ID_ID_VP8                                        (0x00000005)
#define LWC6B0_SET_APPLICATION_ID_ID_HEVC                                       (0x00000007)
#define LWC6B0_SET_APPLICATION_ID_ID_VP9                                        (0x00000009)
#define LWC6B0_SET_APPLICATION_ID_ID_HEVC_PARSER                                (0x0000000C)
#define LWC6B0_SET_APPLICATION_ID_ID_VP9_WITH_PARSER                            (0x00000011)
#define LWC6B0_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LWC6B0_SET_WATCHDOG_TIMER_TIMER                                         31:0
#define LWC6B0_SEMAPHORE_A                                                      (0x00000240)
#define LWC6B0_SEMAPHORE_A_UPPER                                                7:0
#define LWC6B0_SEMAPHORE_B                                                      (0x00000244)
#define LWC6B0_SEMAPHORE_B_LOWER                                                31:0
#define LWC6B0_SEMAPHORE_C                                                      (0x00000248)
#define LWC6B0_SEMAPHORE_C_PAYLOAD                                              31:0
#define LWC6B0_EXELWTE                                                          (0x00000300)
#define LWC6B0_EXELWTE_NOTIFY                                                   0:0
#define LWC6B0_EXELWTE_NOTIFY_DISABLE                                           (0x00000000)
#define LWC6B0_EXELWTE_NOTIFY_ENABLE                                            (0x00000001)
#define LWC6B0_EXELWTE_NOTIFY_ON                                                1:1
#define LWC6B0_EXELWTE_NOTIFY_ON_END                                            (0x00000000)
#define LWC6B0_EXELWTE_NOTIFY_ON_BEGIN                                          (0x00000001)
#define LWC6B0_EXELWTE_PREDICATION                                              2:2
#define LWC6B0_EXELWTE_PREDICATION_DISABLE                                      (0x00000000)
#define LWC6B0_EXELWTE_PREDICATION_ENABLE                                       (0x00000001)
#define LWC6B0_EXELWTE_PREDICATION_OP                                           3:3
#define LWC6B0_EXELWTE_PREDICATION_OP_EQUAL_ZERO                                (0x00000000)
#define LWC6B0_EXELWTE_PREDICATION_OP_NOT_EQUAL_ZERO                            (0x00000001)
#define LWC6B0_EXELWTE_AWAKEN                                                   8:8
#define LWC6B0_EXELWTE_AWAKEN_DISABLE                                           (0x00000000)
#define LWC6B0_EXELWTE_AWAKEN_ENABLE                                            (0x00000001)
#define LWC6B0_SEMAPHORE_D                                                      (0x00000304)
#define LWC6B0_SEMAPHORE_D_STRUCTURE_SIZE                                       0:0
#define LWC6B0_SEMAPHORE_D_STRUCTURE_SIZE_ONE                                   (0x00000000)
#define LWC6B0_SEMAPHORE_D_STRUCTURE_SIZE_FOUR                                  (0x00000001)
#define LWC6B0_SEMAPHORE_D_AWAKEN_ENABLE                                        8:8
#define LWC6B0_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                  (0x00000000)
#define LWC6B0_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                   (0x00000001)
#define LWC6B0_SEMAPHORE_D_OPERATION                                            17:16
#define LWC6B0_SEMAPHORE_D_OPERATION_RELEASE                                    (0x00000000)
#define LWC6B0_SEMAPHORE_D_OPERATION_RESERVED0                                  (0x00000001)
#define LWC6B0_SEMAPHORE_D_OPERATION_RESERVED1                                  (0x00000002)
#define LWC6B0_SEMAPHORE_D_OPERATION_TRAP                                       (0x00000003)
#define LWC6B0_SEMAPHORE_D_FLUSH_DISABLE                                        21:21
#define LWC6B0_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                  (0x00000000)
#define LWC6B0_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                   (0x00000001)
#define LWC6B0_SEMAPHORE_D_CONDITIONAL_TRAP                                     22:22
#define LWC6B0_SEMAPHORE_D_CONDITIONAL_TRAP_FALSE                               (0x00000000)
#define LWC6B0_SEMAPHORE_D_CONDITIONAL_TRAP_TRUE                                (0x00000001)
#define LWC6B0_SET_PREDICATION_OFFSET_UPPER                                     (0x00000308)
#define LWC6B0_SET_PREDICATION_OFFSET_UPPER_OFFSET                              7:0
#define LWC6B0_SET_PREDICATION_OFFSET_LOWER                                     (0x0000030C)
#define LWC6B0_SET_PREDICATION_OFFSET_LOWER_OFFSET                              31:0
#define LWC6B0_SET_CONTROL_PARAMS                                               (0x00000400)
#define LWC6B0_SET_CONTROL_PARAMS_CODEC_TYPE                                    3:0
#define LWC6B0_SET_CONTROL_PARAMS_CODEC_TYPE_MPEG1                              (0x00000000)
#define LWC6B0_SET_CONTROL_PARAMS_CODEC_TYPE_MPEG2                              (0x00000001)
#define LWC6B0_SET_CONTROL_PARAMS_CODEC_TYPE_VC1                                (0x00000002)
#define LWC6B0_SET_CONTROL_PARAMS_CODEC_TYPE_H264                               (0x00000003)
#define LWC6B0_SET_CONTROL_PARAMS_CODEC_TYPE_MPEG4                              (0x00000004)
#define LWC6B0_SET_CONTROL_PARAMS_CODEC_TYPE_DIVX3                              (0x00000004)
#define LWC6B0_SET_CONTROL_PARAMS_CODEC_TYPE_VP8                                (0x00000005)
#define LWC6B0_SET_CONTROL_PARAMS_CODEC_TYPE_HEVC                               (0x00000007)
#define LWC6B0_SET_CONTROL_PARAMS_CODEC_TYPE_VP9                                (0x00000009)
#define LWC6B0_SET_CONTROL_PARAMS_GPTIMER_ON                                    4:4
#define LWC6B0_SET_CONTROL_PARAMS_RET_ERROR                                     5:5
#define LWC6B0_SET_CONTROL_PARAMS_ERR_CONCEAL_ON                                6:6
#define LWC6B0_SET_CONTROL_PARAMS_ERROR_FRM_IDX                                 12:7
#define LWC6B0_SET_CONTROL_PARAMS_MBTIMER_ON                                    13:13
#define LWC6B0_SET_CONTROL_PARAMS_EC_INTRA_FRAME_USING_PSLC                     14:14
#define LWC6B0_SET_CONTROL_PARAMS_IGNORE_SOME_FIELDS_CRC_CHECK                  15:15
#define LWC6B0_SET_CONTROL_PARAMS_EVENT_TRACE_LOGGING_ON                        16:16
#define LWC6B0_SET_CONTROL_PARAMS_ALL_INTRA_FRAME                               17:17
#define LWC6B0_SET_CONTROL_PARAMS_RESERVED                                      31:18
#define LWC6B0_SET_DRV_PIC_SETUP_OFFSET                                         (0x00000404)
#define LWC6B0_SET_DRV_PIC_SETUP_OFFSET_OFFSET                                  31:0
#define LWC6B0_SET_IN_BUF_BASE_OFFSET                                           (0x00000408)
#define LWC6B0_SET_IN_BUF_BASE_OFFSET_OFFSET                                    31:0
#define LWC6B0_SET_PICTURE_INDEX                                                (0x0000040C)
#define LWC6B0_SET_PICTURE_INDEX_INDEX                                          31:0
#define LWC6B0_SET_SLICE_OFFSETS_BUF_OFFSET                                     (0x00000410)
#define LWC6B0_SET_SLICE_OFFSETS_BUF_OFFSET_OFFSET                              31:0
#define LWC6B0_SET_COLOC_DATA_OFFSET                                            (0x00000414)
#define LWC6B0_SET_COLOC_DATA_OFFSET_OFFSET                                     31:0
#define LWC6B0_SET_HISTORY_OFFSET                                               (0x00000418)
#define LWC6B0_SET_HISTORY_OFFSET_OFFSET                                        31:0
#define LWC6B0_SET_DISPLAY_BUF_SIZE                                             (0x0000041C)
#define LWC6B0_SET_DISPLAY_BUF_SIZE_SIZE                                        31:0
#define LWC6B0_SET_HISTOGRAM_OFFSET                                             (0x00000420)
#define LWC6B0_SET_HISTOGRAM_OFFSET_OFFSET                                      31:0
#define LWC6B0_SET_LWDEC_STATUS_OFFSET                                          (0x00000424)
#define LWC6B0_SET_LWDEC_STATUS_OFFSET_OFFSET                                   31:0
#define LWC6B0_SET_DISPLAY_BUF_LUMA_OFFSET                                      (0x00000428)
#define LWC6B0_SET_DISPLAY_BUF_LUMA_OFFSET_OFFSET                               31:0
#define LWC6B0_SET_DISPLAY_BUF_CHROMA_OFFSET                                    (0x0000042C)
#define LWC6B0_SET_DISPLAY_BUF_CHROMA_OFFSET_OFFSET                             31:0
#define LWC6B0_SET_PICTURE_LUMA_OFFSET0                                         (0x00000430)
#define LWC6B0_SET_PICTURE_LUMA_OFFSET0_OFFSET                                  31:0
#define LWC6B0_SET_PICTURE_LUMA_OFFSET1                                         (0x00000434)
#define LWC6B0_SET_PICTURE_LUMA_OFFSET1_OFFSET                                  31:0
#define LWC6B0_SET_PICTURE_LUMA_OFFSET2                                         (0x00000438)
#define LWC6B0_SET_PICTURE_LUMA_OFFSET2_OFFSET                                  31:0
#define LWC6B0_SET_PICTURE_LUMA_OFFSET3                                         (0x0000043C)
#define LWC6B0_SET_PICTURE_LUMA_OFFSET3_OFFSET                                  31:0
#define LWC6B0_SET_PICTURE_LUMA_OFFSET4                                         (0x00000440)
#define LWC6B0_SET_PICTURE_LUMA_OFFSET4_OFFSET                                  31:0
#define LWC6B0_SET_PICTURE_LUMA_OFFSET5                                         (0x00000444)
#define LWC6B0_SET_PICTURE_LUMA_OFFSET5_OFFSET                                  31:0
#define LWC6B0_SET_PICTURE_LUMA_OFFSET6                                         (0x00000448)
#define LWC6B0_SET_PICTURE_LUMA_OFFSET6_OFFSET                                  31:0
#define LWC6B0_SET_PICTURE_LUMA_OFFSET7                                         (0x0000044C)
#define LWC6B0_SET_PICTURE_LUMA_OFFSET7_OFFSET                                  31:0
#define LWC6B0_SET_PICTURE_LUMA_OFFSET8                                         (0x00000450)
#define LWC6B0_SET_PICTURE_LUMA_OFFSET8_OFFSET                                  31:0
#define LWC6B0_SET_PICTURE_LUMA_OFFSET9                                         (0x00000454)
#define LWC6B0_SET_PICTURE_LUMA_OFFSET9_OFFSET                                  31:0
#define LWC6B0_SET_PICTURE_LUMA_OFFSET10                                        (0x00000458)
#define LWC6B0_SET_PICTURE_LUMA_OFFSET10_OFFSET                                 31:0
#define LWC6B0_SET_PICTURE_LUMA_OFFSET11                                        (0x0000045C)
#define LWC6B0_SET_PICTURE_LUMA_OFFSET11_OFFSET                                 31:0
#define LWC6B0_SET_PICTURE_LUMA_OFFSET12                                        (0x00000460)
#define LWC6B0_SET_PICTURE_LUMA_OFFSET12_OFFSET                                 31:0
#define LWC6B0_SET_PICTURE_LUMA_OFFSET13                                        (0x00000464)
#define LWC6B0_SET_PICTURE_LUMA_OFFSET13_OFFSET                                 31:0
#define LWC6B0_SET_PICTURE_LUMA_OFFSET14                                        (0x00000468)
#define LWC6B0_SET_PICTURE_LUMA_OFFSET14_OFFSET                                 31:0
#define LWC6B0_SET_PICTURE_LUMA_OFFSET15                                        (0x0000046C)
#define LWC6B0_SET_PICTURE_LUMA_OFFSET15_OFFSET                                 31:0
#define LWC6B0_SET_PICTURE_LUMA_OFFSET16                                        (0x00000470)
#define LWC6B0_SET_PICTURE_LUMA_OFFSET16_OFFSET                                 31:0
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET0                                       (0x00000474)
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET0_OFFSET                                31:0
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET1                                       (0x00000478)
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET1_OFFSET                                31:0
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET2                                       (0x0000047C)
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET2_OFFSET                                31:0
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET3                                       (0x00000480)
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET3_OFFSET                                31:0
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET4                                       (0x00000484)
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET4_OFFSET                                31:0
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET5                                       (0x00000488)
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET5_OFFSET                                31:0
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET6                                       (0x0000048C)
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET6_OFFSET                                31:0
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET7                                       (0x00000490)
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET7_OFFSET                                31:0
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET8                                       (0x00000494)
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET8_OFFSET                                31:0
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET9                                       (0x00000498)
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET9_OFFSET                                31:0
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET10                                      (0x0000049C)
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET10_OFFSET                               31:0
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET11                                      (0x000004A0)
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET11_OFFSET                               31:0
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET12                                      (0x000004A4)
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET12_OFFSET                               31:0
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET13                                      (0x000004A8)
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET13_OFFSET                               31:0
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET14                                      (0x000004AC)
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET14_OFFSET                               31:0
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET15                                      (0x000004B0)
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET15_OFFSET                               31:0
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET16                                      (0x000004B4)
#define LWC6B0_SET_PICTURE_CHROMA_OFFSET16_OFFSET                               31:0
#define LWC6B0_SET_PIC_SCRATCH_BUF_OFFSET                                       (0x000004B8)
#define LWC6B0_SET_PIC_SCRATCH_BUF_OFFSET_OFFSET                                31:0
#define LWC6B0_SET_EXTERNAL_MVBUFFER_OFFSET                                     (0x000004BC)
#define LWC6B0_SET_EXTERNAL_MVBUFFER_OFFSET_OFFSET                              31:0
#define LWC6B0_H264_SET_MBHIST_BUF_OFFSET                                       (0x00000500)
#define LWC6B0_H264_SET_MBHIST_BUF_OFFSET_OFFSET                                31:0
#define LWC6B0_VP8_SET_PROB_DATA_OFFSET                                         (0x00000540)
#define LWC6B0_VP8_SET_PROB_DATA_OFFSET_OFFSET                                  31:0
#define LWC6B0_VP8_SET_HEADER_PARTITION_BUF_BASE_OFFSET                         (0x00000544)
#define LWC6B0_VP8_SET_HEADER_PARTITION_BUF_BASE_OFFSET_OFFSET                  31:0
#define LWC6B0_HEVC_SET_SCALING_LIST_OFFSET                                     (0x00000580)
#define LWC6B0_HEVC_SET_SCALING_LIST_OFFSET_OFFSET                              31:0
#define LWC6B0_HEVC_SET_TILE_SIZES_OFFSET                                       (0x00000584)
#define LWC6B0_HEVC_SET_TILE_SIZES_OFFSET_OFFSET                                31:0
#define LWC6B0_HEVC_SET_FILTER_BUFFER_OFFSET                                    (0x00000588)
#define LWC6B0_HEVC_SET_FILTER_BUFFER_OFFSET_OFFSET                             31:0
#define LWC6B0_HEVC_SET_SAO_BUFFER_OFFSET                                       (0x0000058C)
#define LWC6B0_HEVC_SET_SAO_BUFFER_OFFSET_OFFSET                                31:0
#define LWC6B0_HEVC_SET_SLICE_INFO_BUFFER_OFFSET                                (0x00000590)
#define LWC6B0_HEVC_SET_SLICE_INFO_BUFFER_OFFSET_OFFSET                         31:0
#define LWC6B0_HEVC_SET_SLICE_GROUP_INDEX                                       (0x00000594)
#define LWC6B0_HEVC_SET_SLICE_GROUP_INDEX_OFFSET                                31:0
#define LWC6B0_VP9_SET_PROB_TAB_BUF_OFFSET                                      (0x000005C0)
#define LWC6B0_VP9_SET_PROB_TAB_BUF_OFFSET_OFFSET                               31:0
#define LWC6B0_VP9_SET_CTX_COUNTER_BUF_OFFSET                                   (0x000005C4)
#define LWC6B0_VP9_SET_CTX_COUNTER_BUF_OFFSET_OFFSET                            31:0
#define LWC6B0_VP9_SET_SEGMENT_READ_BUF_OFFSET                                  (0x000005C8)
#define LWC6B0_VP9_SET_SEGMENT_READ_BUF_OFFSET_OFFSET                           31:0
#define LWC6B0_VP9_SET_SEGMENT_WRITE_BUF_OFFSET                                 (0x000005CC)
#define LWC6B0_VP9_SET_SEGMENT_WRITE_BUF_OFFSET_OFFSET                          31:0
#define LWC6B0_VP9_SET_TILE_SIZE_BUF_OFFSET                                     (0x000005D0)
#define LWC6B0_VP9_SET_TILE_SIZE_BUF_OFFSET_OFFSET                              31:0
#define LWC6B0_VP9_SET_COL_MVWRITE_BUF_OFFSET                                   (0x000005D4)
#define LWC6B0_VP9_SET_COL_MVWRITE_BUF_OFFSET_OFFSET                            31:0
#define LWC6B0_VP9_SET_COL_MVREAD_BUF_OFFSET                                    (0x000005D8)
#define LWC6B0_VP9_SET_COL_MVREAD_BUF_OFFSET_OFFSET                             31:0
#define LWC6B0_VP9_SET_FILTER_BUFFER_OFFSET                                     (0x000005DC)
#define LWC6B0_VP9_SET_FILTER_BUFFER_OFFSET_OFFSET                              31:0
#define LWC6B0_VP9_PARSER_SET_PIC_SETUP_OFFSET                                  (0x000005E0)
#define LWC6B0_VP9_PARSER_SET_PIC_SETUP_OFFSET_OFFSET                           31:0
#define LWC6B0_VP9_PARSER_SET_PREV_PIC_SETUP_OFFSET                             (0x000005E4)
#define LWC6B0_VP9_PARSER_SET_PREV_PIC_SETUP_OFFSET_OFFSET                      31:0
#define LWC6B0_VP9_PARSER_SET_PROB_TAB_BUF_OFFSET                               (0x000005E8)
#define LWC6B0_VP9_PARSER_SET_PROB_TAB_BUF_OFFSET_OFFSET                        31:0
#define LWC6B0_PM_TRIGGER_END                                                   (0x00001114)
#define LWC6B0_PM_TRIGGER_END_V                                                 31:0

#define LWC6B0_ERROR_NONE                                                       (0x00000000)
#define LWC6B0_OS_ERROR_EXELWTE_INSUFFICIENT_DATA                               (0x00000001)
#define LWC6B0_OS_ERROR_SEMAPHORE_INSUFFICIENT_DATA                             (0x00000002)
#define LWC6B0_OS_ERROR_ILWALID_METHOD                                          (0x00000003)
#define LWC6B0_OS_ERROR_ILWALID_DMA_PAGE                                        (0x00000004)
#define LWC6B0_OS_ERROR_UNHANDLED_INTERRUPT                                     (0x00000005)
#define LWC6B0_OS_ERROR_EXCEPTION                                               (0x00000006)
#define LWC6B0_OS_ERROR_ILWALID_CTXSW_REQUEST                                   (0x00000007)
#define LWC6B0_OS_ERROR_APPLICATION                                             (0x00000008)
#define LWC6B0_OS_ERROR_SW_BREAKPT                                              (0x00000009)
#define LWC6B0_OS_INTERRUPT_EXELWTE_AWAKEN                                      (0x00000100)
#define LWC6B0_OS_INTERRUPT_BACKEND_SEMAPHORE_AWAKEN                            (0x00000200)
#define LWC6B0_OS_INTERRUPT_CTX_ERROR_FBIF                                      (0x00000300)
#define LWC6B0_OS_INTERRUPT_LIMIT_VIOLATION                                     (0x00000400)
#define LWC6B0_OS_INTERRUPT_LIMIT_AND_FBIF_CTX_ERROR                            (0x00000500)
#define LWC6B0_OS_INTERRUPT_HALT_ENGINE                                         (0x00000600)
#define LWC6B0_OS_INTERRUPT_TRAP_NONSTALL                                       (0x00000700)
#define LWC6B0_H264_VLD_ERR_SEQ_DATA_INCONSISTENT                               (0x00004001)
#define LWC6B0_H264_VLD_ERR_PIC_DATA_INCONSISTENT                               (0x00004002)
#define LWC6B0_H264_VLD_ERR_SLC_DATA_BUF_ADDR_OUT_OF_BOUNDS                     (0x00004100)
#define LWC6B0_H264_VLD_ERR_BITSTREAM_ERROR                                     (0x00004101)
#define LWC6B0_H264_VLD_ERR_CTX_DMA_ID_CTRL_IN_ILWALID                          (0x000041F8)
#define LWC6B0_H264_VLD_ERR_SLC_HDR_OUT_SIZE_NOT_MULT256                        (0x00004200)
#define LWC6B0_H264_VLD_ERR_SLC_DATA_OUT_SIZE_NOT_MULT256                       (0x00004201)
#define LWC6B0_H264_VLD_ERR_CTX_DMA_ID_FLOW_CTRL_ILWALID                        (0x00004203)
#define LWC6B0_H264_VLD_ERR_CTX_DMA_ID_SLC_HDR_OUT_ILWALID                      (0x00004204)
#define LWC6B0_H264_VLD_ERR_SLC_HDR_OUT_BUF_TOO_SMALL                           (0x00004205)
#define LWC6B0_H264_VLD_ERR_SLC_HDR_OUT_BUF_ALREADY_VALID                       (0x00004206)
#define LWC6B0_H264_VLD_ERR_SLC_DATA_OUT_BUF_TOO_SMALL                          (0x00004207)
#define LWC6B0_H264_VLD_ERR_DATA_BUF_CNT_TOO_SMALL                              (0x00004208)
#define LWC6B0_H264_VLD_ERR_BITSTREAM_EMPTY                                     (0x00004209)
#define LWC6B0_H264_VLD_ERR_FRAME_WIDTH_TOO_LARGE                               (0x0000420A)
#define LWC6B0_H264_VLD_ERR_FRAME_HEIGHT_TOO_LARGE                              (0x0000420B)
#define LWC6B0_H264_VLD_ERR_HIST_BUF_TOO_SMALL                                  (0x00004300)
#define LWC6B0_VC1_VLD_ERR_PIC_DATA_BUF_ADDR_OUT_OF_BOUND                       (0x00005100)
#define LWC6B0_VC1_VLD_ERR_BITSTREAM_ERROR                                      (0x00005101)
#define LWC6B0_VC1_VLD_ERR_PIC_HDR_OUT_SIZE_NOT_MULT256                         (0x00005200)
#define LWC6B0_VC1_VLD_ERR_PIC_DATA_OUT_SIZE_NOT_MULT256                        (0x00005201)
#define LWC6B0_VC1_VLD_ERR_CTX_DMA_ID_CTRL_IN_ILWALID                           (0x00005202)
#define LWC6B0_VC1_VLD_ERR_CTX_DMA_ID_FLOW_CTRL_ILWALID                         (0x00005203)
#define LWC6B0_VC1_VLD_ERR_CTX_DMA_ID_PIC_HDR_OUT_ILWALID                       (0x00005204)
#define LWC6B0_VC1_VLD_ERR_SLC_HDR_OUT_BUF_TOO_SMALL                            (0x00005205)
#define LWC6B0_VC1_VLD_ERR_PIC_HDR_OUT_BUF_ALREADY_VALID                        (0x00005206)
#define LWC6B0_VC1_VLD_ERR_PIC_DATA_OUT_BUF_TOO_SMALL                           (0x00005207)
#define LWC6B0_VC1_VLD_ERR_DATA_INFO_IN_BUF_TOO_SMALL                           (0x00005208)
#define LWC6B0_VC1_VLD_ERR_BITSTREAM_EMPTY                                      (0x00005209)
#define LWC6B0_VC1_VLD_ERR_FRAME_WIDTH_TOO_LARGE                                (0x0000520A)
#define LWC6B0_VC1_VLD_ERR_FRAME_HEIGHT_TOO_LARGE                               (0x0000520B)
#define LWC6B0_VC1_VLD_ERR_PIC_DATA_OUT_BUF_FULL_TIME_OUT                       (0x00005300)
#define LWC6B0_MPEG12_VLD_ERR_SLC_DATA_BUF_ADDR_OUT_OF_BOUNDS                   (0x00006100)
#define LWC6B0_MPEG12_VLD_ERR_BITSTREAM_ERROR                                   (0x00006101)
#define LWC6B0_MPEG12_VLD_ERR_SLC_DATA_OUT_SIZE_NOT_MULT256                     (0x00006200)
#define LWC6B0_MPEG12_VLD_ERR_CTX_DMA_ID_CTRL_IN_ILWALID                        (0x00006201)
#define LWC6B0_MPEG12_VLD_ERR_CTX_DMA_ID_FLOW_CTRL_ILWALID                      (0x00006202)
#define LWC6B0_MPEG12_VLD_ERR_SLC_DATA_OUT_BUF_TOO_SMALL                        (0x00006203)
#define LWC6B0_MPEG12_VLD_ERR_DATA_INFO_IN_BUF_TOO_SMALL                        (0x00006204)
#define LWC6B0_MPEG12_VLD_ERR_BITSTREAM_EMPTY                                   (0x00006205)
#define LWC6B0_MPEG12_VLD_ERR_ILWALID_PIC_STRUCTURE                             (0x00006206)
#define LWC6B0_MPEG12_VLD_ERR_ILWALID_PIC_CODING_TYPE                           (0x00006207)
#define LWC6B0_MPEG12_VLD_ERR_FRAME_WIDTH_TOO_LARGE                             (0x00006208)
#define LWC6B0_MPEG12_VLD_ERR_FRAME_HEIGHT_TOO_LARGE                            (0x00006209)
#define LWC6B0_MPEG12_VLD_ERR_SLC_DATA_OUT_BUF_FULL_TIME_OUT                    (0x00006300)
#define LWC6B0_CMN_VLD_ERR_PDEC_RETURNED_ERROR                                  (0x00007101)
#define LWC6B0_CMN_VLD_ERR_EDOB_FLUSH_TIME_OUT                                  (0x00007102)
#define LWC6B0_CMN_VLD_ERR_EDOB_REWIND_TIME_OUT                                 (0x00007103)
#define LWC6B0_CMN_VLD_ERR_VLD_WD_TIME_OUT                                      (0x00007104)
#define LWC6B0_CMN_VLD_ERR_NUM_SLICES_ZERO                                      (0x00007105)
#define LWC6B0_MPEG4_VLD_ERR_PIC_DATA_BUF_ADDR_OUT_OF_BOUND                     (0x00008100)
#define LWC6B0_MPEG4_VLD_ERR_BITSTREAM_ERROR                                    (0x00008101)
#define LWC6B0_MPEG4_VLD_ERR_PIC_HDR_OUT_SIZE_NOT_MULT256                       (0x00008200)
#define LWC6B0_MPEG4_VLD_ERR_PIC_DATA_OUT_SIZE_NOT_MULT256                      (0x00008201)
#define LWC6B0_MPEG4_VLD_ERR_CTX_DMA_ID_CTRL_IN_ILWALID                         (0x00008202)
#define LWC6B0_MPEG4_VLD_ERR_CTX_DMA_ID_FLOW_CTRL_ILWALID                       (0x00008203)
#define LWC6B0_MPEG4_VLD_ERR_CTX_DMA_ID_PIC_HDR_OUT_ILWALID                     (0x00008204)
#define LWC6B0_MPEG4_VLD_ERR_SLC_HDR_OUT_BUF_TOO_SMALL                          (0x00008205)
#define LWC6B0_MPEG4_VLD_ERR_PIC_HDR_OUT_BUF_ALREADY_VALID                      (0x00008206)
#define LWC6B0_MPEG4_VLD_ERR_PIC_DATA_OUT_BUF_TOO_SMALL                         (0x00008207)
#define LWC6B0_MPEG4_VLD_ERR_DATA_INFO_IN_BUF_TOO_SMALL                         (0x00008208)
#define LWC6B0_MPEG4_VLD_ERR_BITSTREAM_EMPTY                                    (0x00008209)
#define LWC6B0_MPEG4_VLD_ERR_FRAME_WIDTH_TOO_LARGE                              (0x0000820A)
#define LWC6B0_MPEG4_VLD_ERR_FRAME_HEIGHT_TOO_LARGE                             (0x0000820B)
#define LWC6B0_MPEG4_VLD_ERR_PIC_DATA_OUT_BUF_FULL_TIME_OUT                     (0x00051E01)
#define LWC6B0_DEC_ERROR_MPEG12_APPTIMER_EXPIRED                                (0xDEC10001)
#define LWC6B0_DEC_ERROR_MPEG12_MVTIMER_EXPIRED                                 (0xDEC10002)
#define LWC6B0_DEC_ERROR_MPEG12_ILWALID_TOKEN                                   (0xDEC10003)
#define LWC6B0_DEC_ERROR_MPEG12_SLICEDATA_MISSING                               (0xDEC10004)
#define LWC6B0_DEC_ERROR_MPEG12_HWERR_INTERRUPT                                 (0xDEC10005)
#define LWC6B0_DEC_ERROR_MPEG12_DETECTED_VLD_FAILURE                            (0xDEC10006)
#define LWC6B0_DEC_ERROR_MPEG12_PICTURE_INIT                                    (0xDEC10100)
#define LWC6B0_DEC_ERROR_MPEG12_STATEMACHINE_FAILURE                            (0xDEC10101)
#define LWC6B0_DEC_ERROR_MPEG12_ILWALID_CTXID_PIC                               (0xDEC10901)
#define LWC6B0_DEC_ERROR_MPEG12_ILWALID_CTXID_UCODE                             (0xDEC10902)
#define LWC6B0_DEC_ERROR_MPEG12_ILWALID_CTXID_FC                                (0xDEC10903)
#define LWC6B0_DEC_ERROR_MPEG12_ILWALID_CTXID_SLH                               (0xDEC10904)
#define LWC6B0_DEC_ERROR_MPEG12_ILWALID_UCODE_SIZE                              (0xDEC10905)
#define LWC6B0_DEC_ERROR_MPEG12_ILWALID_SLICE_COUNT                             (0xDEC10906)
#define LWC6B0_DEC_ERROR_VC1_APPTIMER_EXPIRED                                   (0xDEC20001)
#define LWC6B0_DEC_ERROR_VC1_MVTIMER_EXPIRED                                    (0xDEC20002)
#define LWC6B0_DEC_ERROR_VC1_ILWALID_TOKEN                                      (0xDEC20003)
#define LWC6B0_DEC_ERROR_VC1_SLICEDATA_MISSING                                  (0xDEC20004)
#define LWC6B0_DEC_ERROR_VC1_HWERR_INTERRUPT                                    (0xDEC20005)
#define LWC6B0_DEC_ERROR_VC1_DETECTED_VLD_FAILURE                               (0xDEC20006)
#define LWC6B0_DEC_ERROR_VC1_TIMEOUT_POLLING_FOR_DATA                           (0xDEC20007)
#define LWC6B0_DEC_ERROR_VC1_PDEC_PIC_END_UNALIGNED                             (0xDEC20008)
#define LWC6B0_DEC_ERROR_VC1_WDTIMER_EXPIRED                                    (0xDEC20009)
#define LWC6B0_DEC_ERROR_VC1_ERRINTSTART                                        (0xDEC20010)
#define LWC6B0_DEC_ERROR_VC1_IQT_ERRINT                                         (0xDEC20011)
#define LWC6B0_DEC_ERROR_VC1_MC_ERRINT                                          (0xDEC20012)
#define LWC6B0_DEC_ERROR_VC1_MC_IQT_ERRINT                                      (0xDEC20013)
#define LWC6B0_DEC_ERROR_VC1_REC_ERRINT                                         (0xDEC20014)
#define LWC6B0_DEC_ERROR_VC1_REC_IQT_ERRINT                                     (0xDEC20015)
#define LWC6B0_DEC_ERROR_VC1_REC_MC_ERRINT                                      (0xDEC20016)
#define LWC6B0_DEC_ERROR_VC1_REC_MC_IQT_ERRINT                                  (0xDEC20017)
#define LWC6B0_DEC_ERROR_VC1_DBF_ERRINT                                         (0xDEC20018)
#define LWC6B0_DEC_ERROR_VC1_DBF_IQT_ERRINT                                     (0xDEC20019)
#define LWC6B0_DEC_ERROR_VC1_DBF_MC_ERRINT                                      (0xDEC2001A)
#define LWC6B0_DEC_ERROR_VC1_DBF_MC_IQT_ERRINT                                  (0xDEC2001B)
#define LWC6B0_DEC_ERROR_VC1_DBF_REC_ERRINT                                     (0xDEC2001C)
#define LWC6B0_DEC_ERROR_VC1_DBF_REC_IQT_ERRINT                                 (0xDEC2001D)
#define LWC6B0_DEC_ERROR_VC1_DBF_REC_MC_ERRINT                                  (0xDEC2001E)
#define LWC6B0_DEC_ERROR_VC1_DBF_REC_MC_IQT_ERRINT                              (0xDEC2001F)
#define LWC6B0_DEC_ERROR_VC1_PICTURE_INIT                                       (0xDEC20100)
#define LWC6B0_DEC_ERROR_VC1_STATEMACHINE_FAILURE                               (0xDEC20101)
#define LWC6B0_DEC_ERROR_VC1_ILWALID_CTXID_PIC                                  (0xDEC20901)
#define LWC6B0_DEC_ERROR_VC1_ILWALID_CTXID_UCODE                                (0xDEC20902)
#define LWC6B0_DEC_ERROR_VC1_ILWALID_CTXID_FC                                   (0xDEC20903)
#define LWC6B0_DEC_ERROR_VC1_ILWAILD_CTXID_SLH                                  (0xDEC20904)
#define LWC6B0_DEC_ERROR_VC1_ILWALID_UCODE_SIZE                                 (0xDEC20905)
#define LWC6B0_DEC_ERROR_VC1_ILWALID_SLICE_COUNT                                (0xDEC20906)
#define LWC6B0_DEC_ERROR_H264_APPTIMER_EXPIRED                                  (0xDEC30001)
#define LWC6B0_DEC_ERROR_H264_MVTIMER_EXPIRED                                   (0xDEC30002)
#define LWC6B0_DEC_ERROR_H264_ILWALID_TOKEN                                     (0xDEC30003)
#define LWC6B0_DEC_ERROR_H264_SLICEDATA_MISSING                                 (0xDEC30004)
#define LWC6B0_DEC_ERROR_H264_HWERR_INTERRUPT                                   (0xDEC30005)
#define LWC6B0_DEC_ERROR_H264_DETECTED_VLD_FAILURE                              (0xDEC30006)
#define LWC6B0_DEC_ERROR_H264_ERRINTSTART                                       (0xDEC30010)
#define LWC6B0_DEC_ERROR_H264_IQT_ERRINT                                        (0xDEC30011)
#define LWC6B0_DEC_ERROR_H264_MC_ERRINT                                         (0xDEC30012)
#define LWC6B0_DEC_ERROR_H264_MC_IQT_ERRINT                                     (0xDEC30013)
#define LWC6B0_DEC_ERROR_H264_REC_ERRINT                                        (0xDEC30014)
#define LWC6B0_DEC_ERROR_H264_REC_IQT_ERRINT                                    (0xDEC30015)
#define LWC6B0_DEC_ERROR_H264_REC_MC_ERRINT                                     (0xDEC30016)
#define LWC6B0_DEC_ERROR_H264_REC_MC_IQT_ERRINT                                 (0xDEC30017)
#define LWC6B0_DEC_ERROR_H264_DBF_ERRINT                                        (0xDEC30018)
#define LWC6B0_DEC_ERROR_H264_DBF_IQT_ERRINT                                    (0xDEC30019)
#define LWC6B0_DEC_ERROR_H264_DBF_MC_ERRINT                                     (0xDEC3001A)
#define LWC6B0_DEC_ERROR_H264_DBF_MC_IQT_ERRINT                                 (0xDEC3001B)
#define LWC6B0_DEC_ERROR_H264_DBF_REC_ERRINT                                    (0xDEC3001C)
#define LWC6B0_DEC_ERROR_H264_DBF_REC_IQT_ERRINT                                (0xDEC3001D)
#define LWC6B0_DEC_ERROR_H264_DBF_REC_MC_ERRINT                                 (0xDEC3001E)
#define LWC6B0_DEC_ERROR_H264_DBF_REC_MC_IQT_ERRINT                             (0xDEC3001F)
#define LWC6B0_DEC_ERROR_H264_PICTURE_INIT                                      (0xDEC30100)
#define LWC6B0_DEC_ERROR_H264_STATEMACHINE_FAILURE                              (0xDEC30101)
#define LWC6B0_DEC_ERROR_H264_ILWALID_CTXID_PIC                                 (0xDEC30901)
#define LWC6B0_DEC_ERROR_H264_ILWALID_CTXID_UCODE                               (0xDEC30902)
#define LWC6B0_DEC_ERROR_H264_ILWALID_CTXID_FC                                  (0xDEC30903)
#define LWC6B0_DEC_ERROR_H264_ILWALID_CTXID_SLH                                 (0xDEC30904)
#define LWC6B0_DEC_ERROR_H264_ILWALID_UCODE_SIZE                                (0xDEC30905)
#define LWC6B0_DEC_ERROR_H264_ILWALID_SLICE_COUNT                               (0xDEC30906)
#define LWC6B0_DEC_ERROR_MPEG4_APPTIMER_EXPIRED                                 (0xDEC40001)
#define LWC6B0_DEC_ERROR_MPEG4_MVTIMER_EXPIRED                                  (0xDEC40002)
#define LWC6B0_DEC_ERROR_MPEG4_ILWALID_TOKEN                                    (0xDEC40003)
#define LWC6B0_DEC_ERROR_MPEG4_SLICEDATA_MISSING                                (0xDEC40004)
#define LWC6B0_DEC_ERROR_MPEG4_HWERR_INTERRUPT                                  (0xDEC40005)
#define LWC6B0_DEC_ERROR_MPEG4_DETECTED_VLD_FAILURE                             (0xDEC40006)
#define LWC6B0_DEC_ERROR_MPEG4_TIMEOUT_POLLING_FOR_DATA                         (0xDEC40007)
#define LWC6B0_DEC_ERROR_MPEG4_PDEC_PIC_END_UNALIGNED                           (0xDEC40008)
#define LWC6B0_DEC_ERROR_MPEG4_WDTIMER_EXPIRED                                  (0xDEC40009)
#define LWC6B0_DEC_ERROR_MPEG4_ERRINTSTART                                      (0xDEC40010)
#define LWC6B0_DEC_ERROR_MPEG4_IQT_ERRINT                                       (0xDEC40011)
#define LWC6B0_DEC_ERROR_MPEG4_MC_ERRINT                                        (0xDEC40012)
#define LWC6B0_DEC_ERROR_MPEG4_MC_IQT_ERRINT                                    (0xDEC40013)
#define LWC6B0_DEC_ERROR_MPEG4_REC_ERRINT                                       (0xDEC40014)
#define LWC6B0_DEC_ERROR_MPEG4_REC_IQT_ERRINT                                   (0xDEC40015)
#define LWC6B0_DEC_ERROR_MPEG4_REC_MC_ERRINT                                    (0xDEC40016)
#define LWC6B0_DEC_ERROR_MPEG4_REC_MC_IQT_ERRINT                                (0xDEC40017)
#define LWC6B0_DEC_ERROR_MPEG4_DBF_ERRINT                                       (0xDEC40018)
#define LWC6B0_DEC_ERROR_MPEG4_DBF_IQT_ERRINT                                   (0xDEC40019)
#define LWC6B0_DEC_ERROR_MPEG4_DBF_MC_ERRINT                                    (0xDEC4001A)
#define LWC6B0_DEC_ERROR_MPEG4_DBF_MC_IQT_ERRINT                                (0xDEC4001B)
#define LWC6B0_DEC_ERROR_MPEG4_DBF_REC_ERRINT                                   (0xDEC4001C)
#define LWC6B0_DEC_ERROR_MPEG4_DBF_REC_IQT_ERRINT                               (0xDEC4001D)
#define LWC6B0_DEC_ERROR_MPEG4_DBF_REC_MC_ERRINT                                (0xDEC4001E)
#define LWC6B0_DEC_ERROR_MPEG4_DBF_REC_MC_IQT_ERRINT                            (0xDEC4001F)
#define LWC6B0_DEC_ERROR_MPEG4_PICTURE_INIT                                     (0xDEC40100)
#define LWC6B0_DEC_ERROR_MPEG4_STATEMACHINE_FAILURE                             (0xDEC40101)
#define LWC6B0_DEC_ERROR_MPEG4_ILWALID_CTXID_PIC                                (0xDEC40901)
#define LWC6B0_DEC_ERROR_MPEG4_ILWALID_CTXID_UCODE                              (0xDEC40902)
#define LWC6B0_DEC_ERROR_MPEG4_ILWALID_CTXID_FC                                 (0xDEC40903)
#define LWC6B0_DEC_ERROR_MPEG4_ILWALID_CTXID_SLH                                (0xDEC40904)
#define LWC6B0_DEC_ERROR_MPEG4_ILWALID_UCODE_SIZE                               (0xDEC40905)
#define LWC6B0_DEC_ERROR_MPEG4_ILWALID_SLICE_COUNT                              (0xDEC40906)
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // clc6b0_h

