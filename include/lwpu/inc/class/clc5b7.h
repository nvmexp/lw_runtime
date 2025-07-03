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

#ifndef clc5b7_h_
#define clc5b7_h_

#include "lwtypes.h"


#ifdef __cplusplus
extern "C" {
#endif

#define LWC5B7_VIDEO_ENCODER                                                             (0x0000C5B7U)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
typedef volatile struct _clc5b7_tag0 {
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
    LwV32 CtxSaveArea;                                                          // 0x0000024C - 0x0000024F
    LwV32 CtxSwitch;                                                            // 0x00000250 - 0x00000253
    LwV32 Reserved04[0x2B];
    LwV32 Execute;                                                              // 0x00000300 - 0x00000303
    LwV32 SemaphoreD;                                                           // 0x00000304 - 0x00000307
    LwV32 Reserved05[0x3E];
    LwV32 SetInRefPic0Luma;                                                     // 0x00000400 - 0x00000403
    LwV32 SetInRefPic1Luma;                                                     // 0x00000404 - 0x00000407
    LwV32 SetInRefPic2Luma;                                                     // 0x00000408 - 0x0000040B
    LwV32 SetInRefPic3Luma;                                                     // 0x0000040C - 0x0000040F
    LwV32 SetInRefPic4Luma;                                                     // 0x00000410 - 0x00000413
    LwV32 SetInRefPic5Luma;                                                     // 0x00000414 - 0x00000417
    LwV32 SetInRefPic6Luma;                                                     // 0x00000418 - 0x0000041B
    LwV32 SetInRefPic7Luma;                                                     // 0x0000041C - 0x0000041F
    LwV32 SetInRefPic8Luma;                                                     // 0x00000420 - 0x00000423
    LwV32 SetInRefPic9Luma;                                                     // 0x00000424 - 0x00000427
    LwV32 SetInRefPic10Luma;                                                    // 0x00000428 - 0x0000042B
    LwV32 SetInRefPic11Luma;                                                    // 0x0000042C - 0x0000042F
    LwV32 SetInRefPic12Luma;                                                    // 0x00000430 - 0x00000433
    LwV32 SetInRefPic13Luma;                                                    // 0x00000434 - 0x00000437
    LwV32 SetInRefPic14Luma;                                                    // 0x00000438 - 0x0000043B
    LwV32 SetInRefPic15Luma;                                                    // 0x0000043C - 0x0000043F
    LwV32 SetInRefPic0Chroma;                                                   // 0x00000440 - 0x00000443
    LwV32 SetInRefPic1Chroma;                                                   // 0x00000444 - 0x00000447
    LwV32 SetInRefPic2Chroma;                                                   // 0x00000448 - 0x0000044B
    LwV32 SetInRefPic3Chroma;                                                   // 0x0000044C - 0x0000044F
    LwV32 SetInRefPic4Chroma;                                                   // 0x00000450 - 0x00000453
    LwV32 SetInRefPic5Chroma;                                                   // 0x00000454 - 0x00000457
    LwV32 SetInRefPic6Chroma;                                                   // 0x00000458 - 0x0000045B
    LwV32 SetInRefPic7Chroma;                                                   // 0x0000045C - 0x0000045F
    LwV32 SetInRefPic8Chroma;                                                   // 0x00000460 - 0x00000463
    LwV32 SetInRefPic9Chroma;                                                   // 0x00000464 - 0x00000467
    LwV32 SetInRefPic10Chroma;                                                  // 0x00000468 - 0x0000046B
    LwV32 SetInRefPic11Chroma;                                                  // 0x0000046C - 0x0000046F
    LwV32 SetInRefPic12Chroma;                                                  // 0x00000470 - 0x00000473
    LwV32 SetInRefPic13Chroma;                                                  // 0x00000474 - 0x00000477
    LwV32 SetInRefPic14Chroma;                                                  // 0x00000478 - 0x0000047B
    LwV32 SetInRefPic15Chroma;                                                  // 0x0000047C - 0x0000047F
    LwV32 Reserved06[0x20];
    LwV32 SetInRefPicLastLuma;                                                  // 0x00000500 - 0x00000503
    LwV32 SetInRefPicGoldenLuma;                                                // 0x00000504 - 0x00000507
    LwV32 SetInRefPicAltrefLuma;                                                // 0x00000508 - 0x0000050B
    LwV32 SetUcodeState;                                                        // 0x0000050C - 0x0000050F
    LwV32 SetIoVp8EncStatus;                                                    // 0x00000510 - 0x00000513
    LwV32 SetOutBitstreamRes;                                                   // 0x00000514 - 0x00000517
    LwV32 SetInRefPicLastChroma;                                                // 0x00000518 - 0x0000051B
    LwV32 SetInRefPicGoldenChroma;                                              // 0x0000051C - 0x0000051F
    LwV32 SetInRefPicAltrefChroma;                                              // 0x00000520 - 0x00000523
    LwV32 SetOutVP9CounterData;                                                 // 0x00000524 - 0x00000527
    LwV32 SetIlwP9ProbabilityData;                                              // 0x00000528 - 0x0000052B
    LwV32 SetIlwP9LwrTemporalData;                                              // 0x0000052C - 0x0000052F
    LwV32 SetIlwP9RefTemporalData;                                              // 0x00000530 - 0x00000533
    LwV32 SetIlwP9CombinedlineBuf;                                              // 0x00000534 - 0x00000537
    LwV32 SetIlwP9FilterlineBuf;                                                // 0x00000538 - 0x0000053B
    LwV32 SetIlwP9FiltercollineBuf;                                             // 0x0000053C - 0x0000053F
    LwV32 Reserved07[0x1];
    LwV32 SetIlwP9MocompPicLuma;                                                // 0x00000544 - 0x00000547
    LwV32 SetIlwP9MocompPicChroma;                                              // 0x00000548 - 0x0000054B
    LwV32 Reserved08[0x6D];
    LwV32 SetControlParams;                                                     // 0x00000700 - 0x00000703
    LwV32 SetPictureIndex;                                                      // 0x00000704 - 0x00000707
    LwV32 SetOutEncryptParams;                                                  // 0x00000708 - 0x0000070B
    LwV32 SetInRCData;                                                          // 0x0000070C - 0x0000070F
    LwV32 SetInDrvPicSetup;                                                     // 0x00000710 - 0x00000713
    LwV32 SetInCEAHintsData;                                                    // 0x00000714 - 0x00000717
    LwV32 SetOutEncStatus;                                                      // 0x00000718 - 0x0000071B
    LwV32 SetOutBitstream;                                                      // 0x0000071C - 0x0000071F
    LwV32 SetIOHistory;                                                         // 0x00000720 - 0x00000723
    LwV32 SetIoRcProcess;                                                       // 0x00000724 - 0x00000727
    LwV32 SetInColocData;                                                       // 0x00000728 - 0x0000072B
    LwV32 SetOutColocData;                                                      // 0x0000072C - 0x0000072F
    LwV32 SetOutRefPicLuma;                                                     // 0x00000730 - 0x00000733
    LwV32 SetInLwrPic;                                                          // 0x00000734 - 0x00000737
    LwV32 SetInMEPredData;                                                      // 0x00000738 - 0x0000073B
    LwV32 SetOutMEPredData;                                                     // 0x0000073C - 0x0000073F
    LwV32 SetInLwrPicChromaU;                                                   // 0x00000740 - 0x00000743
    LwV32 SetInLwrPicChromaV;                                                   // 0x00000744 - 0x00000747
    LwV32 SetInQpMap;                                                           // 0x00000748 - 0x0000074B
    LwV32 SetOutRefPicChroma;                                                   // 0x0000074C - 0x0000074F
    LwV32 SetInPartitionBuf;                                                    // 0x00000750 - 0x00000753
    LwV32 Reserved09[0x1];
    LwV32 SetInLwrPic1TaskStatus;                                               // 0x00000758 - 0x0000075B
    LwV32 SetInLwrPic2TaskStatus;                                               // 0x0000075C - 0x0000075F
    LwV32 SetOutTaskStatus;                                                     // 0x00000760 - 0x00000763
    LwV32 SetInMvHintsTaskStatus;                                               // 0x00000764 - 0x00000767
    LwV32 Reserved10[0x26B];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved11[0x3BA];
} LWC5B7_VIDEO_ENCODERControlPio;
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#define LWC5B7_NOP                                                              (0x00000100U)
#define LWC5B7_NOP_PARAMETER                                                    31:0
#define LWC5B7_NOP_PARAMETER_HIGH_FIELD                                         31U
#define LWC5B7_NOP_PARAMETER_LOW_FIELD                                          0U
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LWC5B7_PM_TRIGGER                                                       (0x00000140U)
#define LWC5B7_PM_TRIGGER_V                                                     31:0
#define LWC5B7_PM_TRIGGER_V_HIGH_FIELD                                          31U
#define LWC5B7_PM_TRIGGER_V_LOW_FIELD                                           0U
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LWC5B7_SET_APPLICATION_ID                                               (0x00000200U)
#define LWC5B7_SET_APPLICATION_ID_ID                                            31:0
#define LWC5B7_SET_APPLICATION_ID_ID_HIGH_FIELD                                 31U
#define LWC5B7_SET_APPLICATION_ID_ID_LOW_FIELD                                  0U
#define LWC5B7_SET_APPLICATION_ID_ID_LWENC_H264                                 (0x00000001U)
#define LWC5B7_SET_APPLICATION_ID_ID_LWENC_VP8                                  (0x00000002U)
#define LWC5B7_SET_APPLICATION_ID_ID_LWENC_H265                                 (0x00000003U)
#define LWC5B7_SET_APPLICATION_ID_ID_LWENC_VP9                                  (0x00000004U)
#define LWC5B7_SET_APPLICATION_ID_ID_LWENC_RC                                   (0x00000005U)
#define LWC5B7_SET_APPLICATION_ID_ID_LWENC_OFS                                  (0x00000006U)
#define LWC5B7_SET_WATCHDOG_TIMER                                               (0x00000204U)
#define LWC5B7_SET_WATCHDOG_TIMER_TIMER                                         31:0
#define LWC5B7_SET_WATCHDOG_TIMER_TIMER_HIGH_FIELD                              31U
#define LWC5B7_SET_WATCHDOG_TIMER_TIMER_LOW_FIELD                               0U
#define LWC5B7_SEMAPHORE_A                                                      (0x00000240U)
#define LWC5B7_SEMAPHORE_A_UPPER                                                7:0
#define LWC5B7_SEMAPHORE_A_UPPER_HIGH_FIELD                                     7U
#define LWC5B7_SEMAPHORE_A_UPPER_LOW_FIELD                                      0U
#define LWC5B7_SEMAPHORE_B                                                      (0x00000244U)
#define LWC5B7_SEMAPHORE_B_LOWER                                                31:0
#define LWC5B7_SEMAPHORE_B_LOWER_HIGH_FIELD                                     31U
#define LWC5B7_SEMAPHORE_B_LOWER_LOW_FIELD                                      0U
#define LWC5B7_SEMAPHORE_C                                                      (0x00000248U)
#define LWC5B7_SEMAPHORE_C_PAYLOAD                                              31:0
#define LWC5B7_SEMAPHORE_C_PAYLOAD_HIGH_FIELD                                   31U
#define LWC5B7_SEMAPHORE_C_PAYLOAD_LOW_FIELD                                    0U
#define LWC5B7_CTX_SAVE_AREA                                                    (0x0000024LW)
#define LWC5B7_CTX_SAVE_AREA_OFFSET                                             27:0
#define LWC5B7_CTX_SAVE_AREA_OFFSET_HIGH_FIELD                                  27U
#define LWC5B7_CTX_SAVE_AREA_OFFSET_LOW_FIELD                                   0U
#define LWC5B7_CTX_SAVE_AREA_CTX_VALID                                          31:28
#define LWC5B7_CTX_SAVE_AREA_CTX_VALID_HIGH_FIELD                               31U
#define LWC5B7_CTX_SAVE_AREA_CTX_VALID_LOW_FIELD                                28U
#define LWC5B7_CTX_SWITCH                                                       (0x00000250U)
#define LWC5B7_CTX_SWITCH_RESTORE                                               0:0
#define LWC5B7_CTX_SWITCH_RESTORE_HIGH_FIELD                                    0U
#define LWC5B7_CTX_SWITCH_RESTORE_LOW_FIELD                                     0U
#define LWC5B7_CTX_SWITCH_RESTORE_FALSE                                         (0x00000000U)
#define LWC5B7_CTX_SWITCH_RESTORE_TRUE                                          (0x00000001U)
#define LWC5B7_CTX_SWITCH_RST_NOTIFY                                            1:1
#define LWC5B7_CTX_SWITCH_RST_NOTIFY_HIGH_FIELD                                 1U
#define LWC5B7_CTX_SWITCH_RST_NOTIFY_LOW_FIELD                                  1U
#define LWC5B7_CTX_SWITCH_RST_NOTIFY_FALSE                                      (0x00000000U)
#define LWC5B7_CTX_SWITCH_RST_NOTIFY_TRUE                                       (0x00000001U)
#define LWC5B7_CTX_SWITCH_RESERVED                                              7:2
#define LWC5B7_CTX_SWITCH_RESERVED_HIGH_FIELD                                   7U
#define LWC5B7_CTX_SWITCH_RESERVED_LOW_FIELD                                    2U
#define LWC5B7_CTX_SWITCH_ASID                                                  23:8
#define LWC5B7_CTX_SWITCH_ASID_HIGH_FIELD                                       23U
#define LWC5B7_CTX_SWITCH_ASID_LOW_FIELD                                        8U
#define LWC5B7_EXELWTE                                                          (0x00000300U)
#define LWC5B7_EXELWTE_NOTIFY                                                   0:0
#define LWC5B7_EXELWTE_NOTIFY_HIGH_FIELD                                        0U
#define LWC5B7_EXELWTE_NOTIFY_LOW_FIELD                                         0U
#define LWC5B7_EXELWTE_NOTIFY_DISABLE                                           (0x00000000U)
#define LWC5B7_EXELWTE_NOTIFY_ENABLE                                            (0x00000001U)
#define LWC5B7_EXELWTE_NOTIFY_ON                                                1:1
#define LWC5B7_EXELWTE_NOTIFY_ON_HIGH_FIELD                                     1U
#define LWC5B7_EXELWTE_NOTIFY_ON_LOW_FIELD                                      1U
#define LWC5B7_EXELWTE_NOTIFY_ON_END                                            (0x00000000U)
#define LWC5B7_EXELWTE_NOTIFY_ON_BEGIN                                          (0x00000001U)
#define LWC5B7_EXELWTE_AWAKEN                                                   8:8
#define LWC5B7_EXELWTE_AWAKEN_HIGH_FIELD                                        8U
#define LWC5B7_EXELWTE_AWAKEN_LOW_FIELD                                         8U
#define LWC5B7_EXELWTE_AWAKEN_DISABLE                                           (0x00000000U)
#define LWC5B7_EXELWTE_AWAKEN_ENABLE                                            (0x00000001U)
#define LWC5B7_SEMAPHORE_D                                                      (0x00000304U)
#define LWC5B7_SEMAPHORE_D_STRUCTURE_SIZE                                       0:0
#define LWC5B7_SEMAPHORE_D_STRUCTURE_SIZE_HIGH_FIELD                            0U
#define LWC5B7_SEMAPHORE_D_STRUCTURE_SIZE_LOW_FIELD                             0U
#define LWC5B7_SEMAPHORE_D_STRUCTURE_SIZE_ONE                                   (0x00000000U)
#define LWC5B7_SEMAPHORE_D_STRUCTURE_SIZE_FOUR                                  (0x00000001U)
#define LWC5B7_SEMAPHORE_D_AWAKEN_ENABLE                                        8:8
#define LWC5B7_SEMAPHORE_D_AWAKEN_ENABLE_HIGH_FIELD                             8U
#define LWC5B7_SEMAPHORE_D_AWAKEN_ENABLE_LOW_FIELD                              8U
#define LWC5B7_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                  (0x00000000U)
#define LWC5B7_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                   (0x00000001U)
#define LWC5B7_SEMAPHORE_D_OPERATION                                            17:16
#define LWC5B7_SEMAPHORE_D_OPERATION_HIGH_FIELD                                 17U
#define LWC5B7_SEMAPHORE_D_OPERATION_LOW_FIELD                                  16U
#define LWC5B7_SEMAPHORE_D_OPERATION_RELEASE                                    (0x00000000U)
#define LWC5B7_SEMAPHORE_D_OPERATION_RESERVED0                                  (0x00000001U)
#define LWC5B7_SEMAPHORE_D_OPERATION_RESERVED1                                  (0x00000002U)
#define LWC5B7_SEMAPHORE_D_OPERATION_TRAP                                       (0x00000003U)
#define LWC5B7_SEMAPHORE_D_FLUSH_DISABLE                                        21:21
#define LWC5B7_SEMAPHORE_D_FLUSH_DISABLE_HIGH_FIELD                             21U
#define LWC5B7_SEMAPHORE_D_FLUSH_DISABLE_LOW_FIELD                              21U
#define LWC5B7_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                  (0x00000000U)
#define LWC5B7_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                   (0x00000001U)
#define LWC5B7_SET_IN_REF_PIC0_LUMA                                             (0x00000400U)
#define LWC5B7_SET_IN_REF_PIC0_LUMA_OFFSET                                      31:0
#define LWC5B7_SET_IN_REF_PIC0_LUMA_OFFSET_HIGH_FIELD                           31U
#define LWC5B7_SET_IN_REF_PIC0_LUMA_OFFSET_LOW_FIELD                            0U
#define LWC5B7_SET_IN_REF_PIC1_LUMA                                             (0x00000404U)
#define LWC5B7_SET_IN_REF_PIC1_LUMA_OFFSET                                      31:0
#define LWC5B7_SET_IN_REF_PIC1_LUMA_OFFSET_HIGH_FIELD                           31U
#define LWC5B7_SET_IN_REF_PIC1_LUMA_OFFSET_LOW_FIELD                            0U
#define LWC5B7_SET_IN_REF_PIC2_LUMA                                             (0x00000408U)
#define LWC5B7_SET_IN_REF_PIC2_LUMA_OFFSET                                      31:0
#define LWC5B7_SET_IN_REF_PIC2_LUMA_OFFSET_HIGH_FIELD                           31U
#define LWC5B7_SET_IN_REF_PIC2_LUMA_OFFSET_LOW_FIELD                            0U
#define LWC5B7_SET_IN_REF_PIC3_LUMA                                             (0x0000040LW)
#define LWC5B7_SET_IN_REF_PIC3_LUMA_OFFSET                                      31:0
#define LWC5B7_SET_IN_REF_PIC3_LUMA_OFFSET_HIGH_FIELD                           31U
#define LWC5B7_SET_IN_REF_PIC3_LUMA_OFFSET_LOW_FIELD                            0U
#define LWC5B7_SET_IN_REF_PIC4_LUMA                                             (0x00000410U)
#define LWC5B7_SET_IN_REF_PIC4_LUMA_OFFSET                                      31:0
#define LWC5B7_SET_IN_REF_PIC4_LUMA_OFFSET_HIGH_FIELD                           31U
#define LWC5B7_SET_IN_REF_PIC4_LUMA_OFFSET_LOW_FIELD                            0U
#define LWC5B7_SET_IN_REF_PIC5_LUMA                                             (0x00000414U)
#define LWC5B7_SET_IN_REF_PIC5_LUMA_OFFSET                                      31:0
#define LWC5B7_SET_IN_REF_PIC5_LUMA_OFFSET_HIGH_FIELD                           31U
#define LWC5B7_SET_IN_REF_PIC5_LUMA_OFFSET_LOW_FIELD                            0U
#define LWC5B7_SET_IN_REF_PIC6_LUMA                                             (0x00000418U)
#define LWC5B7_SET_IN_REF_PIC6_LUMA_OFFSET                                      31:0
#define LWC5B7_SET_IN_REF_PIC6_LUMA_OFFSET_HIGH_FIELD                           31U
#define LWC5B7_SET_IN_REF_PIC6_LUMA_OFFSET_LOW_FIELD                            0U
#define LWC5B7_SET_IN_REF_PIC7_LUMA                                             (0x0000041LW)
#define LWC5B7_SET_IN_REF_PIC7_LUMA_OFFSET                                      31:0
#define LWC5B7_SET_IN_REF_PIC7_LUMA_OFFSET_HIGH_FIELD                           31U
#define LWC5B7_SET_IN_REF_PIC7_LUMA_OFFSET_LOW_FIELD                            0U
#define LWC5B7_SET_IN_REF_PIC8_LUMA                                             (0x00000420U)
#define LWC5B7_SET_IN_REF_PIC8_LUMA_OFFSET                                      31:0
#define LWC5B7_SET_IN_REF_PIC8_LUMA_OFFSET_HIGH_FIELD                           31U
#define LWC5B7_SET_IN_REF_PIC8_LUMA_OFFSET_LOW_FIELD                            0U
#define LWC5B7_SET_IN_REF_PIC9_LUMA                                             (0x00000424U)
#define LWC5B7_SET_IN_REF_PIC9_LUMA_OFFSET                                      31:0
#define LWC5B7_SET_IN_REF_PIC9_LUMA_OFFSET_HIGH_FIELD                           31U
#define LWC5B7_SET_IN_REF_PIC9_LUMA_OFFSET_LOW_FIELD                            0U
#define LWC5B7_SET_IN_REF_PIC10_LUMA                                            (0x00000428U)
#define LWC5B7_SET_IN_REF_PIC10_LUMA_OFFSET                                     31:0
#define LWC5B7_SET_IN_REF_PIC10_LUMA_OFFSET_HIGH_FIELD                          31U
#define LWC5B7_SET_IN_REF_PIC10_LUMA_OFFSET_LOW_FIELD                           0U
#define LWC5B7_SET_IN_REF_PIC11_LUMA                                            (0x0000042LW)
#define LWC5B7_SET_IN_REF_PIC11_LUMA_OFFSET                                     31:0
#define LWC5B7_SET_IN_REF_PIC11_LUMA_OFFSET_HIGH_FIELD                          31U
#define LWC5B7_SET_IN_REF_PIC11_LUMA_OFFSET_LOW_FIELD                           0U
#define LWC5B7_SET_IN_REF_PIC12_LUMA                                            (0x00000430U)
#define LWC5B7_SET_IN_REF_PIC12_LUMA_OFFSET                                     31:0
#define LWC5B7_SET_IN_REF_PIC12_LUMA_OFFSET_HIGH_FIELD                          31U
#define LWC5B7_SET_IN_REF_PIC12_LUMA_OFFSET_LOW_FIELD                           0U
#define LWC5B7_SET_IN_REF_PIC13_LUMA                                            (0x00000434U)
#define LWC5B7_SET_IN_REF_PIC13_LUMA_OFFSET                                     31:0
#define LWC5B7_SET_IN_REF_PIC13_LUMA_OFFSET_HIGH_FIELD                          31U
#define LWC5B7_SET_IN_REF_PIC13_LUMA_OFFSET_LOW_FIELD                           0U
#define LWC5B7_SET_IN_REF_PIC14_LUMA                                            (0x00000438U)
#define LWC5B7_SET_IN_REF_PIC14_LUMA_OFFSET                                     31:0
#define LWC5B7_SET_IN_REF_PIC14_LUMA_OFFSET_HIGH_FIELD                          31U
#define LWC5B7_SET_IN_REF_PIC14_LUMA_OFFSET_LOW_FIELD                           0U
#define LWC5B7_SET_IN_REF_PIC15_LUMA                                            (0x0000043LW)
#define LWC5B7_SET_IN_REF_PIC15_LUMA_OFFSET                                     31:0
#define LWC5B7_SET_IN_REF_PIC15_LUMA_OFFSET_HIGH_FIELD                          31U
#define LWC5B7_SET_IN_REF_PIC15_LUMA_OFFSET_LOW_FIELD                           0U
#define LWC5B7_SET_IN_REF_PIC0_CHROMA                                           (0x00000440U)
#define LWC5B7_SET_IN_REF_PIC0_CHROMA_OFFSET                                    31:0
#define LWC5B7_SET_IN_REF_PIC0_CHROMA_OFFSET_HIGH_FIELD                         31U
#define LWC5B7_SET_IN_REF_PIC0_CHROMA_OFFSET_LOW_FIELD                          0U
#define LWC5B7_SET_IN_REF_PIC1_CHROMA                                           (0x00000444U)
#define LWC5B7_SET_IN_REF_PIC1_CHROMA_OFFSET                                    31:0
#define LWC5B7_SET_IN_REF_PIC1_CHROMA_OFFSET_HIGH_FIELD                         31U
#define LWC5B7_SET_IN_REF_PIC1_CHROMA_OFFSET_LOW_FIELD                          0U
#define LWC5B7_SET_IN_REF_PIC2_CHROMA                                           (0x00000448U)
#define LWC5B7_SET_IN_REF_PIC2_CHROMA_OFFSET                                    31:0
#define LWC5B7_SET_IN_REF_PIC2_CHROMA_OFFSET_HIGH_FIELD                         31U
#define LWC5B7_SET_IN_REF_PIC2_CHROMA_OFFSET_LOW_FIELD                          0U
#define LWC5B7_SET_IN_REF_PIC3_CHROMA                                           (0x0000044LW)
#define LWC5B7_SET_IN_REF_PIC3_CHROMA_OFFSET                                    31:0
#define LWC5B7_SET_IN_REF_PIC3_CHROMA_OFFSET_HIGH_FIELD                         31U
#define LWC5B7_SET_IN_REF_PIC3_CHROMA_OFFSET_LOW_FIELD                          0U
#define LWC5B7_SET_IN_REF_PIC4_CHROMA                                           (0x00000450U)
#define LWC5B7_SET_IN_REF_PIC4_CHROMA_OFFSET                                    31:0
#define LWC5B7_SET_IN_REF_PIC4_CHROMA_OFFSET_HIGH_FIELD                         31U
#define LWC5B7_SET_IN_REF_PIC4_CHROMA_OFFSET_LOW_FIELD                          0U
#define LWC5B7_SET_IN_REF_PIC5_CHROMA                                           (0x00000454U)
#define LWC5B7_SET_IN_REF_PIC5_CHROMA_OFFSET                                    31:0
#define LWC5B7_SET_IN_REF_PIC5_CHROMA_OFFSET_HIGH_FIELD                         31U
#define LWC5B7_SET_IN_REF_PIC5_CHROMA_OFFSET_LOW_FIELD                          0U
#define LWC5B7_SET_IN_REF_PIC6_CHROMA                                           (0x00000458U)
#define LWC5B7_SET_IN_REF_PIC6_CHROMA_OFFSET                                    31:0
#define LWC5B7_SET_IN_REF_PIC6_CHROMA_OFFSET_HIGH_FIELD                         31U
#define LWC5B7_SET_IN_REF_PIC6_CHROMA_OFFSET_LOW_FIELD                          0U
#define LWC5B7_SET_IN_REF_PIC7_CHROMA                                           (0x0000045LW)
#define LWC5B7_SET_IN_REF_PIC7_CHROMA_OFFSET                                    31:0
#define LWC5B7_SET_IN_REF_PIC7_CHROMA_OFFSET_HIGH_FIELD                         31U
#define LWC5B7_SET_IN_REF_PIC7_CHROMA_OFFSET_LOW_FIELD                          0U
#define LWC5B7_SET_IN_REF_PIC8_CHROMA                                           (0x00000460U)
#define LWC5B7_SET_IN_REF_PIC8_CHROMA_OFFSET                                    31:0
#define LWC5B7_SET_IN_REF_PIC8_CHROMA_OFFSET_HIGH_FIELD                         31U
#define LWC5B7_SET_IN_REF_PIC8_CHROMA_OFFSET_LOW_FIELD                          0U
#define LWC5B7_SET_IN_REF_PIC9_CHROMA                                           (0x00000464U)
#define LWC5B7_SET_IN_REF_PIC9_CHROMA_OFFSET                                    31:0
#define LWC5B7_SET_IN_REF_PIC9_CHROMA_OFFSET_HIGH_FIELD                         31U
#define LWC5B7_SET_IN_REF_PIC9_CHROMA_OFFSET_LOW_FIELD                          0U
#define LWC5B7_SET_IN_REF_PIC10_CHROMA                                          (0x00000468U)
#define LWC5B7_SET_IN_REF_PIC10_CHROMA_OFFSET                                   31:0
#define LWC5B7_SET_IN_REF_PIC10_CHROMA_OFFSET_HIGH_FIELD                        31U
#define LWC5B7_SET_IN_REF_PIC10_CHROMA_OFFSET_LOW_FIELD                         0U
#define LWC5B7_SET_IN_REF_PIC11_CHROMA                                          (0x0000046LW)
#define LWC5B7_SET_IN_REF_PIC11_CHROMA_OFFSET                                   31:0
#define LWC5B7_SET_IN_REF_PIC11_CHROMA_OFFSET_HIGH_FIELD                        31U
#define LWC5B7_SET_IN_REF_PIC11_CHROMA_OFFSET_LOW_FIELD                         0U
#define LWC5B7_SET_IN_REF_PIC12_CHROMA                                          (0x00000470U)
#define LWC5B7_SET_IN_REF_PIC12_CHROMA_OFFSET                                   31:0
#define LWC5B7_SET_IN_REF_PIC12_CHROMA_OFFSET_HIGH_FIELD                        31U
#define LWC5B7_SET_IN_REF_PIC12_CHROMA_OFFSET_LOW_FIELD                         0U
#define LWC5B7_SET_IN_REF_PIC13_CHROMA                                          (0x00000474U)
#define LWC5B7_SET_IN_REF_PIC13_CHROMA_OFFSET                                   31:0
#define LWC5B7_SET_IN_REF_PIC13_CHROMA_OFFSET_HIGH_FIELD                        31U
#define LWC5B7_SET_IN_REF_PIC13_CHROMA_OFFSET_LOW_FIELD                         0U
#define LWC5B7_SET_IN_REF_PIC14_CHROMA                                          (0x00000478U)
#define LWC5B7_SET_IN_REF_PIC14_CHROMA_OFFSET                                   31:0
#define LWC5B7_SET_IN_REF_PIC14_CHROMA_OFFSET_HIGH_FIELD                        31U
#define LWC5B7_SET_IN_REF_PIC14_CHROMA_OFFSET_LOW_FIELD                         0U
#define LWC5B7_SET_IN_REF_PIC15_CHROMA                                          (0x0000047LW)
#define LWC5B7_SET_IN_REF_PIC15_CHROMA_OFFSET                                   31:0
#define LWC5B7_SET_IN_REF_PIC15_CHROMA_OFFSET_HIGH_FIELD                        31U
#define LWC5B7_SET_IN_REF_PIC15_CHROMA_OFFSET_LOW_FIELD                         0U
#define LWC5B7_SET_IN_REF_PIC_LAST_LUMA                                         (0x00000500U)
#define LWC5B7_SET_IN_REF_PIC_LAST_LUMA_OFFSET                                  31:0
#define LWC5B7_SET_IN_REF_PIC_LAST_LUMA_OFFSET_HIGH_FIELD                       31U
#define LWC5B7_SET_IN_REF_PIC_LAST_LUMA_OFFSET_LOW_FIELD                        0U
#define LWC5B7_SET_IN_REF_PIC_GOLDEN_LUMA                                       (0x00000504U)
#define LWC5B7_SET_IN_REF_PIC_GOLDEN_LUMA_OFFSET                                31:0
#define LWC5B7_SET_IN_REF_PIC_GOLDEN_LUMA_OFFSET_HIGH_FIELD                     31U
#define LWC5B7_SET_IN_REF_PIC_GOLDEN_LUMA_OFFSET_LOW_FIELD                      0U
#define LWC5B7_SET_IN_REF_PIC_ALTREF_LUMA                                       (0x00000508U)
#define LWC5B7_SET_IN_REF_PIC_ALTREF_LUMA_OFFSET                                31:0
#define LWC5B7_SET_IN_REF_PIC_ALTREF_LUMA_OFFSET_HIGH_FIELD                     31U
#define LWC5B7_SET_IN_REF_PIC_ALTREF_LUMA_OFFSET_LOW_FIELD                      0U
#define LWC5B7_SET_UCODE_STATE                                                  (0x0000050LW)
#define LWC5B7_SET_UCODE_STATE_OFFSET                                           31:0
#define LWC5B7_SET_UCODE_STATE_OFFSET_HIGH_FIELD                                31U
#define LWC5B7_SET_UCODE_STATE_OFFSET_LOW_FIELD                                 0U
#define LWC5B7_SET_IO_VP8_ENC_STATUS                                            (0x00000510U)
#define LWC5B7_SET_IO_VP8_ENC_STATUS_OFFSET                                     31:0
#define LWC5B7_SET_IO_VP8_ENC_STATUS_OFFSET_HIGH_FIELD                          31U
#define LWC5B7_SET_IO_VP8_ENC_STATUS_OFFSET_LOW_FIELD                           0U
#define LWC5B7_SET_OUT_BITSTREAM_RES                                            (0x00000514U)
#define LWC5B7_SET_OUT_BITSTREAM_RES_OFFSET                                     31:0
#define LWC5B7_SET_OUT_BITSTREAM_RES_OFFSET_HIGH_FIELD                          31U
#define LWC5B7_SET_OUT_BITSTREAM_RES_OFFSET_LOW_FIELD                           0U
#define LWC5B7_SET_IN_REF_PIC_LAST_CHROMA                                       (0x00000518U)
#define LWC5B7_SET_IN_REF_PIC_LAST_CHROMA_OFFSET                                31:0
#define LWC5B7_SET_IN_REF_PIC_LAST_CHROMA_OFFSET_HIGH_FIELD                     31U
#define LWC5B7_SET_IN_REF_PIC_LAST_CHROMA_OFFSET_LOW_FIELD                      0U
#define LWC5B7_SET_IN_REF_PIC_GOLDEN_CHROMA                                     (0x0000051LW)
#define LWC5B7_SET_IN_REF_PIC_GOLDEN_CHROMA_OFFSET                              31:0
#define LWC5B7_SET_IN_REF_PIC_GOLDEN_CHROMA_OFFSET_HIGH_FIELD                   31U
#define LWC5B7_SET_IN_REF_PIC_GOLDEN_CHROMA_OFFSET_LOW_FIELD                    0U
#define LWC5B7_SET_IN_REF_PIC_ALTREF_CHROMA                                     (0x00000520U)
#define LWC5B7_SET_IN_REF_PIC_ALTREF_CHROMA_OFFSET                              31:0
#define LWC5B7_SET_IN_REF_PIC_ALTREF_CHROMA_OFFSET_HIGH_FIELD                   31U
#define LWC5B7_SET_IN_REF_PIC_ALTREF_CHROMA_OFFSET_LOW_FIELD                    0U
#define LWC5B7_SET_OUT_VP9_COUNTER_DATA                                         (0x00000524U)
#define LWC5B7_SET_OUT_VP9_COUNTER_DATA_OFFSET                                  31:0
#define LWC5B7_SET_OUT_VP9_COUNTER_DATA_OFFSET_HIGH_FIELD                       31U
#define LWC5B7_SET_OUT_VP9_COUNTER_DATA_OFFSET_LOW_FIELD                        0U
#define LWC5B7_SET_IN_VP9_PROBABILITY_DATA                                      (0x00000528U)
#define LWC5B7_SET_IN_VP9_PROBABILITY_DATA_OFFSET                               31:0
#define LWC5B7_SET_IN_VP9_PROBABILITY_DATA_OFFSET_HIGH_FIELD                    31U
#define LWC5B7_SET_IN_VP9_PROBABILITY_DATA_OFFSET_LOW_FIELD                     0U
#define LWC5B7_SET_IN_VP9_LWR_TEMPORAL_DATA                                     (0x0000052LW)
#define LWC5B7_SET_IN_VP9_LWR_TEMPORAL_DATA_OFFSET                              31:0
#define LWC5B7_SET_IN_VP9_LWR_TEMPORAL_DATA_OFFSET_HIGH_FIELD                   31U
#define LWC5B7_SET_IN_VP9_LWR_TEMPORAL_DATA_OFFSET_LOW_FIELD                    0U
#define LWC5B7_SET_IN_VP9_REF_TEMPORAL_DATA                                     (0x00000530U)
#define LWC5B7_SET_IN_VP9_REF_TEMPORAL_DATA_OFFSET                              31:0
#define LWC5B7_SET_IN_VP9_REF_TEMPORAL_DATA_OFFSET_HIGH_FIELD                   31U
#define LWC5B7_SET_IN_VP9_REF_TEMPORAL_DATA_OFFSET_LOW_FIELD                    0U
#define LWC5B7_SET_IN_VP9_COMBINEDLINE_BUF                                      (0x00000534U)
#define LWC5B7_SET_IN_VP9_COMBINEDLINE_BUF_OFFSET                               31:0
#define LWC5B7_SET_IN_VP9_COMBINEDLINE_BUF_OFFSET_HIGH_FIELD                    31U
#define LWC5B7_SET_IN_VP9_COMBINEDLINE_BUF_OFFSET_LOW_FIELD                     0U
#define LWC5B7_SET_IN_VP9_FILTERLINE_BUF                                        (0x00000538U)
#define LWC5B7_SET_IN_VP9_FILTERLINE_BUF_OFFSET                                 31:0
#define LWC5B7_SET_IN_VP9_FILTERLINE_BUF_OFFSET_HIGH_FIELD                      31U
#define LWC5B7_SET_IN_VP9_FILTERLINE_BUF_OFFSET_LOW_FIELD                       0U
#define LWC5B7_SET_IN_VP9_FILTERCOLLINE_BUF                                     (0x0000053LW)
#define LWC5B7_SET_IN_VP9_FILTERCOLLINE_BUF_OFFSET                              31:0
#define LWC5B7_SET_IN_VP9_FILTERCOLLINE_BUF_OFFSET_HIGH_FIELD                   31U
#define LWC5B7_SET_IN_VP9_FILTERCOLLINE_BUF_OFFSET_LOW_FIELD                    0U
#define LWC5B7_SET_IN_VP9_MOCOMP_PIC_LUMA                                       (0x00000544U)
#define LWC5B7_SET_IN_VP9_MOCOMP_PIC_LUMA_OFFSET                                31:0
#define LWC5B7_SET_IN_VP9_MOCOMP_PIC_LUMA_OFFSET_HIGH_FIELD                     31U
#define LWC5B7_SET_IN_VP9_MOCOMP_PIC_LUMA_OFFSET_LOW_FIELD                      0U
#define LWC5B7_SET_IN_VP9_MOCOMP_PIC_CHROMA                                     (0x00000548U)
#define LWC5B7_SET_IN_VP9_MOCOMP_PIC_CHROMA_OFFSET                              31:0
#define LWC5B7_SET_IN_VP9_MOCOMP_PIC_CHROMA_OFFSET_HIGH_FIELD                   31U
#define LWC5B7_SET_IN_VP9_MOCOMP_PIC_CHROMA_OFFSET_LOW_FIELD                    0U
#define LWC5B7_SET_CONTROL_PARAMS                                               (0x00000700U)
#define LWC5B7_SET_CONTROL_PARAMS_CODEC_TYPE                                    3:0
#define LWC5B7_SET_CONTROL_PARAMS_CODEC_TYPE_HIGH_FIELD                         3U
#define LWC5B7_SET_CONTROL_PARAMS_CODEC_TYPE_LOW_FIELD                          0U
#define LWC5B7_SET_CONTROL_PARAMS_CODEC_TYPE_H264                               (0x00000003U)
#define LWC5B7_SET_CONTROL_PARAMS_FORCE_OUT_PIC                                 8:8
#define LWC5B7_SET_CONTROL_PARAMS_FORCE_OUT_PIC_HIGH_FIELD                      8U
#define LWC5B7_SET_CONTROL_PARAMS_FORCE_OUT_PIC_LOW_FIELD                       8U
#define LWC5B7_SET_CONTROL_PARAMS_FORCE_OUT_COL                                 9:9
#define LWC5B7_SET_CONTROL_PARAMS_FORCE_OUT_COL_HIGH_FIELD                      9U
#define LWC5B7_SET_CONTROL_PARAMS_FORCE_OUT_COL_LOW_FIELD                       9U
#define LWC5B7_SET_CONTROL_PARAMS_MEONLY                                        10:10
#define LWC5B7_SET_CONTROL_PARAMS_MEONLY_HIGH_FIELD                             10U
#define LWC5B7_SET_CONTROL_PARAMS_MEONLY_LOW_FIELD                              10U
#define LWC5B7_SET_CONTROL_PARAMS_SLICE_STAT_ON                                 11:11
#define LWC5B7_SET_CONTROL_PARAMS_SLICE_STAT_ON_HIGH_FIELD                      11U
#define LWC5B7_SET_CONTROL_PARAMS_SLICE_STAT_ON_LOW_FIELD                       11U
#define LWC5B7_SET_CONTROL_PARAMS_GPTIMER_ON                                    12:12
#define LWC5B7_SET_CONTROL_PARAMS_GPTIMER_ON_HIGH_FIELD                         12U
#define LWC5B7_SET_CONTROL_PARAMS_GPTIMER_ON_LOW_FIELD                          12U
#define LWC5B7_SET_CONTROL_PARAMS_STOP_PROB_UPDATE                              13:13
#define LWC5B7_SET_CONTROL_PARAMS_STOP_PROB_UPDATE_HIGH_FIELD                   13U
#define LWC5B7_SET_CONTROL_PARAMS_STOP_PROB_UPDATE_LOW_FIELD                    13U
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LWC5B7_SET_CONTROL_PARAMS_DUMP_CYCLE_COUNT                              14:14
#define LWC5B7_SET_CONTROL_PARAMS_DUMP_CYCLE_COUNT_HIGH_FIELD                   14U
#define LWC5B7_SET_CONTROL_PARAMS_DUMP_CYCLE_COUNT_LOW_FIELD                    14U
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LWC5B7_SET_CONTROL_PARAMS_MPEC_STAT_ON                                  15:15
#define LWC5B7_SET_CONTROL_PARAMS_MPEC_STAT_ON_HIGH_FIELD                       15U
#define LWC5B7_SET_CONTROL_PARAMS_MPEC_STAT_ON_LOW_FIELD                        15U
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LWC5B7_SET_CONTROL_PARAMS_DEBUG_MODE                                    16:16
#define LWC5B7_SET_CONTROL_PARAMS_DEBUG_MODE_HIGH_FIELD                         16U
#define LWC5B7_SET_CONTROL_PARAMS_DEBUG_MODE_LOW_FIELD                          16U
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LWC5B7_SET_CONTROL_PARAMS_SUBFRAME_MODE                                 18:17
#define LWC5B7_SET_CONTROL_PARAMS_SUBFRAME_MODE_HIGH_FIELD                      18U
#define LWC5B7_SET_CONTROL_PARAMS_SUBFRAME_MODE_LOW_FIELD                       17U
#define LWC5B7_SET_CONTROL_PARAMS_SUBFRAME_MODE_NONE                            (0x00000000U)
#define LWC5B7_SET_CONTROL_PARAMS_SUBFRAME_MODE_SLICE_FLUSH                     (0x00000001U)
#define LWC5B7_SET_CONTROL_PARAMS_SUBFRAME_MODE_SLICE_OFFSETS_WITHOUT_FLUSH     (0x00000002U)
#define LWC5B7_SET_CONTROL_PARAMS_SUBFRAME_MODE_SLICE_OFFSETS_WITH_FLUSH        (0x00000003U)
#define LWC5B7_SET_CONTROL_PARAMS_RCSTAT_WRITE                                  19:19
#define LWC5B7_SET_CONTROL_PARAMS_RCSTAT_WRITE_HIGH_FIELD                       19U
#define LWC5B7_SET_CONTROL_PARAMS_RCSTAT_WRITE_LOW_FIELD                        19U
#define LWC5B7_SET_CONTROL_PARAMS_RCSTAT_READ                                   20:20
#define LWC5B7_SET_CONTROL_PARAMS_RCSTAT_READ_HIGH_FIELD                        20U
#define LWC5B7_SET_CONTROL_PARAMS_RCSTAT_READ_LOW_FIELD                         20U
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LWC5B7_SET_CONTROL_PARAMS_ENCRYPT_ON                                    21:21
#define LWC5B7_SET_CONTROL_PARAMS_ENCRYPT_ON_HIGH_FIELD                         21U
#define LWC5B7_SET_CONTROL_PARAMS_ENCRYPT_ON_LOW_FIELD                          21U
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LWC5B7_SET_CONTROL_PARAMS_RCMODE                                        31:24
#define LWC5B7_SET_CONTROL_PARAMS_RCMODE_HIGH_FIELD                             31U
#define LWC5B7_SET_CONTROL_PARAMS_RCMODE_LOW_FIELD                              24U
#define LWC5B7_SET_CONTROL_PARAMS_RCMODE_CONSTQP                                (0x00000000U)
#define LWC5B7_SET_PICTURE_INDEX                                                (0x00000704U)
#define LWC5B7_SET_PICTURE_INDEX_INDEX                                          31:0
#define LWC5B7_SET_PICTURE_INDEX_INDEX_HIGH_FIELD                               31U
#define LWC5B7_SET_PICTURE_INDEX_INDEX_LOW_FIELD                                0U
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LWC5B7_SET_OUT_ENCRYPT_PARAMS                                           (0x00000708U)
#define LWC5B7_SET_OUT_ENCRYPT_PARAMS_OFFSET                                    31:0
#define LWC5B7_SET_OUT_ENCRYPT_PARAMS_OFFSET_HIGH_FIELD                         31U
#define LWC5B7_SET_OUT_ENCRYPT_PARAMS_OFFSET_LOW_FIELD                          0U
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LWC5B7_SET_IN_RCDATA                                                    (0x0000070LW)
#define LWC5B7_SET_IN_RCDATA_OFFSET                                             31:0
#define LWC5B7_SET_IN_RCDATA_OFFSET_HIGH_FIELD                                  31U
#define LWC5B7_SET_IN_RCDATA_OFFSET_LOW_FIELD                                   0U
#define LWC5B7_SET_IN_DRV_PIC_SETUP                                             (0x00000710U)
#define LWC5B7_SET_IN_DRV_PIC_SETUP_OFFSET                                      31:0
#define LWC5B7_SET_IN_DRV_PIC_SETUP_OFFSET_HIGH_FIELD                           31U
#define LWC5B7_SET_IN_DRV_PIC_SETUP_OFFSET_LOW_FIELD                            0U
#define LWC5B7_SET_IN_CEAHINTS_DATA                                             (0x00000714U)
#define LWC5B7_SET_IN_CEAHINTS_DATA_OFFSET                                      31:0
#define LWC5B7_SET_IN_CEAHINTS_DATA_OFFSET_HIGH_FIELD                           31U
#define LWC5B7_SET_IN_CEAHINTS_DATA_OFFSET_LOW_FIELD                            0U
#define LWC5B7_SET_OUT_ENC_STATUS                                               (0x00000718U)
#define LWC5B7_SET_OUT_ENC_STATUS_OFFSET                                        31:0
#define LWC5B7_SET_OUT_ENC_STATUS_OFFSET_HIGH_FIELD                             31U
#define LWC5B7_SET_OUT_ENC_STATUS_OFFSET_LOW_FIELD                              0U
#define LWC5B7_SET_OUT_BITSTREAM                                                (0x0000071LW)
#define LWC5B7_SET_OUT_BITSTREAM_OFFSET                                         31:0
#define LWC5B7_SET_OUT_BITSTREAM_OFFSET_HIGH_FIELD                              31U
#define LWC5B7_SET_OUT_BITSTREAM_OFFSET_LOW_FIELD                               0U
#define LWC5B7_SET_IOHISTORY                                                    (0x00000720U)
#define LWC5B7_SET_IOHISTORY_OFFSET                                             31:0
#define LWC5B7_SET_IOHISTORY_OFFSET_HIGH_FIELD                                  31U
#define LWC5B7_SET_IOHISTORY_OFFSET_LOW_FIELD                                   0U
#define LWC5B7_SET_IO_RC_PROCESS                                                (0x00000724U)
#define LWC5B7_SET_IO_RC_PROCESS_OFFSET                                         31:0
#define LWC5B7_SET_IO_RC_PROCESS_OFFSET_HIGH_FIELD                              31U
#define LWC5B7_SET_IO_RC_PROCESS_OFFSET_LOW_FIELD                               0U
#define LWC5B7_SET_IN_COLOC_DATA                                                (0x00000728U)
#define LWC5B7_SET_IN_COLOC_DATA_OFFSET                                         31:0
#define LWC5B7_SET_IN_COLOC_DATA_OFFSET_HIGH_FIELD                              31U
#define LWC5B7_SET_IN_COLOC_DATA_OFFSET_LOW_FIELD                               0U
#define LWC5B7_SET_OUT_COLOC_DATA                                               (0x0000072LW)
#define LWC5B7_SET_OUT_COLOC_DATA_OFFSET                                        31:0
#define LWC5B7_SET_OUT_COLOC_DATA_OFFSET_HIGH_FIELD                             31U
#define LWC5B7_SET_OUT_COLOC_DATA_OFFSET_LOW_FIELD                              0U
#define LWC5B7_SET_OUT_REF_PIC_LUMA                                             (0x00000730U)
#define LWC5B7_SET_OUT_REF_PIC_LUMA_OFFSET                                      31:0
#define LWC5B7_SET_OUT_REF_PIC_LUMA_OFFSET_HIGH_FIELD                           31U
#define LWC5B7_SET_OUT_REF_PIC_LUMA_OFFSET_LOW_FIELD                            0U
#define LWC5B7_SET_IN_LWR_PIC                                                   (0x00000734U)
#define LWC5B7_SET_IN_LWR_PIC_OFFSET                                            31:0
#define LWC5B7_SET_IN_LWR_PIC_OFFSET_HIGH_FIELD                                 31U
#define LWC5B7_SET_IN_LWR_PIC_OFFSET_LOW_FIELD                                  0U
#define LWC5B7_SET_IN_MEPRED_DATA                                               (0x00000738U)
#define LWC5B7_SET_IN_MEPRED_DATA_OFFSET                                        31:0
#define LWC5B7_SET_IN_MEPRED_DATA_OFFSET_HIGH_FIELD                             31U
#define LWC5B7_SET_IN_MEPRED_DATA_OFFSET_LOW_FIELD                              0U
#define LWC5B7_SET_OUT_MEPRED_DATA                                              (0x0000073LW)
#define LWC5B7_SET_OUT_MEPRED_DATA_OFFSET                                       31:0
#define LWC5B7_SET_OUT_MEPRED_DATA_OFFSET_HIGH_FIELD                            31U
#define LWC5B7_SET_OUT_MEPRED_DATA_OFFSET_LOW_FIELD                             0U
#define LWC5B7_SET_IN_LWR_PIC_CHROMA_U                                          (0x00000740U)
#define LWC5B7_SET_IN_LWR_PIC_CHROMA_U_OFFSET                                   31:0
#define LWC5B7_SET_IN_LWR_PIC_CHROMA_U_OFFSET_HIGH_FIELD                        31U
#define LWC5B7_SET_IN_LWR_PIC_CHROMA_U_OFFSET_LOW_FIELD                         0U
#define LWC5B7_SET_IN_LWR_PIC_CHROMA_V                                          (0x00000744U)
#define LWC5B7_SET_IN_LWR_PIC_CHROMA_V_OFFSET                                   31:0
#define LWC5B7_SET_IN_LWR_PIC_CHROMA_V_OFFSET_HIGH_FIELD                        31U
#define LWC5B7_SET_IN_LWR_PIC_CHROMA_V_OFFSET_LOW_FIELD                         0U
#define LWC5B7_SET_IN_QP_MAP                                                    (0x00000748U)
#define LWC5B7_SET_IN_QP_MAP_OFFSET                                             31:0
#define LWC5B7_SET_IN_QP_MAP_OFFSET_HIGH_FIELD                                  31U
#define LWC5B7_SET_IN_QP_MAP_OFFSET_LOW_FIELD                                   0U
#define LWC5B7_SET_OUT_REF_PIC_CHROMA                                           (0x0000074LW)
#define LWC5B7_SET_OUT_REF_PIC_CHROMA_OFFSET                                    31:0
#define LWC5B7_SET_OUT_REF_PIC_CHROMA_OFFSET_HIGH_FIELD                         31U
#define LWC5B7_SET_OUT_REF_PIC_CHROMA_OFFSET_LOW_FIELD                          0U
#define LWC5B7_SET_IN_PARTITION_BUF                                             (0x00000750U)
#define LWC5B7_SET_IN_PARTITION_BUF_OFFSET                                      31:0
#define LWC5B7_SET_IN_PARTITION_BUF_OFFSET_HIGH_FIELD                           31U
#define LWC5B7_SET_IN_PARTITION_BUF_OFFSET_LOW_FIELD                            0U
#define LWC5B7_SET_IN_LWR_PIC1_TASK_STATUS                                      (0x00000758U)
#define LWC5B7_SET_IN_LWR_PIC1_TASK_STATUS_OFFSET                               31:0
#define LWC5B7_SET_IN_LWR_PIC1_TASK_STATUS_OFFSET_HIGH_FIELD                    31U
#define LWC5B7_SET_IN_LWR_PIC1_TASK_STATUS_OFFSET_LOW_FIELD                     0U
#define LWC5B7_SET_IN_LWR_PIC2_TASK_STATUS                                      (0x0000075LW)
#define LWC5B7_SET_IN_LWR_PIC2_TASK_STATUS_OFFSET                               31:0
#define LWC5B7_SET_IN_LWR_PIC2_TASK_STATUS_OFFSET_HIGH_FIELD                    31U
#define LWC5B7_SET_IN_LWR_PIC2_TASK_STATUS_OFFSET_LOW_FIELD                     0U
#define LWC5B7_SET_OUT_TASK_STATUS                                              (0x00000760U)
#define LWC5B7_SET_OUT_TASK_STATUS_OFFSET                                       31:0
#define LWC5B7_SET_OUT_TASK_STATUS_OFFSET_HIGH_FIELD                            31U
#define LWC5B7_SET_OUT_TASK_STATUS_OFFSET_LOW_FIELD                             0U
#define LWC5B7_SET_IN_MV_HINTS_TASK_STATUS                                      (0x00000764U)
#define LWC5B7_SET_IN_MV_HINTS_TASK_STATUS_OFFSET                               31:0
#define LWC5B7_SET_IN_MV_HINTS_TASK_STATUS_OFFSET_HIGH_FIELD                    31U
#define LWC5B7_SET_IN_MV_HINTS_TASK_STATUS_OFFSET_LOW_FIELD                     0U
#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
#define LWC5B7_PM_TRIGGER_END                                                   (0x00001114U)
#define LWC5B7_PM_TRIGGER_END_V                                                 31:0
#define LWC5B7_PM_TRIGGER_END_V_HIGH_FIELD                                      31U
#define LWC5B7_PM_TRIGGER_END_V_LOW_FIELD                                       0U
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#define LWC5B7_ERROR_NONE                                                       (0x00000000U)
#define LWC5B7_OS_ERROR_EXELWTE_INSUFFICIENT_DATA                               (0x00000001U)
#define LWC5B7_OS_ERROR_SEMAPHORE_INSUFFICIENT_DATA                             (0x00000002U)
#define LWC5B7_OS_ERROR_ILWALID_METHOD                                          (0x00000003U)
#define LWC5B7_OS_ERROR_ILWALID_DMA_PAGE                                        (0x00000004U)
#define LWC5B7_OS_ERROR_UNHANDLED_INTERRUPT                                     (0x00000005U)
#define LWC5B7_OS_ERROR_EXCEPTION                                               (0x00000006U)
#define LWC5B7_OS_ERROR_ILWALID_CTXSW_REQUEST                                   (0x00000007U)
#define LWC5B7_OS_ERROR_APPLICATION                                             (0x00000008U)
#define LWC5B7_OS_INTERRUPT_EXELWTE_AWAKEN                                      (0x00000100U)
#define LWC5B7_OS_INTERRUPT_BACKEND_SEMAPHORE_AWAKEN                            (0x00000200U)
#define LWC5B7_OS_INTERRUPT_CTX_ERROR_FBIF                                      (0x00000300U)
#define LWC5B7_OS_INTERRUPT_LIMIT_VIOLATION                                     (0x00000400U)
#define LWC5B7_OS_INTERRUPT_LIMIT_AND_FBIF_CTX_ERROR                            (0x00000500U)
#define LWC5B7_OS_INTERRUPT_HALT_ENGINE                                         (0x00000600U)
#define LWC5B7_OS_INTERRUPT_TRAP_NONSTALL                                       (0x00000700U)
#define LWC5B7_OS_INTERRUPT_CTX_SAVE_DONE                                       (0x00000800U)
#define LWC5B7_OS_INTERRUPT_CTX_RESTORE_DONE                                    (0x00000900U)
#define LWC5B7_ENC_ERROR_H264_APPTIMER_EXPIRED                                  (0x30000001U)
#define LWC5B7_ENC_ERROR_H264_ILWALID_INPUT                                     (0x30000002U)
#define LWC5B7_ENC_ERROR_H264_HWERR_INTERRUPT                                   (0x30000003U)
#define LWC5B7_ENC_ERROR_H264_BAD_MAGIC                                         (0x30000004U)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // clc5b7_h

