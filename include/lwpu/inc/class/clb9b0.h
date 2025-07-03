// WARNING!!! THIS HEADER INCLUDES SOFTWARE METHODS!!!
// ********** DO NOT USE IN HW TREE.  **********
/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 1993-2014 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


#ifndef clb9b0_h_
#define clb9b0_h_

#include "lwtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LWB9B0_VIDEO_DECODER                                                       (0x0000B9B0U)

typedef volatile struct _clb9b0_tag0 {
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
    LwV32 SetSemaphorePayloadLower;                                             // 0x00000254 - 0x00000257
    LwV32 SetSemaphorePayloadUpper;                                             // 0x00000258 - 0x0000025B
    LwV32 SetMonitoredFenceSignalAddressBaseA;                                  // 0x0000025C - 0x0000025F
    LwV32 SetMonitoredFenceSignalAddressBaseB;                                  // 0x00000260 - 0x00000263
    LwV32 Reserved04[0x27];
    LwV32 Execute;                                                              // 0x00000300 - 0x00000303
    LwV32 SemaphoreD;                                                           // 0x00000304 - 0x00000307
    LwV32 SetPredicationOffsetUpper;                                            // 0x00000308 - 0x0000030B
    LwV32 SetPredicationOffsetLower;                                            // 0x0000030C - 0x0000030F
    LwV32 SetAuxiliaryDataBuffer;                                               // 0x00000310 - 0x00000313
    LwV32 Reserved05[0x3B];
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
    LwV32 SetSubSampleMapOffset;                                                // 0x000004C0 - 0x000004C3
    LwV32 SetSubSampleMapIvOffset;                                              // 0x000004C4 - 0x000004C7
    LwV32 SetIntraTopBufOffset;                                                 // 0x000004C8 - 0x000004CB
    LwV32 SetTileSizeBufOffset;                                                 // 0x000004CC - 0x000004CF
    LwV32 SetFilterBufferOffset;                                                // 0x000004D0 - 0x000004D3
    LwV32 SetCrcStructOffset;                                                   // 0x000004D4 - 0x000004D7
    LwV32 Reserved06[0xA];
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
    LwV32 Reserved10[0x8];
    LwV32 Pass1SetClearHeaderOffset;                                            // 0x00000600 - 0x00000603
    LwV32 Pass1SetReEncryptOffset;                                              // 0x00000604 - 0x00000607
    LwV32 Pass1SetVP8TokenOffset;                                               // 0x00000608 - 0x0000060B
    LwV32 Pass1SetInputDataOffset;                                              // 0x0000060C - 0x0000060F
    LwV32 Pass1SetOutputDataSizeOffset;                                         // 0x00000610 - 0x00000613
    LwV32 Reserved11[0xB];
    LwV32 AV1SetProbTabReadBufOffset;                                           // 0x00000640 - 0x00000643
    LwV32 AV1SetProbTabWriteBufOffset;                                          // 0x00000644 - 0x00000647
    LwV32 AV1SetSegmentReadBufOffset;                                           // 0x00000648 - 0x0000064B
    LwV32 AV1SetSegmentWriteBufOffset;                                          // 0x0000064C - 0x0000064F
    LwV32 AV1SetColMV0ReadBufOffset;                                            // 0x00000650 - 0x00000653
    LwV32 AV1SetColMV1ReadBufOffset;                                            // 0x00000654 - 0x00000657
    LwV32 AV1SetColMV2ReadBufOffset;                                            // 0x00000658 - 0x0000065B
    LwV32 AV1SetColMVWriteBufOffset;                                            // 0x0000065C - 0x0000065F
    LwV32 AV1SetGlobalModelBufOffset;                                           // 0x00000660 - 0x00000663
    LwV32 AV1SetFilmGrainBufOffset;                                             // 0x00000664 - 0x00000667
    LwV32 AV1SetTileStreamInfoBufOffset;                                        // 0x00000668 - 0x0000066B
    LwV32 AV1SetSubStreamEntryBufOffset;                                        // 0x0000066C - 0x0000066F
    LwV32 Reserved12[0x4];
    LwV32 H264SetScalingListOffset;                                             // 0x00000680 - 0x00000683
    LwV32 H264SetVLDHistBufOffset;                                              // 0x00000684 - 0x00000687
    LwV32 H264SetEDOBOffset0;                                                   // 0x00000688 - 0x0000068B
    LwV32 H264SetEDOBOffset1;                                                   // 0x0000068C - 0x0000068F
    LwV32 H264SetEDOBOffset2;                                                   // 0x00000690 - 0x00000693
    LwV32 H264SetEDOBOffset3;                                                   // 0x00000694 - 0x00000697
    LwV32 Reserved13[0x15A];
    LwV32 SetContentInitialVector[4];                                           // 0x00000C00 - 0x00000C0F
    LwV32 SetCtlCount;                                                          // 0x00000C10 - 0x00000C13
    LwV32 SetUpperSrc;                                                          // 0x00000C14 - 0x00000C17
    LwV32 SetLowerSrc;                                                          // 0x00000C18 - 0x00000C1B
    LwV32 SetUpperDst;                                                          // 0x00000C1C - 0x00000C1F
    LwV32 SetLowerDst;                                                          // 0x00000C20 - 0x00000C23
    LwV32 SetBlockCount;                                                        // 0x00000C24 - 0x00000C27
    LwV32 Reserved14[0x36];
    LwV32 PrSetRequestBuffer;                                                   // 0x00000D00 - 0x00000D03
    LwV32 PrSetRequestBufferSize;                                               // 0x00000D04 - 0x00000D07
    LwV32 PrSetResponseBuffer;                                                  // 0x00000D08 - 0x00000D0B
    LwV32 PrSetResponseBufferSize;                                              // 0x00000D0C - 0x00000D0F
    LwV32 PrSetRequestMessageBuffer;                                            // 0x00000D10 - 0x00000D13
    LwV32 PrSetResponseMessageBuffer;                                           // 0x00000D14 - 0x00000D17
    LwV32 PrSetLocalDecryptBuffer;                                              // 0x00000D18 - 0x00000D1B
    LwV32 PrSetLocalDecryptBufferSize;                                          // 0x00000D1C - 0x00000D1F
    LwV32 PrSetContentDecryptInfoBuffer;                                        // 0x00000D20 - 0x00000D23
    LwV32 PrSetReencryptedBitstreamSurface;                                     // 0x00000D24 - 0x00000D27
    LwV32 Reserved15[0x76];
    LwV32 SetSessionKey[4];                                                     // 0x00000F00 - 0x00000F0F
    LwV32 SetContentKey[4];                                                     // 0x00000F10 - 0x00000F1F
    LwV32 Reserved16[0x7D];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved17[0x3BA];
} LWB9B0_VIDEO_DECODERControlPio;

#define LWB9B0_NOP                                                              (0x00000100U)
#define LWB9B0_NOP_PARAMETER                                                    31:0
#define LWB9B0_NOP_PARAMETER_HIGH_FIELD                                         31U
#define LWB9B0_NOP_PARAMETER_LOW_FIELD                                          0U
#define LWB9B0_PM_TRIGGER                                                       (0x00000140U)
#define LWB9B0_PM_TRIGGER_V                                                     31:0
#define LWB9B0_PM_TRIGGER_V_HIGH_FIELD                                          31U
#define LWB9B0_PM_TRIGGER_V_LOW_FIELD                                           0U
#define LWB9B0_SET_APPLICATION_ID                                               (0x00000200U)
#define LWB9B0_SET_APPLICATION_ID_ID                                            31:0
#define LWB9B0_SET_APPLICATION_ID_ID_HIGH_FIELD                                 31U
#define LWB9B0_SET_APPLICATION_ID_ID_LOW_FIELD                                  0U
#define LWB9B0_SET_APPLICATION_ID_ID_MPEG12                                     (0x00000001U)
#define LWB9B0_SET_APPLICATION_ID_ID_VC1                                        (0x00000002U)
#define LWB9B0_SET_APPLICATION_ID_ID_H264                                       (0x00000003U)
#define LWB9B0_SET_APPLICATION_ID_ID_MPEG4                                      (0x00000004U)
#define LWB9B0_SET_APPLICATION_ID_ID_VP8                                        (0x00000005U)
#define LWB9B0_SET_APPLICATION_ID_ID_CTR64                                      (0x00000006U)
#define LWB9B0_SET_APPLICATION_ID_ID_HEVC                                       (0x00000007U)
#define LWB9B0_SET_APPLICATION_ID_ID_NEW_H264                                   (0x00000008U)
#define LWB9B0_SET_APPLICATION_ID_ID_VP9                                        (0x00000009U)
#define LWB9B0_SET_APPLICATION_ID_ID_PASS1                                      (0x0000000AU)
#define LWB9B0_SET_APPLICATION_ID_ID_HEVC_PARSER                                (0x0000000LW)
#define LWB9B0_SET_APPLICATION_ID_ID_UCODE_TEST                                 (0x0000000DU)
#define LWB9B0_SET_APPLICATION_ID_ID_HWDRM_PR_DECRYPTAUDIO                      (0x0000000EU)
#define LWB9B0_SET_APPLICATION_ID_ID_HWDRM_PR_DECRYPTAUDIOMULTIPLE              (0x0000000FU)
#define LWB9B0_SET_APPLICATION_ID_ID_HWDRM_PR_PREPROCESSENCRYPTEDDATA           (0x00000010U)
#define LWB9B0_SET_APPLICATION_ID_ID_VP9_PARSER                                 (0x00000011U)
#define LWB9B0_SET_APPLICATION_ID_ID_AVD                                        (0x00000012U)
#define LWB9B0_SET_WATCHDOG_TIMER                                               (0x00000204U)
#define LWB9B0_SET_WATCHDOG_TIMER_TIMER                                         31:0
#define LWB9B0_SET_WATCHDOG_TIMER_TIMER_HIGH_FIELD                              31U
#define LWB9B0_SET_WATCHDOG_TIMER_TIMER_LOW_FIELD                               0U
#define LWB9B0_SEMAPHORE_A                                                      (0x00000240U)
#define LWB9B0_SEMAPHORE_A_UPPER                                                7:0
#define LWB9B0_SEMAPHORE_A_UPPER_HIGH_FIELD                                     7U
#define LWB9B0_SEMAPHORE_A_UPPER_LOW_FIELD                                      0U
#define LWB9B0_SEMAPHORE_B                                                      (0x00000244U)
#define LWB9B0_SEMAPHORE_B_LOWER                                                31:0
#define LWB9B0_SEMAPHORE_B_LOWER_HIGH_FIELD                                     31U
#define LWB9B0_SEMAPHORE_B_LOWER_LOW_FIELD                                      0U
#define LWB9B0_SEMAPHORE_C                                                      (0x00000248U)
#define LWB9B0_SEMAPHORE_C_PAYLOAD                                              31:0
#define LWB9B0_SEMAPHORE_C_PAYLOAD_HIGH_FIELD                                   31U
#define LWB9B0_SEMAPHORE_C_PAYLOAD_LOW_FIELD                                    0U
#define LWB9B0_CTX_SAVE_AREA                                                    (0x0000024LW)
#define LWB9B0_CTX_SAVE_AREA_OFFSET                                             31:0
#define LWB9B0_CTX_SAVE_AREA_OFFSET_HIGH_FIELD                                  31U
#define LWB9B0_CTX_SAVE_AREA_OFFSET_LOW_FIELD                                   0U
#define LWB9B0_CTX_SWITCH                                                       (0x00000250U)
#define LWB9B0_CTX_SWITCH_OP                                                    1:0
#define LWB9B0_CTX_SWITCH_OP_HIGH_FIELD                                         1U
#define LWB9B0_CTX_SWITCH_OP_LOW_FIELD                                          0U
#define LWB9B0_CTX_SWITCH_OP_CTX_UPDATE                                         (0x00000000U)
#define LWB9B0_CTX_SWITCH_OP_CTX_SAVE                                           (0x00000001U)
#define LWB9B0_CTX_SWITCH_OP_CTX_RESTORE                                        (0x00000002U)
#define LWB9B0_CTX_SWITCH_OP_CTX_FORCERESTORE                                   (0x00000003U)
#define LWB9B0_CTX_SWITCH_CTXID_VALID                                           2:2
#define LWB9B0_CTX_SWITCH_CTXID_VALID_HIGH_FIELD                                2U
#define LWB9B0_CTX_SWITCH_CTXID_VALID_LOW_FIELD                                 2U
#define LWB9B0_CTX_SWITCH_CTXID_VALID_FALSE                                     (0x00000000U)
#define LWB9B0_CTX_SWITCH_CTXID_VALID_TRUE                                      (0x00000001U)
#define LWB9B0_CTX_SWITCH_RESERVED0                                             7:3
#define LWB9B0_CTX_SWITCH_RESERVED0_HIGH_FIELD                                  7U
#define LWB9B0_CTX_SWITCH_RESERVED0_LOW_FIELD                                   3U
#define LWB9B0_CTX_SWITCH_CTX_ID                                                23:8
#define LWB9B0_CTX_SWITCH_CTX_ID_HIGH_FIELD                                     23U
#define LWB9B0_CTX_SWITCH_CTX_ID_LOW_FIELD                                      8U
#define LWB9B0_CTX_SWITCH_RESERVED1                                             31:24
#define LWB9B0_CTX_SWITCH_RESERVED1_HIGH_FIELD                                  31U
#define LWB9B0_CTX_SWITCH_RESERVED1_LOW_FIELD                                   24U
#define LWB9B0_SET_SEMAPHORE_PAYLOAD_LOWER                                      (0x00000254U)
#define LWB9B0_SET_SEMAPHORE_PAYLOAD_LOWER_PAYLOAD_LOWER                        31:0
#define LWB9B0_SET_SEMAPHORE_PAYLOAD_LOWER_PAYLOAD_LOWER_HIGH_FIELD             31U
#define LWB9B0_SET_SEMAPHORE_PAYLOAD_LOWER_PAYLOAD_LOWER_LOW_FIELD              0U
#define LWB9B0_SET_SEMAPHORE_PAYLOAD_UPPER                                      (0x00000258U)
#define LWB9B0_SET_SEMAPHORE_PAYLOAD_UPPER_PAYLOAD_UPPER                        31:0
#define LWB9B0_SET_SEMAPHORE_PAYLOAD_UPPER_PAYLOAD_UPPER_HIGH_FIELD             31U
#define LWB9B0_SET_SEMAPHORE_PAYLOAD_UPPER_PAYLOAD_UPPER_LOW_FIELD              0U
#define LWB9B0_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_A                        (0x0000025LW)
#define LWB9B0_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_A_LOWER                  31:0
#define LWB9B0_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_A_LOWER_HIGH_FIELD       31U
#define LWB9B0_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_A_LOWER_LOW_FIELD        0U
#define LWB9B0_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_B                        (0x00000260U)
#define LWB9B0_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_B_UPPER                  31:0
#define LWB9B0_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_B_UPPER_HIGH_FIELD       31U
#define LWB9B0_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_B_UPPER_LOW_FIELD        0U
#define LWB9B0_EXELWTE                                                          (0x00000300U)
#define LWB9B0_EXELWTE_NOTIFY                                                   0:0
#define LWB9B0_EXELWTE_NOTIFY_HIGH_FIELD                                        0U
#define LWB9B0_EXELWTE_NOTIFY_LOW_FIELD                                         0U
#define LWB9B0_EXELWTE_NOTIFY_DISABLE                                           (0x00000000U)
#define LWB9B0_EXELWTE_NOTIFY_ENABLE                                            (0x00000001U)
#define LWB9B0_EXELWTE_NOTIFY_ON                                                1:1
#define LWB9B0_EXELWTE_NOTIFY_ON_HIGH_FIELD                                     1U
#define LWB9B0_EXELWTE_NOTIFY_ON_LOW_FIELD                                      1U
#define LWB9B0_EXELWTE_NOTIFY_ON_END                                            (0x00000000U)
#define LWB9B0_EXELWTE_NOTIFY_ON_BEGIN                                          (0x00000001U)
#define LWB9B0_EXELWTE_PREDICATION                                              2:2
#define LWB9B0_EXELWTE_PREDICATION_HIGH_FIELD                                   2U
#define LWB9B0_EXELWTE_PREDICATION_LOW_FIELD                                    2U
#define LWB9B0_EXELWTE_PREDICATION_DISABLE                                      (0x00000000U)
#define LWB9B0_EXELWTE_PREDICATION_ENABLE                                       (0x00000001U)
#define LWB9B0_EXELWTE_PREDICATION_OP                                           3:3
#define LWB9B0_EXELWTE_PREDICATION_OP_HIGH_FIELD                                3U
#define LWB9B0_EXELWTE_PREDICATION_OP_LOW_FIELD                                 3U
#define LWB9B0_EXELWTE_PREDICATION_OP_EQUAL_ZERO                                (0x00000000U)
#define LWB9B0_EXELWTE_PREDICATION_OP_NOT_EQUAL_ZERO                            (0x00000001U)
#define LWB9B0_EXELWTE_AWAKEN                                                   8:8
#define LWB9B0_EXELWTE_AWAKEN_HIGH_FIELD                                        8U
#define LWB9B0_EXELWTE_AWAKEN_LOW_FIELD                                         8U
#define LWB9B0_EXELWTE_AWAKEN_DISABLE                                           (0x00000000U)
#define LWB9B0_EXELWTE_AWAKEN_ENABLE                                            (0x00000001U)
#define LWB9B0_SEMAPHORE_D                                                      (0x00000304U)
#define LWB9B0_SEMAPHORE_D_STRUCTURE_SIZE                                       1:0
#define LWB9B0_SEMAPHORE_D_STRUCTURE_SIZE_HIGH_FIELD                            1U
#define LWB9B0_SEMAPHORE_D_STRUCTURE_SIZE_LOW_FIELD                             0U
#define LWB9B0_SEMAPHORE_D_STRUCTURE_SIZE_ONE                                   (0x00000000U)
#define LWB9B0_SEMAPHORE_D_STRUCTURE_SIZE_FOUR                                  (0x00000001U)
#define LWB9B0_SEMAPHORE_D_STRUCTURE_SIZE_TWO                                   (0x00000002U)
#define LWB9B0_SEMAPHORE_D_AWAKEN_ENABLE                                        8:8
#define LWB9B0_SEMAPHORE_D_AWAKEN_ENABLE_HIGH_FIELD                             8U
#define LWB9B0_SEMAPHORE_D_AWAKEN_ENABLE_LOW_FIELD                              8U
#define LWB9B0_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                  (0x00000000U)
#define LWB9B0_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                   (0x00000001U)
#define LWB9B0_SEMAPHORE_D_OPERATION                                            17:16
#define LWB9B0_SEMAPHORE_D_OPERATION_HIGH_FIELD                                 17U
#define LWB9B0_SEMAPHORE_D_OPERATION_LOW_FIELD                                  16U
#define LWB9B0_SEMAPHORE_D_OPERATION_RELEASE                                    (0x00000000U)
#define LWB9B0_SEMAPHORE_D_OPERATION_RESERVED_0                                 (0x00000001U)
#define LWB9B0_SEMAPHORE_D_OPERATION_RESERVED_1                                 (0x00000002U)
#define LWB9B0_SEMAPHORE_D_OPERATION_TRAP                                       (0x00000003U)
#define LWB9B0_SEMAPHORE_D_FLUSH_DISABLE                                        21:21
#define LWB9B0_SEMAPHORE_D_FLUSH_DISABLE_HIGH_FIELD                             21U
#define LWB9B0_SEMAPHORE_D_FLUSH_DISABLE_LOW_FIELD                              21U
#define LWB9B0_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                  (0x00000000U)
#define LWB9B0_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                   (0x00000001U)
#define LWB9B0_SEMAPHORE_D_TRAP_TYPE                                            23:22
#define LWB9B0_SEMAPHORE_D_TRAP_TYPE_HIGH_FIELD                                 23U
#define LWB9B0_SEMAPHORE_D_TRAP_TYPE_LOW_FIELD                                  22U
#define LWB9B0_SEMAPHORE_D_TRAP_TYPE_UNCONDITIONAL                              (0x00000000U)
#define LWB9B0_SEMAPHORE_D_TRAP_TYPE_CONDITIONAL                                (0x00000001U)
#define LWB9B0_SEMAPHORE_D_TRAP_TYPE_CONDITIONAL_EXT                            (0x00000002U)
#define LWB9B0_SEMAPHORE_D_PAYLOAD_SIZE                                         24:24
#define LWB9B0_SEMAPHORE_D_PAYLOAD_SIZE_HIGH_FIELD                              24U
#define LWB9B0_SEMAPHORE_D_PAYLOAD_SIZE_LOW_FIELD                               24U
#define LWB9B0_SEMAPHORE_D_PAYLOAD_SIZE_32BIT                                   (0x00000000U)
#define LWB9B0_SEMAPHORE_D_PAYLOAD_SIZE_64BIT                                   (0x00000001U)
#define LWB9B0_SET_PREDICATION_OFFSET_UPPER                                     (0x00000308U)
#define LWB9B0_SET_PREDICATION_OFFSET_UPPER_OFFSET                              7:0
#define LWB9B0_SET_PREDICATION_OFFSET_UPPER_OFFSET_HIGH_FIELD                   7U
#define LWB9B0_SET_PREDICATION_OFFSET_UPPER_OFFSET_LOW_FIELD                    0U
#define LWB9B0_SET_PREDICATION_OFFSET_LOWER                                     (0x0000030LW)
#define LWB9B0_SET_PREDICATION_OFFSET_LOWER_OFFSET                              31:0
#define LWB9B0_SET_PREDICATION_OFFSET_LOWER_OFFSET_HIGH_FIELD                   31U
#define LWB9B0_SET_PREDICATION_OFFSET_LOWER_OFFSET_LOW_FIELD                    0U
#define LWB9B0_SET_AUXILIARY_DATA_BUFFER                                        (0x00000310U)
#define LWB9B0_SET_AUXILIARY_DATA_BUFFER_OFFSET                                 31:0
#define LWB9B0_SET_AUXILIARY_DATA_BUFFER_OFFSET_HIGH_FIELD                      31U
#define LWB9B0_SET_AUXILIARY_DATA_BUFFER_OFFSET_LOW_FIELD                       0U
#define LWB9B0_SET_CONTROL_PARAMS                                               (0x00000400U)
#define LWB9B0_SET_CONTROL_PARAMS_CODEC_TYPE                                    3:0
#define LWB9B0_SET_CONTROL_PARAMS_CODEC_TYPE_HIGH_FIELD                         3U
#define LWB9B0_SET_CONTROL_PARAMS_CODEC_TYPE_LOW_FIELD                          0U
#define LWB9B0_SET_CONTROL_PARAMS_CODEC_TYPE_MPEG1                              (0x00000000U)
#define LWB9B0_SET_CONTROL_PARAMS_CODEC_TYPE_MPEG2                              (0x00000001U)
#define LWB9B0_SET_CONTROL_PARAMS_CODEC_TYPE_VC1                                (0x00000002U)
#define LWB9B0_SET_CONTROL_PARAMS_CODEC_TYPE_H264                               (0x00000003U)
#define LWB9B0_SET_CONTROL_PARAMS_CODEC_TYPE_MPEG4                              (0x00000004U)
#define LWB9B0_SET_CONTROL_PARAMS_CODEC_TYPE_DIVX3                              (0x00000004U)
#define LWB9B0_SET_CONTROL_PARAMS_CODEC_TYPE_VP8                                (0x00000005U)
#define LWB9B0_SET_CONTROL_PARAMS_CODEC_TYPE_HEVC                               (0x00000007U)
#define LWB9B0_SET_CONTROL_PARAMS_CODEC_TYPE_VP9                                (0x00000009U)
#define LWB9B0_SET_CONTROL_PARAMS_CODEC_TYPE_AV1                                (0x0000000AU)
#define LWB9B0_SET_CONTROL_PARAMS_GPTIMER_ON                                    4:4
#define LWB9B0_SET_CONTROL_PARAMS_GPTIMER_ON_HIGH_FIELD                         4U
#define LWB9B0_SET_CONTROL_PARAMS_GPTIMER_ON_LOW_FIELD                          4U
#define LWB9B0_SET_CONTROL_PARAMS_RET_ERROR                                     5:5
#define LWB9B0_SET_CONTROL_PARAMS_RET_ERROR_HIGH_FIELD                          5U
#define LWB9B0_SET_CONTROL_PARAMS_RET_ERROR_LOW_FIELD                           5U
#define LWB9B0_SET_CONTROL_PARAMS_ERR_CONCEAL_ON                                6:6
#define LWB9B0_SET_CONTROL_PARAMS_ERR_CONCEAL_ON_HIGH_FIELD                     6U
#define LWB9B0_SET_CONTROL_PARAMS_ERR_CONCEAL_ON_LOW_FIELD                      6U
#define LWB9B0_SET_CONTROL_PARAMS_ERROR_FRM_IDX                                 12:7
#define LWB9B0_SET_CONTROL_PARAMS_ERROR_FRM_IDX_HIGH_FIELD                      12U
#define LWB9B0_SET_CONTROL_PARAMS_ERROR_FRM_IDX_LOW_FIELD                       7U
#define LWB9B0_SET_CONTROL_PARAMS_MBTIMER_ON                                    13:13
#define LWB9B0_SET_CONTROL_PARAMS_MBTIMER_ON_HIGH_FIELD                         13U
#define LWB9B0_SET_CONTROL_PARAMS_MBTIMER_ON_LOW_FIELD                          13U
#define LWB9B0_SET_CONTROL_PARAMS_EC_INTRA_FRAME_USING_PSLC                     14:14
#define LWB9B0_SET_CONTROL_PARAMS_EC_INTRA_FRAME_USING_PSLC_HIGH_FIELD          14U
#define LWB9B0_SET_CONTROL_PARAMS_EC_INTRA_FRAME_USING_PSLC_LOW_FIELD           14U
#define LWB9B0_SET_CONTROL_PARAMS_IGNORE_SOME_FIELDS_CRC_CHECK                  15:15
#define LWB9B0_SET_CONTROL_PARAMS_IGNORE_SOME_FIELDS_CRC_CHECK_HIGH_FIELD       15U
#define LWB9B0_SET_CONTROL_PARAMS_IGNORE_SOME_FIELDS_CRC_CHECK_LOW_FIELD        15U
#define LWB9B0_SET_CONTROL_PARAMS_EVENT_TRACE_LOGGING_ON                        16:16
#define LWB9B0_SET_CONTROL_PARAMS_EVENT_TRACE_LOGGING_ON_HIGH_FIELD             16U
#define LWB9B0_SET_CONTROL_PARAMS_EVENT_TRACE_LOGGING_ON_LOW_FIELD              16U
#define LWB9B0_SET_CONTROL_PARAMS_ALL_INTRA_FRAME                               17:17
#define LWB9B0_SET_CONTROL_PARAMS_ALL_INTRA_FRAME_HIGH_FIELD                    17U
#define LWB9B0_SET_CONTROL_PARAMS_ALL_INTRA_FRAME_LOW_FIELD                     17U
#define LWB9B0_SET_CONTROL_PARAMS_TESTRUN_ELW                                   19:18
#define LWB9B0_SET_CONTROL_PARAMS_TESTRUN_ELW_HIGH_FIELD                        19U
#define LWB9B0_SET_CONTROL_PARAMS_TESTRUN_ELW_LOW_FIELD                         18U
#define LWB9B0_SET_CONTROL_PARAMS_TESTRUN_ELW_TRACE3D_RUN                       (0x00000000U)
#define LWB9B0_SET_CONTROL_PARAMS_TESTRUN_ELW_PROD_RUN                          (0x00000001U)
#define LWB9B0_SET_CONTROL_PARAMS_RESERVED                                      25:20
#define LWB9B0_SET_CONTROL_PARAMS_RESERVED_HIGH_FIELD                           25U
#define LWB9B0_SET_CONTROL_PARAMS_RESERVED_LOW_FIELD                            20U
#define LWB9B0_SET_CONTROL_PARAMS_LWDECSIM_SKIP_SCP                             26:26
#define LWB9B0_SET_CONTROL_PARAMS_LWDECSIM_SKIP_SCP_HIGH_FIELD                  26U
#define LWB9B0_SET_CONTROL_PARAMS_LWDECSIM_SKIP_SCP_LOW_FIELD                   26U
#define LWB9B0_SET_CONTROL_PARAMS_ENABLE_ENCRYPT                                27:27
#define LWB9B0_SET_CONTROL_PARAMS_ENABLE_ENCRYPT_HIGH_FIELD                     27U
#define LWB9B0_SET_CONTROL_PARAMS_ENABLE_ENCRYPT_LOW_FIELD                      27U
#define LWB9B0_SET_CONTROL_PARAMS_ENCRYPTMODE                                   31:28
#define LWB9B0_SET_CONTROL_PARAMS_ENCRYPTMODE_HIGH_FIELD                        31U
#define LWB9B0_SET_CONTROL_PARAMS_ENCRYPTMODE_LOW_FIELD                         28U
#define LWB9B0_SET_DRV_PIC_SETUP_OFFSET                                         (0x00000404U)
#define LWB9B0_SET_DRV_PIC_SETUP_OFFSET_OFFSET                                  31:0
#define LWB9B0_SET_DRV_PIC_SETUP_OFFSET_OFFSET_HIGH_FIELD                       31U
#define LWB9B0_SET_DRV_PIC_SETUP_OFFSET_OFFSET_LOW_FIELD                        0U
#define LWB9B0_SET_IN_BUF_BASE_OFFSET                                           (0x00000408U)
#define LWB9B0_SET_IN_BUF_BASE_OFFSET_OFFSET                                    31:0
#define LWB9B0_SET_IN_BUF_BASE_OFFSET_OFFSET_HIGH_FIELD                         31U
#define LWB9B0_SET_IN_BUF_BASE_OFFSET_OFFSET_LOW_FIELD                          0U
#define LWB9B0_SET_PICTURE_INDEX                                                (0x0000040LW)
#define LWB9B0_SET_PICTURE_INDEX_INDEX                                          31:0
#define LWB9B0_SET_PICTURE_INDEX_INDEX_HIGH_FIELD                               31U
#define LWB9B0_SET_PICTURE_INDEX_INDEX_LOW_FIELD                                0U
#define LWB9B0_SET_SLICE_OFFSETS_BUF_OFFSET                                     (0x00000410U)
#define LWB9B0_SET_SLICE_OFFSETS_BUF_OFFSET_OFFSET                              31:0
#define LWB9B0_SET_SLICE_OFFSETS_BUF_OFFSET_OFFSET_HIGH_FIELD                   31U
#define LWB9B0_SET_SLICE_OFFSETS_BUF_OFFSET_OFFSET_LOW_FIELD                    0U
#define LWB9B0_SET_COLOC_DATA_OFFSET                                            (0x00000414U)
#define LWB9B0_SET_COLOC_DATA_OFFSET_OFFSET                                     31:0
#define LWB9B0_SET_COLOC_DATA_OFFSET_OFFSET_HIGH_FIELD                          31U
#define LWB9B0_SET_COLOC_DATA_OFFSET_OFFSET_LOW_FIELD                           0U
#define LWB9B0_SET_HISTORY_OFFSET                                               (0x00000418U)
#define LWB9B0_SET_HISTORY_OFFSET_OFFSET                                        31:0
#define LWB9B0_SET_HISTORY_OFFSET_OFFSET_HIGH_FIELD                             31U
#define LWB9B0_SET_HISTORY_OFFSET_OFFSET_LOW_FIELD                              0U
#define LWB9B0_SET_DISPLAY_BUF_SIZE                                             (0x0000041LW)
#define LWB9B0_SET_DISPLAY_BUF_SIZE_SIZE                                        31:0
#define LWB9B0_SET_DISPLAY_BUF_SIZE_SIZE_HIGH_FIELD                             31U
#define LWB9B0_SET_DISPLAY_BUF_SIZE_SIZE_LOW_FIELD                              0U
#define LWB9B0_SET_HISTOGRAM_OFFSET                                             (0x00000420U)
#define LWB9B0_SET_HISTOGRAM_OFFSET_OFFSET                                      31:0
#define LWB9B0_SET_HISTOGRAM_OFFSET_OFFSET_HIGH_FIELD                           31U
#define LWB9B0_SET_HISTOGRAM_OFFSET_OFFSET_LOW_FIELD                            0U
#define LWB9B0_SET_LWDEC_STATUS_OFFSET                                          (0x00000424U)
#define LWB9B0_SET_LWDEC_STATUS_OFFSET_OFFSET                                   31:0
#define LWB9B0_SET_LWDEC_STATUS_OFFSET_OFFSET_HIGH_FIELD                        31U
#define LWB9B0_SET_LWDEC_STATUS_OFFSET_OFFSET_LOW_FIELD                         0U
#define LWB9B0_SET_DISPLAY_BUF_LUMA_OFFSET                                      (0x00000428U)
#define LWB9B0_SET_DISPLAY_BUF_LUMA_OFFSET_OFFSET                               31:0
#define LWB9B0_SET_DISPLAY_BUF_LUMA_OFFSET_OFFSET_HIGH_FIELD                    31U
#define LWB9B0_SET_DISPLAY_BUF_LUMA_OFFSET_OFFSET_LOW_FIELD                     0U
#define LWB9B0_SET_DISPLAY_BUF_CHROMA_OFFSET                                    (0x0000042LW)
#define LWB9B0_SET_DISPLAY_BUF_CHROMA_OFFSET_OFFSET                             31:0
#define LWB9B0_SET_DISPLAY_BUF_CHROMA_OFFSET_OFFSET_HIGH_FIELD                  31U
#define LWB9B0_SET_DISPLAY_BUF_CHROMA_OFFSET_OFFSET_LOW_FIELD                   0U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET0                                         (0x00000430U)
#define LWB9B0_SET_PICTURE_LUMA_OFFSET0_OFFSET                                  31:0
#define LWB9B0_SET_PICTURE_LUMA_OFFSET0_OFFSET_HIGH_FIELD                       31U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET0_OFFSET_LOW_FIELD                        0U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET1                                         (0x00000434U)
#define LWB9B0_SET_PICTURE_LUMA_OFFSET1_OFFSET                                  31:0
#define LWB9B0_SET_PICTURE_LUMA_OFFSET1_OFFSET_HIGH_FIELD                       31U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET1_OFFSET_LOW_FIELD                        0U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET2                                         (0x00000438U)
#define LWB9B0_SET_PICTURE_LUMA_OFFSET2_OFFSET                                  31:0
#define LWB9B0_SET_PICTURE_LUMA_OFFSET2_OFFSET_HIGH_FIELD                       31U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET2_OFFSET_LOW_FIELD                        0U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET3                                         (0x0000043LW)
#define LWB9B0_SET_PICTURE_LUMA_OFFSET3_OFFSET                                  31:0
#define LWB9B0_SET_PICTURE_LUMA_OFFSET3_OFFSET_HIGH_FIELD                       31U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET3_OFFSET_LOW_FIELD                        0U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET4                                         (0x00000440U)
#define LWB9B0_SET_PICTURE_LUMA_OFFSET4_OFFSET                                  31:0
#define LWB9B0_SET_PICTURE_LUMA_OFFSET4_OFFSET_HIGH_FIELD                       31U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET4_OFFSET_LOW_FIELD                        0U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET5                                         (0x00000444U)
#define LWB9B0_SET_PICTURE_LUMA_OFFSET5_OFFSET                                  31:0
#define LWB9B0_SET_PICTURE_LUMA_OFFSET5_OFFSET_HIGH_FIELD                       31U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET5_OFFSET_LOW_FIELD                        0U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET6                                         (0x00000448U)
#define LWB9B0_SET_PICTURE_LUMA_OFFSET6_OFFSET                                  31:0
#define LWB9B0_SET_PICTURE_LUMA_OFFSET6_OFFSET_HIGH_FIELD                       31U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET6_OFFSET_LOW_FIELD                        0U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET7                                         (0x0000044LW)
#define LWB9B0_SET_PICTURE_LUMA_OFFSET7_OFFSET                                  31:0
#define LWB9B0_SET_PICTURE_LUMA_OFFSET7_OFFSET_HIGH_FIELD                       31U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET7_OFFSET_LOW_FIELD                        0U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET8                                         (0x00000450U)
#define LWB9B0_SET_PICTURE_LUMA_OFFSET8_OFFSET                                  31:0
#define LWB9B0_SET_PICTURE_LUMA_OFFSET8_OFFSET_HIGH_FIELD                       31U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET8_OFFSET_LOW_FIELD                        0U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET9                                         (0x00000454U)
#define LWB9B0_SET_PICTURE_LUMA_OFFSET9_OFFSET                                  31:0
#define LWB9B0_SET_PICTURE_LUMA_OFFSET9_OFFSET_HIGH_FIELD                       31U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET9_OFFSET_LOW_FIELD                        0U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET10                                        (0x00000458U)
#define LWB9B0_SET_PICTURE_LUMA_OFFSET10_OFFSET                                 31:0
#define LWB9B0_SET_PICTURE_LUMA_OFFSET10_OFFSET_HIGH_FIELD                      31U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET10_OFFSET_LOW_FIELD                       0U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET11                                        (0x0000045LW)
#define LWB9B0_SET_PICTURE_LUMA_OFFSET11_OFFSET                                 31:0
#define LWB9B0_SET_PICTURE_LUMA_OFFSET11_OFFSET_HIGH_FIELD                      31U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET11_OFFSET_LOW_FIELD                       0U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET12                                        (0x00000460U)
#define LWB9B0_SET_PICTURE_LUMA_OFFSET12_OFFSET                                 31:0
#define LWB9B0_SET_PICTURE_LUMA_OFFSET12_OFFSET_HIGH_FIELD                      31U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET12_OFFSET_LOW_FIELD                       0U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET13                                        (0x00000464U)
#define LWB9B0_SET_PICTURE_LUMA_OFFSET13_OFFSET                                 31:0
#define LWB9B0_SET_PICTURE_LUMA_OFFSET13_OFFSET_HIGH_FIELD                      31U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET13_OFFSET_LOW_FIELD                       0U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET14                                        (0x00000468U)
#define LWB9B0_SET_PICTURE_LUMA_OFFSET14_OFFSET                                 31:0
#define LWB9B0_SET_PICTURE_LUMA_OFFSET14_OFFSET_HIGH_FIELD                      31U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET14_OFFSET_LOW_FIELD                       0U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET15                                        (0x0000046LW)
#define LWB9B0_SET_PICTURE_LUMA_OFFSET15_OFFSET                                 31:0
#define LWB9B0_SET_PICTURE_LUMA_OFFSET15_OFFSET_HIGH_FIELD                      31U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET15_OFFSET_LOW_FIELD                       0U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET16                                        (0x00000470U)
#define LWB9B0_SET_PICTURE_LUMA_OFFSET16_OFFSET                                 31:0
#define LWB9B0_SET_PICTURE_LUMA_OFFSET16_OFFSET_HIGH_FIELD                      31U
#define LWB9B0_SET_PICTURE_LUMA_OFFSET16_OFFSET_LOW_FIELD                       0U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET0                                       (0x00000474U)
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET0_OFFSET                                31:0
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET0_OFFSET_HIGH_FIELD                     31U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET0_OFFSET_LOW_FIELD                      0U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET1                                       (0x00000478U)
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET1_OFFSET                                31:0
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET1_OFFSET_HIGH_FIELD                     31U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET1_OFFSET_LOW_FIELD                      0U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET2                                       (0x0000047LW)
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET2_OFFSET                                31:0
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET2_OFFSET_HIGH_FIELD                     31U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET2_OFFSET_LOW_FIELD                      0U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET3                                       (0x00000480U)
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET3_OFFSET                                31:0
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET3_OFFSET_HIGH_FIELD                     31U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET3_OFFSET_LOW_FIELD                      0U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET4                                       (0x00000484U)
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET4_OFFSET                                31:0
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET4_OFFSET_HIGH_FIELD                     31U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET4_OFFSET_LOW_FIELD                      0U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET5                                       (0x00000488U)
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET5_OFFSET                                31:0
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET5_OFFSET_HIGH_FIELD                     31U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET5_OFFSET_LOW_FIELD                      0U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET6                                       (0x0000048LW)
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET6_OFFSET                                31:0
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET6_OFFSET_HIGH_FIELD                     31U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET6_OFFSET_LOW_FIELD                      0U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET7                                       (0x00000490U)
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET7_OFFSET                                31:0
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET7_OFFSET_HIGH_FIELD                     31U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET7_OFFSET_LOW_FIELD                      0U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET8                                       (0x00000494U)
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET8_OFFSET                                31:0
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET8_OFFSET_HIGH_FIELD                     31U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET8_OFFSET_LOW_FIELD                      0U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET9                                       (0x00000498U)
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET9_OFFSET                                31:0
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET9_OFFSET_HIGH_FIELD                     31U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET9_OFFSET_LOW_FIELD                      0U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET10                                      (0x0000049LW)
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET10_OFFSET                               31:0
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET10_OFFSET_HIGH_FIELD                    31U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET10_OFFSET_LOW_FIELD                     0U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET11                                      (0x000004A0U)
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET11_OFFSET                               31:0
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET11_OFFSET_HIGH_FIELD                    31U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET11_OFFSET_LOW_FIELD                     0U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET12                                      (0x000004A4U)
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET12_OFFSET                               31:0
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET12_OFFSET_HIGH_FIELD                    31U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET12_OFFSET_LOW_FIELD                     0U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET13                                      (0x000004A8U)
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET13_OFFSET                               31:0
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET13_OFFSET_HIGH_FIELD                    31U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET13_OFFSET_LOW_FIELD                     0U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET14                                      (0x000004ALW)
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET14_OFFSET                               31:0
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET14_OFFSET_HIGH_FIELD                    31U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET14_OFFSET_LOW_FIELD                     0U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET15                                      (0x000004B0U)
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET15_OFFSET                               31:0
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET15_OFFSET_HIGH_FIELD                    31U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET15_OFFSET_LOW_FIELD                     0U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET16                                      (0x000004B4U)
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET16_OFFSET                               31:0
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET16_OFFSET_HIGH_FIELD                    31U
#define LWB9B0_SET_PICTURE_CHROMA_OFFSET16_OFFSET_LOW_FIELD                     0U
#define LWB9B0_SET_PIC_SCRATCH_BUF_OFFSET                                       (0x000004B8U)
#define LWB9B0_SET_PIC_SCRATCH_BUF_OFFSET_OFFSET                                31:0
#define LWB9B0_SET_PIC_SCRATCH_BUF_OFFSET_OFFSET_HIGH_FIELD                     31U
#define LWB9B0_SET_PIC_SCRATCH_BUF_OFFSET_OFFSET_LOW_FIELD                      0U
#define LWB9B0_SET_EXTERNAL_MVBUFFER_OFFSET                                     (0x000004BLW)
#define LWB9B0_SET_EXTERNAL_MVBUFFER_OFFSET_OFFSET                              31:0
#define LWB9B0_SET_EXTERNAL_MVBUFFER_OFFSET_OFFSET_HIGH_FIELD                   31U
#define LWB9B0_SET_EXTERNAL_MVBUFFER_OFFSET_OFFSET_LOW_FIELD                    0U
#define LWB9B0_SET_SUB_SAMPLE_MAP_OFFSET                                        (0x000004C0U)
#define LWB9B0_SET_SUB_SAMPLE_MAP_OFFSET_OFFSET                                 31:0
#define LWB9B0_SET_SUB_SAMPLE_MAP_OFFSET_OFFSET_HIGH_FIELD                      31U
#define LWB9B0_SET_SUB_SAMPLE_MAP_OFFSET_OFFSET_LOW_FIELD                       0U
#define LWB9B0_SET_SUB_SAMPLE_MAP_IV_OFFSET                                     (0x000004C4U)
#define LWB9B0_SET_SUB_SAMPLE_MAP_IV_OFFSET_OFFSET                              31:0
#define LWB9B0_SET_SUB_SAMPLE_MAP_IV_OFFSET_OFFSET_HIGH_FIELD                   31U
#define LWB9B0_SET_SUB_SAMPLE_MAP_IV_OFFSET_OFFSET_LOW_FIELD                    0U
#define LWB9B0_SET_INTRA_TOP_BUF_OFFSET                                         (0x000004C8U)
#define LWB9B0_SET_INTRA_TOP_BUF_OFFSET_OFFSET                                  31:0
#define LWB9B0_SET_INTRA_TOP_BUF_OFFSET_OFFSET_HIGH_FIELD                       31U
#define LWB9B0_SET_INTRA_TOP_BUF_OFFSET_OFFSET_LOW_FIELD                        0U
#define LWB9B0_SET_TILE_SIZE_BUF_OFFSET                                         (0x000004CLW)
#define LWB9B0_SET_TILE_SIZE_BUF_OFFSET_OFFSET                                  31:0
#define LWB9B0_SET_TILE_SIZE_BUF_OFFSET_OFFSET_HIGH_FIELD                       31U
#define LWB9B0_SET_TILE_SIZE_BUF_OFFSET_OFFSET_LOW_FIELD                        0U
#define LWB9B0_SET_FILTER_BUFFER_OFFSET                                         (0x000004D0U)
#define LWB9B0_SET_FILTER_BUFFER_OFFSET_OFFSET                                  31:0
#define LWB9B0_SET_FILTER_BUFFER_OFFSET_OFFSET_HIGH_FIELD                       31U
#define LWB9B0_SET_FILTER_BUFFER_OFFSET_OFFSET_LOW_FIELD                        0U
#define LWB9B0_SET_CRC_STRUCT_OFFSET                                            (0x000004D4U)
#define LWB9B0_SET_CRC_STRUCT_OFFSET_OFFSET                                     31:0
#define LWB9B0_SET_CRC_STRUCT_OFFSET_OFFSET_HIGH_FIELD                          31U
#define LWB9B0_SET_CRC_STRUCT_OFFSET_OFFSET_LOW_FIELD                           0U
#define LWB9B0_H264_SET_MBHIST_BUF_OFFSET                                       (0x00000500U)
#define LWB9B0_H264_SET_MBHIST_BUF_OFFSET_OFFSET                                31:0
#define LWB9B0_H264_SET_MBHIST_BUF_OFFSET_OFFSET_HIGH_FIELD                     31U
#define LWB9B0_H264_SET_MBHIST_BUF_OFFSET_OFFSET_LOW_FIELD                      0U
#define LWB9B0_VP8_SET_PROB_DATA_OFFSET                                         (0x00000540U)
#define LWB9B0_VP8_SET_PROB_DATA_OFFSET_OFFSET                                  31:0
#define LWB9B0_VP8_SET_PROB_DATA_OFFSET_OFFSET_HIGH_FIELD                       31U
#define LWB9B0_VP8_SET_PROB_DATA_OFFSET_OFFSET_LOW_FIELD                        0U
#define LWB9B0_VP8_SET_HEADER_PARTITION_BUF_BASE_OFFSET                         (0x00000544U)
#define LWB9B0_VP8_SET_HEADER_PARTITION_BUF_BASE_OFFSET_OFFSET                  31:0
#define LWB9B0_VP8_SET_HEADER_PARTITION_BUF_BASE_OFFSET_OFFSET_HIGH_FIELD       31U
#define LWB9B0_VP8_SET_HEADER_PARTITION_BUF_BASE_OFFSET_OFFSET_LOW_FIELD        0U
#define LWB9B0_HEVC_SET_SCALING_LIST_OFFSET                                     (0x00000580U)
#define LWB9B0_HEVC_SET_SCALING_LIST_OFFSET_OFFSET                              31:0
#define LWB9B0_HEVC_SET_SCALING_LIST_OFFSET_OFFSET_HIGH_FIELD                   31U
#define LWB9B0_HEVC_SET_SCALING_LIST_OFFSET_OFFSET_LOW_FIELD                    0U
#define LWB9B0_HEVC_SET_TILE_SIZES_OFFSET                                       (0x00000584U)
#define LWB9B0_HEVC_SET_TILE_SIZES_OFFSET_OFFSET                                31:0
#define LWB9B0_HEVC_SET_TILE_SIZES_OFFSET_OFFSET_HIGH_FIELD                     31U
#define LWB9B0_HEVC_SET_TILE_SIZES_OFFSET_OFFSET_LOW_FIELD                      0U
#define LWB9B0_HEVC_SET_FILTER_BUFFER_OFFSET                                    (0x00000588U)
#define LWB9B0_HEVC_SET_FILTER_BUFFER_OFFSET_OFFSET                             31:0
#define LWB9B0_HEVC_SET_FILTER_BUFFER_OFFSET_OFFSET_HIGH_FIELD                  31U
#define LWB9B0_HEVC_SET_FILTER_BUFFER_OFFSET_OFFSET_LOW_FIELD                   0U
#define LWB9B0_HEVC_SET_SAO_BUFFER_OFFSET                                       (0x0000058LW)
#define LWB9B0_HEVC_SET_SAO_BUFFER_OFFSET_OFFSET                                31:0
#define LWB9B0_HEVC_SET_SAO_BUFFER_OFFSET_OFFSET_HIGH_FIELD                     31U
#define LWB9B0_HEVC_SET_SAO_BUFFER_OFFSET_OFFSET_LOW_FIELD                      0U
#define LWB9B0_HEVC_SET_SLICE_INFO_BUFFER_OFFSET                                (0x00000590U)
#define LWB9B0_HEVC_SET_SLICE_INFO_BUFFER_OFFSET_OFFSET                         31:0
#define LWB9B0_HEVC_SET_SLICE_INFO_BUFFER_OFFSET_OFFSET_HIGH_FIELD              31U
#define LWB9B0_HEVC_SET_SLICE_INFO_BUFFER_OFFSET_OFFSET_LOW_FIELD               0U
#define LWB9B0_HEVC_SET_SLICE_GROUP_INDEX                                       (0x00000594U)
#define LWB9B0_HEVC_SET_SLICE_GROUP_INDEX_OFFSET                                31:0
#define LWB9B0_HEVC_SET_SLICE_GROUP_INDEX_OFFSET_HIGH_FIELD                     31U
#define LWB9B0_HEVC_SET_SLICE_GROUP_INDEX_OFFSET_LOW_FIELD                      0U
#define LWB9B0_VP9_SET_PROB_TAB_BUF_OFFSET                                      (0x000005C0U)
#define LWB9B0_VP9_SET_PROB_TAB_BUF_OFFSET_OFFSET                               31:0
#define LWB9B0_VP9_SET_PROB_TAB_BUF_OFFSET_OFFSET_HIGH_FIELD                    31U
#define LWB9B0_VP9_SET_PROB_TAB_BUF_OFFSET_OFFSET_LOW_FIELD                     0U
#define LWB9B0_VP9_SET_CTX_COUNTER_BUF_OFFSET                                   (0x000005C4U)
#define LWB9B0_VP9_SET_CTX_COUNTER_BUF_OFFSET_OFFSET                            31:0
#define LWB9B0_VP9_SET_CTX_COUNTER_BUF_OFFSET_OFFSET_HIGH_FIELD                 31U
#define LWB9B0_VP9_SET_CTX_COUNTER_BUF_OFFSET_OFFSET_LOW_FIELD                  0U
#define LWB9B0_VP9_SET_SEGMENT_READ_BUF_OFFSET                                  (0x000005C8U)
#define LWB9B0_VP9_SET_SEGMENT_READ_BUF_OFFSET_OFFSET                           31:0
#define LWB9B0_VP9_SET_SEGMENT_READ_BUF_OFFSET_OFFSET_HIGH_FIELD                31U
#define LWB9B0_VP9_SET_SEGMENT_READ_BUF_OFFSET_OFFSET_LOW_FIELD                 0U
#define LWB9B0_VP9_SET_SEGMENT_WRITE_BUF_OFFSET                                 (0x000005CLW)
#define LWB9B0_VP9_SET_SEGMENT_WRITE_BUF_OFFSET_OFFSET                          31:0
#define LWB9B0_VP9_SET_SEGMENT_WRITE_BUF_OFFSET_OFFSET_HIGH_FIELD               31U
#define LWB9B0_VP9_SET_SEGMENT_WRITE_BUF_OFFSET_OFFSET_LOW_FIELD                0U
#define LWB9B0_VP9_SET_TILE_SIZE_BUF_OFFSET                                     (0x000005D0U)
#define LWB9B0_VP9_SET_TILE_SIZE_BUF_OFFSET_OFFSET                              31:0
#define LWB9B0_VP9_SET_TILE_SIZE_BUF_OFFSET_OFFSET_HIGH_FIELD                   31U
#define LWB9B0_VP9_SET_TILE_SIZE_BUF_OFFSET_OFFSET_LOW_FIELD                    0U
#define LWB9B0_VP9_SET_COL_MVWRITE_BUF_OFFSET                                   (0x000005D4U)
#define LWB9B0_VP9_SET_COL_MVWRITE_BUF_OFFSET_OFFSET                            31:0
#define LWB9B0_VP9_SET_COL_MVWRITE_BUF_OFFSET_OFFSET_HIGH_FIELD                 31U
#define LWB9B0_VP9_SET_COL_MVWRITE_BUF_OFFSET_OFFSET_LOW_FIELD                  0U
#define LWB9B0_VP9_SET_COL_MVREAD_BUF_OFFSET                                    (0x000005D8U)
#define LWB9B0_VP9_SET_COL_MVREAD_BUF_OFFSET_OFFSET                             31:0
#define LWB9B0_VP9_SET_COL_MVREAD_BUF_OFFSET_OFFSET_HIGH_FIELD                  31U
#define LWB9B0_VP9_SET_COL_MVREAD_BUF_OFFSET_OFFSET_LOW_FIELD                   0U
#define LWB9B0_VP9_SET_FILTER_BUFFER_OFFSET                                     (0x000005DLW)
#define LWB9B0_VP9_SET_FILTER_BUFFER_OFFSET_OFFSET                              31:0
#define LWB9B0_VP9_SET_FILTER_BUFFER_OFFSET_OFFSET_HIGH_FIELD                   31U
#define LWB9B0_VP9_SET_FILTER_BUFFER_OFFSET_OFFSET_LOW_FIELD                    0U
#define LWB9B0_PASS1_SET_CLEAR_HEADER_OFFSET                                    (0x00000600U)
#define LWB9B0_PASS1_SET_CLEAR_HEADER_OFFSET_OFFSET                             31:0
#define LWB9B0_PASS1_SET_CLEAR_HEADER_OFFSET_OFFSET_HIGH_FIELD                  31U
#define LWB9B0_PASS1_SET_CLEAR_HEADER_OFFSET_OFFSET_LOW_FIELD                   0U
#define LWB9B0_PASS1_SET_RE_ENCRYPT_OFFSET                                      (0x00000604U)
#define LWB9B0_PASS1_SET_RE_ENCRYPT_OFFSET_OFFSET                               31:0
#define LWB9B0_PASS1_SET_RE_ENCRYPT_OFFSET_OFFSET_HIGH_FIELD                    31U
#define LWB9B0_PASS1_SET_RE_ENCRYPT_OFFSET_OFFSET_LOW_FIELD                     0U
#define LWB9B0_PASS1_SET_VP8_TOKEN_OFFSET                                       (0x00000608U)
#define LWB9B0_PASS1_SET_VP8_TOKEN_OFFSET_OFFSET                                31:0
#define LWB9B0_PASS1_SET_VP8_TOKEN_OFFSET_OFFSET_HIGH_FIELD                     31U
#define LWB9B0_PASS1_SET_VP8_TOKEN_OFFSET_OFFSET_LOW_FIELD                      0U
#define LWB9B0_PASS1_SET_INPUT_DATA_OFFSET                                      (0x0000060LW)
#define LWB9B0_PASS1_SET_INPUT_DATA_OFFSET_OFFSET                               31:0
#define LWB9B0_PASS1_SET_INPUT_DATA_OFFSET_OFFSET_HIGH_FIELD                    31U
#define LWB9B0_PASS1_SET_INPUT_DATA_OFFSET_OFFSET_LOW_FIELD                     0U
#define LWB9B0_PASS1_SET_OUTPUT_DATA_SIZE_OFFSET                                (0x00000610U)
#define LWB9B0_PASS1_SET_OUTPUT_DATA_SIZE_OFFSET_OFFSET                         31:0
#define LWB9B0_PASS1_SET_OUTPUT_DATA_SIZE_OFFSET_OFFSET_HIGH_FIELD              31U
#define LWB9B0_PASS1_SET_OUTPUT_DATA_SIZE_OFFSET_OFFSET_LOW_FIELD               0U
#define LWB9B0_AV1_SET_PROB_TAB_READ_BUF_OFFSET                                 (0x00000640U)
#define LWB9B0_AV1_SET_PROB_TAB_READ_BUF_OFFSET_OFFSET                          31:0
#define LWB9B0_AV1_SET_PROB_TAB_READ_BUF_OFFSET_OFFSET_HIGH_FIELD               31U
#define LWB9B0_AV1_SET_PROB_TAB_READ_BUF_OFFSET_OFFSET_LOW_FIELD                0U
#define LWB9B0_AV1_SET_PROB_TAB_WRITE_BUF_OFFSET                                (0x00000644U)
#define LWB9B0_AV1_SET_PROB_TAB_WRITE_BUF_OFFSET_OFFSET                         31:0
#define LWB9B0_AV1_SET_PROB_TAB_WRITE_BUF_OFFSET_OFFSET_HIGH_FIELD              31U
#define LWB9B0_AV1_SET_PROB_TAB_WRITE_BUF_OFFSET_OFFSET_LOW_FIELD               0U
#define LWB9B0_AV1_SET_SEGMENT_READ_BUF_OFFSET                                  (0x00000648U)
#define LWB9B0_AV1_SET_SEGMENT_READ_BUF_OFFSET_OFFSET                           31:0
#define LWB9B0_AV1_SET_SEGMENT_READ_BUF_OFFSET_OFFSET_HIGH_FIELD                31U
#define LWB9B0_AV1_SET_SEGMENT_READ_BUF_OFFSET_OFFSET_LOW_FIELD                 0U
#define LWB9B0_AV1_SET_SEGMENT_WRITE_BUF_OFFSET                                 (0x0000064LW)
#define LWB9B0_AV1_SET_SEGMENT_WRITE_BUF_OFFSET_OFFSET                          31:0
#define LWB9B0_AV1_SET_SEGMENT_WRITE_BUF_OFFSET_OFFSET_HIGH_FIELD               31U
#define LWB9B0_AV1_SET_SEGMENT_WRITE_BUF_OFFSET_OFFSET_LOW_FIELD                0U
#define LWB9B0_AV1_SET_COL_MV0_READ_BUF_OFFSET                                  (0x00000650U)
#define LWB9B0_AV1_SET_COL_MV0_READ_BUF_OFFSET_OFFSET                           31:0
#define LWB9B0_AV1_SET_COL_MV0_READ_BUF_OFFSET_OFFSET_HIGH_FIELD                31U
#define LWB9B0_AV1_SET_COL_MV0_READ_BUF_OFFSET_OFFSET_LOW_FIELD                 0U
#define LWB9B0_AV1_SET_COL_MV1_READ_BUF_OFFSET                                  (0x00000654U)
#define LWB9B0_AV1_SET_COL_MV1_READ_BUF_OFFSET_OFFSET                           31:0
#define LWB9B0_AV1_SET_COL_MV1_READ_BUF_OFFSET_OFFSET_HIGH_FIELD                31U
#define LWB9B0_AV1_SET_COL_MV1_READ_BUF_OFFSET_OFFSET_LOW_FIELD                 0U
#define LWB9B0_AV1_SET_COL_MV2_READ_BUF_OFFSET                                  (0x00000658U)
#define LWB9B0_AV1_SET_COL_MV2_READ_BUF_OFFSET_OFFSET                           31:0
#define LWB9B0_AV1_SET_COL_MV2_READ_BUF_OFFSET_OFFSET_HIGH_FIELD                31U
#define LWB9B0_AV1_SET_COL_MV2_READ_BUF_OFFSET_OFFSET_LOW_FIELD                 0U
#define LWB9B0_AV1_SET_COL_MVWRITE_BUF_OFFSET                                   (0x0000065LW)
#define LWB9B0_AV1_SET_COL_MVWRITE_BUF_OFFSET_OFFSET                            31:0
#define LWB9B0_AV1_SET_COL_MVWRITE_BUF_OFFSET_OFFSET_HIGH_FIELD                 31U
#define LWB9B0_AV1_SET_COL_MVWRITE_BUF_OFFSET_OFFSET_LOW_FIELD                  0U
#define LWB9B0_AV1_SET_GLOBAL_MODEL_BUF_OFFSET                                  (0x00000660U)
#define LWB9B0_AV1_SET_GLOBAL_MODEL_BUF_OFFSET_OFFSET                           31:0
#define LWB9B0_AV1_SET_GLOBAL_MODEL_BUF_OFFSET_OFFSET_HIGH_FIELD                31U
#define LWB9B0_AV1_SET_GLOBAL_MODEL_BUF_OFFSET_OFFSET_LOW_FIELD                 0U
#define LWB9B0_AV1_SET_FILM_GRAIN_BUF_OFFSET                                    (0x00000664U)
#define LWB9B0_AV1_SET_FILM_GRAIN_BUF_OFFSET_OFFSET                             31:0
#define LWB9B0_AV1_SET_FILM_GRAIN_BUF_OFFSET_OFFSET_HIGH_FIELD                  31U
#define LWB9B0_AV1_SET_FILM_GRAIN_BUF_OFFSET_OFFSET_LOW_FIELD                   0U
#define LWB9B0_AV1_SET_TILE_STREAM_INFO_BUF_OFFSET                              (0x00000668U)
#define LWB9B0_AV1_SET_TILE_STREAM_INFO_BUF_OFFSET_OFFSET                       31:0
#define LWB9B0_AV1_SET_TILE_STREAM_INFO_BUF_OFFSET_OFFSET_HIGH_FIELD            31U
#define LWB9B0_AV1_SET_TILE_STREAM_INFO_BUF_OFFSET_OFFSET_LOW_FIELD             0U
#define LWB9B0_AV1_SET_SUB_STREAM_ENTRY_BUF_OFFSET                              (0x0000066LW)
#define LWB9B0_AV1_SET_SUB_STREAM_ENTRY_BUF_OFFSET_OFFSET                       31:0
#define LWB9B0_AV1_SET_SUB_STREAM_ENTRY_BUF_OFFSET_OFFSET_HIGH_FIELD            31U
#define LWB9B0_AV1_SET_SUB_STREAM_ENTRY_BUF_OFFSET_OFFSET_LOW_FIELD             0U
#define LWB9B0_H264_SET_SCALING_LIST_OFFSET                                     (0x00000680U)
#define LWB9B0_H264_SET_SCALING_LIST_OFFSET_OFFSET                              31:0
#define LWB9B0_H264_SET_SCALING_LIST_OFFSET_OFFSET_HIGH_FIELD                   31U
#define LWB9B0_H264_SET_SCALING_LIST_OFFSET_OFFSET_LOW_FIELD                    0U
#define LWB9B0_H264_SET_VLDHIST_BUF_OFFSET                                      (0x00000684U)
#define LWB9B0_H264_SET_VLDHIST_BUF_OFFSET_OFFSET                               31:0
#define LWB9B0_H264_SET_VLDHIST_BUF_OFFSET_OFFSET_HIGH_FIELD                    31U
#define LWB9B0_H264_SET_VLDHIST_BUF_OFFSET_OFFSET_LOW_FIELD                     0U
#define LWB9B0_H264_SET_EDOBOFFSET0                                             (0x00000688U)
#define LWB9B0_H264_SET_EDOBOFFSET0_OFFSET                                      31:0
#define LWB9B0_H264_SET_EDOBOFFSET0_OFFSET_HIGH_FIELD                           31U
#define LWB9B0_H264_SET_EDOBOFFSET0_OFFSET_LOW_FIELD                            0U
#define LWB9B0_H264_SET_EDOBOFFSET1                                             (0x0000068LW)
#define LWB9B0_H264_SET_EDOBOFFSET1_OFFSET                                      31:0
#define LWB9B0_H264_SET_EDOBOFFSET1_OFFSET_HIGH_FIELD                           31U
#define LWB9B0_H264_SET_EDOBOFFSET1_OFFSET_LOW_FIELD                            0U
#define LWB9B0_H264_SET_EDOBOFFSET2                                             (0x00000690U)
#define LWB9B0_H264_SET_EDOBOFFSET2_OFFSET                                      31:0
#define LWB9B0_H264_SET_EDOBOFFSET2_OFFSET_HIGH_FIELD                           31U
#define LWB9B0_H264_SET_EDOBOFFSET2_OFFSET_LOW_FIELD                            0U
#define LWB9B0_H264_SET_EDOBOFFSET3                                             (0x00000694U)
#define LWB9B0_H264_SET_EDOBOFFSET3_OFFSET                                      31:0
#define LWB9B0_H264_SET_EDOBOFFSET3_OFFSET_HIGH_FIELD                           31U
#define LWB9B0_H264_SET_EDOBOFFSET3_OFFSET_LOW_FIELD                            0U
#define LWB9B0_SET_CONTENT_INITIAL_VECTOR(b)                                    (0x00000C00 + ((b)*0x00000004))
#define LWB9B0_SET_CONTENT_INITIAL_VECTOR_VALUE                                 31:0
#define LWB9B0_SET_CONTENT_INITIAL_VECTOR_VALUE_HIGH_FIELD                      31U
#define LWB9B0_SET_CONTENT_INITIAL_VECTOR_VALUE_LOW_FIELD                       0U
#define LWB9B0_SET_CTL_COUNT                                                    (0x00000C10U)
#define LWB9B0_SET_CTL_COUNT_VALUE                                              31:0
#define LWB9B0_SET_CTL_COUNT_VALUE_HIGH_FIELD                                   31U
#define LWB9B0_SET_CTL_COUNT_VALUE_LOW_FIELD                                    0U
#define LWB9B0_SET_UPPER_SRC                                                    (0x00000C14U)
#define LWB9B0_SET_UPPER_SRC_OFFSET                                             7:0
#define LWB9B0_SET_UPPER_SRC_OFFSET_HIGH_FIELD                                  7U
#define LWB9B0_SET_UPPER_SRC_OFFSET_LOW_FIELD                                   0U
#define LWB9B0_SET_LOWER_SRC                                                    (0x00000C18U)
#define LWB9B0_SET_LOWER_SRC_OFFSET                                             31:0
#define LWB9B0_SET_LOWER_SRC_OFFSET_HIGH_FIELD                                  31U
#define LWB9B0_SET_LOWER_SRC_OFFSET_LOW_FIELD                                   0U
#define LWB9B0_SET_UPPER_DST                                                    (0x00000C1LW)
#define LWB9B0_SET_UPPER_DST_OFFSET                                             7:0
#define LWB9B0_SET_UPPER_DST_OFFSET_HIGH_FIELD                                  7U
#define LWB9B0_SET_UPPER_DST_OFFSET_LOW_FIELD                                   0U
#define LWB9B0_SET_LOWER_DST                                                    (0x00000C20U)
#define LWB9B0_SET_LOWER_DST_OFFSET                                             31:0
#define LWB9B0_SET_LOWER_DST_OFFSET_HIGH_FIELD                                  31U
#define LWB9B0_SET_LOWER_DST_OFFSET_LOW_FIELD                                   0U
#define LWB9B0_SET_BLOCK_COUNT                                                  (0x00000C24U)
#define LWB9B0_SET_BLOCK_COUNT_VALUE                                            31:0
#define LWB9B0_SET_BLOCK_COUNT_VALUE_HIGH_FIELD                                 31U
#define LWB9B0_SET_BLOCK_COUNT_VALUE_LOW_FIELD                                  0U
#define LWB9B0_PR_SET_REQUEST_BUFFER                                            (0x00000D00U)
#define LWB9B0_PR_SET_REQUEST_BUFFER_OFFSET                                     31:0
#define LWB9B0_PR_SET_REQUEST_BUFFER_OFFSET_HIGH_FIELD                          31U
#define LWB9B0_PR_SET_REQUEST_BUFFER_OFFSET_LOW_FIELD                           0U
#define LWB9B0_PR_SET_REQUEST_BUFFER_SIZE                                       (0x00000D04U)
#define LWB9B0_PR_SET_REQUEST_BUFFER_SIZE_SIZE                                  31:0
#define LWB9B0_PR_SET_REQUEST_BUFFER_SIZE_SIZE_HIGH_FIELD                       31U
#define LWB9B0_PR_SET_REQUEST_BUFFER_SIZE_SIZE_LOW_FIELD                        0U
#define LWB9B0_PR_SET_RESPONSE_BUFFER                                           (0x00000D08U)
#define LWB9B0_PR_SET_RESPONSE_BUFFER_OFFSET                                    31:0
#define LWB9B0_PR_SET_RESPONSE_BUFFER_OFFSET_HIGH_FIELD                         31U
#define LWB9B0_PR_SET_RESPONSE_BUFFER_OFFSET_LOW_FIELD                          0U
#define LWB9B0_PR_SET_RESPONSE_BUFFER_SIZE                                      (0x00000D0LW)
#define LWB9B0_PR_SET_RESPONSE_BUFFER_SIZE_SIZE                                 31:0
#define LWB9B0_PR_SET_RESPONSE_BUFFER_SIZE_SIZE_HIGH_FIELD                      31U
#define LWB9B0_PR_SET_RESPONSE_BUFFER_SIZE_SIZE_LOW_FIELD                       0U
#define LWB9B0_PR_SET_REQUEST_MESSAGE_BUFFER                                    (0x00000D10U)
#define LWB9B0_PR_SET_REQUEST_MESSAGE_BUFFER_OFFSET                             31:0
#define LWB9B0_PR_SET_REQUEST_MESSAGE_BUFFER_OFFSET_HIGH_FIELD                  31U
#define LWB9B0_PR_SET_REQUEST_MESSAGE_BUFFER_OFFSET_LOW_FIELD                   0U
#define LWB9B0_PR_SET_RESPONSE_MESSAGE_BUFFER                                   (0x00000D14U)
#define LWB9B0_PR_SET_RESPONSE_MESSAGE_BUFFER_OFFSET                            31:0
#define LWB9B0_PR_SET_RESPONSE_MESSAGE_BUFFER_OFFSET_HIGH_FIELD                 31U
#define LWB9B0_PR_SET_RESPONSE_MESSAGE_BUFFER_OFFSET_LOW_FIELD                  0U
#define LWB9B0_PR_SET_LOCAL_DECRYPT_BUFFER                                      (0x00000D18U)
#define LWB9B0_PR_SET_LOCAL_DECRYPT_BUFFER_OFFSET                               31:0
#define LWB9B0_PR_SET_LOCAL_DECRYPT_BUFFER_OFFSET_HIGH_FIELD                    31U
#define LWB9B0_PR_SET_LOCAL_DECRYPT_BUFFER_OFFSET_LOW_FIELD                     0U
#define LWB9B0_PR_SET_LOCAL_DECRYPT_BUFFER_SIZE                                 (0x00000D1LW)
#define LWB9B0_PR_SET_LOCAL_DECRYPT_BUFFER_SIZE_SIZE                            31:0
#define LWB9B0_PR_SET_LOCAL_DECRYPT_BUFFER_SIZE_SIZE_HIGH_FIELD                 31U
#define LWB9B0_PR_SET_LOCAL_DECRYPT_BUFFER_SIZE_SIZE_LOW_FIELD                  0U
#define LWB9B0_PR_SET_CONTENT_DECRYPT_INFO_BUFFER                               (0x00000D20U)
#define LWB9B0_PR_SET_CONTENT_DECRYPT_INFO_BUFFER_OFFSET                        31:0
#define LWB9B0_PR_SET_CONTENT_DECRYPT_INFO_BUFFER_OFFSET_HIGH_FIELD             31U
#define LWB9B0_PR_SET_CONTENT_DECRYPT_INFO_BUFFER_OFFSET_LOW_FIELD              0U
#define LWB9B0_PR_SET_REENCRYPTED_BITSTREAM_SURFACE                             (0x00000D24U)
#define LWB9B0_PR_SET_REENCRYPTED_BITSTREAM_SURFACE_OFFSET                      31:0
#define LWB9B0_PR_SET_REENCRYPTED_BITSTREAM_SURFACE_OFFSET_HIGH_FIELD           31U
#define LWB9B0_PR_SET_REENCRYPTED_BITSTREAM_SURFACE_OFFSET_LOW_FIELD            0U
#define LWB9B0_SET_SESSION_KEY(b)                                               (0x00000F00 + ((b)*0x00000004))
#define LWB9B0_SET_SESSION_KEY_VALUE                                            31:0
#define LWB9B0_SET_SESSION_KEY_VALUE_HIGH_FIELD                                 31U
#define LWB9B0_SET_SESSION_KEY_VALUE_LOW_FIELD                                  0U
#define LWB9B0_SET_CONTENT_KEY(b)                                               (0x00000F10 + ((b)*0x00000004))
#define LWB9B0_SET_CONTENT_KEY_VALUE                                            31:0
#define LWB9B0_SET_CONTENT_KEY_VALUE_HIGH_FIELD                                 31U
#define LWB9B0_SET_CONTENT_KEY_VALUE_LOW_FIELD                                  0U
#define LWB9B0_PM_TRIGGER_END                                                   (0x00001114U)
#define LWB9B0_PM_TRIGGER_END_V                                                 31:0
#define LWB9B0_PM_TRIGGER_END_V_HIGH_FIELD                                      31U
#define LWB9B0_PM_TRIGGER_END_V_LOW_FIELD                                       0U

#define LWB9B0_ERROR_NONE                                                       (0x00000000U)
#define LWB9B0_OS_ERROR_EXELWTE_INSUFFICIENT_DATA                               (0x00000001U)
#define LWB9B0_OS_ERROR_SEMAPHORE_INSUFFICIENT_DATA                             (0x00000002U)
#define LWB9B0_OS_ERROR_ILWALID_METHOD                                          (0x00000003U)
#define LWB9B0_OS_ERROR_ILWALID_DMA_PAGE                                        (0x00000004U)
#define LWB9B0_OS_ERROR_UNHANDLED_INTERRUPT                                     (0x00000005U)
#define LWB9B0_OS_ERROR_EXCEPTION                                               (0x00000006U)
#define LWB9B0_OS_ERROR_ILWALID_CTXSW_REQUEST                                   (0x00000007U)
#define LWB9B0_OS_ERROR_APPLICATION                                             (0x00000008U)
#define LWB9B0_OS_ERROR_SW_BREAKPT                                              (0x00000009U)
#define LWB9B0_OS_INTERRUPT_EXELWTE_AWAKEN                                      (0x00000100U)
#define LWB9B0_OS_INTERRUPT_BACKEND_SEMAPHORE_AWAKEN                            (0x00000200U)
#define LWB9B0_OS_INTERRUPT_CTX_ERROR_FBIF                                      (0x00000300U)
#define LWB9B0_OS_INTERRUPT_LIMIT_VIOLATION                                     (0x00000400U)
#define LWB9B0_OS_INTERRUPT_LIMIT_AND_FBIF_CTX_ERROR                            (0x00000500U)
#define LWB9B0_OS_INTERRUPT_HALT_ENGINE                                         (0x00000600U)
#define LWB9B0_OS_INTERRUPT_TRAP_NONSTALL                                       (0x00000700U)
#define LWB9B0_H264_VLD_ERR_SEQ_DATA_INCONSISTENT                               (0x00004001U)
#define LWB9B0_H264_VLD_ERR_PIC_DATA_INCONSISTENT                               (0x00004002U)
#define LWB9B0_H264_VLD_ERR_SLC_DATA_BUF_ADDR_OUT_OF_BOUNDS                     (0x00004100U)
#define LWB9B0_H264_VLD_ERR_BITSTREAM_ERROR                                     (0x00004101U)
#define LWB9B0_H264_VLD_ERR_CTX_DMA_ID_CTRL_IN_ILWALID                          (0x000041F8U)
#define LWB9B0_H264_VLD_ERR_SLC_HDR_OUT_SIZE_NOT_MULT256                        (0x00004200U)
#define LWB9B0_H264_VLD_ERR_SLC_DATA_OUT_SIZE_NOT_MULT256                       (0x00004201U)
#define LWB9B0_H264_VLD_ERR_CTX_DMA_ID_FLOW_CTRL_ILWALID                        (0x00004203U)
#define LWB9B0_H264_VLD_ERR_CTX_DMA_ID_SLC_HDR_OUT_ILWALID                      (0x00004204U)
#define LWB9B0_H264_VLD_ERR_SLC_HDR_OUT_BUF_TOO_SMALL                           (0x00004205U)
#define LWB9B0_H264_VLD_ERR_SLC_HDR_OUT_BUF_ALREADY_VALID                       (0x00004206U)
#define LWB9B0_H264_VLD_ERR_SLC_DATA_OUT_BUF_TOO_SMALL                          (0x00004207U)
#define LWB9B0_H264_VLD_ERR_DATA_BUF_CNT_TOO_SMALL                              (0x00004208U)
#define LWB9B0_H264_VLD_ERR_BITSTREAM_EMPTY                                     (0x00004209U)
#define LWB9B0_H264_VLD_ERR_FRAME_WIDTH_TOO_LARGE                               (0x0000420AU)
#define LWB9B0_H264_VLD_ERR_FRAME_HEIGHT_TOO_LARGE                              (0x0000420BU)
#define LWB9B0_H264_VLD_ERR_HIST_BUF_TOO_SMALL                                  (0x00004300U)
#define LWB9B0_VC1_VLD_ERR_PIC_DATA_BUF_ADDR_OUT_OF_BOUND                       (0x00005100U)
#define LWB9B0_VC1_VLD_ERR_BITSTREAM_ERROR                                      (0x00005101U)
#define LWB9B0_VC1_VLD_ERR_PIC_HDR_OUT_SIZE_NOT_MULT256                         (0x00005200U)
#define LWB9B0_VC1_VLD_ERR_PIC_DATA_OUT_SIZE_NOT_MULT256                        (0x00005201U)
#define LWB9B0_VC1_VLD_ERR_CTX_DMA_ID_CTRL_IN_ILWALID                           (0x00005202U)
#define LWB9B0_VC1_VLD_ERR_CTX_DMA_ID_FLOW_CTRL_ILWALID                         (0x00005203U)
#define LWB9B0_VC1_VLD_ERR_CTX_DMA_ID_PIC_HDR_OUT_ILWALID                       (0x00005204U)
#define LWB9B0_VC1_VLD_ERR_SLC_HDR_OUT_BUF_TOO_SMALL                            (0x00005205U)
#define LWB9B0_VC1_VLD_ERR_PIC_HDR_OUT_BUF_ALREADY_VALID                        (0x00005206U)
#define LWB9B0_VC1_VLD_ERR_PIC_DATA_OUT_BUF_TOO_SMALL                           (0x00005207U)
#define LWB9B0_VC1_VLD_ERR_DATA_INFO_IN_BUF_TOO_SMALL                           (0x00005208U)
#define LWB9B0_VC1_VLD_ERR_BITSTREAM_EMPTY                                      (0x00005209U)
#define LWB9B0_VC1_VLD_ERR_FRAME_WIDTH_TOO_LARGE                                (0x0000520AU)
#define LWB9B0_VC1_VLD_ERR_FRAME_HEIGHT_TOO_LARGE                               (0x0000520BU)
#define LWB9B0_VC1_VLD_ERR_PIC_DATA_OUT_BUF_FULL_TIME_OUT                       (0x00005300U)
#define LWB9B0_MPEG12_VLD_ERR_SLC_DATA_BUF_ADDR_OUT_OF_BOUNDS                   (0x00006100U)
#define LWB9B0_MPEG12_VLD_ERR_BITSTREAM_ERROR                                   (0x00006101U)
#define LWB9B0_MPEG12_VLD_ERR_SLC_DATA_OUT_SIZE_NOT_MULT256                     (0x00006200U)
#define LWB9B0_MPEG12_VLD_ERR_CTX_DMA_ID_CTRL_IN_ILWALID                        (0x00006201U)
#define LWB9B0_MPEG12_VLD_ERR_CTX_DMA_ID_FLOW_CTRL_ILWALID                      (0x00006202U)
#define LWB9B0_MPEG12_VLD_ERR_SLC_DATA_OUT_BUF_TOO_SMALL                        (0x00006203U)
#define LWB9B0_MPEG12_VLD_ERR_DATA_INFO_IN_BUF_TOO_SMALL                        (0x00006204U)
#define LWB9B0_MPEG12_VLD_ERR_BITSTREAM_EMPTY                                   (0x00006205U)
#define LWB9B0_MPEG12_VLD_ERR_ILWALID_PIC_STRUCTURE                             (0x00006206U)
#define LWB9B0_MPEG12_VLD_ERR_ILWALID_PIC_CODING_TYPE                           (0x00006207U)
#define LWB9B0_MPEG12_VLD_ERR_FRAME_WIDTH_TOO_LARGE                             (0x00006208U)
#define LWB9B0_MPEG12_VLD_ERR_FRAME_HEIGHT_TOO_LARGE                            (0x00006209U)
#define LWB9B0_MPEG12_VLD_ERR_SLC_DATA_OUT_BUF_FULL_TIME_OUT                    (0x00006300U)
#define LWB9B0_CMN_VLD_ERR_PDEC_RETURNED_ERROR                                  (0x00007101U)
#define LWB9B0_CMN_VLD_ERR_EDOB_FLUSH_TIME_OUT                                  (0x00007102U)
#define LWB9B0_CMN_VLD_ERR_EDOB_REWIND_TIME_OUT                                 (0x00007103U)
#define LWB9B0_CMN_VLD_ERR_VLD_WD_TIME_OUT                                      (0x00007104U)
#define LWB9B0_CMN_VLD_ERR_NUM_SLICES_ZERO                                      (0x00007105U)
#define LWB9B0_MPEG4_VLD_ERR_PIC_DATA_BUF_ADDR_OUT_OF_BOUND                     (0x00008100U)
#define LWB9B0_MPEG4_VLD_ERR_BITSTREAM_ERROR                                    (0x00008101U)
#define LWB9B0_MPEG4_VLD_ERR_PIC_HDR_OUT_SIZE_NOT_MULT256                       (0x00008200U)
#define LWB9B0_MPEG4_VLD_ERR_PIC_DATA_OUT_SIZE_NOT_MULT256                      (0x00008201U)
#define LWB9B0_MPEG4_VLD_ERR_CTX_DMA_ID_CTRL_IN_ILWALID                         (0x00008202U)
#define LWB9B0_MPEG4_VLD_ERR_CTX_DMA_ID_FLOW_CTRL_ILWALID                       (0x00008203U)
#define LWB9B0_MPEG4_VLD_ERR_CTX_DMA_ID_PIC_HDR_OUT_ILWALID                     (0x00008204U)
#define LWB9B0_MPEG4_VLD_ERR_SLC_HDR_OUT_BUF_TOO_SMALL                          (0x00008205U)
#define LWB9B0_MPEG4_VLD_ERR_PIC_HDR_OUT_BUF_ALREADY_VALID                      (0x00008206U)
#define LWB9B0_MPEG4_VLD_ERR_PIC_DATA_OUT_BUF_TOO_SMALL                         (0x00008207U)
#define LWB9B0_MPEG4_VLD_ERR_DATA_INFO_IN_BUF_TOO_SMALL                         (0x00008208U)
#define LWB9B0_MPEG4_VLD_ERR_BITSTREAM_EMPTY                                    (0x00008209U)
#define LWB9B0_MPEG4_VLD_ERR_FRAME_WIDTH_TOO_LARGE                              (0x0000820AU)
#define LWB9B0_MPEG4_VLD_ERR_FRAME_HEIGHT_TOO_LARGE                             (0x0000820BU)
#define LWB9B0_MPEG4_VLD_ERR_PIC_DATA_OUT_BUF_FULL_TIME_OUT                     (0x00051E01U)
#define LWB9B0_DEC_ERROR_MPEG12_APPTIMER_EXPIRED                                (0xDEC10001U)
#define LWB9B0_DEC_ERROR_MPEG12_MVTIMER_EXPIRED                                 (0xDEC10002U)
#define LWB9B0_DEC_ERROR_MPEG12_ILWALID_TOKEN                                   (0xDEC10003U)
#define LWB9B0_DEC_ERROR_MPEG12_SLICEDATA_MISSING                               (0xDEC10004U)
#define LWB9B0_DEC_ERROR_MPEG12_HWERR_INTERRUPT                                 (0xDEC10005U)
#define LWB9B0_DEC_ERROR_MPEG12_DETECTED_VLD_FAILURE                            (0xDEC10006U)
#define LWB9B0_DEC_ERROR_MPEG12_PICTURE_INIT                                    (0xDEC10100U)
#define LWB9B0_DEC_ERROR_MPEG12_STATEMACHINE_FAILURE                            (0xDEC10101U)
#define LWB9B0_DEC_ERROR_MPEG12_ILWALID_CTXID_PIC                               (0xDEC10901U)
#define LWB9B0_DEC_ERROR_MPEG12_ILWALID_CTXID_UCODE                             (0xDEC10902U)
#define LWB9B0_DEC_ERROR_MPEG12_ILWALID_CTXID_FC                                (0xDEC10903U)
#define LWB9B0_DEC_ERROR_MPEG12_ILWALID_CTXID_SLH                               (0xDEC10904U)
#define LWB9B0_DEC_ERROR_MPEG12_ILWALID_UCODE_SIZE                              (0xDEC10905U)
#define LWB9B0_DEC_ERROR_MPEG12_ILWALID_SLICE_COUNT                             (0xDEC10906U)
#define LWB9B0_DEC_ERROR_VC1_APPTIMER_EXPIRED                                   (0xDEC20001U)
#define LWB9B0_DEC_ERROR_VC1_MVTIMER_EXPIRED                                    (0xDEC20002U)
#define LWB9B0_DEC_ERROR_VC1_ILWALID_TOKEN                                      (0xDEC20003U)
#define LWB9B0_DEC_ERROR_VC1_SLICEDATA_MISSING                                  (0xDEC20004U)
#define LWB9B0_DEC_ERROR_VC1_HWERR_INTERRUPT                                    (0xDEC20005U)
#define LWB9B0_DEC_ERROR_VC1_DETECTED_VLD_FAILURE                               (0xDEC20006U)
#define LWB9B0_DEC_ERROR_VC1_TIMEOUT_POLLING_FOR_DATA                           (0xDEC20007U)
#define LWB9B0_DEC_ERROR_VC1_PDEC_PIC_END_UNALIGNED                             (0xDEC20008U)
#define LWB9B0_DEC_ERROR_VC1_WDTIMER_EXPIRED                                    (0xDEC20009U)
#define LWB9B0_DEC_ERROR_VC1_ERRINTSTART                                        (0xDEC20010U)
#define LWB9B0_DEC_ERROR_VC1_IQT_ERRINT                                         (0xDEC20011U)
#define LWB9B0_DEC_ERROR_VC1_MC_ERRINT                                          (0xDEC20012U)
#define LWB9B0_DEC_ERROR_VC1_MC_IQT_ERRINT                                      (0xDEC20013U)
#define LWB9B0_DEC_ERROR_VC1_REC_ERRINT                                         (0xDEC20014U)
#define LWB9B0_DEC_ERROR_VC1_REC_IQT_ERRINT                                     (0xDEC20015U)
#define LWB9B0_DEC_ERROR_VC1_REC_MC_ERRINT                                      (0xDEC20016U)
#define LWB9B0_DEC_ERROR_VC1_REC_MC_IQT_ERRINT                                  (0xDEC20017U)
#define LWB9B0_DEC_ERROR_VC1_DBF_ERRINT                                         (0xDEC20018U)
#define LWB9B0_DEC_ERROR_VC1_DBF_IQT_ERRINT                                     (0xDEC20019U)
#define LWB9B0_DEC_ERROR_VC1_DBF_MC_ERRINT                                      (0xDEC2001AU)
#define LWB9B0_DEC_ERROR_VC1_DBF_MC_IQT_ERRINT                                  (0xDEC2001BU)
#define LWB9B0_DEC_ERROR_VC1_DBF_REC_ERRINT                                     (0xDEC2001LW)
#define LWB9B0_DEC_ERROR_VC1_DBF_REC_IQT_ERRINT                                 (0xDEC2001DU)
#define LWB9B0_DEC_ERROR_VC1_DBF_REC_MC_ERRINT                                  (0xDEC2001EU)
#define LWB9B0_DEC_ERROR_VC1_DBF_REC_MC_IQT_ERRINT                              (0xDEC2001FU)
#define LWB9B0_DEC_ERROR_VC1_PICTURE_INIT                                       (0xDEC20100U)
#define LWB9B0_DEC_ERROR_VC1_STATEMACHINE_FAILURE                               (0xDEC20101U)
#define LWB9B0_DEC_ERROR_VC1_ILWALID_CTXID_PIC                                  (0xDEC20901U)
#define LWB9B0_DEC_ERROR_VC1_ILWALID_CTXID_UCODE                                (0xDEC20902U)
#define LWB9B0_DEC_ERROR_VC1_ILWALID_CTXID_FC                                   (0xDEC20903U)
#define LWB9B0_DEC_ERROR_VC1_ILWAILD_CTXID_SLH                                  (0xDEC20904U)
#define LWB9B0_DEC_ERROR_VC1_ILWALID_UCODE_SIZE                                 (0xDEC20905U)
#define LWB9B0_DEC_ERROR_VC1_ILWALID_SLICE_COUNT                                (0xDEC20906U)
#define LWB9B0_DEC_ERROR_H264_APPTIMER_EXPIRED                                  (0xDEC30001U)
#define LWB9B0_DEC_ERROR_H264_MVTIMER_EXPIRED                                   (0xDEC30002U)
#define LWB9B0_DEC_ERROR_H264_ILWALID_TOKEN                                     (0xDEC30003U)
#define LWB9B0_DEC_ERROR_H264_SLICEDATA_MISSING                                 (0xDEC30004U)
#define LWB9B0_DEC_ERROR_H264_HWERR_INTERRUPT                                   (0xDEC30005U)
#define LWB9B0_DEC_ERROR_H264_DETECTED_VLD_FAILURE                              (0xDEC30006U)
#define LWB9B0_DEC_ERROR_H264_ERRINTSTART                                       (0xDEC30010U)
#define LWB9B0_DEC_ERROR_H264_IQT_ERRINT                                        (0xDEC30011U)
#define LWB9B0_DEC_ERROR_H264_MC_ERRINT                                         (0xDEC30012U)
#define LWB9B0_DEC_ERROR_H264_MC_IQT_ERRINT                                     (0xDEC30013U)
#define LWB9B0_DEC_ERROR_H264_REC_ERRINT                                        (0xDEC30014U)
#define LWB9B0_DEC_ERROR_H264_REC_IQT_ERRINT                                    (0xDEC30015U)
#define LWB9B0_DEC_ERROR_H264_REC_MC_ERRINT                                     (0xDEC30016U)
#define LWB9B0_DEC_ERROR_H264_REC_MC_IQT_ERRINT                                 (0xDEC30017U)
#define LWB9B0_DEC_ERROR_H264_DBF_ERRINT                                        (0xDEC30018U)
#define LWB9B0_DEC_ERROR_H264_DBF_IQT_ERRINT                                    (0xDEC30019U)
#define LWB9B0_DEC_ERROR_H264_DBF_MC_ERRINT                                     (0xDEC3001AU)
#define LWB9B0_DEC_ERROR_H264_DBF_MC_IQT_ERRINT                                 (0xDEC3001BU)
#define LWB9B0_DEC_ERROR_H264_DBF_REC_ERRINT                                    (0xDEC3001LW)
#define LWB9B0_DEC_ERROR_H264_DBF_REC_IQT_ERRINT                                (0xDEC3001DU)
#define LWB9B0_DEC_ERROR_H264_DBF_REC_MC_ERRINT                                 (0xDEC3001EU)
#define LWB9B0_DEC_ERROR_H264_DBF_REC_MC_IQT_ERRINT                             (0xDEC3001FU)
#define LWB9B0_DEC_ERROR_H264_PICTURE_INIT                                      (0xDEC30100U)
#define LWB9B0_DEC_ERROR_H264_STATEMACHINE_FAILURE                              (0xDEC30101U)
#define LWB9B0_DEC_ERROR_H264_ILWALID_CTXID_PIC                                 (0xDEC30901U)
#define LWB9B0_DEC_ERROR_H264_ILWALID_CTXID_UCODE                               (0xDEC30902U)
#define LWB9B0_DEC_ERROR_H264_ILWALID_CTXID_FC                                  (0xDEC30903U)
#define LWB9B0_DEC_ERROR_H264_ILWALID_CTXID_SLH                                 (0xDEC30904U)
#define LWB9B0_DEC_ERROR_H264_ILWALID_UCODE_SIZE                                (0xDEC30905U)
#define LWB9B0_DEC_ERROR_H264_ILWALID_SLICE_COUNT                               (0xDEC30906U)
#define LWB9B0_DEC_ERROR_MPEG4_APPTIMER_EXPIRED                                 (0xDEC40001U)
#define LWB9B0_DEC_ERROR_MPEG4_MVTIMER_EXPIRED                                  (0xDEC40002U)
#define LWB9B0_DEC_ERROR_MPEG4_ILWALID_TOKEN                                    (0xDEC40003U)
#define LWB9B0_DEC_ERROR_MPEG4_SLICEDATA_MISSING                                (0xDEC40004U)
#define LWB9B0_DEC_ERROR_MPEG4_HWERR_INTERRUPT                                  (0xDEC40005U)
#define LWB9B0_DEC_ERROR_MPEG4_DETECTED_VLD_FAILURE                             (0xDEC40006U)
#define LWB9B0_DEC_ERROR_MPEG4_TIMEOUT_POLLING_FOR_DATA                         (0xDEC40007U)
#define LWB9B0_DEC_ERROR_MPEG4_PDEC_PIC_END_UNALIGNED                           (0xDEC40008U)
#define LWB9B0_DEC_ERROR_MPEG4_WDTIMER_EXPIRED                                  (0xDEC40009U)
#define LWB9B0_DEC_ERROR_MPEG4_ERRINTSTART                                      (0xDEC40010U)
#define LWB9B0_DEC_ERROR_MPEG4_IQT_ERRINT                                       (0xDEC40011U)
#define LWB9B0_DEC_ERROR_MPEG4_MC_ERRINT                                        (0xDEC40012U)
#define LWB9B0_DEC_ERROR_MPEG4_MC_IQT_ERRINT                                    (0xDEC40013U)
#define LWB9B0_DEC_ERROR_MPEG4_REC_ERRINT                                       (0xDEC40014U)
#define LWB9B0_DEC_ERROR_MPEG4_REC_IQT_ERRINT                                   (0xDEC40015U)
#define LWB9B0_DEC_ERROR_MPEG4_REC_MC_ERRINT                                    (0xDEC40016U)
#define LWB9B0_DEC_ERROR_MPEG4_REC_MC_IQT_ERRINT                                (0xDEC40017U)
#define LWB9B0_DEC_ERROR_MPEG4_DBF_ERRINT                                       (0xDEC40018U)
#define LWB9B0_DEC_ERROR_MPEG4_DBF_IQT_ERRINT                                   (0xDEC40019U)
#define LWB9B0_DEC_ERROR_MPEG4_DBF_MC_ERRINT                                    (0xDEC4001AU)
#define LWB9B0_DEC_ERROR_MPEG4_DBF_MC_IQT_ERRINT                                (0xDEC4001BU)
#define LWB9B0_DEC_ERROR_MPEG4_DBF_REC_ERRINT                                   (0xDEC4001LW)
#define LWB9B0_DEC_ERROR_MPEG4_DBF_REC_IQT_ERRINT                               (0xDEC4001DU)
#define LWB9B0_DEC_ERROR_MPEG4_DBF_REC_MC_ERRINT                                (0xDEC4001EU)
#define LWB9B0_DEC_ERROR_MPEG4_DBF_REC_MC_IQT_ERRINT                            (0xDEC4001FU)
#define LWB9B0_DEC_ERROR_MPEG4_PICTURE_INIT                                     (0xDEC40100U)
#define LWB9B0_DEC_ERROR_MPEG4_STATEMACHINE_FAILURE                             (0xDEC40101U)
#define LWB9B0_DEC_ERROR_MPEG4_ILWALID_CTXID_PIC                                (0xDEC40901U)
#define LWB9B0_DEC_ERROR_MPEG4_ILWALID_CTXID_UCODE                              (0xDEC40902U)
#define LWB9B0_DEC_ERROR_MPEG4_ILWALID_CTXID_FC                                 (0xDEC40903U)
#define LWB9B0_DEC_ERROR_MPEG4_ILWALID_CTXID_SLH                                (0xDEC40904U)
#define LWB9B0_DEC_ERROR_MPEG4_ILWALID_UCODE_SIZE                               (0xDEC40905U)
#define LWB9B0_DEC_ERROR_MPEG4_ILWALID_SLICE_COUNT                              (0xDEC40906U)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // clb9b0_h

