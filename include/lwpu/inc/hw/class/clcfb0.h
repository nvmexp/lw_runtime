// WARNING!!! THIS HEADER INCLUDES SOFTWARE METHODS!!!
// ********** DO NOT USE IN HW TREE.  ********** 
/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 1993-2004 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


#ifndef clcfb0_h_
#define clcfb0_h_

#include "lwtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LWCFB0_VIDEO_DECODER                                                       (0x0000CFB0)

typedef volatile struct _clcfb0_tag0 {
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
    LwV32 SetPrSsmContentInfoBufOffset;                                         // 0x000004D8 - 0x000004DB
    LwV32 Reserved06[0x9];
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
    LwV32 VP9SetHintDumpBufOffset;                                              // 0x000005EC - 0x000005EF
    LwV32 Reserved10[0x4];
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
    LwV32 AV1SetHintDumpBufOffset;                                              // 0x00000670 - 0x00000673
    LwV32 Reserved12[0x3];
    LwV32 NewH264SetScalingListOffset;                                          // 0x00000680 - 0x00000683
    LwV32 Reserved13[0x15F];
    LwV32 SetContentInitialVector[4];                                           // 0x00000C00 - 0x00000C0F
    LwV32 SetCtlCount;                                                          // 0x00000C10 - 0x00000C13
    LwV32 SetUpperSrc;                                                          // 0x00000C14 - 0x00000C17
    LwV32 SetLowerSrc;                                                          // 0x00000C18 - 0x00000C1B
    LwV32 SetUpperDst;                                                          // 0x00000C1C - 0x00000C1F
    LwV32 SetLowerDst;                                                          // 0x00000C20 - 0x00000C23
    LwV32 SetBlockCount;                                                        // 0x00000C24 - 0x00000C27
    LwV32 Reserved14[0x36];
    LwV32 PrSetRequestBufOffset;                                                // 0x00000D00 - 0x00000D03
    LwV32 PrSetRequestBufSize;                                                  // 0x00000D04 - 0x00000D07
    LwV32 PrSetResponseBufOffset;                                               // 0x00000D08 - 0x00000D0B
    LwV32 PrSetResponseBufSize;                                                 // 0x00000D0C - 0x00000D0F
    LwV32 PrSetRequestMessageBufOffset;                                         // 0x00000D10 - 0x00000D13
    LwV32 PrSetResponseMessageBufOffset;                                        // 0x00000D14 - 0x00000D17
    LwV32 PrSetLocalDecryptBufOffset;                                           // 0x00000D18 - 0x00000D1B
    LwV32 PrSetLocalDecryptBufSize;                                             // 0x00000D1C - 0x00000D1F
    LwV32 PrSetContentDecryptInfoBufOffset;                                     // 0x00000D20 - 0x00000D23
    LwV32 PrSetReencryptedBitstreamBufOffset;                                   // 0x00000D24 - 0x00000D27
    LwV32 Reserved15[0x36];
    LwV32 DhKeSetChallengeBufOffset;                                            // 0x00000E00 - 0x00000E03
    LwV32 DhKeSetResponseBufOffset;                                             // 0x00000E04 - 0x00000E07
    LwV32 Reserved16[0x3E];
    LwV32 SetSessionKey[4];                                                     // 0x00000F00 - 0x00000F0F
    LwV32 SetContentKey[4];                                                     // 0x00000F10 - 0x00000F1F
    LwV32 Reserved17[0x7D];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved18[0x3BA];
} LWCFB0_VIDEO_DECODERControlPio;

#define LWCFB0_NOP                                                              (0x00000100)
#define LWCFB0_NOP_PARAMETER                                                    31:0
#define LWCFB0_PM_TRIGGER                                                       (0x00000140)
#define LWCFB0_PM_TRIGGER_V                                                     31:0
#define LWCFB0_SET_APPLICATION_ID                                               (0x00000200)
#define LWCFB0_SET_APPLICATION_ID_ID                                            31:0
#define LWCFB0_SET_APPLICATION_ID_ID_MPEG12                                     (0x00000001)
#define LWCFB0_SET_APPLICATION_ID_ID_VC1                                        (0x00000002)
#define LWCFB0_SET_APPLICATION_ID_ID_H264                                       (0x00000003)
#define LWCFB0_SET_APPLICATION_ID_ID_MPEG4                                      (0x00000004)
#define LWCFB0_SET_APPLICATION_ID_ID_VP8                                        (0x00000005)
#define LWCFB0_SET_APPLICATION_ID_ID_CTR64                                      (0x00000006)
#define LWCFB0_SET_APPLICATION_ID_ID_HEVC                                       (0x00000007)
#define LWCFB0_SET_APPLICATION_ID_ID_NEWH264                                    (0x00000008)
#define LWCFB0_SET_APPLICATION_ID_ID_VP9                                        (0x00000009)
#define LWCFB0_SET_APPLICATION_ID_ID_PASS1                                      (0x0000000A)
#define LWCFB0_SET_APPLICATION_ID_ID_HEVC_PARSER                                (0x0000000C)
#define LWCFB0_SET_APPLICATION_ID_ID_UCODE_TEST                                 (0x0000000D)
#define LWCFB0_SET_APPLICATION_ID_ID_HWDRM_PR_DECRYPTAUDIO                      (0x0000000E)
#define LWCFB0_SET_APPLICATION_ID_ID_HWDRM_PR_DECRYPTAUDIOMULTIPLE              (0x0000000F)
#define LWCFB0_SET_APPLICATION_ID_ID_HWDRM_PR_PREPROCESSENCRYPTEDDATA           (0x00000010)
#define LWCFB0_SET_APPLICATION_ID_ID_VP9_WITH_PARSER                            (0x00000011)
#define LWCFB0_SET_APPLICATION_ID_ID_AVD                                        (0x00000012)
#define LWCFB0_SET_APPLICATION_ID_ID_HW_DRM_PR4_DECRYPTCONTENTMULTIPLE          (0x00000013)
#define LWCFB0_SET_APPLICATION_ID_ID_DHKE                                       (0x00000020)
#define LWCFB0_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LWCFB0_SET_WATCHDOG_TIMER_TIMER                                         31:0
#define LWCFB0_SEMAPHORE_A                                                      (0x00000240)
#define LWCFB0_SEMAPHORE_A_UPPER                                                7:0
#define LWCFB0_SEMAPHORE_B                                                      (0x00000244)
#define LWCFB0_SEMAPHORE_B_LOWER                                                31:0
#define LWCFB0_SEMAPHORE_C                                                      (0x00000248)
#define LWCFB0_SEMAPHORE_C_PAYLOAD                                              31:0
#define LWCFB0_CTX_SAVE_AREA                                                    (0x0000024C)
#define LWCFB0_CTX_SAVE_AREA_OFFSET                                             31:0
#define LWCFB0_CTX_SWITCH                                                       (0x00000250)
#define LWCFB0_CTX_SWITCH_OP                                                    1:0
#define LWCFB0_CTX_SWITCH_OP_CTX_UPDATE                                         (0x00000000)
#define LWCFB0_CTX_SWITCH_OP_CTX_SAVE                                           (0x00000001)
#define LWCFB0_CTX_SWITCH_OP_CTX_RESTORE                                        (0x00000002)
#define LWCFB0_CTX_SWITCH_OP_CTX_FORCERESTORE                                   (0x00000003)
#define LWCFB0_CTX_SWITCH_CTXID_VALID                                           2:2
#define LWCFB0_CTX_SWITCH_CTXID_VALID_FALSE                                     (0x00000000)
#define LWCFB0_CTX_SWITCH_CTXID_VALID_TRUE                                      (0x00000001)
#define LWCFB0_CTX_SWITCH_RESERVED0                                             7:3
#define LWCFB0_CTX_SWITCH_CTX_ID                                                23:8
#define LWCFB0_CTX_SWITCH_RESERVED1                                             31:24
#define LWCFB0_SET_SEMAPHORE_PAYLOAD_LOWER                                      (0x00000254)
#define LWCFB0_SET_SEMAPHORE_PAYLOAD_LOWER_PAYLOAD_LOWER                        31:0
#define LWCFB0_SET_SEMAPHORE_PAYLOAD_UPPER                                      (0x00000258)
#define LWCFB0_SET_SEMAPHORE_PAYLOAD_UPPER_PAYLOAD_UPPER                        31:0
#define LWCFB0_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_A                        (0x0000025C)
#define LWCFB0_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_A_LOWER                  31:0
#define LWCFB0_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_B                        (0x00000260)
#define LWCFB0_SET_MONITORED_FENCE_SIGNAL_ADDRESS_BASE_B_UPPER                  31:0
#define LWCFB0_EXELWTE                                                          (0x00000300)
#define LWCFB0_EXELWTE_NOTIFY                                                   0:0
#define LWCFB0_EXELWTE_NOTIFY_DISABLE                                           (0x00000000)
#define LWCFB0_EXELWTE_NOTIFY_ENABLE                                            (0x00000001)
#define LWCFB0_EXELWTE_NOTIFY_ON                                                1:1
#define LWCFB0_EXELWTE_NOTIFY_ON_END                                            (0x00000000)
#define LWCFB0_EXELWTE_NOTIFY_ON_BEGIN                                          (0x00000001)
#define LWCFB0_EXELWTE_PREDICATION                                              2:2
#define LWCFB0_EXELWTE_PREDICATION_DISABLE                                      (0x00000000)
#define LWCFB0_EXELWTE_PREDICATION_ENABLE                                       (0x00000001)
#define LWCFB0_EXELWTE_PREDICATION_OP                                           3:3
#define LWCFB0_EXELWTE_PREDICATION_OP_EQUAL_ZERO                                (0x00000000)
#define LWCFB0_EXELWTE_PREDICATION_OP_NOT_EQUAL_ZERO                            (0x00000001)
#define LWCFB0_EXELWTE_AWAKEN                                                   8:8
#define LWCFB0_EXELWTE_AWAKEN_DISABLE                                           (0x00000000)
#define LWCFB0_EXELWTE_AWAKEN_ENABLE                                            (0x00000001)
#define LWCFB0_SEMAPHORE_D                                                      (0x00000304)
#define LWCFB0_SEMAPHORE_D_STRUCTURE_SIZE                                       1:0
#define LWCFB0_SEMAPHORE_D_STRUCTURE_SIZE_ONE                                   (0x00000000)
#define LWCFB0_SEMAPHORE_D_STRUCTURE_SIZE_FOUR                                  (0x00000001)
#define LWCFB0_SEMAPHORE_D_STRUCTURE_SIZE_TWO                                   (0x00000002)
#define LWCFB0_SEMAPHORE_D_AWAKEN_ENABLE                                        8:8
#define LWCFB0_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                  (0x00000000)
#define LWCFB0_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                   (0x00000001)
#define LWCFB0_SEMAPHORE_D_OPERATION                                            17:16
#define LWCFB0_SEMAPHORE_D_OPERATION_RELEASE                                    (0x00000000)
#define LWCFB0_SEMAPHORE_D_OPERATION_RESERVED_0                                 (0x00000001)
#define LWCFB0_SEMAPHORE_D_OPERATION_RESERVED_1                                 (0x00000002)
#define LWCFB0_SEMAPHORE_D_OPERATION_TRAP                                       (0x00000003)
#define LWCFB0_SEMAPHORE_D_FLUSH_DISABLE                                        21:21
#define LWCFB0_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                  (0x00000000)
#define LWCFB0_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                   (0x00000001)
#define LWCFB0_SEMAPHORE_D_TRAP_TYPE                                            23:22
#define LWCFB0_SEMAPHORE_D_TRAP_TYPE_UNCONDITIONAL                              (0x00000000)
#define LWCFB0_SEMAPHORE_D_TRAP_TYPE_CONDITIONAL                                (0x00000001)
#define LWCFB0_SEMAPHORE_D_TRAP_TYPE_CONDITIONAL_EXT                            (0x00000002)
#define LWCFB0_SEMAPHORE_D_PAYLOAD_SIZE                                         24:24
#define LWCFB0_SEMAPHORE_D_PAYLOAD_SIZE_32BIT                                   (0x00000000)
#define LWCFB0_SEMAPHORE_D_PAYLOAD_SIZE_64BIT                                   (0x00000001)
#define LWCFB0_SET_PREDICATION_OFFSET_UPPER                                     (0x00000308)
#define LWCFB0_SET_PREDICATION_OFFSET_UPPER_OFFSET                              7:0
#define LWCFB0_SET_PREDICATION_OFFSET_LOWER                                     (0x0000030C)
#define LWCFB0_SET_PREDICATION_OFFSET_LOWER_OFFSET                              31:0
#define LWCFB0_SET_AUXILIARY_DATA_BUFFER                                        (0x00000310)
#define LWCFB0_SET_AUXILIARY_DATA_BUFFER_OFFSET                                 31:0
#define LWCFB0_SET_CONTROL_PARAMS                                               (0x00000400)
#define LWCFB0_SET_CONTROL_PARAMS_CODEC_TYPE                                    3:0
#define LWCFB0_SET_CONTROL_PARAMS_CODEC_TYPE_MPEG1                              (0x00000000)
#define LWCFB0_SET_CONTROL_PARAMS_CODEC_TYPE_MPEG2                              (0x00000001)
#define LWCFB0_SET_CONTROL_PARAMS_CODEC_TYPE_VC1                                (0x00000002)
#define LWCFB0_SET_CONTROL_PARAMS_CODEC_TYPE_H264                               (0x00000003)
#define LWCFB0_SET_CONTROL_PARAMS_CODEC_TYPE_MPEG4                              (0x00000004)
#define LWCFB0_SET_CONTROL_PARAMS_CODEC_TYPE_DIVX3                              (0x00000004)
#define LWCFB0_SET_CONTROL_PARAMS_CODEC_TYPE_VP8                                (0x00000005)
#define LWCFB0_SET_CONTROL_PARAMS_CODEC_TYPE_HEVC                               (0x00000007)
#define LWCFB0_SET_CONTROL_PARAMS_CODEC_TYPE_NEWH264                            (0x00000008)
#define LWCFB0_SET_CONTROL_PARAMS_CODEC_TYPE_VP9                                (0x00000009)
#define LWCFB0_SET_CONTROL_PARAMS_CODEC_TYPE_AV1                                (0x0000000A)
#define LWCFB0_SET_CONTROL_PARAMS_GPTIMER_ON                                    4:4
#define LWCFB0_SET_CONTROL_PARAMS_RET_ERROR                                     5:5
#define LWCFB0_SET_CONTROL_PARAMS_ERR_CONCEAL_ON                                6:6
#define LWCFB0_SET_CONTROL_PARAMS_ERROR_FRM_IDX                                 12:7
#define LWCFB0_SET_CONTROL_PARAMS_MBTIMER_ON                                    13:13
#define LWCFB0_SET_CONTROL_PARAMS_EC_INTRA_FRAME_USING_PSLC                     14:14
#define LWCFB0_SET_CONTROL_PARAMS_IGNORE_SOME_FIELDS_CRC_CHECK                  15:15
#define LWCFB0_SET_CONTROL_PARAMS_EVENT_TRACE_LOGGING_ON                        16:16
#define LWCFB0_SET_CONTROL_PARAMS_ALL_INTRA_FRAME                               17:17
#define LWCFB0_SET_CONTROL_PARAMS_TESTRUN_ELW                                   19:18
#define LWCFB0_SET_CONTROL_PARAMS_TESTRUN_ELW_TRACE3D_RUN                       (0x00000000)
#define LWCFB0_SET_CONTROL_PARAMS_TESTRUN_ELW_PROD_RUN                          (0x00000001)
#define LWCFB0_SET_CONTROL_PARAMS_HINT_DUMP_EN                                  20:20
#define LWCFB0_SET_CONTROL_PARAMS_RESERVED                                      25:21
#define LWCFB0_SET_CONTROL_PARAMS_LWDECSIM_SKIP_SCP                             26:26
#define LWCFB0_SET_CONTROL_PARAMS_ENABLE_ENCRYPT                                27:27
#define LWCFB0_SET_CONTROL_PARAMS_ENCRYPTMODE                                   31:28
#define LWCFB0_SET_DRV_PIC_SETUP_OFFSET                                         (0x00000404)
#define LWCFB0_SET_DRV_PIC_SETUP_OFFSET_OFFSET                                  31:0
#define LWCFB0_SET_IN_BUF_BASE_OFFSET                                           (0x00000408)
#define LWCFB0_SET_IN_BUF_BASE_OFFSET_OFFSET                                    31:0
#define LWCFB0_SET_PICTURE_INDEX                                                (0x0000040C)
#define LWCFB0_SET_PICTURE_INDEX_INDEX                                          31:0
#define LWCFB0_SET_SLICE_OFFSETS_BUF_OFFSET                                     (0x00000410)
#define LWCFB0_SET_SLICE_OFFSETS_BUF_OFFSET_OFFSET                              31:0
#define LWCFB0_SET_COLOC_DATA_OFFSET                                            (0x00000414)
#define LWCFB0_SET_COLOC_DATA_OFFSET_OFFSET                                     31:0
#define LWCFB0_SET_HISTORY_OFFSET                                               (0x00000418)
#define LWCFB0_SET_HISTORY_OFFSET_OFFSET                                        31:0
#define LWCFB0_SET_DISPLAY_BUF_SIZE                                             (0x0000041C)
#define LWCFB0_SET_DISPLAY_BUF_SIZE_SIZE                                        31:0
#define LWCFB0_SET_HISTOGRAM_OFFSET                                             (0x00000420)
#define LWCFB0_SET_HISTOGRAM_OFFSET_OFFSET                                      31:0
#define LWCFB0_SET_LWDEC_STATUS_OFFSET                                          (0x00000424)
#define LWCFB0_SET_LWDEC_STATUS_OFFSET_OFFSET                                   31:0
#define LWCFB0_SET_DISPLAY_BUF_LUMA_OFFSET                                      (0x00000428)
#define LWCFB0_SET_DISPLAY_BUF_LUMA_OFFSET_OFFSET                               31:0
#define LWCFB0_SET_DISPLAY_BUF_CHROMA_OFFSET                                    (0x0000042C)
#define LWCFB0_SET_DISPLAY_BUF_CHROMA_OFFSET_OFFSET                             31:0
#define LWCFB0_SET_PICTURE_LUMA_OFFSET0                                         (0x00000430)
#define LWCFB0_SET_PICTURE_LUMA_OFFSET0_OFFSET                                  31:0
#define LWCFB0_SET_PICTURE_LUMA_OFFSET1                                         (0x00000434)
#define LWCFB0_SET_PICTURE_LUMA_OFFSET1_OFFSET                                  31:0
#define LWCFB0_SET_PICTURE_LUMA_OFFSET2                                         (0x00000438)
#define LWCFB0_SET_PICTURE_LUMA_OFFSET2_OFFSET                                  31:0
#define LWCFB0_SET_PICTURE_LUMA_OFFSET3                                         (0x0000043C)
#define LWCFB0_SET_PICTURE_LUMA_OFFSET3_OFFSET                                  31:0
#define LWCFB0_SET_PICTURE_LUMA_OFFSET4                                         (0x00000440)
#define LWCFB0_SET_PICTURE_LUMA_OFFSET4_OFFSET                                  31:0
#define LWCFB0_SET_PICTURE_LUMA_OFFSET5                                         (0x00000444)
#define LWCFB0_SET_PICTURE_LUMA_OFFSET5_OFFSET                                  31:0
#define LWCFB0_SET_PICTURE_LUMA_OFFSET6                                         (0x00000448)
#define LWCFB0_SET_PICTURE_LUMA_OFFSET6_OFFSET                                  31:0
#define LWCFB0_SET_PICTURE_LUMA_OFFSET7                                         (0x0000044C)
#define LWCFB0_SET_PICTURE_LUMA_OFFSET7_OFFSET                                  31:0
#define LWCFB0_SET_PICTURE_LUMA_OFFSET8                                         (0x00000450)
#define LWCFB0_SET_PICTURE_LUMA_OFFSET8_OFFSET                                  31:0
#define LWCFB0_SET_PICTURE_LUMA_OFFSET9                                         (0x00000454)
#define LWCFB0_SET_PICTURE_LUMA_OFFSET9_OFFSET                                  31:0
#define LWCFB0_SET_PICTURE_LUMA_OFFSET10                                        (0x00000458)
#define LWCFB0_SET_PICTURE_LUMA_OFFSET10_OFFSET                                 31:0
#define LWCFB0_SET_PICTURE_LUMA_OFFSET11                                        (0x0000045C)
#define LWCFB0_SET_PICTURE_LUMA_OFFSET11_OFFSET                                 31:0
#define LWCFB0_SET_PICTURE_LUMA_OFFSET12                                        (0x00000460)
#define LWCFB0_SET_PICTURE_LUMA_OFFSET12_OFFSET                                 31:0
#define LWCFB0_SET_PICTURE_LUMA_OFFSET13                                        (0x00000464)
#define LWCFB0_SET_PICTURE_LUMA_OFFSET13_OFFSET                                 31:0
#define LWCFB0_SET_PICTURE_LUMA_OFFSET14                                        (0x00000468)
#define LWCFB0_SET_PICTURE_LUMA_OFFSET14_OFFSET                                 31:0
#define LWCFB0_SET_PICTURE_LUMA_OFFSET15                                        (0x0000046C)
#define LWCFB0_SET_PICTURE_LUMA_OFFSET15_OFFSET                                 31:0
#define LWCFB0_SET_PICTURE_LUMA_OFFSET16                                        (0x00000470)
#define LWCFB0_SET_PICTURE_LUMA_OFFSET16_OFFSET                                 31:0
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET0                                       (0x00000474)
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET0_OFFSET                                31:0
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET1                                       (0x00000478)
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET1_OFFSET                                31:0
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET2                                       (0x0000047C)
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET2_OFFSET                                31:0
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET3                                       (0x00000480)
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET3_OFFSET                                31:0
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET4                                       (0x00000484)
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET4_OFFSET                                31:0
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET5                                       (0x00000488)
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET5_OFFSET                                31:0
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET6                                       (0x0000048C)
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET6_OFFSET                                31:0
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET7                                       (0x00000490)
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET7_OFFSET                                31:0
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET8                                       (0x00000494)
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET8_OFFSET                                31:0
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET9                                       (0x00000498)
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET9_OFFSET                                31:0
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET10                                      (0x0000049C)
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET10_OFFSET                               31:0
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET11                                      (0x000004A0)
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET11_OFFSET                               31:0
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET12                                      (0x000004A4)
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET12_OFFSET                               31:0
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET13                                      (0x000004A8)
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET13_OFFSET                               31:0
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET14                                      (0x000004AC)
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET14_OFFSET                               31:0
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET15                                      (0x000004B0)
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET15_OFFSET                               31:0
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET16                                      (0x000004B4)
#define LWCFB0_SET_PICTURE_CHROMA_OFFSET16_OFFSET                               31:0
#define LWCFB0_SET_PIC_SCRATCH_BUF_OFFSET                                       (0x000004B8)
#define LWCFB0_SET_PIC_SCRATCH_BUF_OFFSET_OFFSET                                31:0
#define LWCFB0_SET_EXTERNAL_MVBUFFER_OFFSET                                     (0x000004BC)
#define LWCFB0_SET_EXTERNAL_MVBUFFER_OFFSET_OFFSET                              31:0
#define LWCFB0_SET_SUB_SAMPLE_MAP_OFFSET                                        (0x000004C0)
#define LWCFB0_SET_SUB_SAMPLE_MAP_OFFSET_OFFSET                                 31:0
#define LWCFB0_SET_SUB_SAMPLE_MAP_IV_OFFSET                                     (0x000004C4)
#define LWCFB0_SET_SUB_SAMPLE_MAP_IV_OFFSET_OFFSET                              31:0
#define LWCFB0_SET_INTRA_TOP_BUF_OFFSET                                         (0x000004C8)
#define LWCFB0_SET_INTRA_TOP_BUF_OFFSET_OFFSET                                  31:0
#define LWCFB0_SET_TILE_SIZE_BUF_OFFSET                                         (0x000004CC)
#define LWCFB0_SET_TILE_SIZE_BUF_OFFSET_OFFSET                                  31:0
#define LWCFB0_SET_FILTER_BUFFER_OFFSET                                         (0x000004D0)
#define LWCFB0_SET_FILTER_BUFFER_OFFSET_OFFSET                                  31:0
#define LWCFB0_SET_CRC_STRUCT_OFFSET                                            (0x000004D4)
#define LWCFB0_SET_CRC_STRUCT_OFFSET_OFFSET                                     31:0
#define LWCFB0_SET_PR_SSM_CONTENT_INFO_BUF_OFFSET                               (0x000004D8)
#define LWCFB0_SET_PR_SSM_CONTENT_INFO_BUF_OFFSET_OFFSET                        31:0
#define LWCFB0_H264_SET_MBHIST_BUF_OFFSET                                       (0x00000500)
#define LWCFB0_H264_SET_MBHIST_BUF_OFFSET_OFFSET                                31:0
#define LWCFB0_VP8_SET_PROB_DATA_OFFSET                                         (0x00000540)
#define LWCFB0_VP8_SET_PROB_DATA_OFFSET_OFFSET                                  31:0
#define LWCFB0_VP8_SET_HEADER_PARTITION_BUF_BASE_OFFSET                         (0x00000544)
#define LWCFB0_VP8_SET_HEADER_PARTITION_BUF_BASE_OFFSET_OFFSET                  31:0
#define LWCFB0_HEVC_SET_SCALING_LIST_OFFSET                                     (0x00000580)
#define LWCFB0_HEVC_SET_SCALING_LIST_OFFSET_OFFSET                              31:0
#define LWCFB0_HEVC_SET_TILE_SIZES_OFFSET                                       (0x00000584)
#define LWCFB0_HEVC_SET_TILE_SIZES_OFFSET_OFFSET                                31:0
#define LWCFB0_HEVC_SET_FILTER_BUFFER_OFFSET                                    (0x00000588)
#define LWCFB0_HEVC_SET_FILTER_BUFFER_OFFSET_OFFSET                             31:0
#define LWCFB0_HEVC_SET_SAO_BUFFER_OFFSET                                       (0x0000058C)
#define LWCFB0_HEVC_SET_SAO_BUFFER_OFFSET_OFFSET                                31:0
#define LWCFB0_HEVC_SET_SLICE_INFO_BUFFER_OFFSET                                (0x00000590)
#define LWCFB0_HEVC_SET_SLICE_INFO_BUFFER_OFFSET_OFFSET                         31:0
#define LWCFB0_HEVC_SET_SLICE_GROUP_INDEX                                       (0x00000594)
#define LWCFB0_HEVC_SET_SLICE_GROUP_INDEX_OFFSET                                31:0
#define LWCFB0_VP9_SET_PROB_TAB_BUF_OFFSET                                      (0x000005C0)
#define LWCFB0_VP9_SET_PROB_TAB_BUF_OFFSET_OFFSET                               31:0
#define LWCFB0_VP9_SET_CTX_COUNTER_BUF_OFFSET                                   (0x000005C4)
#define LWCFB0_VP9_SET_CTX_COUNTER_BUF_OFFSET_OFFSET                            31:0
#define LWCFB0_VP9_SET_SEGMENT_READ_BUF_OFFSET                                  (0x000005C8)
#define LWCFB0_VP9_SET_SEGMENT_READ_BUF_OFFSET_OFFSET                           31:0
#define LWCFB0_VP9_SET_SEGMENT_WRITE_BUF_OFFSET                                 (0x000005CC)
#define LWCFB0_VP9_SET_SEGMENT_WRITE_BUF_OFFSET_OFFSET                          31:0
#define LWCFB0_VP9_SET_TILE_SIZE_BUF_OFFSET                                     (0x000005D0)
#define LWCFB0_VP9_SET_TILE_SIZE_BUF_OFFSET_OFFSET                              31:0
#define LWCFB0_VP9_SET_COL_MVWRITE_BUF_OFFSET                                   (0x000005D4)
#define LWCFB0_VP9_SET_COL_MVWRITE_BUF_OFFSET_OFFSET                            31:0
#define LWCFB0_VP9_SET_COL_MVREAD_BUF_OFFSET                                    (0x000005D8)
#define LWCFB0_VP9_SET_COL_MVREAD_BUF_OFFSET_OFFSET                             31:0
#define LWCFB0_VP9_SET_FILTER_BUFFER_OFFSET                                     (0x000005DC)
#define LWCFB0_VP9_SET_FILTER_BUFFER_OFFSET_OFFSET                              31:0
#define LWCFB0_VP9_PARSER_SET_PIC_SETUP_OFFSET                                  (0x000005E0)
#define LWCFB0_VP9_PARSER_SET_PIC_SETUP_OFFSET_OFFSET                           31:0
#define LWCFB0_VP9_PARSER_SET_PREV_PIC_SETUP_OFFSET                             (0x000005E4)
#define LWCFB0_VP9_PARSER_SET_PREV_PIC_SETUP_OFFSET_OFFSET                      31:0
#define LWCFB0_VP9_PARSER_SET_PROB_TAB_BUF_OFFSET                               (0x000005E8)
#define LWCFB0_VP9_PARSER_SET_PROB_TAB_BUF_OFFSET_OFFSET                        31:0
#define LWCFB0_VP9_SET_HINT_DUMP_BUF_OFFSET                                     (0x000005EC)
#define LWCFB0_VP9_SET_HINT_DUMP_BUF_OFFSET_OFFSET                              31:0
#define LWCFB0_PASS1_SET_CLEAR_HEADER_OFFSET                                    (0x00000600)
#define LWCFB0_PASS1_SET_CLEAR_HEADER_OFFSET_OFFSET                             31:0
#define LWCFB0_PASS1_SET_RE_ENCRYPT_OFFSET                                      (0x00000604)
#define LWCFB0_PASS1_SET_RE_ENCRYPT_OFFSET_OFFSET                               31:0
#define LWCFB0_PASS1_SET_VP8_TOKEN_OFFSET                                       (0x00000608)
#define LWCFB0_PASS1_SET_VP8_TOKEN_OFFSET_OFFSET                                31:0
#define LWCFB0_PASS1_SET_INPUT_DATA_OFFSET                                      (0x0000060C)
#define LWCFB0_PASS1_SET_INPUT_DATA_OFFSET_OFFSET                               31:0
#define LWCFB0_PASS1_SET_OUTPUT_DATA_SIZE_OFFSET                                (0x00000610)
#define LWCFB0_PASS1_SET_OUTPUT_DATA_SIZE_OFFSET_OFFSET                         31:0
#define LWCFB0_AV1_SET_PROB_TAB_READ_BUF_OFFSET                                 (0x00000640)
#define LWCFB0_AV1_SET_PROB_TAB_READ_BUF_OFFSET_OFFSET                          31:0
#define LWCFB0_AV1_SET_PROB_TAB_WRITE_BUF_OFFSET                                (0x00000644)
#define LWCFB0_AV1_SET_PROB_TAB_WRITE_BUF_OFFSET_OFFSET                         31:0
#define LWCFB0_AV1_SET_SEGMENT_READ_BUF_OFFSET                                  (0x00000648)
#define LWCFB0_AV1_SET_SEGMENT_READ_BUF_OFFSET_OFFSET                           31:0
#define LWCFB0_AV1_SET_SEGMENT_WRITE_BUF_OFFSET                                 (0x0000064C)
#define LWCFB0_AV1_SET_SEGMENT_WRITE_BUF_OFFSET_OFFSET                          31:0
#define LWCFB0_AV1_SET_COL_MV0_READ_BUF_OFFSET                                  (0x00000650)
#define LWCFB0_AV1_SET_COL_MV0_READ_BUF_OFFSET_OFFSET                           31:0
#define LWCFB0_AV1_SET_COL_MV1_READ_BUF_OFFSET                                  (0x00000654)
#define LWCFB0_AV1_SET_COL_MV1_READ_BUF_OFFSET_OFFSET                           31:0
#define LWCFB0_AV1_SET_COL_MV2_READ_BUF_OFFSET                                  (0x00000658)
#define LWCFB0_AV1_SET_COL_MV2_READ_BUF_OFFSET_OFFSET                           31:0
#define LWCFB0_AV1_SET_COL_MVWRITE_BUF_OFFSET                                   (0x0000065C)
#define LWCFB0_AV1_SET_COL_MVWRITE_BUF_OFFSET_OFFSET                            31:0
#define LWCFB0_AV1_SET_GLOBAL_MODEL_BUF_OFFSET                                  (0x00000660)
#define LWCFB0_AV1_SET_GLOBAL_MODEL_BUF_OFFSET_OFFSET                           31:0
#define LWCFB0_AV1_SET_FILM_GRAIN_BUF_OFFSET                                    (0x00000664)
#define LWCFB0_AV1_SET_FILM_GRAIN_BUF_OFFSET_OFFSET                             31:0
#define LWCFB0_AV1_SET_TILE_STREAM_INFO_BUF_OFFSET                              (0x00000668)
#define LWCFB0_AV1_SET_TILE_STREAM_INFO_BUF_OFFSET_OFFSET                       31:0
#define LWCFB0_AV1_SET_SUB_STREAM_ENTRY_BUF_OFFSET                              (0x0000066C)
#define LWCFB0_AV1_SET_SUB_STREAM_ENTRY_BUF_OFFSET_OFFSET                       31:0
#define LWCFB0_AV1_SET_HINT_DUMP_BUF_OFFSET                                     (0x00000670)
#define LWCFB0_AV1_SET_HINT_DUMP_BUF_OFFSET_OFFSET                              31:0
#define LWCFB0_NEW_H264_SET_SCALING_LIST_OFFSET                                 (0x00000680)
#define LWCFB0_NEW_H264_SET_SCALING_LIST_OFFSET_OFFSET                          31:0
#define LWCFB0_SET_CONTENT_INITIAL_VECTOR(b)                                    (0x00000C00 + (b)*0x00000004)
#define LWCFB0_SET_CONTENT_INITIAL_VECTOR_VALUE                                 31:0
#define LWCFB0_SET_CTL_COUNT                                                    (0x00000C10)
#define LWCFB0_SET_CTL_COUNT_VALUE                                              31:0
#define LWCFB0_SET_UPPER_SRC                                                    (0x00000C14)
#define LWCFB0_SET_UPPER_SRC_OFFSET                                             7:0
#define LWCFB0_SET_LOWER_SRC                                                    (0x00000C18)
#define LWCFB0_SET_LOWER_SRC_OFFSET                                             31:0
#define LWCFB0_SET_UPPER_DST                                                    (0x00000C1C)
#define LWCFB0_SET_UPPER_DST_OFFSET                                             7:0
#define LWCFB0_SET_LOWER_DST                                                    (0x00000C20)
#define LWCFB0_SET_LOWER_DST_OFFSET                                             31:0
#define LWCFB0_SET_BLOCK_COUNT                                                  (0x00000C24)
#define LWCFB0_SET_BLOCK_COUNT_VALUE                                            31:0
#define LWCFB0_PR_SET_REQUEST_BUF_OFFSET                                        (0x00000D00)
#define LWCFB0_PR_SET_REQUEST_BUF_OFFSET_OFFSET                                 31:0
#define LWCFB0_PR_SET_REQUEST_BUF_SIZE                                          (0x00000D04)
#define LWCFB0_PR_SET_REQUEST_BUF_SIZE_SIZE                                     31:0
#define LWCFB0_PR_SET_RESPONSE_BUF_OFFSET                                       (0x00000D08)
#define LWCFB0_PR_SET_RESPONSE_BUF_OFFSET_OFFSET                                31:0
#define LWCFB0_PR_SET_RESPONSE_BUF_SIZE                                         (0x00000D0C)
#define LWCFB0_PR_SET_RESPONSE_BUF_SIZE_SIZE                                    31:0
#define LWCFB0_PR_SET_REQUEST_MESSAGE_BUF_OFFSET                                (0x00000D10)
#define LWCFB0_PR_SET_REQUEST_MESSAGE_BUF_OFFSET_OFFSET                         31:0
#define LWCFB0_PR_SET_RESPONSE_MESSAGE_BUF_OFFSET                               (0x00000D14)
#define LWCFB0_PR_SET_RESPONSE_MESSAGE_BUF_OFFSET_OFFSET                        31:0
#define LWCFB0_PR_SET_LOCAL_DECRYPT_BUF_OFFSET                                  (0x00000D18)
#define LWCFB0_PR_SET_LOCAL_DECRYPT_BUF_OFFSET_OFFSET                           31:0
#define LWCFB0_PR_SET_LOCAL_DECRYPT_BUF_SIZE                                    (0x00000D1C)
#define LWCFB0_PR_SET_LOCAL_DECRYPT_BUF_SIZE_SIZE                               31:0
#define LWCFB0_PR_SET_CONTENT_DECRYPT_INFO_BUF_OFFSET                           (0x00000D20)
#define LWCFB0_PR_SET_CONTENT_DECRYPT_INFO_BUF_OFFSET_OFFSET                    31:0
#define LWCFB0_PR_SET_REENCRYPTED_BITSTREAM_BUF_OFFSET                          (0x00000D24)
#define LWCFB0_PR_SET_REENCRYPTED_BITSTREAM_BUF_OFFSET_OFFSET                   31:0
#define LWCFB0_DH_KE_SET_CHALLENGE_BUF_OFFSET                                   (0x00000E00)
#define LWCFB0_DH_KE_SET_CHALLENGE_BUF_OFFSET_OFFSET                            31:0
#define LWCFB0_DH_KE_SET_RESPONSE_BUF_OFFSET                                    (0x00000E04)
#define LWCFB0_DH_KE_SET_RESPONSE_BUF_OFFSET_OFFSET                             31:0
#define LWCFB0_SET_SESSION_KEY(b)                                               (0x00000F00 + (b)*0x00000004)
#define LWCFB0_SET_SESSION_KEY_VALUE                                            31:0
#define LWCFB0_SET_CONTENT_KEY(b)                                               (0x00000F10 + (b)*0x00000004)
#define LWCFB0_SET_CONTENT_KEY_VALUE                                            31:0
#define LWCFB0_PM_TRIGGER_END                                                   (0x00001114)
#define LWCFB0_PM_TRIGGER_END_V                                                 31:0

#define LWCFB0_ERROR_NONE                                                       (0x00000000)
#define LWCFB0_OS_ERROR_EXELWTE_INSUFFICIENT_DATA                               (0x00000001)
#define LWCFB0_OS_ERROR_SEMAPHORE_INSUFFICIENT_DATA                             (0x00000002)
#define LWCFB0_OS_ERROR_ILWALID_METHOD                                          (0x00000003)
#define LWCFB0_OS_ERROR_ILWALID_DMA_PAGE                                        (0x00000004)
#define LWCFB0_OS_ERROR_UNHANDLED_INTERRUPT                                     (0x00000005)
#define LWCFB0_OS_ERROR_EXCEPTION                                               (0x00000006)
#define LWCFB0_OS_ERROR_ILWALID_CTXSW_REQUEST                                   (0x00000007)
#define LWCFB0_OS_ERROR_APPLICATION                                             (0x00000008)
#define LWCFB0_OS_ERROR_SW_BREAKPT                                              (0x00000009)
#define LWCFB0_OS_INTERRUPT_EXELWTE_AWAKEN                                      (0x00000100)
#define LWCFB0_OS_INTERRUPT_BACKEND_SEMAPHORE_AWAKEN                            (0x00000200)
#define LWCFB0_OS_INTERRUPT_CTX_ERROR_FBIF                                      (0x00000300)
#define LWCFB0_OS_INTERRUPT_LIMIT_VIOLATION                                     (0x00000400)
#define LWCFB0_OS_INTERRUPT_LIMIT_AND_FBIF_CTX_ERROR                            (0x00000500)
#define LWCFB0_OS_INTERRUPT_HALT_ENGINE                                         (0x00000600)
#define LWCFB0_OS_INTERRUPT_TRAP_NONSTALL                                       (0x00000700)
#define LWCFB0_H264_VLD_ERR_SEQ_DATA_INCONSISTENT                               (0x00004001)
#define LWCFB0_H264_VLD_ERR_PIC_DATA_INCONSISTENT                               (0x00004002)
#define LWCFB0_H264_VLD_ERR_SLC_DATA_BUF_ADDR_OUT_OF_BOUNDS                     (0x00004100)
#define LWCFB0_H264_VLD_ERR_BITSTREAM_ERROR                                     (0x00004101)
#define LWCFB0_H264_VLD_ERR_CTX_DMA_ID_CTRL_IN_ILWALID                          (0x000041F8)
#define LWCFB0_H264_VLD_ERR_SLC_HDR_OUT_SIZE_NOT_MULT256                        (0x00004200)
#define LWCFB0_H264_VLD_ERR_SLC_DATA_OUT_SIZE_NOT_MULT256                       (0x00004201)
#define LWCFB0_H264_VLD_ERR_CTX_DMA_ID_FLOW_CTRL_ILWALID                        (0x00004203)
#define LWCFB0_H264_VLD_ERR_CTX_DMA_ID_SLC_HDR_OUT_ILWALID                      (0x00004204)
#define LWCFB0_H264_VLD_ERR_SLC_HDR_OUT_BUF_TOO_SMALL                           (0x00004205)
#define LWCFB0_H264_VLD_ERR_SLC_HDR_OUT_BUF_ALREADY_VALID                       (0x00004206)
#define LWCFB0_H264_VLD_ERR_SLC_DATA_OUT_BUF_TOO_SMALL                          (0x00004207)
#define LWCFB0_H264_VLD_ERR_DATA_BUF_CNT_TOO_SMALL                              (0x00004208)
#define LWCFB0_H264_VLD_ERR_BITSTREAM_EMPTY                                     (0x00004209)
#define LWCFB0_H264_VLD_ERR_FRAME_WIDTH_TOO_LARGE                               (0x0000420A)
#define LWCFB0_H264_VLD_ERR_FRAME_HEIGHT_TOO_LARGE                              (0x0000420B)
#define LWCFB0_H264_VLD_ERR_HIST_BUF_TOO_SMALL                                  (0x00004300)
#define LWCFB0_VC1_VLD_ERR_PIC_DATA_BUF_ADDR_OUT_OF_BOUND                       (0x00005100)
#define LWCFB0_VC1_VLD_ERR_BITSTREAM_ERROR                                      (0x00005101)
#define LWCFB0_VC1_VLD_ERR_PIC_HDR_OUT_SIZE_NOT_MULT256                         (0x00005200)
#define LWCFB0_VC1_VLD_ERR_PIC_DATA_OUT_SIZE_NOT_MULT256                        (0x00005201)
#define LWCFB0_VC1_VLD_ERR_CTX_DMA_ID_CTRL_IN_ILWALID                           (0x00005202)
#define LWCFB0_VC1_VLD_ERR_CTX_DMA_ID_FLOW_CTRL_ILWALID                         (0x00005203)
#define LWCFB0_VC1_VLD_ERR_CTX_DMA_ID_PIC_HDR_OUT_ILWALID                       (0x00005204)
#define LWCFB0_VC1_VLD_ERR_SLC_HDR_OUT_BUF_TOO_SMALL                            (0x00005205)
#define LWCFB0_VC1_VLD_ERR_PIC_HDR_OUT_BUF_ALREADY_VALID                        (0x00005206)
#define LWCFB0_VC1_VLD_ERR_PIC_DATA_OUT_BUF_TOO_SMALL                           (0x00005207)
#define LWCFB0_VC1_VLD_ERR_DATA_INFO_IN_BUF_TOO_SMALL                           (0x00005208)
#define LWCFB0_VC1_VLD_ERR_BITSTREAM_EMPTY                                      (0x00005209)
#define LWCFB0_VC1_VLD_ERR_FRAME_WIDTH_TOO_LARGE                                (0x0000520A)
#define LWCFB0_VC1_VLD_ERR_FRAME_HEIGHT_TOO_LARGE                               (0x0000520B)
#define LWCFB0_VC1_VLD_ERR_PIC_DATA_OUT_BUF_FULL_TIME_OUT                       (0x00005300)
#define LWCFB0_MPEG12_VLD_ERR_SLC_DATA_BUF_ADDR_OUT_OF_BOUNDS                   (0x00006100)
#define LWCFB0_MPEG12_VLD_ERR_BITSTREAM_ERROR                                   (0x00006101)
#define LWCFB0_MPEG12_VLD_ERR_SLC_DATA_OUT_SIZE_NOT_MULT256                     (0x00006200)
#define LWCFB0_MPEG12_VLD_ERR_CTX_DMA_ID_CTRL_IN_ILWALID                        (0x00006201)
#define LWCFB0_MPEG12_VLD_ERR_CTX_DMA_ID_FLOW_CTRL_ILWALID                      (0x00006202)
#define LWCFB0_MPEG12_VLD_ERR_SLC_DATA_OUT_BUF_TOO_SMALL                        (0x00006203)
#define LWCFB0_MPEG12_VLD_ERR_DATA_INFO_IN_BUF_TOO_SMALL                        (0x00006204)
#define LWCFB0_MPEG12_VLD_ERR_BITSTREAM_EMPTY                                   (0x00006205)
#define LWCFB0_MPEG12_VLD_ERR_ILWALID_PIC_STRUCTURE                             (0x00006206)
#define LWCFB0_MPEG12_VLD_ERR_ILWALID_PIC_CODING_TYPE                           (0x00006207)
#define LWCFB0_MPEG12_VLD_ERR_FRAME_WIDTH_TOO_LARGE                             (0x00006208)
#define LWCFB0_MPEG12_VLD_ERR_FRAME_HEIGHT_TOO_LARGE                            (0x00006209)
#define LWCFB0_MPEG12_VLD_ERR_SLC_DATA_OUT_BUF_FULL_TIME_OUT                    (0x00006300)
#define LWCFB0_CMN_VLD_ERR_PDEC_RETURNED_ERROR                                  (0x00007101)
#define LWCFB0_CMN_VLD_ERR_EDOB_FLUSH_TIME_OUT                                  (0x00007102)
#define LWCFB0_CMN_VLD_ERR_EDOB_REWIND_TIME_OUT                                 (0x00007103)
#define LWCFB0_CMN_VLD_ERR_VLD_WD_TIME_OUT                                      (0x00007104)
#define LWCFB0_CMN_VLD_ERR_NUM_SLICES_ZERO                                      (0x00007105)
#define LWCFB0_MPEG4_VLD_ERR_PIC_DATA_BUF_ADDR_OUT_OF_BOUND                     (0x00008100)
#define LWCFB0_MPEG4_VLD_ERR_BITSTREAM_ERROR                                    (0x00008101)
#define LWCFB0_MPEG4_VLD_ERR_PIC_HDR_OUT_SIZE_NOT_MULT256                       (0x00008200)
#define LWCFB0_MPEG4_VLD_ERR_PIC_DATA_OUT_SIZE_NOT_MULT256                      (0x00008201)
#define LWCFB0_MPEG4_VLD_ERR_CTX_DMA_ID_CTRL_IN_ILWALID                         (0x00008202)
#define LWCFB0_MPEG4_VLD_ERR_CTX_DMA_ID_FLOW_CTRL_ILWALID                       (0x00008203)
#define LWCFB0_MPEG4_VLD_ERR_CTX_DMA_ID_PIC_HDR_OUT_ILWALID                     (0x00008204)
#define LWCFB0_MPEG4_VLD_ERR_SLC_HDR_OUT_BUF_TOO_SMALL                          (0x00008205)
#define LWCFB0_MPEG4_VLD_ERR_PIC_HDR_OUT_BUF_ALREADY_VALID                      (0x00008206)
#define LWCFB0_MPEG4_VLD_ERR_PIC_DATA_OUT_BUF_TOO_SMALL                         (0x00008207)
#define LWCFB0_MPEG4_VLD_ERR_DATA_INFO_IN_BUF_TOO_SMALL                         (0x00008208)
#define LWCFB0_MPEG4_VLD_ERR_BITSTREAM_EMPTY                                    (0x00008209)
#define LWCFB0_MPEG4_VLD_ERR_FRAME_WIDTH_TOO_LARGE                              (0x0000820A)
#define LWCFB0_MPEG4_VLD_ERR_FRAME_HEIGHT_TOO_LARGE                             (0x0000820B)
#define LWCFB0_MPEG4_VLD_ERR_PIC_DATA_OUT_BUF_FULL_TIME_OUT                     (0x00051E01)
#define LWCFB0_DEC_ERROR_MPEG12_APPTIMER_EXPIRED                                (0xDEC10001)
#define LWCFB0_DEC_ERROR_MPEG12_MVTIMER_EXPIRED                                 (0xDEC10002)
#define LWCFB0_DEC_ERROR_MPEG12_ILWALID_TOKEN                                   (0xDEC10003)
#define LWCFB0_DEC_ERROR_MPEG12_SLICEDATA_MISSING                               (0xDEC10004)
#define LWCFB0_DEC_ERROR_MPEG12_HWERR_INTERRUPT                                 (0xDEC10005)
#define LWCFB0_DEC_ERROR_MPEG12_DETECTED_VLD_FAILURE                            (0xDEC10006)
#define LWCFB0_DEC_ERROR_MPEG12_PICTURE_INIT                                    (0xDEC10100)
#define LWCFB0_DEC_ERROR_MPEG12_STATEMACHINE_FAILURE                            (0xDEC10101)
#define LWCFB0_DEC_ERROR_MPEG12_ILWALID_CTXID_PIC                               (0xDEC10901)
#define LWCFB0_DEC_ERROR_MPEG12_ILWALID_CTXID_UCODE                             (0xDEC10902)
#define LWCFB0_DEC_ERROR_MPEG12_ILWALID_CTXID_FC                                (0xDEC10903)
#define LWCFB0_DEC_ERROR_MPEG12_ILWALID_CTXID_SLH                               (0xDEC10904)
#define LWCFB0_DEC_ERROR_MPEG12_ILWALID_UCODE_SIZE                              (0xDEC10905)
#define LWCFB0_DEC_ERROR_MPEG12_ILWALID_SLICE_COUNT                             (0xDEC10906)
#define LWCFB0_DEC_ERROR_VC1_APPTIMER_EXPIRED                                   (0xDEC20001)
#define LWCFB0_DEC_ERROR_VC1_MVTIMER_EXPIRED                                    (0xDEC20002)
#define LWCFB0_DEC_ERROR_VC1_ILWALID_TOKEN                                      (0xDEC20003)
#define LWCFB0_DEC_ERROR_VC1_SLICEDATA_MISSING                                  (0xDEC20004)
#define LWCFB0_DEC_ERROR_VC1_HWERR_INTERRUPT                                    (0xDEC20005)
#define LWCFB0_DEC_ERROR_VC1_DETECTED_VLD_FAILURE                               (0xDEC20006)
#define LWCFB0_DEC_ERROR_VC1_TIMEOUT_POLLING_FOR_DATA                           (0xDEC20007)
#define LWCFB0_DEC_ERROR_VC1_PDEC_PIC_END_UNALIGNED                             (0xDEC20008)
#define LWCFB0_DEC_ERROR_VC1_WDTIMER_EXPIRED                                    (0xDEC20009)
#define LWCFB0_DEC_ERROR_VC1_ERRINTSTART                                        (0xDEC20010)
#define LWCFB0_DEC_ERROR_VC1_IQT_ERRINT                                         (0xDEC20011)
#define LWCFB0_DEC_ERROR_VC1_MC_ERRINT                                          (0xDEC20012)
#define LWCFB0_DEC_ERROR_VC1_MC_IQT_ERRINT                                      (0xDEC20013)
#define LWCFB0_DEC_ERROR_VC1_REC_ERRINT                                         (0xDEC20014)
#define LWCFB0_DEC_ERROR_VC1_REC_IQT_ERRINT                                     (0xDEC20015)
#define LWCFB0_DEC_ERROR_VC1_REC_MC_ERRINT                                      (0xDEC20016)
#define LWCFB0_DEC_ERROR_VC1_REC_MC_IQT_ERRINT                                  (0xDEC20017)
#define LWCFB0_DEC_ERROR_VC1_DBF_ERRINT                                         (0xDEC20018)
#define LWCFB0_DEC_ERROR_VC1_DBF_IQT_ERRINT                                     (0xDEC20019)
#define LWCFB0_DEC_ERROR_VC1_DBF_MC_ERRINT                                      (0xDEC2001A)
#define LWCFB0_DEC_ERROR_VC1_DBF_MC_IQT_ERRINT                                  (0xDEC2001B)
#define LWCFB0_DEC_ERROR_VC1_DBF_REC_ERRINT                                     (0xDEC2001C)
#define LWCFB0_DEC_ERROR_VC1_DBF_REC_IQT_ERRINT                                 (0xDEC2001D)
#define LWCFB0_DEC_ERROR_VC1_DBF_REC_MC_ERRINT                                  (0xDEC2001E)
#define LWCFB0_DEC_ERROR_VC1_DBF_REC_MC_IQT_ERRINT                              (0xDEC2001F)
#define LWCFB0_DEC_ERROR_VC1_PICTURE_INIT                                       (0xDEC20100)
#define LWCFB0_DEC_ERROR_VC1_STATEMACHINE_FAILURE                               (0xDEC20101)
#define LWCFB0_DEC_ERROR_VC1_ILWALID_CTXID_PIC                                  (0xDEC20901)
#define LWCFB0_DEC_ERROR_VC1_ILWALID_CTXID_UCODE                                (0xDEC20902)
#define LWCFB0_DEC_ERROR_VC1_ILWALID_CTXID_FC                                   (0xDEC20903)
#define LWCFB0_DEC_ERROR_VC1_ILWAILD_CTXID_SLH                                  (0xDEC20904)
#define LWCFB0_DEC_ERROR_VC1_ILWALID_UCODE_SIZE                                 (0xDEC20905)
#define LWCFB0_DEC_ERROR_VC1_ILWALID_SLICE_COUNT                                (0xDEC20906)
#define LWCFB0_DEC_ERROR_H264_APPTIMER_EXPIRED                                  (0xDEC30001)
#define LWCFB0_DEC_ERROR_H264_MVTIMER_EXPIRED                                   (0xDEC30002)
#define LWCFB0_DEC_ERROR_H264_ILWALID_TOKEN                                     (0xDEC30003)
#define LWCFB0_DEC_ERROR_H264_SLICEDATA_MISSING                                 (0xDEC30004)
#define LWCFB0_DEC_ERROR_H264_HWERR_INTERRUPT                                   (0xDEC30005)
#define LWCFB0_DEC_ERROR_H264_DETECTED_VLD_FAILURE                              (0xDEC30006)
#define LWCFB0_DEC_ERROR_H264_ERRINTSTART                                       (0xDEC30010)
#define LWCFB0_DEC_ERROR_H264_IQT_ERRINT                                        (0xDEC30011)
#define LWCFB0_DEC_ERROR_H264_MC_ERRINT                                         (0xDEC30012)
#define LWCFB0_DEC_ERROR_H264_MC_IQT_ERRINT                                     (0xDEC30013)
#define LWCFB0_DEC_ERROR_H264_REC_ERRINT                                        (0xDEC30014)
#define LWCFB0_DEC_ERROR_H264_REC_IQT_ERRINT                                    (0xDEC30015)
#define LWCFB0_DEC_ERROR_H264_REC_MC_ERRINT                                     (0xDEC30016)
#define LWCFB0_DEC_ERROR_H264_REC_MC_IQT_ERRINT                                 (0xDEC30017)
#define LWCFB0_DEC_ERROR_H264_DBF_ERRINT                                        (0xDEC30018)
#define LWCFB0_DEC_ERROR_H264_DBF_IQT_ERRINT                                    (0xDEC30019)
#define LWCFB0_DEC_ERROR_H264_DBF_MC_ERRINT                                     (0xDEC3001A)
#define LWCFB0_DEC_ERROR_H264_DBF_MC_IQT_ERRINT                                 (0xDEC3001B)
#define LWCFB0_DEC_ERROR_H264_DBF_REC_ERRINT                                    (0xDEC3001C)
#define LWCFB0_DEC_ERROR_H264_DBF_REC_IQT_ERRINT                                (0xDEC3001D)
#define LWCFB0_DEC_ERROR_H264_DBF_REC_MC_ERRINT                                 (0xDEC3001E)
#define LWCFB0_DEC_ERROR_H264_DBF_REC_MC_IQT_ERRINT                             (0xDEC3001F)
#define LWCFB0_DEC_ERROR_H264_PICTURE_INIT                                      (0xDEC30100)
#define LWCFB0_DEC_ERROR_H264_STATEMACHINE_FAILURE                              (0xDEC30101)
#define LWCFB0_DEC_ERROR_H264_ILWALID_CTXID_PIC                                 (0xDEC30901)
#define LWCFB0_DEC_ERROR_H264_ILWALID_CTXID_UCODE                               (0xDEC30902)
#define LWCFB0_DEC_ERROR_H264_ILWALID_CTXID_FC                                  (0xDEC30903)
#define LWCFB0_DEC_ERROR_H264_ILWALID_CTXID_SLH                                 (0xDEC30904)
#define LWCFB0_DEC_ERROR_H264_ILWALID_UCODE_SIZE                                (0xDEC30905)
#define LWCFB0_DEC_ERROR_H264_ILWALID_SLICE_COUNT                               (0xDEC30906)
#define LWCFB0_DEC_ERROR_MPEG4_APPTIMER_EXPIRED                                 (0xDEC40001)
#define LWCFB0_DEC_ERROR_MPEG4_MVTIMER_EXPIRED                                  (0xDEC40002)
#define LWCFB0_DEC_ERROR_MPEG4_ILWALID_TOKEN                                    (0xDEC40003)
#define LWCFB0_DEC_ERROR_MPEG4_SLICEDATA_MISSING                                (0xDEC40004)
#define LWCFB0_DEC_ERROR_MPEG4_HWERR_INTERRUPT                                  (0xDEC40005)
#define LWCFB0_DEC_ERROR_MPEG4_DETECTED_VLD_FAILURE                             (0xDEC40006)
#define LWCFB0_DEC_ERROR_MPEG4_TIMEOUT_POLLING_FOR_DATA                         (0xDEC40007)
#define LWCFB0_DEC_ERROR_MPEG4_PDEC_PIC_END_UNALIGNED                           (0xDEC40008)
#define LWCFB0_DEC_ERROR_MPEG4_WDTIMER_EXPIRED                                  (0xDEC40009)
#define LWCFB0_DEC_ERROR_MPEG4_ERRINTSTART                                      (0xDEC40010)
#define LWCFB0_DEC_ERROR_MPEG4_IQT_ERRINT                                       (0xDEC40011)
#define LWCFB0_DEC_ERROR_MPEG4_MC_ERRINT                                        (0xDEC40012)
#define LWCFB0_DEC_ERROR_MPEG4_MC_IQT_ERRINT                                    (0xDEC40013)
#define LWCFB0_DEC_ERROR_MPEG4_REC_ERRINT                                       (0xDEC40014)
#define LWCFB0_DEC_ERROR_MPEG4_REC_IQT_ERRINT                                   (0xDEC40015)
#define LWCFB0_DEC_ERROR_MPEG4_REC_MC_ERRINT                                    (0xDEC40016)
#define LWCFB0_DEC_ERROR_MPEG4_REC_MC_IQT_ERRINT                                (0xDEC40017)
#define LWCFB0_DEC_ERROR_MPEG4_DBF_ERRINT                                       (0xDEC40018)
#define LWCFB0_DEC_ERROR_MPEG4_DBF_IQT_ERRINT                                   (0xDEC40019)
#define LWCFB0_DEC_ERROR_MPEG4_DBF_MC_ERRINT                                    (0xDEC4001A)
#define LWCFB0_DEC_ERROR_MPEG4_DBF_MC_IQT_ERRINT                                (0xDEC4001B)
#define LWCFB0_DEC_ERROR_MPEG4_DBF_REC_ERRINT                                   (0xDEC4001C)
#define LWCFB0_DEC_ERROR_MPEG4_DBF_REC_IQT_ERRINT                               (0xDEC4001D)
#define LWCFB0_DEC_ERROR_MPEG4_DBF_REC_MC_ERRINT                                (0xDEC4001E)
#define LWCFB0_DEC_ERROR_MPEG4_DBF_REC_MC_IQT_ERRINT                            (0xDEC4001F)
#define LWCFB0_DEC_ERROR_MPEG4_PICTURE_INIT                                     (0xDEC40100)
#define LWCFB0_DEC_ERROR_MPEG4_STATEMACHINE_FAILURE                             (0xDEC40101)
#define LWCFB0_DEC_ERROR_MPEG4_ILWALID_CTXID_PIC                                (0xDEC40901)
#define LWCFB0_DEC_ERROR_MPEG4_ILWALID_CTXID_UCODE                              (0xDEC40902)
#define LWCFB0_DEC_ERROR_MPEG4_ILWALID_CTXID_FC                                 (0xDEC40903)
#define LWCFB0_DEC_ERROR_MPEG4_ILWALID_CTXID_SLH                                (0xDEC40904)
#define LWCFB0_DEC_ERROR_MPEG4_ILWALID_UCODE_SIZE                               (0xDEC40905)
#define LWCFB0_DEC_ERROR_MPEG4_ILWALID_SLICE_COUNT                              (0xDEC40906)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // clcfb0_h

