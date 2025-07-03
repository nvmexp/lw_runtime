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


#ifndef CL95B1_H
#define CL95B1_H

#include "lwtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LW95B1_VIDEO_MSVLD                                                               (0x000095B1)

typedef volatile struct _cl95b1_tag0 {
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
    LwV32 H264SetSeqPicCtrlInOffset;                                            // 0x00000400 - 0x00000403
    LwV32 H264SetSliceHdrOutBufOffset;                                          // 0x00000404 - 0x00000407
    LwV32 H264SetSliceHdrOutBufSize;                                            // 0x00000408 - 0x0000040B
    LwV32 H264SetSliceDataOutBufOffset;                                         // 0x0000040C - 0x0000040F
    LwV32 H264SetSliceDataOutBufSize;                                           // 0x00000410 - 0x00000413
    LwV32 H264SetMBHistBufOffset;                                               // 0x00000414 - 0x00000417
    LwV32 H264SetMBHistBufSize;                                                 // 0x00000418 - 0x0000041B
    LwV32 Reserved06[0x1];
    LwV32 H264SetBaseSlcHdrOffset;                                              // 0x00000420 - 0x00000423
    LwV32 Reserved07[0x37];
    LwV32 Vc1SetSeqCtrlInOffset;                                                // 0x00000500 - 0x00000503
    LwV32 Vc1SetPicHdrOutBufOffset;                                             // 0x00000504 - 0x00000507
    LwV32 Vc1SetPicDataOutBufOffset;                                            // 0x00000508 - 0x0000050B
    LwV32 Vc1SetPicDataOutBufSize;                                              // 0x0000050C - 0x0000050F
    LwV32 Vc1SetPicScratchBufOffset;                                            // 0x00000510 - 0x00000513
    LwV32 Vc1SetPicScratchBufSize;                                              // 0x00000514 - 0x00000517
    LwV32 Reserved08[0x3A];
    LwV32 Mpeg12SetSeqPicCtrlInOffset;                                          // 0x00000600 - 0x00000603
    LwV32 Mpeg12SetSlcHdrOutBufOffset;                                          // 0x00000604 - 0x00000607
    LwV32 Mpeg12SetSlcDataOutBufOffset;                                         // 0x00000608 - 0x0000060B
    LwV32 Mpeg12SetSlcDataOutBufSize;                                           // 0x0000060C - 0x0000060F
    LwV32 Reserved09[0x3C];
    LwV32 SetControlParams;                                                     // 0x00000700 - 0x00000703
    LwV32 SetDataInfoBufferInOffset;                                            // 0x00000704 - 0x00000707
    LwV32 SetInBufBaseOffset;                                                   // 0x00000708 - 0x0000070B
    LwV32 SetFlowCtrlOutOffset;                                                 // 0x0000070C - 0x0000070F
    LwV32 SetPictureIndex;                                                      // 0x00000710 - 0x00000713
    LwV32 Reserved10[0x13B];
    LwV32 SetContentInitialVector[4];                                           // 0x00000C00 - 0x00000C0F
    LwV32 SetCtlCount;                                                          // 0x00000C10 - 0x00000C13
    LwV32 SetMdecH2MKey;                                                        // 0x00000C14 - 0x00000C17
    LwV32 SetMdecM2HKey;                                                        // 0x00000C18 - 0x00000C1B
    LwV32 SetMdecFrameKey;                                                      // 0x00000C1C - 0x00000C1F
    LwV32 SetUpperSrc;                                                          // 0x00000C20 - 0x00000C23
    LwV32 SetLowerSrc;                                                          // 0x00000C24 - 0x00000C27
    LwV32 SetUpperDst;                                                          // 0x00000C28 - 0x00000C2B
    LwV32 SetLowerDst;                                                          // 0x00000C2C - 0x00000C2F
    LwV32 SetUpperCtl;                                                          // 0x00000C30 - 0x00000C33
    LwV32 SetLowerCtl;                                                          // 0x00000C34 - 0x00000C37
    LwV32 SetBlockCount;                                                        // 0x00000C38 - 0x00000C3B
    LwV32 SetStretchMask;                                                       // 0x00000C3C - 0x00000C3F
    LwV32 Reserved11[0x30];
    LwV32 SetUpperFlowCtrlInselwre;                                             // 0x00000D00 - 0x00000D03
    LwV32 SetLowerFlowCtrlInselwre;                                             // 0x00000D04 - 0x00000D07
    LwV32 Reserved12[0x2];
    LwV32 SetUcodeLoaderParams;                                                 // 0x00000D10 - 0x00000D13
    LwV32 Reserved13[0x1];
    LwV32 SetUpperFlowCtrlSelwre;                                               // 0x00000D18 - 0x00000D1B
    LwV32 SetLowerFlowCtrlSelwre;                                               // 0x00000D1C - 0x00000D1F
    LwV32 Reserved14[0x5];
    LwV32 SetUcodeLoaderOffset;                                                 // 0x00000D34 - 0x00000D37
    LwV32 Reserved15[0x32];
    LwV32 Mp4SetSeqCtrlInOffset;                                                // 0x00000E00 - 0x00000E03
    LwV32 Mp4SetPicHdrOutBufOffset;                                             // 0x00000E04 - 0x00000E07
    LwV32 Mp4SetPicDataOutBufOffset;                                            // 0x00000E08 - 0x00000E0B
    LwV32 Mp4SetPicDataOutBufSize;                                              // 0x00000E0C - 0x00000E0F
    LwV32 Mp4SetPicScratchBufOffset;                                            // 0x00000E10 - 0x00000E13
    LwV32 Mp4SetPicScratchBufSize;                                              // 0x00000E14 - 0x00000E17
    LwV32 Reserved16[0x3A];
    LwV32 SetSessionKey[4];                                                     // 0x00000F00 - 0x00000F0F
    LwV32 SetContentKey[4];                                                     // 0x00000F10 - 0x00000F1F
    LwV32 Reserved17[0x7D];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved18[0x3BA];
} LW95B1_VIDEO_MSVLDControlPio;

#define LW95B1_NOP                                                              (0x00000100)
#define LW95B1_NOP_PARAMETER                                                    31:0
#define LW95B1_NOP_PARAMETER_HIGH_FIELD                                         31U
#define LW95B1_NOP_PARAMETER_LOW_FIELD                                          0U
#define LW95B1_PM_TRIGGER                                                       (0x00000140)
#define LW95B1_PM_TRIGGER_V                                                     31:0
#define LW95B1_PM_TRIGGER_V_HIGH_FIELD                                          31U
#define LW95B1_PM_TRIGGER_V_LOW_FIELD                                           0U
#define LW95B1_SET_APPLICATION_ID                                               (0x00000200)
#define LW95B1_SET_APPLICATION_ID_ID                                            31:0
#define LW95B1_SET_APPLICATION_ID_ID_HIGH_FIELD                                 31U
#define LW95B1_SET_APPLICATION_ID_ID_LOW_FIELD                                  0U
#define LW95B1_SET_APPLICATION_ID_ID_MPEG12                                     (0x00000001)
#define LW95B1_SET_APPLICATION_ID_ID_VC1                                        (0x00000002)
#define LW95B1_SET_APPLICATION_ID_ID_H264                                       (0x00000003)
#define LW95B1_SET_APPLICATION_ID_ID_MPEG4                                      (0x00000004)
#define LW95B1_SET_APPLICATION_ID_ID_CTR64                                      (0x00000005)
#define LW95B1_SET_APPLICATION_ID_ID_STRETCH_CTR64                              (0x00000006)
#define LW95B1_SET_APPLICATION_ID_ID_MDEC_LEGACY                                (0x00000007)
#define LW95B1_SET_APPLICATION_ID_ID_UCODE_LOADER                               (0x00000008)
#define LW95B1_SET_APPLICATION_ID_ID_SCRAMBLER                                  (0x00000009)
#define LW95B1_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LW95B1_SET_WATCHDOG_TIMER_TIMER                                         31:0
#define LW95B1_SET_WATCHDOG_TIMER_TIMER_HIGH_FIELD                              31U
#define LW95B1_SET_WATCHDOG_TIMER_TIMER_LOW_FIELD                               0U
#define LW95B1_SEMAPHORE_A                                                      (0x00000240)
#define LW95B1_SEMAPHORE_A_UPPER                                                7:0
#define LW95B1_SEMAPHORE_A_UPPER_HIGH_FIELD                                     7U
#define LW95B1_SEMAPHORE_A_UPPER_LOW_FIELD                                      0U
#define LW95B1_SEMAPHORE_B                                                      (0x00000244)
#define LW95B1_SEMAPHORE_B_LOWER                                                31:0
#define LW95B1_SEMAPHORE_B_LOWER_HIGH_FIELD                                     31U
#define LW95B1_SEMAPHORE_B_LOWER_LOW_FIELD                                      0U
#define LW95B1_SEMAPHORE_C                                                      (0x00000248)
#define LW95B1_SEMAPHORE_C_PAYLOAD                                              31:0
#define LW95B1_SEMAPHORE_C_PAYLOAD_HIGH_FIELD                                   31U
#define LW95B1_SEMAPHORE_C_PAYLOAD_LOW_FIELD                                    0U
#define LW95B1_EXELWTE                                                          (0x00000300)
#define LW95B1_EXELWTE_NOTIFY                                                   0:0
#define LW95B1_EXELWTE_NOTIFY_HIGH_FIELD                                        0U
#define LW95B1_EXELWTE_NOTIFY_LOW_FIELD                                         0U
#define LW95B1_EXELWTE_NOTIFY_DISABLE                                           (0x00000000)
#define LW95B1_EXELWTE_NOTIFY_ENABLE                                            (0x00000001)
#define LW95B1_EXELWTE_NOTIFY_ON                                                1:1
#define LW95B1_EXELWTE_NOTIFY_ON_HIGH_FIELD                                     1U
#define LW95B1_EXELWTE_NOTIFY_ON_LOW_FIELD                                      1U
#define LW95B1_EXELWTE_NOTIFY_ON_END                                            (0x00000000)
#define LW95B1_EXELWTE_NOTIFY_ON_BEGIN                                          (0x00000001)
#define LW95B1_EXELWTE_PREDICATION                                              2:2
#define LW95B1_EXELWTE_PREDICATION_HIGH_FIELD                                   2U
#define LW95B1_EXELWTE_PREDICATION_LOW_FIELD                                    2U
#define LW95B1_EXELWTE_PREDICATION_DISABLE                                      (0x00000000)
#define LW95B1_EXELWTE_PREDICATION_ENABLE                                       (0x00000001)
#define LW95B1_EXELWTE_PREDICATION_OP                                           3:3
#define LW95B1_EXELWTE_PREDICATION_OP_HIGH_FIELD                                3U
#define LW95B1_EXELWTE_PREDICATION_OP_LOW_FIELD                                 3U
#define LW95B1_EXELWTE_PREDICATION_OP_EQUAL_ZERO                                (0x00000000)
#define LW95B1_EXELWTE_PREDICATION_OP_NOT_EQUAL_ZERO                            (0x00000001)
#define LW95B1_EXELWTE_AWAKEN                                                   8:8
#define LW95B1_EXELWTE_AWAKEN_HIGH_FIELD                                        8U
#define LW95B1_EXELWTE_AWAKEN_LOW_FIELD                                         8U
#define LW95B1_EXELWTE_AWAKEN_DISABLE                                           (0x00000000)
#define LW95B1_EXELWTE_AWAKEN_ENABLE                                            (0x00000001)
#define LW95B1_SEMAPHORE_D                                                      (0x00000304)
#define LW95B1_SEMAPHORE_D_STRUCTURE_SIZE                                       0:0
#define LW95B1_SEMAPHORE_D_STRUCTURE_SIZE_HIGH_FIELD                            0U
#define LW95B1_SEMAPHORE_D_STRUCTURE_SIZE_LOW_FIELD                             0U
#define LW95B1_SEMAPHORE_D_STRUCTURE_SIZE_ONE                                   (0x00000000)
#define LW95B1_SEMAPHORE_D_STRUCTURE_SIZE_FOUR                                  (0x00000001)
#define LW95B1_SEMAPHORE_D_AWAKEN_ENABLE                                        8:8
#define LW95B1_SEMAPHORE_D_AWAKEN_ENABLE_HIGH_FIELD                             8U
#define LW95B1_SEMAPHORE_D_AWAKEN_ENABLE_LOW_FIELD                              8U
#define LW95B1_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                  (0x00000000)
#define LW95B1_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                   (0x00000001)
#define LW95B1_SEMAPHORE_D_OPERATION                                            17:16
#define LW95B1_SEMAPHORE_D_OPERATION_HIGH_FIELD                                 17U
#define LW95B1_SEMAPHORE_D_OPERATION_LOW_FIELD                                  16U
#define LW95B1_SEMAPHORE_D_OPERATION_RELEASE                                    (0x00000000)
#define LW95B1_SEMAPHORE_D_OPERATION_RESERVED0                                  (0x00000001)
#define LW95B1_SEMAPHORE_D_OPERATION_RESERVED1                                  (0x00000002)
#define LW95B1_SEMAPHORE_D_OPERATION_TRAP                                       (0x00000003)
#define LW95B1_SEMAPHORE_D_FLUSH_DISABLE                                        21:21
#define LW95B1_SEMAPHORE_D_FLUSH_DISABLE_HIGH_FIELD                             21U
#define LW95B1_SEMAPHORE_D_FLUSH_DISABLE_LOW_FIELD                              21U
#define LW95B1_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                  (0x00000000)
#define LW95B1_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                   (0x00000001)
#define LW95B1_SET_PREDICATION_OFFSET_UPPER                                     (0x00000308)
#define LW95B1_SET_PREDICATION_OFFSET_UPPER_OFFSET                              7:0
#define LW95B1_SET_PREDICATION_OFFSET_UPPER_OFFSET_HIGH_FIELD                   7U
#define LW95B1_SET_PREDICATION_OFFSET_UPPER_OFFSET_LOW_FIELD                    0U
#define LW95B1_SET_PREDICATION_OFFSET_LOWER                                     (0x0000030C)
#define LW95B1_SET_PREDICATION_OFFSET_LOWER_OFFSET                              31:0
#define LW95B1_SET_PREDICATION_OFFSET_LOWER_OFFSET_HIGH_FIELD                   31U
#define LW95B1_SET_PREDICATION_OFFSET_LOWER_OFFSET_LOW_FIELD                    0U
#define LW95B1_H264_SET_SEQ_PIC_CTRL_IN_OFFSET                                  (0x00000400)
#define LW95B1_H264_SET_SEQ_PIC_CTRL_IN_OFFSET_OFFSET                           31:0
#define LW95B1_H264_SET_SEQ_PIC_CTRL_IN_OFFSET_OFFSET_HIGH_FIELD                31U
#define LW95B1_H264_SET_SEQ_PIC_CTRL_IN_OFFSET_OFFSET_LOW_FIELD                 0U
#define LW95B1_H264_SET_SLICE_HDR_OUT_BUF_OFFSET                                (0x00000404)
#define LW95B1_H264_SET_SLICE_HDR_OUT_BUF_OFFSET_OFFSET                         31:0
#define LW95B1_H264_SET_SLICE_HDR_OUT_BUF_OFFSET_OFFSET_HIGH_FIELD              31U
#define LW95B1_H264_SET_SLICE_HDR_OUT_BUF_OFFSET_OFFSET_LOW_FIELD               0U
#define LW95B1_H264_SET_SLICE_HDR_OUT_BUF_SIZE                                  (0x00000408)
#define LW95B1_H264_SET_SLICE_HDR_OUT_BUF_SIZE_SIZE                             31:0
#define LW95B1_H264_SET_SLICE_HDR_OUT_BUF_SIZE_SIZE_HIGH_FIELD                  31U
#define LW95B1_H264_SET_SLICE_HDR_OUT_BUF_SIZE_SIZE_LOW_FIELD                   0U
#define LW95B1_H264_SET_SLICE_DATA_OUT_BUF_OFFSET                               (0x0000040C)
#define LW95B1_H264_SET_SLICE_DATA_OUT_BUF_OFFSET_OFFSET                        31:0
#define LW95B1_H264_SET_SLICE_DATA_OUT_BUF_OFFSET_OFFSET_HIGH_FIELD             31U
#define LW95B1_H264_SET_SLICE_DATA_OUT_BUF_OFFSET_OFFSET_LOW_FIELD              0U
#define LW95B1_H264_SET_SLICE_DATA_OUT_BUF_SIZE                                 (0x00000410)
#define LW95B1_H264_SET_SLICE_DATA_OUT_BUF_SIZE_SIZE                            31:0
#define LW95B1_H264_SET_SLICE_DATA_OUT_BUF_SIZE_SIZE_HIGH_FIELD                 31U
#define LW95B1_H264_SET_SLICE_DATA_OUT_BUF_SIZE_SIZE_LOW_FIELD                  0U
#define LW95B1_H264_SET_MBHIST_BUF_OFFSET                                       (0x00000414)
#define LW95B1_H264_SET_MBHIST_BUF_OFFSET_OFFSET                                31:0
#define LW95B1_H264_SET_MBHIST_BUF_OFFSET_OFFSET_HIGH_FIELD                     31U
#define LW95B1_H264_SET_MBHIST_BUF_OFFSET_OFFSET_LOW_FIELD                      0U
#define LW95B1_H264_SET_MBHIST_BUF_SIZE                                         (0x00000418)
#define LW95B1_H264_SET_MBHIST_BUF_SIZE_SIZE                                    31:0
#define LW95B1_H264_SET_MBHIST_BUF_SIZE_SIZE_HIGH_FIELD                         31U
#define LW95B1_H264_SET_MBHIST_BUF_SIZE_SIZE_LOW_FIELD                          0U
#define LW95B1_H264_SET_BASE_SLC_HDR_OFFSET                                     (0x00000420)
#define LW95B1_H264_SET_BASE_SLC_HDR_OFFSET_OFFSET                              31:0
#define LW95B1_H264_SET_BASE_SLC_HDR_OFFSET_OFFSET_HIGH_FIELD                   31U
#define LW95B1_H264_SET_BASE_SLC_HDR_OFFSET_OFFSET_LOW_FIELD                    0U
#define LW95B1_VC1_SET_SEQ_CTRL_IN_OFFSET                                       (0x00000500)
#define LW95B1_VC1_SET_SEQ_CTRL_IN_OFFSET_OFFSET                                31:0
#define LW95B1_VC1_SET_SEQ_CTRL_IN_OFFSET_OFFSET_HIGH_FIELD                     31U
#define LW95B1_VC1_SET_SEQ_CTRL_IN_OFFSET_OFFSET_LOW_FIELD                      0U
#define LW95B1_VC1_SET_PIC_HDR_OUT_BUF_OFFSET                                   (0x00000504)
#define LW95B1_VC1_SET_PIC_HDR_OUT_BUF_OFFSET_OFFSET                            31:0
#define LW95B1_VC1_SET_PIC_HDR_OUT_BUF_OFFSET_OFFSET_HIGH_FIELD                 31U
#define LW95B1_VC1_SET_PIC_HDR_OUT_BUF_OFFSET_OFFSET_LOW_FIELD                  0U
#define LW95B1_VC1_SET_PIC_DATA_OUT_BUF_OFFSET                                  (0x00000508)
#define LW95B1_VC1_SET_PIC_DATA_OUT_BUF_OFFSET_OFFSET                           31:0
#define LW95B1_VC1_SET_PIC_DATA_OUT_BUF_OFFSET_OFFSET_HIGH_FIELD                31U
#define LW95B1_VC1_SET_PIC_DATA_OUT_BUF_OFFSET_OFFSET_LOW_FIELD                 0U
#define LW95B1_VC1_SET_PIC_DATA_OUT_BUF_SIZE                                    (0x0000050C)
#define LW95B1_VC1_SET_PIC_DATA_OUT_BUF_SIZE_SIZE                               31:0
#define LW95B1_VC1_SET_PIC_DATA_OUT_BUF_SIZE_SIZE_HIGH_FIELD                    31U
#define LW95B1_VC1_SET_PIC_DATA_OUT_BUF_SIZE_SIZE_LOW_FIELD                     0U
#define LW95B1_VC1_SET_PIC_SCRATCH_BUF_OFFSET                                   (0x00000510)
#define LW95B1_VC1_SET_PIC_SCRATCH_BUF_OFFSET_OFFSET                            31:0
#define LW95B1_VC1_SET_PIC_SCRATCH_BUF_OFFSET_OFFSET_HIGH_FIELD                 31U
#define LW95B1_VC1_SET_PIC_SCRATCH_BUF_OFFSET_OFFSET_LOW_FIELD                  0U
#define LW95B1_VC1_SET_PIC_SCRATCH_BUF_SIZE                                     (0x00000514)
#define LW95B1_VC1_SET_PIC_SCRATCH_BUF_SIZE_OFFSET                              31:0
#define LW95B1_VC1_SET_PIC_SCRATCH_BUF_SIZE_OFFSET_HIGH_FIELD                   31U
#define LW95B1_VC1_SET_PIC_SCRATCH_BUF_SIZE_OFFSET_LOW_FIELD                    0U
#define LW95B1_MPEG12_SET_SEQ_PIC_CTRL_IN_OFFSET                                (0x00000600)
#define LW95B1_MPEG12_SET_SEQ_PIC_CTRL_IN_OFFSET_OFFSET                         31:0
#define LW95B1_MPEG12_SET_SEQ_PIC_CTRL_IN_OFFSET_OFFSET_HIGH_FIELD              31U
#define LW95B1_MPEG12_SET_SEQ_PIC_CTRL_IN_OFFSET_OFFSET_LOW_FIELD               0U
#define LW95B1_MPEG12_SET_SLC_HDR_OUT_BUF_OFFSET                                (0x00000604)
#define LW95B1_MPEG12_SET_SLC_HDR_OUT_BUF_OFFSET_OFFSET                         31:0
#define LW95B1_MPEG12_SET_SLC_HDR_OUT_BUF_OFFSET_OFFSET_HIGH_FIELD              31U
#define LW95B1_MPEG12_SET_SLC_HDR_OUT_BUF_OFFSET_OFFSET_LOW_FIELD               0U
#define LW95B1_MPEG12_SET_SLC_DATA_OUT_BUF_OFFSET                               (0x00000608)
#define LW95B1_MPEG12_SET_SLC_DATA_OUT_BUF_OFFSET_OFFSET                        31:0
#define LW95B1_MPEG12_SET_SLC_DATA_OUT_BUF_OFFSET_OFFSET_HIGH_FIELD             31U
#define LW95B1_MPEG12_SET_SLC_DATA_OUT_BUF_OFFSET_OFFSET_LOW_FIELD              0U
#define LW95B1_MPEG12_SET_SLC_DATA_OUT_BUF_SIZE                                 (0x0000060C)
#define LW95B1_MPEG12_SET_SLC_DATA_OUT_BUF_SIZE_SIZE                            31:0
#define LW95B1_MPEG12_SET_SLC_DATA_OUT_BUF_SIZE_SIZE_HIGH_FIELD                 31U
#define LW95B1_MPEG12_SET_SLC_DATA_OUT_BUF_SIZE_SIZE_LOW_FIELD                  0U
#define LW95B1_SET_CONTROL_PARAMS                                               (0x00000700)
#define LW95B1_SET_CONTROL_PARAMS_CODEC_TYPE                                    3:0
#define LW95B1_SET_CONTROL_PARAMS_CODEC_TYPE_HIGH_FIELD                         3U
#define LW95B1_SET_CONTROL_PARAMS_CODEC_TYPE_LOW_FIELD                          0U
#define LW95B1_SET_CONTROL_PARAMS_CODEC_TYPE_MPEG1                              (0x00000000)
#define LW95B1_SET_CONTROL_PARAMS_CODEC_TYPE_MPEG2                              (0x00000001)
#define LW95B1_SET_CONTROL_PARAMS_CODEC_TYPE_VC1                                (0x00000002)
#define LW95B1_SET_CONTROL_PARAMS_CODEC_TYPE_H264                               (0x00000003)
#define LW95B1_SET_CONTROL_PARAMS_CODEC_TYPE_MPEG4                              (0x00000004)
#define LW95B1_SET_CONTROL_PARAMS_NUM_SLICES                                    15:4
#define LW95B1_SET_CONTROL_PARAMS_NUM_SLICES_HIGH_FIELD                         15U
#define LW95B1_SET_CONTROL_PARAMS_NUM_SLICES_LOW_FIELD                          4U
#define LW95B1_SET_CONTROL_PARAMS_RING_OUTPUT                                   16:16
#define LW95B1_SET_CONTROL_PARAMS_RING_OUTPUT_HIGH_FIELD                        16U
#define LW95B1_SET_CONTROL_PARAMS_RING_OUTPUT_LOW_FIELD                         16U
#define LW95B1_SET_CONTROL_PARAMS_RING_OUTPUT_DISABLE                           (0x00000000)
#define LW95B1_SET_CONTROL_PARAMS_RING_OUTPUT_ENABLE                            (0x00000001)
#define LW95B1_SET_CONTROL_PARAMS_GPTIMER_ON                                    17:17
#define LW95B1_SET_CONTROL_PARAMS_GPTIMER_ON_HIGH_FIELD                         17U
#define LW95B1_SET_CONTROL_PARAMS_GPTIMER_ON_LOW_FIELD                          17U
#define LW95B1_SET_CONTROL_PARAMS_RET_ERROR                                     18:18
#define LW95B1_SET_CONTROL_PARAMS_RET_ERROR_HIGH_FIELD                          18U
#define LW95B1_SET_CONTROL_PARAMS_RET_ERROR_LOW_FIELD                           18U
#define LW95B1_SET_CONTROL_PARAMS_ERR_CONCEAL_ON                                19:19
#define LW95B1_SET_CONTROL_PARAMS_ERR_CONCEAL_ON_HIGH_FIELD                     19U
#define LW95B1_SET_CONTROL_PARAMS_ERR_CONCEAL_ON_LOW_FIELD                      19U
#define LW95B1_SET_CONTROL_PARAMS_NUM_SLICES_MSB                                20:20
#define LW95B1_SET_CONTROL_PARAMS_NUM_SLICES_MSB_HIGH_FIELD                     20U
#define LW95B1_SET_CONTROL_PARAMS_NUM_SLICES_MSB_LOW_FIELD                      20U
#define LW95B1_SET_CONTROL_PARAMS_ENABLE_OUTPUT_ENCRYPT                         21:21
#define LW95B1_SET_CONTROL_PARAMS_ENABLE_OUTPUT_ENCRYPT_HIGH_FIELD              21U
#define LW95B1_SET_CONTROL_PARAMS_ENABLE_OUTPUT_ENCRYPT_LOW_FIELD               21U
#define LW95B1_SET_CONTROL_PARAMS_ENABLE_OUTPUT_ENCRYPT_DISABLE                 (0x00000000)
#define LW95B1_SET_CONTROL_PARAMS_ENABLE_OUTPUT_ENCRYPT_ENABLE                  (0x00000001)
#define LW95B1_SET_CONTROL_PARAMS_ALL_INTRA_FRAME                               22:22
#define LW95B1_SET_CONTROL_PARAMS_ALL_INTRA_FRAME_HIGH_FIELD                    22U
#define LW95B1_SET_CONTROL_PARAMS_ALL_INTRA_FRAME_LOW_FIELD                     22U
#define LW95B1_SET_CONTROL_PARAMS_RESERVED                                      31:23
#define LW95B1_SET_CONTROL_PARAMS_RESERVED_HIGH_FIELD                           31U
#define LW95B1_SET_CONTROL_PARAMS_RESERVED_LOW_FIELD                            23U
#define LW95B1_SET_DATA_INFO_BUFFER_IN_OFFSET                                   (0x00000704)
#define LW95B1_SET_DATA_INFO_BUFFER_IN_OFFSET_OFFSET                            31:0
#define LW95B1_SET_DATA_INFO_BUFFER_IN_OFFSET_OFFSET_HIGH_FIELD                 31U
#define LW95B1_SET_DATA_INFO_BUFFER_IN_OFFSET_OFFSET_LOW_FIELD                  0U
#define LW95B1_SET_IN_BUF_BASE_OFFSET                                           (0x00000708)
#define LW95B1_SET_IN_BUF_BASE_OFFSET_OFFSET                                    31:0
#define LW95B1_SET_IN_BUF_BASE_OFFSET_OFFSET_HIGH_FIELD                         31U
#define LW95B1_SET_IN_BUF_BASE_OFFSET_OFFSET_LOW_FIELD                          0U
#define LW95B1_SET_FLOW_CTRL_OUT_OFFSET                                         (0x0000070C)
#define LW95B1_SET_FLOW_CTRL_OUT_OFFSET_OFFSET                                  31:0
#define LW95B1_SET_FLOW_CTRL_OUT_OFFSET_OFFSET_HIGH_FIELD                       31U
#define LW95B1_SET_FLOW_CTRL_OUT_OFFSET_OFFSET_LOW_FIELD                        0U
#define LW95B1_SET_PICTURE_INDEX                                                (0x00000710)
#define LW95B1_SET_PICTURE_INDEX_INDEX                                          31:0
#define LW95B1_SET_PICTURE_INDEX_INDEX_HIGH_FIELD                               31U
#define LW95B1_SET_PICTURE_INDEX_INDEX_LOW_FIELD                                0U
#define LW95B1_SET_CONTENT_INITIAL_VECTOR(b)                                    (0x00000C00 + (b)*0x00000004)
#define LW95B1_SET_CONTENT_INITIAL_VECTOR_VALUE                                 31:0
#define LW95B1_SET_CONTENT_INITIAL_VECTOR_VALUE_HIGH_FIELD                      31U
#define LW95B1_SET_CONTENT_INITIAL_VECTOR_VALUE_LOW_FIELD                       0U
#define LW95B1_SET_CTL_COUNT                                                    (0x00000C10)
#define LW95B1_SET_CTL_COUNT_VALUE                                              31:0
#define LW95B1_SET_CTL_COUNT_VALUE_HIGH_FIELD                                   31U
#define LW95B1_SET_CTL_COUNT_VALUE_LOW_FIELD                                    0U
#define LW95B1_SET_MDEC_H2_MKEY                                                 (0x00000C14)
#define LW95B1_SET_MDEC_H2_MKEY_HOST_SKEY                                       15:0
#define LW95B1_SET_MDEC_H2_MKEY_HOST_SKEY_HIGH_FIELD                            15U
#define LW95B1_SET_MDEC_H2_MKEY_HOST_SKEY_LOW_FIELD                             0U
#define LW95B1_SET_MDEC_H2_MKEY_HOST_KEY_HASH                                   23:16
#define LW95B1_SET_MDEC_H2_MKEY_HOST_KEY_HASH_HIGH_FIELD                        23U
#define LW95B1_SET_MDEC_H2_MKEY_HOST_KEY_HASH_LOW_FIELD                         16U
#define LW95B1_SET_MDEC_H2_MKEY_DEC_ID                                          31:24
#define LW95B1_SET_MDEC_H2_MKEY_DEC_ID_HIGH_FIELD                               31U
#define LW95B1_SET_MDEC_H2_MKEY_DEC_ID_LOW_FIELD                                24U
#define LW95B1_SET_MDEC_M2_HKEY                                                 (0x00000C18)
#define LW95B1_SET_MDEC_M2_HKEY_MPEG_SKEY                                       15:0
#define LW95B1_SET_MDEC_M2_HKEY_MPEG_SKEY_HIGH_FIELD                            15U
#define LW95B1_SET_MDEC_M2_HKEY_MPEG_SKEY_LOW_FIELD                             0U
#define LW95B1_SET_MDEC_M2_HKEY_SELECTOR                                        23:16
#define LW95B1_SET_MDEC_M2_HKEY_SELECTOR_HIGH_FIELD                             23U
#define LW95B1_SET_MDEC_M2_HKEY_SELECTOR_LOW_FIELD                              16U
#define LW95B1_SET_MDEC_M2_HKEY_MPEG_KEY_HASH                                   31:24
#define LW95B1_SET_MDEC_M2_HKEY_MPEG_KEY_HASH_HIGH_FIELD                        31U
#define LW95B1_SET_MDEC_M2_HKEY_MPEG_KEY_HASH_LOW_FIELD                         24U
#define LW95B1_SET_MDEC_FRAME_KEY                                               (0x00000C1C)
#define LW95B1_SET_MDEC_FRAME_KEY_VALUE                                         15:0
#define LW95B1_SET_MDEC_FRAME_KEY_VALUE_HIGH_FIELD                              15U
#define LW95B1_SET_MDEC_FRAME_KEY_VALUE_LOW_FIELD                               0U
#define LW95B1_SET_UPPER_SRC                                                    (0x00000C20)
#define LW95B1_SET_UPPER_SRC_OFFSET                                             7:0
#define LW95B1_SET_UPPER_SRC_OFFSET_HIGH_FIELD                                  7U
#define LW95B1_SET_UPPER_SRC_OFFSET_LOW_FIELD                                   0U
#define LW95B1_SET_LOWER_SRC                                                    (0x00000C24)
#define LW95B1_SET_LOWER_SRC_OFFSET                                             31:0
#define LW95B1_SET_LOWER_SRC_OFFSET_HIGH_FIELD                                  31U
#define LW95B1_SET_LOWER_SRC_OFFSET_LOW_FIELD                                   0U
#define LW95B1_SET_UPPER_DST                                                    (0x00000C28)
#define LW95B1_SET_UPPER_DST_OFFSET                                             7:0
#define LW95B1_SET_UPPER_DST_OFFSET_HIGH_FIELD                                  7U
#define LW95B1_SET_UPPER_DST_OFFSET_LOW_FIELD                                   0U
#define LW95B1_SET_LOWER_DST                                                    (0x00000C2C)
#define LW95B1_SET_LOWER_DST_OFFSET                                             31:0
#define LW95B1_SET_LOWER_DST_OFFSET_HIGH_FIELD                                  31U
#define LW95B1_SET_LOWER_DST_OFFSET_LOW_FIELD                                   0U
#define LW95B1_SET_UPPER_CTL                                                    (0x00000C30)
#define LW95B1_SET_UPPER_CTL_OFFSET                                             7:0
#define LW95B1_SET_UPPER_CTL_OFFSET_HIGH_FIELD                                  7U
#define LW95B1_SET_UPPER_CTL_OFFSET_LOW_FIELD                                   0U
#define LW95B1_SET_LOWER_CTL                                                    (0x00000C34)
#define LW95B1_SET_LOWER_CTL_OFFSET                                             31:0
#define LW95B1_SET_LOWER_CTL_OFFSET_HIGH_FIELD                                  31U
#define LW95B1_SET_LOWER_CTL_OFFSET_LOW_FIELD                                   0U
#define LW95B1_SET_BLOCK_COUNT                                                  (0x00000C38)
#define LW95B1_SET_BLOCK_COUNT_VALUE                                            31:0
#define LW95B1_SET_BLOCK_COUNT_VALUE_HIGH_FIELD                                 31U
#define LW95B1_SET_BLOCK_COUNT_VALUE_LOW_FIELD                                  0U
#define LW95B1_SET_STRETCH_MASK                                                 (0x00000C3C)
#define LW95B1_SET_STRETCH_MASK_VALUE                                           31:0
#define LW95B1_SET_STRETCH_MASK_VALUE_HIGH_FIELD                                31U
#define LW95B1_SET_STRETCH_MASK_VALUE_LOW_FIELD                                 0U
#define LW95B1_SET_UPPER_FLOW_CTRL_INSELWRE                                     (0x00000D00)
#define LW95B1_SET_UPPER_FLOW_CTRL_INSELWRE_OFFSET                              7:0
#define LW95B1_SET_UPPER_FLOW_CTRL_INSELWRE_OFFSET_HIGH_FIELD                   7U
#define LW95B1_SET_UPPER_FLOW_CTRL_INSELWRE_OFFSET_LOW_FIELD                    0U
#define LW95B1_SET_LOWER_FLOW_CTRL_INSELWRE                                     (0x00000D04)
#define LW95B1_SET_LOWER_FLOW_CTRL_INSELWRE_OFFSET                              31:0
#define LW95B1_SET_LOWER_FLOW_CTRL_INSELWRE_OFFSET_HIGH_FIELD                   31U
#define LW95B1_SET_LOWER_FLOW_CTRL_INSELWRE_OFFSET_LOW_FIELD                    0U
#define LW95B1_SET_UCODE_LOADER_PARAMS                                          (0x00000D10)
#define LW95B1_SET_UCODE_LOADER_PARAMS_BLOCK_COUNT                              7:0
#define LW95B1_SET_UCODE_LOADER_PARAMS_BLOCK_COUNT_HIGH_FIELD                   7U
#define LW95B1_SET_UCODE_LOADER_PARAMS_BLOCK_COUNT_LOW_FIELD                    0U
#define LW95B1_SET_UCODE_LOADER_PARAMS_SELWRITY_PARAM                           15:8
#define LW95B1_SET_UCODE_LOADER_PARAMS_SELWRITY_PARAM_HIGH_FIELD                15U
#define LW95B1_SET_UCODE_LOADER_PARAMS_SELWRITY_PARAM_LOW_FIELD                 8U
#define LW95B1_SET_UPPER_FLOW_CTRL_SELWRE                                       (0x00000D18)
#define LW95B1_SET_UPPER_FLOW_CTRL_SELWRE_OFFSET                                7:0
#define LW95B1_SET_UPPER_FLOW_CTRL_SELWRE_OFFSET_HIGH_FIELD                     7U
#define LW95B1_SET_UPPER_FLOW_CTRL_SELWRE_OFFSET_LOW_FIELD                      0U
#define LW95B1_SET_LOWER_FLOW_CTRL_SELWRE                                       (0x00000D1C)
#define LW95B1_SET_LOWER_FLOW_CTRL_SELWRE_OFFSET                                31:0
#define LW95B1_SET_LOWER_FLOW_CTRL_SELWRE_OFFSET_HIGH_FIELD                     31U
#define LW95B1_SET_LOWER_FLOW_CTRL_SELWRE_OFFSET_LOW_FIELD                      0U
#define LW95B1_SET_UCODE_LOADER_OFFSET                                          (0x00000D34)
#define LW95B1_SET_UCODE_LOADER_OFFSET_OFFSET                                   31:0
#define LW95B1_SET_UCODE_LOADER_OFFSET_OFFSET_HIGH_FIELD                        31U
#define LW95B1_SET_UCODE_LOADER_OFFSET_OFFSET_LOW_FIELD                         0U
#define LW95B1_MP4_SET_SEQ_CTRL_IN_OFFSET                                       (0x00000E00)
#define LW95B1_MP4_SET_SEQ_CTRL_IN_OFFSET_OFFSET                                31:0
#define LW95B1_MP4_SET_SEQ_CTRL_IN_OFFSET_OFFSET_HIGH_FIELD                     31U
#define LW95B1_MP4_SET_SEQ_CTRL_IN_OFFSET_OFFSET_LOW_FIELD                      0U
#define LW95B1_MP4_SET_PIC_HDR_OUT_BUF_OFFSET                                   (0x00000E04)
#define LW95B1_MP4_SET_PIC_HDR_OUT_BUF_OFFSET_OFFSET                            31:0
#define LW95B1_MP4_SET_PIC_HDR_OUT_BUF_OFFSET_OFFSET_HIGH_FIELD                 31U
#define LW95B1_MP4_SET_PIC_HDR_OUT_BUF_OFFSET_OFFSET_LOW_FIELD                  0U
#define LW95B1_MP4_SET_PIC_DATA_OUT_BUF_OFFSET                                  (0x00000E08)
#define LW95B1_MP4_SET_PIC_DATA_OUT_BUF_OFFSET_OFFSET                           31:0
#define LW95B1_MP4_SET_PIC_DATA_OUT_BUF_OFFSET_OFFSET_HIGH_FIELD                31U
#define LW95B1_MP4_SET_PIC_DATA_OUT_BUF_OFFSET_OFFSET_LOW_FIELD                 0U
#define LW95B1_MP4_SET_PIC_DATA_OUT_BUF_SIZE                                    (0x00000E0C)
#define LW95B1_MP4_SET_PIC_DATA_OUT_BUF_SIZE_SIZE                               31:0
#define LW95B1_MP4_SET_PIC_DATA_OUT_BUF_SIZE_SIZE_HIGH_FIELD                    31U
#define LW95B1_MP4_SET_PIC_DATA_OUT_BUF_SIZE_SIZE_LOW_FIELD                     0U
#define LW95B1_MP4_SET_PIC_SCRATCH_BUF_OFFSET                                   (0x00000E10)
#define LW95B1_MP4_SET_PIC_SCRATCH_BUF_OFFSET_OFFSET                            31:0
#define LW95B1_MP4_SET_PIC_SCRATCH_BUF_OFFSET_OFFSET_HIGH_FIELD                 31U
#define LW95B1_MP4_SET_PIC_SCRATCH_BUF_OFFSET_OFFSET_LOW_FIELD                  0U
#define LW95B1_MP4_SET_PIC_SCRATCH_BUF_SIZE                                     (0x00000E14)
#define LW95B1_MP4_SET_PIC_SCRATCH_BUF_SIZE_SIZE                                31:0
#define LW95B1_MP4_SET_PIC_SCRATCH_BUF_SIZE_SIZE_HIGH_FIELD                     31U
#define LW95B1_MP4_SET_PIC_SCRATCH_BUF_SIZE_SIZE_LOW_FIELD                      0U
#define LW95B1_SET_SESSION_KEY(b)                                               (0x00000F00 + (b)*0x00000004)
#define LW95B1_SET_SESSION_KEY_VALUE                                            31:0
#define LW95B1_SET_SESSION_KEY_VALUE_HIGH_FIELD                                 31U
#define LW95B1_SET_SESSION_KEY_VALUE_LOW_FIELD                                  0U
#define LW95B1_SET_CONTENT_KEY(b)                                               (0x00000F10 + (b)*0x00000004)
#define LW95B1_SET_CONTENT_KEY_VALUE                                            31:0
#define LW95B1_SET_CONTENT_KEY_VALUE_HIGH_FIELD                                 31U
#define LW95B1_SET_CONTENT_KEY_VALUE_LOW_FIELD                                  0U
#define LW95B1_PM_TRIGGER_END                                                   (0x00001114)
#define LW95B1_PM_TRIGGER_END_V                                                 31:0
#define LW95B1_PM_TRIGGER_END_V_HIGH_FIELD                                      31U
#define LW95B1_PM_TRIGGER_END_V_LOW_FIELD                                       0U

#define LW95B1_ERROR_NONE                                                       (0x00000000)
#define LW95B1_OS_ERROR_EXELWTE_INSUFFICIENT_DATA                               (0x00000001)
#define LW95B1_OS_ERROR_SEMAPHORE_INSUFFICIENT_DATA                             (0x00000002)
#define LW95B1_OS_ERROR_ILWALID_METHOD                                          (0x00000003)
#define LW95B1_OS_ERROR_ILWALID_DMA_PAGE                                        (0x00000004)
#define LW95B1_OS_ERROR_UNHANDLED_INTERRUPT                                     (0x00000005)
#define LW95B1_OS_ERROR_EXCEPTION                                               (0x00000006)
#define LW95B1_OS_ERROR_ILWALID_CTXSW_REQUEST                                   (0x00000007)
#define LW95B1_OS_ERROR_APPLICATION                                             (0x00000008)
#define LW95B1_OS_ERROR_SW_BREAKPT                                              (0x00000009)
#define LW95B1_OS_INTERRUPT_EXELWTE_AWAKEN                                      (0x00000100)
#define LW95B1_OS_INTERRUPT_BACKEND_SEMAPHORE_AWAKEN                            (0x00000200)
#define LW95B1_OS_INTERRUPT_CTX_ERROR_FBIF                                      (0x00000300)
#define LW95B1_OS_INTERRUPT_LIMIT_VIOLATION                                     (0x00000400)
#define LW95B1_OS_INTERRUPT_LIMIT_AND_FBIF_CTX_ERROR                            (0x00000500)
#define LW95B1_OS_INTERRUPT_HALT_ENGINE                                         (0x00000600)
#define LW95B1_OS_INTERRUPT_TRAP_NONSTALL                                       (0x00000700)
#define LW95B1_H264_VLD_ERR_SEQ_DATA_INCONSISTENT                               (0x00004001)
#define LW95B1_H264_VLD_ERR_PIC_DATA_INCONSISTENT                               (0x00004002)
#define LW95B1_H264_VLD_ERR_SLC_DATA_BUF_ADDR_OUT_OF_BOUNDS                     (0x00004100)
#define LW95B1_H264_VLD_ERR_BITSTREAM_ERROR                                     (0x00004101)
#define LW95B1_H264_VLD_ERR_CTX_DMA_ID_CTRL_IN_ILWALID                          (0x000041F8)
#define LW95B1_H264_VLD_ERR_SLC_HDR_OUT_SIZE_NOT_MULT256                        (0x00004200)
#define LW95B1_H264_VLD_ERR_SLC_DATA_OUT_SIZE_NOT_MULT256                       (0x00004201)
#define LW95B1_H264_VLD_ERR_CTX_DMA_ID_FLOW_CTRL_ILWALID                        (0x00004203)
#define LW95B1_H264_VLD_ERR_CTX_DMA_ID_SLC_HDR_OUT_ILWALID                      (0x00004204)
#define LW95B1_H264_VLD_ERR_SLC_HDR_OUT_BUF_TOO_SMALL                           (0x00004205)
#define LW95B1_H264_VLD_ERR_SLC_HDR_OUT_BUF_ALREADY_VALID                       (0x00004206)
#define LW95B1_H264_VLD_ERR_SLC_DATA_OUT_BUF_TOO_SMALL                          (0x00004207)
#define LW95B1_H264_VLD_ERR_DATA_BUF_CNT_TOO_SMALL                              (0x00004208)
#define LW95B1_H264_VLD_ERR_BITSTREAM_EMPTY                                     (0x00004209)
#define LW95B1_H264_VLD_ERR_FRAME_WIDTH_TOO_LARGE                               (0x0000420A)
#define LW95B1_H264_VLD_ERR_FRAME_HEIGHT_TOO_LARGE                              (0x0000420B)
#define LW95B1_H264_VLD_ERR_HIST_BUF_TOO_SMALL                                  (0x00004300)
#define LW95B1_VC1_VLD_ERR_PIC_DATA_BUF_ADDR_OUT_OF_BOUND                       (0x00005100)
#define LW95B1_VC1_VLD_ERR_BITSTREAM_ERROR                                      (0x00005101)
#define LW95B1_VC1_VLD_ERR_PIC_HDR_OUT_SIZE_NOT_MULT256                         (0x00005200)
#define LW95B1_VC1_VLD_ERR_PIC_DATA_OUT_SIZE_NOT_MULT256                        (0x00005201)
#define LW95B1_VC1_VLD_ERR_CTX_DMA_ID_CTRL_IN_ILWALID                           (0x00005202)
#define LW95B1_VC1_VLD_ERR_CTX_DMA_ID_FLOW_CTRL_ILWALID                         (0x00005203)
#define LW95B1_VC1_VLD_ERR_CTX_DMA_ID_PIC_HDR_OUT_ILWALID                       (0x00005204)
#define LW95B1_VC1_VLD_ERR_SLC_HDR_OUT_BUF_TOO_SMALL                            (0x00005205)
#define LW95B1_VC1_VLD_ERR_PIC_HDR_OUT_BUF_ALREADY_VALID                        (0x00005206)
#define LW95B1_VC1_VLD_ERR_PIC_DATA_OUT_BUF_TOO_SMALL                           (0x00005207)
#define LW95B1_VC1_VLD_ERR_DATA_INFO_IN_BUF_TOO_SMALL                           (0x00005208)
#define LW95B1_VC1_VLD_ERR_BITSTREAM_EMPTY                                      (0x00005209)
#define LW95B1_VC1_VLD_ERR_FRAME_WIDTH_TOO_LARGE                                (0x0000520A)
#define LW95B1_VC1_VLD_ERR_FRAME_HEIGHT_TOO_LARGE                               (0x0000520B)
#define LW95B1_VC1_VLD_ERR_PIC_DATA_OUT_BUF_FULL_TIME_OUT                       (0x00005300)
#define LW95B1_MPEG12_VLD_ERR_SLC_DATA_BUF_ADDR_OUT_OF_BOUNDS                   (0x00006100)
#define LW95B1_MPEG12_VLD_ERR_BITSTREAM_ERROR                                   (0x00006101)
#define LW95B1_MPEG12_VLD_ERR_SLC_DATA_OUT_SIZE_NOT_MULT256                     (0x00006200)
#define LW95B1_MPEG12_VLD_ERR_CTX_DMA_ID_CTRL_IN_ILWALID                        (0x00006201)
#define LW95B1_MPEG12_VLD_ERR_CTX_DMA_ID_FLOW_CTRL_ILWALID                      (0x00006202)
#define LW95B1_MPEG12_VLD_ERR_SLC_DATA_OUT_BUF_TOO_SMALL                        (0x00006203)
#define LW95B1_MPEG12_VLD_ERR_DATA_INFO_IN_BUF_TOO_SMALL                        (0x00006204)
#define LW95B1_MPEG12_VLD_ERR_BITSTREAM_EMPTY                                   (0x00006205)
#define LW95B1_MPEG12_VLD_ERR_ILWALID_PIC_STRUCTURE                             (0x00006206)
#define LW95B1_MPEG12_VLD_ERR_ILWALID_PIC_CODING_TYPE                           (0x00006207)
#define LW95B1_MPEG12_VLD_ERR_FRAME_WIDTH_TOO_LARGE                             (0x00006208)
#define LW95B1_MPEG12_VLD_ERR_FRAME_HEIGHT_TOO_LARGE                            (0x00006209)
#define LW95B1_MPEG12_VLD_ERR_SLC_DATA_OUT_BUF_FULL_TIME_OUT                    (0x00006300)
#define LW95B1_CMN_VLD_ERR_PDEC_RETURNED_ERROR                                  (0x00007101)
#define LW95B1_CMN_VLD_ERR_EDOB_FLUSH_TIME_OUT                                  (0x00007102)
#define LW95B1_CMN_VLD_ERR_EDOB_REWIND_TIME_OUT                                 (0x00007103)
#define LW95B1_CMN_VLD_ERR_VLD_WD_TIME_OUT                                      (0x00007104)
#define LW95B1_CMN_VLD_ERR_NUM_SLICES_ZERO                                      (0x00007105)
#define LW95B1_MPEG4_VLD_ERR_PIC_DATA_BUF_ADDR_OUT_OF_BOUND                     (0x00008100)
#define LW95B1_MPEG4_VLD_ERR_BITSTREAM_ERROR                                    (0x00008101)
#define LW95B1_MPEG4_VLD_ERR_PIC_HDR_OUT_SIZE_NOT_MULT256                       (0x00008200)
#define LW95B1_MPEG4_VLD_ERR_PIC_DATA_OUT_SIZE_NOT_MULT256                      (0x00008201)
#define LW95B1_MPEG4_VLD_ERR_CTX_DMA_ID_CTRL_IN_ILWALID                         (0x00008202)
#define LW95B1_MPEG4_VLD_ERR_CTX_DMA_ID_FLOW_CTRL_ILWALID                       (0x00008203)
#define LW95B1_MPEG4_VLD_ERR_CTX_DMA_ID_PIC_HDR_OUT_ILWALID                     (0x00008204)
#define LW95B1_MPEG4_VLD_ERR_SLC_HDR_OUT_BUF_TOO_SMALL                          (0x00008205)
#define LW95B1_MPEG4_VLD_ERR_PIC_HDR_OUT_BUF_ALREADY_VALID                      (0x00008206)
#define LW95B1_MPEG4_VLD_ERR_PIC_DATA_OUT_BUF_TOO_SMALL                         (0x00008207)
#define LW95B1_MPEG4_VLD_ERR_DATA_INFO_IN_BUF_TOO_SMALL                         (0x00008208)
#define LW95B1_MPEG4_VLD_ERR_BITSTREAM_EMPTY                                    (0x00008209)
#define LW95B1_MPEG4_VLD_ERR_FRAME_WIDTH_TOO_LARGE                              (0x0000820A)
#define LW95B1_MPEG4_VLD_ERR_FRAME_HEIGHT_TOO_LARGE                             (0x0000820B)
#define LW95B1_MPEG4_VLD_ERR_PIC_DATA_OUT_BUF_FULL_TIME_OUT                     (0x00051E01)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // CL95B1_H

