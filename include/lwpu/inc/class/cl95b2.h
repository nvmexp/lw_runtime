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


#include "lwtypes.h"

#ifndef _cl95b2_h_
#define _cl95b2_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LW95B2_VIDEO_MSPDEC                                                              (0x000095B2)

typedef volatile struct _cl95b2_tag0 {
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
    LwV32 H264SetPictureOffset3;                                                // 0x00000400 - 0x00000403
    LwV32 H264SetPictureOffset4;                                                // 0x00000404 - 0x00000407
    LwV32 H264SetPictureOffset5;                                                // 0x00000408 - 0x0000040B
    LwV32 H264SetPictureOffset6;                                                // 0x0000040C - 0x0000040F
    LwV32 H264SetPictureOffset7;                                                // 0x00000410 - 0x00000413
    LwV32 H264SetPictureOffset8;                                                // 0x00000414 - 0x00000417
    LwV32 H264SetPictureOffset9;                                                // 0x00000418 - 0x0000041B
    LwV32 H264SetPictureOffset10;                                               // 0x0000041C - 0x0000041F
    LwV32 H264SetPictureOffset11;                                               // 0x00000420 - 0x00000423
    LwV32 H264SetPictureOffset12;                                               // 0x00000424 - 0x00000427
    LwV32 H264SetPictureOffset13;                                               // 0x00000428 - 0x0000042B
    LwV32 H264SetPictureOffset14;                                               // 0x0000042C - 0x0000042F
    LwV32 H264SetPictureOffset15;                                               // 0x00000430 - 0x00000433
    LwV32 H264SetPictureOffset16;                                               // 0x00000434 - 0x00000437
    LwV32 H264SetSliceCount;                                                    // 0x00000438 - 0x0000043B
    LwV32 H264SetAppleOOLDPictureOffset;                                        // 0x0000043C - 0x0000043F
    LwV32 Reserved06[0xB0];
    LwV32 SetControlParams;                                                     // 0x00000700 - 0x00000703
    LwV32 SetPictureIndex;                                                      // 0x00000704 - 0x00000707
    LwV32 Reserved07[0x2];
    LwV32 SetDrvPicSetupOffset;                                                 // 0x00000710 - 0x00000713
    LwV32 SetVldDecControlOffset;                                               // 0x00000714 - 0x00000717
    LwV32 SetVldSliceDataOffset;                                                // 0x00000718 - 0x0000071B
    LwV32 SetColocDataOffset;                                                   // 0x0000071C - 0x0000071F
    LwV32 SetHistoryOffset;                                                     // 0x00000720 - 0x00000723
    LwV32 SetFlowCtrlOffset;                                                    // 0x00000724 - 0x00000727
    LwV32 Reserved08[0x1];
    LwV32 SetPictureOffset0;                                                    // 0x0000072C - 0x0000072F
    LwV32 SetPictureOffset1;                                                    // 0x00000730 - 0x00000733
    LwV32 SetPictureOffset2;                                                    // 0x00000734 - 0x00000737
    LwV32 Reserved09[0x1];
    LwV32 SetMvmbStatOffset;                                                    // 0x0000073C - 0x0000073F
    LwV32 Reserved10[0x1F2];
    LwV32 H264SetPictureOffset17;                                               // 0x00000F08 - 0x00000F0B
    LwV32 Reserved11[0x7];
    LwV32 SetIlpControlOffset;                                                  // 0x00000F28 - 0x00000F2B
    LwV32 Reserved12[0x3];
    LwV32 SetIlpDataOffset;                                                     // 0x00000F38 - 0x00000F3B
    LwV32 Reserved13[0x76];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved14[0x3BA];
} LW95B2_VIDEO_MSPDECControlPio;

#define LW95B2_NOP                                                              (0x00000100)
#define LW95B2_NOP_PARAMETER                                                    31:0
#define LW95B2_PM_TRIGGER                                                       (0x00000140)
#define LW95B2_PM_TRIGGER_V                                                     31:0
#define LW95B2_SET_APPLICATION_ID                                               (0x00000200)
#define LW95B2_SET_APPLICATION_ID_ID                                            31:0
#define LW95B2_SET_APPLICATION_ID_ID_MPEG12                                     (0x00000001)
#define LW95B2_SET_APPLICATION_ID_ID_VC1                                        (0x00000002)
#define LW95B2_SET_APPLICATION_ID_ID_H264                                       (0x00000003)
#define LW95B2_SET_APPLICATION_ID_ID_MPEG4                                      (0x00000004)
#define LW95B2_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LW95B2_SET_WATCHDOG_TIMER_TIMER                                         31:0
#define LW95B2_SEMAPHORE_A                                                      (0x00000240)
#define LW95B2_SEMAPHORE_A_UPPER                                                7:0
#define LW95B2_SEMAPHORE_B                                                      (0x00000244)
#define LW95B2_SEMAPHORE_B_LOWER                                                31:0
#define LW95B2_SEMAPHORE_C                                                      (0x00000248)
#define LW95B2_SEMAPHORE_C_PAYLOAD                                              31:0
#define LW95B2_EXELWTE                                                          (0x00000300)
#define LW95B2_EXELWTE_NOTIFY                                                   0:0
#define LW95B2_EXELWTE_NOTIFY_DISABLE                                           (0x00000000)
#define LW95B2_EXELWTE_NOTIFY_ENABLE                                            (0x00000001)
#define LW95B2_EXELWTE_NOTIFY_ON                                                1:1
#define LW95B2_EXELWTE_NOTIFY_ON_END                                            (0x00000000)
#define LW95B2_EXELWTE_NOTIFY_ON_BEGIN                                          (0x00000001)
#define LW95B2_EXELWTE_PREDICATION                                              2:2
#define LW95B2_EXELWTE_PREDICATION_DISABLE                                      (0x00000000)
#define LW95B2_EXELWTE_PREDICATION_ENABLE                                       (0x00000001)
#define LW95B2_EXELWTE_PREDICATION_OP                                           3:3
#define LW95B2_EXELWTE_PREDICATION_OP_EQUAL_ZERO                                (0x00000000)
#define LW95B2_EXELWTE_PREDICATION_OP_NOT_EQUAL_ZERO                            (0x00000001)
#define LW95B2_EXELWTE_AWAKEN                                                   8:8
#define LW95B2_EXELWTE_AWAKEN_DISABLE                                           (0x00000000)
#define LW95B2_EXELWTE_AWAKEN_ENABLE                                            (0x00000001)
#define LW95B2_SEMAPHORE_D                                                      (0x00000304)
#define LW95B2_SEMAPHORE_D_STRUCTURE_SIZE                                       0:0
#define LW95B2_SEMAPHORE_D_STRUCTURE_SIZE_ONE                                   (0x00000000)
#define LW95B2_SEMAPHORE_D_STRUCTURE_SIZE_FOUR                                  (0x00000001)
#define LW95B2_SEMAPHORE_D_AWAKEN_ENABLE                                        8:8
#define LW95B2_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                  (0x00000000)
#define LW95B2_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                   (0x00000001)
#define LW95B2_SEMAPHORE_D_OPERATION                                            17:16
#define LW95B2_SEMAPHORE_D_OPERATION_RELEASE                                    (0x00000000)
#define LW95B2_SEMAPHORE_D_OPERATION_RESERVED0                                  (0x00000001)
#define LW95B2_SEMAPHORE_D_OPERATION_RESERVED1                                  (0x00000002)
#define LW95B2_SEMAPHORE_D_OPERATION_TRAP                                       (0x00000003)
#define LW95B2_SEMAPHORE_D_FLUSH_DISABLE                                        21:21
#define LW95B2_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                  (0x00000000)
#define LW95B2_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                   (0x00000001)
#define LW95B2_SET_PREDICATION_OFFSET_UPPER                                     (0x00000308)
#define LW95B2_SET_PREDICATION_OFFSET_UPPER_OFFSET                              7:0
#define LW95B2_SET_PREDICATION_OFFSET_LOWER                                     (0x0000030C)
#define LW95B2_SET_PREDICATION_OFFSET_LOWER_OFFSET                              31:0
#define LW95B2_H264_SET_PICTURE_OFFSET3                                         (0x00000400)
#define LW95B2_H264_SET_PICTURE_OFFSET3_OFFSET                                  31:0
#define LW95B2_H264_SET_PICTURE_OFFSET4                                         (0x00000404)
#define LW95B2_H264_SET_PICTURE_OFFSET4_OFFSET                                  31:0
#define LW95B2_H264_SET_PICTURE_OFFSET5                                         (0x00000408)
#define LW95B2_H264_SET_PICTURE_OFFSET5_OFFSET                                  31:0
#define LW95B2_H264_SET_PICTURE_OFFSET6                                         (0x0000040C)
#define LW95B2_H264_SET_PICTURE_OFFSET6_OFFSET                                  31:0
#define LW95B2_H264_SET_PICTURE_OFFSET7                                         (0x00000410)
#define LW95B2_H264_SET_PICTURE_OFFSET7_OFFSET                                  31:0
#define LW95B2_H264_SET_PICTURE_OFFSET8                                         (0x00000414)
#define LW95B2_H264_SET_PICTURE_OFFSET8_OFFSET                                  31:0
#define LW95B2_H264_SET_PICTURE_OFFSET9                                         (0x00000418)
#define LW95B2_H264_SET_PICTURE_OFFSET9_OFFSET                                  31:0
#define LW95B2_H264_SET_PICTURE_OFFSET10                                        (0x0000041C)
#define LW95B2_H264_SET_PICTURE_OFFSET10_OFFSET                                 31:0
#define LW95B2_H264_SET_PICTURE_OFFSET11                                        (0x00000420)
#define LW95B2_H264_SET_PICTURE_OFFSET11_OFFSET                                 31:0
#define LW95B2_H264_SET_PICTURE_OFFSET12                                        (0x00000424)
#define LW95B2_H264_SET_PICTURE_OFFSET12_OFFSET                                 31:0
#define LW95B2_H264_SET_PICTURE_OFFSET13                                        (0x00000428)
#define LW95B2_H264_SET_PICTURE_OFFSET13_OFFSET                                 31:0
#define LW95B2_H264_SET_PICTURE_OFFSET14                                        (0x0000042C)
#define LW95B2_H264_SET_PICTURE_OFFSET14_OFFSET                                 31:0
#define LW95B2_H264_SET_PICTURE_OFFSET15                                        (0x00000430)
#define LW95B2_H264_SET_PICTURE_OFFSET15_OFFSET                                 31:0
#define LW95B2_H264_SET_PICTURE_OFFSET16                                        (0x00000434)
#define LW95B2_H264_SET_PICTURE_OFFSET16_OFFSET                                 31:0
#define LW95B2_H264_SET_SLICE_COUNT                                             (0x00000438)
#define LW95B2_H264_SET_SLICE_COUNT_COUNT                                       31:0
#define LW95B2_H264_SET_APPLE_OOLDPICTURE_OFFSET                                (0x0000043C)
#define LW95B2_H264_SET_APPLE_OOLDPICTURE_OFFSET_OFFSET                         31:0
#define LW95B2_SET_CONTROL_PARAMS                                               (0x00000700)
#define LW95B2_SET_CONTROL_PARAMS_CODEC_TYPE                                    3:0
#define LW95B2_SET_CONTROL_PARAMS_CODEC_TYPE_MPEG1                              (0x00000000)
#define LW95B2_SET_CONTROL_PARAMS_CODEC_TYPE_MPEG2                              (0x00000001)
#define LW95B2_SET_CONTROL_PARAMS_CODEC_TYPE_VC1                                (0x00000002)
#define LW95B2_SET_CONTROL_PARAMS_CODEC_TYPE_H264                               (0x00000003)
#define LW95B2_SET_CONTROL_PARAMS_CODEC_TYPE_MPEG4                              (0x00000004)
#define LW95B2_SET_CONTROL_PARAMS_CODEC_TYPE_DIVX311                            (0x00000005)
#define LW95B2_SET_CONTROL_PARAMS_RING_OUTPUT                                   4:4
#define LW95B2_SET_CONTROL_PARAMS_RING_OUTPUT_DISABLE                           (0x00000000)
#define LW95B2_SET_CONTROL_PARAMS_RING_OUTPUT_ENABLE                            (0x00000001)
#define LW95B2_SET_CONTROL_PARAMS_DXVA_MODE                                     9:8
#define LW95B2_SET_CONTROL_PARAMS_GPTIMER_ON                                    12:12
#define LW95B2_SET_CONTROL_PARAMS_MVMBCNT                                       13:13
#define LW95B2_SET_CONTROL_PARAMS_MVMBCNT_DISABLE                               (0x00000000)
#define LW95B2_SET_CONTROL_PARAMS_MVMBCNT_ENABLE                                (0x00000001)
#define LW95B2_SET_CONTROL_PARAMS_DEBUG_MODE                                    19:16
#define LW95B2_SET_CONTROL_PARAMS_APPLE_OOLD_ON                                 20:20
#define LW95B2_SET_CONTROL_PARAMS_ERROR_RES_ON                                  24:24
#define LW95B2_SET_CONTROL_PARAMS_ENABLE_INPUT_ENCRYPT                          25:25
#define LW95B2_SET_CONTROL_PARAMS_ENABLE_INPUT_ENCRYPT_DISABLE                  (0x00000000)
#define LW95B2_SET_CONTROL_PARAMS_ENABLE_INPUT_ENCRYPT_ENABLE                   (0x00000001)
#define LW95B2_SET_CONTROL_PARAMS_ERROR_FRM_IDX                                 31:26
#define LW95B2_SET_PICTURE_INDEX                                                (0x00000704)
#define LW95B2_SET_PICTURE_INDEX_INDEX                                          31:0
#define LW95B2_SET_DRV_PIC_SETUP_OFFSET                                         (0x00000710)
#define LW95B2_SET_DRV_PIC_SETUP_OFFSET_OFFSET                                  31:0
#define LW95B2_SET_VLD_DEC_CONTROL_OFFSET                                       (0x00000714)
#define LW95B2_SET_VLD_DEC_CONTROL_OFFSET_OFFSET                                31:0
#define LW95B2_SET_VLD_SLICE_DATA_OFFSET                                        (0x00000718)
#define LW95B2_SET_VLD_SLICE_DATA_OFFSET_OFFSET                                 31:0
#define LW95B2_SET_COLOC_DATA_OFFSET                                            (0x0000071C)
#define LW95B2_SET_COLOC_DATA_OFFSET_OFFSET                                     31:0
#define LW95B2_SET_HISTORY_OFFSET                                               (0x00000720)
#define LW95B2_SET_HISTORY_OFFSET_OFFSET                                        31:0
#define LW95B2_SET_FLOW_CTRL_OFFSET                                             (0x00000724)
#define LW95B2_SET_FLOW_CTRL_OFFSET_OFFSET                                      31:0
#define LW95B2_SET_PICTURE_OFFSET0                                              (0x0000072C)
#define LW95B2_SET_PICTURE_OFFSET0_OFFSET                                       31:0
#define LW95B2_SET_PICTURE_OFFSET1                                              (0x00000730)
#define LW95B2_SET_PICTURE_OFFSET1_OFFSET                                       31:0
#define LW95B2_SET_PICTURE_OFFSET2                                              (0x00000734)
#define LW95B2_SET_PICTURE_OFFSET2_OFFSET                                       31:0
#define LW95B2_SET_MVMB_STAT_OFFSET                                             (0x0000073C)
#define LW95B2_SET_MVMB_STAT_OFFSET_OFFSET                                      31:0
#define LW95B2_H264_SET_PICTURE_OFFSET17                                        (0x00000F08)
#define LW95B2_H264_SET_PICTURE_OFFSET17_OFFSET                                 31:0
#define LW95B2_SET_ILP_CONTROL_OFFSET                                           (0x00000F28)
#define LW95B2_SET_ILP_CONTROL_OFFSET_OFFSET                                    31:0
#define LW95B2_SET_ILP_DATA_OFFSET                                              (0x00000F38)
#define LW95B2_SET_ILP_DATA_OFFSET_OFFSET                                       31:0
#define LW95B2_PM_TRIGGER_END                                                   (0x00001114)
#define LW95B2_PM_TRIGGER_END_V                                                 31:0

#define LW95B2_ERROR_NONE                                                       (0x00000000)
#define LW95B2_OS_ERROR_EXELWTE_INSUFFICIENT_DATA                               (0x00000001)
#define LW95B2_OS_ERROR_SEMAPHORE_INSUFFICIENT_DATA                             (0x00000002)
#define LW95B2_OS_ERROR_ILWALID_METHOD                                          (0x00000003)
#define LW95B2_OS_ERROR_ILWALID_DMA_PAGE                                        (0x00000004)
#define LW95B2_OS_ERROR_UNHANDLED_INTERRUPT                                     (0x00000005)
#define LW95B2_OS_ERROR_EXCEPTION                                               (0x00000006)
#define LW95B2_OS_ERROR_ILWALID_CTXSW_REQUEST                                   (0x00000007)
#define LW95B2_OS_ERROR_APPLICATION                                             (0x00000008)
#define LW95B2_OS_ERROR_SW_BREAKPT                                              (0x00000009)
#define LW95B2_OS_INTERRUPT_EXELWTE_AWAKEN                                      (0x00000100)
#define LW95B2_OS_INTERRUPT_BACKEND_SEMAPHORE_AWAKEN                            (0x00000200)
#define LW95B2_OS_INTERRUPT_CTX_ERROR_FBIF                                      (0x00000300)
#define LW95B2_OS_INTERRUPT_LIMIT_VIOLATION                                     (0x00000400)
#define LW95B2_OS_INTERRUPT_LIMIT_AND_FBIF_CTX_ERROR                            (0x00000500)
#define LW95B2_OS_INTERRUPT_HALT_ENGINE                                         (0x00000600)
#define LW95B2_OS_INTERRUPT_TRAP_NONSTALL                                       (0x00000700)
#define LW95B2_DEC_ERROR_MPEG12_APPTIMER_EXPIRED                                (0xDEC10001)
#define LW95B2_DEC_ERROR_MPEG12_MVTIMER_EXPIRED                                 (0xDEC10002)
#define LW95B2_DEC_ERROR_MPEG12_ILWALID_TOKEN                                   (0xDEC10003)
#define LW95B2_DEC_ERROR_MPEG12_SLICEDATA_MISSING                               (0xDEC10004)
#define LW95B2_DEC_ERROR_MPEG12_HWERR_INTERRUPT                                 (0xDEC10005)
#define LW95B2_DEC_ERROR_MPEG12_DETECTED_VLD_FAILURE                            (0xDEC10006)
#define LW95B2_DEC_ERROR_MPEG12_PICTURE_INIT                                    (0xDEC10100)
#define LW95B2_DEC_ERROR_MPEG12_STATEMACHINE_FAILURE                            (0xDEC10101)
#define LW95B2_DEC_ERROR_MPEG12_ILWALID_CTXID_PIC                               (0xDEC10901)
#define LW95B2_DEC_ERROR_MPEG12_ILWALID_CTXID_UCODE                             (0xDEC10902)
#define LW95B2_DEC_ERROR_MPEG12_ILWALID_CTXID_FC                                (0xDEC10903)
#define LW95B2_DEC_ERROR_MPEG12_ILWALID_CTXID_SLH                               (0xDEC10904)
#define LW95B2_DEC_ERROR_MPEG12_ILWALID_UCODE_SIZE                              (0xDEC10905)
#define LW95B2_DEC_ERROR_MPEG12_ILWALID_SLICE_COUNT                             (0xDEC10906)
#define LW95B2_DEC_ERROR_VC1_APPTIMER_EXPIRED                                   (0xDEC20001)
#define LW95B2_DEC_ERROR_VC1_MVTIMER_EXPIRED                                    (0xDEC20002)
#define LW95B2_DEC_ERROR_VC1_ILWALID_TOKEN                                      (0xDEC20003)
#define LW95B2_DEC_ERROR_VC1_SLICEDATA_MISSING                                  (0xDEC20004)
#define LW95B2_DEC_ERROR_VC1_HWERR_INTERRUPT                                    (0xDEC20005)
#define LW95B2_DEC_ERROR_VC1_DETECTED_VLD_FAILURE                               (0xDEC20006)
#define LW95B2_DEC_ERROR_VC1_TIMEOUT_POLLING_FOR_DATA                           (0xDEC20007)
#define LW95B2_DEC_ERROR_VC1_PDEC_PIC_END_UNALIGNED                             (0xDEC20008)
#define LW95B2_DEC_ERROR_VC1_WDTIMER_EXPIRED                                    (0xDEC20009)
#define LW95B2_DEC_ERROR_VC1_ERRINTSTART                                        (0xDEC20010)
#define LW95B2_DEC_ERROR_VC1_IQT_ERRINT                                         (0xDEC20011)
#define LW95B2_DEC_ERROR_VC1_MC_ERRINT                                          (0xDEC20012)
#define LW95B2_DEC_ERROR_VC1_MC_IQT_ERRINT                                      (0xDEC20013)
#define LW95B2_DEC_ERROR_VC1_REC_ERRINT                                         (0xDEC20014)
#define LW95B2_DEC_ERROR_VC1_REC_IQT_ERRINT                                     (0xDEC20015)
#define LW95B2_DEC_ERROR_VC1_REC_MC_ERRINT                                      (0xDEC20016)
#define LW95B2_DEC_ERROR_VC1_REC_MC_IQT_ERRINT                                  (0xDEC20017)
#define LW95B2_DEC_ERROR_VC1_DBF_ERRINT                                         (0xDEC20018)
#define LW95B2_DEC_ERROR_VC1_DBF_IQT_ERRINT                                     (0xDEC20019)
#define LW95B2_DEC_ERROR_VC1_DBF_MC_ERRINT                                      (0xDEC2001A)
#define LW95B2_DEC_ERROR_VC1_DBF_MC_IQT_ERRINT                                  (0xDEC2001B)
#define LW95B2_DEC_ERROR_VC1_DBF_REC_ERRINT                                     (0xDEC2001C)
#define LW95B2_DEC_ERROR_VC1_DBF_REC_IQT_ERRINT                                 (0xDEC2001D)
#define LW95B2_DEC_ERROR_VC1_DBF_REC_MC_ERRINT                                  (0xDEC2001E)
#define LW95B2_DEC_ERROR_VC1_DBF_REC_MC_IQT_ERRINT                              (0xDEC2001F)
#define LW95B2_DEC_ERROR_VC1_PICTURE_INIT                                       (0xDEC20100)
#define LW95B2_DEC_ERROR_VC1_STATEMACHINE_FAILURE                               (0xDEC20101)
#define LW95B2_DEC_ERROR_VC1_ILWALID_CTXID_PIC                                  (0xDEC20901)
#define LW95B2_DEC_ERROR_VC1_ILWALID_CTXID_UCODE                                (0xDEC20902)
#define LW95B2_DEC_ERROR_VC1_ILWALID_CTXID_FC                                   (0xDEC20903)
#define LW95B2_DEC_ERROR_VC1_ILWAILD_CTXID_SLH                                  (0xDEC20904)
#define LW95B2_DEC_ERROR_VC1_ILWALID_UCODE_SIZE                                 (0xDEC20905)
#define LW95B2_DEC_ERROR_VC1_ILWALID_SLICE_COUNT                                (0xDEC20906)
#define LW95B2_DEC_ERROR_H264_APPTIMER_EXPIRED                                  (0xDEC30001)
#define LW95B2_DEC_ERROR_H264_MVTIMER_EXPIRED                                   (0xDEC30002)
#define LW95B2_DEC_ERROR_H264_ILWALID_TOKEN                                     (0xDEC30003)
#define LW95B2_DEC_ERROR_H264_SLICEDATA_MISSING                                 (0xDEC30004)
#define LW95B2_DEC_ERROR_H264_HWERR_INTERRUPT                                   (0xDEC30005)
#define LW95B2_DEC_ERROR_H264_DETECTED_VLD_FAILURE                              (0xDEC30006)
#define LW95B2_DEC_ERROR_H264_ERRINTSTART                                       (0xDEC30010)
#define LW95B2_DEC_ERROR_H264_IQT_ERRINT                                        (0xDEC30011)
#define LW95B2_DEC_ERROR_H264_MC_ERRINT                                         (0xDEC30012)
#define LW95B2_DEC_ERROR_H264_MC_IQT_ERRINT                                     (0xDEC30013)
#define LW95B2_DEC_ERROR_H264_REC_ERRINT                                        (0xDEC30014)
#define LW95B2_DEC_ERROR_H264_REC_IQT_ERRINT                                    (0xDEC30015)
#define LW95B2_DEC_ERROR_H264_REC_MC_ERRINT                                     (0xDEC30016)
#define LW95B2_DEC_ERROR_H264_REC_MC_IQT_ERRINT                                 (0xDEC30017)
#define LW95B2_DEC_ERROR_H264_DBF_ERRINT                                        (0xDEC30018)
#define LW95B2_DEC_ERROR_H264_DBF_IQT_ERRINT                                    (0xDEC30019)
#define LW95B2_DEC_ERROR_H264_DBF_MC_ERRINT                                     (0xDEC3001A)
#define LW95B2_DEC_ERROR_H264_DBF_MC_IQT_ERRINT                                 (0xDEC3001B)
#define LW95B2_DEC_ERROR_H264_DBF_REC_ERRINT                                    (0xDEC3001C)
#define LW95B2_DEC_ERROR_H264_DBF_REC_IQT_ERRINT                                (0xDEC3001D)
#define LW95B2_DEC_ERROR_H264_DBF_REC_MC_ERRINT                                 (0xDEC3001E)
#define LW95B2_DEC_ERROR_H264_DBF_REC_MC_IQT_ERRINT                             (0xDEC3001F)
#define LW95B2_DEC_ERROR_H264_PICTURE_INIT                                      (0xDEC30100)
#define LW95B2_DEC_ERROR_H264_STATEMACHINE_FAILURE                              (0xDEC30101)
#define LW95B2_DEC_ERROR_H264_ILWALID_CTXID_PIC                                 (0xDEC30901)
#define LW95B2_DEC_ERROR_H264_ILWALID_CTXID_UCODE                               (0xDEC30902)
#define LW95B2_DEC_ERROR_H264_ILWALID_CTXID_FC                                  (0xDEC30903)
#define LW95B2_DEC_ERROR_H264_ILWALID_CTXID_SLH                                 (0xDEC30904)
#define LW95B2_DEC_ERROR_H264_ILWALID_UCODE_SIZE                                (0xDEC30905)
#define LW95B2_DEC_ERROR_H264_ILWALID_SLICE_COUNT                               (0xDEC30906)
#define LW95B2_DEC_ERROR_MPEG4_APPTIMER_EXPIRED                                 (0xDEC40001)
#define LW95B2_DEC_ERROR_MPEG4_MVTIMER_EXPIRED                                  (0xDEC40002)
#define LW95B2_DEC_ERROR_MPEG4_ILWALID_TOKEN                                    (0xDEC40003)
#define LW95B2_DEC_ERROR_MPEG4_SLICEDATA_MISSING                                (0xDEC40004)
#define LW95B2_DEC_ERROR_MPEG4_HWERR_INTERRUPT                                  (0xDEC40005)
#define LW95B2_DEC_ERROR_MPEG4_DETECTED_VLD_FAILURE                             (0xDEC40006)
#define LW95B2_DEC_ERROR_MPEG4_TIMEOUT_POLLING_FOR_DATA                         (0xDEC40007)
#define LW95B2_DEC_ERROR_MPEG4_PDEC_PIC_END_UNALIGNED                           (0xDEC40008)
#define LW95B2_DEC_ERROR_MPEG4_WDTIMER_EXPIRED                                  (0xDEC40009)
#define LW95B2_DEC_ERROR_MPEG4_ERRINTSTART                                      (0xDEC40010)
#define LW95B2_DEC_ERROR_MPEG4_IQT_ERRINT                                       (0xDEC40011)
#define LW95B2_DEC_ERROR_MPEG4_MC_ERRINT                                        (0xDEC40012)
#define LW95B2_DEC_ERROR_MPEG4_MC_IQT_ERRINT                                    (0xDEC40013)
#define LW95B2_DEC_ERROR_MPEG4_REC_ERRINT                                       (0xDEC40014)
#define LW95B2_DEC_ERROR_MPEG4_REC_IQT_ERRINT                                   (0xDEC40015)
#define LW95B2_DEC_ERROR_MPEG4_REC_MC_ERRINT                                    (0xDEC40016)
#define LW95B2_DEC_ERROR_MPEG4_REC_MC_IQT_ERRINT                                (0xDEC40017)
#define LW95B2_DEC_ERROR_MPEG4_DBF_ERRINT                                       (0xDEC40018)
#define LW95B2_DEC_ERROR_MPEG4_DBF_IQT_ERRINT                                   (0xDEC40019)
#define LW95B2_DEC_ERROR_MPEG4_DBF_MC_ERRINT                                    (0xDEC4001A)
#define LW95B2_DEC_ERROR_MPEG4_DBF_MC_IQT_ERRINT                                (0xDEC4001B)
#define LW95B2_DEC_ERROR_MPEG4_DBF_REC_ERRINT                                   (0xDEC4001C)
#define LW95B2_DEC_ERROR_MPEG4_DBF_REC_IQT_ERRINT                               (0xDEC4001D)
#define LW95B2_DEC_ERROR_MPEG4_DBF_REC_MC_ERRINT                                (0xDEC4001E)
#define LW95B2_DEC_ERROR_MPEG4_DBF_REC_MC_IQT_ERRINT                            (0xDEC4001F)
#define LW95B2_DEC_ERROR_MPEG4_PICTURE_INIT                                     (0xDEC40100)
#define LW95B2_DEC_ERROR_MPEG4_STATEMACHINE_FAILURE                             (0xDEC40101)
#define LW95B2_DEC_ERROR_MPEG4_ILWALID_CTXID_PIC                                (0xDEC40901)
#define LW95B2_DEC_ERROR_MPEG4_ILWALID_CTXID_UCODE                              (0xDEC40902)
#define LW95B2_DEC_ERROR_MPEG4_ILWALID_CTXID_FC                                 (0xDEC40903)
#define LW95B2_DEC_ERROR_MPEG4_ILWALID_CTXID_SLH                                (0xDEC40904)
#define LW95B2_DEC_ERROR_MPEG4_ILWALID_UCODE_SIZE                               (0xDEC40905)
#define LW95B2_DEC_ERROR_MPEG4_ILWALID_SLICE_COUNT                              (0xDEC40906)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl95b2_h

