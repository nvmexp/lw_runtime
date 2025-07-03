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


#ifndef cl90b7_h
#define cl90b7_h

#include "lwtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LW90B7_VIDEO_ENCODER                                                             (0x000090B7)

typedef volatile struct _cl90b7_tag0 {
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
    LwV32 Reserved05[0x3E];
    LwV32 SetInRefPic0;                                                         // 0x00000400 - 0x00000403
    LwV32 SetInRefPic1;                                                         // 0x00000404 - 0x00000407
    LwV32 SetInRefPic2;                                                         // 0x00000408 - 0x0000040B
    LwV32 SetInRefPic3;                                                         // 0x0000040C - 0x0000040F
    LwV32 SetInRefPic4;                                                         // 0x00000410 - 0x00000413
    LwV32 SetInRefPic5;                                                         // 0x00000414 - 0x00000417
    LwV32 SetInRefPic6;                                                         // 0x00000418 - 0x0000041B
    LwV32 SetInRefPic7;                                                         // 0x0000041C - 0x0000041F
    LwV32 SetInRefPic8;                                                         // 0x00000420 - 0x00000423
    LwV32 SetInRefPic9;                                                         // 0x00000424 - 0x00000427
    LwV32 SetInRefPic10;                                                        // 0x00000428 - 0x0000042B
    LwV32 SetInRefPic11;                                                        // 0x0000042C - 0x0000042F
    LwV32 SetInRefPic12;                                                        // 0x00000430 - 0x00000433
    LwV32 SetInRefPic13;                                                        // 0x00000434 - 0x00000437
    LwV32 SetInRefPic14;                                                        // 0x00000438 - 0x0000043B
    LwV32 SetInRefPic15;                                                        // 0x0000043C - 0x0000043F
    LwV32 Reserved06[0xB0];
    LwV32 SetControlParams;                                                     // 0x00000700 - 0x00000703
    LwV32 SetPictureIndex;                                                      // 0x00000704 - 0x00000707
    LwV32 Reserved07[0x1];
    LwV32 SetInRCData;                                                          // 0x0000070C - 0x0000070F
    LwV32 SetInDrvPicSetup;                                                     // 0x00000710 - 0x00000713
    LwV32 SetInCEAHintsData;                                                    // 0x00000714 - 0x00000717
    LwV32 SetOutEncStatus;                                                      // 0x00000718 - 0x0000071B
    LwV32 SetOutBitstream;                                                      // 0x0000071C - 0x0000071F
    LwV32 SetIOHistory;                                                         // 0x00000720 - 0x00000723
    LwV32 SetIoRcProcess;                                                       // 0x00000724 - 0x00000727
    LwV32 SetInColocData;                                                       // 0x00000728 - 0x0000072B
    LwV32 SetOutColocData;                                                      // 0x0000072C - 0x0000072F
    LwV32 SetOutRefPic;                                                         // 0x00000730 - 0x00000733
    LwV32 SetInLwrPic;                                                          // 0x00000734 - 0x00000737
    LwV32 SetInMEPredData;                                                      // 0x00000738 - 0x0000073B
    LwV32 SetOutMEPredData;                                                     // 0x0000073C - 0x0000073F
    LwV32 Reserved08[0x275];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved09[0x3BA];
} LW90B7_VIDEO_ENCODERControlPio;

#define LW90B7_NOP                                                              (0x00000100)
#define LW90B7_NOP_PARAMETER                                                    31:0
#define LW90B7_PM_TRIGGER                                                       (0x00000140)
#define LW90B7_PM_TRIGGER_V                                                     31:0
#define LW90B7_SET_APPLICATION_ID                                               (0x00000200)
#define LW90B7_SET_APPLICATION_ID_ID                                            31:0
#define LW90B7_SET_APPLICATION_ID_ID_MSENC_H264                                 (0x00000000)
#define LW90B7_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LW90B7_SET_WATCHDOG_TIMER_TIMER                                         31:0
#define LW90B7_SEMAPHORE_A                                                      (0x00000240)
#define LW90B7_SEMAPHORE_A_UPPER                                                7:0
#define LW90B7_SEMAPHORE_B                                                      (0x00000244)
#define LW90B7_SEMAPHORE_B_LOWER                                                31:0
#define LW90B7_SEMAPHORE_C                                                      (0x00000248)
#define LW90B7_SEMAPHORE_C_PAYLOAD                                              31:0
#define LW90B7_EXELWTE                                                          (0x00000300)
#define LW90B7_EXELWTE_NOTIFY                                                   0:0
#define LW90B7_EXELWTE_NOTIFY_DISABLE                                           (0x00000000)
#define LW90B7_EXELWTE_NOTIFY_ENABLE                                            (0x00000001)
#define LW90B7_EXELWTE_NOTIFY_ON                                                1:1
#define LW90B7_EXELWTE_NOTIFY_ON_END                                            (0x00000000)
#define LW90B7_EXELWTE_NOTIFY_ON_BEGIN                                          (0x00000001)
#define LW90B7_EXELWTE_AWAKEN                                                   8:8
#define LW90B7_EXELWTE_AWAKEN_DISABLE                                           (0x00000000)
#define LW90B7_EXELWTE_AWAKEN_ENABLE                                            (0x00000001)
#define LW90B7_SEMAPHORE_D                                                      (0x00000304)
#define LW90B7_SEMAPHORE_D_STRUCTURE_SIZE                                       0:0
#define LW90B7_SEMAPHORE_D_STRUCTURE_SIZE_ONE                                   (0x00000000)
#define LW90B7_SEMAPHORE_D_STRUCTURE_SIZE_FOUR                                  (0x00000001)
#define LW90B7_SEMAPHORE_D_AWAKEN_ENABLE                                        8:8
#define LW90B7_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                  (0x00000000)
#define LW90B7_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                   (0x00000001)
#define LW90B7_SEMAPHORE_D_OPERATION                                            17:16
#define LW90B7_SEMAPHORE_D_OPERATION_RELEASE                                    (0x00000000)
#define LW90B7_SEMAPHORE_D_OPERATION_RESERVED0                                  (0x00000001)
#define LW90B7_SEMAPHORE_D_OPERATION_RESERVED1                                  (0x00000002)
#define LW90B7_SEMAPHORE_D_OPERATION_TRAP                                       (0x00000003)
#define LW90B7_SEMAPHORE_D_FLUSH_DISABLE                                        21:21
#define LW90B7_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                  (0x00000000)
#define LW90B7_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                   (0x00000001)
#define LW90B7_SET_IN_REF_PIC0                                                  (0x00000400)
#define LW90B7_SET_IN_REF_PIC0_OFFSET                                           31:0
#define LW90B7_SET_IN_REF_PIC1                                                  (0x00000404)
#define LW90B7_SET_IN_REF_PIC1_OFFSET                                           31:0
#define LW90B7_SET_IN_REF_PIC2                                                  (0x00000408)
#define LW90B7_SET_IN_REF_PIC2_OFFSET                                           31:0
#define LW90B7_SET_IN_REF_PIC3                                                  (0x0000040C)
#define LW90B7_SET_IN_REF_PIC3_OFFSET                                           31:0
#define LW90B7_SET_IN_REF_PIC4                                                  (0x00000410)
#define LW90B7_SET_IN_REF_PIC4_OFFSET                                           31:0
#define LW90B7_SET_IN_REF_PIC5                                                  (0x00000414)
#define LW90B7_SET_IN_REF_PIC5_OFFSET                                           31:0
#define LW90B7_SET_IN_REF_PIC6                                                  (0x00000418)
#define LW90B7_SET_IN_REF_PIC6_OFFSET                                           31:0
#define LW90B7_SET_IN_REF_PIC7                                                  (0x0000041C)
#define LW90B7_SET_IN_REF_PIC7_OFFSET                                           31:0
#define LW90B7_SET_IN_REF_PIC8                                                  (0x00000420)
#define LW90B7_SET_IN_REF_PIC8_OFFSET                                           31:0
#define LW90B7_SET_IN_REF_PIC9                                                  (0x00000424)
#define LW90B7_SET_IN_REF_PIC9_OFFSET                                           31:0
#define LW90B7_SET_IN_REF_PIC10                                                 (0x00000428)
#define LW90B7_SET_IN_REF_PIC10_OFFSET                                          31:0
#define LW90B7_SET_IN_REF_PIC11                                                 (0x0000042C)
#define LW90B7_SET_IN_REF_PIC11_OFFSET                                          31:0
#define LW90B7_SET_IN_REF_PIC12                                                 (0x00000430)
#define LW90B7_SET_IN_REF_PIC12_OFFSET                                          31:0
#define LW90B7_SET_IN_REF_PIC13                                                 (0x00000434)
#define LW90B7_SET_IN_REF_PIC13_OFFSET                                          31:0
#define LW90B7_SET_IN_REF_PIC14                                                 (0x00000438)
#define LW90B7_SET_IN_REF_PIC14_OFFSET                                          31:0
#define LW90B7_SET_IN_REF_PIC15                                                 (0x0000043C)
#define LW90B7_SET_IN_REF_PIC15_OFFSET                                          31:0
#define LW90B7_SET_CONTROL_PARAMS                                               (0x00000700)
#define LW90B7_SET_CONTROL_PARAMS_CODEC_TYPE                                    3:0
#define LW90B7_SET_CONTROL_PARAMS_CODEC_TYPE_H264                               (0x00000003)
#define LW90B7_SET_CONTROL_PARAMS_FORCE_OUT_PIC                                 8:8
#define LW90B7_SET_CONTROL_PARAMS_FORCE_OUT_COL                                 9:9
#define LW90B7_SET_CONTROL_PARAMS_MEONLY                                        10:10
#define LW90B7_SET_CONTROL_PARAMS_SLICE_STAT_ON                                 11:11
#define LW90B7_SET_CONTROL_PARAMS_GPTIMER_ON                                    12:12
#define LW90B7_SET_CONTROL_PARAMS_DEBUG_MODE                                    16:16
#define LW90B7_SET_CONTROL_PARAMS_RCMODE                                        31:24
#define LW90B7_SET_CONTROL_PARAMS_RCMODE_CONSTQP                                (0x00000000)
#define LW90B7_SET_PICTURE_INDEX                                                (0x00000704)
#define LW90B7_SET_PICTURE_INDEX_INDEX                                          31:0
#define LW90B7_SET_IN_RCDATA                                                    (0x0000070C)
#define LW90B7_SET_IN_RCDATA_OFFSET                                             31:0
#define LW90B7_SET_IN_DRV_PIC_SETUP                                             (0x00000710)
#define LW90B7_SET_IN_DRV_PIC_SETUP_OFFSET                                      31:0
#define LW90B7_SET_IN_CEAHINTS_DATA                                             (0x00000714)
#define LW90B7_SET_IN_CEAHINTS_DATA_OFFSET                                      31:0
#define LW90B7_SET_OUT_ENC_STATUS                                               (0x00000718)
#define LW90B7_SET_OUT_ENC_STATUS_OFFSET                                        31:0
#define LW90B7_SET_OUT_BITSTREAM                                                (0x0000071C)
#define LW90B7_SET_OUT_BITSTREAM_OFFSET                                         31:0
#define LW90B7_SET_IOHISTORY                                                    (0x00000720)
#define LW90B7_SET_IOHISTORY_OFFSET                                             31:0
#define LW90B7_SET_IO_RC_PROCESS                                                (0x00000724)
#define LW90B7_SET_IO_RC_PROCESS_OFFSET                                         31:0
#define LW90B7_SET_IN_COLOC_DATA                                                (0x00000728)
#define LW90B7_SET_IN_COLOC_DATA_OFFSET                                         31:0
#define LW90B7_SET_OUT_COLOC_DATA                                               (0x0000072C)
#define LW90B7_SET_OUT_COLOC_DATA_OFFSET                                        31:0
#define LW90B7_SET_OUT_REF_PIC                                                  (0x00000730)
#define LW90B7_SET_OUT_REF_PIC_OFFSET                                           31:0
#define LW90B7_SET_IN_LWR_PIC                                                   (0x00000734)
#define LW90B7_SET_IN_LWR_PIC_OFFSET                                            31:0
#define LW90B7_SET_IN_MEPRED_DATA                                               (0x00000738)
#define LW90B7_SET_IN_MEPRED_DATA_OFFSET                                        31:0
#define LW90B7_SET_OUT_MEPRED_DATA                                              (0x0000073C)
#define LW90B7_SET_OUT_MEPRED_DATA_OFFSET                                       31:0
#define LW90B7_PM_TRIGGER_END                                                   (0x00001114)
#define LW90B7_PM_TRIGGER_END_V                                                 31:0

#define LW90B7_ERROR_NONE                                                       (0x00000000)
#define LW90B7_OS_ERROR_EXELWTE_INSUFFICIENT_DATA                               (0x00000001)
#define LW90B7_OS_ERROR_SEMAPHORE_INSUFFICIENT_DATA                             (0x00000002)
#define LW90B7_OS_ERROR_ILWALID_METHOD                                          (0x00000003)
#define LW90B7_OS_ERROR_ILWALID_DMA_PAGE                                        (0x00000004)
#define LW90B7_OS_ERROR_UNHANDLED_INTERRUPT                                     (0x00000005)
#define LW90B7_OS_ERROR_EXCEPTION                                               (0x00000006)
#define LW90B7_OS_ERROR_ILWALID_CTXSW_REQUEST                                   (0x00000007)
#define LW90B7_OS_ERROR_APPLICATION                                             (0x00000008)
#define LW90B7_OS_INTERRUPT_EXELWTE_AWAKEN                                      (0x00000100)
#define LW90B7_OS_INTERRUPT_BACKEND_SEMAPHORE_AWAKEN                            (0x00000200)
#define LW90B7_OS_INTERRUPT_HALT_ENGINE                                         (0x00000600)
#define LW90B7_ENC_ERROR_H264_APPTIMER_EXPIRED                                  (0x30000001)
#define LW90B7_ENC_ERROR_H264_ILWALID_INPUT                                     (0x30000002)
#define LW90B7_ENC_ERROR_H264_HWERR_INTERRUPT                                   (0x30000003)
#define LW90B7_ENC_ERROR_H264_BAD_MAGIC                                         (0x30000004)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // cl90b7_h

