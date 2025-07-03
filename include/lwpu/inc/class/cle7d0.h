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

#ifndef _cle7d0_h_
#define _cle7d0_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LWE7D0_VIDEO_LWJPG                                                               (0x0000E7D0)

typedef volatile struct _cle7d0_tag0 {
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
    LwV32 Reserved05[0xFE];
    LwV32 SetControlParams;                                                     // 0x00000700 - 0x00000703
    LwV32 SetPictureIndex;                                                      // 0x00000704 - 0x00000707
    LwV32 SetInDrvPicSetup;                                                     // 0x00000708 - 0x0000070B
    LwV32 SetOutStatus;                                                         // 0x0000070C - 0x0000070F
    LwV32 SetBitstream;                                                         // 0x00000710 - 0x00000713
    LwV32 SetLwrPic;                                                            // 0x00000714 - 0x00000717
    LwV32 SetLwrPicChromaU;                                                     // 0x00000718 - 0x0000071B
    LwV32 SetLwrPicChromaV;                                                     // 0x0000071C - 0x0000071F
    LwV32 Reserved06[0x27D];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved07[0x3BA];
} LWE7D0_VIDEO_LWJPGControlPio;

#define LWE7D0_NOP                                                              (0x00000100)
#define LWE7D0_NOP_PARAMETER                                                    31:0
#define LWE7D0_PM_TRIGGER                                                       (0x00000140)
#define LWE7D0_PM_TRIGGER_V                                                     31:0
#define LWE7D0_SET_APPLICATION_ID                                               (0x00000200)
#define LWE7D0_SET_APPLICATION_ID_ID                                            31:0
#define LWE7D0_SET_APPLICATION_ID_ID_LWJPG_DECODER                              (0x00000001)
#define LWE7D0_SET_APPLICATION_ID_ID_LWJPG_ENCODER                              (0x00000002)
#define LWE7D0_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LWE7D0_SET_WATCHDOG_TIMER_TIMER                                         31:0
#define LWE7D0_SEMAPHORE_A                                                      (0x00000240)
#define LWE7D0_SEMAPHORE_A_UPPER                                                7:0
#define LWE7D0_SEMAPHORE_B                                                      (0x00000244)
#define LWE7D0_SEMAPHORE_B_LOWER                                                31:0
#define LWE7D0_SEMAPHORE_C                                                      (0x00000248)
#define LWE7D0_SEMAPHORE_C_PAYLOAD                                              31:0
#define LWE7D0_CTX_SAVE_AREA                                                    (0x0000024C)
#define LWE7D0_CTX_SAVE_AREA_OFFSET                                             27:0
#define LWE7D0_CTX_SAVE_AREA_CTX_VALID                                          31:28
#define LWE7D0_CTX_SWITCH                                                       (0x00000250)
#define LWE7D0_CTX_SWITCH_RESTORE                                               0:0
#define LWE7D0_CTX_SWITCH_RESTORE_FALSE                                         (0x00000000)
#define LWE7D0_CTX_SWITCH_RESTORE_TRUE                                          (0x00000001)
#define LWE7D0_CTX_SWITCH_RST_NOTIFY                                            1:1
#define LWE7D0_CTX_SWITCH_RST_NOTIFY_FALSE                                      (0x00000000)
#define LWE7D0_CTX_SWITCH_RST_NOTIFY_TRUE                                       (0x00000001)
#define LWE7D0_CTX_SWITCH_RESERVED                                              7:2
#define LWE7D0_CTX_SWITCH_ASID                                                  23:8
#define LWE7D0_EXELWTE                                                          (0x00000300)
#define LWE7D0_EXELWTE_NOTIFY                                                   0:0
#define LWE7D0_EXELWTE_NOTIFY_DISABLE                                           (0x00000000)
#define LWE7D0_EXELWTE_NOTIFY_ENABLE                                            (0x00000001)
#define LWE7D0_EXELWTE_NOTIFY_ON                                                1:1
#define LWE7D0_EXELWTE_NOTIFY_ON_END                                            (0x00000000)
#define LWE7D0_EXELWTE_NOTIFY_ON_BEGIN                                          (0x00000001)
#define LWE7D0_EXELWTE_AWAKEN                                                   8:8
#define LWE7D0_EXELWTE_AWAKEN_DISABLE                                           (0x00000000)
#define LWE7D0_EXELWTE_AWAKEN_ENABLE                                            (0x00000001)
#define LWE7D0_SEMAPHORE_D                                                      (0x00000304)
#define LWE7D0_SEMAPHORE_D_STRUCTURE_SIZE                                       0:0
#define LWE7D0_SEMAPHORE_D_STRUCTURE_SIZE_ONE                                   (0x00000000)
#define LWE7D0_SEMAPHORE_D_STRUCTURE_SIZE_FOUR                                  (0x00000001)
#define LWE7D0_SEMAPHORE_D_AWAKEN_ENABLE                                        8:8
#define LWE7D0_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                  (0x00000000)
#define LWE7D0_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                   (0x00000001)
#define LWE7D0_SEMAPHORE_D_OPERATION                                            17:16
#define LWE7D0_SEMAPHORE_D_OPERATION_RELEASE                                    (0x00000000)
#define LWE7D0_SEMAPHORE_D_OPERATION_RESERVED0                                  (0x00000001)
#define LWE7D0_SEMAPHORE_D_OPERATION_RESERVED1                                  (0x00000002)
#define LWE7D0_SEMAPHORE_D_OPERATION_TRAP                                       (0x00000003)
#define LWE7D0_SEMAPHORE_D_FLUSH_DISABLE                                        21:21
#define LWE7D0_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                  (0x00000000)
#define LWE7D0_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                   (0x00000001)
#define LWE7D0_SET_CONTROL_PARAMS                                               (0x00000700)
#define LWE7D0_SET_CONTROL_PARAMS_GPTIMER_ON                                    0:0
#define LWE7D0_SET_CONTROL_PARAMS_DUMP_CYCLE_COUNT                              1:1
#define LWE7D0_SET_CONTROL_PARAMS_DEBUG_MODE                                    2:2
#define LWE7D0_SET_PICTURE_INDEX                                                (0x00000704)
#define LWE7D0_SET_PICTURE_INDEX_INDEX                                          31:0
#define LWE7D0_SET_IN_DRV_PIC_SETUP                                             (0x00000708)
#define LWE7D0_SET_IN_DRV_PIC_SETUP_OFFSET                                      31:0
#define LWE7D0_SET_OUT_STATUS                                                   (0x0000070C)
#define LWE7D0_SET_OUT_STATUS_OFFSET                                            31:0
#define LWE7D0_SET_BITSTREAM                                                    (0x00000710)
#define LWE7D0_SET_BITSTREAM_OFFSET                                             31:0
#define LWE7D0_SET_LWR_PIC                                                      (0x00000714)
#define LWE7D0_SET_LWR_PIC_OFFSET                                               31:0
#define LWE7D0_SET_LWR_PIC_CHROMA_U                                             (0x00000718)
#define LWE7D0_SET_LWR_PIC_CHROMA_U_OFFSET                                      31:0
#define LWE7D0_SET_LWR_PIC_CHROMA_V                                             (0x0000071C)
#define LWE7D0_SET_LWR_PIC_CHROMA_V_OFFSET                                      31:0
#define LWE7D0_PM_TRIGGER_END                                                   (0x00001114)
#define LWE7D0_PM_TRIGGER_END_V                                                 31:0

#define LWE7D0_ERROR_NONE                                                       (0x00000000)
#define LWE7D0_OS_ERROR_EXELWTE_INSUFFICIENT_DATA                               (0x00000001)
#define LWE7D0_OS_ERROR_SEMAPHORE_INSUFFICIENT_DATA                             (0x00000002)
#define LWE7D0_OS_ERROR_ILWALID_METHOD                                          (0x00000003)
#define LWE7D0_OS_ERROR_ILWALID_DMA_PAGE                                        (0x00000004)
#define LWE7D0_OS_ERROR_UNHANDLED_INTERRUPT                                     (0x00000005)
#define LWE7D0_OS_ERROR_EXCEPTION                                               (0x00000006)
#define LWE7D0_OS_ERROR_ILWALID_CTXSW_REQUEST                                   (0x00000007)
#define LWE7D0_OS_ERROR_APPLICATION                                             (0x00000008)
#define LWE7D0_OS_INTERRUPT_EXELWTE_AWAKEN                                      (0x00000100)
#define LWE7D0_OS_INTERRUPT_BACKEND_SEMAPHORE_AWAKEN                            (0x00000200)
#define LWE7D0_OS_INTERRUPT_CTX_ERROR_FBIF                                      (0x00000300)
#define LWE7D0_OS_INTERRUPT_LIMIT_VIOLATION                                     (0x00000400)
#define LWE7D0_OS_INTERRUPT_LIMIT_AND_FBIF_CTX_ERROR                            (0x00000500)
#define LWE7D0_OS_INTERRUPT_HALT_ENGINE                                         (0x00000600)
#define LWE7D0_OS_INTERRUPT_TRAP_NONSTALL                                       (0x00000700)
#define LWE7D0_OS_INTERRUPT_CTX_SAVE_DONE                                       (0x00000800)
#define LWE7D0_OS_INTERRUPT_CTX_RESTORE_DONE                                    (0x00000900)
#define LWE7D0_ERROR_JPGAPPTIMER_EXPIRED                                        (0x30000001)
#define LWE7D0_ERROR_JPGILWALID_INPUT                                           (0x30000002)
#define LWE7D0_ERROR_JPGHWERR_INTERRUPT                                         (0x30000003)
#define LWE7D0_ERROR_JPGBAD_MAGIC                                               (0x30000004)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cle7d0_h

