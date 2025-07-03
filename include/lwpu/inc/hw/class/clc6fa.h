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


#include "lwtypes.h"

#ifndef _clc6fa_h_
#define _clc6fa_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LWC6FA_VIDEO_OFA                                                                 (0x0000C6FA)

typedef volatile struct _clc6fa_tag0 {
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
    LwV32 SetPredicationOffsetUpper;                                            // 0x00000308 - 0x0000030B
    LwV32 SetPredicationOffsetLower;                                            // 0x0000030C - 0x0000030F
    LwV32 SetAuxiliaryDataBuffer;                                               // 0x00000310 - 0x00000313
    LwV32 Reserved05[0xFB];
    LwV32 SetPictureIndex;                                                      // 0x00000700 - 0x00000703
    LwV32 SetControlParams;                                                     // 0x00000704 - 0x00000707
    LwV32 SetTotalLevelNum;                                                     // 0x00000708 - 0x0000070B
    struct {
        LwV32 SetLevelIndex;                                                    // 0x0000070C - 0x0000070F
        LwV32 SetDrvSetUpAddr;                                                  // 0x00000710 - 0x00000713
        LwV32 SetLwrrPicAddr;                                                   // 0x00000714 - 0x00000717
        LwV32 SetRefPicAddr;                                                    // 0x00000718 - 0x0000071B
        LwV32 SetHintMvAddr;                                                    // 0x0000071C - 0x0000071F
        LwV32 SetWinnerCostAddr;                                                // 0x00000720 - 0x00000723
        LwV32 SetWinnerFlowAddr;                                                // 0x00000724 - 0x00000727
        LwV32 SetHistoryBufAddr;                                                // 0x00000728 - 0x0000072B
        LwV32 SetTemporaryBufAddr;                                              // 0x0000072C - 0x0000072F
        LwV32 Reserved06[0x3];
    } SetMultiLevels[5];
    LwV32 SetStatusAddr;                                                        // 0x000007FC - 0x000007FF
    LwV32 Reserved07[0x245];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved08[0x3BA];
} LWC6FA_VIDEO_OFAControlPio;

#define LWC6FA_NOP                                                              (0x00000100)
#define LWC6FA_NOP_PARAMETER                                                    31:0
#define LWC6FA_PM_TRIGGER                                                       (0x00000140)
#define LWC6FA_PM_TRIGGER_V                                                     31:0
#define LWC6FA_SET_APPLICATION_ID                                               (0x00000200)
#define LWC6FA_SET_APPLICATION_ID_ID                                            31:0
#define LWC6FA_SET_APPLICATION_ID_ID_OFA                                        (0x00000001)
#define LWC6FA_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LWC6FA_SET_WATCHDOG_TIMER_TIMER                                         31:0
#define LWC6FA_SEMAPHORE_A                                                      (0x00000240)
#define LWC6FA_SEMAPHORE_A_UPPER                                                7:0
#define LWC6FA_SEMAPHORE_B                                                      (0x00000244)
#define LWC6FA_SEMAPHORE_B_LOWER                                                31:0
#define LWC6FA_SEMAPHORE_C                                                      (0x00000248)
#define LWC6FA_SEMAPHORE_C_PAYLOAD                                              31:0
#define LWC6FA_CTX_SAVE_AREA                                                    (0x0000024C)
#define LWC6FA_CTX_SAVE_AREA_OFFSET                                             27:0
#define LWC6FA_CTX_SAVE_AREA_CTX_VALID                                          31:28
#define LWC6FA_CTX_SWITCH                                                       (0x00000250)
#define LWC6FA_CTX_SWITCH_RESTORE                                               0:0
#define LWC6FA_CTX_SWITCH_RESTORE_FALSE                                         (0x00000000)
#define LWC6FA_CTX_SWITCH_RESTORE_TRUE                                          (0x00000001)
#define LWC6FA_CTX_SWITCH_RST_NOTIFY                                            1:1
#define LWC6FA_CTX_SWITCH_RST_NOTIFY_FALSE                                      (0x00000000)
#define LWC6FA_CTX_SWITCH_RST_NOTIFY_TRUE                                       (0x00000001)
#define LWC6FA_CTX_SWITCH_RESERVED                                              7:2
#define LWC6FA_CTX_SWITCH_ASID                                                  23:8
#define LWC6FA_EXELWTE                                                          (0x00000300)
#define LWC6FA_EXELWTE_NOTIFY                                                   0:0
#define LWC6FA_EXELWTE_NOTIFY_DISABLE                                           (0x00000000)
#define LWC6FA_EXELWTE_NOTIFY_ENABLE                                            (0x00000001)
#define LWC6FA_EXELWTE_NOTIFY_ON                                                1:1
#define LWC6FA_EXELWTE_NOTIFY_ON_END                                            (0x00000000)
#define LWC6FA_EXELWTE_NOTIFY_ON_BEGIN                                          (0x00000001)
#define LWC6FA_EXELWTE_PREDICATION                                              2:2
#define LWC6FA_EXELWTE_PREDICATION_DISABLE                                      (0x00000000)
#define LWC6FA_EXELWTE_PREDICATION_ENABLE                                       (0x00000001)
#define LWC6FA_EXELWTE_PREDICATION_OP                                           3:3
#define LWC6FA_EXELWTE_PREDICATION_OP_EQUAL_ZERO                                (0x00000000)
#define LWC6FA_EXELWTE_PREDICATION_OP_NOT_EQUAL_ZERO                            (0x00000001)
#define LWC6FA_EXELWTE_AWAKEN                                                   8:8
#define LWC6FA_EXELWTE_AWAKEN_DISABLE                                           (0x00000000)
#define LWC6FA_EXELWTE_AWAKEN_ENABLE                                            (0x00000001)
#define LWC6FA_SEMAPHORE_D                                                      (0x00000304)
#define LWC6FA_SEMAPHORE_D_STRUCTURE_SIZE                                       0:0
#define LWC6FA_SEMAPHORE_D_STRUCTURE_SIZE_ONE                                   (0x00000000)
#define LWC6FA_SEMAPHORE_D_STRUCTURE_SIZE_FOUR                                  (0x00000001)
#define LWC6FA_SEMAPHORE_D_AWAKEN_ENABLE                                        8:8
#define LWC6FA_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                  (0x00000000)
#define LWC6FA_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                   (0x00000001)
#define LWC6FA_SEMAPHORE_D_OPERATION                                            17:16
#define LWC6FA_SEMAPHORE_D_OPERATION_RELEASE                                    (0x00000000)
#define LWC6FA_SEMAPHORE_D_OPERATION_RESERVED0                                  (0x00000001)
#define LWC6FA_SEMAPHORE_D_OPERATION_RESERVED1                                  (0x00000002)
#define LWC6FA_SEMAPHORE_D_OPERATION_TRAP                                       (0x00000003)
#define LWC6FA_SEMAPHORE_D_FLUSH_DISABLE                                        21:21
#define LWC6FA_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                  (0x00000000)
#define LWC6FA_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                   (0x00000001)
#define LWC6FA_SET_PREDICATION_OFFSET_UPPER                                     (0x00000308)
#define LWC6FA_SET_PREDICATION_OFFSET_UPPER_OFFSET                              7:0
#define LWC6FA_SET_PREDICATION_OFFSET_LOWER                                     (0x0000030C)
#define LWC6FA_SET_PREDICATION_OFFSET_LOWER_OFFSET                              31:0
#define LWC6FA_SET_AUXILIARY_DATA_BUFFER                                        (0x00000310)
#define LWC6FA_SET_AUXILIARY_DATA_BUFFER_OFFSET                                 31:0
#define LWC6FA_SET_PICTURE_INDEX                                                (0x00000700)
#define LWC6FA_SET_PICTURE_INDEX_INDEX                                          31:0
#define LWC6FA_SET_CONTROL_PARAMS                                               (0x00000704)
#define LWC6FA_SET_CONTROL_PARAMS_GPTIMER_ON                                    0:0
#define LWC6FA_SET_CONTROL_PARAMS_DUMP_CYCLE_COUNT                              1:1
#define LWC6FA_SET_CONTROL_PARAMS_DEBUG_MODE                                    2:2
#define LWC6FA_SET_TOTAL_LEVEL_NUM                                              (0x00000708)
#define LWC6FA_SET_TOTAL_LEVEL_NUM_NUMBER                                       31:0
#define LWC6FA_SET_STATUS_ADDR                                                  (0x000007FC)
#define LWC6FA_SET_STATUS_ADDR_OFFSET                                           31:0
#define LWC6FA_PM_TRIGGER_END                                                   (0x00001114)
#define LWC6FA_PM_TRIGGER_END_V                                                 31:0

#define LWC6FA_SET_MULTI_LEVELS_SET_LEVEL_INDEX(a)                              (0x0000070C + (a)*0x00000030)
#define LWC6FA_SET_MULTI_LEVELS_SET_LEVEL_INDEX_INDEX                           31:0
#define LWC6FA_SET_MULTI_LEVELS_SET_DRV_SET_UP_ADDR(a)                          (0x00000710 + (a)*0x00000030)
#define LWC6FA_SET_MULTI_LEVELS_SET_DRV_SET_UP_ADDR_OFFSET                      31:0
#define LWC6FA_SET_MULTI_LEVELS_SET_LWRR_PIC_ADDR(a)                            (0x00000714 + (a)*0x00000030)
#define LWC6FA_SET_MULTI_LEVELS_SET_LWRR_PIC_ADDR_OFFSET                        31:0
#define LWC6FA_SET_MULTI_LEVELS_SET_REF_PIC_ADDR(a)                             (0x00000718 + (a)*0x00000030)
#define LWC6FA_SET_MULTI_LEVELS_SET_REF_PIC_ADDR_OFFSET                         31:0
#define LWC6FA_SET_MULTI_LEVELS_SET_HINT_MV_ADDR(a)                             (0x0000071C + (a)*0x00000030)
#define LWC6FA_SET_MULTI_LEVELS_SET_HINT_MV_ADDR_OFFSET                         31:0
#define LWC6FA_SET_MULTI_LEVELS_SET_WINNER_COST_ADDR(a)                         (0x00000720 + (a)*0x00000030)
#define LWC6FA_SET_MULTI_LEVELS_SET_WINNER_COST_ADDR_OFFSET                     31:0
#define LWC6FA_SET_MULTI_LEVELS_SET_WINNER_FLOW_ADDR(a)                         (0x00000724 + (a)*0x00000030)
#define LWC6FA_SET_MULTI_LEVELS_SET_WINNER_FLOW_ADDR_OFFSET                     31:0
#define LWC6FA_SET_MULTI_LEVELS_SET_HISTORY_BUF_ADDR(a)                         (0x00000728 + (a)*0x00000030)
#define LWC6FA_SET_MULTI_LEVELS_SET_HISTORY_BUF_ADDR_OFFSET                     31:0
#define LWC6FA_SET_MULTI_LEVELS_SET_TEMPORARY_BUF_ADDR(a)                       (0x0000072C + (a)*0x00000030)
#define LWC6FA_SET_MULTI_LEVELS_SET_TEMPORARY_BUF_ADDR_OFFSET                   31:0

#define LWC6FA_ERROR_NONE                                                       (0x00000000)
#define LWC6FA_OS_ERROR_EXELWTE_INSUFFICIENT_DATA                               (0x00000001)
#define LWC6FA_OS_ERROR_SEMAPHORE_INSUFFICIENT_DATA                             (0x00000002)
#define LWC6FA_OS_ERROR_ILWALID_METHOD                                          (0x00000003)
#define LWC6FA_OS_ERROR_ILWALID_DMA_PAGE                                        (0x00000004)
#define LWC6FA_OS_ERROR_UNHANDLED_INTERRUPT                                     (0x00000005)
#define LWC6FA_OS_ERROR_EXCEPTION                                               (0x00000006)
#define LWC6FA_OS_ERROR_ILWALID_CTXSW_REQUEST                                   (0x00000007)
#define LWC6FA_OS_ERROR_APPLICATION                                             (0x00000008)
#define LWC6FA_OS_INTERRUPT_EXELWTE_AWAKEN                                      (0x00000100)
#define LWC6FA_OS_INTERRUPT_BACKEND_SEMAPHORE_AWAKEN                            (0x00000200)
#define LWC6FA_OS_INTERRUPT_CTX_ERROR_FBIF                                      (0x00000300)
#define LWC6FA_OS_INTERRUPT_LIMIT_VIOLATION                                     (0x00000400)
#define LWC6FA_OS_INTERRUPT_LIMIT_AND_FBIF_CTX_ERROR                            (0x00000500)
#define LWC6FA_OS_INTERRUPT_HALT_ENGINE                                         (0x00000600)
#define LWC6FA_OS_INTERRUPT_TRAP_NONSTALL                                       (0x00000700)
#define LWC6FA_OS_INTERRUPT_CTX_SAVE_DONE                                       (0x00000800)
#define LWC6FA_OS_INTERRUPT_CTX_RESTORE_DONE                                    (0x00000900)
#define LWC6FA_ERROR_OFAAPPTIMER_EXPIRED                                        (0x30000001)
#define LWC6FA_ERROR_OFAILWALID_INPUT                                           (0x30000002)
#define LWC6FA_ERROR_OFAHWERR_INTERRUPT                                         (0x30000003)
#define LWC6FA_ERROR_OFABAD_MAGIC                                               (0x30000004)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _clc6fa_h

