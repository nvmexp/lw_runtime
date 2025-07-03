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


#include "lwtypes.h"

#ifndef _clc4d1_h_
#define _clc4d1_h_

#ifdef __cplusplus
extern "C" {
#endif

#define LWC4D1_VIDEO_LWJPG                                                               (0x0000C4D1)

#if (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)
typedef volatile struct _clc4d1_tag0 {
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
    LwV32 SetTotalCoreNum;                                                      // 0x00000704 - 0x00000707
    LwV32 SetInDrvPicSetup;                                                     // 0x00000708 - 0x0000070B
    LwV32 SetOutStatus;                                                         // 0x0000070C - 0x0000070F
    struct{
        LwV32 SetCoreIndex;                                                     // 0x00000710 - 0x00000713
        LwV32 SetBitstream;                                                     // 0x00000714 - 0x00000717
        LwV32 SetLwrPic;                                                        // 0x00000718 - 0x0000071B
        LwV32 SetLwrPicChromaU;                                                 // 0x0000071C - 0x0000071F
        LwV32 SetLwrPicChromaV;                                                 // 0x00000720 - 0x00000723
        // pad to make per-core method count 8 so that the arithmetic in method grouping (METHOD(i)) can work.
        LwV32 Reserved06[3];                                                    // 0x00000724 - 0x0000072F
    } SetPerCore[5];
    // 0x00000710 - 0x0000072F
    // 0x00000730 - 0x0000074F
    // 0x00000750 - 0x0000076F
    // 0x00000770 - 0x0000078F
    // 0x00000790 - 0x000007AF
    LwV32 GPTimerValue;                                                         // 0x000007B0 - 0x000007B3
    LwV32 Reserved07[0x258];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved08[0x3BA];
} LWC4D1_VIDEO_LWJPGControlPio;

#define LWC4D1_NOP                                                              (0x00000100)
#define LWC4D1_NOP_PARAMETER                                                    31:0
#define LWC4D1_PM_TRIGGER                                                       (0x00000140)
#define LWC4D1_PM_TRIGGER_V                                                     31:0
#define LWC4D1_SET_APPLICATION_ID                                               (0x00000200)
#define LWC4D1_SET_APPLICATION_ID_ID                                            31:0
#define LWC4D1_SET_APPLICATION_ID_ID_LWJPG_DECODER                              (0x00000001)
#define LWC4D1_SET_APPLICATION_ID_ID_LWJPG_ENCODER                              (0x00000002)
#define LWC4D1_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LWC4D1_SET_WATCHDOG_TIMER_TIMER                                         31:0
#define LWC4D1_SEMAPHORE_A                                                      (0x00000240)
#define LWC4D1_SEMAPHORE_A_UPPER                                                7:0
#define LWC4D1_SEMAPHORE_B                                                      (0x00000244)
#define LWC4D1_SEMAPHORE_B_LOWER                                                31:0
#define LWC4D1_SEMAPHORE_C                                                      (0x00000248)
#define LWC4D1_SEMAPHORE_C_PAYLOAD                                              31:0
#define LWC4D1_CTX_SAVE_AREA                                                    (0x0000024C)
#define LWC4D1_CTX_SAVE_AREA_OFFSET                                             27:0
#define LWC4D1_CTX_SAVE_AREA_CTX_VALID                                          31:28
#define LWC4D1_CTX_SWITCH                                                       (0x00000250)
#define LWC4D1_CTX_SWITCH_RESTORE                                               0:0
#define LWC4D1_CTX_SWITCH_RESTORE_FALSE                                         (0x00000000)
#define LWC4D1_CTX_SWITCH_RESTORE_TRUE                                          (0x00000001)
#define LWC4D1_CTX_SWITCH_RST_NOTIFY                                            1:1
#define LWC4D1_CTX_SWITCH_RST_NOTIFY_FALSE                                      (0x00000000)
#define LWC4D1_CTX_SWITCH_RST_NOTIFY_TRUE                                       (0x00000001)
#define LWC4D1_CTX_SWITCH_RESERVED                                              7:2
#define LWC4D1_CTX_SWITCH_ASID                                                  23:8
#define LWC4D1_EXELWTE                                                          (0x00000300)
#define LWC4D1_EXELWTE_NOTIFY                                                   0:0
#define LWC4D1_EXELWTE_NOTIFY_DISABLE                                           (0x00000000)
#define LWC4D1_EXELWTE_NOTIFY_ENABLE                                            (0x00000001)
#define LWC4D1_EXELWTE_NOTIFY_ON                                                1:1
#define LWC4D1_EXELWTE_NOTIFY_ON_END                                            (0x00000000)
#define LWC4D1_EXELWTE_NOTIFY_ON_BEGIN                                          (0x00000001)
#define LWC4D1_EXELWTE_AWAKEN                                                   8:8
#define LWC4D1_EXELWTE_AWAKEN_DISABLE                                           (0x00000000)
#define LWC4D1_EXELWTE_AWAKEN_ENABLE                                            (0x00000001)
#define LWC4D1_SEMAPHORE_D                                                      (0x00000304)
#define LWC4D1_SEMAPHORE_D_STRUCTURE_SIZE                                       0:0
#define LWC4D1_SEMAPHORE_D_STRUCTURE_SIZE_ONE                                   (0x00000000)
#define LWC4D1_SEMAPHORE_D_STRUCTURE_SIZE_FOUR                                  (0x00000001)
#define LWC4D1_SEMAPHORE_D_AWAKEN_ENABLE                                        8:8
#define LWC4D1_SEMAPHORE_D_AWAKEN_ENABLE_FALSE                                  (0x00000000)
#define LWC4D1_SEMAPHORE_D_AWAKEN_ENABLE_TRUE                                   (0x00000001)
#define LWC4D1_SEMAPHORE_D_OPERATION                                            17:16
#define LWC4D1_SEMAPHORE_D_OPERATION_RELEASE                                    (0x00000000)
#define LWC4D1_SEMAPHORE_D_OPERATION_RESERVED0                                  (0x00000001)
#define LWC4D1_SEMAPHORE_D_OPERATION_RESERVED1                                  (0x00000002)
#define LWC4D1_SEMAPHORE_D_OPERATION_TRAP                                       (0x00000003)
#define LWC4D1_SEMAPHORE_D_FLUSH_DISABLE                                        21:21
#define LWC4D1_SEMAPHORE_D_FLUSH_DISABLE_FALSE                                  (0x00000000)
#define LWC4D1_SEMAPHORE_D_FLUSH_DISABLE_TRUE                                   (0x00000001)
#define LWC4D1_SET_CONTROL_PARAMS                                               (0x00000700)
#define LWC4D1_SET_CONTROL_PARAMS_GPTIMER_ON                                    0:0
#define LWC4D1_SET_CONTROL_PARAMS_DUMP_CYCLE_COUNT                              1:1
#define LWC4D1_SET_CONTROL_PARAMS_DEBUG_MODE                                    2:2
#define LWC4D1_SET_TOTAL_CORE_NUM                                               (0x00000704)
#define LWC4D1_SET_TOTAL_CORE_NUM_INDEX                                         31:0
#define LWC4D1_SET_IN_DRV_PIC_SETUP                                             (0x00000708)
#define LWC4D1_SET_IN_DRV_PIC_SETUP_OFFSET                                      31:0
#define LWC4D1_SET_PER_CORE_SET_OUT_STATUS                                      (0x0000070C)
#define LWC4D1_SET_PER_CORE_SET_OUT_STATUS_OFFSET                               31:0
#define LWC4D1_SET_PER_CORE_SET_CORE_INDEX(i)                                   (0x00000710+(i)*0x20)
#define LWC4D1_SET_PER_CORE_SET_CORE_INDEX_INDEX                                31:0
#define LWC4D1_SET_PER_CORE_SET_BITSTREAM(i)                                    (0x00000714+(i)*0x20)
#define LWC4D1_SET_PER_CORE_SET_BITSTREAM_OFFSET                                31:0
#define LWC4D1_SET_PER_CORE_SET_LWR_PIC(i)                                      (0x00000718+(i)*0x20)
#define LWC4D1_SET_PER_CORE_SET_LWR_PIC_OFFSET                                  31:0
#define LWC4D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_U(i)                             (0x0000071C+(i)*0x20)
#define LWC4D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_U_OFFSET                         31:0
#define LWC4D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_V(i)                             (0x00000720+(i)*0x20)
#define LWC4D1_SET_PER_CORE_SET_LWR_PIC_CHROMA_V_OFFSET                         31:0
#define LWC4D1_SET_GP_TIMER                                                     (0x000007B0)
#define LWC4D1_SET_GP_TIMER_VALUE                                               31:0
#define LWC4D1_PM_TRIGGER_END                                                   (0x00001114)
#define LWC4D1_PM_TRIGGER_END_V                                                 31:0

#define LWC4D1_ERROR_NONE                                                       (0x00000000)
#define LWC4D1_OS_ERROR_EXELWTE_INSUFFICIENT_DATA                               (0x00000001)
#define LWC4D1_OS_ERROR_SEMAPHORE_INSUFFICIENT_DATA                             (0x00000002)
#define LWC4D1_OS_ERROR_ILWALID_METHOD                                          (0x00000003)
#define LWC4D1_OS_ERROR_ILWALID_DMA_PAGE                                        (0x00000004)
#define LWC4D1_OS_ERROR_UNHANDLED_INTERRUPT                                     (0x00000005)
#define LWC4D1_OS_ERROR_EXCEPTION                                               (0x00000006)
#define LWC4D1_OS_ERROR_ILWALID_CTXSW_REQUEST                                   (0x00000007)
#define LWC4D1_OS_ERROR_APPLICATION                                             (0x00000008)
#define LWC4D1_OS_INTERRUPT_EXELWTE_AWAKEN                                      (0x00000100)
#define LWC4D1_OS_INTERRUPT_BACKEND_SEMAPHORE_AWAKEN                            (0x00000200)
#define LWC4D1_OS_INTERRUPT_CTX_ERROR_FBIF                                      (0x00000300)
#define LWC4D1_OS_INTERRUPT_LIMIT_VIOLATION                                     (0x00000400)
#define LWC4D1_OS_INTERRUPT_LIMIT_AND_FBIF_CTX_ERROR                            (0x00000500)
#define LWC4D1_OS_INTERRUPT_HALT_ENGINE                                         (0x00000600)
#define LWC4D1_OS_INTERRUPT_TRAP_NONSTALL                                       (0x00000700)
#define LWC4D1_OS_INTERRUPT_CTX_SAVE_DONE                                       (0x00000800)
#define LWC4D1_OS_INTERRUPT_CTX_RESTORE_DONE                                    (0x00000900)
#define LWC4D1_ERROR_JPGAPPTIMER_EXPIRED                                        (0x30000001)
#define LWC4D1_ERROR_JPGILWALID_INPUT                                           (0x30000002)
#define LWC4D1_ERROR_JPGHWERR_INTERRUPT                                         (0x30000003)
#define LWC4D1_ERROR_JPGBAD_MAGIC                                               (0x30000004)
#endif // (!defined(LWRM_UNPUBLISHED_OPAQUE) || LWRM_UNPUBLISHED_OPAQUE == 1)

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _clc4d1_h

