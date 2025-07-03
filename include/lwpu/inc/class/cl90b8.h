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

#ifndef _cl90b8_h_
#define _cl90b8_h_

#ifdef __cplusplus
extern "C" {
#endif

#define GF106_DMA_DECOMPRESS                                                      (0x000090B8)

typedef volatile struct _cl90b8_tag0 {
    LwV32 Reserved00[0x40];
    LwV32 Nop;                                                                  // 0x00000100 - 0x00000103
    LwV32 Reserved01[0xF];
    LwV32 PmTrigger;                                                            // 0x00000140 - 0x00000143
    LwV32 Reserved02[0x2F];
    LwV32 SetApplicationID;                                                     // 0x00000200 - 0x00000203
    LwV32 SetWatchdogTimer;                                                     // 0x00000204 - 0x00000207
    LwV32 Reserved03[0xE];
    LwV32 SetSemaphoreA;                                                        // 0x00000240 - 0x00000243
    LwV32 SetSemaphoreB;                                                        // 0x00000244 - 0x00000247
    LwV32 SetSemaphorePayload;                                                  // 0x00000248 - 0x0000024B
    LwV32 Reserved04[0x2D];
    LwV32 LaunchDma;                                                            // 0x00000300 - 0x00000303
    LwV32 Reserved05[0x3F];
    LwV32 OffsetInUpper;                                                        // 0x00000400 - 0x00000403
    LwV32 OffsetInLower;                                                        // 0x00000404 - 0x00000407
    LwV32 OffsetOutUpper;                                                       // 0x00000408 - 0x0000040B
    LwV32 OffsetOutLower;                                                       // 0x0000040C - 0x0000040F
    LwV32 Reserved06[0xCF];
    LwV32 SetSrcSize;                                                           // 0x0000074C - 0x0000074F
    LwV32 SetDstSizeLimit;                                                      // 0x00000750 - 0x00000753
    LwV32 Reserved07[0x270];
    LwV32 PmTriggerEnd;                                                         // 0x00001114 - 0x00001117
    LwV32 Reserved08[0x3BA];
} GF106dma_decompressControlPio;

#define LW90B8_NOP                                                              (0x00000100)
#define LW90B8_NOP_PARAMETER                                                    31:0
#define LW90B8_PM_TRIGGER                                                       (0x00000140)
#define LW90B8_PM_TRIGGER_V                                                     31:0
#define LW90B8_SET_APPLICATION_ID                                               (0x00000200)
#define LW90B8_SET_APPLICATION_ID_ID                                            31:0
#define LW90B8_SET_APPLICATION_ID_ID_NORMAL                                     (0x00000001)
#define LW90B8_SET_WATCHDOG_TIMER                                               (0x00000204)
#define LW90B8_SET_WATCHDOG_TIMER_TIMER                                         31:0
#define LW90B8_SET_SEMAPHORE_A                                                  (0x00000240)
#define LW90B8_SET_SEMAPHORE_A_UPPER                                            7:0
#define LW90B8_SET_SEMAPHORE_B                                                  (0x00000244)
#define LW90B8_SET_SEMAPHORE_B_LOWER                                            31:0
#define LW90B8_SET_SEMAPHORE_PAYLOAD                                            (0x00000248)
#define LW90B8_SET_SEMAPHORE_PAYLOAD_PAYLOAD                                    31:0
#define LW90B8_LAUNCH_DMA                                                       (0x00000300)
#define LW90B8_LAUNCH_DMA_DATA_TRANSFER_TYPE                                    1:0
#define LW90B8_LAUNCH_DMA_DATA_TRANSFER_TYPE_NONE                               (0x00000000)
#define LW90B8_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED                      (0x00000002)
#define LW90B8_LAUNCH_DMA_FLUSH_ENABLE                                          2:2
#define LW90B8_LAUNCH_DMA_FLUSH_ENABLE_FALSE                                    (0x00000000)
#define LW90B8_LAUNCH_DMA_FLUSH_ENABLE_TRUE                                     (0x00000001)
#define LW90B8_LAUNCH_DMA_SEMAPHORE_TYPE                                        4:3
#define LW90B8_LAUNCH_DMA_SEMAPHORE_TYPE_NONE                                   (0x00000000)
#define LW90B8_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_ONE_WORD_SEMAPHORE             (0x00000001)
#define LW90B8_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_FOUR_WORD_SEMAPHORE            (0x00000002)
#define LW90B8_LAUNCH_DMA_INTERRUPT_TYPE                                        6:5
#define LW90B8_LAUNCH_DMA_INTERRUPT_TYPE_NONE                                   (0x00000000)
#define LW90B8_LAUNCH_DMA_INTERRUPT_TYPE_BLOCKING                               (0x00000001)
#define LW90B8_LAUNCH_DMA_INTERRUPT_TYPE_NON_BLOCKING                           (0x00000002)
#define LW90B8_LAUNCH_DMA_DECOMPRESS_ENABLE                                     7:7
#define LW90B8_LAUNCH_DMA_DECOMPRESS_ENABLE_FALSE                               (0x00000000)
#define LW90B8_LAUNCH_DMA_DECOMPRESS_ENABLE_TRUE                                (0x00000001)
#define LW90B8_OFFSET_IN_UPPER                                                  (0x00000400)
#define LW90B8_OFFSET_IN_UPPER_UPPER                                            7:0
#define LW90B8_OFFSET_IN_LOWER                                                  (0x00000404)
#define LW90B8_OFFSET_IN_LOWER_VALUE                                            31:0
#define LW90B8_OFFSET_OUT_UPPER                                                 (0x00000408)
#define LW90B8_OFFSET_OUT_UPPER_UPPER                                           7:0
#define LW90B8_OFFSET_OUT_LOWER                                                 (0x0000040C)
#define LW90B8_OFFSET_OUT_LOWER_VALUE                                           31:0
#define LW90B8_SET_SRC_SIZE                                                     (0x0000074C)
#define LW90B8_SET_SRC_SIZE_V                                                   31:0
#define LW90B8_SET_DST_SIZE_LIMIT                                               (0x00000750)
#define LW90B8_SET_DST_SIZE_LIMIT_V                                             31:0
#define LW90B8_PM_TRIGGER_END                                                   (0x00001114)
#define LW90B8_PM_TRIGGER_END_V                                                 31:0

#ifdef __cplusplus
};     /* extern "C" */
#endif
#endif // _cl90b8_h

