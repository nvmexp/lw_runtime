/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2001-2001 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl206e_h_
#define _cl206e_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

/* class LW20_CHANNEL_DMA */
#define  LW20_CHANNEL_DMA                                          (0x0000206E)
/* LwNotification[] fields and values */
#define LW206E_NOTIFICATION_STATUS_ERROR_BAD_ARGUMENT              (0x2000)
#define LW206E_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT          (0x4000)
/* pio method data structure */
typedef volatile struct _cl206e_tag0 {
 LwV32 Reserved00[0x7c0];
} Lw206eTypedef, Lw20ChannelDma;
#define LW206E_TYPEDEF                                           Lw20ChannelDma
/* pio flow control data structure */
typedef volatile struct _cl206e_tag1 {
 LwV32 Ignored00[0x010];
 LwU32 Put;                     /* put offset, write only           0040-0043*/
 LwU32 Get;                     /* get offset, read only            0044-0047*/
 LwU32 Reference;               /* reference value, read only       0048-004b*/
 LwU32 Ignored01[0x1];
 LwU32 SetReference;            /* set reference value              0050-0053*/
 LwU32 Ignored02[0x3];
 LwU32 SetContextDmaSemaphore;  /* set sema ctxdma, write only      0060-0063*/
 LwU32 SetSemaphoreOffset;      /* set sema offset, write only      0064-0067*/
 LwU32 SetSemaphoreAcquire;     /* set sema acquire, write only     0068-006b*/
 LwU32 SetSemaphoreRelease;     /* set sema release, write only     006c-006f*/
 LwV32 Ignored03[0x3e4];
} Lw206eControl, Lw20ControlDma;
/* fields and values */
#define LW206E_NUMBER_OF_SUBCHANNELS                               (8)
#define LW206E_SET_OBJECT                                          (0x00000000)
#define LW206E_SET_REFERENCE                                       (0x00000050)
#define LW206E_SET_CONTEXT_DMA_SEMAPHORE                           (0x00000060)
#define LW206E_SEMAPHORE_OFFSET                                    (0x00000064)
#define LW206E_SEMAPHORE_ACQUIRE                                   (0x00000068)
#define LW206E_SEMAPHORE_RELEASE                                   (0x0000006c)
#define LW206E_SUBROUTINE_STATE_RESET                              (0x0000009c)
#define LW206E_QUADRO_VERIFY                                       (0x000000a0)
#define LW206E_SPLIT_POINT                                         (0x000000b0)

/* dma method descriptor format */
#define LW206E_DMA_METHOD_ADDRESS                                  12:2
#define LW206E_DMA_METHOD_SUBCHANNEL                               15:13
#define LW206E_DMA_METHOD_COUNT                                    28:18

/* dma opcode format */
#define LW206E_DMA_OPCODE                                          31:29
#define LW206E_DMA_OPCODE_METHOD                                   (0x00000000)
#define LW206E_DMA_OPCODE_NONINC_METHOD                            (0x00000002)
/* dma jump format */
#define LW206E_DMA_OPCODE_JUMP                                     (0x00000001)
#define LW206E_DMA_JUMP_OFFSET                                     28:2

/* dma opcode2 format */
#define LW206E_DMA_OPCODE2                                         1:0
#define LW206E_DMA_OPCODE2_NONE                                    (0x00000000)
/* dma jump_long format */
#define LW206E_DMA_OPCODE2_JUMP_LONG                               (0x00000001)
#define LW206E_DMA_JUMP_LONG_OFFSET                                31:2
/* dma call format */
#define LW206E_DMA_OPCODE2_CALL                                    (0x00000002)
#define LW206E_DMA_CALL_OFFSET                                     31:2

/* dma opcode3 format */
#define LW206E_DMA_OPCODE3                                         17:16
#define LW206E_DMA_OPCODE3_NONE                                    (0x00000000)
/* dma return format */
#define LW206E_DMA_RETURN                                          (0x00020000)
#define LW206E_DMA_OPCODE3_RETURN                                  (0x00000002)

/* dma data format */
#define LW206E_DMA_DATA                                            31:0

/* dma nop format */
#define LW206E_DMA_NOP                                             (0x00000000)

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl206e_h_ */
