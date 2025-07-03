/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2001-2003 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl446e_h_
#define _cl446e_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

/* class LW44_CHANNEL_DMA */
#define  LW44_CHANNEL_DMA                                          (0x0000446E)
/* LwNotification[] fields and values */
#define LW446E_NOTIFICATION_STATUS_ERROR_BAD_ARGUMENT              (0x2000)
#define LW446E_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT          (0x4000)
/* pio method data structure */
typedef volatile struct _cl446e_tag0 {
 LwV32 Reserved00[0x7c0];
} Lw446eTypedef, Lw44ChannelDma;
#define LW446E_TYPEDEF                                           Lw44ChannelDma
/* pio flow control data structure */
typedef volatile struct _cl446e_tag1 {
 LwV32 Ignored00[0x010];
 LwU32 Put;                     /* put offset, write only           0040-0043*/
 LwU32 Get;                     /* get offset, read only            0044-0047*/
 LwU32 Reference;               /* reference value, read only       0048-004b*/
 LwU32 Ignored01[0x1];
 LwU32 SetReference;            /* set reference value              0050-0053*/
 LwU32 TopLevelGet;             /* top level get offset, read only  0054-0057*/
 LwU32 Ignored02[0x2];
 LwU32 SetContextDmaSemaphore;  /* set sema ctxdma, write only      0060-0063*/
 LwU32 SetSemaphoreOffset;      /* set sema offset, write only      0064-0067*/
 LwU32 SetSemaphoreAcquire;     /* set sema acquire, write only     0068-006b*/
 LwU32 SetSemaphoreRelease;     /* set sema release, write only     006c-006f*/
 LwV32 Ignored03[0x4];
 LwU32 Yield;                   /* engine yield, write only         0080-0083*/
 LwV32 Ignored04[0x3df];
} Lw446eControl, Lw44ControlDma;
/* fields and values */
#define LW446E_NUMBER_OF_SUBCHANNELS                               (8)
#define LW446E_SET_OBJECT                                          (0x00000000)
#define LW446E_SET_REFERENCE                                       (0x00000050)
#define LW446E_SET_CONTEXT_DMA_SEMAPHORE                           (0x00000060)
#define LW446E_SEMAPHORE_OFFSET                                    (0x00000064)
#define LW446E_SEMAPHORE_ACQUIRE                                   (0x00000068)
#define LW446E_SEMAPHORE_RELEASE                                   (0x0000006c)
#define LW446E_YIELD                                               (0x00000080)
#define LW446E_QUADRO_VERIFY                                       (0x000000a0)

/* dma method descriptor format */
#define LW446E_DMA_METHOD_ADDRESS                                  12:2
#define LW446E_DMA_METHOD_SUBCHANNEL                               15:13
#define LW446E_DMA_METHOD_COUNT                                    28:18

/* dma opcode format */
#define LW446E_DMA_OPCODE                                          31:29
#define LW446E_DMA_OPCODE_METHOD                                   (0x00000000)
#define LW446E_DMA_OPCODE_NONINC_METHOD                            (0x00000002)
/* dma jump format */
#define LW446E_DMA_OPCODE_JUMP                                     (0x00000001)
#define LW446E_DMA_JUMP_OFFSET                                     28:2

/* dma opcode2 format */
#define LW446E_DMA_OPCODE2                                         1:0
#define LW446E_DMA_OPCODE2_NONE                                    (0x00000000)
/* dma jump_long format */
#define LW446E_DMA_OPCODE2_JUMP_LONG                               (0x00000001)
#define LW446E_DMA_JUMP_LONG_OFFSET                                31:2
/* dma call format */
#define LW446E_DMA_OPCODE2_CALL                                    (0x00000002)
#define LW446E_DMA_CALL_OFFSET                                     31:2

/* dma opcode3 format */
#define LW446E_DMA_OPCODE3                                         17:16
#define LW446E_DMA_OPCODE3_NONE                                    (0x00000000)
/* dma return format */
#define LW446E_DMA_RETURN                                          (0x00020000)
#define LW446E_DMA_OPCODE3_RETURN                                  (0x00000002)

/* dma data format */
#define LW446E_DMA_DATA                                            31:0

/* dma nop format */
#define LW446E_DMA_NOP                                             (0x00000000)

/* dma set subdevice mask format */
#define LW446E_DMA_SET_SUBDEVICE_MASK                              (0x00010000)
#define LW446E_DMA_SET_SUBDEVICE_MASK_VALUE                        15:4
#define LW446E_DMA_OPCODE3_SET_SUBDEVICE_MASK                      (0x00000001)

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl446e_h_ */
