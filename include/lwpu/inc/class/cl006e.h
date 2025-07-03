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

#ifndef _cl006e_h_
#define _cl006e_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

/* class LW10_CHANNEL_DMA */
#define  LW10_CHANNEL_DMA                                          (0x0000006E)
#define LW06E_NOTIFIERS_MAXCOUNT                                   1
/* LwNotification[] fields and values */
#define LW06E_NOTIFICATION_STATUS_ERROR_BAD_ARGUMENT               (0x2000)
#define LW06E_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
/* pio method data structure */
typedef volatile struct _cl006e_tag0 {
 LwV32 Reserved00[0x7c0];
} Lw06eTypedef, Lw10ChannelDma;
#define LW06E_TYPEDEF                                            Lw10ChannelDma
/* pio flow control data structure */
typedef volatile struct _cl006e_tag1 {
 LwV32 Ignored00[0x010];
 LwU32 Put;                     /* put offset, write only           0040-0043*/
 LwU32 Get;                     /* get offset, read only            0044-0047*/
 LwU32 Reference;               /* reference value, read only       0048-004b*/
 LwV32 Ignored01[0x1];
 LwU32 SetReference;            /* reference value, write only      0050-0053*/
 LwV32 Ignored02[0xf];
 LwU32 SwapExtension;           /* swap extension, write only       0090-0093*/
 LwV32 Ignored03[0x3db];
} Lw06eControl, Lw10ControlDma;
/* fields and values */
#define LW06E_NUMBER_OF_SUBCHANNELS                                (8)
#define LW06E_SET_OBJECT                                           (0x00000000)
#define LW06E_SET_REFERENCE                                        (0x00000050)
#define LW06E_QUADRO_VERIFY                                        (0x000000a0)

/* dma method descriptor format */
#define LW06E_DMA_METHOD_ADDRESS                                   12:2
#define LW06E_DMA_METHOD_SUBCHANNEL                                15:13
#define LW06E_DMA_METHOD_COUNT                                     28:18
#define LW06E_DMA_OPCODE                                           31:29
#define LW06E_DMA_OPCODE_METHOD                                    (0x00000000)
#define LW06E_DMA_OPCODE_NONINC_METHOD                             (0x00000002)
/* dma data format */
#define LW06E_DMA_DATA                                             31:0
/* dma jump format */
#define LW06E_DMA_OPCODE_JUMP                                      (0x00000001)
#define LW06E_DMA_JUMP_OFFSET                                      28:2

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl006e_h_ */
