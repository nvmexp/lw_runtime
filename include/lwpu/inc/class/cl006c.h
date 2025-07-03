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

#ifndef _cl006c_h_
#define _cl006c_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

/* class LW04_CHANNEL_DMA */
#define  LW04_CHANNEL_DMA                                          (0x0000006C)
#define LW06C_NOTIFIERS_MAXCOUNT                                   1
/* LwNotification[] fields and values */
#define LW06C_NOTIFICATION_STATUS_ERROR_BAD_ARGUMENT               (0x2000)
#define LW06C_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
/* pio method data structure */
typedef volatile struct _cl006c_tag0 {
 LwV32 Reserved00[0x7c0];
} Lw06cTypedef, Lw04ChannelDma;
#define LW06C_TYPEDEF                                            Lw04ChannelDma
/* pio flow control data structure */
typedef volatile struct _cl006c_tag1 {
 LwV32 Ignored00[0x010];
 LwU32 Put;                     /* put offset, write only           0040-0043*/
 LwU32 Get;                     /* get offset, read only            0044-0047*/
 LwV32 Ignored01[0x002];
 LwU32 StallNotifier;           /* Set stall notifier               0050-0053*/
 LwU32 StallChannel;            /* Stall the channel                0054-0057*/
 LwV32 Ignored02[0x3EA];
} Lw04ControlDma;
/* obsolete stuff */
#define LW4_CHANNEL_DMA                                            (0x0000006C)
#define Lw4ChannelDma                                            Lw04ChannelDma
#define lw4ChannelDma                                            Lw04ChannelDma
#define Lw4ControlDma                                            Lw04ControlDma

/* dma method descriptor format */
#define LW06C_METHOD_ADDRESS                                       12:2
#define LW06C_METHOD_SUBCHANNEL                                    15:13
#define LW06C_METHOD_COUNT                                         28:18
#define LW06C_OPCODE                                               31:29
#define LW06C_OPCODE_METHOD                                        (0x00000000)
#define LW06C_OPCODE_NONINC_METHOD                                 (0x00000002)
/* dma data format */
#define LW06C_DATA                                                 31:0
/* dma jump format */
#define LW06C_OPCODE_JUMP                                          (0x00000001)
#define LW06C_JUMP_OFFSET                                          28:2

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl006c_h_ */
