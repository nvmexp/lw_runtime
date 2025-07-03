/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2001-2005 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl826f_h_
#define _cl826f_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

/* class G82_CHANNEL_GPFIFO */
#define  G82_CHANNEL_GPFIFO                                        (0x0000826F)

/* LwNotification[] indexes */
/* channel id returned in index 0 at alloc time */
#define LW826F_NOTIFIERS_GET_CHANNEL_ID_INDEX                           (0)
#define LW826F_NOTIFIERS_GET_CHANNEL_ID_INFO32_ID                      31:0
/* RC error oclwrring in channel updates index 0 */
#define LW826F_NOTIFIERS_ROBUST_CHANNEL_ERROR_INDEX                     (0)
#define LW826F_NOTIFIERS_ROBUST_CHANNEL_ERROR_INFO32_ERRTYPE           31:0
#define LW826F_NOTIFIERS_ROBUST_CHANNEL_ERROR_STATUS                   15:0
#define LW826F_NOTIFIERS_ROBUST_CHANNEL_ERROR_STATUS_ERROR           0xFFFF
#define LW826F_NOTIFIERS_UNUSED                                         (1)
#define LW826F_NOTIFIERS_GR_DEBUG_INTR                                  (2)
#define LW826F_NOTIFIERS_MAXCOUNT                                       (3)

/* LwNotification[] fields and values */
#define LW826F_NOTIFICATION_STATUS_ERROR_BAD_ARGUMENT              (0x2000)
#define LW826F_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT          (0x4000)
/* pio method data structure */
typedef volatile struct _cl826f_tag0 {
 LwV32 Reserved00[0x7c0];
} Lw826fTypedef, G82ChannelGPFifo;
#define LW826F_TYPEDEF                                         G82ChannelGPFifo
/* dma flow control data structure */
typedef volatile struct _cl826f_tag1 {
 LwU32 Ignored00[0x010];        /*                                  0000-0043*/
 LwU32 Put;                     /* put offset, read/write           0040-0043*/
 LwU32 Get;                     /* get offset, read only            0044-0047*/
 LwU32 Reference;               /* reference value, read only       0048-004b*/
 LwU32 PutHi;                   /* high order put offset bits       004c-004f*/
 LwU32 SetReference;            /* set reference value              0050-0053*/
 LwU32 Ignored02[0x001];        /*                                  0054-0057*/
 LwU32 TopLevelGet;             /* top level get offset, read only  0058-005b*/
 LwU32 TopLevelGetHi;           /* high order top level get bits    005c-005f*/
 LwU32 GetHi;                   /* high order get offset bits       0060-0063*/
 LwU32 Ignored03[0x007];        /*                                  0064-007f*/
 LwU32 Yield;                   /* engine yield, write only         0080-0083*/
 LwU32 Ignored04[0x001];        /*                                  0084-0087*/
 LwU32 GPGet;                   /* GP FIFO get offset, read only    0088-008b*/
 LwU32 GPPut;                   /* GP FIFO put offset               008c-008f*/
 LwU32 Ignored05[0x3dc];
} Lw826fControl, G82ControlGPFifo;
/* fields and values */
#define LW826F_NUMBER_OF_SUBCHANNELS                               (8)
#define LW826F_SET_OBJECT                                          (0x00000000)
#define LW826F_SEMAPHOREA                                          (0x00000010)
#define LW826F_SEMAPHOREA_OFFSET_UPPER                                     7:0
#define LW826F_SEMAPHOREB                                          (0x00000014)
#define LW826F_SEMAPHOREB_OFFSET_LOWER                                   31:00
#define LW826F_SEMAPHOREC                                          (0x00000018)
#define LW826F_SEMAPHOREC_PAYLOAD                                         31:0
#define LW826F_SEMAPHORED                                          (0x0000001C)
#define LW826F_SEMAPHORED_OPERATION                                        2:0
#define LW826F_SEMAPHORED_OPERATION_ACQUIRE                         0x00000001
#define LW826F_SEMAPHORED_OPERATION_RELEASE                         0x00000002
#define LW826F_SEMAPHORED_OPERATION_ACQ_GEQ                         0x00000004
#define LW826F_NON_STALLED_INTERRUPT                               (0x00000020)
#define LW826F_FB_FLUSH                                            (0x00000024)
#define LW826F_SET_REFERENCE                                       (0x00000050)
#define LW826F_SET_CONTEXT_DMA_SEMAPHORE                           (0x00000060)
#define LW826F_SEMAPHORE_OFFSET                                    (0x00000064)
#define LW826F_SEMAPHORE_ACQUIRE                                   (0x00000068)
#define LW826F_SEMAPHORE_RELEASE                                   (0x0000006c)
#define LW826F_YIELD                                               (0x00000080)
#define LW826F_SWITCH_NO_WAIT                                      (0x00000084)
#define LW826F_QUADRO_VERIFY                                       (0x000000a0)
#define LW826F_SPLIT_POINT                                         (0x000000b0)


/* GPFIFO entry format */
#define LW826F_GP_ENTRY__SIZE                                   8
#define LW826F_GP_ENTRY0_DISABLE                              0:0
#define LW826F_GP_ENTRY0_DISABLE_NOT                   0x00000000
#define LW826F_GP_ENTRY0_DISABLE_SKIP                  0x00000001
#define LW826F_GP_ENTRY0_NO_CONTEXT_SWITCH                    1:1
#define LW826F_GP_ENTRY0_NO_CONTEXT_SWITCH_FALSE       0x00000000
#define LW826F_GP_ENTRY0_NO_CONTEXT_SWITCH_TRUE        0x00000001
#define LW826F_GP_ENTRY0_GET                                 31:2
#define LW826F_GP_ENTRY1_GET_HI                               7:0
#define LW826F_GP_ENTRY1_PRIV                                 8:8
#define LW826F_GP_ENTRY1_PRIV_USER                     0x00000000
#define LW826F_GP_ENTRY1_PRIV_KERNEL                   0x00000001
#define LW826F_GP_ENTRY1_LEVEL                                9:9
#define LW826F_GP_ENTRY1_LEVEL_MAIN                    0x00000000
#define LW826F_GP_ENTRY1_LEVEL_SUBROUTINE              0x00000001
#define LW826F_GP_ENTRY1_LENGTH                             31:10

/* dma method descriptor formats */
#define LW826F_DMA_PRIMARY_OPCODE                                  1:0
#define LW826F_DMA_PRIMARY_OPCODE_USES_SECONDARY                   (0x00000000)
#define LW826F_DMA_PRIMARY_OPCODE_RESERVED                         (0x00000003)
#define LW826F_DMA_METHOD_ADDRESS                                  12:2
#define LW826F_DMA_METHOD_SUBCHANNEL                               15:13
#define LW826F_DMA_TERT_OP                                         17:16
#define LW826F_DMA_TERT_OP_GRP0_INC_METHOD                         (0x00000000)
#define LW826F_DMA_TERT_OP_GRP0_SET_SUB_DEV_MASK                   (0x00000001)
#define LW826F_DMA_TERT_OP_GRP0_DOUBLE_HEADER                      (0x00000003)
#define LW826F_DMA_TERT_OP_GRP2_NON_INC_METHOD                     (0x00000000)
#define LW826F_DMA_TERT_OP_GRP2_RESERVED01                         (0x00000001)
#define LW826F_DMA_TERT_OP_GRP2_RESERVED10                         (0x00000002)
#define LW826F_DMA_TERT_OP_GRP2_RESERVED11                         (0x00000003)
#define LW826F_DMA_METHOD_COUNT                                    28:18
#define LW826F_DMA_SEC_OP                                          31:29
#define LW826F_DMA_SEC_OP_GRP0_USE_TERT                            (0x00000000)
#define LW826F_DMA_SEC_OP_GRP2_USE_TERT                            (0x00000002)
#define LW826F_DMA_SEC_OP_GRP3_RESERVED                            (0x00000003)
#define LW826F_DMA_SEC_OP_GRP4_RESERVED                            (0x00000004)
#define LW826F_DMA_SEC_OP_GRP5_RESERVED                            (0x00000005)
#define LW826F_DMA_SEC_OP_GRP6_RESERVED                            (0x00000006)
#define LW826F_DMA_SEC_OP_GRP7_RESERVED                            (0x00000007)
#define LW826F_DMA_LONG_COUNT                                      31:0 
/* dma legacy method descriptor format */
#define LW826F_DMA_OPCODE2                                         1:0
#define LW826F_DMA_OPCODE2_NONE                                    (0x00000000)
#define LW826F_DMA_OPCODE                                          31:29
#define LW826F_DMA_OPCODE_METHOD                                   (0x00000000)
#define LW826F_DMA_OPCODE_NONINC_METHOD                            (0x00000002)
#define LW826F_DMA_OPCODE3_NONE                                    (0x00000000)
/* dma data format */
#define LW826F_DMA_DATA                                            31:0
/* dma double header descriptor format */
#define LW826F_DMA_DH_OPCODE2                                      1:0
#define LW826F_DMA_DH_OPCODE2_NONE                                 (0x00000000)
#define LW826F_DMA_DH_METHOD_ADDRESS                               12:2
#define LW826F_DMA_DH_METHOD_SUBCHANNEL                            15:13
#define LW826F_DMA_DH_OPCODE3                                      17:16
#define LW826F_DMA_DH_OPCODE3_DOUBLE_HEADER                        (0x00000003)
#define LW826F_DMA_DH_OPCODE                                       31:29
#define LW826F_DMA_DH_OPCODE_METHOD                                (0x00000000)
/* dma double header method count format */
#define LW826F_DMA_DH_METHOD_COUNT                                 23:0
/* dma double header data format */
#define LW826F_DMA_DH_DATA                                         31:0
/* dma nop format */
#define LW826F_DMA_NOP                                             (0x00000000)
/* dma set subdevice mask format */
#define LW826F_DMA_SET_SUBDEVICE_MASK                              (0x00010000)
#define LW826F_DMA_SET_SUBDEVICE_MASK_VALUE                        15:4
#define LW826F_DMA_OPCODE3                                         17:16
#define LW826F_DMA_OPCODE3_SET_SUBDEVICE_MASK                      (0x00000001)

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl826f_h_ */
