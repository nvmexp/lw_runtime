/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2006-2007 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _cl906d_h_
#define _cl906d_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"
    
/* class GF100_CHANNEL_PIO  */
/*
 * Documentation for GF100_CHANNEL_PIO can be fouind in dev_fifo.ref,
 * LW_PFIFO_PIO_*, for the privileged PIO regsiters; and in dev_pio.ref
 * for the user-level registers. The definition of an engine's class
 * methods is described elsewhere.
 */
#define  GF100_CHANNEL_PIO                                  (0x0000906D)

/* pio method data structure */
typedef volatile struct _cl906d_tag0 {
    LwU32 Header;                            /*  0x04 - 0x07 */
    LwU32 Data;                              /*  0x08 - 0x0b */
} Lw906dMethod;

typedef volatile struct _cl906d_tag1 {
    LwU32 Status;                            /*  0x00 - 0x03 */
    Lw906dMethod Method;                     /*  0x04 - 0x0b */
    LwV32 Reserved[0x3FD];
} Lw906dChannelPio;

//
// Lw906dChannelPio::Status
// 
#define LW906D_CHANNEL_PIO_STATUS_IDLE                              0x00000000 /* R---V */
#define LW906D_CHANNEL_PIO_STATUS_CTXSW_BUSY                               0:0 /* R-IUF */
#define LW906D_CHANNEL_PIO_STATUS_CTXSW_BUSY_FALSE                  0x00000000 /* R-I-V */
#define LW906D_CHANNEL_PIO_STATUS_CTXSW_BUSY_TRUE                   0x00000001 /* R---V */
#define LW906D_CHANNEL_PIO_STATUS_METHOD_BUSY                              4:4 /* R-IUF */
#define LW906D_CHANNEL_PIO_STATUS_METHOD_BUSY_FALSE                 0x00000000 /* R-I-V */
#define LW906D_CHANNEL_PIO_STATUS_METHOD_BUSY_TRUE                  0x00000001 /* R---V */
#define LW906D_CHANNEL_PIO_STATUS_ERROR                                   11:8 /* R-IUF */
#define LW906D_CHANNEL_PIO_STATUS_ERROR_NONE                        0x00000000 /* R-I-V */
#define LW906D_CHANNEL_PIO_STATUS_ERROR_WRITE_WHILE_DISABLED        0x00000001 /* R---V */
#define LW906D_CHANNEL_PIO_STATUS_ERROR_ENABLED_WHILE_BUSY          0x00000002 /* R---V */
#define LW906D_CHANNEL_PIO_STATUS_ERROR_WRITE_ENGINE_ILWALID        0x00000003 /* R---V */
#define LW906D_CHANNEL_PIO_STATUS_ERROR_WRITE_WHILE_BUSY            0x00000004 /* R---V */
#define LW906D_CHANNEL_PIO_STATUS_ERROR_CODE_N                      0x00000005 /* R---V */


//
// Lw906dChannelPio::Method::Header
// 
#define LW906D_CHANNEL_PIO_METHOD_HDR_ADDR_ALIGN                           1:0 /* C--UF */
#define LW906D_CHANNEL_PIO_METHOD_HDR_ADDR_ALIGN_DWORD              0x00000000 /* C---V */
#define LW906D_CHANNEL_PIO_METHOD_HDR_ADDR                                13:2 /* RWIUF */
#define LW906D_CHANNEL_PIO_METHOD_HDR_ADDR_NULL                     0x00000000 /* RWI-V */
#define LW906D_CHANNEL_PIO_METHOD_HDR_SUBCH                              18:16 /* RWIUF */
#define LW906D_CHANNEL_PIO_METHOD_HDR_SUBCH_0                       0x00000000 /* RWI-V */
#define LW906D_CHANNEL_PIO_METHOD_HDR_PRIV                               20:20 /* RWIUF */
#define LW906D_CHANNEL_PIO_METHOD_HDR_PRIV_USER                     0x00000000 /* RWI-V */
#define LW906D_CHANNEL_PIO_METHOD_HDR_PRIV_KERNEL                   0x00000001 /* RW--V */


//
// Lw906dChannelPio::Method::Data
// 
#define LW906D_CHANNEL_PIO_METHOD_DATA_V                             31:0 /* RWXUF */

//
// Engine Defines.
// 
#define LW906D_CHANNEL_PIO_TARGET_ENGINE_GRAPHICS                   0x00000000
#define LW906D_CHANNEL_PIO_TARGET_ENGINE_MSPDEC                     0x00000001
#define LW906D_CHANNEL_PIO_TARGET_ENGINE_MSPPP                      0x00000002
#define LW906D_CHANNEL_PIO_TARGET_ENGINE_MSVLD                      0x00000003
#define LW906D_CHANNEL_PIO_TARGET_ENGINE_COPY0                      0x00000004
#define LW906D_CHANNEL_PIO_TARGET_ENGINE_COPY1                      0x00000005
#define LW906D_CHANNEL_PIO_TARGET_ENGINE_FIRST                      LW906D_CHANNEL_PIO_TARGET_ENGINE_GRAPHICS
#define LW906D_CHANNEL_PIO_TARGET_ENGINE_LAST                       LW906D_CHANNEL_PIO_TARGET_ENGINE_COPY1
#define LW906D_CHANNEL_PIO_TARGET_ENGINE_SW                         0x0000001F

//
// fields and values
//
#define LW906D_CHANNEL_PIO_NUMBER_OF_SUBCHANNELS                            (8)
#define LW906D_CHANNEL_PIO_SET_OBJECT                              (0x00000000)
#define LW906D_CHANNEL_PIO_SET_OBJECT_CLASS_ID                             15:0
#define LW906D_CHANNEL_PIO_SET_OBJECT_ENGINE_ID                           20:16

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl906d_h_ */

