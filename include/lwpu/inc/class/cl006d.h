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

#ifndef _cl006d_h_
#define _cl006d_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

/* class LW04_CHANNEL_PIO */
#define  LW04_CHANNEL_PIO                                          (0x0000006D)

/* LwNotification[] fields and values */
#define LW06D_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
#define LW06D_NOTIFICATION_STATUS_ERROR_BAD_ARGUMENT               (0x2000)
#define LW06D_NOTIFICATION_STATUS_ERROR_FLOW_CONTROL               (0x0200)

/* pio subchannel method data structure */
typedef volatile struct _cl006d_tag0 {
    LwV32 Reserved00[0x003];
#if LWCPU_IS_BIG_ENDIAN
    LwU32 Free;                    /* 32 bit free count, read only     0010-0013*/
    LwU32 Zero;                    /* zeroes, read only                0014-0017*/
#else
    LwU16 Free;                    /* free count, read only            0010-0011*/
    LwU16 Zero[3];                 /* zeroes, read only                0012-0017*/
#endif
    LwV32 Reserved01[0x03A];
} Lw04ControlPio;

typedef volatile struct _cl006d_tag1 {
    LwV32 SetObject;               /* handle of current object         0000-0003*/
    Lw04ControlPio control;        /* flow control                     0000-00ff*/

} Lw04SubchannelPio;

/* pio channel */
typedef volatile struct _cl006d_tag2 {               /* start of array of subchannels     0000-    */
    Lw04SubchannelPio subchannel[8];    /*subchannel                       0000-1fff*/
} Lw04ChannelPio;                       /* end of array of subchannels           -ffff*/

/* fields and values */
#define LW06D_FIFO_GUARANTEED_SIZE                                 (0x0200)
#define LW06D_FIFO_EMPTY                                           (0x0200)
#define LW06D_FIFO_FULL                                            (0x0000)

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl006d_h_ */

