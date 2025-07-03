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

#ifndef _cl006a_h_
#define _cl006a_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"
#include "class/cl004d.h"

/* class LW03_CHANNEL_PIO */
#define  LW03_CHANNEL_PIO                                          (0x0000006A)

    /* LwNotification[] fields and values */
#define LW06A_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
#define LW06A_NOTIFICATION_STATUS_ERROR_BAD_ARGUMENT               (0x2000)
#define LW06A_NOTIFICATION_STATUS_ERROR_FLOW_CONTROL               (0x0200)

    /* pio subchannel method data structure */
typedef volatile struct _cl006a_tag0 {
    LwV32 Reserved00[0x003];
#if LWCPU_IS_BIG_ENDIAN
    LwU32 Free;                    /* 32 bit free count, read only     0010-0013*/
    LwU32 Zero;                    /* zeroes, read only                0014-0017*/
#else
    LwU16 Free;                    /* free count, read only            0010-0011*/
    LwU16 Zero[3];                 /* zeroes, read only                0012-0017*/
#endif
    LwV32 Reserved01[0x03A];
} Lw03ControlPio;

typedef volatile struct _cl006a_tag1 {
    LwV32 SetObject;               /* handle of current object         0000-0003*/
    Lw03ControlPio control;        /* flow control                     0000-00ff*/

#ifndef _H2INC
    union {                        /* start of class methods           0100-    */
        Lw04dTypedef LW04D_TYPEDEF;
#if defined(__GNUC__ )
/* anon union does not work on GCC */
    }  cls   ;         /* end of class methods                  -1fff*/
#else  /* __GNUC__ */
    }  /* cls */  ;    /* end of class methods                  -1fff*/
#endif /* __GNUC__ */
#endif /* !_H2INC */

} Lw03SubchannelPio;

/* pio channel */
typedef volatile struct _cl006a_tag2 {      /* start of array of subchannels     0000-    */
    Lw03SubchannelPio subchannel[8];/*subchannel                       0000-1fff*/
} Lw03ChannelPio;              /* end of array of subchannels           -ffff*/

/* fields and values */
#define LW06A_FIFO_GUARANTEED_SIZE                                 (0x007C)
#define LW06A_FIFO_EMPTY                                           (0x007C)
#define LW06A_FIFO_FULL                                            (0x0000)

/* obsolete stuff */
#define LW3_CHANNEL_PIO                                            (0x0000006A)
#define Lw3ControlPio                                              Lw03ControlPio
#define Lw3SubchannelPio                                           Lw03SubchannelPio
#define Lw3ChannelPio                                              Lw03ChannelPio
#define LwChannel                                                  Lw03ChannelPio
#define lw03ChannelPio                                             Lw03ChannelPio

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl006a_h_ */

