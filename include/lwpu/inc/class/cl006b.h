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

#ifndef _cl006b_h_
#define _cl006b_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

/* class LW03_CHANNEL_DMA */
#define  LW03_CHANNEL_DMA                                          (0x0000006B)
/* LwNotification[] fields and values */
#define LW06B_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
/* pio method data structure */
typedef volatile struct _cl006b_tag0 {
 LwV32 Reserved00[0x7c0];
} Lw06bTypedef, Lw03ChannelDma;
#define LW06B_TYPEDEF                                            Lw03ChannelDma
#define lw03ChannelDma                                           Lw03ChannelDma

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl006b_h_ */
