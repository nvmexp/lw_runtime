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

#ifndef _cl0030_h_
#define _cl0030_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define  LW01_NULL                                                 (0x00000030)
/* LwNotification[] fields and values */
#define LW030_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
/* pio method data structure */
typedef volatile struct _cl0030_tag0 {
 LwV32 Reserved00[0x7c0];
} Lw030Typedef, Lw01Null;
#define LW030_TYPEDEF                                              Lw01Null
/* obsolete stuff */
#define LW1_NULL                                                   (0x00000030)
#define Lw1Null                                                    Lw01Null
#define lw1Null                                                    Lw01Null
#define lw01Null                                                   Lw01Null

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl0030_h_ */
