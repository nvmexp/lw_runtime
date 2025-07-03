/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2006 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl208f_h_
#define _cl208f_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

/* Class within the subdevice used for diagnostic purpose*/
#define  LW20_SUBDEVICE_DIAG                                       (0x0000208f)

/* event values */
#define LW208F_NOTIFIERS_SW                                        (0)
#define LW208F_NOTIFIERS_MAXCOUNT                                  (1)

/* LwNotification[] fields and values */
#define LW208f_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT          (0x4000)
/* pio method data structure */
typedef volatile struct _cl208f_tag0 {
 LwV32 Reserved00[0x7c0];
} Lw208fTypedef, Lw20SubdeviceDiag;
#define  LW208f_TYPEDEF                                            Lw20SubdeviceDiag

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl208f_h_ */

