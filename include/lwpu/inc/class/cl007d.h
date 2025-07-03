/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2004-2004 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl007d_h_
#define _cl007d_h_

#ifdef __cplusplus
extern "C" {
#endif

#include "lwtypes.h"

#define LW04_SOFTWARE_TEST                                         (0x0000007D)
#define LW07D                                             0x00001fff:0x00000000
/* LwNotification[] elements */
#define LW07D_NOTIFIERS_NOTIFY                                     (0)
#define LW07D_NOTIFIERS_MAXCOUNT                                   (1)
/* LwNotification[] fields and values */
#define LW07D_NOTIFICATION_STATUS_IN_PROGRESS                      (0x8000)
#define LW07D_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
#define LW07D_NOTIFICATION_STATUS_ERROR_BAD_ARGUMENT               (0x2000)
#define LW07D_NOTIFICATION_STATUS_ERROR_ILWALID_STATE              (0x1000)
#define LW07D_NOTIFICATION_STATUS_ERROR_STATE_IN_USE               (0x0800)
#define LW07D_NOTIFICATION_STATUS_DONE_SUCCESS                     (0x0000)

/* pio method data structures */
typedef volatile struct _cl007d_tag0 {
 LwV32 NoOperation;             /* ignored                          0100-0103*/
 LwV32 Notify;                  /* LW07D_NOTIFY_*                   0104-0107*/
 LwV32 Reserved0104[0x78/4];
 LwV32 SetContextDmaNotifies;   /* LW01_CONTEXT_DMA                 0180-0183*/
 LwV32 Reserved0184[0x1f7c/4];
} Lw07dTypedef, Lw04SoftwareTest;

#define LW07D_TYPEDEF                                          Lw04SoftwareTest
/* dma method offsets, fields, and values */
#define LW07D_SET_OBJECT                                           (0x00000000)
#define LW07D_NO_OPERATION                                         (0x00000100)

#ifdef __cplusplus
};     /* extern "C" */
#endif

#endif /* _cl007d_h_ */
