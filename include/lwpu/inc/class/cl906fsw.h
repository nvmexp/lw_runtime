/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2015 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#ifndef _cl906f_sw_h_
#define _cl906f_sw_h_

/* LwNotification[] elements */
#define LW906F_NOTIFIERS_RC                                         (0)
#define LW906F_NOTIFIERS_REFCNT                                     (1)
#define LW906F_NOTIFIERS_NONSTALL                                   (2)
#define LW906F_NOTIFIERS_EVENTBUFFER                                (3)
#define LW906F_NOTIFIERS_IDLECHANNEL                                (4)
#define LW906F_NOTIFIERS_ENDCTX                                     (5)
#define LW906F_NOTIFIERS_SW                                         (6)
#define LW906F_NOTIFIERS_GR_DEBUG_INTR                              (7)
#define LW906F_NOTIFIERS_MAXCOUNT                                   (8)

/* LwNotification[] fields and values */
#define LW906f_NOTIFICATION_STATUS_ERROR_BAD_ARGUMENT              (0x2000)
#define LW906f_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT          (0x4000)

#endif /* _cl906f_sw_h_ */
