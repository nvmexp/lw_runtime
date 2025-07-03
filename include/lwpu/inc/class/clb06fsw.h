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

#ifndef _clb06f_sw_h_
#define _clb06f_sw_h_

#define LWB06F_NOTIFIERS_RC                                         (0)
#define LWB06F_NOTIFIERS_REFCNT                                     (1)
#define LWB06F_NOTIFIERS_NONSTALL                                   (2)
#define LWB06F_NOTIFIERS_EVENTBUFFER                                (3)
#define LWB06F_NOTIFIERS_IDLECHANNEL                                (4)
#define LWB06F_NOTIFIERS_ENDCTX                                     (5)
#define LWB06F_NOTIFIERS_SW                                         (6)
#define LWB06F_NOTIFIERS_GR_DEBUG_INTR                              (7)
#define LWB06F_NOTIFIERS_REPLAYABLE_FAULT                           (8)
#define LWB06F_NOTIFIERS_MAXCOUNT                                   (9)

/* LwNotification[] fields and values */
#define LWB06F_NOTIFICATION_STATUS_ERROR_BAD_ARGUMENT              (0x2000)
#define LWB06F_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT          (0x4000)

#endif /* _clb06f_sw_h_ */
