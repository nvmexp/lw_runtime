/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright 2004-2010 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef _cl_gt21a_tesla_sw_h_
#define _cl_gt21a_tesla_sw_h_

/* This file is *not* auto-generated. */

/* LwNotification[] elements */
#define LW8697_NOTIFIERS_NOTIFY                                     (0)
#define LW8697_NOTIFIERS_SET_MEMORY_SURFACE_ATTR                    (0)
#define LW8697_NOTIFIERS_DEBUG_INTR                                 (1)
#define LW8697_NOTIFIERS_MAXCOUNT                                   (2)

/* LwNotification[] fields and values */
#define LW8697_NOTIFICATION_STATUS_IN_PROGRESS                      (0x8000)
#define LW8697_NOTIFICATION_STATUS_ERROR_PROTECTION_FAULT           (0x4000)
#define LW8697_NOTIFICATION_STATUS_ERROR_BAD_ARGUMENT               (0x2000)
#define LW8697_NOTIFICATION_STATUS_ERROR_ILWALID_STATE              (0x1000)
#define LW8697_NOTIFICATION_STATUS_ERROR_STATE_IN_USE               (0x0800)
#define LW8697_NOTIFICATION_STATUS_DONE_SUCCESS                     (0x0000)

#define LW8697_RUN_GEOM_AS_OCTS                                              0x12a4
#define LW8697_RUN_GEOM_AS_OCTS_ENABLE_WAR                                   0:0
#define LW8697_RUN_GEOM_AS_OCTS_ENABLE_WAR_FALSE                             0x00000000
#define LW8697_RUN_GEOM_AS_OCTS_ENABLE_WAR_TRUE                              0x00000001

#endif /* _cl_gt21a_tesla_sw_h_ */

