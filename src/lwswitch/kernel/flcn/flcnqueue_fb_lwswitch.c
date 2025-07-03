/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2022 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"

#include "flcn/flcnable_lwswitch.h"
#include "flcn/flcn_lwswitch.h"
#include "flcn/flcnqueue_lwswitch.h"

/*!
 * @file   flcnqueue_fb.c
 * @brief  Provides all functions specific to FB Queue (non-DMEM queues).
 *
 * Queues are the primary communication mechanism between the RM and various
 * falcon-based engines such as the PMU and Display Falcon.  The RM requests
 * actions by inserting a data packet (command) into a command queue. This
 * generates an interrupt to the falcon which allows it to wake-up and service
 * the request.  Upon completion of the command, the falcon can optionally
 * write an acknowledgment packet (message) into a separate queue designated
 * for RM-bound messages.  CMDs sent by an FB CMD queue must send a
 * response, as that is required to clear that CMD queue element's "in use bit"
 * and, free the DMEM allocation associated with it.
 *
 * For more information on FB Queue see:
  *     PMU FB Queue (RID-70296)
 * For general queue information, see the HDR of flcnqueue.c.
 * For information specific to DMEM queues, see the HDR of flcnqueue_dmem.c
 *
 * Each queue has distinct "head" and "tail" pointers. The "head" pointer is the
 * index of the queue Element where the next write operation will take place;
 * the "tail" marks the index of the queue Element for the next read.  When the
 * head and tail pointers are equal, the queue is empty.  When non-equal, data
 * exists in the queue that needs to be processed.  Queues are always allocated
 * in the Super Surface in FB.
 */

