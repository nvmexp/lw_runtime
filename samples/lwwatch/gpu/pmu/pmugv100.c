/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2016-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "pmu.h"

#include <volta/gv100/dev_pwr_pri.h>

/*!
 *  Returns the number of queues on the PMU. Note that there are N many
 *  "command queues" and one message queue. The queues are accessed by their
 *  number but it can be assumed that the maximum queue index is the message
 *  queue.
 *
 *  @return Number of queue on PMU.
 */
LwU32
pmuQueueGetNum_GV100(void)
{
    // We have command queues + 1 message queue
    return LW_PPWR_PMU_QUEUE_HEAD__SIZE_1 + 1;
}

/*! Returns the head of the 'command queue' on the PMU. */
LwU32
pmuQueueReadCommandHead_GV100
(
    LwU32 queueId
)
{
    return PMU_REG_RD32(LW_PPWR_PMU_QUEUE_HEAD(queueId));
}

/*! Returns the tail of the 'command queue' on the PMU. */
LwU32
pmuQueueReadCommandTail_GV100
(
    LwU32 queueId
)
{
    return PMU_REG_RD32(LW_PPWR_PMU_QUEUE_TAIL(queueId));
}

const char *
pmuUcodeName_GV100()
{
    return "g_c85b6_gv10x";
}

/*!
 * @return The falcon base address of PMU
 */
LwU32
pmuGetFalconBase_GV100()
{
    return DEVICE_BASE(LW_PPWR);
}

