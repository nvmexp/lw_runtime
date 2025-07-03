/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2018 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
#include "sec2.h"

#include "turing/tu102/dev_sec_pri.h"
#include "turing/tu102/dev_sec_addendum.h"
/*!
 * Object base address initialization
 */
void
sec2ObjBaseAddr_TU10X()
{
    pObjSec2               = &ObjSec2;
    pObjSec2->getRegAddr   = sec2GetRegAddr;
    pObjSec2->readRegAddr  = sec2RegRdAddr;
    pObjSec2->writeRegAddr = sec2RegWrAddr;
    pObjSec2->registerBase = LW_PSEC_FALCON_IRQSSET;
    pObjSec2->fbifBase     = LW_PSEC_FBIF_TRANSCFG(0);
}

/*!
 * Returns LW_PSEC_QUEUE_HEAD__SIZE_1 value
 */
LwU32
sec2GetQueueHeadSize_TU10X()
{
    return LW_PSEC_QUEUE_HEAD__SIZE_1;
}

/*!
 * Returns LW_PSEC_MSGQ_HEAD__SIZE_1 value
 */
LwU32
sec2GetMsgqHeadSize_TU10X()
{
    return LW_PSEC_MSGQ_HEAD__SIZE_1;
}

/*!
 * Returns LW_PSEC_QUEUE_HEAD__SIZE_1 value
 */
LwU32
sec2GetQueueHead_TU10X(LwU32 queueId)
{
    return LW_PSEC_QUEUE_HEAD(queueId);
}

/*!
 * Returns LW_PSEC_QUEUE_TAIL value
 */
LwU32
sec2GetQueueTail_TU10X(LwU32 queueId)
{
    return LW_PSEC_QUEUE_TAIL(queueId);
}

/*!
 * Returns LW_PSEC_MSGQ_HEAD value
 */
LwU32
sec2GetMsgqHead_TU10X(LwU32 queueId)
{
    return LW_PSEC_MSGQ_HEAD(queueId);
}

/*!
 * Returns LW_PSEC_MSGQ_TAIL value
 */
LwU32
sec2GetMsgqTail_TU10X(LwU32 queueId)
{
    return LW_PSEC_MSGQ_TAIL(queueId);
}

/*!
 * Returns LW_PSEC_EMEMC__SIZE_1 value
 */
LwU32
sec2GetEmemcSize_TU10X()
{
    return LW_PSEC_EMEMC__SIZE_1;
}

/*!
 * Returns LW_PSEC_EMEMC(i) value
 */
LwU32
sec2GetEmemc_TU10X(LwU32 port)
{
    return LW_PSEC_EMEMC(port);
}

/*!
 * Returns LW_PSEC_EMEMD(i) value
 */
LwU32
sec2GetEmemd_TU10X(LwU32 port)
{
    return LW_PSEC_EMEMD(port);
}

/*!
 * Returns the physical address of LW_PSEC_MUTEX_ID register.
 */
LwU32
sec2GetMutexId_TU10X()
{
    return LW_PSEC_MUTEX_ID;
}


/*!
 * @brief Checks if SEC2 DEBUG fuse is blown or not
 *
 */
LwBool
sec2IsDebugMode_TU10X()
{
    LwU32 ctlStat =  GPU_REG_RD32(LW_PSEC_SCP_CTL_STAT);

    return !FLD_TEST_DRF(_PSEC, _SCP_CTL_STAT, _DEBUG_MODE, _DISABLED, ctlStat);
}

/*!
 * @return The falcon base address of PMU
 */
LwU32
sec2GetFalconBase_TU10X()
{
    return DEVICE_BASE(LW_PSEC);
}

LwU32
sec2EmemGetNumPorts_TU10X()
{
    return LW_PSEC_EMEMD__SIZE_1;
}

LwU32
sec2GetEmemPortId_TU10X()
{
    return LW_SEC2_EMEM_ACCESS_PORT_LWWATCH;
}

LwU32
sec2EmemGetSize_TU10X()
{
    return GPU_REG_RD_DRF(_PSEC, _HWCFG, _EMEM_SIZE) * FLCN_BLK_ALIGNMENT;
}
