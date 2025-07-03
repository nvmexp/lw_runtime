/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2021 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "common_lwswitch.h"
#include "soe/haldefs_soe_lwswitch.h"
#include "soe/soe_lwswitch.h"
#include "soe/soe_priv_lwswitch.h"

#include "export_lwswitch.h"

LW_STATUS
soeProcessMessages
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    if (pSoe->base.pHal->processMessages == NULL)
    {
        LWSWITCH_ASSERT(0);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    return pSoe->base.pHal->processMessages(device, pSoe);
}

LW_STATUS
soeWaitForInitAck
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    if (pSoe->base.pHal->waitForInitAck == NULL)
    {
        LWSWITCH_ASSERT(0);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    return pSoe->base.pHal->waitForInitAck(device, pSoe);
}



LwU32
soeService_HAL
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    if (pSoe->base.pHal->service == NULL)
    {
        LWSWITCH_ASSERT(0);
        return 0;
    }

    return pSoe->base.pHal->service(device, pSoe);
}

void
soeServiceHalt_HAL
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    if (pSoe->base.pHal->serviceHalt == NULL)
    {
        LWSWITCH_ASSERT(0);
        return;
    }

    pSoe->base.pHal->serviceHalt(device, pSoe);
}

void
soeEmemTransfer_HAL
(
    lwswitch_device *device,
    PSOE             pSoe,
    LwU32            dmemAddr,
    LwU8            *pBuf,
    LwU32            sizeBytes,
    LwU8             port,
    LwBool           bCopyFrom
)
{
    if (pSoe->base.pHal->ememTransfer == NULL)
    {
        LWSWITCH_ASSERT(0);
        return;
    }

    pSoe->base.pHal->ememTransfer(device, pSoe, dmemAddr, pBuf, sizeBytes, port, bCopyFrom);
}

LwU32
soeGetEmemSize_HAL
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    if (pSoe->base.pHal->getEmemSize == NULL)
    {
        LWSWITCH_ASSERT(0);
        return 0;
    }

    return pSoe->base.pHal->getEmemSize(device, pSoe);
}

LwU32
soeGetEmemStartOffset_HAL
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    if (pSoe->base.pHal->getEmemStartOffset == NULL)
    {
        LWSWITCH_ASSERT(0);
        return 0;
    }

    return pSoe->base.pHal->getEmemStartOffset(device, pSoe);
}

LW_STATUS
soeEmemPortToRegAddr_HAL
(
    lwswitch_device *device,
    PSOE             pSoe,
    LwU32            port,
    LwU32           *pEmemCAddr,
    LwU32           *pEmemDAddr
)
{
    if (pSoe->base.pHal->ememPortToRegAddr == NULL)
    {
        LWSWITCH_ASSERT(0);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    return pSoe->base.pHal->ememPortToRegAddr(device, pSoe, port, pEmemCAddr, pEmemDAddr);
}

void
soeServiceExterr_HAL
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    if (pSoe->base.pHal->serviceExterr == NULL)
    {
        LWSWITCH_ASSERT(0);
        return;
    }

    pSoe->base.pHal->serviceExterr(device, pSoe);
}

LW_STATUS
soeGetExtErrRegAddrs_HAL
(
    lwswitch_device *device,
    PSOE             pSoe,
    LwU32           *pExtErrAddr,
    LwU32           *pExtErrStat
)
{
    if (pSoe->base.pHal->getExtErrRegAddrs == NULL)
    {
        LWSWITCH_ASSERT(0);
        return LW_ERR_ILWALID_ARGUMENT;
    }

    return pSoe->base.pHal->getExtErrRegAddrs(device, pSoe, pExtErrAddr, pExtErrStat);
}

LwU32
soeEmemPortSizeGet_HAL
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    if (pSoe->base.pHal->ememPortSizeGet == NULL)
    {
        LWSWITCH_ASSERT(0);
        return 0;
    }

    return pSoe->base.pHal->ememPortSizeGet(device, pSoe);
}

LwBool
soeIsCpuHalted_HAL
(
    lwswitch_device *device,
    PSOE             pSoe
)
{
    if (pSoe->base.pHal->isCpuHalted == NULL)
    {
        LWSWITCH_ASSERT(0);
        return LW_FALSE;
    }

    return pSoe->base.pHal->isCpuHalted(device, pSoe);
}

LwlStatus
soeTestDma_HAL
(
    lwswitch_device *device,
    PSOE            pSoe
)
{
    if (pSoe->base.pHal->testDma == NULL)
    {
        LWSWITCH_ASSERT(0);
        return -LWL_BAD_ARGS;
    }

    return pSoe->base.pHal->testDma(device);
}

LwlStatus
soeSetPexEOM_HAL
(
    lwswitch_device *device,
    LwU8 mode,
    LwU8 nblks,
    LwU8 nerrs,
    LwU8 berEyeSel
)
{
    PSOE pSoe = (PSOE)device->pSoe;
    if (pSoe->base.pHal->setPexEOM == NULL)
    {
        LWSWITCH_ASSERT(0);
        return -LWL_BAD_ARGS;
    }

    return pSoe->base.pHal->setPexEOM(device, mode, nblks, nerrs, berEyeSel);
}

LwlStatus
soeGetPexEomStatus_HAL
(
    lwswitch_device *device,
    LwU8 mode,
    LwU8 nblks,
    LwU8 nerrs,
    LwU8 berEyeSel,
    LwU32 laneMask,
    LwU16 *pEomStatus
)
{
    PSOE pSoe = (PSOE)device->pSoe;
    if (pSoe->base.pHal->getPexEomStatus == NULL)
    {
        LWSWITCH_ASSERT(0);
        return -LWL_BAD_ARGS;
    }

    return pSoe->base.pHal->getPexEomStatus(device, mode, nblks, nerrs, berEyeSel, laneMask, pEomStatus);
}

LwlStatus
soeGetUphyDlnCfgSpace_HAL
(
    lwswitch_device *device,
    LwU32 regAddress,
    LwU32 laneSelectMask,
    LwU16 *pRegValue
)
{
    PSOE pSoe = (PSOE)device->pSoe;
    if (pSoe->base.pHal->getUphyDlnCfgSpace == NULL)
    {
        LWSWITCH_ASSERT(0);
        return -LWL_BAD_ARGS;
    }

    return pSoe->base.pHal->getUphyDlnCfgSpace(device, regAddress, laneSelectMask, pRegValue);
}

LwlStatus
soeForceThermalSlowdown_HAL
(
    lwswitch_device *device,
    LwBool slowdown,
    LwU32  periodUs
)
{
    PSOE pSoe = (PSOE)device->pSoe;
    if (pSoe->base.pHal->forceThermalSlowdown == NULL)
    {
        LWSWITCH_ASSERT(0);
        return -LWL_BAD_ARGS;
    }

    return pSoe->base.pHal->forceThermalSlowdown(device, slowdown, periodUs);
}

LwlStatus
soeSetPcieLinkSpeed_HAL
(
    lwswitch_device *device,
    LwU32 linkSpeed
)
{
    PSOE pSoe = (PSOE)device->pSoe;
    if (pSoe->base.pHal->setPcieLinkSpeed == NULL)
    {
        LWSWITCH_ASSERT(0);
        return -LWL_BAD_ARGS;
    }

    return pSoe->base.pHal->setPcieLinkSpeed(device, linkSpeed);
}
