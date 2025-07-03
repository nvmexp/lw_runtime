/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2016 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "sec2.h"

#include "g_sec2_private.h"

static FLCN_ENGINE_IFACES flcnEngineIfaces_sec2 =
{
    sec2GetFalconCoreIFace_STUB,         // flcnEngGetCoreIFace
    sec2GetFalconBase_STUB,              // flcnEngGetFalconBase
    sec2GetEngineName,                   // flcnEngGetEngineName
    sec2UcodeName_STUB,                  // flcnEngUcodeName
    sec2GetSymFilePath,                  // flcnEngGetSymFilePath
    sec2QueueGetNum_STUB,                // flcnEngQueueGetNum
    sec2QueueRead_STUB,                  // flcnEngQueueRead
    sec2GetDmemAccessPort,               // flcnEngGetDmemAccessPort
    sec2IsDmemRangeAccessible_STUB,      // flcnEngIsDmemRangeAccessible
    sec2EmemGetOffsetInDmemVaSpace_STUB, // flcnEngEmemGetOffsetInDmemVaSpace
    sec2EmemGetSize_STUB,                // flcnEngEmemGetSize
    sec2EmemGetNumPorts_STUB,            // flcnEngEmemGetNumPorts
    sec2EmemRead_STUB,                   // flcnEngEmemRead
    sec2EmemWrite_STUB,                  // flcnEngEmemWrite
}; // flcnEngineIfaces_sec2

/*!
 * @return Falcon core interface
 */
const FLCN_CORE_IFACES *
sec2GetFalconCoreIFace_GM200()
{
    return &flcnCoreIfaces_v05_01;
}

const FLCN_ENGINE_IFACES *
sec2GetFalconEngineIFace_GM200()
{
    const FLCN_CORE_IFACES   *pFCIF = pSec2[indexGpu].sec2GetFalconCoreIFace();
          FLCN_ENGINE_IFACES *pFEIF = NULL;

    // The Falcon Engine interface is supported only when the Core Interface exists.
    if (pFCIF)
    {
        pFEIF = &flcnEngineIfaces_sec2;

        pFEIF->flcnEngGetCoreIFace          = pSec2[indexGpu].sec2GetFalconCoreIFace;
        pFEIF->flcnEngGetFalconBase         = pSec2[indexGpu].sec2GetFalconBase;
        pFEIF->flcnEngQueueGetNum           = pSec2[indexGpu].sec2QueueGetNum;
        pFEIF->flcnEngQueueRead             = pSec2[indexGpu].sec2QueueRead;
        pFEIF->flcnEngIsDmemRangeAccessible = pSec2[indexGpu].sec2IsDmemRangeAccessible;
    }

    return pFEIF;
}

/*!
 * @return Get the number of DMEM carveouts
 */
LwU32
sec2GetDmemNumPrivRanges_GM200(void)
{
    // No register to read this, so hardcode until HW adds support.
    return 0; // No carveouts on SEC2 until bug 1482136 is fixed
}

/*!
 * @return LW_TRUE  DMEM range is accessible
 *         LW_FALSE DMEM range is inaccessible
 */
LwBool
sec2IsDmemRangeAccessible_GM200
(
    LwU32 blkLo,
    LwU32 blkHi
)
{
    // Print out info message
    dprintf("lw: SEC2 is in LS mode. There are no DMEM carveouts that CPU "
            "can access when SEC2 is not in NS mode. See HW bug 1482136.\n");
    return LW_FALSE; // No carveouts on SEC2 until bug 1482136 is fixed
}
