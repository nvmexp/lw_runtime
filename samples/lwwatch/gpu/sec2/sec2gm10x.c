/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2019 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "os.h"

#include "chip.h"
#include "disp.h"
#include "pmu.h"
#include "sig.h"
#include "fb.h"
#include "fifo.h"
#include "inst.h"
#include "clk.h"
#include "acr.h"
#include "falcphys.h"
#include "smbpbi.h"
#include "sec2.h"

#include "g_sec2_private.h"          // (rmconfig) hal/obj setup

#include "maxwell/gm107/dev_sec_pri.h"
#include "maxwell/gm107/dev_master.h"
#include "maxwell/gm107/dev_falcon_v4.h"

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
 * @return The falcon engine interface FLCN_ENGINE_IFACES*
 */
const FLCN_ENGINE_IFACES *
sec2GetFalconEngineIFace_GM107()
{
    const FLCN_CORE_IFACES   *pFCIF = NULL;
          FLCN_ENGINE_IFACES *pFEIF = NULL;

    pFCIF = pSec2[indexGpu].sec2GetFalconCoreIFace();

    // The Falcon Engine interface is supported only when the Core Interface exists.
    if(pFCIF != NULL)
    {
        pFEIF = &flcnEngineIfaces_sec2;

        pFEIF->flcnEngGetCoreIFace          = pSec2[indexGpu].sec2GetFalconCoreIFace;
        pFEIF->flcnEngGetFalconBase         = pSec2[indexGpu].sec2GetFalconBase;
        pFEIF->flcnEngIsDmemRangeAccessible = pSec2[indexGpu].sec2IsDmemRangeAccessible;
    }

    return pFEIF;
}

//-----------------------------------------------------
// sec2IsSupported_GM10X
//-----------------------------------------------------
LwBool
sec2IsSupported_GM107()
{
    return TRUE;
}


/*!
 * @brief Checks if SEC2 DEBUG fuse is blown or not
 *
 */
LwBool
sec2IsDebugMode_GM107()
{
    LwU32 ctlStat =  GPU_REG_RD32(LW_PSEC_SCP_CTL_STAT);

    return !FLD_TEST_DRF(_PSEC, _SCP_CTL_STAT, _DEBUG_MODE, _DISABLED, ctlStat);
}

/*!
 *  Reset SEC2
 */
LW_STATUS sec2MasterReset_GM107()
{
    LwU32 reg32;
    LwU32 timeoutUs = SEC2_RESET_TIMEOUT_US;

    reg32 = GPU_REG_RD32(LW_PMC_ENABLE);
    reg32 = FLD_SET_DRF(_PMC, _ENABLE, _SEC, _DISABLED, reg32);
    GPU_REG_WR32(LW_PMC_ENABLE, reg32);
    reg32 = FLD_SET_DRF(_PMC, _ENABLE, _SEC, _ENABLED, reg32);
    GPU_REG_WR32(LW_PMC_ENABLE, reg32);

    // Wait for SCRUBBING to complete
    while (timeoutUs > 0)
    {
        reg32 = GPU_REG_RD32(LW_PSEC_FALCON_DMACTL);

        if (FLD_TEST_DRF(_PSEC, _FALCON_DMACTL, _DMEM_SCRUBBING, _DONE, reg32) &&
            FLD_TEST_DRF(_PSEC, _FALCON_DMACTL, _IMEM_SCRUBBING, _DONE, reg32))
        {
            break;
        }
        osPerfDelay(20);
        timeoutUs -= 20;
    }

    if (timeoutUs == 0)
    {
        return LW_ERR_GENERIC;
    }

    return LW_OK;
}

/*!
 * @return Falcon core interface
 */
const FLCN_CORE_IFACES *
sec2GetFalconCoreIFace_GM107()
{
    return &flcnCoreIfaces_v04_00;
}

/*!
 * @return The falcon base address of PMU
 */
LwU32
sec2GetFalconBase_GM107()
{
    return DEVICE_BASE(LW_PSEC);
}

/*!
 * Object base address initialization
 */
void
sec2ObjBaseAddr_GM107()
{
    pObjSec2               = &ObjSec2;
    pObjSec2->getRegAddr   = sec2GetRegAddr;
    pObjSec2->readRegAddr  = sec2RegRdAddr;
    pObjSec2->writeRegAddr = sec2RegWrAddr;
    pObjSec2->registerBase = LW_PSEC_FALCON_IRQSSET;
    pObjSec2->fbifBase     = LW_PSEC_FBIF_TRANSCFG(0);
}
