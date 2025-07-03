/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
 
//*****************************************************
//
// DPU Hal Functions
// dpu0205.c
//
//*****************************************************


/* ------------------------ Includes --------------------------------------- */
#include "dpu.h"
#include "dpu/v02_05/dev_disp_falcon.h"

#include "g_dpu_private.h"     // (rmconfig)  implementation prototypes


/* ------------------------ Defines ---------------------------------------- */
/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Globals ---------------------------------------- */
const static FLCN_ENGINE_IFACES flcnEngineIfaces_dpu0205 =
{
    dpuGetFalconCoreIFace_v02_05,               // flcnEngGetCoreIFace    
    dpuGetFalconBase_v02_01,                    // flcnEngGetFalconBase    
    dpuGetEngineName,                           // flcnEngGetEngineName    
    dpuUcodeName_v02_05,                        // flcnEngUcodeName
    dpuGetSymFilePath,                          // flcnEngGetSymFilePath
    dpuQueueGetNum_v02_01,                      // flcnEngQueueGetNum
    dpuQueueRead_v02_01,                        // flcnEngQueueRead
    dpuGetDmemAccessPort,                       // flcnEngGetDmemAccessPort
    dpuIsDmemRangeAccessible_v02_05,            // flcnEngIsDmemRangeAccessible
    dpuEmemGetOffsetInDmemVaSpace_STUB,         // flcnEngEmemGetOffsetInDmemVaSpace
    dpuEmemGetSize_STUB,                        // flcnEngEmemGetSize
    dpuEmemGetNumPorts_STUB,                    // flcnEngEmemGetNumPorts
    dpuEmemRead_STUB,                           // flcnEngEmemRead
    dpuEmemWrite_STUB,                          // flcnEngEmemWrite
};  // falconEngineIfaces_dpu


/* ------------------------ Functions -------------------------------------- */

/*!
 * @return The falcon engine interface FLCN_ENGINE_IFACES*
 */
const FLCN_ENGINE_IFACES *
dpuGetFalconEngineIFace_v02_05()
{
    return &flcnEngineIfaces_dpu0205;
}

const FLCN_CORE_IFACES *
dpuGetFalconCoreIFace_v02_05()
{
    return &flcnCoreIfaces_v05_01;
}

/*!
 * Return Ucode file name
 *
 * @return Ucode file name
 */
const char*
dpuUcodeName_v02_05()
{
    return "g_dpuuc0205";
}

/*!
 * @return Get the number of DMEM carveouts
 */
LwU32
dpuGetDmemNumPrivRanges_v02_05(void)
{
    // No register to read this, so hardcode until HW adds support.
    return 2; // RANGE0/1
}

/*!
 * @return Get the DMEM Priv Range0/1
 */
void
dpuGetDmemPrivRange_v02_05
(
    LwU32  index,
    LwU32 *rangeStart,
    LwU32 *rangeEnd
)
{
    LwU32 reg;

    switch(index)
    {
        case 0:
        {
            reg         = GPU_REG_RD32(LW_PDISP_FALCON_DMEM_PRIV_RANGE0);
            *rangeStart = DRF_VAL(_PDISP_FALCON, _DMEM_PRIV_RANGE0, _START_BLOCK, reg);
            *rangeEnd   = DRF_VAL(_PDISP_FALCON, _DMEM_PRIV_RANGE0, _END_BLOCK, reg);
            break;
        }

        case 1:
        {
            reg         = GPU_REG_RD32(LW_PDISP_FALCON_DMEM_PRIV_RANGE1);
            *rangeStart = DRF_VAL(_PDISP_FALCON, _DMEM_PRIV_RANGE1, _START_BLOCK, reg);
            *rangeEnd   = DRF_VAL(_PDISP_FALCON, _DMEM_PRIV_RANGE1, _END_BLOCK, reg);
            break;
        }

        default:
        {
            *rangeStart = FALCON_DMEM_PRIV_RANGE_ILWALID;
            *rangeEnd   = FALCON_DMEM_PRIV_RANGE_ILWALID;
            dprintf("lw: Invalid index: %d\n", index);
            break;
        }
    }
}

/*!
 * @return LW_TRUE  DMEM range is accessible 
 *         LW_FALSE DMEM range is inaccessible 
 */
LwBool
dpuIsDmemRangeAccessible_v02_05
(
    LwU32 blkLo,
    LwU32 blkHi
)
{
    LwU32  i, numPrivRanges;
    LwU32  rangeStart = FALCON_DMEM_PRIV_RANGE_ILWALID;
    LwU32  rangeEnd   = FALCON_DMEM_PRIV_RANGE_ILWALID;
    LwBool accessAllowed = LW_FALSE;

    numPrivRanges = pDpu[indexGpu].dpuGetDmemNumPrivRanges();

    for (i = 0; i < numPrivRanges; i++)
    {
        pDpu[indexGpu].dpuGetDmemPrivRange(i, &rangeStart, &rangeEnd);

        if (rangeStart >= rangeEnd)
        {
            // invalid range.
            continue;
        }

        if (blkLo >= rangeStart && blkHi <= rangeEnd)
        {
            // We're within range
            accessAllowed = LW_TRUE;
            break;
        }
    }

    if (!accessAllowed)
    {
        // Print out info message
        dprintf("lw: DPU is in LS mode. Requested address range is not within "
                "ranges accessible by CPU.\n");
    }

    return accessAllowed;
}
