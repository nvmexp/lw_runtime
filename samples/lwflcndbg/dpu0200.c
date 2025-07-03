/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// DPU Hal Functions
// dpu0200.c
//
//*****************************************************

/* ------------------------ Includes --------------------------------------- */
#include "hal.h"
#include "os.h"
#include "print.h"
#include "falcon.h"
#include "dpu.h"

#include "disp/v02_00/dev_disp.h"
#include "fermi/gf110/dev_falcon_v4.h"
#include "fermi/gf110/dev_bus.h"
#include "fermi/gf110/dev_master.h"
#include "fermi/gf110/dev_lw_xve.h"
#include "dpu/v02_00/dev_disp_falcon.h"
#include "g_dpu_private.h"     // (rmconfig)  implementation prototypes

/* ------------------------ Defines ---------------------------------------- */
/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Globals ---------------------------------------- */
const static FLCN_ENGINE_IFACES flcnEngineIfaces_dpu0200 =
{
    dpuGetFalconBase_v02_00,                    // flcnEngGetFalconBase
};  // falconEngineIfaces_dpu

/*!
 * @return The falcon engine interface FLCN_ENGINE_IFACES*
 */
const FLCN_ENGINE_IFACES *
dpuGetFalconEngineIFace_v02_00()
{
    return &flcnEngineIfaces_dpu0200;
}

/*!
 * @return The falcon base address of DPU
 */
LwU32
dpuGetFalconBase_v02_00()
{
    return LW_FALCON_DISP_BASE;
}

/*!
 *  Gets falcon engine base, returns  the given registerMap.
 *
 */
void
dpuFlcngdbGetRegMap_v02_00
(
    FLCNGDB_REGISTER_MAP* registerMap
)
{
   const FLCN_ENGINE_IFACES *pFEIF = pDpu[indexGpu].dpuGetFalconEngineIFace();
   flcnGetFlcngdbRegisterMap_v04_00(pFEIF->flcnEngGetFalconBase(), registerMap);
}

/*!
 * @return The maximum number of breakpoints supported
 */
LwU32
dpuFlcngdbMaxBreakpointsGet_v02_00()
{
    return 2;
}
