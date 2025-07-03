/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
 
//*****************************************************
//
// DPU Hal Functions
// dpu0201.c
//
//*****************************************************


/* ------------------------ Includes --------------------------------------- */
#include "hal.h"
#include "falcon.h"
#include "dpu.h"
#include "dpu/v02_01/dev_disp_falcon.h"

#include "g_dpu_private.h"     // (rmconfig)  implementation prototypes


/* ------------------------ Defines ---------------------------------------- */
/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Globals ---------------------------------------- */
const static FLCN_ENGINE_IFACES flcnEngineIfaces_dpu0201 =
{    
    dpuGetFalconBase_v02_00,                    // flcnEngGetFalconBase    
};  // falconEngineIfaces_dpu


/* ------------------------ Functions -------------------------------------- */
/*!
 * @return The falcon engine interface FLCN_ENGINE_IFACES*
 */
const FLCN_ENGINE_IFACES *
dpuGetFalconEngineIFace_v02_01()
{
    return &flcnEngineIfaces_dpu0201;
}

