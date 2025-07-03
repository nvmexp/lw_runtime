/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2012-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */
 
//*****************************************************
//
// lwwatch WinDbg Extension for FECS
// fecsgk104.c
//
//*****************************************************

//
// includes
//
#include "fecs.h"
#include "hal.h"
#include "kepler/gk104/dev_graphics_nobundle.h"

#include "g_fecs_private.h"     // (rmconfig)  implementation prototypes

/* ------------------------ Defines ---------------------------------------- */
/* ------------------------ Types definitions ------------------------------ */
/* ------------------------ Globals ---------------------------------------- */
const static FLCN_ENGINE_IFACES flcnEngineIfaces_fecs_GK104 =
{
    fecsGetFalconCoreIFace_GK104,               // flcnEngGetCoreIFace    
    fecsGetFalconBase_GK104,                    // flcnEngGetFalconBase    
    fecsGetEngineName,                          // flcnEngGetEngineName    
};  // falconEngineIfaces_fecs


/* ------------------------ Functions -------------------------------------- */
/*!
 * @return The falcon engine interface FLCN_ENGINE_IFACES*
 */
const FLCN_ENGINE_IFACES *
fecsGetFalconEngineIFace_GK104()
{
    return &flcnEngineIfaces_fecs_GK104;
}

/*!
 * @return The falcon core interface FLCN_CORE_IFACES*
 */
const FLCN_CORE_IFACES *
fecsGetFalconCoreIFace_GK104()
{
    return &flcnCoreIfaces_v04_00;
}

/*!
 * @return The falcon base address of DPU
 */
LwU32
fecsGetFalconBase_GK104()
{
    return LW_PGRAPH_PRI_FECS_FALCON_IRQSSETC;   /* LW_FALCON_FECS_BASE */
}
