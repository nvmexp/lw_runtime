/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2021 LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// retodd@lwpu.com - 2.1.2002
// hal.h
//
//*****************************************************

#ifndef _HAL_H_
#define _HAL_H_

#include "gr.h"
#include "os.h"

#include "lwwatch-config.h"

#include "g_hal.h"        // (rmconfig) hal interface definitions


typedef enum
{
    LWHAL_IMPL_T124,
    LWHAL_IMPL_T210,
    LWHAL_IMPL_T186,
    LWHAL_IMPL_T194,
    LWHAL_IMPL_T234,
    LWHAL_IMPL_GM107,
    LWHAL_IMPL_GM200,
    LWHAL_IMPL_GM204,
    LWHAL_IMPL_GM206,
    LWHAL_IMPL_GP100,
    LWHAL_IMPL_GP102,
    LWHAL_IMPL_GP104,
    LWHAL_IMPL_GP106,
    LWHAL_IMPL_GP107,
    LWHAL_IMPL_GP108,
    LWHAL_IMPL_GV100,
    LWHAL_IMPL_TU102,
    LWHAL_IMPL_TU104,
    LWHAL_IMPL_TU106,
    LWHAL_IMPL_TU116,
    LWHAL_IMPL_TU117,
    LWHAL_IMPL_GA100,
    LWHAL_IMPL_GA102,
    LWHAL_IMPL_GA103,
    LWHAL_IMPL_GA104,
    LWHAL_IMPL_GA106,
    LWHAL_IMPL_GA107,
    LWHAL_IMPL_AD102,
    LWHAL_IMPL_AD103,
    LWHAL_IMPL_AD104,
    LWHAL_IMPL_AD106,
    LWHAL_IMPL_AD107,
    LWHAL_IMPL_GH100,
    LWHAL_IMPL_GH202,
    LWHAL_IMPL_GB100,
    LWHAL_IMPL_G000,
    LWHAL_IMPL_MAXIMUM,   // NOTE: Normally, this symbol must be at the end of the enum list. 
                          // It is used to allocate arrays and control loop iterations.
} LWHAL_IMPLEMENTATION;

//
// global halObject
//
typedef struct
{
    const LWHAL_IFACE_SETUP *pHal;
    LWHAL_IMPLEMENTATION halImpl;
    LWWATCHCHIPINFOSTRUCT chipInfo;
    PhysAddr instStartAddr;
    LwU32 numGPU;
}Hal;

extern Hal hal;
extern int indexGpu;
//
// prototypes
//
LW_STATUS    initLwWatchHal(LwU32);
char*   getLwhalImplName( LWHAL_IMPLEMENTATION implId );

#endif // _HAL_H_
