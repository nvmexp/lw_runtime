/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2003-2014 by LWPU Corporation.  All rights reserved.  All
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

#include "os.h"

#define MAX_GPUS 10 // if this is changed, please change the code within 
                    // Hals.pm which prints corresponding extern declarations
                    // of engine objects within the g_eng_hal.h files

#include "lwwatch-config.h"

#include "g_hal.h"        // (rmconfig) hal interface definitions


typedef enum
{
    LWHAL_IMPL_T20,
    LWHAL_IMPL_T30,
    LWHAL_IMPL_T114,
    LWHAL_IMPL_T124,
    LWHAL_IMPL_T148,
    LWHAL_IMPL_T210,
    LWHAL_IMPL_LW04,
    LWHAL_IMPL_LW10,
    LWHAL_IMPL_LW11,
    LWHAL_IMPL_LW15,
    LWHAL_IMPL_LW1A,
    LWHAL_IMPL_LW1F,
    LWHAL_IMPL_LW17,
    LWHAL_IMPL_LW18,
    LWHAL_IMPL_LW20,
    LWHAL_IMPL_LW25,
    LWHAL_IMPL_LW28,
    LWHAL_IMPL_LW30,
    LWHAL_IMPL_LW31,
    LWHAL_IMPL_LW34,
    LWHAL_IMPL_LW35,
    LWHAL_IMPL_LW36,
    LWHAL_IMPL_LW40,
    LWHAL_IMPL_LW41,
    LWHAL_IMPL_LW42,
    LWHAL_IMPL_LW43,
    LWHAL_IMPL_LW44,
    LWHAL_IMPL_LW44A,
    LWHAL_IMPL_LW46,
    LWHAL_IMPL_LW47,
    LWHAL_IMPL_LW48,
    LWHAL_IMPL_LW49,
    LWHAL_IMPL_LW4B,
    LWHAL_IMPL_LW4C,
    LWHAL_IMPL_LW4E,
    LWHAL_IMPL_LW63,
    LWHAL_IMPL_LW67,
    LWHAL_IMPL_LW50,
    LWHAL_IMPL_G80,
    LWHAL_IMPL_G82,
    LWHAL_IMPL_G84,
    LWHAL_IMPL_G86,
    LWHAL_IMPL_G92,
    LWHAL_IMPL_G94,
    LWHAL_IMPL_G96,
    LWHAL_IMPL_G98,
    LWHAL_IMPL_GT200,
    LWHAL_IMPL_dGT206,
    LWHAL_IMPL_iGT206,
    LWHAL_IMPL_MCP77,
    LWHAL_IMPL_iGT209,
    LWHAL_IMPL_MCP79,
    LWHAL_IMPL_GT214,
    LWHAL_IMPL_GT215,
    LWHAL_IMPL_GT216,
    LWHAL_IMPL_GT218,
    LWHAL_IMPL_iGT21A,
    LWHAL_IMPL_MCP89,
    LWHAL_IMPL_GF100,
    LWHAL_IMPL_GF100B,
    LWHAL_IMPL_GF104,
    LWHAL_IMPL_GF106,
    LWHAL_IMPL_GF108,
    LWHAL_IMPL_GF110D,
    LWHAL_IMPL_GF110F,
    LWHAL_IMPL_GF110F2,
    LWHAL_IMPL_GF110F3,
    LWHAL_IMPL_GF117,
    LWHAL_IMPL_GF119,
    LWHAL_IMPL_GK104,
    LWHAL_IMPL_GK106,
    LWHAL_IMPL_GK107,
    LWHAL_IMPL_GK107B,
    LWHAL_IMPL_GK110,
    LWHAL_IMPL_GK208,
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
    LWHAL_IMPL_GK20A,
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
    PhysAddr hashTableStartAddr;
    PhysAddr fifoCtxStartAddr;
    U032 numGPU;
}Hal;

extern Hal hal;
extern int indexGpu;
//
// prototypes
// 
LW_STATUS    initLwWatchHal(LwU32);
char*   getLwhalImplName( LWHAL_IMPLEMENTATION implId );

#endif // _HAL_H_
