/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2016 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// falcphys.h
//
//*****************************************************

#ifndef _LWWATCH_FALCPHYS_H_
#define _LWWATCH_FALCPHYS_H_

#include "os.h"
#include "hal.h"
#include "falcon.h"
#include "gpuanalyze.h"

#define LW_FALCON_FBIF_TRANSCFG(i)              (0x00000600+(i)*4)
#define LW_FALCON_FBIF_CTL                      0x00000624
#define LW_PFB_PRI_MMU_PHYS_SELWRE_FALCON_NO    0xF
#define TARGET_BUFFER_SIZE                      256    // 256B
#define CTX_DMA_ID                              0

#define TOTAL_FALCONS                           0x9
#define FECS_FALCON_ID                          0x0
#define PMU_FALCON_ID                           0x1
#define SEC_FALCON_ID                           0x2
#define DPU_FALCON_ID                           0x3
#define HDA_FALCON_ID                           0x4
#define LWDEC_FALCON_ID                         0x5
#define LWENC0_FALCON_ID                        0x6
#define LWENC1_FALCON_ID                        0x7
#define LWENC2_FALCON_ID                        0x8

#define SELWRE_RESET_ADDR_ILWALID               0
#define RESET_PRIV_MASK_ADDR_ILWALID            0

typedef struct def_falcon_prop
{
    char*  name;
    LwU32  regBase;
    LwU32  pmcMask;
    LwU32  selwreFalc;
    LwU32  selwreInitVal;
    LwU32  regSelwreResetAddr;
    LwU32  resetPrivMaskAddr;
} FALCONPROP, *PFALCONPROP;

typedef struct def_falcon_mmu
{
    LwU32 regBase;
    LwU32 fecsInit;
    LwU32 pmuInit;
    LwU32 secInit;
    LwU32 dfalconInit;
    LwU32 afalconInit;
    LwU32 lwdecInit;
    LwU32 lwencInit;
    LwU32 mspppInit;
} FALCMMUPHYS;

void      falcphysDisplayHelp(void);
BOOL      falcphysIsSupported(LwU32 indexGpu);
LW_STATUS falcphysDmaAccessCheck(LwU32 indexGpu);
LwBool    verify(LwU32, LwU64, LwBool, LwBool);
void      physicalDmaAccess(LwU32, LwU64, LwBool, LwBool);
void      falcGetMmuPhysRegConfig(void *pFalcVoid);
void      falcphysGetLwenc2FalcPhysProp(FALCONPROP *);
void      falcphysProgramDmaBase1Reg(FALCONPROP *, LwU32, LwU64);
LwBool    checkEngineIsPresent(LwU32);

#endif // _LWWATCH_FALCPHYS_H_
