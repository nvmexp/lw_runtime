/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2009-2018 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/


//
// includes
//

#include "kepler/gk104/dev_falcon_v1.h"
#include "gpuanalyze.h"

#include "falcon.h"
#include "g_falcon_private.h"     // (rmconfig)  implementation prototypes

//-----------------------------------------------------
//falconTestPC_GK104
// test of falcon pc: check if it is valid, if valid,
// check if it is changing
// @param[in] engineBase - base of engine in register space
// @param[in] engName - name of the engine, e.g. "PMSPDEC"
// @param[out] LW_OK/LW_ERR_GENERIC on success/failure
//-----------------------------------------------------
LW_STATUS falconTestPC_v01_00(LwU32 engineBase, char *engName)
{
    LW_STATUS    status = LW_OK;
    LwU32   pc[3];

    pc[0] = GPU_REG_RD32(engineBase + LW_PFALCON_PRI_PC);
    pc[1] = GPU_REG_RD32(engineBase + LW_PFALCON_PRI_PC);
    pc[2] = GPU_REG_RD32(engineBase + LW_PFALCON_PRI_PC);

    if ( !DRF_VAL(_PFALCON, _PRI_PC, _VALID, pc[0]) || 
         !DRF_VAL(_PFALCON, _PRI_PC, _VALID, pc[1]) ||
         !DRF_VAL(_PFALCON, _PRI_PC, _VALID, pc[2]))
    {
        dprintf("lw: LW_%s_PRI_PC_VALID is false: 0x%x\n", 
                engName, DRF_VAL(_PFALCON, _PRI_PC, _VALID, pc[0]));
        addUnitErr("\t LW_%s_PRI_PC_VALID is false: 0x%x\n", 
                engName, DRF_VAL(_PFALCON, _PRI_PC, _VALID, pc[0]));

        status = LW_ERR_GENERIC;
    }

    //if looks like we're getting valid PCs, check if stuck
    if ((DRF_VAL(_PFALCON, _PRI_PC, _REG, pc[0]) == DRF_VAL(_PFALCON, _PRI_PC, _REG, pc[1])) && 
         (DRF_VAL(_PFALCON, _PRI_PC, _REG, pc[0]) == DRF_VAL(_PFALCON, _PRI_PC, _REG, pc[2])))
    {
        dprintf("lw: LW_%s_PRI_PC appears to be stuck at:    0x%04x\n",
                engName, DRF_VAL(_PFALCON, _PRI_PC, _REG, pc[0]));
        addUnitErr("\t LW_%s_PRI_PC appears to be stuck at:    0x%04x\n",
                engName, DRF_VAL(_PFALCON, _PRI_PC, _REG, pc[0]));

        status = LW_ERR_GENERIC;
    }
    else
    {
        dprintf("lw: LW_%s_PRI_PC is changing. Here's a few iterations...\n", engName);
        dprintf("\t0x%04x\t", DRF_VAL(_PFALCON, _PRI_PC, _REG, pc[0]));
        dprintf("0x%04x\t", DRF_VAL(_PFALCON, _PRI_PC, _REG, pc[1]));
        dprintf("0x%04x\t", DRF_VAL(_PFALCON, _PRI_PC, _REG, pc[2]));
        dprintf("0x%04x\t", GPU_REG_RD32(engineBase + LW_PFALCON_PRI_PC) & DRF_MASK(15:0));
        dprintf("0x%04x\t", GPU_REG_RD32(engineBase + LW_PFALCON_PRI_PC) & DRF_MASK(15:0));
        dprintf("0x%04x\n", GPU_REG_RD32(engineBase + LW_PFALCON_PRI_PC) & DRF_MASK(15:0));
    }
    return status;

}



