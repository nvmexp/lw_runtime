/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2011-2014 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#include <string.h>

#include "kepler/gk104/dev_top.h"
#include "kepler/gk104/dev_ce_pri.h"
#include "gpuanalyze.h"

#include "ce.h"
#include "hal.h"
#include "hwref/lwutil.h"
#include "gpuanalyze.h"
#include "methodParse.h"

#include "kepler/kepler_ce.h"

#include "g_ce_hal.h"

//-----------------------------------------------------
// ceIsValid_GK104
//-----------------------------------------------------
BOOL ceIsValid_GK104( LwU32 indexCe )
{
    return CE_IS_VALID(indexCe);
}


//-----------------------------------------------------
// ceIsSupported_GK104
//-----------------------------------------------------
BOOL ceIsSupported_GK104( LwU32 indexGpu, LwU32 indexCe )
{
    LwU32 numCes;

    if (!pCe[indexGpu].ceIsValid(indexCe))
    {
        return FALSE;
    }
    
    numCes = GPU_REG_RD_DRF(_PTOP, _SCAL_NUM_CES, _VALUE);
    if (indexCe >= numCes)
    {
        return FALSE;
    }

    return pCe[indexGpu].ceIsPresent(indexCe);
}


//-----------------------------------------------------
// ceTestState_GK104 - Test basic ce state
//-----------------------------------------------------
LW_STATUS ceTestState_GK104( LwU32 indexGpu, LwU32 indexCe )
{
    return pCe[indexGpu].ceTestCeState(indexGpu, indexCe);
}


/*!
 *  
 *  check Ce engine status
 *  @param[in]      indexGpu
 *  @param[in]      CE base offset
 *  
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
LW_STATUS ceCheckCeState_GK104( LwU32 indexGpu, LwU32 eng )
{
    LW_STATUS    status = LW_OK;
    char    engName[8];
    LwU32   base;
    LwU32   data32;
    LwU32   regIntr;
    LwU32   regIntrEn;
    
    if (!pCe[indexGpu].ceIsValid(eng))
    {
        dprintf("lw:  Improper engine index\n");
        return LW_ERR_GENERIC;
    }

    //set the CE engine #
    sprintf( engName, "PCE%d", eng);

    base = LW_PCE_CE_BASE(eng);

    //check COP interrupts
    regIntr = GPU_REG_RD32( LW_PCE_COP2_INTR_STATUS + base );
    regIntrEn = GPU_REG_RD32( LW_PCE_COP2_INTR_EN + base );
    regIntr &= regIntrEn;

    if ( DRF_VAL(_PCE, _COP2_INTR_STATUS, _BLOCKPIPE, regIntr) )
    {
        dprintf("lw: + LW_%s_COP2_INTR_STATUS_BLOCKPIPE_ACTIVE\n", engName);
        addUnitErr("\t LW_%s_COP2_INTR_STATUS_BLOCKPIPE_ACTIVE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_INTR_STATUS, _NONBLOCKPIPE, regIntr) )
    {
        dprintf("lw: + LW_%s_COP2_INTR_STATUS_NONBLOCKPIPE_ACTIVE\n", engName);
        addUnitErr("\t LW_%s_COP2_INTR_STATUS_NONBLOCKPIPE_ACTIVE\n", engName);
        status = LW_ERR_GENERIC;
    }

    //check COP status
    data32 = GPU_REG_RD32( LW_PCE_COP2_PIPESTATUS + base );
    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _CTL, data32) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_CTL_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_CTL_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _GSTRIP, data32) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_GSTRIP_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_GSTRIP_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _RALIGN, data32) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_RALIGN_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_RALIGN_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _SWIZ, data32) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_SWIZ_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_SWIZ_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }
    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _WALIGN, data32) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_WALIGN_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_WALIGN_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _GPAD, data32) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_GPAD_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_GPAD_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _RDAT, data32) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_RDAT_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_RDAT_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _RDACK, data32) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_RDACK_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_RDACK_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _WRACK, data32) )
    {
        dprintf("lw: LW_%s_COP2_PIPESTATUS_WRACK_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_WRACK_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _RDAT, data32) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_RDAT_NONIDLE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_RDAT_NONIDLE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _BLOCKINTRPT, data32) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_BLOCKINTRPT_ACTIVE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_BLOCKINTRPT_ACTIVE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _RCMD_STALL, data32) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_RCMD_STALL_ACTIVE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_RCMD_STALL_ACTIVE\n", engName);
        status = LW_ERR_GENERIC;
    }

    if ( DRF_VAL(_PCE, _COP2_PIPESTATUS, _WCMD_STALL, data32) )
    {
        dprintf("lw: + LW_%s_COP2_PIPESTATUS_WCMD_STALL_ACTIVE\n", engName);
        addUnitErr("\t LW_%s_COP2_PIPESTATUS_WCMD_STALL_ACTIVE\n", engName);
        status = LW_ERR_GENERIC;
    }

    return status;
}


LW_STATUS ceTestCeState_GK104( LwU32 indexGpu, LwU32 indexCe )
{
    LW_STATUS status = LW_OK;

    // If the caller asks for an invalid engine, they 
    // really want all of the engines.
    if (pCe[indexGpu].ceIsValid(indexCe))
    {
        if (pCe[indexGpu].ceIsSupported(indexGpu, indexCe))
        {
            status = pCe[indexGpu].ceCheckCeState(indexGpu, indexCe);
        
            if (status == LW_ERR_GENERIC)
            {
                dprintf("lw: ******** CE%d state test FAILED ********\n", indexCe);
            }
            else
            {
                dprintf("lw: ******** CE%d state test succeeded ********\n", indexCe);
            } 
        }
    }
    else
    {
        LwU32 i = 0;

        while (pCe[indexGpu].ceIsValid(i))
        {
            if (pCe[indexGpu].ceIsSupported(indexGpu, i))
            {
                dprintf("\n\tlw: ******** Testing CE%d state... ********\n", i);
        
                status = pCe[indexGpu].ceCheckCeState(indexGpu, i);
        
                if (status == LW_ERR_GENERIC)
                {
                    dprintf("lw: ******** CE%d state test FAILED ********\n", i);
                }   
                else    
                {
                    dprintf("lw: ******** CE%d state test succeeded ********\n", i);
                }
            }
            i++;
        }    
    }

   return status;
}



//-----------------------------------------------------
// ceGetPrivs_GK104 - Returns the CE priv reg space
//-----------------------------------------------------
void *ceGetPrivs_GK104( void )
{
    static dbg_ce cePrivReg[] =
    {
        privInfo_ce(LW_PCE_FAKE_COP_MASK),
        privInfo_ce(LW_PCE_COP_CMD0),
        privInfo_ce(LW_PCE_COP_LINES_TO_COPY),
        privInfo_ce(LW_PCE_COP_SRC_PARAM0),
        privInfo_ce(LW_PCE_COP_DST_PARAM0),
        privInfo_ce(LW_PCE_COP_SRC_PARAM1),
        privInfo_ce(LW_PCE_COP_DST_PARAM1),
        privInfo_ce(LW_PCE_COP_SRC_PARAM2),
        privInfo_ce(LW_PCE_COP_DST_PARAM2),
        privInfo_ce(LW_PCE_COP_SRC_PARAM3),
        privInfo_ce(LW_PCE_COP_DST_PARAM3),
        privInfo_ce(LW_PCE_COP_SRC_PARAM4),
        privInfo_ce(LW_PCE_COP_DST_PARAM4),
        privInfo_ce(LW_PCE_COP_BYTES_PER_SWIZ),
        privInfo_ce(LW_PCE_COP_SWIZZLE_0),
        privInfo_ce(LW_PCE_COP_SWIZZLE_1),
        privInfo_ce(LW_PCE_COP_SWIZZLE_2),
        privInfo_ce(LW_PCE_COP_SWIZZLE_3),
        privInfo_ce(LW_PCE_COP_SWIZZLE_CONSTANT0),
        privInfo_ce(LW_PCE_COP_SWIZZLE_CONSTANT1),
        privInfo_ce(LW_PCE_COP_CMD1),
        privInfo_ce(LW_PCE_COP_BIND),
        privInfo_ce(LW_PCE_COP_SRC_PHYS_MODE),
        privInfo_ce(LW_PCE_COP_DST_PHYS_MODE),
        privInfo_ce(LW_PCE_COP2_PIPESTATUS),
        privInfo_ce(LW_PCE_COP2_INTR_EN),
        privInfo_ce(LW_PCE_COP2_INTR_STATUS),
        privInfo_ce(LW_PCE_PMM),
        privInfo_ce(LW_PCE_ENGCAP),
        privInfo_ce(LW_PCE_FE_BORROW),
        privInfo_ce(LW_PCE_FE_INJECT_DMA),
        privInfo_ce(LW_PCE_FE_INJECT_BIND),
        privInfo_ce(LW_PCE_FE_BIND_STATUS),
        privInfo_ce(LW_PCE_FE_IRQ),
        privInfo_ce(LW_PCE_FE_LAUNCHERR),
        privInfo_ce(LW_PCE_FE_ENGCTL),
        privInfo_ce(LW_PCE_FE_THROTTLE),
        privInfo_ce(0)
    };

    return (void *)cePrivReg;
}

//-----------------------------------------------------
// ceDumpPriv_GK104 - Dumps CE priv reg space
//-----------------------------------------------------
LW_STATUS ceDumpPriv_GK104( LwU32 indexGpu, LwU32 indexCe )
{
    LwU32 u; char m_tag[256];

    dbg_ce *cePrivReg = pCe[indexGpu].ceGetPrivs();

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u CE%d priv registers -- \n", indexGpu, indexCe);
    dprintf("lw:\n");

    for(u=0;;u++)
    {
        if(cePrivReg[u].m_id==0)
        {
            break;
        }

        // Build up the per-ce manual name.
        sprintf(m_tag,"LW_PCE%d%s",indexCe, cePrivReg[u].m_tag+strlen("LW_PCE"));        
        
        cePrintPriv(40,m_tag, LW_PCE_CE_BASE(indexCe)+cePrivReg[u].m_id);
    }
    dprintf("lw:\n");
    return LW_OK; 
}


//-----------------------------------------------------
// ceIsPresent_GK104
//-----------------------------------------------------
BOOL ceIsPresent_GK104( LwU32 indexCe )
{
    LwU32 fsstatus = GPU_REG_RD32(LW_PTOP_FS_STATUS);
    return (DRF_IDX_VAL(_PTOP, _FS_STATUS, _CE_IDX, indexCe, fsstatus) ==
                         LW_PTOP_FS_STATUS_CE_IDX_ENABLE);
}
