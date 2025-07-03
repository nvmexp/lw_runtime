/* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2021 by LWPU Corporation.  All rights reserved.  All information
* contained herein is proprietary and confidential to LWPU Corporation.  Any
* use, reproduction, or disclosure without the written permission of LWPU
* Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

//-----------------------------------------------------
//
// lwdec0403.c - LWDEC 4.3 routines
//
//-----------------------------------------------------

#include "lwdec.h"
#include "chip.h"
#include "g_lwdec_private.h"     // (rmconfig)  implementation prototypes
#include "class/clb8b0.h"
#include "lwdec0403.h"
#include "hopper/gh100/dev_pri_ringstation_sys.h"
#include "hopper/gh100/dev_pri_ringstation_sysb.h"
#include "hopper/gh100/dev_pri_ringstation_sysc.h"

#include "hopper/gh100/dev_lwdec_pri.h"
#include "hopper/gh100/dev_falcon_v4.h"
#include "hopper/gh100/dev_fifo.h"
#include "hopper/gh100/dev_master.h"

#define FALCON_LWDEC_BASE(id)  (LW_FALCON_LWDEC0_BASE + (id * 0x4000))
//-----------------------------------------------------
// lwdecIsValidEngineId_v04_03
//-----------------------------------------------------
BOOL lwdecIsValidEngineId_v04_03(LwU32 indexGpu, LwU32 engineId)
{
    switch (engineId)
    {
    case LWWATCH_LWDEC_0:
    case LWWATCH_LWDEC_1:
    case LWWATCH_LWDEC_2:
    case LWWATCH_LWDEC_3:
    case LWWATCH_LWDEC_4:
    case LWWATCH_LWDEC_5:
    case LWWATCH_LWDEC_6:
    case LWWATCH_LWDEC_7:
        break;
    default:
        dprintf("GH100 supports upto 8 lwdec instances only\n");
        return FALSE;
    }

    return TRUE;
}

//-----------------------------------------------------
// lwdecIsSupported_v04_03
//-----------------------------------------------------
BOOL lwdecIsSupported_v04_03(LwU32 indexGpu, LwU32 engineId)
{
    if (!pLwdec[indexGpu].lwdecIsValidEngineId(indexGpu, engineId))
        return FALSE;

    switch (engineId)
    {
    case LWWATCH_LWDEC_0:
        pLwdecPrivReg[LWWATCH_LWDEC_0] = lwdecPrivReg_v04_03_eng0;
        break;
    case LWWATCH_LWDEC_1:
        pLwdecPrivReg[LWWATCH_LWDEC_1] = lwdecPrivReg_v04_03_eng1;
        break;
    case LWWATCH_LWDEC_2:
        pLwdecPrivReg[LWWATCH_LWDEC_2] = lwdecPrivReg_v04_03_eng2;
        break;
    case LWWATCH_LWDEC_3:
        pLwdecPrivReg[LWWATCH_LWDEC_3] = lwdecPrivReg_v04_03_eng3;
        break;
    case LWWATCH_LWDEC_4:
        pLwdecPrivReg[LWWATCH_LWDEC_4] = lwdecPrivReg_v04_03_eng4;
        break;
    case LWWATCH_LWDEC_5:
        pLwdecPrivReg[LWWATCH_LWDEC_5] = lwdecPrivReg_v04_03_eng5;
        break;
    case LWWATCH_LWDEC_6:
        pLwdecPrivReg[LWWATCH_LWDEC_6] = lwdecPrivReg_v04_03_eng6;
        break;
    case LWWATCH_LWDEC_7:
        pLwdecPrivReg[LWWATCH_LWDEC_7] = lwdecPrivReg_v04_03_eng7;
        break;
    default:
        dprintf("GH100 has 8 LWDEC instances (0-7).\n");
        return FALSE;
        break;
    }

    pLwdecFuseReg = lwdecFuseReg_v04_03;
    pLwdecMethodTable = lwdecMethodTable_v04_03;

    return TRUE;
}

//-----------------------------------------------------
// lwdecIsPrivBlocked_v04_03
//-----------------------------------------------------
BOOL lwdecIsPrivBlocked_v04_03(LwU32 indexGpu, LwU32 engineId)
{
    LwU32 idx;
    LwU32 bitmask;
    LwU32 regSysPrivFsConfig;
    LwU32 privWarnReadDisable  = 0;
    LwU32 privWarnWriteDisable = 0;

    if (!pLwdec[indexGpu].lwdecIsValidEngineId(indexGpu, engineId))
        return TRUE;

    // Bit-fields within LW_PPRIV_SYS_PRIV_FS_CONFIG denote priv access for video.
    // All video engines must have priv access for lwdec command support.
    switch (engineId)
    {
    case LWWATCH_LWDEC_0:
        idx = LW_PPRIV_SYSC_PRI_MASTER_fecs2lwdec_pri0 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYSC_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYSC_PRI_MASTER_fecs2lwdec_pri0 - (idx << 5));
        break;
    case LWWATCH_LWDEC_1:
        idx = LW_PPRIV_SYSC_PRI_MASTER_fecs2lwdec_pri1 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYSC_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYSC_PRI_MASTER_fecs2lwdec_pri1 - (idx << 5));
        break;
    case LWWATCH_LWDEC_2:
        idx = LW_PPRIV_SYSC_PRI_MASTER_fecs2lwdec_pri2 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYSC_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYSC_PRI_MASTER_fecs2lwdec_pri2 - (idx << 5));
        break;
    case LWWATCH_LWDEC_3:
        idx = LW_PPRIV_SYSB_PRI_MASTER_fecs2lwdec_pri3 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYSB_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYSB_PRI_MASTER_fecs2lwdec_pri3 - (idx << 5));
        break;
    case LWWATCH_LWDEC_4:
        idx = LW_PPRIV_SYSB_PRI_MASTER_fecs2lwdec_pri4 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYSB_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYSB_PRI_MASTER_fecs2lwdec_pri4 - (idx << 5));
    case LWWATCH_LWDEC_5:
        idx = LW_PPRIV_SYSB_PRI_MASTER_fecs2lwdec_pri5 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYSB_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYSB_PRI_MASTER_fecs2lwdec_pri5 - (idx << 5));
        break;
    case LWWATCH_LWDEC_6:
        idx = LW_PPRIV_SYSB_PRI_MASTER_fecs2lwdec_pri6 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYSB_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYSB_PRI_MASTER_fecs2lwdec_pri6 - (idx << 5));
        break;
    case LWWATCH_LWDEC_7:
        idx = LW_PPRIV_SYSB_PRI_MASTER_fecs2lwdec_pri7 >> 5;
        regSysPrivFsConfig = GPU_REG_RD32(LW_PPRIV_SYSB_PRIV_FS_CONFIG(idx));
        bitmask = BIT(LW_PPRIV_SYSB_PRI_MASTER_fecs2lwdec_pri7 - (idx << 5));
        break;
    default:
        return TRUE;
    }

    if ((regSysPrivFsConfig & bitmask) != bitmask)
    {
        return TRUE;
    }
    else
    {
        switch (engineId)
        {
        case LWWATCH_LWDEC_0:
            privWarnReadDisable  = (GPU_REG_RD32(LW_FUSE_OPT_LWDEC_PRIV_READ_DIS)  & 0x1);
            privWarnWriteDisable = (GPU_REG_RD32(LW_FUSE_OPT_LWDEC_PRIV_WRITE_DIS) & 0x1); 
            break;
        case LWWATCH_LWDEC_1:
        case LWWATCH_LWDEC_2:
        case LWWATCH_LWDEC_3:
        case LWWATCH_LWDEC_4:
        case LWWATCH_LWDEC_5:
        case LWWATCH_LWDEC_6:
        case LWWATCH_LWDEC_7:
            privWarnReadDisable  = (GPU_REG_RD32(LW_FUSE_OPT_LWDEC_PRIV_READ_DIS)  & 0x2);
            privWarnWriteDisable = (GPU_REG_RD32(LW_FUSE_OPT_LWDEC_PRIV_WRITE_DIS) & 0x2);
            break;
        default:
            return TRUE;
        }
        if (privWarnReadDisable)
        {
            dprintf("WARNING: LWDEC%d: Fixed function HW unit register's priv READ is disabled by fuse, register reads all zeros, only FALCON, RISCV, FBIF, CG, PMM registers are readable depending on PLM settings\n",
                    engineId);
        }
        if (privWarnWriteDisable)
        {
            dprintf("WARNING: LWDEC%d: Fixed function HW unit register's priv WRITE is disabled by fuse, register writes have no effect, only FALCON, RISCV, FBIF, CG, PMM registers are writeable depending on PLM settings\n",
                    engineId); 
        }
    }

    return FALSE;
}

//-----------------------------------------------------
// lwdecGetClassId_v04_03
//-----------------------------------------------------
LwU32
lwdecGetClassId_v04_03(void)
{
    return LWB8B0_VIDEO_DECODER;
}

//----------------------------------------------------------
// lwdecDumpFuse_v04_03 - Dumps LWDEC related fuse registers
//----------------------------------------------------------
LW_STATUS lwdecDumpFuse_v04_03(LwU32 indexGpu)
{
    LwU32 u;

    if (!pLwdecFuseReg)
    {
        dprintf("lw: -- Gpu %u LWDEC error: fuse reg array uninitialized\n", indexGpu);
        return LW_ERR_ILWALID_PARAMETER;
    }

    dprintf("lw:\n");
    dprintf("lw: -- Gpu %u LWDEC related fuse registers -- \n", indexGpu);
    dprintf("lw:\n");

    for(u=0; ; u++)
    {
        if(pLwdecFuseReg[u].m_id == 0)
        {
            break;
        }
        pLwdec[indexGpu].lwdecPrintPriv(61, pLwdecFuseReg[u].m_tag, pLwdecFuseReg[u].m_id);
    }
    return LW_OK; 
}
