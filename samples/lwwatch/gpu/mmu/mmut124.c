/* _lw_COPYRIGHT_BEGIN_
 *
 * Copyright 2013-2018 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _lw_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// mmut124.c
//
//*****************************************************

//
// includes
//
#include "tegrasys.h"
#include "t21x/t210/dev_armc.h"
#include "cheetah/tegra_access.h"

//
// defines
//
#define NUM_PHYS_ADDR_BITS_T124                     34

/*!
 *  Read PDE base address.
 *
 *  @param[in]  asId        Address space ID.
 *  @param[out] pBase       Address of PDE table
 *  
 *  @return LW_ERR_GENERIC on failure, LW_OK on success.
 */
void
mmuGetIommuPdb_T124
(
    LwU32       asId,
    LwU64      *pBase
)
{
    LwU32 oldRegData, newRegData, devIndex = pTegrasys[indexGpu].tegrasysGetDeviceBroadcastIndex(&TegraSysObj[indexGpu], "MC");
    LwU32 ptbData;

    /* TODO: is there any thread-safety issue here? */
    oldRegData = MC_REG_RD32(LW_PMC_SMMU_PTB_ASID, devIndex);
    newRegData = FLD_SET_DRF_NUM(_PMC, _SMMU_PTB_ASID, _LWRRENT_ASID, asId, oldRegData);
    MC_REG_WR32(LW_PMC_SMMU_PTB_ASID, newRegData, devIndex);
    /* TODO: sync write before read? DMB? DSB? BBQ? */
    ptbData = MC_REG_RD32(LW_PMC_SMMU_PTB_DATA, devIndex);
    /* TODO: sync read before write? DMB? DSB? BBQ? */
    MC_REG_WR32(LW_PMC_SMMU_PTB_ASID, oldRegData, devIndex);

    if (pBase)
        *pBase = (LwU64)DRF_VAL(_PMC, _SMMU_PTB_DATA, _ASID_PDE_BASE, ptbData) << 12;
}

/*!
 *  Indicates whether or not a GPU 'physical' address is actually mapped
 *  through the SMMU.  Returns TRUE in this case, and writes the equivalent
 *  SMMU virtual address into *smmuVa, otherwise returns FALSE and leavnes
 *  *smmuVa untouched.
 *
 *  @param[in]  gmmuPa      Physical address from the GPU view
 *  @param[out] smmuVa      SMMU virtual address
 */
BOOL
mmuIsGpuIommuMapped_T124(LwU64 gmmuPa, LwU64 *smmuVa)
{
    if ((LwU32)((gmmuPa) >> NUM_PHYS_ADDR_BITS_T124) & 1)
    {
        if (smmuVa != NULL)
            *smmuVa = gmmuPa & ~(1ull << NUM_PHYS_ADDR_BITS_T124);
        return TRUE;
    }
    return FALSE;
}
