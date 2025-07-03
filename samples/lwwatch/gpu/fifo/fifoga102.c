/* _LWRM_COPYRIGHT_BEGIN_
 *
 *  Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 *  information contained herein is proprietary and confidential to LWPU
 *  Corporation.  Any use, reproduction, or disclosure without the written
 *  permission of LWPU Corporation is prohibited.
 *
 *  _LWRM_COPYRIGHT_END_
 */

#include "ampere/ga102/dev_top.h"
#include "ampere/ga102/dev_runlist.h"
#include "ampere/ga102/dev_pbdma.h"
#include "ampere/ga102/dev_ctrl.h"
#include "ampere/ga102/dev_ram.h"
#include "fb.h"
#include "vmem.h"
#include "vgpu.h"

/**
 * Reads data about esched runlist whose runlist id is @p runlistId.
 *
 * @param in Id of hardware scheduler runlist (further in description runlist).
 * @param out pEngRunlistPtr - base of the runlist pointer.
 * @param out pRunListLenght - length of the runlist.
 * @param out pTgtAperture - target of the runlist.
 * @param out ppRunlistBuffer - allocated buffer in whose memory runlist data will be stored.
 *
 * @note if any of the parameters is NULL, specified parameter will not be initialized.
 */
LW_STATUS
fifoReadRunlistInfo_GA102
(
    LwU32   runlistId,
    LwU64  *pEngRunlistPtr,
    LwU32  *pRunlistLength,
    LwU32  *pTgtAperture,
    LwU32 **ppRunlistBuffer
)
{
    LW_STATUS status;
    LwU32 runlistPriBase;
    LwU32 runlistLength;
    LwU32 tgtAperture;
    LwU32 engRunlistBaseLo;
    LwU32 engRunlistBaseHi;
    LwU32 engRunlistSubmit;
    LwU64 runlistBasePtr;

    status = pFifo[indexGpu].fifoEngineDataXlate(ENGINE_INFO_TYPE_RUNLIST, runlistId,
                                                 ENGINE_INFO_TYPE_RUNLIST_PRI_BASE, &runlistPriBase);
    if (status != LW_OK)
    {
        return status;
    }
    engRunlistBaseLo = GPU_REG_RD32(runlistPriBase + LW_RUNLIST_SUBMIT_BASE_LO);
    engRunlistBaseHi = GPU_REG_RD32(runlistPriBase + LW_RUNLIST_SUBMIT_BASE_HI);
    engRunlistSubmit = GPU_REG_RD32(runlistPriBase + LW_RUNLIST_SUBMIT);

    runlistBasePtr = ((LwU64)DRF_VAL(_RUNLIST, _SUBMIT_BASE_HI, _PTR_HI, engRunlistBaseHi)
                        << DRF_SIZE(LW_RUNLIST_SUBMIT_BASE_LO_PTR_LO))
                     | (LwU64)DRF_VAL(_RUNLIST, _SUBMIT_BASE_LO, _PTR_LO, engRunlistBaseLo);
    runlistBasePtr <<= LW_RUNLIST_SUBMIT_BASE_LO_PTR_ALIGN_SHIFT;
    runlistLength = DRF_VAL(_RUNLIST, _SUBMIT, _LENGTH, engRunlistSubmit);
    tgtAperture = DRF_VAL(_RUNLIST, _SUBMIT_BASE_LO, _TARGET, engRunlistBaseLo);
    if (pEngRunlistPtr)
    {
        *pEngRunlistPtr = runlistBasePtr;
    }
    if (pRunlistLength)
    {
        *pRunlistLength = runlistLength;
    }
    if (pTgtAperture)
    {
        *pTgtAperture = tgtAperture;
    }

    if (NULL != ppRunlistBuffer)
    {
        status = pFifo[indexGpu].fifoAllocateAndFetchRunlist(runlistBasePtr, runlistLength,
                                                             tgtAperture, ppRunlistBuffer);
        if (LW_OK != status)
        {
            if (runlistLength != LW_RUNLIST_SUBMIT_LENGTH_ZERO)
            {
                dprintf("**ERROR: Could not fetch runlist info\n");
                return status;
            }

            *ppRunlistBuffer = NULL;
            return LW_OK;
        }
    }

    return LW_OK;
}
