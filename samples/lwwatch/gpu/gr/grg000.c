/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// grg000.c
//
//*****************************************************

//
// includes
//
#include "g00x/g000/dev_graphics_nobundle.h"
#include "g00x/g000/hwproject.h"
#include "gr.h"

/*!
 * @brief Provides the caller with information about a particular type of GR Aperture
 *
 * Output parameters may be NULL if that particular aperture information is not
 * required.
 *
 * @param[in]  type        type of the Aperture, GR_UNIT_TYPE* macros defined in grunits.h
 * @param[out] pUnitBase   Base address for the first unit Aperture of its kind
 * @param[out] pUnitStride Stride length for the scalable unit
 * @param[out] pUnitBCIdx  Signed index for a broadcast Aperture, relative to Base
 *
 * @return LW_STATUS LW_OK upon success
 *                   LW_ERR_ILWALID_ARGUMENT for an unknown aperture type for this Arch.
 */
LW_STATUS
grGetUnitApertureInformation_G000
(
    GR_UNIT_TYPE type,
    LwU32       *pUnitBase,
    LwU32       *pUnitStride,
    LwS32       *pUnitBCIdx
)
{
    LwU32 unused;
    pUnitBase   = (pUnitBase   != NULL) ? pUnitBase   : &unused;
    pUnitStride = (pUnitStride != NULL) ? pUnitStride : &unused;
    pUnitBCIdx  = (pUnitBCIdx  != NULL) ? pUnitBCIdx  : (LwS32 *)&unused;

    switch(type)
    {
        case GR_UNIT_TYPE_GR:
            *pUnitBase = LW_PGRAPH_BASE;
            *pUnitStride = DRF_SIZE(LW_PGRAPH);
            *pUnitBCIdx = 0;
            break;

        case GR_UNIT_TYPE_GPC:
            *pUnitBase = LW_GPC_IN_GR_BASE;
            *pUnitStride = LW_GPC_PRI_STRIDE;
            *pUnitBCIdx = LW_GPC_PRI_SHARED_INDEX;
            break;

        case GR_UNIT_TYPE_EGPC:
            *pUnitBase = LW_EGPC_IN_GR_BASE;
            *pUnitStride = LW_EGPC_PRI_STRIDE;
            *pUnitBCIdx = LW_EGPC_PRI_SHARED_INDEX;
            break;

        case GR_UNIT_TYPE_ETPC:
        case GR_UNIT_TYPE_TPC:
            *pUnitBase = LW_TPC_IN_GPC_BASE;
            *pUnitStride = LW_TPC_IN_GPC_STRIDE;
            *pUnitBCIdx = LW_TPC_IN_GPC_PRI_SHARED_INDEX;
            break;

        case GR_UNIT_TYPE_ROP:
            *pUnitBase = LW_ROP_IN_GPC_BASE;
            *pUnitStride = LW_ROP_IN_GPC_STRIDE;
            *pUnitBCIdx = LW_ROP_IN_GPC_PRI_SHARED_INDEX;
            break;

        case GR_UNIT_TYPE_PPC:
            *pUnitBase = LW_PPC_IN_GPC_BASE;
            *pUnitStride = LW_PPC_IN_GPC_STRIDE;
            *pUnitBCIdx = LW_PPC_IN_GPC_PRI_SHARED_INDEX;
            break;

        default:
            return LW_ERR_NOT_SUPPORTED;
    }

    return LW_OK;
}
