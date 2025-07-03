/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020-2021 by LWPU Corporation.  All rights reserved.  All information
 * contained herein is proprietary and confidential to LWPU Corporation.  Any
 * use, reproduction, or disclosure without the written permission of LWPU
 * Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */


//*****************************************************
//
// lwwatch extension
// bifgh100.c
//
//*****************************************************

//
// includes
//
#include "bif.h"
#include "hopper/gh100/dev_xtl_ep_pcfg_gpu.h"
#include "hopper/gh100/dev_xtl_ep_pri.h"
#include "gpuanalyze.h"
#include "print.h"

#include "g_bif_private.h"     // (rmconfig)  implementation prototypes

/*!
* @brief Get the current BIF speed
*
* @param[out]  pSpeed  Current bus speed. Must not be NULL. Will be set to
*                      RM_PMU_BIF_LINK_SPEED_ILWALID in case of an error.
*
* @return      LW_OK on success, otherwise an error code.
*/
LW_STATUS
bifGetBusGenSpeed_GH100
(
    LwU32   *pSpeed
)
{
    LwU32 tempRegVal;

    tempRegVal = GPU_REG_RD32(DEVICE_BASE(LW_EP_PCFGM) + LW_EP_PCFG_GPU_LINK_CONTROL_STATUS);

    switch (DRF_VAL(_EP_PCFG_GPU, _LINK_CONTROL_STATUS, _LWRRENT_LINK_SPEED, tempRegVal))
    {
        case LW_EP_PCFG_GPU_LINK_CONTROL_STATUS_LWRRENT_LINK_SPEED_2P5:
            *pSpeed = RM_PMU_BIF_LINK_SPEED_GEN1PCIE;
            break;
        case LW_EP_PCFG_GPU_LINK_CONTROL_STATUS_LWRRENT_LINK_SPEED_5P0:
            *pSpeed = RM_PMU_BIF_LINK_SPEED_GEN2PCIE;
            break;
        case LW_EP_PCFG_GPU_LINK_CONTROL_STATUS_LWRRENT_LINK_SPEED_8P0:
            *pSpeed = RM_PMU_BIF_LINK_SPEED_GEN3PCIE;
            break;
        case LW_EP_PCFG_GPU_LINK_CONTROL_STATUS_LWRRENT_LINK_SPEED_16P0:
            *pSpeed = RM_PMU_BIF_LINK_SPEED_GEN4PCIE;
            break;
        case LW_EP_PCFG_GPU_LINK_CONTROL_STATUS_LWRRENT_LINK_SPEED_32P0:
            *pSpeed = RM_PMU_BIF_LINK_SPEED_GEN5PCIE;
            break;
        default:
            *pSpeed = RM_PMU_BIF_LINK_SPEED_ILWALID;
            return LW_ERR_GENERIC;
    }

    return LW_OK;
}

void bifGetMsiInfo_GH100(void)
{
    dprintf("lwwatch cannot read config registers in PRI on GH100+\n");
    dprintf("please implement me to read LW_EP_PCFG_GPU_MSI_64_HEADER and LW_EP_PCFG_GPU_MSIX_CAP_HEADER through osPciRead32\n");

}
