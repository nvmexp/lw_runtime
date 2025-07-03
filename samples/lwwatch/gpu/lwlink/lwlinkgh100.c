/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#include "lwwatch.h"
#include "lwlink.h"
#include "chip.h"
#include "hal.h"
#include "exts.h"

#include "hopper/gh100/dev_top.h"
#include "hopper/gh100/dev_hshub_SW.h"
#include "hopper/gh100/dev_mmu_hshub_SW.h"
#include "hopper/gh100/dev_graphics_nobundle.h"
#include "hopper/gh100/dev_fb.h"
#include "hopper/gh100/dev_ltc.h"
#include "hopper/gh100/dev_lwldl_ip.h"
#include "hopper/gh100/dev_lwltlc_ip.h"
#include "hopper/gh100/dev_lwlipt_lnk_ip.h"
#include "hopper/gh100/ioctrl_discovery.h"
#include "hopper/gh100/lwlinkip_discovery.h"


/*!
 * @brief Function to get the LWLIPT Base address for a specific link 
 *
 */
LwU32
lwlinkLwliptBaseAddress_GH100(LwU32 link)
{
    //6 links per ioctrl in gh100
    LwU32 localLink = link % 6;
    LwU32 ioctrlNum  = link / 6; 

    LwU32 linkDeviceAddr = LW_DISCOVERY_IOCTRL_UNICAST_0_SW_DEVICE_BASE_LWLIPT_LNK_0 + (localLink * LWLINK_LINK_STRIDE) + (ioctrlNum * LWLINK_IOCTRL_STRIDE);
    return linkDeviceAddr;
}

/*!
 * @brief Function to get the LWLDL Base address for a specific link 
 *
 */
LwU32
lwlinkLwldlBaseAddress_GH100(LwU32 link)
{
    //6 links per ioctrl in gh100
    LwU32 localLink = link % 6;
    LwU32 ioctrlNum  = link / 6; 

    LwU32 linkDeviceAddr = LW_DISCOVERY_IOCTRL_UNICAST_0_SW_DEVICE_BASE_LWLDL_0 + (localLink * LWLINK_LINK_STRIDE) + (ioctrlNum * LWLINK_IOCTRL_STRIDE);
    return linkDeviceAddr;
}

/*!
 * @brief Function to get the LWLTLC Base address for a specific link 
 *
 */ 
LwU32
lwlinkLwltlcBaseAddress_GH100(LwU32 link)
{
    //6 links per ioctrl in gh100
    LwU32 localLink = link % 6;
    LwU32 ioctrlNum  = link / 6;

    LwU32 linkDeviceAddr = LW_DISCOVERY_IOCTRL_UNICAST_0_SW_DEVICE_BASE_LWLTLC_0 + (localLink * LWLINK_LINK_STRIDE) + (ioctrlNum * LWLINK_IOCTRL_STRIDE);
    return linkDeviceAddr;
}

/*!
 * @brief  Function that prints HSHUB configuration
 *         Prints SYS/PEER_CONFIG0
 */
void
lwlinkPrintHshubConfig_GH100(void)
{
    LwU32 sysConfig0;
    LwU32 peerConfig0;
    LwU32 i, j;

    dprintf("\n");

    // Print the SYS_CONFIG0 registers in both HSHUBs
    for (i = 0; i < LW_PFB_HSHUB__SIZE_1; i++)
    {
        sysConfig0 = GPU_REG_RD32(LW_PFB_HSHUB_SYS_CONFIG0(i));
        dprintf("SYS_CONFIG0_LWLINK_MASK of HSHUB %d = 0x%x\n", i, sysConfig0);
    }

    // Print the PEER_CONFIG0 register in both HSHUBs, for each PEER
    for (j = 0; j < LW_PFB_HSHUB_PEER_CONFIG0__SIZE_2; j++)
    {    
        for (i = 0; i < LW_PFB_HSHUB_PEER_CONFIG0__SIZE_1; i++)
        {
            peerConfig0 = GPU_REG_RD32(LW_PFB_HSHUB_PEER_CONFIG0(i,j));
            dprintf("PEER_CONFIG0_LWLINK_MASK of PEER %d of HSHUB %d = 0x%x\n", j, i, peerConfig0);
        }
    }

    // Print mask of SYSMEM links of HSHUB 0 since both HSHUBs have identical information on GA100
    i = 0;
    sysConfig0 = GPU_REG_RD32(LW_PFB_HSHUB_SYS_CONFIG0(i));
    dprintf("Mask of links connected to SYSMEM of HSHUB %d = 0x%x\n", i,
                        DRF_VAL(_PFB_HSHUB, _SYS_CONFIG0, _LWLINK_MASK, sysConfig0));

    // Print mask of links for each PEER
    for (i = 0; i < LW_PFB_HSHUB_PEER_CONFIG0__SIZE_2; i++)
    {
        peerConfig0 = GPU_REG_RD32(LW_PFB_HSHUB_PEER_CONFIG0(0, i));
        dprintf("Mask of links connected to PEER %d = 0x%x\n", i,
                        DRF_VAL(_PFB_HSHUB, _PEER_CONFIG0, _LWLINK_MASK, peerConfig0));
    }
}

/*!
 * @brief Function to print DLPL device state
 *
 */
void lwlinkPrintDlplState_GH100(LwU32 linkId)
{
    LwU32 lwldlBase = pLwlink[indexGpu].lwlinkLwldlBaseAddress(linkId);

    dprintf("LW_LWLDL_TOP_LINK_STATE: 0x%08X ", GPU_REG_RD32(lwldlBase + LW_LWLDL_TOP_LINK_STATE));
    switch (DRF_VAL(_LWLDL, _TOP_LINK_STATE, _STATE, GPU_REG_RD32(lwldlBase + LW_LWLDL_TOP_LINK_STATE)))
    {
        case LW_LWLDL_TOP_LINK_STATE_STATE_INIT:
            dprintf("(INIT)\n");
            break;
        case LW_LWLDL_TOP_LINK_STATE_STATE_HWCFG:
            dprintf("(HWCFG)\n");
            break;
        case LW_LWLDL_TOP_LINK_STATE_STATE_SWCFG:
            dprintf("(SWCFG)\n");
            break;
        case LW_LWLDL_TOP_LINK_STATE_STATE_ACTIVE:
            dprintf("(ACTIVE)\n");
            break;
        case LW_LWLDL_TOP_LINK_STATE_STATE_FAULT:
            dprintf("(FAULT)\n");
            break;
        case LW_LWLDL_TOP_LINK_STATE_STATE_RCVY_AC:
            dprintf("(RCVY_AC)\n");
            break;
        case LW_LWLDL_TOP_LINK_STATE_STATE_RCVY_RX:
            dprintf("(RCVY_RX)\n");
            break;
        case LW_LWLDL_TOP_LINK_STATE_STATE_TRAIN:
            dprintf("(TRAIN)\n");
            break;
        default:
             dprintf("(UNKNOWN)\n");
             break;
    }

    dprintf("LW_LWLDL_TX_SLSM_STATUS_TX: 0x%08X ", GPU_REG_RD32(lwldlBase + LW_LWLDL_TX_SLSM_STATUS_TX));
    switch (DRF_VAL(_LWLDL, _TX_SLSM_STATUS_TX, _PRIMARY_STATE, GPU_REG_RD32(lwldlBase + LW_LWLDL_TX_SLSM_STATUS_TX)))
    {
        case LW_LWLDL_TX_SLSM_STATUS_TX_PRIMARY_STATE_HS:
             dprintf("(HS)\n");
             break;
        case LW_LWLDL_TX_SLSM_STATUS_TX_PRIMARY_STATE_TEST:
             dprintf("(TEST)\n");
             break;
        case LW_LWLDL_TX_SLSM_STATUS_TX_PRIMARY_STATE_TRAIN:
             dprintf("(TRAIN)\n");
             break;
        case LW_LWLDL_TX_SLSM_STATUS_TX_PRIMARY_STATE_SAFE:
             dprintf("(SAFE)\n");
             break;
        case LW_LWLDL_TX_SLSM_STATUS_TX_PRIMARY_STATE_OFF:
             dprintf("(OFF)\n");
             break;
        default:
             dprintf("(UNKNOWN)\n");
             break;
    }

    if (FLD_TEST_DRF(_LWLDL_TX, _SLSM_STATUS_TX, _SUBSTATE, _STABLE, GPU_REG_RD32(lwldlBase + LW_LWLDL_TX_SLSM_STATUS_TX)))
    {
        dprintf("(TX SUBSTATE STABLE)\n");
    }

    dprintf("LW_LWLDL_RX_SLSM_STATUS_RX: 0x%08X ", GPU_REG_RD32(lwldlBase + LW_LWLDL_RX_SLSM_STATUS_RX));
    switch (DRF_VAL(_LWLDL, _RX_SLSM_STATUS_RX, _PRIMARY_STATE, GPU_REG_RD32(lwldlBase + LW_LWLDL_RX_SLSM_STATUS_RX)))
    {
        case LW_LWLDL_RX_SLSM_STATUS_RX_PRIMARY_STATE_HS:
             dprintf("(HS)\n");
             break;
        case LW_LWLDL_RX_SLSM_STATUS_RX_PRIMARY_STATE_TEST:
             dprintf("(TEST)\n");
             break;
        case LW_LWLDL_RX_SLSM_STATUS_RX_PRIMARY_STATE_TRAIN:
             dprintf("(TRAIN)\n");
             break;
        case LW_LWLDL_RX_SLSM_STATUS_RX_PRIMARY_STATE_SAFE:
             dprintf("(SAFE)\n");
             break;
        case LW_LWLDL_RX_SLSM_STATUS_RX_PRIMARY_STATE_OFF:
             dprintf("(OFF)\n");
             break;
        default:
             dprintf("(UNKNOWN)\n");
             break;
    }

    if (FLD_TEST_DRF(_LWLDL_TX, _SLSM_STATUS_TX, _SUBSTATE, _STABLE, GPU_REG_RD32(lwldlBase + LW_LWLDL_RX_SLSM_STATUS_RX)))
    {
        dprintf("(RX SUBSTATE STABLE)\n");
    }
}
