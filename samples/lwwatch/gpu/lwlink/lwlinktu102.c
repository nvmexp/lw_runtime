/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2021 by LWPU Corporation.  All rights reserved.  All
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

#include "turing/tu102/dev_top.h"
#include "turing/tu102/dev_hshub.h"
#include "turing/tu102/dev_master.h"
#include "turing/tu102/dev_trim.h"

#include "turing/tu102/dev_lwl_ip.h"
#include "turing/tu102/dev_lwltlc_ip.h"
#include "turing/tu102/dev_ioctrlmif_ip.h"
#include "turing/tu102/dev_ioctrl_ip.h"
#include "turing/tu102/lwlinkip_discovery.h"

#include "turing/tu102/dev_graphics_nobundle.h"
#include "turing/tu102/dev_fb.h"
#include "turing/tu102/dev_ltc.h"
#include "turing/tu102/dev_lwlipt_ip.h"

//
// lwlinkPrintHshubConfig_GV100
//     Print the HSHUB configuration
//     LW_PFB_HSHUB_CONFIG0/1/2/6/7 capture all the configuration
//
void lwlinkPrintHshubConfig_TU102(void)
{
    LwU32 config0, config1, config2, config6, config7, i;

    config0 = GPU_REG_RD32(LW_PFB_HSHUB_CONFIG0);
    config1 = GPU_REG_RD32(LW_PFB_HSHUB_CONFIG1);
    config2 = GPU_REG_RD32(LW_PFB_HSHUB_CONFIG2);
    config6 = GPU_REG_RD32(LW_PFB_HSHUB_CONFIG6);
    config7 = GPU_REG_RD32(LW_PFB_HSHUB_CONFIG7);

    dprintf("\n");

    dprintf("LW_PFB_HSHUB_CONFIG0 = 0x%08X\n", config0);
    dprintf("LW_PFB_HSHUB_CONFIG1 = 0x%08X\n", config1);
    dprintf("LW_PFB_HSHUB_CONFIG2 = 0x%08X\n", config2);
    dprintf("LW_PFB_HSHUB_CONFIG6 = 0x%08X\n", config6);
    dprintf("LW_PFB_HSHUB_CONFIG7 = 0x%08X\n", config7);

    for (i = 0; i < LW_PFB_HSHUB_CONFIG0_SYSMEM_LWLINK_MASK_ARRAY__SIZE_1; i++)
    {
        if (DRF_VAL(_PFB_HSHUB, _CONFIG0, _SYSMEM_LWLINK_MASK_ARRAY(i), config0))
        {
            dprintf("LINK %u is connected to SYSMEM\n", i);
        }
    }
    dprintf("\n");

    for (i = 0; i < LW_PFB_HSHUB_CONFIG0_PEER_PCIE_MASK_ARRAY__SIZE_1; i++)
    {
        if (!DRF_VAL(_PFB_HSHUB, _CONFIG0, _PEER_PCIE_MASK_ARRAY(i), config0))
        {
            dprintf("PEER %u is connected through LWLINK\n", i);
        }
    }
    dprintf("\n");

    for (i = 0; i < LW_PFB_HSHUB_CONFIG1_PEER_LWLINK_MASK__SIZE_1; i++)
    {
        dprintf("Mask of links used by PEER%u = 0x%08X\n", i,
                 DRF_VAL(_PFB_HSHUB, _CONFIG1, _PEER_LWLINK_MASK(i), config1));
    }
    for (i = 0; i < LW_PFB_HSHUB_CONFIG2_PEER_LWLINK_MASK__SIZE_1; i++)
    {
        dprintf("Mask of links used by PEER%u = 0x%08X\n",
                 i + LW_PFB_HSHUB_CONFIG1_PEER_LWLINK_MASK__SIZE_1,
                 DRF_VAL(_PFB_HSHUB, _CONFIG2, _PEER_LWLINK_MASK(i), config2));
    }
    dprintf("\n");
}

// The main LWLink status dump for TU102+
void lwlinkPrintStatus_TU102(LwBool bComplete)
{
    LwU32 regData, linkReset, linkDisable, linkInitDisable;
    LwU32        linksDiscovered  = 0;
    LwU32        ioctrlPriBase    = 0;
    lwlDiscover *device           = NULL;
    lwlDiscover *pLwlDiscoverList = NULL;
    LwU32        linkId;

    // Discover IOCTRL in device info
    ioctrlPriBase = pLwlink[indexGpu].lwlinkIoctrlDiscovery(NULL);
    if (ioctrlPriBase == 0)
    {
        dprintf("Unable to locate LWLink IP in GPU PTOP device table, exiting.\n");
        return;
    }

    //
    // Dump key IOCTRL info/registers in all cases
    //

    dprintf("\n");

    dprintf("LW_IOCTRL base = 0x%08X\n", ioctrlPriBase);

    dprintf("\n");

    regData = GPU_REG_RD32(ioctrlPriBase + LW_IOCTRL_DISABLE);
    dprintf("LW_IOCTRL_DISABLE = 0x%08X\n", regData);
    linkDisable     = DRF_VAL(_IOCTRL, _DISABLE, _LINKDISABLE, regData);
    linkInitDisable = DRF_VAL(_IOCTRL, _DISABLE, _INITDISABLE, regData);
    
    regData = GPU_REG_RD32(ioctrlPriBase + LW_IOCTRL_RESET);
    dprintf("LW_IOCTRL_RESET = 0x%08X\n", regData);
    linkReset = DRF_VAL(_IOCTRL, _RESET, _LINKRESET, regData);
    
    dprintf("\n");
    
    dprintf("Links disabled in HW (LW_IOCTRL_DISABLE_LINKDISABLE):      0x%08X\n", linkDisable);
    dprintf("Links init disabled in HW (LW_IOCTRL_DISABLE_INITDISABLE): 0x%08X\n", linkInitDisable);
    dprintf("Links NOT in reset (LW_IOCTRL_RESET_LINKRESET):            0x%08X\n", linkReset);
    dprintf("\n");
    
    linksDiscovered = pLwlink[indexGpu].lwlinkLinkDiscovery((void *)&pLwlDiscoverList, 0);
    if (linksDiscovered == 0)
    {
        dprintf("No links were detected in ioctrl\n");
        return;
    }

    // Discovery links in the discovery table, satisfied by a DLPL per link
    for (linkId = 0; linkId < 32; linkId++)
    {
        if (lwlinkDeviceInstanceInfoLookUp(pLwlDiscoverList, linkId, LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_DLPL))
        {
            // Include this link
            linksDiscovered |= (1 << linkId);
        }
    }

    dprintf("Links discovered in IOCTRL Discovery Table: 0x%08X\n", linksDiscovered);
    dprintf("\n");

    // Exit if all links are in reset    
    if (!(linksDiscovered & linkReset))
    {
        dprintf("All links are in reset, exiting.\n");
        dprintf("\n");
    }

    dprintf("Dumping state of discovered links NOT in reset: 0x%08X\n", linksDiscovered & linkReset);
    dprintf("\n");
    
    // Dump the link state for all discovered links not in reset
    for (linkId = 0; linkId < 32; linkId++)
    {
        if (!(linksDiscovered & linkReset & (1 << linkId))) continue;

        dprintf("===== LINK %d =====\n", linkId);    
  
        // Dump the traffic type
        dprintf("\n");
        dprintf("Link %d HSHUB traffic type: ", linkId);    
        pLwlink[indexGpu].lwlinkPrintLinkTrafficType(linkId);
        dprintf("\n");

        // Locate the DLPL for this link
        device = lwlinkDeviceInstanceInfoLookUp(pLwlDiscoverList, linkId, LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_DLPL);
        if (device)
        {
            dprintf("Link %d DLPL information\n", linkId);    
            pLwlink[indexGpu].lwlinkPrintDlplState(device->addr);
        }
        else
        {
            dprintf("ERROR: Failed to locate DLPL device information for link %d!!!\n", linkId);
        }
        dprintf("\n");

        // Thats it for status, only continue for complete state.
        if (!bComplete) continue;

        // Locate the TLC for this link
        device = lwlinkDeviceInstanceInfoLookUp(pLwlDiscoverList, linkId, LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_LWLTLC);
        if (device)
        {
            dprintf("Link %d TLC information\n", linkId);    
            pLwlink[indexGpu].lwlinkPrintTlcState(device->addr);
        }
        else
        {
            dprintf("ERROR: Failed to locate TLC device information for link %d!!!\n", linkId);
        }
        dprintf("\n");

        // Locate the MIF for this link
        device = lwlinkDeviceInstanceInfoLookUp(pLwlDiscoverList, linkId, LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_IOCTRLMIF);
        if (device)
        {
            dprintf("Link %d MIF information\n", linkId);    
            pLwlink[indexGpu].lwlinkPrintMifState(device->addr);
        }
        else
        {
            dprintf("ERROR: Failed to locate MIF device information for link %d!!!\n", linkId);
        }
        dprintf("\n");
    }

    freeLwlDiscoveryList(pLwlDiscoverList);
}
