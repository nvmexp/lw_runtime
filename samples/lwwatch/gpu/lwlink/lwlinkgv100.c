/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2017-2020 by LWPU Corporation.  All rights reserved.  All
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

#include "volta/gv100/dev_top.h"
#include "volta/gv100/dev_hshub.h"
#include "volta/gv100/dev_master.h"
#include "volta/gv100/dev_trim.h"

#include "volta/gv100/dev_lwl_ip.h"
#include "volta/gv100/dev_lwltlc_ip.h"
#include "volta/gv100/dev_ioctrlmif_ip.h"
#include "volta/gv100/dev_ioctrl_ip.h"
#include "volta/gv100/lwlinkip_discovery.h"

//
// lwlinkPrintHshubConfig_GV100
//     Print the HSHUB configuration
//     LW_PFB_HSHUB_CONFIG0/1/2/6/7 capture all the configuration
//
void lwlinkPrintHshubConfig_GV100(void)
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

//
// lwlinkLogHshubErrors_GV100
//
void lwlinkLogHshubErrors_GV100(void)
{
    dprintf("LW_PFB_HSHUB_IG_ERROR = 0x%08X\n", GPU_REG_RD32(LW_PFB_HSHUB_IG_ERROR));
    dprintf("LW_PFB_HSHUB_EG_ERROR = 0x%08X\n", GPU_REG_RD32(LW_PFB_HSHUB_EG_ERROR));
}

//
// lwlinkPrintHshubReqTimeoutInfo_GV100
//
void lwlinkPrintHshubReqTimeoutInfo_GV100(void)
{
    dprintf("Command not lwrrently supported on this GPU (Support is TBD).\n");
}

//
// lwlinkEnableHshubLogging_GV100
//
void lwlinkEnableHshubLogging_GV100(void)
{
    dprintf("Command not lwrrently supported on this GPU (Support is TBD).\n");
}

//
// lwlinkPrintHshubReadTimeoutInfo_GV100
//
void lwlinkPrintHshubReadTimeoutInfo_GV100(void)
{
    dprintf("Command not lwrrently supported on this GPU (Support is TBD).\n");
}

//
// Walks the ioctrl registers to discover lwlink devices.
// This function creates a very simple singly linked list to store device info
//
LwU32 lwlinkLinkDiscovery_GV100(void **pLwlVoid, LwU32 addr)
{
    LwU32 i;
    LwU32 ioctrlDiscoverySize  = 0;
    LwU32 ioctrlInfoType       = DEVICE_INFO_TYPE_ILWALID;
    BOOL  bIoctrlNextDataValid = FALSE;
    LwU32 ioctrlId             = 0;
    LwU32 foundDevices         = 0;
    LwU32 foundIds             = 0;
    LwU32 ioctrlEntry          = 0;
    LwU32 ioctrlEntryAddr      = 0;
    LwU32 ioctrlPriBase        = 0;

    lwlDiscover **pLwlDiscoverListHead = (lwlDiscover **)pLwlVoid;

    // Discover IOCTRL in device info
    ioctrlPriBase = pLwlink[indexGpu].lwlinkIoctrlDiscovery(NULL);

    if (ioctrlPriBase == 0)
    {
        dprintf("No ioctrls were discovered in PTOP_DEVICE_INFO!\n");
        return foundDevices;
    }

    // priBaseAddr + 0 * 4
    ioctrlEntryAddr = ioctrlPriBase;
    ioctrlEntry     = GPU_REG_RD32(ioctrlEntryAddr);
    ioctrlInfoType  = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DEVICE, ioctrlEntry);

    if (ioctrlInfoType == LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_IOCTRL)
    {
        // priBaseAddr + 1 * 4
        ioctrlEntryAddr     = ioctrlPriBase + 4;
        ioctrlEntry         = GPU_REG_RD32(ioctrlEntryAddr);
        ioctrlDiscoverySize = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _IOCTRL_LENGTH,
                                        ioctrlEntry);
    }
    else
    {
        dprintf("Error: first entry in IOCTRL_DISCOVERY is not IOCTRL!\n");
        return foundDevices;
    }

    for (i = 2; i < ioctrlDiscoverySize; ++i)
    {
        ioctrlEntryAddr = ioctrlPriBase + i * 4;
        ioctrlEntry     = GPU_REG_RD32(ioctrlEntryAddr);

        if (FLD_TEST_DRF(_LWLINKIP, _DISCOVERY_COMMON, _ENTRY, _ILWALID, ioctrlEntry))
        {
            dprintf("found invalid ioctrl\n");
            continue;
        }

        switch (DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _ENTRY, ioctrlEntry))
        {
            case LW_LWLINKIP_DISCOVERY_COMMON_ENTRY_DATA1:
            {
                if (bIoctrlNextDataValid)
                {
                    lwlDiscover *pTemp = malloc(sizeof(lwlDiscover));
                    pTemp->ID          = ioctrlId;
                    pTemp->addr        = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _PRI_BASE, ioctrlEntry) <<
                                         LW_LWLINKIP_DISCOVERY_COMMON_PRI_BASE_ALIGN;
                    pTemp->deviceType  = ioctrlInfoType;
                    pTemp->next        = NULL;

                    if (*pLwlDiscoverListHead == NULL)
                    {
                        *pLwlDiscoverListHead = pTemp;
                    }
                    else
                    {
                        lwlDiscover *pIterator = *pLwlDiscoverListHead;
                        while (pIterator->next != NULL)
                        {
                            pIterator = pIterator->next;
                        }
                        pIterator->next = pTemp;
                    }

                    bIoctrlNextDataValid = FALSE;
                }
                break;
            }

            case LW_LWLINKIP_DISCOVERY_COMMON_ENTRY_DATA2:
            {
                if (bIoctrlNextDataValid)
                {
                    // Save any PLL control related state
                    if (DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DLPL_DATA2_TYPE, ioctrlEntry) ==
                        LW_LWLINKIP_DISCOVERY_COMMON_DLPL_DATA2_TYPE_PLLCONTROL)
                    {
                        // dprintf("found data2. Support for PLLs needs to be added.\n");
                    }

                    if (FLD_TEST_DRF(_LWLINKIP, _DISCOVERY_COMMON, _CHAIN, _DISABLE, ioctrlEntry))
                    {
                        bIoctrlNextDataValid = FALSE;
                    }
                }
                break;
            }

            case LW_LWLINKIP_DISCOVERY_COMMON_ENTRY_ENUM:
            {
                ioctrlInfoType = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DEVICE, ioctrlEntry);
                ioctrlId       = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _ID, ioctrlEntry);

                if (DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _CHAIN, ioctrlEntry) ==
                        LW_LWLINKIP_DISCOVERY_COMMON_CHAIN_ENABLE)
                {
                    bIoctrlNextDataValid = TRUE;
                }
                else
                {
                    dprintf("Warning: Found TLC or DLPL in ioctrl, but there is"
                            "no associated data entry. This ioctrl device will be ignored\n");
                    continue;
                }

                if ((foundIds >> ioctrlId) & 1)
                {
                    foundDevices |= (1 << ioctrlId);
                }
                else
                {
                    foundIds |= (1 << ioctrlId);
                }
                break;
            }

            default:
            {
                dprintf("found unknown ioctrl entry\n");
                break;
            }
        }
    }

    return foundDevices;
}

//
// Program the lwlink TL counters
//
void lwlinkProgramTlCounters_GV100(LwS32 linkId)
{
    dprintf("Command not lwrrently supported on this GPU (Need to update for new registers).\n");
}

//
// Reset the lwlink TL counters
//
void lwlinkResetTlCounters_GV100(LwS32 linkId)
{
    dprintf("Command not lwrrently supported on this GPU (Need to update for new registers).\n");
}

//
// Read the lwlink TL counters
//
void lwlinkReadTlCounters_GV100(LwS32 linkId)
{
    dprintf("Command not lwrrently supported on this GPU (Need to update for new registers).\n");
}

//
// lwlinkPrintVerbose_GV100
//
void lwlinkPrintVerbose_GV100(BOOL bPretty)
{
    dprintf("Command not lwrrently supported on this GPU (Need to update for new registers).\n");
}

//
// lwlinkPrintVerbose_GV100
//
void lwlinkDumpUPhy_GV100(void)
{
    dprintf("Command not lwrrently supported on this GPU (UPHY not readable).\n");
}

//
// lwlinkPrintHelp_GV100
//
void lwlinkPrintHelp_GV100(void)
{
    dprintf("usage: lwv lwlink -status\n");
    dprintf("       Prints a top level status (link states, traffic types, etc.)\n");
    dprintf("       lwv lwlink -state\n");
    dprintf("       Prints complete lwlink status (dumps important link registers, traffic types, etc.)\n");
}

// The main LWLink status dump for GV100+
void lwlinkPrintStatus_GV100(LwBool bComplete)
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

// Print the requested link's traffic type
void lwlinkPrintLinkTrafficType_GV100(LwU32 linkId)
{
    LwU32 hsHubConfig0 = GPU_REG_RD32(LW_PFB_HSHUB_CONFIG0);
    LwU32 hsHubConfig1 = GPU_REG_RD32(LW_PFB_HSHUB_CONFIG1);
    LwU32 hsHubConfig2 = GPU_REG_RD32(LW_PFB_HSHUB_CONFIG2);

    LwBool  bNone = LW_TRUE;

    if (DRF_VAL(_PFB_HSHUB_CONFIG0_SYSMEM, _LWLINK, _MASK, hsHubConfig0) & (1 << linkId))
    {
        dprintf("SYSMEM ");
        bNone = LW_FALSE;    
    }

    if (((DRF_VAL(_PFB_HSHUB_CONFIG0_PEER,   _PCIE,   _MASK, hsHubConfig0) & (1 << 0)) == 0) &&
          DRF_VAL(_PFB_HSHUB_CONFIG1_PEER_0, _LWLINK, _MASK, hsHubConfig1) & (1 << linkId))
    {
        dprintf("PEER0 ");    
        bNone = LW_FALSE;    
    }

    if (((DRF_VAL(_PFB_HSHUB_CONFIG0_PEER,   _PCIE,   _MASK, hsHubConfig0) & (1 << 1)) == 0) &&
          DRF_VAL(_PFB_HSHUB_CONFIG1_PEER_1, _LWLINK, _MASK, hsHubConfig1) & (1 << linkId))
    {
        dprintf("PEER1 ");    
        bNone = LW_FALSE;    
    }

    if (((DRF_VAL(_PFB_HSHUB_CONFIG0_PEER,   _PCIE,   _MASK, hsHubConfig0) & (1 << 2)) == 0) &&
          DRF_VAL(_PFB_HSHUB_CONFIG1_PEER_2, _LWLINK, _MASK, hsHubConfig1) & (1 << linkId))
    {
        dprintf("PEER2 ");    
        bNone = LW_FALSE;    
    }

    if (((DRF_VAL(_PFB_HSHUB_CONFIG0_PEER,   _PCIE,   _MASK, hsHubConfig0) & (1 << 3)) == 0) &&
          DRF_VAL(_PFB_HSHUB_CONFIG1_PEER_3, _LWLINK, _MASK, hsHubConfig1) & (1 << linkId))
    {
        dprintf("PEER3 ");    
        bNone = LW_FALSE;    
    }

    if (((DRF_VAL(_PFB_HSHUB_CONFIG0_PEER,   _PCIE,   _MASK, hsHubConfig0) & (1 << 4)) == 0) &&
          DRF_VAL(_PFB_HSHUB_CONFIG2_PEER_4, _LWLINK, _MASK, hsHubConfig2) & (1 << linkId))
    {
        dprintf("PEER4 ");    
        bNone = LW_FALSE;    
    }

    if (((DRF_VAL(_PFB_HSHUB_CONFIG0_PEER,   _PCIE,   _MASK, hsHubConfig0) & (1 << 5)) == 0) &&
          DRF_VAL(_PFB_HSHUB_CONFIG2_PEER_5, _LWLINK, _MASK, hsHubConfig2) & (1 << linkId))
    {
        dprintf("PEER5 ");    
        bNone = LW_FALSE;    
    }

    if (((DRF_VAL(_PFB_HSHUB_CONFIG0_PEER,   _PCIE,   _MASK, hsHubConfig0) & (1 << 6)) == 0) &&
          DRF_VAL(_PFB_HSHUB_CONFIG2_PEER_6, _LWLINK, _MASK, hsHubConfig2) & (1 << linkId))
    {
        dprintf("PEER6 ");    
        bNone = LW_FALSE;    
    }

    if (((DRF_VAL(_PFB_HSHUB_CONFIG0_PEER,   _PCIE,   _MASK, hsHubConfig0) & (1 << 7)) == 0) &&
          DRF_VAL(_PFB_HSHUB_CONFIG2_PEER_7, _LWLINK, _MASK, hsHubConfig2) & (1 << linkId))
    {
        dprintf("PEER7 ");    
        bNone = LW_FALSE;    
    }

    if (bNone)
    {
        dprintf("NONE");    
    }

    dprintf("\n");    
}

// Print DLPL device state
void lwlinkPrintDlplState_GV100(LwU32 addr)
{
    dprintf("LW_PLWL_LINK_STATE: 0x%08X ", GPU_REG_RD32(addr + LW_PLWL_LINK_STATE));
    switch (DRF_VAL(_PLWL, _LINK_STATE, _STATE, GPU_REG_RD32(addr + LW_PLWL_LINK_STATE)))
    {
        case LW_PLWL_LINK_STATE_STATE_INIT:
             dprintf("(INIT)\n");
             break;
        case LW_PLWL_LINK_STATE_STATE_HWCFG:
             dprintf("(HWCFG)\n");
             break;
        case LW_PLWL_LINK_STATE_STATE_SWCFG:
             dprintf("(SWCFG)\n");
             break;
        case LW_PLWL_LINK_STATE_STATE_ACTIVE:
             dprintf("(ACTIVE)\n");
             break;
        case LW_PLWL_LINK_STATE_STATE_FAULT:
             dprintf("(FAULT)\n");
             break;
        case LW_PLWL_LINK_STATE_STATE_RCVY_AC:
             dprintf("(RCVY_AC)\n");
             break;
        case LW_PLWL_LINK_STATE_STATE_RCVY_SW:
             dprintf("(RCVY_SW)\n");
             break;
        case LW_PLWL_LINK_STATE_STATE_RCVY_RX:
             dprintf("(RCVY_RX)\n");
             break;
        default:
             dprintf("(UNKNOWN)\n");
             break;
    }    
    
    dprintf("LW_PLWL_SL0_SLSM_STATUS_TX: 0x%08X ", GPU_REG_RD32(addr + LW_PLWL_SL0_SLSM_STATUS_TX));
    switch (DRF_VAL(_PLWL, _SL0_SLSM_STATUS_TX, _PRIMARY_STATE, GPU_REG_RD32(addr + LW_PLWL_SL0_SLSM_STATUS_TX)))
    {
        case LW_PLWL_SL0_SLSM_STATUS_TX_PRIMARY_STATE_HS:
             dprintf("(HS)\n");
             break;
        case LW_PLWL_SL0_SLSM_STATUS_TX_PRIMARY_STATE_EIGHTH:
             dprintf("(EIGHTH)\n");
             break;
        case LW_PLWL_SL0_SLSM_STATUS_TX_PRIMARY_STATE_TRAIN:
             dprintf("(TRAIN)\n");
             break;
        case LW_PLWL_SL0_SLSM_STATUS_TX_PRIMARY_STATE_SAFE:
             dprintf("(SAFE)\n");
             break;
        case LW_PLWL_SL0_SLSM_STATUS_TX_PRIMARY_STATE_OFF:
             dprintf("(OFF)\n");
             break;
        default:
             dprintf("(UNKNOWN)\n");
             break;
    }    
    
    dprintf("LW_PLWL_SL1_SLSM_STATUS_RX: 0x%08X ", GPU_REG_RD32(addr + LW_PLWL_SL1_SLSM_STATUS_RX));
    switch (DRF_VAL(_PLWL, _SL1_SLSM_STATUS_RX, _PRIMARY_STATE, GPU_REG_RD32(addr + LW_PLWL_SL1_SLSM_STATUS_RX)))
    {
        case LW_PLWL_SL1_SLSM_STATUS_RX_PRIMARY_STATE_HS:
             dprintf("(HS)\n");
             break;
        case LW_PLWL_SL1_SLSM_STATUS_RX_PRIMARY_STATE_EIGHTH:
             dprintf("(EIGHTH)\n");
             break;
        case LW_PLWL_SL1_SLSM_STATUS_RX_PRIMARY_STATE_TRAIN:
             dprintf("(TRAIN)\n");
             break;
        case LW_PLWL_SL1_SLSM_STATUS_RX_PRIMARY_STATE_SAFE:
             dprintf("(SAFE)\n");
             break;
        case LW_PLWL_SL1_SLSM_STATUS_RX_PRIMARY_STATE_OFF:
             dprintf("(OFF)\n");
             break;
        default:
             dprintf("(UNKNOWN)\n");
             break;
    }    
    
    dprintf("\n");
    
    dprintf("LW_PLWL_LINK_CONFIG: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_LINK_CONFIG));
    dprintf("LW_PLWL_LINK_TIMEOUT_A: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_LINK_TIMEOUT_A));
    dprintf("LW_PLWL_LINK_TIMEOUT_B: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_LINK_TIMEOUT_B));
    dprintf("LW_PLWL_CLK_TIMEOUT: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_CLK_TIMEOUT));
    dprintf("LW_PLWL_LINK_CHANGE: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_LINK_CHANGE));
    dprintf("LW_PLWL_SUBLINK_CHANGE: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SUBLINK_CHANGE));
    dprintf("LW_PLWL_LINK_TEST: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_LINK_TEST));
    dprintf("LW_PLWL_INTR: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_INTR));
    dprintf("LW_PLWL_INTR_SW2: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_INTR_SW2));
    dprintf("LW_PLWL_INTR_MINION: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_INTR_MINION));
    dprintf("LW_PLWL_INTR_STALL_EN: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_INTR_STALL_EN));
    dprintf("LW_PLWL_INTR_NONSTALL_EN: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_INTR_NONSTALL_EN));
    dprintf("LW_PLWL_INTR_MINIONR_EN: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_INTR_MINIONR_EN));
    dprintf("LW_PLWL_INTR_MINIONF_EN: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_INTR_MINIONF_EN));
    dprintf("LW_PLWL_INTR_CTL1: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_INTR_CTL1));
    dprintf("LW_PLWL_ERROR_COUNT_CTRL: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_ERROR_COUNT_CTRL));
    dprintf("LW_PLWL_ERROR_COUNT1: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_ERROR_COUNT1));
    dprintf("LW_PLWL_MINION_REQUEST: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_MINION_REQUEST));

    dprintf("LW_PLWL_SL0_TRAIN0_TX: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL0_TRAIN0_TX));
    dprintf("LW_PLWL_SL0_TRAIN1_TX: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL0_TRAIN1_TX));
    dprintf("LW_PLWL_SL0_TRAIN2_TX: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL0_TRAIN2_TX));
    dprintf("LW_PLWL_SL0_TL_CREDITS: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL0_TL_CREDITS));
    dprintf("LW_PLWL_SL0_REPLAY_TIMEOUT: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL0_REPLAY_TIMEOUT));
    dprintf("LW_PLWL_SL0_REPLAY_THRESHOLD: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL0_REPLAY_THRESHOLD));
    dprintf("LW_PLWL_SL0_REPLAY_STATUS: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL0_REPLAY_STATUS));
    dprintf("LW_PLWL_SL0_R4TX_COMMAND: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL0_R4TX_COMMAND));
    dprintf("LW_PLWL_SL0_R4TX_CONTROL: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL0_R4TX_CONTROL));
    dprintf("LW_PLWL_SL0_R4TX_WDATA0: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL0_R4TX_WDATA0));
    dprintf("LW_PLWL_SL0_R4TX_WDATA1: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL0_R4TX_WDATA1));
    dprintf("LW_PLWL_SL0_R4TX_RDATA0: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL0_R4TX_RDATA0));
    dprintf("LW_PLWL_SL0_R4TX_RDATA1: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL0_R4TX_RDATA1));
    dprintf("LW_PLWL_SL0_TXLANECRC: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL0_TXLANECRC));
    dprintf("LW_PLWL_SL0_SCRAM_TX: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL0_SCRAM_TX));

    dprintf("LW_PLWL_SL1_TRAIN0_RX: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL1_TRAIN0_RX));
    dprintf("LW_PLWL_SL1_R4LOCAL_COMMAND: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL1_R4LOCAL_COMMAND));
    dprintf("LW_PLWL_SL1_R4LOCAL_WDATA0: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL1_R4LOCAL_WDATA0));
    dprintf("LW_PLWL_SL1_R4LOCAL_WDATA1: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL1_R4LOCAL_WDATA1));
    dprintf("LW_PLWL_SL1_R4LOCAL_RDATA0: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL1_R4LOCAL_RDATA0));
    dprintf("LW_PLWL_SL1_R4LOCAL_RDATA1: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL1_R4LOCAL_RDATA1));
    dprintf("LW_PLWL_SL1_RXLANECRC: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL1_RXLANECRC));
    dprintf("LW_PLWL_SL1_SLSM_STATUS_RX: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL1_SLSM_STATUS_RX));
    dprintf("LW_PLWL_SL1_SCRAM_RX: 0x%08X\n", GPU_REG_RD32(addr + LW_PLWL_SL1_SCRAM_RX));
}

// Print TLC device state
void lwlinkPrintTlcState_GV100(LwU32 addr)
{
    dprintf("TLC RX:\n");    
    dprintf("LW_LWLTLC_RX_CTRL_BUFFER_SZ_VC0: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_BUFFER_SZ_VC0));
    dprintf("LW_LWLTLC_RX_CTRL_BUFFER_SZ_VC1: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_BUFFER_SZ_VC1));
    dprintf("LW_LWLTLC_RX_CTRL_BUFFER_SZ_VC2: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_BUFFER_SZ_VC2));
    dprintf("LW_LWLTLC_RX_CTRL_BUFFER_SZ_VC3: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_BUFFER_SZ_VC3));
    dprintf("LW_LWLTLC_RX_CTRL_BUFFER_SZ_VC4: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_BUFFER_SZ_VC4));
    dprintf("LW_LWLTLC_RX_CTRL_BUFFER_SZ_VC5: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_BUFFER_SZ_VC5));
    dprintf("LW_LWLTLC_RX_CTRL_BUFFER_SZ_VC6: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_BUFFER_SZ_VC6));
    dprintf("LW_LWLTLC_RX_CTRL_BUFFER_SZ_VC7: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_BUFFER_SZ_VC7));
    dprintf("LW_LWLTLC_RX_CTRL_BUFFER_CREDITS_VC0: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_BUFFER_CREDITS_VC0));
    dprintf("LW_LWLTLC_RX_CTRL_BUFFER_CREDITS_VC1: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_BUFFER_CREDITS_VC1));
    dprintf("LW_LWLTLC_RX_CTRL_BUFFER_CREDITS_VC2: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_BUFFER_CREDITS_VC2));
    dprintf("LW_LWLTLC_RX_CTRL_BUFFER_CREDITS_VC3: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_BUFFER_CREDITS_VC3));
    dprintf("LW_LWLTLC_RX_CTRL_BUFFER_CREDITS_VC4: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_BUFFER_CREDITS_VC4));
    dprintf("LW_LWLTLC_RX_CTRL_BUFFER_CREDITS_VC5: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_BUFFER_CREDITS_VC5));
    dprintf("LW_LWLTLC_RX_CTRL_BUFFER_CREDITS_VC6: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_BUFFER_CREDITS_VC6));
    dprintf("LW_LWLTLC_RX_CTRL_BUFFER_CREDITS_VC7: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_BUFFER_CREDITS_VC7));
    dprintf("LW_LWLTLC_RX_CTRL_LINK_CONFIG: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_LINK_CONFIG));
    dprintf("LW_LWLTLC_RX_CTRL_BUFFER_READY: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_CTRL_BUFFER_READY));
    dprintf("LW_LWLTLC_RX_ERR_CONTAIN_EN_0: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_ERR_CONTAIN_EN_0));
    dprintf("LW_LWLTLC_RX_ERR_CONTAIN_EN_1: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_ERR_CONTAIN_EN_1));
    dprintf("LW_LWLTLC_RX_ERR_LOG_EN_0: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_ERR_LOG_EN_0));
    dprintf("LW_LWLTLC_RX_ERR_LOG_EN_1: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_ERR_LOG_EN_1));
    dprintf("LW_LWLTLC_RX_ERR_REPORT_EN_0: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_ERR_REPORT_EN_0));
    dprintf("LW_LWLTLC_RX_ERR_REPORT_EN_1: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_ERR_REPORT_EN_1));
    dprintf("LW_LWLTLC_RX_ERR_FIRST_0: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_ERR_FIRST_0));
    dprintf("LW_LWLTLC_RX_ERR_STATUS_0: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_ERR_STATUS_0));
    dprintf("LW_LWLTLC_RX_ERR_INJECT_0: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_ERR_INJECT_0));
    dprintf("LW_LWLTLC_RX_ERR_FIRST_1: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_ERR_FIRST_1));
    dprintf("LW_LWLTLC_RX_ERR_STATUS_1: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_ERR_STATUS_1));
    dprintf("LW_LWLTLC_RX_ERR_INJECT_1: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_ERR_INJECT_1));

    dprintf("TLC TX:\n");    
    dprintf("LW_LWLTLC_TX_CTRL_BUFFER_SZ_VC0: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_BUFFER_SZ_VC0));
    dprintf("LW_LWLTLC_TX_CTRL_BUFFER_SZ_VC1: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_BUFFER_SZ_VC1));
    dprintf("LW_LWLTLC_TX_CTRL_BUFFER_SZ_VC2: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_BUFFER_SZ_VC2));
    dprintf("LW_LWLTLC_TX_CTRL_BUFFER_SZ_VC3: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_BUFFER_SZ_VC3));
    dprintf("LW_LWLTLC_TX_CTRL_BUFFER_SZ_VC4: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_BUFFER_SZ_VC4));
    dprintf("LW_LWLTLC_TX_CTRL_BUFFER_SZ_VC5: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_BUFFER_SZ_VC5));
    dprintf("LW_LWLTLC_TX_CTRL_BUFFER_SZ_VC6: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_BUFFER_SZ_VC6));
    dprintf("LW_LWLTLC_TX_CTRL_BUFFER_SZ_VC7: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_BUFFER_SZ_VC7));
    dprintf("LW_LWLTLC_TX_CTRL_BUFFER_CREDITS_VC0: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_BUFFER_CREDITS_VC0));
    dprintf("LW_LWLTLC_TX_CTRL_BUFFER_CREDITS_VC1: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_BUFFER_CREDITS_VC1));
    dprintf("LW_LWLTLC_TX_CTRL_BUFFER_CREDITS_VC2: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_BUFFER_CREDITS_VC2));
    dprintf("LW_LWLTLC_TX_CTRL_BUFFER_CREDITS_VC3: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_BUFFER_CREDITS_VC3));
    dprintf("LW_LWLTLC_TX_CTRL_BUFFER_CREDITS_VC4: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_BUFFER_CREDITS_VC4));
    dprintf("LW_LWLTLC_TX_CTRL_BUFFER_CREDITS_VC5: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_BUFFER_CREDITS_VC5));
    dprintf("LW_LWLTLC_TX_CTRL_BUFFER_CREDITS_VC6: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_BUFFER_CREDITS_VC6));
    dprintf("LW_LWLTLC_TX_CTRL_BUFFER_CREDITS_VC7: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_BUFFER_CREDITS_VC7));
    dprintf("LW_LWLTLC_TX_CTRL_LINK_CONFIG: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_LINK_CONFIG));
    dprintf("LW_LWLTLC_TX_CTRL_REPLAY_BUFFER_CREDIT_LIMIT: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_REPLAY_BUFFER_CREDIT_LIMIT));
    dprintf("LW_LWLTLC_TX_CTRL_BUFFER_READY: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_CTRL_BUFFER_READY));
    dprintf("LW_LWLTLC_TX_ERR_CONTAIN_EN_0: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_ERR_CONTAIN_EN_0));
    dprintf("LW_LWLTLC_TX_ERR_LOG_EN_0: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_ERR_LOG_EN_0));
    dprintf("LW_LWLTLC_TX_ERR_REPORT_EN_0: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_ERR_REPORT_EN_0));
    dprintf("LW_LWLTLC_TX_ERR_FIRST_0: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_ERR_FIRST_0));
    dprintf("LW_LWLTLC_TX_ERR_STATUS_0: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_ERR_STATUS_0));
    dprintf("LW_LWLTLC_TX_ERR_INJECT_0: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_ERR_INJECT_0));

    dprintf("TLC LPWR RX:\n");    
    dprintf("LW_LWLTLC_RX_PWRM_IC_SW_CTRL: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_PWRM_IC_SW_CTRL));
    dprintf("LW_LWLTLC_RX_PWRM_IC: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_PWRM_IC));
    dprintf("LW_LWLTLC_RX_PWRM_IC_LIMIT: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_PWRM_IC_LIMIT));
    dprintf("LW_LWLTLC_RX_PWRM_IC_INC: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_PWRM_IC_INC));
    dprintf("LW_LWLTLC_RX_PWRM_IC_DEC: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_PWRM_IC_DEC));
    dprintf("LW_LWLTLC_RX_PWRM_IC_LP_ENTER_THRESHOLD: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_PWRM_IC_LP_ENTER_THRESHOLD));
    dprintf("LW_LWLTLC_RX_PWRM_IC_LP_EXIT_THRESHOLD: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_RX_PWRM_IC_LP_EXIT_THRESHOLD));

    dprintf("TLC LPWR TX:\n");    
    dprintf("LW_LWLTLC_TX_PWRM_IC_SW_CTRL: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_PWRM_IC_SW_CTRL));
    dprintf("LW_LWLTLC_TX_PWRM_IC: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_PWRM_IC));
    dprintf("LW_LWLTLC_TX_PWRM_IC_LIMIT: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_PWRM_IC_LIMIT));
    dprintf("LW_LWLTLC_TX_PWRM_IC_INC: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_PWRM_IC_INC));
    dprintf("LW_LWLTLC_TX_PWRM_IC_DEC: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_PWRM_IC_DEC));
    dprintf("LW_LWLTLC_TX_PWRM_IC_LP_ENTER_THRESHOLD: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_PWRM_IC_LP_ENTER_THRESHOLD));
    dprintf("LW_LWLTLC_TX_PWRM_IC_LP_EXIT_THRESHOLD: 0x%08X\n", GPU_REG_RD32(addr + LW_LWLTLC_TX_PWRM_IC_LP_EXIT_THRESHOLD));
}

// Print MIF device state
void lwlinkPrintMifState_GV100(LwU32 addr)
{
    dprintf("MIF RX:\n");    
    dprintf("LW_IOCTRLMIF_RX_CTRL_BUFFER_SZ_REQ: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_CTRL_BUFFER_SZ_REQ));
    dprintf("LW_IOCTRLMIF_RX_CTRL_BUFFER_SZ_RSP: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_CTRL_BUFFER_SZ_RSP));
    dprintf("LW_IOCTRLMIF_RX_CTRL_BUFFER_SZ_DGD: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_CTRL_BUFFER_SZ_DGD));
    dprintf("LW_IOCTRLMIF_RX_CTRL_BUFFER_SZ_PRB: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_CTRL_BUFFER_SZ_PRB));
    dprintf("LW_IOCTRLMIF_RX_CTRL_BUFFER_SZ_ATR: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_CTRL_BUFFER_SZ_ATR));
    dprintf("LW_IOCTRLMIF_RX_CTRL_BUFFER_SZ_ATSD: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_CTRL_BUFFER_SZ_ATSD));
    dprintf("LW_IOCTRLMIF_RX_CTRL_BUFFER_CREDITS_REQ: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_CTRL_BUFFER_CREDITS_REQ));
    dprintf("LW_IOCTRLMIF_RX_CTRL_BUFFER_CREDITS_RSP: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_CTRL_BUFFER_CREDITS_RSP));
    dprintf("LW_IOCTRLMIF_RX_CTRL_BUFFER_CREDITS_DGD: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_CTRL_BUFFER_CREDITS_DGD));
    dprintf("LW_IOCTRLMIF_RX_CTRL_BUFFER_CREDITS_PRB: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_CTRL_BUFFER_CREDITS_PRB));
    dprintf("LW_IOCTRLMIF_RX_CTRL_BUFFER_CREDITS_ATR: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_CTRL_BUFFER_CREDITS_ATR));
    dprintf("LW_IOCTRLMIF_RX_CTRL_BUFFER_CREDITS_ATSD: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_CTRL_BUFFER_CREDITS_ATSD));
    dprintf("LW_IOCTRLMIF_RX_CTRL_BUFFER_READY: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_CTRL_BUFFER_READY));
    dprintf("LW_IOCTRLMIF_RX_ERR_STATUS_0: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_ERR_STATUS_0));
    dprintf("LW_IOCTRLMIF_RX_ERR_LOG_EN_0: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_ERR_LOG_EN_0));
    dprintf("LW_IOCTRLMIF_RX_ERR_REPORT_EN_0: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_ERR_REPORT_EN_0));
    dprintf("LW_IOCTRLMIF_RX_ERR_CONTAIN_EN_0: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_ERR_CONTAIN_EN_0));
    dprintf("LW_IOCTRLMIF_RX_ERR_INJECT_0: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_ERR_INJECT_0));
    dprintf("LW_IOCTRLMIF_RX_ERR_FIRST_0: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_ERR_FIRST_0));
    dprintf("LW_IOCTRLMIF_RX_ERR_MISC_0: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_RX_ERR_MISC_0));

    dprintf("MIF TX:\n");    
    dprintf("LW_IOCTRLMIF_TX_CTRL_BUFFER_SZ_REQ: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_CTRL_BUFFER_SZ_REQ));
    dprintf("LW_IOCTRLMIF_TX_CTRL_BUFFER_SZ_RSP: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_CTRL_BUFFER_SZ_RSP));
    dprintf("LW_IOCTRLMIF_TX_CTRL_BUFFER_SZ_DGD: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_CTRL_BUFFER_SZ_DGD));
    dprintf("LW_IOCTRLMIF_TX_CTRL_BUFFER_SZ_PRB: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_CTRL_BUFFER_SZ_PRB));
    dprintf("LW_IOCTRLMIF_TX_CTRL_BUFFER_SZ_ATR: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_CTRL_BUFFER_SZ_ATR));
    dprintf("LW_IOCTRLMIF_TX_CTRL_BUFFER_SZ_ATSD: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_CTRL_BUFFER_SZ_ATSD));
    dprintf("LW_IOCTRLMIF_TX_CTRL_BUFFER_CREDITS_REQ: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_CTRL_BUFFER_CREDITS_REQ));
    dprintf("LW_IOCTRLMIF_TX_CTRL_BUFFER_CREDITS_RSP: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_CTRL_BUFFER_CREDITS_RSP));
    dprintf("LW_IOCTRLMIF_TX_CTRL_BUFFER_CREDITS_DGD: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_CTRL_BUFFER_CREDITS_DGD));
    dprintf("LW_IOCTRLMIF_TX_CTRL_BUFFER_CREDITS_PRB: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_CTRL_BUFFER_CREDITS_PRB));
    dprintf("LW_IOCTRLMIF_TX_CTRL_BUFFER_CREDITS_ATR: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_CTRL_BUFFER_CREDITS_ATR));
    dprintf("LW_IOCTRLMIF_TX_CTRL_BUFFER_CREDITS_ATSD: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_CTRL_BUFFER_CREDITS_ATSD));
    dprintf("LW_IOCTRLMIF_TX_CTRL_BUFFER_READY: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_CTRL_BUFFER_READY));
    dprintf("LW_IOCTRLMIF_TX_CTRL_MISC: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_CTRL_MISC));
    dprintf("LW_IOCTRLMIF_TX_ERR_STATUS_0: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_ERR_STATUS_0));
    dprintf("LW_IOCTRLMIF_TX_ERR_LOG_EN_0: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_ERR_LOG_EN_0));
    dprintf("LW_IOCTRLMIF_TX_ERR_REPORT_EN_0: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_ERR_REPORT_EN_0));
    dprintf("LW_IOCTRLMIF_TX_ERR_CONTAIN_EN_0: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_ERR_CONTAIN_EN_0));
    dprintf("LW_IOCTRLMIF_TX_ERR_INJECT_0: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_ERR_INJECT_0));
    dprintf("LW_IOCTRLMIF_TX_ERR_FIRST_0: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_ERR_FIRST_0));
    dprintf("LW_IOCTRLMIF_TX_ERR_MISC_0: 0x%08X\n", GPU_REG_RD32(addr + LW_IOCTRLMIF_TX_ERR_MISC_0));
}
