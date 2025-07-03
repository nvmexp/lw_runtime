/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2019-2020 by LWPU Corporation.  All rights reserved.  All
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

#include "ampere/ga100/dev_top.h"
#include "ampere/ga100/dev_hshub_SW.h"
#include "ampere/ga100/dev_mmu_hshub_SW.h"
#include "ampere/ga100/dev_graphics_nobundle.h"
#include "ampere/ga100/dev_fb.h"
#include "ampere/ga100/dev_ltc.h"
#include "ampere/ga100/dev_lwldl_ip.h"
#include "ampere/ga100/dev_lwltlc_ip.h"
#include "ampere/ga100/dev_lwlipt_lnk_ip.h"
#include "ampere/ga100/ioctrl_discovery.h"
#include "ampere/ga100/lwlinkip_discovery.h"

/*!
 * @brief  Function that prints HSHUB configuration
 *         Prints SYS/PEER_CONFIG0
 */
void
lwlinkPrintHshubConfig_GA100(void)
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
 * @brief  Function that prints HSHUB configuration
 *         Prints SYS/PEER_CONFIG0
 *         TODO: GPCMMU has broadcast registers. What to do?
 */
void
lwlinkPrintHshubConnectionCfg_GA100(void)
{
    LwU32 fbhubPeer   = GPU_REG_RD32(LW_PFB_FBHUB_PEER_HSHUB_CONNECTION_CFG);
    LwU32 hubmmuPeer  = GPU_REG_RD32(LW_PFB_PRI_MMU_PEER_HSHUB_CONNECTION_CFG);
    LwU32 hsmmu0Peer  = GPU_REG_RD32(LW_PFB_HSMMU0_PRI_MMU_PEER_HSHUB_CONNECTION_CFG);
    LwU32 hsmmu1Peer  = GPU_REG_RD32(LW_PFB_HSMMU1_PRI_MMU_PEER_HSHUB_CONNECTION_CFG);

    LwU32 fbhubSys = GPU_REG_RD32(LW_PFB_FBHUB_SYSMEM_HSHUB_CONNECTION_CFG);
    LwU32 ltcsSys  = GPU_REG_RD32(LW_PLTCG_LTCS_LTSS_SYSMEM_HSHUB_CONNECTION_CFG);

    dprintf("\n");

    // Print the PEER_HSHUB_CONNECTION_CFG registers
    dprintf("*_PEER_HSHUB_CONNECTION_CFG registers:\n");
    dprintf("FBHUB  = 0x%x\n", fbhubPeer);
    dprintf("HUBMMU = 0x%x\n", hubmmuPeer);
    dprintf("HSMMU0 = 0x%x\n", hsmmu0Peer);
    dprintf("HSMMU1 = 0x%x\n", hsmmu1Peer);

    // Print the SYSMEM_HSHUB_CONNECTION_CFG registers
    dprintf("*_SYSMEM_HSHUB_CONNECTION_CFG registers:\n");
    dprintf("FBHUB = 0x%x\n", fbhubSys);
    dprintf("LTCS  = 0x%x\n", ltcsSys);
}

/*!
 * @brief  Function that prints 2x2 MUX configuration
 *         for each HSHUB
 */
void
lwlinkPrintHshubMuxConfig_GA100(void)
{
    LwU32 hshub0Mux = GPU_REG_RD32(LW_PFB_HSHUB_HSHLWLMUX_CONFIG(0));
    LwU32 hshub1Mux = GPU_REG_RD32(LW_PFB_HSHUB_HSHLWLMUX_CONFIG(1));

    dprintf("\n");

    dprintf("MUX settings for Links 0-5  connected to HSHUB 0: %x\n", hshub0Mux);
    dprintf("MUX settings for Links 6-11 connected to HSHUB 1: %x\n", hshub1Mux);
}

/*!
 * @brief Function that print out the ALT training registers for all links 
 *
 */
void 
lwlinkDumpAltTraining_GA100(void)
{
    LwU32 link;

    for(link = 0; link < 12; ++link)
    {
        lwlinkDumpAltTrainingLink_GA100(link);
    }
}

/*!
 * @brief Function that print out the ALT training registers for a specific link 
 *
 */
void 
lwlinkDumpAltTrainingLink_GA100(LwU32 link)
{
    LwU32 linkDeviceAddr = pLwlink[indexGpu].lwlinkLwldlBaseAddress(link);
    
    LwU32 train2Tx = GPU_REG_RD32(LW_LWLDL_TX_TRAIN2_TX + linkDeviceAddr);
    LwU32 train0Rx = GPU_REG_RD32(LW_LWLDL_RX_TRAIN0_RX + linkDeviceAddr); 
    LwU32 train3Rx = GPU_REG_RD32(LW_LWLDL_RX_TRAIN3_RX + linkDeviceAddr);
    LwU32 train4Rx = GPU_REG_RD32(LW_LWLDL_RX_TRAIN4_RX + linkDeviceAddr);
    LwU32 linkTrain = GPU_REG_RD32(LW_LWLDL_TOP_LINK_TRAIN + linkDeviceAddr);
 
    dprintf("Dumping Training Registers for links: %d\n", link);
    dprintf("TRAIN2_TX_LEN: 0x%x\n", DRF_VAL(_LWLDL, _TX_TRAIN2_TX , _TRAIN_LEN, train2Tx));
    dprintf("TRAIN2_TX_LEN_SCL: 0x%x\n", DRF_VAL(_LWLDL, _TX_TRAIN2_TX , _TRAIN_LEN_SCL, train2Tx));
    dprintf("TRAIN0_RX_LEN: 0x%x\n", DRF_VAL(_LWLDL, _RX_TRAIN0_RX, _TRAIN_LEN, train0Rx));
    dprintf("TRAIN0_RX_LEN_SCL: 0x%x\n", DRF_VAL(_LWLDL, _RX_TRAIN0_RX, _TRAIN_LEN_SCL, train0Rx));
    dprintf("TRAIN3_RX_EQ_LEN: 0x%x\n", DRF_VAL(_LWLDL, _RX_TRAIN3_RX, _TRAIN_EQ_LEN, train3Rx));
    dprintf("TRAIN3_RX_EQ_LEN_SCL: 0x%x\n",DRF_VAL(_LWLDL, _RX_TRAIN3_RX, _TRAIN_EQ_LEN_SCL, train3Rx));
    dprintf("TRAIN4_RX_FOM_WAIT_LEN: 0x%x\n", DRF_VAL(_LWLDL, _RX_TRAIN4_RX, _FOM_WAIT_LEN, train4Rx));
    dprintf("TRAIN4_RX_FOM_WAIT_LEN_SCL: 0x%x\n", DRF_VAL(_LWLDL, _RX_TRAIN4_RX, _FOM_WAIT_LEN_SCL, train4Rx));
    dprintf("LINK_TRAIN_BASE: 0x%x\n", DRF_VAL(_LWLDL, _TOP_LINK_TRAIN, _BASE, linkTrain));

}

/*!
 * @brief Function to get the LWLDL Base address for a specific link 
 *
 */
LwU32
lwlinkLwldlBaseAddress_GA100(LwU32 link)
{
    //4 links per ioctrl in ga100
    LwU32 localLink = link % 4;
    LwU32 ioctrlNum  = link / 4;

    LwU32 linkDeviceAddr = LW_DISCOVERY_IOCTRL_UNICAST_0_SW_DEVICE_BASE_LWLDL_0 + (localLink * LWLINK_LINK_STRIDE) + (ioctrlNum * LWLINK_IOCTRL_STRIDE);
    return linkDeviceAddr;
}

/*!
 * @brief Function to get the LWLIPT Base address for a specific link 
 *
 */ 
LwU32
lwlinkLwliptBaseAddress_GA100(LwU32 link)
{
    //4 links per ioctrl in ga100
    LwU32 localLink = link % 4;
    LwU32 ioctrlNum  = link / 4;

    LwU32 linkDeviceAddr = LW_DISCOVERY_IOCTRL_UNICAST_0_SW_DEVICE_BASE_LWLIPT_LNK_0 + (localLink * LWLINK_LINK_STRIDE) + (ioctrlNum * LWLINK_IOCTRL_STRIDE);
    return linkDeviceAddr;
}

LwBool lwlinkIsLinkInReset_GA100(LwU32 linkId)
{
    LwU32 regData;
    LwU32 lwliptBase = pLwlink[indexGpu].lwlinkLwliptBaseAddress(linkId);

    regData = GPU_REG_RD32(lwliptBase + LW_LWLIPT_LNK_RESET_RSTSEQ_LINK_RESET);
    if(DRF_VAL(_LWLIPT_LNK, _RESET_RSTSEQ, _LINK_RESET_LINK_RESET, regData))
    {
        return LW_TRUE;
    }

    return LW_FALSE;
}
/*!
 * @brief Main Function to print Lwlink info
 *
 */
void lwlinkPrintStatus_GA100(LwBool bComplete)
{
    LwU32 linksDiscoveredMask = 0;
    LwU32 linkId;
    lwlDiscover *pLwlDiscoverList = NULL;
    lwlDiscover *device = NULL;
    LwU32 resetMask = 0;
    LwU32 ioctrlCount = 0;
    LwU32 i;

    LwU32 deviceInfoCfg = GPU_REG_RD32(LW_PTOP_DEVICE_INFO_CFG);
    LwU32 numValidRows = DRF_VAL(_PTOP, _DEVICE_INFO_CFG, _NUM_ROWS, deviceInfoCfg);

    LwU32 *pIoctrlPriBase = malloc(numValidRows * sizeof(*pIoctrlPriBase));

    if (NULL == pIoctrlPriBase)
    {
        return;
    }
    memset(pIoctrlPriBase, 0, numValidRows * sizeof(*pIoctrlPriBase));

    // Discover IOCTRL in device info
    ioctrlCount = pLwlink[indexGpu].lwlinkIoctrlDiscovery(pIoctrlPriBase);
    if (ioctrlCount == 0)
    {
        dprintf("No ioctrls were discovered in PTOP_DEVICE_INFO_CFG!\n");
        return;
    }

    for (i = 0; i < ioctrlCount; i++)
    {
        dprintf("\n");
        dprintf("IOCTRL %d base = 0x%08X\n", i, pIoctrlPriBase[i]);
        dprintf("\n");
    }

    //Discover all the links for each of the IOCTRLS
    for (i = 0; i < ioctrlCount; i++)
    {
        linksDiscoveredMask |= pLwlink[indexGpu].lwlinkLinkDiscovery((void *)&pLwlDiscoverList, pIoctrlPriBase[i]);
    }

    if (linksDiscoveredMask == 0)
    {
        dprintf("No links were detected during the discovery process in ioctrl\n");
        return;
    }

    dprintf("\n");
    dprintf("Links discovered in IOCTRL Discovery Table: 0x%08X\n", linksDiscoveredMask);
    dprintf("\n");

    for (linkId = 0; linkId < LWLINK_MAX_LINKS; linkId++)
    {
        if(pLwlink[indexGpu].lwlinkIsLinkInReset(linkId))
        {
            resetMask |= LWBIT(linkId);
        }
    }

    dprintf("\n");
    dprintf("Reset Mask (LW_LWLIPT_LNK_RESET_RSTSEQ_LINK_RESET_LINK_RESET) : 0x%08X\n", resetMask);
    dprintf("\n");

    dprintf("Dumping state of discovered links NOT in reset: 0x%08X\n", linksDiscoveredMask & resetMask);
    dprintf("\n");

    // Dump the HSHUB traffic type
    dprintf("\n");
    dprintf("HSHUB traffic type: ");    
    pLwlink[indexGpu].lwlinkPrintHshubConfig();
    dprintf("\n");

    // Dump the link state for all discovered links not in reset
    for (linkId = 0; linkId < LWLINK_MAX_LINKS; linkId++)
    {
        if (linksDiscoveredMask & resetMask & LWBIT(linkId)) continue;

        dprintf("============ LINK %d ============\n", linkId);    

        // Locate the DLPL for this link
        device = lwlinkDeviceInstanceInfoLookUp(pLwlDiscoverList, linkId, LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_DLPL);
        if (device)
        {
            dprintf("Link %d DLPL information\n", linkId);    
            pLwlink[indexGpu].lwlinkPrintDlplState(linkId );
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
            pLwlink[indexGpu].lwlinkPrintTlcState(linkId);
        }
        else
        {
            dprintf("ERROR: Failed to locate TLC device information for link %d!!!\n", linkId);
        }
        dprintf("\n");

        // Locate the LWLIPT for this link
        device = lwlinkDeviceInstanceInfoLookUp(pLwlDiscoverList, linkId, LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_LWLIPT);
        if (device)
        {
            dprintf("Link %d LWLIPT information\n", linkId);    
            pLwlink[indexGpu].lwlinkPrintLwliptState(linkId);
        }
        else
        {
            dprintf("ERROR: Failed to locate LWLIPT information for link %d!!!\n", linkId);
        }
        dprintf("\n");     
    }

    freeLwlDiscoveryList(pLwlDiscoverList);
}

/*!
 * @brief Function walks PTOP_DEVICE_INFO registers to discover ioctrls.
 * It caches the Base Address of all ioctrl devices and returns -
 * - the count of IOCTRLS for Ampere+
 *
 */
LwU32 lwlinkIoctrlDiscovery_GA100(LwU32 *pIoctrlPriBase)
{
    LwU32 i;
    LwBool bPTopNextDataValid = LW_FALSE;
    LwBool bIoctrlDetected = LW_FALSE;
    LwU32 ioctrlCount = 0;

    LwU32 deviceInfoCfg = GPU_REG_RD32(LW_PTOP_DEVICE_INFO_CFG);
    LwU32 numValidRows = DRF_VAL(_PTOP, _DEVICE_INFO_CFG, _NUM_ROWS, deviceInfoCfg);

    for (i = 0; i < numValidRows; ++i)
    {
        LwU32 tableEntry = GPU_REG_RD32(LW_PTOP_DEVICE_INFO2(i));

        if (FLD_TEST_DRF(_PTOP, _DEVICE_INFO2, _ROW_VALUE, _ILWALID, tableEntry))
        {
            dprintf("DEVICE_INFO2 entry %d is invalid\n", i);
            continue;
        }

        if (FLD_TEST_DRF(_PTOP, _DEVICE_INFO2, _DEV_TYPE_ENUM, _IOCTRL, tableEntry) || bIoctrlDetected)
        {
            dprintf("DEVICE_INFO entry %d is IOCTRL TYPE_ENUM\n", i);
        }
        else
        {
            bIoctrlDetected = LW_FALSE;
            continue;
        }

        if (bPTopNextDataValid || tableEntry != LW_PTOP_DEVICE_INFO2_ROW_VALUE_ILWALID)
        {
            bPTopNextDataValid = FLD_TEST_DRF(_PTOP, _DEVICE_INFO2, _ROW_CHAIN, _MORE, tableEntry);
            bIoctrlDetected = LW_TRUE;

            if(!bPTopNextDataValid)
            {
                // Get the base address of the IOCTRL device
                pIoctrlPriBase[ioctrlCount] = DRF_VAL(_PTOP, _DEVICE_INFO2, _DEV_DEVICE_PRI_BASE,
                    tableEntry) << SF_SHIFT(LW_PTOP_DEVICE_INFO2_DEV_DEVICE_PRI_BASE);
                dprintf("DEVICE entry %d has IOCTRL pri base address = 0x%08x.\n", i, pIoctrlPriBase[ioctrlCount]);
                ioctrlCount++;
                bIoctrlDetected = LW_FALSE;
            }
        }
    }
    return ioctrlCount;
}

/*!
 * @brief Function walks the ioctrl registers to discover lwlink devices.
 * This function creates a very simple singly linked list to store device info
 *
 */
LwU32 lwlinkLinkDiscovery_GA100(void **pLwlVoid, LwU32 ioctrlPriBase)
{
    LwU32 i;
    LwU32 ioctrlDiscoverySize   = 0;
    LwU32 ioctrlInfoType        = DEVICE_INFO_TYPE_ILWALID;
    LwU32 ioctrlId              = 0;
    LwU32 foundDevices          = 0;
    LwU32 foundIds              = 0;
    LwU32 ioctrlEntry           = 0;
    LwU32 ioctrlEntryAddr       = 0;
    LwBool bIoctrlNextDataValid = LW_FALSE;

    lwlDiscover **pLwlDiscoverListHead = (lwlDiscover **)pLwlVoid;

    // priBaseAddr + 0 * 4
    ioctrlEntryAddr = ioctrlPriBase;
    ioctrlEntry     = GPU_REG_RD32(ioctrlEntryAddr);
    ioctrlInfoType  = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON, _DEVICE, ioctrlEntry);

    if (ioctrlInfoType == LW_LWLINKIP_DISCOVERY_COMMON_DEVICE_IOCTRL)
    {
        // priBaseAddr + 1 * 4
        ioctrlEntryAddr     = ioctrlPriBase + 4;
        ioctrlEntry         = GPU_REG_RD32(ioctrlEntryAddr);
        ioctrlDiscoverySize = DRF_VAL(_LWLINKIP, _DISCOVERY_COMMON_DATA1, _IOCTRL_LENGTH,
                                        ioctrlEntry);
    }
    else
    {
        dprintf("Error: first entry in IOCTRL_DISCOVERY is not IOCTRL!\n");
        return foundDevices;
    }

    for (i = 0; i < ioctrlDiscoverySize; ++i)
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
                    if (FLD_TEST_DRF(_LWLINKIP, _DISCOVERY_COMMON, _CHAIN, _DISABLE, ioctrlEntry))
                    {
                        bIoctrlNextDataValid = LW_FALSE;
                    }
                }
                break;
            }

            case LW_LWLINKIP_DISCOVERY_COMMON_ENTRY_DATA2:
            {
                if (bIoctrlNextDataValid)
                {
                    lwlDiscover *pTemp = malloc(sizeof(lwlDiscover));
                    pTemp->ID          = ioctrlId;
                    pTemp->addr        = ioctrlPriBase;
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

                    if (FLD_TEST_DRF(_LWLINKIP, _DISCOVERY_COMMON, _CHAIN, _DISABLE, ioctrlEntry))
                    {
                        bIoctrlNextDataValid = LW_FALSE;
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
                    bIoctrlNextDataValid = LW_TRUE;
                }
                else
                {
                    dprintf("Warning: Found TLC or DLPL in ioctrl, but there is"
                            "no associated data entry. This ioctrl device will be ignored\n");
                    continue;
                }

                if ((foundIds >> ioctrlId) & 1)
                {
                    foundDevices |= LWBIT(ioctrlId);
                }
                else
                {
                    foundIds |= LWBIT(ioctrlId);
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

/*!
 * @brief Function to get the LWLTLC Base address for a specific link 
 *
 */ 
LwU32
lwlinkLwltlcBaseAddress_GA100(LwU32 link)
{
    //4 links per ioctrl in ga100
    LwU32 localLink = link % 4;
    LwU32 ioctrlNum  = link / 4;

    LwU32 linkDeviceAddr = LW_DISCOVERY_IOCTRL_UNICAST_0_SW_DEVICE_BASE_LWLTLC_0 + (localLink * LWLINK_LINK_STRIDE) + (ioctrlNum * LWLINK_IOCTRL_STRIDE);
    return linkDeviceAddr;
}

/*!
 * @brief Function to print TLC device state
 *
 */
void lwlinkPrintTlcState_GA100(LwU32 linkId)
{
    LwU32 regData;
    LwU32 lwltlcBase = pLwlink[indexGpu].lwlinkLwltlcBaseAddress(linkId);

    dprintf("TLC RX:\n");
    regData = GPU_REG_RD32(lwltlcBase + LW_LWLTLC_RX_SYS_CTRL_BUFFER_READY);
    dprintf("LW_LWLTLC_RX_SYS_CTRL_BUFFER_READY  : 0x%08X\n", regData);
    if (FLD_TEST_DRF(_LWLTLC, _RX_SYS_CTRL, _BUFFER_READY_BUFFERRDY, _ENABLE, regData))
    {
        dprintf("RX BUFFER_READY is enabled\n");
    }

    dprintf("TLC TX:\n");
    regData = GPU_REG_RD32(lwltlcBase + LW_LWLTLC_TX_SYS_CTRL_BUFFER_READY);
    dprintf("LW_LWLTLC_TX_SYS_CTRL_BUFFER_READY  : 0x%08X\n", regData);
    if (FLD_TEST_DRF(_LWLTLC, _TX_SYS_CTRL, _BUFFER_READY_BUFFERRDY, _ENABLE, regData))
    {
        dprintf("TX BUFFER_READY is enabled\n");
    }
}

/*!
 * @brief Function to print LWLIPT device state
 *
 */
void lwlinkPrintLwliptState_GA100(LwU32 linkId)
{
    LwU32 linkModeCtrl;
    LwU32 linkLock;
    LwU32 linkClkCtrl;
    LwU32 linkClkCtrlLock;
    LwU32 linkChannelCtrl;
    LwU32 linkChannelCtrlLock;
    LwU32 linkChannelCtrl2;
    LwU32 linkChannelCtrl2Lock;
    LwU32 linkAn1Ctrl;
    LwU32 linkAn1CtrlLock;

    LwU32 lwliptBase = pLwlink[indexGpu].lwlinkLwliptBaseAddress(linkId);

    // Read state from LWLIPT HW
    LwU32 linkState = GPU_REG_RD32(lwliptBase + LW_LWLIPT_LNK_CTRL_LINK_STATE_STATUS);
    linkState = DRF_VAL(_LWLIPT_LNK, _CTRL_LINK_STATE_STATUS, _LWRRENTLINKSTATE, linkState);

    switch(linkState)
    {
        case LW_LWLIPT_LNK_CTRL_LINK_STATE_STATUS_LWRRENTLINKSTATE_ACTIVE:
            dprintf("Link is in High Speed (HS) mode\n");
            break;

        case LW_LWLIPT_LNK_CTRL_LINK_STATE_STATUS_LWRRENTLINKSTATE_L2:
            dprintf("Link is in Sleep mode\n");
            break;

        case LW_LWLIPT_LNK_CTRL_LINK_STATE_STATUS_LWRRENTLINKSTATE_CONTAIN:
            dprintf("Link is in Contain mode\n");
            break;

        default:
            // Lwrrently, only ACTIVE, L2 and CONTAIN states are supported
            dprintf("Invalid state\n");
            break;
    }

    // Read the System Ctrl Registers and Locks
    dprintf("\n");
    linkModeCtrl = GPU_REG_RD32(lwliptBase + LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_MODE_CTRL);
    linkLock = GPU_REG_RD32(lwliptBase + LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_MODE_CTRL_LOCK);
    linkClkCtrl = GPU_REG_RD32(lwliptBase + LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CLK_CTRL);
    linkClkCtrlLock = GPU_REG_RD32(lwliptBase + LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CLK_CTRL_LOCK);
    linkChannelCtrl = GPU_REG_RD32(lwliptBase + LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CHANNEL_CTRL);
    linkChannelCtrlLock = GPU_REG_RD32(lwliptBase + LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CHANNEL_CTRL_LOCK);
    linkChannelCtrl2 = GPU_REG_RD32(lwliptBase + LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CHANNEL_CTRL2);
    linkChannelCtrl2Lock = GPU_REG_RD32(lwliptBase + LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CHANNEL_CTRL2_LOCK);
    linkAn1Ctrl = GPU_REG_RD32(lwliptBase + LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_AN1_CTRL);
    linkAn1CtrlLock = GPU_REG_RD32(lwliptBase + LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_AN1_CTRL_LOCK);

    dprintf("LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_MODE_CTRL          : 0x%08X\n", linkModeCtrl);
    dprintf("LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_MODE_CTRL_LOCK     : 0x%08X\n", linkLock);
    dprintf("LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CLK_CTRL           : 0x%08X\n", linkClkCtrl);
    dprintf("LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CLK_CTRL_LOCK      : 0x%08X\n", linkClkCtrlLock);
    dprintf("LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CHANNEL_CTRL       : 0x%08X\n", linkChannelCtrl);
    dprintf("LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CHANNEL_CTRL_LOCK  : 0x%08X\n", linkChannelCtrlLock);
    dprintf("LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CHANNEL_CTRL2      : 0x%08X\n", linkChannelCtrl2);
    dprintf("LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_CHANNEL_CTRL2_LOCK : 0x%08X\n", linkChannelCtrl2Lock);
    dprintf("LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_AN1_CTRL           : 0x%08X\n", linkAn1Ctrl);
    dprintf("LW_LWLIPT_LNK_CTRL_SYSTEM_LINK_AN1_CTRL_LOCK      : 0x%08X\n", linkAn1CtrlLock);
}
