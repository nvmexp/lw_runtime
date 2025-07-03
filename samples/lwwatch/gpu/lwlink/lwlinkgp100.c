/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2014-2020 by LWPU Corporation.  All rights reserved.  All
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

#include "pascal/gp100/dev_top.h"
#include "pascal/gp100/dev_hshub.h"
#include "pascal/gp100/dev_master.h"
#include "pascal/gp100/dev_trim.h"

#include "pascal/gp100/dev_lwl_ioctrl.h"
#include "pascal/gp100/dev_lwl.h"
#include "pascal/gp100/dev_lwltl.h"
#include "pascal/gp100/dev_lwltl_ip.h"
#include "pascal/gp100/dev_lwl_ip.h"

// Singly linked list head used to store link information
static lwlStatus *pLwlStatusArray = NULL;

// String arrays used to make printing enums easier see inc/lwlink.h for more info
static const char *pStatusText[] = {"OFF", "NORMAL", "SAFE", "ERROR"};
static const char *pTypeText[]   = {"P2P", "SYSMEM", "UNKNOWN", "N/A"};
static const char *pPowerText[]  = {"FULL", "OFF"};
static const char *pModeText[]   = {"SAFE", "FAST", "N/A"};

//
// Walks PTOP_DEVICE_INFO registers to discover ioctrls. Since there are no
// ioctrl_ip manuals, the PRI_BASE address discovered is not actually used. Instead
// it's used to set a flag to let the command know that a ioctrl entry has been found
//
LwU32 lwlinkIoctrlDiscovery_GP100(LwU32 *pAddr)
{
    LwU32 i;
    LwU32 deviceInfoType     = DEVICE_INFO_TYPE_ILWALID;
    LwU32 ioctrlPriBase      = 0;
    BOOL  bPTopNextDataValid = FALSE;

    for (i = 0; i < LW_PTOP_DEVICE_INFO__SIZE_1; ++i)
    {
        LwU32 tableEntry = GPU_REG_RD32(LW_PTOP_DEVICE_INFO(i));

        if (FLD_TEST_DRF(_PTOP, _DEVICE_INFO, _ENTRY, _NOT_VALID, tableEntry))
            continue;

        switch (DRF_VAL(_PTOP, _DEVICE_INFO, _ENTRY, tableEntry))
        {
            case LW_PTOP_DEVICE_INFO_ENTRY_DATA:
                if (bPTopNextDataValid)
                {
                    // Get the base address of the IOCTRL device
                    ioctrlPriBase = DRF_VAL(_PTOP, _DEVICE_INFO, _DATA_PRI_BASE,
                        tableEntry) << LW_PTOP_DEVICE_INFO_DATA_PRI_BASE_ALIGN;

                    bPTopNextDataValid = FALSE;
                }
                break;

            case LW_PTOP_DEVICE_INFO_ENTRY_ENGINE_TYPE:
                deviceInfoType = DRF_VAL(_PTOP, _DEVICE_INFO, _TYPE_ENUM, tableEntry);
                break;

            default:
                // Unsupported engine
                break;
        }

        // If the IOCTRL device is discovered
        if (deviceInfoType == LW_PTOP_DEVICE_INFO_TYPE_ENUM_IOCTRL &&
                FLD_TEST_DRF(_PTOP, _DEVICE_INFO, _CHAIN, _ENABLE, tableEntry))
        {
            deviceInfoType = DEVICE_INFO_TYPE_ILWALID;
            bPTopNextDataValid = TRUE;
        }
    }

    return ioctrlPriBase;
}

//
// Walks the ioctrl registers to discover lwlink devices.
// These are then used to determine the PRI_BASE for device addresses.
// This function creates a very simple singly linked list to store the PRI_BASE addresses
//
LwU32 lwlinkLinkDiscovery_GP100(void **pLwlVoid, LwU32 addr)
{
    LwU32 i;
    LwU32 ioctrlDiscoverySize  = 0;
    LwU32 ioctrlInfoType       = DEVICE_INFO_TYPE_ILWALID;
    BOOL  bIoctrlNextDataValid = FALSE;
    LwU32 ioctrlId             = 0;
    LwU32 foundDevices         = 0;
    LwU32 foundIds             = 0;
    LwU32 ioctrlEntry          = 0;
    lwlDiscover **pLwlDiscoverListHead = (lwlDiscover **)pLwlVoid;

    ioctrlEntry    = GPU_REG_RD32(LW_IOCTRL_DISCOVERY(0));
    ioctrlInfoType = DRF_VAL(_IOCTRL, _DISCOVERY_FIELDS, _DEVICE, ioctrlEntry);

    if (ioctrlInfoType == LW_IOCTRL_DISCOVERY_FIELDS_DEVICE_IOCTRL)
    {
        ioctrlEntry = GPU_REG_RD32(LW_IOCTRL_DISCOVERY(1));
        ioctrlDiscoverySize = DRF_VAL(_IOCTRL, _DISCOVERY_FIELDS, _IOCTRL_LENGTH,
                                        ioctrlEntry);
    }
    else
    {
        dprintf("Error: first entry in IOCTRL_DISCOVERY is not IOCTRL!\n");
        return 0;
    }

    for (i = 2; i < ioctrlDiscoverySize; ++i)
    {
        ioctrlEntry = GPU_REG_RD32(LW_IOCTRL_DISCOVERY(i));

        if (FLD_TEST_DRF(_IOCTRL, _DISCOVERY, _ENTRY, _ILWALID, ioctrlEntry))
        {
            dprintf("found invalid ioctrl\n");
            continue;
        }

        switch (DRF_VAL(_IOCTRL, _DISCOVERY, _ENTRY, ioctrlEntry))
        {
            case LW_IOCTRL_DISCOVERY_ENTRY_DATA1:
            {
                if (bIoctrlNextDataValid)
                {
                    lwlDiscover *pTemp = malloc(sizeof(lwlDiscover));
                    pTemp->ID          = ioctrlId;
                    pTemp->addr        = DRF_VAL(_IOCTRL, _DISCOVERY_FIELDS, _PRI_BASE, ioctrlEntry) <<
                                         LW_IOCTRL_DISCOVERY_FIELDS_PRI_BASE_ALIGN;
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

            case LW_IOCTRL_DISCOVERY_ENTRY_DATA2:
            {
                dprintf("found data2. This shouldn't be in use.\n");
                break;
            }

            case LW_IOCTRL_DISCOVERY_ENTRY_ENUM:
            {
                ioctrlInfoType = DRF_VAL(_IOCTRL, _DISCOVERY_FIELDS, _DEVICE, ioctrlEntry);
                ioctrlId       = DRF_VAL(_IOCTRL, _DISCOVERY_FIELDS, _ID, ioctrlEntry);

                if (ioctrlInfoType == LW_IOCTRL_DISCOVERY_FIELDS_DEVICE_LWLTL ||
                    ioctrlInfoType == LW_IOCTRL_DISCOVERY_FIELDS_DEVICE_LWLINK)
                {
                    if (DRF_VAL(_IOCTRL, _DISCOVERY, _CHAIN, ioctrlEntry) ==
                            LW_IOCTRL_DISCOVERY_CHAIN_ENABLE)
                    {
                        bIoctrlNextDataValid = TRUE;
                    }
                    else
                    {
                        dprintf("Warning: Found lwltl or lwlink in ioctrl, but there is"
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

// Get the link status
LW_STATUS lwlinkGetStatus(LwU32 statusArraySize, void *pLwlVoid)
{
    lwlStatus *pLwlStatusArray = (lwlStatus *)pLwlVoid;
    LwU32      intenReg        = 0;
    LwU32      i;

    for (i = 0; i < statusArraySize; i++)
    {
        if (pLwlStatusArray[i].power == LWL_POFF)
        {
            continue;
        }

        intenReg = GPU_REG_RD32(LW_LWLTL_TL_INTEN_REG + pLwlStatusArray[i].tlPriBase);
        intenReg &= LW_PLWLTL_INTEN_REG_MASK;

        if (intenReg != 0)
        {
            pLwlStatusArray[i].status = LWL_ERROR;
        }
        else if (pLwlStatusArray[i].rxMode == LWL_SAFE ||
                 pLwlStatusArray[i].txMode == LWL_SAFE)
        {
            pLwlStatusArray[i].status = LWL_S_SAFE;
        }
        else if (pLwlStatusArray[i].rxMode == LWL_FAST &&
                 pLwlStatusArray[i].txMode == LWL_FAST &&
                 pLwlStatusArray[i].power  == LWL_FULL &&
                 (pLwlStatusArray[i].type == LWL_P2P ||
                  pLwlStatusArray[i].type == LWL_SYSMEM))
        {
            pLwlStatusArray[i].status = LWL_NORMAL;
        }
    }

    return LW_OK;
}

// Get the link power states
LW_STATUS lwlinkGetPowerStatus(LwU32 statusArraySize, void *pLwlVoid)
{
    lwlStatus *pLwlStatusArray = (lwlStatus *)pLwlVoid;
    LwU32      ioctrlReset     = GPU_REG_RD32(LW_IOCTRL_RESET);
    LwU32      linkReset       = DRF_VAL(_IOCTRL, _RESET, _LINKRESET, ioctrlReset);
    LwU32      i, linkNum;

    for (i = 0; i < statusArraySize; i++)
    {
        linkNum = pLwlStatusArray[i].linkNum;

        if ((linkReset >> linkNum) & 1)
        {
            pLwlStatusArray[i].power = LWL_FULL;
        }
        else
        {
            dprintf("Link %d is powered off. "
                    "No useful information will be displayed \n", linkNum);
        }
    }

    return LW_OK;
}

//
// lwlinkGetHshubStatus
//     Gets HSHUB status for existing links - whether a link is connected
//     to system or a peer GPU
//
LW_STATUS lwlinkGetHshubStatus(LwU32 statusArraySize, void *pLwlVoid)
{
    LwU32 i, j, linkNum, hsHubConfig1, hsHubConfig2;

    lwlStatus *pLwlStatusArray = (lwlStatus *)pLwlVoid;
    LwU32      hshubConfig0    = GPU_REG_RD32(LW_PFB_HSHUB_CONFIG0);
    LwU32      sysMemStatus    = DRF_VAL(_PFB_HSHUB, _CONFIG0, _SYSMEM_LWLINK_MASK, hshubConfig0);
    LwU32      totalPeers      = LW_PFB_HSHUB_CONFIG1_PEER_LWLINK_MASK__SIZE_1 +
                                 LW_PFB_HSHUB_CONFIG2_PEER_LWLINK_MASK__SIZE_1;

    // Allocate memory for total peers available
    LwU32 *pHsHubConfigPeer = malloc(totalPeers * sizeof(LwU32));

    if (pHsHubConfigPeer == NULL)
    {
        dprintf("malloc failed while trying to read the hshub registers\n");
        return LW_ERR_GENERIC;
    }

    // Get the mask of links associated with peers0-3
    hsHubConfig1 = GPU_REG_RD32(LW_PFB_HSHUB_CONFIG1);
    pHsHubConfigPeer[0] = DRF_VAL(_PFB_HSHUB_CONFIG1_PEER, _0, _LWLINK_MASK, hsHubConfig1);
    pHsHubConfigPeer[1] = DRF_VAL(_PFB_HSHUB_CONFIG1_PEER, _1, _LWLINK_MASK, hsHubConfig1);
    pHsHubConfigPeer[2] = DRF_VAL(_PFB_HSHUB_CONFIG1_PEER, _2, _LWLINK_MASK, hsHubConfig1);
    pHsHubConfigPeer[3] = DRF_VAL(_PFB_HSHUB_CONFIG1_PEER, _3, _LWLINK_MASK, hsHubConfig1);

    // Get the mask of links associated with peers4-7
    hsHubConfig2 = GPU_REG_RD32(LW_PFB_HSHUB_CONFIG2);
    pHsHubConfigPeer[4] = DRF_VAL(_PFB_HSHUB_CONFIG2_PEER, _4, _LWLINK_MASK, hsHubConfig2);
    pHsHubConfigPeer[5] = DRF_VAL(_PFB_HSHUB_CONFIG2_PEER, _5, _LWLINK_MASK, hsHubConfig2);
    pHsHubConfigPeer[6] = DRF_VAL(_PFB_HSHUB_CONFIG2_PEER, _6, _LWLINK_MASK, hsHubConfig2);
    pHsHubConfigPeer[7] = DRF_VAL(_PFB_HSHUB_CONFIG2_PEER, _7, _LWLINK_MASK, hsHubConfig2);

    for (i = 0; i < statusArraySize; i++)
    {
        if (pLwlStatusArray[i].power == LWL_POFF)
        {
            continue;
        }

        linkNum = pLwlStatusArray[i].linkNum;

        // Check if the link is a sysmem link
        if ((sysMemStatus >> linkNum) & 1)
        {
            pLwlStatusArray[i].type = LWL_SYSMEM;
        }

        // Check if the link is a peer link
        for (j = 0; j < totalPeers; j++)
        {
            if ((pHsHubConfigPeer[j] >> linkNum) & 1)
            {
                if (pLwlStatusArray[i].type == LWL_SYSMEM)
                {
                    dprintf("Link was already set as SYSMEM, but is also "
                            "set in peer mode. There must be an error\n");
                    pLwlStatusArray[i].type = LWL_UNKNOWN;
                }
                else
                {
                    pLwlStatusArray[i].type = LWL_P2P;
                    pLwlStatusArray[i].peer = j;
                }
            }
        }

        if (pLwlStatusArray[i].type == LWL_T_NA)
        {
            dprintf("Link %d is neither in SYSMEM nor in P2P mode."
                    " Perhaps it is uninitialized?\n", linkNum);
        }
    }

    free(pHsHubConfigPeer);

    return LW_OK;
}

// Get the TX and RX status of the sublinks
LW_STATUS lwlinkGetRxTxMode(LwU32 statusArraySize, void *pLwlVoid)
{
    LwU32      i, linkNum;
    lwlStatus *pLwlStatusArray = (lwlStatus *)pLwlVoid;
    LwU32      txStatus        = 0;
    LwU32      rxStatus        = 0;
    LwU32      txMode          = 0;
    LwU32      rxMode          = 0;

    for (i = 0; i < statusArraySize; i++)
    {
        if (pLwlStatusArray[i].power == LWL_POFF)
        {
            continue;
        }

        linkNum = pLwlStatusArray[i].linkNum;

        // Get the TX SLSM status of the link
        txStatus = GPU_REG_RD32(LW_PLWL_SL0_SLSM_STATUS_TX + pLwlStatusArray[i].dlPriBase);
        txMode   = DRF_VAL(_PLWL, _SL0, _SLSM_STATUS_TX_PRIMARY_STATE, txStatus);

        if (txMode == LW_PLWL_SL0_SLSM_STATUS_TX_PRIMARY_STATE_HS)
        {
            pLwlStatusArray[i].txMode = LWL_FAST;
        }
        else if (txMode == LW_PLWL_SL0_SLSM_STATUS_TX_PRIMARY_STATE_SAFE)
        {
            pLwlStatusArray[i].txMode = LWL_SAFE;
        }
        else
        {
            dprintf("TX mode for link %d is neither fast nor safe. It may be uninitialized."
                    " Leaving it at the default value of N/A\n", linkNum);
        }

        // Get the RX SLSM status of the link
        rxStatus = GPU_REG_RD32(LW_PLWL_SL1_SLSM_STATUS_RX + pLwlStatusArray[i].dlPriBase);
        rxMode   = DRF_VAL(_PLWL, _SL1, _SLSM_STATUS_RX_PRIMARY_STATE, rxStatus);

        if (rxMode == LW_PLWL_SL1_SLSM_STATUS_RX_PRIMARY_STATE_HS)
        {
            pLwlStatusArray[i].rxMode = LWL_FAST;
        }
        else if (rxMode == LW_PLWL_SL1_SLSM_STATUS_RX_PRIMARY_STATE_SAFE)
        {
            pLwlStatusArray[i].rxMode = LWL_SAFE;
        }
        else
        {
            dprintf("RX mode for link %d is neither fast nor safe. It may be uninitialized."
                    " Leaving it at the default value of N/A\n", linkNum);
        }
    }

    return LW_OK;
}

//
// lwlinkPrintHshubConfig_GP100
//     Print the HSHUB configuration
//     LW_PFB_HSHUB_CONFIG0/1/2 capture all the configuration
//
void lwlinkPrintHshubConfig_GP100(void)
{
    LwU32 config0, config1, config2, i;

    config0 = GPU_REG_RD32(LW_PFB_HSHUB_CONFIG0);
    config1 = GPU_REG_RD32(LW_PFB_HSHUB_CONFIG1);
    config2 = GPU_REG_RD32(LW_PFB_HSHUB_CONFIG2);

    dprintf("\n");

    dprintf("LW_PFB_HSHUB_CONFIG0 = 0x%x\n", config0);
    dprintf("LW_PFB_HSHUB_CONFIG1 = 0x%x\n", config1);
    dprintf("LW_PFB_HSHUB_CONFIG2 = 0x%x\n", config2);

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
        dprintf("Mask of links used by PEER%u = 0x%x\n", i,
                 DRF_VAL(_PFB_HSHUB, _CONFIG1, _PEER_LWLINK_MASK(i), config1));
    }
    for (i = 0; i < LW_PFB_HSHUB_CONFIG2_PEER_LWLINK_MASK__SIZE_1; i++)
    {
        dprintf("Mask of links used by PEER%u = 0x%x\n",
                 i + LW_PFB_HSHUB_CONFIG1_PEER_LWLINK_MASK__SIZE_1,
                 DRF_VAL(_PFB_HSHUB, _CONFIG2, _PEER_LWLINK_MASK(i), config2));
    }
    dprintf("\n");
}

//
// lwlinkPrintHshubIdleStatus_GP100
//     Print the HSHUB idle status
//
void lwlinkPrintHshubIdleStatus_GP100(void)
{
    dprintf("LW_PFB_HSHUB_IG_IDLE0 = 0x%x\n", GPU_REG_RD32(LW_PFB_HSHUB_IG_IDLE0));
    dprintf("LW_PFB_HSHUB_IG_IDLE1 = 0x%x\n", GPU_REG_RD32(LW_PFB_HSHUB_IG_IDLE1));
    dprintf("LW_PFB_HSHUB_EG_IDLE0 = 0x%x\n", GPU_REG_RD32(LW_PFB_HSHUB_EG_IDLE0));
    dprintf("LW_PFB_HSHUB_RR_IDLE0 = 0x%x\n", GPU_REG_RD32(LW_PFB_HSHUB_RR_IDLE0));
    dprintf("LW_PFB_HSHUB_RR_IDLE1 = 0x%x\n", GPU_REG_RD32(LW_PFB_HSHUB_RR_IDLE1));
}

//
// lwlinkLogHshubErrors_GP100
//
void lwlinkLogHshubErrors_GP100(void)
{
    dprintf("LW_PFB_HSHUB_IG_ERROR = 0x%x\n", GPU_REG_RD32(LW_PFB_HSHUB_IG_ERROR));
    dprintf("LW_PFB_HSHUB_EG_ERROR = 0x%x\n", GPU_REG_RD32(LW_PFB_HSHUB_EG_ERROR));
}

//
// lwlinkPrintHshubReqTimeoutInfo_GP100
//
void lwlinkPrintHshubReqTimeoutInfo_GP100(void)
{
    LwU32 i, regVal;

    for (i = 0; i < LW_PFB_HSHUB_LWLTL_REQ_TIMEOUT_CFG__SIZE_1; i++)
    {
        regVal = GPU_REG_RD32( LW_PFB_HSHUB_LWLTL_REQ_TIMEOUT_CFG(i) );
        if (FLD_TEST_DRF(_PFB_HSHUB, _LWLTL_REQ_TIMEOUT_CFG, _ENABLE, _ON, regVal) &&
            FLD_TEST_DRF(_PFB_HSHUB, _LWLTL_REQ_TIMEOUT_CFG, _STATUS, _PENDING, regVal))
        {
            dprintf("Dumping request timeout info for link %d:\n", i);

            regVal = GPU_REG_RD32( LW_PFB_HSHUB_LWLTL_REQ_TIMEOUT_0_INFO(i) );
            dprintf("    LW_PFB_HSHUB_LWLTL_REQ_TIMEOUT_0_INFO(%d) = 0x%x\n", i,
                    DRF_VAL(_PFB_HSHUB, _LWLTL_REQ_TIMEOUT_0_INFO, _DATA, regVal));

            regVal = GPU_REG_RD32( LW_PFB_HSHUB_LWLTL_REQ_TIMEOUT_1_INFO(i) );
            dprintf("    LW_PFB_HSHUB_LWLTL_REQ_TIMEOUT_1_INFO(%d) = 0x%x\n", i,
                    DRF_VAL(_PFB_HSHUB, _LWLTL_REQ_TIMEOUT_1_INFO, _DATA, regVal));

            regVal = GPU_REG_RD32( LW_PFB_HSHUB_LWLTL_REQ_TIMEOUT_2_INFO(i) );
            dprintf("    LW_PFB_HSHUB_LWLTL_REQ_TIMEOUT_2_INFO(%d) = 0x%x\n", i,
                    DRF_VAL(_PFB_HSHUB, _LWLTL_REQ_TIMEOUT_2_INFO, _DATA, regVal));
            dprintf("\n");
        }
    }
}

//
// lwlinkEnableHshubLogging_GP100
//
void lwlinkEnableHshubLogging_GP100(void)
{
    LwU32 regVal, i;

    for (i = 0; i < LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_CFG__SIZE_1; i++)
    {
        dprintf("Current logging support for Link %d:\n", i);
        regVal = GPU_REG_RD32(LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_CFG(i));
        dprintf("    LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_CFG_ENABLE = 0x%x\n",
                DRF_VAL(_PFB_HSHUB, _LWLTL_READ_TIMEOUT_CFG, _ENABLE, regVal));
        dprintf("    LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_CFG_ENHANCED_MODE = 0x%x\n",
                DRF_VAL(_PFB_HSHUB, _LWLTL_READ_TIMEOUT_CFG, _ENHANCED_MODE, regVal));
        regVal = GPU_REG_RD32(LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_ACCESS(i));
        dprintf("    LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_ACCESS_ENABLE = 0x%x\n",
                DRF_VAL(_PFB_HSHUB, _LWLTL_READ_TIMEOUT_ACCESS, _ENABLE, regVal));
        regVal = GPU_REG_RD32(LW_PFB_HSHUB_RR_DEBUG_CFG);
        dprintf("    LW_PFB_HSHUB_RR_DEBUG_CFG_ADDR_VISIBLE = 0x%x\n",
                DRF_VAL(_PFB_HSHUB, _RR_DEBUG_CFG, _ADDR_VISIBLE, regVal));

        dprintf("Enabling logging support for Link %d:\n", i);
        regVal = GPU_REG_RD32(LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_ACCESS(i));
        regVal = FLD_SET_DRF_NUM(_PFB_HSHUB, _LWLTL_READ_TIMEOUT_ACCESS, _ENABLE, 1, regVal);
        GPU_REG_WR32(LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_ACCESS(i), regVal);
        dprintf("\n");
    }

    regVal = GPU_REG_RD32(LW_PFB_HSHUB_RR_DEBUG_CFG);
    regVal = FLD_SET_DRF(_PFB_HSHUB, _RR_DEBUG_CFG, _ADDR_VISIBLE, _ON, regVal);
    GPU_REG_WR32(LW_PFB_HSHUB_RR_DEBUG_CFG, regVal);
}

//
// lwlinkPrintHshubReadTimeoutInfo_GP100
//
void lwlinkPrintHshubReadTimeoutInfo_GP100(void)
{
    LwU32 regVal, i;

    for (i = 0; i < LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_ACCESS__SIZE_1; i++)
    {
        dprintf("Read timeout logging for Link %d:\n", i);
        regVal = GPU_REG_RD32(LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_ACCESS(i));
        dprintf("    LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_ACCESS_ADR = 0x%x\n",
                DRF_VAL(_PFB_HSHUB, _LWLTL_READ_TIMEOUT_ACCESS, _ADR, regVal));
        dprintf("    LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_ACCESS_TIMESTAMP = 0x%x\n",
                DRF_VAL(_PFB_HSHUB, _LWLTL_READ_TIMEOUT_ACCESS, _TIMESTAMP, regVal));
        regVal = GPU_REG_RD32(LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_0_INFO(i));
        dprintf("    LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_0_INFO_DATA = 0x%x\n",
                DRF_VAL(_PFB_HSHUB, _LWLTL_READ_TIMEOUT_0_INFO, _DATA, regVal));
        regVal = GPU_REG_RD32(LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_1_INFO(i));
        dprintf("    LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_1_INFO_DATA = 0x%x\n",
                DRF_VAL(_PFB_HSHUB, _LWLTL_READ_TIMEOUT_1_INFO, _DATA, regVal));
        regVal = GPU_REG_RD32(LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_2_INFO(i));
        dprintf("    LW_PFB_HSHUB_LWLTL_READ_TIMEOUT_2_INFO_DATA = 0x%x\n",
                DRF_VAL(_PFB_HSHUB, _LWLTL_READ_TIMEOUT_2_INFO, _DATA, regVal));
        dprintf("\n");
    }
}

void lwlinkPrintVerbose_GP100(BOOL bPretty)
{
    LwU32 val;
    LwU32 i;
    LwU32 linksDiscovered = 0;
    LwU32 statusArraySize = 0;
    lwlDiscover *pLwlDiscoverList = NULL;


    linksDiscovered = pLwlink[indexGpu].lwlinkLinkDiscovery((void *)&pLwlDiscoverList, 0);
    if (linksDiscovered == 0)
    {
        dprintf("No links were detected during the discovery process in ioctrl\n");
        return;
    }

    if (initLwlStatusArray(&statusArraySize, linksDiscovered, pLwlDiscoverList) != LW_OK)
    {
        dprintf("Error encountered while initializing the status array\n");
        return;
    }


    dprintf("***************************************************************\n");
    dprintf("======================\n");
    dprintf("Detailed LWlink Status\n");
    dprintf("======================\n\n");

    
    // print PMC Enable register
    val = GPU_REG_RD32( LW_PMC_ENABLE );
    dprintf("LW_PMC_ENABLE                      val = 0x%x\n", val);

    // Print LWLink PTOP entry
    for (i = 0; i < LW_PTOP_DEVICE_INFO__SIZE_1; ++i)
    {
        val = GPU_REG_RD32( LW_PTOP_DEVICE_INFO(i) );

        if (FLD_TEST_DRF( _PTOP, _DEVICE_INFO, _ENTRY, _NOT_VALID, val))
            continue;

        if (DRF_VAL( _PTOP, _DEVICE_INFO, _ENTRY, val) ==
                LW_PTOP_DEVICE_INFO_ENTRY_ENGINE_TYPE 
                && FLD_TEST_DRF( _PTOP, _DEVICE_INFO, _CHAIN, _ENABLE, val))
        {
            val = GPU_REG_RD32( LW_PTOP_DEVICE_INFO(i + 1) );

            dprintf("LWlink PTOP IOCTRL Entry:          val = 0x%x\n", val);
            break;
        }
    }

    val = GPU_REG_RD32( LW_IOCTRL_STALL_INTR_STATUS );
    dprintf("LW_IOCTRL_STALL_INTR_STATUS        val = 0x%x\n", val);
    val = GPU_REG_RD32( LW_IOCTRL_NONSTALL_INTR_STATUS );
    dprintf("LW_IOCTRL_NONSTALL_INTR_STATUS     val = 0x%x\n", val);


    val = GPU_REG_RD32( LW_IOCTRL_RESET );
    dprintf("LW_IOCTRL_RESET LINKDISABLE = 0x%x    LINKRESET = 0x%x\n",
            DRF_VAL(_IOCTRL, _RESET, _LINKDISABLE, val),
            DRF_VAL(_IOCTRL, _RESET, _LINKRESET, val)
           );


    for (i = 0; i < statusArraySize; i++)
    {
        val = GPU_REG_RD32( LW_IOCTRL_RESET );
        if (!(DRF_VAL(_IOCTRL, _RESET, _LINKRESET, val) & BIT(i)))
        {
            dprintf("Skipping link %d\n", i);
            continue;
        }

        dprintf("==========================\n");
        dprintf("===       LINK %d      ====\n", i);
        dprintf("==========================\n");

        
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_ERROR_COUNT1 );
        dprintf("LW_PLWL_ERROR_COUNT1               val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_SL0_ERROR_COUNT4 );
        dprintf("LW_PLWL_SL0_ERROR_COUNT4           val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_SL1_ERROR_COUNT1 );
        dprintf("LW_PLWL_SL1_ERROR_COUNT1           val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_SL1_ERROR_COUNT2_LANECRC );
        dprintf("LW_PLWL_SL1_ERROR_COUNT2_LANECRC   val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_SL1_ERROR_COUNT3_LANECRC );
        dprintf("LW_PLWL_SL1_ERROR_COUNT3_LANECRC   val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_SL1_ERROR_COUNT5 );
        dprintf("LW_PLWL_SL1_ERROR_COUNT5           val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_INTR );
        dprintf("LW_PLWL_INTR                       val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_INTR_STALL_EN );
        dprintf("LW_PLWL_INTR_STALL_EN              val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_INTR_NONSTALL_EN );
        dprintf("LW_PLWL_INTR_NONSTALL_EN           val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_LINK_STATE );
        dprintf("LW_PLWL_LINK_STATE                 val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXCTRL_REG );
        dprintf("LW_LWLTL_TL_TXCTRL_REG             val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXFLOW_REG );
        dprintf("LW_LWLTL_TL_TXFLOW_REG             val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_RXCTRL_REG );
        dprintf("LW_LWLTL_TL_RXCTRL_REG             val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_RXFLOW_REG );
        dprintf("LW_LWLTL_TL_RXFLOW_REG             val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_ERRLOG_REG );
        dprintf("LW_LWLTL_TL_ERRLOG_REG             val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_ERRCTRL_REG );
        dprintf("LW_LWLTL_TL_ERRCTRL_REG            val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_RXPROTERR_REG );
        dprintf("LW_LWLTL_TL_RXPROTERR_REG          val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXCREDITCNT0_REG );
        dprintf("LW_LWLTL_TL_TXCREDITCNT0_REG       val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXCREDITCNT1_REG );
        dprintf("LW_LWLTL_TL_TXCREDITCNT1_REG       val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXCREDITLIMIT0_REG );
        dprintf("LW_LWLTL_TL_TXCREDITLIMIT0_REG     val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXCREDITLIMIT1_REG );
        dprintf("LW_LWLTL_TL_TXCREDITLIMIT1_REG     val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXCREDITLIMIT2_REG );
        dprintf("LW_LWLTL_TL_TXCREDITLIMIT2_REG     val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXDBG0_REG );
        dprintf("LW_LWLTL_TL_TXDBG0_REG             val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXDBG1_REG );
        dprintf("LW_LWLTL_TL_TXDBG1_REG             val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXDBG2_REG );
        dprintf("LW_LWLTL_TL_TXDBG2_REG             val = 0x%x\n", val);
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXDBG3_REG );
        dprintf("LW_LWLTL_TL_TXDBG3_REG             val = 0x%x\n", val);
    }
    val = GPU_REG_RD32( LW_PTRIM_SYS_LWLINK_UPHY_OUTPUT );
    dprintf("LW_PTRIM_SYS_LWLINK_UHY_OUTPUT     val = 0x%x\n", val);
  
    

    if (!bPretty)
        goto lwlinkPrintVerboseCleanup;

    dprintf("================\n");
    dprintf("IOCTRL Registers\n");
    dprintf("================\n");

    // Print stalling interrupt status
    val = GPU_REG_RD32( LW_IOCTRL_STALL_INTR_STATUS );
    dprintf("IOCTRL DL Stalling interrupt status: 0x%x\n",
            DRF_VAL( _IOCTRL, _STALL_INTR_STATUS, _DL, val )
           );
    dprintf("IOCTRL TL Stalling interrupt status: 0x%x\n",
            DRF_VAL( _IOCTRL, _STALL_INTR_STATUS, _TL, val )
           );

    // print nonstalling interrupt status
    val = GPU_REG_RD32( LW_IOCTRL_NONSTALL_INTR_STATUS );
    dprintf("IOCTRL DL Nonstalling Interrupt Status: 0x%x\n",
            DRF_VAL( _IOCTRL, _NONSTALL_INTR_STATUS, _DL, val)
           );

    for (i = 0; i < statusArraySize; i++)
    {
        val = GPU_REG_RD32( LW_IOCTRL_RESET );
        if (!(DRF_VAL(_IOCTRL, _RESET, _LINKRESET, val) & BIT(i)))
        {
            dprintf("Skipping link %d\n", i);
            continue;
        }

        dprintf("==========================\n");
        dprintf("===       LINK %d      ====\n", i);
        dprintf("==========================\n");

        dprintf("==============\n");
        dprintf("PLWL Registers\n");
        dprintf("==============\n");

        // print count of recovery events encountered
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_ERROR_COUNT1 );
        dprintf("PLWL Recovery Events: 0x%x\n",
                DRF_VAL( _PLWL, _ERROR_COUNT1, _RECOVERY_EVENTS, val)
               );

        // print count of replay events encountered
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_SL0_ERROR_COUNT4 );
        dprintf("PLWL Replay Events: 0x%x\n",
                DRF_VAL( _PLWL, _SL0_ERROR_COUNT4, _REPLAY_EVENTS, val)
               );

        // print count of CRC errors
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_SL1_ERROR_COUNT1 );
        dprintf("PLWL CRC errors: 0x%x\n",
                DRF_VAL( _PLWL, _SL1_ERROR_COUNT1, _FLIT_CRC_ERRORS, val)
               );

        // Print errors detected by per-lane CRC, if enabled
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_SL1_ERROR_COUNT2_LANECRC );
        dprintf("SL1 Lane CRC Errors:    L0: 0x%x    L1: 0x%x    L2: 0x%x    L3: 0x%x\n",
                DRF_VAL( _PLWL, _SL1_ERROR_COUNT2_LANECRC, _L0, val),
                DRF_VAL( _PLWL, _SL1_ERROR_COUNT2_LANECRC, _L1, val),
                DRF_VAL( _PLWL, _SL1_ERROR_COUNT2_LANECRC, _L2, val),
                DRF_VAL( _PLWL, _SL1_ERROR_COUNT2_LANECRC, _L3, val)
               );
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_SL1_ERROR_COUNT3_LANECRC );
        dprintf("                        L4: 0x%x    L5: 0x%x    L6: 0x%x    L7: 0x%x\n",
                DRF_VAL( _PLWL, _SL1_ERROR_COUNT3_LANECRC, _L4, val),
                DRF_VAL( _PLWL, _SL1_ERROR_COUNT3_LANECRC, _L5, val),
                DRF_VAL( _PLWL, _SL1_ERROR_COUNT3_LANECRC, _L6, val),
                DRF_VAL( _PLWL, _SL1_ERROR_COUNT3_LANECRC, _L7, val)
               );

        // Print SL1 lane errors
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_SL1_ERROR_COUNT5 );
        dprintf("SL1 Lane Errors:\n\tShort Rate Count: 0x%x\n\tLong Rate Count: 0x%x\n",
                DRF_VAL( _PLWL, _SL1_ERROR_COUNT5, _SHORT_RATE_COUNTER, val),
                DRF_VAL( _PLWL, _SL1_ERROR_COUNT5, _LONG_RATE_COUNTER, val)
               );

        // Print active interrupts
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_INTR );
        dprintf("PLWL Active Interrupts: \n\t%s%s%s%s%s%s%s%s%s%s%s%s%s\n",
                DRF_VAL( _PLWL, _INTR, _TX_REPLAY, val) ? "TX Replay\n\t" : "",
                DRF_VAL( _PLWL, _INTR, _TX_RECOVERY_SHORT, val) ? "TX Short Recovery\n\t" : "",
                DRF_VAL( _PLWL, _INTR, _TX_RECOVERY_LONG, val) ? "TX Long Recovery\n\t" : "",
                DRF_VAL( _PLWL, _INTR, _TX_FAULT_RAM, val) ? "TX RAM Fault\n\t" : "",
                DRF_VAL( _PLWL, _INTR, _TX_FAULT_INTERFACE, val) ? "TX Interface Fault\n\t" : "",
                DRF_VAL( _PLWL, _INTR, _TX_FAULT_SUBLINK_CHANGE, val) ? "TX Sublink Change Fault\n\t" : "",
                DRF_VAL( _PLWL, _INTR, _RX_FAULT_SUBLINK_CHANGE, val) ? "RX Sublink Change Fault\n\t" : "",
                DRF_VAL( _PLWL, _INTR, _RX_FAULT_DL_PROTOCOL, val) ? "RX DL Protocol Fault\n\t" : "",
                DRF_VAL( _PLWL, _INTR, _RX_SHORT_ERROR_RATE, val) ? "RX Short Error Rate\n\t" : "",
                DRF_VAL( _PLWL, _INTR, _RX_LONG_ERROR_RATE, val) ? "RX Long Error Rate\n\t" : "",
                DRF_VAL( _PLWL, _INTR, _RX_ILA_TRIGGER, val) ? "RX ILA Trigger\n\t" : "",
                DRF_VAL( _PLWL, _INTR, _LTSSM_FAULT, val) ? "LTSSM Fault\n\t" : "",
                DRF_VAL( _PLWL, _INTR, _LTSSM_PROTOCOL, val) ? "LTSSM Protocol\n\t" : ""
               );

        // Print status of stalling interrupts
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_INTR_STALL_EN );
        dprintf("PLWL Stalling Interrupts:\n|%-23s|%-8s|\n", "Name", "Status");
        dprintf("|%-23s|%-8s|\n", "TX_REPLAY",
                DRF_VAL( _PLWL, _INTR_STALL_EN, _TX_REPLAY, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "TX_RECOVERY_SHORT",
                DRF_VAL( _PLWL, _INTR_STALL_EN, _TX_RECOVERY_SHORT, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "TX_RECOVERY_LONG",
                DRF_VAL( _PLWL, _INTR_STALL_EN, _TX_RECOVERY_LONG, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "TX_FAULT_RAM",
                DRF_VAL( _PLWL, _INTR_STALL_EN, _TX_FAULT_RAM, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "TX_FAULT_INTERFACE",
                DRF_VAL( _PLWL, _INTR_STALL_EN, _TX_FAULT_INTERFACE, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "TX_FAULT_SUBLINK_CHANGE",
                DRF_VAL( _PLWL, _INTR_STALL_EN, _TX_FAULT_SUBLINK_CHANGE, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "RX_FAULT_SUBLINK_CHANGE",
                DRF_VAL( _PLWL, _INTR_STALL_EN, _RX_FAULT_SUBLINK_CHANGE, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "RX_FAULT_DL_PROTOCOL",
                DRF_VAL( _PLWL, _INTR_STALL_EN, _RX_FAULT_DL_PROTOCOL, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "RX_SHORT_ERROR_RATE",
                DRF_VAL( _PLWL, _INTR_STALL_EN, _RX_SHORT_ERROR_RATE, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "RX_LONG_ERROR_RATE",
                DRF_VAL( _PLWL, _INTR_STALL_EN, _RX_LONG_ERROR_RATE, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "RX_ILA_TRIGGER",
                DRF_VAL( _PLWL, _INTR_STALL_EN, _RX_ILA_TRIGGER, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "LTSSM_FAULT",
                DRF_VAL( _PLWL, _INTR_STALL_EN, _LTSSM_FAULT, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "LTSSM_PROTOCOL",
                DRF_VAL( _PLWL, _INTR_STALL_EN, _LTSSM_PROTOCOL, val ) ? "Enabled" : "Disabled"
               );

        // Print status of nonstalling interrupts
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_INTR_NONSTALL_EN );
        dprintf("LW_PLWL_INTR_NONSTALL_EN val = 0x%x\n", val);
        dprintf("|%-23s|%-8s|\n", "TX_REPLAY",
                DRF_VAL( _PLWL, _INTR_NONSTALL_EN, _TX_REPLAY, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "TX_RECOVERY_SHORT",
                DRF_VAL( _PLWL, _INTR_NONSTALL_EN, _TX_RECOVERY_SHORT, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "TX_RECOVERY_LONG",
                DRF_VAL( _PLWL, _INTR_NONSTALL_EN, _TX_RECOVERY_LONG, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "TX_FAULT_RAM",
                DRF_VAL( _PLWL, _INTR_NONSTALL_EN, _TX_FAULT_RAM, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "TX_FAULT_INTERFACE",
                DRF_VAL( _PLWL, _INTR_NONSTALL_EN, _TX_FAULT_INTERFACE, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "TX_FAULT_SUBLINK_CHANGE",
                DRF_VAL( _PLWL, _INTR_NONSTALL_EN, _TX_FAULT_SUBLINK_CHANGE, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "RX_FAULT_SUBLINK_CHANGE",
                DRF_VAL( _PLWL, _INTR_NONSTALL_EN, _RX_FAULT_SUBLINK_CHANGE, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "RX_FAULT_DL_PROTOCOL",
                DRF_VAL( _PLWL, _INTR_NONSTALL_EN, _RX_FAULT_DL_PROTOCOL, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "RX_SHORT_ERROR_RATE",
                DRF_VAL( _PLWL, _INTR_NONSTALL_EN, _RX_SHORT_ERROR_RATE, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "RX_LONG_ERROR_RATE",
                DRF_VAL( _PLWL, _INTR_NONSTALL_EN, _RX_LONG_ERROR_RATE, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "RX_ILA_TRIGGER",
                DRF_VAL( _PLWL, _INTR_NONSTALL_EN, _RX_ILA_TRIGGER, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "LTSSM_FAULT",
                DRF_VAL( _PLWL, _INTR_NONSTALL_EN, _LTSSM_FAULT, val ) ? "Enabled" : "Disabled"
               );
        dprintf("|%-23s|%-8s|\n", "LTSSM_PROTOCOL",
                DRF_VAL( _PLWL, _INTR_NONSTALL_EN, _LTSSM_PROTOCOL, val ) ? "Enabled" : "Disabled"
               );

        // Print current link state, as well as the debug substate
        val = GPU_REG_RD32( pLwlStatusArray[i].dlPriBase + LW_PLWL_LINK_STATE );
        dprintf("PLWL Current Link State: ");
        switch ( DRF_VAL( _PLWL, _LINK_STATE, _STATE, val) )
        {
            case LW_PLWL_LINK_STATE_STATE_INIT:
                dprintf("Initialization\n");
                break;
            case LW_PLWL_LINK_STATE_STATE_HWCFG:
                dprintf("HWCFG\n");
                break;
            case LW_PLWL_LINK_STATE_STATE_SWCFG:
                dprintf("SWCFG\n");
                break;
            case LW_PLWL_LINK_STATE_STATE_ACTIVE:
                dprintf("Active\n");
                break;
            case LW_PLWL_LINK_STATE_STATE_FAULT:
                dprintf("Fault\n");
                break;
            case LW_PLWL_LINK_STATE_STATE_RECOVERY:
                dprintf("Recovery\n");
                break;
            default:
                dprintf("Invalid\n");
                break;
        }
        dprintf("Substate: 0x%x\n",
                DRF_VAL( _PLWL, _LINK_STATE, _DBG_SUBSTATE, val)
               );


        dprintf("=================\n");
        dprintf("LWLTL Registers\n");
        dprintf("=================\n");

        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXCTRL_REG );
        dprintf("TX Request Fifo Max Entry: 0x%x\n",
                ((DRF_VAL( _PLWLTL0, _TL_TXCTRL_REG, _REQFIFOMAX, val ) << 2) | 0x3)
               );
        dprintf("TX Response Fifo Max Entry: 0x%x\n",
                ((DRF_VAL( _PLWLTL0, _TL_TXCTRL_REG, _RESPFIFOMAX, val ) << 2) | 0x3)
               );
        dprintf("TX Force AE: 0x%x       Force BE: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXCTRL_REG, _FORCEAE, val ),
                DRF_VAL( _PLWLTL0, _TL_TXCTRL_REG, _FORCEBE, val )
               ); 

        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXFLOW_REG );
        dprintf("HSHUB Available request credits: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXFLOW_REG, _REQCREDITS, val )
               );
        dprintf("HSHUB Available response credits: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXFLOW_REG, _RESPCREDITS, val )
               );

        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_RXCTRL_REG );
        dprintf("RX Response Data Fifo Top Entry: 0x%x\n",
                ((DRF_VAL( _PLWLTL0, _TL_RXCTRL_REG, _RESPDATAMAX, val ) << 2) | 0x3)
               );
        dprintf("RX Request&Response Header Fifo Top Entry: 0x%x\n",
                ((DRF_VAL( _PLWLTL0, _TL_RXCTRL_REG, _HDRMAX, val ) << 2) | 0x3)
               );
        dprintf("RX Buffer Ready: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_RXCTRL_REG, _BUFFERRDY, val )
               ); 
        
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_RXFLOW_REG );
        dprintf("RX Available request header credits: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_RXFLOW_REG, _REQHDRCREDITS, val )
               );
        dprintf("HSHUB Available request data credits: 0x%x\n",
                (DRF_VAL( _PLWLTL0, _TL_RXFLOW_REG, _REQDATACREDITS, val ) << 2)
               );
        dprintf("RX Available response header credits: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_RXFLOW_REG, _RESPHDRCREDITS, val )
               );
        dprintf("HSHUB Available response data credits: 0x%x\n",
                (DRF_VAL( _PLWLTL0, _TL_RXFLOW_REG, _RESPDATACREDITS, val ) << 2)
               );

        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_ERRLOG_REG );
        dprintf("Hardware Recorded Errors: \n|%-25s|%-9s|\n", "Error", "Active");
        dprintf("|%-25s|%-9s|\n", "RX_DL_DATA_PARITY",
                DRF_VAL( _PLWLTL0, _TL_ERRLOG_REG, _RXDLDATAPARITYERR, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "RX_DL_CTRL_PARITY",
                DRF_VAL( _PLWLTL0, _TL_ERRLOG_REG, _RXDLCTRLPARITYERR, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "RX_PROTCOL",
                DRF_VAL( _PLWLTL0, _TL_ERRLOG_REG, _RXPROTOCOLERR, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "RX_OVERFLOW",
                DRF_VAL( _PLWLTL0, _TL_ERRLOG_REG, _RXOVERFLOWERR, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "RX_RAM_DATA_PARITY",
                DRF_VAL( _PLWLTL0, _TL_ERRLOG_REG, _RXRAMDATAPARITYERR, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "RX_RAM_HDR_PARITY",
                DRF_VAL( _PLWLTL0, _TL_ERRLOG_REG, _RXRAMHDRPARITYERR, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "RX_RESP",
                DRF_VAL( _PLWLTL0, _TL_ERRLOG_REG, _RXRESPERR, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "RX_POISON",
                DRF_VAL( _PLWLTL0, _TL_ERRLOG_REG, _RXPOISONERR, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "TX_RAM_DATA_PARITY",
                DRF_VAL( _PLWLTL0, _TL_ERRLOG_REG, _TXRAMDATAPARITYERR, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "TX_RAM_HDR_PARITY",
                DRF_VAL( _PLWLTL0, _TL_ERRLOG_REG, _TXRAMHDRPARITYERR, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "DL_FLOW_PARITY",
                DRF_VAL( _PLWLTL0, _TL_ERRLOG_REG, _DLFLOWPARITYERR, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "DL_HDR_PARITY",
                DRF_VAL( _PLWLTL0, _TL_ERRLOG_REG, _DLHDRPARITYERR, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "TX_CREDIT",
                DRF_VAL( _PLWLTL0, _TL_ERRLOG_REG, _TXCREDITERR, val) ? "Active" : "Inactive"
               );

        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_ERRCTRL_REG );
        dprintf("Fatal Error Controls: \n|%-30s|%-10s|\n", "Error", "Fatal?");
        dprintf("|%-30s|%-10s|\n", "RX_DL_DATA_PARITY",
                DRF_VAL( _PLWLTL0, _TL_ERRCTRL_REG, _RXDLDATAPARITYFATAL, val) ? "Fatal" : "Nonfatal"
               );
        dprintf("|%-30s|%-10s|\n", "RX_DL_CTRL_PARITY",
                DRF_VAL( _PLWLTL0, _TL_ERRCTRL_REG, _RXDLCTRLPARITYFATAL, val) ? "Fatal" : "Nonfatal"
               );
        dprintf("|%-30s|%-10s|\n", "RX_PROTOCOL",
                DRF_VAL( _PLWLTL0, _TL_ERRCTRL_REG, _RXPROTOCOLFATAL, val) ? "Fatal" : "Nonfatal"
               );
        dprintf("|%-30s|%-10s|\n", "RX_OVERFLOW",
                DRF_VAL( _PLWLTL0, _TL_ERRCTRL_REG, _RXOVERFLOWFATAL, val) ? "Fatal" : "Nonfatal"
               );
        dprintf("|%-30s|%-10s|\n", "RX_RAM_DATA_PARITY",
                DRF_VAL( _PLWLTL0, _TL_ERRCTRL_REG, _RXRAMDATAPARITYFATAL, val) ? "Fatal" : "Nonfatal"
               );
        dprintf("|%-30s|%-10s|\n", "RX_RESP",
                DRF_VAL( _PLWLTL0, _TL_ERRCTRL_REG, _RXRESPFATAL, val) ? "Fatal" : "Nonfatal"
               );
        dprintf("|%-30s|%-10s|\n", "RX_POISON",
                DRF_VAL( _PLWLTL0, _TL_ERRCTRL_REG, _RXPOISONFATAL, val) ? "Fatal" : "Nonfatal"
               );
        dprintf("|%-30s|%-10s|\n", "TX_RAM_DATA_PARITY",
                DRF_VAL( _PLWLTL0, _TL_ERRCTRL_REG, _RXRAMDATAPARITYFATAL, val) ? "Fatal" : "Nonfatal"
               );
        dprintf("|%-30s|%-10s|\n", "TX_RAM_HDR_PARITY",
                DRF_VAL( _PLWLTL0, _TL_ERRCTRL_REG, _RXRAMHDRPARITYFATAL, val) ? "Fatal" : "Nonfatal"
               );
        dprintf("|%-30s|%-10s|\n", "DL_FLOW_PARITY",
                DRF_VAL( _PLWLTL0, _TL_ERRCTRL_REG, _DLFLOWPARITYFATAL, val) ? "Fatal" : "Nonfatal"
               );
        dprintf("|%-30s|%-10s|\n", "DL_DATA_PARITY_POIS",
                DRF_VAL( _PLWLTL0, _TL_ERRCTRL_REG, _DLDATAPARITYPOIS, val) ? "Fatal" : "Nonfatal"
               );
        dprintf("|%-30s|%-10s|\n", "DL_HDR_PARITY",
                DRF_VAL( _PLWLTL0, _TL_ERRCTRL_REG, _DLHDRPARITYFATAL, val) ? "Fatal" : "Nonfatal"
               );
        dprintf("|%-30s|%-10s|\n", "TX_CREDIT",
                DRF_VAL( _PLWLTL0, _TL_ERRCTRL_REG, _TXCREDITFATAL, val) ? "Fatal" : "Nonfatal"
               );


        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_RXPROTERR_REG );
        dprintf("RX Protocol Recorded Errors: \n|%-30s|%-9s|\n", "Error", "Active");
        dprintf("|%-25s|%-9s|\n", "Spare",
                DRF_VAL( _PLWLTL0, _TL_RXPROTERR_REG, _SPARE, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "Reserved",
                DRF_VAL( _PLWLTL0, _TL_RXPROTERR_REG, _RSVD, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "AE Flit Detected",
                DRF_VAL( _PLWLTL0, _TL_RXPROTERR_REG, _AE, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "BE flit detected",
                DRF_VAL( _PLWLTL0, _TL_RXPROTERR_REG, _BE, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "Illegal address alignment",
                DRF_VAL( _PLWLTL0, _TL_RXPROTERR_REG, _ALIGN, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "Missing RD or LO",
                DRF_VAL( _PLWLTL0, _TL_RXPROTERR_REG, _RDLO, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "Inconsistent DL Packet Len",
                DRF_VAL( _PLWLTL0, _TL_RXPROTERR_REG, _PKTLEN, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "Spare",
                DRF_VAL( _PLWLTL0, _TL_RXPROTERR_REG, _SPARE1, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "RX Request Header Overflow",
                DRF_VAL( _PLWLTL0, _TL_RXPROTERR_REG, _REQ_HDR_OVF, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "RX Request Data Overflow",
                DRF_VAL( _PLWLTL0, _TL_RXPROTERR_REG, _REQ_DATA_OVF, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "RX Response Header Overflow",
                DRF_VAL( _PLWLTL0, _TL_RXPROTERR_REG, _RESP_HDR_OVF, val) ? "Active" : "Inactive"
               );
        dprintf("|%-25s|%-9s|\n", "RX Response Data Overflow",
                DRF_VAL( _PLWLTL0, _TL_RXPROTERR_REG, _RESP_DATA_OVF, val) ? "Active" : "Inactive"
               );


        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXCREDITCNT0_REG );
        dprintf("Request Header Credits Available: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXCREDITCNT0_REG, _REQHDR, val)
               );
        dprintf("Request Data Credits Available: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXCREDITCNT0_REG, _REQDATA, val)
               );
        dprintf("DL Overflow: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXCREDITCNT0_REG, _DLOVF, val)
               );
        dprintf("Request Header Overflow: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXCREDITCNT0_REG, _REQHDROVF, val)
               );
        dprintf("Request Data Overflow: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXCREDITCNT0_REG, _REQDATAOVF, val)
               );

        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXCREDITCNT1_REG );
        dprintf("Response Header Credits Available: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXCREDITCNT1_REG, _RESPHDR, val)
               );
        dprintf("Response Data Credits Available: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXCREDITCNT1_REG, _RESPDATA, val)
               );
        dprintf("Response Header Overflow: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXCREDITCNT1_REG, _RESPHDROVF, val)
               );
        dprintf("Response Data Overflow: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXCREDITCNT1_REG, _RESPDATAOVF, val)
               );

        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXCREDITLIMIT0_REG );
        dprintf("Request Header Credit Limit: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXCREDITLIMIT0_REG, _REQHDR, val)
               );
        dprintf("Request Data Credit Limit: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXCREDITLIMIT0_REG, _REQDATA, val)
               );

        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXCREDITLIMIT1_REG );
        dprintf("Response Header Credit Limit: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXCREDITLIMIT1_REG, _RESPHDR, val)
               );
        dprintf("Response Data Credit Limit: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXCREDITLIMIT1_REG, _RESPDATA, val)
               );

        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXCREDITLIMIT2_REG );
        dprintf("Replay Credit Limit: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXCREDITLIMIT2_REG, _REPLAY, val)
               );

        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXDBG0_REG );
        dprintf("Next Sched St Ptr: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXDBG0_REG, _SCHEDSTPTR, val)
               );
        dprintf("DL Credit Counter Val: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXDBG0_REG, _DLCREDIT, val)
               );
        dprintf("Request Control Fifo Has Entry? 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXDBG0_REG, _REQCTRLVLD, val)
               );
        dprintf("Request Control Fifo Head Ptr: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXDBG0_REG, _REQCTRLHEAD, val)
               );
        dprintf("TL TX Idle? 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXDBG0_REG, _TXIDLE, val)
               );

        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXDBG1_REG );
        dprintf("Tx Scheduler State: \n|%-6s|%-7s|\n", "Entry", "State");
        dprintf("|%-6s|%-7x|\n", "0",  DRF_VAL( _PLWLTL0, _TL_TXDBG1_REG, _SCHEDST0, val));
        dprintf("|%-6s|%-7x|\n", "1",  DRF_VAL( _PLWLTL0, _TL_TXDBG1_REG, _SCHEDST1, val));
        dprintf("|%-6s|%-7x|\n", "2",  DRF_VAL( _PLWLTL0, _TL_TXDBG1_REG, _SCHEDST2, val));
        dprintf("|%-6s|%-7x|\n", "3",  DRF_VAL( _PLWLTL0, _TL_TXDBG1_REG, _SCHEDST3, val));
        dprintf("|%-6s|%-7x|\n", "4",  DRF_VAL( _PLWLTL0, _TL_TXDBG1_REG, _SCHEDST4, val));
        dprintf("|%-6s|%-7x|\n", "5",  DRF_VAL( _PLWLTL0, _TL_TXDBG1_REG, _SCHEDST5, val));
        dprintf("|%-6s|%-7x|\n", "6",  DRF_VAL( _PLWLTL0, _TL_TXDBG1_REG, _SCHEDST6, val));
        dprintf("|%-6s|%-7x|\n", "7",  DRF_VAL( _PLWLTL0, _TL_TXDBG1_REG, _SCHEDST7, val));
        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXDBG2_REG );
        dprintf("|%-6s|%-7x|\n", "8",  DRF_VAL( _PLWLTL0, _TL_TXDBG2_REG, _SCHEDST8, val));
        dprintf("|%-6s|%-7x|\n", "9",  DRF_VAL( _PLWLTL0, _TL_TXDBG2_REG, _SCHEDST9, val));
        dprintf("|%-6s|%-7x|\n", "10",  DRF_VAL( _PLWLTL0, _TL_TXDBG2_REG, _SCHEDST10, val));
        dprintf("|%-6s|%-7x|\n", "11",  DRF_VAL( _PLWLTL0, _TL_TXDBG2_REG, _SCHEDST11, val));
        dprintf("|%-6s|%-7x|\n", "12",  DRF_VAL( _PLWLTL0, _TL_TXDBG2_REG, _SCHEDST12, val));
        dprintf("|%-6s|%-7x|\n", "13",  DRF_VAL( _PLWLTL0, _TL_TXDBG2_REG, _SCHEDST13, val));
        dprintf("|%-6s|%-7x|\n", "14",  DRF_VAL( _PLWLTL0, _TL_TXDBG2_REG, _SCHEDST14, val));
        dprintf("|%-6s|%-7x|\n", "15",  DRF_VAL( _PLWLTL0, _TL_TXDBG2_REG, _SCHEDST15, val));

        val = GPU_REG_RD32( pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TXDBG3_REG );
        dprintf("Next Sched State: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXDBG3_REG, _SCHEDSTNXT, val)
               );
        dprintf("Response Control Head Entry: 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXDBG3_REG, _RESPCTRLHEAD, val)
               );
        dprintf("Response Control Fifo Valid? 0x%x\n",
                DRF_VAL( _PLWLTL0, _TL_TXDBG3_REG, _RESPCTRLVLD, val)
               );
    }

    dprintf("===============\n");
    dprintf("PTRIM Registers\n");
    dprintf("===============\n");

    // Print clock calibration status
    val = GPU_REG_RD32( LW_PTRIM_SYS_LWLINK_UPHY_OUTPUT );
    dprintf("PLL Calibration State:\n|%-16s|%-9s|\n", "Name", "Asserted");
    dprintf("|%-16s|%-9x|\n", "FLAG_CAL_VALID0",
            DRF_VAL( _PTRIM, _SYS_LWLINK_UPHY_OUTPUT, _FLAG_CAL_VALID0, val)
           );
    dprintf("|%-16s|%-9x|\n", "FLAG_CAL_VALID1",
            DRF_VAL( _PTRIM, _SYS_LWLINK_UPHY_OUTPUT, _FLAG_CAL_VALID1, val)
           );
    dprintf("|%-16s|%-9x|\n", "FLAG_CAL_VALID2",
            DRF_VAL( _PTRIM, _SYS_LWLINK_UPHY_OUTPUT, _FLAG_CAL_VALID2, val)
           );
    dprintf("|%-16s|%-9x|\n", "FLAG_CAL_VALID3",
            DRF_VAL( _PTRIM, _SYS_LWLINK_UPHY_OUTPUT, _FLAG_CAL_VALID3, val)
           );
    dprintf("|%-16s|%-9x|\n", "PLL_OFF0",
            DRF_VAL( _PTRIM, _SYS_LWLINK_UPHY_OUTPUT, _PLL_OFF0, val)
           );
    dprintf("|%-16s|%-9x|\n", "PLL_OFF1",
            DRF_VAL( _PTRIM, _SYS_LWLINK_UPHY_OUTPUT, _PLL_OFF1, val)
           );
    dprintf("|%-16s|%-9x|\n", "PLL_OFF2",
            DRF_VAL( _PTRIM, _SYS_LWLINK_UPHY_OUTPUT, _PLL_OFF2, val)
           );
    dprintf("|%-16s|%-9x|\n", "PLL_OFF3",
            DRF_VAL( _PTRIM, _SYS_LWLINK_UPHY_OUTPUT, _PLL_OFF3, val)
           );

lwlinkPrintVerboseCleanup:
    freeLwlDiscoveryList(pLwlDiscoverList);
    freeLwlStatusArray();
}

//
// lwlinkReadUPhyPLLCfg_GP100
//
LwU16 lwlinkReadUPhyPLLCfg_GP100(LwU32 link, LwU32 addr) 
{
    LwU32 val;
    LwS32 timeoutUs = LWLINK_UPHY_REG_POLL_TIMEOUT;

    val = GPU_REG_RD32(pLwlStatusArray[link].dlPriBase + LW_PLWL_BR0_PLL_CTL_4); 
    val = FLD_SET_DRF_NUM(_PLWL, _BR0_PLL_CTL_4, _CFG_ADDR, addr, val);
    val = FLD_SET_DRF_NUM(_PLWL, _BR0_PLL_CTL_4, _CFG_RDS, 0x1, val);
    GPU_REG_WR32(pLwlStatusArray[link].dlPriBase + LW_PLWL_BR0_PLL_CTL_4, val);

    do
    {
        val = GPU_REG_RD32(pLwlStatusArray[link].dlPriBase 
                            + LW_PLWL_BR0_CFG_STATUS_1);
        
        if (FLD_TEST_DRF_NUM(_PLWL, _BR0_CFG_STATUS_1, _CFG_PLL_BUSY, 0x0, val))
        {
            break;
        }

        osPerfDelay(20);
        timeoutUs -= 20;
    } while (timeoutUs > 0);
    
    val = GPU_REG_RD32(pLwlStatusArray[link].dlPriBase + LW_PLWL_BR0_PLL_CTL_5);
    return (LwU16) (DRF_NUM(_PLWL, _BR0_PLL_CTL_5, _CFG_RDATA, val));
}

//
// lwlinkReadUPhyLaneCfg_GP100
//
LwU16 lwlinkReadUPhyLaneCfg_GP100(LwU32 link, LwU32 lane, LwU32 addr)
{
    LwU32 val;
    LwS32 timeoutUs = LWLINK_UPHY_REG_POLL_TIMEOUT;

    val = GPU_REG_RD32(pLwlStatusArray[link].dlPriBase + LW_PLWL_BR0_PAD_CTL_6(lane)); 
    val = FLD_SET_DRF_NUM(_PLWL, _BR0_PAD_CTL_6, _CFG_ADDR, addr, val);
    val = FLD_SET_DRF_NUM(_PLWL, _BR0_PAD_CTL_6, _CFG_RDS, 0x1, val);
    GPU_REG_WR32(pLwlStatusArray[link].dlPriBase + LW_PLWL_BR0_PAD_CTL_6(lane), val);


    do
    {
        val = GPU_REG_RD32(pLwlStatusArray[link].dlPriBase 
                            + LW_PLWL_BR0_CFG_STATUS_1);
        
        if (!(BIT(lane) & val))
        {
            break;
        }

        osPerfDelay(20);
        timeoutUs -= 20;
    } while (timeoutUs > 0);

    val = GPU_REG_RD32(pLwlStatusArray[link].dlPriBase + LW_PLWL_BR0_PAD_CTL_7(lane));
    return (LwU16) (DRF_NUM(_PLWL, _BR0_PAD_CTL_7, _CFG_RDATA, val));
}

//
// lwlinkWriteUPhyPLLCfg_GP100
//
void lwlinkWriteUPhyPLLCfg_GP100(LwU32 link, LwU32 addr, LwU16 data)
{
    LwU32 val;
    LwS32 timeoutUs = LWLINK_UPHY_REG_POLL_TIMEOUT;

    val = GPU_REG_RD32(pLwlStatusArray[link].dlPriBase + LW_PLWL_BR0_PLL_CTL_4); 
    val = FLD_SET_DRF_NUM(_PLWL, _BR0_PLL_CTL_4, _CFG_WDATA, data, val);
    val = FLD_SET_DRF_NUM(_PLWL, _BR0_PLL_CTL_4, _CFG_ADDR, addr, val);
    val = FLD_SET_DRF_NUM(_PLWL, _BR0_PLL_CTL_4, _CFG_WDS, 0x1, val);
    GPU_REG_WR32(pLwlStatusArray[link].dlPriBase + LW_PLWL_BR0_PLL_CTL_4, val);

    do
    {
        val = GPU_REG_RD32(pLwlStatusArray[link].dlPriBase 
                            + LW_PLWL_BR0_CFG_STATUS_1);
        
        if (FLD_TEST_DRF_NUM(_PLWL, _BR0_CFG_STATUS_1, _CFG_PLL_BUSY, 0x0, val))
        {
            break;
        }

        osPerfDelay(20);
        timeoutUs -= 20;
    } while (timeoutUs > 0);
}

//
// lwlinkWriteUPhyLaneCfg_GP100
//
void lwlinkWriteUPhyLaneCfg_GP100(LwU32 link, LwU32 lane, LwU32 addr, LwU16 data)
{
    LwU32 val;
    LwS32 timeoutUs = LWLINK_UPHY_REG_POLL_TIMEOUT;

    val = GPU_REG_RD32(pLwlStatusArray[link].dlPriBase + LW_PLWL_BR0_PAD_CTL_6(lane)); 
    val = FLD_SET_DRF_NUM(_PLWL, _BR0_PAD_CTL_6, _CFG_WDATA, data, val);
    val = FLD_SET_DRF_NUM(_PLWL, _BR0_PAD_CTL_6, _CFG_ADDR, addr, val);
    val = FLD_SET_DRF_NUM(_PLWL, _BR0_PAD_CTL_6, _CFG_WDS, 0x1, val);
    GPU_REG_WR32(pLwlStatusArray[link].dlPriBase + LW_PLWL_BR0_PAD_CTL_6(lane), val);

    do
    {
        val = GPU_REG_RD32(pLwlStatusArray[link].dlPriBase 
                            + LW_PLWL_BR0_CFG_STATUS_1);
        
        if (!(BIT(lane) & val))
        {
            break;
        }

        osPerfDelay(20);
        timeoutUs -= 20;
    } while (timeoutUs > 0);
}

//
// lwlinkDumpUPhy_GP100
//
void lwlinkDumpUPhy_GP100(void)
{
    LwU32 val;
    LwU32 linkId, laneId;
    LwU32 i, j;
    LwU32 linksDiscovered = 0;
    LwU32 statusArraySize = 0;
    lwlDiscover *pLwlDiscoverList = NULL;

    linksDiscovered = pLwlink[indexGpu].lwlinkLinkDiscovery((void *)&pLwlDiscoverList, 0);
    if (linksDiscovered == 0)
    {
        dprintf("No links were detected during the discovery process in ioctrl\n");
        return;
    }

    if (initLwlStatusArray(&statusArraySize, linksDiscovered, pLwlDiscoverList) != LW_OK)
    {
        dprintf("Error encountered while initializing the status array\n");
        return;
    }

    for (linkId = 0; linkId < statusArraySize; linkId++)
    {
        val = GPU_REG_RD32( LW_IOCTRL_RESET );
        if (!(DRF_VAL(_IOCTRL, _RESET, _LINKRESET, val) & BIT(linkId)))
        {
            dprintf("Skipping link %d\n", linkId);
            continue;
        }

        dprintf("Link %d CLM Regs\n", linkId);

        for (i = 0x00; i < 0x100; i += 0x10)
        {
            dprintf("%2x: ", i);
            for (j = i; j < i + 0x10; ++j)
            {
                dprintf("%4x ", lwlinkReadUPhyPLLCfg_GP100(linkId, j));
            }
            dprintf("\n");
        }

        dprintf("Link %d DLM Regs\n", linkId);
        for (laneId = 0; laneId < 8; laneId++)
        {
            dprintf("Lane %d\n", laneId);
            for (i = 0x00; i < 0x100; i += 0x10)
            {
                dprintf("%2x: ", i);
                for (j = i; j < i + 0x10; ++j)
                {
                    dprintf("%4x ", lwlinkReadUPhyLaneCfg_GP100(linkId, laneId, j));    
                }
                dprintf("\n");
            }
        }
    }

    freeLwlDiscoveryList(pLwlDiscoverList);
    freeLwlStatusArray();
}

//
// Program the lwlink TL counters
//
void lwlinkProgramTlCounters_GP100(LwS32 linkId)
{
    LwU32        linksDiscovered = 0;
    LwU32        statusArraySize = 0;
    LwU32        ioctrlPriBase   = 0;
    LwU32        i, j;
    lwlDiscover *pLwlDiscoverList = NULL;

    // Discover IOCTRL in device info
    ioctrlPriBase = pLwlink[indexGpu].lwlinkIoctrlDiscovery(NULL);

    if (ioctrlPriBase == 0)
    {
        dprintf("No ioctrls were discovered in PTOP_DEVICE_INFO!\n");
        return;
    }

    // Discover lwlink TL and DL/PL in IOCTRL
    linksDiscovered = pLwlink[indexGpu].lwlinkLinkDiscovery((void *)&pLwlDiscoverList, 0);

    if (linksDiscovered == 0)
    {
        dprintf("No links were detected during the discovery process in ioctrl\n");
        return;
    }

    // This creates the statusarray used to store information on the links
    if (initLwlStatusArray(&statusArraySize, linksDiscovered, pLwlDiscoverList) != LW_OK)
    {
        dprintf("Error encountered while initializing the status array\n");
        return;
    }

    // Program the lwlink TL counters
    if (linkId < 0)
    {
        dprintf("Programming all lwlink TL counters\n");
        for (i = 0; i < statusArraySize; i++)
        {
            linkId = pLwlStatusArray[i].linkNum;
            dprintf("Programming lwlink TL counters for link %d\n", linkId);
            for (j = 0; j < LW_LWLTL_TL_TPCNTCTL_REG__SIZE_1; j++)
            {
                GPU_REG_WR32(pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TPCNTCTL_REG(j), 0x1FFFE);
            }
        }
    }
    else
    {
        for (i = 0; i < statusArraySize; i++)
        {
            if (pLwlStatusArray[i].linkNum == linkId)
            {
                dprintf("Programming lwlink TL counters for link %d\n", linkId);
                for (j = 0; j < LW_LWLTL_TL_TPCNTCTL_REG__SIZE_1; j++)
                {
                    GPU_REG_WR32(pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TPCNTCTL_REG(j), 0x1FFFE);
                }
                break;
            }
        }
        if (i == statusArraySize)
        {
            dprintf("Link %d not found in IOCTRL\n", linkId);
        }
    }

    freeLwlDiscoveryList(pLwlDiscoverList);
    freeLwlStatusArray();
}

//
// Reset the lwlink TL counters
//
void lwlinkResetTlCounters_GP100(LwS32 linkId)
{
    LwU32        linksDiscovered = 0;
    LwU32        statusArraySize = 0;
    LwU32        ioctrlPriBase   = 0;
    LwU32        i;
    lwlDiscover *pLwlDiscoverList = NULL;

    // Discover IOCTRL in device info
    ioctrlPriBase = pLwlink[indexGpu].lwlinkIoctrlDiscovery(NULL);

    if (ioctrlPriBase == 0)
    {
        dprintf("No ioctrls were discovered in PTOP_DEVICE_INFO!\n");
        return;
    }

    // Discover lwlink TL and DL/PL in IOCTRL
    linksDiscovered = pLwlink[indexGpu].lwlinkLinkDiscovery((void *)&pLwlDiscoverList, 0);

    if (linksDiscovered == 0)
    {
        dprintf("No links were detected during the discovery process in ioctrl\n");
        return;
    }

    // This creates the statusarray used to store information on the links
    if (initLwlStatusArray(&statusArraySize, linksDiscovered, pLwlDiscoverList) != LW_OK)
    {
        dprintf("Error encountered while initializing the status array\n");
        return;
    }

    // Reset the lwlink TL counters
    if (linkId < 0)
    {
        dprintf("Resetting all lwlink TL counters\n");
        for (i = 0; i < statusArraySize; i++)
        {
            linkId = pLwlStatusArray[i].linkNum;
            dprintf("Resetting lwlink TL counters for link %d\n", linkId);
            GPU_REG_WR32(pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_COUNT_START, 0xF0000);
            GPU_REG_WR32(pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_COUNT_START, 0xF);
        }
    }
    else
    {
        for (i = 0; i < statusArraySize; i++)
        {
            if (pLwlStatusArray[i].linkNum == linkId)
            {
                dprintf("Resetting lwlink TL counters for link %d\n", linkId);
                GPU_REG_WR32(pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_COUNT_START, 0xF0000);
                GPU_REG_WR32(pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_COUNT_START, 0xF);
                break;
            }
        }
        if (i == statusArraySize)
        {
            dprintf("Link %d not found in IOCTRL\n", linkId);
        }
    }

    freeLwlDiscoveryList(pLwlDiscoverList);
    freeLwlStatusArray();
}

//
// Read the lwlink TL counters
//
void lwlinkReadTlCounters_GP100(LwS32 linkId)
{
    LwU32        linksDiscovered = 0;
    LwU32        statusArraySize = 0;
    LwU32        ioctrlPriBase   = 0;
    LwU32        countUnit;
    LwU32        i, j;
    lwlDiscover *pLwlDiscoverList = NULL;

    // Discover IOCTRL in device info
    ioctrlPriBase = pLwlink[indexGpu].lwlinkIoctrlDiscovery(NULL);

    if (ioctrlPriBase == 0)
    {
        dprintf("No ioctrls were discovered in PTOP_DEVICE_INFO!\n");
        return;
    }

    // Discover lwlink TL and DL/PL in IOCTRL
    linksDiscovered = pLwlink[indexGpu].lwlinkLinkDiscovery((void *)&pLwlDiscoverList, 0);

    if (linksDiscovered == 0)
    {
        dprintf("No links were detected during the discovery process in ioctrl\n");
        return;
    }

    // This creates the statusarray used to store information on the links
    if (initLwlStatusArray(&statusArraySize, linksDiscovered, pLwlDiscoverList) != LW_OK)
    {
        dprintf("Error encountered while initializing the status array\n");
        return;
    }

    // Read the lwlink TL counters
    if (linkId < 0)
    {
        dprintf("Reading ALL lwlink TL counters\n");
        for (i = 0; i < statusArraySize; i++)
        {
            linkId = pLwlStatusArray[i].linkNum;
            dprintf("\nCounter values for LINK %d since the last reset\n\n", linkId);

            for (j = 0; j < LW_LWLTL_TL_TPCNTLO_REG__SIZE_1; j++)
            {
                countUnit = DRF_VAL(_LWLTL_TL, _TPCNTCTL_REG, _UNIT,
                            GPU_REG_RD32(pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TPCNTCTL_REG(j)));
                switch (countUnit)
                {
                    case LW_LWLTL_TL_TPCNTCTL_REG_UNIT_CYCLES:
                        dprintf("Count Unit: CYCLES\t");
                        break;
                    case LW_LWLTL_TL_TPCNTCTL_REG_UNIT_PACKETS:
                        dprintf("Count Unit: PACKETS\t");
                        break;
                    case LW_LWLTL_TL_TPCNTCTL_REG_UNIT_FLITS:
                        dprintf("Count Unit: FLITS\t");
                        break;
                    case LW_LWLTL_TL_TPCNTCTL_REG_UNIT_BYTES:
                        dprintf("Count Unit: BYTES\t");
                        break;
                    default:
                        dprintf("Count Unit: UNKNOWN\t");
                }
                dprintf("LW_PLWLTL%d_TL_TPCNTLO_REG(%d) = 0x%x\t",
                        linkId, j,
                        GPU_REG_RD32(pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TPCNTLO_REG(j)));
                dprintf("LW_PLWLTL%d_TL_TPCNTHI_REG(%d) = 0x%x\n",
                        linkId, j,
                        GPU_REG_RD32(pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TPCNTHI_REG(j)));
            }
        }
    }
    else
    {
        for (i = 0; i < statusArraySize; i++)
        {
            if (pLwlStatusArray[i].linkNum == linkId)
            {
                dprintf("\nCounter values for LINK %d since the last reset\n\n", linkId);
                for (j = 0; j < LW_LWLTL_TL_TPCNTLO_REG__SIZE_1; j++)
                {
                    countUnit = DRF_VAL(_LWLTL_TL, _TPCNTCTL_REG, _UNIT,
                                GPU_REG_RD32(pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TPCNTCTL_REG(j)));
                    switch (countUnit)
                    {
                        case LW_LWLTL_TL_TPCNTCTL_REG_UNIT_CYCLES:
                            dprintf("Count Unit: CYCLES\t");
                            break;
                        case LW_LWLTL_TL_TPCNTCTL_REG_UNIT_PACKETS:
                            dprintf("Count Unit: PACKETS\t");
                            break;
                        case LW_LWLTL_TL_TPCNTCTL_REG_UNIT_FLITS:
                            dprintf("Count Unit: FLITS\t");
                            break;
                        case LW_LWLTL_TL_TPCNTCTL_REG_UNIT_BYTES:
                            dprintf("Count Unit: BYTES\t");
                            break;
                        default:
                            dprintf("Count Unit: UNKNOWN\t");
                    }
                    dprintf("LW_PLWLTL%d_TL_TPCNTLO_REG(%d) = 0x%x\t",
                            linkId, j,
                            GPU_REG_RD32(pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TPCNTLO_REG(j)));
                    dprintf("LW_PLWLTL%d_TL_TPCNTHI_REG(%d) = 0x%x\n",
                            linkId, j,
                            GPU_REG_RD32(pLwlStatusArray[i].tlPriBase + LW_LWLTL_TL_TPCNTHI_REG(j)));
                }
            }
        }
        if (i == statusArraySize)
        {
            dprintf("Link %d not found in IOCTRL\n", linkId);
        }
    }

    freeLwlDiscoveryList(pLwlDiscoverList);
    freeLwlStatusArray();
}

// The main print function for GP100 support
void lwlinkPrintStatus_GP100(LwBool bComplete)
{
    LwU32 linksDiscovered = 0;
    LwU32 statusArraySize = 0;
    LwU32 ioctrlPriBase = 0;
    lwlDiscover *pLwlDiscoverList = NULL;

    // Performs a linkdiscovery process to determine number of active links
    // PRI_BASE for ioctrl is lwrrently unused as there is no ioctrl_ip manual
    ioctrlPriBase = pLwlink[indexGpu].lwlinkIoctrlDiscovery(NULL);

    if (ioctrlPriBase == 0)
    {
        dprintf("No ioctrls were discovered in PTOP_DEVICE_INFO!\n");
        return;
    }

    linksDiscovered = pLwlink[indexGpu].lwlinkLinkDiscovery((void *)&pLwlDiscoverList, 0);

    if (linksDiscovered == 0)
    {
        dprintf("No links were detected during the discovery process in ioctrl\n");
        return;
    }

    // This creates the statusarray used to store information on the links
    if (initLwlStatusArray(&statusArraySize, linksDiscovered, pLwlDiscoverList) != LW_OK)
    {
        dprintf("Error encountered while initializing the status array\n");
        return;
    }

    // This reads and fills out the power column for the links
    if (lwlinkGetPowerStatus(statusArraySize, (void *)pLwlStatusArray) != LW_OK)
        goto cleanup;

    // This reads and fills out the rx and tx mode columns for the links
    if (lwlinkGetRxTxMode(statusArraySize, (void *)pLwlStatusArray) != LW_OK)
        goto cleanup;

    // Retrieves hsHubstatus to fill out the type column
    if (lwlinkGetHshubStatus(statusArraySize, (void *)pLwlStatusArray) != LW_OK)
        goto cleanup;

    // Reads the TL interrupt registers as well as other previously fill columns
    // to determine link status
    if (lwlinkGetStatus(statusArraySize, (void *)pLwlStatusArray) != LW_OK)
        goto cleanup;

    // Prints information on all the links
    printLwlStatusArray(statusArraySize);

cleanup:
    // Deletes list and frees memory
    freeLwlDiscoveryList(pLwlDiscoverList);
    freeLwlStatusArray();
}

void lwlinkPrintHelp_GP100(void)
{
    dprintf("usage: lwv lwlink  <# of args> [-v] [-vv] [-dumpuphys]\n");
    dprintf("                       '-v'  : verbose mode\n");
    dprintf("                       '-p' : verbose + pretty printing mode\n");
    dprintf("                       '-dumpuphys' : Dumps the UPHYS registers\n");
    dprintf("Description of the Columns:");
    dprintf("STATUS:        displays the operational status of the link. The default/uninitialized value is OFF.\n");
    dprintf("TYPE:          displays if the link is operating in SYSMEM or P2P mode. The default/uninitialized value is N/A.\n");
    dprintf("POWER:         displays the power status of the link. The default/uninitialized value is OFF.\n");
    dprintf("RX/TX MODE:    displays the RX/TX mode of the link. The default/uninitialized value is N/A.\n");
}

const char *lwlinkStatusText(LWL_STATUS status)
{
    assert(status < sizeof(pStatusText));
    return pStatusText[status];
}

const char *lwlinkTypeText(LWL_TYPE type)
{
    assert(type < sizeof(pTypeText));
    return pTypeText[type];
}

const char *lwlinkPowerText(LWL_POWER power)
{
    assert(power < sizeof(pPowerText));
    return pPowerText[power];
}

const char *lwlinkModeText(LWL_MODE mode)
{
    assert(mode < sizeof(pModeText));
    return pModeText[mode];
}

//
// Prints information stored in lwlstatuslist. There are two print methods
// depending on if the link is P2P or otherwise.
//
void printLwlStatusArray(LwU32 statusArraySize)
{
    LwU32 i;
    dprintf("\n");
    dprintf("LINK   STATUS  TYPE          POWER   RX_MODE TX_MODE\n");
    dprintf("----------------------------------------------------\n");

    for (i = 0; i < statusArraySize; i++) 
    {
        if (pLwlStatusArray[i].type != LWL_P2P)
        {
            dprintf("%-7d%-8s%-14s%-8s%-8s%-7s\n", 
                pLwlStatusArray[i].linkNum,
                pStatusText[pLwlStatusArray[i].status], 
                pTypeText[pLwlStatusArray[i].type], 
                pPowerText[pLwlStatusArray[i].power], 
                pModeText[pLwlStatusArray[i].rxMode],
                pModeText[pLwlStatusArray[i].txMode]);
        }
        else
        {
            int extraSpacing = (pLwlStatusArray[i].peer < 10)? 2:1;
            dprintf("%-7d%-8s%-4s(PEER %d)%*s%-8s%-8s%-7s\n", 
                pLwlStatusArray[i].linkNum, 
                pStatusText[pLwlStatusArray[i].status],
                pTypeText[pLwlStatusArray[i].type], 
                pLwlStatusArray[i].peer, 
                extraSpacing, "", 
                pPowerText[pLwlStatusArray[i].power],
                pModeText[pLwlStatusArray[i].rxMode], 
                pModeText[pLwlStatusArray[i].txMode]);
        }
    }
    dprintf("\n");
}

// Get the TL and DL base addresses for the link
void tldlPriBaseLookUp
(
    lwlDiscover *pLwlDiscoverList,
    LwU32        linkID,
    LwU32       *dlAddr,
    LwU32       *tlAddr
)
{
    lwlDiscover *pLwrr = pLwlDiscoverList;
    while (pLwrr != NULL)
    {
        if (pLwrr->ID == linkID)
        {
            if (pLwrr->deviceType == LWL_LWLINK)
            {
                *dlAddr = pLwrr->addr;
            }
            else if (pLwrr->deviceType == LWL_LWLTL)
            {
                *tlAddr = pLwrr->addr;
            }
        }
        pLwrr = pLwrr->next;
    }
}

// Initializes list of lwlink status nodes based on statusArraySize
LW_STATUS initLwlStatusArray
(
    LwU32       *statusArraySize,
    LwU32        linksDiscovered,
    lwlDiscover *pLwlDiscoverList
)
{
    LwU32 i;
    LwU32 maxLinkNum          = sizeof(LwU32) * 8;
    LwU32 statusArrayPos      = 0;
    LwU32 linksDiscoveredCopy = linksDiscovered;
    LwU32 dlAddr              = 0;
    LwU32 tlAddr              = 0;

    //
    // This figures out how many links there are by finding the number of '1's
    // in the linksDiscovered bitstring
    //
    while (linksDiscovered)
    {
        linksDiscovered &= linksDiscovered - 1;
        *statusArraySize = *statusArraySize + 1;
    }

    // Constructs an array based on the number of links discovered
    pLwlStatusArray = malloc(*statusArraySize * sizeof(lwlStatus));

    if (pLwlStatusArray == NULL)
    {
        dprintf("malloc failed to create the status array, exiting\n");
        return LW_ERR_GENERIC;
    }

    // Sets some default values
    for (i = 0; i < maxLinkNum; i++)
    {
        if ((linksDiscoveredCopy >> i) & 1)
        {
            pLwlStatusArray[statusArrayPos].linkNum = i;
            pLwlStatusArray[statusArrayPos].status  = LWL_OFF;
            pLwlStatusArray[statusArrayPos].type    = LWL_T_NA;
            pLwlStatusArray[statusArrayPos].peer    = 0;
            pLwlStatusArray[statusArrayPos].power   = LWL_POFF;
            pLwlStatusArray[statusArrayPos].rxMode  = LWL_NA;
            pLwlStatusArray[statusArrayPos].txMode  = LWL_NA;

            tldlPriBaseLookUp(pLwlDiscoverList, i, &dlAddr, &tlAddr);

            if (dlAddr != 0 && tlAddr != 0)
            {
                pLwlStatusArray[statusArrayPos].dlPriBase = dlAddr;
                pLwlStatusArray[statusArrayPos].tlPriBase = tlAddr;
            }
            else
            {
                dprintf("Did not find a PriBase address for the DL/TL layers for"
                        "link number: %d\n", i);
                free(pLwlStatusArray);
                return LW_ERR_GENERIC;
            }

            dlAddr = 0;
            tlAddr = 0;

            statusArrayPos++;
        }
    }

    return LW_OK;
}

// Deletes lwlink status list to free memory
void freeLwlStatusArray(void)
{
    free(pLwlStatusArray);
}
