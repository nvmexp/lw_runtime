#include "lwwatch.h"
#include "exts.h"
#include "ibmnpu.h"
#include "lwlink.h"

#include "pascal/gp100/dev_lwl_ip.h"

#include "lw_ref.h"

LwU16 BYTESWAP_16(LwU16 val) 
{
    return (val >> 8) | (val << 8);  
}

LwU32 BYTESWAP_32(LwU32 val) 
{
    return    (val >> 24)  
            | (val << 24)  
            | ((val & 0x00FF0000) >> 8)   
            | ((val & 0x0000FF00) << 8);  
}

LwU64 BYTESWAP_64(LwU64 val) 
{
    return    (val >> 56)  
            | (val << 56)  
            | ((val & 0x00FF000000000000ULL) >> 40)  
            | ((val & 0x0000FF0000000000ULL) >> 24)  
            | ((val & 0x000000FF00000000ULL) >>  8)   
            | ((val & 0x00000000FF000000ULL) <<  8)   
            | ((val & 0x0000000000FF0000ULL) << 24)  
            | ((val & 0x000000000000FF00ULL) << 40); 
}


static ibmnpu_device devices[MAX_IBMNPU_DEVICES][MAX_IBMNPU_BRICKS_PER_DEVICE] = {{{{0}}}};

void onNpuPciDevice(LwU16 domain, LwU8 bus, LwU8 device, LwU8 func)
{
    LwU32 i;
    LwU32 command;
    LwU32 interrupt;

    // Addresses/values in PCI extended configuration space
    LwU32 cfgVal;
    LwU32 capPtr;

    // NPU top-level device index
    LwU8 npuIndex;
    // NPU Brick #.
    LwU8 brickNum;

    // Temporary device to hold our data before we know where to put it.
    ibmnpu_device dev;

    for (npuIndex = 0; npuIndex < MAX_IBMNPU_DEVICES; npuIndex++)
    {
        if (!devices[npuIndex][0].initialized)
        {
            break;
        }
        else if (devices[npuIndex][0].pciInfo.domain == domain)
        {
            break;
        }
    }

    if (MAX_IBMNPU_DEVICES == npuIndex)
    {
        dprintf("WARNING: The number of Processors/NPUs discovered (%d) is greater "
                "than expected! Cannot initialize NPU device "
                "(%04d:%02d:%02d.%02d)\n",
                npuIndex,
                domain, bus, device, func);
        return;
    }


    dprintf("Found NPU Device at (%04d:%02d:%02d.%02d)\n",
                domain, bus, device, func);
    
    dev.initialized = 1;
    dev.pciInfo.domain = domain;
    dev.pciInfo.bus = bus;
    dev.pciInfo.device = device;
    dev.pciInfo.func = func;

    // Determine proc ctrl/status registers and brick num
    cfgVal = NPU_CFG_RD32(&dev, LW_CONFIG_PCI_LW_13);
    capPtr = DRF_VAL(_CONFIG, _PCI_LW_13, _CAP_PTR, cfgVal);

    // Set brickNum to an invalid value, just in case reading it fails.
    brickNum = 0xFF;
    while (0 != capPtr)
    {
        cfgVal = NPU_CFG_RD32(&dev, capPtr + LW_CONFIG_PCI_LW_0);
        if (PCI_VENDOR_CAP_IBMNPU == DRF_VAL(_CONFIG, _PCI_LW_24, _CAP_ID, cfgVal))
        {
            // Retrieve the procedure status/control register addresses
            dev.procStatusAddr = capPtr +
                PCI_VENDOR_CAP_OFFSET_PROC_STATUS + LW_CONFIG_PCI_LW_0;
            dev.procControlAddr = capPtr +
                PCI_VENDOR_CAP_OFFSET_PROC_CONTROL + LW_CONFIG_PCI_LW_0;

            // Determine the brick num for this device
            cfgVal = NPU_CFG_RD32(&dev, capPtr + LW_CONFIG_PCI_LW_0
                                    + PCI_VENDOR_CAP_OFFSET_BRICK_NUM);
            brickNum = DRF_VAL(_IBMNPU, _BRICK, _NUM, cfgVal);
            break;
        }
        
        capPtr = DRF_VAL(_CONFIG, _PCI_LW_24, _NEXT_PTR, cfgVal);
    }

    if (brickNum >= MAX_IBMNPU_BRICKS_PER_DEVICE)
    {
        dprintf("Error: device identified itself as Brick #%d, which is outside "
                    "the expected range.\n", brickNum);
        return;
    }
    else if (devices[npuIndex][brickNum].initialized)
    {
        dprintf("WARNING: Device with brick #%d already initialized. There may "
                "be some loss in link identification.\n", brickNum);
    }

    // Retrieve information / memmap PCI base address registers
    for (i = 0; i < MAX_IBMNPU_BARS_PER_BRICK; i++)
    {
        if (LW_OK == osPciGetBarInfo(domain, bus, device, func, (LwU8)i,
            &(dev.bars[i].bar), &(dev.bars[i].size)))
        {
            osMapDeviceMemory(dev.bars[i].bar,
                                dev.bars[i].size,
                                (MEM_PROT_READ | MEM_PROT_WRITE),
                                &(dev.bars[i].pBar));
        }
        else
        {
            dprintf("ERROR: Could not get BAR info for brick #%d\n",
                    brickNum);
            return;
        }
    }

    // Set up the pri ranges for register rd/wr macros
    dev.dlPriBase.priBase = dev.bars[0].bar + IBMNPU_DLPL_BASE_OFFSET;
    dev.dlPriBase.priLimit = dev.dlPriBase.priBase + IBMNPU_DLPL_SIZE;
    dev.dlPriBase.pMap = dev.bars[0].pBar;

    dev.ntlPriBase.priBase = dev.bars[0].bar + IBMNPU_NTL_BASE_OFFSET;
    dev.ntlPriBase.priLimit = dev.ntlPriBase.priBase + IBMNPU_NTL_SIZE;
    dev.ntlPriBase.pMap = ((char *)dev.bars[0].pBar) + IBMNPU_NTL_BASE_OFFSET;

    // Not every brick has a BAR1.
    if (NULL != dev.bars[1].pBar) 
    {
        dev.phyPriBase.priBase = dev.bars[1].bar + IBMNPU_PHY_BASE_OFFSET;
        dev.phyPriBase.priLimit = dev.phyPriBase.priBase + IBMNPU_PHY_SIZE;
        dev.phyPriBase.pMap = dev.bars[1].pBar;

        // Set up the PHY address range for the other brick
        switch (brickNum)
        {
            case 0:
                i = 1;
                break;
            case 1:
                i = 0;
                break;
            case 4:
                i = 5;
                break;
            case 5:
                i = 4;
                break;
            default:
                dprintf("WARNING: Unrecognized brick #%d: Could not determine "
                        "PHY address range for connected brick. " 
                        "PHY register accesses may fail.\n",
                        brickNum);
                break;
        }

        devices[npuIndex][i].phyPriBase.priBase = dev.bars[1].bar + IBMNPU_PHY_BASE_OFFSET;
        devices[npuIndex][i].phyPriBase.priLimit = dev.phyPriBase.priBase + IBMNPU_PHY_SIZE;
        devices[npuIndex][i].phyPriBase.pMap = dev.bars[1].pBar;
    }

    // set BUS_MASTER and MEMORY_ENABLE to allow PCI memory access and
    // to set the device as a bus master
    command = NPU_CFG_RD32(&dev, LW_CONFIG_PCI_LW_1);

    command |= DRF_DEF(_CONFIG, _PCI_LW_1, _MEMORY_SPACE, _ENABLED)
             | DRF_DEF(_CONFIG, _PCI_LW_1, _BUS_MASTER, _ENABLED);

    NPU_CFG_WR32(&dev, LW_CONFIG_PCI_LW_1, command);

    // Read the interrupt line;
    interrupt = NPU_CFG_RD32(&dev, LW_CONFIG_PCI_LW_15);
    dev.intLine = DRF_VAL(_CONFIG, _PCI_LW_15, _INTR_LINE, interrupt);

    dprintf("Identified NPU #%d brick #%d\n", npuIndex, brickNum);

    

    // Populate the device array with the new device.
    devices[npuIndex][brickNum] = dev;
}

BOOL ibmnpuIsInitialized(void)
{
    // Assumption: if we've initialized, brick #0 will have been found.
    if (1 == devices[0][0].initialized)
    {
        return TRUE;
    }

    return FALSE;
}

BOOL ibmnpuLinkExists(LwS32 npu, LwS32 brick)
{
    if (!ibmnpuIsInitialized())
    {
        ibmnpuInit();
    }

    // -1 is a wildcard to target all discovered devices
    if (-1 == npu && -1 == brick)
    {
        return TRUE;
    }
    else if (-1 == npu)
    {
        return FALSE;
    }
    else if (-1 == brick)
    {
        return FALSE;
    }

    if (npu >= 0
            && npu < MAX_IBMNPU_DEVICES
            && brick >= 0
            && brick < MAX_IBMNPU_BRICKS_PER_DEVICE
            && devices[npu][brick].initialized)
    {
        return TRUE;
    }

    return FALSE;
}

void ibmnpuInit(void)
{
    ibmnpuTeardown();
    dprintf("Beginning NPU initialization...\n");

    osPciFindDevices(PCI_DEVICE_ID_IBMNPU, PCI_VENDOR_ID_IBM,
                        &onNpuPciDevice);

    dprintf("NPU initialization complete.\n");
}

void ibmnpuTeardown(void)
{
    LwU32 i, j, k;

    dprintf("Tearing down NPU device structure.\n");

    
    for (i = 0; i < MAX_IBMNPU_DEVICES; ++i)
    {
        for (j = 0; j < MAX_IBMNPU_BRICKS_PER_DEVICE; j++)
        {
            devices[i][j].initialized = 0;
            for (k = 0; k < MAX_IBMNPU_BARS_PER_BRICK; ++k)
            {
                if (NULL != devices[i][j].bars[k].pBar)
                {
                    osUnMapDeviceMemory(devices[i][j].bars[k].pBar,
                                        devices[i][j].bars[k].size);
                    devices[i][j].bars[k].pBar = NULL;
                }
            }
        }
    }

    dprintf("Teardown complete.\n");
}

ibmnpu_control_return ibmnpuControl
(
    ibmnpu_device *device,
    ibmnpu_control_procedure proc
)
{
    LwU32 val = (LwU32) proc;
    LwS32 timeoutUs = 1000000;

    if (NULL == device
            || 0 == device->procControlAddr
            || 0 == device->procStatusAddr)
    {
        return IBMNPU_PROC_PERMANENT_FAILURE; 
    }

    NPU_CFG_WR32(device, device->procControlAddr, val);
    val = NPU_CFG_RD32(device, device->procControlAddr);

    if (val != (LwU32) proc)
    {
        goto ibmnpu_control_abort;
    }

    do
    {
        val = NPU_CFG_RD32(device, device->procStatusAddr);

        if (FLD_TEST_DRF(_IBMNPU, _PROC_STATUS, _BUSY, _COMPLETE, val))
        {
            break;
        }

        osPerfDelay(20);
        timeoutUs -= 20;
    } while (timeoutUs > 0);

    if (timeoutUs == 0)
    {
        dprintf("IBMNPU procedure timed out. Aborting.\n");
        goto ibmnpu_control_abort;
    }

    if (FLD_TEST_DRF(_IBMNPU, _PROC_STATUS, _RET_CODE, _SUCCESS, val))
    {
        return IBMNPU_PROC_SUCCESS;
    }
    else if (FLD_TEST_DRF(_IBMNPU, _PROC_STATUS, _RET_CODE, _TRANS_FAIL, val))
    {
        return IBMNPU_PROC_TRANSIENT_FAILURE;
    }
    else if (FLD_TEST_DRF(_IBMNPU, _PROC_STATUS, _RET_CODE, _PERM_FAIL, val))
    {
        return IBMNPU_PROC_PERMANENT_FAILURE;
    }
    else if (FLD_TEST_DRF(_IBMNPU, _PROC_STATUS, _RET_CODE, _ABORTED, val))
    {
        return IBMNPU_PROC_ABORTED;
    }
    else
    {
        dprintf("Unknown IBMNPU control procedure return code: 0x%x\n",
                DRF_VAL(_IBMNPU, _PROC_STATUS, _RET_CODE, val));
        return IBMNPU_PROC_PERMANENT_FAILURE;
    }

ibmnpu_control_abort:
    val = (LwU32) IBMNPU_PROC_ABORT;
    NPU_CFG_WR32(device, device->procControlAddr, val);

    return IBMNPU_PROC_ABORTED;
}

LwU32 ibmnpuDLPLRead(ibmnpu_device *device, LwU32 offset)
{
    if (NULL == device || !device->initialized || NULL == device->dlPriBase.pMap)
    {
        goto ibmnpuDLPLReadFail;
    }

    if (device->dlPriBase.priBase + offset < device->dlPriBase.priLimit)
    {
        return BYTESWAP_32((*((LwU32 *)(device->dlPriBase.pMap + offset))));
    }
    else
    {
        dprintf("Attempted to read out of bounds NPU DLPL reg offset 0x%x\n",
                    offset);
        goto ibmnpuDLPLReadFail;
    }

ibmnpuDLPLReadFail:
    return 0xFFFFFFFF;
}

void  ibmnpuDLPLWrite(ibmnpu_device *device, LwU32 offset, LwU32 data)
{
    if (NULL == device || !device->initialized || NULL == device->dlPriBase.pMap)
    {
        return;
    }
    
    if (device->dlPriBase.priBase + offset < device->dlPriBase.priLimit)
    {
        (*((LwU32 *)(device->dlPriBase.pMap + offset))) = BYTESWAP_32(data);
    }
    else 
    {
        dprintf("Attempted to write to out of bounds NPU DLPL reg offset 0x%x\n",
                    offset);
    }

}

LwU64 ibmnpuNTLRead(ibmnpu_device *device, LwU64 offset)
{
    if (NULL == device || !device->initialized || NULL == device->ntlPriBase.pMap)
    {
        goto ibmnpuNTLReadFail;
    }

    if (device->ntlPriBase.priBase + offset < device->ntlPriBase.priLimit)
    {
        return (*((LwU64 *)(device->ntlPriBase.pMap + offset)));
    }
    else
    {
        dprintf("Attempted to read out of bounds NPU NTL reg offset 0x%lx\n",
                    (long unsigned int)offset);
        goto ibmnpuNTLReadFail;
    }

ibmnpuNTLReadFail:
    return 0xFFFFFFFFFFFFFFFFULL;
}

void  ibmnpuNTLWrite(ibmnpu_device *device, LwU64 offset, LwU64 data)
{
    if (NULL == device || !device->initialized || NULL == device->ntlPriBase.pMap)
    {
        return;
    }

    if (device->ntlPriBase.priBase + offset < device->ntlPriBase.priLimit)
    {
        (*((LwU64 *)(device->ntlPriBase.pMap + offset))) = data;
    }
    else
    {
        dprintf("Attempted to write to out of bounds NPU NTL reg offset 0x%lx\n",
                    (long unsigned int)offset);
    }
}

LwU8  ibmnpuCfgRead08(ibmnpu_device *device, LwU32 offset)
{
    LwU8 buffer;

    if (NULL == device || !device->initialized)
    {
        goto ibmnpuCfgRead08Fail;
    }

    if (LW_OK != osPciRead08(device->pciInfo.domain,
                                device->pciInfo.bus,
                                device->pciInfo.device,
                                device->pciInfo.func,
                                &buffer, 
                                offset))
    {
        goto ibmnpuCfgRead08Fail;
    }
    return buffer;

ibmnpuCfgRead08Fail:
    return 0xFF;
}

LwU16 ibmnpuCfgRead16(ibmnpu_device *device, LwU32 offset)
{
    LwU16 buffer;

    if (NULL == device || !device->initialized)
    {
        goto ibmnpuCfgRead16Fail;
    }

    if (LW_OK != osPciRead16(device->pciInfo.domain,
                                device->pciInfo.bus,
                                device->pciInfo.device,
                                device->pciInfo.func,
                                &buffer, 
                                offset))
    {
        goto ibmnpuCfgRead16Fail;
    }
    return buffer;

ibmnpuCfgRead16Fail:
    return 0xFFFF;
}

LwU32 ibmnpuCfgRead32(ibmnpu_device *device, LwU32 offset)
{
    LwU32 buffer;

    if (NULL == device || !device->initialized)
    {
        goto ibmnpuCfgRead32Fail;
    }

    if (LW_OK != osPciRead32(device->pciInfo.domain,
                                device->pciInfo.bus,
                                device->pciInfo.device,
                                device->pciInfo.func,
                                &buffer, 
                                offset))
    {
        goto ibmnpuCfgRead32Fail;
    }
    return buffer;

ibmnpuCfgRead32Fail:
    return 0xFFFFFFFF;
}

void  ibmnpuCfgWrite08(ibmnpu_device *device, LwU32 offset, LwU8 data)
{
    if (NULL == device || !device->initialized)
    {
        return;
    }

    // Not supported yet
    return;
}

void  ibmnpuCfgWrite16(ibmnpu_device *device, LwU32 offset, LwU16 data)
{
    if (NULL == device || !device->initialized)
    {
        return;
    }

    // Not supported yet
    return;
}

void  ibmnpuCfgWrite32(ibmnpu_device *device, LwU32 offset, LwU32 data)
{
    if (NULL == device || !device->initialized)
    {
        return;
    }

    osPciWrite32(device->pciInfo.domain,
                    device->pciInfo.bus,
                    device->pciInfo.device,
                    device->pciInfo.func,
                    data,
                    offset);
}

void ibmnpuCommandLINKS(void)
{
    LwU32 i, j;

    LWL_STATUS  linkStatus;
    LWL_TYPE    linkType;
    LWL_POWER   linkPower;
    LWL_MODE    linkRxMode;
    LWL_MODE    linkTxMode;

    LwU32 txStatus;
    LwU32 rxStatus;

    if (!ibmnpuIsInitialized())
    {
        ibmnpuInit();
    }

    dprintf("\n");
    dprintf("LINK   STATUS  TYPE          POWER   RX_MODE TX_MODE\n");
    dprintf("----------------------------------------------------\n");

    for (i = 0; i < MAX_IBMNPU_DEVICES; i++) 
    {
        for (j = 0; j < MAX_IBMNPU_BRICKS_PER_DEVICE; j++)
        {
            linkStatus = LWL_OFF;
            linkType = LWL_T_NA;
            linkPower = LWL_POFF;
            linkRxMode = LWL_NA;
            linkTxMode = LWL_NA;

            txStatus = 0;
            rxStatus = 0;

            if (!devices[i][j].initialized)
            {
                continue;
            }
        
            txStatus = NPU_DLPL_RD32(&devices[i][j], LW_PLWL_SL0_SLSM_STATUS_TX);
            if (FLD_TEST_DRF(_PLWL, _SL0_SLSM_STATUS_TX, _PRIMARY_STATE, _HS, txStatus))                
            {
                linkTxMode = LWL_FAST;
            }
            else if (FLD_TEST_DRF(_PLWL, _SL0_SLSM_STATUS_TX, _PRIMARY_STATE, _SAFE, txStatus))
            {
                linkTxMode = LWL_SAFE;
            }
            else
            {
                linkTxMode = LWL_NA;
            }

            rxStatus = NPU_DLPL_RD32(&devices[i][j], LW_PLWL_SL1_SLSM_STATUS_RX);
            if (FLD_TEST_DRF(_PLWL, _SL1_SLSM_STATUS_RX, _PRIMARY_STATE, _HS, rxStatus))                
            {
                linkRxMode = LWL_FAST;
            }
            else if (FLD_TEST_DRF(_PLWL, _SL1_SLSM_STATUS_RX, _PRIMARY_STATE, _SAFE, rxStatus))
            {
                linkRxMode = LWL_SAFE;
            }
            else
            {
                linkRxMode = LWL_NA;
            }
            
            // Assumption: NPU links are always sysmem links.
            linkType = LWL_SYSMEM;

            // Assumption: NPU links are always on, and never in reset.
            linkPower = LWL_FULL;

            if (LWL_SAFE == linkTxMode || LWL_SAFE == linkRxMode)
            {
                linkStatus = LWL_S_SAFE;
            }
            else if (  LWL_FAST == linkRxMode
                    && LWL_FAST == linkTxMode
                    && LWL_FULL == linkPower
                    && LWL_SYSMEM == linkType
                    )
            {
                linkStatus = LWL_NORMAL;
            }

            dprintf("%-7d%-8s%-14s%-8s%-8s%-7s\n", 
                j,                              // Link Num
                lwlinkStatusText(linkStatus),   // Status
                lwlinkTypeText(linkType),       // Link Type
                lwlinkPowerText(linkPower),     // Powered
                lwlinkModeText(linkRxMode),     // Rx mode
                lwlinkModeText(linkTxMode));    // Tx mode
        }
    }
    dprintf("\n");

    ibmnpuTeardown();
}

void ibmnpuCommandDEVICES(void)
{
    LwU32 i, j, k;

    if (!ibmnpuIsInitialized())
    {
        ibmnpuInit();
    }

    dprintf("Printing PCI info for all discovered devices...\n");

    for (i = 0; i < MAX_IBMNPU_DEVICES; ++i)
    {
        for (j = 0; j < MAX_IBMNPU_BRICKS_PER_DEVICE; ++j)
        {
            if (devices[i][j].initialized)
            {
                dprintf("NPU device at (%04d:%02d:%02d.%02d)\n",
                            devices[i][j].pciInfo.domain,
                            devices[i][j].pciInfo.bus,
                            devices[i][j].pciInfo.device,
                            devices[i][j].pciInfo.func);

                for (k = 0; k < MAX_IBMNPU_BARS_PER_BRICK; k++)
                {
                    if (0x0 != devices[i][j].bars[k].bar)
                    {
                        dprintf("    Memory at %08lx (size = %lx)\n",
                                    (unsigned long)devices[i][j].bars[k].bar,
                                    (unsigned long)devices[i][j].bars[k].size);
                    }
                }

                dprintf("    Interrupt line 0x%x\n", devices[i][j].intLine);
                dprintf("    Proc Status Addr: 0x%x\n", devices[i][j].procStatusAddr);
                dprintf("    Proc Ctrl Addr: 0x%x\n", devices[i][j].procControlAddr);
            }
        }
    }

    dprintf("Completed walking device list.\n");
    ibmnpuTeardown();
}

void ibmnpuCommandREAD(LwS32 npu, LwS32 brick, LwU32 type, LwU64 offset)
{
    ibmnpu_device *device;
    
    if (!ibmnpuLinkExists(npu, brick))
    {
        ibmnpuMissingLink(npu, brick);
        return;
    }


    device = &devices[npu][brick];

    switch (type)
    {
        case 0:
            dprintf("[%016lx] %08x\n",
                    (long)offset,
                    NPU_DLPL_RD32(device, (LwU32)offset));
            break;
        case 1:
            dprintf("[%016lx] %016lx\n",
                    (long)offset,
                    (long)NPU_NTL_RD64(device, offset));
            break;
        default:
            dprintf("register space %d unsupported. \n", type);
            return;
    }
}

void ibmnpuCommandWRITE(LwS32 npu, LwS32 brick, LwU32 type, LwU64 offset, LwU64 data)
{
    ibmnpu_device *device;
    
    if (!ibmnpuLinkExists(npu, brick))
    {
        ibmnpuMissingLink(npu, brick);
        return;
    }

    device = &devices[npu][brick];

    switch (type)
    {
        case 0:
            NPU_DLPL_WR32(device, (LwU32)offset, (LwU32)data);
            break;
        case 1:
            NPU_NTL_WR64(device, offset, data);
            break;
        default:
            dprintf("register space %d unsupported. \n", type);
            return;
    }
}

void ibmnpuCommandCTRL(LwS32 npu, LwS32 brick, LwU32 proc)
{
    ibmnpu_device *device;
    LwU32 i, j;
    if (!ibmnpuLinkExists(npu, brick))
    {
        ibmnpuMissingLink(npu, brick);
        return;
    }

    
    for (i = 0; i < MAX_IBMNPU_DEVICES; i++)
    {
        for (j = 0; j < MAX_IBMNPU_BRICKS_PER_DEVICE; j++)
        {
            if (!devices[i][j].initialized 
                    || (npu != -1 && ((LwS32)i) != npu) 
                    || (brick != -1 && ((LwS32)j) != brick))
            {
                continue;
            }
    
            device = &devices[npu][brick];
            dprintf("Performing IBMNPU control command 0x%x\n", proc);

            switch (proc)
            {
                case IBMNPU_PROC_ABORT:
                case IBMNPU_PROC_NOOP:
                case IBMNPU_PROC_START_DMA_SYNC:
                case IBMNPU_PROC_RESET_AT:
                case IBMNPU_PROC_NAPLES_PHY_RESET:
                case IBMNPU_PROC_NAPLES_PHY_TX_ZCAL:
                case IBMNPU_PROC_NAPLES_PHY_RX_DCCAL:
                case IBMNPU_PROC_NAPLES_PHY_TX_RXCAL_ENABLE:
                case IBMNPU_PROC_NAPLES_PHY_TX_RXCAL_DISABLE:
                case IBMNPU_PROC_NAPLES_PHY_RX_TRAINING:
                case IBMNPU_PROC_NAPLES_PHY_DL_RESET:
                    dprintf("Procedure returned with status code 0x%x\n",
                            ibmnpuControl(device, (ibmnpu_control_procedure)proc));
                    break;
                default:
                    dprintf("Unknown IBMNPU proc 0x%x\n", proc);
                    break;
            }
        }
    }

    ibmnpuTeardown();
}

void ibmnpuCommandDUMPDLPL(LwS32 npu, LwS32 brick)
{
    LwU32 val;
    LwU32 i, j;

    if (!ibmnpuLinkExists(npu, brick))
    {
        ibmnpuMissingLink(npu, brick);
        return;
    }

    dprintf("***************************************************************\n");
    dprintf("======================\n");
    dprintf("Detailed LWlink Status\n");
    dprintf("======================\n\n");

    
    for (i = 0; i < MAX_IBMNPU_DEVICES; i++)
    {
        for (j = 0; j < MAX_IBMNPU_BRICKS_PER_DEVICE; j++)
        {
            if (!devices[i][j].initialized 
                    || (npu != -1 && ((LwS32)i) != npu)
                    || (brick != -1 && ((LwS32)j) != brick))
            {
                continue;
            }

            // TODO if link is in reset, skip.

            dprintf("==========================\n");
            dprintf("===  NPU %d  LINK %d  ====\n", i, j);
            dprintf("==========================\n");

            
            val = NPU_DLPL_RD32( &devices[i][j], LW_PLWL_ERROR_COUNT1 );
            dprintf("LW_PLWL_ERROR_COUNT1               val = 0x%x\n", val);
            val = NPU_DLPL_RD32( &devices[i][j], LW_PLWL_SL0_ERROR_COUNT4 );
            dprintf("LW_PLWL_SL0_ERROR_COUNT4           val = 0x%x\n", val);
            val = NPU_DLPL_RD32( &devices[i][j], LW_PLWL_SL1_ERROR_COUNT1 );
            dprintf("LW_PLWL_SL1_ERROR_COUNT1           val = 0x%x\n", val);
            val = NPU_DLPL_RD32( &devices[i][j], LW_PLWL_SL1_ERROR_COUNT2_LANECRC );
            dprintf("LW_PLWL_SL1_ERROR_COUNT2_LANECRC   val = 0x%x\n", val);
            val = NPU_DLPL_RD32( &devices[i][j], LW_PLWL_SL1_ERROR_COUNT3_LANECRC );
            dprintf("LW_PLWL_SL1_ERROR_COUNT3_LANECRC   val = 0x%x\n", val);
            val = NPU_DLPL_RD32( &devices[i][j], LW_PLWL_SL1_ERROR_COUNT5 );
            dprintf("LW_PLWL_SL1_ERROR_COUNT5           val = 0x%x\n", val);
            val = NPU_DLPL_RD32( &devices[i][j], LW_PLWL_INTR );
            dprintf("LW_PLWL_INTR                       val = 0x%x\n", val);
            val = NPU_DLPL_RD32( &devices[i][j], LW_PLWL_INTR_STALL_EN );
            dprintf("LW_PLWL_INTR_STALL_EN              val = 0x%x\n", val);
            val = NPU_DLPL_RD32( &devices[i][j], LW_PLWL_INTR_NONSTALL_EN );
            dprintf("LW_PLWL_INTR_NONSTALL_EN           val = 0x%x\n", val);
            val = NPU_DLPL_RD32( &devices[i][j], LW_PLWL_LINK_STATE );
            dprintf("LW_PLWL_LINK_STATE                 val = 0x%x\n", val);
        }
    }

    ibmnpuTeardown();
}

void ibmnpuCommandDUMPNTL(LwS32 npu, LwS32 brick)
{
    if (!ibmnpuLinkExists(npu, brick))
    {
        ibmnpuMissingLink(npu, brick);
        return;
    }

    ibmnpuCommandUnimplemented();
    ibmnpuTeardown();
}

void ibmnpuCommandDUMPUPHYS(LwS32 npu, LwS32 brick)
{
    if (!ibmnpuLinkExists(npu, brick))
    {
        ibmnpuMissingLink(npu, brick);
        return;
    }

    ibmnpuCommandUnimplemented();
    ibmnpuTeardown();
}

void ibmnpuPrintHelp(void)
{
    dprintf("ibmnpu command.\n");
    dprintf("usage: lwv ibmnpu  <# of args> [option] <command>\n");
    dprintf("Commands:\n");
    dprintf(" '-links'                                          : Displays a list of NPU hardware link ids\n");
    dprintf(" '-devices'                                        : Displays all IBMNPU PCI device info\n");
    dprintf(" '-read' <npu#> <link#> <type> <offset>            : Reads a register from the given register space. (0=DL,1=NTL)\n");
    dprintf(" '-write' <npu#> <link#> <type> <offset> <data>    : Writes the given data to the register at the given offset in register space (0=DL,1=NTL)\n");
    dprintf(" '-ctrl [<npu#> <link#>] <proc>'                   : Utilizes the procedure ctrl registers to perform devinit.\n");
    dprintf(" '-dumpdlpl [<npu#> <link#>]                       : Dumps the DL/PL register set of the selected device\n");
    dprintf(" '-dumpntl [<npu#> <link#>]                        : Dumps the NTL register set of the selected device.\n");
    dprintf(" '-dumpuphys [<npu#> <link#>]                      : Dumps the PHY register set of the selected device.\n");
    dprintf("Options:\n");
    dprintf(" '-v'                                              : Verbose mode - Displays more useful info.\n");
}

void ibmnpuMissingLink(LwS32 npu, LwS32 brick)
{
    if (npu < -1)
    {
        dprintf("Only positive npu indices are supported.\n");
    }
    if (brick < -1)
    {
        dprintf("Only positive link ids are supported.\n");
    }
    else
    {
        dprintf("Could not find npu #%d link #%d.\n", npu, brick);
    }
}

void ibmnpuCommandUnimplemented(void)
{
    dprintf("Command not implemented lwrrently.\n");
}
