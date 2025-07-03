/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2005-2014 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

//*****************************************************
//
// lwwatch WinDbg Extension
// pex.c
//
//*****************************************************

//
// includes
//
#include "lwwatch_pex.h"
#include "hwref/lw_pex.h"
#include "print.h"
#include "hwref/lwutil.h"
#include "hwref/dev_lw_xp.h"

//
// globals
//
LwU32 FHBConfigBaseAddr;
BOOL ConfigBaseFound;

// Entry with 0 Vendor ID ends the table
CSINFO chipsetInfoArray[] =
{
    // PCI Express chipset
    {PCI_VENDOR_ID_INTEL,       0x2580,      "Grantsdale",   Intel_25XX_setupFunc},
    {PCI_VENDOR_ID_INTEL,       0x2584,      "Alderwood",    Intel_25XX_setupFunc},
    {PCI_VENDOR_ID_INTEL,       0x2588,      "Intel2588",    Intel_25XX_setupFunc},
    {PCI_VENDOR_ID_INTEL,       0x2590,      "Alviso",       Intel_25XX_setupFunc},
    {PCI_VENDOR_ID_INTEL,       0x359E,      "Tumwater",     Intel_359E_setupFunc},
    {PCI_VENDOR_ID_INTEL,       0x2770,      "Lakeport",     Intel_25XX_setupFunc},
    {PCI_VENDOR_ID_INTEL,       0x2774,      "Glenwood",     Intel_27XX_setupFunc},
    {PCI_VENDOR_ID_INTEL,       0x25E0,      "Glreencreek",     NULL},

    {PCI_VENDOR_ID_LWIDIA,      0x005E,      "nForce4",      Lwidia_CK804_setupFunc},
    {PCI_VENDOR_ID_LWIDIA,      0x0070,      "nForce4 Intel Edition",     Lwidia_C19_setupFunc},
    {PCI_VENDOR_ID_LWIDIA,      0x0071,      "nForce4 Intel Edition",     Lwidia_C19_setupFunc},
    {PCI_VENDOR_ID_LWIDIA,      0x0072,      "nForce4 Intel Edition",     Lwidia_C19_setupFunc},
    {PCI_VENDOR_ID_LWIDIA,      0x0073,      "nForce4 Intel Edition",     Lwidia_C19_setupFunc},
    {PCI_VENDOR_ID_LWIDIA,      0x02F0,      "C51",          Lwidia_C51_setupFunc},
    {PCI_VENDOR_ID_LWIDIA,      0x02F4,      "C51",          Lwidia_C51_setupFunc},
    
    {PCI_VENDOR_ID_SIS,         0x0649,      "649",          SiS_656_setupFunc},
    {PCI_VENDOR_ID_SIS,         0x0656,      "656",          SiS_656_setupFunc},
    {0,                         0,           "Unknown",          NULL}
};

CAP_TABLE capTable[] = 
{
    {PCI_VENDOR_ID_LWIDIA,      0x007E,                         PCIE_AERR_CAP_ID,    0x160},
    {PCI_VENDOR_ID_INTEL,       DEVICE_ID_INTEL_2581_ROOT_PORT, PCIE_AERR_CAP_ID,    0x1C0},
    {PCI_VENDOR_ID_INTEL,       DEVICE_ID_INTEL_2585_ROOT_PORT, PCIE_AERR_CAP_ID,    0x1C0},
    {PCI_VENDOR_ID_INTEL,       0x2589,                         PCIE_AERR_CAP_ID,    0x1C0},
    {PCI_VENDOR_ID_INTEL,       DEVICE_ID_INTEL_2591_ROOT_PORT, PCIE_AERR_CAP_ID,    0x1C0},
    {PCI_VENDOR_ID_INTEL,       DEVICE_ID_INTEL_3597_ROOT_PORT, PCIE_AERR_CAP_ID,    PCIE_ILWALID_CAPABILITY_OFFSET},
    {PCI_VENDOR_ID_INTEL,       0x2775,                         PCIE_AERR_CAP_ID,    0x1C0},
    {PCI_VENDOR_ID_INTEL,       0x2771,                         PCIE_AERR_CAP_ID,    0x1C0},
    {0,                         0,                              0,              PCIE_ILWALID_CAPABILITY_OFFSET}

};

//
// statics
//
static void _getAllInfo(void);
static void _getPexGpuInfo(void);
static void _getSpecifiedInfo(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func);
static int _findGpu(LwU16 * pdomain, LwU8 * pbus, LwU8 * pdev, LwU8 * pfunc);
static int _findUpStreamPort(LwU16 LwrrDom, LwU8 LwrrBus, LwU16 *pUpDom, LwU8 *pUpBus, LwU8 *pUpDev, LwU8 *pUpFunc);
static void _findDevVenID(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func, LwU16 *pVenID, LwU16 *pDevID);
static BOOL _findFHB(LwU16 * pFHBdomain, LwU8 *pFHBbus, LwU8 *pFHBdev, LwU8 *pFHBfunc, LwU16 *pFHBVenID, LwU16 *pFHBDevID);
void static _displayPCIStdHdr(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func);
void static _displayPCIECap(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func);
void static _displayVCCap(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func);
void static _displayAERRCap(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func);
void static _displayAllCapInfo(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func);
static LwU8 _DevType(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func);
static void _displayPortType(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func);
static void _getFHBInfo(void);

//-----------------------------------------------------
// getPexInfo
//
//   ext.c call this when the user specified the "all"
//   argument -- will look at all the PCI-E links in the
//   hiarechy from the GPU to the Root Port
//-----------------------------------------------------
void printPci(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func, LwU32 offset, LwU32 length)
{
    LwU32 BAR0;
    PhysAddr Addr;

    _getFHBInfo();

    osPciRead32( domain, bus, dev, func, &BAR0, PCIE_BAR0_OFFSET);

    if(ConfigBaseFound == TRUE)
    {
        Addr = (Get_PCI_Config_Base(FHBConfigBaseAddr, domain, bus, dev, func) + offset);
        printData(Addr, length);
    }
    else if(BAR0 == lwBar0)
    {
        Addr = lwBar0 + 0x00088000 + offset;
        printData(Addr, length);
    }
}



//-----------------------------------------------------
// printPcie3EvtLogDmp
//
//   ext.c call this when the user want to print Event Dump
//   argument -- None
//   hierarchy from the GPU to the Root Port
//-----------------------------------------------------
void printPcie3EvtLogDmp(void)
{
    unsigned ramWritePtr = 0;
    unsigned ramSize = 768;

    unsigned start_address = 0;
    unsigned end_address = 0;

    unsigned i = 0;
    unsigned logDW0, logDW1, logDW2;

    ramWritePtr = GPU_REG_RD_DRF(_XP3G, _LA_CR_2, _RAM_WRITE_POINTER);

    GPU_FLD_WR_DRF_DEF(_XP3G, _LA_CR_2, _RAM_READ_MODE, _TRACE);
    if(LW_XP3G_LA_CR_2_RAM_WRAP_AROUND_MODE_ENABLED == GPU_REG_RD_DRF(_XP3G, _LA_CR_2, _RAM_WRAP_AROUND_MODE) 
        && TRUE == GPU_REG_RD_DRF(_XP3G, _LA_CR_2, _RAM_FULL))
    {
        start_address = ramWritePtr;
        end_address = ramWritePtr + ramSize;
    }
    else
    {
        start_address = 0;
        end_address = ramWritePtr;
    }

    dprintf("\nFollowing is the log dump in HEX...!!");
    dprintf("\n\nEvent Log Dump \n==================");
    for (i = start_address; i < end_address; i++) 
    { 
        GPU_FLD_WR_DRF_NUM(_XP3G, _LA_CR_2, _RAM_READ_ADDRESS, (i % ramSize));

        logDW0 = GPU_REG_RD_DRF(_XP3G, _LA_EVENT_LOG_DW0, _0);
        logDW1 = GPU_REG_RD_DRF(_XP3G, _LA_EVENT_LOG_DW1, _1);
        logDW2 = GPU_REG_RD_DRF(_XP3G, _LA_EVENT_LOG_DW2, _2);

        dprintf("\n0x%08x : %08x %08x %08x", (i % ramSize), logDW2, logDW1, logDW0);
    } 
    dprintf("\n\nTimeStamps \n==================");
    GPU_FLD_WR_DRF_DEF(_XP3G, _LA_CR_2, _RAM_READ_MODE, _SAMPLE_INTERVALS);
    for (i = 0; i < 3; i++) 
    { 
        GPU_FLD_WR_DRF_NUM(_XP3G, _LA_CR_2, _RAM_READ_ADDRESS, i);

        logDW0 = GPU_REG_RD_DRF(_XP3G, _LA_EVENT_LOG_DW0, _0);
        logDW1 = GPU_REG_RD_DRF(_XP3G, _LA_EVENT_LOG_DW1, _1);
        logDW2 = GPU_REG_RD_DRF(_XP3G, _LA_EVENT_LOG_DW2, _2);

        dprintf("\n0x%08x : %08x %08x %08x", i, logDW2, logDW1, logDW0);
    } 
    dprintf("\n");
}


//-----------------------------------------------------
// getPexInfo
//
//   ext.c call this when the user specified the "all"
//   argument -- will look at all the PCI-E links in the
//   hiarechy from the GPU to the Root Port
//-----------------------------------------------------
void getPexAllInfo(void)
{
    _getFHBInfo();
    _getAllInfo();
}

//-----------------------------------------------------
// getPexInfo
//
//   ext.c call this when the user did specify a bus,
//   dev, func -- implying to look at the PCI-E link 
//   with the device with the given bus/dev/func
//-----------------------------------------------------
void getPexSpecifiedInfo(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func)
{
    _getFHBInfo();
    _getSpecifiedInfo(domain, bus, dev, func);
}

//-----------------------------------------------------
// getPexGPUInfo
//
//   ext.c call this when the user did not specify any
//   argument -- implying to look at the PCI-E link 
//   the GPU called in the init function
//-----------------------------------------------------
void getPexGPUInfo(void)
{
    _getFHBInfo();
    _getPexGpuInfo();
}

//-----------------------------------------------------
// _getFHBInfo
//
//    Gets the First host bridge info and call the
//    relative function to obtain the base address of
//    the PCI Config Space
//-----------------------------------------------------
static void _getFHBInfo(void)
{
    LwU16 FHBdomain = 0;
    LwU8  FHBbus = 0, FHBdev = 0, FHBfunc = 0;
    LwU16 FHBVenID = 0, FHBDevID = 0;
    LwU16 i;

    ConfigBaseFound = FALSE; // initialization

    ConfigBaseFound = _findFHB(&FHBdomain, &FHBbus, &FHBdev, &FHBfunc, &FHBVenID, &FHBDevID);
    if(ConfigBaseFound == TRUE)
        dprintf("lw: FHB  :  DOMAIN  %04x  BUS  %02x  DEVICE  %02x  FUNCTION  %02x\n", FHBdomain, FHBbus, FHBdev, FHBfunc);

    // Search for the correct index to the chipset table
    for (i=0; chipsetInfoArray[i].vendorID && !(chipsetInfoArray[i].vendorID==FHBVenID && chipsetInfoArray[i].deviceID == FHBDevID) ;i++);

    if(chipsetInfoArray[i].vendorID!=0 && chipsetInfoArray[i].setupFunc(FHBdomain, FHBbus, FHBdev, FHBfunc, &FHBConfigBaseAddr) == LW_OK)
        dprintf("lw: FHBConfig Base Address is 0x%08x\n", FHBConfigBaseAddr);
    else if(FindCfgBase(FHBdomain, FHBbus, FHBdev, FHBfunc, &FHBConfigBaseAddr) == LW_OK)
        dprintf("lw: FHBConfig Base Address is 0x%08x\n", FHBConfigBaseAddr);
    else
    {
        dprintf("lw: Warning: FHB Config Base Address Not Found\n");
        ConfigBaseFound = FALSE;
    }

    // Finish finding FHB and base address
}

//-----------------------------------------------------
// _getPexGpuInfo
//
// + Prints out PCI-E info ofthe GPU
//-----------------------------------------------------
static void _getPexGpuInfo(void)
{
    LwU16 domain = 0, updomain = 0;
    LwU8  bus = 0, dev = 0, func = 0, upbus = 0, updev = 0, upfunc = 0;

    dprintf("lw: Searching for device and upstream information...\n");

    // Go get the GPU bus/dev/func
    if(_findGpu(&domain, &bus, &dev, &func)==LW_OK)
        dprintf("lw: GPU:  DOMAIN  %04x  BUS  %02x  DEVICE  %02x  FUNCTION  %02x\n", domain, bus, dev, func);
    else
        dprintf("lw: ERROR not found\n");

    // Go get the Upstream Port bus/dev/func
    if(_findUpStreamPort(domain, bus, &updomain, &upbus, &updev, &upfunc)==LW_ERR_GENERIC)
        dprintf("lw: ERROR not found\n");

    dprintf("\n\n");

    _displayPortType(domain, bus, dev, func);
    _displayAllCapInfo(domain, bus, dev, func);
    _displayPortType(updomain, upbus, updev, upfunc);
    _displayAllCapInfo(updomain, upbus, updev, upfunc);
}

//-----------------------------------------------------
// getPexInfo
//
//   display the pci express link specified by domain/bus/dev/func
//
//-----------------------------------------------------
static void _getSpecifiedInfo(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func)
{
    LwU16 nextdomain = 0;
    LwU8 nextbus = 0, nextdev = 0, nextfunc = 0;
    LwU8 offset;
    LW_STATUS status;

    dprintf("lw: Searching...\n");

    // Check if the user enters a bus/dev/func that is not a valid PCI Express device
    status = FindStdCapOffset(domain, bus, dev, func, PCIE_CAP_ID, &offset);
    if(offset == PCIE_ILWALID_CAPABILITY_OFFSET || status == LW_ERR_GENERIC)
    {
        dprintf("lw: DOMAIN  %04x  BUS  %02x  DEVICE  %02x  FUNCTION  %02x  is not a PCI Express device\n", domain, bus, dev, func);
        return;
    }

    if(!(_DevType(domain, bus, dev, func) == DW_PT || _DevType(domain, bus, dev, func) == ROOT_PT))
    {
        // Go get the Upstream Port bus/dev/func
        if(_findUpStreamPort(domain, bus, &nextdomain, &nextbus, &nextdev, &nextfunc)==LW_ERR_GENERIC)
            dprintf("lw: ERROR Upstream not found\n");
    }
    else if(_DevType(domain, bus, dev, func) == DW_PT || _DevType(domain, bus, dev, func) == ROOT_PT)
    {
        // Go down the hiearchy to get bus/dev/func

        // First need the secondary bus.... this is the bus# we wanna!!
        nextbus = (LwU8) PcieRead32(domain, bus, dev, func, 0x19);
        nextdev = 0;
        nextfunc = 0;
    }

    dprintf("\n\n");

    _displayPortType(domain, bus, dev, func);
    _displayAllCapInfo(domain, bus, dev, func);
    _displayPortType(nextdomain, nextbus, nextdev, nextfunc);
    _displayAllCapInfo(nextdomain, nextbus, nextdev, nextfunc);
}

//-----------------------------------------------------
// _getAllInfo
//
//    displays the entire PCI-Express hiearchy
//
//-----------------------------------------------------
static void _getAllInfo(void)
{
    LwU16 domain = 0, nextdomain = 0;
    LwU8 bus = 0, dev = 0, func = 0;
    LwU8 nextbus = 0, nextdev = 0, nextfunc = 0;
    BOOL ReachTop = FALSE;

    // Go get the GPU bus/dev/func
    if(_findGpu(&domain, &bus, &dev, &func)==LW_OK)
        dprintf("lw: GPU:  DOMAIN  %04x  BUS  %02x  DEVICE  %02x  FUNCTION  %02x\n", domain, bus, dev, func);

    else
        dprintf("lw: ERROR GPU not found\n");

    _displayPortType(domain, bus, dev, func);
    _displayAllCapInfo(domain, bus, dev, func);

    while(ReachTop == FALSE)
    {
        // Go get the Upstream Port bus/dev/func
        if(_findUpStreamPort(domain, bus, &nextdomain, &nextbus, &nextdev, &nextfunc)==LW_OK)
            _displayPortType(nextdomain, nextbus, nextdev, nextfunc);
        else
            dprintf("lw: ERROR not found\n");

        domain = nextdomain;
        bus = nextbus;
        dev = nextdev;
        func = nextfunc;
        _displayAllCapInfo(domain, bus, dev, func);

        if((domain == 0 && bus == 0 && func == 0 && dev == 0)||(_DevType(domain,bus,dev,func)==ROOT_PT))
            ReachTop = TRUE;
    }
}

//-----------------------------------------------------
// _displayPortType
//
//    Displays the type of port given by domain bus dev func
//
//-----------------------------------------------------
static void _displayPortType(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func)
{
    switch(_DevType(domain, bus,dev,func))
    {  
    case PCIE_ENDPT: 
        dprintf("lw:  Info of  DOMAIN  %04x  BUS  %02x  DEVICE  %02x  FUNCTION  %02x\n", domain, bus, dev, func);
        dprintf("lw:  PCI EXPRESS ENDPOINT\n");
        dprintf("lw:  DOWNSTREAM Port of the PCI Express Link\n");
        break;

    case LEG_ENDPT: 
        dprintf("lw:  Info of  DOMAIN  %04x  BUS  %02x  DEVICE  %02x  FUNCTION  %02x\n", domain, bus, dev, func);
        dprintf("lw:  LEGACY PCI EXPRESS ENDPOINT\n");
        dprintf("lw:  DOWNSTREAM Port of the PCI Express Link\n");
        break;

    case ROOT_PT: 
        dprintf("lw:  Info of  DOMAIN  %04x  BUS  %02x  DEVICE  %02x  FUNCTION  %02x\n", domain, bus, dev, func);
        dprintf("lw:  ROOT PORT OF PCI EXPRESS ROOT COMPLEX\n");
        dprintf("lw:  UPSTREAM Port of the PCI Express Link\n");
        break;

    case UP_PT: 
        dprintf("lw:  Info of  DOMAIN  %04x  BUS  %02x  DEVICE  %02x  FUNCTION  %02x\n", domain, bus, dev, func);
        dprintf("lw:  SWITCH UPSTREAM PORT\n");
        dprintf("lw:  DOWNSTREAM Port of the PCI Express Link\n");
        break;

    case DW_PT: 
        dprintf("lw:  Info of  DOMAIN  %04x  BUS  %02x  DEVICE  %02x  FUNCTION  %02x\n", domain, bus, dev, func);
        dprintf("lw:  SWITCH DOWNSTREAM PORT\n");
        dprintf("lw:  UPSTREAM Port of the PCI Express Link\n");
        break;

    default: break;
    }
}

//-----------------------------------------------------
// _findGpu
//
//    Finds the domain/bus/dev/func of the upstream port 
//
//-----------------------------------------------------
static int _findGpu(LwU16 *pdomain, LwU8 *pbus, LwU8 *pdev, LwU8 *pfunc)
{
    LwU32 BAR0;
    BOOL found = FALSE;
    LwU32 domain;
    LwU16 bus;
    LwU8 dev;

    // For a GPU, the function number must be 0.
    for(domain = 0; domain < PCIE_MAX_DOMAIN ; domain++)
    {
        for(bus = 0; bus <  PCIE_MAX_BUS ; bus++)
        {
            for(dev = 0; dev <  PCIE_MAX_DEV ; dev++)
            {
                    osPciRead32( (LwU16) domain, (LwU8) bus, dev, 0, &BAR0, PCIE_BAR0_OFFSET);
                    // check if the BAR0 of this (bus,dev,func) the same as the global variable
                    if(lwBar0 == BAR0)
                    {
                        found = TRUE;
                        *pdomain = (LwU16) domain;
                        *pbus = (LwU8) bus;
                        *pdev = dev;
                        *pfunc = 0;
                        // if the BAR0 are the same, we found it
                        break;
                    }
    
                if(found)
                    break;
            }
            if(found)
                break;
        }
        if(found)
            break;
    }

    if(found)
        return LW_OK;   // success
    else return LW_ERR_GENERIC;  // error
}

//-----------------------------------------------------
// _findUpStreamPort
//
//    Finds the domain/bus/dev/func of the upstream port 
//    (Relative to the PCI Express link)
//-----------------------------------------------------
static int _findUpStreamPort(LwU16 LwrrDomain, LwU8 LwrrBus, LwU16 *pUpDomain, LwU8 *pUpBus, LwU8 *pUpDev, LwU8 *pUpFunc)
{
    BOOL found = FALSE;
    LwU8 secBus;
    LwS16 domain;
    LwS16 bus;
    LwU8 dev, func;
    LwU32 HeaderTypeReg;     // To see if it is single or multifunc -- efficiency!!!

    for(domain = LwrrDomain; domain >= 0 ; domain--)
    {
        for(bus = LwrrBus - 1; bus >= 0 ; bus--)
        {
            for(dev = 0; dev <  PCIE_MAX_DEV ; dev++)
            {
                for(func=0; func <  PCIE_MAX_FUNC ; func++)
                {
                    osPciRead08( (LwU16) domain, (LwU8) bus, dev, func, &secBus, 0x19);
                    // check if the secondary bus of this (bus,dev,func) the same as the bus# of the GPU
                    if(LwrrBus == secBus)
                    {
                        found = TRUE;
                        *pUpDomain = (LwU16) domain;
                        *pUpBus = (LwU8) bus;
                        *pUpDev = dev;
                        *pUpFunc = func;
                        // if the bus numbers are the same, we found it
                        break;
                    }
    
                    osPciRead32( (LwU16) domain, (LwU8) bus, dev, func, &HeaderTypeReg, LW_XVE_MISC_1);
                    if(func == 0 && DRF_VAL(_XVE, _MISC_1, _HEADER_TYPE, HeaderTypeReg) == LW_XVE_MISC_1_HEADER_TYPE_MULTIFUNC)
                        break;
                }
    
                // When we break, we found the correct domain,bus,dev,func# for the upstream port
                
                if(found)
                    break;
            }
            if(found)
                break;
        }
        if(found)
            break;
    }

    if(found == TRUE)
        return LW_OK;   // success
    else return LW_ERR_GENERIC;  // error
}

//-----------------------------------------------------
// _findDevVenID
//
//   Gets the Vendor and Device ID
//   Takes in Domain/Bus/Dev/Func as arguments
//-----------------------------------------------------
static void _findDevVenID(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func, LwU16 *pVenID, LwU16 *pDevID)
{
    LwU32 result;
    osPciRead32( domain, bus, dev, func, &result, 0x0);
    *pVenID = (LwU16) (result & 0xffff);   // extract lowest significant 2 bytes
    *pDevID = (LwU16) (result >> 16);        // extract the upper significant 2 bytes
}

//-----------------------------------------------------
// Intel_25XX_setupFunc
//
//   Finds the PCI Config Base Address for a 
//   Intel 25XX chipset
//-----------------------------------------------------
int Intel_25XX_setupFunc(LwU16 FHBdomain, LwU8 FHBbus, LwU8 FHBdev, LwU8 FHBfunc, LwU32 *pAddr)
{
    LwU32 BaseAddr;
    osPciRead32( FHBdomain, FHBbus, FHBdev, FHBfunc, &BaseAddr, INTEL_25XX_CONFIG_SPACE_BASE);

    // PCI-E enhanced config space is 256M aligned
    *pAddr = BaseAddr & 0xf0000000;

    return LW_OK;
}

//-----------------------------------------------------
// Intel_359E_setupFunc
//
//   Finds the PCI Config Base Address for a 
//   Intel 359E chipset
//-----------------------------------------------------
int Intel_359E_setupFunc(LwU16 FHBdomain, LwU8 FHBbus, LwU8 FHBdev, LwU8 FHBfunc, LwU32 *pAddr)
{
    LwU32 BaseAddr;
    osPciRead32( FHBdomain, FHBbus, FHBdev, FHBfunc, &BaseAddr, INTEL_359E_CONFIG_SPACE_BASE);

    // PCI-E enhanced config space is 256M aligned
    *pAddr = BaseAddr & 0xf0000000;

    return LW_OK;
}

//-----------------------------------------------------
// Intel_27XX_setupFunc
//
//   Finds the PCI Config Base Address for a 
//   Intel 27XX chipset
//-----------------------------------------------------
int Intel_27XX_setupFunc(LwU16 FHBdomain, LwU8 FHBbus, LwU8 FHBdev, LwU8 FHBfunc, LwU32 *pAddr)
{
    LwU32 BaseAddr;
    osPciRead32( FHBdomain, FHBbus, FHBdev, FHBfunc, &BaseAddr, INTEL_25XX_CONFIG_SPACE_BASE);

    // PCI-E enhanced config space is 256M aligned
    *pAddr = BaseAddr & 0xf0000000;

    return LW_OK;
}

//-----------------------------------------------------
// Lwidia_CK804_setupFunc
//
//   Finds the PCI Config Base Address for a 
//   Lwpu CK804 chipset
//-----------------------------------------------------
int Lwidia_CK804_setupFunc(LwU16 FHBdomain, LwU8 FHBbus, LwU8 FHBdev, LwU8 FHBfunc, LwU32 *pAddr)
{
    LwU32 BaseAddr;
    osPciRead32( FHBdomain, FHBbus, FHBdev, FHBfunc, &BaseAddr, LWIDIA_CK804_CONFIG_SPACE_BASE);
    BaseAddr = REF_VAL( LWIDIA_CK804_CONFIG_SPACE_BASE_ADDRESS, BaseAddr);

    if (BaseAddr >= 0x10)
    {
        dprintf("LWRM: *** CK804 or C19 Pcie enhanced configuration access only supports 32 bits address!\n");
        BaseAddr = 0;
        return LW_ERR_GENERIC;
    }

    BaseAddr <<= 28;

    *pAddr = BaseAddr;

    return LW_OK;
}

//-----------------------------------------------------
// Lwidia_C19_setupFunc
//
//   Finds the PCI Config Base Address for a 
//   Lwpu C19 chipset
//-----------------------------------------------------
int Lwidia_C19_setupFunc(LwU16 FHBdomain, LwU8 FHBbus, LwU8 FHBdev, LwU8 FHBfunc, LwU32 *pAddr)
{
    LwU32 BaseAddr;
    osPciRead32( FHBdomain, FHBbus, FHBdev, FHBfunc, &BaseAddr, LWIDIA_C19_CONFIG_SPACE_BASE);
    BaseAddr = REF_VAL( LWIDIA_C19_CONFIG_SPACE_BASE_ADDRESS, BaseAddr);
    
    *pAddr = BaseAddr;

    if(BaseAddr == 0)
        return LW_ERR_GENERIC;

    if (BaseAddr >= 0x10)
    {
        dprintf("LWRM: *** CK804 or C19 Pcie enhanced configuration access only supports 32 bits address!\n");
        *pAddr = 0;
        return LW_ERR_GENERIC;
    }

    BaseAddr <<= 28;

    *pAddr = BaseAddr;

    return LW_OK;
}

//-----------------------------------------------------
// Lwidia_C51_setupFunc
//
//   Finds the PCI Config Base Address for a 
//   Lwpu C51 chipset
//-----------------------------------------------------
int Lwidia_C51_setupFunc(LwU16 FHBdomain, LwU8 FHBbus, LwU8 FHBdev, LwU8 FHBfunc, LwU32 *pAddr)
{
    LwU32 BaseAddr;
    osPciRead32( FHBdomain, FHBbus, FHBdev, FHBfunc, &BaseAddr, LWIDIA_C51_CONFIG_SPACE_BASE);
    BaseAddr = REF_VAL( LWIDIA_C51_CONFIG_SPACE_BASE_ADDRESS, BaseAddr);
    
    BaseAddr <<= 28;
    *pAddr = BaseAddr;
    
    if (BaseAddr >= 0x10)
    {
        dprintf("LWRM: C51 enhanced config access only supports 32 bits address!\n");
        *pAddr = 0;
        return LW_ERR_GENERIC;
    }

    if(BaseAddr == 0)
        return LW_ERR_GENERIC;

    return LW_OK;
}

//-----------------------------------------------------
// SiS_656_setupFunc
//
//   Finds the PCI Config Base Address for a 
//   Sis 656 chipset
//-----------------------------------------------------
int SiS_656_setupFunc(LwU16 FHBdomain, LwU8 FHBbus, LwU8 FHBdev, LwU8 FHBfunc, LwU32 *pAddr)
{
    LwU32 BaseAddr;
    osPciRead32( FHBdomain, FHBbus, FHBdev, FHBfunc, &BaseAddr, SIS_656_CONFIG_SPACE_BASE);
    BaseAddr = REF_VAL( SIS_656_CONFIG_SPACE_BASE_ADDRESS, BaseAddr);

    if(BaseAddr)
    {
        BaseAddr <<= 28;
        *pAddr = BaseAddr;
    }

    return LW_OK;
}

//-----------------------------------------------------
// _findFHB
//
//    Finds the domain/bus/dev/func & Vendor and Device ID
//    for the first host bridge
//-----------------------------------------------------
static BOOL _findFHB(LwU16 *pFHBdomain, LwU8 *pFHBbus, LwU8 *pFHBdev, LwU8 *pFHBfunc, LwU16 *pFHBVenID, LwU16 *pFHBDevID)
{
    LwU16   VenID, DevID;
    LwU32   domain;
    LwU16   bus;
    LwU8    dev, func;
    LwU32   i;
    LwU16   ClassCode;

    for(domain = 0; domain < PCIE_MAX_DOMAIN ; domain++)
    {
        for(bus = 0; bus <  PCIE_MAX_BUS ; bus++)
        {
            for(dev = 0; dev <  PCIE_MAX_DEV ; dev++)
            {
                for(func=0; func <  PCIE_MAX_FUNC ; func++)
                {
                    _findDevVenID((LwU16) domain, (LwU8) bus, dev, func, &VenID, &DevID);
                    if (VenID == PCI_ILWALID_VENDORID)
                        break;           // skip to the next device
    
                    //
                    // For domain 0 bus 0 device 0 and function 0
                    // We will check all known chipsets
                    // Because for some systems, like CK804's North Bridge is memory controller, not host bridge
                    //
                    if (domain == 0 && bus == 0 && dev == 0 && func == 0)
                        for (i=0; chipsetInfoArray[i].vendorID;i++)
                            if (VenID == chipsetInfoArray[i].vendorID && DevID == chipsetInfoArray[i].deviceID)
                                goto foundit;
    
                    osPciRead16( (LwU16) domain, (LwU8) bus, dev, func, &ClassCode, 0xA);
                    if ((ClassCode >> 8) != PCI_CLASS_BRIDGE_DEV)
                        break;           // not a bridge device
    
                    if ((ClassCode & 0xff) != PCI_SUBCLASS_BR_HOST)
                        break;           // not a host bridge

foundit:
                    // Found it
                    *pFHBdomain = (LwU16)domain;
                    *pFHBbus    = (LwU8)bus;
                    *pFHBdev = dev;
                    *pFHBfunc   = func;
                    *pFHBVenID = VenID;
                    *pFHBDevID = DevID;
                    return TRUE;
                }
            }
        }
    }

    // Not found
    return FALSE;
}

//-----------------------------------------------------
// FindExtendedCapOffset
//
//    returns the offset to the capability in the PCI
//    Enhanced  Config Space
//-----------------------------------------------------
LW_STATUS FindExtendedCapOffset(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func, LwU16 capID, LwU16 *capOffset)
{
    LwU32 capHdr;
    LwU16 capIDRead, NextCapOffset;
    LwU16 returnOffset = 0x100;
    LW_STATUS status = LW_OK;

    capHdr = PcieRead32(domain, bus, dev, func, 0x100);

    capIDRead = (LwU16) (capHdr & 0xffff);
    NextCapOffset = (LwU16) (capHdr >> 20);

    while(capIDRead != capID && NextCapOffset != 0)
    {
        returnOffset = NextCapOffset;

        capHdr = PcieRead32(domain, bus, dev, func, NextCapOffset);
        capIDRead = (LwU16) (capHdr & 0xffff);
        NextCapOffset = (LwU16) (capHdr >> 20);
    }

    // This capability does not exist!!
    if(capIDRead != capID)
    {
        status = LW_ERR_GENERIC;
        *capOffset = PCIE_ILWALID_CAPABILITY_OFFSET;
    }
    else
    {
        *capOffset = returnOffset;
    }

    return status;
}

//-----------------------------------------------------
// FindStdCapOffset
//
//    returns the offset to the capability in the PCI
//    Standard  Config Space
//-----------------------------------------------------
LW_STATUS FindStdCapOffset(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func, LwU16 capID, LwU8 *capOffset)
{
    LwU32 capHdr;
    LwU8 capIDRead, NextCapOffset, returnOffset;
    LW_STATUS status = LW_OK;

    //
    // Read Cap Pointer First so that I can know the starting address of the Cap
    // Pointer stored at offset 34h
    //
    LwU8 capPtr = (LwU8) (PcieRead32(domain, bus, dev, func, 0x34) & 0xff);

    returnOffset = capPtr;

    capHdr = PcieRead32(domain, bus, dev, func, capPtr);

    capIDRead = (LwU8) (capHdr & 0xff);

    NextCapOffset = (LwU8) (capHdr >> 8);
    
    while(capIDRead != capID && NextCapOffset != 0)
    {
        returnOffset = NextCapOffset;
        capHdr = PcieRead32(domain, bus, dev, func, NextCapOffset);
        capIDRead = (LwU8) (capHdr & 0xff);
        NextCapOffset = (LwU8) (capHdr >> 8);
    }

    // This capability does not exist!!
    if(capIDRead != capID)
    {
        status = LW_ERR_GENERIC;
        *capOffset = PCIE_ILWALID_CAPABILITY_OFFSET;
    }
    else
    {
        *capOffset = returnOffset;
    }

    return status;
}

//-----------------------------------------------------
// PcieRead32
//
//   return a 32 bit data @ the offset of PCI Config space of domain/bus/dev/func
//   Will try to access offset above 0x100
//-----------------------------------------------------
LwU32 PcieRead32(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func, LwU32 offset)
{
    LwU32 BAR0;
    LwU32 ConfigBaseAddr = Get_PCI_Config_Base(FHBConfigBaseAddr, domain, bus, dev, func);
    LwU32 data32;

    if(osPciRead32( domain, bus, dev, func, &data32, offset) == LW_OK)
        return data32;
 
    if(ConfigBaseFound == TRUE)
        return (RD_PHYS32(ConfigBaseAddr + offset));
    
    osPciRead32( domain, bus, dev, func, &BAR0, PCIE_BAR0_OFFSET);
    
    if(BAR0 == lwBar0)
        return (RD_PHYS32(lwBar0 + 0x00088000 + offset));
    
    return 0;   // unable to get data, return 0 as default
}

//-----------------------------------------------------
// _displayAllCapInfo
//
//    A helper function to display all the capabilities
//    info nicely
//-----------------------------------------------------
void static _displayAllCapInfo(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func)
{
    dprintf("\n");

    dprintf("-START-----------------------------------------------------------------------------------------\n"); 
    dprintf("lw: Displaying DOMAIN  %04x  BUS  %02x, DEVICE  %02x, FUNCTION  %02x\n", domain, bus, dev, func); 

    dprintf("\n");
    _displayPCIStdHdr(domain, bus, dev, func);
    dprintf("\n");
    _displayPCIECap(domain, bus, dev, func);
    dprintf("\n");
    _displayVCCap(domain, bus, dev, func);
    dprintf("\n");
    _displayAERRCap(domain, bus, dev, func);

    dprintf("-END-------------------------------------------------------------------------------------------\n");
    dprintf("\n");
}

//-----------------------------------------------------
// _displayPCIStdHdr
//
//    Display the PCI standard header with important
//    registers
//-----------------------------------------------------
void static _displayPCIStdHdr(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func)
{
    LwU32 tempVar = PcieRead32(domain, bus, dev, func, LW_XVE_DEV_CTRL);
    PhysAddr Addr;
    LwU32 BAR0;

    dprintf("lw: PCI Standard Header\n\n");

    dprintf("lw: LW_XVE_DEV_CTRL_CMD/_STAT:                                0x%08x\n", tempVar);

    if(DRF_VAL(_XVE_DEV, _CTRL_CMD, _IO_SPACE, tempVar) == LW_XVE_DEV_CTRL_CMD_IO_SPACE_ENABLED)
        dprintf("lw:  LW_XVE_DEV_CTRL_CMD_IO_SPACE_ENABLED\n");
        
    if(DRF_VAL(_XVE_DEV, _CTRL_CMD, _MEMORY_SPACE, tempVar) == LW_XVE_DEV_CTRL_CMD_MEMORY_SPACE_ENABLED)
        dprintf("lw:  LW_XVE_DEV_CTRL_CMD_MEMORY_SPACE_ENABLED\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_CMD, _BUS_MASTER, tempVar) == LW_XVE_DEV_CTRL_CMD_BUS_MASTER_ENABLED)
        dprintf("lw:  LW_XVE_DEV_CTRL_CMD_BUS_MASTER_ENABLED\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_CMD, _SPECIAL_CYCLE , tempVar) != LW_XVE_DEV_CTRL_CMD_SPECIAL_CYCLE_DISABLED)
        dprintf("lw:  LW_XVE_DEV_CTRL_CMD_SPECIAL_CYCLE_ENABLED\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_CMD, _MEM_WRITE_AND_ILWALIDATE, tempVar) != LW_XVE_DEV_CTRL_CMD_MEM_WRITE_AND_ILWALIDATE_DISABLED)
        dprintf("lw:  LW_XVE_DEV_CTRL_CMD_MEM_WRITE_AND_ILWALIDATE_ENABLED\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_CMD, _VGA_PALETTE_SNOOP, tempVar) != LW_XVE_DEV_CTRL_CMD_VGA_PALETTE_SNOOP_DISABLED)
        dprintf("lw:  LW_XVE_DEV_CTRL_CMD_VGA_PALETTE_SNOOP_ENABLED\n");

    if(DRF_VAL(_XVE_DEV, _CTRL_CMD, _PERR, tempVar) == LW_XVE_DEV_CTRL_CMD_PERR_ENABLED)
        dprintf("lw:  LW_XVE_DEV_CTRL_CMD_PERR_ENABLED\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_CMD, _IDSEL_STEP, tempVar) != LW_XVE_DEV_CTRL_CMD_IDSEL_STEP_DISABLED)
        dprintf("lw:  LW_XVE_DEV_CTRL_CMD_IDSEL_STEP_ENABLED\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_CMD, _SERR, tempVar) != LW_XVE_DEV_CTRL_CMD_SERR_DISABLED)
        dprintf("lw:  LW_XVE_DEV_CTRL_CMD_SERR_ENABLED\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_CMD, _FAST_BACK2BACK , tempVar) != LW_XVE_DEV_CTRL_CMD_FAST_BACK2BACK_DISABLED)
        dprintf("lw:  LW_XVE_DEV_CTRL_CMD_FAST_BACK2BACK_ENABLED\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_CMD, _INTERRUPT_DISABLE, tempVar) != LW_XVE_DEV_CTRL_CMD_INTERRUPT_DISABLE_INIT)
        dprintf("lw:  LW_XVE_DEV_CTRL_CMD_INTERRUPT_ENABLE_INIT\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_STAT, _INTERRUPT, tempVar) == LW_XVE_DEV_CTRL_STAT_INTERRUPT_PENDING)
        dprintf("lw:  LW_XVE_DEV_CTRL_STAT_INTERRUPT_PENDING\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_STAT, _CAPLIST , tempVar) == LW_XVE_DEV_CTRL_STAT_CAPLIST_PRESENT)
        dprintf("lw:  LW_XVE_DEV_CTRL_STAT_CAPLIST_PRESENT\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_STAT, _66MHZ, tempVar) == LW_XVE_DEV_CTRL_STAT_66MHZ_CAPABLE)
        dprintf("lw:  LW_XVE_DEV_CTRL_STAT_66MHZ_CAPABLE\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_STAT, _FAST_BACK2BACK, tempVar) == LW_XVE_DEV_CTRL_STAT_FAST_BACK2BACK_CAPABLE)
        dprintf("lw:  LW_XVE_DEV_CTRL_STAT_FAST_BACK2BACK_CAPABLE\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_STAT, _MASTER_DATA_PERR, tempVar) == LW_XVE_DEV_CTRL_STAT_MASTER_DATA_PERR_ACTIVE)
        dprintf("lw:  LW_XVE_DEV_CTRL_STAT_MASTER_DATA_PERR_ACTIVE\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_STAT, _DEVSEL_TIMING, tempVar) == LW_XVE_DEV_CTRL_STAT_DEVSEL_TIMING_FAST)
        dprintf("lw:  LW_XVE_DEV_CTRL_STAT_DEVSEL_TIMING_FAST\n");
    else if(DRF_VAL(_XVE_DEV, _CTRL_STAT, _DEVSEL_TIMING, tempVar) == LW_XVE_DEV_CTRL_STAT_DEVSEL_TIMING_MEDIUM)
        dprintf("lw:  LW_XVE_DEV_CTRL_STAT_DEVSEL_TIMING_MEDIUM\n");
    else
        dprintf("lw:  LW_XVE_DEV_CTRL_STAT_DEVSEL_TIMING_SLOW\n");

    if(DRF_VAL(_XVE_DEV, _CTRL_STAT, _SIGNALED_TARGET_ABORT, tempVar) == LW_XVE_DEV_CTRL_STAT_SIGNALED_TARGET_ABORT_YES)
        dprintf("lw:  LW_XVE_DEV_CTRL_STAT_SIGNALED_TARGET_ABORT_YES\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_STAT, _RECEIVED_TARGET_ABORT, tempVar) == LW_XVE_DEV_CTRL_STAT_RECEIVED_TARGET_ABORT_YES)
        dprintf("lw:  LW_XVE_DEV_CTRL_STAT_RECEIVED_TARGET_ABORT_YES\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_STAT, _RECEIVED_MASTER_ABORT, tempVar) == LW_XVE_DEV_CTRL_STAT_RECEIVED_MASTER_ABORT_YES)
        dprintf("lw:  LW_XVE_DEV_CTRL_STAT_RECEIVED_MASTER_ABORT_YES\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_STAT,_SIGNALED_SERR, tempVar) == LW_XVE_DEV_CTRL_STAT_SIGNALED_SERR_ACTIVE)
        dprintf("lw:  LW_XVE_DEV_CTRL_STAT_SIGNALED_SERR_ACTIVE\n");
    
    if(DRF_VAL(_XVE_DEV, _CTRL_STAT,_DETECTED_PERR, tempVar) == LW_XVE_DEV_CTRL_STAT_DETECTED_PERR_ACTIVE)
        dprintf("lw:  LW_XVE_DEV_CTRL_STAT_DETECTED_PERR_ACTIVE\n");
    
    dprintf("\n");

    osPciRead32( domain, bus, dev, func, &BAR0, PCIE_BAR0_OFFSET);

    if(ConfigBaseFound == TRUE)
    {
        Addr = Get_PCI_Config_Base(FHBConfigBaseAddr, domain, bus, dev, func);
        printData(Addr, 0x40);
    }
    else if(BAR0 == lwBar0)
    {
        Addr = lwBar0 + 0x00088000;
        printData(Addr, 0x40);
    }
}

//-----------------------------------------------------
// _DevType
//
//    Return the Device/port type of the device
//    (e.g root port)
//-----------------------------------------------------
static LwU8 _DevType(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func)
{
    LwU32 tempVar;
    LwU8 offset;
    LW_STATUS status = FindStdCapOffset(domain, bus, dev, func, PCIE_CAP_ID, &offset);

    if(offset ==  PCIE_ILWALID_CAPABILITY_OFFSET || status == LW_ERR_GENERIC)
        return PCIE_ENDPT;

    tempVar = PcieRead32(domain, bus, dev, func, offset);
    return (LwU8) (DRF_VAL(_XVR, _PCI_EXPRESS_CAPABILITY, _DEVICE_PORT_TYPE, tempVar));
}

//-----------------------------------------------------
// _displayAERRCap
//
//    Displays the PCI-Express Capability register values
//    Only display the reg with bits set
//-----------------------------------------------------
void static _displayPCIECap(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func)
{
    LwU32 tempVar;
    LwU8 offset;
    LW_STATUS status = FindStdCapOffset(domain, bus, dev, func, PCIE_CAP_ID, &offset);
    LwU32 BAR0;
    PhysAddr Addr;
    LwU8 PwrLimValue, PwrLimScale;

    if(offset == PCIE_ILWALID_CAPABILITY_OFFSET || status == LW_ERR_GENERIC)
        return;

    tempVar = PcieRead32(domain, bus, dev, func, offset+4);
    
    dprintf("lw: PCI Express Capability\n\n");

    dprintf("lw: LW_XVE_DEVICE_CAPABILITY:                                 0x%08x\n", tempVar);

    switch(DRF_VAL(_XVE, _DEVICE_CAPABILITY, _MAX_PAYLOAD_SIZE, tempVar))
    {
        case 0 : dprintf("lw:  MAX_PAYLOAD_SIZE                                         128 byte \n");
                break;
        case 1 : dprintf("lw:  MAX_PAYLOAD_SIZE                                         256 byte \n");
                break;
        case 2 : dprintf("lw:  MAX_PAYLOAD_SIZE                                         512 byte \n");
                break;
        case 3 : dprintf("lw:  MAX_PAYLOAD_SIZE                                         1KB \n");
                break;
        case 4 : dprintf("lw:  MAX_PAYLOAD_SIZE                                         2KB \n");
                break;
        case 5 : dprintf("lw:  MAX_PAYLOAD_SIZE                                         4KB \n");
                break;
        case 6 : dprintf("lw:  MAX_PAYLOAD_SIZE                                         Reserved Encoding Detected \n");
                break;
        case 7 : dprintf("lw:  MAX_PAYLOAD_SIZE                                         Reserved Encoding Detected \n");
                break;
        default: break;
    }

    PwrLimValue = (LwU8) DRF_VAL(_XVE, _DEVICE_CAPABILITY, _CAPTURED_SLOT_POWER_LIMIT_VALUE, tempVar);
    dprintf("lw:  CAPTURED_SLOT_POWER_LIMIT_VALUE is                       %i\n", PwrLimValue);

    PwrLimScale = (LwU8) DRF_VAL(_XVE, _DEVICE_CAPABILITY, _CAPTURED_SLOT_POWER_LIMIT_SCALE, tempVar);
    dprintf("lw:  CAPTURED_SLOT_POWER_LIMIT_SCALE is                       %i\n", PwrLimScale);

    switch (PwrLimScale)
    {
        case 0: dprintf("lw:  MAX Power limit                                          %d W\n", PwrLimValue);
                break;
        case 1: dprintf("lw:  MAX Power limit                                          %d W\n", PwrLimValue / 10);
                break;
        case 2: dprintf("lw:  MAX Power limit                                          %d W\n", PwrLimValue / 100);
                break;
        case 3: dprintf("lw:  MAX Power limit                                          %d W\n", PwrLimValue / 1000);
                break;
        default: break;
    }

    dprintf("\n");

    tempVar = PcieRead32(domain, bus, dev, func, offset+8);

    dprintf("lw: LW_XVE_DEVICE_CONTROL_STATUS:                             0x%08x\n", tempVar);

    if(DRF_VAL(_XVE, _DEVICE_CONTROL_STATUS, _CORR_ERROR_REPORTING_ENABLE, tempVar) != LW_XVE_DEVICE_CONTROL_STATUS_CORR_ERROR_REPORTING_ENABLE_INIT )
        dprintf("lw:  CORR_ERROR_REPORTING_ENABLE_INIT bit set \n");

    if(DRF_VAL(_XVE, _DEVICE_CONTROL_STATUS, _NON_FATAL_ERROR_REPORTING_ENABLE, tempVar) != LW_XVE_DEVICE_CONTROL_STATUS_NON_FATAL_ERROR_REPORTING_ENABLE_INIT )
        dprintf("lw:  NON_FATAL_ERROR_REPORTING_ENABLE bit set \n");

    if(DRF_VAL(_XVE, _DEVICE_CONTROL_STATUS, _FATAL_ERROR_REPORTING_ENABLE, tempVar) != LW_XVE_DEVICE_CONTROL_STATUS_FATAL_ERROR_REPORTING_ENABLE_INIT )
        dprintf("lw:  FATAL_ERROR_REPORTING_ENABLE bit set \n");

    if(DRF_VAL(_XVE, _DEVICE_CONTROL_STATUS, _UNSUPP_REQ_REPORTING_ENABLE, tempVar) != LW_XVE_DEVICE_CONTROL_STATUS_UNSUPP_REQ_REPORTING_ENABLE_INIT )
        dprintf("lw:  UNSUPP_REQ_REPORTING_ENABLE bit set \n");

    if(DRF_VAL(_XVE, _DEVICE_CONTROL_STATUS, _ENABLE_RELAXED_ORDERING, tempVar) != LW_XVE_DEVICE_CONTROL_STATUS_ENABLE_RELAXED_ORDERING_INIT )
        dprintf("lw:  ENABLE_RELAXED_ORDERING bit set \n");

    //***

    switch(DRF_VAL(_XVE, _DEVICE_CONTROL_STATUS, _MAX_PAYLOAD_SIZE, tempVar))
    {
        case 0 : dprintf("lw:  MAX_PAYLOAD_SIZE                                         128 byte \n");
                break;
        case 1 : dprintf("lw:  MAX_PAYLOAD_SIZE                                         256 byte \n");
                break;
        case 2 : dprintf("lw:  MAX_PAYLOAD_SIZE                                         512 byte \n");
                break;
        case 3 : dprintf("lw:  MAX_PAYLOAD_SIZE                                         1024 byte \n");
                break;
        case 4 : dprintf("lw:  MAX_PAYLOAD_SIZE                                         2048 byte \n");
                break;
        case 5 : dprintf("lw:  MAX_PAYLOAD_SIZE                                         4096 byte \n");
                break;
        case 6 : dprintf("lw:  MAX_PAYLOAD_SIZE                                         Reserved Encoding Detected \n");
                break;
        case 7 : dprintf("lw:  MAX_PAYLOAD_SIZE                                         Reserved Encoding Detected \n");
                break;
        default: break;
    }

    //***

    if(DRF_VAL(_XVE, _DEVICE_CONTROL_STATUS, _EXTENDED_TAG_FIELD_ENABLE, tempVar) != LW_XVE_DEVICE_CONTROL_STATUS_EXTENDED_TAG_FIELD_ENABLE_INIT )
        dprintf("lw:  EXTENDED_TAG_FIELD_ENABLE bit set\n");

    if(DRF_VAL(_XVE, _DEVICE_CONTROL_STATUS, _PHANTOM_FUNCTIONS_ENABLE, tempVar) != LW_XVE_DEVICE_CONTROL_STATUS_PHANTOM_FUNCTIONS_ENABLE_INIT )
        dprintf("lw:  PHANTOM_FUNCTIONS_ENABLE bit set \n");

    if(DRF_VAL(_XVE, _DEVICE_CONTROL_STATUS, _AUXILLARY_POWER_PM_ENABLE, tempVar) != LW_XVE_DEVICE_CONTROL_STATUS_AUXILLARY_POWER_PM_ENABLE_INIT  )
        dprintf("lw:  AUXILLARY_POWER_PM_ENABLE bit set \n");

    if(DRF_VAL(_XVE, _DEVICE_CONTROL_STATUS, _ENABLE_NO_SNOOP, tempVar) == LW_XVE_DEVICE_CONTROL_STATUS_ENABLE_NO_SNOOP_INIT )
        dprintf("lw:  ENABLE_NO_SNOOP bit set \n");

    //***

    switch(DRF_VAL(_XVE, _DEVICE_CONTROL_STATUS, _MAX_READ_REQUEST_SIZE, tempVar))
    {
        case 0 : dprintf("lw:  MAX_READ_REQUEST_SIZE                                    128 byte \n");
                break;
        case 1 : dprintf("lw:  MAX_READ_REQUEST_SIZE                                    256 byte \n");
                break;
        case 2 : dprintf("lw:  MAX_READ_REQUEST_SIZE                                    512 byte \n");
                break;
        case 3 : dprintf("lw:  MAX_READ_REQUEST_SIZE                                    1KB \n");
                break;
        case 4 : dprintf("lw:  MAX_READ_REQUEST_SIZE                                    2KB \n");
                break;
        case 5 : dprintf("lw:  MAX_READ_REQUEST_SIZE                                    4KB \n");
                break;
        case 6 : dprintf("lw:  MAX_READ_REQUEST_SIZE                                    Reserved Encoding Detected \n");
                break;
        case 7 : dprintf("lw:  MAX_READ_REQUEST_SIZE                                    Reserved Encoding Detected \n");
                break;
        default: break;
    }

    //***
    
    if(DRF_VAL(_XVE, _DEVICE_CONTROL_STATUS, _CORR_ERROR_DETECTED, tempVar) != LW_XVE_DEVICE_CONTROL_STATUS_CORR_ERROR_DETECTED_INIT )
        dprintf("lw:  CORR_ERROR_DETECTEDE bit set\n");

    if(DRF_VAL(_XVE, _DEVICE_CONTROL_STATUS, _NON_FATAL_ERROR_DETECTED, tempVar) != LW_XVE_DEVICE_CONTROL_STATUS_NON_FATAL_ERROR_DETECTED_INIT )
        dprintf("lw:  NON_FATAL_ERROR_DETECTED bit set \n");

    if(DRF_VAL(_XVE, _DEVICE_CONTROL_STATUS, _UNSUPP_REQUEST_DETECTED, tempVar) != LW_XVE_DEVICE_CONTROL_STATUS_UNSUPP_REQUEST_DETECTED_INIT  )
        dprintf("lw:  UNSUPP_REQUEST_DETECTED bit set \n");

    if(DRF_VAL(_XVE, _DEVICE_CONTROL_STATUS, _AUX_POWER_DETECTED, tempVar) != LW_XVE_DEVICE_CONTROL_STATUS_AUX_POWER_DETECTED_INIT )
        dprintf("lw:  AUX_POWER_DETECTED bit set\n");

    if(DRF_VAL(_XVE, _DEVICE_CONTROL_STATUS, _TRANSACTIONS_PENDING, tempVar) != LW_XVE_DEVICE_CONTROL_STATUS_TRANSACTIONS_PENDING_INIT )
        dprintf("lw:  TRANSACTIONS_PENDING bit set \n");

    dprintf("\n");

    tempVar = PcieRead32(domain, bus, dev, func, offset+12);

    dprintf("lw: LW_XVE_LINK_CAPABILITIES:                                 0x%08x\n", tempVar);

    if(DRF_VAL(_XVE, _LINK_CAPABILITIES, _MAX_LINK_SPEED, tempVar) == 1 )
        dprintf("lw:  MAX_LINK_SPEED                                           2.5Gb/s \n");
    else
        dprintf("lw:  MAX_LINK_SPEED                                           Reserved Encoding Detected \n");

    switch(DRF_VAL(_XVE, _LINK_CAPABILITIES, _MAX_LINK_WIDTH, tempVar))
    {
        case 1 :
        case 2 :
        case 4 :
        case 8 :
        case 12 :
        case 16 :
        case 32 :   dprintf("lw:  MAX_LINK_WIDTH                                           x%i \n", DRF_VAL(_XVE, _LINK_CAPABILITIES, _MAX_LINK_WIDTH, tempVar));
                    break;
        default :   dprintf("lw:  MAX_LINK_WIDTH                                           Reserved Encoding Detected \n");
                    break;
    }

    switch (DRF_VAL(_XVE, _LINK_CAPABILITIES, _ACTIVE_STATE_LINK_PM_SUPPORT, tempVar))
    {
        case 0: dprintf("lw:  ACTIVE_STATE_LINK_PM_SUPPORT                             Reserved Encoding Detected \n");
                break;
        case 1: dprintf("lw:  ACTIVE_STATE_LINK_PM_SUPPORT                             L0s Entry Supported \n");
                break;
        case 2: dprintf("lw:  ACTIVE_STATE_LINK_PM_SUPPORT                             Encodings reserved ");
                break;
        case 3: dprintf("lw:  ACTIVE_STATE_LINK_PM_SUPPORT                             L0s and L1 Supported \n");
                break;
        default: break;
    }

    dprintf("\n");

    tempVar = PcieRead32(domain, bus, dev, func, offset+16);

    dprintf("lw: LW_XVE_LINK_CONTROL_STATUS:                               0x%08x\n", tempVar);

    switch (DRF_VAL(_XVE, _LINK_CONTROL_STATUS, _ACTIVE_STATE_LINK_PM_CONTROL, tempVar))
    {
        case 0: dprintf("lw:  ACTIVE_STATE_LINK_PM_SUPPORT                             Disabled \n");
                break;
        case 1: dprintf("lw:  ACTIVE_STATE_LINK_PM_SUPPORT                             L0s Entry Enabled \n");
                break;
        case 2: dprintf("lw:  ACTIVE_STATE_LINK_PM_SUPPORT                             L1  Entry Enabled ");
                break;
        case 3: dprintf("lw:  ACTIVE_STATE_LINK_PM_SUPPORT                             L0s and L1 Enabled \n");
                break;
        default: break;
    }

    // check switch port also
    if(DRF_VAL(_XVE, _LINK_CONTROL_STATUS, _READ_COMPLETION_BOUNDARY, tempVar) == 0)
        dprintf("lw:  READ_COMPLETION_BOUNDARY                                 64 byte \n");
    else
        dprintf("lw:  READ_COMPLETION_BOUNDARY                                 128 byte \n");

    if(DRF_VAL(_XVE, _LINK_CONTROL_STATUS, _LINK_DISABLE, tempVar) != LW_XVE_LINK_CONTROL_STATUS_LINK_DISABLE_INIT)
        dprintf("lw:  LINK_DISABLE bit set \n");

    if(DRF_VAL(_XVE, _LINK_CONTROL_STATUS, _RETRAIN_LINK, tempVar) != LW_XVE_LINK_CONTROL_STATUS_RETRAIN_LINK_INIT)
        dprintf("lw:  RETRAIN_LINK bit set \n");

    if(DRF_VAL(_XVE, _LINK_CONTROL_STATUS, _COMMON_CLOCK_CONFIGURATION , tempVar) != LW_XVE_LINK_CONTROL_STATUS_COMMON_CLOCK_CONFIGURATION_INIT)
        dprintf("lw:  COMMON_CLOCK_CONFIGURATION Bit Set:                      common reference clock \n");
    else
        dprintf("lw:  COMMON_CLOCK_CONFIGURATION Bit Not Set:                  separate reference clock \n");

    if(DRF_VAL(_XVE, _LINK_CONTROL_STATUS, _EXTENDED_SYNCH, tempVar) != LW_XVE_LINK_CONTROL_STATUS_EXTENDED_SYNCH_INIT)
        dprintf("lw:  EXTENDED_SYNCH bit set \n");

    if(DRF_VAL(_XVE, _LINK_CONTROL_STATUS, _LINK_SPEED, tempVar) == 1 )
        dprintf("lw:  LINK_SPEED                                               2.5Gb/s \n");
    else
        dprintf("lw:  LINK_SPEED                                               Reserved Encoding Detected \n");

    switch(DRF_VAL(_XVE, _LINK_CONTROL_STATUS, _NEGOTIATED_LINK_WIDTH, tempVar))
    {
        case 1 :
        case 2 :
        case 4 :
        case 8 :
        case 12 :
        case 16 :
        case 32 :   dprintf("lw:  NEGOTIATED_LINK_WIDTH                                    x%i \n", DRF_VAL(_XVE, _LINK_CONTROL_STATUS, _NEGOTIATED_LINK_WIDTH, tempVar));
                    break;
        default :   dprintf("lw:  NEGOTIATED_LINK_WIDTH                                    Reserved Encoding Detected \n");
                    break;
    }

    if(DRF_VAL(_XVE, _LINK_CONTROL_STATUS, _TRAINING_ERROR, tempVar) != LW_XVE_LINK_CONTROL_STATUS_TRAINING_ERROR_INIT)
        dprintf("lw:  TRAINING_ERROR bit set \n");

    if(DRF_VAL(_XVE, _LINK_CONTROL_STATUS, _LINK_TRAINING, tempVar) != LW_XVE_LINK_CONTROL_STATUS_LINK_TRAINING_INIT)
        dprintf("lw:  LINK_TRAINING bit set \n");

    if(DRF_VAL(_XVE, _LINK_CONTROL_STATUS, _SLOT_CLOCK_CONFIGURATON, tempVar) == LW_XVE_LINK_CONTROL_STATUS_SLOT_CLOCK_CONFIGURATON_INIT)
        dprintf("lw:  SLOT_CLOCK_CONFIGURATON bit set \n");

    dprintf("\n");

    tempVar = PcieRead32(domain, bus, dev, func, offset+20);

    dprintf("lw: LW_XVE_SLOT_CAPABILITIES:                                 0x%08x\n", tempVar);

    dprintf("lw:  SLOT_POWER_LIMIT_VALUE                                   %i\n", DRF_VAL(_XVE, _SLOT_CAPABILITIES, _SLOT_POWER_LIMIT_VALUE, tempVar));

    dprintf("lw:  SLOT_POWER_LIMIT_SCALE                                   %i\n", DRF_VAL(_XVE, _SLOT_CAPABILITIES, _SLOT_POWER_LIMIT_SCALE, tempVar));

    switch (DRF_VAL(_XVE, _SLOT_CAPABILITIES, _SLOT_POWER_LIMIT_SCALE, tempVar))
    {
        case 0: dprintf("lw:  MAX Power limit                                          %d W\n", DRF_VAL(_XVE, _SLOT_CAPABILITIES, _SLOT_POWER_LIMIT_VALUE, tempVar));
                break;
        case 1: dprintf("lw:  MAX Power limit                                          %d W\n", DRF_VAL(_XVE, _SLOT_CAPABILITIES, _SLOT_POWER_LIMIT_VALUE, tempVar) / 10);
                break;
        case 2: dprintf("lw:  MAX Power limit                                          %d W\n", DRF_VAL(_XVE, _SLOT_CAPABILITIES, _SLOT_POWER_LIMIT_VALUE, tempVar) / 100);
                break;
        case 3: dprintf("lw:  MAX Power limit                                          %d W\n", DRF_VAL(_XVE, _SLOT_CAPABILITIES, _SLOT_POWER_LIMIT_VALUE, tempVar) / 1000);
                break;
        default: break;
    }

    dprintf("\n");

    tempVar = PcieRead32(domain, bus, dev, func, offset+28);

    dprintf("lw: LW_XVE_ROOT_CONTROL:                                      0x%08x\n", tempVar);

    if(DRF_VAL(_XVE, _ROOT_CONTROL, _SERR_CORRECTABLE_ERROR_ENABLE, tempVar) != LW_XVE_ROOT_CONTROL_SERR_CORRECTABLE_ERROR_ENABLE_INIT)
        dprintf("lw:  SERR_CORRECTABLE_ERROR_ENABLE bit set \n");

    if(DRF_VAL(_XVE, _ROOT_CONTROL, _SERR_NON_FATAL_ERROR_ENABLE, tempVar) != LW_XVE_ROOT_CONTROL_SERR_NON_FATAL_ERROR_ENABLE_INIT)
        dprintf("lw:  SERR_NON_FATAL_ERROR_ENABLE bit set \n");

    if(DRF_VAL(_XVE, _ROOT_CONTROL, _SERR_FATAL_ERROR_ENABLE, tempVar) != LW_XVE_ROOT_CONTROL_SERR_FATAL_ERROR_ENABLE_INIT)
        dprintf("lw:  SERR_FATAL_ERROR_ENABLE bit set \n");

    if(DRF_VAL(_XVE, _ROOT_CONTROL, _PME_INTERRUPT_ENABLE, tempVar) != LW_XVE_ROOT_CONTROL_PME_INTERRUPT_ENABLE_INIT)
        dprintf("lw:  PME_INTERRUPT_ENABLE bit set \n");

    dprintf("\n");

    osPciRead32( domain, bus, dev, func, &BAR0, PCIE_BAR0_OFFSET);

    if(ConfigBaseFound == TRUE)
    {
        Addr = (Get_PCI_Config_Base(FHBConfigBaseAddr, domain, bus, dev, func) + offset);
        printData(Addr, 0x40);
    }
    else if(BAR0 == lwBar0)
    {
        Addr = lwBar0 + 0x00088000 + offset;
        printData(Addr, 0x40);
    }
}

//-----------------------------------------------------
// _displayAERRCap
//
//    Displays the Virtual Channel Capability register values
//    Only display the reg with bits set
//-----------------------------------------------------
void static _displayVCCap(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func)
{
    LwU32 tempVar;
    PhysAddr Addr;
    LwU32 BAR0;
    LwU16 offset;
    LW_STATUS status = FindExtendedCapOffset(domain, bus, dev, func, PCIE_VC_CAP_ID, &offset);

    if(offset == PCIE_ILWALID_CAPABILITY_OFFSET || status == LW_ERR_GENERIC)
        return;

    tempVar = PcieRead32(domain, bus, dev, func, offset+4);

    dprintf("lw: Virtual Channel Capability\n\n");

    dprintf("lw: LW_XVE_VCCAP_PVC1:                                        0x%08x\n", tempVar);

    dprintf("lw:  VCCAP_PVC1_EVC:                                          %i \n", DRF_VAL(_XVE, _VCCAP_PVC1, _EVC, tempVar));

    dprintf("lw:  VCCAP_PVC1_LPVC:                                         %i \n", DRF_VAL(_XVE, _VCCAP_PVC1, _LPVC, tempVar));

    if(DRF_VAL(_XVE, _VCCAP_PVC1, _EVC, tempVar) == 0)
        dprintf("lw:  VCCAP_PVC1_REF:                                          00b -- 100ns reference clock \n");
    else
        dprintf("lw:  VCCAP_PVC1_REF:                                          reserved encoding \n");

    switch (DRF_VAL(_XVE, _VCCAP_PVC1, _PATS, tempVar))
    {
        case 0: dprintf("lw:  VCCAP_PVC1_PATS                                          1 bit \n");
                break;
        case 1: dprintf("lw:  VCCAP_PVC1_PATS                                          2 bits \n");
                break;
        case 2: dprintf("lw:  VCCAP_PVC1_PATS                                          4 bits \n");
                break;
        case 3: dprintf("lw:  VCCAP_PVC1_PATS                                          8 bits \n");
                break;
        default: break;
    }

    dprintf("\n");

    tempVar = PcieRead32(domain, bus, dev, func, offset+8);

    dprintf("lw: LW_XVR_VCCAP_PVC2:                                        0x%08x\n", tempVar);

    if(DRF_VAL(_XVR, _VCCAP_PVC2, _ARB, tempVar) & BIT(0))
        dprintf("lw:  VCCAP_PVC2_ARB                                           Bit 0 Set: Hardware fixed arbitration scheme\n");
    if(DRF_VAL(_XVR, _VCCAP_PVC2, _ARB, tempVar) & BIT(1))
        dprintf("lw:  VCCAP_PVC2_ARB                                           Bit 1 Set: WRR arbitration with 32 phases\n");
    if(DRF_VAL(_XVR, _VCCAP_PVC2, _ARB, tempVar) & BIT(2))
        dprintf("lw:  VCCAP_PVC2_ARB                                           Bit 2 Set: WRR arbitration with 64 phases\n");
    if(DRF_VAL(_XVR, _VCCAP_PVC2, _ARB, tempVar) & BIT(3))
        dprintf("lw:  VCCAP_PVC2_ARB                                           Bit 3 Set: WRR arbitration with 128 phases\n");
    if(DRF_VAL(_XVR, _VCCAP_PVC2, _ARB, tempVar) & (BIT(4)|BIT(5)|BIT(6)|BIT(7)))
        dprintf("lw:  VCCAP_PVC2_ARB                                           N/A: Reserved Encodings Detected  \n");

    dprintf("lw:  LW_XVR_VCCAP_PVC2_OFF:                                   0x%08x \n",
            DRF_VAL(_XVR, _VCCAP_PVC2, _OFF, tempVar));

    dprintf("\n");

    tempVar = PcieRead32(domain, bus, dev, func, offset+0x0C);

    dprintf("lw: LW_XVR_VCCAP_PCSR:                                        0x%08x\n", tempVar);

    if(DRF_VAL(_XVR, _VCCAP_PCSR, _LOAD, tempVar) & BIT(0))
        dprintf("lw:  VCCAP_PCSR_LOAD Bit Set\n");

    dprintf("lw:  VCCAP_PCSR_SEL:                                          %i\n", DRF_VAL(_XVR, _VCCAP_PCSR, _SEL, tempVar));

    if(DRF_VAL(_XVR, _VCCAP_PCSR, _STAT, tempVar) & BIT(0))
        dprintf("lw:  VCCAP_PCSR_STAT Bit Set\n");

    dprintf("\n");

    tempVar = PcieRead32(domain, bus, dev, func, offset+0x10);

    dprintf("lw: LW_XVR_VCCAP_VCR0:                                        0x%08x\n", tempVar);

    if(DRF_VAL(_XVR, _VCCAP_VCR0, _REJECT_SNOOP, tempVar) & BIT(0))
        dprintf("lw:  VCCAP_VCR0_REJECT_SNOOP Bit Set\n");

    dprintf("\n");

    tempVar = PcieRead32(domain, bus, dev, func, offset+0x14);

    dprintf("lw: LW_XVR_VCCAP_CTRL0:                                       0x%08x\n", tempVar);

    if(DRF_VAL(_XVR, _VCCAP_CTRL0, _TC0, tempVar) & BIT(0))
        dprintf("lw:  VCCAP_CTRL0_TC0 Bit Set\n");

    dprintf("lw:  VCCAP_CTRL0_MAP:                                         0x%08x\n",
            DRF_VAL(_XVR, _VCCAP_CTRL0, _MAP, tempVar));

    if(DRF_VAL(_XVR, _VCCAP_CTRL0, _LOAD, tempVar) & BIT(0))
        dprintf("lw:  VCCAP_CTRL0_LOAD Bit Set\n");

    dprintf("lw:  VCCAP_CTRL0_SEL:                                         %i\n", DRF_VAL(_XVR, _VCCAP_CTRL0, _SEL, tempVar));

    dprintf("lw:  VCCAP_CTRL0_VCID:                                        0x%03x\n", DRF_VAL(_XVR, _VCCAP_CTRL0, _VCID, tempVar));

    if(DRF_VAL(_XVR, _VCCAP_CTRL0, _VCEN, tempVar) == LW_XVR_VCCAP_CTRL0_VCEN_1)
        dprintf("lw:  VCCAP_CTRL0_VCEN_1\n");

    dprintf("\n");

    tempVar = PcieRead32(domain, bus, dev, func, offset+0x18);

    dprintf("lw: LW_XVR_VCCAP_STAT0:                                       0x%08x\n", tempVar);

    if(DRF_VAL(_XVR, _VCCAP_STAT0, _PATS, tempVar) & BIT(0))
        dprintf("lw:  VCCAP_STAT0_PATS Bit Set\n");

    if(DRF_VAL(_XVR, _VCCAP_STAT0, _VNP, tempVar) & BIT(0))
        dprintf("lw:  VCCAP_STAT0_VNP Bit Set\n");

    dprintf("\n");

    tempVar = PcieRead32(domain, bus, dev, func, offset+0x1C);

    dprintf("lw: LW_XVR_VCCAP_VCR1:                                        0x%08x\n", tempVar);

    if(DRF_VAL(_XVR, _VCCAP_VCR1, _REJECT_SNOOP, tempVar) & BIT(0))
        dprintf("lw:  VCCAP_VCR1_REJECT_SNOOP Bit Set\n");

    dprintf("\n");

    tempVar = PcieRead32(domain, bus, dev, func, offset+0x20);

    dprintf("lw: LW_XVR_VCCAP_CTRL1:                                       0x%08x\n", tempVar);

    if(DRF_VAL(_XVR, _VCCAP_CTRL1, _TC0, tempVar) & BIT(0))
        dprintf("lw:  VCCAP_CTRL1_TC0 Bit Set\n");

    dprintf("lw:  VCCAP_CTRL1_MAP:                                         0x%08x\n",
            DRF_VAL(_XVR, _VCCAP_CTRL1, _MAP, tempVar));

    if(DRF_VAL(_XVR, _VCCAP_CTRL1, _LOAD, tempVar) & BIT(0))
        dprintf("lw:  VCCAP_CTRL1_LOAD Bit Set\n");

    dprintf("lw:  VCCAP_CTRL1_SEL:                                         %i\n", DRF_VAL(_XVR, _VCCAP_CTRL1, _SEL, tempVar));

    dprintf("lw:  VCCAP_CTRL1_VCID:                                        0x%03x\n", DRF_VAL(_XVR, _VCCAP_CTRL1, _VCID, tempVar));

    if(DRF_VAL(_XVR, _VCCAP_CTRL1, _VCEN, tempVar) == LW_XVR_VCCAP_CTRL1_VCEN_1)
        dprintf("lw:  VCCAP_CTRL1_VCEN_1\n");

    dprintf("\n");

    tempVar = PcieRead32(domain, bus, dev, func, offset+0x24);

    dprintf("lw: LW_XVR_VCCAP_STAT1:                                       0x%08x\n", tempVar);

    if(DRF_VAL(_XVR, _VCCAP_STAT1, _PATS, tempVar) & BIT(0))
        dprintf("lw:  VCCAP_STAT1_PATS Bit Set\n");

    if(DRF_VAL(_XVR, _VCCAP_STAT1, _VNP, tempVar) & BIT(0))
        dprintf("lw:  VCCAP_STAT1_VNP Bit Set\n");

    dprintf("\n");

    osPciRead32( domain, bus, dev, func, &BAR0, PCIE_BAR0_OFFSET);

    if(ConfigBaseFound == TRUE)
    {
        Addr = (Get_PCI_Config_Base(FHBConfigBaseAddr, domain, bus, dev, func) + offset);
        printData(Addr, 0x40);
    }
    else if(BAR0 == lwBar0)
    {
        Addr = lwBar0 + 0x00088000 + offset;
        printData(Addr, 0x40);
    }
}

//-----------------------------------------------------
// _displayAERRCap
//
//    Displays the AERR Capability register values
//    Only display the reg with bits set
//-----------------------------------------------------
void static _displayAERRCap(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func)
{
    LwU32 tempVar;
    PhysAddr Addr;
    LwU32 BAR0;
    LwU16 offset;
    LW_STATUS status = FindExtendedCapOffset(domain, bus, dev, func, PCIE_AERR_CAP_ID, &offset);
    LwU16 VenID, DevID;
    LwU32 i;

    if(offset == PCIE_ILWALID_CAPABILITY_OFFSET || status == LW_ERR_GENERIC)
    {
        _findDevVenID(domain, bus, dev, func, &VenID, &DevID);

        for (i=0; capTable[i].vendorID != 0 && !(capTable[i].vendorID==VenID && capTable[i].deviceID == DevID && capTable[i].capID == PCIE_AERR_CAP_ID) ;i++);

        if(capTable[i].vendorID==VenID && capTable[i].deviceID == DevID)
            offset = capTable[i].offset;
        else return;
    }

    dprintf("lw: Advanced Error Reporting Capability\n\n");

    tempVar = PcieRead32(domain, bus, dev, func, offset+4);

    dprintf("lw: LW_XVR_ERPTCAP_UCERR:                                     0x%08x\n", tempVar);

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR, _TRAINING_ERR, tempVar) == LW_XVR_ERPTCAP_UCERR_TRAINING_ERR_TRUE)
        dprintf("lw:  TRAINING_ERR_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR, _DLINK_PROTO_ERR, tempVar) == LW_XVR_ERPTCAP_UCERR_DLINK_PROTO_ERR_TRUE)
        dprintf("lw:  DLINK_PROTO_ERR_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR, _POS_TLP, tempVar) == LW_XVR_ERPTCAP_UCERR_POS_TLP_TRUE)
        dprintf("lw:  POS_TLP_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR, _FC_PROTO_ERR, tempVar) != LW_XVR_ERPTCAP_UCERR_FC_PROTO_ERR_DEFAULT)
        dprintf("lw:  FC_PROTO_ERR Bit Set\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR, _COMP_TO, tempVar) == LW_XVR_ERPTCAP_UCERR_COMP_TO_TRUE)
        dprintf("lw:  COMP_TO_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR, _COMP_ABORT, tempVar) == LW_XVR_ERPTCAP_UCERR_COMP_ABORT_TRUE)
        dprintf("lw:  COMP_ABORT_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR, _UNEXP_COMP, tempVar) == LW_XVR_ERPTCAP_UCERR_UNEXP_COMP_TRUE)
        dprintf("lw:  UNEXP_COMP_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR, _RCV_OVFL, tempVar) != LW_XVR_ERPTCAP_UCERR_RCV_OVFL_DEFAULT)
        dprintf("lw:  RCV_OVFL Bit Set\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR, _MF_TLP, tempVar) == LW_XVR_ERPTCAP_UCERR_MF_TLP_TRUE)
        dprintf("lw:  MF_TLP_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR, _ECRC_ERR, tempVar) == LW_XVR_ERPTCAP_UCERR_ECRC_ERR_TRUE)
        dprintf("lw:  ECRC_ERR_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR, _UNSUP_REQ_ERR, tempVar) == LW_XVR_ERPTCAP_UCERR_UNSUP_REQ_ERR_TRUE)
        dprintf("lw:  UNSUP_REQ_ERR_TRUE\n");

    dprintf("\n");
    tempVar = PcieRead32(domain, bus, dev, func, offset+8);
   
    dprintf("lw: LW_XVR_ERPTCAP_UCERR_MK:                                  0x%08x\n", tempVar);

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_MK, _TRAINING_ERR, tempVar) == LW_XVR_ERPTCAP_UCERR_MK_TRAINING_ERR_MASKED)
        dprintf("lw:  TRAINING_ERR_MASKED\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_MK, _DLINK_PROTO_ERR, tempVar) == LW_XVR_ERPTCAP_UCERR_MK_DLINK_PROTO_ERR_MASKED)
        dprintf("lw:  DLINK_PROTO_ERR_MASKED\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_MK, _POS_TLP, tempVar) == LW_XVR_ERPTCAP_UCERR_MK_POS_TLP_MASKED)
        dprintf("lw:  POS_TLP_MASKED\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_MK, _FC_PROTO_ERR, tempVar) != LW_XVR_ERPTCAP_UCERR_MK_FC_PROTO_ERR_DEFAULT)
        dprintf("lw:  FC_PROTO_ERR Bit Set\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_MK, _COMP_TO, tempVar) == LW_XVR_ERPTCAP_UCERR_MK_COMP_TO_MASKED)
        dprintf("lw:  COMP_TO_MASKED\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_MK, _COMP_ABORT, tempVar) == LW_XVR_ERPTCAP_UCERR_MK_COMP_ABORT_MASKED)
        dprintf("lw:  COMP_ABORT_MASKED\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_MK, _UNEXP_COMP, tempVar) == LW_XVR_ERPTCAP_UCERR_MK_UNEXP_COMP_MASKED)
        dprintf("lw:  UNEXP_COMP_MASKED\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_MK, _RCV_OVFL, tempVar) != LW_XVR_ERPTCAP_UCERR_MK_RCV_OVFL_DEFAULT)
        dprintf("lw:  RCV_OVFL Bit Set\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_MK, _MF_TLP, tempVar) == LW_XVR_ERPTCAP_UCERR_MK_MF_TLP_MASKED)
        dprintf("lw:  MF_TLP_MASKED\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_MK, _ECRC_ERR, tempVar) == LW_XVR_ERPTCAP_UCERR_MK_ECRC_ERR_MASKED)
        dprintf("lw:  ECRC_ERR_MASKED\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_MK, _UNSUP_REQ_ERR, tempVar) == LW_XVR_ERPTCAP_UCERR_MK_UNSUP_REQ_ERR_MASKED)
        dprintf("lw:  UNSUP_REQ_ERR_MASKED\n");

    dprintf("\n");
    tempVar = PcieRead32(domain, bus, dev, func, offset+0xC);
   
    dprintf("lw: LW_XVR_ERPTCAP_UCERR_SEVR:                                0x%08x\n", tempVar);

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_SEVR, _TRAINING_ERR, tempVar) == LW_XVR_ERPTCAP_UCERR_SEVR_TRAINING_ERR_FATAL)
        dprintf("lw:  TRAINING_ERR_FATAL\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_SEVR, _DLINK_PROTO_ERR, tempVar) == LW_XVR_ERPTCAP_UCERR_SEVR_DLINK_PROTO_ERR_FATAL)
        dprintf("lw:  DLINK_PROTO_ERR_FATAL\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_SEVR, _POS_TLP, tempVar) == LW_XVR_ERPTCAP_UCERR_SEVR_POS_TLP_FATAL)
        dprintf("lw:  POS_TLP_FATAL\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_SEVR, _FC_PROTO_ERR, tempVar) != LW_XVR_ERPTCAP_UCERR_SEVR_FC_PROTO_ERR_FATAL)
        dprintf("lw:  FC_PROTO_ERR_FATAL\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_SEVR, _COMP_TO, tempVar) == LW_XVR_ERPTCAP_UCERR_SEVR_COMP_TO_FATAL)
        dprintf("lw:  COMP_TO_FATAL\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_SEVR, _COMP_ABORT, tempVar) == LW_XVR_ERPTCAP_UCERR_SEVR_COMP_ABORT_FATAL)
        dprintf("lw:  COMP_ABORT_FATAL\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_SEVR, _UNEXP_COMP, tempVar) == LW_XVR_ERPTCAP_UCERR_SEVR_UNEXP_COMP_FATAL)
        dprintf("lw:  UNEXP_COMP_FATAL\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_SEVR, _RCV_OVFL, tempVar) != LW_XVR_ERPTCAP_UCERR_SEVR_RCV_OVFL_FATAL)
        dprintf("lw:  RCV_OVFL_FATAL\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_SEVR, _MF_TLP, tempVar) == LW_XVR_ERPTCAP_UCERR_SEVR_MF_TLP_FATAL)
        dprintf("lw:  MF_TLP_FATAL\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_SEVR, _ECRC_ERR, tempVar) == LW_XVR_ERPTCAP_UCERR_SEVR_ECRC_ERR_FATAL)
        dprintf("lw:  ECRC_ERR_FATAL\n");

    if(DRF_VAL(_XVR, _ERPTCAP_UCERR_SEVR, _UNSUP_REQ_ERR, tempVar) == LW_XVR_ERPTCAP_UCERR_SEVR_UNSUP_REQ_ERR_FATAL)
        dprintf("lw:  UNSUP_REQ_ERR_FATAL\n");

    dprintf("\n");
    tempVar = PcieRead32(domain, bus, dev, func, offset+0x10);
   
    dprintf("lw: LW_XVR_ERPTCAP_CERR:                                      0x%08x\n", tempVar);

    if(DRF_VAL(_XVR, _ERPTCAP_CERR, _RCV_ERR, tempVar) == LW_XVR_ERPTCAP_CERR_RCV_ERR_TRUE)
        dprintf("lw:  RCV_ERR_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_CERR, _BAD_TLP, tempVar) == LW_XVR_ERPTCAP_CERR_BAD_TLP_TRUE)
        dprintf("lw:  BAD_TLP_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_CERR, _BAD_DLLP, tempVar) == LW_XVR_ERPTCAP_CERR_BAD_DLLP_TRUE)
        dprintf("lw:  BAD_DLLP_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_CERR, _RPLY_RLOV, tempVar) == LW_XVR_ERPTCAP_CERR_RPLY_RLOV_TRUE)
        dprintf("lw:  RPLY_RLOV_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_CERR, _RPLY_TO, tempVar) == LW_XVR_ERPTCAP_CERR_RPLY_TO_TRUE)
        dprintf("lw:  RPLY_TO_TRUE\n");

    dprintf("\n");
    tempVar = PcieRead32(domain, bus, dev, func, offset+0x14);
   
    dprintf("lw: LW_XVR_ERPTCAP_CERR_MK:                                   0x%08x\n", tempVar);

    if(DRF_VAL(_XVR, _ERPTCAP_CERR_MK, _RCV_ERR, tempVar) == LW_XVR_ERPTCAP_CERR_MK_RCV_ERR_MASKED)
        dprintf("lw:  RCV_ERR_MASKED\n");

    if(DRF_VAL(_XVR, _ERPTCAP_CERR_MK, _BAD_TLP, tempVar) == LW_XVR_ERPTCAP_CERR_MK_BAD_TLP_MASKED)
        dprintf("lw:  BAD_TLP_MASKED\n");

    if(DRF_VAL(_XVR, _ERPTCAP_CERR_MK, _BAD_DLLP, tempVar) == LW_XVR_ERPTCAP_CERR_MK_BAD_DLLP_MASKED)
        dprintf("lw:  BAD_DLLP_MASKED\n");

    if(DRF_VAL(_XVR, _ERPTCAP_CERR_MK, _RPLY_RLOV, tempVar) == LW_XVR_ERPTCAP_CERR_MK_RPLY_RLOV_MASKED)
        dprintf("lw:  RPLY_RLOV_MASKED\n");

    if(DRF_VAL(_XVR, _ERPTCAP_CERR_MK, _RPLY_TO, tempVar) == LW_XVR_ERPTCAP_CERR_MK_RPLY_TO_MASKED)
        dprintf("lw:  RPLY_TO_MASKED\n");

    dprintf("\n");
    tempVar = PcieRead32(domain, bus, dev, func, offset+0x18);
   
    dprintf("lw: LW_XVR_ERPTCAP_ADV_ERR_CAP_CNTL:                          0x%08x\n", tempVar);

    dprintf("lw:  ERR_PTR:                                                 0x%04x\n", DRF_VAL(_XVR, _ERPTCAP_ADV_ERR_CAP_CNTL, _ERR_PTR, tempVar));

    if(DRF_VAL(_XVR, _ERPTCAP_ADV_ERR_CAP_CNTL, _ECRC_GEN_CAP, tempVar) == LW_XVR_ERPTCAP_ADV_ERR_CAP_CNTL_ECRC_GEN_CAP_TRUE)
        dprintf("lw:  ECRC_GEN_CAP_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_ADV_ERR_CAP_CNTL, _ECRC_GEN_EN, tempVar) == LW_XVR_ERPTCAP_ADV_ERR_CAP_CNTL_ECRC_GEN_EN_TRUE)
        dprintf("lw:  ECRC_GEN_EN_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_ADV_ERR_CAP_CNTL, _ECRC_CHK_CAP, tempVar) == LW_XVR_ERPTCAP_ADV_ERR_CAP_CNTL_ECRC_CHK_CAP_TRUE)
        dprintf("lw:  ECRC_CHK_CAP_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_ADV_ERR_CAP_CNTL, _ECRC_CHK_EN, tempVar) == LW_XVR_ERPTCAP_ADV_ERR_CAP_CNTL_ECRC_CHK_EN_TRUE)
        dprintf("lw:  ECRC_CHK_EN_TRUE\n");

    dprintf("\n");
    tempVar = PcieRead32(domain, bus, dev, func, offset+0x1C);
   
    dprintf("lw: LW_XVR_ERPTCAP_HDR_LOG_DW0:                               0x%08x\n", tempVar);

    dprintf("\n");
    tempVar = PcieRead32(domain, bus, dev, func, offset+0x20);

    dprintf("lw: LW_XVR_ERPTCAP_HDR_LOG_DW1:                               0x%08x\n", tempVar);

    dprintf("\n");
    tempVar = PcieRead32(domain, bus, dev, func, offset+0x24);
   
    dprintf("lw: LW_XVR_ERPTCAP_HDR_LOG_DW2:                               0x%08x\n", tempVar);

    dprintf("\n");
    tempVar = PcieRead32(domain, bus, dev, func, offset+0x28);   

    dprintf("lw: LW_XVR_ERPTCAP_HDR_LOG_DW3:                               0x%08x\n", tempVar);

    // Root Error Command Register
    dprintf("\n");
    tempVar = PcieRead32(domain, bus, dev, func, offset+0x2C);  
    
    dprintf("lw: LW_XVR_ERPTCAP_ERR_CMD:                                   0x%08x\n", tempVar);

    if(DRF_VAL(_XVR, _ERPTCAP_ERR_CMD, _COR_ERR_RPT_EN, tempVar) == LW_XVR_ERPTCAP_ERR_CMD_COR_ERR_RPT_EN_TRUE)
        dprintf("lw:  COR_ERR_RPT_EN_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_ERR_CMD, _NONFATAL_ERR_RPT_EN, tempVar) == LW_XVR_ERPTCAP_ERR_CMD_NONFATAL_ERR_RPT_EN_TRUE)
        dprintf("lw:  NONFATAL_ERR_RPT_EN_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_ERR_CMD, _FATAL_ERR_RPT_EN, tempVar) == LW_XVR_ERPTCAP_ERR_CMD_FATAL_ERR_RPT_EN_TRUE)
        dprintf("lw:  FATAL_ERR_RPT_EN_TRUE\n");

    // Root Error Status Register
    dprintf("\n");
    tempVar = PcieRead32(domain, bus, dev, func, offset+0x30);   

    dprintf("lw: LW_XVR_ERPTCAP_ERR_STS:                                   0x%08x\n", tempVar);

    if(DRF_VAL(_XVR, _ERPTCAP_ERR_STS, _COR_RCVD, tempVar) == LW_XVR_ERPTCAP_ERR_STS_COR_RCVD_TRUE)
        dprintf("lw:  COR_RCVD_TRUE\n");

    if(DRF_VAL(_XVR, _ERPTCAP_ERR_STS, _MULT_COR_RCVD, tempVar) == LW_XVR_ERPTCAP_ERR_STS_MULT_COR_RCVD_TRUE)
        dprintf("lw:  MULT_COR_RCVD_TRUE\n");
    
    if(DRF_VAL(_XVR, _ERPTCAP_ERR_STS, _UNCOR_RCVD, tempVar) == LW_XVR_ERPTCAP_ERR_STS_UNCOR_RCVD_TRUE)
        dprintf("lw:  UNCOR_RCVD_TRUE\n");
    
    if(DRF_VAL(_XVR, _ERPTCAP_ERR_STS, _MULT_UNCOR_RCVD, tempVar) == LW_XVR_ERPTCAP_ERR_STS_MULT_UNCOR_RCVD_TRUE)
        dprintf("lw:  MULT_UNCOR_RCVD_TRUE\n");
    
    if(DRF_VAL(_XVR, _ERPTCAP_ERR_STS, _FIRST_FATAL_RCVD, tempVar) == LW_XVR_ERPTCAP_ERR_STS_FIRST_FATAL_RCVD_TRUE)
        dprintf("lw:  FIRST_FATAL_RCVD_TRUE\n");
    
    if(DRF_VAL(_XVR, _ERPTCAP_ERR_STS, _NONFATAL_RCVD, tempVar) == LW_XVR_ERPTCAP_ERR_STS_NONFATAL_RCVD_TRUE)
        dprintf("lw:  NONFATAL_RCVD_TRUE\n");
    
    if(DRF_VAL(_XVR, _ERPTCAP_ERR_STS, _FATAL_RCVD, tempVar) == LW_XVR_ERPTCAP_ERR_STS_FATAL_RCVD_TRUE)
        dprintf("lw:  FATAL_RCVD_TRUE\n");
    
    dprintf("lw:  ADV_ERR_INTR_MSG_NUM                                     0x%02x\n", DRF_VAL(_XVR, _ERPTCAP_ERR_STS, _ADV_ERR_INTR_MSG_NUM, tempVar));

    dprintf("\n");
    tempVar = PcieRead32(domain, bus, dev, func, offset+0x34);   

    dprintf("lw: LW_XVR_ERPTCAP_ERR_ID:                                    0x%08x\n", tempVar);

    dprintf("lw:  ERR_COR                                                  0x%04x\n", DRF_VAL(_XVR, _ERPTCAP_ERR_ID, _ERR_COR, tempVar));

    dprintf("lw:  ERR_UNCOR                                                0x%04x\n", DRF_VAL(_XVR, _ERPTCAP_ERR_ID, _ERR_UNCOR, tempVar));

    dprintf("\n");

    osPciRead32( domain, bus, dev, func, &BAR0, PCIE_BAR0_OFFSET);

    if(ConfigBaseFound == TRUE)
    {
        Addr = (Get_PCI_Config_Base(FHBConfigBaseAddr, domain, bus, dev, func) + offset);
        printData(Addr, 0x40);
    }
    else if(BAR0 == lwBar0)
    {
        Addr = lwBar0 + 0x00088000 + offset;
        printData(Addr, 0x40);
    }
}

//-----------------------------------------------------
// FindCfgBase
//
//   Finds the base address of the PCI Config Space
//   by comparing the data values returned by
//   osPciRead and by physical memory read
//-----------------------------------------------------
LwU32  FindCfgBase(LwU16 FHBdomain, LwU8 FHBbus, LwU8 FHBdev, LwU8 FHBfunc, LwU32 *pBase)
{
#ifndef SIM_BUILD

    LwU32 BaseAddr = 0xF0000000;    // start to check from address 0xF0000000 
    LwU32 Data32, PhysData32;

    osPciRead32( FHBdomain, FHBbus, FHBdev, FHBfunc, &Data32, 0);
    PhysData32 = RD_PHYS32(BaseAddr);

    while((Data32 != PhysData32) && (BaseAddr >= 0x80000000))
    {
        // decrement the base address to 0x80000000, the Config Space must be 256MB-aligned and
        // should be within 0x8-F0000000
        BaseAddr -= 0x10000000;
        osPciRead32( FHBdomain, FHBbus, FHBdev, FHBfunc, &Data32, 0);
        PhysData32 = RD_PHYS32(BaseAddr);
    }

    // Check the header and see if they match or not
    if(Data32 == PhysData32)
    {
        ConfigBaseFound = TRUE;
        *pBase = BaseAddr;
        return LW_OK;
    }

#endif

    // If data not match, base not found!!
    ConfigBaseFound = FALSE;
    return LW_ERR_GENERIC;
}
