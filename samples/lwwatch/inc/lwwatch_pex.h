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
// pex.h
//
//*****************************************************

#ifndef _PEX_H_
#define _PEX_H_

#include "os.h"

#define PCIE_MAX_DOMAIN     65536
#define PCIE_MAX_BUS        256
#define PCIE_MAX_DEV        32
#define PCIE_MAX_FUNC       8
#define PCIE_MAX_OFFSET     0x1000
#define PCIE_BAR0_OFFSET    0x10

#define PCIE_ILWALID_CAPABILITY_OFFSET 0x0

//
// PEX functions
//
void printPci(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func, LwU32 offset, LwU32 length);

void printPcie3EvtLogDmp(void);
void getPexAllInfo(void);
void getPexSpecifiedInfo(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func);
void getPexGPUInfo(void);

int Intel_25XX_setupFunc(LwU16, LwU8, LwU8, LwU8, LwU32*);
int Intel_359E_setupFunc(LwU16, LwU8, LwU8, LwU8, LwU32*);
int Intel_27XX_setupFunc(LwU16, LwU8, LwU8, LwU8, LwU32*);
int Lwidia_CK804_setupFunc(LwU16, LwU8, LwU8, LwU8, LwU32*);
int Lwidia_C19_setupFunc(LwU16, LwU8, LwU8, LwU8, LwU32*);
int Lwidia_C51_setupFunc(LwU16, LwU8, LwU8, LwU8, LwU32*);
int SiS_656_setupFunc(LwU16, LwU8, LwU8, LwU8, LwU32*);

LW_STATUS  FindExtendedCapOffset(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func, LwU16 capID, LwU16 *capOffset);
LW_STATUS  FindStdCapOffset(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func, LwU16 capID, LwU8 *capOffset);
LwU32      PcieRead32(LwU16 domain, LwU8 bus, LwU8 dev, LwU8 func, LwU32 offset);
LW_STATUS       FindCfgBase(LwU16 FHBdomain, LwU8 FHBbus, LwU8 FHBdev, LwU8 FHBfunc, LwU32 *pBase);

typedef struct CSINFO CSINFO;
typedef struct CAP_TABLE CAP_TABLE;

enum
{
    PCIE_ENDPT = 0,
    LEG_ENDPT,
    ROOT_PT = 4,
    UP_PT,
    DW_PT,
    EXP2PCI_BR,
    PCI2EXP_BR
};

struct CSINFO
{
    LwU16 vendorID, deviceID;
    const char *name;
    int (*setupFunc)(LwU16, LwU8, LwU8, LwU8, LwU32*);
};

struct CAP_TABLE
{
    LwU16 vendorID, deviceID;
    LwU16 capID;
    LwU16 offset;
};

#define PCI_Addr_Domain_Start 28
#define PCI_Addr_Bus_Start  20
#define PCI_Addr_Dev_Start  15
#define PCI_Addr_Func_Start  12

#define Get_PCI_Config_Base(FHBConfigBaseAddr, domain, bus, dev, func)     ((FHBConfigBaseAddr) | (domain<<PCI_Addr_Domain_Start) | (bus<< PCI_Addr_Bus_Start) | (dev<< PCI_Addr_Dev_Start) | (func<< PCI_Addr_Func_Start))

#define PCI_VENDOR_ID_AMD       0x1022
#define PCI_VENDOR_ID_ALI       0x10B9 
#define PCI_VENDOR_ID_LWIDIA    0x10DE

#define PCI_VENDOR_ID_INTEL     0x8086
#define PCI_VENDOR_ID_VIA       0x1106
#define PCI_VENDOR_ID_RCC       0x1166
#define PCI_VENDOR_ID_MICRON_1  0x1042
#define PCI_VENDOR_ID_MICRON_2  0x1344
#define PCI_VENDOR_ID_APPLE     0x106B
#define PCI_VENDOR_ID_SIS       0x1039
#define PCI_VENDOR_ID_ATI       0x1002
#define PCI_VENDOR_ID_TRANSMETA 0x1279
#define PCI_VENDOR_ID_HP        0x103C

#define PCI_CLASS_BRIDGE_DEV    0x06
#define PCI_SUBCLASS_BR_HOST    0x00

#define PCI_ILWALID_VENDORID    0xFFFF
#define PCI_ILWALID_DEVICEID    0xFFFF

// Intel Grantsdale definitions
#define DEVICE_ID_INTEL_2580_HOST_BRIDGE    0x2580
#define DEVICE_ID_INTEL_2581_ROOT_PORT      0x2581
#define INTEL_25XX_CONFIG_SPACE_BASE        0x48

// Intel Alderwood definitions
#define DEVICE_ID_INTEL_2584_HOST_BRIDGE    0x2584
#define DEVICE_ID_INTEL_2585_ROOT_PORT      0x2585

// Intel Alviso definitions
#define DEVICE_ID_INTEL_2590_HOST_BRIDGE    0x2590
#define DEVICE_ID_INTEL_2591_ROOT_PORT      0x2591

// Intel Tumwater definitions
#define DEVICE_ID_INTEL_359E_HOST_BRIDGE    0x359E
#define DEVICE_ID_INTEL_3597_ROOT_PORT      0x3597
#define INTEL_359E_CONFIG_SPACE_BASE        0xcc

// LWPU CK804 
#define LWIDIA_CK804_CONFIG_SPACE_BASE                   0x90
#define LWIDIA_CK804_CONFIG_SPACE_BASE_ADDRESS           11:0 // mapped to 39:28
#define LWIDIA_CK804_CONFIG_SPACE_BASE_ADDRESS_ENABLE   12:12 // Bit 12, Enable

// LWPU C19
#define LWIDIA_C19_CONFIG_SPACE_BASE                     0x88
#define LWIDIA_C19_CONFIG_SPACE_BASE_ADDRESS              7:0 // mapped to 39:28
#define LWIDIA_C19_CONFIG_SPACE_BASE_ADDRESS_ENABLE     12:12 // Bit 12, Enable

// LWPU C51
#define LWIDIA_C51_CONFIG_SPACE_BASE                     0x90
#define LWIDIA_C51_CONFIG_SPACE_BASE_ADDRESS             14:3 // mapped to 39:28
#define LWIDIA_C51_CONFIG_SPACE_BASE_ADDRESS_ENABLE     31:31 // Bit 31, Enable

// SiS 656
#define SIS_656_CONFIG_SPACE_BASE           0xE0
#define SIS_656_CONFIG_SPACE_BASE_ADDRESS   3:0     // mapped to 31:28

#endif // _PEX_H_
