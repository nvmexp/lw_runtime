#ifndef _IBMNPU_H_
#define _IBMNPU_H_

#include "os.h"


#define PCI_DEVICE_CLASS_CODE_IBMNPU                0x068000
#define PCI_DEVICE_ID_IBMNPU                        0x04ea
#define PCI_VENDOR_ID_IBM                           0x1014
#define PCI_VENDOR_CAP_IBMNPU                       0x9
#define PCI_VENDOR_CAP_OFFSET_PROC_STATUS           0x4
#define PCI_VENDOR_CAP_OFFSET_PROC_CONTROL          0x8
#define PCI_VENDOR_CAP_OFFSET_BRICK_NUM             0xC


#define MAX_IBMNPU_DEVICES                          2
#define MAX_IBMNPU_BRICKS_PER_DEVICE                6
#define MAX_IBMNPU_BARS_PER_BRICK                   2

#define IBMNPU_DLPL_BASE_OFFSET                     0x0
// DLPL size is 64 KB
#define IBMNPU_DLPL_SIZE                            0x10000
#define IBMNPU_NTL_BASE_OFFSET                      0x10000
// NTL size is 64 KB
#define IBMNPU_NTL_SIZE                             0x10000
#define IBMNPU_PHY_BASE_OFFSET                      0x0
// PHY size is 2 MB
#define IBMNPU_PHY_SIZE                             0x200000

#define LW_IBMNPU_BRICK_NUM                         7:0

#define LW_IBMNPU_PROC_STATUS_BUSY                  30:30
#define LW_IBMNPU_PROC_STATUS_BUSY_IN_PROGRESS      0x00000000
#define LW_IBMNPU_PROC_STATUS_BUSY_COMPLETE         0x00000001
#define LW_IBMNPU_PROC_STATUS_RET_CODE              2:0
#define LW_IBMNPU_PROC_STATUS_RET_CODE_SUCCESS      0x00000000
#define LW_IBMNPU_PROC_STATUS_RET_CODE_TRANS_FAIL   0x00000001
#define LW_IBMNPU_PROC_STATUS_RET_CODE_PERM_FAIL    0x00000002
#define LW_IBMNPU_PROC_STATUS_RET_CODE_ABORTED      0x00000003

#define NPU_DLPL_RD32(dev, off) ibmnpuDLPLRead((dev), (off))
#define NPU_DLPL_WR32(dev, off, data) ibmnpuDLPLWrite((dev), (off), (data))

#define NPU_NTL_RD64(dev, off) ibmnpuNTLRead((dev), (off))
#define NPU_NTL_WR64(dev, off, data) ibmnpuNTLWrite((dev), (off), (data))

#define NPU_CFG_RD8(dev, off) ibmnpuCfgRead08((dev), (off))
#define NPU_CFG_RD16(dev, off) ibmnpuCfgRead16((dev), (off))
#define NPU_CFG_RD32(dev, off) ibmnpuCfgRead32((dev), (off))
#define NPU_CFG_WR8(dev, off, data) ibmnpuCfgWrite08((dev), (off), (data))
#define NPU_CFG_WR16(dev, off, data) ibmnpuCfgWrite16((dev), (off), (data))
#define NPU_CFG_WR32(dev, off, data) ibmnpuCfgWrite32((dev), (off), (data))

//
// Structures
//

typedef struct
{
    LwU16 domain;
    LwU8 bus, device, func;
} PCI_DEV_ENT;

typedef struct
{
    LwU64 bar;
    LwU64 size;
    void *pBar;
} BAR_ENT;

typedef struct
{
    LwU64 priBase;
    LwU64 priLimit;
    char *pMap;
} PRI_RANGE;

typedef struct
{
    PCI_DEV_ENT pciInfo;
    BAR_ENT bars[MAX_IBMNPU_BARS_PER_BRICK];
    LwU8 intLine;
    LwU32 procControlAddr;
    LwU32 procStatusAddr;

    PRI_RANGE dlPriBase;
    PRI_RANGE ntlPriBase;
    PRI_RANGE phyPriBase;

    int initialized;
} ibmnpu_device;

typedef enum
{
    IBMNPU_PROC_ABORT                       = 0x0,
    IBMNPU_PROC_NOOP                        = 0x1,
    IBMNPU_PROC_START_DMA_SYNC              = 0x2,
    IBMNPU_PROC_RESET_AT                    = 0x3,
    IBMNPU_PROC_NAPLES_PHY_RESET            = 0x4,
    IBMNPU_PROC_NAPLES_PHY_TX_ZCAL          = 0x5,
    IBMNPU_PROC_NAPLES_PHY_RX_DCCAL         = 0x6,
    IBMNPU_PROC_NAPLES_PHY_TX_RXCAL_ENABLE  = 0x7,
    IBMNPU_PROC_NAPLES_PHY_TX_RXCAL_DISABLE = 0x8,
    IBMNPU_PROC_NAPLES_PHY_RX_TRAINING      = 0x9,
    IBMNPU_PROC_NAPLES_PHY_DL_RESET         = 0xA
} ibmnpu_control_procedure;

typedef enum
{
    IBMNPU_PROC_SUCCESS                     = 0x0,
    IBMNPU_PROC_TRANSIENT_FAILURE           = 0x1,
    IBMNPU_PROC_PERMANENT_FAILURE           = 0x2,
    IBMNPU_PROC_ABORTED                     = 0x3
} ibmnpu_control_return;

LwU16 BYTESWAP_16(LwU16 val); 
LwU32 BYTESWAP_32(LwU32 val);
LwU64 BYTESWAP_64(LwU64 val);

BOOL ibmnpuIsInitialized(void);
BOOL ibmnpuLinkExists(LwS32 npu, LwS32 brick);
void ibmnpuInit(void);
void ibmnpuTeardown(void);
ibmnpu_control_return ibmnpuControl(ibmnpu_device *device,
                                    ibmnpu_control_procedure proc);

LwU32 ibmnpuDLPLRead(ibmnpu_device *device, LwU32 offset);
void  ibmnpuDLPLWrite(ibmnpu_device *device, LwU32 offset, LwU32 data);
LwU64 ibmnpuNTLRead(ibmnpu_device *device, LwU64 offset);
void  ibmnpuNTLWrite(ibmnpu_device *device, LwU64 offset, LwU64 data);

LwU8  ibmnpuCfgRead08(ibmnpu_device *device, LwU32 offset);
LwU16 ibmnpuCfgRead16(ibmnpu_device *device, LwU32 offset);
LwU32 ibmnpuCfgRead32(ibmnpu_device *device, LwU32 offset);
void  ibmnpuCfgWrite08(ibmnpu_device *device, LwU32 offset, LwU8 data);
void  ibmnpuCfgWrite16(ibmnpu_device *device, LwU32 offset, LwU16 data);
void  ibmnpuCfgWrite32(ibmnpu_device *device, LwU32 offset, LwU32 data);

void ibmnpuCommandLINKS(void);
void ibmnpuCommandDEVICES(void);
void ibmnpuCommandREAD(LwS32 npu, LwS32 brick, LwU32 type, LwU64 offset);
void ibmnpuCommandWRITE(LwS32 npu, LwS32 brick, LwU32 type, LwU64 offset, LwU64 data);
void ibmnpuCommandCTRL(LwS32 npu, LwS32 brick, LwU32 proc);
void ibmnpuCommandDUMPDLPL(LwS32 npu, LwS32 brick);
void ibmnpuCommandDUMPNTL(LwS32 npu, LwS32 brick);
void ibmnpuCommandDUMPUPHYS(LwS32 npu, LwS32 brick);
void ibmnpuPrintHelp();
void ibmnpuMissingLink(LwS32 npu, LwS32 brick);
void ibmnpuCommandUnimplemented(void);

#endif // _IBMNPU_H_
