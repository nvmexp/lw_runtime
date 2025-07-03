/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2002-2020 by LWPU Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to LWPU
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of LWPU Corporation is prohibited.
 *
 * _LWRM_COPYRIGHT_END_
 */

#ifndef _RMCDOBS_H_
#define _RMCDOBS_H_

//******************************************************************************
//
// Module Name: RMCDOBS.H
//
// This file contains obsolete structures and constants that define the resource manager
// specific data for the crash dump file. The record definitions defined here
// are always stored after the crash dump file header. Each record defined here
// is preceded by the LWCD_RECORD structure.
//
//******************************************************************************
#include "lwgputypes.h"
#include "lwcd.h"

// Define the resource manager group record types
typedef enum _RMCD_RECORD_TYPE_OBS
{
    RmGlobalInfo            = 0,    // RM global info. record (Always first)
    RmLwrrentStateInfo      = 1,    // RM current state info. record
    RmFifoErrorInfo         = 2,    // RM FIFO error information record
    GraphicsErrorInfo       = 3,    // RM graphics error information record
    RmFifoErrorInfo_V2      = 5,    // Revised Version 2 of FIFO error information record
    GraphicsErrorInfo_V2    = 6,    // Revised Version 2 of Graphics error information record
    RmGlobalInfo_V2         = 7,    // Revised Version 2 of Global Info
    RmBusErrorInfo          = 8,    // RM Bus Error Info
    RmFifoErrorInfo_V3      = 11,   // Revised Version 3 of Fifo Info
    GraphicsErrorInfo_V3    = 12,    // Revised Version 3 of Graphics Info

    RmRC2SwDbgBreakpoint                = 31,   // Retail logged breakpoint info
    RmRC2SwRmAssert                     = 32,   // Retail logged RM_ASSERT info
    RmRC2GpuTimeout                     = 33,   // Retail logged RM_TIMEOUT events

    GraphicsErrorInfo_V4    = 103,    // Revised Version 4 of Graphics Info
    RmVbiosInfo             = 109,    // Vbios Info
    RmJournalInfo           = 112,    // Rm Journal

    RmRC2SmuCommandInfo     = 114,    // Logs the communication with SMU.
    RmRC2PstateInfo         = 115,
    RmJournalInfo_V2        = 118,    // Rm Journal for header size issue
} RMCD_RECORD_TYPE_OBS;

#define ERROR_TRACE_SIZE (16)
#define GLOBALINFO_FLAGS_OVERCLOCK_MCLK               BIT(0)
#define GLOBALINFO_FLAGS_OVERCLOCK_G                  BIT(1)
#define GLOBALINFO_FLAGS_OVERCLOCK_S                  BIT(2)
#define GLOBALINFO_FLAGS_OVERCLOCK_R                  BIT(3)
#define GLOBALINFO_FLAGS_FUSE_ERROR                   BIT(4)
#define GLOBALINFO_FLAGS_SLI_MODE                     BIT(5)
#define GLOBALINFO_FLAGS_MULTI_MON                    BIT(6)
#define GLOBALINFO_FLAGS_POWERMIZER                   BIT(7)
#define GLOBALINFO_FLAGS_DAMAGE_ROUTINE_CRASH         BIT(8)
#define GLOBALINFO_FLAGS_DRIVER_HACKED_TO_ENABLE_SLI  BIT(10)   // BIT(9) cannot be used as it has been reserved for DEVID hack in mc.proto

// Some of these data structures are obsolete but keep so that we can still work with older drivers
// Define the RmGlobalInfo record structure <Obsolete>
typedef struct _RmGlobalInfo_RECORD
{
    LWCD_RECORD Header;                 // Global information record header
    LwU32       dwPCI_ID;               // Device PCI vendor/device ID value
    LwU32       dwSubSys_ID;            // Subsystem vendor/device ID value
    LwU32       dwChipRevision;         // Chip revision value (PMC Boot 0)
    LwU32       dwStrap;                // Chip boot strapping value
    LwU32       dwNorthBridgeId;        // AGP NB vendor/device ID value
    LwU32       dwBiosRevision;         // BIOS revision value
    LwU32       dwBiosOEMRevision;      // OEM BIOS revision value
    LwU32       dwErrorCount;           // Number of error records to follow
    LwU16       wLW_AgpConf_Status;     // LW AGP configuration status value
    LwU16       wNB_AgpConf_Status;     // NB AGP configuration status value
    LwU32       dwCPU_Type;             // CPU type value (Processor.Type)
} RmGlobalInfo_RECORD, *PRmGlobalInfo_RECORD;

//<Obsolete>
typedef struct _RmGlobalInfo_RECORD_V2
{
    LWCD_RECORD Header;                 // Global information record header
    LwU32       dwPCI_ID;               // Device PCI vendor/device ID value
    LwU32       dwSubSys_ID;            // Subsystem vendor/device ID value
    LwU32       dwChipRevision;         // Chip revision value (PMC Boot 0)
    LwU32       dwStrap;                // Chip boot strapping value
    LwU32       dwNorthBridgeId;        // AGP NB vendor/device ID value
    LwU32       dwBiosRevision;         // BIOS revision value
    LwU32       dwBiosOEMRevision;      // OEM BIOS revision value
    LwU32       dwErrorCount;           // Number of error records to follow
    LwU16       wLW_AgpConf_Status;     // LW AGP configuration status value
    LwU16       wNB_AgpConf_Status;     // NB AGP configuration status value
    LwU32       dwCPU_Type;             // CPU type value (Processor.Type)
    LwU8        bNumPhysicalCPUs;       // Number of Physical CPUs
    LwU8        bNumLogicalCPUs;        // Total number of Logical CPUs
    LwU16       wSpare;                 // Spare to round to 32 byte boundary
} RmGlobalInfo_RECORD_V2, *PRmGlobalInfo_RECORD_V2;

//<Obsolete>
typedef struct
{
    LWCD_RECORD Header;                 // Global information record header
    LwU32       dwPCI_ID;               // Device PCI vendor/device ID value
    LwU32       dwSubSys_ID;            // Subsystem vendor/device ID value
    LwU32       dwChipRevision;         // Chip revision value (PMC Boot 0)
    LwU32       dwStrap;                // Chip boot strapping value
    LwU32       dwNorthBridgeId;        // AGP NB vendor/device ID value
    LwU32       dwBiosRevision;         // BIOS revision value
    LwU32       dwBiosOEMRevision;      // OEM BIOS revision value
    LwU32       dwErrorCount;           // Number of error records to follow
    LwU16       wLW_AgpConf_Status;     // LW AGP configuration status value
    LwU16       wNB_AgpConf_Status;     // NB AGP configuration status value
    LwU32       dwCPU_Type;             // CPU type value (cpuInfo.type)
    LwU32       dwCPU_Caps;             // CPU caps value (cpuInfo.caps)
    LwU8        bNumPhysicalCPUs;       // Number of Physical CPUs
    LwU8        bNumLogicalCPUs;        // Total number of Logical CPUs
    LwU16       wTotalError;            // 0xFFFF means at least 0xFFFF possible more
    LwU8        LastErrors[ERROR_TRACE_SIZE]; // Last 16 Errors
    LwU32       Flags;                  // LW_PBUS_PIPE config, SLI, Multimon
} RmGlobalInfo_RECORD_V3;
typedef RmGlobalInfo_RECORD_V3 *PRmGlobalInfo_RECORD_V3;

//<Obsolete>
// Generic error data (Include in both FIFO and graphics error records)
typedef struct _GENERIC_DATA
{
    LwU16       wLW_AgpConf_Cmd;        // LW AGP configuration at error
    LwU16       wNB_AgpConf_Cmd;        // NB AGP configuration at error
    LwHandle    hChannel;               // Channel handle
    LwU8        cChannelID;             // Channel ID <0-31>
    LwU8        cSubChannel;            // SubChannel <0-7>
    LwU16       wPushBufferSpace;       // Push buffer space
    LwU32       dwTimeHi;               // Error time (High word) <OS Specific>
    LwU32       dwTimeLo;               // Error time (Low word)  <OS Specific>
} GENERIC_DATA, *PGENERIC_DATA;

//<Obsolete>
// Define the RmLwrrentStateInfo record structure
typedef struct _RmLwrrentStateInfo_RECORD
{
    LWCD_RECORD Header;                 // Current state info. record header
    GENERIC_DATA Generic;               // Generic data
    // PCI
    LwU32       dwLW_PBUS_PCI_BAR0;                 // 0x001810 PCI Bar 0 value
    LwU32       dwLW_PBUS_PCI_BAR1;                 // 0x001814 PCI Bar 1 value
    // Fifo
    LwU32       dwLW_PFIFO_INTR_0;                  // 0x002100 FIFO interrupt status
    LwU32       dwLW_PFIFO_INTR_EN_0;               // 0x002140 FIFO interrupt enables
    LwU32       dwLW_PFIFO_CACHES;                  // 0x002500 Cache settings
    LwU32       dwLW_PFIFO_CACHE1_PUSH0;            // 0x003200 Cache 1 pusher access
    LwU32       dwLW_PFIFO_CACHE1_PUSH1;            // 0x003204 Cache 1 pusher channel
    LwU32       dwLW_PFIFO_CACHE1_PULL0;            // 0x003250 Cache 1 puller access/status
    LwU32       dwLW_PFIFO_CACHE1_PUT;              // 0x003210 Cache 1 FIFO put pointer
    LwU32       dwLW_PFIFO_CACHE1_GET;              // 0x003270 Cache 1 FIFO get pointer
    // Master Controller
    LwU32       dwLW_PMC_INTR_0;                    // 0x000100 Master interrupt status
    LwU32       dwLW_PMC_INTR_EN_0;                 // 0x000140 Master interrupt enable
    LwU32       dwLW_PMC_INTR_READ_0;               // 0x000160 Interrupt pin status
    LwU32       dwLW_PMC_ENABLE;                    // 0x000200 Engine enables
    // Graphics Status
    LwU32       dwLW_PGRAPH_STATUS;                 // 0x400700 Graphics engine status
    LwU32       dwLW_PGRAPH_INTR;                   // 0x400100 Graphics engine interrrupt status
    LwU32       dwLW_PGRAPH_CTX_CONTROL;            // 0x400144 Context switch control
    LwU32       dwLW_PGRAPH_CTX_USER;               // 0x400148 User address information
    LwU32       dwLW_PGRAPH_TRAPPED_ADDR;           // 0x400704 Last trapped address (Method/Sub/Chan)
    LwU32       dwLW_PGRAPH_TRAPPED_DATA_LOW;       // 0x400708 Last trapped data value (Low)
    LwU32       dwLW_PGRAPH_TRAPPED_DATA_HIGH;      // 0x40070C Last trapped data value (High)
} RmLwrrentStateInfo_RECORD, *PRmLwrrentStateInfo_RECORD;

//<Obsolete>
// Define the RmLwrrentStateInfo record structure
typedef struct
{
    LWCD_RECORD Header;                             // Current state info. record header
    GENERIC_DATA_V2 Generic;                        // Generic data
    // PCI
    LwU32       dwLW_PBUS_PCI_BAR0;                 // 0x001810 PCI Bar 0 value
    LwU32       dwLW_PBUS_PCI_BAR1;                 // 0x001814 PCI Bar 1 value
    // Fifo
    LwU32       dwLW_PFIFO_INTR_0;                  // 0x002100 FIFO interrupt status
    LwU32       dwLW_PFIFO_INTR_EN_0;               // 0x002140 FIFO interrupt enables
    LwU32       dwLW_PFIFO_CACHES;                  // 0x002500 Cache settings
    LwU32       dwLW_PFIFO_CACHE1_PUSH0;            // 0x003200 Cache 1 pusher access
    LwU32       dwLW_PFIFO_CACHE1_PUSH1;            // 0x003204 Cache 1 pusher channel
    LwU32       dwLW_PFIFO_CACHE1_PULL0;            // 0x003250 Cache 1 puller access/status
    LwU32       dwLW_PFIFO_CACHE1_PUT;              // 0x003210 Cache 1 FIFO put pointer
    LwU32       dwLW_PFIFO_CACHE1_GET;              // 0x003270 Cache 1 FIFO get pointer
    // Master Controller
    LwU32       dwLW_PMC_INTR_0;                    // 0x000100 Master interrupt status
    LwU32       dwLW_PMC_INTR_EN_0;                 // 0x000140 Master interrupt enable
    LwU32       dwLW_PMC_INTR_READ_0;               // 0x000160 Interrupt pin status
    LwU32       dwLW_PMC_ENABLE;                    // 0x000200 Engine enables
    // Graphics Status
    LwU32       dwLW_PGRAPH_STATUS;                 // 0x400700 Graphics engine status
    LwU32       dwLW_PGRAPH_INTR;                   // 0x400100 Graphics engine interrrupt status
    LwU32       dwLW_PGRAPH_CTX_CONTROL;            // 0x400144 Context switch control
    LwU32       dwLW_PGRAPH_CTX_USER;               // 0x400148 User address information
    LwU32       dwLW_PGRAPH_TRAPPED_ADDR;           // 0x400704 Last trapped address (Method/Sub/Chan)
    LwU32       dwLW_PGRAPH_TRAPPED_DATA_LOW;       // 0x400708 Last trapped data value (Low)
    LwU32       dwLW_PGRAPH_TRAPPED_DATA_HIGH;      // 0x40070C Last trapped data value (High)

} RmLwrrentStateInfo_RECORD_V2;
typedef RmLwrrentStateInfo_RECORD_V2 *PRmLwrrentStateInfo_RECORD_V2;

//<Obsolete>
// Define the RmFifoErrorInfo record structure
typedef struct _RmFifoErrorInfo_RECORD
{
    LWCD_RECORD Header;                 // FIFO error information record header
    GENERIC_DATA Generic;               // Generic data
    // Exception State
    LwU32       dwLW_PFIFO_INTR_0;                  // 0x002100 FIFO interrupt status
    LwU32       dwExceptionData;                    // Exception data
    LwU32       dwLW_PFIFO_MODE;                    // FIFO mode???
    // Chip Specific Information
    LwU32       dwLW_PFIFO_CACHE1_DMA_STATE;        // 0x003228 DMA pusher state
    LwU32       dwLW_PFIFO_CACHE1_DMA_DCOUNT;       // 0x0032A0 DMA transfer count
    LwU32       dwLW_PFIFO_CACHE1_DMA_RSVD_SHADOW;  // 0x0032A8 Last reserved DMA command
    LwU32       dwLW_PFIFO_CACHE1_DMA_DATA_SHADOW;  // 0x0032AC Last DMA data fetched
    LwU32       dwLW_PFIFO_CACHE1_DMA_COUNT;        // 0x003364 Remaining dwords in current method
    // DMA Pusher
    LwU32       dwLW_PFIFO_CACHE1_DMA_PUT;          // 0x003240 Cache 1 DMA put pointer
    LwU32       dwLW_PFIFO_CACHE1_DMA_GET;          // 0x003244 Cache 1 DMA get pointer
    LwU32       dwLW_PFIFO_CACHE1_DMA_REF;          // 0x003248 Cache 1 DMA reference count
    LwU32       dwLW_PFIFO_CACHE1_DMA_INSTANCE;     // 0x00322C Cache 1 DMA PRAMIN address
    LwU32       dwLW_PFIFO_CACHE1_DMA_CTL;          // 0x003230 Cache 1 DMA addr. tran. info.
    LwU32       dwLW_PFIFO_CACHE1_DMA_TLB_PTE;      // 0x00323C Cache 1 DMA PTE TLB info.
    // CACHE1 State
    LwU32       dwLW_PFIFO_CACHE1_PUSH1;            // 0x003204 Cache 1 pusher channel
    LwU32       dwLW_PFIFO_CACHE1_PUT;              // 0x003210 Cache 1 FIFO put pointer
    LwU32       dwLW_PFIFO_CACHE1_GET;              // 0x003270 Cache 1 FIFO get pointer
    LwU32       dwLW_PFIFO_CACHE1_PULL0;            // 0x003250 Cache 1 puller access/status
    LwU32       dwLW_PFIFO_CACHE1_ENGINE;           // 0x003280 Cache 1 FIFO engine status
    LwU32       dwLW_PFIFO_CACHE1[0x100];           // 0x003800 Cache1 FIFO contents
    // PushBuffer State
    LwU32       dwLinearAddrPBC;                    // Linear Address of PB Context
    LwU32       dwLinearAddrPBPTE;                  // Linear Address of PB PTE
    LwU32       dwPhysAddrPB;                       // Physical Address of PB
    LwU32       dwPB[0x40];                         // 256 bytes of Push Buffer
    LwU32       dwDMA_Instance[3];                  // DMA Instance??? Same Size on All
} RmFifoErrorInfo_RECORD, *PRmFifoErrorInfo_RECORD;

#define FIFO_CACHE_SIZE (1024 >> 2)
#define PB_DUMP_SIZE (1024 >> 2)

typedef struct _RmFifoErrorData
{
    // Exception State
    LwU32       dwLW_PFIFO_INTR_0;                  // 0x002100 FIFO interrupt status
    LwU32       dwExceptionData;                    // Exception data
    LwU32       dwLW_PFIFO_MODE;                    // FIFO mode???
    // Chip Specific Information
    LwU32       dwLW_PFIFO_CACHE1_DMA_STATE;        // 0x003228 DMA pusher state
    LwU32       dwLW_PFIFO_CACHE1_DMA_DCOUNT;       // 0x0032A0 DMA transfer count
    LwU32       dwLW_PFIFO_CACHE1_DMA_RSVD_SHADOW;  // 0x0032A8 Last reserved DMA command
    LwU32       dwLW_PFIFO_CACHE1_DMA_DATA_SHADOW;  // 0x0032AC Last DMA data fetched
    LwU32       dwLW_PFIFO_CACHE1_DMA_COUNT;        // 0x003364 Remaining dwords in current method
    // DMA Pusher
    LwU32       dwLW_PFIFO_CACHE1_DMA_PUT;          // 0x003240 Cache 1 DMA put pointer
    LwU32       dwLW_PFIFO_CACHE1_DMA_GET;          // 0x003244 Cache 1 DMA get pointer
    LwU32       dwLW_PFIFO_CACHE1_DMA_REF;          // 0x003248 Cache 1 DMA reference count
    LwU32       dwLW_PFIFO_CACHE1_DMA_INSTANCE;     // 0x00322C Cache 1 DMA PRAMIN address
    LwU32       dwLW_PFIFO_CACHE1_DMA_CTL;          // 0x003230 Cache 1 DMA addr. tran. info.
    LwU32       dwLW_PFIFO_CACHE1_DMA_TLB_PTE;      // 0x00323C Cache 1 DMA PTE TLB info.
    // CACHE1 State
    LwU32       dwLW_PFIFO_CACHE1_PUSH1;            // 0x003204 Cache 1 pusher channel
    LwU32       dwLW_PFIFO_CACHE1_PUT;              // 0x003210 Cache 1 FIFO put pointer
    LwU32       dwLW_PFIFO_CACHE1_GET;              // 0x003270 Cache 1 FIFO get pointer
    LwU32       dwLW_PFIFO_CACHE1_PULL0;            // 0x003250 Cache 1 puller access/status
    LwU32       dwLW_PFIFO_CACHE1_ENGINE;           // 0x003280 Cache 1 FIFO engine status
    LwU32       dwLW_PFIFO_CACHE1[FIFO_CACHE_SIZE];           // 0x003800 Cache1 FIFO contents
    // PushBuffer State
    LwU32       dwPhysAddrPBStart;                  // Physical Address of PB Dump Start
    LwU32       dwPhysAddrPBEnd;                    // Physical Address of PB Dump Start + DMA Get
    LwU32       dwPB[PB_DUMP_SIZE];                        // 1024 bytes of Push Buffer
    LwU32       dwDMA_Instance[3];                  // DMA Instance??? Same Size on All
} RMFIFOERRORDATA, *PRMFIFOERRORDATA;

// Define the RmFifoErrorInfo record structure
typedef struct _RmFifoErrorInfo_RECORD_V2
{
    LWCD_RECORD Header;                 // FIFO error information record header
    GENERIC_DATA_V2 Generic;            // Generic data
    RMFIFOERRORDATA FifoData;           // FifoData
} RmFifoErrorInfo_RECORD_V2, *PRmFifoErrorInfo_RECORD_V2;

typedef struct _RmFifoErrorInfo_RECORD_V3
{
    LWCD_RECORD Header;                 // FIFO error information record header
    GENERIC_DATA_V2 Generic;            // Generic data
    RMFIFOERRORDATA FifoData;           // FifoData
    PERF_INFO_STRUCT PerfInfo;          // Perf Info at time of error
} RmFifoErrorInfo_RECORD_V3, *PRmFifoErrorInfo_RECORD_V3;

typedef struct _RmFifoErrorElement {
    struct _RmFifoErrorElement * pNextError;
    RmFifoErrorInfo_RECORD_V2 RmFifoErrorData;
} RMFIFOERRORELEMENT, * PRMFIFOERRORELEMENT;

typedef struct _RmFifoErrorElement_V2 {
    struct _RmFifoErrorElement * pNextError;
    RmFifoErrorInfo_RECORD_V3 RmFifoErrorData;
} RMFIFOERRORELEMENT_V2, * PRMFIFOERRORELEMENT_V2;

//<Obsolete>
// Define the RmGraphicsErrorInfo record structure
typedef struct _RmGraphicsErrorInfo_RECORD
{
    LWCD_RECORD Header;                 // Graphics error info. record header
    GENERIC_DATA Generic;               // Generic data
    // Exception State
    LwU32       dwLW_PGRAPH_INTR;                   // Interrrupt Status
    LwU32       dwInstance;                         //
    LwU32       dwNsource;                          //
    LwU32       dwOffset;                           //
    LwU8        dwChID;                             // 0x400148 Graphics ChID
    // Chip Specific Information
    LwU32       dwLW_PGRAPH_TRAPPED_DATA_LOW;       // 0x400708
    LwU32       dwLW_PGRAPH_TRAPPED_DATA_HIGH;      // 0x40070C
    LwU32       dwLW_PGRAPH_STATUS;                 // 0x400700 Graphics status
} RmGraphicsErrorInfo_RECORD, *PRmGraphicsErrorInfo_RECORD;

// Define the RmGraphicsErrorInfo record structure
typedef struct _RmGraphicsErrorInfo_RECORD_V2
{
    LWCD_RECORD Header;                 // Graphics error info. record header
    GENERIC_DATA_V2 Generic;            // Generic data
    // Exception State
    LwU32       dwLW_PGRAPH_INTR;                   // Interrrupt Status
    // GR Exception Data
    LwU32       classNum;
    LwU32       NotifyInstance;
    LwU32       Nsource;
    LwU32       Instance;
    LwU32       Offset;
    LwU32       Data;
    LwU32       ChID;
    LwU32       MethodStatus;
    LwU32       ObjectName;                         // Obejct Name
    LwU32       dwLW_PGRAPH_STATUS;                 // 0x400700 Graphics status
    LwU32       FFINTFC_FIFO_0[8];
    LwU32       FFINTFC_FIFO_1[8];
    LwU32       FFINTFC_FIFO_2[8];
    LwU32       FFINTFC_FIFO_PTR;
    LwU32       FFINTFC_ST2;
    LwU32       FFINTFC_ST2_DL;
    LwU32       FFINTFC_ST2_DH;
    RMFIFOERRORDATA FifoData;           // FifoData
} RmGraphicsErrorInfo_RECORD_V2, *PRmGraphicsErrorInfo_RECORD_V2;

typedef struct _RmGraphicsErrorInfo_RECORD_V3
{
    LWCD_RECORD Header;                 // Graphics error info. record header
    GENERIC_DATA_V2 Generic;            // Generic data
    // Exception State
    LwU32       dwLW_PGRAPH_INTR;                   // Interrrupt Status
    // GR Exception Data
    LwU32       classNum;
    LwU32       NotifyInstance;
    LwU32       Nsource;
    LwU32       Instance;
    LwU32       Offset;
    LwU32       Data;
    LwU32       ChID;
    LwU32       MethodStatus;
    LwU32       ObjectName;                         // Obejct Name
    LwU32       dwLW_PGRAPH_STATUS;                 // 0x400700 Graphics status
    LwU32       FFINTFC_FIFO_0[8];
    LwU32       FFINTFC_FIFO_1[8];
    LwU32       FFINTFC_FIFO_2[8];
    LwU32       FFINTFC_FIFO_PTR;
    LwU32       FFINTFC_ST2;
    LwU32       FFINTFC_ST2_DL;
    LwU32       FFINTFC_ST2_DH;
    RMFIFOERRORDATA FifoData;           // FifoData
    PERF_INFO_STRUCT PerfInfo;          // Perf Info at time of error
} RmGraphicsErrorInfo_RECORD_V3, *PRmGraphicsErrorInfo_RECORD_V3;

typedef struct _RmGraphicsErrorElement {
    struct _RmGraphicsErrorElement * pNextError;
    RmGraphicsErrorInfo_RECORD_V2 RmGraphicsErrorData;
} RMGRAPHICSERRORELEMENT, * PRMGRAPHICSERRORELEMENT;

typedef struct _RmGraphicsExtErrorInfo_RECORD
{
    RmGraphicsErrorInfo_RECORD_V2 RmGraphicsErrorData;
    RmGraphicsFormatExtError_RECORD_V2 RmGraphicsFormatExtError;
} RmGraphicsExtErrorInfo_RECORD, *PRmGraphicsExtErrorInfo_RECORD;

typedef struct _RmGraphicsExtErrorInfo_RECORD_V2
{
    RmGraphicsErrorInfo_RECORD_V3 RmGraphicsErrorData;
    RmGraphicsFormatExtError_RECORD_V2 RmGraphicsFormatExtError;
} RmGraphicsExtErrorInfo_RECORD_V2, *PRmGraphicsExtErrorInfo_RECORD_V2;

// Work in Progress
typedef struct _RmGraphicsErrorInfo_RECORD_V4
{
    LWCD_RECORD Header;                 // Graphics error info. record header
    GENERIC_DATA_V2 Generic;            // Generic data
    // Exception State
    LwU32       dwLW_PGRAPH_INTR;                   // Interrrupt Status
    // GR Exception Data
    LwU32       classNum;
    LwU32       NotifyInstance;
    LwU32       ExceptionCode;
    LwU32       Instance;
    LwU32       Offset;
    LwU32       Data;
    LwU32       ChID;
    LwU32       MethodStatus;
    LwU32       ObjectName;                         // Obejct Name
    LwU32       dwLW_PGRAPH_STATUS;                 // 0x400700 Graphics status
    LwU32       FFINTFC_FIFO_0[8];
    LwU32       FFINTFC_FIFO_1[8];
    LwU32       FFINTFC_FIFO_2[8];
    LwU32       FFINTFC_FIFO_PTR;
    LwU32       FFINTFC_ST2;
    LwU32       FFINTFC_ST2_DL;
    LwU32       FFINTFC_ST2_DH;
    RMFIFOERRORDATA FifoData;           // FifoData
    PERF_INFO_STRUCT PerfInfo;          // Perf Info at time of error
    LwU32       ClassError;
} RmGraphicsErrorInfo_RECORD_V4, *PRmGraphicsErrorInfo_RECORD_V4;

// Define the RmGraphicsErrorInfo record structure
typedef struct _RmBusErrorInfo_RECORD
{
    LWCD_RECORD Header;                 // Graphics error info. record header
    GENERIC_DATA_V2 Generic;            // Generic data
    LwU32 DMAInfoA[9];                  // DMA A Info
    LwU32 DMAInfoB[9];                  // DMA B Info
    RMFIFOERRORDATA_V2 FifoData;        // FifoData
} RmBusErrorInfo_RECORD, *PRmBusErrorInfo_RECORD;

// RM_ASSERT info
typedef struct
{
    LWCD_RECORD Header;
    LwU32 *     BreakpointAddrHint;                      // address that can identify bp module
} RmRC2SwRmAssert_RECORD;
typedef RmRC2SwRmAssert_RECORD *PRmRC2SwRmAssert_RECORD;

// DBG_BREAKPOINT_LOGGED - retail logged DBG_BREAKPOINT
// RmGroup/RmRC2SwDbgBreakpoint
typedef struct
{
    LWCD_RECORD Header;
    LwU32 *     BreakpointAddrHint;                     // address that can identify bp module
    LwU32       BreakpointLine;                         // source line # of break point within the module
    LwU32       TimesHit;                               // times this bp was hit
} RmRC2SwDbgBreakpoint_RECORD;
typedef RmRC2SwDbgBreakpoint_RECORD *PRmRC2SwDbgBreakpoint_RECORD;

typedef struct
{
    LWCD_RECORD Header;
    LwU32 *     pGpu;                                   //  Address of GPU that timed out
    LwU32 *     DriverPathId;                           // IP address of caller
} RmRC2GpuTimeout_RECORD;
typedef RmRC2GpuTimeout_RECORD *PRmRC2GpuTimeout_RECORD;

typedef struct _RmRC2SmuCommandInfo_RECORD
{
    LwU8 cmdID;
} RmRC2SmuCommandInfo_RECORD, *PRmRC2SmuCommandInfo_RECORD;


typedef struct _RmRC2PstateInfo_RECORD
{
    LwU32 PstateIndex;
} RmRC2PstateInfo_RECORD, *PRmRC2PstateInfo_RECORD;

typedef struct
{
    LWCD_RECORD Header;                 // Global information record header
    LwU32       dwSize;                 // Size of Journal > 64K
    LwU8        rawData[1];             // Byte stream specified by above

} RmJournal_RECORD;
typedef RmJournal_RECORD *PRmJournal_RECORD;

#endif // _RMCDOBS_H_

