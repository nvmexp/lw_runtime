 /****************************************************************************\
|*                                                                            *|
|*      Copyright 2016-2017 LWPU Corporation.  All rights reserved.         *|
|*                                                                            *|
|*  NOTICE TO USER:                                                           *|
|*                                                                            *|
|*  This source code is subject to LWPU ownership rights under U.S. and     *|
|*  international Copyright laws.                                             *|
|*                                                                            *|
|*  This software and the information contained herein is PROPRIETARY and     *|
|*  CONFIDENTIAL to LWPU and is being provided under the terms and          *|
|*  conditions of a Non-Disclosure Agreement. Any reproduction or             *|
|*  disclosure to any third party without the express written consent of      *|
|*  LWPU is prohibited.                                                     *|
|*                                                                            *|
|*  LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE       *|
|*  CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR           *|
|*  IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH       *|
|*  REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF           *|
|*  MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR            *|
|*  PURPOSE. IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL,              *|
|*  INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES            *|
|*  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN        *|
|*  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING       *|
|*  OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE        *|
|*  CODE.                                                                     *|
|*                                                                            *|
|*  U.S. Government End Users. This source code is a "commercial item"        *|
|*  as that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting         *|
|*  of "commercial computer software" and "commercial computer software       *|
|*  documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)     *|
|*  and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through          *|
|*  227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the         *|
|*  source code with only those rights set forth herein.                      *|
|*                                                                            *|
|*  Module: ocarm.h                                                           *|
|*                                                                            *|
 \****************************************************************************/
#ifndef _OCARM_H
#define _OCARM_H

//******************************************************************************
//
//  oca namespace
//
//******************************************************************************
namespace oca
{

//******************************************************************************
//
//  Constants
//
//******************************************************************************
#define ERROR_TRACE_SIZE        16
#define MAX_FBBAS               0x2
#define FBBA_MUX_SEL_MAX        0xF
#define FBBA_RXB_STATUS_SEL_MAX 0x10
#define RDSTAT_MAX              0x2

//
// Generic mini-record type (keep the size at 64bits)
//
typedef struct
{
    LWCD_RECORD     Header;                     // header for mini record
    DWORD           Payload;                    // 32 bit payload value

} LWCDMiniRecord, *PLWCDMiniRecord;

//
// Generic record collection type
// 
typedef struct
{
    LWCD_RECORD     Header;                     // generic header to binary type this in OCA buffer
                                                // size is actual size of this struct + all items in collection
    DWORD           NumRecords;                 // number of records this collection contain
    LWCD_RECORD     FirstRecord;                // first record, its data follow

} LWCDRecordCollection, *PLWCDRecordCollection;

typedef struct LwNotificationRec
{
    struct                                      /*                                   0000-    */
    {
        LwU32           nanoseconds[2];         /* nanoseconds since Jan. 1, 1970       0-   7*/
    } timeStamp;                                /*                                       -0007*/
    LwV32               info32;                 /* info returned depends on method   0008-000b*/
    LwV16               info16;                 /* info returned depends on method   000c-000d*/
    LwV16               status;                 /* user sets bit 15, LW sets status  000e-000f*/

} LwNotification, *PLwNotification;

//******************************************************************************
//
// Resman OCA Record Structures
//
//******************************************************************************
// Define the resource manager group record types
typedef enum
{
    RmGlobalInfo                            = 0,    // RM global info. record
    RmLwrrentStateInfo                      = 1,    // RM current state info. record
    RmFifoErrorInfo                         = 2,    // RM FIFO error information record
    RmGraphicsErrorInfo                     = 3,    // RM graphics error information record
    RmLwrrentStateInfo_V2                   = 4,    // Revised Version 2 RM current state info. record
    RmFifoErrorInfo_V2                      = 5,    // Revised Version 2 RM FIFO error info. record
    RmGraphicsErrorInfo_V2                  = 6,    // Revised Version 3 RM graphics error info. record
    RmGlobalInfo_V2                         = 7,    // Revised Version 2 RM global info. record
    RmBusErrorInfo                          = 8,    // RM Bus Error Info
    RmMissedNotifierInfo                    = 9,    // RM Driver Indicates Missed Notifier 
    RmGlobalInfo_V3                         = 10,   // Revised Version 3 RM Global Info
    RmFifoErrorInfo_V3                      = 11,   // Revised Version 3 RM FIFO error info. record
    RmGraphicsErrorInfo_V3                  = 12,   // Revised Version 3 RM graphics error info. record
    RmBspErrorInfo                          = 13,   // RM BSP Error Info

    // RC2 style 
    RmRC2GlobalCollectionRec                = 15,   // Generic collection for global data
    RmRC2CrashStateCollectionRec            = 16,   // Generic collection for crash state data
    RmRC2RcErrorCollectionRec               = 17,   // Generic collection for RC error record
    RmRC2HwExceptionCollectionRec           = 18,   // Generic collection for hw exception record 
    RmRC2SwExceptionCollectionRec           = 19,   // Generic collection for sw exception record
    RmRC2HwPriTimeoutCollectionRec          = 20,   // PRI_TIMEOUT intr logged collection (tmr read + SAVE regs)
    RmRC2HwBusErrorCollectionRec            = 21,   // Bus (PEX or AGP) error collection
    RmRC2HwTmrErrorCollectionRec            = 22,   // PTIMER bad value read error
    RmRC2HwThermalEventCollectionRec        = 23,   // Thermal event collection record
    RmRC2SaveRestoreFBRetryCollectionRec    = 24,   // SaveRestoreFB retry record (mem test after mclk programming)

    RmRC2GenericCounter                     = 30,   // counter value that may be attached to some records
    RmRC2SwDbgBreakpoint                    = 31,   // Retail logged breakpoint info
    RmRC2SwRmAssert                         = 32,   // Retail logged RM_ASSERT info
    RmRC2GpuTimeout                         = 33,   // Retail logged RM_TIMEOUT events
    RmRC2TempReading                        = 34,   // GPU temp reading
    RmRC2RequestedGpuSignals                = 35,   // Captures requested gpu signals

    // short info records
    RmRC2GpuShortInfo                       = 40,   // GPU short info
    RmRC2BiosShortInfo                      = 41,   // Video Bios short info (version)

    // agp and pci short info
    RmRC2AgpShortInfo                       = 45,   // AGP short info
    RmRC2ShortPci30CfgInfo                  = 46,   // PCI 3.0 config space data
    RmRC2PexCapsStateInfo                   = 47,   // PCI-E general caps/state info
    RmRC2PexAERCapsStateInfo                = 48,   // PCI-E AER (advanced erro reporting) caps/state info
    RmRC2PexVCCapsStateInfo                 = 49,   // PCI-E VC (virtual channel) caps/state info

    // GPU engine HAL records (generated in .list file by rc/lw/refoca.pl)
    RmRC2GpuEnginePBUS_HAL                  = 70,
    RmRC2GpuEnginePFB_HAL                   = 71,
    RmRC2GpuEnginePFIFO_HAL                 = 72,
    RmRC2GpuEnginePGRAPH_HAL                = 73,
    RmRC2GpuEnginePHOST_HAL                 = 74,
    RmRC2GpuEnginePMC_HAL                   = 75,
    RmRC2GpuEnginePTIMER_HAL                = 77,
    // sub unit (split) records
    RmRC2GpuEngineUnitPBUS_DEBUG_HAL        = 90,
    RmRC2GpuEngineUnitPBUS_PCI_HAL          = 91,
    RmRC2GpuEngineUnitPBUS_THERMALCTRL_HAL  = 92,
    RmRC2GpuEngineUnitPBUS_UMA_HAL          = 93,
    RmRC2GpuEngineUnitPFB_GART_HAL          = 94,
    RmRC2GpuEngineUnitPFB_GTLB_HAL          = 95,
    RmRC2GpuEngineUnitPFIFO_C1DUMP_HAL      = 96,
    RmRC2GpuEngineUnitPFIFO_CHANNEL_HAL     = 97,
    RmRC2GpuEngineUnitPFIFO_DEVICE_HAL      = 98,
    RmRC2GpuEngineUnitPGRAPH_FFINTFC_HAL    = 99,
    RmRC2GpuEngineUnitPGRAPH_SEMAPHORE_HAL  = 100,
    RmRC2GpuEngineUnitPTIMER_PRITO_HAL      = 101,
    RmRC2GpuEngineUnitPTIMER_TIME_HAL       = 102,
    RmGraphicsErrorInfo_V4                  = 103,  // Revised Version 4 RM graphics error info. record
    RmErrorBlock                            = 104,  // Error Block

    RmFifoErrorInfo_V4                      = 105,  // Revised Version 4 of FIFO Error Info
    RmGraphicsErrorInfo_V5                  = 106,  // Revised Version 5 of Graphics Error Info
    RmGraphicsErrorInfo_V6                  = 107,  // Revised Version 6 of Graphics Error Info
    RmBusErrorInfo_V2                       = 108,  // Revised Version 2 of Bus Error Info
    RmGraphicsErrorInfo_V7                  = 110,  // Revised Version 7 of Graphics Error Info
    RmSafeGlobalInfo                        = 111,  // RM Global Info
    RmJournalInfo                           = 112,  // RM Journal Info
    RmRingBuffer                            = 113,  // Ring Buffer                                      
    RmRC2SmuCommandInfo                     = 114,  // Logs the communication with SMU.
    RmRC2PstateInfo                         = 115,
    RmVpErrorInfo                           = 116,
    RmVbiosInfo_V2                          = 117,  // Vbios Info Roll for header size issue
    RmRC2SmuErrorInfo                       = 119,  // Logs the errors encountered when communicating with SMU
    RmRC2SwRmAssert_V2                      = 120,  // Retail logged RM_ASSERT info
    RmRC2GpuTimeout_V2                      = 121,  // Retail logged RM_TIMEOUT events
    RmRC2SmuCommandInfo_V2                  = 122,  // Logs the communication with SMU.
    RmRC2PstateInfo_V2                      = 123,
    RmRC2SwDbgBreakpoint_V2                 = 124,  // Uses same format as RmRC2SwRmAssert_V2
    RmPrbErrorInfo                          = 126,  // Protobuf error record
    RmElpgInfo                              = 127,
    RmBadRead                               = 128,  // Record with Bad Read Information
    RmJournalInfo_V3                        = 129,  // Rm Journal for header size issue
    RmSurpriseRemoval                       = 130,  // Surprise Removal
    RmProtoBuf                              = 131,  // ProtoBuf
    RmProtoBuf_V2                           = 132,  // ProtoBuf + LwDump
    RmDclMsg                                = 133,  // One of the optional DlcMsg fields, encoded
    RmJournalEngDump                        = 134,
    DP_ASSERT_HIT                           = 135,
    DP_LOG_CALL                             = 136,
    RmPrbFullDump                           = 137,  // Full LwDump protobuf message

} RMCD_RECORD_TYPE;

typedef struct
{
    LWCD_RECORD         Header;                 // Global information record header
    LwU32               dwSize;                 // Total Protobuf Message Size  

}  RmProtoBuf_RECORD, *PRmProtoBuf_RECORD;

//******************************************************************************
//
//  class CRmProtoBufRecord
//
//******************************************************************************
class CRmProtoBufRecord : public CLwcdRecord
{
// RmProtoBuf_RECORD Type Helpers
TYPE(rmProtoBufRecord)

// RmProtoBuf_RECORD Field Helpers
FIELD(Header)
FIELD(dwSize)

// RmProtoBuf_RECORD Members
MEMBER(dwSize,  DWORD,  0,  public)

private:
        const void*     m_pRmProtoBufRecord;
        const COcaData* m_pOcaData;

public:
                        CRmProtoBufRecord(const CLwcdRecord* pLwcdRecord, const COcaData* pOcaData, ULONG ulRemaining);
virtual                ~CRmProtoBufRecord();

const   void*           rmProtoBufRecord() const    { return m_pRmProtoBufRecord; }
const   COcaData*       ocaData() const             { return m_pOcaData; }

virtual ULONG           size() const;

const   CMemberType&    type() const                { return m_rmProtoBufRecordType; }

}; // CRmProtoBufRecord

typedef struct
{
    LWCD_RECORD         Header;                 // Global information record header
    LwU32               dwPCI_ID;               // Device PCI vendor/device ID value
    LwU32               dwSubSys_ID;            // Subsystem vendor/device ID value
    LwU32               dwChipRevision;         // Chip revision value (PMC Boot 0)
    LwU32               dwStrap;                // Chip boot strapping value
    LwU32               dwNorthBridgeId;        // AGP NB vendor/device ID value
    LwU32               dwBiosRevision;         // BIOS revision value
    LwU32               dwBiosOEMRevision;      // OEM BIOS revision value
    LwU32               dwErrorCount;           // Number of error records to follow
    LwU16               wLW_AgpConf_Status;     // LW AGP configuration status value
    LwU16               wNB_AgpConf_Status;     // NB AGP configuration status value
    LwU32               dwCPU_Type;             // CPU type value (cpuInfo.type)
    LwU32               dwCPU_Caps;             // CPU caps value (cpuInfo.caps)
    LwU8                bNumPhysicalCPUs;       // Number of Physical CPUs
    LwU8                bNumLogicalCPUs;        // Total number of Logical CPUs
    LwU16               wTotalError;            // 0xFFFF means at least 0xFFFF possible more
    LwU8                LastErrors[ERROR_TRACE_SIZE]; // Last 16 Errors
    LwU32               Flags;                  // LW_PBUS_PIPE config, SLI, Multimon

} RmGlobalInfo_RECORD_V3, *PRmGlobalInfo_RECORD_V3;

// Keep these flags the same so that we can use common code
#define PERF_INFO_FLAGS_OVERCLOCK_MCLK  GLOBALINFO_FLAGS_OVERCLOCK_MCLK
#define PERF_INFO_FLAGS_OVERCLOCK_G     GLOBALINFO_FLAGS_OVERCLOCK_G 
#define PERF_INFO_FLAGS_OVERCLOCK_S     GLOBALINFO_FLAGS_OVERCLOCK_S 
#define PERF_INFO_FLAGS_OVERCLOCK_R     GLOBALINFO_FLAGS_OVERCLOCK_R 

typedef struct
{
    LwU32               Voltage;                // Can I reduce this?
    LwU32               McClk;                  // In 10 KHz Units
    LwU32               LwClkG;                 // In 10 KHz Units
    LwU32               LwClkS;                 // In 10 KHz Units
    LwU32               LwClkR;                 // In 10 KHz Units
    LwU8                LocTemp; 
    LwU8                GpuTemp; 
    LwU8                PerfLevel;
    LwU8                Flags;
    LwU8                LwrrentPstate;          // 0=>undefined, 1=>P0, 2=>P1, 4=>P2, 0x8=>P3, 0x100=P8, 0x400=>P10, 0x1000=P12, 0x8000=>P15
    LwBool              ThermSlowdownState;
    LwU8                ThermPerfCapLevel;      // 0=>no, >0=>yes and level
    LwU8                ThermSlowdownLevel;     // in %; 100=>no slowdown, 0=>complete slowdown
    LwBool              ThermAlertGPIOState;
    LwBool              ThermOvertGPIOState;
    LwBool              ThermOnlyGPIOState;
    LwBool              PowerAlertGPIOState;    // 0 = on, 1 = off
    LwU8                FanControl;             // 0=> none, 1=>hw, 2=>sw
    LwU8                FanLevelPWM;            // 0 = OFF,  > 0 => ON and in %
    LwU8                PowerAlertCapLevel;     // 0 is no, >0 is yes and level
    LwU32               reserved;
    LwU32               reserved1;

} PERF_INFO_STRUCT, *PPERF_INFO_STRUCT;

// Generic error data Version 2 (Include in both FIFO and graphics error records)
// Minor Changes -- 
typedef struct 
{
    LwU16               wLW_AgpConf_Cmd;        // LW AGP configuration at error
    LwU16               wNB_AgpConf_Cmd;        // NB AGP configuration at error
    LwHandle            hErrorContext;          // Handle to Error Context
    LwU8                cChannelID;             // Channel ID <0-31>
    LwU8                cErrorType;             // Error Type <
    LwU16               wPushBufferSpace;       // Push buffer space
    LwU32               dwTimeHi;               // Error time (High word) <OS Specific>
    LwU32               dwTimeLo;               // Error time (Low word)  <OS Specific>

} GENERIC_DATA_V2, *PGENERIC_DATA_V2;

// Define the RmLwrrentStateInfo record structure
typedef struct 
{
    LWCD_RECORD         Header;                         // Current state info. record header
    GENERIC_DATA_V2     Generic;                        // Generic data
    // PCI
    LwU32               dwLW_PBUS_PCI_BAR0;             // 0x001810 PCI Bar 0 value
    LwU32               dwLW_PBUS_PCI_BAR1;             // 0x001814 PCI Bar 1 value
    // Fifo
    LwU32               dwLW_PFIFO_INTR_0;              // 0x002100 FIFO interrupt status
    LwU32               dwLW_PFIFO_INTR_EN_0;           // 0x002140 FIFO interrupt enables
    LwU32               dwLW_PFIFO_CACHES;              // 0x002500 Cache settings
    LwU32               dwLW_PFIFO_CACHE1_PUSH0;        // 0x003200 Cache 1 pusher access
    LwU32               dwLW_PFIFO_CACHE1_PUSH1;        // 0x003204 Cache 1 pusher channel
    LwU32               dwLW_PFIFO_CACHE1_PULL0;        // 0x003250 Cache 1 puller access/status
    LwU32               dwLW_PFIFO_CACHE1_PUT;          // 0x003210 Cache 1 FIFO put pointer
    LwU32               dwLW_PFIFO_CACHE1_GET;          // 0x003270 Cache 1 FIFO get pointer
    // Master Controller
    LwU32               dwLW_PMC_INTR_0;                // 0x000100 Master interrupt status
    LwU32               dwLW_PMC_INTR_EN_0;             // 0x000140 Master interrupt enable
    LwU32               dwLW_PMC_INTR_READ_0;           // 0x000160 Interrupt pin status
    LwU32               dwLW_PMC_ENABLE;                // 0x000200 Engine enables
    // Graphics Status
    LwU32               dwLW_PGRAPH_STATUS;             // 0x400700 Graphics engine status
    LwU32               dwLW_PGRAPH_INTR;               // 0x400100 Graphics engine interrrupt status
    LwU32               dwLW_PGRAPH_CTX_CONTROL;        // 0x400144 Context switch control
    LwU32               dwLW_PGRAPH_CTX_USER;           // 0x400148 User address information
    LwU32               dwLW_PGRAPH_TRAPPED_ADDR;       // 0x400704 Last trapped address (Method/Sub/Chan)
    LwU32               dwLW_PGRAPH_TRAPPED_DATA_LOW;   // 0x400708 Last trapped data value (Low)
    LwU32               dwLW_PGRAPH_TRAPPED_DATA_HIGH;  // 0x40070C Last trapped data value (High)

} RmLwrrentStateInfo_RECORD_V2, *PRmLwrrentStateInfo_RECORD_V2;

typedef struct 
{
    // Exception State
    LwU32               dwLW_PFIFO_INTR_0;                  // 0x002100 FIFO interrupt status
    LwU32               dwExceptionData;                    // Exception data
    LwU32               dwLW_PFIFO_MODE;                    // FIFO mode???
    // Chip Specific Information
    LwU32               dwLW_PFIFO_CACHE1_DMA_STATE;        // 0x003228 DMA pusher state
    LwU32               dwLW_PFIFO_CACHE1_DMA_DCOUNT;       // 0x0032A0 DMA transfer count
    LwU32               dwLW_PFIFO_CACHE1_DMA_RSVD_SHADOW;  // 0x0032A8 Last reserved DMA command
    LwU32               dwLW_PFIFO_CACHE1_DMA_DATA_SHADOW;  // 0x0032AC Last DMA data fetched
    LwU32               dwLW_PFIFO_CACHE1_DMA_COUNT;        // 0x003364 Remaining dwords in current method
    // DMA Pusher
    LwU32               dwLW_PFIFO_CACHE1_DMA_PUT;          // 0x003240 Cache 1 DMA put pointer
    LwU32               dwLW_PFIFO_CACHE1_DMA_GET;          // 0x003244 Cache 1 DMA get pointer
    LwU32               dwLW_PFIFO_CACHE1_DMA_REF;          // 0x003248 Cache 1 DMA reference count
    LwU32               dwLW_PFIFO_CACHE1_DMA_INSTANCE;     // 0x00322C Cache 1 DMA PRAMIN address
    LwU32               dwLW_PFIFO_CACHE1_DMA_CTL;          // 0x003230 Cache 1 DMA addr. tran. info.
    LwU32               dwLW_PFIFO_CACHE1_DMA_TLB_PTE;      // 0x00323C Cache 1 DMA PTE TLB info.
    // CACHE1 State
    LwU32               dwLW_PFIFO_CACHE1_PUSH1;            // 0x003204 Cache 1 pusher channel
    LwU32               dwLW_PFIFO_CACHE1_PUT;              // 0x003210 Cache 1 FIFO put pointer
    LwU32               dwLW_PFIFO_CACHE1_GET;              // 0x003270 Cache 1 FIFO get pointer
    LwU32               dwLW_PFIFO_CACHE1_PULL0;            // 0x003250 Cache 1 puller access/status
    LwU32               dwLW_PFIFO_CACHE1_ENGINE;           // 0x003280 Cache 1 FIFO engine status
    LwU32               dwDMA_Instance[3];                  // DMA Instance??? Same Size on All

} RMFIFOERRORDATA_V2, *PRMFIFOERRORDATA_V2;

typedef struct
{
    LWCD_RECORD         Header;                 // FIFO error information record header
    GENERIC_DATA_V2     Generic;                // Generic data
    RMFIFOERRORDATA_V2  FifoData;               // FifoData   
    PERF_INFO_STRUCT    PerfInfo;               // Perf Info at time of error

} RmFifoErrorInfo_RECORD_V4, *PRmFifoErrorInfo_RECORD_V4;

typedef struct
{
    LWCD_RECORD         Header;                 // Protobuf data

} RmPrbInfo_RECORD_V1, *PRmPrbInfo_RECORD_V1;

typedef enum
{
    RM_FORMAT_ERROR     = 0,                    // Format Exception Special Data
    RM_FORMAT_ERROR_V2  = 1                     // Format Exception Special Data

} RmGraphicsSpecialInfoType;

typedef struct
{
    RmGraphicsSpecialInfoType SpecialInfo;
    LwU32               BPIXEL;
    LwU32               CTX_CACHE1[8];

} RmGraphicsFormatExtError_RECORD, *PRmGraphicsFormatExtError_RECORD;

typedef struct
{
    RmGraphicsSpecialInfoType SpecialInfo;
    LwU32               BPIXEL;
    LwU32               CTX_SWITCH1;            // Context Switch 1
    LwU32               Gr_Ctx[4];              // Context Save Area

} RmGraphicsFormatExtError_RECORD_V2, *PRmGraphicsFormatExtError_RECORD_V2;

typedef struct 
{
    LWCD_RECORD         Header;                 // Graphics error info. record header
    GENERIC_DATA_V2     Generic;                // Generic data
    // Exception State
    LwU32               dwLW_PGRAPH_INTR;       // Interrrupt Status
    // GR Exception Data
    LwU32               classNum;
    LwU32               NotifyInstance;
    LwU32               Nsource;
    LwU32               Instance;
    LwU32               Offset;
    LwU32               Data;
    LwU32               ChID;
    LwU32               MethodStatus;
    LwU32               ObjectName;             // Object Name
    LwU32               dwLW_PGRAPH_STATUS;     // 0x400700 Graphics status
    LwU32               FFINTFC_FIFO_0[8];                              
    LwU32               FFINTFC_FIFO_1[8];                             
    LwU32               FFINTFC_FIFO_2[8];
    LwU32               FFINTFC_FIFO_PTR;
    LwU32               FFINTFC_ST2;
    LwU32               FFINTFC_ST2_DL;
    LwU32               FFINTFC_ST2_DH;
    RMFIFOERRORDATA_V2  FifoData;               // FifoData

} RmGraphicsErrorInfo_RECORD_V5, *PRmGraphicsErrorInfo_RECORD_V5;

typedef struct 
{
    LWCD_RECORD         Header;                 // Graphics error info. record header
    GENERIC_DATA_V2     Generic;                // Generic data
    // Exception State
    LwU32               dwLW_PGRAPH_INTR;       // Interrrupt Status
    // GR Exception Data
    LwU32               classNum;
    LwU32               NotifyInstance;
    LwU32               Nsource;
    LwU32               Instance;
    LwU32               Offset;
    LwU32               Data;
    LwU32               ChID;
    LwU32               MethodStatus;
    LwU32               ObjectName;             // Object Name
    LwU32               dwLW_PGRAPH_STATUS;     // 0x400700 Graphics status
    LwU32               FFINTFC_FIFO_0[8];                              
    LwU32               FFINTFC_FIFO_1[8];                             
    LwU32               FFINTFC_FIFO_2[8];
    LwU32               FFINTFC_FIFO_PTR;
    LwU32               FFINTFC_ST2;
    LwU32               FFINTFC_ST2_DL;
    LwU32               FFINTFC_ST2_DH;
    RMFIFOERRORDATA_V2  FifoData;               // FifoData   
    PERF_INFO_STRUCT    PerfInfo;               // Perf Info at time of error

} RmGraphicsErrorInfo_RECORD_V6, *PRmGraphicsErrorInfo_RECORD_V6;

typedef struct
{
    RmGraphicsErrorInfo_RECORD_V5       RmGraphicsErrorData;
    RmGraphicsFormatExtError_RECORD_V2  RmGraphicsFormatExtError;    

} RmGraphicsExtErrorInfo_RECORD_V3, *PRmGraphicsExtErrorInfo_RECORD_V3;

typedef struct
{
    RmGraphicsErrorInfo_RECORD_V6       RmGraphicsErrorData;
    RmGraphicsFormatExtError_RECORD_V2  RmGraphicsFormatExtError;    

} RmGraphicsExtErrorInfo_RECORD_V4, *PRmGraphicsExtErrorInfo_RECORD_V4;

typedef struct
{
    LWCD_RECORD         Header;                 // Graphics error info. record header
    GENERIC_DATA_V2     Generic;                // Generic data
    // Exception State
    LwU32               dwLW_PGRAPH_INTR;       // Interrrupt Status
    // GR Exception Data
    LwU32               classNum;
    LwU32               NotifyInstance;
    LwU32               ExceptionCode;
    LwU32               Instance;
    LwU32               Offset;
    LwU32               Data;
    LwU32               ChID;
    LwU32               MethodStatus;
    LwU32               ObjectName;             // Object Name
    LwU32               dwLW_PGRAPH_STATUS;     // 0x400700 Graphics status
    LwU32               FFINTFC_FIFO_0[8];                              
    LwU32               FFINTFC_FIFO_1[8];                             
    LwU32               FFINTFC_FIFO_2[8];
    LwU32               FFINTFC_FIFO_PTR;
    LwU32               FFINTFC_ST2;
    LwU32               FFINTFC_ST2_DL;
    LwU32               FFINTFC_ST2_DH;
    RMFIFOERRORDATA_V2  FifoData;               // FifoData   
    PERF_INFO_STRUCT    PerfInfo;               // Perf Info at time of error
    LwU32               ClassError;

} RmGraphicsErrorInfo_RECORD_V7, *PRmGraphicsErrorInfo_RECORD_V7;

// Define the RmGraphicsErrorInfo record structure
typedef struct
{
    LWCD_RECORD         Header;                 // Graphics error info. record header
    GENERIC_DATA_V2     Generic;                // Generic data
    LwU32               DMAInfoA[9];            // DMA A Info    
    LwU32               DMAInfoB[9];            // DMA B Info    
    RMFIFOERRORDATA_V2  FifoData;               // FifoData

} RmBusErrorInfo_RECORD_V2, *PRmBusErrorInfo_RECORD_V2;

typedef struct
{
    LWCD_RECORD         Header;                 // Graphics error info. record header
    GENERIC_DATA_V2     Generic;                // Generic data
    LwHandle            hClient;
    LwHandle            hDevice;
    LwU32               lwDispDmaLastFree;
    LwHandle            hDispChannel;           // event_log
    LwHandle            hLastChannel;           // event_log
    LwU32               lwDispDmaCount;         // event_log
    LwU32               lwDispDmaCachedGet;     // event_log
    LwU32               lwDispDmaCachedPut;     // event_log
    LwNotification      Notifier;
    LwU32               notifierObject;
    LwU32               notifierClass;
    LwU32               notifierStatus;

} RmMissedNotifierInfo_RECORD, *PRmMissedNotifierInfo_RECORD;

// Define the RmBSPErrorInfo record structure
typedef struct RmBspErrorInfo_RECORD
{
    LWCD_RECORD         Header;                 // BSP error info. record header
    GENERIC_DATA_V2     Generic;                // Generic data
    LwU32               Method;
    LwU32               Data;
    LwU32               intrStatus;
    LwU32               lwrrentContext;
    LwU32               haltStatus;
    
} RmBspErrorInfo_RECORD, *PRmBspErrorInfo_RECORD;

typedef struct
{
    LWCD_RECORD         Header;                 // BSP error info. record header
    GENERIC_DATA_V2     Generic;                // Generic data
    LwU32               Method;
    LwU32               Data;
    LwU32               eventsPending;
    LwU32               intrStatus;
    LwU32               lwrrentContext;
    LwU32               haltStatus;
    
} RmVpErrorInfo_RECORD, *PRmVpErrorInfo_RECORD;

// ****************************************************************************
//
// New style (collection) records
//
typedef LWCDRecordCollection RmRC2GlobalInfo_RECORD;    // global info data
typedef LWCDRecordCollection RmRC2CrashStateInfo_RECORD;// info state data at crash
typedef LWCDRecordCollection RmRC2RcErrorRec_RECORD;    // RC error record 

// PCI config space info (pci ids, bars, caps)
// RmGroup/RmRC2ShortPci30CfgInfo
typedef struct 
{
    LWCD_RECORD         Header;
    LwU16               VenID;                  // pci vendor id (=10DE)
    LwU16               DevID;                  // pci device id
    LwU16               PciCommand;             // pci command (offset=4)
    LwU16               PciStatus;              // pci status (offset=6)
    LwU32               RevClass;               // pci rev and class (offset=8)
    LwU32               Other;                  // other pci info (cache line, etc) (offset=C)
    LwU32               BAR0;                   // pci BAR0 (base addr register) (offset=10)
    LwU32               BAR1;                   // pci BAR1 (base addr register) (offset=14)
    LwU32               BAR2;                   // pci BAR2 (base addr register) (offset=18)
    LwU32               BAR3;                   // pci BAR3 (base addr register) (offset=1C)
    LwU32               BAR4;                   // pci BAR4 (base addr register) (offset=20)
    LwU32               BAR5;                   // pci BAR5 (base addr register) (offset=24)
    LwU16               SubsysVenID;            // pci subsystem vendor id (offset=2C)
    LwU16               SubsysDevID;            // pci subsystem device id (offset=2E)

} RmRC2ShortPci30CfgInfo_RECORD, *PRmRC2ShortPci30CfgInfo_RECORD;

// GPU global info (pmc_boot_0, straps)
// RmGroup/RmRC2GpuShortInfo
typedef struct 
{
    LWCD_RECORD         Header;
    LwU32               Revision;               // chip revision        PMC_BOOT_0
    LwU32               Strap0;                 // strap0               LW_PEXTDEV_BOOT_0
    LwU32               Strap0AndMask;          // strap0 and mask      LW_PEXTDEV_BOOT_1
    LwU32               Strap0OrMask;           // strap0 or mask       LW_PEXTDEV_BOOT_2
    LwU32               Strap1;                 // strap1               LW_PEXTDEV_BOOT_3
    LwU32               Strap1AndMask;          // strap1 and mask      LW_PEXTDEV_BOOT_4
    LwU32               Strap1OrMask;           // strap1 or mask       LW_PEXTDEV_BOOT_5

} RmRC2GpuShortInfo_RECORD, *PRmRC2GpuShortInfo_RECORD;

// GPU global info (pmc_boot_0, straps)
// RmGroup/RmRC2BiosShortInfo
typedef struct 
{
    LWCD_RECORD         Header;
    LwU32               BiosVersion;            // Vbios version
    LwU32               BiosOemVersion;         // Vbios OEM version
    LwU32               BiosRmDword;            // Bios Rm dword

} RmRC2BiosShortInfo_RECORD, *PRmRC2BiosShortInfo_RECORD;

// AGP short info
// RmGroup/RmRC2AgpShortInfo
typedef struct 
{
    LWCD_RECORD         Header;
    LwU32               northBridgeId;          // AGP north bridge dev/ven ID
    LwU32               NBAgpStatus;            // NB AGP status register
    LwU32               NBAgpCommand;           // NB AGP command register
    LwU32               LWAgpStatus;            // LW AGP status register
    LwU32               LWAgpCommand;           // LW AGP command register

} RmRC2AgpShortInfo_RECORD, *PRmRC2AgpShortInfo_RECORD;

typedef struct 
{
    LWCD_RECORD         Header;
    LwU32               Counter;

} RmRC2GenericCounter_RECORD, *PRmRC2GenericCounter_RECORD;

// RM_ASSERT + DBG_BREAKPOINT info
typedef struct 
{
    LWCD_RECORD         Header;
    LwU64               FirstTime LW_ALIGN_BYTES(8);
    LwU64               LastTime LW_ALIGN_BYTES(8);
    LwU64               BreakpointAddrHint LW_ALIGN_BYTES(8);   // address that can identify bp module
    LwU64               CallStack[10] LW_ALIGN_BYTES(8);        // Call stack when the assert oclwrred.
    LwU32               GPUTag;
    LwU32               Count;

} RmRCCommonAssert_RECORD, *PRmRCCommonAssert_RECORD;

// How Serious is this RM_ASSERT/DBG_BREAKPOINT
// (1) Info -- This is unexpected but we should continue to run 
// (2) Error -- 
// (3) Fatal -- This is hopeless -- FBI Timeout, Bus Error Etc
#define LW_RM_ASSERT_TYPE                           3:0
#define LW_RM_ASSERT_TYPE_INFO                      0x00000001
#define LW_RM_ASSERT_TYPE_ERROR                     0x00000002
#define LW_RM_ASSERT_TYPE_FATAL                     0x00000003

// HW Unit which is having the issue
#define LW_RM_ASSERT_HW_UNIT                        15:8
#define LW_RM_ASSERT_HW_UNIT_NULL                   (0x00)
#define LW_RM_ASSERT_HW_UNIT_GRAPHICS               (0x01)
#define LW_RM_ASSERT_HW_UNIT_COPY0                  (0x02)
#define LW_RM_ASSERT_HW_UNIT_COPY1                  (0x03)
#define LW_RM_ASSERT_HW_UNIT_VP                     (0x04)
#define LW_RM_ASSERT_HW_UNIT_ME                     (0x05)
#define LW_RM_ASSERT_HW_UNIT_PPP                    (0x06)
#define LW_RM_ASSERT_HW_UNIT_BSP                    (0x07)
#define LW_RM_ASSERT_HW_UNIT_MPEG                   (0x08)
#define LW_RM_ASSERT_HW_UNIT_SW                     (0x09)
#define LW_RM_ASSERT_HW_UNIT_CIPHER                 (0x0a)
#define LW_RM_ASSERT_HW_UNIT_VIC                    (0x0b)
#define LW_RM_ASSERT_HW_UNIT_MSENC                  (0x0c)
#define LW_RM_ASSERT_HW_UNIT_LWENC0                 LW_RM_ASSERT_HW_UNIT_MSENC
#define LW_RM_ASSERT_HW_UNIT_LWENC1                 (0x0d)
#define LW_RM_ASSERT_HW_UNIT_HOST                   (0x0e)
#define LW_RM_ASSERT_HW_UNIT_ROM                    (0x0f)
#define LW_RM_ASSERT_HW_UNIT_INSTMEM                (0x10)
#define LW_RM_ASSERT_HW_UNIT_DISP                   (0x11)
#define LW_RM_ASSERT_HW_UNIT_LWENC2                 (0x12)
#define LW_RM_ASSERT_HW_UNIT_ALLENGINES             (0xff)
// SW Module which generated the error
#define LW_RM_ASSERT_SW_MODULE                      15:8

// This is a specific error number which we wish to follow regardless of builds
// We want to use this for backend processing 
// This is also a compromise.  Ideally, each event would have a unique id but 
// instead of doing this we use EIP which is unique per load.  If we subtracted off
// the Module Load Address then it would be unique per build,  Using EIP allows us
// to use the debugger to lookup the source code that corresponds to the event.
#define LW_RM_ASSERT_LEVEL_TAG                      30:16
// Host Errors
#define LW_RM_ASSERT_LEVEL_TAG_BAR1_PAGE_FAULT      (0x0001)
#define LW_RM_ASSERT_LEVEL_TAG_IFB_PAGE_FAULT       (0x0002)
#define LW_RM_ASSERT_LEVEL_TAG_PIO_ERROR            (0x0003)
#define LW_RM_ASSERT_LEVEL_TAG_CHSW_SAVE_ILWALID    (0x0004)
#define LW_RM_ASSERT_LEVEL_TAG_CHSW_ERROR           (0x0005)
#define LW_RM_ASSERT_LEVEL_TAG_FBIRD_TIMEOUT        (0x0006)
#define LW_RM_ASSERT_LEVEL_TAG_CPUQ_FBIBUSY_TIMEOUT (0x0007)
#define LW_RM_ASSERT_LEVEL_TAG_CHSW_FSM_TIMEOUT     (0x0008)
#define LW_RM_ASSERT_LEVEL_TAG_FB_FLUSH_TIMEOUT     (0x0009)
#define LW_RM_ASSERT_LEVEL_TAG_P2PSTATE_TIMEOUT     (0x000a)
#define LW_RM_ASSERT_LEVEL_TAG_VBIOS_CHECKSUM       (0x000b)
#define LW_RM_ASSERT_LEVEL_TAG_DISP_SYNC            (0x000c)

// What is the generating Source -- GPU or SW
#define LW_RM_ASSERT_LEVEL_SOURCE                   31:31
#define LW_RM_ASSERT_LEVEL_SOURCE_SW                0x00000000
#define LW_RM_ASSERT_LEVEL_SOURCE_HW                0x00000001


// RM_ASSERT + DBG_BREAKPOINT info
typedef struct 
{
    RmRCCommonAssert_RECORD Common;
    LwU32               Level;

} RmRC2SwRmAssert2_RECORD, *PRmRC2SwRmAssert2_RECORD;

typedef struct 
{
    LWCD_RECORD         Header;
    LwS32               GpuTemp;
    LwS32               LocTemp;

} RmRC2TempReading_RECORD, *PRmRC2TempReading_RECORD;

typedef enum
{
    MEMORY_BAR0 = 1,
    MEMORY_FB,
    MEMORY_INSTANCE,
    MEMORY_PCI,

} RMCD_BAD_READ_SPACE;

typedef enum
{
    BAD_READ_GPU_OFF_BUS = 1,
    BAD_READ_LOW_POWER,
    BAD_READ_PCI_DEVICE_DISABLED,
    BAD_READ_GPU_RESET,
    BAD_READ_DWORD_SHIFT,
    BAD_READ_UNKNOWN,

} RMCD_BAD_READ_REASON;

typedef struct 
{
    LWCD_RECORD         Header;                 // Header
    LwU32               MemorySpace;            // Which Memory Space
    LwU32               Offset;                 // Offset in Memory Space
    LwU32               Mask;                   // Mask used to detect bad read
    LwU32               Value;                  // Value Return
    LwU32               Reason;                 // Potential Reason why this might have happened

} RmRC2BadRead_RECORD, *PRmRC2BadRead_RECORD;

typedef struct 
{
    LWCD_RECORD         Header;
    LwU32               hostEngTag;
    LwU32               exceptType;
    LwU32               exceptLevel;

} RmRCRecovery_RECORD, *PRmRCRecovery_RECORD;

// Might need to also encode unit #????
typedef enum
{
    PGRAPH_LW50_FE = 1,
    PGRAPH_LW50_MEMFMT,
    PGRAPH_LW50_DA,
    PGRAPH_LW50_STREAM,
    PGRAPH_LW50_SCC,
    PGRAPH_LW50_CRSTR,
    PGRAPH_LW50_SMC,
    PGRAPH_LW50_SM0_1,
    PGRAPH_LW50_PROP,
    PGRAPH_LW50_SM,
    PGRAPH_LW50_TPC,
    PFB_LW50_LIMIT,
    PGRAPH_GT200_FE,
    PGRAPH_GT200_MEMFMT,
    PGRAPH_GT200_DA,
    PGRAPH_GT200_STREAM,
    PGRAPH_GT200_SCC,
    PGRAPH_GT200_CRSTR,
    PGRAPH_GT200_SMC,
    PGRAPH_GT200_SM012,
    PGRAPH_GT200_PROP,
    PGRAPH_GT200_SM,
    PGRAPH_GT200_TPC,
    PFIFO_C1,
    PFIFO_PB,
    SIGDUMP_INFO_LW50,
    GR_STATUS_INFO_LW50,
    SIGDUMP_INFO_G84,
    SIGDUMP_INFO_G86,
    SIGDUMP_INFO_G92,
    SIGDUMP_INFO_G94,
    SIGDUMP_INFO_G96,
    SIGDUMP_INFO_G98,
    SIGDUMP_INFO_GT200,
    PERF_INFO_LW50_V1,
    PERF_INFO_G84_V1,
    PERF_INFO_G94_V1,
    PERF_INFO_GT212_V1,
    THERMAL_TABLE_INFO_V1,
    MXM_TABLE_INFO_V1,
    PERF_INFO_GF100_V1,
    PGRAPH_GF100_PGRAPH,
    PGRAPH_GF100_FE,
    PGRAPH_GF100_MEMFMT,
    PGRAPH_GF100_SCC,
    PGRAPH_GF100_DS,
    PGRAPH_GF100_CWD,
    PGRAPH_GF100_MME,
    PGRAPH_GF100_TPCSM,
    PGRAPH_GF100_GCC,
    PGRAPH_GF100_PROP,
    PGRAPH_GF100_SETUP,
    PGRAPH_GF100_ZLWLL,
    PGRAPH_GF100_TPCL1C,
    PGRAPH_GF100_TPCPE,
    PGRAPH_GF100_TPCTEX,
    PGRAPH_GF100_BEZROP,
    PGRAPH_GF100_CROP, 
    PGRAPH_GF100_PD,
    FBBA_INFO_iGT21A_V1, 
    ELPG_INFO_GT215_V1, 
    ELPG_INFO_iGT21A_V1, 
    GPU_SIGNAL_INFO_V1_TYPE,
    PGRAPH_GK104_SKED,
    PGRAPH_GF10X_TEX_HANG,
    PGRAPH_GF117_PES,
    PGRAPH_GM107_TPCMPC,

} RMCD_UNIT_RECORD_TYPE;

#define RMCD_UNIT_VERSION_1 (1)

typedef struct
{
    LwU8                cType;                  // Encodes GPU/Unit
    LwU8                cVersion;               // Which Version
    LwU16               wSize;                  // Size in Bytes

} UNIT_HEADER, *PUNIT_HEADER;

typedef struct 
{
    UNIT_HEADER         Header;
    LwU32               hww_esr;

} GR_FE_INFO_LW50, *PGR_FE_INFO_LW50;

typedef struct 
{
    UNIT_HEADER         Header;
    LwU32               hww_esr;
    LwU32               esr_limitv_req;
    LwU32               esr_addr_lo;
    LwU32               esr_addr_hi;
    LwU32               esr_ctxdma;

} GR_MEMFMT_INFO_LW50, *PGR_MEMFMT_INFO_LW50;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               Limit[4];

} FB_LIMITV_INFO_LW50, *PFB_LIMITV_INFO_LW50;

typedef struct 
{
    UNIT_HEADER         Header;
    LwU32               hww_esr;

} GR_DA_INFO_LW50, *PGR_DA_INFO_LW50;

typedef struct 
{
    UNIT_HEADER         Header;
    LwU32               hww_esr;
    LwU32               esr_req;
    LwU32               esr_addr_lo;
    LwU32               esr_addr_hi;
    LwU32               esr_ctxdma;

} GR_STREAM_INFO_LW50, *PGR_STREAM_INFO_LW50;

typedef struct 
{
    UNIT_HEADER         Header;
    LwU32               hww_esr;

} GR_SCC_INFO_LW50, *PGR_SCC_INFO_LW50;

typedef struct 
{
    UNIT_HEADER         Header;
    LwU32               hww_esr;
    LwU32               esr_limitv_req;

} GR_CRSTR_INFO_LW50, *PGR_CRSTR_INFO_LW50;

typedef struct 
{
    UNIT_HEADER         Header;
    LwU32               unit;
    LwU32               hww_esr;
    LwU32               esr_ls_raddr;

} GR_SMC_INFO_LW50, *PGR_SMC_INFO_LW50;

typedef struct 
{
    UNIT_HEADER         Header;
    LwU32               unit;
    LwU32               sm_ctx_4;
    LwU32               sm_ctx_5;
    LwU32               sm_ctx_30;
    LwU32               sm_ctx_31;
    LwU32               sm_ctx_8;
    LwU32               sm_ctx_9;
    LwU32               sm_ctx_28;
    LwU32               sm_ctx_29;
    LwU32               smc_debug_1;
    LwU32               smc_debug_2;
    LwU32               smc_debug_3;
    LwU32               smc_debug_4;
    LwU32               smc_debug_5;
    LwU32               sm_ctx_22;
    LwU32               sm_ctx_23;

} GR_SM0_1_INFO_LW50, *PGR_SM0_1_INFO_LW50;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               unit;
    LwU32               hww_esr;
    LwU32               esr_addr_lo;
    LwU32               esr_addr_hi;
    LwU32               esr_coord;
    LwU32               esr_format;
    LwU32               esr_state;

} GR_PROP_INFO_LW50, *PGR_PROP_INFO_LW50;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               hww_esr;

} GR_SM_INFO_LW50, *PGR_SM_INFO_LW50;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               unit;
    LwU32               hww_esr;
    LwU32               esr_req;
    LwU32               esr_addr;
    LwU32               esr_ctxdma;
    LwU32               esr_mmu;

} GR_TPC_INFO_LW50, *PGR_TPC_INFO_LW50;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               C1Size;                 // Size is Dwords Pairs (Method, Data)
    LwU32               Data[1];

} FIFO_C1_INFO, *PFIFO_C1_INFO;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               PBSize;                 // Size is in Dwords
    LwU32               dwPhysAddrPBStart;      // Physical Address of PB Dump Start
    LwU32               dwPhysAddrPBEnd;        // Physical Address of PB Dump Start + DMA Get
    LwU32               Data[1];

} FIFO_PB_INFO, *PFIFO_PB_INFO;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               Data[1];

} GR_SIGDUMP_INFO, *PGR_SIGDUMP_INFO;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               dwLW_PGRAPH_STATUS;
    LwU32               dwLW_PGRAPH_GRFIFO_CONTROL;
    LwU32               dwLW_PGRAPH_INTR;
    LwU32               dwLW_PGRAPH_CTXSW_STATUS;
    LwU32               dwLW_PGRAPH_ACTIVITY0;
    LwU32               dwLW_PGRAPH_ACTIVITY1;
    LwU32               dwLW_PGRAPH_ACTIVITY2;
    LwU32               dwLW_PGRAPH_PRI_SCS_ACTIVITY_STATUS_MISC;
    LwU32               dwLW_PBUS_FS;
    LwU32               dwLW_PGRAPH_PRI_SCS_ACTIVITY_STATUS[1];

} GR_STATUS_INFO, *PGR_STATUS_INFO;

typedef struct
{
    LwU8                id;
    LwU8                version;
    LwU16               size;
    LwU16               offset;

} VBIOS_BIT_TOKEN, *PVBIOS_BIT_TOKEN;

//
// Vbios Data
//   Byte 0 -- Size in 512 chunks
//   Byte 1 - 9 -- Mods Data
//   Zero or More Bit token structures -- If BIOSData 'B' or Internal Data 'i' Available
//   Byte 10 -- Bit Token (6 bytes)
//   Byte 16 -- Raw Vbios Byte Stream (size specified in Bit Token)
//   Byte 16+size -- Bit Token (6 bytes)
//   Byte 16+size+6 -- Raw Vbios Byte Stream 

typedef struct
{
    LWCD_RECORD         Header;                 // Global information record header
    LwU8                rawData[1];

} VBIOS_OCA_DATA, *PVBIOS_OCA_DATA;

typedef struct
{
    UNIT_HEADER         Header; 
    LwU32               Size;
    LwU32               Data[1];

} ADD_FBBA_INFO, *PADD_FBBA_INFO;

typedef struct
{
    UNIT_HEADER         Header; 
    LwU32               Size;
    LwU32               Data[1];

} ADD_ELPG_INFO, *PADD_ELPG_INFO;

// iGT21A specific RC ELPG dump structure
typedef struct
{
    // ELPG status
    LwU32               regLW_PPWR_PMU_ELPG_INTRSTAT0; // 0x10a700 ELPG status for GR
    LwU32               regLW_PPWR_PMU_ELPG_INTRSTAT1; // 0x10a704 ELPG status for VIDEO
    LwU32               regLW_PPWR_PMU_ELPG_INTRSTAT2; // 0x10a778 ELPG status for VIC
    LwU32               regLW_PPWR_PMU_ELPG_STAT;      // 0x10a708

} ADD_ELPG_INFO_iGT21A_V1, *PADD_ELPG_INFO_iGT21A_V1;

// dGT21X specific RC ELPG dump structure
typedef struct
{
    // ELPG status
    LwU32               regLW_PPWR_PMU_ELPG_INTRSTAT0; // 0x10a700 ELPG status for GR
    LwU32               regLW_PPWR_PMU_ELPG_INTRSTAT1; // 0x10a704 ELPG status for VIDEO
    LwU32               regLW_PPWR_PMU_ELPG_STAT;      // 0x10a708
} ADD_ELPG_INFO_GT215_V1, *PADD_ELPG_INFO_GT215_V1;

// Need this typedef since xapi doesn't support multi-dimensional arrays
typedef LwU16 slice_regLW_PFB_FBBA_IQ_STATUS[FBBA_MUX_SEL_MAX];

// iGT21A specific RC FBBA dump structure
typedef struct
{
    // FBBA status
    LwU16               FBBAMask;                 // PBUS_FS_FB_ENABLE mask to find enabled FBBAs
    LwU32               regLW_PFB_ZBCCFG;         // 0x10029C Zero-Bandwidth-Clear configuration
    LwU32               regLW_PFB_NISO_CFG0;      // 0x100C00
    LwU32               regLW_PFB_NISO_CFG1;      // 0x100C30 Cache Flush details
    LwU32               regLW_PFB2_SC_CFG;        // 0x00111200 Default sysmem cache behavior
    LwU32               regLW_PFB2_SC_ALLOC;      // 0x00111204 Allocate cache line behavior
    LwU32               regLW_PFB2_SC_BYPASS;     // 0x00111208 Bypass cache behavior
    LwU32               regLW_PFB2_SC_FLUSHCTL;   // 0x00111228 Priv initiated cache flushes
    LwU32               regLW_PFB2_SC_PROMOTION;  // 0x0011122C Data Promotion behavior
    LwU32               regLW_PFB2_SC_PROMOTION2; // 0x00111230
    LwU32               regLW_PFB2_SC_RCM_CFG;    // 0x00111234  Reduced cache mode config
    LwU32               regLW_PFB2_SC_SWIE_CFG0;  // 0x0011124C SW/Cache interaction errors config
    LwU32               regLW_PFB2_SC_FLUSHINFO0; // 0x00111284 State of Flushing state machine
    LwU32               regLW_PFB2_SC_FLUSHINFO1; // 0x00111288
    LwU32               regLW_PFB2_SC_FLUSHINFO2; // 0x0011128C
    LwU32               regLW_PFB2_SC_FLUSHINFO3; // 0x00111290
    LwU32               regLW_PFB2_SC_RDSTATUS0[RDSTAT_MAX];    // 0x001112CC Status of return data threads within FillView
    
    // 0x001004fC Read back internal status info from each FBBA
    slice_regLW_PFB_FBBA_IQ_STATUS regLW_PFB_FBBA_IQ_STATUS[MAX_FBBAS];
    LwU16               regLW_PFB2_RXB_STATUS[FBBA_RXB_STATUS_SEL_MAX]; // 0x001117E4

} ADD_FBBA_INFO_iGT21A_V1, *PADD_FBBA_INFO_iGT21A_V1;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               pgraph;

} GR_PGRAPH_INFO_GF100, *PGR_PGRAPH_INFO_GF100;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               hwwEsr;

} GR_HWWESR_INFO_GF100, *PGR_HWWESR_INFO_GF100;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               hwwEsr;
    LwU32               constBuff;
    LwU32               constBuffBase;

} GR_SCC_INFO_GF100, *PGR_SCC_INFO_GF100;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               hwwEsr;
    LwU32               lMemSize;

} GR_CWD_INFO_GF100, *PGR_CWD_INFO_GF100;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               hwwEsr;
    LwU32               info;
    LwU32               info2;

} GR_MME_INFO_GF100, *PGR_MME_INFO_GF100;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               hwwEsr;
    LwU32               gpcCounter;
    LwU32               tpcCounter;
    LwU32               hwwWarpEsr;
    LwU32               hwwWarpRptMask;
    LwU32               hwwGlbEsr;
    LwU32               hwwGlbRptMask;
    LwU32               eccStatus;

} GR_TPCSM_INFO_GF100, *PGR_TPCSM_INFO_GF100;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               hwwEsr;
    LwU32               gpcCounter;
    LwU32               badHdrIdx;
    LwU32               badSmpIdx;

} GR_GCC_INFO_GF100, *PGR_GCC_INFO_GF100;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               hwwEsr;
    LwU32               gpcCounter;
    LwU32               esrCoord;
    LwU32               esrFormat;
    LwU32               esrState;

} GR_PROP_INFO_GF100, *PGR_PROP_INFO_GF100;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               hwwEsr;
    LwU32               gpcCounter;

} GR_SETUP_INFO_GF100, *PGR_SETUP_INFO_GF100;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               hwwEsr;
    LwU32               gpcCounter;
    LwU32               ppcCounter;

} GR_PES_INFO_GF117, *PGR_PES_INFO_GF117;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               hwwEsr;
    LwU32               gpcCounter;
    LwU32               hwwInfo0;
    LwU32               hwwInfo1;         
    LwU32               hwwInfo2;

} GR_ZLWLL_INFO_GF100, *PGR_ZLWLL_INFO_GF100;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               hwwEsr;
    LwU32               gpcCounter;
    LwU32               tpcCounter;
    LwU32               hwwEsrAddr;
    LwU32               hwwEsrReq;
    LwU32               l1cEccCsr;

} GR_TPCL1C_INFO_GF100, *PGR_TPCL1C_INFO_GF100;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               hwwEsr;
    LwU32               gpcCounter;
    LwU32               tpcCounter;
    LwU32               vpgAddr;

} GR_TPCMPC_INFO_GM107, *PGR_TPCMPC_INFO_GM107;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               hwwEsr;
    LwU32               gpcCounter;
    LwU32               tpcCounter;

} GR_TPCPE_INFO_GF100, *PGR_TPCPE_INFO_GF100;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               hwwEsr;
    LwU32               gpcCounter;
    LwU32               tpcCounter;
    LwU32               esrReq;
    LwU32               esrAddr;
    LwU32               esrAddr1;
    LwU32               esrMMU;

} GR_TPCTEX_INFO_GF100, *PGR_TPCTEX_INFO_GF100;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               hwwEsr;
    LwU32               beCounter;

} GR_BEZROP_INFO_GF100, *PGR_BEZROP_INFO_GF100;

typedef struct
{
    UNIT_HEADER         Header;
    LwU32               hwwEsr;
    LwU32               beCounter;

} GR_CROP_INFO_GF100, *PGR_CROP_INFO_GF100;

typedef struct
{
    UNIT_HEADER         Header; 
    LwU32               Size;
    LwU32               Data[1];

} THERMAL_TABLE_INFO, *PTHERMAL_TABLE_INFO;

typedef struct
{
    UNIT_HEADER         Header; 
    LwU32               Size;
    LwU32               Data[1];

} MXM_TABLE_INFO, *PMXM_TABLE_INFO;

typedef struct
{
    LWCD_RECORD         Header;                 // Global information record header
    LwU32               dwPCI_ID;               // Device PCI vendor/device ID value
    LwU32               dwSubSys_ID;            // Subsystem vendor/device ID value
    LwU32               dwPmcBoot0;             // Chip Identification
    LwU32               dwPmcBoot42;
    LwU32               dwNorthBridgeId;        // Core Logic vendor/device ID value
    LwU32               dwSubNorthBridgeId;     // Core Logic vendor/device ID value
    LwU32               dwBiosRevision;         // BIOS revision value
    LwU32               dwBiosOEMRevision;      // OEM BIOS revision value
    LwU32               dwErrorCount;           // Number of error records to follow

} RmSafeGlobalInfo_RECORD, *PRmSafeGlobalInfo_RECORD;

typedef struct
{
    LWCD_RECORD         Header;                 // Global information record header
    LwU32               dwSize;                 // Size of Journal > 64K
    LwP64               DriverStart;            // Driver Start
    LwU32               Offset;                 // Offset to Text
    LwU32               pointerSize;            // Size of Pointer 4=x86 and 8=AMD64
    LwU8                rawData[1];             // Byte stream specified by above
                                        
} RmJournal_RECORD_V3, *PRmJournal_RECORD_V3;

#define    RM_EVENT_PG_ILWALID          0x0     // Invalid Event ID
#define    RM_EVENT_PG_INITIATE_PGOFF   0x1     // Power Up Initiated
#define    RM_EVENT_PG_INIT_ACK         0x2     // Received the ack for ELPG_INIT commands 
#define    RM_EVENT_PG_ON               0x3     // Received a PG_ON interrupt during powerdown
#define    RM_EVENT_PG_ON_DONE          0x4     // Received a PG_ON_DONE interrupt during powerdown
#define    RM_EVENT_PG_ENG_RESET        0x5     // Received a PG_ENG_RST interrupt during powerup
#define    RM_EVENT_PG_CTX_RESTORE      0x6     // Received a PG_CTX_RESTORE interrupt during powerup
#define    RM_EVENT_PG_CTX_SAVE         0x7     // Received a PG_CTX_SAVE interrupt during powerdown
#define    RM_EVENT_PG_CTX_SAVE_ABORT   0x8     // Aborted PG_CTX_SAVE process during powerdown
#define    RM_EVENT_PG_DENY_PGON        0x9     // Denied or aborted the powerdown process
#define    RM_EVENT_PG_ENABLE           0xA     // Enable all PG interrupts
#define    RM_EVENT_PG_DISABLE          0xB     // Disable all PG interrupts
#define    RM_EVENT_PG_FLUSH            0xC     // Flush all pending PG events 
#define    RM_EVENT_PG_CFG_ERR          0xD     // Received a PG_CFG_ERR interrupt
#define    RM_EVENT_PG_HOLDOFF          0xE     // Received a Engine holdoff interrupt to initiate powerup
#define    RM_EVENT_PG_ABORTED_POWER_UP 0xF     // Wake for SW abort

#define    WLM_FREEZE                   0x64    // WLM Freeze issued (mthdctx_holdoff applied)
#define    WLM_FREEZE_WAIT_COMPLETE     0x65    // WLM Freeze Wait Complete
#define    WLM_FREEZE_WAIT_TIMEOUT      0x66    // WLM Freeze Wait Timed Out
#define    WLM_UNFREEZE_POWERUP         0x67    // WLM Unfreeze, ELPG Powerup
#define    WLM_UPFREEZE_POWERUP_DONE    0x68    // WLM UnFreeze Power Up complete
#define    WLM_UNFREEZE_CLR_HOLDOFF     0x69    // WLM UnFreeze, Clear Holdoff 
#define    WLM_CLR_MTHD_INTR            0x6A    // WLM clear MTHD INTR
#define    WLM_LOW_ELPG_THRSHLD         0x6B    // WLM setting LOW ELPG Entry  Threshold
#define    WLM_HIGH_ELPG_THRSHLD        0x6C    // WLM setting HIGH ELPG Entry Threshold

#define    LW_PG_EVENTLOG_EVENT         7:0
#define    LW_PG_EVENTLOG_DEFERRED      8:8
#define    LW_PG_EVENTLOG_ENDOFEVENT    9:9

typedef struct _elpg_log_entry
{
    LwU32               timestamp;
    LwU32               event;
    LwU32               engine;
    LwU32               status;

} ELPG_LOG_ENTRY, *PELPG_LOG_ENTRY;

typedef struct
{
    LWCD_RECORD         Header;                 // Global information record header
    ELPG_LOG_ENTRY      elpgLog[1];

} ELPG_OCA_DATA, *PELPG_OCA_DATA;

typedef struct
{
    LWCD_RECORD         Header;                 // Global information record header.    
    LwU8                cEntryRecordType;       // The type of each entry stored in the ring buffer. 
    LwU32               entrySize;              // Size of each entry in the ring buffer. 
    LwU32               maxEntries;
    LwU32               numFilledEntries;       // Number of entries in the ring buffer. 

    // 
    // Raw data of the ring buffer. 
    // Entries in the ring buffer are dumped in head-to-tail order
    // So, entries rawData[0...n-1] are the last n commands logged. 
    // 
    LwU8                rawData[1];

} RmRingBuffer_RECORD, *PRmRingBuffer_RECORD;

typedef struct
{
    UNIT_HEADER         Header; 
    LwU32               signal;
    LwU32               value;

} GPU_SIGNAL_INFO_V1, *PGPU_SIGNAL_INFO_V1;

// Signals enum for GPU_SIGNAL_INFO
typedef enum
{
    GPU_SIGNAL_INFO_SIGNAL_TYPE_ZTLF_PM_ZADDR_WAIT_0 = 0,
    GPU_SIGNAL_INFO_SIGNAL_TYPE_ZTLF_PM_ZADDR_WAIT_1,
    GPU_SIGNAL_INFO_SIGNAL_TYPE_ZTLF_PM_ZADDR_WAIT_2,
    GPU_SIGNAL_INFO_SIGNAL_TYPE_ZTLF_PM_ZADDR_WAIT_3,
    GPU_SIGNAL_INFO_SIGNAL_TYPE_ZTLF_PM_ZADDR_WAIT_4,
    GPU_SIGNAL_INFO_SIGNAL_TYPE_ZTLF_PM_ZADDR_WAIT_5,
    GPU_SIGNAL_INFO_SIGNAL_TYPE_ZTLF_PM_ZADDR_WAIT_6,
    GPU_SIGNAL_INFO_SIGNAL_TYPE_ZTLF_PM_ZADDR_WAIT_7,
    GPU_SIGNAL_INFO_SIGNAL_TYPE_CSB_PM_CADDR_WAIT_0,
    GPU_SIGNAL_INFO_SIGNAL_TYPE_CSB_PM_CADDR_WAIT_1,
    GPU_SIGNAL_INFO_SIGNAL_TYPE_CSB_PM_CADDR_WAIT_2,
    GPU_SIGNAL_INFO_SIGNAL_TYPE_CSB_PM_CADDR_WAIT_3,
    GPU_SIGNAL_INFO_SIGNAL_TYPE_CSB_PM_CADDR_WAIT_4,
    GPU_SIGNAL_INFO_SIGNAL_TYPE_CSB_PM_CADDR_WAIT_5,
    GPU_SIGNAL_INFO_SIGNAL_TYPE_CSB_PM_CADDR_WAIT_6,
    GPU_SIGNAL_INFO_SIGNAL_TYPE_CSB_PM_CADDR_WAIT_7

} GPU_SIGNAL_INFO_SIGNAL_TYPE;

// GF104, GF106, GF108 specific RC dump structures
#define LW_MAX_GPC_COUNT    4
#define LW_MAX_TPC_COUNT    4

typedef struct
{
    LwU32               dwLW_PGRAPH_TEX_M_STATUS_TEXIN[LW_MAX_TPC_COUNT];
    LwU32               dwLW_PGRAPH_TEX_M_2_STATUS_TEXIN[LW_MAX_TPC_COUNT];
    LwU32               dwLW_PGRAPH_TEX_M_STATUS_TM[LW_MAX_TPC_COUNT];
    LwU32               dwLW_PGRAPH_TEX_M_2_STATUS_TM[LW_MAX_TPC_COUNT];
    LwU32               dwLW_PGRAPH_TEX_STATUS_GCC;

} GR_TEX_STATUS, *PGR_TEX_STATUS;

typedef struct
{
    UNIT_HEADER         Header;
    GR_TEX_STATUS       dwLW_GRAPH_TEX_STATUS_INFO[LW_MAX_GPC_COUNT];
    LwBool              dwLW_PGRAPH_2X_TEX_HANG_DETECTED;

} GR_TEX_HANG_SIGNATURE_INFO, *PGR_TEX_HANG_SIGNATURE_INFO;

typedef enum
{
    SURPRISE_REMOVAL_GPU_OFF_BUS = 1,
    SURPRISE_REMOVAL_SSID_MISMATCH,
    SURPRISE_REMOVAL_SYSTEMBIOS_NO_LOAD,
    SURPRISE_REMOVAL_PROM_CHECKSUM,
    SURPRISE_REMOVAL_UNKNOWN,

} RMCD_SURPRISE_REMOVAL_REASON;

// Taken from Appendix G in ROM Data Map -- dev_ext_devices.ref
//#define CHECKSUM0_LENGTH (0x6C-0x58)
// Above line breaks xapi
// Changed for xpai from CHECKSUM0_LENGTH + sizeof(LwU32) to
#define CHECKSUM0_TOTAL_LENGTH (0x18)
typedef struct 
{
    LWCD_RECORD         Header;                 // Header
    LwU32               Reason;                 // Potential Reason why this might have happened
    LwU32               expPCIDeviceID;
    LwU32               expPCISubDeviceID;
    LwU32               lwrPCIDeviceID;
    LwU32               lwrPCISubDeviceID;
    LwU32               SSIDBios;               // Where do we get the SSID from
    LwU32               DebugReg;
    LwU8                SystemImage[CHECKSUM0_TOTAL_LENGTH];    // Grab SSID from Bios + Checksum area

} RmRC2SurpriseRemoval_RECORD, *PRmRC2SurpriseRemoval_RECORD;






//******************************************************************************
//
//  Functions
//
//******************************************************************************






} // oca namespace
//******************************************************************************
//
//  End Of File
//
//******************************************************************************
#endif // _OCARM_H
