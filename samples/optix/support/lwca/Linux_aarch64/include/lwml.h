/*
 * Copyright 1993-2020 LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to LWPU ownership rights under U.S. and 
 * international Copyright laws.  Users and possessors of this source code 
 * are hereby granted a nonexclusive, royalty-free license to use this code 
 * in individual and commercial software.
 *
 * LWPU MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  LWPU DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL LWPU BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein. 
 *
 * Any use of this source code in individual and commercial software must 
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* 
LWML API Reference

The LWPU Management Library (LWML) is a C-based programmatic interface for monitoring and 
managing various states within LWPU Tesla &tm; GPUs. It is intended to be a platform for building
3rd party applications, and is also the underlying library for the LWPU-supported lwpu-smi
tool. LWML is thread-safe so it is safe to make simultaneous LWML calls from multiple threads.

API Documentation

Supported platforms:
- Windows:     Windows Server 2008 R2 64bit, Windows Server 2012 R2 64bit, Windows 7 64bit, Windows 8 64bit, Windows 10 64bit
- Linux:       32-bit and 64-bit
- Hypervisors: Windows Server 2008R2/2012 Hyper-V 64bit, Citrix XenServer 6.2 SP1+, VMware ESX 5.1/5.5

Supported products:
- Full Support
    - All Tesla products, starting with the Fermi architecture
    - All Lwdqro products, starting with the Fermi architecture
    - All vGPU Software products, starting with the Kepler architecture
    - Selected VdChip Titan products
- Limited Support
    - All Vdchip products, starting with the Fermi architecture

The LWML library can be found at \%ProgramW6432\%\\"LWPU Corporation"\\LWSMI\\ on Windows. It is
not be added to the system path by default. To dynamically link to LWML, add this path to the PATH 
elwironmental variable. To dynamically load LWML, call LoadLibrary with this path.

On Linux the LWML library will be found on the standard library path. For 64 bit Linux, both the 32 bit
and 64 bit LWML libraries will be installed.

Online documentation for this library is available at http://docs.lwpu.com/deploy/lwml-api/index.html
*/

#ifndef __lwml_lwml_h__
#define __lwml_lwml_h__

#ifdef __cplusplus
extern "C" {
#endif

/*
 * On Windows, set up methods for DLL export
 * define LWML_STATIC_IMPORT when using lwml_loader library
 */
#if defined _WINDOWS
    #if !defined LWML_STATIC_IMPORT
        #if defined LWML_LIB_EXPORT
            #define DECLDIR __declspec(dllexport)
        #else
            #define DECLDIR __declspec(dllimport)
        #endif
    #else
        #define DECLDIR
    #endif
#else
    #define DECLDIR
#endif

/**
 * LWML API versioning support
 */
#define LWML_API_VERSION            11
#define LWML_API_VERSION_STR        "11"
/**
 * Defining LWML_NO_ULWERSIONED_FUNC_DEFS will disable "auto upgrading" of APIs.
 * e.g. the user will have to call lwmlInit_v2 instead of lwmlInit. Enable this
 * guard if you need to support older versions of the API
 */
#ifndef LWML_NO_ULWERSIONED_FUNC_DEFS
    #define lwmlInit                                lwmlInit_v2
    #define lwmlDeviceGetPciInfo                    lwmlDeviceGetPciInfo_v3
    #define lwmlDeviceGetCount                      lwmlDeviceGetCount_v2
    #define lwmlDeviceGetHandleByIndex              lwmlDeviceGetHandleByIndex_v2
    #define lwmlDeviceGetHandleByPciBusId           lwmlDeviceGetHandleByPciBusId_v2
    #define lwmlDeviceGetLwLinkRemotePciInfo        lwmlDeviceGetLwLinkRemotePciInfo_v2
    #define lwmlDeviceRemoveGpu                     lwmlDeviceRemoveGpu_v2
    #define lwmlDeviceGetGridLicensableFeatures     lwmlDeviceGetGridLicensableFeatures_v3
    #define lwmlEventSetWait                        lwmlEventSetWait_v2
    #define lwmlDeviceGetAttributes                 lwmlDeviceGetAttributes_v2
    #define lwmlComputeInstanceGetInfo              lwmlComputeInstanceGetInfo_v2
    #define lwmlDeviceGetComputeRunningProcesses    lwmlDeviceGetComputeRunningProcesses_v2
    #define lwmlDeviceGetGraphicsRunningProcesses   lwmlDeviceGetGraphicsRunningProcesses_v2
    #define lwmlBlacklistDeviceInfo_t               lwmlExcludedDeviceInfo_t
    #define lwmlGetBlacklistDeviceCount             lwmlGetExcludedDeviceCount
    #define lwmlGetBlacklistDeviceInfoByIndex       lwmlGetExcludedDeviceInfoByIndex
#endif // #ifndef LWML_NO_ULWERSIONED_FUNC_DEFS

/***************************************************************************************************/
/** @defgroup lwmlDeviceStructs Device Structs
 *  @{
 */
/***************************************************************************************************/

/**
 * Special constant that some fields take when they are not available.
 * Used when only part of the struct is not available.
 *
 * Each structure explicitly states when to check for this value.
 */
#define LWML_VALUE_NOT_AVAILABLE (-1)

typedef struct lwmlDevice_st* lwmlDevice_t;

/**
 * Buffer size guaranteed to be large enough for pci bus id
 */
#define LWML_DEVICE_PCI_BUS_ID_BUFFER_SIZE      32

/**
 * Buffer size guaranteed to be large enough for pci bus id for ::busIdLegacy
 */
#define LWML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE   16

/**
 * PCI information about a GPU device.
 */
typedef struct lwmlPciInfo_st
{
    char busIdLegacy[LWML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE]; //!< The legacy tuple domain:bus:device.function PCI identifier (&amp; NULL terminator)
    unsigned int domain;             //!< The PCI domain on which the device's bus resides, 0 to 0xffffffff
    unsigned int bus;                //!< The bus on which the device resides, 0 to 0xff
    unsigned int device;             //!< The device's id on the bus, 0 to 31
    unsigned int pciDeviceId;        //!< The combined 16-bit device id and 16-bit vendor id

    // Added in LWML 2.285 API
    unsigned int pciSubSystemId;     //!< The 32-bit Sub System Device ID

    char busId[LWML_DEVICE_PCI_BUS_ID_BUFFER_SIZE]; //!< The tuple domain:bus:device.function PCI identifier (&amp; NULL terminator)
} lwmlPciInfo_t;

/**
 * PCI format string for ::busIdLegacy
 */
#define LWML_DEVICE_PCI_BUS_ID_LEGACY_FMT           "%04X:%02X:%02X.0"

/**
 * PCI format string for ::busId
 */
#define LWML_DEVICE_PCI_BUS_ID_FMT                  "%08X:%02X:%02X.0"

/**
 * Utility macro for filling the pci bus id format from a lwmlPciInfo_t
 */
#define LWML_DEVICE_PCI_BUS_ID_FMT_ARGS(pciInfo)    (pciInfo)->domain, \
                                                    (pciInfo)->bus,    \
                                                    (pciInfo)->device

/**
 * Detailed ECC error counts for a device.
 *
 * @deprecated  Different GPU families can have different memory error counters
 *              See \ref lwmlDeviceGetMemoryErrorCounter
 */
typedef struct lwmlEccErrorCounts_st 
{
    unsigned long long l1Cache;      //!< L1 cache errors
    unsigned long long l2Cache;      //!< L2 cache errors
    unsigned long long deviceMemory; //!< Device memory errors
    unsigned long long registerFile; //!< Register file errors
} lwmlEccErrorCounts_t;

/** 
 * Utilization information for a device.
 * Each sample period may be between 1 second and 1/6 second, depending on the product being queried.
 */
typedef struct lwmlUtilization_st 
{
    unsigned int gpu;                //!< Percent of time over the past sample period during which one or more kernels was exelwting on the GPU
    unsigned int memory;             //!< Percent of time over the past sample period during which global (device) memory was being read or written
} lwmlUtilization_t;

/** 
 * Memory allocation information for a device.
 */
typedef struct lwmlMemory_st 
{
    unsigned long long total;        //!< Total installed FB memory (in bytes)
    unsigned long long free;         //!< Unallocated FB memory (in bytes)
    unsigned long long used;         //!< Allocated FB memory (in bytes). Note that the driver/GPU always sets aside a small amount of memory for bookkeeping
} lwmlMemory_t;

/**
 * BAR1 Memory allocation Information for a device
 */
typedef struct lwmlBAR1Memory_st
{
    unsigned long long bar1Total;    //!< Total BAR1 Memory (in bytes)
    unsigned long long bar1Free;     //!< Unallocated BAR1 Memory (in bytes)
    unsigned long long bar1Used;     //!< Allocated Used Memory (in bytes)
}lwmlBAR1Memory_t;

/**
 * Information about running compute processes on the GPU
 */
typedef struct lwmlProcessInfo_st
{
    unsigned int        pid;                //!< Process ID
    unsigned long long  usedGpuMemory;      //!< Amount of used GPU memory in bytes.
                                            //! Under WDDM, \ref LWML_VALUE_NOT_AVAILABLE is always reported
                                            //! because Windows KMD manages all the memory and not the LWPU driver
    unsigned int        gpuInstanceId;      //!< If MIG is enabled, stores a valid GPU instance ID. gpuInstanceId is set to
                                            //  0xFFFFFFFF otherwise.
    unsigned int        computeInstanceId;  //!< If MIG is enabled, stores a valid compute instance ID. computeInstanceId is set to
                                            //  0xFFFFFFFF otherwise.
} lwmlProcessInfo_t;

typedef struct lwmlDeviceAttributes_st
{
    unsigned int multiprocessorCount;       //!< Streaming Multiprocessor count
    unsigned int sharedCopyEngineCount;     //!< Shared Copy Engine count
    unsigned int sharedDecoderCount;        //!< Shared Decoder Engine count
    unsigned int sharedEncoderCount;        //!< Shared Encoder Engine count
    unsigned int sharedJpegCount;           //!< Shared JPEG Engine count
    unsigned int sharedOfaCount;            //!< Shared OFA Engine count
    unsigned int gpuInstanceSliceCount;     //!< GPU instance slice count
    unsigned int computeInstanceSliceCount; //!< Compute instance slice count
    unsigned long long memorySizeMB;        //!< Device memory size (in MiB)
} lwmlDeviceAttributes_t;

/**
 * Possible values that classify the remap availability for each bank. The max
 * field will contain the number of banks that have maximum remap availability
 * (all reserved rows are available). None means that there are no reserved
 * rows available.
 */
typedef struct lwmlRowRemapperHistogramValues_st
{
    unsigned int max;
    unsigned int high;
    unsigned int partial;
    unsigned int low;
    unsigned int none;
} lwmlRowRemapperHistogramValues_t;

/**
 * Enum to represent type of bridge chip
 */
typedef enum lwmlBridgeChipType_enum
{
    LWML_BRIDGE_CHIP_PLX = 0,
    LWML_BRIDGE_CHIP_BRO4 = 1           
}lwmlBridgeChipType_t;

/**
 * Maximum number of LwLink links supported 
 */
#define LWML_LWLINK_MAX_LINKS 12

/**
 * Enum to represent the LwLink utilization counter packet units
 */
typedef enum lwmlLwLinkUtilizationCountUnits_enum
{
    LWML_LWLINK_COUNTER_UNIT_CYCLES =  0,     // count by cycles
    LWML_LWLINK_COUNTER_UNIT_PACKETS = 1,     // count by packets
    LWML_LWLINK_COUNTER_UNIT_BYTES   = 2,     // count by bytes
    LWML_LWLINK_COUNTER_UNIT_RESERVED = 3,    // count reserved for internal use
    // this must be last
    LWML_LWLINK_COUNTER_UNIT_COUNT
} lwmlLwLinkUtilizationCountUnits_t;

/**
 * Enum to represent the LwLink utilization counter packet types to count
 *  ** this is ONLY applicable with the units as packets or bytes
 *  ** as specified in \a lwmlLwLinkUtilizationCountUnits_t
 *  ** all packet filter descriptions are target GPU centric
 *  ** these can be "OR'd" together 
 */
typedef enum lwmlLwLinkUtilizationCountPktTypes_enum
{
    LWML_LWLINK_COUNTER_PKTFILTER_NOP        = 0x1,     // no operation packets
    LWML_LWLINK_COUNTER_PKTFILTER_READ       = 0x2,     // read packets
    LWML_LWLINK_COUNTER_PKTFILTER_WRITE      = 0x4,     // write packets
    LWML_LWLINK_COUNTER_PKTFILTER_RATOM      = 0x8,     // reduction atomic requests
    LWML_LWLINK_COUNTER_PKTFILTER_NRATOM     = 0x10,    // non-reduction atomic requests
    LWML_LWLINK_COUNTER_PKTFILTER_FLUSH      = 0x20,    // flush requests
    LWML_LWLINK_COUNTER_PKTFILTER_RESPDATA   = 0x40,    // responses with data
    LWML_LWLINK_COUNTER_PKTFILTER_RESPNODATA = 0x80,    // responses without data
    LWML_LWLINK_COUNTER_PKTFILTER_ALL        = 0xFF     // all packets
} lwmlLwLinkUtilizationCountPktTypes_t;

/** 
 * Struct to define the LWLINK counter controls
 */
typedef struct lwmlLwLinkUtilizationControl_st
{
    lwmlLwLinkUtilizationCountUnits_t units;
    lwmlLwLinkUtilizationCountPktTypes_t pktfilter;
} lwmlLwLinkUtilizationControl_t;

/**
 * Enum to represent LwLink queryable capabilities
 */
typedef enum lwmlLwLinkCapability_enum
{
    LWML_LWLINK_CAP_P2P_SUPPORTED = 0,     // P2P over LWLink is supported
    LWML_LWLINK_CAP_SYSMEM_ACCESS = 1,     // Access to system memory is supported
    LWML_LWLINK_CAP_P2P_ATOMICS   = 2,     // P2P atomics are supported
    LWML_LWLINK_CAP_SYSMEM_ATOMICS= 3,     // System memory atomics are supported
    LWML_LWLINK_CAP_SLI_BRIDGE    = 4,     // SLI is supported over this link
    LWML_LWLINK_CAP_VALID         = 5,     // Link is supported on this device
    // should be last
    LWML_LWLINK_CAP_COUNT
} lwmlLwLinkCapability_t;

/**
 * Enum to represent LwLink queryable error counters
 */
typedef enum lwmlLwLinkErrorCounter_enum
{
    LWML_LWLINK_ERROR_DL_REPLAY   = 0,     // Data link transmit replay error counter
    LWML_LWLINK_ERROR_DL_RECOVERY = 1,     // Data link transmit recovery error counter
    LWML_LWLINK_ERROR_DL_CRC_FLIT = 2,     // Data link receive flow control digit CRC error counter
    LWML_LWLINK_ERROR_DL_CRC_DATA = 3,     // Data link receive data CRC error counter

    // this must be last
    LWML_LWLINK_ERROR_COUNT
} lwmlLwLinkErrorCounter_t;

/**
 * Enum to represent LwLink's remote device type
 */
typedef enum lwmlIntLwLinkDeviceType_enum
{
    LWML_LWLINK_DEVICE_TYPE_GPU     = 0x00,
    LWML_LWLINK_DEVICE_TYPE_IBMNPU  = 0x01,
    LWML_LWLINK_DEVICE_TYPE_SWITCH  = 0x02,
    LWML_LWLINK_DEVICE_TYPE_UNKNOWN = 0xFF,
} lwmlIntLwLinkDeviceType_t;

/**
 * Represents level relationships within a system between two GPUs
 * The enums are spaced to allow for future relationships
 */
typedef enum lwmlGpuLevel_enum
{
    LWML_TOPOLOGY_INTERNAL           = 0, // e.g. Tesla K80
    LWML_TOPOLOGY_SINGLE             = 10, // all devices that only need traverse a single PCIe switch
    LWML_TOPOLOGY_MULTIPLE           = 20, // all devices that need not traverse a host bridge
    LWML_TOPOLOGY_HOSTBRIDGE         = 30, // all devices that are connected to the same host bridge
    LWML_TOPOLOGY_NODE               = 40, // all devices that are connected to the same NUMA node but possibly multiple host bridges
    LWML_TOPOLOGY_SYSTEM             = 50, // all devices in the system

    // there is purposefully no COUNT here because of the need for spacing above
} lwmlGpuTopologyLevel_t;

/* Compatibility for CPU->NODE renaming */
#define LWML_TOPOLOGY_CPU LWML_TOPOLOGY_NODE

/* P2P Capability Index Status*/
typedef enum lwmlGpuP2PStatus_enum
{
    LWML_P2P_STATUS_OK     = 0,
    LWML_P2P_STATUS_CHIPSET_NOT_SUPPORED,
    LWML_P2P_STATUS_GPU_NOT_SUPPORTED,
    LWML_P2P_STATUS_IOH_TOPOLOGY_NOT_SUPPORTED,
    LWML_P2P_STATUS_DISABLED_BY_REGKEY,
    LWML_P2P_STATUS_NOT_SUPPORTED,
    LWML_P2P_STATUS_UNKNOWN

} lwmlGpuP2PStatus_t;

/* P2P Capability Index*/
typedef enum lwmlGpuP2PCapsIndex_enum
{
    LWML_P2P_CAPS_INDEX_READ = 0,
    LWML_P2P_CAPS_INDEX_WRITE,
    LWML_P2P_CAPS_INDEX_LWLINK,
    LWML_P2P_CAPS_INDEX_ATOMICS,
    LWML_P2P_CAPS_INDEX_PROP,
    LWML_P2P_CAPS_INDEX_UNKNOWN
}lwmlGpuP2PCapsIndex_t;

/**
 * Maximum limit on Physical Bridges per Board
 */
#define LWML_MAX_PHYSICAL_BRIDGE                         (128)

/**
 * Information about the Bridge Chip Firmware
 */
typedef struct lwmlBridgeChipInfo_st
{
    lwmlBridgeChipType_t type;                  //!< Type of Bridge Chip 
    unsigned int fwVersion;                     //!< Firmware Version. 0=Version is unavailable
}lwmlBridgeChipInfo_t;

/**
 * This structure stores the complete Hierarchy of the Bridge Chip within the board. The immediate 
 * bridge is stored at index 0 of bridgeInfoList, parent to immediate bridge is at index 1 and so forth.
 */
typedef struct lwmlBridgeChipHierarchy_st
{
    unsigned char  bridgeCount;                 //!< Number of Bridge Chips on the Board
    lwmlBridgeChipInfo_t bridgeChipInfo[LWML_MAX_PHYSICAL_BRIDGE]; //!< Hierarchy of Bridge Chips on the board
}lwmlBridgeChipHierarchy_t;

/**
 *  Represents Type of Sampling Event
 */
typedef enum lwmlSamplingType_enum
{
    LWML_TOTAL_POWER_SAMPLES        = 0, //!< To represent total power drawn by GPU
    LWML_GPU_UTILIZATION_SAMPLES    = 1, //!< To represent percent of time during which one or more kernels was exelwting on the GPU
    LWML_MEMORY_UTILIZATION_SAMPLES = 2, //!< To represent percent of time during which global (device) memory was being read or written
    LWML_ENC_UTILIZATION_SAMPLES    = 3, //!< To represent percent of time during which LWENC remains busy
    LWML_DEC_UTILIZATION_SAMPLES    = 4, //!< To represent percent of time during which LWDEC remains busy            
    LWML_PROCESSOR_CLK_SAMPLES      = 5, //!< To represent processor clock samples
    LWML_MEMORY_CLK_SAMPLES         = 6, //!< To represent memory clock samples
            
    // Keep this last
    LWML_SAMPLINGTYPE_COUNT               
}lwmlSamplingType_t;

/**
 * Represents the queryable PCIe utilization counters
 */
typedef enum lwmlPcieUtilCounter_enum
{
    LWML_PCIE_UTIL_TX_BYTES             = 0, // 1KB granularity
    LWML_PCIE_UTIL_RX_BYTES             = 1, // 1KB granularity
    
    // Keep this last
    LWML_PCIE_UTIL_COUNT
} lwmlPcieUtilCounter_t;

/**
 * Represents the type for sample value returned
 */
typedef enum lwmlValueType_enum 
{
    LWML_VALUE_TYPE_DOUBLE = 0,
    LWML_VALUE_TYPE_UNSIGNED_INT = 1,
    LWML_VALUE_TYPE_UNSIGNED_LONG = 2,
    LWML_VALUE_TYPE_UNSIGNED_LONG_LONG = 3,
    LWML_VALUE_TYPE_SIGNED_LONG_LONG = 4,

    // Keep this last
    LWML_VALUE_TYPE_COUNT
}lwmlValueType_t;


/**
 * Union to represent different types of Value
 */
typedef union lwmlValue_st
{
    double dVal;                    //!< If the value is double
    unsigned int uiVal;             //!< If the value is unsigned int
    unsigned long ulVal;            //!< If the value is unsigned long
    unsigned long long ullVal;      //!< If the value is unsigned long long
    signed long long sllVal;        //!< If the value is signed long long
}lwmlValue_t;

/**
 * Information for Sample
 */
typedef struct lwmlSample_st 
{
    unsigned long long timeStamp;       //!< CPU Timestamp in microseconds
    lwmlValue_t sampleValue;        //!< Sample Value
}lwmlSample_t;

/**
 * Represents type of perf policy for which violation times can be queried 
 */
typedef enum lwmlPerfPolicyType_enum
{
    LWML_PERF_POLICY_POWER = 0,              //!< How long did power violations cause the GPU to be below application clocks
    LWML_PERF_POLICY_THERMAL = 1,            //!< How long did thermal violations cause the GPU to be below application clocks
    LWML_PERF_POLICY_SYNC_BOOST = 2,         //!< How long did sync boost cause the GPU to be below application clocks
    LWML_PERF_POLICY_BOARD_LIMIT = 3,        //!< How long did the board limit cause the GPU to be below application clocks
    LWML_PERF_POLICY_LOW_UTILIZATION = 4,    //!< How long did low utilization cause the GPU to be below application clocks
    LWML_PERF_POLICY_RELIABILITY = 5,        //!< How long did the board reliability limit cause the GPU to be below application clocks

    LWML_PERF_POLICY_TOTAL_APP_CLOCKS = 10,  //!< Total time the GPU was held below application clocks by any limiter (0 - 5 above)
    LWML_PERF_POLICY_TOTAL_BASE_CLOCKS = 11, //!< Total time the GPU was held below base clocks

    // Keep this last
    LWML_PERF_POLICY_COUNT
}lwmlPerfPolicyType_t;

/**
 * Struct to hold perf policy violation status data
 */
typedef struct lwmlViolationTime_st
{
    unsigned long long referenceTime;  //!< referenceTime represents CPU timestamp in microseconds
    unsigned long long violationTime;  //!< violationTime in Nanoseconds
}lwmlViolationTime_t;

/** @} */

/***************************************************************************************************/
/** @defgroup lwmlDeviceEnumvs Device Enums
 *  @{
 */
/***************************************************************************************************/

/** 
 * Generic enable/disable enum. 
 */
typedef enum lwmlEnableState_enum 
{
    LWML_FEATURE_DISABLED    = 0,     //!< Feature disabled 
    LWML_FEATURE_ENABLED     = 1      //!< Feature enabled
} lwmlEnableState_t;

//! Generic flag used to specify the default behavior of some functions. See description of particular functions for details.
#define lwmlFlagDefault     0x00      
//! Generic flag used to force some behavior. See description of particular functions for details.
#define lwmlFlagForce       0x01      

/**
 *  * The Brand of the GPU
 *   */
typedef enum lwmlBrandType_enum
{
    LWML_BRAND_UNKNOWN          = 0, 
    LWML_BRAND_QUADRO           = 1,
    LWML_BRAND_TESLA            = 2,
    LWML_BRAND_LWS              = 3,
    LWML_BRAND_GRID             = 4,   // Deprecated from API reporting. Keeping definition for backward compatibility.
    LWML_BRAND_GEFORCE          = 5,
    LWML_BRAND_TITAN            = 6,
    LWML_BRAND_LWIDIA_VAPPS     = 7,   // LWPU Virtual Applications
    LWML_BRAND_LWIDIA_VPC       = 8,   // LWPU Virtual PC
    LWML_BRAND_LWIDIA_VCS       = 9,   // LWPU Virtual Compute Server
    LWML_BRAND_LWIDIA_VWS       = 10,  // LWPU RTX Virtual Workstation
    LWML_BRAND_LWIDIA_VGAMING   = 11,  // LWPU vGaming
    LWML_BRAND_QUADRO_RTX       = 12,
    LWML_BRAND_LWIDIA_RTX       = 13,
    LWML_BRAND_LWIDIA           = 14,
    LWML_BRAND_GEFORCE_RTX      = 15,  // Unused
    LWML_BRAND_TITAN_RTX        = 16,  // Unused

    // Keep this last
    LWML_BRAND_COUNT
} lwmlBrandType_t;

/**
 * Temperature thresholds.
 */
typedef enum lwmlTemperatureThresholds_enum
{
    LWML_TEMPERATURE_THRESHOLD_SHUTDOWN      = 0, // Temperature at which the GPU will
                                                  // shut down for HW protection
    LWML_TEMPERATURE_THRESHOLD_SLOWDOWN      = 1, // Temperature at which the GPU will
                                                  // begin HW slowdown
    LWML_TEMPERATURE_THRESHOLD_MEM_MAX       = 2, // Memory Temperature at which the GPU will
                                                  // begin SW slowdown
    LWML_TEMPERATURE_THRESHOLD_GPU_MAX       = 3, // GPU Temperature at which the GPU
                                                  // can be throttled below base clock
    LWML_TEMPERATURE_THRESHOLD_ACOUSTIC_MIN  = 4, // Minimum GPU Temperature that can be
                                                  // set as acoustic threshold
    LWML_TEMPERATURE_THRESHOLD_ACOUSTIC_LWRR = 5, // Current temperature that is set as
                                                  // acoustic threshold.
    LWML_TEMPERATURE_THRESHOLD_ACOUSTIC_MAX  = 6, // Maximum GPU temperature that can be
                                                  // set as acoustic threshold.
    // Keep this last
    LWML_TEMPERATURE_THRESHOLD_COUNT
} lwmlTemperatureThresholds_t;

/** 
 * Temperature sensors. 
 */
typedef enum lwmlTemperatureSensors_enum 
{
    LWML_TEMPERATURE_GPU      = 0,    //!< Temperature sensor for the GPU die
    
    // Keep this last
    LWML_TEMPERATURE_COUNT
} lwmlTemperatureSensors_t;

/** 
 * Compute mode. 
 *
 * LWML_COMPUTEMODE_EXCLUSIVE_PROCESS was added in LWCA 4.0.
 * Earlier LWCA versions supported a single exclusive mode, 
 * which is equivalent to LWML_COMPUTEMODE_EXCLUSIVE_THREAD in LWCA 4.0 and beyond.
 */
typedef enum lwmlComputeMode_enum 
{
    LWML_COMPUTEMODE_DEFAULT           = 0,  //!< Default compute mode -- multiple contexts per device
    LWML_COMPUTEMODE_EXCLUSIVE_THREAD  = 1,  //!< Support Removed
    LWML_COMPUTEMODE_PROHIBITED        = 2,  //!< Compute-prohibited mode -- no contexts per device
    LWML_COMPUTEMODE_EXCLUSIVE_PROCESS = 3,  //!< Compute-exclusive-process mode -- only one context per device, usable from multiple threads at a time
    
    // Keep this last
    LWML_COMPUTEMODE_COUNT
} lwmlComputeMode_t;

/** 
 * ECC bit types.
 *
 * @deprecated See \ref lwmlMemoryErrorType_t for a more flexible type
 */
#define lwmlEccBitType_t lwmlMemoryErrorType_t

/**
 * Single bit ECC errors
 *
 * @deprecated Mapped to \ref LWML_MEMORY_ERROR_TYPE_CORRECTED
 */
#define LWML_SINGLE_BIT_ECC LWML_MEMORY_ERROR_TYPE_CORRECTED

/**
 * Double bit ECC errors
 *
 * @deprecated Mapped to \ref LWML_MEMORY_ERROR_TYPE_UNCORRECTED
 */
#define LWML_DOUBLE_BIT_ECC LWML_MEMORY_ERROR_TYPE_UNCORRECTED

/**
 * Memory error types
 */
typedef enum lwmlMemoryErrorType_enum
{
    /**
     * A memory error that was corrected
     * 
     * For ECC errors, these are single bit errors
     * For Texture memory, these are errors fixed by resend
     */
    LWML_MEMORY_ERROR_TYPE_CORRECTED = 0,
    /**
     * A memory error that was not corrected
     * 
     * For ECC errors, these are double bit errors
     * For Texture memory, these are errors where the resend fails
     */
    LWML_MEMORY_ERROR_TYPE_UNCORRECTED = 1,
    
    
    // Keep this last
    LWML_MEMORY_ERROR_TYPE_COUNT //!< Count of memory error types

} lwmlMemoryErrorType_t;

/** 
 * ECC counter types. 
 *
 * Note: Volatile counts are reset each time the driver loads. On Windows this is once per boot. On Linux this can be more frequent.
 *       On Linux the driver unloads when no active clients exist. If persistence mode is enabled or there is always a driver 
 *       client active (e.g. X11), then Linux also sees per-boot behavior. If not, volatile counts are reset each time a compute app
 *       is run.
 */
typedef enum lwmlEccCounterType_enum 
{
    LWML_VOLATILE_ECC      = 0,      //!< Volatile counts are reset each time the driver loads.
    LWML_AGGREGATE_ECC     = 1,      //!< Aggregate counts persist across reboots (i.e. for the lifetime of the device)
    
    // Keep this last
    LWML_ECC_COUNTER_TYPE_COUNT      //!< Count of memory counter types
} lwmlEccCounterType_t;

/** 
 * Clock types. 
 * 
 * All speeds are in Mhz.
 */
typedef enum lwmlClockType_enum 
{
    LWML_CLOCK_GRAPHICS  = 0,        //!< Graphics clock domain
    LWML_CLOCK_SM        = 1,        //!< SM clock domain
    LWML_CLOCK_MEM       = 2,        //!< Memory clock domain
    LWML_CLOCK_VIDEO     = 3,        //!< Video encoder/decoder clock domain
    
    // Keep this last
    LWML_CLOCK_COUNT //!< Count of clock types
} lwmlClockType_t;

/**
 * Clock Ids.  These are used in combination with lwmlClockType_t
 * to specify a single clock value.
 */
typedef enum lwmlClockId_enum
{
    LWML_CLOCK_ID_LWRRENT            = 0,   //!< Current actual clock value
    LWML_CLOCK_ID_APP_CLOCK_TARGET   = 1,   //!< Target application clock
    LWML_CLOCK_ID_APP_CLOCK_DEFAULT  = 2,   //!< Default application clock target
    LWML_CLOCK_ID_LWSTOMER_BOOST_MAX = 3,   //!< OEM-defined maximum clock rate

    //Keep this last
    LWML_CLOCK_ID_COUNT //!< Count of Clock Ids.
} lwmlClockId_t;

/** 
 * Driver models. 
 *
 * Windows only.
 */
typedef enum lwmlDriverModel_enum 
{
    LWML_DRIVER_WDDM      = 0,       //!< WDDM driver model -- GPU treated as a display device
    LWML_DRIVER_WDM       = 1        //!< WDM (TCC) model (recommended) -- GPU treated as a generic device
} lwmlDriverModel_t;

/**
 * Allowed PStates.
 */
typedef enum lwmlPStates_enum 
{
    LWML_PSTATE_0               = 0,       //!< Performance state 0 -- Maximum Performance
    LWML_PSTATE_1               = 1,       //!< Performance state 1 
    LWML_PSTATE_2               = 2,       //!< Performance state 2
    LWML_PSTATE_3               = 3,       //!< Performance state 3
    LWML_PSTATE_4               = 4,       //!< Performance state 4
    LWML_PSTATE_5               = 5,       //!< Performance state 5
    LWML_PSTATE_6               = 6,       //!< Performance state 6
    LWML_PSTATE_7               = 7,       //!< Performance state 7
    LWML_PSTATE_8               = 8,       //!< Performance state 8
    LWML_PSTATE_9               = 9,       //!< Performance state 9
    LWML_PSTATE_10              = 10,      //!< Performance state 10
    LWML_PSTATE_11              = 11,      //!< Performance state 11
    LWML_PSTATE_12              = 12,      //!< Performance state 12
    LWML_PSTATE_13              = 13,      //!< Performance state 13
    LWML_PSTATE_14              = 14,      //!< Performance state 14
    LWML_PSTATE_15              = 15,      //!< Performance state 15 -- Minimum Performance 
    LWML_PSTATE_UNKNOWN         = 32       //!< Unknown performance state
} lwmlPstates_t;

/**
 * GPU Operation Mode
 *
 * GOM allows to reduce power usage and optimize GPU throughput by disabling GPU features.
 *
 * Each GOM is designed to meet specific user needs.
 */
typedef enum lwmlGom_enum
{
    LWML_GOM_ALL_ON                    = 0, //!< Everything is enabled and running at full speed

    LWML_GOM_COMPUTE                   = 1, //!< Designed for running only compute tasks. Graphics operations
                                            //!< are not allowed

    LWML_GOM_LOW_DP                    = 2  //!< Designed for running graphics applications that don't require
                                            //!< high bandwidth double precision
} lwmlGpuOperationMode_t;

/** 
 * Available infoROM objects.
 */
typedef enum lwmlInforomObject_enum 
{
    LWML_INFOROM_OEM            = 0,       //!< An object defined by OEM
    LWML_INFOROM_ECC            = 1,       //!< The ECC object determining the level of ECC support
    LWML_INFOROM_POWER          = 2,       //!< The power management object

    // Keep this last
    LWML_INFOROM_COUNT                     //!< This counts the number of infoROM objects the driver knows about
} lwmlInforomObject_t;

/** 
 * Return values for LWML API calls. 
 */
typedef enum lwmlReturn_enum 
{
    // cppcheck-suppress *
    LWML_SUCCESS = 0,                        //!< The operation was successful
    LWML_ERROR_UNINITIALIZED = 1,            //!< LWML was not first initialized with lwmlInit()
    LWML_ERROR_ILWALID_ARGUMENT = 2,         //!< A supplied argument is invalid
    LWML_ERROR_NOT_SUPPORTED = 3,            //!< The requested operation is not available on target device
    LWML_ERROR_NO_PERMISSION = 4,            //!< The current user does not have permission for operation
    LWML_ERROR_ALREADY_INITIALIZED = 5,      //!< Deprecated: Multiple initializations are now allowed through ref counting
    LWML_ERROR_NOT_FOUND = 6,                //!< A query to find an object was unsuccessful
    LWML_ERROR_INSUFFICIENT_SIZE = 7,        //!< An input argument is not large enough
    LWML_ERROR_INSUFFICIENT_POWER = 8,       //!< A device's external power cables are not properly attached
    LWML_ERROR_DRIVER_NOT_LOADED = 9,        //!< LWPU driver is not loaded
    LWML_ERROR_TIMEOUT = 10,                 //!< User provided timeout passed
    LWML_ERROR_IRQ_ISSUE = 11,               //!< LWPU Kernel detected an interrupt issue with a GPU
    LWML_ERROR_LIBRARY_NOT_FOUND = 12,       //!< LWML Shared Library couldn't be found or loaded
    LWML_ERROR_FUNCTION_NOT_FOUND = 13,      //!< Local version of LWML doesn't implement this function
    LWML_ERROR_CORRUPTED_INFOROM = 14,       //!< infoROM is corrupted
    LWML_ERROR_GPU_IS_LOST = 15,             //!< The GPU has fallen off the bus or has otherwise become inaccessible
    LWML_ERROR_RESET_REQUIRED = 16,          //!< The GPU requires a reset before it can be used again
    LWML_ERROR_OPERATING_SYSTEM = 17,        //!< The GPU control device has been blocked by the operating system/cgroups
    LWML_ERROR_LIB_RM_VERSION_MISMATCH = 18, //!< RM detects a driver/library version mismatch
    LWML_ERROR_IN_USE = 19,                  //!< An operation cannot be performed because the GPU is lwrrently in use
    LWML_ERROR_MEMORY = 20,                  //!< Insufficient memory
    LWML_ERROR_NO_DATA = 21,                 //!< No data
    LWML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22,  //!< The requested vgpu operation is not available on target device, becasue ECC is enabled
    LWML_ERROR_INSUFFICIENT_RESOURCES = 23,  //!< Ran out of critical resources, other than memory
    LWML_ERROR_FREQ_NOT_SUPPORTED = 24,  //!< Ran out of critical resources, other than memory
    LWML_ERROR_UNKNOWN = 999                 //!< An internal driver error oclwrred
} lwmlReturn_t;

/**
 * See \ref lwmlDeviceGetMemoryErrorCounter
 */
typedef enum lwmlMemoryLocation_enum
{
    LWML_MEMORY_LOCATION_L1_CACHE        = 0,    //!< GPU L1 Cache
    LWML_MEMORY_LOCATION_L2_CACHE        = 1,    //!< GPU L2 Cache
    LWML_MEMORY_LOCATION_DRAM            = 2,    //!< Turing+ DRAM
    LWML_MEMORY_LOCATION_DEVICE_MEMORY   = 2,    //!< GPU Device Memory
    LWML_MEMORY_LOCATION_REGISTER_FILE   = 3,    //!< GPU Register File
    LWML_MEMORY_LOCATION_TEXTURE_MEMORY  = 4,    //!< GPU Texture Memory
    LWML_MEMORY_LOCATION_TEXTURE_SHM     = 5,    //!< Shared memory
    LWML_MEMORY_LOCATION_CBU             = 6,    //!< CBU
    LWML_MEMORY_LOCATION_SRAM            = 7,    //!< Turing+ SRAM
    // Keep this last
    LWML_MEMORY_LOCATION_COUNT              //!< This counts the number of memory locations the driver knows about
} lwmlMemoryLocation_t;

/**
 * Causes for page retirement
 */
typedef enum lwmlPageRetirementCause_enum
{
    LWML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS = 0, //!< Page was retired due to multiple single bit ECC error
    LWML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR = 1,           //!< Page was retired due to double bit ECC error

    // Keep this last
    LWML_PAGE_RETIREMENT_CAUSE_COUNT
} lwmlPageRetirementCause_t;

/**
 * API types that allow changes to default permission restrictions
 */
typedef enum lwmlRestrictedAPI_enum
{
    LWML_RESTRICTED_API_SET_APPLICATION_CLOCKS = 0,   //!< APIs that change application clocks, see lwmlDeviceSetApplicationsClocks 
                                                      //!< and see lwmlDeviceResetApplicationsClocks
    LWML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS = 1,  //!< APIs that enable/disable Auto Boosted clocks
                                                      //!< see lwmlDeviceSetAutoBoostedClocksEnabled
    // Keep this last
    LWML_RESTRICTED_API_COUNT
} lwmlRestrictedAPI_t;

/** @} */

/***************************************************************************************************/
/** @addtogroup virtualGPU
 *  @{
 */
/***************************************************************************************************/
/** @defgroup lwmlVirtualGpuEnums vGPU Enums
 *  @{
 */
/***************************************************************************************************/

/*!
 * GPU virtualization mode types.
 */
typedef enum lwmlGpuVirtualizationMode {
    LWML_GPU_VIRTUALIZATION_MODE_NONE = 0,  //!< Represents Bare Metal GPU
    LWML_GPU_VIRTUALIZATION_MODE_PASSTHROUGH = 1,  //!< Device is associated with GPU-Passthorugh
    LWML_GPU_VIRTUALIZATION_MODE_VGPU = 2,  //!< Device is associated with vGPU inside virtual machine.
    LWML_GPU_VIRTUALIZATION_MODE_HOST_VGPU = 3,  //!< Device is associated with VGX hypervisor in vGPU mode
    LWML_GPU_VIRTUALIZATION_MODE_HOST_VSGA = 4,  //!< Device is associated with VGX hypervisor in vSGA mode
} lwmlGpuVirtualizationMode_t;

/**
 * Host vGPU modes
 */
typedef enum lwmlHostVgpuMode_enum
{
    LWML_HOST_VGPU_MODE_NON_SRIOV    = 0,     //!< Non SR-IOV mode
    LWML_HOST_VGPU_MODE_SRIOV        = 1      //!< SR-IOV mode
} lwmlHostVgpuMode_t;

/*!
 * Types of VM identifiers
 */
typedef enum lwmlVgpuVmIdType {
    LWML_VGPU_VM_ID_DOMAIN_ID = 0, //!< VM ID represents DOMAIN ID
    LWML_VGPU_VM_ID_UUID = 1,      //!< VM ID represents UUID
} lwmlVgpuVmIdType_t;

/**
 * vGPU GUEST info state.
 */
typedef enum lwmlVgpuGuestInfoState_enum
{
    LWML_VGPU_INSTANCE_GUEST_INFO_STATE_UNINITIALIZED = 0,  //!< Guest-dependent fields uninitialized
    LWML_VGPU_INSTANCE_GUEST_INFO_STATE_INITIALIZED   = 1,  //!< Guest-dependent fields initialized
} lwmlVgpuGuestInfoState_t;

/**
 * vGPU software licensable features
 */
typedef enum {
    LWML_GRID_LICENSE_FEATURE_CODE_VGPU = 1,         //!< Virtual GPU
    LWML_GRID_LICENSE_FEATURE_CODE_VWORKSTATION = 2  //!< Virtual Workstation
} lwmlGridLicenseFeatureCode_t;

/** @} */

/***************************************************************************************************/

/** @defgroup lwmlVgpuConstants vGPU Constants
 *  @{
 */
/***************************************************************************************************/

/**
 * Buffer size guaranteed to be large enough for \ref lwmlVgpuTypeGetLicense
 */
#define LWML_GRID_LICENSE_BUFFER_SIZE       128

#define LWML_VGPU_NAME_BUFFER_SIZE          64

#define LWML_GRID_LICENSE_FEATURE_MAX_COUNT 3

#define ILWALID_GPU_INSTANCE_PROFILE_ID     0xFFFFFFFF

#define ILWALID_GPU_INSTANCE_ID             0xFFFFFFFF

/*!
 * Macros for vGPU instance's virtualization capabilities bitfield.
 */
#define LWML_VGPU_VIRTUALIZATION_CAP_MIGRATION         0:0
#define LWML_VGPU_VIRTUALIZATION_CAP_MIGRATION_NO      0x0
#define LWML_VGPU_VIRTUALIZATION_CAP_MIGRATION_YES     0x1

/*!
 * Macros for pGPU's virtualization capabilities bitfield.
 */
#define LWML_VGPU_PGPU_VIRTUALIZATION_CAP_MIGRATION         0:0
#define LWML_VGPU_PGPU_VIRTUALIZATION_CAP_MIGRATION_NO      0x0
#define LWML_VGPU_PGPU_VIRTUALIZATION_CAP_MIGRATION_YES     0x1

/** @} */

/***************************************************************************************************/
/** @defgroup lwmlVgpuStructs vGPU Structs
 *  @{
 */
/***************************************************************************************************/

typedef unsigned int lwmlVgpuTypeId_t;

typedef unsigned int lwmlVgpuInstance_t;

/**
 * Structure to store Utilization Value and vgpuInstance
 */
typedef struct lwmlVgpuInstanceUtilizationSample_st
{
    lwmlVgpuInstance_t vgpuInstance;    //!< vGPU Instance
    unsigned long long timeStamp;       //!< CPU Timestamp in microseconds
    lwmlValue_t smUtil;                 //!< SM (3D/Compute) Util Value
    lwmlValue_t memUtil;                //!< Frame Buffer Memory Util Value
    lwmlValue_t enlwtil;                //!< Encoder Util Value
    lwmlValue_t delwtil;                //!< Decoder Util Value
} lwmlVgpuInstanceUtilizationSample_t;

/**
 * Structure to store Utilization Value, vgpuInstance and subprocess information
 */
typedef struct lwmlVgpuProcessUtilizationSample_st
{
    lwmlVgpuInstance_t vgpuInstance;                //!< vGPU Instance
    unsigned int pid;                               //!< PID of process running within the vGPU VM
    char processName[LWML_VGPU_NAME_BUFFER_SIZE];   //!< Name of process running within the vGPU VM
    unsigned long long timeStamp;                   //!< CPU Timestamp in microseconds
    unsigned int smUtil;                            //!< SM (3D/Compute) Util Value
    unsigned int memUtil;                           //!< Frame Buffer Memory Util Value
    unsigned int enlwtil;                           //!< Encoder Util Value
    unsigned int delwtil;                           //!< Decoder Util Value
} lwmlVgpuProcessUtilizationSample_t;

/**
 * Structure to store utilization value and process Id
 */
typedef struct lwmlProcessUtilizationSample_st
{
    unsigned int pid;                   //!< PID of process
    unsigned long long timeStamp;       //!< CPU Timestamp in microseconds
    unsigned int smUtil;                //!< SM (3D/Compute) Util Value
    unsigned int memUtil;               //!< Frame Buffer Memory Util Value
    unsigned int enlwtil;               //!< Encoder Util Value
    unsigned int delwtil;               //!< Decoder Util Value
} lwmlProcessUtilizationSample_t;

/**
 * Structure containing vGPU software licensable feature information
 */
typedef struct lwmlGridLicensableFeature_st
{
    lwmlGridLicenseFeatureCode_t    featureCode;                                 //!< Licensed feature code
    unsigned int                    featureState;                                //!< Non-zero if feature is lwrrently licensed, otherwise zero
    char                            licenseInfo[LWML_GRID_LICENSE_BUFFER_SIZE];  //!< Deprecated.
    char                            productName[LWML_GRID_LICENSE_BUFFER_SIZE];
    unsigned int                    featureEnabled;                              //!< Non-zero if feature is enabled, otherwise zero
} lwmlGridLicensableFeature_t;

/**
 * Structure to store vGPU software licensable features
 */
typedef struct lwmlGridLicensableFeatures_st
{
    int                         isGridLicenseSupported;                                       //!< Non-zero if vGPU Software Licensing is supported on the system, otherwise zero
    unsigned int                licensableFeaturesCount;                                      //!< Entries returned in \a gridLicensableFeatures array
    lwmlGridLicensableFeature_t gridLicensableFeatures[LWML_GRID_LICENSE_FEATURE_MAX_COUNT];  //!< Array of vGPU software licensable features.
} lwmlGridLicensableFeatures_t;

/**
 * Simplified chip architecture
 */
#define LWML_DEVICE_ARCH_KEPLER    2 // Devices based on the LWPU Kepler architecture
#define LWML_DEVICE_ARCH_MAXWELL   3 // Devices based on the LWPU Maxwell architecture
#define LWML_DEVICE_ARCH_PASCAL    4 // Devices based on the LWPU Pascal architecture
#define LWML_DEVICE_ARCH_VOLTA     5 // Devices based on the LWPU Volta architecture
#define LWML_DEVICE_ARCH_TURING    6 // Devices based on the LWPU Turing architecture

#define LWML_DEVICE_ARCH_AMPERE    7 // Devices based on the LWPU Ampere architecture

#define LWML_DEVICE_ARCH_HOPPER    8 // Devices based on the LWPU Hopper architecture

#define LWML_DEVICE_ARCH_UNKNOWN   0xffffffff // Anything else, presumably something newer

typedef unsigned int lwmlDeviceArchitecture_t;

/** @} */
/** @} */

/***************************************************************************************************/
/** @defgroup lwmlFieldValueEnums Field Value Enums
 *  @{
 */
/***************************************************************************************************/

/**
 * Field Identifiers.
 *
 * All Identifiers pertain to a device. Each ID is only used once and is guaranteed never to change.
 */
#define LWML_FI_DEV_ECC_LWRRENT           1   //!< Current ECC mode. 1=Active. 0=Inactive
#define LWML_FI_DEV_ECC_PENDING           2   //!< Pending ECC mode. 1=Active. 0=Inactive
/* ECC Count Totals */
#define LWML_FI_DEV_ECC_SBE_VOL_TOTAL     3   //!< Total single bit volatile ECC errors
#define LWML_FI_DEV_ECC_DBE_VOL_TOTAL     4   //!< Total double bit volatile ECC errors
#define LWML_FI_DEV_ECC_SBE_AGG_TOTAL     5   //!< Total single bit aggregate (persistent) ECC errors
#define LWML_FI_DEV_ECC_DBE_AGG_TOTAL     6   //!< Total double bit aggregate (persistent) ECC errors
/* Individual ECC locations */
#define LWML_FI_DEV_ECC_SBE_VOL_L1        7   //!< L1 cache single bit volatile ECC errors
#define LWML_FI_DEV_ECC_DBE_VOL_L1        8   //!< L1 cache double bit volatile ECC errors
#define LWML_FI_DEV_ECC_SBE_VOL_L2        9   //!< L2 cache single bit volatile ECC errors
#define LWML_FI_DEV_ECC_DBE_VOL_L2        10  //!< L2 cache double bit volatile ECC errors
#define LWML_FI_DEV_ECC_SBE_VOL_DEV       11  //!< Device memory single bit volatile ECC errors
#define LWML_FI_DEV_ECC_DBE_VOL_DEV       12  //!< Device memory double bit volatile ECC errors
#define LWML_FI_DEV_ECC_SBE_VOL_REG       13  //!< Register file single bit volatile ECC errors
#define LWML_FI_DEV_ECC_DBE_VOL_REG       14  //!< Register file double bit volatile ECC errors
#define LWML_FI_DEV_ECC_SBE_VOL_TEX       15  //!< Texture memory single bit volatile ECC errors
#define LWML_FI_DEV_ECC_DBE_VOL_TEX       16  //!< Texture memory double bit volatile ECC errors
#define LWML_FI_DEV_ECC_DBE_VOL_CBU       17  //!< CBU double bit volatile ECC errors
#define LWML_FI_DEV_ECC_SBE_AGG_L1        18  //!< L1 cache single bit aggregate (persistent) ECC errors
#define LWML_FI_DEV_ECC_DBE_AGG_L1        19  //!< L1 cache double bit aggregate (persistent) ECC errors
#define LWML_FI_DEV_ECC_SBE_AGG_L2        20  //!< L2 cache single bit aggregate (persistent) ECC errors
#define LWML_FI_DEV_ECC_DBE_AGG_L2        21  //!< L2 cache double bit aggregate (persistent) ECC errors
#define LWML_FI_DEV_ECC_SBE_AGG_DEV       22  //!< Device memory single bit aggregate (persistent) ECC errors
#define LWML_FI_DEV_ECC_DBE_AGG_DEV       23  //!< Device memory double bit aggregate (persistent) ECC errors
#define LWML_FI_DEV_ECC_SBE_AGG_REG       24  //!< Register File single bit aggregate (persistent) ECC errors
#define LWML_FI_DEV_ECC_DBE_AGG_REG       25  //!< Register File double bit aggregate (persistent) ECC errors
#define LWML_FI_DEV_ECC_SBE_AGG_TEX       26  //!< Texture memory single bit aggregate (persistent) ECC errors
#define LWML_FI_DEV_ECC_DBE_AGG_TEX       27  //!< Texture memory double bit aggregate (persistent) ECC errors
#define LWML_FI_DEV_ECC_DBE_AGG_CBU       28  //!< CBU double bit aggregate ECC errors

/* Page Retirement */
#define LWML_FI_DEV_RETIRED_SBE           29  //!< Number of retired pages because of single bit errors
#define LWML_FI_DEV_RETIRED_DBE           30  //!< Number of retired pages because of double bit errors
#define LWML_FI_DEV_RETIRED_PENDING       31  //!< If any pages are pending retirement. 1=yes. 0=no.

/* LwLink Flit Error Counters */
#define LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L0    32 //!< LWLink flow control CRC  Error Counter for Lane 0
#define LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L1    33 //!< LWLink flow control CRC  Error Counter for Lane 1
#define LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L2    34 //!< LWLink flow control CRC  Error Counter for Lane 2
#define LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L3    35 //!< LWLink flow control CRC  Error Counter for Lane 3
#define LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L4    36 //!< LWLink flow control CRC  Error Counter for Lane 4
#define LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L5    37 //!< LWLink flow control CRC  Error Counter for Lane 5
#define LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL 38 //!< LWLink flow control CRC  Error Counter total for all Lanes

/* LwLink CRC Data Error Counters */
#define LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L0    39 //!< LWLink data CRC Error Counter for Lane 0
#define LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L1    40 //!< LWLink data CRC Error Counter for Lane 1
#define LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L2    41 //!< LWLink data CRC Error Counter for Lane 2
#define LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L3    42 //!< LWLink data CRC Error Counter for Lane 3
#define LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L4    43 //!< LWLink data CRC Error Counter for Lane 4
#define LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L5    44 //!< LWLink data CRC Error Counter for Lane 5
#define LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_TOTAL 45 //!< LwLink data CRC Error Counter total for all Lanes

/* LwLink Replay Error Counters */
#define LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L0      46 //!< LWLink Replay Error Counter for Lane 0
#define LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L1      47 //!< LWLink Replay Error Counter for Lane 1
#define LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L2      48 //!< LWLink Replay Error Counter for Lane 2
#define LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L3      49 //!< LWLink Replay Error Counter for Lane 3
#define LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L4      50 //!< LWLink Replay Error Counter for Lane 4
#define LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L5      51 //!< LWLink Replay Error Counter for Lane 5
#define LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_TOTAL   52 //!< LWLink Replay Error Counter total for all Lanes

/* LwLink Recovery Error Counters */
#define LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L0    53 //!< LWLink Recovery Error Counter for Lane 0
#define LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L1    54 //!< LWLink Recovery Error Counter for Lane 1
#define LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L2    55 //!< LWLink Recovery Error Counter for Lane 2
#define LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L3    56 //!< LWLink Recovery Error Counter for Lane 3
#define LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L4    57 //!< LWLink Recovery Error Counter for Lane 4
#define LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L5    58 //!< LWLink Recovery Error Counter for Lane 5
#define LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_TOTAL 59 //!< LWLink Recovery Error Counter total for all Lanes

/* LwLink Bandwidth Counters */
/*
 * LWML_FI_DEV_LWLINK_BANDWIDTH_* field values are now deprecated.
 * Please use the following field values instead:
 * LWML_FI_DEV_LWLINK_THROUGHPUT_DATA_TX
 * LWML_FI_DEV_LWLINK_THROUGHPUT_DATA_RX
 * LWML_FI_DEV_LWLINK_THROUGHPUT_RAW_TX
 * LWML_FI_DEV_LWLINK_THROUGHPUT_RAW_RX
 */
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L0     60 //!< LWLink Bandwidth Counter for Counter Set 0, Lane 0
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L1     61 //!< LWLink Bandwidth Counter for Counter Set 0, Lane 1
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L2     62 //!< LWLink Bandwidth Counter for Counter Set 0, Lane 2
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L3     63 //!< LWLink Bandwidth Counter for Counter Set 0, Lane 3
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L4     64 //!< LWLink Bandwidth Counter for Counter Set 0, Lane 4
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L5     65 //!< LWLink Bandwidth Counter for Counter Set 0, Lane 5
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C0_TOTAL  66 //!< LWLink Bandwidth Counter Total for Counter Set 0, All Lanes

/* LwLink Bandwidth Counters */
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C1_L0     67 //!< LWLink Bandwidth Counter for Counter Set 1, Lane 0
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C1_L1     68 //!< LWLink Bandwidth Counter for Counter Set 1, Lane 1
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C1_L2     69 //!< LWLink Bandwidth Counter for Counter Set 1, Lane 2
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C1_L3     70 //!< LWLink Bandwidth Counter for Counter Set 1, Lane 3
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C1_L4     71 //!< LWLink Bandwidth Counter for Counter Set 1, Lane 4
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C1_L5     72 //!< LWLink Bandwidth Counter for Counter Set 1, Lane 5
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C1_TOTAL  73 //!< LWLink Bandwidth Counter Total for Counter Set 1, All Lanes

/* LWML Perf Policy Counters */
#define LWML_FI_DEV_PERF_POLICY_POWER              74   //!< Perf Policy Counter for Power Policy
#define LWML_FI_DEV_PERF_POLICY_THERMAL            75   //!< Perf Policy Counter for Thermal Policy
#define LWML_FI_DEV_PERF_POLICY_SYNC_BOOST         76   //!< Perf Policy Counter for Sync boost Policy
#define LWML_FI_DEV_PERF_POLICY_BOARD_LIMIT        77   //!< Perf Policy Counter for Board Limit
#define LWML_FI_DEV_PERF_POLICY_LOW_UTILIZATION    78   //!< Perf Policy Counter for Low GPU Utilization Policy
#define LWML_FI_DEV_PERF_POLICY_RELIABILITY        79   //!< Perf Policy Counter for Reliability Policy
#define LWML_FI_DEV_PERF_POLICY_TOTAL_APP_CLOCKS   80   //!< Perf Policy Counter for Total App Clock Policy
#define LWML_FI_DEV_PERF_POLICY_TOTAL_BASE_CLOCKS  81   //!< Perf Policy Counter for Total Base Clocks Policy

/* Memory temperatures */
#define LWML_FI_DEV_MEMORY_TEMP  82 //!< Memory temperature for the device

/* Energy Counter */
#define LWML_FI_DEV_TOTAL_ENERGY_CONSUMPTION 83 //!< Total energy consumption for the GPU in mJ since the driver was last reloaded

/* LWLink Speed */
#define LWML_FI_DEV_LWLINK_SPEED_MBPS_L0     84  //!< LWLink Speed in MBps for Link 0
#define LWML_FI_DEV_LWLINK_SPEED_MBPS_L1     85  //!< LWLink Speed in MBps for Link 1
#define LWML_FI_DEV_LWLINK_SPEED_MBPS_L2     86  //!< LWLink Speed in MBps for Link 2
#define LWML_FI_DEV_LWLINK_SPEED_MBPS_L3     87  //!< LWLink Speed in MBps for Link 3
#define LWML_FI_DEV_LWLINK_SPEED_MBPS_L4     88  //!< LWLink Speed in MBps for Link 4
#define LWML_FI_DEV_LWLINK_SPEED_MBPS_L5     89  //!< LWLink Speed in MBps for Link 5
#define LWML_FI_DEV_LWLINK_SPEED_MBPS_COMMON 90  //!< Common LWLink Speed in MBps for active links

#define LWML_FI_DEV_LWLINK_LINK_COUNT        91  //!< Number of LWLinks present on the device

#define LWML_FI_DEV_RETIRED_PENDING_SBE      92  //!< If any pages are pending retirement due to SBE. 1=yes. 0=no.
#define LWML_FI_DEV_RETIRED_PENDING_DBE      93  //!< If any pages are pending retirement due to DBE. 1=yes. 0=no.

#define LWML_FI_DEV_PCIE_REPLAY_COUNTER             94  //!< PCIe replay counter
#define LWML_FI_DEV_PCIE_REPLAY_ROLLOVER_COUNTER    95  //!< PCIe replay rollover counter

/* LwLink Flit Error Counters */
#define LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L6     96 //!< LWLink flow control CRC  Error Counter for Lane 6
#define LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L7     97 //!< LWLink flow control CRC  Error Counter for Lane 7
#define LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L8     98 //!< LWLink flow control CRC  Error Counter for Lane 8
#define LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L9     99 //!< LWLink flow control CRC  Error Counter for Lane 9
#define LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L10   100 //!< LWLink flow control CRC  Error Counter for Lane 10
#define LWML_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L11   101 //!< LWLink flow control CRC  Error Counter for Lane 11

/* LwLink CRC Data Error Counters */
#define LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L6    102 //!< LWLink data CRC Error Counter for Lane 6
#define LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L7    103 //!< LWLink data CRC Error Counter for Lane 7
#define LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L8    104 //!< LWLink data CRC Error Counter for Lane 8
#define LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L9    105 //!< LWLink data CRC Error Counter for Lane 9
#define LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L10   106 //!< LWLink data CRC Error Counter for Lane 10
#define LWML_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L11   107 //!< LWLink data CRC Error Counter for Lane 11

/* LwLink Replay Error Counters */
#define LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L6      108 //!< LWLink Replay Error Counter for Lane 6
#define LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L7      109 //!< LWLink Replay Error Counter for Lane 7
#define LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L8      110 //!< LWLink Replay Error Counter for Lane 8
#define LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L9      111 //!< LWLink Replay Error Counter for Lane 9
#define LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L10     112 //!< LWLink Replay Error Counter for Lane 10
#define LWML_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L11     113 //!< LWLink Replay Error Counter for Lane 11

/* LwLink Recovery Error Counters */
#define LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L6    114 //!< LWLink Recovery Error Counter for Lane 6
#define LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L7    115 //!< LWLink Recovery Error Counter for Lane 7
#define LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L8    116 //!< LWLink Recovery Error Counter for Lane 8
#define LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L9    117 //!< LWLink Recovery Error Counter for Lane 9
#define LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L10   118 //!< LWLink Recovery Error Counter for Lane 10
#define LWML_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L11   119 //!< LWLink Recovery Error Counter for Lane 11

/* LwLink Bandwidth Counters */
/*
 * LWML_FI_DEV_LWLINK_BANDWIDTH_* field values are now deprecated.
 * Please use the following field values instead:
 * LWML_FI_DEV_LWLINK_THROUGHPUT_DATA_TX
 * LWML_FI_DEV_LWLINK_THROUGHPUT_DATA_RX
 * LWML_FI_DEV_LWLINK_THROUGHPUT_RAW_TX
 * LWML_FI_DEV_LWLINK_THROUGHPUT_RAW_RX
 */
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L6     120 //!< LWLink Bandwidth Counter for Counter Set 0, Lane 6
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L7     121 //!< LWLink Bandwidth Counter for Counter Set 0, Lane 7
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L8     122 //!< LWLink Bandwidth Counter for Counter Set 0, Lane 8
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L9     123 //!< LWLink Bandwidth Counter for Counter Set 0, Lane 9
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L10    124 //!< LWLink Bandwidth Counter for Counter Set 0, Lane 10
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C0_L11    125 //!< LWLink Bandwidth Counter for Counter Set 0, Lane 11

/* LwLink Bandwidth Counters */
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C1_L6     126 //!< LWLink Bandwidth Counter for Counter Set 1, Lane 6
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C1_L7     127 //!< LWLink Bandwidth Counter for Counter Set 1, Lane 7
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C1_L8     128 //!< LWLink Bandwidth Counter for Counter Set 1, Lane 8
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C1_L9     129 //!< LWLink Bandwidth Counter for Counter Set 1, Lane 9
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C1_L10    130 //!< LWLink Bandwidth Counter for Counter Set 1, Lane 10
#define LWML_FI_DEV_LWLINK_BANDWIDTH_C1_L11    131 //!< LWLink Bandwidth Counter for Counter Set 1, Lane 11

/* LWLink Speed */
#define LWML_FI_DEV_LWLINK_SPEED_MBPS_L6     132  //!< LWLink Speed in MBps for Link 6
#define LWML_FI_DEV_LWLINK_SPEED_MBPS_L7     133  //!< LWLink Speed in MBps for Link 7
#define LWML_FI_DEV_LWLINK_SPEED_MBPS_L8     134  //!< LWLink Speed in MBps for Link 8
#define LWML_FI_DEV_LWLINK_SPEED_MBPS_L9     135  //!< LWLink Speed in MBps for Link 9
#define LWML_FI_DEV_LWLINK_SPEED_MBPS_L10    136  //!< LWLink Speed in MBps for Link 10
#define LWML_FI_DEV_LWLINK_SPEED_MBPS_L11    137  //!< LWLink Speed in MBps for Link 11

/**
 * LWLink throughput counters field values
 *
 * Link ID needs to be specified in the scopeId field in lwmlFieldValue_t.
 * A scopeId of UINT_MAX returns aggregate value summed up across all links
 * for the specified counter type in fieldId.
 */
#define LWML_FI_DEV_LWLINK_THROUGHPUT_DATA_TX      138 //!< LWLink TX Data throughput in KiB
#define LWML_FI_DEV_LWLINK_THROUGHPUT_DATA_RX      139 //!< LWLink RX Data throughput in KiB
#define LWML_FI_DEV_LWLINK_THROUGHPUT_RAW_TX       140 //!< LWLink TX Data + protocol overhead in KiB
#define LWML_FI_DEV_LWLINK_THROUGHPUT_RAW_RX       141 //!< LWLink RX Data + protocol overhead in KiB

/* Row Remapper */
#define LWML_FI_DEV_REMAPPED_COR        142 //!< Number of remapped rows due to correctable errors
#define LWML_FI_DEV_REMAPPED_UNC        143 //!< Number of remapped rows due to uncorrectable errors
#define LWML_FI_DEV_REMAPPED_PENDING    144 //!< If any rows are pending remapping. 1=yes 0=no
#define LWML_FI_DEV_REMAPPED_FAILURE    145 //!< If any rows failed to be remapped 1=yes 0=no

/**
 * Remote device LWLink ID
 *
 * Link ID needs to be specified in the scopeId field in lwmlFieldValue_t.
 */
#define LWML_FI_DEV_LWLINK_REMOTE_LWLINK_ID     146 //!< Remote device LWLink ID

/**
 * LWSwitch: connected LWLink count
 */
#define LWML_FI_DEV_LWSWITCH_CONNECTED_LINK_COUNT   147  //!< Number of LWLinks connected to LWSwitch

#define LWML_FI_MAX 148 //!< One greater than the largest field ID defined above

/**
 * Information for a Field Value Sample
 */
typedef struct lwmlFieldValue_st
{
    unsigned int fieldId;       //!< ID of the LWML field to retrieve. This must be set before any call that uses this struct. See the constants starting with LWML_FI_ above.
    unsigned int scopeId;       //!< Scope ID can represent data used by LWML depending on fieldId's context. For example, for LWLink throughput counter data, scopeId can represent linkId.
    long long timestamp;        //!< CPU Timestamp of this value in microseconds since 1970
    long long latencyUsec;      //!< How long this field value took to update (in usec) within LWML. This may be averaged across several fields that are serviced by the same driver call.
    lwmlValueType_t valueType;  //!< Type of the value stored in value
    lwmlReturn_t lwmlReturn;    //!< Return code for retrieving this value. This must be checked before looking at value, as value is undefined if lwmlReturn != LWML_SUCCESS
    lwmlValue_t value;          //!< Value for this field. This is only valid if lwmlReturn == LWML_SUCCESS
} lwmlFieldValue_t;


/** @} */

/***************************************************************************************************/
/** @defgroup lwmlUnitStructs Unit Structs
 *  @{
 */
/***************************************************************************************************/

typedef struct lwmlUnit_st* lwmlUnit_t;

/** 
 * Description of HWBC entry 
 */
typedef struct lwmlHwbcEntry_st 
{
    unsigned int hwbcId;
    char firmwareVersion[32];
} lwmlHwbcEntry_t;

/** 
 * Fan state enum. 
 */
typedef enum lwmlFanState_enum 
{
    LWML_FAN_NORMAL       = 0,     //!< Fan is working properly
    LWML_FAN_FAILED       = 1      //!< Fan has failed
} lwmlFanState_t;

/** 
 * Led color enum. 
 */
typedef enum lwmlLedColor_enum 
{
    LWML_LED_COLOR_GREEN       = 0,     //!< GREEN, indicates good health
    LWML_LED_COLOR_AMBER       = 1      //!< AMBER, indicates problem
} lwmlLedColor_t;


/** 
 * LED states for an S-class unit.
 */
typedef struct lwmlLedState_st 
{
    char cause[256];               //!< If amber, a text description of the cause
    lwmlLedColor_t color;          //!< GREEN or AMBER
} lwmlLedState_t;

/** 
 * Static S-class unit info.
 */
typedef struct lwmlUnitInfo_st 
{
    char name[96];                      //!< Product name
    char id[96];                        //!< Product identifier
    char serial[96];                    //!< Product serial number
    char firmwareVersion[96];           //!< Firmware version
} lwmlUnitInfo_t;

/** 
 * Power usage information for an S-class unit.
 * The power supply state is a human readable string that equals "Normal" or contains
 * a combination of "Abnormal" plus one or more of the following:
 *    
 *    - High voltage
 *    - Fan failure
 *    - Heatsink temperature
 *    - Current limit
 *    - Voltage below UV alarm threshold
 *    - Low-voltage
 *    - SI2C remote off command
 *    - MOD_DISABLE input
 *    - Short pin transition 
*/
typedef struct lwmlPSUInfo_st 
{
    char state[256];                 //!< The power supply state
    unsigned int current;            //!< PSU current (A)
    unsigned int voltage;            //!< PSU voltage (V)
    unsigned int power;              //!< PSU power draw (W)
} lwmlPSUInfo_t;

/** 
 * Fan speed reading for a single fan in an S-class unit.
 */
typedef struct lwmlUnitFanInfo_st 
{
    unsigned int speed;              //!< Fan speed (RPM)
    lwmlFanState_t state;            //!< Flag that indicates whether fan is working properly
} lwmlUnitFanInfo_t;

/** 
 * Fan speed readings for an entire S-class unit.
 */
typedef struct lwmlUnitFanSpeeds_st 
{
    lwmlUnitFanInfo_t fans[24];      //!< Fan speed data for each fan
    unsigned int count;              //!< Number of fans in unit
} lwmlUnitFanSpeeds_t;

/** @} */

/***************************************************************************************************/
/** @addtogroup lwmlEvents 
 *  @{
 */
/***************************************************************************************************/

/** 
 * Handle to an event set
 */
typedef struct lwmlEventSet_st* lwmlEventSet_t;

/** @defgroup lwmlEventType Event Types
 * @{
 * Event Types which user can be notified about.
 * See description of particular functions for details.
 *
 * See \ref lwmlDeviceRegisterEvents and \ref lwmlDeviceGetSupportedEventTypes to check which devices 
 * support each event.
 *
 * Types can be combined with bitwise or operator '|' when passed to \ref lwmlDeviceRegisterEvents
 */
//! Event about single bit ECC errors
/**
 * \note A corrected texture memory error is not an ECC error, so it does not generate a single bit event
 */
#define lwmlEventTypeSingleBitEccError     0x0000000000000001LL

//! Event about double bit ECC errors
/**
 * \note An uncorrected texture memory error is not an ECC error, so it does not generate a double bit event
 */
#define lwmlEventTypeDoubleBitEccError     0x0000000000000002LL

//! Event about PState changes
/**
 *  \note On Fermi architecture PState changes are also an indicator that GPU is throttling down due to
 *  no work being exelwted on the GPU, power capping or thermal capping. In a typical situation,
 *  Fermi-based GPU should stay in P0 for the duration of the exelwtion of the compute process.
 */
#define lwmlEventTypePState                0x0000000000000004LL

//! Event that Xid critical error oclwrred
#define lwmlEventTypeXidCriticalError      0x0000000000000008LL

//! Event about clock changes
/**
 * Kepler only
 */
#define lwmlEventTypeClock                 0x0000000000000010LL

//! Event about AC/Battery power source changes
#define lwmlEventTypePowerSourceChange     0x0000000000000080LL

//! Event about MIG configuration changes
#define lwmlEventMigConfigChange           0x0000000000000100LL

//! Mask with no events
#define lwmlEventTypeNone                  0x0000000000000000LL

//! Mask of all events
#define lwmlEventTypeAll (lwmlEventTypeNone    \
        | lwmlEventTypeSingleBitEccError       \
        | lwmlEventTypeDoubleBitEccError       \
        | lwmlEventTypePState                  \
        | lwmlEventTypeClock                   \
        | lwmlEventTypeXidCriticalError        \
        | lwmlEventTypePowerSourceChange       \
        | lwmlEventMigConfigChange             \
        )
/** @} */

/** 
 * Information about oclwrred event
 */
typedef struct lwmlEventData_st
{
    lwmlDevice_t        device;             //!< Specific device where the event oclwrred
    unsigned long long  eventType;          //!< Information about what specific event oclwrred
    unsigned long long  eventData;          //!< Stores XID error for the device in the event of lwmlEventTypeXidCriticalError,
                                            //   eventData is 0 for any other event. eventData is set as 999 for unknown xid error.
    unsigned int        gpuInstanceId;      //!< If MIG is enabled and lwmlEventTypeXidCriticalError event is attributable to a GPU
                                            //   instance, stores a valid GPU instance ID. gpuInstanceId is set to 0xFFFFFFFF
                                            //   otherwise.
    unsigned int        computeInstanceId;  //!< If MIG is enabled and lwmlEventTypeXidCriticalError event is attributable to a
                                            //   compute instance, stores a valid compute instance ID. computeInstanceId is set to
                                            //   0xFFFFFFFF otherwise.
} lwmlEventData_t;

/** @} */

/***************************************************************************************************/
/** @addtogroup lwmlClocksThrottleReasons
 *  @{
 */
/***************************************************************************************************/

/** Nothing is running on the GPU and the clocks are dropping to Idle state
 * \note This limiter may be removed in a later release
 */
#define lwmlClocksThrottleReasonGpuIdle                   0x0000000000000001LL

/** GPU clocks are limited by current setting of applications clocks
 *
 * @see lwmlDeviceSetApplicationsClocks
 * @see lwmlDeviceGetApplicationsClock
 */
#define lwmlClocksThrottleReasonApplicationsClocksSetting 0x0000000000000002LL

/** 
 * @deprecated Renamed to \ref lwmlClocksThrottleReasonApplicationsClocksSetting 
 *             as the name describes the situation more aclwrately.
 */
#define lwmlClocksThrottleReasonUserDefinedClocks         lwmlClocksThrottleReasonApplicationsClocksSetting 

/** SW Power Scaling algorithm is reducing the clocks below requested clocks 
 *
 * @see lwmlDeviceGetPowerUsage
 * @see lwmlDeviceSetPowerManagementLimit
 * @see lwmlDeviceGetPowerManagementLimit
 */
#define lwmlClocksThrottleReasonSwPowerCap                0x0000000000000004LL

/** HW Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
 * 
 * This is an indicator of:
 *   - temperature being too high
 *   - External Power Brake Assertion is triggered (e.g. by the system power supply)
 *   - Power draw is too high and Fast Trigger protection is reducing the clocks
 *   - May be also reported during PState or clock change
 *      - This behavior may be removed in a later release.
 *
 * @see lwmlDeviceGetTemperature
 * @see lwmlDeviceGetTemperatureThreshold
 * @see lwmlDeviceGetPowerUsage
 */
#define lwmlClocksThrottleReasonHwSlowdown                0x0000000000000008LL

/** Sync Boost
 *
 * This GPU has been added to a Sync boost group with lwpu-smi or DCGM in
 * order to maximize performance per watt. All GPUs in the sync boost group
 * will boost to the minimum possible clocks across the entire group. Look at
 * the throttle reasons for other GPUs in the system to see why those GPUs are
 * holding this one at lower clocks.
 *
 */
#define lwmlClocksThrottleReasonSyncBoost                 0x0000000000000010LL

/** SW Thermal Slowdown
 *
 * This is an indicator of one or more of the following:
 *  - Current GPU temperature above the GPU Max Operating Temperature
 *  - Current memory temperature above the Memory Max Operating Temperature
 *
 */
#define lwmlClocksThrottleReasonSwThermalSlowdown         0x0000000000000020LL

/** HW Thermal Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
 * 
 * This is an indicator of:
 *   - temperature being too high
 *
 * @see lwmlDeviceGetTemperature
 * @see lwmlDeviceGetTemperatureThreshold
 * @see lwmlDeviceGetPowerUsage
 */
#define lwmlClocksThrottleReasonHwThermalSlowdown         0x0000000000000040LL

/** HW Power Brake Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
 * 
 * This is an indicator of:
 *   - External Power Brake Assertion being triggered (e.g. by the system power supply)
 *
 * @see lwmlDeviceGetTemperature
 * @see lwmlDeviceGetTemperatureThreshold
 * @see lwmlDeviceGetPowerUsage
 */
#define lwmlClocksThrottleReasonHwPowerBrakeSlowdown      0x0000000000000080LL

/** GPU clocks are limited by current setting of Display clocks
 *
 * @see bug 1997531
 */
#define lwmlClocksThrottleReasonDisplayClockSetting       0x0000000000000100LL

/** Bit mask representing no clocks throttling
 *
 * Clocks are as high as possible.
 * */
#define lwmlClocksThrottleReasonNone                      0x0000000000000000LL

/** Bit mask representing all supported clocks throttling reasons 
 * New reasons might be added to this list in the future
 */
#define lwmlClocksThrottleReasonAll (lwmlClocksThrottleReasonNone \
      | lwmlClocksThrottleReasonGpuIdle                           \
      | lwmlClocksThrottleReasonApplicationsClocksSetting         \
      | lwmlClocksThrottleReasonSwPowerCap                        \
      | lwmlClocksThrottleReasonHwSlowdown                        \
      | lwmlClocksThrottleReasonSyncBoost                         \
      | lwmlClocksThrottleReasonSwThermalSlowdown                 \
      | lwmlClocksThrottleReasonHwThermalSlowdown                 \
      | lwmlClocksThrottleReasonHwPowerBrakeSlowdown              \
      | lwmlClocksThrottleReasonDisplayClockSetting               \
)
/** @} */

/***************************************************************************************************/
/** @defgroup lwmlAccountingStats Accounting Statistics
 *  @{
 *
 *  Set of APIs designed to provide per process information about usage of GPU.
 *
 *  @note All accounting statistics and accounting mode live in lwpu driver and reset 
 *        to default (Disabled) when driver unloads.
 *        It is advised to run with persistence mode enabled.
 *
 *  @note Enabling accounting mode has no negative impact on the GPU performance.
 */
/***************************************************************************************************/

/**
 * Describes accounting statistics of a process.
 */
typedef struct lwmlAccountingStats_st {
    unsigned int gpuUtilization;                //!< Percent of time over the process's lifetime during which one or more kernels was exelwting on the GPU.
                                                //! Utilization stats just like returned by \ref lwmlDeviceGetUtilizationRates but for the life time of a
                                                //! process (not just the last sample period).
                                                //! Set to LWML_VALUE_NOT_AVAILABLE if lwmlDeviceGetUtilizationRates is not supported
    
    unsigned int memoryUtilization;             //!< Percent of time over the process's lifetime during which global (device) memory was being read or written.
                                                //! Set to LWML_VALUE_NOT_AVAILABLE if lwmlDeviceGetUtilizationRates is not supported
    
    unsigned long long maxMemoryUsage;          //!< Maximum total memory in bytes that was ever allocated by the process.
                                                //! Set to LWML_VALUE_NOT_AVAILABLE if lwmlProcessInfo_t->usedGpuMemory is not supported
    

    unsigned long long time;                    //!< Amount of time in ms during which the compute context was active. The time is reported as 0 if 
                                                //!< the process is not terminated
    
    unsigned long long startTime;               //!< CPU Timestamp in usec representing start time for the process
    
    unsigned int isRunning;                     //!< Flag to represent if the process is running (1 for running, 0 for terminated)

    unsigned int reserved[5];                   //!< Reserved for future use
} lwmlAccountingStats_t;

/** @} */

/***************************************************************************************************/
/** @defgroup lwmlEncoderStructs Encoder Structs
 *  @{
 */
/***************************************************************************************************/

/**
 * Represents type of encoder for capacity can be queried
 */
typedef enum lwmlEncoderQueryType_enum
{
    LWML_ENCODER_QUERY_H264 = 0,        //!< H264 encoder
    LWML_ENCODER_QUERY_HEVC = 1,        //!< HEVC encoder
}lwmlEncoderType_t;

/**
 * Structure to hold encoder session data
 */
typedef struct lwmlEncoderSessionInfo_st
{
    unsigned int       sessionId;       //!< Unique session ID
    unsigned int       pid;             //!< Owning process ID
    lwmlVgpuInstance_t vgpuInstance;    //!< Owning vGPU instance ID (only valid on vGPU hosts, otherwise zero)
    lwmlEncoderType_t  codecType;       //!< Video encoder type
    unsigned int       hResolution;     //!< Current encode horizontal resolution
    unsigned int       vResolution;     //!< Current encode vertical resolution
    unsigned int       averageFps;      //!< Moving average encode frames per second
    unsigned int       averageLatency;  //!< Moving average encode latency in microseconds
}lwmlEncoderSessionInfo_t;

/** @} */

/***************************************************************************************************/
/** @defgroup lwmlFBCStructs Frame Buffer Capture Structures
*  @{
*/
/***************************************************************************************************/

/**
 * Represents frame buffer capture session type
 */
typedef enum lwmlFBCSessionType_enum
{
    LWML_FBC_SESSION_TYPE_UNKNOWN = 0,     //!< Unknwon
    LWML_FBC_SESSION_TYPE_TOSYS,           //!< ToSys
    LWML_FBC_SESSION_TYPE_LWDA,            //!< Lwca
    LWML_FBC_SESSION_TYPE_VID,             //!< Vid
    LWML_FBC_SESSION_TYPE_HWENC,           //!< HEnc
} lwmlFBCSessionType_t;

/**
 * Structure to hold frame buffer capture sessions stats
 */
typedef struct lwmlFBCStats_st
{
    unsigned int      sessionsCount;    //!< Total no of sessions
    unsigned int      averageFPS;       //!< Moving average new frames captured per second
    unsigned int      averageLatency;   //!< Moving average new frame capture latency in microseconds
} lwmlFBCStats_t;

#define LWML_LWFBC_SESSION_FLAG_DIFFMAP_ENABLED                0x00000001    //!< Bit specifying differential map state.
#define LWML_LWFBC_SESSION_FLAG_CLASSIFICATIONMAP_ENABLED      0x00000002    //!< Bit specifying classification map state.
#define LWML_LWFBC_SESSION_FLAG_CAPTURE_WITH_WAIT_NO_WAIT      0x00000004    //!< Bit specifying if capture was requested as non-blocking call.
#define LWML_LWFBC_SESSION_FLAG_CAPTURE_WITH_WAIT_INFINITE     0x00000008    //!< Bit specifying if capture was requested as blocking call.
#define LWML_LWFBC_SESSION_FLAG_CAPTURE_WITH_WAIT_TIMEOUT      0x00000010    //!< Bit specifying if capture was requested as blocking call with timeout period.

/**
 * Structure to hold FBC session data
 */
typedef struct lwmlFBCSessionInfo_st
{
    unsigned int          sessionId;                           //!< Unique session ID
    unsigned int          pid;                                 //!< Owning process ID
    lwmlVgpuInstance_t    vgpuInstance;                        //!< Owning vGPU instance ID (only valid on vGPU hosts, otherwise zero)
    unsigned int          displayOrdinal;                      //!< Display identifier
    lwmlFBCSessionType_t  sessionType;                         //!< Type of frame buffer capture session
    unsigned int          sessionFlags;                        //!< Session flags (one or more of LWML_LWFBC_SESSION_FLAG_XXX).
    unsigned int          hMaxResolution;                      //!< Max horizontal resolution supported by the capture session
    unsigned int          vMaxResolution;                      //!< Max vertical resolution supported by the capture session
    unsigned int          hResolution;                         //!< Horizontal resolution requested by caller in capture call
    unsigned int          vResolution;                         //!< Vertical resolution requested by caller in capture call
    unsigned int          averageFPS;                          //!< Moving average new frames captured per second
    unsigned int          averageLatency;                      //!< Moving average new frame capture latency in microseconds
} lwmlFBCSessionInfo_t;

/** @} */

/***************************************************************************************************/
/** @defgroup lwmlDrainDefs definitions related to the drain state
 *  @{
 */
/***************************************************************************************************/

/**
 *  Is the GPU device to be removed from the kernel by lwmlDeviceRemoveGpu()
 */
typedef enum lwmlDetachGpuState_enum
{
    LWML_DETACH_GPU_KEEP         = 0,
    LWML_DETACH_GPU_REMOVE,
} lwmlDetachGpuState_t;

/**
 *  Parent bridge PCIe link state requested by lwmlDeviceRemoveGpu()
 */
typedef enum lwmlPcieLinkState_enum
{
    LWML_PCIE_LINK_KEEP         = 0,
    LWML_PCIE_LINK_SHUT_DOWN,
} lwmlPcieLinkState_t;

/** @} */

/***************************************************************************************************/
/** @defgroup lwmlInitializationAndCleanup Initialization and Cleanup
 * This chapter describes the methods that handle LWML initialization and cleanup.
 * It is the user's responsibility to call \ref lwmlInit_v2() before calling any other methods, and 
 * lwmlShutdown() once LWML is no longer being used.
 *  @{
 */
/***************************************************************************************************/

#define LWML_INIT_FLAG_NO_GPUS      1   //!< Don't fail lwmlInit() when no GPUs are found
#define LWML_INIT_FLAG_NO_ATTACH    2   //!< Don't attach GPUs

/**
 * Initialize LWML, but don't initialize any GPUs yet.
 *
 * \note lwmlInit_v3 introduces a "flags" argument, that allows passing boolean values
 *       modifying the behaviour of lwmlInit().
 * \note In LWML 5.319 new lwmlInit_v2 has replaced lwmlInit"_v1" (default in LWML 4.304 and older) that
 *       did initialize all GPU devices in the system.
 *       
 * This allows LWML to communicate with a GPU
 * when other GPUs in the system are unstable or in a bad state.  When using this API, GPUs are
 * discovered and initialized in lwmlDeviceGetHandleBy* functions instead.
 * 
 * \note To contrast lwmlInit_v2 with lwmlInit"_v1", LWML 4.304 lwmlInit"_v1" will fail when any detected GPU is in
 *       a bad or unstable state.
 * 
 * For all products.
 *
 * This method, should be called once before ilwoking any other methods in the library.
 * A reference count of the number of initializations is maintained.  Shutdown only oclwrs
 * when the reference count reaches zero.
 * 
 * @return 
 *         - \ref LWML_SUCCESS                   if LWML has been properly initialized
 *         - \ref LWML_ERROR_DRIVER_NOT_LOADED   if LWPU driver is not running
 *         - \ref LWML_ERROR_NO_PERMISSION       if LWML does not have permission to talk to the driver
 *         - \ref LWML_ERROR_UNKNOWN             on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlInit_v2(void);

/**
 * lwmlInitWithFlags is a variant of lwmlInit(), that allows passing a set of boolean values
 *       modifying the behaviour of lwmlInit().
 *       Other than the "flags" parameter it is completely similar to \ref lwmlInit_v2.
 *       
 * For all products.
 *
 * @param flags                                 behaviour modifier flags
 *
 * @return 
 *         - \ref LWML_SUCCESS                   if LWML has been properly initialized
 *         - \ref LWML_ERROR_DRIVER_NOT_LOADED   if LWPU driver is not running
 *         - \ref LWML_ERROR_NO_PERMISSION       if LWML does not have permission to talk to the driver
 *         - \ref LWML_ERROR_UNKNOWN             on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlInitWithFlags(unsigned int flags);

/**
 * Shut down LWML by releasing all GPU resources previously allocated with \ref lwmlInit_v2().
 * 
 * For all products.
 *
 * This method should be called after LWML work is done, once for each call to \ref lwmlInit_v2()
 * A reference count of the number of initializations is maintained.  Shutdown only oclwrs
 * when the reference count reaches zero.  For backwards compatibility, no error is reported if
 * lwmlShutdown() is called more times than lwmlInit().
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if LWML has been properly shut down
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlShutdown(void);

/** @} */

/***************************************************************************************************/
/** @defgroup lwmlErrorReporting Error reporting
 * This chapter describes helper functions for error reporting routines.
 *  @{
 */
/***************************************************************************************************/

/**
 * Helper method for colwerting LWML error codes into readable strings.
 *
 * For all products.
 *
 * @param result                               LWML error code to colwert
 *
 * @return String representation of the error.
 *
 */
const DECLDIR char* lwmlErrorString(lwmlReturn_t result);
/** @} */


/***************************************************************************************************/
/** @defgroup lwmlConstants Constants
 *  @{
 */
/***************************************************************************************************/

/**
 * Buffer size guaranteed to be large enough for \ref lwmlDeviceGetInforomVersion and \ref lwmlDeviceGetInforomImageVersion
 */
#define LWML_DEVICE_INFOROM_VERSION_BUFFER_SIZE       16

/**
 * Buffer size guaranteed to be large enough for storing GPU identifiers.
 */
#define LWML_DEVICE_UUID_BUFFER_SIZE                  80

/**
 * Buffer size guaranteed to be large enough for \ref lwmlDeviceGetUUID
 */
#define LWML_DEVICE_UUID_V2_BUFFER_SIZE               96

/**
 * Buffer size guaranteed to be large enough for \ref lwmlDeviceGetBoardPartNumber
 */
#define LWML_DEVICE_PART_NUMBER_BUFFER_SIZE           80

/**
 * Buffer size guaranteed to be large enough for \ref lwmlSystemGetDriverVersion
 */
#define LWML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE        80

/**
 * Buffer size guaranteed to be large enough for \ref lwmlSystemGetLWMLVersion
 */
#define LWML_SYSTEM_LWML_VERSION_BUFFER_SIZE          80

/**
 * Buffer size guaranteed to be large enough for storing GPU device names.
 */
#define LWML_DEVICE_NAME_BUFFER_SIZE                  64

/**
 * Buffer size guaranteed to be large enough for \ref lwmlDeviceGetName
 */
#define LWML_DEVICE_NAME_V2_BUFFER_SIZE               96

/**
 * Buffer size guaranteed to be large enough for \ref lwmlDeviceGetSerial
 */
#define LWML_DEVICE_SERIAL_BUFFER_SIZE                30

/**
 * Buffer size guaranteed to be large enough for \ref lwmlDeviceGetVbiosVersion
 */
#define LWML_DEVICE_VBIOS_VERSION_BUFFER_SIZE         32

/** @} */

/***************************************************************************************************/
/** @defgroup lwmlSystemQueries System Queries
 * This chapter describes the queries that LWML can perform against the local system. These queries
 * are not device-specific.
 *  @{
 */
/***************************************************************************************************/

/**
 * Retrieves the version of the system's graphics driver.
 * 
 * For all products.
 *
 * The version identifier is an alphanumeric string.  It will not exceed 80 characters in length
 * (including the NULL terminator).  See \ref lwmlConstants::LWML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE.
 *
 * @param version                              Reference in which to return the version identifier
 * @param length                               The maximum allowed length of the string returned in \a version
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a version has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a version is NULL
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a length is too small 
 */
lwmlReturn_t DECLDIR lwmlSystemGetDriverVersion(char *version, unsigned int length);

/**
 * Retrieves the version of the LWML library.
 * 
 * For all products.
 *
 * The version identifier is an alphanumeric string.  It will not exceed 80 characters in length
 * (including the NULL terminator).  See \ref lwmlConstants::LWML_SYSTEM_LWML_VERSION_BUFFER_SIZE.
 *
 * @param version                              Reference in which to return the version identifier
 * @param length                               The maximum allowed length of the string returned in \a version
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a version has been set
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a version is NULL
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a length is too small 
 */
lwmlReturn_t DECLDIR lwmlSystemGetLWMLVersion(char *version, unsigned int length);

/**
 * Retrieves the version of the LWCA driver.
 *
 * For all products.
 *
 * The LWCA driver version returned will be retreived from the lwrrently installed version of LWCA.
 * If the lwca library is not found, this function will return a known supported version number.
 *
 * @param lwdaDriverVersion                    Reference in which to return the version identifier
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a lwdaDriverVersion has been set
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a lwdaDriverVersion is NULL
 */
lwmlReturn_t DECLDIR lwmlSystemGetLwdaDriverVersion(int *lwdaDriverVersion);

/**
 * Retrieves the version of the LWCA driver from the shared library.
 *
 * For all products.
 *
 * The returned LWCA driver version by calling lwDriverGetVersion()
 *
 * @param lwdaDriverVersion                    Reference in which to return the version identifier
 *
 * @return
 *         - \ref LWML_SUCCESS                  if \a lwdaDriverVersion has been set
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a lwdaDriverVersion is NULL
 *         - \ref LWML_ERROR_LIBRARY_NOT_FOUND  if \a liblwda.so.1 or liblwda.dll is not found
 *         - \ref LWML_ERROR_FUNCTION_NOT_FOUND if \a lwDriverGetVersion() is not found in the shared library
 */
lwmlReturn_t DECLDIR lwmlSystemGetLwdaDriverVersion_v2(int *lwdaDriverVersion);

/**
 * Macros for colwerting the LWCA driver version number to Major and Minor version numbers.
 */
#define LWML_LWDA_DRIVER_VERSION_MAJOR(v) ((v)/1000)
#define LWML_LWDA_DRIVER_VERSION_MINOR(v) (((v)%1000)/10)

/**
 * Gets name of the process with provided process id
 *
 * For all products.
 *
 * Returned process name is cropped to provided length.
 * name string is encoded in ANSI.
 *
 * @param pid                                  The identifier of the process
 * @param name                                 Reference in which to return the process name
 * @param length                               The maximum allowed length of the string returned in \a name
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a name has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a name is NULL or \a length is 0.
 *         - \ref LWML_ERROR_NOT_FOUND         if process doesn't exists
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlSystemGetProcessName(unsigned int pid, char *name, unsigned int length);

/** @} */

/***************************************************************************************************/
/** @defgroup lwmlUnitQueries Unit Queries
 * This chapter describes that queries that LWML can perform against each unit. For S-class systems only.
 * In each case the device is identified with an lwmlUnit_t handle. This handle is obtained by 
 * calling \ref lwmlUnitGetHandleByIndex().
 *  @{
 */
/***************************************************************************************************/

 /**
 * Retrieves the number of units in the system.
 *
 * For S-class products.
 *
 * @param unitCount                            Reference in which to return the number of units
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a unitCount has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a unitCount is NULL
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlUnitGetCount(unsigned int *unitCount);

/**
 * Acquire the handle for a particular unit, based on its index.
 *
 * For S-class products.
 *
 * Valid indices are derived from the \a unitCount returned by \ref lwmlUnitGetCount(). 
 *   For example, if \a unitCount is 2 the valid indices are 0 and 1, corresponding to UNIT 0 and UNIT 1.
 *
 * The order in which LWML enumerates units has no guarantees of consistency between reboots.
 *
 * @param index                                The index of the target unit, >= 0 and < \a unitCount
 * @param unit                                 Reference in which to return the unit handle
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a unit has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a index is invalid or \a unit is NULL
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlUnitGetHandleByIndex(unsigned int index, lwmlUnit_t *unit);

/**
 * Retrieves the static information associated with a unit.
 *
 * For S-class products.
 *
 * See \ref lwmlUnitInfo_t for details on available unit info.
 *
 * @param unit                                 The identifier of the target unit
 * @param info                                 Reference in which to return the unit information
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a info has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a unit is invalid or \a info is NULL
 */
lwmlReturn_t DECLDIR lwmlUnitGetUnitInfo(lwmlUnit_t unit, lwmlUnitInfo_t *info);

/**
 * Retrieves the LED state associated with this unit.
 *
 * For S-class products.
 *
 * See \ref lwmlLedState_t for details on allowed states.
 *
 * @param unit                                 The identifier of the target unit
 * @param state                                Reference in which to return the current LED state
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a state has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a unit is invalid or \a state is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this is not an S-class product
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 * 
 * @see lwmlUnitSetLedState()
 */
lwmlReturn_t DECLDIR lwmlUnitGetLedState(lwmlUnit_t unit, lwmlLedState_t *state);

/**
 * Retrieves the PSU stats for the unit.
 *
 * For S-class products.
 *
 * See \ref lwmlPSUInfo_t for details on available PSU info.
 *
 * @param unit                                 The identifier of the target unit
 * @param psu                                  Reference in which to return the PSU information
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a psu has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a unit is invalid or \a psu is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this is not an S-class product
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlUnitGetPsuInfo(lwmlUnit_t unit, lwmlPSUInfo_t *psu);

/**
 * Retrieves the temperature readings for the unit, in degrees C.
 *
 * For S-class products.
 *
 * Depending on the product, readings may be available for intake (type=0), 
 * exhaust (type=1) and board (type=2).
 *
 * @param unit                                 The identifier of the target unit
 * @param type                                 The type of reading to take
 * @param temp                                 Reference in which to return the intake temperature
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a temp has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a unit or \a type is invalid or \a temp is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this is not an S-class product
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlUnitGetTemperature(lwmlUnit_t unit, unsigned int type, unsigned int *temp);

/**
 * Retrieves the fan speed readings for the unit.
 *
 * For S-class products.
 *
 * See \ref lwmlUnitFanSpeeds_t for details on available fan speed info.
 *
 * @param unit                                 The identifier of the target unit
 * @param fanSpeeds                            Reference in which to return the fan speed information
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a fanSpeeds has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a unit is invalid or \a fanSpeeds is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this is not an S-class product
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlUnitGetFanSpeedInfo(lwmlUnit_t unit, lwmlUnitFanSpeeds_t *fanSpeeds);

/**
 * Retrieves the set of GPU devices that are attached to the specified unit.
 *
 * For S-class products.
 *
 * The \a deviceCount argument is expected to be set to the size of the input \a devices array.
 *
 * @param unit                                 The identifier of the target unit
 * @param deviceCount                          Reference in which to provide the \a devices array size, and
 *                                             to return the number of attached GPU devices
 * @param devices                              Reference in which to return the references to the attached GPU devices
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a deviceCount and \a devices have been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a deviceCount indicates that the \a devices array is too small
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a unit is invalid, either of \a deviceCount or \a devices is NULL
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlUnitGetDevices(lwmlUnit_t unit, unsigned int *deviceCount, lwmlDevice_t *devices);

/**
 * Retrieves the IDs and firmware versions for any Host Interface Cards (HICs) in the system.
 * 
 * For S-class products.
 *
 * The \a hwbcCount argument is expected to be set to the size of the input \a hwbcEntries array.
 * The HIC must be connected to an S-class system for it to be reported by this function.
 *
 * @param hwbcCount                            Size of hwbcEntries array
 * @param hwbcEntries                          Array holding information about hwbc
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a hwbcCount and \a hwbcEntries have been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if either \a hwbcCount or \a hwbcEntries is NULL
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a hwbcCount indicates that the \a hwbcEntries array is too small
 */
lwmlReturn_t DECLDIR lwmlSystemGetHicVersion(unsigned int *hwbcCount, lwmlHwbcEntry_t *hwbcEntries);
/** @} */

/***************************************************************************************************/
/** @defgroup lwmlDeviceQueries Device Queries
 * This chapter describes that queries that LWML can perform against each device.
 * In each case the device is identified with an lwmlDevice_t handle. This handle is obtained by  
 * calling one of \ref lwmlDeviceGetHandleByIndex_v2(), \ref lwmlDeviceGetHandleBySerial(),
 * \ref lwmlDeviceGetHandleByPciBusId_v2(). or \ref lwmlDeviceGetHandleByUUID(). 
 *  @{
 */
/***************************************************************************************************/

 /**
 * Retrieves the number of compute devices in the system. A compute device is a single GPU.
 * 
 * For all products.
 *
 * Note: New lwmlDeviceGetCount_v2 (default in LWML 5.319) returns count of all devices in the system
 *       even if lwmlDeviceGetHandleByIndex_v2 returns LWML_ERROR_NO_PERMISSION for such device.
 *       Update your code to handle this error, or use LWML 4.304 or older lwml header file.
 *       For backward binary compatibility reasons _v1 version of the API is still present in the shared
 *       library.
 *       Old _v1 version of lwmlDeviceGetCount doesn't count devices that LWML has no permission to talk to.
 *
 * @param deviceCount                          Reference in which to return the number of accessible devices
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a deviceCount has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a deviceCount is NULL
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetCount_v2(unsigned int *deviceCount);

/**
 * Get attributes (engine counts etc.) for the given LWML device handle.
 *
 * @note This API lwrrently only supports MIG device handles.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               LWML device handle
 * @param attributes                           Device attributes
 *
 * @return
 *        - \ref LWML_SUCCESS                  if \a device attributes were successfully retrieved
 *        - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a device handle is invalid
 *        - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *        - \ref LWML_ERROR_NOT_SUPPORTED      if this query is not supported by the device
 *        - \ref LWML_ERROR_UNKNOWN            on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetAttributes_v2(lwmlDevice_t device, lwmlDeviceAttributes_t *attributes);

/**
 * Acquire the handle for a particular device, based on its index.
 * 
 * For all products.
 *
 * Valid indices are derived from the \a accessibleDevices count returned by 
 *   \ref lwmlDeviceGetCount_v2(). For example, if \a accessibleDevices is 2 the valid indices  
 *   are 0 and 1, corresponding to GPU 0 and GPU 1.
 *
 * The order in which LWML enumerates devices has no guarantees of consistency between reboots. For that reason it
 *   is recommended that devices be looked up by their PCI ids or UUID. See 
 *   \ref lwmlDeviceGetHandleByUUID() and \ref lwmlDeviceGetHandleByPciBusId_v2().
 *
 * Note: The LWML index may not correlate with other APIs, such as the LWCA device index.
 *
 * Starting from LWML 5, this API causes LWML to initialize the target GPU
 * LWML may initialize additional GPUs if:
 *  - The target GPU is an SLI slave
 * 
 * Note: New lwmlDeviceGetCount_v2 (default in LWML 5.319) returns count of all devices in the system
 *       even if lwmlDeviceGetHandleByIndex_v2 returns LWML_ERROR_NO_PERMISSION for such device.
 *       Update your code to handle this error, or use LWML 4.304 or older lwml header file.
 *       For backward binary compatibility reasons _v1 version of the API is still present in the shared
 *       library.
 *       Old _v1 version of lwmlDeviceGetCount doesn't count devices that LWML has no permission to talk to.
 *
 *       This means that lwmlDeviceGetHandleByIndex_v2 and _v1 can return different devices for the same index.
 *       If you don't touch macros that map old (_v1) versions to _v2 versions at the top of the file you don't
 *       need to worry about that.
 *
 * @param index                                The index of the target GPU, >= 0 and < \a accessibleDevices
 * @param device                               Reference in which to return the device handle
 * 
 * @return 
 *         - \ref LWML_SUCCESS                  if \a device has been set
 *         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a index is invalid or \a device is NULL
 *         - \ref LWML_ERROR_INSUFFICIENT_POWER if any attached devices have improperly attached external power cables
 *         - \ref LWML_ERROR_NO_PERMISSION      if the user doesn't have permission to talk to this device
 *         - \ref LWML_ERROR_IRQ_ISSUE          if LWPU kernel detected an interrupt issue with the attached GPUs
 *         - \ref LWML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
 *
 * @see lwmlDeviceGetIndex
 * @see lwmlDeviceGetCount
 */
lwmlReturn_t DECLDIR lwmlDeviceGetHandleByIndex_v2(unsigned int index, lwmlDevice_t *device);

/**
 * Acquire the handle for a particular device, based on its board serial number.
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * This number corresponds to the value printed directly on the board, and to the value returned by
 *   \ref lwmlDeviceGetSerial().
 *
 * @deprecated Since more than one GPU can exist on a single board this function is deprecated in favor 
 *             of \ref lwmlDeviceGetHandleByUUID.
 *             For dual GPU boards this function will return LWML_ERROR_ILWALID_ARGUMENT.
 *
 * Starting from LWML 5, this API causes LWML to initialize the target GPU
 * LWML may initialize additional GPUs as it searches for the target GPU
 *
 * @param serial                               The board serial number of the target GPU
 * @param device                               Reference in which to return the device handle
 * 
 * @return 
 *         - \ref LWML_SUCCESS                  if \a device has been set
 *         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a serial is invalid, \a device is NULL or more than one
 *                                              device has the same serial (dual GPU boards)
 *         - \ref LWML_ERROR_NOT_FOUND          if \a serial does not match a valid device on the system
 *         - \ref LWML_ERROR_INSUFFICIENT_POWER if any attached devices have improperly attached external power cables
 *         - \ref LWML_ERROR_IRQ_ISSUE          if LWPU kernel detected an interrupt issue with the attached GPUs
 *         - \ref LWML_ERROR_GPU_IS_LOST        if any GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
 *
 * @see lwmlDeviceGetSerial
 * @see lwmlDeviceGetHandleByUUID
 */
lwmlReturn_t DECLDIR lwmlDeviceGetHandleBySerial(const char *serial, lwmlDevice_t *device);

/**
 * Acquire the handle for a particular device, based on its globally unique immutable UUID associated with each device.
 *
 * For all products.
 *
 * @param uuid                                 The UUID of the target GPU or MIG instance
 * @param device                               Reference in which to return the device handle or MIG device handle
 * 
 * Starting from LWML 5, this API causes LWML to initialize the target GPU
 * LWML may initialize additional GPUs as it searches for the target GPU
 *
 * @return 
 *         - \ref LWML_SUCCESS                  if \a device has been set
 *         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a uuid is invalid or \a device is null
 *         - \ref LWML_ERROR_NOT_FOUND          if \a uuid does not match a valid device on the system
 *         - \ref LWML_ERROR_INSUFFICIENT_POWER if any attached devices have improperly attached external power cables
 *         - \ref LWML_ERROR_IRQ_ISSUE          if LWPU kernel detected an interrupt issue with the attached GPUs
 *         - \ref LWML_ERROR_GPU_IS_LOST        if any GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
 *
 * @see lwmlDeviceGetUUID
 */
lwmlReturn_t DECLDIR lwmlDeviceGetHandleByUUID(const char *uuid, lwmlDevice_t *device);

/**
 * Acquire the handle for a particular device, based on its PCI bus id.
 * 
 * For all products.
 *
 * This value corresponds to the lwmlPciInfo_t::busId returned by \ref lwmlDeviceGetPciInfo_v3().
 *
 * Starting from LWML 5, this API causes LWML to initialize the target GPU
 * LWML may initialize additional GPUs if:
 *  - The target GPU is an SLI slave
 *
 * \note LWML 4.304 and older version of lwmlDeviceGetHandleByPciBusId"_v1" returns LWML_ERROR_NOT_FOUND 
 *       instead of LWML_ERROR_NO_PERMISSION.
 *
 * @param pciBusId                             The PCI bus id of the target GPU
 * @param device                               Reference in which to return the device handle
 * 
 * @return 
 *         - \ref LWML_SUCCESS                  if \a device has been set
 *         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a pciBusId is invalid or \a device is NULL
 *         - \ref LWML_ERROR_NOT_FOUND          if \a pciBusId does not match a valid device on the system
 *         - \ref LWML_ERROR_INSUFFICIENT_POWER if the attached device has improperly attached external power cables
 *         - \ref LWML_ERROR_NO_PERMISSION      if the user doesn't have permission to talk to this device
 *         - \ref LWML_ERROR_IRQ_ISSUE          if LWPU kernel detected an interrupt issue with the attached GPUs
 *         - \ref LWML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, lwmlDevice_t *device);

/**
 * Retrieves the name of this device. 
 * 
 * For all products.
 *
 * The name is an alphanumeric string that denotes a particular product, e.g. Tesla &tm; C2070. It will not
 * exceed 96 characters in length (including the NULL terminator).  See \ref
 * lwmlConstants::LWML_DEVICE_NAME_V2_BUFFER_SIZE.
 *
 * When used with MIG device handles the API returns MIG device names which can be used to identify devices
 * based on their attributes.
 *
 * @param device                               The identifier of the target device
 * @param name                                 Reference in which to return the product name
 * @param length                               The maximum allowed length of the string returned in \a name
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a name has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, or \a name is NULL
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a length is too small
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetName(lwmlDevice_t device, char *name, unsigned int length);

/**
 * Retrieves the brand of this device.
 *
 * For all products.
 *
 * The type is a member of \ref lwmlBrandType_t defined above.
 *
 * @param device                               The identifier of the target device
 * @param type                                 Reference in which to return the product brand type
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a name has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, or \a type is NULL
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetBrand(lwmlDevice_t device, lwmlBrandType_t *type);

/**
 * Retrieves the LWML index of this device.
 *
 * For all products.
 * 
 * Valid indices are derived from the \a accessibleDevices count returned by 
 *   \ref lwmlDeviceGetCount_v2(). For example, if \a accessibleDevices is 2 the valid indices  
 *   are 0 and 1, corresponding to GPU 0 and GPU 1.
 *
 * The order in which LWML enumerates devices has no guarantees of consistency between reboots. For that reason it
 *   is recommended that devices be looked up by their PCI ids or GPU UUID. See 
 *   \ref lwmlDeviceGetHandleByPciBusId_v2() and \ref lwmlDeviceGetHandleByUUID().
 *
 * When used with MIG device handles this API returns indices that can be
 * passed to \ref lwmlDeviceGetMigDeviceHandleByIndex to retrieve an identical handle.
 * MIG device indices are unique within a device.
 *
 * Note: The LWML index may not correlate with other APIs, such as the LWCA device index.
 *
 * @param device                               The identifier of the target device
 * @param index                                Reference in which to return the LWML index of the device
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a index has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, or \a index is NULL
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlDeviceGetHandleByIndex()
 * @see lwmlDeviceGetCount()
 */
lwmlReturn_t DECLDIR lwmlDeviceGetIndex(lwmlDevice_t device, unsigned int *index);

/**
 * Retrieves the globally unique board serial number associated with this device's board.
 *
 * For all products with an inforom.
 *
 * The serial number is an alphanumeric string that will not exceed 30 characters (including the NULL terminator).
 * This number matches the serial number tag that is physically attached to the board.  See \ref
 * lwmlConstants::LWML_DEVICE_SERIAL_BUFFER_SIZE.
 *
 * @param device                               The identifier of the target device
 * @param serial                               Reference in which to return the board/module serial number
 * @param length                               The maximum allowed length of the string returned in \a serial
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a serial has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, or \a serial is NULL
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a length is too small
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetSerial(lwmlDevice_t device, char *serial, unsigned int length);


/***************************************************************************************************/

/** @defgroup lwmlAffinity CPU and Memory Affinity
 *  This chapter describes LWML operations that are associated with CPU and memory
 *  affinity.
 *  @{
 */
/***************************************************************************************************/

//! Scope of NUMA node for affinity queries
#define LWML_AFFINITY_SCOPE_NODE     0
//! Scope of processor socket for affinity queries
#define LWML_AFFINITY_SCOPE_SOCKET   1

typedef unsigned int lwmlAffinityScope_t;

/**
 * Retrieves an array of unsigned ints (sized to nodeSetSize) of bitmasks with
 * the ideal memory affinity within node or socket for the device.
 * For example, if NUMA node 0, 1 are ideal within the socket for the device and nodeSetSize ==  1,
 *     result[0] = 0x3
 *
 * \note If requested scope is not applicable to the target topology, the API
 *       will fall back to reporting the memory affinity for the immediate non-I/O
 *       ancestor of the device.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               The identifier of the target device
 * @param nodeSetSize                          The size of the nodeSet array that is safe to access
 * @param nodeSet                              Array reference in which to return a bitmask of NODEs, 64 NODEs per
 *                                             unsigned long on 64-bit machines, 32 on 32-bit machines
 * @param scope                                Scope that change the default behavior
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a NUMA node Affinity has been filled
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, nodeSetSize == 0, nodeSet is NULL or scope is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */

lwmlReturn_t DECLDIR lwmlDeviceGetMemoryAffinity(lwmlDevice_t device, unsigned int nodeSetSize, unsigned long *nodeSet, lwmlAffinityScope_t scope);

/**
 * Retrieves an array of unsigned ints (sized to cpuSetSize) of bitmasks with the
 * ideal CPU affinity within node or socket for the device.
 * For example, if processors 0, 1, 32, and 33 are ideal for the device and cpuSetSize == 2,
 *     result[0] = 0x3, result[1] = 0x3
 *
 * \note If requested scope is not applicable to the target topology, the API
 *       will fall back to reporting the CPU affinity for the immediate non-I/O
 *       ancestor of the device.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               The identifier of the target device
 * @param cpuSetSize                           The size of the cpuSet array that is safe to access
 * @param cpuSet                               Array reference in which to return a bitmask of CPUs, 64 CPUs per 
 *                                                 unsigned long on 64-bit machines, 32 on 32-bit machines
 * @param scope                                Scope that change the default behavior
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a cpuAffinity has been filled
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, cpuSetSize == 0, cpuSet is NULL or sope is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */

lwmlReturn_t DECLDIR lwmlDeviceGetCpuAffinityWithinScope(lwmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet, lwmlAffinityScope_t scope);

/**
 * Retrieves an array of unsigned ints (sized to cpuSetSize) of bitmasks with the ideal CPU affinity for the device
 * For example, if processors 0, 1, 32, and 33 are ideal for the device and cpuSetSize == 2,
 *     result[0] = 0x3, result[1] = 0x3
 * This is equivalent to calling \ref lwmlDeviceGetCpuAffinityWithinScope with \ref LWML_AFFINITY_SCOPE_NODE.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               The identifier of the target device
 * @param cpuSetSize                           The size of the cpuSet array that is safe to access
 * @param cpuSet                               Array reference in which to return a bitmask of CPUs, 64 CPUs per 
 *                                                 unsigned long on 64-bit machines, 32 on 32-bit machines
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a cpuAffinity has been filled
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, cpuSetSize == 0, or cpuSet is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetCpuAffinity(lwmlDevice_t device, unsigned int cpuSetSize, unsigned long *cpuSet);

/**
 * Sets the ideal affinity for the calling thread and device using the guidelines 
 * given in lwmlDeviceGetCpuAffinity().  Note, this is a change as of version 8.0.  
 * Older versions set the affinity for a calling process and all children.
 * Lwrrently supports up to 1024 processors.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               The identifier of the target device
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if the calling process has been successfully bound
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceSetCpuAffinity(lwmlDevice_t device);

/**
 * Clear all affinity bindings for the calling thread.  Note, this is a change as of version
 * 8.0 as older versions cleared the affinity for a calling process and all children.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               The identifier of the target device
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if the calling process has been successfully unbound
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceClearCpuAffinity(lwmlDevice_t device);

/**
 * Retrieve the common ancestor for two devices
 * For all products.
 * Supported on Linux only.
 *
 * @param device1                              The identifier of the first device
 * @param device2                              The identifier of the second device
 * @param pathInfo                             A \ref lwmlGpuTopologyLevel_t that gives the path type
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a pathInfo has been set
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device1, or \a device2 is invalid, or \a pathInfo is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device or OS does not support this feature
 *         - \ref LWML_ERROR_UNKNOWN           an error has oclwrred in underlying topology discovery
 */

/** @} */
lwmlReturn_t DECLDIR lwmlDeviceGetTopologyCommonAncestor(lwmlDevice_t device1, lwmlDevice_t device2, lwmlGpuTopologyLevel_t *pathInfo);

/**
 * Retrieve the set of GPUs that are nearest to a given device at a specific interconnectivity level
 * For all products.
 * Supported on Linux only.
 *
 * @param device                               The identifier of the first device
 * @param level                                The \ref lwmlGpuTopologyLevel_t level to search for other GPUs
 * @param count                                When zero, is set to the number of matching GPUs such that \a deviceArray 
 *                                             can be malloc'd.  When non-zero, \a deviceArray will be filled with \a count
 *                                             number of device handles.
 * @param deviceArray                          An array of device handles for GPUs found at \a level
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a deviceArray or \a count (if initially zero) has been set
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device, \a level, or \a count is invalid, or \a deviceArray is NULL with a non-zero \a count
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device or OS does not support this feature
 *         - \ref LWML_ERROR_UNKNOWN           an error has oclwrred in underlying topology discovery
 */
lwmlReturn_t DECLDIR lwmlDeviceGetTopologyNearestGpus(lwmlDevice_t device, lwmlGpuTopologyLevel_t level, unsigned int *count, lwmlDevice_t *deviceArray);

/**
 * Retrieve the set of GPUs that have a CPU affinity with the given CPU number
 * For all products.
 * Supported on Linux only.
 *
 * @param cpuNumber                            The CPU number
 * @param count                                When zero, is set to the number of matching GPUs such that \a deviceArray 
 *                                             can be malloc'd.  When non-zero, \a deviceArray will be filled with \a count
 *                                             number of device handles.
 * @param deviceArray                          An array of device handles for GPUs found with affinity to \a cpuNumber
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a deviceArray or \a count (if initially zero) has been set
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a cpuNumber, or \a count is invalid, or \a deviceArray is NULL with a non-zero \a count
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device or OS does not support this feature
 *         - \ref LWML_ERROR_UNKNOWN           an error has oclwrred in underlying topology discovery
 */
lwmlReturn_t DECLDIR lwmlSystemGetTopologyGpuSet(unsigned int cpuNumber, unsigned int *count, lwmlDevice_t *deviceArray);

/**
 * Retrieve the status for a given p2p capability index between a given pair of GPU 
 * 
 * @param device1                              The first device 
 * @param device2                              The second device
 * @param p2pIndex                             p2p Capability Index being looked for between \a device1 and \a device2
 * @param p2pStatus                            Reference in which to return the status of the \a p2pIndex
 *                                             between \a device1 and \a device2
 * @return 
 *         - \ref LWML_SUCCESS         if \a p2pStatus has been populated
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT     if \a device1 or \a device2 or \a p2pIndex is invalid or \a p2pStatus is NULL
 *         - \ref LWML_ERROR_UNKNOWN              on any unexpected error
 */ 
lwmlReturn_t DECLDIR lwmlDeviceGetP2PStatus(lwmlDevice_t device1, lwmlDevice_t device2, lwmlGpuP2PCapsIndex_t p2pIndex,lwmlGpuP2PStatus_t *p2pStatus);

/**
 * Retrieves the globally unique immutable UUID associated with this device, as a 5 part hexadecimal string,
 * that augments the immutable, board serial identifier.
 *
 * For all products.
 *
 * The UUID is a globally unique identifier. It is the only available identifier for pre-Fermi-architecture products.
 * It does NOT correspond to any identifier printed on the board.  It will not exceed 96 characters in length
 * (including the NULL terminator).  See \ref lwmlConstants::LWML_DEVICE_UUID_V2_BUFFER_SIZE.
 *
 * When used with MIG device handles the API returns globally unique UUIDs which can be used to identify MIG
 * devices across both GPU and MIG devices. UUIDs are immutable for the lifetime of a MIG device.
 *
 * @param device                               The identifier of the target device
 * @param uuid                                 Reference in which to return the GPU UUID
 * @param length                               The maximum allowed length of the string returned in \a uuid
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a uuid has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, or \a uuid is NULL
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a length is too small 
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetUUID(lwmlDevice_t device, char *uuid, unsigned int length);

/**
 * Retrieve the MDEV UUID of a vGPU instance.
 *
 * The MDEV UUID is a globally unique identifier of the mdev device assigned to the VM, and is returned as a 5-part hexadecimal string,
 * not exceeding 80 characters in length (including the NULL terminator).
 * MDEV UUID is displayed only on KVM platform.
 * See \ref lwmlConstants::LWML_DEVICE_UUID_BUFFER_SIZE.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param mdevUuid                 Pointer to caller-supplied buffer to hold MDEV UUID
 * @param size                     Size of buffer in bytes
 *
 * @return
 *         - \ref LWML_SUCCESS                 successful completion
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_NOT_SUPPORTED     on any hypervisor other than KVM
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuInstance is 0, or \a mdevUuid is NULL
 *         - \ref LWML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a size is too small
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetMdevUUID(lwmlVgpuInstance_t vgpuInstance, char *mdevUuid, unsigned int size);

/**
 * Retrieves minor number for the device. The minor number for the device is such that the Lwpu device node file for 
 * each GPU will have the form /dev/lwpu[minor number].
 *
 * For all products.
 * Supported only for Linux
 *
 * @param device                                The identifier of the target device
 * @param minorNumber                           Reference in which to return the minor number for the device
 * @return
 *         - \ref LWML_SUCCESS                 if the minor number is successfully retrieved
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a minorNumber is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetMinorNumber(lwmlDevice_t device, unsigned int *minorNumber);

/**
 * Retrieves the the device board part number which is programmed into the board's InfoROM
 *
 * For all products.
 *
 * @param device                                Identifier of the target device
 * @param partNumber                            Reference to the buffer to return
 * @param length                                Length of the buffer reference
 *
 * @return
 *         - \ref LWML_SUCCESS                  if \a partNumber has been set
 *         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref LWML_ERROR_NOT_SUPPORTED      if the needed VBIOS fields have not been filled
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a device is invalid or \a serial is NULL
 *         - \ref LWML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetBoardPartNumber(lwmlDevice_t device, char* partNumber, unsigned int length);

/**
 * Retrieves the version information for the device's infoROM object.
 *
 * For all products with an inforom.
 *
 * Fermi and higher parts have non-volatile on-board memory for persisting device info, such as aggregate 
 * ECC counts. The version of the data structures in this memory may change from time to time. It will not
 * exceed 16 characters in length (including the NULL terminator).
 * See \ref lwmlConstants::LWML_DEVICE_INFOROM_VERSION_BUFFER_SIZE.
 *
 * See \ref lwmlInforomObject_t for details on the available infoROM objects.
 *
 * @param device                               The identifier of the target device
 * @param object                               The target infoROM object
 * @param version                              Reference in which to return the infoROM version
 * @param length                               The maximum allowed length of the string returned in \a version
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a version has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a version is NULL
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a length is too small 
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not have an infoROM
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlDeviceGetInforomImageVersion
 */
lwmlReturn_t DECLDIR lwmlDeviceGetInforomVersion(lwmlDevice_t device, lwmlInforomObject_t object, char *version, unsigned int length);

/**
 * Retrieves the global infoROM image version
 *
 * For all products with an inforom.
 *
 * Image version just like VBIOS version uniquely describes the exact version of the infoROM flashed on the board 
 * in contrast to infoROM object version which is only an indicator of supported features.
 * Version string will not exceed 16 characters in length (including the NULL terminator).
 * See \ref lwmlConstants::LWML_DEVICE_INFOROM_VERSION_BUFFER_SIZE.
 *
 * @param device                               The identifier of the target device
 * @param version                              Reference in which to return the infoROM image version
 * @param length                               The maximum allowed length of the string returned in \a version
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a version has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a version is NULL
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a length is too small 
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not have an infoROM
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlDeviceGetInforomVersion
 */
lwmlReturn_t DECLDIR lwmlDeviceGetInforomImageVersion(lwmlDevice_t device, char *version, unsigned int length);

/**
 * Retrieves the checksum of the configuration stored in the device's infoROM.
 *
 * For all products with an inforom.
 *
 * Can be used to make sure that two GPUs have the exact same configuration.
 * Current checksum takes into account configuration stored in PWR and ECC infoROM objects.
 * Checksum can change between driver releases or when user changes configuration (e.g. disable/enable ECC)
 *
 * @param device                               The identifier of the target device
 * @param checksum                             Reference in which to return the infoROM configuration checksum
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a checksum has been set
 *         - \ref LWML_ERROR_CORRUPTED_INFOROM if the device's checksum couldn't be retrieved due to infoROM corruption
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a checksum is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error 
 */
lwmlReturn_t DECLDIR lwmlDeviceGetInforomConfigurationChecksum(lwmlDevice_t device, unsigned int *checksum);

/**
 * Reads the infoROM from the flash and verifies the checksums.
 *
 * For all products with an inforom.
 *
 * @param device                               The identifier of the target device
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if infoROM is not corrupted
 *         - \ref LWML_ERROR_CORRUPTED_INFOROM if the device's infoROM is corrupted
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error 
 */
lwmlReturn_t DECLDIR lwmlDeviceValidateInforom(lwmlDevice_t device);

/**
 * Retrieves the display mode for the device.
 *
 * For all products.
 *
 * This method indicates whether a physical display (e.g. monitor) is lwrrently connected to
 * any of the device's connectors.
 *
 * See \ref lwmlEnableState_t for details on allowed modes.
 *
 * @param device                               The identifier of the target device
 * @param display                              Reference in which to return the display mode
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a display has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a display is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetDisplayMode(lwmlDevice_t device, lwmlEnableState_t *display);

/**
 * Retrieves the display active state for the device.
 *
 * For all products.
 *
 * This method indicates whether a display is initialized on the device.
 * For example whether X Server is attached to this device and has allocated memory for the screen.
 *
 * Display can be active even when no monitor is physically attached.
 *
 * See \ref lwmlEnableState_t for details on allowed modes.
 *
 * @param device                               The identifier of the target device
 * @param isActive                             Reference in which to return the display active state
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a isActive has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a isActive is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetDisplayActive(lwmlDevice_t device, lwmlEnableState_t *isActive);

/**
 * Retrieves the persistence mode associated with this device.
 *
 * For all products.
 * For Linux only.
 *
 * When driver persistence mode is enabled the driver software state is not torn down when the last 
 * client disconnects. By default this feature is disabled. 
 *
 * See \ref lwmlEnableState_t for details on allowed modes.
 *
 * @param device                               The identifier of the target device
 * @param mode                                 Reference in which to return the current driver persistence mode
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a mode has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a mode is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlDeviceSetPersistenceMode()
 */
lwmlReturn_t DECLDIR lwmlDeviceGetPersistenceMode(lwmlDevice_t device, lwmlEnableState_t *mode);

/**
 * Retrieves the PCI attributes of this device.
 * 
 * For all products.
 *
 * See \ref lwmlPciInfo_t for details on the available PCI info.
 *
 * @param device                               The identifier of the target device
 * @param pci                                  Reference in which to return the PCI info
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a pci has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a pci is NULL
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetPciInfo_v3(lwmlDevice_t device, lwmlPciInfo_t *pci);

/**
 * Retrieves the maximum PCIe link generation possible with this device and system
 *
 * I.E. for a generation 2 PCIe device attached to a generation 1 PCIe bus the max link generation this function will
 * report is generation 1.
 * 
 * For Fermi &tm; or newer fully supported devices.
 * 
 * @param device                               The identifier of the target device
 * @param maxLinkGen                           Reference in which to return the max PCIe link generation
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a maxLinkGen has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a maxLinkGen is null
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if PCIe link information is not available
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetMaxPcieLinkGeneration(lwmlDevice_t device, unsigned int *maxLinkGen);

/**
 * Retrieves the maximum PCIe link width possible with this device and system
 *
 * I.E. for a device with a 16x PCIe bus width attached to a 8x PCIe system bus this function will report
 * a max link width of 8.
 * 
 * For Fermi &tm; or newer fully supported devices.
 * 
 * @param device                               The identifier of the target device
 * @param maxLinkWidth                         Reference in which to return the max PCIe link generation
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a maxLinkWidth has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a maxLinkWidth is null
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if PCIe link information is not available
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetMaxPcieLinkWidth(lwmlDevice_t device, unsigned int *maxLinkWidth);

/**
 * Retrieves the current PCIe link generation
 * 
 * For Fermi &tm; or newer fully supported devices.
 * 
 * @param device                               The identifier of the target device
 * @param lwrrLinkGen                          Reference in which to return the current PCIe link generation
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a lwrrLinkGen has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a lwrrLinkGen is null
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if PCIe link information is not available
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetLwrrPcieLinkGeneration(lwmlDevice_t device, unsigned int *lwrrLinkGen);

/**
 * Retrieves the current PCIe link width
 * 
 * For Fermi &tm; or newer fully supported devices.
 * 
 * @param device                               The identifier of the target device
 * @param lwrrLinkWidth                        Reference in which to return the current PCIe link generation
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a lwrrLinkWidth has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a lwrrLinkWidth is null
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if PCIe link information is not available
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetLwrrPcieLinkWidth(lwmlDevice_t device, unsigned int *lwrrLinkWidth);

/**
 * Retrieve PCIe utilization information.
 * This function is querying a byte counter over a 20ms interval and thus is the 
 *   PCIe throughput over that interval.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * This method is not supported in virtual machines running virtual GPU (vGPU).
 *
 * @param device                               The identifier of the target device
 * @param counter                              The specific counter that should be queried \ref lwmlPcieUtilCounter_t
 * @param value                                Reference in which to return throughput in KB/s
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a value has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device or \a counter is invalid, or \a value is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetPcieThroughput(lwmlDevice_t device, lwmlPcieUtilCounter_t counter, unsigned int *value);

/**  
 * Retrieve the PCIe replay counter.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param value                                Reference in which to return the counter's value
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a value has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, or \a value is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetPcieReplayCounter(lwmlDevice_t device, unsigned int *value);

/**
 * Retrieves the current clock speeds for the device.
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * See \ref lwmlClockType_t for details on available clock information.
 *
 * @param device                               The identifier of the target device
 * @param type                                 Identify which clock domain to query
 * @param clock                                Reference in which to return the clock speed in MHz
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a clock has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a clock is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device cannot report the specified clock
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetClockInfo(lwmlDevice_t device, lwmlClockType_t type, unsigned int *clock);

/**
 * Retrieves the maximum clock speeds for the device.
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * See \ref lwmlClockType_t for details on available clock information.
 *
 * \note On GPUs from Fermi family current P0 clocks (reported by \ref lwmlDeviceGetClockInfo) can differ from max clocks
 *       by few MHz.
 *
 * @param device                               The identifier of the target device
 * @param type                                 Identify which clock domain to query
 * @param clock                                Reference in which to return the clock speed in MHz
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a clock has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a clock is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device cannot report the specified clock
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetMaxClockInfo(lwmlDevice_t device, lwmlClockType_t type, unsigned int *clock);

/**
 * Retrieves the current setting of a clock that applications will use unless an overspec situation oclwrs.
 * Can be changed using \ref lwmlDeviceSetApplicationsClocks.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param clockType                            Identify which clock domain to query
 * @param clockMHz                             Reference in which to return the clock in MHz
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a clockMHz has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a clockMHz is NULL or \a clockType is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetApplicationsClock(lwmlDevice_t device, lwmlClockType_t clockType, unsigned int *clockMHz);

/**
 * Retrieves the default applications clock that GPU boots with or 
 * defaults to after \ref lwmlDeviceResetApplicationsClocks call.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param clockType                            Identify which clock domain to query
 * @param clockMHz                             Reference in which to return the default clock in MHz
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a clockMHz has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a clockMHz is NULL or \a clockType is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * \see lwmlDeviceGetApplicationsClock
 */
lwmlReturn_t DECLDIR lwmlDeviceGetDefaultApplicationsClock(lwmlDevice_t device, lwmlClockType_t clockType, unsigned int *clockMHz);

/**
 * Resets the application clock to the default value
 *
 * This is the applications clock that will be used after system reboot or driver reload.
 * Default value is constant, but the current value an be changed using \ref lwmlDeviceSetApplicationsClocks.
 *
 * On Pascal and newer hardware, if clocks were previously locked with \ref lwmlDeviceSetApplicationsClocks,
 * this call will unlock clocks. This returns clocks their default behavior ofautomatically boosting above
 * base clocks as thermal limits allow.
 *
 * @see lwmlDeviceGetApplicationsClock
 * @see lwmlDeviceSetApplicationsClocks
 *
 * For Fermi &tm; or newer non-VdChip fully supported devices and Maxwell or newer VdChip devices.
 *
 * @param device                               The identifier of the target device
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if new settings were successfully set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceResetApplicationsClocks(lwmlDevice_t device);

/**
 * Retrieves the clock speed for the clock specified by the clock type and clock ID.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param clockType                            Identify which clock domain to query
 * @param clockId                              Identify which clock in the domain to query
 * @param clockMHz                             Reference in which to return the clock in MHz
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a clockMHz has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a clockMHz is NULL or \a clockType is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetClock(lwmlDevice_t device, lwmlClockType_t clockType, lwmlClockId_t clockId, unsigned int *clockMHz);

/**
 * Retrieves the customer defined maximum boost clock speed specified by the given clock type.
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param clockType                            Identify which clock domain to query
 * @param clockMHz                             Reference in which to return the clock in MHz
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a clockMHz has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a clockMHz is NULL or \a clockType is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device or the \a clockType on this device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetMaxLwstomerBoostClock(lwmlDevice_t device, lwmlClockType_t clockType, unsigned int *clockMHz);

/**
 * Retrieves the list of possible memory clocks that can be used as an argument for \ref lwmlDeviceSetApplicationsClocks.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param count                                Reference in which to provide the \a clocksMHz array size, and
 *                                             to return the number of elements
 * @param clocksMHz                            Reference in which to return the clock in MHz
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a count and \a clocksMHz have been populated 
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a count is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a count is too small (\a count is set to the number of
 *                                                required elements)
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlDeviceSetApplicationsClocks
 * @see lwmlDeviceGetSupportedGraphicsClocks
 */
lwmlReturn_t DECLDIR lwmlDeviceGetSupportedMemoryClocks(lwmlDevice_t device, unsigned int *count, unsigned int *clocksMHz);

/**
 * Retrieves the list of possible graphics clocks that can be used as an argument for \ref lwmlDeviceSetApplicationsClocks.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param memoryClockMHz                       Memory clock for which to return possible graphics clocks
 * @param count                                Reference in which to provide the \a clocksMHz array size, and
 *                                             to return the number of elements
 * @param clocksMHz                            Reference in which to return the clocks in MHz
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a count and \a clocksMHz have been populated 
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_NOT_FOUND         if the specified \a memoryClockMHz is not a supported frequency
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a clock is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a count is too small 
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlDeviceSetApplicationsClocks
 * @see lwmlDeviceGetSupportedMemoryClocks
 */
lwmlReturn_t DECLDIR lwmlDeviceGetSupportedGraphicsClocks(lwmlDevice_t device, unsigned int memoryClockMHz, unsigned int *count, unsigned int *clocksMHz);

/**
 * Retrieve the current state of Auto Boosted clocks on a device and store it in \a isEnabled
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates
 * to maximize performance as thermal limits allow.
 *
 * On Pascal and newer hardware, Auto Aoosted clocks are controlled through application clocks.
 * Use \ref lwmlDeviceSetApplicationsClocks and \ref lwmlDeviceResetApplicationsClocks to control Auto Boost
 * behavior.
 *
 * @param device                               The identifier of the target device
 * @param isEnabled                            Where to store the current state of Auto Boosted clocks of the target device
 * @param defaultIsEnabled                     Where to store the default Auto Boosted clocks behavior of the target device that the device will
 *                                                 revert to when no applications are using the GPU
 *
 * @return
 *         - \ref LWML_SUCCESS                 If \a isEnabled has been been set with the Auto Boosted clocks state of \a device
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a isEnabled is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support Auto Boosted clocks
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 */
lwmlReturn_t DECLDIR lwmlDeviceGetAutoBoostedClocksEnabled(lwmlDevice_t device, lwmlEnableState_t *isEnabled, lwmlEnableState_t *defaultIsEnabled);

/**
 * Try to set the current state of Auto Boosted clocks on a device.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates
 * to maximize performance as thermal limits allow. Auto Boosted clocks should be disabled if fixed clock
 * rates are desired.
 *
 * Non-root users may use this API by default but can be restricted by root from using this API by calling
 * \ref lwmlDeviceSetAPIRestriction with apiType=LWML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS.
 * Note: Persistence Mode is required to modify current Auto Boost settings, therefore, it must be enabled.
 *
 * On Pascal and newer hardware, Auto Boosted clocks are controlled through application clocks.
 * Use \ref lwmlDeviceSetApplicationsClocks and \ref lwmlDeviceResetApplicationsClocks to control Auto Boost
 * behavior.
 *
 * @param device                               The identifier of the target device
 * @param enabled                              What state to try to set Auto Boosted clocks of the target device to
 *
 * @return
 *         - \ref LWML_SUCCESS                 If the Auto Boosted clocks were successfully set to the state specified by \a enabled
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support Auto Boosted clocks
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 */
lwmlReturn_t DECLDIR lwmlDeviceSetAutoBoostedClocksEnabled(lwmlDevice_t device, lwmlEnableState_t enabled);

/**
 * Try to set the default state of Auto Boosted clocks on a device. This is the default state that Auto Boosted clocks will
 * return to when no compute running processes (e.g. LWCA application which have an active context) are running
 *
 * For Kepler &tm; or newer non-VdChip fully supported devices and Maxwell or newer VdChip devices.
 * Requires root/admin permissions.
 *
 * Auto Boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates
 * to maximize performance as thermal limits allow. Auto Boosted clocks should be disabled if fixed clock
 * rates are desired.
 *
 * On Pascal and newer hardware, Auto Boosted clocks are controlled through application clocks.
 * Use \ref lwmlDeviceSetApplicationsClocks and \ref lwmlDeviceResetApplicationsClocks to control Auto Boost
 * behavior.
 *
 * @param device                               The identifier of the target device
 * @param enabled                              What state to try to set default Auto Boosted clocks of the target device to
 * @param flags                                Flags that change the default behavior. Lwrrently Unused.
 *
 * @return
 *         - \ref LWML_SUCCESS                 If the Auto Boosted clock's default state was successfully set to the state specified by \a enabled
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_NO_PERMISSION     If the calling user does not have permission to change Auto Boosted clock's default state.
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support Auto Boosted clocks
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 */
lwmlReturn_t DECLDIR lwmlDeviceSetDefaultAutoBoostedClocksEnabled(lwmlDevice_t device, lwmlEnableState_t enabled, unsigned int flags);


/**
 * Retrieves the intended operating speed of the device's fan.
 *
 * Note: The reported speed is the intended fan speed.  If the fan is physically blocked and unable to spin, the
 * output will not match the actual fan speed.
 * 
 * For all discrete products with dedicated fans.
 *
 * The fan speed is expressed as a percentage of the product's maximum noise tolerance fan speed.
 * This value may exceed 100% in certain cases.
 *
 * @param device                               The identifier of the target device
 * @param speed                                Reference in which to return the fan speed percentage
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a speed has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a speed is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not have a fan
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetFanSpeed(lwmlDevice_t device, unsigned int *speed);


/**
 * Retrieves the intended operating speed of the device's specified fan.
 *
 * Note: The reported speed is the intended fan speed. If the fan is physically blocked and unable to spin, the
 * output will not match the actual fan speed.
 *
 * For all discrete products with dedicated fans.
 *
 * The fan speed is expressed as a percentage of the product's maximum noise tolerance fan speed.
 * This value may exceed 100% in certain cases.
 *
 * @param device                                The identifier of the target device
 * @param fan                                   The index of the target fan, zero indexed.
 * @param speed                                 Reference in which to return the fan speed percentage
 *
 * @return
 *        - \ref LWML_SUCCESS                   if \a speed has been set
 *        - \ref LWML_ERROR_UNINITIALIZED       if the library has not been successfully initialized
 *        - \ref LWML_ERROR_ILWALID_ARGUMENT    if \a device is invalid, \a fan is not an acceptable index, or \a speed is NULL
 *        - \ref LWML_ERROR_NOT_SUPPORTED       if the device does not have a fan or is newer than Maxwell
 *        - \ref LWML_ERROR_GPU_IS_LOST         if the target GPU has fallen off the bus or is otherwise inaccessible
 *        - \ref LWML_ERROR_UNKNOWN             on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetFanSpeed_v2(lwmlDevice_t device, unsigned int fan, unsigned int * speed);


/**
 * Retrieves the current temperature readings for the device, in degrees C. 
 * 
 * For all products.
 *
 * See \ref lwmlTemperatureSensors_t for details on available temperature sensors.
 *
 * @param device                               The identifier of the target device
 * @param sensorType                           Flag that indicates which sensor reading to retrieve
 * @param temp                                 Reference in which to return the temperature reading
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a temp has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, \a sensorType is invalid or \a temp is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not have the specified sensor
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetTemperature(lwmlDevice_t device, lwmlTemperatureSensors_t sensorType, unsigned int *temp);

/**
 * Retrieves the temperature threshold for the GPU with the specified threshold type in degrees C.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * See \ref lwmlTemperatureThresholds_t for details on available temperature thresholds.
 *
 * @param device                               The identifier of the target device
 * @param thresholdType                        The type of threshold value queried
 * @param temp                                 Reference in which to return the temperature reading
 * @return
 *         - \ref LWML_SUCCESS                 if \a temp has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, \a thresholdType is invalid or \a temp is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not have a temperature sensor or is unsupported
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetTemperatureThreshold(lwmlDevice_t device, lwmlTemperatureThresholds_t thresholdType, unsigned int *temp);

/**
 * Sets the temperature threshold for the GPU with the specified threshold type in degrees C.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * See \ref lwmlTemperatureThresholds_t for details on available temperature thresholds.
 *
 * @param device                               The identifier of the target device
 * @param thresholdType                        The type of threshold value to be set
 * @param temp                                 Reference which hold the value to be set
 * @return
 *         - \ref LWML_SUCCESS                 if \a temp has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, \a thresholdType is invalid or \a temp is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not have a temperature sensor or is unsupported
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceSetTemperatureThreshold(lwmlDevice_t device, lwmlTemperatureThresholds_t thresholdType, int *temp);

/**
 * Retrieves the current performance state for the device. 
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * See \ref lwmlPstates_t for details on allowed performance states.
 *
 * @param device                               The identifier of the target device
 * @param pState                               Reference in which to return the performance state reading
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a pState has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a pState is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetPerformanceState(lwmlDevice_t device, lwmlPstates_t *pState);

/**
 * Retrieves current clocks throttling reasons.
 *
 * For all fully supported products.
 *
 * \note More than one bit can be enabled at the same time. Multiple reasons can be affecting clocks at once.
 *
 * @param device                                The identifier of the target device
 * @param clocksThrottleReasons                 Reference in which to return bitmask of active clocks throttle
 *                                                  reasons
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a clocksThrottleReasons has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a clocksThrottleReasons is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlClocksThrottleReasons
 * @see lwmlDeviceGetSupportedClocksThrottleReasons
 */
lwmlReturn_t DECLDIR lwmlDeviceGetLwrrentClocksThrottleReasons(lwmlDevice_t device, unsigned long long *clocksThrottleReasons);

/**
 * Retrieves bitmask of supported clocks throttle reasons that can be returned by 
 * \ref lwmlDeviceGetLwrrentClocksThrottleReasons
 *
 * For all fully supported products.
 *
 * This method is not supported in virtual machines running virtual GPU (vGPU).
 *
 * @param device                               The identifier of the target device
 * @param supportedClocksThrottleReasons       Reference in which to return bitmask of supported
 *                                              clocks throttle reasons
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a supportedClocksThrottleReasons has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a supportedClocksThrottleReasons is NULL
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlClocksThrottleReasons
 * @see lwmlDeviceGetLwrrentClocksThrottleReasons
 */
lwmlReturn_t DECLDIR lwmlDeviceGetSupportedClocksThrottleReasons(lwmlDevice_t device, unsigned long long *supportedClocksThrottleReasons);

/**
 * Deprecated: Use \ref lwmlDeviceGetPerformanceState. This function exposes an incorrect generalization.
 *
 * Retrieve the current performance state for the device. 
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * See \ref lwmlPstates_t for details on allowed performance states.
 *
 * @param device                               The identifier of the target device
 * @param pState                               Reference in which to return the performance state reading
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a pState has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a pState is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetPowerState(lwmlDevice_t device, lwmlPstates_t *pState);

/**
 * This API has been deprecated.
 *
 * Retrieves the power management mode associated with this device.
 *
 * For products from the Fermi family.
 *     - Requires \a LWML_INFOROM_POWER version 3.0 or higher.
 *
 * For from the Kepler or newer families.
 *     - Does not require \a LWML_INFOROM_POWER object.
 *
 * This flag indicates whether any power management algorithm is lwrrently active on the device. An 
 * enabled state does not necessarily mean the device is being actively throttled -- only that 
 * that the driver will do so if the appropriate conditions are met.
 *
 * See \ref lwmlEnableState_t for details on allowed modes.
 *
 * @param device                               The identifier of the target device
 * @param mode                                 Reference in which to return the current power management mode
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a mode has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a mode is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetPowerManagementMode(lwmlDevice_t device, lwmlEnableState_t *mode);

/**
 * Retrieves the power management limit associated with this device.
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * The power limit defines the upper boundary for the card's power draw. If
 * the card's total power draw reaches this limit the power management algorithm kicks in.
 *
 * This reading is only available if power management mode is supported. 
 * See \ref lwmlDeviceGetPowerManagementMode.
 *
 * @param device                               The identifier of the target device
 * @param limit                                Reference in which to return the power management limit in milliwatts
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a limit has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a limit is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetPowerManagementLimit(lwmlDevice_t device, unsigned int *limit);

/**
 * Retrieves information about possible values of power management limits on this device.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param minLimit                             Reference in which to return the minimum power management limit in milliwatts
 * @param maxLimit                             Reference in which to return the maximum power management limit in milliwatts
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a minLimit and \a maxLimit have been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a minLimit or \a maxLimit is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlDeviceSetPowerManagementLimit
 */
lwmlReturn_t DECLDIR lwmlDeviceGetPowerManagementLimitConstraints(lwmlDevice_t device, unsigned int *minLimit, unsigned int *maxLimit);

/**
 * Retrieves default power management limit on this device, in milliwatts.
 * Default power management limit is a power management limit that the device boots with.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param defaultLimit                         Reference in which to return the default power management limit in milliwatts
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a defaultLimit has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a defaultLimit is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetPowerManagementDefaultLimit(lwmlDevice_t device, unsigned int *defaultLimit);

/**
 * Retrieves power usage for this GPU in milliwatts and its associated cirlwitry (e.g. memory)
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * On Fermi and Kepler GPUs the reading is accurate to within +/- 5% of current power draw.
 *
 * It is only available if power management mode is supported. See \ref lwmlDeviceGetPowerManagementMode.
 *
 * @param device                               The identifier of the target device
 * @param power                                Reference in which to return the power usage information
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a power has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a power is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support power readings
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetPowerUsage(lwmlDevice_t device, unsigned int *power);

/**
 * Retrieves total energy consumption for this GPU in millijoules (mJ) since the driver was last reloaded
 *
 * For Volta &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param energy                               Reference in which to return the energy consumption information
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a energy has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a energy is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support energy readings
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetTotalEnergyConsumption(lwmlDevice_t device, unsigned long long *energy);

/**
 * Get the effective power limit that the driver enforces after taking into account all limiters
 *
 * Note: This can be different from the \ref lwmlDeviceGetPowerManagementLimit if other limits are set elsewhere
 * This includes the out of band power limit interface
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                           The device to communicate with
 * @param limit                            Reference in which to return the power management limit in milliwatts
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a limit has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a limit is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetEnforcedPowerLimit(lwmlDevice_t device, unsigned int *limit);

/**
 * Retrieves the current GOM and pending GOM (the one that GPU will switch to after reboot).
 *
 * For GK110 M-class and X-class Tesla &tm; products from the Kepler family.
 * Modes \ref LWML_GOM_LOW_DP and \ref LWML_GOM_ALL_ON are supported on fully supported VdChip products.
 * Not supported on Lwdqro &reg; and Tesla &tm; C-class products.
 *
 * @param device                               The identifier of the target device
 * @param current                              Reference in which to return the current GOM
 * @param pending                              Reference in which to return the pending GOM
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a mode has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a current or \a pending is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlGpuOperationMode_t
 * @see lwmlDeviceSetGpuOperationMode
 */
lwmlReturn_t DECLDIR lwmlDeviceGetGpuOperationMode(lwmlDevice_t device, lwmlGpuOperationMode_t *current, lwmlGpuOperationMode_t *pending);

/**
 * Retrieves the amount of used, free and total memory available on the device, in bytes.
 * 
 * For all products.
 *
 * Enabling ECC reduces the amount of total available memory, due to the extra required parity bits.
 * Under WDDM most device memory is allocated and managed on startup by Windows.
 *
 * Under Linux and Windows TCC, the reported amount of used memory is equal to the sum of memory allocated 
 * by all active channels on the device.
 *
 * See \ref lwmlMemory_t for details on available memory info.
 *
 * @note In MIG mode, if device handle is provided, the API returns aggregate
 *       information, only if the caller has appropriate privileges. Per-instance
 *       information can be queried by using specific MIG device handles.
 *
 * @param device                               The identifier of the target device
 * @param memory                               Reference in which to return the memory information
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a memory has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a memory is NULL
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetMemoryInfo(lwmlDevice_t device, lwmlMemory_t *memory);

/**
 * Retrieves the current compute mode for the device.
 *
 * For all products.
 *
 * See \ref lwmlComputeMode_t for details on allowed compute modes.
 *
 * @param device                               The identifier of the target device
 * @param mode                                 Reference in which to return the current compute mode
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a mode has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a mode is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlDeviceSetComputeMode()
 */
lwmlReturn_t DECLDIR lwmlDeviceGetComputeMode(lwmlDevice_t device, lwmlComputeMode_t *mode);

/**
 * Retrieves the LWCA compute capability of the device.
 *
 * For all products.
 *
 * Returns the major and minor compute capability version numbers of the
 * device.  The major and minor versions are equivalent to the
 * LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR and
 * LW_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR attributes that would be
 * returned by LWCA's lwDeviceGetAttribute().
 *
 * @param device                               The identifier of the target device
 * @param major                                Reference in which to return the major LWCA compute capability
 * @param minor                                Reference in which to return the minor LWCA compute capability
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a major and \a minor have been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a major or \a minor are NULL
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetLwdaComputeCapability(lwmlDevice_t device, int *major, int *minor);

/**
 * Retrieves the current and pending ECC modes for the device.
 *
 * For Fermi &tm; or newer fully supported devices.
 * Only applicable to devices with ECC.
 * Requires \a LWML_INFOROM_ECC version 1.0 or higher.
 *
 * Changing ECC modes requires a reboot. The "pending" ECC mode refers to the target mode following
 * the next reboot.
 *
 * See \ref lwmlEnableState_t for details on allowed modes.
 *
 * @param device                               The identifier of the target device
 * @param current                              Reference in which to return the current ECC mode
 * @param pending                              Reference in which to return the pending ECC mode
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a current and \a pending have been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or either \a current or \a pending is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlDeviceSetEccMode()
 */
lwmlReturn_t DECLDIR lwmlDeviceGetEccMode(lwmlDevice_t device, lwmlEnableState_t *current, lwmlEnableState_t *pending);

/**
 * Retrieves the device boardId from 0-N.
 * Devices with the same boardId indicate GPUs connected to the same PLX.  Use in conjunction with 
 *  \ref lwmlDeviceGetMultiGpuBoard() to decide if they are on the same board as well.
 *  The boardId returned is a unique ID for the current configuration.  Uniqueness and ordering across 
 *  reboots and system configurations is not guaranteed (i.e. if a Tesla K40c returns 0x100 and
 *  the two GPUs on a Tesla K10 in the same system returns 0x200 it is not guaranteed they will 
 *  always return those values but they will always be different from each other).
 *  
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param boardId                              Reference in which to return the device's board ID
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a boardId has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a boardId is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetBoardId(lwmlDevice_t device, unsigned int *boardId);

/**
 * Retrieves whether the device is on a Multi-GPU Board
 * Devices that are on multi-GPU boards will set \a multiGpuBool to a non-zero value.
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param multiGpuBool                         Reference in which to return a zero or non-zero value
 *                                                 to indicate whether the device is on a multi GPU board
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a multiGpuBool has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a multiGpuBool is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetMultiGpuBoard(lwmlDevice_t device, unsigned int *multiGpuBool);

/**
 * Retrieves the total ECC error counts for the device.
 *
 * For Fermi &tm; or newer fully supported devices.
 * Only applicable to devices with ECC.
 * Requires \a LWML_INFOROM_ECC version 1.0 or higher.
 * Requires ECC Mode to be enabled.
 *
 * The total error count is the sum of errors across each of the separate memory systems, i.e. the total set of 
 * errors across the entire device.
 *
 * See \ref lwmlMemoryErrorType_t for a description of available error types.\n
 * See \ref lwmlEccCounterType_t for a description of available counter types.
 *
 * @param device                               The identifier of the target device
 * @param errorType                            Flag that specifies the type of the errors. 
 * @param counterType                          Flag that specifies the counter-type of the errors. 
 * @param eccCounts                            Reference in which to return the specified ECC errors
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a eccCounts has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device, \a errorType or \a counterType is invalid, or \a eccCounts is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlDeviceClearEccErrorCounts()
 */
lwmlReturn_t DECLDIR lwmlDeviceGetTotalEccErrors(lwmlDevice_t device, lwmlMemoryErrorType_t errorType, lwmlEccCounterType_t counterType, unsigned long long *eccCounts);

/**
 * Retrieves the detailed ECC error counts for the device.
 *
 * @deprecated   This API supports only a fixed set of ECC error locations
 *               On different GPU architectures different locations are supported
 *               See \ref lwmlDeviceGetMemoryErrorCounter
 *
 * For Fermi &tm; or newer fully supported devices.
 * Only applicable to devices with ECC.
 * Requires \a LWML_INFOROM_ECC version 2.0 or higher to report aggregate location-based ECC counts.
 * Requires \a LWML_INFOROM_ECC version 1.0 or higher to report all other ECC counts.
 * Requires ECC Mode to be enabled.
 *
 * Detailed errors provide separate ECC counts for specific parts of the memory system.
 *
 * Reports zero for unsupported ECC error counters when a subset of ECC error counters are supported.
 *
 * See \ref lwmlMemoryErrorType_t for a description of available bit types.\n
 * See \ref lwmlEccCounterType_t for a description of available counter types.\n
 * See \ref lwmlEccErrorCounts_t for a description of provided detailed ECC counts.
 *
 * @param device                               The identifier of the target device
 * @param errorType                            Flag that specifies the type of the errors. 
 * @param counterType                          Flag that specifies the counter-type of the errors. 
 * @param eccCounts                            Reference in which to return the specified ECC errors
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a eccCounts has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device, \a errorType or \a counterType is invalid, or \a eccCounts is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlDeviceClearEccErrorCounts()
 */
lwmlReturn_t DECLDIR lwmlDeviceGetDetailedEccErrors(lwmlDevice_t device, lwmlMemoryErrorType_t errorType, lwmlEccCounterType_t counterType, lwmlEccErrorCounts_t *eccCounts);

/**
 * Retrieves the requested memory error counter for the device.
 *
 * For Fermi &tm; or newer fully supported devices.
 * Requires \a LWML_INFOROM_ECC version 2.0 or higher to report aggregate location-based memory error counts.
 * Requires \a LWML_INFOROM_ECC version 1.0 or higher to report all other memory error counts.
 *
 * Only applicable to devices with ECC.
 *
 * Requires ECC Mode to be enabled.
 *
 * @note On MIG-enabled GPUs, per instance information can be queried using specific
 *       MIG device handles. Per instance information is lwrrently only supported for
 *       non-DRAM uncorrectable volatile errors. Querying volatile errors using device
 *       handles is lwrrently not supported.
 *
 * See \ref lwmlMemoryErrorType_t for a description of available memory error types.\n
 * See \ref lwmlEccCounterType_t for a description of available counter types.\n
 * See \ref lwmlMemoryLocation_t for a description of available counter locations.\n
 * 
 * @param device                               The identifier of the target device
 * @param errorType                            Flag that specifies the type of error.
 * @param counterType                          Flag that specifies the counter-type of the errors. 
 * @param locationType                         Specifies the location of the counter. 
 * @param count                                Reference in which to return the ECC counter
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a count has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device, \a bitTyp,e \a counterType or \a locationType is
 *                                             invalid, or \a count is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support ECC error reporting in the specified memory
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetMemoryErrorCounter(lwmlDevice_t device, lwmlMemoryErrorType_t errorType,
                                                   lwmlEccCounterType_t counterType,
                                                   lwmlMemoryLocation_t locationType, unsigned long long *count);

/**
 * Retrieves the current utilization rates for the device's major subsystems.
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * See \ref lwmlUtilization_t for details on available utilization rates.
 *
 * \note During driver initialization when ECC is enabled one can see high GPU and Memory Utilization readings.
 *       This is caused by ECC Memory Scrubbing mechanism that is performed during driver initialization.
 *
 * @note On MIG-enabled GPUs, querying device utilization rates is not lwrrently supported.
 *
 * @param device                               The identifier of the target device
 * @param utilization                          Reference in which to return the utilization information
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a utilization has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a utilization is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetUtilizationRates(lwmlDevice_t device, lwmlUtilization_t *utilization);

/**
 * Retrieves the current utilization and sampling size in microseconds for the Encoder
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @note On MIG-enabled GPUs, querying encoder utilization is not lwrrently supported.
 *
 * @param device                               The identifier of the target device
 * @param utilization                          Reference to an unsigned int for encoder utilization info
 * @param samplingPeriodUs                     Reference to an unsigned int for the sampling period in US
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a utilization has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, \a utilization is NULL, or \a samplingPeriodUs is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetEncoderUtilization(lwmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs);

/**
 * Retrieves the current capacity of the device's encoder, as a percentage of maximum encoder capacity with valid values in the range 0-100.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param device                            The identifier of the target device
 * @param encoderQueryType                  Type of encoder to query
 * @param encoderCapacity                   Reference to an unsigned int for the encoder capacity
 * 
 * @return
 *         - \ref LWML_SUCCESS                  if \a encoderCapacity is fetched
 *         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a encoderCapacity is NULL, or \a device or \a encoderQueryType
 *                                              are invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED      if device does not support the encoder specified in \a encodeQueryType
 *         - \ref LWML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetEncoderCapacity (lwmlDevice_t device, lwmlEncoderType_t encoderQueryType, unsigned int *encoderCapacity);

/**
 * Retrieves the current encoder statistics for a given device.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param device                            The identifier of the target device
 * @param sessionCount                      Reference to an unsigned int for count of active encoder sessions
 * @param averageFps                        Reference to an unsigned int for trailing average FPS of all active sessions
 * @param averageLatency                    Reference to an unsigned int for encode latency in microseconds
 * 
 * @return
 *         - \ref LWML_SUCCESS                  if \a sessionCount, \a averageFps and \a averageLatency is fetched
 *         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a sessionCount, or \a device or \a averageFps,
 *                                              or \a averageLatency is NULL
 *         - \ref LWML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetEncoderStats (lwmlDevice_t device, unsigned int *sessionCount,
                                                unsigned int *averageFps, unsigned int *averageLatency);

/**
 * Retrieves information about active encoder sessions on a target device.
 *
 * An array of active encoder sessions is returned in the caller-supplied buffer pointed at by \a sessionInfos. The
 * array elememt count is passed in \a sessionCount, and \a sessionCount is used to return the number of sessions
 * written to the buffer.
 *
 * If the supplied buffer is not large enough to accomodate the active session array, the function returns
 * LWML_ERROR_INSUFFICIENT_SIZE, with the element count of lwmlEncoderSessionInfo_t array required in \a sessionCount.
 * To query the number of active encoder sessions, call this function with *sessionCount = 0.  The code will return
 * LWML_SUCCESS with number of active encoder sessions updated in *sessionCount.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param device                            The identifier of the target device
 * @param sessionCount                      Reference to caller supplied array size, and returns the number of sessions.
 * @param sessionInfos                      Reference in which to return the session information
 * 
 * @return
 *         - \ref LWML_SUCCESS                  if \a sessionInfos is fetched
 *         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE  if \a sessionCount is too small, array element count is returned in \a sessionCount
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a sessionCount is NULL.
 *         - \ref LWML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_NOT_SUPPORTED      if this query is not supported by \a device
 *         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetEncoderSessions(lwmlDevice_t device, unsigned int *sessionCount, lwmlEncoderSessionInfo_t *sessionInfos);

/**
 * Retrieves the current utilization and sampling size in microseconds for the Decoder
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @note On MIG-enabled GPUs, querying decoder utilization is not lwrrently supported.
 *
 * @param device                               The identifier of the target device
 * @param utilization                          Reference to an unsigned int for decoder utilization info
 * @param samplingPeriodUs                     Reference to an unsigned int for the sampling period in US
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a utilization has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, \a utilization is NULL, or \a samplingPeriodUs is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetDecoderUtilization(lwmlDevice_t device, unsigned int *utilization, unsigned int *samplingPeriodUs);

/**
* Retrieves the active frame buffer capture sessions statistics for a given device.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param device                            The identifier of the target device
* @param fbcStats                          Reference to lwmlFBCStats_t structure contianing LwFBC stats
*
* @return
*         - \ref LWML_SUCCESS                  if \a fbcStats is fetched
*         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a fbcStats is NULL
*         - \ref LWML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
*/
lwmlReturn_t DECLDIR lwmlDeviceGetFBCStats(lwmlDevice_t device, lwmlFBCStats_t *fbcStats);

/**
* Retrieves information about active frame buffer capture sessions on a target device.
*
* An array of active FBC sessions is returned in the caller-supplied buffer pointed at by \a sessionInfo. The
* array element count is passed in \a sessionCount, and \a sessionCount is used to return the number of sessions
* written to the buffer.
*
* If the supplied buffer is not large enough to accomodate the active session array, the function returns
* LWML_ERROR_INSUFFICIENT_SIZE, with the element count of lwmlFBCSessionInfo_t array required in \a sessionCount.
* To query the number of active FBC sessions, call this function with *sessionCount = 0.  The code will return
* LWML_SUCCESS with number of active FBC sessions updated in *sessionCount.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @note hResolution, vResolution, averageFPS and averageLatency data for a FBC session returned in \a sessionInfo may
*       be zero if there are no new frames captured since the session started.
*
* @param device                            The identifier of the target device
* @param sessionCount                      Reference to caller supplied array size, and returns the number of sessions.
* @param sessionInfo                       Reference in which to return the session information
*
* @return
*         - \ref LWML_SUCCESS                  if \a sessionInfo is fetched
*         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref LWML_ERROR_INSUFFICIENT_SIZE  if \a sessionCount is too small, array element count is returned in \a sessionCount
*         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a sessionCount is NULL.
*         - \ref LWML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
*         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
*/
lwmlReturn_t DECLDIR lwmlDeviceGetFBCSessions(lwmlDevice_t device, unsigned int *sessionCount, lwmlFBCSessionInfo_t *sessionInfo);

/**
 * Retrieves the current and pending driver model for the device.
 *
 * For Fermi &tm; or newer fully supported devices.
 * For windows only.
 *
 * On Windows platforms the device driver can run in either WDDM or WDM (TCC) mode. If a display is attached
 * to the device it must run in WDDM mode. TCC mode is preferred if a display is not attached.
 *
 * See \ref lwmlDriverModel_t for details on available driver models.
 *
 * @param device                               The identifier of the target device
 * @param current                              Reference in which to return the current driver model
 * @param pending                              Reference in which to return the pending driver model
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if either \a current and/or \a pending have been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or both \a current and \a pending are NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the platform is not windows
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 * 
 * @see lwmlDeviceSetDriverModel()
 */
lwmlReturn_t DECLDIR lwmlDeviceGetDriverModel(lwmlDevice_t device, lwmlDriverModel_t *current, lwmlDriverModel_t *pending);

/**
 * Get VBIOS version of the device.
 *
 * For all products.
 *
 * The VBIOS version may change from time to time. It will not exceed 32 characters in length 
 * (including the NULL terminator).  See \ref lwmlConstants::LWML_DEVICE_VBIOS_VERSION_BUFFER_SIZE.
 *
 * @param device                               The identifier of the target device
 * @param version                              Reference to which to return the VBIOS version
 * @param length                               The maximum allowed length of the string returned in \a version
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a version has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, or \a version is NULL
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a length is too small 
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetVbiosVersion(lwmlDevice_t device, char *version, unsigned int length);

/**
 * Get Bridge Chip Information for all the bridge chips on the board.
 * 
 * For all fully supported products.
 * Only applicable to multi-GPU products.
 * 
 * @param device                                The identifier of the target device
 * @param bridgeHierarchy                       Reference to the returned bridge chip Hierarchy
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if bridge chip exists
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, or \a bridgeInfo is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if bridge chip not supported on the device
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 * 
 */
lwmlReturn_t DECLDIR lwmlDeviceGetBridgeChipInfo(lwmlDevice_t device, lwmlBridgeChipHierarchy_t *bridgeHierarchy);

/**
 * Get information about processes with a compute context on a device
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * This function returns information only about compute running processes (e.g. LWCA application which have
 * active context). Any graphics applications (e.g. using OpenGL, DirectX) won't be listed by this function.
 *
 * To query the current number of running compute processes, call this function with *infoCount = 0. The
 * return code will be LWML_ERROR_INSUFFICIENT_SIZE, or LWML_SUCCESS if none are running. For this call
 * \a infos is allowed to be NULL.
 *
 * The usedGpuMemory field returned is all of the memory used by the application.
 *
 * Keep in mind that information returned by this call is dynamic and the number of elements might change in
 * time. Allocate more space for \a infos table in case new compute processes are spawned.
 *
 * @note In MIG mode, if device handle is provided, the API returns aggregate information, only if
 *       the caller has appropriate privileges. Per-instance information can be queried by using
 *       specific MIG device handles.
 *       Querying per-instance information using MIG device handles is not supported if the device is in vGPU Host virtualization mode.
 *
 * @param device                               The device handle or MIG device handle
 * @param infoCount                            Reference in which to provide the \a infos array size, and
 *                                             to return the number of returned elements
 * @param infos                                Reference in which to return the process information
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a infoCount and \a infos have been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a infoCount indicates that the \a infos array is too small
 *                                             \a infoCount will contain minimal amount of space necessary for
 *                                             the call to complete
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, either of \a infoCount or \a infos is NULL
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this query is not supported by \a device
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see \ref lwmlSystemGetProcessName
 */
lwmlReturn_t DECLDIR lwmlDeviceGetComputeRunningProcesses_v2(lwmlDevice_t device, unsigned int *infoCount, lwmlProcessInfo_t *infos);

/**
 * Get information about processes with a graphics context on a device
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * This function returns information only about graphics based processes
 * (eg. applications using OpenGL, DirectX)
 *
 * To query the current number of running graphics processes, call this function with *infoCount = 0. The
 * return code will be LWML_ERROR_INSUFFICIENT_SIZE, or LWML_SUCCESS if none are running. For this call
 * \a infos is allowed to be NULL.
 *
 * The usedGpuMemory field returned is all of the memory used by the application.
 *
 * Keep in mind that information returned by this call is dynamic and the number of elements might change in
 * time. Allocate more space for \a infos table in case new graphics processes are spawned.
 *
 * @note In MIG mode, if device handle is provided, the API returns aggregate information, only if
 *       the caller has appropriate privileges. Per-instance information can be queried by using
 *       specific MIG device handles.
 *       Querying per-instance information using MIG device handles is not supported if the device is in vGPU Host virtualization mode.
 *
 * @param device                               The identifier of the target device
 * @param infoCount                            Reference in which to provide the \a infos array size, and
 *                                             to return the number of returned elements
 * @param infos                                Reference in which to return the process information
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a infoCount and \a infos have been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a infoCount indicates that the \a infos array is too small
 *                                             \a infoCount will contain minimal amount of space necessary for
 *                                             the call to complete
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, either of \a infoCount or \a infos is NULL
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this query is not supported by \a device
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see \ref lwmlSystemGetProcessName
 */
lwmlReturn_t DECLDIR lwmlDeviceGetGraphicsRunningProcesses_v2(lwmlDevice_t device, unsigned int *infoCount, lwmlProcessInfo_t *infos);

/**
 * Check if the GPU devices are on the same physical board.
 *
 * For all fully supported products.
 *
 * @param device1                               The first GPU device
 * @param device2                               The second GPU device
 * @param onSameBoard                           Reference in which to return the status.
 *                                              Non-zero indicates that the GPUs are on the same board.
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a onSameBoard has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a dev1 or \a dev2 are invalid or \a onSameBoard is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this check is not supported by the device
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the either GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceOnSameBoard(lwmlDevice_t device1, lwmlDevice_t device2, int *onSameBoard);

/**
 * Retrieves the root/admin permissions on the target API. See \a lwmlRestrictedAPI_t for the list of supported APIs.
 * If an API is restricted only root users can call that API. See \a lwmlDeviceSetAPIRestriction to change current permissions.
 *
 * For all fully supported products.
 *
 * @param device                               The identifier of the target device
 * @param apiType                              Target API type for this operation
 * @param isRestricted                         Reference in which to return the current restriction 
 *                                             LWML_FEATURE_ENABLED indicates that the API is root-only
 *                                             LWML_FEATURE_DISABLED indicates that the API is accessible to all users
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a isRestricted has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, \a apiType incorrect or \a isRestricted is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this query is not supported by the device or the device does not support
 *                                                 the feature that is being queried (E.G. Enabling/disabling Auto Boosted clocks is
 *                                                 not supported by the device)
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlRestrictedAPI_t
 */
lwmlReturn_t DECLDIR lwmlDeviceGetAPIRestriction(lwmlDevice_t device, lwmlRestrictedAPI_t apiType, lwmlEnableState_t *isRestricted);

/**
 * Gets recent samples for the GPU.
 * 
 * For Kepler &tm; or newer fully supported devices.
 * 
 * Based on type, this method can be used to fetch the power, utilization or clock samples maintained in the buffer by 
 * the driver.
 * 
 * Power, Utilization and Clock samples are returned as type "unsigned int" for the union lwmlValue_t.
 * 
 * To get the size of samples that user needs to allocate, the method is ilwoked with samples set to NULL. 
 * The returned samplesCount will provide the number of samples that can be queried. The user needs to 
 * allocate the buffer with size as samplesCount * sizeof(lwmlSample_t).
 * 
 * lastSeenTimeStamp represents CPU timestamp in microseconds. Set it to 0 to fetch all the samples maintained by the 
 * underlying buffer. Set lastSeenTimeStamp to one of the timeStamps retrieved from the date of the previous query 
 * to get more recent samples.
 * 
 * This method fetches the number of entries which can be accommodated in the provided samples array, and the 
 * reference samplesCount is updated to indicate how many samples were actually retrieved. The advantage of using this 
 * method for samples in contrast to polling via existing methods is to get get higher frequency data at lower polling cost.
 *
 * @note On MIG-enabled GPUs, querying the following sample types, LWML_GPU_UTILIZATION_SAMPLES, LWML_MEMORY_UTILIZATION_SAMPLES
 *       LWML_ENC_UTILIZATION_SAMPLES and LWML_DEC_UTILIZATION_SAMPLES, is not lwrrently supported.
 *
 * @param device                        The identifier for the target device
 * @param type                          Type of sampling event
 * @param lastSeenTimeStamp             Return only samples with timestamp greater than lastSeenTimeStamp. 
 * @param sampleValType                 Output parameter to represent the type of sample value as described in lwmlSampleVal_t
 * @param sampleCount                   Reference to provide the number of elements which can be queried in samples array
 * @param samples                       Reference in which samples are returned
 
 * @return 
 *         - \ref LWML_SUCCESS                 if samples are successfully retrieved
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, \a samplesCount is NULL or 
 *                                             reference to \a sampleCount is 0 for non null \a samples
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_NOT_FOUND         if sample entries are not found
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetSamples(lwmlDevice_t device, lwmlSamplingType_t type, unsigned long long lastSeenTimeStamp,
        lwmlValueType_t *sampleValType, unsigned int *sampleCount, lwmlSample_t *samples);

/**
 * Gets Total, Available and Used size of BAR1 memory.
 * 
 * BAR1 is used to map the FB (device memory) so that it can be directly accessed by the CPU or by 3rd party 
 * devices (peer-to-peer on the PCIE bus). 
 *
 * @note In MIG mode, if device handle is provided, the API returns aggregate
 *       information, only if the caller has appropriate privileges. Per-instance
 *       information can be queried by using specific MIG device handles.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param bar1Memory                           Reference in which BAR1 memory
 *                                             information is returned.
 *
 * @return
 *         - \ref LWML_SUCCESS                 if BAR1 memory is successfully retrieved
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, \a bar1Memory is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 */
lwmlReturn_t DECLDIR lwmlDeviceGetBAR1MemoryInfo(lwmlDevice_t device, lwmlBAR1Memory_t *bar1Memory);

/**
 * Gets the duration of time during which the device was throttled (lower than requested clocks) due to power 
 * or thermal constraints.
 *
 * The method is important to users who are tying to understand if their GPUs throttle at any point during their applications. The
 * difference in violation times at two different reference times gives the indication of GPU throttling event. 
 *
 * Violation for thermal capping is not supported at this time.
 * 
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param perfPolicyType                       Represents Performance policy which can trigger GPU throttling
 * @param violTime                             Reference to which violation time related information is returned 
 *                                         
 *
 * @return
 *         - \ref LWML_SUCCESS                 if violation time is successfully retrieved
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, \a perfPolicyType is invalid, or \a violTime is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *
 */
lwmlReturn_t DECLDIR lwmlDeviceGetViolationStatus(lwmlDevice_t device, lwmlPerfPolicyType_t perfPolicyType, lwmlViolationTime_t *violTime);

/**
 * @}
 */

/** @addtogroup lwmlAccountingStats
 *  @{
 */

/**
 * Queries the state of per process accounting mode.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * See \ref lwmlDeviceGetAccountingStats for more details.
 * See \ref lwmlDeviceSetAccountingMode
 *
 * @param device                               The identifier of the target device
 * @param mode                                 Reference in which to return the current accounting mode
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if the mode has been successfully retrieved 
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a mode are NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetAccountingMode(lwmlDevice_t device, lwmlEnableState_t *mode);

/**
 * Queries process's accounting stats.
 *
 * For Kepler &tm; or newer fully supported devices.
 * 
 * Accounting stats capture GPU utilization and other statistics across the lifetime of a process.
 * Accounting stats can be queried during life time of the process and after its termination.
 * The time field in \ref lwmlAccountingStats_t is reported as 0 during the lifetime of the process and 
 * updated to actual running time after its termination.
 * Accounting stats are kept in a cirlwlar buffer, newly created processes overwrite information about old
 * processes.
 *
 * See \ref lwmlAccountingStats_t for description of each returned metric.
 * List of processes that can be queried can be retrieved from \ref lwmlDeviceGetAccountingPids.
 *
 * @note Accounting Mode needs to be on. See \ref lwmlDeviceGetAccountingMode.
 * @note Only compute and graphics applications stats can be queried. Monitoring applications stats can't be
 *         queried since they don't contribute to GPU utilization.
 * @note In case of pid collision stats of only the latest process (that terminated last) will be reported
 *
 * @warning On Kepler devices per process statistics are accurate only if there's one process running on a GPU.
 * 
 * @param device                               The identifier of the target device
 * @param pid                                  Process Id of the target process to query stats for
 * @param stats                                Reference in which to return the process's accounting stats
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if stats have been successfully retrieved
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a stats are NULL
 *         - \ref LWML_ERROR_NOT_FOUND         if process stats were not found
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if \a device doesn't support this feature or accounting mode is disabled
 *                                              or on vGPU host.
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlDeviceGetAccountingBufferSize
 */
lwmlReturn_t DECLDIR lwmlDeviceGetAccountingStats(lwmlDevice_t device, unsigned int pid, lwmlAccountingStats_t *stats);

/**
 * Queries list of processes that can be queried for accounting stats. The list of processes returned 
 * can be in running or terminated state.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * To just query the number of processes ready to be queried, call this function with *count = 0 and
 * pids=NULL. The return code will be LWML_ERROR_INSUFFICIENT_SIZE, or LWML_SUCCESS if list is empty.
 * 
 * For more details see \ref lwmlDeviceGetAccountingStats.
 *
 * @note In case of PID collision some processes might not be accessible before the cirlwlar buffer is full.
 *
 * @param device                               The identifier of the target device
 * @param count                                Reference in which to provide the \a pids array size, and
 *                                               to return the number of elements ready to be queried
 * @param pids                                 Reference in which to return list of process ids
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if pids were successfully retrieved
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a count is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if \a device doesn't support this feature or accounting mode is disabled
 *                                              or on vGPU host.
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a count is too small (\a count is set to
 *                                                 expected value)
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlDeviceGetAccountingBufferSize
 */
lwmlReturn_t DECLDIR lwmlDeviceGetAccountingPids(lwmlDevice_t device, unsigned int *count, unsigned int *pids);

/**
 * Returns the number of processes that the cirlwlar buffer with accounting pids can hold.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * This is the maximum number of processes that accounting information will be stored for before information
 * about oldest processes will get overwritten by information about new processes.
 *
 * @param device                               The identifier of the target device
 * @param bufferSize                           Reference in which to provide the size (in number of elements)
 *                                               of the cirlwlar buffer for accounting stats.
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if buffer size was successfully retrieved
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a bufferSize is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature or accounting mode is disabled
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 * 
 * @see lwmlDeviceGetAccountingStats
 * @see lwmlDeviceGetAccountingPids
 */
lwmlReturn_t DECLDIR lwmlDeviceGetAccountingBufferSize(lwmlDevice_t device, unsigned int *bufferSize);

/** @} */

/** @addtogroup lwmlDeviceQueries
 *  @{
 */

/**
 * Returns the list of retired pages by source, including pages that are pending retirement
 * The address information provided from this API is the hardware address of the page that was retired.  Note
 * that this does not match the virtual address used in LWCA, but will match the address information in XID 63
 * 
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                            The identifier of the target device
 * @param cause                             Filter page addresses by cause of retirement
 * @param pageCount                         Reference in which to provide the \a addresses buffer size, and
 *                                          to return the number of retired pages that match \a cause
 *                                          Set to 0 to query the size without allocating an \a addresses buffer
 * @param addresses                         Buffer to write the page addresses into
 * 
 * @return
 *         - \ref LWML_SUCCESS                 if \a pageCount was populated and \a addresses was filled
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a pageCount indicates the buffer is not large enough to store all the
 *                                             matching page addresses.  \a pageCount is set to the needed size.
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, \a pageCount is NULL, \a cause is invalid, or 
 *                                             \a addresses is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetRetiredPages(lwmlDevice_t device, lwmlPageRetirementCause_t cause,
    unsigned int *pageCount, unsigned long long *addresses);

/**
 * Returns the list of retired pages by source, including pages that are pending retirement
 * The address information provided from this API is the hardware address of the page that was retired.  Note
 * that this does not match the virtual address used in LWCA, but will match the address information in XID 63
 *
 * \note lwmlDeviceGetRetiredPages_v2 adds an additional timestamps paramter to return the time of each page's
 *       retirement.
 * 
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                            The identifier of the target device
 * @param cause                             Filter page addresses by cause of retirement
 * @param pageCount                         Reference in which to provide the \a addresses buffer size, and
 *                                          to return the number of retired pages that match \a cause
 *                                          Set to 0 to query the size without allocating an \a addresses buffer
 * @param addresses                         Buffer to write the page addresses into
 * @param timestamps                        Buffer to write the timestamps of page retirement, additional for _v2
 * 
 * @return
 *         - \ref LWML_SUCCESS                 if \a pageCount was populated and \a addresses was filled
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a pageCount indicates the buffer is not large enough to store all the
 *                                             matching page addresses.  \a pageCount is set to the needed size.
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, \a pageCount is NULL, \a cause is invalid, or 
 *                                             \a addresses is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetRetiredPages_v2(lwmlDevice_t device, lwmlPageRetirementCause_t cause,
    unsigned int *pageCount, unsigned long long *addresses, unsigned long long *timestamps);

/**
 * Check if any pages are pending retirement and need a reboot to fully retire.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                            The identifier of the target device
 * @param isPending                         Reference in which to return the pending status
 * 
 * @return
 *         - \ref LWML_SUCCESS                 if \a isPending was populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a isPending is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetRetiredPagesPendingStatus(lwmlDevice_t device, lwmlEnableState_t *isPending);

/**
 * Get number of remapped rows. The number of rows reported will be based on
 * the cause of the remapping. isPending indicates whether or not there are
 * pending remappings. A reset will be required to actually remap the row.
 * failureOclwrred will be set if a row remapping ever failed in the past. A
 * pending remapping won't affect future work on the GPU since
 * error-containment and dynamic page blacklisting will take care of that.
 *
 * @note On MIG-enabled GPUs with active instances, querying the number of
 * remapped rows is not supported
 *
 * For Ampere &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param corrRows                             Reference for number of rows remapped due to correctable errors
 * @param uncRows                              Reference for number of rows remapped due to uncorrectable errors
 * @param isPending                            Reference for whether or not remappings are pending
 * @param failureOclwrred                      Reference that is set when a remapping has failed in the past
 *
 * @return
 *         - \ref LWML_SUCCESS                 Upon success
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  If \a corrRows, \a uncRows, \a isPending or \a failureOclwrred is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     If MIG is enabled or if the device doesn't support this feature
 *         - \ref LWML_ERROR_UNKNOWN           Unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetRemappedRows(lwmlDevice_t device, unsigned int *corrRows, unsigned int *uncRows,
                                               unsigned int *isPending, unsigned int *failureOclwrred);

/**
 * Get the row remapper histogram. Returns the remap availability for each bank
 * on the GPU.
 *
 * @param device                               Device handle
 * @param values                               Histogram values
 *
 * @return
 *        - \ref LWML_SUCCESS                  On success
 *        - \ref LWML_ERROR_UNKNOWN            On any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetRowRemapperHistogram(lwmlDevice_t device, lwmlRowRemapperHistogramValues_t *values);

/**
 * Get architecture for device
 *
 * @param device                               The identifier of the target device
 * @param arch                                 Reference where architecture is returned, if call successful.
 *                                             Set to LWML_DEVICE_ARCH_* upon success
 *
 * @return
 *         - \ref LWML_SUCCESS                 Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  If \a device or \a arch (output refererence) are invalid
 */
lwmlReturn_t DECLDIR lwmlDeviceGetArchitecture(lwmlDevice_t device, lwmlDeviceArchitecture_t *arch);

/** @} */

/***************************************************************************************************/
/** @defgroup lwmlUnitCommands Unit Commands
 *  This chapter describes LWML operations that change the state of the unit. For S-class products.
 *  Each of these requires root/admin access. Non-admin users will see an LWML_ERROR_NO_PERMISSION
 *  error code when ilwoking any of these methods.
 *  @{
 */
/***************************************************************************************************/

/**
 * Set the LED state for the unit. The LED can be either green (0) or amber (1).
 *
 * For S-class products.
 * Requires root/admin permissions.
 *
 * This operation takes effect immediately.
 * 
 *
 * <b>Current S-Class products don't provide unique LEDs for each unit. As such, both front 
 * and back LEDs will be toggled in unison regardless of which unit is specified with this command.</b>
 *
 * See \ref lwmlLedColor_t for available colors.
 *
 * @param unit                                 The identifier of the target unit
 * @param color                                The target LED color
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if the LED color has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a unit or \a color is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this is not an S-class product
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 * 
 * @see lwmlUnitGetLedState()
 */
lwmlReturn_t DECLDIR lwmlUnitSetLedState(lwmlUnit_t unit, lwmlLedColor_t color);

/** @} */

/***************************************************************************************************/
/** @defgroup lwmlDeviceCommands Device Commands
 *  This chapter describes LWML operations that change the state of the device.
 *  Each of these requires root/admin access. Non-admin users will see an LWML_ERROR_NO_PERMISSION
 *  error code when ilwoking any of these methods.
 *  @{
 */
/***************************************************************************************************/

/**
 * Set the persistence mode for the device.
 *
 * For all products.
 * For Linux only.
 * Requires root/admin permissions.
 *
 * The persistence mode determines whether the GPU driver software is torn down after the last client
 * exits.
 *
 * This operation takes effect immediately. It is not persistent across reboots. After each reboot the
 * persistence mode is reset to "Disabled".
 *
 * See \ref lwmlEnableState_t for available modes.
 *
 * After calling this API with mode set to LWML_FEATURE_DISABLED on a device that has its own NUMA
 * memory, the given device handle will no longer be valid, and to continue to interact with this
 * device, a new handle should be obtained from one of the lwmlDeviceGetHandleBy*() APIs. This
 * limitation is lwrrently only applicable to devices that have a coherent LWLink connection to
 * system memory.
 *
 * @param device                               The identifier of the target device
 * @param mode                                 The target persistence mode
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if the persistence mode was set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a mode is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlDeviceGetPersistenceMode()
 */
lwmlReturn_t DECLDIR lwmlDeviceSetPersistenceMode(lwmlDevice_t device, lwmlEnableState_t mode);

/**
 * Set the compute mode for the device.
 *
 * For all products.
 * Requires root/admin permissions.
 *
 * The compute mode determines whether a GPU can be used for compute operations and whether it can
 * be shared across contexts.
 *
 * This operation takes effect immediately. Under Linux it is not persistent across reboots and
 * always resets to "Default". Under windows it is persistent.
 *
 * Under windows compute mode may only be set to DEFAULT when running in WDDM
 *
 * @note On MIG-enabled GPUs, compute mode would be set to DEFAULT and changing it is not supported.
 *
 * See \ref lwmlComputeMode_t for details on available compute modes.
 *
 * @param device                               The identifier of the target device
 * @param mode                                 The target compute mode
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if the compute mode was set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a mode is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlDeviceGetComputeMode()
 */
lwmlReturn_t DECLDIR lwmlDeviceSetComputeMode(lwmlDevice_t device, lwmlComputeMode_t mode);

/**
 * Set the ECC mode for the device.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Only applicable to devices with ECC.
 * Requires \a LWML_INFOROM_ECC version 1.0 or higher.
 * Requires root/admin permissions.
 *
 * The ECC mode determines whether the GPU enables its ECC support.
 *
 * This operation takes effect after the next reboot.
 *
 * See \ref lwmlEnableState_t for details on available modes.
 *
 * @param device                               The identifier of the target device
 * @param ecc                                  The target ECC mode
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if the ECC mode was set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a ecc is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlDeviceGetEccMode()
 */
lwmlReturn_t DECLDIR lwmlDeviceSetEccMode(lwmlDevice_t device, lwmlEnableState_t ecc);  

/**
 * Clear the ECC error and other memory error counts for the device.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Only applicable to devices with ECC.
 * Requires \a LWML_INFOROM_ECC version 2.0 or higher to clear aggregate location-based ECC counts.
 * Requires \a LWML_INFOROM_ECC version 1.0 or higher to clear all other ECC counts.
 * Requires root/admin permissions.
 * Requires ECC Mode to be enabled.
 *
 * Sets all of the specified ECC counters to 0, including both detailed and total counts.
 *
 * This operation takes effect immediately.
 *
 * See \ref lwmlMemoryErrorType_t for details on available counter types.
 *
 * @param device                               The identifier of the target device
 * @param counterType                          Flag that indicates which type of errors should be cleared.
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if the error counts were cleared
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a counterType is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see 
 *      - lwmlDeviceGetDetailedEccErrors()
 *      - lwmlDeviceGetTotalEccErrors()
 */
lwmlReturn_t DECLDIR lwmlDeviceClearEccErrorCounts(lwmlDevice_t device, lwmlEccCounterType_t counterType);

/**
 * Set the driver model for the device.
 *
 * For Fermi &tm; or newer fully supported devices.
 * For windows only.
 * Requires root/admin permissions.
 *
 * On Windows platforms the device driver can run in either WDDM or WDM (TCC) mode. If a display is attached
 * to the device it must run in WDDM mode.  
 *
 * It is possible to force the change to WDM (TCC) while the display is still attached with a force flag (lwmlFlagForce).
 * This should only be done if the host is subsequently powered down and the display is detached from the device
 * before the next reboot. 
 *
 * This operation takes effect after the next reboot.
 * 
 * Windows driver model may only be set to WDDM when running in DEFAULT compute mode.
 *
 * Change driver model to WDDM is not supported when GPU doesn't support graphics acceleration or 
 * will not support it after reboot. See \ref lwmlDeviceSetGpuOperationMode.
 *
 * See \ref lwmlDriverModel_t for details on available driver models.
 * See \ref lwmlFlagDefault and \ref lwmlFlagForce
 *
 * @param device                               The identifier of the target device
 * @param driverModel                          The target driver model
 * @param flags                                Flags that change the default behavior
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if the driver model has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a driverModel is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the platform is not windows or the device does not support this feature
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 * 
 * @see lwmlDeviceGetDriverModel()
 */
lwmlReturn_t DECLDIR lwmlDeviceSetDriverModel(lwmlDevice_t device, lwmlDriverModel_t driverModel, unsigned int flags);

typedef enum lwmlClockLimitId_enum {
    LWML_CLOCK_LIMIT_ID_RANGE_START = 0xffffff00,
    LWML_CLOCK_LIMIT_ID_TDP,
    LWML_CLOCK_LIMIT_ID_UNLIMITED
} lwmlClockLimitId_t;

/**
 * Set clocks that device will lock to.
 *
 * Sets the clocks that the device will be running at to the value in the range of minGpuClockMHz to maxGpuClockMHz.
 * Setting this will supercede application clock values and take effect regardless if a lwca app is running.
 * See /ref lwmlDeviceSetApplicationsClocks
 *
 * Can be used as a setting to request constant performance.
 *
 * This can be called with a pair of integer clock frequencies in MHz, or a pair of /ref lwmlClockLimitId_t values.
 * See the table below for valid combinations of these values.
 *
 * minGpuClock | maxGpuClock | Effect
 * ------------+-------------+--------------------------------------------------
 *     tdp     |     tdp     | Lock clock to TDP
 *  unlimited  |     tdp     | Upper bound is TDP but clock may drift below this
 *     tdp     |  unlimited  | Lower bound is TDP but clock may boost above this
 *  unlimited  |  unlimited  | Unlocked (== lwmlDeviceResetGpuLockedClocks)
 *
 * If one arg takes one of these values, the other must be one of these values as
 * well. Mixed numeric and symbolic calls return LWML_ERROR_ILWALID_ARGUMENT.
 *
 * Requires root/admin permissions.
 *
 * After system reboot or driver reload applications clocks go back to their default value.
 * See \ref lwmlDeviceResetGpuLockedClocks.
 *
 * For Volta &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param minGpuClockMHz                       Requested minimum gpu clock in MHz
 * @param maxGpuClockMHz                       Requested maximum gpu clock in MHz
 *
 * @return
 *         - \ref LWML_SUCCESS                 if new settings were successfully set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a minGpuClockMHz and \a maxGpuClockMHz
 *                                                 is not a valid clock combination
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceSetGpuLockedClocks(lwmlDevice_t device, unsigned int minGpuClockMHz, unsigned int maxGpuClockMHz);

/**
 * Resets the gpu clock to the default value
 *
 * This is the gpu clock that will be used after system reboot or driver reload.
 * Default values are idle clocks, but the current values can be changed using \ref lwmlDeviceSetApplicationsClocks.
 *
 * @see lwmlDeviceSetGpuLockedClocks
 *
 * For Volta &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 *
 * @return
 *         - \ref LWML_SUCCESS                 if new settings were successfully set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceResetGpuLockedClocks(lwmlDevice_t device);

/**
 * Set memory clocks that device will lock to.
 *
 * Sets the device's memory clocks to the value in the range of minMemClockMHz to maxMemClockMHz.
 * Setting this will supersede application clock values and take effect regardless of whether a lwca app is running.
 * See /ref lwmlDeviceSetApplicationsClocks
 *
 * Can be used as a setting to request constant performance.
 *
 * Requires root/admin permissions.
 *
 * After system reboot or driver reload applications clocks go back to their default value.
 * See \ref lwmlDeviceResetMemoryLockedClocks.
 *
 * For Ampere &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param minMemClockMHz                       Requested minimum memory clock in MHz
 * @param maxMemClockMHz                       Requested maximum memory clock in MHz
 *
 * @return
 *         - \ref LWML_SUCCESS                 if new settings were successfully set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a minGpuClockMHz and \a maxGpuClockMHz
 *                                                 is not a valid clock combination
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceSetMemoryLockedClocks(lwmlDevice_t device, unsigned int minMemClockMHz, unsigned int maxMemClockMHz);

/**
 * Resets the memory clock to the default value
 *
 * This is the memory clock that will be used after system reboot or driver reload.
 * Default values are idle clocks, but the current values can be changed using \ref lwmlDeviceSetApplicationsClocks.
 *
 * @see lwmlDeviceSetMemoryLockedClocks
 *
 * For Ampere &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 *
 * @return
 *         - \ref LWML_SUCCESS                 if new settings were successfully set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceResetMemoryLockedClocks(lwmlDevice_t device);

/**
 * Set clocks that applications will lock to.
 *
 * Sets the clocks that compute and graphics applications will be running at.
 * e.g. LWCA driver requests these clocks during context creation which means this property
 * defines clocks at which LWCA applications will be running unless some overspec event
 * oclwrs (e.g. over power, over thermal or external HW brake).
 *
 * Can be used as a setting to request constant performance.
 *
 * On Pascal and newer hardware, this will automatically disable automatic boosting of clocks.
 *
 * On K80 and newer Kepler and Maxwell GPUs, users desiring fixed performance should also call
 * \ref lwmlDeviceSetAutoBoostedClocksEnabled to prevent clocks from automatically boosting
 * above the clock value being set.
 *
 * For Kepler &tm; or newer non-VdChip fully supported devices and Maxwell or newer VdChip devices.
 * Requires root/admin permissions.
 *
 * See \ref lwmlDeviceGetSupportedMemoryClocks and \ref lwmlDeviceGetSupportedGraphicsClocks
 * for details on how to list available clocks combinations.
 *
 * After system reboot or driver reload applications clocks go back to their default value.
 * See \ref lwmlDeviceResetApplicationsClocks.
 *
 * @param device                               The identifier of the target device
 * @param memClockMHz                          Requested memory clock in MHz
 * @param graphicsClockMHz                     Requested graphics clock in MHz
 *
 * @return
 *         - \ref LWML_SUCCESS                 if new settings were successfully set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a memClockMHz and \a graphicsClockMHz
 *                                                 is not a valid clock combination
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceSetApplicationsClocks(lwmlDevice_t device, unsigned int memClockMHz, unsigned int graphicsClockMHz);

/**
 * Set new power limit of this device.
 * 
 * For Kepler &tm; or newer fully supported devices.
 * Requires root/admin permissions.
 *
 * See \ref lwmlDeviceGetPowerManagementLimitConstraints to check the allowed ranges of values.
 *
 * \note Limit is not persistent across reboots or driver unloads.
 * Enable persistent mode to prevent driver from unloading when no application is using the device.
 *
 * @param device                               The identifier of the target device
 * @param limit                                Power management limit in milliwatts to set
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a limit has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a defaultLimit is out of range
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlDeviceGetPowerManagementLimitConstraints
 * @see lwmlDeviceGetPowerManagementDefaultLimit
 */
lwmlReturn_t DECLDIR lwmlDeviceSetPowerManagementLimit(lwmlDevice_t device, unsigned int limit);

/**
 * Sets new GOM. See \a lwmlGpuOperationMode_t for details.
 *
 * For GK110 M-class and X-class Tesla &tm; products from the Kepler family.
 * Modes \ref LWML_GOM_LOW_DP and \ref LWML_GOM_ALL_ON are supported on fully supported VdChip products.
 * Not supported on Lwdqro &reg; and Tesla &tm; C-class products.
 * Requires root/admin permissions.
 * 
 * Changing GOMs requires a reboot. 
 * The reboot requirement might be removed in the future.
 *
 * Compute only GOMs don't support graphics acceleration. Under windows switching to these GOMs when
 * pending driver model is WDDM is not supported. See \ref lwmlDeviceSetDriverModel.
 * 
 * @param device                               The identifier of the target device
 * @param mode                                 Target GOM
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a mode has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a mode incorrect
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support GOM or specific mode
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlGpuOperationMode_t
 * @see lwmlDeviceGetGpuOperationMode
 */
lwmlReturn_t DECLDIR lwmlDeviceSetGpuOperationMode(lwmlDevice_t device, lwmlGpuOperationMode_t mode);

/**
 * Changes the root/admin restructions on certain APIs. See \a lwmlRestrictedAPI_t for the list of supported APIs.
 * This method can be used by a root/admin user to give non-root/admin access to certain otherwise-restricted APIs.
 * The new setting lasts for the lifetime of the LWPU driver; it is not persistent. See \a lwmlDeviceGetAPIRestriction
 * to query the current restriction settings.
 * 
 * For Kepler &tm; or newer fully supported devices.
 * Requires root/admin permissions.
 *
 * @param device                               The identifier of the target device
 * @param apiType                              Target API type for this operation
 * @param isRestricted                         The target restriction
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a isRestricted has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a apiType incorrect
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support changing API restrictions or the device does not support
 *                                                 the feature that api restrictions are being set for (E.G. Enabling/disabling auto 
 *                                                 boosted clocks is not supported by the device)
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlRestrictedAPI_t
 */
lwmlReturn_t DECLDIR lwmlDeviceSetAPIRestriction(lwmlDevice_t device, lwmlRestrictedAPI_t apiType, lwmlEnableState_t isRestricted);

/**
 * @}
 */
 
/** @addtogroup lwmlAccountingStats
 *  @{
 */

/**
 * Enables or disables per process accounting.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Requires root/admin permissions.
 *
 * @note This setting is not persistent and will default to disabled after driver unloads.
 *       Enable persistence mode to be sure the setting doesn't switch off to disabled.
 * 
 * @note Enabling accounting mode has no negative impact on the GPU performance.
 *
 * @note Disabling accounting clears all accounting pids information.
 *
 * @note On MIG-enabled GPUs, accounting mode would be set to DISABLED and changing it is not supported.
 *
 * See \ref lwmlDeviceGetAccountingMode
 * See \ref lwmlDeviceGetAccountingStats
 * See \ref lwmlDeviceClearAccountingPids
 *
 * @param device                               The identifier of the target device
 * @param mode                                 The target accounting mode
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if the new mode has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device or \a mode are invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceSetAccountingMode(lwmlDevice_t device, lwmlEnableState_t mode);

/**
 * Clears accounting information about all processes that have already terminated.
 *
 * For Kepler &tm; or newer fully supported devices.
 * Requires root/admin permissions.
 *
 * See \ref lwmlDeviceGetAccountingMode
 * See \ref lwmlDeviceGetAccountingStats
 * See \ref lwmlDeviceSetAccountingMode
 *
 * @param device                               The identifier of the target device
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if accounting information has been cleared 
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device are invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceClearAccountingPids(lwmlDevice_t device);

/** @} */

/***************************************************************************************************/
/** @defgroup LwLink LwLink Methods
 * This chapter describes methods that LWML can perform on LWLINK enabled devices.
 *  @{
 */
/***************************************************************************************************/

/**
 * Retrieves the state of the device's LwLink for the link specified
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the LwLink link to be queried
 * @param isActive                             \a lwmlEnableState_t where LWML_FEATURE_ENABLED indicates that
 *                                             the link is active and LWML_FEATURE_DISABLED indicates it 
 *                                             is inactive
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a isActive has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device or \a link is invalid or \a isActive is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetLwLinkState(lwmlDevice_t device, unsigned int link, lwmlEnableState_t *isActive);

/**
 * Retrieves the version of the device's LwLink for the link specified
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the LwLink link to be queried
 * @param version                              Requested LwLink version
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a version has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device or \a link is invalid or \a version is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetLwLinkVersion(lwmlDevice_t device, unsigned int link, unsigned int *version);

/**
 * Retrieves the requested capability from the device's LwLink for the link specified
 * Please refer to the \a lwmlLwLinkCapability_t structure for the specific caps that can be queried
 * The return value should be treated as a boolean.
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the LwLink link to be queried
 * @param capability                           Specifies the \a lwmlLwLinkCapability_t to be queried
 * @param capResult                            A boolean for the queried capability indicating that feature is available
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a capResult has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device, \a link, or \a capability is invalid or \a capResult is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetLwLinkCapability(lwmlDevice_t device, unsigned int link,
                                                   lwmlLwLinkCapability_t capability, unsigned int *capResult); 

/**
 * Retrieves the PCI information for the remote node on a LwLink link 
 * Note: pciSubSystemId is not filled in this function and is indeterminate
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the LwLink link to be queried
 * @param pci                                  \a lwmlPciInfo_t of the remote node for the specified link                            
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a pci has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device or \a link is invalid or \a pci is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetLwLinkRemotePciInfo_v2(lwmlDevice_t device, unsigned int link, lwmlPciInfo_t *pci);

/**
 * Retrieves the specified error counter value
 * Please refer to \a lwmlLwLinkErrorCounter_t for error counters that are available
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the LwLink link to be queried
 * @param counter                              Specifies the LwLink counter to be queried
 * @param counterValue                         Returned counter value
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a counter has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device, \a link, or \a counter is invalid or \a counterValue is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetLwLinkErrorCounter(lwmlDevice_t device, unsigned int link,
                                                     lwmlLwLinkErrorCounter_t counter, unsigned long long *counterValue);

/**
 * Resets all error counters to zero
 * Please refer to \a lwmlLwLinkErrorCounter_t for the list of error counters that are reset
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the LwLink link to be queried
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if the reset is successful
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device or \a link is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceResetLwLinkErrorCounters(lwmlDevice_t device, unsigned int link);

/**
 * Deprecated: Setting utilization counter control is no longer supported.
 *
 * Set the LWLINK utilization counter control information for the specified counter, 0 or 1.
 * Please refer to \a lwmlLwLinkUtilizationControl_t for the structure definition.  Performs a reset
 * of the counters if the reset parameter is non-zero.
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param counter                              Specifies the counter that should be set (0 or 1).
 * @param link                                 Specifies the LwLink link to be queried
 * @param control                              A reference to the \a lwmlLwLinkUtilizationControl_t to set
 * @param reset                                Resets the counters on set if non-zero
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if the control has been set successfully
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device, \a counter, \a link, or \a control is invalid 
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceSetLwLinkUtilizationControl(lwmlDevice_t device, unsigned int link, unsigned int counter,
                                                           lwmlLwLinkUtilizationControl_t *control, unsigned int reset);

/**
 * Deprecated: Getting utilization counter control is no longer supported.
 *
 * Get the LWLINK utilization counter control information for the specified counter, 0 or 1.
 * Please refer to \a lwmlLwLinkUtilizationControl_t for the structure definition
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param counter                              Specifies the counter that should be set (0 or 1).
 * @param link                                 Specifies the LwLink link to be queried
 * @param control                              A reference to the \a lwmlLwLinkUtilizationControl_t to place information
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if the control has been set successfully
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device, \a counter, \a link, or \a control is invalid 
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetLwLinkUtilizationControl(lwmlDevice_t device, unsigned int link, unsigned int counter,
                                                           lwmlLwLinkUtilizationControl_t *control);


/**
 * Deprecated: Use \ref lwmlDeviceGetFieldValues with LWML_FI_DEV_LWLINK_THROUGHPUT_* as field values instead.
 *
 * Retrieve the LWLINK utilization counter based on the current control for a specified counter.
 * In general it is good practice to use \a lwmlDeviceSetLwLinkUtilizationControl
 *  before reading the utilization counters as they have no default state
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the LwLink link to be queried
 * @param counter                              Specifies the counter that should be read (0 or 1).
 * @param rxcounter                            Receive counter return value
 * @param txcounter                            Transmit counter return value
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if \a rxcounter and \a txcounter have been successfully set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device, \a counter, or \a link is invalid or \a rxcounter or \a txcounter are NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetLwLinkUtilizationCounter(lwmlDevice_t device, unsigned int link, unsigned int counter, 
                                                           unsigned long long *rxcounter, unsigned long long *txcounter);

/**
 * Deprecated: Freezing LWLINK utilization counters is no longer supported.
 *
 * Freeze the LWLINK utilization counters 
 * Both the receive and transmit counters are operated on by this function
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the LwLink link to be queried
 * @param counter                              Specifies the counter that should be frozen (0 or 1).
 * @param freeze                               LWML_FEATURE_ENABLED = freeze the receive and transmit counters
 *                                             LWML_FEATURE_DISABLED = unfreeze the receive and transmit counters
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if counters were successfully frozen or unfrozen
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device, \a link, \a counter, or \a freeze is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceFreezeLwLinkUtilizationCounter (lwmlDevice_t device, unsigned int link, 
                                            unsigned int counter, lwmlEnableState_t freeze);

/**
 * Deprecated: Resetting LWLINK utilization counters is no longer supported.
 *
 * Reset the LWLINK utilization counters 
 * Both the receive and transmit counters are operated on by this function
 *
 * For Pascal &tm; or newer fully supported devices.
 *
 * @param device                               The identifier of the target device
 * @param link                                 Specifies the LwLink link to be reset
 * @param counter                              Specifies the counter that should be reset (0 or 1)
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if counters were successfully reset
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device, \a link, or \a counter is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceResetLwLinkUtilizationCounter (lwmlDevice_t device, unsigned int link, unsigned int counter);

/**
* Get the LWLink device type of the remote device connected over the given link.
*
* @param device                                The device handle of the target GPU
* @param link                                  The LWLink link index on the target GPU
* @param pLwLinkDeviceType                     Pointer in which the output remote device type is returned
*
* @return
*         - \ref LWML_SUCCESS                  if \a pLwLinkDeviceType has been set
*         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref LWML_ERROR_NOT_SUPPORTED      if LWLink is not supported
*         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a device or \a link is invalid, or
*                                              \a pLwLinkDeviceType is NULL
*         - \ref LWML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is
*                                              otherwise inaccessible
*         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
*/
lwmlReturn_t DECLDIR lwmlDeviceGetLwLinkRemoteDeviceType(lwmlDevice_t device, unsigned int link, lwmlIntLwLinkDeviceType_t *pLwLinkDeviceType);

/** @} */

/***************************************************************************************************/
/** @defgroup lwmlEvents Event Handling Methods
 * This chapter describes methods that LWML can perform against each device to register and wait for 
 * some event to occur.
 *  @{
 */
/***************************************************************************************************/

/**
 * Create an empty set of events.
 * Event set should be freed by \ref lwmlEventSetFree
 *
 * For Fermi &tm; or newer fully supported devices.
 * @param set                                  Reference in which to return the event handle
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if the event has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a set is NULL
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 * 
 * @see lwmlEventSetFree
 */
lwmlReturn_t DECLDIR lwmlEventSetCreate(lwmlEventSet_t *set);

/**
 * Starts recording of events on a specified devices and add the events to specified \ref lwmlEventSet_t
 *
 * For Fermi &tm; or newer fully supported devices.
 * Ecc events are available only on ECC enabled devices (see \ref lwmlDeviceGetTotalEccErrors)
 * Power capping events are available only on Power Management enabled devices (see \ref lwmlDeviceGetPowerManagementMode)
 *
 * For Linux only.
 *
 * \b IMPORTANT: Operations on \a set are not thread safe
 *
 * This call starts recording of events on specific device.
 * All events that oclwrred before this call are not recorded.
 * Checking if some event oclwrred can be done with \ref lwmlEventSetWait_v2
 *
 * If function reports LWML_ERROR_UNKNOWN, event set is in undefined state and should be freed.
 * If function reports LWML_ERROR_NOT_SUPPORTED, event set can still be used. None of the requested eventTypes
 *     are registered in that case.
 *
 * @param device                               The identifier of the target device
 * @param eventTypes                           Bitmask of \ref lwmlEventType to record
 * @param set                                  Set to which add new event types
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if the event has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a eventTypes is invalid or \a set is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the platform does not support this feature or some of requested event types
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 * 
 * @see lwmlEventType
 * @see lwmlDeviceGetSupportedEventTypes
 * @see lwmlEventSetWait
 * @see lwmlEventSetFree
 */
lwmlReturn_t DECLDIR lwmlDeviceRegisterEvents(lwmlDevice_t device, unsigned long long eventTypes, lwmlEventSet_t set);

/**
 * Returns information about events supported on device
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * Events are not supported on Windows. So this function returns an empty mask in \a eventTypes on Windows.
 *
 * @param device                               The identifier of the target device
 * @param eventTypes                           Reference in which to return bitmask of supported events
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if the eventTypes has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a eventType is NULL
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 * 
 * @see lwmlEventType
 * @see lwmlDeviceRegisterEvents
 */
lwmlReturn_t DECLDIR lwmlDeviceGetSupportedEventTypes(lwmlDevice_t device, unsigned long long *eventTypes);

/**
 * Waits on events and delivers events
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * If some events are ready to be delivered at the time of the call, function returns immediately.
 * If there are no events ready to be delivered, function sleeps till event arrives 
 * but not longer than specified timeout. This function in certain conditions can return before
 * specified timeout passes (e.g. when interrupt arrives)
 *
 * On Windows, in case of xid error, the function returns the most recent xid error type seen by the system.
 * If there are multiple xid errors generated before lwmlEventSetWait is ilwoked then the last seen xid error
 * type is returned for all xid error events.
 *
 * On Linux, every xid error event would return the associated event data and other information if applicable.
 *
 * In MIG mode, if device handle is provided, the API reports all the events for the available instances,
 * only if the caller has appropriate privileges. In absence of required privileges, only the events which
 * affect all the instances (i.e. whole device) are reported.
 *
 * This API does not lwrrently support per-instance event reporting using MIG device handles.
 *
 * @param set                                  Reference to set of events to wait on
 * @param data                                 Reference in which to return event data
 * @param timeoutms                            Maximum amount of wait time in milliseconds for registered event
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if the data has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a data is NULL
 *         - \ref LWML_ERROR_TIMEOUT           if no event arrived in specified timeout or interrupt arrived
 *         - \ref LWML_ERROR_GPU_IS_LOST       if a GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 * 
 * @see lwmlEventType
 * @see lwmlDeviceRegisterEvents
 */
lwmlReturn_t DECLDIR lwmlEventSetWait_v2(lwmlEventSet_t set, lwmlEventData_t * data, unsigned int timeoutms);

/**
 * Releases events in the set
 *
 * For Fermi &tm; or newer fully supported devices.
 *
 * @param set                                  Reference to events to be released 
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if the event has been successfully released
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 * 
 * @see lwmlDeviceRegisterEvents
 */
lwmlReturn_t DECLDIR lwmlEventSetFree(lwmlEventSet_t set);

/** @} */

/***************************************************************************************************/
/** @defgroup lwmlZPI Drain states 
 * This chapter describes methods that LWML can perform against each device to control their drain state
 * and recognition by LWML and LWPU kernel driver. These methods can be used with out-of-band tools to
 * power on/off GPUs, enable robust reset scenarios, etc.
 *  @{
 */
/***************************************************************************************************/

/**
 * Modify the drain state of a GPU.  This method forces a GPU to no longer accept new incoming requests.
 * Any new LWML process will no longer see this GPU.  Persistence mode for this GPU must be turned off before
 * this call is made.
 * Must be called as administrator.
 * For Linux only.
 * 
 * For Pascal &tm; or newer fully supported devices.
 * Some Kepler devices supported.
 *
 * @param pciInfo                              The PCI address of the GPU drain state to be modified
 * @param newState                             The drain state that should be entered, see \ref lwmlEnableState_t
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if counters were successfully reset
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a lwmlIndex or \a newState is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_NO_PERMISSION     if the calling process has insufficient permissions to perform operation
 *         - \ref LWML_ERROR_IN_USE            if the device has persistence mode turned on
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceModifyDrainState (lwmlPciInfo_t *pciInfo, lwmlEnableState_t newState);

/**
 * Query the drain state of a GPU.  This method is used to check if a GPU is in a lwrrently draining
 * state.
 * For Linux only.
 * 
 * For Pascal &tm; or newer fully supported devices.
 * Some Kepler devices supported.
 *
 * @param pciInfo                              The PCI address of the GPU drain state to be queried
 * @param lwrrentState                         The current drain state for this GPU, see \ref lwmlEnableState_t
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if counters were successfully reset
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a lwmlIndex or \a lwrrentState is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceQueryDrainState (lwmlPciInfo_t *pciInfo, lwmlEnableState_t *lwrrentState);

/**
 * This method will remove the specified GPU from the view of both LWML and the LWPU kernel driver
 * as long as no other processes are attached. If other processes are attached, this call will return
 * LWML_ERROR_IN_USE and the GPU will be returned to its original "draining" state. Note: the
 * only situation where a process can still be attached after lwmlDeviceModifyDrainState() is called
 * to initiate the draining state is if that process was using, and is still using, a GPU before the 
 * call was made. Also note, persistence mode counts as an attachment to the GPU thus it must be disabled
 * prior to this call.
 *
 * For long-running LWML processes please note that this will change the enumeration of current GPUs.
 * For example, if there are four GPUs present and GPU1 is removed, the new enumeration will be 0-2.
 * Also, device handles after the removed GPU will not be valid and must be re-established.
 * Must be run as administrator. 
 * For Linux only.
 *
 * For Pascal &tm; or newer fully supported devices.
 * Some Kepler devices supported.
 *
 * @param pciInfo                              The PCI address of the GPU to be removed
 * @param gpuState                             Whether the GPU is to be removed, from the OS
 *                                             see \ref lwmlDetachGpuState_t
 * @param linkState                            Requested upstream PCIe link state, see \ref lwmlPcieLinkState_t
 *
 * @return
 *         - \ref LWML_SUCCESS                 if counters were successfully reset
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a lwmlIndex is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_IN_USE            if the device is still in use and cannot be removed
 */
lwmlReturn_t DECLDIR lwmlDeviceRemoveGpu_v2(lwmlPciInfo_t *pciInfo, lwmlDetachGpuState_t gpuState, lwmlPcieLinkState_t linkState);

/**
 * Request the OS and the LWPU kernel driver to rediscover a portion of the PCI subsystem looking for GPUs that
 * were previously removed. The portion of the PCI tree can be narrowed by specifying a domain, bus, and device.  
 * If all are zeroes then the entire PCI tree will be searched.  Please note that for long-running LWML processes
 * the enumeration will change based on how many GPUs are discovered and where they are inserted in bus order.
 *
 * In addition, all newly discovered GPUs will be initialized and their ECC scrubbed which may take several seconds
 * per GPU. Also, all device handles are no longer guaranteed to be valid post discovery.
 *
 * Must be run as administrator.
 * For Linux only.
 * 
 * For Pascal &tm; or newer fully supported devices.
 * Some Kepler devices supported.
 *
 * @param pciInfo                              The PCI tree to be searched.  Only the domain, bus, and device
 *                                             fields are used in this call.
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if counters were successfully reset
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a pciInfo is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the operating system does not support this feature
 *         - \ref LWML_ERROR_OPERATING_SYSTEM  if the operating system is denying this feature
 *         - \ref LWML_ERROR_NO_PERMISSION     if the calling process has insufficient permissions to perform operation
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceDiscoverGpus (lwmlPciInfo_t *pciInfo);

/** @} */

/***************************************************************************************************/
/** @defgroup lwmlFieldValueQueries Field Value Queries
 *  This chapter describes LWML operations that are associated with retrieving Field Values from LWML
 *  @{
 */
/***************************************************************************************************/

/**
 * Request values for a list of fields for a device. This API allows multiple fields to be queried at once.
 * If any of the underlying fieldIds are populated by the same driver call, the results for those field IDs
 * will be populated from a single call rather than making a driver call for each fieldId.
 *
 * @param device                               The device handle of the GPU to request field values for
 * @param valuesCount                          Number of entries in values that should be retrieved
 * @param values                               Array of \a valuesCount structures to hold field values.
 *                                             Each value's fieldId must be populated prior to this call
 *
 * @return
 *         - \ref LWML_SUCCESS                 if any values in \a values were populated. Note that you must
 *                                             check the lwmlReturn field of each value for each individual
 *                                             status
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a values is NULL
 */
lwmlReturn_t DECLDIR lwmlDeviceGetFieldValues(lwmlDevice_t device, int valuesCount, lwmlFieldValue_t *values);


/** @} */

/***************************************************************************************************/
/** @defgroup vGPU Enums, Constants and Structs
 *  @{
 */
/** @} */
/***************************************************************************************************/

/***************************************************************************************************/
/** @defgroup lwmlVirtualGpuQueries vGPU APIs
 * This chapter describes operations that are associated with LWPU vGPU Software products.
 *  @{
 */
/***************************************************************************************************/

/**
 * This method is used to get the virtualization mode corresponding to the GPU.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                    Identifier of the target device
 * @param pVirtualMode              Reference to virtualization mode. One of LWML_GPU_VIRTUALIZATION_?
 * 
 * @return 
 *         - \ref LWML_SUCCESS                  if \a pVirtualMode is fetched
 *         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a device is invalid or \a pVirtualMode is NULL
 *         - \ref LWML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetVirtualizationMode(lwmlDevice_t device, lwmlGpuVirtualizationMode_t *pVirtualMode);

/**
 * Queries if SR-IOV host operation is supported on a vGPU supported device.
 *
 * Checks whether SR-IOV host capability is supported by the device and the
 * driver, and indicates device is in SR-IOV mode if both of these conditions
 * are true.
 *
 * @param device                                The identifier of the target device
 * @param pHostVgpuMode                         Reference in which to return the current vGPU mode
 *
 * @return
 *         - \ref LWML_SUCCESS                  if device's vGPU mode has been successfully retrieved
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a device handle is 0 or \a pVgpuMode is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED      if \a device doesn't support this feature.
 *         - \ref LWML_ERROR_UNKNOWN            if any unexpected error oclwrred
 */
lwmlReturn_t DECLDIR lwmlDeviceGetHostVgpuMode(lwmlDevice_t device, lwmlHostVgpuMode_t *pHostVgpuMode);

/**
 * This method is used to set the virtualization mode corresponding to the GPU.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                    Identifier of the target device
 * @param virtualMode               virtualization mode. One of LWML_GPU_VIRTUALIZATION_?
 *
 * @return 
 *         - \ref LWML_SUCCESS                  if \a pVirtualMode is set
 *         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a device is invalid or \a pVirtualMode is NULL
 *         - \ref LWML_ERROR_GPU_IS_LOST        if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_NOT_SUPPORTED      if setting of virtualization mode is not supported.
 *         - \ref LWML_ERROR_NO_PERMISSION      if setting of virtualization mode is not allowed for this client.
 */
lwmlReturn_t DECLDIR lwmlDeviceSetVirtualizationMode(lwmlDevice_t device, lwmlGpuVirtualizationMode_t virtualMode);

/**
 * Retrieve the vGPU Software licensable features.
 *
 * Identifies whether the system supports vGPU Software Licensing. If it does, return the list of licensable feature(s)
 * and their current license status.
 *
 * @param device                    Identifier of the target device
 * @param pGridLicensableFeatures   Pointer to structure in which vGPU software licensable features are returned
 *
 * @return
 *         - \ref LWML_SUCCESS                 if licensable features are successfully retrieved
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a pGridLicensableFeatures is NULL
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetGridLicensableFeatures_v3(lwmlDevice_t device, lwmlGridLicensableFeatures_t *pGridLicensableFeatures);

/**
 * Retrieves the current utilization and process ID
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * Reads recent utilization of GPU SM (3D/Compute), framebuffer, video encoder, and video decoder for processes running.
 * Utilization values are returned as an array of utilization sample structures in the caller-supplied buffer pointed at
 * by \a utilization. One utilization sample structure is returned per process running, that had some non-zero utilization
 * during the last sample period. It includes the CPU timestamp at which  the samples were recorded. Individual utilization values
 * are returned as "unsigned int" values.
 *
 * To read utilization values, first determine the size of buffer required to hold the samples by ilwoking the function with
 * \a utilization set to NULL. The caller should allocate a buffer of size
 * processSamplesCount * sizeof(lwmlProcessUtilizationSample_t). Ilwoke the function again with the allocated buffer passed
 * in \a utilization, and \a processSamplesCount set to the number of entries the buffer is sized for.
 *
 * On successful return, the function updates \a processSamplesCount with the number of process utilization sample
 * structures that were actually written. This may differ from a previously read value as instances are created or
 * destroyed.
 *
 * lastSeenTimeStamp represents the CPU timestamp in microseconds at which utilization samples were last read. Set it to 0
 * to read utilization based on all the samples maintained by the driver's internal sample buffer. Set lastSeenTimeStamp
 * to a timeStamp retrieved from a previous query to read utilization since the previous query.
 *
 * @note On MIG-enabled GPUs, querying process utilization is not lwrrently supported.
 *
 * @param device                    The identifier of the target device
 * @param utilization               Pointer to caller-supplied buffer in which guest process utilization samples are returned
 * @param processSamplesCount       Pointer to caller-supplied array size, and returns number of processes running
 * @param lastSeenTimeStamp         Return only samples with timestamp greater than lastSeenTimeStamp.

 * @return
 *         - \ref LWML_SUCCESS                 if \a utilization has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, \a utilization is NULL, or \a samplingPeriodUs is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support this feature
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetProcessUtilization(lwmlDevice_t device, lwmlProcessUtilizationSample_t *utilization,
                                              unsigned int *processSamplesCount, unsigned long long lastSeenTimeStamp);

/** @} */

/***************************************************************************************************/
/** @defgroup lwmlVgpu vGPU Management
 * @{
 *
 * This chapter describes APIs supporting LWPU vGPU.
 */
/***************************************************************************************************/

/**
 * Retrieve the supported vGPU types on a physical GPU (device).
 *
 * An array of supported vGPU types for the physical GPU indicated by \a device is returned in the caller-supplied buffer
 * pointed at by \a vgpuTypeIds. The element count of lwmlVgpuTypeId_t array is passed in \a vgpuCount, and \a vgpuCount
 * is used to return the number of vGPU types written to the buffer.
 *
 * If the supplied buffer is not large enough to accomodate the vGPU type array, the function returns
 * LWML_ERROR_INSUFFICIENT_SIZE, with the element count of lwmlVgpuTypeId_t array required in \a vgpuCount.
 * To query the number of vGPU types supported for the GPU, call this function with *vgpuCount = 0.
 * The code will return LWML_ERROR_INSUFFICIENT_SIZE, or LWML_SUCCESS if no vGPU types are supported.
 *
 * @param device                   The identifier of the target device
 * @param vgpuCount                Pointer to caller-supplied array size, and returns number of vGPU types
 * @param vgpuTypeIds              Pointer to caller-supplied array in which to return list of vGPU types
 *
 * @return
 *         - \ref LWML_SUCCESS                      successful completion
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE      \a vgpuTypeIds buffer is too small, array element count is returned in \a vgpuCount
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT       if \a vgpuCount is NULL or \a device is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED          if vGPU is not supported by the device
 *         - \ref LWML_ERROR_UNKNOWN                on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetSupportedVgpus(lwmlDevice_t device, unsigned int *vgpuCount, lwmlVgpuTypeId_t *vgpuTypeIds);

/**
 * Retrieve the lwrrently creatable vGPU types on a physical GPU (device).
 *
 * An array of creatable vGPU types for the physical GPU indicated by \a device is returned in the caller-supplied buffer
 * pointed at by \a vgpuTypeIds. The element count of lwmlVgpuTypeId_t array is passed in \a vgpuCount, and \a vgpuCount
 * is used to return the number of vGPU types written to the buffer.
 *
 * The creatable vGPU types for a device may differ over time, as there may be restrictions on what type of vGPU types
 * can conlwrrently run on a device.  For example, if only one vGPU type is allowed at a time on a device, then the creatable
 * list will be restricted to whatever vGPU type is already running on the device.
 *
 * If the supplied buffer is not large enough to accomodate the vGPU type array, the function returns
 * LWML_ERROR_INSUFFICIENT_SIZE, with the element count of lwmlVgpuTypeId_t array required in \a vgpuCount.
 * To query the number of vGPU types createable for the GPU, call this function with *vgpuCount = 0.
 * The code will return LWML_ERROR_INSUFFICIENT_SIZE, or LWML_SUCCESS if no vGPU types are creatable.
 *
 * @param device                   The identifier of the target device
 * @param vgpuCount                Pointer to caller-supplied array size, and returns number of vGPU types
 * @param vgpuTypeIds              Pointer to caller-supplied array in which to return list of vGPU types
 *
 * @return
 *         - \ref LWML_SUCCESS                      successful completion
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE      \a vgpuTypeIds buffer is too small, array element count is returned in \a vgpuCount
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT       if \a vgpuCount is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED          if vGPU is not supported by the device
 *         - \ref LWML_ERROR_UNKNOWN                on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetCreatableVgpus(lwmlDevice_t device, unsigned int *vgpuCount, lwmlVgpuTypeId_t *vgpuTypeIds);

/**
 * Retrieve the class of a vGPU type. It will not exceed 64 characters in length (including the NUL terminator).
 * See \ref lwmlConstants::LWML_DEVICE_NAME_BUFFER_SIZE.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param vgpuTypeClass            Pointer to string array to return class in
 * @param size                     Size of string
 *
 * @return
 *         - \ref LWML_SUCCESS                   successful completion
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT    if \a vgpuTypeId is invalid, or \a vgpuTypeClass is NULL
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE   if \a size is too small
 *         - \ref LWML_ERROR_UNKNOWN             on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuTypeGetClass(lwmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeClass, unsigned int *size);

/**
 * Retrieve the vGPU type name.
 *
 * The name is an alphanumeric string that denotes a particular vGPU, e.g. GRID M60-2Q. It will not
 * exceed 64 characters in length (including the NUL terminator).  See \ref
 * lwmlConstants::LWML_DEVICE_NAME_BUFFER_SIZE.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param vgpuTypeName             Pointer to buffer to return name
 * @param size                     Size of buffer
 *
 * @return
 *         - \ref LWML_SUCCESS                 successful completion
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a name is NULL
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a size is too small
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuTypeGetName(lwmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeName, unsigned int *size);

/**
 * Retrieve the GPU Instance Profile ID for the given vGPU type ID.
 * The API will return a valid GPU Instance Profile ID for the MIG capable vGPU types, else ILWALID_GPU_INSTANCE_PROFILE_ID is
 * returned.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param gpuInstanceProfileId     GPU Instance Profile ID
 *
 * @return
 *         - \ref LWML_SUCCESS                 successful completion
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if \a device is not in vGPU Host virtualization mode
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a gpuInstanceProfileId is NULL
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuTypeGetGpuInstanceProfileId(lwmlVgpuTypeId_t vgpuTypeId, unsigned int *gpuInstanceProfileId);

/**
 * Retrieve the device ID of a vGPU type.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param deviceID                 Device ID and vendor ID of the device contained in single 32 bit value
 * @param subsystemID              Subsytem ID and subsytem vendor ID of the device contained in single 32 bit value
 *
 * @return
 *         - \ref LWML_SUCCESS                 successful completion
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a deviceId or \a subsystemID are NULL
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuTypeGetDeviceID(lwmlVgpuTypeId_t vgpuTypeId, unsigned long long *deviceID, unsigned long long *subsystemID);

/**
 * Retrieve the vGPU framebuffer size in bytes.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param fbSize                   Pointer to framebuffer size in bytes
 *
 * @return
 *         - \ref LWML_SUCCESS                 successful completion
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a fbSize is NULL
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuTypeGetFramebufferSize(lwmlVgpuTypeId_t vgpuTypeId, unsigned long long *fbSize);

/**
 * Retrieve count of vGPU's supported display heads.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param numDisplayHeads          Pointer to number of display heads
 *
 * @return
 *         - \ref LWML_SUCCESS                 successful completion
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a numDisplayHeads is NULL
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuTypeGetNumDisplayHeads(lwmlVgpuTypeId_t vgpuTypeId, unsigned int *numDisplayHeads);

/**
 * Retrieve vGPU display head's maximum supported resolution.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param displayIndex             Zero-based index of display head
 * @param xdim                     Pointer to maximum number of pixels in X dimension
 * @param ydim                     Pointer to maximum number of pixels in Y dimension
 *
 * @return
 *         - \ref LWML_SUCCESS                 successful completion
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a xdim or \a ydim are NULL, or \a displayIndex
 *                                             is out of range.
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuTypeGetResolution(lwmlVgpuTypeId_t vgpuTypeId, unsigned int displayIndex, unsigned int *xdim, unsigned int *ydim);

/**
 * Retrieve license requirements for a vGPU type
 *
 * The license type and version required to run the specified vGPU type is returned as an alphanumeric string, in the form
 * "<license name>,<version>", for example "GRID-Virtual-PC,2.0". If a vGPU is runnable with* more than one type of license,
 * the licenses are delimited by a semicolon, for example "GRID-Virtual-PC,2.0;GRID-Virtual-WS,2.0;GRID-Virtual-WS-Ext,2.0".
 *
 * The total length of the returned string will not exceed 128 characters, including the NUL terminator.
 * See \ref lwmlVgpuConstants::LWML_GRID_LICENSE_BUFFER_SIZE.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param vgpuTypeLicenseString    Pointer to buffer to return license info
 * @param size                     Size of \a vgpuTypeLicenseString buffer
 *
 * @return
 *         - \ref LWML_SUCCESS                 successful completion
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a vgpuTypeLicenseString is NULL
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a size is too small
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuTypeGetLicense(lwmlVgpuTypeId_t vgpuTypeId, char *vgpuTypeLicenseString, unsigned int size);

/**
 * Retrieve the static frame rate limit value of the vGPU type
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param frameRateLimit           Reference to return the frame rate limit value
 * @return
 *         - \ref LWML_SUCCESS                 successful completion
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if frame rate limiter is turned off for the vGPU type
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a frameRateLimit is NULL
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuTypeGetFrameRateLimit(lwmlVgpuTypeId_t vgpuTypeId, unsigned int *frameRateLimit);

/**
 * Retrieve the maximum number of vGPU instances creatable on a device for given vGPU type
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                   The identifier of the target device
 * @param vgpuTypeId               Handle to vGPU type
 * @param vgpuInstanceCount        Pointer to get the max number of vGPU instances
 *                                 that can be created on a deicve for given vgpuTypeId
 * @return
 *         - \ref LWML_SUCCESS                 successful completion
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuTypeId is invalid or is not supported on target device,
 *                                             or \a vgpuInstanceCount is NULL
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuTypeGetMaxInstances(lwmlDevice_t device, lwmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCount);

/**
 * Retrieve the maximum number of vGPU instances supported per VM for given vGPU type
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuTypeId               Handle to vGPU type
 * @param vgpuInstanceCountPerVm   Pointer to get the max number of vGPU instances supported per VM for given \a vgpuTypeId
 * @return
 *         - \ref LWML_SUCCESS                 successful completion
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuTypeId is invalid, or \a vgpuInstanceCountPerVm is NULL
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuTypeGetMaxInstancesPerVm(lwmlVgpuTypeId_t vgpuTypeId, unsigned int *vgpuInstanceCountPerVm);

/**
 * Retrieve the active vGPU instances on a device.
 *
 * An array of active vGPU instances is returned in the caller-supplied buffer pointed at by \a vgpuInstances. The
 * array elememt count is passed in \a vgpuCount, and \a vgpuCount is used to return the number of vGPU instances
 * written to the buffer.
 *
 * If the supplied buffer is not large enough to accomodate the vGPU instance array, the function returns
 * LWML_ERROR_INSUFFICIENT_SIZE, with the element count of lwmlVgpuInstance_t array required in \a vgpuCount.
 * To query the number of active vGPU instances, call this function with *vgpuCount = 0.  The code will return
 * LWML_ERROR_INSUFFICIENT_SIZE, or LWML_SUCCESS if no vGPU Types are supported.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param device                   The identifier of the target device
 * @param vgpuCount                Pointer which passes in the array size as well as get
 *                                 back the number of types
 * @param vgpuInstances            Pointer to array in which to return list of vGPU instances
 *
 * @return
 *         - \ref LWML_SUCCESS                  successful completion
 *         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a device is invalid, or \a vgpuCount is NULL
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE  if \a size is too small
 *         - \ref LWML_ERROR_NOT_SUPPORTED      if vGPU is not supported by the device
 *         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetActiveVgpus(lwmlDevice_t device, unsigned int *vgpuCount, lwmlVgpuInstance_t *vgpuInstances);

/**
 * Retrieve the VM ID associated with a vGPU instance.
 *
 * The VM ID is returned as a string, not exceeding 80 characters in length (including the NUL terminator).
 * See \ref lwmlConstants::LWML_DEVICE_UUID_BUFFER_SIZE.
 *
 * The format of the VM ID varies by platform, and is indicated by the type identifier returned in \a vmIdType.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param vmId                     Pointer to caller-supplied buffer to hold VM ID
 * @param size                     Size of buffer in bytes
 * @param vmIdType                 Pointer to hold VM ID type
 *
 * @return
 *         - \ref LWML_SUCCESS                 successful completion
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vmId or \a vmIdType is NULL, or \a vgpuInstance is 0
 *         - \ref LWML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a size is too small
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetVmID(lwmlVgpuInstance_t vgpuInstance, char *vmId, unsigned int size, lwmlVgpuVmIdType_t *vmIdType);

/**
 * Retrieve the UUID of a vGPU instance.
 *
 * The UUID is a globally unique identifier associated with the vGPU, and is returned as a 5-part hexadecimal string,
 * not exceeding 80 characters in length (including the NULL terminator).
 * See \ref lwmlConstants::LWML_DEVICE_UUID_BUFFER_SIZE.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param uuid                     Pointer to caller-supplied buffer to hold vGPU UUID
 * @param size                     Size of buffer in bytes
 *
 * @return
 *         - \ref LWML_SUCCESS                 successful completion
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuInstance is 0, or \a uuid is NULL
 *         - \ref LWML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a size is too small
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetUUID(lwmlVgpuInstance_t vgpuInstance, char *uuid, unsigned int size);

/**
 * Retrieve the LWPU driver version installed in the VM associated with a vGPU.
 *
 * The version is returned as an alphanumeric string in the caller-supplied buffer \a version. The length of the version
 * string will not exceed 80 characters in length (including the NUL terminator).
 * See \ref lwmlConstants::LWML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE.
 *
 * lwmlVgpuInstanceGetVmDriverVersion() may be called at any time for a vGPU instance. The guest VM driver version is
 * returned as "Not Available" if no LWPU driver is installed in the VM, or the VM has not yet booted to the point where the
 * LWPU driver is loaded and initialized.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param version                  Caller-supplied buffer to return driver version string
 * @param length                   Size of \a version buffer
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a version has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuInstance is 0
 *         - \ref LWML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a length is too small
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetVmDriverVersion(lwmlVgpuInstance_t vgpuInstance, char* version, unsigned int length);

/**
 * Retrieve the framebuffer usage in bytes.
 *
 * Framebuffer usage is the amont of vGPU framebuffer memory that is lwrrently in use by the VM.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             The identifier of the target instance
 * @param fbUsage                  Pointer to framebuffer usage in bytes
 *
 * @return
 *         - \ref LWML_SUCCESS                 successful completion
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuInstance is 0, or \a fbUsage is NULL
 *         - \ref LWML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetFbUsage(lwmlVgpuInstance_t vgpuInstance, unsigned long long *fbUsage);

/**
 * Retrieve the current licensing state of the vGPU instance.
 *
 * If the vGPU is lwrrently licensed, \a licensed is set to 1, otherwise it is set to 0.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param licensed                 Reference to return the licensing status
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a licensed has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuInstance is 0, or \a licensed is NULL
 *         - \ref LWML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetLicenseStatus(lwmlVgpuInstance_t vgpuInstance, unsigned int *licensed);

/**
 * Retrieve the vGPU type of a vGPU instance.
 *
 * Returns the vGPU type ID of vgpu assigned to the vGPU instance.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param vgpuTypeId               Reference to return the vgpuTypeId
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a vgpuTypeId has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuInstance is 0, or \a vgpuTypeId is NULL
 *         - \ref LWML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetType(lwmlVgpuInstance_t vgpuInstance, lwmlVgpuTypeId_t *vgpuTypeId);

/**
 * Retrieve the frame rate limit set for the vGPU instance.
 *
 * Returns the value of the frame rate limit set for the vGPU instance
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param frameRateLimit           Reference to return the frame rate limit
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a frameRateLimit has been set
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if frame rate limiter is turned off for the vGPU type
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuInstance is 0, or \a frameRateLimit is NULL
 *         - \ref LWML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetFrameRateLimit(lwmlVgpuInstance_t vgpuInstance, unsigned int *frameRateLimit);

/**
 * Retrieve the current ECC mode of vGPU instance.
 *
 * @param vgpuInstance            The identifier of the target vGPU instance
 * @param eccMode                 Reference in which to return the current ECC mode
 *
 * @return
 *         - \ref LWML_SUCCESS                 if the vgpuInstance's ECC mode has been successfully retrieved
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuInstance is 0, or \a mode is NULL
 *         - \ref LWML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the vGPU doesn't support this feature
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetEccMode(lwmlVgpuInstance_t vgpuInstance, lwmlEnableState_t *eccMode);

/**
 * Retrieve the encoder capacity of a vGPU instance, as a percentage of maximum encoder capacity with valid values in the range 0-100.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param encoderCapacity          Reference to an unsigned int for the encoder capacity
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a encoderCapacity has been retrived
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuInstance is 0, or \a encoderQueryType is invalid
 *         - \ref LWML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetEncoderCapacity(lwmlVgpuInstance_t vgpuInstance, unsigned int *encoderCapacity);

/**
 * Set the encoder capacity of a vGPU instance, as a percentage of maximum encoder capacity with valid values in the range 0-100.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param vgpuInstance             Identifier of the target vGPU instance
 * @param encoderCapacity          Unsigned int for the encoder capacity value
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a encoderCapacity has been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuInstance is 0, or \a encoderCapacity is out of range of 0-100.
 *         - \ref LWML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceSetEncoderCapacity(lwmlVgpuInstance_t vgpuInstance, unsigned int  encoderCapacity);

/**
 * Retrieves the current encoder statistics of a vGPU Instance
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param vgpuInstance                      Identifier of the target vGPU instance
 * @param sessionCount                      Reference to an unsigned int for count of active encoder sessions
 * @param averageFps                        Reference to an unsigned int for trailing average FPS of all active sessions
 * @param averageLatency                    Reference to an unsigned int for encode latency in microseconds
 *
 * @return
 *         - \ref LWML_SUCCESS                  if \a sessionCount, \a averageFps and \a averageLatency is fetched
 *         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a sessionCount , or \a averageFps or \a averageLatency is NULL
 *                                              or \a vgpuInstance is 0.
 *         - \ref LWML_ERROR_NOT_FOUND          if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetEncoderStats(lwmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount,
                                                     unsigned int *averageFps, unsigned int *averageLatency);

/**
 * Retrieves information about all active encoder sessions on a vGPU Instance.
 *
 * An array of active encoder sessions is returned in the caller-supplied buffer pointed at by \a sessionInfo. The
 * array element count is passed in \a sessionCount, and \a sessionCount is used to return the number of sessions
 * written to the buffer.
 *
 * If the supplied buffer is not large enough to accomodate the active session array, the function returns
 * LWML_ERROR_INSUFFICIENT_SIZE, with the element count of lwmlEncoderSessionInfo_t array required in \a sessionCount.
 * To query the number of active encoder sessions, call this function with *sessionCount = 0. The code will return
 * LWML_SUCCESS with number of active encoder sessions updated in *sessionCount.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param vgpuInstance                      Identifier of the target vGPU instance
 * @param sessionCount                      Reference to caller supplied array size, and returns
 *                                          the number of sessions.
 * @param sessionInfo                       Reference to caller supplied array in which the list
 *                                          of session information us returned.
 *
 * @return
 *         - \ref LWML_SUCCESS                  if \a sessionInfo is fetched
 *         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE  if \a sessionCount is too small, array element count is
                                                returned in \a sessionCount
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a sessionCount is NULL, or \a vgpuInstance is 0.
 *         - \ref LWML_ERROR_NOT_FOUND          if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetEncoderSessions(lwmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, lwmlEncoderSessionInfo_t *sessionInfo);

/**
* Retrieves the active frame buffer capture sessions statistics of a vGPU Instance
*
* For Maxwell &tm; or newer fully supported devices.
*
* @param vgpuInstance                      Identifier of the target vGPU instance
* @param fbcStats                          Reference to lwmlFBCStats_t structure contianing LwFBC stats
*
* @return
*         - \ref LWML_SUCCESS                  if \a fbcStats is fetched
*         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a vgpuInstance is 0, or \a fbcStats is NULL
*         - \ref LWML_ERROR_NOT_FOUND          if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
*/
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetFBCStats(lwmlVgpuInstance_t vgpuInstance, lwmlFBCStats_t *fbcStats);

/**
* Retrieves information about active frame buffer capture sessions on a vGPU Instance.
*
* An array of active FBC sessions is returned in the caller-supplied buffer pointed at by \a sessionInfo. The
* array element count is passed in \a sessionCount, and \a sessionCount is used to return the number of sessions
* written to the buffer.
*
* If the supplied buffer is not large enough to accomodate the active session array, the function returns
* LWML_ERROR_INSUFFICIENT_SIZE, with the element count of lwmlFBCSessionInfo_t array required in \a sessionCount.
* To query the number of active FBC sessions, call this function with *sessionCount = 0.  The code will return
* LWML_SUCCESS with number of active FBC sessions updated in *sessionCount.
*
* For Maxwell &tm; or newer fully supported devices.
*
* @note hResolution, vResolution, averageFPS and averageLatency data for a FBC session returned in \a sessionInfo may
*       be zero if there are no new frames captured since the session started.
*
* @param vgpuInstance                      Identifier of the target vGPU instance
* @param sessionCount                      Reference to caller supplied array size, and returns the number of sessions.
* @param sessionInfo                       Reference in which to return the session information
*
* @return
*         - \ref LWML_SUCCESS                  if \a sessionInfo is fetched
*         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a vgpuInstance is 0, or \a sessionCount is NULL.
*         - \ref LWML_ERROR_NOT_FOUND          if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref LWML_ERROR_INSUFFICIENT_SIZE  if \a sessionCount is too small, array element count is returned in \a sessionCount
*         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
*/
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetFBCSessions(lwmlVgpuInstance_t vgpuInstance, unsigned int *sessionCount, lwmlFBCSessionInfo_t *sessionInfo);

/**
* Retrieve the GPU Instance ID for the given vGPU Instance.
* The API will return a valid GPU Instance ID for MIG backed vGPU Instance, else ILWALID_GPU_INSTANCE_ID is returned.
*
* For Kepler &tm; or newer fully supported devices.
*
* @param vgpuInstance                      Identifier of the target vGPU instance
* @param gpuInstanceId                     GPU Instance ID
*
* @return
*         - \ref LWML_SUCCESS                  successful completion
*         - \ref LWML_ERROR_UNINITIALIZED      if the library has not been successfully initialized
*         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a vgpuInstance is 0, or \a gpuInstanceId is NULL.
*         - \ref LWML_ERROR_NOT_FOUND          if \a vgpuInstance does not match a valid active vGPU instance on the system
*         - \ref LWML_ERROR_UNKNOWN            on any unexpected error
*/
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetGpuInstanceId(lwmlVgpuInstance_t vgpuInstance, unsigned int *gpuInstanceId);

/** @} */

/***************************************************************************************************/
/** @defgroup lwml vGPU Migration
 * This chapter describes operations that are associated with vGPU Migration.
 *  @{
 */
/***************************************************************************************************/

/**
 * Structure representing range of vGPU versions.
 */
typedef struct lwmlVgpuVersion_st
{
    unsigned int milwersion; //!< Minimum vGPU version.
    unsigned int maxVersion; //!< Maximum vGPU version.
} lwmlVgpuVersion_t;

/**
 * vGPU metadata structure.
 */
typedef struct lwmlVgpuMetadata_st
{
    unsigned int             version;                                                    //!< Current version of the structure
    unsigned int             revision;                                                   //!< Current revision of the structure
    lwmlVgpuGuestInfoState_t guestInfoState;                                             //!< Current state of Guest-dependent fields
    char                     guestDriverVersion[LWML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE]; //!< Version of driver installed in guest
    char                     hostDriverVersion[LWML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];  //!< Version of driver installed in host
    unsigned int             reserved[6];                                                //!< Reserved for internal use
    unsigned int             vgpuVirtualizationCaps;                                     //!< vGPU virtualizaion capabilities bitfileld
    unsigned int             guestVgpuVersion;                                           //!< vGPU version of guest driver
    unsigned int             opaqueDataSize;                                             //!< Size of opaque data field in bytes
    char                     opaqueData[4];                                              //!< Opaque data
} lwmlVgpuMetadata_t;

/**
 * Physical GPU metadata structure
 */
typedef struct lwmlVgpuPgpuMetadata_st
{
    unsigned int            version;                                                    //!< Current version of the structure
    unsigned int            revision;                                                   //!< Current revision of the structure
    char                    hostDriverVersion[LWML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];  //!< Host driver version
    unsigned int            pgpuVirtualizationCaps;                                     //!< Pgpu virtualizaion capabilities bitfileld
    unsigned int            reserved[5];                                                //!< Reserved for internal use
    lwmlVgpuVersion_t       hostSupportedVgpuRange;                                     //!< vGPU version range supported by host driver
    unsigned int            opaqueDataSize;                                             //!< Size of opaque data field in bytes
    char                    opaqueData[4];                                              //!< Opaque data
} lwmlVgpuPgpuMetadata_t;

/**
 * vGPU VM compatibility codes
 */
typedef enum lwmlVgpuVmCompatibility_enum
{
    LWML_VGPU_VM_COMPATIBILITY_NONE         = 0x0,    //!< vGPU is not runnable
    LWML_VGPU_VM_COMPATIBILITY_COLD         = 0x1,    //!< vGPU is runnable from a cold / powered-off state (ACPI S5)
    LWML_VGPU_VM_COMPATIBILITY_HIBERNATE    = 0x2,    //!< vGPU is runnable from a hibernated state (ACPI S4)
    LWML_VGPU_VM_COMPATIBILITY_SLEEP        = 0x4,    //!< vGPU is runnable from a sleeped state (ACPI S3)
    LWML_VGPU_VM_COMPATIBILITY_LIVE         = 0x8,    //!< vGPU is runnable from a live/paused (ACPI S0)
} lwmlVgpuVmCompatibility_t;

/**
 *  vGPU-pGPU compatibility limit codes
 */
typedef enum lwmlVgpuPgpuCompatibilityLimitCode_enum
{
    LWML_VGPU_COMPATIBILITY_LIMIT_NONE          = 0x0,           //!< Compatibility is not limited.
    LWML_VGPU_COMPATIBILITY_LIMIT_HOST_DRIVER   = 0x1,           //!< ompatibility is limited by host driver version.
    LWML_VGPU_COMPATIBILITY_LIMIT_GUEST_DRIVER  = 0x2,           //!< Compatibility is limited by guest driver version.
    LWML_VGPU_COMPATIBILITY_LIMIT_GPU           = 0x4,           //!< Compatibility is limited by GPU hardware.
    LWML_VGPU_COMPATIBILITY_LIMIT_OTHER         = 0x80000000,    //!< Compatibility is limited by an undefined factor.
} lwmlVgpuPgpuCompatibilityLimitCode_t;

/**
 * vGPU-pGPU compatibility structure
 */
typedef struct lwmlVgpuPgpuCompatibility_st
{
    lwmlVgpuVmCompatibility_t               vgpuVmCompatibility;    //!< Compatibility of vGPU VM. See \ref lwmlVgpuVmCompatibility_t
    lwmlVgpuPgpuCompatibilityLimitCode_t    compatibilityLimitCode; //!< Limiting factor for vGPU-pGPU compatibility. See \ref lwmlVgpuPgpuCompatibilityLimitCode_t
} lwmlVgpuPgpuCompatibility_t;

/**
 * Returns vGPU metadata structure for a running vGPU. The structure contains information about the vGPU and its associated VM
 * such as the lwrrently installed LWPU guest driver version, together with host driver version and an opaque data section
 * containing internal state.
 *
 * lwmlVgpuInstanceGetMetadata() may be called at any time for a vGPU instance. Some fields in the returned structure are
 * dependent on information obtained from the guest VM, which may not yet have reached a state where that information
 * is available. The current state of these dependent fields is reflected in the info structure's \ref lwmlVgpuGuestInfoState_t field.
 *
 * The VMM may choose to read and save the vGPU's VM info as persistent metadata associated with the VM, and provide
 * it to Virtual GPU Manager when creating a vGPU for subsequent instances of the VM.
 *
 * The caller passes in a buffer via \a vgpuMetadata, with the size of the buffer in \a bufferSize. If the vGPU Metadata structure
 * is too large to fit in the supplied buffer, the function returns LWML_ERROR_INSUFFICIENT_SIZE with the size needed
 * in \a bufferSize.
 *
 * @param vgpuInstance             vGPU instance handle
 * @param vgpuMetadata             Pointer to caller-supplied buffer into which vGPU metadata is written
 * @param bufferSize               Size of vgpuMetadata buffer
 *
 * @return
 *         - \ref LWML_SUCCESS                   vGPU metadata structure was successfully returned
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE   vgpuMetadata buffer is too small, required size is returned in \a bufferSize
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT    if \a bufferSize is NULL or \a vgpuInstance is 0; if \a vgpuMetadata is NULL and the value of \a bufferSize is not 0.
 *         - \ref LWML_ERROR_NOT_FOUND           if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref LWML_ERROR_UNKNOWN             on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetMetadata(lwmlVgpuInstance_t vgpuInstance, lwmlVgpuMetadata_t *vgpuMetadata, unsigned int *bufferSize);

/**
 * Returns a vGPU metadata structure for the physical GPU indicated by \a device. The structure contains information about
 * the GPU and the lwrrently installed LWPU host driver version that's controlling it, together with an opaque data section
 * containing internal state.
 *
 * The caller passes in a buffer via \a pgpuMetadata, with the size of the buffer in \a bufferSize. If the \a pgpuMetadata
 * structure is too large to fit in the supplied buffer, the function returns LWML_ERROR_INSUFFICIENT_SIZE with the size needed
 * in \a bufferSize.
 *
 * @param device                The identifier of the target device
 * @param pgpuMetadata          Pointer to caller-supplied buffer into which \a pgpuMetadata is written
 * @param bufferSize            Pointer to size of \a pgpuMetadata buffer
 *
 * @return
 *         - \ref LWML_SUCCESS                   GPU metadata structure was successfully returned
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE   pgpuMetadata buffer is too small, required size is returned in \a bufferSize
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT    if \a bufferSize is NULL or \a device is invalid; if \a pgpuMetadata is NULL and the value of \a bufferSize is not 0.
 *         - \ref LWML_ERROR_NOT_SUPPORTED       vGPU is not supported by the system
 *         - \ref LWML_ERROR_UNKNOWN             on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetVgpuMetadata(lwmlDevice_t device, lwmlVgpuPgpuMetadata_t *pgpuMetadata, unsigned int *bufferSize);

/**
 * Takes a vGPU instance metadata structure read from \ref lwmlVgpuInstanceGetMetadata(), and a vGPU metadata structure for a
 * physical GPU read from \ref lwmlDeviceGetVgpuMetadata(), and returns compatibility information of the vGPU instance and the
 * physical GPU.
 *
 * The caller passes in a buffer via \a compatibilityInfo, into which a compatibility information structure is written. The
 * structure defines the states in which the vGPU / VM may be booted on the physical GPU. If the vGPU / VM compatibility
 * with the physical GPU is limited, a limit code indicates the factor limiting compability.
 * (see \ref lwmlVgpuPgpuCompatibilityLimitCode_t for details).
 *
 * Note: vGPU compatibility does not take into account dynamic capacity conditions that may limit a system's ability to
 *       boot a given vGPU or associated VM.
 *
 * @param vgpuMetadata          Pointer to caller-supplied vGPU metadata structure
 * @param pgpuMetadata          Pointer to caller-supplied GPU metadata structure
 * @param compatibilityInfo     Pointer to caller-supplied buffer to hold compatibility info
 *
 * @return
 *         - \ref LWML_SUCCESS                   vGPU metadata structure was successfully returned
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT    if \a vgpuMetadata or \a pgpuMetadata or \a bufferSize are NULL
 *         - \ref LWML_ERROR_UNKNOWN             on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlGetVgpuCompatibility(lwmlVgpuMetadata_t *vgpuMetadata, lwmlVgpuPgpuMetadata_t *pgpuMetadata, lwmlVgpuPgpuCompatibility_t *compatibilityInfo);

/**
 * Returns the properties of the physical GPU indicated by the device in an ascii-encoded string format.
 *
 * The caller passes in a buffer via \a pgpuMetadata, with the size of the buffer in \a bufferSize. If the
 * string is too large to fit in the supplied buffer, the function returns LWML_ERROR_INSUFFICIENT_SIZE with the size needed
 * in \a bufferSize.
 *
 * @param device                The identifier of the target device
 * @param pgpuMetadata          Pointer to caller-supplied buffer into which \a pgpuMetadata is written
 * @param bufferSize            Pointer to size of \a pgpuMetadata buffer
 *
 * @return
 *         - \ref LWML_SUCCESS                   GPU metadata structure was successfully returned
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE   \a pgpuMetadata buffer is too small, required size is returned in \a bufferSize
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT    if \a bufferSize is NULL or \a device is invalid; if \a pgpuMetadata is NULL and the value of \a bufferSize is not 0.
 *         - \ref LWML_ERROR_NOT_SUPPORTED       if vGPU is not supported by the system
 *         - \ref LWML_ERROR_UNKNOWN             on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetPgpuMetadataString(lwmlDevice_t device, char *pgpuMetadata, unsigned int *bufferSize);

/*
 * Virtual GPU (vGPU) version
 *
 * The LWPU vGPU Manager and the guest drivers are tagged with a range of supported vGPU versions. This determines the range of LWPU guest driver versions that
 * are compatible for vGPU feature support with a given LWPU vGPU Manager. For vGPU feature support, the range of supported versions for the LWPU vGPU Manager 
 * and the guest driver must overlap. Otherwise, the guest driver fails to load in the VM.
 *
 * When the LWPU guest driver loads, either when the VM is booted or when the driver is installed or upgraded, a negotiation oclwrs between the guest driver
 * and the LWPU vGPU Manager to select the highest mutually compatible vGPU version. The negotiated vGPU version stays the same across VM migration.
 */

/**
 * Query the ranges of supported vGPU versions.
 *
 * This function gets the linear range of supported vGPU versions that is preset for the LWPU vGPU Manager and the range set by an administrator.
 * If the preset range has not been overridden by \ref lwmlSetVgpuVersion, both ranges are the same.
 *
 * The caller passes pointers to the following \ref lwmlVgpuVersion_t structures, into which the LWPU vGPU Manager writes the ranges:
 * 1. \a supported structure that represents the preset range of vGPU versions supported by the LWPU vGPU Manager.
 * 2. \a current structure that represents the range of supported vGPU versions set by an administrator. By default, this range is the same as the preset range.
 *
 * @param supported  Pointer to the structure in which the preset range of vGPU versions supported by the LWPU vGPU Manager is written
 * @param current    Pointer to the structure in which the range of supported vGPU versions set by an administrator is written
 *
 * @return
 * - \ref LWML_SUCCESS                 The vGPU version range structures were successfully obtained.
 * - \ref LWML_ERROR_NOT_SUPPORTED     The API is not supported.
 * - \ref LWML_ERROR_ILWALID_ARGUMENT  The \a supported parameter or the \a current parameter is NULL.
 * - \ref LWML_ERROR_UNKNOWN           An error oclwrred while the data was being fetched.
 */
lwmlReturn_t DECLDIR lwmlGetVgpuVersion(lwmlVgpuVersion_t *supported, lwmlVgpuVersion_t *current);

/**
 * Override the preset range of vGPU versions supported by the LWPU vGPU Manager with a range set by an administrator.
 *
 * This function configures the LWPU vGPU Manager with a range of supported vGPU versions set by an administrator. This range must be a subset of the
 * preset range that the LWPU vGPU Manager supports. The custom range set by an administrator takes precedence over the preset range and is advertised to
 * the guest VM for negotiating the vGPU version. See \ref lwmlGetVgpuVersion for details of how to query the preset range of versions supported.
 *
 * This function takes a pointer to vGPU version range structure \ref lwmlVgpuVersion_t as input to override the preset vGPU version range that the LWPU vGPU Manager supports.
 *
 * After host system reboot or driver reload, the range of supported versions reverts to the range that is preset for the LWPU vGPU Manager.
 *
 * @note 1. The range set by the administrator must be a subset of the preset range that the LWPU vGPU Manager supports. Otherwise, an error is returned.
 *       2. If the range of supported guest driver versions does not overlap the range set by the administrator, the guest driver fails to load.
 *       3. If the range of supported guest driver versions overlaps the range set by the administrator, the guest driver will load with a negotiated 
 *          vGPU version that is the maximum value in the overlapping range.
 *       4. No VMs must be running on the host when this function is called. If a VM is running on the host, the call to this function fails.
 *
 * @param vgpuVersion   Pointer to a caller-supplied range of supported vGPU versions.
 *
 * @return
 * - \ref LWML_SUCCESS                 The preset range of supported vGPU versions was successfully overridden.
 * - \ref LWML_ERROR_NOT_SUPPORTED     The API is not supported.
 * - \ref LWML_ERROR_IN_USE            The range was not overridden because a VM is running on the host.
 * - \ref LWML_ERROR_ILWALID_ARGUMENT  The \a vgpuVersion parameter specifies a range that is outside the range supported by the LWPU vGPU Manager or if \a vgpuVersion is NULL.
 */
lwmlReturn_t DECLDIR lwmlSetVgpuVersion(lwmlVgpuVersion_t *vgpuVersion);

/** @} */

/***************************************************************************************************/
/** @defgroup lwmlUtil vGPU Utilization and Accounting 
 * This chapter describes operations that are associated with vGPU Utilization and Accounting.
 *  @{
 */
/***************************************************************************************************/

/**
 * Retrieves current utilization for vGPUs on a physical GPU (device).
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * Reads recent utilization of GPU SM (3D/Compute), framebuffer, video encoder, and video decoder for vGPU instances running
 * on a device. Utilization values are returned as an array of utilization sample structures in the caller-supplied buffer
 * pointed at by \a utilizationSamples. One utilization sample structure is returned per vGPU instance, and includes the
 * CPU timestamp at which the samples were recorded. Individual utilization values are returned as "unsigned int" values
 * in lwmlValue_t unions. The function sets the caller-supplied \a sampleValType to LWML_VALUE_TYPE_UNSIGNED_INT to
 * indicate the returned value type.
 *
 * To read utilization values, first determine the size of buffer required to hold the samples by ilwoking the function with
 * \a utilizationSamples set to NULL. The function will return LWML_ERROR_INSUFFICIENT_SIZE, with the current vGPU instance
 * count in \a vgpuInstanceSamplesCount, or LWML_SUCCESS if the current vGPU instance count is zero. The caller should allocate
 * a buffer of size vgpuInstanceSamplesCount * sizeof(lwmlVgpuInstanceUtilizationSample_t). Ilwoke the function again with
 * the allocated buffer passed in \a utilizationSamples, and \a vgpuInstanceSamplesCount set to the number of entries the
 * buffer is sized for.
 *
 * On successful return, the function updates \a vgpuInstanceSampleCount with the number of vGPU utilization sample
 * structures that were actually written. This may differ from a previously read value as vGPU instances are created or
 * destroyed.
 *
 * lastSeenTimeStamp represents the CPU timestamp in microseconds at which utilization samples were last read. Set it to 0
 * to read utilization based on all the samples maintained by the driver's internal sample buffer. Set lastSeenTimeStamp
 * to a timeStamp retrieved from a previous query to read utilization since the previous query.
 *
 * @param device                        The identifier for the target device
 * @param lastSeenTimeStamp             Return only samples with timestamp greater than lastSeenTimeStamp.
 * @param sampleValType                 Pointer to caller-supplied buffer to hold the type of returned sample values
 * @param vgpuInstanceSamplesCount      Pointer to caller-supplied array size, and returns number of vGPU instances
 * @param utilizationSamples            Pointer to caller-supplied buffer in which vGPU utilization samples are returned

 * @return
 *         - \ref LWML_SUCCESS                 if utilization samples are successfully retrieved
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, \a vgpuInstanceSamplesCount or \a sampleValType is
 *                                             NULL, or a sample count of 0 is passed with a non-NULL \a utilizationSamples
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if supplied \a vgpuInstanceSamplesCount is too small to return samples for all
 *                                             vGPU instances lwrrently exelwting on the device
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if vGPU is not supported by the device
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_NOT_FOUND         if sample entries are not found
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetVgpuUtilization(lwmlDevice_t device, unsigned long long lastSeenTimeStamp,
                                                  lwmlValueType_t *sampleValType, unsigned int *vgpuInstanceSamplesCount,
                                                  lwmlVgpuInstanceUtilizationSample_t *utilizationSamples);

/**
 * Retrieves current utilization for processes running on vGPUs on a physical GPU (device).
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * Reads recent utilization of GPU SM (3D/Compute), framebuffer, video encoder, and video decoder for processes running on
 * vGPU instances active on a device. Utilization values are returned as an array of utilization sample structures in the
 * caller-supplied buffer pointed at by \a utilizationSamples. One utilization sample structure is returned per process running
 * on vGPU instances, that had some non-zero utilization during the last sample period. It includes the CPU timestamp at which
 * the samples were recorded. Individual utilization values are returned as "unsigned int" values.
 *
 * To read utilization values, first determine the size of buffer required to hold the samples by ilwoking the function with
 * \a utilizationSamples set to NULL. The function will return LWML_ERROR_INSUFFICIENT_SIZE, with the current vGPU instance
 * count in \a vgpuProcessSamplesCount. The caller should allocate a buffer of size
 * vgpuProcessSamplesCount * sizeof(lwmlVgpuProcessUtilizationSample_t). Ilwoke the function again with
 * the allocated buffer passed in \a utilizationSamples, and \a vgpuProcessSamplesCount set to the number of entries the
 * buffer is sized for.
 *
 * On successful return, the function updates \a vgpuSubProcessSampleCount with the number of vGPU sub process utilization sample
 * structures that were actually written. This may differ from a previously read value depending on the number of processes that are active
 * in any given sample period.
 *
 * lastSeenTimeStamp represents the CPU timestamp in microseconds at which utilization samples were last read. Set it to 0
 * to read utilization based on all the samples maintained by the driver's internal sample buffer. Set lastSeenTimeStamp
 * to a timeStamp retrieved from a previous query to read utilization since the previous query.
 *
 * @param device                        The identifier for the target device
 * @param lastSeenTimeStamp             Return only samples with timestamp greater than lastSeenTimeStamp.
 * @param vgpuProcessSamplesCount       Pointer to caller-supplied array size, and returns number of processes running on vGPU instances
 * @param utilizationSamples            Pointer to caller-supplied buffer in which vGPU sub process utilization samples are returned

 * @return
 *         - \ref LWML_SUCCESS                 if utilization samples are successfully retrieved
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, \a vgpuProcessSamplesCount or a sample count of 0 is
 *                                             passed with a non-NULL \a utilizationSamples
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if supplied \a vgpuProcessSamplesCount is too small to return samples for all
 *                                             vGPU instances lwrrently exelwting on the device
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if vGPU is not supported by the device
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_NOT_FOUND         if sample entries are not found
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetVgpuProcessUtilization(lwmlDevice_t device, unsigned long long lastSeenTimeStamp,
                                                         unsigned int *vgpuProcessSamplesCount,
                                                         lwmlVgpuProcessUtilizationSample_t *utilizationSamples);
/**
 * Queries the state of per process accounting mode on vGPU.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * @param vgpuInstance            The identifier of the target vGPU instance
 * @param mode                    Reference in which to return the current accounting mode
 *
 * @return
 *         - \ref LWML_SUCCESS                 if the mode has been successfully retrieved
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuInstance is 0, or \a mode is NULL
 *         - \ref LWML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the vGPU doesn't support this feature
 *         - \ref LWML_ERROR_DRIVER_NOT_LOADED if LWPU driver is not running on the vGPU instance
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetAccountingMode(lwmlVgpuInstance_t vgpuInstance, lwmlEnableState_t *mode);

/**
 * Queries list of processes running on vGPU that can be queried for accounting stats. The list of processes
 * returned can be in running or terminated state.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * To just query the maximum number of processes that can be queried, call this function with *count = 0 and
 * pids=NULL. The return code will be LWML_ERROR_INSUFFICIENT_SIZE, or LWML_SUCCESS if list is empty.
 *
 * For more details see \ref lwmlVgpuInstanceGetAccountingStats.
 *
 * @note In case of PID collision some processes might not be accessible before the cirlwlar buffer is full.
 *
 * @param vgpuInstance            The identifier of the target vGPU instance
 * @param count                   Reference in which to provide the \a pids array size, and
 *                                to return the number of elements ready to be queried
 * @param pids                    Reference in which to return list of process ids
 *
 * @return
 *         - \ref LWML_SUCCESS                 if pids were successfully retrieved
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuInstance is 0, or \a count is NULL
 *         - \ref LWML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the vGPU doesn't support this feature or accounting mode is disabled
 *         - \ref LWML_ERROR_INSUFFICIENT_SIZE if \a count is too small (\a count is set to expected value)
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see lwmlVgpuInstanceGetAccountingPids
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetAccountingPids(lwmlVgpuInstance_t vgpuInstance, unsigned int *count, unsigned int *pids);

/**
 * Queries process's accounting stats.
 *
 * For Maxwell &tm; or newer fully supported devices.
 *
 * Accounting stats capture GPU utilization and other statistics across the lifetime of a process, and
 * can be queried during life time of the process or after its termination.
 * The time field in \ref lwmlAccountingStats_t is reported as 0 during the lifetime of the process and
 * updated to actual running time after its termination.
 * Accounting stats are kept in a cirlwlar buffer, newly created processes overwrite information about old
 * processes.
 *
 * See \ref lwmlAccountingStats_t for description of each returned metric.
 * List of processes that can be queried can be retrieved from \ref lwmlVgpuInstanceGetAccountingPids.
 *
 * @note Accounting Mode needs to be on. See \ref lwmlVgpuInstanceGetAccountingMode.
 * @note Only compute and graphics applications stats can be queried. Monitoring applications stats can't be
 *         queried since they don't contribute to GPU utilization.
 * @note In case of pid collision stats of only the latest process (that terminated last) will be reported
 *
 * @param vgpuInstance            The identifier of the target vGPU instance
 * @param pid                     Process Id of the target process to query stats for
 * @param stats                   Reference in which to return the process's accounting stats
 *
 * @return
 *         - \ref LWML_SUCCESS                 if stats have been successfully retrieved
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuInstance is 0, or \a stats is NULL
 *         - \ref LWML_ERROR_NOT_FOUND         if \a vgpuInstance does not match a valid active vGPU instance on the system
 *                                             or \a stats is not found
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the vGPU doesn't support this feature or accounting mode is disabled
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceGetAccountingStats(lwmlVgpuInstance_t vgpuInstance, unsigned int pid, lwmlAccountingStats_t *stats);

/**
 * Clears accounting information of the vGPU instance that have already terminated.
 *
 * For Maxwell &tm; or newer fully supported devices.
 * Requires root/admin permissions.
 *
 * @note Accounting Mode needs to be on. See \ref lwmlVgpuInstanceGetAccountingMode.
 * @note Only compute and graphics applications stats are reported and can be cleared since monitoring applications
 *         stats don't contribute to GPU utilization.
 *
 * @param vgpuInstance            The identifier of the target vGPU instance
 *
 * @return
 *         - \ref LWML_SUCCESS                 if accounting information has been cleared
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a vgpuInstance is invalid
 *         - \ref LWML_ERROR_NO_PERMISSION     if the user doesn't have permission to perform this operation
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the vGPU doesn't support this feature or accounting mode is disabled
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlVgpuInstanceClearAccountingPids(lwmlVgpuInstance_t vgpuInstance);
/** @} */

/***************************************************************************************************/
/** @defgroup lwmlExcludedGpuQueries Excluded GPU Queries
 * This chapter describes LWML operations that are associated with excluded GPUs.
 *  @{
 */
/***************************************************************************************************/

/**
 * Excluded GPU device information
 **/
typedef struct lwmlExcludedDeviceInfo_st
{
    lwmlPciInfo_t pciInfo;                   //!< The PCI information for the excluded GPU
    char uuid[LWML_DEVICE_UUID_BUFFER_SIZE]; //!< The ASCII string UUID for the excluded GPU
} lwmlExcludedDeviceInfo_t;

 /**
 * Retrieves the number of excluded GPU devices in the system.
 * 
 * For all products.
 *
 * @param deviceCount                          Reference in which to return the number of excluded devices
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a deviceCount has been set
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a deviceCount is NULL
 */
lwmlReturn_t DECLDIR lwmlGetExcludedDeviceCount(unsigned int *deviceCount);

/**
 * Acquire the device information for an excluded GPU device, based on its index.
 * 
 * For all products.
 *
 * Valid indices are derived from the \a deviceCount returned by 
 *   \ref lwmlGetExcludedDeviceCount(). For example, if \a deviceCount is 2 the valid indices  
 *   are 0 and 1, corresponding to GPU 0 and GPU 1.
 *
 * @param index                                The index of the target GPU, >= 0 and < \a deviceCount
 * @param info                                 Reference in which to return the device information
 * 
 * @return 
 *         - \ref LWML_SUCCESS                  if \a device has been set
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT   if \a index is invalid or \a info is NULL
 *
 * @see lwmlGetExcludedDeviceCount
 */
lwmlReturn_t DECLDIR lwmlGetExcludedDeviceInfoByIndex(unsigned int index, lwmlExcludedDeviceInfo_t *info);

/** @} */

/***************************************************************************************************/
/** @defgroup lwmlMultiInstanceGPU Multi Instance GPU Management
 * This chapter describes LWML operations that are associated with Multi Instance GPU management.
 *  @{
 */
/***************************************************************************************************/

/**
 * Disable Multi Instance GPU mode.
 */
#define LWML_DEVICE_MIG_DISABLE 0x0

/**
 * Enable Multi Instance GPU mode.
 */
#define LWML_DEVICE_MIG_ENABLE 0x1

/**
 * GPU instance profiles.
 *
 * These macros should be passed to \ref lwmlDeviceGetGpuInstanceProfileInfo to retrieve the
 * detailed information about a GPU instance such as profile ID, engine counts.
 */
#define LWML_GPU_INSTANCE_PROFILE_1_SLICE 0x0
#define LWML_GPU_INSTANCE_PROFILE_2_SLICE 0x1
#define LWML_GPU_INSTANCE_PROFILE_3_SLICE 0x2
#define LWML_GPU_INSTANCE_PROFILE_4_SLICE 0x3
#define LWML_GPU_INSTANCE_PROFILE_7_SLICE 0x4
#define LWML_GPU_INSTANCE_PROFILE_8_SLICE 0x5
#define LWML_GPU_INSTANCE_PROFILE_6_SLICE 0x6
#define LWML_GPU_INSTANCE_PROFILE_COUNT   0x7

typedef struct lwmlGpuInstancePlacement_st
{
    unsigned int start;
    unsigned int size;
} lwmlGpuInstancePlacement_t;

typedef struct lwmlGpuInstanceProfileInfo_st
{
    unsigned int id;                  //!< Unique profile ID within the device
    unsigned int isP2pSupported;      //!< Peer-to-Peer support
    unsigned int sliceCount;          //!< GPU Slice count
    unsigned int instanceCount;       //!< GPU instance count
    unsigned int multiprocessorCount; //!< Streaming Multiprocessor count
    unsigned int copyEngineCount;     //!< Copy Engine count
    unsigned int decoderCount;        //!< Decoder Engine count
    unsigned int encoderCount;        //!< Encoder Engine count
    unsigned int jpegCount;           //!< JPEG Engine count
    unsigned int ofaCount;            //!< OFA Engine count
    unsigned long long memorySizeMB;  //!< Memory size in MBytes
} lwmlGpuInstanceProfileInfo_t;

typedef struct lwmlGpuInstanceInfo_st
{
    lwmlDevice_t device;                      //!< Parent device
    unsigned int id;                          //!< Unique instance ID within the device
    unsigned int profileId;                   //!< Unique profile ID within the device
    lwmlGpuInstancePlacement_t placement;     //!< Placement for this instance
} lwmlGpuInstanceInfo_t;

typedef struct lwmlGpuInstance_st* lwmlGpuInstance_t;

/**
 * Compute instance profiles.
 *
 * These macros should be passed to \ref lwmlGpuInstanceGetComputeInstanceProfileInfo to retrieve the
 * detailed information about a compute instance such as profile ID, engine counts
 */
#define LWML_COMPUTE_INSTANCE_PROFILE_1_SLICE 0x0
#define LWML_COMPUTE_INSTANCE_PROFILE_2_SLICE 0x1
#define LWML_COMPUTE_INSTANCE_PROFILE_3_SLICE 0x2
#define LWML_COMPUTE_INSTANCE_PROFILE_4_SLICE 0x3
#define LWML_COMPUTE_INSTANCE_PROFILE_7_SLICE 0x4
#define LWML_COMPUTE_INSTANCE_PROFILE_8_SLICE 0x5
#define LWML_COMPUTE_INSTANCE_PROFILE_6_SLICE 0x6
#define LWML_COMPUTE_INSTANCE_PROFILE_COUNT   0x7

#define LWML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED 0x0 //!< All the engines except multiprocessors would be shared
#define LWML_COMPUTE_INSTANCE_ENGINE_PROFILE_COUNT  0x1

typedef struct lwmlComputeInstancePlacement_st
{
    unsigned int start;
    unsigned int size;
} lwmlComputeInstancePlacement_t;

typedef struct lwmlComputeInstanceProfileInfo_st
{
    unsigned int id;                    //!< Unique profile ID within the GPU instance
    unsigned int sliceCount;            //!< GPU Slice count
    unsigned int instanceCount;         //!< Compute instance count
    unsigned int multiprocessorCount;   //!< Streaming Multiprocessor count
    unsigned int sharedCopyEngineCount; //!< Shared Copy Engine count
    unsigned int sharedDecoderCount;    //!< Shared Decoder Engine count
    unsigned int sharedEncoderCount;    //!< Shared Encoder Engine count
    unsigned int sharedJpegCount;       //!< Shared JPEG Engine count
    unsigned int sharedOfaCount;        //!< Shared OFA Engine count
} lwmlComputeInstanceProfileInfo_t;

typedef struct lwmlComputeInstanceInfo_st
{
    lwmlDevice_t device;                      //!< Parent device
    lwmlGpuInstance_t gpuInstance;            //!< Parent GPU instance
    unsigned int id;                          //!< Unique instance ID within the GPU instance
    unsigned int profileId;                   //!< Unique profile ID within the GPU instance
    lwmlComputeInstancePlacement_t placement; //!< Placement for this instance within the GPU instance's slice range {0, sliceCount}
} lwmlComputeInstanceInfo_t;

typedef struct lwmlComputeInstance_st* lwmlComputeInstance_t;

/**
 * Set MIG mode for the device.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Requires root user.
 *
 * This mode determines whether a GPU instance can be created.
 *
 * This API may unbind or reset the device to activate the requested mode. Thus, the attributes associated with the
 * device, such as minor number, might change. The caller of this API is expected to query such attributes again.
 *
 * On certain platforms like pass-through virtualization, where reset functionality may not be exposed directly, VM
 * reboot is required. \a activationStatus would return \ref LWML_ERROR_RESET_REQUIRED for such cases.
 *
 * \a activationStatus would return the appropriate error code upon unsuccessful activation. For example, if device
 * unbind fails because the device isn't idle, \ref LWML_ERROR_IN_USE would be returned. The caller of this API
 * is expected to idle the device and retry setting the \a mode.
 *
 * @note On Windows, only disabling MIG mode is supported. \a activationStatus would return \ref
 *       LWML_ERROR_NOT_SUPPORTED as GPU reset is not supported on Windows through this API.
 *
 * @param device                               The identifier of the target device
 * @param mode                                 The mode to be set, \ref LWML_DEVICE_MIG_DISABLE or
 *                                             \ref LWML_DEVICE_MIG_ENABLE
 * @param activationStatus                     The activationStatus status
 *
 * @return
 *         - \ref LWML_SUCCESS                 Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  If \a device,\a mode or \a activationStatus are invalid
 *         - \ref LWML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 *         - \ref LWML_ERROR_NOT_SUPPORTED     If \a device doesn't support MIG mode
 */
lwmlReturn_t DECLDIR lwmlDeviceSetMigMode(lwmlDevice_t device, unsigned int mode, lwmlReturn_t *activationStatus);

/**
 * Get MIG mode for the device.
 *
 * For Ampere &tm; or newer fully supported devices.
 *
 * Changing MIG modes may require device unbind or reset. The "pending" MIG mode refers to the target mode following the
 * next activation trigger.
 *
 * @param device                               The identifier of the target device
 * @param lwrrentMode                          Returns the current mode, \ref LWML_DEVICE_MIG_DISABLE or
 *                                             \ref LWML_DEVICE_MIG_ENABLE
 * @param pendingMode                          Returns the pending mode, \ref LWML_DEVICE_MIG_DISABLE or
 *                                             \ref LWML_DEVICE_MIG_ENABLE
 *
 * @return
 *         - \ref LWML_SUCCESS                 Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  If \a device, \a lwrrentMode or \a pendingMode are invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     If \a device doesn't support MIG mode
 */
lwmlReturn_t DECLDIR lwmlDeviceGetMigMode(lwmlDevice_t device, unsigned int *lwrrentMode, unsigned int *pendingMode);

/**
 * Get GPU instance profile information.
 *
 * Information provided by this API is immutable throughout the lifetime of a MIG mode.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param device                               The identifier of the target device
 * @param profile                              One of the LWML_GPU_INSTANCE_PROFILE_*
 * @param info                                 Returns detailed profile information
 *
 * @return
 *         - \ref LWML_SUCCESS                 Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  If \a device, \a profile or \a info are invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled or \a profile isn't supported
 *         - \ref LWML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
lwmlReturn_t DECLDIR lwmlDeviceGetGpuInstanceProfileInfo(lwmlDevice_t device, unsigned int profile,
                                                         lwmlGpuInstanceProfileInfo_t *info);

/**
 * Get GPU instance placements.
 *
 * A placement represents the location of a GPU instance within a device. This API only returns all the possible
 * placements for the given profile.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param device                               The identifier of the target device
 * @param profileId                            The GPU instance profile ID. See \ref lwmlDeviceGetGpuInstanceProfileInfo
 * @param placements                           Returns placements, the buffer must be large enough to accommodate
 *                                             the instances supported by the profile.
 *                                             See \ref lwmlDeviceGetGpuInstanceProfileInfo
 * @param count                                The count of returned placements
 *
 * @return
 *         - \ref LWML_SUCCESS                 Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  If \a device, \a profileId, \a placements or \a count are invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled or \a profileId isn't supported
 *         - \ref LWML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
lwmlReturn_t DECLDIR lwmlDeviceGetGpuInstancePossiblePlacements(lwmlDevice_t device, unsigned int profileId,
                                                                lwmlGpuInstancePlacement_t *placements,
                                                                unsigned int *count);

/**
 * Get GPU instance profile capacity.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param device                               The identifier of the target device
 * @param profileId                            The GPU instance profile ID. See \ref lwmlDeviceGetGpuInstanceProfileInfo
 * @param count                                Returns remaining instance count for the profile ID
 *
 * @return
 *         - \ref LWML_SUCCESS                 Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  If \a device, \a profileId or \a count are invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled or \a profileId isn't supported
 *         - \ref LWML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
lwmlReturn_t DECLDIR lwmlDeviceGetGpuInstanceRemainingCapacity(lwmlDevice_t device, unsigned int profileId,
                                                               unsigned int *count);

/**
 * Create GPU instance.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * If the parent device is unbound, reset or the GPU instance is destroyed explicitly, the GPU instance handle would
 * become invalid. The GPU instance must be recreated to acquire a valid handle.
 *
 * @param device                               The identifier of the target device
 * @param profileId                            The GPU instance profile ID. See \ref lwmlDeviceGetGpuInstanceProfileInfo
 * @param gpuInstance                          Returns the GPU instance handle
 *
 * @return
 *         - \ref LWML_SUCCESS                       Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED           If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT        If \a device, \a profile, \a profileId or \a gpuInstance are invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED           If \a device doesn't have MIG mode enabled or in vGPU guest
 *         - \ref LWML_ERROR_NO_PERMISSION           If user doesn't have permission to perform the operation
 *         - \ref LWML_ERROR_INSUFFICIENT_RESOURCES  If the requested GPU instance could not be created
 */
lwmlReturn_t DECLDIR lwmlDeviceCreateGpuInstance(lwmlDevice_t device, unsigned int profileId,
                                                 lwmlGpuInstance_t *gpuInstance);

/**
 * Create GPU instance with the specified placement.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * If the parent device is unbound, reset or the GPU instance is destroyed explicitly, the GPU instance handle would
 * become invalid. The GPU instance must be recreated to acquire a valid handle.
 *
 * @param device                               The identifier of the target device
 * @param profileId                            The GPU instance profile ID. See \ref lwmlDeviceGetGpuInstanceProfileInfo
 * @param placement                            The requested placement. See \ref lwmlDeviceGetGpuInstancePossiblePlacements
 * @param gpuInstance                          Returns the GPU instance handle
 *
 * @return
 *         - \ref LWML_SUCCESS                       Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED           If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT        If \a device, \a profile, \a profileId, \a placement or \a gpuInstance
 *                                                   are invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED           If \a device doesn't have MIG mode enabled or in vGPU guest
 *         - \ref LWML_ERROR_NO_PERMISSION           If user doesn't have permission to perform the operation
 *         - \ref LWML_ERROR_INSUFFICIENT_RESOURCES  If the requested GPU instance could not be created
 */
lwmlReturn_t DECLDIR lwmlDeviceCreateGpuInstanceWithPlacement(lwmlDevice_t device, unsigned int profileId,
                                                              const lwmlGpuInstancePlacement_t *placement,
                                                              lwmlGpuInstance_t *gpuInstance);
/**
 * Destroy GPU instance.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param gpuInstance                          The GPU instance handle
 *
 * @return
 *         - \ref LWML_SUCCESS                 Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  If \a gpuInstance is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled or in vGPU guest
 *         - \ref LWML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 *         - \ref LWML_ERROR_IN_USE            If the GPU instance is in use. This error would be returned if processes
 *                                             (e.g. LWCA application) or compute instances are active on the
 *                                             GPU instance.
 */
lwmlReturn_t DECLDIR lwmlGpuInstanceDestroy(lwmlGpuInstance_t gpuInstance);

/**
 * Get GPU instances for given profile ID.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param device                               The identifier of the target device
 * @param profileId                            The GPU instance profile ID. See \ref lwmlDeviceGetGpuInstanceProfileInfo
 * @param gpuInstances                         Returns pre-exiting GPU instances, the buffer must be large enough to
 *                                             accommodate the instances supported by the profile.
 *                                             See \ref lwmlDeviceGetGpuInstanceProfileInfo
 * @param count                                The count of returned GPU instances
 *
 * @return
 *         - \ref LWML_SUCCESS                 Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  If \a device, \a profileId, \a gpuInstances or \a count are invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled
 *         - \ref LWML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
lwmlReturn_t DECLDIR lwmlDeviceGetGpuInstances(lwmlDevice_t device, unsigned int profileId,
                                               lwmlGpuInstance_t *gpuInstances, unsigned int *count);

/**
 * Get GPU instances for given instance ID.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param device                               The identifier of the target device
 * @param id                                   The GPU instance ID
 * @param gpuInstance                          Returns GPU instance
 *
 * @return
 *         - \ref LWML_SUCCESS                 Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  If \a device, \a id or \a gpuInstance are invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled
 *         - \ref LWML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 *         - \ref LWML_ERROR_NOT_FOUND         If the GPU instance is not found.
 */
lwmlReturn_t DECLDIR lwmlDeviceGetGpuInstanceById(lwmlDevice_t device, unsigned int id, lwmlGpuInstance_t *gpuInstance);

/**
 * Get GPU instance information.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param gpuInstance                          The GPU instance handle
 * @param info                                 Return GPU instance information
 *
 * @return
 *         - \ref LWML_SUCCESS                 Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  If \a gpuInstance or \a info are invalid
 *         - \ref LWML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
lwmlReturn_t DECLDIR lwmlGpuInstanceGetInfo(lwmlGpuInstance_t gpuInstance, lwmlGpuInstanceInfo_t *info);

/**
 * Get compute instance profile information.
 *
 * Information provided by this API is immutable throughout the lifetime of a MIG mode.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param gpuInstance                          The identifier of the target GPU instance
 * @param profile                              One of the LWML_COMPUTE_INSTANCE_PROFILE_*
 * @param engProfile                           One of the LWML_COMPUTE_INSTANCE_ENGINE_PROFILE_*
 * @param info                                 Returns detailed profile information
 *
 * @return
 *         - \ref LWML_SUCCESS                 Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  If \a gpuInstance, \a profile, \a engProfile or \a info are invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     If \a profile isn't supported
 *         - \ref LWML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
lwmlReturn_t DECLDIR lwmlGpuInstanceGetComputeInstanceProfileInfo(lwmlGpuInstance_t gpuInstance, unsigned int profile,
                                                                  unsigned int engProfile,
                                                                  lwmlComputeInstanceProfileInfo_t *info);

/**
 * Get compute instance profile capacity.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param gpuInstance                          The identifier of the target GPU instance
 * @param profileId                            The compute instance profile ID.
 *                                             See \ref lwmlGpuInstanceGetComputeInstanceProfileInfo
 * @param count                                Returns remaining instance count for the profile ID
 *
 * @return
 *         - \ref LWML_SUCCESS                 Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  If \a gpuInstance, \a profileId or \a availableCount are invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     If \a profileId isn't supported
 *         - \ref LWML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
lwmlReturn_t DECLDIR lwmlGpuInstanceGetComputeInstanceRemainingCapacity(lwmlGpuInstance_t gpuInstance,
                                                                        unsigned int profileId, unsigned int *count);

/**
 * Create compute instance.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * If the parent device is unbound, reset or the parent GPU instance is destroyed or the compute instance is destroyed
 * explicitly, the compute instance handle would become invalid. The compute instance must be recreated to acquire
 * a valid handle.
 *
 * @param gpuInstance                          The identifier of the target GPU instance
 * @param profileId                            The compute instance profile ID.
 *                                             See \ref lwmlGpuInstanceGetComputeInstanceProfileInfo
 * @param computeInstance                      Returns the compute instance handle
 *
 * @return
 *         - \ref LWML_SUCCESS                       Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED           If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT        If \a gpuInstance, \a profile, \a profileId or \a computeInstance
 *                                                   are invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED           If \a profileId isn't supported
 *         - \ref LWML_ERROR_NO_PERMISSION           If user doesn't have permission to perform the operation
 *         - \ref LWML_ERROR_INSUFFICIENT_RESOURCES  If the requested compute instance could not be created
 */
lwmlReturn_t DECLDIR lwmlGpuInstanceCreateComputeInstance(lwmlGpuInstance_t gpuInstance, unsigned int profileId,
                                                          lwmlComputeInstance_t *computeInstance);

/**
 * Destroy compute instance.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param computeInstance                      The compute instance handle
 *
 * @return
 *         - \ref LWML_SUCCESS                 Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  If \a computeInstance is invalid
 *         - \ref LWML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 *         - \ref LWML_ERROR_IN_USE            If the compute instance is in use. This error would be returned if
 *                                             processes (e.g. LWCA application) are active on the compute instance.
 */
lwmlReturn_t DECLDIR lwmlComputeInstanceDestroy(lwmlComputeInstance_t computeInstance);

/**
 * Get compute instances for given profile ID.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param gpuInstance                          The identifier of the target GPU instance
 * @param profileId                            The compute instance profile ID.
 *                                             See \ref lwmlGpuInstanceGetComputeInstanceProfileInfo
 * @param computeInstances                     Returns pre-exiting compute instances, the buffer must be large enough to
 *                                             accommodate the instances supported by the profile.
 *                                             See \ref lwmlGpuInstanceGetComputeInstanceProfileInfo
 * @param count                                The count of returned compute instances
 *
 * @return
 *         - \ref LWML_SUCCESS                 Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  If \a gpuInstance, \a profileId, \a computeInstances or \a count
 *                                             are invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     If \a profileId isn't supported
 *         - \ref LWML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
lwmlReturn_t DECLDIR lwmlGpuInstanceGetComputeInstances(lwmlGpuInstance_t gpuInstance, unsigned int profileId,
                                                        lwmlComputeInstance_t *computeInstances, unsigned int *count);

/**
 * Get compute instance for given instance ID.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 * Requires privileged user.
 *
 * @param gpuInstance                          The identifier of the target GPU instance
 * @param id                                   The compute instance ID
 * @param computeInstance                      Returns compute instance
 *
 * @return
 *         - \ref LWML_SUCCESS                 Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  If \a device, \a ID or \a computeInstance are invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     If \a device doesn't have MIG mode enabled
 *         - \ref LWML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 *         - \ref LWML_ERROR_NOT_FOUND         If the compute instance is not found.
 */
lwmlReturn_t DECLDIR lwmlGpuInstanceGetComputeInstanceById(lwmlGpuInstance_t gpuInstance, unsigned int id,
                                                           lwmlComputeInstance_t *computeInstance);

/**
 * Get compute instance information.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param computeInstance                      The compute instance handle
 * @param info                                 Return compute instance information
 *
 * @return
 *         - \ref LWML_SUCCESS                 Upon success
 *         - \ref LWML_ERROR_UNINITIALIZED     If library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  If \a computeInstance or \a info are invalid
 *         - \ref LWML_ERROR_NO_PERMISSION     If user doesn't have permission to perform the operation
 */
lwmlReturn_t DECLDIR lwmlComputeInstanceGetInfo_v2(lwmlComputeInstance_t computeInstance, lwmlComputeInstanceInfo_t *info);

/**
 * Test if the given handle refers to a MIG device.
 *
 * A MIG device handle is an LWML abstraction which maps to a MIG compute instance.
 * These overloaded references can be used (with some restrictions) interchangeably
 * with a GPU device handle to execute queries at a per-compute instance granularity.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               LWML handle to test
 * @param isMigDevice                          True when handle refers to a MIG device
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a device status was successfully retrieved
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device handle or \a isMigDevice reference is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this check is not supported by the device
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceIsMigDeviceHandle(lwmlDevice_t device, unsigned int *isMigDevice);

/**
 * Get GPU instance ID for the given MIG device handle.
 *
 * GPU instance IDs are unique per device and remain valid until the GPU instance is destroyed.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               Target MIG device handle
 * @param id                                   GPU instance ID
 *
 * @return
 *         - \ref LWML_SUCCESS                 if instance ID was successfully retrieved
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device or \a id reference is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetGpuInstanceId(lwmlDevice_t device, unsigned int *id);

/**
 * Get compute instance ID for the given MIG device handle.
 *
 * Compute instance IDs are unique per GPU instance and remain valid until the compute instance
 * is destroyed.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               Target MIG device handle
 * @param id                                   Compute instance ID
 *
 * @return
 *         - \ref LWML_SUCCESS                 if instance ID was successfully retrieved
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device or \a id reference is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetComputeInstanceId(lwmlDevice_t device, unsigned int *id);

/**
 * Get the maximum number of MIG devices that can exist under a given parent LWML device.
 *
 * Returns zero if MIG is not supported or enabled.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               Target device handle
 * @param count                                Count of MIG devices
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a count was successfully retrieved
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device or \a count reference is invalid
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetMaxMigDeviceCount(lwmlDevice_t device, unsigned int *count);

/**
 * Get MIG device handle for the given index under its parent LWML device.
 *
 * If the compute instance is destroyed either explicitly or by destroying,
 * resetting or unbinding the parent GPU instance or the GPU device itself
 * the MIG device handle would remain invalid and must be requested again
 * using this API. Handles may be reused and their properties can change in
 * the process.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param device                               Reference to the parent GPU device handle
 * @param index                                Index of the MIG device
 * @param migDevice                            Reference to the MIG device handle
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a migDevice handle was successfully created
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device, \a index or \a migDevice reference is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref LWML_ERROR_NOT_FOUND         if no valid MIG device was found at \a index
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetMigDeviceHandleByIndex(lwmlDevice_t device, unsigned int index,
                                                         lwmlDevice_t *migDevice);

/**
 * Get parent device handle from a MIG device handle.
 *
 * For Ampere &tm; or newer fully supported devices.
 * Supported on Linux only.
 *
 * @param migDevice                            MIG device handle
 * @param device                               Device handle
 *
 * @return
 *         - \ref LWML_SUCCESS                 if \a device handle was successfully created
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a migDevice or \a device is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if this query is not supported by the device
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceGetDeviceHandleFromMigDeviceHandle(lwmlDevice_t migDevice, lwmlDevice_t *device);

/** @} */

/**
 * LWML API versioning support
 */

#ifdef LWML_NO_ULWERSIONED_FUNC_DEFS
lwmlReturn_t DECLDIR lwmlInit(void);
lwmlReturn_t DECLDIR lwmlDeviceGetCount(unsigned int *deviceCount);
lwmlReturn_t DECLDIR lwmlDeviceGetHandleByIndex(unsigned int index, lwmlDevice_t *device);
lwmlReturn_t DECLDIR lwmlDeviceGetHandleByPciBusId(const char *pciBusId, lwmlDevice_t *device);
lwmlReturn_t DECLDIR lwmlDeviceGetPciInfo(lwmlDevice_t device, lwmlPciInfo_t *pci);
lwmlReturn_t DECLDIR lwmlDeviceGetPciInfo_v2(lwmlDevice_t device, lwmlPciInfo_t *pci);
lwmlReturn_t DECLDIR lwmlDeviceGetLwLinkRemotePciInfo(lwmlDevice_t device, unsigned int link, lwmlPciInfo_t *pci);
lwmlReturn_t DECLDIR lwmlDeviceGetGridLicensableFeatures(lwmlDevice_t device, lwmlGridLicensableFeatures_t *pGridLicensableFeatures);
lwmlReturn_t DECLDIR lwmlDeviceGetGridLicensableFeatures_v2(lwmlDevice_t device, lwmlGridLicensableFeatures_t *pGridLicensableFeatures);
lwmlReturn_t DECLDIR lwmlDeviceRemoveGpu(lwmlPciInfo_t *pciInfo);
lwmlReturn_t DECLDIR lwmlEventSetWait(lwmlEventSet_t set, lwmlEventData_t * data, unsigned int timeoutms);
lwmlReturn_t DECLDIR lwmlDeviceGetAttributes(lwmlDevice_t device, lwmlDeviceAttributes_t *attributes);
lwmlReturn_t DECLDIR lwmlComputeInstanceGetInfo(lwmlComputeInstance_t computeInstance, lwmlComputeInstanceInfo_t *info);
lwmlReturn_t DECLDIR lwmlDeviceGetComputeRunningProcesses(lwmlDevice_t device, unsigned int *infoCount, lwmlProcessInfo_t *infos);
lwmlReturn_t DECLDIR lwmlDeviceGetGraphicsRunningProcesses(lwmlDevice_t device, unsigned int *infoCount, lwmlProcessInfo_t *infos);
#endif // #ifdef LWML_NO_ULWERSIONED_FUNC_DEFS

#if defined(LWML_NO_ULWERSIONED_FUNC_DEFS)
// We don't define APIs to run new versions if this guard is present so there is
// no need to undef
#elif defined(__LWML_API_VERSION_INTERNAL)
#undef lwmlDeviceGetGraphicsRunningProcesses
#undef lwmlDeviceGetComputeRunningProcesses
#undef lwmlDeviceGetAttributes
#undef lwmlComputeInstanceGetInfo
#undef lwmlEventSetWait
#undef lwmlDeviceGetGridLicensableFeatures
#undef lwmlDeviceRemoveGpu
#undef lwmlDeviceGetLwLinkRemotePciInfo
#undef lwmlDeviceGetPciInfo
#undef lwmlDeviceGetCount
#undef lwmlDeviceGetHandleByIndex
#undef lwmlDeviceGetHandleByPciBusId
#undef lwmlInit
#undef lwmlBlacklistDeviceInfo_t
#undef lwmlGetBlacklistDeviceCount
#undef lwmlGetBlacklistDeviceInfoByIndex
#endif

#ifdef __cplusplus
}
#endif

#endif
