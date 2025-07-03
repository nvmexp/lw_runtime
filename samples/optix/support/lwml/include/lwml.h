/*
 * Copyright 1993-2016 LWPU Corporation.  All rights reserved.
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
    - All GRID products, starting with the Kepler architecture
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
#define LWML_API_VERSION            8
#define LWML_API_VERSION_STR        "8"
#define lwmlInit                    lwmlInit_v2
#define lwmlDeviceGetPciInfo        lwmlDeviceGetPciInfo_v2
#define lwmlDeviceGetCount          lwmlDeviceGetCount_v2
#define lwmlDeviceGetHandleByIndex  lwmlDeviceGetHandleByIndex_v2
#define lwmlDeviceGetHandleByPciBusId lwmlDeviceGetHandleByPciBusId_v2

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
#define LWML_DEVICE_PCI_BUS_ID_BUFFER_SIZE   16

/**
 * PCI information about a GPU device.
 */
typedef struct lwmlPciInfo_st 
{
    char busId[LWML_DEVICE_PCI_BUS_ID_BUFFER_SIZE]; //!< The tuple domain:bus:device.function PCI identifier (&amp; NULL terminator)
    unsigned int domain;             //!< The PCI domain on which the device's bus resides, 0 to 0xffff
    unsigned int bus;                //!< The bus on which the device resides, 0 to 0xff
    unsigned int device;             //!< The device's id on the bus, 0 to 31
    unsigned int pciDeviceId;        //!< The combined 16-bit device id and 16-bit vendor id
    
    // Added in LWML 2.285 API
    unsigned int pciSubSystemId;     //!< The 32-bit Sub System Device ID
    
    // LWPU reserved for internal use only
    unsigned int reserved0;
    unsigned int reserved1;
    unsigned int reserved2;
    unsigned int reserved3;
} lwmlPciInfo_t;

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
    unsigned int pid;                 //!< Process ID
    unsigned long long usedGpuMemory; //!< Amount of used GPU memory in bytes.
                                      //! Under WDDM, \ref LWML_VALUE_NOT_AVAILABLE is always reported
                                      //! because Windows KMD manages all the memory and not the LWPU driver
} lwmlProcessInfo_t;


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
#define LWML_LWLINK_MAX_LINKS 4

/**
 * Enum to represent the LwLink utilization counter packet units
 */
typedef enum lwmlLwLinkUtilizationCountUnits_enum
{
    LWML_LWLINK_COUNTER_UNIT_CYCLES =  0,     // count by cycles
    LWML_LWLINK_COUNTER_UNIT_PACKETS = 1,     // count by packets
    LWML_LWLINK_COUNTER_UNIT_BYTES   = 2,     // count by bytes

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
    LWML_LWLINK_ERROR_DL_REPLAY   = 0,     // Data link replay error counter
    LWML_LWLINK_ERROR_DL_RECOVERY = 1,     // Data link recovery error counter

    // this must be last
    LWML_LWLINK_ERROR_COUNT
} lwmlLwLinkErrorCounter_t;

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
    LWML_TOPOLOGY_CPU                = 40, // all devices that are connected to the same CPU but possibly multiple host bridges
    LWML_TOPOLOGY_SYSTEM             = 50, // all devices in the system

    // there is purposefully no COUNT here because of the need for spacing above
} lwmlGpuTopologyLevel_t;


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
    LWML_PERF_POLICY_POWER = 0,
    LWML_PERF_POLICY_THERMAL = 1,
    LWML_PERF_POLICY_SYNC_BOOST = 2,

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
    LWML_BRAND_UNKNOWN = 0, 
    LWML_BRAND_QUADRO  = 1,
    LWML_BRAND_TESLA   = 2,
    LWML_BRAND_LWS     = 3,
    LWML_BRAND_GRID    = 4,
    LWML_BRAND_GEFORCE = 5,

    // Keep this last
    LWML_BRAND_COUNT
} lwmlBrandType_t;

/**
 * Temperature thresholds.
 */
typedef enum lwmlTemperatureThresholds_enum
{
    LWML_TEMPERATURE_THRESHOLD_SHUTDOWN = 0,    // Temperature at which the GPU will shut down
                                                // for HW protection
    LWML_TEMPERATURE_THRESHOLD_SLOWDOWN = 1,    // Temperature at which the GPU will begin slowdown
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
    LWML_CLOCK_COUNT //<! Count of clock types
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
    LWML_CLOCK_ID_COUNT //<! Count of Clock Ids.
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
    LWML_SUCCESS = 0,                   //!< The operation was successful
    LWML_ERROR_UNINITIALIZED = 1,       //!< LWML was not first initialized with lwmlInit()
    LWML_ERROR_ILWALID_ARGUMENT = 2,    //!< A supplied argument is invalid
    LWML_ERROR_NOT_SUPPORTED = 3,       //!< The requested operation is not available on target device
    LWML_ERROR_NO_PERMISSION = 4,       //!< The current user does not have permission for operation
    LWML_ERROR_ALREADY_INITIALIZED = 5, //!< Deprecated: Multiple initializations are now allowed through ref counting
    LWML_ERROR_NOT_FOUND = 6,           //!< A query to find an object was unsuccessful
    LWML_ERROR_INSUFFICIENT_SIZE = 7,   //!< An input argument is not large enough
    LWML_ERROR_INSUFFICIENT_POWER = 8,  //!< A device's external power cables are not properly attached
    LWML_ERROR_DRIVER_NOT_LOADED = 9,   //!< LWPU driver is not loaded
    LWML_ERROR_TIMEOUT = 10,            //!< User provided timeout passed
    LWML_ERROR_IRQ_ISSUE = 11,          //!< LWPU Kernel detected an interrupt issue with a GPU
    LWML_ERROR_LIBRARY_NOT_FOUND = 12,  //!< LWML Shared Library couldn't be found or loaded
    LWML_ERROR_FUNCTION_NOT_FOUND = 13, //!< Local version of LWML doesn't implement this function
    LWML_ERROR_CORRUPTED_INFOROM = 14,  //!< infoROM is corrupted
    LWML_ERROR_GPU_IS_LOST = 15,        //!< The GPU has fallen off the bus or has otherwise become inaccessible
    LWML_ERROR_RESET_REQUIRED = 16,     //!< The GPU requires a reset before it can be used again
    LWML_ERROR_OPERATING_SYSTEM = 17,   //!< The GPU control device has been blocked by the operating system/cgroups
    LWML_ERROR_LIB_RM_VERSION_MISMATCH = 18,   //!< RM detects a driver/library version mismatch
    LWML_ERROR_IN_USE = 19,             //!< An operation cannot be performed because the GPU is lwrrently in use
    LWML_ERROR_UNKNOWN = 999            //!< An internal driver error oclwrred
} lwmlReturn_t;

/**
 * Memory locations
 *
 * See \ref lwmlDeviceGetMemoryErrorCounter
 */
typedef enum lwmlMemoryLocation_enum
{
    LWML_MEMORY_LOCATION_L1_CACHE = 0,       //!< GPU L1 Cache
    LWML_MEMORY_LOCATION_L2_CACHE = 1,       //!< GPU L2 Cache
    LWML_MEMORY_LOCATION_DEVICE_MEMORY = 2,  //!< GPU Device Memory
    LWML_MEMORY_LOCATION_REGISTER_FILE = 3,  //!< GPU Register File
    LWML_MEMORY_LOCATION_TEXTURE_MEMORY = 4, //!< GPU Texture Memory
    
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
    LWML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS = 1,  //!< APIs that enable/disable auto boosted clocks
                                                      //!< see lwmlDeviceSetAutoBoostedClocksEnabled
    // Keep this last
    LWML_RESTRICTED_API_COUNT
} lwmlRestrictedAPI_t;

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

//! Mask with no events
#define lwmlEventTypeNone                  0x0000000000000000LL
//! Mask of all events
#define lwmlEventTypeAll (lwmlEventTypeNone    \
        | lwmlEventTypeSingleBitEccError       \
        | lwmlEventTypeDoubleBitEccError       \
        | lwmlEventTypePState                  \
        | lwmlEventTypeClock                   \
        | lwmlEventTypeXidCriticalError        \
        )
/** @} */

/** 
 * Information about oclwrred event
 */
typedef struct lwmlEventData_st
{
    lwmlDevice_t        device;         //!< Specific device where the event oclwrred
    unsigned long long  eventType;      //!< Information about what specific event oclwrred
    unsigned long long  eventData;      //!< Stores last XID error for the device in the event of lwmlEventTypeXidCriticalError, 
                                        //  eventData is 0 for any other event. eventData is set as 999 for unknown xid error.
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
#define lwmlClocksThrottleReasonApplicationsClocksSetting   0x0000000000000002LL

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

/** Some other unspecified factor is reducing the clocks */
#define lwmlClocksThrottleReasonUnknown                   0x8000000000000000LL

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
      | lwmlClocksThrottleReasonUnknown                           \
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
/** @defgroup lwmlInitializationAndCleanup Initialization and Cleanup
 * This chapter describes the methods that handle LWML initialization and cleanup.
 * It is the user's responsibility to call \ref lwmlInit() before calling any other methods, and 
 * lwmlShutdown() once LWML is no longer being used.
 *  @{
 */
/***************************************************************************************************/

/**
 * Initialize LWML, but don't initialize any GPUs yet.
 *
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
lwmlReturn_t DECLDIR lwmlInit(void);

/**
 * Shut down LWML by releasing all GPU resources previously allocated with \ref lwmlInit().
 * 
 * For all products.
 *
 * This method should be called after LWML work is done, once for each call to \ref lwmlInit()
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
 * Buffer size guaranteed to be large enough for \ref lwmlDeviceGetUUID
 */
#define LWML_DEVICE_UUID_BUFFER_SIZE                  80

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
 * Buffer size guaranteed to be large enough for \ref lwmlDeviceGetName
 */
#define LWML_DEVICE_NAME_BUFFER_SIZE                  64

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
 * calling one of \ref lwmlDeviceGetHandleByIndex(), \ref lwmlDeviceGetHandleBySerial(),
 * \ref lwmlDeviceGetHandleByPciBusId(). or \ref lwmlDeviceGetHandleByUUID(). 
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
lwmlReturn_t DECLDIR lwmlDeviceGetCount(unsigned int *deviceCount);

/**
 * Acquire the handle for a particular device, based on its index.
 * 
 * For all products.
 *
 * Valid indices are derived from the \a accessibleDevices count returned by 
 *   \ref lwmlDeviceGetCount(). For example, if \a accessibleDevices is 2 the valid indices  
 *   are 0 and 1, corresponding to GPU 0 and GPU 1.
 *
 * The order in which LWML enumerates devices has no guarantees of consistency between reboots. For that reason it
 *   is recommended that devices be looked up by their PCI ids or UUID. See 
 *   \ref lwmlDeviceGetHandleByUUID() and \ref lwmlDeviceGetHandleByPciBusId().
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
lwmlReturn_t DECLDIR lwmlDeviceGetHandleByIndex(unsigned int index, lwmlDevice_t *device);

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
 * @param uuid                                 The UUID of the target GPU
 * @param device                               Reference in which to return the device handle
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
 * This value corresponds to the lwmlPciInfo_t::busId returned by \ref lwmlDeviceGetPciInfo().
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
lwmlReturn_t DECLDIR lwmlDeviceGetHandleByPciBusId(const char *pciBusId, lwmlDevice_t *device);

/**
 * Retrieves the name of this device. 
 * 
 * For all products.
 *
 * The name is an alphanumeric string that denotes a particular product, e.g. Tesla &tm; C2070. It will not
 * exceed 64 characters in length (including the NULL terminator).  See \ref
 * lwmlConstants::LWML_DEVICE_NAME_BUFFER_SIZE.
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
 *   \ref lwmlDeviceGetCount(). For example, if \a accessibleDevices is 2 the valid indices  
 *   are 0 and 1, corresponding to GPU 0 and GPU 1.
 *
 * The order in which LWML enumerates devices has no guarantees of consistency between reboots. For that reason it
 *   is recommended that devices be looked up by their PCI ids or GPU UUID. See 
 *   \ref lwmlDeviceGetHandleByPciBusId() and \ref lwmlDeviceGetHandleByUUID().
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

/**
 * Retrieves an array of unsigned ints (sized to cpuSetSize) of bitmasks with the ideal CPU affinity for the device
 * For example, if processors 0, 1, 32, and 33 are ideal for the device and cpuSetSize == 2,
 *     result[0] = 0x3, result[1] = 0x3
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
 * Sets the ideal affinity for a device using the guidelines given in lwmlDeviceGetCpuAffinity()
 * Lwrrently supports up to 64 processors.
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
 * Clear all affinity bindings
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
 * Retrieves the globally unique immutable UUID associated with this device, as a 5 part hexadecimal string,
 * that augments the immutable, board serial identifier.
 *
 * For all products.
 *
 * The UUID is a globally unique identifier. It is the only available identifier for pre-Fermi-architecture products.
 * It does NOT correspond to any identifier printed on the board.  It will not exceed 80 characters in length
 * (including the NULL terminator).  See \ref lwmlConstants::LWML_DEVICE_UUID_BUFFER_SIZE.
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
lwmlReturn_t DECLDIR lwmlDeviceGetPciInfo(lwmlDevice_t device, lwmlPciInfo_t *pci);

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
 * This method is not supported on virtualized GPU elwironments.
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
 *         - \ref LWML_SUCCESS                 if \a value and \a rollover have been set
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, or \a value or \a rollover are NULL
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
 * For newer than Maxwell &tm; fully supported devices.
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
 * Retrieve the current state of auto boosted clocks on a device and store it in \a isEnabled
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * Auto boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates
 * to maximize performance as thermal limits allow.
 *
 * @param device                               The identifier of the target device
 * @param isEnabled                            Where to store the current state of auto boosted clocks of the target device
 * @param defaultIsEnabled                     Where to store the default auto boosted clocks behavior of the target device that the device will
 *                                                 revert to when no applications are using the GPU
 *
 * @return
 *         - \ref LWML_SUCCESS                 If \a isEnabled has been been set with the auto boosted clocks state of \a device
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid or \a isEnabled is NULL
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support auto boosted clocks
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 */
lwmlReturn_t DECLDIR lwmlDeviceGetAutoBoostedClocksEnabled(lwmlDevice_t device, lwmlEnableState_t *isEnabled, lwmlEnableState_t *defaultIsEnabled);

/**
 * Try to set the current state of auto boosted clocks on a device.
 *
 * For Kepler &tm; or newer fully supported devices.
 *
 * Auto boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates
 * to maximize performance as thermal limits allow. Auto boosted clocks should be disabled if fixed clock
 * rates are desired.
 * Non-root users may use this API by default but can be restricted by root from using this API by calling
 * \ref lwmlDeviceSetAPIRestriction with apiType=LWML_RESTRICTED_API_SET_AUTO_BOOSTED_CLOCKS.
 * Note: Persistence Mode is required to modify current Auto boost settings, therefore, it must be enabled.
 *
 * @param device                               The identifier of the target device
 * @param enabled                              What state to try to set auto boosted clocks of the target device to
 *
 * @return
 *         - \ref LWML_SUCCESS                 If the auto boosted clocks were successfully set to the state specified by \a enabled
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support auto boosted clocks
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 */
lwmlReturn_t DECLDIR lwmlDeviceSetAutoBoostedClocksEnabled(lwmlDevice_t device, lwmlEnableState_t enabled);

/**
 * Try to set the default state of auto boosted clocks on a device. This is the default state that auto boosted clocks will
 * return to when no compute running processes (e.g. LWCA application which have an active context) are running
 *
 * For Kepler &tm; or newer non-VdChip fully supported devices and Maxwell or newer VdChip devices.
 * Requires root/admin permissions.
 *
 * Auto boosted clocks are enabled by default on some hardware, allowing the GPU to run at higher clock rates
 * to maximize performance as thermal limits allow. Auto boosted clocks should be disabled if fixed clock
 * rates are desired.
 *
 * @param device                               The identifier of the target device
 * @param enabled                              What state to try to set default auto boosted clocks of the target device to
 * @param flags                                Flags that change the default behavior. Lwrrently Unused.
 *
 * @return
 *         - \ref LWML_SUCCESS                 If the auto boosted clock's default state was successfully set to the state specified by \a enabled
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_NO_PERMISSION     If the calling user does not have permission to change auto boosted clock's default state.
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device does not support auto boosted clocks
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
 * The fan speed is expressed as a percent of the maximum, i.e. full speed is 100%.
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
 * This method is not supported on virtualized GPU elwironments.
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
 * @param device                               The identifier of the target device
 * @param memory                               Reference in which to return the memory information
 * 
 * @return 
 *         - \ref LWML_SUCCESS                 if \a memory has been populated
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
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
 * Retrieves the current utilization and sampling size in microseconds for the Decoder
 *
 * For Kepler &tm; or newer fully supported devices.
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
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, either of \a infoCount or \a infos is NULL
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see \ref lwmlSystemGetProcessName
 */
lwmlReturn_t DECLDIR lwmlDeviceGetComputeRunningProcesses(lwmlDevice_t device, unsigned int *infoCount, lwmlProcessInfo_t *infos);

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
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a device is invalid, either of \a infoCount or \a infos is NULL
 *         - \ref LWML_ERROR_GPU_IS_LOST       if the target GPU has fallen off the bus or is otherwise inaccessible
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 *
 * @see \ref lwmlSystemGetProcessName
 */
lwmlReturn_t DECLDIR lwmlDeviceGetGraphicsRunningProcesses(lwmlDevice_t device, unsigned int *infoCount, lwmlProcessInfo_t *infos);

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
 *                                                 the feature that is being queried (E.G. Enabling/disabling auto boosted clocks is
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
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature or accounting mode is disabled
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
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature or accounting mode is disabled
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
 * For newer than Maxwell &tm; fully supported devices.
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
 * For newer than Maxwell &tm; fully supported devices.
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
 * For newer than Maxwell &tm; fully supported devices.
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
 * For newer than Maxwell &tm; fully supported devices.
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
lwmlReturn_t DECLDIR lwmlDeviceGetLwLinkRemotePciInfo(lwmlDevice_t device, unsigned int link, lwmlPciInfo_t *pci);

/**
 * Retrieves the specified error counter value
 * Please refer to \a lwmlLwLinkErrorCounter_t for error counters that are available
 *
 * For newer than Maxwell &tm; fully supported devices.
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
 * Set the LWLINK utilization counter control information for the specified counter, 0 or 1.
 * Please refer to \a lwmlLwLinkUtilizationControl_t for the structure definition.  Performs a reset
 * of the counters if the reset parameter is non-zero.
 *
 * For newer than Maxwell &tm; fully supported devices.
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
 * Get the LWLINK utilization counter control information for the specified counter, 0 or 1.
 * Please refer to \a lwmlLwLinkUtilizationControl_t for the structure definition
 *
 * For newer than Maxwell &tm; fully supported devices.
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
 * Retrieve the LWLINK utilization counter based on the current control for a specified counter.
 * In general it is good practice to use \a lwmlDeviceSetLwLinkUtilizationControl
 *  before reading the utilization counters as they have no default state
 *
 * For newer than Maxwell &tm; fully supported devices.
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
 * Freeze the LWLINK utilization counters 
 * Both the receive and transmit counters are operated on by this function
 *
 * For newer than Maxwell &tm; fully supported devices.
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
 * Reset the LWLINK utilization counters 
 * Both the receive and transmit counters are operated on by this function
 *
 * For newer than Maxwell &tm; fully supported devices.
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
 * Checking if some event oclwrred can be done with \ref lwmlEventSetWait
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
 * In case of xid error, the function returns the most recent xid error type seen by the system. If there are multiple
 * xid errors generated before lwmlEventSetWait is ilwoked then the last seen xid error type is returned for all
 * xid error events.
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
lwmlReturn_t DECLDIR lwmlEventSetWait(lwmlEventSet_t set, lwmlEventData_t * data, unsigned int timeoutms);

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
 * Any new LWML process will see a gap in the enumeration where this GPU should exist as any call to that
 * GPU outside of the drain state APIs will fail.
 * Must be called as administrator.
 * For Linux only.
 * 
 * For newer than Maxwell &tm; fully supported devices.
 * Some Kepler devices supported.
 *
 * @param lwmlIndex                            The ID of the target device
 * @param newState                             The drain state that should be entered, see \ref lwmlEnableState_t
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if counters were successfully reset
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a lwmlIndex or \a newState is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_NO_PERMISSION     if the calling process has insufficient permissions to perform operation
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceModifyDrainState (unsigned int lwmlIndex, lwmlEnableState_t newState);

/**
 * Query the drain state of a GPU.  This method is used to check if a GPU is in a lwrrently draining
 * state.
 * For Linux only.
 * 
 * For newer than Maxwell &tm; fully supported devices.
 * Some Kepler devices supported.
 *
 * @param lwmlIndex                            The ID of the target device
 * @param lwrrentState                         The current drain state for this GPU, see \ref lwmlEnableState_t
 *
 * @return 
 *         - \ref LWML_SUCCESS                 if counters were successfully reset
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a lwmlIndex or \a lwrrentState is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_UNKNOWN           on any unexpected error
 */
lwmlReturn_t DECLDIR lwmlDeviceQueryDrainState (unsigned int lwmlIndex, lwmlEnableState_t *lwrrentState);

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
 * For newer than Maxwell &tm; fully supported devices.
 * Some Kepler devices supported.
 *
 * @param lwmlIndex                            The ID of the target device
 *
 * @return
 *         - \ref LWML_SUCCESS                 if counters were successfully reset
 *         - \ref LWML_ERROR_UNINITIALIZED     if the library has not been successfully initialized
 *         - \ref LWML_ERROR_ILWALID_ARGUMENT  if \a lwmlIndex is invalid
 *         - \ref LWML_ERROR_NOT_SUPPORTED     if the device doesn't support this feature
 *         - \ref LWML_ERROR_IN_USE            if the device is still in use and cannot be removed
 */
lwmlReturn_t DECLDIR lwmlDeviceRemoveGpu (unsigned int lwmlIndex);

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
 * For newer than Maxwell &tm; fully supported devices.
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

/**
 * LWML API versioning support
 */
#if defined(__LWML_API_VERSION_INTERNAL)
#undef lwmlDeviceGetPciInfo
#undef lwmlDeviceGetCount
#undef lwmlDeviceGetHandleByIndex
#undef lwmlDeviceGetHandleByPciBusId
#undef lwmlInit
#endif

#ifdef __cplusplus
}
#endif

#endif
