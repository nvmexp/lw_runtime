/*
 * Copyright 1993-2018 LWPU Corporation.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 */

#ifndef DCGMFIELDS_H
#define DCGMFIELDS_H

#ifdef __cplusplus
extern "C" {
#endif
    
/***************************************************************************************************/
/** @defgroup dcgmFieldTypes Field Types
 *  Field Types are a single byte.
 *  @{
 */
/***************************************************************************************************/        

/**
 * Blob of binary data representing a structure
 */
#define DCGM_FT_BINARY    'b'
    
/**
 * 8-byte double precision
 */
#define DCGM_FT_DOUBLE    'd'
    
/**
 * 8-byte signed integer
 */
#define DCGM_FT_INT64     'i'
    
/**
 * Null-terminated ASCII Character string
 */
#define DCGM_FT_STRING    's'
    
/**
 * 8-byte signed integer usec since 1970
 */
#define DCGM_FT_TIMESTAMP 't'
    
/** @} */    
    

/***************************************************************************************************/
/** @defgroup dcgmFieldScope Field Scope
 *  Represents field association with entity scope or global scope.
 *  @{
 */
/***************************************************************************************************/     

/**
 * Field is global (ex: driver version)
 */
#define DCGM_FS_GLOBAL  0

/**
 * Field is associated with an entity (GPU, VGPU...etc)
 */
#define DCGM_FS_ENTITY  1

/**
 * Field is associated with a device. Deprecated. Use DCGM_FS_ENTITY
 */
#define DCGM_FS_DEVICE  DCGM_FS_ENTITY

/**
 * DCGM_FI_DEV_LWDA_COMPUTE_CAPABILITY is 16 bits of major version followed by
 * 16 bits of the minor version. These macros separate the two.
 */
#define DCGM_LWDA_COMPUTE_CAPABILITY_MAJOR(x) ((uint64_t)(x) & 0xFFFF0000)
#define DCGM_LWDA_COMPUTE_CAPABILITY_MINOR(x) ((uint64_t)(x) & 0x0000FFFF)

/**
 * DCGM_FI_DEV_CLOCK_THROTTLE_REASONS is a bitmap of why the clock is throttled.
 * These macros are masks for relevant throttling, and are a 1:1 map to the LWML
 * reasons dolwmented in lwml.h. The notes for the header are copied blow:
 */
/** Nothing is running on the GPU and the clocks are dropping to Idle state
 * \note This limiter may be removed in a later release
 */
#define DCGM_CLOCKS_THROTTLE_REASON_GPU_IDLE        0x0000000000000001LL
/** GPU clocks are limited by current setting of applications clocks
 */
#define DCGM_CLOCKS_THROTTLE_REASON_CLOCKS_SETTING  0x0000000000000002LL
/** SW Power Scaling algorithm is reducing the clocks below requested clocks 
 */
#define DCGM_CLOCKS_THROTTLE_REASON_SW_POWER_CAP    0x0000000000000004LL
/** HW Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
 *
 *This is an indicator of:
 * - temperature being too high
 * - External Power Brake Assertion is triggered (e.g. by the system power supply)
 * - Power draw is too high and Fast Trigger protection is reducing the clocks
 * - May be also reported during PState or clock change
 * - This behavior may be removed in a later release.
 */
#define DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN     0x0000000000000008LL
/** Sync Boost
 *
 * This GPU has been added to a Sync boost group with lwpu-smi or DCGM in
 * order to maximize performance per watt. All GPUs in the sync boost group
 * will boost to the minimum possible clocks across the entire group. Look at
 * the throttle reasons for other GPUs in the system to see why those GPUs are
 * holding this one at lower clocks.
 */
#define DCGM_CLOCKS_THROTTLE_REASON_SYNC_BOOST      0x0000000000000010LL
/** SW Thermal Slowdown
 *
 * This is an indicator of one or more of the following:
 *  - Current GPU temperature above the GPU Max Operating Temperature
 *  - Current memory temperature above the Memory Max Operating Temperature
 */
#define DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL      0x0000000000000020LL
/** HW Thermal Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
 *
 * This is an indicator of:
 *  - temperature being too high
 */
#define DCGM_CLOCKS_THROTTLE_REASON_HW_THERMAL      0x0000000000000040LL
/** HW Power Brake Slowdown (reducing the core clocks by a factor of 2 or more) is engaged
 *
 * This is an indicator of:
 *  - External Power Brake Assertion being triggered (e.g. by the system power supply)
 */
#define DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE  0x0000000000000080LL
/** GPU clocks are limited by current setting of Display clocks
 */
#define DCGM_CLOCKS_THROTTLE_REASON_DISPLAY_CLOCKS  0x0000000000000100LL

/** @} */

/***************************************************************************************************/
/** @defgroup dcgmFieldEntity Field Entity
 *  Represents field association with a particular entity
 *  @{
 */
/***************************************************************************************************/

/**
 * Enum of possible field entity groups
 */
typedef enum dcgm_field_entity_group_t
{
    DCGM_FE_NONE = 0, /** Field is not associated with an entity. Field scope should be DCGM_FS_GLOBAL */
    DCGM_FE_GPU,      /** Field is associated with a GPU entity */
    DCGM_FE_VGPU,     /** Field is associated with a VGPU entity */
    DCGM_FE_SWITCH,   /** Field is associated with a Switch entity */

    DCGM_FE_COUNT     /** Number of elements in this enumeration. Keep this entry last */
} dcgm_field_entity_group_t;

/**
 * Represents an identifier for an entity within a field entity. For instance, this is the gpuId for DCGM_FE_GPU.
 */
typedef unsigned int dcgm_field_eid_t;

/** @} */

/***************************************************************************************************/
/** @defgroup dcgmFieldIdentifiers Field Identifiers
 *  Field Identifiers
 *  @{
 */
/***************************************************************************************************/
    
/**
 * NULL field
 */    
#define DCGM_FI_UNKNOWN                   0
    
/**
 * Driver Version
 */
#define DCGM_FI_DRIVER_VERSION            1
    
/* Underlying LWML version */
#define DCGM_FI_LWML_VERSION              2
    
/*
 * Process Name
 */
#define DCGM_FI_PROCESS_NAME              3
    
/**
 * Number of Devices on the node
 */    
#define DCGM_FI_DEV_COUNT                 4

/**
 * Name of the GPU device
 */
#define DCGM_FI_DEV_NAME                  50
    
/**
 * Device Brand
 */
#define DCGM_FI_DEV_BRAND                 51
    
/**
 * LWML index of this GPU
 */
#define DCGM_FI_DEV_LWML_INDEX            52

/**
 * Device Serial Number
 */
#define DCGM_FI_DEV_SERIAL                53

/**
 * UUID corresponding to the device
 */
#define DCGM_FI_DEV_UUID                  54

/**
 * Device node minor number /dev/lwpu#
 */
#define DCGM_FI_DEV_MINOR_NUMBER          55

/**
 * OEM inforom version
 */
#define DCGM_FI_DEV_OEM_INFOROM_VER       56

/**
 * PCI attributes for the device
 */
#define DCGM_FI_DEV_PCI_BUSID             57

/**
 * The combined 16-bit device id and 16-bit vendor id
 */
#define DCGM_FI_DEV_PCI_COMBINED_ID       58
    
/**
 * The 32-bit Sub System Device ID
 */
#define DCGM_FI_DEV_PCI_SUBSYS_ID         59

/**
 * Topology of all GPUs on the system via PCI (static)
 */
#define DCGM_FI_GPU_TOPOLOGY_PCI          60

/**
 * Topology of all GPUs on the system via LWLINK (static)
 */
#define DCGM_FI_GPU_TOPOLOGY_LWLINK       61

/**
 * Affinity of all GPUs on the system (static)
 */
#define DCGM_FI_GPU_TOPOLOGY_AFFINITY     62

/**
 * Lwca compute capability for the device.
 * The major version is the upper 32 bits and 
 * the minor version is the lower 32 bits.
 */
#define DCGM_FI_DEV_LWDA_COMPUTE_CAPABILITY 63

/**
 * Compute mode for the device
 */
#define DCGM_FI_DEV_COMPUTE_MODE          65


/**
 * Device CPU affinity. part 1/8 = cpus 0 - 63
 */
#define DCGM_FI_DEV_CPU_AFFINITY_0        70
    
/**
 * Device CPU affinity. part 1/8 = cpus 64 - 127
 */      
#define DCGM_FI_DEV_CPU_AFFINITY_1        71
    
/**
 * Device CPU affinity. part 2/8 = cpus 128 - 191
 */      
#define DCGM_FI_DEV_CPU_AFFINITY_2        72

/**
 * Device CPU affinity. part 3/8 = cpus 192 - 255
 */
#define DCGM_FI_DEV_CPU_AFFINITY_3        73

/**
 * ECC inforom version
 */
#define DCGM_FI_DEV_ECC_INFOROM_VER       80

/**
 * Power management object inforom version
 */
#define DCGM_FI_DEV_POWER_INFOROM_VER     81

/**
 * Inforom image version
 */
#define DCGM_FI_DEV_INFOROM_IMAGE_VER     82

/**
 * Inforom configuration checksum
 */
#define DCGM_FI_DEV_INFOROM_CONFIG_CHECK  83

/**
 * Reads the infoROM from the flash and verifies the checksums
 */
#define DCGM_FI_DEV_INFOROM_CONFIG_VALID  84

/**
 * VBIOS version of the device
 */
#define DCGM_FI_DEV_VBIOS_VERSION         85

/**
 * Total BAR1 of the GPU in MB
 */
#define DCGM_FI_DEV_BAR1_TOTAL            90

/**
 * Sync boost settings on the node
 */
#define DCGM_FI_SYNC_BOOST                91

/**
 * Used BAR1 of the GPU in MB
 */
#define DCGM_FI_DEV_BAR1_USED             92

/**
 * Free BAR1 of the GPU in MB
 */
#define DCGM_FI_DEV_BAR1_FREE             93

/**
 * SM clock for the device
 */
#define DCGM_FI_DEV_SM_CLOCK              100

/**
 * Memory clock for the device
 */
#define DCGM_FI_DEV_MEM_CLOCK             101

/**
 * Video encoder/decoder clock for the device
 */
#define DCGM_FI_DEV_VIDEO_CLOCK           102

/**
 * SM Application clocks
 */
#define DCGM_FI_DEV_APP_SM_CLOCK          110

/**
 * Memory Application clocks
 */
#define DCGM_FI_DEV_APP_MEM_CLOCK         111

/**
 * Current clock throttle reasons (bitmask of DCGM_CLOCKS_THROTTLE_REASON_*)
 */
#define DCGM_FI_DEV_CLOCK_THROTTLE_REASONS 112

/**
 * Maximum supported SM clock for the device
 */
#define DCGM_FI_DEV_MAX_SM_CLOCK          113

/**
 * Maximum supported Memory clock for the device
 */
#define DCGM_FI_DEV_MAX_MEM_CLOCK         114

/**
 * Maximum supported Video encoder/decoder clock for the device
 */
#define DCGM_FI_DEV_MAX_VIDEO_CLOCK       115

/**
 * Auto-boost for the device (1 = enabled. 0 = disabled)
 */
#define DCGM_FI_DEV_AUTOBOOST             120

/**
 * Supported clocks for the device
 */
#define DCGM_FI_DEV_SUPPORTED_CLOCKS      130

/**
 * Memory temperature for the device
 */
#define DCGM_FI_DEV_MEMORY_TEMP           140

/**
 * Current temperature readings for the device, in degrees C
 */
#define DCGM_FI_DEV_GPU_TEMP              150

/**
 * Power usage for the device in Watts
 */
#define DCGM_FI_DEV_POWER_USAGE           155

/**
 * Total energy consumption for the GPU in mJ since the driver was last reloaded
 */
#define DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION 156

/**
 * Slowdown temperature for the device
 */
#define DCGM_FI_DEV_SLOWDOWN_TEMP         158

/**
 * Shutdown temperature for the device
 */
#define DCGM_FI_DEV_SHUTDOWN_TEMP         159

/**
 * Current Power limit for the device
 */
#define DCGM_FI_DEV_POWER_MGMT_LIMIT      160

/**
 * Minimum power management limit for the device
 */
#define DCGM_FI_DEV_POWER_MGMT_LIMIT_MIN  161

/**
 * Maximum power management limit for the device
 */
#define DCGM_FI_DEV_POWER_MGMT_LIMIT_MAX  162

/**
 * Default power management limit for the device
 */
#define DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF  163

/**
 * Effective power limit that the driver enforces after taking into account all limiters
 */
#define DCGM_FI_DEV_ENFORCED_POWER_LIMIT  164

/**
 * Performance state (P-State) 0-15. 0=highest
 */
#define DCGM_FI_DEV_PSTATE                190

/**
 * Fan speed for the device in percent 0-100
 */
#define DCGM_FI_DEV_FAN_SPEED             191

/**
 * PCIe Tx utilization information
 */
#define DCGM_FI_DEV_PCIE_TX_THROUGHPUT    200
    
/**
 * PCIe Rx utilization information
 */    
#define DCGM_FI_DEV_PCIE_RX_THROUGHPUT    201
    
/**
 * PCIe replay counter
 */
#define DCGM_FI_DEV_PCIE_REPLAY_COUNTER   202

/**
 * GPU Utilization
 */
#define DCGM_FI_DEV_GPU_UTIL              203

/**
 * Memory Utilization
 */
#define DCGM_FI_DEV_MEM_COPY_UTIL         204

/**
 * Process accounting stats.
 *
 * This field is only supported when the host engine is running as root unless you
 * enable accounting ahead of time. Accounting mode can be enabled by
 * running "lwpu-smi -am 1" as root on the same node the host engine is running on.
 */
#define DCGM_FI_DEV_ACCOUNTING_DATA       205

/**
 * Encoder Utilization
 */
#define DCGM_FI_DEV_ENC_UTIL              206

/**
 * Decoder Utilization
 */
#define DCGM_FI_DEV_DEC_UTIL              207

/**
 * Memory utilization samples
 */
#define DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES 210

/*
 * SM utilization samples
 */
#define DCGM_FI_DEV_GPU_UTIL_SAMPLES      211

/**
 * Graphics processes running on the GPU.
 */
#define DCGM_FI_DEV_GRAPHICS_PIDS         220

/**
 * Compute processes running on the GPU.
 */
#define DCGM_FI_DEV_COMPUTE_PIDS          221

/**
 * XID errors. The value is the specific XID error
 */
#define DCGM_FI_DEV_XID_ERRORS            230

/**
 * PCIe Max Link Generation
 */
#define DCGM_FI_DEV_PCIE_MAX_LINK_GEN     235

/**
 * PCIe Max Link Width
 */
#define DCGM_FI_DEV_PCIE_MAX_LINK_WIDTH   236

/**
 * PCIe Current Link Generation
 */
#define DCGM_FI_DEV_PCIE_LINK_GEN         237

/**
 * PCIe Current Link Width
 */
#define DCGM_FI_DEV_PCIE_LINK_WIDTH       238

/**
 * Power Violation time in usec
 */
#define DCGM_FI_DEV_POWER_VIOLATION       240

/**
 * Thermal Violation time in usec
 */
#define DCGM_FI_DEV_THERMAL_VIOLATION     241

/**
 * Sync Boost Violation time in usec
 */
#define DCGM_FI_DEV_SYNC_BOOST_VIOLATION  242

/**
 * Board violation limit.
 */
#define DCGM_FI_DEV_BOARD_LIMIT_VIOLATION 243

/**
 *Low utilisation violation limit.
 */
#define DCGM_FI_DEV_LOW_UTIL_VIOLATION    244

/**
 *Reliability violation limit.
 */
#define DCGM_FI_DEV_RELIABILITY_VIOLATION 245

/**
 * App clock violation limit.
 */
#define DCGM_FI_DEV_TOTAL_APP_CLOCKS_VIOLATION 246

/**
 * Base clock violation limit.
 */
#define DCGM_FI_DEV_TOTAL_BASE_CLOCKS_VIOLATION 247

/**
 * Total Frame Buffer of the GPU in MB
 */
#define DCGM_FI_DEV_FB_TOTAL          250

/**
 * Free Frame Buffer in MB
 */
#define DCGM_FI_DEV_FB_FREE           251

/**
 * Used Frame Buffer in MB
 */
#define DCGM_FI_DEV_FB_USED           252

/**
 * Current ECC mode for the device
 */
#define DCGM_FI_DEV_ECC_LWRRENT           300
    
/**
 * Pending ECC mode for the device
 */    
#define DCGM_FI_DEV_ECC_PENDING           301
    
/**
 * Total single bit volatile ECC errors
 */    
#define DCGM_FI_DEV_ECC_SBE_VOL_TOTAL     310
    
/**
 * Total double bit volatile ECC errors
 */        
#define DCGM_FI_DEV_ECC_DBE_VOL_TOTAL     311
    
/**
 * Total single bit aggregate (persistent) ECC errors
 * Note: monotonically increasing
 */    
#define DCGM_FI_DEV_ECC_SBE_AGG_TOTAL     312
    
/**
 * Total double bit aggregate (persistent) ECC errors
 * Note: monotonically increasing
 */    
#define DCGM_FI_DEV_ECC_DBE_AGG_TOTAL     313
    
/**
 * L1 cache single bit volatile ECC errors
 */
#define DCGM_FI_DEV_ECC_SBE_VOL_L1        314
    
/**
 * L1 cache double bit volatile ECC errors
 */
#define DCGM_FI_DEV_ECC_DBE_VOL_L1        315
    
/**
 * L2 cache single bit volatile ECC errors
 */
#define DCGM_FI_DEV_ECC_SBE_VOL_L2        316
    
/**
 * L2 cache double bit volatile ECC errors
 */
#define DCGM_FI_DEV_ECC_DBE_VOL_L2        317
    
/**
 * Device memory single bit volatile ECC errors
 */
#define DCGM_FI_DEV_ECC_SBE_VOL_DEV       318

/**
 * Device memory double bit volatile ECC errors
 */
#define DCGM_FI_DEV_ECC_DBE_VOL_DEV       319
    
/**
 * Register file single bit volatile ECC errors
 */
#define DCGM_FI_DEV_ECC_SBE_VOL_REG       320
    
/**
 * Register file double bit volatile ECC errors
 */    
#define DCGM_FI_DEV_ECC_DBE_VOL_REG       321
    
/**
 * Texture memory single bit volatile ECC errors
 */        
#define DCGM_FI_DEV_ECC_SBE_VOL_TEX       322
    
/**
 * Texture memory double bit volatile ECC errors
 */            
#define DCGM_FI_DEV_ECC_DBE_VOL_TEX       323
    
/**
 * L1 cache single bit aggregate (persistent) ECC errors
 * Note: monotonically increasing
 */       
#define DCGM_FI_DEV_ECC_SBE_AGG_L1        324
    
/**
 * L1 cache double bit aggregate (persistent) ECC errors
 * Note: monotonically increasing
 */           
#define DCGM_FI_DEV_ECC_DBE_AGG_L1        325
    
/**
 * L2 cache single bit aggregate (persistent) ECC errors
 * Note: monotonically increasing
 */               
#define DCGM_FI_DEV_ECC_SBE_AGG_L2        326

/**
 * L2 cache double bit aggregate (persistent) ECC errors
 * Note: monotonically increasing
 */                   
#define DCGM_FI_DEV_ECC_DBE_AGG_L2        327
    
/**
 * Device memory single bit aggregate (persistent) ECC errors
 * Note: monotonically increasing
 */                       
#define DCGM_FI_DEV_ECC_SBE_AGG_DEV       328
    
/**
 * Device memory double bit aggregate (persistent) ECC errors
 * Note: monotonically increasing
 */                       
#define DCGM_FI_DEV_ECC_DBE_AGG_DEV       329
    
/**
 * Register File single bit aggregate (persistent) ECC errors
 * Note: monotonically increasing
 */                           
#define DCGM_FI_DEV_ECC_SBE_AGG_REG       330
    
/**
 * Register File double bit aggregate (persistent) ECC errors
 * Note: monotonically increasing
 */
#define DCGM_FI_DEV_ECC_DBE_AGG_REG       331
    
/**
 * Texture memory single bit aggregate (persistent) ECC errors
 * Note: monotonically increasing
 */
#define DCGM_FI_DEV_ECC_SBE_AGG_TEX       332

/**
 * Texture memory double bit aggregate (persistent) ECC errors
 * Note: monotonically increasing
 */    
#define DCGM_FI_DEV_ECC_DBE_AGG_TEX       333
    
/**
 * Number of retired pages because of single bit errors
 * Note: monotonically increasing
 */
#define DCGM_FI_DEV_RETIRED_SBE           390

/**
 * Number of retired pages because of double bit errors
 * Note: monotonically increasing
 */
#define DCGM_FI_DEV_RETIRED_DBE           391

/**
 * Number of pages pending retirement
 */
#define DCGM_FI_DEV_RETIRED_PENDING       392

/*
* LW Link flow control CRC  Error Counter for Lane 0
*/
#define DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L0        400

/*
* LW Link flow control CRC  Error Counter for Lane 1
*/
#define DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L1        401

/*
* LW Link flow control CRC  Error Counter for Lane 2
*/
#define DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L2        402

/*
* LW Link flow control CRC  Error Counter for Lane 3
*/
#define DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L3        403

/*
* LW Link flow control CRC  Error Counter for Lane 4
*/
#define DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L4        404

/*
* LW Link flow control CRC  Error Counter for Lane 5
*/
#define DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L5        405

/*
* LW Link flow control CRC  Error Counter total for all Lanes
*/
#define DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_TOTAL     409

/*
* LW Link data CRC Error Counter for Lane 0
*/
#define DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L0      410

/*
* LW Link data CRC Error Counter for Lane 1
*/
#define DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L1      411

/*
* LW Link data CRC Error Counter for Lane 2
*/
#define DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L2      412

/*
* LW Link data CRC Error Counter for Lane 3
*/
#define DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L3      413

/*
* LW Link data CRC Error Counter for Lane 4
*/
#define DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L4      414

/*
* LW Link data CRC Error Counter for Lane 5
*/
#define DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L5      415

/*
* LW Link data CRC Error Counter total for all Lanes
*/
#define DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_TOTAL   419

/*
* LW Link Replay Error Counter for Lane 0
*/
#define DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L0          420

/*
* LW Link Replay Error Counter for Lane 1
*/
#define DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L1          421

/*
* LW Link Replay Error Counter for Lane 2
*/
#define DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L2          422

/*
* LW Link Replay Error Counter for Lane 3
*/
#define DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L3          423

/*
* LW Link Replay Error Counter for Lane 4
*/
#define DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L4          424

/*
* LW Link Replay Error Counter for Lane 5
*/
#define DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L5          425

/*
* LW Link Replay Error Counter total for all Lanes
*/
#define DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_TOTAL       429

/*
* LW Link Recovery Error Counter for Lane 0
*/
#define DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L0        430

/*
* LW Link Recovery Error Counter for Lane 1
*/
#define DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L1        431

/*
* LW Link Recovery Error Counter for Lane 2
*/
#define DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L2        432

/*
* LW Link Recovery Error Counter for Lane 3
*/
#define DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L3        433

/*
* LW Link Recovery Error Counter for Lane 4
*/
#define DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L4        434

/*
* LW Link Recovery Error Counter for Lane 5
*/
#define DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L5        435

/*
* LW Link Recovery Error Counter total for all Lanes
*/
#define DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_TOTAL     439

/*
* LW Link Bandwidth Counter for Lane 0
*/
#define DCGM_FI_DEV_LWLINK_BANDWIDTH_L0                   440

/*
* LW Link Bandwidth Counter for Lane 1
*/
#define DCGM_FI_DEV_LWLINK_BANDWIDTH_L1                   441

/*
* LW Link Bandwidth Counter for Lane 2
*/
#define DCGM_FI_DEV_LWLINK_BANDWIDTH_L2                   442

/*
* LW Link Bandwidth Counter for Lane 3
*/
#define DCGM_FI_DEV_LWLINK_BANDWIDTH_L3                   443

/*
* LW Link Bandwidth Counter for Lane 4
*/
#define DCGM_FI_DEV_LWLINK_BANDWIDTH_L4                   444

/*
* LW Link Bandwidth Counter for Lane 5
*/
#define DCGM_FI_DEV_LWLINK_BANDWIDTH_L5                   445

/*
* LW Link Bandwidth Counter total for all Lanes
*/
#define DCGM_FI_DEV_LWLINK_BANDWIDTH_TOTAL                449

/*
* GPU LWLink error information
*/
#define DCGM_FI_DEV_GPU_LWLINK_ERRORS                     450

/*
 * LW Link flow control CRC  Error Counter for Lane 6
 */
#define DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L6        451

/*
 * LW Link flow control CRC  Error Counter for Lane 7
 */
#define DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L7        452

/*
 * LW Link flow control CRC  Error Counter for Lane 8
 */
#define DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L8        453

/*
 * LW Link flow control CRC  Error Counter for Lane 9
 */
#define DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L9        454

/*
 * LW Link flow control CRC  Error Counter for Lane 10
 */
#define DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L10       455

/*
 * LW Link flow control CRC  Error Counter for Lane 11
 */
#define DCGM_FI_DEV_LWLINK_CRC_FLIT_ERROR_COUNT_L11       456

/*
* LW Link data CRC Error Counter for Lane 6
*/
#define DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L6      457

/*
* LW Link data CRC Error Counter for Lane 7
*/
#define DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L7      458

/*
* LW Link data CRC Error Counter for Lane 8
*/
#define DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L8      459

/*
* LW Link data CRC Error Counter for Lane 9
*/
#define DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L9      460

/*
* LW Link data CRC Error Counter for Lane 10
*/
#define DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L10     461

/*
* LW Link data CRC Error Counter for Lane 11
*/
#define DCGM_FI_DEV_LWLINK_CRC_DATA_ERROR_COUNT_L11     462

/*
* LW Link Replay Error Counter for Lane 6
*/
#define DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L6        463

/*
* LW Link Replay Error Counter for Lane 7
*/
#define DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L7        464

/*
* LW Link Replay Error Counter for Lane 8
*/
#define DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L8        465

/*
* LW Link Replay Error Counter for Lane 9
*/
#define DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L9        466

/*
* LW Link Replay Error Counter for Lane 10
*/
#define DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L10       467

/*
* LW Link Replay Error Counter for Lane 11
*/
#define DCGM_FI_DEV_LWLINK_REPLAY_ERROR_COUNT_L11       468

/*
* LW Link Recovery Error Counter for Lane 6
*/
#define DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L6      469

/*
* LW Link Recovery Error Counter for Lane 7
*/
#define DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L7      470

/*
* LW Link Recovery Error Counter for Lane 8
*/
#define DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L8      471

/*
* LW Link Recovery Error Counter for Lane 9
*/
#define DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L9      472

/*
* LW Link Recovery Error Counter for Lane 10
*/
#define DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L10     473

/*
* LW Link Recovery Error Counter for Lane 11
*/
#define DCGM_FI_DEV_LWLINK_RECOVERY_ERROR_COUNT_L11     474

/*
* LW Link Bandwidth Counter for Lane 6
*/
#define DCGM_FI_DEV_LWLINK_BANDWIDTH_L6                 475

/*
* LW Link Bandwidth Counter for Lane 7
*/
#define DCGM_FI_DEV_LWLINK_BANDWIDTH_L7                 476

/*
* LW Link Bandwidth Counter for Lane 8
*/
#define DCGM_FI_DEV_LWLINK_BANDWIDTH_L8                 477

/*
* LW Link Bandwidth Counter for Lane 9
*/
#define DCGM_FI_DEV_LWLINK_BANDWIDTH_L9                 476

/*
* LW Link Bandwidth Counter for Lane 10
*/
#define DCGM_FI_DEV_LWLINK_BANDWIDTH_L10                477

/*
* LW Link Bandwidth Counter for Lane 11
*/
#define DCGM_FI_DEV_LWLINK_BANDWIDTH_L11                478

/**
 * Virtualization Mode corresponding to the GPU
 */
#define DCGM_FI_DEV_VIRTUAL_MODE                          500

/**
 * Includes Count and Static info of vGPU types supported on a device
 */
#define DCGM_FI_DEV_SUPPORTED_TYPE_INFO                   501

/**
 * Includes Count and lwrrently Creatable vGPU types on a device
 */
#define DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS               502

/**
 * Includes Count and lwrrently Active vGPU Instances on a device
 */
#define DCGM_FI_DEV_VGPU_INSTANCE_IDS                     503

/**
 * Utilization values for vGPUs running on the device
 */
#define DCGM_FI_DEV_VGPU_UTILIZATIONS                     504

/**
 * Utilization values for processes running within vGPU VMs using the device
 */
#define DCGM_FI_DEV_VGPU_PER_PROCESS_UTILIZATION          505

/**
 * Current encoder statistics for a given device
 */
#define DCGM_FI_DEV_ENC_STATS                             506

/**
 * Statistics of current active frame buffer capture sessions on a given device
 */
#define DCGM_FI_DEV_FBC_STATS                             507

/**
 * Information about active frame buffer capture sessions on a target device
 */
#define DCGM_FI_DEV_FBC_SESSIONS_INFO                     508
/**
 * VM ID of the vGPU instance
 */
#define DCGM_FI_DEV_VGPU_VM_ID                            520

/**
 * VM name of the vGPU instance
 */
#define DCGM_FI_DEV_VGPU_VM_NAME                          521

/**
 * vGPU type of the vGPU instance
 */
#define DCGM_FI_DEV_VGPU_TYPE                             522

/**
 * UUID of the vGPU instance
 */
#define DCGM_FI_DEV_VGPU_UUID                             523

/**
 * Driver version of the vGPU instance
 */
#define DCGM_FI_DEV_VGPU_DRIVER_VERSION                   524

/**
 * Memory usage of the vGPU instance
 */
#define DCGM_FI_DEV_VGPU_MEMORY_USAGE                     525

/**
 * License status of the vGPU instance
 */
#define DCGM_FI_DEV_VGPU_LICENSE_STATUS                   526

/**
 * Frame rate limit of the vGPU instance
 */
#define DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT                 527

/**
 * Current encoder statistics of the vGPU instance
 */
#define DCGM_FI_DEV_VGPU_ENC_STATS                        528

/**
 * Information about all active encoder sessions on the vGPU instance
 */
#define DCGM_FI_DEV_VGPU_ENC_SESSIONS_INFO                529

/**
 * Statistics of current active frame buffer capture sessions on the vGPU instance
 */
#define DCGM_FI_DEV_VGPU_FBC_STATS                        530

/**
 * Information about active frame buffer capture sessions on the vGPU instance
 */
#define DCGM_FI_DEV_VGPU_FBC_SESSIONS_INFO                531

/**
 * PCI Id of the vGPU instance
 */
#define DCGM_FI_DEV_VGPU_PCI_ID                           532

/**
 * Starting field ID of the vGPU instance
 */
#define DCGM_FI_FIRST_VGPU_FIELD_ID                       520

/**
 * Last field ID of the vGPU instance
 */
#define DCGM_FI_LAST_VGPU_FIELD_ID                        570

/**
 * For now max vGPU field Ids taken as difference of DCGM_FI_LAST_VGPU_FIELD_ID and DCGM_FI_LAST_VGPU_FIELD_ID i.e. 50
 */
#define DCGM_FI_MAX_VGPU_FIELDS     DCGM_FI_LAST_VGPU_FIELD_ID - DCGM_FI_FIRST_VGPU_FIELD_ID

/**
 * Starting ID for all the internal fields
 */
#define DCGM_FI_INTERNAL_FIELDS_0_START                   600

/**
 * Last ID for all the internal fields
 */

/**
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch entity field IDs start here.</p>
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 0</p>
*/

#define DCGM_FI_INTERNAL_FIELDS_0_END                     699


/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P00               700
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P00               701
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P00              702
/** 
* Max latency bin
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 1</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P00               703

/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P01               704
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P01               705
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P01              706
/** 
* Max latency bin
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 2</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P01               707

/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P02               708
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P02               709
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P02              710
/** 
* Max latency bin
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 3</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P02               711

/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P03               712
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P03               713
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P03              714
/** 
* Max latency bin
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 4</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P03               715

/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P04               716
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P04               717
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P04              718
/** 
* Max latency bin
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 5</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P04               719

/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P05               720
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P05               721
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P05              722
/** 
* Max latency bin
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 6</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P05               723

/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P06               724
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P06               725
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P06              726
/** 
* Max latency bin
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 7</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P06               727

/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P07               728
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P07               729
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P07              730
/** 
* Max latency bin
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 8</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P07               731

/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P08               732
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P08               733
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P08              734
/** 
* Max latency bin
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 9</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P08               735

/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P09               736
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P09               737
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P09              738
/** 
* Max latency bin
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 10</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P09               739

/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P10               740
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P10               741
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P10              742
/** 
* Max latency bin
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 11</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P10               743

/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P11               744
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P11               745
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P11              746
/** 
* Max latency bin
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 12</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P11               747

/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P12               748
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P12               749
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P12              750
/** 
* Max latency bin
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 13</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P12               751

/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P13               752
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P13               753
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P13              754
/** 
* Max latency bin
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 14</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P13               755

/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P14               756
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P14               757
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P14              758
/** 
* Max latency bin
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 15</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P14               759

/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P15               760
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P15               761
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P15              762
/** 
* Max latency bin
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 16</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P15               763

/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P16               764
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P16               765
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P16              766
/** 
* Max latency bin
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch latency bins for port 17</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P16               767

/**
* <p>Low latency bin</p>
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_LOW_P17               768
/** 
* Medium latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MED_P17               769
/** 
* High latency bin
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_HIGH_P17              770
/** 
* <p>Max latency bin</p>
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch Tx and Rx Counter 0 for each port</p>
* <p>By default, Counter 0 counts bytes.</p> 
*/
#define DCGM_FI_DEV_LWSWITCH_LATENCY_MAX_P17               771

/**
* <p>LWSwitch Tx Bandwidth Counter 0 for port 0</p>
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P00            780
/**
* LWSwitch Rx Bandwidth Counter 0 for port 0
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P00            781

/**
* LWSwitch Tx Bandwidth Counter 0 for port 1
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P01            782
/**
* LWSwitch Rx Bandwidth Counter 0 for port 1
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P01            783

/**
* LWSwitch Tx Bandwidth Counter 0 for port 2
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P02            784
/**
* LWSwitch Rx Bandwidth Counter 0 for port 2
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P02            785

/**
* LWSwitch Tx Bandwidth Counter 0 for port 3
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P03            786
/**
* LWSwitch Rx Bandwidth Counter 0 for port 3
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P03            787

/**
* LWSwitch Tx Bandwidth Counter 0 for port 4
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P04            788
/**
* LWSwitch Rx Bandwidth Counter 0 for port 4
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P04            789

/**
* LWSwitch Tx Bandwidth Counter 0 for port 5
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P05            790
/**
* LWSwitch Rx Bandwidth Counter 0 for port 5
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P05            791

/**
* LWSwitch Tx Bandwidth Counter 0 for port 6
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P06            792
/**
* LWSwitch Rx Bandwidth Counter 0 for port 6
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P06            793

/**
* LWSwitch Tx Bandwidth Counter 0 for port 7
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P07            794
/**
* LWSwitch Rx Bandwidth Counter 0 for port 7
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P07            795

/**
* LWSwitch Tx Bandwidth Counter 0 for port 8
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P08            796
/**
* LWSwitch Rx Bandwidth Counter 0 for port 8
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P08            797

/**
* LWSwitch Tx Bandwidth Counter 0 for port 9
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P09            798
/**
* LWSwitch Rx Bandwidth Counter 0 for port 9
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P09            799

/**
* LWSwitch Tx Bandwidth Counter 0 for port 10
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P10            800
/**
* LWSwitch Rx Bandwidth Counter 0 for port 10
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P10            801

/**
* LWSwitch Tx Bandwidth Counter 0 for port 11
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P11            802
/**
* LWSwitch Rx Bandwidth Counter 0 for port 11
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P11            803
 
/**
* LWSwitch Tx Bandwidth Counter 0 for port 12
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P12            804
/**
* LWSwitch Rx Bandwidth Counter 0 for port 12
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P12            805

/**
* LWSwitch Tx Bandwidth Counter 0 for port 13
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P13            806
/**
* LWSwitch Rx Bandwidth Counter 0 for port 13
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P13            807

/**
* LWSwitch Tx Bandwidth Counter 0 for port 14
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P14            808
/**
* LWSwitch Rx Bandwidth Counter 0 for port 14
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P14            809

/**
* LWSwitch Tx Bandwidth Counter 0 for port 15
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P15            810
/**
* LWSwitch Rx Bandwidth Counter 0 for port 15
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P15            811

/**
* LWSwitch Tx Bandwidth Counter 0 for port 16
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P16            812
/**
* LWSwitch Rx Bandwidth Counter 0 for port 16
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P16            813

/**
* LWSwitch Tx Bandwidth Counter 0 for port 17
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_0_P17            814
/**
* <p>LWSwitch Rx Bandwidth Counter 0 for port 17</p>
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>LWSwitch Tx and RX Bandwidth Counter 1 for each port</p>
* <p>By default, Counter 1 counts packets.</p> 
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_0_P17            815

/**
* <p>LWSwitch Tx Bandwidth Counter 1 for port 0</p>
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P00            820
/**
* LWSwitch Rx Bandwidth Counter 1 for port 0
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P00            821

/**
* LWSwitch Tx Bandwidth Counter 1 for port 1
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P01            822
/**
* LWSwitch Rx Bandwidth Counter 1 for port 1
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P01            823

/**
* LWSwitch Tx Bandwidth Counter 1 for port 2
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P02            824
/**
* LWSwitch Rx Bandwidth Counter 1 for port 2
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P02            825

/**
* LWSwitch Tx Bandwidth Counter 1 for port 3
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P03            826
/**
* LWSwitch Rx Bandwidth Counter 1 for port 3
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P03            827

/**
* LWSwitch Tx Bandwidth Counter 1 for port 4
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P04            828
/**
* LWSwitch Rx Bandwidth Counter 1 for port 4
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P04            829

/**
* LWSwitch Tx Bandwidth Counter 1 for port 5
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P05            830
/**
* LWSwitch Rx Bandwidth Counter 1 for port 5
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P05            831

/**
* LWSwitch Tx Bandwidth Counter 1 for port 6
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P06            832
/**
* LWSwitch Rx Bandwidth Counter 1 for port 6
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P06            833

/**
* LWSwitch Tx Bandwidth Counter 1 for port 7
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P07            834
/**
* LWSwitch Rx Bandwidth Counter 1 for port 7
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P07            835

/**
* LWSwitch Tx Bandwidth Counter 1 for port 8
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P08            836
/**
* LWSwitch Rx Bandwidth Counter 1 for port 8
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P08            837

/**
* LWSwitch Tx Bandwidth Counter 1 for port 9
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P09            838
/**
* LWSwitch Rx Bandwidth Counter 1 for port 9
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P09            839

/**
* LWSwitch Tx Bandwidth Counter 0 for port 10
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P10            840
/**
* LWSwitch Rx Bandwidth Counter 1 for port 10
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P10            841

/**
* LWSwitch Tx Bandwidth Counter 1 for port 11
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P11            842
/**
* LWSwitch Rx Bandwidth Counter 1 for port 11
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P11            843

/**
* LWSwitch Tx Bandwidth Counter 1 for port 12
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P12            844
/**
* LWSwitch Rx Bandwidth Counter 1 for port 12
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P12            845

/**
* LWSwitch Tx Bandwidth Counter 0 for port 13
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P13            846
/**
* LWSwitch Rx Bandwidth Counter 1 for port 13
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P13            847

/**
* LWSwitch Tx Bandwidth Counter 1 for port 14
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P14            848
/**
* LWSwitch Rx Bandwidth Counter 1 for port 14
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P14            849

/**
* LWSwitch Tx Bandwidth Counter 1 for port 15
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P15            850
/**
* LWSwitch Rx Bandwidth Counter 1 for port 15
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P15            851

/**
* LWSwitch Tx Bandwidth Counter 1 for port 16
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P16            852
/**
* LWSwitch Rx Bandwidth Counter 1 for port 16
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P16            853

/**
* LWSwitch Tx Bandwidth Counter 1 for port 17
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_TX_1_P17            854
/**
* LWSwitch Rx Bandwidth Counter 1 for port 17
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* <p>&nbsp;</p>
* LWSwitch error counters
*/
#define DCGM_FI_DEV_LWSWITCH_BANDWIDTH_RX_1_P17            855

/**
* LWSwitch fatal error information.
* Note: value field indicates the specific SXid reported
*/
#define DCGM_FI_DEV_LWSWITCH_FATAL_ERRORS                  856

/**
* LWSwitch non fatal error information.
* Note: value field indicates the specific SXid reported
*/
#define DCGM_FI_DEV_LWSWITCH_NON_FATAL_ERRORS              857

/**
 * Starting field ID of the LWSwitch instance
 */
#define DCGM_FI_FIRST_LWSWITCH_FIELD_ID                    700

/**
 * Last field ID of the LWSwitch instance
 */
#define DCGM_FI_LAST_LWSWITCH_FIELD_ID                     860

/**
 * For now max LWSwitch field Ids taken as difference of DCGM_FI_LAST_LWSWITCH_FIELD_ID and DCGM_FI_FIRST_LWSWITCH_FIELD_ID + 1 i.e. 200
 */
#define DCGM_FI_MAX_LWSWITCH_FIELDS     DCGM_FI_LAST_LWSWITCH_FIELD_ID - DCGM_FI_FIRST_LWSWITCH_FIELD_ID + 1

/**
 * Profiling Fields. These all start with DCGM_FI_PROF_*
 */

/**
 * Ratio of time the graphics engine is active. The graphics engine is 
 * active if a graphics/compute context is bound and the graphics pipe or 
 * compute pipe is busy.
 */
#define DCGM_FI_PROF_GR_ENGINE_ACTIVE                      1001

/**
 * The ratio of cycles an SM has at least 1 warp assigned 
 * (computed from the number of cycles and elapsed cycles) 
 */
#define DCGM_FI_PROF_SM_ACTIVE                             1002

/**
 * The ratio of number of warps resident on an SM. 
 * (number of resident as a ratio of the theoretical 
 * maximum number of warps per elapsed cycle)
 */
#define DCGM_FI_PROF_SM_OCLWPANCY                          1003

/**
 * The ratio of cycles the tensor (HMMA) pipe is active 
 * (off the peak sustained elapsed cycles)
 */
#define DCGM_FI_PROF_PIPE_TENSOR_ACTIVE                    1004

/**
 * The ratio of cycles the device memory interface is 
 * active sending or receiving data.
 */
#define DCGM_FI_PROF_DRAM_ACTIVE                           1005

/**
 * Ratio of cycles the fp64 pipe is active.
 */
#define DCGM_FI_PROF_PIPE_FP64_ACTIVE                      1006

/**
 * Ratio of cycles the fp32 pipe is active.
 */
#define DCGM_FI_PROF_PIPE_FP32_ACTIVE                      1007

/**
 * Ratio of cycles the fp16 pipe is active. This does not include HMMA.
 */
#define DCGM_FI_PROF_PIPE_FP16_ACTIVE                      1008

/**
 * The number of bytes of active PCIe tx (transmit) data including both header and payload.
 * 
 * Note that this is from the perspective of the GPU, so copying data from device to host (DtoH)
 * would be reflected in this metric.
 */
#define DCGM_FI_PROF_PCIE_TX_BYTES                         1009

/**
 * The number of bytes of active PCIe rx (read) data including both header and payload.
 * 
 * Note that this is from the perspective of the GPU, so copying data from host to device (HtoD)
 * would be reflected in this metric.
 */
#define DCGM_FI_PROF_PCIE_RX_BYTES                         1010

/**
 * The number of bytes of active LwLink tx (transmit) data including both header and payload.
 */
#define DCGM_FI_PROF_LWLINK_TX_BYTES                       1011

/**
 * The number of bytes of active LwLink rx (read) data including both header and payload.
 */
#define DCGM_FI_PROF_LWLINK_RX_BYTES                       1012

/**
 * 1 greater than maximum fields above. This is the 1 greater than the maximum field id that could be allocated
 */
#define DCGM_FI_MAX_FIELDS                1013


/** @} */

/*****************************************************************************/

/**
 * Structure for formating the output for dmon.
 * Used as a member in dcgm_field_meta_p
 */
typedef struct
{
    char    shortName[10];  /* Short name corresponding to field. This short name 
                               is used to identify columns in dmon output.*/
    char    unit[4];        /* The unit of value. Eg: C(elsius), W(att), MB/s*/
    short   width;          /* Maximum width/number of digits that a value for field can have.*/
} dcgm_field_output_format_t,*dcgm_field_output_format_p;

/**
 * Structure to store meta data for the field
 */

typedef struct
{
    unsigned short fieldId;     /* Field identifier. DCGM_FI_? #define */
    char           fieldType;   /* Field type. DCGM_FT_? #define */
    unsigned char  size;        /* field size in bytes (raw value size). 0=variable (like DCGM_FT_STRING) */
    char           tag[48];     /* Tag for this field for serialization like 'device_temperature' */
    int            scope;       /* Field scope. DCGM_FS_? #define of this field's association */
    int            lwmlFieldId; /* Optional LWML field this DCGM field maps to. 0 = no mapping. Otherwise,
                                   this should be a LWML_FI_? #define from lwml.h */
    
    dcgm_field_output_format_p valueFormat; /* pointer to the structure that holds the formatting the values for fields */
} dcgm_field_meta_t, *dcgm_field_meta_p;

/***************************************************************************************************/
/** @addtogroup dcgmFieldIdentifiers
 *  @{
 */
/***************************************************************************************************/

/**
 * Get a pointer to the metadata for a field by its field ID. See DCGM_FI_? for a list of field IDs.
 * @param fieldId     IN:   One of the field IDs (DCGM_FI_?)
 * @return
 *      0       On Failure
 *      > 0     Pointer to field metadata structure if found.
 */
dcgm_field_meta_p DcgmFieldGetById(unsigned short fieldId);

/**
 * Get a pointer to the metadata for a field by its field tag.
 * @param tag       IN: Tag for the field of interest
 * @return
 *  0             On failure or not found
 *  > 0           Pointer to field metadata structure if found
 */
dcgm_field_meta_p DcgmFieldGetByTag(char *tag);

/**
 * Initialize the DcgmFields module. Call this once from inside
 * your program
 * @return 
 *  0                On success
 *  <0               On error
 */
int DcgmFieldsInit(void);

/**
 * Terminates the DcgmFields module. Call this once from inside your program
 * @return 
 *  0            On success
 *  <0           On error
 */
int DcgmFieldsTerm(void);

/**
 * Get the string version of a entityGroupId
 *
 * Returns         Pointer to a string like GPU/LwSwitch..etc
 *                 Null on error
 */
char *DcgmFieldsGetEntityGroupString(dcgm_field_entity_group_t entityGroupId);

/** @} */  


#ifdef __cplusplus
}
#endif


#endif //DCGMFIELDS_H
