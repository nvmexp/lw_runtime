/* 
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2009-2020 by LWPU Corporation.  All rights reserved.  All
* information contained herein is proprietary and confidential to LWPU
* Corporation.  Any use, reproduction, or disclosure without the written
* permission of LWPU Corporation is prohibited.
*
* _LWRM_COPYRIGHT_END_
*/

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl208f/ctrl208ffb.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl208f/ctrl208fbase.h"

/*
 * LW208F_CTRL_CMD_FB_GET_INFO
 *
 * This command returns fb engine information for the associated GPU.
 * The client specifies what information to query through 'index' param.
 * On success, the information is stored in the 'data' param.
 *
 *   index
 *     Specify what information to query. Please see below for valid values of
 *     indexes for this command.
 *   data
 *     On success, this param will hold the data that the client queried for.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW208F_CTRL_CMD_FB_GET_INFO (0x208f0501) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FB_INTERFACE_ID << 8) | LW208F_CTRL_FB_GET_INFO_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_FB_GET_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW208F_CTRL_FB_GET_INFO_PARAMS {
    LwU32 index;
    LW_DECLARE_ALIGNED(LwU64 data, 8);
} LW208F_CTRL_FB_GET_INFO_PARAMS;

/* valid fb info index values */
#define LW208F_CTRL_FB_INFO_INDEX_FREE_CONTIG_COMPRESSION_SIZE (0x00000001)

#define LW208F_CTRL_FB_INFO_INDEX_MAX                          LW208F_CTRL_FB_INFO_INDEX_FREE_CONTIG_COMPRESSION_SIZE

/*
 * LW208F_CTRL_CMD_FB_GET_ZBC_REFCOUNT
 *
 * This command gets the ZBC reference count associated with a given
 * compression tag address.  It is not supported on GPUs which support class
 * GF100_ZBC_CLEAR as it is specific to a different hardware implementation.
 * 
 *   compTagAddress
 *     The input parameter indicating the compression tag address for which the
 *     associated ZBC refcount should be looked up.
 *   zbcRefCount
 *     An array of reference counts for the ZBC clear values associated with
 *     compTagAddress.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW208F_CTRL_CMD_FB_GET_ZBC_REFCOUNT                    (0x208f0505) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FB_INTERFACE_ID << 8) | 0x5" */ // Deprecated, removed form RM

#define LW208F_CTRL_FB_GET_ZBC_REFCOUNT_MAX_REFCOUNTS          2
typedef struct LW208F_CTRL_FB_GET_ZBC_REFCOUNT_PARAMS {
    LwU32 compTagAddress;
    LwU32 zbcRefCount[LW208F_CTRL_FB_GET_ZBC_REFCOUNT_MAX_REFCOUNTS];
} LW208F_CTRL_FB_GET_ZBC_REFCOUNT_PARAMS;

/*
 * LW208F_CTRL_CMD_FB_CTRL_GPU_CACHE
 *
 * This command controls the state of a cache which all GPU memory accesses go
 * through.  If supported, it allows changing of the power state and the write
 * mode.  This is only supported when LW_VERIF_FEATURES is defined.  An error
 * will be returned if the requested combination of settings is not possible.
 *
 *   writeMode
 *     Specifies the write mode of the cache.  Possible values are defined in
 *     LW208F_CTRL_FB_CTRL_GPU_CACHE_WRITE_MODE. Passing _DEFAULT means to
 *     maintain the current write mode.  It is illegal to change the write mode
 *     while the cache is disabled or in the same call as a request to disable
 *     it.
 *   powerState
 *     Specifies the power state of the cache.  Possible values are defined in
 *     LW208F_CTRL_FB_CTRL_GPU_CACHE_POWER_STATE.  Passing _DEFAULT means
 *     to maintain the current power state.
 *   rcmState
 *     Specifies the reduced cache mode of the cache.  Possible values are
 *     defined in LW208F_CTRL_FB_CTRL_GPU_CACHE_RCM_STATE.  Passing _DEFAULT
 *     means to maintain the current RCM state.
 *   vgaCacheMode
 *     Specifies whether or not to enable VGA out-of-cache mode.  Possible
 *     values are defined in LW208F_CTRL_FB_CTRL_GPU_CACHE_VGA_MODE.  Passing
 *     _DEFAULT means to maintain the current VGA caching mode.  
 *   cacheReset
 *     Triggers a hardware reset of the cache.  Possible values are defined in
 *     LW208F_CTRL_FB_CTRL_GPU_CACHE_CACHE_RESET.  Passing _DEFAULT does
 *     nothing while passing _RESET clears all data in the cache.
 *   flags
 *     Contains flags to control the details of how transitions should be
 *     handled.  Possible values are defined in
 *     LW208F_CTRL_FB_CTRL_GPU_CACHE_FLAGS.  Passing _DEFAULT for any of
 *     the fields means to use the defaults specified by the Resource Manager.
 *     Note not all options are available for all transitions.  Flags that are
 *     set but not applicable will be silently ignored.
 *   bypassMode
 *     (Fermi only) Specifies the bypass mode of the L2 cache.  Normal GPU
 *     operation is _DISABLE.  For TEST ONLY, setting _ENABLE enables a debug
 *     mode where all transactions miss in L2 and no writes are combined,
 *     essentially disabling the caching feature of the L2 cache.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW208F_CTRL_CMD_FB_CTRL_GPU_CACHE (0x208f0506) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FB_INTERFACE_ID << 8) | LW208F_CTRL_FB_CTRL_GPU_CACHE_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_FB_CTRL_GPU_CACHE_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW208F_CTRL_FB_CTRL_GPU_CACHE_PARAMS {
    LwU32 writeMode;
    LwU32 powerState;
    LwU32 rcmState;
    LwU32 vgaCacheMode;
    LwU32 cacheReset;
    LwU32 flags;
    LwU32 bypassMode;
} LW208F_CTRL_FB_CTRL_GPU_CACHE_PARAMS;

/* valid values for writeMode */
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_WRITE_MODE_DEFAULT      (0x00000000)
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_WRITE_MODE_WRITETHROUGH (0x00000001)
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_WRITE_MODE_WRITEBACK    (0x00000002)

/* valid values for powerState */
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_POWER_STATE_DEFAULT     (0x00000000)
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_POWER_STATE_ENABLED     (0x00000001)
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_POWER_STATE_DISABLED    (0x00000002)

/* valid values for rcmState */
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_RCM_STATE_DEFAULT       (0x00000000)
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_RCM_STATE_FULL          (0x00000001)
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_RCM_STATE_REDUCED       (0x00000002)
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_RCM_STATE_ZERO_CACHE    (0x00000003)

/* valid values for vgaCacheMode */
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_VGA_MODE_DEFAULT        (0x00000000)
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_VGA_MODE_ENABLED        (0x00000001)
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_VGA_MODE_DISABLED       (0x00000002)

/* valid values for cacheReset */
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_CACHE_RESET_DEFAULT     (0x00000000)
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_CACHE_RESET_RESET       (0x00000001)

/* valid fields and values for flags */
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_FLAGS_MODE                   1:0
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_FLAGS_MODE_DEFAULT      (0x00000000)
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_FLAGS_MODE_RM           (0x00000001)
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_FLAGS_MODE_PMU          (0x00000002)

/* valid values for bypassMode */
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_BYPASS_MODE_DEFAULT     (0x00000000)
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_BYPASS_MODE_DISABLED    (0x00000001)
#define LW208F_CTRL_FB_CTRL_GPU_CACHE_BYPASS_MODE_ENABLED     (0x00000002)



/*
 * LW208F_CTRL_CMD_FB_SET_STATE
 *
 * This command is used to put fb engine in a state requested by the caller. 
 * 
 *   state
 *     This parameter specifies the desired engine state:
 *       LW208F_CTRL_FB_SET_STATE_STOPPED
 *         This value stops/halts the fb engine.
 *       LW208F_CTRL_FB_SET_STATE_RESTART      
 *         This value restarts fb from a stopped state. 
 *   
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW208F_CTRL_CMD_FB_SET_STATE                          (0x208f0508) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FB_INTERFACE_ID << 8) | LW208F_CTRL_FB_SET_STATE_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_FB_SET_STATE_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW208F_CTRL_FB_SET_STATE_PARAMS {
    LwU32 state;
} LW208F_CTRL_FB_SET_STATE_PARAMS;

/* valid values for state */
#define LW208F_CTRL_FB_SET_STATE_STOPPED  (0x00000000)
#define LW208F_CTRL_FB_SET_STATE_RESTART  (0x00000001)

/*
 * LW208F_CTRL_CMD_GPU_ECC_SCRUB_DIAG
 * 
 * This command reads all the settings internal to scrubbing (both asynchronous
 * and synchronous.
 *
 * Lwrrently implemented: FB offset scrubber has completed, FB offset that scrubber
 * is completing to, whether or not asynchronous scrubbing is enabled.
 * 
 *   fbOffsetCompleted
 *      This is the offset into FB that the scrubber has completed up to at the
 *      time this function is ilwoked. Note that the scrubber is top-down. Therefore
 *      the memory that remains unscrubbed is from 0x0 to fbOffsetCompleted.
 *
 *   fbEndOffset
 *       This is the offset of the base of the last block that ECC asynchronous
 *       scrubber has been tasked to scrub.
 *
 *   bAsyncScru bDisabled
 *       This is LW_TRUE if asynchronous scrubbing is disabled and LW_FALSE if
 *       asynchronous scrubbing is enabled.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW208F_CTRL_CMD_FB_ECC_SCRUB_DIAG (0x208f0509) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FB_INTERFACE_ID << 8) | LW208F_CTRL_CMD_FB_ECC_SCRUB_DIAG_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_CMD_FB_ECC_SCRUB_DIAG_PARAMS_MESSAGE_ID (0x9U)

typedef struct LW208F_CTRL_CMD_FB_ECC_SCRUB_DIAG_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 fbOffsetCompleted, 8);
    LW_DECLARE_ALIGNED(LwU64 fbEndOffset, 8);
    LwBool bAsyncScrubDisabled;
} LW208F_CTRL_CMD_FB_ECC_SCRUB_DIAG_PARAMS;

/*
 * LW208F_CTRL_CMD_GPU_ECC_ASYNCH_SCRUB_REGION
 *
 * This command launches the ECC scrubber in asynchronous mode. The scrubber, as
 * in normal operation, will continue to operate until all of FB (excluding
 * dedicated system memory) has been scrubbed. Like usual operation, scrubbing is
 * only done on silicon.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW208F_CTRL_CMD_FB_ECC_ASYNC_SCRUB_REGION (0x208f050a) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FB_INTERFACE_ID << 8) | LW208F_CTRL_CMD_FB_ECC_ASYNC_SCRUB_REGION_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_CMD_FB_ECC_ASYNC_SCRUB_REGION_PARAMS_MESSAGE_ID (0xAU)

typedef struct LW208F_CTRL_CMD_FB_ECC_ASYNC_SCRUB_REGION_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 startBlock, 8);
    LW_DECLARE_ALIGNED(LwU64 endBlock, 8);
} LW208F_CTRL_CMD_FB_ECC_ASYNC_SCRUB_REGION_PARAMS;

/*
 * LW208F_CTRL_CMD_GPU_ECC_ERROR_INFO
 * This is a structure that is defined here for diag/debug purposes in mods.
 * It is used to return the error information as part of the callback to 
 * kernel clients registering for SBE/DBE callbacks.
 */

typedef struct LW208F_CTRL_CMD_FB_ECC_ERROR_INFO {
    LwU32 row;
    LwU32 bank;
    LwU32 col;
    LwU32 extBank;
    LwU32 xbarAddress;
    LW_DECLARE_ALIGNED(LwU64 physAddress, 8);
} LW208F_CTRL_CMD_FB_ECC_ERROR_INFO;

/*
 * LW208F_CTRL_CMD_FB_ECC_GET_FORWARD_MAP_ADDRESS
 *
 * This command is used to forward map a physical FB address into a RBC
 * address. HW uses RBC addresses to inject test ECC errors.
 *
 *   pAddr
 *     The physical FB address to forwrd map
 *   row
 *     The "row" portion of the RBC address
 *   bank
 *     The "bank" portion of the RBC address
 *   col
 *     The "col" portion of the RBC address derived from ECC_ERR_INJECTION_ADDR
 *   extBank
 *     The "extBank" portion of the RBC address
 *   rank
 *     The "rank" of memory
 *   sublocation
 *     The physical sublocation the RBC address maps to (subpartition or channel
       depending on the architecture)
 *   partition
 *     The physical FBPA that the RBC address maps to
 *   rbcAddress
 *     The register value for ECC_ADDR
 *   rbcAddressExt
 *     The register value for ECC_ADDR_EXT
 *   rbcAddressExt2
 *     The register value for ECC_ADDR_EXT2 (GH100+ only)
 *   eccCol
 *     The "col" portion of the RBC address derived from ECC_ADDR. This value
 *     differs from the col value above since col returns the 32-byte column
 *     address of the physical address while eccCol returns the bit address
 *     that HW reports when receiving an ECC error.
 *
 * Possible status return values are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW208F_CTRL_CMD_FB_ECC_GET_FORWARD_MAP_ADDRESS (0x208f050c) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FB_INTERFACE_ID << 8) | LW208F_CTRL_CMD_FB_ECC_GET_FORWARD_MAP_ADDRESS_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_CMD_FB_ECC_GET_FORWARD_MAP_ADDRESS_PARAMS_MESSAGE_ID (0xLW)

typedef struct LW208F_CTRL_CMD_FB_ECC_GET_FORWARD_MAP_ADDRESS_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 pAddr, 8);
    LwU32 row;
    LwU32 bank;
    LwU32 col;
    LwU32 extBank;
    LwU32 rank;
    LwU32 sublocation;
    LwU32 partition;
    LwU32 writeKillPtr0;
    LwU32 injectionAddr;
    LwU32 injectionAddrExt;
    LwU32 rbcAddress;
    LwU32 rbcAddressExt;
    LwU32 rbcAddressExt2;
    LwU32 eccCol;
} LW208F_CTRL_CMD_FB_ECC_GET_FORWARD_MAP_ADDRESS_PARAMS;

/*
 * LW208F_CTRL_CMD_FB_ECC_SET_KILL_PTR
 *
 * This command sets the kill pointer for the specified DRAM address.  If
 * the kill pointer is set to LW208F_CTRL_FB_ERROR_TYPE_CORRECTABLE
 * or LW208F_CTRL_FB_ERROR_TYPE_UNCORRECTABLE, accesses to the specified
 * address will result in ECC errors until all kill pointers are unset
 * using LW208F_CTRL_FB_ERROR_TYPE_NONE.
 *
 * Only one kill pointer can be set at a time.  Setting a kill pointer will
 * clear all lwrrently set kill pointers and set the new kill pointer.
 * Calling LW208F_CTRL_FB_ERROR_TYPE_NONE simply clears all lwrrently set
 * kill pointers.
 *
 *   errorType
 *      The type of kill pointer to set.  LW208F_CTRL_FB_ERROR_TYPE_CORRECTABLE
 *      will set a single kill pointer resulting in a correctable error.
 *      LW208F_CTRL_FB_ERROR_TYPE_UNCORRECTABLE will set both kill pointers
 *      resulting in an uncorrectable error.  LW208F_CTRL_FB_ERROR_TYPE_NONE
 *      will clear all kill pointers, which stops the associated addresses
 *      from generating ECC errors if LW208F_CTRL_FB_ERROR_TYPE_CORRECTABLE
 *      or LW208F_CTRL_FB_ERROR_TYPE_UNCORRECTABLE was previously set.
 *      Only one kill pointer can be set at a time and setting a new
 *      kill pointer will clear the previous kill pointer.
 *
 *    address
 *      The physical DRAM address to be targeted by the kill pointer
 */
#define LW208F_CTRL_CMD_FB_ECC_SET_KILL_PTR (0x208f050e) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FB_INTERFACE_ID << 8) | LW208F_CTRL_FB_ECC_SET_KILL_PTR_PARAMS_MESSAGE_ID" */

typedef enum LW208F_CTRL_FB_ERROR_TYPE {
    LW208F_CTRL_FB_ERROR_TYPE_CORRECTABLE = 0,
    LW208F_CTRL_FB_ERROR_TYPE_UNCORRECTABLE = 1,
    LW208F_CTRL_FB_ERROR_TYPE_NONE = 2,
} LW208F_CTRL_FB_ERROR_TYPE;

#define LW208F_CTRL_FB_ECC_SET_KILL_PTR_PARAMS_MESSAGE_ID (0xEU)

typedef struct LW208F_CTRL_FB_ECC_SET_KILL_PTR_PARAMS {
    LW208F_CTRL_FB_ERROR_TYPE errorType;
    LW_DECLARE_ALIGNED(LwU64 address, 8);
} LW208F_CTRL_FB_ECC_SET_KILL_PTR_PARAMS;

/*
 * LW208F_CTRL_FB_INJECT_LTC_ECC_ERROR_LOC_*
 *
 * Defines to allow LTC ECC error injection to specific LTC location subtype(s).
 *
 *    - LW208F_CTRL_FB_INJECT_LTC_ECC_ERROR_LOC_ANY           : Inject to any subtype location
 *
 *    - LW208F_CTRL_FB_INJECT_LTC_ECC_ERROR_LOC_DSTG          : Inject LTC Data errors
 *
 *    - LW208F_CTRL_FB_INJECT_LTC_ECC_ERROR_LOC_RSTG          : Inject LTC CBC Parity errors
 *
 *    - LW208F_CTRL_FB_INJECT_LTC_ECC_ERROR_LOC_TSTG          : Inject LTC Tag Parity errors
 *
 *    - LW208F_CTRL_FB_INJECT_LTC_ECC_ERROR_LOC_CLIENT_POISON : Inject client poison
*/
#define LW208F_CTRL_FB_INJECT_LTC_ECC_ERROR_LOC                         3:0
#define LW208F_CTRL_FB_INJECT_LTC_ECC_ERROR_LOC_ANY (0)
#define LW208F_CTRL_FB_INJECT_LTC_ECC_ERROR_LOC_DSTG                    (LWBIT(0))
#define LW208F_CTRL_FB_INJECT_LTC_ECC_ERROR_LOC_RSTG                    (LWBIT(1))
#define LW208F_CTRL_FB_INJECT_LTC_ECC_ERROR_LOC_TSTG                    (LWBIT(2))
#define LW208F_CTRL_FB_INJECT_LTC_ECC_ERROR_LOC_CLIENT_POISON           (LWBIT(3))

/*
 * LW208F_CTRL_CMD_FB_INJECT_LTC_ECC_ERROR
 *
 * This API allows a client to inject ECC errors in the L2.
 *
 *   ltc:
 *      The physical LTC number to inject the error into.
 *   slice:
 *      THe physical slice number within the LTC to inject the error into.
 *   locationMask
 *      LTC location subtype(s) where error is to be injected. (Valid on Ampere and later)
 *   errorType
 *      Type of error to inject
 *      LW208F_CTRL_FB_ERROR_TYPE_CORRECTABLE for SBE.
 *      LW208F_CTRL_FB_ERROR_TYPE_UNCORRECTABLE for DBE.
 *
 */
#define LW208F_CTRL_CMD_FB_INJECT_LTC_ECC_ERROR     (0x208f050f) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FB_INTERFACE_ID << 8) | LW208F_CTRL_FB_INJECT_LTC_ECC_ERROR_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_FB_INJECT_LTC_ECC_ERROR_PARAMS_MESSAGE_ID (0xFU)

typedef struct LW208F_CTRL_FB_INJECT_LTC_ECC_ERROR_PARAMS {
    LwU8                      ltc;
    LwU8                      slice;
    LwU8                      locationMask;
    LW208F_CTRL_FB_ERROR_TYPE errorType;
} LW208F_CTRL_FB_INJECT_LTC_ECC_ERROR_PARAMS;

/*
 * LW208F_CTRL_CMD_FB_ECC_INJECTION_SUPPORTED
 *
 * Reports if error injection is supported for a given HW unit
 *
 * location [in]:
 *      The ECC protected unit for which ECC injection support is being checked.
 *      The location type is defined by LW208F_CTRL_FB_ECC_INJECTION_SUPPORTED_LOC.
 *
 * bCorrectableSupported [out]:
 *      Boolean value that shows if correcatable errors can be injected.
 *
 * bUncorrectableSupported [out]:
 *      Boolean value that shows if uncorrecatable errors can be injected.
 *
 * Return values:
 *      LW_OK on success
 *      LW_ERR_ILWALID_ARGUMENT if the requested location is invalid.
 *      LW_ERR_INSUFFICIENT_PERMISSIONS if priv write not enabled.
 *      LW_ERR_NOT_SUPPORTED otherwise
 *
 *
 */
#define LW208F_CTRL_CMD_FB_ECC_INJECTION_SUPPORTED (0x208f0510) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FB_INTERFACE_ID << 8) | LW208F_CTRL_FB_ECC_INJECTION_SUPPORTED_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_FB_ECC_INJECTION_SUPPORTED_PARAMS_MESSAGE_ID (0x10U)

typedef struct LW208F_CTRL_FB_ECC_INJECTION_SUPPORTED_PARAMS {
    LwU8   location;
    LwBool bCorrectableSupported;
    LwBool bUncorrectableSupported;
} LW208F_CTRL_FB_ECC_INJECTION_SUPPORTED_PARAMS;

#define LW208F_CTRL_FB_ECC_INJECTION_SUPPORTED_LOC          0:0
#define LW208F_CTRL_FB_ECC_INJECTION_SUPPORTED_LOC_LTC  (0x00000000)
#define LW208F_CTRL_FB_ECC_INJECTION_SUPPORTED_LOC_DRAM (0x00000001)

/*
 * LW208F_CTRL_CMD_FB_ECC_SET_KILL_PTR
 *
 * This command sets the write kill for the specified DRAM address.  If set,
 * writes to the specified address won't update the ECC checkbits.  When unset,
 * writes the specified address will update the ECC checkbits.
 *
 * Only one write kill register can be set at a time.  Setting a write kill
 * will clear all lwrrently set write kills and set the new write kill.
 * Calling this ctrl call with setWriteKill = false simply clears all lwrrently
 * set write kills.
 *
 *   setWriteKill
 *      When true, the ECC checkbits for the specified address won't update on
 *      writes. When false, the ECC checkbits for the specified address will
 *      revert to normal behavior and update on all writes.
 *
 *   address
 *      The physical DRAM address to be targeted by the write kill
 */
#define LW208F_CTRL_CMD_FB_ECC_SET_WRITE_KILL           (0x208f0511) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FB_INTERFACE_ID << 8) | LW208F_CTRL_FB_ECC_SET_WRITE_KILL_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_FB_ECC_SET_WRITE_KILL_PARAMS_MESSAGE_ID (0x11U)

typedef struct LW208F_CTRL_FB_ECC_SET_WRITE_KILL_PARAMS {
    LwBool setWriteKill;
    LW_DECLARE_ALIGNED(LwU64 address, 8);
} LW208F_CTRL_FB_ECC_SET_WRITE_KILL_PARAMS;

/*
 * LW208F_CTRL_FB_REMAPPING_ADDRESS_INFO
 *
 *   physicalAddress
 *     Physical address to be remapped
 *   source
 *     The reason for retirement. Valid values for this parameter are
 *     from LW2080_CTRL_FB_REMAPPED_ROW_SOURCE_*
 *   status
 *     Reason for row remapping failure. Valid values are:
 *       LW208F_CTRL_FB_REMAP_ROW_STATUS_OK
 *         No error
 *       LW208F_CTRL_FB_REMAP_ROW_STATUS_REMAPPING_PENDING
 *         The remapping is pending
 *       LW208F_CTRL_FB_REMAP_ROW_STATUS_TABLE_FULL
 *         Table full
 *       LW208F_CTRL_FB_REMAP_ROW_STATUS_ALREADY_REMAPPED
 *         Attempting to remap a reserved row
 *       LW208F_CTRL_FB_REMAP_ROW_STATUS_INTERNAL_ERROR
 *         Some other RM failure
 */
typedef struct LW208F_CTRL_FB_REMAPPING_ADDRESS_INFO {
    LW_DECLARE_ALIGNED(LwU64 physicalAddress, 8);
    LwU8  source;
    LwU32 status;
} LW208F_CTRL_FB_REMAPPING_ADDRESS_INFO;

/* valid values for status */
#define LW208F_CTRL_FB_REMAP_ROW_STATUS_OK                (0x00000000)
#define LW208F_CTRL_FB_REMAP_ROW_STATUS_REMAPPING_PENDING (0x00000001)
#define LW208F_CTRL_FB_REMAP_ROW_STATUS_TABLE_FULL        (0x00000002)
#define LW208F_CTRL_FB_REMAP_ROW_STATUS_ALREADY_REMAPPED  (0x00000003)
#define LW208F_CTRL_FB_REMAP_ROW_STATUS_INTERNAL_ERROR    (0x00000004)

#define LW208F_CTRL_FB_REMAPPED_ROWS_MAX_ROWS             (0x00000200)
/*
 * LW208F_CTRL_CMD_FB_REMAP_ROW
 *
 * This command will write entries to Inforom. During init the entries will be
 * read and used to remap a row.
 *
 *   addressList
 *     This input parameter is an array of LW208F_CTRL_FB_REMAPPING_ADDRESS_INFO
 *     structures containing information used for row remapping. Valid entries
 *     are adjacent
 *   validEntries
 *     This input parameter specifies the number of valid entries in the
 *     address array
 *   numEntriesAdded
 *     This output parameter specifies how many validEntries were successfully
 *     added to the Inforom
 *
 *   Possible status values returned are:
 *     LW_OK
 *     LW_ERR_ILWALID_ARGUMENT
 */
#define LW208F_CTRL_CMD_FB_REMAP_ROW                      (0x208f0512) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FB_INTERFACE_ID << 8) | LW208F_CTRL_FB_REMAP_ROW_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_FB_REMAP_ROW_PARAMS_MESSAGE_ID (0x12U)

typedef struct LW208F_CTRL_FB_REMAP_ROW_PARAMS {
    LW_DECLARE_ALIGNED(LW208F_CTRL_FB_REMAPPING_ADDRESS_INFO addressList[LW208F_CTRL_FB_REMAPPED_ROWS_MAX_ROWS], 8);
    LwU32 validEntries;
    LwU32 numEntriesAdded;
} LW208F_CTRL_FB_REMAP_ROW_PARAMS;

/*
 * LW208F_CTRL_CMD_FB_REVERSE_MAP_RBC_ADDR_TO_PA
 *
 * This command takes rbc addresses and colwerts them to a physical address.
 *
 *   address
 *     Output containing the physical address
 *   rbcAddress
 *     Whole register value of ECC_ADDR
 *   rbcAddressExt
 *     Whole register value of ECC_ADDR_EXT
 *   rbcAddressExt2
 *     Whole register value of ECC_ADDR_EXT2
 *   partition
 *     Virtual partition index
 *   sublocation
 *     The virtual sublocation index (subpartition/channel)
 *
 *   Possible status values returned are:
 *     LW_OK
 *     LW_ERR_NOT_SUPPORTED
 */
#define LW208F_CTRL_CMD_FB_REVERSE_MAP_RBC_ADDR_TO_PA (0x208f0513) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FB_INTERFACE_ID << 8) | LW208F_CTRL_FB_REVERSE_MAP_RBC_ADDR_TO_PA_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_FB_REVERSE_MAP_RBC_ADDR_TO_PA_PARAMS_MESSAGE_ID (0x13U)

typedef struct LW208F_CTRL_FB_REVERSE_MAP_RBC_ADDR_TO_PA_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 address, 8);
    LwU32 rbcAddress;
    LwU32 rbcAddressExt;
    LwU32 rbcAddressExt2;
    LwU32 partition;
    LwU32 sublocation;
} LW208F_CTRL_FB_REVERSE_MAP_RBC_ADDR_TO_PA_PARAMS;

/**
 * LW208F_CTRL_CMD_FB_TOGGLE_PHYSICAL_ADDRESS_ECC_ON_OFF
 *
 * This command will colwert a physical address when ECC is on to the physical
 * address when ECC is off or vice versa
 *
 * @params[in] LwU64 inputAddress
 *     Input physical address
 *
 * @params[in] LwBool eccOn
 *     Whether or not input physical address is with ECC on or off
 *
 * @params[out] LwU64 outputAddress
 *     Output physical address
 */
#define LW208F_CTRL_CMD_FB_TOGGLE_PHYSICAL_ADDRESS_ECC_ON_OFF (0x208f0514) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FB_INTERFACE_ID << 8) | LW208F_CTRL_FB_TOGGLE_PHYSICAL_ADDRESS_ECC_ON_OFF_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_FB_TOGGLE_PHYSICAL_ADDRESS_ECC_ON_OFF_PARAMS_MESSAGE_ID (0x14U)

typedef struct LW208F_CTRL_FB_TOGGLE_PHYSICAL_ADDRESS_ECC_ON_OFF_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 inputAddress, 8);
    LwBool eccOn;
    LW_DECLARE_ALIGNED(LwU64 outputAddress, 8);
} LW208F_CTRL_FB_TOGGLE_PHYSICAL_ADDRESS_ECC_ON_OFF_PARAMS;

/*
 * LW208F_CTRL_CMD_FB_CLEAR_REMAPPED_ROWS_PARAMS;
 *
 * This command clears remapping entries from the Inforom's row remapping table.
 *
 *   sourceMask
 *     This is a bit mask of LW2080_CTRL_FB_REMAPPED_ROW_SOURCE. Rows
 *     remapped from the specified sources will be cleared/removed from the
 *     Inforom RRL object entries list.
 *
 *   Possbile status values returned are:
 *     LW_OK
 *     LW_ERR_NOT_SUPPORTED
 */
#define LW208F_CTRL_CMD_FB_CLEAR_REMAPPED_ROWS (0x208f0515) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FB_INTERFACE_ID << 8) | LW208F_CTRL_FB_CLEAR_REMAPPED_ROWS_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_FB_CLEAR_REMAPPED_ROWS_PARAMS_MESSAGE_ID (0x15U)

typedef struct LW208F_CTRL_FB_CLEAR_REMAPPED_ROWS_PARAMS {
    LwU32 sourceMask;
} LW208F_CTRL_FB_CLEAR_REMAPPED_ROWS_PARAMS;

/*
 * LW208F_CTRL_CMD_FB_GET_NUM_SUBLOCATIONS;
 *
 * Returns the number of dram sublocations. The framebuffer is split into
 * partitions and within those partitions the sublocation access is either on a
 * subpartition basis or a channel basis.
 *
 * numSublocations [out]
 *   Equal to subpartitions for pre Hopper and channels Hopper+
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW208F_CTRL_CMD_FB_GET_NUM_SUBLOCATIONS (0x208f0516) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_DIAG_FB_INTERFACE_ID << 8) | LW208F_CTRL_FB_GET_NUM_SUBLOCATIONS_PARAMS_MESSAGE_ID" */

#define LW208F_CTRL_FB_GET_NUM_SUBLOCATIONS_PARAMS_MESSAGE_ID (0x16U)

typedef struct LW208F_CTRL_FB_GET_NUM_SUBLOCATIONS_PARAMS {
    LwU32 numSublocations;
} LW208F_CTRL_FB_GET_NUM_SUBLOCATIONS_PARAMS;

/* _ctrl208ffb_h_ */
