/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2018-2022 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrlb0cc/ctrlb0ccprofiler.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlb0cc/ctrlb0ccbase.h"


/*!
 * LWB0CC_CTRL_CMD_RESERVE_HWPM_LEGACY
 *
 * Reserves the HWPM legacy PM system for use by the calling client.
 * This PM system will only be accessible if this reservation is
 * taken.
 *
 * If a device level reservation is held by another client, then this command
 * will fail regardless of reservation scope.
 *
 * This reservation can be released with @ref LWB0CC_CTRL_CMD_RELEASE_HWPM_LEGACY.
 *
 */
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*!
 * Context level reservation through B1CC is lwrrently disabled for GSP based
 * RM projects hence this comment is being unpublished
 *
 * Reservation scope (device or context) is determined by the class (B1CC/B2CC)
 * that a client instantiates.
 *
 * If one or more context level reservations are held by other clients, then
 * this command will fail if reservation scope is device or if a context
 * level reservation is already take for this context.
 *
 */
 /* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


#define LWB0CC_CTRL_CMD_RESERVE_HWPM_LEGACY (0xb0cc0101) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_PROFILER_INTERFACE_ID << 8) | LWB0CC_CTRL_RESERVE_HWPM_LEGACY_PARAMS_MESSAGE_ID" */

#define LWB0CC_CTRL_RESERVE_HWPM_LEGACY_PARAMS_MESSAGE_ID (0x1U)

typedef struct LWB0CC_CTRL_RESERVE_HWPM_LEGACY_PARAMS {
    /*!
     * [in] Enable ctxsw for HWPM.
     */
    LwBool ctxsw;
} LWB0CC_CTRL_RESERVE_HWPM_LEGACY_PARAMS;

/*!
 * LWB0CC_CTRL_CMD_RELEASE_HWPM_LEGACY
 *
 * Releases the reservation taken with @ref LWB0CC_CTRL_CMD_RESERVE_HWPM_LEGACY.
 *
 * This command does not take any parameters.
 *
 */
#define LWB0CC_CTRL_CMD_RELEASE_HWPM_LEGACY  (0xb0cc0102) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_PROFILER_INTERFACE_ID << 8) | 0x2" */

/*!
 * LWB0CC_CTRL_CMD_RESERVE_PM_AREA_SMPC
 *
 * Reserves the SMPC PM system for use by the calling client.
 * This PM system will only be accessible if this reservation is
 * taken.
 *
 * Reservation scope and rules are same as for @ref LWB0CC_CTRL_CMD_RESERVE_HWPM_LEGACY.
 *
 * This reservation can be released with @ref LWB0CC_CTRL_CMD_RELEASE_PM_AREA_SMPC.
 *
 */
#define LWB0CC_CTRL_CMD_RESERVE_PM_AREA_SMPC (0xb0cc0103) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_PROFILER_INTERFACE_ID << 8) | LWB0CC_CTRL_RESERVE_PM_AREA_SMPC_PARAMS_MESSAGE_ID" */

#define LWB0CC_CTRL_RESERVE_PM_AREA_SMPC_PARAMS_MESSAGE_ID (0x3U)

typedef struct LWB0CC_CTRL_RESERVE_PM_AREA_SMPC_PARAMS {
    /*!
     * [in] Enable ctxsw for SMPC.
     */
    LwBool ctxsw;
} LWB0CC_CTRL_RESERVE_PM_AREA_SMPC_PARAMS;

/*!
 * LWB0CC_CTRL_CMD_RELEASE_PM_AREA_SMPC
 *
 * Releases the reservation taken with @ref LWB0CC_CTRL_CMD_RESERVE_PM_AREA_SMPC.
 *
 * This command does not take any parameters.
 *
 */
#define LWB0CC_CTRL_CMD_RELEASE_PM_AREA_SMPC (0xb0cc0104) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_PROFILER_INTERFACE_ID << 8) | 0x4" */

/*!
 * LWB0CC_CTRL_CMD_ALLOC_PMA_STREAM
 *
 * Allocates PMA VA and map it to the buffers for streaming records and for
 * for streaming the updated bytes available in the buffer.
 *
 */
#define LWB0CC_CTRL_CMD_ALLOC_PMA_STREAM     (0xb0cc0105) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_PROFILER_INTERFACE_ID << 8) | 0x5" */

/*!
 * Defines the maximum size of PMA buffer for streamout. It can be up to 4GB minus one page
 * reserved for streaming mem_bytes (see @ref LWB0CC_PMA_BYTES_AVAILABLE_SIZE).
 */
#define LWB0CC_PMA_BUFFER_SIZE_MAX           (0xffe00000ULL) /* finn: Evaluated from "(4 * 1024 * 1024 * 1024 - 2 * 1024 * 1024)" */
#define LWB0CC_PMA_BYTES_AVAILABLE_SIZE      (0x1000) /* finn: Evaluated from "(4 * 1024)" */

typedef struct LWB0CC_CTRL_ALLOC_PMA_STREAM_PARAMS {
    /*!
     * [in] Memory handle (RW memory) for streaming records.
     * Size of this must be >= @ref pmaBufferOffset + @ref pmaBufferSize.
     */
    LwHandle hMemPmaBuffer;

    /*!
     * [in] Start offset of PMA buffer (offset in @ref hMemPmaBuffer).
     */
    LW_DECLARE_ALIGNED(LwU64 pmaBufferOffset, 8);

    /*!
     * [in] size of the buffer. This must be <= LWB0CC_PMA_BUFFER_SIZE_MAX.
     */
    LW_DECLARE_ALIGNED(LwU64 pmaBufferSize, 8);

    /*!
     * [in] Memory handle (RO memory) for streaming number of bytes available.
     * Size of this must be of at least @ref pmaBytesAvailableOffset +
     * @ref LWB0CC_PMA_BYTES_AVAILABLE_SIZE.
     */
    LwHandle hMemPmaBytesAvailable;

    /*!
     * [in] Start offset of PMA bytes available buffer (offset in @ref hMemPmaBytesAvailable).
     */
    LW_DECLARE_ALIGNED(LwU64 pmaBytesAvailableOffset, 8);

    /*!
     * [in] Enable ctxsw for PMA stream.
     */
    LwBool   ctxsw;

    /*!
     * [out] The PMA Channel Index associated with a given PMA stream.
     */
    LwU32    pmaChannelIdx;

    /*!
     * [out] PMA buffer VA. Note that this is a HWPM Virtual address.
     */
    LW_DECLARE_ALIGNED(LwU64 pmaBufferVA, 8);
} LWB0CC_CTRL_ALLOC_PMA_STREAM_PARAMS;

/*!
 * LWB0CC_CTRL_CMD_FREE_PMA_STREAM
 *
 * Releases (unmap and free) PMA stream allocated through
 * @ref LWB0CC_CTRL_CMD_ALLOC_PMA_STREAM.
 *
 */
#define LWB0CC_CTRL_CMD_FREE_PMA_STREAM (0xb0cc0106) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_PROFILER_INTERFACE_ID << 8) | LWB0CC_CTRL_FREE_PMA_STREAM_PARAMS_MESSAGE_ID" */

#define LWB0CC_CTRL_FREE_PMA_STREAM_PARAMS_MESSAGE_ID (0x6U)

typedef struct LWB0CC_CTRL_FREE_PMA_STREAM_PARAMS {
    /*!
     * [in] The PMA channel index associated with a given PMA stream.
     */
    LwU32 pmaChannelIdx;
} LWB0CC_CTRL_FREE_PMA_STREAM_PARAMS;

/*!
 * LWB0CC_CTRL_CMD_BIND_PM_RESOURCES
 *
 * Binds all PM resources reserved through @ref LWB0CC_CTRL_CMD_RESERVE_*
 * and with @ref LWB0CC_CTRL_CMD_ALLOC_PMA_STREAM with PMA engine.
 * After this call, interface is ready for programming a collection
 * of counters.
 * @Note: Any new PM resource reservation via LWB0CC_CTRL_CMD_RESERVE_* or
 * @ref LWB0CC_CTRL_CMD_ALLOC_PMA_STREAM request after this call will fail,
 * clients need to unbind (see @ref LWB0CC_CTRL_CMD_UNBIND_PM_RESOURCES) to
 * reserve more resources.
 *
 * This can be unbound with @ref LWB0CC_CTRL_CMD_UNBIND_PM_RESOURCES.
 *
 */
#define LWB0CC_CTRL_CMD_BIND_PM_RESOURCES         (0xb0cc0107) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_PROFILER_INTERFACE_ID << 8) | 0x7" */

/*!
 * LWB0CC_CTRL_CMD_UNBIND_PM_RESOURCES
 *
 * Unbinds PM resources that were bound with @ref LWB0CC_CTRL_CMD_BIND_PM_RESOURCES
 *
 */
#define LWB0CC_CTRL_CMD_UNBIND_PM_RESOURCES       (0xb0cc0108) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_PROFILER_INTERFACE_ID << 8) | 0x8" */

/*!
 * LWB0CC_CTRL_CMD_PMA_STREAM_UPDATE_GET_PUT
 *
 * This command updates bytes consumed by the SW and optionally gets the
 * current available bytes in the buffer.
 *
 */
#define LWB0CC_CTRL_CMD_PMA_STREAM_UPDATE_GET_PUT (0xb0cc0109) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_PROFILER_INTERFACE_ID << 8) | LWB0CC_CTRL_PMA_STREAM_UPDATE_GET_PUT_PARAMS_MESSAGE_ID" */

#define LWB0CC_AVAILABLE_BYTES_DEFAULT_VALUE      0xFFFFFFFF
#define LWB0CC_CTRL_PMA_STREAM_UPDATE_GET_PUT_PARAMS_MESSAGE_ID (0x9U)

typedef struct LWB0CC_CTRL_PMA_STREAM_UPDATE_GET_PUT_PARAMS {
    /*!
     * [in] Total bytes consumed by SW since last update.
     */
    LW_DECLARE_ALIGNED(LwU64 bytesConsumed, 8);

    /*!
     * [in] Initiate streaming of the bytes available (see @ref hMemPmaBytesAvailable).
     * RM will set the memory for streaming (see @ref hMemPmaBytesAvailable) to LWB0CC_AVAILABLE_BYTES_DEFAULT_VALUE and
     * client can optionally wait (see @ref bWait) for it to change from this value.
     */
    LwBool bUpdateAvailableBytes;

    /*!
     * [in] Waits for available bytes to get updated
     */
    LwBool bWait;

    /*!
     * [out] Bytes available in the PMA buffer (see @ref hMemPmaBuffer) for SW to consume.
     * This will only be populated if both bUpdateAvailableBytes and bWait are set
     * to TRUE.
     */
    LW_DECLARE_ALIGNED(LwU64 bytesAvailable, 8);

    /*!
     * [in] If set to TRUE, current put pointer will be returned in @ref putPtr.
     */
    LwBool bReturnPut;

    /*!
     * [out] Current PUT pointer (MEM_HEAD).
     * This will only be populated if bReturnPut is set to TRUE.
     */
    LW_DECLARE_ALIGNED(LwU64 putPtr, 8);

    /*!
     * [in] The PMA Channel Index associated with a given PMA stream.
     */
    LwU32  pmaChannelIdx;
} LWB0CC_CTRL_PMA_STREAM_UPDATE_GET_PUT_PARAMS;

/*!
 * Maximum number of register operations allowed in a single request.
 * @NOTE: @ref LWB0CC_REGOPS_MAX_COUNT is chosen to keep struct size
 * of @ref LWB0CC_CTRL_EXEC_REG_OPS_PARAMS under 4KB.
 */
#define LWB0CC_REGOPS_MAX_COUNT      (124)

/*!
 * LWB0CC_CTRL_CMD_EXEC_REG_OPS
 *
 * This command is used to submit an array containing one or more
 * register operations for processing.  Each entry in the
 * array specifies a single read or write operation. Each entry is
 * checked for validity in the initial pass: Only registers from PM area
 * are allowed using this interface and only register from PM systems for
 * which user has a valid reservation are allowed (see @ref LWB0CC_CTRL_CMD_RESERVE_*).
 * Operation type (@ref  LW2080_CTRL_GPU_REG_OP_TYPE_*) is not required to be passed in.
 */
#define LWB0CC_CTRL_CMD_EXEC_REG_OPS (0xb0cc010a) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_PROFILER_INTERFACE_ID << 8) | LWB0CC_CTRL_EXEC_REG_OPS_PARAMS_MESSAGE_ID" */

/*!
 * Structure definition for register operation. See @ref LW2080_CTRL_GPU_REG_OP.
 */
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
typedef LW2080_CTRL_GPU_REG_OP LWB0CC_GPU_REG_OP;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*!
 * Enumeration of different REG_OPS modes. This mode determines how a failure
 * of a regop is handled in a batch of regops.
 */
typedef enum LWB0CC_REGOPS_MODE {

    /*!
     * Either all regops will be exelwted or none of them will be exelwted.
     * Failing regop will have the appropriate status (see @ref LWB0CC_GPU_REG_OP::regStatus).
     */
    LWB0CC_REGOPS_MODE_ALL_OR_NONE = 0,
    /*!
     * All regops will be attempted and the ones that failed will have the
     * the appropriate status (see @ref LWB0CC_GPU_REG_OP::regStatus).
     */
    LWB0CC_REGOPS_MODE_CONTINUE_ON_ERROR = 1,
} LWB0CC_REGOPS_MODE;

#define LWB0CC_CTRL_EXEC_REG_OPS_PARAMS_MESSAGE_ID (0xAU)

typedef struct LWB0CC_CTRL_EXEC_REG_OPS_PARAMS {
    /*!
     * [in] Number of valid entries in the regOps array. This value cannot
     * exceed LWB0CC_REGOPS_MAX_COUNT.
     */
    LwU32              regOpCount;

    /*!
     * [in] Specifies the mode for the entire operation see @ref LWB0CC_REGOPS_MODE.
     */
    LWB0CC_REGOPS_MODE mode;

    /*!
     * [out] Provides status for the entire operation. This is only valid for
     * mode @ref LWB0CC_REGOPS_MODE_CONTINUE_ON_ERROR.
     */
    LwBool             bPassed;

    /*!
     * [out] This is lwrrently not populated.
     */
    LwBool             bDirect;

    /*!
     * [in/out] An array (of fixed size LWB0CC_REGOPS_MAX_COUNT) of register read or write
     * operations (see @ref LWB0CC_GPU_REG_OP)
     *
     */
#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
    LWB0CC_GPU_REG_OP  regOps[LWB0CC_REGOPS_MAX_COUNT];
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

} LWB0CC_CTRL_EXEC_REG_OPS_PARAMS;

/*!
 * LWB0CC_CTRL_CMD_RESERVE_PM_AREA_PC_SAMPLER
 *
 * Reserves the PC sampler system for use by the calling client.
 *
 * This reservation can be released with @ref LWB0CC_CTRL_CMD_RELEASE_PM_AREA_PC_SAMPLER.
 *
 * This command does not take any parameters.
 *
 * PC sampler is always context switched with a GR context, so reservation scope is
 * always context. This requires that profiler object is instantiated with a valid GR
 * context. See @ref LWB2CC_ALLOC_PARAMETERS.
 */

#define LWB0CC_CTRL_CMD_RESERVE_PM_AREA_PC_SAMPLER (0xb0cc010b) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_PROFILER_INTERFACE_ID << 8) | 0xB" */

/*!
 * LWB0CC_CTRL_CMD_RELEASE_PM_AREA_PC_SAMPLER
 *
 * Releases the reservation taken with @ref LWB0CC_CTRL_CMD_RESERVE_PM_AREA_PC_SAMPLER.
 *
 * This command does not take any parameters.
 *
 */
#define LWB0CC_CTRL_CMD_RELEASE_PM_AREA_PC_SAMPLER (0xb0cc010c) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_PROFILER_INTERFACE_ID << 8) | 0xC" */

/*!
 * LWB0CC_CTRL_CMD_GET_TOTAL_HS_CREDITS
 *
 * Gets the total high speed streaming credits available for the client.
 *
 * This command can only be performed after a bind using LWB0CC_CTRL_CMD_BIND_PM_RESOURCES.
 *
 */
#define LWB0CC_CTRL_CMD_GET_TOTAL_HS_CREDITS       (0xb0cc010d) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_PROFILER_INTERFACE_ID << 8) | LWB0CC_CTRL_GET_TOTAL_HS_CREDITS_PARAMS_MESSAGE_ID" */

#define LWB0CC_CTRL_GET_TOTAL_HS_CREDITS_PARAMS_MESSAGE_ID (0xDU)

typedef struct LWB0CC_CTRL_GET_TOTAL_HS_CREDITS_PARAMS {
    LwU32 numCredits;
} LWB0CC_CTRL_GET_TOTAL_HS_CREDITS_PARAMS;

/*!
 * LWB0CC_CTRL_CMD_SET_HS_CREDITS_CHIPLET
 *
 * Sets per chiplet (pmm router) credits for high speed streaming for a pma channel.
 *
 * @note: This command resets the current credits to 0 before setting the new values also
 *        if programming fails, it will reset credits to 0 for all the chiplets.
 *
 */
#define LWB0CC_CTRL_CMD_SET_HS_CREDITS (0xb0cc010e) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_PROFILER_INTERFACE_ID << 8) | 0xE" */

typedef enum LWB0CC_CHIPLET_TYPE {
    LWB0CC_CHIPLET_TYPE_ILWALID = 0,
    LWB0CC_CHIPLET_TYPE_FBP = 1,
    LWB0CC_CHIPLET_TYPE_GPC = 2,
    LWB0CC_CHIPLET_TYPE_SYS = 3,
} LWB0CC_CHIPLET_TYPE;

typedef enum LWB0CC_HS_CREDITS_CMD_STATUS {
    LWB0CC_HS_CREDITS_CMD_STATUS_OK = 0,
    /*!
     * More credits are requested than the total credits. Total credits can be queried using @ref LWB0CC_CTRL_CMD_GET_TOTAL_HS_CREDITS
     */
    LWB0CC_HS_CREDITS_CMD_STATUS_ILWALID_CREDITS = 1,
    /*!
     * Chiplet index is invalid.
     */
    LWB0CC_HS_CREDITS_CMD_STATUS_ILWALID_CHIPLET = 2,
} LWB0CC_HS_CREDITS_CMD_STATUS;

typedef struct LWB0CC_CTRL_PMA_STREAM_HS_CREDITS_INFO {
    /*!
     * Specifies the chiplet type @ref LWB0CC_CHIPLET_TYPE.
     */
    LwU8  chipletType;

    /*!
     * Specifies the logical index of the chiplet.
     */
    LwU8  chipletIndex;

    /*!
     * Specifies the number of credits for the chiplet.
     */
    LwU16 numCredits;
} LWB0CC_CTRL_PMA_STREAM_HS_CREDITS_INFO;

typedef struct LWB0CC_CTRL_PMA_STREAM_HS_CREDITS_STATUS {
    /*!
     * Status for the command @ref LWB0CC_HS_CREDITS_CMD_STATUS.
     */
    LwU8 status;

    /*!
     * Index of the failing @ref LWB0CC_CTRL_SET_HS_CREDITS_PARAMS::creditInfo entry. This
     * is only relevant if status is LWB0CC_HS_CREDITS_CMD_STATUS_ILWALID_CHIPLET.
     */
    LwU8 entryIndex;
} LWB0CC_CTRL_PMA_STREAM_HS_CREDITS_STATUS;

#define LWB0CC_MAX_CREDIT_INFO_ENTRIES (63)

typedef struct LWB0CC_CTRL_SET_HS_CREDITS_PARAMS {
    /*!
     * [in] The PMA Channel Index associated with a given PMA stream.
     */
    LwU8                                     pmaChannelIdx;

    /*!
     * [in] Number of valid entries in creditInfo.
     */
    LwU8                                     numEntries;

    /*!
     * [out] Provides status for the entire operation.
     */
    LWB0CC_CTRL_PMA_STREAM_HS_CREDITS_STATUS statusInfo;

    /*!
     * [in] Credit programming per chiplet
     */
    LWB0CC_CTRL_PMA_STREAM_HS_CREDITS_INFO   creditInfo[LWB0CC_MAX_CREDIT_INFO_ENTRIES];
} LWB0CC_CTRL_SET_HS_CREDITS_PARAMS;

/*!
 * LWB0CC_CTRL_CMD_GET_HS_CREDITS
 *
 * Gets per chiplet (pmm router) high speed streaming credits for a pma channel.
 *
 */
#define LWB0CC_CTRL_CMD_GET_HS_CREDITS (0xb0cc010f) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_PROFILER_INTERFACE_ID << 8) | 0xF" */

typedef LWB0CC_CTRL_SET_HS_CREDITS_PARAMS LWB0CC_CTRL_GET_HS_CREDITS_PARAMS;

/*!
 * LWB0CC_CTRL_CMD_ALLOC_PMA_STREAM_KERNEL
 *
 * Kernel-level API called by KMD in WSL to map buffers for: 1) the PMA to stream
 * profiling data into, and 2) for updating the bytes available in that record buffer.
 *
 */
#define LWB0CC_CTRL_CMD_ALLOC_PMA_STREAM_KERNEL (0xb0cc0110) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_PROFILER_INTERFACE_ID << 8) | 0x10" */

typedef struct LWB0CC_CTRL_ALLOC_PMA_STREAM_KERNEL_PARAMS {
    /*!
     * [in] Virtual Address of the PMA record buffer.
     */
    LW_DECLARE_ALIGNED(LwU64 pmaBufferCpuVA, 8);

    /*!
     * [in] Start offset of PMA buffer.
     */
    LW_DECLARE_ALIGNED(LwU64 pmaBufferOffset, 8);

    /*!
     * [in] Size of the buffer. This must be <= LWB0CC_PMA_BUFFER_SIZE_MAX.
     */
    LW_DECLARE_ALIGNED(LwU64 pmaBufferSize, 8);

    /*!
     * [in] Virtual Address of the PMA bytes available buffer.
     */
    LW_DECLARE_ALIGNED(LwU64 pmaMemBytesCpuVA, 8);

    /*!
     * [in] Start offset of PMA bytes available buffer.
     */
    LW_DECLARE_ALIGNED(LwU64 pmaMemBytesAvailableOffset, 8);

    /*!
     * [in] Enable ctxsw for PMA stream.
     */
    LwBool ctxsw;

    /*!
     * [out] The PMA Channel Index associated with a given PMA stream.
     */
    LwU32  pmaChannelIdx;

    /*!
     * [out] PMA buffer VA. Note that this is a HWPM Virtual Address.
     */
    LW_DECLARE_ALIGNED(LwU64 pmaBufferVA, 8);
} LWB0CC_CTRL_ALLOC_PMA_STREAM_KERNEL_PARAMS;

/*!
 * LWB0CC_CTRL_CMD_FREE_PMA_STREAM_KERNEL
 *
 * Releases (unmap and free) PMA stream resources allocated through
 * @ref LWB0CC_CTRL_CMD_RESERVE_PMA_KERNEL.
 *
 */
#define LWB0CC_CTRL_CMD_FREE_PMA_STREAM_KERNEL (0xb0cc0111) /* finn: Evaluated from "(FINN_MAXWELL_PROFILER_PROFILER_INTERFACE_ID << 8) | LWB0CC_CTRL_FREE_PMA_STREAM_KERNEL_PARAMS_MESSAGE_ID" */

#define LWB0CC_CTRL_FREE_PMA_STREAM_KERNEL_PARAMS_MESSAGE_ID (0x11U)

typedef struct LWB0CC_CTRL_FREE_PMA_STREAM_KERNEL_PARAMS {
    /*!
     * [in] The PMA channel index associated with a given PMA stream.
     */
    LwU32 pmaChannelIdx;
} LWB0CC_CTRL_FREE_PMA_STREAM_KERNEL_PARAMS;

/* _ctrlb0ccprofiler_h_ */
