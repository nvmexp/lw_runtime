/*
 * _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2008-2021 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl85b6/ctrl85b6pmu.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl85b6/ctrl85b6base.h"

/* PMU control core-related control commands */

/*
 * LW85B6_CTRL_PMU_SURFACE_INFO
 *
 *  hMemory
 *    The memory handle associated with the surface being described.
 *
 *  offset
 *    A PMU surface may be a subregion of a larger allocation.  This offset
 *    marks the start of the surface relative to the start of the memory
 *    allocation.
 *
 *  size
 *    Used in conjunction with the offset (above) to mark the end of the
 *    surface.
 */
typedef struct LW85B6_CTRL_PMU_SURFACE_INFO {
    LwHandle hMemory;
    LwU32    offset;
    LwU32    size;
} LW85B6_CTRL_PMU_SURFACE_INFO;


/*
 * LW85B6_CTRL_CMD_PMU_RESET
 *
 * This command forces the PMU to reset.  As an option, the PMU may restarted
 * using the previously loaded/running ucode image.
 *
 *  bReboot
 *    When LW_TRUE, the PMU will reboot using the previously loaded/running
 *    ucode image. When LW__FALSE, the PMU will remain in the reset state.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW85B6_CTRL_CMD_PMU_RESET (0x85b60103) /* finn: Evaluated from "(FINN_GT212_SUBDEVICE_PMU_PMU_INTERFACE_ID << 8) | LW85B6_CTRL_PMU_RESET_PARAMS_MESSAGE_ID" */

#define LW85B6_CTRL_PMU_RESET_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW85B6_CTRL_PMU_RESET_PARAMS {
    LwBool bReboot;
} LW85B6_CTRL_PMU_RESET_PARAMS;


/*
 * LW85B6_CTRL_CMD_PMU_BIND_APERTURE
 *
 * This command binds binds the aperture associated with the given memory
 * to the PMU by setting up the PMU instance block (tesla2-only) and writing
 * to the appropriate PMU HW registers.
 *
 *  vAddr
 *    The PMU virtual address for the aperture being bound.
 *
 *  hMemory
 *    The memory handle for the surfaces whose apertures are being bound to the
 *    PMU.
 *
 *  apertureIndex
 *    The DMA index issued in the PMU's DMA request when accessing (r/w) the
 *    aperture.
 *
 * Possible status values resturned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW85B6_CTRL_CMD_PMU_BIND_APERTURE (0x85b60104) /* finn: Evaluated from "(FINN_GT212_SUBDEVICE_PMU_PMU_INTERFACE_ID << 8) | LW85B6_CTRL_PMU_BIND_APERTURE_PARAMS_MESSAGE_ID" */

#define LW85B6_CTRL_PMU_BIND_APERTURE_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW85B6_CTRL_PMU_BIND_APERTURE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 vAddr, 8);
    LwHandle hMemory;
    LwU32    apertureIndex;
} LW85B6_CTRL_PMU_BIND_APERTURE_PARAMS;


/*
 * LW85B6_CTRL_CMD_PMU_UNIT_*
 *
 * Definition for unit ids used when registering for events and retrieving
 * events
*/
#define LW85B6_CTRL_PMU_UNIT_REWIND             (0x00000000)
#define LW85B6_CTRL_PMU_UNIT_I2C                (0x00000001)
#define LW85B6_CTRL_PMU_UNIT_SEQ                (0x00000002)
#define LW85B6_CTRL_PMU_UNIT_ELPG               (0x00000003)
#define LW85B6_CTRL_PMU_UNIT_GPIO               (0x00000004)
#define LW85B6_CTRL_PMU_UNIT_CEC                (0x00000005)
#define LW85B6_CTRL_PMU_UNIT_MEM                (0x00000006)
#define LW85B6_CTRL_PMU_UNIT_INIT               (0x00000007)
#define LW85B6_CTRL_PMU_UNIT_COUNT              (0x00000008)

/*
 * LW85B6_CTRL_CMD_PMU_EVENT_QUEUE_OFFSET_*
 *
 * Definition for offset in bytes within the event queue for head and tail
 * pointers and the actual queue data
*/
#define LW85B6_CTRL_PMU_EVENT_QUEUE_OFFSET_HEAD (0x00000000)
#define LW85B6_CTRL_PMU_EVENT_QUEUE_OFFSET_TAIL (0x00000004)
#define LW85B6_CTRL_PMU_EVENT_QUEUE_OFFSET_DATA (0x00000008)

/*
 * LW85B6_CTRL_CMD_PMU_EVENT_REGISTER
 *
 * This command allows registering for events from the PMU.
 *
 *  unitID
 *    The PMU unit ID to receive events on.
 *
 *  evtQueue
 *    Structure containing the tracking data for the event queue that will be
 *    populated with the events that are being registered for.
 *
 *  evtDesc
 *    [output] Will populated with a unique identifier/descriptor that
 *    identifies this registration.  This value must be kept and presented when
 *    unregistering for the event.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_INSUFFICIENT_RESOURCES
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW85B6_CTRL_CMD_PMU_EVENT_REGISTER      (0x85b60105) /* finn: Evaluated from "(FINN_GT212_SUBDEVICE_PMU_PMU_INTERFACE_ID << 8) | 0x5" */

typedef struct LW85B6_CTRL_PMU_EVENT_REGISTER_PARAMS {
    LwU8                         unitId;
    LW85B6_CTRL_PMU_SURFACE_INFO evtQueue;
    LwU32                        evtDesc;
} LW85B6_CTRL_PMU_EVENT_REGISTER_PARAMS;

/*
 * LW85B6_CTRL_CMD_PMU_EVENT_UNREGISTER
 *
 * This command allows unregistering for events from the PMU.  This command
 * receives the same parameter structure as was used when the event was first
 * registered for (see LW85B6_CTRL_PMU_EVENT_REGISTER_PARAMS).
 *
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 */
#define LW85B6_CTRL_CMD_PMU_EVENT_UNREGISTER (0x85b60106) /* finn: Evaluated from "(FINN_GT212_SUBDEVICE_PMU_PMU_INTERFACE_ID << 8) | 0x6" */

/*
 * Defines the max size of the RPC structure.
 */
#define LW85B6_CTRL_PMU_RPC_SIZE_MAX         2048

/*
 * LW85B6_CTRL_CMD_PMU_RPC_EXELWTE_SYNC
 *
 * This command allows exelwtion of the synchronous PMU RPCs (Remote Procedure
 * Calls).  It is supported only in MODS, and no input validation is performed
 * allowing testing of wide range of an invalid input.
 *
 *  exelwtionStatus [OUT]
 *    Status of the RPC exelwtion.  Value is from the RM if RM rejected
 *    the RPC request, otherwise the PMU exelwtion status is mapped.
 *
 *  sizeRpc [IN]
 *    Size of the RPC request RAW data in @ref rpc.
 *
 *  sizeScratch [IN]
 *    Scratch buffer size appended at the end of the payload, to be used by
 *    the PMU ucode.
 *
 *  queueId [IN]
 *    The logical identifier for the command queue this RPC is destined for.
 *    See RM_PMU_COMMAND_QUEUE_<xyz> for a list of logical queue identifiers.
 *
 *  rpc [INOUT]
 *    RPC request RAW data (including RPC header).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_ILWALID_REQUEST
 */
#define LW85B6_CTRL_CMD_PMU_RPC_EXELWTE_SYNC (0x85b60107) /* finn: Evaluated from "(FINN_GT212_SUBDEVICE_PMU_PMU_INTERFACE_ID << 8) | LW85B6_CTRL_PMU_RPC_EXELWTE_PARAMS_MESSAGE_ID" */

#define LW85B6_CTRL_PMU_RPC_EXELWTE_PARAMS_MESSAGE_ID (0x7U)

typedef struct LW85B6_CTRL_PMU_RPC_EXELWTE_PARAMS {
    LwU32 exelwtionStatus;
    LwU16 sizeRpc;
    LwU16 sizeScratch;
    LwU8  queueId;
    LwU8  rpc[LW85B6_CTRL_PMU_RPC_SIZE_MAX];
} LW85B6_CTRL_PMU_RPC_EXELWTE_PARAMS;

/*
 * LW85B6_CTRL_CMD_PMU_UCODE_STATE
 *
 * This command is used to retrieve the internal state of the PMU ucode.  It
 * may be used to determine when the PMU is ready to accept and process
 * commands.
 *
 *  ucodeState
 *    [output] the internal PMU ucode state
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW85B6_CTRL_CMD_PMU_UCODE_STATE (0x85b6010a) /* finn: Evaluated from "(FINN_GT212_SUBDEVICE_PMU_PMU_INTERFACE_ID << 8) | LW85B6_CTRL_PMU_UCODE_STATE_PARAMS_MESSAGE_ID" */

#define LW85B6_CTRL_PMU_UCODE_STATE_PARAMS_MESSAGE_ID (0xAU)

typedef struct LW85B6_CTRL_PMU_UCODE_STATE_PARAMS {
    LwU32 ucodeState;
} LW85B6_CTRL_PMU_UCODE_STATE_PARAMS;

/*
 * LW85B6_CTRL_CMD_PMU_UCODE_STATE_*
 *
 * Definition of the various internal PMU ucode states.
 *
 *  LW85B6_CTRL_PMU_UCODE_STATE_NONE
 *    Indicates that the PMU ucode has not been loaded (ie. the PMU has not
 *    been bootstrapped).  The PMU in not accepting commands at this point.
 *
 *  LW85B6_CTRL_PMU_UCODE_STATE_LOADED
 *    Indicates that the PMU ucode has been loaded but the PMU has not yet been
 *    started. The PMU is not accepting commands at this point.
 *
 *  LW85B6_CTRL_PMU_UCODE_STATE_RUNNING
 *    Indicates that the PMU ucode has been loaded and that the PMU is
 *    lwrrently exelwting its bootstrapping process. The PMU is not not
 *    accepting commands at this point.
 *
 *  LW85B6_CTRL_PMU_UCODE_STATE_FAILED
 *    Indicates that the PMU is fully bootstrapped but failed while handling
 *    PMU Init.
 *
 *  LW85B6_CTRL_PMU_UCODE_STATE_READY
 *    Indicates that the PMU is fully bootstrapped and ready to accept and
 *    process commands.
 */
#define LW85B6_CTRL_PMU_UCODE_STATE_NONE        (0x00000000)
#define LW85B6_CTRL_PMU_UCODE_STATE_LOADED      (0x00000001)
#define LW85B6_CTRL_PMU_UCODE_STATE_RUNNING     (0x00000002)
#define LW85B6_CTRL_PMU_UCODE_STATE_FAILED      (0x00000003)
#define LW85B6_CTRL_PMU_UCODE_STATE_READY       (0x00000004)


/*
 * LW85B6_CTRL_CMD_PMU_DETACH
 *
 * In the "detached" state, the PMU will cease all DMA operations as well as
 * all communication with the RM, and perform only vital operations such as GPU
 * fan-control. Once detached, the only way to return to a fully functional
 * state is to reset and rebootstrap.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_PARAM_STRUCT
 */
#define LW85B6_CTRL_CMD_PMU_DETACH              (0x85b6010b) /* finn: Evaluated from "(FINN_GT212_SUBDEVICE_PMU_PMU_INTERFACE_ID << 8) | 0xB" */

/*
 * Number of supported DMEM overlays (overestimated to allow future growth).
 */
#define LW85B6_CTRL_PMU_DMEM_INFO_OVL_COUNT_MAX 128

/*
 * Buffer size (bytes) retrieve run time VectorCast code coverage data
 */
#define LW85B6_CTRL_PMU_VC_DATA_SIZE_MAX        1000000U

/*
 * Buffer size (bytes) to retrieve VectorCast data from PMU via RPC
 */
#define LW85B6_CTRL_PMU_VC_RPC_DATA_SIZE_MAX    2000U

/*
 * LW85B6_CTRL_CMD_PMU_VC_DATA_GET
 *
 * This is command is used to retrieve the VectorCast code coverage data.
 *
 *  size        [OUT]
 *    Once complete, the output states the VectorCast code coverage date size.
 *
 *  bFormatted  [OUT]
 *    Indicates if the data copied from RM is formatted output or raw data
 *
 *  bReset      [IN]
 *    Flag to tell PMU to reset all vectorcast coverage data buffers
 *
 *  bDone       [OUT]
 *    Indicates there is no more data to be transferred
 *
 *  dataFile    [OUT]
 *    Buffer to hold VectorCast code coverage data.
 *
 * Possible status values returned are:
 *    LW_OK                         if successful
 *    LW_ERR_NOT_SUPPORTED          if feature is not supported
 *    LW_ERR_INSUFFICIENT_RESOURCES if feature is supported but not turned on
 */
#define LW85B6_CTRL_CMD_PMU_VC_DATA_GET         (0x85b60111) /* finn: Evaluated from "(FINN_GT212_SUBDEVICE_PMU_PMU_INTERFACE_ID << 8) | LW85B6_CTRL_PMU_VC_DATA_GET_PARAMS_MESSAGE_ID" */

#define LW85B6_CTRL_PMU_VC_DATA_GET_PARAMS_MESSAGE_ID (0x11U)

typedef struct LW85B6_CTRL_PMU_VC_DATA_GET_PARAMS {
    LwU32  size;
    LwBool bFormatted;
    LwBool bReset;
    LwBool bDone;
    LwU8   dataFile[LW85B6_CTRL_PMU_VC_DATA_SIZE_MAX];
} LW85B6_CTRL_PMU_VC_DATA_GET_PARAMS;

#define LW85B6_CTRL_PMU_FUZZ_SUPER_SURFACE_INPUT_DATA_SIZE_MAX 256

/*
 * LW85B6_CTRL_CMD_PMU_FUZZ_BOARDOBJ_SUPER_SURFACE
 *
 * This interface populates boardobj super surface with provided values.
 * It fuzzes specific struct in the RM_PMU_SUPER_SURFACE struct.
 * All those structs are consisted of header and entries.
 *
 *  input        [IN]
 *    Represents a buffer of data that is used to populate super surface.
 *
 *  inputSize     [IN]
 *    Size of buffer pInput.
 *
 *  structOffset  [IN]
 *    Offset of the struct that is being fuzzed in the RM_PMU_SUPER_SURFACE struct.
 *
 *  hdrSize       [IN]
 *    Header size of the struct being fuzzed.
 *
 *  entrySize     [IN]
 *    Entry size of the struct being fuzzed
 *
 *  structSize    [IN]
 *    Size of the struct being fuzzed
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_INSUFFICIENT_RESOURCES
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW85B6_CTRL_CMD_PMU_FUZZ_BOARDOBJ_SUPER_SURFACE        (0x85b6010f) /* finn: Evaluated from "(FINN_GT212_SUBDEVICE_PMU_PMU_INTERFACE_ID << 8) | LW85B6_CTRL_PMU_FUZZ_BOARDOBJ_SUPER_SURFACE_PARAMS_MESSAGE_ID" */

#define LW85B6_CTRL_PMU_FUZZ_BOARDOBJ_SUPER_SURFACE_PARAMS_MESSAGE_ID (0xFU)

typedef struct LW85B6_CTRL_PMU_FUZZ_BOARDOBJ_SUPER_SURFACE_PARAMS {
    LwU8  input[LW85B6_CTRL_PMU_FUZZ_SUPER_SURFACE_INPUT_DATA_SIZE_MAX];
    LwU32 inputSize;
    LwU32 structOffset;
    LwU32 hdrSize;
    LwU32 entrySize;
    LwU32 structSize;
} LW85B6_CTRL_PMU_FUZZ_BOARDOBJ_SUPER_SURFACE_PARAMS;

/* SDK_CTRL85B6PMU_H */
