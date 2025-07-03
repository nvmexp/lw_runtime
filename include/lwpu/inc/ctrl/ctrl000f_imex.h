/*
* _LWRM_COPYRIGHT_BEGIN_
*
* Copyright 2022 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl000f_imex.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#include "ctrl/ctrlxxxx.h"
#include "class/cl00fa.h"

/*
 * LW000F_CTRL_CMD_GET_FABRIC_EVENTS
 *
 * Queries the internal fabric object for import lifetime
 * events.
 *
 *   eventArray [OUT]
 *     An array of import lifetime events
 *   numEvents [OUT]
 *     The number of valid events in eventArray
 *   bMoreEvents [OUT]
 *     Whether there are any remaining events to be queried
 *
 * Possible status values returned are:
 *    LW_ERR_NOT_SUPPORTED
 *    LW_OK
 */
#define LW000F_CTRL_CMD_GET_FABRIC_EVENTS     (0xf0201U) /* finn: Evaluated from "(FINN_FABRIC_MANAGER_SESSION_IMEX_INTERFACE_ID << 8) | LW000F_CTRL_GET_FABRIC_EVENTS_PARAMS_MESSAGE_ID" */

#define LW000F_CTRL_FABRIC_EVENT_TYPE_IMPORT  0U
#define LW000F_CTRL_FABRIC_EVENT_TYPE_RELEASE 1U

typedef struct LW000F_CTRL_FABRIC_EVENT {
    LwU32    eventType;
    LwU32    fabricBaseAddr;
    LwU16    nodeId;
    LwU32    gpuId;
    LwHandle hExportClient;
    LwHandle hExportObject;
    LW_DECLARE_ALIGNED(LwU64 size, 8);
    LwHandle hImportClient;
    LwHandle hImportObject;
} LW000F_CTRL_FABRIC_EVENT;

#define LW000F_CTRL_GET_FABRIC_EVENTS_ARRAY_SIZE 500U

#define LW000F_CTRL_GET_FABRIC_EVENTS_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW000F_CTRL_GET_FABRIC_EVENTS_PARAMS {
    LW_DECLARE_ALIGNED(LW000F_CTRL_FABRIC_EVENT eventArray[LW000F_CTRL_GET_FABRIC_EVENTS_ARRAY_SIZE], 8);
    LwU32  numEvents;
    LwBool bMoreEvents;
} LW000F_CTRL_GET_FABRIC_EVENTS_PARAMS;

/*
 * LW000F_CTRL_CMD_SET_FABRIC_NODE_ID
 *
 * Sets the global fabric node ID.
 *
 * nodeId [IN]
 *   The node ID value
 *
 * Possible status values returned are:
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_OK
 */
#define LW000F_CTRL_CMD_SET_FABRIC_NODE_ID (0xf0202U) /* finn: Evaluated from "(FINN_FABRIC_MANAGER_SESSION_IMEX_INTERFACE_ID << 8) | LW000F_CTRL_SET_FABRIC_NODE_ID_PARAMS_MESSAGE_ID" */

#define LW000F_CTRL_SET_FABRIC_NODE_ID_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW000F_CTRL_SET_FABRIC_NODE_ID_PARAMS {
    LwU16 nodeId;
} LW000F_CTRL_SET_FABRIC_NODE_ID_PARAMS;

/*
 * LW000F_CTRL_CMD_GET_FABRIC_EVENTS_V2
 *
 * Queries the fabric object for events.
 *
 *  eventArray [OUT]
 *    An array of import lifetime events.
 *
 *  numEvents [OUT]
 *    The number of valid events in eventArray.
 *
 *  bMoreEvents [OUT]
 *    Whether there are any remaining events to be queried.
 *
 * Possible status values returned are:
 *    LW_ERR_NOT_SUPPORTED
 *    LW_OK
 */
#define LW000F_CTRL_CMD_GET_FABRIC_EVENTS_V2          (0xf0203U) /* finn: Evaluated from "(FINN_FABRIC_MANAGER_SESSION_IMEX_INTERFACE_ID << 8) | LW000F_CTRL_GET_FABRIC_EVENTS_V2_PARAMS_MESSAGE_ID" */

/* Event types */
#define LW000F_CTRL_FABRIC_EVENT_V2_TYPE_MEM_IMPORT   0U
#define LW000F_CTRL_FABRIC_EVENT_V2_TYPE_MEM_UNIMPORT 1U

/*
 *  exportNodeId
 *    ID of the exporter node where memory will be imported.
 *
 *  gpuId
 *    ID of the memory owner GPU.
 *
 *  exportUuid
 *    Universally unique identifier of the export object. This is extracted
 *    from a fabric packet.
 *
 *  index
 *    Index of the export object to which the memory object is attached.
 */
typedef struct LW000F_CTRL_FABRIC_MEM_IMPORT_EVENT_DATA {
    LwU16 exportNodeId;
    LwU32 gpuId;
    LwU8  exportUuid[LW_FABRIC_UUID_LEN];
    LwU16 index;
} LW000F_CTRL_FABRIC_MEM_IMPORT_EVENT_DATA;

/*
 *  exportNodeId
 *    ID of the exporter node where memory will be unimported.
 *
 *  importEventId
 *    ID of the corresponding import event.
 */
typedef struct LW000F_CTRL_FABRIC_MEM_UNIMPORT_EVENT_DATA {
    LwU16 exportNodeId;
    LW_DECLARE_ALIGNED(LwU64 importEventId, 8);
} LW000F_CTRL_FABRIC_MEM_UNIMPORT_EVENT_DATA;

/*
 *  type
 *    Event type, one of LW000F_CTRL_FABRIC_EVENT_V2_TYPE_*.
 *
 *  id
 *    A monotonically increasing event ID.
 *
 *  data
 *    Event data
 */
typedef struct LW000F_CTRL_FABRIC_EVENT_V2 {
    LwU8 type;
    LW_DECLARE_ALIGNED(LwU64 id, 8);

    union {
        LW000F_CTRL_FABRIC_MEM_IMPORT_EVENT_DATA import;
        LW_DECLARE_ALIGNED(LW000F_CTRL_FABRIC_MEM_UNIMPORT_EVENT_DATA unimport, 8);
    } data;
} LW000F_CTRL_FABRIC_EVENT_V2;

#define LW000F_CTRL_GET_FABRIC_EVENTS_V2_ARRAY_SIZE 128U

#define LW000F_CTRL_GET_FABRIC_EVENTS_V2_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW000F_CTRL_GET_FABRIC_EVENTS_V2_PARAMS {
    LW_DECLARE_ALIGNED(LW000F_CTRL_FABRIC_EVENT_V2 eventArray[LW000F_CTRL_GET_FABRIC_EVENTS_V2_ARRAY_SIZE], 8);
    LwU32  numEvents;
    LwBool bMoreEvents;
} LW000F_CTRL_GET_FABRIC_EVENTS_V2_PARAMS;

/*
 * LW000F_CTRL_CMD_FINISH_MEM_UNIMPORT
 *
 * Notifies the unimport sequence is finished.
 *
 *  tokenArray [IN]
 *    An array of tokens that finished the unimport sequence.
 *
 *  numTokens [IN]
 *    The number of valid tokens in the tokenArray.
 *
 * Possible status values returned are:
 *    LW_ERR_OBJECT_NOT_FOUND
 *    LW_ERR_NOT_SUPPORTED
 *    LW_OK
 */
#define LW000F_CTRL_CMD_FINISH_MEM_UNIMPORT (0xf0204U) /* finn: Evaluated from "(FINN_FABRIC_MANAGER_SESSION_IMEX_INTERFACE_ID << 8) | LW000F_CTRL_FINISH_MEM_UNIMPORT_PARAMS_MESSAGE_ID" */

/*
 *  unimportEventId
 *    ID of the unimport event.
 */
typedef struct LW000F_CTRL_FABRIC_UNIMPORT_TOKEN {
    LW_DECLARE_ALIGNED(LwU64 unimportEventId, 8);
} LW000F_CTRL_FABRIC_UNIMPORT_TOKEN;

#define LW000F_CTRL_FINISH_MEM_UNIMPORT_ARRAY_SIZE 256U

#define LW000F_CTRL_FINISH_MEM_UNIMPORT_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW000F_CTRL_FINISH_MEM_UNIMPORT_PARAMS {
    LW_DECLARE_ALIGNED(LW000F_CTRL_FABRIC_UNIMPORT_TOKEN tokenArray[LW000F_CTRL_FINISH_MEM_UNIMPORT_ARRAY_SIZE], 8);
    LwU32 numTokens;
} LW000F_CTRL_FINISH_MEM_UNIMPORT_PARAMS;

/* _ctrl000f_imex.h_ */
