/* _LWRM_COPYRIGHT_BEGIN_
 *
 * Copyright 2011-2015 by LWPU Corporation.  All rights reserved.  All
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
// Source file: ctrl/ctrl503c.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"
#include "ctrl/ctrl503c/ctrl503cbase.h"

/*
 * LW503C_CTRL_CMD_REGISTER_VA_SPACE
 *
 * This command registers the specified GPU VA space with the given
 * LW50_THIRD_PARTY_P2P object, and returns a token that
 * uniquely identifies the VA space within the object's parent
 * client.
 *
 * Its parameter structure has the following fields:
 *
 *   hVASpace
 *      This field specifies the GPU VA space to be registered
 *      with the third-party P2P object.
 *
 *   vaSpaceToken
 *     Upon successful completion of the regristration attempt,
 *     this field holds the new VA space identifier.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LWOS_STATUS_ILWALID_OBJECT_HANDLE
 *   LWOS_STATUS_ILWALID_ARGUMENT
 *   LW_ERR_INSUFFICIENT_RESOURCES
 */
#define LW503C_CTRL_CMD_REGISTER_VA_SPACE (0x503c0102) /* finn: Evaluated from "(FINN_LW50_THIRD_PARTY_P2P_P2P_INTERFACE_ID << 8) | LW503C_CTRL_REGISTER_VA_SPACE_PARAMS_MESSAGE_ID" */

#define LW503C_CTRL_REGISTER_VA_SPACE_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW503C_CTRL_REGISTER_VA_SPACE_PARAMS {
    LwHandle hVASpace;
    LW_DECLARE_ALIGNED(LwU64 vaSpaceToken, 8);
} LW503C_CTRL_REGISTER_VA_SPACE_PARAMS;


/*
 * LW503C_CTRL_CMD_UNREGISTER_VA_SPACE
 *
 * This command unregisters (a previously registered) GPU VA space.
 *
 * Its parameter structure has the following field:
 *
 *   hVASpace
 *      This field specifies the GPU VA space to be
 *      unregistered.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */
#define LW503C_CTRL_CMD_UNREGISTER_VA_SPACE (0x503c0103) /* finn: Evaluated from "(FINN_LW50_THIRD_PARTY_P2P_P2P_INTERFACE_ID << 8) | LW503C_CTRL_UNREGISTER_VA_SPACE_PARAMS_MESSAGE_ID" */

#define LW503C_CTRL_UNREGISTER_VA_SPACE_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW503C_CTRL_UNREGISTER_VA_SPACE_PARAMS {
    LwHandle hVASpace;
} LW503C_CTRL_UNREGISTER_VA_SPACE_PARAMS;


/*
 * LW503C_CTRL_CMD_REGISTER_VIDMEM
 *
 * This command registers a video memory allocation with the given
 * LW50_THIRD_PARTY_P2P object.  Registration of video memory
 * allocations is required if they are to be made accessible via the
 * third-party P2P infrastructure.
 *
 * The vidmem allocation is made available to the users of the third-party P2P
 * APIs. It's exposed at the range specified by address and size starting at the
 * specified offset within the physical allocation. The same physical memory is
 * exposed as the LwRmMapMemoryDma() API would make accessible to the GPU if
 * used with equivalent parameters. Notably this API doesn't create any virtual
 * mappings nor verifies that any mappings are present, it only registers the
 * memory for the purpose of the third-party P2P infrastructure.
 *
 * The address range specified by address and size cannot overlap any previously
 * registered ranges for the given LW50_THIRD_PARTY_P2P object.
 *
 * Its parameter structure has the following field:
 *
 *   hMemory
 *      This field specifies the video memory allocation to be
 *      registered with the third-party P2P object.
 *
 *   address
 *      The address to register the video memory allocation at. Has to be
 *      aligned to 64K.
 *
 *   size
 *      Size in bytes, has to be non-0 and aligned to 64K. Offset + size cannot
 *      be larger than the vidmem allocation.
 *
 *   offset
 *      Offset within the video memory allocation where the registered address
 *      range starts. Has to be aligned to 64K.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_INSUFFICIENT_RESOURCES
 */
#define LW503C_CTRL_CMD_REGISTER_VIDMEM (0x503c0104) /* finn: Evaluated from "(FINN_LW50_THIRD_PARTY_P2P_P2P_INTERFACE_ID << 8) | LW503C_CTRL_REGISTER_VIDMEM_PARAMS_MESSAGE_ID" */

#define LW503C_CTRL_REGISTER_VIDMEM_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW503C_CTRL_REGISTER_VIDMEM_PARAMS {
    LwHandle hMemory;
    LW_DECLARE_ALIGNED(LwU64 address, 8);
    LW_DECLARE_ALIGNED(LwU64 size, 8);
    LW_DECLARE_ALIGNED(LwU64 offset, 8);
} LW503C_CTRL_REGISTER_VIDMEM_PARAMS;


/*
 * LW503C_CTRL_CMD_UNREGISTER_VIDMEM
 *
 * This command unregisters (a previously registered) video memory
 * allocation.
 *
 * Its parameter structure has the following field:
 *
 *   hMemory
 *      This field specifies the video memory allocation to be
 *      unregistered.
 *
 *  Possible status values returned are:
 *    LW_OK
 *    LW_ERR_ILWALID_ARGUMENT
 */
#define LW503C_CTRL_CMD_UNREGISTER_VIDMEM (0x503c0105) /* finn: Evaluated from "(FINN_LW50_THIRD_PARTY_P2P_P2P_INTERFACE_ID << 8) | LW503C_CTRL_UNREGISTER_VIDMEM_PARAMS_MESSAGE_ID" */

#define LW503C_CTRL_UNREGISTER_VIDMEM_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW503C_CTRL_UNREGISTER_VIDMEM_PARAMS {
    LwHandle hMemory;
} LW503C_CTRL_UNREGISTER_VIDMEM_PARAMS;

/*
 * LW503C_CTRL_CMD_REGISTER_PID
 * 
 * This command registers the PID of the process that allocated
 * the RM client identified by the hClient argument with the 
 * third-party P2P object, granting this process access to any
 * underlying video memory.
 *
 * Its parameter structure has the following field:
 *
 *   hClient
 *      This field specifies the client id and should be the handle
 *      to a valid LW01_ROOT_USER instance.
 *
 * Possible status values returned are:
 *    LWOS_STATUS_SUCCES
 *    LW_ERR_ILWALID_ARGUMENT
 */
#define LW503C_CTRL_CMD_REGISTER_PID (0x503c0106) /* finn: Evaluated from "(FINN_LW50_THIRD_PARTY_P2P_P2P_INTERFACE_ID << 8) | LW503C_CTRL_REGISTER_PID_PARAMS_MESSAGE_ID" */

#define LW503C_CTRL_REGISTER_PID_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW503C_CTRL_REGISTER_PID_PARAMS {
    LwHandle hClient;
} LW503C_CTRL_REGISTER_PID_PARAMS;

/* _ctrl503c_h_ */
