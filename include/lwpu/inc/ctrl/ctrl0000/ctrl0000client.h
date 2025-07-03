/*
 * SPDX-FileCopyrightText: Copyright (c) 2006-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl0000/ctrl0000client.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl0000/ctrl0000base.h"

#include "ctrl/ctrlxxxx.h"
#include "class/cl0000.h"
#include "rs_access.h"

/*
 * LW0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE
 *
 * This command may be used to query memory address space type associated with an object 
 *
 * Parameters:
 *    hObject[IN]
 *     handle of the object to look up
 *    addrSpaceType[OUT]
 *     addrSpaceType with associated memory descriptor
 *
 * Possible status values are:
 *   LW_OK
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_OBJECT
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE (0xd01) /* finn: Evaluated from "(FINN_LW01_ROOT_CLIENT_INTERFACE_ID << 8) | LW0000_CTRL_CLIENT_GET_ADDR_SPACE_TYPE_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_CLIENT_GET_ADDR_SPACE_TYPE_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW0000_CTRL_CLIENT_GET_ADDR_SPACE_TYPE_PARAMS {
    LwHandle hObject;                /* [in]  - Handle of object to look up */
    LwU32    mapFlags;                  /* [in]  - Flags that will be used when mapping the object */
    LwU32    addrSpaceType;             /* [out] - Memory Address Space Type */
} LW0000_CTRL_CLIENT_GET_ADDR_SPACE_TYPE_PARAMS;

#define LW0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE_ILWALID 0x00000000
#define LW0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE_SYSMEM  0x00000001
#define LW0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE_VIDMEM  0x00000002
#define LW0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE_REGMEM  0x00000003
#define LW0000_CTRL_CMD_CLIENT_GET_ADDR_SPACE_TYPE_FABRIC  0x00000004

/*
 * LW0000_CTRL_CMD_CLIENT_GET_HANDLE_INFO
 *
 * This command may be used to query information on a handle
 */
#define LW0000_CTRL_CMD_CLIENT_GET_HANDLE_INFO             (0xd02) /* finn: Evaluated from "(FINN_LW01_ROOT_CLIENT_INTERFACE_ID << 8) | LW0000_CTRL_CLIENT_GET_HANDLE_INFO_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_CLIENT_GET_HANDLE_INFO_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0000_CTRL_CLIENT_GET_HANDLE_INFO_PARAMS {
    LwHandle hObject;         /* [in]  - Handle of object to look up */
    LwU32    index;           /* [in]  - Type of lookup */

    union {
        LwHandle hResult; /* [out] - Result of lookup when result is a handle type */
        LW_DECLARE_ALIGNED(LwU64 iResult, 8); /* [out] - Result of lookup when result is a integer */
    } data;
} LW0000_CTRL_CLIENT_GET_HANDLE_INFO_PARAMS;

#define LW0000_CTRL_CMD_CLIENT_GET_HANDLE_INFO_INDEX_ILWALID 0x00000000
#define LW0000_CTRL_CMD_CLIENT_GET_HANDLE_INFO_INDEX_PARENT  0x00000001
#define LW0000_CTRL_CMD_CLIENT_GET_HANDLE_INFO_INDEX_CLASSID 0x00000002

/*
 * LW0000_CTRL_CMD_CLIENT_GET_ACCESS_RIGHTS
 *
 * This command may be used to get this client's access rights for an object
 * The object to which access rights are checked does not have to be owned by
 * the client calling the command, it is owned by the hClient parameter
 */
#define LW0000_CTRL_CMD_CLIENT_GET_ACCESS_RIGHTS             (0xd03) /* finn: Evaluated from "(FINN_LW01_ROOT_CLIENT_INTERFACE_ID << 8) | LW0000_CTRL_CLIENT_GET_ACCESS_RIGHTS_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_CLIENT_GET_ACCESS_RIGHTS_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW0000_CTRL_CLIENT_GET_ACCESS_RIGHTS_PARAMS {
    LwHandle       hObject;                /* [in]  - Handle of object to look up */
    LwHandle       hClient;                /* [in]  - Handle of client which owns hObject */
    RS_ACCESS_MASK maskResult;       /* [out] - Result of lookup */
} LW0000_CTRL_CLIENT_GET_ACCESS_RIGHTS_PARAMS;

/*
 * LW0000_CTRL_CMD_CLIENT_SET_INHERITED_SHARE_POLICY
 *
 * DEPRECATED: Calls LW0000_CTRL_CMD_CLIENT_SHARE_OBJECT with hObject=hClient
 *
 * This command will modify a client's inherited share policy list
 * The policy is applied in the same way that LwRmShare applies policies,
 * except to the client's inherited policy list instead of an object's policy list
 */
#define LW0000_CTRL_CMD_CLIENT_SET_INHERITED_SHARE_POLICY (0xd04) /* finn: Evaluated from "(FINN_LW01_ROOT_CLIENT_INTERFACE_ID << 8) | LW0000_CTRL_CLIENT_SET_INHERITED_SHARE_POLICY_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_CLIENT_SET_INHERITED_SHARE_POLICY_PARAMS_MESSAGE_ID (0x4U)

typedef struct LW0000_CTRL_CLIENT_SET_INHERITED_SHARE_POLICY_PARAMS {
    RS_SHARE_POLICY sharePolicy;       /* [in] - Share Policy to apply */
} LW0000_CTRL_CLIENT_SET_INHERITED_SHARE_POLICY_PARAMS;

/*
 * LW0000_CTRL_CMD_CLIENT_GET_CHILD_HANDLE
 *
 * This command may be used to get a handle of a child of a given type
 */
#define LW0000_CTRL_CMD_CLIENT_GET_CHILD_HANDLE (0xd05) /* finn: Evaluated from "(FINN_LW01_ROOT_CLIENT_INTERFACE_ID << 8) | LW0000_CTRL_CMD_CLIENT_GET_CHILD_HANDLE_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_CMD_CLIENT_GET_CHILD_HANDLE_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW0000_CTRL_CMD_CLIENT_GET_CHILD_HANDLE_PARAMS {
    LwHandle hParent;                /* [in]  - Handle of parent object */
    LwU32    classId;                /* [in]  - Class ID of the child object */
    LwHandle hObject;                /* [out] - Handle of the child object (0 if not found) */
} LW0000_CTRL_CMD_CLIENT_GET_CHILD_HANDLE_PARAMS;

/*
 * LW0000_CTRL_CMD_CLIENT_SHARE_OBJECT
 *
 * This command is meant to imitate the LwRmShare API.
 * Applies a share policy to an object, which should be owned by the caller's client.
 * The policy is applied in the same way that LwRmShare applies policies.
 *
 * This ctrl command is only meant to be used in older branches. For releases after R450,
 * use LwRmShare directly instead.
 */
#define LW0000_CTRL_CMD_CLIENT_SHARE_OBJECT (0xd06) /* finn: Evaluated from "(FINN_LW01_ROOT_CLIENT_INTERFACE_ID << 8) | LW0000_CTRL_CLIENT_SHARE_OBJECT_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_CLIENT_SHARE_OBJECT_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW0000_CTRL_CLIENT_SHARE_OBJECT_PARAMS {
    LwHandle        hObject;                /* [in]  - Handle of object to share */
    RS_SHARE_POLICY sharePolicy;     /* [in]  - Share Policy to apply */
} LW0000_CTRL_CLIENT_SHARE_OBJECT_PARAMS;

/* _ctrl0000client_h_ */

