/*
 * SPDX-FileCopyrightText: Copyright (c) 2009-2021 LWPU CORPORATION & AFFILIATES. All rights reserved.
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
// Source file: ctrl/ctrl0000/ctrl0000unix.finn
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
/* LW01_ROOT (client) Linux control commands and parameters */

/*
 * LW0000_CTRL_CMD_OS_UNIX_FLUSH_USER_CACHE
 *
 * This command may be used to force a cache flush for a range of virtual addresses in 
 * memory. Can be used for either user or kernel addresses.
 *
 *   offset, length
 *     These parameters specify the offset within the memory block
 *     and the number of bytes to flush/ilwalidate
 *   cacheOps
 *     This parameter flags whether to flush, ilwalidate or do both.
 *     Possible values are:
 *       LW0000_CTRL_OS_UNIX_FLAGS_USER_CACHE_FLUSH
 *       LW0000_CTRL_OS_UNIX_FLAGS_USER_CACHE_ILWALIDATE
 *       LW0000_CTRL_OS_UNIX_FLAGS_USER_CACHE_FLUSH_ILWALIDATE
 *   hDevice
 *     This parameter is the handle to the device
 *   hObject
 *     This parameter is the handle to the memory structure being operated on.
 *   internalOnly
 *     Intended for internal use unless client is running in MODS UNIX environment, 
 *     in which case this parameter specify the virtual address of the memory block 
 *     to flush.
 *
 * Possible status values are:
 *   LW_OK
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_COMMAND
 *   LW_ERR_ILWALID_LIMIT
 *   LW_ERR_NOT_SUPPORTED
 */
#define LW0000_CTRL_CMD_OS_UNIX_FLUSH_USER_CACHE (0x3d02) /* finn: Evaluated from "(FINN_LW01_ROOT_OS_UNIX_INTERFACE_ID << 8) | LW0000_CTRL_OS_UNIX_FLUSH_USER_CACHE_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_OS_UNIX_FLUSH_USER_CACHE_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW0000_CTRL_OS_UNIX_FLUSH_USER_CACHE_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 offset, 8);
    LW_DECLARE_ALIGNED(LwU64 length, 8);
    LwU32    cacheOps;
    LwHandle hDevice;
    LwHandle hObject;
    LW_DECLARE_ALIGNED(LwU64 internalOnly, 8);
} LW0000_CTRL_OS_UNIX_FLUSH_USER_CACHE_PARAMS;

#define LW0000_CTRL_OS_UNIX_FLAGS_USER_CACHE_FLUSH            (0x00000001)
#define LW0000_CTRL_OS_UNIX_FLAGS_USER_CACHE_ILWALIDATE       (0x00000002)
#define LW0000_CTRL_OS_UNIX_FLAGS_USER_CACHE_FLUSH_ILWALIDATE (0x00000003)


/*
 * LW0000_CTRL_CMD_OS_UNIX_GET_CONTROL_FILE_DESCRIPTOR
 *
 * This command is used to get the control file descriptor.
 *
 * Possible status values returned are:
 *   LW_OK
 *
 */
#define LW0000_CTRL_CMD_OS_UNIX_GET_CONTROL_FILE_DESCRIPTOR   (0x3d04) /* finn: Evaluated from "(FINN_LW01_ROOT_OS_UNIX_INTERFACE_ID << 8) | 0x4" */

typedef struct LW0000_CTRL_OS_UNIX_GET_CONTROL_FILE_DESCRIPTOR_PARAMS {
    LwS32 fd;
} LW0000_CTRL_OS_UNIX_GET_CONTROL_FILE_DESCRIPTOR_PARAMS;

typedef enum LW0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE {
    LW0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_NONE = 0,
    LW0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM = 1,
} LW0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE;

typedef struct LW0000_CTRL_OS_UNIX_EXPORT_OBJECT {
    LW0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE type;

    union {
        struct {
            LwHandle hDevice;
            LwHandle hParent;
            LwHandle hObject;
        } rmObject;
    } data;
} LW0000_CTRL_OS_UNIX_EXPORT_OBJECT;

/*
 * LW0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD
 *
 * This command may be used to export LW0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE
 * object to file descriptor.
 *
 * Note that the 'fd' parameter is an input parameter at the kernel level, but
 * an output parameter for usermode RMAPI clients -- the RMAPI library will
 * open a new FD automatically if a usermode RMAPI client exports an object.
 *
 * Kernel-mode RM clients can export an object to an FD in two steps:
 * 1. User client calls this RMControl with the flag 'EMPTY_FD_TRUE' to create
 *    an empty FD to receive the object, then passes that FD to the kernel-mode
 *    RM client.
 * 2. Kernel-mode RM client fills in the rest of the
 *    LW0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_PARAMS as usual and calls RM to
 *    associate its desired RM object with the empty FD from its usermode
 *    client.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_PARAMETER
 */
#define LW0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD (0x3d05) /* finn: Evaluated from "(FINN_LW01_ROOT_OS_UNIX_INTERFACE_ID << 8) | LW0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_PARAMS_MESSAGE_ID (0x5U)

typedef struct LW0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_PARAMS {
    LW0000_CTRL_OS_UNIX_EXPORT_OBJECT object; /* IN */
    LwS32                             fd;                                 /* IN/OUT */
    LwU32                             flags;                              /* IN */
} LW0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_PARAMS;

/*
 * If EMPTY_FD is TRUE, the 'fd' will be created but no object will be
 * associated with it.  The hDevice parameter is still required, to determine
 * the correct device node on which to create the file descriptor.
 * (An empty FD can then be passed to a kernel-mode driver to associate it with
 * an actual object.)
 */
#define LW0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_FLAGS_EMPTY_FD       0:0
#define LW0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_FLAGS_EMPTY_FD_FALSE (0x00000000)
#define LW0000_CTRL_OS_UNIX_EXPORT_OBJECT_TO_FD_FLAGS_EMPTY_FD_TRUE  (0x00000001)

/*
 * LW0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD
 *
 * This command may be used to import back
 * LW0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE object from file descriptor.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_PARAMETER
 */
#define LW0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD                (0x3d06) /* finn: Evaluated from "(FINN_LW01_ROOT_OS_UNIX_INTERFACE_ID << 8) | LW0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS {
    LwS32                             fd;                                 /* IN */
    LW0000_CTRL_OS_UNIX_EXPORT_OBJECT object; /* IN */
} LW0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS;

/*
 * LW0000_CTRL_CMD_OS_GET_GPU_INFO
 *
 * This command will query the OS specific info for the specified GPU.
 *
 *  gpuId
 *    This parameter should specify a valid GPU ID value.  If there
 *    is no GPU present with the specified ID, a status of
 *    LW_ERR_ILWALID_ARGUMENT is returned.
 *  minorNum
 *    This parameter returns minor number of device node.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 */

#define LW0000_CTRL_CMD_OS_GET_GPU_INFO (0x3d07) /* finn: Evaluated from "(FINN_LW01_ROOT_OS_UNIX_INTERFACE_ID << 8) | 0x7" */

typedef struct LW0000_CTRL_OS_GET_GPU_INFO_PARAMS {
    LwU32 gpuId;    /* IN */
    LwU32 minorNum; /* OUT */
} LW0000_CTRL_OS_GET_GPU_INFO_PARAMS;

/*
 * LW0000_CTRL_CMD_OS_UNIX_GET_EXPORT_OBJECT_INFO
 *
 * This command will query the deviceInstance for the specified FD
 * which is referencing an exported object.
 *
 *  fd
 *    File descriptor parameter is referencing an exported object on a Unix system.
 *
 *  deviceInstatnce
 *    This parameter returns a deviceInstance on which the object is located.
 *
 *  maxObjects
 *    This parameter returns the maximum number of object handles that may be
 *    contained in the file descriptor.
 *
 *  metadata
 *    This parameter returns the user metadata passed into the
 *    _EXPORT_OBJECTS_TO_FD control call.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_ARGUMENT
 *   LW_ERR_OBJECT_NOT_FOUND
 */

#define LW0000_CTRL_CMD_OS_UNIX_GET_EXPORT_OBJECT_INFO (0x3d08) /* finn: Evaluated from "(FINN_LW01_ROOT_OS_UNIX_INTERFACE_ID << 8) | LW0000_CTRL_OS_UNIX_GET_EXPORT_OBJECT_INFO_PARAMS_MESSAGE_ID" */

#define LW0000_OS_UNIX_EXPORT_OBJECT_FD_BUFFER_SIZE    64

#define LW0000_CTRL_OS_UNIX_GET_EXPORT_OBJECT_INFO_PARAMS_MESSAGE_ID (0x8U)

typedef struct LW0000_CTRL_OS_UNIX_GET_EXPORT_OBJECT_INFO_PARAMS {
    LwS32 fd;               /* IN  */
    LwU32 deviceInstance;   /* OUT */
    LwU16 maxObjects;       /* OUT */
    LwU8  metadata[LW0000_OS_UNIX_EXPORT_OBJECT_FD_BUFFER_SIZE]; /* OUT */
} LW0000_CTRL_OS_UNIX_GET_EXPORT_OBJECT_INFO_PARAMS;

/*
 * LW0000_CTRL_CMD_OS_UNIX_REFRESH_RMAPI_DEVICE_LIST
 *
 * This command will re-fetch probed GPUs information and update RMAPI library's
 * internal detected GPU context information accordingly. Without this, GPUs
 * attached to RM after RMAPI client initialization will not be accessible and
 * all RMAPI library calls will fail on them.
 * Lwrrently this is used by LWSwitch Fabric Manager in conjunction with LWSwitch
 * Shared Virtualization feature where GPUs are hot-plugged to OS/RM (by Hypervisor)
 * and Fabric Manager is signaled externally by the Hypervisor to initialize those GPUs.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_ILWALID_STATE
 *   LW_ERR_OPERATING_SYSTEM
 */

#define LW0000_CTRL_CMD_OS_UNIX_REFRESH_RMAPI_DEVICE_LIST       (0x3d09) /* finn: Evaluated from "(FINN_LW01_ROOT_OS_UNIX_INTERFACE_ID << 8) | 0x9" */

/*
 * This control call has been deprecated. It will be deleted soon.
 * Use LW0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECT_TO_FD (singular) or
 * LW0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECTS_TO_FD (plural) instead.
 */
#define LW0000_CTRL_CMD_OS_UNIX_CREATE_EXPORT_OBJECT_FD         (0x3d0a) /* finn: Evaluated from "(FINN_LW01_ROOT_OS_UNIX_INTERFACE_ID << 8) | LW0000_CTRL_OS_UNIX_CREATE_EXPORT_OBJECT_FD_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_OS_UNIX_CREATE_EXPORT_OBJECT_FD_BUFFER_SIZE LW0000_OS_UNIX_EXPORT_OBJECT_FD_BUFFER_SIZE

#define LW0000_CTRL_OS_UNIX_CREATE_EXPORT_OBJECT_FD_PARAMS_MESSAGE_ID (0xAU)

typedef struct LW0000_CTRL_OS_UNIX_CREATE_EXPORT_OBJECT_FD_PARAMS {
    LwHandle hDevice;                                                       /* IN */
    LwU16    maxObjects;                                                       /* IN */
    LwU8     metadata[LW0000_CTRL_OS_UNIX_CREATE_EXPORT_OBJECT_FD_BUFFER_SIZE]; /* IN */
    LwS32    fd;                                                               /* IN/OUT */
} LW0000_CTRL_OS_UNIX_CREATE_EXPORT_OBJECT_FD_PARAMS;

/*
 * LW0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECTS_TO_FD
 *
 * Exports RM handles to an fd that was provided, also creates an FD if
 * requested.
 *
 * The objects in the 'handles' array are exported into the fd
 * as the range [index, index + numObjects).
 *
 * If index + numObjects is greater than the maxObjects value used
 * to create the file descriptor, LW_ERR_ILWALID_PARAMETER is returned.
 *
 * If 'numObjects and 'index' overlap with a prior call, the newer call's RM object
 * handles will overwrite the previously exported handles from the previous call.
 * This overlapping behavior can also be used to unexport a handle by setting
 * the appropriate object in 'objects' to 0.
 *
 *  fd
 *    A file descriptor. If -1, a new FD will be created.
 *
 *  hDevice
 *    The owning device of the objects to be exported (must be the same for
 *    all objects).
 *
 *  maxObjects
 *    The total number of objects that the client wishes to export to the FD.
 *    This parameter will be honored only when the FD is getting created.
 *
 *  metadata
 *    A buffer for clients to write some metadata to and pass to the importing
 *    client. This parameter will be honored only when the FD is getting
 *    created.
 *
 *  objects
 *    Array of RM object handles to export to the fd.
 *
 *  numObjects
 *    The number of handles the user wishes to export in this call.
 *
 *  index
 *    The index into the export fd at which to start exporting the handles in
 *    'objects' (for use in iterative calls).
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OUT_OF_RANGE
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 */
#define LW0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECTS_TO_FD         (0x3d0b) /* finn: Evaluated from "(FINN_LW01_ROOT_OS_UNIX_INTERFACE_ID << 8) | LW0000_CTRL_OS_UNIX_EXPORT_OBJECTS_TO_FD_PARAMS_MESSAGE_ID" */

#define LW0000_CTRL_OS_UNIX_EXPORT_OBJECTS_TO_FD_MAX_OBJECTS 512

#define LW0000_CTRL_OS_UNIX_EXPORT_OBJECTS_TO_FD_PARAMS_MESSAGE_ID (0xBU)

typedef struct LW0000_CTRL_OS_UNIX_EXPORT_OBJECTS_TO_FD_PARAMS {
    LwS32    fd;                                                               /* IN/OUT */
    LwHandle hDevice;                                                       /* IN */
    LwU16    maxObjects;                                                       /* IN */
    LwU8     metadata[LW0000_OS_UNIX_EXPORT_OBJECT_FD_BUFFER_SIZE];             /* IN */
    LwHandle objects[LW0000_CTRL_OS_UNIX_EXPORT_OBJECTS_TO_FD_MAX_OBJECTS]; /* IN */
    LwU16    numObjects;                                                       /* IN */
    LwU16    index;                                                            /* IN */
} LW0000_CTRL_OS_UNIX_EXPORT_OBJECTS_TO_FD_PARAMS;

/*
 * LW0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECTS_FROM_FD
 *
 * This command can be used to import back RM handles
 * that were exported to an fd using the
 * LW0000_CTRL_CMD_OS_UNIX_EXPORT_OBJECTS_TO_FD control call.
 *
 * If index + numObjects is greater than the maxObjects value used
 * to create the file descriptor, LW_ERR_ILWALID_PARAMETER is returned
 * and no objects are imported.
 *
 * For each valid handle in the 'objects' array parameter at index 'i',
 * the corresponding object handle at index ('i' + 'index') contained by
 * the fd will be imported. If the object at index ('i' + 'index') has
 * not been exported into the fd, no object will be imported.
 *
 * If any of handles contained in the 'objects' array parameter are invalid
 * and the corresponding export object handle is valid,
 * LW_ERR_ILWALID_PARAMETER will be returned and no handles will be imported.
 *
 *  fd
 *    The export fd on which to import handles out of.
 *
 *  hParent
 *    The parent RM handle of which all of the exported objects will
 *    be duped under.
 *
 *  objects
 *    An array of RM handles. The exported objects will be duped under
 *    these handles during the import process.
 *
 * objectTypes
 *    An array of RM handle types. The type _NONE will be returned if
 *    the object was not imported. Other possible object types are
 *    mentioned below.
 *
 *  numObjects
 *    The number of valid object handles in the 'objects' array. This should
 *    be set to the number of objects that the client wishes to import.
 *
 *  index
 *    The index into the fd in which to start importing from. For
 *    use in iterative calls.
 *
 * Possible status values returned are:
 *   LW_OK
 *   LW_ERR_OUT_OF_RANGE
 *   LW_ERR_NOT_SUPPORTED
 *   LW_ERR_OBJECT_NOT_FOUND
 *   LW_ERR_ILWALID_OBJECT_HANDLE
 *   LW_ERR_ILWALID_PARAMETER
 */
#define LW0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECTS_FROM_FD       (0x3d0c) /* finn: Evaluated from "(FINN_LW01_ROOT_OS_UNIX_INTERFACE_ID << 8) | LW0000_CTRL_OS_UNIX_IMPORT_OBJECTS_FROM_FD_PARAMS_MESSAGE_ID" */

//
// TODO Bump this back up to 512 after the FLA revamp is complete
//
#define LW0000_CTRL_OS_UNIX_IMPORT_OBJECTS_TO_FD_MAX_OBJECTS 128

#define LW0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_TYPE_NONE      0
#define LW0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_TYPE_VIDMEM    1
#define LW0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_TYPE_SYSMEM    2
#define LW0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_TYPE_FABRIC    3

#define LW0000_CTRL_OS_UNIX_IMPORT_OBJECTS_FROM_FD_PARAMS_MESSAGE_ID (0xLW)

typedef struct LW0000_CTRL_OS_UNIX_IMPORT_OBJECTS_FROM_FD_PARAMS {
    LwS32    fd;                                                                   /* IN  */
    LwHandle hParent;                                                           /* IN  */
    LwHandle objects[LW0000_CTRL_OS_UNIX_IMPORT_OBJECTS_TO_FD_MAX_OBJECTS];     /* IN  */
    LwU8     objectTypes[LW0000_CTRL_OS_UNIX_IMPORT_OBJECTS_TO_FD_MAX_OBJECTS];     /* OUT */
    LwU16    numObjects;                                                           /* IN  */
    LwU16    index;                                                                /* IN  */
} LW0000_CTRL_OS_UNIX_IMPORT_OBJECTS_FROM_FD_PARAMS;

/* _ctrl0000unix_h_ */
