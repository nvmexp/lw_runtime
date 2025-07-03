/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
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

 /***************************************************************************\
|*                                                                           *|
|*                         LW Architecture Interface                         *|
|*                                                                           *|
|*  <lwos.h>  defines the Operating System function and ioctl interfaces to  *|
|*  LWPU's Unified Media Architecture (TM).                                *|
|*                                                                           *|
 \***************************************************************************/

#ifndef LWOS_INCLUDED
#define LWOS_INCLUDED
#ifdef __cplusplus
extern "C" {
#endif

#include "lwstatus.h"

#include "lwgputypes.h"
#include "rs_access.h"

/* local defines here */
#define FILE_DEVICE_LW      0x00008000
#define LW_IOCTL_FCT_BASE   0x00000800

// This is the maximum number of subdevices supported in an SLI
// configuration.
#define LWOS_MAX_SUBDEVICES 8

/* Define to indicate the use of Unified status codes - bug 200043705*/
#define UNIFIED_LW_STATUS 1

 /***************************************************************************\
|*                              LW OS Functions                              *|
 \***************************************************************************/

/*
   Result codes for RM APIs, shared for all the APIs

   *** IMPORTANT ***

   Ensure that no LWOS_STATUS value has the highest bit set. That bit
   is used to passthrough the LWOS_STATUS on code expecting an RM_STATUS.
*/
#define LWOS_STATUS LW_STATUS

#define LWOS_STATUS_SUCCESS                                     LW_OK
#define LWOS_STATUS_ERROR_CARD_NOT_PRESENT                      LW_ERR_CARD_NOT_PRESENT
#define LWOS_STATUS_ERROR_DUAL_LINK_INUSE                       LW_ERR_DUAL_LINK_INUSE
#define LWOS_STATUS_ERROR_GENERIC                               LW_ERR_GENERIC
#define LWOS_STATUS_ERROR_GPU_NOT_FULL_POWER                    LW_ERR_GPU_NOT_FULL_POWER
#define LWOS_STATUS_ERROR_ILLEGAL_ACTION                        LW_ERR_ILLEGAL_ACTION
#define LWOS_STATUS_ERROR_IN_USE                                LW_ERR_STATE_IN_USE
#define LWOS_STATUS_ERROR_INSUFFICIENT_RESOURCES                LW_ERR_INSUFFICIENT_RESOURCES
#define LWOS_STATUS_ERROR_ILWALID_ACCESS_TYPE                   LW_ERR_ILWALID_ACCESS_TYPE
#define LWOS_STATUS_ERROR_ILWALID_ARGUMENT                      LW_ERR_ILWALID_ARGUMENT
#define LWOS_STATUS_ERROR_ILWALID_BASE                          LW_ERR_ILWALID_BASE
#define LWOS_STATUS_ERROR_ILWALID_CHANNEL                       LW_ERR_ILWALID_CHANNEL
#define LWOS_STATUS_ERROR_ILWALID_CLASS                         LW_ERR_ILWALID_CLASS
#define LWOS_STATUS_ERROR_ILWALID_CLIENT                        LW_ERR_ILWALID_CLIENT
#define LWOS_STATUS_ERROR_ILWALID_COMMAND                       LW_ERR_ILWALID_COMMAND
#define LWOS_STATUS_ERROR_ILWALID_DATA                          LW_ERR_ILWALID_DATA
#define LWOS_STATUS_ERROR_ILWALID_DEVICE                        LW_ERR_ILWALID_DEVICE
#define LWOS_STATUS_ERROR_ILWALID_DMA_SPECIFIER                 LW_ERR_ILWALID_DMA_SPECIFIER
#define LWOS_STATUS_ERROR_ILWALID_EVENT                         LW_ERR_ILWALID_EVENT
#define LWOS_STATUS_ERROR_ILWALID_FLAGS                         LW_ERR_ILWALID_FLAGS
#define LWOS_STATUS_ERROR_ILWALID_FUNCTION                      LW_ERR_ILWALID_FUNCTION
#define LWOS_STATUS_ERROR_ILWALID_HEAP                          LW_ERR_ILWALID_HEAP
#define LWOS_STATUS_ERROR_ILWALID_INDEX                         LW_ERR_ILWALID_INDEX
#define LWOS_STATUS_ERROR_ILWALID_LIMIT                         LW_ERR_ILWALID_LIMIT
#define LWOS_STATUS_ERROR_ILWALID_METHOD                        LW_ERR_ILWALID_METHOD
#define LWOS_STATUS_ERROR_ILWALID_OBJECT_BUFFER                 LW_ERR_BUFFER_TOO_SMALL
#define LWOS_STATUS_ERROR_ILWALID_OBJECT_ERROR                  LW_ERR_ILWALID_OBJECT
#define LWOS_STATUS_ERROR_ILWALID_OBJECT_HANDLE                 LW_ERR_ILWALID_OBJECT_HANDLE
#define LWOS_STATUS_ERROR_ILWALID_OBJECT_NEW                    LW_ERR_ILWALID_OBJECT_NEW
#define LWOS_STATUS_ERROR_ILWALID_OBJECT_OLD                    LW_ERR_ILWALID_OBJECT_OLD
#define LWOS_STATUS_ERROR_ILWALID_OBJECT_PARENT                 LW_ERR_ILWALID_OBJECT_PARENT
#define LWOS_STATUS_ERROR_ILWALID_OFFSET                        LW_ERR_ILWALID_OFFSET
#define LWOS_STATUS_ERROR_ILWALID_OWNER                         LW_ERR_ILWALID_OWNER
#define LWOS_STATUS_ERROR_ILWALID_PARAM_STRUCT                  LW_ERR_ILWALID_PARAM_STRUCT
#define LWOS_STATUS_ERROR_ILWALID_PARAMETER                     LW_ERR_ILWALID_PARAMETER
#define LWOS_STATUS_ERROR_ILWALID_POINTER                       LW_ERR_ILWALID_POINTER
#define LWOS_STATUS_ERROR_ILWALID_REGISTRY_KEY                  LW_ERR_ILWALID_REGISTRY_KEY
#define LWOS_STATUS_ERROR_ILWALID_STATE                         LW_ERR_ILWALID_STATE
#define LWOS_STATUS_ERROR_ILWALID_STRING_LENGTH                 LW_ERR_ILWALID_STRING_LENGTH
#define LWOS_STATUS_ERROR_ILWALID_XLATE                         LW_ERR_ILWALID_XLATE
#define LWOS_STATUS_ERROR_IRQ_NOT_FIRING                        LW_ERR_IRQ_NOT_FIRING
#define LWOS_STATUS_ERROR_MULTIPLE_MEMORY_TYPES                 LW_ERR_MULTIPLE_MEMORY_TYPES
#define LWOS_STATUS_ERROR_NOT_SUPPORTED                         LW_ERR_NOT_SUPPORTED
#define LWOS_STATUS_ERROR_OPERATING_SYSTEM                      LW_ERR_OPERATING_SYSTEM
#define LWOS_STATUS_ERROR_LIB_RM_VERSION_MISMATCH               LW_ERR_LIB_RM_VERSION_MISMATCH
#define LWOS_STATUS_ERROR_PROTECTION_FAULT                      LW_ERR_PROTECTION_FAULT
#define LWOS_STATUS_ERROR_TIMEOUT                               LW_ERR_TIMEOUT
#define LWOS_STATUS_ERROR_TOO_MANY_PRIMARIES                    LW_ERR_TOO_MANY_PRIMARIES
#define LWOS_STATUS_ERROR_IRQ_EDGE_TRIGGERED                    LW_ERR_IRQ_EDGE_TRIGGERED
#define LWOS_STATUS_ERROR_ILWALID_OPERATION                     LW_ERR_ILWALID_OPERATION
#define LWOS_STATUS_ERROR_NOT_COMPATIBLE                        LW_ERR_NOT_COMPATIBLE
#define LWOS_STATUS_ERROR_MORE_PROCESSING_REQUIRED              LW_WARN_MORE_PROCESSING_REQUIRED
#define LWOS_STATUS_ERROR_INSUFFICIENT_PERMISSIONS              LW_ERR_INSUFFICIENT_PERMISSIONS
#define LWOS_STATUS_ERROR_TIMEOUT_RETRY                         LW_ERR_TIMEOUT_RETRY
#define LWOS_STATUS_ERROR_NOT_READY                             LW_ERR_NOT_READY
#define LWOS_STATUS_ERROR_GPU_IS_LOST                           LW_ERR_GPU_IS_LOST
#define LWOS_STATUS_ERROR_IN_FULLCHIP_RESET                     LW_ERR_GPU_IN_FULLCHIP_RESET
#define LWOS_STATUS_ERROR_ILWALID_LOCK_STATE                    LW_ERR_ILWALID_LOCK_STATE
#define LWOS_STATUS_ERROR_ILWALID_ADDRESS                       LW_ERR_ILWALID_ADDRESS
#define LWOS_STATUS_ERROR_ILWALID_IRQ_LEVEL                     LW_ERR_ILWALID_IRQ_LEVEL
#define LWOS_STATUS_ERROR_MEMORY_TRAINING_FAILED                LW_ERR_MEMORY_TRAINING_FAILED
#define LWOS_STATUS_ERROR_BUSY_RETRY                            LW_ERR_BUSY_RETRY
#define LWOS_STATUS_ERROR_INSUFFICIENT_POWER                    LW_ERR_INSUFFICIENT_POWER
#define LWOS_STATUS_ERROR_OBJECT_NOT_FOUND                      LW_ERR_OBJECT_NOT_FOUND
#define LWOS_STATUS_ERROR_RESOURCE_LOST                         LW_ERR_RESOURCE_LOST
#define LWOS_STATUS_ERROR_BUFFER_TOO_SMALL                      LW_ERR_BUFFER_TOO_SMALL
#define LWOS_STATUS_ERROR_RESET_REQUIRED                        LW_ERR_RESET_REQUIRED
#define LWOS_STATUS_ERROR_ILWALID_REQUEST                       LW_ERR_ILWALID_REQUEST

#define LWOS_STATUS_ERROR_PRIV_SEC_VIOLATION                    LW_ERR_PRIV_SEC_VIOLATION
#define LWOS_STATUS_ERROR_GPU_IN_DEBUG_MODE                     LW_ERR_GPU_IN_DEBUG_MODE

/*
    Note:
        This version of the architecture has been changed to allow the
        RM to return a client handle that will subsequently used to
        identify the client.  LwAllocRoot() returns the handle.  All
        other functions must specify this client handle.

*/
/* macro LW01_FREE */
#define  LW01_FREE                                                 (0x00000000)

/* NT ioctl data structure */
typedef struct
{
  LwHandle  hRoot;
  LwHandle  hObjectParent;
  LwHandle  hObjectOld;
  LwV32     status;
} LWOS00_PARAMETERS;

/* valid hClass values. */
#define  LW01_ROOT                                                 (0x00000000)
//
// Redefining it here to maintain consistency with current code
// This is also defined in class cl0001.h
//
#define  LW01_ROOT_NON_PRIV                                        (0x00000001)

// Deprecated, please use LW01_ROOT_CLIENT
#define  LW01_ROOT_USER                                            LW01_ROOT_CLIENT

//
// This will eventually replace LW01_ROOT_USER in RM client code. Please use this
// RM client object type for any new RM client object allocations that are being
// added.
//
#define  LW01_ROOT_CLIENT                                          (0x00000041)

/* macro LW01_ALLOC_MEMORY */
#define  LW01_ALLOC_MEMORY                                         (0x00000002)

/* parameter values */
#define LWOS02_FLAGS_PHYSICALITY                                   7:4
#define LWOS02_FLAGS_PHYSICALITY_CONTIGUOUS                        (0x00000000)
#define LWOS02_FLAGS_PHYSICALITY_NONCONTIGUOUS                     (0x00000001)
#define LWOS02_FLAGS_LOCATION                                      11:8
#define LWOS02_FLAGS_LOCATION_PCI                                  (0x00000000)
#define LWOS02_FLAGS_LOCATION_AGP                                  (0x00000001)
#define LWOS02_FLAGS_LOCATION_VIDMEM                               (0x00000002)
#define LWOS02_FLAGS_COHERENCY                                     15:12
#define LWOS02_FLAGS_COHERENCY_UNCACHED                            (0x00000000)
#define LWOS02_FLAGS_COHERENCY_CACHED                              (0x00000001)
#define LWOS02_FLAGS_COHERENCY_WRITE_COMBINE                       (0x00000002)
#define LWOS02_FLAGS_COHERENCY_WRITE_THROUGH                       (0x00000003)
#define LWOS02_FLAGS_COHERENCY_WRITE_PROTECT                       (0x00000004)
#define LWOS02_FLAGS_COHERENCY_WRITE_BACK                          (0x00000005)
#define LWOS02_FLAGS_ALLOC                                         17:16
#define LWOS02_FLAGS_ALLOC_NONE                                    (0x00000001)
#define LWOS02_FLAGS_GPU_CACHEABLE                                 18:18
#define LWOS02_FLAGS_GPU_CACHEABLE_NO                              (0x00000000)
#define LWOS02_FLAGS_GPU_CACHEABLE_YES                             (0x00000001)
// If requested, RM will create a kernel mapping of this memory.
// Default is no map.
#define LWOS02_FLAGS_KERNEL_MAPPING                                19:19
#define LWOS02_FLAGS_KERNEL_MAPPING_NO_MAP                         (0x00000000)
#define LWOS02_FLAGS_KERNEL_MAPPING_MAP                            (0x00000001)
#define LWOS02_FLAGS_ALLOC_NISO_DISPLAY                            20:20
#define LWOS02_FLAGS_ALLOC_NISO_DISPLAY_NO                         (0x00000000)
#define LWOS02_FLAGS_ALLOC_NISO_DISPLAY_YES                        (0x00000001)

//
// If the flag is set, the RM will only allow read-only CPU user mappings to the
// allocation.
//
#define LWOS02_FLAGS_ALLOC_USER_READ_ONLY                          21:21
#define LWOS02_FLAGS_ALLOC_USER_READ_ONLY_NO                       (0x00000000)
#define LWOS02_FLAGS_ALLOC_USER_READ_ONLY_YES                      (0x00000001)

//
// If the flag is set, the RM will only allow read-only DMA mappings to the
// allocation.
//
#define LWOS02_FLAGS_ALLOC_DEVICE_READ_ONLY                        22:22
#define LWOS02_FLAGS_ALLOC_DEVICE_READ_ONLY_NO                     (0x00000000)
#define LWOS02_FLAGS_ALLOC_DEVICE_READ_ONLY_YES                    (0x00000001)

//
// If the flag is set, the IO memory allocation can be registered with the RM if
// the RM regkey peerMappingOverride is set or the client is privileged.
//
// See Bug 1630288 "[PeerSync] threat related to GPU.." for more details.
//
#define LWOS02_FLAGS_PEER_MAP_OVERRIDE                             23:23
#define LWOS02_FLAGS_PEER_MAP_OVERRIDE_DEFAULT                     (0x00000000)
#define LWOS02_FLAGS_PEER_MAP_OVERRIDE_REQUIRED                    (0x00000001)

// If the flag is set RM will assume the memory pages are of type syncpoint.
#define LWOS02_FLAGS_ALLOC_TYPE_SYNCPOINT                          24:24
#define LWOS02_FLAGS_ALLOC_TYPE_SYNCPOINT_APERTURE                 (0x00000001)

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Allow client allocations to go to protected/unprotected video/system memory.
// When Ampere Protected Model aka APM or Confidential Compute is enabled and
// DEFAULT flag is set by client, allocations go to protected memory. When
// protected memory is not enabled, allocations go to unprotected memory.
// If APM or CC is not enabled, it is a bug for a client to set the PROTECTED
// flag to YES
//
#define LWOS02_FLAGS_MEMORY_PROTECTION                             26:25
#define LWOS02_FLAGS_MEMORY_PROTECTION_DEFAULT                     (0x00000000)
#define LWOS02_FLAGS_MEMORY_PROTECTION_PROTECTED                   (0x00000001)
#define LWOS02_FLAGS_MEMORY_PROTECTION_UNPROTECTED                 (0x00000002)
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

//
// If _NO_MAP is requested, the RM in supported platforms will not map the
// allocated system or IO memory into user space. The client can later map
// memory through the RmMapMemory() interface.
// If _NEVER_MAP is requested, the RM will never map the allocated system or
// IO memory into user space
//
#define LWOS02_FLAGS_MAPPING                                       31:30
#define LWOS02_FLAGS_MAPPING_DEFAULT                               (0x00000000)
#define LWOS02_FLAGS_MAPPING_NO_MAP                                (0x00000001)
#define LWOS02_FLAGS_MAPPING_NEVER_MAP                             (0x00000002)

// -------------------------------------------------------------------------------------

/* parameters */
typedef struct
{
    LwHandle    hRoot;
    LwHandle    hObjectParent;
    LwHandle    hObjectNew;
    LwV32       hClass;
    LwV32       flags;
    LwP64       pMemory LW_ALIGN_BYTES(8);
    LwU64       limit LW_ALIGN_BYTES(8);
    LwV32       status;
} LWOS02_PARAMETERS;

/* parameter values */
#define LWOS03_FLAGS_ACCESS                                        1:0
#define LWOS03_FLAGS_ACCESS_READ_WRITE                             (0x00000000)
#define LWOS03_FLAGS_ACCESS_READ_ONLY                              (0x00000001)
#define LWOS03_FLAGS_ACCESS_WRITE_ONLY                             (0x00000002)

#define LWOS03_FLAGS_PREALLOCATE                                   2:2
#define LWOS03_FLAGS_PREALLOCATE_DISABLE                           (0x00000000)
#define LWOS03_FLAGS_PREALLOCATE_ENABLE                            (0x00000001)

#define LWOS03_FLAGS_GPU_MAPPABLE                                  15:15
#define LWOS03_FLAGS_GPU_MAPPABLE_DISABLE                          (0x00000000)
#define LWOS03_FLAGS_GPU_MAPPABLE_ENABLE                           (0x00000001)

// ------------------------------------------------------------------------------------
// This flag is required for a hack to be placed inside DD that allows it to
// access a dummy ctxdma as a block linear surface. Refer bug 1562766 for details.
//
// This flag is deprecated, use LWOS03_FLAGS_PTE_KIND.
//
#define LWOS03_FLAGS_PTE_KIND_BL_OVERRIDE                          16:16
#define LWOS03_FLAGS_PTE_KIND_BL_OVERRIDE_FALSE                    (0x00000000)
#define LWOS03_FLAGS_PTE_KIND_BL_OVERRIDE_TRUE                     (0x00000001)

/*
 * This field allows to specify the page kind. If the page kind
 * is not specified then the page kind associated with the memory will be used.
 *
 * In cheetah display driver stack, the page kind remains unknown at the time
 * of memory allocation/import, the page kind can only be known when display
 * driver client creates a framebuffer from allocated/imported memory.
 *
 * This field compatible with LWOS03_FLAGS_PTE_KIND_BL_OVERRIDE flag.
 */
#define LWOS03_FLAGS_PTE_KIND                                      17:16
#define LWOS03_FLAGS_PTE_KIND_NONE                                 (0x00000000)
#define LWOS03_FLAGS_PTE_KIND_BL                                   (0x00000001)
#define LWOS03_FLAGS_PTE_KIND_PITCH                                (0x00000002)

#define LWOS03_FLAGS_TYPE                                          23:20
#define LWOS03_FLAGS_TYPE_NOTIFIER                                 (0x00000001)

/*
 * This is an alias into the LSB of the TYPE field which
 * actually indicates if a Kernel Mapping should be created.
 * If the RM should have access to the memory then Enable this
 * flag.
 *
 * Note that the LW_OS03_FLAGS_MAPPING is an alias to
 * the LSB of the LW_OS03_FLAGS_TYPE. And in fact if
 * type is LW_OS03_FLAGS_TYPE_NOTIFIER (bit 20 set)
 * then it implicitly means that LW_OS03_FLAGS_MAPPING
 * is _MAPPING_KERNEL. If the client wants to have a
 * Kernel Mapping, it should use the _MAPPING_KERNEL
 * flag set and the _TYPE_NOTIFIER should be used only
 * with NOTIFIERS.
 */

#define LWOS03_FLAGS_MAPPING                                       20:20
#define LWOS03_FLAGS_MAPPING_NONE                                  (0x00000000)
#define LWOS03_FLAGS_MAPPING_KERNEL                                (0x00000001)

#define LWOS03_FLAGS_CACHE_SNOOP                                   28:28
#define LWOS03_FLAGS_CACHE_SNOOP_ENABLE                            (0x00000000)
#define LWOS03_FLAGS_CACHE_SNOOP_DISABLE                           (0x00000001)

// HASH_TABLE:ENABLE means that the context DMA is automatically bound into all
// channels in the client.  This can lead to excessive hash table usage.
// HASH_TABLE:DISABLE means that the context DMA must be explicitly bound into
// any channel that needs to use it via LwRmBindContextDma.
// HASH_TABLE:ENABLE is not supported on LW50 and up, and HASH_TABLE:DISABLE should
// be preferred for all new code.
#define LWOS03_FLAGS_HASH_TABLE                                    29:29
#define LWOS03_FLAGS_HASH_TABLE_ENABLE                             (0x00000000)
#define LWOS03_FLAGS_HASH_TABLE_DISABLE                            (0x00000001)

/* macro LW01_ALLOC_OBJECT */
#define  LW01_ALLOC_OBJECT                                         (0x00000005)

/* parameters */
typedef struct
{
    LwHandle hRoot;
    LwHandle hObjectParent;
    LwHandle hObjectNew;
    LwV32    hClass;
    LwV32    status;
} LWOS05_PARAMETERS;

/* Valid values for hClass in Lw01AllocEvent */
/* Note that LW01_EVENT_OS_EVENT is same as LW01_EVENT_WIN32_EVENT */
/* TODO: delete the WIN32 name */
#define  LW01_EVENT_KERNEL_CALLBACK                                (0x00000078)
#define  LW01_EVENT_OS_EVENT                                       (0x00000079)
#define  LW01_EVENT_WIN32_EVENT                             LW01_EVENT_OS_EVENT
#define  LW01_EVENT_KERNEL_CALLBACK_EX                             (0x0000007E)

/* NOTE: LW01_EVENT_KERNEL_CALLBACK is deprecated. Please use LW01_EVENT_KERNEL_CALLBACK_EX. */
/* For use with LW01_EVENT_KERNEL_CALLBACK. */
/* LWOS10_EVENT_KERNEL_CALLBACK data structure storage needs to be retained by the caller. */
typedef void (*Callback1ArgVoidReturn)(void *arg);
typedef void (*Callback5ArgVoidReturn)(void *arg1, void *arg2, LwHandle hEvent, LwU32 data, LwU32 status);

/* NOTE: the 'void* arg' below is ok (but unfortunate) since this interface
   can only be used by other kernel drivers which must share the same ptr-size */
typedef struct
{
    Callback1ArgVoidReturn  func;
    void                   *arg;
} LWOS10_EVENT_KERNEL_CALLBACK;

/* For use with LW01_EVENT_KERNEL_CALLBACK_EX. */
/* LWOS10_EVENT_KERNEL_CALLBACK_EX data structure storage needs to be retained by the caller. */
/* NOTE: the 'void* arg' below is ok (but unfortunate) since this interface
   can only be used by other kernel drivers which must share the same ptr-size */
typedef struct
{
    Callback5ArgVoidReturn  func;
    void                   *arg;
} LWOS10_EVENT_KERNEL_CALLBACK_EX;

/* Setting this bit in index will set the Event to a Broadcast type */
/* i.e. each subdevice under a device needs to see the Event before it's signaled */
#define LW01_EVENT_BROADCAST                                       (0x80000000)

/* allow non-root resman client to create LW01_EVENT_KERNEL_CALLBACK events */
/* -- this works in debug/develop drivers only (for security reasons)*/
#define LW01_EVENT_PERMIT_NON_ROOT_EVENT_KERNEL_CALLBACK_CREATION  (0x40000000)

/* RM event should be triggered only by the specified subdevice; see cl0005.h
 * for details re: how to specify subdevice. */
#define LW01_EVENT_SUBDEVICE_SPECIFIC                              (0x20000000)

/* RM should trigger the event but shouldn't do the book-keeping of data
 * associated with that event */
#define LW01_EVENT_WITHOUT_EVENT_DATA                              (0x10000000)

/* RM event should be triggered only by the non-stall interrupt */
#define LW01_EVENT_NONSTALL_INTR                                   (0x08000000)

/* RM event was allocated from client RM, post events back to client RM */
#define LW01_EVENT_CLIENT_RM                                       (0x04000000)

/* function OS0D */
/* DEPRECATED INTERFACE - try to avoid, do not add or enhance CFG interfaces */
#define  LW01_CONFIG_GET                                           (0x0000000D)

/* parameters */
typedef struct
{
  LwHandle  hClient;
  LwHandle  hDevice;
  LwV32     index;
  LwV32     value;
  LwV32     status;
} LWOS13_PARAMETERS;

/* function OS0E */
/* DEPRECATED INTERFACE - try to avoid, do not add or enhance CFG interfaces */
#define  LW01_CONFIG_SET                                           (0x0000000E)

/* parameters */
typedef struct
{
  LwHandle  hClient;
  LwHandle  hDevice;
  LwV32     index;
  LwV32     oldValue;
  LwV32     newValue;
  LwV32     status;
} LWOS14_PARAMETERS;

/* function OS17 */
/* DEPRECATED INTERFACE - try to avoid, do not add or enhance CFGEX interfaces */
#define  LW04_CONFIG_GET_EX                                        (0x00000011)

/* parameters */
typedef struct
{
    LwHandle hClient;
    LwHandle hDevice;
    LwV32    index;
    LwU32    paramSize;
    LwP64    paramStructPtr LW_ALIGN_BYTES(8);
    LwV32    status;
} LWOS_CONFIG_GET_EX_PARAMS;

/* function OS18 */
/* DEPRECATED INTERFACE - try to avoid, do not add or enhance CFGEX interfaces */
#define  LW04_CONFIG_SET_EX                                        (0x00000012)

/* parameters */
typedef struct
{
    LwHandle hClient;
    LwHandle hDevice;
    LwV32    index;
    LwU32    paramSize;
    LwP64    paramStructPtr LW_ALIGN_BYTES(8);
    LwV32    status;
} LWOS_CONFIG_SET_EX_PARAMS;

/* function OS19 */
#define  LW04_I2C_ACCESS                                           (0x00000013)

#define LWOS_I2C_ACCESS_MAX_BUFFER_SIZE  2048

/* parameters */
typedef struct
{
    LwHandle hClient;
    LwHandle hDevice;
    LwU32    paramSize;
    LwP64    paramStructPtr LW_ALIGN_BYTES(8);
    LwV32    status;
} LWOS_I2C_ACCESS_PARAMS;

/* current values for command */
#define LWOS20_COMMAND_unused0001                  0x0001
#define LWOS20_COMMAND_unused0002                  0x0002
#define LWOS20_COMMAND_STRING_PRINT                0x0003

/* function OS21 */
#define  LW04_ALLOC                                                (0x00000015)

/* parameters */
typedef struct
{
    LwHandle hRoot;
    LwHandle hObjectParent;
    LwHandle hObjectNew;
    LwV32    hClass;
    LwP64    pAllocParms LW_ALIGN_BYTES(8);
    LwV32    status;
} LWOS21_PARAMETERS;

/* New struct with rights requested */
typedef struct
{
    LwHandle hRoot;                               // [IN] client handle
    LwHandle hObjectParent;                       // [IN] parent handle of new object
    LwHandle hObjectNew;                          // [INOUT] new object handle, 0 to generate
    LwV32    hClass;                              // [in] class num of new object
    LwP64    pAllocParms LW_ALIGN_BYTES(8);       // [IN] class-specific alloc parameters
    LwP64    pRightsRequested LW_ALIGN_BYTES(8);  // [IN] RS_ACCESS_MASK to request rights, or NULL
    LwV32    status;                              // [OUT] status
} LWOS64_PARAMETERS;

/* RM Alloc header
 *
 * Replacement for LWOS21/64_PARAMETERS where embedded pointers are not allowed.
 * Input layout for RM Alloc user space calls should be
 *
 * +--- LWOS62_PARAMETERS ---+--- RM Alloc parameters ---+
 * +--- LWOS65_PARAMETERS ---+--- Rights Requested ---+--- RM Alloc parameters ---+
 *
 * LWOS62_PARAMETERS::paramsSize is the size of RM Alloc parameters
 * If LWOS65_PARAMETERS::maskSize is 0, Rights Requested will not be present in memory.
 *
 */
typedef struct
{
    LwHandle hRoot;             // [IN]  client handle
    LwHandle hObjectParent;     // [IN]  parent handle of the new object
    LwHandle hObjectNew;        // [IN]  new object handle
    LwV32    hClass;            // [IN]  class num of the new object
    LwU32    paramSize;         // [IN]  size in bytes of the RM alloc parameters
    LwV32    status;            // [OUT] status
} LWOS62_PARAMETERS;

#define LWOS65_PARAMETERS_VERSION_MAGIC 0x77FEF81E

typedef struct
{
    LwHandle hRoot;             // [IN]  client handle
    LwHandle hObjectParent;     // [IN]  parent handle of the new object
    LwHandle hObjectNew;        // [INOUT]  new object handle, 0 to generate
    LwV32    hClass;            // [IN]  class num of the new object
    LwU32    paramSize;         // [IN]  size in bytes of the RM alloc parameters
    LwU32    versionMagic;      // [IN]  LWOS65_PARAMETERS_VERISON_MAGIC
    LwU32    maskSize;          // [IN]  size in bytes of access mask, or 0 if NULL
    LwV32    status;            // [OUT] status
} LWOS65_PARAMETERS;

/* function OS30 */
#define LW04_IDLE_CHANNELS                                         (0x0000001E)

/* parameter values */
#define LWOS30_FLAGS_BEHAVIOR                                      3:0
#define LWOS30_FLAGS_BEHAVIOR_SPIN                                 (0x00000000)
#define LWOS30_FLAGS_BEHAVIOR_SLEEP                                (0x00000001)
#define LWOS30_FLAGS_BEHAVIOR_QUERY                                (0x00000002)
#define LWOS30_FLAGS_BEHAVIOR_FORCE_BUSY_CHECK                     (0x00000003)
#define LWOS30_FLAGS_CHANNEL                                       7:4
#define LWOS30_FLAGS_CHANNEL_LIST                                  (0x00000000)
#define LWOS30_FLAGS_CHANNEL_SINGLE                                (0x00000001)
#define LWOS30_FLAGS_IDLE                                          30:8
#define LWOS30_FLAGS_IDLE_PUSH_BUFFER                              (0x00000001)
#define LWOS30_FLAGS_IDLE_CACHE1                                   (0x00000002)
#define LWOS30_FLAGS_IDLE_GRAPHICS                                 (0x00000004)
#define LWOS30_FLAGS_IDLE_MPEG                                     (0x00000008)
#define LWOS30_FLAGS_IDLE_MOTION_ESTIMATION                        (0x00000010)
#define LWOS30_FLAGS_IDLE_VIDEO_PROCESSOR                          (0x00000020)
#define LWOS30_FLAGS_IDLE_MSPDEC                                   (0x00000020)
#define LWOS30_FLAGS_IDLE_BITSTREAM_PROCESSOR                      (0x00000040)
#define LWOS30_FLAGS_IDLE_MSVLD                                    (0x00000040)
#define LWOS30_FLAGS_IDLE_LWDEC0                                   LWOS30_FLAGS_IDLE_MSVLD
#define LWOS30_FLAGS_IDLE_CIPHER_DMA                               (0x00000080)
#define LWOS30_FLAGS_IDLE_SEC                                      (0x00000080)
#define LWOS30_FLAGS_IDLE_CALLBACKS                                (0x00000100)
#define LWOS30_FLAGS_IDLE_MSPPP                                    (0x00000200)
#define LWOS30_FLAGS_IDLE_CE0                                      (0x00000400)
#define LWOS30_FLAGS_IDLE_CE1                                      (0x00000800)
#define LWOS30_FLAGS_IDLE_CE2                                      (0x00001000)
#define LWOS30_FLAGS_IDLE_CE3                                      (0x00002000)
#define LWOS30_FLAGS_IDLE_CE4                                      (0x00004000)
#define LWOS30_FLAGS_IDLE_CE5                                      (0x00008000)
#define LWOS30_FLAGS_IDLE_VIC                                      (0x00010000)
#define LWOS30_FLAGS_IDLE_MSENC                                    (0x00020000)
#define LWOS30_FLAGS_IDLE_LWENC0                                    LWOS30_FLAGS_IDLE_MSENC
#define LWOS30_FLAGS_IDLE_LWENC1                                   (0x00040000)
#define LWOS30_FLAGS_IDLE_LWENC2                                   (0x00080000)
#define LWOS30_FLAGS_IDLE_LWJPG                                    (0x00100000)
#define LWOS30_FLAGS_IDLE_LWDEC1                                   (0x00200000)
#define LWOS30_FLAGS_IDLE_LWDEC2                                   (0x00400000)
#define LWOS30_FLAGS_IDLE_ACTIVECHANNELS                           (0x00800000)
#define LWOS30_FLAGS_IDLE_ALL_ENGINES (LWOS30_FLAGS_IDLE_GRAPHICS | \
                                       LWOS30_FLAGS_IDLE_MPEG | \
                                       LWOS30_FLAGS_IDLE_MOTION_ESTIMATION | \
                                       LWOS30_FLAGS_IDLE_VIDEO_PROCESSOR | \
                                       LWOS30_FLAGS_IDLE_BITSTREAM_PROCESSOR |  \
                                       LWOS30_FLAGS_IDLE_CIPHER_DMA  | \
                                       LWOS30_FLAGS_IDLE_MSPDEC      | \
                                       LWOS30_FLAGS_IDLE_LWDEC0      | \
                                       LWOS30_FLAGS_IDLE_SEC         | \
                                       LWOS30_FLAGS_IDLE_MSPPP       | \
                                       LWOS30_FLAGS_IDLE_CE0         | \
                                       LWOS30_FLAGS_IDLE_CE1         | \
                                       LWOS30_FLAGS_IDLE_CE2         | \
                                       LWOS30_FLAGS_IDLE_CE3         | \
                                       LWOS30_FLAGS_IDLE_CE4         | \
                                       LWOS30_FLAGS_IDLE_CE5         | \
                                       LWOS30_FLAGS_IDLE_LWENC0      | \
                                       LWOS30_FLAGS_IDLE_LWENC1      | \
                                       LWOS30_FLAGS_IDLE_LWENC2      | \
                                       LWOS30_FLAGS_IDLE_VIC         | \
                                       LWOS30_FLAGS_IDLE_LWJPG       | \
                                       LWOS30_FLAGS_IDLE_LWDEC1      | \
                                       LWOS30_FLAGS_IDLE_LWDEC2)
#define LWOS30_FLAGS_WAIT_FOR_ELPG_ON                              31:31
#define LWOS30_FLAGS_WAIT_FOR_ELPG_ON_NO                           (0x00000000)
#define LWOS30_FLAGS_WAIT_FOR_ELPG_ON_YES                          (0x00000001)

/* parameters */
typedef struct
{
    LwHandle hClient;
    LwHandle hDevice;
    LwHandle hChannel;
    LwV32    numChannels;

    LwP64    phClients LW_ALIGN_BYTES(8);
    LwP64    phDevices LW_ALIGN_BYTES(8);
    LwP64    phChannels LW_ALIGN_BYTES(8);

    LwV32    flags;
    LwV32    timeout;
    LwV32    status;
} LWOS30_PARAMETERS;

/* function OS32 */
typedef void (*BindResultFunc)(void * pVoid, LwU32 gpuMask, LwU32 bState, LwU32 bResult);

#define LW04_VID_HEAP_CONTROL                                      (0x00000020)
/*************************************************************************
************************ New Heap Interface ******************************
*************************************************************************/
// LWOS32 Descriptor types
//
// LWOS32_DESCRIPTOR_TYPE_OS_DMA_BUF_PTR: The dma-buf object
// pointer, provided by the linux kernel buffer sharing sub-system.
// This descriptor can only be used by kernel space rm-clients.
//
#define LWOS32_DESCRIPTOR_TYPE_VIRTUAL_ADDRESS          0
#define LWOS32_DESCRIPTOR_TYPE_OS_PAGE_ARRAY            1
#define LWOS32_DESCRIPTOR_TYPE_OS_IO_MEMORY             2
#define LWOS32_DESCRIPTOR_TYPE_OS_PHYS_ADDR             3
#define LWOS32_DESCRIPTOR_TYPE_OS_FILE_HANDLE           4
#define LWOS32_DESCRIPTOR_TYPE_OS_DMA_BUF_PTR           5
#define LWOS32_DESCRIPTOR_TYPE_OS_SGT_PTR               6
// LWOS32 function
#define LWOS32_FUNCTION_ALLOC_DEPTH_WIDTH_HEIGHT        1
#define LWOS32_FUNCTION_ALLOC_SIZE                      2
#define LWOS32_FUNCTION_FREE                            3
// #define LWOS32_FUNCTION_HEAP_PURGE                   4
#define LWOS32_FUNCTION_INFO                            5
#define LWOS32_FUNCTION_ALLOC_TILED_PITCH_HEIGHT        6
// #define LWOS32_FUNCTION_DESTROY                      7
// #define LWOS32_FUNCTION_RETAIN                       9
// #define LWOS32_FUNCTION_REALLOC                      10
#define LWOS32_FUNCTION_DUMP                            11
// #define LWOS32_FUNCTION_INFO_TYPE_ALLOC_BLOCKS       12
#define LWOS32_FUNCTION_ALLOC_SIZE_RANGE                14
#define LWOS32_FUNCTION_REACQUIRE_COMPR                 15
#define LWOS32_FUNCTION_RELEASE_COMPR                   16
// #define LWOS32_FUNCTION_MODIFY_DEFERRED_TILES        17
#define LWOS32_FUNCTION_GET_MEM_ALIGNMENT               18
#define LWOS32_FUNCTION_HW_ALLOC                        19
#define LWOS32_FUNCTION_HW_FREE                         20
// #define LWOS32_FUNCTION_SET_OFFSET                   21
// #define LWOS32_FUNCTION_IS_TILED                     22
// #define LWOS32_FUNCTION_ENABLE_RESOURCE              23
// #define LWOS32_FUNCTION_BIND_COMPR                   24
#define LWOS32_FUNCTION_ALLOC_OS_DESCRIPTOR             27

typedef struct
{
    LwP64 sgt LW_ALIGN_BYTES(8);
    LwP64 gem LW_ALIGN_BYTES(8);
} LWOS32_DESCRIPTOR_TYPE_OS_SGT_PTR_PARAMETERS;

#define LWOS32_FLAGS_BLOCKINFO_VISIBILITY_CPU (0x00000001)
typedef struct
{
    LwU64 startOffset LW_ALIGN_BYTES(8);
    LwU64 size LW_ALIGN_BYTES(8);
    LwU32 flags;
} LWOS32_BLOCKINFO;

// LWOS32 IVC-heap number delimiting value
#define LWOS32_IVC_HEAP_NUMBER_DONT_ALLOCATE_ON_IVC_HEAP 0 // When IVC heaps are present,
                                                           // IVC-heap number specified
                                                           // as part of 'LWOS32_PARAMETERS'
                                                           // which is less or equal to this
                                                           // constant indicates that allocation
                                                           // should not be done on IVC heap.
                                                           // Explanation of IVC-heap number is
                                                           // under 'AllocSize' structure below.

typedef struct
{
  LwHandle  hRoot;                      // [IN]  - root object handle
  LwHandle  hObjectParent;              // [IN]  - device handle
  LwU32     function;                   // [IN]  - heap function, see below FUNCTION* defines
  LwHandle  hVASpace;                   // [IN]  - VASpace handle
  LwS16     ivcHeapNumber;              // [IN] - When IVC heaps are present: either 1) number of the IVC heap
                                        //        shared between two VMs or 2) number indicating that allocation
                                        //        should not be done on an IVC heap. Values greater than constant
                                        //        'LWOS32_IVC_HEAP_NUMBER_DONT_ALLOCATE_ON_IVC_HEAP' define set 1)
                                        //        and values less or equal to that constant define set 2).
                                        //        When IVC heaps are present, correct IVC-heap number must be specified.
                                        //        When IVC heaps are absent, IVC-heap number is diregarded.
                                        //        RM provides for each VM a bitmask of heaps with each bit
                                        //        specifying the other peer that can use the partition.
                                        //        Each bit set to one can be enumerated, such that the bit
                                        //        with lowest significance is enumerated with one.
                                        //        'ivcHeapNumber' parameter specifies this enumeration value.
                                        //        This value is used to uniquely identify a heap shared between
                                        //        two particular VMs.
                                        //        Illustration:
                                        //                                bitmask: 1  1  0  1  0 = 0x1A
                                        //        possible 'ivcHeapNumber' values: 3, 2,    1
  LwV32     status;                     // [OUT] - returned LWOS32* status code, see below STATUS* defines
  LwU64     total LW_ALIGN_BYTES(8);    // [OUT] - returned total size of heap
  LwU64     free  LW_ALIGN_BYTES(8);    // [OUT] - returned free space available in heap

  union
  {
      // LWOS32_FUNCTION_ALLOC_DEPTH_WIDTH_HEIGHT
      struct
      {
          LwU32     owner;              // [IN]  - memory owner ID
          LwHandle  hMemory;            // [IN/OUT] - unique memory handle - IN only if MEMORY_HANDLE_PROVIDED is set (otherwise generated)
          LwU32     type;               // [IN]  - surface type, see below TYPE* defines
          LwU32     flags;              // [IN]  - allocation modifier flags, see below ALLOC_FLAGS* defines
          LwU32     depth;              // [IN]  - depth of surface in bits
          LwU32     width;              // [IN]  - width of surface in pixels
          LwU32     height;             // [IN]  - height of surface in pixels
          LwU32     attr;               // [IN/OUT] - surface attributes requested, and surface attributes allocated
          LwU32     format;             // [IN/OUT] - format requested, and format allocated
          LwU32     comprCovg;          // [IN/OUT] - compr covg requested, and allocated
          LwU32     zlwllCovg;          // [OUT] - zlwll covg allocated
          LwU32     partitionStride;    // [IN/OUT] - 0 means "RM" chooses
          LwU64     size      LW_ALIGN_BYTES(8); // [IN/OUT]  - size of allocation - also returns the actual size allocated
          LwU64     alignment LW_ALIGN_BYTES(8); // [IN]  - requested alignment - LWOS32_ALLOC_FLAGS_ALIGNMENT* must be on
          LwU64     offset    LW_ALIGN_BYTES(8); // [IN/OUT]  - desired offset if LWOS32_ALLOC_FLAGS_FIXED_ADDRESS_ALLOCATE is on AND returned offset
          LwU64     limit     LW_ALIGN_BYTES(8); // [OUT] - returned surface limit
          LwP64     address LW_ALIGN_BYTES(8);// [OUT] - returned address
          LwU64     rangeBegin LW_ALIGN_BYTES(8); // [IN]  - allocated memory will be limited to the range
          LwU64     rangeEnd   LW_ALIGN_BYTES(8); // [IN]  - from rangeBegin to rangeEnd, inclusive.
          LwU32     attr2;              // [IN/OUT] - surface attributes requested, and surface attributes allocated
          LwU32     ctagOffset;         // [IN] - comptag offset for this surface (see LWOS32_ALLOC_COMPTAG_OFFSET)
      } AllocDepthWidthHeight;

      // LWOS32_FUNCTION_ALLOC_SIZE
      struct
      {
          LwU32     owner;              // [IN]  - memory owner ID
          LwHandle  hMemory;            // [IN/OUT]  - unique memory handle - IN only if MEMORY_HANDLE_PROVIDED is set (otherwise generated)
          LwU32     type;               // [IN]  - surface type, see below TYPE* defines
          LwU32     flags;              // [IN]  - allocation modifier flags, see below ALLOC_FLAGS* defines
          LwU32     attr;               // [IN/OUT] - surface attributes requested, and surface attributes allocated
          LwU32     format;             // [IN/OUT] - format requested, and format allocated
          LwU32     comprCovg;          // [IN/OUT] - compr covg requested, and allocated
          LwU32     zlwllCovg;          // [OUT] - zlwll covg allocated
          LwU32     partitionStride;    // [IN/OUT] - 0 means "RM" chooses
          LwU32     width;              // [IN] - width "hint" used for zlwll region allocations
          LwU32     height;             // [IN] - height "hint" used for zlwll region allocations
          LwU64     size      LW_ALIGN_BYTES(8); // [IN/OUT]  - size of allocation - also returns the actual size allocated
          LwU64     alignment LW_ALIGN_BYTES(8); // [IN]  - requested alignment - LWOS32_ALLOC_FLAGS_ALIGNMENT* must be on
          LwU64     offset    LW_ALIGN_BYTES(8); // [IN/OUT]  - desired offset if LWOS32_ALLOC_FLAGS_FIXED_ADDRESS_ALLOCATE is on AND returned offset
          LwU64     limit     LW_ALIGN_BYTES(8); // [OUT] - returned surface limit
          LwP64     address LW_ALIGN_BYTES(8);// [OUT] - returned address
          LwU64     rangeBegin LW_ALIGN_BYTES(8); // [IN]  - allocated memory will be limited to the range
          LwU64     rangeEnd   LW_ALIGN_BYTES(8); // [IN]  - from rangeBegin to rangeEnd, inclusive.
          LwU32     attr2;              // [IN/OUT] - surface attributes requested, and surface attributes allocated
          LwU32     ctagOffset;         // [IN] - comptag offset for this surface (see LWOS32_ALLOC_COMPTAG_OFFSET)
      } AllocSize;

      // LWOS32_FUNCTION_ALLOC_TILED_PITCH_HEIGHT
      struct
      {
          LwU32     owner;              // [IN]  - memory owner ID
          LwHandle  hMemory;            // [IN/OUT]  - unique memory handle - IN only if MEMORY_HANDLE_PROVIDED is set (otherwise generated)
          LwU32     type;               // [IN]  - surface type, see below TYPE* defines
          LwU32     flags;              // [IN]  - allocation modifier flags, see below ALLOC_FLAGS* defines
          LwU32     height;             // [IN]  - height of surface in pixels
          LwS32     pitch;              // [IN/OUT] - desired pitch AND returned actual pitch allocated
          LwU32     attr;               // [IN/OUT] - surface attributes requested, and surface attributes allocated
          LwU32     width;               //[IN] - width of surface in pixels
          LwU32     format;             // [IN/OUT] - format requested, and format allocated
          LwU32     comprCovg;          // [IN/OUT] - compr covg requested, and allocated
          LwU32     zlwllCovg;          // [OUT] - zlwll covg allocated
          LwU32     partitionStride;    // [IN/OUT] - 0 means "RM" chooses
          LwU64     size      LW_ALIGN_BYTES(8); // [IN/OUT]  - size of allocation - also returns the actual size allocated
          LwU64     alignment LW_ALIGN_BYTES(8); // [IN]  - requested alignment - LWOS32_ALLOC_FLAGS_ALIGNMENT* must be on
          LwU64     offset    LW_ALIGN_BYTES(8); // [IN/OUT]  - desired offset if LWOS32_ALLOC_FLAGS_FIXED_ADDRESS_ALLOCATE is on AND returned offset
          LwU64     limit     LW_ALIGN_BYTES(8); // [OUT] - returned surface limit
          LwP64     address LW_ALIGN_BYTES(8);// [OUT] - returned address
          LwU64     rangeBegin LW_ALIGN_BYTES(8); // [IN]  - allocated memory will be limited to the range
          LwU64     rangeEnd   LW_ALIGN_BYTES(8); // [IN]  - from rangeBegin to rangeEnd, inclusive.
          LwU32     attr2;              // [IN/OUT] - surface attributes requested, and surface attributes allocated
          LwU32     ctagOffset;         // [IN] - comptag offset for this surface (see LWOS32_ALLOC_COMPTAG_OFFSET)
      } AllocTiledPitchHeight;

      // LWOS32_FUNCTION_FREE
      struct
      {
          LwU32     owner;              // [IN]  - memory owner ID
          LwHandle  hMemory;            // [IN]  - unique memory handle
          LwU32     flags;              // [IN]  - heap free flags (must be LWOS32_FREE_FLAGS_MEMORY_HANDLE_PROVIDED)
      } Free;

      // LWOS32_FUNCTION_RELEASE_COMPR
      struct
      {
          LwU32     owner;           // [IN]  - memory owner ID
          LwU32     flags;           // [IN]  - must be LWOS32_RELEASE_COMPR_FLAGS_MEMORY_HANDLE_PROVIDED
          LwHandle  hMemory;         // [IN]  - unique memory handle (valid if _RELEASE_COMPR_FLAGS_MEMORY_HANDLE_PROVIDED
      } ReleaseCompr;

      // LWOS32_FUNCTION_REACQUIRE_COMPR
      struct
      {
          LwU32     owner;           // [IN]  - memory owner ID
          LwU32     flags;           // [IN]  - must be LWOS32_REACQUIRE_COMPR_FLAGS_MEMORY_HANDLE_PROVIDED
          LwHandle  hMemory;         // [IN]  - unique memory handle (valid if _REACQUIRE_COMPR_FLAGS_MEMORY_HANDLE_PROVIDED
      } ReacquireCompr;

      // LWOS32_FUNCTION_INFO
      struct
      {
          LwU32 attr;               // [IN] - memory heap attributes requested
          LwU64 offset LW_ALIGN_BYTES(8); // [OUT] - base of largest free block
          LwU64 size   LW_ALIGN_BYTES(8); // [OUT] - size of largest free block
          LwU64 base   LW_ALIGN_BYTES(8);   // [OUT] - returned heap phys base
      } Info;

      // LWOS32_FUNCTION_DUMP
      struct
      {
          LwU32 flags;              // [IN] - see _DUMP_FLAGS
          // [IN]  - if NULL, numBlocks is the returned number of blocks in
          //         heap, else returns all blocks in eHeap
          //         if non-NULL points to a buffer that is at least numBlocks
          //         * sizeof(LWOS32_HEAP_DUMP_BLOCK) bytes.
          LwP64 pBuffer LW_ALIGN_BYTES(8);
          // [IN/OUT] - if pBuffer is NULL, will number of blocks in heap
          //            if pBuffer is non-NULL, is input containing the size of
          //            pBuffer in units of LWOS32_HEAP_DUMP_BLOCK.  This must
          //            be greater than or equal to the number of blocks in the
          //            heap.
          LwU32 numBlocks;
      } Dump;

      // LWOS32_FUNCTION_DESTROY - no extra parameters needed

      // LWOS32_FUNCTION_ALLOC_SIZE_RANGE
      struct
      {
          LwU32     owner;              // [IN]  - memory owner ID
          LwHandle  hMemory;            // [IN]  - unique memory handle
          LwU32     type;               // [IN]  - surface type, see below TYPE* defines
          LwU32     flags;              // [IN]  - allocation modifier flags, see below ALLOC_FLAGS* defines
          LwU32     attr;               // [IN/OUT] - surface attributes requested, and surface attributes allocated
          LwU32     format;             // [IN/OUT] - format requested, and format allocated
          LwU32     comprCovg;          // [IN/OUT] - compr covg requested, and allocated
          LwU32     zlwllCovg;          // [OUT] - zlwll covg allocated
          LwU32     partitionStride;    // [IN/OUT] - 0 means "RM" chooses
          LwU64     size      LW_ALIGN_BYTES(8);  // [IN/OUT]  - size of allocation - also returns the actual size allocated
          LwU64     alignment LW_ALIGN_BYTES(8);  // [IN]  - requested alignment - LWOS32_ALLOC_FLAGS_ALIGNMENT* must be on
          LwU64     offset    LW_ALIGN_BYTES(8);  // [IN/OUT]  - desired offset if LWOS32_ALLOC_FLAGS_FIXED_ADDRESS_ALLOCATE is on AND returned offset
          LwU64     limit     LW_ALIGN_BYTES(8);  // [OUT] - returned surface limit
          LwU64     rangeBegin LW_ALIGN_BYTES(8); // [IN]  - allocated memory will be limited to the range
          LwU64     rangeEnd   LW_ALIGN_BYTES(8); // [IN]  - from rangeBegin to rangeEnd, inclusive.
          LwP64     address LW_ALIGN_BYTES(8);// [OUT] - returned address
          LwU32     attr2;              // [IN/OUT] - surface attributes requested, and surface attributes allocated
          LwU32     ctagOffset;         // [IN] - comptag offset for this surface (see LWOS32_ALLOC_COMPTAG_OFFSET)
      } AllocSizeRange;

      // additions for Longhorn
#define LWAL_MAX_BANKS (4)
#define LWAL_MAP_DIRECTION             0:0
#define LWAL_MAP_DIRECTION_DOWN 0x00000000
#define LWAL_MAP_DIRECTION_UP   0x00000001

      // LWOS32_FUNCTION_GET_MEM_ALIGNMENT
      struct
      {
          LwU32 alignType;                                 // Input
          LwU32 alignAttr;
          LwU32 alignInputFlags;
          LwU64 alignSize LW_ALIGN_BYTES(8);
          LwU32 alignHeight;
          LwU32 alignWidth;
          LwU32 alignPitch;
          LwU32 alignPad;
          LwU32 alignMask;
          LwU32 alignOutputFlags[LWAL_MAX_BANKS];           // We could compress this information but it is probably not that big of a deal
          LwU32 alignBank[LWAL_MAX_BANKS];
          LwU32 alignKind;
          LwU32 alignAdjust;                                // Output -- If non-zero the amount we need to adjust the offset
          LwU32 alignAttr2;
      } AllocHintAlignment;

      struct
      {
          LwU32     allocOwner;              // [IN]  - memory owner ID
          LwHandle  allochMemory;            // [IN/OUT] - unique memory handle - IN only if MEMORY_HANDLE_PROVIDED is set (otherwise generated)
          LwU32     flags;
          LwU32     allocType;               // Input
          LwU32     allocAttr;
          LwU32     allocInputFlags;
          LwU64     allocSize       LW_ALIGN_BYTES(8);
          LwU32     allocHeight;
          LwU32     allocWidth;
          LwU32     allocPitch;
          LwU32     allocMask;
          LwU32     allocComprCovg;
          LwU32     allocZlwllCovg;
          LwP64     bindResultFunc  LW_ALIGN_BYTES(8);         // BindResultFunc
          LwP64     pHandle         LW_ALIGN_BYTES(8);
          LwHandle  hResourceHandle;                          // Handle to RM container
          LwU32     retAttr;                                  // Output Indicates the resources that we allocated
          LwU32     kind;
          LwU64     osDeviceHandle  LW_ALIGN_BYTES(8);
          LwU32     allocAttr2;
          LwU32     retAttr2;                                 // Output Indicates the resources that we allocated
          LwU64     allocAddr       LW_ALIGN_BYTES(8);
          // [out] from GMMU_COMPR_INFO in drivers/common/shared/inc/mmu/gmmu_fmt.h
          struct
          {
              LwU32 compPageShift;
              LwU32 compressedKind;
              LwU32 compTagLineMin;
              LwU32 compPageIndexLo;
              LwU32 compPageIndexHi;
              LwU32 compTagLineMultiplier;
          } comprInfo;
          // [out] fallback uncompressed kind.
          LwU32  uncompressedKind;
      } HwAlloc;

      // LWOS32_FUNCTION_HW_FREE
      struct
      {
          LwHandle  hResourceHandle;                          // Handle to RM Resource Info
          LwU32     flags;                                    // Indicate if HW Resources and/or Memory
      } HwFree;
// Updated interface check.
#define LW_RM_OS32_ALLOC_OS_DESCRIPTOR_WITH_OS32_ATTR   1

      // LWOS32_FUNCTION_ALLOC_OS_DESCRIPTOR
      struct
      {
          LwHandle  hMemory;                      // [IN/OUT] - unique memory handle - IN only if MEMORY_HANDLE_PROVIDED is set (otherwise generated)
          LwU32     type;                         // [IN]  - surface type, see below TYPE* defines
          LwU32     flags;                        // [IN]  - allocation modifier flags, see below ALLOC_FLAGS* defines
          LwU32     attr;                         // [IN]  - attributes for memory placement/properties, see below
          LwU32     attr2;                        // [IN]  - attributes GPU_CACHEABLE
          LwP64     descriptor LW_ALIGN_BYTES(8); // [IN]  - descriptor address
          LwU64     limit      LW_ALIGN_BYTES(8); // [IN]  - allocated size -1
          LwU32     descriptorType;               // [IN]  - descriptor type(Virtual | lwmap Handle)
      } AllocOsDesc;

  } data;
} LWOS32_PARAMETERS;

typedef struct
{
    LwU32 owner;    // owner id - LWOS32_BLOCK_TYPE_FREE or defined by client during heap_alloc
    LwU32 format;   // arch specific format/kind
    LwU64 begin LW_ALIGN_BYTES(8); // start of allocated memory block
    LwU64 align LW_ALIGN_BYTES(8); // actual start of usable memory, aligned to chip specific boundary
    LwU64 end   LW_ALIGN_BYTES(8); // end of usable memory.  end - align + 1 = size of block
} LWOS32_HEAP_DUMP_BLOCK;


#define LWOS32_DELETE_RESOURCES_ALL                     0

// type field
#define LWOS32_TYPE_IMAGE                               0
#define LWOS32_TYPE_DEPTH                               1
#define LWOS32_TYPE_TEXTURE                             2
#define LWOS32_TYPE_VIDEO                               3
#define LWOS32_TYPE_FONT                                4
#define LWOS32_TYPE_LWRSOR                              5
#define LWOS32_TYPE_DMA                                 6
#define LWOS32_TYPE_INSTANCE                            7
#define LWOS32_TYPE_PRIMARY                             8
#define LWOS32_TYPE_ZLWLL                               9
#define LWOS32_TYPE_UNUSED                              10
#define LWOS32_TYPE_SHADER_PROGRAM                      11
#define LWOS32_TYPE_OWNER_RM                            12
#define LWOS32_TYPE_NOTIFIER                            13
#define LWOS32_TYPE_RESERVED                            14
#define LWOS32_TYPE_PMA                                 15
#define LWOS32_TYPE_STENCIL                             16
#define LWOS32_NUM_MEM_TYPES                            17

// Surface attribute field - bitmask of requested attributes the surface
// should have.
// This value is updated to reflect what was actually allocated, and so this
// field must be checked after every allocation to determine what was
// allocated. Pass in the ANY tags to indicate that RM should fall back but
// still succeed the alloc.
// for example, if tiled_any is passed in, but no tile ranges are available,
// RM will allocate normal memory and indicate that in the returned attr field.
// Each returned attribute will have the REQUIRED field set if that attribute
// applies to the allocated surface.

#define LWOS32_ATTR_NONE                                0x00000000

#define LWOS32_ATTR_DEPTH                                      2:0
#define LWOS32_ATTR_DEPTH_UNKNOWN                       0x00000000
#define LWOS32_ATTR_DEPTH_8                             0x00000001
#define LWOS32_ATTR_DEPTH_16                            0x00000002
#define LWOS32_ATTR_DEPTH_24                            0x00000003
#define LWOS32_ATTR_DEPTH_32                            0x00000004
#define LWOS32_ATTR_DEPTH_64                            0x00000005
#define LWOS32_ATTR_DEPTH_128                           0x00000006

#define LWOS32_ATTR_COMPR_COVG                                 3:3
#define LWOS32_ATTR_COMPR_COVG_DEFAULT                  0x00000000
#define LWOS32_ATTR_COMPR_COVG_PROVIDED                 0x00000001

// Surface description - number of AA samples
// This number should only reflect AA done in hardware, not in software. For
// example, OpenGL's 8x AA mode is a mix of 2x hardware multisample and 2x2
// software supersample.
// OpenGL should specify ATTR_AA_SAMPLES of 2 in this case, not 8, because
// the hardware will be programmed to run in 2x AA mode.
// Note that X_VIRTUAL_Y means X real samples with Y samples total (i.e. Y
// does not indicate the number of virtual samples).  For instance, what
// arch and HW describe as LW_PGRAPH_ANTI_ALIAS_SAMPLES_MODE_2X2_VC_12
// corresponds to LWOS32_ATTR_AA_SAMPLES_4_VIRTUAL_16 here.

#define LWOS32_ATTR_AA_SAMPLES                                 7:4
#define LWOS32_ATTR_AA_SAMPLES_1                        0x00000000
#define LWOS32_ATTR_AA_SAMPLES_2                        0x00000001
#define LWOS32_ATTR_AA_SAMPLES_4                        0x00000002
#define LWOS32_ATTR_AA_SAMPLES_4_ROTATED                0x00000003
#define LWOS32_ATTR_AA_SAMPLES_6                        0x00000004
#define LWOS32_ATTR_AA_SAMPLES_8                        0x00000005
#define LWOS32_ATTR_AA_SAMPLES_16                       0x00000006
#define LWOS32_ATTR_AA_SAMPLES_4_VIRTUAL_8              0x00000007
#define LWOS32_ATTR_AA_SAMPLES_4_VIRTUAL_16             0x00000008
#define LWOS32_ATTR_AA_SAMPLES_8_VIRTUAL_16             0x00000009
#define LWOS32_ATTR_AA_SAMPLES_8_VIRTUAL_32             0x0000000A

// Tiled region
#define LWOS32_ATTR_TILED                                      9:8
#define LWOS32_ATTR_TILED_NONE                          0x00000000
#define LWOS32_ATTR_TILED_REQUIRED                      0x00000001
#define LWOS32_ATTR_TILED_ANY                           0x00000002
#define LWOS32_ATTR_TILED_DEFERRED                      0x00000003

// Zlwll region (LW40 and up)
// If ATTR_ZLWLL is REQUIRED or ANY and ATTR_DEPTH is UNKNOWN, the
// allocation will fail.
// If ATTR_DEPTH or ATTR_AA_SAMPLES is not accurate, erroneous rendering
// may result.
#define LWOS32_ATTR_ZLWLL                                    11:10
#define LWOS32_ATTR_ZLWLL_NONE                          0x00000000
#define LWOS32_ATTR_ZLWLL_REQUIRED                      0x00000001
#define LWOS32_ATTR_ZLWLL_ANY                           0x00000002
#define LWOS32_ATTR_ZLWLL_SHARED                        0x00000003

// Compression (LW20 and up)
// If ATTR_COMPR is REQUIRED or ANY and ATTR_DEPTH is UNKNOWN, the
// allocation will fail.
// If ATTR_DEPTH or ATTR_AA_SAMPLES is not accurate, performance will
// suffer heavily
#define LWOS32_ATTR_COMPR                                    13:12
#define LWOS32_ATTR_COMPR_NONE                          0x00000000
#define LWOS32_ATTR_COMPR_REQUIRED                      0x00000001
#define LWOS32_ATTR_COMPR_ANY                           0x00000002
#define LWOS32_ATTR_COMPR_PLC_REQUIRED                  LWOS32_ATTR_COMPR_REQUIRED
#define LWOS32_ATTR_COMPR_PLC_ANY                       LWOS32_ATTR_COMPR_ANY
#define LWOS32_ATTR_COMPR_DISABLE_PLC_ANY               0x00000003

// Format
// _BLOCK_LINEAR is only available for lw50+.
#define LWOS32_ATTR_FORMAT                                   17:16
// Macros representing the low/high bits of LWOS32_ATTR_FORMAT
// bit range. These provide direct access to the range limits
// without needing to split the low:high representation via
// ternary operator, thereby avoiding MISRA 14.3 violation.
#define LWOS32_ATTR_FORMAT_LOW_FIELD                            16
#define LWOS32_ATTR_FORMAT_HIGH_FIELD                           17
#define LWOS32_ATTR_FORMAT_PITCH                        0x00000000
#define LWOS32_ATTR_FORMAT_SWIZZLED                     0x00000001
#define LWOS32_ATTR_FORMAT_BLOCK_LINEAR                 0x00000002

#define LWOS32_ATTR_Z_TYPE                                   18:18
#define LWOS32_ATTR_Z_TYPE_FIXED                        0x00000000
#define LWOS32_ATTR_Z_TYPE_FLOAT                        0x00000001

#define LWOS32_ATTR_ZS_PACKING                               21:19
#define LWOS32_ATTR_ZS_PACKING_S8                       0x00000000 // Z24S8 and S8 share definition
#define LWOS32_ATTR_ZS_PACKING_Z24S8                    0x00000000
#define LWOS32_ATTR_ZS_PACKING_S8Z24                    0x00000001
#define LWOS32_ATTR_ZS_PACKING_Z32                      0x00000002
#define LWOS32_ATTR_ZS_PACKING_Z24X8                    0x00000003
#define LWOS32_ATTR_ZS_PACKING_X8Z24                    0x00000004
#define LWOS32_ATTR_ZS_PACKING_Z32_X24S8                0x00000005
#define LWOS32_ATTR_ZS_PACKING_X8Z24_X24S8              0x00000006
#define LWOS32_ATTR_ZS_PACKING_Z16                      0x00000007
// NOTE: ZS packing and color packing fields are overlaid
#define LWOS32_ATTR_COLOR_PACKING                       LWOS32_ATTR_ZS_PACKING
#define LWOS32_ATTR_COLOR_PACKING_A8R8G8B8              0x00000000
#define LWOS32_ATTR_COLOR_PACKING_X8R8G8B8              0x00000001

#ifdef LW_VERIF_FEATURES
// If _ON the format will be taken directly from the 'format' field
// and used to program the HW.
#define LWOS32_ATTR_FORMAT_OVERRIDE                          22:22
#define LWOS32_ATTR_FORMAT_OVERRIDE_OFF                 0x00000000
#define LWOS32_ATTR_FORMAT_OVERRIDE_ON                  0x00000001
#endif



//
// For virtual allocs to choose page size for the region. Specifying
// _DEFAULT will select a virtual page size that allows for a surface
// to be mixed between video and system memory and allow the surface
// to be migrated between video and system memory. For tesla chips,
// 4KB will be used. For fermi chips with dual page tables, a virtual
// address with both page tables will be used.
//
// For physical allocation on chips with page swizzle this field is
// used to select the page swizzle.  This later also sets the virtual
// page size, but does not have influence over selecting a migratable
// virtual address. That must be selected when mapping the physical
// memory.
//
// BIG_PAGE  = 64 KB on PASCAL
//           = 64 KB or 128 KB on pre_PASCAL chips
//
// HUGE_PAGE = 2 MB on PASCAL+
//           = 2 MB or 512 MB on AMPERE+
//           = not supported on pre_PASCAL chips.
//
// To request for a HUGE page size,
// set LWOS32_ATTR_PAGE_SIZE to _HUGE and LWOS32_ATTR2_PAGE_SIZE_HUGE to
// the desired size.
//
#define LWOS32_ATTR_PAGE_SIZE                                24:23
#define LWOS32_ATTR_PAGE_SIZE_DEFAULT                   0x00000000
#define LWOS32_ATTR_PAGE_SIZE_4KB                       0x00000001
#define LWOS32_ATTR_PAGE_SIZE_BIG                       0x00000002
#define LWOS32_ATTR_PAGE_SIZE_HUGE                      0x00000003

#define LWOS32_ATTR_LOCATION                                 26:25
#define LWOS32_ATTR_LOCATION_VIDMEM                     0x00000000
#define LWOS32_ATTR_LOCATION_PCI                        0x00000001
#define LWOS32_ATTR_LOCATION_AGP                        0x00000002
#define LWOS32_ATTR_LOCATION_ANY                        0x00000003

//
// _DEFAULT implies _CONTIGUOUS for video memory lwrrently, but
// may be changed to imply _NONCONTIGUOUS in the future.
// _ALLOW_NONCONTIGUOUS enables falling back to the noncontiguous
// vidmem allocator if contig allocation fails.
//
#define LWOS32_ATTR_PHYSICALITY                              28:27
#define LWOS32_ATTR_PHYSICALITY_DEFAULT                 0x00000000
#define LWOS32_ATTR_PHYSICALITY_NONCONTIGUOUS           0x00000001
#define LWOS32_ATTR_PHYSICALITY_CONTIGUOUS              0x00000002
#define LWOS32_ATTR_PHYSICALITY_ALLOW_NONCONTIGUOUS     0x00000003

#define LWOS32_ATTR_COHERENCY                                31:29
#define LWOS32_ATTR_COHERENCY_UNCACHED                  0x00000000
#define LWOS32_ATTR_COHERENCY_CACHED                    0x00000001
#define LWOS32_ATTR_COHERENCY_WRITE_COMBINE             0x00000002
#define LWOS32_ATTR_COHERENCY_WRITE_THROUGH             0x00000003
#define LWOS32_ATTR_COHERENCY_WRITE_PROTECT             0x00000004
#define LWOS32_ATTR_COHERENCY_WRITE_BACK                0x00000005

// ATTR2 fields
#define LWOS32_ATTR2_NONE                               0x00000000

//
// DEFAULT          - Let lower level drivers pick optimal page kind.
// PREFER_NO_ZBC    - Prefer other types of compression over ZBC when
//                    selecting page kind.
// PREFER_ZBC       - Prefer ZBC over other types of compression when
//                    selecting page kind.
// REQUIRE_ONLY_ZBC - Require a page kind that enables ZBC but disables
//                    other types of compression (i.e. 2C page kind).
// INVALID          - Aliases REQUIRE_ONLY_ZBC, which is not supported
//                    by all RM implementations.
//
#define LWOS32_ATTR2_ZBC                                       1:0
#define LWOS32_ATTR2_ZBC_DEFAULT                        0x00000000
#define LWOS32_ATTR2_ZBC_PREFER_NO_ZBC                  0x00000001
#define LWOS32_ATTR2_ZBC_PREFER_ZBC                     0x00000002
#define LWOS32_ATTR2_ZBC_REQUIRE_ONLY_ZBC               0x00000003
#define LWOS32_ATTR2_ZBC_ILWALID                        0x00000003

//
// DEFAULT  - Highest performance cache policy that is coherent with the highest
//            performance CPU mapping.  Typically this is gpu cached for video
//            memory and gpu uncached for system memory.
// YES      - Enable gpu caching if supported on this surface type.  For system
//            memory this will not be coherent with direct CPU mappings.
// NO       - Disable gpu caching if supported on this surface type.
// INVALID  - Clients should never set YES and NO simultaneously.
//
#define LWOS32_ATTR2_GPU_CACHEABLE                             3:2
#define LWOS32_ATTR2_GPU_CACHEABLE_DEFAULT              0x00000000
#define LWOS32_ATTR2_GPU_CACHEABLE_YES                  0x00000001
#define LWOS32_ATTR2_GPU_CACHEABLE_NO                   0x00000002
#define LWOS32_ATTR2_GPU_CACHEABLE_ILWALID              0x00000003

//
// DEFAULT  - GPU-dependent cache policy
// YES      - Enable gpu caching for p2p mem
// NO       - Disable gpu caching for p2p mem
//
#define LWOS32_ATTR2_P2P_GPU_CACHEABLE                         5:4
#define LWOS32_ATTR2_P2P_GPU_CACHEABLE_DEFAULT          0x00000000
#define LWOS32_ATTR2_P2P_GPU_CACHEABLE_YES              0x00000001
#define LWOS32_ATTR2_P2P_GPU_CACHEABLE_NO               0x00000002

// This applies to virtual allocs only.  See LWOS46_FLAGS_32BIT_POINTER.
#define LWOS32_ATTR2_32BIT_POINTER                             6:6
#define LWOS32_ATTR2_32BIT_POINTER_DISABLE              0x00000000
#define LWOS32_ATTR2_32BIT_POINTER_ENABLE               0x00000001

//
// Indicates address colwersion to be used, which affects what
// pitch alignment needs to be used
//
#define LWOS32_ATTR2_TILED_TYPE                                7:7
#define LWOS32_ATTR2_TILED_TYPE_LINEAR                  0x00000000
#define LWOS32_ATTR2_TILED_TYPE_XY                      0x00000001

//
// Force SMMU mapping on GPU physical allocation in CheetAh
// SMMU mapping for GPU physical allocation decided internally by RM
// This attribute provide an override to RM policy for verification purposes.
//
#define LWOS32_ATTR2_SMMU_ON_GPU                               10:8
#define LWOS32_ATTR2_SMMU_ON_GPU_DEFAULT                 0x00000000
#define LWOS32_ATTR2_SMMU_ON_GPU_DISABLE                 0x00000001
#define LWOS32_ATTR2_SMMU_ON_GPU_ENABLE                  0x00000002

//
// Make comptag allocation aligned to compression cacheline size.
// Specifying this attribute will make RM allocate comptags worth an entire
// comp cacheline. The allocation will be offset aligned to number of comptags/comp cacheline.
//
#define LWOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN                11:11
#define LWOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN_OFF              0x0
#define LWOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN_ON               0x1
#define LWOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN_DEFAULT            \
                                   LWOS32_ATTR2_ALLOC_COMPCACHELINE_ALIGN_OFF

// Allocation preferred in high or low priority memory
#define LWOS32_ATTR2_PRIORITY                                 13:12
#define LWOS32_ATTR2_PRIORITY_DEFAULT                           0x0
#define LWOS32_ATTR2_PRIORITY_HIGH                              0x1
#define LWOS32_ATTR2_PRIORITY_LOW                               0x2

// PMA: Allocation is an RM internal allocation (RM-only)
#define LWOS32_ATTR2_INTERNAL                                 14:14
#define LWOS32_ATTR2_INTERNAL_NO                                0x0
#define LWOS32_ATTR2_INTERNAL_YES                               0x1

// Allocate 2C instead of 2CZ
#define LWOS32_ATTR2_PREFER_2C                                15:15
#define LWOS32_ATTR2_PREFER_2C_NO                        0x00000000
#define LWOS32_ATTR2_PREFER_2C_YES                       0x00000001

// Allocation used by display engine; RM verifies display engine has enough
// address bits or remapper available.
#define LWOS32_ATTR2_NISO_DISPLAY                             16:16
#define LWOS32_ATTR2_NISO_DISPLAY_NO                     0x00000000
#define LWOS32_ATTR2_NISO_DISPLAY_YES                    0x00000001

//
// !!WARNING!!!
//
// This flag is introduced as a temporary WAR to enable color compression
// without ZBC.
//
// This dangerous flag can be used by UMDs to instruct RM to skip the zbc
// table refcounting that RM does today, when the chosen PTE kind has ZBC
// support.
//
// Lwrrently we do not have a safe per process zbc slot management and
// refcounting mechanism between RM and UMD and hence, any process can
// access any other process's zbc entry in the global zbc table (without mask)
// Inorder to flush the ZBC table for slot reuse RM cannot track which
// process is using which zbc slot. Hence RM has a global refcount for the
// zbc table to flush and reuse the entries if the PTE kind supports zbc.
//
// This scheme poses a problem if there are apps that are persistent such as
// the desktop components that can have color compression enabled which will
// always keep the refcount active. Since these apps can live without
// ZBC, UMD can disable ZBC using masks.
//
// In such a case, if UMD so chooses to disable ZBC, this flag should be used
// to skip refcounting as by default RM would refcount the ZBC table.
//
// NOTE: There is no way for RM to enforce/police this, and we totally rely
// on UMD to use a zbc mask in the pushbuffer method to prevent apps from
// accessing the ZBC table.
//
#define LWOS32_ATTR2_ZBC_SKIP_ZBCREFCOUNT                     17:17
#define LWOS32_ATTR2_ZBC_SKIP_ZBCREFCOUNT_NO             0x00000000
#define LWOS32_ATTR2_ZBC_SKIP_ZBCREFCOUNT_YES            0x00000001

// Allocation requires ISO bandwidth guarantees
#define LWOS32_ATTR2_ISO                                      18:18
#define LWOS32_ATTR2_ISO_NO                              0x00000000
#define LWOS32_ATTR2_ISO_YES                             0x00000001

//
// Turn off blacklist feature for video memory allocation
// This attribute should be used only by Kernel client (KMD), to mask
// the blacklisted pages for the allocation. This is done so that the clients
// will manage the above masked blacklisted pages after the allocation. It will
// return to RM's pool after the allocation was free-d.RmVidHeapCtrl returns
// LW_ERR_INSUFFICIENT_PERMISSIONS if it is being called by non-kernel clients.
//

// XXX Project ReLingo - This term is marked for deletion. Use PAGE_OFFLINING.
#define LWOS32_ATTR2_BLACKLIST                                19:19
#define LWOS32_ATTR2_BLACKLIST_ON                        0x00000000
#define LWOS32_ATTR2_BLACKLIST_OFF                       0x00000001
#define LWOS32_ATTR2_PAGE_OFFLINING                           19:19
#define LWOS32_ATTR2_PAGE_OFFLINING_ON                   0x00000000
#define LWOS32_ATTR2_PAGE_OFFLINING_OFF                  0x00000001

//
// For virtual allocs to choose the HUGE page size for the region.
// LWOS32_ATTR_PAGE_SIZE must be set to _HUGE to use this.
// Lwrrently, the default huge page is 2MB, so a request with _DEFAULT
// set will always be interpreted as 2MB.
// Not supported on pre_AMPERE chips.
//
#define LWOS32_ATTR2_PAGE_SIZE_HUGE                           21:20
#define LWOS32_ATTR2_PAGE_SIZE_HUGE_DEFAULT              0x00000000
#define LWOS32_ATTR2_PAGE_SIZE_HUGE_2MB                  0x00000001
#define LWOS32_ATTR2_PAGE_SIZE_HUGE_512MB                0x00000002

// Allow read-only or read-write user CPU mappings
#define LWOS32_ATTR2_PROTECTION_USER                          22:22
#define LWOS32_ATTR2_PROTECTION_USER_READ_WRITE          0x00000000
#define LWOS32_ATTR2_PROTECTION_USER_READ_ONLY           0x00000001

// Allow read-only or read-write device mappings
#define LWOS32_ATTR2_PROTECTION_DEVICE                        23:23
#define LWOS32_ATTR2_PROTECTION_DEVICE_READ_WRITE        0x00000000
#define LWOS32_ATTR2_PROTECTION_DEVICE_READ_ONLY         0x00000001

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
// 
// Allow allocations to come from the EGM(Extended GPU Memory) pool
// Ideally, EGM will be a large carveout of TH500 CPU_MEM which
// will be accessed and managed as GPU_MEM by RM.
// Note:
// a.For GH100 HW verification purposes (without TH500), EGM memory
// will be allocated out of vidmem but mapped as peermem using
// the peerID reserved for local EGM.
// b.The LWOS32_ATTR2_USE_EGM flag will need to be used only with
// vidmem apertures. This flag if used with other apertures will
// result in an allocation failure with an invalid arg error.
// c.If a GPU does not support EGM and this bit is set, the allocation
// will fail with an invalid argument error.
// d.Lwrrently, both vidmem and egm allocations will be redirected to
// vidmem for HW verification. For production SW, we will introduce
// a new memory allocator class for EGM and a new PMA to manage EGM
// where allocations requested from EGM will be directed to this new
// PMA and will report an allocation failure if the allocation could
// not be satisfied from EGM.
//
#define LWOS32_ATTR2_USE_EGM                                 24:24
#define LWOS32_ATTR2_USE_EGM_FALSE                      0x00000000
#define LWOS32_ATTR2_USE_EGM_TRUE                       0x00000001
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
//
// Allow client allocations to go to protected/unprotected video/system memory.
// When Ampere Protected Model aka APM or Confidential Compute is enabled and
// DEFAULT flag is set by client, allocations go to protected memory. When
// protected memory is not enabled, allocations go to unprotected memory.
// If APM or CC is not enabled, it is a bug for a client to set the PROTECTED
// flag to YES
//
#define LWOS32_ATTR2_MEMORY_PROTECTION                       26:25
#define LWOS32_ATTR2_MEMORY_PROTECTION_DEFAULT          0x00000000
#define LWOS32_ATTR2_MEMORY_PROTECTION_PROTECTED        0x00000001
#define LWOS32_ATTR2_MEMORY_PROTECTION_UNPROTECTED      0x00000002
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)

//
// Force the allocation to go to guest subheap.
// This flag is used by vmiop plugin to allocate from GPA
//
#define LWOS32_ATTR2_ALLOCATE_FROM_SUBHEAP                   27:27
#define LWOS32_ATTR2_ALLOCATE_FROM_SUBHEAP_NO           0x00000000
#define LWOS32_ATTR2_ALLOCATE_FROM_SUBHEAP_YES          0x00000001

/**
 * LWOS32 ALLOC_FLAGS
 *
 *      LWOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT
 *
 *      LWOS32_ALLOC_FLAGS_FORCE_MEM_GROWS_UP
 *
 *      LWOS32_ALLOC_FLAGS_FORCE_MEM_GROWS_DOWN
 *
 *      LWOS32_ALLOC_FLAGS_FORCE_ALIGN_HOST_PAGE
 *
 *      LWOS32_ALLOC_FLAGS_FIXED_ADDRESS_ALLOCATE
 *
 *      LWOS32_ALLOC_FLAGS_BANK_HINT
 *
 *      LWOS32_ALLOC_FLAGS_BANK_FORCE
 *
 *      LWOS32_ALLOC_FLAGS_ALIGNMENT_HINT
 *
 *      LWOS32_ALLOC_FLAGS_ALIGNMENT_FORCE
 *
 *      LWOS32_ALLOC_FLAGS_BANK_GROW_UP
 *          Only relevant if bank_hint or bank_force are set
 *
 *      LWOS32_ALLOC_FLAGS_BANK_GROW_DOWN
 *          Only relevant if bank_hint or bank_force are set
 *
 *      LWOS32_ALLOC_FLAGS_LAZY
 *          Lazy allocation (deferred pde, pagetable creation)
 *
 *      LWOS32_ALLOC_FLAGS_NO_SCANOUT
 *          Set if surface will never be scanned out
 *
 *      LWOS32_ALLOC_FLAGS_PITCH_FORCE
 *          Fail alloc if supplied pitch is not aligned
 *
 *      LWOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED
 *          Memory handle provided to be associated with this allocation
 *
 *      LWOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED
 *          By default memory is mapped into the CPU address space
 *
 *      LWOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM
 *          Allocate persistent video memory
 *
 *      LWOS32_ALLOC_FLAGS_USE_BEGIN_END
 *          Use rangeBegin & rangeEnd fields in allocs other than size/range
 *
 *      LWOS32_ALLOC_FLAGS_TURBO_CIPHER_ENCRYPTED
 *          Allocate TurboCipher encrypted region
 *
 *      LWOS32_ALLOC_FLAGS_VIRTUAL
 *          Allocate virtual memory address space
 *
 *      LWOS32_ALLOC_FLAGS_FORCE_INTERNAL_INDEX
 *          Force allocation internal index
 *
 *      LWOS32_ALLOC_FLAGS_ZLWLL_COVG_SPECIFIED
 *          This flag is depreciated and allocations will fail.
 *
 *      LWOS32_ALLOC_FLAGS_EXTERNALLY_MANAGED
 *          Must be used with LWOS32_ALLOC_FLAGS_VIRTUAL.
 *          Page tables for this allocation will be managed outside of RM.
 *
 *      LWOS32_ALLOC_FLAGS_FORCE_DEDICATED_PDE
 *
 *      LWOS32_ALLOC_FLAGS_PROTECTED
 *          Allocate in a protected memory region if available
 *
 *      LWOS32_ALLOC_FLAGS_KERNEL_MAPPING_MAP
 *          Map kernel os descriptor
 *
 *      LWOS32_ALLOC_FLAGS_MAXIMIZE_ADDRESS_SPACE
 *          On WDDM all address spaces are created with MINIMIZE_PTETABLE_SIZE
 *          to reduce the overhead of private address spaces per application,
 *          at the cost of holes in the virtual address space.
 *
 *          Shaders have short pointers that are required to be within a
 *          GPU dependent 32b range.
 *
 *          MAXIMIZE_ADDRESS_SPACE will reverse the MINIMIZE_PTE_TABLE_SIZE
 *          flag with certain restrictions:
 *          - This flag only has an effect when the allocation has the side
 *            effect of creating a new PDE.  It does not affect existing PDEs.
 *          - The first few PDEs of the address space are kept minimum to allow
 *            small applications to use fewer resources.
 *          - By default this operations on the 0-4GB address range.
 *          - If USE_BEGIN_END is specified the setting will apply to the
 *            specified range instead of the first 4GB.
 *
 *      LWOS32_ALLOC_FLAGS_SPARSE
 *          Denote that a virtual address range is "sparse". Must be used with
 *          LWOS32_ALLOC_FLAGS_VIRTUAL. Creation of a "sparse" virtual address range
 *          denotes that an unmapped virtual address range should "not" fault but simply
 *          return 0's.
 *
 *      LWOS32_ALLOC_FLAGS_ALLOCATE_KERNEL_PRIVILEGED
 *          This a special flag that can be used only by kernel(root) clients
 *          to allocate memory out of a protected region of the address space
 *          If this flag is set by non kernel clients then the allocation will
 *          fail.
 *
 *      LWOS32_ALLOC_FLAGS_SKIP_RESOURCE_ALLOC
 *
 *      LWOS32_ALLOC_FLAGS_PREFER_PTES_IN_SYSMEMORY
 *          If new pagetable need to be allocated prefer them in sysmem (if supported by the gpu)
 *
 *      LWOS32_ALLOC_FLAGS_SKIP_ALIGN_PAD
 *          As per KMD request to eliminate extra allocation
 *
 *      LWOS32_ALLOC_FLAGS_WPR1
 *          Allocate in a WPR1 region if available
 *
 *      LWOS32_ALLOC_FLAGS_ZLWLL_DONT_ALLOCATE_SHARED_1X
 *          If using zlwll sharing and this surface is fsaa, then don't allocate an additional non-FSAA region.
 *
 *      LWOS32_ALLOC_FLAGS_WPR2
 *          Allocate in a WPR1 region if available
 */
#define LWOS32_ALLOC_FLAGS_IGNORE_BANK_PLACEMENT        0x00000001
#define LWOS32_ALLOC_FLAGS_FORCE_MEM_GROWS_UP           0x00000002
#define LWOS32_ALLOC_FLAGS_FORCE_MEM_GROWS_DOWN         0x00000004
#define LWOS32_ALLOC_FLAGS_FORCE_ALIGN_HOST_PAGE        0x00000008
#define LWOS32_ALLOC_FLAGS_FIXED_ADDRESS_ALLOCATE       0x00000010
#define LWOS32_ALLOC_FLAGS_BANK_HINT                    0x00000020
#define LWOS32_ALLOC_FLAGS_BANK_FORCE                   0x00000040
#define LWOS32_ALLOC_FLAGS_ALIGNMENT_HINT               0x00000080
#define LWOS32_ALLOC_FLAGS_ALIGNMENT_FORCE              0x00000100
#define LWOS32_ALLOC_FLAGS_BANK_GROW_UP                 0x00000000
#define LWOS32_ALLOC_FLAGS_BANK_GROW_DOWN               0x00000200
#define LWOS32_ALLOC_FLAGS_LAZY                         0x00000400
// unused                                               0x00000800
#define LWOS32_ALLOC_FLAGS_NO_SCANOUT                   0x00001000
#define LWOS32_ALLOC_FLAGS_PITCH_FORCE                  0x00002000
#define LWOS32_ALLOC_FLAGS_MEMORY_HANDLE_PROVIDED       0x00004000
#define LWOS32_ALLOC_FLAGS_MAP_NOT_REQUIRED             0x00008000
#define LWOS32_ALLOC_FLAGS_PERSISTENT_VIDMEM            0x00010000
#define LWOS32_ALLOC_FLAGS_USE_BEGIN_END                0x00020000
#define LWOS32_ALLOC_FLAGS_TURBO_CIPHER_ENCRYPTED       0x00040000
#define LWOS32_ALLOC_FLAGS_VIRTUAL                      0x00080000
#define LWOS32_ALLOC_FLAGS_FORCE_INTERNAL_INDEX         0x00100000
#define LWOS32_ALLOC_FLAGS_ZLWLL_COVG_SPECIFIED         0x00200000
#define LWOS32_ALLOC_FLAGS_EXTERNALLY_MANAGED           0x00400000
#define LWOS32_ALLOC_FLAGS_FORCE_DEDICATED_PDE          0x00800000
#define LWOS32_ALLOC_FLAGS_PROTECTED                    0x01000000
#define LWOS32_ALLOC_FLAGS_KERNEL_MAPPING_MAP           0x02000000 // TODO BUG 2488679: fix alloc flag aliasing
#define LWOS32_ALLOC_FLAGS_MAXIMIZE_ADDRESS_SPACE       0x02000000
#define LWOS32_ALLOC_FLAGS_SPARSE                       0x04000000
#define LWOS32_ALLOC_FLAGS_USER_READ_ONLY               0x04000000 // TODO BUG 2488682: remove this after KMD transition
#define LWOS32_ALLOC_FLAGS_DEVICE_READ_ONLY             0x08000000 // TODO BUG 2488682: remove this after KMD transition
#define LWOS32_ALLOC_FLAGS_ALLOCATE_KERNEL_PRIVILEGED   0x08000000
#define LWOS32_ALLOC_FLAGS_SKIP_RESOURCE_ALLOC          0x10000000
#define LWOS32_ALLOC_FLAGS_PREFER_PTES_IN_SYSMEMORY     0x20000000
#define LWOS32_ALLOC_FLAGS_SKIP_ALIGN_PAD               0x40000000
#define LWOS32_ALLOC_FLAGS_WPR1                         0x40000000 // TODO BUG 2488672: fix alloc flag aliasing
#define LWOS32_ALLOC_FLAGS_ZLWLL_DONT_ALLOCATE_SHARED_1X 0x80000000
#define LWOS32_ALLOC_FLAGS_WPR2                         0x80000000 // TODO BUG 2488672: fix alloc flag aliasing

// Internal flags used for RM's allocation paths
#define LWOS32_ALLOC_INTERNAL_FLAGS_CLIENTALLOC         0x00000001 // RM internal flags - not sure if this should be exposed even. Keeping it here.
#define LWOS32_ALLOC_INTERNAL_FLAGS_SKIP_SCRUB          0x00000004 // RM internal flags - not sure if this should be exposed even. Keeping it here.
#define LWOS32_ALLOC_FLAGS_MAXIMIZE_4GB_ADDRESS_SPACE LWOS32_ALLOC_FLAGS_MAXIMIZE_ADDRESS_SPACE // Legacy name

//
// Bitmask of flags that are only valid for virtual allocations.
//
#define LWOS32_ALLOC_FLAGS_VIRTUAL_ONLY         ( \
    LWOS32_ALLOC_FLAGS_VIRTUAL                  | \
    LWOS32_ALLOC_FLAGS_LAZY                     | \
    LWOS32_ALLOC_FLAGS_EXTERNALLY_MANAGED       | \
    LWOS32_ALLOC_FLAGS_SPARSE                   | \
    LWOS32_ALLOC_FLAGS_MAXIMIZE_ADDRESS_SPACE   | \
    LWOS32_ALLOC_FLAGS_PREFER_PTES_IN_SYSMEMORY )

// COMPR_COVG_* allows for specification of what compression resources
// are required (_MIN) and necessary (_MAX).  Default behavior is for
// RM to provide as much as possible, including none if _ANY is allowed.
// Values for min/max are (0-100, a %) * _COVG_SCALE (so max value is
// 100*100==10000).  _START is used to specify the % offset into the
// region to begin the requested coverage.
// _COVG_BITS allows specification of the number of comptags per ROP tile.
// A value of 0 is default and allows RM to choose based upon MMU/FB rules.
// All other values for _COVG_BITS are arch-specific.
// Note: LWOS32_ATTR_COMPR_COVG_PROVIDED must be set for this feature
// to be available (verif-only).
#define LWOS32_ALLOC_COMPR_COVG_SCALE                           10
#define LWOS32_ALLOC_COMPR_COVG_BITS                           1:0
#define LWOS32_ALLOC_COMPR_COVG_BITS_DEFAULT            0x00000000
#define LWOS32_ALLOC_COMPR_COVG_BITS_1                  0x00000001
#define LWOS32_ALLOC_COMPR_COVG_BITS_2                  0x00000002
#define LWOS32_ALLOC_COMPR_COVG_BITS_4                  0x00000003
#define LWOS32_ALLOC_COMPR_COVG_MAX                           11:2
#define LWOS32_ALLOC_COMPR_COVG_MIN                          21:12
#define LWOS32_ALLOC_COMPR_COVG_START                        31:22


// Note: LWOS32_ALLOC_FLAGS_ZLWLL_COVG_SPECIFIED must be set for this feature
// to be enabled.
// If FALLBACK_ALLOW is set, a fallback from LOW_RES_Z or LOW_RES_ZS
// to HIGH_RES_Z is allowed if the surface can't be fully covered.
#define LWOS32_ALLOC_ZLWLL_COVG_FORMAT                         3:0
#define LWOS32_ALLOC_ZLWLL_COVG_FORMAT_LOW_RES_Z        0x00000000
#define LWOS32_ALLOC_ZLWLL_COVG_FORMAT_HIGH_RES_Z       0x00000002
#define LWOS32_ALLOC_ZLWLL_COVG_FORMAT_LOW_RES_ZS       0x00000003
#define LWOS32_ALLOC_ZLWLL_COVG_FALLBACK                       4:4
#define LWOS32_ALLOC_ZLWLL_COVG_FALLBACK_DISALLOW       0x00000000
#define LWOS32_ALLOC_ZLWLL_COVG_FALLBACK_ALLOW          0x00000001


// _ALLOC_COMPTAG_OFFSET allows the caller to specify the starting
// offset for the comptags for a given surface, primarily for test only.
// To specify an offset, set _USAGE_FIXED or _USAGE_MIN in conjunction
// with _START.
//
// _USAGE_FIXED sets a surface's comptagline to start at the given
// starting value.  If the offset has already been assigned, then
// the alloc call fails.
//
// _USAGE_MIN sets a surface's comptagline to start at the given
// starting value or higher, depending on comptagline availability.
// In this case, if the offset has already been assigned, the next
// available comptagline (in increasing order) will be assigned.
//
// For Fermi, up to 2^17 comptags may be allowed, but the actual,
// usable limit depends on the size of the compbit backing store.
//
// For Pascal, up to 2 ^ 18 comptags may be allowed
// From Turing. up to 2 ^ 20 comptags may be allowed
//
// See also field ctagOffset in struct LWOS32_PARAMETERS.
#define LWOS32_ALLOC_COMPTAG_OFFSET_START                     19:0
#define LWOS32_ALLOC_COMPTAG_OFFSET_START_DEFAULT       0x00000000
#define LWOS32_ALLOC_COMPTAG_OFFSET_USAGE                    31:30
#define LWOS32_ALLOC_COMPTAG_OFFSET_USAGE_DEFAULT       0x00000000
#define LWOS32_ALLOC_COMPTAG_OFFSET_USAGE_OFF           0x00000000
#define LWOS32_ALLOC_COMPTAG_OFFSET_USAGE_FIXED         0x00000001
#define LWOS32_ALLOC_COMPTAG_OFFSET_USAGE_MIN           0x00000002


// REALLOC flags field
#define LWOS32_REALLOC_FLAGS_GROW_ALLOCATION            0x00000000
#define LWOS32_REALLOC_FLAGS_SHRINK_ALLOCATION          0x00000001
#define LWOS32_REALLOC_FLAGS_REALLOC_UP                 0x00000000 // towards/from high memory addresses
#define LWOS32_REALLOC_FLAGS_REALLOC_DOWN               0x00000002 // towards/from memory address 0

// RELEASE_COMPR, REACQUIRE_COMPR flags field
#define LWOS32_RELEASE_COMPR_FLAGS_MEMORY_HANDLE_PROVIDED   0x000000001

#define LWOS32_REACQUIRE_COMPR_FLAGS_MEMORY_HANDLE_PROVIDED 0x000000001


// FREE flags field
#define LWOS32_FREE_FLAGS_MEMORY_HANDLE_PROVIDED        0x00000001

// DUMP flags field
#define LWOS32_DUMP_FLAGS_TYPE                                 1:0
#define LWOS32_DUMP_FLAGS_TYPE_FB                       0x00000000
#define LWOS32_DUMP_FLAGS_TYPE_CLIENT_PD                0x00000001
#define LWOS32_DUMP_FLAGS_TYPE_CLIENT_VA                0x00000002
#define LWOS32_DUMP_FLAGS_TYPE_CLIENT_VAPTE             0x00000003

#define LWOS32_BLOCK_TYPE_FREE                          0xFFFFFFFF
#define LWOS32_ILWALID_BLOCK_FREE_OFFSET                0xFFFFFFFF

#define LWOS32_MEM_TAG_NONE                             0x00000000

/*
 * LW_CONTEXT_DMA_ALLOCATION_PARAMS - Allocation params to create context dma
   through LwRmAlloc.
 */
typedef struct
{
    LwHandle hSubDevice;
    LwV32    flags;
    LwHandle hMemory;
    LwU64    offset LW_ALIGN_BYTES(8);
    LwU64    limit LW_ALIGN_BYTES(8);
} LW_CONTEXT_DMA_ALLOCATION_PARAMS;

/*
 * LW_MEMORY_ALLOCATION_PARAMS - Allocation params to create memory through
 * LwRmAlloc. Flags are populated with LWOS32_ defines.
 */
typedef struct
{
    LwU32     owner;                        // [IN]  - memory owner ID
    LwU32     type;                         // [IN]  - surface type, see below TYPE* defines
    LwU32     flags;                        // [IN]  - allocation modifier flags, see below ALLOC_FLAGS* defines

    LwU32     width;                        // [IN]  - width of surface in pixels
    LwU32     height;                       // [IN]  - height of surface in pixels
    LwS32     pitch;                        // [IN/OUT] - desired pitch AND returned actual pitch allocated

    LwU32     attr;                         // [IN/OUT] - surface attributes requested, and surface attributes allocated
    LwU32     attr2;                        // [IN/OUT] - surface attributes requested, and surface attributes allocated

    LwU32     format;                       // [IN/OUT] - format requested, and format allocated
    LwU32     comprCovg;                    // [IN/OUT] - compr covg requested, and allocated
    LwU32     zlwllCovg;                    // [OUT] - zlwll covg allocated

    LwU64     rangeLo   LW_ALIGN_BYTES(8);  // [IN]  - allocated memory will be limited to the range
    LwU64     rangeHi   LW_ALIGN_BYTES(8);  // [IN]  - from rangeBegin to rangeEnd, inclusive.

    LwU64     size      LW_ALIGN_BYTES(8);  // [IN/OUT]  - size of allocation - also returns the actual size allocated
    LwU64     alignment LW_ALIGN_BYTES(8);  // [IN]  - requested alignment - LWOS32_ALLOC_FLAGS_ALIGNMENT* must be on
    LwU64     offset    LW_ALIGN_BYTES(8);  // [IN/OUT]  - desired offset if LWOS32_ALLOC_FLAGS_FIXED_ADDRESS_ALLOCATE is on AND returned offset
    LwU64     limit     LW_ALIGN_BYTES(8);  // [OUT] - returned surface limit
    LwP64     address   LW_ALIGN_BYTES(8);  // [OUT] - returned address

    LwU32     ctagOffset;                   // [IN] - comptag offset for this surface (see LWOS32_ALLOC_COMPTAG_OFFSET)
    LwHandle  hVASpace;                     // [IN]  - VASpace handle. Used when flag is VIRTUAL.

    LwU32     internalflags;                // [IN]  - internal flags to change allocation behaviors from internal paths

    LwU32     tag;                          // [IN] - memory tag used for debugging
} LW_MEMORY_ALLOCATION_PARAMS;

/*
 * LW_OS_DESC_MEMORY_ALLOCATION_PARAMS - Allocation params to create OS
 * described memory through LwRmAlloc. Flags are populated with LWOS32_ defines.
 */
typedef struct
{
    LwU32     type;                         // [IN]  - surface type, see below TYPE* defines
    LwU32     flags;                        // [IN]  - allocation modifier flags, see below ALLOC_FLAGS* defines
    LwU32     attr;                         // [IN]  - attributes for memory placement/properties, see below
    LwU32     attr2;                        // [IN]  - attributes GPU_CACHEABLE
    LwP64     descriptor LW_ALIGN_BYTES(8); // [IN]  - descriptor address
    LwU64     limit      LW_ALIGN_BYTES(8); // [IN]  - allocated size -1
    LwU32     descriptorType;               // [IN]  - descriptor type(Virtual | lwmap Handle)
    LwU32     tag;                          // [IN]  - memory tag used for debugging
} LW_OS_DESC_MEMORY_ALLOCATION_PARAMS;

/*
 * LW_USER_LOCAL_DESC_MEMORY_ALLOCATION_PARAMS - Allocation params to create a memory
 * object from user allocated video memory. Flags are populated with LWOS32_*
 * defines.
 */
typedef struct
{
    LwU32     flags;                        // [IN]  - allocation modifier flags, see LWOS02_FLAGS* defines
    LwU64     physAddr   LW_ALIGN_BYTES(8); // [IN]  - physical address
    LwU64     size       LW_ALIGN_BYTES(8); // [IN]  - mem size
    LwU32     tag;                          // [IN]  - memory tag used for debugging
    LwBool    bGuestAllocated;              // [IN]  - Set if memory is guest allocated (mapped by VMMU)
} LW_USER_LOCAL_DESC_MEMORY_ALLOCATION_PARAMS;

/*
 * LW_MEMORY_HW_RESOURCES_ALLOCATION_PARAMS - Allocation params to create
 * memory HW resources through LwRmAlloc. Flags are populated with LWOS32_
 * defines.
 */
typedef struct
{
    LwU32     owner;                        // [IN]  - memory owner ID
    LwU32     flags;                        // [IN]  - allocation modifier flags, see below ALLOC_FLAGS* defines
    LwU32     type;                         // [IN]  - surface type, see below TYPE* defines

    LwU32     attr;                         // [IN/OUT] - surface attributes requested, and surface attributes allocated
    LwU32     attr2;                        // [IN/OUT] - surface attributes requested, and surface attributes allocated

    LwU32     height;
    LwU32     width;
    LwU32     pitch;
    LwU32     alignment;
    LwU32     comprCovg;
    LwU32     zlwllCovg;

    LwU32     kind;

    LwP64     bindResultFunc  LW_ALIGN_BYTES(8);  // BindResultFunc
    LwP64     pHandle         LW_ALIGN_BYTES(8);
    LwU64     osDeviceHandle  LW_ALIGN_BYTES(8);
    LwU64     size            LW_ALIGN_BYTES(8);
    LwU64     allocAddr       LW_ALIGN_BYTES(8);

    // [out] from GMMU_COMPR_INFO in drivers/common/shared/inc/mmu/gmmu_fmt.h
    LwU32 compPageShift;
    LwU32 compressedKind;
    LwU32 compTagLineMin;
    LwU32 compPageIndexLo;
    LwU32 compPageIndexHi;
    LwU32 compTagLineMultiplier;

    // [out] fallback uncompressed kind.
    LwU32  uncompressedKind;

    LwU32     tag;                          // [IN] - memory tag used for debugging
} LW_MEMORY_HW_RESOURCES_ALLOCATION_PARAMS;

/* function OS33 */
#define LW04_MAP_MEMORY                                 (0x00000021)

// Legacy map and unmap memory flags that don't use DRF_DEF scheme
#define LW04_MAP_MEMORY_FLAGS_NONE                      (0x00000000)
#define LW04_MAP_MEMORY_FLAGS_USER                      (0x00004000)

// New map and unmap memory flags.  These flags are used for both LwRmMapMemory
// and for LwRmUnmapMemory.

// Mappings can have restricted permissions (read-only, write-only).  Some
// RM implementations may choose to ignore these flags, or they may work
// only for certain memory spaces (system, AGP, video memory); in such cases,
// you may get a read/write mapping even if you asked for a read-only or
// write-only mapping.
#define LWOS33_FLAGS_ACCESS                                        1:0
#define LWOS33_FLAGS_ACCESS_READ_WRITE                             (0x00000000)
#define LWOS33_FLAGS_ACCESS_READ_ONLY                              (0x00000001)
#define LWOS33_FLAGS_ACCESS_WRITE_ONLY                             (0x00000002)

// Persistent mappings are no longer supported
#define LWOS33_FLAGS_PERSISTENT                                    4:4
#define LWOS33_FLAGS_PERSISTENT_DISABLE                            (0x00000000)
#define LWOS33_FLAGS_PERSISTENT_ENABLE                             (0x00000001)

// This flag is a hack to work around bug 150889.  It disables the error
// checking in the RM that verifies that the client is not trying to map
// memory past the end of the memory object.  This error checking needs to
// be shut off in some cases for a PAE bug workaround in certain kernels.
#define LWOS33_FLAGS_SKIP_SIZE_CHECK                               8:8
#define LWOS33_FLAGS_SKIP_SIZE_CHECK_DISABLE                       (0x00000000)
#define LWOS33_FLAGS_SKIP_SIZE_CHECK_ENABLE                        (0x00000001)

// Normally, a mapping is created in the same memory space as the client -- in
// kernel space for a kernel RM client, or in user space for a user RM client.
// However, a kernel RM client can specify MEM_SPACE:USER to create a user-space
// mapping in the current RM client.
#define LWOS33_FLAGS_MEM_SPACE                                     14:14
#define LWOS33_FLAGS_MEM_SPACE_CLIENT                              (0x00000000)
#define LWOS33_FLAGS_MEM_SPACE_USER                                (0x00000001)

// The client can ask for direct memory mapping (i.e. no BAR1) if remappers and
// blocklinear are not required. RM can do direct mapping in this case if
// carveout is available.
// DEFAULT:   Use direct mapping if available and no address/data translation
//            is necessary; reflected otherwise
// DIRECT:    Use direct mapping if available, even if some translation is
//            necessary (the client is responsible for translation)
// REFLECTED: Always use reflected mapping
#define LWOS33_FLAGS_MAPPING                                       16:15
#define LWOS33_FLAGS_MAPPING_DEFAULT                               (0x00000000)
#define LWOS33_FLAGS_MAPPING_DIRECT                                (0x00000001)
#define LWOS33_FLAGS_MAPPING_REFLECTED                             (0x00000002)

// The client requests a fifo mapping but doesn't know the offset or length
// DEFAULT:   Do error check length and offset
// ENABLE:    Don't error check length and offset but have the RM fill them in
#define LWOS33_FLAGS_FIFO_MAPPING                                  17:17
#define LWOS33_FLAGS_FIFO_MAPPING_DEFAULT                          (0x00000000)
#define LWOS33_FLAGS_FIFO_MAPPING_ENABLE                           (0x00000001)

// The client can require that the CPU mapping be to a specific CPU address
// (akin to MAP_FIXED for mmap).
// DISABLED: RM will map the allocation at a CPU VA that RM selects.
// ENABLED:  RM will map the allocation at the CPU VA specified by the address
//           pass-back parameter to LwRmMapMemory
// NOTES:
// - Used for controlling CPU addresses in LWCA's unified CPU+GPU virtual
//   address space
// - Only valid on LwRmMapMemory
// - Only implemented on Linux
#define LWOS33_FLAGS_MAP_FIXED                                     18:18
#define LWOS33_FLAGS_MAP_FIXED_DISABLE                             (0x00000000)
#define LWOS33_FLAGS_MAP_FIXED_ENABLE                              (0x00000001)

// The client can specify to the RM that the CPU virtual address range for an
// allocation should remain reserved after the allocation is unmapped.
// DISABLE:   When this mapping is destroyed, RM will unmap the CPU virtual
//            address space used by this allocation.  On Linux this corresponds
//            to calling munmap on the CPU VA region.
// ENABLE:    When the map object is freed, RM will leave the CPU virtual
//            address space used by allocation reserved.  On Linux this means
//            that RM will overwrite the previous mapping with an anonymous
//            mapping of instead calling munmap.
// NOTES:
// - When combined with MAP_FIXED, this allows the client to exert
//   significant control over the CPU heap
// - Used in LWCA's unified CPU+GPU virtual address space
// - Only valid on LwRmMapMemory (specifies RM's behavior whenever the
//   mapping is destroyed, regardless of mechanism)
// - Only implemented on Linux
#define LWOS33_FLAGS_RESERVE_ON_UNMAP                              19:19
#define LWOS33_FLAGS_RESERVE_ON_UNMAP_DISABLE                      (0x00000000)
#define LWOS33_FLAGS_RESERVE_ON_UNMAP_ENABLE                       (0x00000001)

// Systems with a coherent LWLINK2 connection between the CPU and GPU
// have the option of directly mapping video memory over that connection.
// During mapping you may specify a preference.
//
#define LWOS33_FLAGS_BUS                               21:20
#define LWOS33_FLAGS_BUS_ANY                           0
#define LWOS33_FLAGS_BUS_LWLINK_COHERENT               1
#define LWOS33_FLAGS_BUS_PCIE                          2

// Internal use only
#define LWOS33_FLAGS_OS_DESCRIPTOR                                 22:22
#define LWOS33_FLAGS_OS_DESCRIPTOR_DISABLE                         (0x00000000)
#define LWOS33_FLAGS_OS_DESCRIPTOR_ENABLE                          (0x00000001)

/* parameters */
typedef struct
{
    LwHandle hClient;
    LwHandle hDevice;            // device or sub-device handle
    LwHandle hMemory;            // handle to memory object if provided -- NULL if not
    LwU64    offset LW_ALIGN_BYTES(8);
    LwU64    length LW_ALIGN_BYTES(8);
    LwP64    pLinearAddress LW_ALIGN_BYTES(8);     // pointer for returned address
    LwU32    status;
    LwU32    flags;
} LWOS33_PARAMETERS;


/* function OS34 */
#define LW04_UNMAP_MEMORY                                          (0x00000022)

/* parameters */
typedef struct
{
    LwHandle hClient;
    LwHandle hDevice;
    LwHandle hMemory;
    LwP64    pLinearAddress LW_ALIGN_BYTES(8);     // ptr to virtual address of mapped memory
    LwU32    status;
    LwU32    flags;
} LWOS34_PARAMETERS;

/* function OS37 */
#define LW04_UPDATE_CONTEXT_DMA                                    (0x00000025)

/* parameters */
typedef struct
{
    LwHandle hClient;
    LwHandle hDevice;
    LwHandle hDma;
    LwHandle hDmaPteArray;          // ctx dma for pte's
    LwV32    dmaFirstPage;          // first page in "real" context dma to update
    LwV32    pteArrayOffset;        // first pte to use from input pte array
    LwV32    pteCount;              // count of PTE entries to update
    LwHandle hResourceHandle;       // bind data handle
    LwV32    status;
} LWOS37_PARAMETERS;

/* function OS38 */
#define LW04_ACCESS_REGISTRY                                       (0x00000026)

/* parameter values */
#define LWOS38_ACCESS_TYPE_READ_DWORD                                        1
#define LWOS38_ACCESS_TYPE_WRITE_DWORD                                       2
#define LWOS38_ACCESS_TYPE_READ_BINARY                                       6
#define LWOS38_ACCESS_TYPE_WRITE_BINARY                                      7

#define LWOS38_MAX_REGISTRY_STRING_LENGTH                                  256
#define LWOS38_MAX_REGISTRY_BINARY_LENGTH                                  256

/* parameters */
typedef struct
{
    LwHandle hClient;
    LwHandle hObject;
    LwV32    AccessType;

    LwV32    DevNodeLength;
    LwP64    pDevNode LW_ALIGN_BYTES(8);

    LwV32    ParmStrLength;
    LwP64    pParmStr LW_ALIGN_BYTES(8);

    LwV32    BinaryDataLength;
    LwP64    pBinaryData LW_ALIGN_BYTES(8);

    LwV32    Data;
    LwV32    Entry;
    LwV32    status;
} LWOS38_PARAMETERS;

#define  LW04_ALLOC_CONTEXT_DMA                                    (0x00000027)

/* parameter values are the same as LWOS03 -- not repeated here */

/* parameters */
typedef struct
{
    LwHandle hObjectParent;
    LwHandle hSubDevice;
    LwHandle hObjectNew;
    LwV32    hClass;
    LwV32    flags;
    LwU32    selector;
    LwHandle hMemory;
    LwU64    offset LW_ALIGN_BYTES(8);
    LwU64    limit LW_ALIGN_BYTES(8);
    LwV32    status;
} LWOS39_PARAMETERS;


#define LW04_GET_EVENT_DATA                                        (0x00000028)

typedef struct
{
    LwHandle hObject;
    LwV32    NotifyIndex;

    //
    // Holds same information as that of lwgputypes.h::LwNotification's
    // info32 and info16.
    //
    LwV32    info32;
    LwU16    info16;
} LwUnixEvent;

/* parameters */
typedef struct
{
    LwP64 pEvent LW_ALIGN_BYTES(8);
    LwV32 MoreEvents;
    LwV32 status;
} LWOS41_PARAMETERS;

/* function LWOS43        -- deleted 4/09 */
/* #define LW04_UNIFIED_FREE                                       (0x0000002B)  */


#define  LWSIM01_BUS_XACT                                          (0x0000002C)

/* parameters */
typedef struct
{
    LwHandle hClient; // n/a lwrrently
    LwHandle hDevice; // n/a lwrrently
    LwU32    offset;  // phy bus offset
    LwU32    bar;     // ~0 := phy addr, {0..2} specify gpu bar
    LwU32    bytes;   // # of bytes
    LwU32    write;   // 0 := read request
    LwU32    data;    // in/out based upon 'write'
    LwU32    status;
} LWOS2C_PARAMETERS;

/* function LWOS2D        -- deleted 4/09 */
/* #define  LWSIM01_BUS_GET_IFACES                                 (0x0000002D)  */


/* function OS46 */
#define LW04_MAP_MEMORY_DMA                                        (0x0000002E)

/* parameter values */
#define LWOS46_FLAGS_ACCESS                                        1:0
#define LWOS46_FLAGS_ACCESS_READ_WRITE                             (0x00000000)
#define LWOS46_FLAGS_ACCESS_READ_ONLY                              (0x00000001)
#define LWOS46_FLAGS_ACCESS_WRITE_ONLY                             (0x00000002)

//
// Compute shaders support both 32b and 64b pointers. This allows mappings
// to be restricted to the bottom 4GB of the address space. How _DISABLE
// is handled is chip specific and may force a pointer above 4GB.
//
#define LWOS46_FLAGS_32BIT_POINTER                                 2:2
#define LWOS46_FLAGS_32BIT_POINTER_DISABLE                         (0x00000000)
#define LWOS46_FLAGS_32BIT_POINTER_ENABLE                          (0x00000001)

#define LWOS46_FLAGS_PAGE_KIND                                     3:3
#define LWOS46_FLAGS_PAGE_KIND_PHYSICAL                            (0x00000000)
#define LWOS46_FLAGS_PAGE_KIND_VIRTUAL                             (0x00000001)

#define LWOS46_FLAGS_CACHE_SNOOP                                   4:4
#define LWOS46_FLAGS_CACHE_SNOOP_DISABLE                           (0x00000000)
#define LWOS46_FLAGS_CACHE_SNOOP_ENABLE                            (0x00000001)

// The client requests a CPU kernel mapping so that SW class could use it
// DEFAULT: Don't map CPU address
// ENABLE:  Map CPU address
#define LWOS46_FLAGS_KERNEL_MAPPING                                5:5
#define LWOS46_FLAGS_KERNEL_MAPPING_NONE                           (0x00000000)
#define LWOS46_FLAGS_KERNEL_MAPPING_ENABLE                         (0x00000001)

//
// Compute shader access control.
// GPUs that support this feature set the LW0080_CTRL_DMA_CAPS_SHADER_ACCESS_SUPPORTED
// property. These were first supported in Kepler. _DEFAULT will match the ACCESS field.
//
#define LWOS46_FLAGS_SHADER_ACCESS                                 7:6
#define LWOS46_FLAGS_SHADER_ACCESS_DEFAULT                         (0x00000000)
#define LWOS46_FLAGS_SHADER_ACCESS_READ_ONLY                       (0x00000001)
#define LWOS46_FLAGS_SHADER_ACCESS_WRITE_ONLY                      (0x00000002)
#define LWOS46_FLAGS_SHADER_ACCESS_READ_WRITE                      (0x00000003)

//
// How the PAGE_SIZE field is interpreted is architecture specific.
//
// On Lwrie chips it is ignored.
//
// On Tesla it is used to guide is used to select which type PDE
// to use. By default the RM will select 4KB for system memory
// and BIG (64KB) for video memory. BOTH is not supported.
//
// Likewise on Fermi this used to select the PDE type. Fermi cannot
// mix page sizes to a single mapping so the page size is determined
// at surface alloation time. 4KB or BIG may be specified but they
// must match the page size selected at allocation time.  DEFAULT
// allows the RM to select either a single page size or both PDE,
// while BOTH forces the RM to select a dual page size PDE.
//
// BIG_PAGE  = 64 KB on PASCAL
//           = 64 KB or 128 KB on pre_PASCAL chips
//
// HUGE_PAGE = 2 MB on PASCAL
//           = not supported on pre_PASCAL chips.
//
#define LWOS46_FLAGS_PAGE_SIZE                                     11:8
#define LWOS46_FLAGS_PAGE_SIZE_DEFAULT                             (0x00000000)
#define LWOS46_FLAGS_PAGE_SIZE_4KB                                 (0x00000001)
#define LWOS46_FLAGS_PAGE_SIZE_BIG                                 (0x00000002)
#define LWOS46_FLAGS_PAGE_SIZE_BOTH                                (0x00000003)
#define LWOS46_FLAGS_PAGE_SIZE_HUGE                                (0x00000004)

#ifdef LW_VERIF_FEATURES
#define LWOS46_FLAGS_PRIV                                          12:12
#define LWOS46_FLAGS_PRIV_DISABLE                                  (0x00000000)
#define LWOS46_FLAGS_PRIV_ENABLE                                   (0x00000001)
#endif

// Some systems allow the device to use the system L3 cache when accessing the
// system memory. For example, the iGPU on T19X can allocate from the system L3
// provided the SoC L3 cache is configured for device allocation.
//
// LWOS46_FLAGS_SYSTEM_L3_ALLOC_DEFAULT - Use the default L3 allocation
// policy. When using this policy, device memory access will be coherent with
// non-snooping devices such as the display on CheetAh.
//
// LWOS46_FLAGS_SYSTEM_L3_ALLOC_ENABLE_HINT - Enable L3 allocation if possible.
// When L3 allocation is enabled, device memory access may be cached, and the
// memory access will be coherent only with other snoop-enabled access. This
// flag is a hint and will be ignored if the system does not support L3
// allocation for the device. LWOS46_FLAGS_CACHE_SNOOP_ENABLE must also be set
// for this flag to be effective.
//
// Note: This flag is implemented only by rmapi_tegra. It is not implemented by
// Resman.
//
#define LWOS46_FLAGS_SYSTEM_L3_ALLOC                               13:13
#define LWOS46_FLAGS_SYSTEM_L3_ALLOC_DEFAULT                       (0x00000000)
#define LWOS46_FLAGS_SYSTEM_L3_ALLOC_ENABLE_HINT                   (0x00000001)

#define LWOS46_FLAGS_DMA_OFFSET_GROWS                              14:14
#define LWOS46_FLAGS_DMA_OFFSET_GROWS_UP                           (0x00000000)
#define LWOS46_FLAGS_DMA_OFFSET_GROWS_DOWN                         (0x00000001)

//
// DMA_OFFSET_FIXED is overloaded for two purposes.
//
// 1. For CTXDMA mappings that use DMA_UNICAST_REUSE_ALLOC_FALSE,
//    DMA_OFFSET_FIXED_TRUE indicates to use the dmaOffset parameter
//    for a fixed address allocation out of the VA space heap.
//    DMA_OFFSET_FIXED_FALSE indicates dmaOffset input will be ignored.
//
// 2. For CTXDMA mappings that use DMA_UNICAST_REUSE_ALLOC_TRUE and
//    for *ALL* non-CTXDMA mappings, DMA_OFFSET_FIXED_TRUE indicates
//    to treat the input dmaOffset as an absolute virtual address
//    instead of an offset relative to the virtual allocation being
//    mapped into. Whether relative or absolute, the resulting
//    virtual address *must* be contained within the specified
//    virtual allocation.
//
//    Internally, it is also required that the virtual address be aligned
//    to the page size of the mapping (obviously cannot map sub-pages).
//    For client flexibility the physical offset does not require page alignment.
//    This is handled by adding the physical misalignment
//    (internally called pteAdjust) to the returned virtual address.
//    The *input* dmaOffset can account for this pteAdjust (or not),
//    but the returned virtual address always will.
//
#define LWOS46_FLAGS_DMA_OFFSET_FIXED                              15:15
#define LWOS46_FLAGS_DMA_OFFSET_FIXED_FALSE                        (0x00000000)
#define LWOS46_FLAGS_DMA_OFFSET_FIXED_TRUE                         (0x00000001)

#define LWOS46_FLAGS_PTE_COALESCE_LEVEL_CAP                        19:16
#define LWOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_DEFAULT                (0x00000000)
#define LWOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_1                      (0x00000001)
#define LWOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_2                      (0x00000002)
#define LWOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_4                      (0x00000003)
#define LWOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_8                      (0x00000004)
#define LWOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_16                     (0x00000005)
#define LWOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_32                     (0x00000006)
#define LWOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_64                     (0x00000007)
#define LWOS46_FLAGS_PTE_COALESCE_LEVEL_CAP_128                    (0x00000008)
#define LWOS46_FLAGS_P2P                                           27:20

#define LWOS46_FLAGS_P2P_ENABLE                                    21:20
#define LWOS46_FLAGS_P2P_ENABLE_NO                                 (0x00000000)
#define LWOS46_FLAGS_P2P_ENABLE_YES                                (0x00000001)
#define LWOS46_FLAGS_P2P_ENABLE_NONE                               LWOS46_FLAGS_P2P_ENABLE_NO
#define LWOS46_FLAGS_P2P_ENABLE_SLI                                LWOS46_FLAGS_P2P_ENABLE_YES
#define LWOS46_FLAGS_P2P_ENABLE_NOSLI                              (0x00000002)
#ifdef LW_VERIF_FEATURES
// For loopbacks only, bug 200203056
// LWOS46_FLAGS_P2P_LOOPBACK_PEER_ID is used to specify peerid
#define LWOS46_FLAGS_P2P_ENABLE_LOOPBACK                           (0x00000003)
#endif
// Subdevice ID. Reserved 3 bits for the possibility of 8-way SLI
#define LWOS46_FLAGS_P2P_SUBDEVICE_ID                              24:22
#define LWOS46_FLAGS_P2P_SUBDEV_ID_SRC                             LWOS46_FLAGS_P2P_SUBDEVICE_ID
#ifdef LW_VERIF_FEATURES
// peerId is available as bits 24:22 when loopback is enabled
#define LWOS46_FLAGS_P2P_LOOPBACK_PEER_ID                          LWOS46_FLAGS_P2P_SUBDEVICE_ID
#endif
#define LWOS46_FLAGS_P2P_SUBDEV_ID_TGT                             27:25
#define LWOS46_FLAGS_TLB_LOCK                                      28:28
#define LWOS46_FLAGS_TLB_LOCK_DISABLE                              (0x00000000)
#define LWOS46_FLAGS_TLB_LOCK_ENABLE                               (0x00000001)
#define LWOS46_FLAGS_DMA_UNICAST_REUSE_ALLOC                       29:29
#define LWOS46_FLAGS_DMA_UNICAST_REUSE_ALLOC_FALSE                 (0x00000000)
#define LWOS46_FLAGS_DMA_UNICAST_REUSE_ALLOC_TRUE                  (0x00000001)
#define LWOS46_FLAGS_DR_SURF                                       30:30
#define LWOS46_FLAGS_DR_SURF_FALSE                                 (0x00000000)
#define LWOS46_FLAGS_DR_SURF_TRUE                                  (0x00000001)
//
// This flag must be used with caution. Improper use can leave stale entries in the TLB,
// and allow access to memory no longer owned by the RM client or cause page faults.
// Also see corresponding flag for LwUnmapMemoryDma.
//
#define LWOS46_FLAGS_DEFER_TLB_ILWALIDATION                        31:31
#define LWOS46_FLAGS_DEFER_TLB_ILWALIDATION_FALSE                  (0x00000000)
#define LWOS46_FLAGS_DEFER_TLB_ILWALIDATION_TRUE                   (0x00000001)

/* parameters */
typedef struct
{
    LwHandle hClient;                // [IN] client handle
    LwHandle hDevice;                // [IN] device handle for mapping
    LwHandle hDma;                   // [IN] dma handle for mapping
    LwHandle hMemory;                // [IN] memory handle for mapping
    LwU64    offset LW_ALIGN_BYTES(8);     // [IN] offset of region
    LwU64    length LW_ALIGN_BYTES(8);     // [IN] limit of region
    LwV32    flags;                  // [IN] flags
    LwU64    dmaOffset LW_ALIGN_BYTES(8);  // [OUT] offset of mapping
                                           // [IN] if FLAGS_DMA_OFFSET_FIXED_TRUE
                                           //      *OR* hDma is NOT a CTXDMA handle
                                           //      (see LWOS46_FLAGS_DMA_OFFSET_FIXED)
    LwV32    status;                 // [OUT] status
} LWOS46_PARAMETERS;


/* function OS47 */
#define LW04_UNMAP_MEMORY_DMA                                      (0x0000002F)

#define LWOS47_FLAGS_DEFER_TLB_ILWALIDATION                        0:0
#define LWOS47_FLAGS_DEFER_TLB_ILWALIDATION_FALSE                  (0x00000000)
#define LWOS47_FLAGS_DEFER_TLB_ILWALIDATION_TRUE                   (0x00000001)

/* parameters */
typedef struct
{
    LwHandle hClient;                // [IN] client handle
    LwHandle hDevice;                // [IN] device handle for mapping
    LwHandle hDma;                   // [IN] dma handle for mapping
    LwHandle hMemory;                // [IN] memory handle for mapping
    LwV32    flags;                  // [IN] flags
    LwU64    dmaOffset LW_ALIGN_BYTES(8);  // [IN] dma offset from LW04_MAP_MEMORY_DMA
    LwV32    status;                 // [OUT] status
} LWOS47_PARAMETERS;


#define LW04_BIND_CONTEXT_DMA                                      (0x00000031)
/* parameters */
typedef struct
{
    LwHandle    hClient;                // [IN] client handle
    LwHandle    hChannel;               // [IN] channel handle for binding
    LwHandle    hCtxDma;                // [IN] ctx dma handle for binding
    LwV32       status;                 // [OUT] status
} LWOS49_PARAMETERS;


/* function OS54 */
#define LW04_CONTROL                                               (0x00000036)

#define LWOS54_FLAGS_NONE                                          (0x00000000)
#define LWOS54_FLAGS_IRQL_RAISED                                   (0x00000001)
#define LWOS54_FLAGS_LOCK_BYPASS                                   (0x00000002)

/* parameters */
typedef struct
{
    LwHandle hClient;
    LwHandle hObject;
    LwV32    cmd;
    LwU32    flags;
    LwP64    params LW_ALIGN_BYTES(8);
    LwU32    paramsSize;
    LwV32    status;
} LWOS54_PARAMETERS;

/* RM Control header
 *
 * Replacement for LWOS54_PARAMETERS where embedded pointers are not allowed.
 * Input layout for user space RM Control calls should be:
 *
 * +--- LWOS63_PARAMETERS ---+--- RM Control parameters ---+
 *
 * LWOS63_PARAMETERS::paramsSize is the size of RM Control parameters
 *
 */
typedef struct
{
    LwHandle hClient;       // [IN]  client handle
    LwHandle hObject;       // [IN]  object handle
    LwV32    cmd;           // [IN]  control command ID
    LwU32    paramsSize;    // [IN]  size in bytes of the RM Control parameters
    LwV32    status;        // [OUT] status
} LWOS63_PARAMETERS;


/* function OS55 */
#define LW04_DUP_OBJECT                                             (0x00000037)

/* parameters */
typedef struct
{
  LwHandle  hClient;                // [IN]  destination client handle
  LwHandle  hParent;                // [IN]  parent of new object
  LwHandle  hObject;                // [INOUT] destination (new) object handle
  LwHandle  hClientSrc;             // [IN]  source client handle
  LwHandle  hObjectSrc;             // [IN]  source (old) object handle
  LwU32     flags;                  // [IN]  flags
  LwU32     status;                 // [OUT] status
} LWOS55_PARAMETERS;

#define LW04_DUP_HANDLE_FLAGS_NONE                                  (0x00000000)
#define LW04_DUP_HANDLE_FLAGS_REJECT_KERNEL_DUP_PRIVILEGE           (0x00000001) // If set, prevents an RM kernel client from duping unconditionally
                                                                                 // NOTE: Do not declare a LW04_DUP_HANDLE_FLAGS_* value of 0x00000008
                                                                                 // until Bug 2859347 is resolved! This is due to conflicting usage
                                                                                 // of RS_RES_DUP_PARAMS_INTERNAL.flags to pass
                                                                                 // LWOS32_ALLOC_INTERNAL_FLAGS_FLA_MEMORY to an object constructor.

/* function OS56 */
#define LW04_UPDATE_DEVICE_MAPPING_INFO                             (0x00000038)

/* parameters */
typedef struct
{
    LwHandle hClient;
    LwHandle hDevice;
    LwHandle hMemory;
    LwP64    pOldCpuAddress LW_ALIGN_BYTES(8);
    LwP64    pNewCpuAddress LW_ALIGN_BYTES(8);
    LwV32    status;
} LWOS56_PARAMETERS;

/* function OS57 */
#define LW04_SHARE                                             (0x0000003E)

/* parameters */
typedef struct
{
  LwHandle        hClient;        // [IN]  owner client handle
  LwHandle        hObject;        // [IN]  resource to share
  RS_SHARE_POLICY sharePolicy;    // [IN]  share policy entry
  LwU32           status;         // [OUT] status
} LWOS57_PARAMETERS;

/* parameters */
typedef struct
{
    LwU32 deviceReference;
    LwU32 head;
    LwU32 state;
    LwU8  forceMonitorState;
    LwU8  bForcePerfBiosLevel;
    LwU8  bIsD3HotTransition;    // [OUT] To tell client if it's a D3Hot transition
    LwU32 fastBootPowerState;
} LWPOWERSTATE_PARAMETERS, *PLWPOWERSTATE_PARAMETERS;

 /***************************************************************************\
|*                          Object Allocation Parameters                     *|
 \***************************************************************************/

// GR engine creation parameters
typedef struct {
    LwU32   version;    // set to 0x2
    LwU32   flags;      // input param from a rm client (no flags are lwrrently defined)
    LwU32   size;       // sizeof(LW_GR_ALLOCATION_PARAMETERS)
    LwU32   caps;       // output param for a rm client - class dependent
} LW_GR_ALLOCATION_PARAMETERS;

//
// LwAlloc parameters for LW03_DEVICE_XX class
//    hClientShare
//      For LW50+ this can be set to virtual address space for this
//      device. On previous chips this field is ignored. There are
//      three possible settings
//          LW01_NULL_OBJECT - Use the default global VA space
//          Handle to current client - Create a new private address space
//          Handle to another client - Attach to other clients address space
//    flags
//          MAP_PTE_GLOBALLY           Deprecated.
//          MINIMIZE_PTETABLE_SIZE     Pass hint to DMA HAL to use partial page
//                                     tables. Depending on allocation pattern
//                                     this may actually use more instance memory.
//          RETRY_PTE_ALLOC_IN_SYS     Fallback to PTEs allocation in sysmem. This
//                                     is now enabled by default.
//          VASPACE_SIZE               Honor vaSpaceSize field.
//
//          MAP_PTE                    Deprecated.
//
//          VASPACE_IS_MIRRORED        This flag will tell RM to create a mirrored
//                                     kernel PDB for the address space associated
//                                     with this device. When this flag is set
//                                     the address space covered by the top PDE
//                                     is restricted and cannot be allocated out of.
//
//
//          VASPACE_BIG_PAGE_SIZE_64k  ***Warning this flag will be deprecated do not use*****
//          VASPACE_BIG_PAGE_SIZE_128k This flag will choose the big page size of the VASPace
//                                     to 64K/128k if the system supports a configurable size.
//                                     If the system does not support a configurable size then
//                                     defaults will be chosen.
//                                     If the user sets both these bits then this API will fail.
//
//          SHARED_MANAGEMENT
//              *** Warning: This will be deprecated - see LW_VASPACE_ALLOCATION_PARAMETERS. ***
//
//
//    hTargetClient/hTargetDevice
//      Deprecated. Can be deleted once client code has removed references.
//
//    vaBase
//        *** Warning: This will be deprecated - see LW_VASPACE_ALLOCATION_PARAMETERS. ***
//
//    vaSpaceSize
//      Set the size of the VA space used for this client if allocating
//      a new private address space. Is expressed as a size such as
//      (1<<32) for a 32b address space. Reducing the size of the address
//      space allows the dma chip specific code to reduce the instance memory
//      used for page tables.
//
//    vaMode
//      The vaspace allocation mode. There are three modes supported:
//      1. SINGLE_VASPACE
//      An old abstraction that provides a single VA space under a
//      device and it's allocated implicityly when an object requires a VA
//      space. Typically, this VA space is also shared across clients.
//
//      2. OPTIONAL_MULTIPLE_VASPACES
//      Global + multiple private va spaces. In this mode, the old abstraction,
//      a single vaspace under a device that is allocated implicitly is still
//      being supported. A private VA space is an entity under a device, which/
//      cannot be shared with other clients, but multiple channels under the
//      same device can still share a private VA space.
//      Private VA spaces (class:90f1,FERMI_VASPACE_A) can be allocated as
//      objects through RM APIs. This mode requires the users to know what they
//      are doing in terms of using VA spaces. Page fault can easily occur if
//      one is not careful with a mixed of an implicit VA space and multiple
//      VA spaces.
//
//      3. MULTIPLE_VASPACES
//      In this mode, all VA spaces have to be allocated explicitly through RM
//      APIs and users have to specify which VA space to use for each object.
//      This case prevents users to use context dma, which is not supported and
//      can be misleading if used. Therefore, it's more a safeguard mode to
//      prevent people making mistakes that are hard to debug.
//
//      DEFAULT MODE: 2. OPTIONAL_MULTIPLE_VASPACES
//
// See LW0080_ALLOC_PARAMETERS for allocation parameter structure.
//

#define LW_DEVICE_ALLOCATION_SZNAME_MAXLEN    128
#define LW_DEVICE_ALLOCATION_FLAGS_NONE                            (0x00000000)
#define LW_DEVICE_ALLOCATION_FLAGS_MAP_PTE_GLOBALLY                (0x00000001)
#define LW_DEVICE_ALLOCATION_FLAGS_MINIMIZE_PTETABLE_SIZE          (0x00000002)
#define LW_DEVICE_ALLOCATION_FLAGS_RETRY_PTE_ALLOC_IN_SYS          (0x00000004)
#define LW_DEVICE_ALLOCATION_FLAGS_VASPACE_SIZE                    (0x00000008)
#define LW_DEVICE_ALLOCATION_FLAGS_MAP_PTE                         (0x00000010)
#define LW_DEVICE_ALLOCATION_FLAGS_VASPACE_IS_TARGET               (0x00000020)
#define LW_DEVICE_ALLOCATION_FLAGS_VASPACE_SHARED_MANAGEMENT       (0x00000100)
#define LW_DEVICE_ALLOCATION_FLAGS_VASPACE_BIG_PAGE_SIZE_64k       (0x00000200)
#define LW_DEVICE_ALLOCATION_FLAGS_VASPACE_BIG_PAGE_SIZE_128k      (0x00000400)
#define LW_DEVICE_ALLOCATION_FLAGS_RESTRICT_RESERVED_VALIMITS      (0x00000800)

/*
 *TODO: Delete this flag once LWCA moves to the ctrl call
 */
#define LW_DEVICE_ALLOCATION_FLAGS_VASPACE_IS_MIRRORED             (0x00000040)

// XXX LW_DEVICE_ALLOCATION_FLAGS_VASPACE_PTABLE_PMA_MANAGED should not
//     should not be exposed to clients. It should be the default RM
//     behavior.
//
//     Until it is made the default, certain clients such as OpenGL
//     might still need PTABLE allocations to go through PMA, so this
//     flag has been temporary exposed.
//
//     See bug 1880192
#define LW_DEVICE_ALLOCATION_FLAGS_VASPACE_PTABLE_PMA_MANAGED      (0x00001000)

//
// Indicates this device is being created by guest and requires a
// HostVgpuDeviceKernel creation in client.
//
#define LW_DEVICE_ALLOCATION_FLAGS_HOST_VGPU_DEVICE                (0x00002000)

//
// Indicates this device is being created for VGPU plugin use.
// Requires a HostVgpuDevice handle to indicate the guest on which
// this plugin operates.
//
#define LW_DEVICE_ALLOCATION_FLAGS_PLUGIN_CONTEXT                  (0x00004000)

#define LW_DEVICE_ALLOCATION_VAMODE_OPTIONAL_MULTIPLE_VASPACES     (0x00000000)
#define LW_DEVICE_ALLOCATION_VAMODE_SINGLE_VASPACE                 (0x00000001)
#define LW_DEVICE_ALLOCATION_VAMODE_MULTIPLE_VASPACES              (0x00000002)

/*
 * LW_CHANNELGPFIFO_ALLOCATION_PARAMETERS.flags values.
 *
 * These flags may apply to all channel types: PIO, DMA, and GPFIFO.
 * They are also designed so that zero is always the correct default.
 *
 *   LWOS04_FLAGS_CHANNEL_TYPE:
 *     This flag specifies the type of channel to allocate.  Legal values
 *     for this flag include:
 *
 *       LWOS04_FLAGS_CHANNEL_TYPE_PHYSICAL:
 *         This flag specifies that a physical channel is to be allocated.
 *
 *       LWOS04_FLAGS_CHANNEL_TYPE_VIRTUAL:
 *         OBSOLETE - NOT SUPPORTED
 *
 *       LWOS04_FLAGS_CHANNEL_TYPE_PHYSICAL_FOR_VIRTUAL:
 *         OBSOLETE - NOT SUPPORTED
 */

/* valid LWOS04_FLAGS_CHANNEL_TYPE values */
#define LWOS04_FLAGS_CHANNEL_TYPE                                  1:0
#define LWOS04_FLAGS_CHANNEL_TYPE_PHYSICAL                         0x00000000
#define LWOS04_FLAGS_CHANNEL_TYPE_VIRTUAL                          0x00000001  // OBSOLETE
#define LWOS04_FLAGS_CHANNEL_TYPE_PHYSICAL_FOR_VIRTUAL             0x00000002  // OBSOLETE

/*
 *    LWOS04_FLAGS_VPR:
 *     This flag specifies if channel is intended for work with
 *     Video Protected Regions (VPR)
 *
 *       LWOS04_FLAGS_VPR_TRUE:
 *         The channel will only write to protected memory regions.
 *
 *       LWOS04_FLAGS_VPR_FALSE:
 *         The channel will never read from protected memory regions.
 */
#define LWOS04_FLAGS_VPR                                           2:2
#define LWOS04_FLAGS_VPR_FALSE                                     0x00000000
#define LWOS04_FLAGS_VPR_TRUE                                      0x00000001

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
/*
 *    LWOS04_FLAGS_CC_SELWRE:
 *     This flag specifies if channel is intended to be used for 
 *     encryption/decryption of data between SYSMEM <-> VIDMEM. Only CE
 *     & SEC2 Channels are capable of handling encrypted content and this
 *     flag will be ignored when CC is disabled or for chips that are not CC
 *     Capable.
 *     Reusing VPR index since VPR & CC are mutually exclusive.
 *
 *       LWOS04_FLAGS_CC_SELWRE_TRUE:
 *         The channel will support CC Encryption/Decryption
 *
 *       LWOS04_FLAGS_CC_SELWRE_FALSE:
 *         The channel will not support CC Encryption/Decryption
 */
#define LWOS04_FLAGS_CC_SELWRE                                     2:2
#define LWOS04_FLAGS_CC_SELWRE_FALSE                               0x00000000
#define LWOS04_FLAGS_CC_SELWRE_TRUE                                0x00000001
#endif 
/*
 *    LWOS04_FLAGS_CHANNEL_SKIP_MAP_REFCOUNTING:
 *     This flag specifies if the channel can skip refcounting of potentially
 *     accessed mappings on job kickoff.  This flag is only meaningful for
 *     kernel drivers which perform refcounting of memory mappings.
 *
 *       LWOS04_FLAGS_CHANNEL_SKIP_MAP_REFCOUNTING_FALSE:
 *         The channel cannot not skip refcounting of memory mappings
 *
 *       LWOS04_FLAGS_CHANNEL_SKIP_MAP_REFCOUNTING_TRUE:
 *         The channel can skip refcounting of memory mappings
 */
#define LWOS04_FLAGS_CHANNEL_SKIP_MAP_REFCOUNTING                  3:3
#define LWOS04_FLAGS_CHANNEL_SKIP_MAP_REFCOUNTING_FALSE            0x00000000
#define LWOS04_FLAGS_CHANNEL_SKIP_MAP_REFCOUNTING_TRUE             0x00000001

/*
 *     LWOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE:
 *       This flag specifies which "runqueue" the allocated channel will be
 *       exelwted on in a TSG.  Channels on different runqueues within a TSG
 *       may be able to feed methods into the engine simultaneously.
 *       Non-default values are only supported on GP10x and later and only for
 *       channels within a TSG.
 */
#define LWOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE                       4:4
#define LWOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_DEFAULT               0x00000000
#define LWOS04_FLAGS_GROUP_CHANNEL_RUNQUEUE_ONE                   0x00000001

/*
 *     LWOS04_FLAGS_PRIVILEGED_CHANNEL:
 *       This flag tells RM whether to give the channel admin privilege. This
 *       flag will only take effect if the client is GSP-vGPU plugin. It is
 *       needed so that guest can update page tables in physical mode and do
 *       scrubbing.
 */
#define LWOS04_FLAGS_PRIVILEGED_CHANNEL                           5:5
#define LWOS04_FLAGS_PRIVILEGED_CHANNEL_FALSE                     0x00000000
#define LWOS04_FLAGS_PRIVILEGED_CHANNEL_TRUE                      0x00000001

/*
 *     LWOS04_FLAGS_DELAY_CHANNEL_SCHEDULING:
 *       This flags tells RM not to schedule a newly created channel within a
 *       channel group immediately even if channel group is lwrrently scheduled.
 *       Channel will not be scheduled until LWA06F_CTRL_GPFIFO_SCHEDULE is
 *       ilwoked. This is used eg. for LWCA which needs to do additional
 *       initialization before starting up a channel.
 *       Default is FALSE.
 */
#define LWOS04_FLAGS_DELAY_CHANNEL_SCHEDULING                     6:6
#define LWOS04_FLAGS_DELAY_CHANNEL_SCHEDULING_FALSE               0x00000000
#define LWOS04_FLAGS_DELAY_CHANNEL_SCHEDULING_TRUE                0x00000001

/*
 *     LWOS04_FLAGS_DENY_PHYSICAL_MODE_CE:
 *       This flag specifies whether or not to deny access to the physical
 *       mode of CopyEngine regardless of whether or not the client handle
 *       is admin. If set to true, this channel allocation will always result
 *       in an unprivileged channel. If set to false, the privilege of the channel
 *       will depend on the privilege level of the client handle.
 *       This is primarily meant for vGPU since all client handles
 *       granted to guests are admin.
 */
#define LWOS04_FLAGS_CHANNEL_DENY_PHYSICAL_MODE_CE                7:7
#define LWOS04_FLAGS_CHANNEL_DENY_PHYSICAL_MODE_CE_FALSE          0x00000000
#define LWOS04_FLAGS_CHANNEL_DENY_PHYSICAL_MODE_CE_TRUE           0x00000001

/*
 *     LWOS04_FLAGS_CHANNEL_USERD_INDEX_VALUE
 *
 *        This flag specifies the channel offset in terms of within a page of
 *        USERD. For example, value 3 means the 4th channel within a USERD page.
 *        Given the USERD size is 512B, we will have 8 channels total, so 3 bits
 *        are reserved.
 *
 *        When _USERD_INDEX_FIXED_TRUE is set but INDEX_PAGE_FIXED_FALSE is set,
 *        it will ask for a new USERD page.
 *
 */
#define LWOS04_FLAGS_CHANNEL_USERD_INDEX_VALUE                    10:8

#define LWOS04_FLAGS_CHANNEL_USERD_INDEX_FIXED                    11:11
#define LWOS04_FLAGS_CHANNEL_USERD_INDEX_FIXED_FALSE              0x00000000
#define LWOS04_FLAGS_CHANNEL_USERD_INDEX_FIXED_TRUE               0x00000001

/*
 *     LWOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_VALUE
 *
 *        This flag specifies the channel offset in terms of USERD page. When
 *        this PAGE_FIXED_TRUE is set, the INDEX_FIXED_FALSE bit should also
 *        be set, otherwise ILWALID_STATE will be returned.
 *
 *        And the field _USERD_INDEX_VALUE will be used to request the specific
 *        offset within a USERD page.
 */

#define LWOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_VALUE               20:12

#define LWOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_FIXED               21:21
#define LWOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_FIXED_FALSE         0x00000000
#define LWOS04_FLAGS_CHANNEL_USERD_INDEX_PAGE_FIXED_TRUE          0x00000001

/*
 *     LWOS04_FLAGS_DENY_AUTH_LEVEL_PRIV
 *       This flag specifies whether or not to deny access to the privileged
 *       host methods TLB_ILWALIDATE and ACCESS_COUNTER_CLR
 */
#define LWOS04_FLAGS_CHANNEL_DENY_AUTH_LEVEL_PRIV                 22:22
#define LWOS04_FLAGS_CHANNEL_DENY_AUTH_LEVEL_PRIV_FALSE           0x00000000
#define LWOS04_FLAGS_CHANNEL_DENY_AUTH_LEVEL_PRIV_TRUE            0x00000001

/*
 *    LWOS04_FLAGS_CHANNEL_SKIP_SCRUBBER
 *
 *       This flag specifies scrubbing should be skipped for any internal
 *       allocations made for this channel from PMA using ctx buf pools.
 *       Only kernel clients are allowed to use this setting.
 */
#define LWOS04_FLAGS_CHANNEL_SKIP_SCRUBBER                        23:23
#define LWOS04_FLAGS_CHANNEL_SKIP_SCRUBBER_FALSE                  0x00000000
#define LWOS04_FLAGS_CHANNEL_SKIP_SCRUBBER_TRUE                   0x00000001

/*
 *    LWOS04_FLAGS_CHANNEL_CLIENT_MAP_FIFO
 *
 *       This flag specifies that the client is expected to map USERD themselves
 *       and RM need not do so.
 */
#define LWOS04_FLAGS_CHANNEL_CLIENT_MAP_FIFO                      24:24
#define LWOS04_FLAGS_CHANNEL_CLIENT_MAP_FIFO_FALSE                0x00000000
#define LWOS04_FLAGS_CHANNEL_CLIENT_MAP_FIFO_TRUE                 0x00000001

/*
 *    LWOS04_FLAGS_SET_EVICT_LAST_CE_PREFETCH_CHANNEL
 */
#define LWOS04_FLAGS_SET_EVICT_LAST_CE_PREFETCH_CHANNEL           25:25
#define LWOS04_FLAGS_SET_EVICT_LAST_CE_PREFETCH_CHANNEL_FALSE     0x00000000
#define LWOS04_FLAGS_SET_EVICT_LAST_CE_PREFETCH_CHANNEL_TRUE      0x00000001

/*
 *    LWOS04_FLAGS_CHANNEL_VGPU_PLUGIN_CONTEXT
 *
 *       This flag specifies whether the channel calling context is from CPU
 *       VGPU plugin.
 */
#define LWOS04_FLAGS_CHANNEL_VGPU_PLUGIN_CONTEXT                  26:26
#define LWOS04_FLAGS_CHANNEL_VGPU_PLUGIN_CONTEXT_FALSE            0x00000000
#define LWOS04_FLAGS_CHANNEL_VGPU_PLUGIN_CONTEXT_TRUE             0x00000001

 /*
  *     LWOS04_FLAGS_CHANNEL_PBDMA_ACQUIRE_TIMEOUT
  *
  *        This flag specifies the channel PBDMA ACQUIRE timeout option.
  *        _FALSE to disable it, _TRUE to enable it.
  *        When this flag is enabled, if a host semaphore acquire does not
  *        complete in about 2 sec, it will time out and trigger a RC error.
  */
#define LWOS04_FLAGS_CHANNEL_PBDMA_ACQUIRE_TIMEOUT                 27:27
#define LWOS04_FLAGS_CHANNEL_PBDMA_ACQUIRE_TIMEOUT_FALSE           0x00000000
#define LWOS04_FLAGS_CHANNEL_PBDMA_ACQUIRE_TIMEOUT_TRUE            0x00000001

/*
 *     LWOS04_FLAGS_GROUP_CHANNEL_THREAD:
 *       This flags specifies the thread id in which an allocated channel
 *       will be exelwted in a TSG. The relationship between the thread id
 *       in A TSG and respective definitions are implementation specific.
 *       Also, not all classes will be supported at thread > 0.
 *       This field cannot be used on non-TSG channels and must be set to
 *       the default value (0) in that case. If thread > 0 on a non-TSG
 *       channel, the allocation will fail
 */
#define LWOS04_FLAGS_GROUP_CHANNEL_THREAD                          29:28
#define LWOS04_FLAGS_GROUP_CHANNEL_THREAD_DEFAULT                  0x00000000
#define LWOS04_FLAGS_GROUP_CHANNEL_THREAD_ONE                      0x00000001
#define LWOS04_FLAGS_GROUP_CHANNEL_THREAD_TWO                      0x00000002

#define LWOS04_FLAGS_MAP_CHANNEL                                   30:30
#define LWOS04_FLAGS_MAP_CHANNEL_FALSE                             0x00000000
#define LWOS04_FLAGS_MAP_CHANNEL_TRUE                              0x00000001

#define LWOS04_FLAGS_SKIP_CTXBUFFER_ALLOC                          31:31
#define LWOS04_FLAGS_SKIP_CTXBUFFER_ALLOC_FALSE                    0x00000000
#define LWOS04_FLAGS_SKIP_CTXBUFFER_ALLOC_TRUE                     0x00000001

#ifdef LW_VERIF_FEATURES

/*
 * LW_CHANNELGPFIFO_ALLOCATION_PARAMETERS.verifFlags values.
 *
 * These flags may apply to all channel types: PIO, DMA, and GPFIFO.
 * They are also designed so that zero is always the correct default.
 * On G82+, INST_MEM applies to both inst and cache1.
 */
#define LWOS04_VERIF_FLAGS_INST_MEM_LOC                              1:0
#define LWOS04_VERIF_FLAGS_INST_MEM_LOC_DEFAULT                      0x00000000
#define LWOS04_VERIF_FLAGS_INST_MEM_LOC_VID                          0x00000001
#define LWOS04_VERIF_FLAGS_INST_MEM_LOC_COH                          0x00000002
#define LWOS04_VERIF_FLAGS_INST_MEM_LOC_NCOH                         0x00000003

#define LWOS04_VERIF_FLAGS_HASH_TABLE_SIZE                           4:2
#define LWOS04_VERIF_FLAGS_HASH_TABLE_SIZE_DEFAULT                   0x00000000
#define LWOS04_VERIF_FLAGS_HASH_TABLE_SIZE_4KB                       0x00000001
#define LWOS04_VERIF_FLAGS_HASH_TABLE_SIZE_8KB                       0x00000002
#define LWOS04_VERIF_FLAGS_HASH_TABLE_SIZE_16KB                      0x00000003
#define LWOS04_VERIF_FLAGS_HASH_TABLE_SIZE_32KB                      0x00000004
#define LWOS04_VERIF_FLAGS_HASH_TABLE_SIZE_64                        0x00000001
#define LWOS04_VERIF_FLAGS_HASH_TABLE_SIZE_128                       0x00000002
#define LWOS04_VERIF_FLAGS_HASH_TABLE_SIZE_256                       0x00000003
#define LWOS04_VERIF_FLAGS_HASH_TABLE_SIZE_512                       0x00000004

#define LWOS04_VERIF_FLAGS_TIMESLICE_SELECT_OVERRIDE                 5:5
#define LWOS04_VERIF_FLAGS_TIMESLICE_SELECT_OVERRIDE_DISABLED        0x00000000
#define LWOS04_VERIF_FLAGS_TIMESLICE_SELECT_OVERRIDE_ENABLED         0x00000001

#define LWOS04_VERIF_FLAGS_TIMESLICE_TIMESCALE_OVERRIDE              6:6
#define LWOS04_VERIF_FLAGS_TIMESLICE_TIMESCALE_OVERRIDE_DISABLED     0x00000000
#define LWOS04_VERIF_FLAGS_TIMESLICE_TIMESCALE_OVERRIDE_ENABLED      0x00000001

#define LWOS04_VERIF_FLAGS_TIMESLICE_TIMEOUT                         7:7
#define LWOS04_VERIF_FLAGS_TIMESLICE_TIMEOUT_ENABLED                 0x00000000
#define LWOS04_VERIF_FLAGS_TIMESLICE_TIMEOUT_DISABLED                0x00000001

#define LWOS04_VERIF_FLAGS_TIMESLICE_SELECT                          24:8
#define LWOS04_VERIF_FLAGS_TIMESLICE_TIMESCALE                       28:25

/* This only applies to G82+. */
#define LWOS04_VERIF_FLAGS_RAMFC_MEM_LOC                             30:29
#define LWOS04_VERIF_FLAGS_RAMFC_MEM_LOC_DEFAULT                     0x00000000
#define LWOS04_VERIF_FLAGS_RAMFC_MEM_LOC_VID                         0x00000001
#define LWOS04_VERIF_FLAGS_RAMFC_MEM_LOC_COH                         0x00000002
#define LWOS04_VERIF_FLAGS_RAMFC_MEM_LOC_NCOH                        0x00000003

/*
 * Only supported on Fermi+.  Controls how RM allocates a channel's
 * hardware ID.
 *
 *   DEFAULT:  Use RM's default selection algorithm.
 *   GROWUP:   RM will allocate the lowest available ID.
 *   GROWDOWN: RM will allocate the highest available ID.
 *   PROVIDED: The ID to use is provided in the ALLOC_ID field.  The allocation
 *             will fail if this ID is already in use.
 *
 */
#define LWOS04_VERIF_FLAGS2_ALLOC_ID_MODE                            1:0
#define LWOS04_VERIF_FLAGS2_ALLOC_ID_MODE_DEFAULT                    0x00000000
#define LWOS04_VERIF_FLAGS2_ALLOC_ID_MODE_GROWUP                     0x00000001
#define LWOS04_VERIF_FLAGS2_ALLOC_ID_MODE_GROWDOWN                   0x00000002
#define LWOS04_VERIF_FLAGS2_ALLOC_ID_MODE_PROVIDED                   0x00000003

#define LWOS04_VERIF_FLAGS2_ALLOC_ID                                 15:4

#endif

typedef struct
{
    LwU64           base LW_ALIGN_BYTES(8);
    LwU64           size LW_ALIGN_BYTES(8);
    LwU32           addressSpace;
    LwU32           cacheAttrib;
} LW_MEMORY_DESC_PARAMS;

typedef struct
{
    LwHandle    hObjectError;                                        // error context DMA
    LwHandle    hObjectBuffer;                                       // no longer used
    LwU64       gpFifoOffset LW_ALIGN_BYTES(8);                      // offset to beginning of GP FIFO
    LwU32       gpFifoEntries;                                       // number of GP FIFO entries
    LwU32       flags;
    LwHandle    hContextShare;                                       // context share handle
    LwHandle    hVASpace;                                            // VASpace for the channel
    LwHandle    hUserdMemory[LWOS_MAX_SUBDEVICES];                   // handle to UserD memory object for channel, ignored if hUserdMemory[0]=0
    LwU64       userdOffset[LWOS_MAX_SUBDEVICES] LW_ALIGN_BYTES(8);  // offset to beginning of UserD within hUserdMemory[x]
    LwU32       engineType;                                          // engine type(LW2080_ENGINE_TYPE_*) with which this channel is associated
    LwU32       cid;                                                 // Channel identifier that is unique for the duration of a RM session
    LwU32       subDeviceId;                                         // One-hot encoded bitmask to match SET_SUBDEVICE_MASK methods
    LwHandle    hObjectEccError;                                     // ECC error context DMA
#ifdef LW_VERIF_FEATURES
/* Keep this at the end of the data structure, since
 * LW_VERIF_FEATURES is not consistently defined
 * for resman clients
 */
    LwU32       verifFlags;
    LwU32       verifFlags2;
#endif
    LW_MEMORY_DESC_PARAMS instanceMem;
    LW_MEMORY_DESC_PARAMS userdMem;
    LW_MEMORY_DESC_PARAMS ramfcMem;
    LW_MEMORY_DESC_PARAMS mthdbufMem;

    LwHandle              hPhysChannelGroup;    // reserved
    LwU32                 internalFlags;        // reserved
    LW_MEMORY_DESC_PARAMS errorNotifierMem;     // reserved
    LW_MEMORY_DESC_PARAMS eccErrorNotifierMem;  // reserved
    LwU32                 ProcessID;            // reserved
    LwU32                 SubProcessID;         // reserved
} LW_CHANNELGPFIFO_ALLOCATION_PARAMETERS;

#define LW_CHANNELGPFIFO_NOTIFICATION_TYPE_ERROR                0x00000000
#define LW_CHANNELGPFIFO_NOTIFICATION_TYPE_WORK_SUBMIT_TOKEN    0x00000001
#define LW_CHANNELGPFIFO_NOTIFICATION_TYPE__SIZE_1              2
#define LW_CHANNELGPFIFO_NOTIFICATION_STATUS_VALUE              14:0
#define LW_CHANNELGPFIFO_NOTIFICATION_STATUS_IN_PROGRESS        15:15
#define LW_CHANNELGPFIFO_NOTIFICATION_STATUS_IN_PROGRESS_TRUE   0x1
#define LW_CHANNELGPFIFO_NOTIFICATION_STATUS_IN_PROGRESS_FALSE  0x0

typedef struct
{
    LW_CHANNELGPFIFO_ALLOCATION_PARAMETERS gpfifoAllocationParams;
    LwHandle hKernelChannel;
} LW_PHYSICALCHANNEL_ALLOC_PARAMS;

typedef struct
{
    LwHandle hRunlistBase;                   // Handle to physmem runlist base
    LwU32    engineID;                       // Engine associated with the runlist
} LW_CHANNELRUNLIST_ALLOCATION_PARAMETERS;

typedef struct
{
    LwV32    channelInstance;            // One of the n channel instances of a given channel type.
                                         // Note that core channel has only one instance
                                         // while all others have two (one per head).
    LwHandle hObjectBuffer;              // ctx dma handle for DMA push buffer
    LwHandle hObjectNotify;              // ctx dma handle for an area (of type LwNotification defined in sdk/lwpu/inc/lwtypes.h) where RM can write errors/notifications
    LwU32    offset;                     // Initial offset for put/get, usually zero.
    LwP64    pControl LW_ALIGN_BYTES(8); // pControl gives virt addr of UDISP GET/PUT regs

    LwU32    flags;
#ifdef LW_VERIF_FEATURES
#define LW50VAIO_CHANNELDMA_ALLOCATION_FLAGS_ALLOW_MULTIPLE_PB_FOR_CLIENT      0:0
#define LW50VAIO_CHANNELDMA_ALLOCATION_FLAGS_ALLOW_MULTIPLE_PB_FOR_CLIENT_NO   0x00000000
#define LW50VAIO_CHANNELDMA_ALLOCATION_FLAGS_ALLOW_MULTIPLE_PB_FOR_CLIENT_YES  0x00000001
#endif
#define LW50VAIO_CHANNELDMA_ALLOCATION_FLAGS_CONNECT_PB_AT_GRAB                1:1
#define LW50VAIO_CHANNELDMA_ALLOCATION_FLAGS_CONNECT_PB_AT_GRAB_YES            0x00000000
#define LW50VAIO_CHANNELDMA_ALLOCATION_FLAGS_CONNECT_PB_AT_GRAB_NO             0x00000001

} LW50VAIO_CHANNELDMA_ALLOCATION_PARAMETERS;

typedef struct
{
    LwV32    channelInstance;            // One of the n channel instances of a given channel type.
                                         // All PIO channels have two instances (one per head).
    LwHandle hObjectNotify;              // ctx dma handle for an area (of type LwNotification defined in sdk/lwpu/inc/lwtypes.h) where RM can write errors.
    LwP64    pControl LW_ALIGN_BYTES(8); // pControl gives virt addr of control region for PIO channel
} LW50VAIO_CHANNELPIO_ALLOCATION_PARAMETERS;

// Used for allocating a channel group
typedef struct
{
    LwHandle hObjectError;               // Error notifier for TSG
    LwHandle hObjectEccError;            // ECC Error notifier for TSG
    LwHandle hVASpace;                   // VA space handle for TSG
    LwU32    engineType;                 // Engine to which all channels in this TSG are associated with
    LwBool   bIsCallingContextVgpuPlugin;
} LW_CHANNEL_GROUP_ALLOCATION_PARAMETERS;

/*
* @params:
* @engineId         : Engine to which the software runlist be associated with.
* @maxTSGs          : Maximum number of TSG entries that will be submitted in this software runlist
*                     The size of the runlist buffer will be determined by
*                     2 *                  // double buffer
*                     maxTSGs           *  // determined by KMD
*                     maxChannelPerTSG  *  // Determined by RM
*                     sizeof(RunlistEntry) // Determined by HW format
* @qosIntrEnableMask: QOS Interrupt bitmask that needs to be enabled for the SW runlist defined below.
*/
typedef struct
{
    LwU32    engineId;          //(IN)
    LwU32    maxTSGs;           //(IN)  // Size of the RM could return error if the request cannot be accommodated.
    LwU32    qosIntrEnableMask; //(IN)  // Bitmask for QOS interrupts that needs to be enabled
} LW_SWRUNLIST_ALLOCATION_PARAMS;

#define LW_SWRUNLIST_QOS_INTR_NONE                                   0x00000000
#define LW_SWRUNLIST_QOS_INTR_RUNLIST_AND_ENG_IDLE_ENABLE            LWBIT32(0)
#define LW_SWRUNLIST_QOS_INTR_RUNLIST_IDLE_ENABLE                    LWBIT32(1)
#define LW_SWRUNLIST_QOS_INTR_RUNLIST_ACQUIRE_ENABLE                 LWBIT32(2)
#define LW_SWRUNLIST_QOS_INTR_RUNLIST_ACQUIRE_AND_ENG_IDLE_ENABLE    LWBIT32(3)

typedef struct
{
    LwU32 size;
    LwU32 caps;
} LW_ME_ALLOCATION_PARAMETERS;

typedef struct
{
    LwU32 size;
    LwU32 prohibitMultipleInstances;
    LwU32 engineInstance;               // Select LWDEC0 or LWDEC1 or LWDEC2
} LW_BSP_ALLOCATION_PARAMETERS;

//
// These are referenced by mdiag mods tests, but do not appear to be used during
// in the RM any longer
//
#define  LW_VP_ALLOCATION_FLAGS_STANDARD_UCODE                               (0x00000000)
#define  LW_VP_ALLOCATION_FLAGS_STATIC_UCODE                                 (0x00000001)
#define  LW_VP_ALLOCATION_FLAGS_DYNAMIC_UCODE                                (0x00000002)

//
// LW_VP_ALLOCATION_PARAMETERS.flags
//
// LW_VP_ALLOCATION_FLAGS_AVP_CLIENT are used by CheetAh to specify if
// the current allocation with be used by Video or Audio
//
#define  LW_VP_ALLOCATION_FLAGS_AVP_CLIENT_VIDEO            (0x00000000)
#define  LW_VP_ALLOCATION_FLAGS_AVP_CLIENT_AUDIO            (0x00000001)

typedef struct
{
    LwU32       size;
    LwU32       caps;
    LwU32       flags;
    LwU32       altUcode;
    LwP64       rawUcode            LW_ALIGN_BYTES(8);
    LwU32       rawUcodeSize;
    LwU32       numSubClasses;
    LwU32       numSubSets;
    LwP64       subClasses          LW_ALIGN_BYTES(8);
    LwU32       prohibitMultipleInstances;
    LwP64       pControl            LW_ALIGN_BYTES(8);  // Used by CheetAh to return a mapping to LwE276Control
    LwHandle    hMemoryCmdBuffer    LW_ALIGN_BYTES(8);  // Used by CheetAh to specify cmd buffer
    LwU64       offset              LW_ALIGN_BYTES(8);  // Used by CheetAh to specify an offset into the cmd buffer

} LW_VP_ALLOCATION_PARAMETERS;

typedef struct
{
    LwU32 size;
    LwU32 prohibitMultipleInstances;
} LW_PPP_ALLOCATION_PARAMETERS;

typedef struct
{
    LwU32 size;
    LwU32 prohibitMultipleInstances;  // Prohibit multiple allocations of MSENC?
    LwU32 engineInstance;             // Select MSENC/LWENC0 or LWENC1 or LWENC2
} LW_MSENC_ALLOCATION_PARAMETERS;

typedef struct
{
    LwU32 size;
    LwU32 prohibitMultipleInstances;  // Prohibit multiple allocations of SEC2?
} LW_SEC2_ALLOCATION_PARAMETERS;

typedef struct
{
    LwU32 size;
    LwU32 prohibitMultipleInstances;  // Prohibit multiple allocations of LWJPG?
    LwU32 engineInstance;
} LW_LWJPG_ALLOCATION_PARAMETERS;

typedef struct
{
    LwU32 size;
    LwU32 prohibitMultipleInstances;  // Prohibit multiple allocations of OFA?
} LW_OFA_ALLOCATION_PARAMETERS;

#define LW04_BIND_ARBITRARY_CONTEXT_DMA                                      (0x00000039)

/* parameters */

#define LW04_GET_MEMORY_INFO                                        (0x0000003A)

typedef struct
{
    LwHandle    hClient;                // [IN] client handle
    LwHandle    hDevice;                // [IN] device handle for mapping
    LwHandle    hMemory;                // [IN] memory handle for mapping
    LwU64       offset   LW_ALIGN_BYTES(8);  // [IN] offset of region
    LwU64       physAddr LW_ALIGN_BYTES(8);  // [OUT] Physical Addr
    LwV32       status;                 // [OUT] status
} LWOS58_PARAMETERS;

/* function OS59 */
#define LW04_MAP_MEMORY_DMA_OFFSET                                    (0x0000003B)

/* parameters */
typedef struct
{
    LwHandle    hClient;                // [IN] client handle
    LwHandle    hDevice;                // [IN] device handle for mapping
    LwHandle    hDma;                   // [IN] dma handle for mapping
    LwU32       dmaFirstPage;           // [IN] numPages
    LwU32       numPages;               // [IN] numPages
    LwV32       flags;                  // [IN] flags
    LwU64       offset LW_ALIGN_BYTES(8);  // [IN] Dma Offset
    LwHandle    hDmaPteArray;           // ctx dma for pte's
    LwV32       status;                 // [OUT] status
} LWOS59_PARAMETERS;

/* function OS60 */
#define LW04_UNMAP_MEMORY_DMA_OFFSET                                      (0x0000003C)
/* parameters */
typedef struct
{
    LwHandle    hClient;                // [IN] client handle
    LwHandle    hDevice;                // [IN] device handle for mapping
    LwHandle    hDma;                   // [IN] dma handle for mapping
    LwU32       numPages;               // [IN] numPages
    LwU64       dmaOffset LW_ALIGN_BYTES(8);  // [IN] dmaOffset
    LwV32       status;                 // [OUT] status
} LWOS60_PARAMETERS;


#define LW04_ADD_VBLANK_CALLBACK                          (0x0000003D)

#include "class/cl9010.h" // for OSVBLANKCALLBACKPROC

/* parameters */
/* NOTE: the "void* pParm's" below are ok (but unfortunate) since this interface
   can only be used by other kernel drivers which must share the same ptr-size */
typedef struct
{
    LwHandle             hClient;     // [IN] client handle
    LwHandle             hDevice;     // [IN] device handle for mapping
    LwHandle             hVblank;     // [IN] Vblank handle for control
    OSVBLANKCALLBACKPROC pProc;       // Routine to call at vblank time

    LwV32                LogicalHead; // Logical Head
    void                *pParm1;
    void                *pParm2;
    LwU32                bAdd;        // Add or Delete
    LwV32                status;      // [OUT] status
} LWOS61_PARAMETERS;

/**
 * @brief LwAlloc parameters for VASPACE classes
 *
 * Used to create a new private virtual address space.
 *
 * index
 *       CheetAh: With TEGRA_VASPACE_A, index specifies the IOMMU
 *       virtual address space to be created. Based on the
 *       index, RM/LWMEM will decide the HW ASID to be used with
 *       this VA Space. "index" takes values from the
 *       LWMEM_CLIENT_* defines in
 *       "drivers/common/inc/cheetah/memory/ioctl.h".
 *
 *       Big GPU: With FERMI_VASPACE_A, see LW_VASPACE_ALLOCATION_INDEX_GPU_*.
 *
 * flags
 *       MINIMIZE_PTETABLE_SIZE Pass hint to DMA HAL to use partial page tables.
 *                              Depending on allocation pattern this may actually
 *                              use more instance memory.
 *
 *       RETRY_PTE_ALLOC_IN_SYS Fallback to PTEs allocation in sysmem. This is now
 *       enabled by default.
 *
 *       SHARED_MANAGEMENT
 *          Indicates management of the VA space is shared with another
 *          component (e.g. driver layer, OS, etc.).
 *
 *          The initial VA range from vaBase (inclusive) through vaSize (exclusive)
 *          is managed by RM. The range must be aligned to a top-level PDE's VA
 *          coverage since backing page table levels for this range are managed by RM.
 *          All normal RM virtual memory management APIs work within this range.
 *
 *          An external component can manage the remaining VA ranges,
 *          from 0 (inclusive) to vaBase (exclusive) and from vaSize (inclusive) up to the
 *          maximum VA limit supported by HW.
 *          Management of these ranges includes VA sub-allocation and the
 *          backing lower page table levels.
 *
 *          The top-level page directory is special since it is a shared resource.
 *          Management of the page directory is as follows:
 *          1. Initially RM allocates a page directory for RM-managed PDEs.
 *          2. The external component may create a full page directory and commit it
 *             with LW0080_CTRL_CMD_DMA_SET_PAGE_DIRECTORY.
 *             This will copy the RM-managed PDEs from the RM-managed page directory
 *             into the external page directory and commit channels to the external page directory.
 *             After this point RM will update the external page directory directly for
 *             operations that modify RM-managed PDEs.
 *          3. The external component may use LW0080_CTRL_CMD_DMA_SET_PAGE_DIRECTORY repeatedly
 *             if it needs to update the page directory again (e.g. to resize or migrate).
 *             This will copy the RM-managed PDEs from the old external page directory
 *             into the new external page directory and commit channels to the new page directory.
 *          4. The external component may restore management of the page directory back to
 *             RM with LW0080_CTRL_CMD_DMA_UNSET_PAGE_DIRECTORY.
 *             This will copy the RM-managed PDEs from the external page directory
 *             into the RM-managed page directory and commit channels to the RM-managed page directory.
 *             After this point RM will update the RM-managed page directory for
 *             operations that modify RM-managed PDEs.
 *          Note that operations (2) and (4) are symmetric - the RM perspective of management is identical
 *          before and after a sequence of SET => ... => UNSET.
 *
 *       IS_MIRRORED      <to be deprecated once LWCA uses EXTERNALLY_MANAGED>
 *                        This flag will tell RM to create a mirrored
 *                        kernel PDB for the address space associated
 *                        with this device. When this flag is set
 *                        the address space covered by the top PDE
 *                        is restricted and cannot be allocated out of.
 *       ENABLE_PAGE_FAULTING
 *                        Enable page faulting if the architecture supports it.
 *                        As of now page faulting is only supported for compute on pascal+.
 *       IS_EXTERNALLY_OWNED
 *                        This vaspace that has been allocated will be managed by
 *                        an external driver. RM will not own the pagetables for this vaspace.
 *
 *       ENABLE_LWLINK_ATS
 *                        Enables VA translation for this address space using LWLINK ATS.
 *                        Note, the GMMU page tables still exist and take priority over LWLINK ATS.
 *                        VA space object creation will fail if:
 *                        - hardware support is not available (LW_ERR_NOT_SUPPORTED)
 *                        - incompatible options IS_MIRRORED or IS_EXTERNALLY_OWNED are set (LW_ERR_ILWALID_ARGUMENT)
 *       IS_FLA
 *                        Sets FLA flag for this VASPACE
 *
 *       ALLOW_ZERO_ADDRESS
 *                        Allows VASPACE Range to start from zero
 *       SKIP_SCRUB_MEMPOOL
 *                        Skip scrubbing in MemPool
 *
 * vaBase [in, out]
 *       On input, the lowest usable base address of the VA space.
 *       If 0, RM will pick a default value - 0 is always reserved to respresent NULL pointers.
 *       The value must be aligned to the largest page size of the VA space.
 *       Larger values aid in debug since offsets added to NULL pointers will still fault.
 *
 *       On output, the actual usable base address is returned.
 *
 * vaSize [in,out]
 *       On input, requested size of the virtual address space in bytes.
 *       Requesting a smaller size reduces the memory required for the initial
 *       page directory, but the VAS may be resized later (LW0080_CTRL_DMA_SET_VA_SPACE_SIZE).
 *       If 0, the default VA space size will be used.
 *
 *       On output, the actual size of the VAS in bytes.
 *       NOTE: This corresponds to the VA_LIMIT + 1, so the usable size is (vaSize - vaBase).
 *
 * bigPageSIze
 *       Set the size of the big page in this address space object. Current HW supports
 *       either 64k or 128k as the size of the big page. HW that support multiple big
 *       page size per address space will use this size. Hw that do not support this feature
 *       will override to the default big page size that is supported by the system.
 *       If the big page size value is set to ZERO then we will pick the default page size
 *       of the system.
 **/
typedef struct
{
    LwU32   index;
    LwV32   flags;
    LwU64   vaSize LW_ALIGN_BYTES(8);
    LwU64   vaStartInternal LW_ALIGN_BYTES(8);
    LwU64   vaLimitInternal LW_ALIGN_BYTES(8);
    LwU32   bigPageSize;
    LwU64   vaBase LW_ALIGN_BYTES(8);
} LW_VASPACE_ALLOCATION_PARAMETERS;

#define LW_VASPACE_ALLOCATION_FLAGS_NONE                            (0x00000000)
#define LW_VASPACE_ALLOCATION_FLAGS_MINIMIZE_PTETABLE_SIZE                BIT(0)
#define LW_VASPACE_ALLOCATION_FLAGS_RETRY_PTE_ALLOC_IN_SYS                BIT(1)
#define LW_VASPACE_ALLOCATION_FLAGS_SHARED_MANAGEMENT                     BIT(2)
#define LW_VASPACE_ALLOCATION_FLAGS_IS_EXTERNALLY_OWNED                   BIT(3)
#define LW_VASPACE_ALLOCATION_FLAGS_ENABLE_LWLINK_ATS                     BIT(4)
#define LW_VASPACE_ALLOCATION_FLAGS_IS_MIRRORED                           BIT(5)
#define LW_VASPACE_ALLOCATION_FLAGS_ENABLE_PAGE_FAULTING                  BIT(6)
#define LW_VASPACE_ALLOCATION_FLAGS_VA_INTERNAL_LIMIT                     BIT(7)
#define LW_VASPACE_ALLOCATION_FLAGS_ALLOW_ZERO_ADDRESS                    BIT(8)
#define LW_VASPACE_ALLOCATION_FLAGS_IS_FLA                                BIT(9)
#define LW_VASPACE_ALLOCATION_FLAGS_SKIP_SCRUB_MEMPOOL                    BIT(10)
#define LW_VASPACE_ALLOCATION_FLAGS_OPTIMIZE_PTETABLE_MEMPOOL_USAGE       BIT(11)

#define LW_VASPACE_ALLOCATION_INDEX_GPU_NEW                                 0x00 //<! Create new VASpace, by default
#define LW_VASPACE_ALLOCATION_INDEX_GPU_HOST                                0x01 //<! Acquire reference to BAR1 VAS.
#define LW_VASPACE_ALLOCATION_INDEX_GPU_GLOBAL                              0x02 //<! Acquire reference to global VAS.
#define LW_VASPACE_ALLOCATION_INDEX_GPU_DEVICE                              0x03 //<! Acquire reference to device vaspace
#define LW_VASPACE_ALLOCATION_INDEX_GPU_FLA                                 0x04 //<! Acquire reference to FLA VAS.
#define LW_VASPACE_ALLOCATION_INDEX_GPU_MAX                                 0x05 //<! Increment this on adding index entries


#define LW_VASPACE_BIG_PAGE_SIZE_64K                                (64 * 1024)
#define LW_VASPACE_BIG_PAGE_SIZE_128K                               (128 * 1024)

/**
 * @brief LwAlloc parameters for FERMI_CONTEXT_SHARE_A class
 *
 * Used to create a new context share object for use by a TSG channel.
 * Context share is now used to represent a subcontext within a TSG.
 * Refer subcontexts-rm-design.docx for more details.
 *
 * hVASpace
 *          Handle of VA Space object associated with the context share.
 *          All channels using the same using the context share the same va space.
 *
 * flags
 *          Options for the context share allocation.
 *
 *          LW_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT
 *              Used to specify the subcontext slot
 *              SYNC
 *                  Use synchronous graphics & compute subcontext
 *                  In VOLTA+ chips, this represent VEID 0
 *                  In pre-VOLTA chips, this represent SCG type 0
 *              ASYNC
 *                  Use asynchronous compute subcontext
 *                  In VOLTA+ chips, this represent a VEID greater than 0
 *                  In pre-VOLTA chips, this represent SCG type 1
 *              SPECIFIED
 *                  Force the VEID specified in the subctxId parameter.
 *                  This flag is intended for verif. i.e testing VEID reuse etc.
 *
 * subctxId
 *          As input, it is used to specify the subcontext ID, when the _SPECIFIED flag is set.
 *          As output, it is used to return the subcontext ID allocated by RM.
 *          This field is intended for verif.
 **/

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
/* Refer https://p4viewer.lwpu.com/get///sw/docs/resman/chips/Volta/Subcontexts/subcontexts-rm-design.docx for more details. */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

typedef struct
{
    LwHandle hVASpace;
    LwU32    flags;
    LwU32    subctxId;
} LW_CTXSHARE_ALLOCATION_PARAMETERS;

#define LW_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT                      1:0
#define LW_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_SYNC                 (0x00000000)
#define LW_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_ASYNC                (0x00000001)
#define LW_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_SPECIFIED            (0x00000002)

/**
 * @brief RmTimeoutControl parameters
 *
 * Used to set various timeout-related features in RM.
 *
 * cmd
 *   The timeout-related command to issue to RM.
 *
 * value
 *   Used by command, such as the timeout to be set, in milliseconds.
 **/

typedef struct
{
    LwU32 cmd;
    LwU32 timeoutInMs;
    LwU32 deviceInstance;
} LW_TIMEOUT_CONTROL_PARAMETERS;

#define LW_TIMEOUT_CONTROL_CMD_SET_DEVICE_TIMEOUT                   (0x00000002)
#define LW_TIMEOUT_CONTROL_CMD_RESET_DEVICE_TIMEOUT                 (0x00000003)

// LW_TIMEOUT_CONTROL_CMD_SET_DEVICE_TIMEOUT sets a maximum timeout value for
// any RM call on a specific device on any thread. It uses 'timeoutInMs'
// as the target timeout and 'deviceInstance' as the target device.

// LW_TIMEOUT_CONTROL_CMD_RESET_DEVICE_TIMEOUT resets the device timeout to its
// default value. It uses 'deviceInstance' as the target device.

/**
 * @brief GspTestGetRpcMessageData parameters
 *
 * This API is used by the user-mode GSP firmware RM to get RPC message data
 * from the kernel-mode GSP client RM.
 *
 * This API is only supported in the GSP testbed environment.
 *
 *  blockNum
 *    Specifies block number of message data to return.  A value of 0
 *    indicates that the (default) message header and body should be returned
 *    in the buffer.  If additional RPC-specific data is required it can
 *    be read by continually incrementing the block number and reading the
 *    next block in sequence.
 *  msgBufferSize
 *    Size (in bytes) of buffer pointed to by pMsgBuffer.
 *  pMsgBuffer
 *    Address of user-buffer into  which RPC message data will be copied.
 *  status
 *    Returns status of call.
 **/
typedef struct
{
    LwU32 blockNum;                      // [IN] block # of data to get
    LwU32 bufferSize;                    // [IN] size of pBuffer
    LwP64 pBuffer LW_ALIGN_BYTES(8);     // [OUT] buffer returning data
    LwV32 status;                        // [OUT] status of call
} LW_GSP_TEST_GET_MSG_BLOCK_PARAMETERS;

/**
 * @brief GspTestSendRpcMessageResponse parameters
 *
 * This API is used to by the user-mode GSP firmware RM to send an RPC message
 * response to the kernel-mode GSP client RM.
 *
 * This API is only supported in the GSP testbed environment.
 *
 *  bufferSize
 *    Size (in bytes) of buffer pointed to by pBuffer.
 *  pBuffer
 *    Address of user-buffer from which RPC response data will be copied.
 *  status
 *    Returns status of call.
 **/
typedef struct
{
    LwU32 bufferSize;                   // [IN] size of response data buffer
    LwP64 pBuffer LW_ALIGN_BYTES(8);    // [IN] response data buffer
    LwV32 status;                       // [OUT] status of call
} LW_GSP_TEST_SEND_MSG_RESPONSE_PARAMETERS;

/**
 * @brief GspTestSendEventNotification parameters
 *
 * This API is used by the user-mode GSP firmware RM to send an event
 * notification to the kernel-mode GSP client RM.
 *
 * This API is only supported in the GSP testbed environment.
 *
 *  hParentClient
 *    Specifies handle of client that owns object associated with event.
 *  hSrcResource
 *    Specifies handle of object associated with event.
 *  hClass
 *    Specifies class number (type) of event.
 *  notifyIndex
 *    Specifies notifier index associated with event.
 *  status
 *    Returns status of call.
 **/
typedef struct
{
    LwHandle hParentClient;             // [IN] handle of client
    LwHandle hSrcResource;              // [IN] handle of object
    LwU32 hClass;                       // [IN] class number of event
    LwU32 notifyIndex;                  // [IN] notifier index
    LwV32 status;                       // [OUT] status of call
} LW_GSP_TEST_SEND_EVENT_NOTIFICATION_PARAMETERS;

/*
 * LW_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_COH
 * LW_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_DEFAULT
 *           Location is Coherent System memory (also the default option)
 * LW_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_NCOH
 *           Location is Non-Coherent System memory
 * LW_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_VID
 *           Location is FB
 *
 * Lwrrently only used by MODS for the V1 VAB interface. To be deleted.
 */
typedef enum
{
    LW_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_DEFAULT = 0,
    LW_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_COH,
    LW_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_NCOH,
    LW_VIDMEM_ACCESS_BIT_BUFFER_ADDR_SPACE_VID
} LW_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS_ADDR_SPACE;

/**
 * @brief Multiclient vidmem access bit allocation params
 */
typedef struct
{
    /* [OUT] Dirty/Access tracking */
    LwBool bDirtyTracking;
    /* [OUT] Current tracking granularity */
    LwU32 granularity;
    /* [OUT] 512B Access bit mask with 1s set on
       bits that are reserved for this client */
    LW_DECLARE_ALIGNED(LwU64 accessBitMask[64], 8);
    /* Number of entries of vidmem access buffer. Used by VAB v1 - to be deleted */
    LwU32 noOfEntries;
    /* Address space of the vidmem access bit buffer. Used by VAB v1 - to be deleted */
    LW_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS_ADDR_SPACE addrSpace;
} LW_VIDMEM_ACCESS_BIT_ALLOCATION_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
/**
 * @brief HopperUsermodeAParams
 * This set of optionalparameters is passed in on allocation of
 * HOPPER_USERMODE_A object to specify whether a BAR1/GMMU
 * privileged/non-privileged mapping is needed.
 */

typedef struct
{
    /**
     * [IN] Whether to allocate GMMU/BAR1 mapping or BAR0 mapping.
     * This flag is ignored on self-hosted chips.
     */
    LwBool bBar1Mapping;
    /* [IN] Whether to allocate the PRIV page or regular VF page */
    LwBool bPriv;
} LW_HOPPER_USERMODE_A_PARAMS;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


#ifdef __cplusplus
};
#endif
#endif /* LWOS_INCLUDED */
