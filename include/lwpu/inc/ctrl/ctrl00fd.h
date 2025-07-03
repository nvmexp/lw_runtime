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
// Source file: ctrl/ctrl00fd.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrlxxxx.h"

#define LW00FD_CTRL_CMD(cat,idx)       LWXXXX_CTRL_CMD(0x00fd, LW00FD_CTRL_##cat, idx)

/* LW00FD command categories (6bits) */
#define LW00FD_CTRL_RESERVED         (0x00)
#define LW00FD_CTRL_MULTICAST_FABRIC (0x01)

/*
 * LW00FD_CTRL_CMD_NULL
 *
 * This command does nothing.
 * This command does not take any parameters.
 *
 * Possible status values returned are:
 *   LW_OK
 */
#define LW00FD_CTRL_CMD_NULL         (0xfd0000) /* finn: Evaluated from "(FINN_LW_MEMORY_MULTICAST_FABRIC_RESERVED_INTERFACE_ID << 8) | 0x0" */



/*
 * LW00FD_CTRL_CMD_GET_INFO
 *
 * Queries multicast memory fabric allocation attributes.
 *
 *  alignment [OUT]
 *    Alignment for the allocation.
 *
 *  allocSize [OUT]
 *    Size of the allocation.
 *
 *  pageSize [OUT]
 *    Page size of the allocation.
 *
 *  numMaxGpus [OUT]
 *    Maximum number of attachable GPUs
 *
 *  numAttachedGpus [OUT]
 *    Number of GPUs lwrrently attached
 *
 */
#define LW00FD_CTRL_CMD_GET_INFO (0xfd0101) /* finn: Evaluated from "(FINN_LW_MEMORY_MULTICAST_FABRIC_FABRIC_INTERFACE_ID << 8) | LW00FD_CTRL_GET_INFO_PARAMS_MESSAGE_ID" */

#define LW00FD_CTRL_GET_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW00FD_CTRL_GET_INFO_PARAMS {
    LW_DECLARE_ALIGNED(LwU64 alignment, 8);
    LW_DECLARE_ALIGNED(LwU64 allocSize, 8);
    LwU32 pageSize;
    LwU32 numMaxGpus;
    LwU32 numAttachedGpus;
} LW00FD_CTRL_GET_INFO_PARAMS;

/*
 * LW00FD_CTRL_CMD_ATTACH_MEM
 *
 * Attaches the physical memory handle and in turn the memory
 * owner of the physical memory to the Multicast FLA object.
 *
 *  hSubdevice [IN]
 *    Subdevice handle of the owner GPU
 *
 *  ownerGpuId [IN]
 *    GPU ID of the memory owner.
 *
 *  hVidMem [IN]
 *    Physical memory handle to be attached.
 *
 *  offset [IN]
 *    Offset into the physical memory descriptor.
 *
 *  flags [IN]
 *    For future use only.
 *
 *  Restrictions:
 *  a.Memory belonging to only LWSwitch P2P supported GPUs
 *    which can do multicast can be attached
 *  b.2MB page size is allowed
 *  c.Memory size from the provided offset should be less than
 *    or equal to Multicast FLA alloc size
 *  d.Memory of an already attached GPU should not be attached
 *  e.Only vidmem physical memory handle can be attached
 *
 */
#define LW00FD_CTRL_CMD_ATTACH_MEM (0xfd0102) /* finn: Evaluated from "(FINN_LW_MEMORY_MULTICAST_FABRIC_FABRIC_INTERFACE_ID << 8) | LW00FD_CTRL_ATTACH_MEM_PARAMS_MESSAGE_ID" */

#define LW00FD_CTRL_ATTACH_MEM_PARAMS_MESSAGE_ID (0x2U)

typedef struct LW00FD_CTRL_ATTACH_MEM_PARAMS {
    LwHandle hSubdevice;
    LwU32    ownerGpuId;
    LwHandle hVidMem;
    LW_DECLARE_ALIGNED(LwU64 offset, 8);
    LwU32    flags;
} LW00FD_CTRL_ATTACH_MEM_PARAMS;

/*
 * LW00FD_CTRL_CMD_REGISTER_EVENT
 *
 * Allows clients to optionally register for events after the Multicast
 * FLA object is duped under another client.
 *
 *  pOsEvent [IN]
 *    OS event handle created with LwRmAllocOsEvent().
 *
 */
#define LW00FD_CTRL_CMD_REGISTER_EVENT (0xfd0103) /* finn: Evaluated from "(FINN_LW_MEMORY_MULTICAST_FABRIC_FABRIC_INTERFACE_ID << 8) | LW00FD_CTRL_REGISTER_EVENT_PARAMS_MESSAGE_ID" */

#define LW00FD_CTRL_REGISTER_EVENT_PARAMS_MESSAGE_ID (0x3U)

typedef struct LW00FD_CTRL_REGISTER_EVENT_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pOsEvent, 8);
} LW00FD_CTRL_REGISTER_EVENT_PARAMS;

/* _ctrl00fd_h_ */
