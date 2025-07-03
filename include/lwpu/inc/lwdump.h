////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// Copyright lWpu Corporation.  All rights reserved.                        //
//                                                                            //
// The information contained herein is proprietary and confidential to        //
// lWpu Corporation.  Use, reproduction or disclosure to any third party    //
// is subject to written pre-approval by lWpu Corporation.                  //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////
//
//  Module: lwDump.h
//      Shared definitions for HW/SW dumping facility residing in resman/rc.
//
// **************************************************************************

#ifndef _LWDUMP_H_
#define _LWDUMP_H_

#include "lwtypes.h"

//-------------------------------------------------------------------------
// DEFINTIONS
//-------------------------------------------------------------------------

#define LWDUMP_SUB_ALLOC_NOT_ENCODED      0x01
#define LWDUMP_SUB_ALLOC_VALID            0x02
#define LWDUMP_SUB_ALLOC_HAS_MORE         0x04

#define LWDUMP_DEBUG_BUFFER_MAX_SIZE            4096   // max of 4K per buffer
#define LWDUMP_DEBUG_BUFFER_MAX_SUBALLOCATIONS  256

#define LWDUMP_CONFIG_SIGNATURE "LWDUMPCONFIGSIG" // multiple of 8 bytes

typedef enum
{
    // The following components are GPU instance specific:
    LWDUMP_COMPONENT_DEBUG_BUFFERS = 0,
    LWDUMP_COMPONENT_ENG_MC,
    LWDUMP_COMPONENT_ENG_FIFO,
    LWDUMP_COMPONENT_ENG_GRAPHICS,
    LWDUMP_COMPONENT_ENG_FB,
    LWDUMP_COMPONENT_ENG_DISP,
    LWDUMP_COMPONENT_ENG_FAN,
    LWDUMP_COMPONENT_ENG_THERMAL,
    LWDUMP_COMPONENT_ENG_FUSE,
    LWDUMP_COMPONENT_ENG_VBIOS,
    LWDUMP_COMPONENT_ENG_PERF,
    LWDUMP_COMPONENT_ENG_BUS,
    LWDUMP_COMPONENT_ENG_PMU,
    LWDUMP_COMPONENT_ENG_ALL,
    LWDUMP_COMPONENT_ENG_CE,
    LWDUMP_COMPONENT_ENG_GPU,
    LWDUMP_COMPONENT_ENG_LPWR,
    LWDUMP_COMPONENT_ENG_LWD,
    LWDUMP_COMPONENT_ENG_VOLT,
    LWDUMP_COMPONENT_ENG_CLK,
    LWDUMP_COMPONENT_ENG_SEC2,
    LWDUMP_COMPONENT_ENG_LWLINK,
    LWDUMP_COMPONENT_ENG_BSP,
    LWDUMP_COMPONENT_ENG_DPU,
    LWDUMP_COMPONENT_ENG_FBFLCN,
    LWDUMP_COMPONENT_ENG_HDA,
    LWDUMP_COMPONENT_ENG_MSENC,
    LWDUMP_COMPONENT_ENG_GSP,
    LWDUMP_COMPONENT_ENG_INFOROM,
    LWDUMP_COMPONENT_ENG_GCX,
    // The following components are global to the system:
    LWDUMP_COMPONENT_SYS_RCDB = 0x400,
    LWDUMP_COMPONENT_SYS_SYSINFO,
    LWDUMP_COMPONENT_SYS_ALL,
    // The following components are lwlog related.
    LWDUMP_COMPONENT_LWLOG_RM = 0x800,
    LWDUMP_COMPONENT_LWLOG_ALL,
    // Reserved
    LWDUMP_COMPONENT_LWLOG_RESERVED = 0xB00,
} LWDUMP_COMPONENT;

#define LWDUMP_IS_GPU_COMPONENT(c)   ((c) < LWDUMP_COMPONENT_SYS_RCDB)
#define LWDUMP_IS_SYS_COMPONENT(c)   (((c) >= LWDUMP_COMPONENT_SYS_RCDB) && \
                                      ((c) < LWDUMP_COMPONENT_LWLOG_RM))
#define LWDUMP_IS_LWLOG_COMPONENT(c) (((c) >= LWDUMP_COMPONENT_LWLOG_RM) && \
                                      ((c) < LWDUMP_COMPONENT_LWLOG_RESERVED))

typedef enum
{
    LWDUMP_BUFFER_PROVIDED = 0,                // Dump buffer provided by caller
    LWDUMP_BUFFER_ALLOCATE,                    // Dump buffer to be allocated
    LWDUMP_BUFFER_COUNT,                       // Just count, no buffer needed
} LWDUMP_BUFFER_POLICY;

typedef enum
{
    LWDUMP_STATUS_IDLE,
    LWDUMP_STATUS_ERROR,
    LWDUMP_STATUS_COUNT_REQUESTED,
    LWDUMP_STATUS_COUNT_COMPLETE,
    LWDUMP_STATUS_DUMP_REQUESTED,
    LWDUMP_STATUS_DUMP_BUFFER_FULL,
    LWDUMP_STATUS_DUMP_END_OF_MSG,
    LWDUMP_STATUS_DUMP_COMPLETE,
} LWDUMP_STATUS;

//
// The following structures require that all elements are precisely sized
// and aligned on natural boundaries.
//

typedef struct
{
    LwP64 address LW_ALIGN_BYTES(8);
    LwU32 size;
    LwU32 lwrNumBytes;
} LWDUMP_BUFFER;

// Max number of bytes that can be returned in a dump buffer
#define LWDUMP_MAX_DUMP_SIZE (1 << 20) // 1 MB

typedef struct
{
    LwU8 sigHead[sizeof(LWDUMP_CONFIG_SIGNATURE)] LW_ALIGN_BYTES(8);

    LwP64 debuggerControlFuncAddr LW_ALIGN_BYTES(8);
    LWDUMP_BUFFER buffer;
    LwU32 gpuSelect;
    LwU32 component;
    LwU32 dumpStatus;
    LwU32 rmStatus;

    LwU8 sigTail[sizeof(LWDUMP_CONFIG_SIGNATURE)] LW_ALIGN_BYTES(8);
} LWDUMP_CONFIG;

typedef struct
{
     LwU16     length;
     LwU16     start;
     LwU16     end;
     LwU16     flags;
     LwU16     tag;
} LWDUMP_SUB_ALLOC_HEADER;

//
// Export is needed to allow remote kernel debuggers without symbols
// to find global LWDUMP_CONFIG variables in driver export address table.
//
#if (defined(_WIN32) || defined(_WIN64))
#define LWDUMP_EXPORT __declspec(dllexport)
#else
#define LWDUMP_EXPORT
#endif

#endif // _LWDUMP_H_

