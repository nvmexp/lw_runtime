/*
 * LWIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2016-2021, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 *
 * LWIDIA_COPYRIGHT_END
 */

#ifndef interfaceUtils_INCLUDED
#define interfaceUtils_INCLUDED

#include <stdLocal.h>
#include "stdThreads.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NUM_MUTEXES 6

#define LWPTXCOMPILER_IFACE_LOCK  0              // no longer used, can be repurposed.
#define CALL_JIT_ENTRY_IFACE_LOCK 1
#define SET_JIT_ENTRY_POINT_LOCK  2
#define PTX_FULL_LOCK             3
#define PTX_GPUINFO_LOCK          4
#define PTX_GPUCODEGEN_LOCK       5
#define PTX_NO_LOCK               NUM_MUTEXES

void interface_mutex_enter(unsigned int mutexID);
void interface_mutex_exit(unsigned int mutexID);

#ifdef __cplusplus
}
#endif

#endif // interfaceUtils_INCLUDED
