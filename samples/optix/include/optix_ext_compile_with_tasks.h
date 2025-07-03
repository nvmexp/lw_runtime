/*
 * Copyright (c) 2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

/**
 * @file   optix_ext_compile_new_backend.h
 * @author LWPU Corporation
 * @brief  OptiX public API header
 */

#ifndef __optix_optix_ext_compile_with_tasks_h__
#define __optix_optix_ext_compile_with_tasks_h__

#include "optix_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/// The algorithm for splitting a module in multiple tasks allocates functions to each
/// task based on the function's "cost". This "cost" is lwrrently callwlated as the number
/// of basic blocks, because many of the more costly compiler optimizations are big-O in
/// the number of basic blocks. This heuristic may be adjusted in the future.
///
/// In order to help tune this heuristic the following value can be queried and set.
///
/// The value is set on the OptixDeviceContext and will affect any subsequent calls to
/// #optixTaskExelwte().
OptixResult optixExtCompileWithTasksSetMinBinSize( OptixDeviceContext context, unsigned int minBinSize );
OptixResult optixExtCompileWithTasksGetMinBinSize( OptixDeviceContext context, unsigned int* minBinSize );

#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
// When changing the ABI version make sure you know exactly what you are doing. See
// apps/optix/exp/functionTable/functionTable.cpp for instructions. See
// https://confluence.lwpu.com/display/RAV/ABI+Versions+in+the+Wild for released ABI versions.
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
#define OPTIX_EXT_COMPILE_WITH_TASKS_ABI_VERSION 7002

typedef struct OptixExtCompileWithTasksFunctionTable
{
    OptixResult ( *optixExtCompileWithTasksSetMinBinSize )( OptixDeviceContext context, unsigned int minBinSize );
    OptixResult ( *optixExtCompileWithTasksGetMinBinSize )( OptixDeviceContext context, unsigned int* minBinSize );
} OptixExtCompileWithTasksFunctionTable;


#ifdef __cplusplus
}
#endif

#endif /* __optix_optix_ext_compile_with_tasks_h__ */
