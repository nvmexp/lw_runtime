/*
 * Copyright (c) 2021, LWPU CORPORATION.  All rights reserved.
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

#include <optix_host.h>
#include <optix_ext_compile_with_tasks.h>
#include <optix_types.h>

#include <prodlib/system/Logger.h>

#include <cstring>

namespace optix_exp {

namespace {

// This struct is just a permanent copy of the then-current struct
// OptixExtCompileNewBackendFunctionTable in optix_ext_compile_with_tasks.h.
//
// We could use an array of void* here, but the explicit types prevent mistakes like ordering
// problems or signature changes of functions used in tables of released ABI versions.
struct FunctionTableExtCompileWithTasks_7001
{
    OptixResult ( *optixModuleCreateFromPTXWithTasks )( OptixDeviceContext                 context,
                                                        const OptixModuleCompileOptions*   moduleCompileOptions,
                                                        const OptixPipelineCompileOptions* pipelineCompileOptions,
                                                        const char*                        PTX,
                                                        size_t                             PTXsize,
                                                        char*                              logString,
                                                        size_t*                            logStringSize,
                                                        OptixModule*                       module,
                                                        OptixTask*                         firstTask );

    OptixResult ( *optixModuleGetCompilationState )( OptixModule module, OptixModuleCompileState* state );
    OptixResult ( *optixTaskExelwte )( OptixTask     task,
                                       OptixTask*    additionalTasks,
                                       unsigned int  maxNumAdditionalTasks,
                                       unsigned int* numAdditionalTasksCreated );

    OptixResult ( *optixExtCompileWithTasksSetMinBinSize )( OptixDeviceContext context, unsigned int minBinSize );
    OptixResult ( *optixExtCompileWithTasksGetMinBinSize )( OptixDeviceContext context, unsigned int* minBinSize );
};

FunctionTableExtCompileWithTasks_7001 g_functionTableExtCompileWithTasks_7001 = {
    // clang-format off
    optixModuleCreateFromPTXWithTasks,
    optixModuleGetCompilationState,
    optixTaskExelwte,
    optixExtCompileWithTasksSetMinBinSize,
    optixExtCompileWithTasksGetMinBinSize
    // clang-format on
};

struct FunctionTableExtCompileWithTasks_lwrrent
{
    OptixResult ( *optixExtCompileWithTasksSetMinBinSize )( OptixDeviceContext context, unsigned int minBinSize );
    OptixResult ( *optixExtCompileWithTasksGetMinBinSize )( OptixDeviceContext context, unsigned int* minBinSize );
};

FunctionTableExtCompileWithTasks_lwrrent g_functionTableExtCompileWithTasks_lwrrent = {
    // clang-format off
    optixExtCompileWithTasksSetMinBinSize,
    optixExtCompileWithTasksGetMinBinSize
    // clang-format on
};
}

OptixResult fillFunctionTableExtCompileWithTasks_lwrrent( int abi, void* functionTable, size_t sizeOfFunctionTable )
{
    switch( abi )
    {
        case 7001:

            if( sizeOfFunctionTable != sizeof( FunctionTableExtCompileWithTasks_7001 ) )
            {
#if defined( OPTIX_ENABLE_LOGGING )
                lerr << "sizeOfFunctionTable != sizeof( FunctionTableExtCompileWithTasks_7001 ) ( " << sizeOfFunctionTable << " != " << sizeof( FunctionTableExtCompileWithTasks_7001 ) << ")\n";
#endif
                return OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH;
            }

            memcpy( functionTable, &g_functionTableExtCompileWithTasks_7001, sizeof( FunctionTableExtCompileWithTasks_7001 ) );
            break;
        case OPTIX_EXT_COMPILE_WITH_TASKS_ABI_VERSION:

            if( sizeOfFunctionTable != sizeof( FunctionTableExtCompileWithTasks_lwrrent ) )
            {
#if defined( OPTIX_ENABLE_LOGGING )
                lerr << "sizeOfFunctionTable != sizeof( FunctionTableExtCompileWithTasks_lwrrent ) ( " << sizeOfFunctionTable << " != " << sizeof( FunctionTableExtCompileWithTasks_lwrrent ) << ")\n";
#endif
                return OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH;
            }

            memcpy( functionTable, &g_functionTableExtCompileWithTasks_lwrrent, sizeof( FunctionTableExtCompileWithTasks_lwrrent ) );
            break;
        default:
#if defined( OPTIX_ENABLE_LOGGING )
            lerr << "Unknown ABI version : " << abi << "\n";
#endif
            return OPTIX_ERROR_UNSUPPORTED_ABI_VERSION;
    }
    return OPTIX_SUCCESS;
};
}
