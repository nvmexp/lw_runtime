/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
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

#include <optix_ext_compile_no_inline.h>
#include <optix_types.h>

#include <cstring>

namespace optix_exp {

namespace {

// This struct is just a permanent copy of the then-current struct OptixExtCompileNoInlineFunctionTable in
// optix_ext_compile_no_inline.h.
//
// We could use an array of void* here, but the explicit types prevent mistakes like ordering
// problems or signature changes of functions used in tables of released ABI versions.
struct FunctionTableExtCompileNoInline_lwrrent
{
    OptixResult ( *optixExtCompileSetNoInlineEnabled )( OptixDeviceContext contextAPI, bool enabled );
    bool ( *optixExtCompileIsNoInlineEnabled )( OptixDeviceContext contextAPI );
};

FunctionTableExtCompileNoInline_lwrrent g_functionTableExtCompileNoInline_lwrrent = {
    // clang-format off
    optixExtCompileNoInlineSetEnabled,
    optixExtCompileNoInlineIsEnabled
    // clang-format on
};
}

OptixResult fillFunctionTableExtCompileNoInline_lwrrent( void* functionTable, size_t sizeOfFunctionTable )
{
    if( sizeOfFunctionTable != sizeof( FunctionTableExtCompileNoInline_lwrrent ) )
        return OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH;

    memcpy( functionTable, &g_functionTableExtCompileNoInline_lwrrent, sizeof( FunctionTableExtCompileNoInline_lwrrent ) );
    return OPTIX_SUCCESS;
}
}
