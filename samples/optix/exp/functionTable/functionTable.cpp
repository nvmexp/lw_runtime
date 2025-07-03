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

// For optixQueryFunctionTable()
#include <optix_types.h>
// For the ABI versions of the main API and the extension APIs
#include <optix_ext_compile_no_inline.h>
#include <optix_ext_compile_with_tasks.h>
#include <optix_ext_compute_instance_aabbs.h>
#include <optix_ext_feature_query.h>
#include <optix_ext_knobs.h>
#include <optix_ext_ptx_encryption.h>
#include <optix_ext_compile_new_backend.h>
#include <optix_function_table.h>

#include <exp/context/ForceDeprecatedCompiler.h>

#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

#include <corelib/system/System.h>

#include <lwselwreloadlibrary/SelwreExelwtableModule.h>

#if !defined(_WIN32)
#include <common/inc/lwUnixVersion.h>
#endif

// Undefine any existing definition for OPTIXAPI
#if defined( OPTIXAPI )
#undef OPTIXAPI
#endif

// On Windows, we're going to export functions with the generated .def file, so don't
// double export them here and define OPTIXAPI to be empty.
#if defined( _WIN32 )
#define OPTIXAPI

// On Linux flavors, we need to set the visibility attribute to 'default'.
#elif defined( __linux__ ) || defined( __CYGWIN__ )
#define OPTIXAPI __attribute__( ( visibility( "default" ) ) )
#elif defined( __APPLE__ ) && defined( __MACH__ )
#define OPTIXAPI __attribute__( ( visibility( "default" ) ) )

#else
#error "CODE FOR THIS OS HAS NOT YET BEEN DEFINED"
#endif

// Instructions for changing the ABI version
// =========================================
//
// Most important: Do *not* break released ABIs.
//          See https://confluence.lwpu.com/display/RAV/ABI+Versions+in+the+Wild for released ABI versions.
//
// 1. Steps that are *always* needed when changing the ABI version:
// - Increase number of #define OPTIX_ABI_VERSION XX in optix_function_table.h (duh)
// - Add new entry to the OptixABI enum class in OptixABI.h
// - Add that new enum value to the checkABIVersion function in DeviceContext.cpp
// - Add definition for the new DeviceContext create function in DeviceContext.cpp: OPTIX_DEVICE_CONTEXT_CREATE_IMPL( XX );
// - Add comment in history of ABI changes before optixQueryFunctionTable below.
//
// 2a. ========== Additional steps when the previous ABI version *has not been released* AND if the new ABI version does not change the host API: ===========
// - Add declaration for the new DeviceContext create function in functionTable_lwrrent.cpp:
//      OptixResult OPTIX_DEVICE_CONTEXT_CREATE_NAME( XX )(LWcontext, const OptixDeviceContextOptions*, OptixDeviceContext*);
//   and make sure that fillFunctionTable_lwrrent adds the right function to the function table:
//      case XX:
//          ftable->optixDeviceContextCreate = OPTIX_DEVICE_CONTEXT_CREATE_NAME( XX );
//          break;
// - add the new ABI version to the versions that call optix_exp::fillFunctionTable_lwrrent in optixQueryFunctionTable below
//
// 2b. ========== Additional steps when the previous version *has been released* OR the new ABI version changes the host API: ============================
// If the current ABI version (OPTIX_ABI_VERSION in optix_function_table.h) has been released, we need to maintain it.
// Maintaining the current ABI requires extra steps to archive it before incrementing OPTIX_ABI_VERSION. Do *not* skip
// these steps, otherwise the old ABI will become unsupported. The wiki page referenced above should mention all
// released ABI versions. Ask for approval if you are unsure or believe that the wiki page is not up-to-date.
//
// Steps to archive the current ABI version:
// - $ export lwrrentVersion=... # (pre-increase value of OPTIX_ABI_VERSION goes here)
//   $ cd exp/functionTable/
//   $ p4 integrate functionTable_lwrrent.cpp functionTable_$lwrrentVersion.cpp
//   $ p4 edit functionTable_$lwrrentVersion.cpp
//   $ sed -i s/Table_lwrrent/Table_$lwrrentVersion/g functionTable_$lwrrentVersion.cpp
//   $ sed -i "s/ABI_VERSION_TO_USE OPTIX_ABI_VERSION/ABI_VERSION_TO_USE $lwrrentVersion/g" functionTable_$lwrrentVersion.cpp
// - add functionTable_$lwrrentVersion.cpp to functionTable.lwmk
// - add fillFunctionTable_$lwrrentVersion() to the forward declarations below
// - in functionTable_lwrrent.cpp:
//      - Remove all OptixResult OPTIX_DEVICE_CONTEXT_CREATE_NAME( XX )(LWcontext, const OptixDeviceContextOptions*, OptixDeviceContext*); lines except one
//      - Change that one line to specify the new ABI version for `XX`
//      - Remove all cases from fillFunctionTable_$lwrrentVersion except for one
//      - Change that one case to handle the new ABI version
// - edit optixQueryFunctionTable() below to add an entry for fillFunctionTable_$lwrrentVersion() (use integral constants
//   in the condition, not OPTIX_ABI_VERSION)
// - leave fillFunctionTable_lwrrent() as the last entry in optixQueryFunctionTable() for abiId == OPTIX_ABI_VERSION
//
// After these steps have been performed (or it has been determined that the current ABI version does not need to be
// maintained, e.g., because it was a short-lived ABI that was never released), you can increment OPTIX_ABI_VERSION.
//
// When making the actual ABI changes make sure *not* to change any methods that are used (directly or indirectly) by
// past ABI versions. For signature changes of API functions this means changing optixFoo to optixFoo_v2 in
// functionTable_lwrrent.cpp and implementing optixFoo_v2() *in* *adddition* to optixFoo(). Depending on the changes
// you might need to duplicate callees as well.

namespace optix_exp {

OptixResult fillFunctionTable_lwrrent( int abi, void* functionTable, size_t sizeOfFunctionTable );
OptixResult fillFunctionTable_20( int abi, void* functionTable, size_t sizeOfFunctionTable );
OptixResult fillFunctionTable_22( int abi, void* functionTable, size_t sizeOfFunctionTable );
OptixResult fillFunctionTable_25( int abi, void* functionTable, size_t sizeOfFunctionTable );
OptixResult fillFunctionTable_38( int abi, void* functionTable, size_t sizeOfFunctionTable );
OptixResult fillFunctionTable_43( int abi, void* functionTable, size_t sizeOfFunctionTable );
OptixResult fillFunctionTable_52( int abi, void* functionTable, size_t sizeOfFunctionTable );

OptixResult fillFunctionTableExtCompileNoInline_lwrrent( void* functionTable, size_t sizeOfFunctionTable );

OptixResult fillFunctionTableExtKnobs_lwrrent( void* functionTable, size_t sizeOfFunctionTable );

OptixResult fillFunctionTableExtPtxEncryption_lwrrent( void* functionTable, size_t sizeOfFunctionTable );

OptixResult fillFunctionTableExtComputeInstanceAabbs_lwrrent( void* functionTable, size_t sizeOfFunctionTable );

OptixResult fillFunctionTableExtCompileNewBackend_lwrrent( void* functionTable, size_t sizeOfFunctionTable );

OptixResult fillFunctionTableExtCompileWithTasks_lwrrent( int abi, void* functionTable, size_t sizeOfFunctionTable );

OptixResult fillFunctionTableExtFeatureQuery_lwrrent( void* functionTable, size_t sizeOfFunctionTable );

}  // namespace optix_exp

static std::mutex g_knobInitialization;

static bool g_knobsInitialized = false;

// Initialize knobs once (after static initialization, but as soon as possible).
static void initializeKnobs()
{
    try
    {
        // Intentionally ignore return value. Knob initialization is a by-product of the optixQueryFunctionTable() API
        // call and knob failures should not affect that API call. The knob registry caches failure messages anyway to
        // make them available later when a logger is available.
        std::lock_guard<std::mutex> lock( g_knobInitialization );
        if( !g_knobsInitialized )
        {
            knobRegistry().initializeKnobs();
            g_knobsInitialized = true;
        }
    }
    catch( ... )
    {
        // Ignore all exceptions for the same reasons.
    }
}


// History of ABI version changes (i.e., API or PTX support changed, w/, w/o backwards compatibility):
// 18 - baseline [first ABI version used by an official product]
// 19 - no API change, PTX support changed w/ backwards compatibility
// 20 - denoiser, new PTX functions for ... (e.g. exceptions)
// 21 - rtcore fat instances
// 22 - added pixelStrideInBytes to OptixImage2D (no backwards compatibility)
// 23 - increase instancing limits (make use of added bits of fat instances)
// 24 - remove OptixPipelineLinkOptions::overrideUsesMotionBlur
// 25 - add transformFormat to triangle
// 26 - add optixBuiltinISModuleGet
// 27 - replace OptixLwrveType and OptixBuiltinISModuleType by OptixPrimitiveType. add OptixPipelineCompileOptions::usesPrimitiveTypeFlags.
// 28 - removed pixelFormat from OptixDenoiserOptions, renamed sizes in OptixDenoiserSizes.
// 29 - guarantee abi break between r440 GA5 (no public lwrves support) and GA6 (public lwrves support).
// 30 - add optixGetLinearLwrveVertexData for piecewise linear lwrves.
// 31 - add optixGetExceptionIlwalidRay
// 32 - Replace support for optixGetPrimitiveIndex in EX shader by optixGetSbtGASIndex.
// 33 - add optixGetExceptionParameterMismatch
// 34 - Change OptixCompileOptimizationLevel and OptixCompileDebugLevel to add a DEFAULT of 0, added 0x2* enum values.
// 35 - Throw new OPTIX_EXCEPTION_CODE_BUILTIN_IS_MISMATCH exception when builtin IS doesn't match the GAS buildinput.
// 36 - Do not produce an error on optixThrowException if user exceptions are disabled.
// 37 - add OptixDeviceContextValidationMode, extend OptixDeviceContextOptions accordingly
// 38 - Added OptixModuleCompileBoundValueEntry to OptixModuleCompileOptions
// 39 - Added optixDenoiserComputeAverageColor to host API, added hdrAverageColor in struct OptixDenoiserParams
// 40 - (originally 41) Fix compile optimization level to O2 for any old ABI due to bug 3034630
// 41 - (originally 43) Remove OptixBuildInputInstanceArray::aabbs, OptixBuildInputInstanceArray::numAabbs
// 42 - (originally 40) Added optixGetInstanceTraversableFromIAS and optixGetInstanceChildFromHandle to device API
// 43 - (originally 42) Add texture footprint intrinsic in sparse texture extension API.
// 44 - add optixDenoiserCreateWithUserModel, optixDenoiserCreate and optixDenoiserIlwoke signature changed, removed optixDenoiserSetModel
// 45 - switch default lwrve intersector to phantom
// 46 - increase max payload values to 32
// 47 - add payload type semantics
// 48 - add distinct per-module payload type semantics
// 49 - remove OPTIX_INSTANCE_FLAG_DISABLE_TRANSFORM flag from the interface (never really worked)
// 50 - enable payload types in the public API
// 51 - add build flags to OptixBuiltinISOptions
// 52 - support Catmull-Rom splines
// 53 - add VM builder, optixModuleCreateFromPTXWithTasks, optixModuleGetCompilationState, optixTaskExelwte
// 54 - add lwrve end cap flags
// 55 - Added OptixCompileDebugLevel::OPTIX_COMPILE_DEBUG_LEVEL_MODERATE, renamed OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO to OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL
// 56 - Added optixPrivateGetCompileTimeConstant device function
// 57 - Added OptixGeometryFlags::OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_LWLLING
// 58 - support sphere primitives
// 59 - Added computeAverageColorSizeInBytes, computeIntensitySizeInBytes to struct OptixDenoiserSizes
// 60 - Added hiddenPreviousOutput, hiddenOutput to OptixDenoiserGuideLayer
// !Please leave the following list of extension ABI versions unchanged, it is edited by the generator script! Add ABI versions above.
// 1001 - OPTIX_EXT_KNOBS_ABI_VERSION
// 2001 - OPTIX_EXT_PTX_ENCRYPTION_ABI_VERSION
// 3001 - OPTIX_EXT_COMPILE_NO_INLINE_ABI_VERSION
// 4001 - OPTIX_EXT_COMPUTE_INSTANCE_AABBS_ABI_VERSION
// 6001 - OPTIX_EXT_COMPILE_NEW_BACKEND_ABI_VERSION
// 7001 - OPTIX_EXT_COMPILE_WITH_TASKS_ABI_VERSION
// 7002 - OPTIX_EXT_COMPILE_WITH_TASKS_ABI_VERSION, removed optixModuleCreateFromPTXWithTasks, optixModuleGetCompilationState, optixTaskExelwte
// 8001 - OPTIX_EXT_FEATURE_QUERY_ABI_VERSION
extern "C" OptixResult OPTIXAPI optixQueryFunctionTable( int                             abiId,
                                                         unsigned int                    numOptions,
                                                         OptixQueryFunctionTableOptions* optionKeys,
                                                         const void**                    optiolwalues,
                                                         void*                           functionTable,
                                                         size_t                          sizeOfFunctionTable )
{
    initializeKnobs();

    lprint << "optixQueryFunctionTable: Requested ABI " << abiId << '\n';

    if( numOptions != 0 )
        return OPTIX_ERROR_ILWALID_ENTRY_FUNCTION_OPTIONS;

    // main API

    if( abiId <= 17 )
        return OPTIX_ERROR_UNSUPPORTED_ABI_VERSION;

    if( abiId <= 20 )
        return optix_exp::fillFunctionTable_20( abiId, functionTable, sizeOfFunctionTable );

    if( abiId <= 22 )
        return optix_exp::fillFunctionTable_22( abiId, functionTable, sizeOfFunctionTable );

    if( abiId <= 25 )
        return optix_exp::fillFunctionTable_25( abiId, functionTable, sizeOfFunctionTable );

    if( abiId <= 38 )
        return optix_exp::fillFunctionTable_38( abiId, functionTable, sizeOfFunctionTable );

    if( abiId <= 43 )
        return optix_exp::fillFunctionTable_43( abiId, functionTable, sizeOfFunctionTable );

    if( abiId <= 52 )
        return optix_exp::fillFunctionTable_52( abiId, functionTable, sizeOfFunctionTable );

    if( abiId <= OPTIX_ABI_VERSION )
        return optix_exp::fillFunctionTable_lwrrent( abiId, functionTable, sizeOfFunctionTable );

    // Extension API for knobs

    if( abiId == OPTIX_EXT_KNOBS_ABI_VERSION )
        return optix_exp::fillFunctionTableExtKnobs_lwrrent( functionTable, sizeOfFunctionTable );

    // Extension API for PTX encryption

    if( abiId == OPTIX_EXT_PTX_ENCRYPTION_ABI_VERSION )
        return optix_exp::fillFunctionTableExtPtxEncryption_lwrrent( functionTable, sizeOfFunctionTable );

    // Extension API for no-inline

    if( abiId == OPTIX_EXT_COMPILE_NO_INLINE_ABI_VERSION )
        return optix_exp::fillFunctionTableExtCompileNoInline_lwrrent( functionTable, sizeOfFunctionTable );

    if( abiId == OPTIX_EXT_COMPUTE_INSTANCE_AABBS_ABI_VERSION )
        return optix_exp::fillFunctionTableExtComputeInstanceAabbs_lwrrent( functionTable, sizeOfFunctionTable );

    if( abiId == OPTIX_EXT_COMPILE_NEW_BACKEND_ABI_VERSION )
        return optix_exp::fillFunctionTableExtCompileNewBackend_lwrrent( functionTable, sizeOfFunctionTable );

    if( abiId == 7001 || abiId == OPTIX_EXT_COMPILE_WITH_TASKS_ABI_VERSION )
        return optix_exp::fillFunctionTableExtCompileWithTasks_lwrrent( abiId, functionTable, sizeOfFunctionTable );

    if( abiId == OPTIX_EXT_FEATURE_QUERY_ABI_VERSION )
        return optix_exp::fillFunctionTableExtFeatureQuery_lwrrent( functionTable, sizeOfFunctionTable );

    return OPTIX_ERROR_UNSUPPORTED_ABI_VERSION;
}
