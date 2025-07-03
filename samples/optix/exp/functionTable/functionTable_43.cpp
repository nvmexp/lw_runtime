/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
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
#include <exp/functionTable/optix_host_denoiser_v1.h>
#include <prodlib/system/Logger.h>

#include <cstring>

// When bumping OPTIX_ABI_VERSION, add a new entry for OPTIX_DEVICE_CONTEXT_CREATE_NAME
// and below in the switch statement.  Additionally, OPTIX_DEVICE_CONTEXT_CREATE_IMPL(
// $lwrrentVersion ); needs to be added to DeviceContext.cpp to create the implementation
// of the function.  See functionTable.cpp for additional information.

#define OPTIX_CREATE_CREATE_FUNCTION_NAME( x ) optixDeviceContextCreate_##x
#define OPTIX_DEVICE_CONTEXT_CREATE_NAME( abi ) OPTIX_CREATE_CREATE_FUNCTION_NAME( abi )
extern "C" {
OptixResult OPTIX_DEVICE_CONTEXT_CREATE_NAME( 39 )( LWcontext, const OptixDeviceContextOptions*, OptixDeviceContext* );
OptixResult OPTIX_DEVICE_CONTEXT_CREATE_NAME( 40 )( LWcontext, const OptixDeviceContextOptions*, OptixDeviceContext* );
OptixResult OPTIX_DEVICE_CONTEXT_CREATE_NAME( 41 )( LWcontext, const OptixDeviceContextOptions*, OptixDeviceContext* );
OptixResult OPTIX_DEVICE_CONTEXT_CREATE_NAME( 42 )( LWcontext, const OptixDeviceContextOptions*, OptixDeviceContext* );
OptixResult OPTIX_DEVICE_CONTEXT_CREATE_NAME( 43 )( LWcontext, const OptixDeviceContextOptions*, OptixDeviceContext* );
}


namespace optix_exp {

namespace {

// This struct is just a permanent copy of the then-current struct OptixFunctionTable in
// optix_function_table.h.
//
// We could use an array of void* here, but the explicit types prevent mistakes like ordering
// problems or signature changes of functions used in tables of released ABI versions.
struct FunctionTable_43
{
    const char* ( *optixGetErrorName )( OptixResult result );
    const char* ( *optixGetErrorString )( OptixResult result );

    /* Device context */

    OptixResult ( *optixDeviceContextCreate )( LWcontext fromContext, const OptixDeviceContextOptions* options, OptixDeviceContext* context );

    OptixResult ( *optixDeviceContextDestroy )( OptixDeviceContext context );

    OptixResult ( *optixDeviceContextGetProperty )( OptixDeviceContext context, OptixDeviceProperty property, void* value, size_t sizeInBytes );

    OptixResult ( *optixDeviceContextSetLogCallback )( OptixDeviceContext context,
                                                       OptixLogCallback   callbackFunction,
                                                       void*              callbackData,
                                                       unsigned int       callbackLevel );

    OptixResult ( *optixDeviceContextSetCacheEnabled )( OptixDeviceContext context, int enabled );

    OptixResult ( *optixDeviceContextSetCacheLocation )( OptixDeviceContext context, const char* location );

    OptixResult ( *optixDeviceContextSetCacheDatabaseSizes )( OptixDeviceContext context, size_t lowWaterMark, size_t highWaterMark );

    OptixResult ( *optixDeviceContextGetCacheEnabled )( OptixDeviceContext context, int* enabled );

    OptixResult ( *optixDeviceContextGetCacheLocation )( OptixDeviceContext context, char* location, size_t locationSize );

    OptixResult ( *optixDeviceContextGetCacheDatabaseSizes )( OptixDeviceContext context, size_t* lowWaterMark, size_t* highWaterMark );

    /* Modules */

    OptixResult ( *optixModuleCreateFromPTX )( OptixDeviceContext                 context,
                                               const OptixModuleCompileOptions*   moduleCompileOptions,
                                               const OptixPipelineCompileOptions* pipelineCompileOptions,
                                               const char*                        PTX,
                                               size_t                             PTXsize,
                                               char*                              logString,
                                               size_t*                            logStringSize,
                                               OptixModule*                       module );

    OptixResult ( *optixModuleDestroy )( OptixModule module );

    OptixResult ( *optixBuiltinISModuleGet )( OptixDeviceContext                 context,
                                              const OptixModuleCompileOptions*   moduleCompileOptions,
                                              const OptixPipelineCompileOptions* pipelineCompileOptions,
                                              const OptixBuiltinISOptions*       builtinISOptions,
                                              OptixModule*                       builtinModule );

    /* Program groups */

    OptixResult ( *optixProgramGroupCreate )( OptixDeviceContext              context,
                                              const OptixProgramGroupDesc*    programDescriptions,
                                              unsigned int                    numProgramGroups,
                                              const OptixProgramGroupOptions* options,
                                              char*                           logString,
                                              size_t*                         logStringSize,
                                              OptixProgramGroup*              programGroups );

    OptixResult ( *optixProgramGroupDestroy )( OptixProgramGroup programGroup );

    OptixResult ( *optixProgramGroupGetStackSize )( OptixProgramGroup programGroup, OptixStackSizes* stackSizes );

    /* Pipeline */

    OptixResult ( *optixPipelineCreate )( OptixDeviceContext                 context,
                                          const OptixPipelineCompileOptions* pipelineCompileOptions,
                                          const OptixPipelineLinkOptions*    pipelineLinkOptions,
                                          const OptixProgramGroup*           programGroups,
                                          unsigned int                       numProgramGroups,
                                          char*                              logString,
                                          size_t*                            logStringSize,
                                          OptixPipeline*                     pipeline );

    OptixResult ( *optixPipelineDestroy )( OptixPipeline pipeline );

    OptixResult ( *optixPipelineSetStackSize )( OptixPipeline pipeline,
                                                unsigned int  directCallableStackSizeFromTraversal,
                                                unsigned int  directCallableStackSizeFromState,
                                                unsigned int  continuationStackSize,
                                                unsigned int  maxTraversableGraphDepth );

    /* Acceleration structures */

    OptixResult ( *optixAccelComputeMemoryUsage )( OptixDeviceContext            context,
                                                   const OptixAccelBuildOptions* accelOptions,
                                                   const OptixBuildInput*        buildInputs,
                                                   unsigned int                  numBuildInputs,
                                                   OptixAccelBufferSizes*        bufferSizes );

    OptixResult ( *optixAccelBuild )( OptixDeviceContext            context,
                                      LWstream                      stream,
                                      const OptixAccelBuildOptions* accelOptions,
                                      const OptixBuildInput*        buildInputs,
                                      unsigned int                  numBuildInputs,
                                      LWdeviceptr                   tempBuffer,
                                      size_t                        tempBufferSizeInBytes,
                                      LWdeviceptr                   outputBuffer,
                                      size_t                        outputBufferSizeInBytes,
                                      OptixTraversableHandle*       outputHandle,
                                      const OptixAccelEmitDesc*     emittedProperties,
                                      unsigned int                  numEmittedProperties );

    OptixResult ( *optixAccelGetRelocationInfo )( OptixDeviceContext context, OptixTraversableHandle handle, OptixAccelRelocationInfo* info );


    OptixResult ( *optixAccelCheckRelocationCompatibility )( OptixDeviceContext              context,
                                                             const OptixAccelRelocationInfo* info,
                                                             int*                            compatible );

    OptixResult ( *optixAccelRelocate )( OptixDeviceContext              context,
                                         LWstream                        stream,
                                         const OptixAccelRelocationInfo* info,
                                         LWdeviceptr                     instanceTraversableHandles,
                                         size_t                          numInstanceTraversableHandles,
                                         LWdeviceptr                     targetAccel,
                                         size_t                          targetAccelSizeInBytes,
                                         OptixTraversableHandle*         targetHandle );


    OptixResult ( *optixAccelCompact )( OptixDeviceContext      context,
                                        LWstream                stream,
                                        OptixTraversableHandle  inputHandle,
                                        LWdeviceptr             outputBuffer,
                                        size_t                  outputBufferSizeInBytes,
                                        OptixTraversableHandle* outputHandle );

    OptixResult ( *optixColwertPointerToTraversableHandle )( OptixDeviceContext      onDevice,
                                                             LWdeviceptr             pointer,
                                                             OptixTraversableType    traversableType,
                                                             OptixTraversableHandle* traversableHandle );

    /* Launch */

    OptixResult ( *optixSbtRecordPackHeader )( OptixProgramGroup programGroup, void* sbtRecordHeaderHostPointer );

    OptixResult ( *optixLaunch )( OptixPipeline                  pipeline,
                                  LWstream                       stream,
                                  LWdeviceptr                    pipelineParams,
                                  size_t                         pipelineParamsSize,
                                  const OptixShaderBindingTable* sbt,
                                  unsigned int                   width,
                                  unsigned int                   height,
                                  unsigned int                   depth );

    /* Denoiser */

    OptixResult ( *optixDenoiserCreate_v1 )( OptixDeviceContext context, const OptixDenoiserOptions_v1* options, OptixDenoiser* returnHandle );
    OptixResult ( *optixDenoiserDestroy )( OptixDenoiser handle );

    OptixResult ( *optixDenoiserComputeMemoryResources )( const OptixDenoiser handle,
                                                          unsigned int        maximumInputWidth,
                                                          unsigned int        maximumInputHeight,
                                                          OptixDenoiserSizes* returnSizes );

    OptixResult ( *optixDenoiserSetup )( OptixDenoiser denoiser,
                                         LWstream      stream,
                                         unsigned int  inputWidth,
                                         unsigned int  inputHeight,
                                         LWdeviceptr   state,
                                         size_t        stateSizeInBytes,
                                         LWdeviceptr   scratch,
                                         size_t        scratchSizeInBytes );


    OptixResult ( *optixDenoiserIlwoke_v1 )( OptixDenoiser              denoiser,
                                             LWstream                   stream,
                                             const OptixDenoiserParams* params,
                                             LWdeviceptr                denoiserState,
                                             size_t                     denoiserStateSizeInBytes,
                                             const OptixImage2D*        inputLayers,
                                             unsigned int               numInputLayers,
                                             unsigned int               inputOffsetX,
                                             unsigned int               inputOffsetY,
                                             const OptixImage2D*        outputLayer,
                                             LWdeviceptr                scratch,
                                             size_t                     scratchSizeInBytes );


    OptixResult ( *optixDenoiserSetModel )( OptixDenoiser handle, OptixDenoiserModelKind_v1 kind, void* data, size_t sizeInBytes );

    OptixResult ( *optixDenoiserComputeIntensity )( OptixDenoiser       handle,
                                                    LWstream            stream,
                                                    const OptixImage2D* inputImage,
                                                    LWdeviceptr         outputIntensity,
                                                    LWdeviceptr         scratch,
                                                    size_t              scratchSizeInBytes );

    OptixResult ( *optixDenoiserComputeAverageColor )( OptixDenoiser       handle,
                                                       LWstream            stream,
                                                       const OptixImage2D* inputImage,
                                                       LWdeviceptr         outputAverageColor,
                                                       LWdeviceptr         scratch,
                                                       size_t              scratchSizeInBytes );
};

FunctionTable_43 g_functionTable_43 = {
    // clang-format off
    optixGetErrorName,
    optixGetErrorString,
    nullptr,
    optixDeviceContextDestroy,
    optixDeviceContextGetProperty,
    optixDeviceContextSetLogCallback,
    optixDeviceContextSetCacheEnabled,
    optixDeviceContextSetCacheLocation,
    optixDeviceContextSetCacheDatabaseSizes,
    optixDeviceContextGetCacheEnabled,
    optixDeviceContextGetCacheLocation,
    optixDeviceContextGetCacheDatabaseSizes,
    optixModuleCreateFromPTX,
    optixModuleDestroy,
    optixBuiltinISModuleGet,
    optixProgramGroupCreate,
    optixProgramGroupDestroy,
    optixProgramGroupGetStackSize,
    optixPipelineCreate,
    optixPipelineDestroy,
    optixPipelineSetStackSize,
    optixAccelComputeMemoryUsage,
    optixAccelBuild,
    optixAccelGetRelocationInfo,
    optixAccelCheckRelocationCompatibility,
    optixAccelRelocate,
    optixAccelCompact,
    optixColwertPointerToTraversableHandle,
    optixSbtRecordPackHeader,
    optixLaunch,
    optixDenoiserCreate_v1,
    optixDenoiserDestroy,
    optixDenoiserComputeMemoryResources,
    optixDenoiserSetup,
    optixDenoiserIlwoke_v1,
    optixDenoiserSetModel_v1,
    optixDenoiserComputeIntensity,
    optixDenoiserComputeAverageColor
    // clang-format on
};
}

OptixResult fillFunctionTable_43( int abi, void* functionTable, size_t sizeOfFunctionTable )
{
    if( sizeOfFunctionTable != sizeof( FunctionTable_43 ) )
    {
#if defined( OPTIX_ENABLE_LOGGING )
        lerr << "sizeOfFunctionTable != sizeof( FunctionTable_43 ) ( " << sizeOfFunctionTable << " != " << sizeof( FunctionTable_43 ) << ")\n";
#endif
        return OPTIX_ERROR_FUNCTION_TABLE_SIZE_MISMATCH;
    }

    FunctionTable_43* ftable = reinterpret_cast<FunctionTable_43*>( functionTable );
    memcpy( ftable, &g_functionTable_43, sizeOfFunctionTable );
    switch( abi )
    {
        case 39:
            ftable->optixDeviceContextCreate = OPTIX_DEVICE_CONTEXT_CREATE_NAME( 39 );
            break;
        case 40:
            ftable->optixDeviceContextCreate = OPTIX_DEVICE_CONTEXT_CREATE_NAME( 40 );
            break;
        case 41:
            ftable->optixDeviceContextCreate = OPTIX_DEVICE_CONTEXT_CREATE_NAME( 41 );
            break;
        case 42:
            ftable->optixDeviceContextCreate = OPTIX_DEVICE_CONTEXT_CREATE_NAME( 42 );
            break;
        case 43:
            ftable->optixDeviceContextCreate = OPTIX_DEVICE_CONTEXT_CREATE_NAME( 43 );
            break;
        default:
#if defined( OPTIX_ENABLE_LOGGING )
            lerr << "Unknown ABI version : " << abi << "\n";
#endif
            return OPTIX_ERROR_UNSUPPORTED_ABI_VERSION;
    }

    return OPTIX_SUCCESS;
}
}
