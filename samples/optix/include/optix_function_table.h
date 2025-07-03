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

/// @file
/// @author LWPU Corporation
/// @brief  OptiX public API header

#ifndef __optix_optix_function_table_h__
#define __optix_optix_function_table_h__

/// The OptiX ABI version.
#ifdef OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
/// When changing the ABI version make sure you know exactly what you are doing. See
/// apps/optix/exp/functionTable/functionTable.cpp for instructions. See
/// https://confluence.lwpu.com/display/RAV/ABI+Versions+in+the+Wild for released ABI versions.
#endif  // OPTIX_OPTIONAL_FEATURE_OPTIX7_INTERNAL_DOLWMENTATION
#define OPTIX_ABI_VERSION 60

#ifndef OPTIX_DEFINE_ABI_VERSION_ONLY

#include "optix_types.h"

#if !defined( OPTIX_DONT_INCLUDE_LWDA )
// If OPTIX_DONT_INCLUDE_LWDA is defined, lwca driver types must be defined through other
// means before including optix headers.
#include <lwca.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// \defgroup optix_function_table Function Table
/// \brief OptiX Function Table

/** \addtogroup optix_function_table
@{
*/

/// The function table containing all API functions.
///
/// See #optixInit() and #optixInitWithHandle().
typedef struct OptixFunctionTable
{
    /// \name Error handling
    //@ {

    /// See ::optixGetErrorName().
    const char* ( *optixGetErrorName )( OptixResult result );

    /// See ::optixGetErrorString().
    const char* ( *optixGetErrorString )( OptixResult result );

    //@ }
    /// \name Device context
    //@ {

    /// See ::optixDeviceContextCreate().
    OptixResult ( *optixDeviceContextCreate )( LWcontext fromContext, const OptixDeviceContextOptions* options, OptixDeviceContext* context );

    /// See ::optixDeviceContextDestroy().
    OptixResult ( *optixDeviceContextDestroy )( OptixDeviceContext context );

    /// See ::optixDeviceContextGetProperty().
    OptixResult ( *optixDeviceContextGetProperty )( OptixDeviceContext context, OptixDeviceProperty property, void* value, size_t sizeInBytes );

    /// See ::optixDeviceContextSetLogCallback().
    OptixResult ( *optixDeviceContextSetLogCallback )( OptixDeviceContext context,
                                                       OptixLogCallback   callbackFunction,
                                                       void*              callbackData,
                                                       unsigned int       callbackLevel );

    /// See ::optixDeviceContextSetCacheEnabled().
    OptixResult ( *optixDeviceContextSetCacheEnabled )( OptixDeviceContext context, int enabled );

    /// See ::optixDeviceContextSetCacheLocation().
    OptixResult ( *optixDeviceContextSetCacheLocation )( OptixDeviceContext context, const char* location );

    /// See ::optixDeviceContextSetCacheDatabaseSizes().
    OptixResult ( *optixDeviceContextSetCacheDatabaseSizes )( OptixDeviceContext context, size_t lowWaterMark, size_t highWaterMark );

    /// See ::optixDeviceContextGetCacheEnabled().
    OptixResult ( *optixDeviceContextGetCacheEnabled )( OptixDeviceContext context, int* enabled );

    /// See ::optixDeviceContextGetCacheLocation().
    OptixResult ( *optixDeviceContextGetCacheLocation )( OptixDeviceContext context, char* location, size_t locationSize );

    /// See ::optixDeviceContextGetCacheDatabaseSizes().
    OptixResult ( *optixDeviceContextGetCacheDatabaseSizes )( OptixDeviceContext context, size_t* lowWaterMark, size_t* highWaterMark );

    //@ }
    /// \name Modules
    //@ {

    /// See ::optixModuleCreateFromPTX().
    OptixResult ( *optixModuleCreateFromPTX )( OptixDeviceContext                 context,
                                               const OptixModuleCompileOptions*   moduleCompileOptions,
                                               const OptixPipelineCompileOptions* pipelineCompileOptions,
                                               const char*                        PTX,
                                               size_t                             PTXsize,
                                               char*                              logString,
                                               size_t*                            logStringSize,
                                               OptixModule*                       module );

    /// See ::optixModuleCreateFromPTXWithTasks().
    OptixResult ( *optixModuleCreateFromPTXWithTasks )( OptixDeviceContext                 context,
                                                        const OptixModuleCompileOptions*   moduleCompileOptions,
                                                        const OptixPipelineCompileOptions* pipelineCompileOptions,
                                                        const char*                        PTX,
                                                        size_t                             PTXsize,
                                                        char*                              logString,
                                                        size_t*                            logStringSize,
                                                        OptixModule*                       module,
                                                        OptixTask*                         firstTask );

    /// See ::optixModuleGetCompilationState().
    OptixResult ( *optixModuleGetCompilationState )( OptixModule module, OptixModuleCompileState* state );

    /// See ::optixModuleDestroy().
    OptixResult ( *optixModuleDestroy )( OptixModule module );

    /// See ::optixBuiltinISModuleGet().
    OptixResult( *optixBuiltinISModuleGet )( OptixDeviceContext                 context,
                                             const OptixModuleCompileOptions*   moduleCompileOptions,
                                             const OptixPipelineCompileOptions* pipelineCompileOptions,
                                             const OptixBuiltinISOptions*       builtinISOptions,
                                             OptixModule*                       builtinModule);

    //@ }
    /// \name Tasks
    //@ {

    /// See ::optixTaskExelwte().
    OptixResult ( *optixTaskExelwte )( OptixTask     task,
                                       OptixTask*    additionalTasks,
                                       unsigned int  maxNumAdditionalTasks,
                                       unsigned int* numAdditionalTasksCreated );
    //@ }
    /// \name Program groups
    //@ {

    /// See ::optixProgramGroupCreate().
    OptixResult ( *optixProgramGroupCreate )( OptixDeviceContext              context,
                                              const OptixProgramGroupDesc*    programDescriptions,
                                              unsigned int                    numProgramGroups,
                                              const OptixProgramGroupOptions* options,
                                              char*                           logString,
                                              size_t*                         logStringSize,
                                              OptixProgramGroup*              programGroups );

    /// See ::optixProgramGroupDestroy().
    OptixResult ( *optixProgramGroupDestroy )( OptixProgramGroup programGroup );

    /// See ::optixProgramGroupGetStackSize().
    OptixResult ( *optixProgramGroupGetStackSize )( OptixProgramGroup programGroup, OptixStackSizes* stackSizes );

    //@ }
    /// \name Pipeline
    //@ {

    /// See ::optixPipelineCreate().
    OptixResult ( *optixPipelineCreate )( OptixDeviceContext                 context,
                                          const OptixPipelineCompileOptions* pipelineCompileOptions,
                                          const OptixPipelineLinkOptions*    pipelineLinkOptions,
                                          const OptixProgramGroup*           programGroups,
                                          unsigned int                       numProgramGroups,
                                          char*                              logString,
                                          size_t*                            logStringSize,
                                          OptixPipeline*                     pipeline );

    /// See ::optixPipelineDestroy().
    OptixResult ( *optixPipelineDestroy )( OptixPipeline pipeline );

    /// See ::optixPipelineSetStackSize().
    OptixResult ( *optixPipelineSetStackSize )( OptixPipeline pipeline,
                                                unsigned int  directCallableStackSizeFromTraversal,
                                                unsigned int  directCallableStackSizeFromState,
                                                unsigned int  continuationStackSize,
                                                unsigned int  maxTraversableGraphDepth );

    //@ }
    /// \name Acceleration structures
    //@ {

    /// See ::optixAccelComputeMemoryUsage().
    OptixResult ( *optixAccelComputeMemoryUsage )( OptixDeviceContext            context,
                                                   const OptixAccelBuildOptions* accelOptions,
                                                   const OptixBuildInput*        buildInputs,
                                                   unsigned int                  numBuildInputs,
                                                   OptixAccelBufferSizes*        bufferSizes );

    /// See ::optixAccelBuild().
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

    /// See ::optixAccelGetRelocationInfo().
    OptixResult ( *optixAccelGetRelocationInfo )( OptixDeviceContext context, OptixTraversableHandle handle, OptixAccelRelocationInfo* info );


    /// See ::optixAccelCheckRelocationCompatibility().
    OptixResult ( *optixAccelCheckRelocationCompatibility )( OptixDeviceContext              context,
                                                             const OptixAccelRelocationInfo* info,
                                                             int*                            compatible );

    /// See ::optixAccelRelocate().
    OptixResult ( *optixAccelRelocate )( OptixDeviceContext              context,
                                         LWstream                        stream,
                                         const OptixAccelRelocationInfo* info,
                                         LWdeviceptr                     instanceTraversableHandles,
                                         size_t                          numInstanceTraversableHandles,
                                         LWdeviceptr                     targetAccel,
                                         size_t                          targetAccelSizeInBytes,
                                         OptixTraversableHandle*         targetHandle );


    /// See ::optixAccelCompact().
    OptixResult ( *optixAccelCompact )( OptixDeviceContext      context,
                                        LWstream                stream,
                                        OptixTraversableHandle  inputHandle,
                                        LWdeviceptr             outputBuffer,
                                        size_t                  outputBufferSizeInBytes,
                                        OptixTraversableHandle* outputHandle );

    /// See ::optixColwertPointerToTraversableHandle().
    OptixResult ( *optixColwertPointerToTraversableHandle )( OptixDeviceContext      onDevice,
                                                             LWdeviceptr             pointer,
                                                             OptixTraversableType    traversableType,
                                                             OptixTraversableHandle* traversableHandle );

#ifndef OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    /// See ::optixVisibilityMapArrayComputeMemoryUsage().
    OptixResult ( *optixVisibilityMapArrayComputeMemoryUsage )( OptixDeviceContext                       context,
                                                                const OptixVisibilityMapArrayBuildInput* buildInput,
                                                                OptixMicromeshBufferSizes*               bufferSizes );

    /// See ::optixVisibilityMapArrayBuild().
    OptixResult ( *optixVisibilityMapArrayBuild )( OptixDeviceContext                       context,
                                                   LWstream                                 stream,
                                                   const OptixVisibilityMapArrayBuildInput* buildInput,
                                                   const OptixMicromeshBuffers*             buffers,
                                                   const OptixMicromeshEmitDesc*            emittedProperties,
                                                   unsigned int                             numEmittedProperties );

    /// See ::optixDisplacedMicromeshArrayComputeMemoryUsage().
    OptixResult ( *optixDisplacedMicromeshArrayComputeMemoryUsage )( OptixDeviceContext                            context,
                                                                     const OptixDisplacedMicromeshArrayBuildInput* buildInput,
                                                                     OptixMicromeshBufferSizes*                    bufferSizes );

    /// See ::optixDisplacedMicromeshArrayBuild().
    OptixResult ( *optixDisplacedMicromeshArrayBuild )( OptixDeviceContext                            context,
                                                        LWstream                                      stream,
                                                        const OptixDisplacedMicromeshArrayBuildInput* buildInput,
                                                        const OptixMicromeshBuffers*                  buffers,
                                                        const OptixMicromeshEmitDesc*                 emittedProperties,
                                                        unsigned int numEmittedProperties );
#else // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    void ( *reserved1 )( void );
    void ( *reserved2 )( void );
#endif // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
#else  // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
    void ( *reserved1 )( void );
    void ( *reserved2 )( void );
#endif  // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD

    //@ }
    /// \name Launch
    //@ {

    /// See ::optixColwertPointerToTraversableHandle().
    OptixResult ( *optixSbtRecordPackHeader )( OptixProgramGroup programGroup, void* sbtRecordHeaderHostPointer );

    /// See ::optixColwertPointerToTraversableHandle().
    OptixResult ( *optixLaunch )( OptixPipeline                  pipeline,
                                  LWstream                       stream,
                                  LWdeviceptr                    pipelineParams,
                                  size_t                         pipelineParamsSize,
                                  const OptixShaderBindingTable* sbt,
                                  unsigned int                   width,
                                  unsigned int                   height,
                                  unsigned int                   depth );

    //@ }
    /// \name Denoiser
    //@ {

    /// See ::optixDenoiserCreate().
    OptixResult ( *optixDenoiserCreate )( OptixDeviceContext context, OptixDenoiserModelKind modelKind, const OptixDenoiserOptions* options, OptixDenoiser* returnHandle );

    /// See ::optixDenoiserDestroy().
    OptixResult ( *optixDenoiserDestroy )( OptixDenoiser handle );

    /// See ::optixDenoiserComputeMemoryResources().
    OptixResult ( *optixDenoiserComputeMemoryResources )( const OptixDenoiser handle,
                                                          unsigned int        maximumInputWidth,
                                                          unsigned int        maximumInputHeight,
                                                          OptixDenoiserSizes* returnSizes );

    /// See ::optixDenoiserSetup().
    OptixResult ( *optixDenoiserSetup )( OptixDenoiser denoiser,
                                         LWstream      stream,
                                         unsigned int  inputWidth,
                                         unsigned int  inputHeight,
                                         LWdeviceptr   state,
                                         size_t        stateSizeInBytes,
                                         LWdeviceptr   scratch,
                                         size_t        scratchSizeInBytes );

    /// See ::optixDenoiserIlwoke().
    OptixResult ( *optixDenoiserIlwoke )( OptixDenoiser                   denoiser,
                                          LWstream                        stream,
                                          const OptixDenoiserParams*      params,
                                          LWdeviceptr                     denoiserState,
                                          size_t                          denoiserStateSizeInBytes,
                                          const OptixDenoiserGuideLayer * guideLayer,
                                          const OptixDenoiserLayer *      layers,
                                          unsigned int                    numLayers,
                                          unsigned int                    inputOffsetX,
                                          unsigned int                    inputOffsetY,
                                          LWdeviceptr                     scratch,
                                          size_t                          scratchSizeInBytes );

    /// See ::optixDenoiserComputeIntensity().
    OptixResult ( *optixDenoiserComputeIntensity )( OptixDenoiser       handle,
                                                    LWstream            stream,
                                                    const OptixImage2D* inputImage,
                                                    LWdeviceptr         outputIntensity,
                                                    LWdeviceptr         scratch,
                                                    size_t              scratchSizeInBytes );

    /// See ::optixDenoiserComputeAverageColor().
    OptixResult ( *optixDenoiserComputeAverageColor )( OptixDenoiser       handle,
                                                       LWstream            stream,
                                                       const OptixImage2D* inputImage,
                                                       LWdeviceptr         outputAverageColor,
                                                       LWdeviceptr         scratch,
                                                       size_t              scratchSizeInBytes );

    /// See ::optixDenoiserCreateWithUserModel().
    OptixResult ( *optixDenoiserCreateWithUserModel )( OptixDeviceContext context, const void * data, size_t dataSizeInBytes, OptixDenoiser* returnHandle );
    //@ }

} OptixFunctionTable;

/*@}*/  // end group optix_function_table

#ifdef __cplusplus
}
#endif

#endif /* OPTIX_DEFINE_ABI_VERSION_ONLY */

#endif /* __optix_optix_function_table_h__ */
