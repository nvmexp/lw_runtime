// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once

#include <rtcore/interface/types.h>

#if defined( WIN32 )
#include <sal.h>
#endif

struct RTCetblRTCore_st;

namespace optix {

class Context;

#if defined( __GNUC__ ) && ( __GNUC__ >= 4 )
#define CHECK_RTCORE_RESULT __attribute__( ( warn_unused_result ) )
#elif defined( _MSC_VER ) && ( _MSC_VER >= 1700 )
#define CHECK_RTCORE_RESULT _Check_return_
#else
#define CHECK_RTCORE_RESULT
#endif

// Use RTCoreAPI to get return codes and no exceptions.
class RTCoreAPI
{
  public:
    static void setRtcoreLibraryVariant( bool useLibraryFromSdk );

    RTCoreAPI()  = default;
    ~RTCoreAPI() = default;

    // Not an API function.  Two phase initialization because we don't want to throw exceptions from the c'tor.
    // This function is thread safe and re-entrant.
    CHECK_RTCORE_RESULT RtcResult finishConstruction();

    // Not an API function.  Initializes rtcore once per optix library.  This function is
    // thread safe and re-entrant.  Calls finishConstruction().
    CHECK_RTCORE_RESULT RtcResult initializeRTCoreLibraryWithKnobs();

    CHECK_RTCORE_RESULT RtcResult getVersion( int* major,   /* [out] pointer to output (optional) */
                                              int* minor,   /* [out] pointer to output (optional) */
                                              int* build ); /* [out] pointer to build CL (optional) */

    CHECK_RTCORE_RESULT RtcResult rtcGetBuildUUID( Rtlw32 uuid[4] ); /* [out] pointer to output */

    CHECK_RTCORE_RESULT RtcResult init( int            debugLogLevel, /* [in] active log level in [0,100] */
                                        PFNRTCDEBUGLOG debugLogCb,    /* [in] log function callback (optional) */
                                        const char*    debugKnobs );  /* [in] debug knob overrides (optional) */

    CHECK_RTCORE_RESULT RtcResult deviceContextCreateForLWDA( LWcontext context, /* [in] the LWCA context associated with the device context */
                                                              const RtcDeviceProperties* properties, /* [in] device properties supplied by the product */
                                                              RtcDeviceContext* devctx ); /* [out] the device context to be created */

    CHECK_RTCORE_RESULT RtcResult deviceContextDestroy( RtcDeviceContext devctx ); /* [in] the device context to be destroyed */

    CHECK_RTCORE_RESULT RtcResult deviceContextGetLimit( RtcDeviceContext devctx, /* [in] the device context to query the limit for */
                                                         RtcLimit limit,   /* [in] the limit to query */
                                                         Rtlw64*  value ); /* [out] pointer to the retured value */

    CHECK_RTCORE_RESULT RtcResult deviceContextGetCompatibilityIdentifier( RtcDeviceContext devctx, /* [in] the device context to query the identifier for */
                                                                           RtcDeviceContextCompatibilityType type, /* [in] the type of compatibility queried */
                                                                           Rtlwuid* identifier ); /* [out] the device and driver identifier for the selected type */

    CHECK_RTCORE_RESULT RtcResult deviceContextCheckCompatibility( RtcDeviceContext devctx, /* [in] the device context to match the identifier against */
                                                                   RtcDeviceContextCompatibilityType type, /* [in] type of compatibility check */
                                                                   const Rtlwuid* identifier ); /* [in] the device and driver identifier for the selected type */

    CHECK_RTCORE_RESULT RtcResult commandListCreateForLWDA( RtcDeviceContext devctx, /* [in] the device context associated with the command list */
                                                            LWstream stream, /* [in] the parent LWCA stream */
                                                            RtcCommandList* cmdlist ); /* [out] the RTcore command list to be created */

    CHECK_RTCORE_RESULT RtcResult commandListDestroy( RtcCommandList cmdlist ); /* [in] the command list to be destroyed */

    CHECK_RTCORE_RESULT RtcResult compileModule( RtcDeviceContext context, /* [in] the device context the module is for */
                                                 const RtcCompileOptions* options, /* [in] options */
                                                 const char* inputSerializedModuleBuffer, /* [in] the input serialized module buffer according to the LWVM-RT spec */
                                                 Rtlw64             bufferSize, /* [in] size of serialized buffer */
                                                 RtcCompiledModule* compiledModule ); /* [out] the result module */

    CHECK_RTCORE_RESULT RtcResult compileNamedConstant( RtcDeviceContext context, /* [in] the device context the module is for */
                                                        const char* symbolName, /* [in] name of the constant */
                                                        int         nbytes,     /* [in] size in bytes of the constant */
                                                        RtcCompiledModule* compiledModule ); /* [out] the result module */

    CHECK_RTCORE_RESULT RtcResult compiledModuleGetCachedBlob( RtcCompiledModule compiledModule, /* [in] the module to serialize */
                                                               Rtlw64 bufferSize, /* [in] size in bytes of the buffer pointer to by 'blob' (0 if 'blob' is null). */
                                                               void* blob, /* [out] pointer to a destination buffer receiving the blob data (optional) */
                                                               Rtlw64* blobSize ); /* [out] amount of storage in bytes required to hold the blob data (optional) */

#if RTCORE_API_VERSION >= 25
    CHECK_RTCORE_RESULT RtcResult compiledModuleGetStackSize( RtcCompiledModule compiledModule, /* [in] the module */
                                                              Rtlw32 entryIndex, /* [in] the index of the function in the module */
                                                              Rtlw32* directStackFrameSize, /* [out] amount of storage in bytes required for the direct (ABI low level) stack */
                                                              Rtlw32* continuationStackFrameSize ); /* [out] amount of storage in bytes required for the continuation (rtcore SW level) stack */
#else
    CHECK_RTCORE_RESULT RtcResult compiledModuleGetStackSize( RtcCompiledModule compiledModule, /* [in] the module */
                                                              const char* symbolName, /* [in] the name of the function in the module */
                                                              Rtlw32* directStackFrameSize, /* [out] amount of storage in bytes required for the direct (ABI low level) stack */
                                                              Rtlw32* continuationStackFrameSize ); /* [out] amount of storage in bytes required for the continuation (rtcore SW level) stack */
#endif

    CHECK_RTCORE_RESULT RtcResult compiledModuleFromCachedBlob( RtcDeviceContext context, /* [in] the device context the module is for */
                                                                const void* blob, /* [in] the blob data to deserialize */
                                                                Rtlw64 blobSize, /* [in] the size in bytes of the buffer pointed to by 'blob' */
                                                                RtcCompiledModule* compiledModule ); /* [out] the result module */

    CHECK_RTCORE_RESULT RtcResult compiledModuleDestroy( RtcCompiledModule module ); /* [in] the module to be destroyed */

    CHECK_RTCORE_RESULT RtcResult accelComputeMemoryUsage( RtcDeviceContext context, /* [in] device context of the pipeline */
                                                           const RtcAccelOptions* accelOptions, /* [in] accel options */
                                                           unsigned int numItemArrays, /* [in] number of elements in buildInputs */
                                                           const RtcBuildInput* buildInputs, /* [in] an array of RtcBuildInput objects */
                                                           const RtcBuildInputOverrides* const* overrides, /* [in] an array of RtcBuildInputOverride objects, May be NULL, entries may be NULL */
                                                           RtcAccelBufferSizes* bufferSizes ); /* [out] fills in buffer sizes */

    CHECK_RTCORE_RESULT RtcResult accelBuild( RtcCommandList commandList, /* [in] command list in which to enqueue build kernels */
                                              const RtcAccelOptions* accelOptions, /* [in] accel options */
                                              unsigned int numItemArrays, /* [in] number of elements in buildInputs */
                                              const RtcBuildInput* buildInputs, /* [in] an array of RtcBuildInput objects */
                                              const RtcBuildInputOverrides* const* overrides, /* [in] an array of RtcBuildInputOverride objects, May be NULL, entries may be NULL */
                                              const RtcAccelBuffers* buffers, /* [in] the buffers used for build */
                                              unsigned int numEmittedProperties, /* [in] number of post-build properties to populate (may be zero) */
                                              const RtcAccelEmitDesc* emittedProperties ); /* [in/out] types of requested properties and output buffers */

    CHECK_RTCORE_RESULT RtcResult accelEmitProperties( RtcCommandList commandList,     /* [in] command list */
                                                       RtcGpuVA*      sourceAccels,    /* [in] input accels */
                                                       unsigned int   numSourceAccels, /* [in] number of elements */
                                                       RtcAccelPropertyType type, /* [in] type of information requested */
                                                       RtcGpuVA resultBuffer, /* [out] output buffer for the properties */
                                                       Rtlw64   resultBufferSize ); /* [in] size of output buffer */

    CHECK_RTCORE_RESULT RtcResult accelCopy( RtcCommandList commandList,  /* [in] command list */
                                             RtcGpuVA       sourceBuffer, /* [in] input accel */
                                             RtcCopyMode mode, /* [in] specify the output format of the copied accel */
                                             RtcGpuVA    resultBuffer,       /* [out] copied accel */
                                             Rtlw64      resultBufferSize ); /* [in] size of cloned accel */

    CHECK_RTCORE_RESULT RtcResult accelRelocate( RtcCommandList commandList,       /* [in] command list */
                                                 RtcGpuVA       traversableVAs,    /* [in] List of updated top->bottom level references for the relocated accel.
                                                                      Used for top-level accels only.
                                                                      Order and number of traversables must match the original build. */
                                                 Rtlw32         numTraversableVAs, /* [in] number of traversable VAs */
                                                 RtcGpuVA       accelBuffer,       /* [in/out] input accel */
                                                 Rtlw64         accelBufferSize ); /* [in] Optional result buffer size. Must be ~0ULL if the size is unknown. Used for validation only. */

    CHECK_RTCORE_RESULT RtcResult colwertPointerToTraversableHandle(
        RtcDeviceContext   context,         /* [in] device context */
        RtcGpuVA           pointer,         /* [in] pointer to traversalbe allocated in RtcDeviceContext */
        RtcTraversableType traversableType, /* [in] Type of RtcTraversableHandle to create */
        RtcAccelType accelType, /* [in] Type of accel if traversableType is RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL, ignored otherwise */
        RtcTraversableHandle* traversableHandle ); /* [out] traversable handle. traversableHandle must be in host memory */

    CHECK_RTCORE_RESULT RtcResult colwertTraversableHandleToPointer( RtcDeviceContext context, /* [in] device context */
                                                                     RtcTraversableHandle traversableHandle, /* [in] traversable handle. */
                                                                     RtcGpuVA* pointer, /* [out] pointer to traversalbe allocated in RtcDeviceContext */
                                                                     RtcTraversableType* traversableType, /* [out] Type of RtcTraversableHandle to create */
                                                                     RtcAccelType* accelType ); /* [out] Type of accel if traversableType is RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL, ignored otherwise */

    CHECK_RTCORE_RESULT RtcResult getSbtRecordHeaderSize( Rtlw64* nbytes ); /* [out] size in bytes of the sbt record header */

#if RTCORE_API_VERSION >= 25
    CHECK_RTCORE_RESULT RtcResult packSbtRecordHeader(
        RtcDeviceContext        context, /* [in] the device context the module is for */
        const RtcCompiledModule moduleGlobalOrCH, /* [in] the module containing the 1st entry function (any global type, or CH if packing a hit record) */
        Rtlw32                  entryFunctionIndexGlobalOrCH, /* [in] the index of the 1st entry function */
        const RtcCompiledModule moduleAH, /* [in] the module containing the any hit entry function (only if packing a hit record) */
        Rtlw32                  entryFunctionIndexAH, /* [in] the index of the any hit entry function */
        const RtcCompiledModule moduleIS, /* [in] the module containing the intersection entry function (only if packing a hit record) */
        Rtlw32                  entryFunctionIndexIS, /* [in] the index of the intersection entry function */
        void*                   sbtHeaderHostPointer );                 /* [out] the result sbt record header */
#else
    CHECK_RTCORE_RESULT RtcResult packSbtRecordHeader(
        RtcDeviceContext        context, /* [in] the device context the module is for */
        const RtcCompiledModule moduleGlobalOrCH, /* [in] the module containing the 1st entry function (any global type, or CH if packing a hit record) */
        const char*             entryFunctionNameGlobalOrCH, /* [in] the name of the 1st entry function */
        const RtcCompiledModule moduleAH, /* [in] the module containing the any hit entry function (only if packing a hit record) */
        const char*             entryFunctionNameAH, /* [in] the name of the any hit entry function */
        const RtcCompiledModule moduleIS, /* [in] the module containing the intersection entry function (only if packing a hit record) */
        const char* entryFunctionNameIS,    /* [in] the name of the intersection entry function */
        void*                   sbtHeaderHostPointer );                /* [out] the result sbt record header */
#endif
    CHECK_RTCORE_RESULT RtcResult pipelineCreate( RtcDeviceContext context, /* [in] the device context the pipeline is for */
                                                  const RtcPipelineOptions* pipelineOptions, /* [in] pipeline options */
                                                  const RtcCompileOptions*  compileOptions,  /* [in] compile options */
                                                  const RtcCompiledModule* modules, /* [in] the list of modules to create a pipeline from */
                                                  int                      moduleCount, /* [in] number of modules */
                                                  RtcPipeline* pipeline );              /* [out] the result pipeline */

    CHECK_RTCORE_RESULT RtcResult pipelineDestroy( RtcPipeline pipeline ); /* [in] the pipeline to be destroyed */

    CHECK_RTCORE_RESULT RtcResult pipelineGetInfo( RtcPipeline pipeline, /* [in] pipeline to query information for */
                                                   RtcPipelineInfoType type, /* [in] what type of information to query about the pipeline */
                                                   Rtlw64 dataSize, /* [in] size of the data structure receiving the information */
                                                   void* data ); /* [out] type-specific struct containing the queried information */

    CHECK_RTCORE_RESULT RtcResult pipelineGetLaunchBufferInfo( RtcPipeline pipeline, /* [in] pipeline to query launch buffer information for */
                                                               Rtlw64* nbytes, /* [out] launch buffer size requirement */
                                                               Rtlw64* align ); /* [out] launch buffer alignment requirement */

    CHECK_RTCORE_RESULT RtcResult pipelineGetNamedConstantInfo( RtcPipeline pipeline, /* [in] the pipeline to query the info for */
                                                                const char* symbolName, /* [in] name of the constant */
                                                                Rtlw64* offset, /* [out] offset relative to launch buffer start */
                                                                Rtlw64* nbytes ); /* [out] size of the constant */

    CHECK_RTCORE_RESULT RtcResult pipelineGetScratchBufferInfo3D( RtcPipeline pipeline, /* [in] pipeline to query launch buffer information for */
                                                                  RtcS64 width, /* [in] number of elements to compute, must match rtcLaunch parameter */
                                                                  RtcS64 height, /* [in] number of elements to compute */
                                                                  RtcS64 depth, /* [in] number of elements to compute */
                                                                  Rtlw64* nbytesMin, /* [out] minimum scratch buffer size requirement */
                                                                  Rtlw64* nbytes, /* [out] requested scratch buffer size for efficient exelwtion */
                                                                  Rtlw64* align ); /* [out] scratch buffer alignment requirement */

    CHECK_RTCORE_RESULT RtcResult pipelineGetStackSize(
        RtcPipeline pipeline, /* [in] pipeline to query stack size information  */
        Rtlw32*     directCallableStackSizeFromTraversal, /* [out] size in bytes of direct (ABI level) stack required for callables from IS/AH shaders */
        Rtlw32*     directCallableStackSizeFromState, /* [out] size in bytes of direct (ABI level) stack required for callables from RG/CH/MS shaders */
        Rtlw32* continuationStackSize, /* [out] size in bytes of continuation (rtcore SW level) stack required for rtcLaunch */
        Rtlw32* maxTraversableGraphDepth ); /* [out] Maximum depth of a traversable graph passed to trace. 0 means the default of 2 */

    CHECK_RTCORE_RESULT RtcResult pipelineSetStackSize(
        RtcPipeline pipeline, /* [in] pipeline to set the stack size */
        Rtlw32      directCallableStackSizeFromTraversal, /* [in] size in bytes of direct (ABI level) stack required for callables from IS/AH shaders */
        Rtlw32      directCallableStackSizeFromState, /* [in] size in bytes of direct (ABI level) stack required for callables from RG/CH/MS shaders */
        Rtlw32 continuationStackSize, /* [in} size in bytes of continuation (rtcore SW level) stack required for rtcLaunch */
        Rtlw32 maxTraversableGraphDepth ); /* [in] Maximum depth of a traversable graph passed to trace. 0 means the default of 2 */

    CHECK_RTCORE_RESULT RtcResult
    launch3D( RtcCommandList cmdlist,         /* [in] the command list to enqueue the launch into */
              RtcPipeline    pipeline,        /* [in] the pipeline to be launched */
              RtcGpuVA       launchBufferVA,  /* [in] device address of the launch buffer for this launch */
              RtcGpuVA       scratchBufferVA, /* [in] device address of the scratch buffer for this launch */
              RtcGpuVA raygenSbtRecordVA, /* [in] device address of the SBT record of the ray gen program to start at */
              RtcGpuVA exceptionSbtRecordVA, /* [in] device address of the SBT record of the exception shader */
              RtcGpuVA firstMissSbtRecordVA, /* [in] device address of the SBT record of the first miss program */
              Rtlw32   missSbtRecordSize,    /* [in] size of a single SBT record in bytes */
              Rtlw32   missSbtRecordCount,   /* [in] size of SBT in records, a size of 0 means the size is unknown */
              RtcGpuVA firstInstanceSbtRecordVA, /* [in] device address of the SBT record of the first hit program */
              Rtlw32   instanceSbtRecordSize,    /* [in] size of a single SBT record in bytes */
              Rtlw32   instanceSbtRecordCount, /* [in] size of SBT in records, a size of 0 means the size is unknown */
              RtcGpuVA firstCallableSbtRecordVA, /* [in] device address of the SBT record of the first callable program */
              Rtlw32   callableSbtRecordSize,    /* [in] size of a single SBT record in bytes */
              Rtlw32   callableSbtRecordCount, /* [in] size of SBT in records, a size of 0 means the size is unknown */
              RtcGpuVA toolsOutputVA, /* [in] device address where exception and profiling information shoud be written */
              Rtlw64   toolsOutputSize,          /* [in] size of the tools output buffer */
              Rtlw64   scratchBufferSizeInBytes, /* [in] size of the scratch buffer in bytes */
              RtcS64   width,                    /* [in] number of elements to compute */
              RtcS64   height,                   /* [in] number of elements to compute */
              RtcS64   depth );                  /* [in] number of elements to compute */


#if RTCORE_API_VERSION >= 25
    CHECK_RTCORE_RESULT RtcResult compiledModuleGetHash( RtcCompiledModule compiledModule, /* [in] the module */
                                                         Rtlw64* hash ); /* [out] the hash value from the module SASS binary */

    CHECK_RTCORE_RESULT RtcResult compiledModuleGetEntryFunctionIndex( RtcCompiledModule module, /* [in] the  module */
                                                                       const char* entryFunctionName, /* [in] the entry function name in the module */
                                                                       Rtlw32* entryFunctionIndex ); /* [out] the index of the function in the module */

    CHECK_RTCORE_RESULT RtcResult compiledModuleGetEntryFunctionName( RtcCompiledModule  module,                 /* [in] the  module */
        Rtlw32             entryFunctionIndex,     /* [in] the index of the function in the module to get name (optional) */
        Rtlw32             nameBufferSize,         /* [in] size in bytes of the buffer pointer to by 'nameBuffer' (0 if 'nameBuffer' is null)(optional) */
        char*              nameBuffer,             /* [out] the entry function name (0 terminated) in the module at the index (optional) */
        Rtlw32*            entryFunctionNameSize );/* [out] the size in bytes of entry function name (including \0 termination)  in the module at the index (optional) */

    CHECK_RTCORE_RESULT RtcResult compiledModuleGetEntryFunctionName( RtcCompiledModule  module, /* [in] the  module */
        Rtlw32*            entryFunctionCount );   /* [out] the number of entry functions in the module */
#endif

#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
#if RTCORE_API_VERSION >= 31
    CHECK_RTCORE_RESULT RtcResult visibilityMapArrayComputeMemoryUsage( RtcDeviceContext context, /* [in] device context of the pipeline */
                                                                        const RtcVisibilityMapArrayBuildInput* buildInputs, /* [in] a build input object containing an array of input objects */
                                                                        RtcMicromeshBufferSizes* bufferSizes ); /* [out] fills in buffer sizes */

    CHECK_RTCORE_RESULT RtcResult visibilityMapArrayBuild(
        RtcCommandList                         commandList, /* [in] command list in which to enqueue build kernels */
        const RtcVisibilityMapArrayBuildInput* buildInput,  /* [in] a build input object containing an array of input objects */
        const RtcMicromeshBuffers*             buffers,     /* [in] the buffers used for build */
        unsigned int numEmittedProperties, /* [in] number of post-build properties to populate (may be zero) */
        const RtcMicromeshEmitDesc* emittedProperties ); /* [in/out] types of requested properties and output buffers */

    CHECK_RTCORE_RESULT RtcResult displacedMicromeshArrayComputeMemoryUsage( RtcDeviceContext context, /* [in] device context of the pipeline */
                                                                             const RtcDisplacedMicromeshArrayBuildInput* buildInputs, /* [in] a build input object containing an array of input objects */
                                                                             RtcMicromeshBufferSizes* bufferSizes ); /* [out] fills in buffer sizes */

    CHECK_RTCORE_RESULT RtcResult displacedMicromeshArrayBuild(
        RtcCommandList commandList, /* [in] command list in which to enqueue build kernels */
        const RtcDisplacedMicromeshArrayBuildInput* buildInput, /* [in] a build input object containing an array of input objects */
        const RtcMicromeshBuffers*                  buffers, /* [in] the buffers used for build */
        unsigned int numEmittedProperties, /* [in] number of post-build properties to populate (may be zero) */
        const RtcMicromeshEmitDesc* emittedProperties ); /* [in/out] types of requested properties and output buffers */

    CHECK_RTCORE_RESULT RtcResult micromeshEmitProperties( RtcCommandList commandList, /* [in] command list */
                                                           const RtcGpuVA* sourceMmArrays, /* [in] input micromesh arrays */
                                                           unsigned int numSourceMmArrays, /* [in] number of elements */
                                                           RtcMicromeshPropertyType type, /* [in] type of information requested */
                                                           RtcGpuVA resultBuffer, /* [out] output buffer for the properties */
                                                           Rtlw64 resultBufferSize ); /* [in] Optional result buffer size. Must be ~0ULL if the size is unknown. */

    CHECK_RTCORE_RESULT RtcResult micromeshCopy( RtcCommandList commandList, /* [in] command list */
                                                 RtcGpuVA sourceBuffer, /* [in] input micromesh array */
                                                 RtcMicromeshCopyMode mode, /* [in] specify the output format of the copied micromesh array */
                                                 RtcGpuVA resultBuffer, /* [out] copied micromesh array */
                                                 Rtlw64   resultBufferSize ); /* [in] optional result buffer size. Must be ~0ULL if the size is unknown. Used for validation only. */

    CHECK_RTCORE_RESULT RtcResult micromeshRelocate( RtcCommandList commandList, /* [in] command list */
                                                     RtcGpuVA mmArrayBuffer, /* [in/out] micromesh array to relocate. */
                                                     Rtlw64   mmArrayBufferSize ); /* [in] optional result buffer size. Must be ~0ULL if the size is unknown. Used for validation only. */

#endif  // RTCORE_API_VERSION >= 31
#endif  // LWCFG( GLOBAL_FEATURE_GR1354_MICROMESH )

  private:
    static bool       m_useLibraryFromSdk;
    RTCetblRTCore_st* m_exports = nullptr;
};

// Use RTCore to get optional return codes and exceptions on errors otherwise.
class RTCore
{
  public:
    static void setRtcoreLibraryVariant( bool useLibraryFromSdk );

    RTCore();
    ~RTCore() = default;

    // Not an API function.  Initializes rtcore once per optix library.  This function is
    // thread safe and re-entrant.
    void initializeRTCoreLibraryWithKnobs( RtcResult* returnResult = nullptr );

    void getVersion( int*       major, /* [out] pointer to output (optional) */
                     int*       minor, /* [out] pointer to output (optional) */
                     int*       build, /* [out] pointer to build CL (optional) */
                     RtcResult* returnResult = nullptr );

    void rtcGetBuildUUID( Rtlw32     uuid[4], /* [out] pointer to output */
                          RtcResult* returnResult = nullptr );

    void init( int            debugLogLevel, /* [in] active log level in [0,100] */
               PFNRTCDEBUGLOG debugLogCb,    /* [in] log function callback (optional) */
               const char*    debugKnobs,    /* [in] debug knob overrides (optional) */
               RtcResult*     returnResult = nullptr );


    void deviceContextCreateForLWDA( LWcontext context, /* [in] the LWCA context associated with the device context */
                                     const RtcDeviceProperties* properties, /* [in] device properties supplied by the product */
                                     RtcDeviceContext*          devctx, /* [out] the device context to be created */
                                     RtcResult*                 returnResult = nullptr );

    void deviceContextDestroy( RtcDeviceContext devctx, /* [in] the device context to be destroyed */
                               RtcResult*       returnResult = nullptr );

    void deviceContextGetLimit( RtcDeviceContext devctx, /* [in] the device context to query the limit for */
                                RtcLimit         limit,  /* [in] the limit to query */
                                Rtlw64*          value,  /* [out] pointer to the retured value */
                                RtcResult*       returnResult = nullptr );


    void deviceContextGetCompatibilityIdentifier( RtcDeviceContext devctx, /* [in] the device context to query the identifier for */
                                                  RtcDeviceContextCompatibilityType type, /* [in] the type of compatibility queried */
                                                  Rtlwuid* identifier, /* [out] the device and driver identifier for the selected type */
                                                  RtcResult* returnResult = nullptr );

    void deviceContextCheckCompatibility( RtcDeviceContext devctx, /* [in] the device context to match the identifier against */
                                          RtcDeviceContextCompatibilityType type, /* [in] type of compatibility check */
                                          const Rtlwuid* identifier, /* [in] the device and driver identifier for the selected type */
                                          RtcResult* returnResult = nullptr );

    void commandListCreateForLWDA( RtcDeviceContext devctx, /* [in] the device context associated with the command list */
                                   LWstream         stream, /* [in] the parent LWCA stream */
                                   RtcCommandList* cmdlist, /* [out] the RTcore command list to be created */
                                   RtcResult*      returnResult = nullptr );

    void commandListDestroy( RtcCommandList cmdlist, /* [in] the command list to be destroyed */
                             RtcResult*     returnResult = nullptr );

    void compileModule( RtcDeviceContext         context, /* [in] the device context the module is for */
                        const RtcCompileOptions* options, /* [in] options */
                        const char* inputSerializedModuleBuffer, /* [in] the input serialized module buffer according to the LWVM-RT spec */
                        Rtlw64             bufferSize,     /* [in] size of serialized buffer */
                        RtcCompiledModule* compiledModule, /* [out] the result module */
                        RtcResult*         returnResult = nullptr );

    void compileNamedConstant( RtcDeviceContext   context,        /* [in] the device context the module is for */
                               const char*        symbolName,     /* [in] name of the constant */
                               int                nbytes,         /* [in] size in bytes of the constant */
                               RtcCompiledModule* compiledModule, /* [out] the result module */
                               RtcResult*         returnResult = nullptr );

    void compiledModuleGetCachedBlob( RtcCompiledModule compiledModule, /* [in] the module to serialize */
                                      Rtlw64 bufferSize, /* [in] size in bytes of the buffer pointer to by 'blob' (0 if 'blob' is null). */
                                      void* blob, /* [out] pointer to a destination buffer receiving the blob data (optional) */
                                      Rtlw64* blobSize, /* [out] amount of storage in bytes required to hold the blob data (optional) */
                                      RtcResult* returnResult = nullptr );

#if RTCORE_API_VERSION >= 25
    void compiledModuleGetStackSize( RtcCompiledModule compiledModule, /* [in] the module */
                                     Rtlw32            index, /* [in] the index of the function in the module */
                                     Rtlw32* directStackFrameSize, /* [out] amount of storage in bytes required for the direct (ABI low level) stack */
                                     Rtlw32* continuationStackFrameSize, /* [out] amount of storage in bytes required for the continuation (rtcore SW level) stack */
                                     RtcResult* returnResult = nullptr );
#else
    void compiledModuleGetStackSize( RtcCompiledModule compiledModule, /* [in] the module */
                                     const char*       symbolName,     /* [in] the name of the function in the module */
                                     Rtlw32* directStackFrameSize, /* [out] amount of storage in bytes required for the direct (ABI low level) stack */
                                     Rtlw32* continuationStackFrameSize, /* [out] amount of storage in bytes required for the continuation (rtcore SW level) stack */
                                     RtcResult* returnResult = nullptr );
#endif

    void compiledModuleFromCachedBlob( RtcDeviceContext context, /* [in] the device context the module is for */
                                       const void*      blob,    /* [in] the blob data to deserialize */
                                       Rtlw64 blobSize, /* [in] the size in bytes of the buffer pointed to by 'blob' */
                                       RtcCompiledModule* compiledModule, /* [out] the result module */
                                       RtcResult*         returnResult = nullptr );

    void compiledModuleDestroy( RtcCompiledModule module, /* [in] the module to be destroyed */
                                RtcResult*        returnResult = nullptr );

    void accelComputeMemoryUsage( RtcDeviceContext       context,       /* [in] device context of the pipeline */
                                  const RtcAccelOptions* accelOptions,  /* [in] accel options */
                                  unsigned int           numItemArrays, /* [in] number of elements in buildInputs */
                                  const RtcBuildInput*   buildInputs,   /* [in] an array of RtcBuildInput objects */
                                  const RtcBuildInputOverrides* const* overrides, /* [in] an array of RtcBuildInputOverride objects, May be NULL, entries may be NULL */
                                  RtcAccelBufferSizes* bufferSizes, /* [out] fills in buffer sizes */
                                  RtcResult*           returnResult = nullptr );

    void accelBuild( RtcCommandList         commandList,   /* [in] command list in which to enqueue build kernels */
                     const RtcAccelOptions* accelOptions,  /* [in] accel options */
                     unsigned int           numItemArrays, /* [in] number of elements in buildInputs */
                     const RtcBuildInput*   buildInputs,   /* [in] an array of RtcBuildInput objects */
                     const RtcBuildInputOverrides* const* overrides, /* [in] an array of RtcBuildInputOverride objects, May be NULL, entries may be NULL */
                     const RtcAccelBuffers* buffers, /* [in] the buffers used for build */
                     unsigned int numEmittedProperties, /* [in] number of post-build properties to populate (may be zero) */
                     const RtcAccelEmitDesc* emittedProperties, /* [in/out] types of requested properties and output buffers */
                     RtcResult*              returnResult = nullptr );

    void accelEmitProperties( RtcCommandList       commandList,      /* [in] command list */
                              RtcGpuVA*            sourceAccels,     /* [in] input accels */
                              unsigned int         numSourceAccels,  /* [in] number of elements */
                              RtcAccelPropertyType type,             /* [in] type of information requested */
                              RtcGpuVA             resultBuffer,     /* [out] output buffer for the properties */
                              Rtlw64               resultBufferSize, /* [in] size of output buffer */
                              RtcResult*           returnResult = nullptr );

    void accelCopy( RtcCommandList commandList,      /* [in] command list */
                    RtcGpuVA       sourceBuffer,     /* [in] input accel */
                    RtcCopyMode    mode,             /* [in] specify the output format of the copied accel */
                    RtcGpuVA       resultBuffer,     /* [out] copied accel */
                    Rtlw64         resultBufferSize, /* [in] size of cloned accel */
                    RtcResult*     returnResult = nullptr );

    void accelRelocate( RtcCommandList commandList,       /* [in] command list */
                        RtcGpuVA       traversableVAs,    /* [in] List of updated top->bottom level references for the relocated accel.
                                                                      Used for top-level accels only.
                                                                      Order and number of traversables must match the original build. */
                        Rtlw32         numTraversableVAs, /* [in] number of traversable VAs */
                        RtcGpuVA       accelBuffer,       /* [in/out] input accel */
                        Rtlw64 accelBufferSize, /* [in] Optional result buffer size. Must be ~0ULL if the size is unknown. Used for validation only. */
                        RtcResult* returnResult = nullptr );

    void colwertPointerToTraversableHandle( RtcDeviceContext context, /* [in] device context */
                                            RtcGpuVA pointer, /* [in] pointer to traversalbe allocated in RtcDeviceContext */
                                            RtcTraversableType traversableType, /* [in] Type of RtcTraversableHandle to create */
                                            RtcAccelType accelType, /* [in] Type of accel if traversableType is RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL, ignored otherwise */
                                            RtcTraversableHandle* traversableHandle, /* [out] traversable handle. traversableHandle must be in host memory */
                                            RtcResult* returnResult = nullptr );

    void colwertTraversableHandleToPointer( RtcDeviceContext     context,           /* [in] device context */
                                            RtcTraversableHandle traversableHandle, /* [ouint] traversable handle. */
                                            RtcGpuVA* pointer, /* [out] pointer to traversalbe allocated in RtcDeviceContext */
                                            RtcTraversableType* traversableType, /* [out] Type of RtcTraversableHandle to create */
                                            RtcAccelType* accelType, /* [ioutn] Type of accel if traversableType is RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL, ignored otherwise */
                                            RtcResult* returnResult = nullptr );

    void getSbtRecordHeaderSize( Rtlw64*    nbytes, /* [out] size in bytes of the sbt record header */
                                 RtcResult* returnResult = nullptr );

#if RTCORE_API_VERSION >= 25
    void packSbtRecordHeader( RtcDeviceContext        context, /* [in] the device context the module is for */
                              const RtcCompiledModule moduleGlobalOrCH, /* [in] the module containing the 1st entry function (any global type, or CH if packing a hit record) */
                              Rtlw32 entryFunctionIndexGlobalOrCH, /* [in] the index of the 1st entry function */
                              const RtcCompiledModule moduleAH, /* [in] the module containing the any hit entry function (only if packing a hit record) */
                              Rtlw32 entryFunctionIndexAH, /* [in] the index of the any hit entry function */
                              const RtcCompiledModule moduleIS, /* [in] the module containing the intersection entry function (only if packing a hit record) */
                              Rtlw32     entryFunctionIndexIS, /* [in] the index of the intersection entry function */
                              void*      sbtHeaderHostPointer, /* [out] the result sbt record header */
                              RtcResult* returnResult = nullptr );
#else
    void packSbtRecordHeader( RtcDeviceContext        context, /* [in] the device context the module is for */
                              const RtcCompiledModule moduleGlobalOrCH, /* [in] the module containing the 1st entry function (any global type, or CH if packing a hit record) */
                              const char* entryFunctionNameGlobalOrCH, /* [in] the name of the 1st entry function */
                              const RtcCompiledModule moduleAH, /* [in] the module containing the any hit entry function (only if packing a hit record) */
                              const char* entryFunctionNameAH, /* [in] the name of the any hit entry function */
                              const RtcCompiledModule moduleIS, /* [in] the module containing the intersection entry function (only if packing a hit record) */
                              const char* entryFunctionNameIS,  /* [in] the name of the intersection entry function */
                              void*       sbtHeaderHostPointer, /* [out] the result sbt record header */
                              RtcResult*  returnResult = nullptr );
#endif

    void pipelineCreate( RtcDeviceContext          context,         /* [in] the device context the pipeline is for */
                         const RtcPipelineOptions* pipelineOptions, /* [in] pipeline options */
                         const RtcCompileOptions*  compileOptions,  /* [in] compile options */
                         const RtcCompiledModule*  modules,     /* [in] the list of modules to create a pipeline from */
                         int                       moduleCount, /* [in] number of modules */
                         RtcPipeline*              pipeline,    /* [out] the result pipeline */
                         RtcResult*                returnResult = nullptr );

    void pipelineDestroy( RtcPipeline pipeline, /* [in] the pipeline to be destroyed */
                          RtcResult*  returnResult = nullptr );

    void pipelineGetInfo( RtcPipeline         pipeline, /* [in] pipeline to query information for */
                          RtcPipelineInfoType type,     /* [in] what type of information to query about the pipeline */
                          Rtlw64              dataSize, /* [in] size of the data structure receiving the information */
                          void*               data, /* [out] type-specific struct containing the queried information */
                          RtcResult*          returnResult = nullptr );

    void pipelineGetLaunchBufferInfo( RtcPipeline pipeline, /* [in] pipeline to query launch buffer information for */
                                      Rtlw64*     nbytes,   /* [out] launch buffer size requirement */
                                      Rtlw64*     align,    /* [out] launch buffer alignment requirement */
                                      RtcResult*  returnResult = nullptr );

    void pipelineGetNamedConstantInfo( RtcPipeline pipeline,   /* [in] the pipeline to query the info for */
                                       const char* symbolName, /* [in] name of the constant */
                                       Rtlw64*     offset,     /* [out] offset relative to launch buffer start */
                                       Rtlw64*     nbytes,     /* [out] size of the constant */
                                       RtcResult*  returnResult = nullptr );

    void pipelineGetScratchBufferInfo3D( RtcPipeline pipeline, /* [in] pipeline to query launch buffer information for */
                                         RtcS64 width, /* [in] number of elements to compute, must match rtcLaunch parameter */
                                         RtcS64  height,    /* [in] number of elements to compute */
                                         RtcS64  depth,     /* [in] number of elements to compute */
                                         Rtlw64* nbytesMin, /* [out] minimum scratch buffer size requirement */
                                         Rtlw64* nbytes, /* [out] requested scratch buffer size for efficient exelwtion */
                                         Rtlw64*    align, /* [out] scratch buffer alignment requirement */
                                         RtcResult* returnResult = nullptr );

    void pipelineGetStackSize( RtcPipeline pipeline, /* [in] pipeline to query stack size information  */
                               Rtlw32*     directCallableStackSizeFromTraversal, /* [out] size in bytes of direct (ABI level) stack required for callables from IS/AH shaders */
                               Rtlw32*     directCallableStackSizeFromState, /* [out] size in bytes of direct (ABI level) stack required for callables from RG/CH/MS shaders */
                               Rtlw32* continuationStackSize, /* [out] size in bytes of continuation (rtcore SW level) stack required for rtcLaunch */
                               Rtlw32* maxTraversableGraphDepth, /* [out] Maximum depth of a traversable graph passed to trace. 0 means the default of 2 */
                               RtcResult* returnResult = nullptr );

    void pipelineSetStackSize( RtcPipeline pipeline, /* [in] pipeline to set the stack size */
                               Rtlw32      directCallableStackSizeFromTraversal, /* [in] size in bytes of direct (ABI level) stack required for callables from IS/AH shaders */
                               Rtlw32      directCallableStackSizeFromState, /* [in] size in bytes of direct (ABI level) stack required for callables from RG/CH/MS shaders */
                               Rtlw32 continuationStackSize, /* [in} size in bytes of continuation (rtcore SW level) stack required for rtcLaunch */
                               Rtlw32 maxTraversableGraphDepth, /* [in] Maximum depth of a traversable graph passed to trace. 0 means the default of 2 */
                               RtcResult* returnResult = nullptr );

    void launch3D( RtcCommandList cmdlist,         /* [in] the command list to enqueue the launch into */
                   RtcPipeline    pipeline,        /* [in] the pipeline to be launched */
                   RtcGpuVA       launchBufferVA,  /* [in] device address of the launch buffer for this launch */
                   RtcGpuVA       scratchBufferVA, /* [in] device address of the scratch buffer for this launch */
                   RtcGpuVA raygenSbtRecordVA, /* [in] device address of the SBT record of the ray gen program to start at */
                   RtcGpuVA exceptionSbtRecordVA, /* [in] device address of the SBT record of the exception shader */
                   RtcGpuVA firstMissSbtRecordVA, /* [in] device address of the SBT record of the first miss program */
                   Rtlw32   missSbtRecordSize,    /* [in] size of a single SBT record in bytes */
                   Rtlw32   missSbtRecordCount, /* [in] size of SBT in records, a size of 0 means the size is unknown */
                   RtcGpuVA firstInstanceSbtRecordVA, /* [in] device address of the SBT record of the first hit program */
                   Rtlw32   instanceSbtRecordSize,    /* [in] size of a single SBT record in bytes */
                   Rtlw32 instanceSbtRecordCount, /* [in] size of SBT in records, a size of 0 means the size is unknown */
                   RtcGpuVA firstCallableSbtRecordVA, /* [in] device address of the SBT record of the first callable program */
                   Rtlw32 callableSbtRecordSize, /* [in] size of a single SBT record in bytes */
                   Rtlw32 callableSbtRecordCount, /* [in] size of SBT in records, a size of 0 means the size is unknown */
                   RtcGpuVA toolsOutputVA, /* [in] device address where exception and profiling information shoud be written */
                   Rtlw64     toolsOutputSize,          /* [in] size of the tools output buffer */
                   Rtlw64     scratchBufferSizeInBytes, /* [in] size of the scratch buffer in bytes */
                   RtcS64     width,                    /* [in] number of elements to compute */
                   RtcS64     height,                   /* [in] number of elements to compute */
                   RtcS64     depth,                    /* [in] number of elements to compute */
                   RtcResult* returnResult = nullptr );

#if RTCORE_API_VERSION >= 25
    void compiledModuleGetHash( RtcCompiledModule compiledModule, /* [in] the module */
                                Rtlw64*           hash,           /* [out] the hash value from the module SASS binary */
                                RtcResult*        returnResult = nullptr );

    void compiledModuleGetEntryFunctionIndex( RtcCompiledModule module,      /* [in] the  module */
                                              const char* entryFunctionName, /* [in] the entry function name in the module */
                                              Rtlw32* entryFunctionIndex, /* [out] the index of the function in the module */
                                              RtcResult* returnResult = nullptr );

    void compiledModuleGetEntryFunctionName( RtcCompiledModule module, /* [in] the  module */
                                             Rtlw32 entryFunctionIndex, /* [in] the index of the function in the module to get name (optional) */
                                             Rtlw32 nameBufferSize, /* [in] size in bytes of the buffer pointer to by 'nameBuffer' (0 if 'nameBuffer' is null)(optional) */
                                             char* nameBuffer, /* [out] the entry function name (0 terminated) in the module at the index (optional) */
                                             Rtlw32*    entryFunctionNameSize, /* [out] the size in bytes of entry function name (including \0 termination)  in the module at the index (optional) */
                                             RtcResult* returnResult = nullptr );
#endif
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
#if RTCORE_API_VERSION >= 31
    void visibilityMapArrayComputeMemoryUsage( RtcDeviceContext context, /* [in] device context of the pipeline */
                                               const RtcVisibilityMapArrayBuildInput* buildInput, /* [in] a build input object containing an array of input objects */
                                               RtcMicromeshBufferSizes* bufferSizes, /* [out] fills in buffer sizes */
                                               RtcResult*               returnResult = nullptr );

    void visibilityMapArrayBuild( RtcCommandList commandList, /* [in] command list in which to enqueue build kernels */
                                  const RtcVisibilityMapArrayBuildInput* buildInput, /* [in] a build input object containing an array of input objects */
                                  const RtcMicromeshBuffers* buffers, /* [in] the buffers used for build */
                                  unsigned int numEmittedProperties, /* [in] number of post-build properties to populate (may be zero) */
                                  const RtcMicromeshEmitDesc* emittedProperties, /* [in/out] types of requested properties and output buffers */
                                  RtcResult* returnResult = nullptr );

    void displacedMicromeshArrayComputeMemoryUsage( RtcDeviceContext context, /* [in] device context of the pipeline */
                                                    const RtcDisplacedMicromeshArrayBuildInput* buildInput, /* [in] a build input object containing an array of input objects */
                                                    RtcMicromeshBufferSizes* bufferSizes, /* [out] fills in buffer sizes */
                                                    RtcResult*               returnResult = nullptr );

    void displacedMicromeshArrayBuild( RtcCommandList commandList, /* [in] command list in which to enqueue build kernels */
                                       const RtcDisplacedMicromeshArrayBuildInput* buildInput, /* [in] a build input object containing an array of input objects */
                                       const RtcMicromeshBuffers* buffers, /* [in] the buffers used for build */
                                       unsigned int numEmittedProperties, /* [in] number of post-build properties to populate (may be zero) */
                                       const RtcMicromeshEmitDesc* emittedProperties, /* [in/out] types of requested properties and output buffers */
                                       RtcResult* returnResult = nullptr );

    void micromeshEmitProperties( RtcCommandList           commandList,       /* [in] command list */
                                  const RtcGpuVA*          sourceMmArrays,    /* [in] input micromesh arrays */
                                  unsigned int             numSourceMmArrays, /* [in] number of elements */
                                  RtcMicromeshPropertyType type,              /* [in] type of information requested */
                                  RtcGpuVA                 resultBuffer, /* [out] output buffer for the properties */
                                  Rtlw64 resultBufferSize, /* [in] Optional result buffer size. Must be ~0ULL if the size is unknown. */
                                  RtcResult* returnResult = nullptr );

    void micromeshCopy( RtcCommandList       commandList,  /* [in] command list */
                        RtcGpuVA             sourceBuffer, /* [in] input micromesh array */
                        RtcMicromeshCopyMode mode, /* [in] specify the output format of the copied micromesh array */
                        RtcGpuVA             resultBuffer, /* [out] copied micromesh array */
                        Rtlw64               resultBufferSize, /* [in] optional result buffer size. Must be ~0ULL if the size is unknown. Used for validation only. */
                        RtcResult*           returnResult = nullptr );

    void micromeshRelocate( RtcCommandList commandList,   /* [in] command list */
                            RtcGpuVA       mmArrayBuffer, /* [in/out] micromesh array to relocate. */
                            Rtlw64         mmArrayBufferSize, /* [in] optional result buffer size. Must be ~0ULL if the size is unknown. Used for validation only. */
                            RtcResult*     returnResult = nullptr );

#endif  // RTCORE_API_VERSION >= 31
#endif  // LWCFG( GLOBAL_FEATURE_GR1354_MICROMESH )

  private:
    RTCoreAPI m_api;
};

}  // namespace optix
