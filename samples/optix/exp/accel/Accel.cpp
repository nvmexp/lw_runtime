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

#include <optix_types.h>

#include <rtcore/interface/rtcore.h>
#include <rtcore/interface/types.h>

#include <exp/accel/InstanceAabbComputer.h>
#include <exp/accel/RtcAccelBuilder.h>
#include <exp/accel/RtcVmBuilder.h>
#include <exp/accel/RtcDmmBuilder.h>
#include <exp/builtinIS/LwrveAabb.h>
#include <exp/context/DeviceContext.h>
#include <exp/context/ErrorHandling.h>
#include <exp/functionTable/deviceTypeTranslate.h>

#include <corelib/misc/String.h>
#include <corelib/system/LwdaDriver.h>
#include <corelib/system/Preprocessor.h>
#include <prodlib/misc/LWTXProfiler.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/misc/LwdaStopwatch.h>
#include <prodlib/misc/HostStopwatch.h>

#include <src/LWCA/Memory.h>

#include <Util/ContainerAlgorithm.h>

#include <lwca.h>

#include <cmath>
#include <vector>
#include <algorithm>

#if defined( DEBUG ) || defined( DEVELOP )
#define OPTIX_ENABLE_REGISTER_AS
#endif

#if defined( DEBUG ) || ( defined( DEVELOP ) && defined( ENABLE_DEVELOP_ASSERTS ) )
#define OPTIX_ENABLE_AS_ASSERTS
#endif

namespace
{
    Knob<bool>         k_enableInstanceAabbComputer( RT_DSTRING( "o7.accel.enableInstanceAabbComputer" ), false, RT_DSTRING( "Enables the instance aabb computer extension. Otherwise it is a near no-op." ) );
    Knob<bool>         k_dumpBvhs( RT_DSTRING( "o7.accel.dumpBvhs" ), false, RT_DSTRING( "Enable to dump BVHs to disk." ) );
    Knob<std::string>  k_bvhDumpPath( RT_DSTRING( "o7.accel.dumpBvhPath" ), ".", RT_DSTRING( "Output path for BVH dumps. Defaults to ./" ) );
}

namespace optix_exp {

// OptixAccelRelocationInfo is lwrrently 32 bytes and 8 byte aligned.
//
// We need to keep track of the RtcTraversableType and RtcAccelType in order to construct
// the correct kind of traversable handle after relocation.
struct RelocationInfo
{
    Rtlwuid            rtcId;            // Bytes [ 0-15]
    RtcTraversableType traversableType;  // Bytes [16-19]
    RtcAccelType       accelType;        // Bytes [20-23]

    // James: I considered storing the original pointer in case we need it in the future.
    // I also considered some kind of device ID to verify that if you don't specify the
    // instanceTraversableHandles that you needed to be doing relocation on the same
    // device.  DeviceContext* would be unsafe, because we can't guarantee the pointer
    // would be valid at the time of use.
};

static OptixResult validateAccelOptions( const OptixAccelBuildOptions* accelOptions, ErrorDetails& errDetails )
{
    if( accelOptions->buildFlags
        & ~( OPTIX_BUILD_FLAG_ALLOW_UPDATE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
             | OPTIX_BUILD_FLAG_PREFER_FAST_BUILD | OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_ALLOW_RANDOM_INSTANCE_ACCESS
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
            | OPTIX_BUILD_FLAG_ALLOW_VM_UPDATE | OPTIX_BUILD_FLAG_ALLOW_DMM_UPDATE | OPTIX_BUILD_FLAG_ALLOW_DISABLE_VMS | OPTIX_BUILD_FLAG_IAS_USES_VM_REPLACEMENTS | OPTIX_BUILD_FLAG_IAS_REFERENCES_GAS_WITH_DMM
#endif // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
            ) )
    {
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, R"msg("accelOptions.buildFlags" contains invalid flags)msg" );
    }

    if( !( accelOptions->operation == OPTIX_BUILD_OPERATION_BUILD || accelOptions->operation == OPTIX_BUILD_OPERATION_UPDATE ) )
    {
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, R"msg("accelOptions.operation" is not a valid build operation)msg" );
    }

    if( accelOptions->operation == OPTIX_BUILD_OPERATION_UPDATE )
    {
        if( ( accelOptions->buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE ) == 0 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          R"msg("accelOptions.operation" is OPTIX_BUILD_OPERATION_UPDATE but "accelOptions.buildFlags" does not specify OPTIX_BUILD_FLAG_ALLOW_UPDATE)msg" );
    }

    if( accelOptions->motionOptions.numKeys > 1 )
    {
        if( accelOptions->motionOptions.flags & ~( OPTIX_MOTION_FLAG_START_VANISH | OPTIX_MOTION_FLAG_END_VANISH ) )
        {
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, R"msg("accelOptions.motionOptions.flags" contains invalid flags)msg" );
        }

        if( accelOptions->motionOptions.timeBegin == INFINITY )
        {
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, R"msg("accelOptions.motionOptions.timeBegin" is infinity)msg" );
        }
        if( accelOptions->motionOptions.timeEnd == INFINITY )
        {
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, R"msg("accelOptions.motionOptions.timeEnd" is infinity)msg" );
        }

        if( std::isnan( accelOptions->motionOptions.timeBegin ) )
        {
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, R"msg("accelOptions.motionOptions.timeBegin" is NaN)msg" );
        }
        if( std::isnan( accelOptions->motionOptions.timeEnd ) )
        {
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, R"msg("accelOptions.motionOptions.timeEnd" is NaN)msg" );
        }

        if( accelOptions->motionOptions.timeBegin > accelOptions->motionOptions.timeEnd )
        {
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          R"msg("accelOptions.motionOptions.timeBegin" is larger than "accelOptions.motionOptions.timeEnd")msg" );
        }
    }

    return OPTIX_SUCCESS;
}

static OptixResult validateAccelOptions( const OptixAccelBuildOptions* accelOptions,
                                         const OptixTraversableHandle* outputHandle,
                                         ErrorDetails&                 errDetails )
{
    if( accelOptions->operation == OPTIX_BUILD_OPERATION_BUILD && outputHandle == nullptr )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      R"msg("accelOptions.operation" is OPTIX_BUILD_OPERATION_BUILD but "outputHandle" is null)msg" );
    return validateAccelOptions( accelOptions, errDetails );
}

static OptixResult validateEmittedProperties( const OptixAccelBuildOptions* accelOptions,
                                              const OptixAccelEmitDesc*     emittedProperties,
                                              unsigned int                  numEmittedProperties,
                                              ErrorDetails&                 errDetails )
{
    if( !emittedProperties )
    {
        if( numEmittedProperties )
        {
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          R"msg("numEmittedProperties" is non-zero but "emittedProperties" is null)msg" );
        }
    }
    else if( !numEmittedProperties )
    {
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      R"msg("emittedProperties" is non-null but "numEmittedProperties" is 0)msg" );
    }

    for( unsigned int i = 0; i < numEmittedProperties; ++i )
    {
        if( emittedProperties[i].result == 0 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          corelib::stringf( R"msg("emittedProperties[%u].result" is zero)msg", i ) );

        switch( emittedProperties[i].type )
        {
            case OPTIX_PROPERTY_TYPE_COMPACTED_SIZE:
                if( ( accelOptions->buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION ) == 0 )
                {
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  corelib::stringf( R"msg(Invalid value (%u) for "emittedProperties[%u].type": Querying compacted size, but build flag "OPTIX_BUILD_FLAG_ALLOW_COMPACTION" is not set.)msg",
                                                                    static_cast<unsigned int>( emittedProperties[i].type ), i ) );
                }
                if( emittedProperties[i].result % sizeof( RtcGpuVA ) != 0 )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, corelib::stringf( R"msg("emittedProperties[%u].result" is not a multiple of %zu)msg",
                                                                                               i, sizeof( RtcGpuVA ) ) );
                break;

            case OPTIX_PROPERTY_TYPE_AABBS:
                if( emittedProperties[i].result % OPTIX_AABB_BUFFER_BYTE_ALIGNMENT != 0 )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, corelib::stringf( R"msg("emittedProperties[%u].result" is not a multiple of OPTIX_AABB_BUFFER_BYTE_ALIGNMENT)msg",
                                                                                               i ) );
                break;

            default:
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                              corelib::stringf( R"msg(Invalid value (%u) for "emittedProperties[%u].type")msg",
                                                                static_cast<unsigned int>( emittedProperties[i].type ), i ) );
        }
    }

    return OPTIX_SUCCESS;
}

static OptixResult getRelocationInfo( DeviceContext* context, OptixTraversableHandle handle, OptixAccelRelocationInfo* info, ErrorDetails& errDetails )
{
    memset( info, 0, sizeof( OptixAccelRelocationInfo ) );
    static_assert( sizeof( OptixAccelRelocationInfo ) >= sizeof( RelocationInfo ),
                   "OptixAccelRelocationInfo is too small" );
    RelocationInfo reloc = {};

    if( const RtcResult rtcResult = context->getRtcore().deviceContextGetCompatibilityIdentifier(
            context->getRtcDeviceContext(), RTC_COMPATIBILITY_TYPE_ACCEL, &reloc.rtcId ) )
        return errDetails.logDetails( rtcResult, "Error retrieving compatibility information from RTX" );

    RtcGpuVA sourceBuffer = 0;

    if( const RtcResult rtcResult = context->getRtcore().colwertTraversableHandleToPointer(
            context->getRtcDeviceContext(), handle, &sourceBuffer, &reloc.traversableType, &reloc.accelType ) )
        return errDetails.logDetails( rtcResult, "Failed to colwert traversable handle to pointer" );

    if( reloc.traversableType != RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL && reloc.traversableType != RTC_TRAVERSABLE_TYPE_BOTTOM_LEVEL_ACCEL
        && reloc.traversableType != RTC_TRAVERSABLE_TYPE_EXTENDED_BOTTOM_LEVEL_ACCEL )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "Traversable handle is not from an acceleration structure build" );

    *reinterpret_cast<RelocationInfo*>( info ) = reloc;

    return OPTIX_SUCCESS;
}

static OptixResult registerAS( 
    DeviceContext*         context,
    LWstream               stream,
    OptixTraversableHandle handle,
    LWdeviceptr            buffer,
    size_t                 bufferSizeInBytes,
    ErrorDetails&          errDetails )
{
    if( !k_dumpBvhs.get() )
        return OPTIX_SUCCESS;

#if defined( OPTIX_ENABLE_REGISTER_AS )
    if( const LWresult result = corelib::lwdaDriver().LwStreamSynchronize( stream ) )
        return errDetails.logDetails( result, "Failed to synchronize stream." );

    optix_exp::DeviceContextLogger& clog = context->getLogger();

    size_t version = context->registerTraversable( buffer );

    RtcGpuVA           traversableBuffer = 0;
    RtcTraversableType traversableType   = RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL;
    RtcAccelType       accelType         = RTC_ACCEL_TYPE_NOACCEL;

    if( const RtcResult rtcResult = context->getRtcore().colwertTraversableHandleToPointer(
            context->getRtcDeviceContext(), handle, &traversableBuffer, &traversableType, &accelType ) )
        return errDetails.logDetails( rtcResult, "Failed to colwert traversable handle to pointer" );

    OptixAccelRelocationInfo info;
    if( const OptixResult result = optix_exp::getRelocationInfo( context, handle, &info, errDetails ) )
        return result;

    std::vector<char> mem;
    mem.resize(bufferSizeInBytes + sizeof(OptixAccelRelocationInfo));
    memcpy( mem.data(), (void*)&info, sizeof(OptixAccelRelocationInfo) );
    if( const LWresult result = corelib::lwdaDriver().LwMemcpyDtoH( mem.data() + sizeof(OptixAccelRelocationInfo), buffer, bufferSizeInBytes ) )
        return errDetails.logDetails( result, "Failed to copy BVH to host." );

    std::string bvhDumpPath = k_bvhDumpPath.get();
    std::replace( bvhDumpPath.begin(), bvhDumpPath.end(), '\\', '/' );
    if( bvhDumpPath.rfind( '/' ) == ( bvhDumpPath.length() - 1 ) )
        bvhDumpPath = bvhDumpPath.substr( 0, bvhDumpPath.length() - 1 );

    std::string bvhFileName = corelib::stringf( "%s/bvhdump-%u-%08llu-%08llx.o7b", bvhDumpPath.c_str(),
                                                traversableType, (unsigned long long)version, (unsigned long long)handle );

    FILE* file = fopen( bvhFileName.c_str(), "wb" );
    if( !file )
        return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, "Failed to open BVH dump output file." );
    fwrite( mem.data(), 1, mem.size(), file );
    fclose( file );    
#endif

    return OPTIX_SUCCESS;
}


static OptixResult registerTraversables( 
    DeviceContext*  context,
    LWstream        stream,
    const char*     traversableHandles,
    size_t          numTraversableHandles,
    size_t          strideInBytes,
    ErrorDetails&   errDetails )
{
    if( !k_dumpBvhs.get() )
        return OPTIX_SUCCESS;

#if defined( OPTIX_ENABLE_REGISTER_AS )
    if( const LWresult result = corelib::lwdaDriver().LwStreamSynchronize( stream ) )
        return errDetails.logDetails( result, "Failed to synchronize stream." );

    for( size_t i = 0; i < numTraversableHandles; ++i )
    {
        OptixTraversableHandle handle = *reinterpret_cast<const OptixTraversableHandle*>(traversableHandles + i * strideInBytes);

        // dump the chain up to (excl.) the first AS
        while(handle)
        {
            OptixTraversableHandle childHandle = 0;

            RtcGpuVA           traversableBuffer = 0;
            RtcTraversableType traversableType   = RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL;
            RtcAccelType       accelType         = RTC_ACCEL_TYPE_NOACCEL;
            if( const RtcResult rtcResult = context->getRtcore().colwertTraversableHandleToPointer(
                context->getRtcDeviceContext(), handle, &traversableBuffer, &traversableType, &accelType ) )
                return errDetails.logDetails( rtcResult, "Failed to colwert traversable handle to pointer" );
            
            size_t baseSizeInBytes = 64;

            std::vector<char> traversableMem;
            traversableMem.resize(baseSizeInBytes);
            if( const LWresult result = corelib::lwdaDriver().LwMemcpyDtoH( traversableMem.data(), traversableBuffer, baseSizeInBytes ) )
                return errDetails.logDetails( result, "Failed to copy traversable header to host." );

            if( traversableType == RTC_TRAVERSABLE_TYPE_STATIC_TRANSFORM )
            {
                // fixed size
                const OptixStaticTransform* staticTransform = reinterpret_cast<const OptixStaticTransform*>(traversableMem.data());
                childHandle = staticTransform->child;
            }
            else if( traversableType == RTC_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM )
            {
                // download any extra matrix keys
                const OptixMatrixMotionTransform* matrixTransform = reinterpret_cast<const OptixMatrixMotionTransform*>(traversableMem.data());
                childHandle = matrixTransform->child;

                size_t traversableSizeInBytes = sizeof(OptixMatrixMotionTransform) + (std::max(2u,(unsigned)matrixTransform->motionOptions.numKeys) - 2u) * 12 * sizeof(float);
                if( traversableSizeInBytes > baseSizeInBytes )
                {
                    traversableMem.resize(traversableSizeInBytes);
                    if( const LWresult result = corelib::lwdaDriver().LwMemcpyDtoH( traversableMem.data() + baseSizeInBytes, traversableBuffer + baseSizeInBytes, traversableSizeInBytes - baseSizeInBytes ) )
                        return errDetails.logDetails( result, "Failed to copy matrix motion transform keys to host." );
                }                    
            }
            else if( traversableType == RTC_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM )
            {
                // download any extra srt keys
                const OptixSRTMotionTransform* srtTransform = reinterpret_cast<const OptixSRTMotionTransform*>(traversableMem.data());
                childHandle = srtTransform->child;

                size_t traversableSizeInBytes = sizeof(OptixSRTMotionTransform) + (std::max(2u,(unsigned)srtTransform->motionOptions.numKeys) - 2u) * sizeof(OptixSRTData);
                if( traversableSizeInBytes > baseSizeInBytes )
                {
                    traversableMem.resize(traversableSizeInBytes);
                    if( const LWresult result = corelib::lwdaDriver().LwMemcpyDtoH( traversableMem.data() + baseSizeInBytes, traversableBuffer + baseSizeInBytes, traversableSizeInBytes - baseSizeInBytes ) )
                        return errDetails.logDetails( result, "Failed to copy srt motion transform keys to host." );
                }                    
            }
            else
            {
                // only dump transforms. AS have already been dumped at build/compact/relocate.
                break;
            }

            size_t version = context->registerTraversable( traversableBuffer );

            std::string bvhDumpPath = k_bvhDumpPath.get();
            std::replace( bvhDumpPath.begin(), bvhDumpPath.end(), '\\', '/' );
            if( bvhDumpPath.rfind( '/' ) == ( bvhDumpPath.length() - 1 ) )
                bvhDumpPath = bvhDumpPath.substr( 0, bvhDumpPath.length() - 1 );

            std::string filename = corelib::stringf( "%s/bvhdump-%u-%08llu-%08llx.o7b", bvhDumpPath.c_str(),
                                                        traversableType, (unsigned long long)version, (unsigned long long)handle );

            FILE* file = fopen( filename.c_str(), "wb" );
            if( !file )
                return errDetails.logDetails( OPTIX_ERROR_FILE_IO_ERROR, "Failed to open traversable dump output file." );
            fwrite( traversableMem.data(), 1, traversableMem.size(), file );
            fclose( file );  

            handle = childHandle;
        }
    }
#endif

    return OPTIX_SUCCESS;
}

// Indicates whether the given pointer was allocated with lwdaMalloc() or lwMemAlloc() (in contrast to
// lwdaMallocManaged(), lwMemHostRegister(), lwMemHostAlloc(), and plain host pointers on the stack and heap).
//
// Pointers from lwMemHostRegister() with LW_MEMHOSTREGISTER_IOMEMORY as part of the flags have not been tested since
// it is not clear how to make lwMemHostRegister() succeed in that case.
static bool isLWDeviceptrFromLwdaMalloc( LWdeviceptr ptr )
{
    bool     managed = false;
    LWresult managedResult =
        corelib::lwdaDriver().LwPointerGetAttribute( &managed, LW_POINTER_ATTRIBUTE_IS_MANAGED, (LWdeviceptr)ptr );

    // Recognizes pointers from lwdaMallocManaged()
    if( ( managedResult == LWDA_SUCCESS ) && managed )
        return false;

    unsigned int memoryType = ~0;
    LWresult     memoryTypeResult =
        corelib::lwdaDriver().LwPointerGetAttribute( &memoryType, LW_POINTER_ATTRIBUTE_MEMORY_TYPE, (LWdeviceptr)ptr );

    // Recognizes pointers from lwMemHostRegister() and lwMemHostAlloc()
    if( ( memoryTypeResult != LWDA_SUCCESS ) || ( memoryType != LW_MEMORYTYPE_DEVICE ) )
        return false;

    // Should be a pointer from lwdaMalloc() or lwMemAlloc()
    return true;
}

static OptixResult buildAccel( DeviceContext*                context,
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
                               unsigned int                  numEmittedProperties,
                               ErrorDetails&                 errDetails )
{
    OPTIX_CHECK_VALIDATION_MODE_LWRRENT_LWDA_CONTEXT( context );
    OPTIX_CHECK_VALIDATION_MODE_STREAM_STATE( context, stream );
    if( const OptixResult result = validateAccelOptions( accelOptions, outputHandle, errDetails ) )
    {
        return result;
    }

    if( const OptixResult result = validateEmittedProperties( accelOptions, emittedProperties, numEmittedProperties, errDetails ) )
    {
        return result;
    }

    if( !isLWDeviceptrFromLwdaMalloc( outputBuffer ) )
    {
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      R"msg("outputBuffer" does not point to memory allocated with lwdaMalloc() or lwMemAlloc())msg" );
    }

    RtcAccelBuilder accelBuilder( context, accelOptions, false, errDetails );
    if( const OptixResult result = accelBuilder.init( buildInputs, numBuildInputs ) )
    {
        return result;
    }

    OptixAccelBufferSizes bufferSizes = {};
#if defined( OPTIX_ENABLE_AS_ASSERTS )
    if( const OptixResult result = accelBuilder.computeMemoryUsage( &bufferSizes ) )
    {
        return result;
    }
    // can't validate output size for update of compacted ASes
    if( ( accelOptions->operation == OPTIX_BUILD_OPERATION_UPDATE ) && ( accelOptions->buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION ) )
        bufferSizes.outputSizeInBytes = 0ull;
#endif

    if( const OptixResult result = accelBuilder.validateTempBuffer( tempBuffer, tempBufferSizeInBytes, bufferSizes ) )
    {
        return result;
    }

    if( const OptixResult result = accelBuilder.validateOutputBuffer( outputBuffer, outputBufferSizeInBytes, bufferSizes ) )
    {
        return result;
    }

    if( const OptixResult result = accelBuilder.build( stream, tempBuffer, tempBufferSizeInBytes, outputBuffer, outputBufferSizeInBytes,
                                                       outputHandle, emittedProperties, numEmittedProperties ) )
    {
        return result;
    }

#if defined( OPTIX_ENABLE_REGISTER_AS )
    if( k_dumpBvhs.get() && numBuildInputs > 0)
    {
        // The user completely manages the transform traversables, so we have to register and dump them all at build time to
        // make sure the dumped traversable matches with the build AS.
        if( buildInputs[0].type == OPTIX_BUILD_INPUT_TYPE_INSTANCES || buildInputs[0].type == OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS)
        {
            size_t instanceSizeInBytes            = translateABI_getOptixInstanceTypeSize( context->getAbiVersion() );
            size_t traversableHandleOffsetInBytes = translateABI_getOptixInstanceTraversableHandleFieldOffet( context->getAbiVersion() );

            size_t numInstances = buildInputs[0].instanceArray.numInstances;
            size_t instancesSizeInBytes = numInstances * instanceSizeInBytes;
            std::vector<char> instanceMem;
            instanceMem.resize(instancesSizeInBytes);

            if( buildInputs[0].type == OPTIX_BUILD_INPUT_TYPE_INSTANCES )
            {
                if( const LWresult result = corelib::lwdaDriver().LwMemcpyDtoH( instanceMem.data(), buildInputs[0].instanceArray.instances, instancesSizeInBytes ) )
                    return errDetails.logDetails( result, "Failed to copy input instances to host." );
                const OptixInstance* instance = reinterpret_cast< const OptixInstance* >( instanceMem.data() );

                if( const OptixResult result = registerTraversables( context, stream, instanceMem.data() + traversableHandleOffsetInBytes, numInstances, instanceSizeInBytes, errDetails ) )
                    return result;
            }
            else if (buildInputs[0].type == OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS)
            {
                std::vector<char> mem;
                size_t instancePointersSizeInBytes = numInstances * sizeof(size_t );
                mem.resize( instancePointersSizeInBytes );
                if( const LWresult result = corelib::lwdaDriver().LwMemcpyDtoH( mem.data(), buildInputs[0].instanceArray.instances, instancePointersSizeInBytes ) )
                    return errDetails.logDetails( result, "Failed to copy input instance pointers to host." );

                for( size_t i = 0; i < numInstances; ++i )
                {
                    LWdeviceptr instancePtr = *reinterpret_cast<LWdeviceptr*>(mem.data() + i * sizeof(size_t));
                    if( const LWresult result = corelib::lwdaDriver().LwMemcpyDtoH( instanceMem.data() + i * instanceSizeInBytes, instancePtr, instanceSizeInBytes ) )
                        return errDetails.logDetails( result, "Failed to copy input instance to host." );
                }

                if( const OptixResult result = registerTraversables( context, stream, instanceMem.data() + traversableHandleOffsetInBytes, numInstances, instanceSizeInBytes, errDetails ) )
                    return result;
            }
        }
    }
#endif

    if( const OptixResult result = registerAS( context, stream, *outputHandle, outputBuffer, outputBufferSizeInBytes, errDetails ) )
        return result;

    return OPTIX_SUCCESS;
}

static OptixResult computeAccelMemoryUsage( DeviceContext*                context,
                                            const OptixAccelBuildOptions* accelOptions,
                                            const OptixBuildInput*        buildInputs,
                                            unsigned int                  numBuildInputs,
                                            OptixAccelBufferSizes*        bufferSizes,
                                            ErrorDetails&                 errDetails )
{
    if( const OptixResult result = validateAccelOptions( accelOptions, errDetails ) )
    {
        return result;
    }

    RtcAccelBuilder accelBuilder( context, accelOptions, true, errDetails );

    if( const OptixResult result = accelBuilder.init( buildInputs, numBuildInputs ) )
    {
        return result;
    }

    if( const OptixResult result = accelBuilder.computeMemoryUsage( bufferSizes ) )
    {
        return result;
    }
    return OPTIX_SUCCESS;
}

static OptixResult compactAccel( DeviceContext*          context,
                                 LWstream                stream,
                                 OptixTraversableHandle  inputHandle,
                                 LWdeviceptr             outputBuffer,
                                 size_t                  outputBufferSizeInBytes,
                                 OptixTraversableHandle* outputHandle,
                                 ErrorDetails&           errDetails )
{
    OPTIX_CHECK_VALIDATION_MODE_LWRRENT_LWDA_CONTEXT( context );
    OPTIX_CHECK_VALIDATION_MODE_STREAM_STATE( context, stream );
    RtcCopyMode rtcMode = RTC_COPY_MODE_COMPACT;

    if( outputBuffer % OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT != 0 )
    {
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      R"msg("outputBuffer" is not a multiple of OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT)msg" );
    }

    if( !isLWDeviceptrFromLwdaMalloc( outputBuffer ) )
    {
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      R"msg("outputBuffer" does not point to memory allocated with lwdaMalloc() or lwMemAlloc())msg" );
    }

    RtcGpuVA           sourceBuffer    = 0;
    RtcTraversableType traversableType = RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL;
    RtcAccelType       accelType       = RTC_ACCEL_TYPE_NOACCEL;

    if( const RtcResult rtcResult = context->getRtcore().colwertTraversableHandleToPointer(
            context->getRtcDeviceContext(), inputHandle, &sourceBuffer, &traversableType, &accelType ) )
        return errDetails.logDetails( rtcResult, "Failed to colwert traversable handle to pointer" );

    if( traversableType != RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL && traversableType != RTC_TRAVERSABLE_TYPE_BOTTOM_LEVEL_ACCEL
        && traversableType != RTC_TRAVERSABLE_TYPE_EXTENDED_BOTTOM_LEVEL_ACCEL )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "Input traversable handle is not from an acceleration structure build." );

    optix_exp::ScopedCommandList commandList( context );

    if( const OptixResult result = commandList.init( stream, errDetails ) )
        return result;

    LWdeviceptr optixSourceBuffer =
        sourceBuffer - ( ( traversableType == RTC_TRAVERSABLE_TYPE_EXTENDED_BOTTOM_LEVEL_ACCEL ) ? EXTENDED_ACCEL_HEADER_SIZE : 0 );
    LWdeviceptr optixOutputBuffer = outputBuffer;
    LWdeviceptr rtcSourceBuffer   = sourceBuffer;
    LWdeviceptr rtcOutputBuffer =
        outputBuffer + ( ( traversableType == RTC_TRAVERSABLE_TYPE_EXTENDED_BOTTOM_LEVEL_ACCEL ) ? EXTENDED_ACCEL_HEADER_SIZE : 0 );

    size_t optixOutputBufferSizeInBytes = outputBufferSizeInBytes;
    size_t rtcOutputBufferSizeInBytes =
        outputBufferSizeInBytes
        - ( ( traversableType == RTC_TRAVERSABLE_TYPE_EXTENDED_BOTTOM_LEVEL_ACCEL ) ? EXTENDED_ACCEL_HEADER_SIZE : 0 );

    const RtcResult rtcResult = context->getRtcore().accelCopy( commandList.get(), rtcSourceBuffer, rtcMode,
                                                                rtcOutputBuffer, rtcOutputBufferSizeInBytes );

    // Always destroy the command list, regardless of error to accelRelocate
    OptixResult result = commandList.destroy( errDetails );

    if( rtcResult )
        return errDetails.logDetails( rtcResult, "Failed to compact acceleration structure data" );
    if( result )
        return result;

    if( traversableType == RTC_TRAVERSABLE_TYPE_EXTENDED_BOTTOM_LEVEL_ACCEL )
    {
        // Copy the extended header in front of the BVH and the data behind the BVH.
        compactExtendedBuffer( stream, optixSourceBuffer, optixOutputBuffer, rtcOutputBufferSizeInBytes );
    }

    RtcTraversableHandle rtcTraversableHandle = 0;

    if( const RtcResult rtcResult = context->getRtcore().colwertPointerToTraversableHandle(
            context->getRtcDeviceContext(), rtcOutputBuffer, traversableType, accelType, &rtcTraversableHandle ) )
        return errDetails.logDetails( rtcResult, "Failed to colwert output pointer to traversable handle" );

    *outputHandle = rtcTraversableHandle;

    if( const OptixResult result = registerAS( context, stream, rtcTraversableHandle, outputBuffer, outputBufferSizeInBytes, errDetails ) )
    {
        return result;
    }

    return OPTIX_SUCCESS;
}

static OptixResult colwertPointerToTraversableHandle( DeviceContext*          context,
                                                      LWdeviceptr             pointer,
                                                      OptixTraversableType    traversableType,
                                                      OptixTraversableHandle* traversableHandle,
                                                      ErrorDetails&           errDetails )
{
    if( pointer % OPTIX_TRANSFORM_BYTE_ALIGNMENT != 0 )
    {
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, R"msg("pointer" is not a multiple of OPTIX_TRANSFORM_BYTE_ALIGNMENT)msg" );
    }

    RtcTraversableType rtcTraversableType;

    switch( traversableType )
    {
        case OPTIX_TRAVERSABLE_TYPE_STATIC_TRANSFORM:
            rtcTraversableType = RTC_TRAVERSABLE_TYPE_STATIC_TRANSFORM;
            break;
        case OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM:
            rtcTraversableType = RTC_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM;
            break;
        case OPTIX_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM:
            rtcTraversableType = RTC_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM;
            break;
        default:
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          corelib::stringf( R"msg(Invalid value (0x%x) for argument "traversableType")msg",
                                                            static_cast<unsigned int>( traversableType ) ) );
    }
    RtcTraversableHandle rtcTraversableHandle = 0;


    if( const RtcResult rtcResult = context->getRtcore().colwertPointerToTraversableHandle(
            context->getRtcDeviceContext(), pointer, rtcTraversableType, RTC_ACCEL_TYPE_NOACCEL,
            &rtcTraversableHandle ) ) /* [out] traversable handle. traversableHandle must be in host memory */
        return errDetails.logDetails( rtcResult, "Error colwerting pointer to traversable handle" );

    *traversableHandle = rtcTraversableHandle;
    return OPTIX_SUCCESS;
}

static OptixResult checkRelocationCompatibility( DeviceContext* context, const OptixAccelRelocationInfo* info, int* compatible, ErrorDetails& errDetails )
{
    const RelocationInfo* reloc = reinterpret_cast<const RelocationInfo*>( info );
    const RtcResult       rtcResult =
        context->getRtcore().deviceContextCheckCompatibility( context->getRtcDeviceContext(),
                                                              RTC_COMPATIBILITY_TYPE_ACCEL, &reloc->rtcId );
    if( rtcResult == RTC_SUCCESS )
        *compatible = 1;
    else if( rtcResult == RTC_ERROR_ILWALID_VERSION || rtcResult == RTC_ERROR_NOT_SUPPORTED )
        *compatible = 0;
    else
        return errDetails.logDetails( rtcResult, "Error from RTX checking compatibility information" );
    return OPTIX_SUCCESS;
}

static OptixResult relocateAccel( DeviceContext*                  context,
                                  LWstream                        stream,
                                  const OptixAccelRelocationInfo* info,
                                  LWdeviceptr                     instanceTraversableHandles,
                                  size_t                          numInstanceTraversableHandles,
                                  LWdeviceptr                     targetAccel,
                                  size_t                          targetAccelSizeInBytes,
                                  OptixTraversableHandle*         targetHandle,
                                  ErrorDetails&                   errDetails )
{
    OPTIX_CHECK_VALIDATION_MODE_LWRRENT_LWDA_CONTEXT( context );
    OPTIX_CHECK_VALIDATION_MODE_STREAM_STATE( context, stream );
    if( targetAccel % OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT != 0 )
    {
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      R"msg("targetAccel" is not a multiple of OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT)msg" );
    }
    int compatible = 0;
    if( const OptixResult result = checkRelocationCompatibility( context, info, &compatible, errDetails ) )
        return result;
    if( !compatible )
        return errDetails.logDetails( OPTIX_ERROR_ACCEL_NOT_COMPATIBLE,
                                      R"msg(Relocation to requested device is not compatible with the original acceleration structure)msg" );

    const RelocationInfo* reloc = reinterpret_cast<const RelocationInfo*>( info );

    // We can't be sure that the TLAS doesn't already have zero elements, so zero input is
    // technically legal.  We can make sure that these are zero for bottom level accels.
    if( reloc->traversableType == RTC_TRAVERSABLE_TYPE_BOTTOM_LEVEL_ACCEL
        || reloc->traversableType == RTC_TRAVERSABLE_TYPE_EXTENDED_BOTTOM_LEVEL_ACCEL )
    {
        if( instanceTraversableHandles != 0 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          R"msg("instanceTraversableHandles" must be null for bottom level acceleration structures)msg" );
        if( numInstanceTraversableHandles != 0 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          R"msg("numInstanceTraversableHandles" must be zero for bottom level acceleration structures)msg" );
    }

    if( !isLWDeviceptrFromLwdaMalloc( targetAccel ) )
    {
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      R"msg("targetAccel" does not point to memory allocated with lwdaMalloc() or lwMemAlloc())msg" );
    }

    optix_exp::ScopedCommandList commandList( context );

    if( const OptixResult result = commandList.init( stream, errDetails ) )
        return result;

    LWdeviceptr rtcTargetAccel =
        targetAccel
        + ( reloc->traversableType == RTC_TRAVERSABLE_TYPE_EXTENDED_BOTTOM_LEVEL_ACCEL ? EXTENDED_ACCEL_HEADER_SIZE : 0 );
    size_t rtcTargetAccelSizeInBytes =
        targetAccelSizeInBytes
        - ( reloc->traversableType == RTC_TRAVERSABLE_TYPE_EXTENDED_BOTTOM_LEVEL_ACCEL ? EXTENDED_ACCEL_HEADER_SIZE : 0 );

    const RtcResult rtcResult =
        context->getRtcore().accelRelocate( commandList.get(), instanceTraversableHandles,
                                            numInstanceTraversableHandles, rtcTargetAccel, rtcTargetAccelSizeInBytes );

    // Always destroy the command list, regardless of error to accelRelocate
    OptixResult result = commandList.destroy( errDetails );

    if( rtcResult )
        return errDetails.logDetails( rtcResult, "Failed to relocate acceleration structure with RTX" );
    if( result )
        return result;

    RtcTraversableHandle rtcTraversableHandle = 0;

    if( const RtcResult cvtResult = context->getRtcore().colwertPointerToTraversableHandle(
            context->getRtcDeviceContext(), rtcTargetAccel, reloc->traversableType, reloc->accelType, &rtcTraversableHandle ) )
        return errDetails.logDetails( cvtResult, "Failed to colwert output pointer to traversable handle" );

    *targetHandle = rtcTraversableHandle;

    if( k_dumpBvhs.get() )
    {
#if defined( OPTIX_ENABLE_REGISTER_AS )
        // The user completely manages the transform traversables, so we have to register and dump all relocated traversables
        // at relocation time to make sure the dumped traversable matches with the relocated AS.
        if( reloc->traversableType == RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL && instanceTraversableHandles != 0 )
        {
            std::vector<char> mem;
            size_t sizeInBytes = numInstanceTraversableHandles * sizeof(RtcTraversableHandle);
            mem.resize(sizeInBytes);
            if( const LWresult result = corelib::lwdaDriver().LwMemcpyDtoH( mem.data(), instanceTraversableHandles, sizeInBytes ) )
                return errDetails.logDetails( result, "Failed to copy instance handles to host." );

            if( const OptixResult result = registerTraversables( context, stream, mem.data(), numInstanceTraversableHandles, sizeof( RtcTraversableHandle ), errDetails ) )
                return result;
        }
#endif
    }

    if( const OptixResult result = registerAS( context, stream, rtcTraversableHandle, targetAccel, targetAccelSizeInBytes, errDetails ) )
    {
        return result;
    }

    return OPTIX_SUCCESS;
}


#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
static OptixResult visibilityMapArrayComputeMemoryUsage( DeviceContext*                           context,
                                                         const OptixVisibilityMapArrayBuildInput* buildInput,
                                                         OptixMicromeshBufferSizes*               bufferSizes,
                                                         ErrorDetails&                            errDetails )
{
    RtcVmBuilder vmBuilder( context, true, errDetails );

    if( const OptixResult result = vmBuilder.init( buildInput ) )
    {
        return result;
    }

    if( const OptixResult result = vmBuilder.computeMemoryUsage( bufferSizes ) )
    {
        return result;
    }
    return OPTIX_SUCCESS;
}

static OptixResult visibilityMapArrayBuild( DeviceContext*                           context,
                                            LWstream                                 stream,
                                            const OptixVisibilityMapArrayBuildInput* buildInput,
                                            const OptixMicromeshBuffers*             buffers,
                                            const OptixMicromeshEmitDesc*            emittedProperties,
                                            unsigned int                             numEmittedProperties,
                                            ErrorDetails&                            errDetails )
{
    RtcVmBuilder vmBuilder( context, false, errDetails );

    if( const OptixResult result = vmBuilder.init( buildInput ) )
    {
        return result;
    }

    if( const OptixResult result = vmBuilder.build( stream, buffers, emittedProperties, numEmittedProperties ) )
    {
        return result;
    }
    return OPTIX_SUCCESS;
}

static OptixResult displacedMicromeshArrayComputeMemoryUsage( DeviceContext*                           context,
                                                              const OptixDisplacedMicromeshArrayBuildInput* buildInput,
                                                              OptixMicromeshBufferSizes*                    bufferSizes,
                                                              ErrorDetails&                                 errDetails )
{
    RtcDmmBuilder dmmBuilder( context, true, errDetails );

    if( const OptixResult result = dmmBuilder.init( buildInput ) )
    {
        return result;
    }

    if( const OptixResult result = dmmBuilder.computeMemoryUsage( bufferSizes ) )
    {
        return result;
    }
    return OPTIX_SUCCESS;
}

static OptixResult displacedMicromeshArrayBuild( DeviceContext*                                context,
                                                 LWstream                                      stream,
                                                 const OptixDisplacedMicromeshArrayBuildInput* buildInput,
                                                 const OptixMicromeshBuffers*                  buffers,
                                                 const OptixMicromeshEmitDesc*                 emittedProperties,
                                                 unsigned int                                  numEmittedProperties,
                                                 ErrorDetails&                                 errDetails )
{
    RtcDmmBuilder dmmBuilder( context, false, errDetails );

    if( const OptixResult result = dmmBuilder.init( buildInput ) )
    {
        return result;
    }

    if( const OptixResult result = dmmBuilder.build( stream, buffers, emittedProperties, numEmittedProperties ) )
    {
        return result;
    }
    return OPTIX_SUCCESS;
}
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)


}  // namespace optix_exp

extern "C" OptixResult optixAccelComputeMemoryUsage( OptixDeviceContext            contextAPI,
                                                     const OptixAccelBuildOptions* accelOptions,
                                                     const OptixBuildInput*        buildInputs,
                                                     unsigned int                  numBuildInputs,
                                                     OptixAccelBufferSizes*        bufferSizes )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::ACCEL_COMPUTE_MEMORY_USAGE );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( accelOptions );
    OPTIX_CHECK_NULL_ARGUMENT( buildInputs );
    OPTIX_CHECK_ZERO_ARGUMENT( numBuildInputs );
    OPTIX_CHECK_NULL_ARGUMENT( bufferSizes );

    try
    {
        optix_exp::ErrorDetails errDetails;
        if( const OptixResult result =
                computeAccelMemoryUsage( context, accelOptions, buildInputs, numBuildInputs, bufferSizes, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixAccelBuild( OptixDeviceContext            contextAPI,
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
                                        unsigned int                  numEmittedProperties )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();

    prodlib::HostStopwatch stopWatch;
    optix_exp::StreamMetricTimer scopeGuard( context, stream, "accel_build_time_GPU_ms" );

    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::ACCEL_BUILD );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( accelOptions );
    OPTIX_CHECK_NULL_ARGUMENT( buildInputs );
    OPTIX_CHECK_ZERO_ARGUMENT( numBuildInputs );
    OPTIX_CHECK_ZERO_ARGUMENT( outputBuffer );
    OPTIX_CHECK_ZERO_ARGUMENT( outputBufferSizeInBytes );

    try
    {
        optix_exp::ErrorDetails errDetails;
        if( const OptixResult result = buildAccel( context, stream, accelOptions, buildInputs, numBuildInputs, tempBuffer,
                                                   tempBufferSizeInBytes, outputBuffer, outputBufferSizeInBytes,
                                                   outputHandle, emittedProperties, numEmittedProperties, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }

        if( optix_exp::Metrics* metrics = context->getMetrics() )
        {
            double duration = stopWatch.getElapsed();
            metrics->logFloat( "accel_build_time_CPU_ms", duration, errDetails );
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixAccelGetRelocationInfo( OptixDeviceContext contextAPI, OptixTraversableHandle handle, OptixAccelRelocationInfo* info )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::ACCEL_GET_RELOCATION_INFO );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_ZERO_ARGUMENT( handle );
    OPTIX_CHECK_NULL_ARGUMENT( info );

    try
    {
        optix_exp::ErrorDetails errDetails;
        if( const OptixResult result = optix_exp::getRelocationInfo( context, handle, info, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixAccelCheckRelocationCompatibility( OptixDeviceContext              contextAPI,
                                                               const OptixAccelRelocationInfo* info,
                                                               int*                            compatible )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::ACCEL_CHECK_RELOCATION_COMPATIBILITY );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( info );
    OPTIX_CHECK_NULL_ARGUMENT( compatible );

    try
    {
        optix_exp::ErrorDetails errDetails;
        if( const OptixResult result = optix_exp::checkRelocationCompatibility( context, info, compatible, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixAccelRelocate( OptixDeviceContext              contextAPI,
                                           LWstream                        stream,
                                           const OptixAccelRelocationInfo* info,
                                           LWdeviceptr                     instanceTraversableHandles,
                                           size_t                          numInstanceTraversableHandles,
                                           LWdeviceptr                     targetAccel,
                                           size_t                          targetAccelSizeInBytes,
                                           OptixTraversableHandle*         targetHandle )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::ACCEL_RELOCATE );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( info );
    OPTIX_CHECK_ZERO_ARGUMENT( targetAccel );
    OPTIX_CHECK_ZERO_ARGUMENT( targetAccelSizeInBytes );
    OPTIX_CHECK_NULL_ARGUMENT( targetHandle );

    try
    {
        optix_exp::ErrorDetails errDetails;
        if( const OptixResult result = optix_exp::relocateAccel( context, stream, info, instanceTraversableHandles,
                                                                 numInstanceTraversableHandles, targetAccel,
                                                                 targetAccelSizeInBytes, targetHandle, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}


extern "C" OptixResult optixAccelCompact( OptixDeviceContext      contextAPI,
                                          LWstream                stream,
                                          OptixTraversableHandle  inputHandle,
                                          LWdeviceptr             outputBuffer,
                                          size_t                  outputBufferSizeInBytes,
                                          OptixTraversableHandle* outputHandle )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::ACCEL_COMPACT );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_ZERO_ARGUMENT( inputHandle );
    OPTIX_CHECK_ZERO_ARGUMENT( outputBuffer );
    OPTIX_CHECK_ZERO_ARGUMENT( outputBufferSizeInBytes );
    OPTIX_CHECK_NULL_ARGUMENT( outputHandle );

    try
    {
        optix_exp::ErrorDetails errDetails;
        if( const OptixResult result = optix_exp::compactAccel( context, stream, inputHandle, outputBuffer,
                                                                outputBufferSizeInBytes, outputHandle, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;


    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixColwertPointerToTraversableHandle( OptixDeviceContext      contextAPI,
                                                               LWdeviceptr             pointer,
                                                               OptixTraversableType    traversableType,
                                                               OptixTraversableHandle* traversableHandle )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::COLWERT_POINTER_TO_TRAVERSABLE_HANDLE );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( traversableHandle );
    OPTIX_CHECK_ZERO_ARGUMENT( pointer );

    try
    {
        optix_exp::ErrorDetails errDetails;
        if( const OptixResult result = optix_exp::colwertPointerToTraversableHandle( context, pointer, traversableType,
                                                                                     traversableHandle, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixAccelInstanceAabbsComputeMemoryUsage( OptixDeviceContext        contextAPI,
                                                                  LWstream                  stream,
                                                                  const OptixMotionOptions* iasMotionOptions,
                                                                  LWdeviceptr               instances,
                                                                  unsigned int              numInstances,
                                                                  LWdeviceptr               tempBufferSizeInBytes )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::ACCEL_INSTANCE_AABBS_COMPUTE_MEMORY_USAGE );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_ZERO_ARGUMENT( tempBufferSizeInBytes );

    try
    {
        optix_exp::ErrorDetails errDetails;
        // note that we still need to run
        if( !k_enableInstanceAabbComputer.get() )
        {
            // avoid any issues due to uninitialized memory for the user
            size_t tempMemSize = 0;
            if( const LWresult result = corelib::lwdaDriver().LwMemcpyHtoDAsync( tempBufferSizeInBytes, &tempMemSize, sizeof( size_t ), stream ) )
            {
                OptixResult res = errDetails.logDetails( result, "Copying data to device failed." );
                clog.sendError( errDetails );
                return res;
            }
        }
        else
        {
            OPTIX_CHECK_NULL_ARGUMENT( iasMotionOptions );
            OPTIX_CHECK_ZERO_ARGUMENT( instances );
            OPTIX_CHECK_ZERO_ARGUMENT( numInstances );

            if( const OptixResult result = optix_exp::instanceAabbsComputeMemoryUsage(
                    context, stream, iasMotionOptions, instances, numInstances, tempBufferSizeInBytes, errDetails ) )
            {
                return result;
            }
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixAccelInstanceAabbsCompute( OptixDeviceContext        contextAPI,
                                                       LWstream                  stream,
                                                       const OptixMotionOptions* iasMotionOptions,
                                                       LWdeviceptr               instances,
                                                       unsigned int              numInstances,
                                                       LWdeviceptr               tempBuffer,
                                                       size_t                    tempBufferSizeInBytes,
                                                       LWdeviceptr               outputAabbBuffer,
                                                       size_t                    outputAabbBufferSizeInBytes )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::ACCEL_INSTANCE_AABBS_COMPUTE );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    if( k_enableInstanceAabbComputer.get() )
    {

        OPTIX_CHECK_NULL_ARGUMENT( iasMotionOptions );
        OPTIX_CHECK_ZERO_ARGUMENT( instances );
        OPTIX_CHECK_ZERO_ARGUMENT( numInstances );
        OPTIX_CHECK_ZERO_ARGUMENT( tempBuffer );
        OPTIX_CHECK_ZERO_ARGUMENT( tempBufferSizeInBytes );
        OPTIX_CHECK_ZERO_ARGUMENT( outputAabbBuffer );
        OPTIX_CHECK_ZERO_ARGUMENT( outputAabbBufferSizeInBytes );

        try
        {
            optix_exp::ErrorDetails errDetails;
            if( const OptixResult result = optix_exp::instanceAabbsCompute( context, stream, iasMotionOptions, instances, numInstances,
                tempBuffer, tempBufferSizeInBytes, outputAabbBuffer,
                outputAabbBufferSizeInBytes, errDetails ) )
            {
                clog.sendError( errDetails );
                return result;
            }
        }
        OPTIX_API_EXCEPTION_CHECK;
    }

    return OPTIX_SUCCESS;
}

#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
extern "C" OptixResult optixVisibilityMapArrayComputeMemoryUsage( OptixDeviceContext                       contextAPI,
                                                                  const OptixVisibilityMapArrayBuildInput* buildInput,
                                                                  OptixMicromeshBufferSizes*               bufferSizes )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::VISIBILITY_MAP_ARRAY_COMPUTE_MEMORY_USAGE );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( buildInput );
    OPTIX_CHECK_NULL_ARGUMENT( bufferSizes );

    try
    {
        optix_exp::ErrorDetails errDetails;
        if( const OptixResult result = visibilityMapArrayComputeMemoryUsage( context, buildInput, bufferSizes, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixVisibilityMapArrayBuild( OptixDeviceContext                       contextAPI,
                                                     LWstream                                 stream,
                                                     const OptixVisibilityMapArrayBuildInput* buildInput,
                                                     const OptixMicromeshBuffers*             buffers,
                                                     const OptixMicromeshEmitDesc*            emittedProperties,
                                                     unsigned int                             numEmittedProperties )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::VISIBILITY_MAP_ARRAY_BUILD );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( buildInput );
    OPTIX_CHECK_NULL_ARGUMENT( buffers );

    try
    {
        optix_exp::ErrorDetails errDetails;
        if( const OptixResult result = visibilityMapArrayBuild( context, stream, buildInput, buffers, emittedProperties, numEmittedProperties, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixDisplacedMicromeshArrayComputeMemoryUsage( OptixDeviceContext contextAPI,
                                                                       const OptixDisplacedMicromeshArrayBuildInput* buildInput,
                                                                       OptixMicromeshBufferSizes* bufferSizes )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::DISPLACED_MICROMESH_ARRAY_COMPUTE_MEMORY_USAGE );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( buildInput );
    OPTIX_CHECK_NULL_ARGUMENT( bufferSizes );

    try
    {
        optix_exp::ErrorDetails errDetails;
        if( const OptixResult result = displacedMicromeshArrayComputeMemoryUsage( context, buildInput, bufferSizes, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixDisplacedMicromeshArrayBuild( OptixDeviceContext                            contextAPI,
                                                          LWstream                                      stream,
                                                          const OptixDisplacedMicromeshArrayBuildInput* buildInput,
                                                          const OptixMicromeshBuffers*                  buffers,
                                                          const OptixMicromeshEmitDesc* emittedProperties,
                                                          unsigned int                  numEmittedProperties )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::DISPLACED_MICROMESH_ARRAY_BUILD );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( buildInput );
    OPTIX_CHECK_NULL_ARGUMENT( buffers );

    try
    {
        optix_exp::ErrorDetails errDetails;
        if( const OptixResult result = displacedMicromeshArrayBuild( context, stream, buildInput, buffers,
                                                                     emittedProperties, numEmittedProperties, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
