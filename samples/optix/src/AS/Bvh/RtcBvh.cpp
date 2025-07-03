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

#include <src/AS/Bvh/RtcBvh.h>

#include <corelib/math/MathUtil.h>
#include <src/LWCA/Memory.h>
#include <src/Context/Context.h>
#include <src/Context/RTCore.h>
#include <src/Context/SharedProgramManager.h>
#include <src/Device/LWDADevice.h>
#include <src/Device/DeviceManager.h>
#include <src/ExelwtionStrategy/CORTTypes.h>
#include <src/Memory/MemoryManager.h>
#include <src/Objects/Acceleration.h>
#include <src/Objects/Group.h>
#include <src/Util/ContainerAlgorithm.h>
#include <src/Util/Enum2String.h>

#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/system/Knobs.h>

// clang-format off
namespace {
    Knob<bool> k_fastTrace( RT_DSTRING( "acceleration.fastTrace" ), false, RT_DSTRING( "Set the build flag for the bvh builder. Increases build time, should decrease trace time. Fast trace takes precedence over fast build." ) );
    Knob<bool> k_fastBuild( RT_DSTRING( "acceleration.fastBuild" ), false, RT_DSTRING( "Set the build flag for the bvh builder. Decreases build time, should increase trace time. Fast trace takes precedence over fast build." ) );
    Knob<bool> k_lowMemory( RT_DSTRING( "acceleration.lowMemory" ), false, RT_DSTRING( "Set the build flag for the bvh builder. Decreases peak memory consumption for bvh build." ) );
}
// clang-format on

using namespace optix;
using namespace corelib;
using namespace prodlib;

static RtcVertexFormat rtcoreVertexFormat( RTformat format )
{
    switch( format )
    {
        case RT_FORMAT_FLOAT3:
            return RTC_VERTEX_FORMAT_FLOAT3;
        case RT_FORMAT_HALF3:
            return RTC_VERTEX_FORMAT_HALF3;
        case RT_FORMAT_FLOAT2:
            return RTC_VERTEX_FORMAT_FLOAT2;
        case RT_FORMAT_HALF2:
            return RTC_VERTEX_FORMAT_HALF2;
        default:
            // this should be catched earlier
            throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid position format for triangle build: ", format2string( format ) );
            break;
    }
}

RtcAccelType rtcoreAccelType( RtcBvhAccelType accelType, int motionSteps )
{
    switch( accelType )
    {
        case RTC_BVH_ACCEL_TYPE_NOACCEL:
            if( motionSteps > 1 )
                return RTC_ACCEL_TYPE_MBVH2;
            return RTC_ACCEL_TYPE_NOACCEL;
        case RTC_BVH_ACCEL_TYPE_BVH2:
            if( motionSteps > 1 )
                return RTC_ACCEL_TYPE_MBVH2;
            return RTC_ACCEL_TYPE_BVH2;
        case RTC_BVH_ACCEL_TYPE_BVH8:
            if( motionSteps > 1 )
                throw IlwalidValue( RT_EXCEPTION_INFO, "Accel build type 'BVH8' does not support motion" );
            return RTC_ACCEL_TYPE_BVH8;
        case RTC_BVH_ACCEL_TYPE_TTU:
            if( motionSteps > 1 )
                return RTC_ACCEL_TYPE_MTTU;
            return RTC_ACCEL_TYPE_TTU;
        default:
            // this should be catched earlier
            throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid accel build type" );
    };
}

RtcBvh::RtcBvh( Acceleration* accel, bool isGeometryGroup, RtcBvhAccelType type )
    : Builder( accel, isGeometryGroup )
    , m_accelType( type )
    , m_bvhSize( 0 )
{
    BufferDimensions dims( RT_FORMAT_UNSIGNED_BYTE, sizeof( unsigned char ), 0, 0, 0, 0 );
    m_accelBuffer     = accel->getContext()->getMemoryManager()->allocateMBuffer( dims, MBufferPolicy::gpuLocal, this );
    m_traversableType = RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL;
}

RtcBvh::~RtcBvh()
{
    if( m_event.get() )
        m_event.destroy();
}

//------------------------------------------------------------------------------
void RtcBvh::setRtcAccelType( RtcBvhAccelType type )
{
    m_accelType = type;
}

//------------------------------------------------------------------------------
void RtcBvh::setProperty( const std::string& name, const std::string& value )
{
    if( name == "compact" )
        m_allowCompaction = corelib::from_string<bool>( value );
}

//------------------------------------------------------------------------------
Program* RtcBvh::getVisitProgram( bool hasMotionAabbs, bool bakePrimitiveEntities )
{
    SharedProgramManager* spm   = m_acceleration->getContext()->getSharedProgramManager();
    Program*              visit = spm->getRtcBvhDummyTraverserProgram();
    return visit;
}

//------------------------------------------------------------------------------
Program* RtcBvh::getBoundingBoxProgram( bool hasMotionAabbs )
{
    SharedProgramManager* spm    = m_acceleration->getContext()->getSharedProgramManager();
    Program*              bounds = spm->getBoundsRuntimeProgram( "bounds_rtcbvh", false, hasMotionAabbs );
    return bounds;
}

//------------------------------------------------------------------------------
size_t RtcBvh::setupBuffers( const BuildParameters&                            params,
                             const RtcAccelOptions&                            options,
                             const std::vector<RtcBuildInput>&                 buildInputs,
                             const std::vector<const RtcBuildInputOverrides*>* overrides )
{
    Context* context = m_acceleration->getContext();
    // TODO(jbigler) what to do when we have mixed devices
    RtcDeviceContext devContext =
        deviceCast<LWDADevice>( m_acceleration->getContext()->getDeviceManager()->allDevices()[params.buildDevices[0]] )->rtcContext();
    RtcAccelBufferSizes bufferSizes;
    context->getRTCore()->accelComputeMemoryUsage( devContext, &options, buildInputs.size(), buildInputs.data(),
                                                   overrides ? overrides->data() : nullptr, &bufferSizes );

    RT_ASSERT( !buildInputs.empty() );
    if( buildInputs[0].type == RTC_BUILD_INPUT_TYPE_INSTANCE_ARRAY || buildInputs[0].type == RTC_BUILD_INPUT_TYPE_INSTANCE_POINTERS )
        m_traversableType = RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL;
    else
        m_traversableType = RTC_TRAVERSABLE_TYPE_BOTTOM_LEVEL_ACCEL;

    if( !options.refit )
    {
        // make sure we can get an 16-byte aligned address to store the root (motion) aabb of the bvh
        m_bvhSize     = roundUp( (size_t)bufferSizes.outputSizeInBytes, size_t( 16 ) );
        m_motionSteps = params.motionSteps;
        BufferDimensions dims( RT_FORMAT_UNSIGNED_BYTE, sizeof( unsigned char ), 1,
                               m_bvhSize + m_motionSteps * sizeof( RtcEmittedAccelPropertyAabb ), 1, 1 );
        context->getMemoryManager()->changeSize( m_accelBuffer, dims );
    }
    else
    {
        RT_ASSERT( m_accelBuffer->getDimensions().width() > 0 );
    }

    size_t tempSize = ( options.refit ? bufferSizes.tempUpdateSizeInBytes : bufferSizes.tempSizeInBytes )
                      + sizeof( size_t );  // additional mem to store compaction size
    tempSize = roundUp( tempSize, size_t( 16 ) );
    return tempSize;
}

bool geometryInstanceDataNeedsOverride( const GeometryInstanceData& gid )
{
    const GeometryTriangles* gt = gid.getTriangles();
    return ( gt && gt->getMaterialCount() > 1 ) || gid.g->getPrimitiveIndexOffset();
}

void countInputs( size_t& numInputs, size_t& numInputOverrides, const std::vector<GeometryInstanceData>& gidata )
{
    numInputs         = 0;
    numInputOverrides = 0;
    for( size_t i = 0; i < gidata.size(); ++i )
    {
        const GeometryInstanceData& gid = gidata[i];
        numInputs += gid.getMaterialCount();
        if( geometryInstanceDataNeedsOverride( gid ) )
            numInputOverrides++;
    }
}

BuildSetupRequest RtcBvh::setupForBuild( const BuildParameters&                   params,
                                         unsigned int                             totalPrims,
                                         const std::vector<GeometryInstanceData>& gidata )
{
    size_t numInputs         = 0;
    size_t numInputOverrides = 0;
    countInputs( numInputs, numInputOverrides, gidata );

    if( params.motionSteps > 1
        && !( m_accelType == RTC_BVH_ACCEL_TYPE_TTU || m_accelType == RTC_BVH_ACCEL_TYPE_BVH2 || m_accelType == RTC_BVH_ACCEL_TYPE_NOACCEL ) )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Motion steps > 1 not supported in this builder" );
    // temporary extra check, bug 2196236
    if( params.motionSteps > 1 && m_accelType == RTC_BVH_ACCEL_TYPE_NOACCEL )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Motion steps > 1 not supported in this builder" );
    if( params.motionSteps > 1 && !m_acceleration->getContext()->RtxUniversalTraversalEnabled() )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Motion steps > 1 requires utrav traversal override" );

    std::vector<RtcBuildInput>                 buildInputs( numInputs );
    std::vector<const RtcBuildInputOverrides*> buildInputOverridePointers( numInputs, nullptr );
    std::vector<RtcBuildInputOverrides>        buildInputOverrides( numInputOverrides );

    size_t inputIndex         = 0;
    size_t inputOverrideIndex = 0;
    for( size_t i = 0; i < gidata.size(); ++i )
    {
        const GeometryInstanceData& gid = gidata[i];
        const GeometryTriangles*    gt  = gid.getTriangles();

        memset(&buildInputs[inputIndex], 0, sizeof(RtcBuildInput));
        buildInputs[inputIndex].type  = RTC_BUILD_INPUT_TYPE_MOTION_AABB_ARRAYS;
        RtcBuildInputAabbArray& aabbs = buildInputs[inputIndex].aabbArray;
        aabbs.aabbBuffer              = reinterpret_cast<RtcGpuVA>( nullptr );
        aabbs.numAabbs                = gidata[inputIndex].primCount;
        aabbs.strideInBytes           = sizeof( cort::Aabb ) * params.motionSteps;
        aabbs.flags                   = gidata[inputIndex].flags;
        aabbs.numSubBoxes             = 0;
        static_assert( sizeof( cort::Aabb ) == sizeof( RtcAabb ), "size difference is bad" );

        if( geometryInstanceDataNeedsOverride( gid ) )
        {
            buildInputOverrides[inputOverrideIndex].primitiveIndexBias   = gid.g->getPrimitiveIndexOffset();
            buildInputOverrides[inputOverrideIndex].numGeometries        = gid.getMaterialCount();
            buildInputOverrides[inputOverrideIndex].geometryOffsetBuffer = reinterpret_cast<RtcGpuVA>( nullptr );

            if( gt )
            {
                buildInputOverrides[inputOverrideIndex].geometryOffsetStrideInBytes = gt->getMaterialIndexByteStride();
                buildInputOverrides[inputOverrideIndex].geometryOffsetSizeInBytes =
                    gt->getMaterialIndexFormat() == RT_FORMAT_UNSIGNED_INT ?
                        4 :
                        ( gt->getMaterialIndexFormat() == RT_FORMAT_UNSIGNED_SHORT ? 2 : 1 );


                aabbs.flags = gt->getGeometryFlags()[0];

                for( size_t gi = 1; gi < buildInputOverrides[inputOverrideIndex].numGeometries; ++gi )
                {
                    buildInputs[inputIndex + gi]                    = buildInputs[inputIndex];
                    buildInputs[inputIndex + gi].aabbArray.numAabbs = 0;

                    if( gt->getGeometryFlags().empty() )
                    {
                        Rtlw32 flags                                 = gt->getGeometryFlags()[gi];
                        buildInputs[inputIndex + gi].aabbArray.flags = flags;
                    }
                }
            }

            buildInputOverridePointers[inputIndex] = &buildInputOverrides[inputOverrideIndex];
            inputOverrideIndex++;
        }

        inputIndex += gid.getMaterialCount();
    }
    if( gidata.empty() )
    {
        buildInputs.resize( 1 );
        memset(&buildInputs[0], 0, sizeof(RtcBuildInput));
        buildInputs[0].type           = RTC_BUILD_INPUT_TYPE_MOTION_AABB_ARRAYS;
        RtcBuildInputAabbArray& aabbs = buildInputs[0].aabbArray;
        aabbs.aabbBuffer              = reinterpret_cast<RtcGpuVA>( nullptr );
        aabbs.numAabbs                = 0;
        aabbs.strideInBytes           = sizeof( cort::Aabb );
        aabbs.flags                   = RTC_GEOMETRY_FLAG_NONE;
        aabbs.numSubBoxes             = 0;
    }

    RtcAccelOptions options;
    setAccelOptions( params, options, false );
    size_t tempSize = setupBuffers( params, options, buildInputs, &buildInputOverridePointers );
    return BuildSetupRequest( /*willRefit=*/options.refit, totalPrims, params.motionSteps, /*motionStride=*/0, tempSize,
                              /*needAabbs=*/true, /*needAabbsOnCpu=*/false );
}

namespace {
template <typename T>
void fillCommonRtcTriangleArray( T& tris, const GeometryInstanceData& gid, const TriangleData& td, LWDADevice* lwdaDevice, bool setupForBuild )
{
    if( td.indices )
    {
        tris.numIndices  = gid.primCount * 3;
        tris.numVertices = ( td.vertices->getDimensions().width() * td.vertices->getDimensions().elementSize() )
                           / size_t( td.vertexStride );
    }
    else
    {
        tris.numVertices = gid.primCount * 3;
        tris.numIndices  = 0;
    }
    // NOTE, cannot set vertex buffer! Is motion dependent!
    tris.vertexFormat        = rtcoreVertexFormat( td.positionFormat );
    tris.vertexStrideInBytes = td.vertexStride;
    tris.indexBuffer         = td.indices && !setupForBuild ?
                           ( RtcGpuVA )( td.indices->getAccess( lwdaDevice ).getLinearPtr() + td.indexBufferOffset ) :
                           reinterpret_cast<RtcGpuVA>( nullptr );
    tris.indexStrideInBytes = td.indices ? td.triIndicesStride : 0;
    tris.indexSizeInBytes   = td.indices ? ( td.triIndicesFormat == RT_FORMAT_UNSIGNED_INT3 ? 4 : 2 ) : 0;
    tris.flags              = gid.flags;
    tris.transform          = reinterpret_cast<RtcGpuVA>( nullptr );
}
}

BuildSetupRequest RtcBvh::setupForBuild( const BuildParameters&                   params,
                                         unsigned int                             totalPrims,
                                         const std::vector<GeometryInstanceData>& gidata,
                                         const std::vector<TriangleData>&         tridata )
{
    //if( params.motionSteps > 1 )
    //    throw IlwalidValue( RT_EXCEPTION_INFO, "Motion steps > 1 not supported for triangles." );

    size_t numInputs         = 0;
    size_t numInputOverrides = 0;
    countInputs( numInputs, numInputOverrides, gidata );

    std::vector<RtcBuildInput>                 buildInputs( numInputs );
    std::vector<const RtcBuildInputOverrides*> buildInputOverridePointers( numInputs, nullptr );
    std::vector<RtcBuildInputOverrides>        buildInputOverrides( numInputOverrides );
    const bool hasMotion = ( params.motionSteps >= 2 );  // allow 0 or 1 to disable motion

    size_t inputIndex         = 0;
    size_t inputOverrideIndex = 0;
    for( size_t i = 0; i < gidata.size(); ++i )
    {
        const GeometryInstanceData& gid = gidata[i];
        const TriangleData&         td  = tridata[i];
        const GeometryTriangles&    gt  = *gid.getTriangles();  // must have valid geometry triangles

        memset(&buildInputs[inputIndex], 0, sizeof(RtcBuildInput));
        buildInputs[inputIndex].type = hasMotion ? RTC_BUILD_INPUT_TYPE_MOTION_TRIANGLE_ARRAYS : RTC_BUILD_INPUT_TYPE_TRIANGLE_ARRAY;
        RtcBuildInputTriangleArray& tris = buildInputs[inputIndex].triangleArray;
        fillCommonRtcTriangleArray( tris, gid, td, nullptr, true );
        tris.vertexBuffer = reinterpret_cast<RtcGpuVA>( nullptr );

        if( geometryInstanceDataNeedsOverride( gid ) )
        {
            buildInputOverrides[inputOverrideIndex].numGeometries               = gt.getMaterialCount();
            buildInputOverrides[inputOverrideIndex].geometryOffsetBuffer        = reinterpret_cast<RtcGpuVA>( nullptr );
            buildInputOverrides[inputOverrideIndex].geometryOffsetStrideInBytes = gt.getMaterialIndexByteStride();
            buildInputOverrides[inputOverrideIndex].geometryOffsetSizeInBytes =
                gt.getMaterialIndexFormat() == RT_FORMAT_UNSIGNED_INT ?
                    4 :
                    ( gt.getMaterialIndexFormat() == RT_FORMAT_UNSIGNED_SHORT ? 2 : 1 );
            buildInputOverrides[inputOverrideIndex].primitiveIndexBias = gt.getPrimitiveIndexOffset();

            for( size_t gi = 0; gi < buildInputOverrides[inputOverrideIndex].numGeometries; ++gi )
            {
                if( gi > 0 )
                {
                    buildInputs[inputIndex + gi]                           = buildInputs[inputIndex];
                    buildInputs[inputIndex + gi].triangleArray.numVertices = 0;
                    buildInputs[inputIndex + gi].triangleArray.numIndices  = 0;
                }

                Rtlw32 flags                                     = gt.getGeometryFlags()[gi];
                buildInputs[inputIndex + gi].triangleArray.flags = flags;
            }

            buildInputOverridePointers[inputIndex] = &buildInputOverrides[inputOverrideIndex];
            inputOverrideIndex++;
        }

        inputIndex += gt.getMaterialCount();
    }

    if( gidata.empty() )
    {
        buildInputs.resize( 1 );
        memset(&buildInputs[0], 0, sizeof(RtcBuildInput));
        buildInputs[0].type              = RTC_BUILD_INPUT_TYPE_TRIANGLE_ARRAY;
        RtcBuildInputTriangleArray& tris = buildInputs[0].triangleArray;
        tris.numVertices                 = 0;
        tris.numIndices                  = 0;
        tris.vertexBuffer                = reinterpret_cast<RtcGpuVA>( nullptr );
        tris.vertexFormat                = RTC_VERTEX_FORMAT_FLOAT3;
        tris.vertexStrideInBytes         = sizeof( float3 );
        tris.indexBuffer                 = reinterpret_cast<RtcGpuVA>( nullptr );
        tris.indexStrideInBytes          = sizeof( int3 );
        tris.indexSizeInBytes            = sizeof( int );
        tris.flags                       = RTC_GEOMETRY_FLAG_NONE;
        tris.transform                   = reinterpret_cast<RtcGpuVA>( nullptr );
    }

    RtcAccelOptions options;
    setAccelOptions( params, options, true );
    size_t tempSize = setupBuffers( params, options, buildInputs, &buildInputOverridePointers );

    return BuildSetupRequest( /*willRefit=*/options.refit, totalPrims, params.motionSteps, /*motionStride=*/0, tempSize,
                              /*needAabbs=*/false, /*needAabbsOnCpu=*/false );
}

BuildSetupRequest RtcBvh::setupForBuild( const BuildParameters& params, const GroupData& groupdata )
{
    if( params.motionSteps > 1
        && !( m_accelType == RTC_BVH_ACCEL_TYPE_TTU || m_accelType == RTC_BVH_ACCEL_TYPE_BVH2 || m_accelType == RTC_BVH_ACCEL_TYPE_NOACCEL ) )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Motion steps > 1 not supported in this builder" );
    // temporary extra check, bug 2196236
    if( params.motionSteps > 1 && m_accelType == RTC_BVH_ACCEL_TYPE_NOACCEL )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Motion steps > 1 not supported in this builder" );
    if( params.motionSteps > 1 && !m_acceleration->getContext()->RtxUniversalTraversalEnabled() )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Motion steps > 1 requires utrav traversal override" );

    std::vector<RtcBuildInput> buildInputs( 1 );
    memset(&buildInputs[0], 0, sizeof(RtcBuildInput));
    buildInputs[0].type                  = RTC_BUILD_INPUT_TYPE_INSTANCE_ARRAY;
    RtcBuildInputInstanceArray& instance = buildInputs[0].instanceArray;
    instance.instanceDescs               = reinterpret_cast<RtcGpuVA>( nullptr );
    instance.numInstances                = groupdata.childCount;

    RtcAccelOptions options;
    setAccelOptions( params, options, false );
    // TODO: refitting instances
    options.refit   = false;
    size_t tempSize = setupBuffers( params, options, buildInputs );
    return BuildSetupRequest( /*willRefit=*/false, groupdata.childCount, params.motionSteps, /*motionStride=*/0,
                              tempSize, /*needAabbs=*/false, /*needAabbsOnCpu=*/false );
}

//------------------------------------------------------------------------------
bool RtcBvh::shouldCompact( const BuildParameters& params, const RtcAccelOptions& options )
{
    return m_allowCompaction && m_accelType != RTC_BVH_ACCEL_TYPE_BVH2 && !options.refit;
}

void RtcBvh::copyCompactedSize( RtcAccelBuffers& buffers, LWDADevice* lwdaDevice, bool& compactedSizeCopied )
{
    LWdeviceptr compactedSizePtr = (LWdeviceptr)buffers.temp + buffers.tempSizeInBytes - sizeof( size_t );
    // If this is the first device, copy the compacted size to the host
    if( !compactedSizeCopied )
    {
        // Initiate copy and record the event
        lwdaDevice->makeLwrrent();
        lwca::memcpyDtoHAsync( &m_compactedSize, compactedSizePtr, sizeof( size_t ), lwdaDevice->primaryStream() );
        // This is an asynchronous copy, so you can't read the value until you
        // synchronize on this event.

        m_event = lwca::Event::create();
        m_event.record( lwdaDevice->primaryStream() );

        compactedSizeCopied = true;
    }
}

void RtcBvh::build( const BuildParameters& params, const BuildSetup& setup, const std::vector<GeometryInstanceData>& gidata )
{
    RT_ASSERT( params.motionSteps == m_motionSteps );

    Context* context = m_acceleration->getContext();

    MemoryManager* mm = context->getMemoryManager();

    size_t numInputs         = 0;
    size_t numInputOverrides = 0;
    countInputs( numInputs, numInputOverrides, gidata );

    RT_ASSERT( m_traversableType == RTC_TRAVERSABLE_TYPE_BOTTOM_LEVEL_ACCEL );
    RT_ASSERT( m_motionSteps == params.motionSteps );

    bool compactedSizeCopied = false;
    for( const auto& perDevice : setup.perDevice )
    {
        LWDADevice*    lwdaDevice  = perDevice.buildDevice;
        RtcCommandList commandList = lwdaDevice->primaryRtcCommandList();
        lwdaDevice->makeLwrrent();

        RtcAccelBuffers buffers;
        buffers.input             = (RtcGpuVA)0;
        buffers.output            = (RtcGpuVA)mm->getWritablePointerForBuild( m_accelBuffer, lwdaDevice, false );
        buffers.outputSizeInBytes = m_bvhSize;
        buffers.temp              = (RtcGpuVA)perDevice.deviceTempPtr;
        buffers.tempSizeInBytes   = perDevice.tempSize - sizeof( size_t );

        std::vector<std::vector<RtcGpuVA>> tempPerGTBufferList;

        std::vector<RtcBuildInput>                 buildInputs( numInputs );
        std::vector<const RtcBuildInputOverrides*> buildInputOverridePointers( numInputs, nullptr );
        std::vector<RtcBuildInputOverrides>        buildInputOverrides( numInputOverrides );

        size_t inputIndex         = 0;
        size_t inputOverrideIndex = 0;
        for( size_t i = 0; i < gidata.size(); ++i )
        {
            const GeometryInstanceData& gid = gidata[i];
            const GeometryTriangles*    gt  = gid.getTriangles();

            std::vector<RtcGpuVA> aabbBuffers( params.motionSteps );

            // motion aabb are stored interleaved in device memory.
            // the per motion step aabb buffers are overlayed by offseting the buffer pointer and using a stride to skip the aabbs of the other interleaved motion steps
            LWdeviceptr aabbBuffer = ( RtcGpuVA )( perDevice.deviceAabbPtr + gidata[i].primStart * params.motionSteps );

            for( unsigned int j = 0; j < params.motionSteps; ++j )
                aabbBuffers[j]  = aabbBuffer + j * sizeof( float ) * 6;

            tempPerGTBufferList.emplace_back( aabbBuffers );

            memset(&buildInputs[inputIndex], 0, sizeof(RtcBuildInput));
            buildInputs[inputIndex].type  = RTC_BUILD_INPUT_TYPE_MOTION_AABB_ARRAYS;
            RtcBuildInputAabbArray& aabbs = buildInputs[inputIndex].aabbArray;
            aabbs.aabbBuffer              = (RtcGpuVA)tempPerGTBufferList.back().data();
            aabbs.numAabbs                = gidata[i].primCount;
            aabbs.strideInBytes           = sizeof( cort::Aabb ) * params.motionSteps;
            aabbs.flags                   = gidata[i].flags;
            aabbs.numSubBoxes             = 0;

            if( geometryInstanceDataNeedsOverride( gid ) )
            {
                buildInputOverrides[inputOverrideIndex].primitiveIndexBias = gid.g->getPrimitiveIndexOffset();
                buildInputOverrides[inputOverrideIndex].numGeometries      = gid.getMaterialCount();

                if( gt )
                {
                    buildInputOverrides[inputOverrideIndex].geometryOffsetBuffer =
                        gt->getMaterialIndexBuffer() ?
                            ( RtcGpuVA )( gt->getMaterialIndexBuffer()->getMBuffer()->getAccess( lwdaDevice ).getLinearPtr()
                                          + gt->getMaterialIndexBufferByteOffset() ) :
                            reinterpret_cast<RtcGpuVA>( nullptr );
                    buildInputOverrides[inputOverrideIndex].geometryOffsetStrideInBytes = gt->getMaterialIndexByteStride();
                    buildInputOverrides[inputOverrideIndex].geometryOffsetSizeInBytes =
                        gt->getMaterialIndexFormat() == RT_FORMAT_UNSIGNED_INT ?
                            4 :
                            ( gt->getMaterialIndexFormat() == RT_FORMAT_UNSIGNED_SHORT ? 2 : 1 );

                    for( size_t gi = 0; gi < buildInputOverrides[inputOverrideIndex].numGeometries; ++gi )
                    {
                        if( gi > 0 )
                        {
                            buildInputs[inputIndex + gi]                    = buildInputs[inputIndex];
                            buildInputs[inputIndex + gi].aabbArray.numAabbs = 0;
                        }

                        buildInputs[inputIndex + gi].aabbArray.flags = gt->getGeometryFlags()[gi];
                    }
                }
                else
                {
                    buildInputOverrides[inputOverrideIndex].geometryOffsetBuffer = 0;
                }

                buildInputOverridePointers[inputIndex] = &buildInputOverrides[inputOverrideIndex];

                inputOverrideIndex++;
            }

            inputIndex += gid.getMaterialCount();
        }
        if( buildInputs.empty() )
        {
            buildInputs.resize( 1 );
            memset(&buildInputs[0], 0, sizeof(RtcBuildInput));
            buildInputs[0].type           = RTC_BUILD_INPUT_TYPE_MOTION_AABB_ARRAYS;
            RtcBuildInputAabbArray& aabbs = buildInputs[0].aabbArray;
            aabbs.aabbBuffer              = reinterpret_cast<RtcGpuVA>( nullptr );
            aabbs.numAabbs                = 0;
            aabbs.strideInBytes           = sizeof( cort::Aabb );
            aabbs.flags                   = RTC_GEOMETRY_FLAG_NONE;
            aabbs.numSubBoxes             = 0;
        }

        RtcAccelOptions options;
        setAccelOptions( params, options, false );
        if( shouldCompact( params, options ) )
        {
            RtcAccelEmitDesc emit;
            emit.type = RTC_PROPERTY_TYPE_COMPACTED_SIZE;
            // note, tempSizeInBytes is a multiple of 16!
            // hence resultVA is 8-byte aligned assuming alignment of the temp buffer
            emit.resultVA = ( RtcGpuVA )( perDevice.deviceTempPtr + buffers.tempSizeInBytes - sizeof( size_t ) );
            context->getRTCore()->accelBuild( commandList, &options, buildInputs.size(), buildInputs.data(),
                                              buildInputOverridePointers.data(), &buffers, 1, &emit );
            copyCompactedSize( buffers, lwdaDevice, compactedSizeCopied );
        }
        else
        {
            context->getRTCore()->accelBuild( commandList, &options, buildInputs.size(), buildInputs.data(),
                                              buildInputOverridePointers.data(), &buffers, 0, nullptr );
        }

        m_lastBuildSupportsRefit      = options.buildFlags & RTC_BUILD_FLAG_ALLOW_UPDATE;
        m_lastBuildHasUniversalFormat = options.useUniversalFormat;

        context->getRTCore()->accelEmitProperties( commandList, &buffers.output, 1,
                                                   m_motionSteps > 1 ? RTC_PROPERTY_TYPE_MOTION_AABBS : RTC_PROPERTY_TYPE_AABB,
                                                   buffers.output + m_bvhSize, sizeof( RtcEmittedAccelPropertyAabb ) * m_motionSteps );
#if 0
        // debug
        lwca::Context::synchronize();
        RtcEmittedAccelPropertyAabb* aabbs = new RtcEmittedAccelPropertyAabb[m_motionSteps];
        lwca::memcpyDtoH(aabbs, (LWdeviceptr)(buffers.output + buffers.outputSizeInBytes), m_motionSteps*sizeof(RtcEmittedAccelPropertyAabb));
        for(unsigned int i=0; i<m_motionSteps; ++i)
        {
            ltemp << "motion step "<< i << " Aabb = ("<<aabbs[i].Aabb[0]<<", "<<aabbs[i].Aabb[1]<<", "<<aabbs[i].Aabb[2]<<") ("<<aabbs[i].Aabb[3]<<", "<<aabbs[i].Aabb[4]<<", "<<aabbs[i].Aabb[5]<<")\n";
        }
        delete aabbs;
#endif
    }
}


void RtcBvh::build( const BuildParameters&                   params,
                    const BuildSetup&                        setup,
                    const std::vector<GeometryInstanceData>& gidata,
                    const std::vector<TriangleData>&         tridata )
{
    RT_ASSERT( params.motionSteps == m_motionSteps );

    Context* context = m_acceleration->getContext();

    MemoryManager* mm = context->getMemoryManager();

    bool compactedSizeCopied = false;

    size_t numInputs         = 0;
    size_t numInputOverrides = 0;
    countInputs( numInputs, numInputOverrides, gidata );

    RT_ASSERT( m_traversableType == RTC_TRAVERSABLE_TYPE_BOTTOM_LEVEL_ACCEL );
    RT_ASSERT( m_motionSteps == params.motionSteps );

    for( const auto& perDevice : setup.perDevice )
    {
        LWDADevice*    lwdaDevice  = perDevice.buildDevice;
        RtcCommandList commandList = lwdaDevice->primaryRtcCommandList();
        lwdaDevice->makeLwrrent();

        RtcAccelBuffers buffers;
        buffers.input             = (RtcGpuVA)0;
        buffers.output            = (RtcGpuVA)mm->getWritablePointerForBuild( m_accelBuffer, lwdaDevice, false );
        buffers.outputSizeInBytes = m_bvhSize;
        buffers.temp              = (RtcGpuVA)perDevice.deviceTempPtr;
        buffers.tempSizeInBytes   = perDevice.tempSize - sizeof( size_t );

        std::vector<std::vector<RtcGpuVA>> tempPerGTBufferList;

        std::vector<RtcBuildInput>                 buildInputs( numInputs );
        std::vector<const RtcBuildInputOverrides*> buildInputOverridePointers( numInputs, nullptr );
        std::vector<RtcBuildInputOverrides>        buildInputOverrides( numInputOverrides );
        const bool hasMotion = ( m_motionSteps >= 2 );  // allow 0 or 1 to disable motion

        size_t inputIndex         = 0;
        size_t inputOverrideIndex = 0;
        for( size_t i = 0; i < gidata.size(); ++i )
        {
            const GeometryInstanceData& gid = gidata[i];
            const TriangleData&         td  = tridata[i];
            const GeometryTriangles&    gt  = *gid.getTriangles();

            memset(&buildInputs[inputIndex], 0, sizeof(RtcBuildInput));

            unsigned int gtMotionSteps = gt.getMotionSteps();
            RT_ASSERT( m_motionSteps == gtMotionSteps );

            RtcBuildInputTriangleArray& tris = buildInputs[inputIndex].triangleArray;
            fillCommonRtcTriangleArray( tris, gid, td, lwdaDevice, false );
            if( hasMotion )
            {
                buildInputs[inputIndex].type = RTC_BUILD_INPUT_TYPE_MOTION_TRIANGLE_ARRAYS;
                std::vector<RtcGpuVA> vertexBuffers( gtMotionSteps );
                if( gt.hasMultiBufferMotion() )
                {
                    // user provided a separate buffer for each motion step
                    const auto offset = td.vertexBufferOffset;  // lambda capture initializers only available with C++14
                    algorithm::transform( gt.getVertexBuffers(), vertexBuffers.begin(),
                                          [lwdaDevice, offset]( const GeometryTriangles::TrianglesPtr& t ) {
                                              return (RtcGpuVA)t->getMBuffer()->getAccess( lwdaDevice ).getLinearPtr() + offset;
                                          } );
                }
                else
                {
                    // user provided motion vertices are stored interleaved in device memory.
                    // the per motion step vertex buffers are overlayed by offseting the buffer pointer and using a stride to skip the vertices of the other interleaved motion steps
                    vertexBuffers[0] = ( RtcGpuVA )( td.vertices->getAccess( lwdaDevice ).getLinearPtr() + td.vertexBufferOffset );
                    for( unsigned int j = 1; j < gtMotionSteps; ++j )
                    {
                        vertexBuffers[j] = vertexBuffers[0] + j * gt.getVertexMotionByteStride();
                    }
                }
                tempPerGTBufferList.emplace_back( vertexBuffers );
                tris.vertexBuffer = (RtcGpuVA)tempPerGTBufferList.back().data();
            }
            else
            {
                buildInputs[inputIndex].type = RTC_BUILD_INPUT_TYPE_TRIANGLE_ARRAY;
                tris.vertexBuffer = ( RtcGpuVA )( td.vertices->getAccess( lwdaDevice ).getLinearPtr() + td.vertexBufferOffset );
            }

            if( geometryInstanceDataNeedsOverride( gid ) )
            {
                buildInputOverrides[inputOverrideIndex].numGeometries = gt.getMaterialCount();
                buildInputOverrides[inputOverrideIndex].geometryOffsetBuffer =
                    gt.getMaterialIndexBuffer() ?
                        ( RtcGpuVA )( gt.getMaterialIndexBuffer()->getMBuffer()->getAccess( lwdaDevice ).getLinearPtr()
                                      + gt.getMaterialIndexBufferByteOffset() ) :
                        reinterpret_cast<RtcGpuVA>( nullptr );
                buildInputOverrides[inputOverrideIndex].geometryOffsetStrideInBytes = gt.getMaterialIndexByteStride();
                buildInputOverrides[inputOverrideIndex].geometryOffsetSizeInBytes =
                    gt.getMaterialIndexFormat() == RT_FORMAT_UNSIGNED_INT ?
                        4 :
                        ( gt.getMaterialIndexFormat() == RT_FORMAT_UNSIGNED_SHORT ? 2 : 1 );
                buildInputOverrides[inputOverrideIndex].primitiveIndexBias = gt.getPrimitiveIndexOffset();

                for( size_t gi = 0; gi < buildInputOverrides[inputOverrideIndex].numGeometries; ++gi )
                {
                    if( gi > 0 )
                    {
                        buildInputs[inputIndex + gi]                           = buildInputs[inputIndex];
                        buildInputs[inputIndex + gi].triangleArray.numVertices = 0;
                        buildInputs[inputIndex + gi].triangleArray.numIndices  = 0;
                    }

                    Rtlw32 flags = gt.getGeometryFlags()[gi];

                    buildInputs[inputIndex + gi].triangleArray.flags = flags;
                }

                buildInputOverridePointers[inputIndex] = &buildInputOverrides[inputOverrideIndex];
                inputOverrideIndex++;
            }

            inputIndex += gt.getMaterialCount();
        }
        if( gidata.empty() )
        {
            buildInputs.resize( 1 );
            buildInputs[0].type              = RTC_BUILD_INPUT_TYPE_TRIANGLE_ARRAY;
            RtcBuildInputTriangleArray& tris = buildInputs[0].triangleArray;
            tris.numVertices                 = 0;
            tris.numIndices                  = 0;
            tris.vertexBuffer                = reinterpret_cast<RtcGpuVA>( nullptr );
            tris.vertexFormat                = RTC_VERTEX_FORMAT_FLOAT3;
            tris.vertexStrideInBytes         = sizeof( float3 );
            tris.indexBuffer                 = reinterpret_cast<RtcGpuVA>( nullptr );
            tris.indexStrideInBytes          = sizeof( int3 );
            tris.indexSizeInBytes            = sizeof( int );
            tris.flags                       = RTC_GEOMETRY_FLAG_NONE;
            tris.transform                   = reinterpret_cast<RtcGpuVA>( nullptr );
        }

        RtcAccelOptions options;
        setAccelOptions( params, options, true );
        if( shouldCompact( params, options ) )
        {
            RtcAccelEmitDesc emit;
            emit.type = RTC_PROPERTY_TYPE_COMPACTED_SIZE;
            // note, tempSizeInBytes is a multiple of 16!
            // hence resultVA is 8-byte aligned assuming alignment of the temp buffer
            emit.resultVA = ( RtcGpuVA )( perDevice.deviceTempPtr + buffers.tempSizeInBytes - sizeof( size_t ) );
            context->getRTCore()->accelBuild( commandList, &options, buildInputs.size(), buildInputs.data(),
                                              buildInputOverridePointers.data(), &buffers, 1, &emit );
            copyCompactedSize( buffers, lwdaDevice, compactedSizeCopied );
        }
        else
        {
            context->getRTCore()->accelBuild( commandList, &options, buildInputs.size(), buildInputs.data(),
                                              buildInputOverridePointers.data(), &buffers, 0, nullptr );
        }

        m_lastBuildSupportsRefit      = options.buildFlags & RTC_BUILD_FLAG_ALLOW_UPDATE;
        m_lastBuildHasUniversalFormat = options.useUniversalFormat;

        context->getRTCore()->accelEmitProperties( commandList, &buffers.output, 1,
                                                   m_motionSteps > 1 ? RTC_PROPERTY_TYPE_MOTION_AABBS : RTC_PROPERTY_TYPE_AABB,
                                                   buffers.output + m_bvhSize, sizeof( RtcEmittedAccelPropertyAabb ) * m_motionSteps );

#if 0
        // debug
        lwca::Context::synchronize();
        RtcEmittedAccelPropertyAabb* aabbs = new RtcEmittedAccelPropertyAabb[m_motionSteps];
        lwca::memcpyDtoH(aabbs, (LWdeviceptr)(buffers.output + buffers.outputSizeInBytes), m_motionSteps*sizeof(RtcEmittedAccelPropertyAabb));
        for(unsigned int i=0; i<m_motionSteps; ++i)
        {
            ltemp << "motion step "<< i << " Aabb = ("<<aabbs[i].Aabb[0]<<", "<<aabbs[i].Aabb[1]<<", "<<aabbs[i].Aabb[2]<<") ("<<aabbs[i].Aabb[3]<<", "<<aabbs[i].Aabb[4]<<", "<<aabbs[i].Aabb[5]<<")\n";
        }
        delete aabbs;
#endif
    }
}

void RtcBvh::build( const BuildParameters& params, const BuildSetup& setup, const GroupData& groupData )
{
    RT_ASSERT( params.motionSteps == m_motionSteps );

    Context* context = m_acceleration->getContext();

    MemoryManager* mm = context->getMemoryManager();

    // Sync instance descriptors for rtcore if necessary. Array is
    // indexed by the allDeviceIndex but valid only for active
    // devices.
    Group* grp = managedObjectCast<Group>( m_acceleration->getAnyAbstractGroup() );
    RT_ASSERT( grp != nullptr );
    grp->syncInstanceDescriptors();

    RT_ASSERT( m_traversableType == RTC_TRAVERSABLE_TYPE_TOP_LEVEL_ACCEL );
    RT_ASSERT( m_motionSteps == params.motionSteps );

    for( const auto& perDevice : setup.perDevice )
    {
        LWDADevice*    lwdaDevice  = perDevice.buildDevice;
        RtcCommandList commandList = lwdaDevice->primaryRtcCommandList();
        lwdaDevice->makeLwrrent();

        RtcInstance* id = groupData.childCount == 0 ? nullptr : grp->getInstanceDescriptorDevicePtr( lwdaDevice );

        RtcAccelBuffers buffers;
        buffers.input             = (RtcGpuVA)0;
        buffers.output            = (RtcGpuVA)mm->getWritablePointerForBuild( m_accelBuffer, lwdaDevice, false );
        buffers.outputSizeInBytes = m_bvhSize;
        buffers.temp              = (RtcGpuVA)perDevice.deviceTempPtr;
        buffers.tempSizeInBytes   = perDevice.tempSize;

        RtcBuildInput buildInput = {};
        buildInput.type                      = RTC_BUILD_INPUT_TYPE_INSTANCE_ARRAY;
        RtcBuildInputInstanceArray& instance = buildInput.instanceArray;
        instance.instanceDescs               = (RtcGpuVA)id;
        // Aabb count is assumed to be childCount * motionSteps
        instance.numInstances = groupData.childCount;

        RtcAccelOptions options;
        setAccelOptions( params, options, false );
        // TODO: refitting instances
        options.refit = false;

        context->getRTCore()->accelBuild( commandList, &options, 1, &buildInput, nullptr, &buffers, 0, nullptr );

        m_lastBuildSupportsRefit      = options.buildFlags & RTC_BUILD_FLAG_ALLOW_UPDATE;
        m_lastBuildHasUniversalFormat = options.useUniversalFormat;

        // RTC_PROPERTY_TYPE_MOTION_AABBS only works for UTRAV_BVH2 and OPTIX_BVH2 bvh encoding
        context->getRTCore()->accelEmitProperties( commandList, &buffers.output, 1,
                                                   m_motionSteps > 1 ? RTC_PROPERTY_TYPE_MOTION_AABBS : RTC_PROPERTY_TYPE_AABB,
                                                   buffers.output + m_bvhSize, sizeof( RtcEmittedAccelPropertyAabb ) * m_motionSteps );

#if 0
        // debug
        lwca::Context::synchronize();
        RtcEmittedAccelPropertyAabb* aabbs = new RtcEmittedAccelPropertyAabb[m_motionSteps];
        lwca::memcpyDtoH(aabbs, (LWdeviceptr)(buffers.output + buffers.outputSizeInBytes), m_motionSteps*sizeof(RtcEmittedAccelPropertyAabb));
        for(unsigned int i=0; i<m_motionSteps; ++i)
        {
            ltemp << "motion step "<< i << " Aabb = ("<<aabbs[i].Aabb[0]<<", "<<aabbs[i].Aabb[1]<<", "<<aabbs[i].Aabb[2]<<") ("<<aabbs[i].Aabb[3]<<", "<<aabbs[i].Aabb[4]<<", "<<aabbs[i].Aabb[5]<<")\n";
        }
        delete aabbs;
#endif
    }
}

RtcTraversableHandle RtcBvh::computeTraversableHandle( const MAccess& access ) const
{
    if( access.getKind() == MAccess::NONE )
        return 0;

    RtcGpuVA devPtr = 0;

    if( m_accelBuffer->getDimensions().zeroSized() )
        return 0;

    devPtr = reinterpret_cast<RtcGpuVA>( access.getLinearPtr() );

    Context*         context    = m_acceleration->getContext();
    RtcDeviceContext rtcContext = context->getDeviceManager()->primaryLWDADevice()->rtcContext();

    RtcTraversableHandle travHandle = 0;
    if( m_acceleration->getContext()->RtxUniversalTraversalEnabled() )
        context->getRTCore()->colwertPointerToTraversableHandle( rtcContext, devPtr, m_traversableType,
                                                                 rtcoreAccelType( m_accelType, m_motionSteps ), &travHandle );
    else
        travHandle = devPtr;

    return travHandle;
}

RtcTraversableHandle RtcBvh::getTraversableHandle( unsigned int allDeviceIndex ) const
{
    return computeTraversableHandle( m_accelBuffer->getAccess( allDeviceIndex ) );
}

void RtcBvh::eventMBufferMAccessDidChange( const MBuffer* buffer, const Device* device, const MAccess& oldMBA, const MAccess& newMBA )
{
    RtcTraversableHandle newHandle      = computeTraversableHandle( newMBA );
    const unsigned int   allDeviceIndex = device->allDeviceListIndex();
    m_acceleration->traversableHandleDidChange( newHandle, allDeviceIndex );
}

void RtcBvh::setAccelOptions( const BuildParameters& params, RtcAccelOptions& options, bool bakeTriangles )
{
    memset( &options, 0, sizeof( RtcAccelOptions ) );

    options.accelType = rtcoreAccelType( m_accelType, params.motionSteps );
    if( params.refitEnabled )
        options.buildFlags = RTC_BUILD_FLAG_ALLOW_UPDATE;
    else
        options.buildFlags = 0;

    if( k_fastTrace.get() )
        options.buildFlags |= RTC_BUILD_FLAG_PREFER_FAST_TRACE;
    if( k_fastBuild.get() )
        options.buildFlags |= RTC_BUILD_FLAG_PREFER_FAST_BUILD;
    if( m_allowCompaction )
        options.buildFlags |= RTC_BUILD_FLAG_ALLOW_COMPACTION;
    if( k_lowMemory.get() )
        options.buildFlags |= RTC_BUILD_FLAG_LOW_MEMORY;

    const bool useUniversalFormat = m_acceleration->getContext()->RtxUniversalTraversalEnabled();

    options.useUniversalFormat = useUniversalFormat;
    options.refit              = m_lastBuildSupportsRefit && params.refitEnabled && params.shouldRefitThisFrame
                    && ( m_lastBuildHasUniversalFormat == useUniversalFormat );
    options.usePrimBits                = true;
    options.useRemapForPrimBits        = false;
    options.enableBuildReordering      = false;
    options.clampAabbsToValidRange     = true;
    options.bakeTriangles              = bakeTriangles;
    options.highPrecisionMath          = true;
    options.useProvizBuilderStrategies = true;
    options.motionSteps                = params.motionSteps;
    options.motionTimeBegin            = params.motionTimeBegin;
    options.motionTimeEnd              = params.motionTimeEnd;
    options.motionFlags                = 0;
    if( !params.motionBorderModeClampBegin )
        options.motionFlags |= RTC_MOTION_FLAG_START_VANISH;
    if( !params.motionBorderModeClampEnd )
        options.motionFlags |= RTC_MOTION_FLAG_END_VANISH;
}

void RtcBvh::copyMotionAabbs( const std::vector<Builder*>& builders, char* deviceOutputBuffer, LWDADevice* lwdaDevice )
{
    if( builders.empty() )
        return;

    const RtcBvh&  firstBvh = *static_cast<const RtcBvh*>( builders[0] );
    Context*       context  = firstBvh.m_acceleration->getContext();
    MemoryManager* mm       = context->getMemoryManager();

    lwdaDevice->makeLwrrent();
    lwca::Event event = lwca::Event::create();
    event.record( lwdaDevice->primaryStream() );
    for( const Builder* b : builders )
    {
        const RtcBvh* rtcBvh = dynamic_cast<const RtcBvh*>( b );
        char*         bvhAabbs =
            mm->getWritablePointerForBuild( rtcBvh->m_accelBuffer, lwdaDevice, false )
            + ( ( rtcBvh->m_allowCompaction && rtcBvh->m_compactedSize > 0 ) ? rtcBvh->m_compactedSize : rtcBvh->m_bvhSize );
        const size_t byteCount = rtcBvh->m_motionSteps * sizeof( RtcEmittedAccelPropertyAabb );
        lwca::memcpyDtoDAsync( reinterpret_cast<LWdeviceptr>( deviceOutputBuffer ),
                               reinterpret_cast<LWdeviceptr>( bvhAabbs ), byteCount, lwdaDevice->primaryStream() );
        deviceOutputBuffer += byteCount;
    }
    // Make sure the async copy of the aabbs finished
    event.synchronize();
    event.destroy();
}

void RtcBvh::compactBuffers()
{
    if( m_event.get() )  // This event only gets created if compaction is enabled, i.e., if shouldCompact returned true
    {
        // Make sure the async copy of the compacted size finished
        m_event.synchronize();
        m_event.destroy();

        // we are storing additional data behind the bvh, make sure that whatever comes
        // after has proper alignment
        m_compactedSize = roundUp( m_compactedSize, size_t( 16 ) );

        Rtlw64           resultBufferSize = m_compactedSize + m_motionSteps * sizeof( RtcEmittedAccelPropertyAabb );
        BufferDimensions newDims( RT_FORMAT_UNSIGNED_BYTE, sizeof( unsigned char ), 1, resultBufferSize, 1, 1 );

        // Setup variables to capture in lambda scope
        Context*       context = m_acceleration->getContext();
        MemoryManager* mm      = context->getMemoryManager();

        // Setup callback to copy the acceleration structure during compaction
        // this will auto-sync iff there is a copy
        auto copyAccel = [&]( LWDADevice* lwdaDevice, LWdeviceptr sourceBuffer, LWdeviceptr resultBuffer ) -> void {
            lwdaDevice->makeLwrrent();
            RtcResult* returnResult = nullptr;
            context->getRTCore()->accelCopy( lwdaDevice->primaryRtcCommandList(), sourceBuffer, RTC_COPY_MODE_COMPACT,
                                             resultBuffer, resultBufferSize, returnResult );
            lwca::memcpyDtoDAsync( resultBuffer + m_compactedSize, sourceBuffer + m_bvhSize,
                                   m_motionSteps * sizeof( RtcEmittedAccelPropertyAabb ), lwdaDevice->primaryStream() );
        };

        // Compact the MBuffer
        mm->changeSize( m_accelBuffer, newDims, copyAccel );

        // Reflect the updated size
        m_bvhSize = m_compactedSize;
    }
}

//------------------------------------------------------------------------------
void RtcBvh::finalizeAfterBuildPrimitives( const BuildParameters& )
{
    compactBuffers();
}

//------------------------------------------------------------------------------
void RtcBvh::finalizeAfterBuildTriangles( const BuildParameters& )
{
    compactBuffers();
}

//------------------------------------------------------------------------------
void RtcBvh::finalizeAfterBuildGroups( const BuildParameters& )
{
    // No trimming for groups at the moment
}
