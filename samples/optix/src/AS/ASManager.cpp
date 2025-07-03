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

#include <AS/ASManager.h>
#include <AS/Builder.h>
#include <AS/ComputeAabb_ptx_bin.h>
#include <Context/Context.h>
#include <Context/SharedProgramManager.h>
#include <Device/CPUDevice.h>
#include <Device/LWDADevice.h>
#include <Device/Device.h>
#include <Exceptions/ExceptionHelpers.h>
#include <Exceptions/TimeoutException.h>
#include <ExelwtionStrategy/CommonRuntime.h>
#include <ExelwtionStrategy/FrameTask.h>
#include <Memory/MemoryManager.h>
#include <Objects/Acceleration.h>
#include <Objects/Buffer.h>
#include <Objects/Geometry.h>
#include <Objects/GeometryInstance.h>
#include <Objects/GlobalScope.h>
#include <Objects/Group.h>
#include <Objects/Program.h>
#include <Objects/Selector.h>
#include <Objects/Transform.h>
#include <Objects/Variable.h>
#include <Objects/VariableType.h>
#include <Util/ContainerAlgorithm.h>
#include <Util/Metrics.h>
#include <Util/MotionAabb.h>
#include <Util/ResampleMotion.h>

#include <corelib/math/MathUtil.h>
#include <corelib/misc/Cast.h>
#include <corelib/misc/String.h>
#include <corelib/system/System.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/misc/TimeViz.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>
#include <prodlib/system/Thread.h>

#include <rtcore/interface/types.h>

using namespace corelib;
using namespace optix;
using namespace prodlib;

namespace {
// clang-format off
Knob<int>   k_compactionGranularity( RT_DSTRING( "acceleration.compactionGranularity" ), 10000000, RT_DSTRING( "Primitive threshold at which a build/compaction wave of acceleration structures is triggered" ) );
Knob<float> k_initialTempBufferFraction( RT_DSTRING( "asmanager.tempBufferFraction" ), 0.005f, RT_DSTRING( "Temp buffer size as a fraction of total device memory" ) );
Knob<float> k_targetPasses( RT_DSTRING( "asmanager.targetPasses" ), 4, RT_DSTRING( "Number of temp buffer passes for many accels" ) );
// clang-format on
}  // namespace

static const size_t AABB_ALIGNMENT = std::max( 16, RTC_AABB_BUFFER_BYTE_ALIGNMENT );

ASManager::ASManager( Context* context )
    : m_context( context )
{
}

ASManager::~ASManager()
{
}

void ASManager::addOrRemoveDirtyGeometryGroupAccel( Acceleration* accel, bool add )
{
    if( m_dirtyGeometryGroupAccels.itemIsInList( accel ) == add )
        return;

    RT_ASSERT_MSG( !m_building, "dirty list modified while build in progress" );

    if( add )
        m_dirtyGeometryGroupAccels.addItem( accel );
    else
        m_dirtyGeometryGroupAccels.removeItem( accel );
}

void ASManager::addOrRemoveDirtyGroupAccel( Acceleration* accel, bool add )
{
    if( m_dirtyGroupAccels.itemIsInList( accel ) == add )
        return;

    RT_ASSERT_MSG( !m_building, "dirty list modified while build in progress" );

    if( add )
        m_dirtyGroupAccels.addItem( accel );
    else
        m_dirtyGroupAccels.removeItem( accel );
}

void ASManager::addOrRemoveDirtyGroup( AbstractGroup* group, bool add )
{
    if( m_dirtyGroups.itemIsInList( group ) == add )
        return;

    RT_ASSERT_MSG( !m_building, "dirty list modified while build in progress" );

    if( add )
        m_dirtyGroups.addItem( group );
    else
        m_dirtyGroups.removeItem( group );
}

void ASManager::addOrRemoveDirtyTopLevelTraversable( Acceleration* accel, bool add )
{
    if( add )
        m_dirtyTopLevelTraversables.addItem( accel );
    else
        m_dirtyTopLevelTraversables.removeItem( accel );
}

// Accelerations are sorted such that they can built in the order that they
// appear in the accels vector. Since the a node's bounding box sometimes comes
// from its accel, accels lower in the tree need to built before those higher in
// the tree. GeometryGroups never have accels below them so they are built first,
// followed by Groups in order of group height (number of groups below with root
// at top) in the tree. Since accels can be shared they need to be uniquified.
void ASManager::getSortedDirtyAccels( std::vector<Acceleration*>& sortedAccels, const std::vector<Acceleration*>& dirtyGroupAccels )
{
    TIMEVIZ_FUNC;

    sortedAccels = dirtyGroupAccels;

    auto compareAccels = []( const Acceleration* left, const Acceleration* right ) -> bool {
        return left->getMaxAccelerationHeight() < right->getMaxAccelerationHeight();
    };
    algorithm::sort( sortedAccels, compareAccels );
}

unsigned int ASManager::buildAccelerationStructures( const std::vector<Acceleration*>& accels, DeviceSet& buildDevices, CPUDevice* cpuDevice )
{
    // Allocate buffers and determine the amount of temporary space
    // needed for each build.
    bool anyNeedAabbs       = false;
    bool anyNeedMotionAabbs = false;
    for( Acceleration* accel : accels )
    {
        RT_ASSERT_MSG( accel->isAttached(), "Unattached accel should not be built" );
        accel->setupForBuild( buildDevices );
        anyNeedAabbs |= accel->getBuildSetupRequest().needAabbs;
        anyNeedMotionAabbs |= accel->hasMotionAabbs();
    }

    // Allocate space for temporaries.
    resizeTempBuffer( accels, buildDevices );

    // Sync memory and tables. Note: we could conserve transfers by
    // syncing only the build device. This would require some updates
    // to memory manager. However, it is likely to provide little gain
    // since the data will need to get transferred for the final
    // launch regardless.
    if( anyNeedAabbs )
    {
        llog( 20 ) << "Preparing for AABB launch at entry point: " << m_context->getAabbEntry() << '\n';
        setupPrograms( anyNeedMotionAabbs );
        m_context->launchPrepare( m_context->getAabbEntry(), 1, buildDevices, /*numLaunchDevices=*/1 );
    }
    else
    {
        m_context->lockMemoryManager();
    }

    // During the build it is illegal to mark any other accelerations
    // as dirty.
    m_building = true;

    // Build level 0
    unsigned int totalPrimitives = 0;
    buildAccels( accels, buildDevices, cpuDevice, totalPrimitives );

    // Release memory manager
    if( anyNeedAabbs )
    {
        llog( 20 ) << "ASManager: completing AABB launches\n";
        m_context->launchComplete();
    }
    else
    {
        m_context->unlockMemoryManager();
    }

    for( Acceleration* accel : accels )
        accel->finalizeAfterBuild( buildDevices );

    return totalPrimitives;
}

void ASManager::buildAccelerationStructures()
{
    if( m_dirtyGeometryGroupAccels.empty() && m_dirtyGroupAccels.empty() && m_dirtyGroups.empty() )
        return;
    if( !m_ringBuffer )
        setMinRingBufferSize();

    TIMEVIZ_FUNC;
    llog( 20 ) << "ASManager: build begin\n";
    timerTick t0 = getTimerTick();

    // Fill children arrays
    llog( 20 ) << "ASManager: filling " << m_dirtyGroups.size() << " child arrays\n";
    for( AbstractGroup* group : m_dirtyGroups )
        group->fillChildren();

    // Get build devices
    DeviceManager* dm        = m_context->getDeviceManager();
    CPUDevice*     cpuDevice = dm->cpuDevice();
    DeviceSet      buildDevices;
    if( m_context->useRtxDataModel() )
        buildDevices = DeviceSet( dm->activeDevices() );
    else
        buildDevices.insert( dm->primaryLWDADevice() );

    // Build GeometryGroups and Groups
    unsigned int totalPrimitives = 0;
    unsigned int granularity     = k_compactionGranularity.get();
    unsigned int idx             = 0;
    unsigned int wave            = 0;

    // Build in waves to reduce the high-watermark for memory
    // (compaction oclwrs by default at the end of each wave)
    while( idx < m_dirtyGeometryGroupAccels.size() )
    {
        std::vector<Acceleration*> toBuild;
        unsigned int               lwrrentPrimitives = 0;
        while( idx < m_dirtyGeometryGroupAccels.size() )
        {
            Acceleration* as = m_dirtyGeometryGroupAccels.getList()[idx++];
            toBuild.push_back( as );
            lwrrentPrimitives += as->getTotalPrimitiveCount();
            if( lwrrentPrimitives >= granularity )
                break;
        }

        llog( 20 ) << "ASManager: setting up " << toBuild.size() << " geometry accels for build\n";
        buildAccelerationStructures( toBuild, buildDevices, cpuDevice );
        wave++;
        totalPrimitives += lwrrentPrimitives;
    }
    std::vector<Acceleration*> sortedAccels;
    getSortedDirtyAccels( sortedAccels, m_dirtyGroupAccels.getList() );
    llog( 20 ) << "ASManager: setting up " << sortedAccels.size() << " group accels for build\n";
    totalPrimitives += buildAccelerationStructures( sortedAccels, buildDevices, cpuDevice );

    // Write the top-level traversables
    m_context->lockMemoryManager();
    for( Acceleration* as : m_dirtyTopLevelTraversables )
        as->writeTopLevelTraversable();
    m_context->unlockMemoryManager();

    // After builds are complete we now have the full time ranges for
    // motion groups.
    for( Acceleration* as : m_dirtyGroupAccels )
        as->updateTimeRangeForMotionGroup();

    // Stats
    float dt         = getDeltaMilliseconds( t0 );
    float mprims_sec = totalPrimitives * 0.001f / dt;
    Metrics::logInt( "build_accels_count", m_dirtyGroupAccels.size() + m_dirtyGeometryGroupAccels.size() );
    Metrics::logFloat( "build_accels_msec", dt );
    Metrics::logFloat( "build_Mprims_per_sec", mprims_sec );

    // Reset dirty
    m_dirtyGroupAccels.clear();
    m_dirtyGeometryGroupAccels.clear();
    m_dirtyGroups.clear();
    m_dirtyTopLevelTraversables.clear();

    m_building = false;
    llog( 20 ) << "ASManager: build done. " << totalPrimitives * 0.000001f << "M primitives in " << dt << " ms, "
               << mprims_sec << " M primitives/second\n";
}

static char* ringAllocate( size_t bytesRequested, char*& lwrPtr, size_t& bytesRemaining, char* ringBegin, size_t ringSize )
{
    if( bytesRequested > bytesRemaining )
    {
        RT_ASSERT( bytesRequested <= ringSize );
        lwrPtr         = ringBegin;
        bytesRemaining = ringSize;
    }
    char* ptr = lwrPtr;
    lwrPtr += bytesRequested;
    bytesRemaining -= bytesRequested;
    return ptr;
}


void ASManager::buildAccels( const std::vector<Acceleration*>& accels, DeviceSet buildDevices, Device* cpuDevice, unsigned int& totalPrimitives )
{

    std::vector<cort::Aabb> cpuAabbs;

    for( Acceleration* accel : accels )
    {
        RT_ASSERT_MSG( accel->isAttached(), "Unattached accel should not be built" );

        const BuildSetupRequest& request = accel->getBuildSetupRequest();
        totalPrimitives += request.totalPrims;

        // Allocate both AABBs and temp buffers in a single
        // allocation, otherwise it is not guaranteed to fit.
        size_t aabbSize = sizeof( cort::Aabb ) * (size_t)request.totalPrims * (size_t)request.motionSteps;
        aabbSize        = roundUp( aabbSize, size_t( AABB_ALIGNMENT ) );
        if( !request.needAabbs )
            aabbSize = 0;

        std::vector<BuildSetup::PerDevice> perDevice;
        for( int allDeviceIndex : buildDevices )
        {
            LWDADevice* buildDevice = deviceCast<LWDADevice>( m_context->getDeviceManager()->allDevices()[allDeviceIndex] );

            size_t tempSize  = roundUp( request.tempSize, size_t( AABB_ALIGNMENT ) );
            size_t totalSize = roundUp( tempSize + aabbSize, size_t( RTC_ACCEL_BUFFER_BYTE_ALIGNMENT ) );
            char*  ringBegin = m_ringBuffer->getAccess( buildDevice ).getLinearPtr();
            char*  ringPtr =
                reinterpret_cast<char*>( align( reinterpret_cast<size_t>( ringBegin ), RTC_ACCEL_BUFFER_BYTE_ALIGNMENT ) );
            size_t ringBufferSize = m_ringBuffer->getDimensions().getTotalSizeInBytes();
            // Include bytes that we didn't use for alignment shift in our remaining byte callwlation.
            size_t ringBytesRemaining = ringBufferSize - ( ringPtr - ringBegin );
            char*  tempPtr = ringAllocate( totalSize, ringPtr, ringBytesRemaining, ringBegin, ringBufferSize );
            char*  aabbPtr = tempPtr + tempSize;

            // Allocate space for CPU Aabbs if necessary. AS will copy them in place.
            if( request.needAabbsOnCpu )
                cpuAabbs.resize( request.totalPrims );

            BuildSetup::PerDevice pd( request.tempSize, buildDevice, tempPtr, (cort::Aabb*)aabbPtr, cpuDevice, cpuAabbs.data() );
            perDevice.push_back( pd );
        }
        // Issue the build
        BuildSetup setup( request.willRefit, request.totalPrims, perDevice );
        accel->build( setup );
    }
}

void ASManager::resizeTempBuffer( const std::vector<Acceleration*>& accels, const DeviceSet& buildDevices )
{
    size_t maxTemp   = 0;
    size_t totalTemp = 0;
    for( auto accel : accels )
    {
        const BuildSetupRequest& setup = accel->getBuildSetupRequest();
        size_t aabbSize = setup.needAabbs ? sizeof( cort::Aabb ) * setup.totalPrims * (size_t)setup.motionSteps : 0;
        aabbSize        = roundUp( aabbSize, size_t( AABB_ALIGNMENT ) );
        size_t tempSize = roundUp( setup.tempSize + aabbSize, ( size_t )( RTC_ACCEL_BUFFER_BYTE_ALIGNMENT ) );
        maxTemp         = std::max( tempSize, maxTemp );
        totalTemp += tempSize;
    }

    float  targetPasses = k_targetPasses.get();
    size_t bufferSize   = std::max( maxTemp, ( size_t )( totalTemp / targetPasses ) );
    bufferSize          = std::max( m_minRingBufferSize, bufferSize );
    // Round up to 64kb. Add buffer alignment so we have space to align pointers if necessary.
    bufferSize = roundUp( bufferSize + RTC_ACCEL_BUFFER_BYTE_ALIGNMENT, size_t( 64 * 1024 ) );

    if( !m_ringBuffer )
    {
        // Allocate buffer
        BufferDimensions dims( RT_FORMAT_BYTE, 1, 1, bufferSize, 1, 1 );
        MemoryManager*   mm = m_context->getMemoryManager();
        m_ringBuffer        = mm->allocateMBuffer( dims, MBufferPolicy::gpuLocal, buildDevices );
    }
    else if( bufferSize != m_ringBuffer->getDimensions().getTotalSizeInBytes() )
    {
        MemoryManager* mm = m_context->getMemoryManager();
        mm->changeSize( m_ringBuffer, BufferDimensions( RT_FORMAT_BYTE, 1, 1, bufferSize, 1, 1 ) );
    }
}

void ASManager::setupPrograms( bool supportMotion )
{
    TIMEVIZ_FUNC;

    // If there any group or geometry group requires motion, we use
    // the the more expensive entry point that is capable of gathering
    // motion aabbs. Otherwise, use the more streamlined version to
    // reduce compile times. Note that this could switch back and
    // forth between the two versions. Revisit if recompiles become
    // problem.
    const std::string progName = supportMotion ? "compute_motion_aabbs" : "compute_aabb";

    Program* aabbCompute = m_context->getSharedProgramManager()->getProgram( data::getComputeAabbSources(), progName );
    m_context->getGlobalScope()->setAabbComputeProgram( aabbCompute );

    Program* exception =
        m_context->getSharedProgramManager()->getProgram( data::getComputeAabbSources(), "compute_aabb_exception" );
    m_context->getGlobalScope()->setAabbExceptionProgram( exception );

    m_needsInitialSetup = false;
}

void ASManager::setupInitialPrograms()
{
    // We defer creating the AABB programs until they are actually needed.
    // If we reach a point where they're needed but haven't been incidentally
    // created (e.g., as a result of a call to buildAccelerationStructures),
    // then we create them here with motion disabled.
    if( m_needsInitialSetup )
    {
        setupPrograms( false );
        m_needsInitialSetup = false;
    }
}

void ASManager::setMinRingBufferSize()
{
    Device* buildDevice = m_context->getDeviceManager()->primaryDevice();

    // Initial temporary buffer is a fraction of device memory
    size_t memorySize   = buildDevice->getTotalMemorySize();
    float  fraction     = k_initialTempBufferFraction.get();
    m_minRingBufferSize = ( size_t )( memorySize * fraction );

    // Round up to 64kb
    m_minRingBufferSize = ( m_minRingBufferSize + 0xffffLL ) & ~0xffffLL;
}

void ASManager::preSetActiveDevices( const DeviceArray& removedDevices )
{
    m_ringBuffer.reset();
}
