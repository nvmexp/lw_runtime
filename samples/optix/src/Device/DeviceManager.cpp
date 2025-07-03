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

#include <Device/DeviceManager.h>

#include <AS/ASManager.h>
#include <LWCA/ComputeCapability.h>
#include <LWCA/Memory.h>
#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Context/PlanManager.h>
#include <Context/ProfileManager.h>
#include <Context/SBTManager.h>
#include <Context/TableManager.h>
#include <Context/UpdateManager.h>
#include <Device/APIDeviceAttributes.h>
#include <Device/CPUDevice.h>
#include <Device/LWDADevice.h>
#include <Device/Device.h>
#include <Device/DriverVersion.h>
#include <Exceptions/NoDevice.h>
#include <ExelwtionStrategy/RTX/RTXES.h>
#include <Memory/DemandLoad/PagingService.h>
#include <Memory/MemoryManager.h>
#include <Util/ContainerAlgorithm.h>

#include <corelib/misc/String.h>
#include <corelib/system/LwdaDriver.h>
#include <corelib/system/ExelwtableModule.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/LwdaError.h>
#include <prodlib/exceptions/IlwalidDevice.h>
#include <prodlib/exceptions/IlwalidOperation.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/misc/TimeViz.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

#include <llvm/ADT/SetVector.h>

#include <functional>
#include <iterator>
#include <limits>
#include <mutex>

#ifdef _WIN32
#include <prodlib/misc/GLFunctions.h>
#include <prodlib/misc/GLInternalFormats.h>
#endif  // _WIN32

using namespace optix;
using namespace prodlib;
using namespace corelib;
using namespace optix::lwca;


namespace {
// clang-format off
  Knob<bool>        k_forceCPUFallback(        RT_DSTRING("deviceManager.forceCPUFallback"),        false, RT_DSTRING( "Enable CPU fallback always" ) );
  Knob<bool>        k_allowCPUFallback(        RT_DSTRING("deviceManager.allowCPUFallback"),        false, RT_DSTRING( "Allow CPU fallback" ) );
  Knob<bool>        k_forceForeignInterop(     RT_DSTRING("deviceManager.forceForeignInterop"),     false, RT_DSTRING( "Force interop to use the foreign device (non-lwca) path" ) );
  Knob<int>         k_minSupportedDevice(      RT_DSTRING("deviceManager.minSupportedDevice"),      0,     RT_DSTRING( "Set minimum allowed device SM" ) );
  Knob<int>         k_maxSupportedDevice(      RT_DSTRING("deviceManager.maxSupportedDevice"),      0,     RT_DSTRING( "Set maximum allowed device SM" ) );
  Knob<int>         k_forceSmVersion(          RT_DSTRING("deviceManager.forceSmVersion"),          0,     RT_DSTRING( "Force a specific SM target (20,30,35,etc.)" ) );
  Knob<std::string> k_visibleLwdaDevices(      RT_DSTRING("deviceManager.visibleLwdaDevices"),      "",    RT_DSTRING( "Filter the LWCA devices visible to OptiX based on a comma-separated list of LWCA ordinals" ) );
  Knob<bool>        k_printDevices(            RT_DSTRING("deviceManager.printDevices"),            false, RT_DSTRING( "Dump list of supported devices" ) );
  Knob<bool>        k_compileOnly(             RT_DSTRING("compile.compileOnly"),                   false, RT_DSTRING( "Just compile the kernel with the forced sm version and quit" ) );
  Knob<bool>        k_p2pAllowed(              RT_DSTRING("mem.p2pAllowed"),                        true,  RT_DSTRING( "Allow use of LWCA peer to peer memory access" ) );
// clang-format on

// Returns true if the value is found in the container.
// Use this when you don't care about the position of the value
// within the container.
//
template <typename Container, typename Value>
bool contains( const Container& c, Value val )
{
    return algorithm::find( c, val ) != c.end();
}

struct DeviceAttributesContainer
{

    const APIDeviceAttributes& getAPIDeviceAttributes( int optixOrdinal );
    unsigned int getDeviceCount();

  private:
    void initialize();

    //
    std::mutex                       m_initializationLock;
    int                              m_initialized = false;
    std::vector<APIDeviceAttributes> m_attributes;
};

static DeviceAttributesContainer g_deviceAttributes;

}  // namespace

DeviceManager::DeviceManager( optix::Context* context, bool useRtxDataModel )
    : m_context( context )
{
    setupDevices( useRtxDataModel );
}

DeviceManager::~DeviceManager()
{
    for( Device* device : m_allDevices )
        delete device;
}

unsigned int DeviceManager::getDeviceCount()
{
    return g_deviceAttributes.getDeviceCount();
}

const DeviceArray& DeviceManager::allDevices() const
{
    return m_allDevices;
}

const DeviceArray& DeviceManager::visibleDevices() const
{
    return m_visibleDevices;
}

const DeviceArray& DeviceManager::activeDevices() const
{
    return m_activeDevices;
}

const DeviceArray& DeviceManager::uniqueActiveDevices() const
{
    return m_uniqueActiveDevices;
}

void DeviceManager::setActiveDevices( const DeviceArray& newDevices )
{
    if( m_context->useRtxDataModel() )
    {
        const bool anyRTCoreCapableDevices = algorithm::any_of( newDevices, std::mem_fn( &Device::supportsTTU ) );
        const bool allRTCoreCapableDevices = algorithm::all_of( newDevices, std::mem_fn( &Device::supportsTTU ) );
        if( anyRTCoreCapableDevices && !allRTCoreCapableDevices )
        {
            throw IlwalidDevice( RT_EXCEPTION_INFO, "All devices must be RTCore capable." );
        }
    }


    // Select a new GL interop device the next time someone asks for one.
    m_selectNewGlInteropDevice = true;

    // Find out which of these devices were active already.
    // Use a SetVector to have a deterministic order of the devices in the set.
    // This is then used below to update the active set with the new devices
    // without adding potential duplicates.
    llvm::SetVector<Device*> newDeviceSet( newDevices.begin(), newDevices.end() );

    llvm::SetVector<Device*> alreadyActive;
    for( DeviceArray::const_iterator AD = m_activeDevices.begin(), ADE = m_activeDevices.end(); AD != ADE; ++AD )
    {
        if( newDeviceSet.count( *AD ) )
            alreadyActive.insert( *AD );
    }

    // Create lists with the removed and newly added devices to pass to listeners.
    DeviceArray removedDevices;
    DeviceArray addedDevices;

    for( Device* device : m_activeDevices )
    {
        if( !alreadyActive.count( device ) )
            removedDevices.push_back( device );
    }

    // Iterate over a set to make sure we don't add duplicates.
    for( Device* D : newDeviceSet )
    {
        if( !alreadyActive.count( D ) )
            addedDevices.push_back( D );
    }

    // Bail out if nothing has changed.
    if( removedDevices.empty() && addedDevices.empty() )
    {
        return;
    }

    const DeviceArray oldDevices = m_activeDevices;
    m_context->getUpdateManager()->eventActiveDevicesWillChange( oldDevices, newDevices );

    // Delete all the plans that refer to the devices being removed. This is necessary
    // because the plans keep references to LWCA modules that are part of the devices.
    m_context->getPlanManager()->removePlansForDevices( removedDevices );

    // Make sure the SBTManager and RTXES don't have references to modules that should
    // be removed.
    m_context->getRTXExelwtionStrategy()->removeProgramsForDevices( removedDevices );
    m_context->getSBTManager()->removeProgramsForDevices( removedDevices );

    // Disable the lwrrently active plan. This is because we may have added devices to the list,
    // which should ilwalidate the current plan anyway.
    m_context->getPlanManager()->activate( nullptr );

    m_context->getSBTManager()->preSetActiveDevices( removedDevices );

    // ObjectManager::preSetActiveDevices must be called before TableManager::preSetActiveDevices,
    // since the TableManager resets the deviceDependentTable. Otherwise this causes issues when the
    // ObjectManager releases the traversable buffer, and the Group then attempts to update that
    // traversable header in the table that had just been reset.
    m_context->getObjectManager()->preSetActiveDevices( removedDevices );
    m_context->getTableManager()->preSetActiveDevices( removedDevices );
    m_context->getProfileManager()->preSetActiveDevices( removedDevices );
    m_context->getASManager()->preSetActiveDevices( removedDevices );
    m_context->getPagingManager()->preSetActiveDevices( removedDevices );

    // Inform the RTX exelwtion strategy about the removed devices so that it can clean up resources.
    m_context->getRTXExelwtionStrategy()->preSetActiveDevices( removedDevices );

    // Update the active devices array without adding duplicates.
    // TODO: We should throw an exception on duplicates, not just eat them
    m_activeDevices.clear();
    m_activeDevices.insert( m_activeDevices.end(), newDeviceSet.begin(), newDeviceSet.end() );

    // Find the unique devices
    setUniqueActiveDevices();

    // Re-number the devices so the active ones have a proper "active index".
    numberDevices();

    // Update the support for TTU
    setActiveDevicesSupportTTU();

    // Update support for LWCA sparse textures. Dispatch an event to notify
    // listeners if support changed.
    bool activeDevicesSupportLwdaSparseTexturesOld = activeDevicesSupportLwdaSparseTextures();
    setActiveDevicesSupportLwdaSparseTextures();
    if( activeDevicesSupportLwdaSparseTextures() != activeDevicesSupportLwdaSparseTexturesOld )
        m_context->getUpdateManager()->eventActiveDevicesSupportLwdaSparseTexturesDidChange( activeDevicesSupportLwdaSparseTextures() );

    // Enable added devices before triggering new allocations in the
    // memory manager.
    for( Device* device : addedDevices )
        device->enable();

    // Update peer access for new devices
    enablePeerAccess( removedDevices );

    // Memory manager needs to allocate memory for buffers etc. on each device.
    DeviceSet activeDeviceSet( m_activeDevices );
    m_context->getMemoryManager()->setActiveDevices( activeDeviceSet );

    // Disable removed devices.
    for( Device* device : removedDevices )
        device->disable();

    m_context->getSBTManager()->postSetActiveDevices();

    m_context->getTableManager()->postSetActiveDevices();
    m_context->getProfileManager()->postSetActiveDevices();
    m_context->getObjectManager()->postSetActiveDevices( removedDevices );
}

void DeviceManager::enableActiveDevices()
{
    TIMEVIZ_FUNC;
    // Some devices may have been disabled.  Loop through all devices
    // and either enable or disable as appropriate.
    bool enabledDevicesChanged = false;
    for( auto device : m_allDevices )
    {
        if( device->isActive() && !device->isEnabled() )
        {
            device->enable();
            enabledDevicesChanged = true;
        }
        else if( !device->isActive() && device->isEnabled() )
        {
            disableDeviceAndUpdatePeerAccess( device );
            enabledDevicesChanged = true;
        }
    }

    // Peer access might have changed, update it.
    if( enabledDevicesChanged )
        enablePeerAccess();
}

void DeviceManager::enablePeerAccess()
{
    enablePeerAccess( DeviceSet() );
}

void DeviceManager::enablePeerAccess( DeviceSet devicesToIgnore )
{
    TIMEVIZ_FUNC;
    if( !k_p2pAllowed.get() )
    {
        // Just put each device in its own island.
        m_activeDeviceLwlinkIslands.clear();
        for( const auto& dev : m_activeDevices )
            m_activeDeviceLwlinkIslands.push_back( DeviceSet( dev ) );
        return;
    }

    bool      peerAccessChanged = false;
    DeviceSet lwdaActiveSet;
    for( LWDADevice* lwdaDeviceX : LWDADeviceArrayView( m_activeDevices ) )
    {
        lwdaActiveSet.insert( lwdaDeviceX );

        for( LWDADevice* lwdaDeviceY : LWDADeviceArrayView( m_activeDevices ) )
        {
            if( lwdaDeviceX == lwdaDeviceY )
                continue;

            if( lwdaDeviceX->canAccessPeer( *lwdaDeviceY ) && !lwdaDeviceX->isPeerAccessEnabled( *lwdaDeviceY ) )
            {
                // Even though this device can access its peer, we still may
                // not be able to enable P2P between them due to limited
                // hardware resources (e.g. LWDA_ERROR_TOO_MANY_PEERS)
                if( lwdaDeviceX->enablePeerAccess( *lwdaDeviceY ) )
                {
                    lwdaDeviceX->dumpShort( llog_stream( 10 ) );
                    peerAccessChanged = true;
                    llog( 10 ) << "           CAN access peer device=" << lwdaDeviceY->lwdaDevice().get() << std::endl;
                }
                else
                    llog( 10 ) << "           CAN access, but unable to enable for peer device="
                               << lwdaDeviceY->lwdaDevice().get() << std::endl;
            }
        }
    }


    // clang-format off
  /* Example.
  If devices access peers like:          then p2p access mask is:       and islands are [0, 1, 2, 3] and [4, 5, 6, 7],
    ---------                              0 1 2 3 4 5 6 7              where p2p access is posible between any devices
   /         \                           0 x x x x x
  5 - 4 - 0 - 1                          1 x x x x   x
  | x |   | x |                          2 x x x x     x
  6 - 7 - 3 - 2                          3 x x x x       x
  \           /                          4 x       x x x x
   -----------                           5   x     x x x x
                                         6     x   x x x x
                                         7       x x x x x
  */
    // clang-format on

    // Compute LWLink islands.
    if( peerAccessChanged || m_activeDeviceLwlinkIslands.empty() )
    {
        m_activeDeviceLwlinkIslands.clear();
        DeviceSet allKnownIslands;
        // linearly search for a device which is not in any island
        for( LWDADevice* devX : LWDADeviceArrayView( m_activeDevices ) )
        {
            if( allKnownIslands.isSet( devX ) || devicesToIgnore.isSet( devX ) )
                continue;

            DeviceSet    devSetFromX = devX->lwlinkAccessible() & ~allKnownIslands & lwdaActiveSet;
            DeviceSet    devSetX     = devSetFromX | DeviceSet( devX );
            unsigned int cardinal    = devSetX.count();  // maximum island size

            DeviceSet island;
            // we don't search all combinations of devices, we greedy search device island with maximum size
            while( cardinal > 0 )
            {
                island = DeviceSet( devX );
                for( int y : devSetFromX )
                {
                    LWDADevice* devY = deviceCast<LWDADevice>( m_allDevices[y] );
                    if( !devY )
                        continue;
                    DeviceSet devSetY = ( devY->lwlinkAccessible() | DeviceSet( devY ) ) & ~allKnownIslands & lwdaActiveSet;
                    if( ( devSetX & devSetY ).count() >= cardinal )
                        island.insert( devY );
                }
                if( island.count() == cardinal )
                {
                    RT_ASSERT( ( allKnownIslands & island ).empty() );
                    allKnownIslands |= island;
                    m_activeDeviceLwlinkIslands.push_back( island );
                    break;
                }
                cardinal--;
            }
        }
        RT_ASSERT( allKnownIslands == lwdaActiveSet );

        // Sanity check.
        for( size_t x = 0; x < m_activeDeviceLwlinkIslands.size(); x++ )
        {
            for( size_t y = x + 1; y < m_activeDeviceLwlinkIslands.size(); y++ )
            {
                RT_ASSERT( ( m_activeDeviceLwlinkIslands[x] & m_activeDeviceLwlinkIslands[y] ).empty() );
            }
        }

        // By default, print the island config only if it's "interesting".
        {
            const int   lev = anyActiveDevicesAreLwlinkEnabled() ? log::LEV_PRINT : 10;
            std::string msg = "Peer-to-peer islands: ";
            for( const auto& island : m_activeDeviceLwlinkIslands )
                msg += island.toString() + " ";
            llog( lev ) << msg << "\n";
            m_context->getUsageReport().getPreambleStream() << msg << std::endl;
        }

        // Check if the devices are fully connected via P2P.
        const size_t numPeers = lwdaActiveSet.count() - 1;
        // If the device has no peers, it can't be connected to anything.
        if( numPeers == 0 )
        {
            m_fullyConnected = false;
            return;
        }
        else
            m_fullyConnected = true;

        for( const LWDADevice* lwdaDev : LWDADeviceArrayView( m_activeDevices ) )
        {
            if( lwdaDev->peerEnabled().count() != numPeers )
            {
                m_fullyConnected = false;
                break;
            }
        }
        llog( 10 ) << "Fully connected: " << m_fullyConnected << "\n";
    }
}

void DeviceManager::setupDevices( bool useRtxDataModel )
{
    // Check display driver version
    DriverVersion lwrrentDriverVersion;
    DriverVersion requiredDriverVersion( getMinimumRequiredDriverVersion() );
    if( !lwrrentDriverVersion.isValid() )
    {
        lwarn << "Failed to retrieve the LWPU driver version\n";
    }
    else if( lwrrentDriverVersion < requiredDriverVersion )
    {
        lerr << "LWPU driver version " << lwrrentDriverVersion.toString() << " found. "
             << "Minimum required LWPU driver version is " << requiredDriverVersion.toString() << ".\n";
    }

    // Set up all, visible and active lists
    createAllDevices();

    filterVisibleDevices();

    setDefaultActiveDevices( useRtxDataModel );

    numberDevices();

    setActiveDevicesSupportTTU();
    setActiveDevicesSupportLwdaSparseTextures();

    // Print device info
    if( k_printDevices.get() )
    {
        if( lwdaDriver().lwdaAvailable() )
        {
            int version;
            CALL_LWDA_DRIVER_THROW( lwdaDriver().LwDriverGetVersion( &version ) );
#ifdef __APPLE__
            lprint << "Using LWCA driver: " << version / 1000 << "." << version % 100 << '\n';
#else
            lprint << "Using LWCA driver: " << version / 100 << "." << version % 100 << '\n';
#endif
        }
        else
        {
            lprint << "LWCA unavailable\n";
        }
        lprint << "Dumping " << m_visibleDevices.size() << " supported devices (of " << m_allDevices.size() << ")...\n";
        for( Device* device : m_visibleDevices )
            device->dump( lprint_stream );
    }

    // Make sure we have at least one good device
    if( m_activeDevices.empty() && !k_forceCPUFallback.get() )
        throw NoDevice( RT_EXCEPTION_INFO, "No LWPU OptiX capable device could be found." );
}

void DeviceManager::createAllDevices()
{
    RT_ASSERT_MSG( !k_forceCPUFallback.get() || k_allowCPUFallback.get(),
                   "If the knob deviceManager.forceCPUFallback is set to true, then deviceManager.allowCPUFallback "
                   "must be set to true also" );

    ComputeCapability forceSmVersion( k_forceSmVersion.get() );

    // Create a device for each LWCA device whether it is viable or not
    if( k_compileOnly.get() )
    {
        RT_ASSERT_MSG( forceSmVersion != SM_NONE(), "SM version must be specified with compile.compileOnly" );
        // make a fake LWCA device
        m_allDevices.push_back( new LWDADevice( m_context, -1, forceSmVersion ) );
    }
    else if( !k_forceCPUFallback.get() )
    {
        // Attempt to initialize LWCA if we have a LWCA driver
        if( lwdaDriver().lwdaAvailable() )
        {
            LWresult result = lwdaDriver().LwInit( 0 );
            if( result != LWDA_ERROR_NO_DEVICE && result != LWDA_SUCCESS )
                CALL_LWDA_DRIVER_THROW( result );

            if( result == LWDA_SUCCESS )
            {
                // Query available LWCA devices
                const int count = lwca::Device::getCount();
                for( int ordinal = 0; ordinal < count; ++ordinal )
                    m_allDevices.push_back( new LWDADevice( m_context, ordinal, forceSmVersion ) );
            }
        }
    }

    // Create a device for CPU
    m_allDevices.push_back( new CPUDevice( m_context ) );
}

void DeviceManager::filterVisibleDevices()
{
    // Get the min/max supported configurations
    ComputeCapability minSupportedDevice( k_minSupportedDevice.get() );
    ComputeCapability maxSupportedDevice( k_maxSupportedDevice.get() );
    if( minSupportedDevice > SM_NONE() || maxSupportedDevice != SM_NONE() )
    {
        lprint << "restricting supported devices from " << minSupportedDevice.toString() << " to "
               << ( maxSupportedDevice == SM_NONE() ? "max" : maxSupportedDevice.toString() ) << "\n";
    }
    if( minSupportedDevice == SM_NONE() )
    {
        const int lwvmMilwersion = 50;  // SM 5.0 (Maxwell) is required for optix. SM 3.0 is required for lwvm.
        minSupportedDevice       = ComputeCapability( lwvmMilwersion );
    }
    if( maxSupportedDevice == SM_NONE() )
    {
        const int maxVersion = 999;  // SM version 99.9 - should last a while
        maxSupportedDevice   = ComputeCapability( maxVersion );
    }

    // If specified, read the list of allowed LWCA ordinals.
    std::vector<int> allowedLwdaOrdinals;
    if( !k_visibleLwdaDevices.get().empty() )
    {
        const std::vector<std::string> ordinalsStr = tokenize( k_visibleLwdaDevices.get(), "," );
        for( const auto& s : ordinalsStr )
            allowedLwdaOrdinals.push_back( from_string<int>( s ) );
    }

    for( Device* device : m_allDevices )
    {
        // Remove CPU device if fallback is not enabled
        if( deviceCast<CPUDevice>( device ) && !k_allowCPUFallback.get() )
            continue;

        if( LWDADevice* lwdaDevice = deviceCast<LWDADevice>( device ) )
        {
            // Remove LWCA devices that do not meet the minimum/maximum specs
            ComputeCapability smversion = lwdaDevice->computeCapability();
            if( smversion > maxSupportedDevice )
                continue;
            if( smversion < minSupportedDevice )
                continue;

            // Remove LWCA devices that are filtered out via knob
            if( !allowedLwdaOrdinals.empty() )
            {
                if( contains( allowedLwdaOrdinals, lwdaDevice->lwdaOrdinal() ) )
                    continue;
            }
        }

        // Add the device
        m_visibleDevices.push_back( device );
    }
}

void DeviceManager::setDefaultActiveDevices( bool useRtxDataModel )
{
    m_activeDevices.clear();
    if( m_visibleDevices.empty() )
        return;  // No devices - flag an error elsewhere

    if( !useRtxDataModel )
    {
        // make all visible devices active by default.
        for( Device* const vis : m_visibleDevices )
        {
            // Make the CPU device active by default only if it is forced.
            if( deviceCast<CPUDevice>( vis ) && !k_forceCPUFallback.get() )
                continue;
            m_activeDevices.push_back( vis );
        }
    }
    else
    {
        setRtxDefaultActiveDevices();
    }

    setUniqueActiveDevices();
}

// By default select all devices with the highest SM version that are RTCore capable.
// If no RTCore capable devices are present, then select all the devices with the
// highest SM version.
//
// Use stable_partition and stable_sort to ensure that relative ordering of equivalent
// devices is preserved from the initial set of visible devices.
//
void DeviceManager::setRtxDefaultActiveDevices()
{
    // Start with all visible devices.
    m_activeDevices = m_visibleDevices;

    // First partition into LWCA capable and non-capable devices.
    const auto beginActiveDevices = m_activeDevices.begin();
    const auto endActiveDevices   = m_activeDevices.end();
    const auto beginNonLWDADevices = std::stable_partition( beginActiveDevices, endActiveDevices, []( const Device* device ) {
        return deviceCast<const LWDADevice>( device ) != nullptr;
    } );

    if( beginNonLWDADevices == beginActiveDevices )
    {
        // We have no LWCA devices, so clear everything.
        m_activeDevices.clear();
        return;
    }

    // Further partition into RTCore capable devices within the LWCA capable devices.
    auto endCompatibleDevices =
        std::stable_partition( beginActiveDevices, beginNonLWDADevices, std::mem_fn( &Device::supportsTTU ) );

    // Use all the LWCA devices if there are no RTCore capable devices.
    if( endCompatibleDevices == beginActiveDevices )
    {
        endCompatibleDevices = beginNonLWDADevices;
    }

    // Sort by compute capability, with highest capability first.
    std::stable_sort( beginActiveDevices, endCompatibleDevices, []( const Device* lhs, const Device* rhs ) {
        const LWDADevice* lhsLwda = deviceCast<const LWDADevice>( lhs );
        const LWDADevice* rhsLwda = deviceCast<const LWDADevice>( rhs );
        return lhsLwda->computeCapability() > rhsLwda->computeCapability();
    } );

    // Find the last device compatible with the device with highest capability.
    endCompatibleDevices = std::find_if( beginActiveDevices, endCompatibleDevices, [this]( Device* device ) {
        return !device->isCompatibleWith( m_activeDevices[0] );
    } );

    // Remove devices that aren't compatible
    m_activeDevices.erase( endCompatibleDevices, endActiveDevices );
}

// Filter the active devices down to a set of unique devices
void DeviceManager::setUniqueActiveDevices()
{
    m_uniqueActiveDevices.clear();
    for( Device* device : m_activeDevices )
    {
        Device* compatibleDevice = getFirstCompatibleDevice( device );
        // Add if not already in the unique devices array
        if( !contains( m_uniqueActiveDevices, compatibleDevice ) )
            m_uniqueActiveDevices.push_back( compatibleDevice );
    }
}

optix::Device* DeviceManager::getFirstCompatibleDevice( optix::Device* other ) const
{
    for( Device* device : m_activeDevices )
        if( device->isCompatibleWith( other ) )
            return device;
    return nullptr;
}

namespace {

// Starting with zero, set an increasing device list index on each device in a DeviceArray
void enumerateDevices( const DeviceArray& devices, const std::function<void( optix::Device*, unsigned int )>& setDeviceListIndex )
{
    unsigned int index = 0U;
    for( optix::Device* const device : devices )
    {
        setDeviceListIndex( device, index );
        ++index;
    }
}

}  // namespace

void DeviceManager::numberDevices()
{
    RT_ASSERT_MSG( m_visibleDevices.size() <= m_allDevices.size(),
                   "Visible devices should be a subset of all devices" );
    RT_ASSERT_MSG( m_activeDevices.size() <= m_visibleDevices.size(),
                   "Active devices should be a subset of visible devices" );

    for( Device* device : m_allDevices )
        device->resetNumbering();

    enumerateDevices( m_allDevices, std::mem_fn( &Device::setAllDeviceListIndex ) );
    enumerateDevices( m_visibleDevices, std::mem_fn( &Device::setVisibleDeviceListIndex ) );
    enumerateDevices( m_activeDevices, std::mem_fn( &Device::setActiveDeviceListIndex ) );
    enumerateDevices( m_uniqueActiveDevices, std::mem_fn( &Device::setUniqueDeviceListIndex ) );

    // Also number the other active devices with their unique index
    for( Device* device : m_activeDevices )
    {
        Device* compatibleDevice = getFirstCompatibleDevice( device );
        device->setUniqueDeviceListIndex( compatibleDevice->uniqueDeviceListIndex() );
    }
}

void DeviceManager::shutdownDevices()
{
    // Disable all devices
    DeviceArray devices;
    setActiveDevices( devices );
}

void DeviceManager::printAvailableDevices() const
{
    std::ostream& out = lprint_stream;  // accessing the stream once produces better output when the log timing is on
    out << "OptiX has " << m_visibleDevices.size() << " devices available for use: ";
    for( size_t i = 0; i < m_visibleDevices.size(); ++i )
        out << ( i == 0 ? "" : ", " ) << m_visibleDevices[i]->deviceName();
    out << std::endl;
}

const std::vector<DeviceSet>& DeviceManager::getLwlinkIslands() const
{
    return m_activeDeviceLwlinkIslands;
}

DeviceSet DeviceManager::getLwlinkIsland( optix::Device* device ) const
{
    for( auto island : m_activeDeviceLwlinkIslands )
    {
        if( island.isSet( device ) )
        {
            return island;
        }
    }
    RT_ASSERT_FAIL_MSG( "No p2p island for the device found" );
}

bool DeviceManager::anyActiveDevicesAreLwlinkEnabled() const
{
    if( !k_p2pAllowed.get() )
        return false;

    for( LWDADevice* lwdaDev : LWDADeviceArrayView( m_activeDevices ) )
    {
        if( !lwdaDev->lwlinkAccessible().empty() )
            return true;
    }
    return false;
}

bool DeviceManager::isPeerToPeerFullyConnected() const
{
    if( !k_p2pAllowed.get() )
        return false;
    return m_fullyConnected;
}

bool DeviceManager::activeDevicesSupportTTU() const
{
    return m_activeDevicesSupportTTU;
}

bool DeviceManager::activeDevicesSupportMotionTTU() const
{
    return m_activeDevicesSupportMotionTTU;
}

void DeviceManager::setActiveDevicesSupportTTU()
{
    m_activeDevicesSupportTTU = !m_activeDevices.empty();
    for( const Device* device : m_activeDevices )
        m_activeDevicesSupportTTU   = m_activeDevicesSupportTTU && device->supportsTTU();
    m_activeDevicesSupportMotionTTU = !m_activeDevices.empty();
    for( const Device* device : m_activeDevices )
        m_activeDevicesSupportMotionTTU = m_activeDevicesSupportMotionTTU && device->supportsMotionTTU();
}

bool DeviceManager::activeDevicesSupportLwdaSparseTextures() const
{
    return m_activeDevicesSupportLwdaSparseTextures;
}

bool DeviceManager::activeDevicesSupportTextureFootprint() const
{
    return m_activeDevicesSupportTextureFootprint;
}

void DeviceManager::setActiveDevicesSupportLwdaSparseTextures()
{
    m_activeDevicesSupportTextureFootprint   = !m_activeDevices.empty();
    m_activeDevicesSupportLwdaSparseTextures = !m_activeDevices.empty();
    for( const LWDADevice* lwdaDevice : LWDADeviceArrayView( m_activeDevices ) )
    {
        m_activeDevicesSupportLwdaSparseTextures =
            m_activeDevicesSupportLwdaSparseTextures && lwdaDevice->supportsLwdaSparseTextures();
        m_activeDevicesSupportTextureFootprint = m_activeDevicesSupportTextureFootprint && lwdaDevice->supportsTextureFootprint();
    }
}

LWDADevice* DeviceManager::leastFreeMemLWDADevice( size_t* bytesFree, size_t* bytesTotal ) const
{
    size_t      minFree   = std::numeric_limits<size_t>::max();
    LWDADevice* resultDev = nullptr;

    for( LWDADevice* lwdaDev : LWDADeviceArrayView( m_activeDevices ) )
    {
        size_t totalMem = 0, freeMem = 0;
        lwdaDev->makeLwrrent();
        memGetInfo( &freeMem, &totalMem );

        if( freeMem < minFree )
        {
            minFree   = freeMem;
            resultDev = lwdaDev;

            if( bytesFree )
                *bytesFree = freeMem;
            if( bytesTotal )
                *bytesTotal = totalMem;
        }
    }

    RT_ASSERT_MSG( resultDev, "No active LWCA device found" );
    return resultDev;
}

optix::Device* DeviceManager::primaryDevice()
{
    RT_ASSERT_MSG( !m_activeDevices.empty(), "No active devices found" );
    return m_activeDevices[0];
}

LWDADevice* DeviceManager::primaryLWDADevice()
{
    // TODO: primary LWCA devices can change during the lifetime of the application.
    // For zero copy, it is important that the primary devices are active and the same
    // during registration/deregistration, as it is done per LWCA context.
    //
    // We either make the primaryLWDAdevice ilwariant (i.e. the first visible lwca device)
    // and enable/disable it on demand when needed, or we need another mechanism to track
    // the context used for registration when deregistering a zero copy buffer.
    for( LWDADevice* lwdaDevice : LWDADeviceArrayView( m_activeDevices ) )
    {
        return lwdaDevice;
    }
    return nullptr;
}

CPUDevice* DeviceManager::cpuDevice()
{
    for( Device* device : m_allDevices )
    {
        CPUDevice* cpudevice = deviceCast<CPUDevice>( device );
        if( cpudevice )
            return cpudevice;
    }
    RT_ASSERT_FAIL_MSG( "CPU device unexepctedly not found in allDevices" );
}

static optix::Device* findLwdaDevice( const optix::lwca::Device& device, const DeviceArray& deviceArray )
{
    for( LWDADevice* d : LWDADeviceArrayView( deviceArray ) )
        if( d->lwdaDevice() == device )
            return d;
    return nullptr;
}

optix::Device* DeviceManager::glInteropDevice()
{
#ifdef __APPLE__
    // There is no support for lwGLGetDevices on Mac OS X, so we will just hope and pray
    // this is the right device to return.  Note that this is the same thing that old rtmain
    // returned.
    if( m_visibleDevices.empty() )
        return 0;
    return m_visibleDevices[0];
#else

    if( !m_selectNewGlInteropDevice )
        return m_glInteropDevice;

    m_selectNewGlInteropDevice = false;
    m_glInteropDevice          = cpuDevice();

    if( k_forceForeignInterop.get() )
        return m_glInteropDevice;

    std::vector<lwca::Device> glDevices( lwca::Device::getCount() );
    unsigned                  glDeviceCount = 0;
    LWresult                  returnResult  = LWDA_SUCCESS;
    lwca::Device::GLGetDevices( &glDeviceCount, &glDevices[0], (unsigned)glDevices.size(), LW_GL_DEVICE_LIST_ALL, &returnResult );
    if( returnResult == LWDA_ERROR_NO_DEVICE )
        return m_glInteropDevice;
    else if( returnResult != LWDA_SUCCESS )
        throw LwdaError( RT_EXCEPTION_INFO, "lwGLGetDevices", returnResult );

    if( glDevices.empty() )
        return m_glInteropDevice;

    DeviceSet glDeviceSet = DeviceSet();
    for( lwca::Device& lwrrLwdaDev : glDevices )
    {
        const Device* lwrrDev = findLwdaDevice( lwrrLwdaDev, m_visibleDevices );
        if( lwrrDev )
            glDeviceSet.insert( lwrrDev );
    }

    if( glDeviceSet.empty() )
        throw IlwalidOperation( RT_EXCEPTION_INFO,
                                "LWCA devices corresponding to GL context not found in OptiX device list" );

    DeviceSet glActiveDevices = glDeviceSet & DeviceSet( m_activeDevices );
    if( !glActiveDevices.empty() )
    {
        m_glInteropDevice = m_allDevices[*glActiveDevices.begin()];

#ifdef _WIN32
        // Prefer the GPU associated with the GL context for interop.

        if( GL::ExtensionExtExternalObjectsAvailable() && GL::ExtensionExtExternalObjectsWin32Available() )
        {
            // Get the Windows display adapter ID associated with the context.
            char contextLuidBuf[GL_LUID_SIZE_EXT] = {};
            GL::GetUnsignedBytevEXT( GL_DEVICE_LUID_EXT, contextLuidBuf );
            LUID* glContextLuid = reinterpret_cast<LUID*>( contextLuidBuf );

            // Get the mask showing which adapter device is being used.
            GLint nodeMask = 0;
            GL::GetIntegerv( GL_DEVICE_NODE_MASK_EXT, &nodeMask );
            unsigned int glContextNodeMask = static_cast<unsigned int>( nodeMask );

            for( unsigned int devIdx : glActiveDevices )
            {
                if( LWDADevice* lwrrLwdaDev = deviceCast<LWDADevice>( m_allDevices[devIdx] ) )
                {
                    char         devLuidBuf[8] = {};
                    unsigned int devNodeMask   = 0;
                    lwrrLwdaDev->lwdaDevice().getLuidAndNodeMask( devLuidBuf, &devNodeMask );
                    LUID* devLuid = reinterpret_cast<LUID*>( devLuidBuf );

                    bool luidsAreEqual =
                        glContextLuid->LowPart == devLuid->LowPart && glContextLuid->HighPart == devLuid->HighPart;
                    if( luidsAreEqual && glContextNodeMask == devNodeMask )
                    {
                        m_glInteropDevice = m_allDevices[devIdx];
                        break;
                    }
                }
            }
        }
#endif  // _WIN32
    }

    // If we haven't set the m_glInteropDevice by now, we are trying to use a GL device that is visible
    // but not active, and fall back to the CPU. There is an ongoing dislwssion if this is a common enough
    // case to be addressed, for example, when a user has one device dedicated for optix and another
    // for regular display. We could extend GLinterop and MultiGPU to do device2device copies to use
    // the GL device even when not active (from the user point of view). This may have wider implications
    // that need to be addressed.
    return m_glInteropDevice;
#endif
}

#ifdef _WIN32
int DeviceManager::wglDevice( HGPULW hGpu )
{
    DeviceManager temp( nullptr, false );

    LWresult     returnResult = LWDA_SUCCESS;
    lwca::Device glDevice     = lwca::Device::lwWGLGetDevice( hGpu, &returnResult );
    if( returnResult != LWDA_SUCCESS )
        throw LwdaError( RT_EXCEPTION_INFO, "lwWGLGetDevice", returnResult );

    Device* device = findLwdaDevice( glDevice, temp.m_visibleDevices );
    if( !device )
        throw IlwalidOperation( RT_EXCEPTION_INFO,
                                "LWCA device corresponding to GL context not found in OptiX device list" );

    return device->visibleDeviceListIndex();
}
#endif  // _WIN32

void DeviceManager::getDeviceAttribute( int ordinal, RTdeviceattribute attrib, RTsize size, void* p )
{
    getAPIDeviceAttributes( ordinal ).getAttribute( attrib, size, p );
}

const APIDeviceAttributes& DeviceManager::getAPIDeviceAttributes( int ordinal )
{
    return g_deviceAttributes.getAPIDeviceAttributes( ordinal );
}

DriverVersion DeviceManager::getMinimumRequiredDriverVersion()
{
#if defined( _WIN32 )
    return DriverVersion( 415, 0 );
#elif defined( __linux__ )
    return DriverVersion( 415, 0 );
#elif defined( __APPLE__ )
    return DriverVersion( 0, 0 );
#else
#error Not implemented for this platform
    return DriverVersion( 0, 0 );
#endif
}

void DeviceManager::disableDeviceAndUpdatePeerAccess( optix::Device* device )
{
    // Peer devices can no longer access this device. Mark their access as invalid.
    if( deviceCast<LWDADevice>( device ) )
    {
        for( LWDADevice* peerLwdaDev : LWDADeviceArrayView( m_activeDevices ) )
        {
            peerLwdaDev->ilwalidatePeerAccess( *device );
        }
    }

    device->disable();
}

void DeviceManager::setStreamPoolSize( const int value )
{
    for( LWDADevice* deviceX : LWDADeviceArrayView( m_activeDevices ) )
    {
        if( deviceX->isEnabled() )
            deviceX->setStreamPoolSize( value );
    }
}

void DeviceAttributesContainer::initialize()
{
    if( m_initialized )
        return;

    std::lock_guard<std::mutex> lock( m_initializationLock );
    if( m_initialized )
        return;

    DeviceManager temp( nullptr, false );
    m_attributes.reserve( temp.visibleDevices().size() );

    for( size_t devId = 0; devId < temp.visibleDevices().size(); ++devId )
    {
        APIDeviceAttributes attributes;
        optix::Device*      device = temp.visibleDevices()[devId];
        device->getAPIDeviceAttributes( attributes );
        m_attributes.push_back( attributes );
    }

    m_initialized = true;
}

unsigned int DeviceAttributesContainer::getDeviceCount()
{
    initialize();
    return m_attributes.size();
}

const APIDeviceAttributes& DeviceAttributesContainer::getAPIDeviceAttributes( int optixOrdinal )
{
    initialize();
    if( optixOrdinal >= static_cast<int>( m_attributes.size() ) || optixOrdinal < 0 )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Ordinal argument is outside bounds of number of devices" );

    return m_attributes[optixOrdinal];
}
