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

#include <LWCA/ComputeCapability.h>
#include <LWCA/Context.h>
#include <LWCA/Device.h>
#include <LWCA/Event.h>
#include <LWCA/Module.h>
#include <LWCA/Stream.h>
#include <Control/ManagedLWDAModule.h>
#include <Device/Device.h>
#include <Device/DeviceSet.h>
#include <Util/DevicePtr.h>

#include <internal/optix_declarations.h>

#include <rtcore/interface/types.h>

#include <prodlib/misc/lwpti/LwptiProfiler.h>

#include <iterator>
#include <map>
#include <vector>

namespace optix {

struct TextureDescriptor;

class LWDADevice : public Device
{
  public:
    LWDADevice( Context* context, int lwdaOrdinal, lwca::ComputeCapability overrideSMVersion );
    ~LWDADevice() override;

    // Methods inherited from Device
    void        enable() override;
    void        disable() override;
    std::string deviceName() const override;
    void dump( std::ostream& out ) const override;
    void dumpShort( std::ostream& out ) const;

    // Make this devices context the lwrrently active device
    void makeLwrrent();

    // Return the LWCA device
    const lwca::Device& lwdaDevice() const;

    // Return the LWCA context
    const lwca::Context& lwdaContext() const;

    // Return the RTcore context
    RtcDeviceContext rtcContext() const;

    // Return the RTcore command list
    RtcCommandList primaryRtcCommandList() const;

    // Return the LWCA device ordinal
    int lwdaOrdinal() const;

    // Return the primary stream for this device
    lwca::Stream primaryStream() const;

    // Return the stream from the pool corresponding to the provided stream index. The index will
    // rotate over the available streams, so both 0 and 2 will return the first stream in a pool
    // of size 2. This makes it possible to use an increasing counter to request the next
    // stream in the pool.
    lwca::Stream getLwdaStream( unsigned int index ) const;

    // Return the list from the pool corresponding to the provided stream index. The index will
    // rotate over the available lists, so both 0 and 2 will return the first stream in a pool
    // of size 2. This makes it possible to use an increasing counter to request the next
    // stream in the pool. The command list of a given index will be created using the stream of
    // the same index.
    RtcCommandList getRtcCommandList( unsigned int index ) const;

    // Sync the primary stream on this device to the primaryStream on the otherDevice
    void syncToDevice( LWDADevice* otherDevice );

    // Sync the provided stream with the primary stream of this device.
    void syncStreamToPrimaryStream( lwca::Stream stream );

    // Return the compute capability of the LWCA device
    lwca::ComputeCapability computeCapability() const;

    // Modules managed by this device.  ManagedLWDAModule can be used
    // like a const lwca::Module*.
    ManagedLWDAModule moduleFromPTX( const std::string& ptx, int launchCounter );

    // TODO: retire these allocators, go through the mem manager instead.
    // Allocate a managed pointer on the device.  Memory will be freed
    // automatically when shared pointer is destroyed.
    template <typename T>
    DevicePtr<T> alloc();

    // Allocate a managed array (not a LWCA array) on the device.
    // Memory will be freed automatically when shared pointer is
    // destroyed.
    template <typename T>
    DevicePtr<T> alloc( size_t numElements );

    // Return the available memory on this device
    size_t getAvailableMemory() const override;

    size_t getTotalMemorySize() const override;

    // Bindless texture is supported on SM_30+
    bool supportsHWBindlessTexture() const override;

    bool supportsTTU() const override;

    bool supportsMotionTTU() const override;

    // LWCA sparse textures are supported on Pascal+
    bool supportsLwdaSparseTextures() const override;
    // Texture footprint instruction is supported on Turing+
    bool supportsTextureFootprint() const override;

    // Queries if a device may directly access a peer device's memory.
    bool canAccessPeer( const Device& peerDev ) const override;

    // Queries if a device may access a peer device's memory via LWLink.
    bool canAccessPeerViaLwlink( const Device& peerDev ) const;

    // Disables direct access to memory allocations in a peer context and unregisters any registered allocations.
    virtual void disablePeerAccess( const Device& peerDev );

    // Marks this device's access to the given peer device as invalid.
    virtual void ilwalidatePeerAccess( const Device& peerDev );

    // Enables direct access to memory allocations in a peer context. Return true if peer access 
    // was successfully enabled, false otherwise.
    virtual bool enablePeerAccess( const Device& peerDev );

    // Queries devices which may be directly accessed (enabled p2p).
    virtual DeviceSet peerEnabled() const;

    // Returns devices that are accessible via LWLink.
    virtual DeviceSet lwlinkAccessible() const;

    // Return true if direct access to memory allocations in a peer context was enabled.
    bool isPeerAccessEnabled( const Device& peerDev ) const;

    // Number of bound textures supported
    unsigned int maximumBoundTextures() const;

    // Number of bindless textures supported
    unsigned int maximumBindlessTextures() const;

    // Query an attribute on this device
    void getAPIDeviceAttributes( APIDeviceAttributes& attributes ) const override;

    // Returns true if the devices are compatible (same kind and same compute capability)
    bool isCompatibleWith( const Device* otherDevice ) const override;

    void setPrintBufferSize( size_t bufferSize );

    void setStreamPoolSize( const int poolSize );

    static bool smVersionSupportsLDG( int smVersion );

    prodlib::LwptiProfiler& getLwptiProfiler() { return m_lwptiProfiler; };

  private:
    int                         m_lwdaOrdinal;
    lwca::ComputeCapability     m_overrideSMVersion;
    lwca::ComputeCapability     m_SMVersion;
    unsigned int                m_architecture               = 0;
    unsigned int                m_architectureImplementation = 0;
    bool                        m_supportsTTU                = false;
    bool                        m_supportsLwdaSparseTextures = false;
    lwca::Device                m_device;
    lwca::Context               m_lwdaContext;
    std::vector<lwca::Stream>   m_streams;
    unsigned int                m_streamCounter = 0;
    lwca::Event                 m_event;
    RtcDeviceContext            m_rtcDeviceContext = nullptr;
    std::vector<RtcCommandList> m_rtcCommandLists;

    // Which peer devices direct access is enabled.
    DeviceSet m_peerAccessEnabled;

    // Peer devices accessible via LWLink.
    DeviceSet m_lwlinkAccessible;

    bool m_isOriBackendOnKeplerAvailable = false;
    bool m_isCompileFastAvailable        = false;

    // The following lwjit options are avaiable only with lwca 8 header files.
    // Since we do not want to rely on lwca 8 we create constants using as baseline the last option available in lwca 7.5
    // We then dynamically check if these options are available for the current device.
    const LWjit_option JIT_OPTION_ORI_ON_KEPLER = static_cast<LWjit_option>( LW_JIT_CACHE_MODE + 1 );
    const LWjit_option JIT_OPTION_COMPILE_FAST  = static_cast<LWjit_option>( LW_JIT_CACHE_MODE + 2 );

    // Returns true if the given binary jit option is avaiable for a device with the given major compute capability.
    bool isJITOptionAvailable( LWjit_option option, int computeCapabilityMajor );

    // Returns true if the given binary jit option is avaialble.
    bool isJITOptionAvailable( LWjit_option option );

    // Get the max number of registers for compilation (-1 = no limit)
    int getMaxRegisters() const;

    //
    // Compile and cache loaded LWCA modules
    //
    struct ModuleCacheEntry
    {
        size_t            hash;
        ManagedLWDAModule module;
    };
    typedef std::map<size_t, ModuleCacheEntry> CachedModulesMap;
    CachedModulesMap m_cachedModules;
    void             pruneModuleCache();

    lwca::Module createFromMemoryPtxLoadData( const std::string& ptx, int maxRegCount );
    lwca::Module compileModuleFromPTX( const std::string& ptx, int launchCounter );

    //
    // Manage memory allocations
    //
    struct AllocationHandle
    {
        AllocationHandle()
            : deviceptr( 0 )
            , size( 0 )
        {
        }
        LWdeviceptr deviceptr;
        size_t      size;
    };
    AllocationHandle* allocate( size_t size );
    void detachAllocationHandle( LWDADevice::AllocationHandle* ah );

    prodlib::LwptiProfiler m_lwptiProfiler;

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( DeviceType type ) const override;

    static const DeviceType m_deviceType{LWDA_DEVICE};
};

template <typename T>
DevicePtr<T> LWDADevice::alloc()
{
    AllocationHandle* ah = allocate( sizeof( T ) );
    return DevicePtr<T>( ah->deviceptr, [this, ah]( T* ) { detachAllocationHandle( ah ); } );
}

template <typename T>
DevicePtr<T> LWDADevice::alloc( size_t numElements )
{
    AllocationHandle* ah = allocate( sizeof( T ) * numElements );
    return DevicePtr<T>( ah->deviceptr, [this, ah]( T* ) { detachAllocationHandle( ah ); } );
}

inline bool LWDADevice::isA( DeviceType type ) const
{
    return type == m_deviceType || Device::isA( type );
}

// An adapter class for a DeviceArray that allows you to write a range-for loop
// over only the LWDADevice pointers in a DeviceArray.
//
class LWDADeviceArrayView
{
  public:
    // The iterator used by the range-for loop.  This is a simple model that doesn't
    // implement the full facility of the Iterator concept.  It's only just enough
    // to allow the range-for loop to work.
    class const_iterator
    {
      public:
        explicit const_iterator( const DeviceArray& devices );
        const_iterator()                            = default;
        const_iterator( const const_iterator& rhs ) = default;
        const_iterator( const_iterator&& rhs )      = default;
        ~const_iterator()                           = default;
        const_iterator& operator=( const const_iterator& rhs ) = default;
        const_iterator& operator=( const_iterator&& rhs ) = default;

        const_iterator operator++();
        LWDADevice* operator*() const;
        bool operator==( const const_iterator& rhs ) const;

      private:
        const DeviceArray*     m_devices = nullptr;
        DeviceArray::size_type m_index   = 0U;
    };

    explicit LWDADeviceArrayView( const DeviceArray& devices )
        : m_devices( devices )
    {
    }
    LWDADeviceArrayView( const LWDADeviceArrayView& rhs ) = delete;
    LWDADeviceArrayView( LWDADeviceArrayView&& rhs )      = delete;
    ~LWDADeviceArrayView()                                = default;
    LWDADeviceArrayView& operator=( const LWDADeviceArrayView& rhs ) = delete;
    LWDADeviceArrayView& operator=( LWDADeviceArrayView&& rhs ) = delete;

    const_iterator begin() const { return const_iterator{m_devices}; }

    const_iterator end() const { return {}; }

  private:
    const DeviceArray& m_devices;
};

inline LWDADeviceArrayView::const_iterator::const_iterator( const DeviceArray& devices )
    : m_devices( &devices )
{
    while( m_index < m_devices->size() && deviceCast<LWDADevice>( ( *m_devices )[m_index] ) == nullptr )
    {
        ++m_index;
    }
}

inline LWDADeviceArrayView::const_iterator LWDADeviceArrayView::const_iterator::operator++()
{
    if( m_devices && m_index < m_devices->size() )
    {
        while( ++m_index < m_devices->size() )
        {
            if( deviceCast<LWDADevice>( ( *m_devices )[m_index] ) != nullptr )
            {
                break;
            }
        }
    }
    return *this;
}

inline LWDADevice* LWDADeviceArrayView::const_iterator::operator*() const
{
    return m_devices && m_index < m_devices->size() ? deviceCast<LWDADevice>( ( *m_devices )[m_index] ) : nullptr;
}

inline bool LWDADeviceArrayView::const_iterator::operator==( const const_iterator& rhs ) const
{
    // Two iterators are the same if any of these conditions are true:
    // 1) They both point to the same index within the same DeviceArray.
    // 2) One iterator is default constructed and the other iterator points at the last element of its DeviceArray.
    // 3) Both iterators are default constructed.
    return ( m_index == rhs.m_index && m_devices == rhs.m_devices )
           || ( m_devices != nullptr && m_index == m_devices->size() && rhs.m_devices == nullptr )
           || ( m_devices == nullptr && rhs.m_devices && rhs.m_index == rhs.m_devices->size() )
           || ( m_devices == nullptr && rhs.m_devices == nullptr );
}

inline bool operator!=( const LWDADeviceArrayView::const_iterator& lhs, const LWDADeviceArrayView::const_iterator& rhs )
{
    return !( lhs == rhs );
}

}  // namespace optix
