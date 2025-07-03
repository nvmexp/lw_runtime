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

#include <LWCA/Context.h>
#include <LWCA/Device.h>
#include <Device/Device.h>
#include <Device/DeviceSet.h>
#include <Device/DriverVersion.h>

#include <corelib/misc/Concepts.h>

#include <o6/optix.h>

#include <memory>
#include <vector>


namespace optix {

class Context;
class CPUDevice;
class LWDADevice;
struct APIDeviceAttributes;

//
// Manage devices, including CPU, GPU and (eventually) Nitro.
//

class DeviceManager : private corelib::NonCopyable
{
  public:
    DeviceManager( Context* context, bool useRtxDataModel );

    ~DeviceManager();

    // Returns the number of OptiX-compatible devices, can be called without any context
    static unsigned int getDeviceCount();

    // Queries a device attribute from an OptiX ordinal (not a LWCA
    // ordinal, this also supports CPU devices)
    static void getDeviceAttribute( int ordinal, RTdeviceattribute attrib, RTsize size, void* p );
    static const APIDeviceAttributes& getAPIDeviceAttributes( int ordinal );

    // Returns the minimum required driver version.
    static DriverVersion getMinimumRequiredDriverVersion();

    // Pointers to all devices, whether they are visible in the API or
    // not (including CPU devices)
    const DeviceArray& allDevices() const;

    // Pointers to OptiX API visible devices
    const DeviceArray& visibleDevices() const;

    // The lwrrently active devices
    const DeviceArray& activeDevices() const;

    // A set of unique devices (from the active set)
    const DeviceArray& uniqueActiveDevices() const;

    // Change the active devices
    void setActiveDevices( const DeviceArray& newDevices );

    // Get the primary device (usually the first device)
    Device* primaryDevice();

    // Get the primary LWCA device (usually the first device).  Returns NULL if no LWCA
    // devices present.
    LWDADevice* primaryLWDADevice();

    // Find the CPU device
    CPUDevice* cpuDevice();

    // Find the GL interop device
    Device* glInteropDevice();

#ifdef _WIN32
    // Find the GL device index in visible device list on Windows
    static int wglDevice( HGPULW hGpu );
#endif

    // Return the LWCA device that lwrrently has the least amount of free memory
    LWDADevice* leastFreeMemLWDADevice( size_t* bytesFree = nullptr, size_t* bytesTotal = nullptr ) const;

    // Enable the active devices, including any context creation necessary.
    // Also disable devices if they are inactive.
    void enableActiveDevices();

    // Shutdown the devices cleanly
    void shutdownDevices();

    // Print the list of available devices
    void printAvailableDevices() const;

    // Return the peer-to-peer islands for the lwrrently active devices.
    const std::vector<DeviceSet>& getLwlinkIslands() const;

    // Return the peer-to-peer island the specified device belongs to.
    DeviceSet getLwlinkIsland( Device* device ) const;

    // Return whether any two devices can communicate via Lwlink (i.e.
    // whether or not it's worth attempting P2P with input buffers)
    bool anyActiveDevicesAreLwlinkEnabled() const;

    // Return whether or all devices are fully interconnected.
    bool isPeerToPeerFullyConnected() const;

    // Returns true if the active devices support the TTU
    bool activeDevicesSupportTTU() const;
    bool activeDevicesSupportMotionTTU() const;

    bool activeDevicesSupportLwdaSparseTextures() const;
    bool activeDevicesSupportTextureFootprint() const;

    // Set the maximum number of streams per LWCA device
    void setStreamPoolSize( const int value );

  private:
    Context* m_context = nullptr;

    DeviceArray m_allDevices;
    DeviceArray m_visibleDevices;
    DeviceArray m_activeDevices;
    DeviceArray m_uniqueActiveDevices;

    // The device used for Gl interop.
    Device* m_glInteropDevice = nullptr;

    // Whether or not we should select a new interop device.
    bool m_selectNewGlInteropDevice = true;

    // Peer to peer islands of active devices. An island is a set of devices
    // that can communicate via fast peer-to-peer, e.g. LWLINK.
    std::vector<DeviceSet> m_activeDeviceLwlinkIslands;
    bool                   m_checkPeerAccessOnEnable = true;

    // Whether or not all of the devices can access each other via P2P.
    bool m_fullyConnected = false;

    bool m_activeDevicesSupportTTU       = false;
    bool m_activeDevicesSupportMotionTTU = false;
    // Sets m_activeDevicesSupportTTU if the active devices support the TTU
    void setActiveDevicesSupportTTU();

    bool m_activeDevicesSupportLwdaSparseTextures = false;
    bool m_activeDevicesSupportTextureFootprint   = false;
    void setActiveDevicesSupportLwdaSparseTextures();

    // Initialization
    void setupDevices( bool useRtxDataModel );
    void createAllDevices();
    void filterVisibleDevices();
    void setDefaultActiveDevices( bool useRtxDataModel );
    void    setRtxDefaultActiveDevices();
    void    setUniqueActiveDevices();
    Device* getFirstCompatibleDevice( Device* other ) const;
    void numberDevices();
    void enablePeerAccess( DeviceSet devicesToIgnore );
    void enablePeerAccess();

    // Used to make sure a device's peers are updated when it is disabled.
    void disableDeviceAndUpdatePeerAccess( optix::Device* device );
};
}
