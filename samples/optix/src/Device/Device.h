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

#include <iosfwd>
#include <string>
#include <vector>


namespace optix {

class Context;
struct APIDeviceAttributes;

enum DeviceType
{
    ABSTRACT_DEVICE,
    LWDA_DEVICE,
    CPU_DEVICE
};

class Device
{
  public:
    Device( const Device& ) = delete;
    Device& operator=( const Device& ) = delete;
    virtual ~Device();


    // We use the following definition to distinguish active and enable: a device
    // is active if it is in the active device list (can potentially be used), a
    // device is enabled if the device is going to be used, as in the LWCA
    // context is created.  This means a device can be active but not enabled
    // yet. Enabled can also be interpreted as "readyForUse".  It is however
    // diffilwlt to express the action of disabling a device (what is the action
    // that makes something not ready?) so we keep it short to the enable/disable
    // pair.

    // Return true if this device is in the "active devices" list.
    // Note that enable() may not have been called yet, since that
    // is done lazily.
    bool isActive() const;

    // Return true if the device is enabled, i.e., enable() has been called.
    bool isEnabled() const;

    // Enable the device, including any necessary context
    // creation. Note that this method can be called on an already
    // active device.
    virtual void enable() = 0;

    // Disable the device and free up any associated memory. Note
    // that this method can be called on a previously inactive device.
    virtual void disable() = 0;

    // Get the index of this device in the "all devices" list
    unsigned int allDeviceListIndex() const;

    // Get the index of this device in the "visible devices" list
    unsigned int visibleDeviceListIndex() const;

    // Get the index of this device in the "active devices" list
    unsigned int activeDeviceListIndex() const;

    // Get the index of this device in the "unique devices" list
    unsigned int uniqueDeviceListIndex() const;

    // Return a string with the brief name of the device
    virtual std::string deviceName() const = 0;

    // Print all available information about the device to the specified stream
    virtual void dump( std::ostream& out ) const = 0;

    // Get the amount of free memory available on this device. Virtual overloads for both CPU and LWCA
    // devices are provided
    virtual size_t getAvailableMemory() const = 0;
    virtual size_t getTotalMemorySize() const = 0;

    // does this support HW bindless textures?
    virtual bool supportsHWBindlessTexture() const = 0;

    //
    virtual bool supportsTTU() const       = 0;
    virtual bool supportsMotionTTU() const = 0;

    virtual bool supportsLwdaSparseTextures() const = 0;
    virtual bool supportsTextureFootprint() const = 0;

    // Queries if a device may directly access a peer device's memory.
    // SGP: why is this Device& when ALL other things use Device*?
    virtual bool canAccessPeer( const Device& peerDev ) const = 0;

    // Get a device-specific API attribute (either CPU or GPU specific)
    virtual void getAPIDeviceAttributes( APIDeviceAttributes& attributes ) const = 0;

    // Returns true if the devices are compatible (same kind and same compute capability)
    virtual bool isCompatibleWith( const Device* otherDevice ) const = 0;

    static const unsigned int ILWALID_DEVICE = 0xffff;


  protected:
    std::vector<int> getCompatibleOrdinals() const;
    virtual void dump( std::ostream& out, const std::string& header ) const;
    Device( Context* context );
    Context* m_context;
    bool     m_enabled;

    friend class DeviceManager;
    void resetNumbering();
    void setAllDeviceListIndex( unsigned int i );
    void setVisibleDeviceListIndex( unsigned int i );
    void setActiveDeviceListIndex( unsigned int i );
    void setUniqueDeviceListIndex( unsigned int i );

  private:
    unsigned int m_allDeviceListIndex;
    unsigned int m_visibleDeviceListIndex;
    unsigned int m_activeDeviceListIndex;
    unsigned int m_uniqueDeviceListIndex;

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    virtual bool isA( DeviceType type ) const;

    static const DeviceType m_deviceType{ABSTRACT_DEVICE};
};

template <typename T>
T* deviceCast( Device* device )
{
    return device && device->isA( T::m_deviceType ) ? reinterpret_cast<T*>( device ) : nullptr;
}

template <typename T>
const T* deviceCast( const Device* device )
{
    return device && device->isA( T::m_deviceType ) ? reinterpret_cast<const T*>( device ) : nullptr;
}

inline bool Device::isA( DeviceType type ) const
{
    return type == m_deviceType;
}

typedef std::vector<Device*> DeviceArray;
}  // namespace optix
