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

#include <Device/Device.h>


namespace optix {

class CPUDevice : public Device
{
  public:
    CPUDevice( Context* context );
    ~CPUDevice() override;

    // Methods inherited from Device
    void        enable() override;
    void        disable() override;
    std::string deviceName() const override;
    void dump( std::ostream& out ) const override;

    // Set the number of CPU threads to use on the CPU
    void setNumThreads( int numThreads );

    // Query the current number of CPU threads to use on the CPU
    int getNumThreads() const;

    // Get the amount of available memory
    size_t getAvailableMemory() const override;

    size_t getTotalMemorySize() const override;

    // does this support HW bindless textures?
    bool supportsHWBindlessTexture() const override;

    bool supportsTTU() const override;

    bool supportsMotionTTU() const override;

    bool supportsLwdaSparseTextures() const override;
    bool supportsTextureFootprint() const override;

    // Queries if a device may directly access a peer device's memory.
    bool canAccessPeer( const Device& peerDev ) const override;

    // Query an attribute on this device
    void getAPIDeviceAttributes( APIDeviceAttributes& attributes ) const override;

    // Returns true if the devices are compatible (same kind and same compute capability)
    bool isCompatibleWith( const Device* otherDevice ) const override;


  private:
    int m_numThreads;
    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( DeviceType type ) const override;

    static const DeviceType m_deviceType{CPU_DEVICE};
};

inline bool CPUDevice::isA( DeviceType type ) const
{
    return type == m_deviceType || Device::isA( type );
}

}  // namespace optix
