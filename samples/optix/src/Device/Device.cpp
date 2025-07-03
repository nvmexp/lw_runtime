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

#include <Device/Device.h>

#include <Context/Context.h>
#include <Device/DeviceManager.h>

#include <prodlib/exceptions/Assert.h>

#include <sstream>

using namespace optix;


Device::Device( Context* context )
    : m_context( context )
    , m_enabled( false )
{
    resetNumbering();
}

Device::~Device()
{
}

bool Device::isActive() const
{
    return m_activeDeviceListIndex != ILWALID_DEVICE;
}

bool Device::isEnabled() const
{
    return m_enabled;
}

unsigned int Device::allDeviceListIndex() const
{
    RT_ASSERT( m_allDeviceListIndex != ILWALID_DEVICE );
    return m_allDeviceListIndex;
}

unsigned int Device::visibleDeviceListIndex() const
{
    RT_ASSERT( m_visibleDeviceListIndex != ILWALID_DEVICE );
    return m_visibleDeviceListIndex;
}

unsigned int Device::activeDeviceListIndex() const
{
    RT_ASSERT( m_activeDeviceListIndex != ILWALID_DEVICE );
    return m_activeDeviceListIndex;
}

unsigned int Device::uniqueDeviceListIndex() const
{
    RT_ASSERT( m_uniqueDeviceListIndex != ILWALID_DEVICE );
    return m_uniqueDeviceListIndex;
}

void Device::resetNumbering()
{
    m_allDeviceListIndex = m_visibleDeviceListIndex = m_activeDeviceListIndex = m_uniqueDeviceListIndex = ILWALID_DEVICE;
}

void Device::setAllDeviceListIndex( unsigned int i )
{
    m_allDeviceListIndex = i;
}

void Device::setVisibleDeviceListIndex( unsigned int i )
{
    m_visibleDeviceListIndex = i;
}

void Device::setActiveDeviceListIndex( unsigned int i )
{
    m_activeDeviceListIndex = i;
}

void Device::setUniqueDeviceListIndex( unsigned int i )
{
    m_uniqueDeviceListIndex = i;
}

std::vector<int> Device::getCompatibleOrdinals() const
{
    DeviceManager    dm{nullptr, false};
    std::vector<int> compatibleOrdinals;
    compatibleOrdinals.push_back( 0 );
    const DeviceArray& visibleDevices              = dm.visibleDevices();
    const bool         thisDeviceSupportsTTU       = supportsTTU();
    const bool         thisDeviceSupportsMotionTTU = supportsMotionTTU();
    for( size_t i = 0; i < visibleDevices.size(); ++i )
    {
        if( visibleDevices[i]->supportsTTU() == thisDeviceSupportsTTU && visibleDevices[i]->supportsMotionTTU() == thisDeviceSupportsMotionTTU )
        {
            compatibleOrdinals.push_back( static_cast<int>( i ) );
        }
    }
    compatibleOrdinals[0] = static_cast<int>( compatibleOrdinals.size() ) - 1;
    return compatibleOrdinals;
}

void Device::dump( std::ostream& out, const std::string& title ) const
{
    out << title << " Device [allIndex " << m_allDeviceListIndex << "]";
    if( !isActive() )
        out << "  INACTIVE";
    if( !isEnabled() )
        out << "  NOT ENABLED";
    out << std::endl;
    out << "  visible device number: " << m_visibleDeviceListIndex << std::endl;
    if( isActive() )
        out << "  active device number : " << m_activeDeviceListIndex << std::endl;
}
