// Copyright (c) 2018, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <ExelwtionStrategy/ExelwtionStrategy.h>
#include <ExelwtionStrategy/LaunchResources.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix;

LaunchResources::LaunchResources( ExelwtionStrategy* es, const DeviceSet& devices )
    : m_exelwtionStrategy( es )
    , m_devices( devices )
{
    m_hostFrameStatus.resize( devices.count() );
}

LaunchResources::~LaunchResources()
{
}

unsigned int LaunchResources::getNumberOfActiveDevices()
{
    return m_devices.count();
}

cort::FrameStatus& LaunchResources::getFrameStatusHostPtr( int activeDeviceListIndex )
{
    RT_ASSERT( static_cast<unsigned int>( activeDeviceListIndex ) < m_hostFrameStatus.size() );
    return m_hostFrameStatus[activeDeviceListIndex];
}

void LaunchResources::resetHostFrameStatus()
{
    m_hostFrameStatus.clear();
    m_hostFrameStatus.resize( m_devices.count() );
}

const DeviceSet& LaunchResources::getDevices() const
{
    return m_devices;
}
