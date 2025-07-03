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

#include <LWCA/Device.h>
#include <Context/Context.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <ExelwtionStrategy/ExelwtionStrategy.h>
#include <ExelwtionStrategy/RTX/RTXLaunchResources.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/system/Logger.h>

using namespace optix;

RTXLaunchResources::RTXLaunchResources( ExelwtionStrategy* es, const DeviceSet& devices, const optix::lwca::Stream& syncStream )
    : LaunchResources( es, devices )
    , m_syncStream( syncStream )
{
    m_deviceFrameStatus.resize( devices.count(), 0 );
    m_launchBuffers.resize( devices.count(), std::make_pair( 0, 0 ) );
    m_scratchBuffers.resize( devices.count(), std::make_pair( 0, 0 ) );
}

RTXLaunchResources::~RTXLaunchResources()
{
    // This needs to be called by the lowest subclass destructor, otherwise instance members
    // will be deallocated before they are handled by the ExelwtionStrategy in charge,
    // and bad things will happen.
    m_exelwtionStrategy->releaseLaunchResources( this );
}

lwca::Stream RTXLaunchResources::getStream( int allDeviceListIndex ) const
{
    Device* device = m_exelwtionStrategy->getContext()->getDeviceManager()->allDevices()[allDeviceListIndex];
    RT_ASSERT( device );
    LWDADevice* lwdaDevice = deviceCast<LWDADevice>( device );
    RT_ASSERT( lwdaDevice );
    return lwdaDevice->getLwdaStream( m_streamIndices[m_devices.getArrayPosition( allDeviceListIndex )] );
}

RtcCommandList RTXLaunchResources::getRtcCommandList( int allDeviceListIndex ) const
{
    Device* device = m_exelwtionStrategy->getContext()->getDeviceManager()->allDevices()[allDeviceListIndex];
    RT_ASSERT( device );
    LWDADevice* lwdaDevice = deviceCast<LWDADevice>( device );
    RT_ASSERT( lwdaDevice );
    return lwdaDevice->getRtcCommandList( m_streamIndices[m_devices.getArrayPosition( allDeviceListIndex )] );
}

const lwca::Stream& RTXLaunchResources::getSyncStream() const
{
    return m_syncStream;
}
