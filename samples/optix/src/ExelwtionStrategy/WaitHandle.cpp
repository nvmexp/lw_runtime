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

#include <Exceptions/LaunchFailed.h>
#include <ExelwtionStrategy/LaunchResources.h>
#include <ExelwtionStrategy/WaitHandle.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix;

WaitHandle::WaitHandle( std::shared_ptr<LaunchResources> launchResources )
    : m_launchResources( launchResources )
{
}

WaitHandle::~WaitHandle()
{
}

void WaitHandle::checkFrameStatus() const
{
    unsigned int numActiveDevices = m_launchResources->getNumberOfActiveDevices();
    for( unsigned int i = 0; i < numActiveDevices; ++i )
    {
        cort::FrameStatus& status = m_launchResources->getFrameStatusHostPtr( i );
        if( status.failed != cort::FrameStatus::FRAME_STATUS_NO_ERROR )
            throw LaunchFailed( RT_EXCEPTION_INFO, "Bad frame status: " + std::to_string( status.failed ) );
    }
}

cort::FrameStatus* WaitHandle::getFrameStatusHostPtr( int activeDeviceListIndex )
{
    return &( m_launchResources->getFrameStatusHostPtr( activeDeviceListIndex ) );
}

std::shared_ptr<LaunchResources>& WaitHandle::getLaunchResources()
{
    return m_launchResources;
}

void WaitHandle::releaseResources()
{
    m_launchResources.reset();
}
