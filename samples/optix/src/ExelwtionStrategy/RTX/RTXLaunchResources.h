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

#pragma once

#include <LWCA/Stream.h>
#include <Device/MaxDevices.h>
#include <ExelwtionStrategy/LaunchResources.h>
#include <Util/DevicePtr.h>
#include <rtcore/interface/types.h>

#include <atomic>
#include <memory>
#include <utility>
#include <vector>

namespace optix {

class RTXLaunchResources : public LaunchResources
{
  public:
    RTXLaunchResources( ExelwtionStrategy* es, const DeviceSet& devices, const lwca::Stream& syncStream );
    virtual ~RTXLaunchResources();

    // The streams used by this launch (one per device)
    lwca::Stream getStream( int allDeviceListIndex ) const;
    RtcCommandList getRtcCommandList( int allDeviceListIndex ) const;

    // The synchronization stream to use, or empty if no stream is set. The launch must make sure
    // that any work queued up on this stream finishes first.
    const lwca::Stream& getSyncStream() const;

  protected:
    // RTXES is responsible for acquire/release of RTXLaunchResources. RTXFrameTask is responsible of the launch.
    // RTXWaitHandle is using the events
    friend class RTXES;
    friend class RTXFrameTask;
    friend class RTXWaitHandle;

    typedef std::pair<LWdeviceptr, size_t> devptr_size_t;

    std::vector<LWdeviceptr>   m_deviceFrameStatus;
    std::vector<devptr_size_t> m_launchBuffers;
    std::vector<devptr_size_t> m_scratchBuffers;

    const lwca::Stream m_syncStream;

    std::vector<unsigned int> m_streamIndices;

    // Events used by the RTXWaitHandle
    std::vector<lwca::Event> m_t0;
    std::vector<lwca::Event> m_t1;
};
}
