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

#include <Device/DeviceSet.h>
#include <ExelwtionStrategy/FrameStatus.h>

#include <memory>
#include <vector>

namespace optix {

class ExelwtionStrategy;

// Base class for objects grouping resources needed per launch.
//
// Each subclass per exelwtion strategy can extend this to allocate, and
// potentially share and reuse, resources needed during a launch.
//
// Each ExelwtionStrategy class is responsible for the management of Launch Resources,
// so it is there that individual resources are allocated, released and reused.
//
// When more than one launch of the same frame task is in flight, the launch resources
// contain the unique elements for that launch that remain reserved for the duration
// of the launch.
//
// In addition to the unique elements of the launch, shared elements across launches
// can be set in a launch resource, as determined by the exelwtion strategy. For example,
// launches that are determined to be exelwted one after another may use the same
// resources, as they are not overlapping. This is determined by the exelwtion strategy.
//
// Some resources needed by all exelwtion strategies have been promoted to this
// class. Therefore, Single, Null and Megakernel LaunchResources have been optimized
// away and use the base class directly.
//
class LaunchResources
{
  public:
    LaunchResources( ExelwtionStrategy* es, const DeviceSet& devices );
    virtual ~LaunchResources();

    // Disallow copying
    LaunchResources( const LaunchResources& ) = delete;
    LaunchResources& operator=( const LaunchResources& ) = delete;

    // Access the frame status
    unsigned int       getNumberOfActiveDevices();
    cort::FrameStatus& getFrameStatusHostPtr( int activeDeviceListIndex );
    void resetHostFrameStatus();

    const DeviceSet& getDevices() const;

  protected:
    ExelwtionStrategy* m_exelwtionStrategy;
    DeviceSet          m_devices;

    std::vector<cort::FrameStatus> m_hostFrameStatus;

    //private:
};
}
