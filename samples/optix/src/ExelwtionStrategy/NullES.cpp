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

#include <Context/Context.h>
#include <Device/DeviceManager.h>
#include <ExelwtionStrategy/FrameTask.h>
#include <ExelwtionStrategy/NullES.h>
#include <memory>
#include <prodlib/exceptions/Assert.h>

using namespace optix;

NullES::NullES( Context* context )
    : ExelwtionStrategy( context )
{
}

NullES::~NullES()
{
}

std::unique_ptr<Plan> NullES::createPlan( unsigned int entry, int dimensionality, const DeviceSet& devices, int numLaunchDevices ) const
{
    return std::unique_ptr<Plan>( new NullESPlan( m_context, devices ) );
}

NullESPlan::NullESPlan( Context* context, const DeviceSet& devices )
    : Plan( context, devices )
{
}

NullESPlan::~NullESPlan()
{
}

std::string NullESPlan::summaryString() const
{
    return "NullES" + status();
}

bool NullESPlan::supportsLaunchConfiguration( unsigned int entry, int dimensionality, const DeviceSet& devices, int numLaunchDevices ) const
{
    return true;
}

bool NullESPlan::isCompatibleWith( const Plan* otherPlan ) const
{
    const NullESPlan* other = dynamic_cast<const NullESPlan*>( otherPlan );
    if( !other )
        return false;

    // All are compatible
    return true;
}

void NullESPlan::compile() const
{
    std::unique_ptr<FrameTask> task( new NullESFrameTask( m_context ) );
    setTask( std::move( task ) );
}

void NullESPlan::update() const
{
}

NullESFrameTask::NullESFrameTask( Context* context )
    : FrameTask( context )
{
}

NullESFrameTask::~NullESFrameTask()
{
}

void NullESFrameTask::activate()
{
}

void NullESFrameTask::deactivate()
{
}

void NullESFrameTask::launch( std::shared_ptr<WaitHandle> waiter,
                              unsigned int                entry,
                              int                         dimensionality,
                              RTsize                      width,
                              RTsize                      height,
                              RTsize                      depth,
                              unsigned int                subframe_index,
                              const cort::AabbRequest&    aabbParams )
{
}

std::shared_ptr<WaitHandle> optix::NullESFrameTask::acquireWaitHandle( std::shared_ptr<LaunchResources>& launchResources )
{
    return std::shared_ptr<WaitHandle>( new NullWaitHandle( launchResources ) );
}

NullWaitHandle::NullWaitHandle( std::shared_ptr<LaunchResources> launchResources )
    : WaitHandle( launchResources )
{
}

NullWaitHandle::~NullWaitHandle()
{
}

void NullWaitHandle::block()
{
}

float NullWaitHandle::getElapsedMilliseconds() const
{
    return 0;
}
