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

#include <ExelwtionStrategy/ExelwtionStrategy.h>
#include <ExelwtionStrategy/FrameTask.h>
#include <ExelwtionStrategy/LaunchResources.h>

using namespace optix;

ExelwtionStrategy::ExelwtionStrategy( Context* context )
    : m_context( context )
{
}

ExelwtionStrategy::~ExelwtionStrategy()
{
}


std::shared_ptr<LaunchResources> ExelwtionStrategy::acquireLaunchResources( const DeviceSet& devices,
                                                                            const FrameTask* /*ft*/,
                                                                            const optix::lwca::Stream& syncStream,
                                                                            const unsigned int /*width*/,
                                                                            const unsigned int /*height*/,
                                                                            const unsigned int /*depth*/ )
{
    std::shared_ptr<LaunchResources> res( new LaunchResources( this, devices ) );
    return res;
}

void ExelwtionStrategy::releaseLaunchResources( LaunchResources* /*launchResources*/ )
{
    // no-op.
    // Default LaunchResources are cleared in the destructor and require no special handling.
}

Context* ExelwtionStrategy::getContext()
{
    return m_context;
}
