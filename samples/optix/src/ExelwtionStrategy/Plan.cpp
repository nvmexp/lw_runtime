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
#include <ExelwtionStrategy/FrameTask.h>
#include <ExelwtionStrategy/Plan.h>
#include <prodlib/system/Logger.h>
#include <sstream>

using namespace optix;

Plan::Plan( Context* context, const DeviceSet& devices )
    : m_context( context )
    , m_devices( devices )
    , m_kernelLaunchCounter( context->getKernelLaunchCount() )
{
    m_context->getUpdateManager()->registerUpdateListener( this );
}

Plan::~Plan()
{
    m_context->getUpdateManager()->unregisterUpdateListener( this );
}

bool Plan::isValid() const
{
    return m_isValid;
}

void Plan::revalidate()
{
    m_isValid = true;
}

void Plan::ilwalidatePlan()
{
    if( m_isValid )
    {
        llog( 20 ) << "Ilwalidated plan: " << summaryString() << '\n';
    }

    m_isValid = false;
}

std::string Plan::status() const
{
    std::ostringstream out;
    out << " [";
    out << ( m_isValid ? "valid" : "invalid" );
    out << ", ";
    out << ( hasBeenCompiled() ? "compiled" : "uncompiled" );
    out << "]";
    return out.str();
}

FrameTask* Plan::getTask() const
{
    return m_task.get();
}

void Plan::setTask( std::unique_ptr<FrameTask> task ) const
{
    m_task = std::move( task );
}

bool Plan::hasBeenCompiled() const
{
    return m_task != nullptr;
}

DeviceSet Plan::getDevices() const
{
    return m_devices;
}

void Plan::setKernelLaunchCounter( size_t counter )
{
    m_kernelLaunchCounter = counter;
}

void Plan::eventNuclear()
{
    ilwalidatePlan();
}
