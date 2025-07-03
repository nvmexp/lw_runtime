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

#include <Context/UpdateManager.h>
#include <Device/DeviceSet.h>
#include <memory>


//
// When adding a property to a plan implementation, you need to do the following:
//
// 1. Add data members to hold the property -- but only the minimum set of
//    things required to produce a unique kernel.
// 2. Extend isCompatibleWith to distinguish the specialization from other plans.
// 3. Add reporting in summaryString for the new information.
// 4. Subscribe to all events that will potentially ilwalidate that assumption
//    and either update the plan or ilwalidate it.  In many cases the event system
//    will need to be extended to facilitate this.
// 5. Use the data in Compile to produce a better kernel.
//


namespace optix {
class Context;
class FrameTask;

class Plan : public UpdateEventListenerNop
{
  public:
    Plan& operator=( const Plan& ) = delete;
    Plan( const Plan& )            = delete;

    ~Plan() override;

    virtual std::string summaryString() const = 0;
    virtual bool supportsLaunchConfiguration( unsigned int entry, int dimensionality, const DeviceSet& devices, int numLaunchDevices ) const = 0;
    virtual bool isCompatibleWith( const Plan* otherPlan ) const = 0;
    bool isValid() const;

    virtual void compile() const = 0;
    void         revalidate();
    FrameTask*   getTask() const;
    bool         hasBeenCompiled() const;
    DeviceSet    getDevices() const;

    // Sets the counter which can be used to do things such as printout out files.
    void setKernelLaunchCounter( size_t counter );

    // Ilwalidate the plan. This is public so that delegates can cause
    // plan ilwalidations
    void ilwalidatePlan();

    // Nuclear event always ilwalidates the plan
    void eventNuclear();

  protected:
    Plan( Context* context, const DeviceSet& devices );
    void setTask( std::unique_ptr<FrameTask> task ) const;
    std::string status() const;

    Context*        m_context = nullptr;
    const DeviceSet m_devices;
    int             m_kernelLaunchCounter = 0;

  private:
    mutable std::unique_ptr<FrameTask> m_task;
    bool                               m_isValid = true;
};
}
