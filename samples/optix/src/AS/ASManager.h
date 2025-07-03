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

#include <Device/DeviceManager.h>
#include <Objects/Acceleration.h>
#include <Objects/Buffer.h>
#include <Objects/Group.h>
#include <Objects/Program.h>
#include <Util/IndexedVector.h>
#include <Util/Trace.h>
#include <prodlib/system/Thread.h>

#include <o6/optix.h>

namespace optix {
class Buffer;
class Context;

class ASManager
{
  public:
    ASManager( Context* context );
    ~ASManager();


    // Interface from Acceleration to manage dirty lists.
    void addOrRemoveDirtyGeometryGroupAccel( Acceleration* accel, bool add );
    void addOrRemoveDirtyGroupAccel( Acceleration* accel, bool add );

    // Interface from AbstractGroup to manage dirty children buffers.
    void addOrRemoveDirtyGroup( AbstractGroup* group, bool add );
    void addOrRemoveDirtyTopLevelTraversable( Acceleration* accel, bool add );

    // Interface from context.
    void buildAccelerationStructures();
    void setupPrograms( bool supportMotion );
    void setupInitialPrograms();

    // Interface from device manager.
    void preSetActiveDevices( const DeviceArray& removedDevices );


  private:
    Context* m_context = nullptr;

    size_t        m_minRingBufferSize = 0;
    MBufferHandle m_ringBuffer;
    bool          m_building          = false;
    bool          m_needsInitialSetup = true;

    // Dirty geometry group accels that need to be built
    IndexedVector<Acceleration*, Acceleration::dirtyListIndex_fn> m_dirtyGeometryGroupAccels;

    // Dirty group accels that need to be built
    IndexedVector<Acceleration*, Acceleration::dirtyListIndex_fn> m_dirtyGroupAccels;

    // Dirty groups that need to have their children buffers filled
    IndexedVector<AbstractGroup*, AbstractGroup::dirtyListIndex_fn> m_dirtyGroups;

    // Dirty top level traversables
    IndexedVector<Acceleration*, Acceleration::dirtyTopLevelTraversableListIndex_fn> m_dirtyTopLevelTraversables;

    // Utility functions
    unsigned int buildAccelerationStructures( const std::vector<Acceleration*>& accels, DeviceSet& buildDevices, CPUDevice* cpuDevice );
    void getSortedDirtyAccels( std::vector<Acceleration*>& sortedAccels, const std::vector<Acceleration*>& dirtyGroupAccels );
    void setMinRingBufferSize();
    void resizeTempBuffer( const std::vector<Acceleration*>& accels, const DeviceSet& buildDevices );
    void buildAccels( const std::vector<Acceleration*>& accels, DeviceSet buildDevices, Device* cpuDevice, unsigned int& totalPrimitives );
};

}  // namespace optix
