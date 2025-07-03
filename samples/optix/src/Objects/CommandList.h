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

#include <LWCA/Event.h>
#include <Objects/ManagedObject.h>
#include <Util/ReusableIDMap.h>

#include <string>

namespace optix {

class PostprocessingStage;

class CommandList : public ManagedObject
{
    //////////////////////////////////////////////////////////////////////////
  public:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // CTOR / DTOR
    //------------------------------------------------------------------------
    CommandList( Context* context );
    ~CommandList() override;


    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    // Append a postprocessing stage to the command list
    void appendPostprocessingStage( PostprocessingStage* stage, RTsize launch_width, RTsize launch_height );

    // Append a 1D launch to the command list
    void appendLaunch( unsigned int entryIndex, RTsize launch_width );

    // Append a 2D launch to the command list
    void appendLaunch( unsigned int entryIndex, RTsize launch_width, RTsize launch_height );

    // Append a 3D launch to the command list
    void appendLaunch( unsigned int entryIndex, RTsize launch_width, RTsize launch_height, RTsize launch_depth );

    // Sets the lwca synchronization stream for this command list.
    void setLwdaStream( void* stream );

    // Sets the lwca synchronization stream for this command list.
    void getLwdaStream( void** stream );

    // Finalize the command list so that it can be called, later
    void finalize();

    // Sets the devices to use for this command list. Must be a subset of the active devices.
    void setDevices( std::vector<unsigned int>& deviceList );

    // Gets the devices lwrrently set for this command list.
    std::vector<unsigned int> getDevices();

    // Excelwte the command list. Can only be called after finalizing it
    void execute() const;

    // LinkedPtr relationship management
    void detachLinkedChild( const LinkedPtr_Link* link );

  private:
    // Exelwtes the command list asynchronously.
    void exelwteAsync() const;

    // Utility functions to sync memory manager
    void activateMemoryManager( bool& active ) const;
    void deactivateMemoryManager( bool& active ) const;

    struct CommandDescriptor
    {
        unsigned int stageOrEntryIndex;
        RTsize       width;
        RTsize       height;
        RTsize       depth;
        unsigned int dim;
        bool         isLaunch;
        CommandDescriptor( unsigned int stage_or_entry_index_, unsigned int dim_, RTsize launch_width_, RTsize launch_height_, RTsize launch_depth_, bool is_launch_ )
            : stageOrEntryIndex( stage_or_entry_index_ )
            , width( launch_width_ )
            , height( launch_height_ )
            , depth( launch_depth_ )
            , dim( dim_ )
            , isLaunch( is_launch_ ){};
    };

    std::vector<LinkedPtr<CommandList, PostprocessingStage>> m_stages;

    std::vector<CommandDescriptor> m_commands;

    bool m_isFinalized = false;

    // True if the command list should be exelwted asynchronously.
    bool m_exelwteAsync = false;

    // The synchronization stream set by the application for this command list.
    LWstream m_syncStream = nullptr;

    // The devices to use when launching. Empty means launch on all active devices.
    std::vector<unsigned int> m_devices;

    ReusableID m_id;


    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_COMMAND_LIST};
};

inline bool CommandList::isA( ManagedObjectType type ) const
{
    return type == m_objectType || ManagedObject::isA( type );
}

}  // namespace optix
