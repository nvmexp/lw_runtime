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

#include <Memory/MBuffer.h>
#include <Objects/LexicalScope.h>
#include <corelib/system/ExelwtableModule.h>

namespace optix {

class LWDADevice;

class PostprocessingStage : public LexicalScope
{
    //////////////////////////////////////////////////////////////////////////
  public:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // CTOR / DTOR
    //------------------------------------------------------------------------
    PostprocessingStage( Context* context, const std::string& builtin_name );
    ~PostprocessingStage() override;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    // Called once for each stage when finalized to perform one time initialization needed
    // before subsequent calls to launch().
    //
    // Concrete implementations of PostprocessingStage must implement this method
    // to do the work.
    virtual void initialize( RTsize width, RTsize height ) = 0;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    // Launch the stage in 2d. The buffers are specified as variables. Other variables can be
    // used to control the behavior. This method can not be called directly from the application but
    // will be called by Optix when exelwtes a command list.
    //
    // Concrete implementations of PostprocessingStage must implement this method
    // to do the work.
    //
    // NOTE: Input and output buffers are set as variables with fixed names! (TODO: dolwmented where?)
    // Concrete implementations of PostprocessingStage must implement this method
    // to do the work.
    // Concrete implementations of PostprocessingStage must implement doLaunch to contain the actual
    // functionality. This function is a wrapper to provide logging, profiling, etc.
    //
    void launch( RTsize width, RTsize height );

    // Throws an exception if object is invalid
    void validate() const override;

    //------------------------------------------------------------------------
    // LinkedPtr relationship mangement
    //------------------------------------------------------------------------
    void detachFromParents() override;
    void detachLinkedChild( const LinkedPtr_Link* link ) override;

    const std::string getName() const { return m_name; }

    // Called by Buffer when mapped for write. To get this callback stage implementations need
    // to add a Linked_ptr to the buffer and override this method, as well as detachLinkedChild()
    // to do cleanup when the buffer is destroyed.
    virtual void bufferWasMapped( LinkedPtr_Link* link );

    // Called by linked child buffer when anything about its format changes. To get this callback
    // stage implementations need to add a Linked_ptr to the buffer and override this method, as
    // well as detachLinkedChild() to do cleanup when the buffer is destroyed.
    virtual void bufferFormatDidChange( LinkedPtr_Link* link );

  protected:
    virtual void doLaunch( RTsize width, RTsize height ) = 0;
    // A (not necessarily unique) name to identify stages e.g. in profile output
    std::string m_name;

    // Override from LexicalScope. Marks buffers used by post-processing stages so that the buffer
    // policies are updated correctly.
    void bufferVariableValueDidChange( Variable* var, Buffer* oldBuffer, Buffer* newBuffer ) override;

    // Utility function that checks if the provided buffer is in slow zero copy memory and makes a
    // GPU copy in that case. The provided device must be the current lwca device.
    // Returns either the same device pointer or a pointer to a copy on the GPU. If the pointer is
    // different lwdaFree must be called on it when not needed anymore.
    void* copyToGpuIfNeeded( MBufferHandle& buffer, void* devicePtr, LWDADevice* device );

  private:
    //------------------------------------------------------------------------
    // Object record management
    //------------------------------------------------------------------------
    size_t getRecordBaseSize() const override;
    void   reallocateRecord() override;
    void   notifyParents_offsetDidChange() const override;

    //------------------------------------------------------------------------
    // Unresolved reference property
    //------------------------------------------------------------------------
    void sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool addedToUnresolvedSet ) const override;

    //------------------------------------------------------------------------
    // Attachment
    //------------------------------------------------------------------------
    // Notify children of a change in attachment
    void sendPropertyDidChange_Attachment( bool added ) const override;

    //------------------------------------------------------------------------
    // Direct Caller
    //------------------------------------------------------------------------
    void sendPropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const override;

    // Not the same as the LexicalScope::m_id.
    ReusableID m_stageId;

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_POSTPROC_STAGE};
};

inline bool PostprocessingStage::isA( ManagedObjectType type ) const
{
    return type == m_objectType || LexicalScope::isA( type );
}

}  // namespace optix
