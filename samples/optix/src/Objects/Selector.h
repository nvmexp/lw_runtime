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

#include <Objects/GraphNode.h>

#include <Memory/MBufferListener.h>
#include <optixu/optixu_aabb.h>

#include <vector>

namespace optix {
class LWDADevice;
class MotionAabb;

class Selector : public GraphNode, public MBufferListener
{
    //////////////////////////////////////////////////////////////////////////
  public:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // CTOR/DTOR
    //------------------------------------------------------------------------
    Selector( Context* context );
    ~Selector();


    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    // Child management
    void setChildCount( unsigned int count );
    unsigned int getChildCount() const;
    void setChild( unsigned int index, GraphNode* child );
    GraphNode* getChild( unsigned int index ) const;

    // Validation
    virtual void validate() const;


    //------------------------------------------------------------------------
    // Internal API
    //------------------------------------------------------------------------


    // Child management
    ObjectClass getChildType( unsigned int index ) const;

    // Colwenience function to get a child cast to the given type.
    template <typename T>
    T* getChild( unsigned int index ) const;


    //------------------------------------------------------------------------
    // LinkedPtr relationship mangement
    //------------------------------------------------------------------------
    virtual void detachLinkedChild( const LinkedPtr_Link* link );


    //////////////////////////////////////////////////////////////////////////
  private:
    //////////////////////////////////////////////////////////////////////////


    //------------------------------------------------------------------------
    // Object record access and management
    //------------------------------------------------------------------------
    virtual size_t getRecordBaseSize() const;
    virtual void   writeRecord() const;

  public:
    // Fill the data buffers that reference the children
    void fillChildren() const;

  private:
    //------------------------------------------------------------------------
    // Unresolved reference property
    //------------------------------------------------------------------------
    // Override receive to handle attachment
    virtual void receivePropertyDidChange_UnresolvedReference( const LexicalScope* child, VariableReferenceID refid, bool added );

    // References are propagated to GlobalScope
    virtual void sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool addedToUnresolvedSet ) const;

    // Override variable declaration to handle attachment
    virtual void variableDeclarationDidChange( VariableReferenceID refid, bool added );

    // Compute th resolve set with proper attachment
    virtual void computeUnresolvedOutputForDebugging( GraphProperty<VariableReferenceID, false>& out ) const;


    //------------------------------------------------------------------------
    // Attachment
    //------------------------------------------------------------------------
    // Notify children of a change in attachment
    virtual void sendPropertyDidChange_Attachment( bool added ) const;

    // Output set resolution is affected by attachment at the
    // GeometryInstance so that unresolved references do not propagate
    // to the global scope for nodes not attached.  See also GeometryInstance.
    virtual void attachmentDidChange( bool newAttached );

    //------------------------------------------------------------------------
    // RtxUniversalTraversal property
    //------------------------------------------------------------------------

    // Notify children of a change in traversal mode
    virtual void sendPropertyDidChange_RtxUniversalTraversal() const override;

    //------------------------------------------------------------------------
    // Direct Caller
    //------------------------------------------------------------------------
  public:
    void attachOrDetachProperty_DirectCaller_toChildren( Program* program, bool added ) const override;

  private:
    //------------------------------------------------------------------------
    // Trace Caller
    //------------------------------------------------------------------------
    void sendPropertyDidChange_TraceCaller( CanonicalProgramID cpid, bool added ) const override;

    //------------------------------------------------------------------------
    // Traversable support
    //------------------------------------------------------------------------
  public:
    RtcTraversableHandle getTraversableHandle( unsigned int allDeviceIndex ) const override;
    void childTraversableHandleDidChange( LinkedPtr_Link* child, TravSource source, unsigned int allDeviceIndex, RtcTraversableHandle travHandle ) override;

    TraversableDataForTest getTraversableDataForTest( unsigned int allDeviceIndex ) const override;

  protected:
    void updateTraversable() override;
    void eventMBufferMAccessDidChange( const MBuffer* buffer, const Device* device, const MAccess& oldMBA, const MAccess& newMBA ) override;

  private:
    RtcTraversableHandle getTraversableHandleFromMAccess( const LWDADevice* lwdaDevice, const MAccess& access ) const;
    BufferDimensions getTraversableDimensions() const;
    void             allocateTraversables();
    void             resizeTraversables();
    void writeTraversable( unsigned int allDeviceIndex );
    void writeChildTraversable( unsigned int allDeviceIndex, unsigned int childIndex, RtcTraversableHandle childHandle );

    //------------------------------------------------------------------------
    // Member data
    //------------------------------------------------------------------------
  private:
    std::vector<LinkedPtr<Selector, GraphNode>> m_children;
    std::unique_ptr<Buffer> m_childrenBuffer;

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_SELECTOR};
};

inline bool Selector::isA( ManagedObjectType type ) const
{
    return type == m_objectType || GraphNode::isA( type );
}

template <typename T>
inline T* Selector::getChild( unsigned int index ) const
{
    ManagedObject* child = getChild( index );
    RT_ASSERT( managedObjectCast<T>( child ) );
    return static_cast<T*>( child );
}

}  // namespace optix
