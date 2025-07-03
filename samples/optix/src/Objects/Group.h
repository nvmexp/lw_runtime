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

#include <Context/Context.h>
#include <Memory/InstanceDescriptorTable.h>
#include <Memory/MBufferListener.h>
#include <Objects/Acceleration.h>
#include <Objects/GraphNode.h>
#include <Objects/Transform.h>

#include <memory>
#include <vector>

namespace optix {
class ASManager;
class Context;
class GeometryGroup;
class MotionAabb;

//
// AbstractGroup.
//

class AbstractGroup : public GraphNode
{
    //////////////////////////////////////////////////////////////////////////
  public:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // CTOR/DTOR
    //------------------------------------------------------------------------
  protected:
    AbstractGroup( Context* context, ObjectClass objClass );
    ~AbstractGroup();

  public:
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    // Set / get acceleration structure
    virtual void setAcceleration( Acceleration* acceleration );
    Acceleration* getAcceleration() const;

    // Child management
    LexicalScope* getChild( unsigned int index ) const;
    unsigned int getChildCount() const;
    void setChild( unsigned int index, LexicalScope* child );
    virtual void setChildCount( unsigned int count );

    void             setVisibilityMask( RTvisibilitymask );
    RTvisibilitymask getVisibilityMask() const;
    void             setFlags( RTinstanceflags );
    RTinstanceflags  getFlags() const;

    //------------------------------------------------------------------------
    // Internal API
    //------------------------------------------------------------------------

    // Return if the concrete type is GeometryGroup.
    virtual bool isGeometryGroup() const = 0;

    // Child management
    ObjectClass getChildType( unsigned int index ) const;

    // Colwenience function to get a child cast to the given type.
    template <typename T>
    T* getChild( unsigned int index ) const;

    // Fill the children buffer with the object record offsets
    void fillChildren();

    // Get the children buffer (used for the builder)
    MBufferHandle getChildren();


    //------------------------------------------------------------------------
    // LinkedPtr relationship management
    //------------------------------------------------------------------------
  public:
    virtual void detachLinkedChild( const LinkedPtr_Link* link );
    void childOffsetDidChange( const LinkedPtr_Link* link ) override;


    //------------------------------------------------------------------------
    // Object record management
    //------------------------------------------------------------------------
  protected:
    virtual size_t getRecordBaseSize() const;
    virtual void   writeRecord() const;
    virtual void   offsetDidChange() const;

    //------------------------------------------------------------------------
    // AbstractGroup support methods
    //------------------------------------------------------------------------
    virtual void checkForRelwrsiveGroup( GraphNode* child ) = 0;
    virtual void attachOrDetachChild( LexicalScope* child, LinkedPtr_Link* childLink, bool attached ) = 0;
    virtual void updateRtCoreData( unsigned int index ) = 0;
    virtual void reallocateInstances();


    //------------------------------------------------------------------------
    // SBTRecord management
    //------------------------------------------------------------------------
  public:
    void rayTypeCountDidChange();
    void materialCountDidChange( const LinkedPtr_Link* giLink );

    void geometryDidChange( const LinkedPtr_Link* giLink ) const;
    void geometryIntersectionDidChange( const LinkedPtr_Link* giLink ) const;

    void materialDidChange( const LinkedPtr_Link* giLink, unsigned int materialIndex ) const;
    void materialOffsetDidChange( const LinkedPtr_Link* giLink, unsigned int materialIndex ) const;
    void materialClosestHitProgramDidChange( const LinkedPtr_Link* giLink, unsigned int materialIndex, unsigned int rayTypeIndex ) const;
    void materialAnyHitProgramDidChange( const LinkedPtr_Link* giLink, unsigned int materialIndex, unsigned int rayTypeIndex ) const;

    unsigned int getSBTRecordIndex() const;

    struct SBTManagerIndex_fn
    {
        int& operator()( const AbstractGroup* g ) { return g->m_SBTManagerIndex; }
    };
    struct SBTManagerUpdateIndex_fn
    {
        int& operator()( const AbstractGroup* g ) { return g->m_SBTManagerUpdateIndex; }
    };

  private:
    void getGGandIndexFromLink( const GeometryGroup*& gg, unsigned int& index, const LinkedPtr_Link* link ) const;
    virtual void updateChildTraversable( unsigned int childIndex ) = 0;


    //------------------------------------------------------------------------
    // Unresolved reference property
    //------------------------------------------------------------------------
    // Attach to scope parents (Acceleration)
  private:
    virtual void attachOrDetachProperty_UnresolvedReference( Acceleration* acceleration, bool attached ) const;

    // Notify all scope parents (Acceleration) of changes in unresolved
    // references.
    virtual void sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool addedToUnresolvedSet ) const;


    //------------------------------------------------------------------------
    // Attachment
    //------------------------------------------------------------------------
    // Notify children of a change in attachment

  private:
    virtual void sendPropertyDidChange_Attachment( bool added ) const;

    //------------------------------------------------------------------------
    // RtxUniversalTraversal property
    //------------------------------------------------------------------------

    // Notify children of a change in traversal mode
  private:
    void sendPropertyDidChange_RtxUniversalTraversal() const override;

    //------------------------------------------------------------------------
    // Direct Caller
    //------------------------------------------------------------------------
  public:
    void attachOrDetachProperty_DirectCaller_toChildren( Program* program, bool added ) const override;

    //------------------------------------------------------------------------
    // Trace Caller
    //------------------------------------------------------------------------
  protected:
    void sendPropertyDidChange_TraceCaller( CanonicalProgramID cpid, bool added ) const override;

    //------------------------------------------------------------------------
    // Acceleration height property
    //------------------------------------------------------------------------
  protected:
    // Needed to also notify AS objects
    virtual void sendPropertyDidChange_AccelerationHeight( int height, bool added ) const;
    // Returns 1 for group nodes, 0 for others
    virtual int accelerationHeightOffset() const { return 1; }

    //------------------------------------------------------------------------
    // HasMotionAabbs property
    //------------------------------------------------------------------------
  protected:
    // Needed to also notify AS objects
    virtual void sendPropertyDidChange_HasMotionAabbs( bool added ) const;

    //------------------------------------------------------------------------
    // Geometry flags and visibility mask properties
    //------------------------------------------------------------------------
  public:
    virtual bool getVisibilityMaskInstanceFlags( RTvisibilitymask&, RTinstanceflags& ) const;
    void attachOrDetachProperty_VisibilityMaskInstanceFlags( GraphNode* parent, LinkedPtr_Link* child ) override;

    //------------------------------------------------------------------------
    // Traversable support
    //------------------------------------------------------------------------
  public:
    virtual void asTraversableHandleDidChange( RtcTraversableHandle travHandle, unsigned int allDeviceIndex ) = 0;

    //------------------------------------------------------------------------
    // Helper functions
    //------------------------------------------------------------------------
  protected:
    unsigned int getChildIndexFromLink( const LinkedPtr_Link* link ) const;

    //------------------------------------------------------------------------
    // Member data
    //------------------------------------------------------------------------
  protected:
    LinkedPtr<AbstractGroup, Acceleration> m_acceleration;
    std::shared_ptr<size_t> m_SBTIndex;
    size_t                  numSetChildren{0};

    std::vector<LinkedPtr<AbstractGroup, LexicalScope>> m_children;

    RTinstanceflags  m_flags          = RT_INSTANCE_FLAG_NONE;
    RTvisibilitymask m_visibilityMask = RT_VISIBILITY_ALL;

  private:
    std::unique_ptr<Buffer> m_childrenBuffer;

    mutable int m_SBTManagerIndex       = -1;
    mutable int m_SBTManagerUpdateIndex = -1;

    // Used for dirty list in ASManager
    int m_dirtyListIndex = -1;

    friend class ASManager;
    struct dirtyListIndex_fn
    {
        int& operator()( AbstractGroup* g ) { return g->m_dirtyListIndex; }
    };

    // Let NodegraphPrinter see the GraphProperties
    friend class NodegraphPrinter;

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_ABSTRACT_GROUP};
};


//
// Group. This is a minimal class that delegates most responsibilities to AbstractGroup
//

class Group : public AbstractGroup
{
  public:
    Group( Context* context );

    virtual ~Group();

    bool isGeometryGroup() const override { return false; }
    void validate() const override;

    void setChildCount( unsigned int newCount ) override;

    //------------------------------------------------------------------------
    // InstanceDescriptor support (RTX)
    //------------------------------------------------------------------------
  public:
    InstanceDescriptorHost::DeviceIndependent getInstanceDescriptor( unsigned int child ) const;
    void setInstanceDescriptor( unsigned int child, const InstanceDescriptorHost::DeviceIndependent& descriptor );

    InstanceDescriptorHost::DeviceDependent getInstanceDescriptorDeviceDependent( unsigned int child,
                                                                                  unsigned int allDevicesindex ) const;

    void setInstanceDescriptorDeviceDependent( unsigned int                                   child,
                                               unsigned int                                   allDevicesIndex,
                                               const InstanceDescriptorHost::DeviceDependent& descriptor );

    RtcInstance* getInstanceDescriptorDevicePtr( Device* device ) const;
    const char* getInstanceDescriptorTableDevicePtr( unsigned int allDevicesIndex ) const;
    void syncInstanceDescriptors();

    //------------------------------------------------------------------------
    // InstanceTransform property
    //------------------------------------------------------------------------
  public:
    void receiveProperty_InstanceTransform( LinkedPtr_Link* child, Matrix4x4 transform ) override;
    void instanceTransformDidChange( LinkedPtr_Link* parentLink, Matrix4x4 transform ) override;
    void receivePropertyDidChange_VisibilityMaskInstanceFlags( LinkedPtr_Link*, RTvisibilitymask, RTinstanceflags ) override;

    //------------------------------------------------------------------------
    // Traversable support
    //------------------------------------------------------------------------
  public:
    RtcTraversableHandle getTraversableHandle( unsigned int allDeviceIndex ) const override;
    void asTraversableHandleDidChange( RtcTraversableHandle travHandle, unsigned int allDeviceIndex ) override;
    void childTraversableHandleDidChange( LinkedPtr_Link* child, TravSource source, unsigned int allDeviceIndex, RtcTraversableHandle travHandle ) override;

    TraversableDataForTest getTraversableDataForTest( unsigned int allDeviceIndex ) const override;

    void childSBTIndexDidChange( LinkedPtr_Link* parentLink, unsigned int oldSBTIndex, unsigned int newSBTIndex ) override;

  protected:
    void updateChildTraversable( unsigned int childIndex ) override;
    void updateTraversable() override;

    void checkForRelwrsiveGroup( GraphNode* child ) override;
    void attachOrDetachChild( LexicalScope* child, LinkedPtr_Link* childLink, bool attached ) override;
    void updateRtCoreData( unsigned index ) override;

    mutable InstanceDescriptorTable m_instanceDescriptors;

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_GROUP};
};


//
// GeometryGroup. This is a minimal class that delegates most responsibilities to AbstractGroup
//

class GeometryGroup : public AbstractGroup, public MBufferListener
{
  public:
    GeometryGroup( Context* context );
    virtual ~GeometryGroup();


    bool isGeometryGroup() const override { return true; }
    void validate() const override;

    //------------------------------------------------------------------------
    // SBTIndex
    //------------------------------------------------------------------------
    unsigned int getSBTIndex() const override;

    //------------------------------------------------------------------------
    // Traversable support
    //------------------------------------------------------------------------
  public:
    TravSource           getTraversableSource() const override;
    RtcTraversableHandle getTraversableHandle( unsigned int allDeviceIndex ) const override;
    void asTraversableHandleDidChange( RtcTraversableHandle travHandle, unsigned int allDeviceIndex ) override;
    void writeTopLevelTraversable( const cort::Aabb* aabbDevicePtr, LWDADevice* lwdaDevice );
    void attachTraversableHandle( GraphNode* parent, LinkedPtr_Link* child ) override;

    TraversableDataForTest getTraversableDataForTest( unsigned int allDeviceIndex ) const override;

    void preSetActiveDevices( const DeviceSet& removedDeviceSet );
    void postSetActiveDevices();


    //------------------------------------------------------------------------
    // RtxUniversalTraversal property
    //------------------------------------------------------------------------
  private:
    virtual void rtxUniversalTraversalDidChange() override;

  protected:
    void updateTraversable() override;
    void eventMBufferMAccessDidChange( const MBuffer* buffer, const Device* device, const MAccess& oldMBA, const MAccess& newMBA ) override;
    void writeTraversable( RtcTraversableHandle travHandle, unsigned int allDeviceIndex );
    void updateChildTraversable( unsigned int childIndex ) override;

    void checkForRelwrsiveGroup( GraphNode* child ) override;
    void attachOrDetachChild( LexicalScope* oldChild, LinkedPtr_Link* childLink, bool attachNotDetach ) override;
    void updateRtCoreData( unsigned index ) override;
    void reallocateInstances() override;

  private:
    // These three are indexed by device->allDeviceListIndex.
    std::vector<MBufferHandle>        m_topLevelTempBuffers;
    std::vector<MBufferHandle>        m_topLevelInstances;
    std::vector<RtcTraversableHandle> m_asTraversables;

    MBufferHandle createTopLevelInstanceBuffer( Device* device );

  public:
    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_GEOMETRY_GROUP};
};

//
// AbstractGroup implementation.
//

template <typename T>
inline T* AbstractGroup::getChild( unsigned int index ) const
{
    LexicalScope* child = getChild( index );

    // Note: this is an unsafe cast because of a performance issue a long time ago
    return static_cast<T*>( child );
}

inline unsigned int AbstractGroup::getChildCount() const
{
    return static_cast<unsigned int>( m_children.size() );
}

inline RTvisibilitymask AbstractGroup::getVisibilityMask() const
{
    return m_visibilityMask;
}

inline RTinstanceflags AbstractGroup::getFlags() const
{
    return m_flags;
}

inline MBufferHandle AbstractGroup::getChildren()
{
    return m_childrenBuffer->getMBuffer();
}

inline unsigned int GeometryGroup::getSBTIndex() const
{
    return !m_context->useRtxDataModel() || m_children.empty() || !m_SBTIndex ? 0 : *m_SBTIndex;
}

//------------------------------------------------------------------------
// RTTI
//------------------------------------------------------------------------

inline bool AbstractGroup::isA( ManagedObjectType type ) const
{
    return type == m_objectType || GraphNode::isA( type );
}

inline bool GeometryGroup::isA( ManagedObjectType type ) const
{
    return type == m_objectType || AbstractGroup::isA( type );
}

inline bool Group::isA( ManagedObjectType type ) const
{
    return type == m_objectType || AbstractGroup::isA( type );
}

}  // end namespace optix
