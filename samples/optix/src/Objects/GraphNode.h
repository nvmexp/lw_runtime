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
#include <Objects/Program.h>  // LinkedPtr<,Program> needs to know about Program
#include <optixu/optixu_aabb.h>
#include <optixu/optixu_matrix_namespace.h>

#include <rtcore/interface/types.h>

#include <vector>

namespace optix {
class Acceleration;
class Context;
class MotionAabb;

// Base class for scene graph node types.
class GraphNode : public LexicalScope
{
    //////////////////////////////////////////////////////////////////////////
  public:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // CTOR / DTOR
    //------------------------------------------------------------------------
  protected:
    GraphNode( Context* context, ObjectClass objClass );
    ~GraphNode() override;

  public:
    //------------------------------------------------------------------------
    // Public API
    //-----------------------------------------------------------------------
    // Set/get the visit program. Note that this is also the public API
    // for Selector
    void setVisitProgram( Program* program );
    Program* getVisitProgram() const;

    // throws exceptions in case of invalid
    void validate() const override;

    // Set/get the bounds program, used for computing aabbs for groups.
    void setBoundingBoxProgram( Program* program );
    Program* getBoundingBoxProgram() const;

    //------------------------------------------------------------------------
    // LinkedPtr relationship mangement
    //------------------------------------------------------------------------
    void detachFromParents() override;

  protected:
    void detachLinkedProgram( const LinkedPtr_Link* link );

  public:
    //////////////////////////////////////////////////////////////////////////
  protected:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // Object record management
    //------------------------------------------------------------------------
    void writeRecord() const override;
    void notifyParents_offsetDidChange() const override;


    //------------------------------------------------------------------------
    // Attachment
    //------------------------------------------------------------------------
    // Notify children of a change in attachment
  protected:
    virtual void sendPropertyDidChange_Attachment( bool added ) const;

    //------------------------------------------------------------------------
    // RtxUniversalTraversal property
    //------------------------------------------------------------------------
  public:
    // Returns true iff the rtx requires universal traversal
    // when this graph node is directly attached to a variable.
    virtual bool rtxTraversableNeedUniversalTraversal() const;

    //------------------------------------------------------------------------
    // Direct Caller
    //------------------------------------------------------------------------
    // This isn't pure virtual, because it can be called during
    // GraphNode CTOR and the polymorphic child classes don't exist yet.
    // In this case it is OK to call a shell because there are no
    // children during construction.
  protected:
    virtual void attachOrDetachProperty_DirectCaller_toChildren( Program* program, bool added ) const;

  public:
    void sendPropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const override;

  private:
    // Propagate direct caller to visit program
    void attachOrDetachProperty_DirectCaller( Program* program, bool added ) const;

  protected:
    //------------------------------------------------------------------------
    // Trace Caller
    //------------------------------------------------------------------------
    // Called when a child is attached to a GraphNode
    void attachOrDetachProperty_TraceCaller( LexicalScope* atScope, bool added ) const;

  public:
    void receivePropertyDidChange_TraceCaller( CanonicalProgramID cpid, bool added );
    virtual void sendPropertyDidChange_TraceCaller( CanonicalProgramID cpid, bool added ) const = 0;

  protected:
    //------------------------------------------------------------------------
    // Acceleration height property
    //------------------------------------------------------------------------
    void receivePropertyDidChange_AccelerationHeight( int height, bool added );
    virtual void sendPropertyDidChange_AccelerationHeight( int height, bool added ) const;
    // Returns 1 for group nodes, 0 for others
    virtual int accelerationHeightOffset() const { return 0; }

  public:
    // GeometryGroup has a height of 1. Nodes above GeometryGroup will
    // have a height of 1 or more.
    int          getMaxAccelerationHeight() const;
    virtual void attachOrDetachProperty_AccelerationHeight( GraphNode* child, bool attached ) const;
    virtual void attachOrDetachProperty_AccelerationHeight( Acceleration* child, bool attached ) const;

  protected:
    //------------------------------------------------------------------------
    // Transform height property
    //------------------------------------------------------------------------
  public:
    // Transform height starting at zero. GeometryGroup has a height of
    // zero. Nodes above will have a height of 0 until the first
    // Transform parent is encountered.
    int          getMaxTransformHeight() const;
    virtual void attachOrDetachProperty_TransformHeight( GraphNode* child, bool attached ) const;

  private:
    void receivePropertyDidChange_TransformHeight( int height, bool added );
    void sendPropertyDidChange_TransformHeight( int height, bool added ) const;
    virtual int transformHeightOffset() const { return 0; }

  protected:
    //------------------------------------------------------------------------
    // HasMotionAabbs property
    //------------------------------------------------------------------------
    virtual void sendPropertyDidChange_HasMotionAabbs( bool added ) const;
    virtual bool hasMotionKeys() const { return false; }

  public:
    // Returns true if node has motion steps or the subgraph under it has motion steps
    bool hasMotionAabbs() const;
    void attachOrDetachProperty_HasMotionAabbs( GraphNode* child, bool attached ) const;
    void attachOrDetachProperty_HasMotionAabbs( Acceleration* child, bool attached ) const;
    void receivePropertyDidChange_HasMotionAabbs( bool added );

    //------------------------------------------------------------------------
    // Geometry flags and visibility mask properties
    //------------------------------------------------------------------------
    virtual bool getVisibilityMaskInstanceFlags( RTvisibilitymask&, RTinstanceflags& ) const;
    virtual void sendPropertyDidChange_VisibilityMaskInstanceFlags( GraphNode*      parentNode,
                                                                    LinkedPtr_Link* parentLink,
                                                                    RTvisibilitymask,
                                                                    RTinstanceflags ) const;
    virtual void sendPropertyDidChange_VisibilityMaskInstanceFlags( RTvisibilitymask, RTinstanceflags ) const;
    virtual void attachOrDetachProperty_VisibilityMaskInstanceFlags( GraphNode* parent, LinkedPtr_Link* child );
    virtual void receivePropertyDidChange_VisibilityMaskInstanceFlags( LinkedPtr_Link*, RTvisibilitymask, RTinstanceflags );

    //------------------------------------------------------------------------
    // InstanceTransform property
    //------------------------------------------------------------------------
  public:
    virtual Matrix4x4 getInstanceTransform() const;
    void attachOrDetachProperty_InstanceTransform( GraphNode* parent, LinkedPtr_Link* child );
    virtual void receiveProperty_InstanceTransform( LinkedPtr_Link* child, Matrix4x4 transform );
    virtual void instanceTransformDidChange( LinkedPtr_Link* child, Matrix4x4 transform );

  protected:
    void sendPropertyDidChange_InstanceTransform( Matrix4x4 instanceTransform );

    //------------------------------------------------------------------------
    // SBTIndex property
    //------------------------------------------------------------------------
  public:
    virtual unsigned int getSBTIndex() const;
    void notifyParents_SBTIndexDidChange( unsigned int oldSBTIndex, unsigned int newSBTIndex );
    virtual void childSBTIndexDidChange( LinkedPtr_Link* parentLink, unsigned int oldSBTIndex, unsigned int newSBTIndex );
    void attachOrDetachProperty_SBTIndex( GraphNode* parent, LinkedPtr_Link* child );

    //------------------------------------------------------------------------
    // Traversable properties
    //------------------------------------------------------------------------
  public:
    // Source of a traversable handle changed notification
    enum class TravSource
    {
        GEOMGROUP_ACCEL,   // The Acceleration's handle on a GeometryGroup changed.
        GEOMGROUP_DIRECT,  // The GeometryGroup's direct traversable changed.
        TRANSFORM_DIRECT,  // The Static Transform's direct traversable changed.
        OTHER_DIRECT       // The direct traversable on (selector, group, motion transform) changed.
    };
    virtual TravSource getTraversableSource() const;
    virtual void childTraversableHandleDidChange( LinkedPtr_Link* child, TravSource source, unsigned int allDeviceIndex, RtcTraversableHandle travHandle );
    virtual void attachTraversableHandle( GraphNode* parent, LinkedPtr_Link* child );
    virtual void receivePropertyDidChange_TransformRequiresTraversable( GraphNode* fromParent, bool attached );

  protected:
    void sendDidChange_TraversableHandle( TravSource source, unsigned int allDeviceIndex, RtcTraversableHandle travHandle );

  private:
    void variableChildTraversableHandleDidChange( TravSource source, unsigned int allDeviceIndex, RtcTraversableHandle travHandle );

  public:
    struct TraversableDataForTest
    {
        size_t             m_size;  // size of the data
        RtcTraversableType m_type;  // type of the data
        union
        {
            const RtcTravSelector*            m_selector;
            const RtcTravStaticTransform*     m_staticTransform;
            const RtcMatrixMotionTransform*   m_matrixMotionTransform;
            const RtcSRTMotionTransform*      m_srtMotionTransform;
            const RtcTravBottomLevelInstance* m_bottomLevelInstance;
        };  // pointer to the traversable data

        ReusableIDValue m_traversableId;  // traversable id for variable, or -1 if no variable
    };

    virtual TraversableDataForTest getTraversableDataForTest( unsigned int allDeviceIndex ) const;
    virtual void releaseTraversableDataForTest( unsigned int allDeviceIndex );

    //------------------------------------------------------------------------
    // Attached to variable property
    //------------------------------------------------------------------------
  public:
    void receivePropertyDidChange_AttachedToVariable( bool attached );

    //------------------------------------------------------------------------
    // Traversable support
    //------------------------------------------------------------------------
  public:
    virtual void receivePropertyDidChange_RequiresTraversable( GraphNode* fromParent, bool added );
    virtual RtcTraversableHandle getTraversableHandle( unsigned int allDeviceIndex ) const = 0;
    int getRequiresTraversableCountForTest() const { return m_requiresTraversable.count(); }
    int getTransformRequiresTraversableCountForTest() const { return m_transformRequiresTraversable.count(); }

  protected:
    virtual void updateTraversable() = 0;


    //------------------------------------------------------------------------
    // Helper functions
    //------------------------------------------------------------------------
    // Throws an error if adding newChild as a child will cause a cycle
    // in the graph.
  protected:
    void checkForRelwrsiveGraph( GraphNode* newChild ) const;

    // Returns true if node is an ancestor of this.  Note: this->hasAncestor(this) returns true
    bool hasAncestor( const GraphNode* node ) const;


    //------------------------------------------------------------------------
    // Member data
    //------------------------------------------------------------------------
  private:
    LinkedPtr<GraphNode, Program> m_visitProgram;
    LinkedPtr<GraphNode, Program> m_boundingBoxProgram;

    GraphProperty<CanonicalProgramID> m_traceCaller;

  protected:
    GraphProperty<int, true> m_accelerationHeight;

    GraphProperty<int, true> m_transformHeight;

    GraphPropertySingle<int> m_hasMotionAabbs;

    GraphPropertySingle<int> m_requiresTraversable;
    GraphPropertySingle<int> m_transformRequiresTraversable;

    GraphPropertySingle<int> m_attachedToVariable;

    ReusableID m_traversableId;  // Used only for graphnodes connected directly to a variable

    std::vector<MBufferHandle> m_traversables;

    friend class NodegraphPrinter;  // Allow nodegraph printer to print sets

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_GRAPH_NODE};
};

inline bool GraphNode::isA( ManagedObjectType type ) const
{
    return type == m_objectType || LexicalScope::isA( type );
}

}  // namespace optix
