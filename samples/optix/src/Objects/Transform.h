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

#include <Memory/MBufferListener.h>
#include <Objects/GraphNode.h>

#include <optixu/optixu_matrix.h>
#include <optixu/optixu_quaternion.h>

namespace optix {

class LWDADevice;
class Program;

class Transform : public GraphNode, public MBufferListener
{
  public:
    //////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // CTOR/DTOR
    //------------------------------------------------------------------------
    Transform( Context* context );
    ~Transform() override;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    // Child management
    void setChild( GraphNode* child );
    GraphNode* getChild() const;

    // The transform matrix
    void setMatrix( const float* matrix, bool transpose );
    void setIlwerseMatrix( const float* ilwerse_matrix, bool transpose );
    void getMatrix( float* matrix, bool transpose ) const;
    void getIlwerseMatrix( float* ilwerse_matrix, bool transpose ) const;

    // Motion blur data
    void setMotionRange( float timeBegin, float timeEnd );
    void getMotionRange( float& timeBegin, float& timeEnd ) const;

    void setMotionBorderMode( RTmotionbordermode beginMode, RTmotionbordermode endMode );
    void getMotionBorderMode( RTmotionbordermode& beginMode, RTmotionbordermode& endMode );

    void setKeys( int keyCount, RTmotionkeytype keyType, const float* keys );
    unsigned        getKeyCount() const;
    RTmotionkeytype getKeyType() const;
    void getKeys( float* keys ) const;

    // Validation
    void validate() const override;

    //------------------------------------------------------------------------
    // Internal API
    //------------------------------------------------------------------------


    // Child management
    ObjectClass getChildType() const;
    bool        hasChild() const;

    // Returns true if the node has motion keys
    bool hasMotionKeys() const override { return !m_keys.empty(); }
    // Motion
    MotionAabb computeMotionAabb( const MotionAabb& childAabb ) const;

    static unsigned getKeySize( RTmotionkeytype keyType );

    //------------------------------------------------------------------------
    // LinkedPtr relationship management
    //------------------------------------------------------------------------
  public:
    void detachLinkedChild( const LinkedPtr_Link* link ) override;

    //////////////////////////////////////////////////////////////////////////
  private:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // Object record management
    //------------------------------------------------------------------------
    size_t getRecordBaseSize() const override;
    void   writeRecord() const override;
    void   updateMotionData() const;


    //------------------------------------------------------------------------
    // Unresolved reference property
    //------------------------------------------------------------------------
    // Notify all the scope parent (GlobalScope) of changes* in
    // unresolved references.
    void sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool addedToUnresolvedSet ) const override;


    //------------------------------------------------------------------------
    // Attachment
    //------------------------------------------------------------------------
    // Notify children of a change in attachment
    void sendPropertyDidChange_Attachment( bool added ) const override;

    // Global HasMotionTransforms property is affected by attachment of Transform to graph
    void attachmentDidChange( bool newAttached ) override;

    //------------------------------------------------------------------------
    // RtxUniversalTraversal property
    //------------------------------------------------------------------------

    // Notify children of a change in traversal mode
    virtual void sendPropertyDidChange_RtxUniversalTraversal() const override;

    // Returns true iff the rtx requires universal traversal
    // when this graph node is directly attached to a variable.
    virtual bool rtxTraversableNeedUniversalTraversal() const override;

    //------------------------------------------------------------------------
    // Geometry flags and visibility mask properties
    //------------------------------------------------------------------------
  public:
    virtual bool getVisibilityMaskInstanceFlags( RTvisibilitymask&, RTinstanceflags& ) const;
    void attachOrDetachProperty_VisibilityMaskInstanceFlags( GraphNode* parent, LinkedPtr_Link* child ) override;

    //------------------------------------------------------------------------
    // Direct Caller
    //------------------------------------------------------------------------
  public:
    void attachOrDetachProperty_DirectCaller_toChildren( Program* program, bool added ) const override;


    //------------------------------------------------------------------------
    // InstanceTransform property
    //------------------------------------------------------------------------
    void receiveProperty_InstanceTransform( LinkedPtr_Link* child, Matrix4x4 childTransform ) override;
    void instanceTransformDidChange( LinkedPtr_Link* parent, Matrix4x4 transform ) override;

  protected:
    Matrix4x4 getInstanceTransform() const override;


    //------------------------------------------------------------------------
    // SBTIndex property
    //------------------------------------------------------------------------
  public:
    unsigned int getSBTIndex() const override;
    void childSBTIndexDidChange( LinkedPtr_Link* parentLink, unsigned int oldSBTIndex, unsigned int newSBTIndex ) override;

    //------------------------------------------------------------------------
    // Traversable support
    //------------------------------------------------------------------------
  public:
    TravSource getTraversableSource() const override;
    void receivePropertyDidChange_RequiresTraversable( GraphNode* fromParent, bool attached ) override;
    void receivePropertyDidChange_TransformRequiresTraversable( GraphNode* transform, bool attached ) override;
    RtcTraversableHandle getTraversableHandle( unsigned int allDeviceIndex ) const override;
    void childTraversableHandleDidChange( LinkedPtr_Link* child, TravSource source, unsigned int allDeviceIndex, RtcTraversableHandle childHandle ) override;
    bool requiresDirectTraversable() const;

    TraversableDataForTest getTraversableDataForTest( unsigned int allDeviceIndex ) const override;

  protected:
    void             updateTraversable() override;
    BufferDimensions getTraversableDimensions() const;
    void eventMBufferMAccessDidChange( const MBuffer* buffer, const Device* device, const MAccess& oldMBA, const MAccess& newMBA ) override;

  private:
    void attachOrDetachProperty_RequiresTraversable( GraphNode* oldChild, bool added );
    void attachOrDetachProperty_TransformRequiresTraversable( GraphNode* oldChild, bool added );
    void forwardDidChange_TraversableHandle( TravSource source, unsigned int allDeviceIndex, RtcTraversableHandle travHandle );
    RtcTraversableHandle getTraversableHandleForDevice( const LWDADevice* lwdaDevice ) const;
    RtcTraversableHandle getTraversableHandleFromMAccess( const LWDADevice* lwdaDevice, const MAccess& access ) const;
    void writeTraversable( unsigned int allDeviceIndex );
    void writeTraversable( unsigned int allDeviceIndex, RtcTraversableHandle childHandle );
    void allocateTraversables();
    void resizeTraversables();

    //------------------------------------------------------------------------
    // Trace Caller
    //------------------------------------------------------------------------
  protected:
    void sendPropertyDidChange_TraceCaller( CanonicalProgramID cpid, bool added ) const override;

    //------------------------------------------------------------------------
    // Transform height property
    //------------------------------------------------------------------------
  protected:
    // Returns 1 for transform nodes, 0 for others
    int transformHeightOffset() const override { return 1; }


    //------------------------------------------------------------------------
    // Helper functions
    //------------------------------------------------------------------------
    // Return the first non-transform child down the graph. Used for
    // building a tighter bounding box than would be possible by
    // transforming each AABB individually. If joinedMatrix is non-null,
    // return the compound transform.
    GraphNode* getNonTransformChild( Matrix4x4* joinedMatrix ) const;

    void handleIrregularKeys( const MotionAabb& inputMaabb, unsigned int keycount, bool addIrregularTimes, MotionAabb& outMaabb ) const;

    //------------------------------------------------------------------------
    // Member data
    //------------------------------------------------------------------------
  private:
    Matrix<4, 4>                    m_transform;
    Matrix<4, 4>                    m_ilwerse_transform;
    LinkedPtr<Transform, GraphNode> m_child;

    // MBlur data
    mutable std::unique_ptr<Buffer> m_motionData;  // nullptr if not using motion transform
    std::vector<float>              m_keys;
    RTmotionkeytype                 m_keyType         = RT_MOTIONKEYTYPE_NONE;
    RTmotionbordermode              m_beginBorderMode = RT_MOTIONBORDERMODE_CLAMP;
    RTmotionbordermode              m_endBorderMode   = RT_MOTIONBORDERMODE_CLAMP;
    float                           m_timeBegin       = 0.0f;
    float                           m_timeEnd         = 1.0f;

    enum class MotionDataOffset
    {
        KEY_TYPE          = 0,
        BEGIN_BORDER_MODE = 1,
        END_BORDER_MODE   = 2,
        TIME_BEGIN        = 3,
        TIME_END          = 4,
        NUM_KEYS          = 5,
        KEYS              = 6
    };

    // The set of variables that are unresolved at the output from this
    // scope. Marked as mutable so that sendPropertyDidChange can
    // access.
    mutable GraphProperty<VariableReferenceID, false> m_unresolvedRemaining;

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_TRANSFORM};
};

inline bool Transform::isA( ManagedObjectType type ) const
{
    return type == m_objectType || GraphNode::isA( type );
}

}  // namespace optix
