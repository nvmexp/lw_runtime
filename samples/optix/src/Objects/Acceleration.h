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

#include <Objects/LexicalScope.h>

#include <AS/Builder.h>
#include <Util/PropertySet.h>

#include <optixu/optixu_aabb.h>

namespace optix {

class ASManager;
class AbstractGroup;
struct BvhNode;

class Acceleration : public LexicalScope
{
    //////////////////////////////////////////////////////////////////////////
  public:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // CTOR/DTOR
    //------------------------------------------------------------------------
    Acceleration( Context* context );
    ~Acceleration() override;

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    // Set/get builder instance.
    void setBuilder( const std::string& builder );
    std::string getBuilderName() const;
    std::string getBuilderType() const { return m_builderType; }

    // get traverser string.
    std::string getTraverser() const;

    // Set/get an acceleration-specific property.
    void setProperty( const std::string& name, const std::string& value );
    std::string getProperty( const std::string& name ) const;

    // Build data getting/setting for AS caching
    size_t getDataSize() const;
    void getData( void* data ) const;
    void setData( const void* data, size_t size );

    // Dirty flag control.
    void markDirty( bool internallyDirty = false );
    void clearDirty();
    bool isDirty() const;
    bool isDirtyExternal() const;  // Is it visibly dirty from the API?

    void markTopLevelTraversableDirty();
    void clearTopLevelTraversableDirty();

    // Throw exceptions if the object is not valid
    void validate() const override;

    // Build the top-level traversables for this accel
    void writeTopLevelTraversable();

    //------------------------------------------------------------------------
    // Internal API
    //------------------------------------------------------------------------

    // Called by Group or GeoemtryGroup to indicate which type of leaf
    // is required. Must be RT_OBJECT_GEOMETRY_GROUP or
    // RT_OBJECT_GROUP.
    void setLeafKind( ObjectClass kind );

    // return true if it is attached to GeometryGroups (as opposed to Groups)
    bool isAttachedToGeometryGroups() const;

    // Tools for iterating over the groups this Acceleration is attached to.
    // Note that we assume that all linked pointers to this object are (abstract) groups.
    typedef LinkedPointerType::const_iterator AbstractGroupIterator;
    AbstractGroupIterator                     abstractGroupBegin() const;  // iterator for processing groups
    AbstractGroupIterator                     abstractGroupEnd() const;    //     "            "        "
    AbstractGroup* getAbstractGroup( AbstractGroupIterator it ) const;     // get a group given the iterator
    int            getAbstractGroupCount() const;  // get number of groups we're attached to (Note: O(N))
    AbstractGroup* getAnyAbstractGroup() const;    // get any of the groups we're attached to

    // Used by testing to replace the bvh. Works only with Bvh.
    void overrideBvh_forTesting( const std::vector<BvhNode>& nodes, const std::vector<int>& indices );

    // Total primitive count of attached groups
    unsigned int getTotalPrimitiveCount() const;

    //------------------------------------------------------------------------
    // Group interface
    //------------------------------------------------------------------------

    // Change the baking state if a new AS is attached.
    void updateBakingState();

    // Return the visit/bounding box programs that the attached groups
    // should use.
    Program* getLwrrentVisitProgram() const;
    Program* getLwrrentBoundingBoxProgram() const;

    // If one of the group's children gets a new object record address and baking is
    // enabled, the AS will need to be rebuilt.
    void groupsChildOffsetDidChange();


    //------------------------------------------------------------------------
    // Traversable support
    //------------------------------------------------------------------------
  public:
    RtcTraversableHandle getTraversableHandle( unsigned int allDeviceIndex ) const;


    // The set of references remaining after variable resolution. Launching with any
    // unresolved references is not legal.
    const GraphProperty<VariableReferenceID, false>& getRemainingUnresolvedReferences() const;

    //------------------------------------------------------------------------
    // AS Manager interface
    //------------------------------------------------------------------------
  public:
    // The maximum tree height at which this acceleration is
    // attached. GeometryGroup is level 0. Groups with only
    // GeometryGroup children will return 1.
    int getMaxAccelerationHeight() const;


    // This is the primary protocol with ASManager for builds.
    // ASManager first ilwokes setupForBuild(), where the builder must
    // create any buffers required to support the build. The builder
    // creates a BuildSetupRequest that will be used to control
    // temporary space required and whether AABBs are needed by the
    // builder. Once allocations have been satisfied, the ASManager
    // will call build(). During build() no other memory allocations
    // are possible. After memory allocations are available again, the
    // ASManager will call finalizeAfterBuild(). Then, the ASManager will
    // call writeTopLevelTraversable to ensure each acceleration structure
    // will be accessible if the BVH has a single level. Finally, ASManager
    // will call updateTimeRangeForMotionGroup() on group-level
    // accelerations once all builds are complete and the object
    // records are again available for writing.
    void setupForBuild( DeviceSet buildDevices );
    const BuildSetupRequest& getBuildSetupRequest() const;
    void build( const BuildSetup& setup );
    void finalizeAfterBuild( DeviceSet buildDevices );
    void updateTimeRangeForMotionGroup();

    // If the builder changes the traversable handle, it must ilwoke
    // this method to propagate the change appropriately.
    void traversableHandleDidChange( RtcTraversableHandle newHandle, unsigned int allDeviceIndex );


    //------------------------------------------------------------------------
    // LinkedPtr relationship management
    //------------------------------------------------------------------------
  public:
    void detachFromParents() override;
    void detachLinkedChild( const LinkedPtr_Link* link ) override;


    //////////////////////////////////////////////////////////////////////////
  private:
    //////////////////////////////////////////////////////////////////////////

    //------------------------------------------------------------------------
    // Object record management
    //------------------------------------------------------------------------
    size_t getRecordBaseSize() const override;
    void   writeRecord() const override;
    void   notifyParents_offsetDidChange() const override;


    //------------------------------------------------------------------------
    // Unresolved reference property
    //------------------------------------------------------------------------
    // Notify all the scope parent (GlobalScope) of changes* in
    // unresolved references.
    void sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool addedToUnresolvedSet ) const override;


    //------------------------------------------------------------------------
    // Attachment
    //------------------------------------------------------------------------
    // Notify children of a change in entrypoint reachability. Note: AS
    // has no children so this is a no-op.
    void sendPropertyDidChange_Attachment( bool added ) const override;

    void attachmentDidChange( bool newAttached ) override;

    //------------------------------------------------------------------------
    // RtxUniversalTraversal
    //------------------------------------------------------------------------

    void rtxUniversalTraversalDidChange();

    //------------------------------------------------------------------------
    // Direct Caller
    //------------------------------------------------------------------------
    void sendPropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const override;


    //------------------------------------------------------------------------
    // Acceleration height property
    //------------------------------------------------------------------------
  public:
    void receivePropertyDidChange_AccelerationHeight( int height, bool added );

    //------------------------------------------------------------------------
    // HasMotionAabbs property
    //------------------------------------------------------------------------
  public:
    void receivePropertyDidChange_HasMotionAabbs( bool added );


    //------------------------------------------------------------------------
    // Traversable support
    //------------------------------------------------------------------------
  private:
    void sendTraversableHandleToParents();

  public:
    // This will be called by the builder when the allocation changes.  When the AS
    // changes the builder, the old builder is destroyed along with any allocation it
    // held.  When the new builder creates its allocations, it will call this function on
    // the AS.
    void buildersTraversableHandleDidChange( RtcTraversableHandle travHandle, int allDeviceIndex );

  private:
    //------------------------------------------------------------------------
    // Helper functions
    //------------------------------------------------------------------------
    // Builder factory
    std::unique_ptr<Builder> createBuilder( const std::string& name );

    // Setup a new builder when either the builder is changed from the
    // API or when the leafKind becomes known.
    void updateBuilder();

    // Get a new visit program from the builder and if necessary
    // communicate it to the attached groups.
    void updateVisitProgram();

    // Add or remove this AS from the ASManager dirty list
    void updateDirtyList();

    // Serialize the acceleration.
    void serialize( Serializer& serializer ) const;

    // Create data vector with data from each GeometryInstance that is
    // communicated to the builder when setting up a GeometryGroup
    // build.
    void setupGeometryInstanceData();

    // Will build need to account for motion aabbs
    bool hasMotionAabbs() const;

  public:
    bool hasMotionAabbs_publicMethodForTesting() const;

  private:
    // Implement setup/build from ASManager
    bool setupForBuildFromTriangles();
    void setupForBuildFromAABBs();
    void setupForBuildGroup();
    void buildGeometryGroup( const BuildSetup& setup );
    void buildGroup( const BuildSetup& setup );
    void computeAABBs( const BuildParameters& params, const BuildSetup& setup );
    void resampleAABBs( const BuildSetup& setup );
    size_t computeResampleTempSize( float buildTimeBegin, float buildTimeEnd ) const;
    size_t      computeMotionAABBRequestSize() const;
    static void gatherMotionAabbs( GraphNode* node,
                                   std::map<int, MotionAabb>& all_maabbs,
                                   const std::map<int, int>&  gatherList,
                                   const std::vector<Aabb>& gatheredAabbs );
    void printAabbs( unsigned int totalPrims, unsigned int buildMotionSteps, const std::vector<Aabb>& aabbs ) const;

    //------------------------------------------------------------------------
    // Member data
    //------------------------------------------------------------------------
  private:
    // Dirty state
    enum DirtyState
    {
        NON_DIRTY,
        DIRTY_INTERNAL,  // Not visibly dirty from the API, but needs rebuild
        DIRTY            // Visibly dirty from the API and internally
    };

    DirtyState m_dirty;

    // Builder and associated properties
    std::string              m_builderType;
    std::string              m_builderTypeParam;
    std::unique_ptr<Builder> m_builder;
    std::string              m_vertexBufferName   = "vertex_buffer";   // name of geometry's vertex buffer variables
    std::string              m_indexBufferName    = "index_buffer";    // name of geometry's index buffer variables
    int                      m_vertexBufferStride = sizeof( float3 );  // offset (in bytes) between vertices
    int                      m_indexBufferStride  = sizeof( int3 );    // offset (in bytes) between indices

    static const unsigned int ILWALID_BUILDER_VERSION = 0xFFFFFFFF;

    static const int MAX_MOTION_STEPS_PER_TRIANGLE = 255;

    ObjectClass m_leafKind           = RT_OBJECT_UNKNOWN;
    Program*    m_visitProgram       = nullptr;
    Program*    m_boundingBoxProgram = nullptr;

    // String/value pairs of user-visible properties
    PropertySet m_properties;

    // Maximum height of attached acceleration structures
    GraphProperty<int, true> m_accelerationHeight;

    // Will acceleration have motion aabbs
    GraphPropertySingle<int> m_hasMotionAabbs;

    // Number of geometry instances
    GraphProperty<int, true> m_childCount;

    // The set of variables that are unresolved at the output from this
    // scope. Marked as mutable so that sendPropertyDidChange can
    // access.
    mutable GraphProperty<VariableReferenceID, false> m_unresolvedRemaining;

    // Build temporarys, held between setupForBuild() and
    // build(). Note that when TriangleData is used, it is *in
    // addition* to GeometryInstanceData.
    BuildParameters                   m_buildParameters;
    BuildSetupRequest                 m_buildSetupRequest;
    std::vector<GeometryInstanceData> m_geometryInstanceData;
    std::vector<TriangleData>         m_triangleData;
    std::unique_ptr<GroupData>        m_groupData;

    // If more than one AS is attached, we may need to trigger a
    // rebuild to unbake entities. Default is baked. Baking is also
    // disabled for refit.
    bool m_bakePrimitiveEntities = true;

    // Triangle builds are not used for some builders (can also be
    // disabled via a knob)
    bool m_enableTriangleBuilds = true;

    // Refine and refine builder parameters. "refit number" is a complicated beast. Values:
    // 0: Refit off
    // 1: Refit every frame
    // 2+: Refit, except every nth frame
    int          m_refitNumber             = 0;
    int          m_framesSinceLastBuild    = 0;
    int          m_lastBuildPrimitiveCount = 0;
    unsigned int m_lastBuildMotionSteps    = 0;

    // Let NodegraphPrinter see the GraphProperties
    friend class LayoutPrinter;
    friend class NodegraphPrinter;

    // Used for dirty list in ASManager
    friend class ASManager;
    int m_dirtyListIndex                    = -1;
    int m_dirtyTopLevelTraversableListIndex = -1;

    // Information so writeTopLevelTraversable can be called after build
    struct TopLevelTraversableInput
    {
        AbstractGroup* group;
        cort::Aabb*    deviceAabbPtr;
        LWDADevice*    buildDevice;
    };
    std::vector<TopLevelTraversableInput> m_topLevelTraversableInputs;

    // The number of motionSteps requested through the AS
    // property. Actual number of motion steps will depend on whether
    // motion AABBs are available.
    int m_requestedMotionSteps = -1;

    struct dirtyListIndex_fn
    {
        int& operator()( Acceleration* accel ) { return accel->m_dirtyListIndex; }
    };

    struct dirtyTopLevelTraversableListIndex_fn
    {
        int& operator()( Acceleration* accel ) { return accel->m_dirtyTopLevelTraversableListIndex; }
    };

    //------------------------------------------------------------------------
    // RTTI
    //------------------------------------------------------------------------
  public:
    bool isA( ManagedObjectType type ) const override;

    static const ManagedObjectType m_objectType{MO_TYPE_ACCELERATION};
};

inline bool Acceleration::isA( ManagedObjectType type ) const
{
    return type == m_objectType || LexicalScope::isA( type );
}

}  // namespace optix
