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

#include <Objects/Acceleration.h>

#include <g_lwconfig.h>

#include <AS/ASManager.h>
#include <AS/BuildTypes.h>
#include <AS/Builder.h>
#include <AS/Bvh/RtcBvh.h>
#include <AS/ResampleMotionAabb.h>
#include <LWCA/Memory.h>
#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Context/ProgramManager.h>
#include <Device/LWDADevice.h>
#include <Exceptions/VariableNotFound.h>
#include <Exceptions/VersionMismatch.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <ExelwtionStrategy/WaitHandle.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <Objects/GeometryInstance.h>
#include <Objects/GeometryTriangles.h>
#include <Objects/GlobalScope.h>
#include <Objects/Selector.h>
#include <Objects/Transform.h>
#include <Util/LinkedPtrHelpers.h>
#include <Util/Metrics.h>
#include <Util/MotionAabb.h>
#include <Util/Serializer.h>

#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/ValidationError.h>
#include <prodlib/system/Knobs.h>

#include <functional>
#include <queue>

// clang-format off
namespace {
  Knob<bool> k_printAabbs( RT_DSTRING( "acceleration.printAabbs" ), false, RT_DSTRING( "Print the AABBs if they were requested. Not all ASes will request them (e.g. Trbvh for triangles)." ) );
  Knob<std::string> k_builderOverride( RT_DSTRING( "acceleration.builderOverride" ), "", RT_DSTRING( "Override any builder (except NoAccel) to the given builder" ) );
  Knob<bool> k_bakeChildPointers( RT_DSTRING( "acceleration.bakeChildPointers" ), true, RT_DSTRING( "Turn child pointer baking on/off" ) );
  Knob<bool> k_enableTriangleBuilds( RT_DSTRING( "acceleration.enableTriangleBuilds" ), true, RT_DSTRING( "When disabled only AABB builds will be performed" ) );
  HiddenPublicKnob<std::string> k_traversalOverride( RT_PUBLIC_DSTRING( "rtx.traversalOverride" ), "", RT_PUBLIC_DSTRING( "Override traversal to the given traversal" ) );
  Knob<bool> k_enableMKTriangleBuilds( RT_DSTRING( "acceleration.enableMKTriangleBuilds" ), false, RT_DSTRING( "When enabled try to read Acceleration properties to query vertex and index buffers for triangle builds" ) );
}
// clang-format on

using namespace optix;
using namespace prodlib;
using namespace corelib;
static Variable* lookupBufferVariable( const GeometryInstance* ginst, unsigned short token );

//------------------------------------------------------------------------
// CTOR/DTOR
//------------------------------------------------------------------------
Acceleration::Acceleration( Context* context )
    : LexicalScope( context, RT_OBJECT_ACCELERATION )
    , m_dirty( DIRTY )
{
    reallocateRecord();
}

Acceleration::~Acceleration()
{
    RT_ASSERT_MSG( m_accelerationHeight.empty(), "Accceleration object destroyed while height property remains" );
    clearDirty();
    clearTopLevelTraversableDirty();
    deleteVariables();
}

//------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------

void Acceleration::setBuilder( const std::string& builderTypeParam )
{
    m_builderTypeParam = builderTypeParam;

    std::string builderType = builderTypeParam;
    if( !k_builderOverride.isDefault() && m_builderType != "NoAccel" )
    {
        builderType = k_builderOverride.get();
    }

    if( m_context->useRtxDataModel() )
    {
// Here we simply need to make sure we provide reasonable defaults, but allow most combinations.
#if LWCFG( GLOBAL_ARCH_TURING ) || LWCFG( GLOBAL_ARCH_AMPERE )
        bool        useTTU     = m_context->getDeviceManager()->activeDevicesSupportTTU();
        std::string defaultBvh = useTTU ? "TTU" : "Trbvh";
        if( builderType == "TTU" && !useTTU )
            throw IlwalidValue( RT_EXCEPTION_INFO, "Specified TTU builder, but TTU support isn't present" );
#else
        bool        useTTU     = false;
        std::string defaultBvh = "Trbvh";
#endif

        if( m_context->RtxUniversalTraversalEnabled() && k_traversalOverride.isDefault() )
        {
            // We can use MTTU for universal traversal on Turing (non-motion) and Ampere (all).
            // MTTU only supports TTU traversal so force TTU AS builds.
            if( useTTU ) // TTU traversal
                builderType = defaultBvh;
            else if( builderType != "NoAccel" )  // Bvh2 traversal
                builderType = defaultBvh;
        }
        else if( k_traversalOverride.isDefault() )
        {
            if( useTTU )  // TTU traversal
                builderType = defaultBvh;
            else if( builderType != "NoAccel" )  // Bvh2 traversal
                builderType = defaultBvh;
        }
        else if( k_traversalOverride.get() == "Utrav" )
        {
            // Allow the builder override to do whatever, but use TTU where we can
            if( k_builderOverride.isDefault() && builderType != "NoAccel" )
                builderType = defaultBvh;
        }
        else if( k_traversalOverride.get() == "Bvh2" )
        {
            if( builderType != "NoAccel" )
                builderType = "Trbvh";
        }
        else if( k_traversalOverride.get() == "Bvh8" )
        {
            builderType = "Bvh8";
        }
#if LWCFG( GLOBAL_ARCH_TURING ) || LWCFG( GLOBAL_ARCH_AMPERE )
        else if( k_traversalOverride.get() == "TTU" )
        {
            if( !useTTU )
                throw IlwalidValue( RT_EXCEPTION_INFO, "TTU traversal override specified, but not supported" );
            builderType = "TTU";
        }
#endif
#if LWCFG( GLOBAL_ARCH_AMPERE )
        else if( k_traversalOverride.get() == "MTTU" )
        {
            if( !useTTU )
                throw IlwalidValue( RT_EXCEPTION_INFO, "MTTU traversal override specified, but not supported" );
            builderType = "TTU";
        }
#endif
        else
        {
            throw IlwalidValue( RT_EXCEPTION_INFO, "Unsupported traversal override : " + k_traversalOverride.get() );
        }
    }  // end useRtxDataModel

#if defined( OPTIX_ENABLE_LOGGING )
    if( builderTypeParam != builderType )
    {
        llog( 100 ) << "Overriding builder type from " << builderTypeParam << " to " << builderType << std::endl;
    }
#endif

    if( builderType == m_builderType )
        return;

    m_builderType = builderType;
    updateBuilder();
}


std::string Acceleration::getBuilderName() const
{
    if( !m_builder )
        return "";
    return m_builder->getName();
}

std::string Acceleration::getTraverser() const
{
    if( getBuilderName() == "NoAccel" )
        return "NoAccel";

    return "Bvh";
}

void Acceleration::setProperty( const std::string& name, const std::string& value )
{
    m_properties.set( name, value );

    // Look for dolwmented properties
    if( name == "vertex_buffer_name" )
        m_vertexBufferName = value;
    else if( name == "index_buffer_name" )
        m_indexBufferName = value;
    else if( name == "vertex_buffer_stride" )
        m_vertexBufferStride = corelib::from_string<int>( value );
    else if( name == "index_buffer_stride" )
        m_indexBufferStride = corelib::from_string<int>( value );
    else if( name == "motion_steps" )
    {
        m_requestedMotionSteps = corelib::from_string<int>( value );
        m_requestedMotionSteps = std::max( 1, m_requestedMotionSteps );
    }
    else if( name == "refine" )
    {
        // Note: refine is lwrrently not wired up (since goldenrod /
        // optix 4.0)
    }
    else if( name == "refit" )
    {
        m_refitNumber                  = corelib::from_string<int>( value );
        m_buildParameters.refitEnabled = ( m_refitNumber != 0 );
        updateBakingState();
    }

    if( m_builder )
    {
        // Push the property to the builder
        m_builder->setProperty( name, value );
        updateVisitProgram();
    }
}

std::string Acceleration::getProperty( const std::string& name ) const
{
    return m_properties.get<std::string>( name, "" );
}

size_t Acceleration::getDataSize() const
{
    if( isDirty() )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Cannot get data size from an acceleration structure marked dirty." );

    // Create a dummy serializer to compute the data size.
    Serializer size_aclwm;
    serialize( size_aclwm );
    return size_aclwm.getSize();
}

void Acceleration::getData( void* data ) const
{
    if( isDirty() )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Cannot get data from an acceleration structure marked dirty." );

    // Create the real serializer and copy the data.
    Serializer serializer( data );
    serialize( serializer );
}

void Acceleration::setData( const void* data, size_t size )
{
    // Create deserializer.
    Deserializer deserializer( data );

    // Deserialize API version.
    unsigned int api_version;
    if( size < sizeof( api_version ) )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Error in data stream (way too short)" );
    deserialize( deserializer, api_version );

    // Data before 0.8.1 didn't have stream size info.
    if( api_version < 81 )
        throw VersionMismatch( RT_EXCEPTION_INFO, "Data stream version too old" );

    // Deserialize data size and make sure it fits the provided stream (mismatch can
    // occur e.g. if there's been an IO error or some other corruption).
    uInt64 deser_size;
    if( size < sizeof( api_version ) + sizeof( deser_size ) )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Error in data stream (way too short)" );
    deserialize( deserializer, deser_size );
    if( deser_size != static_cast<uInt64>( size ) )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Error in data stream (size mismatch)" );

    // Deserialize builder and traverser.
    std::string  builder_name;
    std::string  traverser_name;
    unsigned int builder_version;
    deserialize( deserializer, builder_name );
    deserialize( deserializer, builder_version );
    deserialize( deserializer, traverser_name );

    // Check if builder name and version are valid.
    unsigned int version = m_builder->getVersion();
    if( version == ILWALID_BUILDER_VERSION )
        throw VersionMismatch( RT_EXCEPTION_INFO, "Unknown builder type in data stream" );
    if( version != builder_version )
        throw VersionMismatch( RT_EXCEPTION_INFO, "Builder version mismatch in data stream" );

    // The builder is OK, set it. This will reset the builder's properties,
    // but if the stream contains some they will be deserialized below.
    setBuilder( builder_name );

    // Deserialize remaining stuff.
    deserialize( deserializer, m_properties );

    // Dirty state becomes clean from the view of the external API,
    // but internally in need of a rebuild.
    clearDirty();
    markDirty( /*internallyDirty=*/true );

    // Validate since we changed the builder
    subscribeForValidation();
}

void Acceleration::markDirty( bool internallyDirty )
{
    if( m_dirty != DIRTY )
        m_dirty = internallyDirty ? DIRTY_INTERNAL : DIRTY;

    updateDirtyList();
}

void Acceleration::clearDirty()
{
    m_dirty = NON_DIRTY;

    updateDirtyList();
}

void Acceleration::updateDirtyList()
{
    // If the AS is not attached to a group or geometry group, do nothing for now.
    if( m_leafKind == RT_OBJECT_UNKNOWN )
        return;

    bool add = isAttached() && m_dirty != NON_DIRTY;
    if( m_leafKind == RT_OBJECT_GROUP )
        getContext()->getASManager()->addOrRemoveDirtyGroupAccel( this, add );
    else
        getContext()->getASManager()->addOrRemoveDirtyGeometryGroupAccel( this, add );
}

void Acceleration::markTopLevelTraversableDirty()
{
    getContext()->getASManager()->addOrRemoveDirtyTopLevelTraversable( this, true );
}

void Acceleration::clearTopLevelTraversableDirty()
{
    getContext()->getASManager()->addOrRemoveDirtyTopLevelTraversable( this, false );
}

bool Acceleration::isDirty() const
{
    // Unattached Accels should not be considered dirty, since we don't want to do anything to them.
    // Partilwlarly, their build buffers will not be allocated, so we don't want to try to build them.
    return ( m_dirty != NON_DIRTY ) && isAttached();
}
bool Acceleration::isDirtyExternal() const
{
    return m_dirty == DIRTY;
}

void Acceleration::validate() const
{
    // Attachment isn't always required to perform validation.  rtBufferValidate doesn't require
    // attachment, but ValidationManager does.  Do the attachment check in ValidationManager.

    LexicalScope::validate();

    // Make sure the user specified a builder
    if( m_builderType.empty() )
        throw ValidationError( RT_EXCEPTION_INFO, "Builder not set" );

    if( !m_unresolvedRemaining.empty() )
    {
        const VariableReference* varRef =
            m_context->getProgramManager()->getVariableReferenceById( m_unresolvedRemaining.front() );
        throw VariableNotFound( RT_EXCEPTION_INFO, this, "Unresolved reference to variable " + varRef->getInputName()
                                                             + " from " + varRef->getParent()->getInputFunctionName() );
    }
}

//------------------------------------------------------------------------
// Internal API
//------------------------------------------------------------------------

void Acceleration::setLeafKind( ObjectClass leafKind )
{
    if( m_leafKind == leafKind )
        return;
    RT_ASSERT_MSG( m_leafKind == RT_OBJECT_UNKNOWN, "Conflicting types for Acceleration leaf kind" );

    // Now that the leafkind is known, create the builder
    m_leafKind = leafKind;
    updateBuilder();
}

bool Acceleration::isAttachedToGeometryGroups() const
{
    RT_ASSERT_MSG( m_leafKind != RT_OBJECT_UNKNOWN, "Acceleration attachment not yet known" );
    return m_leafKind == RT_OBJECT_GEOMETRY_GROUP;
}


Acceleration::AbstractGroupIterator Acceleration::abstractGroupBegin() const
{
    return m_linkedPointers.begin();
}

Acceleration::AbstractGroupIterator Acceleration::abstractGroupEnd() const
{
    return m_linkedPointers.end();
}

AbstractGroup* Acceleration::getAbstractGroup( AbstractGroupIterator it ) const
{
    AbstractGroup* src = getLinkToAccelerationFrom<AbstractGroup>( *it );
    RT_ASSERT( src );  // make sure it's a linked ptr to a group type
    return src;
}

int Acceleration::getAbstractGroupCount() const
{
    return static_cast<int>( m_linkedPointers.size() );
}

AbstractGroup* Acceleration::getAnyAbstractGroup() const
{
    RT_ASSERT_MSG( isAttached(), "Abstract group requested for unattached group" );
    for( AbstractGroupIterator iter = abstractGroupBegin(); iter != abstractGroupEnd(); ++iter )
    {
        AbstractGroup* group = getAbstractGroup( iter );
        if( group->isAttached() )
            return group;
    }
    RT_ASSERT_FAIL_MSG( "No attached groups found" );  // First attatched. If none then throw
}

void Acceleration::overrideBvh_forTesting( const std::vector<BvhNode>& nodes, const std::vector<int>& indices )
{
    m_builder->overrideBvh_forTesting( nodes, indices );
}

unsigned int Acceleration::getTotalPrimitiveCount() const
{
    for( AbstractGroupIterator iter = abstractGroupBegin(); iter != abstractGroupEnd(); ++iter )
    {
        AbstractGroup* group = getAbstractGroup( iter );
        if( group->isAttached() )
        {
            unsigned int n = group->getChildCount();
            if( group->isGeometryGroup() )
            {
                unsigned int total = 0;
                for( unsigned int i = 0; i < n; ++i )
                {
                    if( GeometryInstance* gi = group->getChild<GeometryInstance>( i ) )
                        if( Geometry* geom = gi->getGeometry() )
                            total += geom->getPrimitiveCount();
                }
                return total;
            }
            else
            {
                return n;
            }
        }
    }
    return 0;
}

void Acceleration::updateBakingState()
{
    bool newBake = k_bakeChildPointers.get() && getAbstractGroupCount() <= 1 && m_refitNumber == 0;
    if( newBake != m_bakePrimitiveEntities )
    {
        m_bakePrimitiveEntities = newBake;
        updateVisitProgram();
        markDirty( /*internallyDirty=*/true );
    }
}

void Acceleration::groupsChildOffsetDidChange()
{
    if( m_bakePrimitiveEntities )
        markDirty( /*internallyDirty=*/true );
}

Program* Acceleration::getLwrrentVisitProgram() const
{
    return m_visitProgram;
}

Program* Acceleration::getLwrrentBoundingBoxProgram() const
{
    return m_boundingBoxProgram;
}

RtcTraversableHandle Acceleration::getTraversableHandle( unsigned int allDeviceIndex ) const
{
    if( !m_context->useRtxDataModel() )
        return 0U;

    if( m_builder )
        return m_builder->getTraversableHandle( allDeviceIndex );
    else
        return 0U;
}

void Acceleration::traversableHandleDidChange( RtcTraversableHandle newHandle, unsigned int allDeviceIndex )
{
    for( const auto parentLink : m_linkedPointers )
    {
        if( AbstractGroup* parent = getLinkToAccelerationFrom<AbstractGroup>( parentLink ) )
            parent->asTraversableHandleDidChange( newHandle, allDeviceIndex );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to Acceleration" );
    }
}


//------------------------------------------------------------------------
// AS Manager interface
//------------------------------------------------------------------------

int Acceleration::getMaxAccelerationHeight() const
{
    if( m_accelerationHeight.empty() )
        return -1;  // No Groups have attached to this acceleration

    return m_accelerationHeight.back();
}

bool Acceleration::hasMotionAabbs() const
{
    return !m_hasMotionAabbs.empty();
}

bool optix::Acceleration::hasMotionAabbs_publicMethodForTesting() const
{
    return hasMotionAabbs();
}

void Acceleration::setupForBuild( DeviceSet buildDevices )
{
    RT_ASSERT( m_builder != nullptr );
    RT_ASSERT( m_leafKind == RT_OBJECT_GROUP || m_leafKind == RT_OBJECT_GEOMETRY_GROUP );

    m_buildParameters.buildDevices = buildDevices;

    // Set up build parameters for motion blur, using an infinite time
    // range. The time range will get overwritten for motion aabbs.
    // Note that m_requestedMotionSteps may be negative!
    // This signals, that the number of motion steps is set lazily (see setupForBuildFromTriangles) and m_buildParameters.motionSteps may be overwritten
    m_buildParameters.motionSteps = hasMotionAabbs() ? ( m_requestedMotionSteps < 1 ? 2 : m_requestedMotionSteps ) : 1;
    m_buildParameters.motionTimeBegin = -std::numeric_limits<float>::max();
    m_buildParameters.motionTimeEnd   = std::numeric_limits<float>::max();

    if( getContext()->useRtxDataModel() )
    {
        RtcBvhAccelType type = RTC_BVH_ACCEL_TYPE_BVH2;

#if LWCFG( GLOBAL_ARCH_TURING ) || LWCFG( GLOBAL_ARCH_AMPERE )
        if( m_builderType == "TTU" )
        {
            if( m_context->getDeviceManager()->activeDevicesSupportMotionTTU() || m_buildParameters.motionSteps == 1 )
            {
                type = RTC_BVH_ACCEL_TYPE_TTU;
            }
            else
            {
                lprint << "TTU does not support motion blur, falling back to default builder" << std::endl;
            }
        }
#endif
        if( m_builderType == "NoAccel" )
            type = RTC_BVH_ACCEL_TYPE_NOACCEL;
        else if( m_builderType == "Bvh8" )
            type = RTC_BVH_ACCEL_TYPE_BVH8;

        static_cast<RtcBvh*>( m_builder.get() )->setRtcAccelType( type );
    }

    // Decide if we should try to refit. Note: setup might squash the
    // refit if the primitive count does not match. Furthermore, the
    // builder might also decline to refit.
    m_buildParameters.shouldRefitThisFrame = false;
    if( m_refitNumber != 0 )
    {
        // Determine if we are eligible for refit
        bool canRefit = true;

        // No refit for motion blur
        if( hasMotionAabbs() )
            canRefit = false;

        // If elegible for refit and it is not time to rebuild, then refit
        if( canRefit && ( m_refitNumber == 1 || m_buildParameters.framesSinceLastBuild < m_refitNumber ) )
            m_buildParameters.shouldRefitThisFrame = true;
    }

    // Setup the appropriate build.
    if( m_leafKind == RT_OBJECT_GROUP )
    {
        // Build for a group
        setupForBuildGroup();
    }
    else
    {
        // Create the data used by both build paths
        setupGeometryInstanceData();

        // Attempt to setup for triangle build. If that fails, use the
        // AABB path. Note that the builder may choose to use the AABB
        // path anyway.
        if( !m_enableTriangleBuilds || !setupForBuildFromTriangles() )
            setupForBuildFromAABBs();
    }

    if( hasMotionAabbs() )
    {
        // Save motion blur data
        cort::MotionAccelerationRecord* mar = getObjectRecord<cort::MotionAccelerationRecord>();
        mar->motionSteps                    = m_buildParameters.motionSteps;
        mar->motionStride                   = m_buildSetupRequest.motionStride;

        // Determine temp space needed for resampling aabbs
        if( m_leafKind == RT_OBJECT_GEOMETRY_GROUP )
        {
            // Update time range in object record
            float buildTimeBegin = m_buildParameters.motionTimeBegin;
            float buildTimeEnd   = m_buildParameters.motionTimeEnd;
            mar->timeBegin       = buildTimeBegin;
            mar->timeEnd         = buildTimeEnd;

            // Ensure that temp request is big enough
            size_t resampleTempSize      = computeResampleTempSize( buildTimeBegin, buildTimeEnd );
            m_buildSetupRequest.tempSize = std::max( m_buildSetupRequest.tempSize, resampleTempSize * sizeof( Aabb ) );
        }
        else
        {
            // Motion requests require space for AABBs and the motion requests
            size_t motionSize            = computeMotionAABBRequestSize();
            m_buildSetupRequest.tempSize = std::max( m_buildSetupRequest.tempSize, motionSize );
        }
    }
}

void Acceleration::build( const BuildSetup& setup )
{
    RT_ASSERT( m_builder != nullptr );
    RT_ASSERT( m_leafKind == RT_OBJECT_GROUP || m_leafKind == RT_OBJECT_GEOMETRY_GROUP );

    if( m_leafKind == RT_OBJECT_GROUP )
    {
        buildGroup( setup );
    }
    else
    {
        buildGeometryGroup( setup );
    }

    // Update frame counter for refit.
    if( setup.willRefit )
        ++m_buildParameters.framesSinceLastBuild;
    else
        m_buildParameters.framesSinceLastBuild = 0;

    // Clear dirty but do not take it out of the list. ASManager will
    // clear the entire list when finished.
    m_dirty = NON_DIRTY;
}

bool Acceleration::setupForBuildFromTriangles()
{
    m_triangleData.clear();
    AbstractGroup* gg = managedObjectCast<GeometryGroup>( getAnyAbstractGroup() );
    RT_ASSERT( gg != nullptr );

    int giCount = gg->getChildCount();
    if( giCount == 0 )
        return false;
    const GeometryInstance*  gi       = gg->getChild<GeometryInstance>( 0 );
    const GeometryTriangles* geomTris = managedObjectCast<GeometryTriangles>( gi->getGeometry() );
    if( geomTris != nullptr )
    {
        std::vector<TriangleData> tridata( giCount );

        int                motionSteps = 1;
        RTmotionbordermode beginMode, endMode;
        float              timeBegin, timeEnd;
        for( int i = 0; i < giCount; ++i )
        {
            // GG was validated to either contain only Geometry or GeometryTriangles as geometries
            const GeometryInstance*  gi       = gg->getChild<GeometryInstance>( i );
            const GeometryTriangles* geomTris = static_cast<GeometryTriangles*>( gi->getGeometry() );

            if( hasMotionAabbs() )
            {
                if( m_requestedMotionSteps >= 1 && geomTris->getMotionSteps() != m_requestedMotionSteps )
                    return false;

                // all geometries must agree on their motion setting
                if( i == 0 )
                {
                    motionSteps = geomTris->getMotionSteps();
                    geomTris->getMotionBorderMode( beginMode, endMode );
                    geomTris->getMotionRange( timeBegin, timeEnd );
                }
                else
                {
                    if( motionSteps != geomTris->getMotionSteps() )
                        return false;

                    RTmotionbordermode geometryBeginMode, geometryEndMode;
                    geomTris->getMotionBorderMode( geometryBeginMode, geometryEndMode );

                    if( geometryBeginMode != beginMode || geometryEndMode != endMode )
                        return false;

                    float geometryTimeBegin, geometryTimeEnd;
                    geomTris->getMotionRange( geometryTimeBegin, geometryTimeEnd );

                    if( geometryTimeBegin != timeBegin || geometryTimeEnd != timeEnd )
                        return false;
                }
            }

            TriangleData& td      = tridata[i];
            td.vertices           = geomTris->getVertexBuffers().front()->getMBuffer();
            td.vertexBufferOffset = geomTris->getVertexBufferByteOffset();
            td.vertexStride       = geomTris->getVertexByteStride();
            td.positionFormat     = geomTris->getPositionFormat();
            if( !geomTris->isIndexedTriangles() )
                td.indices = nullptr;
            else
            {
                td.indices           = geomTris->getIndexBuffer()->getMBuffer();
                td.indexBufferOffset = geomTris->getIndexBufferByteOffset();
                td.triIndicesStride  = geomTris->getTriIndicesByteStride();
                td.triIndicesFormat  = geomTris->getTriIndicesFormat();
            }
        }

        m_buildParameters.motionSteps = motionSteps;

        // the number of motion steps is limited to 8-bit for native triangle motion blur
        if( motionSteps > MAX_MOTION_STEPS_PER_TRIANGLE )
            return false;

        // Success: save the triangle data and setup the builder
        m_triangleData = std::move( tridata );
    }
    else if( k_enableMKTriangleBuilds.get() )
    {
        // Find tokens for the triangle buffer names
        ObjectManager* om                = m_context->getObjectManager();
        unsigned short indexBufferToken  = om->getTokenForVariableName( m_indexBufferName );
        unsigned short vertexBufferToken = om->getTokenForVariableName( m_vertexBufferName );
        unsigned short ilwalid_token     = IDMap<std::string, unsigned short>::ILWALID_INDEX;
        if( indexBufferToken == ilwalid_token || vertexBufferToken == ilwalid_token )
            // If the tokens are not present, then these variables are not used in the program
            return false;

        int                       giCount = gg->getChildCount();
        std::vector<TriangleData> tridata( giCount );
        for( int i = 0; i < giCount; ++i )
        {
            const GeometryInstance* gi         = gg->getChild<GeometryInstance>( i );
            const Geometry*         geom       = gi->getGeometry();
            int                     primCount  = geom->getPrimitiveCount();
            int                     primOffset = geom->getPrimitiveIndexOffset();

            // Find buffer variables
            Variable* ivar = lookupBufferVariable( gi, indexBufferToken );
            if( ivar && !ivar->getType().isBuffer() )
                ivar       = nullptr;
            Variable* vvar = lookupBufferVariable( gi, vertexBufferToken );
            if( vvar && !vvar->getType().isBuffer() )
                vvar = nullptr;

            // Sanity check buffer info
            if( ivar && vvar )
            {
                // Indices and vertices
                size_t expectedSize = ( primCount + primOffset ) * m_indexBufferStride;
                size_t bufferSize   = ivar->getBuffer()->getTotalSizeInBytes();
                if( expectedSize > bufferSize )
                    throw IlwalidValue( RT_EXCEPTION_INFO,
                                        "Invalid index buffer size. Possible reasons: incorrect primitive "
                                        "count or stride too large." );
            }
            else if( !ivar && vvar )
            {
                // Vertices only
                size_t expectedSize = ( primCount + primOffset ) * 3 * m_vertexBufferStride;
                size_t bufferSize   = vvar->getBuffer()->getTotalSizeInBytes();
                if( expectedSize > bufferSize )
                    throw IlwalidValue( RT_EXCEPTION_INFO,
                                        "Invalid vertex buffer size. Possible reasons: stride too large with "
                                        "no index buffer." );
            }
            else if( ivar && !vvar )
            {
                // Indices only
                ureport1( m_context->getUsageReport(), "USAGE HINT" )
                    << "Acceleration found index_buffer but no vertex_buffer.  Falling back to non-triangle build."
                    << std::endl;
                return false;
            }
            else
            {
                // Neither indices nor vertices
                // Fall back to AABB path
                return false;
            }

            // Create auxiliary triangle information for builder.
            TriangleData& td    = tridata[i];
            td.indices          = ivar ? ivar->getBuffer()->getMBuffer() : nullptr;
            td.vertices         = vvar->getBuffer()->getMBuffer();
            td.triIndicesStride = m_indexBufferStride;
            td.vertexStride     = m_vertexBufferStride;
        }

        // Success: save the triangle data and setup the builder
        m_triangleData = std::move( tridata );
    }
    else
        return false;

    // Squash refit if we have the wrong number of primitives.
    int totalPrims = m_geometryInstanceData.empty() ? 0 : m_geometryInstanceData.back().primStart
                                                              + m_geometryInstanceData.back().primCount;
    if( totalPrims != m_lastBuildPrimitiveCount || m_buildParameters.motionSteps != m_lastBuildMotionSteps )
        m_buildParameters.shouldRefitThisFrame = false;
    m_lastBuildPrimitiveCount                  = totalPrims;
    m_lastBuildMotionSteps                     = m_buildParameters.motionSteps;

    m_buildSetupRequest = m_builder->setupForBuild( m_buildParameters, totalPrims, m_geometryInstanceData, m_triangleData );
    return true;
}

void Acceleration::setupForBuildFromAABBs()
{
    // Squash refit if we have the wrong number of primitives.
    int totalPrims = m_geometryInstanceData.empty() ? 0 : m_geometryInstanceData.back().primStart
                                                              + m_geometryInstanceData.back().primCount;
    if( totalPrims != m_lastBuildPrimitiveCount || m_buildParameters.motionSteps != m_lastBuildMotionSteps )
        m_buildParameters.shouldRefitThisFrame = false;
    m_lastBuildPrimitiveCount                  = totalPrims;
    m_lastBuildMotionSteps                     = m_buildParameters.motionSteps;

    // Ilwoke setup on the builder
    m_buildSetupRequest = m_builder->setupForBuild( m_buildParameters, totalPrims, m_geometryInstanceData );
}

void Acceleration::finalizeAfterBuild( DeviceSet buildDevices )
{
    RT_ASSERT( m_builder != nullptr );

    // Setup the appropriate build.
    if( m_leafKind == RT_OBJECT_GROUP )
    {
        m_builder->finalizeAfterBuildGroups( m_buildParameters );
    }
    else
    {
        if( m_triangleData.empty() )
            m_builder->finalizeAfterBuildPrimitives( m_buildParameters );
        else
            m_builder->finalizeAfterBuildTriangles( m_buildParameters );

        // Free up temporaries
        m_triangleData.clear();
    }
}

#define STATIC_ASSERT( a, b ) static_assert( (unsigned)a == (unsigned)b, "" );
STATIC_ASSERT( RTC_GEOMETRY_FLAG_NONE, RT_GEOMETRY_FLAG_NONE );
STATIC_ASSERT( RTC_GEOMETRY_FLAG_OPAQUE, RT_GEOMETRY_FLAG_DISABLE_ANYHIT );
STATIC_ASSERT( RTC_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_ILWOCATION, RT_GEOMETRY_FLAG_NO_SPLITTING );
#undef STATIC_ASSERT

void Acceleration::setupGeometryInstanceData()
{
    AbstractGroup* gg = managedObjectCast<GeometryGroup>( getAnyAbstractGroup() );
    RT_ASSERT( gg != nullptr );

    int giCount = gg->getChildCount();

    m_geometryInstanceData.clear();
    size_t                            totalPrims = 0;
    std::vector<GeometryInstanceData> gidata( giCount );
    float                             timeBegin = std::numeric_limits<float>::max();
    float                             timeEnd   = -std::numeric_limits<float>::max();
    for( int i = 0; i < giCount; ++i )
    {
        const GeometryInstance* gi         = gg->getChild<GeometryInstance>( i );
        const Geometry*         geom       = gi->getGeometry();
        int                     primCount  = geom->getPrimitiveCount();
        int                     primOffset = geom->getPrimitiveIndexOffset();

        // Fill information for builder
        GeometryInstanceData& data = gidata[i];
        data.primCount             = primCount;
        data.primOffset            = primOffset;
        data.primStart             = totalPrims;
        data.giOffsetOrChildIndex  = 0;  //  giOffset is filled at build time since it might change.
        data.giOffset              = 0;
        data.motionSteps           = geom->getMotionSteps();
        data.flags                 = geom->getFlags();
        data.g                     = geom;
        geom->getMotionRange( data.timeBegin, data.timeEnd );

        // Expand motion of builder
        if( data.motionSteps > 1 )
        {
            timeBegin = std::min( timeBegin, data.timeBegin );
            timeEnd   = std::max( timeEnd, data.timeEnd );
        }

        totalPrims += primCount;
    }

    // Save build parameters for motion blur
    m_buildParameters.motionTimeBegin = timeBegin;
    m_buildParameters.motionTimeEnd   = timeEnd;

    // Success: save the data and setup the builder
    m_geometryInstanceData = std::move( gidata );
}

namespace {
struct AabbResample
{
    cort::Aabb*        aabbs;
    cort::Aabb*        destPtr;
    optix::LWDADevice* buildDevice;
};
}  // end anonymous namespace


void Acceleration::computeAABBs( const BuildParameters& params, const BuildSetup& setup )
{
    const timerTick t0 = getTimerTick();

    // Ilwoke the bounds programs on each GeometryInstance.
    unsigned int buildMotionSteps = params.motionSteps;
    float        buildTimeBegin   = params.motionTimeBegin;
    float        buildTimeEnd     = params.motionTimeEnd;

    const int                                numDevices = static_cast<int>( setup.perDevice.size() );
    std::vector<unsigned int>                tempStart( numDevices, 0 );
    std::vector<std::shared_ptr<WaitHandle>> waitHandles;

    for( unsigned int i = 0; i < m_geometryInstanceData.size(); ++i )
    {
        const GeometryInstanceData& gid             = m_geometryInstanceData[i];
        int                         numPrims        = gid.primCount;
        unsigned int                giOffset        = gid.giOffset;
        unsigned int                geomMotionSteps = gid.motionSteps;
        float                       geomTimeBegin   = gid.timeBegin;
        float                       geomTimeEnd     = gid.timeEnd;

        // Determine whether we simply need the union (1:n and n:1
        // mappings) and if we need to resample aabbs. If there is no
        // motion blur (both motion steps == 1) then we use the
        // non-union path since it will be slightly more
        // efficient. Without motion blur, the lighter-weight kernels
        // runs that ignores these parameters anyway.
        const bool computeUnion = ( geomMotionSteps == 1 ) ^ ( buildMotionSteps == 1 );
        const bool geometryNeedsResample =
            geomMotionSteps > 1 && ( ( geomTimeBegin != buildTimeBegin || geomTimeEnd != buildTimeEnd )
                                     || ( buildMotionSteps > 1 && geomMotionSteps != buildMotionSteps ) );

        std::vector<AabbResample> resampleData;
        resampleData.reserve( geometryNeedsResample ? numDevices : 0 );
        for( int pdidx = 0; pdidx < numDevices; ++pdidx )
        {
            // Compute AABB start address. If we need to resample, compute
            // the bounding boxes into a temporary staging area. The
            // staging area is lwrrently non-overlapping so that this code
            // could safely use multiple streams in the future. However,
            // temp area overlaps builder temp space.
            cort::Aabb* aabbs   = setup.perDevice[pdidx].deviceAabbPtr + gid.primStart * buildMotionSteps;
            cort::Aabb* destPtr = aabbs;
            if( geometryNeedsResample )
            {
                destPtr = (cort::Aabb*)setup.perDevice[pdidx].deviceTempPtr + tempStart[pdidx];
                tempStart[pdidx] += numPrims * geomMotionSteps;
                resampleData.push_back( AabbResample{aabbs, destPtr, setup.perDevice[pdidx].buildDevice} );
            }

            // Perform a kernel launch to ilwoke the AABB
            // program. ASManager has already made the right entry point
            // active.
            cort::AabbRequest aabbRequest( false, giOffset, buildMotionSteps, geomMotionSteps, computeUnion, destPtr, nullptr );
            waitHandles.push_back( m_context->launchFrameAsync( numPrims, 1, 1, DeviceSet( setup.perDevice[pdidx].buildDevice ),
                                                                aabbRequest, lwca::Stream() ) );
        }

        // Make sure all launches complete before continuing
        for( auto& waitHandle : waitHandles )
            waitHandle->block();
        waitHandles.clear();

        // If necessary, resample the bounding boxes using a device
        // kernel.
        for( const auto& resample : resampleData )
        {
            resample.buildDevice->makeLwrrent();
            const lwca::ComputeCapability sm_ver = resample.buildDevice->computeCapability();
            LWstream                      stream = resample.buildDevice->primaryStream().get();
            resampleMotionAabbsWithRegularDistributionDevice( sm_ver, stream, numPrims, (optix::Aabb*)resample.destPtr, geomMotionSteps,
                                                              geomTimeBegin, geomTimeEnd, (optix::Aabb*)resample.aabbs,
                                                              buildMotionSteps, buildTimeBegin, buildTimeEnd );
        }
    }  // End loop over GIs


    if( m_buildSetupRequest.needAabbsOnCpu )
    {
        // If the AABBs are needs on CPU, then copy them back from the
        // device.
        size_t totalSize = setup.totalPrims * sizeof( cort::Aabb ) * buildMotionSteps;
        for( const auto& perDevice : setup.perDevice )
            lwca::memcpyDtoH( perDevice.cpuAabbPtr, (LWdeviceptr)perDevice.deviceAabbPtr, totalSize );
    }

    const float dt = getDeltaMilliseconds( t0 );
    Metrics::logFloat( "build_aabbs_msec", dt );

#if defined( DEBUG ) || defined( DEVELOP )
    if( k_printAabbs.get() )
    {
        lprint << "GeometryGroup AABBS for AS ID: " << getScopeID() << '\n';
        // Print AABBs if requested
        std::vector<Aabb> aabbs( setup.totalPrims * buildMotionSteps );
        size_t            totalSize = setup.totalPrims * sizeof( cort::Aabb ) * buildMotionSteps;
        for( const auto& perDevice : setup.perDevice )
        {
            lwca::memcpyDtoH( aabbs.data(), (LWdeviceptr)perDevice.deviceAabbPtr, totalSize );
            printAabbs( setup.totalPrims, buildMotionSteps, aabbs );
        }
    }
#endif
}

void Acceleration::printAabbs( unsigned int totalPrims, unsigned int buildMotionSteps, const std::vector<Aabb>& aabbs ) const
{
#if defined( DEBUG ) || defined( DEVELOP )
    for( size_t i = 0; i < totalPrims; ++i )
    {
        for( size_t j = 0; j < buildMotionSteps; ++j )
        {
            const size_t index = i * buildMotionSteps + j;
            lprint
                << "aabb[" << i << "][" << j << "] : "
                << "(" << aabbs[index].m_min.x << ", " << aabbs[index].m_min.y << ", " << aabbs[index].m_min.z << ")  "
                << "(" << aabbs[index].m_max.x << ", " << aabbs[index].m_max.y << ", " << aabbs[index].m_max.z << ")\n";
        }
    }
    lprint << '\n';
#endif
}

void Acceleration::buildGeometryGroup( const BuildSetup& setup )
{
    // Update the GI offsets before the build to ensure that we have a
    // current GI offset.
    AbstractGroup* group = getAnyAbstractGroup();
    for( unsigned int i = 0; i < m_geometryInstanceData.size(); ++i )
    {
        unsigned int offset                            = group->getChild( i )->getRecordOffset();
        m_geometryInstanceData[i].giOffsetOrChildIndex = m_bakePrimitiveEntities ? offset : i;
        m_geometryInstanceData[i].giOffset             = offset;
    }

    // Compute AAbbs if they were requested
    if( m_buildSetupRequest.needAabbs )
        computeAABBs( m_buildParameters, setup );

    // Issue the build
    if( m_triangleData.empty() )
        m_builder->build( m_buildParameters, setup, m_geometryInstanceData );
    else
        m_builder->build( m_buildParameters, setup, m_geometryInstanceData, m_triangleData );

    if( m_context->useRtxDataModel() )
    {
        bool topLevelTraversableIsDirty = false;
        for( AbstractGroupIterator iter = abstractGroupBegin(); iter != abstractGroupEnd(); ++iter )
        {
            AbstractGroup* group = getAbstractGroup( iter );
            if( group->isAttached() )
            {
                for( const auto& perDeviceSetup : setup.perDevice )
                {
                    topLevelTraversableIsDirty = true;
                    m_topLevelTraversableInputs.push_back(
                        TopLevelTraversableInput{group, perDeviceSetup.deviceAabbPtr, perDeviceSetup.buildDevice} );
                }
            }
        }
        if( topLevelTraversableIsDirty )
            markTopLevelTraversableDirty();
    }

    // Reset the temp arrays
    m_geometryInstanceData.clear();
}

void Acceleration::writeTopLevelTraversable()
{
    for( const auto& topLevelTraversableInput : m_topLevelTraversableInputs )
        managedObjectCast<GeometryGroup>( topLevelTraversableInput.group )
            ->writeTopLevelTraversable( topLevelTraversableInput.deviceAabbPtr, topLevelTraversableInput.buildDevice );

    m_topLevelTraversableInputs.clear();
}

void Acceleration::setupForBuildGroup()
{
    AbstractGroup* group = managedObjectCast<Group>( getAnyAbstractGroup() );
    RT_ASSERT( group != nullptr );

    std::unique_ptr<GroupData> groupdata( new GroupData );
    groupdata->childCount         = group->getChildCount();
    groupdata->children           = group->getChildren();
    groupdata->bakeChildAddresses = m_bakePrimitiveEntities;

    // Squash refit if necessary
    if( groupdata->childCount != m_lastBuildPrimitiveCount || m_buildParameters.motionSteps != m_lastBuildMotionSteps )
        m_buildParameters.shouldRefitThisFrame = false;
    m_lastBuildPrimitiveCount                  = groupdata->childCount;
    m_lastBuildMotionSteps                     = m_buildParameters.motionSteps;

    // Save the data and setup the builder
    m_groupData         = std::move( groupdata );
    m_buildSetupRequest = m_builder->setupForBuild( m_buildParameters, *m_groupData );
}

void Acceleration::buildGroup( const BuildSetup& setup )
{
    if( getContext()->useRtxDataModel() )
    {
        // rtcore computes the instance AABBs, no need to do that here.
        // however, we need to find the time begin/end times!
        AbstractGroup*     parentGroup    = managedObjectCast<Group>( getAnyAbstractGroup() );
        const unsigned int childCount     = parentGroup->getChildCount();
        float              buildTimeBegin = std::numeric_limits<float>::max();
        float              buildTimeEnd   = -std::numeric_limits<float>::max();

        // Loop over all children, compute/gather MotionAABBs, and find range for
        // buildTimeBegin/buildTimeEnd and copy child MotionAABBs into a linear array.
        for( unsigned int i = 0; i < childCount; ++i )
        {
            GraphNode* child = parentGroup->getChild<GraphNode>( i );
            while( child )
            {
                if( child->getClass() == RT_OBJECT_TRANSFORM )
                {
                    Transform* xform = static_cast<Transform*>( child );
                    if( xform->hasMotionKeys() )
                    {
                        float timeBegin, timeEnd;
                        xform->getMotionRange( timeBegin, timeEnd );
                        buildTimeBegin = std::min( buildTimeBegin, timeBegin );
                        buildTimeEnd   = std::max( buildTimeEnd, timeEnd );
                    }

                    child = xform->getChild();
                }
                //// not supported in rtx mode
                //else if( node->getClass() == RT_OBJECT_SELECTOR )
                //{
                //}
                else
                {
                    RT_ASSERT_MSG( child->getClass() == RT_OBJECT_GROUP || child->getClass() == RT_OBJECT_GEOMETRY_GROUP,
                                   "Invalid object found in graph" );
                    AbstractGroup* group = static_cast<AbstractGroup*>( child );

                    // Retrieve the motion range and build steps
                    const float timeBegin = group->getAcceleration()->m_buildParameters.motionTimeBegin;
                    const float timeEnd   = group->getAcceleration()->m_buildParameters.motionTimeEnd;
                    buildTimeBegin        = std::min( buildTimeBegin, timeBegin );
                    buildTimeEnd          = std::max( buildTimeEnd, timeEnd );

                    child = nullptr;
                }
            }
        }
        m_buildParameters.motionTimeBegin = buildTimeBegin;
        m_buildParameters.motionTimeEnd   = buildTimeEnd;
    }
    else if( m_buildSetupRequest.needAabbs )
    {
        if( m_buildParameters.motionSteps == 1 )
        {
            // No motion blur. Perform a kernel launch to ilwoke the AABB
            // program. ASManager has already made the right entry point
            // active.
            AbstractGroup* group       = managedObjectCast<Group>( getAnyAbstractGroup() );
            int            childCount  = group->getChildCount();
            unsigned int   groupOffset = LexicalScope::getSafeOffset( group );

            for( const auto& perDevice : setup.perDevice )
            {
                cort::Aabb*       aabbs = perDevice.deviceAabbPtr;
                cort::AabbRequest aabbRequest( true, groupOffset, 0, 0, false, aabbs, nullptr );
                RT_ASSERT_MSG( perDevice.buildDevice,
                               "TODO, if CPU builder, where do we get the build device from for aabb launch" );
                m_context->launchFrame( childCount, 1, 1, DeviceSet( perDevice.buildDevice ), aabbRequest );
            }

            if( m_buildSetupRequest.needAabbsOnCpu )
            {
                // Copy to back to CPU if necessary.
                size_t totalSize = childCount * sizeof( cort::Aabb );
                for( const auto& perDevice : setup.perDevice )
                    lwca::memcpyDtoH( perDevice.cpuAabbPtr, (LWdeviceptr)perDevice.deviceAabbPtr, totalSize );
            }
#if defined( DEBUG ) || defined( DEVELOP )
            if( k_printAabbs.get() )
            {
                lprint << "Group AABBS for AS ID: " << getScopeID() << '\n';
                // Print AABBs if requested
                std::vector<Aabb> aabbs( setup.totalPrims * m_buildParameters.motionSteps );
                size_t            totalSize = setup.totalPrims * sizeof( cort::Aabb ) * m_buildParameters.motionSteps;
                for( const auto& perDevice : setup.perDevice )
                {
                    lwca::memcpyDtoH( aabbs.data(), (LWdeviceptr)perDevice.deviceAabbPtr, totalSize );
                    printAabbs( setup.totalPrims, m_buildParameters.motionSteps, aabbs );
                }
            }
#endif
        }
        else
        {
            // Motion blur - compute resampled bounding boxes.
            resampleAABBs( setup );
        }
    }


    // Issue the build
    m_builder->build( m_buildParameters, setup, *m_groupData );
    m_groupData.reset();
}

size_t Acceleration::computeResampleTempSize( float buildTimeBegin, float buildTimeEnd ) const
{
    unsigned int buildMotionSteps = m_buildParameters.motionSteps;

    unsigned int resampleTempSize = 0;
    for( unsigned int i = 0; i < m_geometryInstanceData.size(); ++i )
    {
        unsigned int geomMotionSteps = m_geometryInstanceData[i].motionSteps;
        float        geomTimeBegin   = m_geometryInstanceData[i].timeBegin;
        float        geomTimeEnd     = m_geometryInstanceData[i].timeEnd;

        const bool geometryNeedsResample =
            geomMotionSteps > 1 && ( ( geomTimeBegin != buildTimeBegin || geomTimeEnd != buildTimeEnd )
                                     || ( buildMotionSteps > 1 && geomMotionSteps != buildMotionSteps ) );
        if( geometryNeedsResample )
            resampleTempSize += m_geometryInstanceData[i].primCount * m_geometryInstanceData[i].motionSteps;
    }
    return resampleTempSize;
}

static void visitAbstractGroup( AbstractGroup* parentGroup, std::function<void( AbstractGroup* )> fn )
{
    const unsigned int childCount = parentGroup->getChildCount();

    std::set<int>          visited;
    std::queue<GraphNode*> q;
    for( unsigned int i = 0; i < childCount; ++i )
    {
        GraphNode* child = parentGroup->getChild<GraphNode>( i );
        q.push( child );
    }

    while( !q.empty() )
    {
        GraphNode* gn = q.front();
        q.pop();

        while( gn && gn->getClass() == RT_OBJECT_TRANSFORM )
            gn = static_cast<Transform*>( gn )->getChild();

        if( !gn )
            continue;

        if( gn->getClass() == RT_OBJECT_SELECTOR )
        {
            // Selector: Insert children into queue.
            Selector*    selector = static_cast<Selector*>( gn );
            unsigned int n        = selector->getChildCount();

            for( unsigned int i = 0; i < n; ++i )
            {
                GraphNode* child   = selector->getChild( i );
                int        childid = child->getScopeID();
                if( visited.count( childid ) == 0 )
                    q.push( child );
            }
        }
        else
        {
            // Group: Add to size
            RT_ASSERT_MSG( gn->getClass() == RT_OBJECT_GROUP || gn->getClass() == RT_OBJECT_GEOMETRY_GROUP,
                           "Invalid object found in graph" );
            AbstractGroup* group   = static_cast<AbstractGroup*>( gn );
            int            groupid = group->getScopeID();
            if( visited.count( groupid ) == 0 )
            {
                visited.insert( groupid );
                fn( group );
            }
        }
    }
}

size_t Acceleration::computeMotionAABBRequestSize() const
{
    AbstractGroup* group = managedObjectCast<Group>( getAnyAbstractGroup() );

    // Walk the tree to determine how many AABBs we need to bring back from the device
    unsigned int totalSize    = 0;
    unsigned int requestCount = 0;
    visitAbstractGroup( group, [&]( AbstractGroup* group ) {
        totalSize += group->getAcceleration()->m_buildParameters.motionSteps;
        requestCount++;
    } );

    return totalSize * sizeof( Aabb ) + requestCount * sizeof( int2 );
}

void Acceleration::resampleAABBs( const BuildSetup& setup )
{
    AbstractGroup* parentGroup = managedObjectCast<Group>( getAnyAbstractGroup() );
    unsigned int   totalSize   = 0;
    std::map<int, int> gatherList;  // Map scopeid -> beginning of motion aabbs
    std::vector<cort::uint2> requestList;
    std::vector<Builder*>    blGroups;

    // Compute the unique set of acceleration objects for which we
    // need to find aabbs. Note that this will probably be fairly
    // expensive but will hopefully be used with a limited number of
    // children.
    visitAbstractGroup( parentGroup, [&]( AbstractGroup* group ) {
        // Add to request list and remember the location in
        // the map/request lists.
        const int groupid = group->getScopeID();
        gatherList.emplace( groupid, totalSize );
        requestList.push_back( {(unsigned int)group->getRecordOffset(), totalSize} );
        totalSize += group->getAcceleration()->m_buildParameters.motionSteps;
        blGroups.push_back( group->getAcceleration()->m_builder.get() );
    } );

    const BuildSetup::PerDevice& perDevice           = setup.perDevice[0];
    cort::Aabb*                  deviceGatheredAabbs = reinterpret_cast<cort::Aabb*>( perDevice.deviceTempPtr );
    if( getContext()->useRtxDataModel() )
    {
        RtcBvh::copyMotionAabbs( blGroups, perDevice.deviceTempPtr, perDevice.buildDevice );
    }
    else
    {
        // here, we assume that all devices have an identical copy of all AS'es
        // gather the root AABBs on one device only
        // for( const auto& perDevice : setup.perDevice )
        {
            // Push the requests to the device
            cort::uint2* deviceRequests = (cort::uint2*)( perDevice.deviceTempPtr + totalSize * sizeof( Aabb ) );
            lwca::memcpyHtoD( (LWdeviceptr)deviceRequests, requestList.data(), requestList.size() * 2 * sizeof( unsigned int ) );

            // Gather motion aabbs on the device
            cort::AabbRequest aabbRequest( true, 0, 0, 0, false, deviceGatheredAabbs, deviceRequests );
            m_context->launchFrame( requestList.size(), 1, 1, DeviceSet( perDevice.buildDevice ), aabbRequest );
        }
    }

    // Copy the Aabbs back from the device
    std::vector<Aabb> aabbs( totalSize );
    lwca::memcpyDtoH( aabbs.data(), (LWdeviceptr)deviceGatheredAabbs, totalSize * sizeof( cort::Aabb ) );

#if defined( DEBUG ) || defined( DEVELOP )
    // Print if requested
    if( k_printAabbs.get() )
    {
        lprint << "MotionGroup child AABBS for AS ID: " << getScopeID() << '\n';
        printAabbs( totalSize, 1, aabbs );
    }
#endif

    // Revisit the list of nodes and cache the MotionAABBs for each
    // node. This time we relwrsively transform the nodes.  By the end
    // we will have MotionAABBs for all of the children. In some
    // use-cases it may be beneficial to retain this cache across
    // builds, but doing so would add additional complexity. Revisit
    // if such use-cases materialize.
    std::map<int, MotionAabb> all_maabbs;

    // Put the aabbs into a vector and find bounds for buildTimeBegin,
    // buildTimeEnd.  For groups, setup does not hold the motion range
    // of the builder because the child accels were not guaranteed to
    // be built until now.  Traverse the graph and collect it.  Also
    // see ticket OP-1951 (graph property for motion time range),
    // marked as 'won't fix'.
    const unsigned int      childCount = parentGroup->getChildCount();
    std::vector<MotionAabb> child_maabbs( childCount );
    float                   buildTimeBegin = std::numeric_limits<float>::max();
    float                   buildTimeEnd   = -std::numeric_limits<float>::max();

    // Loop over all children, compute/gather MotionAABBs, and find range for
    // buildTimeBegin/buildTimeEnd and copy child MotionAABBs into a linear array.
    for( unsigned int i = 0; i < childCount; ++i )
    {
        GraphNode* child = parentGroup->getChild<GraphNode>( i );
        if( m_context->useRtxDataModel() )
        {
            // skip all static transforms that get folded into the instance transform
            // the builder will apply the instance transform to the input AABB
            while( child->getClass() == RT_OBJECT_TRANSFORM )
            {
                Transform* xform = static_cast<Transform*>( child );
                if( xform->hasMotionKeys() )
                    break;
                child = xform->getChild();
            }
        }
        gatherMotionAabbs( child, all_maabbs, gatherList, aabbs );
        MotionAabb& maabb = all_maabbs.at( child->getScopeID() );
        if( !maabb.isStatic() )
        {
            buildTimeBegin    = std::min( buildTimeBegin, maabb.timeFirstKey() );
            buildTimeEnd      = std::max( buildTimeEnd, maabb.timeLastKey() );
        }
        child_maabbs[i]   = maabb;
    }

    // Transfer (copy) or resample motion aabbs from buildData->entities into fixed
    //  set of regularly distributed aabbs within time span [buildTimeBegin, buildTimeEnd]
    //  that serve as input to the bvh builder.
    unsigned int buildMotionSteps = m_buildParameters.motionSteps;
    unsigned int totalSteps       = buildMotionSteps * childCount;
    aabbs.resize( totalSteps );
    for( size_t i = 0; i < child_maabbs.size(); ++i )
    {
        const MotionAabb& maabb = child_maabbs[i];
        if( !maabb.isValid() || maabb.isStatic() )
        {
            std::fill_n( &aabbs[i * buildMotionSteps], buildMotionSteps, child_maabbs[i].aabbUnion() );  // union includes check for 0 aabbs / ilwalidity
        }
        else if( maabb.keyCount() == buildMotionSteps && maabb.keysAreRegularlyDistributed()
                 && maabb.timeFirstKey() == buildTimeBegin && maabb.timeLastKey() == buildTimeEnd )
        {
            memcpy( &aabbs[i * buildMotionSteps], maabb.aabbs().data(), sizeof( Aabb ) * buildMotionSteps );
        }
        else
        {
            resampleMotionAabbsHost( maabb.aabbs().data(), maabb.keyCount(), maabb.keyTimes().data(), maabb.keyTimes().size(),
                                     &aabbs[i * buildMotionSteps], buildMotionSteps, buildTimeBegin, buildTimeEnd );
        }
    }

    // upload sampled aabbs to device for the build
    for( const auto& perDevice : setup.perDevice )
        lwca::memcpyHtoD( (LWdeviceptr)perDevice.deviceAabbPtr, aabbs.data(), totalSteps * sizeof( cort::Aabb ) );

    // Save the time range
    m_buildParameters.motionTimeBegin = buildTimeBegin;
    m_buildParameters.motionTimeEnd   = buildTimeEnd;

#if defined( DEBUG ) || defined( DEVELOP )
    // Print resampled aabbs
    if( k_printAabbs.get() )
    {
        lprint << "Resampled motion AABBS for AS ID: " << getScopeID() << '\n';
        lprint << "Time range: " << buildTimeBegin << " - " << buildTimeEnd << '\n';
        printAabbs( childCount, buildMotionSteps, aabbs );
    }
#endif
}


void Acceleration::updateTimeRangeForMotionGroup()
{
    if( !hasMotionAabbs() )
        return;

    // Update time range in object record. Must be done after build
    // when object record is available again.
    cort::MotionAccelerationRecord* mar = getObjectRecord<cort::MotionAccelerationRecord>();
    mar->timeBegin                      = m_buildParameters.motionTimeBegin;
    mar->timeEnd                        = m_buildParameters.motionTimeEnd;
}

static MotionAabb mergeMotionAabbs( const std::vector<MotionAabb>& aabbs )
{
    if( aabbs.empty() )
        return MotionAabb();
    else if( aabbs.size() == 1 )
        return aabbs[0];
    else
    {
        // Combine the AABBs of children.  collect and check if all
        // motion aabbs align we do a fast path for the case that all
        // children have regularly distributed keys and fully align
        // with each other (equal begin/end, key count) start with the
        // first child as base loop over remaining children, merge
        // with first if they align perfectly
        std::vector<MotionAabb> toMerge( 1, aabbs[0] );
        for( size_t i = 1; i < aabbs.size(); ++i )
        {
            MotionAabb&       first = toMerge[0];
            const MotionAabb& next  = aabbs[i];
            if( ( first.isStatic() && next.isStatic() ) || first.keysAlign( next ) )
            {
                for( size_t k = 0; k < first.keyCount(); ++k )
                {
                    first.aabb( k ).include( next.aabb( k ) );
                }
            }
            else
                toMerge.push_back( next );
        }
        if( toMerge.size() > 1 )
            return MotionAabb( toMerge );
        else
            return toMerge[0];
    }
}

void Acceleration::gatherMotionAabbs( GraphNode* node,
                                      std::map<int, MotionAabb>& all_maabbs,
                                      const std::map<int, int>&  gatherList,
                                      const std::vector<Aabb>& gatheredAabbs )
{
    if( all_maabbs.count( node->getScopeID() ) != 0 )
        return;

    if( node->getClass() == RT_OBJECT_TRANSFORM )
    {
        Transform* xform = static_cast<Transform*>( node );
        // Visit child
        GraphNode* child = xform->getChild();
        gatherMotionAabbs( child, all_maabbs, gatherList, gatheredAabbs );

        // Transform and insert into the map
        MotionAabb m = xform->computeMotionAabb( all_maabbs[child->getScopeID()] );
        all_maabbs.emplace( xform->getScopeID(), m );
    }
    else if( node->getClass() == RT_OBJECT_SELECTOR )
    {
        // Visit selector children and merge boxes
        Selector*               selector = static_cast<Selector*>( node );
        unsigned int            n        = selector->getChildCount();
        std::vector<MotionAabb> child_maabbs( n );
        for( unsigned int i = 0; i < n; ++i )
        {
            GraphNode* child = selector->getChild( i );
            gatherMotionAabbs( child, all_maabbs, gatherList, gatheredAabbs );
            child_maabbs[i] = all_maabbs[child->getScopeID()];
        }

        // Compute bounds and insert into the map
        MotionAabb m = mergeMotionAabbs( child_maabbs );
        all_maabbs.emplace( selector->getScopeID(), m );
    }
    else
    {
        RT_ASSERT_MSG( node->getClass() == RT_OBJECT_GROUP || node->getClass() == RT_OBJECT_GEOMETRY_GROUP,
                       "Invalid object found in graph" );
        AbstractGroup* group   = static_cast<AbstractGroup*>( node );
        unsigned int   groupid = group->getScopeID();


        // Look up resampled aabbs gathered from the device.
        RT_ASSERT_MSG( gatherList.count( groupid ) != 0, "Group not found in gather list" );
        int gatherBegin = gatherList.at( groupid );

        // Retrieve the motion range and build steps
        Acceleration* accel       = group->getAcceleration();
        unsigned int  motionSteps = accel->m_buildParameters.motionSteps;
        float         timeBegin   = accel->m_buildParameters.motionTimeBegin;
        float         timeEnd     = accel->m_buildParameters.motionTimeEnd;
        RT_ASSERT_MSG( gatherBegin + motionSteps <= gatheredAabbs.size(), "Gathered AABB out of range" );

        // copy the aabbs gathered from the device and colwert them to
        // a motion aabb
        std::vector<Aabb> aabbs( motionSteps );
        for( unsigned int i = 0; i < motionSteps; ++i )
            aabbs[i]        = gatheredAabbs[gatherBegin + i];
        MotionAabb m( timeBegin, timeEnd, aabbs );
        all_maabbs.emplace( groupid, m );
    }
}

const BuildSetupRequest& Acceleration::getBuildSetupRequest() const
{
    return m_buildSetupRequest;
}


//------------------------------------------------------------------------
// LinkedPtr relationship management
//------------------------------------------------------------------------

void Acceleration::detachFromParents()
{
    auto iter = m_linkedPointers.begin();
    while( iter != m_linkedPointers.end() )
    {
        LinkedPtr_Link* parentLink = *iter;

        if( AbstractGroup* parent = getLinkToAccelerationFrom<AbstractGroup>( parentLink ) )
            parent->detachLinkedChild( parentLink );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to Acceleration" );

        iter = m_linkedPointers.begin();
    }
}

void Acceleration::detachLinkedChild( const LinkedPtr_Link* link )
{
    RT_ASSERT_FAIL_MSG( "Acceleration object does not have LinkedPtr children" );
}


//------------------------------------------------------------------------
// Object record management
//------------------------------------------------------------------------

size_t Acceleration::getRecordBaseSize() const
{
    if( hasMotionAabbs() )
        return sizeof( cort::MotionAccelerationRecord );
    else
        return sizeof( cort::AccelerationRecord );
}

void Acceleration::writeRecord() const
{
    if( !recordIsAllocated() )
        return;
    cort::AccelerationRecord* as = getObjectRecord<cort::AccelerationRecord>();
    RT_ASSERT( as != nullptr );
    LexicalScope::writeRecord();

    // Note: remainder of motion record will be written in setupForBuild
}

void Acceleration::notifyParents_offsetDidChange() const
{
    for( auto parentLink : m_linkedPointers )
    {
        if( AbstractGroup* parent = getLinkToAccelerationFrom<AbstractGroup>( parentLink ) )
            parent->childOffsetDidChange( parentLink );
        else
            RT_ASSERT_FAIL_MSG( "Invalid parent link to Acceleration" );
    }
}

//------------------------------------------------------------------------
// Unresolved reference property
//------------------------------------------------------------------------

void Acceleration::sendPropertyDidChange_UnresolvedReference( VariableReferenceID refid, bool added ) const
{
    // Acceleration has no parent, but we need to keep track of the unresolved references
    // for validation.  Note, unlike GlobalScope which has to take into consideration
    // default values, we should not have any default values.
    bool changed = m_unresolvedRemaining.addOrRemoveProperty( refid, added );
    if( changed && added )
    {
        // const cast required because subscribe is non-const
        Acceleration* nonconst_this = const_cast<Acceleration*>( this );
        nonconst_this->subscribeForValidation();
    }
}

//------------------------------------------------------------------------
// Attachment
//------------------------------------------------------------------------

void Acceleration::sendPropertyDidChange_Attachment( bool added ) const
{
    // Acceleration has no children for attachment
}

void Acceleration::attachmentDidChange( bool newAttached )
{
    LexicalScope::attachmentDidChange( newAttached );
    updateDirtyList();
}

//------------------------------------------------------------------------
// RtxUniversalTraversal
//------------------------------------------------------------------------

void Acceleration::rtxUniversalTraversalDidChange()
{
    markDirty( /*internallyDirty=*/true );

    // A different builder may be required for the current traversal method.
    setBuilder( m_builderTypeParam );

    if( m_builder )
    {
        // Even if the builder did not change, switching traversal method changes the traversable handle encoding
        // from raw pointer to decorated handle.
        for( Device* dev : m_context->getDeviceManager()->activeDevices() )
        {
            const unsigned int allDeviceIndex = dev->allDeviceListIndex();
            traversableHandleDidChange( m_builder->getTraversableHandle( allDeviceIndex ), allDeviceIndex );
        }
    }
}

//------------------------------------------------------------------------
// Direct Caller
//------------------------------------------------------------------------

void Acceleration::sendPropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added ) const
{
    // Acceleration should not have any direct caller changes
    RT_ASSERT_FAIL_MSG( "Acceleration direct caller changed" );
}


//------------------------------------------------------------------------
// Acceleration height property
//------------------------------------------------------------------------

void Acceleration::receivePropertyDidChange_AccelerationHeight( int height, bool added )
{
    // No further action is necessary. Used only during build.
    m_accelerationHeight.addOrRemoveProperty( height, added );
}

//------------------------------------------------------------------------
// HasMotionAabbs property
//------------------------------------------------------------------------

void Acceleration::receivePropertyDidChange_HasMotionAabbs( bool added )
{
    bool changed = m_hasMotionAabbs.addOrRemoveProperty( added );
    if( changed )
    {
        reallocateRecord();
        updateVisitProgram();
    }
}


//------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------
static bool isNameForBvhBuilder( const std::string& name )
{
    return name == "Bvh"             // Bvh specified explicitly
           || name == "BvhCompact";  // remap BvhCompact -> Bvh
}

static bool isNameForTrBvhBuilder( const std::string& name )
{
    return name == "Trbvh"              // Trbvh specified explicitly
           || name == "MedianBvh"       // remap MedianBvh -> Trbvh
           || name == "TriangleKdTree"  // remap TriangleKdTree -> Trbvh
           || name == "Lbvh";           // remap Lbvh -> Trbvh
}

std::unique_ptr<Builder> Acceleration::createBuilder( const std::string& name )
{
    RT_ASSERT_MSG( m_leafKind == RT_OBJECT_GROUP || m_leafKind == RT_OBJECT_GEOMETRY_GROUP,
                   "Leafkind is not yet determined" );

    m_enableTriangleBuilds = k_enableTriangleBuilds.get();
    // For backward compatibility with old builders we need to disable triangle builds
    if( name == "Lbvh" )
        m_enableTriangleBuilds = false;
    // sarine_diamond trace fails, because it set the vertex_buffer incorrectly.  Set this
    // for both cases that use Bvh.
    if( name == "Bvh" || name == "BvhCompact" )
        m_enableTriangleBuilds = false;

    const bool isGeometryGroup = ( m_leafKind == RT_OBJECT_GEOMETRY_GROUP );

    RtcBvhAccelType accelType = RTC_BVH_ACCEL_TYPE_BVH2;

    if( name == "NoAccel" )
        accelType = RTC_BVH_ACCEL_TYPE_NOACCEL;
    else if( name == "Bvh8" )
        accelType = RTC_BVH_ACCEL_TYPE_BVH8;
#if LWCFG( GLOBAL_ARCH_TURING ) || LWCFG( GLOBAL_ARCH_AMPERE )
    else if( name == "TTU" )
        accelType = RTC_BVH_ACCEL_TYPE_TTU;
#endif

    return std::unique_ptr<Builder>( new RtcBvh( this, isGeometryGroup, accelType ) );
}

void Acceleration::updateBuilder()
{
    // Wait to create the builder until the leafkind is known
    if( m_leafKind == RT_OBJECT_UNKNOWN )
        return;

    // Try to get a builder instance.
    std::unique_ptr<Builder> builder = createBuilder( m_builderType );
    if( !builder )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid builder type: ", m_builderType );

    // Setup the new instance and add the properties
    m_builder.swap( builder );
    builder.reset();

    for( auto prop : m_properties.getPropertyMap() )
        m_builder->setProperty( prop.first, prop.second );

    // Update the visit program if necessary
    updateVisitProgram();

    // The new builder doesn't have any data, so we need to mark ourselves dirty.
    markDirty();
    subscribeForValidation();
}


void Acceleration::updateVisitProgram()
{
    // If shutting down, do not re-establish the visit program
    if( m_context->shuttingDown() )
        return;

    if( !m_builder )
        return;

    bool     motionTraversal = hasMotionAabbs() && m_requestedMotionSteps > 1;
    Program* visit           = m_builder->getVisitProgram( motionTraversal, m_bakePrimitiveEntities );
    if( m_visitProgram != visit )
    {
        m_visitProgram = visit;

        for( AbstractGroupIterator it = abstractGroupBegin(), end = abstractGroupEnd(); it != end; ++it )
        {
            AbstractGroup* ag = getAbstractGroup( it );
            ag->setVisitProgram( visit );
        }
    }

    Program* bounds = m_builder->getBoundingBoxProgram( motionTraversal );
    if( m_boundingBoxProgram != bounds )
    {
        m_boundingBoxProgram = bounds;

        for( AbstractGroupIterator it = abstractGroupBegin(), end = abstractGroupEnd(); it != end; ++it )
        {
            AbstractGroup* ag = getAbstractGroup( it );
            ag->setBoundingBoxProgram( bounds );
        }
    }

    subscribeForValidation();
}

void Acceleration::serialize( Serializer& serializer ) const
{
    // Start off with API version, in case things become incompatible.
    optix::serialize( serializer, static_cast<unsigned int>( OPTIX_VERSION ) );

    // Insert dummy field for storing the data size.
    uInt64* size_ptr = (uInt64*)serializer.getWriteLocation();
    optix::serialize( serializer, static_cast<uInt64>( 0 ) );

    // Serialize required state.
    optix::serialize( serializer, std::string( m_builder->getName() ) );
    optix::serialize( serializer, m_builder->getVersion() );
    optix::serialize( serializer, getTraverser() );
    optix::serialize( serializer, m_properties );

    // Write size field if we're actually serializing.
    if( size_ptr )
        *size_ptr = (uInt64)serializer.getSize();
}

// Colwenience function to find index/vertex buffers on a GI
static Variable* lookupBufferVariable( const GeometryInstance* ginst, unsigned short token )
{
    Variable* var = nullptr;

    // First, search geometry instance.
    if( !( var = ginst->getVariableByToken( token ) ) )
        // Second, search geometry.
        if( !( var = ginst->getGeometry()->getVariableByToken( token ) ) )
            // Third, search context.
            var = ginst->getContext()->getGlobalScope()->getVariableByToken( token );

    if( var )
    {
        RT_ASSERT( var->getType().isBuffer() );
    }

    return var;
}
