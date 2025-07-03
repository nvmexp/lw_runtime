// Copyright (c) 2017 LWPU Corporation.  All rights reserved.
//
// LWPU Corporation and its licensors retain all intellectual property and proprietary
// rights in and to this software, related documentation and any modifications thereto.
// Any use, reproduction, disclosure or distribution of this software and related
// documentation without an express license agreement from LWPU Corporation is strictly
// prohibited.
//
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
// AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
// INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
// SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
// LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
// BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
// INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES

#include "ForceRtxExelwtionStrategy.h"

#include <prodlib/system/Knobs.h>
#include <srcTests.h>

#include <AS/ASManager.h>
#include <Context/Context.h>
#include <Context/TableManager.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <Memory/MemoryManager.h>
#include <Objects/Group.h>
#include <Objects/Selector.h>
#include <Objects/Transform.h>
#include <Util/ContainerAlgorithm.h>

#include <algorithm>
#include <array>

using namespace optix;
using namespace testing;

static bool isValidTraversableHandleForUtrav( RtcTraversableHandle handle )
{
    return 0U != ( handle & 0x7 );
}

static bool isValidTraversableHandle( RtcTraversableHandle handle )
{
    return 0U != handle;
}

template <std::size_t N>
void setIdentityMatrix12Keys( std::array<float, N>& keys, int numKeys )
{
    assert( static_cast<int>( N ) >= numKeys * 12 );
    for( int k = 0; k < numKeys; ++k )
    {
        for( int r = 0; r < 3; ++r )
        {
            for( int c = 0; c < 4; ++c )
            {
                keys[12 * k + 4 * r + c] = ( r == c ) ? 1.0f : 0.0f;
            }
        }
    }
}

template <std::size_t N>
void setIdentitySRTMatrixKeys( std::array<float, N>& keys, int numKeys )
{
    assert( static_cast<int>( N ) >= numKeys * 16 );
    // clang-format off
    static const std::array<float, 16> identity{
        // S
        1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f,
        // R (quaternion representation)
        0.f, 0.f, 0.f, 1.f,
        // T
        0.f, 0.f, 0.f
    };
    // clang-format on
    for( int k = 0; k < numKeys; ++k )
    {
        algorithm::copy( identity, &keys[k * 16] );
    }
}

static Matrix4x4 fromMatrix4x3( const float* data )
{
    Matrix4x4 result = Matrix4x4::identity();
    std::copy( &data[0], &data[12], result.getData() );
    return result;
}

namespace {

class TraversableDataLock
{
  public:
    TraversableDataLock( GraphNode* node, unsigned int allDeviceIndex )
        : m_node( node )
        , m_allDeviceIndex( allDeviceIndex )
    {
        m_data = m_node->getTraversableDataForTest( m_allDeviceIndex );
    }
    ~TraversableDataLock() { m_node->releaseTraversableDataForTest( m_allDeviceIndex ); }

    GraphNode::TraversableDataForTest m_data;

  private:
    GraphNode*   m_node;
    unsigned int m_allDeviceIndex;
};

struct RTXTraversableHandle : Test
{
    void SetUp() override
    {
        forceRtxExelwtionStrategy();
        m_context = createContext();
        ASSERT_NE( nullptr, m_context );
        ASSERT_EQ( RT_SUCCESS, rtContextSetRayGenerationProgram( m_context_api, 0U,
                                                                 createProgramByName( "SBTIndexEmptyRayGen" ) ) );
        m_context->getDeviceManager()->enableActiveDevices();
        m_group = createGroup( true );
        m_accel = createAcceleration();
        m_group->setAcceleration( m_accel );
        m_geomGroup    = createGeometryGroup();
        m_geomInstance = createGeometryInstance();
        m_deviceIndex  = m_context->getDeviceManager()->primaryLWDADevice()->activeDeviceListIndex();
    }

    void TearDown() override { rtContextDestroy( m_context_api ); }

    Context* createContext()
    {
        if( rtContextCreate( &m_context_api ) != RT_SUCCESS )
        {
            return nullptr;
        }
        Context* context = reinterpret_cast<Context*>( m_context_api );
        context->setRayTypeCount( 1 );
        context->setEntryPointCount( 1 );
        rtContextDeclareVariable( m_context_api, "dummy", &m_dummyVar_api );
        return context;
    }

    RTprogram createProgramByName( const char* programName )
    {
        RTprogram program = nullptr;
        rtProgramCreateFromPTXFile( m_context_api, m_ptxPath.c_str(), programName, &program );
        return program;
    }

    RTprogram createProgramByName( const char* fileName, const char* programName )
    {
        RTprogram program = nullptr;
        rtProgramCreateFromPTXFile( m_context_api, ptxPath( "test_Objects", fileName ).c_str(), programName, &program );
        return program;
    }

    Group* createGroup( bool attachVariable )
    {
        RTgroup group_api;
        rtGroupCreate( m_context_api, &group_api );
        if( attachVariable )
        {
            rtVariableSetObject( m_dummyVar_api, group_api );
        }
        return reinterpret_cast<Group*>( group_api );
    }

    Transform* createTransform()
    {
        RTtransform transform_api;
        rtTransformCreate( m_context_api, &transform_api );
        return reinterpret_cast<Transform*>( transform_api );
    }

    Transform* createTranslation( float x = 1.0f, float y = 1.0f, float z = 1.0f )
    {
        Transform*   transform = createTransform();
        Matrix4x4    m         = Matrix4x4().translate( make_float3( x, y, z ) );
        const float* data      = m.getData();
        transform->setMatrix( data, false );
        return transform;
    }

    Transform* createMatrixMotionTransform()
    {
        Transform* transform = createTransform();
        std::array<float, 12 * 2> keys{};
        setIdentityMatrix12Keys( keys, 2 );
        transform->setMotionRange( 0.0f, 1.0f );
        transform->setKeys( 2, RT_MOTIONKEYTYPE_MATRIX_FLOAT12, &keys[0] );
        return transform;
    }

    Acceleration* createAcceleration()
    {
        RTacceleration acceleration_api;
        rtAccelerationCreate( m_context_api, &acceleration_api );
        rtAccelerationSetBuilder( acceleration_api, "Trbvh" );
        return reinterpret_cast<Acceleration*>( acceleration_api );
    }

    GeometryGroup* createGeometryGroup()
    {
        RTgeometrygroup geomGroup_api;
        rtGeometryGroupCreate( m_context_api, &geomGroup_api );
        GeometryGroup* geomGroup = reinterpret_cast<GeometryGroup*>( geomGroup_api );
        geomGroup->setAcceleration( createAcceleration() );
        return geomGroup;
    }

    GeometryInstance* createGeometryInstance()
    {
        RTgeometryinstance geomInstance_api;
        rtGeometryInstanceCreate( m_context_api, &geomInstance_api );
        RTgeometry geom_api;
        rtGeometryCreate( m_context_api, &geom_api );
        rtGeometrySetPrimitiveCount( geom_api, 0U );
        rtGeometrySetBoundingBoxProgram( geom_api, createProgramByName( "SBTIndexEmptyBounds" ) );
        rtGeometrySetIntersectionProgram( geom_api, createProgramByName( "SBTIndexEmptyIntersect" ) );
        rtGeometryInstanceSetGeometry( geomInstance_api, geom_api );
        return reinterpret_cast<GeometryInstance*>( geomInstance_api );
    }

    Variable* declareVariable( const char* name )
    {
        RTvariable var_api;
        rtContextDeclareVariable( m_context_api, name, &var_api );
        return reinterpret_cast<Variable*>( var_api );
    }

    Selector* createSelector()
    {
        RTselector selector_api;
        rtSelectorCreate( m_context_api, &selector_api );
        rtSelectorSetVisitProgram( selector_api,
                                   createProgramByName( "GroupInstanceProperties.lw", "SelectorEmptyVisit" ) );
        return reinterpret_cast<Selector*>( selector_api );
    }

    void forceTraversableHandleGeneration()
    {
        m_context->getASManager()->buildAccelerationStructures();
    }

    // attaching a variable to a node puts an entry in the TraversabelHeader table
    bool traversableMatchesTraversableHeaderEntry( GraphNode* node )
    {
        TraversableDataLock data( node, m_deviceIndex );
        return traversableMatchesTraversableHeaderEntry( data, node );
    }

    bool traversableMatchesTraversableHeaderEntry( const TraversableDataLock& data, GraphNode* node )
    {
        return node->getTraversableHandle( m_deviceIndex )
               == m_context->getTableManager()->getTraversableHandleForTest( data.m_data.m_traversableId, m_deviceIndex );
    }

    RTcontext         m_context_api  = nullptr;
    Context*          m_context      = nullptr;
    std::string       m_ptxPath      = ptxPath( "test_Objects", "SBTIndex.lw" );
    Acceleration*     m_accel        = nullptr;
    Group*            m_group        = nullptr;
    RTvariable        m_dummyVar_api = nullptr;
    GeometryGroup*    m_geomGroup    = nullptr;
    GeometryInstance* m_geomInstance = nullptr;
    unsigned int      m_deviceIndex  = 0U;
};

struct RTXTraversableGraphProperties : RTXTraversableHandle
{
};

struct RTXTraversableHandleWithUTrav : RTXTraversableHandle
{
    RTXTraversableHandleWithUTrav()
        : m_knobES( "rtx.traversalOverride", std::string( "Utrav" ) )
    {
    }
    ScopedKnobSetter m_knobES;
};

}  // namespace

TEST_F_DEV( RTXTraversableHandleWithUTrav, updates_group_when_attaching_geom_group_to_transform )
{
    Transform* transform = createTranslation( 10.0f, 12.0f, 14.0f );
    m_group->setChildCount( 1 );
    m_group->setChild( 0, transform );
    forceTraversableHandleGeneration();
    InstanceDescriptorHost::DeviceDependent desc = m_group->getInstanceDescriptorDeviceDependent( 0, m_deviceIndex );
    ASSERT_FALSE( isValidTraversableHandleForUtrav( desc.accelOrTraversableHandle ) );
    ASSERT_FALSE( isValidTraversableHandleForUtrav( m_geomGroup->getTraversableHandle( m_deviceIndex ) ) );

    transform->setChild( m_geomGroup );
    forceTraversableHandleGeneration();

    desc = m_group->getInstanceDescriptorDeviceDependent( 0, m_deviceIndex );
    ASSERT_TRUE( isValidTraversableHandleForUtrav( desc.accelOrTraversableHandle ) );
}

TEST_F_DEV( RTXTraversableHandleWithUTrav, updates_group_when_attaching_transform_to_group )
{
    m_group->setChildCount( 1 );
    Transform* transform = createTranslation( 10.0f, 12.0f, 14.0f );
    transform->setChild( m_geomGroup );
    forceTraversableHandleGeneration();
    InstanceDescriptorHost::DeviceDependent desc = m_group->getInstanceDescriptorDeviceDependent( 0, m_deviceIndex );
    // If the child of m_group hasn't been set, accelOrTraversableHandle is not initialized, don't check its value.
    // ASSERT_FALSE( isValidTraversableHandleForUtrav( desc.accelOrTraversableHandle ) );
    ASSERT_FALSE( isValidTraversableHandleForUtrav( m_geomGroup->getTraversableHandle( m_deviceIndex ) ) );

    m_group->setChild( 0, transform );
    forceTraversableHandleGeneration();

    desc = m_group->getInstanceDescriptorDeviceDependent( 0, m_deviceIndex );
    ASSERT_TRUE( isValidTraversableHandleForUtrav( desc.accelOrTraversableHandle ) );
}

TEST_F_DEV( RTXTraversableHandleWithUTrav, updates_group_when_geom_group_accel_changes )
{
    Transform* transform = createTranslation( 10.0f, 12.0f, 14.0f );
    m_group->setChildCount( 1 );
    m_group->setChild( 0, transform );
    transform->setChild( m_geomGroup );
    forceTraversableHandleGeneration();
    InstanceDescriptorHost::DeviceDependent desc = m_group->getInstanceDescriptorDeviceDependent( 0, m_deviceIndex );
    const RtcTraversableHandle oldHandle = desc.accelOrTraversableHandle;
    ASSERT_TRUE( isValidTraversableHandleForUtrav( oldHandle ) );
    Acceleration* newAccel = createAcceleration();

    m_geomGroup->setAcceleration( newAccel );
    forceTraversableHandleGeneration();

    desc = m_group->getInstanceDescriptorDeviceDependent( 0, m_deviceIndex );
    ASSERT_TRUE( isValidTraversableHandleForUtrav( desc.accelOrTraversableHandle ) );
}

TEST_F_DEV( RTXTraversableHandleWithUTrav, updates_group_through_multiple_transforms_when_geom_group_accel_changes )
{
    Transform* transform = createTranslation( 10.0f, 12.0f, 14.0f );
    m_group->setChildCount( 1 );
    m_group->setChild( 0, transform );
    Transform* transform2 = createTranslation( 1.0f, 1.0f, 1.0f );
    transform->setChild( transform2 );
    transform2->setChild( m_geomGroup );
    forceTraversableHandleGeneration();
    InstanceDescriptorHost::DeviceDependent desc = m_group->getInstanceDescriptorDeviceDependent( 0, m_deviceIndex );
    const RtcTraversableHandle oldHandle = desc.accelOrTraversableHandle;
    ASSERT_TRUE( isValidTraversableHandleForUtrav( oldHandle ) );
    Acceleration* newAccel = createAcceleration();

    m_geomGroup->setAcceleration( newAccel );
    forceTraversableHandleGeneration();

    desc = m_group->getInstanceDescriptorDeviceDependent( 0, m_deviceIndex );
    ASSERT_TRUE( isValidTraversableHandleForUtrav( desc.accelOrTraversableHandle ) );
}

TEST_F_DEV( RTXTraversableHandle, variable_creates_traversable_on_geom_group )
{
    Variable* var = declareVariable( "var" );
    forceTraversableHandleGeneration();
    ASSERT_FALSE( isValidTraversableHandle( m_geomGroup->getTraversableHandle( m_deviceIndex ) ) );

    var->setGraphNode( m_geomGroup );
    forceTraversableHandleGeneration();

    EXPECT_TRUE( isValidTraversableHandle( m_geomGroup->getTraversableHandle( m_deviceIndex ) ) );
    EXPECT_TRUE( traversableMatchesTraversableHeaderEntry( m_geomGroup ) );
}

// This test, if not run with UTrav enabled, would fail in
//   EXPECT_TRUE( isValidTraversableHandleForUtrav( newHandle ) );
// RtcTravBottomLevelInstance is only set up when universal traversal is enabled. None of the other
// traversers support direct traversal of a RtcTravBottomLevelInstance so its setup is skipped.
// OptiX decides to use universal traversal automagically on launch, based on the scene graph.
// This code would not be triggered in a unit test, hence we use the special RTXTraversableHandleWithUTrav
// to ensure that UTrav and the setup of RtcTravBottomLevelInstance is enabled explicitly.
TEST_F_DEV( RTXTraversableHandleWithUTrav, updates_geom_group_direct_traversable_when_accel_changes )
{
    Variable* var = declareVariable( "var" );
    var->setGraphNode( m_geomGroup );
    forceTraversableHandleGeneration();
    const RtcTraversableHandle oldHandle = m_geomGroup->getTraversableHandle( m_deviceIndex );
    ASSERT_TRUE( isValidTraversableHandleForUtrav( oldHandle ) );
    Acceleration* newAccel = createAcceleration();

    m_geomGroup->setAcceleration( newAccel );
    forceTraversableHandleGeneration();

    TraversableDataLock        ggData( m_geomGroup, m_deviceIndex );
    const RtcTraversableHandle newHandle = ggData.m_data.m_bottomLevelInstance->accel;
    EXPECT_TRUE( isValidTraversableHandleForUtrav( newHandle ) );
    EXPECT_NE( oldHandle, newHandle );
    EXPECT_TRUE( traversableMatchesTraversableHeaderEntry( ggData, m_geomGroup ) );
}

TEST_F_DEV( RTXTraversableHandle, variable_creates_traversable_on_transform )
{
    Transform* transform = createTranslation();
    Variable*  var       = declareVariable( "var" );
    ASSERT_FALSE( isValidTraversableHandle( transform->getTraversableHandle( m_deviceIndex ) ) );

    var->setGraphNode( transform );
    forceTraversableHandleGeneration();

    EXPECT_TRUE( isValidTraversableHandle( transform->getTraversableHandle( m_deviceIndex ) ) );
    TraversableDataLock data( transform, m_deviceIndex );
    EXPECT_EQ( RTC_TRAVERSABLE_TYPE_STATIC_TRANSFORM, data.m_data.m_type );
    EXPECT_EQ( sizeof( RtcTravStaticTransform ), data.m_data.m_size );
    const RtcStaticTransform* xformData = data.m_data.m_staticTransform;
    EXPECT_FALSE( isValidTraversableHandle( xformData->child ) );
    EXPECT_EQ( Matrix4x4::translate( make_float3( 1.0f, 1.0f, 1.0f ) ), fromMatrix4x3( xformData->transform ) );
    EXPECT_EQ( Matrix4x4::translate( make_float3( -1.0f, -1.0f, -1.0f ) ), fromMatrix4x3( xformData->ilwTransform ) );
    EXPECT_TRUE( traversableMatchesTraversableHeaderEntry( data, transform ) );
}

TEST_F_DEV( RTXTraversableHandle, selector_creates_traversable_on_transform )
{
    Selector* selector = createSelector();
    selector->setChildCount( 1 );
    Transform* xform = createTransform();
    ASSERT_FALSE( isValidTraversableHandle( xform->getTraversableHandle( m_deviceIndex ) ) );

    selector->setChild( 0, xform );
    forceTraversableHandleGeneration();

    const RtcTraversableHandle newHandle = xform->getTraversableHandle( m_deviceIndex );
    EXPECT_TRUE( isValidTraversableHandle( newHandle ) );
    TraversableDataLock selData( selector, m_deviceIndex );
    EXPECT_EQ( newHandle, selData.m_data.m_selector->children[0] );
}

TEST_F_DEV( RTXTraversableHandle, motion_transform_has_traversable_handle )
{
    Transform* transform = createTranslation();
    ASSERT_FALSE( isValidTraversableHandle( transform->getTraversableHandle( m_deviceIndex ) ) );
    std::array<float, 12 * 2> keys{};
    setIdentityMatrix12Keys( keys, 2 );

    transform->setMotionRange( 0.0f, 1.0f );
    transform->setKeys( 2, RT_MOTIONKEYTYPE_MATRIX_FLOAT12, &keys[0] );
    forceTraversableHandleGeneration();

    ASSERT_TRUE( isValidTraversableHandle( transform->getTraversableHandle( m_deviceIndex ) ) );
    TraversableDataLock data( transform, m_deviceIndex );
    ASSERT_EQ( RTC_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM, data.m_data.m_type );
    ASSERT_EQ( sizeof( RtcMatrixMotionTransform ), data.m_data.m_size );
    const RtcMatrixMotionTransform* xformData = data.m_data.m_matrixMotionTransform;
    ASSERT_FALSE( isValidTraversableHandle( xformData->child ) );
    ASSERT_EQ( 2U, xformData->numKeys );
    ASSERT_EQ( 0x0U, xformData->flags );
    ASSERT_EQ( 0.0f, xformData->timeBegin );
    ASSERT_EQ( 1.0f, xformData->timeEnd );
    ASSERT_EQ( Matrix4x4::identity(), fromMatrix4x3( xformData->transform[0] ) );
    ASSERT_EQ( Matrix4x4::identity(), fromMatrix4x3( xformData->transform[1] ) );
}

TEST_F_DEV( RTXTraversableHandle, motion_transform_to_static_transform_removes_traversable_handle )
{
    Transform*            transform = createTranslation();
    constexpr std::size_t numKeys   = 2;
    constexpr std::size_t numFloats = numKeys * 12;
    std::array<float, numFloats> keys{};
    transform->setMotionRange( 0.0f, 1.0f );
    setIdentityMatrix12Keys<numFloats>( keys, numKeys );
    transform->setKeys( numKeys, RT_MOTIONKEYTYPE_MATRIX_FLOAT12, &keys[0] );
    forceTraversableHandleGeneration();
    ASSERT_TRUE( isValidTraversableHandle( transform->getTraversableHandle( m_deviceIndex ) ) );

    transform->setKeys( 1, RT_MOTIONKEYTYPE_MATRIX_FLOAT12, &keys[0] );
    forceTraversableHandleGeneration();

    ASSERT_FALSE( isValidTraversableHandle( transform->getTraversableHandle( m_deviceIndex ) ) );
}

TEST_F_DEV( RTXTraversableHandle, motion_transform_type_changes_traversable_handle )
{
    Transform*            transform = createTranslation();
    constexpr std::size_t numKeys   = 2;
    constexpr std::size_t numFloats = numKeys * 16;
    transform->setMotionRange( 0.0f, 1.0f );
    std::array<float, numFloats> keys{};
    setIdentityMatrix12Keys<numFloats>( keys, numKeys );
    transform->setKeys( numKeys, RT_MOTIONKEYTYPE_MATRIX_FLOAT12, &keys[0] );
    forceTraversableHandleGeneration();
    const RtcTraversableHandle oldHandle = transform->getTraversableHandle( m_deviceIndex );
    ASSERT_TRUE( isValidTraversableHandle( oldHandle ) );

    setIdentitySRTMatrixKeys( keys, numKeys );
    transform->setKeys( numKeys, RT_MOTIONKEYTYPE_SRT_FLOAT16, &keys[0] );
    forceTraversableHandleGeneration();

    const RtcTraversableHandle newHandle = transform->getTraversableHandle( m_deviceIndex );
    // No handle without Utrav
    ASSERT_FALSE( isValidTraversableHandle( newHandle ) );
    // TODO: uncomment this assert when traversable type bits are incorporated into RtcTraversableHandle
    ASSERT_NE( oldHandle, newHandle );
    TraversableDataLock data( transform, m_deviceIndex );
    ASSERT_EQ( RTC_TRAVERSABLE_TYPE_SRT_MOTION_TRANSFORM, data.m_data.m_type );
    ASSERT_EQ( sizeof( RtcSRTMotionTransform ), data.m_data.m_size );
    const RtcSRTMotionTransform* xformData = data.m_data.m_srtMotionTransform;
    ASSERT_FALSE( isValidTraversableHandle( xformData->child ) );
    ASSERT_EQ( 2U, xformData->numKeys );
    ASSERT_EQ( 0x0U, xformData->flags );
    ASSERT_EQ( 0.0f, xformData->timeBegin );
    ASSERT_EQ( 1.0f, xformData->timeEnd );
    std::array<float, 16> identityQuaternion{};
    setIdentitySRTMatrixKeys( identityQuaternion, 1 );
    ASSERT_THAT( identityQuaternion, ElementsAreArray( xformData->quaternion[0], 16 ) );
    ASSERT_THAT( identityQuaternion, ElementsAreArray( xformData->quaternion[1], 16 ) );
}

TEST_F_DEV( RTXTraversableHandle, transform_direct_traversable_changed_notifies_parent_group_and_instance_handle_remains_unchanged )
{
    Variable*  var       = declareVariable( "var" );
    Transform* transform = createTranslation();
    m_group->setChildCount( 1 );
    m_group->setChild( 0, transform );
    InstanceDescriptorHost::DeviceDependent desc = m_group->getInstanceDescriptorDeviceDependent( 0, m_deviceIndex );
    ASSERT_FALSE( isValidTraversableHandle( desc.accelOrTraversableHandle ) );

    var->setGraphNode( transform );
    forceTraversableHandleGeneration();

    desc = m_group->getInstanceDescriptorDeviceDependent( 0, m_deviceIndex );
    EXPECT_FALSE( isValidTraversableHandle( desc.accelOrTraversableHandle ) );
    EXPECT_TRUE( traversableMatchesTraversableHeaderEntry( transform ) );
}

TEST_F( RTXTraversableHandle, transform_direct_traversable_holds_direct_traversable_of_child_geometry_group )
{
    // V1 -> T1 -> T2 -> GG
    //       V2 ---/
    Transform* xform1 = createTranslation();
    m_group->setChildCount( 1 );
    m_group->setChild( 0, xform1 );
    Variable* var1 = declareVariable( "var1" );
    var1->setGraphNode( xform1 );
    Transform* xform2 = createTranslation();
    xform2->setChild( m_geomGroup );
    Variable* var2 = declareVariable( "var2" );
    var2->setGraphNode( xform2 );

    xform1->setChild( xform2 );
    forceTraversableHandleGeneration();

    EXPECT_TRUE( isValidTraversableHandle( xform1->getTraversableHandle( m_deviceIndex ) ) );
    EXPECT_TRUE( isValidTraversableHandle( xform2->getTraversableHandle( m_deviceIndex ) ) );
    const RtcTraversableHandle ggTravHandle = m_geomGroup->getTraversableHandle( m_deviceIndex );
    TraversableDataLock        xform1Data( xform1, m_deviceIndex );
    EXPECT_EQ( ggTravHandle, xform1Data.m_data.m_staticTransform->child );
    EXPECT_TRUE( traversableMatchesTraversableHeaderEntry( xform1Data, xform1 ) );
    TraversableDataLock xform2Data( xform2, m_deviceIndex );
    EXPECT_EQ( ggTravHandle, xform2Data.m_data.m_staticTransform->child );
    EXPECT_TRUE( traversableMatchesTraversableHeaderEntry( xform2Data, xform2 ) );
}

TEST_F( RTXTraversableHandleWithUTrav, group_notifies_parent_transform_of_direct_traversable_on_set_child )
{
    Transform* xform = createTranslation();
    Variable*  var   = declareVariable( "var" );
    var->setGraphNode( xform );
    forceTraversableHandleGeneration();
    const RtcTraversableHandle groupHandle = m_group->getTraversableHandle( m_deviceIndex );
    ASSERT_TRUE( isValidTraversableHandleForUtrav( groupHandle ) );

    xform->setChild( m_group );

    TraversableDataLock xformData( xform, m_deviceIndex );
    EXPECT_EQ( groupHandle, xformData.m_data.m_staticTransform->child );
    EXPECT_TRUE( traversableMatchesTraversableHeaderEntry( xformData, xform ) );
}

TEST_F( RTXTraversableHandleWithUTrav, geometry_group_notifies_parent_group_of_traversables_on_set_child )
{
    m_group->setChildCount( 1 );
    Variable* var = declareVariable( "var" );
    var->setGraphNode( m_geomGroup );
    forceTraversableHandleGeneration();
    InstanceDescriptorHost::DeviceDependent desc = m_group->getInstanceDescriptorDeviceDependent( 0, m_deviceIndex );
    // If the child of m_group hasn't been set, accelOrTraversableHandle is not initialized, don't check its value.
    // ASSERT_FALSE( isValidTraversableHandleForUtrav( desc.accelOrTraversableHandle ) );
    ASSERT_TRUE( isValidTraversableHandleForUtrav( m_geomGroup->getTraversableHandle( m_deviceIndex ) ) );

    m_group->setChild( 0, m_geomGroup );

    desc = m_group->getInstanceDescriptorDeviceDependent( 0, m_deviceIndex );
    EXPECT_TRUE( isValidTraversableHandleForUtrav( desc.accelOrTraversableHandle ) );
    EXPECT_TRUE( traversableMatchesTraversableHeaderEntry( m_geomGroup ) );
}

TEST_F( RTXTraversableHandleWithUTrav, geometry_group_notifies_parent_transform_of_traversables_on_set_child )
{
    m_group->setChildCount( 1 );
    Transform* xform = createTranslation();
    m_group->setChild( 0, xform );
    Variable* var = declareVariable( "var" );
    var->setGraphNode( m_geomGroup );
    forceTraversableHandleGeneration();
    InstanceDescriptorHost::DeviceDependent desc = m_group->getInstanceDescriptorDeviceDependent( 0, m_deviceIndex );
    ASSERT_FALSE( isValidTraversableHandleForUtrav( desc.accelOrTraversableHandle ) );
    ASSERT_TRUE( isValidTraversableHandleForUtrav( m_geomGroup->getTraversableHandle( m_deviceIndex ) ) );

    xform->setChild( m_geomGroup );

    desc = m_group->getInstanceDescriptorDeviceDependent( 0, m_deviceIndex );
    EXPECT_TRUE( isValidTraversableHandleForUtrav( desc.accelOrTraversableHandle ) );
    EXPECT_TRUE( traversableMatchesTraversableHeaderEntry( m_geomGroup ) );
}

TEST_F( RTXTraversableHandle, transform_collapses_matrices_for_direct_traversable )
{
    // G -> T[v] -> T[v] -> GG
    Transform* xform1 = createTranslation();
    Variable*  var1   = declareVariable( "var1" );
    var1->setGraphNode( xform1 );
    Transform* xform2 = createTranslation();
    xform1->setChild( xform2 );
    Variable* var2 = declareVariable( "var2" );
    var2->setGraphNode( xform2 );
    m_group->setChildCount( 1 );
    m_group->setChild( 0, xform1 );

    xform2->setChild( m_geomGroup );
    forceTraversableHandleGeneration();

    TraversableDataLock xform1Data( xform1, m_deviceIndex );
    EXPECT_EQ( RTC_TRAVERSABLE_TYPE_STATIC_TRANSFORM, xform1Data.m_data.m_type );
    EXPECT_EQ( sizeof( RtcTravStaticTransform ), xform1Data.m_data.m_size );
    EXPECT_TRUE( isValidTraversableHandle( xform1Data.m_data.m_staticTransform->child ) );
    EXPECT_EQ( m_geomGroup->getTraversableHandle( m_deviceIndex ), xform1Data.m_data.m_staticTransform->child );
    EXPECT_EQ( Matrix4x4::translate( make_float3( 2.0f, 2.0f, 2.0f ) ),
               fromMatrix4x3( xform1Data.m_data.m_staticTransform->transform ) );
    EXPECT_EQ( Matrix4x4::translate( make_float3( -2.0f, -2.0f, -2.0f ) ),
               fromMatrix4x3( xform1Data.m_data.m_staticTransform->ilwTransform ) );
    EXPECT_TRUE( traversableMatchesTraversableHeaderEntry( xform1Data, xform1 ) );
    EXPECT_TRUE( traversableMatchesTraversableHeaderEntry( xform2 ) );
}

TEST_F( RTXTraversableHandle, variable_on_transform_gets_direct_traversable_from_motion_transform )
{
    Transform* xform1 = createTranslation();
    Variable*  var    = declareVariable( "var" );
    var->setGraphNode( xform1 );
    Transform* xform2 = createMatrixMotionTransform();

    xform1->setChild( xform2 );
    forceTraversableHandleGeneration();

    EXPECT_TRUE( isValidTraversableHandle( xform1->getTraversableHandle( m_deviceIndex ) ) );
    TraversableDataLock xform1Data( xform1, m_deviceIndex );
    EXPECT_TRUE( isValidTraversableHandle( xform1Data.m_data.m_staticTransform->child ) );
    EXPECT_EQ( xform2->getTraversableHandle( m_deviceIndex ), xform1Data.m_data.m_staticTransform->child );
    EXPECT_TRUE( traversableMatchesTraversableHeaderEntry( xform1Data, xform1 ) );
}

TEST_F( RTXTraversableHandle, motion_transform_requires_direct_traversable_on_geom_group )
{
    Transform* xform = createMatrixMotionTransform();

    xform->setChild( m_geomGroup );
    forceTraversableHandleGeneration();

    const RtcTraversableHandle geomGroupHandle = m_geomGroup->getTraversableHandle( m_deviceIndex );
    EXPECT_TRUE( isValidTraversableHandle( geomGroupHandle ) );
    TraversableDataLock xformData( xform, m_deviceIndex );
    EXPECT_TRUE( isValidTraversableHandle( xformData.m_data.m_matrixMotionTransform->child ) );
    EXPECT_EQ( geomGroupHandle, xformData.m_data.m_matrixMotionTransform->child );
}

TEST_F( RTXTraversableHandle, variable_on_transform_requires_direct_traversable_on_geom_group )
{
    Transform* xform = createTranslation();
    Variable*  var   = declareVariable( "var" );
    var->setGraphNode( xform );

    xform->setChild( m_geomGroup );
    forceTraversableHandleGeneration();

    const RtcTraversableHandle geomGroupHandle = m_geomGroup->getTraversableHandle( m_deviceIndex );
    EXPECT_TRUE( isValidTraversableHandle( geomGroupHandle ) );
    TraversableDataLock xformData( xform, m_deviceIndex );
    EXPECT_TRUE( isValidTraversableHandle( xformData.m_data.m_staticTransform->child ) );
    EXPECT_EQ( geomGroupHandle, xformData.m_data.m_staticTransform->child );
    EXPECT_TRUE( traversableMatchesTraversableHeaderEntry( xformData, xform ) );
}

TEST_F( RTXTraversableHandle, variable_on_transform_chain_requires_direct_traversable_on_geom_group )
{
    Transform* xform1 = createTranslation();
    Variable*  var    = declareVariable( "var" );
    var->setGraphNode( xform1 );
    Transform* xform2 = createTranslation();
    xform1->setChild( xform2 );

    xform2->setChild( m_geomGroup );
    forceTraversableHandleGeneration();

    const RtcTraversableHandle geomGroupHandle = m_geomGroup->getTraversableHandle( m_deviceIndex );
    EXPECT_TRUE( isValidTraversableHandle( geomGroupHandle ) );
    TraversableDataLock xform1Data( xform1, m_deviceIndex );
    EXPECT_TRUE( isValidTraversableHandle( xform1Data.m_data.m_staticTransform->child ) );
    EXPECT_EQ( geomGroupHandle, xform1Data.m_data.m_staticTransform->child );
    EXPECT_FALSE( isValidTraversableHandle( xform2->getTraversableHandle( m_deviceIndex ) ) );
    EXPECT_TRUE( traversableMatchesTraversableHeaderEntry( xform1Data, xform1 ) );
}

TEST_F( RTXTraversableHandleWithUTrav, variable_on_transform_gets_group_direct_traversable )
{
    Transform* xform = createTransform();
    Variable*  var   = declareVariable( "var" );
    var->setGraphNode( xform );

    xform->setChild( m_group );
    forceTraversableHandleGeneration();

    const RtcTraversableHandle groupHandle = m_group->getTraversableHandle( m_deviceIndex );
    EXPECT_TRUE( isValidTraversableHandleForUtrav( groupHandle ) );
    TraversableDataLock xformData( xform, m_deviceIndex );
    EXPECT_TRUE( isValidTraversableHandleForUtrav( xformData.m_data.m_staticTransform->child ) );
    EXPECT_EQ( groupHandle, xformData.m_data.m_staticTransform->child );
    EXPECT_TRUE( traversableMatchesTraversableHeaderEntry( xformData, xform ) );
}

TEST_F( RTXTraversableHandle, motion_transform_gets_transform_traversable )
{
    Transform* xform1 = createMatrixMotionTransform();
    Transform* xform2 = createTransform();

    xform1->setChild( xform2 );
    forceTraversableHandleGeneration();

    TraversableDataLock xform1Data( xform1, m_deviceIndex );
    EXPECT_TRUE( isValidTraversableHandle( xform1Data.m_data.m_matrixMotionTransform->child ) );
    EXPECT_EQ( xform2->getTraversableHandle( m_deviceIndex ), xform1Data.m_data.m_matrixMotionTransform->child );
}

TEST_F( RTXTraversableHandle, variable_has_transform_direct_traversable )
{
    Transform* xform = createTranslation();
    m_group->setChildCount( 1 );
    m_group->setChild( 0, xform );
    Variable* var = declareVariable( "var" );
    xform->setChild( m_geomGroup );

    var->setGraphNode( xform );
    forceTraversableHandleGeneration();

    ASSERT_TRUE( traversableMatchesTraversableHeaderEntry( xform ) );
}

TEST_F( RTXTraversableHandleWithUTrav, group_has_traversable_of_motion_transform )
{
    Transform* xform = createMatrixMotionTransform();
    m_group->setChildCount( 1 );

    m_group->setChild( 0, xform );
    forceTraversableHandleGeneration();

    const RtcTraversableHandle xformHandle = xform->getTraversableHandle( m_deviceIndex );
    ASSERT_TRUE( isValidTraversableHandleForUtrav( xformHandle ) );
    InstanceDescriptorHost::DeviceDependent desc = m_group->getInstanceDescriptorDeviceDependent( 0, m_deviceIndex );
    ASSERT_EQ( xformHandle, desc.accelOrTraversableHandle );
}

TEST_F( RTXTraversableHandleWithUTrav, group_has_traversable_of_group )
{
    Group*        group2 = createGroup( false );
    Acceleration* accel2 = createAcceleration();
    group2->setAcceleration( accel2 );
    m_group->setChildCount( 1 );

    m_group->setChild( 0, group2 );
    forceTraversableHandleGeneration();

    const RtcTraversableHandle group2Handle = group2->getTraversableHandle( m_deviceIndex );
    ASSERT_TRUE( isValidTraversableHandleForUtrav( group2Handle ) );
    InstanceDescriptorHost::DeviceDependent desc = m_group->getInstanceDescriptorDeviceDependent( 0, m_deviceIndex );
    ASSERT_EQ( group2Handle, desc.accelOrTraversableHandle );
}

//
// TODO: OP-2027 handle disconnect case with transform requires traversable graph property
TEST_F( RTXTraversableHandle, geom_group_disconnected_from_variable_transform_has_no_direct_traversable )
{
    Variable*  var   = declareVariable( "var" );
    Transform* xform = createTranslation();
    var->setGraphNode( xform );
    xform->setChild( m_geomGroup );
    forceTraversableHandleGeneration();
    ASSERT_TRUE( isValidTraversableHandle( m_geomGroup->getTraversableHandle( m_deviceIndex ) ) );

    xform->setChild( nullptr );

    EXPECT_FALSE( isValidTraversableHandle( m_geomGroup->getTraversableHandle( m_deviceIndex ) ) );
}

// TODO: OP-2027 handle disconnect case with transform requires traversable graph property
TEST_F( RTXTraversableHandle, geom_group_disconnected_from_motion_transform_has_no_direct_traversable )
{
    Transform* xform = createMatrixMotionTransform();
    xform->setChild( m_geomGroup );
    forceTraversableHandleGeneration();
    ASSERT_TRUE( isValidTraversableHandle( m_geomGroup->getTraversableHandle( m_deviceIndex ) ) );

    xform->setChild( nullptr );

    EXPECT_FALSE( isValidTraversableHandle( m_geomGroup->getTraversableHandle( m_deviceIndex ) ) );
}

TEST_F( RTXTraversableHandle, variable_disconnected_from_transform_removes_traversable_from_geom_group )
{
    Variable*  var   = declareVariable( "var" );
    Transform* xform = createTranslation();
    var->setGraphNode( xform );
    xform->setChild( m_geomGroup );
    forceTraversableHandleGeneration();
    ASSERT_TRUE( isValidTraversableHandle( m_geomGroup->getTraversableHandle( m_deviceIndex ) ) );

    var->setGraphNode( nullptr );

    EXPECT_FALSE( isValidTraversableHandle( m_geomGroup->getTraversableHandle( m_deviceIndex ) ) );
}

TEST_F( RTXTraversableHandle, transform_disconnected_from_motion_transform_has_no_direct_traversable )
{
    Transform* motion = createMatrixMotionTransform();
    Transform* xform  = createTranslation();
    motion->setChild( xform );
    forceTraversableHandleGeneration();
    ASSERT_TRUE( isValidTraversableHandle( xform->getTraversableHandle( m_deviceIndex ) ) );

    motion->setChild( nullptr );

    EXPECT_FALSE( isValidTraversableHandle( xform->getTraversableHandle( m_deviceIndex ) ) );
}

TEST_F( RTXTraversableHandle, transform_disconnected_from_variable_has_no_direct_traversable )
{
    Transform* xform = createTranslation();
    Variable*  var   = declareVariable( "var" );
    var->setGraphNode( xform );
    forceTraversableHandleGeneration();
    ASSERT_TRUE( isValidTraversableHandle( xform->getTraversableHandle( m_deviceIndex ) ) );

    var->setGraphNode( nullptr );

    EXPECT_FALSE( isValidTraversableHandle( xform->getTraversableHandle( m_deviceIndex ) ) );
}

TEST_F( RTXTraversableGraphProperties, var_on_transform_induces_RT_and_TRT )
{
    Transform* xform = createTranslation();
    Variable*  var   = declareVariable( "var" );

    var->setGraphNode( xform );

    EXPECT_EQ( 1, xform->getRequiresTraversableCountForTest() );
    EXPECT_EQ( 1, xform->getTransformRequiresTraversableCountForTest() );
}

TEST_F( RTXTraversableGraphProperties, var_on_transform_child_transform_induces_TRT )
{
    Transform* xform1 = createTranslation();
    Variable*  var    = declareVariable( "var" );
    var->setGraphNode( xform1 );
    Transform* xform2 = createTranslation();

    xform1->setChild( xform2 );

    EXPECT_EQ( 0, xform2->getRequiresTraversableCountForTest() );
    EXPECT_EQ( 1, xform2->getTransformRequiresTraversableCountForTest() );
}

TEST_F( RTXTraversableGraphProperties, var_on_transform_child_geom_group_induces_RT_and_TRT )
{
    Transform* xform = createTranslation();
    Variable*  var   = declareVariable( "var" );
    var->setGraphNode( xform );

    xform->setChild( m_geomGroup );

    EXPECT_EQ( 1, m_geomGroup->getRequiresTraversableCountForTest() );
    EXPECT_EQ( 1, m_geomGroup->getTransformRequiresTraversableCountForTest() );
}

TEST_F( RTXTraversableGraphProperties, disconnect_var_on_transform_removes_RT_and_TRT )
{
    Transform* xform = createTranslation();
    Variable*  var   = declareVariable( "var" );
    var->setGraphNode( xform );
    ASSERT_EQ( 1, xform->getRequiresTraversableCountForTest() );
    ASSERT_EQ( 1, xform->getTransformRequiresTraversableCountForTest() );

    var->setGraphNode( nullptr );

    EXPECT_EQ( 0, xform->getRequiresTraversableCountForTest() );
    EXPECT_EQ( 0, xform->getTransformRequiresTraversableCountForTest() );
}

TEST_F( RTXTraversableGraphProperties, disconnecting_var_on_transform_child_transform_removes_TRT )
{
    Transform* xform1 = createTranslation();
    Variable*  var    = declareVariable( "var" );
    var->setGraphNode( xform1 );
    Transform* xform2 = createTranslation();
    xform1->setChild( xform2 );
    ASSERT_EQ( 0, xform2->getRequiresTraversableCountForTest() );
    ASSERT_EQ( 1, xform2->getTransformRequiresTraversableCountForTest() );

    var->setGraphNode( nullptr );

    EXPECT_EQ( 0, xform2->getRequiresTraversableCountForTest() );
    EXPECT_EQ( 0, xform2->getTransformRequiresTraversableCountForTest() );
}

TEST_F( RTXTraversableGraphProperties, disconnecting_var_on_transform_child_geom_group_removes_RT_and_TRT )
{
    Transform* xform = createTranslation();
    Variable*  var   = declareVariable( "var" );
    var->setGraphNode( xform );
    xform->setChild( m_geomGroup );
    ASSERT_EQ( 1, m_geomGroup->getRequiresTraversableCountForTest() );
    ASSERT_EQ( 1, m_geomGroup->getTransformRequiresTraversableCountForTest() );

    var->setGraphNode( nullptr );

    EXPECT_EQ( 0, m_geomGroup->getRequiresTraversableCountForTest() );
    EXPECT_EQ( 0, m_geomGroup->getTransformRequiresTraversableCountForTest() );
}

TEST_F( RTXTraversableGraphProperties, motion_transform_induces_RT_and_TRT )
{
    Transform* xform = createMatrixMotionTransform();

    EXPECT_EQ( 1, xform->getRequiresTraversableCountForTest() );
    EXPECT_EQ( 1, xform->getTransformRequiresTraversableCountForTest() );
}

TEST_F( RTXTraversableGraphProperties, motion_transform_child_geom_group_induces_RT_and_TRT )
{
    Transform* xform = createMatrixMotionTransform();

    xform->setChild( m_geomGroup );

    EXPECT_EQ( 2, m_geomGroup->getRequiresTraversableCountForTest() );
    EXPECT_EQ( 1, m_geomGroup->getTransformRequiresTraversableCountForTest() );
}

TEST_F( RTXTraversableGraphProperties, motion_to_static_transform_child_geom_group_removes_RT_and_TRT )
{
    Transform* xform = createMatrixMotionTransform();
    xform->setChild( m_geomGroup );
    ASSERT_EQ( 2, m_geomGroup->getRequiresTraversableCountForTest() );
    ASSERT_EQ( 1, m_geomGroup->getTransformRequiresTraversableCountForTest() );

    std::array<float, 12> keys{};
    setIdentityMatrix12Keys<12>( keys, 1 );
    xform->setKeys( 1, RT_MOTIONKEYTYPE_MATRIX_FLOAT12, &keys[0] );

    EXPECT_EQ( 0, m_geomGroup->getRequiresTraversableCountForTest() );
    EXPECT_EQ( 0, m_geomGroup->getTransformRequiresTraversableCountForTest() );
}

TEST_F( RTXTraversableGraphProperties, var_on_transform_child_selector_induces_RT_and_TRT )
{
    Transform* xform = createTranslation();
    Variable*  var   = declareVariable( "var" );
    var->setGraphNode( xform );
    Selector* selector = createSelector();

    xform->setChild( selector );

    EXPECT_EQ( 2, selector->getRequiresTraversableCountForTest() );
    EXPECT_EQ( 1, selector->getTransformRequiresTraversableCountForTest() );
}

TEST_F( RTXTraversableGraphProperties, var_on_transform_child_group_induces_RT_and_TRT )
{
    Transform* xform = createTranslation();
    Variable*  var   = declareVariable( "var" );
    var->setGraphNode( xform );
    Group* group = createGroup( false );

    xform->setChild( group );

    EXPECT_EQ( 2, group->getRequiresTraversableCountForTest() );
    EXPECT_EQ( 1, group->getTransformRequiresTraversableCountForTest() );
}

TEST_F( RTXTraversableGraphProperties, selector_always_requires_traversable )
{
    Selector* selector = createSelector();

    EXPECT_EQ( 1, selector->getRequiresTraversableCountForTest() );
    EXPECT_EQ( 0, selector->getTransformRequiresTraversableCountForTest() );
}

TEST_F( RTXTraversableGraphProperties, group_always_requires_traversable )
{
    Group* group = createGroup( false );

    EXPECT_EQ( 1, group->getRequiresTraversableCountForTest() );
    EXPECT_EQ( 0, group->getTransformRequiresTraversableCountForTest() );
}
