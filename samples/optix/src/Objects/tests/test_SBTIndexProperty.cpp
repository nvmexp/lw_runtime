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

#include <srcTests.h>

#include "Objects/Selector.h"
#include <Context/Context.h>
#include <Device/DeviceManager.h>
#include <Objects/GeometryInstance.h>
#include <Objects/Group.h>
#include <o6/optix.h>
#include <prodlib/system/Knobs.h>

using namespace optix;
using namespace testing;

namespace {

struct RTXSBTInstanceFixture : Test
{
    void SetUp() override
    {
        optix::forceRtxExelwtionStrategy();
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

    Transform* createTranslation( float x, float y, float z )
    {
        Transform*   transform = createTransform();
        Matrix4x4    m         = Matrix4x4().translate( make_float3( x, y, z ) );
        const float* data      = m.getData();
        transform->setMatrix( data, false );
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

    RTprogram createProgramByName( const char* programName )
    {
        RTprogram program = nullptr;
        rtProgramCreateFromPTXFile( m_context_api, m_ptxPath.c_str(), programName, &program );
        return program;
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

    Material* createMaterial()
    {
        RTmaterial material_api;
        rtMaterialCreate( m_context_api, &material_api );
        rtMaterialSetAnyHitProgram( material_api, 0, createProgramByName( "SBTIndexEmptyAnyHit" ) );
        return reinterpret_cast<Material*>( material_api );
    }

    Selector* createSelector()
    {
        RTselector selector_api;
        rtSelectorCreate( m_context_api, &selector_api );
        rtSelectorSetVisitProgram( selector_api, createProgramByName( "SBTIndexEmptyVisit" ) );
        return reinterpret_cast<Selector*>( selector_api );
    }

    void connectGeometryToGeometryGroup( GeometryGroup* geomGroup, GeometryInstance* geomInstance, Material* material )
    {
        geomGroup->setChildCount( 1 );
        geomGroup->setChild( 0, geomInstance );
        geomInstance->setMaterialCount( 1 );
        geomInstance->setMaterial( 0, material );
    }

    std::string       m_ptxPath      = ptxPath( "test_Objects", "SBTIndex.lw" );
    RTcontext         m_context_api  = nullptr;
    Context*          m_context      = nullptr;
    Acceleration*     m_accel        = nullptr;
    Group*            m_group        = nullptr;
    GeometryGroup*    m_geomGroup    = nullptr;
    GeometryInstance* m_geomInstance = nullptr;
    RTvariable        m_dummyVar_api = nullptr;

  private:
};

}  // namespace

TEST_F_DEV( RTXSBTInstanceFixture, propagates_sbt_index_to_top_level_as_on_material_count_change )
{
    Transform* transform = createTranslation( 10.0f, 12.0f, 14.0f );
    m_group->setChildCount( 1 );
    m_group->setChild( 0, transform );
    connectGeometryToGeometryGroup( m_geomGroup, m_geomInstance, createMaterial() );
    transform->setChild( m_geomGroup );
    connectGeometryToGeometryGroup( createGeometryGroup(), createGeometryInstance(), createMaterial() );
    ASSERT_EQ( RT_SUCCESS, rtContextLaunch1D( m_context_api, 0, 4 ) );
    ASSERT_FALSE( m_accel->isDirty() );

    // force gg into sbt reallocation
    m_geomInstance->setMaterialCount( 2 );

    ASSERT_TRUE( m_accel->isDirty() );
}

// TODO: OP-1977 This won't pass in RTX until motion blur is implemented
TEST_F_DEV( RTXSBTInstanceFixture, DISABLED_sbt_index_does_not_propagate_through_motion_transform )
{
    Transform* transform = createTranslation( 10.0f, 12.0f, 14.0f );
    float      keys[2 * 12]{};
    transform->setKeys( 2, RT_MOTIONKEYTYPE_MATRIX_FLOAT12, keys );
    m_group->setChildCount( 1 );
    m_group->setChild( 0, transform );
    connectGeometryToGeometryGroup( m_geomGroup, m_geomInstance, createMaterial() );
    transform->setChild( m_geomGroup );
    connectGeometryToGeometryGroup( createGeometryGroup(), createGeometryInstance(), createMaterial() );
    ASSERT_EQ( RT_SUCCESS, rtContextLaunch1D( m_context_api, 0, 4 ) );
    ASSERT_FALSE( m_accel->isDirty() );

    // force gg into sbt reallocation
    m_geomInstance->setMaterialCount( 2 );

    ASSERT_FALSE( m_accel->isDirty() );
}

// TODO: OP-1931 This won't pass in RTX until selectors are supported
TEST_F_DEV( RTXSBTInstanceFixture, DISABLED_sbt_index_does_not_propagate_through_selector )
{
    Transform* transform = createTranslation( 10.0f, 12.0f, 14.0f );
    m_group->setChildCount( 1 );
    m_group->setChild( 0, transform );
    connectGeometryToGeometryGroup( m_geomGroup, m_geomInstance, createMaterial() );
    Selector* selector = createSelector();
    transform->setChild( selector );
    selector->setChildCount( 1 );
    selector->setChild( 0, m_geomGroup );
    connectGeometryToGeometryGroup( createGeometryGroup(), createGeometryInstance(), createMaterial() );
    ASSERT_EQ( RT_SUCCESS, rtContextLaunch1D( m_context_api, 0, 4 ) );
    ASSERT_FALSE( m_accel->isDirty() );

    // force gg into sbt reallocation
    m_geomInstance->setMaterialCount( 2 );

    ASSERT_FALSE( m_accel->isDirty() );
}

TEST_F_DEV( RTXSBTInstanceFixture, propagates_sbt_index_when_transform_child_attached )
{
    Transform* transform = createTranslation( 10.0f, 12.0f, 14.0f );
    m_group->setChildCount( 1 );
    m_group->setChild( 0, transform );
    connectGeometryToGeometryGroup( m_geomGroup, m_geomInstance, createMaterial() );
    transform->setChild( m_geomGroup );
    connectGeometryToGeometryGroup( createGeometryGroup(), createGeometryInstance(), createMaterial() );
    ASSERT_EQ( RT_SUCCESS, rtContextLaunch1D( m_context_api, 0, 4 ) );
    ASSERT_FALSE( m_accel->isDirty() );

    // force gg into sbt reallocation while unattached, then reattach it
    transform->setChild( nullptr );
    m_geomInstance->setMaterialCount( 2 );
    transform->setChild( m_geomGroup );

    ASSERT_TRUE( m_accel->isDirty() );
}

TEST_F_DEV( RTXSBTInstanceFixture, propagates_sbt_index_when_group_child_attached )
{
    Transform* transform = createTranslation( 10.0f, 12.0f, 14.0f );
    m_group->setChildCount( 1 );
    m_group->setChild( 0, transform );
    connectGeometryToGeometryGroup( m_geomGroup, m_geomInstance, createMaterial() );
    transform->setChild( m_geomGroup );
    connectGeometryToGeometryGroup( createGeometryGroup(), createGeometryInstance(), createMaterial() );
    ASSERT_EQ( RT_SUCCESS, rtContextLaunch1D( m_context_api, 0, 4 ) );
    ASSERT_FALSE( m_accel->isDirty() );

    // force gg into sbt reallocation while unattached, then reattach it
    m_group->setChild( 0, nullptr );
    m_geomInstance->setMaterialCount( 2 );
    m_group->setChild( 0, transform );

    ASSERT_TRUE( m_accel->isDirty() );
}
