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

#include <Context/Context.h>
#include <Device/DeviceManager.h>
#include <Objects/Group.h>

using namespace optix;
using namespace testing;

namespace {

struct RTXGroupFixture : Test
{
    RTXGroupFixture()
        : Test()
        , m_context_api( nullptr )
        , m_context( nullptr )
        , m_group( nullptr )
    {
    }

    void SetUp() override
    {
        optix::forceRtxExelwtionStrategy();
        ASSERT_EQ( RT_SUCCESS, rtContextCreate( &m_context_api ) );
        m_context = reinterpret_cast<Context*>( m_context_api );
        m_context->getDeviceManager()->enableActiveDevices();
        m_group = createGroup();
    }

    void TearDown() override { rtContextDestroy( m_context_api ); }

    Group* createGroup()
    {
        RTgroup group_api;
        rtGroupCreate( m_context_api, &group_api );
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

    RTcontext m_context_api;
    Context*  m_context;
    Group*    m_group;

  private:
};

}  // namespace

TEST_F_DEV( RTXGroupFixture, propagates_instance_matrix_from_child_transform )
{
    Transform* transform = createTranslation( 10.0f, 12.0f, 14.0f );
    m_group->setChildCount( 1 );

    m_group->setChild( 0, transform );

    InstanceDescriptorHost::DeviceIndependent desc = m_group->getInstanceDescriptor( 0 );
    float                                     data[16];
    transform->getMatrix( &data[0], false );
    for( size_t i = 0; i < 12; ++i )
    {
        ASSERT_EQ( desc.transform[i], data[i] );
    }
}

TEST_F_DEV( RTXGroupFixture, propagates_instance_matrix_from_chained_child_transforms )
{
    Transform* transform1 = createTranslation( 10.0f, 12.0f, 14.0f );
    Transform* transform2 = createTranslation( 20.0f, 22.0f, 24.0f );
    transform1->setChild( transform2 );
    m_group->setChildCount( 1 );

    m_group->setChild( 0, transform1 );

    Matrix4x4 combined = Matrix4x4::translate( make_float3( 10.0f, 12.0f, 14.0f ) )
                         * Matrix4x4::translate( make_float3( 20.0f, 22.0f, 24.0f ) );
    InstanceDescriptorHost::DeviceIndependent desc         = m_group->getInstanceDescriptor( 0 );
    const float*                              combinedData = combined.getData();
    for( size_t i = 0; i < 12; ++i )
    {
        ASSERT_EQ( desc.transform[i], combinedData[i] );
    }
}

TEST_F_DEV( RTXGroupFixture, propagates_instance_matrix_when_subgraph_joined_to_transform )
{
    Transform* transform1 = createTranslation( 10.0f, 12.0f, 14.0f );
    Transform* transform2 = createTranslation( 20.0f, 22.0f, 24.0f );
    Transform* transform3 = createTranslation( 5.0f, 5.0f, 5.0f );
    m_group->setChildCount( 1 );
    m_group->setChild( 0, transform1 );
    transform1->setChild( transform2 );

    transform2->setChild( transform3 );

    Matrix4x4 combined = Matrix4x4::translate( make_float3( 10.0f, 12.0f, 14.0f ) )
                         * Matrix4x4::translate( make_float3( 20.0f, 22.0f, 24.0f ) )
                         * Matrix4x4::translate( make_float3( 5.0f, 5.0f, 5.0f ) );
    InstanceDescriptorHost::DeviceIndependent desc         = m_group->getInstanceDescriptor( 0 );
    const float*                              combinedData = combined.getData();
    for( size_t i = 0; i < 12; ++i )
    {
        ASSERT_EQ( desc.transform[i], combinedData[i] );
    }
}

TEST_F_DEV( RTXGroupFixture, propagates_instance_matrix_when_subgraph_joined_to_group )
{
    Transform* transform1 = createTranslation( 10.0f, 12.0f, 14.0f );
    Transform* transform2 = createTranslation( 20.0f, 22.0f, 24.0f );
    transform1->setChild( transform2 );
    m_group->setChildCount( 1 );

    m_group->setChild( 0, transform1 );

    Matrix4x4 combined = Matrix4x4::translate( make_float3( 10.0f, 12.0f, 14.0f ) )
                         * Matrix4x4::translate( make_float3( 20.0f, 22.0f, 24.0f ) );
    InstanceDescriptorHost::DeviceIndependent desc         = m_group->getInstanceDescriptor( 0 );
    const float*                              combinedData = combined.getData();
    for( size_t i = 0; i < 12; ++i )
    {
        ASSERT_EQ( desc.transform[i], combinedData[i] );
    }
}

TEST_F_DEV( RTXGroupFixture, propagates_instance_matrix_to_all_parents )
{
    Transform* transform1 = createTranslation( 10.0f, 12.0f, 14.0f );
    Transform* transform2 = createTranslation( 20.0f, 22.0f, 24.0f );
    m_group->setChildCount( 1 );
    m_group->setChild( 0, transform1 );
    Group* group2 = createGroup();
    group2->setChildCount( 1 );
    group2->setChild( 0, transform1 );

    transform1->setChild( transform2 );

    Matrix4x4 combined = Matrix4x4::translate( make_float3( 10.0f, 12.0f, 14.0f ) )
                         * Matrix4x4::translate( make_float3( 20.0f, 22.0f, 24.0f ) );
    InstanceDescriptorHost::DeviceIndependent desc         = m_group->getInstanceDescriptor( 0 );
    const float*                              combinedData = combined.getData();
    for( size_t i = 0; i < 12; ++i )
    {
        ASSERT_EQ( desc.transform[i], combinedData[i] );
    }
    desc = group2->getInstanceDescriptor( 0 );
    for( size_t i = 0; i < 12; ++i )
    {
        ASSERT_EQ( desc.transform[i], combinedData[i] );
    }
}

TEST_F_DEV( RTXGroupFixture, set_matrix_propagates_instance_matrix )
{
    Transform* transform = createTranslation( 10.0f, 12.0f, 14.0f );
    m_group->setChildCount( 1 );
    m_group->setChild( 0, transform );

    Matrix4x4    m    = Matrix4x4().translate( make_float3( 20.0f, 22.0f, 24.0f ) );
    const float* data = m.getData();
    transform->setMatrix( data, false );

    InstanceDescriptorHost::DeviceIndependent desc = m_group->getInstanceDescriptor( 0 );
    for( size_t i = 0; i < 12; ++i )
    {
        ASSERT_EQ( desc.transform[i], data[i] );
    }
}

TEST_F_DEV( RTXGroupFixture, instance_matrix_aclwmulation_stops_before_motion_transform )
{
    Transform* transform       = createTranslation( 10.0f, 12.0f, 14.0f );
    Transform* motionTransform = createTranslation( 20.0f, 22.0f, 24.0f );
    m_group->setChildCount( 1 );
    m_group->setChild( 0, transform );
    const int numKeys = 2;
    float     motionKeys[12 * numKeys]{};
    motionTransform->setKeys( numKeys, RT_MOTIONKEYTYPE_MATRIX_FLOAT12, &motionKeys[0] );
    Transform* transform2 = createTranslation( 40.0f, 44.0f, 48.0f );
    motionTransform->setChild( transform2 );

    transform->setChild( motionTransform );

    float data[16];
    transform->getMatrix( &data[0], false );
    InstanceDescriptorHost::DeviceIndependent desc = m_group->getInstanceDescriptor( 0 );
    for( size_t i = 0; i < 12; ++i )
    {
        ASSERT_EQ( desc.transform[i], data[i] );
    }
}

TEST_F_DEV( RTXGroupFixture, removing_all_motion_keys_propagates_identity_instance_transform )
{
    Transform* transform = createTranslation( 10.0f, 12.0f, 14.0f );
    m_group->setChildCount( 1 );
    m_group->setChild( 0, transform );
    Transform* motionTransform = createTranslation( 20.0f, 22.0f, 24.0f );
    transform->setChild( motionTransform );
    const int numKeys = 2;
    float     motionKeys[12 * numKeys]{};
    motionTransform->setKeys( numKeys, RT_MOTIONKEYTYPE_MATRIX_FLOAT12, &motionKeys[0] );
    Transform* transform2 = createTranslation( 40.0f, 44.0f, 48.0f );
    motionTransform->setChild( transform2 );

    motionTransform->setKeys( 0, RT_MOTIONKEYTYPE_MATRIX_FLOAT12, nullptr );

    Matrix4x4 combined = Matrix4x4::translate( make_float3( 10.0f, 12.0f, 14.0f ) )
                         * Matrix4x4::translate( make_float3( 40.0f, 44.0f, 48.0f ) );
    float*                                    data = combined.getData();
    InstanceDescriptorHost::DeviceIndependent desc = m_group->getInstanceDescriptor( 0 );
    for( size_t i = 0; i < 12; ++i )
    {
        ASSERT_EQ( desc.transform[i], data[i] );
    }
}

TEST_F_DEV( RTXGroupFixture, removing_all_but_one_motion_key_propagates_instance_transform )
{
    Transform* transform = createTranslation( 10.0f, 12.0f, 14.0f );
    m_group->setChildCount( 1 );
    m_group->setChild( 0, transform );
    Transform* motionTransform = createTranslation( 20.0f, 22.0f, 24.0f );
    transform->setChild( motionTransform );
    const int numKeys = 2;
    float     motionKeys[12 * numKeys]{};
    motionTransform->setKeys( numKeys, RT_MOTIONKEYTYPE_MATRIX_FLOAT12, &motionKeys[0] );
    Transform* transform2 = createTranslation( 40.0f, 44.0f, 48.0f );
    motionTransform->setChild( transform2 );

    Matrix4x4 motionUpdatedTransform = Matrix4x4::translate( make_float3( 20.0f, 22.0f, 24.0f ) );
    motionTransform->setKeys( 1, RT_MOTIONKEYTYPE_MATRIX_FLOAT12, motionUpdatedTransform.getData() );

    Matrix4x4 combined = Matrix4x4::translate( make_float3( 10.0f, 12.0f, 14.0f ) )
                         * Matrix4x4::translate( make_float3( 20.0f, 22.0f, 24.0f ) )
                         * Matrix4x4::translate( make_float3( 40.0f, 44.0f, 48.0f ) );
    float*                                    data = combined.getData();
    InstanceDescriptorHost::DeviceIndependent desc = m_group->getInstanceDescriptor( 0 );
    for( size_t i = 0; i < 12; ++i )
    {
        ASSERT_EQ( desc.transform[i], data[i] );
    }
}

TEST_F_DEV( RTXGroupFixture, adding_motion_keys_propagates_instance_transform )
{
    Transform* transform = createTranslation( 10.0f, 12.0f, 14.0f );
    m_group->setChildCount( 1 );
    m_group->setChild( 0, transform );
    Transform* motionTransform = createTranslation( 20.0f, 22.0f, 24.0f );
    transform->setChild( motionTransform );
    Transform* transform2 = createTranslation( 40.0f, 44.0f, 48.0f );
    motionTransform->setChild( transform2 );

    const int numKeys = 2;
    float     motionKeys[12 * numKeys]{};
    motionTransform->setKeys( numKeys, RT_MOTIONKEYTYPE_MATRIX_FLOAT12, &motionKeys[0] );

    Matrix4x4                                 combined = Matrix4x4::translate( make_float3( 10.0f, 12.0f, 14.0f ) );
    float*                                    data     = combined.getData();
    InstanceDescriptorHost::DeviceIndependent desc     = m_group->getInstanceDescriptor( 0 );
    for( size_t i = 0; i < 12; ++i )
    {
        ASSERT_EQ( desc.transform[i], data[i] );
    }
}
