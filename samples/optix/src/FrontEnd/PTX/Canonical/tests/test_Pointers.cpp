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

#include <optixpp_namespace.h>

#include <gtest/gtest.h>
#include <srcTests.h>

#include <optix_world.h>
#include <optixu/optixu_math.h>
#include <optixu/optixu_matrix_namespace.h>

using namespace optix;
using namespace testing;

class TestPointers : public Test
{
  public:
    Context m_context;
    Buffer  m_bufferFloat;
    Buffer  m_bufferFloat3;
    Buffer  m_bufferPointer;
    int     m_bufferFloatId  = 0;
    int     m_bufferFloat3Id = 0;

    const float EPS            = 1e-3f;
    const int   INDEX          = 2;
    const float EXPECTED_VALUE = 10.f;
    int         CONTROL        = 0;

    void SetUp()
    {
        m_context = Context::create();
        m_context->setRayTypeCount( 1 );
        m_context->setEntryPointCount( 1 );

        const int ELEMENTS_NUMBER = 10;

        m_context["bufferIndex"]->setInt( INDEX );
        m_context["value"]->setFloat( EXPECTED_VALUE );
        m_context["control"]->setInt( CONTROL );

        // Init Float buffer.
        m_bufferFloat = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, ELEMENTS_NUMBER );
        m_context["bufferFloat"]->set( m_bufferFloat );

        float* ptr = static_cast<float*>( m_bufferFloat->map() );
        for( int index = 0; index < ELEMENTS_NUMBER; ++index )
            ptr[index] = index + 0.14f;
        m_bufferFloat->unmap();

        // Init Float3 buffer.
        m_bufferFloat3 = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, ELEMENTS_NUMBER );
        m_context["bufferFloat3"]->set( m_bufferFloat3 );

        float3* ptrFloat3 = static_cast<float3*>( m_bufferFloat3->map() );
        for( int index       = 0; index < ELEMENTS_NUMBER; ++index )
            ptrFloat3[index] = optix::make_float3( 0.14f + index, 1.14f + index, 2.14f + index );
        m_bufferFloat3->unmap();

        // Init Pointer buffer.
        Buffer m_bufferPointer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_USER, 1 );
        m_bufferPointer->setElementSize( 8 );
        m_context["addressBuffer"]->set( m_bufferPointer );

        m_context["bufferFloatId"]->setInt( m_bufferFloat->getId() );
        m_context["bufferFloat3Id"]->setInt( m_bufferFloat3->getId() );
    }

    void TearDown() { m_context->destroy(); }

    void setup_TestPointers( const std::string& programName );
};

void TestPointers::setup_TestPointers( const std::string& programName )
{
    std::string ptx_path( ptxPath( "test_Pointers", "pointers.lw" ) );
    Program     rayGen = m_context->createProgramFromPTXFile( ptx_path, programName );

    m_context->setRayGenerationProgram( 0, rayGen );
}

// Tests using normal buffers.
// -----------------------------------------------------------------------------
TEST_F( TestPointers, CanPassPointerToBufferElementToFunction )
{
    setup_TestPointers( "pass_buffer_element_pointer_to_function" );

    m_context->launch( 0, 1 );

    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[2], EXPECTED_VALUE, EXPECTED_VALUE * EPS );
    m_bufferFloat->unmap();
}

TEST_F( TestPointers, CanPassPointerToBufferSubElementToFunction )
{
    setup_TestPointers( "pass_buffer_subelement_pointer_to_function" );

    m_context->launch( 0, 1 );

    float3* ptr = static_cast<float3*>( m_bufferFloat3->map() );
    EXPECT_NEAR( ptr[INDEX].x, INDEX + 0.14f, ( INDEX + 0.14f ) * EPS );
    EXPECT_NEAR( ptr[INDEX].y, EXPECTED_VALUE, EXPECTED_VALUE * EPS );
    EXPECT_NEAR( ptr[INDEX].z, INDEX + 2.14f, ( INDEX + 2.14f ) * EPS );
    m_bufferFloat3->unmap();
}

TEST_F( TestPointers, CanPassPointerToBufferElementToFunctionAfterSwitch_IfBranch )
{
    setup_TestPointers( "pass_buffer_pointer_to_function_after_switch" );

    // Test the if-branch.
    CONTROL = 10;
    m_context["control"]->setInt( CONTROL );
    m_context->launch( 0, 1 );
    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[2], EXPECTED_VALUE, EXPECTED_VALUE * EPS );
    m_bufferFloat->unmap();
}

TEST_F( TestPointers, CanPassPointerToBufferElementToFunctionAfterSwitch_ElseBranch )
{
    setup_TestPointers( "pass_buffer_pointer_to_function_after_switch" );
    // Test the else-branch.
    CONTROL = 20;
    m_context["control"]->setInt( CONTROL );
    m_context->launch( 0, 1 );
    float3* ptr3 = static_cast<float3*>( m_bufferFloat3->map() );
    EXPECT_NEAR( ptr3[INDEX].x, INDEX + 0.14f, ( INDEX + 0.14f ) * EPS );
    EXPECT_NEAR( ptr3[INDEX].y, EXPECTED_VALUE, EXPECTED_VALUE * EPS );
    EXPECT_NEAR( ptr3[INDEX].z, INDEX + 2.14f, ( INDEX + 2.14f ) * EPS );
    m_bufferFloat3->unmap();
}

TEST_F( TestPointers, CanPassPointerToBufferSubElementToFunctionWithArithmetics )
{
    setup_TestPointers( "pass_buffer_subelement_pointer_to_function_with_arithmetics" );

    m_context->launch( 0, 1 );

    float3* ptr = static_cast<float3*>( m_bufferFloat3->map() );
    EXPECT_NEAR( ptr[INDEX].x, INDEX + 0.14f, ( INDEX + 0.14f ) * EPS );
    EXPECT_NEAR( ptr[INDEX].y, EXPECTED_VALUE, EXPECTED_VALUE * EPS );
    EXPECT_NEAR( ptr[INDEX].z, INDEX + 2.14f, ( INDEX + 2.14f ) * EPS );
    m_bufferFloat3->unmap();
}

TEST_F( TestPointers, CanReturnPointerToBufferFromFunction )
{
    setup_TestPointers( "return_buffer_element_pointer_from_function" );

    m_context->launch( 0, 1 );

    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[INDEX], EXPECTED_VALUE, EXPECTED_VALUE * EPS );
    m_bufferFloat->unmap();
}

TEST_F( TestPointers, CanStorePointerToBufferElementInBuffer )
{
    setup_TestPointers( "store_pointer_to_buffer_in_buffer" );

    m_context->launch( 0, 1 );

    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[INDEX], EXPECTED_VALUE, EXPECTED_VALUE * EPS );
    m_bufferFloat->unmap();
}

TEST_F( TestPointers, CanPassGlobalVariableToFunction )
{
    setup_TestPointers( "pass_global_variable_to_function" );

    m_context->launch( 0, 1 );

    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[0], EXPECTED_VALUE, EXPECTED_VALUE * EPS );
    m_bufferFloat->unmap();
}

// Tests using buffer ids.
// -----------------------------------------------------------------------------
TEST_F( TestPointers, CanPassPointerToBufferIdElementToFunction )
{
    setup_TestPointers( "pass_buffer_id_element_pointer_to_function" );

    m_context->launch( 0, 1 );

    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[2], EXPECTED_VALUE, EXPECTED_VALUE * EPS );
    m_bufferFloat->unmap();
}

TEST_F( TestPointers, CanPassPointerToBufferIdSubElementToFunction )
{
    setup_TestPointers( "pass_buffer_id_subelement_pointer_to_function" );

    m_context->launch( 0, 1 );

    float3* ptr = static_cast<float3*>( m_bufferFloat3->map() );
    EXPECT_NEAR( ptr[INDEX].x, INDEX + 0.14f, ( INDEX + 0.14f ) * EPS );
    EXPECT_NEAR( ptr[INDEX].y, EXPECTED_VALUE, EXPECTED_VALUE * EPS );
    EXPECT_NEAR( ptr[INDEX].z, INDEX + 2.14f, ( INDEX + 2.14f ) * EPS );
    m_bufferFloat3->unmap();
}

TEST_F( TestPointers, CanPassPointerToBufferIdSubElementToFunctionWithArithmetics )
{
    setup_TestPointers( "pass_buffer_id_subelement_pointer_to_function_with_arithmetics" );

    m_context->launch( 0, 1 );

    float3* ptr = static_cast<float3*>( m_bufferFloat3->map() );
    EXPECT_NEAR( ptr[INDEX].x, INDEX + 0.14f, ( INDEX + 0.14f ) * EPS );
    EXPECT_NEAR( ptr[INDEX].y, EXPECTED_VALUE, EXPECTED_VALUE * EPS );
    EXPECT_NEAR( ptr[INDEX].z, INDEX + 2.14f, ( INDEX + 2.14f ) * EPS );
    m_bufferFloat3->unmap();
}

TEST_F( TestPointers, CanReturnPointerToBufferIdFromFunction )
{
    setup_TestPointers( "return_buffer_id_element_pointer_from_function" );

    m_context->launch( 0, 1 );

    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[INDEX], EXPECTED_VALUE, EXPECTED_VALUE * EPS );
    m_bufferFloat->unmap();
}

TEST_F( TestPointers, CanStorePointerToBufferIdElementInBuffer )
{
    setup_TestPointers( "store_pointer_to_buffer_id_in_buffer" );

    m_context->launch( 0, 1 );

    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[INDEX], EXPECTED_VALUE, EXPECTED_VALUE * EPS );
    m_bufferFloat->unmap();
}

// Tests on aliasing different types of pointers.
// -----------------------------------------------------------------------------
TEST_F( TestPointers, CannotAliasPointerToVariableWithPointerToBuffer )
{
    EXPECT_ANY_THROW( setup_TestPointers( "alias_buffer_and_variable_pointers" ) );
}

// -----------------------------------------------------------------------------
TEST_F( TestPointers, CanAliasStackAndVariablePointers_If )
{
    setup_TestPointers( "alias_stack_and_variable_pointers" );

    m_context["control"]->setInt( 5 );
    m_context->launch( 0, 1 );

    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[0], 55.56f, 55.56f * EPS );
    m_bufferFloat->unmap();
}
// -----------------------------------------------------------------------------
TEST_F( TestPointers, CanAliasStackAndVariablePointers_Else )
{
    setup_TestPointers( "alias_stack_and_variable_pointers" );

    m_context["control"]->setInt( 15 );
    m_context->launch( 0, 1 );

    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[0], EXPECTED_VALUE, EXPECTED_VALUE * EPS );
    m_bufferFloat->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestPointers, CanLoadFromRawOrBufferPointers_If )
{
    setup_TestPointers( "alias_escaped_and_buffer_pointer_load" );

    m_context["control"]->setInt( 5 );
    m_context->launch( 0, 1 );

    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[0], 3.14, 3.14 * EPS );
    m_bufferFloat->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestPointers, CanLoadFromRawOrBufferPointers_Else )
{
    setup_TestPointers( "alias_escaped_and_buffer_pointer_load" );

    m_context["control"]->setInt( 15 );
    m_context->launch( 0, 1 );

    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[0], 1.14, 1.14 * EPS );
    m_bufferFloat->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestPointers, CanStoreFromRawOrBufferPointers_If )
{
    setup_TestPointers( "alias_escaped_and_buffer_pointer_store" );

    m_context["control"]->setInt( 5 );
    m_context->launch( 0, 1 );

    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[3], 54.32, 54.32 * EPS );
    m_bufferFloat->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestPointers, CanStoreFromRawOrBufferPointers_Else )
{
    setup_TestPointers( "alias_escaped_and_buffer_pointer_store" );

    m_context["control"]->setInt( 15 );
    m_context->launch( 0, 1 );

    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[1], 54.32, 54.32 * EPS );
    m_bufferFloat->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestPointers, CanEscapeRawOrBufferPointers_If )
{
    setup_TestPointers( "alias_escaped_and_buffer_pointer_function" );

    m_context["control"]->setInt( 5 );
    m_context->launch( 0, 1 );

    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[4], EXPECTED_VALUE, EXPECTED_VALUE * EPS );
    m_bufferFloat->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestPointers, CanEscapeRawOrBufferPointers_Else )
{
    setup_TestPointers( "alias_escaped_and_buffer_pointer_function" );

    m_context["control"]->setInt( 15 );
    m_context->launch( 0, 1 );

    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[2], EXPECTED_VALUE, EXPECTED_VALUE * EPS );
    m_bufferFloat->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestPointers, CanAtomicallyModifyRawOrBufferPointers_If )
{
    setup_TestPointers( "alias_escaped_and_buffer_pointer_atomic" );

    m_context["control"]->setInt( 5 );
    m_context->launch( 0, 1 );

    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[3], 6.14f, 6.14f * EPS );
    m_bufferFloat->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestPointers, CanAtomicallyModifyRawOrBufferPointers_Else )
{
    setup_TestPointers( "alias_escaped_and_buffer_pointer_atomic" );

    m_context["control"]->setInt( 15 );
    m_context->launch( 0, 1 );

    float* ptr = static_cast<float*>( m_bufferFloat->map() );
    EXPECT_NEAR( ptr[1], 4.14f, 4.14f * EPS );
    m_bufferFloat->unmap();
}

// -----------------------------------------------------------------------------
// This tests:
// 1. Passing a pointer to the payload to a function.
// 2. The payload is a pointer to a buffer element.
class TestPayloadPointers : public Test
{
  public:
    Context m_context;
    Buffer  m_outputBuffer;
    Program m_rayGen;
    Program m_intersection;
    Program m_closestHit;
    Program m_boundingBox;

    float4      EXPECTED_COLOR;
    const float EPS = 1e-3f;

    void SetUp()
    {
        m_context = Context::create();
        m_context->setRayTypeCount( 1 );
        m_context->setEntryPointCount( 1 );

        EXPECTED_COLOR = make_float4( 1.f, 0.f, 0.f, 1.f );

        m_outputBuffer = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, 1, 1 );
        m_context["outputBuffer"]->set( m_outputBuffer );
        float4* ptr = static_cast<float4*>( m_outputBuffer->map() );
        ptr[0]      = make_float4( 0.0f, 0.f, 0.f, 0.f );
        m_outputBuffer->unmap();
    }

    void TearDown() { m_context->destroy(); }

    void setup_TestPayloadPointers( std::string file )
    {
        std::string ptx_path( ptxPath( "test_Pointers", file ) );
        m_rayGen       = m_context->createProgramFromPTXFile( ptx_path, "rayGen" );
        m_intersection = m_context->createProgramFromPTXFile( ptx_path, "intersection" );
        m_closestHit   = m_context->createProgramFromPTXFile( ptx_path, "closestHit" );
        m_boundingBox  = m_context->createProgramFromPTXFile( ptx_path, "boundingBox" );

        m_context->setRayGenerationProgram( 0, m_rayGen );

        Material material = m_context->createMaterial();
        material->setClosestHitProgram( 0, m_closestHit );

        Geometry geo = m_context->createGeometry();
        geo->setPrimitiveCount( 1 );
        geo->setIntersectionProgram( m_intersection );
        geo->setBoundingBoxProgram( m_boundingBox );

        GeometryInstance geoInstance = m_context->createGeometryInstance();
        geoInstance->setMaterialCount( 1 );
        geoInstance->setMaterial( 0, material );
        geoInstance->setGeometry( geo );

        GeometryGroup geoGroup = m_context->createGeometryGroup();
        geoGroup->setChildCount( 1 );
        geoGroup->setChild( 0, geoInstance );
        geoGroup->setAcceleration( m_context->createAcceleration( "NoAccel", "NoAccel" ) );

        Transform transform = m_context->createTransform();
        transform->setChild( geoGroup );

        Group topLevelGroup = m_context->createGroup();
        topLevelGroup->setChildCount( 1 );
        topLevelGroup->setChild( 0, transform );
        topLevelGroup->setAcceleration( m_context->createAcceleration( "NoAccel", "NoAccel" ) );

        m_context["topObject"]->set( topLevelGroup );
    }
};

// -----------------------------------------------------------------------------
TEST_F( TestPayloadPointers, CanPassPointerToPayloadToFunction )
{
    setup_TestPayloadPointers( "pointer_to_payload_to_function.lw" );

    m_context->launch( 0, 1 );

    float4* ptr = static_cast<float4*>( m_outputBuffer->map() );
    EXPECT_NEAR( ptr[0].x, EXPECTED_COLOR.x, EPS * EXPECTED_COLOR.x );
    EXPECT_NEAR( ptr[0].y, EXPECTED_COLOR.y, EPS * EXPECTED_COLOR.y );
    EXPECT_NEAR( ptr[0].z, EXPECTED_COLOR.z, EPS * EXPECTED_COLOR.z );
    EXPECT_NEAR( ptr[0].w, EXPECTED_COLOR.w, EPS * EXPECTED_COLOR.w );
    m_outputBuffer->unmap();
}

// -----------------------------------------------------------------------------
TEST_F( TestPayloadPointers, CanPassPointerToBufferInPayload )
{
    setup_TestPayloadPointers( "pointer_to_buffer_in_payload.lw" );

    m_context->launch( 0, 1 );

    float4* ptr = static_cast<float4*>( m_outputBuffer->map() );
    EXPECT_NEAR( ptr[0].x, EXPECTED_COLOR.x, EPS * EXPECTED_COLOR.x );
    EXPECT_NEAR( ptr[0].y, EXPECTED_COLOR.y, EPS * EXPECTED_COLOR.y );
    EXPECT_NEAR( ptr[0].z, EXPECTED_COLOR.z, EPS * EXPECTED_COLOR.z );
    EXPECT_NEAR( ptr[0].w, EXPECTED_COLOR.w, EPS * EXPECTED_COLOR.w );
    m_outputBuffer->unmap();
}

// -----------------------------------------------------------------------------
struct PTXModule
{
    const char* description;
    const char* functionName;
    const char* code;
};

#define PTX_MODULE( functionName, ... )                                                                                \
    {                                                                                                                  \
        "", functionName, #__VA_ARGS__                                                                                 \
    }


class TestPTXPointers : public Test
{
  public:
    Context     m_context;
    Buffer      m_bufferFloat;
    Buffer      m_bufferFloat3;
    Buffer      m_outputBuffer;
    int         m_bufferFloatId  = 0;
    int         m_bufferFloat3Id = 0;
    PTXModule   module;
    int         INDEX = 2;
    const float EPS   = 1e-3f;

    void SetUp()
    {
        m_context = Context::create();
        m_context->setRayTypeCount( 1 );
        m_context->setEntryPointCount( 1 );

        const int ELEMENTS_NUMBER = 9;

        m_context["bufferIndex"]->setInt( INDEX );

        // Init Float buffer.
        m_bufferFloat = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT, ELEMENTS_NUMBER );
        m_context["bufferFloat"]->set( m_bufferFloat );

        float* ptr = static_cast<float*>( m_bufferFloat->map() );
        for( int index = 0; index < ELEMENTS_NUMBER; ++index )
            ptr[index] = index + 0.14f;
        m_bufferFloat->unmap();

        // Init Float3 buffer.
        m_bufferFloat3 = m_context->createBuffer( RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT3, ELEMENTS_NUMBER );
        m_context["bufferFloat3"]->set( m_bufferFloat3 );

        float3* ptrFloat3 = static_cast<float3*>( m_bufferFloat3->map() );
        for( int index       = 0; index < ELEMENTS_NUMBER; ++index )
            ptrFloat3[index] = optix::make_float3( 0.14f + index, 1.14f + index, 2.14f + index );
        m_bufferFloat3->unmap();

        m_outputBuffer = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, 1 );
        m_context["outputBuffer"]->set( m_outputBuffer );

        m_context["bufferFloatId"]->setInt( m_bufferFloat->getId() );
        m_context["bufferFloat3Id"]->setInt( m_bufferFloat3->getId() );
    }

    void TearDown() { m_context->destroy(); }

    void setup_TestPTXPointers( const std::string& code, const std::string& programName )
    {
        Program rayGen = m_context->createProgramFromPTXString( code.c_str(), programName.c_str() );
        m_context->setRayGenerationProgram( 0, rayGen );
    }
};

// Notice that, in the call to _rt_buffer_get_id_64, the third argument is the size of a buffer element.
// In this case it is set to 1, meaning that we treat the buffer as a buffer of bytes.
// This test forces the pointer to escape.
//
// Notice further that BufferFloatId is declared as buffer of ints (floats would make more sense),
// but the _rt_buffer_get_id_64 call uses an element size of 1. Such PTX code does not result from
// our headers (which generate sizeof(T) as element size). While the handcrafted PTX code computes
// the same address in the end, it can easily trigger the RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS
// exception since the indices used here might be higher than the buffer sizes used to create the
// buffer on the host side. In this example, ELEMENTS_NUMBER = 3 would be sufficient for INDEX = 2,
// but we use ELEMENTS_NUMBER = 9 to avoid triggering the exception. The exception check could be
// improved by carrying the element size used during buffer creation in the BufferHeader.

// clang-format off
PTXModule readFloatFromBufferOfBytes = PTX_MODULE( "readFloatFromBufferOfBytes",
    .version 4.2
    .target sm_20
    .address_size 64

    .global .align 1 .b8 outputBuffer[1];
    .global .align 4 .u32 bufferFloatId;
    .global .align 4 .u32 bufferIndex;
    .global .align 4 .b8 _ZN21rti_internal_typeinfo13bufferFloatIdE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
    .global .align 4 .b8 _ZN21rti_internal_typeinfo11bufferIndexE[8] = {82, 97, 121, 0, 4, 0, 0, 0};
    .global .align 1 .b8 _ZN21rti_internal_typename13bufferFloatIdE[4] = {105, 110, 116, 0};
    .global .align 1 .b8 _ZN21rti_internal_typename11bufferIndexE[4] = {105, 110, 116, 0};
    .global .align 4 .u32 _ZN21rti_internal_typeenum13bufferFloatIdE = 4919;
    .global .align 4 .u32 _ZN21rti_internal_typeenum11bufferIndexE = 4919;
    .global .align 1 .b8 _ZN21rti_internal_semantic13bufferFloatIdE[1];
    .global .align 1 .b8 _ZN21rti_internal_semantic11bufferIndexE[1];
    .global .align 1 .b8 _ZN23rti_internal_annotation13bufferFloatIdE[1];
    .global .align 1 .b8 _ZN23rti_internal_annotation11bufferIndexE[1];

    .visible .entry _Z26readFloatFromBufferOfBytesv()
    {
      .reg .f32   %f<2>;
      .reg .s32   %r<11>;
      .reg .s64   %rd<13>;

      ldu.global.u32  %r1, [bufferFloatId];
      ldu.global.u32  %r6, [bufferIndex];
      cvt.s32.u32   %r8, %r6;
      mov.s32   %r9, 4;
      mul.wide.s32   %rd2, %r9, %r8; // multiply bufferIndex by 4
      mov.u32   %r4, 1;
      mov.u32   %r5, 4;
      mov.u32   %r7, 1; // use element size of 1 instead of 4
      mov.u64   %rd11, 0;
      call (%rd1), _rt_buffer_get_id_64, (%r1, %r4, %r7, %rd2, %rd11, %rd11, %rd11);
      ld.f32  %f1, [%rd1];
      mov.u64   %rd12, outputBuffer;
      cvta.global.u64   %rd7, %rd12;
      call (%rd6), _rt_buffer_get_64, (%rd7, %r4, %r5, %rd11, %rd11, %rd11, %rd11);
      st.f32  [%rd6], %f1;
      ret;
    }
);
// clang-format on

TEST_F( TestPTXPointers, CanReadBufferOfBytesAsFloat )
{
    setup_TestPTXPointers( readFloatFromBufferOfBytes.code, readFloatFromBufferOfBytes.functionName );

    m_context->launch( 0, 1 );

    float*      inputPtr       = static_cast<float*>( m_bufferFloat->map() );
    const float EXPECTED_VALUE = inputPtr[INDEX];
    m_bufferFloat->unmap();

    float*      outputPtr    = static_cast<float*>( m_outputBuffer->map() );
    const float ACTUAL_VALUE = outputPtr[0];
    m_outputBuffer->unmap();

    EXPECT_NEAR( ACTUAL_VALUE, EXPECTED_VALUE, EPS * EXPECTED_VALUE );
}
