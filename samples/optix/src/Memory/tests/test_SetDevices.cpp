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

#include <prodlib/system/Knobs.h>
#include <srcTests.h>

#include <optix_world.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

using namespace optix;
using namespace testing;


#define REQUIRE_NUM_DEVICES( N )                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if( !requireAtLeastDevices( N ) )                                                                              \
            return;                                                                                                    \
    } while( false )

class TestSetDevices : public Test
{
  public:
    Context      m_context;
    Buffer       m_input;
    Buffer       m_output;
    unsigned int m_numDevices;

    unsigned int m_expectedValue;
    unsigned int m_errorValue;

    GeometryGroup makeGeometryGroup()
    {
        std::string ptx_path = ptxPath( "test_Memory", "buffer.lw" );
        Program     ch       = m_context->createProgramFromPTXFile( ptx_path, "ch_simple" );
        Material    material = m_context->createMaterial();
        material->setClosestHitProgram( 0, ch );

        Geometry geometry = m_context->createGeometry();
        geometry->setPrimitiveCount( 1u );

        Program bounds    = m_context->createProgramFromPTXFile( ptx_path, "bounds" );
        Program intersect = m_context->createProgramFromPTXFile( ptx_path, "intersect" );

        geometry->setBoundingBoxProgram( bounds );
        geometry->setIntersectionProgram( intersect );

        GeometryInstance gi = m_context->createGeometryInstance( geometry, &material, &material + 1 );
        GeometryGroup    gg = m_context->createGeometryGroup( &gi, &gi + 1 );
        Acceleration     as = m_context->createAcceleration( "NoAccel", "NoAccel" );
        gg->setAcceleration( as );

        return gg;
    }

    void SetUp() override
    {
        m_input   = nullptr;
        m_output  = nullptr;
        m_context = Context::create();
        m_context->setRayTypeCount( 1 );
        m_context->setEntryPointCount( 1 );
        m_numDevices = m_context->getEnabledDeviceCount();

        m_expectedValue = 0xAAAA0000;
        m_errorValue    = 0xFFFF5555;

        std::string ptx_path = ptxPath( "test_Memory", "buffer.lw" );

        std::string raygenName = "rg";
        Program     rayGen     = m_context->createProgramFromPTXFile( ptx_path, raygenName );
        m_context->setRayGenerationProgram( 0, rayGen );

        m_input = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1 );
        m_context["input"]->set( m_input );

        m_output = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1 );
        m_context["output"]->set( m_output );

        int* input = static_cast<int*>( m_input->map() );
        input[0]   = static_cast<int>( m_expectedValue );
        m_input->unmap();

        m_context["top_object"]->set( makeGeometryGroup() );

        // Initialize output, too.
        resetOutputBuffer();
    }

    void TearDown() override
    {
        if( m_context )
            m_context->destroy();

        m_output  = nullptr;
        m_input   = nullptr;
        m_context = nullptr;
    }

    void resetOutputBuffer();
    bool bufferContentMatch( Buffer& buf, unsigned int expectedValue );
    bool requireAtLeastDevices( unsigned int N );
    void setDevices( bool first_gpu, bool second_gpu );
};

class TestSetDevicesWithCpu : public TestSetDevices
{
  public:
    unsigned int m_cpuDeviceIndex = 0U;

    TestSetDevicesWithCpu()
        : m_allowCPUFallbackKnob( "deviceManager.allowCPUFallback", true )
    {
    }

    void SetUp() override
    {
        TestSetDevices::SetUp();
        m_cpuDeviceIndex = m_numDevices - 1;
    }

    void setDevices( bool first_gpu, bool second_gpu, bool cpu );

  private:
    ScopedKnobSetter m_allowCPUFallbackKnob;
};

void TestSetDevices::resetOutputBuffer()
{
    int* output = static_cast<int*>( m_output->map() );
    output[0]   = static_cast<int>( m_errorValue );
    m_output->unmap();
}

bool TestSetDevices::bufferContentMatch( Buffer& buf, unsigned int expectedValue )
{
    unsigned int* results = static_cast<unsigned int*>( buf->map() );
    bool          match   = ( results[0] == expectedValue );
    m_output->unmap();

    return match;
}

bool TestSetDevices::requireAtLeastDevices( unsigned int N )
{
    if( m_numDevices < N )
    {
        std::cout << "\n\n\t[WARNING] Test could not complete since not enough devices were found! (expected: " << N
                  << ", detected: " << m_numDevices << ")\n\n";
        return false;
    }

    return true;
}

void TestSetDevices::setDevices( bool first_gpu, bool second_gpu )
{
    std::vector<int> devices;

    if( first_gpu )
        devices.push_back( 0 );

    if( second_gpu )
        devices.push_back( 1 );

    m_context->setDevices( devices.begin(), devices.end() );
}

void TestSetDevicesWithCpu::setDevices( bool first_gpu, bool second_gpu, bool cpu )
{
    std::vector<int> devices;

    if( first_gpu )
        devices.push_back( 0 );

    if( second_gpu )
        devices.push_back( 1 );

    if( cpu )
        devices.push_back( m_cpuDeviceIndex );

    m_context->setDevices( devices.begin(), devices.end() );
}

////////////////////////////////////////////////////////////////////////////////
/// Launch kernels with a single one of the available devices.
////////////////////////////////////////////////////////////////////////////////

TEST_F( TestSetDevices, Default )
{
    m_context->launch( 0, 1 /* width */ );

    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestSetDevices, CanLaunchWithFirstGpu )
{
    REQUIRE_NUM_DEVICES( 1 );

    // Disable all but the first GPU device.
    setDevices( true, false );

    m_context->launch( 0, 1 /* width */ );

    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestSetDevices, CanLaunchWithSecondGpu )
{
    REQUIRE_NUM_DEVICES( 2 );

    // Disable all but the second GPU device.
    setDevices( false, true );

    m_context->launch( 0, 1 /* width */ );

    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestSetDevices, CanLaunchWithTwoGPUs )
{
    REQUIRE_NUM_DEVICES( 2 );

    // Disable all but the first two GPU devices.
    setDevices( true, true );

    m_context->launch( 0, 1 /* width */ );

    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );
}

////////////////////////////////////////////////////////////////////////////////
/// Call setDevices with the same set that is lwrrently active.
////////////////////////////////////////////////////////////////////////////////

// requires CPU fallback
TEST_F_DEV( TestSetDevicesWithCpu, CanResetActiveCpuDevice )
{
    setDevices( false, false, true );
    setDevices( false, false, true );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestSetDevices, CanResetActiveFirstGpu )
{
    REQUIRE_NUM_DEVICES( 1 );

    setDevices( true, false );
    setDevices( true, false );
}

////////////////////////////////////////////////////////////////////////////////

// requires CPU fallback
TEST_F_DEV( TestSetDevicesWithCpu, CanResetActiveFirstGpuAfterCPUOnlyWasActive )
{
    REQUIRE_NUM_DEVICES( 2 );

    setDevices( false, false, true );
    setDevices( true, false, false );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestSetDevices, CanResetActiveSecondGpu )
{
    REQUIRE_NUM_DEVICES( 2 );

    setDevices( false, true );
    setDevices( false, true );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F_DEV( TestSetDevicesWithCpu, CanResetActiveCpuGpu )
{
    REQUIRE_NUM_DEVICES( 2 );

    setDevices( false, true, true );
    setDevices( false, true, true );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestSetDevices, CanResetActiveGpuDevices )
{
    REQUIRE_NUM_DEVICES( 2 );

    setDevices( true, true );
    setDevices( true, true );
}

////////////////////////////////////////////////////////////////////////////////
/// Launch, then disable the active device(s) and activate a previously disabled
/// one, then map the buffer. Buffer contents should have been transferred.
////////////////////////////////////////////////////////////////////////////////

// requires CPU fallback
TEST_F_DEV( TestSetDevicesWithCpu, MemorySyncedWhenSwitchingFromFirstGpuToCpu )
{
    REQUIRE_NUM_DEVICES( 2 );

    // Disable all but the first GPU device.
    setDevices( true, false, false );

    // Write 13 to the output.
    m_context->launch( 0, 1 /* width */ );

    // Disable all but the CPU device.
    setDevices( false, false, true );

    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestSetDevices, MemorySyncedWhenSwitchingFromFistGpuToSecondGpu )
{
    // Note this test is to check that the memory is properly
    // synced to the host on device deactivation.
    // It does NOT tests for the memory to be valid in the second GPU.

    REQUIRE_NUM_DEVICES( 2 );

    // Disable all but the first GPU device.
    setDevices( true, false );

    // Write 13 to the output.
    m_context->launch( 0, 1 /* width */ );

    // Disable all but the second GPU device.
    setDevices( false, true );

    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );
}

////////////////////////////////////////////////////////////////////////////////

// This test will only be operational when the CPU fallback works again (OP-131).
TEST_F( TestSetDevicesWithCpu, DISABLED_MemorySyncedWhenCpuAfterLaunch )
{
    REQUIRE_NUM_DEVICES( 2 );

    // Disable all but the CPU device.
    setDevices( false, false, true );

    // Write 13 to the output.
    m_context->launch( 0, 1 /* width */ );

    // Disable all but a single GPU device.
    setDevices( true, false, false );

    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );
}

////////////////////////////////////////////////////////////////////////////////

// requires CPU fallback
TEST_F_DEV( TestSetDevicesWithCpu, MemorySyncedWhenFirstGpuRemovedAfterLaunch )
{
    REQUIRE_NUM_DEVICES( 1 );

    // Disable all but a single GPU device.
    setDevices( true, false, false );

    // Write 13 to the output.
    m_context->launch( 0, 1 /* width */ );

    // Disable all but the CPU device.
    setDevices( false, false, true );

    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestSetDevices, CanHandleDuplicateDevicesInList )
{
    REQUIRE_NUM_DEVICES( 1 );

    // Disable all but a single GPU device, but add this one to the list multiple
    // times.
    // done explicitly because this is a special case.
    std::vector<int> devices;
    devices.push_back( 0 );
    devices.push_back( 0 );
    devices.push_back( 0 );
    m_context->setDevices( devices.begin(), devices.end() );

    // Write 13 to the output.
    m_context->launch( 0, 1 /* width */ );

    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F_DEV( TestSetDevicesWithCpu, MemorySyncedWhenSwitchingMultipleTimesAfterLaunch )
{
    REQUIRE_NUM_DEVICES( 1 );

    // Disable all but a single GPU device.
    setDevices( true, false, false );

    // Write 13 to the output.
    m_context->launch( 0, 1 /* width */ );

    // Disable all but the CPU device.
    setDevices( false, false, true );

    // Disable all but the GPU device again.
    setDevices( true, false, false );

    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );
}

////////////////////////////////////////////////////////////////////////////////

// requires CPU fallback
TEST_F_DEV( TestSetDevicesWithCpu, MemorySyncedWhenSwitchingMultipleTimesAndToDifferentGpuAfterLaunch )
{
    REQUIRE_NUM_DEVICES( 3 );

    // Disable all but a single GPU device.
    setDevices( true, false, false );

    // Write 13 to the output.
    m_context->launch( 0, 1 /* width */ );

    // Disable all but the CPU device.
    setDevices( false, false, true );

    // Disable all but the other GPU device.
    setDevices( false, true, false );

    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );
}

////////////////////////////////////////////////////////////////////////////////

// requires CPU fallback
TEST_F_DEV( TestSetDevicesWithCpu, MemorySyncedWhenSwitchingMultipleTimesAndToDifferentGpuAfterLaunchWithMappingInBetween )
{
    REQUIRE_NUM_DEVICES( 2 );

    // Disable all but a single GPU device.
    setDevices( true, false, false );

    // Write 13 to the output.
    m_context->launch( 0, 1 /* width */ );

    // Disable all but the CPU device.
    setDevices( false, false, true );

    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );

    // Disable all but the other GPU device.
    setDevices( false, true, false );

    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestSetDevices, LaunchFirstGpuThenSecondGpu )
{
    REQUIRE_NUM_DEVICES( 2 );

    // Disable all but the first GPU device.
    setDevices( true, false );

    // Write 13 to the output.
    m_context->launch( 0, 1 /* width */ );

    // check output
    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );

    // reset output
    resetOutputBuffer();

    // Disable all but the second GPU device.
    setDevices( false, true );

    // Write 13 to the output with the second GPU
    m_context->launch( 0, 1 /* width */ );

    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestSetDevices, LaunchFirstGpuThenSecondGpuThenFirstGpu )
{
    REQUIRE_NUM_DEVICES( 2 );

    // Disable all but the first GPU device.
    setDevices( true, false );

    // Write 13 to the output.
    m_context->launch( 0, 1 /* width */ );

    // check output
    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );

    // reset output
    resetOutputBuffer();

    // Disable all but the second GPU device.
    setDevices( false, true );

    // Write 13 to the output with the second GPU
    m_context->launch( 0, 1 /* width */ );

    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );

    // reset output
    resetOutputBuffer();

    // Disable all but the first GPU device.
    setDevices( true, false );

    // Write 13 to the output.
    m_context->launch( 0, 1 /* width */ );

    // check output
    EXPECT_TRUE( bufferContentMatch( m_output, m_expectedValue ) );
}

////////////////////////////////////////////////////////////////////////////////
