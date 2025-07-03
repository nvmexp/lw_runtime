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

#include <srcTests.h>

#include <prodlib/system/Knobs.h>

#include <optixpp_namespace.h>

using namespace optix;
using namespace testing;


// The name of the exelwtable in which this test will be included.
// Used to locate the test's PTX files.
static const char* EXELWTABLE_NAME = "test_Device";


#define REQUIRE_NUM_DEVICES( N )                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if( !requireAtLeastDevices( N ) )                                                                              \
            return;                                                                                                    \
    } while( false )

class TestMultiGpu : public testing::Test
{
  public:
    Context      m_context;
    Buffer       m_input;
    Buffer       m_output;
    unsigned int m_numDevices;
    unsigned int m_cpuDeviceIndex;

    // Initialization of non-static is not supported in VS2013.
    std::vector<int> FIRST_GPU;                   // = { 0 };
    std::vector<int> SECOND_GPU;                  // = { 1 };
    std::vector<int> FIRST_AND_SECOND_GPU;        // = { 0, 1 };
    std::vector<int> FIRST_AND_THIRD_GPU;         // = { 0, 2 };
    std::vector<int> SECOND_AND_THIRD_GPU;        // = { 1, 2 };
    std::vector<int> FIRST_SECOND_AND_THIRD_GPU;  // = { 0, 1, 2 };
    std::vector<int> CPU;                         // filled during setup
    std::vector<int> TWO_GPUS_AND_CPU;            // filled during setup
    std::vector<int> THREE_GPUS_AND_CPU;          // filled during setup

    TestMultiGpu() {}

    void SetUp()
    {
        m_input   = nullptr;
        m_output  = nullptr;
        m_context = Context::create();
        m_context->setRayTypeCount( 1 );
        m_context->setEntryPointCount( 1 );
        m_numDevices = m_context->getDeviceCount();

        FIRST_GPU.push_back( 0 );
        SECOND_GPU.push_back( 1 );
        FIRST_AND_SECOND_GPU.push_back( 0 );
        FIRST_AND_SECOND_GPU.push_back( 1 );
        FIRST_AND_THIRD_GPU.push_back( 0 );
        FIRST_AND_THIRD_GPU.push_back( 2 );
        SECOND_AND_THIRD_GPU.push_back( 1 );
        SECOND_AND_THIRD_GPU.push_back( 2 );
        FIRST_SECOND_AND_THIRD_GPU.push_back( 0 );
        FIRST_SECOND_AND_THIRD_GPU.push_back( 1 );
        FIRST_SECOND_AND_THIRD_GPU.push_back( 2 );

        m_input = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 1 );
        m_context["input"]->set( m_input );

        m_output = m_context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_INT, 1 );
        m_context["output"]->set( m_output );
    }

    void TearDown()
    {
        if( m_context )
            m_context->destroy();

        m_output  = nullptr;
        m_input   = nullptr;
        m_context = nullptr;
    }

    void resetOutputBuffer();

    bool requireAtLeastDevices( unsigned int N );
    void setDevices( std::vector<int>& devices );
    void setupProgram( const std::string& ptxFile, const std::string raygenName );
};

bool TestMultiGpu::requireAtLeastDevices( unsigned int N )
{
    if( m_numDevices < N )
    {
        std::cout << "\n\n\t[WARNING] Test could not complete since not enough devices were found! (expected: " << N
                  << ", detected: " << m_numDevices << ")\n\n";
        return false;
    }

    return true;
}

void TestMultiGpu::setDevices( std::vector<int>& devices )
{
    m_context->setDevices( devices.begin(), devices.end() );
}

void TestMultiGpu::setupProgram( const std::string& ptxFile, const std::string raygenName )
{
    const std::string ptx_path = ptxPath( EXELWTABLE_NAME, ptxFile );

    Program rayGen = m_context->createProgramFromPTXFile( ptx_path, raygenName );
    m_context->setRayGenerationProgram( 0, rayGen );
}

class TestMultiGpuWithCpu : public TestMultiGpu
{
  public:
    TestMultiGpuWithCpu()
        : m_allowCPUFallbackKnob( "deviceManager.allowCPUFallback", true )
    {
    }

    void SetUp()
    {
        TestMultiGpu::SetUp();
        m_cpuDeviceIndex = m_numDevices - 1;
        CPU.push_back( m_cpuDeviceIndex );
        TWO_GPUS_AND_CPU = FIRST_AND_SECOND_GPU;
        TWO_GPUS_AND_CPU.push_back( m_cpuDeviceIndex );
        THREE_GPUS_AND_CPU.push_back( 0 );
        THREE_GPUS_AND_CPU.push_back( 1 );
        THREE_GPUS_AND_CPU.push_back( 2 );
        THREE_GPUS_AND_CPU.push_back( m_cpuDeviceIndex );
    }

  private:
    ScopedKnobSetter m_allowCPUFallbackKnob;
};

//// These are preparatory tests. This should also be covered by testSetDevices.

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestMultiGpu, CanEnableTwoGpus )
{
    REQUIRE_NUM_DEVICES( 2 );
    setDevices( FIRST_AND_SECOND_GPU );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F_DEV( TestMultiGpuWithCpu, CanEnableTwoGpusAndCpu )
{
    REQUIRE_NUM_DEVICES( 3 );
    setDevices( TWO_GPUS_AND_CPU );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F_DEV( TestMultiGpuWithCpu, EnableAndDisableTwoGpus )
{
    REQUIRE_NUM_DEVICES( 2 );
    setDevices( CPU );
    setDevices( FIRST_AND_SECOND_GPU );
    setDevices( CPU );
}

////////////////////////////////////////////////////////////////////////////////

//// Launch in two devices with zero dimension.

// This tests that a zero in the launch dimension is handled properly.
// The test should pass, and the launch should be internally skipped.
// Events handling the task should work correctly.

TEST_F( TestMultiGpu, LaunchTwoGpusWithZeroSize )
{
    REQUIRE_NUM_DEVICES( 2 );
    setupProgram( "buffer.lw", "rg" );
    setDevices( FIRST_AND_SECOND_GPU );

    m_context->launch( 0, 0 /* width */ );
}

////////////////////////////////////////////////////////////////////////////////

//// Launch in two devices (1D).

TEST_F( TestMultiGpu, Launch1DTwoGpus )
{
    REQUIRE_NUM_DEVICES( 2 );
    setupProgram( "buffer.lw", "rg" );
    setDevices( FIRST_AND_SECOND_GPU );

    m_context->launch( 0, 1 /* width */ );
}


////////////////////////////////////////////////////////////////////////////////

//// Launch in two devices (1D).

// This tests that we can launch in rapid succession.
// Events should handle the synchronization and wait at the right times
// for the launch to finish before continuing with the next one.

TEST_F( TestMultiGpu, Launch1DTwoGpusMultipleTimes )
{
    REQUIRE_NUM_DEVICES( 3 );
    setupProgram( "buffer.lw", "rg" );
    setDevices( FIRST_AND_SECOND_GPU );

    // 4 times here is arbitrary.
    m_context->launch( 0, 1 /* width */ );
    m_context->launch( 0, 1 /* width */ );
    m_context->launch( 0, 1 /* width */ );
    m_context->launch( 0, 1 /* width */ );
}

////////////////////////////////////////////////////////////////////////////////

//// Launch in two devices (2D).

TEST_F( TestMultiGpu, Launch2DTwoGpus )
{
    REQUIRE_NUM_DEVICES( 3 );
    setupProgram( "buffer.lw", "rg" );
    setDevices( FIRST_AND_SECOND_GPU );

    m_context->launch( 0, 1 /* width */, 1 /* height */ );
}


////////////////////////////////////////////////////////////////////////////////

//// Launch in two devices (2D).

// This tests that we can launch in rapid succession.
// Events should handle the synchronization and wait at the right times
// for the launch to finish before continuing with the next one.

TEST_F( TestMultiGpu, Launch2DTwoGpusMultipleTimes )
{
    REQUIRE_NUM_DEVICES( 3 );
    setupProgram( "buffer.lw", "rg" );
    setDevices( FIRST_AND_SECOND_GPU );

    // 4 times here is arbitrary.
    m_context->launch( 0, 1 /* width */, 1 /* height */ );
    m_context->launch( 0, 1 /* width */, 1 /* height */ );
    m_context->launch( 0, 1 /* width */, 1 /* height */ );
    m_context->launch( 0, 1 /* width */, 1 /* height */ );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestMultiGpu, CanChangeRegistarForZeroCopyBuffers )
{
    REQUIRE_NUM_DEVICES( 3 );
    setupProgram( "buffer.lw", "rg" );
    setDevices( FIRST_AND_SECOND_GPU );
    m_context->launch( 0, 1 /* width */, 1 /* height */ );
    setDevices( SECOND_AND_THIRD_GPU );
    m_context->launch( 0, 1 /* width */, 1 /* height */ );
}

////////////////////////////////////////////////////////////////////////////////

TEST_F( TestMultiGpu, CanAddGpuToActiveDevices )
{
    REQUIRE_NUM_DEVICES( 3 );
    setupProgram( "buffer.lw", "rg" );
    setDevices( FIRST_GPU );
    m_context->launch( 0, 1 /* width */, 1 /* height */ );
    setDevices( FIRST_AND_SECOND_GPU );
    m_context->launch( 0, 1 /* width */, 1 /* height */ );
    setDevices( FIRST_SECOND_AND_THIRD_GPU );
    m_context->launch( 0, 1 /* width */, 1 /* height */ );
    setDevices( SECOND_AND_THIRD_GPU );
    m_context->launch( 0, 1 /* width */, 1 /* height */ );
    setDevices( FIRST_SECOND_AND_THIRD_GPU );
    m_context->launch( 0, 1 /* width */, 1 /* height */ );
}
