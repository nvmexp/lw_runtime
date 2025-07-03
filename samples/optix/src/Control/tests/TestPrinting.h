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

#include <srcTests.h>

#include <prodlib/system/Knobs.h>
#include <prodlib/system/System.h>

#include <optix_world.h>
#include <optixpp_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include <fcntl.h>  // O_WRONLY etc.
#include <sstream>  // std::stringstream
#include <string>

using namespace optix;


class TestPrintingNoKnobs : public testing::Test
{
  public:
    Context m_context;
    Buffer  m_input;

    TestPrintingNoKnobs() {}

    void SetUp()
    {
        // Construct a semi-random filename for the log file based on the amount of
        // host memory. This is to help avoid issues if two test instances happen to
        // be run simultaneously by providing unique files for capturing stdout.
        //
        // The amount of free mem is not significant, it is being used simply as a
        // somewhat random number that is likely to vary from run to run.
        const size_t total_mem = prodlib::getAvailableSystemMemoryInBytes();
        m_filename = std::string( "test_PrintManager_" ) + std::to_string( total_mem ) + std::string( ".log" );
    }

    void TearDown()
    {
        if( m_context )
            m_context->destroy();

        // Remove temporary file if it exists.
        if( FILE* file = fopen( m_filename.c_str(), "r" ) )
        {
            fclose( file );
            if( remove( m_filename.c_str() ) != 0 )
                fprintf( stderr, "Error deleting temporary log file.\n" );
        }
    }

    void setupProgram( const char* ptxFile, const char* raygenName );
    void setupProgramFromPTXString( const char* ptxString, const char* raygenName );
    void startCapture();
    void getOutputString();
    void endCapture();

    std::string launch( int width );

    // TEMPORARY: Launch the kernel without redirecting stdout to a file.
    // Device prints should appear in the robot log. This is to help diagnose
    // issues with the test machines. See lwbugs/3521134
    void launchNoCapture( int width );

  private:
    int         m_file;
    int         m_stdout;
    std::string m_outputString;
    std::string m_filename = "test_PrintManager.log";
};
