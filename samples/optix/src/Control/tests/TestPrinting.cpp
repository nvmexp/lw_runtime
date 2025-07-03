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

#include <Control/tests/TestPrinting.h>

void TestPrintingNoKnobs::setupProgram( const char* ptxFile, const char* raygenName )
{
    // Delay context creation until this point such that overrides for the
    // exelwtion strategy are already in place.
    m_context = Context::create();
    m_context->setRayTypeCount( 1 );
    m_context->setEntryPointCount( 1 );

    std::string ptx_path( ptxPath( "test_Control", ptxFile ) );

    ASSERT_TRUE( raygenName );
    Program rayGen = m_context->createProgramFromPTXFile( ptx_path, raygenName );
    m_context->setRayGenerationProgram( 0, rayGen );

    m_input = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 13 );
    m_context["input"]->set( m_input );

    // Enable printing.
    m_context->setPrintEnabled( true );

    int* input = static_cast<int*>( m_input->map() );
    int  val   = 13;
    input[0]   = val;
    m_input->unmap();
}

void TestPrintingNoKnobs::setupProgramFromPTXString( const char* ptxString, const char* raygenName )
{
    // Delay context creation until this point such that overrides for the
    // exelwtion strategy are already in place.
    m_context = Context::create();
    m_context->setRayTypeCount( 1 );
    m_context->setEntryPointCount( 1 );

    ASSERT_TRUE( raygenName );
    Program rayGen = m_context->createProgramFromPTXString( ptxString, raygenName );
    m_context->setRayGenerationProgram( 0, rayGen );

    // Enable printing.
    m_context->setPrintEnabled( true );

    // Need to create buffer to avoid initialization bug. (?)
    m_input = m_context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_INT, 13 );
}

#define FD_STDOUT 1
void TestPrintingNoKnobs::startCapture()
{
    int ret = 0;
    // Flush everything before redirection.
    ret = fflush( stdout );
    ASSERT_TRUE( ret == 0 );

    // Store the original stdout file descriptor.
    m_stdout = dup( FD_STDOUT );
    ASSERT_TRUE( m_stdout >= 0 );

    // Open a file stream as the new stdout target.
    m_file = open( m_filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0660 );
    ASSERT_TRUE( m_file >= 0 );

    // Create an alias for stdout and close the original one.
    ret = dup2( m_file, FD_STDOUT );
    ASSERT_TRUE( ret >= 0 );

    // Write a sentinel character to stdout.
    fprintf( stdout, "^" );

    // Flush the sentinel character to ensure that it's the first thing in the
    // captured output.
    ret = fflush( stdout );
    ASSERT_TRUE( ret == 0 );
}

void TestPrintingNoKnobs::getOutputString()
{
    // Read the file contents into a string.
    std::ifstream     file( m_filename );
    std::stringstream buffer;
    buffer << file.rdbuf();

    // Check the string for the start and end sentinels.
    const std::string rawString = buffer.str();
    ASSERT_TRUE( rawString.size() >= 2 );
    ASSERT_TRUE( rawString.front() == '^' && rawString.back() == '$' );

    // Return the string with the sentinel characters stripped.
    m_outputString = rawString.substr( 1, rawString.size() - 2 );
}

void TestPrintingNoKnobs::endCapture()
{
    // Write a sentinel character to stdout.
    fprintf( stdout, "$" );
    
    int ret = 0;
    // Flush everything to the file.
    ret = fflush( stdout );
    ASSERT_TRUE( ret == 0 );

    // Restore the original stdout from the stored file descriptor.
    ret = dup2( m_stdout, FD_STDOUT );
    ASSERT_TRUE( ret >= 0 );

    // We don't need the file descriptor anymore.
    ret = close( m_file );
    ASSERT_TRUE( ret == 0 );

    // We don't need the file descriptor of the copied stdout anymore.
    ret = close( m_stdout );
    ASSERT_TRUE( ret == 0 );
}
#undef FD_STDOUT

std::string TestPrintingNoKnobs::launch( int width )
{
    startCapture();
    m_context->launch( 0, width );
    endCapture();
    getOutputString();
    return m_outputString;
}

void TestPrintingNoKnobs::launchNoCapture( int width )
{
    fflush( stdout );
    m_context->launch( 0, width );
    fflush( stdout );
}
