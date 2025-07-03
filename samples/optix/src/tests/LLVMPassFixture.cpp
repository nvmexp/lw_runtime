// Copyright (c) 2020, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <tests/LLVMPassFixture.h>

namespace {

std::string loadFile( const std::string& filePath )
{
    std::ifstream fileToRead( filePath );
    if( !fileToRead.is_open() )
        return "";

    std::ostringstream fileContents;
    fileContents << fileToRead.rdbuf();
    return fileContents.str();
}

}  // namespace

namespace optix {

std::unique_ptr<Module> LLVMPassFixture::parseIR( const std::string& source )
{
    std::unique_ptr<MemoryBuffer> mb( MemoryBuffer::getMemBuffer( source.c_str() ) );
    SMDiagnostic                  err;
    std::unique_ptr<Module>       module( llvm::parseIR( mb->getMemBufferRef(), err, m_context ) );
    if( !module )
        err.print( "" /*filename*/, llvm::errs() );
    return module;
}

void LLVMPassFixture::SetUp()
{
    std::string inputPath = dataPath() + "/" + getTestIrDirectory() + "/" + m_testName + "-input.ll";

    std::string inputSource = loadFile( inputPath );
    if( inputSource.empty() )
        FAIL() << "Unable to load source from file " << inputPath;

    m_inputModule = parseIR( inputSource );
    if( !m_inputModule )
        FAIL() << "Unable to parse input module.";
}

void LLVMPassFixture::runPassOnInputModule()
{
    legacy::PassManager PM;
    PM.add( createPassToTest() );
    PM.run( *m_inputModule );
}

void LLVMPassFixture::verifyPassResults()
{
    // Make sure the pass didn't break the input module.
    verifyModule( *m_inputModule );

    // Dump the result to a string for comparison.
    std::string        resultSource;
    raw_string_ostream resultStream( resultSource );
    resultStream << *m_inputModule;

    // Make sure the result matches what we expect.
    std::string expectedPath   = dataPath() + "/" + getTestIrDirectory() + "/" + m_testName + "-expected.ll";
    std::string expectedSource = loadFile( expectedPath );
    if( expectedSource.empty() )
        FAIL() << "Unable to load source from file " << expectedPath;

    EXPECT_STREQ( expectedSource.c_str(), resultStream.str().c_str() );
}

}  // namespace optix
