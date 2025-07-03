// Copyright (c) 2019, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <corelib/compiler/PeepholePass.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <gtest/gtest.h>

using namespace llvm;

class TestPeepholePass : public testing::Test
{
  public:
    void Test( const char* source, const char* expected )
    {
        // Parse IR
        std::unique_ptr<Module> module( parseIR( source ) );
        if( module )
        {
            // Run the peephole optimizer
            legacy::PassManager MPM;
            MPM.add( corelib::createPeepholePass() );
            MPM.run( *module );

            // Compare to expected IR.
            std::string        sourceOut;
            raw_string_ostream stream( sourceOut );
            stream << *module;
            EXPECT_STREQ( expected, stream.str().c_str() );

            verifyModule( *module );  // aborts on failure
        }
    }

  private:
    LLVMContext m_context;

    // Parse LLVM IR from string.
    std::unique_ptr<Module> parseIR( const char* source )
    {
        std::unique_ptr<MemoryBuffer> mb( MemoryBuffer::getMemBuffer( source ) );
        SMDiagnostic err;
        std::unique_ptr<Module> module( llvm::parseIR( mb->getMemBufferRef(), err, m_context ) );
        if( !module )
        {
            err.print( "" /*filename*/, llvm::errs() );
        }
        return module;
    }
};

TEST_F( TestPeepholePass, TestAddressArith )
{
    Test(
        "define <4 x i32>* @test() {"
        "  %r1 = alloca [64 x i8], align 16"
        "  %r2 = ptrtoint [64 x i8]* %r1 to i64"
        "  %r3 = add i64 16, %r2"
        "  %r4 = inttoptr i64 %r3 to <4 x i32>*"
        "  ret <4 x i32>* %r4"
        "}",
        "\n"
        "define <4 x i32>* @test() {\n"
        "  %r1 = alloca [64 x i8], align 16\n"
        "  %1 = bitcast [64 x i8]* %r1 to <4 x i32>*\n"
        "  %r4 = getelementptr inbounds <4 x i32>, <4 x i32>* %1, i64 1\n"
        "  ret <4 x i32>* %r4\n"
        "}\n" );

    Test(
        "define <4 x i32>* @test() {"
        "  %r1 = alloca [64 x i8], align 16"
        "  %r2 = ptrtoint [64 x i8]* %r1 to i64"
        "  %r3 = add i64 17, %r2"
        "  %r4 = inttoptr i64 %r3 to <4 x i32>*"
        "  ret <4 x i32>* %r4"
        "}",
        "\n"
        "define <4 x i32>* @test() {\n"
        "  %r1 = alloca [64 x i8], align 16\n"
        "  %1 = bitcast [64 x i8]* %r1 to i8*\n"
        "  %2 = getelementptr inbounds i8, i8* %1, i64 17\n"
        "  %r4 = bitcast i8* %2 to <4 x i32>*\n"
        "  ret <4 x i32>* %r4\n"
        "}\n" );

    Test(
        "declare i64 @foo()"
        "define float* @test() {"
        "  %r1 = alloca [64 x i8], align 16"
        "  %r2 = ptrtoint [64 x i8]* %r1 to i64"
        "  %r3 = add i64 %r2, 32"
        "  %r4 = call i64 @foo()"
        "  %r5 = add i64 %r3, %r4"
        "  %r6 = inttoptr i64 %r5 to float*"
        "  ret float* %r6"
        "}",
        "\n"
        "declare i64 @foo()\n"
        "\n"
        "define float* @test() {\n"
        "  %r1 = alloca [64 x i8], align 16\n"
        "  %r4 = call i64 @foo()\n"
        "  %1 = add i64 32, %r4\n"
        "  %2 = bitcast [64 x i8]* %r1 to i8*\n"
        "  %3 = getelementptr inbounds i8, i8* %2, i64 %1\n"
        "  %r6 = bitcast i8* %3 to float*\n"
        "  ret float* %r6\n"
        "}\n" );

    Test(
        "define <4 x i32>* @test() {"
        "  %r1 = alloca [64 x i8], align 16"
        "  %r2 = ptrtoint [64 x i8]* %r1 to i64"
        "  %r3 = add i64 %r2, 16"
        "  %r4 = inttoptr i64 %r3 to <4 x i32>*"
        "  ret <4 x i32>* %r4"
        "}",
        "\n"
        "define <4 x i32>* @test() {\n"
        "  %r1 = alloca [64 x i8], align 16\n"
        "  %1 = bitcast [64 x i8]* %r1 to <4 x i32>*\n"
        "  %r4 = getelementptr inbounds <4 x i32>, <4 x i32>* %1, i64 1\n"
        "  ret <4 x i32>* %r4\n"
        "}\n" );

    Test(
        "define <4 x i32>* @test() {"
        "  %r1 = alloca [64 x i8], align 16"
        "  %r2 = ptrtoint [64 x i8]* %r1 to i64"
        "  %r3 = add i64 %r2, 16"
        "  %r4 = add i64 32, %r3"
        "  %r5 = inttoptr i64 %r4 to <4 x i32>*"
        "  ret <4 x i32>* %r5"
        "}",
        "\n"
        "define <4 x i32>* @test() {\n"
        "  %r1 = alloca [64 x i8], align 16\n"
        "  %1 = bitcast [64 x i8]* %r1 to <4 x i32>*\n"
        "  %r5 = getelementptr inbounds <4 x i32>, <4 x i32>* %1, i64 3\n"
        "  ret <4 x i32>* %r5\n"
        "}\n" );

    Test(
        "define <4 x i32>* @test() {"
        "  %r1 = alloca [64 x i8], align 16"
        "  %r2 = ptrtoint [64 x i8]* %r1 to i64"
        "  %r3 = inttoptr i64 %r2 to <4 x i32>*"
        "  ret <4 x i32>* %r3"
        "}",
        "\n"
        "define <4 x i32>* @test() {\n"
        "  %r1 = alloca [64 x i8], align 16\n"
        "  %r3 = bitcast [64 x i8]* %r1 to <4 x i32>*\n"
        "  ret <4 x i32>* %r3\n"
        "}\n" );
}

TEST_F( TestPeepholePass, TestBitcast )
{
    Test(
        "define float* @test(i32* %p) {"
        "  %r1 = bitcast i32* %p to i64*"
        "  %r2 = bitcast i64* %r1 to float*"
        "  ret float* %r2"
        "}",
        "\n"
        "define float* @test(i32* %p) {\n"
        "  %r2 = bitcast i32* %p to float*\n"
        "  ret float* %r2\n"
        "}\n" );
}
