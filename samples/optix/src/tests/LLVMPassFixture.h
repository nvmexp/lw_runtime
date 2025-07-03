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

#pragma once

#include <srcTests.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

using namespace llvm;

namespace optix {

class LLVMPassFixture : public testing::Test, public testing::WithParamInterface<std::string>
{
  public:
    LLVMPassFixture() { m_testName = GetParam(); }

    void SetUp() override;
    void runPassOnInputModule();
    void verifyPassResults();

    // Each test must specify the subdirectory in which the input and expected
    // IR is located.
    virtual std::string getTestIrDirectory() = 0;

    // Each test must provide a method to create the LLVM pass it uses.
    virtual llvm::Pass* createPassToTest() = 0;

  protected:
    std::string m_testName;

  private:
    LLVMContext       m_context;
    std::unique_ptr<Module> m_inputModule;

    // Parse LLVM IR from string.
    std::unique_ptr<Module> parseIR( const std::string& source );
};

}  // optix


