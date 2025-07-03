//
// Copyright (c) 2019, LWPU CORPORATION.
//
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES
//

#pragma once

#include <srcTests.h>

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <string>

namespace optix {
namespace testing {

// Test fixture for testing intrinsics
class IntrinsicNameTest : public ::testing::Test
{
  public:
    explicit IntrinsicNameTest( const std::string& name )
        : IntrinsicNameTest( "", name, "" )
    {
    }
    IntrinsicNameTest( const std::string& prefix, const std::string& suffix )
        : IntrinsicNameTest( prefix, prefix + suffix, suffix )
    {
    }

    llvm::Function* createFunction( const std::string& name )
    {
        return llvm::Function::Create( m_fnType, llvm::Function::ExternalLinkage, name.c_str() );
    }
    llvm::Function* createFunction() { return createFunction( m_name ); }

    llvm::CallInst* createCall( const std::string& name )
    {
        llvm::Function* fn = createFunction( name );
        return m_builder.CreateCall( fn );
    }
    llvm::CallInst* createCall() { return createCall( m_name ); }

    llvm::Value* createValue( const std::string& name ) { return createCall( name ); }

    llvm::Value* createValue() { return createValue( m_name ); }

    llvm::LLVMContext m_context;
    llvm::IRBuilder<>   m_builder;
    llvm::FunctionType* m_fnType = llvm::FunctionType::get( llvm::Type::getVoidTy( m_context ), false );
    llvm::BasicBlock*   m_block  = llvm::BasicBlock::Create( m_context, "entry" );
    const std::string   m_prefix;
    const std::string   m_suffix;
    const std::string   m_name;

  private:
    IntrinsicNameTest( const std::string& prefix, const std::string& name, const std::string& suffix )
        : m_builder( m_context )
        , m_prefix( prefix )
        , m_suffix( suffix )
        , m_name( name )
    {
        m_builder.SetInsertPoint( m_block );
    }
};

// Test fixture for testing intrinsics with unique name suffixes
class UniqueNameTest : public IntrinsicNameTest
{
  public:
    UniqueNameTest( const std::string& prefix, const std::string& uniqueName, const char* uniqueNameSuffix )
        : IntrinsicNameTest( prefix, '.' + uniqueName + uniqueNameSuffix )
        , m_uniqueName( uniqueName )
        , m_uniqueNameSuffix( uniqueNameSuffix )
    {
    }

    const std::string m_uniqueName;
    const std::string m_uniqueNameSuffix;
};

// Test fixture for testing intrinsics with variable reference unique names
class VariableReferenceUniqueNameTest : public UniqueNameTest
{
  public:
    VariableReferenceUniqueNameTest( const std::string& prefix, const char* suffix = "" )
        : UniqueNameTest( prefix, "arbitrary_text_ptx0xdeadbeef.no_more_dots", suffix )
    {
    }
};

// Test fixture for testing intrinsics with canonical program unique names
class CanonicalProgramUniqueNameTest : public UniqueNameTest
{
  public:
    CanonicalProgramUniqueNameTest( const std::string& prefix )
        : UniqueNameTest( prefix, "arbitrary_text_ptx0xdeadbeef", ".prd666b" )
    {
    }
};

}  // namespace testing
}  // namespace optix
