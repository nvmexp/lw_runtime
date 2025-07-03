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

#include <ExelwtionStrategy/RTX/RTXDemandBufferSpecializer.h>

#include <ExelwtionStrategy/RTX/RTXIntrinsics.h>
#include <Util/ContainerAlgorithm.h>

#include <srcTests.h>

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <algorithm>
#include <string>

using namespace optix;
using namespace testing;

namespace {

class RTXDemandBufferSpecializerTest : public Test
{
  public:
    void SetUp() override
    {
        declareOutputFunction();
        createCallToInputFunction();
    }

    RTXDemandBufferSpecializer m_specializer;
    llvm::LLVMContext&         m_context = llvm::getGlobalContext();
    llvm::Module               m_module{"test", m_context};
    llvm::Type*                m_i32Ty      = llvm::Type::getInt32Ty( m_context );
    llvm::Type*                m_i64Ty      = llvm::Type::getInt64Ty( m_context );
    llvm::Type*                m_statePtrTy = llvm::PointerType::get( m_i32Ty, 1 );
    llvm::IRBuilder<>          m_builder{m_context};
    llvm::Function*            m_parentFn = createParentFunction();
    std::string m_inputFnName  = RtxiLoadOrRequestBufferElement::getFunctionName( "foo_ptx0xdeadbeef.other_stuff" );
    std::string m_outputFnName = "RTX_requestBufferElement2";

    llvm::Function* createParentFunction()
    {
        std::vector<llvm::Type*> paramTys{m_statePtrTy};
        llvm::FunctionType*      fnType = llvm::FunctionType::get( m_i32Ty, paramTys, false );
        std::string              fnName{"parent_fn"};
        llvm::Function*          fn = llvm::cast<llvm::Function>( m_module.getOrInsertFunction( fnName, fnType ) );
        llvm::BasicBlock*        entryBlock = llvm::BasicBlock::Create( m_context, "entry", fn );
        m_builder.SetInsertPoint( entryBlock );
        return fn;
    }

    void declareOutputFunction()
    {
        std::vector<llvm::Type*> paramTys{m_statePtrTy, m_i32Ty, m_i32Ty, m_i64Ty, m_i64Ty};  // statePtr, bufferId, elementSize, x, y
        llvm::FunctionType* fnType = llvm::FunctionType::get( llvm::Type::getInt8PtrTy( m_context ), paramTys, false );
        m_module.getOrInsertFunction( m_outputFnName, fnType );
    }

    void createCallToInputFunction()
    {
        // statePtr, bufferId, elementSize, ptr, x, y
        std::vector<llvm::Type*> paramTys{m_statePtrTy, m_i32Ty, m_i32Ty, m_i64Ty, m_i64Ty, m_i64Ty};
        llvm::FunctionType*      fnType = llvm::FunctionType::get( m_i32Ty, paramTys, false );
        llvm::Function* fn       = llvm::cast<llvm::Function>( m_module.getOrInsertFunction( m_inputFnName, fnType ) );
        llvm::Value*    statePtr = llvm::ConstantPointerNull::get( llvm::cast<llvm::PointerType>( m_statePtrTy ) );
        llvm::Value*    bufferId = llvm::UndefValue::get( m_i32Ty );
        llvm::Value*    elementSize = llvm::ConstantInt::get( m_i32Ty, 4, false /*isSigned*/ );
        llvm::Value*    ptr         = llvm::UndefValue::get( m_i64Ty );
        llvm::Value*    x           = llvm::UndefValue::get( m_i64Ty );
        llvm::Value*    y           = llvm::UndefValue::get( m_i64Ty );
        std::vector<llvm::Value*> args{statePtr, bufferId, elementSize, ptr, x, y};
        m_builder.SetInsertPoint( m_parentFn->begin() );
        llvm::Value* callInst = m_builder.CreateCall( fn, args );
        m_builder.CreateRet( callInst );
    }

    static unsigned int basicBlockCount( const llvm::Function* fn ) { return std::distance( fn->begin(), fn->end() ); }

    static bool containsCallToFunction( const llvm::BasicBlock& block, const std::string& name )
    {
        for( const llvm::Instruction& inst : block )
        {
            if( const llvm::CallInst* callInst = llvm::dyn_cast_or_null<llvm::CallInst>( &inst ) )
            {
                if( name == callInst->getCalledFunction()->getName() )
                    return true;
            }
        }
        return false;
    }
};

}  // namespace

TEST_F( RTXDemandBufferSpecializerTest, module_verifies_after_specialization )
{
    m_specializer.runOnFunction( m_parentFn );

    ASSERT_FALSE( llvm::verifyModule( m_module ) );
}

TEST_F( RTXDemandBufferSpecializerTest, replaces_call_to_intrinsic_with_if_then_else_basic_blocks )
{
    ASSERT_EQ( 1U, basicBlockCount( m_parentFn ) );

    m_specializer.runOnFunction( m_parentFn );

    ASSERT_EQ( 4U, basicBlockCount( m_parentFn ) );
}

TEST_F( RTXDemandBufferSpecializerTest, call_instruction_to_input_intrinsic_removed )
{
    m_specializer.runOnFunction( m_parentFn );

    for( const llvm::BasicBlock& block : *m_parentFn )
    {
        ASSERT_FALSE( containsCallToFunction( block, m_inputFnName ) );
    }
}

TEST_F( RTXDemandBufferSpecializerTest, first_basic_block_calls_output_function )
{
    m_specializer.runOnFunction( m_parentFn );

    const llvm::BasicBlock& firstBlock = *m_parentFn->begin();
    ASSERT_TRUE( containsCallToFunction( firstBlock, m_outputFnName ) );
}
