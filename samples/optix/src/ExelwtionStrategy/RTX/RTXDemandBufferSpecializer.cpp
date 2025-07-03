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

#include <corelib/compiler/LLVMUtil.h>
#include <prodlib/exceptions/Assert.h>

#include <vector>

namespace optix {

void RTXDemandBufferSpecializer::runOnFunction( llvm::Function* function )
{
    llvm::Module*                module = function->getParent();
    std::vector<llvm::CallInst*> toDelete;
    for( llvm::Function* fn : corelib::getFunctions( module ) )
    {
        if( !RtxiLoadOrRequestBufferElement::isIntrinsic( fn ) )
            continue;

        for( llvm::CallInst* callInst : corelib::getCallsToFunction( fn ) )
        {
            runOnCall( module, callInst, toDelete );
        }
    }
    for( llvm::CallInst* callInst : toDelete )
    {
        callInst->eraseFromParent();
    }
}

void RTXDemandBufferSpecializer::runOnCall( llvm::Module* module, llvm::CallInst* callInst, std::vector<llvm::CallInst*>& toDelete )
{
    RtxiLoadOrRequestBufferElement* element = llvm::cast<RtxiLoadOrRequestBufferElement>( callInst );
    corelib::CoreIRBuilder          builder{callInst};
    llvm::Value*                    elementSize = element->getElementSize();
    std::vector<llvm::Value*>       args;
    args.push_back( element->getStatePtr() );
    args.push_back( element->getBufferId() );
    args.push_back( elementSize );
    args.push_back( element->getX() );
    const unsigned int dimensionality = element->getDimensionality();
    if( dimensionality > 1 )
        args.push_back( element->getY() );
    if( dimensionality > 2 )
        args.push_back( element->getZ() );
    llvm::Function* fn = module->getFunction( ( "RTX_requestBufferElement" + std::to_string( dimensionality ) ).c_str() );
    RT_ASSERT( fn != nullptr );
    llvm::Value* fnResult = builder.CreateCall( fn, args, "element" );

    const unsigned int elementSizeInBytes = corelib::getConstantValueOrAssert( elementSize );

    llvm::Type* elementType = llvm::ArrayType::get( llvm::Type::getInt8Ty( callInst->getContext() ), elementSizeInBytes );
    llvm::PointerType* elementPtrType = llvm::PointerType::get( elementType, corelib::AddressSpace::ADDRESS_SPACE_GENERIC );

    llvm::Value* destPtr = builder.CreateIntToPtr( element->getPtr(), elementPtrType, "destPtr" );

    llvm::Value* castedResult = builder.CreatePointerCast( fnResult, elementPtrType );

    llvm::Value* zeroValue  = llvm::ConstantPointerNull::get( elementPtrType );
    llvm::Value* condition  = builder.CreateICmp( llvm::CmpInst::ICMP_NE, castedResult, zeroValue );
    llvm::Value* isResident = builder.CreateZExt( condition, llvm::Type::getInt32Ty( callInst->getContext() ) );

    llvm::TerminatorInst*      thenTerm   = nullptr;
    llvm::TerminatorInst*      elseTerm   = nullptr;
    llvm::Instruction*         splitPoint = nullptr;
    llvm::Instruction*         cmpInst    = llvm::cast<llvm::Instruction>( condition );
    llvm::BasicBlock::iterator it( cmpInst );
    ++it;
    llvm::SplitBlockAndInsertIfThenElse( condition, &*it, &thenTerm, &elseTerm );

    corelib::CoreIRBuilder thenBuilder{thenTerm};
    llvm::Value*           loadElement = thenBuilder.CreateLoad( castedResult );
    thenBuilder.CreateStore( loadElement, destPtr );

    callInst->replaceAllUsesWith( isResident );
    RT_ASSERT( callInst->use_empty() );

    toDelete.push_back( callInst );
}

}  // namespace optix
