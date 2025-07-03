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

// This Pass has been inspired by the patch submitted to llvm-commits by Eli Bendersky:
// http://lists.cs.uiuc.edu/pipermail/llvm-commits/Week-of-Mon-20140324/210384.html
// Note that the patch has not been submitted to mainline LLVM.

#include <ExelwtionStrategy/Compile.h>

#include <llvm/Support/raw_ostream.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

#include <corelib/compiler/CoreIRBuilder.h>
#include <corelib/compiler/LLVMUtil.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/math/Bits.h>
#include <prodlib/misc/TimeViz.h>

using namespace llvm;
using namespace optix;

// Support functions.
static Function* getPrintfFunction( Module* module );
static Function* insertVprintfDeclaration( Module* module );
static int computeVprintfBufferSize( CallInst* printfCall, const DataLayout& dataLayout );
static Value* createAndFillVprintfBuffer( CallInst* printfCall, int bufferSize, const DataLayout& dataLayout );

// Constant data.
static int               VPRINTF_BUFFER_ALIGMENT = 8;  // In Bytes.
static int               VPRINTF_BUFFER_ROUND_UP = 8;  // In Bytes.
static const std::string VPRINTF_NAME            = "vprintf";
static const std::string RT_PRINTF_NAME          = "rt_printf";

// -----------------------------------------------------------------------------
void optix::printfToVprintf( Module* module )
{
    using Types = Type* [];

    Function* printf  = getPrintfFunction( module );
    Function* vprintf = insertVprintfDeclaration( module );

    // Printf function is not used by the module, do nothing.
    if( printf == nullptr )
        return;

    std::vector<CallInst*> toDelete;
    DataLayout             dataLayout( module );

    // Go over the calls to printf.
    for( auto I = printf->user_begin(), E = printf->user_end(); I != E; ++I )
    {
        CallInst* printfCall = dyn_cast<CallInst>( *I );
        RT_ASSERT_MSG( printfCall != nullptr, "rt_printf can only be used in a call instruction" );
        corelib::CoreIRBuilder irb{printfCall};

        toDelete.push_back( printfCall );

        int bufferSize = computeVprintfBufferSize( printfCall, dataLayout );

        Value* buffer = nullptr;
        if( bufferSize == 0 )
        {
            // If the buffer is empty we pass a nullptr to vprintf.
            buffer = ConstantPointerNull::get( irb.getInt8PtrTy() );
        }
        else
        {
            buffer = createAndFillVprintfBuffer( printfCall, bufferSize, dataLayout );
            buffer = irb.CreateBitCast( buffer, irb.getInt8PtrTy() );
        }

        RT_ASSERT( buffer );

        Value* formatString = printfCall->getArgOperand( 0 );
        // Generate the alternative call to vprintf and replace the original.
        FunctionType* vprintfType = vprintf->getFunctionType();
        FunctionType* lwrrent_vprintfType =
            FunctionType::get( irb.getInt32Ty(), Types{formatString->getType(), buffer->getType()}, false );
        Value* call = vprintf;
        if( vprintfType != lwrrent_vprintfType )
        {
            RT_ASSERT( vprintf->getType()->isPointerTy() );
            call = ConstantExpr::getBitCast( vprintf, PointerType::get( lwrrent_vprintfType, 0 ) );
        }
        Value*    vprintfArgs[] = {formatString, buffer};
        CallInst* vprintfCall   = irb.CreateCall( call, vprintfArgs );

        printfCall->replaceAllUsesWith( vprintfCall );
    }

    for( const auto& printfCall : toDelete )
        printfCall->eraseFromParent();
}

// -----------------------------------------------------------------------------
static Function* getPrintfFunction( Module* module )
{
    Function* printf = module->getFunction( RT_PRINTF_NAME );
    if( printf == nullptr )
        return nullptr;

    LLVMContext& context = module->getContext();

    // Check the type of the function.
    FunctionType* functionType = printf->getFunctionType();
    RT_ASSERT( functionType->getNumParams() == 1 && functionType->isVarArg()
               && functionType->getReturnType() == Type::getInt32Ty( context )
               && functionType->getParamType( 0 ) == Type::getInt8PtrTy( context )
               && "rt_printf has the wrong signature" );

    return printf;
}

// -----------------------------------------------------------------------------
static Function* insertVprintfDeclaration( Module* module )
{
    Function*    vprintf = module->getFunction( VPRINTF_NAME );
    LLVMContext& context = module->getContext();

    Type*         arg8Types[] = {Type::getInt8PtrTy( context ), Type::getInt8PtrTy( context )};
    FunctionType* vprintfType = FunctionType::get( Type::getInt32Ty( context ), arg8Types, false );

    if( vprintf != nullptr )
    {
        // If we already have a vprintf function in the module check that the type matches.
        if( vprintf->getFunctionType() != vprintfType )
        {
            Type*         arg64Types[] = {Type::getInt64Ty( context ), Type::getInt64Ty( context )};
            FunctionType* vprintfType2 = FunctionType::get( Type::getInt32Ty( context ), arg64Types, false );
            if( vprintf->getFunctionType() != vprintfType2 )
            {
                std::string              os;
                llvm::raw_string_ostream o( os );
                o << "Type of vprintf (" << *vprintf->getFunctionType() << ") doesn't match either \"" << *vprintfType
                  << "\" nor \"" << *vprintfType2 << "\"\n";
                throw prodlib::AssertionFailure( RT_EXCEPTION_INFO, os );
            }
        }
        return vprintf;
    }

    GlobalValue::LinkageTypes externalLinkage = Function::ExternalLinkage;
    return Function::Create( vprintfType, externalLinkage, VPRINTF_NAME, module );
}

// -----------------------------------------------------------------------------
static int computeVprintfBufferSize( CallInst* printfCall, const DataLayout& dataLayout )
{
    int bufferSize = 0;
    for( int operandIndex = 1, operandEnd = printfCall->getNumArgOperands(); operandIndex < operandEnd; ++operandIndex )
    {
        Value* operand     = printfCall->getArgOperand( operandIndex );
        Type*  operandType = operand->getType();
        RT_ASSERT( operandType->isSized() );
        // The size of the buffer has to be updated for each argument to manage cases like this:
        // i32 i64 i32, which needs a bigger size than the case: i32 i32 i64.
        bufferSize = prodlib::align( bufferSize, dataLayout.getPrefTypeAlignment( operandType ) );
        // This assumes that types have already been promoted i.e., char -> int, float -> double.
        bufferSize += dataLayout.getTypeAllocSize( operandType );
    }
    bufferSize = prodlib::align( bufferSize, VPRINTF_BUFFER_ROUND_UP );

    return bufferSize;
}

// -----------------------------------------------------------------------------
static Value* createAndFillVprintfBuffer( CallInst* printfCall, int bufferSize, const DataLayout& dataLayout )
{
    corelib::CoreIRBuilder irb{corelib::getFirstNonAlloca( printfCall->getParent()->getParent() )};
    AllocaInst* buffer = irb.CreateAlloca( ArrayType::get( irb.getInt8Ty(), bufferSize ), nullptr, "vprintf.buffer" );
    buffer->setAlignment( VPRINTF_BUFFER_ALIGMENT );

    irb.SetInsertPoint( printfCall );
    int offset = 0;
    for( int operandIndex = 1, operandEnd = printfCall->getNumArgOperands(); operandIndex < operandEnd; ++operandIndex )
    {
        Value* operand     = printfCall->getArgOperand( operandIndex );
        Type*  operandType = operand->getType();
        RT_ASSERT( operandType->isSized() );

        if( offset != 0 )
            offset = prodlib::align( offset, dataLayout.getPrefTypeAlignment( operandType ) );

        Value* values[2] = {irb.getInt32( 0 ), irb.getInt32( offset )};
        Value* gepInst   = irb.CreateGEP( buffer, values, "arg.pointer" );

        Value* castInst = irb.CreateBitCast( gepInst, operandType->getPointerTo(), "arg.pointer.casted" );
        irb.CreateStore( operand, castInst, false );

        offset += dataLayout.getTypeAllocSize( operandType );
    }

    return buffer;
}
