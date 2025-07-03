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

#include <FrontEnd/PTX/LowerCarryInstructionsPass.h>

#include <FrontEnd/Canonical/FrontEndHelpers.h>
#include <prodlib/exceptions/Assert.h>

#include <corelib/compiler/CoreIRBuilder.h>
#include <corelib/compiler/LLVMUtil.h>

#include <llvm/Pass.h>

namespace {

// Helper functions for strings
bool startsWith( const std::string& someString, const std::string& prefix )
{
    return someString.size() >= prefix.size() && someString.substr( 0, prefix.size() ).compare( prefix ) == 0;
}

bool endsWith( const std::string& someString, const std::string& suffix )
{
    if( someString.size() < suffix.size() )
        return false;

    size_t startIndex = someString.size() - suffix.size();
    size_t numChars   = someString.size() - startIndex;
    return someString.substr( startIndex, numChars ).compare( suffix ) == 0;
}

std::string removePrefix( const std::string& someString, const std::string& prefix )
{
    return someString.substr( prefix.size(), someString.size() - prefix.size() );
}

std::vector<std::string> split( const std::string& someString, const std::string& pattern )
{
    std::vector<std::string> splitString;

    size_t start = 0;
    size_t end   = 0;
    while( end != std::string::npos )
    {
        end             = someString.find( pattern, start );
        size_t numChars = end == std::string::npos ? someString.size() - start : end - start;
        splitString.push_back( someString.substr( start, numChars ) );
        start = end + 1;
    }

    return splitString;
}

// If true, look for and replace raw inline PTX, otherwise look for
// optix.ptx... wrappers
#define PARSE_ASM_DIRECTLY 0

class LowerCarryInstructionsPass : public llvm::FunctionPass
{
  public:
    static char ID;

    LowerCarryInstructionsPass()
        : llvm::FunctionPass( ID ){};
    bool runOnFunction( llvm::Function& function ) override;
};

char LowerCarryInstructionsPass::ID = 0;

typedef enum class ArithmeticOperation { UNKNOWN = 0, ADD, SUB, MAD } ArithmeticOperation;

struct PtxCarryInstruction
{
    ArithmeticOperation op;
    bool                hasCarryIn;
    bool                hasCarryOut;
    bool                isSigned;
    bool                isHi;
};

/*
 * Retrieve PTX instruction from given inline ASM string.
 *
 * Example:
 *      " addc.s16 $0, $1, $2;"
 * Becomes:
 *     "addc.s16"
 */
std::string getInstructionFromInlineAsm( const std::string& asmString )
{
    // Inline asm might have beginning whitespace
    size_t nameStart = asmString.find_first_not_of( ' ' );
    size_t nameEnd   = asmString.find_first_of( ' ', nameStart );
    return asmString.substr( nameStart, nameEnd - nameStart );
}

const std::vector<std::string> carryInstructionNames = {"add.cc", "addc", "sub.cc",    "subc",
                                                        "mad.cc", "madc", "mad.lo.cc", "mad.hi.cc"};

bool isCarryInstruction( const std::string& instruction )
{
    for( const std::string& carryInstName : carryInstructionNames )
    {
        if( startsWith( instruction, carryInstName ) )
            return true;
    }

    return false;
}

bool isInstructionSigned( const std::string& instruction )
{
    // If the instruction's type specifier begins with "s", return true.
    // Example: addc.s16 returns true, addc.u16 returns false
    if( instruction.size() < 3 )
        return false;
    return instruction.substr( instruction.size() - 3, 3 )[0] == 's';
}

ArithmeticOperation getInstructionOp( const std::string& instruction )
{
    if( startsWith( instruction, "add" ) )
        return ArithmeticOperation::ADD;
    else if( startsWith( instruction, "sub" ) )
        return ArithmeticOperation::SUB;
    else if( startsWith( instruction, "mad" ) )
        return ArithmeticOperation::MAD;

    return ArithmeticOperation::UNKNOWN;
}

bool parseInstruction( const std::string& instruction, PtxCarryInstruction& outInstruction )
{
    if( !isCarryInstruction( instruction ) )
        return false;

    // Carry instructions are in the form:
    //   {operation}[c][.cc].[us]{type width}

    std::vector<std::string> components = split( instruction, "." );
    RT_ASSERT( components.size() >= 2 );

    outInstruction.op = getInstructionOp( components[0] );

    if( outInstruction.op == ArithmeticOperation::MAD )
    {
        outInstruction.isHi        = components[1].compare( "hi" ) == 0;
        outInstruction.hasCarryOut = components[2].compare( "cc" ) == 0;
    }
    else
        outInstruction.hasCarryOut = components[1].compare( "cc" ) == 0;

    outInstruction.hasCarryIn = endsWith( components[0], "c" );
    outInstruction.isSigned   = startsWith( components[components.size() - 1], "s" );

    return true;
}

bool parseInlineAsmCarryInstruction( llvm::CallInst* call, PtxCarryInstruction& outInstruction )
{
    if( !call->isInlineAsm() )
        return false;

    llvm::InlineAsm*   inlineAsm   = llvm::dyn_cast<llvm::InlineAsm>( call->getCalledValue() );
    const std::string& asmString   = inlineAsm->getAsmString();
    const std::string  instruction = getInstructionFromInlineAsm( asmString );

    return parseInstruction( instruction, outInstruction );
}

bool parseOptixCarryInstruction( llvm::CallInst* call, PtxCarryInstruction& outInstruction )
{
    if( !startsWith( call->getCalledValue()->getName().str(), "optix.ptx" ) )
        return false;
    const std::string instruction = removePrefix( call->getCalledValue()->getName().str(), "optix.ptx." );

    return parseInstruction( instruction, outInstruction );
}

llvm::Value* addCarryBitToFunction( llvm::LLVMContext& context, corelib::CoreIRBuilder builder, llvm::Function& function, const std::string& name )
{
    builder.SetInsertPoint( &*function.getEntryBlock().getFirstInsertionPt() );

    llvm::Type*  carryBitType    = llvm::Type::getInt1Ty( context );
    llvm::Value* carryAllocaSize = llvm::ConstantInt::get( llvm::Type::getInt32Ty( context ), 1 );
    llvm::Value* carryAlloca     = builder.CreateAlloca( carryBitType, carryAllocaSize, name );

    builder.SetInsertPoint( &*optix::getSafeInsertionPoint( &function ) );
    // Based on observation (but not dolwmented), LWCA initializes the CC.CF register to 1
    builder.CreateStore( llvm::ConstantInt::get( llvm::Type::getInt1Ty( context ), 1 ), carryAlloca );

    return carryAlloca;
}

llvm::Value* getCarryValue( llvm::LLVMContext& context, corelib::CoreIRBuilder builder, llvm::Value* carryAlloca, llvm::CallInst* call )
{
    llvm::Value* lwrrentCarryAsBool = builder.CreateLoad( carryAlloca );
    return builder.CreateZExt( lwrrentCarryAsBool, llvm::Type::getInt32Ty( context ), "lwrrCarry" );
}


llvm::Value* createHiOrLoMultiply( llvm::LLVMContext&         context,
                                   corelib::CoreIRBuilder&    builder,
                                   const PtxCarryInstruction& instruction,
                                   llvm::Value*               leftValue,
                                   llvm::Value*               rightValue )
{
    llvm::Type* i64Ty = llvm::Type::getInt64Ty( context );

    // If the instruction has carry-out, use unsigned intrinsics, because we're
    // adding bytes that don't contain the sign bit.
    const bool useUnsignedIntrinsic = !instruction.isSigned || instruction.hasCarryOut;

    // Assume the values are i32s. Extend them to 64 bits.
    llvm::Value* firstValueAsI64;
    llvm::Value* secondValueAsI64;
    if( useUnsignedIntrinsic )
    {
        firstValueAsI64  = builder.CreateZExt( leftValue, i64Ty );
        secondValueAsI64 = builder.CreateZExt( rightValue, i64Ty );
    }
    else
    {
        firstValueAsI64  = builder.CreateSExt( leftValue, i64Ty );
        secondValueAsI64 = builder.CreateSExt( rightValue, i64Ty );
    }

    llvm::Value* mulResult = builder.CreateMul( firstValueAsI64, secondValueAsI64 );

    // Extract either the high or low bits of the result.
    llvm::Type*  vectorTy    = llvm::VectorType::get( llvm::IntegerType::get( context, 32 ), 2 );
    llvm::Value* resAsVector = builder.CreateBitCast( mulResult, vectorTy );

    int          extractIndex      = instruction.isHi ? 1 : 0;
    llvm::Value* extractIndexValue = llvm::ConstantInt::get( llvm::Type::getInt32Ty( context ), extractIndex );

    return builder.CreateExtractElement( resAsVector, extractIndexValue );
}

bool LowerCarryInstructionsPass::runOnFunction( llvm::Function& function )
{
    llvm::LLVMContext&     context = function.getContext();
    corelib::CoreIRBuilder builder( context );

    // Insert an alloca into the beginning of the function to serve as the
    // "carry bit". We'll rely on mem2reg or DCE to eliminate this later.
    //
    // NOTE: Based on observation, LWCA maintains separate carry bits for carry
    // and borrow out.
    llvm::Value* carryOutAlloca  = nullptr;
    llvm::Value* borrowOutAlloca = nullptr;

    for( llvm::Function::iterator func_it = function.begin(), func_end = function.end(); func_it != func_end; ++func_it )
    {
        // Keep a list of calls to remove, so we don't ilwalidate our iterator.
        std::vector<llvm::CallInst*> callsToRemove;

        for( llvm::BasicBlock::iterator bb_it = func_it->begin(), bb_end = func_it->end(); bb_it != bb_end; ++bb_it )
        {
            llvm::CallInst* call = llvm::dyn_cast<llvm::CallInst>( bb_it );
            if( call == nullptr )
                continue;

            PtxCarryInstruction instruction{};
#if PARSE_ASM_DIRECTLY
            if( !parseInlineAsmCarryInstruction( call, instruction ) )
#else
            if( !parseOptixCarryInstruction( call, instruction ) )
#endif
                continue;

            const bool usesAddIntrinsic = instruction.op == ArithmeticOperation::ADD || instruction.op == ArithmeticOperation::MAD;

            llvm::Value* carryOrBorrowAlloca;
            if( usesAddIntrinsic )
            {
                if( carryOutAlloca == nullptr )
                    carryOutAlloca  = addCarryBitToFunction( context, builder, function, "carryOut" );
                carryOrBorrowAlloca = carryOutAlloca;
            } else
            {
                if( borrowOutAlloca == nullptr )
                    borrowOutAlloca  = addCarryBitToFunction( context, builder, function, "borrowOut" );
                carryOrBorrowAlloca = borrowOutAlloca;
            }

            builder.SetInsertPoint( call );

            // Get the left and right operands for the add or sub operation. If
            // this is a MAD instruction, the add operands are the product of
            // operands a and b, along with operand c. Otherwise, operands a
            // and b are used directly in the add or sub.
            llvm::Value* leftAddOrSubOperand;
            llvm::Value* rightAddOrSubOperand;
            if( instruction.op == ArithmeticOperation::MAD )
            {
                leftAddOrSubOperand =
                    createHiOrLoMultiply( context, builder, instruction, call->getOperand( 0 ), call->getOperand( 1 ) );
                rightAddOrSubOperand = call->getOperand( 2 );
            }
            else
            {
                leftAddOrSubOperand  = call->getOperand( 0 );
                rightAddOrSubOperand = call->getOperand( 1 );
            }

            // If the instruction has carry-out, use unsigned intrinsics, because we're
            // adding bytes that don't contain the sign bit.
            const bool useUnsignedIntrinsic = !instruction.isSigned || instruction.hasCarryOut;

            // Determine which arithmetic operation we should peform on the
            // operands (e.g. signed or unsigned add or sub)
            llvm::Intrinsic::ID intrinsicID;
            if( usesAddIntrinsic )
                intrinsicID = useUnsignedIntrinsic ? llvm::Intrinsic::uadd_with_overflow : llvm::Intrinsic::sadd_with_overflow;
            else
                intrinsicID = useUnsignedIntrinsic ? llvm::Intrinsic::usub_with_overflow : llvm::Intrinsic::ssub_with_overflow;

            // Retrieve the LLVM intrinsic for the add or sub operation, making
            // sure its declaration is present in the module.
            std::vector<llvm::Type*> overloadedArgs = {
                llvm::IntegerType::get( context, leftAddOrSubOperand->getType()->getIntegerBitWidth() )};
            llvm::Function* addOrSubFunc = llvm::Intrinsic::getDeclaration( function.getParent(), intrinsicID, overloadedArgs );
            RT_ASSERT_MSG( addOrSubFunc, "Could't retrieve declaration for add or sub w/ overflow intrinsic." );

            // Apply carry/borrow in. PTX add/sub/mad instructions add it to
            // the second operand. If that add operation overflows, the result
            // is wrapped and the carry flag is set.
            llvm::Value* lwrrentCarry = nullptr;
            if( instruction.hasCarryIn )
            {
                lwrrentCarry = getCarryValue( context, builder, carryOrBorrowAlloca, call );

                llvm::Intrinsic::ID addIntrinsicID =
                    useUnsignedIntrinsic ? llvm::Intrinsic::uadd_with_overflow : llvm::Intrinsic::sadd_with_overflow;
                llvm::Function* addFunc = llvm::Intrinsic::getDeclaration( function.getParent(), addIntrinsicID, overloadedArgs );

                std::vector<llvm::Value*> carryInCallArgs  = {rightAddOrSubOperand, lwrrentCarry};
                llvm::Value*              carryInAddResult = builder.CreateCall( addFunc, carryInCallArgs );

                rightAddOrSubOperand = builder.CreateExtractValue( carryInAddResult, 0, "carryInAddResult" );
                lwrrentCarry         = builder.CreateExtractValue( carryInAddResult, 1, "carryInAddOverflow" );
            }

            std::vector<llvm::Value*> callArgs      = {leftAddOrSubOperand, rightAddOrSubOperand};
            llvm::Value*              addOrSubValue = builder.CreateCall( addOrSubFunc, callArgs );

            llvm::Value* addOrSubResult = builder.CreateExtractValue( addOrSubValue, 0, "addOrSubResult" );
            llvm::Value* carryOut       = builder.CreateExtractValue( addOrSubValue, 1, "addOrSubOverflow" );

            // If we have carry out, store that in the alloca
            if( instruction.hasCarryOut )
            {
                if( lwrrentCarry )
                    carryOut = builder.CreateOr( lwrrentCarry, carryOut );

                builder.CreateStore( carryOut, carryOrBorrowAlloca );
            }

            // Replace all of the inline asm's uses with the new value and mark the old instruction for deletion.
            call->replaceAllUsesWith( addOrSubResult );
            callsToRemove.push_back( call );
        }

        for( llvm::CallInst* call : callsToRemove )
            call->eraseFromParent();
    }

    // We modified the function if we inserted an alloca for the carry bit
    return carryOutAlloca != nullptr || borrowOutAlloca != nullptr;
}
}  // namespace

namespace optix {

bool lowerCarryInstructionsPassHandlesIntrinsic( llvm::StringRef intrinsicName )
{
    const std::string intrinsicInstruction = removePrefix( intrinsicName.str(), "optix.ptx." );
    for( const std::string& carryInstruction : carryInstructionNames )
    {
        if( startsWith( intrinsicInstruction, carryInstruction ) )
            return true;
    }
    return false;
}

llvm::FunctionPass* createLowerCarryInstructionsPass()
{
    return new LowerCarryInstructionsPass();
}

}  // namespace optix
