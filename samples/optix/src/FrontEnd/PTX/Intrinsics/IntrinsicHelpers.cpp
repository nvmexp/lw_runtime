//
//  Copyright (c) 2021 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//

#include <FrontEnd/PTX/Intrinsics/IntrinsicHelpers.h>

#include <FrontEnd/PTX/Intrinsics/PTXToLLVMHelpers.h>
#include <prodlib/exceptions/Assert.h>

namespace optix {
namespace PTXIntrinsics {

InstructionSignature expandTexInstructionSignature( llvm::LLVMContext&           context,
                                                    const InstructionSignature&  signature,
                                                    const PTXIntrinsicModifiers& modifiers,
                                                    const ptxInstructionCode     opCode,
                                                    const bool                   isSparse )
{
    InstructionSignature newSignature = signature;

    // Adjust return type to match vector modifier
    if( modifiers.vectorSize != VectorSize::unspecified )
    {
        const int vecSize = vectorSizeToInt( modifiers.vectorSize );
        newSignature[0].ptxType  = "v" + std::to_string( vecSize ) + "." + newSignature[0].ptxType;
        newSignature[0].llvmType = llvm::VectorType::get( newSignature[0].llvmType, vecSize );
    }

    // Add bool to return type if this is a sparse instruction
    if( isSparse )
    {
        llvm::Type* i1Ty = llvm::IntegerType::get( context, 1 );
        llvm::SmallVector<llvm::Type*, 2> structElements = {newSignature[0].llvmType, i1Ty};
        newSignature[0].llvmType = llvm::StructType::get( context, structElements );
    }

    const int coordVectorSize = getVectorSizeForDimensionality( modifiers.texDim );

    // The first argument to array calls is a struct containing an index plus the coordinates
    // e.g. a2d is {i32, f32, f32, f32}
    if( isArrayDimensionality( modifiers.texDim ) )
    {
        OperandType& firstCoordTy = newSignature[2];

        llvm::Type* arrayIdxType = llvm::IntegerType::get( context, 32 );
        llvm::SmallVector<llvm::Type*, 4> structMembers = {arrayIdxType};

        const int structSize = coordVectorSize == 1 ? 2 : 4;
        for( int i = 1; i < structSize; ++i )
            structMembers.push_back( firstCoordTy.llvmType );

        firstCoordTy.llvmType = llvm::StructType::get( context, structMembers );
    }

    // Adjust all further coordinates to take vectors appropriate for the coordinate size
    if( coordVectorSize > 1 )
    {
        const size_t coordStartIndex = isArrayDimensionality( modifiers.texDim ) ? 3 : 2;
        // tex.level instructions have a trailing scalar argument specifying the mip level
        const size_t coordEndIndex = opCode == ptx_tex_level_Instr ? newSignature.size() - 1 : newSignature.size();
        for( size_t i                = coordStartIndex; i < coordEndIndex; ++i )
            newSignature[i].llvmType = llvm::VectorType::get( newSignature[i].llvmType, coordVectorSize );
    }

    // Adjust derivative inputs so they're all vectors
    if( opCode == ptx_tex_grad_Instr )
    {
        const size_t derivStartIndex = 3;
        const size_t derivEndIndex   = 4;
        for( size_t i = derivStartIndex; i <= derivEndIndex; ++i )
        {
            if( !newSignature[i].llvmType->isVectorTy() && !newSignature[i].llvmType->isStructTy() )
                newSignature[i].llvmType = llvm::VectorType::get( newSignature[i].llvmType, coordVectorSize );
        }
    }

    return newSignature;
}

InstructionSignature expandTld4InstructionSignature( llvm::LLVMContext&           context,
                                                     const InstructionSignature&  signature,
                                                     const PTXIntrinsicModifiers& modifiers,
                                                     const ptxInstructionCode     opCode,
                                                     const bool                   isSparse )
{
    InstructionSignature newSignature = signature;

    // Adjust return type to match vector modifier (which is always 4)
    newSignature[0].ptxType  = "v" + std::to_string( 4 ) + "." + newSignature[0].ptxType;
    newSignature[0].llvmType = llvm::VectorType::get( newSignature[0].llvmType, 4 );

    // Add bool to return type if this is a sparse instruction
    if( isSparse )
    {
        llvm::Type* i1Ty = llvm::IntegerType::get( context, 1 );
        llvm::SmallVector<llvm::Type*, 2> structElements = {newSignature[0].llvmType, i1Ty};
        newSignature[0].llvmType = llvm::StructType::get( context, structElements );
    }

    const int coordVectorSize = getVectorSizeForDimensionality( modifiers.texDim );

    // The first argument to array calls is a struct containing an index plus the coordinates
    // e.g. a2d is {i32, f32, f32, f32}
    if( isArrayDimensionality( modifiers.texDim ) )
    {
        OperandType& firstCoordTy = newSignature[2];

        llvm::Type* arrayIdxType = llvm::IntegerType::get( context, 32 );
        llvm::SmallVector<llvm::Type*, 4> structMembers = {arrayIdxType};

        const int structSize = coordVectorSize == 1 ? 2 : 4;
        for( int i = 1; i < structSize; ++i )
            structMembers.push_back( firstCoordTy.llvmType );

        firstCoordTy.llvmType = llvm::StructType::get( context, structMembers );
    }

    // Adjust all further coordinates to take vectors appropriate for the coordinate size
    if( coordVectorSize > 1 )
    {
        const size_t coordStartIndex = isArrayDimensionality( modifiers.texDim ) ? 3 : 2;
        // tex.level instructions have a trailing scalar argument specifying the mip level
        const size_t coordEndIndex = opCode == ptx_tex_level_Instr ? newSignature.size() - 1 : newSignature.size();
        for( size_t i                = coordStartIndex; i < coordEndIndex; ++i )
            newSignature[i].llvmType = llvm::VectorType::get( newSignature[i].llvmType, coordVectorSize );
    }

    return newSignature;
}

InstructionSignature expandSurfaceInstructionSignature( llvm::LLVMContext&           context,
                                                        const InstructionSignature&  signature,
                                                        const PTXIntrinsicModifiers& modifiers )
{
    InstructionSignature newSignature = signature;

    OperandType& returnType = newSignature[0];
    if( returnType.llvmType != llvm::Type::getVoidTy( context ) && modifiers.vectorSize != VectorSize::unspecified )
        returnType.llvmType = llvm::VectorType::get( returnType.llvmType, vectorSizeToInt( modifiers.vectorSize ) );

    // All arguments (excluding the image address) should be vectors whose size cooresponds to the dimensionality of the surface
    const int coordVectorSize = getVectorSizeForDimensionality( modifiers.texDim );

    // argument at index 1 is the surface handle; it doesn't need modification

    // Special case for array dimensionalities: the first coordinate argument
    // has an i32 array index register packed in with the coordinate values, so
    // it ends up being a vector arguments.
    OperandType& firstCoordType = newSignature[2];
    if( isArrayDimensionality( modifiers.texDim ) )
    {
        const int   firstCoordVectorSize = coordVectorSize == 1 ? 2 : 4;
        llvm::Type* vectorType           = llvm::VectorType::get( firstCoordType.llvmType, firstCoordVectorSize );
        firstCoordType.llvmType          = vectorType;
    }
    else if( coordVectorSize > 1 )
        firstCoordType.llvmType = llvm::VectorType::get( firstCoordType.llvmType, coordVectorSize );

    // Further arguments are values to store; they are vector sizes matching our vector modifier
    if( modifiers.vectorSize != VectorSize::unspecified )
    {
        for( size_t i = 3; i < newSignature.size(); ++i )
        {
            OperandType& lwrrType = newSignature[i];
            lwrrType.llvmType     = llvm::VectorType::get( lwrrType.llvmType, vectorSizeToInt( modifiers.vectorSize ) );
        }
    }

    return newSignature;
}

InstructionSignature expandLdStInstructionSignature( llvm::LLVMContext&            context,
                                                     const ptxInstructionTemplate& instTemplate,
                                                     const InstructionSignature&   signature,
                                                     const PTXIntrinsicModifiers&  modifiers )
{
    InstructionSignature newSignature = signature;

    const int valTypeIdx = (ptxInstructionCode)instTemplate->code == ptx_ld_Instr ? 0 : 2;
    const int ptrTypeIdx = 1;

    // Adjust the function's return type so it matches the vector size specified by the modifiers.
    if( modifiers.vectorSize != VectorSize::unspecified )
        newSignature[valTypeIdx].llvmType =
            llvm::VectorType::get( newSignature[valTypeIdx].llvmType, vectorSizeToInt( modifiers.vectorSize ) );

    // Adjust the instruction's signature so the memory type points to
    // the proper address space, based on the address space in the modifiers.
    newSignature[ptrTypeIdx].llvmType =
        llvm::PointerType::get( newSignature[valTypeIdx].llvmType, ptxToLwvmAddressSpace( modifiers.addressSpace ) );

    return newSignature;
}

static void doubleOperandSize( llvm::LLVMContext& context, OperandType& operand )
{
    if( llvm::IntegerType* intTy = llvm::dyn_cast<llvm::IntegerType>( operand.llvmType ) )
    {
        unsigned int newBitWidth = intTy->getBitWidth() * 2;
        llvm::Type*  newIntType  = llvm::IntegerType::get( context, newBitWidth );
        // Use the first letter of the previous ptx type, since it might be "u", "s", or "b"
        std::string newPtxType = operand.ptxType[0] + std::to_string( newBitWidth );

        operand.llvmType = newIntType;
        operand.ptxType  = newPtxType;
        return;
    }

    llvm::Type* halfTy   = llvm::Type::getHalfTy( context );
    llvm::Type* floatTy  = llvm::Type::getFloatTy( context );
    llvm::Type* doubleTy = llvm::Type::getDoubleTy( context );

    if( operand.llvmType == halfTy )
    {
        operand.llvmType = floatTy;
        operand.ptxType  = "f32";
        return;
    }
    else if( operand.llvmType == floatTy )
    {
        operand.llvmType = doubleTy;
        operand.ptxType  = "f64";
        return;
    }

    throw std::runtime_error( "Trying to double invalid type" );
}

InstructionSignature expandMathInstructionSignature( llvm::LLVMContext& context, const InstructionSignature& signature, bool hasDoubleRes )
{
    InstructionSignature newSignature = signature;

    // Special case: if DOUBLERES is set, operands 0 and 3 will be double their specified sizes
    // DOUBLERES should only be set for mad and mul wide instructions
    if( hasDoubleRes )
    {
        doubleOperandSize( context, newSignature[0] );
        doubleOperandSize( context, newSignature[3] );
    }

    return newSignature;
}

InstructionSignature expandSetOrSetpInstructionSignature( llvm::LLVMContext& context, const InstructionSignature& signature, const bool hasPredOutput )
{
    InstructionSignature newSignature = signature;

    // If this instruction has optional predicate output, expand the return type to an {i1, i1} struct
    if( hasPredOutput )
    {
        llvm::Type*              i1Ty          = llvm::IntegerType::get( context, 1 );
        std::vector<llvm::Type*> structMembers = {i1Ty, i1Ty};
        newSignature[0].llvmType               = llvm::StructType::get( context, structMembers );
    }

    return newSignature;
}

}  // namespace PTXIntrinsics
}  // namespace optix
