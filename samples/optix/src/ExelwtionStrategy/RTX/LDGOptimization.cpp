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

#include <ExelwtionStrategy/RTX/LDGOptimization.h>

#include <FrontEnd/Canonical/VariableReference.h>

#include <corelib/compiler/CoreIRBuilder.h>
#include <prodlib/exceptions/Assert.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>

using namespace optix;
using namespace llvm;

static unsigned getNumElements( const Type* type )
{
    if( isa<VectorType>( type ) )
        return type->getVectorNumElements();
    if( isa<StructType>( type ) )
        return type->getStructNumElements();
    if( isa<ArrayType>( type ) )
        return type->getArrayNumElements();
    RT_ASSERT_FAIL_MSG( "getNumElements() should never be called for a non-aggregate type" );
}

// Return true if all contained types are the same.
// This is trivially the case for every type except structs.
static bool elementTypesMatch( const Type* type )
{
    RT_ASSERT( !isa<FunctionType>( type ) );

    const StructType* structType = dyn_cast<StructType>( type );
    if( !structType )
        return true;

    const Type* elementType = structType->getElementType( 0 );
    for( unsigned i = 1, e = structType->getNumElements(); i < e; i++ )
        if( structType->getElementType( i ) != elementType )
            return false;

    return true;
}

// Returns true if the type is a vector that is understood by the LWVM LDG
// intrinsic.
static bool isLDGVectorType( const Type* ty )
{
    if( !isa<VectorType>( ty ) )
        return false;

    // LDG only supports loading vectors with 2 or 4 elements.
    const unsigned numElements = getNumElements( ty );
    if( ( numElements != 2 ) && ( numElements != 4 ) )
        return false;

    // LDG only supports loading vectors of element types int, float, or pointer.
    const Type* elementType = ty->getContainedType( 0 );
    if( !elementType->isIntegerTy() && !elementType->isFloatingPointTy() && !elementType->isPointerTy() )
    {
        return false;
    }

    // LDG only supports loading vectors of sizes up to 128 bit.
    const unsigned eltBits = elementType->getPrimitiveSizeInBits();
    if( ( numElements * eltBits ) > 128 )
        return false;

    // LDG only supports loading vectors of element types i8 or larger.
    return eltBits >= 8;
}

static unsigned gcd( const unsigned a, const unsigned b )
{
    if( b == 0 )
        return a;
    return gcd( b, a % b );
}

static bool alignmentOkayForVectorLoad( const Type* ty, const unsigned alignment, const DataLayout& DL )
{
    RT_ASSERT_MSG( isa<VectorType>( ty ) || isa<StructType>( ty ) || isa<ArrayType>( ty ),
                   "must not call alignmentOkayForVectorLoad() for non-aggregate type" );

    // If this is a non-homogeneous struct, bail out without asking.
    if( !elementTypesMatch( ty ) )
        return false;

    Type*          elementType = ty->getContainedType( 0 );
    const unsigned elemAlign   = DL.getABITypeAlignment( elementType );
    const unsigned vecAlign    = getNumElements( ty ) * elemAlign;
    const unsigned minAlign    = gcd( alignment, vecAlign );
    return minAlign == vecAlign;
}

// Create 'numLoads' loads of sub-elements of homogeneous aggregate structures.
// 'load' is the original load of the source structure.
// 'desiredType' is the return type of each of the loads to be created, or NULL,
// in which case each element is loaded with its actual type.
// 'index' is the current index into the original structure where to start.
// The results of all these loads are merged together into 'res', which is again
// of the same type as the original source structure.
// 'splitLoads' is filled with the created loads.
static void createLoads( LoadInst* load, const unsigned numLoads, VectorType* vectorTy, unsigned index, Value*& res, std::vector<LoadInst*>& splitLoads )
{
    const unsigned size = vectorTy ? vectorTy->getVectorNumElements() : 1;
    Value*         ptr  = load->getPointerOperand();
    Type* ptrTy = vectorTy ? vectorTy->getPointerTo( cast<PointerType>( ptr->getType() )->getAddressSpace() ) : nullptr;
    Type* ty    = load->getType();

    corelib::CoreIRBuilder irb( load );
    // If 'vectorTy' is NULL, this loop creates scalar loads of all elements.
    // This allows us to use this function both for homogeneous aggregate values
    // and non-homogeneous structs.
    LLVMContext& context = load->getContext();
    Value*       zero    = ConstantInt::get( IntegerType::getInt32Ty( context ), 0 );
    for( unsigned i = 0; i < numLoads; ++i, index += size )
    {
        Value* idx = ConstantInt::get( IntegerType::getInt32Ty( context ), index );

        // Create a GEP that points to the first element of the vector.
        Value* args[2]{zero, idx};
        Value* gep = irb.CreateGEP( ptr, args, "split.gep" );

        // If we are loading multiple subsequent values as a vector, bitcast the
        // pointer to the desired vector type.
        if( vectorTy && gep->getType() != ptrTy )
            gep = irb.CreateBitCast( gep, ptrTy, "split.bc" );

        // Load the element(s) (this load will be replaced by an LDG intrinsic).
        LoadInst* lde = irb.CreateLoad( gep, "split.ld" );
        splitLoads.push_back( lde );

        // Merge the loaded value(s) into the final result.
        // If we are creating scalar loads, the loop iterates exactly once and no
        // extraction is performed (since the loaded value is no vector).
        for( unsigned j = 0; j < size; ++j )
        {
            Value* insertVal = lde;

            // If we loaded more than one value, extract the one of the current index.
            if( vectorTy )
            {
                Value* tmpIdx = ConstantInt::get( IntegerType::getInt32Ty( context ), j );
                insertVal     = irb.CreateExtractElement( lde, tmpIdx, "split.ext" );
            }

            // Insert the value into the final result.
            Value* tmpIdx = ConstantInt::get( IntegerType::getInt32Ty( context ), index + j );
            if( ty->isVectorTy() )
                res = irb.CreateInsertElement( res, insertVal, tmpIdx, "split.copy" );
            else
                res = irb.CreateInsertValue( res, insertVal, index + j, "split.copy" );
        }
    }
}

// Note that this function also does the right thing if the original load was
// vectorized and is aligned properly, i.e., create a vector LDG. It does
// perform a redundant copy of the vector, but this is easily removed by later
// optimizations.
// TODO: It is possible to even try to create 2- or 4-element
//       vector loads for a subset of the elements of a non-homogeneous struct.
//       Not sure if this is worth the effort (need to track alignment, decide
//       which elements to combine, etc.) so I'll skip it for now.
static Instruction* splitLoad( LoadInst* load, std::vector<LoadInst*>& splitLoads )
{
    // Try to split into types that can be vectorized (e.g. [8 x i32] into two
    // vector loads of type <4 x i32>).
    RT_ASSERT_MSG( isa<VectorType>( load->getType() ) || isa<StructType>( load->getType() ) || isa<ArrayType>( load->getType() ),
                   "must not call splitLoadIntoVectors() for load of non-aggregate type" );

    DataLayout     DL( load->getParent()->getParent()->getParent() );
    Type*          ty          = load->getType();
    const unsigned alignment   = load->getAlignment();
    const unsigned numElements = getNumElements( ty );

    // Check if the alignment allows to load vectors (if the source is a non-
    // homogeneous struct, alignmentOkayForVectorLoad() will return false).
    Type*       elementType       = ty->getContainedType( 0 );
    VectorType* fullVecType       = nullptr;
    VectorType* halfVecType       = nullptr;
    bool        canUseFullVectors = false;
    bool        canUseHalfVectors = false;

    // We can only do vector loads if the contained type is a valid element type,
    // if the element type is at least 8 bits large, if the full vector type
    // is at most 128 bit large, and if the alignment of the original load signals
    // that a vector load would be safe.
    if( VectorType::isValidElementType( elementType ) )
    {
        fullVecType       = VectorType::get( elementType, 4 );
        halfVecType       = VectorType::get( elementType, 2 );
        canUseFullVectors = elementType->getPrimitiveSizeInBits() >= 8 && fullVecType->getPrimitiveSizeInBits() <= 128
                            && alignmentOkayForVectorLoad( fullVecType, alignment, DL );
        canUseHalfVectors = elementType->getPrimitiveSizeInBits() >= 8 && halfVecType->getPrimitiveSizeInBits() <= 128
                            && alignmentOkayForVectorLoad( halfVecType, alignment, DL );
    }

    // Since there are only 2- and 4-element vector LDGs, we use a greedy approach
    // to determine which elements of the aggregate value to load together. If
    // alignment and/or structure of the value does not allow vector loads, we
    // load every element separately.
    const unsigned numFullVectors = canUseFullVectors ? numElements / 4 : 0;
    const unsigned numHalfVectors = canUseHalfVectors ? ( numElements - ( numFullVectors * 4 ) ) / 2 : 0;
    const unsigned numScalars     = numElements - ( ( numFullVectors * 4 ) + ( numHalfVectors * 2 ) );

    Value* res = UndefValue::get( ty );

    // Create 4-element vector loads.
    if( numFullVectors )
        createLoads( load, numFullVectors, fullVecType, 0, res, splitLoads );

    // Create 2-element vector loads.
    if( numHalfVectors )
        createLoads( load, numHalfVectors, halfVecType, numFullVectors * 4, res, splitLoads );

    // Create scalar loads (signaled by NULL vector type).
    if( numScalars )
        createLoads( load, numScalars, nullptr, numElements - numScalars, res, splitLoads );

    return cast<Instruction>( res );
}

// This is based on LWVM (LWPTXlduTransform.cpp).
// Create an LDG intrinsic for the given load.
// the load's uses still have to be replaced by the returned instruction, and
// the load removed.
// The LDG intrinsic can only handle int/float/pointer/vector, any value with a
// different type is either interpreted as a vector (structs or arrays with 2 or
// 4 elements), or has to be split into multiple ones that can be expressed.
// TODO: Use vector load for structs/arrays with different number of
//       elements but matching alignment and size (e.g. [ 16 x i8 ]).
Instruction* optix::genLDGIntrinsic( LoadInst* load )
{
    Function*    parent  = load->getParent()->getParent();
    Module*      module  = parent->getParent();
    LLVMContext& context = parent->getContext();
    RT_ASSERT( !module->getDataLayoutStr().empty() );

    // Check what kind of LDG intrinsic we need to generate.
    Intrinsic::ID id;
    bool          isPointer = false;
    Type*         ty        = load->getType();
    if( ty->isIntegerTy() )
    {
        id = Intrinsic::lwvm_ldg_global_i;
    }
    else if( ty->isFloatingPointTy() )
    {
        id = Intrinsic::lwvm_ldg_global_f;
    }
    else if( ty->isPointerTy() )
    {
        isPointer = true;
        id        = Intrinsic::lwvm_ldg_global_p;
    }
    else if( isLDGVectorType( ty ) && alignmentOkayForVectorLoad( ty, load->getAlignment(), DataLayout( module ) ) )
    {
        // Create a single vector intrinsic that loads the entire value.
        Type* elementType = ty->getContainedType( 0 );
        if( elementType->isIntegerTy() )
            id = Intrinsic::lwvm_ldg_global_i;
        else if( elementType->isFloatingPointTy() )
            id = Intrinsic::lwvm_ldg_global_f;
        else
            id = Intrinsic::lwvm_ldg_global_p;
    }
    else
    {
        // If the type and alignment allow to treat the loaded value as a vector,
        // we create a single vector intrinsic that loads the entire value, and then
        // create a value of the appropriate type again afterwards.
        // If the type is not vectorizable (e.g. non-homogeneous structs), or if the
        // alignment does not allow a vector load, we create scalar loads of the
        // elements of the aggregate value and combine the result back to the
        // desired type.
        // If the type is partially vectorizable (e.g. large but homogeneous
        // structures), we create a combination of vector and scalar loads and
        // combine the result back to the desired type.
        std::vector<LoadInst*> splitLoads;
        Instruction*           res = splitLoad( load, splitLoads );

        for( LoadInst* splitLoad : splitLoads )
        {
            Instruction* splitRes = genLDGIntrinsic( splitLoad );
            splitLoad->replaceAllUsesWith( splitRes );
            splitLoad->eraseFromParent();
        }

        return res;
    }

    Value*         ptr          = load->getPointerOperand();
    PointerType*   ptrType      = cast<PointerType>( ptr->getType() );
    const unsigned ptrAddrSpace = ptrType->getAddressSpace();
    Type*          returnType   = load->getType();

    corelib::CoreIRBuilder irb( load );

    if( isPointer )
    {
        // The intrinsic does not like multi-level pointer types, so have to colwert
        // to single level pointer first.
        const unsigned retAddrSpace = cast<PointerType>( load->getType() )->getAddressSpace();
        returnType                  = PointerType::get( Type::getInt8Ty( context ), retAddrSpace );
        ptrType                     = PointerType::get( Type::getInt32Ty( context ), ptrAddrSpace );
        ptr                         = irb.CreateBitCast( ptr, ptrType, "ldg.in.bc" );
    }

    // Get the intrinsic function.
    Type*     intrinsicTypes[2]{returnType, ptrType};
    Function* ldgFunc = Intrinsic::getDeclaration( module, id, intrinsicTypes );

    // Create the intrinsic call.
    Value*    alignment = ConstantInt::get( IntegerType::getInt32Ty( context ), load->getAlignment() );
    Value*    args[2]{ptr, alignment};
    CallInst* ldgCall = irb.CreateCall( ldgFunc, args );
    ldgCall->takeName( load );
    ldgCall->setDebugLoc( load->getDebugLoc() );

    // If this is a pointer LDG, we may have to bitcast it back to the appropriate
    // type.
    if( isPointer && load->getType() != ldgCall->getType() )
        return static_cast<Instruction*>( irb.CreateBitCast( ldgCall, load->getType(), "ldg.res.bc" ) );

    return ldgCall;
}
