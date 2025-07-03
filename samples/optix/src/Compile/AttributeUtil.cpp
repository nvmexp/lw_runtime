// Copyright LWPU Corporation 2014
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <Compile/AttributeUtil.h>

#include <Context/ProgramManager.h>
#include <FrontEnd/Canonical/FrontEndHelpers.h>
#include <FrontEnd/Canonical/IntrinsicsManager.h>
#include <FrontEnd/Canonical/VariableReference.h>

#include <corelib/compiler/CoreIRBuilder.h>
#include <corelib/compiler/LLVMUtil.h>

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>

#include <vector>

using namespace optix;
using namespace llvm;
using namespace corelib;

static const VariableReference* getAttributeForIntrinsic( const ProgramManager* pm, llvm::Function& attributeFunction, bool* isSet )
{
    StringRef uniqueName = "";
    if( SetAttributeValue::isIntrinsic( &attributeFunction ) )
    {
        bool success = SetAttributeValue::parseUniqueName( &attributeFunction, uniqueName );
        RT_ASSERT( success );
        *isSet = true;
    }
    else if( GetAttributeValue::isIntrinsic( &attributeFunction ) )
    {
        bool success = GetAttributeValue::parseUniqueName( &attributeFunction, uniqueName );
        RT_ASSERT( success );
        *isSet = false;
    }
    else
    {
        return nullptr;
    }
    const VariableReference* varref = pm->getVariableReferenceByUniversallyUniqueName( uniqueName );
    RT_ASSERT( varref != nullptr );
    return varref;
}

static std::string getTypedName( const Type* type )
{
    switch( type->getTypeID() )
    {
        case Type::FloatTyID:
            return std::string( "float" );
        case Type::IntegerTyID:
            return std::string( "i" ) + std::to_string( type->getIntegerBitWidth() );
        case Type::ArrayTyID:
            return std::string( "array_" ) + getTypedName( type->getArrayElementType() ) + "_x_"
                   + std::to_string( type->getArrayNumElements() );
        case Type::VectorTyID:
            return std::string( "vector_" ) + getTypedName( type->getVectorElementType() ) + "_x_"
                   + std::to_string( type->getVectorNumElements() );
        default:
            RT_ASSERT_FAIL_MSG( "Unknown type for computing typed name" );
    }
}

static llvm::Type* getAttributeType( const VariableReference* varref, llvm::Type* i8Ty )
{
    unsigned int allocaSize    = varref->getType().computeSize();
    Type*        defaultType   = ArrayType::get( i8Ty, allocaSize );
    Type*        attributeType = getCleanType( varref->getType(), defaultType, nullptr );
    return attributeType;
}

static std::string getAllocaName( const VariableReference* varref, bool useUniqueName, llvm::Type* i8Ty )
{
    if( useUniqueName )
        return varref->getUniversallyUniqueName();
    return varref->getInputName() + "_" + getTypedName( getAttributeType( varref, i8Ty ) );
}

void optix::patchAttributesToLocalAllocas( const ProgramManager* pm,
                                           llvm::Function*       inFunction,
                                           bool                  useUniqueName,
                                           std::map<std::string, llvm::AllocaInst*>* returnAllocas )
{
    LLVMContext&           llvmContext = inFunction->getContext();
    IntegerType*           i8Ty        = Type::getInt8Ty( llvmContext );
    BasicBlock&            entryBlock  = inFunction->getEntryBlock();
    corelib::CoreIRBuilder irb( &entryBlock );

    std::vector<Value*> toRemove;
    std::map<std::string, llvm::AllocaInst*> allocas;
    for( Function& attributeFunction : *inFunction->getParent() )
    {
        bool                     isSet  = false;
        const VariableReference* varref = getAttributeForIntrinsic( pm, attributeFunction, &isSet );
        if( !varref )
            continue;

        // If necessary, create an alloca to contain the attribute
        std::string allocaName = getAllocaName( varref, useUniqueName, i8Ty );
        if( allocas.count( allocaName ) == 0 )
        {
            Type* attributeType = getAttributeType( varref, i8Ty );
            irb.SetInsertPoint( corelib::getFirstNonAlloca( inFunction ) );
            llvm::AllocaInst* alloca = irb.CreateAlloca( attributeType );
            alloca->setName( allocaName );
            alloca->setAlignment( 16 );
            allocas.insert( std::make_pair( allocaName, alloca ) );
        }

        // Update set/gets to loads and stores in the alloca
        AllocaInst* attributeAlloca = allocas.at( allocaName );

        for( CallInst* call : getCallsToFunction( &attributeFunction, inFunction ) )
        {
            Value* ptr              = attributeAlloca;
            Type*  attributeType    = isSet ? call->getOperand( 2 )->getType() : call->getType();
            Type*  attributePtrType = attributeType->getPointerTo();
            Value* elementOffset    = call->getOperand( 1 );


            // Colwert pointer if necessary.
            irb.SetInsertPoint( call );
            if( isa<Constant>( elementOffset ) && cast<Constant>( elementOffset )->isZeroValue() )
            {
                ptr = irb.CreateBitCast( ptr, attributePtrType );
            }
            else if( ptr->getType() != attributePtrType )
            {
                Type* i8PtrTy = PointerType::get( i8Ty, 0 );

                ptr = irb.CreateBitCast( ptr, i8PtrTy );
                ptr = irb.CreateInBoundsGEP( ptr, elementOffset );
                ptr = irb.CreateBitCast( ptr, attributePtrType );
            }
            // Otherwise the offset is guaranteed to be zero.

            // Create the load or store
            if( isSet )
            {
                Value* toStore = call->getOperand( 2 );
                irb.CreateStore( toStore, ptr );
            }
            else
            {
                Value* attribute = irb.CreateLoad( ptr );
                call->replaceAllUsesWith( attribute );
            }
            call->eraseFromParent();
        }

        if( attributeFunction.use_empty() )
            toRemove.push_back( &attributeFunction );
    }

    removeValues( toRemove );
    if( returnAllocas )
        *returnAllocas = std::move( allocas );
}
