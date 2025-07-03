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

#include <FrontEnd/PTX/Canonical/UberPointer.h>

#include <Context/LLVMManager.h>
#include <ExelwtionStrategy/Compile.h>  // TODO: this is only needed for enum AtomicOpType, we should find a better place for that.
#include <FrontEnd/Canonical/IntrinsicsManager.h>
#include <Util/ContainerAlgorithm.h>

#include <corelib/compiler/CoreIRBuilder.h>
#include <corelib/compiler/LLVMUtil.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/CompileError.h>

#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

#include <cassert>

using namespace optix;
using namespace llvm;
using namespace corelib;
using namespace prodlib;

// Support functions.
//------------------------------------------------------------------------------
static bool isAtomicFunction( StringRef functionName );
static AtomicOpType getAtomicOperator( StringRef functionName );
static std::string getPTXAtomicFunction( AtomicOpType op, Type* type );

//------------------------------------------------------------------------------
// Given the pointer allocaptr add the offset value to it.
// This is to reference the piece of data we are interested in within a buffer element.
static Value* addOffset( DataLayout& DL, Value* allocaptr, const StringRef& name, Type* ptrtype, Value* offset, Instruction* insertBefore )
{
    corelib::CoreIRBuilder irb{insertBefore};
    Value*                 offsetptr;
    if( offset )
    {
        // All the computation is performed on pointers to i8s.
        Type* i8PtrType = irb.getInt8PtrTy();
        // cast to char*
        Value* i8Ptr = irb.CreateBitCast( allocaptr, i8PtrType, name + ".i8ptr" );
        // add offset
        Value* i8Offsetptr = irb.CreateGEP( i8Ptr, offset, name + ".i8offsetptr" );
        // cast from char*
        offsetptr = irb.CreateBitCast( i8Offsetptr, ptrtype, name + ".valueptr" );
    }
    else
    {
        // If no offset is specified return the base pointer.
        offsetptr = irb.CreateBitCast( allocaptr, ptrtype, name + ".valueptr" );
    }
    return offsetptr;
}

//------------------------------------------------------------------------------
template <UberPointer::Field F>
Type* UberPointer::getType( const LLVMManager* llvm_manager )
{
    // Pull the types from the struct.
    Type*    ty  = llvm_manager->getUberPointerType();
    unsigned idx = F;
    return ExtractValueInst::getIndexedType( ty, idx );
}

StructType* UberPointer::getType( const LLVMManager* llvm_manager )
{
    return llvm_manager->getUberPointerType();
}

// Return the type of the field of the UberPointer struct specified by the template parameter.
// struct UberPointer {
//   size_t offset;  // Offset within the element. This is useful in cases in which we have a struct or a vector type
//                   // element and we whant to access a subelement of it.
//   int kind;       // Index into UberPointerSet, this is used as an identifier for the UberPointer and
//                   // it is used to retrieve on which variable reference the UberPointer is referring to.
//   int id;         // This is used to store the buffer id of the buffer the UberPointer is referring to, in the case
//                   // of bindless buffers for example. This can also be used to store the exception index.
//   size_t3 buffer_index; // Coordinates of the element of the buffer accessed by the UberPointer.
//   int elementSize;    // Size in Bytes data type of the global variable we are trying to access. This is used for
//                       // atomic instructions accessing a field of a vector element. This is needed because we need
//                       // the field size to compute the element address.
//   uint_64 raw_pointer; // This field is used in case we have to create a UberPointer out of a Raw pointer.
// };
template <UberPointer::Field F>
Type* UberPointer::createType( const LLVMManager* llvm_manager )
{
    switch( F )
    {
        case Offset:
            return llvm_manager->getSizeTType();
        case UPKind:
            return llvm_manager->getI32Type();
        case ID:
            return llvm_manager->getI32Type();
        case BufferIndex:
            return llvm_manager->getSize3Type();
        case ElementSize:
            return llvm_manager->getI32Type();
        case RawPointer:
            return llvm_manager->getI64Type();
    }
    return nullptr;
}

// Return the type of the UberPointer struct.
StructType* UberPointer::createType( const LLVMManager* llvm_manager )
{
    std::string name       = "struct.cort::UberPointer";
    Type*       elements[] = {createType<Offset>( llvm_manager ),      createType<UPKind>( llvm_manager ),
                        createType<ID>( llvm_manager ),          createType<BufferIndex>( llvm_manager ),
                        createType<ElementSize>( llvm_manager ), createType<RawPointer>( llvm_manager )};
    StructType* ty = StructType::create( elements, name );
    return ty;
}

// Extract from the given UberPointer the value of the field specified by the template parameter.
// Since the UberPointer is not a constant we have to genereate an ExtractValue instruction.
template <UberPointer::Field F>
Value* UberPointer::get( Value* UberP, const Twine& NameStr, Instruction* insertBefore )
{
    assert( insertBefore );
    unsigned idx = F;
    return corelib::CoreIRBuilder{insertBefore}.CreateExtractValue( UberP, idx, NameStr );
}

// Extract from the given UberPointer the value of the field specified by the template parameter.
// The UberPointer is a constant, therefore we can return a constant.
template <UberPointer::Field F>
Constant* UberPointer::get( Constant* UberP )
{
    unsigned idx = F;
    return ConstantExpr::getExtractValue( UberP, idx );
}

// Set the field of the UberPointer specified by the template parameter.
// To do this we have to create an insert value instruction.
template <UberPointer::Field F>
Value* UberPointer::set( Value* UberP, Value* new_value, const Twine& NameStr, Instruction* insertBefore )
{
    assert( insertBefore );
    unsigned idx = F;
    return corelib::CoreIRBuilder{insertBefore}.CreateInsertValue( UberP, new_value, idx, NameStr );
}

// Set the field of the UberPointer specified by the template parameter.
// The UberPointer and the value to set are both constants, therefore we can return a constant.
template <UberPointer::Field F>
Constant* UberPointer::set( Constant* UberP, Constant* new_value )
{
    unsigned idx = F;
    return ConstantExpr::getInsertValue( UberP, new_value, idx );
}

// Create a UberPointer. Since all the given fields are constants we can generate a constant.
Constant* UberPointer::getUberPointer( LLVMManager* llvm_manager, int upkind, size_t elementSize, size_t initialOffset )
{
    StructType* ST     = cast<StructType>( UberPointer::getType( llvm_manager ) );
    Constant*   UberPC = UndefValue::get( ST );
    UberPC             = set<Offset>( UberPC, ConstantInt::get( getType<Offset>( llvm_manager ), initialOffset ) );
    UberPC             = set<UPKind>( UberPC, ConstantInt::get( getType<UPKind>( llvm_manager ), upkind ) );
    UberPC = set<ElementSize>( UberPC, ConstantInt::get( getType<ElementSize>( llvm_manager ), elementSize ) );
    return UberPC;
}

// Create a UberPointer. Since all the given fields are constants we can generate a constant.
Constant* UberPointer::getUberPointer( LLVMManager* llvm_manager, int upkind, Constant* index, size_t elementSize, size_t initialOffset )
{
    Constant* UberPC = getUberPointer( llvm_manager, upkind, elementSize, initialOffset );
    UberPC           = set<ID>( UberPC, index );
    return UberPC;
}

// Create a UberPointer.
Value* UberPointer::getUberPointer( LLVMManager*               llvm_manager,
                                    int                        upkind,
                                    const std::vector<Value*>& bufferIndices,
                                    size_t                     elementSize,
                                    size_t                     initialOffset,
                                    Instruction*               insertBefore )
{
    Constant* UberPC = getUberPointer( llvm_manager, upkind, elementSize, initialOffset );
    bool allConst    = algorithm::all_of( bufferIndices, []( Value* value ) { return isa<Constant>( value ); } );

    Type*  bufferIndexTy = getType<BufferIndex>( llvm_manager );
    Value* UberP         = nullptr;
    if( allConst )
    {
        // If all the indices used to access the memory location are constants we can generate a constant version of the UberPointer.
        Constant* buffer_index = UndefValue::get( bufferIndexTy );
        for( unsigned i = 0, size = bufferIndices.size(); i < size; ++i )
        {
            unsigned idx = i;
            buffer_index = ConstantExpr::getInsertValue( buffer_index, dyn_cast<Constant>( bufferIndices[i] ), idx );
        }
        UberP = set<BufferIndex>( UberPC, buffer_index );
    }
    else
    {
        // If not, we have to create explicit insert instructions.
        corelib::CoreIRBuilder irb{insertBefore};
        Value*                 buffer_index = UndefValue::get( bufferIndexTy );
        for( unsigned i = 0, size = bufferIndices.size(); i < size; ++i )
        {
            unsigned idx = i;
            buffer_index = irb.CreateInsertValue( buffer_index, bufferIndices[i], idx );
        }
        UberP = set<BufferIndex>( UberPC, buffer_index, "", insertBefore );
    }

    return UberP;
}

// Create a UberPointer. In particular set the ID field to store the value that contains the buffer id this UberPointer
// is referring to.
Value* UberPointer::getUberPointer( LLVMManager*               llvm_manager,
                                    int                        upkind,
                                    Value*                     bufferId,
                                    const std::vector<Value*>& args,
                                    size_t                     elementSize,
                                    size_t                     initialOffset,
                                    Instruction*               insertBefore )
{
    Value* uberPointer = getUberPointer( llvm_manager, upkind, args, elementSize, initialOffset, insertBefore );
    uberPointer        = set<ID>( uberPointer, bufferId, "", insertBefore );
    return uberPointer;
}

Value* UberPointer::getRawUberPointer( LLVMManager* llvm_manager, int upkind, Value* raw_pointer, Instruction* insertBefore )
{
    Value* uberPointer = getUberPointer( llvm_manager, upkind, 0, 0 );
    uberPointer        = set<RawPointer>( uberPointer, raw_pointer, "", insertBefore );
    return uberPointer;
}

// UberPointerTransform
// -----------------------------------------------------------------------------

const std::string UberPointerTransform::UBER_POINTER_MEM_READ           = "_UberPointerMemRead";
const std::string UberPointerTransform::UBER_POINTER_MEM_READ_TO_ALLOCA = "_UberPointerMemReadToAlloca";
const std::string UberPointerTransform::UBER_POINTER_MEM_WRITE          = "_UberPointerMemWrite";
const std::string UberPointerTransform::UBER_POINTER_ATOMIC_OP          = "_UberPointerMemAtomicOp";
const std::string UberPointerTransform::UBER_POINTER_GET_ADDRESS        = "_UberPointerGetAddress";


UberPointerTransform::UberPointerTransform( Module* module, LLVMManager* llvm_manager, const std::string& name, UberPointer::PointeeType pointeeType )
    : m_module( module )
    , m_llvmManager( llvm_manager )
    // This is the name of the global variable replaced during the transformation,
    // it is used to mangle the name of instructions genereated during translation,
    // this is to make the code more readable.
    , m_name( name )
    , m_hasLoads( false )
    , m_hasStores( false )
    , m_pointerEscapes( false )
    , m_pointeeType( pointeeType )
{
}

void UberPointerTransform::translate( Value* start, int upkind, size_t elementSize, size_t initialOffset )
{
    Value* UberP = UberPointer::getUberPointer( m_llvmManager, upkind, elementSize, initialOffset );
    m_visited.clear();
    translate( start, UberP );
}

void UberPointerTransform::translate( Value* start, int upkind, Constant* index, size_t elementSize, size_t initialOffset )
{
    Value* UberP = UberPointer::getUberPointer( m_llvmManager, upkind, index, elementSize, initialOffset );
    m_visited.clear();
    translate( start, UberP );
}

void UberPointerTransform::translate( Instruction* start, int upkind, const std::vector<Value*>& args, size_t elementSize, size_t initialOffset )
{
    Instruction* insertBefore = start;
    Value* UberP = UberPointer::getUberPointer( m_llvmManager, upkind, args, elementSize, initialOffset, insertBefore );
    m_visited.clear();
    translate( start, UberP );
}

void UberPointerTransform::translate( Instruction* start, int upkind, Value* bufferId, const std::vector<Value*>& args, size_t elementSize, size_t initialOffset )
{
    Instruction* insertBefore = start;
    Value* UberP = UberPointer::getUberPointer( m_llvmManager, upkind, bufferId, args, elementSize, initialOffset, insertBefore );
    m_visited.clear();
    translate( start, UberP );
}

// Replace all the uses of the value V_start to use the UberPointer UberP_in.
// V_start can be, for example, the pointer to a memory location.
// Using forward data flow, we replace all the uses of the pointers computed using the current pointer with UberPointers.

// This function should be refactored to use Instruction::getOpcode instead of dyn_cast if possible, as suggested by the LLVM programming guide:
// http://llvm.org/docs/ProgrammersManual.html#the-isa-cast-and-dyn-cast-templates
void UberPointerTransform::translate( Value* V_start, Value* UberP_in )
{
    Value* V = V_start;
    // If we have already visisted this object, no need to process it again
    if( !std::get<1>( m_visited.insert( V ) ) )
        return;

    // Iterate over all the uses of the current value.
    for( Value::user_iterator UI = V->user_begin(), UE = V->user_end(); UI != UE; )
    {
        Value* U     = *UI++;
        Value* UberP = UberP_in;
        if( LoadInst* L = dyn_cast<LoadInst>( U ) )
        {
            // The pointer is used in a load instruction.
            m_hasLoads                = true;
            Instruction* insertBefore = L;
            Function*    function     = insertBefore->getParent()->getParent();
            Type*        loadType     = L->getType();
            // A call to _UberPointerMemRead is a replacement for the original load.
            Value* newValue = createUberMemAccessCall( UberP, insertBefore, function, loadType, AccessType::READ, nullptr );

            L->replaceAllUsesWith( newValue );
            L->eraseFromParent();

            m_insertedLoads.push_back( newValue );
        }
        else if( StoreInst* S = dyn_cast<StoreInst>( U ) )
        {
            if( V == S->getPointerOperand() )
            {
                // In this case we are using the UberPointer to store to memory.
                m_hasStores               = true;
                Instruction* insertBefore = S;
                Function*    function     = insertBefore->getParent()->getParent();
                Type*        storedType   = S->getValueOperand()->getType();

                // A call to _UberPointerMemWrite is a replaceement for the original store.
                createUberMemAccessCall( UberP, insertBefore, function, storedType, AccessType::WRITE, S->getValueOperand() );
                S->eraseFromParent();
            }
            else
            {
                // In this case we are storing the pointer. We have to escape.
                Instruction* insertBefore = S;
                Function*    caller       = insertBefore->getParent()->getParent();
                Value* rawPtr = createUberMemAccessCall( UberP, insertBefore, caller, nullptr, AccessType::GET_ADDRESS, nullptr );
                m_pointerEscapes = true;

                Type* retType = S->getValueOperand()->getType();

                // Cast to the right type.
                if( retType != m_llvmManager->getI64Type() )
                {
                    RT_ASSERT_MSG( retType->isPointerTy(),
                                   "We are returning a pointer of unsupported type while translating (" + m_name
                                       + ")." );
                    rawPtr = corelib::CoreIRBuilder{insertBefore}.CreateIntToPtr( rawPtr, retType );
                }

                S->setOperand( 0, rawPtr );
            }
        }
        else if( BinaryOperator* B = dyn_cast<BinaryOperator>( U ) )
        {
            RT_ASSERT( V == B->getOperand( 0 ) || V == B->getOperand( 1 ) );
            Type* i64Ty = Type::getInt64Ty( m_llvmManager->llvmContext() );
            RT_ASSERT_MSG( B->getType() == i64Ty,
                           "Pointer arithmetic using a non-64 bit data type while translating (" + m_name + ")." );
            Instruction*           insertBefore = B;
            Function*              caller       = insertBefore->getParent()->getParent();
            corelib::CoreIRBuilder irb{insertBefore};

            int pointerPosition = 0;
            int otherOpPosition = 1;
            if( V != B->getOperand( pointerPosition ) )
                std::swap( pointerPosition, otherOpPosition );

            Value* otherOp = B->getOperand( otherOpPosition );

            // If we are dealing with a buffer, and the operation isn't (pointer + constant)
            // escape the pointer.  We need to do this before we update the raw_pointer field in
            // order to avoid duplication of the binary operation.
            if( ( m_pointeeType == UberPointer::PointeeType::BufferID || m_pointeeType == UberPointer::PointeeType::Buffer )
                && ( B->getOpcode() != BinaryOperator::Add || !isa<Constant>( otherOp ) ) )
            {
                Value* rawPtr = createUberMemAccessCall( UberP, insertBefore, caller, nullptr, AccessType::GET_ADDRESS, nullptr );
                m_pointerEscapes = true;
                B->setOperand( pointerPosition, rawPtr );
                continue;
            }

            // Update the raw_pointer field.
            Value* newOperands[2];
            newOperands[pointerPosition] = UberPointer::get<UberPointer::RawPointer>( UberP, "raw", insertBefore );
            newOperands[otherOpPosition] = B->getOperand( otherOpPosition );
            Value* raw_pointer           = irb.CreateBinOp( B->getOpcode(), newOperands[0], newOperands[1] );
            UberP = UberPointer::set<UberPointer::RawPointer>( UberP, raw_pointer, Twine( "UberP." ) + m_name, insertBefore );

            // If we are not pointing to a buffer there is not need to use div/mod.
            // Just update the offset.
            // This check relies on the fact that pointers to buffers cannot alias with anyother pointers except local memory ones.
            if( m_pointeeType != UberPointer::PointeeType::BufferID && m_pointeeType != UberPointer::PointeeType::Buffer )
            {
                Value* oldOffset = UberPointer::get<UberPointer::Offset>( UberP, "", insertBefore );

                Value* newOperands2[2];
                newOperands2[pointerPosition] = oldOffset;
                newOperands2[otherOpPosition] = B->getOperand( otherOpPosition );

                Value* newOffset = irb.CreateBinOp( B->getOpcode(), newOperands2[0], newOperands2[1] );
                Value* newUp = UberPointer::set<UberPointer::Offset>( UberP, newOffset, Twine( "UberP." ) + m_name, insertBefore );
                translate( B, newUp );
                continue;
            }

            // The use of pointer arithmetics is restricted.
            // Lwrrently we support only additions and 1D buffers.
            // Given this expression: B = pointer + A;
            // The uber pointer is updated in this way:
            // tmp_offset = UP.offset + A % sizeof(T);
            // UP.index.x = UP.index.x + A / sizeof(T) + tmp_offset / sizeof(T) - sign(tmpOffset);
            // UP.offset = (tmp_offset + sizeof(T)) % sizeof(T);
            // Here sign returns 0 if the argument is positive, 1 if the argument is negative.
            // This implementation should take into account the case in which A is negative.
            // Notice that for constant values of A the / and % operations should be optimized away since sizeof(T) is a
            // constant extracted from the UberPointer.

            Value* oldOffset = UberPointer::get<UberPointer::Offset>( UberP, "offset." + m_name, insertBefore );

            Value* eltSize = UberPointer::get<UberPointer::ElementSize>( UberP, "elt.size." + m_name, insertBefore );
            eltSize        = irb.CreateZExt( eltSize, i64Ty );

            Value* xDiff   = irb.CreateSDiv( otherOp, eltSize, "x.index." + m_name );
            Value* xOffset = irb.CreateSRem( otherOp, eltSize, "x.offset." + m_name );

            Value* oldIndex = UberPointer::get<UberPointer::BufferIndex>( UberP, "buffer.index." + m_name, insertBefore );
            Value* oldX     = irb.CreateExtractValue( oldIndex, 0, "x." + m_name );
            Value* tmpX     = irb.CreateAdd( oldX, xDiff, "tmp.x." + m_name );

            Value* tmpOffset        = irb.CreateAdd( oldOffset, xOffset, "tmp.offset." + m_name );
            Value* tmpOffsetShifted = irb.CreateAdd( tmpOffset, eltSize, "tmp.offset.shifted." + m_name );

            Type*    offsetType = UberPointer::getType<UberPointer::Offset>( m_llvmManager );
            unsigned shift      = m_llvmManager->llvmDataLayout().getTypeSizeInBits( offsetType ) - 1;
            // Keep in mind that this operation does not compute the sign of tmpOffset for large unsigned values.
            // I think it is ok anyways since unsigned values so large to break this are not legit for pointer arithmetics.
            Value* tmpOffsetSign =
                irb.CreateLShr( tmpOffset, ConstantInt::get( UberPointer::getType<UberPointer::Offset>( m_llvmManager ), shift ),
                                "tmp.offset.sign." + m_name );
            Value* tmpOffsetDiv = irb.CreateSDiv( tmpOffset, eltSize, "x.offset.div." + m_name );
            Value* tmpXShifted  = irb.CreateSub( tmpOffsetDiv, tmpOffsetSign, "x.offset.shifted." + m_name );

            Value* newOffset = irb.CreateSRem( tmpOffsetShifted, eltSize, "x.offset.rem." + m_name );
            Value* newX      = irb.CreateAdd( tmpX, tmpXShifted, "new.x." + m_name );

            Value* newIndex = irb.CreateInsertValue( oldIndex, newX, 0, "new.index." + m_name );

            Value* newUp = UberPointer::set<UberPointer::BufferIndex>( UberP, newIndex, "", insertBefore );
            newUp        = UberPointer::set<UberPointer::Offset>( newUp, newOffset, "new.up", insertBefore );

            translate( B, newUp );
        }
        else if( ConstantExpr* C = dyn_cast<ConstantExpr>( U ) )
        {
            if( C->isCast() )
            {
                RT_ASSERT( C->getNumOperands() == 1 );
                translate( C, UberP );
            }
            else if( GEPOperator* GEP = dyn_cast<GEPOperator>( C ) )
            {
                unsigned BitWidth = m_llvmManager->llvmDataLayout().getPointerTypeSizeInBits( GEP->getType() );
                APInt    immOff( BitWidth, 0 );
                if( !GEP->aclwmulateConstantOffset( m_llvmManager->llvmDataLayout(), immOff ) )
                    throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( GEP ),
                                        "Cannot compute offset for expression while translating (" + m_name + ")" );
                int       offset   = static_cast<int>( immOff.getSExtValue() );
                Constant* const_UP = dyn_cast<Constant>( UberP );
                RT_ASSERT( const_UP );
                Constant* old_offset = UberPointer::get<UberPointer::Offset>( const_UP );
                Constant* new_offset = ConstantExpr::getAdd(
                    old_offset, ConstantInt::get( UberPointer::getType<UberPointer::Offset>( m_llvmManager ), offset ) );
                Constant* new_UberP = UberPointer::set<UberPointer::Offset>( const_UP, new_offset );
                translate( C, new_UberP );
            }
            else
            {

                // See if we can simply replace the pointer with the UberPointer's offset
                int  us        = -1;
                bool typesGood = true;
                for( unsigned i = 0, ie = C->getNumOperands(); i < ie; ++i )
                {
                    if( C->getOperand( i ) == V )
                        us = i;
                    if( C->getOperand( i )->getType() != UberPointer::getType<UberPointer::Offset>( m_llvmManager ) )
                        typesGood = false;
                }

                if( us >= 0 && typesGood )
                {
                    Constant* const_UP = dyn_cast<Constant>( UberP );
                    RT_ASSERT( const_UP );
                    Constant* old_offset = UberPointer::get<UberPointer::Offset>( const_UP );
                    Constant* new_offset = C->getWithOperandReplaced( us, old_offset );
                    Constant* new_UberP  = UberPointer::set<UberPointer::Offset>( const_UP, new_offset );
                    translate( C, new_UberP );
                }
                else
                {
                    // We could also add support for other types of constant expressions (such as
                    // select if the need arose).
                    throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( C ),
                                        "Unexpected constant in analysis of variable usage while translating (" + m_name
                                            + ")" );
                }
            }
        }
        else if( CallInst* CI = dyn_cast<CallInst>( U ) )
        {
            Instruction*           insertBefore = CI;
            Function*              caller       = insertBefore->getParent()->getParent();
            corelib::CoreIRBuilder irb{insertBefore};

            if( CI->getCalledValue() == V )
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ),
                                    "Illegal use of pointer from OptiX variable or buffer used as a function pointer "
                                    "while "
                                    "translating ("
                                        + m_name + ")." );


            // Why are we treating pointers to callable programs differently than pointers to
            // other functions?  At the very lease we should refactor this code with the code at
            // the bottom of this case, so the treatment can be unified.
            if( CI->getCalledFunction() == nullptr )
            {
                Value* rawPtr = createUberMemAccessCall( UberP, insertBefore, caller, nullptr, AccessType::GET_ADDRESS, nullptr );
                m_pointerEscapes = true;

                std::vector<unsigned> positions = getArgPositions( CI, V );
                for( auto position : positions )
                {
                    Value* argument     = CI->getArgOperand( position );
                    Type*  argType      = argument->getType();
                    Value* inputPointer = rawPtr;

                    // Cast to the right type.
                    if( argType != m_llvmManager->getI64Type() )
                    {
                        RT_ASSERT_MSG( argType->isPointerTy(),
                                       "We are passing a pointer as parameter of incompatible type while translating ("
                                           + m_name + ")." );
                        inputPointer = irb.CreateIntToPtr( inputPointer, argType );
                    }

                    CI->setArgOperand( position, inputPointer );
                }
                continue;
            }


            Function* callee = CI->getCalledFunction();
            RT_ASSERT( callee != nullptr );

            StringRef functionName = callee->getName();

            // Cast.
            if( functionName.startswith( "_castUberP" ) )
            {
                CI->replaceAllUsesWith( UberP );
                continue;
            }

            // Atomic instruction, aguments: address and a value.
            // Atomically change the value stored in the address.
            // Returns the content of the memory in the given address before the modification.
            if( isAtomicFunction( functionName ) )
            {
                const AtomicOpType atomicOp = getAtomicOperator( functionName );

                m_hasStores          = true;
                Value* secondOperand = nullptr;
                if( atomicOp == AtomicOpType::CAS )
                    secondOperand = CI->getArgOperand( 2 );
                else
                    secondOperand       = CI->getArgOperand( 1 );
                Type* returnType        = CI->getCalledFunction()->getReturnType();
                Type* secondOperandType = secondOperand->getType();

                RT_ASSERT_MSG( returnType == secondOperandType,
                               "Malformed call to optix.ptx.atom found while translating (" + m_name + ")" );
                auto i32Ty = m_llvmManager->getI32Type();

                // Create the array of types for the parameters of the MemAccess function.
                Type* paramTypes[] = {m_llvmManager->getStatePtrType(),
                                      UberPointer::getType( m_llvmManager ),
                                      i32Ty,
                                      m_llvmManager->getI8Type(),
                                      secondOperandType,
                                      secondOperandType};
                FunctionType* memFunType = FunctionType::get( returnType, paramTypes, false );

                // Notice that the type of the atomic operand is always casted to an int. So no name mangling is required.
                std::string memFunName    = UBER_POINTER_ATOMIC_OP + "." + getTypeName( returnType );
                Function*   memFun        = cast<Function>( m_module->getOrInsertFunction( memFunName, memFunType ) );
                Value*      atomicOpConst = ConstantInt::get( i32Ty, static_cast<int>( atomicOp ) );

                Value* compareValue = nullptr;
                if( atomicOp == AtomicOpType::CAS )
                    compareValue = CI->getArgOperand( 1 );
                else
                    compareValue = UndefValue::get( secondOperandType );

                LLVMContext& llvmContext    = m_llvmManager->llvmContext();
                Value*       subElementType = nullptr;
                if( returnType == m_llvmManager->getI32Type() )
                    subElementType = ConstantInt::get( llvmContext, APInt( 8, static_cast<int>( AtomicSubElementType::INT32 ) ) );
                else if( returnType == m_llvmManager->getFloatType() )
                    subElementType =
                        ConstantInt::get( llvmContext, APInt( 8, static_cast<int>( AtomicSubElementType::FLOAT32 ) ) );
                else if( returnType == m_llvmManager->getI64Type() )
                    subElementType = ConstantInt::get( llvmContext, APInt( 8, static_cast<int>( AtomicSubElementType::INT64 ) ) );
                else
                {
                    throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ),
                                        "Atomics attempted on buffer with unsupported type." );
                }

                // Create the vector of arguments for the call instruction.
                Value* args[] = {caller->arg_begin(), UberP,        atomicOpConst,
                                 subElementType,      compareValue, secondOperand};

                // Call the function.
                CallInst* call = CallInst::Create( memFun, args, "" );
                call->insertAfter( CI );

                CI->replaceAllUsesWith( call );

                // We have to explicitly remove the function call from the function since the LLVM optimizer cannot remove it by itself.
                // This is not required for normal stores since dead stores can be optimized out.
                CI->eraseFromParent();

                continue;
            }

            // Print write.
            if( functionName == "rt_print_write32" )
            {
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ),
                                    "Pointers to rtVariables and rtBuffers (" + m_name
                                        + ") cannot be printed with rtPrintf." );
            }

            // We replace the original call with a special call that is then canonicalized by C14n::canonicalizeComplexFunctions.
            if( functionName.startswith( "_rt_trace_" ) )
            {
                std::string   rtTraceLwrrentPayload = functionName.str() + "_global_payload";
                Function*     calleeFn              = CI->getCalledFunction();
                FunctionType* type                  = calleeFn->getFunctionType();
                Module*       module                = calleeFn->getParent();
                Constant* tmp = module->getOrInsertFunction( rtTraceLwrrentPayload, type, calleeFn->getAttributes() );
                Function* toCall = dyn_cast<Function>( tmp );

                std::vector<Value*> args;
                unsigned            argNumber = CI->getNumArgOperands();
                args.reserve( argNumber );
                for( unsigned index = 0; index < argNumber; ++index )
                    args.push_back( CI->getArgOperand( index ) );

                CallInst* newCall = irb.CreateCall( toCall, args );
                CI->replaceAllUsesWith( newCall );
                CI->eraseFromParent();

                continue;
            }

            // Handle here all calls to plain functions.
            // Extract the pointer field from the uber pointer and cast it appropriately to the type of the argument.
            // Mark the pointer as escaped.

            // Create a call to _UberPointerGetAddress, this function is a place holder that identifies that a pointer escapes.
            // rawPtr is a i64.
            Value* rawPtr = createUberMemAccessCall( UberP, insertBefore, caller, nullptr, AccessType::GET_ADDRESS, nullptr );
            m_pointerEscapes = true;

            std::vector<unsigned> positions = getArgPositions( CI, V );
            for( auto position : positions )
            {
                Value* argument     = CI->getArgOperand( position );
                Type*  argType      = argument->getType();
                Value* inputPointer = rawPtr;

                // Cast to the right type.
                if( argType != m_llvmManager->getI64Type() )
                {
                    RT_ASSERT_MSG( argType->isPointerTy(),
                                   "Non-64 bit int found passed as a pointer argument while translating (" + m_name
                                       + ")" );
                    inputPointer = irb.CreateIntToPtr( inputPointer, argType );
                }

                CI->setArgOperand( position, inputPointer );
            }
        }
        else if( CastInst* CICast = dyn_cast<CastInst>( U ) )
        {
            translate( CICast, UberP );
        }
        else if( PHINode* PN = dyn_cast<PHINode>( U ) )
        {
            corelib::CoreIRBuilder irb{PN};
            unsigned int           numIncoming   = PN->getNumIncomingValues();
            Type*                  uberPointerTy = UberPointer::getType( m_llvmManager );
            PHINode*               new_PN        = irb.CreatePHI( uberPointerTy, numIncoming, "UberP.phi" );
            for( unsigned int i = 0; i < numIncoming; ++i )
            {
                Value*      incoming = PN->getIncomingValue( i );
                BasicBlock* block    = PN->getIncomingBlock( i );
                if( incoming == V )
                {
                    // here's the value we are chasing
                    new_PN->addIncoming( UberP, block );
                }
                else
                {
                    // We want to insert the cast in the incoming basic block just before the
                    // terminator.  Note that if 'incoming' is already an UberPointer then
                    // createUberPointerCast simply returns it (i.e. not cast inserted).
                    incoming = createUberPointerCast( incoming, block->getTerminator() );
                    new_PN->addIncoming( incoming, block );
                }
            }
            translate( PN, new_PN );
        }
        else if( SelectInst* SI = dyn_cast<SelectInst>( U ) )
        {
            RT_ASSERT( SI->getNumOperands() == 3 );
            RT_ASSERT( SI->getCondition() != V );                               // Can't use UberP in condition
            RT_ASSERT( V == SI->getOperand( 1 ) || V == SI->getOperand( 2 ) );  // V needs to be one of the arguments.

            // Figure out who is whom
            int us = 1, them = 2;
            if( V != SI->getOperand( us ) )
                std::swap( us, them );

            Instruction*           insertBefore = SI;
            corelib::CoreIRBuilder irb{insertBefore};
            Value*                 new_operands[3];
            new_operands[0]    = SI->getCondition();
            new_operands[us]   = UberP;
            new_operands[them] = createUberPointerCast( SI->getOperand( them ), insertBefore );

            Value* new_SI = irb.CreateSelect( new_operands[0], new_operands[1], new_operands[2], "UberP.select" );
            translate( SI, new_SI );
        }
        else if( GetElementPtrInst* GEP = dyn_cast<GetElementPtrInst>( U ) )
        {
            // This case is tricky.  Since we can't compute the constant offset (presumably
            // because the GEP has a non-const argument), we need to compute the offset by
            // taking the difference between the pointer to the sub-element and the pointer to
            // the beginning of element.
            //
            // y = gep(p1,0,i);
            // y_uber.offset = y-gep(p1,0,0);
            //
            // This is lwrrently unimplemented until we get a case for this.
            errs() << "GEP: " << *GEP << "\n";
            RT_ASSERT_MSG( false,
                           "Unexpected instruction translation to UberPointer while translating (" + m_name + ")" );
        }
        else if( CmpInst* CI = dyn_cast<CmpInst>( U ) )
        {
            RT_ASSERT( V == CI->getOperand( 0 ) || V == CI->getOperand( 1 ) );  // V needs to be one of the arguments.

            // Figure out who is whom
            int us = 0, them = 1;
            if( V != CI->getOperand( us ) )
                std::swap( us, them );

            // Handle the special case of comparing to NULL
            ConstantInt* themConstantInt = dyn_cast<ConstantInt>( CI->getOperand( them ) );
            if( themConstantInt && themConstantInt->isZero() )
            {
                ConstantInt* nonZero = llvm::ConstantInt::get( themConstantInt->getType(), 1, CI->isSigned() );
                CI->setOperand( us, nonZero );
            }
            // All other comparisons are not allowed
            else
            {
                throw CompileError(RT_EXCEPTION_INFO, LLVMErrorInfo(U),
					"Pointer to rtVariable (" + m_name + ") cannot be used in a comparison except for the special case of comparison to zero (NULL check).");
            }
        }
        else if( ReturnInst* retInst = dyn_cast<ReturnInst>( U ) )
        {
            Instruction* insertBefore = retInst;
            Function*    caller       = insertBefore->getParent()->getParent();
            Value* rawPtr = createUberMemAccessCall( UberP, insertBefore, caller, nullptr, AccessType::GET_ADDRESS, nullptr );
            m_pointerEscapes = true;

            Type* retType = retInst->getOperand( 0 )->getType();

            // Cast to the right type.
            if( retType != m_llvmManager->getI64Type() )
            {
                RT_ASSERT_MSG( retType->isPointerTy(),
                               "Returning a pointer of unsupported type while translating (" + m_name + ")." );
                rawPtr = corelib::CoreIRBuilder{insertBefore}.CreateIntToPtr( rawPtr, retType );
            }

            retInst->setOperand( 0, rawPtr );
        }
        else
        {
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( U ),
                                "UberPointer::translate - Unexpected instruction translation to UberPointer while "
                                "translating ("
                                    + m_name + ")." );
        }
    }
}

// Create a call to one of the four functions _UberPointerMemRead, _UberPointerMemWrite, _UberPointerMemAtomicOp,
// _UberPointerGetAddress. These are placeholder functions that identify a load or store and keep track of what element
// of a buffer is accessed through the UberPointer. _UberPointerMemRead returns the value read from memory, while
// _UberPointerMemWrite returns void. These functions calls do not have references to the global variable that is
// accessed. This information is saved in the runtime memory. The UberPointer kind allows us to perform a lookup in the
// UberPointerSet, this contains references to the setter and getter functions whose name contain the variable reference
// id, variable token and variable name of the buffer that is used.
CallInst* UberPointerTransform::createUberMemAccessCall( Value*       UberP,
                                                         Instruction* insertBefore,
                                                         Function*    function,
                                                         Type*        type,
                                                         AccessType   accessType,
                                                         Value*       storeValue )
{
    LLVMContext& context = m_llvmManager->llvmContext();

    // Create the vector of types for the parameters of the MemAccess function.
    std::vector<Type*> paramTypes = {m_llvmManager->getStatePtrType(), UberPointer::getType( m_llvmManager )};
    if( accessType == AccessType::WRITE )
        paramTypes.push_back( type );
    else if( accessType == AccessType::ATOMIC_OP )
        paramTypes.push_back( Type::getInt8PtrTy( context ) );

    FunctionType* memFunType = FunctionType::get( Type::getInt8PtrTy( context ), paramTypes, false );
    std::string   functionName;
    switch( accessType )
    {
        case AccessType::READ:
        {
            functionName = UBER_POINTER_MEM_READ + "." + getTypeName( type );
            memFunType   = FunctionType::get( type, paramTypes, false );
            break;
        }
        case AccessType::WRITE:
            functionName = UBER_POINTER_MEM_WRITE + "." + getTypeName( type );
            memFunType   = FunctionType::get( Type::getVoidTy( context ), paramTypes, false );
            break;
        case AccessType::ATOMIC_OP:
            functionName = UBER_POINTER_ATOMIC_OP + "." + getTypeName( type );
            break;
        case AccessType::GET_ADDRESS:
            functionName = UBER_POINTER_GET_ADDRESS;
            memFunType   = FunctionType::get( m_llvmManager->getI64Type(), paramTypes, false );
            break;
        default:
            RT_ASSERT_FAIL_MSG( "Access type unknown found while translating (" + m_name + ")" );
            break;
    }

    Function* memFun = cast<Function>( m_module->getOrInsertFunction( functionName, memFunType ) );

    // Create the vector of arguments for the call instruction.
    std::vector<Value*> args{function->arg_begin(), UberP};
    if( accessType == AccessType::WRITE || accessType == AccessType::ATOMIC_OP )
        args.push_back( storeValue );

    // Call the function.
    assert( insertBefore );
    return corelib::CoreIRBuilder{insertBefore}.CreateCall( memFun, args );
}

Value* UberPointerTransform::createUberPointerCast( Value* V, Instruction* insertBefore )
{
    Type* uberPointerTy = UberPointer::getType( m_llvmManager );

    // Trivial colwersion
    if( V->getType() == uberPointerTy )
        return V;

    bool isPointerTy = V->getType()->isPointerTy();
    // Allow both pointer and pointer sized integers
    RT_ASSERT( isPointerTy || V->getType() == m_llvmManager->getSizeTType() );

    // Build the cast function and call arguments.  The function should take a char*
    // and return an UberPointer.  Any pointer argument into the function should be
    // cast to char* to avoid having to create different versions of the function.
    Type*         paramTypes  = m_llvmManager->getI64Type();
    FunctionType* castFunType = FunctionType::get( uberPointerTy, paramTypes, false );

    // Set the nounwind and readnone attributes on the function.
    // This allows llvm to remove any calls to this function where the result
    // of the call is dead.  During the process of colwerting pointers into UberPointer
    // values, there are a number of dangling calls created.
    Function* castFun = cast<Function>( m_module->getOrInsertFunction( "_castUberP", castFunType ) );
    castFun->setDoesNotThrow();
    castFun->setDoesNotAccessMemory();

    corelib::CoreIRBuilder irb{insertBefore};
    Value*                 args = nullptr;
    if( isPointerTy )
        args = irb.CreatePtrToInt( V, m_llvmManager->getI64Type(), "cast.uberp" );
    else
        args = V;

    // Call the cast function
    return irb.CreateCall( castFun, args );
}

// UberPointerSet::Info Class.
// -----------------------------------------------------------------------------
void UberPointerSet::finalizeUberPointerGetsAndSets( Module* module )
{
    // Preserve this order in the finalization.
    // Finalize Writes first, then finalize reads.
    // This is because writes are expanded into reads which store to stack memory.
    std::vector<Function*> functions = getFunctions( module );

    for( Function* function : functions )
        if( function->getName().startswith( UberPointerTransform::UBER_POINTER_MEM_WRITE ) )
            finalizeUberPointerGetsOrSets( function, true );

    if( Function* function = module->getFunction( UberPointerTransform::UBER_POINTER_MEM_READ_TO_ALLOCA ) )
        finalizeUberPointerGetsToAlloca( function );

    for( Function* function : functions )
        if( function->getName().startswith( UberPointerTransform::UBER_POINTER_MEM_READ ) )
            finalizeUberPointerGetsOrSets( function, false );

    for( Function& function : *module )
    {
        if( function.getName().startswith( UberPointerTransform::UBER_POINTER_ATOMIC_OP ) )
        {
            finalizeUberPointerAtomic( &function );
        }
    }

    Function* getAddressFunction = module->getFunction( UberPointerTransform::UBER_POINTER_GET_ADDRESS );
    if( getAddressFunction )
        finalizeUberPointerGetAddress( getAddressFunction );
}

// -----------------------------------------------------------------------------
CallInst* UberPointerSet::generateRawAtomicCall( Value*       uberP,
                                                 Value*       operatorCode,
                                                 Value*       subElementType,
                                                 Value*       compareOperand,
                                                 Value*       operand,
                                                 const Info&  info,
                                                 Instruction* insertBefore )
{
    ConstantInt* atomicOperator = dyn_cast<ConstantInt>( operatorCode );
    RT_ASSERT( atomicOperator != nullptr );

    AtomicOpType op = static_cast<AtomicOpType>( atomicOperator->getZExtValue() );

    Type* type = operand->getType();

    corelib::CoreIRBuilder irb{insertBefore};

    AtomicSubElementType subType = valueToSubElementType( subElementType );
    switch( subType )
    {
        case AtomicSubElementType::INT32:
        {
            operand = irb.CreateTrunc( operand, m_llvmManager->getI32Type() );

            compareOperand = irb.CreateTrunc( compareOperand, m_llvmManager->getI32Type() );

            type = m_llvmManager->getI32Type();
        }
        break;
        case AtomicSubElementType::FLOAT32:
            operand = irb.CreateTrunc( operand, m_llvmManager->getI32Type() );
            operand = irb.CreateBitCast( operand, m_llvmManager->getFloatType() );

            compareOperand = irb.CreateTrunc( compareOperand, m_llvmManager->getI32Type() );
            compareOperand = irb.CreateBitCast( compareOperand, m_llvmManager->getFloatType() );

            type = m_llvmManager->getFloatType();
            break;
        case AtomicSubElementType::INT64:
            // No colwersion needed.
            break;
        default:
            RT_ASSERT_FAIL_MSG( "Unsupported AtomicSubElementType." );
    }

    std::string name           = getPTXAtomicFunction( op, type );
    Module*     module         = insertBefore->getParent()->getParent()->getParent();
    Function*   atomicFunction = module->getFunction( name );
    // If the name does not match the original function this triggers.
    RT_ASSERT( atomicFunction != nullptr );

    Value* rawPtr            = UberPointer::get<UberPointer::RawPointer>( uberP, "", insertBefore );
    rawPtr                   = irb.CreateIntToPtr( rawPtr, type->getPointerTo() );
    std::vector<Value*> args = {rawPtr};
    if( op == AtomicOpType::CAS )
        args.push_back( compareOperand );

    args.push_back( operand );

    return irb.CreateCall( atomicFunction, args );
}

// -----------------------------------------------------------------------------
CallInst* UberPointerSet::generateSingleAtomicCall( Function*    atomicFunc,
                                                    Value*       statePtr,
                                                    Value*       uberP,
                                                    Value*       operatorCode,
                                                    Value*       subElementType,
                                                    Value*       compareOperand,
                                                    Value*       operand,
                                                    const Info&  info,
                                                    Instruction* insertBefore )
{
    if( atomicFunc == nullptr )
        return generateRawAtomicCall( uberP, operatorCode, subElementType, compareOperand, operand, info, insertBefore );

    RT_ASSERT( atomicFunc != nullptr );
    RT_ASSERT_MSG( info.type == UberPointer::PointeeType::Buffer || info.type == UberPointer::PointeeType::BufferID,
                   "Performing atomic operation on unsupported pointer found while translating (" + info.name + ")." );

    auto module = atomicFunc->getParent();

    Value* bufferId    = UberPointer::get<UberPointer::ID>( uberP, "UberPointer.id", insertBefore );
    Value* offset      = UberPointer::get<UberPointer::Offset>( uberP, "UberP.offset", insertBefore );
    Value* index       = UberPointer::get<UberPointer::BufferIndex>( uberP, "UberP.index", insertBefore );
    Value* elementSize = UberPointer::get<UberPointer::ElementSize>( uberP, "UberP.eltSize", insertBefore );

    corelib::CoreIRBuilder irb{insertBefore};

    if( info.type == UberPointer::PointeeType::BufferID )
    {
        AtomicSetBufferElementFromIdBuilder asbId( module );

        asbId.setCanonicalState( statePtr )
            .setBufferId( bufferId )
            .setOffset( offset )
            .setElementSize( elementSize )
            .setSubElementType( subElementType )
            .setOperand( operand )
            .setCompareOperand( compareOperand )
            .setOperation( operatorCode );

        unsigned dimensions = asbId.getDimensionality( atomicFunc );
        // This switch deliberately omits break statements.
        switch( dimensions )
        {
            case 3:
            {
                Value* z = irb.CreateExtractValue( index, 2, info.name + ".z" );
                asbId.setZ( z );
            }
            case 2:
            {
                Value* y = irb.CreateExtractValue( index, 1, info.name + ".y" );
                asbId.setY( y );
            }
            case 1:
            {
                Value* x = irb.CreateExtractValue( index, 0, info.name + ".x" );
                asbId.setX( x );
                break;
            }
            default:
                RT_ASSERT_FAIL();
                break;
        }

        return asbId.create( atomicFunc, insertBefore );
    }
    else if( info.type == UberPointer::PointeeType::Buffer )
    {
        AtomicSetBufferElementBuilder asb( module );

        asb.setCanonicalState( statePtr )
            .setOffset( offset )
            .setElementSize( elementSize )
            .setOperand( operand )
            .setSubElementType( subElementType )
            .setCompareOperand( compareOperand )
            .setOperation( operatorCode );

        unsigned dimensions = asb.getDimensionality( atomicFunc );
        // This switch deliberately omits break statements.
        switch( dimensions )
        {
            case 3:
            {
                Value* z = irb.CreateExtractValue( index, 2, info.name + ".z" );
                asb.setZ( z );
            }
            case 2:
            {
                Value* y = irb.CreateExtractValue( index, 1, info.name + ".y" );
                asb.setY( y );
            }
            case 1:
            {
                Value* x = irb.CreateExtractValue( index, 0, info.name + ".x" );
                asb.setX( x );
                break;
            }
            default:
                RT_ASSERT_FAIL();
                break;
        }

        return asb.create( atomicFunc, insertBefore );
    }
    else
    {
        RT_ASSERT_FAIL_MSG( "Performing atomic operation on unsupported pointer found while translating (" + info.name
                            + ")." );
    }
}

void UberPointerSet::generateSingleGetOrSet( int idx, Value* UberP, Value* statePtr, Value* allocaptr, Type* type, Value* toStore, Instruction* insertBefore )
{
    RT_ASSERT( idx >= 0 && idx < static_cast<int>( m_uptrList.size() ) );
    Info& info = m_uptrList[idx];

    corelib::CoreIRBuilder irb{insertBefore};

    // Build arguments
    SmallVector<Value*, 5> args;
    args.push_back( statePtr );
    switch( info.type )
    {
        case UberPointer::PointeeType::Variable:
        {
            generateSingleGetVariable( info, UberP, statePtr, allocaptr, type, insertBefore );
            return;
        }
        case UberPointer::PointeeType::Payload:
        {
            generateSingleGetOrSetPayload( info, UberP, statePtr, allocaptr, type, toStore, insertBefore );
            return;
        }
        case UberPointer::PointeeType::LaunchIndex:
        case UberPointer::PointeeType::LaunchDim:
        case UberPointer::PointeeType::LwrrentRay:
        case UberPointer::PointeeType::LwrrentTime:
        case UberPointer::PointeeType::TMax:
        case UberPointer::PointeeType::SubframeIndex:
        {
            // No additional arguments needed.
            break;
        }
        case UberPointer::PointeeType::Attribute:
        {
            generateSingleGetOrSetAttribute( info, UberP, statePtr, allocaptr, type, toStore, insertBefore );
            return;
        }
        case UberPointer::PointeeType::Buffer:
        {
            generateSingleGetOrSetBufferElement( info, UberP, statePtr, allocaptr, type, toStore, insertBefore );
            return;
        }
        case UberPointer::PointeeType::BufferID:
        {
            generateSingleGetOrSetBufferElementFromId( info, UberP, statePtr, allocaptr, type, toStore, insertBefore );
            return;
        }
        case UberPointer::PointeeType::ExceptionDetail:
        {
            Value* tempIdx = UberPointer::get<UberPointer::ID>( UberP, info.name + ".idx", insertBefore );
            args.push_back( tempIdx );
        }
        break;
        case UberPointer::PointeeType::Raw:
        {
            if( type == nullptr )
                return;

            Value* pointer = UberPointer::get<UberPointer::RawPointer>( UberP, "raw_pointer", insertBefore );

            if( toStore == nullptr )
            {
                pointer            = irb.CreateIntToPtr( pointer, type->getPointerTo() );
                Value* loadedValue = irb.CreateLoad( pointer );
                Value* typedptr    = irb.CreateBitCast( allocaptr, type->getPointerTo(), info.name + ".ptr" );
                irb.CreateStore( loadedValue, typedptr );
            }
            else
            {
                pointer         = irb.CreateIntToPtr( pointer, type->getPointerTo() );
                Value* typedptr = irb.CreateBitCast( pointer, type->getPointerTo(), info.name + ".ptr" );
                irb.CreateStore( toStore, typedptr );
            }
            return;
        }
        case UberPointer::PointeeType::Unknown:
        {
            RT_ASSERT_FAIL_MSG( "UberPointer PointeeType cannot be Unknown during finalization while translating ("
                                + info.name + ")." );
            break;
        }
    }

    if( toStore != nullptr )
    {
        RT_ASSERT( info.setter != nullptr );
        FunctionType* fntype   = info.setter->getFunctionType();
        int           n        = fntype->getNumParams();
        Type*         ptrType  = fntype->getParamType( n - 1 )->getPointerTo();
        Value*        typedptr = irb.CreateBitCast( allocaptr, ptrType, info.name );
        Value*        value    = irb.CreateLoad( typedptr, info.name + ".value" );
        args.push_back( value );
        irb.CreateCall( info.setter, args );
    }
    else
    {
        RT_ASSERT( info.getter != nullptr );
        Type*  valueType = info.getter->getReturnType();
        Value* typedptr  = irb.CreateBitCast( allocaptr, valueType->getPointerTo(), info.name + ".ptr" );
        Value* value     = irb.CreateCall( info.getter, args, info.name );
        irb.CreateStore( value, typedptr );
    }
}

// -----------------------------------------------------------------------------
void UberPointerSet::generateSingleGetOrSetPayload( Info& info, Value* UberP, Value* statePtr, Value* allocaptr, Type* type, Value* toStore, Instruction* insertBefore )
{
    // For the payload we enable the use of sub-element accesses.
    // This means that we don't have a single function function to access the payload.
    // We will have multiple functions with different names, one for each type of the subfields for the payload.
    // This new approach does not match well with the current UberPointer infrastructure,
    // this is because we generate a unique access functions per payload during canonicalization (getOrAddUberPointerKind).
    // To cirlwmvent this limitation without restructuring the whole UberPointer infrastructure we build accessor functions on the fly.
    // These will be:
    // getPayloadValue.cp${CP_ID}.prd.${ElementSize}b.${GlobalName}.${typeMangledName}(statePtr, offset);
    // setPayloadValue.cp${CP_ID}.prd.${ElementSize}b.${GlobalName}.${typeMangledName}(statePtr, offset, valueToStore);

    std::vector<Value*>    args   = {statePtr};
    Module*                module = insertBefore->getParent()->getParent()->getParent();
    corelib::CoreIRBuilder irb{insertBefore};
    if( toStore != nullptr )
    {
        RT_ASSERT( info.setter != nullptr );
        FunctionType* fntype = info.setter->getFunctionType();

        RT_ASSERT( toStore->getType() == type );

        std::vector<Type*> params;
        auto               E = fntype->param_end();
        --E;
        for( FunctionType::param_iterator I = fntype->param_begin(); I != E; ++I )
            params.push_back( *I );
        params.push_back( m_llvmManager->getI64Type() );
        params.push_back( type );

        FunctionType* newFnType = FunctionType::get( m_llvmManager->getVoidType(), params, false );
        std::string   newName   = info.setter->getName().str() + "." + getTypeName( type );
        Function*     newFunction =
            dyn_cast<Function>( module->getOrInsertFunction( newName, newFnType, info.setter->getAttributes() ) );

        Value* offset = UberPointer::get<UberPointer::Offset>( UberP, "", insertBefore );
        args.push_back( offset );
        args.push_back( toStore );

        irb.CreateCall( newFunction, args );
    }
    else
    {
        RT_ASSERT( info.getter != nullptr );
        FunctionType*      fntype = info.getter->getFunctionType();
        std::vector<Type*> params = {};
        auto               E      = fntype->param_end();
        for( FunctionType::param_iterator I = fntype->param_begin(); I != E; ++I )
            params.push_back( *I );
        params.push_back( m_llvmManager->getI64Type() );

        FunctionType* newFnType = FunctionType::get( type, params, false );
        std::string   newName   = info.getter->getName().str() + "." + getTypeName( type );
        Function*     newFunction =
            dyn_cast<Function>( module->getOrInsertFunction( newName, newFnType, info.getter->getAttributes() ) );

        Value* offset = UberPointer::get<UberPointer::Offset>( UberP, "", insertBefore );
        args.push_back( offset );
        Value* loadedValue = irb.CreateCall( newFunction, args );

        Value* typedptr = irb.CreateBitCast( allocaptr, type->getPointerTo(), info.name + ".ptr" );
        irb.CreateStore( loadedValue, typedptr );
    }
}

void UberPointerSet::generateSingleGetOrSetAttribute( Info&              info,
                                                      llvm::Value*       UberP,
                                                      llvm::Value*       statePtr,
                                                      llvm::Value*       allocaptr,
                                                      llvm::Type*        type,
                                                      llvm::Value*       toStore,
                                                      llvm::Instruction* insertBefore )
{
    RT_ASSERT( type != nullptr );
    Module* module = insertBefore->getParent()->getParent()->getParent();

    Value*                 offset = UberPointer::get<UberPointer::Offset>( UberP, "", insertBefore );
    std::vector<Value*>    args   = {statePtr, offset};
    corelib::CoreIRBuilder irb{insertBefore};
    if( toStore != nullptr )
    {
        RT_ASSERT( info.setter != nullptr );
        Type*         paramTypes[] = {m_llvmManager->getStatePtrType(), m_llvmManager->getI64Type(), type};
        FunctionType* newFnType    = FunctionType::get( m_llvmManager->getVoidType(), paramTypes, false );
        std::string   newName      = info.setter->getName().str() + "." + getTypeName( type );
        Function*     newFunction =
            dyn_cast<Function>( module->getOrInsertFunction( newName, newFnType, info.setter->getAttributes() ) );
        args.push_back( toStore );
        irb.CreateCall( newFunction, args );
    }
    else
    {
        RT_ASSERT( info.getter != nullptr );
        Type*         paramTypes[] = {m_llvmManager->getStatePtrType(), m_llvmManager->getI64Type()};
        FunctionType* newFnType    = FunctionType::get( type, paramTypes, false );
        std::string   newName      = info.getter->getName().str() + "." + getTypeName( type );
        Function*     newFunction =
            dyn_cast<Function>( module->getOrInsertFunction( newName, newFnType, info.getter->getAttributes() ) );
        Value* value    = irb.CreateCall( newFunction, args, info.name );
        Value* typedptr = irb.CreateBitCast( allocaptr, type->getPointerTo(), info.name + ".ptr" );
        irb.CreateStore( value, typedptr );
    }
}

// -----------------------------------------------------------------------------
void UberPointerSet::generateSingleGetOrSetBufferElement( Info&        info,
                                                          Value*       UberP,
                                                          Value*       statePtr,
                                                          Value*       allocaptr,
                                                          Type*        type,
                                                          Value*       toStore,
                                                          Instruction* insertBefore )
{
    // getBufferElement...${typeMangledName}(statePtr, elementSize, offset, x, y, z,);
    // setBufferElement...${typeMangledName}(statePtr, elementSize, offset, x, y, z, valueToStore);

    Module*                module = insertBefore->getParent()->getParent()->getParent();
    corelib::CoreIRBuilder irb{insertBefore};

    Value* idx         = UberPointer::get<UberPointer::BufferIndex>( UberP, info.name + ".idx", insertBefore );
    Value* elementSize = UberPointer::get<UberPointer::ElementSize>( UberP, info.name + ".elt.size", insertBefore );
    Value* offset      = UberPointer::get<UberPointer::Offset>( UberP, "", insertBefore );
    std::vector<Value*> args = {statePtr, elementSize, offset};

    RT_ASSERT( info.getter != nullptr );
    unsigned int dimensions = info.getter->getFunctionType()->getNumParams() - 1;
    RT_ASSERT( 1 <= dimensions && dimensions <= 3 );
    std::vector<Type*> paramTypes = {m_llvmManager->getStatePtrType(), m_llvmManager->getI32Type()};

    if( toStore != nullptr )
    {
        RT_ASSERT( info.setter != nullptr );
        RT_ASSERT( toStore->getType() == type );

        // Param types: statePtrType (canonical state pointer), i32Ty (elementSize), i64Ty (offset), ElementType (type of the element to store), i64Ty ... (x, y, z).
        paramTypes.push_back( m_llvmManager->getI64Type() );
        paramTypes.insert( std::end( paramTypes ), dimensions, m_llvmManager->getI64Type() );
        paramTypes.push_back( type );

        FunctionType* newFnType = FunctionType::get( m_llvmManager->getVoidType(), paramTypes, false );
        std::string   newName   = info.setter->getName().str() + "." + getTypeName( type );
        Function*     newFunction =
            dyn_cast<Function>( module->getOrInsertFunction( newName, newFnType, info.setter->getAttributes() ) );

        for( unsigned int i = 0u; i < dimensions; i++ )
            args.push_back( irb.CreateExtractValue( idx, i, info.name + ".idx" ) );
        args.push_back( toStore );

        irb.CreateCall( newFunction, args );
    }
    else
    {
        // Param types: statePtrType (canonical state pointer), i32Ty (elementSize), i64Ty (offset), i64Ty ... (x, y, z).
        paramTypes.push_back( m_llvmManager->getI64Type() );
        paramTypes.insert( std::end( paramTypes ), dimensions, m_llvmManager->getI64Type() );

        FunctionType* newFnType = FunctionType::get( type, paramTypes, false );
        std::string   newName   = info.getter->getName().str() + "." + getTypeName( type );
        Function*     newFunction =
            dyn_cast<Function>( module->getOrInsertFunction( newName, newFnType, info.getter->getAttributes() ) );

        for( unsigned int i = 0u; i < dimensions; i++ )
            args.push_back( irb.CreateExtractValue( idx, i, info.name + ".idx" ) );

        Value* loadedValue = irb.CreateCall( newFunction, args );

        Value* typedptr = irb.CreateBitCast( allocaptr, type->getPointerTo(), info.name + ".ptr" );
        irb.CreateStore( loadedValue, typedptr );
    }
}

// -----------------------------------------------------------------------------
void UberPointerSet::generateSingleGetVariable( Info& info, Value* UberP, Value* statePtr, Value* allocaptr, Type* type, Instruction* insertBefore )
{
    RT_ASSERT( info.getter != nullptr );
    Module* module = insertBefore->getParent()->getParent()->getParent();

    corelib::CoreIRBuilder irb{insertBefore};
    Value*                 offset = UberPointer::get<UberPointer::Offset>( UberP, "", insertBefore );
    Value*                 args[] = {statePtr, offset, info.variableDefault};
    Type* paramTypes[] = {m_llvmManager->getStatePtrType(), m_llvmManager->getI64Type(), info.variableDefault->getType()};

    FunctionType* newFnType = FunctionType::get( type, paramTypes, false );
    std::string   newName   = info.getter->getName().str() + "." + getTypeName( type );
    Function*     newFunction =
        dyn_cast<Function>( module->getOrInsertFunction( newName, newFnType, info.getter->getAttributes() ) );

    Value* variableValue = irb.CreateCall( newFunction, args );

    Value* typedptr = irb.CreateBitCast( allocaptr, type->getPointerTo(), info.name + ".ptr" );
    irb.CreateStore( variableValue, typedptr );
}

// -----------------------------------------------------------------------------
void UberPointerSet::generateSingleGetOrSetBufferElementFromId( Info&        info,
                                                                Value*       UberP,
                                                                Value*       statePtr,
                                                                Value*       allocaptr,
                                                                Type*        type,
                                                                Value*       toStore,
                                                                Instruction* insertBefore )
{
    // getBufferElementFromId(statePtr, bufferId, elementSize, offset, x, y, z,);
    // setBufferElementFromId(statePtr, bufferId, elementSize, offset, x, y, z, valueToStore);
    Module*                module = insertBefore->getParent()->getParent()->getParent();
    corelib::CoreIRBuilder irb{insertBefore};

    Value* idx         = UberPointer::get<UberPointer::BufferIndex>( UberP, info.name + ".idx", insertBefore );
    Value* bufferId    = UberPointer::get<UberPointer::ID>( UberP, info.name + ".id", insertBefore );
    Value* elementSize = UberPointer::get<UberPointer::ElementSize>( UberP, info.name + ".elt.size", insertBefore );
    Value* offset      = UberPointer::get<UberPointer::Offset>( UberP, "", insertBefore );
    std::vector<Value*> args = {statePtr, bufferId, elementSize, offset};

    RT_ASSERT( info.getter != nullptr );
    unsigned int dimensions = info.getter->getFunctionType()->getNumParams() - 2;
    RT_ASSERT( 1 <= dimensions && dimensions <= 3 );
    Type*              i32Ty      = m_llvmManager->getI32Type();
    Type*              i64Ty      = m_llvmManager->getI64Type();
    std::vector<Type*> paramTypes = {m_llvmManager->getStatePtrType(), i32Ty, i32Ty, i64Ty};

    if( toStore != nullptr )
    {
        RT_ASSERT( info.setter != nullptr );
        RT_ASSERT( toStore->getType() == type );

        // Param types: statePtrType (canonical state pointer), i32Ty (bufferId), i32Ty (elementSize), i64Ty (offset), ElementType (type of the element to store), i64Ty ... (x, y, z).
        paramTypes.insert( std::end( paramTypes ), dimensions, m_llvmManager->getI64Type() );
        paramTypes.push_back( type );

        FunctionType* newFnType = FunctionType::get( m_llvmManager->getVoidType(), paramTypes, false );
        std::string   newName   = info.setter->getName().str() + "." + getTypeName( type );
        Function*     newFunction =
            dyn_cast<Function>( module->getOrInsertFunction( newName, newFnType, info.setter->getAttributes() ) );

        for( unsigned int i = 0u; i < dimensions; i++ )
            args.push_back( irb.CreateExtractValue( idx, i, info.name + ".idx" ) );
        args.push_back( toStore );

        irb.CreateCall( newFunction, args );
    }
    else
    {
        // Param types: statePtrType (canonical state pointer), i32Ty (bufferId), i32Ty (elementSize), i64Ty (offset), i64Ty ... (x, y, z).
        paramTypes.insert( std::end( paramTypes ), dimensions, m_llvmManager->getI64Type() );

        FunctionType* newFnType = FunctionType::get( type, paramTypes, false );
        std::string   newName   = info.getter->getName().str() + "." + getTypeName( type );
        Function*     newFunction =
            dyn_cast<Function>( module->getOrInsertFunction( newName, newFnType, info.getter->getAttributes() ) );

        for( unsigned int i = 0u; i < dimensions; i++ )
            args.push_back( irb.CreateExtractValue( idx, i, info.name + ".idx" ) );

        Value* loadedValue = irb.CreateCall( newFunction, args );

        Value* typedptr = irb.CreateBitCast( allocaptr, type->getPointerTo(), info.name + ".ptr" );
        irb.CreateStore( loadedValue, typedptr );
    }
}

// -----------------------------------------------------------------------------
Value* UberPointerSet::generateSingleGetAddr( int kind, Value* uberP, Value* statePtr, Instruction* insertBefore )
{
    RT_ASSERT( kind >= 0 && kind < static_cast<int>( m_uptrList.size() ) );
    Info&                  info   = m_uptrList[kind];
    auto                   module = insertBefore->getParent()->getParent()->getParent();
    corelib::CoreIRBuilder irb{insertBefore};

    Function* getAddress = info.getAddress;

    Value* elementSize = UberPointer::get<UberPointer::ElementSize>( uberP, "get.address.element.size", insertBefore );
    Value* index       = UberPointer::get<UberPointer::BufferIndex>( uberP, "UberP.index", insertBefore );
    Value* offset      = UberPointer::get<UberPointer::Offset>( uberP, "UberP.offset", insertBefore );
    CallInst* call     = nullptr;

    switch( info.type )
    {
        case UberPointer::PointeeType::Buffer:
        {
            GetBufferElementAddressBuilder builder( module );

            unsigned dimensions = builder.getDimensionality( getAddress );

            builder.setCanonicalState( statePtr ).setElementSize( elementSize );

            // This switch deliberately omits break statements.
            switch( dimensions )
            {
                case 3:
                {
                    Value* z = irb.CreateExtractValue( index, 2, info.name + ".z" );
                    builder.setZ( z );
                }
                case 2:
                {
                    Value* y = irb.CreateExtractValue( index, 1, info.name + ".y" );
                    builder.setY( y );
                }
                case 1:
                {
                    Value* x = irb.CreateExtractValue( index, 0, info.name + ".x" );
                    builder.setX( x );
                    break;
                }
                default:
                {
                    RT_ASSERT_FAIL();
                    break;
                }
            }

            RT_ASSERT_MSG( getAddress != nullptr, "Address accessor function not created." );
            call = builder.create( getAddress, insertBefore );
            break;
        }
        case UberPointer::PointeeType::BufferID:
        {
            Value* bufferId = UberPointer::get<UberPointer::ID>( uberP, "UberP.id", insertBefore );

            GetBufferElementAddressFromIdBuilder builder( module );

            unsigned dimensions = builder.getDimensionality( getAddress );

            builder.setCanonicalState( statePtr ).setElementSize( elementSize ).setBufferId( bufferId );

            // This switch deliberately omits break statements.
            switch( dimensions )
            {
                case 3:
                {
                    Value* z = irb.CreateExtractValue( index, 2, info.name + ".z" );
                    builder.setZ( z );
                }
                case 2:
                {
                    Value* y = irb.CreateExtractValue( index, 1, info.name + ".y" );
                    builder.setY( y );
                }
                case 1:
                {
                    Value* x = irb.CreateExtractValue( index, 0, info.name + ".x" );
                    builder.setX( x );
                    break;
                }
                default:
                {
                    RT_ASSERT_FAIL();
                    break;
                }
            }

            RT_ASSERT_MSG( getAddress != nullptr, "Address accessor function not created." );
            call = builder.create( getAddress, insertBefore );
            break;
        }
        case UberPointer::PointeeType::Payload:
        {
            RT_ASSERT_MSG( getAddress != nullptr, "Address accessor function not created." );
            call = GetPayloadAddressCall::create( getAddress, statePtr, insertBefore );
            break;
        }
        case UberPointer::PointeeType::Attribute:
        {
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( uberP ),
                                "Taking the address of an attribute (" + info.name + ") is not supported." );
            break;
        }
        case UberPointer::PointeeType::LwrrentRay:
        {
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( uberP ),
                                "Taking the address of the current ray is not supported." );
            break;
        }
        case UberPointer::PointeeType::LwrrentTime:
        {
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( uberP ),
                                "Taking the address of the current time is not supported." );
            break;
        }
        case UberPointer::PointeeType::Raw:
        {
            return UberPointer::get<UberPointer::RawPointer>( uberP, "raw", insertBefore );
        }
        case UberPointer::PointeeType::Variable:
        {
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( uberP ),
                                "Taking the address of an rtVariable (" + info.name + ") is not supported." );
        }
        default:
        {
            RT_ASSERT_FAIL_MSG( "Escaping pointer of unsupported type while translating (" + info.name + ")." );
            break;
        }
    }
    Type*  i8PtrType   = irb.getInt8PtrTy();
    Value* i8Ptr       = irb.CreateIntToPtr( call, i8PtrType, ".toi8" );
    Value* i8Offsetptr = irb.CreateGEP( i8Ptr, offset, "i8offsetptr" );
    return irb.CreatePtrToInt( i8Offsetptr, m_llvmManager->getI64Type() );
}

int UberPointerSet::getOrInsertRawAccessEntry( Module* module )
{
    Type*         voidTy       = m_llvmManager->getVoidType();
    FunctionType* functionType = FunctionType::get( voidTy, {}, false );
    Function*     dummyGetter  = dyn_cast<Function>( module->getOrInsertFunction( "raw_uber_get", functionType ) );
    return getOrAddUberPointerKind( "raw", dummyGetter, nullptr, nullptr, nullptr, UberPointer::PointeeType::Raw, nullptr );
}

void UberPointerSet::checkAliasingRules( const llvm::BitVector& validKinds )
{
    // Make sure that we are not aliasing pointers to buffers with pointers of other types.
    bool isABuffer    = false;
    bool isNotABuffer = false;
    for( int kind = validKinds.find_first(); kind != -1; kind = validKinds.find_next( kind ) )
    {
        RT_ASSERT( kind >= 0 && kind < static_cast<int>( m_uptrList.size() ) );
        Info&                    info = m_uptrList[kind];
        UberPointer::PointeeType type = info.type;

        if( type == UberPointer::PointeeType::Buffer || type == UberPointer::PointeeType::BufferID )
            isABuffer = true;
        else
        {
            if( type != UberPointer::PointeeType::Raw )
                isNotABuffer = true;
        }
    }

    if( isABuffer == true && isNotABuffer == true )
        throw CompileError( RT_EXCEPTION_INFO,
                            "Pointers to buffers found to be aliasing with pointers to other types." );
}

void UberPointerSet::finalizeUberPointerAtomic( Function* function )
{
    LLVMContext&        llvmContext = function->getContext();
    std::vector<Value*> toDelete;
    unsigned            operandsNumber = function->arg_size();

    // Cache all the calls to the accessor function so not to risk to ilwalidate the iterator while modifying the code.
    for( CallInst* CI : getCallsToFunction( function ) )
    {
        corelib::CoreIRBuilder irb{CI};

        Value* statePtr        = CI->getOperand( 0 );
        Value* uberP           = CI->getOperand( 1 );
        Value* operatorCode    = CI->getOperand( 2 );
        Value* subElementType  = CI->getOperand( 3 );
        Value* originalOperand = CI->getOperand( operandsNumber - 1 );

        Value* operand        = nullptr;
        Value* compareOperand = nullptr;

        // The atomic setter is now always use i64 operands and i64 return value, so need to
        // colwert the operand in the call to i64
        if( originalOperand->getType() == irb.getInt64Ty() )
        {
            operand        = originalOperand;
            compareOperand = CI->getOperand( operandsNumber - 2 );
        }
        else if( originalOperand->getType() == irb.getInt32Ty() )
        {
            // zext i32 to i64
            operand        = irb.CreateZExt( originalOperand, irb.getInt64Ty(), "" );
            compareOperand = irb.CreateZExt( CI->getOperand( operandsNumber - 2 ), irb.getInt64Ty(), "" );
        }
        else if( originalOperand->getType() == irb.getFloatTy() )
        {
            // Colwert float to i64
            operand = irb.CreateBitCast( originalOperand, irb.getInt32Ty(), "" );
            operand = irb.CreateZExt( operand, irb.getInt64Ty() );

            compareOperand = irb.CreateBitCast( CI->getOperand( operandsNumber - 2 ), irb.getInt32Ty(), "" );
            compareOperand = irb.CreateZExt( compareOperand, irb.getInt64Ty() );
        }


        int             n = m_uptrInfo.size();
        llvm::BitVector validKinds( n, false );
        findValidUberPointerKinds( validKinds, uberP );
        int validKindsNumber = validKinds.count();

        // Make sure the atomic operation is done on a valid pointee type (Buffer, BufferID, and Raw)
        for( int kind = validKinds.find_first(); kind != -1; kind = validKinds.find_next( kind ) )
        {
            Info& info = m_uptrList[kind];
            if( !( info.type == UberPointer::PointeeType::Buffer || info.type == UberPointer::PointeeType::BufferID
                   || info.type == UberPointer::PointeeType::Raw ) )
            {
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ),
                                    "Atomic operation attempted on invalid pointee type." );
            }
        }

        Instruction* insertBefore = CI;
        if( validKindsNumber == 1 )
        {
            Info& info = m_uptrList[validKinds.find_first()];

            Function* atomicFunc = info.atomic;
            RT_ASSERT( atomicFunc != nullptr );
            CallInst* newCall = generateSingleAtomicCall( atomicFunc, statePtr, uberP, operatorCode, subElementType,
                                                          compareOperand, operand, info, insertBefore );
            Value* result = nullptr;

            if( CI->getType() == irb.getFloatTy() )
            {
                Value* trunk = irb.CreateTrunc( newCall, irb.getInt32Ty() );
                result       = irb.CreateBitCast( trunk, CI->getType(), "" );
            }
            else
                result = irb.CreateTrunc( newCall, CI->getType(), "" );

            CI->replaceAllUsesWith( result );
            CI->eraseFromParent();
        }
        else
        {
            checkAliasingRules( validKinds );

            Function* usingFunction = insertBefore->getParent()->getParent();

            // The return type of the atomic setter is always i64, so create an i64 alloca.
            auto allocaptr =
                corelib::CoreIRBuilder{corelib::getFirstNonAlloca( usingFunction )}.CreateAlloca( irb.getInt64Ty(),
                                                                                                  irb.getInt32( 1 ) );
            allocaptr->setAlignment( 16 );

            Value*      selector    = UberPointer::get<UberPointer::UPKind>( uberP, "uberptr.selector", insertBefore );
            BasicBlock* lwrrentBB   = insertBefore->getParent();
            BasicBlock* successorBB = lwrrentBB->splitBasicBlock( insertBefore );
            irb.SetInsertPoint( insertBefore );

            // Split basic block inserts an unconditional branch to the old code.  Remove it.
            Instruction* branch = lwrrentBB->getTerminator();
            RT_ASSERT( branch && isa<BranchInst>( branch ) );
            branch->eraseFromParent();

            // Switch requires a default case.  Make a basic block and mark it as unreachable.
            BasicBlock* trapBlock = BasicBlock::Create( llvmContext, "uberptr.illegal", usingFunction, successorBB );
            corelib::CoreIRBuilder{trapBlock}.CreateUnreachable();

            // Insert the switch, which will become the new terminator instruction
            BasicBlock* insertBB = trapBlock;  // Insert other blocks before here
            SwitchInst* sin = corelib::CoreIRBuilder{lwrrentBB}.CreateSwitch( selector, trapBlock, validKindsNumber );

            // Generate load/stores for the valid possibilities
            for( int kind = validKinds.find_first(); kind != -1; kind = validKinds.find_next( kind ) )
            {
                RT_ASSERT( kind >= 0 && kind < static_cast<int>( m_uptrList.size() ) );
                Info&     info       = m_uptrList[kind];
                Function* atomicFunc = info.atomic;

                // Create a new block for this case
                BasicBlock* newBlock = BasicBlock::Create( llvmContext, "uberptr." + info.name, usingFunction, insertBB );
                ConstantInt* idxval = ConstantInt::get( Type::getInt32Ty( llvmContext ), kind );
                sin->addCase( idxval, newBlock );

                // Insert unconditional branch back to the original code
                Instruction*           insertBeforeBranch = corelib::CoreIRBuilder{newBlock}.CreateBr( successorBB );
                corelib::CoreIRBuilder irb{insertBeforeBranch};
                Value* newCall = generateSingleAtomicCall( atomicFunc, statePtr, uberP, operatorCode, subElementType,
                                                           compareOperand, operand, info, insertBeforeBranch );
                Value* tmp = irb.CreateBitCast( allocaptr, CI->getType()->getPointerTo() );

                if( atomicFunc != nullptr )
                {
                    // newCall is always 64 bit if atomicFunction isn't nullptr so need to cast it to the
                    // type of CI.
                    if( CI->getType() == irb.getFloatTy() )
                    {
                        Value* trunk = irb.CreateTrunc( newCall, irb.getInt32Ty() );
                        newCall      = irb.CreateBitCast( trunk, CI->getType(), "" );
                    }
                    else
                    {
                        newCall = irb.CreateTrunc( newCall, CI->getType(), "" );
                    }
                }
                else
                {
                    // In this case a raw atomic call is created which always follows the type of CI.
                    RT_ASSERT( CI->getType() == newCall->getType() );
                }

                irb.CreateStore( newCall, tmp );
            }

            Instruction* insertPoint = successorBB->getFirstNonPHI();
            // Create a load from the alloca.
            Value* result = corelib::CoreIRBuilder{insertPoint}.CreateLoad( allocaptr );

            if( CI->getType() == irb.getFloatTy() )
            {
                Value* trunk = irb.CreateTrunc( result, irb.getInt32Ty() );
                result       = irb.CreateBitCast( trunk, irb.getFloatTy(), "" );
            }
            else
                result = irb.CreateTrunc( result, CI->getType(), "" );

            CI->replaceAllUsesWith( result );
            CI->eraseFromParent();
        }
    }
}

void UberPointerSet::finalizeUberPointerGetsOrSets( Function* fn, bool isPut )
{
    Module*             module      = fn->getParent();
    LLVMContext&        llvmContext = module->getContext();
    std::vector<Value*> toDelete;

    getOrInsertRawAccessEntry( module );

    for( CallInst* CI : getCallsToFunction( fn ) )
    {
        Value* statePtr = CI->getOperand( 0 );
        Value* UberP    = CI->getOperand( 1 );

        // Compute which getters (or setters) are necessary for this uberpointer reference.
        int             n = m_uptrInfo.size();
        llvm::BitVector validKinds( n, false );
        findValidUberPointerKinds( validKinds, UberP );

        // Determine the size of the alloca as the largest of the possible types
        size_t                   maxSize = 0;
        DataLayout               DL( module );
        UberPointer::PointeeType pointeeType = UberPointer::PointeeType::Unknown;
        for( int kind = validKinds.find_first(); kind != -1; kind = validKinds.find_next( kind ) )
        {
            RT_ASSERT( kind >= 0 && kind < static_cast<int>( m_uptrList.size() ) );
            Info& info      = m_uptrList[kind];
            Type* valueType = info.getter->getReturnType();

            if( valueType->isVoidTy() )
                continue;

            size_t size = DL.getTypeStoreSize( valueType );
            if( size > maxSize )
                maxSize = size;

            pointeeType = info.type;
        }
        RT_ASSERT( maxSize != 0 );
        int validKindsNumber = validKinds.count();

        Instruction* insertBefore  = CI;
        Function*    usingFunction = insertBefore->getParent()->getParent();
        Value*       allocaptr     = nullptr;
        Type*        type          = nullptr;
        Value*       toStore       = nullptr;

        corelib::CoreIRBuilder irb{corelib::getFirstNonAlloca( usingFunction )};
        if( pointeeType == UberPointer::PointeeType::Payload || pointeeType == UberPointer::PointeeType::Buffer
            || pointeeType == UberPointer::PointeeType::BufferID || pointeeType == UberPointer::PointeeType::Attribute
            || pointeeType == UberPointer::PointeeType::Variable )
        {
            if( !isPut )
            {
                type    = fn->getReturnType();  // This type may not match the variable (e.g. <4 x float> instead of <3 x float> in sample5_test0).
                auto ap = irb.CreateAlloca( irb.getInt8Ty(),
                                            /*ArraySize*/ irb.getInt32( DL.getTypeAllocSize( type ) ) );
                ap->setAlignment( 16 );
                allocaptr = ap;
            }
            else
            {
                type    = CI->getArgOperand( 2 )->getType();
                toStore = CI->getArgOperand( CI->getNumArgOperands() - 1 );
            }
        }
        else
        {
            if( !isPut )
            {
                type = fn->getReturnType();
                // Create an alloca to hold return values
                auto ap = irb.CreateAlloca( irb.getInt8Ty(),
                                            /*ArraySize*/ irb.getInt32( maxSize ) );
                ap->setAlignment( 16 );
                allocaptr = ap;
            }
            else
            {
                irb.SetInsertPoint( insertBefore );
                toStore = CI->getArgOperand( CI->getNumArgOperands() - 1 );
                type    = CI->getArgOperand( 2 )->getType();
                // Create call to UberPointerMemRead.
                Type*         paramTypes[] = {m_llvmManager->getStatePtrType(), UberPointer::getType( m_llvmManager )};
                std::string   functionName = UberPointerTransform::UBER_POINTER_MEM_READ_TO_ALLOCA;
                FunctionType* memFunType =
                    FunctionType::get( Type::getInt8PtrTy( m_llvmManager->llvmContext() ), paramTypes, false );
                Function* memFun       = cast<Function>( module->getOrInsertFunction( functionName, memFunType ) );
                Value*    args[]       = {statePtr, UberP};
                CallInst* memRead      = irb.CreateCall( memFun, args );
                Value*    UberP_offset = UberPointer::get<UberPointer::Offset>( UberP, "UberP.offset", insertBefore );
                Value*    storePtr = addOffset( m_llvmManager->llvmDataLayout(), memRead, "", type->getPointerTo( 0 ),
                                             UberP_offset, insertBefore );
                // Replace the store
                irb.CreateStore( CI->getArgOperand( CI->getNumArgOperands() - 1 ), storePtr );
                allocaptr = memRead;
            }
        }

        if( validKindsNumber == 1 )
        {
            // There is only a single reference associated with this get/put
            // so generate it directly.  Technically the optimizer will
            // figure this out, but save it some work in the common case.
            generateSingleGetOrSet( validKinds.find_first(), UberP, statePtr, allocaptr, type, toStore, insertBefore );
        }
        else
        {
            // There are multiple getters applicable to this reference.
            // Dynamically switch between them.
            checkAliasingRules( validKinds );

            // Insert the discriminant and split the block in preparation
            // for adding the switch.
            Value*      selector    = UberPointer::get<UberPointer::UPKind>( UberP, "uberptr.selector", insertBefore );
            BasicBlock* lwrrentBB   = insertBefore->getParent();
            BasicBlock* successorBB = lwrrentBB->splitBasicBlock( insertBefore );

            // Split basic block inserts an unconditional branch to the old code.  Remove it.
            Instruction* branch = lwrrentBB->getTerminator();
            RT_ASSERT( branch && isa<BranchInst>( branch ) );
            branch->eraseFromParent();

            // Switch requires a default case.  Make a basic block and mark it as unreachable.
            BasicBlock* trapBlock = BasicBlock::Create( llvmContext, "uberptr.illegal", usingFunction, successorBB );
            corelib::CoreIRBuilder{trapBlock}.CreateUnreachable();

            // Insert the switch, which will become the new terminator instruction
            BasicBlock* insertBB = trapBlock;  // Insert other blocks before here
            SwitchInst* sin = corelib::CoreIRBuilder{lwrrentBB}.CreateSwitch( selector, trapBlock, validKindsNumber );

            // Generate load/stores for the valid possibilities
            for( int kind = validKinds.find_first(); kind != -1; kind = validKinds.find_next( kind ) )
            {
                RT_ASSERT( kind >= 0 && kind < static_cast<int>( m_uptrList.size() ) );
                Info& info = m_uptrList[kind];

                // Create a new block for this case
                BasicBlock* newBlock = BasicBlock::Create( llvmContext, "uberptr." + info.name, usingFunction, insertBB );
                ConstantInt* idxval = ConstantInt::get( Type::getInt32Ty( llvmContext ), kind );
                sin->addCase( idxval, newBlock );

                // Insert unconditional branch back to the original code
                Instruction* insertBeforeBranch = corelib::CoreIRBuilder{newBlock}.CreateBr( successorBB );

                // Generate the load/store
                generateSingleGetOrSet( kind, UberP, statePtr, allocaptr, type, toStore, insertBeforeBranch );
            }
        }

        if( !isPut )
        {
            irb.SetInsertPoint( insertBefore );
            if( pointeeType == UberPointer::PointeeType::Payload || pointeeType == UberPointer::PointeeType::Buffer
                || pointeeType == UberPointer::PointeeType::BufferID || pointeeType == UberPointer::PointeeType::Attribute
                || pointeeType == UberPointer::PointeeType::Variable )
            {
                auto      tmp     = irb.CreateBitCast( allocaptr, type->getPointerTo( 0 ) );
                LoadInst* newLoad = irb.CreateLoad( tmp );
                CI->replaceAllUsesWith( newLoad );
            }
            else
            {
                Value* UberP_offset = UberPointer::get<UberPointer::Offset>( UberP, "UberP.offset", insertBefore );
                Value* loadPtr = addOffset( m_llvmManager->llvmDataLayout(), allocaptr, "", type->getPointerTo( 0 ),
                                            UberP_offset, insertBefore );
                LoadInst* newLoad = irb.CreateLoad( loadPtr );
                CI->replaceAllUsesWith( newLoad );
            }
        }

        toDelete.push_back( CI );
    }

    // Delete the call instructions
    removeValues( toDelete );
}

void UberPointerSet::finalizeUberPointerGetsToAlloca( Function* fn )
{
    Module*             module      = fn->getParent();
    LLVMContext&        llvmContext = module->getContext();
    std::vector<Value*> toDelete;

    // Cache all the calls to the accessor function so not to risk to ilwalidate the iterator while modifying the code.
    for( CallInst* CI : getCallsToFunction( fn ) )
    {
        // Argument to _UberPointerMemAccess is the UberPointer
        Value* UberP = CI->getOperand( 1 );

        // Compute which getters (or setters) are necessary for this uberpointer reference
        int             n = m_uptrInfo.size();
        llvm::BitVector validKinds( n, false );
        findValidUberPointerKinds( validKinds, UberP );

        // Determine the size of the alloca as the largest of the possible types
        size_t     maxSize = 0;
        DataLayout DL( module );
        for( int kind = validKinds.find_first(); kind != -1; kind = validKinds.find_next( kind ) )
        {
            RT_ASSERT( kind >= 0 && kind < static_cast<int>( m_uptrList.size() ) );
            Info& info      = m_uptrList[kind];
            Type* valueType = info.getter->getReturnType();

            if( valueType->isVoidTy() )
                continue;

            size_t size = DL.getTypeStoreSize( valueType );
            if( size > maxSize )
                maxSize = size;
        }
        RT_ASSERT( maxSize != 0 );
        int validKindsNumber = validKinds.count();

        Instruction*           insertBefore  = CI;
        Function*              usingFunction = insertBefore->getParent()->getParent();
        corelib::CoreIRBuilder irb{corelib::getFirstNonAlloca( usingFunction )};
        AllocaInst*            allocaptr = irb.CreateAlloca( irb.getInt8Ty(), irb.getInt32( maxSize ) );
        allocaptr->setAlignment( 16 );

        Value* statePtr = CI->getOperand( 0 );
        if( validKindsNumber == 1 )
        {
            // There is only a single reference associated with this get/put
            // so generate it directly.  Technically the optimizer will
            // figure this out, but save it some work in the common case.
            generateSingleGetOrSet( validKinds.find_first(), UberP, statePtr, allocaptr, nullptr, nullptr, insertBefore );
        }
        else
        {
            // There are multiple getters applicable to this reference.
            // Dynamically switch between them.
            checkAliasingRules( validKinds );

            // Insert the discriminant and split the block in preparation
            // for adding the switch.
            Value*      selector    = UberPointer::get<UberPointer::UPKind>( UberP, "uberptr.selector", insertBefore );
            BasicBlock* lwrrentBB   = insertBefore->getParent();
            BasicBlock* successorBB = lwrrentBB->splitBasicBlock( insertBefore );

            // Split basic block inserts an unconditional branch to the old code.  Remove it.
            Instruction* branch = lwrrentBB->getTerminator();
            RT_ASSERT( branch && isa<BranchInst>( branch ) );
            branch->eraseFromParent();

            // Switch requires a default case.  Make a basic block and mark it as unreachable.
            BasicBlock* trapBlock = BasicBlock::Create( llvmContext, "uberptr.illegal", usingFunction, successorBB );
            corelib::CoreIRBuilder{trapBlock}.CreateUnreachable();

            // Insert the switch, which will become the new terminator instruction
            BasicBlock* insertBB = trapBlock;  // Insert other blocks before here
            SwitchInst* sin = corelib::CoreIRBuilder{lwrrentBB}.CreateSwitch( selector, trapBlock, validKindsNumber );

            // Generate load/stores for the valid possibilities
            for( int kind = validKinds.find_first(); kind != -1; kind = validKinds.find_next( kind ) )
            {
                RT_ASSERT( kind >= 0 && kind < static_cast<int>( m_uptrList.size() ) );
                Info& info = m_uptrList[kind];

                // Create a new block for this case
                BasicBlock* newBlock = BasicBlock::Create( llvmContext, "uberptr." + info.name, usingFunction, insertBB );
                ConstantInt* idxval = ConstantInt::get( Type::getInt32Ty( llvmContext ), kind );
                sin->addCase( idxval, newBlock );

                // Insert unconditional branch back to the original code
                Instruction* insertBeforeBranch = corelib::CoreIRBuilder{newBlock}.CreateBr( successorBB );

                // Generate the load/store
                generateSingleGetOrSet( kind, UberP, statePtr, allocaptr, nullptr, nullptr, insertBeforeBranch );
            }
        }

        CI->replaceAllUsesWith( allocaptr );
        allocaptr->takeName( CI );
        toDelete.push_back( CI );
    }

    // Delete the call instructions
    removeValues( toDelete );
}

void UberPointerSet::finalizeUberPointerGetAddress( Function* fn )
{
    Module*             module      = fn->getParent();
    LLVMContext&        llvmContext = module->getContext();
    std::vector<Value*> toDelete;

    for( CallInst* CI : getCallsToFunction( fn ) )
    {
        // Argument to _UberPointerMemAccess is the UberPointer
        Value* UberP = CI->getOperand( 1 );

        // Compute which getters (or setters) are necessary for this uberpointer reference
        llvm::BitVector validKinds( m_uptrInfo.size(), false );
        findValidUberPointerKinds( validKinds, UberP );
        int validKindsNumber = validKinds.count();

        Instruction*           insertBefore  = CI;
        Function*              usingFunction = insertBefore->getParent()->getParent();
        corelib::CoreIRBuilder irb{corelib::getFirstNonAlloca( usingFunction )};
        AllocaInst*            allocaptr = irb.CreateAlloca( irb.getInt64Ty(), irb.getInt32( 1 ) );
        allocaptr->setAlignment( 16 );

        Value* statePtr = CI->getOperand( 0 );
        if( validKindsNumber == 1 )
        {
            Value* ptr = generateSingleGetAddr( validKinds.find_first(), UberP, statePtr, insertBefore );

            CI->replaceAllUsesWith( ptr );
            CI->eraseFromParent();
            continue;
        }
        else
        {
            // There are multiple getters applicable to this reference.
            // Dynamically switch between them.
            checkAliasingRules( validKinds );

            // Insert the discriminant and split the block in preparation
            // for adding the switch.
            Value*      selector    = UberPointer::get<UberPointer::UPKind>( UberP, "uberptr.selector", insertBefore );
            BasicBlock* lwrrentBB   = insertBefore->getParent();
            BasicBlock* successorBB = lwrrentBB->splitBasicBlock( insertBefore );

            // Split basic block inserts an unconditional branch to the old code.  Remove it.
            Instruction* branch = lwrrentBB->getTerminator();
            RT_ASSERT( branch && isa<BranchInst>( branch ) );
            branch->eraseFromParent();

            // Switch requires a default case.  Make a basic block and mark it as unreachable.
            BasicBlock* trapBlock = BasicBlock::Create( llvmContext, "uberptr.illegal", usingFunction, successorBB );
            corelib::CoreIRBuilder{trapBlock}.CreateUnreachable();

            // Insert the switch, which will become the new terminator instruction
            BasicBlock* insertBB = trapBlock;  // Insert other blocks before here
            SwitchInst* sin = corelib::CoreIRBuilder{lwrrentBB}.CreateSwitch( selector, trapBlock, validKindsNumber );

            // Generate load/stores for the valid possibilities
            for( int kind = validKinds.find_first(); kind != -1; kind = validKinds.find_next( kind ) )
            {
                RT_ASSERT( kind >= 0 && kind < static_cast<int>( m_uptrList.size() ) );
                Info& info = m_uptrList[kind];

                // Create a new block for this case
                BasicBlock* newBlock = BasicBlock::Create( llvmContext, "uberptr." + info.name, usingFunction, insertBB );
                ConstantInt* idxval = ConstantInt::get( Type::getInt32Ty( llvmContext ), kind );
                sin->addCase( idxval, newBlock );

                // Insert unconditional branch back to the original code
                Instruction* insertBeforeBranch = corelib::CoreIRBuilder{newBlock}.CreateBr( successorBB );

                // Generate the load/store
                Value* ptr = generateSingleGetAddr( kind, UberP, statePtr, insertBeforeBranch );

                corelib::CoreIRBuilder{insertBeforeBranch}.CreateStore( ptr, allocaptr );
            }

            Value* new_ptr = corelib::CoreIRBuilder{insertBefore}.CreateLoad( allocaptr );
            CI->replaceAllUsesWith( new_ptr );
            new_ptr->takeName( CI );
        }
        toDelete.push_back( CI );
    }

    // Delete the call instructions
    removeValues( toDelete );
}

// The 'kind' of a UberPointer is an identifier of the type of a UberPointer.
// It identifies a getter and a setter and all the accessor functions.
int UberPointerSet::getOrAddUberPointerKind( const std::string&       name,
                                             Function*                getter,
                                             Function*                setter,
                                             Function*                atomic,
                                             Function*                getAddress,
                                             UberPointer::PointeeType type,
                                             Value*                   defaultValue )
{
    // Create a key object, to be used in the lookup in the UberPointer map.
    // Pay attention that only the setter and getter names are used in the map lookup, the other fields of the key are ignored.
    Info                  key( name, type, getter, setter, atomic, getAddress, defaultValue );
    InfoMapType::iterator iter = m_uptrInfo.find( key );
    if( iter != m_uptrInfo.end() )
    {
        RT_ASSERT( iter->first.name == key.name );
        RT_ASSERT( iter->first.type == key.type );
        RT_ASSERT( iter->first.variableDefault == key.variableDefault );
        RT_ASSERT( iter->first.atomic == key.atomic );
        RT_ASSERT( iter->first.getAddress == key.getAddress );
        return iter->second;
    }

    int kind = m_uptrList.size();
    m_uptrList.push_back( key );
    m_uptrInfo.insert( std::make_pair( key, kind ) );
    return kind;
}

// UberPointerSet::Info Class.
// -----------------------------------------------------------------------------
UberPointerSet::Info::Info( const std::string&       name,
                            UberPointer::PointeeType type,
                            Function*                getter,
                            Function*                setter,
                            Function*                atomic,
                            Function*                getAddress,
                            Value*                   variableDefault )
    : name( name )
    , type( type )
    , getter( getter )
    , setter( setter )
    , atomic( atomic )
    , getAddress( getAddress )
    , variableDefault( variableDefault )
{
}

// Comparison function used to insert Info structs into the UberPointer map.
bool UberPointerSet::InfoCompare::operator()( const UberPointerSet::Info& a, const UberPointerSet::Info& b ) const
{
    // First compare the getter name. By construction the getter is non-NULL.
    if( a.getter->getName() < b.getter->getName() )
        return true;
    else if( a.getter->getName() > b.getter->getName() )
        return false;

    if( a.setter && !b.setter )
        return true;
    else if( !a.setter && b.setter )
        return false;

    if( a.setter && b.setter )
        return a.setter->getName() < b.setter->getName();

    return false;
}

// Support functions.
//------------------------------------------------------------------------------
bool isAtomicFunction( StringRef functionName )
{
    return functionName.startswith( "optix.ptx.atom" );
}

//------------------------------------------------------------------------------
// TODO: This operation should be replaced with an intrinsic lookup. At the moment this is very expensive.
// For this, take a look at how LLVM implements intrinsics lookup.
AtomicOpType getAtomicOperator( StringRef functionName )
{
    Regex optixAtomicRegex = Regex( "optix\\.ptx\\.atom\\.(.+)\\.(.+)$" );
    SmallVector<StringRef, 6> matches;
    if( !optixAtomicRegex.match( functionName, &matches ) )
        RT_ASSERT_FAIL_MSG( "Function name does not match expected optix atomic name" );
    if( matches.size() != 3 )
        RT_ASSERT_FAIL_MSG( "Function name does not match expected optix atomic name" );

    if( matches[1] == "add" )
        return AtomicOpType::ADD;
    if( matches[1] == "sub" )
        return AtomicOpType::SUB;
    if( matches[1] == "exch" )
        return AtomicOpType::EXCH;
    if( matches[1] == "min" )
        return AtomicOpType::MIN;
    if( matches[1] == "max" )
        return AtomicOpType::MAX;
    if( matches[1] == "inc" )
        return AtomicOpType::INC;
    if( matches[1] == "dec" )
        return AtomicOpType::DEC;
    if( matches[1] == "cas" )
        return AtomicOpType::CAS;
    if( matches[1] == "and" )
        return AtomicOpType::AND;
    if( matches[1] == "or" )
        return AtomicOpType::OR;
    if( matches[1] == "xor" )
        return AtomicOpType::XOR;

    RT_ASSERT_FAIL_MSG( "Unrecognized operator type" );
    return AtomicOpType::INVALID;
}

//------------------------------------------------------------------------------
void UberPointerSet::findValidUberPointerKinds( llvm::BitVector& validKinds, llvm::Value* UberP )
{
    if( Constant* C = dyn_cast<Constant>( UberP ) )
    {
        // Common simple case
        int upkind = getConstantValueOrAssert( UberPointer::get<UberPointer::UPKind>( C ) );
        RT_ASSERT( upkind >= 0 && upkind < static_cast<int>( validKinds.size() ) );
        validKinds.set( upkind );
    }
    else
    {
        SmallVector<Value*, 4> Worklist;
        SmallSet<Value*, 16>   Visited;

        Worklist.push_back( UberP );

        while( !Worklist.empty() )
        {
            Value* V = Worklist.pop_back_val();
            RT_ASSERT( V->getType() == UberP->getType() );

            // Gather all the UberP and process
            if( Constant* CV = dyn_cast<Constant>( V ) )
            {
                // Figure out what the size is
                int upkind = getConstantValueOrAssert( UberPointer::get<UberPointer::UPKind>( CV ) );
                RT_ASSERT( upkind >= 0 && upkind < static_cast<int>( validKinds.size() ) );
                validKinds.set( upkind );
            }
            else if( SelectInst* SI = dyn_cast<SelectInst>( V ) )
            {
                RT_ASSERT( SI->getCondition() != V );  // Can't use UberP as the condition
                if( std::get<1>( Visited.insert( SI->getTrueValue() ) ) )
                    Worklist.push_back( SI->getTrueValue() );
                if( std::get<1>( Visited.insert( SI->getFalseValue() ) ) )
                    Worklist.push_back( SI->getFalseValue() );
            }
            else if( PHINode* PN = dyn_cast<PHINode>( V ) )
            {
                unsigned int numIncoming = PN->getNumIncomingValues();
                for( unsigned int i = 0; i < numIncoming; ++i )
                {
                    Value* incoming = PN->getIncomingValue( i );
                    if( std::get<1>( Visited.insert( incoming ) ) )
                        Worklist.push_back( incoming );
                }
            }
            else if( InsertValueInst* IVI = dyn_cast<InsertValueInst>( V ) )
            {
                Value* agg = IVI->getAggregateOperand();
                if( std::get<1>( Visited.insert( agg ) ) )
                    Worklist.push_back( agg );
            }
            else if( CallInst* callInst = dyn_cast<CallInst>( V ) )
            {
                Function* callee = callInst->getCalledFunction();
                if( callee->getName() == "_castUberP" )
                {
                    Module* module = callee->getParent();
                    int     kind   = getOrInsertRawAccessEntry( module );

                    Value* rawPointer = callInst->getArgOperand( 0 );
                    Value* uberP      = UberPointer::getRawUberPointer( m_llvmManager, kind, rawPointer, callInst );
                    callInst->replaceAllUsesWith( uberP );
                    validKinds.set( kind );
                }
            }
            else
            {
                throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( V ),
                                    "Find valid UberPointer kinds - Unexpected instruction in analysis of variable "
                                    "usage" );
            }
        }
    }
    RT_ASSERT_MSG( validKinds.any(), "Did not find any valid uses of UberPointer" );
}

//------------------------------------------------------------------------------
std::string getPTXAtomicFunction( AtomicOpType op, Type* type )
{
    std::string name = "optix.ptx.atom.";
    std::string size = std::to_string( type->getScalarSizeInBits() );

    switch( op )
    {
        case AtomicOpType::ADD:
            if( type->isFloatTy() )
                name += "add.f32";
            else
                name += "add.s" + size;
            break;
        case AtomicOpType::SUB:
            name += "sub.s32";
            break;
        case AtomicOpType::EXCH:
            name += "exch.b" + size;
            break;
        case AtomicOpType::MIN:
            name += "min.s" + size;
            break;
        case AtomicOpType::MAX:
            name += "max.s" + size;
            break;
        case AtomicOpType::INC:
            name += "inc.u32";
            break;
        case AtomicOpType::DEC:
            name += "dec.u32";
            break;
        case AtomicOpType::CAS:
            name += "cas.b" + size;
            break;
        case AtomicOpType::AND:
            name += "and.b" + size;
            break;
        case AtomicOpType::OR:
            name += "or.b" + size;
            break;
        case AtomicOpType::XOR:
            name += "xor.b" + size;
            break;
        case AtomicOpType::INVALID:
            RT_ASSERT_FAIL_MSG( "Invalid atomic operation found" );
            break;
        default:
            RT_ASSERT_FAIL_MSG( "Unrecognized atomic operation" );
    }
    return name;
}
