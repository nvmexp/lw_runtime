// Copyright (c) 2019, LWPU CORPORATION.
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
#include <FrontEnd/Canonical/IntrinsicsManager.h>

#include <Context/LLVMManager.h>
#include <Context/ProgramManager.h>
#include <ExelwtionStrategy/Compile.h>
#include <FrontEnd/Canonical/IntrinsicsAssertions.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <Util/ContainerAlgorithm.h>

#include <corelib/compiler/CoreIRBuilder.h>
#include <corelib/compiler/LLVMUtil.h>
#include <prodlib/exceptions/Assert.h>

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>

#include <llvm/Support/Regex.h>

using namespace optix;
using namespace llvm;

// -----------------------------------------------------------------------------
unsigned short int optix::getOptixAtomicToken( const Function* function, const ProgramManager* programManager )
{
    const std::string        uniqueName = AtomicSetBufferElement::parseUniqueName( function->getName() );
    const VariableReference* varref     = programManager->getVariableReferenceByUniversallyUniqueName( uniqueName );
    return varref->getVariableToken();
}

// -----------------------------------------------------------------------------
bool optix::isOptixIntrinsic( const Function* function )
{
    return OptixIntrinsic::isIntrinsic( function );
}

// -----------------------------------------------------------------------------
bool optix::isOptixAtomicIntrinsic( const Function* function )
{
    return AtomicSetBufferElementFromId::isIntrinsic( function ) || AtomicSetBufferElement::isIntrinsic( function );
}

// -----------------------------------------------------------------------------
bool optix::isPayloadGet( const Function* function )
{
    return function->getName().startswith( GET_PAYLOAD_PREFIX );
}

// -----------------------------------------------------------------------------
bool optix::isPayloadSet( const Function* function )
{
    return function->getName().startswith( SET_PAYLOAD_PREFIX );
}

// -----------------------------------------------------------------------------
const std::string OptixIntrinsic::PREFIX{"optixi_"};

bool OptixIntrinsic::classof( const CallInst* inst )
{
    if( const Function* calledFunction = inst->getCalledFunction() )
        return isIntrinsic( calledFunction );
    return false;
}

bool OptixIntrinsic::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool OptixIntrinsic::isIntrinsic( const llvm::Function* function )
{
    return function->getName().startswith( PREFIX );
}


// -----------------------------------------------------------------------------
const std::string AtomicSetBufferElement::PREFIX{"optixi_atomicSetBufferElement"};
Regex             AtomicSetBufferElement::nameRegex{PREFIX + "\\.(" + optixi_VariableReferenceUniqueNameRegex + ")$"};

bool AtomicSetBufferElement::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    if( calledFunction == nullptr )
        return false;

    // Check if the name matches the expected one.
    SmallVector<StringRef, 2> matches;
    StringRef functionName = calledFunction->getName();
    if( !nameRegex.match( functionName, &matches ) )
        return false;

    return matches.size() == 2;
}

bool AtomicSetBufferElement::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool AtomicSetBufferElement::isIntrinsic( const Function* function )
{
    return nameRegex.match( function->getName() );
}

unsigned AtomicSetBufferElement::getDimensionality( const Function* function )
{
    unsigned callArgsNumber = function->arg_size();
    unsigned dimensions     = callArgsNumber - MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT( 1 <= dimensions && dimensions <= 3 );
    return dimensions;
}

std::string AtomicSetBufferElement::parseUniqueName() const
{
    return parseUniqueName( getCalledFunction()->getName() );
}

std::string AtomicSetBufferElement::parseUniqueName( StringRef name )
{
    SmallVector<StringRef, 2> matches;
    RT_ASSERT( nameRegex.match( name, &matches ) );
    RT_ASSERT( matches.size() == 2 );
    return matches[1];
}

std::string AtomicSetBufferElement::createUniqueName( const VariableReference* vref )
{
    return PREFIX + '.' + vref->getUniversallyUniqueName();
}

unsigned AtomicSetBufferElement::getDimensionality() const
{
    return getDimensionality( getCalledFunction() );
}

Value* AtomicSetBufferElement::getStatePtr() const
{
    return getArgOperand( Args::CanonicalState );
}

Value* AtomicSetBufferElement::getOperation()
{
    return getArgOperand( Args::Operation );
}

Value* AtomicSetBufferElement::getCompareOperand()
{
    return getArgOperand( Args::CompareOperand );
}

Value* AtomicSetBufferElement::getOperand()
{
    return getArgOperand( Args::Operand );
}

Value* AtomicSetBufferElement::getElementSize() const
{
    return getArgOperand( Args::ElementSize );
}

Value* AtomicSetBufferElement::getOffset() const
{
    return getArgOperand( Args::Offset );
}

Value* AtomicSetBufferElement::getX() const
{
    return getArgOperand( Args::x );
}

Value* AtomicSetBufferElement::getY() const
{
    return getArgOperand( Args::y );
}

Value* AtomicSetBufferElement::getZ() const
{
    return getArgOperand( Args::z );
}

Value* AtomicSetBufferElement::getIndex( unsigned dimension ) const
{
    return getArgOperand( Args::x + dimension );
}

Value* AtomicSetBufferElement::getSubElementType()
{
    return getArgOperand( Args::SubElementType );
}

// -----------------------------------------------------------------------------
AtomicSetBufferElementBuilder::AtomicSetBufferElementBuilder( Module* module )
    : m_module( module )
{
}

unsigned AtomicSetBufferElementBuilder::getDimensionality( const Function* function )
{
    return AtomicSetBufferElement::getDimensionality( function );
}

FunctionType* AtomicSetBufferElementBuilder::createType( Type* opType, unsigned dimensions, LLVMManager* llvmManager )
{
    std::vector<Type*> argsType( AtomicSetBufferElement::Args::END );

    // Mandatory arguments.
    argsType[AtomicSetBufferElement::Args::CanonicalState] = llvmManager->getStatePtrType();
    argsType[AtomicSetBufferElement::Args::Operation]      = llvmManager->getI32Type();
    argsType[AtomicSetBufferElement::Args::CompareOperand] = opType;
    argsType[AtomicSetBufferElement::Args::Operand]        = opType;
    argsType[AtomicSetBufferElement::Args::SubElementType] = llvmManager->getI8Type();
    argsType[AtomicSetBufferElement::Args::ElementSize]    = llvmManager->getI32Type();
    argsType[AtomicSetBufferElement::Args::Offset]         = llvmManager->getSizeTType();
    argsType[AtomicSetBufferElement::Args::x]              = llvmManager->getI64Type();

    // Optional arguments.
    if( dimensions > 1 )
        argsType[AtomicSetBufferElement::Args::y] = llvmManager->getI64Type();
    if( dimensions > 2 )
        argsType[AtomicSetBufferElement::Args::z] = llvmManager->getI64Type();

    argsType.resize( AtomicSetBufferElement::MANDATORY_ARGS_NUMBER + dimensions - 1 );

    RT_ASSERT_ARG_TYPES_NON_NULL( argsType, "an atomic setter." );

    FunctionType* functionType = FunctionType::get( opType, argsType, false );

    return functionType;
}

Constant* AtomicSetBufferElementBuilder::createFunction( Type* opType, VariableReference* varRef, unsigned dimensions, LLVMManager* llvmManager )
{
    auto        functionType = createType( opType, dimensions, llvmManager );
    std::string name         = AtomicSetBufferElement::createUniqueName( varRef );
    return m_module->getOrInsertFunction( name, functionType );
}

CallInst* AtomicSetBufferElementBuilder::create( Function* function, Instruction* insertBefore )
{
    unsigned numParams    = function->getFunctionType()->getNumParams();
    unsigned expectedDims = numParams - AtomicSetBufferElement::MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT_MSG( expectedDims >= 1 && expectedDims <= 3, "Malformed function " + function->getName().str() );

    std::vector<Value*> args( AtomicSetBufferElement::Args::END );

    // Mandatory arguments.
    args[AtomicSetBufferElement::Args::CanonicalState] = m_canonicalState;
    args[AtomicSetBufferElement::Args::Operation]      = m_operation;
    args[AtomicSetBufferElement::Args::CompareOperand] = m_compareOperand;
    args[AtomicSetBufferElement::Args::Operand]        = m_operand;
    args[AtomicSetBufferElement::Args::ElementSize]    = m_elementSize;
    args[AtomicSetBufferElement::Args::SubElementType] = m_subElementType;
    args[AtomicSetBufferElement::Args::Offset]         = m_offset;
    args[AtomicSetBufferElement::Args::x]              = m_x;

    unsigned dimensions = 1;

    // Optional arguments.
    if( m_y != nullptr )
    {
        args[AtomicSetBufferElement::Args::y] = m_y;
        dimensions                            = 2;
    }

    if( m_z != nullptr )
    {
        args[AtomicSetBufferElement::Args::z] = m_z;
        dimensions                            = 3;
    }

    RT_ASSERT( dimensions == expectedDims );
    args.resize( AtomicSetBufferElement::MANDATORY_ARGS_NUMBER + dimensions - 1 );

    RT_ASSERT_ARGS_NON_NULL( args, "creating an atomic settter." );

    corelib::CoreIRBuilder irb{insertBefore};
    return irb.CreateCall( function, args );
}

AtomicSetBufferElementBuilder& AtomicSetBufferElementBuilder::setCanonicalState( Value* canonicalState )
{
    m_canonicalState = canonicalState;
    return *this;
}

AtomicSetBufferElementBuilder& AtomicSetBufferElementBuilder::setOperation( Value* operation )
{
    m_operation = operation;
    return *this;
}
AtomicSetBufferElementBuilder& AtomicSetBufferElementBuilder::setCompareOperand( Value* compareOperand )
{
    m_compareOperand = compareOperand;
    return *this;
}
AtomicSetBufferElementBuilder& AtomicSetBufferElementBuilder::setOperand( Value* operand )
{
    m_operand = operand;
    return *this;
}
AtomicSetBufferElementBuilder& AtomicSetBufferElementBuilder::setElementSize( Value* elementSize )
{
    m_elementSize = elementSize;
    return *this;
}
AtomicSetBufferElementBuilder& AtomicSetBufferElementBuilder::setSubElementType( Value* subElementType )
{
    m_subElementType = subElementType;
    return *this;
}
AtomicSetBufferElementBuilder& AtomicSetBufferElementBuilder::setOffset( Value* offset )
{
    m_offset = offset;
    return *this;
}
AtomicSetBufferElementBuilder& AtomicSetBufferElementBuilder::setX( Value* x )
{
    m_x = x;
    return *this;
}
AtomicSetBufferElementBuilder& AtomicSetBufferElementBuilder::setY( Value* y )
{
    m_y = y;
    return *this;
}
AtomicSetBufferElementBuilder& AtomicSetBufferElementBuilder::setZ( Value* z )
{
    m_z = z;
    return *this;
}

// -----------------------------------------------------------------------------
const std::string AtomicSetBufferElementFromId::PREFIX{"optixi_atomicSetBufferElementFromId"};
Regex AtomicSetBufferElementFromId::nameRegex{PREFIX + "\\." + optixi_VariableReferenceUniqueNameRegex + "$"};

bool AtomicSetBufferElementFromId::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    if( calledFunction == nullptr )
        return false;

    // Check if the name matches the expected one.
    return calledFunction->getName().startswith( PREFIX );
}

bool AtomicSetBufferElementFromId::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool AtomicSetBufferElementFromId::isIntrinsic( const Function* function )
{
    return function->getName().startswith( PREFIX );
}

std::string AtomicSetBufferElementFromId::createUniqueName( unsigned int dimensions, size_t elementSize )
{
    return PREFIX + std::to_string( dimensions ) + '.' + std::to_string( elementSize );
}

unsigned AtomicSetBufferElementFromId::getDimensionality( const Function* function )
{
    unsigned callArgsNumber = function->arg_size();
    unsigned dimensions     = callArgsNumber - MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT( 1 <= dimensions && dimensions <= 3 );
    return dimensions;
}

unsigned AtomicSetBufferElementFromId::getDimensionality()
{
    return getDimensionality( getCalledFunction() );
}

Value* AtomicSetBufferElementFromId::getStatePtr()
{
    return getArgOperand( Args::CanonicalState );
}

Value* AtomicSetBufferElementFromId::getBufferId()
{
    return getArgOperand( Args::BufferId );
}

Value* AtomicSetBufferElementFromId::getOperation()
{
    return getArgOperand( Args::Operation );
}

Value* AtomicSetBufferElementFromId::getCompareOperand()
{
    return getArgOperand( Args::CompareOperand );
}

Value* AtomicSetBufferElementFromId::getOperand()
{
    return getArgOperand( Args::Operand );
}

Value* AtomicSetBufferElementFromId::getElementSize()
{
    return getArgOperand( Args::ElementSize );
}

Value* AtomicSetBufferElementFromId::getOffset()
{
    return getArgOperand( Args::Offset );
}

Value* AtomicSetBufferElementFromId::getX()
{
    return getArgOperand( Args::x );
}

Value* AtomicSetBufferElementFromId::getY()
{
    return getArgOperand( Args::y );
}

Value* AtomicSetBufferElementFromId::getZ()
{
    return getArgOperand( Args::z );
}

Value* AtomicSetBufferElementFromId::getIndex( unsigned dimension )
{
    return getArgOperand( Args::x + dimension );
}

Value* AtomicSetBufferElementFromId::getSubElementType()
{
    return getArgOperand( Args::SubElementType );
}

// -----------------------------------------------------------------------------
AtomicSetBufferElementFromIdBuilder::AtomicSetBufferElementFromIdBuilder( Module* module )
    : m_module( module )
{
}

unsigned AtomicSetBufferElementFromIdBuilder::getDimensionality( const Function* function )
{
    return AtomicSetBufferElementFromId::getDimensionality( function );
}

FunctionType* AtomicSetBufferElementFromIdBuilder::createType( Type* opType, unsigned dimensions, LLVMManager* llvmManager )
{
    std::vector<Type*> argsType( AtomicSetBufferElementFromId::Args::END );

    // Mandatory arguments.
    argsType[AtomicSetBufferElementFromId::Args::CanonicalState] = llvmManager->getStatePtrType();
    argsType[AtomicSetBufferElementFromId::Args::BufferId]       = llvmManager->getI32Type();
    argsType[AtomicSetBufferElementFromId::Args::Operation]      = llvmManager->getI32Type();
    argsType[AtomicSetBufferElementFromId::Args::SubElementType] = llvmManager->getI8Type();
    argsType[AtomicSetBufferElementFromId::Args::CompareOperand] = opType;
    argsType[AtomicSetBufferElementFromId::Args::Operand]        = opType;
    argsType[AtomicSetBufferElementFromId::Args::ElementSize]    = llvmManager->getI32Type();
    argsType[AtomicSetBufferElementFromId::Args::Offset]         = llvmManager->getSizeTType();
    argsType[AtomicSetBufferElementFromId::Args::x]              = llvmManager->getI64Type();

    // Optional arguments.
    if( dimensions > 1 )
        argsType[AtomicSetBufferElementFromId::Args::y] = llvmManager->getI64Type();
    if( dimensions > 2 )
        argsType[AtomicSetBufferElementFromId::Args::z] = llvmManager->getI64Type();

    argsType.resize( AtomicSetBufferElementFromId::MANDATORY_ARGS_NUMBER + dimensions - 1 );

    RT_ASSERT_ARG_TYPES_NON_NULL( argsType, "an atomic setter from id." );

    FunctionType* functionType = FunctionType::get( opType, argsType, false );

    return functionType;
}

Constant* AtomicSetBufferElementFromIdBuilder::createFunction( Type* opType, size_t elementSize, unsigned dimensions, LLVMManager* llvmManager )
{
    auto functionType = createType( opType, dimensions, llvmManager );
    return m_module->getOrInsertFunction( AtomicSetBufferElementFromId::createUniqueName( dimensions, elementSize ), functionType );
}

CallInst* AtomicSetBufferElementFromIdBuilder::create( Function* function, Instruction* insertBefore )
{
    unsigned numParams    = function->getFunctionType()->getNumParams();
    unsigned expectedDims = numParams - AtomicSetBufferElementFromId::MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT_MSG( expectedDims >= 1 && expectedDims <= 3, "Malformed function " + function->getName().str() );

    std::vector<Value*> args( AtomicSetBufferElementFromId::Args::END );

    // Mandatory arguments.
    args[AtomicSetBufferElementFromId::Args::CanonicalState] = m_canonicalState;
    args[AtomicSetBufferElementFromId::Args::BufferId]       = m_bufferId;
    args[AtomicSetBufferElementFromId::Args::Operation]      = m_operation;
    args[AtomicSetBufferElementFromId::Args::CompareOperand] = m_compareOperand;
    args[AtomicSetBufferElementFromId::Args::Operand]        = m_operand;
    args[AtomicSetBufferElementFromId::Args::SubElementType] = m_subElementType;
    args[AtomicSetBufferElementFromId::Args::ElementSize]    = m_elementSize;
    args[AtomicSetBufferElementFromId::Args::Offset]         = m_offset;
    args[AtomicSetBufferElementFromId::Args::x]              = m_x;

    unsigned dimensions = 1;

    // Optional arguments.
    if( m_y != nullptr )
    {
        args[AtomicSetBufferElementFromId::Args::y] = m_y;
        dimensions                                  = 2;
    }

    if( m_z != nullptr )
    {
        args[AtomicSetBufferElementFromId::Args::z] = m_z;
        dimensions                                  = 3;
    }

    RT_ASSERT( dimensions == expectedDims );
    args.resize( AtomicSetBufferElementFromId::MANDATORY_ARGS_NUMBER + dimensions - 1 );

    RT_ASSERT_ARGS_NON_NULL( args, "an atomic setter from id." );

    corelib::CoreIRBuilder irb{insertBefore};
    return irb.CreateCall( function, args );
}

AtomicSetBufferElementFromIdBuilder& AtomicSetBufferElementFromIdBuilder::setCanonicalState( Value* canonicalState )
{
    m_canonicalState = canonicalState;
    return *this;
}

AtomicSetBufferElementFromIdBuilder& AtomicSetBufferElementFromIdBuilder::setBufferId( Value* bufferId )
{
    m_bufferId = bufferId;
    return *this;
}

AtomicSetBufferElementFromIdBuilder& AtomicSetBufferElementFromIdBuilder::setOperation( Value* operation )
{
    m_operation = operation;
    return *this;
}
AtomicSetBufferElementFromIdBuilder& AtomicSetBufferElementFromIdBuilder::setCompareOperand( Value* compareOperand )
{
    m_compareOperand = compareOperand;
    return *this;
}
AtomicSetBufferElementFromIdBuilder& AtomicSetBufferElementFromIdBuilder::setOperand( Value* operand )
{
    m_operand = operand;
    return *this;
}
AtomicSetBufferElementFromIdBuilder& AtomicSetBufferElementFromIdBuilder::setElementSize( Value* elementSize )
{
    m_elementSize = elementSize;
    return *this;
}
AtomicSetBufferElementFromIdBuilder& AtomicSetBufferElementFromIdBuilder::setSubElementType( Value* subElementType )
{
    m_subElementType = subElementType;
    return *this;
}
AtomicSetBufferElementFromIdBuilder& AtomicSetBufferElementFromIdBuilder::setOffset( Value* offset )
{
    m_offset = offset;
    return *this;
}
AtomicSetBufferElementFromIdBuilder& AtomicSetBufferElementFromIdBuilder::setX( Value* x )
{
    m_x = x;
    return *this;
}
AtomicSetBufferElementFromIdBuilder& AtomicSetBufferElementFromIdBuilder::setY( Value* y )
{
    m_y = y;
    return *this;
}
AtomicSetBufferElementFromIdBuilder& AtomicSetBufferElementFromIdBuilder::setZ( Value* z )
{
    m_z = z;
    return *this;
}

// -----------------------------------------------------------------------------
Regex TraceGlobalPayloadCall::nameRegex =
    Regex( "optixi_trace_global_payload\\." + optixi_CanonicalProgramUniqueNameRegex + "\\.prd([0-9]+)b$" );

bool TraceGlobalPayloadCall::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    if( calledFunction == nullptr )
        return false;

    SmallVector<llvm::StringRef, 2> matches;
    return matchName( calledFunction->getName(), matches );
}

bool TraceGlobalPayloadCall::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool TraceGlobalPayloadCall::isIntrinsic( const Function* function )
{
    SmallVector<llvm::StringRef, 2> matches;
    return matchName( function->getName(), matches );
}

int TraceGlobalPayloadCall::getPayloadSize( const Function* function )
{
    SmallVector<llvm::StringRef, 2> matches;
    RT_ASSERT( matchName( function->getName(), matches ) );
    int payloadSize = 0;
    matches[1].getAsInteger( 10, payloadSize );
    return payloadSize;
}

bool TraceGlobalPayloadCall::matchName( StringRef name, SmallVector<llvm::StringRef, 2>& matches )
{
    if( !nameRegex.match( name, &matches ) )
        return false;
    return matches.size() == 2;
}

Value* TraceGlobalPayloadCall::getStatePtr() const
{
    return getArgOperand( Args::CanonicalState );
}
Value* TraceGlobalPayloadCall::getNode() const
{
    return getArgOperand( Args::Node );
}
Value* TraceGlobalPayloadCall::getOx() const
{
    return getArgOperand( Args::Ox );
}
Value* TraceGlobalPayloadCall::getOy() const
{
    return getArgOperand( Args::Oy );
}
Value* TraceGlobalPayloadCall::getOz() const
{
    return getArgOperand( Args::Oz );
}
Value* TraceGlobalPayloadCall::getDx() const
{
    return getArgOperand( Args::Dx );
}
Value* TraceGlobalPayloadCall::getDy() const
{
    return getArgOperand( Args::Dy );
}
Value* TraceGlobalPayloadCall::getDz() const
{
    return getArgOperand( Args::Dz );
}
Value* TraceGlobalPayloadCall::getRayType() const
{
    return getArgOperand( Args::RayType );
}
Value* TraceGlobalPayloadCall::getTMin() const
{
    return getArgOperand( Args::Tmin );
}
Value* TraceGlobalPayloadCall::getTMax() const
{
    return getArgOperand( Args::Tmax );
}
Value* TraceGlobalPayloadCall::getTime() const
{
    return getArgOperand( Args::Time );
}
Value* TraceGlobalPayloadCall::getHasTime() const
{
    return getArgOperand( Args::HasTime );
}
Value* TraceGlobalPayloadCall::getRayMask() const
{
    return getArgOperand( Args::RayMask );
}
Value* TraceGlobalPayloadCall::getRayFlags() const
{
    return getArgOperand( Args::RayFlags );
}
Value* TraceGlobalPayloadCall::getElementSize() const
{
    return getArgOperand( Args::ElementSize );
}

// -----------------------------------------------------------------------------

llvm::Type* TraceGlobalPayloadBuilder::getRayFlagsType( llvm::LLVMContext& ctx )
{
    return Type::getIntNTy( ctx, 8 );
}

llvm::Value* TraceGlobalPayloadBuilder::getDefaultRayFlags( llvm::LLVMContext& ctx )
{
    return ConstantInt::getNullValue( getRayFlagsType( ctx ) );
}

llvm::Type* TraceGlobalPayloadBuilder::getRayMaskType( llvm::LLVMContext& ctx )
{
    return Type::getIntNTy( ctx, 8 );
}

llvm::Value* TraceGlobalPayloadBuilder::getDefaultRayMask( llvm::LLVMContext& ctx )
{
    return ConstantInt::getAllOnesValue( getRayMaskType( ctx ) );
}

TraceGlobalPayloadBuilder::TraceGlobalPayloadBuilder( Module* module )
    : m_module( module )
{
}

FunctionType* TraceGlobalPayloadBuilder::createType( LLVMManager* llvmManager )
{
    std::vector<Type*> argsType( TraceGlobalPayloadCall::Args::END );
    auto&              ctx = llvmManager->llvmContext();

    // Mandatory arguments.
    argsType[TraceGlobalPayloadCall::Args::CanonicalState] = llvmManager->getStatePtrType();
    argsType[TraceGlobalPayloadCall::Args::Node]           = llvmManager->getI32Type();
    argsType[TraceGlobalPayloadCall::Args::Ox]             = llvmManager->getFloatType();
    argsType[TraceGlobalPayloadCall::Args::Oy]             = llvmManager->getFloatType();
    argsType[TraceGlobalPayloadCall::Args::Oz]             = llvmManager->getFloatType();
    argsType[TraceGlobalPayloadCall::Args::Dx]             = llvmManager->getFloatType();
    argsType[TraceGlobalPayloadCall::Args::Dy]             = llvmManager->getFloatType();
    argsType[TraceGlobalPayloadCall::Args::Dz]             = llvmManager->getFloatType();
    argsType[TraceGlobalPayloadCall::Args::RayType]        = llvmManager->getI32Type();
    argsType[TraceGlobalPayloadCall::Args::Tmin]           = llvmManager->getFloatType();
    argsType[TraceGlobalPayloadCall::Args::Tmax]           = llvmManager->getFloatType();
    argsType[TraceGlobalPayloadCall::Args::Time]           = llvmManager->getFloatType();
    argsType[TraceGlobalPayloadCall::Args::HasTime]        = llvmManager->getI32Type();
    argsType[TraceGlobalPayloadCall::Args::RayMask]        = getRayMaskType( ctx );
    argsType[TraceGlobalPayloadCall::Args::RayFlags]       = getRayFlagsType( ctx );
    argsType[TraceGlobalPayloadCall::Args::ElementSize]    = llvmManager->getI32Type();

    RT_ASSERT_ARG_TYPES_NON_NULL( argsType, "a trace with global payload from id." );

    FunctionType* functionType = FunctionType::get( llvmManager->getVoidType(), argsType, false );

    return functionType;
}

Function* TraceGlobalPayloadBuilder::createFunction( const std::string& name, LLVMManager* llvmManager )
{
    auto functionType = createType( llvmManager );
    return dyn_cast<Function>( m_module->getOrInsertFunction( name, functionType ) );
}

CallInst* TraceGlobalPayloadBuilder::create( const std::string& cpUUName, int elementSize, LLVMManager* llvmManager, Instruction* insertBefore )
{
    std::string name     = "optixi_trace_global_payload." + cpUUName + ".prd" + std::to_string( elementSize ) + "b";
    auto        function = createFunction( name, llvmManager );
    function->addFnAttr( Attribute::AlwaysInline );

    std::vector<Value*> args( TraceGlobalPayloadCall::Args::END );

    args[TraceGlobalPayloadCall::Args::CanonicalState] = m_canonicalState;
    args[TraceGlobalPayloadCall::Args::Node]           = m_node;
    args[TraceGlobalPayloadCall::Args::Ox]             = m_ox;
    args[TraceGlobalPayloadCall::Args::Oy]             = m_oy;
    args[TraceGlobalPayloadCall::Args::Oz]             = m_oz;
    args[TraceGlobalPayloadCall::Args::Dx]             = m_dx;
    args[TraceGlobalPayloadCall::Args::Dy]             = m_dy;
    args[TraceGlobalPayloadCall::Args::Dz]             = m_dz;
    args[TraceGlobalPayloadCall::Args::RayType]        = m_rayType;
    args[TraceGlobalPayloadCall::Args::Tmin]           = m_tMin;
    args[TraceGlobalPayloadCall::Args::Tmax]           = m_tMax;
    args[TraceGlobalPayloadCall::Args::Time]           = m_time;
    args[TraceGlobalPayloadCall::Args::HasTime]        = m_hasTime;
    args[TraceGlobalPayloadCall::Args::RayMask]        = m_rayMask;
    args[TraceGlobalPayloadCall::Args::RayFlags]       = m_rayFlags;
    args[TraceGlobalPayloadCall::Args::ElementSize]    = m_elementSize;

    RT_ASSERT_ARGS_NON_NULL( args, "a trace with global payload from id." );

    corelib::CoreIRBuilder irb{insertBefore};
    return irb.CreateCall( function, args );
}

TraceGlobalPayloadBuilder& TraceGlobalPayloadBuilder::setCanonicalState( Value* canonicalState )
{
    m_canonicalState = canonicalState;
    return *this;
}

TraceGlobalPayloadBuilder& TraceGlobalPayloadBuilder::setNode( Value* node )
{
    m_node = node;
    return *this;
}

TraceGlobalPayloadBuilder& TraceGlobalPayloadBuilder::setOx( Value* ox )
{
    m_ox = ox;
    return *this;
}
TraceGlobalPayloadBuilder& TraceGlobalPayloadBuilder::setOy( Value* oy )
{
    m_oy = oy;
    return *this;
}
TraceGlobalPayloadBuilder& TraceGlobalPayloadBuilder::setOz( Value* oz )
{
    m_oz = oz;
    return *this;
}
TraceGlobalPayloadBuilder& TraceGlobalPayloadBuilder::setDx( Value* dx )
{
    m_dx = dx;
    return *this;
}
TraceGlobalPayloadBuilder& TraceGlobalPayloadBuilder::setDy( Value* dy )
{
    m_dy = dy;
    return *this;
}
TraceGlobalPayloadBuilder& TraceGlobalPayloadBuilder::setDz( Value* dz )
{
    m_dz = dz;
    return *this;
}
TraceGlobalPayloadBuilder& TraceGlobalPayloadBuilder::setRayType( Value* rayType )
{
    m_rayType = rayType;
    return *this;
}
TraceGlobalPayloadBuilder& TraceGlobalPayloadBuilder::setTMin( Value* tMin )
{
    m_tMin = tMin;
    return *this;
}
TraceGlobalPayloadBuilder& TraceGlobalPayloadBuilder::setTMax( Value* tMax )
{
    m_tMax = tMax;
    return *this;
}
TraceGlobalPayloadBuilder& TraceGlobalPayloadBuilder::setTime( Value* t )
{
    m_time = t;
    return *this;
}
TraceGlobalPayloadBuilder& TraceGlobalPayloadBuilder::setHasTime( Value* hasTime )
{
    m_hasTime = hasTime;
    return *this;
}
TraceGlobalPayloadBuilder& TraceGlobalPayloadBuilder::setRayMask( llvm::Value* v )
{
    m_rayMask = v;
    return *this;
}
TraceGlobalPayloadBuilder& TraceGlobalPayloadBuilder::setRayFlags( llvm::Value* v )
{
    m_rayFlags = v;
    return *this;
}
TraceGlobalPayloadBuilder& TraceGlobalPayloadBuilder::setElementSize( Value* elementSize )
{
    m_elementSize = elementSize;
    return *this;
}

// -----------------------------------------------------------------------------
const std::string GetBufferElementAddress::PREFIX{"optixi_getBufferElementAddress"};
Regex             GetBufferElementAddress::nameRegex{PREFIX + "\\.(" + optixi_VariableReferenceUniqueNameRegex + ")$"};

bool GetBufferElementAddress::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    if( calledFunction == nullptr )
        return false;

    return matchName( calledFunction->getName() );
}

bool GetBufferElementAddress::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool GetBufferElementAddress::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

bool GetBufferElementAddress::isIntrinsic( const Function* function, const std::string& varRefUniqueName )
{
    // MTA OP-1082
    // Avoid the string concatenation if the name doesn't start with PREFIX.
    // It'd be better to have a boolean m_isPrefix instead which gets initialized in the constructor.
    llvm::StringRef name = function->getName();
    if( name.startswith( PREFIX ) )
    {
        // Note that optixi_getBufferElementAddress.varRefUUName is not followed by the type,
        // so the startswith check would produce wrong results.
        return name.str() == PREFIX + "." + varRefUniqueName;
    }
    return false;
}

unsigned GetBufferElementAddress::getDimensionality( const Function* function )
{
    unsigned callArgsNumber = function->arg_size();
    unsigned dimensions     = callArgsNumber - MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT( 1 <= dimensions && dimensions <= 3 );
    return dimensions;
}

unsigned GetBufferElementAddress::getDimensionality() const
{
    return getDimensionality( getCalledFunction() );
}

std::string GetBufferElementAddress::parseUniqueName() const
{
    return parseUniqueName( getCalledFunction()->getName() );
}

std::string GetBufferElementAddress::parseUniqueName( StringRef name )
{
    SmallVector<StringRef, 2> matches;
    RT_ASSERT( nameRegex.match( name, &matches ) );
    RT_ASSERT( matches.size() == 2 );
    return matches[1];
}

std::string GetBufferElementAddress::createUniqueName( const VariableReference* varRef )
{
    return PREFIX + '.' + varRef->getUniversallyUniqueName();
}

bool GetBufferElementAddress::matchName( StringRef name )
{
    SmallVector<llvm::StringRef, 2> matches;
    if( !nameRegex.match( name, &matches ) )
        return false;
    return matches.size() == 2;
}

Value* GetBufferElementAddress::getStatePtr() const
{
    return getArgOperand( Args::CanonicalState );
}

Value* GetBufferElementAddress::getElementSize() const
{
    return getArgOperand( Args::ElementSize );
}

Value* GetBufferElementAddress::getX() const
{
    return getArgOperand( Args::x );
}

Value* GetBufferElementAddress::getY() const
{
    return getArgOperand( Args::y );
}

Value* GetBufferElementAddress::getZ() const
{
    return getArgOperand( Args::z );
}

Value* GetBufferElementAddress::getIndex( unsigned dimension ) const
{
    return getArgOperand( Args::x + dimension );
}

Value* GetBufferElementAddress::getOffset() const
{
    return ConstantInt::get( IntegerType::getInt64Ty( getCalledFunction()->getContext() ), 0 );
}

// -----------------------------------------------------------------------------
GetBufferElementAddressBuilder::GetBufferElementAddressBuilder( Module* module )
    : m_module( module )
{
}

FunctionType* GetBufferElementAddressBuilder::createType( unsigned dimensions, LLVMManager* llvmManager )
{
    std::vector<Type*> argsType( GetBufferElementAddress::Args::END );

    // Mandatory arguments.
    argsType[GetBufferElementAddress::Args::CanonicalState] = llvmManager->getStatePtrType();
    argsType[GetBufferElementAddress::Args::ElementSize]    = llvmManager->getI32Type();
    argsType[GetBufferElementAddress::Args::x]              = llvmManager->getI64Type();

    // Optional arguments.
    if( dimensions > 1 )
        argsType[GetBufferElementAddress::Args::y] = llvmManager->getI64Type();
    if( dimensions > 2 )
        argsType[GetBufferElementAddress::Args::z] = llvmManager->getI64Type();

    argsType.resize( GetBufferElementAddress::MANDATORY_ARGS_NUMBER + dimensions - 1 );

    RT_ASSERT_ARG_TYPES_NON_NULL( argsType, "optixi_getBufferElementAddress" );

    FunctionType* functionType = FunctionType::get( llvmManager->getI64Type(), argsType, false );

    return functionType;
}

Function* GetBufferElementAddressBuilder::createFunction( VariableReference* varRef, unsigned dimensions, LLVMManager* llvmManager )
{
    auto        functionType = createType( dimensions, llvmManager );
    std::string name         = GetBufferElementAddress::createUniqueName( varRef );
    Function*   result       = dyn_cast<Function>( m_module->getOrInsertFunction( name, functionType ) );
    // This two properties should make sure that optixi_getBufferElementAddress is optimized away is its output is not used.
    result->setDoesNotAccessMemory();
    result->setDoesNotThrow();
    return result;
}

unsigned GetBufferElementAddressBuilder::getDimensionality( const Function* function ) const
{
    return GetBufferElementAddress::getDimensionality( function );
}

CallInst* GetBufferElementAddressBuilder::create( Function* function, Instruction* insertBefore )
{
    unsigned numParams    = function->getFunctionType()->getNumParams();
    unsigned expectedDims = numParams - GetBufferElementAddress::MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT_MSG( expectedDims >= 1 && expectedDims <= 3, "Malformed function " + function->getName().str() );

    std::vector<Value*> args( GetBufferElementAddress::Args::END );

    args[GetBufferElementAddress::Args::CanonicalState] = m_canonicalState;
    args[GetBufferElementAddress::Args::ElementSize]    = m_elementSize;
    args[GetBufferElementAddress::Args::x]              = m_x;

    unsigned dimensions = 1;

    if( m_y != nullptr )
    {
        args[GetBufferElementAddress::Args::y] = m_y;
        dimensions                             = 2;
    }

    if( m_z != nullptr )
    {
        args[GetBufferElementAddress::Args::z] = m_z;
        dimensions                             = 3;
    }

    RT_ASSERT( dimensions == expectedDims );
    args.resize( GetBufferElementAddress::MANDATORY_ARGS_NUMBER + dimensions - 1 );

    RT_ASSERT_ARGS_NON_NULL( args, "optixi_getBufferElementAddress" );

    corelib::CoreIRBuilder irb{insertBefore};
    return irb.CreateCall( function, args );
}

GetBufferElementAddressBuilder& GetBufferElementAddressBuilder::setCanonicalState( Value* canonicalState )
{
    m_canonicalState = canonicalState;
    return *this;
}

GetBufferElementAddressBuilder& GetBufferElementAddressBuilder::setElementSize( Value* elementSize )
{
    m_elementSize = elementSize;
    return *this;
}

GetBufferElementAddressBuilder& GetBufferElementAddressBuilder::setX( Value* x )
{
    m_x = x;
    return *this;
}

GetBufferElementAddressBuilder& GetBufferElementAddressBuilder::setY( Value* y )
{
    m_y = y;
    return *this;
}

GetBufferElementAddressBuilder& GetBufferElementAddressBuilder::setZ( Value* z )
{
    m_z = z;
    return *this;
}

// -----------------------------------------------------------------------------
const std::string GetBufferElementAddressFromId::PREFIX{"optixi_getBufferElementAddressFromId"};

bool GetBufferElementAddressFromId::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    if( calledFunction == nullptr )
        return false;

    return matchName( calledFunction->getName() );
}

bool GetBufferElementAddressFromId::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool GetBufferElementAddressFromId::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

std::string GetBufferElementAddressFromId::createUniqueName( unsigned int dimensionality )
{
    return PREFIX + '.' + std::to_string( dimensionality );
}

unsigned GetBufferElementAddressFromId::getDimensionality( const Function* function )
{
    unsigned callArgsNumber = function->arg_size();
    unsigned dimensions     = callArgsNumber - MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT( 1 <= dimensions && dimensions <= 3 );
    return dimensions;
}

unsigned GetBufferElementAddressFromId::getDimensionality() const
{
    return getDimensionality( getCalledFunction() );
}

bool GetBufferElementAddressFromId::matchName( StringRef name )
{
    return name.startswith( PREFIX );
}

Value* GetBufferElementAddressFromId::getStatePtr() const
{
    return getArgOperand( Args::CanonicalState );
}

Value* GetBufferElementAddressFromId::getBufferId() const
{
    return getArgOperand( Args::BufferId );
}

Value* GetBufferElementAddressFromId::getElementSize() const
{
    return getArgOperand( Args::ElementSize );
}

Value* GetBufferElementAddressFromId::getX() const
{
    return getArgOperand( Args::x );
}

Value* GetBufferElementAddressFromId::getY() const
{
    return getArgOperand( Args::y );
}

Value* GetBufferElementAddressFromId::getZ() const
{
    return getArgOperand( Args::z );
}

Value* GetBufferElementAddressFromId::getIndex( unsigned dimension ) const
{
    return getArgOperand( Args::x + dimension );
}

// -----------------------------------------------------------------------------
GetBufferElementAddressFromIdBuilder::GetBufferElementAddressFromIdBuilder( Module* module )
    : m_module( module )
{
}

FunctionType* GetBufferElementAddressFromIdBuilder::createType( unsigned dimensions, LLVMManager* llvmManager )
{
    std::vector<Type*> argsType( GetBufferElementAddressFromId::Args::END );

    // Mandatory arguments.
    argsType[GetBufferElementAddressFromId::Args::CanonicalState] = llvmManager->getStatePtrType();
    argsType[GetBufferElementAddressFromId::Args::BufferId]       = llvmManager->getI32Type();
    argsType[GetBufferElementAddressFromId::Args::ElementSize]    = llvmManager->getI32Type();
    argsType[GetBufferElementAddressFromId::Args::x]              = llvmManager->getI64Type();

    // Optional arguments.
    if( dimensions > 1 )
        argsType[GetBufferElementAddressFromId::Args::y] = llvmManager->getI64Type();
    if( dimensions > 2 )
        argsType[GetBufferElementAddressFromId::Args::z] = llvmManager->getI64Type();

    argsType.resize( GetBufferElementAddressFromId::MANDATORY_ARGS_NUMBER + dimensions - 1 );

    RT_ASSERT_ARG_TYPES_NON_NULL( argsType, "optixi_get_buffer_id_element_address" );

    FunctionType* functionType = FunctionType::get( llvmManager->getI64Type(), argsType, false );

    return functionType;
}

Function* GetBufferElementAddressFromIdBuilder::createFunction( unsigned dimensions, LLVMManager* llvmManager )
{
    auto        functionType = createType( dimensions, llvmManager );
    std::string name         = GetBufferElementAddressFromId::createUniqueName( dimensions );
    Function*   result       = dyn_cast<Function>( m_module->getOrInsertFunction( name, functionType ) );
    // These two properties should make sure that optixi_get_buffer_id_element_address is optimized away if its result is not used.
    result->setDoesNotAccessMemory();
    result->setDoesNotThrow();
    return result;
}

unsigned GetBufferElementAddressFromIdBuilder::getDimensionality( const Function* function ) const
{
    return GetBufferElementAddressFromId::getDimensionality( function );
}

CallInst* GetBufferElementAddressFromIdBuilder::create( Function* function, Instruction* insertBefore )
{
    unsigned numParams    = function->getFunctionType()->getNumParams();
    unsigned expectedDims = numParams - GetBufferElementAddressFromId::MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT_MSG( expectedDims >= 1 && expectedDims <= 3, "Malformed function " + function->getName().str() );

    std::vector<Value*> args( GetBufferElementAddressFromId::Args::END );

    args[GetBufferElementAddressFromId::Args::CanonicalState] = m_canonicalState;
    args[GetBufferElementAddressFromId::Args::BufferId]       = m_id;
    args[GetBufferElementAddressFromId::Args::ElementSize]    = m_elementSize;
    args[GetBufferElementAddressFromId::Args::x]              = m_x;

    unsigned dimensions = 1;

    if( m_y != nullptr )
    {
        args[GetBufferElementAddressFromId::Args::y] = m_y;
        dimensions                                   = 2;
    }

    if( m_z != nullptr )
    {
        args[GetBufferElementAddressFromId::Args::z] = m_z;
        dimensions                                   = 3;
    }

    RT_ASSERT( dimensions == expectedDims );
    args.resize( GetBufferElementAddressFromId::MANDATORY_ARGS_NUMBER + dimensions - 1 );

    RT_ASSERT_ARGS_NON_NULL( args, "optixi_get_buffer_id_element_address" );

    corelib::CoreIRBuilder irb{insertBefore};
    return irb.CreateCall( function, args );
}

GetBufferElementAddressFromIdBuilder& GetBufferElementAddressFromIdBuilder::setCanonicalState( Value* canonicalState )
{
    m_canonicalState = canonicalState;
    return *this;
}

GetBufferElementAddressFromIdBuilder& GetBufferElementAddressFromIdBuilder::setBufferId( Value* id )
{
    m_id = id;
    return *this;
}

GetBufferElementAddressFromIdBuilder& GetBufferElementAddressFromIdBuilder::setElementSize( Value* elementSize )
{
    m_elementSize = elementSize;
    return *this;
}

GetBufferElementAddressFromIdBuilder& GetBufferElementAddressFromIdBuilder::setX( Value* x )
{
    m_x = x;
    return *this;
}

GetBufferElementAddressFromIdBuilder& GetBufferElementAddressFromIdBuilder::setY( Value* y )
{
    m_y = y;
    return *this;
}

GetBufferElementAddressFromIdBuilder& GetBufferElementAddressFromIdBuilder::setZ( Value* z )
{
    m_z = z;
    return *this;
}

// -----------------------------------------------------------------------------
const std::string GetPayloadAddressCall::GET_PAYLOAD_ADDRESS = "optixi_get_payload_address";

llvm::Value* GetPayloadAddressCall::getStatePtr() const
{
    return getArgOperand( Args::CanonicalState );
}

bool GetPayloadAddressCall::classof( const llvm::CallInst* inst )
{
    if( const Function* calledFunction = inst->getCalledFunction() )
        return isOptixIntrinsic( calledFunction );
    return false;
}

bool GetPayloadAddressCall::classof( const llvm::Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool GetPayloadAddressCall::isIntrinsic( const llvm::Function* function )
{
    return function->getName() == GET_PAYLOAD_ADDRESS;
}

FunctionType* GetPayloadAddressCall::createType( LLVMManager* llvmManager )
{
    std::vector<Type*> argsType( GetPayloadAddressCall::Args::END );

    argsType[GetPayloadAddressCall::Args::CanonicalState] = llvmManager->getStatePtrType();

    FunctionType* functionType = FunctionType::get( llvmManager->getI64Type(), argsType, false );

    return functionType;
}

llvm::CallInst* GetPayloadAddressCall::create( llvm::Function* function, llvm::Value* statePtr, Instruction* insertBefore )
{
    std::vector<Value*> args( GetPayloadAddressCall::Args::END );

    args[GetPayloadAddressCall::Args::CanonicalState] = statePtr;

    corelib::CoreIRBuilder irb{insertBefore};
    return irb.CreateCall( function, args );
}

Function* GetPayloadAddressCall::createFunction( LLVMManager* llvmManager, Module* module )
{
    FunctionType* functionType = createType( llvmManager );
    Constant*     constant     = module->getOrInsertFunction( GET_PAYLOAD_ADDRESS, functionType );
    return dyn_cast<Function>( constant );
}

// -----------------------------------------------------------------------------
const std::string GetBufferElement::PREFIX{"optixi_getBufferElement"};
llvm::Regex       GetBufferElement::nameRegex{PREFIX + "\\.(" + optixi_VariableReferenceUniqueNameRegex + ")\\..+$"};

bool GetBufferElement::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    if( calledFunction == nullptr )
        return false;

    return matchName( calledFunction->getName() );
}

bool GetBufferElement::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool GetBufferElement::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

bool GetBufferElement::isIntrinsic( const Function* function, const std::string& varRefUniqueName )
{
    return optix::isIntrinsic( function->getName(), PREFIX, varRefUniqueName );
}

bool GetBufferElement::matchName( StringRef name )
{
    SmallVector<StringRef, 2> matches;
    if( !nameRegex.match( name, &matches ) )
        return false;
    return matches.size() == 2;
}

Value* GetBufferElement::getStatePtr() const
{
    return getArgOperand( Args::CanonicalState );
}

Value* GetBufferElement::getElementSize() const
{
    return getArgOperand( Args::ElementSize );
}

Value* GetBufferElement::getX() const
{
    return getArgOperand( Args::x );
}

Value* GetBufferElement::getY() const
{
    return getArgOperand( Args::y );
}

Value* GetBufferElement::getZ() const
{
    return getArgOperand( Args::z );
}

Value* GetBufferElement::getIndex( unsigned dimension ) const
{
    return getArgOperand( Args::x + dimension );
}

Value* GetBufferElement::getOffset() const
{
    return getArgOperand( Args::offset );
}

std::string GetBufferElement::parseUniqueName() const
{
    return parseUniqueName( getCalledFunction()->getName() );
}

std::string GetBufferElement::parseUniqueName( StringRef name )
{
    SmallVector<StringRef, 2> matches;
    RT_ASSERT( nameRegex.match( name, &matches ) );
    RT_ASSERT( matches.size() == 2 );
    return matches[1];
}

std::string GetBufferElement::createUniqueName( const VariableReference* vref )
{
    return PREFIX + "." + vref->getUniversallyUniqueName();
}

unsigned GetBufferElement::getDimensionality( const Function* function )
{
    unsigned callArgsNumber = function->arg_size();
    unsigned dimensions     = callArgsNumber - MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT( 1 <= dimensions && dimensions <= 3 );
    return dimensions;
}

unsigned GetBufferElement::getDimensionality() const
{
    return getDimensionality( getCalledFunction() );
}

// -----------------------------------------------------------------------------
const std::string LoadOrRequestBufferElement::PREFIX{"optixi_loadOrRequestBufferElement"};
Regex LoadOrRequestBufferElement::nameRegex{PREFIX + "\\.(" + optixi_VariableReferenceUniqueNameRegex + ")$"};

bool LoadOrRequestBufferElement::classof( const CallInst* inst )
{
    if( const Function* calledFunction = inst->getCalledFunction() )
        return matchName( calledFunction->getName() );

    return false;
}

bool LoadOrRequestBufferElement::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool LoadOrRequestBufferElement::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

bool LoadOrRequestBufferElement::isIntrinsic( const Function* function, const std::string& varRefUniqueName )
{
    return optix::isIntrinsic( function->getName(), PREFIX, varRefUniqueName );
}

bool LoadOrRequestBufferElement::matchName( StringRef name )
{
    SmallVector<StringRef, 2> matches;
    if( !nameRegex.match( name, &matches ) )
        return false;
    return matches.size() == 2;
}

llvm::Value* LoadOrRequestBufferElement::getStatePtr() const
{
    return getArgOperand( Args::CanonicalState );
}

unsigned int LoadOrRequestBufferElement::getDimensionality() const
{
    return getDimensionality( getCalledFunction() );
}

Value* LoadOrRequestBufferElement::getElementSize() const
{
    return getArgOperand( Args::ElementSize );
}

Value* LoadOrRequestBufferElement::getX() const
{
    return getArgOperand( Args::x );
}

Value* LoadOrRequestBufferElement::getY() const
{
    return getArgOperand( Args::y );
}

Value* LoadOrRequestBufferElement::getZ() const
{
    return getArgOperand( Args::z );
}

Value* LoadOrRequestBufferElement::getPointer() const
{
    return getArgOperand( Args::Pointer );
}

std::string LoadOrRequestBufferElement::parseUniqueName() const
{
    return parseUniqueName( getCalledFunction()->getName() );
}

std::string LoadOrRequestBufferElement::parseUniqueName( StringRef name )
{
    SmallVector<StringRef, 2> matches;
    RT_ASSERT( nameRegex.match( name, &matches ) );
    RT_ASSERT( matches.size() == 2 );
    return matches[1];
}

std::string LoadOrRequestBufferElement::createUniqueName( const VariableReference* vref )
{
    return PREFIX + '.' + vref->getUniversallyUniqueName();
}

unsigned int LoadOrRequestBufferElement::getDimensionality( const Function* function )
{
    return function->getFunctionType()->getNumParams() - MANDATORY_ARGS_NUMBER + 1;
}

// -----------------------------------------------------------------------------

Regex LoadOrRequestTextureElement::nameRegex{"optixi_texture(Lod|Grad)?LoadOrRequest[1-3]"};

bool LoadOrRequestTextureElement::classof( const CallInst* inst )
{
    if( const Function* calledFunction = inst->getCalledFunction() )
        return matchName( calledFunction->getName() );

    return false;
}

bool LoadOrRequestTextureElement::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool LoadOrRequestTextureElement::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

bool LoadOrRequestTextureElement::matchName( StringRef name )
{
    return nameRegex.match( name );
}

unsigned int LoadOrRequestTextureElement::getDimensionality() const
{
    return getCalledFunction()->getFunctionType()->getNumParams() - MANDATORY_ARGS_NUMBER + 1;
}

LoadOrRequestTextureElement::Kind LoadOrRequestTextureElement::getKind() const
{
    SmallVector<StringRef, 2> matches;
    RT_ASSERT( nameRegex.match( getCalledFunction()->getName(), &matches ) );
    RT_ASSERT( matches.size() == 2 );
    const std::string& kindStr = matches[1];
    if( kindStr == "Grad" )
        return Grad;
    else if( kindStr == "Lod" )
        return Lod;
    RT_ASSERT( kindStr.empty() );
    return Nomip;
}

// -----------------------------------------------------------------------------
const std::string SetBufferElement::PREFIX{"optixi_setBufferElement"};
llvm::Regex       SetBufferElement::nameRegex{PREFIX + "\\.(" + optixi_VariableReferenceUniqueNameRegex + ")\\..+$"};

//optixi_setBufferElement.function_ptx0x12a344f.output.i32

bool SetBufferElement::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    if( calledFunction == nullptr )
        return false;

    return matchName( calledFunction->getName() );
}

bool SetBufferElement::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool SetBufferElement::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

bool SetBufferElement::isIntrinsic( const Function* function, const std::string& varRefUniqueName )
{
    return optix::isIntrinsic( function->getName(), PREFIX, varRefUniqueName );
}

bool SetBufferElement::matchName( StringRef name )
{
    SmallVector<StringRef, 2> matches;
    if( !nameRegex.match( name, &matches ) )
        return false;
    return matches.size() == 2;
}

Value* SetBufferElement::getStatePtr() const
{
    return getArgOperand( Args::CanonicalState );
}

Value* SetBufferElement::getElementSize() const
{
    return getArgOperand( Args::ElementSize );
}

Value* SetBufferElement::getX() const
{
    return getArgOperand( Args::x );
}

Value* SetBufferElement::getY() const
{
    return getArgOperand( Args::y );
}

Value* SetBufferElement::getZ() const
{
    return getArgOperand( Args::z );
}

Value* SetBufferElement::getIndex( unsigned dimension ) const
{
    return getArgOperand( Args::x + dimension );
}

Value* SetBufferElement::getOffset() const
{
    return getArgOperand( Args::offset );
}

Value* SetBufferElement::getValueToSet() const
{
    return getArgOperand( Args::offset + getDimensionality() + 1 );
}

std::string SetBufferElement::parseUniqueName() const
{
    return parseUniqueName( getCalledFunction()->getName() );
}

std::string SetBufferElement::parseUniqueName( StringRef name )
{
    SmallVector<StringRef, 2> matches;
    RT_ASSERT( nameRegex.match( name, &matches ) );
    RT_ASSERT( matches.size() == 2 );
    return matches[1];
}

std::string SetBufferElement::createUniqueName( const VariableReference* vref )
{
    return PREFIX + '.' + vref->getUniversallyUniqueName();
}

unsigned SetBufferElement::getDimensionality( const Function* function )
{
    unsigned callArgsNumber = function->arg_size();
    unsigned dimensions     = callArgsNumber - MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT( 1 <= dimensions && dimensions <= 3 );
    return dimensions;
}

unsigned SetBufferElement::getDimensionality() const
{
    return getDimensionality( getCalledFunction() );
}

// -----------------------------------------------------------------------------
const std::string GetBufferElementFromId::PREFIX{"optixi_getBufferElementFromId"};
llvm::Regex       GetBufferElementFromId::nameRegex{PREFIX + "\\..+$"};

//optixi_setBufferElement.1.i32.i32

bool GetBufferElementFromId::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    if( calledFunction == nullptr )
        return false;

    return matchName( calledFunction->getName() );
}

bool GetBufferElementFromId::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool GetBufferElementFromId::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

std::string GetBufferElementFromId::createUniqueName( unsigned int dimensionality, const llvm::Type* valueType )
{
    // TODO: corelib shouldn't take valueType by non-const pointer
    return PREFIX + '.' + std::to_string( dimensionality ) + '.' + corelib::getTypeName( const_cast<llvm::Type*>( valueType ) );
}

bool GetBufferElementFromId::matchName( StringRef name )
{
    return name.startswith( PREFIX );
}

Value* GetBufferElementFromId::getStatePtr() const
{
    return getArgOperand( Args::CanonicalState );
}

Value* GetBufferElementFromId::getBufferId() const
{
    return getArgOperand( Args::BufferId );
}

Value* GetBufferElementFromId::getElementSize() const
{
    return getArgOperand( Args::ElementSize );
}

Value* GetBufferElementFromId::getX() const
{
    return getArgOperand( Args::x );
}

Value* GetBufferElementFromId::getY() const
{
    return getArgOperand( Args::y );
}

Value* GetBufferElementFromId::getZ() const
{
    return getArgOperand( Args::z );
}

Value* GetBufferElementFromId::getIndex( unsigned dimension ) const
{
    return getArgOperand( Args::x + dimension );
}

Value* GetBufferElementFromId::getOffset() const
{
    return getArgOperand( Args::offset );
}

unsigned GetBufferElementFromId::getDimensionality( const Function* function )
{
    unsigned callArgsNumber = function->arg_size();
    unsigned dimensions     = callArgsNumber - MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT( 1 <= dimensions && dimensions <= 3 );
    return dimensions;
}

unsigned GetBufferElementFromId::getDimensionality() const
{
    return getDimensionality( getCalledFunction() );
}

// -----------------------------------------------------------------------------
const std::string SetBufferElementFromId::PREFIX{"optixi_setBufferElementFromId"};
llvm::Regex       SetBufferElementFromId::nameRegex{PREFIX + "\\..+$"};

bool SetBufferElementFromId::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    if( calledFunction == nullptr )
        return false;

    return matchName( calledFunction->getName() );
}

bool SetBufferElementFromId::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool SetBufferElementFromId::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

std::string SetBufferElementFromId::createUniqueName( unsigned int dimensionality, const llvm::Type* valueType )
{
    // TODO: corelib shouldn't take valueType by non-const pointer
    return PREFIX + '.' + std::to_string( dimensionality ) + '.' + corelib::getTypeName( const_cast<llvm::Type*>( valueType ) );
}

bool SetBufferElementFromId::matchName( StringRef name )
{
    return name.startswith( PREFIX );
}

Value* SetBufferElementFromId::getStatePtr() const
{
    return getArgOperand( Args::CanonicalState );
}

Value* SetBufferElementFromId::getBufferId() const
{
    return getArgOperand( Args::BufferId );
}

Value* SetBufferElementFromId::getElementSize() const
{
    return getArgOperand( Args::ElementSize );
}

Value* SetBufferElementFromId::getX() const
{
    return getArgOperand( Args::x );
}

Value* SetBufferElementFromId::getY() const
{
    return getArgOperand( Args::y );
}

Value* SetBufferElementFromId::getZ() const
{
    return getArgOperand( Args::z );
}

Value* SetBufferElementFromId::getIndex( unsigned dimension ) const
{
    return getArgOperand( Args::x + dimension );
}

Value* SetBufferElementFromId::getOffset() const
{
    return getArgOperand( Args::offset );
}

Value* SetBufferElementFromId::getValueToSet() const
{
    return getArgOperand( Args::offset + getDimensionality() + 1 );
}

unsigned SetBufferElementFromId::getDimensionality( const Function* function )
{
    unsigned callArgsNumber = function->arg_size();
    unsigned dimensions     = callArgsNumber - MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT( 1 <= dimensions && dimensions <= 3 );
    return dimensions;
}

unsigned SetBufferElementFromId::getDimensionality() const
{
    return getDimensionality( getCalledFunction() );
}

// -----------------------------------------------------------------------------
const std::string SetAttributeValue::PREFIX{"optixi_setAttributeValue"};
llvm::Regex       SetAttributeValue::nameRegex{PREFIX + "\\.(" + optixi_VariableReferenceUniqueNameRegex + ")\\..+$"};

bool SetAttributeValue::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    if( calledFunction == nullptr )
        return false;

    // Check if the name matches the expected one.
    SmallVector<StringRef, 2> matches;
    StringRef functionName = calledFunction->getName();
    if( !nameRegex.match( functionName, &matches ) )
        return false;
    return matches.size() == 2;
}

bool SetAttributeValue::isIntrinsic( const Function* function )
{
    return nameRegex.match( function->getName() );
}

// FIXME: this function initialized the output parameters even if we don't have a full match.
bool SetAttributeValue::parseUniqueName( const llvm::Function* function, llvm::StringRef& uniqueName )
{
    StringRef name           = function->getName();
    const int MATCHES_NUMBER = 2;
    SmallVector<llvm::StringRef, MATCHES_NUMBER> matches;
    if( !nameRegex.match( name, &matches ) )
        return false;
    if( matches.size() != MATCHES_NUMBER )
        return false;
    uniqueName = matches[1];
    return true;
}

bool SetAttributeValue::parseUniqueName( llvm::StringRef& uniqueName )
{
    return SetAttributeValue::parseUniqueName( getCalledFunction(), uniqueName );
}

// -----------------------------------------------------------------------------
const std::string GetAttributeValue::PREFIX{"optixi_getAttributeValue"};
llvm::Regex       GetAttributeValue::nameRegex{PREFIX + "\\.(" + optixi_VariableReferenceUniqueNameRegex + ")\\..+$"};

bool GetAttributeValue::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    if( calledFunction == nullptr )
        return false;

    // Check if the name matches the expected one.
    SmallVector<StringRef, 2> matches;
    StringRef functionName = calledFunction->getName();
    if( !nameRegex.match( functionName, &matches ) )
        return false;
    return matches.size() == 2;
}

bool GetAttributeValue::isIntrinsic( const Function* function )
{
    return nameRegex.match( function->getName() );
}

bool GetAttributeValue::parseUniqueName( const llvm::Function* function, llvm::StringRef& uniqueName )
{
    StringRef name           = function->getName();
    const int MATCHES_NUMBER = 2;
    SmallVector<llvm::StringRef, MATCHES_NUMBER> matches;
    if( !nameRegex.match( name, &matches ) )
        return false;
    if( matches.size() != MATCHES_NUMBER )
        return false;
    uniqueName = matches[1];
    return true;
}

bool GetAttributeValue::parseUniqueName( llvm::StringRef& uniqueName )
{
    return GetAttributeValue::parseUniqueName( getCalledFunction(), uniqueName );
}

// -----------------------------------------------------------------------------
namespace {
const std::string uniqueName = "[0-9A-Za-z_]+";
}
const std::string ReportFullIntersection::PREFIX{"optixi_reportFullIntersection"};
llvm::Regex       ReportFullIntersection::nameRegex{PREFIX + "\\.(" + uniqueName + ")\\..+$"};

bool ReportFullIntersection::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    if( calledFunction == nullptr )
        return false;

    // Check if the name matches the expected one.
    SmallVector<StringRef, 2> matches;
    StringRef functionName = calledFunction->getName();
    if( !nameRegex.match( functionName, &matches ) )
        return false;
    return matches.size() == 2;
}

bool ReportFullIntersection::isIntrinsic( const Function* function )
{
    return nameRegex.match( function->getName() );
}

bool ReportFullIntersection::parseUniqueName( const llvm::Function* function, llvm::StringRef& name )
{
    StringRef fname          = function->getName();
    const int MATCHES_NUMBER = 2;
    SmallVector<llvm::StringRef, MATCHES_NUMBER> matches;
    if( !nameRegex.match( fname, &matches ) )
        return false;
    if( matches.size() != MATCHES_NUMBER )
        return false;
    name = matches[1];
    return true;
}

bool ReportFullIntersection::parseUniqueName( llvm::StringRef& name )
{
    return ReportFullIntersection::parseUniqueName( getCalledFunction(), name );
}

llvm::Value* ReportFullIntersection::getHitT() const
{
    return getArgOperand( Args::HitT );
}

llvm::Value* ReportFullIntersection::getMaterialIndex() const
{
    return getArgOperand( Args::MaterialIndex );
}

llvm::Value* ReportFullIntersection::getHitKind() const
{
    return getArgOperand( Args::HitKind );
}

llvm::Value* ReportFullIntersection::getAttributeData() const
{
    return getArgOperand( Args::AttributeData );
}

// -----------------------------------------------------------------------------
const std::string IsPotentialIntersection::PREFIX{"optixi_isPotentialIntersection"};
llvm::Regex       IsPotentialIntersection::nameRegex{PREFIX + "$"};

bool IsPotentialIntersection::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    if( calledFunction == nullptr )
        return false;

    // Check if the name matches the expected one.
    SmallVector<StringRef, 2> matches;
    StringRef functionName = calledFunction->getName();
    if( !nameRegex.match( functionName, &matches ) )
        return false;
    return matches.size() == 1;
}

bool IsPotentialIntersection::isIntrinsic( const Function* function )
{
    return nameRegex.match( function->getName() );
}

llvm::Value* IsPotentialIntersection::getHitT() const
{
    return getArgOperand( Args::HitT );
}
