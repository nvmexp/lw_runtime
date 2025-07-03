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

#include <ExelwtionStrategy/RTX/RTXIntrinsics.h>

#include <Context/LLVMManager.h>
#include <ExelwtionStrategy/Compile.h>
#include <FrontEnd/Canonical/IntrinsicsAssertions.h>

#include <corelib/compiler/LLVMUtil.h>

using namespace corelib;
using namespace llvm;

namespace optix {

namespace {

std::string DimensionalityRegex = "[123]";
std::string CounterRegex        = "[0-9]+";

}  // namespace

// --- RtxiGetBufferId -----------------------------------------------------------------------------

const std::string RtxiGetBufferId::PREFIX = "rtxiGetBufferId";

Regex RtxiGetBufferId::nameRegex = Regex( PREFIX + "\\.(" + optixi_VariableReferenceUniqueNameRegex + ")$" );

std::string RtxiGetBufferId::getVarRefUniqueName()
{
    StringRef functionName = getCalledFunction()->getName();
    SmallVector<StringRef, 2> matches;
    bool result = nameRegex.match( functionName, &matches );
    RT_ASSERT( result && ( matches.size() == 2 ) );
    (void)result;
    return matches[1];
}

std::string RtxiGetBufferId::getFunctionName( const std::string& varRefUniqueName )
{
    return PREFIX + "." + varRefUniqueName;
}

bool RtxiGetBufferId::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    return calledFunction && matchName( calledFunction->getName() );
}

bool RtxiGetBufferId::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool RtxiGetBufferId::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

bool RtxiGetBufferId::isIntrinsic( const Function* function, const std::string& varRefUniqueName )
{
    return optix::isIntrinsic( function->getName(), PREFIX, varRefUniqueName );
}

bool RtxiGetBufferId::matchName( StringRef functionName )
{
    SmallVector<StringRef, 2> matches;
    return nameRegex.match( functionName, &matches ) && matches.size() == 2;
}

// --- RtxiGetBufferIdBuilder ----------------------------------------------------------------------

RtxiGetBufferIdBuilder::RtxiGetBufferIdBuilder( llvm::Module* module, llvm::Type* statePtrTy, llvm::Instruction* insertBefore )
    : m_module( module )
    , m_statePtrTy( statePtrTy )
    , m_builder( insertBefore )
{
}

CallInst* RtxiGetBufferIdBuilder::createCall( const std::string& varRefUniqueName )
{
    Function* f = createFunction( varRefUniqueName );

    std::vector<Value*> args( RtxiGetBufferId::Args::END );
    args[RtxiGetBufferId::Args::StatePtr] = m_statePtr;

    RT_ASSERT_ARGS_NON_NULL( args, "rtxiGetBufferId" );

    return m_builder.CreateCall( f, args, "bufferId" );
}

Function* RtxiGetBufferIdBuilder::createFunction( const std::string& varRefUniqueName )
{
    const std::string& name       = RtxiGetBufferId::getFunctionName( varRefUniqueName );
    FunctionType*      functionTy = createType();
    Function*          result     = dyn_cast<Function>( m_module->getOrInsertFunction( name, functionTy ) );
    result->addFnAttr( Attribute::NoUnwind );
    return result;
}

FunctionType* RtxiGetBufferIdBuilder::createType()
{
    Type* argTy = m_statePtrTy;
    return FunctionType::get( m_builder.getInt32Ty(), argTy, false );
}

// --- RtxiGetBufferElement ------------------------------------------------------------------------

const std::string RtxiGetBufferSize::PREFIX = "rtxiGetBufferSize";

Regex RtxiGetBufferSize::nameRegex = Regex( PREFIX + "\\.(" + optixi_VariableReferenceUniqueNameRegex + ")$" );

std::string RtxiGetBufferSize::getVarRefUniqueName()
{
    StringRef functionName = getCalledFunction()->getName();
    SmallVector<StringRef, 2> matches;
    bool result = nameRegex.match( functionName, &matches );
    RT_ASSERT( result && ( matches.size() == 2 ) );
    (void)result;
    return matches[1];
}

std::string RtxiGetBufferSize::getFunctionName( const std::string& varRefUniqueName )
{
    return PREFIX + "." + varRefUniqueName;
}

bool RtxiGetBufferSize::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    return calledFunction && matchName( calledFunction->getName() );
}

bool RtxiGetBufferSize::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool RtxiGetBufferSize::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

bool RtxiGetBufferSize::isIntrinsic( const Function* function, const std::string& varRefUniqueName )
{
    return optix::isIntrinsic( function->getName(), PREFIX, varRefUniqueName );
}

bool RtxiGetBufferSize::matchName( StringRef functionName )
{
    SmallVector<StringRef, 2> matches;
    return nameRegex.match( functionName, &matches ) && matches.size() == 2;
}

// --- RtxiGetBufferSizeBuilder --------------------------------------------------------------------

RtxiGetBufferSizeBuilder::RtxiGetBufferSizeBuilder( llvm::Module* module, llvm::Type* size3Ty, llvm::Type* statePtrTy, llvm::Instruction* insertBefore )
    : m_module( module )
    , m_size3Ty( size3Ty )
    , m_statePtrTy( statePtrTy )
    , m_builder( insertBefore )
{
}

CallInst* RtxiGetBufferSizeBuilder::createCall( const std::string& varRefUniqueName )
{
    Function* f = createFunction( varRefUniqueName );

    std::vector<Value*> args( RtxiGetBufferSize::Args::END );

    args[RtxiGetBufferSize::Args::StatePtr] = m_statePtr;
    args[RtxiGetBufferSize::Args::BufferId] = m_bufferId;

    args.resize( RtxiGetBufferSize::MANDATORY_ARGS_NUMBER );

    RT_ASSERT_ARGS_NON_NULL( args, "rtxiGetBufferElement" );

    return m_builder.CreateCall( f, args, "bufferSize" );
}

Function* RtxiGetBufferSizeBuilder::createFunction( const std::string& varRefUniqueName )
{
    const std::string& name       = RtxiGetBufferSize::getFunctionName( varRefUniqueName );
    FunctionType*      functionTy = createType();
    Function*          result     = dyn_cast<Function>( m_module->getOrInsertFunction( name, functionTy ) );
    result->addFnAttr( Attribute::NoUnwind );
    return result;
}

FunctionType* RtxiGetBufferSizeBuilder::createType()
{
    std::vector<Type*> argsType( RtxiGetBufferSize::Args::END );

    argsType[RtxiGetBufferSize::Args::StatePtr] = m_statePtrTy;
    argsType[RtxiGetBufferSize::Args::BufferId] = m_builder.getInt32Ty();

    argsType.resize( RtxiGetBufferSize::MANDATORY_ARGS_NUMBER );

    RT_ASSERT_ARG_TYPES_NON_NULL( argsType, "rtxiGetBufferSize" );

    return FunctionType::get( m_size3Ty, argsType, false );
}

// --- RtxiGetBufferElement ------------------------------------------------------------------------

const std::string RtxiGetBufferElement::PREFIX = "rtxiGetBufferElement";

Regex RtxiGetBufferElement::nameRegex = Regex( PREFIX + "\\.(" + optixi_VariableReferenceUniqueNameRegex + ")\\..+$" );

std::string RtxiGetBufferElement::getVarRefUniqueName()
{
    StringRef functionName = getCalledFunction()->getName();
    SmallVector<StringRef, 2> matches;
    bool result = nameRegex.match( functionName, &matches );
    RT_ASSERT( result && ( matches.size() == 2 ) );
    (void)result;
    return matches[1];
}

unsigned int RtxiGetBufferElement::getDimensionality() const
{
    unsigned int callArgsNumber = getCalledFunction()->arg_size();
    unsigned int dimensionality = callArgsNumber - MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );
    return dimensionality;
}

std::string RtxiGetBufferElement::getFunctionName( const std::string& varRefUniqueName, const std::string& elementTypeName )
{
    return PREFIX + "." + varRefUniqueName + "." + elementTypeName;
}

bool RtxiGetBufferElement::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    return calledFunction && matchName( calledFunction->getName() );
}

bool RtxiGetBufferElement::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool RtxiGetBufferElement::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

bool RtxiGetBufferElement::isIntrinsic( const Function* function, const std::string& varRefUniqueName )
{
    return optix::isIntrinsic( function->getName(), PREFIX, varRefUniqueName );
}

bool RtxiGetBufferElement::matchName( StringRef functionName )
{
    SmallVector<StringRef, 2> matches;
    return nameRegex.match( functionName, &matches ) && matches.size() == 2;
}

// --- RtxiGetBufferElementBuilder -----------------------------------------------------------------

RtxiGetBufferElementBuilder::RtxiGetBufferElementBuilder( llvm::Module*      module,
                                                          llvm::Type*        returnTy,
                                                          llvm::Type*        statePtrTy,
                                                          llvm::Instruction* insertBefore )
    : m_module( module )
    , m_returnTy( returnTy )
    , m_statePtrTy( statePtrTy )
    , m_builder( insertBefore )
{
}

CallInst* RtxiGetBufferElementBuilder::createCall( const std::string& varRefUniqueName, int dimensionality )
{
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );

    Function* f = createFunction( varRefUniqueName, dimensionality );

    std::vector<Value*> args( RtxiGetBufferElement::Args::END );

    args[RtxiGetBufferElement::Args::StatePtr]    = m_statePtr;
    args[RtxiGetBufferElement::Args::BufferId]    = m_bufferId;
    args[RtxiGetBufferElement::Args::ElementSize] = m_elementSize;
    args[RtxiGetBufferElement::Args::Offset]      = m_offset;
    args[RtxiGetBufferElement::Args::X]           = m_x;

    if( dimensionality > 1 )
        args[RtxiGetBufferElement::Args::Y] = m_y;
    if( dimensionality > 2 )
        args[RtxiGetBufferElement::Args::Z] = m_z;

    args.resize( RtxiGetBufferElement::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_ARGS_NON_NULL( args, "rtxiGetBufferElement" );

    return m_builder.CreateCall( f, args, "bufferElement" );
}

Function* RtxiGetBufferElementBuilder::createFunction( const std::string& varRefUniqueName, int dimensionality )
{
    const std::string  elementTypeName = corelib::getTypeName( m_returnTy );
    const std::string& name            = RtxiGetBufferElement::getFunctionName( varRefUniqueName, elementTypeName );
    FunctionType*      functionTy      = createType( dimensionality );
    Function*          result          = dyn_cast<Function>( m_module->getOrInsertFunction( name, functionTy ) );
    result->addFnAttr( Attribute::NoUnwind );
    return result;
}

FunctionType* RtxiGetBufferElementBuilder::createType( int dimensionality )
{
    std::vector<Type*> argsType( RtxiGetBufferElement::Args::END );

    argsType[RtxiGetBufferElement::Args::StatePtr]    = m_statePtrTy;
    argsType[RtxiGetBufferElement::Args::BufferId]    = m_builder.getInt32Ty();
    argsType[RtxiGetBufferElement::Args::ElementSize] = m_builder.getInt32Ty();
    argsType[RtxiGetBufferElement::Args::Offset]      = m_builder.getInt64Ty();
    argsType[RtxiGetBufferElement::Args::X]           = m_builder.getInt64Ty();

    if( dimensionality > 1 )
        argsType[RtxiGetBufferElement::Args::Y] = m_builder.getInt64Ty();
    if( dimensionality > 2 )
        argsType[RtxiGetBufferElement::Args::Z] = m_builder.getInt64Ty();

    argsType.resize( RtxiGetBufferElement::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_ARG_TYPES_NON_NULL( argsType, "rtxiGetBufferElement" );

    return FunctionType::get( m_returnTy, argsType, false );
}

// --- RtxiLoadOrRequestBufferElement --------------------------------------------------------------

const std::string RtxiLoadOrRequestBufferElement::PREFIX = "rtxiLoadOrRequestBufferElement";

Regex RtxiLoadOrRequestBufferElement::nameRegex =
    Regex( PREFIX + "\\.(" + optixi_VariableReferenceUniqueNameRegex + ")$" );

std::string RtxiLoadOrRequestBufferElement::getVarRefUniqueName()
{
    StringRef functionName = getCalledFunction()->getName();
    SmallVector<StringRef, 2> matches;
    bool result = nameRegex.match( functionName, &matches );
    RT_ASSERT( result && ( matches.size() == 2 ) );
    (void)result;
    return matches[1];
}

unsigned int RtxiLoadOrRequestBufferElement::getDimensionality() const
{
    unsigned int callArgsNumber = getCalledFunction()->arg_size();
    unsigned int dimensionality = callArgsNumber - MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );
    return dimensionality;
}

std::string RtxiLoadOrRequestBufferElement::getFunctionName( const std::string& varRefUniqueName )
{
    return PREFIX + '.' + varRefUniqueName;
}

bool RtxiLoadOrRequestBufferElement::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    return calledFunction && matchName( calledFunction->getName() );
}

bool RtxiLoadOrRequestBufferElement::classof( const llvm::Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool RtxiLoadOrRequestBufferElement::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

bool RtxiLoadOrRequestBufferElement::isIntrinsic( const Function* function, const std::string& varRefUniqueName )
{
    return optix::isIntrinsic( function->getName(), PREFIX, varRefUniqueName );
}

bool RtxiLoadOrRequestBufferElement::matchName( StringRef functionName )
{
    SmallVector<StringRef, 2> matches;
    return nameRegex.match( functionName, &matches ) && matches.size() == 2;
}

RtxiLoadOrRequestBufferElementBuilder::RtxiLoadOrRequestBufferElementBuilder( llvm::Module*      module,
                                                                              llvm::Type*        returnTy,
                                                                              llvm::Type*        statePtrTy,
                                                                              llvm::Instruction* insertBefore )
    : m_module( module )
    , m_returnTy( returnTy )
    , m_statePtrTy( statePtrTy )
    , m_builder( insertBefore )
{
}

llvm::CallInst* RtxiLoadOrRequestBufferElementBuilder::createCall( const std::string& varRefUniqueName, int dimensionality )
{
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );

    Function* f = createFunction( varRefUniqueName, dimensionality );

    std::vector<Value*> args( RtxiLoadOrRequestBufferElement::Args::END );

    args[RtxiLoadOrRequestBufferElement::Args::StatePtr]    = m_statePtr;
    args[RtxiLoadOrRequestBufferElement::Args::BufferId]    = m_bufferId;
    args[RtxiLoadOrRequestBufferElement::Args::ElementSize] = m_elementSize;
    args[RtxiLoadOrRequestBufferElement::Args::Ptr]         = m_ptr;
    args[RtxiLoadOrRequestBufferElement::Args::X]           = m_x;

    if( dimensionality > 1 )
        args[RtxiLoadOrRequestBufferElement::Args::Y] = m_y;
    if( dimensionality > 2 )
        args[RtxiLoadOrRequestBufferElement::Args::Z] = m_z;

    args.resize( RtxiLoadOrRequestBufferElement::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_MSG( std::all_of( args.begin(), args.end(), []( Value* value ) { return value != nullptr; } ),
                   "Argument not initialized when creating rtxiLoadOrRequestBufferElement" );

    return m_builder.CreateCall( f, args, "loadOrRequestBufferElement" );
}

llvm::Function* RtxiLoadOrRequestBufferElementBuilder::createFunction( const std::string& varRefUniqueName, int dimensionality )
{
    const std::string& name       = RtxiLoadOrRequestBufferElement::getFunctionName( varRefUniqueName );
    FunctionType*      functionTy = createType( dimensionality );
    Function*          result     = dyn_cast<Function>( m_module->getOrInsertFunction( name, functionTy ) );
    result->addFnAttr( Attribute::NoUnwind );
    return result;
}

llvm::FunctionType* RtxiLoadOrRequestBufferElementBuilder::createType( int dimensionality )
{
    std::vector<Type*> argsType( RtxiLoadOrRequestBufferElement::Args::END );

    argsType[RtxiLoadOrRequestBufferElement::Args::StatePtr]    = m_statePtrTy;
    argsType[RtxiLoadOrRequestBufferElement::Args::BufferId]    = m_builder.getInt32Ty();
    argsType[RtxiLoadOrRequestBufferElement::Args::ElementSize] = m_builder.getInt32Ty();
    argsType[RtxiLoadOrRequestBufferElement::Args::Ptr]         = m_builder.getInt64Ty();
    argsType[RtxiLoadOrRequestBufferElement::Args::X]           = m_builder.getInt64Ty();
    if( dimensionality > 1 )
        argsType[RtxiLoadOrRequestBufferElement::Args::Y] = m_builder.getInt64Ty();
    if( dimensionality > 2 )
        argsType[RtxiLoadOrRequestBufferElement::Args::Z] = m_builder.getInt64Ty();

    argsType.resize( RtxiLoadOrRequestBufferElement::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_MSG( std::all_of( argsType.begin(), argsType.end(), []( Type* type ) { return type != nullptr; } ),
                   "Argument type not initialized when creating rtxiGetBufferElement" );

    return FunctionType::get( m_returnTy, argsType, false );
}

// --- RtxiSetBufferElement ------------------------------------------------------------------------

const std::string RtxiSetBufferElement::PREFIX = "rtxiSetBufferElement";

Regex RtxiSetBufferElement::nameRegex = Regex( PREFIX + "\\.(" + optixi_VariableReferenceUniqueNameRegex + ")\\..+$" );

std::string RtxiSetBufferElement::getVarRefUniqueName()
{
    StringRef functionName = getCalledFunction()->getName();
    SmallVector<StringRef, 2> matches;
    bool result = nameRegex.match( functionName, &matches );
    RT_ASSERT( result && ( matches.size() == 2 ) );
    (void)result;
    return matches[1];
}

unsigned int RtxiSetBufferElement::getDimensionality() const
{
    unsigned int callArgsNumber = getCalledFunction()->arg_size();
    unsigned int dimensionality = callArgsNumber - MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );
    return dimensionality;
}

std::string RtxiSetBufferElement::getFunctionName( const std::string& varRefUniqueName, const std::string& elementTypeName )
{
    return PREFIX + "." + varRefUniqueName + "." + elementTypeName;
}

bool RtxiSetBufferElement::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    return calledFunction && matchName( calledFunction->getName() );
}

bool RtxiSetBufferElement::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool RtxiSetBufferElement::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

bool RtxiSetBufferElement::isIntrinsic( const Function* function, const std::string& varRefUniqueName )
{
    return optix::isIntrinsic( function->getName(), PREFIX, varRefUniqueName );
}

bool RtxiSetBufferElement::matchName( StringRef functionName )
{
    SmallVector<StringRef, 2> matches;
    return nameRegex.match( functionName, &matches ) && matches.size() == 2;
}

// --- RtxiSetBufferElementBuilder -----------------------------------------------------------------

RtxiSetBufferElementBuilder::RtxiSetBufferElementBuilder( llvm::Module*      module,
                                                          llvm::Type*        statePtrTy,
                                                          llvm::Type*        valueToSetTy,
                                                          llvm::Instruction* insertBefore )
    : m_module( module )
    , m_statePtrTy( statePtrTy )
    , m_valueToSetTy( valueToSetTy )
    , m_builder( insertBefore )
{
}

CallInst* RtxiSetBufferElementBuilder::createCall( const std::string& varRefUniqueName, int dimensionality )
{
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );

    Function* f = createFunction( varRefUniqueName, dimensionality );

    std::vector<Value*> args( RtxiSetBufferElement::Args::END );

    args[RtxiSetBufferElement::Args::StatePtr]    = m_statePtr;
    args[RtxiSetBufferElement::Args::BufferId]    = m_bufferId;
    args[RtxiSetBufferElement::Args::ElementSize] = m_elementSize;
    args[RtxiSetBufferElement::Args::Offset]      = m_offset;
    args[RtxiSetBufferElement::Args::X]           = m_x;

    if( dimensionality > 1 )
        args[RtxiSetBufferElement::Args::Y] = m_y;
    if( dimensionality > 2 )
        args[RtxiSetBufferElement::Args::Z] = m_z;

    args[RtxiSetBufferElement::Args::Offset + dimensionality + 1] = m_valueToSet;

    args.resize( RtxiSetBufferElement::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_ARGS_NON_NULL( args, "rtxiSetBufferElement" );

    return m_builder.CreateCall( f, args );
}

Function* RtxiSetBufferElementBuilder::createFunction( const std::string& varRefUniqueName, int dimensionality )
{
    const std::string  elementTypeName = corelib::getTypeName( m_valueToSetTy );
    const std::string& name            = RtxiSetBufferElement::getFunctionName( varRefUniqueName, elementTypeName );
    FunctionType*      functionTy      = createType( dimensionality );
    Function*          result          = dyn_cast<Function>( m_module->getOrInsertFunction( name, functionTy ) );
    result->addFnAttr( Attribute::NoUnwind );
    return result;
}

FunctionType* RtxiSetBufferElementBuilder::createType( int dimensionality )
{
    std::vector<Type*> argsType( RtxiSetBufferElement::Args::END );

    argsType[RtxiSetBufferElement::Args::StatePtr]    = m_statePtrTy;
    argsType[RtxiSetBufferElement::Args::BufferId]    = m_builder.getInt32Ty();
    argsType[RtxiSetBufferElement::Args::ElementSize] = m_builder.getInt32Ty();
    argsType[RtxiSetBufferElement::Args::Offset]      = m_builder.getInt64Ty();
    argsType[RtxiSetBufferElement::Args::X]           = m_builder.getInt64Ty();

    if( dimensionality > 1 )
        argsType[RtxiSetBufferElement::Args::Y] = m_builder.getInt64Ty();
    if( dimensionality > 2 )
        argsType[RtxiSetBufferElement::Args::Z] = m_builder.getInt64Ty();

    argsType[RtxiSetBufferElement::Args::Offset + dimensionality + 1] = m_valueToSetTy;

    argsType.resize( RtxiSetBufferElement::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_ARG_TYPES_NON_NULL( argsType, "rtxiSetBufferElement" );

    return FunctionType::get( m_builder.getVoidTy(), argsType, false );
}

// --- RtxiGetBufferElementAddress -----------------------------------------------------------------

const std::string RtxiGetBufferElementAddress::PREFIX = "rtxiGetBufferElementAddress";

Regex RtxiGetBufferElementAddress::nameRegex =
    Regex( PREFIX + "\\.(" + optixi_VariableReferenceUniqueNameRegex + ")$" );

std::string RtxiGetBufferElementAddress::getVarRefUniqueName()
{
    StringRef functionName = getCalledFunction()->getName();
    SmallVector<StringRef, 2> matches;
    bool result = nameRegex.match( functionName, &matches );
    RT_ASSERT( result && ( matches.size() == 2 ) );
    (void)result;
    return matches[1];
}

unsigned int RtxiGetBufferElementAddress::getDimensionality() const
{
    unsigned int callArgsNumber = getCalledFunction()->arg_size();
    unsigned int dimensionality = callArgsNumber - MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );
    return dimensionality;
}

std::string RtxiGetBufferElementAddress::getFunctionName( const std::string& varRefUniqueName )
{
    return PREFIX + "." + varRefUniqueName;
}

bool RtxiGetBufferElementAddress::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    return calledFunction && matchName( calledFunction->getName() );
}

bool RtxiGetBufferElementAddress::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool RtxiGetBufferElementAddress::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

bool RtxiGetBufferElementAddress::isIntrinsic( const Function* function, const std::string& varRefUniqueName )
{
    return optix::isIntrinsic( function->getName(), PREFIX, varRefUniqueName );
}

bool RtxiGetBufferElementAddress::matchName( StringRef functionName )
{
    SmallVector<StringRef, 2> matches;
    return nameRegex.match( functionName, &matches ) && matches.size() == 2;
}

// --- RtxiGetBufferElementAddressBuilder ----------------------------------------------------------

RtxiGetBufferElementAddressBuilder::RtxiGetBufferElementAddressBuilder( llvm::Module* module, llvm::Type* statePtrTy, llvm::Instruction* insertBefore )
    : m_module( module )
    , m_statePtrTy( statePtrTy )
    , m_builder( insertBefore )
{
}

CallInst* RtxiGetBufferElementAddressBuilder::createCall( const std::string& varRefUniqueName, int dimensionality )
{
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );

    Function* f = createFunction( varRefUniqueName, dimensionality );

    std::vector<Value*> args( RtxiGetBufferElementAddress::Args::END );

    args[RtxiGetBufferElementAddress::Args::StatePtr]    = m_statePtr;
    args[RtxiGetBufferElementAddress::Args::BufferId]    = m_bufferId;
    args[RtxiGetBufferElementAddress::Args::ElementSize] = m_elementSize;
    args[RtxiGetBufferElementAddress::Args::Offset]      = m_offset;
    args[RtxiGetBufferElementAddress::Args::X]           = m_x;

    if( dimensionality > 1 )
        args[RtxiGetBufferElementAddress::Args::Y] = m_y;
    if( dimensionality > 2 )
        args[RtxiGetBufferElementAddress::Args::Z] = m_z;

    args.resize( RtxiGetBufferElementAddress::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_ARGS_NON_NULL( args, "rtxiGetBufferElementAddress" );

    return m_builder.CreateCall( f, args, "bufferElementAddress" );
}

Function* RtxiGetBufferElementAddressBuilder::createFunction( const std::string& varRefUniqueName, int dimensionality )
{
    const std::string& name       = RtxiGetBufferElementAddress::getFunctionName( varRefUniqueName );
    FunctionType*      functionTy = createType( dimensionality );
    Function*          result     = dyn_cast<Function>( m_module->getOrInsertFunction( name, functionTy ) );
    result->addFnAttr( Attribute::NoUnwind );
    return result;
}

FunctionType* RtxiGetBufferElementAddressBuilder::createType( int dimensionality )
{
    std::vector<Type*> argsType( RtxiGetBufferElementAddress::Args::END );

    argsType[RtxiGetBufferElementAddress::Args::StatePtr]    = m_statePtrTy;
    argsType[RtxiGetBufferElementAddress::Args::BufferId]    = m_builder.getInt32Ty();
    argsType[RtxiGetBufferElementAddress::Args::ElementSize] = m_builder.getInt32Ty();
    argsType[RtxiGetBufferElementAddress::Args::Offset]      = m_builder.getInt64Ty();
    argsType[RtxiGetBufferElementAddress::Args::X]           = m_builder.getInt64Ty();

    if( dimensionality > 1 )
        argsType[RtxiGetBufferElementAddress::Args::Y] = m_builder.getInt64Ty();
    if( dimensionality > 2 )
        argsType[RtxiGetBufferElementAddress::Args::Z] = m_builder.getInt64Ty();

    argsType.resize( RtxiGetBufferElementAddress::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_ARG_TYPES_NON_NULL( argsType, "rtxiGetBufferElementAddress" );

    return FunctionType::get( m_builder.getInt64Ty(), argsType, false );
}

// --- RtxiAtomicSetBufferElement ------------------------------------------------------------------

const std::string RtxiAtomicSetBufferElement::PREFIX = "rtxiAtomicSetBufferElement";

Regex RtxiAtomicSetBufferElement::nameRegex = Regex( PREFIX + "\\.(" + optixi_VariableReferenceUniqueNameRegex + ")$" );

std::string RtxiAtomicSetBufferElement::getVarRefUniqueName()
{
    StringRef functionName = getCalledFunction()->getName();
    SmallVector<StringRef, 2> matches;
    bool result = nameRegex.match( functionName, &matches );
    RT_ASSERT( result && ( matches.size() == 2 ) );
    (void)result;
    return matches[1];
}

unsigned int RtxiAtomicSetBufferElement::getDimensionality() const
{
    unsigned int callArgsNumber = getCalledFunction()->arg_size();
    unsigned int dimensionality = callArgsNumber - MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );
    return dimensionality;
}

std::string RtxiAtomicSetBufferElement::getFunctionName( const std::string& varRefUniqueName )
{
    return PREFIX + "." + varRefUniqueName;
}

bool RtxiAtomicSetBufferElement::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    return calledFunction && matchName( calledFunction->getName() );
}

bool RtxiAtomicSetBufferElement::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool RtxiAtomicSetBufferElement::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

bool RtxiAtomicSetBufferElement::isIntrinsic( const Function* function, const std::string& varRefUniqueName )
{
    return optix::isIntrinsic( function->getName(), PREFIX, varRefUniqueName );
}

bool RtxiAtomicSetBufferElement::matchName( StringRef functionName )
{
    SmallVector<StringRef, 2> matches;
    return nameRegex.match( functionName, &matches ) && matches.size() == 2;
}

// --- RtxiAtomicSetBufferElementBuilder -----------------------------------------------------------

RtxiAtomicSetBufferElementBuilder::RtxiAtomicSetBufferElementBuilder( llvm::Module*      module,
                                                                      llvm::Type*        statePtrTy,
                                                                      llvm::Type*        opTy,
                                                                      llvm::Instruction* insertBefore )
    : m_module( module )
    , m_statePtrTy( statePtrTy )
    , m_opTy( opTy )
    , m_builder( insertBefore )
{
}

CallInst* RtxiAtomicSetBufferElementBuilder::createCall( const std::string& varRefUniqueName, int dimensionality )
{
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );

    Function* f = createFunction( varRefUniqueName, dimensionality );

    std::vector<Value*> args( RtxiAtomicSetBufferElement::Args::END );

    args[RtxiAtomicSetBufferElement::Args::StatePtr]       = m_statePtr;
    args[RtxiAtomicSetBufferElement::Args::BufferId]       = m_bufferId;
    args[RtxiAtomicSetBufferElement::Args::ElementSize]    = m_elementSize;
    args[RtxiAtomicSetBufferElement::Args::Offset]         = m_offset;
    args[RtxiAtomicSetBufferElement::Args::Operation]      = m_operation;
    args[RtxiAtomicSetBufferElement::Args::CompareOperand] = m_compareOperand;
    args[RtxiAtomicSetBufferElement::Args::Operand]        = m_operand;
    args[RtxiAtomicSetBufferElement::Args::SubElementType] = m_subElementType;
    args[RtxiAtomicSetBufferElement::Args::X]              = m_x;

    if( dimensionality > 1 )
        args[RtxiAtomicSetBufferElement::Args::Y] = m_y;
    if( dimensionality > 2 )
        args[RtxiAtomicSetBufferElement::Args::Z] = m_z;

    args.resize( RtxiAtomicSetBufferElement::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_ARGS_NON_NULL( args, "rtxiAtomicSetBufferElement" );

    return m_builder.CreateCall( f, args );
}

Function* RtxiAtomicSetBufferElementBuilder::createFunction( const std::string& varRefUniqueName, int dimensionality )
{
    const std::string& name       = RtxiAtomicSetBufferElement::getFunctionName( varRefUniqueName );
    FunctionType*      functionTy = createType( dimensionality );
    Function*          result     = dyn_cast<Function>( m_module->getOrInsertFunction( name, functionTy ) );
    result->addFnAttr( Attribute::NoUnwind );
    return result;
}

FunctionType* RtxiAtomicSetBufferElementBuilder::createType( int dimensionality )
{
    std::vector<Type*> argsType( RtxiAtomicSetBufferElement::Args::END );

    argsType[RtxiAtomicSetBufferElement::Args::StatePtr]       = m_statePtrTy;
    argsType[RtxiAtomicSetBufferElement::Args::BufferId]       = m_builder.getInt32Ty();
    argsType[RtxiAtomicSetBufferElement::Args::ElementSize]    = m_builder.getInt32Ty();
    argsType[RtxiAtomicSetBufferElement::Args::Offset]         = m_builder.getInt64Ty();
    argsType[RtxiAtomicSetBufferElement::Args::Operation]      = m_builder.getInt32Ty();
    argsType[RtxiAtomicSetBufferElement::Args::CompareOperand] = m_opTy;
    argsType[RtxiAtomicSetBufferElement::Args::Operand]        = m_opTy;
    argsType[RtxiAtomicSetBufferElement::Args::SubElementType] = m_builder.getInt8Ty();
    argsType[RtxiAtomicSetBufferElement::Args::X]              = m_builder.getInt64Ty();

    if( dimensionality > 1 )
        argsType[RtxiAtomicSetBufferElement::Args::Y] = m_builder.getInt64Ty();
    if( dimensionality > 2 )
        argsType[RtxiAtomicSetBufferElement::Args::Z] = m_builder.getInt64Ty();

    argsType.resize( RtxiAtomicSetBufferElement::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_ARG_TYPES_NON_NULL( argsType, "rtxiAtomicSetBufferElement" );

    return FunctionType::get( m_builder.getInt64Ty(), argsType, false );
}

// --- RtxiGetBufferSizeFromId ---------------------------------------------------------------------

const std::string RtxiGetBufferSizeFromId::PREFIX = "rtxiGetBufferSizeFromId";

Regex RtxiGetBufferSizeFromId::nameRegex = Regex( PREFIX + "$" );

std::string RtxiGetBufferSizeFromId::getFunctionName()
{
    return PREFIX;
}

bool RtxiGetBufferSizeFromId::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    return calledFunction && matchName( calledFunction->getName() );
}

bool RtxiGetBufferSizeFromId::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool RtxiGetBufferSizeFromId::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

bool RtxiGetBufferSizeFromId::matchName( StringRef functionName )
{
    SmallVector<StringRef, 2> matches;
    return nameRegex.match( functionName, &matches ) && matches.size() == 1;
}

// --- RtxiGetBufferSizeFromIdBuilder --------------------------------------------------------------

RtxiGetBufferSizeFromIdBuilder::RtxiGetBufferSizeFromIdBuilder( llvm::Module*      module,
                                                                llvm::Type*        size3Ty,
                                                                llvm::Type*        statePtrTy,
                                                                llvm::Instruction* insertBefore )
    : m_module( module )
    , m_size3Ty( size3Ty )
    , m_statePtrTy( statePtrTy )
    , m_builder( insertBefore )
{
}

CallInst* RtxiGetBufferSizeFromIdBuilder::createCall()
{
    Function* f = createFunction();

    std::vector<Value*> args( RtxiGetBufferSizeFromId::Args::END );

    args[RtxiGetBufferSizeFromId::Args::StatePtr] = m_statePtr;
    args[RtxiGetBufferSizeFromId::Args::BufferId] = m_bufferId;

    args.resize( RtxiGetBufferSizeFromId::MANDATORY_ARGS_NUMBER );

    RT_ASSERT_ARGS_NON_NULL( args, "rtxiGetBufferSizeFromId" );

    return m_builder.CreateCall( f, args, "bufferSizeFromId" );
}

Function* RtxiGetBufferSizeFromIdBuilder::createFunction()
{
    const std::string& name       = RtxiGetBufferSizeFromId::getFunctionName();
    FunctionType*      functionTy = createType();
    Function*          result     = dyn_cast<Function>( m_module->getOrInsertFunction( name, functionTy ) );
    result->addFnAttr( Attribute::NoUnwind );
    return result;
}

FunctionType* RtxiGetBufferSizeFromIdBuilder::createType()
{
    std::vector<Type*> argsType( RtxiGetBufferSizeFromId::Args::END );

    argsType[RtxiGetBufferSizeFromId::Args::StatePtr] = m_statePtrTy;
    argsType[RtxiGetBufferSizeFromId::Args::BufferId] = m_builder.getInt32Ty();

    argsType.resize( RtxiGetBufferSizeFromId::MANDATORY_ARGS_NUMBER );

    RT_ASSERT_ARG_TYPES_NON_NULL( argsType, "rtxiGetBufferSizeFromId" );

    return FunctionType::get( m_size3Ty, argsType, false );
}

// --- RtxiGetBufferElementFromId ------------------------------------------------------------------

const std::string RtxiGetBufferElementFromId::PREFIX = "rtxiGetBufferElementFromId";

Regex RtxiGetBufferElementFromId::nameRegex = Regex( PREFIX + "\\.(" + DimensionalityRegex + ")\\..+$" );

unsigned int RtxiGetBufferElementFromId::getDimensionality() const
{
    unsigned int callArgsNumber = getCalledFunction()->arg_size();
    unsigned int dimensionality = callArgsNumber - MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );
    return dimensionality;
}

std::string RtxiGetBufferElementFromId::getFunctionName( int dimensionality, const std::string& elementTypeName )
{
    return PREFIX + "." + std::to_string( dimensionality ) + "." + elementTypeName;
}

bool RtxiGetBufferElementFromId::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    return calledFunction && matchName( calledFunction->getName() );
}

bool RtxiGetBufferElementFromId::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool RtxiGetBufferElementFromId::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

bool RtxiGetBufferElementFromId::matchName( StringRef functionName )
{
    SmallVector<StringRef, 2> matches;
    return nameRegex.match( functionName, &matches ) && matches.size() == 2;
}

// --- RtxiGetBufferElementFromIdBuilder -----------------------------------------------------------

RtxiGetBufferElementFromIdBuilder::RtxiGetBufferElementFromIdBuilder( llvm::Module*      module,
                                                                      llvm::Type*        returnTy,
                                                                      llvm::Type*        statePtrTy,
                                                                      llvm::Instruction* insertBefore )
    : m_module( module )
    , m_returnTy( returnTy )
    , m_statePtrTy( statePtrTy )
    , m_builder( insertBefore )
{
}

CallInst* RtxiGetBufferElementFromIdBuilder::createCall( int dimensionality )
{
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );

    Function* f = createFunction( dimensionality );

    std::vector<Value*> args( RtxiGetBufferElementFromId::Args::END );

    args[RtxiGetBufferElementFromId::Args::StatePtr]    = m_statePtr;
    args[RtxiGetBufferElementFromId::Args::BufferId]    = m_bufferId;
    args[RtxiGetBufferElementFromId::Args::ElementSize] = m_elementSize;
    args[RtxiGetBufferElementFromId::Args::Offset]      = m_offset;
    args[RtxiGetBufferElementFromId::Args::X]           = m_x;

    if( dimensionality > 1 )
        args[RtxiGetBufferElementFromId::Args::Y] = m_y;
    if( dimensionality > 2 )
        args[RtxiGetBufferElementFromId::Args::Z] = m_z;

    args.resize( RtxiGetBufferElementFromId::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_ARGS_NON_NULL( args, "rtxiGetBufferElementFromId" );

    return m_builder.CreateCall( f, args, "bufferElementFromId" );
}

Function* RtxiGetBufferElementFromIdBuilder::createFunction( int dimensionality )
{
    const std::string  elementTypeName = corelib::getTypeName( m_returnTy );
    const std::string& name            = RtxiGetBufferElementFromId::getFunctionName( dimensionality, elementTypeName );
    FunctionType*      functionTy      = createType( dimensionality );
    Function*          result          = dyn_cast<Function>( m_module->getOrInsertFunction( name, functionTy ) );
    result->addFnAttr( Attribute::NoUnwind );
    return result;
}

FunctionType* RtxiGetBufferElementFromIdBuilder::createType( int dimensionality )
{
    std::vector<Type*> argsType( RtxiGetBufferElementFromId::Args::END );

    argsType[RtxiGetBufferElementFromId::Args::StatePtr]    = m_statePtrTy;
    argsType[RtxiGetBufferElementFromId::Args::BufferId]    = m_builder.getInt32Ty();
    argsType[RtxiGetBufferElementFromId::Args::ElementSize] = m_builder.getInt32Ty();
    argsType[RtxiGetBufferElementFromId::Args::Offset]      = m_builder.getInt64Ty();
    argsType[RtxiGetBufferElementFromId::Args::X]           = m_builder.getInt64Ty();

    if( dimensionality > 1 )
        argsType[RtxiGetBufferElementFromId::Args::Y] = m_builder.getInt64Ty();
    if( dimensionality > 2 )
        argsType[RtxiGetBufferElementFromId::Args::Z] = m_builder.getInt64Ty();

    argsType.resize( RtxiGetBufferElementFromId::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_ARG_TYPES_NON_NULL( argsType, "rtxiGetBufferElementFromId" );

    return FunctionType::get( m_returnTy, argsType, false );
}

// --- RtxiSetBufferElementFromId ------------------------------------------------------------------

const std::string RtxiSetBufferElementFromId::PREFIX = "rtxiSetBufferElementFromId";

Regex RtxiSetBufferElementFromId::nameRegex = Regex( PREFIX + "\\.(" + DimensionalityRegex + ")\\..+$" );

unsigned int RtxiSetBufferElementFromId::getDimensionality() const
{
    unsigned int callArgsNumber = getCalledFunction()->arg_size();
    unsigned int dimensionality = callArgsNumber - MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );
    return dimensionality;
}

std::string RtxiSetBufferElementFromId::getFunctionName( int dimensionality, const std::string& elementTypeName )
{
    return PREFIX + "." + std::to_string( dimensionality ) + "." + elementTypeName;
}

bool RtxiSetBufferElementFromId::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    return calledFunction && matchName( calledFunction->getName() );
}

bool RtxiSetBufferElementFromId::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool RtxiSetBufferElementFromId::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

bool RtxiSetBufferElementFromId::matchName( StringRef functionName )
{
    SmallVector<StringRef, 2> matches;
    return nameRegex.match( functionName, &matches ) && matches.size() == 2;
}

// --- RtxiSetBufferElementFromIdBuilder -----------------------------------------------------------

RtxiSetBufferElementFromIdBuilder::RtxiSetBufferElementFromIdBuilder( llvm::Module*      module,
                                                                      llvm::Type*        statePtrTy,
                                                                      llvm::Type*        valueToSetTy,
                                                                      llvm::Instruction* insertBefore )
    : m_module( module )
    , m_statePtrTy( statePtrTy )
    , m_valueToSetTy( valueToSetTy )
    , m_builder( insertBefore )
{
}

CallInst* RtxiSetBufferElementFromIdBuilder::createCall( int dimensionality )
{
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );

    Function* f = createFunction( dimensionality );

    std::vector<Value*> args( RtxiSetBufferElementFromId::Args::END );

    args[RtxiSetBufferElementFromId::Args::StatePtr]    = m_statePtr;
    args[RtxiSetBufferElementFromId::Args::BufferId]    = m_bufferId;
    args[RtxiSetBufferElementFromId::Args::ElementSize] = m_elementSize;
    args[RtxiSetBufferElementFromId::Args::Offset]      = m_offset;
    args[RtxiSetBufferElementFromId::Args::X]           = m_x;

    if( dimensionality > 1 )
        args[RtxiSetBufferElementFromId::Args::Y] = m_y;
    if( dimensionality > 2 )
        args[RtxiSetBufferElementFromId::Args::Z] = m_z;

    args[RtxiSetBufferElementFromId::Args::Offset + dimensionality + 1] = m_valueToSet;

    args.resize( RtxiSetBufferElementFromId::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_ARGS_NON_NULL( args, "rtxiSetBufferElementFromId" );

    return m_builder.CreateCall( f, args );
}

Function* RtxiSetBufferElementFromIdBuilder::createFunction( int dimensionality )
{
    const std::string  elementTypeName = corelib::getTypeName( m_valueToSetTy );
    const std::string& name            = RtxiSetBufferElementFromId::getFunctionName( dimensionality, elementTypeName );
    FunctionType*      functionTy      = createType( dimensionality );
    Function*          result          = dyn_cast<Function>( m_module->getOrInsertFunction( name, functionTy ) );
    result->addFnAttr( Attribute::NoUnwind );
    return result;
}

FunctionType* RtxiSetBufferElementFromIdBuilder::createType( int dimensionality )
{
    std::vector<Type*> argsType( RtxiSetBufferElementFromId::Args::END );

    argsType[RtxiSetBufferElementFromId::Args::StatePtr]    = m_statePtrTy;
    argsType[RtxiSetBufferElementFromId::Args::BufferId]    = m_builder.getInt32Ty();
    argsType[RtxiSetBufferElementFromId::Args::ElementSize] = m_builder.getInt32Ty();
    argsType[RtxiSetBufferElementFromId::Args::Offset]      = m_builder.getInt64Ty();
    argsType[RtxiSetBufferElementFromId::Args::X]           = m_builder.getInt64Ty();

    if( dimensionality > 1 )
        argsType[RtxiSetBufferElementFromId::Args::Y] = m_builder.getInt64Ty();
    if( dimensionality > 2 )
        argsType[RtxiSetBufferElementFromId::Args::Z] = m_builder.getInt64Ty();

    argsType[RtxiSetBufferElementFromId::Args::Offset + dimensionality + 1] = m_valueToSetTy;

    argsType.resize( RtxiSetBufferElementFromId::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_ARG_TYPES_NON_NULL( argsType, "rtxiSetBufferElementFromId" );

    return FunctionType::get( m_builder.getVoidTy(), argsType, false );
}

// --- RtxiGetBufferElementAddressFromId -----------------------------------------------------------

const std::string RtxiGetBufferElementAddressFromId::PREFIX = "rtxiGetBufferElementAddressFromId";

Regex RtxiGetBufferElementAddressFromId::nameRegex = Regex( PREFIX + "\\.(" + DimensionalityRegex + ")$" );

unsigned int RtxiGetBufferElementAddressFromId::getDimensionality() const
{
    unsigned int callArgsNumber = getCalledFunction()->arg_size();
    unsigned int dimensionality = callArgsNumber - MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );
    return dimensionality;
}

std::string RtxiGetBufferElementAddressFromId::getFunctionName( int dimensionality )
{
    return PREFIX + "." + std::to_string( dimensionality );
}

bool RtxiGetBufferElementAddressFromId::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    return calledFunction && matchName( calledFunction->getName() );
}

bool RtxiGetBufferElementAddressFromId::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool RtxiGetBufferElementAddressFromId::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

bool RtxiGetBufferElementAddressFromId::matchName( StringRef functionName )
{
    SmallVector<StringRef, 2> matches;
    return nameRegex.match( functionName, &matches ) && matches.size() == 2;
}

// --- RtxiGetBufferElementAddressFromIdBuilder ----------------------------------------------------

RtxiGetBufferElementAddressFromIdBuilder::RtxiGetBufferElementAddressFromIdBuilder( llvm::Module*      module,
                                                                                    llvm::Type*        statePtrTy,
                                                                                    llvm::Instruction* insertBefore )
    : m_module( module )
    , m_statePtrTy( statePtrTy )
    , m_builder( insertBefore )
{
}

CallInst* RtxiGetBufferElementAddressFromIdBuilder::createCall( int dimensionality )
{
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );

    Function* f = createFunction( dimensionality );

    std::vector<Value*> args( RtxiGetBufferElementAddressFromId::Args::END );

    args[RtxiGetBufferElementAddressFromId::Args::StatePtr]    = m_statePtr;
    args[RtxiGetBufferElementAddressFromId::Args::BufferId]    = m_bufferId;
    args[RtxiGetBufferElementAddressFromId::Args::ElementSize] = m_elementSize;
    args[RtxiGetBufferElementAddressFromId::Args::X]           = m_x;

    if( dimensionality > 1 )
        args[RtxiGetBufferElementAddressFromId::Args::Y] = m_y;
    if( dimensionality > 2 )
        args[RtxiGetBufferElementAddressFromId::Args::Z] = m_z;

    args.resize( RtxiGetBufferElementAddressFromId::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_ARGS_NON_NULL( args, "rtxiGetBufferElementAddressFromId" );

    return m_builder.CreateCall( f, args, "bufferElementAddressFromId" );
}

Function* RtxiGetBufferElementAddressFromIdBuilder::createFunction( int dimensionality )
{
    const std::string& name       = RtxiGetBufferElementAddressFromId::getFunctionName( dimensionality );
    FunctionType*      functionTy = createType( dimensionality );
    Function*          result     = dyn_cast<Function>( m_module->getOrInsertFunction( name, functionTy ) );
    result->addFnAttr( Attribute::NoUnwind );
    return result;
}

FunctionType* RtxiGetBufferElementAddressFromIdBuilder::createType( int dimensionality )
{
    std::vector<Type*> argsType( RtxiGetBufferElementAddressFromId::Args::END );

    argsType[RtxiGetBufferElementAddressFromId::Args::StatePtr]    = m_statePtrTy;
    argsType[RtxiGetBufferElementAddressFromId::Args::BufferId]    = m_builder.getInt32Ty();
    argsType[RtxiGetBufferElementAddressFromId::Args::ElementSize] = m_builder.getInt32Ty();
    argsType[RtxiGetBufferElementAddressFromId::Args::X]           = m_builder.getInt64Ty();

    if( dimensionality > 1 )
        argsType[RtxiGetBufferElementAddressFromId::Args::Y] = m_builder.getInt64Ty();
    if( dimensionality > 2 )
        argsType[RtxiGetBufferElementAddressFromId::Args::Z] = m_builder.getInt64Ty();

    argsType.resize( RtxiGetBufferElementAddressFromId::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_ARG_TYPES_NON_NULL( argsType, "rtxiGetBufferElementAddressFromId" );

    return FunctionType::get( m_builder.getInt64Ty(), argsType, false );
}

// --- RtxiAtomicSetBufferElementFromId ------------------------------------------------------------

const std::string RtxiAtomicSetBufferElementFromId::PREFIX = "rtxiAtomicSetBufferElementFromId";

Regex RtxiAtomicSetBufferElementFromId::nameRegex = Regex( PREFIX + "\\.(" + DimensionalityRegex + ")$" );

unsigned int RtxiAtomicSetBufferElementFromId::getDimensionality() const
{
    unsigned int callArgsNumber = getCalledFunction()->arg_size();
    unsigned int dimensionality = callArgsNumber - MANDATORY_ARGS_NUMBER + 1;
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );
    return dimensionality;
}

std::string RtxiAtomicSetBufferElementFromId::getFunctionName( int dimensionality )
{
    return PREFIX + "." + std::to_string( dimensionality );
}

bool RtxiAtomicSetBufferElementFromId::classof( const CallInst* inst )
{
    const Function* calledFunction = inst->getCalledFunction();
    return calledFunction && matchName( calledFunction->getName() );
}

bool RtxiAtomicSetBufferElementFromId::classof( const Value* value )
{
    return isa<CallInst>( value ) && classof( cast<CallInst>( value ) );
}

bool RtxiAtomicSetBufferElementFromId::isIntrinsic( const Function* function )
{
    return matchName( function->getName() );
}

bool RtxiAtomicSetBufferElementFromId::matchName( StringRef functionName )
{
    SmallVector<StringRef, 2> matches;
    return nameRegex.match( functionName, &matches ) && matches.size() == 2;
}

// --- RtxiAtomicSetBufferElementFromIdBuilder -----------------------------------------------------

RtxiAtomicSetBufferElementFromIdBuilder::RtxiAtomicSetBufferElementFromIdBuilder( llvm::Module*      module,
                                                                                  llvm::Type*        statePtrTy,
                                                                                  llvm::Type*        opTy,
                                                                                  llvm::Instruction* insertBefore )
    : m_module( module )
    , m_statePtrTy( statePtrTy )
    , m_opTy( opTy )
    , m_builder( insertBefore )
{
}

CallInst* RtxiAtomicSetBufferElementFromIdBuilder::createCall( int dimensionality )
{
    RT_ASSERT( 1 <= dimensionality && dimensionality <= 3 );

    Function* f = createFunction( dimensionality );

    std::vector<Value*> args( RtxiAtomicSetBufferElementFromId::Args::END );

    args[RtxiAtomicSetBufferElementFromId::Args::StatePtr]       = m_statePtr;
    args[RtxiAtomicSetBufferElementFromId::Args::BufferId]       = m_bufferId;
    args[RtxiAtomicSetBufferElementFromId::Args::ElementSize]    = m_elementSize;
    args[RtxiAtomicSetBufferElementFromId::Args::Offset]         = m_offset;
    args[RtxiAtomicSetBufferElementFromId::Args::Operation]      = m_operation;
    args[RtxiAtomicSetBufferElementFromId::Args::CompareOperand] = m_compareOperand;
    args[RtxiAtomicSetBufferElementFromId::Args::Operand]        = m_operand;
    args[RtxiAtomicSetBufferElementFromId::Args::SubElementType] = m_subElementType;
    args[RtxiAtomicSetBufferElementFromId::Args::X]              = m_x;

    if( dimensionality > 1 )
        args[RtxiAtomicSetBufferElementFromId::Args::Y] = m_y;
    if( dimensionality > 2 )
        args[RtxiAtomicSetBufferElementFromId::Args::Z] = m_z;

    args.resize( RtxiAtomicSetBufferElementFromId::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_ARGS_NON_NULL( args, "rtxiAtomicSetBufferElement" );

    return m_builder.CreateCall( f, args );
}

Function* RtxiAtomicSetBufferElementFromIdBuilder::createFunction( int dimensionality )
{
    const std::string& name       = RtxiAtomicSetBufferElementFromId::getFunctionName( dimensionality );
    FunctionType*      functionTy = createType( dimensionality );
    Function*          result     = dyn_cast<Function>( m_module->getOrInsertFunction( name, functionTy ) );
    result->addFnAttr( Attribute::NoUnwind );
    return result;
}

FunctionType* RtxiAtomicSetBufferElementFromIdBuilder::createType( int dimensionality )
{
    std::vector<Type*> argsType( RtxiAtomicSetBufferElementFromId::Args::END );

    argsType[RtxiAtomicSetBufferElementFromId::Args::StatePtr]       = m_statePtrTy;
    argsType[RtxiAtomicSetBufferElementFromId::Args::BufferId]       = m_builder.getInt32Ty();
    argsType[RtxiAtomicSetBufferElementFromId::Args::ElementSize]    = m_builder.getInt32Ty();
    argsType[RtxiAtomicSetBufferElementFromId::Args::Offset]         = m_builder.getInt64Ty();
    argsType[RtxiAtomicSetBufferElementFromId::Args::Operation]      = m_builder.getInt32Ty();
    argsType[RtxiAtomicSetBufferElementFromId::Args::CompareOperand] = m_opTy;
    argsType[RtxiAtomicSetBufferElementFromId::Args::Operand]        = m_opTy;
    argsType[RtxiAtomicSetBufferElementFromId::Args::SubElementType] = m_builder.getInt8Ty();
    argsType[RtxiAtomicSetBufferElementFromId::Args::X]              = m_builder.getInt64Ty();

    if( dimensionality > 1 )
        argsType[RtxiAtomicSetBufferElementFromId::Args::Y] = m_builder.getInt64Ty();
    if( dimensionality > 2 )
        argsType[RtxiAtomicSetBufferElementFromId::Args::Z] = m_builder.getInt64Ty();

    argsType.resize( RtxiAtomicSetBufferElementFromId::MANDATORY_ARGS_NUMBER + dimensionality - 1 );

    RT_ASSERT_ARG_TYPES_NON_NULL( argsType, "rtxiAtomicSetBufferElement" );

    return FunctionType::get( m_builder.getInt64Ty(), argsType, false );
}

}  // namespace optix
