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

#include <corelib/compiler/LLVMUtil.h>

#include <prodlib/exceptions/CompileError.h>
#include <prodlib/system/Knobs.h>

#include <Context/Context.h>
#include <Context/LLVMManager.h>
#include <ExelwtionStrategy/Compile.h>
#include <ExelwtionStrategy/RTX/RTXExceptionInstrumenter.h>
#include <ExelwtionStrategy/RTX/RTXIntrinsics.h>
#include <FrontEnd/Canonical/FrontEndHelpers.h>
#include <FrontEnd/Canonical/IntrinsicsManager.h>
#include <FrontEnd/Canonical/LineInfo.h>
#include <Util/ContainerAlgorithm.h>

using namespace corelib;
using namespace llvm;

namespace {
// clang-format off
// Knob precedence (in decreasing priority, default value counts as unset):
// - enable individuals
// - enable all
// - disable all
PublicKnob<bool> k_enableAllExceptions( RT_PUBLIC_DSTRING("compile.enableAllExceptions"), false, RT_PUBLIC_DSTRING("Force creation of exception handling code for all exceptions that we support." ) );
PublicKnob<bool> k_disableAllExceptions( RT_PUBLIC_DSTRING("compile.disableAllExceptions"), false, RT_PUBLIC_DSTRING("Do not create code to handle any exceptions." ) );
Knob<bool> k_enableIlwalidBufferIdHandling( RT_DSTRING("compile.enableIlwalidBufferIdHandling"), false, RT_DSTRING("Create code to handle invalid buffer id exceptions." ) );
Knob<bool> k_enableIlwalidTextureIdHandling( RT_DSTRING("compile.enableIlwalidTextureIdHandling"), false, RT_DSTRING("Create code to handle invalid texture id exceptions." ) );
Knob<bool> k_enableIlwalidProgramIdHandling( RT_DSTRING("compile.enableIlwalidProgramIdHandling"), false, RT_DSTRING("Create code to handle invalid program id exceptions." ) );
Knob<bool> k_enableBufferIndexOutOfBoundsHandling( RT_DSTRING("compile.enableBufferIndexOutOfBoundsHandling"), false, RT_DSTRING("Create code to handle buffer index out of bounds exceptions." ) );
Knob<bool> k_enableIndexOutOfBoundsHandling( RT_DSTRING("compile.enableIndexOutOfBoundsHandling"), false, RT_DSTRING("Create code to handle index out of bounds exceptions." ) );
Knob<bool> k_enableIlwalidRayHandling( RT_DSTRING("compile.enableIlwalidRayHandling"), false, RT_DSTRING("Create code to handle invalid ray exceptions." ) );
Knob<bool> k_enablePayloadAccessOutOfBoundsHandling( RT_DSTRING("compile.enablePayloadAccessOutOfBoundsHandling"), false, RT_DSTRING("Create code to handle payload offset out of bounds exceptions." ) );

Knob<std::string> k_saveLLVM( RT_DSTRING("rtx.saveLLVM"), "", RT_DSTRING( "Save LLVM stages during compilation" ) );
// clang-format on
}

namespace optix {

RTXExceptionInstrumenter::RTXExceptionInstrumenter( SemanticType stype,
                                                    uint64_t     exceptionFlags,
                                                    uint64_t     maxPayloadSize,
                                                    bool         payloadInRegisters,
                                                    int          launchCounterForDebugging )
    : m_entryFunction( nullptr )
    , m_module( nullptr )
    , m_stype( stype )
    , m_exceptionFlags( exceptionFlags )
    , m_maxPayloadSize( maxPayloadSize )
    , m_payloadInRegisters( payloadInRegisters )
    , m_launchCounterForDebugging( launchCounterForDebugging )
{
}

void RTXExceptionInstrumenter::runOnFunction( Function* entryFunction )
{
    m_entryFunction = entryFunction;
    m_module        = m_entryFunction->getParent();
    initializeRuntimeFunctions();

    int dumpId = 0;

    // Add semantic type to dumped file names for consistency with rtcore and to disambiguate
    // different uses of the null program.
    std::string semanticType = optix::semanticTypeToString( m_stype );
    algorithm::transform( semanticType, semanticType.begin(), ::tolower );
    std::string dumpFunctionName = "_" + semanticType + "__" + std::string( entryFunction->getName() );

    dump( dumpFunctionName, dumpId++, "init" );

    // Instrument code for the buffer-related exceptions.
    //
    // Iterate over functions and transform calls to optixi buffer intrinsics into corresponding RTX
    // intrinsics. This needs to happen even if all exceptions are disabled.
    std::vector<Function*> toRemove;
    for( Function& F : *m_module )
    {
        if( !F.isDeclaration() )
            continue;

        bool transformed = true;

        if( F.getName().startswith( GET_BUFFER_SIZE + "." ) )
            transformGetBufferSize( &F );
        else if( GetBufferElement::isIntrinsic( &F ) )
            transformGetBufferElement( &F );
        else if( SetBufferElement::isIntrinsic( &F ) )
            transformSetBufferElement( &F );
        else if( GetBufferElementAddress::isIntrinsic( &F ) )
            transformGetBufferElementAddress( &F );
        else if( AtomicSetBufferElement::isIntrinsic( &F ) )
            transformAtomicSetBufferElement( &F );
        else if( LoadOrRequestBufferElement::isIntrinsic( &F ) )
            transformLoadOrRequestBufferElement( &F );
        else if( LoadOrRequestTextureElement::isIntrinsic( &F ) )
            transformLoadOrRequestTextureElement( &F );
        else if( F.getName().startswith( GET_BUFFER_SIZE_ID + "." ) )
            transformGetBufferSizeFromId( &F );
        else if( GetBufferElementFromId::isIntrinsic( &F ) )
            transformGetBufferElementFromId( &F );
        else if( SetBufferElementFromId::isIntrinsic( &F ) )
            transformSetBufferElementFromId( &F );
        else if( GetBufferElementAddressFromId::isIntrinsic( &F ) )
            transformGetBufferElementAddressFromId( &F );
        else if( AtomicSetBufferElementFromId::isIntrinsic( &F ) )
            transformAtomicSetBufferElementFromId( &F );

        else
            transformed = false;

        if( transformed )
            toRemove.push_back( &F );
    }

    for( Function* F : toRemove )
        F->eraseFromParent();

    // Instrument code for remaining exceptions.
    if( textureIdIlwalidEnabled() )
        insertTextureIdIlwalidCheck();
    if( programIdIlwalidEnabled() )
        insertProgramIdIlwalidCheck();
    if( ilwalidRayEnabled() )
        insertIlwalidRayCheck();
    if( indexOutOfBoundsEnabled() )
        insertIndexOutOfBoundsCheck();
    if( payloadOffsetOutOfBoundsEnabled() )
        insertPayloadAccessOutOfBoundsCheck();

    // This check cannot be disabled (same for user exceptions).
    insertExceptionCodeOutOfBoundsCheck();

    dump( dumpFunctionName, dumpId++, "after_instrumentation" );
}

// Relwrsively look for type of the given name
static Type* findTypeWithPrefix( Type* t, const std::string& prefix )
{
    while( isa<PointerType>( t ) )
        t          = t->getPointerElementType();
    StructType* st = dyn_cast<StructType>( t );
    if( !st )
        return nullptr;
    if( st->getName() == prefix || st->getName().startswith( prefix + "." ) )
        return st;
    for( StructType::element_iterator b = st->element_begin(), e = st->element_end(); b != e; ++b )
    {
        Type* elt = *b;
        if( Type* ret = findTypeWithPrefix( elt, prefix ) )
            return ret;
    }
    return nullptr;
}

static Type* findType( Type* t, const std::string& name )
{
    return findTypeWithPrefix( t, "struct.cort::" + name );
}

void RTXExceptionInstrumenter::initializeRuntimeFunctions()
{
    Module*      module      = m_entryFunction->getParent();
    LLVMContext& llvmContext = module->getContext();
    Type*        statePtrTy  = m_entryFunction->getFunctionType()->getParamType( 0 );
    Type*        i32Ty       = Type::getInt32Ty( llvmContext );
    Type*        i64Ty       = Type::getInt64Ty( llvmContext );
    Type*        f32Ty       = Type::getFloatTy( llvmContext );

    m_checkTextureIdFunc =
        findOrCreateRuntimeFunction( m_module, "_ZN4cort31Exception_checkTextureIdIlwalidEPNS_14CanonicalStateEj",
                                     i32Ty, {statePtrTy, i32Ty} );

    m_checkBufferIdFunc =
        findOrCreateRuntimeFunction( m_module, "_ZN4cort30Exception_checkBufferIdIlwalidEPNS_14CanonicalStateEj", i32Ty,
                                     {statePtrTy, i32Ty} );

    m_checkProgramIdFunc =
        findOrCreateRuntimeFunction( m_module, "_ZN4cort31Exception_checkProgramIdIlwalidEPNS_14CanonicalStateEj",
                                     i32Ty, {statePtrTy, i32Ty} );

    m_getGeometryInstanceFunc = findOrCreateRuntimeFunction( m_module, "_ZN4cort25getGeometryInstanceHandleEv", i32Ty, {} );

    m_getNumMaterialsFunc =
        findOrCreateRuntimeFunction( m_module, "_ZN4cort32GeometryInstance_getNumMaterialsEPNS_14CanonicalStateEj",
                                     i32Ty, {statePtrTy, i32Ty} );

    m_getNumMaterialsFunc =
        findOrCreateRuntimeFunction( m_module, "_ZN4cort32GeometryInstance_getNumMaterialsEPNS_14CanonicalStateEj",
                                     i32Ty, {statePtrTy, i32Ty} );

    Type* voidTy         = Type::getVoidTy( llvmContext );
    Type* i32_23Ty       = ArrayType::get( i32Ty, 23 );
    m_throwExceptionFunc = findOrCreateRuntimeFunction( m_module, "lw.rt.throw.exception", voidTy, {i32Ty, i32_23Ty} );

    m_throwBufferIndexOutOfBoundsException =
        findOrCreateRuntimeFunction( m_module, "RTX_throwBufferIndexOutOfBoundsException", voidTy,
                                     {i32Ty, i64Ty, i64Ty, i64Ty, i64Ty, i32Ty, i32Ty, i32Ty} );
    m_throwExceptionCodeOutOfBoundsException =
        findOrCreateRuntimeFunction( m_module, "RTX_throwExceptionCodeOutOfBoundsException", voidTy,
                                     {i32Ty, i64Ty, i32Ty, i32Ty, i32Ty} );
    m_throwIlwalidIdException =
        findOrCreateRuntimeFunction( m_module, "RTX_throwIlwalidIdException", voidTy, {i32Ty, i64Ty, i32Ty, i32Ty} );
    m_throwIlwalidRayException =
        findOrCreateRuntimeFunction( m_module, "RTX_throwIlwalidRayException", voidTy,
                                     {i32Ty, i64Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, f32Ty, i32Ty, f32Ty, f32Ty} );
    m_throwMaterialIndexOutOfBoundsException =
        findOrCreateRuntimeFunction( m_module, "RTX_throwMaterialIndexOutOfBoundsException", voidTy, {i32Ty, i64Ty, i64Ty, i64Ty} );
    m_throwPayloadAccessOutOfBoundsException =
        findOrCreateRuntimeFunction( m_module, "RTX_throwPayloadAccessOutOfBoundsException", voidTy,
                                     {i32Ty, i64Ty, i64Ty, i64Ty, i64Ty, i64Ty} );
}

void RTXExceptionInstrumenter::transformGetBufferSize( Function* F )
{
    std::vector<CallInst*> toRemove;
    for( CallInst* CI : getCallsToFunction( F ) )
    {
        StringRef variableName;
        bool      result = parseBufferSizeName( F->getName(), variableName );
        RT_ASSERT( result );
        (void)result;

        Value* statePtr = CI->getArgOperand( 0 );

        RtxiGetBufferIdBuilder builder1( m_module, statePtr->getType(), CI );
        builder1.setStatePtr( statePtr );
        CallInst* bufferId = builder1.createCall( variableName );

        if( bufferIdIlwalidEnabled() )
            insertBufferIdIlwalidCheck( statePtr, bufferId, CI );

        RtxiGetBufferSizeBuilder builder2( m_module, F->getReturnType(), statePtr->getType(), CI );
        builder2.setStatePtr( statePtr );
        builder2.setBufferId( bufferId );
        CallInst* element = builder2.createCall( variableName );
        element->takeName( CI );

        CI->replaceAllUsesWith( element );
        toRemove.push_back( CI );
    }

    for( CallInst* CI : toRemove )
        CI->eraseFromParent();
}

void RTXExceptionInstrumenter::transformGetBufferElement( Function* F )
{
    std::vector<CallInst*> toRemove;
    for( CallInst* CI : getCallsToFunction( F ) )
    {
        GetBufferElement*  getBufferElement = cast<GetBufferElement>( CI );
        const std::string& variableName     = getBufferElement->parseUniqueName();
        int                dimensionality   = getBufferElement->getDimensionality();

        Value* statePtr    = getBufferElement->getStatePtr();
        Value* elementSize = getBufferElement->getElementSize();
        Value* offset      = getBufferElement->getOffset();
        Value* x           = getBufferElement->getX();
        Value* y           = dimensionality > 1 ? getBufferElement->getY() : nullptr;
        Value* z           = dimensionality > 2 ? getBufferElement->getZ() : nullptr;

        RtxiGetBufferIdBuilder builder1( m_module, statePtr->getType(), CI );
        builder1.setStatePtr( statePtr );
        CallInst* bufferId = builder1.createCall( variableName );

        if( bufferIdIlwalidEnabled() )
            insertBufferIdIlwalidCheck( statePtr, bufferId, CI );

        if( bufferIndexOutOfBoundsEnabled() )
        {
            Value* indices[3] = {x, y, z};
            insertBufferIndexOutOfBoundsCheck( statePtr, bufferId, dimensionality, elementSize, indices, variableName, CI );
        }

        RtxiGetBufferElementBuilder builder2( m_module, F->getReturnType(), statePtr->getType(), CI );
        builder2.setStatePtr( statePtr );
        builder2.setBufferId( bufferId );
        builder2.setElementSize( elementSize );
        builder2.setOffset( offset );
        builder2.setX( x );
        builder2.setY( y );
        builder2.setZ( z );
        CallInst* element = builder2.createCall( variableName, dimensionality );
        element->takeName( CI );

        CI->replaceAllUsesWith( element );
        toRemove.push_back( CI );
    }

    for( CallInst* CI : toRemove )
        CI->eraseFromParent();
}

void RTXExceptionInstrumenter::transformSetBufferElement( Function* F )
{
    std::vector<CallInst*> toRemove;
    for( CallInst* CI : getCallsToFunction( F ) )
    {
        SetBufferElement*  setBufferElement = cast<SetBufferElement>( CI );
        const std::string& variableName     = setBufferElement->parseUniqueName();
        int                dimensionality   = setBufferElement->getDimensionality();

        Value* statePtr    = setBufferElement->getStatePtr();
        Value* elementSize = setBufferElement->getElementSize();
        Value* offset      = setBufferElement->getOffset();
        Value* x           = setBufferElement->getX();
        Value* y           = dimensionality > 1 ? setBufferElement->getY() : nullptr;
        Value* z           = dimensionality > 2 ? setBufferElement->getZ() : nullptr;
        Value* valueToSet  = setBufferElement->getValueToSet();

        RtxiGetBufferIdBuilder builder1( m_module, statePtr->getType(), CI );
        builder1.setStatePtr( statePtr );
        CallInst* bufferId = builder1.createCall( variableName );

        if( bufferIdIlwalidEnabled() )
            insertBufferIdIlwalidCheck( statePtr, bufferId, CI );

        if( bufferIndexOutOfBoundsEnabled() )
        {
            Value* indices[3] = {x, y, z};
            insertBufferIndexOutOfBoundsCheck( statePtr, bufferId, dimensionality, elementSize, indices, variableName, CI );
        }

        RtxiSetBufferElementBuilder builder2( m_module, statePtr->getType(), valueToSet->getType(), CI );
        builder2.setStatePtr( statePtr );
        builder2.setBufferId( bufferId );
        builder2.setElementSize( elementSize );
        builder2.setOffset( offset );
        builder2.setX( x );
        builder2.setY( y );
        builder2.setZ( z );
        builder2.setValueToSet( valueToSet );
        CallInst* element = builder2.createCall( variableName, dimensionality );
        element->takeName( CI );

        CI->replaceAllUsesWith( element );
        toRemove.push_back( CI );
    }

    for( CallInst* CI : toRemove )
        CI->eraseFromParent();
}

void RTXExceptionInstrumenter::transformGetBufferElementAddress( Function* F )
{
    std::vector<CallInst*> toRemove;
    for( CallInst* CI : getCallsToFunction( F ) )
    {
        GetBufferElementAddress* getBufferElementAddress = cast<GetBufferElementAddress>( CI );
        const std::string&       variableName            = getBufferElementAddress->parseUniqueName();
        int                      dimensionality          = getBufferElementAddress->getDimensionality();

        Value* statePtr    = getBufferElementAddress->getStatePtr();
        Value* elementSize = getBufferElementAddress->getElementSize();
        Value* offset      = getBufferElementAddress->getOffset();
        Value* x           = getBufferElementAddress->getX();
        Value* y           = dimensionality > 1 ? getBufferElementAddress->getY() : nullptr;
        Value* z           = dimensionality > 2 ? getBufferElementAddress->getZ() : nullptr;

        RtxiGetBufferIdBuilder builder1( m_module, statePtr->getType(), CI );
        builder1.setStatePtr( statePtr );
        CallInst* bufferId = builder1.createCall( variableName );

        if( bufferIdIlwalidEnabled() )
            insertBufferIdIlwalidCheck( statePtr, bufferId, CI );

        if( bufferIndexOutOfBoundsEnabled() )
        {
            Value* indices[3] = {x, y, z};
            insertBufferIndexOutOfBoundsCheck( statePtr, bufferId, dimensionality, elementSize, indices, variableName, CI );
        }

        RtxiGetBufferElementAddressBuilder builder2( m_module, statePtr->getType(), CI );
        builder2.setStatePtr( statePtr );
        builder2.setBufferId( bufferId );
        builder2.setElementSize( elementSize );
        builder2.setOffset( offset );
        builder2.setX( x );
        builder2.setY( y );
        builder2.setZ( z );
        CallInst* element = builder2.createCall( variableName, dimensionality );
        element->takeName( CI );

        CI->replaceAllUsesWith( element );
        toRemove.push_back( CI );
    }

    for( CallInst* CI : toRemove )
        CI->eraseFromParent();
}

void RTXExceptionInstrumenter::transformAtomicSetBufferElement( Function* F )
{
    std::vector<CallInst*> toRemove;
    for( CallInst* CI : getCallsToFunction( F ) )
    {
        AtomicSetBufferElement* atomicSetBufferElement = cast<AtomicSetBufferElement>( CI );
        const std::string&      variableName           = atomicSetBufferElement->parseUniqueName();
        int                     dimensionality         = atomicSetBufferElement->getDimensionality();

        Value* statePtr       = atomicSetBufferElement->getStatePtr();
        Value* elementSize    = atomicSetBufferElement->getElementSize();
        Value* offset         = atomicSetBufferElement->getOffset();
        Value* operation      = atomicSetBufferElement->getOperation();
        Value* compareOperand = atomicSetBufferElement->getCompareOperand();
        Value* operand        = atomicSetBufferElement->getOperand();
        Value* subElementType = atomicSetBufferElement->getSubElementType();
        Value* x              = atomicSetBufferElement->getX();
        Value* y              = dimensionality > 1 ? atomicSetBufferElement->getY() : nullptr;
        Value* z              = dimensionality > 2 ? atomicSetBufferElement->getZ() : nullptr;

        RtxiGetBufferIdBuilder builder1( m_module, statePtr->getType(), CI );
        builder1.setStatePtr( statePtr );
        CallInst* bufferId = builder1.createCall( variableName );

        if( bufferIdIlwalidEnabled() )
            insertBufferIdIlwalidCheck( statePtr, bufferId, CI );

        if( bufferIndexOutOfBoundsEnabled() )
        {
            Value* indices[3] = {x, y, z};
            insertBufferIndexOutOfBoundsCheck( statePtr, bufferId, dimensionality, elementSize, indices, variableName, CI );
        }

        RtxiAtomicSetBufferElementBuilder builder2( m_module, statePtr->getType(), operand->getType(), CI );
        builder2.setStatePtr( statePtr );
        builder2.setBufferId( bufferId );
        builder2.setElementSize( elementSize );
        builder2.setOffset( offset );
        builder2.setOperation( operation );
        builder2.setCompareOperand( compareOperand );
        builder2.setOperand( operand );
        builder2.setSubElementType( subElementType );
        builder2.setX( x );
        builder2.setY( y );
        builder2.setZ( z );
        CallInst* element = builder2.createCall( variableName, dimensionality );
        element->takeName( CI );

        CI->replaceAllUsesWith( element );
        toRemove.push_back( CI );
    }

    for( CallInst* CI : toRemove )
        CI->eraseFromParent();
}

// Transform
//
//      i32 optixi_loadOrRequestBuffer.variableUniversallyUniqueName(
//          statePtrTy canonicalState,
//          i32 elementSize,
//          u64 ptr,
//          u64 x, u64 y )
//
// for dimensionality = 2 into
//
//      i32 bufferId = i32 rtxiGetBufferId.variableUniversallyUniqueName(
//              statePtrTy canonicalState )
//      i32 rtxiLoadOrRequestBufferElement.variableUniversallyUniqueName(
//              statePtrTy canonicalState,
//              i32 bufferId,
//              i32 elementSize,
//              u64 ptr,
//              u64 x, u64 y )
//
void RTXExceptionInstrumenter::transformLoadOrRequestBufferElement( llvm::Function* function )
{
    std::vector<CallInst*> toRemove;
    for( CallInst* callInst : getCallsToFunction( function ) )
    {
        LoadOrRequestBufferElement* srcElement     = cast<LoadOrRequestBufferElement>( callInst );
        const std::string           variableName   = srcElement->parseUniqueName();
        const int                   dimensionality = srcElement->getDimensionality();
        RT_ASSERT( dimensionality >= 1 && dimensionality <= 3 );

        Value* statePtr    = srcElement->getStatePtr();
        Type*  statePtrTy  = statePtr->getType();
        Value* valuePtr    = srcElement->getPointer();
        Value* elementSize = srcElement->getElementSize();
        Value* x           = srcElement->getX();
        Value* y           = dimensionality > 1 ? srcElement->getY() : nullptr;
        Value* z           = dimensionality > 2 ? srcElement->getZ() : nullptr;

        RtxiGetBufferIdBuilder builder1( m_module, statePtrTy, srcElement );
        builder1.setStatePtr( statePtr );
        CallInst* bufferId = builder1.createCall( variableName );

        if( bufferIdIlwalidEnabled() )
            insertBufferIdIlwalidCheck( statePtr, bufferId, srcElement );

        if( bufferIndexOutOfBoundsEnabled() )
        {
            Value* indices[3] = {x, y, z};
            insertBufferIndexOutOfBoundsCheck( statePtr, bufferId, dimensionality, nullptr /*elementSize*/, indices,
                                               variableName, callInst );
        }

        RtxiLoadOrRequestBufferElementBuilder builder2( m_module, function->getReturnType(), statePtrTy, srcElement );
        builder2.setStatePtr( statePtr );
        builder2.setBufferId( bufferId );
        builder2.setElementSize( elementSize );
        builder2.setPtr( valuePtr );
        builder2.setX( x );
        builder2.setY( y );
        builder2.setZ( z );
        CallInst* element = builder2.createCall( variableName, dimensionality );
        element->takeName( srcElement );

        srcElement->replaceAllUsesWith( element );
        toRemove.push_back( srcElement );
    }

    for( CallInst* CI : toRemove )
        CI->eraseFromParent();
}

// Replace calls to optixi_textureLoadOrRequest (etc.) with calls to RTX_textureLoadOrRequest (etc.)
void RTXExceptionInstrumenter::transformLoadOrRequestTextureElement( llvm::Function* function )
{
    for( CallInst* callInst : getCallsToFunction( function ) )
    {
        // Get function name with RTX_ prefix instead of optixi_
        std::string oldFuncName( function->getName() );
        std::string oldPrefix( "optixi_" );
        std::string newFuncName = std::string( "RTX_" ) + oldFuncName.substr( oldPrefix.size() );

        // Get or create RTX function
        Module*   module      = function->getParent();
        Constant* newFunction = module->getOrInsertFunction( newFuncName, function->getFunctionType() );

        // Update the function call.
        callInst->setCalledFunction( newFunction );
    }
}

void RTXExceptionInstrumenter::transformGetBufferSizeFromId( Function* F )
{
    std::vector<CallInst*> toRemove;
    for( CallInst* CI : getCallsToFunction( F ) )
    {
        Value* statePtr = CI->getArgOperand( 0 );
        Value* bufferId = CI->getArgOperand( 1 );

        if( bufferIdIlwalidEnabled() )
            insertBufferIdIlwalidCheck( statePtr, bufferId, CI );

        RtxiGetBufferSizeFromIdBuilder builder2( m_module, F->getReturnType(), statePtr->getType(), CI );
        builder2.setStatePtr( statePtr );
        builder2.setBufferId( bufferId );
        CallInst* element = builder2.createCall();
        element->takeName( CI );

        CI->replaceAllUsesWith( element );
        toRemove.push_back( CI );
    }

    for( CallInst* CI : toRemove )
        CI->eraseFromParent();
}

void RTXExceptionInstrumenter::transformGetBufferElementFromId( Function* F )
{
    std::vector<CallInst*> toRemove;
    for( CallInst* CI : getCallsToFunction( F ) )
    {
        GetBufferElementFromId* getBufferElementFromId = cast<GetBufferElementFromId>( CI );
        int                     dimensionality         = getBufferElementFromId->getDimensionality();

        Value* statePtr    = getBufferElementFromId->getStatePtr();
        Value* bufferId    = getBufferElementFromId->getBufferId();
        Value* elementSize = getBufferElementFromId->getElementSize();
        Value* offset      = getBufferElementFromId->getOffset();
        Value* x           = getBufferElementFromId->getX();
        Value* y           = dimensionality > 1 ? getBufferElementFromId->getY() : nullptr;
        Value* z           = dimensionality > 2 ? getBufferElementFromId->getZ() : nullptr;

        if( bufferIdIlwalidEnabled() )
            insertBufferIdIlwalidCheck( statePtr, bufferId, CI );

        if( bufferIndexOutOfBoundsEnabled() )
        {
            Value* indices[3] = {x, y, z};
            insertBufferIndexOutOfBoundsCheck( statePtr, bufferId, dimensionality, elementSize, indices, "", CI );
        }

        RtxiGetBufferElementFromIdBuilder builder2( m_module, F->getReturnType(), statePtr->getType(), CI );
        builder2.setStatePtr( statePtr );
        builder2.setBufferId( bufferId );
        builder2.setElementSize( elementSize );
        builder2.setOffset( offset );
        builder2.setX( x );
        builder2.setY( y );
        builder2.setZ( z );
        CallInst* element = builder2.createCall( dimensionality );
        element->takeName( CI );

        CI->replaceAllUsesWith( element );
        toRemove.push_back( CI );
    }

    for( CallInst* CI : toRemove )
        CI->eraseFromParent();
}

void RTXExceptionInstrumenter::transformSetBufferElementFromId( Function* F )
{
    std::vector<CallInst*> toRemove;
    for( CallInst* CI : getCallsToFunction( F ) )
    {
        SetBufferElementFromId* setBufferElementFromId = cast<SetBufferElementFromId>( CI );
        int                     dimensionality         = setBufferElementFromId->getDimensionality();

        Value* statePtr    = setBufferElementFromId->getStatePtr();
        Value* bufferId    = setBufferElementFromId->getBufferId();
        Value* elementSize = setBufferElementFromId->getElementSize();
        Value* offset      = setBufferElementFromId->getOffset();
        Value* x           = setBufferElementFromId->getX();
        Value* y           = dimensionality > 1 ? setBufferElementFromId->getY() : nullptr;
        Value* z           = dimensionality > 2 ? setBufferElementFromId->getZ() : nullptr;
        Value* valueToSet  = setBufferElementFromId->getValueToSet();

        if( bufferIdIlwalidEnabled() )
            insertBufferIdIlwalidCheck( statePtr, bufferId, CI );

        if( bufferIndexOutOfBoundsEnabled() )
        {
            Value* indices[3] = {x, y, z};
            insertBufferIndexOutOfBoundsCheck( statePtr, bufferId, dimensionality, elementSize, indices, "", CI );
        }

        RtxiSetBufferElementFromIdBuilder builder2( m_module, statePtr->getType(), valueToSet->getType(), CI );
        builder2.setStatePtr( statePtr );
        builder2.setBufferId( bufferId );
        builder2.setElementSize( elementSize );
        builder2.setOffset( offset );
        builder2.setX( x );
        builder2.setY( y );
        builder2.setZ( z );
        builder2.setValueToSet( valueToSet );
        CallInst* element = builder2.createCall( dimensionality );
        element->takeName( CI );

        CI->replaceAllUsesWith( element );
        toRemove.push_back( CI );
    }

    for( CallInst* CI : toRemove )
        CI->eraseFromParent();
}

void RTXExceptionInstrumenter::transformGetBufferElementAddressFromId( Function* F )
{
    std::vector<CallInst*> toRemove;
    for( CallInst* CI : getCallsToFunction( F ) )
    {
        GetBufferElementAddressFromId* getBufferElementAddressFromId = cast<GetBufferElementAddressFromId>( CI );
        int                            dimensionality = getBufferElementAddressFromId->getDimensionality();

        Value* statePtr    = getBufferElementAddressFromId->getStatePtr();
        Value* bufferId    = getBufferElementAddressFromId->getBufferId();
        Value* elementSize = getBufferElementAddressFromId->getElementSize();
        Value* x           = getBufferElementAddressFromId->getX();
        Value* y           = dimensionality > 1 ? getBufferElementAddressFromId->getY() : nullptr;
        Value* z           = dimensionality > 2 ? getBufferElementAddressFromId->getZ() : nullptr;

        if( bufferIdIlwalidEnabled() )
            insertBufferIdIlwalidCheck( statePtr, bufferId, CI );

        if( bufferIndexOutOfBoundsEnabled() )
        {
            Value* indices[3] = {x, y, z};
            insertBufferIndexOutOfBoundsCheck( statePtr, bufferId, dimensionality, elementSize, indices, "", CI );
        }

        RtxiGetBufferElementAddressFromIdBuilder builder2( m_module, statePtr->getType(), CI );
        builder2.setStatePtr( statePtr );
        builder2.setBufferId( bufferId );
        builder2.setElementSize( elementSize );
        builder2.setX( x );
        builder2.setY( y );
        builder2.setZ( z );
        CallInst* element = builder2.createCall( dimensionality );
        element->takeName( CI );

        CI->replaceAllUsesWith( element );
        toRemove.push_back( CI );
    }

    for( CallInst* CI : toRemove )
        CI->eraseFromParent();
}

void RTXExceptionInstrumenter::transformAtomicSetBufferElementFromId( Function* F )
{
    std::vector<CallInst*> toRemove;
    for( CallInst* CI : getCallsToFunction( F ) )
    {
        AtomicSetBufferElementFromId* atomicSetBufferElementFromId = cast<AtomicSetBufferElementFromId>( CI );
        int                           dimensionality               = atomicSetBufferElementFromId->getDimensionality();

        Value* statePtr       = atomicSetBufferElementFromId->getStatePtr();
        Value* bufferId       = atomicSetBufferElementFromId->getBufferId();
        Value* elementSize    = atomicSetBufferElementFromId->getElementSize();
        Value* offset         = atomicSetBufferElementFromId->getOffset();
        Value* operation      = atomicSetBufferElementFromId->getOperation();
        Value* compareOperand = atomicSetBufferElementFromId->getCompareOperand();
        Value* operand        = atomicSetBufferElementFromId->getOperand();
        Value* subElementType = atomicSetBufferElementFromId->getSubElementType();
        Value* x              = atomicSetBufferElementFromId->getX();
        Value* y              = dimensionality > 1 ? atomicSetBufferElementFromId->getY() : nullptr;
        Value* z              = dimensionality > 2 ? atomicSetBufferElementFromId->getZ() : nullptr;

        if( bufferIdIlwalidEnabled() )
            insertBufferIdIlwalidCheck( statePtr, bufferId, CI );

        if( bufferIndexOutOfBoundsEnabled() )
        {
            Value* indices[3] = {x, y, z};
            insertBufferIndexOutOfBoundsCheck( statePtr, bufferId, dimensionality, elementSize, indices, "", CI );
        }

        RtxiAtomicSetBufferElementFromIdBuilder builder2( m_module, statePtr->getType(), operand->getType(), CI );
        builder2.setStatePtr( statePtr );
        builder2.setBufferId( bufferId );
        builder2.setElementSize( elementSize );
        builder2.setOffset( offset );
        builder2.setOperation( operation );
        builder2.setCompareOperand( compareOperand );
        builder2.setOperand( operand );
        builder2.setSubElementType( subElementType );
        builder2.setX( x );
        builder2.setY( y );
        builder2.setZ( z );
        CallInst* element = builder2.createCall( dimensionality );
        element->takeName( CI );

        CI->replaceAllUsesWith( element );
        toRemove.push_back( CI );
    }

    for( CallInst* CI : toRemove )
        CI->eraseFromParent();
}

void RTXExceptionInstrumenter::insertTextureIdIlwalidCheck()
{
    for( Function& F : *m_module )
    {
        if( !F.isDeclaration() )
            continue;

        if( F.getName().startswith( "optixi_getTexture" ) && F.getName().endswith( "ValueFromId" ) )
            insertTextureIdIlwalidCheck( &F, m_checkTextureIdFunc );
    }
}

void RTXExceptionInstrumenter::insertTextureIdIlwalidCheck( Function* toInstrument, Function* checkFunc )
{
    for( CallInst* CI : getCallsToFunction( toInstrument ) )
    {
        Value* statePtr = CI->getArgOperand( 0 );
        Value* id       = CI->getArgOperand( 1 );
        insertIlwalidIdCheck( RT_EXCEPTION_TEXTURE_ID_ILWALID, checkFunc, statePtr, id, CI );
    }
}

void RTXExceptionInstrumenter::insertBufferIdIlwalidCheck( Value* statePtr, Value* bufferId, Instruction* insertBefore )
{
    insertIlwalidIdCheck( RT_EXCEPTION_BUFFER_ID_ILWALID, m_checkBufferIdFunc, statePtr, bufferId, insertBefore );
}

void RTXExceptionInstrumenter::insertProgramIdIlwalidCheck()
{
    for( Function& F : *m_module )
    {
        if( !F.isDeclaration() )
            continue;

        if( F.getName().startswith( "optixi_callBound." ) || F.getName().startswith( "optixi_callBindless." ) )
            insertProgramIdIlwalidCheck( &F, m_checkProgramIdFunc );
    }
}

void RTXExceptionInstrumenter::insertProgramIdIlwalidCheck( Function* toInstrument, Function* checkFunc )
{
    for( CallInst* CI : getCallsToFunction( toInstrument ) )
    {
        Value* statePtr = CI->getArgOperand( 0 );
        Value* id       = CI->getArgOperand( 1 );
        insertIlwalidIdCheck( RT_EXCEPTION_PROGRAM_ID_ILWALID, checkFunc, statePtr, id, CI );
    }
}

void RTXExceptionInstrumenter::insertIlwalidIdCheck( RTexception exception, Function* checkFunc, Value* statePtr, Value* id, Instruction* insertBefore )
{
    CoreIRBuilder builder{insertBefore};

    Value* args[]  = {statePtr, id};
    Value* idCheck = builder.CreateCall( checkFunc, args, "idCheck" );

    // The identifier is valid if the value is 0.
    RT_ASSERT( id->getType()->isIntegerTy( 32 ) );
    RT_ASSERT( idCheck->getType()->isIntegerTy( 32 ) );

    Value* description = getSourceLocation( insertBefore );

    Value* exceptionArgs[] = {builder.getInt32( exception ), description, id, idCheck};
    builder.CreateCall( m_throwIlwalidIdException, exceptionArgs );
}

void RTXExceptionInstrumenter::insertBufferIndexOutOfBoundsCheck( Value*             statePtr,
                                                                  Value*             bufferId,
                                                                  int                dimensionality,
                                                                  Value*             elementSize,
                                                                  Value*             indices[3],
                                                                  const std::string& variableName,
                                                                  Instruction*       insertBefore )
{
    CoreIRBuilder builder{insertBefore};

    for( int i     = dimensionality; i < 3; ++i )
        indices[i] = builder.getInt64( 0 );

    Value* description = getBufferDetails( variableName, insertBefore );

    Value* exceptionArgs[] = {builder.getInt32( RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS ),
                              description,
                              indices[0],
                              indices[1],
                              indices[2],
                              builder.getInt32( dimensionality ),
                              elementSize,
                              bufferId};
    builder.CreateCall( m_throwBufferIndexOutOfBoundsException, exceptionArgs );
}

void RTXExceptionInstrumenter::insertIlwalidRayCheck()
{
    for( const auto& F : corelib::getFunctions( m_module ) )
    {
        if( !F->isDeclaration() || !F->getName().startswith( "optixi_" ) )
            continue;
        if( TraceGlobalPayloadCall::isIntrinsic( F ) || parseTraceName( F->getName() ) )
        {
            FunctionType* fntype = F->getFunctionType();
            RT_ASSERT( fntype->getNumParams() == 16 );

            for( CallInst* CI : getCallsToFunction( F ) )
                insertIlwalidRayCheck( CI );
        }
    }
}

void RTXExceptionInstrumenter::insertIlwalidRayCheck( CallInst* trace )
{
    corelib::CoreIRBuilder builder{trace};

    Value* description = getSourceLocation( trace );
    Value* ox          = trace->getArgOperand( 2 );
    Value* oy          = trace->getArgOperand( 3 );
    Value* oz          = trace->getArgOperand( 4 );
    Value* dx          = trace->getArgOperand( 5 );
    Value* dy          = trace->getArgOperand( 6 );
    Value* dz          = trace->getArgOperand( 7 );
    Value* rayType     = trace->getArgOperand( 8 );
    Value* tmin        = trace->getArgOperand( 9 );
    Value* tmax        = trace->getArgOperand( 10 );

    Value* exceptionArgs[] = {
        builder.getInt32( RT_EXCEPTION_ILWALID_RAY ), description, ox, oy, oz, dx, dy, dz, rayType, tmin, tmax};
    builder.CreateCall( m_throwIlwalidRayException, exceptionArgs );
}

void RTXExceptionInstrumenter::insertIndexOutOfBoundsCheck()
{
    std::vector<CallInst*> calls;
    for( Function& F : *m_module )
    {
        if( !F.isDeclaration() )
            continue;

        if( ReportFullIntersection::isIntrinsic( &F ) )
        {
            const std::vector<CallInst*>& c = getCallsToFunction( &F );
            calls.insert( calls.end(), c.begin(), c.end() );
        }
    }

    if( calls.empty() )
        return;

    for( CallInst* call : calls )
    {
        ReportFullIntersection* rfi = dyn_cast<ReportFullIntersection>( call );
        RT_ASSERT( rfi );

        CoreIRBuilder builder{call};

        Value* geometryInstance = builder.CreateCall( m_getGeometryInstanceFunc, None, "geometryInstance" );
        Value* statePtr         = call->getArgOperand( 0 );
        Value* args[]           = { statePtr, geometryInstance };
        Value* numMaterials     = builder.CreateCall( m_getNumMaterialsFunc, args, "numMaterials" );
        RT_ASSERT( numMaterials->getType()->isIntegerTy( 32 ) );

        Value* index = rfi->getMaterialIndex();
        RT_ASSERT( index->getType()->isIntegerTy( 32 ) );

        Value* description = getSourceLocation( call );

        Value* exceptionArgs[] = {builder.getInt32( RT_EXCEPTION_INDEX_OUT_OF_BOUNDS ), description,
                                  builder.CreateZExt( numMaterials, builder.getInt64Ty() ),
                                  builder.CreateZExt( index, builder.getInt64Ty() )};
        builder.CreateCall( m_throwMaterialIndexOutOfBoundsException, exceptionArgs );
    }
}

void RTXExceptionInstrumenter::insertPayloadAccessOutOfBoundsCheck()
{
    DataLayout DL( m_module );

    for( const auto& F : corelib::getFunctions( m_module ) )
    {
        if( !F->isDeclaration() )
            continue;

        bool isSet = isPayloadSet( F );
        bool isGet = isPayloadGet( F );
        if( !isSet && !isGet )
            continue;

        for( CallInst* call : getCallsToFunction( F ) )
        {
            CoreIRBuilder builder{call};

            // Obtain the payload size directly from the corresponding payload register. If the
            // payload has been promoted to memory, the size is in the next register after a
            // payload of maximum size (over all ilwolved programs). If the payload remains in
            // memory, the size is in the next register after the payload pointer, i.e., in payload
            // register 2.
            Type*         i32Ty      = builder.getInt32Ty();
            FunctionType* funcType   = FunctionType::get( i32Ty, i32Ty, false );
            Function*     func       = insertOrCreateFunction( m_module, "lw.rt.read.payload.i32", funcType );
            Value*        sizeOffset = builder.getInt32( m_payloadInRegisters ? m_maxPayloadSize / 4 : 2 );
            RT_ASSERT_MSG( m_maxPayloadSize % 4 == 0, "Fix odd-sized maximum payload" );
            Value* payloadSize = builder.CreateCall( func, sizeOffset, "payloadSize" );
            payloadSize        = builder.CreateZExt( payloadSize, builder.getInt64Ty() );

            Value* description = getSourceLocation( call );
            Value* valueOffset = call->getArgOperand( 1 );
            Type*  valueType = isGet ? call->getCalledFunction()->getReturnType() : call->getArgOperand( 2 )->getType();
            Value* valueSize = builder.getInt64( DL.getTypeAllocSize( valueType ) );
            Value* valueEnd  = builder.CreateAdd( valueOffset, valueSize, "valueEnd" );

            Value* exceptionArgs[] = {builder.getInt32( RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS ),
                                      description,
                                      valueOffset,
                                      valueSize,
                                      payloadSize,
                                      valueEnd};
            builder.CreateCall( m_throwPayloadAccessOutOfBoundsException, exceptionArgs );
        }
    }
}

void RTXExceptionInstrumenter::insertExceptionCodeOutOfBoundsCheck()
{
    Function* func = m_module->getFunction( "optixi_throw" );

    for( CallInst* call : getCallsToFunction( func ) )
    {
        CoreIRBuilder builder{call};

        Value* description   = getSourceLocation( call );
        Value* exceptionCode = call->getArgOperand( 1 );

        Value* exceptionArgs[] = {builder.getInt32( RT_EXCEPTION_USER_EXCEPTION_CODE_OUT_OF_BOUNDS ), description, exceptionCode,
                                  builder.getInt32( RT_EXCEPTION_USER ), builder.getInt32( RT_EXCEPTION_USER_MAX )};
        builder.CreateCall( m_throwExceptionCodeOutOfBoundsException, exceptionArgs );
    }
}

bool RTXExceptionInstrumenter::textureIdIlwalidEnabled() const
{
    return exceptionEnabled( k_enableIlwalidTextureIdHandling.get(), RT_EXCEPTION_TEXTURE_ID_ILWALID );
}

bool RTXExceptionInstrumenter::programIdIlwalidEnabled() const
{
    return exceptionEnabled( k_enableIlwalidProgramIdHandling.get(), RT_EXCEPTION_PROGRAM_ID_ILWALID );
}

bool RTXExceptionInstrumenter::bufferIdIlwalidEnabled() const
{
    return exceptionEnabled( k_enableIlwalidBufferIdHandling.get(), RT_EXCEPTION_BUFFER_ID_ILWALID );
}

bool RTXExceptionInstrumenter::bufferIndexOutOfBoundsEnabled() const
{
    return exceptionEnabled( k_enableBufferIndexOutOfBoundsHandling.get(), RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS );
}

bool RTXExceptionInstrumenter::ilwalidRayEnabled() const
{
    return exceptionEnabled( k_enableIlwalidRayHandling.get(), RT_EXCEPTION_ILWALID_RAY );
}

bool RTXExceptionInstrumenter::indexOutOfBoundsEnabled() const
{
    return exceptionEnabled( k_enableIndexOutOfBoundsHandling.get(), RT_EXCEPTION_INDEX_OUT_OF_BOUNDS );
}

bool RTXExceptionInstrumenter::payloadOffsetOutOfBoundsEnabled() const
{
    return exceptionEnabled( k_enablePayloadAccessOutOfBoundsHandling.get(), RT_EXCEPTION_PAYLOAD_ACCESS_OUT_OF_BOUNDS );
}

bool RTXExceptionInstrumenter::exceptionEnabled( bool exceptionKnob, RTexception exception ) const
{
    if( m_stype == ST_EXCEPTION )
        return false;
    if( k_enableAllExceptions.get() )
        return true;
    if( k_disableAllExceptions.get() )
        return false;
    if( exceptionKnob )
        return true;
    return Context::getExceptionEnabled( m_exceptionFlags, exception );
}

bool RTXExceptionInstrumenter::hasProductSpecificExceptionsEnabled( SemanticType stype, uint64_t exceptionFlags )
{
    if( stype == ST_EXCEPTION )
        return false;
    return Context::hasProductSpecificExceptionsEnabled( exceptionFlags );
}

Value* RTXExceptionInstrumenter::getBufferDetails( const std::string& variableName, Instruction* instruction )
{
    corelib::CoreIRBuilder builder{instruction};

    std::string str = getExactSourceLocationAsString( instruction );
    if( !variableName.empty() )
    {
        if( !str.empty() )
            str += ", ";
        str += "\"" + variableName.substr( variableName.rfind( "." ) + 1 ) + "\"";
    }
    else if( str.empty() )
    {
        str = "n/a (ilwoke lwcc with the -lineinfo option)";
    }

    Value*& result = m_stringsCache[str];
    if( result != nullptr )
        return result;

    Constant* constant   = ConstantDataArray::getString( m_module->getContext(), str, true );
    Type*     constantTy = constant->getType();
    Constant* gv = new GlobalVariable( *m_module, constantTy, true, GlobalValue::PrivateLinkage, constant, "bufferName",
                                       nullptr, GlobalVariable::NotThreadLocal, ADDRESS_SPACE_GLOBAL, false );
    Constant* genericPtr = ConstantExpr::getAddrSpaceCast( gv, constantTy->getPointerTo() );
    result               = builder.CreatePtrToInt( genericPtr, builder.getInt64Ty() );
    return result;
}

Value* RTXExceptionInstrumenter::getSourceLocation( Instruction* instruction )
{
    corelib::CoreIRBuilder builder{instruction};

    std::string str = getApproximateSourceLocationAsString( instruction );
    if( str.empty() )
        str = "n/a (ilwoke lwcc with the -lineinfo option, or no useful information for that block present)";

    Value*& result = m_stringsCache[str];
    if( result != nullptr )
        return result;

    Constant* constant   = ConstantDataArray::getString( m_module->getContext(), str, true );
    Type*     constantTy = constant->getType();
    Constant* gv = new GlobalVariable( *m_module, constantTy, true, GlobalValue::PrivateLinkage, constant, "location",
                                       nullptr, GlobalVariable::NotThreadLocal, ADDRESS_SPACE_GLOBAL, false );
    Constant* genericPtr = ConstantExpr::getAddrSpaceCast( gv, constantTy->getPointerTo() );
    result               = builder.CreatePtrToInt( genericPtr, builder.getInt64Ty() );
    return result;
}

std::string RTXExceptionInstrumenter::getApproximateSourceLocationAsString( Instruction* instruction )
{
    std::string result = getExactSourceLocationAsString( instruction );
    if( !result.empty() )
        return result;

    // Obtain iterator to instruction after \p instruction. Initializing the iterator directly from
    // the instruction does not work. If \p instruction is the last instruction of the basic block,
    // the final increment operation after the loop causes a crash.
    BasicBlock::iterator next     = instruction->getParent()->begin();
    BasicBlock::iterator next_end = instruction->getParent()->end();
    while( &( *next ) != instruction )
        ++next;
    ++next;

    // Obtain iterator to instruction before \p instruction. Initializing the iterator directly from
    // the instruction does not work. If \p instruction is the first instruction of the basic block,
    // the final increment operation after the loop causes a crash.
    BasicBlock::reverse_iterator prev     = instruction->getParent()->rbegin();
    BasicBlock::reverse_iterator prev_end = instruction->getParent()->rend();
    while( &( *prev ) != instruction )
        ++prev;
    ++prev;

    // Walk in both directions until we find a source location or there are no more instructions in
    // the basic block.
    while( prev != prev_end || next != next_end )
    {
        if( prev != prev_end )
        {
            result = getExactSourceLocationAsString( &( *prev ) );
            if( !result.empty() )
                return result + " (approximately)";
            ++prev;
        }

        if( next != next_end )
        {
            result = getExactSourceLocationAsString( &( *next ) );
            if( !result.empty() )
                return result + " (approximately)";
            ++next;
        }
    }

    return std::string();
}

std::string RTXExceptionInstrumenter::getExactSourceLocationAsString( Instruction* instruction )
{
    const DebugLoc& debugLoc = instruction->getDebugLoc();

    DILocation* diLocation = debugLoc.get();
    if( !diLocation )
        return std::string();
    const std::string filename = diLocation->getFilename();

    if( filename.empty() )
        return std::string();

    if( filenameIsBlacklisted( filename ) )
        return std::string();

    const std::string directory = diLocation->getDirectory();

    if( directoryIsBlacklisted( directory ) )
        return std::string();

    const unsigned int line   = diLocation->getLine();
    const unsigned int column = diLocation->getColumn();

#ifdef WIN32
#define SEPARATOR '\\'
#else
#define SEPARATOR '/'
#endif

    return directory + SEPARATOR + filename + ":" + std::to_string( line ) + ":" + std::to_string( column );

#undef SEPARATOR
}

bool RTXExceptionInstrumenter::filenameIsBlacklisted( const std::string& filename )
{
    return filename == "optix_device.h" || filename == "optix_internal.h";
}

bool RTXExceptionInstrumenter::directoryIsBlacklisted( const std::string& directory )
{
    return directory == getGeneratedCodeDirectory();
}

void RTXExceptionInstrumenter::dump( const std::string& functionName, int dumpId, const std::string& suffix )
{
    addMissingLineInfoAndDump( m_module, k_saveLLVM.get(), suffix, dumpId, m_launchCounterForDebugging,
                               functionName + "-RTXExceptionInstrumenter" );
}

}  // namespace optix
