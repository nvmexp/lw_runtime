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

#include <Context/Context.h>
#include <Context/ProgramManager.h>
#include <ExelwtionStrategy/Compile.h>  // Regex parsing of getBufferElement() etc.
#include <ExelwtionStrategy/RuntimeExceptionInstrumenter.h>
#include <FrontEnd/Canonical/FrontEndHelpers.h>
#include <FrontEnd/Canonical/IntrinsicsManager.h>
#include <FrontEnd/Canonical/VariableReference.h>
#include <Objects/SemanticType.h>
#include <Util/Enum2String.h>           // exception2string()
#include <corelib/compiler/LLVMUtil.h>  // getConstantValueOrThrow()
#include <corelib/misc/String.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/CompileError.h>
#include <prodlib/system/Knobs.h>

#include <llvm/ADT/APFloat.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

using namespace optix;
using namespace llvm;
using namespace corelib;
using namespace prodlib;

namespace {
// clang-format off
// Knob precedence:
// - enable individuals
// - disable individuals
// - enable all
// - disable all
PublicKnob<bool> k_enableAllExceptions( RT_PUBLIC_DSTRING("compile.enableAllExceptions"), false, RT_PUBLIC_DSTRING("Force creation of exception handling code for all exceptions that we support." ) );
PublicKnob<bool> k_disableAllExceptions( RT_PUBLIC_DSTRING("compile.disableAllExceptions"), false, RT_PUBLIC_DSTRING("Do not create code to handle any exceptions." ) );
Knob<bool> k_enableStackOverflowHandling( RT_DSTRING("compile.enableStackOverflowHandling"), false, RT_DSTRING("Create code to handle stack overflow exceptions." ) );
Knob<bool> k_enableIlwalidBufferIdHandling( RT_DSTRING("compile.enableIlwalidBufferIdHandling"), false, RT_DSTRING("Create code to handle invalid buffer id exceptions." ) );
Knob<bool> k_enableIlwalidTextureIdHandling( RT_DSTRING("compile.enableIlwalidTextureIdHandling"), false, RT_DSTRING("Create code to handle invalid texture id exceptions." ) );
Knob<bool> k_enableIlwalidProgramIdHandling( RT_DSTRING("compile.enableIlwalidProgramIdHandling"), false, RT_DSTRING("Create code to handle invalid program id exceptions." ) );
Knob<bool> k_enableBufferIndexOutOfBoundsHandling( RT_DSTRING("compile.enableBufferIndexOutOfBoundsHandling"), false, RT_DSTRING("Create code to handle buffer index out of bounds exceptions." ) );
Knob<bool> k_enableIndexOutOfBoundsHandling( RT_DSTRING("compile.enableIndexOutOfBoundsHandling"), false, RT_DSTRING("Create code to handle index out of bounds exceptions." ) );
Knob<bool> k_enableInternalErrorHandling( RT_DSTRING("compile.enableInternalErrorHandling"), false, RT_DSTRING("Create code to handle internal error exceptions." ) );
Knob<bool> k_enableIlwalidRayHandling( RT_DSTRING("compile.enableIlwalidRayHandling"), false, RT_DSTRING("Create code to handle invalid ray exceptions." ) );
// clang-format on
}

static Type* getCanonicalStateType( Module* module )
{
    // Just query an arbitrary function of which we know it has that type and has no interstate.
    Function* func = getFunctionOrAssert( module, "_ZN4cort22Global_getStatusReturnEPNS_14CanonicalStateE" );
    RT_ASSERT( func->getFunctionType()->getNumParams() == 1 );
    const FunctionType* funcTy = func->getFunctionType();
    return funcTy->getParamType( 0 );
}

static Type* getCanonicalStateMKType( Module* module )
{
    // Query the name directly, in contrast to CS it is always the same.
    Type* type = module->getTypeByName( "struct.Megakernel::MKCanonicalState" );
    RT_ASSERT( type );
    return type->getPointerTo();
}

RuntimeExceptionInstrumenter::RuntimeExceptionInstrumenter( Module*               module,
                                                            const uint64_t        exceptionFlags,
                                                            const ProgramManager& programManager,
                                                            const FunctionSet&    stateFunctions,
                                                            const FunctionSet&    exceptionFunctions,
                                                            const bool            isMegakernel )
    : m_isMegakernel( isMegakernel )
    , m_module( module )
    , m_exceptionFlags( exceptionFlags )
    , m_programManager( programManager )
    , m_stateFunctions( stateFunctions )
    , m_exceptionFunctions( exceptionFunctions )
    , m_throwingFunctions()
    , m_canonicalStateType( getCanonicalStateType( module ) )
    , m_canonicalStateMKType( isMegakernel ? getCanonicalStateMKType( module ) : nullptr )
{
    Constant* constant = ConstantDataArray::getString( m_module->getContext(), "n/a", true );
    m_gv = new GlobalVariable( *m_module, constant->getType(), true, GlobalValue::PrivateLinkage, constant, "na" );
}


static void createExceptionBlocks( Instruction* inst, BasicBlock*& okayBlock, BasicBlock*& exceptionBlock )
{
    okayBlock      = inst->getParent()->splitBasicBlock( inst, "noException" );
    exceptionBlock = BasicBlock::Create( inst->getContext(), "throwException", okayBlock->getParent(), okayBlock );
}

static CallInst* createExceptionGetCodeCall( Value* state, corelib::CoreIRBuilder& builder )
{
    Module*   module               = builder.GetInsertBlock()->getParent()->getParent();
    Function* getExceptionCodeFunc = getFunctionOrAssert( module, "_ZN4cort17Exception_getCodeEPNS_14CanonicalStateE" );
    RT_ASSERT( getExceptionCodeFunc->getFunctionType()->getNumParams() == 1 );
    return builder.CreateCall( getExceptionCodeFunc, state, "code" );
}

static CallInst* createExceptionSetCodeCall( Value* state, Value* exceptiolwal, corelib::CoreIRBuilder& builder )
{
    Module*   module = builder.GetInsertBlock()->getParent()->getParent();
    Function* setExceptionCodeFunc =
        getFunctionOrAssert( module, "_ZN4cort17Exception_setCodeEPNS_14CanonicalStateEj" );
    RT_ASSERT( setExceptionCodeFunc->getFunctionType()->getNumParams() == 2 );

    Value* args[2] = {state, exceptiolwal};
    return builder.CreateCall( setExceptionCodeFunc, args );
}

static CallInst* createExceptionSetDetailCall( Value* state, Value* detail, int which, corelib::CoreIRBuilder& builder )
{
    RT_ASSERT( which <= 10 );
    Module*   module = builder.GetInsertBlock()->getParent()->getParent();
    Function* setExceptionDetailFunc =
        getFunctionOrAssert( module, "_ZN4cort19Exception_setDetailEPNS_14CanonicalStateEjj" );
    RT_ASSERT( setExceptionDetailFunc->getFunctionType()->getNumParams() == 3 );

    RT_ASSERT( detail->getType()->isIntegerTy( 32 ) );

    Value* args[] = {state, detail, builder.getInt32( which )};
    return builder.CreateCall( setExceptionDetailFunc, args );
}

static CallInst* createExceptionSetDetail64Call( Value* state, Value* detail64, int which, corelib::CoreIRBuilder& builder )
{
    RT_ASSERT( which <= 10 );
    Module*   module = builder.GetInsertBlock()->getParent()->getParent();
    Function* setExceptionDetail64Func =
        getFunctionOrAssert( module, "_ZN4cort21Exception_setDetail64EPNS_14CanonicalStateEyj" );
    RT_ASSERT( setExceptionDetail64Func->getFunctionType()->getNumParams() == 3 );

    RT_ASSERT( detail64->getType()->isIntegerTy( 64 ) );

    Value* args[] = {state, detail64, builder.getInt32( which )};
    return builder.CreateCall( setExceptionDetail64Func, args );
}

static CallInst* createExceptionThrowCall( Value* state, corelib::CoreIRBuilder& builder )
{
    Module*   module             = builder.GetInsertBlock()->getParent()->getParent();
    Function* throwExceptionFunc = getFunctionOrAssert( module, "_ZN4cort15Exception_throwEPNS_14CanonicalStateE" );
    return builder.CreateCall( throwExceptionFunc, state, "exceptionStateID" );
}

// If the current function is an exception state function, return -1.
// If the current function is a state function, return the exception state ID.
// Otherwise create a simple return and mark the function as "throws".
void RuntimeExceptionInstrumenter::createThrow( Value* state, corelib::CoreIRBuilder& builder )
{
    Function* parentFunc = builder.GetInsertBlock()->getParent();

    // If the current function is an exception state function, immediately return
    // with the returnStateID (MegakernelES).
    // TODO: It might be nice for users if we printed a warning in such a case.
    if( isExceptionFunction( parentFunc ) )
    {
        if( !m_isMegakernel )
        {
            builder.CreateRetVoid();
            return;
        }

        // Compute the return state ID.
        Function* continuationReturnFunc =
            getFunctionOrAssert( m_module, "_ZN10Megakernel18continuationReturnEPN4cort14CanonicalStateE" );

        Value* returnStateID = builder.CreateCall( continuationReturnFunc, state, "returnStateID" );
        builder.CreateRet( returnStateID );
        return;
    }

    // If the current function is a state function, call Exception_throw() and
    // return its result.
    if( isStateFunction( parentFunc ) )
    {
        if( !m_isMegakernel )
        {
            createExceptionThrowCall( state, builder );
            builder.CreateRetVoid();
            return;
        }

        // Create the exception handling call.
        Value* stateID = createExceptionThrowCall( state, builder );
        RT_ASSERT( parentFunc->getReturnType() == stateID->getType() );
        builder.CreateRet( stateID );
        return;
    }

    // Otherwise, just create a return.

    // If the parent function of this call returns void, create a simple return.
    // Otherwise, return undef (it will never be used anyway).
    if( parentFunc->getReturnType()->isVoidTy() )
        builder.CreateRetVoid();
    else
        builder.CreateRet( UndefValue::get( parentFunc->getReturnType() ) );

    // Mark the function as "throws". The return of the state ID is created for
    // all these functions later.
    m_throwingFunctions.insert( parentFunc );
}

bool RuntimeExceptionInstrumenter::isCanonicalState( Value* value ) const
{
    return value->getType() == m_canonicalStateType || value->getType() == m_canonicalStateMKType;
}

bool RuntimeExceptionInstrumenter::isStateFunction( Function* func ) const
{
    return m_stateFunctions.count( func );
}

bool RuntimeExceptionInstrumenter::isExceptionFunction( Function* func ) const
{
    return m_exceptionFunctions.count( func );
}

// Insert the following code behind 'call':
// if ( Exception_getCode( state ) != 0 )
//   return [undef];
//
// or, in case of a call in a state function:
//
// if ( Exception_getCode( state ) != 0 )
//   return Exception_throw( state );
void RuntimeExceptionInstrumenter::insertTryCatch( CallInst* call )
{
    Instruction* insertBefore = &(* ++( BasicBlock::iterator( call ) ) );
    RT_ASSERT( ( ++BasicBlock::iterator( call ) ) != call->getParent()->end() );

    Function* getCodeFunc = getFunctionOrAssert( m_module, "_ZN4cort17Exception_getCodeEPNS_14CanonicalStateE" );
    RT_ASSERT( getCodeFunc->getFunctionType()->getNumParams() == 1 );

    BasicBlock* checkBlock     = insertBefore->getParent();
    BasicBlock* okayBlock      = nullptr;
    BasicBlock* exceptionBlock = nullptr;
    createExceptionBlocks( insertBefore, okayBlock, exceptionBlock );

    TerminatorInst*        ti = checkBlock->getTerminator();
    corelib::CoreIRBuilder builder( ti );

    Value* state = call->getArgOperand( 0 );
    RT_ASSERT( isCanonicalState( state ) );

    // Cast to the right state type if necessary.
    Type* stateTy = getCodeFunc->getFunctionType()->getParamType( 0 );
    if( state->getType() != getCodeFunc->getFunctionType()->getParamType( 0 ) )
        state = builder.CreateBitCast( state, stateTy );

    Value* code      = builder.CreateCall( getCodeFunc, state, "exception.code" );
    Value* condition = builder.CreateICmpNE( code, builder.getInt32( 0 ), "exceptionThrown" );
    builder.CreateCondBr( condition, exceptionBlock, okayBlock );
    ti->eraseFromParent();

    builder.SetInsertPoint( exceptionBlock );

    createThrow( state, builder );
}

// Create the following code:
// if ( condition )
// {
//   Exception_setCode( state, exception );
//   Exception_setDetail( state, detail0, 0 );
//   Exception_setDetail( state, detail1, 1 );
//   ...
//   Exception_setDetail( state, detailN, N );
//   Exception_setDetail64( state, detail0, 0 );
//   Exception_setDetail64( state, detail1, 1 );
//   ...
//   Exception_setDetail64( state, detailN, N );
//   return Exception_throw( state );
// }
void RuntimeExceptionInstrumenter::createException( Value*               state,
                                                    Value*               condition,
                                                    RTexception          exception,
                                                    Instruction*         insertBefore,
                                                    std::vector<Value*>* detail,
                                                    std::vector<Value*>* detail64 )
{
    BasicBlock* checkBlock     = insertBefore->getParent();
    BasicBlock* okayBlock      = nullptr;
    BasicBlock* exceptionBlock = nullptr;
    createExceptionBlocks( insertBefore, okayBlock, exceptionBlock );

    TerminatorInst*        ti = checkBlock->getTerminator();
    corelib::CoreIRBuilder builder( ti );

    builder.CreateCondBr( condition, exceptionBlock, okayBlock );
    ti->eraseFromParent();

    // Set the exception code.
    builder.SetInsertPoint( exceptionBlock );
    createExceptionSetCodeCall( state, builder.getInt32( exception ), builder );

    // If desired, set the exception detail.
    if( detail )
    {
        builder.SetInsertPoint( exceptionBlock );
        for( int i = 0, e = detail->size(); i < e; ++i )
        {
            Value* det = ( *detail )[i];
            if( !det )
                continue;
            createExceptionSetDetailCall( state, det, i, builder );
        }
    }

    if( detail64 )
    {
        builder.SetInsertPoint( exceptionBlock );
        for( int i = 0, e = detail64->size(); i < e; ++i )
        {
            Value* det = ( *detail64 )[i];
            if( !det )
                continue;
            createExceptionSetDetail64Call( state, det, i, builder );
        }
    }

    createThrow( state, builder );
}


void RuntimeExceptionInstrumenter::createStackOverflow( Value* state, ConstantInt* frameSize, Instruction* insertBefore )
{
    // If frame size is 0, we don't need an overflow check.
    if( frameSize->getZExtValue() == 0 )
        return;

    Function* checkStackOverflowFunc =
        getFunctionOrAssert( m_module, "_ZN4cort28Exception_checkStackOverflowEPNS_14CanonicalStateEj" );
    RT_ASSERT( checkStackOverflowFunc->getFunctionType()->getNumParams() == 2 );

    corelib::CoreIRBuilder irb{insertBefore};
    Value*                 args[]       = {state, frameSize};
    CallInst*              overflowCond = irb.CreateCall( checkStackOverflowFunc, args, "willOverflow" );

    createException( state, overflowCond, RT_EXCEPTION_STACK_OVERFLOW, insertBefore );
}

void RuntimeExceptionInstrumenter::createIdIlwalid( Function* checkIdFunc, RTexception exception, Value* state, Value* id, Instruction* insertBefore )
{
    RT_ASSERT( checkIdFunc );
    RT_ASSERT( checkIdFunc->getFunctionType()->getNumParams() == 2 );

    corelib::CoreIRBuilder builder( insertBefore );

    Value* args[]  = {state, id};
    Value* idError = builder.CreateCall( checkIdFunc, args, "idError" );

    // The identifier is valid if the returned error is 0.
    RT_ASSERT( id->getType()->isIntegerTy( 32 ) );
    RT_ASSERT( idError->getType()->isIntegerTy( 32 ) );
    Value* idIlwalidCond = builder.CreateICmpNE( idError, builder.getInt32( 0 ), "isIlwalid" );

    std::vector<Value*> detail;
    detail.push_back( id );
    detail.push_back( idError );

    std::vector<Value*> detail64;
    detail64.push_back( builder.CreatePtrToInt( m_gv, builder.getInt64Ty() ) );  // location not supported

    createException( state, idIlwalidCond, exception, insertBefore, &detail, &detail64 );
}


// Insert stack overflow checks before every call to stackAllocate().
// NOTE: This could be put directly into stackAllocate() in the runtime.
//       However, that would require us to do some other special handling later,
//       since we can't "double return" from the state function from within
//       stackAllocate().
void RuntimeExceptionInstrumenter::insertStackOverflowHandler()
{
    if( !Context::getExceptionEnabled( m_exceptionFlags, RT_EXCEPTION_STACK_OVERFLOW )
        && !k_enableStackOverflowHandling.get() && !k_enableAllExceptions.get() )
    {
        llog( 20 ) << "Disabled exception handling for stack overflow exceptions.\n";
        return;
    }

    if( !m_isMegakernel )
    {
        lwarn << "Exception handling for stack overflow exceptions not supported.\n";
        return;
    }

    // Create stack overflow handling code before every stack alloc function call.
    Function* stackAllocFunc =
        getFunctionOrAssert( m_module, "_ZN10Megakernel13stackAllocateEPN4cort14CanonicalStateEj" );
    Function* stackAllocRegFn =
        getFunctionOrAssert( m_module, "_ZN10Megakernel22stackAllocate_registerEbPN4cort14CanonicalStateEj" );
    for( CallInst* stackAllocCall : getCallsToFunction( stackAllocFunc ) )
    {
        // Don't create stack overflow checks inside exception programs (before
        // exelwtion, the stack is reset, so there should always be enough memory).
        Function* parentFunc = stackAllocCall->getParent()->getParent();
        if( isExceptionFunction( parentFunc ) )
            continue;
        if( parentFunc == stackAllocRegFn )
            continue;

        // Get the stack frame size for this function from the stackAllocate() call.
        Value* frameSizeVal = stackAllocCall->getArgOperand( stackAllocCall->getNumArgOperands() - 1 );
        RT_ASSERT( isa<ConstantInt>( frameSizeVal ) );
        ConstantInt* sizeConst = cast<ConstantInt>( frameSizeVal );

        // Get the canonical state (first argument of parent function).
        Value* state = stackAllocCall->getArgOperand( 0 );

        createStackOverflow( state, sizeConst, stackAllocCall );
    }
}

// TODO: Call Exception_checkIdIlwalid directly with the tableSize,
//       which should be obtained here from the TableManager in the OptiX
//       Context instead of being stored in the CanonicalState (it is statically
//       known and never changes).
//       The only thing required for this is to hand down the context to this
//       class. We should store the size or the TableManager as a member variable.
void RuntimeExceptionInstrumenter::insertIdIlwalidHandler( RTexception exception, const char* funcName )
{
    Function* func      = getFunctionOrAssert( m_module, funcName );
    Function* checkFunc = nullptr;
    switch( exception )
    {
        case RT_EXCEPTION_BUFFER_ID_ILWALID:
            checkFunc =
                getFunctionOrAssert( m_module, "_ZN4cort30Exception_checkBufferIdIlwalidEPNS_14CanonicalStateEj" );
            break;

        case RT_EXCEPTION_TEXTURE_ID_ILWALID:
            checkFunc =
                getFunctionOrAssert( m_module, "_ZN4cort31Exception_checkTextureIdIlwalidEPNS_14CanonicalStateEj" );
            break;

        case RT_EXCEPTION_PROGRAM_ID_ILWALID:
            checkFunc =
                getFunctionOrAssert( m_module, "_ZN4cort31Exception_checkProgramIdIlwalidEPNS_14CanonicalStateEj" );
            break;

        default:
            throw CompileError( RT_EXCEPTION_INFO, "Bad exception type for insertIdIlwalidHandler() " + exception2string( exception ) );
    }

    for( CallInst* call : getCallsToFunction( func ) )
    {
        // Get the canonical state and the identifier from the call.
        Value* state = call->getArgOperand( 0 );
        Value* id    = call->getArgOperand( 1 );

        createIdIlwalid( checkFunc, exception, state, id, call );
    }
}

// Mark all calls that begin with 'namePrefix' as 'throwing'.
void RuntimeExceptionInstrumenter::markAsThrowing( const char* namePrefix )
{
    for( Function& F : *m_module )
        if( F.getName().startswith( namePrefix ) )
            m_throwingFunctions.insert( &F );
}

// Insert invalid id checks before every call to getBufferHeader(FromConst).
void RuntimeExceptionInstrumenter::insertBufferIdIlwalidHandler()
{
    if( !Context::getExceptionEnabled( m_exceptionFlags, RT_EXCEPTION_BUFFER_ID_ILWALID )
        && !k_enableIlwalidBufferIdHandling.get() && !k_enableAllExceptions.get() )
    {
        llog( 20 ) << "Disabled exception handling for invalid buffer id exceptions.\n";
        return;
    }

    for( Function& F : *m_module )
    {
        if( GetBufferElementFromId::isIntrinsic( &F ) || SetBufferElementFromId::isIntrinsic( &F ) )
        {
            insertIdIlwalidHandler( RT_EXCEPTION_BUFFER_ID_ILWALID, F.getName().str().c_str() );
        }
    }

    // In addition, all get/set buffer calls have to be marked as "throwing".
    markAsThrowing( "optixi_getBufferElementFromId" );
    markAsThrowing( "optixi_setBufferElementFromId" );
}

// Insert invalid id checks before every call to getTextureHeader(FromConst).
void RuntimeExceptionInstrumenter::insertTextureIdIlwalidHandler()
{
    if( !Context::getExceptionEnabled( m_exceptionFlags, RT_EXCEPTION_TEXTURE_ID_ILWALID )
        && !k_enableIlwalidTextureIdHandling.get() && !k_enableAllExceptions.get() )
    {
        llog( 20 ) << "Disabled exception handling for invalid texture id exceptions.\n";
        return;
    }

    insertIdIlwalidHandler( RT_EXCEPTION_TEXTURE_ID_ILWALID,
                            "_ZN4cort30Global_getTextureSamplerHeaderEPNS_14CanonicalStateEj" );
    if( m_isMegakernel )
        insertIdIlwalidHandler( RT_EXCEPTION_TEXTURE_ID_ILWALID, "Megakernel_getTextureHeaderFromConst" );

    // In addition, all get texture calls have to be marked as "throwing".
    markAsThrowing( "optixi_getTexture" );
}

// Insert invalid id checks before every call to getProgramHeader(FromConst).
void RuntimeExceptionInstrumenter::insertProgramIdIlwalidHandler()
{
    if( !Context::getExceptionEnabled( m_exceptionFlags, RT_EXCEPTION_PROGRAM_ID_ILWALID )
        && !k_enableIlwalidProgramIdHandling.get() && !k_enableAllExceptions.get() )
    {
        llog( 20 ) << "Disabled exception handling for invalid program id exceptions.\n";
        return;
    }

    Function* checkIlwalidProgramIdDummyFunc =
        getFunctionOrAssert( m_module, "_ZN4cort31Exception_checkIlwalidProgramIdEPNS_14CanonicalStateEj" );

    Function* checkFunc =
        getFunctionOrAssert( m_module, "_ZN4cort31Exception_checkProgramIdIlwalidEPNS_14CanonicalStateEj" );

    // Replace every use of this function by exception handling code.
    for( CallInst* call : getCallsToFunction( checkIlwalidProgramIdDummyFunc ) )
    {
        // Get the state.
        unsigned i     = 0;
        Value*   state = call->getArgOperand( i++ );
        Value*   id    = call->getArgOperand( i++ );

        // Check for an invalid id.
        createIdIlwalid( checkFunc, RT_EXCEPTION_PROGRAM_ID_ILWALID, state, id, call );

        // Erase the dummy call.
        call->eraseFromParent();
    }
}

template <class BufferAccess>
void RuntimeExceptionInstrumenter::instrumentBufferAccessFunction( Function* F, Function* getBufferSizeFunc, Function* getBufferAddressFunc )
{
    std::string  varrefUniqueName = BufferAccess::parseUniqueName( F->getName() );
    unsigned int dim              = BufferAccess::getDimensionality( F );

    // Fetch the variable reference.
    const VariableReference* varref = m_programManager.getVariableReferenceByUniversallyUniqueName( varrefUniqueName );

    // The element size of the buffer is returned by the 'numElements' function.
    RT_ASSERT( varref->getType().baseType() == VariableType::Buffer );
    const int elementSize = varref->getType().numElements();
    Value*    tokelwal    = ConstantInt::get( Type::getInt16Ty( m_module->getContext() ), varref->getVariableToken() );

    // For every use of this function, create exception handling code *before* the use.
    for( CallInst* CI : getCallsToFunction( F ) )
    {
        BufferAccess* call = dyn_cast<BufferAccess>( CI );
        if( !call )
            continue;

        corelib::CoreIRBuilder builder( call );

        // Get the state.
        Value* state = call->getStatePtr();

        // Get the size of the buffer (returns a struct of 3x i64 values).
        Value*    args[2] = {state, tokelwal};
        CallInst* size    = builder.CreateCall( getBufferSizeFunc, args, "bufferSize" );

        // Check if the access is okay.
        // Get the indices and compare them to the corresponding size.
        // Use the disjunction of the results as the condition.
        Value* condition = nullptr;
        for( unsigned int i = 0; i < dim; ++i )
        {
            // Update the condition.
            Value* sizeI  = builder.CreateExtractValue( size, i, "size" );
            Value* indexI = call->getIndex( i );
            RT_ASSERT( sizeI->getType()->isIntegerTy( 64 ) );
            RT_ASSERT( indexI->getType()->isIntegerTy( 64 ) );
            Value* cmpI = builder.CreateICmpUGE( indexI, sizeI, "outOfBounds" );
            if( !condition )
                condition = cmpI;
            else
                condition = builder.CreateOr( condition, cmpI, "" );
        }

        if( Instruction* condI = dyn_cast<Instruction>( condition ) )
            condI->setName( "bufferIndexOutOfBounds" );

        // Create exception detail64:
        // - unused (formerly buffer address)
        // - size x/y/z
        // - accessed index x/y/z
        std::vector<Value*> detail64;
        detail64.resize( 7 );

        // Update the detail64 with the buffer name (unsupported here)
        detail64[0] = builder.CreatePtrToInt( m_gv, builder.getInt64Ty() );

        // Update the detail64 with size and index of this dimension.
        for( unsigned int i = 0; i < dim; ++i )
        {
            Value* sizeI  = builder.CreateExtractValue( size, i, "size" );
            Value* indexI = call->getIndex( i );
            RT_ASSERT( sizeI->getType()->isIntegerTy( 64 ) );
            RT_ASSERT( indexI->getType()->isIntegerTy( 64 ) );
            detail64[i + 1]     = sizeI;   // indices 1/2/3
            detail64[i + 3 + 1] = indexI;  // indices 4/5/6
        }

        // Update the detail64 with the remaining size and index (set to 0) (if dim < 3).
        for( unsigned int i = dim; i < 3; ++i )
        {
            detail64[i + 1]     = builder.CreateExtractValue( size, i, "size" );  // indices 2/3
            detail64[i + 3 + 1] = builder.getInt64( 0 );                          // indices 5/6
        }

        // Create exception detail:
        // - dimensionality
        // - element size
        // - buffer ID
        std::vector<Value*> detail;
        detail.resize( 3 );
        detail[0] = builder.getInt32( dim );
        detail[1] = builder.getInt32( elementSize );
        detail[2] = ConstantInt::getAllOnesValue( builder.getInt32Ty() );  // buffer ID unsupported here

        // Create the exception handling code.
        createException( state, condition, RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS, call, &detail, &detail64 );
    }
}

// Insert buffer index out of bounds checks before every call to get/setBufferElement.
// TODO: Are get/setBufferElement() the only possibilities to access a
//       buffer with an index?
void RuntimeExceptionInstrumenter::insertBufferIndexOutOfBoundsHandler()
{
    if( !Context::getExceptionEnabled( m_exceptionFlags, RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS )
        && !k_enableBufferIndexOutOfBoundsHandling.get() && !k_enableAllExceptions.get() )
    {
        llog( 20 ) << "Disabled exception handling for buffer index out of bounds exceptions.\n";
        return;
    }

    Function* getBufferSizeFunc = getFunctionOrAssert( m_module, "_ZN4cort14Buffer_getSizeEPNS_14CanonicalStateEt" );
    Function* getBufferAddressFunc =
        getFunctionOrAssert( m_module, "_ZN4cort23Global_getBufferAddressEPNS_14CanonicalStateEt" );

    // Iterate over oclwrrences of @optixi_getBufferElement... and @optixi_setBufferElement...
    for( Function& F : *m_module )
    {
        if( !F.isDeclaration() )
            continue;

        if( GetBufferElement::isIntrinsic( &F ) )
            instrumentBufferAccessFunction<GetBufferElement>( &F, getBufferSizeFunc, getBufferAddressFunc );
        else if( SetBufferElement::isIntrinsic( &F ) )
            instrumentBufferAccessFunction<SetBufferElement>( &F, getBufferSizeFunc, getBufferAddressFunc );
        else
            continue;
    }
}

void RuntimeExceptionInstrumenter::insertMaterialIndexOutOfBoundsHandler()
{
    Function* func  = getFunctionOrAssert( m_module, "_ZN4cort28GeometryInstance_getMaterialEPNS_14CanonicalStateEjj" );
    Function* funcC = m_module->getFunction( "Megakernel_GeometryInstance_getMaterialFromConst" );
    Function* sizeFunc =
        getFunctionOrAssert( m_module, "_ZN4cort32GeometryInstance_getNumMaterialsEPNS_14CanonicalStateEj" );
    RT_ASSERT( !m_isMegakernel || funcC != nullptr );

    Function* funcs[2] = {func, funcC};
    for( Function* F : funcs )
    {
        if( !F )
            continue;

        for( CallInst* call : getCallsToFunction( F ) )
        {
            corelib::CoreIRBuilder builder( call );

            Value* state        = call->getArgOperand( 0 );
            Value* geomInstance = call->getArgOperand( 1 );
            Value* args[2]      = {state, geomInstance};
            Value* numMaterials = builder.CreateCall( sizeFunc, args, "numMaterials" );
            RT_ASSERT( numMaterials->getType()->isIntegerTy( 32 ) );

            Value* idx = call->getArgOperand( 2 );
            RT_ASSERT( idx->getType()->isIntegerTy( 32 ) );

            Value* condition = builder.CreateICmpUGE( idx, numMaterials, "outOfBounds" );

            // Create exception detail:
            // - unused (formerly material address)
            // - size
            // - index
            std::vector<Value*> detail64;
            detail64.push_back( builder.CreatePtrToInt( m_gv, builder.getInt64Ty() ) );  // location not supported
            detail64.push_back( builder.CreateZExt( numMaterials, builder.getInt64Ty() ) );
            detail64.push_back( builder.CreateZExt( idx, builder.getInt64Ty() ) );

            createException( state, condition, RT_EXCEPTION_INDEX_OUT_OF_BOUNDS, call, nullptr, &detail64 );
        }
    }
}

// NOTE: getChildNode() is implemented using Buffer_getElement1dFromId(), which
//       means it will throw a buffer index out of bounds exception if the child
//       is invalid and RT_EXCEPTION_INDEX_OUT_OF_BOUNDS is not enabled but
//       RT_EXCEPTION_BUFFER_INDEX_OUT_OF_BOUNDS is.
void RuntimeExceptionInstrumenter::insertNodeIndexOutOfBoundsHandler()
{
    Function* func     = getFunctionOrAssert( m_module, "_ZN4cort21Selector_getChildNodeEPNS_14CanonicalStateEj" );
    Function* sizeFunc = getFunctionOrAssert( m_module, "_ZN4cort23Selector_getNumChildrenEPNS_14CanonicalStateE" );

    for( CallInst* call : getCallsToFunction( func ) )
    {
        corelib::CoreIRBuilder builder( call );

        // Get the canonical state and the child index.
        Value* state = call->getArgOperand( 0 );
        Value* child = call->getArgOperand( 1 );
        RT_ASSERT( child->getType()->isIntegerTy( 32 ) );

        // Get the number of children.
        Value* numChildren = builder.CreateCall( sizeFunc, state, "numChildren" );
        RT_ASSERT( numChildren->getType()->isIntegerTy( 64 ) );

        // Compare values.
        Value* child64   = builder.CreateZExt( child, builder.getInt64Ty() );
        Value* condition = builder.CreateICmpUGE( child64, numChildren, "outOfBounds" );

        // Create exception detail:
        // - unused (formerly child address)
        // - size
        // - index
        std::vector<Value*> detail64;
        detail64.push_back( builder.CreatePtrToInt( m_gv, builder.getInt64Ty() ) );  // location not supported
        detail64.push_back( numChildren );
        detail64.push_back( child64 );

        createException( state, condition, RT_EXCEPTION_INDEX_OUT_OF_BOUNDS, call, nullptr, &detail64 );
    }
}

// Insert index out of bounds checks for various types of indices, e.g. material
// and node indices. To this end, instrument calls to the following functions:
// - GeometryInstance_getMaterial() ( originates from rtReportIntersection() )
// - Selector_getChildNode() ( originates from rtIntersectChild() )
// - Runtime_intersectNode() ?
// - Runtime_intersectPrimitive() ?
// - Runtime_computePrimitiveAABB() ?
// - GlobalScope_getException() ?
void RuntimeExceptionInstrumenter::insertIndexOutOfBoundsHandler()
{
    if( !Context::getExceptionEnabled( m_exceptionFlags, RT_EXCEPTION_INDEX_OUT_OF_BOUNDS )
        && !k_enableIndexOutOfBoundsHandling.get() && !k_enableAllExceptions.get() )
    {
        llog( 20 ) << "Disabled exception handling for index out of bounds exceptions.\n";
        return;
    }

    insertMaterialIndexOutOfBoundsHandler();
    insertNodeIndexOutOfBoundsHandler();

    // In addition, all get texture calls have to be marked as "throwing".
    // TODO: Check if these are really necessary (for any ES).
    markAsThrowing( "optixi_reportIntersection" );
    markAsThrowing( "optixi_intersectPrimitive" );
}

void RuntimeExceptionInstrumenter::insertIlwalidRayHandler()
{
    if( !Context::getExceptionEnabled( m_exceptionFlags, RT_EXCEPTION_ILWALID_RAY ) && !k_enableIlwalidRayHandling.get()
        && !k_enableAllExceptions.get() )
    {
        llog( 20 ) << "Disabled exception handling for invalid ray exceptions.\n";
        return;
    }

    Function* checkIlwalidRayFunc =
        getFunctionOrAssert( m_module, "_ZN4cort25Exception_checkIlwalidRayEPNS_14CanonicalStateEffffffjff" );

    // For every use of this function, create exception handling code *before* the use.
    for( CallInst* call : getCallsToFunction( checkIlwalidRayFunc ) )
    {
        // Get the state.
        unsigned i       = 0;
        Value*   state   = call->getArgOperand( i++ );
        Value*   ox      = call->getArgOperand( i++ );
        Value*   oy      = call->getArgOperand( i++ );
        Value*   oz      = call->getArgOperand( i++ );
        Value*   dx      = call->getArgOperand( i++ );
        Value*   dy      = call->getArgOperand( i++ );
        Value*   dz      = call->getArgOperand( i++ );
        Value*   rayType = call->getArgOperand( i++ );
        Value*   tmin    = call->getArgOperand( i++ );
        Value*   tmax    = call->getArgOperand( i++ );

        corelib::CoreIRBuilder builder( call );

        Constant* infP = ConstantFP::getInfinity( builder.getFloatTy(), false /* negative */ );
        Constant* infN = ConstantFP::getInfinity( builder.getFloatTy(), true /* negative */ );

        // True if value is NaN or +Inf.
        Value* cmpOX = builder.CreateFCmpUEQ( ox, infP );
        Value* cmpOY = builder.CreateFCmpUEQ( oy, infP );
        Value* cmpOZ = builder.CreateFCmpUEQ( oz, infP );
        Value* cmpDX = builder.CreateFCmpUEQ( dx, infP );
        Value* cmpDY = builder.CreateFCmpUEQ( dy, infP );
        Value* cmpDZ = builder.CreateFCmpUEQ( dz, infP );

        // True if value is NaN or -Inf.
        Value* cmpOXn = builder.CreateFCmpUEQ( ox, infN );
        Value* cmpOYn = builder.CreateFCmpUEQ( oy, infN );
        Value* cmpOZn = builder.CreateFCmpUEQ( oz, infN );
        Value* cmpDXn = builder.CreateFCmpUEQ( dx, infN );
        Value* cmpDYn = builder.CreateFCmpUEQ( dy, infN );
        Value* cmpDZn = builder.CreateFCmpUEQ( dz, infN );

        // True if value is negative.
        Value* cmpRayType = builder.CreateICmpSLT( rayType, builder.getInt32( 0 ) );

        // True if value is NaN.
        Value* cmpTmin = builder.CreateFCmpUNE( tmin, tmin );

        // True if value is NaN.
        Value* cmpTmax = builder.CreateFCmpUNE( tmax, tmax );

        // Reduce result.
        Value* condition = builder.CreateOr( cmpOX, cmpOY );
        condition        = builder.CreateOr( condition, cmpOZ );
        condition        = builder.CreateOr( condition, cmpDX );
        condition        = builder.CreateOr( condition, cmpDY );
        condition        = builder.CreateOr( condition, cmpDZ );
        condition        = builder.CreateOr( condition, cmpDXn );
        condition        = builder.CreateOr( condition, cmpDYn );
        condition        = builder.CreateOr( condition, cmpDZn );
        condition        = builder.CreateOr( condition, cmpOXn );
        condition        = builder.CreateOr( condition, cmpOYn );
        condition        = builder.CreateOr( condition, cmpOZn );
        condition        = builder.CreateOr( condition, cmpRayType );
        condition        = builder.CreateOr( condition, cmpTmin );
        condition        = builder.CreateOr( condition, cmpTmax, "ilwalidRay" );

        // Create exception detail:
        // - origin x/y/z
        // - direction x/y/z
        // - ray type
        // - tmin
        // - tmax
        // NOTE: We use a double array that has the same size as the expected array
        //       of 9 x i64. It is later reinterpreted as such an array instead of
        //       every value being colwerted. rtPrintExceptionDetail() has to make
        //       sure that it reinterprets the values properly again.
        std::vector<Value*> detail;
        detail.push_back( builder.CreateBitCast( ox, builder.getInt32Ty() ) );
        detail.push_back( builder.CreateBitCast( oy, builder.getInt32Ty() ) );
        detail.push_back( builder.CreateBitCast( oz, builder.getInt32Ty() ) );
        detail.push_back( builder.CreateBitCast( dx, builder.getInt32Ty() ) );
        detail.push_back( builder.CreateBitCast( dy, builder.getInt32Ty() ) );
        detail.push_back( builder.CreateBitCast( dz, builder.getInt32Ty() ) );
        detail.push_back( rayType );
        detail.push_back( builder.CreateBitCast( tmin, builder.getInt32Ty() ) );
        detail.push_back( builder.CreateBitCast( tmax, builder.getInt32Ty() ) );

        std::vector<Value*> detail64;
        detail64.push_back( builder.CreatePtrToInt( m_gv, builder.getInt64Ty() ) );  // location not supported

        // Create the exception handling code.
        createException( state, condition, RT_EXCEPTION_ILWALID_RAY, call, &detail, &detail64 );
    }
}

// rtmain used this to report invalid VPCs.
void RuntimeExceptionInstrumenter::insertInternalErrorHandler()
{
    if( !Context::getExceptionEnabled( m_exceptionFlags, RT_EXCEPTION_INTERNAL_ERROR )
        && !k_enableInternalErrorHandling.get() && !k_enableAllExceptions.get() )
    {
        llog( 20 ) << "Disabled exception handling for internal error exceptions.\n";
        return;
    }

    // TODO: Exception handling not implemented for internal error exceptions
}

void RuntimeExceptionInstrumenter::insertUserExceptionHandler()
{
    // optixi_throw contains a call to Exception_setCode(), so all we need to do
    // for this is to mark optixi_throw() as "throwing", which will force creation
    // of exception handling code during insertTryCatch().
    Function* optixiThrowFunc = getFunctionOrAssert( m_module, "optixi_throw" );
    RT_ASSERT( optixiThrowFunc->getFunctionType()->getNumParams() == 2 );
    m_throwingFunctions.insert( optixiThrowFunc );

    for( CallInst* call : getCallsToFunction( optixiThrowFunc ) )
    {
        CoreIRBuilder builder{call};

        Value* state         = call->getArgOperand( 0 );
        Value* exceptionCode = call->getArgOperand( 1 );
        Value* tooSmall = builder.CreateICmpULT( exceptionCode, builder.getInt32( RT_EXCEPTION_USER ), "tooSmall" );
        Value* tooLarge = builder.CreateICmpUGT( exceptionCode, builder.getInt32( RT_EXCEPTION_USER_MAX ), "tooLarge" );
        Value* condition = builder.CreateOr( tooSmall, tooLarge, "ilwalidUserExceptionCode" );

        std::vector<Value*> details( 1 );
        details[0] = exceptionCode;

        std::vector<Value*> details64( 1 );
        details64[0] = builder.CreatePtrToInt( m_gv, builder.getInt64Ty() );  // location not supported

        createException( state, condition, RT_EXCEPTION_USER_EXCEPTION_CODE_OUT_OF_BOUNDS, call, &details, &details64 );
    }
}

// Prevent infinite throw-catch loops due to exception handling code generated
// inside the exception program (where the exception code is set already).
// Create the following code at the beginning of every exception function:
// unsigned int code = Exception_getCode( state );
// Exception_setCode( state, 0 );
// Then replace all uses of optixi_getExceptionCode() by 'code'.
void RuntimeExceptionInstrumenter::insertClearCodeInExceptionPrograms()
{
    Function* getCodeFunc = getFunctionOrAssert( m_module, "optixi_getExceptionCode" );

    // Sanity check: optixi_getExceptionCode() must only be called in exception functions.
    for( CallInst* call : getCallsToFunction( getCodeFunc ) )
    {
        if( !isExceptionFunction( call->getParent()->getParent() ) )
            throw CompileError( RT_EXCEPTION_INFO,
                                "rtGetExceptionCode() can only be called inside exception functions" );
    }

    std::vector<Instruction*> toDelete;
    for( Function* func : m_exceptionFunctions )
    {
        Value*                 state = func->arg_begin();
        BasicBlock::iterator   I     = getSafeInsertionPoint( func );
        corelib::CoreIRBuilder builder( &( *I ) );

        // Create new getCode and setCode calls at the beginning of the function.
        Value* code = createExceptionGetCodeCall( state, builder );
        createExceptionSetCodeCall( state, builder.getInt32( 0 ), builder );

        // Replace all uses of the original code by the one stored before the setCode call.
        for( inst_iterator I2 = inst_begin( func ), E = inst_end( func ); I2 != E; ++I2 )
        {
            CallInst* call = dyn_cast<CallInst>( &*I2 );
            if( !call )
                continue;

            Function* callee = call->getCalledFunction();
            if( !callee || callee != getCodeFunc )
                continue;

            call->replaceAllUsesWith( code );
            toDelete.push_back( call );
        }
    }

    for( Instruction* i : toDelete )
        i->eraseFromParent();
}

bool RuntimeExceptionInstrumenter::mayThrow( Function* function )
{
    return m_throwingFunctions.count( function );
}

// Iterate over all functions that may throw an exception (marked while we were
// inserting exception handling code). At every use of such a function, create
// exception handling code. Transitively mark all functions as "may
// throw" up to (but not including) the top level (state functions).
// NOTE: Exception_throw can technically produce an invalid program id exception
//       itself. Depending on the order in which we generate the exception
//       handling code, this could result in a loop of creating Exception_throw
//       calls to handle exceptions thrown by Exception_throw. Thus, we disable
//       this explicitly.
// NOTE: Along the same lines, we must not create exception handling code inside
//       Exception_throw, since at this point the code has already been set and
//       thus will always throw (and then return an invalid state ID instead of
//       the exception state ID).
// TODO: Sanity-check that every function that may call
//       Exception_setCode() is marked as "throwing"?
void RuntimeExceptionInstrumenter::insertTryCatch()
{
    Function* throwExceptionFunc = getFunctionOrAssert( m_module, "_ZN4cort15Exception_throwEPNS_14CanonicalStateE" );

    // All "throwing" functions require try-catch guards.
    std::vector<Function*> workList;
    workList.insert( workList.end(), m_throwingFunctions.begin(), m_throwingFunctions.end() );

    std::set<Function*> visited;
    while( !workList.empty() )
    {
        Function* throwingFunc = workList.back();
        workList.pop_back();
        RT_ASSERT( !isStateFunction( throwingFunc ) );

        // It can happen that functions that were part of m_throwingFunctions are
        // added to the workList as the parent of another call, so we have to catch
        // that case here instead of using an assertion.
        if( visited.count( throwingFunc ) )
            continue;
        visited.insert( throwingFunc );

        // Don't treat Exception_throw() as a throwing function or we may end up
        // in a loop (or at least multiple nested exception handling routines).
        RT_ASSERT( throwingFunc != throwExceptionFunc );

        for( CallInst* call : getCallsToFunction( throwingFunc ) )
        {
            // Don't create exception handling code inside Exception_throw().
            Function* parentFunc = call->getParent()->getParent();
            if( parentFunc == throwExceptionFunc )
                continue;

            insertTryCatch( call );

            // If the parent of the call is no state function, append it to the work list.
            if( !isStateFunction( parentFunc ) )
                workList.push_back( parentFunc );
        }
    }
}

void RuntimeExceptionInstrumenter::run()
{
    if( !k_enableAllExceptions.get() && ( k_disableAllExceptions.get() || !Context::hasAnyExceptionEnabled( m_exceptionFlags ) ) )
    {
        llog( 20 ) << "All exception handling disabled in context.\n";
        return;
    }

    // We have to treat ilwokeProgram() as "throwing" since we have
    // to unwind the call stack all the way up to the entry state. In MegakernelES
    // this is not necessary, since the stack is cleared at once.
    if( !m_isMegakernel )
    {
        Function* ilwokeFunc =
            getFunctionOrAssert( m_module,
                                 "_ZN4cort21Runtime_ilwokeProgramEPNS_14CanonicalStateEjN5optix12SemanticTypeEjj" );
        m_throwingFunctions.insert( ilwokeFunc );
    }

    // In the exception program, make sure we can not enter infinite exception
    // loops due to the code being set to some value and queried by exception
    // handling code inside the exception program.
    // We do this by simply clearing the code, and when inserting exception
    // handling code in an exception function, return -1 instead of the exception
    // state ID.
    insertClearCodeInExceptionPrograms();

    insertStackOverflowHandler();

    // Check if we need to implement generic exception handling, i.e., initialize
    // state->exception.code with 0 to indicate "no exception thrown", and guard
    // every call that may potentially throw with a check of that value (kind of a
    // try-catch block). If the value is non-zero, return immediately up to the
    // top level: the state function. At that level, create a continuation call
    // with the exception state ID.
    if( Context::hasOnlyStackOverflowEnabled( m_exceptionFlags ) && !k_enableAllExceptions.get() )
        return;

    insertBufferIndexOutOfBoundsHandler();
    insertIndexOutOfBoundsHandler();
    insertIlwalidRayHandler();
    insertInternalErrorHandler();
    insertUserExceptionHandler();

    // Insert these late so we also instrument calls that we added for other
    // exception handling (e.g. BufferIndexOutOfBounds needs to query the size of
    // a buffer - if the buffer id is also invalid, however, this would crash).
    insertBufferIdIlwalidHandler();
    insertTextureIdIlwalidHandler();
    insertProgramIdIlwalidHandler();

    // TODO: Create exception for invalid variable token (prevent a
    // hard-to-trace-back crash due to Runtime_lookupVariableAddress returning nullptr).
    //const char* lookupFnName = "_ZN4cort29Runtime_lookupVariableAddressEPNS_14CanonicalStateEtPc";

    insertTryCatch();
}
