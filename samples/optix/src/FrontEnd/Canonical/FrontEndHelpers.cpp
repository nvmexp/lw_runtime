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

#include <FrontEnd/Canonical/FrontEndHelpers.h>

#include <Compile/FindAttributeSegments.h>
#include <Context/LLVMManager.h>
#include <FrontEnd/Canonical/IntrinsicsManager.h>
#include <FrontEnd/PTX/DataLayout.h>
#include <Objects/VariableType.h>
#include <corelib/compiler/CoreIRBuilder.h>
#include <corelib/compiler/LLVMUtil.h>
#include <corelib/misc/String.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/CompileError.h>
#include <prodlib/system/Knobs.h>

#include <exp/context/ErrorHandling.h>

#include <llvm/Analysis/Passes.h>  // createTypeBasedAliasAnalysisPass()
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicInst.h>  // MemSetInst
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Utils/Cloning.h>

using namespace corelib;
using namespace prodlib;
using namespace llvm;
using namespace optix;

namespace {
// clang-format off
 Knob<bool>        k_disableTerminateRay      ( RT_DSTRING("compile.disableTerminateRay"),                  false, RT_DSTRING( "Ignore all calls to rtTerminateRay." ) );
 Knob<bool>        k_disableIgnoreIntersection( RT_DSTRING("compile.disableIgnoreIntersection"),            false, RT_DSTRING( "Ignore all calls to rtIgnoreIntersection." ) );
 Knob<bool>        k_cleanupCallableProgramSignatures( RT_DSTRING("c14n.cleanupCallableProgramSignatures"),  true, RT_DSTRING( "Cleanup parameters to callable programs" ) );
 Knob<bool>        k_enableRelwrsiveFunctions ( RT_DSTRING("c14n.enableRelwrsiveFunctions"),                false, RT_DSTRING( "Enable relwrsive calls in input PTX" ) );
 Knob<bool>        k_useContinuationCallables ( RT_DSTRING( "rtx.useContinuationCallables" ),               false, RT_DSTRING( "Use Continuation Callables for all callable programs" ) );

// clang-format on
}  // namespace


// -----------------------------------------------------------------------------
void optix::replaceExitWithReturn( Function* function )
{
    Module* module = function->getParent();

    // Get exit spots
    Function* optix_exit = module->getFunction( "optix.ptx.exit" );
    if( !optix_exit )
        return;

    std::vector<Instruction*> toDelete;
    for( Value::user_iterator ret = optix_exit->user_begin(), e = optix_exit->user_end(); ret != e; ++ret )
    {
        CallSite CS( *ret );
        if( !CS )
            throw CompileError( RT_EXCEPTION_INFO, "Illegal use of exit call" );

        // Check if the exit lies inside the function
        Instruction* I      = CS.getInstruction();
        Function*    parent = I->getParent()->getParent();

        // Make sure there is no exit call inside another function
        if( parent != function )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( *ret ), "Illegal call to exit from device function" );

        // Remove subsequent unreachable instructions and colwert exit to a return
        Instruction* next      = I->getNextNode();
        BasicBlock*  block     = I->getParent();
        Instruction* lastInstr = &block->getInstList().back();
        toDelete.push_back( I );

        RT_ASSERT( next );
        if( next->getOpcode() == Instruction::Unreachable )
        {
            toDelete.push_back( next );
            if( next == lastInstr )
                next = nullptr;  // Expected case
            else
                next = next->getNextNode();
        }

        RT_ASSERT( function->getReturnType()->isVoidTy() );

        // Insert ret before next instruction and split the BB if well formed
        if( next != nullptr && block->getTerminator() && next != lastInstr )
        {
            block->splitBasicBlock( next );
            toDelete.push_back( &block->back() );
        }
        corelib::CoreIRBuilder{block}.CreateRetVoid();

        function->removeFnAttr( Attribute::NoReturn );
    }

    // Security measures to prevent iterator ilwalidation: store all
    // instructions in a vector and then replace all their uses before
    // complete deletion. If these are not properly handled, subsequent
    // GDCE passes will fail.
    Type*  voidType = Type::getVoidTy( module->getContext() );
    Value* undefVal = UndefValue::get( voidType );
    for( Instruction* inst : toDelete )
    {
        inst->replaceAllUsesWith( undefVal );
        inst->eraseFromParent();
    }

    optix_exit->eraseFromParent();
}

// -----------------------------------------------------------------------------
void optix::removeFunctionPointerInitializersOfGlobalVariables( llvm::Module* module )
{
    LLVMContext& context = module->getContext();
    Type*        i64Ty   = Type::getInt64Ty( context );

    for( Module::global_iterator I = module->global_begin(), E = module->global_end(); I != E; ++I )
    {
        // Recognize global variables storing function pointers by initializers which are constants expressions of
        // type i64 with opcode of a PtrToInst instruction.
        GlobalVariable* gv = dyn_cast<GlobalVariable>( I );
        if( !gv )
            continue;
        Constant* c = gv->getInitializer();
        if( !c )
            continue;
        if( c->getType() != i64Ty )
            continue;
        ConstantExpr* ce = dyn_cast<ConstantExpr>( c );
        if( !ce )
            continue;
        unsigned opcode = ce->getOpcode();
        if( opcode != Instruction::PtrToInt )
            continue;
        gv->setInitializer( UndefValue::get( i64Ty ) );
        if( ce->use_empty() )
            ce->destroyConstant();
    }
}

// -----------------------------------------------------------------------------
Function* optix::addStateParameter( Function* func, LLVMManager* llvmManager )
{

    // Update the function signatures to the canonical form
    Type*     statePtrTy = llvmManager->getStatePtrType();
    Function* newFunc    = updateSignatures( statePtrTy, func );

    // Replace the optixi_getState() function for the C14nRuntime functions
    replaceGetState( newFunc->getParent(), statePtrTy );

    return newFunc;
}

// -----------------------------------------------------------------------------
void optix::replaceGetState( Module* module, Type* statePtrTy )
{
    Function* fn = module->getFunction( "optixi_getState" );
    if( !fn )
        return;
    RT_ASSERT( fn->isDeclaration() );

    // The optixi_getState function is used to provide access to the state
    // pointer in the C14n runtime stubs.  The prior "updateSignatures"
    // pass added the state pointer as the first argument to every
    // function.  Change uses of optixi_getState() to use that argument
    // directly.
    std::vector<Value*> toDelete;
    RT_ASSERT( fn->getReturnType() );
    for( CallInst* CI : getCallsToFunction( fn ) )
    {
        if( CI->getType() != statePtrTy )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( CI ), "Invalid use of function optixi_getState()" );
        Value* statePtr = CI->getParent()->getParent()->arg_begin();
        RT_ASSERT( statePtr->getType() == statePtrTy );
        CI->replaceAllUsesWith( statePtr );
        toDelete.push_back( CI );
    }

    removeValues( toDelete );
    RT_ASSERT( fn->use_empty() );
    fn->eraseFromParent();
}

const char* optix::annotationForSemanticType( const SemanticType ST )
{
    // Strings returned here must match those used in in parseLwvmRtAnnotations in coreCompile.cpp
    switch( ST )
    {
        case ST_RAYGEN:
            return "raygen";
        case ST_EXCEPTION:
            return "exception";
        case ST_MISS:
            return "miss";
        case ST_INTERSECTION:
            return "intersection";
        case ST_CLOSEST_HIT:
            return "closesthit";
        case ST_ANY_HIT:
            return "anyhit";
        case ST_BINDLESS_CALLABLE_PROGRAM:
            if( k_useContinuationCallables.get() )
                return "continuationcallable";
            else
                return "directcallable";
        case ST_BOUND_CALLABLE_PROGRAM:
            return "continuationcallable";
        case ST_ATTRIBUTE:
        case ST_NODE_VISIT:
        case ST_BOUNDING_BOX:
        case ST_INTERNAL_AABB_ITERATOR:
        case ST_INTERNAL_AABB_EXCEPTION:
        case ST_ILWALID:
        case ST_INHERITED_HEAVYWEIGHT_BINDLESS_CALLABLE:
            RT_ASSERT_FAIL_MSG( "rtx not available for semantic type" );
    }
    RT_ASSERT_FAIL_MSG( "invalid semantic type" );
}

// -----------------------------------------------------------------------------
void optix::ensureFirstOrderForm( llvm::Module* module )
{
    SmallVector<Function*, 16> workList;

    if( Function* function = module->getFunction( "optixi_isPotentialIntersection" ) )
        workList.push_back( function );
    if( Function* function = module->getFunction( "optixi_reportIntersection" ) )
        workList.push_back( function );

    // Gather the list of functions to be processed by starting with the
    // required functions

    // Add the special trace dispatcher functions
    for( Function& F : *module )
    {
        if( !F.isDeclaration() )
            continue;

        if( SetAttributeValue::isIntrinsic( &F ) || GetAttributeValue::isIntrinsic( &F )
            || TraceGlobalPayloadCall::isIntrinsic( &F ) )
        {
            workList.push_back( &F );
        }
    }

    for( Function* F : workList )
    {
        inlineAllCallersOfFunction( F, true );
    }
}

// -----------------------------------------------------------------------------
void optix::addTypeLinkageFunction( Module* module, LLVMManager* llvmManager )
{
    // Insert a dummy function to help the type mapping. The canonical program has only
    // non-dotted typenames. These will only get mapped to destination types by matching a
    // global. So we place a function in the common runtime called "optixi_linkTypes" that
    // takes all the types that need to be matched as parameters and a corresponding
    // function declaration in the source module for the link.
    //
    // This function needs to match the one found in optixi.cpp, so there is a connection between
    // the runtime and the canonical program.
    FunctionBuilder( module, "optixi_linkTypes" )
        .voidTy()
        .type( llvmManager->getStatePtrType() )
        // Use pointer type, so LLVM doesn't split up the arguments (e.g. float4 -> float,float,...)
        .type( llvmManager->getFloat4Type()->getPointerTo() )
        .build();
}

// -----------------------------------------------------------------------------
static void removeGlobalAddressSpaceModifiers( Function* function, SmallVector<Function*, 4>& stack )
{
    if( std::find( stack.rbegin(), stack.rend(), function ) != stack.rend() )
    {
        std::string functionName( function->getName().data() );
        if( !k_enableRelwrsiveFunctions.get() )
            throw CompileError( RT_EXCEPTION_INFO, "Found relwrsive call to " + functionName );
        llog( 25 ) << "Found relwrsive call to " << functionName << "\n";
        return;
    }

    // First, relwrse into all called functions for which we have the source
    // code. For example, test_Pointers has some cases where the payload
    // is set from within a function.
    for( inst_iterator I = inst_begin( function ), IE = inst_end( function ); I != IE; ++I )
    {
        CallInst* call = dyn_cast<CallInst>( &*I );
        if( !call )
            continue;
        Function* callee = call->getCalledFunction();
        if( !callee || callee->isDeclaration() )
            continue;
        stack.push_back( function );
        removeGlobalAddressSpaceModifiers( callee, stack );
        stack.pop_back();
    }

    // Collect all instructions that return a pointer to global address space.
    std::set<Instruction*> mutateInsts;
    for( inst_iterator I = inst_begin( function ), IE = inst_end( function ); I != IE; ++I )
    {
        Instruction* inst    = &*I;
        Type*        oldType = inst->getType();

        // Ignore instructions that return something that is not a pointer.
        if( !oldType->isPointerTy() )
            continue;

        // Ignore instructions that return a pointer to a non-global address space.
        if( oldType->getPointerAddressSpace() != ADDRESS_SPACE_GLOBAL )
            continue;

        // Ignore instructions that are operands of a call.
        // This oclwrred in fastao_generated_fastambocc.lw.ptx in test_C14n
        // SLOW_Catalog/CanonicalizationFromFileSucceeds.Test/450, where the result
        // of an inttoptr inst is passed to optix.ptx.atom.u32.global.add().
        bool isUsedInCall = false;
        for( Instruction::user_iterator U = inst->user_begin(), UE = inst->user_end(); U != UE; ++U )
        {
            if( !isa<CallInst>( *U ) )
                continue;
            isUsedInCall = true;
            break;
        }
        if( isUsedInCall )
            continue;

        RT_ASSERT_MSG( !isa<CallInst>( inst ), "there must not be any calls that return pointers" );

        mutateInsts.insert( inst );
    }

    // Shrink the set of candidates to those for which we can safely change the
    // type without messing with phis and selects that require matching types.
    // NOTE: This complexity is lwrrently not exercised since we run this pass
    //       after optimizations. If run before, there are cases with dead global
    //       values as incoming values to phis etc.
    bool changed = true;
    while( changed )
    {
        changed = false;
        std::set<Instruction*> ignoreInsts;
        for( Instruction* inst : mutateInsts )
        {
            // Ignore phi instructions that have actual global values incoming. They
            // will be removed later, but we must not create invalid IR by mutating the
            // return type that will not match the operands anymore afterwards.
            if( PHINode* phi = dyn_cast<PHINode>( inst ) )
            {
                bool ignore = false;
                for( int i = 0, e = phi->getNumIncomingValues(); i < e; ++i )
                {
                    Value* val = phi->getIncomingValue( i );
                    if( !isa<GlobalValue>( val ) )
                        continue;
                    ignore = true;
                    break;
                }

                if( ignore )
                {
                    ignoreInsts.insert( phi );
                    changed = true;

                    // Make sure the types of the other incoming values are not changed, too.
                    for( int i = 0, e = phi->getNumIncomingValues(); i < e; ++i )
                    {
                        Value* val = phi->getIncomingValue( i );
                        if( Instruction* valI = dyn_cast<Instruction>( val ) )
                            ignoreInsts.insert( valI );
                    }
                }
            }

            // Ignore selects for which at least one incoming value is an actual global.
            if( SelectInst* selectI = dyn_cast<SelectInst>( inst ) )
            {
                Value* op0 = selectI->getTrueValue();
                Value* op1 = selectI->getFalseValue();

                if( isa<GlobalValue>( op0 ) || isa<GlobalValue>( op1 ) )
                {
                    ignoreInsts.insert( selectI );
                    changed = true;

                    // Make sure the type of the other incoming value is not changed.
                    if( Instruction* op0I = dyn_cast<Instruction>( op0 ) )
                        ignoreInsts.insert( op0I );
                    if( Instruction* op1I = dyn_cast<Instruction>( op1 ) )
                        ignoreInsts.insert( op1I );
                }
            }
        }

        // Remove all ignored instructions from the candidates.
        for( Instruction* inst : ignoreInsts )
        {
            mutateInsts.erase( inst );
        }
    }

    // Mutate the type of all those instructions to return a generic pointer.
    for( Instruction* inst : mutateInsts )
    {
        Type* oldType = inst->getType();
        Type* newType = PointerType::getUnqual( oldType->getPointerElementType() );
        inst->mutateType( newType );

        if( !isa<AddrSpaceCastInst>( inst ) )
            continue;

        // Address space casts may also cast the type. Thus, if we remove the
        // address space, we may have to replace the instruction by a bitcast.
        // If the operand has an address space different from the default,
        // though, we still have to use an addrspacecast, just with a different
        // return type.
        // Note that we did mutate the addrspacecast's type above so that we
        // can safely use replaceAllUsesWith.
        corelib::CoreIRBuilder irb{inst};
        Value*                 op      = inst->getOperand( 0 );
        Value*                 newCast = nullptr;
        if( op->getType()->getPointerAddressSpace() != 0 )
            newCast = irb.CreateAddrSpaceCast( op, newType );
        else
            newCast = irb.CreateBitCast( op, newType );
        inst->replaceAllUsesWith( newCast );
        inst->eraseFromParent();
    }
}

void optix::removeGlobalAddressSpaceModifiers( Function* function )
{
    SmallVector<Function*, 4> stack;
    ::removeGlobalAddressSpaceModifiers( function, stack );
}

// -----------------------------------------------------------------------------
void optix::removeRedundantAddrSpaceCasts( Function* function )
{
    std::vector<Instruction*> workList;

    // Collect all starting instructions, i.e., all uses of allocas.
    for( inst_iterator I = inst_begin( function ), IE = inst_end( function ); I != IE; ++I )
    {
        AllocaInst* allocaInst = dyn_cast<AllocaInst>( &*I );
        if( !allocaInst )
            continue;

        RT_ASSERT_MSG( allocaInst->getType()->getPointerAddressSpace() == 0,
                       "must not have allocations in non-default address space" );

        for( Instruction::user_iterator U = allocaInst->user_begin(), UE = allocaInst->user_end(); U != UE; ++U )
        {
            RT_ASSERT( isa<Instruction>( *U ) );
            workList.push_back( cast<Instruction>( *U ) );
        }
    }

    // To handle phis where multiple incoming values need to be fixed, we
    // use a delay mechanism: We collect such phis and re-insert them
    // into the work list once it is empty and relwrsion would normally stop.
    // We re-iterate them at most 2 times to make sure we also cover some
    // cases with connected phis (to handle this properly the delayed phis
    // would have to be sorted by dependency). Those phis that remain in
    // the delayed set after the second traversal cannot be fixed and so we
    // have to re-introduce addrspacecasts before every operand that was
    // modified.
    SetVector<PHINode*> delayedPhis;
    unsigned int        numDeferredTraversals = 0;

    std::set<Instruction*>    visited;
    std::vector<Instruction*> deleteVec;
    while( !workList.empty() || numDeferredTraversals < 2 )
    {
        if( workList.empty() )
        {
            if( delayedPhis.empty() )
                break;

            workList.insert( workList.begin(), delayedPhis.begin(), delayedPhis.end() );
            for( PHINode* phi : delayedPhis )
                visited.erase( phi );
            delayedPhis.clear();
            ++numDeferredTraversals;
        }

        Instruction* inst = workList.back();
        workList.pop_back();

        if( !visited.insert( inst ).second )
            continue;

        // Collect all uses before we do any replacements.
        std::vector<Instruction*> uses;
        for( Instruction::user_iterator U = inst->user_begin(), UE = inst->user_end(); U != UE; ++U )
        {
            RT_ASSERT( isa<Instruction>( *U ) );
            uses.push_back( cast<Instruction>( *U ) );
        }

        switch( inst->getOpcode() )
        {
            case Instruction::AddrSpaceCast:
            {
                AddrSpaceCastInst* castInst     = cast<AddrSpaceCastInst>( inst );
                Type*              srcTy        = castInst->getSrcTy();
                Type*              dstTy        = castInst->getDestTy();
                const unsigned     srcAddrSpace = srcTy->getPointerAddressSpace();
                const unsigned     dstAddrSpace = dstTy->getPointerAddressSpace();
                RT_ASSERT( srcAddrSpace == 0 );
                Value* op = castInst->getOperand( 0 );

                // If the source and destination types match, the operand must be
                // in address space 0 since we must have touched it before. In such
                // a case, simply replace the addrspacecast by its operand.
                if( srcTy == dstTy )
                {
                    RT_ASSERT( dstAddrSpace == 0 );
                    castInst->replaceAllUsesWith( op );
                }
                else
                {
                    // This instruction casts to a different pointer type, so it has to be
                    // replaced with a bitcast (with the appropriate address space).
                    Type*  newType = PointerType::getUnqual( dstTy->getPointerElementType() );
                    Value* bci     = corelib::CoreIRBuilder{castInst}.CreateBitCast( op, newType );
                    castInst->mutateType( newType );  // This way the replacement will always work.
                    castInst->replaceAllUsesWith( bci );
                }

                // Either way, the addrspacecast is deleted.
                deleteVec.push_back( castInst );
                break;
            }

            case Instruction::IntToPtr:
            {
                IntToPtrInst* castInst = cast<IntToPtrInst>( inst );

                // Replace by new IntToPtrInst without address space if necessary.
                Type*          dstTy        = castInst->getDestTy();
                const unsigned dstAddrSpace = dstTy->getPointerAddressSpace();
                if( dstAddrSpace != 0 )
                {
                    RT_ASSERT( dstAddrSpace == ADDRESS_SPACE_LOCAL );
                    Value* op      = castInst->getOperand( 0 );
                    Type*  newType = PointerType::getUnqual( dstTy->getPointerElementType() );
                    Value* newCast = corelib::CoreIRBuilder{castInst}.CreateIntToPtr( op, newType );
                    castInst->mutateType( newType );  // This way the replacement will always work.
                    castInst->replaceAllUsesWith( newCast );
                    deleteVec.push_back( castInst );
                }
                break;
            }

            case Instruction::BitCast:
            case Instruction::GetElementPtr:
            {
                // BitCasts can be of non-pointer type after ptrtoint etc.
                // Simply relwrse into the uses in such a case.
                Type* oldRetType = inst->getType();
                if( !oldRetType->isPointerTy() )
                    break;

                // Make sure the instruction returns the right type if its operand changed
                // in an earlier iteration.
                if( oldRetType->getPointerAddressSpace() != 0 )
                {
                    RT_ASSERT_MSG( inst->getOperand( 0 )->getType()->getPointerAddressSpace() == 0,
                                   "expected operand to be modified already" );
                    Type* oldElemTy = inst->getType()->getPointerElementType();
                    Type* newType   = PointerType::getUnqual( oldElemTy );
                    inst->mutateType( newType );
                }
                break;
            }

            case Instruction::PHI:
            {
                // PHIs can be of non-pointer type after ptrtoint etc.
                // Simply relwrse into the uses in such a case.
                Type* oldRetType = inst->getType();
                if( !oldRetType->isPointerTy() )
                    break;

                // If the instruction already returns the right type, simply relwrse
                // into the uses.
                if( oldRetType->getPointerAddressSpace() == 0 )
                    break;

                PHINode* phi       = cast<PHINode>( inst );
                Type*    oldElemTy = oldRetType->getPointerElementType();
                Type*    newType   = PointerType::getUnqual( oldElemTy );

                // Otherwise, we have to make sure that all incoming values have the
                // right addrspace. Since this is not easily possible within this
                // relwrsion we re-insert the phi at the end of the work list and do
                // not relwrse into its uses right now.
                bool hasBadOperand = false;
                for( int i = 0, e = phi->getNumIncomingValues(); i < e; ++i )
                {
                    Value* incVal   = phi->getIncomingValue( i );
                    Type*  incValTy = incVal->getType();
                    if( incValTy->isPointerTy() && incValTy->getPointerAddressSpace() == 0 )
                        continue;

                    // If this operand is a constant, we can change it on the fly.
                    if( Constant* incValC = dyn_cast<Constant>( incVal ) )
                    {
                        incValC->mutateType( newType );
                        continue;
                    }

                    hasBadOperand = true;
                    break;
                }

                if( hasBadOperand )
                {
                    // We have to make sure we do not end in an infinite loop when re-
                    // inserting the phi into the work list. Therefore, we use a delay
                    // mechanism for the phi's that we need to take another look at.
                    RT_ASSERT( !delayedPhis.count( phi ) );
                    delayedPhis.insert( phi );
                    continue;
                }

                // We've fixed all incoming values, so now it is time to fix the type
                // of the phi itself. Relwrse into the uses afterwards.
                phi->mutateType( newType );

                break;
            }

            case Instruction::Call:
            {
                // There are some custom cases we need to handle as we come across them.
                if( MemIntrinsic* call = dyn_cast<MemIntrinsic>( inst ) )
                {
                    // 1) memset intrinsics depend on the types of their arguments, so they
                    //    have to be recreated in case an argument type changed
                    //    (delcross_visibility, polyphonic_bake, traversal_backface,
                    //    traversal_test0, traversal_transfers exercise this).
                    Value*       dest       = call->getDest();
                    Value*       val        = call->getArgOperand( 1 );
                    Value*       length     = call->getLength();
                    unsigned int align      = call->getDestAlignment();
                    bool         isVolatile = call->isVolatile();

                    corelib::CoreIRBuilder builder( call );
                    CallInst*              newInst = nullptr;
                    if( isa<MemSetInst>( inst ) )
                        newInst = builder.CreateMemSet( dest, val, length, align, isVolatile );
                    else if( isa<MemCpyInst>( inst ) )
                        newInst = builder.CreateMemCpy( dest, align, val, align, length, isVolatile );
                    else if( isa<MemMoveInst>( inst ) )
                        newInst = builder.CreateMemMove( dest, align, val, align, length, isVolatile );
                    RT_ASSERT( newInst );

                    call->replaceAllUsesWith( newInst );
                    deleteVec.push_back( call );

                    // Relwrse into the uses.
                    break;
                }

                // For all other calls we stop relwrsion.
                continue;
            }

            case Instruction::Load:
            case Instruction::Store:
            case Instruction::Ilwoke:
            case Instruction::Br:
            case Instruction::Ret:
            {
                // Do not relwrse into uses of load/store/ilwoke/branch/ret.
                continue;
            }

            default:
            {
                // Nothing to do for pointer arithmetic etc., only relwrse into uses.

                // Make sure we did not miss any instruction we should handle.
                RT_ASSERT_MSG( !inst->getType()->isPointerTy() || inst->getType()->getPointerAddressSpace() == 0,
                               "unexpected pointer instruction encountered" );

                break;
            }
        }

        workList.insert( workList.end(), uses.begin(), uses.end() );
    }

    // Those phis that remain in the delayed set after the second traversal
    // cannot be fixed and so we have to re-introduce addrspacecasts before
    // every operand that was modified.
    for( PHINode* phi : delayedPhis )
    {
        RT_ASSERT( phi->getType()->isPointerTy() );
        const unsigned int phiAddrSpace = phi->getType()->getPointerAddressSpace();
        for( int i = 0, e = phi->getNumIncomingValues(); i < e; ++i )
        {
            Value* incVal   = phi->getIncomingValue( i );
            Type*  incValTy = incVal->getType();
            RT_ASSERT( incValTy->isPointerTy() && incValTy->getPointerElementType() == phi->getType()->getPointerElementType() );
            if( incValTy->getPointerAddressSpace() != phiAddrSpace )
                continue;

            // All incoming values that we modified above are instructions.
            RT_ASSERT( isa<Instruction>( incVal ) );
            corelib::CoreIRBuilder irb{getInstructionAfter( cast<Instruction>( incVal ) )};
            Value*                 newCast = irb.CreateAddrSpaceCast( incVal, phi->getType() );
            phi->setIncomingValue( i, newCast );
        }
    }

    for( Instruction* inst : deleteVec )
        inst->eraseFromParent();

    // It can be beneficial to run SROA again
    Module*     module = function->getParent();
    legacy::PassManager MPM;
    MPM.add( createSROAPass() );
    MPM.run( *module );

    // Make sure we didn't mess anything up.
    // TODO(Kincaid): Supply a string_ostream for error printouts.
    RT_ASSERT( !verifyFunction( *function ) );
}

// -----------------------------------------------------------------------------
// TODO: Double-check what phases are beneficial.
void optix::optimizeModuleAfterGetSetOpt( Module* module )
{
    legacy::PassManager MPM;
    MPM.add( createTypeBasedAAWrapperPass() );
    MPM.add( createInstructionCombiningPass() );
    MPM.add( createSROAPass() );
    MPM.add( createGVNPass() );
    MPM.add( createInstructionCombiningPass() );
    MPM.add( createDeadCodeEliminationPass() );
    MPM.run( *module );
}

// -----------------------------------------------------------------------------
bool optix::globalVariableIsWritten( const GlobalVariable* value )
{
    SmallVector<const Value*, 4> workList;
    SmallSet<const Value*, 16>   visited;
    workList.push_back( value );
    while( !workList.empty() )
    {
        const Value* V = workList.pop_back_val();

        bool addDependents;
        if( isa<CmpInst>( V ) || isa<LoadInst>( V ) || isa<BranchInst>( V ) || isa<ReturnInst>( V ) )
        {
            // Do not follow these dependencies
            addDependents = false;
        }
        else if( isa<Constant>( V ) || isa<UnaryInstruction>( V ) || isa<PHINode>( V ) || isa<BinaryOperator>( V )
                 || isa<SelectInst>( V ) || isa<InsertValueInst>( V ) )
        {
            addDependents = true;
        }
        else if( const CallInst* CI = dyn_cast<CallInst>( V ) )
        {
            const Function* fn = CI->getCalledFunction();
            if( fn && fn->getName().startswith( "optix.ptx" ) )
            {
                // Fall through below
                addDependents = true;
            }
            else
            {
                // Allow these because usually they are benign, but technically a pointer can escape
                addDependents = false;
            }
        }
        else
        {
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( V ), "Unknown usage of symbol: " + value->getName().str() );
        }

        if( addDependents )
        {
            for( Value::const_user_iterator i = V->user_begin(), e = V->user_end(); i != e; ++i )
            {
                const Value* user = *i;
                if( const StoreInst* SI = dyn_cast<StoreInst>( user ) )
                {
                    if( SI->getPointerOperand() == V )
                        return true;  // used as a pointer
                }
                else
                {
                    if( std::get<1>( visited.insert( user ) ) )
                        workList.push_back( user );
                }
            }
        }
    }
    return false;
}

// -----------------------------------------------------------------------------
void optix::alignUserAllocas( Function* function )
{
    for( inst_iterator I = inst_begin( function ), E = inst_end( function ); I != E; ++I )
    {
        Instruction* inst = &*I;
        if( AllocaInst* allocaInst = dyn_cast<AllocaInst>( inst ) )
        {
            allocaInst->setAlignment( std::max( 16u, allocaInst->getAlignment() ) );
        }
    }
}

// -----------------------------------------------------------------------------
void optix::earlyCheckIntersectionProgram( Function* function )
{
    Module*   module = function->getParent();
    Function* rtPI   = module->getFunction( "_rt_potential_intersection" );
    Function* rtRI   = module->getFunction( "_rt_report_intersection" );

    if( (bool)rtPI != (bool)rtRI )
        throw CompileError( RT_EXCEPTION_INFO,
                            "In intersection programs, all calls to rtPotentialIntersection must be paired with a call "
                            "to "
                            "rtReportIntersection." );

    std::vector<CallInst*> rtPICalls = getCallsToFunction( rtPI );
    std::vector<CallInst*> rtRICalls = getCallsToFunction( rtRI );

    for( CallInst* rtPICall : rtPICalls )
    {
        if( rtPICall->getParent()->getParent() != function )
        {
            throw CompileError( RT_EXCEPTION_INFO, "Only intersection programs can call rtPotentialIntersection." );
        }
    }

    for( CallInst* rtRICall : rtRICalls )
    {
        if( rtRICall->getParent()->getParent() != function )
        {
            throw CompileError( RT_EXCEPTION_INFO, "Only intersection programs can call rtReportIntersection." );
        }
    }

    if( !rtPI )
        return;  // Not an intersection program

    // This throws a compiler exception if the rtPI and rtRI are misused.
    findAttributeSegments( function, rtPICalls, rtRICalls );
}

// -----------------------------------------------------------------------------
static std::vector<CallInst*> getCallsToAttributeFunctions( Function* intersectionFunction )
{
    Module*        module = intersectionFunction->getParent();
    Functiolwector attributeFunctions;
    for( Function& function : *module )
    {
        if( function.getName().startswith( "optixi_getAttributeValue" )
            || function.getName().startswith( "optixi_setAttributeValue" ) )
        {
            attributeFunctions.push_back( &function );
        }
    }

    std::vector<CallInst*> attributeCalls;
    for( Function* function : attributeFunctions )
    {
        std::vector<CallInst*> calls = getCallsToFunction( function, intersectionFunction );
        attributeCalls.insert( attributeCalls.end(), calls.begin(), calls.end() );
    }
    return attributeCalls;
}

// -----------------------------------------------------------------------------
void optix::checkIntersectionProgram( Function* intersectionFunction )
{
    Module*   module = intersectionFunction->getParent();
    Function* rtPI   = module->getFunction( "optixi_isPotentialIntersection" );
    Function* rtRI   = module->getFunction( "optixi_reportIntersection" );

    if( (bool)rtPI != (bool)rtRI )
        throw CompileError( RT_EXCEPTION_INFO,
                            "In intersection programs, all calls to rtPotentialIntersection must be paired with a call "
                            "to "
                            "rtReportIntersection." );

    // Not an intersection progam.
    if( !rtPI )
        return;

    std::vector<CallInst*> attributeCalls;

    PostDominatorTree pDomTree;
    DominatorTree     domTree;
    pDomTree.recallwlate( *intersectionFunction );
    domTree.recallwlate( *intersectionFunction );

    AttributeSegmentVector attributeSegments = findAttributeSegments( intersectionFunction, rtPI, rtRI );
    for( CallInst* callInst : getCallsToAttributeFunctions( intersectionFunction ) )
    {
        bool isValid = attributeSegments.empty();
        for( AttributeSegment& as : attributeSegments )
        {
            if( domTree.dominates( as.rtPI, callInst ) && postdominates( pDomTree, as.rtRI, callInst ) )
                isValid = true;
        }

        if( !isValid )
        {
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( callInst ),
                                "In intersection programs, attributes can be read/written only between calls to "
                                "rtPotentialIntersection and "
                                "rtReportIntersection" );
        }
    }
}

// -----------------------------------------------------------------------------
bool optix::hasMotionIndexArg( Function* boundsFunction )
{
    Module*   module = boundsFunction->getParent();
    Function* rtMI   = module->getFunction( "optixi_getMotionIndexArgToComputeAABB" );

    if( rtMI )
    {
        for( CallInst* rtMICall : getCallsToFunction( rtMI ) )
        {
            if( rtMICall->getParent()->getParent() == boundsFunction )
            {
                return true;
            }
        }
    }
    return false;
}

// -----------------------------------------------------------------------------
namespace {
typedef std::pair<bool, DISubprogram*> DIDFound;
}  // namespace

// Returns a debug info descriptor associated with a function if present
static DIDFound getDIDescriptorIfPresent( Function* targetFn )
{
    DISubprogram* subProgram = targetFn->getSubprogram();
    if( subProgram )
        return DIDFound( true, subProgram );
    else
        return DIDFound( false, nullptr );
}

// -----------------------------------------------------------------------------
static Function* changeFunctionSignature( Function* function, FunctionType* CanonicalSignature )
{
    Module*   module = function->getParent();
    Function* NF     = Function::Create( CanonicalSignature, function->getLinkage() );
    NF->copyAttributesFrom( function );
    module->getFunctionList().push_back( NF );
    NF->takeName( function );

    RT_ASSERT( function->use_empty() );  // We're not expecting anyone to use an RT_PROGRAM.. except what happens next!

    // Actually splice the old function into the new one
    NF->getBasicBlockList().splice( NF->begin(), function->getBasicBlockList() );

    // Transfer the use of the cort argument to the new argument, also transfer the name
    Function::arg_iterator oldCortIt = function->arg_begin();
    Function::arg_iterator newCortIt = NF->arg_begin();

    oldCortIt->replaceAllUsesWith( newCortIt );
    newCortIt->takeName( oldCortIt );

    // Patch the pointer to LLVM function in debug info descriptor if present - this is faster than the DAE pass' one
    // TODO(Kincaid): Is this the right thing to do here?
    NF->setSubprogram( function->getSubprogram() );

    // Now that the old function is completely dead, delete it
    function->eraseFromParent();
    return NF;
}

static inline bool isBoundsProgram( FunctionType* FTy )
{
    const int n = FTy->getNumParams();
    if( n == 3 || n == 4 )
    {
        // required prim_index param
        if( FTy->getParamType( 1 )->isIntegerTy() )
        {
            // optional motion_index param
            if( n == 3 || ( n == 4 && FTy->getParamType( 2 )->isIntegerTy() ) )
            {
                // required pointer param
                if( ( FTy->getFunctionParamType( n - 1 )->isPointerTy()
                      && FTy->getFunctionParamType( n - 1 )->getPointerElementType()->isFloatTy() )
                    || FTy->getFunctionParamType( n - 1 )->isIntegerTy( sizeof( void* ) * 8 ) )
                {
                    return true;
                }
            }
        }
    }
    return false;
}

static inline void replaceIntegerArgWithCortFunction( Function* function, Function::arg_iterator arg, const std::string& cortName, Type* statePtrType )
{
    Module*      module      = function->getParent();
    LLVMContext& llvmContext = function->getContext();

    std::vector<Type*> params;
    params.push_back( statePtrType );
    FunctionType* intGetterTy    = FunctionType::get( Type::getInt32Ty( llvmContext ), params, false );
    Constant*     intGetterConst = module->getOrInsertFunction( cortName, intGetterTy );
    Function*     intGetterFn    = cast<Function>( intGetterConst );
    intGetterFn->setOnlyReadsMemory();
    intGetterFn->setDoesNotThrow();
    Value* args     = function->arg_begin();  // State parameter
    Value* arg0Call = corelib::CoreIRBuilder{&*getSafeInsertionPoint( function )}.CreateCall( intGetterFn, args );
    arg->replaceAllUsesWith( arg0Call );
}

// -----------------------------------------------------------------------------
Function* optix::handleKernelParameters( Function* function, Type* statePtrType )
{
    Module*       module = function->getParent();
    FunctionType* FTy    = function->getFunctionType();
    RT_ASSERT( FTy->getReturnType()->isVoidTy() );        // Required by canonicalization
    RT_ASSERT( FTy->getParamType( 0 ) == statePtrType );  // Required by canonicalization
    LLVMContext& llvmContext = function->getContext();
    if( FTy->getNumParams() == 2 && FTy->getParamType( 1 )->isIntegerTy() )
    {
        // Known signature: intersection program

        Function::arg_iterator arg = function->arg_begin();
        arg++;  // Skip the cort param

        // Replace all uses with the appropriate cort routine
        {
            // int primitiveIndex
            replaceIntegerArgWithCortFunction( function, arg, "optixi_getPrimitiveArgToIntersect", statePtrType );
        }
    }
    else if( isBoundsProgram( FTy ) )
    {
        // Known signature: aabb program

        Function::arg_iterator arg = function->arg_begin();
        arg++;  // Skip the cort param

        // Replace all uses with the appropriate cort routine
        {
            // int aabbPrimitive
            replaceIntegerArgWithCortFunction( function, arg, "optixi_getPrimitiveArgToComputeAABB", statePtrType );
        }

        const bool hasMotionIndex = ( FTy->getNumParams() == 4 );
        if( hasMotionIndex )
        {
            // int motionIndex
            arg++;
            replaceIntegerArgWithCortFunction( function, arg, "optixi_getMotionIndexArgToComputeAABB", statePtrType );
        }

        {
            // float* aabbPtr
            arg++;
            std::string        aaBBPrimitiveGetterName = "optixi_getAABBArgToComputeAABB";
            std::vector<Type*> params;
            params.push_back( statePtrType );
            Type*         ptrToFloat         = Type::getFloatPtrTy( llvmContext );
            FunctionType* intAABBptrGetterTy = FunctionType::get( ptrToFloat, params, false );
            Constant*     intAABBgetConst = module->getOrInsertFunction( aaBBPrimitiveGetterName, intAABBptrGetterTy );
            Function*     intAABBptrGetterFn = cast<Function>( intAABBgetConst );
            intAABBptrGetterFn->setOnlyReadsMemory();
            intAABBptrGetterFn->setDoesNotThrow();
            Value*               args        = function->arg_begin();  // State parameter
            BasicBlock::iterator insertBefore = getSafeInsertionPoint( function );
            Value*               arg0Call = corelib::CoreIRBuilder{&*insertBefore}.CreateCall( intAABBptrGetterFn, args );

            ++insertBefore;
            Value*               arg0Colw =
                corelib::CoreIRBuilder{&*insertBefore}.CreatePtrToInt( arg0Call, arg->getType(), "aabbPtr" );
            arg->replaceAllUsesWith( arg0Colw );

            // Walk the entire dependence tree of the pointer, remove all colwersions
            // to global address space that were introduced by lwcc, and colwert all
            // other dependant operations to use pointers in the generic address space.
            // This is required since the AABB is passed as a pointer to generic
            // memory, but the AABB program is declared as __global__, which makes
            // lwcc generate code that expects the pointer argument to be in global
            // address space, too.
            // NOTE: This is now subsumed by removeGlobalAddressSpaceModifiers.
            //changeAddressSpaceTo( arg0Colw, ADDRESS_SPACE_GENERIC );
        }
    }
    else if( FTy->getNumParams() == 1 )
    {
        return function;  // No need to handle this here
    }
    else
    {
        std::string        sig;
        raw_string_ostream sd( sig );
        sd << FTy;
        sd.flush();
        throw CompileError( RT_EXCEPTION_INFO, "Function signature not recognized: " + sig );
    }

    // Save the previous attributes for the parameter we're interested in
    // Create a new function with the right signature, the same name and correct set of attributes of the previous one
    std::vector<Type*> params;
    params.push_back( statePtrType );
    FunctionType* CanonicalSignature = FunctionType::get( Type::getVoidTy( llvmContext ), params, false );
    Function*     NF                 = changeFunctionSignature( function, CanonicalSignature );
    return NF;
}

static Value* stripCasts( Value* ptr )
{
    if( CastInst* cast = dyn_cast<CastInst>( ptr ) )
    {
        return stripCasts( cast->getOperand( 0 ) );
    }
    return ptr;
}

// -----------------------------------------------------------------------------
OptixResult optix::handleCallableProgramParameters( Function*& function, optix_exp::ErrorDetails& errDetails )
{
    // Optionally skip this optimization for debugging
    if( !k_cleanupCallableProgramSignatures.get() )
        return OPTIX_SUCCESS;

    // Determine the new (cleaned) type of the function
    FunctionType* oldFType = function->getFunctionType();
    FunctionType* newFType = getCleanFunctionType( oldFType );
    if( newFType == oldFType )
        return OPTIX_SUCCESS;

    // Update the function signature and connect the arguments
    Module*   module = function->getParent();
    Function* NF     = Function::Create( newFType, function->getLinkage() );
    NF->copyAttributesFrom( function );
    module->getFunctionList().push_back( NF );
    NF->takeName( function );

    // Splice the old function body into the new one
    NF->getBasicBlockList().splice( NF->begin(), function->getBasicBlockList() );

    // Transfer the use of old arguments and names
    for( Function::arg_iterator oldArg = function->arg_begin(), oldArgE = function->arg_end(), newArg = NF->arg_begin();
         oldArg != oldArgE; ++oldArg, ++newArg )
    {
        newArg->takeName( oldArg );
        if( !NF->isDeclaration() )
        {
            if( oldArg->getType() == newArg->getType() )
            {
                oldArg->replaceAllUsesWith( newArg );
            }
            else
            {
                // Collect all uses of the parameter. Note: This should always be only one (see comment below).
                llvm::SetVector<llvm::Instruction*> uses;
                for( auto useIt = oldArg->user_begin(); useIt != oldArg->user_end(); ++useIt )
                {
                    llvm::Instruction* useI = llvm::dyn_cast<llvm::Instruction>( *useIt );
                    if( !useI )
                    {
                        errDetails.m_compilerFeedback << "Error: Invalid use of argument "
                                                      << std::distance( function->arg_begin(), oldArg )
                                                      << " in callable program " << NF->getName().str() << ".\n";
                        return OPTIX_ERROR_ILWALID_PTX;
                    }
                    if( useI->getParent()->getParent() != NF )
                    {
                        errDetails.m_compilerFeedback
                            << "Error: Argument " << std::distance( function->arg_begin(), oldArg )
                            << " of callable program " << NF->getName().str() << " is used outside the function.\n";
                        return OPTIX_ERROR_ILWALID_PTX;
                    }
                    uses.insert( useI );
                }
                if( uses.size() == 1 && llvm::isa<llvm::StoreInst>( uses[0] ) )
                {
                    // For casting parameter types, the front end inserts allocas, stores to
                    // those and loads again. This is the only use of the original parameter.
                    // Instead of adding a second alloca to cast from the cleaned up type to the original
                    // one, we use the existing alloca and store directly from the cleaned up parameter
                    // into there.
                    llvm::StoreInst*  store  = llvm::cast<llvm::StoreInst>( uses[0] );
                    llvm::AllocaInst* alloca = llvm::cast<llvm::AllocaInst>( stripCasts( store->getPointerOperand() ) );
                    if( alloca )
                    {
                        corelib::CoreIRBuilder irb{store};
                        llvm::Value* castedPtr = irb.CreateBitCast( alloca, newArg->getType()->getPointerTo() );
                        irb.CreateStore( newArg, castedPtr );
                        store->eraseFromParent();
                        uses.clear();
                    }
                    // Else case actually should be an error, I think, but the code below catches if
                    // there is actually a legit case for this.
                }
                if( !uses.empty() )
                {
                    // I don't think this can be reached, but I am not sure (also see comment above).
                    corelib::CoreIRBuilder irb{corelib::getFirstNonAlloca( NF )};
                    llvm::AllocaInst*      alloca  = irb.CreateAlloca( oldArg->getType(), nullptr, "paramCastAlloca" );
                    Value*                 castPtr = irb.CreateBitCast( alloca, newArg->getType()->getPointerTo() );
                    irb.CreateStore( newArg, castPtr );
                    for( llvm::Instruction* useI : uses )
                    {

                        if( llvm::PHINode* PN = llvm::dyn_cast<llvm::PHINode>( useI ) )
                        {
                            unsigned int numIncoming = PN->getNumIncomingValues();
                            for( unsigned int i = 0; i < numIncoming; ++i )
                            {
                                llvm::Value* incoming = PN->getIncomingValue( i );
                                if( incoming == oldArg )
                                {
                                    llvm::BasicBlock*  block        = PN->getIncomingBlock( i );
                                    llvm::Instruction* insertBefore = &block->back();
                                    irb.SetInsertPoint( insertBefore );
                                    llvm::Value* value = irb.CreateLoad( alloca );
                                    useI->replaceUsesOfWith( oldArg, value );
                                }
                            }
                        }
                        else
                        {
                            irb.SetInsertPoint( useI );
                            llvm::Value* value = irb.CreateLoad( alloca );
                            useI->replaceUsesOfWith( oldArg, value );
                        }
                    }
                }
            }
        }
    }

    // Update return instructions if necessary
    Type* newReturnType = newFType->getReturnType();
    if( newReturnType != oldFType->getReturnType() )
    {
        std::vector<Value*> toDelete;
        for( inst_iterator I = inst_begin( NF ), IE = inst_end( NF ); I != IE; ++I )
        {
            ReturnInst* ri = dyn_cast<ReturnInst>( &*I );
            if( !ri )
                continue;

            Value*                 returlwalue = ri->getReturlwalue();
            Value*                 casted      = castThroughAlloca( returlwalue, newReturnType, ri );
            corelib::CoreIRBuilder irb{ri};
            irb.CreateRet( casted );
            toDelete.push_back( ri );
        }
        removeValues( toDelete );
    }

    // Patch the pointer to LLVM function in debug info descriptor if present - this is faster than the DAE pass' one
    NF->setSubprogram( function->getSubprogram() );

    // Move any LWVM metadata from the old to the new function.
    if( llvm::NamedMDNode* lwvmMd = function->getParent()->getNamedMetadata( "lwvm.annotations" ) )
        corelib::replaceMetadataUses( lwvmMd, function, NF );

    // Replace calls.
    for( CallInst* call : corelib::getCallsToFunction( function ) )
    {
        corelib::CoreIRBuilder irb{call};
        unsigned int           argc = call->getNumArgOperands();
        std::vector<Value*>    newArgs;
        for( unsigned int i = 0; i < argc; ++i )
        {
            llvm::Value* oldArg = call->getArgOperand( i );
            llvm::Type*  newTy  = newFType->getParamType( i );
            if( oldArg->getType() == newTy )
            {
                newArgs.push_back( oldArg );
            }
            else
            {
                Value* casted = castThroughAlloca( oldArg, newTy, call );
                newArgs.push_back( casted );
            }
        }
        Value* newValue = irb.CreateCall( NF, newArgs );
        if( newValue->getType() != call->getType() )
        {
            newValue = castThroughAlloca( newValue, call->getType(), call );
        }
        call->replaceAllUsesWith( newValue );
        call->eraseFromParent();
    }

    // Now that the old function is completely dead, delete it
    function->eraseFromParent();
    function = NF;
    return OPTIX_SUCCESS;
}

llvm::Value* optix::castThroughAlloca( Value* srcValue, Type* destType, Instruction* insertBefore )
{
    llvm::Function* inFunction = insertBefore->getParent()->getParent();
    llvm::Type*     srcType    = srcValue->getType();

    llvm::DataLayout       DL( inFunction->getParent()->getDataLayout() );
    corelib::CoreIRBuilder irb{corelib::getFirstNonAlloca( inFunction )};
    llvm::AllocaInst*      alloca = irb.CreateAlloca( destType );
    const unsigned int alignment  = std::max( DL.getPrefTypeAlignment( srcType ), DL.getPrefTypeAlignment( destType ) );
    alloca->setAlignment( alignment );
    irb.SetInsertPoint( insertBefore );
    llvm::ConstantInt* allocaSize = irb.getInt64( DL.getTypeAllocSize( alloca->getAllocatedType() ) );
    irb.CreateLifetimeStart( alloca, allocaSize );
    llvm::Value* castPtr = irb.CreateBitCast( alloca, srcType->getPointerTo() );
    irb.CreateAlignedStore( srcValue, castPtr, alignment );
    llvm::Value* destValue = irb.CreateAlignedLoad( alloca, alignment );
    irb.CreateLifetimeEnd( alloca, allocaSize );
    return destValue;
}

VariableType optix::parseTypename( const std::string& typename_string )
{
    StringRef t( typename_string );
    if( t.startswith( "volatile " ) )
    {
        // We don't care anymore if this is volatile
        t = t.drop_front( 9 );
    }
    if( t.startswith( "optix::" ) )
    {
        // Look for types both in and out of the optix:: namespace.
        t = t.drop_front( 7 );
    }

    // parse a trailing number
    StringRef base   = t;
    size_t    nbegin = t.find_last_not_of( "0123456789" );
    if( nbegin != StringRef::npos )
        nbegin++;
    unsigned int n = 1;
    if( nbegin < t.size() && !t.substr( nbegin ).getAsInteger( 10, n ) )
    {
        if( n == 0 || n > 4 )
        {
            // Vec elements are limited to 1-4
            n = 1;
        }
        else
        {
            base = t.slice( 0, nbegin );
        }
    }

    if( base.equals( "float" ) )
    {
        return VariableType( VariableType::Float, n );
    }
    else if( base == "int" )
    {
        return VariableType( VariableType::Int, n );
    }
    else if( base == "uint" )
    {
        return VariableType( VariableType::Uint, n );
    }
    else if( base == "longlong" )
    {
        return VariableType( VariableType::LongLong, n );
    }
    else if( base == "ulonglong" )
    {
        return VariableType( VariableType::ULongLong, n );
    }
    else if( t == "unsigned" || t == "unsigned int" )
    {
        return VariableType( VariableType::Uint, 1 );
    }
    else if( t == "long long int" || t == "long long" )
    {
        return VariableType( VariableType::LongLong, 1 );
    }
    else if( t == "unsigned long long int" || t == "unsigned long long" )
    {
        return VariableType( VariableType::ULongLong, 1 );
    }
    else if( t == "Ray" )
    {
        return VariableType( VariableType::Ray, 1 );
    }
    else if( t == "rtObject" )
    {
        return VariableType( VariableType::GraphNode, 1 );
    }
    else if( t.startswith( "rtBufferId" ) )
    {
        return VariableType( VariableType::BufferId, 1 );
    }
    else
    {
        // User type is sized in bytes - filled in by caller
        return VariableType( VariableType::UserData, 0 );
    }
}

void optix::canonicalizeTerminateRay( Function* function, Type* statePtrType, ValueVector& toDelete )
{
    if( function->arg_size() != 1 || function->isVarArg() )
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( function ), "Malformed call to " + function->getName().str() );

    Module*      module  = function->getParent();
    LLVMContext& context = module->getContext();

    std::vector<CallInst*> calls = getCallsToFunction( function );
    if( k_disableTerminateRay.get() )
    {
        // Simply remove the call.
        std::for_each( calls.begin(), calls.end(), []( CallInst* CI ) { RT_ASSERT( CI->use_empty() ); } );
        toDelete.insert( toDelete.end(), calls.begin(), calls.end() );
        return;
    }

    // First, replace uses of rtTerminateRay by optixi_terminateRay and a return.
    // Mark every function that may call rtTerminateRay as "may terminate".
    Function* terminateRayFn = module->getFunction( "optixi_terminateRay" );
    RT_ASSERT( terminateRayFn );
    SetVector<Function*> mayTerminateSet;
    for( CallInst* CI : calls )
    {
        // Mark the original call for removal.
        toDelete.push_back( CI );

        // Create a call to optixi_terminateRay.
        BasicBlock* parentBlock = CI->getParent();
        Function*   parentFn    = parentBlock->getParent();
        Value*      statePtr    = parentFn->arg_begin();
        corelib::CoreIRBuilder{CI}.CreateCall( terminateRayFn, statePtr );

        // If the instruction following the call is a return, we are done.
        Instruction* next = getInstructionAfter( CI );
        if( isa<ReturnInst>( next ) )
            continue;

        // Otherwise, split the block and create a new return. We leave the new
        // block and everything that depends on it dead for later removal.
        parentBlock->splitBasicBlock( next );
        parentBlock->getTerminator()->eraseFromParent();

        corelib::CoreIRBuilder irb{parentBlock};
        if( parentFn->getReturnType()->isVoidTy() )
            irb.CreateRetVoid();
        else
            irb.CreateRet( UndefValue::get( parentFn->getReturnType() ) );

        // Mark function as "may terminate".
        mayTerminateSet.insert( parentFn );
    }

    // Second, instrument every call to a function that may terminate with a check
    // and unwind. Build the transitive closure over all functions that may
    // terminate along the way, i.e., mark every function in which we instrument a
    // call as "may terminate", too.
    // TODO: This is not required if we always force inlining of all
    //       levels that contain terminate calls.
    std::vector<Function*> workList( mayTerminateSet.begin(), mayTerminateSet.end() );

    Type*         i1Ty        = Type::getInt1Ty( context );
    Type*         i32Ty       = Type::getInt32Ty( context );
    FunctionType* checkFnTy   = FunctionType::get( i1Ty, statePtrType, false /* isVarArgs */ );
    FunctionType* unwindFnTy  = FunctionType::get( i32Ty, statePtrType, false /* isVarArgs */ );
    Constant*     checkConst  = module->getOrInsertFunction( "optixi_terminateRayRequested", checkFnTy );
    Constant*     unwindConst = module->getOrInsertFunction( "optixi_terminateRayUnwind", unwindFnTy );
    Function*     checkFn     = cast<Function>( checkConst );
    Function*     unwindFn    = cast<Function>( unwindConst );
    RT_ASSERT( checkFn != nullptr );
    RT_ASSERT( unwindFn != nullptr );

    while( !workList.empty() )
    {
        Function* terminatingFunc = workList.back();
        workList.pop_back();

        for( CallInst* CI : getCallsToFunction( terminatingFunc ) )
        {
            // If we've not already marked the parent as "may terminate", do it now.
            BasicBlock* parentBlock = CI->getParent();
            Function*   parentFn    = parentBlock->getParent();
            if( !mayTerminateSet.count( parentFn ) )
            {
                mayTerminateSet.insert( parentFn );
                workList.push_back( parentFn );
            }

            // If the instruction following the call is a return, we don't have to
            // generate the check + unwind since we will return anyway. It is
            // important that the parent function has already been marked as "may
            // terminate" since it still requires instrumentation.
            Instruction* next = getInstructionAfter( CI );
            if( isa<ReturnInst>( next ) )
                continue;

            // Create a call to optixi_terminateRayRequested, a branch, and inside the
            // "then" part a call to optixi_terminateRayUnwind followed by a
            // return.

            // Split the block.
            BasicBlock* noUnwindBlock = parentBlock->splitBasicBlock( next, "noUnwindAfterTerminateRay" );
            BasicBlock* unwindBlock =
                BasicBlock::Create( context, "unwindAfterTerminateRay", noUnwindBlock->getParent(), noUnwindBlock );

            // Modify the new end of the parent block.
            TerminatorInst*        ti = parentBlock->getTerminator();
            corelib::CoreIRBuilder builder( ti );

            // Get the CanonicalState pointer.
            Value* statePtr = parentFn->arg_begin();

            // Cast to the right state type if necessary.
            Type* stateTy = checkFn->getFunctionType()->getParamType( 0 );
            if( statePtr->getType() != checkFn->getFunctionType()->getParamType( 0 ) )
                statePtr = builder.CreateBitCast( statePtr, stateTy );

            // Replace the terminator of the parent block (unconditional branch that
            // was inserted by splitBasicBlock) by a conditional branch.
            Value* unwindRequired = builder.CreateCall( checkFn, statePtr, "unwindRequired" );
            builder.CreateCondBr( unwindRequired, unwindBlock, noUnwindBlock );
            ti->eraseFromParent();

            // Create the call to optixi_terminateRayUnwind.
            builder.SetInsertPoint( unwindBlock );
            builder.CreateCall( unwindFn, statePtr, "" );

            // Create the return. If the function has a non-void return type, simply
            // return an undef value (it won't be used anyway).
            if( parentFn->getReturnType()->isVoidTy() )
                builder.CreateRetVoid();
            else
                builder.CreateRet( UndefValue::get( parentFn->getReturnType() ) );
        }
    }
}

void optix::canonicalizeIgnoreIntersection( Function* function, ValueVector& toDelete )
{
    if( function->arg_size() != 1 || function->isVarArg() )
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( function ), "Malformed call to " + function->getName().str() );

    Module* module = function->getParent();

    std::vector<CallInst*> calls = getCallsToFunction( function );
    if( k_disableIgnoreIntersection.get() )
    {
        // Simply remove the call.
        std::for_each( calls.begin(), calls.end(), []( CallInst* CI ) { RT_ASSERT( CI->use_empty() ); } );
        toDelete.insert( toDelete.end(), calls.begin(), calls.end() );
        return;
    }

    // Replace uses of rtIgnoreIntersection by optixi_ignoreIntersection and a return.
    Function* ignoreFn = module->getFunction( "optixi_ignoreIntersection" );
    RT_ASSERT( ignoreFn != nullptr );
    for( CallInst* CI : calls )
    {
        // Mark the original call for removal.
        toDelete.push_back( CI );

        // Create a call to optixi_ignoreIntersection.
        BasicBlock* parentBlock = CI->getParent();
        Function*   parentFn    = parentBlock->getParent();
        Value*      statePtr    = parentFn->arg_begin();
        corelib::CoreIRBuilder{CI}.CreateCall( ignoreFn, statePtr );

        // If the instruction following the call is a return, we are done.
        Instruction* next = getInstructionAfter( CI );
        if( isa<ReturnInst>( next ) )
            continue;

        // Otherwise, split the block and create a new return. We leave the new
        // block and everything that depends on it dead for later removal.
        parentBlock->splitBasicBlock( next );
        parentBlock->getTerminator()->eraseFromParent();

        // The function is a canonical program, so it returns void.
        RT_ASSERT( parentFn->getReturnType()->isVoidTy() );
        corelib::CoreIRBuilder{parentBlock}.CreateRetVoid();
    }
}

//------------------------------------------------------------------------------
// Clone a function and any dependencies from one module to
// another. Simply mark all global values and functions with internal
// linkage and perform dead code elimination.  For large modules, it
// may be faster to build this directly, but since canonicalization is
// an initialization step we leave that for the future.
Function* optix::makePartialClone( Function* oldFunc, LLVMManager* llvmManager, ValueToValueMapTy* vMap )
{
    Module* newModule =
        vMap ? CloneModule( *oldFunc->getParent(), *vMap ).release() : CloneModule( *oldFunc->getParent() ).release();
    Function* newFunc = newModule->getFunction( oldFunc->getName() );

    // Add references to some different types in the destination module so that
    // the corresponding types from the C14nRuntime source module match up.
    FunctionBuilder( newModule, "optixi_linkAllTypes" )
        .voidTy()
        .type( llvmManager->getStatePtrType() )
        .type( llvmManager->getSize3Type() )
        .type( llvmManager->getUint3Type() )
        .type( llvmManager->getFloat2Type() )
        .type( llvmManager->getFloat3Type() )
        .type( llvmManager->getFloat4Type() )
        .type( llvmManager->getUberPointerType() )
        .type( llvmManager->getBufferHeaderType() )
        .type( llvmManager->getOptixRayType() )
        .build();

    // Link in the runtime library
    Linker  linker( *newModule );
    Module* lib = llvmManager->getC14nRuntime();
    linkOrThrow( linker, lib, true, "Failed to link C14n runtime: " );

    stripAllBut( newFunc, false /*resetCallingColw*/ );

    return newFunc;
}

//------------------------------------------------------------------------------
void optix::makeUserFunctionsNoInline( llvm::Module* module )
{
    for( Module::iterator I = module->begin(), E = module->end(); I != E; ++I )
    {
        Function*         F        = &*I;
        const std::string funcName = F->getName();
        if( F->isDeclaration() || stringBeginsWith( funcName, "_rt_" ) || stringBeginsWith( funcName, "_rti_" )
            || stringBeginsWith( funcName, "optix." ) || stringBeginsWith( funcName, "optixi_" ) )
        {
            continue;
        }

        F->addFnAttr( Attribute::NoInline );
        F->removeFnAttr( Attribute::AlwaysInline );

        llog( 25 ) << "Noinling func: " << funcName << std::endl;
    }
}

//------------------------------------------------------------------------------
Type* optix::getCleanType( const VariableType& vtype, Type* defaultType, Type* optixRayType )
{
    LLVMContext& llvmContext = defaultType->getContext();
    Type*        baseType    = nullptr;
    switch( vtype.baseType() )
    {
        case VariableType::Float:
            baseType = Type::getFloatTy( llvmContext );
            break;
        case VariableType::Int:
            baseType = Type::getInt32Ty( llvmContext );
            break;
        case VariableType::Uint:
            baseType = Type::getInt32Ty( llvmContext );
            break;
        case VariableType::LongLong:
            baseType = Type::getInt64Ty( llvmContext );
            break;
        case VariableType::ULongLong:
            baseType = Type::getInt64Ty( llvmContext );
            break;
        case VariableType::Ray:
            baseType = optixRayType;
            break;
        default:
            return getCleanType( defaultType );
    }
    if( vtype.numElements() == 1 )
        return baseType;
    else
        return VectorType::get( baseType, vtype.numElements() );
}

//------------------------------------------------------------------------------
// TODO: Provide an alignment argument to prevent returning vector types?
//       See comment in canonicalizeComplexFunctions.
Type* optix::getCleanType( Type* type )
{
    LLVMContext& llvmContext = type->getContext();
    // If we have a simple byte array, change it to a word array.
    if( type->isArrayTy() && type->getArrayElementType() == Type::getInt8Ty( llvmContext ) )
    {
        unsigned int numElements = type->getArrayNumElements();
        if( numElements == 1 )
            return Type::getInt8Ty( llvmContext );
        else if( numElements == 2 )
            return Type::getInt16Ty( llvmContext );
        else if( numElements == 4 )
            return Type::getInt32Ty( llvmContext );
        else if( numElements == 8 )
            return Type::getInt64Ty( llvmContext );
        // TODO: The question whether to use arrays or vectors, and for what
        //       sizes, is really one for LWVM... Only using vectors for types
        //       with 128 bits seems to strike a balance for now, but we should
        //       revisit this. Eventually, it would be preferable to get rid of
        //       this "type guessing" alltogether.
        else if( numElements == 16 )
            return VectorType::get( Type::getInt32Ty( llvmContext ), 4 );
        else if( numElements % 4 == 0 )
            return ArrayType::get( Type::getInt32Ty( llvmContext ), numElements / 4 );
        else if( numElements % 2 == 0 )
            return ArrayType::get( Type::getInt16Ty( llvmContext ), numElements / 2 );
    }
    return type;
}

//------------------------------------------------------------------------------
Type* optix::getCleanTypeForArg( Type* type )
{
    LLVMContext& llvmContext = type->getContext();
    // If we have a simple byte array, change it to a word array.
    if( type->isArrayTy() && type->getArrayElementType() == Type::getInt8Ty( llvmContext ) )
    {
        unsigned int numElements = type->getArrayNumElements();
        if( numElements == 1 )
            return Type::getInt8Ty( llvmContext );
        else if( numElements == 2 )
            return type;  // leave as [2 x i8]
        else if( numElements == 4 )
            // If it's an array, don't jam it into a single type, because it is more than likely
            // made up of smaller pieces.
            return ArrayType::get( Type::getInt16Ty( llvmContext ), numElements / 2 );
        else if( numElements % 4 == 0 )
            return ArrayType::get( Type::getInt32Ty( llvmContext ), numElements / 4 );
        else if( numElements % 2 == 0 )
            return ArrayType::get( Type::getInt16Ty( llvmContext ), numElements / 2 );
    }
    return type;
}

//------------------------------------------------------------------------------
// Replace byte array argument types with larger types if possible
FunctionType* optix::getCleanFunctionType( FunctionType* oldType )
{
    // Determine new argument types
    SmallVector<Type*, 10> newParamTypes;
    for( FunctionType::param_iterator P = oldType->param_begin(), PE = oldType->param_end(); P != PE; ++P )
    {
        Type* newType = getCleanTypeForArg( *P );
        newParamTypes.push_back( newType );
    }
    Type* returnType = oldType->getReturnType();
    if( StructType* ST = dyn_cast<StructType>( returnType ) )
    {
        if( ST->getNumElements() == 1 )
        {
            Type* cleanReturnType = getCleanTypeForArg( ST->getElementType( 0 ) );
            returnType            = StructType::get( oldType->getContext(), ArrayRef<Type*>{cleanReturnType} );
        }
    }
    return FunctionType::get( returnType, newParamTypes, oldType->isVarArg() );
}

//-----------------------------------------------------------------------------
void optix::linkOrThrow( Linker& linker, Module* module, bool preserveModule, const std::string& msg )
{
    // TODO(Kincaid): How do we get the error diagnostics from the linker?
    module->setDataLayout( optix::createDataLayoutForLwrrentProcess() );
    std::unique_ptr<llvm::Module> moduleToLink;
    if( preserveModule )
        moduleToLink = std::move( llvm::CloneModule( *module ) );
    else
        moduleToLink.reset( module );
    if( linker.linkInModule( std::move( moduleToLink ), llvm::Linker::StrictMatch ) )
        throw CompileError( RT_EXCEPTION_INFO, msg );
}


//-----------------------------------------------------------------------------
void optix::linkOrThrow( Linker& linker, const Module* module, bool preserveModule, const std::string& msg )
{
    // Cast the module to a non-const version since the LLVM linker does not have
    // a const version.
    Module* nonconstModule = const_cast<Module*>( module );
    linkOrThrow( linker, nonconstModule, preserveModule, msg );
}

//-----------------------------------------------------------------------------
// Get an insertion point in the entry block of the given function, after any alloca instructions.
// Inserting branches after alloca instructions is important, otherwise the LLVM mem2reg
// optimization won't promote stack-allocated variables to registers.
// TODO jbigler - move this to optix/src/Compile after triangle api branch lands.
BasicBlock::iterator optix::getSafeInsertionPoint( Function* function )
{
    BasicBlock::iterator point = function->getEntryBlock().getFirstInsertionPt();
    while( isa<AllocaInst>( *point ) )
        ++point;
    return point;
}
