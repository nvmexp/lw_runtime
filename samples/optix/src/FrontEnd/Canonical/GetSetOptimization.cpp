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

#include <FrontEnd/Canonical/GetSetOptimization.h>

#include <corelib/compiler/CoreIRBuilder.h>
#include <corelib/compiler/LLVMUtil.h>
#include <prodlib/exceptions/Assert.h>

#include <llvm/Analysis/CFG.h>  // pred_begin()
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>

using namespace corelib;
using namespace llvm;

static bool isUberPointerSet( const CallInst* call )
{
    if( !call )
        return false;

    const Function* callee = call->getCalledFunction();
    if( !callee )
        return false;

    const StringRef& name = callee->getName();

    // TODO: Which ones can we / do we want to optimize?
    return name.startswith( "optixi_setAttributeValue" ) || name.startswith( "optixi_setBufferElement" )
           //|| name.startswith("optixi_setLwrrentAcceleration")
           || name.startswith( "optixi_setPayloadValue" );
}

static bool isUberPointerGet( const CallInst* call )
{
    if( !call )
        return false;

    const Function* callee = call->getCalledFunction();
    if( !callee )
        return false;

    const StringRef& name = callee->getName();

    // TODO: Which ones can we / do we want to optimize?
    return name.startswith( "optixi_getAttributeValue" ) || name.startswith( "optixi_getVariableValue" )
           || name.startswith( "optixi_getPayloadValue" ) || name.startswith( "optixi_getBufferElement" )
           || name.startswith( "optixi_getBufferSize" ) || name.startswith( "optixi_getLwrrentRay" )
           || name.startswith( "optixi_getLwrrentTmax" ) || name.startswith( "optixi_getLaunchDim" )
           || name.startswith( "optixi_getLaunchIndex" ) || name.startswith( "optixi_getSubframeIndex" )
           //|| name.startswith("optixi_getPrimitiveArgToIntersect")
           //|| name.startswith("optixi_getPrimitiveArgToIntersectAABB")
           //|| name.startswith("optixi_getPrimitiveIndexOffset")
           //|| name.startswith("optixi_getExceptionCode")
           //|| name.startswith("optixi_getExceptionDetail")
           //|| name.startswith("optixi_getExceptionDetail64")
           //|| name.startswith("optixi_getFrameStatus")
           //|| name.startswith("optixi_getState")
           || name.startswith( "optixi_getTexture_" );
}

static bool isSameUberPointerTypeName( const StringRef& strA, const StringRef& strB )
{
    RT_ASSERT( strA.startswith( "optixi_get" ) || strA.startswith( "optixi_set" ) );
    RT_ASSERT( strB.startswith( "optixi_get" ) || strB.startswith( "optixi_set" ) );

    // If the strings match exactly, this is obviously the same type.
    if( strA.equals( strB ) )
        return true;

    // If the first dot is not in the same place, the types are different.
    // If there is no dot, this must be a call to getLaunchDim etc.
    const size_t dotIdxA = strA.find( '.' );
    const size_t dotIdxB = strB.find( '.' );
    if( dotIdxA != dotIdxB )
        return false;

    const bool noDot = dotIdxA == StringRef::npos;

    // If whatever comes between "optixi_get" or "optixi_set" and the first dot
    // does not match, the types are different.
    const StringRef substrA = noDot ? strA.substr( 10 ) : strA.substr( 10, dotIdxA - 10 );
    const StringRef substrB = noDot ? strB.substr( 10 ) : strB.substr( 10, dotIdxB - 10 );
    if( !substrA.equals( substrB ) )
        return false;

    // Finally, if the descriptors behind the first dot do not match, the types
    // are different (which can never be if there is no dot ;) ).
    if( noDot )
        return true;
    const StringRef substrA2 = strA.substr( dotIdxA );
    const StringRef substrB2 = strB.substr( dotIdxB );
    return substrA2.equals( substrB2 );
}

// Find out if two "get" or "set" call sites refer to the same memory location.
// TODO: Reuse code from Compile.cpp for this.
static bool isSameUberPointerType( const CallInst* call, const CallInst* source )
{
    // Make sure the descriptor (mangled into the name) matches.
    const StringRef& name    = call->getCalledFunction()->getName();
    const StringRef& srcName = source->getCalledFunction()->getName();
    if( !isSameUberPointerTypeName( name, srcName ) )
        return false;

    // Make sure the arguments match (except for the last one in case of a get-set
    // comparison).
    // We must not check the argument that is the stored value in case of a "set".
    // TODO: Is it true that there is at most one argument difference?
    //       Is it true that the last operand is always the stored value in case
    //       of a "set"? If any of these assumptions is not true, this will not
    //       work correctly (but will only result in missed optimization
    //       opportunities).
    const int numArgsCall   = call->getNumArgOperands();
    const int numArgsSource = source->getNumArgOperands();
    RT_ASSERT( abs( numArgsCall - numArgsSource ) <= 1 );

    const bool callIsStore   = isUberPointerSet( call );
    const bool sourceIsStore = isUberPointerSet( source );
    RT_ASSERT( callIsStore || isUberPointerGet( call ) );
    RT_ASSERT( sourceIsStore || isUberPointerGet( source ) );

    // If both are "set" operations, they have the same number of arguments, but
    // we must not test the last one. If one operation is not a "set", test as
    // many arguments as that operation has.
    const int numArgsToTest = callIsStore && sourceIsStore ? numArgsCall - 1 : callIsStore ? numArgsSource : numArgsCall;

    // TODO: This should do some structural equivalence testing, not
    //       just rely on the calls to have the exact same arguments. For example,
    //       a value passed to both operations might be rematerialized, resulting
    //       in two different values but the calls still match.
    for( int i = 0, e = numArgsToTest; i < e; ++i )
    {
        if( call->getArgOperand( i ) != source->getArgOperand( i ) )
            return false;
    }

    return true;
}

// Check if 'inst' is a "set" operation.
// If 'source' is not NULL, the "set" also has to match the source, i.e., store
// to the exact same buffer/attribute/etc.
static bool isStore( const Instruction* inst, const CallInst* source )
{
    if( !isa<CallInst>( inst ) )
        return false;

    const CallInst* call = cast<CallInst>( inst );
    if( !isUberPointerSet( call ) )
        return false;

    if( !source )
        return true;

    return isSameUberPointerType( call, source );
}

// Check if 'inst' is a "get" operation.
// If 'source' is not NULL, the "get" also has to match the source, i.e., load
// from the exact same buffer/attribute/etc.
static bool isLoad( const Instruction* inst, const CallInst* source )
{
    if( !isa<CallInst>( inst ) )
        return false;

    const CallInst* call = cast<CallInst>( inst );
    if( !isUberPointerGet( call ) )
        return false;

    if( !source )
        return true;

    return isSameUberPointerType( call, source );
}

static bool isCallSite( const Instruction* inst )
{
    // TODO: Only return true if this is an actual call site.
    //       The current behavior is conservative until we figured out how.

    // Get the function which produces a call site if called.
    //Function* function = inst->getParent()->getParent();
    //Module*   module   = function->getParent();
    //Function* callSiteCallee = module->getFunction("_ZN4cort34CanonicalProgram_callIndirect_stubEPNS_14CanonicalStateEtN5optix12SemanticTypeE");
    //RT_ASSERT_MSG (callSiteCallee, "Unable to find any users of callCanonicalProgram_impl");
    //RT_ASSERT (callSiteCallee->getFunctionType()->getNumParams() == 3);

    // Conservative approach: All calls that are not get/set are treated as call sites.
    const CallInst* call = dyn_cast<CallInst>( inst );
    return isa<CallInst>( inst ) && !isUberPointerGet( call ) && !isUberPointerSet( call );
}

// Scan the block, forward or backward, starting either at 'source' if the block
// is its parent, or at the beginning/end of the block.
// Return the first "set" (if 'lookingForStore') or "get" (if 'lookingForLoad')
// CallInst that matches 'source' (if any). If any instruction that prevents
// optimization is encountered first, return NULL and mark the current path as
// "invalid". Examples are call sites, other "set" operations (if
// 'mustnotCrossOtherStore'), or other "get" operations (if
// 'mustNotCrossOtherLoad'),
// If neither such a bad instruction nor a "set" or "get" is found, return NULL.
static CallInst* scanBlock( BasicBlock* block,
                            CallInst*   source,
                            const bool  scanBackwards,
                            const bool  lookingForStore,
                            const bool  lookingForLoad,
                            const bool  mustNotCrossOtherStore,
                            const bool  mustNotCrossOtherLoad,
                            bool&       ilwalidPath )
{
    // If we scan backwards and start with the first instruction of the block,
    // there's nothing to do. This prevents the following code from falling off
    // the instruction list.
    if( scanBackwards && source == &*block->begin() )
        return nullptr;

    // This code looks a bit ugly, but it allows us to use a loop that either
    // scans the block forward or backward without duplicating code.
    // If we scan the parent block of 'source', scan everything between the
    // instruction and the start or end of the block. Otherwise, scan everything.
    // If we scan backwards, start at the instruction before 'source', otherwise
    // at the instruction behind it.
    Instruction* lwrInst =
        block == &*( source->getParent() ) ?
            ( scanBackwards ? &*( --( BasicBlock::iterator( source ) ) ) : &*( ++( BasicBlock::iterator( source ) ) ) ) :
            ( scanBackwards ? block->getTerminator() : &*block->begin() );

    // If we scan backwards, end at the first instruction of the block, otherwise
    // at the terminator.
    Instruction* lastInst = scanBackwards ? &*block->begin() : block->getTerminator();

    while( true )
    {
        // If this is a store and we are looking for one, the path is valid.
        if( lookingForStore && isStore( lwrInst, source ) )
            return cast<CallInst>( lwrInst );

        // If this is a load and we are looking for one, the path is valid.
        if( lookingForLoad && isLoad( lwrInst, source ) )
            return cast<CallInst>( lwrInst );

        // If this is a store and we must not cross one, the path is invalid.
        if( mustNotCrossOtherStore && isStore( lwrInst, nullptr ) )
        {
            ilwalidPath = true;
            return nullptr;
        }

        // If this is a store and we must not cross one, the path is invalid.
        if( mustNotCrossOtherLoad && isLoad( lwrInst, nullptr ) )
        {
            ilwalidPath = true;
            return nullptr;
        }

        // If this is a call site, the path is invalid.
        if( isCallSite( lwrInst ) )
        {
            ilwalidPath = true;
            return nullptr;
        }

        if( lwrInst == lastInst )
            break;

        if( scanBackwards )
            lwrInst = &*( --( BasicBlock::iterator( lwrInst ) ) );
        else
            lwrInst = &*( ++( BasicBlock::iterator( lwrInst ) ) );
    }

    // We did not find anything in this block that makes the path valid or invalid.
    return nullptr;
}

typedef std::map<BasicBlock*, CallInst*> ReachedMap;
typedef std::vector<CallInst*> ReachedInstVec;

// Return the next reached set/get or NULL if none found or a barrier was hit.
static CallInst* scanDFS( BasicBlock*     block,
                          CallInst*       source,
                          const bool      scanBackwards,
                          const bool      lookingForStore,
                          const bool      lookingForLoad,
                          const bool      mustNotCrossOtherStore,
                          const bool      mustNotCrossOtherLoad,
                          const bool      mustNotHaveEmptyPath,
                          const bool      localOnly,
                          bool&           ilwalidPath,
                          ReachedMap&     reachedMap,
                          ReachedInstVec& reachedInsts )
{
    // Check if we've seen this block already.
    // If so, return the associated set/get.
    ReachedMap::iterator it = reachedMap.find( block );
    if( it != reachedMap.end() )
        return it->second;

    // Scan this block.
    CallInst* inst = scanBlock( block, source, scanBackwards, lookingForStore, lookingForLoad, mustNotCrossOtherStore,
                                mustNotCrossOtherLoad, ilwalidPath );

    // If there's something in that block that prevents optimization, stop immediately.
    if( ilwalidPath )
        return nullptr;

    // Store the mapping of this block to the instruction (or NULL) so we only
    // traverse every edge of the CFG exactly once.
    reachedMap[block] = inst;

    // If we found a set/get in this block, return it.
    // Also, update the list of sets/gets so we later know which ones we reached.
    if( inst )
    {
        reachedInsts.push_back( inst );
        return inst;
    }

    // If there is no predecessor/successor, return NULL and mark the path
    // invalid if required.
    if( ( scanBackwards && pred_begin( block ) == pred_end( block ) )
        || ( !scanBackwards && block->getTerminator()->getNumSuccessors() == 0 ) )
    {
        if( mustNotHaveEmptyPath )
            ilwalidPath = true;
        return nullptr;
    }

    // If we only optimize locally within a basic block, return now.
    if( localOnly )
    {
        ilwalidPath = true;
        return nullptr;
    }

    // Otherwise, keep looking in the predecessors/successors.
    bool hasEmptyPath = false;
    if( scanBackwards )
    {
        for( pred_iterator P = pred_begin( block ), PE = pred_end( block ); P != PE; ++P )
        {
            BasicBlock* predBB = *P;
            CallInst*   predInst =
                scanDFS( predBB, source, scanBackwards, lookingForStore, lookingForLoad, mustNotCrossOtherStore,
                         mustNotCrossOtherLoad, mustNotHaveEmptyPath, localOnly, ilwalidPath, reachedMap, reachedInsts );

            // If there's something on that path that prevents optimization, stop immediately.
            if( ilwalidPath )
                return nullptr;

            // If there's a path from this predecessor without a set/get and this does
            // not prevent optimization (otherwise the path would have been marked as
            // invalid), just remember that there was an empty path.
            if( !predInst )
                hasEmptyPath = true;
        }
    }
    else
    {
        TerminatorInst* ti = block->getTerminator();
        for( int i = 0, e = ti->getNumSuccessors(); i < e; ++i )
        {
            BasicBlock* succBB = ti->getSuccessor( i );
            CallInst*   succInst =
                scanDFS( succBB, source, scanBackwards, lookingForStore, lookingForLoad, mustNotCrossOtherStore,
                         mustNotCrossOtherLoad, mustNotHaveEmptyPath, localOnly, ilwalidPath, reachedMap, reachedInsts );

            // If there's something on that path that prevents optimization, stop immediately.
            if( ilwalidPath )
                return nullptr;

            // If there's a path from this predecessor without a set/get and this does
            // not prevent optimization (otherwise the path would have been marked as
            // invalid), just remember that there was an empty path.
            if( !succInst )
                hasEmptyPath = true;
        }
    }

    // Return something that indicates whether we found an empty path or not.
    return hasEmptyPath ? nullptr : reachedInsts.back();
}

// - forward scan from each set x
// - if on every outgoing path there is another set y and no call site or get
//   is between x and y, remove x
static bool eliminateDeadStoreForward( CallInst* set, const bool localOnly )
{
    ReachedMap     reachedMap;
    ReachedInstVec reachedInsts;

    bool ilwalidPath = false;
    scanDFS( set->getParent(), set, false /* scanBackwards */, true /* lookingForStore */, false /* lookingForLoad */,
             false /* mustNotCrossOtherStore */, true /* mustNotCrossOtherLoad */, true /* mustNotHaveEmptyPath */,
             localOnly, ilwalidPath, reachedMap, reachedInsts );

    if( ilwalidPath )
        return false;

    RT_ASSERT( !reachedInsts.empty() );

    // Remove the dead store.
    set->eraseFromParent();

    return true;
}

static bool isConnectedThroughPhis( const Instruction* inst, const PHINode* phi )
{
    std::vector<const PHINode*> workList;
    workList.push_back( phi );

    while( !workList.empty() )
    {
        const PHINode* lwrPhi = workList.back();
        workList.pop_back();

        for( int i = 0, e = lwrPhi->getNumIncomingValues(); i < e; ++i )
        {
            if( lwrPhi->getIncomingValue( i ) == inst )
                return true;
        }

        for( int i = 0, e = lwrPhi->getNumOperands(); i < e; ++i )
        {
            const PHINode* opPhi = dyn_cast<PHINode>( lwrPhi->getOperand( i ) );
            if( opPhi )
                workList.push_back( opPhi );
        }
    }

    return false;
}

// - backward scan from each set x
// - if the stored value originates from a load y from the same address on
//   every incoming path, remove x if there are no call sites or other sets
//   between x and any y
static bool eliminateDeadStoreBackward( CallInst* set, const bool localOnly )
{
    ReachedMap     reachedMap;
    ReachedInstVec reachedInsts;

    bool ilwalidPath = false;
    scanDFS( set->getParent(), set, true /* scanBackwards */, false /* lookingForStore */, true /* lookingForLoad */,
             true /* mustNotCrossOtherStore */, false /* mustNotCrossOtherLoad */, true /* mustNotHaveEmptyPath */,
             localOnly, ilwalidPath, reachedMap, reachedInsts );

    if( ilwalidPath )
        return false;

    RT_ASSERT( !reachedInsts.empty() );

    // We can only remove the store if the stored value is always loaded by one
    // of the reachedInsts.
    // The stored value is always the last argument of the set.
    if( reachedInsts.size() == 1 )
    {
        CallInst* reached = reachedInsts[0];
        RT_ASSERT( isUberPointerGet( reached ) );
        if( set->getArgOperand( set->getNumArgOperands() - 1 ) == reached )
        {
            // Remove the dead store.
            set->eraseFromParent();
            return true;
        }
    }

    Value*   storedVal = set->getArgOperand( set->getNumArgOperands() - 1 );
    PHINode* storedPhi = dyn_cast<PHINode>( storedVal );

    for( CallInst* reached : reachedInsts )
    {
        if( reached == storedVal )
            continue;

        if( !storedPhi || !isConnectedThroughPhis( reached, storedPhi ) )
            return false;
    }

    // The set has a preceeding get of the same value on all incoming paths,
    // so the store is dead - remove it.
    set->eraseFromParent();
    return true;
}

static bool eliminateDeadStore( CallInst* set, const bool localOnly )
{
    return eliminateDeadStoreForward( set, localOnly ) || eliminateDeadStoreBackward( set, localOnly );
}

// Remove stores that are overwritten:
// st x, p <-
// st y, p
// - sort all sets in topological order (optimize high ones first)
// - forward scan from each set x
// - if on every outgoing path there is another set y and no call site or get
//   is between x and y, remove x
//
// Remove stores that directly store a value we've just loaded back to the
// same address:
// x = ld p
// st x, p  <-
//
// Also follow through phi nodes:
// y0 = ld p           y1 = ld p
//       z = phi(y0,y1)
//       st z, p  <-
// - sort all sets in reverse topological order (optimize low ones first)
// - backward scan from each set x
// - if the stored value originates from a load y from the same address on
//   every incoming path, remove x if there are no call sites or other sets
//   between x and any y
//
// If 'localOnly' is set, we do not do any control-flow dependent optimization.
static int eliminateDeadStores( Function* function, const bool localOnly )
{
    typedef std::vector<BasicBlock*> BlockVec;
    BlockVec                         workList;

    workList.push_back( &function->getEntryBlock() );

    std::set<BasicBlock*> visited;
    int                   storesRemoved = 0;
    while( !workList.empty() )
    {
        BasicBlock* block = workList.back();
        workList.pop_back();

        if( visited.count( block ) )
            continue;
        visited.insert( block );

        std::vector<CallInst*> stores;
        for( BasicBlock::iterator I = block->begin(), IE = block->end(); I != IE; ++I )
        {
            CallInst* store = dyn_cast<CallInst>( I );
            if( !store || !isUberPointerSet( store ) )
                continue;
            stores.push_back( store );
        }
        for( CallInst* store : stores )
        {
            if( eliminateDeadStore( store, localOnly ) )
                ++storesRemoved;
        }

        TerminatorInst* ti = block->getTerminator();
        for( int i = 0, e = ti->getNumSuccessors(); i < e; ++i )
            workList.push_back( ti->getSuccessor( i ) );
    }

    return storesRemoved;
}

static bool eliminateRedundantLoad( CallInst* get, const bool localOnly )
{
    ReachedMap     reachedMap;
    ReachedInstVec reachedInsts;

    bool ilwalidPath = false;
    scanDFS( get->getParent(), get, true /* scanBackwards */, true /* lookingForStore */, true /* lookingForLoad */,
             true /* mustNotCrossOtherStore */, false /* mustNotCrossOtherLoad */, true /* mustNotHaveEmptyPath */,
             localOnly, ilwalidPath, reachedMap, reachedInsts );

    if( ilwalidPath )
        return false;

    RT_ASSERT( !reachedInsts.empty() );

    // Remove the redundant load:
    // - Create alloca
    // - For every "get" in the reached instructions, store its returned value.
    // - For every "set" in the reached instructions, store its value operand.
    // - Load from the alloca and replace the redundant load with that value.
    // - mem2reg will later clean the code up and produce valid SSA.
    AllocaInst* ai =
        corelib::CoreIRBuilder{corelib::getFirstNonAlloca( get->getParent()->getParent() )}.CreateAlloca( get->getType() );

    for( CallInst* reached : reachedInsts )
    {
        RT_ASSERT( isUberPointerSet( reached ) || isUberPointerGet( reached ) );

        // The stored value is always the last argument of the set.
        corelib::CoreIRBuilder irb{corelib::getInstructionAfter( reached )};
        if( isUberPointerSet( reached ) )
            irb.CreateStore( reached->getArgOperand( reached->getNumArgOperands() - 1 ), ai );
        else
            irb.CreateStore( reached, ai );
    }

    LoadInst* li = corelib::CoreIRBuilder{get}.CreateLoad( ai );
    get->replaceAllUsesWith( li );
    get->eraseFromParent();

    return true;
}

// Remove loads for which the loaded value is available already:
// a = ld p
// b = ld p <- replace by a
//
// st a, p
// b = ld p <- replace by a
//
// y0 = ld p       y1 = ld p
//         x = ld p <- replace by x = phi(y0,y1)
//
// st y0, p       st y1, p
//         x = ld p <- replace by x = phi(y0,y1)
//
// y0 = ld p       st y1, p
//         x = ld p <- replace by x = phi(y0,y1)
// - sort all gets in reverse topological order (optimize low ones first)
// - backward scan from each get x
// - if on every incoming path there is another get/set y and no call site or
//   set in between x and y, replace x by an SSA value derived from get-returns,
//   set-operands, and phis that connect them
//
// If 'localOnly' is set, we do not do any control-flow dependent optimization.
static int eliminateRedundantLoads( Function* function, const bool localOnly )
{
    typedef std::vector<BasicBlock*> BlockVec;
    BlockVec                         workList;

    // Initialize work list with blocks ending with return
    for( Function::iterator BB = function->begin(), BBE = function->end(); BB != BBE; ++BB )
    {
        if( isa<ReturnInst>( BB->getTerminator() ) )
            workList.push_back( &*BB );
    }

    std::set<BasicBlock*> visited;
    int                   loadsRemoved = 0;
    while( !workList.empty() )
    {
        BasicBlock* block = workList.back();
        workList.pop_back();
        if( !visited.insert( block ).second )
            continue;

        // Collect loads
        std::vector<CallInst*> loads;
        for( BasicBlock::iterator I = block->begin(), IE = block->end(); I != IE; ++I )
        {
            CallInst* load = dyn_cast<CallInst>( I );
            if( !load || !isUberPointerGet( load ) )
                continue;
            loads.push_back( load );
        }

        // Process loads in reverse order.
        for( std::vector<CallInst *>::reverse_iterator L = loads.rbegin(), LE = loads.rend(); L != LE; ++L )
        {
            if( eliminateRedundantLoad( *L, localOnly ) )
                ++loadsRemoved;
        }

        for( pred_iterator P = pred_begin( block ), PE = pred_end( block ); P != PE; ++P )
        {
            if( !visited.count( *P ) )
                workList.push_back( *P );
        }
    }

    return loadsRemoved;
}

// If 'localOnly' is set, we do not do any control-flow dependent optimization.
void optix::optimizeUberPointerGetsAndSets( llvm::Function* function, const bool localOnly )
{
    //----------------------------------------------------------------------------
    // GENERAL NOTES & IDEAS
    //
    // - have to consider other get/set calls due to possible aliasing
    //   - different types can not alias (e.g. buffer can't alias payload)
    //   - only buffers and payload can alias, attributes and variables can't.
    // - can ignore other loads/stores
    //   - all potentially aliasing ones are transformed into get/set calls
    // - can ignore calls other than call sites
    // - can we do more for get calls from read-only buffers/attributes/...?
    //   - we only know about read-only at planning time
    //   - could analyze code and treat buffers/variables/etc. that are never
    //     written as read-only
    // - have to consider potential increase of register pressure:
    //   a = ...
    //   st a, p // lifetime of "a" ends here
    //   ...     // other code
    //   x = ld p
    //   ... = x
    //   ---------------------
    //   a = ...
    //   st a, p
    //   ...     // other code
    //   ... = a // lifetime of "a" now ends here
    // - if there is a path with a call site and a second path without, we could
    //   optimize the second path, but not before the CFG has been split...
    //   - could duplicate block (and introduce phi in new join block if "get")
    //----------------------------------------------------------------------------

    // Run DSE and RLE until a fixed point is reached.
    int  storesRemoved = 0;
    int  loadsRemoved  = 0;
    bool changed       = true;
    while( changed )
    {
        changed = false;

        // Dead Store Elimination
        const int removedS = eliminateDeadStores( function, localOnly );
        if( removedS )
            changed = true;
        storesRemoved += removedS;

        // Redundant Load Elimination
        const int removedL = eliminateRedundantLoads( function, localOnly );
        if( removedL )
            changed = true;
        loadsRemoved += removedL;
    }

    if( storesRemoved )
    {
        llog( 20 ) << "Removed " << storesRemoved << " \"set\" operation(s).\n";
    }
    if( loadsRemoved )
    {
        llog( 20 ) << "Removed " << loadsRemoved << " \"get\" operation(s).\n";
    }
}
