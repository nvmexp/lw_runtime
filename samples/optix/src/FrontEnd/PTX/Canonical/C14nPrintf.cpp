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

#include <FrontEnd/PTX/Canonical/C14n.h>

#include <Context/LLVMManager.h>
#include <FrontEnd/PTX/PTXFrontEnd.h>
#include <corelib/compiler/LLVMUtil.h>
#include <prodlib/exceptions/CompileError.h>
#include <prodlib/misc/TimeViz.h>

#include <internal/optix_declarations.h>
#include <internal/optix_defines.h>

#include <llvm/ADT/SmallSet.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/ValueMap.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>

using namespace optix;
using namespace llvm;
using namespace corelib;
using namespace prodlib;

// Mapping between a function and call instructions to "_rt_print_active".
typedef std::map<Function*, std::vector<CallInst*>> FuncToCallsMap;
// Map calls to PWF to PWAs.
typedef std::pair<std::pair<CallInst*, int>, std::vector<CallInst*>> PWFPair;
typedef std::vector<PWFPair> PrintWriteMap;

static void removeBlockFromPHIs( BasicBlock* lwrrentBlock, BasicBlock* toRemove );
static int getPrintWriteFormat( CallInst* printWriteCall );
Value* getFormatString( CallInst* printStartCallInst );
static CallInst* isPrintWriteCall( Instruction* inst, const Function* printWrite32Function );
static std::vector<CallInst*> collectPrintWrites( const std::vector<BasicBlock*>& printStartRegionBlocks,
                                                  const Function*                 printWrite32Function,
                                                  std::vector<Value*>&            toDelete );
static std::vector<BasicBlock*> collectBlocksInRegion( BasicBlock* header, BasicBlock* exit );
static BasicBlock* isolatePrintStartRegion( BasicBlock*                     printStartBlock,
                                            BasicBlock*                     exitingBlock,
                                            const std::vector<BasicBlock*>& printStartRegionBlocks );

static void createPrintWriteMap( const std::vector<CallInst*>& printWrites,
                                 LoopInfo&                     loopInfo,
                                 PostDominatorTree&            pDomTree,
                                 PrintWriteMap&                printWriteMap );
static int getExpectedPWAs( int format );
static std::vector<Value*> createAndFillArgBuffer( corelib::CoreIRBuilder& builder,
                                                   const PrintWriteMap&    printWriteMap,
                                                   LoopInfo&               loopInfo,
                                                   const DataLayout&       dataLayout );

#if 0
// For debugging only.
static void dumpPrintWriteMap( const PrintWriteMap& map );
#endif

//------------------------------------------------------------------------------
void C14n::canonicalizeFunctionPrintActive( llvm::Module* module, llvm::Function* printActiveFunction, std::vector<Value*>& toDelete )
{
    if( printActiveFunction->arg_size() != 0 || printActiveFunction->isVarArg() )
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( printActiveFunction ),
                            "Malformed call to " + printActiveFunction->getName().str() );
    m_globalsToRemove.push_back( printActiveFunction );

    LLVMContext&  llvmContext = module->getContext();
    FunctionType* isPrintEnabledType =
        FunctionType::get( Type::getInt32Ty( llvmContext ), m_llvmManager->getStatePtrType(), false );
    Function* isPrintEnabledFunction =
        dyn_cast<Function>( module->getOrInsertFunction( "optixi_isPrintingEnabled", isPrintEnabledType ) );
    RT_ASSERT( isPrintEnabledFunction != nullptr );

    for( Value::user_iterator iter = printActiveFunction->user_begin(), end = printActiveFunction->user_end(); iter != end; ++iter )
    {
        CallInst* printActiveCallInst = dyn_cast<CallInst>( *iter );
        Function* caller              = printActiveCallInst->getParent()->getParent();

        // Replace the call to rt_print_active, with the corresponding canonical function.
        llvm::Value* state = caller->arg_begin();
        llvm::Value* get   = corelib::CoreIRBuilder{printActiveCallInst}.CreateCall( isPrintEnabledFunction, state );

        printActiveCallInst->replaceAllUsesWith( get );
        toDelete.push_back( printActiveCallInst );
    }
}

//------------------------------------------------------------------------------
void C14n::canonicalizeFunctionPrintStart( llvm::Module* module, llvm::Function* printStartFunction, std::vector<Value*>& toDelete )
{
    TIMEVIZ_FUNC;

    // This function transforms calls to rt_print_start and to rt_print_write into a call to rt_printf.
    // We do that by identifying the region of the CFG controlled by the rt_print_start.
    // All the rt_print_write in the region belong to the same rtPrintf statement.

    RT_ASSERT( printStartFunction != nullptr );
    m_globalsToRemove.push_back( printStartFunction );

    LLVMContext&     llvmContext = module->getContext();
    llvm::DataLayout dataLayout( module );
    Type*            i32Ty          = Type::getInt32Ty( llvmContext );
    Type*            paramTypes[1]  = {Type::getInt8PtrTy( llvmContext )};
    FunctionType*    printfType     = FunctionType::get( i32Ty, paramTypes, true );
    Function*        printfFunction = dyn_cast<Function>( module->getOrInsertFunction( "rt_printf", printfType ) );
    RT_ASSERT( printfFunction != nullptr );
    PostDominatorTree pDomTree;
    DominatorTree     domTree;
    Function*         printWrite32Function = module->getFunction( "_rt_print_write32" );

    // Group the ilwocations to rt_print_active on a per-function basis.
    FuncToCallsMap callsPerFunction;
    for( auto iter = printStartFunction->user_begin(), end = printStartFunction->user_end(); iter != end; ++iter )
    {
        CallInst* printStartCallInst = dyn_cast<CallInst>( *iter );
        if( !printStartCallInst )
            throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( printStartCallInst ),
                                "Invalid use of function: " + printStartFunction->getName().str() );
        RT_ASSERT( printStartCallInst->getCalledFunction() == printStartFunction );

        Function* caller = printStartCallInst->getParent()->getParent();
        // The [] map operator returns an empty vector when caller is not in the map.
        // This allows not to initialize the entry of the maps.
        callsPerFunction[caller].push_back( printStartCallInst );
    }

    // Go over all the functions that contain calls to rt_print_start.
    for( auto& iter : callsPerFunction )
    {
        Function*               caller    = iter.first;
        std::vector<CallInst*>& callInsts = iter.second;

        // Go over the calls to print_start.
        for( const auto& printStartCallInst : callInsts )
        {
            // Keep dom tree and pdom tree up-to-date while modifying the cfg.
            // This might be expensive, revisit if it becomes a problem.
            pDomTree.recallwlate( *caller );
            domTree.recallwlate( *caller );
            LoopInfo loopInfo( domTree );

            BasicBlock* printStartBlock = printStartCallInst->getParent();
            BasicBlock* exitingBlock    = pDomTree.getNode( printStartBlock )->getIDom()->getBlock();
            Value*      formatString    = getFormatString( printStartCallInst );

            TerminatorInst* printStartBlockTI = printStartBlock->getTerminator();
            RT_ASSERT( isa<BranchInst>( printStartBlockTI ) );
            // The printf does not have any arguments.
            if( !cast<BranchInst>( printStartBlockTI )->isConditional() )
            {
                RT_ASSERT( printStartBlockTI->getSuccessor( 0 ) == exitingBlock );

                // Call the rt_printf function.  Do not replace uses of the return value.
                Value* rtPrintfArgs = formatString;
                corelib::CoreIRBuilder{printStartCallInst}.CreateCall( printfFunction, rtPrintfArgs );
                toDelete.push_back( printStartCallInst );

                continue;
            }

            BasicBlock* firstWriteBlock = printStartBlockTI->getSuccessor( 0 );
            if( firstWriteBlock == exitingBlock )
                firstWriteBlock = printStartBlockTI->getSuccessor( 1 );

            // Collect all "C" blocks in the order that they are exelwted.
            // There may still be loops in that code, but these are guaranteed to only
            // ever execute their body once (optimization only failed to remove them).
            std::vector<BasicBlock*> printStartRegionBlocks = collectBlocksInRegion( printStartBlock, exitingBlock );

            // Build arguments for printf by finding the related calls to
            // rt_print_write32 in all "C" blocks.

            // TODO: we want write32s to be sorted top-down.
            // If we find a sample in which collectBlocksInRegion return blocks in the wrong order we
            // have to sort write32s.
            // This can be done using the normal std::sort function using a custom comparison that relies on domination and
            // post-domination.
            // In particular: if A dom B -> A < B
            //                if B pdom A -> A < B
            // These two rules should cover all the cases that we can see in the print_start region.

            std::vector<CallInst*> printWrites = collectPrintWrites( printStartRegionBlocks, printWrite32Function, toDelete );
            toDelete.push_back( printStartCallInst );

            BasicBlock* printfBlock = isolatePrintStartRegion( printStartBlock, exitingBlock, printStartRegionBlocks );
            RT_ASSERT( printfBlock );

            // There is a conditional branch from printStartBlock to printfBlock / firstWriteBlock.
            // Replace it with an unconditional one to firstWrite.
            corelib::CoreIRBuilder{printStartBlock}.CreateBr( firstWriteBlock );
            printStartBlockTI->eraseFromParent();
            // Update PHI nodes in printfBlock.
            removeBlockFromPHIs( printfBlock, printStartBlock );

            // rt_print_write32 calls can have 2 different meanings:
            // 1) define the format of the argument that is going to be written, called PWF.
            // 2) actually write the argument, called PWA.
            // We first have to identify which calls are PWF and the corresponding PWAs.
            PrintWriteMap printWriteMap;
            createPrintWriteMap( printWrites, loopInfo, pDomTree, printWriteMap );

            corelib::CoreIRBuilder builder( &*printStartBlock->getParent()->getEntryBlock().begin() );
            auto rtPrintfArgsBuffers = createAndFillArgBuffer( builder, printWriteMap, loopInfo, dataLayout );
            // Create the actual call to rt_printf in the printf block.
            builder.SetInsertPoint( printfBlock->getTerminator() );
            // Call the function. Do not replace uses of the return value.
            std::vector<Value*> rtPrintfArgs = {formatString};

            for( const auto& buffer : rtPrintfArgsBuffers )
            {
                Value* arg = builder.CreateLoad( buffer, "" );
                rtPrintfArgs.push_back( arg );
            }

            builder.CreateCall( printfFunction, rtPrintfArgs );
        }

        // We do not update dominator information.
        // This is because the modifications done to the CFG are local to a region only.
        // Subsequent queries to the dominator are not going to touch the same region again.
    }
}

//------------------------------------------------------------------------------
#if 0
// For debugging only.
void dumpPrintWriteMap( const PrintWriteMap& map )
{
    for( const auto& mI : map )
    {
        errs() << "PWF: " << *mI.first.first << "-" << mI.first.second << " ";
        std::vector<CallInst*> pwaCalls = mI.second;
        errs() << "PWAs:\n";
        for( CallInst* call : pwaCalls )
        {
            call->dump();
        }
        errs() << "--------------------------------\n";
    }
}
#endif

//------------------------------------------------------------------------------
static void removeBlockFromPHIs( BasicBlock* lwrrentBlock, BasicBlock* toRemove )
{
    for( BasicBlock::iterator iter = lwrrentBlock->begin(); isa<PHINode>( iter ); ++iter )
    {
        PHINode* node = dyn_cast<PHINode>( iter );
        node->removeIncomingValue( toRemove );
    }
}

//------------------------------------------------------------------------------
static CallInst* isPrintWriteCall( Instruction* inst, const Function* printWrite32Function )
{
    if( CallInst* callInst = dyn_cast<CallInst>( inst ) )
    {
        if( callInst->getCalledFunction() == printWrite32Function )
        {
            return callInst;
        }
    }
    return nullptr;
}

//------------------------------------------------------------------------------
std::vector<CallInst*> collectPrintWrites( const std::vector<BasicBlock*>& printStartRegionBlocks,
                                           const Function*                 printWrite32Function,
                                           std::vector<Value*>&            toDelete )
{
    std::vector<CallInst*> printWrites;
    for( BasicBlock* block : printStartRegionBlocks )
    {
        for( Instruction& inst : *block )
        {
            if( CallInst* printWriteCall = isPrintWriteCall( &inst, printWrite32Function ) )
            {
                printWrites.push_back( printWriteCall );
                toDelete.push_back( printWriteCall );
            }
        }
    }
    return printWrites;
}

//------------------------------------------------------------------------------
// 0 = 32 bit integer value
// 1 = 64 bit integer value
// 2 = 32 bit float value
// 3 = 64 bit double value
static int getPrintWriteFormat( CallInst* printWriteCall )
{
    Value* formatValue      = printWriteCall->getArgOperand( 0 );
    int    argumentFormat   = -1;
    bool   colwertionResult = getConstantValue( formatValue, argumentFormat );
    RT_ASSERT_MSG( colwertionResult, "Expected: " + printWriteCall->getName().str() + " to have constant argument." );
    RT_ASSERT_MSG( 0 <= argumentFormat && argumentFormat <= 3, "Argument format value should be between 0 and 3." );
    return argumentFormat;
}

//------------------------------------------------------------------------------
// Collect all blocks between 'block' and 'endBlock' in topological order.
// This code assumes that 'endBlock' post dominates 'block'.
// This code assumes that the only cycles in that region are self-edges.
static void collectBlocksInRegionImpl( BasicBlock* block, BasicBlock* exit, std::vector<BasicBlock*>& outputBlocks, std::set<BasicBlock*>& visited )
{
    // If we found the end block, return without adding it.
    if( block == exit )
        return;

    if( !visited.insert( block ).second )
        return;

    // Relwrse into successor blocks (ignore self edges).
    TerminatorInst* ti = block->getTerminator();
    for( int i = 0, e = ti->getNumSuccessors(); i < e; ++i )
    {
        BasicBlock* succBB = ti->getSuccessor( i );
        if( succBB == block )
            continue;
        collectBlocksInRegionImpl( succBB, exit, outputBlocks, visited );
    }

    // Append to list (reversed topological order).
    outputBlocks.push_back( block );
}

//------------------------------------------------------------------------------
static std::vector<BasicBlock*> collectBlocksInRegion( BasicBlock* header, BasicBlock* exit )
{
    std::vector<BasicBlock*> outputBlocks;
    std::set<BasicBlock*>    visited;
    collectBlocksInRegionImpl( header, exit, outputBlocks, visited );
    // Reverse the topological order.
    std::reverse( outputBlocks.begin(), outputBlocks.end() );
    return outputBlocks;
}

void visitBasicBlock( BasicBlock& BB );

//------------------------------------------------------------------------------
// Split the critical edge from printStartBlock to exiting.
// This will create a new exiting block for all the brances in the current region.
// We will place the vprint in this block.
BasicBlock* isolatePrintStartRegion( BasicBlock* printStartBlock, BasicBlock* exitingBlock, const std::vector<BasicBlock*>& printStartRegionBlocks )
{
    TerminatorInst* printStartTI = printStartBlock->getTerminator();
    RT_ASSERT( printStartTI->getSuccessor( 0 ) == exitingBlock );

    BasicBlock* printfBlock = SplitCriticalEdge( printStartTI, 0 );
    if( !printfBlock )
    {
        throw CompileError( RT_EXCEPTION_INFO, LLVMErrorInfo( printStartTI ),
                            "Cannot split critical edge for _rt_print_start block" );
    }
    printfBlock->setName( "rt_printf.block" );

    llvm::ValueMap<llvm::Value*, llvm::Value*> phiMap;
    // Add redundat phi nodes to vprintfblock.
    for( BasicBlock::iterator iter = exitingBlock->begin(); isa<PHINode>( iter ); ++iter )
    {
        PHINode* phiNode = cast<PHINode>( iter );

        unsigned int vprintIndex   = phiNode->getBasicBlockIndex( printfBlock );
        Value*       incomingValue = phiNode->getIncomingValue( vprintIndex );
        PHINode*     newPhi = corelib::CoreIRBuilder{&*printfBlock->begin()}.CreatePHI( incomingValue->getType(), 1,
                                                                                  "rt_printf." + incomingValue->getName() );
        newPhi->addIncoming( incomingValue, printStartBlock );
        phiNode->setIncomingValue( vprintIndex, newPhi );
        phiMap[phiNode] = newPhi;
    }

    // Replace branches to exiting with branches to printfBlock.
    for( BasicBlock* block : printStartRegionBlocks )
    {
        TerminatorInst* terminator = block->getTerminator();
        int             children   = terminator->getNumSuccessors();
        for( int child = 0; child < children; ++child )
        {
            if( terminator->getSuccessor( child ) != exitingBlock )
                continue;

            terminator->setSuccessor( child, printfBlock );
            // Update the PHI nodes in exiting and in vprintf block.
            for( BasicBlock::iterator iter = exitingBlock->begin(); isa<PHINode>( iter ); ++iter )
            {
                PHINode* phiNode = cast<PHINode>( iter );
                PHINode* newPhi  = cast<PHINode>( phiMap[phiNode] );

                // Add the incoming value to the phi node in vprintf and remove it from the one in exiting.
                unsigned int blockIndex    = phiNode->getBasicBlockIndex( block );
                Value*       incomingValue = phiNode->getIncomingValue( blockIndex );
                newPhi->addIncoming( incomingValue, block );
                phiNode->removeIncomingValue( blockIndex, true );
            }
        }
    }

    return printfBlock;
}

//------------------------------------------------------------------------------
static int getExpectedPWAs( int format )
{
    if( format == 0 || format == 2 )
        return 1;
    else
        return 2;
}

//------------------------------------------------------------------------------
Value* getGlobalPointer( Value* value )
{
    while( ConstantExpr* constExpr = dyn_cast<ConstantExpr>( value ) )
    {
        if( constExpr->getOpcode() == Instruction::PtrToInt || constExpr->getOpcode() == Instruction::IntToPtr
            || constExpr->getOpcode() == Instruction::AddrSpaceCast )
        {
            value = constExpr->getOperand( 0 );
        }
    }
    return value;
}

//------------------------------------------------------------------------------
// This function traslates the format string given in input to print_start into a format string suitable for rt_printf
// and vprintf. In particular vprintf requires the format string to be allocate in the global address space.
// Older implementation of LWCC might allocate format strings into the constant address space,  this makes the call to vprintf crash.
// So, in here we detect if the format string given in input to the print_start is allocated into the constant address space,
// if yes we create a new global variable allocate in the global address space.
// We do not touch the original global variable, it should be optimized away.
// The identification of the global variable is very strict and not flexible, we might have to revisit this
// if we encounter other patterns in old traces or in user-generated PTX code.
Value* getFormatString( CallInst* printStartCallInst )
{
    Value*  formatString = printStartCallInst->getOperand( 0 );
    Module* module       = printStartCallInst->getParent()->getParent()->getParent();

    corelib::CoreIRBuilder irb{printStartCallInst};
    if( !isa<ConstantExpr>( formatString ) )
        return irb.CreateIntToPtr( formatString, irb.getInt8PtrTy() );

    Value* ptr = getGlobalPointer( formatString );
    if( GlobalVariable* global = dyn_cast<GlobalVariable>( ptr ) )
    {
        Type* type = global->getType();
        if( type->isPointerTy() && type->getPointerAddressSpace() == ADDRESS_SPACE_CONST )
        {
            // Create a new global variable, this is a clone of the input one, but it lives in global address space.
            GlobalVariable* newGlobal =
                new GlobalVariable( *module, global->getType()->getElementType(), global->isConstant(),
                                    global->getLinkage(), global->getInitializer(), global->getName(), nullptr,
                                    global->getThreadLocalMode(), ADDRESS_SPACE_GLOBAL );
            Constant* result = ConstantExpr::getAddrSpaceCast( newGlobal, irb.getInt8PtrTy() );
            return result;
        }
    }

    // TODO: Not sure about what to do here. I have never seen a case that trigger this code path.
    return irb.CreateIntToPtr( formatString, irb.getInt8PtrTy() );
}

//------------------------------------------------------------------------------
static void createPrintWriteMap( const std::vector<CallInst*>& printWrites, LoopInfo& loopInfo, PostDominatorTree& pDomTree, PrintWriteMap& printWriteMap )
{
    printWriteMap.clear();
    CallInst*   lwrrentPWF = printWrites[0];
    BasicBlock* pDom       = pDomTree.getNode( lwrrentPWF->getParent() )->getIDom()->getBlock();
    // expectedPWAs is the number of PWAs that are controlled by the current PWF if the PWAs belong to the same
    // block of the current PWF. This is 2 if the argument is 64 bits, 1 if its 32.
    // ExpectedPWAs is releveant in the case in which the loop has been completelly unrolled and the blocks have
    // been merged into 1.
    int     format       = getPrintWriteFormat( lwrrentPWF );
    int     expectedPWAs = getExpectedPWAs( format );
    PWFPair newPair      = std::make_pair( std::make_pair( lwrrentPWF, format ), std::vector<CallInst*>() );
    auto    lwrrentPair  = printWriteMap.insert( printWriteMap.end(), newPair );

    int index = 0;

    while( index < static_cast<int>( printWrites.size() ) - 1 )
    {
        CallInst*   pw      = printWrites[++index];
        BasicBlock* pwBlock = pw->getParent();
        // If the PWF and the current PWA are in the same block.
        // This is for a situation like the following:
        // ------------
        // |   PWF    |
        // |   PWA    |
        // |   PWA    |
        // |   PWF    |
        // |   PWA    |
        // ------------
        // i.e., multiple PWs are in the same block.
        if( pwBlock == lwrrentPWF->getParent() )
        {
            if( expectedPWAs > 0 )
            {
                // Insert the current PWA under the current PWF.
                lwrrentPair->second.push_back( pw );
                --expectedPWAs;
            }
            else if( expectedPWAs == 0 )
            {
                // We are done collecting PWAs for the current PWF.
                // The current pw is a PWF.
                lwrrentPWF       = pw;
                pDom             = pDomTree.getNode( lwrrentPWF->getParent() )->getIDom()->getBlock();
                int printFormat  = getPrintWriteFormat( lwrrentPWF );
                expectedPWAs     = getExpectedPWAs( printFormat );
                PWFPair newPair2 = std::make_pair( std::make_pair( lwrrentPWF, printFormat ), std::vector<CallInst*>() );
                lwrrentPair      = printWriteMap.insert( printWriteMap.end(), newPair2 );
            }
            else
            {
                // expectedPWAs < 0
                RT_ASSERT_MSG(
                    false, "A print_write that defines the print argument controls more print_writes than expected." );
            }
        }
        else
        {
            Loop* pwLoop   = loopInfo.getLoopFor( lwrrentPWF->getParent() );
            Loop* pDomLoop = loopInfo.getLoopFor( pDom );

            if( pwBlock == pDom )
            {
                if( pwLoop == pDomLoop )
                {
                    // This is for a situation where the PWAs are in different blocks than then PWF.
                    // We are are in the post dominator of the current pwf.
                    // We assume that this means that this pw is actually a PWF.
                    lwrrentPWF      = pw;
                    pDom            = pDomTree.getNode( lwrrentPWF->getParent() )->getIDom()->getBlock();
                    int printFormat = getPrintWriteFormat( lwrrentPWF );
                    expectedPWAs    = getExpectedPWAs( printFormat );
                    PWFPair newPair2 = std::make_pair( std::make_pair( lwrrentPWF, printFormat ), std::vector<CallInst*>() );
                    lwrrentPair      = printWriteMap.insert( printWriteMap.end(), newPair2 );
                }
                else
                {
                    lwrrentPair->second.push_back( pw );
                    // expectedPWAs is irrelevant now.
                    expectedPWAs = 0;

                    if( index == static_cast<int>( printWrites.size() ) - 1 )
                    {
                        break;
                    }

                    lwrrentPWF      = printWrites[++index];
                    pDom            = pDomTree.getNode( lwrrentPWF->getParent() )->getIDom()->getBlock();
                    int printFormat = getPrintWriteFormat( lwrrentPWF );
                    expectedPWAs    = getExpectedPWAs( printFormat );
                    PWFPair newPair2 = std::make_pair( std::make_pair( lwrrentPWF, printFormat ), std::vector<CallInst*>() );
                    lwrrentPair      = printWriteMap.insert( printWriteMap.end(), newPair2 );
                }
            }
            else  // We are in a block controlled by the current PWF.
            {
                lwrrentPair->second.push_back( pw );
                // expectedPWAs is irrelevant now.
                expectedPWAs = 0;
            }
        }
    }
}

//------------------------------------------------------------------------------
static std::vector<Value*> createAndFillArgBuffer( corelib::CoreIRBuilder& builder,
                                                   const PrintWriteMap&    printWriteMap,
                                                   LoopInfo&               loopInfo,
                                                   const DataLayout&       dataLayout )
{
    RT_ASSERT( !printWriteMap.empty() );

    std::vector<Value*> printfArgs;
    Function*           function = printWriteMap[0].second[0]->getParent()->getParent();


    for( const auto& mI : printWriteMap )
    {
        CallInst*              pwfCall   = mI.first.first;
        int                    argFormat = mI.first.second;
        std::vector<CallInst*> pwaCalls  = mI.second;
        RT_ASSERT( pwaCalls.size() == 1 || pwaCalls.size() == 2 );

        Value* toStore     = nullptr;
        Type*  toStoreType = nullptr;

        switch( argFormat )
        {
            case 0:
            {
                // The argument is an integer.
                CallInst* pwaCall = pwaCalls[0];


                Loop* loop    = loopInfo.getLoopFor( pwaCall->getParent() );
                Loop* pwfLoop = loopInfo.getLoopFor( pwfCall->getParent() );

                toStoreType = builder.getInt32Ty();
                builder.SetInsertPoint( pwaCall );

                // We are not in a loop.
                if( loop == pwfLoop )
                {
                    toStore = pwaCall->getOperand( 0 );
                }
                else
                {
                    // We are in a loop.
                    Value* pwaArgument = pwaCall->getOperand( 0 );
                    if( isa<PHINode>( pwaArgument ) )
                    {
                        PHINode* phi = cast<PHINode>( pwaArgument );
                        RT_ASSERT( phi->getNumIncomingValues() == 2 );
                        for( unsigned int index = 0, end = phi->getNumIncomingValues(); index != end; ++index )
                        {
                            BasicBlock* block = phi->getIncomingBlock( index );
                            if( loopInfo.getLoopFor( block ) != loop )
                            {
                                toStore = phi->getIncomingValue( index );
                            }
                        }
                    }
                    else
                    {
                        RT_ASSERT( isa<LoadInst>( pwaArgument ) );
                        toStore = pwaCall->getOperand( 0 );
                    }
                }

                break;
            }
            case 2:
            {
                // The argument is a float. We have to cast it to a double.
                CallInst* pwaCall = pwaCalls[0];
                toStore           = pwaCall->getOperand( 0 );
                builder.SetInsertPoint( pwaCall );
                toStore     = builder.CreateBitCast( toStore, builder.getFloatTy() );
                toStore     = builder.CreateFPExt( toStore, builder.getDoubleTy() );
                toStoreType = builder.getDoubleTy();
                break;
            }
            case 1:
            case 3:
            {
                Value* firstValue  = nullptr;
                Value* secondValue = nullptr;

                // The loop has been peeled.
                if( pwaCalls.size() == 2 )
                {
                    CallInst* firstPwa  = pwaCalls[0];
                    CallInst* secondPwa = pwaCalls[1];
                    builder.SetInsertPoint( secondPwa );

                    firstValue  = firstPwa->getOperand( 0 );
                    secondValue = secondPwa->getOperand( 0 );

                    toStore     = UndefValue::get( VectorType::get( builder.getInt32Ty(), 2 ) );
                    toStore     = builder.CreateInsertElement( toStore, firstValue, builder.getInt32( 0 ) );
                    toStore     = builder.CreateInsertElement( toStore, secondValue, builder.getInt32( 1 ) );
                    toStore     = builder.CreateBitCast( toStore, builder.getInt64Ty() );
                    toStoreType = builder.getInt64Ty();
                }
                else if( pwaCalls.size() == 1 )
                {
                    // The loop has not been unrolled, we have a single call in a loop.
                    CallInst* pwaCall = pwaCalls[0];

                    // We have to check that we are in a loop, otherwise there is a problem.
                    BasicBlock* pwaBlock = pwaCall->getParent();
                    Loop*       loop     = loopInfo.getLoopFor( pwaBlock );
                    RT_ASSERT_MSG( loop != nullptr, "Expected _print_write[argument] to be in a loop" );

                    Value* pwaArgument = pwaCall->getOperand( 0 );

                    if( PHINode* phi = dyn_cast<PHINode>( pwaArgument ) )
                    {
                        RT_ASSERT( phi->getNumIncomingValues() == 2 );

                        for( unsigned int index = 0, end = phi->getNumIncomingValues(); index != end; ++index )
                        {
                            BasicBlock* block = phi->getIncomingBlock( index );
                            if( loopInfo.getLoopFor( block ) != loop )
                            {
                                firstValue = phi->getIncomingValue( index );
                            }
                            else
                            {
                                secondValue = phi->getIncomingValue( index );
                            }
                        }

                        RT_ASSERT( isa<Instruction>( secondValue ) );
                        Instruction*         insertionPoint = cast<Instruction>( secondValue );
                        BasicBlock::iterator iter( insertionPoint );
                        ++iter;
                        builder.SetInsertPoint( &*iter );
                    }
                    else
                    {
                        RT_ASSERT( isa<LoadInst>( pwaArgument ) );
                        builder.SetInsertPoint( pwaCall );

                        firstValue          = pwaArgument;
                        LoadInst* firstLoad = cast<LoadInst>( pwaArgument );

                        // Get the address of the load.
                        Value*   address          = firstLoad->getPointerOperand();
                        unsigned addressSpace     = firstLoad->getPointerAddressSpace();
                        Value*   intAddress       = builder.CreatePtrToInt( address, builder.getInt64Ty() );
                        Value*   secondIntAddress = builder.CreateAdd( intAddress, builder.getInt64( 4 ) );
                        Value*   secondAddress =
                            builder.CreateIntToPtr( secondIntAddress, builder.getInt32Ty()->getPointerTo( addressSpace ) );

                        secondValue = builder.CreateLoad( secondAddress );
                    }

                    toStore     = UndefValue::get( VectorType::get( builder.getInt32Ty(), 2 ) );
                    toStore     = builder.CreateInsertElement( toStore, firstValue, builder.getInt32( 0 ) );
                    toStore     = builder.CreateInsertElement( toStore, secondValue, builder.getInt32( 1 ) );
                    toStore     = builder.CreateBitCast( toStore, builder.getInt64Ty() );
                    toStoreType = builder.getInt64Ty();
                }
                else
                {
                    RT_ASSERT_MSG( false,
                                   "Number of _print_write[argument] controlled by a single _print_write[format] is "
                                   "not supported." );
                }
                break;
            }
            default:
                RT_ASSERT_FAIL_MSG( "Unsupported format for print_write" );
                break;
        }

        RT_ASSERT( toStore );
        RT_ASSERT( toStoreType );

        // Create room for storing the argument.
        AllocaInst* argBuffer =
            corelib::CoreIRBuilder{corelib::getFirstNonAlloca( function )}.CreateAlloca( toStoreType, builder.getInt64( 1 ) );
        argBuffer->setAlignment( dataLayout.getPrefTypeAlignment( toStoreType ) );

        builder.CreateStore( toStore, argBuffer, false );
        printfArgs.push_back( argBuffer );

        Loop* pwfLoop = loopInfo.getLoopFor( pwfCall->getParent() );

        for( auto& pwaCall : pwaCalls )
        {
            // Determine if the current PWA belongs to the loop of 64 bit writes.
            // If so disable the loop.
            // This is needed because the loop trip count is based on the output of print_write, since we remove
            // print_write from the code we have to disable the loop too.
            // This modification does not work if we are going to print values larger then 64 bits.
            BasicBlock* pwaBlock = pwaCall->getParent();
            Loop*       loop     = loopInfo.getLoopFor( pwaBlock );
            // No loop.
            if( loop == pwfLoop )
                continue;

            BasicBlock* latch   = loop->getLoopLatch();
            BasicBlock* exiting = loop->getExitingBlock();
            BasicBlock* header  = loop->getHeader();
            SmallVector<BasicBlock*, 4> exitingBlocks;
            loop->getExitingBlocks( exitingBlocks );

            if( latch == exiting )
            {
                TerminatorInst* terminatorInst = latch->getTerminator();
                BranchInst*     branchInst     = dyn_cast<BranchInst>( terminatorInst );
                branchInst->setOperand( 0, builder.getFalse() );
            }
            else if( header == exiting )
            {
                TerminatorInst* terminatorInst = header->getTerminator();
                BranchInst*     branchInst     = dyn_cast<BranchInst>( terminatorInst );
                branchInst->setOperand( 0, builder.getTrue() );

                TerminatorInst* latchBranch = latch->getTerminator();
                latchBranch->setOperand( 0, branchInst->getSuccessor( 1 ) );
                // Remove redudandat phi entries.
                for( BasicBlock::iterator I = pwaBlock->begin(); isa<PHINode>( I ); ++I )
                {
                    PHINode* phi      = cast<PHINode>( I );
                    unsigned toRemove = phi->getBasicBlockIndex( latch );
                    phi->removeIncomingValue( toRemove );
                }
            }
            else
            {
                RT_ASSERT_FAIL();
            }

            break;
        }
    }

    RT_ASSERT( printfArgs.size() == printWriteMap.size() );

    return printfArgs;
}
