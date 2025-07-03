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

#include <Compile/SaveSetOptimizer.h>

#include <FrontEnd/Canonical/IntrinsicsManager.h>
#include <Util/ContainerAlgorithm.h>

#include <corelib/adt/Digraph.h>
#include <corelib/compiler/LLVMUtil.h>
#include <corelib/misc/String.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/system/Knobs.h>

#include <llvm/ADT/MapVector.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>

#include <llvm/IR/InstIterator.h>

#include <llvm/Support/raw_ostream.h>

using namespace optix;
using namespace corelib;
using namespace prodlib;
using namespace llvm;

namespace {
// clang-format off
  Knob<bool>        k_save(                RT_DSTRING("sso.save"),         false, RT_DSTRING("Save data during saveset optimization" ) );
  Knob<int>         k_ssoLogLevel(         RT_DSTRING("sso.logLevel"),     50,    RT_DSTRING("Set the log level of the save set optimization phase" ) );
  Knob<int>         k_frontierDepth(       RT_DSTRING("vw.frontierDepth"), 1,     RT_DSTRING("How many instructions beyond the frontier to save" ) );
  Knob<int>         k_iterations(          RT_DSTRING("vw.iterations"),    0,     RT_DSTRING("Iterations for save set optimization" ) );
  Knob<std::string> k_forceStore(          RT_DSTRING("vw.forceStore"),    "",    RT_DSTRING("Format: [cpid] node0 node1 ... [cpid] node0 ..." ) );

  Knob<bool>        k_enableCallableProgramArgRemat(RT_DSTRING("vw.enableCallableProgramArgRemat"), true, RT_DSTRING("Enable rematerialization of callable program arguments." ) );
  Knob<bool>        k_enableGetVariableValueRemat(  RT_DSTRING("vw.enableGetVariableValueRemat"),   false, RT_DSTRING("Enable rematerialization of rtGetVariableValue calls." ) );
  Knob<bool>        k_enableExperimentalRemat(      RT_DSTRING("vw.enableExperimentalRemat"),       false, RT_DSTRING("Enable rematerialization of various calls such as getLwrrentRay, getAttributeValue, and getBufferElement." ) );

  Knob<bool>        k_chooseBestLwtBySize( RT_DSTRING("vw.chooseBestLwtBySize"), false, RT_DSTRING("Choose the minimal save set by size rather than cost (only for sft.computeMinimalSaveSet bruteForce)." ) );
  Knob<int>         k_bruteForceLwtSizeLimit( RT_DSTRING("vw.bruteForceLwtSizeLimit"), 20, RT_DSTRING("Set the maximum size of lwts after which the brute force algorithm bails out. Set to -1 to disable." ) );
// clang-format on
}

/**********************************************************************
 *
 * Continuation optimizer base class
 *
 **********************************************************************/

SaveSetOptimizer::SaveSetOptimizer( const DataLayout& DL )
    : m_DL( DL )
{
    m_valid                    = false;
    m_allowPrimitiveIndexRemat = true;
}

const InstSetVector& SaveSetOptimizer::getSaveSet() const
{
    RT_ASSERT( m_valid );
    return m_saveSet;
}

const InstSetVector& SaveSetOptimizer::getRematSet() const
{
    RT_ASSERT( m_valid );
    return m_rematSet;
}

void SaveSetOptimizer::disallowPrimitiveIndexRemat()
{
    RT_ASSERT( !m_valid );
    m_allowPrimitiveIndexRemat = false;
}

/**********************************************************************
 *
 * Nop optimizer
 *
 **********************************************************************/

namespace {
class NopSaveSetOptimizer : public SaveSetOptimizer
{
  public:
    NopSaveSetOptimizer( const DataLayout& DL );
    void run( const InstSetVector& restoreSet, const std::string& idString ) override;
};
}


NopSaveSetOptimizer::NopSaveSetOptimizer( const DataLayout& DL )
    : SaveSetOptimizer( DL )
{
}

void NopSaveSetOptimizer::run( const InstSetVector& restoreSet, const std::string& idString )
{
    RT_ASSERT( !m_valid );
    // No-op: the saveset and rematset are both equal to the restoreSet )
    m_saveSet  = restoreSet;
    m_rematSet = restoreSet;
    m_valid    = true;
}


/**********************************************************************
 *
 * Rematerialization costs: lwrrently shared between algorithms.
 *
 **********************************************************************/

static const int LOAD_COST     = 95;
static const int STORE_COST    = 215;
static const int INFINITE_COST = 10000000;
static const int DEFAULT_COST  = STORE_COST;

//------------------------------------------------------------------------------
static unsigned int instructionSize( Instruction* I, const DataLayout& DL )
{
    // Round up in bytes
    return ( DL.getTypeAllocSize( I->getType() ) + 3 ) / 4;
}

//------------------------------------------------------------------------------
// %frame.ptr = call i8* @_ZN10Megakernel15stackGetPointerEPN4cort14CanonicalStateE(%struct.Interstate.66 %0, %"struct.cort::CanonicalState.62"* %1)
// %3 = call i32 @stackSizePlaceholder()
// %4 = getelementptr inbounds i8* %frame.ptr, i32 %3
// %argFrame = bitcast i8* %4 to { i32, i32, i32 }*
// %argPtr = getelementptr inbounds { i32, i32, i32 }* %argFrame, i32 0, i32 1
// %5 = load i32* %argPtr
static bool isCallableProgramArgLoad( LoadInst* I )
{
    GetElementPtrInst* argPtr = dyn_cast<GetElementPtrInst>( I->getPointerOperand() );
    if( !argPtr )
        return false;

    Value* frame = argPtr->getPointerOperand();
    while( CastInst* castInst = dyn_cast<CastInst>( frame ) )
        frame = castInst->getOperand( 0 );

    GetElementPtrInst* frameGEP = dyn_cast<GetElementPtrInst>( frame );
    if( !frameGEP )
        return false;
    if( frameGEP->getNumIndices() != 1 )
        return false;

    CallInst* stackGetPtr = dyn_cast<CallInst>( frameGEP->getPointerOperand() );
    CallInst* stackSize   = dyn_cast<CallInst>( frameGEP->idx_begin() );

    if( !stackGetPtr || !stackSize )
        return false;

    if( !stackGetPtr->getCalledFunction()->getName().startswith(
            "_ZN10Megakernel15stackGetPointerEPN4cort14CanonicalStateE" )
        && !stackGetPtr->getCalledFunction()->getName().startswith( "rewrite_stackGetPointer" ) )
    {
        return false;
    }

    return stackSize->getCalledFunction()->getName().startswith( "stackSizePlaceholder" );
}

//------------------------------------------------------------------------------
static int rematCost( Instruction* I, bool allowPrimitiveIndexRemat, const DataLayout& DL )
{
    unsigned int size = instructionSize( I, DL );

    if( isa<BinaryOperator>( I ) && !I->mayHaveSideEffects() )
        return 10 * size;

    if( isa<CastInst>( I ) )
        return 1;

    switch( I->getOpcode() )
    {
        case Instruction::Alloca:
            return 2 * size;

        case Instruction::GetElementPtr:
            return 2;

        case Instruction::ExtractValue:
        case Instruction::ExtractElement:
            return 2;

        case Instruction::Call:
        {
            CallInst*        call     = dyn_cast<CallInst>( I );
            Function*        func     = call->getCalledFunction();
            const StringRef& funcName = func->getName();

            if( funcName.startswith( "optix.ptx" ) && func->hasFnAttribute( Attribute::ReadNone ) )
                return 2;

            if( funcName.startswith( "rewrite_stackGetPointer" ) )
                return 1;

            if( funcName.startswith( "stackSizePlaceholder" ) )
                return 1;

            if( allowPrimitiveIndexRemat && funcName.find( "optixi_getPrimitiveArgToIntersect" ) != StringRef::npos )
                return LOAD_COST * size;

            if( k_enableGetVariableValueRemat.get() && funcName.find( "optixi_getVariableValue" ) != StringRef::npos )
                return LOAD_COST;

            if( GetBufferElement::isIntrinsic( func ) && func->hasFnAttribute( Attribute::ReadNone ) )
                return LOAD_COST;

            if( GetBufferElementFromId::isIntrinsic( func ) && func->hasFnAttribute( Attribute::ReadNone ) )
                return LOAD_COST;

            // These are experimental test that are not lwrrently deemed safe or desirable.
            if( k_enableExperimentalRemat.get() )
            {
                if( funcName.find( "optixi_getBufferSize" ) != StringRef::npos )
                    return LOAD_COST;
                if( funcName.find( "optixi_getLwrrentTmax" ) != StringRef::npos )
                    return LOAD_COST;
                if( funcName.find( "optixi_getTexture_tex_2dValue" ) != StringRef::npos )
                    return LOAD_COST;
                if( funcName.find( "optixi_transformTuple" ) != StringRef::npos )
                    return LOAD_COST;
                if( funcName.find( "optixi_getAttributeValue" ) != StringRef::npos )
                    return LOAD_COST;
                // It is not easily possible to rematerialize getLwrrentRay.
                //if( funcName.find("optixi_getLwrrentRay") != StringRef::npos )
                //return LOAD_COST;
            }

            return INFINITE_COST;
        }

        case Instruction::Load:
        {
            if( k_enableCallableProgramArgRemat.get() && isCallableProgramArgLoad( cast<LoadInst>( I ) ) )
                return LOAD_COST * size;
            return INFINITE_COST;
        }

        default:
            return INFINITE_COST;
    }
}


/**********************************************************************
 *
 * ValueWeb continuation optimizer
 *
 **********************************************************************/

namespace {
class ValueWebSaveSetOptimizer : public SaveSetOptimizer
{
  public:
    ValueWebSaveSetOptimizer( const DataLayout& DL );
    void run( const InstSetVector& restoreSet, const std::string& idString ) override;

  private:
    struct ValueWebNode
    {
        enum Disposition
        {
            Stored,
            Rematerialized,
            Dropped  // To be dropped from the value web
        } disposition;
        bool   required;  // part of the original restoreSet
        int    cost;
        size_t idx;
        ValueWebNode();
        ValueWebNode( size_t idx, unsigned int size, bool required );
    };
    typedef MapVector<Instruction*, ValueWebNode> ValueWebType;

    void populateValueWeb( const InstSetVector& restoreSet, ValueWebType& valueWeb, InstSetVector& edgeNodes );
    bool updateRematCost( ValueWebType& valueWeb, Instruction* inst, ValueWebNode& vw );
    bool hasRematUse( ValueWebType& valueWeb, Instruction* I );
    void saveDotFile( const std::string& fname, const ValueWebType& valueWeb, const InstSetVector& edgeNodes ) const;
    void saveInstructionGraph( const std::string& filename, ValueWebType& valueWeb ) const;
};
}


//------------------------------------------------------------------------------
ValueWebSaveSetOptimizer::ValueWebSaveSetOptimizer( const DataLayout& DL )
    : SaveSetOptimizer( DL )
{
}

//------------------------------------------------------------------------------
ValueWebSaveSetOptimizer::ValueWebNode::ValueWebNode( size_t idx, unsigned int size, bool required )
    : disposition( Stored )
    , required( required )
    , cost( DEFAULT_COST * size )
    , idx( idx )
{
}

//------------------------------------------------------------------------------
ValueWebSaveSetOptimizer::ValueWebNode::ValueWebNode()
    : disposition( Stored )
    , required( false )
    , cost( 0 )
    , idx( 0 )
{
}

//------------------------------------------------------------------------------
void ValueWebSaveSetOptimizer::run( const InstSetVector& restoreSet, const std::string& idString )
{
    m_valid = true;
    ValueWebType   valueWeb;
    InstSetVector edgeNodes;

    // Initialize the value web, marking all live values as "stored", which means
    // "saved/restored via the stack".
    populateValueWeb( restoreSet, valueWeb, edgeNodes );

    // Simple algorithm to optimize the save set. Iteratively try to make values
    // rematerialized and drop those that are not necessary for rematerialization.
    // TODO: Make this more efficient with work lists.
    bool anyChanged = true;
    while( anyChanged )
    {
        anyChanged = false;

        // Determine if rematerialization is cheaper than storing a value.
        bool colwerged = false;
        while( !colwerged )
        {
            colwerged = true;
            for( ValueWebType::iterator it = valueWeb.begin(); it != valueWeb.end(); ++it )
            {
                Instruction*  inst = it->first;
                ValueWebNode& vw   = it->second;

                if( vw.disposition == ValueWebNode::Dropped )
                    continue;

                if( updateRematCost( valueWeb, inst, vw ) )
                {
                    colwerged  = false;
                    anyChanged = true;
                }
            }
        }

        // Clean up the web:
        // Drop a value from the web if is not "required", not marked as remat, and
        // if none of its uses is rematerialized. This means that the value is
        // not required, directly or indirectly.
        colwerged = false;
        while( !colwerged )
        {
            colwerged = true;
            for( ValueWebType::iterator iter = valueWeb.begin(); iter != valueWeb.end(); ++iter )
            {
                Instruction*  inst = iter->first;
                ValueWebNode& vw   = iter->second;

                if( vw.required || vw.disposition != ValueWebNode::Stored || hasRematUse( valueWeb, inst ) )
                    continue;

                vw.disposition = ValueWebNode::Dropped;
                colwerged      = false;
                anyChanged     = true;
            }
        }
    }


    if( k_save.get() )
    {
        std::string saveFilename = "valueweb-" + idString + ".dot";
        saveDotFile( saveFilename, valueWeb, edgeNodes );
        saveInstructionGraph( "inst-" + saveFilename, valueWeb );
    }


    // Construct saveSet and rematSet from the valueWeb
    SmallVector<Instruction*, 16> workList;
    SmallSet<Instruction*, 16>    processed;

    // Initialize the work list from the initial live values.
    for( Instruction* inst : restoreSet )
    {
        workList.push_back( inst );
        processed.insert( inst );
    }

    while( !workList.empty() )
    {
        Instruction*        inst = workList.pop_back_val();
        const ValueWebNode& vw   = valueWeb[inst];
        RT_ASSERT( vw.disposition == ValueWebNode::Stored || vw.disposition == ValueWebNode::Rematerialized );

        // All values that are restored or rematerialized are placed in the remat set.
        m_rematSet.insert( inst );

        // If the value is marked as "store", it goes onto the stack and we are done.
        if( vw.disposition == ValueWebNode::Stored )
        {
            m_saveSet.insert( inst );
            continue;
        }

        // Otherwise, it is rematerialized, so we relwrse into its operands which by
        // definition must all be part of the web, too.
        for( Instruction::op_iterator O = inst->op_begin(), OE = inst->op_end(); O != OE; ++O )
        {
            Instruction* opI = dyn_cast<Instruction>( *O );
            // Make sure we do not add the same instruction multiple times.
            if( opI && std::get<1>( processed.insert( opI ) ) )
                workList.push_back( opI );
        }
    }
}

//------------------------------------------------------------------------------
// Initialize valueWeb with required values in the restore set and
// add values that should be considered for rematerialization.
// Edge nodes are the frontier of operands that can't be rematerialized.
void ValueWebSaveSetOptimizer::populateValueWeb( const InstSetVector& restoreSet, ValueWebType& valueWeb, InstSetVector& edgeNodes )
{
    SmallVector<Instruction*, 16> workList;
    for( Instruction* inst : restoreSet )
    {
        unsigned int size = instructionSize( inst, m_DL );
        valueWeb.insert( std::make_pair( inst, ValueWebNode( valueWeb.size(), size, /*required*/ true ) ) );
        if( rematCost( inst, m_allowPrimitiveIndexRemat, m_DL ) < INFINITE_COST )
            workList.push_back( inst );
        else
            edgeNodes.insert( inst );
    }

    // Traverse operands to put all values in the ValueWeb that can possibly be
    // rematerialized.
    while( !workList.empty() )
    {
        Instruction* inst = workList.pop_back_val();
        for( Instruction::op_iterator O = inst->op_begin(), OE = inst->op_end(); O != OE; ++O )
        {
            Instruction* opInst = dyn_cast<Instruction>( *O );
            if( !opInst || valueWeb.count( opInst ) )
                continue;

            // Add the instruction to the web.
            unsigned int size = instructionSize( inst, m_DL );
            valueWeb.insert( std::make_pair( opInst, ValueWebNode( valueWeb.size(), size, /*required*/ false ) ) );

            // If this instruction can't be rematerialized, we've reached an edge node.
            // Otherwise, relwrse.
            if( rematCost( opInst, m_allowPrimitiveIndexRemat, m_DL ) < INFINITE_COST )
                workList.push_back( opInst );
            else
                edgeNodes.insert( opInst );
        }
    }
}

//------------------------------------------------------------------------------
// inst is an instruction and vw its ValueWebNode.
// Returns true if there was a change.
bool ValueWebSaveSetOptimizer::updateRematCost( ValueWebType& valueWeb, Instruction* inst, ValueWebNode& vw )
{
    RT_ASSERT( vw.disposition != ValueWebNode::Dropped );

    // The rematerialization cost is the cost of the instruction itself plus the
    // cost of storing its operands minus the cost of storing the instruction.

    // Compute the cost of the instruction itself.
    int newCost = rematCost( inst, m_allowPrimitiveIndexRemat, m_DL );

    // Add the cost of storing each operand.
    // Some operands are free, e.g. constants, parameters, and values that already
    // have to be stored anyway. Other operands can't be added for various
    // reasons (INFINITE_COST), in which case the current instruction can't be
    // rematerialized. The middle ground is rematerializable instructions that are
    // not yet part of the web, and so impose additional cost when added.
    for( Instruction::op_iterator O = inst->op_begin(), OE = inst->op_end(); O != OE; ++O )
    {
        int          opCost;
        Instruction* opI = dyn_cast<Instruction>( *O );
        if( !opI )       // e.g. constant, parameter
            opCost = 0;  // free
        else if( !valueWeb.count( opI ) )
            opCost = INFINITE_COST;
        else
        {
            ValueWebNode& opvw = valueWeb[opI];
            if( opvw.required )
                opCost = 0;  // Required values have no additional cost
            else if( opvw.disposition == ValueWebNode::Dropped )
                opCost = INFINITE_COST;
            else
                opCost = opvw.cost;
        }

        // Note: costs are not really linear which one of the problems with this method
        if( opCost == INFINITE_COST )
        {
            newCost = INFINITE_COST;
            break;
        }
        newCost += opCost;
    }

    // If the instruction itself or any of its operands can't be rematerialized,
    // we can't do anything.
    if( newCost == INFINITE_COST )
        return false;

    // If the instruction previously was considered to be stored and the cost for
    // rematerialization is lower, change the disposition. If the instruction was
    // already marked for rematerialization, we only update the cost if it became
    // lower (e.g. because an operand was moved into the web).
    if( ( newCost <= vw.cost && vw.disposition == ValueWebNode::Stored ) || newCost < vw.cost )
    {
        vw.disposition = ValueWebNode::Rematerialized;
        vw.cost        = newCost;
        return true;
    }

    return false;
}

//------------------------------------------------------------------------------
// Returns true if no uses in the valueWeb are rematerialized
bool ValueWebSaveSetOptimizer::hasRematUse( ValueWebType& valueWeb, Instruction* I )
{
    for( Value::user_iterator U = I->user_begin(), UE = I->user_end(); U != UE; ++U )
    {
        Instruction* UI = dyn_cast<Instruction>( *U );
        if( !valueWeb.count( UI ) )
            continue;  // Ignore uses not in the web

        ValueWebNode& vw = valueWeb[UI];
        if( vw.disposition == ValueWebNode::Rematerialized )
            return true;
    }
    return false;
}

//------------------------------------------------------------------------------
void ValueWebSaveSetOptimizer::saveDotFile( const std::string& fname, const ValueWebType& valueWeb, const InstSetVector& edgeNodes ) const
{
    if( valueWeb.empty() )
        return;
    Function* func = valueWeb.begin()->first->getParent()->getParent();

    std::ofstream dot( fname.c_str() );
    dot << "digraph " << func->getName().str() << "{\n";
    dot << "    node [style=filled]\n";

    // Draw regular nodes
    DenseMap<Instruction*, int> nodeNumbers;
    for( const auto& NI : valueWeb )
    {
        Instruction*        I  = NI.first;
        const ValueWebNode& vw = NI.second;

        int n                = (int)nodeNumbers.size();
        nodeNumbers[I]       = n;
        std::string nodename = "n" + stringf( "%d", n );
        std::string disposition;
        std::string color;
        switch( vw.disposition )
        {
            case ValueWebNode::Stored:
                disposition = "stored";
                color       = vw.required ? "red" : "\"#FFA0A0\"";
                break;
            case ValueWebNode::Rematerialized:
                disposition = "rematerialized";
                color       = vw.required ? "yellow" : "\"#FFFFA0\"";
                break;
            case ValueWebNode::Dropped:
                disposition = "dropped";
                color       = "gray";
                break;
        }
        dot << '\t' << nodename << " [ shape=rectangle ";
        if( vw.required )
            dot << " style=\"filled,rounded\"";
        dot << " fillcolor=" << color;
        dot << " label=\"" << disposition << ":" << vw.idx << ( vw.required ? " (in restore set)" : "" ) << "\\n"
            << " total cost: " << vw.cost << "\\n"
            << " remat cost: " << rematCost( I, m_allowPrimitiveIndexRemat, m_DL ) << "\\n"
            << " " << instructionToEscapedString( I ) << "\"]\n";
    }

    // Draw "edge" nodes - those available for rematerialization with infinite cost
    for( Instruction* inst : edgeNodes )
    {
        int n                = (int)nodeNumbers.size();
        nodeNumbers[inst]    = n;
        std::string nodename = "n" + stringf( "%d", n );
        dot << '\t' << nodename << " [ shape=rectangle style=dashed label=\"" << instructionToEscapedString( inst ) << "\"]\n";
    }

    // Connections from regular nodes to regular nodes and edge nodes
    for( ValueWebType::const_iterator NI = valueWeb.begin(), NIE = valueWeb.end(); NI != NIE; ++NI )
    {
        Instruction* I       = NI->first;
        std::string  to_name = "n" + stringf( "%d", nodeNumbers[I] );
        for( Instruction::op_iterator O = I->op_begin(), OE = I->op_end(); O != OE; ++O )
        {
            Instruction* opI = dyn_cast<Instruction>( *O );
            if( valueWeb.count( opI ) != 0 )
            {
                std::string from_name = "n" + stringf( "%d", nodeNumbers[opI] );
                dot << '\t' << from_name << " -> " << to_name << "\n";
            }
            if( edgeNodes.count( opI ) != 0 )
            {
                std::string from_name = "n" + stringf( "%d", nodeNumbers[opI] );
                dot << '\t' << from_name << " -> " << to_name << " [style=dashed]\n";
            }
        }
    }

    // Connections from edge nodes to edge nodes
    for( Instruction* inst : edgeNodes )
    {
        std::string to_name = "n" + stringf( "%d", nodeNumbers[inst] );
        for( Instruction::op_iterator O = inst->op_begin(), OE = inst->op_end(); O != OE; ++O )
        {
            Instruction* opI = dyn_cast<Instruction>( *O );
            if( edgeNodes.count( opI ) != 0 )
            {
                std::string from_name = "n" + stringf( "%d", nodeNumbers[opI] );
                dot << '\t' << from_name << " -> " << to_name << " [style=dashed]\n";
            }
        }
    }
    dot << "}\n";
}


//------------------------------------------------------------------------------
void ValueWebSaveSetOptimizer::saveInstructionGraph( const std::string& filename, ValueWebType& valueWeb ) const
{
    if( valueWeb.empty() )
        return;
    bool      showBlocks = false;
    Function* func       = valueWeb.begin()->first->getParent()->getParent();

    std::ofstream dot( filename.c_str() );
    dot << "digraph " << func->getName().str() << "{\n";
    dot << "    node [shape=ellipse style=filled fillcolor=white]\n";

    // creates nodes
    DenseMap<Instruction*, int> instToIdx;
    DenseMap<BasicBlock*, int>  blockToIdx;
    BasicBlock* lwrBlock = nullptr;
    for( inst_iterator I = inst_begin( func ), IE = inst_end( func ); I != IE; ++I )
    {
        Instruction* inst = &*I;
        int          idx  = (int)instToIdx.size();
        instToIdx[inst]   = idx;

        if( showBlocks && lwrBlock != inst->getParent() )
        {
            if( lwrBlock != nullptr )
                dot << "  }\n";
            lwrBlock             = inst->getParent();
            int bIdx             = (int)blockToIdx.size();
            blockToIdx[lwrBlock] = bIdx;
            dot << "  subgraph cluster_b" << bIdx << " {\n";
        }
        dot << "    i" << idx << " [label=i" << idx;
        if( valueWeb.count( inst ) )
        {
            ValueWebNode& vw = valueWeb[inst];
            std::string   color;
            switch( vw.disposition )
            {
                case ValueWebNode::Stored:
                    color = vw.required ? "red" : "\"#FFA0A0\"";
                    break;
                case ValueWebNode::Rematerialized:
                    color = vw.required ? "yellow" : "\"#FFFFA0\"";
                    break;
                case ValueWebNode::Dropped:
                    color = "gray";
                    break;
            }
            if( !color.empty() )
                dot << " fillcolor=" << color;
            if( vw.required )
                dot << " shape=rectangle";
        }
        dot << " tooltip=\"" << instructionToEscapedString( inst ) << "\"";
        dot << "]\n";
    }
    if( showBlocks )
        dot << "  }\n";

    // create edges
    for( inst_iterator I = inst_begin( func ), IE = inst_end( func ); I != IE; ++I )
    {
        int dstIdx = instToIdx[&*I];
        for( Instruction::op_iterator O = I->op_begin(), OE = I->op_end(); O != OE; ++O )
        {
            if( Instruction* opI = dyn_cast<Instruction>( *O ) )
            {
                int srcIdx = instToIdx[opI];
                dot << "    i" << srcIdx << " -> i" << dstIdx << "\n";
            }
        }
    }

    dot << "}\n";
}


/**********************************************************************
 *
 * Randomized continuation optimizer
 *
 **********************************************************************/

namespace {
class RandomizedSaveSetOptimizer : public SaveSetOptimizer
{
  public:
    RandomizedSaveSetOptimizer( const DataLayout& DL );
    void run( const InstSetVector& restoreSet, const std::string& idString ) override;

  private:
    struct NodeData
    {
        Instruction* inst;
        size_t       idx;
        int          cost;
        int          rematCost;      // how much to rematerialize just this node
        int          rematTreeCost;  // how much would it cost to rematerialize this node and uncommitted nodes in its operand tree (committed nodes are already available)
        int numBadOps;  // number of operands which are not themselves rematerializable or depend on values not rematerizable
        enum Disposition
        {
            Uncommitted,
            Stored,
            Rematerialized,
        } disposition;
        bool required;

        NodeData() {}
        NodeData( Instruction* inst, size_t idx, int rematCost, int numBadOps = 0 )
            : inst( inst )
            , idx( idx )
            , cost( 0 )
            , rematCost( rematCost )
            , rematTreeCost( 0 )
            , numBadOps( numBadOps )
            , disposition( Uncommitted )
            , required( false )
        {
        }

        std::string name() { return stringf( "n%zu", idx ); }

        static std::string defaultAttributes() { return "node [shape=rectangle, style=filled]"; }

        std::string attributes()
        {
            const char* dispStr = "";
            const char* color   = "";
            switch( disposition )
            {
                case Uncommitted:
                    dispStr = "uncommitted";
                    color   = numBadOps ? "gray" : "lightblue";
                    break;
                case Stored:
                    dispStr = "stored";
                    color   = required ? "red" : "\"#FFA0A0\"";
                    break;
                case Rematerialized:
                    dispStr = "rematerialized";
                    color   = required ? "yellow" : "\"#FFFFA0\"";
                    break;
            }
            return stringf(
                "[fillcolor=%s %s "
                "label=\"%s:%zu\\n"
                "cost / remat / tree: %d / %d / %d\\n"
                "numBadOps:%d\\n"
                "%s\"]",
                color, required ? "style=\"filled,rounded\"" : "", dispStr, idx, cost, rematCost, rematTreeCost,
                numBadOps, instructionToEscapedString( inst ).c_str() );
        }

        bool committed() { return disposition != Uncommitted; }
    };

    struct EdgeData
    {
        EdgeData() {}

        static std::string defaultAttributes() { return ""; }
        std::string        attributes() { return ""; }
    };

    typedef Digraph<NodeData, EdgeData, Instruction*> DigraphTy;
    typedef DigraphTy::Node    Node;
    typedef DigraphTy::Edge    Edge;
    typedef std::vector<Node*> NodeList;

    DigraphTy   m_graph;
    NodeList    m_frontierNodes;  // Nodes with operands that cannot be rematerialized.
    int         m_totalCost;      // The sum of the costs of committed nodes in the graph
    std::string m_idString;       // A string to identify the file when printing

    // Fill the graph nodes and edges and propagate poison info. The nodes are
    // added to the graph in a topological order.
    void populateGraph( const InstSetVector& restoreSet );
    Node* populateGraph( Instruction* inst, const InstSetVector& restoreSet );

    // Update rematTreeCost for a node.
    void updateTreeCost( Node* node );

    // Considers the node for each value in the saveSet and, if its rematTreeCost
    // is low enough, removes the node fromt the save set and colwerts it and any
    // 'Uncommitted' nodes in its operand tree to 'Rematerialized'.
    // Returns true if any node was changed.
    bool rematerialize( NodeList& storedSet );

    // Make all 'Uncommitted' nodes 'Stored' and repeatedly run rematerialize()
    // until we hit a steady state. We get different results by traversing the
    // nodes in different orders.
    void tryAddingRandomStores( int iterations );

    void saveDot( const std::string& filename );

    void saveNodeData( std::vector<NodeData>& nodeData );
    void restoreNodeData( const std::vector<NodeData>& nodeData );

    // Changes a node's disposition and updates m_totalCost
    void updateDisposition( NodeData& nd, NodeData::Disposition disposition );

    // Update the numBadOps counts for the dependents of the newStoredNodes.
    // Accumulate required 'Stored' nodes that had their numBadOps go to zero in
    // 'colwertibleStoredNodes'. These are candidates for rematerialization.
    void updateNumBadOps( const NodeList& newlyStoredNodes, NodeList& colwertibleStoredNodes );

    // Update the numBadOps counts for the dependents of nodes that have been
    // changed to 'Uncommitted'
    void updateNumBadOps( const NodeList& newlyUncommittedNodes );

    // Colwert any of the given nodes to 'Uncommitted' that are not connected
    // to required 'Rematerialized' nodes
    void lwllUnusedNodes( const NodeList& nodes );
};

}  // namespace


//------------------------------------------------------------------------------
RandomizedSaveSetOptimizer::RandomizedSaveSetOptimizer( const DataLayout& DL )
    : SaveSetOptimizer( DL )
{
}


//------------------------------------------------------------------------------
void RandomizedSaveSetOptimizer::run( const InstSetVector& restoreSet, const std::string& idString )
{
    m_valid     = true;
    m_totalCost = 0;
    populateGraph( restoreSet );

    // Initialize saveSet and m_totalCost. saveSet could be initialized from the
    // directly from the restoreSet, but doing it this way puts the instructions
    // in a topological order.
    NodeList storedNodes;
    for( Node* N : m_graph )
    {
        NodeData& nd = N->data();
        if( nd.disposition == NodeData::Stored )
        {
            storedNodes.push_back( N );
            m_totalCost += nd.cost;
        }
    }

    errs() << "REDUCER totalCost (initial): " << m_totalCost << "\n";
    // Remove easily rematerialized instructions (those which are dependent on
    // other nodes in the restore set or operand trees that terminate in
    // rematerializable instructions.
    while( rematerialize( storedNodes ) )
    {
    }
    errs() << "REDUCER totalCost (local min): " << m_totalCost << "\n";

    // We are now in a local minimum of the cost function. We might be able to
    // get to a lower net cost by introducing stores in the Uncommitted nodes which
    // would make some of the saveSet nodes rematerializable.
    srand( 0xBEEFFACE );  // TODO: We should use an RNG that is the same on all platforms
    tryAddingRandomStores( k_iterations.get() );
    errs() << "REDUCER totalCost (final): " << m_totalCost << "\n";

    // Save the graph
    if( k_save.get() )
    {
        // Update the cost of uncommitted nodes
        for( Node* N : m_graph )
        {
            NodeData& nd = N->data();
            if( nd.disposition == NodeData::Uncommitted )
                updateTreeCost( N );
        }

        saveDot( "valueweb-" + m_idString + " .dot" );
    }

    // Build rematSet and saveSet
    m_saveSet.clear();
    for( Node* N : m_graph )
    {
        NodeData& nd = N->data();
        if( nd.disposition == NodeData::Rematerialized || nd.disposition == NodeData::Stored )
        {
            m_rematSet.insert( nd.inst );
            if( nd.disposition == NodeData::Stored )
                m_saveSet.insert( nd.inst );
        }
    }
}


//------------------------------------------------------------------------------
void RandomizedSaveSetOptimizer::populateGraph( const InstSetVector& restoreSet )
{
    for( InstSetVector::iterator I = restoreSet.begin(), IE = restoreSet.end(); I != IE; ++I )
    {
        Instruction* inst = *I;
        Node*        node = m_graph.getNode( inst );
        if( !node )
            node = populateGraph( inst, restoreSet );
    }
}

//------------------------------------------------------------------------------
RandomizedSaveSetOptimizer::Node* RandomizedSaveSetOptimizer::populateGraph( Instruction* inst, const InstSetVector& restoreSet )
{
    int numBadOps     = 0;
    int instRematCost = rematCost( inst, m_allowPrimitiveIndexRemat, m_DL );
    if( instRematCost != INFINITE_COST )
    {
        // Traverse operands
        for( Instruction::op_iterator O = inst->op_begin(), OE = inst->op_end(); O != OE; ++O )
        {
            Instruction* opInst = dyn_cast<Instruction>( *O );
            if( opInst && m_graph.getNode( opInst ) == nullptr )
            {
                if( rematCost( opInst, m_allowPrimitiveIndexRemat, m_DL ) < INFINITE_COST )
                    populateGraph( opInst, restoreSet );
                else if( restoreSet.count( opInst ) )
                    populateGraph( opInst, restoreSet );
                else
                {
                    numBadOps++;
                }
            }
        }
    }

    // Add to graph
    Node* node = m_graph.addNode( NodeData( inst, m_graph.size(), instRematCost, numBadOps ), inst );
    if( restoreSet.count( inst ) )
    {
        NodeData& nd = node->data();
        updateDisposition( nd, NodeData::Stored );
        nd.required = true;
    }
    if( numBadOps )
        m_frontierNodes.push_back( node );

    // Add operand edges
    numBadOps = 0;
    for( Instruction::op_iterator O = inst->op_begin(), OE = inst->op_end(); O != OE; ++O )
    {
        if( Instruction* opInst = dyn_cast<Instruction>( *O ) )
        {
            if( Node* opNode = m_graph.getNode( opInst ) )
            {
                NodeData& nd = opNode->data();
                if( nd.numBadOps && nd.disposition != NodeData::Stored )
                    numBadOps++;
                m_graph.addEdge( opNode, node, EdgeData() );
            }
        }
    }

    // Propagate numBadOps
    node->data().numBadOps += numBadOps;

    return node;
}

//------------------------------------------------------------------------------
void RandomizedSaveSetOptimizer::updateTreeCost( Node* node )
{
    SmallVector<Node*, 16> workList;
    SmallPtrSet<Node*, 16> visited;

    int rematTreeCost = 0;
    visited.insert( node );
    workList.push_back( node );
    while( !workList.empty() )
    {
        Node*     lwr   = workList.pop_back_val();
        NodeData& lwrNd = lwr->data();
        rematTreeCost += lwrNd.rematCost;
        for( Node::iterator E = lwr->pred_begin(), EE = lwr->pred_end(); E != EE; ++E )
        {
            Node* op = ( *E )->from();
            if( std::get<1>( visited.insert( op ) ) && !op->data().committed() )
                workList.push_back( op );
        }
    }
    node->data().rematTreeCost = rematTreeCost;
}

//------------------------------------------------------------------------------
static void dbgInsertForceStore( std::set<int>& indices, const std::string& idString )
{
    if( k_forceStore.get().empty() )
        return;

    std::vector<std::string> groups = tokenize( k_forceStore.get(), "[" );
    for( const std::string& group : groups )
    {
        if( idString != group.c_str() )
            continue;

        std::vector<std::string> vals = tokenize( group, "], " );
        for( size_t v = 1; v < vals.size(); ++v )
            indices.insert( atoi( vals[v].c_str() ) );
    }
}

//------------------------------------------------------------------------------
template <typename C, typename T>
static void remove( C& container, T& item )
{
    container.erase( algorithm::find( container, item ) );
}

//------------------------------------------------------------------------------
bool RandomizedSaveSetOptimizer::rematerialize( NodeList& nodes )
{
    std::set<int> dbgForceStore;
    dbgInsertForceStore( dbgForceStore, m_idString );

    SmallVector<Node*, 16> workList;
    SmallPtrSet<Node*, 16> visited;
    SmallVector<Node*, 16> toRemove;
    for( Node* node : nodes )
    {
        NodeData& nd = node->data();
        RT_ASSERT( nd.disposition == NodeData::Stored );
        updateTreeCost( node );
        if( nd.numBadOps || nd.rematTreeCost > nd.cost || dbgForceStore.count( nd.idx ) )
            continue;
        toRemove.push_back( node );

        // Rematerialize tree
        visited.insert( node );
        workList.push_back( node );
        while( !workList.empty() )
        {
            Node*     lwr   = workList.pop_back_val();
            NodeData& lwrNd = lwr->data();
            updateDisposition( lwrNd, NodeData::Rematerialized );

            // traverse operands
            for( Node::iterator E = lwr->pred_begin(), EE = lwr->pred_end(); E != EE; ++E )
            {
                Node* op = ( *E )->from();
                if( std::get<1>( visited.insert( op ) ) && !op->data().committed() )
                    workList.push_back( op );
            }
        }
    }

    for( size_t i = 0; i < toRemove.size(); ++i )
        remove( nodes, toRemove[i] );

    return !toRemove.empty();
}

//------------------------------------------------------------------------------
void RandomizedSaveSetOptimizer::tryAddingRandomStores( int iterations )
{
    bool dbgSave = false;
    if( dbgSave )
        saveDot( "initial.dot" );

    // Save 'Uncommitted' nodes as the initial pool of stored node candidates
    NodeList initialPool;
    for( Node* N : m_graph )
    {
        if( !N->data().committed() )
            initialPool.push_back( N );
    }
    if( initialPool.empty() )
        return;  // nothing to do

    std::vector<NodeData> initialNodeData;
    saveNodeData( initialNodeData );
    int initialCost = m_totalCost;

    // Save the current best solution
    int                   bestCost = m_totalCost;
    std::vector<NodeData> bestNodeData;
    saveNodeData( bestNodeData );

    // Try randomized iterations
    for( int i = 0; i < iterations; ++i )
    {
        NodeList pool = initialPool;
        restoreNodeData( initialNodeData );
        m_totalCost = initialCost;

        // Make random selection of random size
        NodeList newStoredNodes;
        int      count = 1 + ( rand() % pool.size() );
        for( int n = 0; n < count; ++n )
        {
            // Choose a node
            int idx = rand() % pool.size();

            // Mark stored and increase cost
            updateDisposition( pool[idx]->data(), NodeData::Stored );
            newStoredNodes.push_back( pool[idx] );

            // Remove and shrink the pool
            pool[idx] = pool.back();
            pool.pop_back();
        }

        // Figure out which nodes we should try to rematerialize
        NodeList colwertibleStoredNodes;
        updateNumBadOps( newStoredNodes, colwertibleStoredNodes );
        if( colwertibleStoredNodes.empty() )
            continue;

        if( dbgSave )
            saveDot( "pre.dot" );
        while( rematerialize( colwertibleStoredNodes ) )
        {
        }
        if( dbgSave )
            saveDot( "remat.dot" );
        lwllUnusedNodes( newStoredNodes );
        if( dbgSave )
            saveDot( "post.dot" );

        if( m_totalCost < bestCost )
        {
            errs() << "it: " << i << " REDUCER totalCost: " << m_totalCost << "\n";
            bestCost = m_totalCost;
            saveNodeData( bestNodeData );
        }
    }

    // Restore best solution found
    m_totalCost = bestCost;
    restoreNodeData( bestNodeData );
}


//------------------------------------------------------------------------------
void RandomizedSaveSetOptimizer::saveDot( const std::string& filename )
{
    Function* func = ( *m_graph.begin() )->data().inst->getParent()->getParent();

    std::ofstream dot( filename.c_str() );
    dot << "digraph " << func->getName().str() << "{\n";

    std::string indent = "  ";
    dot << indent << NodeData::defaultAttributes() << "\n";
    for( Node* N : m_graph )
    {
        NodeData& node = N->data();
        dot << indent << node.name() << " " << node.attributes() << "\n";
        for( Node::iterator E = N->begin(), EE = N->end(); E != EE; ++E )
        {
            EdgeData& ed  = ( *E )->data();
            NodeData& src = ( *E )->from()->data();
            NodeData& dst = ( *E )->to()->data();
            dot << indent << src.name() << " -> " << dst.name() << " " << ed.attributes() << "\n";
        }
    }


    // Add outer nodes beyond the frontier
    std::deque<Instruction*> workList;
    for( Node* node : m_frontierNodes )
        workList.push_back( node->data().inst );

    DenseMap<Instruction*, std::string> outerNames;
    for( int d = 0; d < k_frontierDepth.get() && !workList.empty(); ++d )
    {
        size_t count = workList.size();
        for( size_t i = 0; i < count; ++i )  // process only the current layer
        {
            Instruction* inst = workList.front();
            workList.pop_front();

            // Add operands
            std::string dstName = ( d == 0 ) ? m_graph.getNode( inst )->data().name() : outerNames[inst];
            for( Instruction::op_iterator O = inst->op_begin(), OE = inst->op_end(); O != OE; ++O )
            {
                Instruction* opInst = dyn_cast<Instruction>( *O );
                if( !opInst || m_graph.getNode( opInst ) )
                    continue;

                std::string srcName;
                if( outerNames.count( opInst ) == 0 )
                {
                    srcName            = stringf( "outer%u", outerNames.size() );
                    outerNames[opInst] = srcName;
                    dot << indent << srcName << "[style=dashed label=\"" << instructionToEscapedString( opInst )
                        << "\"]\n";
                    if( d < k_frontierDepth.get() - 1 )
                        workList.push_back( opInst );
                }
                else
                    srcName = outerNames[opInst];

                dot << indent << srcName << " -> " << dstName << " [style=dashed]\n";
            }
        }
    }

    dot << "}\n";
}

//------------------------------------------------------------------------------
void RandomizedSaveSetOptimizer::saveNodeData( std::vector<NodeData>& nodeData )
{
    nodeData.resize( m_graph.size() );
    int i = 0;
    for( Node* N : m_graph )
        nodeData[i++] = N->data();
}

//------------------------------------------------------------------------------
void RandomizedSaveSetOptimizer::restoreNodeData( const std::vector<NodeData>& nodeData )
{
    int i = 0;
    for( Node* N : m_graph )
        N->data() = nodeData[i++];
}

//------------------------------------------------------------------------------
void RandomizedSaveSetOptimizer::updateDisposition( NodeData& nd, NodeData::Disposition disposition )
{
    if( nd.disposition == disposition )
        return;

    m_totalCost -= nd.cost;

    nd.disposition = disposition;
    switch( disposition )
    {
        case NodeData::Uncommitted:
            nd.cost = 0;
            break;
        case NodeData::Rematerialized:
            nd.cost = nd.rematCost;
            break;
        case NodeData::Stored:
            nd.cost = STORE_COST;
            break;
    }

    m_totalCost += nd.cost;

    RT_ASSERT( !( nd.disposition == NodeData::Rematerialized && nd.numBadOps ) );  // We can't rematerialize a node with bad ops
}

//------------------------------------------------------------------------------
void RandomizedSaveSetOptimizer::updateNumBadOps( const NodeList& newStoredNodes, NodeList& colwertibleStoredNodes )
{
    SmallVector<Node*, 16> workList( newStoredNodes.begin(), newStoredNodes.end() );
    while( !workList.empty() )
    {
        Node* lwrNode = workList.pop_back_val();
        for( Edge* E : *lwrNode )
        {
            Node*     child   = E->to();
            NodeData& childNd = child->data();
            if( childNd.numBadOps )
            {
                childNd.numBadOps--;
                if( !childNd.numBadOps )  // This node is no longer dependent on unavailable values
                {
                    if( childNd.disposition == NodeData::Stored )
                    {
                        if( childNd.required )
                            colwertibleStoredNodes.push_back( child );
                    }
                    else
                        workList.push_back( child );  // propagate
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
void RandomizedSaveSetOptimizer::updateNumBadOps( const NodeList& newlyUncommittedNodes )
{
    SmallVector<Node*, 16> workList( newlyUncommittedNodes.begin(), newlyUncommittedNodes.end() );
    while( !workList.empty() )
    {
        Node* lwrNode = workList.pop_back_val();
        if( !lwrNode->data().numBadOps )
            continue;
        for( Edge* E : *lwrNode )
        {
            Node*     child   = E->to();
            NodeData& childNd = child->data();
            if( !childNd.numBadOps && !childNd.committed() )
                workList.push_back( child );  // propagate
            childNd.numBadOps++;
        }
    }
}

//------------------------------------------------------------------------------
void RandomizedSaveSetOptimizer::lwllUnusedNodes( const NodeList& nodes )
{
    for( Node* node : nodes )
    {
        NodeData& nd = node->data();
        if( nd.required )
            continue;

        // Look at dependents
        bool unused = true;
        SmallVector<Node*, 16> workList( 1, node );
        SmallPtrSet<Node*, 16> visited;
        visited.insert( node );  // not really necessary because the graph is acyclic
        while( unused && !workList.empty() )
        {
            for( Edge* E : *workList.pop_back_val() )
            {
                Node*     child   = E->to();
                NodeData& childNd = child->data();
                if( childNd.disposition == NodeData::Rematerialized )
                {
                    if( childNd.required )
                    {
                        unused = false;
                        break;
                    }
                    else if( std::get<1>( visited.insert( child ) ) )
                        workList.push_back( child );
                }
            }
        }

        if( unused )
            updateDisposition( nd, NodeData::Uncommitted );
    }

    updateNumBadOps( nodes );
}


/**********************************************************************
 *
 * Brute force optimizer
 *
 **********************************************************************/

namespace {
// A cut is a valid set of instructions from which all required values can be rematerialized.
typedef InstSetVector Cut;

// TODO: This shares quite some code with the ValueWebSaveSetOptimizer, unify them.
class BruteForceSaveSetOptimizer : public SaveSetOptimizer
{
  public:
    BruteForceSaveSetOptimizer( const DataLayout& DL );
    void run( const InstSetVector& restoreSet, const std::string& idString ) override;

  private:
    struct ValueWebNode
    {
        enum Disposition
        {
            Stored,
            Rematerialized,
            Dropped  // To be dropped from the value web
        } disposition;
        bool   required;  // part of the original restoreSet
        int    cost;
        size_t idx;
        ValueWebNode();
        ValueWebNode( size_t idx, unsigned int size, bool required );
    };
    typedef MapVector<Instruction*, ValueWebNode> ValueWebType;

    // Create a value web by walking up the dependencies of the instructions in the
    // restore set up to instructions that can't be rematerialized (edge nodes) and
    // creating a node for every instruction we traversed.
    void populateValueWeb( const InstSetVector& restoreSet, ValueWebType& valueWeb, InstSetVector& edgeNodes );

    // Find the best cut wrt rematerialization cost. A new cut is created based on
    // the current cut by following a def-use chain of an instruction in the cut
    // and removing the def from the new cut. This is super expensive since it
    // explores the full search space of possible lwts.
    Cut* findBestLwt( const InstSetVector& restoreSet, const InstSetVector& edgeNodes, const ValueWebType& valueWeb ) const;

    // Remove redundant nodes from a cut.
    void minimizeLwt( Cut& cut, const InstSetVector& restoreSet ) const;

    // Create a value web from a given cut.
    // Every value in the cut is marked as "stored", all others as "remat" unless
    // they cannot be rematerialized (in which case they are "stored", too).
    void createValueWeb( const Cut* cut, ValueWebType& valueWeb ) const;

    // Test if the instruction has a use that is rematerialized.
    bool hasRematUse( ValueWebType& valueWeb, Instruction* I ) const;

    // Compute the total cost of a value web.
    int computeTotalCost( const ValueWebType& valueWeb ) const;
};
}


//------------------------------------------------------------------------------
BruteForceSaveSetOptimizer::BruteForceSaveSetOptimizer( const DataLayout& DL )
    : SaveSetOptimizer( DL )
{
}

//------------------------------------------------------------------------------
BruteForceSaveSetOptimizer::ValueWebNode::ValueWebNode( size_t idx, unsigned int size, bool required )
    : disposition( Stored )
    , required( required )
    , cost( DEFAULT_COST * size )
    , idx( idx )
{
}

//------------------------------------------------------------------------------
BruteForceSaveSetOptimizer::ValueWebNode::ValueWebNode()
    : disposition( Stored )
    , required( false )
    , cost( 0 )
    , idx( 0 )
{
}

//------------------------------------------------------------------------------
void BruteForceSaveSetOptimizer::run( const InstSetVector& restoreSet, const std::string& idString )
{
    m_valid = true;
    ValueWebType   valueWeb;
    InstSetVector edgeNodes;

    // Initialize the value web, marking all live values as "restored", which means
    // "saved/restored via the stack".
    populateValueWeb( restoreSet, valueWeb, edgeNodes );

    llog( k_ssoLogLevel.get() ) << "\nSearching best cut for web with " << valueWeb.size()
                                << " nodes, initial cut #values: " << restoreSet.size() << ".\n";

    if( k_bruteForceLwtSizeLimit.get() > 0 && (int)restoreSet.size() > k_bruteForceLwtSizeLimit.get() )
    {
        llog( k_ssoLogLevel.get() ) << "  cut size threshold crossed, falling back to default value web.\n";
        ValueWebSaveSetOptimizer copt( m_DL );
        copt.run( restoreSet, idString + ".fallback" );
        m_saveSet  = copt.getSaveSet();
        m_rematSet = copt.getRematSet();
        return;
    }

    // Find the best possible cut.
    Cut* bestLwt = findBestLwt( restoreSet, edgeNodes, valueWeb );
    RT_ASSERT( edgeNodes.empty() || bestLwt );

    // Update the value web to reflect the best cut.
    createValueWeb( bestLwt, valueWeb );
    delete bestLwt;

    // Construct saveSet and rematSet from the valueWeb
    SmallVector<Instruction*, 16> workList;
    SmallSet<Instruction*, 16>    processed;

    // Initialize the work list from the initial live values.
    for( Instruction* inst : restoreSet )
    {
        workList.push_back( inst );
        processed.insert( inst );
    }

    while( !workList.empty() )
    {
        Instruction*        inst = workList.pop_back_val();
        const ValueWebNode& vw   = valueWeb[inst];
        RT_ASSERT( vw.disposition == ValueWebNode::Stored || vw.disposition == ValueWebNode::Rematerialized );

        // All values that are restored or rematerialized are placed in the remat set.
        m_rematSet.insert( inst );

        // If the value is marked as "store", it goes onto the stack and we are done.
        if( vw.disposition == ValueWebNode::Stored )
        {
            m_saveSet.insert( inst );
            continue;
        }

        // Otherwise, it is rematerialized, so we relwrse into its operands which by
        // definition must all be part of the web, too.
        for( Instruction::op_iterator O = inst->op_begin(), OE = inst->op_end(); O != OE; ++O )
        {
            Instruction* opI = dyn_cast<Instruction>( *O );
            // Make sure we do not add the same instruction multiple times.
            if( opI && std::get<1>( processed.insert( opI ) ) )
                workList.push_back( opI );
        }
    }

    if( prodlib::log::active( k_ssoLogLevel.get() ) )
    {
        // Compute default value web for comparison.
        ValueWebSaveSetOptimizer* copt = new ValueWebSaveSetOptimizer( m_DL );
        copt->run( restoreSet, idString + ".default" );
        const InstSetVector& otherSaveSet  = copt->getSaveSet();
        const InstSetVector& otherRematSet = copt->getRematSet();
        llog( k_ssoLogLevel.get() ) << "delta(saves) = " << ( (int)m_saveSet.size() - (int)otherSaveSet.size() )
                                    << ", delta(remats) = " << ( (int)m_rematSet.size() - (int)otherRematSet.size() ) << "\n";
        delete copt;
    }
}

//------------------------------------------------------------------------------
// Initialize valueWeb with required values in the restore set and
// add values that should be considered for rematerialization.
// Edge nodes are the frontier of operands that can't be rematerialized.
void BruteForceSaveSetOptimizer::populateValueWeb( const InstSetVector& restoreSet, ValueWebType& valueWeb, InstSetVector& edgeNodes )
{
    SmallVector<Instruction*, 16> workList;
    for( Instruction* inst : restoreSet )
    {
        unsigned int size = instructionSize( inst, m_DL );
        valueWeb.insert( std::make_pair( inst, ValueWebNode( valueWeb.size(), size, /*required*/ true ) ) );
        if( rematCost( inst, m_allowPrimitiveIndexRemat, m_DL ) < INFINITE_COST )
            workList.push_back( inst );
        else
            edgeNodes.insert( inst );
    }

    // Traverse operands to put all values in the ValueWeb that can possibly be
    // rematerialized.
    while( !workList.empty() )
    {
        Instruction* inst = workList.pop_back_val();
        for( Instruction::op_iterator O = inst->op_begin(), OE = inst->op_end(); O != OE; ++O )
        {
            Instruction* opInst = dyn_cast<Instruction>( *O );
            if( !opInst || valueWeb.count( opInst ) )
                continue;

            // Add the instruction to the web.
            unsigned int size = instructionSize( inst, m_DL );
            valueWeb.insert( std::make_pair( opInst, ValueWebNode( valueWeb.size(), size, /*required*/ false ) ) );

            // If this instruction can't be rematerialized, we've reached an edge node.
            // Otherwise, relwrse.
            if( rematCost( opInst, m_allowPrimitiveIndexRemat, m_DL ) < INFINITE_COST )
                workList.push_back( opInst );
            else
                edgeNodes.insert( opInst );
        }
    }
}

//---------------------------------------------------------------------------
#if defined( __OPTIX_TEST_UNNECESSARY_BRUTEFORCE_RELWRSION )
static bool isVisitedLwt( Cut* cut, std::set<Cut*>& visited )
{
    RT_ASSERT( !visited.count( cut ) );

    // TODO: This is quick and very dirty. Use a hash for comparison instead.
    for( InstSetVector* visitedLwt : visited )
    {
        if( visitedLwt->size() != cut->size() )
            continue;

        bool isEqual = true;
        for( Instruction* inst : *visitedLwt )
        {
            if( cut->count( inst ) )
                continue;
            isEqual = false;
            break;
        }
        if( isEqual )
            return true;
    }

    return false;
}
#endif

//---------------------------------------------------------------------------
// From: http://burtleburtle.net/bob/c/lookup3.c
// By Bob Jenkins, 1996.  bob_jenkins@burtleburtle.net.  You may use this
// code any way you wish, private, educational, or commercial.  It's free.
// clang-format off
#define _optix_hashsize(n) ((uint32_t)1<<(n))
#define _optix_hashmask(n) (_optix_hashsize(n)-1)
#define _optix_rot(x,k) (((x)<<(k)) | ((x)>>(32-(k))))
#define _optix_mix(a,b,c) \
{ \
  (a) -= (c);  (a) ^= _optix_rot(c, 4);  (c) += (b); \
  (b) -= (a);  (b) ^= _optix_rot(a, 6);  (a) += (c); \
  (c) -= (b);  (c) ^= _optix_rot(b, 8);  (b) += (a); \
  (a) -= (c);  (a) ^= _optix_rot(c,16);  (c) += (b); \
  (b) -= (a);  (b) ^= _optix_rot(a,19);  (a) += (c); \
  (c) -= (b);  (c) ^= _optix_rot(b, 4);  (b) += (a); \
}
#define _optix_final(a,b,c) \
{ \
  (c) ^= (b); (c) -= _optix_rot(b,14); \
  (a) ^= (c); (a) -= _optix_rot(c,11); \
  (b) ^= (a); (b) -= _optix_rot(a,25); \
  (c) ^= (b); (c) -= _optix_rot(b,16); \
  (a) ^= (c); (a) -= _optix_rot(c,4);  \
  (b) ^= (a); (b) -= _optix_rot(a,14); \
  (c) ^= (b); (c) -= _optix_rot(b,24); \
}
// clang-format on

static uint32_t hashword( const uint32_t* k,        /* the key, an array of uint32_t values */
                          size_t          length,   /* the length of the key, in uint32_ts */
                          uint32_t        initval ) /* the previous hash, or an arbitrary value */
{
    uint32_t a, b, c;

    /* Set up the internal state */
    a = b = c = 0xdeadbeef + ( ( (uint32_t)length ) << 2 ) + initval;

    /*------------------------------------------------- handle most of the key */
    while( length > 3 )
    {
        a += k[0];
        b += k[1];
        c += k[2];
        _optix_mix( a, b, c );
        length -= 3;
        k += 3;
    }

    /*------------------------------------------- handle the last 3 uint32_t's */
    switch( length ) /* all the case statements fall through */
    {
        case 3:
            c += k[2];
        case 2:
            b += k[1];
        case 1:
            a += k[0];
            _optix_final( a, b, c );
        case 0: /* case 0: nothing left to add */
            break;
    }
    /*------------------------------------------------------ report the result */
    return c;
}

static uint32_t getLwtHash( const Cut* cut, const std::map<const Instruction*, uint32_t>& instMapping )
{
    InstVector tmp;
    tmp.insert( tmp.end(), cut->begin(), cut->end() );
    algorithm::sort( tmp );

    uint32_t*    k = new uint32_t[cut->size()]();
    unsigned int i = 0;
    for( const Instruction* inst : tmp )
    {
        RT_ASSERT( instMapping.count( inst ) );
        k[i++] = instMapping.find( inst )->second;
    }
    uint32_t hash = hashword( k, cut->size(), 1 );
    delete[] k;
    return hash;
}
#undef _optix_hashsize
#undef _optix_hashmask
#undef _optix_rot
#undef _optix_mix
#undef _optix_final

//---------------------------------------------------------------------------
#if 0
// For debugging only.
static void printSet( const InstSetVector& set, const char* msg )
{
    outs() << msg << "-----------------\n";
    for( Instruction* inst : set )
    {
        outs() << " * " << *inst << "\n";
    }
}
#endif

//---------------------------------------------------------------------------
Cut* BruteForceSaveSetOptimizer::findBestLwt( const InstSetVector& restoreSet, const InstSetVector& edgeNodes, const ValueWebType& valueWeb ) const
{
    Cut* bestLwt      = nullptr;
    Cut* smallestLwt  = nullptr;  // Only tracked for comparison.
    int  bestCost     = INT_MAX;  // Corresponds to bestLwt.
    int  bestSize     = INT_MAX;  // Corresponds to bestLwt.
    int  smallestCost = INT_MAX;  // Corresponds to smallestLwt.
    int  smallestSize = INT_MAX;  // Corresponds to smallestLwt.

    std::map<const Instruction*, uint32_t> instMapping;
    for( ValueWebType::const_iterator it = valueWeb.begin(); it != valueWeb.end(); ++it )
    {
        RT_ASSERT( it->second.idx < valueWeb.size() );
        instMapping[it->first] = (uint32_t)it->second.idx;
    }

    // Build initial graph cut from the restore set.
    // Ignore instructions that can be rematerialized "for free", i.e., from
    // constants, function arguments, and other values in the restore set.
    Cut* initialLwt = new Cut();
    for( Instruction* inst : restoreSet )
    {
        bool canRematForFree = true;
        for( Instruction::op_iterator O = inst->op_begin(), OE = inst->op_end(); O != OE; ++O )
        {
            Instruction* opI = dyn_cast<Instruction>( *O );
            if( !opI || restoreSet.count( opI ) )
                continue;
            canRematForFree = false;
            break;
        }
        if( canRematForFree )
            continue;

        initialLwt->insert( inst );
    }

    std::vector<Cut*> workList;
    workList.push_back( initialLwt );

    std::set<uint32_t> visited;
    visited.insert( getLwtHash( initialLwt, instMapping ) );

    std::set<Cut*> visitedLwts;
    visitedLwts.insert( initialLwt );

    int          numIterations = 0;
    ValueWebType tmpWeb( valueWeb );
    while( !workList.empty() )
    {
        Cut* lwrrentLwt = workList.back();
        workList.pop_back();

        // Determine the cost of the current cut.
        createValueWeb( lwrrentLwt, tmpWeb );
        const int lwrrentLwtCost = computeTotalCost( tmpWeb );

        // Determine the size of the current cut.
        int lwrrentLwtSize = 0;
        for( Instruction* inst : *lwrrentLwt )
            lwrrentLwtSize += instructionSize( inst, m_DL );

        // Update the best cut if necessary.
        if( lwrrentLwtCost < bestCost )
        {
            bestCost = lwrrentLwtCost;
            bestSize = lwrrentLwtSize;
            bestLwt  = lwrrentLwt;
        }

        // Update the smallest cut if necessary.
        if( lwrrentLwtSize < smallestSize )
        {
            smallestCost = lwrrentLwtCost;
            smallestSize = lwrrentLwtSize;
            smallestLwt  = lwrrentLwt;
        }

        if( ( k_chooseBestLwtBySize.get() && smallestLwt == lwrrentLwt ) || bestLwt == lwrrentLwt )
        {
            llog( k_ssoLogLevel.get() ) << "Found better cut: cost " << lwrrentLwtCost << ", size: " << lwrrentLwtSize
                                        << ", #values: " << lwrrentLwt->size() << "\n";
        }

        for( Instruction* inst : *lwrrentLwt )
        {
            if( edgeNodes.count( inst ) )
                continue;

            for( Instruction::op_iterator O = inst->op_begin(), OE = inst->op_end(); O != OE; ++O )
            {
                Instruction* opI = dyn_cast<Instruction>( *O );
                if( !opI )
                    continue;

                InstSetVector* nextLwt = new InstSetVector( *lwrrentLwt );
                nextLwt->insert( opI );

                // Minimize the cut. This will usually remove 'inst' unless it has multiple
                // predecessors. In such a case, the subsequent cut will be mostly useless,
                // but further relwrsion may yield better lwts. For example, when all other
                // predecessors of 'inst' have been added to the cut, it will be removed.
                minimizeLwt( *nextLwt, restoreSet );

                // If we've already seen this cut, make sure we do not evaluate it again.
                if( !visited.insert( getLwtHash( nextLwt, instMapping ) ).second )
                {
                    delete nextLwt;
                    continue;
                }

#ifdef __OPTIX_TEST_UNNECESSARY_BRUTEFORCE_RELWRSION
                if( isVisitedLwt( nextLwt, visitedLwts ) )
                {
                    llog( k_ssoLogLevel.get() ) << "should not relwrse!\n";
                    delete nextLwt;
                    continue;
                }
                visitedLwts.insert( nextLwt );
#endif

                workList.push_back( nextLwt );
            }
        }

#ifndef __OPTIX_TEST_UNNECESSARY_BRUTEFORCE_RELWRSION
        if( lwrrentLwt != bestLwt && lwrrentLwt != smallestLwt )
            delete lwrrentLwt;
#endif

#ifdef __OPTIX_PRINT_BRUTEFORCE_ITERATIONS
        if( numIterations % 10000 == 0 )
        {
            llog( k_ssoLogLevel.get() ) << "iteration " << numIterations << " finished.\n";
        }
#endif
        ++numIterations;
    }

#ifdef __OPTIX_TEST_UNNECESSARY_BRUTEFORCE_RELWRSION
    // Delete all lwts except for the best one.
    for( Cut* cut : visitedLwts )
    {
        if( cut != bestLwt && cut != smallestLwt )
            delete cut;
    }
#endif

    if( prodlib::log::active( k_ssoLogLevel.get() ) )
    {
        // Print some statistics.
        llog( k_ssoLogLevel.get() ) << "Evaluated " << visited.size() << " cut(s) in " << numIterations
                                    << " iterations. ";
        if( bestLwt != nullptr )
        {
            llog( k_ssoLogLevel.get() ) << "Best: cost " << bestCost << ", size: " << bestSize
                                        << ", #values: " << bestLwt->size() << " ";
        }
        if( smallestLwt != nullptr )
        {
            llog( k_ssoLogLevel.get() ) << "Smallest: cost " << smallestCost << ", size: " << smallestSize
                                        << ", #values: " << smallestLwt->size() << "\n";
        }
    }

    if( k_chooseBestLwtBySize.get() )
    {
        if( bestLwt != smallestLwt )
            delete bestLwt;
        return smallestLwt;
    }
    else
    {
        if( bestLwt != smallestLwt )
            delete smallestLwt;
        return bestLwt;
    }
}

//------------------------------------------------------------------------------
void BruteForceSaveSetOptimizer::minimizeLwt( Cut& cut, const InstSetVector& restoreSet ) const
{
    // Collect all instructions in the cut for which all uses are in the cut, too.
    InstSetVector toRemove;

    bool changed = true;
    while( changed )
    {
        changed = false;

        for( Instruction* inst : cut )
        {
            // We do not remove from the cut until we have collected all instructions,
            // since removal may introduce a bias if we remove an instruction before
            // testing if one of its uses is a redundant instruction, too.
            if( toRemove.count( inst ) )
                continue;

            bool redundant = true;
            for( Instruction::op_iterator O = inst->op_begin(), OE = inst->op_end(); O != OE; ++O )
            {
                Instruction* opI = dyn_cast<Instruction>( *O );
                if( !opI )
                    continue;
                if( cut.count( opI ) )
                    continue;
                redundant = false;
                break;
            }

            if( redundant )
            {
                toRemove.insert( inst );
                changed = true;
            }
        }
    }

    for( Instruction* inst : toRemove )
        cut.remove( inst );
}


void BruteForceSaveSetOptimizer::createValueWeb( const Cut* cut, ValueWebType& valueWeb ) const
{
    // Use the given value web as basis for all values we need to take into account.
    // Mark all values in the cut as "stored", everything else that can be
    // rematerialized as "rematerialized", and drop all values that are not required
    // on the fly.
    for( auto& it : valueWeb )
    {
        Instruction*  inst = it.first;
        ValueWebNode& vw   = it.second;

        // This can be called with cut being a nullptr, in which case everything is rematerialized.
        if( cut && cut->count( inst ) )
        {
            vw.disposition = ValueWebNode::Stored;
            vw.cost        = STORE_COST * instructionSize( inst, m_DL );
            continue;
        }

        vw.disposition = ValueWebNode::Rematerialized;
        vw.cost        = rematCost( inst, m_allowPrimitiveIndexRemat, m_DL );
    }

    // Clean up the web:
    // Drop a value from the web if is not "required", not marked as save, and
    // if none of its uses is rematerialized. This means that the value is
    // not required, directly or indirectly.
    bool changed = true;
    while( changed )
    {
        changed = false;
        for( ValueWebType::iterator it = valueWeb.begin(); it != valueWeb.end(); ++it )
        {
            Instruction*  inst = it->first;
            ValueWebNode& vw   = it->second;
            if( vw.required || vw.disposition != ValueWebNode::Rematerialized || hasRematUse( valueWeb, inst ) )
                continue;

            vw.disposition = ValueWebNode::Dropped;
            changed        = true;
        }
    }

    // If we can't rematerialize a value it also has to be stored.
    // NOTE: Don't do this before cleaning the web from unnecessary values.
    // TODO: Could it be useful to make these instructions part of the cut?
    for( auto& it : valueWeb )
    {
        Instruction*  inst = it.first;
        ValueWebNode& vw   = it.second;
        if( vw.disposition != ValueWebNode::Rematerialized )
            continue;

        if( vw.cost < INFINITE_COST )
            continue;

        vw.disposition = ValueWebNode::Stored;
        vw.cost        = STORE_COST * instructionSize( inst, m_DL );
    }
}

//------------------------------------------------------------------------------
bool BruteForceSaveSetOptimizer::hasRematUse( ValueWebType& valueWeb, Instruction* I ) const
{
    for( Value::user_iterator U = I->user_begin(), UE = I->user_end(); U != UE; ++U )
    {
        Instruction* UI = dyn_cast<Instruction>( *U );
        if( !valueWeb.count( UI ) )
            continue;  // Ignore uses not in the web

        ValueWebNode& vw = valueWeb[UI];
        if( vw.disposition == ValueWebNode::Rematerialized )
            return true;
    }
    return false;
}

//------------------------------------------------------------------------------
int BruteForceSaveSetOptimizer::computeTotalCost( const ValueWebType& valueWeb ) const
{
    int cost = 0;
    for( const auto& it : valueWeb )
    {
        const ValueWebNode& vw = it.second;
        if( vw.disposition == ValueWebNode::Dropped )
            continue;

        RT_ASSERT( vw.cost < INFINITE_COST );
        cost += vw.cost;
    }
    return cost;
}


/**********************************************************************
 *
 * Factory
 *
 **********************************************************************/

std::unique_ptr<SaveSetOptimizer> SaveSetOptimizer::create( const std::string& name, const DataLayout& DL )
{
    if( name == "nop" )
        return std::unique_ptr<SaveSetOptimizer>( new NopSaveSetOptimizer( DL ) );
    else if( name == "valueWeb" )
        return std::unique_ptr<SaveSetOptimizer>( new ValueWebSaveSetOptimizer( DL ) );
    else if( name == "bruteForce" )
        return std::unique_ptr<SaveSetOptimizer>( new BruteForceSaveSetOptimizer( DL ) );
    else if( name == "randomized" )
        return std::unique_ptr<SaveSetOptimizer>( new RandomizedSaveSetOptimizer( DL ) );
    else
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Invalid component" );
}
