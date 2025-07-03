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

#include <Context/BindingManager.h>
#include <Context/Context.h>
#include <Context/ObjectManager.h>
#include <Context/ProgramManager.h>
#include <Context/SharedProgramManager.h>
#include <Context/TableManager.h>
#include <Context/UpdateManager.h>
#include <Context/ValidationManager.h>
#include <Exceptions/IllegalSymbol.h>
#include <Exceptions/TypeMismatch.h>
#include <Exceptions/VariableNotFound.h>
#include <Exceptions/VariableRedeclared.h>
#include <ExelwtionStrategy/CORTTypes.h>
#include <FrontEnd/Canonical/CanonicalProgram.h>
#include <FrontEnd/Canonical/Mangle.h>
#include <Objects/GeometryInstance.h>
#include <Objects/GlobalScope.h>
#include <Objects/LexicalScope.h>
#include <Objects/Variable.h>
#include <corelib/misc/Cast.h>
#include <corelib/misc/String.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/exceptions/ValidationError.h>
#include <prodlib/math/Bits.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

#include <algorithm>
#include <cctype>
#include <cmath>

using namespace optix;
using namespace prodlib;
using namespace corelib;

namespace {
// clang-format off
Knob<bool> k_scopeTrace( RT_DSTRING( "scope.trace" ), false, RT_DSTRING( "Trace unresolved reference set events" ) );
// clang-format on
}  // namespace

// Utilities for constructing the implicit binaryTree representing the scope's dynamicVariableTable.
namespace {

// Struct used during construction of dynamicVariableTable, holding both a variable's token id and
// its offset into the scope's dynamicVariableTable.
struct VariableEntry
{
    VariableEntry( unsigned short t, unsigned short o )
        : m_token( t )
        , m_offset( o )
    {
    }
    unsigned short m_token;
    unsigned short m_offset;
};
// Comparison operator for above struct.
bool operator<( const VariableEntry& one, const VariableEntry& other )
{
    return one.m_token < other.m_token;
}

// ceilingIntDivision for proper median computations below.
inline size_t Div( size_t x, size_t y )
{
    if( x == 0 )
        return 0;
    // q = (x + y - 1) / y; might overflow in x+y
    return 1 + ( ( x - 1 ) / y );
}

// Is v odd or even?
inline bool is_odd( size_t v )
{
    return v % 2;
}

// Return the proper size of the array representing the dynamicVariableTable for a given number
// of entries. Background: To allow proper binary search in implicit binTrees, one goes from any
// inner node with index i by index 2*i+1 or 2*i+2 to its children, which have to exist both. On
// the leaf level it might happen though, that there is just one child to its parent, depending
// on the number of variables. In such cases we have to add one artificial node to the binTree.
// This has to be taken into accound when allocating the memory for the table. Fortunately this
// seems to boil down to
// - when there are an uneven number of nodes, all is well
// - when there are even number of nodes, we have one leaf w/o a direct sibling -> we have to add one
size_t getProperDynamicVariableTableSize( size_t nodeCount )
{
    return is_odd( nodeCount ) ? nodeCount : nodeCount + 1;
}

// Helper struct to keep track of split ranges in implicitBinTreeTransform().
struct Range
{
    Range( size_t l, size_t r, size_t d )
        : left( l )
        , right( r )
        , depth( d )
    {
    }
    // Left for debugging purposes only.
    void dump() const
    {
        llog( 75 ) << "Range [" << left << ", " << right << "]\n";
        llog( 75 ) << "\tdepth " << depth << ", ie something between (" << minElems() << ", " << maxElems()
                   << ") elements\n";
        llog( 75 ) << "\tcount " << count() << "\n";
        llog( 75 ) << "\tleaves " << leaves() << ", maxLeaves " << maxLeaves() << "\n";
    }
    size_t left;
    size_t right;
    size_t depth;

    size_t count() const { return right - left + 1; }
    size_t leaves() const { return count() + 1 - std::pow( 2, depth ); }
    size_t minElems() const { return std::pow( 2, depth ); }
    size_t maxElems() const { return std::pow( 2, depth + 1 ) - 1; }
    size_t maxLeaves() const { return ( maxElems() + 1 ) / 2; }
};

// Transform the given sorted array into an implicit binary tree, actually a complete binary tree with two additions
// 1. all leafs have the highest order bit in VariableEntry::m_token marked
// 2. to have complete subtrees (otherwise the search will get out of range), potentially one dummy entry added
//    - so that any node with a left child as a right child as well
//    - its value is identical to the parental one to avoid any unwanted search effects
// Taking 1. and 2. together, every node is either an internal node with two children or a leaf node.
// Note that you have to unmark the leaf values before accessing these.
void implicitBinTreeTransform( const std::vector<VariableEntry>& sortedValues, unsigned short* binTree )
{
    assert( std::is_sorted( sortedValues.begin(), sortedValues.end() ) );
    if( sortedValues.empty() )
        return;

    std::queue<Range> ranges;
    short             depth = static_cast<short>( std::log2( sortedValues.size() + 1 ) );
    ranges.push( Range( 0, sortedValues.size() - 1, depth ) );

    unsigned short* startPtr = binTree;
    while( !ranges.empty() )
    {
        size_t l = ranges.front().left;
        size_t r = ranges.front().right;
        size_t d = ranges.front().depth;
        ranges.pop();

        // found a leaf
        if( l == r )
        {
            *binTree = sortedValues[l].m_token;
            cort::markAsVariableTableLeaf( binTree++ );
            *binTree++ = sortedValues[l].m_offset;
            continue;
        }
        // right side is lacking a leaf, so we have to fill in some dummy entry which does not disturb the order.
        // Choosing parent entry for that, and add it as a dummy leaf.
        if( l > r )
        {
            *binTree = sortedValues[r].m_token;
            cort::markAsVariableTableLeaf( binTree++ );
            *binTree++ = sortedValues[r].m_offset;
            continue;
        }
        // compute median
        size_t m = l + Div( r - l, 2 );
        if( m == (size_t)-1 )
            continue;

        // check split with found median
        Range left( l, m - 1, d - 1 );
        Range right( m + 1, r, d - 1 );
        // if right side contains leaves while left side is not completly filled, move median one to the right
        // TODO For some bigger numbers this increment by one might be a performance issue - probably some
        // smart computation might give us the correct shift. But for now I don't expect that many variables...
        while( right.leaves() && left.leaves() < left.maxLeaves() )
        {
            m += 1;
            left  = Range( l, m - 1, d - 1 );
            right = Range( m + 1, r, d - 1 );
        }
        assert( !right.leaves() || left.leaves() == left.maxLeaves() );

        *binTree++ = sortedValues[m].m_token;
        *binTree++ = sortedValues[m].m_offset;

        ranges.push( left );
        ranges.push( right );
    }
    // correctness check that written memory is equal to allocated memory
    RT_ASSERT( binTree - startPtr == static_cast<ptrdiff_t>( getProperDynamicVariableTableSize( sortedValues.size() ) * 2 ) );
}

// Helper util to retrieve the variable entries out of the scope's m_tokenToVarMap.
std::vector<VariableEntry> getVariableEntries( const std::map<unsigned short, Variable*>& vars )
{
    std::vector<VariableEntry> res;
    res.reserve( vars.size() );
    for( auto var : vars )
    {
        unsigned short token  = var.first;
        unsigned short offset = cort::ILWALIDOFFSET;
        if( var.second->getType().isValid() )
        {
            try
            {
                offset = range_cast<unsigned short>( var.second->getScopeOffset() );
            }
            catch( const prodlib::IlwalidValue& )
            {
                throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, "Offset for variable " + var.second->getName()
                                                                    + " exceeded 64 KB range within scope: ",
                                             var.second->getScopeOffset() );
            }
        }
        res.push_back( VariableEntry( token, offset ) );
    }
    return res;
}

}  // namespace


//------------------------------------------------------------------------
// CTOR / DTOR
//------------------------------------------------------------------------

LexicalScope::LexicalScope( Context* context, ObjectClass objClass )
    : ManagedObject( context, objClass )
{
    m_id = m_context->getObjectManager()->registerObject( this );
    subscribeForValidation();
}

LexicalScope::~LexicalScope()
{
    // All objects must have linked pointers detached before the destructor is called.
    RT_ASSERT_MSG( m_linkedPointers.empty(), "Scope destroyed while references remain" );
    RT_ASSERT_MSG( !isAttached(), "LexicalScope is not fully detached before destruction" );
    if( getClass() != RT_OBJECT_PROGRAM )
    {
        // Because canonical programs are not detached, Program objects
        // might have remaining unresolved references and variable
        // resolutions. All other objects should be clean.
        RT_ASSERT_MSG( m_resolvedSet.empty(), "Scope destroyed while resolutions remain" );
        RT_ASSERT_MSG( m_unresolvedSet.empty(), "Scope destroyed while unresolved references remain" );
    }

    RT_ASSERT_MSG( m_variables.empty(), "Variables weren't removed in derived class before ~LexicalScope()" );

    // Remove ourselves from the validation queue
    m_context->getValidationManager()->unsubscribeForValidation( this );

    RT_ASSERT_MSG( m_validationIndex == -1, "Failed to remove object from validation list" );
}


//------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------

Variable* LexicalScope::declareVariable( const std::string& name )
{
    // Get the token number
    checkName( name );
    std::string    mangled_name = canonicalMangleVariableName( name );
    unsigned short token        = m_context->getObjectManager()->registerVariableName( mangled_name );
    if( m_tokenToVarMap.count( token ) != 0 )
        throw VariableRedeclared( RT_EXCEPTION_INFO, this, getVariableByToken( token ) );

    // Create the variable and index entry
    unsigned int index = m_variables.size();
    Variable*    ptr   = new Variable( this, mangled_name, index, token );
    m_variables.push_back( ptr );
    m_tokenToVarMap.insert( std::make_pair( token, ptr ) );

    // Propagate symbol resolution based on this declaration.
    variableDeclarationDidChange( ptr, true );

    // No need to reallocate the object record - variables are empty until set

    return ptr;
}

void LexicalScope::removeVariable( Variable* variable )
{
    if( !variable )
        return;
    if( variable->getScope() != this )
        throw VariableNotFound( RT_EXCEPTION_INFO, this, variable->getName() );

    VariableMapType::iterator map_iter = m_tokenToVarMap.find( variable->getToken() );
    if( map_iter == m_tokenToVarMap.end() )
        throw VariableNotFound( RT_EXCEPTION_INFO, this, variable->getName() );
    RT_ASSERT_MSG( map_iter->second == variable, "Variable argument's token doesn't match pointer in m_tokenToVarMap" );

    // Set the value to null for types that point to objects.  All the types are specified in the switch,
    // since this function is used by the public API.
    switch( variable->getType().baseType() )
    {
        case VariableType::Buffer:
        case VariableType::DemandBuffer:
            variable->setBuffer( nullptr );
            break;
        case VariableType::GraphNode:
            variable->setGraphNode( nullptr );
            break;
        case VariableType::Program:
            variable->setProgram( nullptr );
            break;
        case VariableType::TextureSampler:
            variable->setTextureSampler( nullptr );
            break;
        case VariableType::ProgramId:
        case VariableType::BufferId:
        case VariableType::Float:
        case VariableType::Int:
        case VariableType::Uint:
        case VariableType::LongLong:
        case VariableType::ULongLong:
        case VariableType::UserData:
        case VariableType::Unknown:
            // Nothing to do
            break;
        default:
            throw AssertionFailure( RT_EXCEPTION_INFO, "Unknown type for variable: " + variable->getName() );
    }

    // Propagate symbol resolution based on removing this declaration.
    variableDeclarationDidChange( variable, false );

    // Remove the variable from both the vector and the map
    VariableArrayType::iterator vec_iter = std::find( m_variables.begin(), m_variables.end(), variable );
    RT_ASSERT( vec_iter != m_variables.end() );
    m_variables.erase( vec_iter );
    m_tokenToVarMap.erase( map_iter );

    delete variable;

    reallocateRecord();
}

Variable* LexicalScope::queryVariable( const std::string& name ) const
{
    return getVariableByToken( m_context->getObjectManager()->getTokenForVariableName( canonicalMangleVariableName( name ) ) );
}

Variable* LexicalScope::getVariableByIndex( unsigned int index ) const
{
    if( index >= m_variables.size() )
        throw IlwalidValue( RT_EXCEPTION_INFO, "Variable index out of range: " + std::to_string( index ) );
    return m_variables[index];
}

unsigned int LexicalScope::getVariableCount() const
{
    // NOTE: this returns the number of variables that have been declared
    // through the host API.  It does not include undeclared variables
    // present in PTX if the lexical scope is an rtProgram (i.e. this
    // function doesn't handle reflection).
    // See OP-938 for more details.
    return range_cast<unsigned int>( m_variables.size() );
}

static void checkVarTypeMatching( const VariableReference* varref, const Variable* var )
{
    if( varref->getType().baseType() == VariableType::Unknown && !varref->isInitialized() )
        throw ValidationError( RT_EXCEPTION_INFO, "Unknown type for variable: " + var->getName() + "." );

    if( varref->getType().baseType() == VariableType::ProgramId
        && ( ( var->getType().baseType() == VariableType::Int ) || ( var->getType().baseType() == VariableType::Uint ) ) )
        return;

    if( varref->getType().baseType() == VariableType::BufferId
        && ( ( var->getType().baseType() == VariableType::Int ) || ( var->getType().baseType() == VariableType::Uint ) ) )
        return;

    if( varref->getType().baseType() == VariableType::Program && varref->getType() != var->getType() )
    {
        ProgramManager* pm           = var->getScope()->getContext()->getProgramManager();
        std::string     refSig       = pm->getFunctionSignatureForToken( varref->getType().programSignatureToken() );
        std::string     varSig       = pm->getFunctionSignatureForToken( var->getType().programSignatureToken() );
        std::string exceptionMessage = "Program variable \"" + var->getName() + "\"" + " assigned signature " + varSig
                                       + ".  Should be " + refSig + ".";
        throw TypeMismatch( RT_EXCEPTION_INFO, exceptionMessage );
    }

    if( varref->getType() != var->getType() )
    {
        std::string exceptionMessage = "Variable \"" + var->getName() + "\"" + " assigned type " + var->getType().toString()
                                       + ".  Should be " + varref->getType().toString() + ".";
        throw TypeMismatch( RT_EXCEPTION_INFO, exceptionMessage );
    }
}

void LexicalScope::validate() const
{
    // All resolved variables must type check.
    const ProgramManager* pm = m_context->getProgramManager();
    for( auto refid : m_resolvedSet )
    {
        const VariableReference* varRef = pm->getVariableReferenceById( refid );
        Variable*                var    = getVariableByToken( varRef->getVariableToken() );
        RT_ASSERT( var );
        checkVarTypeMatching( varRef, var );
    }

    // All object-type variables must have a value (whether they are
    // resolved or not).
    for( auto variable : m_variables )
    {
        switch( variable->getType().baseType() )
        {
            case VariableType::Buffer:
            case VariableType::DemandBuffer:
                if( !variable->getBuffer() )
                    throw ValidationError( RT_EXCEPTION_INFO,
                                           "Buffer variable '" + variable->getName() + "' is not set" );
                break;

            case VariableType::GraphNode:
                if( !variable->getGraphNode() )
                    throw ValidationError( RT_EXCEPTION_INFO,
                                           "Node variable '" + variable->getName() + "' is not set" );
                break;

            case VariableType::Program:
                if( !variable->getProgram() )
                    throw ValidationError( RT_EXCEPTION_INFO,
                                           "Program variable '" + variable->getName() + "' is not set" );
                break;

            case VariableType::TextureSampler:
                if( !variable->getTextureSampler() )
                    throw ValidationError( RT_EXCEPTION_INFO,
                                           "TextureSampler variable '" + variable->getName() + "' is not set" );
                break;

            case VariableType::ProgramId:
            case VariableType::BufferId:
                // Processed as ids
                break;

            case VariableType::Float:
            case VariableType::Int:
            case VariableType::Uint:
            case VariableType::LongLong:
            case VariableType::ULongLong:
            case VariableType::UserData:
            case VariableType::Unknown:
                // Nothing to check
                break;

            default:
                throw ValidationError( RT_EXCEPTION_INFO, "Internal error in variable: " + variable->getName() );
        }
    }
}

//------------------------------------------------------------------------
// Internal API
//------------------------------------------------------------------------

int LexicalScope::getScopeID() const
{
    return *m_id;
}

Variable* LexicalScope::getOrDeclareVariable( const std::string& name )
{
    Variable* var = queryVariable( name );
    if( !var )
        var = declareVariable( name );

    return var;
}

Variable* LexicalScope::getOrDeclareVariable( const std::string& name, const VariableType& type )
{
    Variable* var = getOrDeclareVariable( name );
    var->setOrCheckType( type );
    return var;
}

Variable* LexicalScope::getVariableByToken( unsigned short token ) const
{
    VariableMapType::const_iterator iter = m_tokenToVarMap.find( token );
    if( iter == m_tokenToVarMap.end() )
        return nullptr;
    RT_ASSERT( iter->second != nullptr );
    return iter->second;
}

//------------------------------------------------------------------------
// LinkedPtr relationship mangement
//------------------------------------------------------------------------

void LexicalScope::childOffsetDidChange( const LinkedPtr_Link* link )
{
    // Precision is probably not worthwhile here. Just rewrite the
    // object record.  Should a performance problem arise, start by
    // deferring object record writes before attempting to update
    // individual elements.
    writeRecord();
}


//------------------------------------------------------------------------
// Notifications from variable (used ONLY by variable)
//------------------------------------------------------------------------

void LexicalScope::variableSizeDidChange( Variable* var )
{
    // Note: this just calls reallocateRecord but could be used to
    // perform a lazy update if necessary.
    reallocateRecord();
}

void LexicalScope::variableTypeDidChange( Variable* var )
{
    subscribeForValidation();
}

void LexicalScope::variableValueDidChange( Variable* var )
{
    // create overload with old and new int value -> specialize set<int, 1>
    // Note: being conservative here, but we could choose to lazily update if necessary.
    writeRecord();
}

void LexicalScope::variableValueDidChange( Variable* var, int oldValue, int newValue )
{
    // Note: being conservative here, but we could choose to lazily update if necessary.
    writeRecord();
    if( var->getType().isProgramOrProgramId() && var->getType().baseType() != VariableType::Program )
    {
        // We cannot distinguish between a variable of type int and a program id,
        // this needs to be done for every variable of type int. It gets filtered
        // through the variable references below.
        const ProgramManager* pm = m_context->getProgramManager();
        for( const auto& refid : pm->getReferencesForVariable( var->getToken() ) )
        {
            const VariableReference* varref = pm->getVariableReferenceById( refid );
            if( varref->getType().baseType() != VariableType::ProgramId )
                continue;
            // Establish link to bindless callable from variable.
            Program* program = m_context->getObjectManager()->getProgramByIdNoThrow( newValue );

            if( !program && newValue > 0 )
            {
                // program id must be valid or 0, otherwise it is a programming bug.
                ureport2( m_context->getUsageReport(), "CALLABLE TRACKING" )
                    << "Warning: Found invalid program ID in variable \"" << var->getName() << "\": " << newValue
                    << ". Setting it to 0.\n";
            }

            var->setBindlessProgram( program );

            if( oldValue > 0 )
                m_context->getBindingManager()->addOrRemove_ProgramIdBinding( refid, oldValue, false );
            m_context->getBindingManager()->addOrRemove_ProgramIdBinding( refid, newValue, true );
        }
    }
}

void LexicalScope::variableValueDidChange( Variable* var, GraphNode* oldNode, GraphNode* newNode )
{
    // Note: being conservative here, but we could choose to lazily update if necessary.
    writeRecord();
    subscribeForValidation();

    // Update graphnode bindings.  See comment about performance in reallocateRecord
    const ProgramManager* pm = m_context->getProgramManager();
    for( const auto& refid : pm->getReferencesForVariable( var->getToken() ) )
    {
        if( !m_resolvedSet.contains( refid ) )
            continue;

        if( newNode )
            m_context->getBindingManager()->addOrRemove_GraphNodeBinding( refid, newNode->getScopeID(), true );

        if( oldNode )
            m_context->getBindingManager()->addOrRemove_GraphNodeBinding( refid, oldNode->getScopeID(), false );
    }

    // Adjust attachment after we have processed the existing set of references.  This set
    // can change during attachment.  The changes will be processed during attachment.
    if( oldNode )
        this->attachOrDetachProperty_Attachment( oldNode, false );
    if( newNode )
        this->attachOrDetachProperty_Attachment( newNode, true );


    if( oldNode )
    {
        oldNode->receivePropertyDidChange_AttachedToVariable( false );
        oldNode->receivePropertyDidChange_RequiresTraversable( nullptr, false );
    }
    if( newNode )
    {
        newNode->receivePropertyDidChange_AttachedToVariable( true );
        newNode->receivePropertyDidChange_RequiresTraversable( nullptr, true );
    }
}

void LexicalScope::variableValueDidChange( Variable* var, Program* oldProgram, Program* newProgram )
{
    // Note: being conservative here, but we could choose to lazily update if necessary.
    writeRecord();
    subscribeForValidation();

    // Adjust semantic type and attachment properties
    if( oldProgram )
    {
        this->attachOrDetachProperty_Attachment( oldProgram, false );
        oldProgram->receivePropertyDidChange_UsedAsSemanticType( ST_BOUND_CALLABLE_PROGRAM, false );
    }

    if( newProgram )
    {
        newProgram->receivePropertyDidChange_UsedAsSemanticType( ST_BOUND_CALLABLE_PROGRAM, true );
        this->attachOrDetachProperty_Attachment( newProgram, true );
    }

    // Update program bindings.  See comment about performance in reallocateRecord
    for( const auto& refid : m_context->getProgramManager()->getReferencesForVariable( var->getToken() ) )
    {
        if( !m_resolvedSet.contains( refid ) )
            continue;

        if( newProgram )
            m_context->getBindingManager()->addOrRemove_ProgramBinding( this, newProgram, refid, true );
        if( oldProgram )
            m_context->getBindingManager()->addOrRemove_ProgramBinding( this, oldProgram, refid, false );
    }

    // Process and graph edits caused by changing the value of the
    // variable (required only for changing program variables)
    m_context->getBindingManager()->processVirtualParentQueue();
}

void LexicalScope::variableValueDidChange( Variable* var, Buffer* oldBuffer, Buffer* newBuffer )
{
    // Note: being conservative here, but we could choose to lazily update if necessary.
    writeRecord();
    subscribeForValidation();

    // Update buffer bindings.  See comment about performance in reallocateRecord
    const ProgramManager* pm = m_context->getProgramManager();
    for( const auto& refid : pm->getReferencesForVariable( var->getToken() ) )
    {
        if( !m_resolvedSet.contains( refid ) )
            continue;

        if( newBuffer )
            m_context->getBindingManager()->addOrRemove_BufferBinding( refid, newBuffer->getId(), true );

        if( oldBuffer )
            m_context->getBindingManager()->addOrRemove_BufferBinding( refid, oldBuffer->getId(), false );
    }

    // Change attachment on the buffer
    if( oldBuffer )
        this->attachOrDetachProperty_Attachment( oldBuffer, false );
    if( newBuffer )
        this->attachOrDetachProperty_Attachment( newBuffer, true );

    // Notify PostprocessingStage subclasses about the buffer variable change.
    bufferVariableValueDidChange( var, oldBuffer, newBuffer );
}

void LexicalScope::variableValueDidChange( Variable* var, TextureSampler* oldTexture, TextureSampler* newTexture )
{
    // Note: being conservative here, but we could choose to lazily update if necessary.
    writeRecord();
    subscribeForValidation();

    // Update texture bindings.  See comment about performance in reallocateRecord
    for( const auto& refid : m_context->getProgramManager()->getReferencesForVariable( var->getToken() ) )
    {
        if( !m_resolvedSet.contains( refid ) )
            continue;

        if( newTexture )
            m_context->getBindingManager()->addOrRemove_TextureBinding( refid, newTexture->getId(), true );
        if( oldTexture )
            m_context->getBindingManager()->addOrRemove_TextureBinding( refid, oldTexture->getId(), false );
    }

    // Change attachment on the texture sampler
    if( oldTexture )
        this->attachOrDetachProperty_Attachment( oldTexture, false );
    if( newTexture )
        this->attachOrDetachProperty_Attachment( newTexture, true );
}


size_t LexicalScope::getSafeOffset( const LexicalScope* scope )
{
    if( !scope )
        return 0;
    if( !scope->recordIsAllocated() )
        return LexicalScope::ILWALID_OFFSET;
    return scope->getRecordOffset();
}


//------------------------------------------------------------------------
// Object record management
//------------------------------------------------------------------------

char* LexicalScope::getObjectRecord() const
{
    char* objectRecord = m_context->getTableManager()->getObjectRecordHostPointer();
    return objectRecord + getRecordOffset();
}

unsigned short LexicalScope::getDynamicVariableTableCount() const
{
    return m_context->getObjectManager()->getVariableTokenCount();
}

void LexicalScope::offsetDidChange() const
{
    notifyParents_offsetDidChange();
}

size_t LexicalScope::getRecordOffset() const
{
    return *m_recordOffset;
}

size_t LexicalScope::getRecordSize() const
{
    return m_recordSize;
}

bool LexicalScope::recordIsAllocated() const
{
    return m_recordOffset != nullptr;
}

void LexicalScope::releaseRecord()
{
    m_recordOffset.reset();
    releaseDynamicVariableTable();
}

void LexicalScope::reallocateRecord()
{
    // Assume that object is 16 byte aligned.
    unsigned int start = static_cast<unsigned int>( getRecordBaseSize() );

    // dynamicVariableTable needs only be updated whenever a variable or a variable offset changes
    bool needToRewriteDylwarTable = false;

    for( Variable* variable : m_variables )
    {
        unsigned int oldOffset = variable->getScopeOffset();
        if( variable->getType().isValid() )
        {
            // Pad if necessary
            start = align( start, variable->getAlignment() );
            variable->setScopeOffset( start );
            start += variable->getSize();
        }
        else
        {  // TODO: treat unknown as size==0, rethink this when we delay writeRecord till compile time
            variable->setScopeOffset( start );
        }

        // Register the new binding locations if necessary
        unsigned int newOffset = variable->getScopeOffset();
        if( oldOffset != newOffset )
        {
            needToRewriteDylwarTable                              = true;
            const ProgramManager*                              pm = m_context->getProgramManager();
            const ProgramManager::VariableReferenceIDListType& refids = pm->getReferencesForVariable( variable->getToken() );
            for( unsigned int refid : refids )
            {
                // Note, this could be inefficient if there are many programs
                // with identical variable names.  Revisit if this shows up in
                // a profile.  Consider storing the valid binding list with
                // the variable.
                if( m_resolvedSet.contains( refid ) )
                {

                    if( k_scopeTrace.get() )
                    {
                        lprint << "scope: moving binding for reference " << refid  // clang-format fail
                               << " from " << VariableReferenceBinding( getClass(), oldOffset ).toString()  //
                               << " to " << VariableReferenceBinding( getClass(), newOffset ).toString() << '\n';
                    }

                    // Variable is resolved by this scope, so remove the old
                    // binding and add a new one.
                    m_context->getBindingManager()->addOrRemove_VariableBinding(
                        refid, VariableReferenceBinding( getClass(), oldOffset ), false );
                    m_context->getBindingManager()->addOrRemove_VariableBinding(
                        refid, VariableReferenceBinding( getClass(), newOffset ), true );
                }
            }
        }
    }

    if( start != m_recordSize || !m_recordOffset )
    {
        // Note that we can get here even with !needToRewriteDylwarTable and !m_variables.empty(),
        // if for instance only the last variable will be set for the first time. Ie, all offsets
        // stay unchanged, but the formerly invalid variable gets now added to start.
        needToRewriteDylwarTable = true;

        // Release old handle first so that the old block will be available for reuse
        size_t old_recordOffset = m_recordOffset ? *m_recordOffset : size_t( -1 );
        m_recordOffset.reset();
        m_recordSize   = start;
        m_recordOffset = m_context->getTableManager()->allocateObjectRecord( m_recordSize );

        // optimization to opt out when called from GlobalScope::reallocateRecord() and allocation
        // at offset 0 failed. Otherwise unneccesary updates and writes would happen.
        if( *m_recordOffset && this == m_context->getGlobalScope() )
            return;

        // Notify all objects that maintain pointers to us that our offset has changed
        if( old_recordOffset != *m_recordOffset )
            offsetDidChange();
    }

    // important to call the virtual function here and not for instance just writeObjectRecord().
    // Otherwise the writeRecord() calls to Program, Material, Geometry etc would be missing.
    writeRecord();
    if( needToRewriteDylwarTable )
    {
        // Can we avoid the reallocation? The call to reallocateDynamicVariableTable() will decide.
        reallocateDynamicVariableTable();
        writeDynamicVariableTable();
    }
}

void LexicalScope::writeRecord() const
{
    writeObjectRecord();
}

void LexicalScope::writeObjectRecord() const
{
    if( !recordIsAllocated() )
        return;
    cort::LexicalScopeRecord* ls = getObjectRecord<cort::LexicalScopeRecord>();
    RT_ASSERT( ls != nullptr );
    for( auto variable : m_variables )
    {
        // TODO: treat unknown as size==0 or just skip it, rethink this later
        if( !variable->getType().isValid() )
            continue;  // I found an unset variable, nevermind.. carry on

        variable->writeRecord( (char*)ls );
    }
}

//------------------------------------------------------------------------
// DynamicVariableTable record management
//------------------------------------------------------------------------

void LexicalScope::reallocateDynamicVariableTable()
{
    if( m_tokenToVarMap.empty() )
        return;

    // each table entry holds two short, one for the token id and the internal offset each. Note
    // that we might have to increase the size of the table a bit to allow for proper binary search
    // in the array representing the table - hence we are using a dedicated function for that.
    size_t dynamicVariableTableSize = getProperDynamicVariableTableSize( m_tokenToVarMap.size() ) * 2 * sizeof( unsigned short );
    if( dynamicVariableTableSize != m_dynamicVariableTableSize || !m_dynamicVariableTableOffset )
    {
        if( !m_dynamicVariableTableOffset && !dynamicVariableTableSize )
        {
            RT_ASSERT_FAIL_MSG( "How can this happen when !m_tokenToVarMap.empty()?" );
            return;
        }
        // Release old handle first so that the old block will be available for reuse
        //size_t old_dVTOffset = m_dynamicVariableTableOffset ? *m_dynamicVariableTableOffset : size_t( -1 );
        m_dynamicVariableTableOffset.reset();
        m_dynamicVariableTableSize = dynamicVariableTableSize;
        m_dynamicVariableTableOffset = m_context->getTableManager()->allocateDynamicVariableTable( m_dynamicVariableTableSize );

        // are there any objects maintaining pointers to us that require notification about the offset change?
        // No, hence no notfication on top of the ones inside reallocateRecord(). Otherwise something along
        // if( old_dVTOffset != *m_dynamicVariableTableOffset ) anyUpdateRequired()?
    }
}

void LexicalScope::writeDynamicVariableTable() const
{
    cort::LexicalScopeRecord* ls = getObjectRecord<cort::LexicalScopeRecord>();
    RT_ASSERT( ls != nullptr );

    if( !dynamicVariableTableIsAllocated() )
    {
        // don't leave the dynamicVariableTable offset in the record uninitialized
        // Otherwise the device code has no chance to find out whether a dynamic variable
        // table exists
        ls->dynamicVariableTable = -1;
        return;
    }

    // store offset in record
    ls->dynamicVariableTable = *m_dynamicVariableTableOffset;

    // write entries into table - here we rely on a properly allocated table
    unsigned short* table = (unsigned short*)getDynamicVariableTable();
    implicitBinTreeTransform( getVariableEntries( m_tokenToVarMap ), table );
}

char* LexicalScope::getDynamicVariableTable() const
{
    char* dynamicVariableTable = m_context->getTableManager()->getDynamicVariableTableHostPointer();
    return dynamicVariableTable + *m_dynamicVariableTableOffset;
}

bool LexicalScope::dynamicVariableTableIsAllocated() const
{
    return m_dynamicVariableTableOffset != nullptr;
}

void LexicalScope::releaseDynamicVariableTable()
{
    m_dynamicVariableTableOffset.reset();
}

//------------------------------------------------------------------------
// Unresolved reference property
//------------------------------------------------------------------------

static inline void computeReferenceResolutionLogic( bool bIN, bool bV, bool& bR, bool& bOUT )
{
    /*
   * IN = union(all children.out)             // Counting set.
   * R = intersect(IN, V)                     // Resolution change
   * OUT = IN - V                             // Notify Context
   */

    bR   = bIN && bV;
    bOUT = bIN && !bV;
}

void LexicalScope::updateResolvedSet( VariableReferenceID refid, bool addToResolvedSet )
{
    bool changed = m_resolvedSet.addOrRemoveProperty( refid, addToResolvedSet );
    if( changed && addToResolvedSet )
        // Schedule type-check (only necessary for additions)
        subscribeForValidation();

    // Note: this function can be made faster by indexing the variables
    // by token instead of string name.
    const ProgramManager*    pm       = m_context->getProgramManager();
    const VariableReference* varref   = pm->getVariableReferenceById( refid );
    Variable*                variable = getVariableByToken( varref->getVariableToken() );

    if( k_scopeTrace.get() )
    {
        lprint << "scope: " << ( addToResolvedSet ? "creating" : "dissolving" ) << " binding for reference " << refid
               << " at " << VariableReferenceBinding( getClass(), variable->getScopeOffset() ).toString() << '\n';
    }

    // Add/remove bindings to variable values
    m_context->getBindingManager()->addOrRemove_VariableBinding(
        refid, VariableReferenceBinding( getClass(), variable->getScopeOffset() ), addToResolvedSet );

    switch( variable->getType().baseType() )
    {
        case VariableType::Buffer:
        case VariableType::DemandBuffer:
            if( Buffer* buffer = variable->getBuffer() )
                m_context->getBindingManager()->addOrRemove_BufferBinding( refid, buffer->getId(), addToResolvedSet );
            break;
        case VariableType::GraphNode:
            if( GraphNode* graphNode = variable->getGraphNode() )
                m_context->getBindingManager()->addOrRemove_GraphNodeBinding( refid, graphNode->getScopeID(), addToResolvedSet );
            break;
        case VariableType::Program:
            if( Program* program = variable->getProgram() )
                m_context->getBindingManager()->addOrRemove_ProgramBinding( this, program, refid, addToResolvedSet );
            break;
        case VariableType::TextureSampler:
            if( TextureSampler* ts = variable->getTextureSampler() )
                m_context->getBindingManager()->addOrRemove_TextureBinding( refid, ts->getId(), addToResolvedSet );
            break;
        case VariableType::Int:
            if( varref->getType().baseType() == VariableType::ProgramId )
            {
                //establish link to bindless callable from variable
                int      programId = variable->get<int>();
                Program* program   = m_context->getObjectManager()->getProgramByIdNoThrow( programId );

                if( !program && programId > 0 )
                {
                    // program id must be valid or 0, otherwise it is a programming bug.
                    ureport2( m_context->getUsageReport(), "CALLABLE TRACKING" )
                        << "Warning: Found invalid program ID in variable \"" << variable->getName()
                        << "\": " << programId << ". Setting it to 0.\n";
                }
                variable->setBindlessProgram( program );

                m_context->getBindingManager()->addOrRemove_ProgramIdBinding( refid, programId, addToResolvedSet );
            }
            break;
        default:
            break;
    }
}

void LexicalScope::variableDeclarationDidChange( Variable* var, bool variableWasAdded )
{
    const ProgramManager*                              pm     = m_context->getProgramManager();
    const ProgramManager::VariableReferenceIDListType& refids = pm->getReferencesForVariable( var->getToken() );
    for( unsigned int refid : refids )
    {
        variableDeclarationDidChange( refid, variableWasAdded );
    }
}

void LexicalScope::variableDeclarationDidChange( VariableReferenceID refid, bool variableWasAdded )
{
    scopeTrace( "begin variableDeclarationDidChange", refid, variableWasAdded );

    // Callwlate new/old input bits
    bool old_IN = m_unresolvedSet.contains( refid );
    bool new_IN = old_IN;
    bool old_V  = !variableWasAdded;
    bool new_V  = variableWasAdded;

    // Callwlate derived sets
    bool old_R, old_OUT;
    computeReferenceResolutionLogic( old_IN, old_V, old_R, old_OUT );
    bool new_R, new_OUT;
    computeReferenceResolutionLogic( new_IN, new_V, new_R, new_OUT );

    // Propagate changes if necessary
    if( old_OUT != new_OUT )
        sendPropertyDidChange_UnresolvedReference( refid, new_OUT );
    if( old_R != new_R )
        updateResolvedSet( refid, new_R );

    scopeTrace( "end variableDeclarationDidChange", refid, variableWasAdded );
}

void LexicalScope::computeUnresolvedOutputForDebugging( GraphProperty<VariableReferenceID, false>& out ) const
{
    for( auto refid : m_unresolvedSet )
    {
        bool bIN = true;
        bool bV  = haveVariableForReference( refid );
        // Callwlate derived sets
        bool bR, bOUT;
        computeReferenceResolutionLogic( bIN, bV, bR, bOUT );
        if( bOUT )
            out.addOrRemoveProperty( refid, true );
    }
}

bool LexicalScope::haveResolutionForReference( VariableReferenceID refid ) const
{
    return m_resolvedSet.contains( refid );
}

const GraphProperty<VariableReferenceID, false>& LexicalScope::getResolvedReferenceSet() const
{
    return m_resolvedSet;
}

void LexicalScope::receivePropertyDidChange_UnresolvedReference( const LexicalScope* child, VariableReferenceID refid, bool added )
{
    scopeTrace( "begin receivePropertyDidChange_UnresolvedReference", refid, added, child );

    // Callwlate new/old input bits
    bool setChanged = m_unresolvedSet.addOrRemoveProperty( refid, added );
    bool old_IN     = !added || !setChanged;
    bool new_IN     = added || !setChanged;
    bool old_V      = haveVariableForReference( refid );
    bool new_V      = old_V;

    // Callwlate derived sets
    bool old_R, old_OUT;
    computeReferenceResolutionLogic( old_IN, old_V, old_R, old_OUT );
    bool new_R, new_OUT;
    computeReferenceResolutionLogic( new_IN, new_V, new_R, new_OUT );

    // Propagate changes if necessary
    if( old_OUT != new_OUT )
        scopeTrace( "receivePropertyDidChange_UnresolvedReference send", refid, new_OUT, child );

    if( old_OUT != new_OUT )
        sendPropertyDidChange_UnresolvedReference( refid, new_OUT );
    if( old_R != new_R )
        updateResolvedSet( refid, new_R );

    scopeTrace( "end receivePropertyDidChange_UnresolvedReference", refid, added, child );
}


//------------------------------------------------------------------------
// Attachment property
//------------------------------------------------------------------------

void LexicalScope::sendPropertyDidChangeToVariables_Attachment( bool added ) const
{
    // We need to notify select attached objects of attachment
    for( Variable* variable : m_variables )
    {
        switch( variable->getType().baseType() )
        {
            case VariableType::Buffer:
            case VariableType::DemandBuffer:
                if( Buffer* buffer = variable->getBuffer() )
                    buffer->receivePropertyDidChange_Attachment( added );
                break;
            case VariableType::GraphNode:
                if( GraphNode* node = variable->getGraphNode() )
                    node->receivePropertyDidChange_Attachment( added );
                break;
            case VariableType::Program:
                if( Program* program = variable->getProgram() )
                    program->receivePropertyDidChange_Attachment( added );
                break;
            case VariableType::TextureSampler:
                if( TextureSampler* texture = variable->getTextureSampler() )
                    texture->receivePropertyDidChange_Attachment( added );
                break;
            default:
                break;
        }
    }
}

bool LexicalScope::isAttached() const
{
    return !m_attachment.empty();
}

void LexicalScope::attachOrDetachProperty_Attachment( LexicalScope* dest, bool attached ) const
{
    // Only pay attention to links from attached nodes
    if( isAttached() )
        dest->receivePropertyDidChange_Attachment( attached );
}

void LexicalScope::attachOrDetachProperty_Attachment( Buffer* dest, bool attached ) const
{
    if( isAttached() )
        dest->receivePropertyDidChange_Attachment( attached );
}

void LexicalScope::attachOrDetachProperty_Attachment( TextureSampler* dest, bool attached ) const
{
    if( isAttached() )
        dest->receivePropertyDidChange_Attachment( attached );
}

void LexicalScope::receivePropertyDidChange_Attachment( bool added )
{
    bool setChanged = m_attachment.addOrRemoveProperty( added );
    if( setChanged )
    {
        sendPropertyDidChangeToVariables_Attachment( added );
        sendPropertyDidChange_Attachment( added );
        attachmentDidChange( added );
        if( added )
        {
            // Update rtx traversal mode when a node is attached so all attached nodes agree on the traversal mode
            const bool universalChanged = ( m_rtxUniversalTraversal != m_context->RtxUniversalTraversalEnabled() );
            if( universalChanged )
            {
                m_rtxUniversalTraversal = m_context->RtxUniversalTraversalEnabled();
                rtxUniversalTraversalDidChange();
            }

            // Re-validate the object
            subscribeForValidation();
        }
    }
}

//------------------------------------------------------------------------
// RtxUniversalTraversal
//------------------------------------------------------------------------

void LexicalScope::attachOrDetachProperty_RtxUniversalTraversal( LexicalScope* dest, bool attached ) const
{
    // Only pay attention to links from attached nodes
    // Unattached nodes may retain their current universal state
    if( isAttached() && attached )
        dest->receivePropertyDidChange_RtxUniversalTraversal();
}

void LexicalScope::receivePropertyDidChange_RtxUniversalTraversal()
{
    bool setChanged = ( m_rtxUniversalTraversal != m_context->RtxUniversalTraversalEnabled() );
    if( setChanged )
    {
        m_rtxUniversalTraversal = m_context->RtxUniversalTraversalEnabled();
        sendPropertyDidChange_RtxUniversalTraversal();
        rtxUniversalTraversalDidChange();
    }
}

//------------------------------------------------------------------------
// Direct Caller
//------------------------------------------------------------------------

void LexicalScope::receivePropertyDidChange_DirectCaller( CanonicalProgramID cpid, bool added )
{
    bool changed = m_directCaller.addOrRemoveProperty( cpid, added );
    if( changed )
        sendPropertyDidChange_DirectCaller( cpid, added );
}


//------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------

Program* LexicalScope::getSharedNullProgram() const
{
    return m_context->getSharedProgramManager()->getNullProgram();
}

void LexicalScope::subscribeForValidation()
{
    m_context->getValidationManager()->subscribeForValidation( this );
}

bool LexicalScope::haveVariableForReference( VariableReferenceID refid ) const
{
    // Note: this function can be made faster by indexing the variables
    // by token instead of string name.  Some of the calls can also be
    // short-cirlwited.
    const ProgramManager*    pm     = m_context->getProgramManager();
    const VariableReference* varref = pm->getVariableReferenceById( refid );
    return getVariableByToken( varref->getVariableToken() ) != nullptr;
}

void LexicalScope::deleteVariables()
{
    //
    // Variables belong to the object that created them
    //
    while( !m_variables.empty() )
        removeVariable( m_variables.back() );
    RT_ASSERT( m_variables.empty() );
}

void LexicalScope::scopeTrace( const char* message, VariableReferenceID refid, bool added, const LexicalScope* fromScope ) const
{
    if( k_scopeTrace.get() )
    {
        lprint << "scope ID: " << getScopeID() << " " << getNameForClass( getClass() ) << " " << message;
        if( refid == (VariableReferenceID)~0 )
        {
            lprint << ( added ? " attached" : " detached" );
        }
        else
        {
            lprint << " r" << refid << " " << ( added ? "added" : "removed" );
        }

        if( fromScope )
        {
            lprint << " (from scope ID " << fromScope->getScopeID() << ")";
        }
        lprint << '\n';
    }
}

void LexicalScope::checkName( const std::string& name ) const
{
    std::string::const_iterator iter = name.begin();
    if( iter == name.end() )
        throw IllegalSymbol( RT_EXCEPTION_INFO, "(Empty variable name found)" );

    // Symbol must begin with an letter or underscore
    if( !std::isalpha( *iter ) && *iter != '_' )
        throw IllegalSymbol( RT_EXCEPTION_INFO, name );
    ++iter;

    // Remainder must be [A-Za-z0-9_:]
    // Colon added to support namespaces
    for( ; iter != name.end(); ++iter )
        if( !std::isalnum( *iter ) && *iter != '_' && *iter != ':' )
            throw IllegalSymbol( RT_EXCEPTION_INFO, name );
}

void LexicalScope::bufferVariableValueDidChange( Variable* var, Buffer* oldBuffer, Buffer* newBuffer )
{
    // Empty default implementation.
}
