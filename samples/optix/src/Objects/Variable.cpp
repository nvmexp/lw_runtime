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

#include <Objects/Variable.h>

#include <Exceptions/TypeMismatch.h>
#include <prodlib/exceptions/Assert.h>
#include <prodlib/exceptions/IlwalidValue.h>

#include <cstring>  // memset, memcpy
#include <sstream>

using namespace optix;
using namespace prodlib;
using namespace corelib;

Variable::Variable( LexicalScope* scope, const std::string& name, unsigned int index, unsigned short token )
    : m_class( RT_OBJECT_VARIABLE )
    , m_scope( scope )
    , m_name( name )
    , m_token( token )
    , m_index( index )
    , m_scopeOffset( ILWALID_OFFSET )
    , m_type( VariableType::Unknown, 0 )
    , m_matrixDim( make_uint2( 0, 0 ) )
    , m_ptr( m_databuf )
{
    memset( &m_databuf, 0x00, sizeof( m_databuf ) );
}

Variable::~Variable()
{
    if( m_ptr != m_databuf )
    {
        delete[] static_cast<char*>( m_ptr );
        m_ptr = m_databuf;
    }

    m_type        = VariableType( VariableType::Unknown, 0 );
    m_scopeOffset = ILWALID_OFFSET;
}

ObjectClass Variable::getClass() const
{
    return m_class;
}

LexicalScope* Variable::getScope() const
{
    return m_scope;
}

const std::string& Variable::getName() const
{
    return m_name;
}

unsigned int Variable::getIndex() const
{
    return m_index;
}

unsigned int Variable::getScopeOffset() const
{
    return m_scopeOffset;
}

void Variable::setScopeOffset( unsigned int newOffset )
{
    m_scopeOffset = newOffset;
}

void Variable::writeRecord( char* scope_base ) const
{
    char* var_base = scope_base + getScopeOffset();
    switch( m_type.baseType() )
    {
        case VariableType::Float:
        case VariableType::Int:
        case VariableType::Uint:
        case VariableType::LongLong:
        case VariableType::ULongLong:
        case VariableType::ProgramId:
        case VariableType::BufferId:
            memcpy( var_base, m_ptr, getSize() );
            break;
        case VariableType::UserData:
            getUserData( getSize(), var_base );
            break;
        case VariableType::Buffer:
        case VariableType::DemandBuffer:
        {
            Buffer* buffer = getBuffer();
            int     id     = buffer ? buffer->getId() : 0;
            RT_ASSERT( sizeof( id ) == getSize() );
            *reinterpret_cast<int*>( var_base ) = id;
            break;
        }
        case VariableType::TextureSampler:
        {
            TextureSampler* sampler = getTextureSampler();
            int             id      = sampler ? sampler->getId() : 0;
            RT_ASSERT( sizeof( id ) == getSize() );
            *reinterpret_cast<int*>( var_base ) = id;
            break;
        }
        case VariableType::Program:
        {
            Program* program = getProgram();
            int      id      = program ? program->getId() : 0;
            RT_ASSERT( sizeof( id ) == getSize() );
            *reinterpret_cast<int*>( var_base ) = id;
            break;
        }
        case VariableType::GraphNode:
        {
            GraphNode*   node = getGraphNode();
            unsigned int off  = LexicalScope::getSafeOffset( node );
            RT_ASSERT( sizeof( off ) == getSize() );
            *reinterpret_cast<int*>( var_base ) = off;
            break;
        }
        case VariableType::Ray:
        case VariableType::Unknown:
        default:
            throw IlwalidValue( RT_EXCEPTION_INFO, "Illegal variable type in writeRecord" );
            break;
    }
}

void Variable::detachLinkedChild( const LinkedPtr_Link* link )
{
    if( link == &m_graphNode )
        setGraphNode( nullptr );

    else if( link == &m_buffer )
        setBuffer( nullptr );

    else if( link == &m_program )
        setProgram( nullptr );

    else if( link == &m_textureSampler )
        setTextureSampler( nullptr );

    else if( link == &m_bindlessProgram )
        setBindlessProgram( nullptr );

    else
        RT_ASSERT_FAIL_MSG( "Invalid child link in detachLinkedChild" );
}

unsigned int Variable::getSize() const
{
    return m_type.computeSize();
}

unsigned int Variable::getAlignment() const
{
    return m_type.computeAlignment();
}

unsigned short Variable::getToken() const
{
    return m_token;
}

VariableType Variable::getType() const
{
    return m_type;
}

bool Variable::isMatrix() const
{
    return m_matrixDim.x != 0;
}

bool Variable::isGraphNode() const
{
    return getType().isGraphNode();
}

bool Variable::isBuffer() const
{
    return getType().isBuffer();
}

bool Variable::isTextureSampler() const
{
    return getType().isTextureSampler();
}

bool Variable::isProgram() const
{
    return getType().isProgram();
}

void Variable::setUserData( size_t size, const void* ptr )
{
    const bool changed = !m_type.isValid();
    setOrCheckType( VariableType( VariableType::UserData, size ) );

    // Copy data
    if( changed || ( memcmp( m_ptr, ptr, size ) != 0 ) )
    {
        memcpy( m_ptr, ptr, size );
        getScope()->variableValueDidChange( this );
    }
}

void Variable::setGraphNode( GraphNode* node )
{
    setOrCheckType( VariableType( VariableType::GraphNode, 1 ) );
    GraphNode* oldNode = m_graphNode.get();

    if( oldNode == node )
        return;

    m_graphNode.set( this, node );
    getScope()->variableValueDidChange( this, oldNode, node );
}

void Variable::setProgram( Program* newProgram )
{
    VariableType newType;
    if( newProgram )
        newType = VariableType::createForCallableProgramVariable( newProgram->getFunctionSignature() );
    else
        newType = VariableType::createForProgramVariable();

    setOrCheckType( newType );
    Program* oldProgram = m_program.get();

    if( oldProgram == newProgram )
        return;

    m_program.set( this, newProgram );

    // Always override the type, since the signature may have changed.
    m_type = newType;
    getScope()->variableTypeDidChange( this );
    getScope()->variableValueDidChange( this, oldProgram, newProgram );
}

void Variable::setBindlessProgram( Program* newProgram )
{
    Program* oldProgram = m_bindlessProgram.get();

    if( oldProgram != newProgram )
        m_bindlessProgram.set( this, newProgram );

    if( !newProgram && get<int>() > 0 )
    {
        // reset variable value to avoid pointing
        // to a non existing program
        int* val = static_cast<int*>( m_ptr );
        val[0]   = 0;
        getScope()->variableValueDidChange( this );
    }
}

void Variable::setBuffer( Buffer* buffer )
{
    setOrCheckType( VariableType::createForBufferVariable( buffer && buffer->isDemandLoad() ) );
    Buffer* oldBuffer = m_buffer.get();

    if( oldBuffer == buffer )
        return;

    m_buffer.set( this, buffer );
    if( buffer )
    {
        bufferFormatDidChange();
    }
    getScope()->variableValueDidChange( this, oldBuffer, buffer );
}

void Variable::setTextureSampler( TextureSampler* ts )
{
    setOrCheckType( VariableType::createForTextureSamplerVariable() );
    TextureSampler* oldTexture = m_textureSampler.get();

    if( oldTexture == ts )
        return;

    m_textureSampler.set( this, ts );
    if( ts )
    {
        textureSamplerFormatDidChange();
    }
    getScope()->variableValueDidChange( this, oldTexture, ts );
}

void Variable::getUserData( size_t size, void* ptr ) const
{
    checkType( VariableType( VariableType::UserData, size ) );
    memcpy( ptr, m_ptr, size );
}

uint2 Variable::getMatrixDim() const
{
    return m_matrixDim;
}

GraphNode* Variable::getGraphNode() const
{
    RT_ASSERT( getType() == VariableType( VariableType::GraphNode, 1 ) );
    return m_graphNode.get();
}

Program* Variable::getProgram() const
{
    RT_ASSERT( getType().baseType() == VariableType::Program );  // we check signature in validation manager
    return m_program.get();
}

Buffer* Variable::getBuffer() const
{
    RT_ASSERT( getType().isBuffer() );
    return m_buffer.get();
}

TextureSampler* Variable::getTextureSampler() const
{
    RT_ASSERT( getType().isTextureSampler() );
    return m_textureSampler.get();
}

void Variable::setOrCheckType( const VariableType& type )
{
    if( m_type.isValid() )
    {
        checkType( type );
    }
    else
    {
        size_t size = type.computeSize();
        if( size <= sizeof( m_databuf ) )
        {
            m_ptr = m_databuf;
        }
        else
        {
            m_ptr = new char[size];
        }
        m_type = type;

        // Reallocate this variable (setting a new type always changes the
        // size)
        getScope()->variableTypeDidChange( this );
        getScope()->variableSizeDidChange( this );
    }
}

void Variable::checkType( const VariableType& type ) const
{
    // Buffers, programs and texture samplers are allowed to change size, but not base type
    if( isBuffer() && type.isBuffer() )
        return;
    if( isTextureSampler() && type.isTextureSampler() )
        return;
    // Program signature validation we do in validation manager
    if( isProgram() && type.isProgram() )
        return;

    // Everything else has an immutable type
    if( type == m_type )
        return;

    std::ostringstream out;
    out << "Variable \"" << m_name << "\" assigned value of type \"" << type.toString() << "\" to variable of type \""
        << m_type.toString() << "\"";
    throw TypeMismatch( RT_EXCEPTION_INFO, out.str() );
}

void Variable::bufferFormatDidChange()
{
    RT_ASSERT( isBuffer() );
    int elementSize = 0;
    if( m_buffer->isElementSizeInitialized() )
        elementSize = m_buffer->getElementSize();
    m_type          = VariableType( m_type.baseType(), elementSize, m_buffer->getDimensionality() );
    getScope()->variableTypeDidChange( this );
}

void Variable::textureSamplerFormatDidChange()
{
    RT_ASSERT( isTextureSampler() );
    // Texture samplers' elementSize is always expected to be 1
    // They are untyped, so this is just by convention
    m_type = VariableType( VariableType::TextureSampler, 1, m_textureSampler->getDimensionality() );
    getScope()->variableTypeDidChange( this );
}

void Variable::graphNodeOffsetDidChange()
{
    // This variable points to a LexicalScope object in the object record.  Tell the
    // containing scope that the value (offset) of this variable has changed.
    getScope()->variableValueDidChange( this );
}
