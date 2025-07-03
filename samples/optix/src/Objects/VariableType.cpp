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

#include <Objects/VariableType.h>
#include <Util/PersistentStream.h>
#include <prodlib/exceptions/Assert.h>

using namespace optix;

static const unsigned int TYPE_SHIFT           = 0u;
static const unsigned int TYPE_MASK            = 0xfu;  // 4 bits
static const unsigned int DIMENSIONALITY_SHIFT = 4u;
static const unsigned int DIMENSIONALITY_MASK  = 0x3u;  // 2 bits
static const unsigned int SIZE_SHIFT           = 8u;
static const unsigned int SIZE_MASK            = 0x00ffffffu;  // 24 bits
// 2 unused bits

// Here are some static asserts to make sure that we have the expected sizes
static_assert( sizeof( VariableType ) == 4, "Invalid size" );
static_assert( VariableType::Unknown <= TYPE_MASK, "Too few bits for variable types" );

VariableType::VariableType()
{
    pack( Unknown, 0, 0 );
}

VariableType::VariableType( Type type, unsigned int size, unsigned int dimensionality )
{
    pack( type, size, dimensionality );
}

VariableType::~VariableType()
{
}

bool VariableType::operator==( const VariableType& b ) const
{
    return m_packedType == b.m_packedType;
}

bool VariableType::operator!=( const VariableType& b ) const
{
    return m_packedType != b.m_packedType;
}

VariableType::Type VariableType::baseType() const
{
    return static_cast<Type>( ( m_packedType >> TYPE_SHIFT ) & TYPE_MASK );
}

unsigned int VariableType::numElements() const
{
    return ( m_packedType >> SIZE_SHIFT ) & SIZE_MASK;
}

unsigned int VariableType::programSignatureToken() const
{
    RT_ASSERT( baseType() == Program );
    return numElements();
}

unsigned int VariableType::bufferDimensionality() const
{
    return ( m_packedType >> DIMENSIONALITY_SHIFT ) & DIMENSIONALITY_MASK;
}

bool VariableType::isValid() const
{
    return baseType() != Unknown;
}

bool VariableType::isGraphNode() const
{
    return baseType() == GraphNode;
}

bool VariableType::isBuffer() const
{
    return baseType() == Buffer || isDemandBuffer();
}

bool VariableType::isDemandBuffer() const
{
    return baseType() == DemandBuffer;
}

bool VariableType::isTextureSampler() const
{
    return baseType() == TextureSampler;
}

bool VariableType::isProgram() const
{
    return baseType() == Program;
}

bool VariableType::isProgramId() const
{
    return baseType() == ProgramId;
}

bool VariableType::isBufferId() const
{
    return baseType() == BufferId;
}

bool VariableType::isProgramOrProgramId() const
{
    // Bindless callable program IDs come from variables of type int
    return ( baseType() == Program ) || ( baseType() == Int ) || ( baseType() == Uint ) || ( baseType() == ProgramId );
}

bool VariableType::isTypeWithValidDefaultValue() const
{
    switch( baseType() )
    {
        // clang-format off
        case VariableType::Float:          return true;
        case VariableType::Int:            return true;
        case VariableType::Uint:           return true;
        case VariableType::LongLong:       return true;
        case VariableType::ULongLong:      return true;
        case VariableType::ProgramId:      return true;
        case VariableType::BufferId:       return true;
        case VariableType::UserData:       return true;
        case VariableType::Buffer:         return false;
        case VariableType::DemandBuffer:   return false;
        case VariableType::GraphNode:      return false;
        case VariableType::Program:        return false;
        case VariableType::TextureSampler: return false;
        case VariableType::Ray:            return false;
        case VariableType::Unknown:        return false;
            // clang-format on
    }
    return false;
}


size_t VariableType::computeSize() const
{
    switch( baseType() )
    {
        // clang-format off
        case VariableType::Unknown:        return 0;
        case VariableType::Float:          return 4*numElements();
        case VariableType::Int:            return 4*numElements();
        case VariableType::Uint:           return 4*numElements();
        case VariableType::LongLong:       return 8*numElements();
        case VariableType::ULongLong:      return 8*numElements();
        case VariableType::Buffer:         return 4;
        case VariableType::DemandBuffer:   return 4;
        case VariableType::GraphNode:      return 4;
        case VariableType::Program:        return 4;
        case VariableType::ProgramId:      return 4;
        case VariableType::BufferId:       return 4;
        case VariableType::TextureSampler: return 4;
        case VariableType::UserData:       return numElements();
        case VariableType::Ray:            return 4*9;
            // clang-format on
    }
    return 0;
}

size_t VariableType::computeAlignment() const
{
    size_t size = computeSize();
    if( size >= 9 )
        return 16;
    else if( size >= 5 )
        return 8;
    else if( size >= 3 )
        return 4;
    else if( size >= 2 )
        return 2;
    else
        return 1;
}

std::string VariableType::toString() const
{
    std::string result;
    switch( baseType() )
    {
        // clang-format off
        case Float:          result += "float";          break;
        case Int:            result += "int";            break;
        case Uint:           result += "uint";           break;
        case LongLong:       result += "longlong";       break;
        case ULongLong:      result += "ulonglong";      break;
        case UserData:       result += "User data";      break;
        case Buffer:         result += "Buffer";         break;
        case DemandBuffer:   result += "Demand Buffer";  break;
        case GraphNode:      result += "Node object";    break;
        case Program:        result += "Program object"; break;
        case ProgramId:      result += "Program ID";     break;
        case BufferId:       result += "Buffer ID";      break;
        case TextureSampler: result += "Texture object"; break;
        case Unknown:        result += "Unknown type";   break;
        default:             result += "INVALID type";   break;
            // clang-format on
    }

    // Add additional info for some types
    if( baseType() == UserData )
    {
        result += "[" + std::to_string( numElements() ) + " bytes]";
    }
    else if( isBuffer() || isTextureSampler() )
    {
        result +=
            "(" + std::to_string( bufferDimensionality() ) + "d, " + std::to_string( numElements() ) + " byte element)";
    }
    else if( numElements() != 1 && baseType() != Unknown )
    {
        result += std::to_string( numElements() );
    }
    return result;
}

void VariableType::pack( Type type, unsigned int size, unsigned int dimensionality )
{
    if( type == Buffer || type == DemandBuffer || type == TextureSampler )
    {
        RT_ASSERT( dimensionality > 0 );
    }
    else if( type == Program )
    {
        RT_ASSERT( dimensionality == 0 );
    }
    else
    {
        RT_ASSERT( dimensionality == 0 );
        if( size == 0 )
            RT_ASSERT( type == Unknown || type == UserData );
        else if( size == 1 )
            RT_ASSERT( type != Unknown );
        else if( size != 1 )
            RT_ASSERT( type == UserData || type == Float || type == Int || type == Uint || type == LongLong || type == ULongLong );
    }

    RT_ASSERT( static_cast<unsigned int>( type ) <= TYPE_MASK );
    RT_ASSERT( size <= SIZE_MASK );
    RT_ASSERT( dimensionality <= DIMENSIONALITY_MASK );
    m_packedType = ( static_cast<unsigned int>( type ) << TYPE_SHIFT ) | ( size << SIZE_SHIFT )
                   | ( dimensionality << DIMENSIONALITY_SHIFT );
}


VariableType VariableType::createForBufferVariable( bool isDemandLoaded )
{
    VariableType vtForBuffer;
    vtForBuffer.m_packedType =
        ( static_cast<unsigned int>( isDemandLoaded ? VariableType::DemandBuffer : VariableType::Buffer ) << TYPE_SHIFT );
    return vtForBuffer;
}

VariableType VariableType::createForTextureSamplerVariable()
{
    VariableType vtForTexSampler;
    vtForTexSampler.m_packedType = ( static_cast<unsigned int>( VariableType::TextureSampler ) << TYPE_SHIFT );
    return vtForTexSampler;
}

VariableType VariableType::createForProgramVariable()
{
    return VariableType( VariableType::Program, /*signature*/ (unsigned short)~0 );
}

VariableType VariableType::createForCallableProgramVariable( unsigned sig )
{
    return VariableType( VariableType::Program, sig );
}

void optix::readOrWrite( PersistentStream* stream, VariableType* vtype, const char* label )
{
    readOrWrite( stream, &vtype->m_packedType, label );
}
