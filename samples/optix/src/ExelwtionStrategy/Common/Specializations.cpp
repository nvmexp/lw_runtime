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

#include <ExelwtionStrategy/Common/Specializations.h>

#include <Context/ProgramManager.h>

#include <exp/context/DiskCache.h>

#include <Util/PersistentStream.h>
#include <Util/optixUuid.h>

#include <sstream>

using namespace optix;

std::string Specializations::summaryString( const ProgramManager* pm ) const
{
    std::ostringstream out;
    std::map<short, VariableReferenceID> printed;
    out << " {VS:";
    for( const auto& refid_vs : m_varspec )
    {
        VariableReferenceID           refid  = refid_vs.first;
        const VariableSpecialization& vs     = refid_vs.second;
        const VariableReference*      varref = pm->getVariableReferenceById( refid );
        if( !varref )
        {
            out << " " << refid << ":(null)";
            continue;
        }

        // Skip attributes
        if( varref->getInputName().compare( 0, 11, "_attribute_" ) == 0 )
            continue;

        unsigned short token = varref->getVariableToken();
        if( printed.count( token ) )
        {
            const VariableSpecialization& other = m_varspec.at( printed.at( token ) );
            if( other == vs || vs.lookupKind == VariableSpecialization::Unused )
                continue;
        }
        else
        {
            printed.insert( std::make_pair( token, refid ) );
        }
        out << ' ' << varref->getInputName();

        if( varref->getType().isBuffer() )
            out << ":buf";
        else if( varref->getType().isTextureSampler() )
            out << ":tex";

        switch( vs.lookupKind )
        {
            case VariableSpecialization::SingleScope:
                if( vs.singleBinding.isDefaultValue() )
                    out << "@default_value";
                else
                    out << "@" << vs.singleBinding.toString();
                break;
            case VariableSpecialization::SingleId:
                out << "@" << vs.singleId;
                break;
            case VariableSpecialization::GenericLookup:
                out << ":generic";
                break;
            case VariableSpecialization::Unused:
                out << ":unused";
        }


        switch( vs.accessKind )
        {
            case VariableSpecialization::HWTextureOnly:
                out << "/hwtex";
                break;
            case VariableSpecialization::SWTextureOnly:
                out << "/swtex";
                break;
            case VariableSpecialization::TexHeap:
                out << "/texheap";
                break;
            case VariableSpecialization::TexHeapSingleOffset:
                out << "/texheap+" << vs.singleOffset;
                break;
            case VariableSpecialization::PitchedLinearPreferLDG:
                out << "/ldg";
                break;
            case VariableSpecialization::PitchedLinear:
                out << "/ldst";
                break;
            case VariableSpecialization::GenericAccess:
                out << "/generic";
                break;
        }
    }
    out << "}";

    return out.str();
}

bool Specializations::isCompatibleWith( const Specializations& other ) const
{
    if( m_exceptionFlags != other.m_exceptionFlags )
        return false;

    if( m_printEnabled != other.m_printEnabled )
        return false;

    if( m_maxTransformDepth != other.m_maxTransformDepth )
        return false;

    if( m_varspec != other.m_varspec )
        return false;

    return true;
}

bool Specializations::operator<( const Specializations& other ) const
{
    if( m_exceptionFlags != other.m_exceptionFlags )
        return m_exceptionFlags < other.m_exceptionFlags;

    if( m_printEnabled != other.m_printEnabled )
        return m_printEnabled < other.m_printEnabled;

    if( m_maxTransformDepth != other.m_maxTransformDepth )
        return m_maxTransformDepth < other.m_maxTransformDepth;

    if( m_varspec.size() != other.m_varspec.size() )
        return m_varspec.size() < other.m_varspec.size();

    auto iter1 = m_varspec.begin();
    auto iter2 = other.m_varspec.begin();
    while( iter1 != m_varspec.end() && iter2 != other.m_varspec.end() )
    {
        if( iter1->first != iter2->first )
            return iter1->first < iter2->first;

        if( iter1->second != iter2->second )
            return iter1->second < iter2->second;

        ++iter1;
        ++iter2;
    }
    return false;
}

void Specializations::mergeVariableSpecializations( const Specializations& other )
{
    for( auto& spec : other.m_varspec )
    {
        if( m_varspec.count( spec.first ) == 1 )
        {
            RT_ASSERT( m_varspec.at( spec.first ) == spec.second );
        }
        m_varspec.insert( spec );
    }
}

void optix::readOrWrite( PersistentStream* stream, Specializations* spec, const char* label )
{
    auto                       tmp     = stream->pushObject( label, "Specializations" );
    static const unsigned int* version = getOptixUUID();
    stream->readOrWriteObjectVersion( version );
    readOrWrite( stream, &spec->m_exceptionFlags, "m_exceptionFlags" );
    readOrWrite( stream, &spec->m_printEnabled, "m_printEnabled" );
    readOrWrite( stream, &spec->m_minTransformDepth, "m_minTransformDepth" );
    readOrWrite( stream, &spec->m_maxTransformDepth, "m_maxTransformDepth" );
    readOrWrite( stream, &spec->m_varspec, "m_varspec" );
}
