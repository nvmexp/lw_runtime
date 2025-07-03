// Copyright LWPU Corporation 2008
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <corelib/misc/String.h>
#include <corelib/system/System.h>
#include <prodlib/misc/IomanipHelpers.h>
#include <prodlib/system/Knobs.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Functionality required to retrieve and handle environment variables.
#if defined( _WIN32 )
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
extern char** elwiron;
#endif

namespace {
// clang-format off
    Knob<bool> k_ignoreKnobChangesViaApi( RT_DSTRING( "o7.ignoreKnobChangesViaApi" ), false, RT_DSTRING( "Ignores knob changes with the extension API for knobs." ) );
// clang-format on
}  // namespace

std::string KnobBase::sourceToString( Source s )
{
    switch( s )
    {
        case Source::DEFAULT:
            return "DEFAULT";
        case Source::FILE:
            return "FILE";
        case Source::ENVIRONMENT:
            return "ENVIRONMENT";
        case Source::API:
            return "API";
        case Source::FILE_OR_ELWIRONMENT:
            return "FILE_OR_ELWIRONMENT";
        case Source::SCOPED_KNOB_SETTER:
            return "SCOPED_KNOB_SETTER";
        case Source::MIXED:
            return "MIXED";
    }
    return "UNKNOWN VALUE FOR SOURCE";
}

// Returns the list of environment variable strings.
static std::vector<std::string> getElwironmentStrings()
{
    std::vector<std::string> vars;
#if defined( _WIN32 )
    char* elws = GetElwironmentStrings();
    char* ptr  = elws;
    while( *ptr )
    {
        std::string val = ptr;
        vars.push_back( val );
        ptr += val.size() + 1;
    }
    FreeElwironmentStrings( elws );
#else
    int i = 0;
    while( char* s = elwiron[i++] )
        vars.push_back( s );
#endif
    return vars;
}

std::string KnobRegistry::getOptixPropsLocation()
{
    std::string result;

    if( corelib::getelw( "OPTIX_PROPS_PATH", result ) )
    {
        result += "/";
    }
    else
    {
        result = corelib::getLwrrentDir();
        if( !result.empty() )
            result += "/";
    }

    result += "optix.props";
    llog( 6 ) << "Looking for optix.props in " << result << "\n";
    return result;
}


void KnobRegistry::registerKnob( KnobBase* knob )
{
    std::lock_guard<std::mutex> lock( m_mutex );

    const std::string& name = knob->getName();

    if( m_finalized )
    {
        m_errorString += "Knob \"" + knob->getName() + "\" is registered after the knob registry has been finalized\n";
        return;
    }

    // Make sure that public knobs have a name (avoids mistakes where DT_STRING is used instead of DT_PUBLIC_STRING for
    // public or public hidden knobs).
    if( name.empty() && knob->getKind() != KnobBase::Kind::DEVELOPER )
    {
        m_errorString += "A public knob was registered with an empty name.\n";
        return;
    }

    // Ignore knobs with empty name, e.g., developer knobs in release builds.
    if( name.empty() )
        return;

    // Make sure that all instances of knobs of this name are equal.
    auto it = m_knobs.find( name );
    if( it != m_knobs.end() && !knob->isEqual( it->second ) )
    {
        m_errorString += "Knob \"" + knob->getName() + "\" is registered more than once, with differing attributes\n";
        return;
    }

    m_knobs.emplace( name, knob );
}

OptixResult KnobRegistry::initializeKnobs()
{
#if defined( OPTIX_ENABLE_KNOBS )
    std::lock_guard<std::mutex> lock( m_mutex );

    if( m_finalized )
        return OPTIX_ERROR_ILWALID_OPERATION;

    if( OptixResult result = setKnobsFromFileLocked( getOptixPropsLocation(), /*IOFailureIsError*/ false ) )
        return result;

    if( OptixResult result = setKnobsFromElwironmentLocked() )
        return result;

    return OPTIX_SUCCESS;
#else
    return OPTIX_ERROR_ILWALID_OPERATION;
#endif
}

static std::string trim( const std::string& s )
{
    auto start = std::find_if_not( s.begin(), s.end(), []( int c ) { return std::isspace( c ); } );
    auto end   = std::find_if_not( s.rbegin(), s.rend(), []( int c ) { return std::isspace( c ); } ).base();
    return end <= start ? std::string() : std::string( start, end );
}

OptixResult KnobRegistry::initializeKnobs( const std::string& knobsString )
{
    std::lock_guard<std::mutex> lock( m_mutex );

    if( m_finalized )
        return OPTIX_ERROR_ILWALID_OPERATION;

    OptixResult result = OPTIX_SUCCESS;

    std::istringstream in( knobsString );
    while( in )
    {
        std::string line = trim( corelib::getline_stripcomments( in ) );
        if( !line.empty() )
        {
            std::string name, value;
            if( corelib::get_key_value( line, name, value ) )
            {
                OptixResult r = setKnobLocked( name, value, KnobBase::Source::FILE_OR_ELWIRONMENT );
                result        = result ? result : r;
            }
        }
    }

    return result;
}

OptixResult KnobRegistry::setKnobsFromFile( const std::string& filename, bool IOFailureIsError )
{
    std::lock_guard<std::mutex> lock( m_mutex );

    if( k_ignoreKnobChangesViaApi.get() )
        return OPTIX_SUCCESS;

    if( m_finalized )
        return OPTIX_ERROR_ILWALID_OPERATION;

    return setKnobsFromFileLocked( filename, IOFailureIsError );
}

OptixResult KnobRegistry::setKnobsFromFileLocked( const std::string& filename, bool IOFailureIsError )
{
    OptixResult result = OPTIX_SUCCESS;

    std::ifstream in( filename.c_str() );
    if( IOFailureIsError && in.fail() )
    {
        m_errorString += "File I/O error for file \"" + filename + "\"\n";
        return OPTIX_ERROR_FILE_IO_ERROR;
    }

    while( in )
    {
        const std::string line = trim( corelib::getline_stripcomments( in ) );
        if( line.empty() )
            continue;

        std::string name;
        std::string value;
        corelib::get_key_value( line, name, value );
        OptixResult r = setKnobLocked( name, value, KnobBase::Source::FILE );
        result        = result ? result : r;
    }

    return result;
}

OptixResult KnobRegistry::setKnobsFromElwironmentLocked()
{
    OptixResult result = OPTIX_SUCCESS;

    for( const std::string& elw : getElwironmentStrings() )
    {
        const size_t firstEqualsPosition = elw.find_first_of( '=', 0 );
        if( firstEqualsPosition == std::string::npos )
            continue;

        std::string name  = elw.substr( 0, firstEqualsPosition );
        std::string value = elw.substr( firstEqualsPosition + 1, elw.size() );

        if( corelib::stringBeginsWith( name, "optix_" ) )
        {
            // Some shells, e.g., bash, do not allow environment variables with dots. As workaround, we support names
            // with dots replaced by underscores. This means that names using underscores can not be set via environment
            // variable with "optix_" prefix (but such names are lwrrently not in use).
            std::replace( name.begin(), name.end(), '_', '.' );
        }

        if( corelib::stringBeginsWith( name, "optix." ) )
        {
            OptixResult r = setKnobLocked( name.substr( 6 ), value, KnobBase::Source::ENVIRONMENT );
            result        = result ? result : r;
        }
    }

    return result;
}

OptixResult KnobRegistry::setKnob( const std::string& name, const std::string& value, KnobBase::Source source )
{
    std::lock_guard<std::mutex> lock( m_mutex );

    if( k_ignoreKnobChangesViaApi.get() )
        return OPTIX_SUCCESS;

    if( m_finalized )
        return OPTIX_ERROR_ILWALID_OPERATION;

    return setKnobLocked( name, value, source );
}

OptixResult KnobRegistry::setKnobLocked( const std::string& name, const std::string& value, KnobBase::Source source )
{
    if( isDynamicRtcoreKnob( name ) )
    {
        m_rtcoreKnobs[name] = std::make_pair( value, source );
        return OPTIX_SUCCESS;
    }

    auto range = m_knobs.equal_range( name );
    if( range.first == range.second )
    {
        m_errorString += "Knob \"" + name + "\" does not exist\n";
        return OPTIX_ERROR_ILWALID_VALUE;
    }

    for( auto it = range.first; it != range.second; ++it )
        if( OptixResult result = it->second->setUntyped( value, source ) )
        {
            m_errorString += "Invalid value for knob \"" + name + "\"\n";
            return result;
        }

    return OPTIX_SUCCESS;
}

void KnobRegistry::finalizeKnobs( std::string& errorString )
{
    std::lock_guard<std::mutex> lock( m_mutex );

    // Repeated calls are not considered as error.
    if( m_finalized )
    {
        errorString.clear();
        return;
    }

    transformDynamicRtcoreKnobs();
    exportKnobsToFile();

    errorString = m_errorString;
    m_errorString.clear();

    m_finalized = true;
}

void KnobRegistry::printKnobs( std::ostream& out ) const
{
    std::lock_guard<std::mutex> lock( m_mutex );
    printKnobsLocked( out );
}

void KnobRegistry::printKnobsLocked( std::ostream& out ) const
{
    prodlib::IOSSaver saver( out );
    out << "\n";

    std::string lastKnobKey;
    for( auto nameKnob : m_knobs )
    {

#if !defined( OPTIX_ENABLE_KNOBS )
        // skip develop or public hidden knobs in release builds
        if( nameKnob.second->getKind() != KnobBase::Kind::PUBLIC )
            continue;
#endif

        // skip multiple instances of the same name
        if( nameKnob.first == lastKnobKey )
            continue;

        // comment out the default knobs so this makes for a nice fresh config file
        if( nameKnob.second->isDefault() )
            out << "// ";

        nameKnob.second->print( out, true, true );
        out << std::endl;

        lastKnobKey = nameKnob.first;
    }
}

void KnobRegistry::printNonDefaultKnobs( std::ostream& out ) const
{
    std::lock_guard<std::mutex> lock( m_mutex );

    prodlib::IOSSaver saver( out );

    bool hasNonDefault = false;
    bool hasSetKnobs   = false;
    for( auto nameKnob : m_knobs )
    {
        if( !nameKnob.second->isDefault() )
        {
            hasNonDefault = true;
            break;
        }
        if( nameKnob.second->isSet() )
        {
            hasSetKnobs = true;
            break;
        }
    }

    if( !hasNonDefault && !hasSetKnobs )
    {
        out << "All knobs on default.\n";
        return;
    }

    out << "Non-default knobs:\n";
    std::string lastKnobKey;
    for( auto nameKnob : m_knobs )
    {
        const KnobBase* knob = nameKnob.second;

        if( knob->isDefault() )
            continue;

        // skip multiple instances of the same name
        if( nameKnob.first == lastKnobKey )
            continue;

        out << "  ";
        knob->print( out, false, false, true );
        out << "\n";

        lastKnobKey = nameKnob.first;
    }

    out << "Set but default knobs:\n";
    for( auto nameKnob : m_knobs )
    {
        const KnobBase* knob = nameKnob.second;

        if( !( knob->isSet() && knob->isDefault() ) )
            continue;

        // skip multiple instances of the same name
        if( nameKnob.first == lastKnobKey )
            continue;

        out << "  ";
        knob->print( out, false, false, true );
        out << "\n";

        lastKnobKey = nameKnob.first;
    }
}

std::vector<std::string> KnobRegistry::getNonDefaultKnobs() const
{
    std::lock_guard<std::mutex> lock( m_mutex );

    std::vector<std::string> strings;
    std::string              lastKnobKey;
    for( const auto& nameKnob : m_knobs )
    {
        const KnobBase* knob = nameKnob.second;

        if( knob->isDefault() )
            continue;

        // skip multiple instances of the same name
        if( nameKnob.first == lastKnobKey )
            continue;

        std::ostringstream out;
        knob->print( out, false );
        strings.push_back( out.str() );

        lastKnobKey = nameKnob.first;
    }

    return strings;
}

bool KnobRegistry::isDynamicRtcoreKnob( const std::string& name )
{
    return corelib::stringBeginsWith( name, "rtcore." ) && ( name != "rtcore.knobs" );
}

void KnobRegistry::transformDynamicRtcoreKnobs()
{
    // Check whether the "rtcore.knobs" knob exists, e.g., it does not exist in release builds.
    auto range = m_knobs.equal_range( "rtcore.knobs" );
    if( range.first == range.second )
    {
        if( !m_rtcoreKnobs.empty() )
            m_errorString += "Dynamic rtcore knobs used, but there is no \"rtcore.knobs\" knob\n";
        return;
    }

    // Append combined name/value pairs to "rtcore.knobs" value
    Knob<std::string>* knob              = dynamic_cast<Knob<std::string>*>( range.first->second );
    std::string        combinedKeyValues = knob->get();
    KnobBase::Source   source            = knob->getSource();

    // Add all name/value pairs in m_rtcoreKnobs into one string (stripping the "rtcore." prefix from the names).
    for( auto nameValue : m_rtcoreKnobs )
    {
        const std::string& name  = nameValue.first.substr( 7 );
        const std::string& value = nameValue.second.first;
        if( !combinedKeyValues.empty() )
            combinedKeyValues += ",";
        combinedKeyValues += name + ":\"" + value + "\"";
        if( source == KnobBase::Source::DEFAULT )
            source = nameValue.second.second;
        else if( source != nameValue.second.second )
            source = KnobBase::Source::MIXED;
    }
    m_rtcoreKnobs.clear();

    setKnobLocked( "rtcore.knobs", combinedKeyValues, source );
}

void KnobRegistry::exportKnobsToFile() const
{
#if defined( OPTIX_ENABLE_KNOBS )
    const std::string& filename = getOptixPropsLocation();

    std::ifstream in( filename );
    if( in )
    {
        in.seekg( 0, std::ios::end );
        const size_t size = in.tellg();
        if( size != 0 )
            return;
    }
    else
    {
        if( std::getelw( "OPTIX_PROPS" ) == nullptr )
            return;
    }
    in.close();

    std::ofstream out( filename );
    printKnobsLocked( out );
#endif
}

#include <prodlib/system/KnobsImpl.h>

#define INSTANTIATE_FOR_TYPE( type )                                                                                   \
    template OptixResult KnobRegistry::setKnobTyped( const std::string& name, const type& value, type& oldValue,       \
                                                     KnobBase::Source source, KnobBase::Source& oldSource );

INSTANTIATE_FOR_TYPE( bool );
INSTANTIATE_FOR_TYPE( float );
INSTANTIATE_FOR_TYPE( int );
INSTANTIATE_FOR_TYPE( unsigned int );
INSTANTIATE_FOR_TYPE( unsigned long );
INSTANTIATE_FOR_TYPE( unsigned long long );
INSTANTIATE_FOR_TYPE( std::string );

#undef INSTANTIATE_FOR_TYPE


KnobRegistry& knobRegistry()
{
    static KnobRegistry g_knobRegistry;
    return g_knobRegistry;
}
