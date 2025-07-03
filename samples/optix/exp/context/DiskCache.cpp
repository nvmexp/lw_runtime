// Copyright LWPU Corporation 2017
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <exp/context/DeviceContext.h>
#include <exp/context/DiskCache.h>
#include <exp/context/ErrorHandling.h>

#include <Util/PersistentStream.h>
#include <Util/digest_md5.h>

#include <corelib/misc/String.h>
#include <corelib/system/System.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/system/Knobs.h>
#include <prodlib/system/System.h>

#include <algorithm>
#include <fstream>
#include <iomanip>

#ifdef _WIN32
#include <ShlObj.h>  // To retrieve the LocalAppData folder on Windows
#endif
#ifdef __linux__
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>
#endif

using namespace prodlib;

namespace {
// clang-format off
PublicKnob<bool>  k_cacheEnabled( RT_PUBLIC_DSTRING( "diskcache.enabled" ), true, RT_PUBLIC_DSTRING( "Enable / disable the on-disk compilation cache" ) );
Knob<std::string> k_overrideCacheDirectory( RT_DSTRING( "diskcache.overrideDirectory"), "<DEFAULT>", RT_DSTRING( "Force the cache database file to be created at the specified location." ) );
Knob<bool>        k_humanReadable( RT_DSTRING( "diskcache.exportHumanReadableFiles" ), true, RT_DSTRING( "Create human-readable cache entries side-by-side with the normal binary versions. The directory specified with diskcache.humanReadableFilesDirectory must exist." ) );
Knob<std::string> k_cacheDirectory( RT_DSTRING( "diskcache.humanReadableFilesDirectory" ), "optixcache", RT_DSTRING( "Directory name to store human readable text files for each cache entry for debugging and optimizations of the caching mechanism." ));
Knob<size_t>      k_lowWaterMark( RT_DSTRING( "diskcache.lowWaterMark" ), 1u << 30, RT_DSTRING( "The cache size will be reduced to the low-water mark if the high-water mark is exceeded. The value 0 disables garbage collection. The default value is 1GB." ));
Knob<size_t>      k_highWaterMark( RT_DSTRING( "diskcache.highWaterMark" ), 1u << 31, RT_DSTRING( "The cache size will be reduced to the low-water mark if the high-water mark is exceeded. The value 0 disables garbage collection. The default value is 2GB." ));
// clang-format on
}  // namespace

namespace {
struct PersistentWriter : public optix::PersistentStream
{
    PersistentWriter( const std::string& key, optix_exp::DiskCacheDatabase* database );

    void flush( optix_exp::DeviceContextLogger& clog ) override;

    void readOrWriteObjectVersion( const unsigned int* version ) override;
    void readOrWrite( char* data, size_t size, const char* label, Format format ) override;

    std::string                   m_key;
    std::stringbuf                m_buffer;
    optix_exp::DiskCacheDatabase* m_database;
};

struct PersistentWriterWithReadableCopy : public PersistentWriter
{
    static std::string generateFileName( const std::string& key );
    PersistentWriterWithReadableCopy( const std::string& key, optix_exp::DiskCacheDatabase* database );

    void readOrWriteObjectVersion( const unsigned int* version ) override;
    void readOrWrite( char* data, size_t size, const char* label, Format format ) override;
    void writeHexString( char* data, size_t size );

    void pushLabel( const char* label, const char* classname ) override;
    void popLabel() override;

    std::ofstream out_human;
    int           m_indent = 0;
};

struct PersistentReader : public optix::PersistentStream
{
    PersistentReader( std::unique_ptr<optix_exp::DiskCacheDatabase::ByteBuffer> buffer );

    void readOrWriteObjectVersion( const unsigned int* expectedVersion ) override;
    void readOrWrite( char* data, size_t size, const char* label, Format format ) override;

    std::unique_ptr<const optix_exp::DiskCacheDatabase::ByteBuffer> m_buffer;
    size_t                                                          m_lwrsor;
};

struct MD5Hasher : public optix::PersistentStream
{
    MD5Hasher();

    std::string getDigestString() const override;
    void readOrWriteObjectVersion( const unsigned int* version ) override;
    void readOrWrite( char* data, size_t size, const char* label, Format format ) override;

    MI::DIGEST::MD5 hasher;
};
}  // namespace


#ifdef __linux__
std::string getUserName()
{
    // Notes about alternatives to getpwuid_r():
    // - getpwuid() is not thread-safe
    // - lwserid() lwts off usernames after the 8th character
    // - getlogin() is not thread-safe
    // - getlogin_r() returns the username of the controlling terminal which might be different from the effective
    //   user, or there might be no controlling terminal
    // - environment variables USER and LOGNAME could be changed by the user

    size_t            bufferSize = 32768;  // The buffer also contains the pw_gecos field which might need some space.
    std::vector<char> buffer( bufferSize );
    struct passwd     pwd;
    struct passwd*    result;
    int               error = getpwuid_r( geteuid(), &pwd, &buffer[0], bufferSize, &result );
    return ( error == 0 && result == &pwd ) ? std::string( result->pw_name ) : std::string();
}
#endif

/// Determine file path for the diskcache file for all 3 platforms.
/// * First check if an override location is being set by a knob.
/// * Then check OPTIX_CACHE_PATH environment variable.
/// * Finally, check if the path argument is non-default, which means that the
///   application is attempting to set a new location via the Context Attribute API.
///   An exception will be thrown by the Context if setting the path though the API
///   fails for any reason.
/// * If it is not present, use the following default directories:
///     Linux:      /var/tmp/OptixCache_$USER/cache.db (1st choice, or without "_$USER")
///                 /tmp/OptixCache_$USER/cache.db     (2nd choice, or without "_$USER")
///     Windows:    %LOCALAPPDATA%\\LWPU\\OptixCache\\cache.db
///     Mac:        /Library/Application Support/LWPU/OptixCache/cache.db
///   For Linux and Windows fall back to the current working directory
///   if the default locations are not visible in the environment and writable.
static OptixResult getDatabaseFilePath( const std::string&              path,
                                        const std::string&              fileName,
                                        std::string&                    filePath,
                                        optix_exp::DeviceContextLogger& clog,
                                        optix_exp::ErrorDetails&        errDetails )
{
    // Determine the target cache location, depending on whether the knob or elw
    // var are set. Use the platform-specific default path if no override is set.
    std::string elwPath;
    bool        elwSet     = corelib::getelw( "OPTIX_CACHE_PATH", elwPath );
    std::string knobPath   = k_overrideCacheDirectory.get();
    bool        knobSet    = !k_overrideCacheDirectory.isDefault();
    bool        nonDefault = ( knobSet || elwSet || path != "<USE DEFAULT>" );

    std::string targetPath = ( knobSet ) ? knobPath : ( elwSet ) ? elwPath : path;
    std::replace( targetPath.begin(), targetPath.end(), '\\', '/' );

    std::string optixCacheDir;
    if( nonDefault )
    {
        optixCacheDir = targetPath;
    }
    else
    {
        std::string lwDir;
#ifdef __linux__
        std::string userName = getUserName();
        if( userName.empty() )
        {
            clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                           "Failed to obtain username for current process, using cache directory without username as "
                           "suffix." );
        }
        else
        {
            userName = "_" + userName;
        }

        lwDir         = "/var/tmp";
        optixCacheDir = lwDir + "/OptixCache" + userName;

        // Eagerly try to create the OptixCache dir to determine whether or not
        // we need to try /tmp.
        if( !prodlib::dirExists( optixCacheDir.c_str() ) && !prodlib::createDir( optixCacheDir.c_str() ) )
        {
            lwDir         = "/tmp";
            optixCacheDir = lwDir + "/OptixCache" + userName;
        }
#elif _WIN32
        // Try to use LOCALAPPDATA, unlike the LWCA cache which is in APPDATA (roaming data).
        // We do not want a potentially large diskcache file to be part of roaming data.
        if( !corelib::getelw( "LOCALAPPDATA", lwDir ) )
        {
            // Variable is not present, use GetKnownFolderPath instead
            wchar_t* appDataWStr = nullptr;
            if( SHGetKnownFolderPath( FOLDERID_LocalAppData, KF_FLAG_DEFAULT, nullptr, &appDataWStr ) != S_OK )
            {
                clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                               corelib::stringf( "Could not retrieve LocalAppData folder path. "
                                                 "Storing %s in the current working directory.",
                                                 fileName.c_str() )
                                   .c_str() );

                filePath = fileName;
                return OPTIX_SUCCESS;
            }
            // Colwert the wide-char string to a regular char-string (assuming that the path falls within ASCII range)
            const size_t strLength = wcslen( appDataWStr );
            lwDir.resize( strLength );
            wcstombs( &lwDir[0], appDataWStr, strLength );
            CoTaskMemFree( appDataWStr );
        }
        lwDir += "/LWPU";
        optixCacheDir = lwDir + "/OptixCache";
#else
        lwDir         = "/Library/Application Support/LWPU";
        optixCacheDir = lwDir + "/OptixCache";
#endif
        // Create the lwpu folder if it doesn't exist yet
        if( !prodlib::dirExists( lwDir.c_str() ) )
        {
            if( !prodlib::createDir( lwDir.c_str() ) )
            {
                clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                               corelib::stringf( "Could not create lwpu directory: \"%s\" "
                                                 "Storing %s in the current working directory.",
                                                 lwDir.c_str(), fileName.c_str() )
                                   .c_str() );
                filePath = fileName;
                return OPTIX_SUCCESS;
            }
        }
    }

    // Create the OptixCache folder if it doesn't exist yet
    if( !prodlib::dirExists( optixCacheDir.c_str() ) && ( !prodlib::createDirectories( optixCacheDir.c_str() ) ) )
    {
        if( nonDefault )
        {
            // return filePath, so the user can query it in the error case
            // to find out which path failed.
            filePath = optixCacheDir + "/" + fileName;

            if( knobSet )
                return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_ILWALID_PATH,
                                              "Invalid DiskCache path override knob: " + knobPath );
            else if( elwSet )
                return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_ILWALID_PATH, "Invalid OPTIX_CACHE_PATH value: " + elwPath );
            else
                return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_ILWALID_PATH, "Unable to set OptiX cache path: " + path );
        }
        else
        {
            clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                           corelib::stringf( "Could not create directory: \"%s\" "
                                             "Storing %s in the current working directory.",
                                             optixCacheDir.c_str(), fileName.c_str() )
                               .c_str() );
            filePath = fileName;
            return OPTIX_SUCCESS;
        }
    }
    filePath = optixCacheDir + "/" + fileName;
    return OPTIX_SUCCESS;
}

namespace optix_exp {
OptixResult DiskCache::init( DeviceContextLogger& clog, ErrorDetails& errDetails, const std::string& fileName, const std::string& path )
{
    return init( clog, errDetails, fileName, path, k_lowWaterMark.get(), k_highWaterMark.get() );
}

OptixResult DiskCache::init( DeviceContextLogger& clog,
                             ErrorDetails&        errDetails,
                             const std::string&   fileName,
                             const std::string&   path,
                             size_t               lowWaterMark,
                             size_t               highWaterMark )
{
    std::string elwSizeStr;
    if( !k_highWaterMark.isSet() && corelib::getelw( "OPTIX_CACHE_MAXSIZE", elwSizeStr ) )
    {
        long long int elwSize      = 0;
        bool          elwSizeValid = ( sscanf( elwSizeStr.c_str(), "%lld", &elwSize ) == 1 ) && elwSize >= 0;
        if( !elwSizeValid )
        {
            clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DISK CACHE",
                           corelib::stringf( "Invalid OPTIX_CACHE_MAXSIZE: \"%s\". "
                                             "Value will be ignored.",
                                             elwSizeStr.c_str() )
                               .c_str() );
        }
        else if( elwSize == 0 )
        {
            m_active        = false;
            m_disabledByElw = true;
            m_sizeSetByElw  = true;
            clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Print, "DISK CACHE",
                           "OPTIX_CACHE_MAXSIZE is set to 0. "
                           "Disabling the OptiX disk cache. The cache contents will not be changed." );
            return OPTIX_SUCCESS;
        }
        else
        {
            highWaterMark  = static_cast<size_t>( elwSize );
            lowWaterMark   = static_cast<size_t>( elwSize ) / 2;
            m_sizeSetByElw = true;
            clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Print, "DISK CACHE",
                           corelib::stringf( "OPTIX_CACHE_MAXSIZE is set to %lld. "
                                             "Setting high water mark to %zu bytes and low water mark to %zu bytes.",
                                             elwSize, highWaterMark, lowWaterMark )
                               .c_str() );
        }
    }

    if( OptixResult result = getDatabaseFilePath( path, fileName, m_lwrrentDatabasePath, clog, errDetails ) )
        return result;
    if( OptixResult result = m_database.init( m_lwrrentDatabasePath, lowWaterMark, highWaterMark, clog, errDetails ) )
    {
        return errDetails.logDetails( result, "Unable to initialize the OptiX cache database at path " + m_lwrrentDatabasePath
                                                  + ". Delete or rename the file if this problem persists." );
    }

    m_active = k_cacheEnabled.get() && m_database.isOpen();

    m_exportReadableFiles = m_active && k_humanReadable.get() && prodlib::dirExists( k_cacheDirectory.get().c_str() );
    if( k_cacheEnabled.get() && !m_database.isOpen() )
        return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR,
                                      "Unable to open OptiX cache database: " + m_lwrrentDatabasePath );
    return OPTIX_SUCCESS;
}

OptixResult DiskCache::destroy( DeviceContextLogger& clog )
{
    return m_database.destroy( clog );
}

OptixResult DiskCache::setPath( const std::string& path, const std::string& fileName, DeviceContextLogger& clog, ErrorDetails& errDetails )
{
    // Don't set the path if the disk cache has been disabled via the environment variable.
    if( m_disabledByElw )
    {
        m_lwrrentDatabasePath = "";
        return OPTIX_SUCCESS;
    }

    // Do not destroy database right away. Destroy and init is done atomically using
    // DiskCacheDatabase::setPath to ensure thread safety.
    std::string oldPath = m_lwrrentDatabasePath;
    OptixResult result  = getDatabaseFilePath( path, fileName, m_lwrrentDatabasePath, clog, errDetails );
    if( result )
    {
        // Given path is bad. Deactivate cache.
        m_active = false;
        // Close previous database connection
        m_database.destroy( clog );
        return result;
    }
    if( oldPath == m_lwrrentDatabasePath )
    {
        // New cache location is the old cache location.
        // TODO: Is there a valid use case to reinitialize anyways?
        return OPTIX_SUCCESS;
    }

    if( OptixResult result = m_database.setPath( m_lwrrentDatabasePath, clog, errDetails ) )
    {
        m_active = false;
        return result;
    }

    m_active              = m_active && k_cacheEnabled.get() && m_database.isOpen();
    m_exportReadableFiles = m_exportReadableFiles && m_active;

    if( k_cacheEnabled.get() && !m_database.isOpen() )
        return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR,
                                      "Unable to open OptiX cache database: " + m_lwrrentDatabasePath );
    return OPTIX_SUCCESS;
}

bool DiskCache::isActive() const
{
    return m_active;
}

void DiskCache::setIsActive( bool active )
{
    m_active = active;
}

bool DiskCache::isDisabledByElwironment() const
{
    return m_disabledByElw;
}

bool DiskCache::isSizeSetByElwironment() const
{
    return m_sizeSetByElw;
}

std::string DiskCache::getPath( const std::string& fileName ) const
{
    // Strip the filename from the path and return the containing directory.
    size_t found = m_lwrrentDatabasePath.find( fileName );
    if( found != std::string::npos )
        return m_lwrrentDatabasePath.substr( 0, std::max( size_t( 0 ), found - 1 ) );
    else
        return m_lwrrentDatabasePath;
};

OptixResult DiskCache::find( const std::string&                        key,
                             std::unique_ptr<optix::PersistentStream>& stream,
                             DeviceContextLogger&                      clog,
                             ErrorDetails&                             errDetails ) const
{
    stream.reset();
    if( !m_active )
        return OPTIX_SUCCESS;

    std::unique_ptr<DiskCacheDatabase::ByteBuffer> buffer = m_database.findCacheEntry( key.c_str(), clog );
    if( buffer )
    {
        stream.reset( new PersistentReader( std::move( buffer ) ) );
        if( stream->error() )
        {
            return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_ILWALID_DATA, "Read error: Empty cache data for key " + key );
        }
        return OPTIX_SUCCESS;
    }

    return OPTIX_SUCCESS;
}

OptixResult DiskCache::insert( const std::string& key, std::unique_ptr<optix::PersistentStream>& stream, ErrorDetails& errDetails )
{
    stream.reset();
    if( !m_active )
        return OPTIX_SUCCESS;

    if( m_exportReadableFiles )
    {
        stream.reset( new PersistentWriterWithReadableCopy( key, &m_database ) );
        if( stream->error() )
        {
            stream.reset();
            return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR,
                                          "Could not open human readable file: "
                                              + PersistentWriterWithReadableCopy::generateFileName( key ) );
        }
    }
    else
    {
        stream.reset( new PersistentWriter( key, &m_database ) );
        if( stream->error() )
        {
            // lwrrently not possible
            stream.reset();
            return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR, "Could not create database writer" );
        }
    }

    return OPTIX_SUCCESS;
}

std::unique_ptr<optix::PersistentStream> DiskCache::createHasher() const
{
    return std::unique_ptr<optix::PersistentStream>( new MD5Hasher() );
}

OptixResult DiskCache::setMemoryLimits( size_t lowWaterMark, size_t highWaterMark, DeviceContextLogger& clog, ErrorDetails& errDetails )
{
    if( m_sizeSetByElw )
    {
        clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Print, "DISK CACHE",
                       "The disk cache memory limits have been set by the OPTIX_CACHE_MAXSIZE environment variable. "
                       "The API call will be ignored." );
        return OPTIX_SUCCESS;
    }
    else
    {
        return m_database.setMemoryLimits( lowWaterMark, highWaterMark, clog, errDetails );
    }
}

void DiskCache::getMemoryLimits( size_t& lowWaterMark, size_t& highWaterMark ) const
{
    if( m_disabledByElw )
    {
        lowWaterMark  = 0;
        highWaterMark = 0;
    }
    else
    {
        m_database.getMemoryLimits( lowWaterMark, highWaterMark );
    }
}
}  // namespace optix_exp

PersistentReader::PersistentReader( std::unique_ptr<optix_exp::DiskCacheDatabase::ByteBuffer> buffer )
    : PersistentStream( Reading )
    , m_buffer( std::move( buffer ) )
    , m_lwrsor( 0 )
{
    m_error = m_buffer->empty();
}

void PersistentReader::readOrWriteObjectVersion( const unsigned int* expectedVersion )
{
    unsigned int version[4] = {0, 0, 0, 0};
    optix::readOrWrite( this, &version[0], "version[0]" );
    optix::readOrWrite( this, &version[1], "version[1]" );
    optix::readOrWrite( this, &version[2], "version[2]" );
    optix::readOrWrite( this, &version[3], "version[3]" );
    if( version[0] != expectedVersion[0] || version[1] != expectedVersion[1] || version[2] != expectedVersion[2]
        || version[3] != expectedVersion[3] )
    {
        m_error = true;
    }
}

void PersistentReader::readOrWrite( char* data, size_t size, const char* label, Format format )
{
    if( m_error )
        return;
    if( m_lwrsor + size > m_buffer->size() )
    {
        m_error = true;
        return;
    }
    memcpy( data, &( *m_buffer )[m_lwrsor], size );
    m_lwrsor += size;
}

std::string PersistentWriterWithReadableCopy::generateFileName( const std::string& key )
{
    return k_cacheDirectory.get() + "/" + key + ".optixcache.txt";
}

PersistentWriterWithReadableCopy::PersistentWriterWithReadableCopy( const std::string& key, optix_exp::DiskCacheDatabase* database )
    : PersistentWriter( key, database )
{
    const std::string filename_human = generateFileName( key );
    out_human.open( filename_human.c_str() );
    m_error |= out_human.fail();
}

void PersistentWriterWithReadableCopy::readOrWriteObjectVersion( const unsigned int* version )
{
    PersistentWriter::readOrWriteObjectVersion( version );
}

namespace {
template <class T>
void simpleOutput( std::ostream& out, char* data, size_t size )
{
    T* ptr = (T*)data;
    for( size_t i = 0; i < size / sizeof( T ); ++i )
    {
        if( i != 0 )
            out << ' ';
        out << *ptr++;
    }
}
}  // namespace

void PersistentWriterWithReadableCopy::writeHexString( char* data, size_t size )
{
    static const size_t maxSize = 80;
    if( size <= maxSize )
    {
        for( size_t i = 0; i < size; ++i )
            out_human << std::setfill( '0' ) << std::setw( 2 ) << std::hex << (unsigned int)(unsigned char)data[i] << ' ';
    }
    else
    {
        // print head and tail only
        for( size_t i = 0; i < 20; ++i )
            out_human << std::setfill( '0' ) << std::setw( 2 ) << std::hex << (unsigned int)(unsigned char)data[i] << ' ';

        out_human << " ... ";
        for( size_t i = size - 3; i < size; ++i )
            out_human << std::setfill( '0' ) << std::setw( 2 ) << std::hex << (unsigned int)(unsigned char)data[i] << ' ';
    }
}

void PersistentWriterWithReadableCopy::readOrWrite( char* data, size_t size, const char* label, Format format )
{
    PersistentWriter::readOrWrite( data, size, label, format );
    if( m_error )
        return;
    if( !label )
        return;
    for( int i = 0; i < m_indent; i++ )
        out_human << "  ";
    out_human << label << " = ";

    std::ios::fmtflags saveflags( out_human.flags() );
    switch( format )
    {
        case Opaque:
            writeHexString( data, size );
            break;
        case None:
            break;
        case String:
            out_human << '"' << data << '"';
            break;
        case Bool:
            for( size_t i = 0; i < size; ++i )
                out_human << ( data[i] ? "true" : "false" ) << ' ';
            break;
        case Char:
            writeHexString( data, size );
            break;
        case Int:
            simpleOutput<int>( out_human, data, size );
            break;
        case UInt:
            simpleOutput<unsigned int>( out_human, data, size );
            break;
        case Short:
            simpleOutput<short>( out_human, data, size );
            break;
        case UShort:
            simpleOutput<unsigned short>( out_human, data, size );
            break;
        case LongLong:
            simpleOutput<long long>( out_human, data, size );
            break;
        case ULong:
            simpleOutput<unsigned long>( out_human, data, size );
            break;
        case ULongLong:
            simpleOutput<unsigned long long>( out_human, data, size );
            break;
    }
    out_human.flags( saveflags );
    out_human << '\n';
    if( out_human.fail() )
        m_error = true;
}

void PersistentWriterWithReadableCopy::pushLabel( const char* label, const char* classname )
{
    for( int i = 0; i < m_indent; i++ )
        out_human << "  ";
    out_human << label << ":" << classname << " = {\n";
    m_indent++;
}

void PersistentWriterWithReadableCopy::popLabel()
{
    --m_indent;
    for( int i = 0; i < m_indent; i++ )
        out_human << "  ";
    out_human << "}\n";
}

PersistentWriter::PersistentWriter( const std::string& key, optix_exp::DiskCacheDatabase* database )
    : PersistentStream( Writing )
    , m_key( key )
    , m_buffer()
    , m_database( database )
{
}

void PersistentWriter::flush( optix_exp::DeviceContextLogger& clog )
{
    const std::string buffer_data( m_buffer.str() );
    m_database->insertCacheEntry( m_key.c_str(), buffer_data.c_str(), buffer_data.size(), clog );
}

void PersistentWriter::readOrWriteObjectVersion( const unsigned int* version )
{
    optix::readOrWrite( this, &version[0], "version[0]" );
    optix::readOrWrite( this, &version[1], "version[1]" );
    optix::readOrWrite( this, &version[2], "version[2]" );
    optix::readOrWrite( this, &version[3], "version[3]" );
}

void PersistentWriter::readOrWrite( char* data, size_t size, const char* label, Format format )
{
    if( m_error )
        return;
    if( m_buffer.sputn( data, static_cast<std::streamsize>( size ) ) != static_cast<std::streamsize>( size ) )
        m_error = true;
}


MD5Hasher::MD5Hasher()
    : PersistentStream( Hashing )
{
}

void MD5Hasher::readOrWriteObjectVersion( const unsigned int* version )
{
    hasher.update( (char*)&version[0], sizeof( version[0] ) );
    hasher.update( (char*)&version[1], sizeof( version[1] ) );
    hasher.update( (char*)&version[2], sizeof( version[2] ) );
    hasher.update( (char*)&version[3], sizeof( version[3] ) );
}

void MD5Hasher::readOrWrite( char* data, size_t size, const char* label, Format format )
{
    hasher.update( data, size );
}

std::string MD5Hasher::getDigestString() const
{
    // Make a copy of the hasher and finalize it. This allows the
    // string to be printed in intermediate results.
    MI::DIGEST::MD5 copy = hasher;
    copy.finalize();
    char buf[33];
    buf[32] = '\0';
    copy.hex_digest( buf );
    return std::string( buf );
}
