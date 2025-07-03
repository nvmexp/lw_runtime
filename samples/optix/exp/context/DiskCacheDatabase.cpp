// Copyright LWPU Corporation 2018
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <Exceptions/DatabaseFilePermissions.h>
#include <Util/LWML.h>
#include <exp/context/DiskCache.h>
#include <exp/context/DiskCacheDatabase.h>
#include <private/optix_version_string.h>

#include <corelib/misc/String.h>
#include <corelib/system/System.h>

#include <exp/context/DeviceContext.h>
#include <exp/context/ErrorHandling.h>

#include <prodlib/system/System.h>

#include <support/sqlite/sqlite3.h>

#include <chrono>
#include <cstdio>  // for std::remove
#include <cstring>

#define OPTIX_ERROR_FAIL( call )                                                                                       \
    if( OptixResult result = call )                                                                                    \
        return result;

std::string bytesToString( size_t size )
{
    char buffer[256];
    if( size >= 1073741824.0 )
        snprintf( buffer, sizeof( buffer ) - 1, "%.1f GiB", size / 1073741824.0 );
    else if( size >= 1048576.0 )
        snprintf( buffer, sizeof( buffer ) - 1, "%.1f MiB", size / 1048576.0 );
    else if( size >= 1024 )
        snprintf( buffer, sizeof( buffer ) - 1, "%.1f KiB", size / 1024.0 );
    else
        snprintf( buffer, sizeof( buffer ) - 1, "%d Bytes", static_cast<int>( size ) );
    return buffer;
}

namespace optix_exp {

DiskCacheDatabase::DiskCacheDatabase()
    : m_driverVersion( optix::LWML::driverVersion() )
{
}

OptixResult DiskCacheDatabase::init( const std::string&   filename,
                                     size_t               lowWaterMark,
                                     size_t               highWaterMark,
                                     DeviceContextLogger& clog,
                                     ErrorDetails&        errDetails )
{
    return init( true, filename, lowWaterMark, highWaterMark, clog, errDetails );
}

OptixResult DiskCacheDatabase::init( bool                 retry,
                                     const std::string&   filename,
                                     size_t               lowWaterMark,
                                     size_t               highWaterMark,
                                     DeviceContextLogger& clog,
                                     ErrorDetails&        errDetails )
{
    if( m_sqlite )
    {
        //already initialized
        return OPTIX_SUCCESS;
    }
    if( sqlite3_open( filename.c_str(), &m_sqlite ) != SQLITE_OK )
    {
        // SQLite returns in most cases a valid handle, even if an error oclwrred.
        // Close the handle again and reset the pointer to NULL;
        if( m_sqlite != nullptr )
        {
            sqlite3_close( m_sqlite );
            m_sqlite = nullptr;
        }
        return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR,
                                      std::string( "Could not open database file: " ) + filename );
    }

    // Set the max. time sqlite is allowed to wait for a lock in order to be able to write to disk.
    if( sqlite3_busy_timeout( m_sqlite, static_cast<int>( m_timeoutInSeconds * 1000 ) ) != SQLITE_OK )
    {
        // Not sure what could go wrong - the sqlite documentation does not mention the result codes
        clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                       corelib::stringf( "Could not set sqlite3 busy timeout: %s", sqlite3_errmsg( m_sqlite ) ).c_str() );
    }
    clog.callback( 13, "DiskCacheDatabase", corelib::stringf( "Opened database: \"%s\"", filename.c_str() ).c_str() );

    // Make sure the database file is writable by the current process.
    if( !prodlib::fileIsWritable( filename.c_str() ) )
    {
        sqlite3_close( m_sqlite );
        m_sqlite = nullptr;
        return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_PERMISSION_ERROR,
                                      std::string( "Database file is not writable: " ) + filename );
    }

    setMemoryLimitsNoLock( lowWaterMark, highWaterMark, clog, errDetails );

    // If configure fails because the database is busy, keep trying until the timeout is reached,
    // or until an actual error oclwrs.
    DiskCacheDatabase::QueryResult result = QueryResult::CACHE_SUCCESS;
    {      
        using namespace std::chrono;
        auto waitStart = steady_clock::now();
        for( ;; )
        {
            result = configure( clog );
            if( result != DiskCacheDatabase::QueryResult::CACHE_BUSY )
                break;
            if( duration_cast<duration<double>>( steady_clock::now() - waitStart ).count() > m_timeoutInSeconds )
                break;
        }
    }

    if( result != QueryResult::CACHE_SUCCESS )
    {
        if( m_sqlite )
            sqlite3_close( m_sqlite );
        m_sqlite = nullptr;

        if( result == QueryResult::CACHE_CORRUPT )
        {
            // In the case of a corrupt cache, we need to delete the on-disk files and attempt to
            // re-initialize the database. If we're unable to delete the files (e.g., there are
            // other open database connections), then we need to disable caching and notify the
            // user.
            
            clog.callback( DeviceContextLogger::Print, "DiskCacheDatabase",
                           "Corrupt cache file detected. Deleting cache files." );

            if( !deleteDbFiles( filename, clog ) )
            {
                return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR,
                                              std::string( "Failed to remove corrupt database file \"" ) + filename
                                              + "\"." );
            }
            if( retry )
            {
                return init( false, filename, lowWaterMark, highWaterMark, clog, errDetails );
            }
            else
            {
                return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR,
                                              "Failed to recover corrupt database." );
            }
        }
        if( result == QueryResult::CACHE_BUSY )
        {
            // It's possible that configuration will fail if there is contention with other threads
            // or processes that keeps the cache in a busy state. If that oclwrs, we'll make one
            // additional attempt at initialization before reporting failure.
            
            if( retry )
            {
                return init( false, filename, lowWaterMark, highWaterMark, clog, errDetails );
            }
            else
            {
                return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR,
                                              "Failed to access the database. Too many simultaneous connections." );
            }
        }

        return errDetails.logDetails( OPTIX_ERROR_DISK_CACHE_DATABASE_ERROR,
                                      "Error when configuring the database." );
    }

    reportSize( "Opened", clog );
    return OPTIX_SUCCESS;
}

OptixResult DiskCacheDatabase::destroy( DeviceContextLogger& clog )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    return destroyLocked( clog );
}

OptixResult DiskCacheDatabase::destroyLocked( DeviceContextLogger& clog )
{
    if( m_sqlite != nullptr )
    {
        reportSize( "Closed", clog );
        sqlite3_close( m_sqlite );
        m_sqlite = nullptr;
    }

    clog.callback( 13, "DiskCacheDatabase", "closed" );
    return OPTIX_SUCCESS;
}

OptixResult DiskCacheDatabase::setPath( const std::string& filename, DeviceContextLogger& clog, ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    if( m_sqlite != nullptr )
    {
        destroyLocked( clog );
    }
    return init( filename, m_lowWaterMark, m_highWaterMark, clog, errDetails );
}

// Execute an sqlite query, and return the result if the operation is not successful.
// In the event that the result is CACHE_BUSY, the query will be retried until the timeout is reached.
#define CONFIGURE_CHECK( query )                                                                                       \
    {                                                                                                                  \
        using namespace std::chrono;                                                                                   \
        auto waitStart = steady_clock::now();                                                                          \
        for( ;; )                                                                                                      \
        {                                                                                                              \
            result = exelwteQuery( clog, query );                                                                      \
            if( result != DiskCacheDatabase::QueryResult::CACHE_BUSY )                                                 \
                break;                                                                                                 \
            if( duration_cast<duration<double>>( steady_clock::now() - waitStart ).count() > m_timeoutInSeconds )      \
                break;                                                                                                 \
        }                                                                                                              \
        if( result != DiskCacheDatabase::QueryResult::CACHE_SUCCESS )                                                  \
        {                                                                                                              \
            return result;                                                                                             \
        }                                                                                                              \
    }

DiskCacheDatabase::QueryResult DiskCacheDatabase::configure( DeviceContextLogger& clog )
{
    clog.callback( 13, "DiskCacheDatabase", "Configuring database..." );

    QueryResult result = QueryResult::CACHE_SUCCESS;

    // General configurations
    CONFIGURE_CHECK( "PRAGMA page_size = 65536;" );

    // Enable relwrsive triggers otherwise DELETE triggers will not fire ON a REPLACE
    CONFIGURE_CHECK( "PRAGMA relwrsive_triggers='ON'" );

    // Switch off the synchronous mode, because otherwise on some systems the
    // transactions are very slow because they wait for a fsync. Since this is
    // a cache, loss of the cache data in case of a computer crash is not an issue.
    CONFIGURE_CHECK( "PRAGMA synchronous = OFF;" );

    // Switch to write-ahead-logging which is supposed to make conlwrrent read / write operations
    // faster. Those can occur if multiple processes are rendering in parallel on the same machine
    // which is something which is supposedly used by some lwstomers who do batch rendering.
    if( !exelwteCommand( clog, "PRAGMA journal_mode=WAL;", "wal" ) )
    {
        clog.callback( 13, "DiskCacheDatabase", "unable to enable write-ahead logging" );
    }

    // Create cache info table: stores additional information to each cache entry
    CONFIGURE_CHECK(
                  "CREATE TABLE IF NOT EXISTS cache_info ("
                  "key VARCHAR(1024) UNIQUE ON CONFLICT REPLACE, "
                  "optix_version VARCHAR(32), "
                  "driver_version VARCHAR(32), "
                  "size INTEGER, "
                  "timestamp INTEGER);" );

    // Create cache data: stores the actual shader cache data
    CONFIGURE_CHECK(
                  "CREATE TABLE IF NOT EXISTS cache_data ("
                  "key VARCHAR(1024) UNIQUE ON CONFLICT REPLACE, "
                  "value BLOB);" );

    // Create globals table: stores arbitrary global variables
    CONFIGURE_CHECK(
                  "CREATE TABLE IF NOT EXISTS globals ("
                  "key VARCHAR(256) UNIQUE ON CONFLICT REPLACE, "
                  "value TEXT);" );

    // Indices
    CONFIGURE_CHECK( "CREATE INDEX IF NOT EXISTS cache_data_key ON cache_data(key);" );
    CONFIGURE_CHECK( "CREATE INDEX IF NOT EXISTS cache_info_key ON cache_info(key);" );

    // Make sure to delete the corresponding entry from cache_info or cache_data if data from one
    // of these tables is deleted. To ensure consistency even if the database is opened through another application.
    CONFIGURE_CHECK(
                  "CREATE TRIGGER IF NOT EXISTS cache_data_delete_info_trigger AFTER DELETE ON cache_data "
                  "FOR EACH ROW "
                  "BEGIN "
                  "DELETE FROM cache_info WHERE key=OLD.key;"
                  "END;" );
    CONFIGURE_CHECK(
                  "CREATE TRIGGER IF NOT EXISTS cache_info_delete_data_trigger AFTER DELETE ON cache_info "
                  "FOR EACH ROW "
                  "BEGIN "
                  "DELETE FROM cache_data WHERE key=OLD.key;"
                  "END;" );

    // Create triggers to automatically keep track of the total cache data size in the globals table
    CONFIGURE_CHECK(
                  "CREATE TRIGGER IF NOT EXISTS total_data_size_delete_trigger AFTER DELETE ON cache_info "
                  "FOR EACH ROW "
                  "BEGIN "
                  "UPDATE globals SET value=value - OLD.size WHERE key='total_data_size';"
                  "END;" );
    CONFIGURE_CHECK(
                  "CREATE TRIGGER IF NOT EXISTS total_data_size_insert_trigger AFTER INSERT ON cache_info "
                  "FOR EACH ROW "
                  "BEGIN "
                  "UPDATE globals SET value=value + NEW.size WHERE key='total_data_size';"
                  "END;" );

    // Set the version of the database schema, if it doesn't already match
    // the version we're trying to set
    if( getSchemaVersion( clog ) != s_schemaVersion )
        setSchemaVersion( s_schemaVersion, clog );

    // Update the total data size global to make sure it is consistent with the actually cached data.
    // In theory it should always stay consistent but technically it is possible to open the database and
    // modify the global values.
    CONFIGURE_CHECK(
                  "INSERT INTO globals (key, value) "
                  "VALUES ('total_data_size', (SELECT COALESCE(SUM(size), 0) FROM cache_info));" );

    clog.callback( 13, "DiskCacheDatabase",
                   corelib::stringf( "Configuration done (schema version %d)", getSchemaVersion( clog ) ).c_str() );
    clog.callback(
        13, "DiskCacheDatabase",
        corelib::stringf( "Current total cache size: %s", bytesToString( queryTotalDataSize( clog ) ).c_str() ).c_str() );
    return QueryResult::CACHE_SUCCESS;
}

#undef CONFIGURE_CHECK

int DiskCacheDatabase::getSchemaVersion( DeviceContextLogger& clog ) const
{
    const char* command = "PRAGMA user_version;";

    sqlite3_stmt* statement = nullptr;
    if( sqlite3_prepare( m_sqlite, command, std::strlen( command ), &statement, nullptr ) != SQLITE_OK )
    {
        clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                       corelib::stringf( "Could not query the database schema version. %s\nStatement is: %s",
                                         sqlite3_errmsg( m_sqlite ), command )
                           .c_str() );
        return -1;
    }
    if( sqlite3_step( statement ) == SQLITE_ROW )
    {
        int version = static_cast<int>( sqlite3_column_int64( statement, 0 ) );
        sqlite3_finalize( statement );
        return version;
    }

    sqlite3_finalize( statement );
    return -1;
}

bool DiskCacheDatabase::setSchemaVersion( int version, DeviceContextLogger& clog )
{
    return exelwteQuery( clog, "PRAGMA user_version = %d;", version ) == QueryResult::CACHE_SUCCESS;
}

bool DiskCacheDatabase::updateCacheAccessTimestamp( const char* key, DeviceContextLogger& clog )
{
    if( !m_sqlite )
        return false;

    return exelwteQuery( clog, "UPDATE cache_info SET timestamp=strftime('%%s', 'now') WHERE key=%Q;", key )
           == QueryResult::CACHE_SUCCESS;
}

size_t DiskCacheDatabase::queryTotalDataSize( DeviceContextLogger& clog ) const
{
    const char* command = "SELECT value FROM globals WHERE key='total_data_size';";

    sqlite3_stmt* statement = nullptr;
    if( sqlite3_prepare( m_sqlite, command, std::strlen( command ), &statement, nullptr ) != SQLITE_OK )
    {
        clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                       corelib::stringf( "Could not query the database size: %s\nStatement is: %s", sqlite3_errmsg( m_sqlite ), command )
                           .c_str() );
        return -1;
    }

    if( sqlite3_step( statement ) == SQLITE_ROW )
    {
        size_t totalSize = static_cast<size_t>( sqlite3_column_int64( statement, 0 ) );
        sqlite3_finalize( statement );
        return totalSize;
    }

    sqlite3_finalize( statement );

    return 0;
}

void DiskCacheDatabase::selectGarbageCollectionItems( size_t minTotalSize, std::vector<KeySizePair>& items, DeviceContextLogger& clog ) const
{
    const char* command = "SELECT key, size FROM cache_info ORDER BY timestamp ASC;";

    sqlite3_stmt* statement = nullptr;
    if( sqlite3_prepare( m_sqlite, command, std::strlen( command ), &statement, nullptr ) != SQLITE_OK )
    {
        clog.callback(
            optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
            corelib::stringf( "Garbage collection failed: %s\nStatement is %s", sqlite3_errmsg( m_sqlite ), command ).c_str() );
        return;
    }

    size_t totalSize = 0;
    while( sqlite3_step( statement ) == SQLITE_ROW && totalSize < minTotalSize )
    {
        const char* key  = (const char*)sqlite3_column_text( statement, 0 );
        size_t      size = static_cast<size_t>( sqlite3_column_int64( statement, 1 ) );

        items.push_back( KeySizePair( key, size ) );
        totalSize += size;
    }

    sqlite3_finalize( statement );
}

bool DiskCacheDatabase::deleteCacheEntry( const std::string& key, DeviceContextLogger& clog )
{
    return exelwteQuery( clog, "DELETE FROM cache_data WHERE key=%Q;", key.c_str() ) == QueryResult::CACHE_SUCCESS;
}

void DiskCacheDatabase::runGarbageCollection( size_t newEntrySize, DeviceContextLogger& clog )
{
    const size_t totalDataSize = queryTotalDataSize( clog );
    if( m_lowWaterMark == 0 || m_highWaterMark == 0 || totalDataSize + newEntrySize <= m_highWaterMark )
        return;

    // Get the list of cache entries that should be deleted in order to delete as many elements as necessary to reach
    // the low-water mark again.

    std::vector<KeySizePair> selectedItems;
    const size_t             requestedSize = totalDataSize + newEntrySize - m_lowWaterMark;

    selectGarbageCollectionItems( requestedSize, selectedItems, clog );

    clog.callback( 12, "DiskCacheDatabase",
                   corelib::stringf( "Running garbage collection:\n"
                                     "  Current cache size: %s\n"
                                     "  Low-water / high-water: %s / %s\n"
                                     "  Requested size: %s\n"
                                     "  Selected items: %zu\n",
                                     bytesToString( totalDataSize ).c_str(), bytesToString( m_lowWaterMark ).c_str(),
                                     bytesToString( m_highWaterMark ).c_str(), bytesToString( requestedSize ).c_str(),
                                     selectedItems.size() )
                       .c_str() );

    // Delete the selected items

    size_t deletedItems     = 0;
    size_t deletedItemsSize = 0;
    for( KeySizePair& item : selectedItems )
    {
        if( deleteCacheEntry( item.first, clog ) )
        {
            deletedItems++;
            deletedItemsSize += item.second;
        }
    }

    clog.callback( 12, "DiskCacheDatabase", corelib::stringf( "Cleaned up %zu"
                                                              " cache entries with a total size of %s"
                                                              " - New cache size: %s",
                                                              deletedItems, bytesToString( deletedItemsSize ).c_str(),
                                                              bytesToString( queryTotalDataSize( clog ) ).c_str() )
                                                .c_str() );

    // We need to issue a VALWUM comamnd to reclaim disk space following garbage
    // collection.  Otherwise, the on-disk database file will continute to grow
    // far beyond the garbage collection limit.
    exelwteCommand( clog, "VALWUM;" );

    reportSize( "Ran garbage collection on", clog );
}

bool DiskCacheDatabase::isOpen() const
{
    std::lock_guard<std::mutex> lock( m_mutex );
    return m_sqlite != nullptr;
}

OptixResult DiskCacheDatabase::setMemoryLimits( size_t lowWaterMark, size_t highWaterMark, DeviceContextLogger& clog, ErrorDetails& errDetails )
{
    std::lock_guard<std::mutex> lock( m_mutex );
    return setMemoryLimitsNoLock( lowWaterMark, highWaterMark, clog, errDetails );
}

OptixResult DiskCacheDatabase::setMemoryLimitsNoLock( size_t lowWaterMark, size_t highWaterMark, DeviceContextLogger& clog, ErrorDetails& errDetails )
{
    // The low-water mark must be greater than the high-water mark but the values 0 are accepted to disable
    // gc.
    if( lowWaterMark > highWaterMark && lowWaterMark > 0 && highWaterMark > 0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "lowWaterMark cannot be larger than highWaterMark" );

    m_lowWaterMark  = lowWaterMark;
    m_highWaterMark = highWaterMark;

    clog.callback( 13, "DiskCacheDatabase",
                   corelib::stringf( "Set memory limits for garbage collection to:  low-water=%s"
                                     ", high-water=%s",
                                     bytesToString( m_lowWaterMark ).c_str(), bytesToString( m_highWaterMark ).c_str() )
                       .c_str() );

    return OPTIX_SUCCESS;
}

void DiskCacheDatabase::getMemoryLimits( size_t& lowWaterMark, size_t& highWaterMark ) const
{
    std::lock_guard<std::mutex> lock( m_mutex );

    lowWaterMark  = m_lowWaterMark;
    highWaterMark = m_highWaterMark;
}

size_t DiskCacheDatabase::getTotalDataSize( DeviceContextLogger& clog ) const
{
    std::lock_guard<std::mutex> lock( m_mutex );
    return queryTotalDataSize( clog );
}

bool DiskCacheDatabase::insertCacheEntry( const char* key, const void* buffer, size_t bufferSize, DeviceContextLogger& clog )
{
    std::lock_guard<std::mutex> lock( m_mutex );

    if( !m_sqlite )
        return false;

    if( m_highWaterMark != 0 && m_lowWaterMark != 0 && bufferSize > m_highWaterMark )
    {
        clog.callback( DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                       corelib::stringf( "Failed to insert cache entry: Entry size (%zu bytes) exceeds cache high "
                                         "water mark (%zu bytes)",
                                         bufferSize, m_highWaterMark )
                           .c_str() );
        return false;
    }

    runGarbageCollection( bufferSize, clog );

    // Begin a new "immediate" transaction to guarantee that no subsequent
    // operations on the same database through the next COMMIT will return
    // SQLITE_BUSY.
    exelwteQuery( clog, "BEGIN IMMEDIATE;" );

    // Insert cache data
    {
        const char*   command    = "INSERT INTO cache_data (key, value) VALUES (?, ?);";
        sqlite3_stmt* statement  = nullptr;
        int           sqliteCode = sqlite3_prepare( m_sqlite, command, std::strlen( command ), &statement, nullptr );

        if( sqliteCode != SQLITE_OK )
        {
            clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                           corelib::stringf( "Failed to insert into the cache. Could not prepare sqlite statement: %s "
                                             "(code: %d)\nStatement is %s",
                                             sqlite3_errmsg( m_sqlite ), sqliteCode, command )
                               .c_str() );
            // Close opened transaction.
            exelwteQuery( clog, "ROLLBACK TRANSACTION" );
            return false;
        }

        // Bind the cache key parameter
        sqliteCode = sqlite3_bind_text( statement, 1, key, -1, nullptr );
        if( sqliteCode != SQLITE_OK )
        {
            clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                           corelib::stringf( "Failed to insert into the cache. Could not bind string to sqlite "
                                             "statement: %s (code: %d)",
                                             sqlite3_errmsg( m_sqlite ), sqliteCode )
                               .c_str() );
            sqlite3_finalize( statement );
            exelwteQuery( clog, "ROLLBACK TRANSACTION" );
            return false;
        }

        // Bind the binary blob to the second parameter of the statement
        sqliteCode = sqlite3_bind_blob( statement, 2, buffer, bufferSize, nullptr );
        if( sqliteCode != SQLITE_OK )
        {
            clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                           corelib::stringf( "Failed to insert into the cache. Could not bind binary blob to sqlite "
                                             "statement: %s (code: %d)",
                                             sqlite3_errmsg( m_sqlite ), sqliteCode )
                               .c_str() );
            sqlite3_finalize( statement );
            exelwteQuery( clog, "ROLLBACK TRANSACTION" );
            return false;
        }
        sqliteCode = sqlite3_step( statement );
        if( sqliteCode != SQLITE_DONE )
        {
            if( sqliteCode == SQLITE_ERROR )
            {
                // In the legacy interface, a specific error code is not produced until
                // sqlite3_reset is called after failure of sqlite3_step
                // (see "Goofy Interface Alert" in sqlite3_step documentation).
                sqliteCode = sqlite3_reset( statement );
            }
            clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                           corelib::stringf( "Failed to insert into the cache. Query failed. %s (code: %d)",
                                             sqlite3_errmsg( m_sqlite ), sqliteCode )
                               .c_str() );
            sqlite3_finalize( statement );
            exelwteQuery( clog, "ROLLBACK TRANSACTION" );
            if( sqliteCode == SQLITE_CORRUPT || sqliteCode == SQLITE_NOTADB )
            {
                // If corruption is detected, try to delete the cache files and reinitialize the database.
                // If that fails, disable caching and notify the user.

                // Grab the filename before closing the database.
                const std::string filename( sqlite3_db_filename( m_sqlite, nullptr ) );

                clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                               corelib::stringf( "Database corruption detected. Deleting cache file: \"%s\".", filename.c_str() ).c_str() );

                // Close the corrupt database.
                destroyLocked( clog );

                ErrorDetails errDetails;
                if( !deleteDbFiles( filename, clog ) )
                {
                    clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                                   "Unable to delete cache file. Disabling caching." );
                }
                else if( init( filename, m_lowWaterMark, m_highWaterMark, clog, errDetails ) != OPTIX_SUCCESS )
                {
                    clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                                   "Unable to re-initialize cache database. Disabling caching." );
                }
            }
            return false;
        }

        sqlite3_finalize( statement );
    }

    // Insert additional cache info
    exelwteQuery( clog,
                  "INSERT INTO cache_info (key, optix_version, driver_version, size, timestamp) "
                  "VALUES (%Q, %Q, %Q, %d, strftime('%%s', 'now'));",
                  key, OPTIX_VERSION_STRING, m_driverVersion.c_str(), bufferSize );

    exelwteQuery( clog, "COMMIT TRANSACTION;" );

    return true;
}

std::unique_ptr<DiskCacheDatabase::ByteBuffer> DiskCacheDatabase::findCacheEntry( const char* key, DeviceContextLogger& clog )
{
    std::lock_guard<std::mutex> lock( m_mutex );

    if( !m_sqlite )
        return nullptr;

    // Select the cache data along with the data size from the database
    char* command = sqlite3_mprintf(
        "SELECT size, value FROM cache_info "
        "INNER JOIN cache_data ON cache_info.key=cache_data.key "
        "WHERE cache_info.key=%Q LIMIT 1;",
        key );

    sqlite3_stmt* statement = nullptr;
    if( sqlite3_prepare( m_sqlite, command, std::strlen( command ), &statement, nullptr ) != SQLITE_OK )
    {
        clog.callback(
            optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
            corelib::stringf( "Failed to search the cache. %s\nStatement is %s", sqlite3_errmsg( m_sqlite ), command ).c_str() );
        sqlite3_free( command );
        return nullptr;
    }

    sqlite3_free( command );

    if( sqlite3_step( statement ) == SQLITE_ROW )
    {
        // Cache hit
        const size_t expectedBufferSize( static_cast<size_t>( sqlite3_column_int64( statement, 0 ) ) );

        const void* blob = sqlite3_column_blob( statement, 1 );
        if( blob == nullptr )
        {
            clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                           corelib::stringf( "Cached data for key \"%s\" is null", key ).c_str() );
            return nullptr;
        }

        std::unique_ptr<ByteBuffer> buffer( new ByteBuffer( expectedBufferSize ) );
        memcpy( &( *buffer )[0], blob, expectedBufferSize );

        sqlite3_finalize( statement );

        updateCacheAccessTimestamp( key, clog );

        return buffer;
    }

    // Cache miss
    sqlite3_finalize( statement );

    return nullptr;
}

DiskCacheDatabase::QueryResult DiskCacheDatabase::exelwteQuery( DeviceContextLogger& clog, const char* fmt, ... )
{
    if( !m_sqlite )
        return QueryResult::CACHE_FAILED;

    va_list args;
    va_start( args, fmt );
    char* command = sqlite3_vmprintf( fmt, args );
    va_end( args );

    char* errMsg     = nullptr;
    int   sqliteCode = sqlite3_exec( m_sqlite, command, nullptr, nullptr, &errMsg );
    if( sqliteCode == SQLITE_BUSY )
    {
        sqlite3_free( command );
        sqlite3_free( errMsg );
        return QueryResult::CACHE_BUSY;
    }
    if( sqliteCode != SQLITE_OK )
    {
        clog.callback(
            optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
            corelib::stringf( "Failed to execute query: %s (code: %d)\nStatement: %s\n", errMsg, sqliteCode, command ).c_str() );
        sqlite3_free( command );
        sqlite3_free( errMsg );
        if( sqliteCode == SQLITE_CORRUPT || sqliteCode == SQLITE_NOTADB )
            return QueryResult::CACHE_CORRUPT;
        else
            return QueryResult::CACHE_FAILED;
    }
    sqlite3_free( command );
    sqlite3_free( errMsg );
    return QueryResult::CACHE_SUCCESS;
}

bool DiskCacheDatabase::exelwteCommand( DeviceContextLogger& clog, const char* command, const char* expected )
{
    if( !m_sqlite )
        return false;

    // This struct is used as an in/out parameter by the row-processing callback.
    struct ColumnMatch
    {
        std::string expected; // The expected string to be matched with the column value
        bool        match;    // The result of the match
    };

    // A callback function is needed to process the rows returned by the query.
    auto callback = []( void* ptr, int numCols, char** colVals, char** colNames ) {
        ColumnMatch* cm = reinterpret_cast<ColumnMatch*>( ptr );
        if( !cm->expected.empty() && ( numCols < 1 || cm->expected != colVals[0] ) )
            cm->match = false;
        return 1;
    };

    ColumnMatch cm         = { std::string( expected ? expected : "" ), true };
    char*       errMsg     = nullptr;
    int         sqliteCode = sqlite3_exec( m_sqlite, command, callback, &cm, &errMsg );
    if( sqliteCode == SQLITE_ERROR )
    {
        clog.callback(
            optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
            corelib::stringf( "Query exelwtion failed: %s (code: %d)", errMsg, sqliteCode ).c_str() );
        sqlite3_free( errMsg );
        return false;
    }
    sqlite3_free( errMsg );
    if( expected != nullptr && cm.match == false )
        return false;
    else
        return true;
}

void DiskCacheDatabase::reportSize( const std::string& preface, DeviceContextLogger& clog )
{
    const char* filename = sqlite3_db_filename( m_sqlite, nullptr );
    clog.callback( 4, "DISK CACHE", corelib::stringf( "%s database: \"%s\"", preface.c_str(), filename ).c_str() );
    clog.callback(
        4, "DISK CACHE",
        corelib::stringf( "    Cache data size: \"%s\"", bytesToString( queryTotalDataSize( clog ) ).c_str() ).c_str() );
}

bool DiskCacheDatabase::deleteDbFiles( const std::string& filename, DeviceContextLogger& clog )
{
    if( prodlib::fileExists( filename.c_str() ) && std::remove( filename.c_str() ) )
    {
        clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                       corelib::stringf( "Unable to remove database file \"%s\".", filename.c_str() ).c_str() );
        return false;
    }

    // SQLITE creates a pair of additional files for write-ahead logging (WAL) and to coordinate
    // shared access from multiple clients (SHM). We should delete these files if they exist, but
    // only consider it an error if the WAL file is not deleted.
    const std::string walFile( filename + "-wal" );
    if( prodlib::fileExists( walFile.c_str() ) && std::remove( walFile.c_str() ) )
    {
        clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                       corelib::stringf( "Unable to remove WAL file \"%s\".", walFile.c_str() ).c_str() );
        return false;
    }

    const std::string shmFile( filename + "-shm" );
    if( prodlib::fileExists( shmFile.c_str() ) && std::remove( shmFile.c_str() ) )
    {
        clog.callback( optix_exp::DeviceContextLogger::LOG_LEVEL::Warning, "DiskCacheDatabase",
                       corelib::stringf( "Unable to remove SHM file \"%s\".", shmFile.c_str() ).c_str() );
    }
    return true;
}

bool DiskCacheDatabase::beginImmediateForTest( DeviceContextLogger& clog )
{
    return exelwteQuery( clog, "BEGIN IMMEDIATE;" ) == QueryResult::CACHE_SUCCESS;
}

bool DiskCacheDatabase::commitTransactionForTest( DeviceContextLogger& clog )
{
    return exelwteQuery( clog, "COMMIT TRANSACTION;" ) == QueryResult::CACHE_SUCCESS;
}

}  // namespace optix_exp
