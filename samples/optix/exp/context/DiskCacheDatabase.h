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

#pragma once

#define __OPTIX_INCLUDE_INTERNAL_HEADERS__
#include <optix_7_types.h>
#undef __OPTIX_INCLUDE_INTERNAL_HEADERS__

#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct sqlite3;

namespace optix_exp {
class DeviceContextLogger;
class ErrorDetails;

///
/// This class handles all SQLite transactions to store or retrieve cache entries.
/// It is also responsible for the creation, configuration and update of the database
/// scheme as well as handling the low- and high-water mark limits of the database.
///
class DiskCacheDatabase
{
  public:
    typedef std::vector<char> ByteBuffer;

    DiskCacheDatabase();
    DiskCacheDatabase( const DiskCacheDatabase& ) = delete;
    DiskCacheDatabase& operator=( const DiskCacheDatabase& ) = delete;
    ~DiskCacheDatabase()                                     = default;

    OptixResult init( const std::string& filename, size_t lowWaterMark, size_t highWaterMark, DeviceContextLogger& clog, ErrorDetails& errDetails );
    OptixResult destroy( DeviceContextLogger& clog );
    OptixResult setPath( const std::string& filename, DeviceContextLogger& clog, ErrorDetails& errDetails );

    /// Returns true if the database has been opened successfully
    bool isOpen() const;

    /// Sets the memory limits for the garbage collection
    ///
    /// The DiskCacheDatabase uses a garbage collection scheme to attempt to keep the total amount of
    /// memory, i.e., disk space, below a configurable limit called high water mark.
    /// If that limit is exceeded the garbage collection reduces memory usage until another limit,
    /// the low water mark, is reached (or no further memory reduction is possible).
    ///
    /// \return Returns true if the memory limits have been changed successfully and false in case of invalid
    ///         parameter values.
    OptixResult setMemoryLimits( size_t lowWaterMark, size_t highWaterMark, DeviceContextLogger& clog, ErrorDetails& errDetails );

    /// Returns the memory limits for the garbage collection (0 means disabled).
    void getMemoryLimits( size_t& lowWaterMark, size_t& highWaterMark ) const;

    /// Returns the total size of the cached data without the size of the meta-information and SQLite overhead.
    size_t getTotalDataSize( DeviceContextLogger& clog ) const;

    /// Inserts a cache entry with the given key into the database and returns true in case of success.
    bool insertCacheEntry( const char* key, const void* buffer, size_t bufferSize, DeviceContextLogger& clog );

    /// Finds a cache entry to a given key in the database. Returns the value of the cache key
    /// or a nullptr.
    std::unique_ptr<ByteBuffer> findCacheEntry( const char* key, DeviceContextLogger& clog );

  protected:
    // Expose a few things for testing.
    sqlite3* getSQLiteForTest() const { return m_sqlite; }
    bool     beginImmediateForTest( DeviceContextLogger& clog );
    bool     commitTransactionForTest( DeviceContextLogger& clog );

  private:
    enum class QueryResult
    {
        CACHE_SUCCESS,
        CACHE_FAILED,
        CACHE_CORRUPT,
        CACHE_BUSY
    };

    static const int s_schemaVersion = 1;

    OptixResult init( bool retry, const std::string& filename, size_t lowWaterMark, size_t highWaterMark, DeviceContextLogger& clog, ErrorDetails& errDetails );

    /// Sets the memory limits without locking the mutex. Mutex handling is left to the caller.
    OptixResult setMemoryLimitsNoLock( size_t lowWaterMark, size_t highWaterMark, DeviceContextLogger& clog, ErrorDetails& errDetails );

    /// Closes the database connection. Mutex handling is left to the caller.
    OptixResult destroyLocked( DeviceContextLogger& clog );

    /// Exelwtes a SQLite query and returns true in case of success
    QueryResult exelwteQuery( DeviceContextLogger& clog, const char* fmt, ... );

    /// Exelwtes a single SQLite command and returns true in case of success.
    /// For commands that return a string result, pass in an optional string
    /// to compare with the result, and return false if the strings don't match.
    bool exelwteCommand( DeviceContextLogger& clog, const char* command, const char* expected = nullptr );

    /// Configures / updates the database scheme
    QueryResult configure( DeviceContextLogger& clog );

    /// Returns the schema version of the opened database
    int getSchemaVersion( DeviceContextLogger& clog ) const;

    /// Sets the schema version of the opened database
    bool setSchemaVersion( int version, DeviceContextLogger& clog );

    /// Updates the last access time of the given cache key
    bool updateCacheAccessTimestamp( const char* key, DeviceContextLogger& clog );

    /// Returns the total size of the cached data in bytes
    size_t queryTotalDataSize( DeviceContextLogger& clog ) const;

    typedef std::pair<std::string, size_t> KeySizePair;

    /// Creates a list of cache keys for deletion and returns the total size of the elements
    void selectGarbageCollectionItems( size_t minTotalSize, std::vector<KeySizePair>& items, DeviceContextLogger& clog ) const;

    /// Deletes the given cache entry and it's meta data from the database
    bool deleteCacheEntry( const std::string& key, DeviceContextLogger& clog );

    /// Removes least recently used elements until the low water mark has been reached or no
    /// elements could be removed any more.
    void runGarbageCollection( size_t newEntrySize, DeviceContextLogger& clog );

    /// Report the size of the database on open and close.
    void reportSize( const std::string& preface, DeviceContextLogger& clog );

    /// Attempts to delete the on-disk database files
    bool deleteDbFiles( const std::string& filename, DeviceContextLogger& clog );

  private:
    const std::string  m_driverVersion;
    mutable std::mutex m_mutex;
    sqlite3*           m_sqlite           = nullptr;
    size_t             m_lowWaterMark     = 0;
    size_t             m_highWaterMark    = 0;
    double             m_timeoutInSeconds = 10.0;
};
}  // namespace optix_exp
