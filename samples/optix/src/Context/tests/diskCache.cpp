
/*
 * Copyright (c) 2021, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <exp/context/DeviceContext.h>
#include <exp/context/DiskCacheDatabase.h>
#include <exp/context/ErrorHandling.h>

#include <Exceptions/DatabaseFilePermissions.h>
#include <corelib/system/System.h>
#include <prodlib/system/Logger.h>
#include <prodlib/system/System.h>

#include <gtest/gtest.h>
#include <gtest/internal/gtest-filepath.h>

#include <support/sqlite/sqlite3.h>

#include <chrono>
#include <thread>
#include <vector>

namespace {

// DiskCacheDatabaseProbe increases visibility of test methods
// from protected to public.
class DiskCacheDatabaseProbe : public optix_exp::DiskCacheDatabase
{
  public:
    using DiskCacheDatabase::getSQLiteForTest;
    using DiskCacheDatabase::beginImmediateForTest;
    using DiskCacheDatabase::commitTransactionForTest;
};

}  // namespace

namespace optix {
// Needs to be in optix namespace for FRIEND_TEST to work
class DiskCacheDatabaseTest : public testing::Test
{
  public:
    virtual void SetUp()
    {
        // Create a dummy file for the db
        corelib::writeFile( "test.db", "", 0 );
    }

    virtual void TearDown() { remove( "test.db" ); }
};

TEST_F( DiskCacheDatabaseTest, OpensDatabase )
{
    {
        DiskCacheDatabaseProbe         db;
        optix_exp::DeviceContextLogger clog;
        optix_exp::ErrorDetails        errDetails;
        db.init( "test.db", 0, 0, clog, errDetails );
        EXPECT_TRUE( db.isOpen() );
        db.destroy( clog );
    }
}

TEST( DiskCacheDatabaseNotOpenedTest, HandlesNotOpenedDatabase )
{
    EXPECT_FALSE( testing::internal::FilePath( "dir_should_not_exist" ).DirectoryExists() );
    DiskCacheDatabaseProbe         db;
    optix_exp::ErrorDetails        errDetails;
    optix_exp::DeviceContextLogger clog;
    EXPECT_NE( db.init( "dir_should_not_exist/test.db", 0, 0, clog, errDetails ), OPTIX_SUCCESS );
}

TEST_F( DiskCacheDatabaseTest, InsertsAndFindsCacheEntry )
{
    DiskCacheDatabaseProbe         db;
    optix_exp::DeviceContextLogger clog;
    optix_exp::ErrorDetails        errDetails;
    db.init( "test.db", 0, 0, clog, errDetails );
    EXPECT_TRUE( db.insertCacheEntry( "key", "123", 3, clog ) ) << "1st insert command";
    EXPECT_TRUE( db.insertCacheEntry( "key", "123", 3, clog ) ) << "Insert the same key again";
    EXPECT_TRUE( db.insertCacheEntry( "key-2", "1234", 4, clog ) ) << "Insert another key";

    std::unique_ptr<optix_exp::DiskCacheDatabase::ByteBuffer> buffer( db.findCacheEntry( "key", clog ) );
    EXPECT_NE( buffer, nullptr ) << "Find does not return a nullptr";
    EXPECT_EQ( buffer->size(), 3u ) << "The returned buffer has the expected size";
    EXPECT_EQ( strncmp( (const char*)( buffer->data() ), "123", 3 ), 0 )
        << "The returned buffer has the expected content";
    db.destroy( clog );
}

TEST_F( DiskCacheDatabaseTest, TestGarbageCollection )
{
    DiskCacheDatabaseProbe         db;
    optix_exp::DeviceContextLogger clog;
    optix_exp::ErrorDetails        errDetails;
    db.init( "test.db", 0, 0, clog, errDetails );
    EXPECT_TRUE( db.isOpen() );

    const size_t targetLowWaterMark  = 5u * 1024u;
    const size_t targetHighWaterMark = 10u * 1024u;

    // Test low-water mark and high-water mark limit values
    {
        size_t lowWaterMark  = 0;
        size_t highWaterMark = 0;

        // Test defaults
        db.getMemoryLimits( lowWaterMark, highWaterMark );
        EXPECT_EQ( lowWaterMark, 0u ) << "Default value of low-water mark is 0";
        EXPECT_EQ( highWaterMark, 0u ) << "Default value of high-water mark is 0";

        // Low-water mark needs to be < high-water mark
        EXPECT_FALSE( db.setMemoryLimits( 100, 1, clog, errDetails ) == OPTIX_SUCCESS );
        db.getMemoryLimits( lowWaterMark, highWaterMark );
        EXPECT_EQ( lowWaterMark, 0u ) << "Low-water mark is still 0 after setting invalid value";
        EXPECT_EQ( highWaterMark, 0u ) << "High-water mark is still 0 after setting invalid value";

        // Test valid values
        EXPECT_TRUE( db.setMemoryLimits( targetLowWaterMark, targetHighWaterMark, clog, errDetails ) == OPTIX_SUCCESS );
        db.getMemoryLimits( lowWaterMark, highWaterMark );
        EXPECT_EQ( lowWaterMark, targetLowWaterMark ) << "Low-water mark is set correctly to " << lowWaterMark;
        EXPECT_EQ( highWaterMark, targetHighWaterMark ) << "High-water mark is set correctly to " << highWaterMark;
    }

    // Generate a random blob for testing
    std::vector<unsigned char> blob( 1024u );

    // Test actual garbage collection
    {
        EXPECT_EQ( db.getTotalDataSize( clog ), 0u ) << "Total data size of the cache db is 0 in the beginning";

        // Fill up the cache until the size reaches the high-water mark
        for( size_t i = 0; i < targetHighWaterMark; i += blob.size() )
        {
            EXPECT_TRUE( db.insertCacheEntry( std::to_string( i ).c_str(), blob.data(), blob.size(), clog ) );
            EXPECT_EQ( db.getTotalDataSize( clog ), i + blob.size() );
        }

        // Since the cache size has reached the high-water mark, adding a new cache entry should trigger garbage collection
        // and reduce the cache size to the value of low-water mark.
        EXPECT_TRUE( db.insertCacheEntry( "new-key", blob.data(), blob.size(), clog ) );
        EXPECT_EQ( db.getTotalDataSize( clog ), targetLowWaterMark );

        // This means also that the first 5 elements have been removed from the database
        for( size_t i = 0; i < targetLowWaterMark; i += blob.size() )
        {
            std::string key = std::to_string( i );
            EXPECT_EQ( db.findCacheEntry( key.c_str(), clog ), nullptr ) << "Element with key " << key
                                                                         << " not in cache db anymore due to gc";
        }

        // Test high water mark not being exceeded
        // Fill up the cache until the size reaches the high-water mark - blob.size()
        for( size_t i = targetLowWaterMark; i < targetHighWaterMark - blob.size(); i += blob.size() )
        {
            EXPECT_TRUE( db.insertCacheEntry( ( std::to_string( i ) + "-2" ).c_str(), blob.data(), blob.size(), clog ) );
            EXPECT_EQ( db.getTotalDataSize( clog ), i + blob.size() );
        }
        // Add an entry that causes the cache to almost reach the high water mark
        // without actually hitting it.
        std::vector<unsigned char> blob2( 1023u );
        EXPECT_TRUE( db.insertCacheEntry( "small-blob", blob2.data(), blob2.size(), clog ) );
        // In next size checks we will no longer hit the water marks exactly,
        // but will need to take the diff of this smaller entry into account.
        int    blobSizeDiff      = blob.size() - blob2.size();
        size_t expectedCacheSize = targetHighWaterMark - blobSizeDiff;
        EXPECT_EQ( db.getTotalDataSize( clog ), expectedCacheSize );

        // Insert new entry which exceeds the remaining free space should trigger
        // garbage collection. Afterwards the database size will be
        // targetLowWaterMark - blobSizeDiff.
        EXPECT_TRUE( db.insertCacheEntry( "new-key-2", blob.data(), blob.size(), clog ) );
        expectedCacheSize = targetLowWaterMark - blobSizeDiff;
        EXPECT_EQ( db.getTotalDataSize( clog ), expectedCacheSize );

        // Test equal values
        size_t lowWaterMark           = 0;
        size_t highWaterMark          = 0;
        size_t newTargetHighWaterMark = targetLowWaterMark;
        EXPECT_TRUE( db.setMemoryLimits( targetLowWaterMark, newTargetHighWaterMark, clog, errDetails ) == OPTIX_SUCCESS );
        db.getMemoryLimits( lowWaterMark, highWaterMark );
        EXPECT_EQ( lowWaterMark, targetLowWaterMark ) << "Low-water mark is set correctly to " << lowWaterMark;
        EXPECT_EQ( highWaterMark, newTargetHighWaterMark ) << "High-water mark is set correctly to " << highWaterMark;
        EXPECT_EQ( lowWaterMark, highWaterMark );
        // After the previous test, the current cache size is higher than the low water mark, which
        // is the new high water mark, so adding another entry should trigger gc again.
        EXPECT_TRUE( db.insertCacheEntry( "new-key-3", blob.data(), blob.size(), clog ) );
        // GC needs to free up enough space so the new entry fits without exceeding the high water mark.
        // So the new cache size should be: targetLowWaterMark - blobSizeDiff
        expectedCacheSize = targetLowWaterMark - blobSizeDiff;
        EXPECT_EQ( db.getTotalDataSize( clog ), targetLowWaterMark - blobSizeDiff );
        // Another element was removed from the cache (the 6th one).
        std::string key = std::to_string( targetLowWaterMark );
        EXPECT_EQ( db.findCacheEntry( key.c_str(), clog ), nullptr ) << "Element with key " << key
                                                                     << " not in cache db anymore due to gc";

        // Test too large cache entry for high water mark fails
        std::vector<unsigned char> largeBlob( newTargetHighWaterMark + 1 );
        EXPECT_FALSE( db.insertCacheEntry( "large-blob-key", largeBlob.data(), largeBlob.size(), clog ) );
        // size should be unchanged.
        EXPECT_EQ( db.getTotalDataSize( clog ), expectedCacheSize );
        EXPECT_EQ( db.findCacheEntry( "large-blob-key", clog ), nullptr )
            << "Element with key large-blob-key not in cache db";
        // Test large entry works with gc disabled
        EXPECT_TRUE( db.setMemoryLimits( 0, 0, clog, errDetails ) == OPTIX_SUCCESS );
        EXPECT_TRUE( db.insertCacheEntry( "large-blob-key", largeBlob.data(), largeBlob.size(), clog ) );
        EXPECT_EQ( db.getTotalDataSize( clog ), expectedCacheSize + largeBlob.size() );
        EXPECT_NE( db.findCacheEntry( "large-blob-key", clog ), nullptr )
            << "Element with key large-blob-key in cache db";
    }
    db.destroy( clog );
}

void insertManyCacheEntries( DiskCacheDatabaseProbe* db )
{
    optix_exp::DeviceContextLogger clog;
    for( int i = 0; i < 1000; ++i )
    {
        std::vector<unsigned char> blob( 1024u );
        ASSERT_TRUE( db->insertCacheEntry( std::to_string( i ).c_str(), blob.data(), blob.size(), clog ) );

        std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
    }
}

TEST_F( DiskCacheDatabaseTest, TestMultithreading )
{
    // Open the same database file within different DiskCacheDatabase objects

    DiskCacheDatabaseProbe         db0;
    optix_exp::DeviceContextLogger clog;
    optix_exp::ErrorDetails        errDetails;
    db0.init( "test.db", 0, 0, clog, errDetails );
    ASSERT_TRUE( db0.isOpen() );

    DiskCacheDatabaseProbe db1;
    db1.init( "test.db", 0, 0, clog, errDetails );
    ASSERT_TRUE( db1.isOpen() );

    std::thread t0( insertManyCacheEntries, &db0 );
    std::thread t1( insertManyCacheEntries, &db1 );

    t0.join();
    t1.join();
    db0.destroy( clog );
    db1.destroy( clog );
}

TEST_F( DiskCacheDatabaseTest, TestWAL )
{
    DiskCacheDatabaseProbe         db1;
    optix_exp::DeviceContextLogger clog;
    optix_exp::ErrorDetails        errDetails;
    db1.init( "test.db", 0, 0, clog, errDetails );
    DiskCacheDatabaseProbe db2;
    db2.init( "test.db", 0, 0, clog, errDetails );
    EXPECT_TRUE( db1.isOpen() );
    EXPECT_TRUE( db2.isOpen() );
    ASSERT_TRUE( prodlib::fileExists( "test.db" ) );
    ASSERT_TRUE( prodlib::fileExists( "test.db-wal" ) );
    ASSERT_TRUE( prodlib::fileExists( "test.db-shm" ) );
    db1.destroy( clog );
    db2.destroy( clog );
}

TEST_F( DiskCacheDatabaseTest, TestSQLiteTimeout )
{
    DiskCacheDatabaseProbe         db0;
    optix_exp::DeviceContextLogger clog0;
    optix_exp::ErrorDetails        errDetails;
    db0.init( "test.db", 0, 0, clog0, errDetails );
    ASSERT_TRUE( db0.isOpen() );

    DiskCacheDatabaseProbe         db1;
    optix_exp::DeviceContextLogger clog1;
    db1.init( "test.db", 0, 0, clog1, errDetails );
    ASSERT_TRUE( db1.isOpen() );

    constexpr int TIMEOUT = 500;

    // lower sqlite timeout to reduce testing time
    ASSERT_TRUE( sqlite3_busy_timeout( db0.getSQLiteForTest(), TIMEOUT ) == SQLITE_OK );
    ASSERT_TRUE( sqlite3_busy_timeout( db1.getSQLiteForTest(), TIMEOUT ) == SQLITE_OK );

    bool result1, result2;

    auto timeoutTrigger = [TIMEOUT]( DiskCacheDatabaseProbe* db, bool& result, optix_exp::DeviceContextLogger& clog ) {
        result = db->beginImmediateForTest( clog );
        std::this_thread::sleep_for( std::chrono::milliseconds( TIMEOUT + 5000 ) );
        if( result )
            db->commitTransactionForTest( clog );
    };

    std::thread t0( timeoutTrigger, &db0, std::ref( result1 ), std::ref( clog0 ) );
    std::thread t1( timeoutTrigger, &db1, std::ref( result2 ), std::ref( clog1 ) );

    t0.join();
    t1.join();

    int errCode1 = sqlite3_errcode( db0.getSQLiteForTest() );
    int errCode2 = sqlite3_errcode( db1.getSQLiteForTest() );

    // Only one of the two calls should succeed. Cannot predict thread interleaving.
    // The failing one should fail with SQLITE_BUSY.
    if( result1 )
    {
        EXPECT_FALSE( result2 );
        EXPECT_EQ( SQLITE_BUSY, errCode2 );
    }
    else
    {
        EXPECT_TRUE( result2 );
        EXPECT_EQ( SQLITE_BUSY, errCode1 );
    }
    db0.destroy( clog0 );
    db1.destroy( clog1 );
}
}
