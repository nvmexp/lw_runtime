
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

#include <corelib/system/System.h>
#include <prodlib/system/System.h>
#include <optix_world.h>

#include <gtest/gtest.h>
#include <gtest/internal/gtest-filepath.h>
#include <srcTests.h>

#include <algorithm>
#include <cstdlib>

//------------------------------------------------------------------------------
// TODO: These are really blackbox tests of the public API, so they should
//       probably be moved somewhere else.
namespace {
class DiskCacheAPI : public testing::Test
{
  public:
    DiskCacheAPI() {}

    void SetUp() override { m_context = optix::Context::create(); }

    void TearDown() override
    {
        if( m_context )
            m_context->destroy();

        cleanupDirectory( m_location );
    }

    void createPrograms()
    {
        std::string ptxFilePath = ptxPath( m_target, "diskCache.lw" );
        for( int i = 0; i < 20; ++i )
        {
            std::string pgmName = "unique_program_" + std::to_string( i );
            m_context->createProgramFromPTXFile( ptxFilePath, pgmName );
        }
    }

    void initializeDirectory( const std::string path )
    {
        prodlib::createDir( path.c_str() );
        std::string cacheFile = path + "/optixcache.db";
        remove( cacheFile.c_str() );
    }

    // Delete the optixcache.db file in the directory, and try to remove the directory
    void cleanupDirectory( const std::string path )
    {
        std::string cacheFile = path + "/optixcache.db";
        remove( cacheFile.c_str() );
        remove( path.c_str() );
    }

    optix::Context m_context;
    std::string    m_target   = "test_Context";
    std::string    m_location = "";
};

class DiskCacheAPIElwVar : public DiskCacheAPI
{
public:
    void SetUp() override {}
};
}  // namespace

TEST_F( DiskCacheAPI, SetLocation )
{
    m_location = corelib::getLwrrentDir() + "/test_diskCacheAPI_tmp";
    std::replace( m_location.begin(), m_location.end(), '\\', '/' );
    initializeDirectory( m_location );

    m_context->setDiskCacheLocation( m_location );
    ASSERT_EQ( m_location, m_context->getDiskCacheLocation() );

    std::string cacheFile = m_context->getDiskCacheLocation() + "/optixcache.db";
    ASSERT_TRUE( prodlib::fileExists( cacheFile.c_str() ) );
}

TEST_F( DiskCacheAPI, SetLocationWithUnicode )
{
    m_location = corelib::getLwrrentDir() + "/test_diskCacheAPI_ünicode/";
    std::replace( m_location.begin(), m_location.end(), '\\', '/' );
    initializeDirectory( m_location );

    m_context->setDiskCacheLocation( m_location );
    ASSERT_EQ( m_location, m_context->getDiskCacheLocation() );
    ASSERT_TRUE( prodlib::dirExists( m_location.c_str() ) );

    std::string cacheFile = m_context->getDiskCacheLocation() + "/optixcache.db";
    ASSERT_TRUE( prodlib::fileExists( cacheFile.c_str() ) );
}

TEST_F( DiskCacheAPI, SetIlwalidLocationThrows )
{
#ifdef _WIN32
    m_location = "C:/?bogus?";
#else
    m_location = "/dev/null/bogus/";
#endif
    ASSERT_ANY_THROW( m_context->setDiskCacheLocation( m_location ) );
}

TEST_F( DiskCacheAPI, PopulateNewCache )
{
    m_location = corelib::getLwrrentDir() + "/test_diskCacheAPI_tmp";
    initializeDirectory( m_location );

    ASSERT_NO_THROW( m_context->setDiskCacheLocation( m_location ) );

    // Create a bunch of programs, which should be added to the new cache and
    // reflected as an increase in the size of the database file.
    std::string cacheFile   = m_context->getDiskCacheLocation() + "/optixcache.db";
    long long   initialSize = corelib::fileSize( cacheFile.c_str() );
    createPrograms();

    // Since write-ahead logging is enabled, we need to destroy the Context
    // to flush the entries from the journal files to the main database file.
    if( m_context )
    {
        m_context->destroy();
        m_context = nullptr;
    }

    long long finalSize = corelib::fileSize( cacheFile.c_str() );
    ASSERT_TRUE( finalSize > initialSize );
}

TEST_F( DiskCacheAPI, SetMemoryLimits )
{
    m_location = corelib::getLwrrentDir() + "/test_diskCacheAPI_tmp";
    initializeDirectory( m_location );

    m_context->setDiskCacheLocation( m_location );
    std::string cacheFile = m_context->getDiskCacheLocation() + "/optixcache.db";

    // Setup the memory limits for garbage collection
    size_t lowLimitIn  = 1 << 15;  // 32 KB
    size_t highLimitIn = 1 << 17;  // 128 KB
    ASSERT_NO_THROW( m_context->setDiskCacheMemoryLimits( lowLimitIn, highLimitIn ) );

    // Make sure the limits were set correctly
    RTsize lowLimitOut  = 0;
    RTsize highLimitOut = 0;
    ASSERT_NO_THROW( m_context->getDiskCacheMemoryLimits( lowLimitOut, highLimitOut ) );
    ASSERT_EQ( lowLimitIn, lowLimitOut );
    ASSERT_EQ( highLimitIn, highLimitOut );
}

TEST_F( DiskCacheAPI, SetIlwalidMemoryLimitsThrows )
{
    m_location = corelib::getLwrrentDir() + "/test_diskCacheAPI_tmp";
    initializeDirectory( m_location );

    m_context->setDiskCacheLocation( m_location );

    // If either limit is non-zero, the high limit must exceed the low limit.
    // Attempting to set invalid limits should trigger an exception.
    size_t lowLimitIn  = 1 << 17;  // 128 KB
    size_t highLimitIn = 1 << 15;  // 32 KB
    ASSERT_ANY_THROW( m_context->setDiskCacheMemoryLimits( lowLimitIn, highLimitIn ) );
}

static void setElwVar( const std::string& varName, const std::string& val )
{
#ifdef WIN32
    _putelw( std::string( varName + "=" + val ).c_str() );
#else
    if( val.empty() )
        unsetelw( varName.c_str() );
    else
        setelw( varName.c_str(), val.c_str(), 1 );
#endif
}

TEST_F( DiskCacheAPIElwVar, SetLocationWithElw )
{
    m_location = corelib::getLwrrentDir() + "/test_diskCacheAPI_tmp";
    std::replace( m_location.begin(), m_location.end(), '\\', '/' );
    initializeDirectory( m_location );

    setElwVar( "OPTIX_CACHE_PATH", m_location );

    ASSERT_NO_THROW( m_context = optix::Context::create() );

    ASSERT_EQ( m_location, m_context->getDiskCacheLocation() );

    std::string cacheFile = m_context->getDiskCacheLocation() + "/optixcache.db";
    ASSERT_TRUE( prodlib::fileExists( cacheFile.c_str() ) );
    setElwVar( "OPTIX_CACHE_PATH", "" );
}

TEST_F( DiskCacheAPIElwVar, SetSizeWithElw )
{
    m_location = corelib::getLwrrentDir() + "/test_diskCacheAPI_tmp";
    std::replace( m_location.begin(), m_location.end(), '\\', '/' );
    initializeDirectory( m_location );

    setElwVar( "OPTIX_CACHE_PATH", m_location );

    RTsize highLimitIn = 1 << 22;
    RTsize lowLimitIn  = highLimitIn / 2;
    setElwVar( "OPTIX_CACHE_MAXSIZE", std::to_string( highLimitIn ).c_str() );

    ASSERT_NO_THROW( m_context = optix::Context::create() );

    // Make sure the limits were set correctly
    RTsize lowLimitOut  = 0;
    RTsize highLimitOut = 0;
    ASSERT_NO_THROW( m_context->getDiskCacheMemoryLimits( lowLimitOut, highLimitOut ) );
    ASSERT_EQ( lowLimitIn, lowLimitOut );
    ASSERT_EQ( highLimitIn, highLimitOut );

    setElwVar( "OPTIX_CACHE_MAXSIZE", "" );
    setElwVar( "OPTIX_CACHE_PATH", "" );
}

TEST_F( DiskCacheAPIElwVar, ElwSizeTakesPrecedence )
{
    m_location = corelib::getLwrrentDir() + "/test_diskCacheAPI_tmp";
    std::replace( m_location.begin(), m_location.end(), '\\', '/' );
    initializeDirectory( m_location );

    setElwVar( "OPTIX_CACHE_PATH", m_location );

    RTsize highLimitIn = 1 << 22;
    RTsize lowLimitIn  = highLimitIn / 2;
    setElwVar( "OPTIX_CACHE_MAXSIZE", std::to_string( highLimitIn ).c_str() );

    ASSERT_NO_THROW( m_context = optix::Context::create() );

    RTsize apiHighLimitIn = 1 << 24;
    RTsize apiLowLimitIn  = apiHighLimitIn / 2;

    ASSERT_NO_THROW( m_context->setDiskCacheMemoryLimits( apiLowLimitIn, apiHighLimitIn ) );

    // Make sure the limits set via the environment are preserved
    RTsize lowLimitOut  = 0;
    RTsize highLimitOut = 0;
    ASSERT_NO_THROW( m_context->getDiskCacheMemoryLimits( lowLimitOut, highLimitOut ) );
    ASSERT_EQ( lowLimitIn, lowLimitOut );
    ASSERT_EQ( highLimitIn, highLimitOut );

    setElwVar( "OPTIX_CACHE_MAXSIZE", "" );
    setElwVar( "OPTIX_CACHE_PATH", "" );
}

TEST_F( DiskCacheAPIElwVar, DisableWithElw )
{
    setElwVar( "OPTIX_CACHE_MAXSIZE", "0" );
    
    // This directory and the cache file *should* never be created
    m_location = corelib::getLwrrentDir() + "/test_diskCacheAPI_tmp";
    std::replace( m_location.begin(), m_location.end(), '\\', '/' );

    setElwVar( "OPTIX_CACHE_PATH", m_location );

    ASSERT_NO_THROW( m_context = optix::Context::create() );

    std::string locationOut;
    ASSERT_NO_THROW( locationOut = m_context->getDiskCacheLocation() );
    ASSERT_EQ( "", locationOut );

    std::string cacheFile;
    ASSERT_NO_THROW( cacheFile = m_context->getDiskCacheLocation() + "/optixcache.db" );
    ASSERT_FALSE( prodlib::fileExists( cacheFile.c_str() ) );

    setElwVar( "OPTIX_CACHE_MAXSIZE", "" );
    setElwVar( "OPTIX_CACHE_PATH", "" );
}

TEST_F( DiskCacheAPIElwVar, SetLocationWhenDisabled )
{
    setElwVar( "OPTIX_CACHE_MAXSIZE", "0" );
    ASSERT_NO_THROW( m_context = optix::Context::create() );

    m_location = corelib::getLwrrentDir() + "/test_diskCacheAPI_tmp";
    std::replace( m_location.begin(), m_location.end(), '\\', '/' );

    std::string locationOut;
    ASSERT_NO_THROW( m_context->setDiskCacheLocation( m_location ) );
    ASSERT_NO_THROW( locationOut = m_context->getDiskCacheLocation() );
    ASSERT_EQ( "", locationOut );

    setElwVar( "OPTIX_CACHE_MAXSIZE", "" );
}

TEST_F( DiskCacheAPIElwVar, SetLimitsWhenDisabled )
{
    setElwVar( "OPTIX_CACHE_MAXSIZE", "0" );
    ASSERT_NO_THROW( m_context = optix::Context::create() );
    
    RTsize lowLimitIn   = 0;
    RTsize highLimitIn  = 0;
    RTsize lowLimitOut  = 0;
    RTsize highLimitOut = 0;
    ASSERT_NO_THROW( m_context->setDiskCacheMemoryLimits( lowLimitIn, highLimitIn ) );
    ASSERT_NO_THROW( m_context->getDiskCacheMemoryLimits( lowLimitOut, highLimitOut ) );
    ASSERT_EQ( lowLimitIn, lowLimitOut );
    ASSERT_EQ( highLimitIn, highLimitOut );

    setElwVar( "OPTIX_CACHE_MAXSIZE", "" );
}
