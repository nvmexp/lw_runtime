//
//  Copyright (c) 2019 LWPU Corporation.  All rights reserved.
//
//  LWPU Corporation and its licensors retain all intellectual property and proprietary
//  rights in and to this software, related documentation and any modifications thereto.
//  Any use, reproduction, disclosure or distribution of this software and related
//  documentation without an express license agreement from LWPU Corporation is strictly
//  prohibited.
//
//  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
//  AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
//  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
//  PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
//  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
//  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
//  BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
//  INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
//  SUCH DAMAGES
//
//
//

#include <exptest/testTools.h>

#include <srcTestsConfig.h>

#if !defined( _WIN32 )
#include <dirent.h>
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <windows.h>
#endif

#include <fstream>
#include <list>
#include <mutex>

// In order to return const char*, we will hold all the strings in a list to maintain the memory.
// The lock is to prevent bad things from happening in multi-threaded elwironments.
std::mutex             s_stringLock;
std::list<std::string> s_strings;

static bool dirExists( const char* path )
{
#if defined( _WIN32 )
    DWORD attrib = GetFileAttributes( path );
    return ( attrib != ILWALID_FILE_ATTRIBUTES ) && ( attrib & FILE_ATTRIBUTE_DIRECTORY );
#else
    DIR* dir = opendir( path );
    if( dir == NULL )
        return false;
    else
    {
        closedir( dir );
        return true;
    }
#endif
}

namespace exptest {

const char* ptxPath( const std::string& target, const std::string& base )
{
    std::string path;

    // Allow for overrides.
    if( const char* dir = getelw( "OPTIX_TEST_PTX_DIR" ) )
    {
        path = dir;
    }
    else if( dirExists( SRC_TESTS_PTX_DIR ) )
    {
        // Return hardcoded path if it exists.
        path = std::string( SRC_TESTS_PTX_DIR );
    }
    else
    {
        // Last resort.
        path = ".";
    }

    path += "/" + target + "_generated_" + base + ".ptx";

    std::lock_guard<std::mutex> lock( s_stringLock );
    s_strings.push_back( path );
    return s_strings.back().c_str();
}

const char* dataPath()
{
    // Allow for overrides. In the QA setup, OPTIX_TEST_DATA_DIR points to "bin/data" inside test.zip.
    if( const char* dir = getelw( "OPTIX_TEST_DATA_DIR" ) )
        return dir;

    // Return hardcoded path if it exists. SRC_TESTS_DATA_DIR points to "tests/data" in the source tree.
    if( dirExists( SRC_TESTS_DATA_DIR ) )
        return SRC_TESTS_DATA_DIR;

    // Last resort.
    return ".";
}

const char* goldPath( const std::string& testDirectory, const std::string& fileName )
{
    std::string path;

    // Allow for overrides. In the QA setup, OPTIX_TEST_DATA_DIR points to "bin/data" inside test.zip.
    if( const char* dir = getelw( "OPTIX_TEST_DATA_DIR" ) )
    {
        path = std::string( dir ) + "/../../tests/Unit_exp/" + testDirectory;
    }
    else if( dirExists( SRC_TESTS_DATA_DIR ) )
    {
        // Return hardcoded path if it exists. SRC_TESTS_DATA_DIR points to "tests/data" in the source tree.
        path = std::string( SRC_TESTS_DATA_DIR ) + "/../Unit_exp/" + testDirectory;
    }
    else
    {
        // Last resort.
        path = ".";
    }

    path += "/" + fileName;

    std::lock_guard<std::mutex> lock( s_stringLock );
    s_strings.push_back( path );
    return s_strings.back().c_str();
}

const char* readPTX( const char* testName, const char* fileName )
{
    std::string ptx_path( ptxPath( testName, fileName ) );

    std::ifstream inputPtx( ptx_path );
    EXPECT_FALSE( !inputPtx );

    std::stringstream ptx;
    ptx << inputPtx.rdbuf();
    EXPECT_FALSE( inputPtx.fail() );

    std::lock_guard<std::mutex> lock( s_stringLock );
    s_strings.push_back( ptx.str() );
    return s_strings.back().c_str();
}
}
