
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

#pragma once

#include <lwda_runtime.h>

#include <gtest/gtest.h>

#include <exptest/exptestapi.h>

#include <srcTestsConfig.h>

#include <optix.h>

#include <exception>
#include <sstream>
#include <string>

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

template <>
struct SbtRecord<void>
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

namespace exptest {

// Overloading on the first argument ensures the correct function is called from the macros below.

inline void check( OptixResult res, const char* call, const char* file, unsigned int line )
{
    if( res != OPTIX_SUCCESS )
    {
        FAIL() << "Optix call in " << file << ", line " << line << " (" << call << ") failed with code " << res;
    }
}

inline void check_throw( OptixResult res, const char* call, const char* file, unsigned int line )
{
    if( res != OPTIX_SUCCESS )
    {
        std::stringstream s;
        s << "Optix call in " << file << ", line " << line << " (" << call << ") failed with code " << res;
        throw std::runtime_error( s.str() );
    }
}

inline void check( lwdaError_t error, const char* call, const char* file, unsigned int line )
{
    if( error != lwdaSuccess )
    {
        FAIL() << "LWCA call in " << file << ", line " << line << " (" << call << ") failed with code " << error << ": "
               << lwdaGetErrorString( error );
    }
}

inline void check( LWresult error, const char* call, const char* file, unsigned int line )
{
    if( error != LWDA_SUCCESS )
    {
        const char* str;
        lwGetErrorString( error, &str );
        FAIL() << "LWCA call in " << file << ", line " << line << " (" << call << ") failed with code " << error << ": " << str;
    }
}

inline void syncCheck( const char* file, unsigned int line )
{
    lwdaDeviceSynchronize();
    const lwdaError_t error = lwdaGetLastError();
    if( error != lwdaSuccess )
    {
        FAIL() << "LWCA sync check in " << file << ": line " << line << " failed with code " << error << ": "
               << lwdaGetErrorString( error );
    }
}

#define OPTIX_CHECK( call ) exptest::check( call, #call, __FILE__, __LINE__ )
#define OPTIX_CHECK_THROW( call ) exptest::check_throw( call, #call, __FILE__, __LINE__ )
#define LWDA_CHECK( call ) exptest::check( call, #call, __FILE__, __LINE__ )
#define LWDA_SYNC_CHECK() exptest::syncCheck( __FILE__, __LINE__ )
#define LW_CHECK( call ) exptest::check( call, #call, __FILE__, __LINE__ )

inline void lwdaInitialize()
{
    LWDA_CHECK( lwdaFree( 0 ) );
}

EXPTESTAPI const char* ptxPath( const std::string& target, const std::string& base );
EXPTESTAPI const char* dataPath();
EXPTESTAPI const char* goldPath( const std::string& testDirectory, const std::string& fileName );
EXPTESTAPI const char* readPTX( const char* testName, const char* fileName );

}  // namespace exptest
