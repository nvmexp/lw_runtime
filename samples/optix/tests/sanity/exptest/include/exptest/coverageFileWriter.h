
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

#include <optix.h>

#include <fstream>
#include <string>

#include <exptest/exptestapi.h>

#define ADD_COVERAGE_RESULT( covered ) CoverageFileWriter::getInstance().addCoverage( covered );

#define INJECT_API_CALL_COVERAGE()                                                                                     \
    const char*             fileName = OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE_FILENAME;                             \
    const char*             testName = OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE_TESTNAME;                             \
    CoverageFileWriterGuard guard( fileName, testName );

#define SETUP_API_CALL_COVERAGE( ptr )                                                                                 \
    LWDA_CHECK( lwdaMalloc( (void**)&ptr, OPTIX_Undefined_COVERAGE_ID ) );                                               \
    LWDA_CHECK( lwdaMemset( (void*)ptr, 0, OPTIX_Undefined_COVERAGE_ID ) );

#define ANALYZE_API_CALL_COVERAGE( ptr )                                                                                         \
    char coveredDeviceAPIFunctions[OPTIX_Undefined_COVERAGE_ID] = { 0 };                                                           \
    LWDA_CHECK( lwdaMemcpy( (void*)coveredDeviceAPIFunctions, (void*)ptr, OPTIX_Undefined_COVERAGE_ID, lwdaMemcpyDeviceToHost ) ); \
    ADD_COVERAGE_RESULT( coveredDeviceAPIFunctions );                                                                            \
    LWDA_CHECK( lwdaFree( (void*)ptr ) );

// The writer - when enabled - keeps track of each test's coverage of the OptiX 7 device API calls. Each test can then use the
// macro definitions above and has to add a corresponding char* entry into the Params definition.
class CoverageFileWriter
{
  public:
    EXPTESTAPI static void init( const std::string& fileName, const std::string& testName, bool withReport );
    EXPTESTAPI static void exit();
    // Singleton.
    EXPTESTAPI static CoverageFileWriter& getInstance();

    EXPTESTAPI void stopTest();
    EXPTESTAPI void close();

    // Add coverage to local current coverage.
    EXPTESTAPI void addCoverage( char* covered );

  private:
    static std::string s_lwrrentTest;
    static std::string s_outFileName;
    static bool        s_doReport;
    std::fstream       m_outFile;
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    char m_coverageValuesGlobal[OPTIX_Undefined_COVERAGE_ID];
    char m_coverageValuesLocal[OPTIX_Undefined_COVERAGE_ID] = { 0 };
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE

    CoverageFileWriter();
    void write( char* covValues, const std::string& name );
    void report();
};

struct CoverageFileWriterGuard
{
    CoverageFileWriterGuard( const std::string& fileName, const std::string& testName, bool withReport = false )
    {
        CoverageFileWriter::init( fileName, testName, withReport );
    }
    ~CoverageFileWriterGuard() { CoverageFileWriter::exit(); }
};

#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
// Utility function to translate a OPTIX_COVERED_DEVICE_API_CALLS value into a name.
const char* getDeviceAPICallName( OptixDeviceAPICallCoverageID val );
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
