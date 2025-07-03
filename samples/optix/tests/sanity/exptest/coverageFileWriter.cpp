
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

#define OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
#define OPTIX_OPTIONAL_FEATURE_OPTIX7_LWRVES

#include <exptest/coverageFileWriter.h>
#include <private/optix_7_enum_printers.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>

using namespace std;

std::string CoverageFileWriter::s_outFileName;
std::string CoverageFileWriter::s_lwrrentTest;
bool        CoverageFileWriter::s_doReport;

void CoverageFileWriter::init( const std::string& fileName, const std::string& testName, bool withReport )
{
    CoverageFileWriter::s_outFileName = fileName;
    CoverageFileWriter::s_lwrrentTest = testName;
    CoverageFileWriter::s_doReport    = withReport;
}


void CoverageFileWriter::exit()
{
    getInstance().stopTest();
    getInstance().close();
}


std::string getLastLine( std::fstream& in )
{
    // while not exactly optimal, it should be sufficient for the file sizes the writer deals with
    string line;
    // skip empty lines
    while( in >> std::ws && std::getline( in, line ) )
        ;

    return line;
}

// clang-format off
CoverageFileWriter::CoverageFileWriter()
  : m_outFile{ s_outFileName, std::ios::out | std::ios::app }
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
  , m_coverageValuesGlobal{ 0 }
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
// clang-format on
{
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    // read last line's bitset as current global coverage state
    if( m_outFile )
    {
        // for getting read access, create an input file stream on the same file
        fstream     inFile{ s_outFileName, std::ios::in };
        std::string lastLine = getLastLine( inFile );
        // extract bitset
        if( lastLine.size() >= OPTIX_Undefined_COVERAGE_ID )
        {
            string lwrrentGlobalCoverage = lastLine.substr( 0, OPTIX_Undefined_COVERAGE_ID );
            // interpret string as bitset of '0's and '1's
            for( size_t i = 0; i < OPTIX_Undefined_COVERAGE_ID; ++i )
                if( lwrrentGlobalCoverage[i] == '0' )
                    m_coverageValuesGlobal[i] = 0;
                else
                    m_coverageValuesGlobal[i] = 1;
        }
    }
    else
        cerr << "No coverage file \"" << s_outFileName << "\" could be opened, no coverage data will be written" << endl;
#endif  // OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
}


CoverageFileWriter& CoverageFileWriter::getInstance()
{
    static CoverageFileWriter writer;
    return writer;
}


void CoverageFileWriter::write( char* covValues, const std::string& name )
{
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    if( !m_outFile )
        return;
    for( size_t i = 0; i < OPTIX_Undefined_COVERAGE_ID; ++i )
    {
        m_outFile << ( covValues[i] ? std::string( "1" ) : std::string( "0" ) );
    }
    m_outFile << "\t" << name << std::endl;
#endif  //OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
}


void CoverageFileWriter::stopTest()
{
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    // write local cov values
    write( m_coverageValuesLocal, s_lwrrentTest );
    // add to global values
    for( size_t i = 0; i < OPTIX_Undefined_COVERAGE_ID; ++i )
        if( m_coverageValuesLocal[i] )
            m_coverageValuesGlobal[i] = static_cast<char>( 1 );
#endif  //OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
}

void CoverageFileWriter::close()
{
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    // write collected global values, actually the union of all local ones
    write( m_coverageValuesGlobal, "Result" );
    if( s_doReport )
    {
        report();
        // write collected global values again such that following runs can read the last data
        write( m_coverageValuesGlobal, "Result - repeated" );
    }
    if( m_outFile )
        m_outFile.close();
#endif  //OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
}

void CoverageFileWriter::report()
{
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    if( !m_outFile )
        return;
    m_outFile << "Report of uncovered device API calls" << std::endl;
    for( size_t i = 0; i < OPTIX_Undefined_COVERAGE_ID; ++i )
        if( !m_coverageValuesGlobal[i] )
            m_outFile << "\t" << toString( static_cast<OptixDeviceAPICallCoverageID>( i ) ) << std::endl;
#endif  //OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
}

void CoverageFileWriter::addCoverage( char* covered )
{
#ifdef OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
    for( size_t i = 0; i < OPTIX_Undefined_COVERAGE_ID; ++i )
        if( covered[i] )
            m_coverageValuesLocal[i] = static_cast<char>( 1 );
#endif  //OPTIX_OPTIONAL_FEATURE_TEST_CALL_COVERAGE
}
