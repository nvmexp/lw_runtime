// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <gmock/gmock.h>

#include <srcTestConfig.h>

#include <shared/Exceptions/AlreadyMapped.h>
#include <shared/Util/Knobs.h>
#include <shared/Util/Preprocessor.h>
#include <shared/Util/Timer.h>

#include <Control/ErrorManager.h>
#include <Objects/Context.h>
#include <c-api/rtapi.h>

#include <string>

using namespace optix;
using namespace prodlib;
using namespace testing;


TEST( ErrorManager, returnDescriptiveStringWithErrorCode )
{
    std::string str;

    str = ErrorManager::getErrorString_static( RT_SUCCESS );
    ASSERT_EQ( str, std::string( "Success (no errors)" ) );

    str = ErrorManager::getErrorString_static( RT_TIMEOUT_CALLBACK );
    ASSERT_EQ( str, std::string( "Application's timeout callback requested the API call to terminate unfinished" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_CONTEXT_CREATION_FAILED );
    ASSERT_EQ( str, std::string( "Context creation failed" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_ILWALID_CONTEXT );
    ASSERT_EQ( str, std::string( "Invalid context" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_ILWALID_VALUE );
    ASSERT_EQ( str, std::string( "Invalid value" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_MEMORY_ALLOCATION_FAILED );
    ASSERT_EQ( str, std::string( "Memory allocation failed" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_TYPE_MISMATCH );
    ASSERT_EQ( str, std::string( "Type mismatch" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_VARIABLE_NOT_FOUND );
    ASSERT_EQ( str, std::string( "Variable not found" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_VARIABLE_REDECLARED );
    ASSERT_EQ( str, std::string( "Variable redeclared" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_ILLEGAL_SYMBOL );
    ASSERT_EQ( str, std::string( "Illegal symbol" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_ILWALID_SOURCE );
    ASSERT_EQ( str, std::string( "Parse error" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_VERSION_MISMATCH );
    ASSERT_EQ( str, std::string( "Version mismatch" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_OBJECT_CREATION_FAILED );
    ASSERT_EQ( str, std::string( "Object creation failed" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_LAUNCH_FAILED );
    ASSERT_EQ( str, std::string( "Launch failed" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_NO_DEVICE );
    ASSERT_EQ( str, std::string( "A supported LWPU GPU could not be found" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_ILWALID_DEVICE );
    ASSERT_EQ( str, std::string( "Invalid device" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_ILWALID_IMAGE );
    ASSERT_EQ( str, std::string( "Invalid image" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_FILE_NOT_FOUND );
    ASSERT_EQ( str, std::string( "File not found" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_ALREADY_MAPPED );
    ASSERT_EQ( str, std::string( "Already mapped" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_ILWALID_DRIVER_VERSION );
    ASSERT_EQ( str,
               std::string( "A supported LWPU driver cannot be found. Please see the release notes for supported "
                            "drivers." ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_RESOURCE_NOT_REGISTERED );
    ASSERT_EQ( str, std::string( "An OptiX resource was not registered properly" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_RESOURCE_ALREADY_REGISTERED );
    ASSERT_EQ( str, std::string( "An OptiX resource has already been registered" ) );

    str = ErrorManager::getErrorString_static( RT_ERROR_UNKNOWN );
    ASSERT_EQ( str, std::string( "Unknown error" ) );

    str = ErrorManager::getErrorString_static( (RTresult)0x1223556 );
    ASSERT_EQ( str, std::string( "Unknown error" ) );

    str = ErrorManager::getErrorString_static( (RTresult)-0x1223556 );
    ASSERT_EQ( str, std::string( "Unknown error" ) );
}

TEST( ErrorManager, storesErrorFromRtResultCode )
{
    ErrorManager errorManager = ErrorManager();
    RTresult     result       = RT_ERROR_ALREADY_MAPPED;
    std::string  error        = errorManager.getErrorString_static( result );

    errorManager.setErrorString( ( RTAPI_FUNC ), error, result );

    ASSERT_EQ( errorManager.getErrorString( result ), std::string( "Already mapped (Details: Function "
                                                                   "\"ErrorManager_storesErrorFromRtResultCode_Test::"
                                                                   "TestBody\" detected error: Already mapped)" ) );
}

TEST( ErrorManager, DISABLED_storesErrorFromOptixException )
{
    ErrorManager  errorManager  = ErrorManager();
    RTresult      result        = RT_ERROR_ALREADY_MAPPED;
    ExceptionInfo exceptionInfo = ExceptionInfo( "test.file", 1 );
    AlreadyMapped exception     = AlreadyMapped( exceptionInfo, "Buffer is already mapped." );

    errorManager.setErrorString( ( RTAPI_FUNC ), exception );

    ASSERT_EQ( errorManager.getErrorString( result ).substr( 0, 176 ),
               std::string( "Already mapped (Details: Function "
                            "\"ErrorManager_storesErrorFromOptixException_Test::TestBody\" caught exception: Buffer is "
                            "already mapped., file:test.file(1) id: [1]\n===========" ) );
}

TEST( ErrorManager, storesErrorFromStdException )
{
    ErrorManager   errorManager = ErrorManager();
    RTresult       result       = RT_ERROR_UNKNOWN;
    std::exception e;
    std::string    error = errorManager.getErrorString_static( result );

    errorManager.setErrorString( ( RTAPI_FUNC ), e );

    ASSERT_EQ( errorManager.getErrorString( result ), std::string( "Unknown error (Details: Function "
                                                                   "\"ErrorManager_storesErrorFromStdException_Test::"
                                                                   "TestBody\" caught C++ standard exception: "
                                                                   "Unknown exception)" ) );
}

TEST( ErrorManager, returnUnsavedAndSavedErrorCode )
{
    ErrorManager errorManager = ErrorManager();
    RTresult     savedResult  = RT_ERROR_ALREADY_MAPPED;
    std::string  error        = errorManager.getErrorString_static( savedResult );
    errorManager.setErrorString( ( RTAPI_FUNC ), error, savedResult );
    RTresult    unsavedResult = RT_ERROR_ILWALID_CONTEXT;
    std::string str;

    str = errorManager.getErrorString( unsavedResult );
    ASSERT_EQ( str, std::string( "Invalid context" ) );

    str = errorManager.getErrorString( savedResult );
    ASSERT_EQ( str,
               std::string( "Already mapped (Details: Function "
                            "\"ErrorManager_returnUnsavedAndSavedErrorCode_Test::TestBody\" detected error: Already "
                            "mapped)" ) );

    str = errorManager.getErrorString( (RTresult)0x1223556 );
    ASSERT_EQ( str, std::string( "Unknown error" ) );
}
