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

#include <optix.h>
#include <optix_stubs.h>

#include <optix_ext_knobs_function_table_definition.h>
#include <optix_ext_knobs_stubs.h>
//#include <optix_function_table_definition.h>

#include "CommonAsserts.h"
#include <exptest/exptest.h>
#include <exptest/loggingHelpers.h>
#include <exptest/testTools.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <rtcoreVersionForDevice.h>

#include <src/Util/LWMLWrapper.h>

#include <lwda_runtime.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>

using namespace testing;

// Common class, do basic Context initialization
template <typename BASE>
struct DeviceContextConstruction : BASE
{
    void SetUp() override
    {
        exptest::lwdaInitialize();
        OPTIX_CHECK( optixInit() );

        LWcontext lwCtx = 0;  // zero means take the current context
        ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( lwCtx, 0, &m_context ) );
        ASSERT_NE( m_context, nullptr );
        ASSERT_OPTIX_SUCCESS( optixDeviceContextSetLogCallback( m_context, &OptixRecordingLogger::callback, &m_logger, 2 ) );
    }
    void TearDown() override {}

    OptixDeviceContext   m_context;
    OptixRecordingLogger m_logger;
};


template <typename BASE>
struct DeviceContextWithCleanupBase : DeviceContextConstruction<BASE>
{
    void TearDown() override
    {
        if( this->m_context )
        {
            ASSERT_OPTIX_SUCCESS( optixDeviceContextDestroy( this->m_context ) );
        }
    }
};


// Common base class for majority of test fixture classes.
struct DeviceContextWithCleanup : DeviceContextWithCleanupBase<Test>
{
};


//---------------------------------------------------------------------------
// optixDeviceContextCreate
//---------------------------------------------------------------------------

// Try to run these tests first.  If not, then they will pick up previous LWCA contexts.
TEST( O7_API_optixDeviceContextCreate, CreateWithNoLwrrentContextFAILS )
{
    lwCtxSetLwrrent( NULL );  // Ensure LWCA context not init. from prev tests.
    OPTIX_CHECK( optixInit() );
    OptixDeviceContext context;    // DON'T Initialize LWCA
    LWcontext          lwCtx = 0;  // zero means take the current context

    OptixResult optixRes = optixDeviceContextCreate( lwCtx, 0, &context );

    ASSERT_EQ( OPTIX_ERROR_LWDA_NOT_INITIALIZED, optixRes );
}

// This doesn't always fail if there was a valid LWCA context before running this test.
TEST( O7_API_optixDeviceContextCreate, CreateWithNoLwrrentContextAfterDeviceResetFAILS )
{
    lwCtxSetLwrrent( NULL );  // Ensure LWCA context not init. from prev tests.
    LWDA_CHECK( lwdaDeviceReset() );
    OPTIX_CHECK( optixInit() );
    OptixDeviceContext context;    // DON'T Initialize LWCA
    LWcontext          lwCtx = 0;  // zero means take the current context

    OptixResult optixRes = optixDeviceContextCreate( lwCtx, 0, &context );

    ASSERT_EQ( OPTIX_ERROR_LWDA_NOT_INITIALIZED, optixRes );
}

TEST( O7_API_optixDeviceContextCreate, CreateWithLwdaContext )
{
    exptest::lwdaInitialize();
    OPTIX_CHECK( optixInit() );

    LWcontext lwCtx;
    ASSERT_EQ( LWDA_SUCCESS, lwCtxGetLwrrent( &lwCtx ) );

    OptixDeviceContext context;

    ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( lwCtx, 0, &context ) );
    ASSERT_NE( context, nullptr );
    ASSERT_OPTIX_SUCCESS( optixDeviceContextDestroy( context ) );
}

TEST( O7_API_optixDeviceContextCreate, CreateWithLwrrentContext )
{
    exptest::lwdaInitialize();
    OPTIX_CHECK( optixInit() );

    OptixDeviceContext context;
    LWcontext          lwCtx = 0;  // zero means take the current context
    ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( lwCtx, 0, &context ) );
    ASSERT_NE( context, nullptr );
    ASSERT_OPTIX_SUCCESS( optixDeviceContextDestroy( context ) );
}

TEST( O7_API_optixDeviceContextCreate, CreateWithEmptyOptions )
{
    exptest::lwdaInitialize();
    OPTIX_CHECK( optixInit() );

    OptixDeviceContext context;
    LWcontext          lwCtx = 0;  // zero means take the current context

    OptixDeviceContextOptions options = {};
    ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( lwCtx, &options, &context ) );
    ASSERT_NE( context, nullptr );
    ASSERT_OPTIX_SUCCESS( optixDeviceContextDestroy( context ) );
}

TEST( O7_API_optixDeviceContextCreate, CreateWithLogger )
{
    exptest::lwdaInitialize();
    ASSERT_OPTIX_SUCCESS( optixInit() );

    OptixDeviceContext        context;
    OptixLogger               logger( std::cerr );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &OptixLogger::callback;
    options.logCallbackData           = &logger;
    options.logCallbackLevel          = 4;
    LWcontext lwCtx                   = 0;  // zero means take the current context

    ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( lwCtx, &options, &context ) );
    ASSERT_NE( context, nullptr );
    ASSERT_OPTIX_SUCCESS( optixDeviceContextDestroy( context ) );
}


TEST( O7_API_optixDeviceContextCreate, CreateWithIlwalidLogger )
{
    exptest::lwdaInitialize();
    ASSERT_OPTIX_SUCCESS( optixInit() );

    OptixDeviceContext        context = {};
    OptixLogger               logger( std::cerr );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &OptixLogger::callback;
    options.logCallbackData           = &logger;
    options.logCallbackLevel          = 5;  // == LOG_LEVEL::Invalid;
    LWcontext lwCtx                   = 0;  // zero means take the current context

    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixDeviceContextCreate( lwCtx, &options, &context ) );
}


TEST( O7_API_optixDeviceContextCreate, RunWithIlwalidKnob )
{
    exptest::lwdaInitialize();

    void* handle;
    ASSERT_OPTIX_SUCCESS( optixInitWithHandle( &handle ) );
    ASSERT_OPTIX_SUCCESS( optixExtKnobsInit( handle ) );
    // this is neccessary as OptiX keeps unfortunately static data even between multiple program runs
    // If this test runs first, the knob setting goes through and returns ilwalid_value. If not run first,
    // it does not run through and returns ilwalid_operation as the knob registry is already finalized.
    OptixResult res = optixExtKnobsSetKnob( "foo", "bar" );
    ASSERT_TRUE( res == OPTIX_ERROR_ILWALID_VALUE || res == OPTIX_ERROR_ILWALID_OPERATION );

    OptixDeviceContext context;
    //OptixLogger               logger( std::cerr );
    OptixRecordingLogger      logger;
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &OptixRecordingLogger::callback;
    options.logCallbackData           = &logger;
    options.logCallbackLevel          = 4;
    LWcontext lwCtx                   = 0;  // zero means take the current context

    ASSERT_OPTIX_SUCCESS( optixDeviceContextCreate( lwCtx, &options, &context ) );
    // see aformentioned comment - this test case is only valid if run first/stand-alone
    if( res == OPTIX_ERROR_ILWALID_VALUE )
    {
        EXPECT_THAT( logger.getMessagesAsOneString().c_str(), testing::HasSubstr( "Knob \"foo\" does not exist" ) );
    }
    ASSERT_OPTIX_SUCCESS( optixDeviceContextDestroy( context ) );
}


//---------------------------------------------------------------------------
// optixDeviceContextDestroy
//---------------------------------------------------------------------------

struct O7_API_optixDeviceContextDestroy : DeviceContextConstruction<Test>
{
};


TEST_F( O7_API_optixDeviceContextDestroy, RunWithSuccess )
{
    ASSERT_OPTIX_SUCCESS( optixDeviceContextDestroy( m_context ) );
}


// This test is disabled because it segfaults. Not nice.
TEST_F( O7_API_optixDeviceContextDestroy, DISABLED_DestroyNotCreatedFakePtr )
{
    OptixDeviceContext context  = {( OptixDeviceContext )( -1 )};
    OptixResult        optixRes = optixDeviceContextDestroy( context );
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixRes );
}


TEST_F( O7_API_optixDeviceContextDestroy, DestroyNullptrContext )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT, optixDeviceContextDestroy( nullptr ) );
}


// Tests of DeviceContext::destroy() regarding pipeline, program group, module, and denoiser will
// be covered in the dedicated tests, e.g. in O7_API_optixModuleCreate::TearDown() etc


//---------------------------------------------------------------------------
// optixDeviceContextGetProperty
//---------------------------------------------------------------------------

// Unfortunately we cannot reuse DeviceContextWithCleanup easily - multiple inheritance from Test (as
// TestWithParam inherits from Test itself) would lead to ambiguities inside gtest code. Templates to
// the rescue? Yes, providing one level of indirection via DeviceContextWithCleanupBase allows the code sharing.
struct O7_API_optixDeviceContextGetProperty : DeviceContextWithCleanupBase<TestWithParam<OptixDeviceProperty>>
{
};


TEST_P( O7_API_optixDeviceContextGetProperty, NoContext )
{
    unsigned int value;
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT, optixDeviceContextGetProperty( 0, GetParam(), &value, sizeof( value ) ) );
}

TEST_P( O7_API_optixDeviceContextGetProperty, NullptrValue )
{
    OptixResult optixRes = optixDeviceContextGetProperty( m_context, GetParam(), 0, 0 );
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixRes );
}

TEST_P( O7_API_optixDeviceContextGetProperty, ZeroSize )
{
    int         value;
    OptixResult optixRes = optixDeviceContextGetProperty( m_context, GetParam(), &value, 0 );
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixRes );
}

static bool getLwrrentDevicePciId( const unsigned int pciBusId, std::string& pciId )
{
    optix::LWML::Wrapper lwml;
    try
    {
        lwml.load();
        unsigned int  lwmlDeviceCount;
        lwmlDevice_t  device;
        lwmlPciInfo_t pci;
        if( lwml.deviceGetCount( &lwmlDeviceCount ) != LWML_SUCCESS )
            return false;
        for( int i = 0; i < lwmlDeviceCount; ++i )
        {
            if( lwml.deviceGetHandleByIndex( i, &device ) != LWML_SUCCESS || lwml.deviceGetPciInfo( device, &pci ) != LWML_SUCCESS )
                return false;
            if( pci.bus == pciBusId )
            {
                std::stringstream ss;
                ss << std::hex << ( pci.pciDeviceId );
                std::string pciDeviceIdString = ss.str();
                pciId = pciDeviceIdString.substr( 0, 4 ); // Device ID is first 4 chars of pciDeviceId
                return true;
            }
        }
        lwml.unload();
    }
    catch( const std::exception& e )
    {
        std::cerr << e.what();
        return false;
    }
    return true;
}

TEST_P( O7_API_optixDeviceContextGetProperty, GetValue )
{
	const int preTuringRTCoreVersion = 0;
    unsigned int value;
    OptixResult  optixRes = optixDeviceContextGetProperty( m_context, GetParam(), &value, sizeof( value ) );
    // which is equal to poor man's OPTIX_DEVICE_PROPERTY_UNDEF
    if( GetParam() == ( OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH - 1 ) )
        ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixRes );
    else
        ASSERT_EQ( OPTIX_SUCCESS, optixRes );

    // The values must match the limits in the optix7 documentation.
    switch( GetParam() )
    {
    case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH:
    {
        EXPECT_EQ( value, 31 );
    } break;
    case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH:
    {
        EXPECT_EQ( value, 31 );
    } break;
    case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS:
    {
        EXPECT_EQ( value, (1 << 29) );
    } break;
    case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS:
    {
        EXPECT_EQ( value, (1 << 28) );
    } break;
    case OPTIX_DEVICE_PROPERTY_RTCORE_VERSION:
    {
    	lwdaDeviceProp prop;
    	int lwrrentDevice;
    	LWDA_CHECK( lwdaGetDevice( &lwrrentDevice ) );
        LWDA_CHECK( lwdaGetDeviceProperties( &prop, lwrrentDevice ) );

		std::string pciId;
		ASSERT_TRUE( getLwrrentDevicePciId( prop.pciBusID, pciId ) );

		// Devices with SM prior to 70 should report RTCore version 0.
		if( prop.major < 7 )
			EXPECT_EQ( value, preTuringRTCoreVersion );
    	// RTCore version will be 10 for devices where hasTTU == true (some but not all Turing).
        // RTCore version will be 20 for devices where hasMotionTTU == true (some but not all Ampere).
        // RTCore version will be 0 for any other Turing or later devices that do not have RTCores.
		else
		{
			// Dump this bit to help with debug on the test farm.
			const int rtcoreVersion = rtcoreVersionForDevice[pciId];
			std::cerr << "Device ID: " << pciId << " RTCore version found: " << rtcoreVersion << std::endl;
			EXPECT_EQ( value, rtcoreVersion ) << "Incorrect RTCore Version for device with ID: " << pciId << "\n";
		}
    } break;
    case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID:
    {
        EXPECT_EQ( value, (1 << 28) - 1 );
    } break;
    case OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK:
    {
        EXPECT_EQ( value, 8 );
    } break;
    case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS:
    {
        EXPECT_EQ( value, (1 << 28) );
    } break;
    case OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET:
    {
        EXPECT_EQ( value, (1 << 28) - 1 );
    } break;
    };
}

TEST_P( O7_API_optixDeviceContextGetProperty, GetValueOnULL )
{
    unsigned long long value_as_ull;
    OptixResult optixRes = optixDeviceContextGetProperty( m_context, GetParam(), &value_as_ull, sizeof( value_as_ull ) );
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixRes );
}

void PrintTo( OptixDeviceProperty value, std::ostream* str )
{
    *str << toString( value );
}

INSTANTIATE_TEST_SUITE_P( O7_API,
                          O7_API_optixDeviceContextGetProperty,
                          ::testing::Values( OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH,
                                             OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH,
                                             OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS,
                                             OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS,
                                             OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID,
                                             OPTIX_DEVICE_PROPERTY_LIMIT_NUM_BITS_INSTANCE_VISIBILITY_MASK,
                                             OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,
                                             OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_RECORDS_PER_GAS,
                                             OPTIX_DEVICE_PROPERTY_LIMIT_MAX_SBT_OFFSET,
                                             // poor man's OPTIX_DEVICE_PROPERTY_UNDEF
                                             OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRACE_DEPTH - 1 ) );

//---------------------------------------------------------------------------
// optixDeviceContextSetLogCallback
//---------------------------------------------------------------------------

struct O7_API_optixDeviceContextSetLogCallback : DeviceContextWithCleanup
{
};

TEST_F( O7_API_optixDeviceContextSetLogCallback, SetAllZeros )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT, optixDeviceContextSetLogCallback( 0, 0, 0, 0 ) );
}

TEST_F( O7_API_optixDeviceContextSetLogCallback, SetZerosValidContext )
{
    OptixResult optixRes = optixDeviceContextSetLogCallback( m_context, 0, 0, 0 );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );
}

TEST_F( O7_API_optixDeviceContextSetLogCallback, SetFunctionNoData )
{
    OptixResult optixRes = optixDeviceContextSetLogCallback( m_context, &OptixLogger::callback, 0, 0 );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );
}

TEST_F( O7_API_optixDeviceContextSetLogCallback, SetLoggerIlwalidLevel )
{
    OptixResult optixRes = optixDeviceContextSetLogCallback( m_context, &OptixLogger::callback, &m_logger, -1 );
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixRes );
}

TEST_F( O7_API_optixDeviceContextSetLogCallback, SetLoggerLevelTooHigh )
{
    OptixResult optixRes = optixDeviceContextSetLogCallback( m_context, &OptixLogger::callback, &m_logger, 100 );
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixRes );
}

TEST_F( O7_API_optixDeviceContextSetLogCallback, SetLoggerLevel0 )
{
    OptixResult optixRes = optixDeviceContextSetLogCallback( m_context, &OptixLogger::callback, &m_logger, 0 );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );
}

TEST_F( O7_API_optixDeviceContextSetLogCallback, SetLoggerLevel4 )
{
    OptixResult optixRes = optixDeviceContextSetLogCallback( m_context, &OptixLogger::callback, &m_logger, 4 );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );
}

//---------------------------------------------------------------------------
// optixDeviceContextGetCacheEnabled
//---------------------------------------------------------------------------

struct O7_API_optixDeviceContextGetCacheEnabled : DeviceContextWithCleanup
{
};

TEST_F( O7_API_optixDeviceContextGetCacheEnabled, SetAllZeros )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT, optixDeviceContextGetCacheEnabled( 0, 0 ) );
}

TEST_F( O7_API_optixDeviceContextGetCacheEnabled, Nullptr )
{
    OptixResult optixRes = optixDeviceContextGetCacheEnabled( m_context, 0 );
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixRes );
}

TEST_F( O7_API_optixDeviceContextGetCacheEnabled, GetValue )
{
    int         value    = -1;
    OptixResult optixRes = optixDeviceContextGetCacheEnabled( m_context, &value );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );
    ASSERT_LE( 0, value );
    ASSERT_GE( 1, value );
}

//---------------------------------------------------------------------------
// optixDeviceContextSetCacheEnabled
//---------------------------------------------------------------------------

struct O7_API_optixDeviceContextSetCacheEnabled : DeviceContextWithCleanup
{
};

TEST_F( O7_API_optixDeviceContextSetCacheEnabled, SetAllZeros )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT, optixDeviceContextSetCacheEnabled( 0, 0 ) );
}

TEST_F( O7_API_optixDeviceContextSetCacheEnabled, DisableCache )
{
    OptixResult optixRes;
    int         new_value;

    optixRes = optixDeviceContextSetCacheEnabled( m_context, 0 );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );

    optixRes = optixDeviceContextGetCacheEnabled( m_context, &new_value );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );
    ASSERT_EQ( 0, new_value );
}

TEST_F( O7_API_optixDeviceContextSetCacheEnabled, EnableCache )
{
    OptixResult optixRes;
    int         new_value;

    optixRes = optixDeviceContextSetCacheEnabled( m_context, 1 );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );

    optixRes = optixDeviceContextGetCacheEnabled( m_context, &new_value );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );
    ASSERT_EQ( 1, new_value );
}

TEST_F( O7_API_optixDeviceContextSetCacheEnabled, ToggleValue )
{
    OptixResult optixRes;
    int         old_value, new_value;
    optixRes = optixDeviceContextGetCacheEnabled( m_context, &old_value );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );

    optixRes = optixDeviceContextSetCacheEnabled( m_context, ( old_value == 0 ) ? 1 : 0 );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );

    optixRes = optixDeviceContextGetCacheEnabled( m_context, &new_value );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );
    ASSERT_EQ( ( old_value == 0 ) ? 1 : 0, new_value );
}

TEST_F( O7_API_optixDeviceContextSetCacheEnabled, ValueTooLow )
{
    OptixResult optixRes = optixDeviceContextSetCacheEnabled( m_context, -1 );
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixRes );
}

TEST_F( O7_API_optixDeviceContextSetCacheEnabled, ValueTooHigh )
{
    OptixResult optixRes = optixDeviceContextSetCacheEnabled( m_context, 2 );
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixRes );
}

//---------------------------------------------------------------------------
// optixDeviceContextGetCacheLocation
//---------------------------------------------------------------------------

struct O7_API_optixDeviceContextGetCacheLocation : DeviceContextWithCleanup
{
};

TEST_F( O7_API_optixDeviceContextGetCacheLocation, SetAllZeros )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT, optixDeviceContextGetCacheLocation( 0, 0, 0 ) );
}

TEST_F( O7_API_optixDeviceContextGetCacheLocation, NullContext )
{
    const int STRING_LENGTH = 256;
    char      s[STRING_LENGTH];
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT, optixDeviceContextGetCacheLocation( nullptr, s, STRING_LENGTH ) );
}

TEST_F( O7_API_optixDeviceContextGetCacheLocation, NullLocation )
{
    const int   STRING_LENGTH = 256;
    OptixResult optixRes      = optixDeviceContextGetCacheLocation( m_context, 0, STRING_LENGTH );
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixRes );
}

TEST_F( O7_API_optixDeviceContextGetCacheLocation, NullptrLocation )
{
    OptixResult optixRes = optixDeviceContextGetCacheLocation( m_context, nullptr, 0 );
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixRes );
}

TEST_F( O7_API_optixDeviceContextGetCacheLocation, ZeroLocationSize )
{
    const int STRING_LENGTH = 256;
    char      s[STRING_LENGTH];
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixDeviceContextGetCacheLocation( m_context, s, 0 ) );
}

TEST_F( O7_API_optixDeviceContextGetCacheLocation, SmallString8C )
{
    const int   STRING_LENGTH = 8;
    char        s[STRING_LENGTH];
    OptixResult optixRes = optixDeviceContextGetCacheLocation( m_context, s, STRING_LENGTH );
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixRes );
}

TEST_F( O7_API_optixDeviceContextGetCacheLocation, LongString256C )
{
    const int   STRING_LENGTH = 256;
    char        s[STRING_LENGTH];
    OptixResult optixRes = optixDeviceContextGetCacheLocation( m_context, s, STRING_LENGTH );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );
}

// I would expect this to segfault.
TEST_F( O7_API_optixDeviceContextGetCacheLocation, DISABLED_SizeLongerThanActualSize )
{
    const int   STRING_LENGTH = 16;
    const int   WRONG_LENGTH  = 256;
    char        s[STRING_LENGTH];
    OptixResult optixRes = optixDeviceContextGetCacheLocation( m_context, s, WRONG_LENGTH );
    ASSERT_EQ( OPTIX_ERROR_NOT_SUPPORTED, optixRes );
}

//---------------------------------------------------------------------------
// optixDeviceContextSetCacheLocation
//---------------------------------------------------------------------------

struct O7_API_optixDeviceContextSetCacheLocation : DeviceContextWithCleanup
{
};

TEST_F( O7_API_optixDeviceContextSetCacheLocation, SetAllZeros )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT, optixDeviceContextSetCacheLocation( 0, 0 ) );
}

TEST_F( O7_API_optixDeviceContextSetCacheLocation, NullptrLocation )
{
    OptixResult optixRes = optixDeviceContextSetCacheLocation( m_context, 0 );
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixRes );
}

TEST_F( O7_API_optixDeviceContextSetCacheLocation, SetEmptyString )
{
    const char* empty    = "";
    OptixResult optixRes = optixDeviceContextSetCacheLocation( m_context, empty );
    ASSERT_EQ( OPTIX_ERROR_DISK_CACHE_ILWALID_PATH, optixRes );
}

TEST_F( O7_API_optixDeviceContextSetCacheLocation, SetAndVerify )
{
    const char* new_location  = "testpath";
    const int   STRING_LENGTH = 256;
    char        get_location[STRING_LENGTH];
    OptixResult optixRes;
    optixRes = optixDeviceContextSetCacheLocation( m_context, new_location );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );
    optixRes = optixDeviceContextGetCacheLocation( m_context, get_location, STRING_LENGTH );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );
    ASSERT_STREQ( get_location, new_location );
}

//---------------------------------------------------------------------------
// optixDeviceContextGetCacheDatabaseSizes
//---------------------------------------------------------------------------

struct O7_API_optixDeviceContextGetCacheDatabaseSizes : DeviceContextWithCleanup
{
};

TEST_F( O7_API_optixDeviceContextGetCacheDatabaseSizes, SetAllZeros )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT, optixDeviceContextGetCacheDatabaseSizes( 0, 0, 0 ) );
}

TEST_F( O7_API_optixDeviceContextGetCacheDatabaseSizes, LowHighBothNullptr )
{
    OptixResult optixRes = optixDeviceContextGetCacheDatabaseSizes( m_context, 0, 0 );
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixRes );
}


//---------------------------------------------------------------------------
// optixDeviceContextSetCacheDatabaseSizes
//---------------------------------------------------------------------------

struct O7_API_optixDeviceContextSetCacheDatabaseSizes : DeviceContextWithCleanup
{
};

TEST_F( O7_API_optixDeviceContextSetCacheDatabaseSizes, SetAllZeros )
{
    ASSERT_EQ( OPTIX_ERROR_ILWALID_DEVICE_CONTEXT, optixDeviceContextSetCacheDatabaseSizes( 0, 0, 0 ) );
}

TEST_F( O7_API_optixDeviceContextSetCacheDatabaseSizes, LowHighBothZero )
{
    OptixResult optixRes = optixDeviceContextSetCacheDatabaseSizes( m_context, 0, 0 );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );
}

TEST_F( O7_API_optixDeviceContextSetCacheDatabaseSizes, SetAndVerify )
{
    const size_t WATERMARK = 100;
    size_t       low_watermark, high_watermark;
    OptixResult  optixRes;
    optixRes = optixDeviceContextSetCacheDatabaseSizes( m_context, WATERMARK, WATERMARK + 1 );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );
    optixRes = optixDeviceContextGetCacheDatabaseSizes( m_context, &low_watermark, &high_watermark );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );
    ASSERT_EQ( WATERMARK, low_watermark );
    ASSERT_EQ( WATERMARK + 1, high_watermark );
}

// if both low and high watermarks are set to the same value, is it correct?
TEST_F( O7_API_optixDeviceContextSetCacheDatabaseSizes, LowHighEqual )
{
    const size_t WATERMARK = 100;
    OptixResult  optixRes  = optixDeviceContextSetCacheDatabaseSizes( m_context, WATERMARK, WATERMARK );
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );
}

TEST_F( O7_API_optixDeviceContextSetCacheDatabaseSizes, LowHigherThanHigh )
{
    const size_t WATERMARK = 100;
    OptixResult  optixRes  = optixDeviceContextSetCacheDatabaseSizes( m_context, WATERMARK + 1, WATERMARK );
    ASSERT_EQ( OPTIX_ERROR_ILWALID_VALUE, optixRes );
}

TEST_F( O7_API_optixDeviceContextSetCacheDatabaseSizes, LowHigherThanHighWhichIsZero )
{
    const size_t WATERMARK = 0;
    // this triggers an otherwise unreachable check result in DeviceContext::setDiskCacheMemoryLimits()
    OptixResult optixRes = optixDeviceContextSetCacheDatabaseSizes( m_context, WATERMARK + 1, WATERMARK );
    // this works while setting either one limit to zero disables garbage collection
    ASSERT_EQ( OPTIX_SUCCESS, optixRes );
}
