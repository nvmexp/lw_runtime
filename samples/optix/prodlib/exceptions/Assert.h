// Copyright LWPU Corporation 2008
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

#include <corelib/system/Preprocessor.h>

#include <prodlib/exceptions/AssertionFailure.h>
#include <prodlib/system/Logger.h>

#include <exception>
#include <string>

// Runtime assertion.
namespace prodlib {

// This function exists purely to provide a place for you to put a breakpoint to catch all
// failed assertions thrown by any of the macros in this file.
inline void assertFailureDebugHook()
{
}

inline void assertCondition( const char* file, unsigned int line, const char* expr )
{
    assertFailureDebugHook();
    throw AssertionFailure( ExceptionInfo( file, line, true ), expr );
}

inline void assertMessage( const char* file, unsigned int line, const char* expr, const std::string& msg )
{
    assertFailureDebugHook();
    throw AssertionFailure( ExceptionInfo( file, line, true ), std::string( expr ) + " : " + msg );
}

#if defined( DEBUG ) || defined( DEVELOP )
inline void assertNoThrow( const char* file, unsigned int line, const char* expr, const char* msg )
{
    assertFailureDebugHook();
    lfatal << "Assertion failed in " RT_FILE_NAME ":" << __LINE__ << " : " << expr << " : " << msg << std::endl;
    std::terminate();
}
#endif
}

#define RT_ASSERT( condition )                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        if( !static_cast<bool>( condition ) )                                                                          \
        {                                                                                                              \
            prodlib::assertCondition( RT_FILE_NAME, __LINE__, #condition );                                            \
        }                                                                                                              \
    } while( false )

// Evaluate the condition before evaluting msg arguments
#define RT_ASSERT_MSG( condition, msg )                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        if( !static_cast<bool>( condition ) )                                                                          \
        {                                                                                                              \
            prodlib::assertMessage( RT_FILE_NAME, __LINE__, #condition, msg );                                         \
        }                                                                                                              \
    } while( false )

#define RT_ASSERT_FAIL_MSG_IMPL( msg )                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        prodlib::assertFailureDebugHook();                                                                             \
        throw prodlib::AssertionFailure( RT_EXCEPTION_INFO, msg );                                                     \
    } while( false )

#define RT_ASSERT_FAIL_MSG( msg )                                                                                      \
    RT_ASSERT_FAIL_MSG_IMPL( std::string( "Unconditional assertion failure: " ) + ( msg ) )
#define RT_ASSERT_FAIL() RT_ASSERT_FAIL_MSG_IMPL( "Unconditional assertion failure" )

// Should be used when asserting in destructors where we don't want to throw.
// This doesn't do anything in public builds, because we don't want to ever
// terminate() a client app in the wild. Therefore the throwing variant should
// be used whenever throwing isn't a concern.
#if defined( DEBUG ) || defined( DEVELOP )
#define RT_ASSERT_NOTHROW( condition, msg )                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        if( !static_cast<bool>( condition ) )                                                                          \
        {                                                                                                              \
            prodlib::assertNoThrow( RT_FILE_NAME, __LINE__, #condition, msg );                                         \
        }                                                                                                              \
    } while( false )
#else
#define RT_ASSERT_NOTHROW( condition, msg )
#endif
