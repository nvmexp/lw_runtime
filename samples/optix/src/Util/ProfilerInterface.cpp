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

#include <Util/ProfilerInterface.h>

// Turn this on when you want profile code
//#define USE_PROFILER

#ifdef USE_PROFILER
#if _MSC_VER < 1900  // visual studio 15 (2017) has built-in profiling support and does not ship profiling SDK
#define USE_VSPERF
#endif
#endif

#ifdef USE_VSPERF
#define VSPERF_NO_DEFAULT_LIB
#include <../../Team Tools/Performance Tools/x64/PerfSDK/VSPerf.h>
// I don't know how to use a relative path here, and I don't want to have to add library linker paths to the whole project.
#pragma comment( lib,                                                                                                  \
                 "c:/Program Files (x86)/Microsoft Visual Studio 12.0/Team Tools/Performance Tools/x64/PerfSDK/VSPerf.lib" )
#include <Strsafe.h>
#if DEBUG
#include <Windows.h>  // for OutputDebugString and friends
#endif
#endif


void optix::ProfilerInterface::startProfiler()
{
#ifdef USE_VSPERF
// StartProfile and StopProfile control the
// Start/Stop state for the profiling level.
// The default initial value of Start/Stop is 1.
// The initial value can be changed in the registry.
// Each call to StartProfile sets Start/Stop to 1;
// each call to StopProfile sets it to 0.

#ifdef DEBUG
    // Variables used to print output.
    HRESULT hResult;
    TCHAR   tchBuffer[256];
#endif

    // Declare enumeration to hold return value of
    // the call to StartProfile.
    PROFILE_COMMAND_STATUS profileResult;

    profileResult = StartProfile( PROFILE_THREADLEVEL, PROFILE_LWRRENTID );

#ifdef DEBUG
    // Format and print result.
    LPCTSTR pszFormat = TEXT( "%s %d.\0" );
    TCHAR*  pszTxt    = TEXT( "StartProfile returned" );
    hResult           = StringCchPrintf( tchBuffer, 256, pszFormat, pszTxt, profileResult );

    OutputDebugString( tchBuffer );
#endif
#endif
}

void optix::ProfilerInterface::stopProfiler()
{
#ifdef USE_VSPERF
// StartProfile and StopProfile control the
// Start/Stop state for the profiling level.
// The default initial value of Start/Stop is 1.
// The initial value can be changed in the registry.
// Each call to StartProfile sets Start/Stop to 1;
// each call to StopProfile sets it to 0.

#ifdef DEBUG
    // Variables used to print output.
    HRESULT hResult;
    TCHAR   tchBuffer[256];
#endif

    // Declare enumeration to hold return value of
    // the call to StartProfile.
    PROFILE_COMMAND_STATUS profileResult;

    profileResult = StopProfile( PROFILE_THREADLEVEL, PROFILE_LWRRENTID );

#ifdef DEBUG
    // Format and print result.
    LPCTSTR pszFormat = TEXT( "%s %d.\0" );
    TCHAR*  pszTxt    = TEXT( "StopProfile returned" );
    hResult           = StringCchPrintf( tchBuffer, 256, pszFormat, pszTxt, profileResult );

    OutputDebugString( tchBuffer );
#endif
#endif
}

void optix::ProfilerInterface::suspendProfiler()
{
#ifdef USE_VSPERF
// The initial value of the Suspend/Resume counter is 0.
// Each call to SuspendProfile adds 1 to the
// Suspend/Resume count; each call
// to ResumeProfile subtracts 1.

#ifdef DEBUG
    // Variables used to print output
    HRESULT hResult;
    TCHAR   tchBuffer[256];
#endif

    // Declare enumeration to hold result of call
    // to SuspendProfile
    PROFILE_COMMAND_STATUS profileResult;

    profileResult = SuspendProfile( PROFILE_GLOBALLEVEL, PROFILE_LWRRENTID );

#ifdef DEBUG
    // Format and print result.
    LPCTSTR pszFormat = TEXT( "%s %d.\0" );
    TCHAR*  pszTxt    = TEXT( "SuspendProfile returned" );
    hResult           = StringCchPrintf( tchBuffer, 256, pszFormat, pszTxt, profileResult );

    OutputDebugString( tchBuffer );
#endif
#endif
}

void optix::ProfilerInterface::resumeProfiler()
{
#ifdef USE_VSPERF
// The initial value of the Suspend/Resume counter is 0.
// Each call to SuspendProfile adds 1 to the Suspend/Resume
// count; each call to ResumeProfile subtracts 1.

#ifdef DEBUG
    // Variables used to print output.
    HRESULT hResult;
    TCHAR   tchBuffer[256];
#endif

    // Declare enumeration to hold result of call to ResumeProfile
    PROFILE_COMMAND_STATUS profileResult;

    profileResult = ResumeProfile( PROFILE_GLOBALLEVEL, PROFILE_LWRRENTID );

#ifdef DEBUG
    // Format and print result.
    LPCTSTR pszFormat = TEXT( "%s %d.\0" );
    TCHAR*  pszTxt    = TEXT( "ResumeProfile returned" );
    hResult           = StringCchPrintf( tchBuffer, 256, pszFormat, pszTxt, profileResult );

    OutputDebugString( tchBuffer );
#endif
#endif
}
