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

#include <fstream>
#include <memory>
#include <string>

// Use a single defined macro to globally control whether or not to log.
#if defined( DEBUG ) || defined( DEVELOP )
#if !defined( OPTIX_ENABLE_LOGGING )
#define OPTIX_ENABLE_LOGGING
#endif
#endif

namespace prodlib {
namespace log {

// Interface for logging messages and printing them out to a stream.
//
// Upon construction, the environment variables OPTIX_LOG_LEVEL and OPTIX_LOG_FILE are
// read to configure how much and where logging information is stored.  The log level
// can be overriden later by calling setLogLevel().
//
// OPTIX_LOG_LEVEL is a number between 0 and 100.  0 has the least logging, and 100 has
// all the logging.  This value defaults to 4 and will be clamped if out of bounds.
//
// OPTIX_LOG_FILE:  "stdout" for standard out
//                  "stderr" for standard error (default)
//                  <filename> for output to a file
//
// To log a message, use one of the provided macros to obtain an ostream:
//   lfatal
//   lerr
//   lwarn
//   lprint
//   ltemp
//   llog(level)
//
// All of these except llog will log to their associated level. llog accepts a parameter that
// is used to specify a level directly.
//
// An example of use: lprint << "Testing output\n";
//
// In addition if you need to pass the log stream as a paramter, you can use the _stream
// variant of the macros:
//
// printStats( lprint_stream );
//
// Log messages can be prepended with various information (msg level, timestamp, source position).
// To turn these on, use the following knobs:
//    log.printLevel
//    log.printTimestamp
//    log.printOrigin
//
// Logging of higher levels than "print" can be restricted to originate from specific files.
// This is done using the knob:
//    log.restrictToFiles <filename1> ; <filename2> ; ...
//
// There is a global singleton Logger, accessed via logger().
// You can sort the log file with "sort -n".
//
// If you have a lot of work to do to create log messages, you can call active() to
// determine if the log will be outputed or not.
//
// If the logging code contains proprietary information, emits a lot of code, or would be
// slow to run, you should wrap it in `#if defined( OPTIX_ENABLE_LOGGING )`.
//
// Log level colwentions:
//
// LEVEL    SHORTLWT   DESCRIPTION
// ------------------------------------------------------------------------
//
//  0                   Do not use.
//
//  1       fatal       For assertions. Always indicates a bug in optix.
//
//  2       error       Regular errors, e.g. invalid user parameter, failure
//                      to open a file, etc. We may one day expose messages
//                      on this level to the user through a debug API.
//
//  3       warning     Warnings: indicates that Optix might not behave exactly
//                      as intended by the user or may perform slower than expected.
//                      We may one day expose messages on this level to the user
//                      through a debug API.
//
//  4       print       Stuff you want Optix to print on a default run without
//                      any knobs set (i.e. very little). Should be used for
//                      messages that are turned on explicitly through knobs.
//
//  5       temp        Temporary debug output.  This is one larger than print,
//                      so you can disable temporary output and still get the default
//                      output.
//                      This is the highest level that's on by default.
//
//  6 - 9               Do not use.
//
//  10 - 19             Messages that are at per-context-creation frequency
//                      (roughly once per run).
//
//  20 - 29             Messages at canonicalization/compilation frequency
//                      (O(10) per run).
//
//  30 - 39             Messages at launch, acceleration build, and API object
//                      creation frequency (O(100-1000) per run).
//
//  40 - 49             Messages that can occur many times per launch (e.g.
//                      every API call).
//
//  >= 50               Debugging details.
//

static const int LEV_FATAL   = 1;
static const int LEV_ERROR   = 2;
static const int LEV_WARNING = 3;
static const int LEV_PRINT   = 4;
static const int LEV_TEMP    = 5;

int  level();
bool active( int level = 0 );

// Shouldn't be used directly (use the macros below).
std::ostream& stream( int level, const char* file, int line );

}  // end namespace log
}  // end namespace prodlib

// TODO: Unfortunately this lif_active macro causes warnings in some situations
//       on Mac (see https://jirasw/browse/OP-1247).
#define lif_active( lev_ )                                                                                             \
    if( !prodlib::log::active( lev_ ) )                                                                                \
        ;                                                                                                              \
    else

// Make sure that we do not leak the file names in release builds.
#ifndef RT_FILE_NAME
#if defined( OPTIX_ENABLE_LOGGING )
#define RT_FILE_NAME __FILE__
#else
#define RT_FILE_NAME "<internal>"
#endif
#endif

// Use these directly if you need the ostream object, e.g. to pass to a function
#define lfatal_stream prodlib::log::stream( prodlib::log::LEV_FATAL, RT_FILE_NAME, __LINE__ )
#define lerr_stream prodlib::log::stream( prodlib::log::LEV_ERROR, RT_FILE_NAME, __LINE__ )
#define lwarn_stream prodlib::log::stream( prodlib::log::LEV_WARNING, RT_FILE_NAME, __LINE__ )
#define lprint_stream prodlib::log::stream( prodlib::log::LEV_PRINT, RT_FILE_NAME, __LINE__ )
#define ltemp_stream prodlib::log::stream( prodlib::log::LEV_TEMP, RT_FILE_NAME, __LINE__ )
#define llog_stream( lev_ ) prodlib::log::stream( ( lev_ ), RT_FILE_NAME, __LINE__ )

// Log macros. Use these directly.
#define lfatal lif_active( prodlib::log::LEV_FATAL ) lfatal_stream
#define lerr lif_active( prodlib::log::LEV_ERROR ) lerr_stream
#define lwarn lif_active( prodlib::log::LEV_WARNING ) lwarn_stream
#define lprint lif_active( prodlib::log::LEV_PRINT ) lprint_stream
#define ltemp lif_active( prodlib::log::LEV_TEMP ) ltemp_stream
#define llog( lev_ ) lif_active( lev_ ) llog_stream( lev_ )
