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
#pragma once

#include <iostream>
#include <mutex>
#include <sstream>

namespace optix {
namespace demandLoad {

extern std::mutex g_demandLoadLogMutex;

// Normal logging: high level operations that take place once per context launch, or for unusual information messages.
// Medium verbose logging: medium level operations that take place once per device.
// Verbose logging: full details of all operations.
bool isLogActive();
bool isLogMediumVerboseActive();
bool isLogVerboseActive();

std::ostream& logStream();
std::ostream& logMediumVerboseStream();
std::ostream& logVerboseStream();

#define LOG_NORMAL( things_ )                                                                                          \
    do                                                                                                                 \
    {                                                                                                                  \
        if( optix::demandLoad::isLogActive() )                                                                         \
        {                                                                                                              \
            std::lock_guard<std::mutex> lock( optix::demandLoad::g_demandLoadLogMutex );                               \
            std::ostringstream          str;                                                                           \
            str << things_; /* NOLINT(bugprone-macro-parentheses) */                                                   \
            optix::demandLoad::logStream() << str.str();                                                               \
        }                                                                                                              \
    } while( 0 )

#define LOG_MEDIUM_VERBOSE( things_ )                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if( optix::demandLoad::isLogMediumVerboseActive() )                                                            \
        {                                                                                                              \
            std::lock_guard<std::mutex> lock( optix::demandLoad::g_demandLoadLogMutex );                               \
            std::ostringstream          str;                                                                           \
            str << things_; /* NOLINT(bugprone-macro-parentheses) */                                                   \
            optix::demandLoad::logMediumVerboseStream() << str.str();                                                  \
        }                                                                                                              \
    } while( 0 )

#define LOG_VERBOSE( things_ )                                                                                         \
    do                                                                                                                 \
    {                                                                                                                  \
        if( optix::demandLoad::isLogVerboseActive() )                                                                  \
        {                                                                                                              \
            std::lock_guard<std::mutex> lock( optix::demandLoad::g_demandLoadLogMutex );                               \
            std::ostringstream          str;                                                                           \
            str << things_; /* NOLINT(bugprone-macro-parentheses) */                                                   \
            optix::demandLoad::logVerboseStream() << str.str();                                                        \
        }                                                                                                              \
    } while( 0 )

}  // namespace demandLoad
}  // namespace optix
