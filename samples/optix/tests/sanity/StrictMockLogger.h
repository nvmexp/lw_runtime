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

#include <gmock/gmock.h>

const unsigned int LOG_LEVEL_ERROR = 2;

struct Logger
{
    virtual ~Logger() {}

    virtual void log( unsigned int level, const char* tag, const char* message ) = 0;
};

struct MockLogger : Logger
{
    MOCK_METHOD3( log, void( unsigned int, const char*, const char* message ) );

    static void apiCallback( unsigned int level, const char* tag, const char* message, void* callbackData )
    {
        static_cast<MockLogger*>( callbackData )->log( level, tag, message );
    }
};

using StrictMockLogger = ::testing::StrictMock<MockLogger>;

#define EXPECT_LOG_MSG( msg_ ) EXPECT_CALL( m_logger, log( LOG_LEVEL_ERROR, _, HasSubstr( msg_ ) ) )
