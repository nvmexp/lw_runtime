//
//  Copyright (c) 2020 LWPU Corporation.  All rights reserved.
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

#include <srcTests.h>

#include <Util/JsonEscape.h>

using namespace testing;

TEST( TestJsonEscape, no_special_chars_copied_unchanged )
{
    const std::string noSpecials{
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[]{}|;:',./<>?`~"};

    EXPECT_EQ( noSpecials, optix::escapeJsonString( noSpecials ) );
}

TEST( TestJsonEscape, double_quotes_are_escaped )
{
    EXPECT_EQ( std::string{R"escaped(before\"and\"after)escaped"}, optix::escapeJsonString( R"raw(before"and"after)raw" ) );
}

TEST( TestJsonEscape, only_double_quote_is_escaped )
{
    EXPECT_EQ( std::string{R"escaped(\")escaped"}, optix::escapeJsonString( R"raw(")raw" ) );
}

TEST( TestJsonEscape, backslashes_are_escaped )
{
    EXPECT_EQ( std::string{R"escaped(before\\and\\after)escaped"}, optix::escapeJsonString( R"raw(before\and\after)raw" ) );
}

TEST( TestJsonEscape, only_backslash_is_escaped )
{
    EXPECT_EQ( std::string{R"escaped(\\)escaped"}, optix::escapeJsonString( R"raw(\)raw" ) );
}

TEST( TestJsonEscape, mixed_specials_are_escaped )
{
    EXPECT_EQ( std::string{R"escaped(\\\")escaped"}, optix::escapeJsonString( R"raw(\")raw" ) );
}
