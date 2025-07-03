/*
 * Copyright (c) 2019, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <prodlib/misc/String.h>

#include <gtest/gtest.h>

#include <cstring>

TEST( Strnstr, NeedleLenZero )
{
    const char* hayStack    = "hayStack";
    const char* needle      = "";
    size_t      hayStackLen = strlen( hayStack );

    EXPECT_EQ( prodlib::strNStr( hayStack, needle, hayStackLen ), hayStack );
}

TEST( Strnstr, HayStackLenZero )
{
    const char* hayStack    = "";
    const char* needle      = "needle";
    size_t      hayStackLen = strlen( hayStack );

    EXPECT_EQ( prodlib::strNStr( hayStack, needle, hayStackLen ), nullptr );
}

TEST( Strnstr, HayStackLenAndNeedleLenZero )
{
    const char* hayStack    = "";
    const char* needle      = "";
    size_t      hayStackLen = strlen( hayStack );

    EXPECT_EQ( prodlib::strNStr( hayStack, needle, hayStackLen ), hayStack );
}

TEST( Strnstr, MatchFront )
{
    const char* hayStack    = "hayStack";
    const char* needle      = "hay";
    size_t      hayStackLen = strlen( hayStack );

    EXPECT_EQ( prodlib::strNStr( hayStack, needle, hayStackLen ), hayStack );
}

TEST( Strnstr, MatchMiddle )
{
    const char* hayStack    = "hayStack";
    const char* needle      = "ySta";
    size_t      hayStackLen = strlen( hayStack );

    EXPECT_EQ( prodlib::strNStr( hayStack, needle, hayStackLen ), hayStack + 2 );
}

TEST( Strnstr, MatchEnd )
{
    const char* hayStack    = "hayStack";
    const char* needle      = "Stack";
    size_t      hayStackLen = strlen( hayStack );

    EXPECT_EQ( prodlib::strNStr( hayStack, needle, hayStackLen ), hayStack + 3 );
}

TEST( Strnstr, MatchAtSecondInitialCharachterMatch )
{
    const char* hayStack    = "hayStack";
    const char* needle      = "ac";
    size_t      hayStackLen = strlen( hayStack );

    EXPECT_EQ( prodlib::strNStr( hayStack, needle, hayStackLen ), hayStack + 5 );
}

TEST( Strnstr, DoNotExceedHayStackLen )
{
    // Reduce hayStackLen, but keep a matching character at hayStack[hayStackLen], such that a non-match is a
    // good indicator that hayStackLen was not exceeded.

    const char* hayStack    = "hayStack";
    const char* needle      = "Stack";
    size_t      hayStackLen = strlen( hayStack ) - 1;

    EXPECT_EQ( prodlib::strNStr( hayStack, needle, hayStackLen ), nullptr );
}
