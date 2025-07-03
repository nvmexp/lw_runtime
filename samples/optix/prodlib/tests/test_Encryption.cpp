/*
 * Copyright (c) 2018, LWPU CORPORATION. All rights reserved.
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

#include <prodlib/misc/Encryption.h>

#include <gtest/gtest.h>

#include <cstdint>

// clang-format off
struct Encryption : public testing::Test
{
    // Special cases and some random values
    const std::vector<std::uint32_t> m_input = { 
        0x00000000, 0xffffffff,
        0xb800221f, 0x5af5f21d, 
        0xecda6206, 0x34ea8ef3, 
        0xfcd5d683, 0xb549ca10, 
        0x55778482, 0x08960641,
        0xa2415ee3, 0xe732b876, 
        0xd72b6e8c, 0x8ee5ed75, 
        0x2bafbc16, 0x51fc6df8, 
        0x51c7b668, 0x090cd756,
        0x913ce612
    };

    const std::uint32_t m_key[4] = { 0x3a984218, 0x2b10f73c, 0x2312bacb, 0x1c0c2670 };
};
// clang-format on


TEST_F( Encryption, MinimallyObfuscates )
{
    // Encrypt in place
    std::vector<std::uint32_t> buf = m_input;
    prodlib::tea_encrypt( reinterpret_cast<unsigned char*>( buf.data() ), sizeof( std::uint32_t ) * buf.size(), m_key );

    // Check for minimal obfuscation: every byte is different after encryption
    for( size_t i = 0; i < m_input.size(); ++i )
    {
        EXPECT_FALSE( buf[i] == m_input[i] );
    }
}


TEST_F( Encryption, IsReversible )
{
    // Encrypt in place
    std::vector<std::uint32_t> buf = m_input;
    prodlib::tea_encrypt( reinterpret_cast<unsigned char*>( buf.data() ), sizeof( std::uint32_t ) * buf.size(), m_key );

    // Reverse
    prodlib::tea_decrypt( reinterpret_cast<unsigned char*>( buf.data() ), sizeof( std::uint32_t ) * buf.size(), m_key );

    EXPECT_TRUE( buf == m_input );
}


int main( int argc, char* argv[] )
{
    testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS();
}
