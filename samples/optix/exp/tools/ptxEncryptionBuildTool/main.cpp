/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#include <fstream>
#include <iostream>
#include <sstream>

#include <optix_ext_ptx_encryption_utilities.h>

#include "secrets.h"

#define OPTIX_PTX_ENCRYPTION_STANDALONE

[[noreturn]] void printUsageAndExit( const char* argv0 )
{
    std::cerr << "Usage:" << std::endl
              << "  " << argv0 << " -h | --help" << std::endl
              << "  " << argv0 << " [option <value>]..." << std::endl
              << "Options:" << std::endl
              << "  -h  | --help     Print this usage message" << std::endl
              << "  -if | --infile   Input file [string] (default: '')" << std::endl
              << "  -of | --outfile  Output file [string] (default: '')" << std::endl;
    exit( 1 );
}

void clearMemory( void* buffer, size_t length )
{
#ifdef _WIN32
    RtlSelwreZeroMemory( buffer, length );
#else
    // TODO Investigate whether memset_s() can be used instead.
    memset( buffer, 0, length );
#endif
}

std::vector<unsigned char> getSessionKey()
{
    // This private key was deduced by applying the same formula used by the
    // EncryptionManager during its generation of the session key, and using
    // the public key defined in secrets.h:
    //
    // privateKey = hexify(sha256(strcat(vendorPublicKey, lwidiaSecretKey)))
    const size_t  secretKeyLength                = 64;
    unsigned char secretKey[secretKeyLength + 1] = "f8e8a4f88a5288bf007518eb01379f0537e2b8a642ffdfb42a498b40d79e9c49";

    // Generate session key from hashed secret key + salts
    const size_t               sessionKeyInputLength = SALT_LENGTH + secretKeyLength + SALT_LENGTH;
    std::vector<unsigned char> sessionKeyInput( sessionKeyInputLength );
    memcpy( sessionKeyInput.data(), secrets::optixSalt, SALT_LENGTH );
    memcpy( &sessionKeyInput[SALT_LENGTH], secretKey, secretKeyLength );
    memcpy( &sessionKeyInput[SALT_LENGTH + secretKeyLength], secrets::vendorSalt, SALT_LENGTH );
    clearMemory( secretKey, secretKeyLength );

    std::vector<unsigned char> sessionKey( 32 );
    optix::detail::sha256( sessionKeyInput.data(), static_cast<unsigned int>( sessionKeyInputLength ), sessionKey.data() );
    clearMemory( sessionKeyInput.data(), sessionKeyInputLength );

    for( size_t i = 0; i < 16; ++i )
        sessionKey[i] += sessionKey[i + 16];
    sessionKey.resize( 16 );

    return sessionKey;
}

std::string encryptPtx( std::string ptx, std::vector<unsigned char>& sessionKey )
{
    // Encrypt PTX
    uint32_t teaKey[4];
    memcpy( &teaKey[0], sessionKey.data(), sizeof( teaKey ) );

    std::vector<unsigned char> result( ptx.size() );
    memcpy( result.data(), ptx.data(), ptx.size() );

    // Encrypt 8 byte blocks with TEA
    const uint32_t      MAGIC  = 0x9e3779b9;
    const unsigned char KEY[7] = {164, 195, 147, 255, 203, 161, 184};

    const size_t n = result.size() / static_cast<size_t>( 8 );
    uint32_t*    v = reinterpret_cast<uint32_t*>( result.data() );
    for( size_t i = 0; i < n; ++i )
    {
        uint32_t v0 = v[2 * i];
        uint32_t v1 = v[2 * i + 1];
        uint32_t s0 = 0;

        for( uint32_t n = 0; n < 16; n++ )
        {
            s0 += MAGIC;
            v0 += ( ( v1 << 4 ) + teaKey[0] ) ^ ( v1 + s0 ) ^ ( ( v1 >> 5 ) + teaKey[1] );
            v1 += ( ( v0 << 4 ) + teaKey[2] ) ^ ( v0 + s0 ) ^ ( ( v0 >> 5 ) + teaKey[3] );
        }

        v[2 * i]     = v0;
        v[2 * i + 1] = v1;
    }

    // Slightly obfuscate leftover bytes (at most 7) with simple xor.
    for( size_t i = 8 * n, k = 0; i < result.size(); ++i, ++k )
        result[i] = result[i] ^ KEY[k];

    // Replace '\0' by '\1\1' and '\1' by '\1\2'.
    std::string encodedPtx;
    for( char c : result )
    {
        if( c == '\0' || c == '\1' )
        {
            encodedPtx.push_back( '\1' );
            encodedPtx.push_back( c + 1 );
        }
        else
            encodedPtx.push_back( c );
    }

    std::string prefix       = "eptx0001";
    std::string encryptedPtx = prefix + encodedPtx;

    return encryptedPtx;
}

int main( int argc, char** argv )
{
    std::stringstream ptxBuffer;
    std::string       outfileName;

    for( int i = 1; i < argc; ++i )
    {
        std::string arg( argv[i] );

        if( arg == "-h" || arg == "--help" )
        {
            printUsageAndExit( argv[0] );
        }
        else if( ( arg == "-if" || arg == "--infile" ) && i + 1 < argc )
        {
            std::ifstream infile( argv[++i], std::fstream::in | std::fstream::binary );
            if( infile.is_open() )
            {
                ptxBuffer << infile.rdbuf();
                infile.close();
            }
        }
        else if( ( arg == "-of" || arg == "--outfile" ) && i + 1 < argc )
        {
            outfileName = argv[++i];
        }
        else
        {
            std::cerr << "Bad option: " << arg << std::endl;
            printUsageAndExit( argv[0] );
        }
    }

    std::vector<unsigned char> sessionKey   = getSessionKey();
    std::string                encryptedPtx = encryptPtx( ptxBuffer.str(), sessionKey );

    // Write encrypted PTX to file
    std::ofstream outfile( outfileName, std::fstream::out | std::fstream::binary );
    if( outfile.is_open() )
    {
        outfile << encryptedPtx;
        outfile.close();
    }
}
