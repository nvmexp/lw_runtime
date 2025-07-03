/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
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

#include <exp/context/EncryptionManager.h>

#include <prodlib/exceptions/Assert.h>
#include <prodlib/misc/Encryption.h>

#include <cstring>

#define OPTIX_PTX_ENCRYPTION_STANDALONE
#include <optix_ext_ptx_encryption_utilities.h>  // for sha256()

namespace optix_exp {

const std::string EncryptionManager::s_prefix = "eptx0001";

EncryptionManager::EncryptionManager()
{
    optix::detail::generateSalt( m_optixSalt );
}

OptixResult EncryptionManager::getOptixSalt( void* salt, size_t saltLength, ErrorDetails& errDetails )
{
    RT_ASSERT( salt );

    if( saltLength != s_saltLength )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "Wrong length for OptiX salt." );

    std::lock_guard<std::mutex> lock( m_mutex );

    memcpy( salt, &m_optixSalt[0], saltLength );

    return OPTIX_SUCCESS;
}

OptixResult EncryptionManager::setOptixSalt( const void* salt, size_t saltLength, ErrorDetails& errDetails )
{
    RT_ASSERT( salt );

    if( saltLength != s_saltLength )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "Wrong length for OptiX salt." );

    std::lock_guard<std::mutex> lock( m_mutex );

    if( isEncryptionEnabledLocked() )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_OPERATION,
                                      "OptiX salt can't be changed after encryption has been enabled." );

    memcpy( &m_optixSalt[0], salt, saltLength );
    m_weakVariant = true;

    return OPTIX_SUCCESS;
}

OptixResult EncryptionManager::setVendorSalt( const void* salt, size_t saltLength, ErrorDetails& errDetails )
{
    RT_ASSERT( salt );

    if( saltLength != s_saltLength )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "Wrong length for vendor salt." );

    std::lock_guard<std::mutex> lock( m_mutex );

    if( isEncryptionEnabledLocked() )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_OPERATION,
                                      "Vendor salt can't be changed after encryption has been enabled." );

    memcpy( &m_vendorSalt[0], salt, saltLength );
    m_vendorSaltSet = true;

    generateSessionKey();

    return OPTIX_SUCCESS;
}

OptixResult EncryptionManager::setPublicVendorKey( const void* buffer, size_t bufferLength, ErrorDetails& errDetails )
{
    RT_ASSERT( buffer );

    std::lock_guard<std::mutex> lock( m_mutex );

    if( isEncryptionEnabledLocked() )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_OPERATION,
                                      "Public vendor key can't be changed after encryption has been enabled." );

    m_publicVendorKey.resize( bufferLength );
    memcpy( &m_publicVendorKey[0], buffer, bufferLength );
    m_publicVendorKeySet = true;

    generateSessionKey();

    return OPTIX_SUCCESS;
}

bool EncryptionManager::isEncryptionEnabled() const
{
    std::lock_guard<std::mutex> lock( m_mutex );
    return isEncryptionEnabledLocked();
}

bool EncryptionManager::isEncryptionEnabledLocked() const
{
    return m_vendorSaltSet && m_publicVendorKeySet;
}

bool EncryptionManager::isWeakVariant() const
{
    std::lock_guard<std::mutex> lock( m_mutex );
    return m_weakVariant;
}

bool EncryptionManager::hasEncryptionPrefix( const prodlib::StringView& input ) const
{
    if( input.size() < s_prefix.size() )
        return false;

    if( memcmp( input.data(), s_prefix.data(), s_prefix.size() ) != 0 )
        return false;

    return true;
}

OptixResult EncryptionManager::decrypt( const prodlib::StringView& encrypted, std::vector<char>& decryptedData, ErrorDetails& errDetails ) const
{
    if( !isEncryptionEnabled() )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_OPERATION, "Decryption is not enabled" );
    if( !hasEncryptionPrefix( encrypted ) )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "Input is not OptiX encrypted" );

    if( OptixResult result = decode( encrypted, decryptedData, errDetails ) )
        return result;

    decryptTea( decryptedData.data(), decryptedData.size() );

    // Add a null terminator.  Note that the decoded string is not null terminated (decryptTea uses
    // the size and works in place), but it was reserved with an extra byte, so this push_back does
    // not cause a realloc.
    decryptedData.push_back( '\0' );

    return OPTIX_SUCCESS;
}

OptixResult EncryptionManager::decryptString( const prodlib::StringView& encryptedStr,
                                              char*                      decryptedData,
                                              size_t&                    decryptedDataSize,
                                              size_t                     windowSize,
                                              size_t&                    consumed,
                                              ErrorDetails&              errDetails )
{
    // first decode windowSize chars
    if( OptixResult result = decodeString( encryptedStr, decryptedData, decryptedDataSize, windowSize, consumed, errDetails ) )
        return result;
    decryptTea( decryptedData, decryptedDataSize );
    return OPTIX_SUCCESS;
}

OptixResult EncryptionManager::decryptStringWithPrefix( const prodlib::StringView& encryptedStr,
                                                        char*                      decryptedData,
                                                        size_t&                    decryptedDataSize,
                                                        size_t                     windowSize,
                                                        size_t&                    consumed,
                                                        ErrorDetails&              errDetails )
{
    if( !isEncryptionEnabled() )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_OPERATION, "Decryption is not enabled" );
    if( !hasEncryptionPrefix( encryptedStr ) )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "Input is not OptiX encrypted" );

    // first strip the prefix
    prodlib::StringView prefixStripped( encryptedStr.data() + s_prefix.size(), encryptedStr.size() - s_prefix.size() );

    // decrypt the rest
    if( OptixResult result = decryptString(prefixStripped, decryptedData, decryptedDataSize, windowSize, consumed, errDetails) )
        return result;

    // we have consumed the prefix
    consumed += s_prefix.size();
    return OPTIX_SUCCESS;
}

size_t EncryptionManager::getVendorSaltLength() const
{
    return sizeof( m_vendorSalt ) / sizeof( m_vendorSalt[0] );
}

size_t EncryptionManager::getPublicVendorKeyLength() const
{
    return m_publicVendorKey.size();
}

void EncryptionManager::generateSessionKey()
{
    if( !m_vendorSaltSet || !m_publicVendorKeySet )
        return;

    // Generate SVK = hash(PVK, NSK).
    const size_t               publicVendorKeyLength      = getPublicVendorKeyLength();
    const size_t               secretLwidiaKeyLength      = 64;
    const size_t               secretVendorKeyInputLength = publicVendorKeyLength + secretLwidiaKeyLength;
    std::vector<unsigned char> secretVendorKeyInput( secretVendorKeyInputLength );
    memcpy( &secretVendorKeyInput[0], &m_publicVendorKey[0], publicVendorKeyLength );
    getSecretLwidiaKey( &secretVendorKeyInput[publicVendorKeyLength], secretLwidiaKeyLength );

    const size_t               secretVendorKeyLength = 32;
    std::vector<unsigned char> secretVendorKey( secretVendorKeyLength );
    optix::detail::sha256( &secretVendorKeyInput[0], secretVendorKeyInputLength, &secretVendorKey[0] );
    clearMemory( &secretVendorKeyInput[0], secretVendorKeyInputLength );

    // Generate SVK' = hexify(SVK). This form of the computed SVK should be identical to the one
    // that was given to the vendor before.
    const size_t secretVendorKeyHexLength = 64;
    // Make buffer one byte larger to avoid truncation in the hexify() call below.
    unsigned char secretVendorKeyHex[secretVendorKeyHexLength + 1];
    for( int i = 0; i < 32; ++i )
        hexify( secretVendorKeyHex + 2 * i, static_cast<unsigned char>( secretVendorKey[i] ) );
    clearMemory( &secretVendorKey[0], secretVendorKeyLength );

    // Generate SK = hash(OS, SVK', VS).
    const size_t               optixSaltLength       = s_saltLength;
    const size_t               vendorSaltLength      = s_saltLength;
    const size_t               sessionKeyInputLength = optixSaltLength + secretVendorKeyHexLength + vendorSaltLength;
    std::vector<unsigned char> sessionKeyInput( sessionKeyInputLength );
    memcpy( &sessionKeyInput[0], m_optixSalt, optixSaltLength );
    memcpy( &sessionKeyInput[optixSaltLength], secretVendorKeyHex, secretVendorKeyHexLength );
    memcpy( &sessionKeyInput[optixSaltLength + secretVendorKeyHexLength], m_vendorSalt, vendorSaltLength );
    clearMemory( secretVendorKeyHex, secretVendorKeyHexLength );

    m_sessionKey.resize( 32 );
    optix::detail::sha256( &sessionKeyInput[0], sessionKeyInputLength, &m_sessionKey[0] );
    clearMemory( &sessionKeyInput[0], sessionKeyInputLength );

    for( size_t i = 0; i < 16; ++i )
        m_sessionKey[i] += m_sessionKey[i + 16];
    m_sessionKey.resize( 16 );
}

OptixResult EncryptionManager::decode( const prodlib::StringView& input, std::vector<char>& outputData, ErrorDetails& errDetails )
{
    // Ilwariant (enforced by single caller decrypt()): isEncryptionEnabled() && hasEncryptionPrefix( input )

    // We want to allocate storage without initializing storage, so use reserve(), not resize()
    // An extra byte is reserved for a null terminator, which is added after decryptTea() is called.
    outputData.clear();
    outputData.reserve( input.size() - s_prefix.size() + 1 );

    const char* const inputStr = input.data();
    const size_t      inputLen = input.size();

    for( size_t i = s_prefix.size(); i < inputLen; ++i )
    {
        unsigned char c = inputStr[i];
        if( c == '\1' )
        {
            ++i;
            if( i >= inputLen )
            {
                outputData.clear();
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                              "Encrypted PTX has invalid position of escape character." );
            }

            c = inputStr[i];
            if( c != '\1' && c != '\2' )
            {
                outputData.clear();
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "Encrypted PTX has invalid escape sequence." );
            }

            c -= 1;
        }
        outputData.push_back( c );
    }

    return OPTIX_SUCCESS;
}

OptixResult EncryptionManager::decodeString( const prodlib::StringView& encryptedStr,
                                             char*                      decodedData,
                                             size_t&                    decoded,
                                             size_t                     windowSize,
                                             size_t&                    consumed,
                                             ErrorDetails&              errDetails )
{
    const char* const inputStr = encryptedStr.data();
    decoded                    = 0;
    consumed                   = 0;
    for( ; decoded < windowSize && consumed < encryptedStr.size(); ++consumed )
    {
        unsigned char c = inputStr[consumed];
        if( c == '\1' )
        {
            ++consumed;
            if( consumed >= encryptedStr.size() )
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                              "Encrypted PTX has invalid position of escape character." );

            c = inputStr[consumed];
            if( c != '\1' && c != '\2' )
                return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "Encrypted PTX has invalid escape sequence." );

            c -= 1;
        }
        decodedData[decoded++] = c;
    }

    return OPTIX_SUCCESS;
}

void EncryptionManager::decryptTea( char* data, size_t length ) const
{
    // Ilwariant (enforced by single caller decrypt()): isEncryptionEnabled()

    uint32_t teaKey[4];
    RT_ASSERT( m_sessionKey.size() == sizeof( teaKey ) );
    memcpy( &teaKey[0], &m_sessionKey[0], sizeof( teaKey ) );

    prodlib::tea_decrypt( reinterpret_cast<unsigned char*>( data ), length, teaKey );
}

void EncryptionManager::clearMemory( void* buffer, size_t length )
{
#ifdef _WIN32
    RtlSelwreZeroMemory( buffer, length );
#else
    // TODO Investigate whether memset_s() can be used instead.
    memset( buffer, 0, length );
#endif
}

void EncryptionManager::hexify( unsigned char* buffer, unsigned char input )
{
#ifdef _WIN32
    _snprintf_s( reinterpret_cast<char*>( buffer ), 3, _TRUNCATE, "%02x", input );
#else
    snprintf( reinterpret_cast<char*>( buffer ), 3, "%02x", input );
#endif
}

void EncryptionManager::getSecretLwidiaKey( void* buffer, size_t bufferLength )
{
    RT_ASSERT( bufferLength == 64 );

    // buffer: const char *s = "-3343556356fgfgfdessss-(--9355-489795-2333354:[]}}{[]523552%GWEf";
    // doing int *a = (int*)s; for(i = 0; i < 16; ++i) printf("%d ", *a++) produces the
    // int array we use here so it will not show up as strings in the final binary at least.
    int                   s_one[]   = {875770669, 909456691, 1714828595, 1718052455, 1936942436};
    int                   apa[]     = {120929, 90906691, 6661717, 251828595};
    int                   s_two[]   = {674067315, 859385133, 875377973};
    int                   apa2[]    = {13733, 906691, 2261717, 333595};
    int                   apa3[]    = {223429, 92430424, 4242, 244234};
    int                   s_three[] = {959920440, 858926389, 892547891, 1566259764, 1534819709};
    int                   s_four[]  = {858928477, 624047413};
    int                   five      = 1715820359;
    std::unique_ptr<char> keyrr( new char[64] );
    char*                 key2 = keyrr.get();

    char* key = static_cast<char*>( buffer );
    memcpy( key, s_one, sizeof( s_one ) );
    memcpy( key + sizeof( s_one ), s_two, sizeof( s_two ) );
    memcpy( key2 + 10, apa, sizeof( apa ) );
    memcpy( key + sizeof( s_one ) + sizeof( s_two ), apa2, sizeof( apa2 ) );
    memcpy( key2 + sizeof( apa ), apa2, sizeof( apa2 ) );
    memcpy( key + sizeof( s_one ) + sizeof( s_two ), s_three, sizeof( s_three ) );
    memcpy( key2 + sizeof( apa2 ), apa3, sizeof( apa3 ) );
    memcpy( key + sizeof( s_one ) + sizeof( s_two ) + sizeof( s_three ), s_four, sizeof( s_four ) );
    memcpy( key + sizeof( s_one ) + sizeof( s_two ) + sizeof( s_three ) + sizeof( s_four ), &five, 4 );
}
}  // namespace optix_exp

// C interface
bool decryptString( void* encrypter, const char* encrypted, size_t encrypted_len, char* decrypted, size_t* decrypted_len, size_t window_size, size_t* consumed )
{
    // pre-condition: prefix is already skipped over
    optix_exp::EncryptionManager* encMgr = static_cast<optix_exp::EncryptionManager*>( encrypter );

    optix_exp::ErrorDetails errDetails;
    if( encMgr->decryptString( { encrypted, encrypted_len }, decrypted, *decrypted_len, window_size, *consumed, errDetails ) )
        return false;
    return true;
}
