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

#include <Memory/DemandLoad/RequestHandler/BlockStamper.h>

#include <Memory/DemandLoad/PagingServiceLogging.h>
#include <Util/ContainerAlgorithm.h>

#include <prodlib/misc/RTFormatUtil.h>

#include <private/optix_6_enum_printers.h>

#include <lwda_fp16.h>

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>

namespace optix {

namespace {

constexpr unsigned int FONT_WIDTH  = 5;
constexpr unsigned int FONT_HEIGHT = 7;

class BlockStamper
{
  public:
    BlockStamper( const RTmemoryblock& block, unsigned int xOffset, unsigned int yOffset )
        : m_block( block )
        , m_xOffset( xOffset )
        , m_yOffset( yOffset )
        , m_blackPixel( getBlackPixel( block.format ) )
        , m_whitePixel( getWhitePixel( block.format ) )
        , m_bytesPerPixel( prodlib::getElementSize( block.format ) )
    {
    }

    void stampId( unsigned int id );
    void         stampDebugPattern();
    unsigned int checkDebugPattern( std::vector<unsigned int>& byteOffsets, std::vector<unsigned int>& pixelCoords, unsigned int& pixelCount );
    void stampRed();

  private:
    static const unsigned char* getBlackPixel( RTformat format );
    static const unsigned char* getWhitePixel( RTformat format );
    const unsigned char* getRedPixel();
    void stampPixel( const unsigned char* source, unsigned char* dest )
    {
        std::copy( &source[0], &source[m_bytesPerPixel], dest );
    }
    void stampPixelSpan( const unsigned char* source, unsigned char* dest, unsigned int spanLength = 1 )
    {
        for( unsigned int i = 0; i < spanLength; ++i )
        {
            stampPixel( source, dest );
            dest += m_bytesPerPixel;
        }
    }
    unsigned int stampCharacter( char c, unsigned int x );

    const RTmemoryblock&       m_block;
    const unsigned int         m_xOffset;
    const unsigned int         m_yOffset;
    const unsigned char* const m_blackPixel;
    const unsigned char* const m_whitePixel;
    const unsigned int         m_bytesPerPixel;
};

const unsigned char* BlockStamper::getBlackPixel( RTformat format )
{
#if defined( OPTIX_ENABLE_LOGGING )
    // This is only used to know when you need to add more cases to the switch statements below
    // in order to get useful visible output.  Therefore, drop it from release builds.
    if( demandLoad::isLogActive() )
    {
        static std::vector<RTformat> seenFormats{RT_FORMAT_UNSIGNED_BYTE,  RT_FORMAT_UNSIGNED_BYTE2, RT_FORMAT_BYTE3,
                                                 RT_FORMAT_UNSIGNED_BYTE4, RT_FORMAT_HALF,           RT_FORMAT_HALF2,
                                                 RT_FORMAT_HALF3,          RT_FORMAT_HALF4,          RT_FORMAT_FLOAT,
                                                 RT_FORMAT_FLOAT2,         RT_FORMAT_FLOAT3,         RT_FORMAT_FLOAT4};
        // Request processing may be multithreaded.
        static std::mutex s_mutex;
        {
            std::lock_guard<std::mutex> lock( s_mutex );
            if( algorithm::find( seenFormats, format ) == seenFormats.end() )
            {
                LOG_NORMAL( "BlockStamper::getBlackPixel: Encountered format " << toString( format ) << '\n' );
                seenFormats.push_back( format );
            }
        }
    }
#endif

    switch( format )
    {
        case RT_FORMAT_UNSIGNED_BYTE:
        case RT_FORMAT_UNSIGNED_BYTE2:
        case RT_FORMAT_UNSIGNED_BYTE3:
        case RT_FORMAT_UNSIGNED_BYTE4:
        {
            static const unsigned char black[4] = {0, 0, 0, 255};
            return &black[0];
        }
        case RT_FORMAT_HALF:
        case RT_FORMAT_HALF2:
        case RT_FORMAT_HALF3:
        case RT_FORMAT_HALF4:
        {
            static const __half black[4] = {__float2half( 0.0f ), __float2half( 0.0f ), __float2half( 0.0f ), __float2half( 1.0f )};
            return reinterpret_cast<const unsigned char*>( &black[0] );
        }
        case RT_FORMAT_FLOAT:
        case RT_FORMAT_FLOAT2:
        case RT_FORMAT_FLOAT3:
        case RT_FORMAT_FLOAT4:
        {
            static const float black[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            return reinterpret_cast<const unsigned char*>( &black[0] );
        }
        default:
        {
            static const unsigned char black[16]{};
            return &black[0];
        }
    }
}

const unsigned char* BlockStamper::getWhitePixel( RTformat format )
{
    switch( format )
    {
        case RT_FORMAT_UNSIGNED_BYTE:
        case RT_FORMAT_UNSIGNED_BYTE2:
        case RT_FORMAT_UNSIGNED_BYTE3:
        case RT_FORMAT_UNSIGNED_BYTE4:
        {
            static const unsigned char white[4] = {255, 255, 255, 255};
            return &white[0];
        }
        case RT_FORMAT_HALF:
        case RT_FORMAT_HALF2:
        case RT_FORMAT_HALF3:
        case RT_FORMAT_HALF4:
        {
            static const __half white[4] = {__float2half( 1.0f ), __float2half( 1.0f ), __float2half( 1.0f ), __float2half( 1.0f )};
            return reinterpret_cast<const unsigned char*>( &white[0] );
        }
        case RT_FORMAT_FLOAT:
        case RT_FORMAT_FLOAT2:
        case RT_FORMAT_FLOAT3:
        case RT_FORMAT_FLOAT4:
        {
            static const float white[4] = {1.0f, 1.0f, 1.0f, 1.0f};
            return reinterpret_cast<const unsigned char*>( &white[0] );
        }
        default:
        {
            static const unsigned char white[16] = {255, 255, 255, 255, 255, 255, 255, 255,
                                                    255, 255, 255, 255, 255, 255, 255, 255};
            return &white[0];
        }
    }
}

const unsigned char* BlockStamper::getRedPixel()
{
    switch( m_block.format )
    {
        case RT_FORMAT_UNSIGNED_BYTE:
        case RT_FORMAT_UNSIGNED_BYTE2:
        case RT_FORMAT_UNSIGNED_BYTE3:
        case RT_FORMAT_UNSIGNED_BYTE4:
        {
            static const unsigned char red[4] = {255, 0, 0, 255};
            return &red[0];
        }
        case RT_FORMAT_HALF:
        case RT_FORMAT_HALF2:
        case RT_FORMAT_HALF3:
        case RT_FORMAT_HALF4:
        {
            static const __half red[4] = {__float2half( 1.0f ), __float2half( 0.0f ), __float2half( 0.0f ), __float2half( 1.0f )};
            return reinterpret_cast<const unsigned char*>( &red[0] );
        }
        case RT_FORMAT_FLOAT:
        case RT_FORMAT_FLOAT2:
        case RT_FORMAT_FLOAT3:
        case RT_FORMAT_FLOAT4:
        {
            static const float red[4] = {1.0f, 0.0f, 0.0f, 1.0f};
            return reinterpret_cast<const unsigned char*>( &red[0] );
        }
        default:
        {
            static const unsigned char red[16] = {255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            return &red[0];
        }
    }
}

unsigned int BlockStamper::stampCharacter( char c, unsigned int x )
{
    static const char* const font[10][FONT_HEIGHT] = {
        // clang-format off
        { // 0
            " XXX ",
            "X   X",
            "X  XX",
            "X X X",
            "XX  X",
            "X   X",
            " XXX ",
        },
        { // 1
            "  X  ",
            " XX  ",
            "  X  ",
            "  X  ",
            "  X  ",
            "  X  ",
            "XXXXX",
        },
        { // 2
            " XXX ",
            "X   X",
            "    X",
            "  XX ",
            " X   ",
            "X    ",
            "XXXXX",
        },
        { // 3
            " XXX ",
            "X   X",
            "    X",
            "  XXX",
            "    X",
            "X   X",
            " XXX ",
        },
        { // 4
            "    X",
            "   XX",
            "  X X",
            " X  X",
            "XXXXX",
            "    X",
            "    X",
        },
        { // 5
            "XXXXX",
            "X    ",
            "XXXX ",
            "    X",
            "    X",
            "X   X",
            " XXX ",
        },
        { // 6
            "  XX ",
            " X   ",
            "X    ",
            "XXXX ",
            "X   X",
            "X   X",
            " XXX ",
        },
        { // 7
            "XXXXX",
            "    X",
            "   X ",
            "  X  ",
            " X   ",
            " X   ",
            " X   ",
        },
        { // 8
            " XXX ",
            "X   X",
            "X   X",
            " XXX ",
            "X   X",
            "X   X",
            " XXX ",
        },
        { // 9
            " XXX ",
            "X   X",
            "X   X",
            " XXXX",
            "    X",
            "   X ",
            " XX  ",
        },
        // clang-format on
    };

    unsigned char* const tileStart =
        static_cast<unsigned char*>( m_block.baseAddress ) + m_yOffset * m_block.rowPitch + x * m_bytesPerPixel;
    // Border scanline of 0.
    for( unsigned int col = 0; col < FONT_WIDTH + 2; ++col )
    {
        stampPixelSpan( m_blackPixel, &tileStart[col * m_bytesPerPixel] );
    }
    for( unsigned int row = 0; row < FONT_HEIGHT; ++row )
    {
        const char*          glyphRow = font[c - '0'][row];
        unsigned char* const scanline = tileStart + ( FONT_HEIGHT - row ) * m_block.rowPitch;

        stampPixelSpan( m_blackPixel, &scanline[0] );  // Border pixel of zero.
        for( unsigned int col = 0; col < FONT_WIDTH; ++col )
        {
            const unsigned char* pixel = glyphRow[col] == 'X' ? m_whitePixel : m_blackPixel;
            stampPixelSpan( pixel, &scanline[( col + 1 ) * m_bytesPerPixel] );
        }
        stampPixelSpan( m_blackPixel, &scanline[( FONT_WIDTH + 1 ) * m_bytesPerPixel] );  // Border pixel of zero.
    }
    // Border scanline of 0.
    for( unsigned int col = 0; col < FONT_WIDTH + 2; ++col )
    {
        stampPixelSpan( m_blackPixel, &tileStart[( FONT_HEIGHT + 1 ) * m_block.rowPitch + col * m_bytesPerPixel] );
    }
    return FONT_WIDTH + 2;
}

void BlockStamper::stampId( unsigned int id )
{
    const std::string digits = std::to_string( id );
    if( m_block.width < digits.size() * ( FONT_WIDTH + 2 ) || m_block.height < FONT_HEIGHT + 2 )
    {
        LOG_NORMAL( "Skipping stamp " << id << ", block dimensions " << m_block.width << 'x' << m_block.height
                                      << " too small\n" );
        return;
    }

    unsigned int x = m_xOffset;
    for( char c : digits )
    {
        x += stampCharacter( c, x );
    }
}

const unsigned char g_debugPattern[] = {0xDE, 0xAD, 0xBE, 0xEF, 0xFA, 0xCE, 0xF0, 0x0D};

void BlockStamper::stampDebugPattern()
{
    unsigned char* scanline = static_cast<unsigned char*>( m_block.baseAddress );
    for( unsigned int y = 0; y < m_block.height; ++y )
    {
        for( unsigned int x = 0; x < m_block.width * m_bytesPerPixel; ++x )
        {
            scanline[x] = g_debugPattern[x % sizeof( g_debugPattern )];
        }
        scanline += m_block.rowPitch;
    }
}

unsigned int BlockStamper::checkDebugPattern( std::vector<unsigned int>& byteOffsets,
                                              std::vector<unsigned int>& pixelCoords,
                                              unsigned int&              pixelCount )
{
    unsigned int   matchingBytes = 0;
    unsigned char* scanline      = static_cast<unsigned char*>( m_block.baseAddress );
    for( unsigned int y = 0; y < m_block.height; ++y )
    {
        for( unsigned int x = 0; x < m_block.width; ++x )
        {
            unsigned int pixelBytes = 0;
            for( unsigned int b = 0; b < m_bytesPerPixel; ++b )
            {
                if( scanline[x + b] == g_debugPattern[( x + b ) % sizeof( g_debugPattern )] )
                {
                    ++pixelBytes;
                    byteOffsets.push_back( y * m_block.rowPitch + x * m_bytesPerPixel + b );
                }
            }
            if( pixelBytes > 0 )
            {
                pixelCoords.push_back( m_block.x + x );
                pixelCoords.push_back( m_block.y + y );
            }
            if( pixelBytes == m_bytesPerPixel )
            {
                ++pixelCount;
            }
            matchingBytes += pixelBytes;
        }
        scanline += m_block.rowPitch;
    }
    return matchingBytes;
}

void BlockStamper::stampRed()
{
    const unsigned char* pixel    = getRedPixel();
    unsigned char*       scanline = static_cast<unsigned char*>( m_block.baseAddress );
    for( unsigned int y = 0; y < m_block.height; ++y )
    {
        stampPixelSpan( pixel, scanline, m_block.width );
        scanline += m_block.rowPitch;
    }
}

}  // namespace

void stampMemoryBlockWithId( const RTmemoryblock& block, unsigned int id, unsigned int xOffset, unsigned int yOffset )
{
    BlockStamper stamper( block, xOffset, yOffset );
    stamper.stampId( id );
}

void stampMemoryBlockWithDebugPattern( const RTmemoryblock& block )
{
    BlockStamper stamper( block, 0, 0 );
    stamper.stampDebugPattern();
}

unsigned int checkMemoryBlockForDebugPattern( const RTmemoryblock&       block,
                                              std::vector<unsigned int>& byteOffsets,
                                              std::vector<unsigned int>& pixelCoords,
                                              unsigned int&              pixelCount )
{
    BlockStamper stamper( block, 0, 0 );
    return stamper.checkDebugPattern( byteOffsets, pixelCoords, pixelCount );
}

void stampMemoryBlockWithRed( const RTmemoryblock& block )
{
    BlockStamper stamper( block, 0, 0 );
    return stamper.stampRed();
}

}  // namespace optix
