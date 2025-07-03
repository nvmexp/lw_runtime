
/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
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

#include <exptest/Image.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>

Image::Image( const std::string& filename )
{
    readPPM( filename );
}

Image::Image( void* data, size_t width, size_t height, OptixPixelFormat format )
{
    init( data, width, height, format );
}

void Image::init( void* data, size_t width, size_t height, OptixPixelFormat format )
{
    delete[] m_data;

    m_nx   = static_cast<unsigned int>( width );
    m_ny   = static_cast<unsigned int>( height );
    m_data = new float[width * height * 3];

    // Colwert data to array of floats
    switch( format )
    {
        case OPTIX_PIXEL_FORMAT_FLOAT3:
        {
            float* fdata = reinterpret_cast<float*>( data );
            // This buffer is upside down
            for( int j = static_cast<int>( height ) - 1; j >= 0; --j )
            {
                float* dst = m_data + 3 * width * ( height - 1 - j );
                float* src = fdata + width * j;
                for( unsigned int i = 0; i < width; ++i )
                {
                    // write the pixel to all 3 channels
                    *dst++ = *src;
                    *dst++ = *src;
                    *dst++ = *src++;
                }
            }

            break;
        }
        case OPTIX_PIXEL_FORMAT_FLOAT4:
        {
            float* fdata = reinterpret_cast<float*>( data );
            // This buffer is upside down
            for( int j = static_cast<int>( height ) - 1; j >= 0; --j )
            {
                float* dst = m_data + 3 * width * ( height - 1 - j );
                float* src = fdata + 4 * width * j;
                for( unsigned int i = 0; i < width; ++i )
                {
                    for( int k = 0; k < 3; ++k )
                    {
                        *dst++ = *src++;
                    }
                    // skip alpha
                    ++src;
                }
            }
            break;
        }
        case OPTIX_PIXEL_FORMAT_UCHAR4:
        {
            // Data is BGRA and upside down, so we need to swizzle to RGB
            unsigned char* udata = reinterpret_cast<unsigned char*>( data );
            for( int j = static_cast<int>( height ) - 1; j >= 0; --j )
            {
                float*         dst = m_data + 3 * width * ( height - 1 - j );
                unsigned char* src = udata + ( 4 * width * j );
                for( unsigned int i = 0; i < width; i++ )
                {
                    *dst++ = static_cast<float>( *( src + 2 ) ) / 255.0f;
                    *dst++ = static_cast<float>( *( src + 1 ) ) / 255.0f;
                    *dst++ = static_cast<float>( *( src + 0 ) ) / 255.0f;
                    src += 4;  // skip alpha
                }
            }
            break;
        }
        default:
        {
            delete[] m_data;
            m_data = nullptr;
            std::cerr << "Image::Image( Buffer ) passed buffer with format other than OPTIX_PIXEL_FORMAT_FLOAT3, "
                         "OPTIX_PIXEL_FORMAT_FLOAT4, or OPTIX_PIXEL_FORMAT_UCHAR4"
                      << std::endl;
        }
    }
}

void Image::compare( const Image& i0, const Image& i1, float tol, int& num_errors, float& avg_error, float& max_error )
{
    if( i0.width() != i1.width() || i0.height() != i1.height() )
    {
        throw std::string( "Image::compare passed images of differing dimensions!" );
    }
    num_errors = 0;
    max_error  = 0.0f;
    avg_error  = 0.0f;
    for( unsigned int i = 0; i < i0.width() * i0.height(); ++i )
    {
        float error[3] = {
            fabsf( i0.m_data[3 * i + 0] - i1.m_data[3 * i + 0] ), fabsf( i0.m_data[3 * i + 1] - i1.m_data[3 * i + 1] ),
            fabsf( i0.m_data[3 * i + 2] - i1.m_data[3 * i + 2] ),
        };
        max_error = std::max( max_error, std::max( error[0], std::max( error[1], error[2] ) ) );
        avg_error += error[0] + error[1] + error[2];
        if( error[0] > tol || error[1] > tol || error[2] > tol )
            ++num_errors;
    }
    avg_error /= static_cast<float>( i0.width() * i0.height() * 3 );
}

void Image::compare( const std::string& filename0, const std::string& filename1, float tol, int& num_errors, float& avg_error, float& max_error )
{
    Image i0( filename0 );
    Image i1( filename1 );
    if( i0.failed() )
    {
        std::stringstream ss;
        ss << "Image::compare() failed to load image file '" << filename0 << "'";
        throw ss.str();
    }
    if( i1.failed() )
    {
        std::stringstream ss;
        ss << "Image::compare() failed to load image file '" << filename1 << "'";
        throw ss.str();
    }
    compare( Image( filename0 ), Image( filename1 ), tol, num_errors, avg_error, max_error );
}

bool Image::writePPM( const std::string& filename, bool float_format )
{
    try
    {

        std::ofstream out( filename.c_str(), std::ios::out | std::ios::binary );
        if( !out )
        {
            std::cerr << "Image::writePPM failed to open outfile '" << filename << "'" << std::endl;
            return false;
        }

        if( float_format )
        {

            out << "P7\n" << m_nx << " " << m_ny << "\n" << FLT_MAX << std::endl;
            out.write( reinterpret_cast<char*>( m_data ), m_nx * m_ny * 3 * sizeof( float ) );
        }
        else
        {
            out << "P6\n" << m_nx << " " << m_ny << "\n255" << std::endl;
            for( unsigned int i = 0; i < m_nx * m_ny * 3; ++i )
            {
                float         val  = m_data[i];
                unsigned char cval = val < 0.0f ? 0u : val > 1.0f ? 255u : static_cast<unsigned char>( val * 255.0f );
                out.put( cval );
            }
        }

        return true;
    }
    catch( ... )
    {
        std::cerr << "Failed to write ppm '" << filename << "'" << std::endl;
        return false;
    }
}

Image::~Image()
{
    if( m_data )
        delete[] m_data;
}

void Image::getLine( std::ifstream& file_in, std::string& s )
{
    for( ;; )
    {
        if( !std::getline( file_in, s ) )
            return;
        std::string::size_type index = s.find_first_not_of( "\n\r\t " );
        if( index != std::string::npos && s[index] != '#' )
            break;
    }
}

void Image::readPPM( const std::string& filename )
{
    delete[] m_data;
    m_data = nullptr;

    if( filename.empty() )
        return;

    // Open file
    try
    {
        std::ifstream file_in( filename.c_str(), std::ifstream::in | std::ifstream::binary );
        if( !file_in )
        {
            std::cerr << "Image( '" << filename << "' ) failed to open file." << std::endl;
            return;
        }

        // Check magic number to make sure we have an ascii or binary PPM
        std::string line, magic_number;
        getLine( file_in, line );
        std::istringstream iss1( line );
        iss1 >> magic_number;
        if( magic_number != "P6" && magic_number != "P3" && magic_number != "P7" )
        {
            std::cerr << "Image( '" << filename << "' ) unknown magic number: " << magic_number
                      << ".  Only P3, P6 and P7 supported." << std::endl;
            return;
        }

        // width, height
        getLine( file_in, line );
        std::istringstream iss2( line );
        iss2 >> m_nx >> m_ny;

        // max channel value
        getLine( file_in, line );
        std::istringstream iss3( line );
        iss3 >> m_maxVal;

        m_data = new float[m_nx * m_ny * 3];

        if( magic_number == "P3" )
        {
            unsigned int num_elements = m_nx * m_ny * 3;
            unsigned int count        = 0;

            while( count < num_elements )
            {
                getLine( file_in, line );
                std::istringstream iss( line );

                while( iss.good() )
                {
                    unsigned int c;
                    iss >> c;
                    m_data[count++] = static_cast<float>( c ) / 255.0f;
                }
            }
        }
        else if( magic_number == "P6" )
        {

            unsigned char* charm_data = new unsigned char[m_nx * m_ny * 3];
            file_in.read( reinterpret_cast<char*>( charm_data ), m_nx * m_ny * 3 );
            for( unsigned int i = 0u; i < m_nx * m_ny * 3; ++i )
            {
                m_data[i] = static_cast<float>( charm_data[i] ) / 255.0f;
            }
            delete[] charm_data;
        }
        else if( magic_number == "P7" )
        {

            file_in.read( reinterpret_cast<char*>( m_data ), m_nx * m_ny * 3 * sizeof( float ) );
        }
    }
    catch( ... )
    {
        std::cerr << "Image( '" << filename << "' ) failed to load" << std::endl;
        if( m_data )
            delete[] m_data;
        m_data = nullptr;
    }
}
