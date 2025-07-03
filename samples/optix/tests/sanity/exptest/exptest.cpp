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

#include <exptest/exptest.h>
#include <optix_function_table_definition.h>
#include <optix_host.h>
#include <optix_stubs.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

OptixResult saveBufferPPM( const char* filename, size_t width, size_t height, OptixPixelFormat format, BufferOrientation orientation, void* buffer, bool disable_srgb_colwersion )
{
    if( !filename || width < 1 || height < 1 || !buffer )
        return OPTIX_ERROR_ILWALID_VALUE;

    const float gamma_ilw = 1.0f / 2.2f;

    std::vector<unsigned char> tmp( width * height * 3 );

    switch( format )
    {
        case OPTIX_PIXEL_FORMAT_UCHAR3:
        case OPTIX_PIXEL_FORMAT_UCHAR4:
        {

            unsigned char* src        = static_cast<unsigned char*>( buffer );
            unsigned char* dst        = &tmp[0];
            int            stride     = format == OPTIX_PIXEL_FORMAT_UCHAR3 ? 3 : 4;
            int            row_stride = 0;

            if( orientation == BUFFER_ORIENTATION_BOTTOM_UP )
            {
                src += stride * width * ( height - 1 );
                row_stride = -2 * static_cast<int>( width ) * stride;
            }

            for( int y = static_cast<int>( height ) - 1; y >= 0; --y )
            {
                for( size_t x = 0; x < width; ++x )
                {
                    *dst++ = *( src + 0 );
                    *dst++ = *( src + 1 );
                    *dst++ = *( src + 2 );
                    src += stride;
                }
                src += row_stride;
            }

            break;
        }

        case OPTIX_PIXEL_FORMAT_FLOAT3:
        case OPTIX_PIXEL_FORMAT_FLOAT4:
        {

            float*         src        = static_cast<float*>( buffer );
            unsigned char* dst        = &tmp[0];
            int            stride     = format == OPTIX_PIXEL_FORMAT_FLOAT3 ? 3 : 4;
            int            row_stride = 0;

            if( orientation == BUFFER_ORIENTATION_BOTTOM_UP )
            {
                src += stride * width * ( height - 1 );
                row_stride = -2 * static_cast<int>( width ) * stride;
            }

            for( int y = static_cast<int>( height ) - 1; y >= 0; --y )
            {
                for( size_t x = 0; x < width; ++x )
                {
                    for( int k = 0; k < 3; ++k )
                    {
                        int x;
                        if( disable_srgb_colwersion )
                            x = static_cast<int>( ( *( src + k ) * 255.0f ) );
                        else
                            x  = static_cast<int>( std::pow( *( src + k ), gamma_ilw ) * 255.0f );
                        *dst++ = static_cast<unsigned char>( x < 0 ? 0 : x > 0xff ? 0xff : x );
                    }
                    src += stride;
                }
                src += row_stride;
            }

            break;
        }

        default:
            return OPTIX_ERROR_ILWALID_VALUE;
    }

    std::ofstream file( filename, std::ios::out | std::ios::binary );
    if( !file.is_open() )
        return OPTIX_ERROR_UNKNOWN;

    file << "P6" << std::endl;
    file << width << " " << height << std::endl;
    file << 255 << std::endl;
    file.write( reinterpret_cast<char*>( &tmp[0] ), tmp.size() );
    // TODO error checking

    file.close();

    return OPTIX_SUCCESS;
}

class OptiXLoader
{
  public:
    OptiXLoader() { optixInit(); }
};

// Use a static global variable that will make sure the optix dll is loaded when exptest is loaded.
static OptiXLoader g_loader;
