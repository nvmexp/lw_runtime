
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

#pragma once

#include <optixu/optixpp_namespace.h>

#include <iosfwd>
#include <string>


#ifndef SRCTESTSAPI
#define SRCTESTSAPI
#endif

//-----------------------------------------------------------------------------
//
// Image class declaration.
//
//-----------------------------------------------------------------------------

namespace optix {

class Image
{
  public:
    SRCTESTSAPI Image() = default;

    // Initialize image from PPM file
    SRCTESTSAPI explicit Image( const std::string& filename );

    // Initialize image from optix buffer
    SRCTESTSAPI explicit Image( optix::Buffer buffer );

    // Initialize image from buffer
    SRCTESTSAPI Image( void* data, RTsize width, RTsize height, RTformat format );

    SRCTESTSAPI ~Image();

    SRCTESTSAPI void init( optix::Buffer buffer );
    SRCTESTSAPI void init( void* data, RTsize width, RTsize height, RTformat format );

    SRCTESTSAPI void readPPM( const std::string& filename );

    // Store image object to disk in PPM format (or PPM-like raw float format)
    SRCTESTSAPI bool writePPM( const std::string& filename, bool float_format = true );


    SRCTESTSAPI bool         failed() const { return _raster == nullptr; }
    SRCTESTSAPI unsigned int width() const { return _nx; }
    SRCTESTSAPI unsigned int height() const { return _ny; }
    SRCTESTSAPI float*       raster() const { return _raster; }

    SRCTESTSAPI static void compare( const Image& i0, const Image& i1, float tol, int& num_errors, float& avg_error, float& max_error );
    SRCTESTSAPI static void compare( const std::string& filename0,
                                     const std::string& filename1,
                                     float              tol,
                                     int&               num_errors,
                                     float&             avg_error,
                                     float&             max_error );

  private:
    unsigned int _nx      = 0u;
    unsigned int _ny      = 0u;
    unsigned int _max_val = 0u;
    float*       _raster  = nullptr;  // r,g,b triples

    Image( const Image& );  // forbidden
    static void getLine( std::ifstream& file_in, std::string& s );
};

}  // namespace optix
