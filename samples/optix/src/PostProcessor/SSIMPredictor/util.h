/******************************************************************************
 * Copyright 2019 LWPU Corporation. All rights reserved.
 *****************************************************************************/

// Internal utilities

#pragma once

#include <vector>

namespace LW {
namespace SSIM {

// exposed for debug purpose only
static inline unsigned int RoundUp( unsigned int n, unsigned int d )
{
    return ( n + d - 1 ) / d;
}

// integer rounded up to the next multiple
static inline unsigned int round_up( unsigned int n, unsigned int d = 32 )
{
    return d * ( ( n + d - 1 ) / d );
}

// colwert tensor data between nchw and nhwc
template <typename T>
void vec_nhwc_to_nchw( std::vector<T>& v, int width, int height )
{
    std::vector<T> cp( v );

    int c = int( v.size() / ( width * height ) );
    for( int k = 0; k < c; ++k )
    {
        for( int i = 0; i < width; ++i )
        {
            for( int j = 0; j < height; ++j )
            {
                int i_nchw = k * height * width + j * width + i;
                int i_nhwc = j * c * width + i * c + k;
                v[i_nchw]  = cp[i_nhwc];
            }
        }
    }
}


}  // namespace SSIM
}  // namespace LW
