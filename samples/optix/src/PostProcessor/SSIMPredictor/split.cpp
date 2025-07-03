/******************************************************************************
 * Copyright 2018 LWPU Corporation. All rights reserved.
 *****************************************************************************/

#include "pch.h"

#include <algorithm>
#include <vector>


#include "common.h"
#include "util.h"

namespace LW {
namespace SSIM {

// Split image into vertical tiles (strips)
// RGB image buffer is given in data
// output image will be written into outData (outData != data)
// image resolution is given in width x height,
// number of tiles n_tiles

void Forward_base::split_image( int                        width,   // input image
                                int                        height,  // input image
                                int                        shrink_factor,
                                int                        tile_height,  // input tiles
                                std::vector<Forward_tile>& tiles )
{
    int overlap = tile_height < height ? Forward_base::m_overlap : 0;
    int h_up    = RoundUp( height, Forward_base::m_overlap ) * Forward_base::m_overlap;
    MI_ASSERT( overlap % shrink_factor == 0 );


    int inp_y = 0;
    int inp_h = tile_height + 2 * overlap;

    int copied = 0;
    while( copied < height )
    {
        int          yoffset = inp_y == 0 ? 0 : overlap;
        Forward_tile tilep;
        tilep.data_offset  = ( inp_y - yoffset ) * width;
        tilep.out_offset_h = ( inp_y / shrink_factor );

        tilep.copy_h = inp_y == 0 ? std::min( tile_height + overlap, h_up ) :
                                    copied + tile_height + overlap < height ? tile_height : h_up - copied;
        tilep.copy_h /= shrink_factor;  // in output scale
        tilep.data_h  = inp_y == 0 ? std::min( inp_h, height ) : std::min( inp_h, height + overlap - copied );
        tilep.yoffset = yoffset;

        inp_y += inp_y == 0 ? tile_height + overlap : tile_height;
        copied += tilep.copy_h * shrink_factor;

        tiles.push_back( tilep );
    }
}

}  // namespace SSIM
}  // namespace LW
