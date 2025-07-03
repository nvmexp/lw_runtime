#pragma once

#include <vector>
#include "Errors.h"

namespace darkroom
{
    /*
        Each captured tile has its own properties, they are captured in this structure
    */
    struct TileInfo
    {
        float pitch;
        float yaw;
        float horizontalFov;
        float blendFactor;
    };

    using LoadTilesEquirectCallback = void(*)(const std::vector<size_t>& tileNumbers);
    using UnloadTilesEquirectCallback = void(*)(const std::vector<size_t>& tileNumbers);

    /*
        - Tiles were captured using SPHERICAL_PANORAMA
        - Each tile should have the same dimensions
        - tiles points to an array of pointers to tiles of length 'tileCount'
        - tileInfo points to an array of TileInfo objects of length 'tileCount'
        - result pointer should point to enough memory for result image - its dimensions can be easily callwlated:
            resultWidth x resultHeight x 3 (BGR)
        - sphericalEquirect spawns 'std::thread::hardware_conlwrrency()' amount of threads if threadCount is 0
        - tileWidth, tileHeight all pointers and tileCount should be non zero
        - all pointers, tileCount, resultWidth/Height, tileWidth/Height shouldn't be zero
    */
    template<typename T>
    Error sphericalEquirect(const T* const * const tiles,
        LoadTilesEquirectCallback loadTilesCallback,
        UnloadTilesEquirectCallback unloadTilesCallback,
        const TileInfo* tileInfo,
        const unsigned int tileCount,
        const unsigned int tileWidth, const unsigned int tileHeight,
        T* const result,
        const unsigned int resultWidth, const unsigned int resultHeight,
        const unsigned int threadCount = 0);
}
