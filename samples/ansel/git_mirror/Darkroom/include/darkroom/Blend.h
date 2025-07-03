#pragma once
#include <vector>
#include <stdint.h>

#include "darkroom/Errors.h"
#include "darkroom/PixelFormat.h"

namespace darkroom
{
    using LoadTilesCallback = void(*)(const std::vector<size_t>& tileNumbers);
    using UnloadTilesCallback = void(*)(const std::vector<size_t>& tileNumbers);
    template<typename T>
    using RemapOutputCallback = T*(*)(uint64_t start, uint64_t end, void* lwrrentMapping);
    using ProgressCallback = void(*)(float);
    /*
      - Tiles were captured using HIGHRES mode
      - Because of that tileCount is a square of an odd natural number (9, 25, 49, 81, 121 are supported), which corresponds to a highresMultiplier of (2, 3, 4, 5, 6)
      - Each tile should have the same dimensions
      - Conselwtive tiles are overlapping exactly 50%
      - tiles points to an array of pointers to tiles of length 'tileCount'
      - tileWidth, tileHeight, tileCount and all pointers should be non zero

      In case conserveMemory is set to false:
      - result pointer should point to enough memory for result image - its dimensions can be easily callwlated:
      viewport dimensions x highresMultiplier x 3 (BGR)

      In case conserveMemory is set to true:

      - result should point to a bmp file data (memory mapped file or just memory) with bmpheader offset (54) added.
      This function does not fill the header, it only fills BGR data.

      In both cases result pointer could point to a memory mapped file

      The load/unloadTilesCallback callbacks are called whenever this function needs to load/unload tiles
      The callback is free to unload tiles that are no longer needed and load those which will be needed before the next call.
      See HighresBlender implementation as an example of such logic.
    */
    template<typename T>
    Error blendHighres(const T* const * const tiles,
        const unsigned int tileCount,
        const unsigned int tileWidth,
        const unsigned int tileHeight,
        T* result,
        const unsigned int outputPitch,
        LoadTilesCallback loadTilesCallback,
        UnloadTilesCallback unloadTilesCallback,
        RemapOutputCallback<T> remapCallback,
        ProgressCallback progressCallback,
        bool conserveMemory,
        BufferFormat format = BufferFormat::BGR8,
        const unsigned int threadCount = 0);

    /*
        Determines if number of tiles is valid for highres blending
    */
    Error isHighresAllowed(unsigned int shotsNum);
}
