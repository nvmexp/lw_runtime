#pragma once
#include <vector>
#include "darkroom/Errors.h"

namespace darkroom
{
    class FFTChannelPersistentData
    {
    public:
        virtual ~FFTChannelPersistentData() = 0;
    };

    class FFTPersistentData
    {
    public:
        FFTChannelPersistentData * pPersistentDataRegular = nullptr;
        FFTChannelPersistentData * pPersistentDataHighres = nullptr;

        std::vector<unsigned char> regularPart;
        std::vector<unsigned char> highresPart;
        std::vector<unsigned char> outputPart;
    };

    void initFrequencyTransferProcessing(FFTPersistentData * pPersistentData);
    void deinitFrequencyTransferProcessing(FFTPersistentData * pPersistentData);

    // regularSizeW, regularSizeH - regular image width and height
    // highresSizeW, highresSizeH - highres image width and height
    // highresTileOffsetW, highresTileOffsetH, highresTileSizeW, highresTileSizeH specify a tile within highres image
    // outHighresTilePaddingOffsetW, outHighresTilePaddingOffsetH - pointers to resulting paddings on the beginning sides of the tile (e.g. left and bottom paddings)
    // outHighresTilePaddingSizeW, outHighresTilePaddingSizeH - pointers to resulting paddings on the ending sides of the tile (e.g. right and top paddings)
    void getFrequencyTransferTilePaddings(
        uint32_t regularSizeW, uint32_t regularSizeH,
        uint32_t highresSizeW, uint32_t highresSizeH,
        uint32_t highresTileOffsetW, uint32_t highresTileOffsetH,
        uint32_t highresTileSizeW, uint32_t highresTileSizeH,
        uint32_t * outHighresTilePaddingOffsetW, uint32_t * outHighresTilePaddingOffsetH, uint32_t * outHighresTilePaddingSizeW, uint32_t * outHighresTilePaddingSizeH);

    // regular - pointer to the start of the regular RGB/BGR image
    // highres - pointer to the start of the highres RGB/BGR image
    // Only one mode should be used for both regular and highres images - RGB or BGR.
    // highresTileOffsetW, highresTileOffsetH, highresTileSizeW, highresTileSizeH specify a tile within highres image
    // alpha - coefficient that determines how much spectrum interpolation to allow (0.0 - sharp cutoff, 1.0 - gradual blending)
    // highresTileFixed can be the same as highres or should point to the buffer of the same size
    template<typename T>
    darkroom::Error processFrequencyTransfer(FFTPersistentData * pPersistentData, const T * regular, uint32_t regularSizeW, uint32_t regularSizeH,
        T * highres, uint32_t highresSizeW, uint32_t highresSizeH, uint32_t highresTileOffsetW, uint32_t highresTileOffsetH, uint32_t highresTileSizeW, uint32_t highresTileSizeH,
        T * highresTileFixed, float alpha);
}
