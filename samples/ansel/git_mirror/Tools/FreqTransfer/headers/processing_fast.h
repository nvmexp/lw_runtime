#pragma once

enum class FreqTransferInterpShape
{
    kCIRCLE = 0,
    kRECTANGLE = 1,
    kELLIPSE = 2,
    
    kNUM_ENTRIES
};

void processDataFast(
    FFTChannelPersistentData * pPersistentDataRegular, float * regularData, int regularSizeW, int regularSizeH, double regularScale,
    FFTChannelPersistentData * pPersistentDataHighres, float * highresData, int highresSizeW, int highresSizeH,
    float * dataOut, float alpha, float maxFreq, FreqTransferInterpShape interpShapeType = FreqTransferInterpShape::kCIRCLE)
{
    std::vector<ChannelPair> channelPairsList;

    FFTChannel * regularR = nullptr;
    FFTChannel * regularG = nullptr;
    FFTChannel * regularB = nullptr;
#if (FFT_LIB == FFT_LIB_FFTW)
    regularR = new FFTChannelFFTW;
    regularG = new FFTChannelFFTW;
    regularB = new FFTChannelFFTW;
#elif (FFT_LIB == FFT_LIB_LWFFT)
    regularR = new FFTChannelLwFFT;
    regularG = new FFTChannelLwFFT;
    regularB = new FFTChannelLwFFT;
#else
    regularR = new FFTChannelKissFFT;
    regularG = new FFTChannelKissFFT;
    regularB = new FFTChannelKissFFT;
#endif

    regularR->setPersistentDataPointer(pPersistentDataRegular);
    regularR->setName("Regular.R");
    regularR->setSizes(regularSizeW, regularSizeH);
    regularG->setPersistentDataPointer(pPersistentDataRegular);
    regularG->setName("Regular.G");
    regularG->setSizes(regularSizeW, regularSizeH);
    regularB->setPersistentDataPointer(pPersistentDataRegular);
    regularB->setName("Regular.B");
    regularB->setSizes(regularSizeW, regularSizeH);

    FFTChannel * highresR = nullptr;
    FFTChannel * highresG = nullptr;
    FFTChannel * highresB = nullptr;
#if (FFT_LIB == FFT_LIB_FFTW)
    highresR = new FFTChannelFFTW;
    highresG = new FFTChannelFFTW;
    highresB = new FFTChannelFFTW;
#elif (FFT_LIB == FFT_LIB_LWFFT)
    highresR = new FFTChannelLwFFT;
    highresG = new FFTChannelLwFFT;
    highresB = new FFTChannelLwFFT;
#else
    highresR = new FFTChannelKissFFT;
    highresG = new FFTChannelKissFFT;
    highresB = new FFTChannelKissFFT;
#endif

    highresR->setPersistentDataPointer(pPersistentDataHighres);
    highresR->setName("HighRes.R");
    highresR->setSizes(highresSizeW, highresSizeH);
    highresG->setPersistentDataPointer(pPersistentDataHighres);
    highresG->setName("HighRes.G");
    highresG->setSizes(highresSizeW, highresSizeH);
    highresB->setPersistentDataPointer(pPersistentDataHighres);
    highresB->setName("HighRes.B");
    highresB->setSizes(highresSizeW, highresSizeH);

    channelPairsList.push_back(ChannelPair(regularR, highresR));
    channelPairsList.push_back(ChannelPair(regularG, highresG));
    channelPairsList.push_back(ChannelPair(regularB, highresB));

    const uint regular_channels = 3;
    const uint highres_channels = 3;

    size_t regularChannelSize = regularSizeW*regularSizeH;
    regularR->setDataPointer(regularData);
    regularG->setDataPointer(regularData + regularChannelSize);
    regularB->setDataPointer(regularData + 2*regularChannelSize);

    size_t highresChannelSize = highresSizeW*highresSizeH;
    highresR->setDataPointer(highresData);
    highresG->setDataPointer(highresData + highresChannelSize);
    highresB->setDataPointer(highresData + 2*highresChannelSize);
    highresR->setDataOutPointer(dataOut);
    highresG->setDataOutPointer(dataOut + highresChannelSize);
    highresB->setDataOutPointer(dataOut + 2*highresChannelSize);

#define DBG_SPECTRUM_SAVE 0
#if (DBG_SPECTRUM_SAVE == 1)
    static int tileNum = 0;
    wchar_t tileSpectrumName0[128];
    wchar_t tileSpectrumName1[128];
    wsprintf(tileSpectrumName0, L"tile%03d_spec0.bmp", tileNum);
    wsprintf(tileSpectrumName1, L"tile%03d_spec1.bmp", tileNum);
    ++tileNum;

    unsigned char * spectrumOutput0 = nullptr;
    unsigned char * spectrumOutput1 = nullptr;
#endif


#if (DBG_VERBOSE == 1)
    printf("Processing [%dx%d]:\n", highresSizeW, highresSizeH);
#endif
    Timer perfTimer;
    perfTimer.start();
    for (size_t ch = 0; ch < channelPairsList.size(); ++ch)
    {
        FFTChannel * regularCh = channelPairsList[ch].regular;
        FFTChannel * highresCh = channelPairsList[ch].highres;

        regularCh->init();
        highresCh->init();

#if (DBG_SPECTRUM_SAVE == 1)
        if (spectrumOutput0 == nullptr)
        {
            const size_t spectrumChannelSize = (int)highresR->getSpectrumWidthFFT() * (int)highresR->getHeightFFT();
            spectrumOutput0 = (unsigned char *)malloc(spectrumChannelSize * highres_channels * sizeof(unsigned char));
            spectrumOutput1 = (unsigned char *)malloc(spectrumChannelSize * highres_channels * sizeof(unsigned char));
        }
#endif

        regularCh->runFFT();
        highresCh->runFFT();

        const float regular_numElements = (float)regularCh->getWidthFFT()*regularCh->getHeightFFT();
        const float highres_numElements = (float)highresCh->getWidthFFT()*highresCh->getHeightFFT();
        regularCh->retrieveDataSpectrum();
        highresCh->retrieveDataSpectrum();

#if (DBG_SPECTRUM_SAVE == 1)
        const float highres_numElementsSq = sqrtf(highres_numElements);
        for (size_t i = 0; i < highresCh->getSpectrumWidthFFT(); ++i)
        {
            for (size_t j = 0; j < highresCh->getHeightFFT(); ++j)
            {
                size_t idx = (i+j*highresCh->getSpectrumWidthFFT())*3;
                spectrumOutput0[idx+ch] = getOutputSpectrumVal( getModulus(highresR->getSpectrumValOffset((int)i, (int)j)) / highres_numElementsSq );
            }
        }
#endif

        // Blend the two
        {
            struct blendingData
            {
                uint spectrumW, spectrumH;
                int centerI, centerJ;
            };

            blendingData blendRegular;
            blendRegular.spectrumW = regularCh->getSpectrumWidthFFT();
            blendRegular.spectrumH = regularCh->getHeightFFT();
            blendRegular.centerI = 0;
            blendRegular.centerJ = regularCh->getHeightFFT() / 2;

            blendingData blendHighres;
            blendHighres.spectrumW = highresCh->getSpectrumWidthFFT();
            blendHighres.spectrumH = highresCh->getHeightFFT();
            blendHighres.centerI = 0;
            blendHighres.centerJ = highresCh->getHeightFFT() / 2;

            assert(blendRegular.spectrumW <= blendHighres.spectrumW);
            assert(blendRegular.spectrumH <= blendHighres.spectrumH);

            // R2C FT width is half width already
            // TODO: add scaling?
            const int halfSizeW = blendRegular.spectrumW;
            const int halfSizeH = (blendRegular.spectrumH >> 1);

            // Assuming size(blendRegular) <= size(blendHighres)
            for (uint i = 0, iend = halfSizeW; i < iend; ++i)
            {
                for (uint j = 0, jend = halfSizeH; j < jend; ++j)
                {
                    //uint iRefl = blendRegular.spectrumW - i;
                    uint jRefl = (blendRegular.spectrumH - 1) - j;
                    //uint iReflHR = blendHighres.spectrumW - i;
                    uint jReflHR = (blendHighres.spectrumH - 1) - j;

                    // TODO: second half is just a reflection in our case

                    const double numElements = ((double)regularCh->getWidthFFT()*regularCh->getHeightFFT());
                    const double numElementsHR = ((double)highresCh->getWidthFFT()*highresCh->getHeightFFT());
                    double scale = numElementsHR / numElements;

                    // Distance callwlation is easier wrt center
                    int jShifted = j + halfSizeH;
                    if (jShifted < 0)
                        jShifted = j - blendRegular.spectrumH;

                    // FFT spectrum only has 1, 4 quadrants due to real-valued input
                    const int centerI = 0;
                    const int centerJ = halfSizeH;

                    // Alpha = 0.5 - aggressive preset
                    // Alpha = 0.75 - normal preset
                    // Alpha = 1.0 - weak preset
                    if (alpha == 0.0f)
                    {
                        alpha = 1e-6f;
                    }

                    bool applyInterpolation = false;
                    float interp = 1.0f;
                    if (interpShapeType == FreqTransferInterpShape::kCIRCLE)
                    {
                        const int filterSize = int(2*maxFreq*((halfSizeH < halfSizeW) ? halfSizeH : halfSizeW) / regularScale);
                        int dist = (int)sqrtf((float)sqr(i-centerI) + (float)sqr(jShifted-centerJ));
                        if (dist < filterSize)
                        {
                            applyInterpolation = true;
                            interp = (filterSize - dist) / float(alpha*(2*filterSize));
                        }
                    }
                    else if (interpShapeType == FreqTransferInterpShape::kRECTANGLE)
                    {
                        int filterDistW = abs((int)i - centerI);
                        int filterDistH = abs(jShifted - centerJ);

                        int filterSizeW = int(2*maxFreq*halfSizeW / regularScale);
                        int filterSizeH = int(2*maxFreq*halfSizeH / regularScale);

                        if ((filterDistW < filterSizeW) && (filterDistH < filterSizeH))
                        {
                            applyInterpolation = true;
                            float interpW = (filterSizeW - filterDistW) / float(alpha*2*filterSizeW);
                            float interpH = (filterSizeH - filterDistH) / float(alpha*2*filterSizeH);
                            interp = (interpW < interpH) ? interpW : interpH;
                        }
                    }
                    else if (interpShapeType == FreqTransferInterpShape::kELLIPSE)
                    {
                        int filterDistW = abs((int)i - centerI);
                        int filterDistH = abs(jShifted - centerJ);

                        int filterSizeW = int(2*maxFreq*halfSizeW / regularScale);
                        int filterSizeH = int(2*maxFreq*halfSizeH / regularScale);

                        float filterDistW_N = (filterDistW) / (float)filterSizeW;
                        float filterDistH_N = (filterDistH) / (float)filterSizeH;

                        float filterDist_N = sqrtf(filterDistW_N*filterDistW_N + filterDistH_N*filterDistH_N);

                        if (filterDist_N < 1.0f)
                        {
                            applyInterpolation = true;
                            interp = (1.0f - filterDist_N) / float(alpha*2);
                        }
                    }


                    if (applyInterpolation)
                    {
                        const float maxInterp = 1.0f;//lwtoffFreq;
                        if (interp > maxInterp)
                            interp = maxInterp;
                        if (interp > 1.0f)
                            interp = 1.0f;

                        // Cosine interpolation
                        //	it can be treated as special case of Hann window scaled by alpha
                        interp = 0.5f * (1 - cosf((float)M_PI * interp));

                        {
                            fComplex regularSpectrumVal = regularCh->getSpectrumVal(i, j);
                            fComplex highresSpectrumVal = highresCh->getSpectrumVal(i, j);

                            highresSpectrumVal = complexLerp(highresSpectrumVal, mul(scale, regularSpectrumVal), interp);
                            highresCh->updateSpectrumVal(i, j, highresSpectrumVal);
                        }
                        {
                            fComplex regularSpectrumVal = regularCh->getSpectrumVal(i, jRefl);
                            fComplex highresSpectrumVal = highresCh->getSpectrumVal(i, jReflHR);

                            highresSpectrumVal = complexLerp(highresSpectrumVal, mul(scale, regularSpectrumVal), interp);
                            highresCh->updateSpectrumVal(i, jReflHR, highresSpectrumVal);
                        }
                    }
                }
            }
        }

#if (DBG_SPECTRUM_SAVE == 1)
        for (size_t i = 0; i < highresCh->getSpectrumWidthFFT(); ++i)
        {
            for (size_t j = 0; j < highresCh->getHeightFFT(); ++j)
            {
                size_t idx = (i+j*highresCh->getSpectrumWidthFFT())*3;
                spectrumOutput1[idx+ch] = getOutputSpectrumVal( getModulus(highresR->getSpectrumValOffset((int)i, (int)j)) / highres_numElementsSq );
            }
        }
#endif

        regularCh->uploadDataSpectrum();
        highresCh->uploadDataSpectrum();

        highresCh->runIlwFFT();
        highresCh->getResult();

        regularCh->freeIntermediateData();
        highresCh->freeIntermediateData();
    }
    double elapsedTime = perfTimer.time();
#if (DBG_VERBOSE == 1)
    printf("done.\n");
#endif

#if (DBG_SPECTRUM_SAVE == 1)

    darkroom::saveBmp(spectrumOutput0, tileSpectrumName0, (int)highresR->getSpectrumWidthFFT(), (int)highresR->getHeightFFT(), darkroom::BufferFormat::BGR8);
    darkroom::saveBmp(spectrumOutput1, tileSpectrumName1, (int)highresR->getSpectrumWidthFFT(), (int)highresR->getHeightFFT(), darkroom::BufferFormat::BGR8);
    free(spectrumOutput0);
    free(spectrumOutput1);

#endif

#if (DBG_VERBOSE == 1)
    printf("processing time = %.2fms\n", elapsedTime);

    printf("Tearing down\n");
#endif
    for (size_t ch = 0; ch < channelPairsList.size(); ++ch)
    {
        channelPairsList[ch].regular->deinit();
        channelPairsList[ch].highres->deinit();
    }
#if (DBG_VERBOSE == 1)
    printf("done.\n");
#endif

    SAFE_DELETE(highresR);
    SAFE_DELETE(highresG);
    SAFE_DELETE(highresB);

    SAFE_DELETE(regularR);
    SAFE_DELETE(regularG);
    SAFE_DELETE(regularB);
}

template <typename T>
void copyFixedData(
    float * src, uint tilePaddedSizeW, uint tilePaddedSizeH, uint tilePaddingOffsetW, uint tilePaddingOffsetH, uint tileDataSizeW, uint tileDataSizeH,
    T * dst, uint highresSizeW, uint highresSizeH, uint highresOffsetW, uint highresOffsetH
    )
{
    assert(false);
    return;
}

template <>
void copyFixedData<unsigned char>(
        float * src, uint tilePaddedSizeW, uint tilePaddedSizeH, uint tilePaddingOffsetW, uint tilePaddingOffsetH, uint tileDataSizeW, uint tileDataSizeH,
        unsigned char * dst, uint highresSizeW, uint highresSizeH, uint highresOffsetW, uint highresOffsetH
        )
{
    size_t tileOutChannelSize = tilePaddedSizeW*tilePaddedSizeH;

    float * tileOutR = src;
    float * tileOutG = tileOutR+tileOutChannelSize;
    float * tileOutB = tileOutG+tileOutChannelSize;

    const uint numChannels = 3;
    for (int x = 0; x < (int)tileDataSizeW; ++x)
    {
        int padX = x + tilePaddingOffsetW;
        for (int y = 0; y < (int)tileDataSizeH; ++y)
        {
            int padY = y + tilePaddingOffsetH;
            dst[(x+highresOffsetW + (y+highresOffsetH)*highresSizeW)*numChannels+2] = saturate_uc(tileOutB[padX+padY*tilePaddedSizeW]);
            dst[(x+highresOffsetW + (y+highresOffsetH)*highresSizeW)*numChannels+1] = saturate_uc(tileOutG[padX+padY*tilePaddedSizeW]);
            dst[(x+highresOffsetW + (y+highresOffsetH)*highresSizeW)*numChannels  ] = saturate_uc(tileOutR[padX+padY*tilePaddedSizeW]);
        }
    }
}

template <>
void copyFixedData<float>(
    float * src, uint tilePaddedSizeW, uint tilePaddedSizeH, uint tilePaddingOffsetW, uint tilePaddingOffsetH, uint tileDataSizeW, uint tileDataSizeH,
    float * dst, uint highresSizeW, uint highresSizeH, uint highresOffsetW, uint highresOffsetH
    )
{
    size_t tileOutChannelSize = tilePaddedSizeW*tilePaddedSizeH;

    float * tileOutR = src;
    float * tileOutG = tileOutR+tileOutChannelSize;
    float * tileOutB = tileOutG+tileOutChannelSize;

    const uint numChannels = 3;
    for (int x = 0; x < (int)tileDataSizeW; ++x)
    {
        int padX = x + tilePaddingOffsetW;
        for (int y = 0; y < (int)tileDataSizeH; ++y)
        {
            int padY = y + tilePaddingOffsetH;
            dst[(x+highresOffsetW + (y+highresOffsetH)*highresSizeW)*numChannels+2] = tileOutB[padX+padY*tilePaddedSizeW];
            dst[(x+highresOffsetW + (y+highresOffsetH)*highresSizeW)*numChannels+1] = tileOutG[padX+padY*tilePaddedSizeW];
            dst[(x+highresOffsetW + (y+highresOffsetH)*highresSizeW)*numChannels  ] = tileOutR[padX+padY*tilePaddedSizeW];
        }
    }
}

/*
    This function performs FFT processing of a single tile. The function performs necessary padding and data copying,
    the actual FFT is happening in the processData* function call.

    This function upscales regular (downscaled) image, which means that efficiency padding could be implemented,
    as scaled up regular tile and highres tile maintain 1-to-1 pixel mapping, allowing for arbitrary paddings, not
    constrained by the multipliers.

    This makes the function more memory hungry than non-scaling alternative, and slower for the factor-independent
    FFT implementations; as well as for best-case scenarios of factor-dependent FFTs. However, for factor-dependent
    FFTs, this function provides consistent timing, without significant performance drops when primes are factors
    of the input size (sometimes dropping factor-dependent FFTs performance more ~20x), which makes this function
    better suitable for work with factor-dependent FFTs.

    alpha is the interpolation/blending parameter (the higher it is - the less affect transfer will have)
    maxFreq is the frequency transfer distance, default is 0.5 (Nyquist freq)
    interpShapeType is the area shape of transferred spectrum, rectangle - biggest area for certain
    maxFreq, circle - safest transfer, although requires nearly 1:1 aspect of the tiles
*/
template <typename T>
void processTilingFast(
    FFTPersistentData * pPersistentData,
    T * regular, int regularSizeW, int regularSizeH,
    T * highres, int highresSizeW, int highresSizeH,
    int highresTileOffsetW, int highresTileOffsetH, int highresTileSizeW, int highresTileSizeH,
    T * highresTileFixed, float alpha, float maxFreq,
    FreqTransferInterpShape interpShapeType = FreqTransferInterpShape::kCIRCLE
)
{
    const uint numChannels = 3;

    // Tiling
    // Lwrrently regular has symmetrical padding
    //	and highres could have asymmetrical padding due to potential sub-regular-pixel offsets
    const uint tilePaddingInRegular = 8;

    // Callwlate tile size
    const uint tileOutW = highresTileSizeW;
    const uint tileOutH = highresTileSizeH;

    uint highresMultW = (uint)ceil(highresSizeW / (float)regularSizeW);
    uint highresMultH = (uint)ceil(highresSizeH / (float)regularSizeH);

    double regularScale = highresMultW;

    // Regular sizes/offsets callwlation
    ///////////////////////////////////////////////////////////////////////////////
    // Callwlating regular tile box
    //	the box should be callwlated taking into account that highres tile could have a box
    //	that doesn't directly maps onto regular pixel grid, thus we need to be conservative
    const uint regularTileOutOffsetW = (uint)floor(highresTileOffsetW / (float)highresMultW);
    const uint regularTileOutOffsetH = (uint)floor(highresTileOffsetH / (float)highresMultH);
    const uint regularTileOutSizeW = (uint)ceil((highresTileSizeW + highresTileOffsetW) / (float)highresMultW) - regularTileOutOffsetW;
    const uint regularTileOutSizeH = (uint)ceil((highresTileSizeH + highresTileOffsetH) / (float)highresMultH) - regularTileOutOffsetH;

    // Paddings for the tile in regular space
    uint regularTilePaddingOffsetW = tilePaddingInRegular;
    uint regularTilePaddingOffsetH = tilePaddingInRegular;
    uint regularTilePaddingSizeW = tilePaddingInRegular;
    uint regularTilePaddingSizeH = tilePaddingInRegular;

    // We want to make our regular padded tile offset&size even
    //	this will guarantee that highres tile will also be even
    //	this subsequently will guarantee that none of the FFT libs will fail
    if ((regularTileOutOffsetW - regularTilePaddingOffsetW)&1)
    {
        ++regularTilePaddingOffsetW;
    }
    if ((regularTileOutOffsetH - regularTilePaddingOffsetH)&1)
    {
        ++regularTilePaddingOffsetH;
    }
    if ((regularTileOutSizeW + regularTilePaddingOffsetW + regularTilePaddingSizeW)&1)
    {
        ++regularTilePaddingSizeW;
    }
    if ((regularTileOutSizeH + regularTilePaddingOffsetH + regularTilePaddingSizeH)&1)
    {
        ++regularTilePaddingSizeH;
    }

    const uint regularTileOutWPad = regularTileOutSizeW + regularTilePaddingOffsetW + regularTilePaddingSizeW;
    const uint regularTileOutHPad = regularTileOutSizeH + regularTilePaddingOffsetH + regularTilePaddingSizeH;

    // Highres sizes/offsets callwlation
    ///////////////////////////////////////////////////////////////////////////////
    // In case highres tile doesn't fit onto pixel borders f regular, we need additional padding
    //	this padding could be different on each edge of the highres tile
    const int highresAdditionalPaddingOffsetW = highresTileOffsetW - regularTileOutOffsetW * highresMultW;
    const int highresAdditionalPaddingOffsetH = highresTileOffsetH - regularTileOutOffsetH * highresMultH;
    const int highresAdditionalPaddingSizeW = (regularTileOutSizeW+regularTileOutOffsetW) * highresMultW - (highresTileSizeW+highresTileOffsetW);
    const int highresAdditionalPaddingSizeH = (regularTileOutSizeH+regularTileOutOffsetH) * highresMultH - (highresTileSizeH+highresTileOffsetH);

    assert(highresAdditionalPaddingOffsetW >= 0);
    assert(highresAdditionalPaddingOffsetH >= 0);
    assert(highresAdditionalPaddingSizeW >= 0);
    assert(highresAdditionalPaddingSizeH >= 0);

    uint highresTilePaddingOffsetW = regularTilePaddingOffsetW*highresMultW + highresAdditionalPaddingOffsetW;
    uint highresTilePaddingOffsetH = regularTilePaddingOffsetH*highresMultH + highresAdditionalPaddingOffsetH;
    uint highresTilePaddingSizeW = regularTilePaddingSizeW*highresMultW + highresAdditionalPaddingSizeW;
    uint highresTilePaddingSizeH = regularTilePaddingSizeH*highresMultH + highresAdditionalPaddingSizeH;

    uint tileOutWPad = tileOutW + highresTilePaddingOffsetW + highresTilePaddingSizeW;
    uint tileOutHPad = tileOutH + highresTilePaddingOffsetH + highresTilePaddingSizeH;

    uint highresEffectivenessPaddingW = 0;
    uint highresEffectivenessPaddingH = 0;
    bool isEffectivenessPaddingAllowed = true;
    if (isEffectivenessPaddingAllowed)
    {
#if (FFT_LIB == FFT_LIB_FFTW)
        // TODO:
        highresEffectivenessPaddingW = 0;
        highresEffectivenessPaddingH = 0;
#elif (FFT_LIB == FFT_LIB_LWFFT)
        // TODO:
        highresEffectivenessPaddingW = 0;
        highresEffectivenessPaddingH = 0;
#else
        highresEffectivenessPaddingW = kiss_fftr_next_fast_size_real(tileOutWPad) - tileOutWPad;
        highresEffectivenessPaddingH = kiss_fftr_next_fast_size_real(tileOutHPad) - tileOutHPad;
#endif

#if 0
        // Black padding
#else
        // Same padding
        highresTilePaddingSizeW += highresEffectivenessPaddingW;
        highresTilePaddingSizeH += highresEffectivenessPaddingH;
        highresEffectivenessPaddingW = 0;
        highresEffectivenessPaddingH = 0;
#endif

        tileOutWPad = tileOutW + highresTilePaddingOffsetW + highresTilePaddingSizeW + highresEffectivenessPaddingW;
        tileOutHPad = tileOutH + highresTilePaddingOffsetH + highresTilePaddingSizeH + highresEffectivenessPaddingH;
    }

    size_t tileOutChannelSize = tileOutWPad*tileOutHPad;

    pPersistentData->highresPart.resize(tileOutChannelSize*numChannels*sizeof(float));
    float * highresPart = reinterpret_cast<float *>(&pPersistentData->highresPart[0]);

    memset(highresPart, 0, tileOutChannelSize*numChannels*sizeof(float));

    float * highresPartR = highresPart;
    float * highresPartG = highresPartR+tileOutChannelSize;
    float * highresPartB = highresPartG+tileOutChannelSize;

    // Output 
    pPersistentData->outputPart.resize(tileOutChannelSize*numChannels*sizeof(float));
    float * tileOut = reinterpret_cast<float *>(&pPersistentData->outputPart[0]);
    memset(tileOut, 0, tileOutChannelSize*numChannels*sizeof(float));

    // Upscaled regular tile allocation
    ///////////////////////////////////////////////////////////////////////////////
    pPersistentData->regularPart.resize(tileOutChannelSize*numChannels*sizeof(float));
    float * regularPart = reinterpret_cast<float *>(&pPersistentData->regularPart[0]);
    float * regularPartR = regularPart;
    float * regularPartG = regularPartR+tileOutChannelSize;
    float * regularPartB = regularPartG+tileOutChannelSize;

#define PADDING_SAME		1
#define PADDING_MIRROR		2
#define PADDING_OVERLAP		3
#define PADDING_TYPE		PADDING_OVERLAP


    // Copying data and introduce padding
    ///////////////////////////////////////////////////////////////////////////////

#if (DBG_VERBOSE == 1)
    printf("Copying the input data to channels\n");
#endif
    // Copy highres part
#if (PADDING_TYPE == PADDING_OVERLAP)
    for (int x = -(int)highresTilePaddingOffsetW; x < (int)(tileOutW + highresTilePaddingSizeW); ++x)
    {
        for (int y = -(int)highresTilePaddingOffsetH; y < (int)(tileOutH + highresTilePaddingSizeH); ++y)
        {
            int padX = x + highresTilePaddingOffsetW;
            int padY = y + highresTilePaddingOffsetH;

            int imageX = x + highresTileOffsetW;
            int imageY = y + highresTileOffsetH;

            if (imageX < 0) imageX = 0;
            if (imageX >= highresSizeW) imageX = highresSizeW - 1;

            if (imageY < 0) imageY = 0;
            if (imageY >= highresSizeH) imageY = highresSizeH - 1;

            highresPartB[padX+padY*tileOutWPad] = highres[(imageX + imageY*highresSizeW)*numChannels+2];
            highresPartG[padX+padY*tileOutWPad] = highres[(imageX + imageY*highresSizeW)*numChannels+1];
            highresPartR[padX+padY*tileOutWPad] = highres[(imageX + imageY*highresSizeW)*numChannels  ];
        }
    }
#else
    for (uint x = 0; x < tileOutW; ++x)
    {
        for (uint y = 0; y < tileOutH; ++y)
        {
            int padX = x + highresTilePaddingOffsetW;
            int padY = y + highresTilePaddingOffsetH;
            highresPartB[padX+padY*tileOutWPad] = highres[(x+highresTileOffsetW + (y+highresTileOffsetH)*highresSizeW)*numChannels+2];
            highresPartG[padX+padY*tileOutWPad] = highres[(x+highresTileOffsetW + (y+highresTileOffsetH)*highresSizeW)*numChannels+1];
            highresPartR[padX+padY*tileOutWPad] = highres[(x+highresTileOffsetW + (y+highresTileOffsetH)*highresSizeW)*numChannels  ];
        }
    }
#endif

    uint tileOutWPad_noEffPad = tileOutWPad - highresEffectivenessPaddingW;
    uint tileOutHPad_noEffPad = tileOutHPad - highresEffectivenessPaddingH;

    // Fill in the padding in highres tile, padding is asymmetric so we cannot unify offset/size paddings
    // Offset-height padding
    for (uint x = 0; x < tileOutWPad_noEffPad; ++x)
    {
        for (uint y = 0; y < highresTilePaddingOffsetH; ++y)
        {
#if (PADDING_TYPE == PADDING_SAME)
            uint wrapIdx, wrapX = x;

            if (wrapX < highresTilePaddingOffsetW)
                wrapX = highresTilePaddingOffsetW;
            else if (wrapX > tileOutWPad - 1 - highresTilePaddingSizeW - highresEffectivenessPaddingW)
                wrapX = tileOutWPad - 1 - highresTilePaddingSizeW - highresEffectivenessPaddingW;

            wrapIdx = wrapX+highresTilePaddingOffsetH*tileOutWPad;
            highresPartB[x+y*tileOutWPad] = highresPartB[wrapIdx];
            highresPartG[x+y*tileOutWPad] = highresPartG[wrapIdx];
            highresPartR[x+y*tileOutWPad] = highresPartR[wrapIdx];
#elif (PADDING_TYPE == PADDING_MIRROR)
            uint wrapIdx, wrapX = x;

            if (wrapX < highresTilePaddingOffsetW)
                wrapX = 2 * highresTilePaddingOffsetW - wrapX;
            else if (wrapX > tileOutWPad_noEffPad - 1 - highresTilePaddingSizeW)
                wrapX = 2 * (tileOutWPad_noEffPad - 1 - highresTilePaddingSizeW) - wrapX;

            wrapIdx = wrapX+(2 * highresTilePaddingOffsetH - y)*tileOutWPad;
            highresPartB[x+y*tileOutWPad] = highresPartB[wrapIdx];
            highresPartG[x+y*tileOutWPad] = highresPartG[wrapIdx];
            highresPartR[x+y*tileOutWPad] = highresPartR[wrapIdx];
#endif
        }
    }
    // Size-height padding
    for (uint x = 0; x < tileOutWPad_noEffPad; ++x)
    {
        for (uint y = 0; y < highresTilePaddingSizeH; ++y)
        {
#if (PADDING_TYPE == PADDING_SAME)
            uint wrapIdx, wrapX = x;

            if (wrapX < highresTilePaddingOffsetW)
                wrapX = highresTilePaddingOffsetW;
            else if (wrapX > tileOutWPad - 1 - highresTilePaddingSizeW - highresEffectivenessPaddingW)
                wrapX = tileOutWPad - 1 - highresTilePaddingSizeW - highresEffectivenessPaddingW;

            wrapIdx = wrapX+(tileOutHPad - 1 - highresTilePaddingSizeH - highresEffectivenessPaddingH)*tileOutWPad;
            highresPartB[x+(tileOutHPad - 1 - y - highresEffectivenessPaddingH)*tileOutWPad] = highresPartB[wrapIdx];
            highresPartG[x+(tileOutHPad - 1 - y - highresEffectivenessPaddingH)*tileOutWPad] = highresPartG[wrapIdx];
            highresPartR[x+(tileOutHPad - 1 - y - highresEffectivenessPaddingH)*tileOutWPad] = highresPartR[wrapIdx];
#elif (PADDING_TYPE == PADDING_MIRROR)
            uint wrapIdx, wrapX = x;

            if (wrapX < highresTilePaddingOffsetW)
                wrapX = 2 * highresTilePaddingOffsetW - wrapX;
            else if (wrapX > tileOutWPad_noEffPad - 1 - highresTilePaddingSizeW)
                wrapX = 2 * (tileOutWPad_noEffPad - 1 - highresTilePaddingSizeW) - wrapX;

            // No need to mul by 2 here as y = (y_real - size_y)
            wrapIdx = wrapX+((tileOutHPad_noEffPad - 1 - highresTilePaddingSizeH) - (highresTilePaddingSizeH - 1 - y))*tileOutWPad;
            highresPartB[x+(tileOutHPad_noEffPad - 1 - y)*tileOutWPad] = highresPartB[wrapIdx];
            highresPartG[x+(tileOutHPad_noEffPad - 1 - y)*tileOutWPad] = highresPartG[wrapIdx];
            highresPartR[x+(tileOutHPad_noEffPad - 1 - y)*tileOutWPad] = highresPartR[wrapIdx];
#endif
        }
    }

    // Offset-width padding
    for (uint x = 0; x < highresTilePaddingOffsetW; ++x)
    {
        for (uint y = highresTilePaddingOffsetH; y < tileOutHPad - highresTilePaddingSizeH - highresEffectivenessPaddingH; ++y)
        {
#if (PADDING_TYPE == PADDING_SAME)
            uint wrapIdx, wrapY = y;

            wrapIdx = highresTilePaddingOffsetW+wrapY*tileOutWPad;
            highresPartB[x+y*tileOutWPad] = highresPartB[wrapIdx];
            highresPartG[x+y*tileOutWPad] = highresPartG[wrapIdx];
            highresPartR[x+y*tileOutWPad] = highresPartR[wrapIdx];
#elif (PADDING_TYPE == PADDING_MIRROR)
            uint wrapIdx, wrapY = y;

            wrapIdx = (2*highresTilePaddingOffsetW-x)+wrapY*tileOutWPad;
            highresPartB[x+y*tileOutWPad] = highresPartB[wrapIdx];
            highresPartG[x+y*tileOutWPad] = highresPartG[wrapIdx];
            highresPartR[x+y*tileOutWPad] = highresPartR[wrapIdx];
#endif
        }
    }
    // Size-width padding
    for (uint x = 0; x < highresTilePaddingSizeW; ++x)
    {
        for (uint y = highresTilePaddingOffsetH; y < tileOutHPad - highresTilePaddingSizeH - highresEffectivenessPaddingH; ++y)
        {
#if (PADDING_TYPE == PADDING_SAME)
            uint wrapIdx, wrapY = y;

            wrapIdx = (tileOutWPad - 1 - highresTilePaddingSizeW - highresEffectivenessPaddingW)+wrapY*tileOutWPad;
            highresPartB[(tileOutWPad - 1 - x - highresEffectivenessPaddingW)+y*tileOutWPad] = highresPartB[wrapIdx];
            highresPartG[(tileOutWPad - 1 - x - highresEffectivenessPaddingW)+y*tileOutWPad] = highresPartG[wrapIdx];
            highresPartR[(tileOutWPad - 1 - x - highresEffectivenessPaddingW)+y*tileOutWPad] = highresPartR[wrapIdx];
#elif (PADDING_TYPE == PADDING_MIRROR)
            uint wrapIdx, wrapY = y;

            // No need to mul by 2 here as x = (x_real - size_x)
            wrapIdx = ((tileOutWPad_noEffPad - 1 - highresTilePaddingSizeW)-(highresTilePaddingSizeW - 1 - x))+wrapY*tileOutWPad;
            highresPartB[(tileOutWPad_noEffPad - 1 - x)+y*tileOutWPad] = highresPartB[wrapIdx];
            highresPartG[(tileOutWPad_noEffPad - 1 - x)+y*tileOutWPad] = highresPartG[wrapIdx];
            highresPartR[(tileOutWPad_noEffPad - 1 - x)+y*tileOutWPad] = highresPartR[wrapIdx];
#endif
        }
    }

    // Copy regular part
#if (PADDING_TYPE == PADDING_OVERLAP)
    for (int x = -(int)highresTilePaddingOffsetW; x < (int)(tileOutW + highresTilePaddingSizeW); ++x)
    {
        for (int y = -(int)highresTilePaddingOffsetH; y < (int)(tileOutH + highresTilePaddingSizeH); ++y)
        {
            int padX = x + highresTilePaddingOffsetW;
            int padY = y + highresTilePaddingOffsetH;

            int imageX = x + highresTileOffsetW;
            int imageY = y + highresTileOffsetH;

            if (imageX < 0) imageX = 0;
            if (imageX >= highresSizeW) imageX = highresSizeW - 1;

            if (imageY < 0) imageY = 0;
            if (imageY >= highresSizeH) imageY = highresSizeH - 1;

            int regularIdx = (imageX/highresMultW + (imageY/highresMultH)*regularSizeW);

            regularPartB[padX+padY*tileOutWPad] = regular[regularIdx*numChannels+2];
            regularPartG[padX+padY*tileOutWPad] = regular[regularIdx*numChannels+1];
            regularPartR[padX+padY*tileOutWPad] = regular[regularIdx*numChannels  ];
        }
    }
#else
    for (uint x = 0; x < tileOutW; ++x)
    {
        for (uint y = 0; y < tileOutH; ++y)
        {
            int padX = x + highresTilePaddingOffsetW;
            int padY = y + highresTilePaddingOffsetH;

#if (PADDING_TYPE == PADDING_SAME)
            int regularIdx = ((x+highresTileOffsetW)/highresMultW + ((y+highresTileOffsetH)/highresMultH)*regularSizeW);

            regularPartB[padX+padY*tileOutWPad] = regular[regularIdx*numChannels+2];
            regularPartG[padX+padY*tileOutWPad] = regular[regularIdx*numChannels+1];
            regularPartR[padX+padY*tileOutWPad] = regular[regularIdx*numChannels  ];
#elif (PADDING_TYPE == PADDING_MIRROR)
            int regularIdx = ((x+highresTileOffsetW)/highresMultW + ((y+highresTileOffsetH)/highresMultH)*regularSizeW);

            regularPartB[padX+padY*tileOutWPad] = regular[regularIdx*numChannels+2];
            regularPartG[padX+padY*tileOutWPad] = regular[regularIdx*numChannels+1];
            regularPartR[padX+padY*tileOutWPad] = regular[regularIdx*numChannels  ];
#endif
        }
    }
#endif

    // Fill in the padding in highres tile, padding is asymmetric so we cannot unify offset/size paddings
    // Offset-height padding
    for (uint x = 0; x < tileOutWPad_noEffPad; ++x)
    {
        for (uint y = 0; y < highresTilePaddingOffsetH; ++y)
        {
#if (PADDING_TYPE == PADDING_SAME)
            uint wrapIdx, wrapX = x;

            if (wrapX < highresTilePaddingOffsetW)
                wrapX = highresTilePaddingOffsetW;
            else if (wrapX > tileOutWPad - 1 - highresTilePaddingSizeW - highresEffectivenessPaddingW)
                wrapX = tileOutWPad - 1 - highresTilePaddingSizeW - highresEffectivenessPaddingW;

            wrapIdx = wrapX+highresTilePaddingOffsetH*tileOutWPad;
            regularPartB[x+y*tileOutWPad] = regularPartB[wrapIdx];
            regularPartG[x+y*tileOutWPad] = regularPartG[wrapIdx];
            regularPartR[x+y*tileOutWPad] = regularPartR[wrapIdx];
#elif (PADDING_TYPE == PADDING_MIRROR)
            uint wrapIdx, wrapX = x;

            if (wrapX < highresTilePaddingOffsetW)
                wrapX = 2 * highresTilePaddingOffsetW - wrapX;
            else if (wrapX > tileOutWPad_noEffPad - 1 - highresTilePaddingSizeW)
                wrapX = 2 * (tileOutWPad_noEffPad - 1 - highresTilePaddingSizeW) - wrapX;

            wrapIdx = wrapX+(2*highresTilePaddingOffsetH - y)*tileOutWPad;
            regularPartB[x+y*tileOutWPad] = regularPartB[wrapIdx];
            regularPartG[x+y*tileOutWPad] = regularPartG[wrapIdx];
            regularPartR[x+y*tileOutWPad] = regularPartR[wrapIdx];
#endif
        }
    }
    // Size-height padding
    for (uint x = 0; x < tileOutWPad_noEffPad; ++x)
    {
        for (uint y = 0; y < highresTilePaddingSizeH; ++y)
        {
#if (PADDING_TYPE == PADDING_SAME)
            uint wrapIdx, wrapX = x;

            if (wrapX < highresTilePaddingOffsetW)
                wrapX = highresTilePaddingOffsetW;
            else if (wrapX > tileOutWPad - 1 - highresTilePaddingSizeW - highresEffectivenessPaddingW)
                wrapX = tileOutWPad - 1 - highresTilePaddingSizeW - highresEffectivenessPaddingW;

            wrapIdx = wrapX+(tileOutHPad - 1 - highresTilePaddingSizeH - highresEffectivenessPaddingH)*tileOutWPad;
            regularPartB[x+(tileOutHPad - 1 - y - highresEffectivenessPaddingH)*tileOutWPad] = regularPartB[wrapIdx];
            regularPartG[x+(tileOutHPad - 1 - y - highresEffectivenessPaddingH)*tileOutWPad] = regularPartG[wrapIdx];
            regularPartR[x+(tileOutHPad - 1 - y - highresEffectivenessPaddingH)*tileOutWPad] = regularPartR[wrapIdx];
#elif (PADDING_TYPE == PADDING_MIRROR)
            uint wrapIdx, wrapX = x;

            if (wrapX < highresTilePaddingOffsetW)
                wrapX = 2 * highresTilePaddingOffsetW - wrapX;
            else if (wrapX > tileOutWPad_noEffPad - 1 - highresTilePaddingSizeW)
                wrapX = 2 * (tileOutWPad_noEffPad - 1 - highresTilePaddingSizeW) - wrapX;

            // No need to mul by 2 here as y = (y_real - size_y)
            wrapIdx = wrapX+((tileOutHPad_noEffPad - 1 - highresTilePaddingSizeH) - (highresTilePaddingSizeH - 1 - y))*tileOutWPad;
            regularPartB[x+(tileOutHPad_noEffPad - 1 - y)*tileOutWPad] = regularPartB[wrapIdx];
            regularPartG[x+(tileOutHPad_noEffPad - 1 - y)*tileOutWPad] = regularPartG[wrapIdx];
            regularPartR[x+(tileOutHPad_noEffPad - 1 - y)*tileOutWPad] = regularPartR[wrapIdx];
#endif
        }
    }

    // Offset-width padding
    for (uint x = 0; x < highresTilePaddingOffsetW; ++x)
    {
        for (uint y = highresTilePaddingOffsetH; y < tileOutHPad - highresTilePaddingSizeH - highresEffectivenessPaddingH; ++y)
        {
#if (PADDING_TYPE == PADDING_SAME)
            uint wrapIdx, wrapY = y;

            wrapIdx = highresTilePaddingOffsetW+wrapY*tileOutWPad;
            regularPartB[x+y*tileOutWPad] = regularPartB[wrapIdx];
            regularPartG[x+y*tileOutWPad] = regularPartG[wrapIdx];
            regularPartR[x+y*tileOutWPad] = regularPartR[wrapIdx];
#elif (PADDING_TYPE == PADDING_MIRROR)
            uint wrapIdx, wrapY = y;

            wrapIdx = (2*highresTilePaddingOffsetW-x)+wrapY*tileOutWPad;
            regularPartB[x+y*tileOutWPad] = regularPartB[wrapIdx];
            regularPartG[x+y*tileOutWPad] = regularPartG[wrapIdx];
            regularPartR[x+y*tileOutWPad] = regularPartR[wrapIdx];
#endif
        }
    }
    // Size-width padding
    for (uint x = 0; x < highresTilePaddingSizeW; ++x)
    {
        for (uint y = highresTilePaddingOffsetH; y < tileOutHPad - highresTilePaddingSizeH - highresEffectivenessPaddingH; ++y)
        {
#if (PADDING_TYPE == PADDING_SAME)
            uint wrapIdx, wrapY = y;

            wrapIdx = (tileOutWPad - 1 - highresTilePaddingSizeW - highresEffectivenessPaddingW)+wrapY*tileOutWPad;
            regularPartB[(tileOutWPad - 1 - x - highresEffectivenessPaddingW)+y*tileOutWPad] = regularPartB[wrapIdx];
            regularPartG[(tileOutWPad - 1 - x - highresEffectivenessPaddingW)+y*tileOutWPad] = regularPartG[wrapIdx];
            regularPartR[(tileOutWPad - 1 - x - highresEffectivenessPaddingW)+y*tileOutWPad] = regularPartR[wrapIdx];
#elif (PADDING_TYPE == PADDING_MIRROR)
            uint wrapIdx, wrapY = y;

            // No need to mul by 2 here as x = (x_real - size_x)
            wrapIdx = ((tileOutWPad_noEffPad - 1 - highresTilePaddingSizeW)-(highresTilePaddingSizeW - 1 - x))+wrapY*tileOutWPad;
            regularPartB[(tileOutWPad_noEffPad - 1 - x)+y*tileOutWPad] = regularPartB[wrapIdx];
            regularPartG[(tileOutWPad_noEffPad - 1 - x)+y*tileOutWPad] = regularPartG[wrapIdx];
            regularPartR[(tileOutWPad_noEffPad - 1 - x)+y*tileOutWPad] = regularPartR[wrapIdx];
#endif
        }
    }
#if (DBG_VERBOSE == 1)
    printf("done.\n");
#endif

    // DBG
    //////////////////////////////////////////////////////
    if (0)
    {
        {
            static int tileNum = 0;
            wchar_t tileName[128];
            wsprintf(tileName, L"tile%03d.bmp", tileNum);
            ++tileNum;

            const size_t channelSize = (int)tileOutWPad * (int)tileOutHPad;
            unsigned char * tmp = (unsigned char *)malloc(channelSize * 3 * sizeof(unsigned char));

            for (size_t i = 0; i < channelSize; ++i)
            {
                tmp[i*3  ] = (unsigned char)highresPartR[i];
                tmp[i*3+1] = (unsigned char)highresPartG[i];
                tmp[i*3+2] = (unsigned char)highresPartB[i];
            }

            darkroom::saveBmp(tmp, tileName, (int)tileOutWPad, (int)tileOutHPad, darkroom::BufferFormat::BGR8);
            free(tmp);
        }
        {
            static int tileNum = 0;
            wchar_t tileName[128];
            wsprintf(tileName, L"tile%03d_reg.bmp", tileNum);
            ++tileNum;

            const size_t channelSize = (int)tileOutWPad * (int)tileOutHPad;
            unsigned char * tmp = (unsigned char *)malloc(channelSize * 3 * sizeof(unsigned char));

            for (size_t i = 0; i < channelSize; ++i)
            {
                tmp[i*3  ] = (unsigned char)regularPartR[i];
                tmp[i*3+1] = (unsigned char)regularPartG[i];
                tmp[i*3+2] = (unsigned char)regularPartB[i];
            }

            darkroom::saveBmp(tmp, tileName, (int)tileOutWPad, (int)tileOutHPad, darkroom::BufferFormat::BGR8);
            free(tmp);
        }
    }


    // Fix up the highres tile
    ///////////////////////////////////////////////////////////////////////////////

    processDataFast(
        pPersistentData->pPersistentDataRegular, regularPart, (int)tileOutWPad, (int)tileOutHPad, regularScale,
        pPersistentData->pPersistentDataHighres, highresPart, (int)tileOutWPad, (int)tileOutHPad,
        tileOut, alpha, maxFreq, interpShapeType
    );


    // Copy processed part
    ///////////////////////////////////////////////////////////////////////////////

    copyFixedData(
        tileOut, tileOutWPad, tileOutHPad, highresTilePaddingOffsetW, highresTilePaddingOffsetH, highresTileSizeW, highresTileSizeH,
        highresTileFixed, highresSizeW, highresSizeH, highresTileOffsetW, highresTileOffsetH
        );
}

#undef DBG_SPECTRUM_SAVE