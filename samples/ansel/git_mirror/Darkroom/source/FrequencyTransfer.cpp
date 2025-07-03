#include <KissFFT/kiss_fft.h>
#include <KissFFT/kiss_fftndr.h>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>
#include <assert.h>

#include "complex.h"

#include "darkroom/FrequencyTransfer.h"

#define SAFE_FREE(x)    if (x) { free(x); (x) = nullptr; }
#define SAFE_DELETE(x)  if (x) { delete x; (x) = nullptr; }

namespace
{
    using darkroom::FFTChannelPersistentData;

    typedef struct
    {
        float x;
        float y;
    } fComplex;

    fComplex mul(double k, const fComplex & c)
    {
        fComplex result;
        result.x = (float)(k * c.x);
        result.y = (float)(k * c.y);
        return result;
    }

    float getModulus(const fComplex & c)
    {
        return sqrtf(c.x*c.x + c.y*c.y);
    }

    fComplex complexLerp(const fComplex & Ca, const fComplex & Cb, float coeff)
    {
        fComplex C;
        C.x = Ca.x * (1.0f - coeff) + Cb.x * coeff;
        C.y = Ca.y * (1.0f - coeff) + Cb.y * coeff;
        return C;
    }

    class FFTChannel
    {
    public:

        virtual void setName(const char * name) = 0;
        virtual void setSizes(uint32_t w, uint32_t h) = 0;

        virtual void init() = 0;

        virtual void freeIntermediateData() = 0;

        virtual void deinit() = 0;

        virtual void runFFT() = 0;

        virtual void runIlwFFT() = 0;

        virtual void retrieveDataSpectrum() = 0;
        virtual void uploadDataSpectrum() = 0;

        virtual fComplex getSpectrumValOffset(int x, int y) = 0;
        virtual fComplex getSpectrumVal(int x, int y) = 0;
        virtual void updateSpectrumValOffset(int x, int y, fComplex val) = 0;
        virtual void updateSpectrumVal(int x, int y, fComplex val) = 0;

        virtual void setDataPointer(float * dataPtr) = 0;
        virtual float * getData() = 0;
        virtual void setDataOutPointer(float * dataPtr) = 0;
        virtual float * getDataOut() = 0;
        virtual uint32_t getWidth() = 0;
        virtual uint32_t getHeight() = 0;
        virtual uint32_t getWidthFFT() = 0;
        virtual uint32_t getHeightFFT() = 0;
        virtual uint32_t getSpectrumWidthFFT() = 0;

        virtual void getResult() = 0;

        virtual ~FFTChannel() {}

        virtual void setPersistentDataPointer(FFTChannelPersistentData *) = 0;
        virtual FFTChannelPersistentData * getPersistentDataPointer() = 0;
    };

    class ChannelPair
    {
    public:
        FFTChannel * regular = nullptr;
        FFTChannel * highres = nullptr;

        ChannelPair(FFTChannel * reg, FFTChannel * hr) :
            regular(reg),
            highres(hr)
        {
        }
    };

    class FFTChannelPDataKissFFT : public FFTChannelPersistentData
    {
    public:

        // FFT result spectrum
        std::vector<unsigned char> dataSpectrumStorage;

        int kissfft_cfg_dims[2];
        kiss_fftndr_cfg kissfft_cfg = nullptr;
        kiss_fftndr_cfg kissfft_icfg = nullptr;

        virtual ~FFTChannelPDataKissFFT()
        {
            kiss_fft_cleanup();

            SAFE_FREE(kissfft_cfg);
            SAFE_FREE(kissfft_icfg);
        }
    };

    class FFTChannelKissFFT : public FFTChannel
    {
    public:

        std::string dbgName;

        // Input size
        uint32_t width = 0xFFFFFFFF, height = 0xFFFFFFFF;
        uint32_t widthFFT = 0xFFFFFFFF, heightFFT = 0xFFFFFFFF, spectrumWidthFFT = 0xFFFFFFFF;

        // Input data
        float * data = nullptr;
        // Output data
        float * dataOut = nullptr;

        FFTChannelPDataKissFFT * persistentDataKFFT = nullptr;

        void setDataPointer(float * dataPtr) override
        {
            data = dataPtr;
        }
        float * getData() override
        {
            return data;
        }
        void setDataOutPointer(float * dataOutPtr) override
        {
            dataOut = dataOutPtr;
        }
        float * getDataOut() override
        {
            return dataOut;
        }
        uint32_t getWidth() override
        {
            return width;
        }
        uint32_t getHeight() override
        {
            return height;
        }
        uint32_t getWidthFFT() override
        {
            return widthFFT;
        }
        uint32_t getHeightFFT() override
        {
            return heightFFT;
        }
        uint32_t getSpectrumWidthFFT() override
        {
            return spectrumWidthFFT;
        }

        void setName(const char * name) override
        {
            dbgName = name;
        }

        void setSizes(uint32_t w, uint32_t h) override
        {
            width = w;
            height = h;
            widthFFT = width;
            heightFFT = height;

            uint32_t dims[2] = { height, width };
            if (!persistentDataKFFT->kissfft_icfg || !persistentDataKFFT->kissfft_cfg ||
                persistentDataKFFT->kissfft_cfg_dims[0] != int(dims[0]) || persistentDataKFFT->kissfft_cfg_dims[1] != int(dims[1]))
            {
                SAFE_FREE(persistentDataKFFT->kissfft_cfg);
                SAFE_FREE(persistentDataKFFT->kissfft_icfg);

                persistentDataKFFT->kissfft_cfg = kiss_fftndr_alloc(reinterpret_cast<int*>(dims), 2, 0, 0, 0);
                persistentDataKFFT->kissfft_icfg = kiss_fftndr_alloc(reinterpret_cast<int*>(dims), 2, 1, 0, 0);

                persistentDataKFFT->kissfft_cfg_dims[0] = int(dims[0]);
                persistentDataKFFT->kissfft_cfg_dims[1] = int(dims[1]);
            }
        }

        void init() override
        {
            spectrumWidthFFT = (width / 2 + 1);

            // Allocating space for CPU data spectrum
            persistentDataKFFT->dataSpectrumStorage.resize(spectrumWidthFFT * heightFFT * sizeof(kiss_fft_cpx));
        }

        void freeIntermediateData() override
        {
        }

        void deinit() override
        {
        }

        void runFFT() override
        {
            kiss_fftndr(persistentDataKFFT->kissfft_cfg, data, reinterpret_cast<kiss_fft_cpx *>(&persistentDataKFFT->dataSpectrumStorage[0]));
        }

        void runIlwFFT() override
        {
            kiss_fftndri(persistentDataKFFT->kissfft_icfg, reinterpret_cast<kiss_fft_cpx *>(&persistentDataKFFT->dataSpectrumStorage[0]), dataOut);
        }

        void retrieveDataSpectrum() override { }
        void uploadDataSpectrum() override { }

        fComplex getSpectrumValOffset(int x, int y) override
        {
            assert(x >= 0);
            assert(y >= 0);
            assert(x < int(spectrumWidthFFT));
            assert(y < int(heightFFT));

            int yShifted = y - int((heightFFT >> 1));
            if (yShifted < 0)
                yShifted = yShifted + int(heightFFT);

            fComplex retVal;
            // fftwf_complex == float[2]
            kiss_fft_cpx * retValFFTW = reinterpret_cast<kiss_fft_cpx *>(&persistentDataKFFT->dataSpectrumStorage[0]) + x + yShifted*spectrumWidthFFT;

            retVal.x = retValFFTW->r;
            retVal.y = retValFFTW->i;

            return retVal;
        }
        fComplex getSpectrumVal(int x, int y) override
        {
            assert(x >= 0);
            assert(y >= 0);
            assert(x < int(spectrumWidthFFT));
            assert(y < int(heightFFT));

            fComplex retVal;
            // fftwf_complex == float[2]
            kiss_fft_cpx * retValFFTW = reinterpret_cast<kiss_fft_cpx *>(&persistentDataKFFT->dataSpectrumStorage[0]) + x + y*spectrumWidthFFT;

            retVal.x = retValFFTW->r;
            retVal.y = retValFFTW->i;

            return retVal;
        }

        void updateSpectrumValOffset(int x, int y, fComplex val) override
        {
            assert(x >= 0);
            assert(y >= 0);
            assert(x < int(spectrumWidthFFT));
            assert(y < int(heightFFT));

            int yShifted = y - int((heightFFT >> 1));
            if (yShifted < 0)
                yShifted = yShifted + int(heightFFT);

            // fftwf_complex == float[2]
            kiss_fft_cpx * retValFFTW = reinterpret_cast<kiss_fft_cpx *>(&persistentDataKFFT->dataSpectrumStorage[0]) + x + yShifted*spectrumWidthFFT;

            retValFFTW->r = val.x;
            retValFFTW->i = val.y;
        }
        void updateSpectrumVal(int x, int y, fComplex val) override
        {
            assert(x >= 0);
            assert(y >= 0);
            assert(x < int(spectrumWidthFFT));
            assert(y < int(heightFFT));

            // fftwf_complex == float[2]
            kiss_fft_cpx * retValFFTW = reinterpret_cast<kiss_fft_cpx *>(&persistentDataKFFT->dataSpectrumStorage[0]) + x + y*spectrumWidthFFT;

            retValFFTW->r = val.x;
            retValFFTW->i = val.y;
        }

        void getResult() override
        {
            const float numElements = (float)widthFFT*heightFFT;
            for (uint32_t i = 0; i < widthFFT; ++i)
            {
                for (uint32_t j = 0; j < heightFFT; ++j)
                {
                    /*
                    lwFFT performs un-normalized FFTs;
                    that is, performing a forward FFT on an input data set followed by an ilwerse FFT on the resulting set yields data that is equal to the input,
                    scaled by the number of elements. Scaling either transform by the reciprocal of the size of the data set is left for the user to perform as seen fit.
                    */
                    dataOut[i + j*widthFFT] = (dataOut[i + j*widthFFT] / numElements);
                }
            }
        }

        virtual void setPersistentDataPointer(FFTChannelPersistentData * persistentData) override
        {
            persistentDataKFFT = static_cast<FFTChannelPDataKissFFT *>(persistentData);
        }
        virtual FFTChannelPersistentData * getPersistentDataPointer() override
        {
            return persistentDataKFFT;
        }

        static FFTChannelPersistentData * allocatePersistentData()
        {
            return new FFTChannelPDataKissFFT;
        }
        static void freePersistentData(FFTChannelPersistentData * data)
        {
            delete data;
        }
    };

    template <typename T> T sqr(T val) { return val*val; }

    template <typename T>
    darkroom::Error copyFixedData(
        float * src, uint32_t tilePaddedSizeW, uint32_t tilePaddedSizeH, uint32_t tilePaddingOffsetW, uint32_t tilePaddingOffsetH, uint32_t tileDataSizeW, uint32_t tileDataSizeH,
        T * dst, uint32_t highresSizeW, uint32_t highresOffsetW, uint32_t highresOffsetH
    )
    {
        assert(false);
        return darkroom::Error::kOperationFailed;
    }

    template <>
    darkroom::Error copyFixedData<unsigned char>(
        float * src, uint32_t tilePaddedSizeW, uint32_t tilePaddedSizeH, uint32_t tilePaddingOffsetW, uint32_t tilePaddingOffsetH, uint32_t tileDataSizeW, uint32_t tileDataSizeH,
        unsigned char * dst, uint32_t highresSizeW, uint32_t highresOffsetW, uint32_t highresOffsetH
        )
    {
        if (src == nullptr)
            return darkroom::Error::kIlwalidArgument;

        if (dst == nullptr)
            return darkroom::Error::kIlwalidArgument;

        const auto saturate_uc = [](float val) -> unsigned char
        {
            if (val < 0)
                return 0;
            if (val > 255)
                return 255;

            return static_cast<unsigned char>(val);
        };

        size_t tileOutChannelSize = tilePaddedSizeW*tilePaddedSizeH;

        float * tileOutR = src;
        float * tileOutG = tileOutR + tileOutChannelSize;
        float * tileOutB = tileOutG + tileOutChannelSize;

        const uint32_t numChannels = 3;
        for (int64_t x = 0; x < (int64_t)tileDataSizeW; ++x)
        {
            int64_t padX = x + int64_t(tilePaddingOffsetW);
            for (int64_t y = 0; y < int64_t(tileDataSizeH); ++y)
            {
                int64_t padY = y + int64_t(tilePaddingOffsetH);
                dst[(x + highresOffsetW + (y + highresOffsetH)*highresSizeW)*numChannels + 2] = saturate_uc(tileOutB[padX + padY*tilePaddedSizeW]);
                dst[(x + highresOffsetW + (y + highresOffsetH)*highresSizeW)*numChannels + 1] = saturate_uc(tileOutG[padX + padY*tilePaddedSizeW]);
                dst[(x + highresOffsetW + (y + highresOffsetH)*highresSizeW)*numChannels] = saturate_uc(tileOutR[padX + padY*tilePaddedSizeW]);
            }
        }

        return darkroom::Error::kSuccess;
    }

    template <>
    darkroom::Error copyFixedData<float>(
        float * src, uint32_t tilePaddedSizeW, uint32_t tilePaddedSizeH, uint32_t tilePaddingOffsetW, uint32_t tilePaddingOffsetH, uint32_t tileDataSizeW, uint32_t tileDataSizeH,
        float * dst, uint32_t highresSizeW, uint32_t highresOffsetW, uint32_t highresOffsetH
        )
    {
        if (src == nullptr)
            return darkroom::Error::kIlwalidArgument;

        if (dst == nullptr)
            return darkroom::Error::kIlwalidArgument;

        size_t tileOutChannelSize = tilePaddedSizeW*tilePaddedSizeH;

        const float * tileOutR = src;
        const float * tileOutG = tileOutR + tileOutChannelSize;
        const float * tileOutB = tileOutG + tileOutChannelSize;

        const uint32_t numChannels = 3;
        for (int64_t x = 0; x < (int64_t)tileDataSizeW; ++x)
        {
            int64_t padX = x + int64_t(tilePaddingOffsetW);
            for (int64_t y = 0; y < (int64_t)tileDataSizeH; ++y)
            {
                int64_t padY = y + int64_t(tilePaddingOffsetH);
                dst[(x + highresOffsetW + (y + highresOffsetH)*highresSizeW)*numChannels + 2] = tileOutB[padX + padY*tilePaddedSizeW];
                dst[(x + highresOffsetW + (y + highresOffsetH)*highresSizeW)*numChannels + 1] = tileOutG[padX + padY*tilePaddedSizeW];
                dst[(x + highresOffsetW + (y + highresOffsetH)*highresSizeW)*numChannels] = tileOutR[padX + padY*tilePaddedSizeW];
            }
        }

        return darkroom::Error::kSuccess;
    }

    enum class FreqTransferInterpShape
    {
        kCIRCLE = 0,
        kRECTANGLE = 1,
        kELLIPSE = 2,

        kNUM_ENTRIES
    };

    darkroom::Error processDataFast(
        FFTChannelPersistentData * pPersistentDataRegular, float * regularData, uint32_t regularSizeW, uint32_t regularSizeH, double regularScale,
        FFTChannelPersistentData * pPersistentDataHighres, float * highresData, uint32_t highresSizeW, uint32_t highresSizeH,
        float * dataOut, float alpha, float maxFreq, FreqTransferInterpShape interpShapeType = FreqTransferInterpShape::kCIRCLE)
    {
        if (pPersistentDataRegular == nullptr)
            return darkroom::Error::kIlwalidArgument;

        if (regularData == nullptr)
            return darkroom::Error::kIlwalidArgument;

        if (pPersistentDataHighres == nullptr)
            return darkroom::Error::kIlwalidArgument;

        if (highresData == nullptr)
            return darkroom::Error::kIlwalidArgument;

        std::vector<ChannelPair> channelPairsList;

        FFTChannel * regularR = nullptr;
        FFTChannel * regularG = nullptr;
        FFTChannel * regularB = nullptr;
        regularR = new FFTChannelKissFFT;
        regularG = new FFTChannelKissFFT;
        regularB = new FFTChannelKissFFT;

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
        highresR = new FFTChannelKissFFT;
        highresG = new FFTChannelKissFFT;
        highresB = new FFTChannelKissFFT;

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

        const uint32_t regular_channels = 3;
        const uint32_t highres_channels = 3;

        size_t regularChannelSize = regularSizeW*regularSizeH;
        regularR->setDataPointer(regularData);
        regularG->setDataPointer(regularData + regularChannelSize);
        regularB->setDataPointer(regularData + 2 * regularChannelSize);

        size_t highresChannelSize = highresSizeW*highresSizeH;
        highresR->setDataPointer(highresData);
        highresG->setDataPointer(highresData + highresChannelSize);
        highresB->setDataPointer(highresData + 2 * highresChannelSize);
        highresR->setDataOutPointer(dataOut);
        highresG->setDataOutPointer(dataOut + highresChannelSize);
        highresB->setDataOutPointer(dataOut + 2 * highresChannelSize);

        for (size_t ch = 0; ch < channelPairsList.size(); ++ch)
        {
            FFTChannel * regularCh = channelPairsList[ch].regular;
            FFTChannel * highresCh = channelPairsList[ch].highres;

            regularCh->init();
            highresCh->init();

            regularCh->runFFT();
            highresCh->runFFT();

            const float regular_numElements = (float)regularCh->getWidthFFT()*regularCh->getHeightFFT();
            const float highres_numElements = (float)highresCh->getWidthFFT()*highresCh->getHeightFFT();
            regularCh->retrieveDataSpectrum();
            highresCh->retrieveDataSpectrum();

            // Blend the two
            {
                struct blendingData
                {
                    uint32_t spectrumW, spectrumH;
                    int centerI, centerJ;
                };

                blendingData blendRegular;
                blendRegular.spectrumW = regularCh->getSpectrumWidthFFT();
                blendRegular.spectrumH = regularCh->getHeightFFT();
                blendRegular.centerI = 0;
                blendRegular.centerJ = int(regularCh->getHeightFFT() / 2);

                blendingData blendHighres;
                blendHighres.spectrumW = highresCh->getSpectrumWidthFFT();
                blendHighres.spectrumH = highresCh->getHeightFFT();
                blendHighres.centerI = 0;
                blendHighres.centerJ = int(highresCh->getHeightFFT() / 2);

                assert(blendRegular.spectrumW <= blendHighres.spectrumW);
                assert(blendRegular.spectrumH <= blendHighres.spectrumH);

                // R2C FT width is half width already
                // TODO: add scaling?
                const int halfSizeW = int(blendRegular.spectrumW);
                const int halfSizeH = int((blendRegular.spectrumH >> 1));

                // Assuming size(blendRegular) <= size(blendHighres)
                for (uint32_t i = 0, iend = uint32_t(halfSizeW); i < iend; ++i)
                {
                    for (uint32_t j = 0, jend = uint32_t(halfSizeH); j < jend; ++j)
                    {
                        //uint32_t iRefl = blendRegular.spectrumW - i;
                        uint32_t jRefl = (blendRegular.spectrumH - 1) - j;
                        //uint32_t iReflHR = blendHighres.spectrumW - i;
                        uint32_t jReflHR = (blendHighres.spectrumH - 1) - j;

                        // TODO: second half is just a reflection in our case

                        const double numElements = ((double)regularCh->getWidthFFT()*regularCh->getHeightFFT());
                        const double numElementsHR = ((double)highresCh->getWidthFFT()*highresCh->getHeightFFT());
                        double scale = numElementsHR / numElements;

                        // Distance callwlation is easier wrt center
                        int jShifted = int(j) + halfSizeH;
                        if (jShifted < 0)
                            jShifted = int(j) - int(blendRegular.spectrumH);

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
                            // it can be treated as special case of Hann window scaled by alpha
                            interp = 0.5f * (1 - cosf((float)M_PI * interp));

                            {
                                fComplex regularSpectrumVal = regularCh->getSpectrumVal(int(i), int(j));
                                fComplex highresSpectrumVal = highresCh->getSpectrumVal(int(i), int(j));

                                highresSpectrumVal = complexLerp(highresSpectrumVal, mul(scale, regularSpectrumVal), interp);
                                highresCh->updateSpectrumVal(int(i), int(j), highresSpectrumVal);
                            }
                            {
                                fComplex regularSpectrumVal = regularCh->getSpectrumVal(int(i), int(jRefl));
                                fComplex highresSpectrumVal = highresCh->getSpectrumVal(int(i), int(jReflHR));

                                highresSpectrumVal = complexLerp(highresSpectrumVal, mul(scale, regularSpectrumVal), interp);
                                highresCh->updateSpectrumVal(int(i), int(jReflHR), highresSpectrumVal);
                            }
                        }
                    }
                }
            }

            regularCh->uploadDataSpectrum();
            highresCh->uploadDataSpectrum();

            highresCh->runIlwFFT();
            highresCh->getResult();

            regularCh->freeIntermediateData();
            highresCh->freeIntermediateData();
        }

        for (size_t ch = 0; ch < channelPairsList.size(); ++ch)
        {
            channelPairsList[ch].regular->deinit();
            channelPairsList[ch].highres->deinit();
        }

        SAFE_DELETE(highresR);
        SAFE_DELETE(highresG);
        SAFE_DELETE(highresB);

        SAFE_DELETE(regularR);
        SAFE_DELETE(regularG);
        SAFE_DELETE(regularB);

        return darkroom::Error::kSuccess;
    }

    // Use these two functions ONLY when you use processData directly
    // for general case, use initFFTTiledProcessing/deinitFFTTiledProcessing
    void initFFTProcessing(FFTChannelPersistentData ** ppPersistentDataRegular, FFTChannelPersistentData ** ppPersistentDataHighres)
    {
        *ppPersistentDataRegular = FFTChannelKissFFT::allocatePersistentData();
        *ppPersistentDataHighres = FFTChannelKissFFT::allocatePersistentData();
    }

    void deinitFFTProcessing(FFTChannelPersistentData * pPersistentDataRegular, FFTChannelPersistentData * pPersistentDataHighres)
    {
        FFTChannelKissFFT::freePersistentData(pPersistentDataRegular);
        FFTChannelKissFFT::freePersistentData(pPersistentDataHighres);
    }
}

namespace darkroom
{
    FFTChannelPersistentData::~FFTChannelPersistentData() {}

    void initFrequencyTransferProcessing(FFTPersistentData * pPersistentData)
    {
        initFFTProcessing(&pPersistentData->pPersistentDataRegular, &pPersistentData->pPersistentDataHighres);
    }

    void deinitFrequencyTransferProcessing(FFTPersistentData * pPersistentData)
    {
        deinitFFTProcessing(pPersistentData->pPersistentDataRegular, pPersistentData->pPersistentDataHighres);
    }

    void callwlateHighresMults(
        uint32_t regularSizeW, uint32_t regularSizeH,
        uint32_t highresSizeW, uint32_t highresSizeH,
        uint32_t * highresMultW, uint32_t * highresMultH
        )
    {
        if (highresMultW)
            *highresMultW = (uint32_t)ceil(highresSizeW / (float)regularSizeW);
        if (highresMultH)
            *highresMultH = (uint32_t)ceil(highresSizeH / (float)regularSizeH);
    }

    void getFrequencyTransferTilePaddings(
            uint32_t regularSizeW, uint32_t regularSizeH,
            uint32_t highresSizeW, uint32_t highresSizeH,
            uint32_t highresTileOffsetW, uint32_t highresTileOffsetH,
            uint32_t highresTileSizeW, uint32_t highresTileSizeH,
            uint32_t * outHighresTilePaddingOffsetW, uint32_t * outHighresTilePaddingOffsetH, uint32_t * outHighresTilePaddingSizeW, uint32_t * outHighresTilePaddingSizeH
            )
    {
        uint32_t highresMultW, highresMultH;
        callwlateHighresMults(
            regularSizeW, regularSizeH,
            highresSizeW, highresSizeH,
            &highresMultW, &highresMultH
            );

        // Tiling
        // Lwrrently regular has symmetrical padding
        // and highres could have asymmetrical padding due to potential sub-regular-pixel offsets
        const uint32_t tilePaddingInRegular = 8;

        // Regular sizes/offsets callwlation
        ///////////////////////////////////////////////////////////////////////////////
        // Callwlating regular tile box
        // the box should be callwlated taking into account that highres tile could have a box
        // that doesn't directly maps onto regular pixel grid, thus we need to be conservative
        const uint32_t regularTileOutOffsetW = (uint32_t)floor(highresTileOffsetW / (float)highresMultW);
        const uint32_t regularTileOutOffsetH = (uint32_t)floor(highresTileOffsetH / (float)highresMultH);
        const uint32_t regularTileOutSizeW = (uint32_t)ceil((highresTileSizeW + highresTileOffsetW) / (float)highresMultW) - regularTileOutOffsetW;
        const uint32_t regularTileOutSizeH = (uint32_t)ceil((highresTileSizeH + highresTileOffsetH) / (float)highresMultH) - regularTileOutOffsetH;

        // Paddings for the tile in regular space
        uint32_t regularTilePaddingOffsetW = tilePaddingInRegular;
        uint32_t regularTilePaddingOffsetH = tilePaddingInRegular;
        uint32_t regularTilePaddingSizeW = tilePaddingInRegular;
        uint32_t regularTilePaddingSizeH = tilePaddingInRegular;

        // We want to make our regular padded tile offset&size even
        // this will guarantee that highres tile will also be even
        // this subsequently will guarantee that none of the FFT libs will fail
        if ((regularTileOutOffsetW - regularTilePaddingOffsetW) & 1)
        {
            ++regularTilePaddingOffsetW;
        }
        if ((regularTileOutOffsetH - regularTilePaddingOffsetH) & 1)
        {
            ++regularTilePaddingOffsetH;
        }
        if ((regularTileOutSizeW + regularTilePaddingOffsetW + regularTilePaddingSizeW) & 1)
        {
            ++regularTilePaddingSizeW;
        }
        if ((regularTileOutSizeH + regularTilePaddingOffsetH + regularTilePaddingSizeH) & 1)
        {
            ++regularTilePaddingSizeH;
        }

        const uint32_t regularTileOutWPad = regularTileOutSizeW + regularTilePaddingOffsetW + regularTilePaddingSizeW;
        const uint32_t regularTileOutHPad = regularTileOutSizeH + regularTilePaddingOffsetH + regularTilePaddingSizeH;

        // Highres sizes/offsets callwlation
        ///////////////////////////////////////////////////////////////////////////////
        // In case highres tile doesn't fit onto pixel borders f regular, we need additional padding
        // this padding could be different on each edge of the highres tile
        const int highresAdditionalPaddingOffsetW = int(highresTileOffsetW - regularTileOutOffsetW * highresMultW);
        const int highresAdditionalPaddingOffsetH = int(highresTileOffsetH - regularTileOutOffsetH * highresMultH);
        const int highresAdditionalPaddingSizeW = int((regularTileOutSizeW + regularTileOutOffsetW) * highresMultW - (highresTileSizeW + highresTileOffsetW));
        const int highresAdditionalPaddingSizeH = int((regularTileOutSizeH + regularTileOutOffsetH) * highresMultH - (highresTileSizeH + highresTileOffsetH));

        assert(highresAdditionalPaddingOffsetW >= 0);
        assert(highresAdditionalPaddingOffsetH >= 0);
        assert(highresAdditionalPaddingSizeW >= 0);
        assert(highresAdditionalPaddingSizeH >= 0);

        uint32_t highresTilePaddingOffsetW = regularTilePaddingOffsetW*highresMultW + highresAdditionalPaddingOffsetW;
        uint32_t highresTilePaddingOffsetH = regularTilePaddingOffsetH*highresMultH + highresAdditionalPaddingOffsetH;
        uint32_t highresTilePaddingSizeW = regularTilePaddingSizeW*highresMultW + highresAdditionalPaddingSizeW;
        uint32_t highresTilePaddingSizeH = regularTilePaddingSizeH*highresMultH + highresAdditionalPaddingSizeH;

        const uint32_t tileOutW = highresTileSizeW;
        const uint32_t tileOutH = highresTileSizeH;
        uint32_t tileOutWPad = tileOutW + highresTilePaddingOffsetW + highresTilePaddingSizeW;
        uint32_t tileOutHPad = tileOutH + highresTilePaddingOffsetH + highresTilePaddingSizeH;

        const bool isEffectivenessPaddingAllowed = true;
        if (isEffectivenessPaddingAllowed)
        {
            uint32_t highresEffectivenessPaddingW = uint32_t(kiss_fftr_next_fast_size_real(int(tileOutWPad))) - tileOutWPad;
            uint32_t highresEffectivenessPaddingH = uint32_t(kiss_fftr_next_fast_size_real(int(tileOutHPad))) - tileOutHPad;

            // Same padding
            highresTilePaddingSizeW += highresEffectivenessPaddingW;
            highresTilePaddingSizeH += highresEffectivenessPaddingH;
        }

        if (outHighresTilePaddingOffsetW)
            *outHighresTilePaddingOffsetW = highresTilePaddingOffsetW;
        if (outHighresTilePaddingOffsetH)
            *outHighresTilePaddingOffsetH = highresTilePaddingOffsetH;
        if (outHighresTilePaddingSizeW)
            *outHighresTilePaddingSizeW = highresTilePaddingSizeW;
        if (outHighresTilePaddingSizeH)
            *outHighresTilePaddingSizeH = highresTilePaddingSizeH;
    }

    template<typename T>
    darkroom::Error processFrequencyTransfer(FFTPersistentData * pPersistentData, const T * regular, uint32_t regularSizeW, uint32_t regularSizeH,
        T * highres, uint32_t highresSizeW, uint32_t highresSizeH, uint32_t highresTileOffsetW, uint32_t highresTileOffsetH,
        uint32_t highresTileSizeW, uint32_t highresTileSizeH, T * highresTileFixed, float alpha)
    {
        if (pPersistentData == nullptr)
            return darkroom::Error::kIlwalidArgument;

        if (regular == nullptr)
            return darkroom::Error::kIlwalidArgument;

        if (highres == nullptr)
            return darkroom::Error::kIlwalidArgument;

        if (highresTileFixed == nullptr)
            return darkroom::Error::kIlwalidArgument;

        darkroom::Error status = darkroom::Error::kSuccess;
        const uint32_t numChannels = 3;

        // Callwlate tile size
        const uint32_t tileOutW = highresTileSizeW;
        const uint32_t tileOutH = highresTileSizeH;

        //uint32_t highresMultW = (uint32_t)ceil(highresSizeW / (float)regularSizeW);
        //uint32_t highresMultH = (uint32_t)ceil(highresSizeH / (float)regularSizeH);
        uint32_t highresMultW, highresMultH;
        callwlateHighresMults(
            regularSizeW, regularSizeH,
            highresSizeW, highresSizeH,
            &highresMultW, &highresMultH
            );

        double regularScale = highresMultW;

        uint32_t highresTilePaddingOffsetW;
        uint32_t highresTilePaddingOffsetH;
        uint32_t highresTilePaddingSizeW;
        uint32_t highresTilePaddingSizeH;
        getFrequencyTransferTilePaddings(
            regularSizeW, regularSizeH,
            highresSizeW, highresSizeH,
            highresTileOffsetW, highresTileOffsetH,
            highresTileSizeW, highresTileSizeH,
            &highresTilePaddingOffsetW, &highresTilePaddingOffsetH, &highresTilePaddingSizeW, &highresTilePaddingSizeH
            );

        uint32_t tileOutWPad = tileOutW + highresTilePaddingOffsetW + highresTilePaddingSizeW;
        uint32_t tileOutHPad = tileOutH + highresTilePaddingOffsetH + highresTilePaddingSizeH;

        size_t tileOutChannelSize = tileOutWPad*tileOutHPad;

        pPersistentData->highresPart.resize(tileOutChannelSize*numChannels * sizeof(float));
        float * highresPart = reinterpret_cast<float *>(&pPersistentData->highresPart[0]);

        memset(highresPart, 0, tileOutChannelSize*numChannels * sizeof(float));

        float * highresPartR = highresPart;
        float * highresPartG = highresPartR + tileOutChannelSize;
        float * highresPartB = highresPartG + tileOutChannelSize;

        // Output 
        pPersistentData->outputPart.resize(tileOutChannelSize*numChannels * sizeof(float));
        float * tileOut = reinterpret_cast<float *>(&pPersistentData->outputPart[0]);
        memset(tileOut, 0, tileOutChannelSize*numChannels * sizeof(float));

        // Upscaled regular tile allocation
        ///////////////////////////////////////////////////////////////////////////////
        pPersistentData->regularPart.resize(tileOutChannelSize*numChannels * sizeof(float));
        float * regularPart = reinterpret_cast<float *>(&pPersistentData->regularPart[0]);
        float * regularPartR = regularPart;
        float * regularPartG = regularPartR + tileOutChannelSize;
        float * regularPartB = regularPartG + tileOutChannelSize;


        // Copying data and introduce padding
        ///////////////////////////////////////////////////////////////////////////////

        // Copy highres part
        for (int x = -(int)highresTilePaddingOffsetW; x < (int)(tileOutW + highresTilePaddingSizeW); ++x)
        {
            for (int y = -(int)highresTilePaddingOffsetH; y < (int)(tileOutH + highresTilePaddingSizeH); ++y)
            {
                int padX = x + (int)highresTilePaddingOffsetW;
                int padY = y + (int)highresTilePaddingOffsetH;

                int64_t imageX = x + (int64_t)highresTileOffsetW;
                int64_t imageY = y + (int64_t)highresTileOffsetH;

                if (imageX < 0) imageX = 0;
                if (imageX >= (int)highresSizeW) imageX = (int)highresSizeW - 1;

                if (imageY < 0) imageY = 0;
                if (imageY >= (int)highresSizeH) imageY = (int)highresSizeH - 1;

                highresPartB[padX+padY*tileOutWPad] = highres[(imageX + imageY*highresSizeW)*numChannels+2];
                highresPartG[padX+padY*tileOutWPad] = highres[(imageX + imageY*highresSizeW)*numChannels+1];
                highresPartR[padX+padY*tileOutWPad] = highres[(imageX + imageY*highresSizeW)*numChannels  ];
            }
        }

        // Copy regular part
        for (int x = -(int)highresTilePaddingOffsetW; x < (int)(tileOutW + highresTilePaddingSizeW); ++x)
        {
            for (int y = -(int)highresTilePaddingOffsetH; y < (int)(tileOutH + highresTilePaddingSizeH); ++y)
            {
                int padX = x + (int)highresTilePaddingOffsetW;
                int padY = y + (int)highresTilePaddingOffsetH;

                int imageX = x + (int)highresTileOffsetW;
                int imageY = y + (int)highresTileOffsetH;

                if (imageX < 0) imageX = 0;
                if (imageX >= (int)highresSizeW) imageX = (int)highresSizeW - 1;

                if (imageY < 0) imageY = 0;
                if (imageY >= (int)highresSizeH) imageY = (int)highresSizeH - 1;

                int regularIdx = (imageX/(int)highresMultW + (imageY/(int)highresMultH)*(int)regularSizeW);

                regularPartB[padX+padY*tileOutWPad] = regular[regularIdx*numChannels+2];
                regularPartG[padX+padY*tileOutWPad] = regular[regularIdx*numChannels+1];
                regularPartR[padX+padY*tileOutWPad] = regular[regularIdx*numChannels  ];
            }
        }

        // Fix up the highres tile
        ///////////////////////////////////////////////////////////////////////////////
        status = processDataFast(pPersistentData->pPersistentDataRegular, regularPart, tileOutWPad, tileOutHPad, regularScale,
            pPersistentData->pPersistentDataHighres, highresPart, tileOutWPad, tileOutHPad, tileOut, alpha, 0.5f, FreqTransferInterpShape::kELLIPSE);

        if (status != darkroom::Error::kSuccess)
            return status;

        // Copy processed part
        ///////////////////////////////////////////////////////////////////////////////
        status = copyFixedData(
            tileOut, tileOutWPad, tileOutHPad, highresTilePaddingOffsetW, highresTilePaddingOffsetH, highresTileSizeW, highresTileSizeH,
            highresTileFixed, highresSizeW, highresTileOffsetW, highresTileOffsetH
            );

        if (status != darkroom::Error::kSuccess)
            return status;
        
        return darkroom::Error::kSuccess;
    }

    template darkroom::Error processFrequencyTransfer<unsigned char>(FFTPersistentData * pPersistentData, const unsigned char* regular, uint32_t regularSizeW, uint32_t regularSizeH,
        unsigned char* highres, uint32_t highresSizeW, uint32_t highresSizeH, uint32_t highresTileOffsetW, uint32_t highresTileOffsetH, uint32_t highresTileSizeW, uint32_t highresTileSizeH,
        unsigned char* highresTileFixed, float alpha);

    template darkroom::Error processFrequencyTransfer<float>(FFTPersistentData * pPersistentData, const float* regular, uint32_t regularSizeW, uint32_t regularSizeH,
        float* highres, uint32_t highresSizeW, uint32_t highresSizeH, uint32_t highresTileOffsetW, uint32_t highresTileOffsetH, uint32_t highresTileSizeW, uint32_t highresTileSizeH,
        float* highresTileFixed, float alpha);
}
