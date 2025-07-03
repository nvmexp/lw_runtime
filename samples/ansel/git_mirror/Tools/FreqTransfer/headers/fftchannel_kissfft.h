#pragma once

#include <KissFFT/kiss_fft.h>
#include <KissFFT/kiss_fftndr.h>
#include <string>
#include <assert.h>

#include "complex.h"

#include "fftchannel.h"

#define DBG_LOG_CHANNEL		0

#if (DBG_LOG_CHANNEL == 1)
#	define LOG_DEBUG(fmt, ...)	printf(fmt, ...)
#else
#	define LOG_DEBUG(fmt, ...)
#endif

#define SAFE_KISSFFT_FREE(x)	if (x) { free(x); (x) = nullptr; }

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
    int width = -1, height = -1;
    int widthFFT = -1, heightFFT = -1, spectrumWidthFFT = -1;

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
    int getWidth() override
    {
        return width;
    }
    int getHeight() override
    {
        return height;
    }
    int getWidthFFT() override
    {
        return widthFFT;
    }
    int getHeightFFT() override
    {
        return heightFFT;
    }
    int getSpectrumWidthFFT() override
    {
        return spectrumWidthFFT;
    }

    void setName(const char * name) override
    {
        LOG_DEBUG("Channel:\n");
        dbgName = name;
        LOG_DEBUG("...assigning channel name \"%s\"\n", name);
    }

    void setSizes(int w, int h) override
    {
        LOG_DEBUG("Channel \"%s\":\n", dbgName.c_str());

        width = w;
        height = h;
        widthFFT = width;
        heightFFT = height;

        int dims[2] = { height, width };
        if (!persistentDataKFFT->kissfft_icfg || !persistentDataKFFT->kissfft_cfg ||
            persistentDataKFFT->kissfft_cfg_dims[0] != dims[0] || persistentDataKFFT->kissfft_cfg_dims[1] != dims[1])
        {
            SAFE_FREE(persistentDataKFFT->kissfft_cfg);
            SAFE_FREE(persistentDataKFFT->kissfft_icfg);

            persistentDataKFFT->kissfft_cfg = kiss_fftndr_alloc(dims, 2, 0, 0, 0);
            persistentDataKFFT->kissfft_icfg = kiss_fftndr_alloc(dims, 2, 1, 0, 0);

            persistentDataKFFT->kissfft_cfg_dims[0] = dims[0];
            persistentDataKFFT->kissfft_cfg_dims[1] = dims[1];
        }

        LOG_DEBUG("...channel \"%s\" sizes: %d x %d [%d x %d]\n", dbgName.c_str(), width, height, widthFFT, heightFFT);
    }

    void init() override
    {
        LOG_DEBUG("Channel \"%s\":\n", dbgName.c_str());

        spectrumWidthFFT = (width/2+1);

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
        LOG_DEBUG("...FFT on \"%s\":\n", dbgName.c_str());

        kiss_fftndr(persistentDataKFFT->kissfft_cfg, data, reinterpret_cast<kiss_fft_cpx *>(&persistentDataKFFT->dataSpectrumStorage[0]));

        LOG_DEBUG("   done.\n");
    }

    void runIlwFFT() override
    {
        LOG_DEBUG("...iFFT on \"%s\":\n", dbgName.c_str());

        kiss_fftndri(persistentDataKFFT->kissfft_icfg, reinterpret_cast<kiss_fft_cpx *>(&persistentDataKFFT->dataSpectrumStorage[0]), dataOut);

        LOG_DEBUG("   done.\n");
    }

    void retrieveDataSpectrum() override
    {
    }
    void uploadDataSpectrum() override
    {
    }

    fComplex getSpectrumValOffset(int x, int y) override
    {
        assert(x >= 0);
        assert(y >= 0);
        assert(x < spectrumWidthFFT);
        assert(y < heightFFT);

        int yShifted = y - (heightFFT >> 1);
        if (yShifted < 0)
            yShifted = yShifted + heightFFT;

        fComplex retVal;
        // fftwf_complex == float[2]
        kiss_fft_cpx * retValFFTW = reinterpret_cast<kiss_fft_cpx *>(&persistentDataKFFT->dataSpectrumStorage[0]) + x+yShifted*spectrumWidthFFT;

        retVal.x = retValFFTW->r;
        retVal.y = retValFFTW->i;

        return retVal;
    }
    fComplex getSpectrumVal(int x, int y) override
    {
        assert(x >= 0);
        assert(y >= 0);
        assert(x < spectrumWidthFFT);
        assert(y < heightFFT);

        fComplex retVal;
        // fftwf_complex == float[2]
        kiss_fft_cpx * retValFFTW = reinterpret_cast<kiss_fft_cpx *>(&persistentDataKFFT->dataSpectrumStorage[0]) + x+y*spectrumWidthFFT;

        retVal.x = retValFFTW->r;
        retVal.y = retValFFTW->i;

        return retVal;
    }

    void updateSpectrumValOffset(int x, int y, fComplex val) override
    {
        assert(x >= 0);
        assert(y >= 0);
        assert(x < spectrumWidthFFT);
        assert(y < heightFFT);

        int yShifted = y - (heightFFT >> 1);
        if (yShifted < 0)
            yShifted = yShifted + heightFFT;

        // fftwf_complex == float[2]
        kiss_fft_cpx * retValFFTW = reinterpret_cast<kiss_fft_cpx *>(&persistentDataKFFT->dataSpectrumStorage[0]) + x+yShifted*spectrumWidthFFT;

        retValFFTW->r = val.x;
        retValFFTW->i = val.y;
    }
    void updateSpectrumVal(int x, int y, fComplex val) override
    {
        assert(x >= 0);
        assert(y >= 0);
        assert(x < spectrumWidthFFT);
        assert(y < heightFFT);

        // fftwf_complex == float[2]
        kiss_fft_cpx * retValFFTW = reinterpret_cast<kiss_fft_cpx *>(&persistentDataKFFT->dataSpectrumStorage[0]) + x+y*spectrumWidthFFT;

        retValFFTW->r = val.x;
        retValFFTW->i = val.y;
    }

    void getResult() override
    {
        LOG_DEBUG("...fetching result of \"%s\":\n", dbgName.c_str());
        LOG_DEBUG("   scaling FFT result:\n");
        const float numElements = (float)widthFFT*heightFFT;
        for (int i = 0; i < widthFFT; ++i)
        {
            for (int j = 0; j < heightFFT; ++j)
            {
                /*
                lwFFT performs un-normalized FFTs;
                that is, performing a forward FFT on an input data set followed by an ilwerse FFT on the resulting set yields data that is equal to the input,
                scaled by the number of elements. Scaling either transform by the reciprocal of the size of the data set is left for the user to perform as seen fit.
                */
                dataOut[i+j*widthFFT] = (dataOut[i+j*widthFFT] / numElements);
            }
        }
        LOG_DEBUG("   done.\n");
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

#undef SAFE_KISSFFT_FREE