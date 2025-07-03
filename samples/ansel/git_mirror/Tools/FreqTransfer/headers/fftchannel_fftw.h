#pragma once

#include <fftw3.h>
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

#define SAFE_FFTWF_FREE(x)	if (x) { fftwf_free(x); (x) = nullptr; }

class FFTChannelPDataFFTW : public FFTChannelPersistentData
{
public:

    // FFT result spectrum
    std::vector<unsigned char> dataSpectrumStorage;


    virtual ~FFTChannelPDataFFTW()
    {
    }
};

class FFTChannelFFTW : public FFTChannel
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

    FFTChannelPDataFFTW * persistentDataFFTW = nullptr;

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
        LOG_DEBUG("...channel \"%s\" sizes: %d x %d [%d x %d]\n", dbgName.c_str(), width, height, widthFFT, heightFFT);
    }

    void init() override
    {
        LOG_DEBUG("Channel \"%s\":\n", dbgName.c_str());

        spectrumWidthFFT = (width/2+1);

        // Allocating space for CPU data spectrum
        persistentDataFFTW->dataSpectrumStorage.resize(spectrumWidthFFT * heightFFT * sizeof(fftwf_complex));
    }

    void freeIntermediateData() override
    {
    }

    void deinit() override
    {
        LOG_DEBUG("Channel \"%s\":\n", dbgName.c_str());

        freeIntermediateData();
    }

    void runFFT() override
    {
        LOG_DEBUG("...FFT on \"%s\":\n", dbgName.c_str());

        fftwf_plan p;
        p = fftwf_plan_dft_r2c_2d(height, width, data, reinterpret_cast<fftwf_complex *>(&persistentDataFFTW->dataSpectrumStorage[0]), FFTW_ESTIMATE);
        
        fftwf_exelwte(p);
        //checkLwdaErrors(lwfftExecR2C(fftPlanFwd, (lwfftReal *)d_PaddedData, (lwfftComplex *)d_DataSpectrum));
        
        fftwf_destroy_plan(p);

        LOG_DEBUG("   done.\n");
    }

    void runIlwFFT() override
    {
        LOG_DEBUG("...iFFT on \"%s\":\n", dbgName.c_str());

        fftwf_plan p;
        p = fftwf_plan_dft_c2r_2d(height, width, reinterpret_cast<fftwf_complex *>(&persistentDataFFTW->dataSpectrumStorage[0]), dataOut, FFTW_ESTIMATE);

        fftwf_exelwte(p);
        //checkLwdaErrors(lwfftExecC2R(fftPlanIlw, (lwfftComplex *)d_DataSpectrum, (lwfftReal *)d_PaddedData));

        fftwf_destroy_plan(p);

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
        fftwf_complex * retValFFTW = reinterpret_cast<fftwf_complex *>(&persistentDataFFTW->dataSpectrumStorage[0]) + x+yShifted*spectrumWidthFFT;

        retVal.x = (*retValFFTW)[0];
        retVal.y = (*retValFFTW)[1];

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
        fftwf_complex * retValFFTW = reinterpret_cast<fftwf_complex *>(&persistentDataFFTW->dataSpectrumStorage[0]) + x+y*spectrumWidthFFT;

        retVal.x = (*retValFFTW)[0];
        retVal.y = (*retValFFTW)[1];

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
        fftwf_complex * retValFFTW = reinterpret_cast<fftwf_complex *>(&persistentDataFFTW->dataSpectrumStorage[0]) + x+yShifted*spectrumWidthFFT;

        (*retValFFTW)[0] = val.x;
        (*retValFFTW)[1] = val.y;
    }
    void updateSpectrumVal(int x, int y, fComplex val) override
    {
        assert(x >= 0);
        assert(y >= 0);
        assert(x < spectrumWidthFFT);
        assert(y < heightFFT);

        // fftwf_complex == float[2]
        fftwf_complex * retValFFTW = reinterpret_cast<fftwf_complex *>(&persistentDataFFTW->dataSpectrumStorage[0]) + x+y*spectrumWidthFFT;

        (*retValFFTW)[0] = val.x;
        (*retValFFTW)[1] = val.y;
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
        persistentDataFFTW = static_cast<FFTChannelPDataFFTW *>(persistentData);
    }
    virtual FFTChannelPersistentData * getPersistentDataPointer() override
    {
        return persistentDataFFTW;
    }

    static FFTChannelPersistentData * allocatePersistentData()
    {
        return new FFTChannelPDataFFTW;
    }
    static void freePersistentData(FFTChannelPersistentData * data)
    {
        delete data;
    }
};

#undef SAFE_FFTWF_FREE