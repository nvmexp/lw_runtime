#pragma once

#include <string>
#include <assert.h>

#include <lwda_runtime.h>
#include <lwfft.h>

#include "fftchannel.h"
#include "helpers_lwda.h"

#define DBG_LOG_CHANNEL		0

#if (DBG_LOG_CHANNEL == 1)
#	define LOG_DEBUG(fmt, ...)	printf(fmt, ...)
#else
#	define LOG_DEBUG(fmt, ...)
#endif

class FFTChannelPDataLwFFT : public FFTChannelPersistentData
{
public:

    // FFT result spectrum
    std::vector<unsigned char> dataSpectrumStorage;

    virtual ~FFTChannelPDataLwFFT()
    {
    }
};

class FFTChannelLwFFT : public FFTChannel
{
public:

    std::string dbgName;

    // TODO: create automated FFT plan managers
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    lwfftHandle
        fftPlanFwd,
        fftPlanIlw;

    // Input size
    int width = -1, height = -1;
    // FFT padded size
    int widthFFT = -1, heightFFT = -1, spectrumWidthFFT = -1;

    // [cpu] Input data
    float * h_Data = nullptr;
    // [cpu] FFT padded output data
    float * h_DataOut = nullptr;

    // [gpu] Input data;
    float * d_Data = nullptr;
    // [gpu] FFT padded input data;
    float * d_PaddedData = nullptr;

    // [gpu] FFT result spectrum
    fComplex * d_DataSpectrum = nullptr;
    // [cpu] FFT result spectrum
    //fComplex * h_DataSpectrum = nullptr;
    FFTChannelPDataLwFFT * persistentDataLwFFT = nullptr;

    void setDataPointer(float * dataPtr) override
    {
        h_Data = dataPtr;
    }
    float * getData() override
    {
        return h_Data;
    }
    void setDataOutPointer(float * dataOutPtr) override
    {
        h_DataOut = dataOutPtr;
    }
    float * getDataOut() override
    {
        return h_DataOut;
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
#if (USE_PADDING_FFT == 1)
        widthFFT = snapTransformSize(width - 1);
        heightFFT = snapTransformSize(height - 1);
#else
        widthFFT = width;
        heightFFT = height;
#endif
        LOG_DEBUG("...channel \"%s\" sizes: %d x %d [%d x %d]\n", dbgName.c_str(), width, height, widthFFT, heightFFT);

#if 0
        if (h_Data || h_DataOut)
        {
            LOG_DEBUG("..deallocating CPU data:\n");
            SAFE_FREE(h_Data);
            SAFE_FREE(h_DataOut);
            LOG_DEBUG("   done.\n");
        }

        LOG_DEBUG("...allocating CPU data:\n");
        h_Data		= (float *)malloc(width		* height	* sizeof(float));
        h_DataOut	= (float *)malloc(widthFFT	* heightFFT	* sizeof(float));
        LOG_DEBUG("   done.\n");
#endif
    }

    void init() override
    {
        LOG_DEBUG("Channel \"%s\":\n", dbgName.c_str());

        if (d_Data || d_PaddedData || d_DataSpectrum)
        {
            LOG_DEBUG("...deallocating GPU data:\n");
            SAFE_LWFREE(d_Data);
            SAFE_LWFREE(d_PaddedData);
            SAFE_LWFREE(d_DataSpectrum);
            LOG_DEBUG("   done.\n");
        }

        LOG_DEBUG("...allocating GPU data:\n");
        spectrumWidthFFT = (widthFFT/2+1);
        checkLwdaErrors(lwdaMalloc((void **)&d_Data,			width			* height	* sizeof(float)));
        checkLwdaErrors(lwdaMalloc((void **)&d_PaddedData,		widthFFT		* heightFFT	* sizeof(float)));
        checkLwdaErrors(lwdaMalloc((void **)&d_DataSpectrum,	spectrumWidthFFT* heightFFT * sizeof(fComplex)));
        LOG_DEBUG("   done.\n");

        // Allocating space for CPU data spectrum
#if 0
        if (h_DataSpectrum)
        {
            SAFE_FREE(h_DataSpectrum);
        }
        h_DataSpectrum = (fComplex *)malloc(spectrumWidthFFT * heightFFT * sizeof(fComplex));
#endif

        persistentDataLwFFT->dataSpectrumStorage.resize(spectrumWidthFFT * heightFFT * sizeof(fComplex));

        LOG_DEBUG("...uploading to GPU and padding the input data:\n");
#if (USE_PADDING_FFT == 1)
        checkLwdaErrors(lwdaMemcpy(d_Data, h_Data, width * height * sizeof(float), lwdaMemcpyHostToDevice));
        checkLwdaErrors(lwdaMemset(d_PaddedData, 0, widthFFT * heightFFT * sizeof(float)));

        padDataClampToBorder(
            d_PaddedData,
            d_Data,
            heightFFT,
            widthFFT,
            height,
            width,
            0,
            0,
            0,
            0
            );
#else
        checkLwdaErrors(lwdaMemcpy(d_PaddedData, h_Data, width * height * sizeof(float), lwdaMemcpyHostToDevice));
#endif

#if (DBG_FORCED_SYNCS == 1)
        checkLwdaErrors(lwdaDeviceSynchronize());
#endif

        LOG_DEBUG("   done.\n");

        // TODO: create automated FFT plan managers
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        LOG_DEBUG("Creating R2C & C2R FFT plans for %i x %i\n", widthFFT, heightFFT);
        checkLwdaErrors(lwfftPlan2d(&fftPlanFwd, heightFFT, widthFFT, LWFFT_R2C));
        checkLwdaErrors(lwfftPlan2d(&fftPlanIlw, heightFFT, widthFFT, LWFFT_C2R));
        LOG_DEBUG("done.\n");
    }

    void freeIntermediateData() override
    {
        LOG_DEBUG("Channel \"%s\":\n", dbgName.c_str());

        LOG_DEBUG("...deallocating GPU data:\n");
        SAFE_LWFREE(d_DataSpectrum);
        SAFE_LWFREE(d_PaddedData);
        SAFE_LWFREE(d_Data);
        LOG_DEBUG("   done.\n");
    }

    void deinit() override
    {
        LOG_DEBUG("Channel \"%s\":\n", dbgName.c_str());

        checkLwdaErrors(lwfftDestroy(fftPlanIlw));
        checkLwdaErrors(lwfftDestroy(fftPlanFwd));

        freeIntermediateData();

        LOG_DEBUG("...deallocating CPU data:\n");
        //SAFE_FREE(h_DataSpectrum);
        //SAFE_FREE(h_DataOut);
        //SAFE_FREE(h_Data);
        LOG_DEBUG("   done.\n");
    }

    void runFFT() override
    {
        LOG_DEBUG("...FFT on \"%s\":\n", dbgName.c_str());
        checkLwdaErrors(lwfftExecR2C(fftPlanFwd, (lwfftReal *)d_PaddedData, (lwfftComplex *)d_DataSpectrum));
        LOG_DEBUG("   done.\n");
    }

    void runIlwFFT() override
    {
        LOG_DEBUG("...iFFT on \"%s\":\n", dbgName.c_str());
        checkLwdaErrors(lwfftExecC2R(fftPlanIlw, (lwfftComplex *)d_DataSpectrum, (lwfftReal *)d_PaddedData));
        LOG_DEBUG("   done.\n");
    }

    void retrieveDataSpectrum() override
    {
        LOG_DEBUG("...fetching data spectrum of \"%s\":\n", dbgName.c_str());
#if (DBG_FORCED_SYNCS == 1)
        checkLwdaErrors(lwdaDeviceSynchronize());
#endif
        checkLwdaErrors(lwdaMemcpy(reinterpret_cast<fComplex *>(&persistentDataLwFFT->dataSpectrumStorage[0]), d_DataSpectrum, spectrumWidthFFT * heightFFT * sizeof(lwfftComplex), lwdaMemcpyDeviceToHost));
        LOG_DEBUG("   done.\n");
    }
    void uploadDataSpectrum() override
    {
        LOG_DEBUG("...uploading data spectrum of \"%s\":\n", dbgName.c_str());
#if (DBG_FORCED_SYNCS == 1)
        checkLwdaErrors(lwdaDeviceSynchronize());
#endif
        checkLwdaErrors(lwdaMemcpy(d_DataSpectrum, reinterpret_cast<fComplex *>(&persistentDataLwFFT->dataSpectrumStorage[0]), spectrumWidthFFT * heightFFT * sizeof(lwfftComplex), lwdaMemcpyHostToDevice));
        LOG_DEBUG("   done.\n");
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
        return reinterpret_cast<fComplex *>(&persistentDataLwFFT->dataSpectrumStorage[0])[x+yShifted*spectrumWidthFFT];
    }
    fComplex getSpectrumVal(int x, int y) override
    {
        assert(x >= 0);
        assert(y >= 0);
        assert(x < spectrumWidthFFT);
        assert(y < heightFFT);

        return reinterpret_cast<fComplex *>(&persistentDataLwFFT->dataSpectrumStorage[0])[x+y*spectrumWidthFFT];
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
        reinterpret_cast<fComplex *>(&persistentDataLwFFT->dataSpectrumStorage[0])[x+yShifted*spectrumWidthFFT] = val;
    }
    void updateSpectrumVal(int x, int y, fComplex val) override
    {
        assert(x >= 0);
        assert(y >= 0);
        assert(x < spectrumWidthFFT);
        assert(y < heightFFT);

        reinterpret_cast<fComplex *>(&persistentDataLwFFT->dataSpectrumStorage[0])[x+y*spectrumWidthFFT] = val;
    }


    void getResult() override
    {
        LOG_DEBUG("...fetching result of \"%s\":\n", dbgName.c_str());
#if (DBG_FORCED_SYNCS == 1)
        checkLwdaErrors(lwdaDeviceSynchronize());
#endif
        checkLwdaErrors(lwdaMemcpy(h_DataOut, d_PaddedData, widthFFT * heightFFT * sizeof(float), lwdaMemcpyDeviceToHost));

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
                h_DataOut[i+j*widthFFT] = (h_DataOut[i+j*widthFFT] / numElements);
            }
        }
        LOG_DEBUG("   done.\n");
    }

    virtual void setPersistentDataPointer(FFTChannelPersistentData * persistentData) override
    {
        persistentDataLwFFT = static_cast<FFTChannelPDataLwFFT *>(persistentData);
    }
    virtual FFTChannelPersistentData * getPersistentDataPointer() override
    {
        return persistentDataLwFFT;
    }

    static FFTChannelPersistentData * allocatePersistentData()
    {
        return new FFTChannelPDataLwFFT;
    }
    static void freePersistentData(FFTChannelPersistentData * data)
    {
        delete data;
    }
};
