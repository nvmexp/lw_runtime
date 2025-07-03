#pragma once

#include <string>

#include "complex.h"

class FFTChannelPersistentData
{
public:

    virtual ~FFTChannelPersistentData() {}
};

class FFTChannel
{
public:

    virtual void setName(const char * name) = 0;
    virtual void setSizes(int w, int h) = 0;

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
    virtual int getWidth() = 0;
    virtual int getHeight() = 0;
    virtual int getWidthFFT() = 0;
    virtual int getHeightFFT() = 0;
    virtual int getSpectrumWidthFFT() = 0;

    virtual void getResult() = 0;

    virtual ~FFTChannel() {}

    virtual void setPersistentDataPointer(FFTChannelPersistentData *) = 0;
    virtual FFTChannelPersistentData * getPersistentDataPointer() = 0;
};
