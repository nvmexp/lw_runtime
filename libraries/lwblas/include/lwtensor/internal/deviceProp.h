#pragma once

#include <lwtensor/internal/defines.h>

struct lwdaDeviceProp;

namespace LWTENSOR_NAMESPACE
{
    /**
     * \brief Subset of lwdaDeviceProp
     */
    class DeviceProp : public Initializable<133>
    {
        public: 
        DeviceProp& operator=(const lwdaDeviceProp &prop);

        size_t totalGlobalMem;
        size_t sharedMemPerBlock;
        int regsPerBlock;
        int maxThreadsPerBlock;
        int maxThreadsDim[3];
        int maxGridSize[3];
        int clockRate;
        size_t totalConstMem;
        int major;
        int minor;
        int multiProcessorCount;
        int memoryClockRate;
        int memoryBusWidth;
        int l2CacheSize;
        int maxThreadsPerMultiProcessor;
        size_t sharedMemPerMultiprocessor;
        int regsPerMultiprocessor;
        int singleToDoublePrecisionPerfRatio;
    };

}
