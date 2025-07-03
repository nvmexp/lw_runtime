#include <lwda_runtime.h>

#include <lwtensor/internal/context.h>
#include <lwtensor/internal/util.h>
#include <lwtensor/internal/defines.h>

namespace LWTENSOR_NAMESPACE 
{
    DeviceProp& DeviceProp::operator=(const lwdaDeviceProp &prop)
    {
        this->totalGlobalMem = prop.totalGlobalMem;
        this->sharedMemPerBlock = prop.sharedMemPerBlock;
        this->regsPerBlock = prop.regsPerBlock;
        this->maxThreadsPerBlock = prop.maxThreadsPerBlock;
        for(int i=0; i < 3; ++i)
        {
            this->maxThreadsDim[i] = prop.maxThreadsDim[i];
            this->maxThreadsDim[i] = prop.maxThreadsDim[i];
            this->maxGridSize[i] = prop.maxGridSize[i];
        }
        this->clockRate = prop.clockRate;
        this->totalConstMem = prop.totalConstMem;
        this->major = prop.major;
        this->minor = prop.minor;
        this->multiProcessorCount = prop.multiProcessorCount;
        this->memoryClockRate = prop.memoryClockRate;
        this->memoryBusWidth = prop.memoryBusWidth;
        this->l2CacheSize = prop.l2CacheSize;
        this->maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
        this->sharedMemPerMultiprocessor = prop.sharedMemPerMultiprocessor;
        this->regsPerMultiprocessor = prop.regsPerMultiprocessor;
        this->singleToDoublePrecisionPerfRatio = prop.singleToDoublePrecisionPerfRatio;
        this->setInitialized();
        return *this;
    }

    Context::Context()
    {
        this->unsetInitialized();
    }

    Context::~Context()
    {
        this->unsetInitialized();
    }

    lwtensorStatus_t Context::init(const int lwrrentDevice) noexcept
    {
        this->unsetInitialized();

        HANDLE_ERROR(lwdaGetLastError());

        activeGpuId_ = lwrrentDevice;

        if (getelw("LWTENSOR_LOG_LEVEL") != nullptr)
        {
            logLevel_ = atoi(getelw("LWTENSOR_LOG_LEVEL"));
        }
        else
        {
            logLevel_ = 0; // i.e., no logging
        }

        if (getelw("LWTENSOR_DISABLE_PLAN_CACHE") != nullptr && atoi(getelw("LWTENSOR_DISABLE_PLAN_CACHE")) == 1)
        {
            disableCache_ = true;
        }
        else
        {
            disableCache_ = false;
        }

        if (getelw("LWTENSOR_DISABLE_LWBLAS") != nullptr && atoi(getelw("LWTENSOR_DISABLE_LWBLAS")) == 1)
        {
            disableLwblas_= true;
        }
        else
        {
            disableLwblas_ = false;
        }

        if (getelw("LWTENSOR_FORCE_C_EQUAL_D") != nullptr && atoi(getelw("LWTENSOR_FORCE_C_EQUAL_D")) == 1)
        {
            forceEqualCD_ = true;
        }
        else
        {
            forceEqualCD_ = false;
        }

        lwdaDeviceProp prop;
        HANDLE_ERROR(lwdaGetDeviceProperties(&prop, activeGpuId_));
        deviceProp_ = prop;

#ifdef LWTENSOR_EXPOSE_INTERNAL
        const char * ccOverride = getelw("LWTENSOR_CC_OVERRIDE");
        if (ccOverride != nullptr)
        {
            int cc = atoi(ccOverride);
            deviceProp_.major = cc / 10;
            deviceProp_.minor = cc % 10;
        }
#endif

        if(getelw("LWTENSOR_LOGINFO_DBG") != nullptr && atoi(getelw("LWTENSOR_LOGINFO_DBG")) == 1)
        {
            this->handleError_   = static_cast<lwtensorStatus_t (*)(const lwtensorStatus_t, const std::string &)>(&handleError_log);
        }
        else
        {
            this->handleError_   = static_cast<lwtensorStatus_t (*)(const lwtensorStatus_t, const std::string &)>(&handleError);
        }

        const char * enableTF32Flag = getelw("LWIDIA_TF32_OVERRIDE");
        if (enableTF32Flag != nullptr && strcmp(enableTF32Flag, "0") == 0)
        {
          this->forceEnableTF32_ = false;
          this->forceDisableTF32_ = true;
        }
        else if (enableTF32Flag != nullptr && strcmp(enableTF32Flag, "1") == 0)
        {
          this->forceEnableTF32_ = true;
          this->forceDisableTF32_ = false;
        }
        else
        {
          this->forceEnableTF32_ = false;
          this->forceDisableTF32_ = false;
        }

        this->contractionPlanCache_ = nullptr;

        this->setInitialized();

        return LWTENSOR_STATUS_SUCCESS;
    }

    bool Context::hasValidPlanCache() const
    {
        return (! disableCache_) && contractionPlanCache_ != nullptr && contractionPlanCache_->isInitialized();
    }

    lwtensorStatus_t Context::attachPlanCachelines( PlanCacheline cachelines[],
            const uint32_t numCachelines)
    {
        if( contractionPlanCache_ != nullptr )
        {
            RETURN_STATUS(this->logError(LWTENSOR_STATUS_NOT_SUPPORTED, "You must detach the cachelines before you can attach new ones."));
        }
        if( cachelines == nullptr || numCachelines == 0 )
        {
            RETURN_STATUS(this->logError(LWTENSOR_STATUS_ILWALID_VALUE, "Cachelines invalid."));
        }
        contractionPlanCache_ = new ContractionPlanCache;
        return contractionPlanCache_->attachCachelines(cachelines, numCachelines);
    }

    lwtensorStatus_t Context::detachPlanCachelines()
    {
        if( contractionPlanCache_ == nullptr )
        {
            RETURN_STATUS(this->logError(LWTENSOR_STATUS_NOT_SUPPORTED, "You must attach cachlines before you can detach them."));
        }
        delete contractionPlanCache_;
        contractionPlanCache_ = nullptr;
        return LWTENSOR_STATUS_SUCCESS;
    }

    lwtensorStatus_t Context::writeCacheToFile(const char filename[]) const
    {
        if (this->hasValidPlanCache())
        {
            return contractionPlanCache_->writeToFile(filename);
        }
        else
        {
            RETURN_STATUS(this->logError(LWTENSOR_STATUS_ILWALID_VALUE, "The cache is not valid (are you sure that you've attached cache lines?)."));
        }
    }

    lwtensorStatus_t Context::readCacheFromFile(const char filename[], uint32_t &numCachelinesRead)
    {
        if (this->hasValidPlanCache())
        {
            return contractionPlanCache_->readFromFile(filename, numCachelinesRead);
        }
        else
        {
            RETURN_STATUS(this->logError(LWTENSOR_STATUS_ILWALID_VALUE, "The cache is not valid (are you sure that you've attached cache lines?)."));
        }
    }
}

