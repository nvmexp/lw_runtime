#pragma once

#include <string>

#include <lwtensor/types.h>
#include <lwtensor/internal/contractionPlanCache.h>
#include <lwtensor/internal/deviceProp.h>
#include <lwtensor/internal/defines.h>

struct lwdaDeviceProp;


namespace LWTENSOR_NAMESPACE
{

    /**
     *  \brief Context holds common information used throughout this library.
     *
     *  \details The only common information hold by this struct is lwrrently the number of
     *  streaming multi-processors (SMs) for the active LWCA device; all other information
     *  (e.g., stream) are exposed via the public API directly.
     *  \req None
     *  \ilwariants Nothing shall changes after initialization
     */
    class Context : public Initializable<42>
    {
        public:
            /**
             * \brief Default contructor initializes all member variables.
             * \req None
             * \pre None
             * \changes_elw allocate resources
             * \throw internal exception if fail to acquire a valid device ID through lwdaGetDevice.
             * \exception-guarantee basic
             * \behavior blocking, not reentrant, and thread safe
             */
            Context();

            /**
             * \brief Destructor responsible for freeing any allocated resources.
             * \req None
             * \pre None
             * \changes_elw allocate resources
             * \exception-guarantee nothrow
             * \behavior blocking, not reentrant, and thread safe
             */
            ~Context();

            /**
             * \brief Initialize all member variables.
             * \req None
             * \pre None
             * \changes_elw None
             * \return error code
             * \behavior blocking, not reentrant, and thread safe
             */
            lwtensorStatus_t init(const int lwrrentDevice) noexcept;

            /**
             * \brief Returns the device properties of the selected gpu Id.
             * \req None
             * \pre None
             * \return Returns the device properties of the selected gpu Id.
             * \exception-guarantee nothrow
             * \behavior blocking, not reentrant, and thread safe
             */
            const DeviceProp* getDeviceProp() const noexcept { return &deviceProp_; }

            /**
            * \brief Returns an error code, and if LWTENSOR_LOGINFO_DBG == 1, prints an error message.
            * \req None
            * \pre err is a valid lwtensorStatus_t, and desc is a valid string containing the message.
            * \return The processed error code err.
            * \exception-guarantee nothrow
            * \behavior blocking, not reentrant, and thread safe
            */
            lwtensorStatus_t logError(const lwtensorStatus_t err, const std::string &desc) const noexcept
            { 
                return this->handleError_(err, desc);
            }

            /**
            * \brief Selector to force TF32 kernels
            */
            bool isTF32ForceEnabled() const noexcept { return forceEnableTF32_; }
            /**
            * \brief Selector to force non-TF32 kernels
            */
            bool isTF32ForceDisabled() const noexcept { return forceDisableTF32_; }

            using PlanCacheline = typename ContractionPlanCache::Cacheline;
            /**
             * Attaches cachelines to cache
             */
            lwtensorStatus_t attachPlanCachelines( PlanCacheline cachelines[],
                                             const uint32_t numCachelines);
            /**
             * Detaches cachelines from cache
             */
            lwtensorStatus_t detachPlanCachelines();

            lwtensorStatus_t writeCacheToFile(const char filename[]) const;

            lwtensorStatus_t readCacheFromFile(const char filename[], uint32_t &numCachelinesRead);

            bool hasValidPlanCache() const;

            ContractionPlanCache* getCache() const { return contractionPlanCache_; }

            int32_t getDeviceId() const noexcept { return activeGpuId_; }

            bool supportsBF16andTF32() const noexcept { return this->getDeviceProp()->major >= 8; }

            bool isLwblasEnabled() const noexcept { return !disableLwblas_; }

            bool isForcedCD() const noexcept { return forceEqualCD_; }

            int32_t logLevel_; ///< for API logging
        private:
            
            bool disableCache_;
            bool forceEqualCD_;
            bool disableLwblas_;
            DeviceProp deviceProp_;
            int32_t activeGpuId_; ///< gpu-id for which this handle was created
            bool forceEnableTF32_;
            bool forceDisableTF32_;
            lwtensorStatus_t (*handleError_)(const lwtensorStatus_t, const std::string &);
            mutable ContractionPlanCache *contractionPlanCache_;
    };

    // Check that Context fits in lwtensorHandle_t
    static_assert(sizeof(Context) <= sizeof(lwtensorHandle_t),
                  "Size of Context greater than lwtensorHandle_t");
}
