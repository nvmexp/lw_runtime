#pragma once

#include <list>
#include <utility> // for std::pair

#include <lwtensor/internal/cache.h>
#include <lwtensor/internal/typesEx.h> // for ContractionPlan
#include <lwtensor/internal/exceptions.h>
#include <lwtensor/internal/defines.h>

namespace LWTENSOR_NAMESPACE
{

    class ContractionPlanCache : public Cache<size_t, std::pair<ContractionPlan, float>>
    {
        public:
            // Value = ContractionPlan, measuredTime (float)
            using Plan = ContractionPlan;
            using Parent = Cache<Key, Value>;
            using Cacheline = typename Parent::Cacheline;
            static_assert(sizeof(Cacheline) <= sizeof(lwtensorPlanCacheline_t),
                    "Size of cacheline greater than lwtensorPlanCacheline_t");

            ContractionPlanCache()
            {
                for (unsigned int i = 0; i < kNumMeasurements_; ++i)
                {
                    if (measurements_[i].init() != LWTENSOR_STATUS_SUCCESS)
                    {
                        throw InternalError("Couldn't initialize measurement\n");
                    }
                    availableMeasurements_.pushBack(&measurements_[i]);
                }
            }

            ~ContractionPlanCache()
            {
                for (unsigned int i = 0; i < kNumMeasurements_; ++i)
                {
                    measurements_[i].finalize();
                }
            }

            /**
             * This function measures the time of the provided contraction plan (i.e.,
             * records the events) and it searches through the existing events to see if
             * some have completed already -- completed events will be used to update the
             * cache (if the corresponding plan is better than the plan that is lwrrently
             * stored in the cache.
             */
            lwtensorStatus_t measureAndUpdate(const Plan* plan, Plan::Params &params)
            {
                const std::lock_guard<Mutex> lock(mutex_);

                // update cache (i.e., search for completed measurements and update cache accordingly)
                if (plan->getRequiresMeasurement())
                {
                    this->update();
                }

                lwtensorStatus_t status = LWTENSOR_STATUS_NOT_SUPPORTED;

                if (plan->getRequiresMeasurement() && ! availableMeasurements_.isEmpty())
                {
                    auto measurement = availableMeasurements_.getFront();
                    // consume measurement
                    availableMeasurements_.popFront();
                    outstandingMeasurements_.pushBack(measurement);

                    // execute, measure, update
                    status = (*measurement)(plan, params); 
                }
                else
                {
                    status = (*plan)(params); // no measurements available skip cache-update
                }
                return status;
            }

            lwtensorStatus_t incrementAlgoCount(const Key &key) noexcept
            {
                const std::lock_guard<Mutex> lock(mutex_);

                const auto it = lookup_.find(key);
                if (it != lookup_.end())
                {
                    // cache hit
                    const auto cacheline = it->second;
                    cacheline->incrementAlgoCount();
                    return LWTENSOR_STATUS_SUCCESS;
                }else
                {
                    RETURN_STATUS(LWTENSOR_STATUS_NOT_SUPPORTED);
                }
            }

            lwtensorStatus_t readFromFile(const char filename[], uint32_t &numCachelinesRead)
            {
                const std::lock_guard<Mutex> lock(mutex_);
                outstandingMeasurements_.clear();
                availableMeasurements_.clear();
                for (unsigned int i = 0; i < kNumMeasurements_; ++i)
                {
                    availableMeasurements_.pushBack(&measurements_[i]);
                }
                return Parent::readFromFile(filename, numCachelinesRead);
            }

            lwtensorStatus_t writeToFile(const char filename[]) const
            {
                const std::lock_guard<Mutex> lock(mutex_);
                return Parent::writeToFile(filename);
            }

        private:
            struct Measurement : public IntrusiveList<Measurement>::Member
            {
                Measurement(){}

                lwtensorStatus_t operator() (const Plan* plan, Plan::Params &params)
                {
                    auto stream = params.stream_;
                    HANDLE_ERROR( this->recordStart(stream) );
                    // launch kernel
                    auto status = (*plan)(params); // no measurements available skip cache-update
                    HANDLE_ERROR( this->recordStop(stream) );
                    this->plan_ = *plan; // store plan
                    return status;
                }

                lwtensorStatus_t init()
                {
                    HANDLE_ERROR(lwdaEventCreate(&start_));
                    HANDLE_ERROR(lwdaEventCreate(&stop_));
                    return LWTENSOR_STATUS_SUCCESS;
                }
                lwtensorStatus_t finalize()
                {
                    HANDLE_ERROR(lwdaEventDestroy(start_));
                    HANDLE_ERROR(lwdaEventDestroy(stop_));
                    return LWTENSOR_STATUS_SUCCESS;
                }

                lwtensorStatus_t recordStart(lwdaStream_t stream)
                {
                    HANDLE_ERROR(lwdaEventRecord(start_, stream));
                    return LWTENSOR_STATUS_SUCCESS;
                }

                lwtensorStatus_t recordStop(lwdaStream_t stream)
                {
                    HANDLE_ERROR(lwdaEventRecord(stop_, stream));
                    return LWTENSOR_STATUS_SUCCESS;
                }

                const Plan& getPlan() const noexcept
                {
                    return plan_;
                }

                /**
                 * Returns the elapsed time in seconds
                 */
                float getTime() const
                {
                    float time;
                    auto status = lwdaEventElapsedTime(&time, start_, stop_);
                    if (status == lwdaSuccess)
                    {
                        return time * 1e-3f;
                    } else
                    {
                        //TODO better error handling?
                        lwdaGetLastError(); // reset error
                        return 0.0f;
                    }
                }

                void ilwalidate()
                {
                    plan_.unsetInitialized();
                }

                private:

                lwdaEvent_t start_;
                lwdaEvent_t stop_;
                Plan plan_; // we must store the plan associated to this measurement
            };

            /**
             * Searches for completed measurements (part of outstandingMeasurements_), updates the cache (if the new timing is better than the cached one), and pushes the measurement back to availableMeasurements_
             */
            lwtensorStatus_t update()
            {
                for( auto it = outstandingMeasurements_.begin(); it != outstandingMeasurements_.end(); )
                {
                    Measurement* measurement = *it;
                    float time = measurement->getTime();
                    if (time > 0.0f)
                    {
                        const auto &plan = measurement->getPlan();
                        const auto key = plan.getKey();

                        uint32_t algoCount = 0;
                        Value value;
                        if (this->get(key, value, algoCount))
                        {
                            // cache-hit
                            const float timeCached = value.second;
                            if (timeCached > time)
                            {
                                // only update cache if the new measurement is faster than
                                // the cached plan
                                this->put(key, std::pair<Plan,float>(plan, time), algoCount); // leave algoCount unchanged
                            }
                        }

                        // free-up measurement
                        it = outstandingMeasurements_.erase(it); // advance iterator
                        availableMeasurements_.pushBack(measurement);
                    }
                    else
                    {
                        it++; // advance iterator
                    }
                }
                return LWTENSOR_STATUS_SUCCESS;
            }

            static const uint32_t kNumMeasurements_ = 8;
            Measurement measurements_[kNumMeasurements_]; ///< used to measure elapsed time of a kernel
            IntrusiveList<Measurement> outstandingMeasurements_; ///< this list keeps track of the outstanding measurements
            IntrusiveList<Measurement> availableMeasurements_; ///< this list keeps track of the free measurements

    };
    static_assert(sizeof(ContractionPlanCache) <= sizeof(lwtensorPlanCache_t),
            "Size of PlanCache greater than lwtensorPlanCache_t");
}
