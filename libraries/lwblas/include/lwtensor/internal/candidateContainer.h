#pragma once

#include <lwtensor/internal/initializable.h>
#include <lwtensor/internal/typesEx.h>
#include <lwtensor/internal/context.h>
#include <lwtensor/internal/defines.h>

namespace LWTENSOR_NAMESPACE
{
template<typename T, typename Array, typename Params, typename CandidateInfo, int end, int idx>
struct ContainerInit
{
    void operator()(const T &kernels, Array &arr) const
    {
        arr[idx] = (CandidateTyped<Params, CandidateInfo>*)&std::get<idx>(kernels);
        ContainerInit<T, Array, Params, CandidateInfo, end, idx+1>{}(kernels, arr);
    }
};

template<typename T, typename Array, typename Params, typename CandidateInfo, int end>
struct ContainerInit<T, Array, Params, CandidateInfo, end, end>
{
    void operator()(const T &kernels, Array &arr) const {(void)kernels; (void)arr;}
};

template<typename Params_>
class CandidateContainer
{
    public:
    using Params = Params_; //ContractionDescriptorInternal;

    virtual void init() = 0;

    /// execute
    virtual lwtensorStatus_t operator()(const Context *ctx,
            const Params &params,
            const void* alpha, const void *A,
            const void* beta, const void *B,
            const void* gamma,  const void *C, void *D,
            void* workspace, uint64_t workspaceSize, lwdaStream_t stream, const int32_t candidateIdx) const = 0;

    /// get specific kernel
    virtual lwtensorStatus_t getCandidateFromRank(const Context *ctx,
            const Params& params,
            const size_t workspaceSize,
            const int32_t kernelRank,
            int32_t &candidateIdx) const = 0;

    /// get kernel selected by heuristic
    virtual lwtensorStatus_t getCandidateHeuristic(const Context *ctx,
            const Params& params,
            const size_t workspaceSize,
            int32_t &candidateIdx) const = 0;

    virtual lwtensorStatus_t isApplicable(const Context *ctx, const Params& params) const = 0;

    virtual lwtensorStatus_t getCandidatePtr(const int32_t candidateIdx,
                                             const Candidate<Params> *&candidate) const = 0;
};

/**
 * This class encapsulates a collection of contraction kernels for given Traits (types, arch, ...)
 */
template<typename Traits, // should be ContractionTraits
         typename Params,
         typename CandidateInfo,
         typename CandidateHeuristic,
         typename ...Kernels>
class CandidateContainerTyped : public CandidateContainer<Params>
{
    public:
        using Traits_ = Traits;

        static const int kNumKernels = sizeof...(Kernels);

        CandidateContainerTyped()
            : heuristic_()
        {
            ContainerInit<std::tuple<Kernels...>,
                          std::array<CandidateTyped<Params, CandidateInfo>*, kNumKernels>,
                          Params,
                          CandidateInfo,
                          kNumKernels, 0>{}(kernels_, candidates_);
        }

        void init()
        {
            for (auto& candidate : candidates_)
            {
                candidate->init();
            }
        }

        /**
         * Performs the tensor contraction for the given candidate
         * \param[in] candidateIdx Index into the candidate array (similar to algo)
         */
        lwtensorStatus_t operator()( const Context *ctx,
                const Params &params,
                const void* alpha, const void *A,
                const void* beta, const void *B,
                const void* gamma,  const void *C, void *D,
                void* workspace, uint64_t workspaceSize, lwdaStream_t stream,
                const int32_t candidateIdx) const
        {
            if (candidateIdx >= kNumKernels)
            {
                RETURN_STATUS(LWTENSOR_STATUS_ILWALID_VALUE);
            }
            assert(candidateIdx >= 0 && candidateIdx < kNumKernels);
            return (*candidates_[candidateIdx])(ctx, params,
                                                alpha, A,
                                                beta, B,
                                                gamma, C, D,
                                             workspace, workspaceSize, stream);
        }

        virtual lwtensorStatus_t isApplicable(const Context *ctx, const Params& params) const
        {
            const DeviceProp* deviceProp = ctx->getDeviceProp();
            return Traits_::isApplicable(deviceProp, params);
        }

        /**
         * Select the candidateIdx belonging to the kernelRank'th ranked canidate
         * \param[in] kernelRank
         * \param[out] candidateIdx
         */
        lwtensorStatus_t getCandidateFromRank(const Context *ctx,
                               const Params& params,
                               const size_t workspaceSize,
                               const int32_t kernelRank,
                               int32_t &candidateIdx) const
        {
            const DeviceProp* deviceProp = ctx->getDeviceProp();
            assert(Traits_::isApplicable(deviceProp, params) == LWTENSOR_STATUS_SUCCESS); // we are checking the applicabilty of the container inside of computeEngine.h

            std::array<int, kNumKernels> applicableCandidates;
            int numberOfCandidates = 0;

            for (int idx = 0; idx < kNumKernels; idx++)
            {
                const CandidateTyped<Params, CandidateInfo>* candidateTmp = candidates_[idx];

                if (candidateTmp->isApplicable(ctx, params, workspaceSize))
                {
                    applicableCandidates[numberOfCandidates] = idx;
                    numberOfCandidates++;
                }

            }

            const int MAX_NUM_FEATURES = 32;
            assert(heuristic_.numberOfFeatures() <= MAX_NUM_FEATURES);

            std::array<float, kNumKernels * MAX_NUM_FEATURES> features;
            std::array<float, kNumKernels> scores;

            for (int i = 0; i < numberOfCandidates; ++i)
            {
                heuristic_.computeFeatures(params, candidates_[applicableCandidates[i]]->getCandidateInfo(), deviceProp, features.data() + i * heuristic_.numberOfFeatures());
            }

            heuristic_.evaluate(numberOfCandidates, features.data(), scores.data());

            // sort applicableCandidates w.r.t. scores
            for (int i = 0; i < numberOfCandidates - 1; ++i)
            {
                int bestId = i;
                float bestScore = scores[i];

                for (int j = i + 1; j < numberOfCandidates; ++j)
                {
                    if (scores[j] < bestScore)
                    {
                        bestId = j;
                        bestScore = scores[j];
                    }
                }

                std::swap(applicableCandidates[bestId], applicableCandidates[i]);
                std::swap(scores[bestId], scores[i]);
            }

            if (kernelRank >= 0 && kernelRank < numberOfCandidates)
            {
                candidateIdx = applicableCandidates[kernelRank];
                return LWTENSOR_STATUS_SUCCESS;
            }
            return LWTENSOR_STATUS_NOT_SUPPORTED;
        }

        /**
         * \param[out] candidateIdx
         */
        lwtensorStatus_t getCandidateHeuristic(const Context* ctx,
                                               const Params& params,
                                               const size_t workspaceSize,
                                               int32_t &candidateIdx) const
        {
            return getCandidateFromRank(ctx, params, workspaceSize, 0, candidateIdx);
        }

        lwtensorStatus_t getCandidatePtr(
                const int32_t candidateIdx, const Candidate<Params> *&candidate) const
        {
            if (candidateIdx >= 0 && candidateIdx < candidates_.size())
            {
                candidate = candidates_[candidateIdx];
                return LWTENSOR_STATUS_SUCCESS;
            }
            RETURN_STATUS(LWTENSOR_STATUS_INTERNAL_ERROR);
        }

    private:
        std::tuple<Kernels...> kernels_;

        CandidateHeuristic heuristic_;

    public:
        std::array<CandidateTyped<Params, CandidateInfo>*, kNumKernels> candidates_;
};
}
