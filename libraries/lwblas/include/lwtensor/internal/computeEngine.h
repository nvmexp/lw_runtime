#pragma once

#include <lwtensor/internal/initializable.h>
#include <lwtensor/internal/candidateContainer.h>
#include <lwtensor/internal/defines.h>
namespace LWTENSOR_NAMESPACE
{

template<typename Params_>
class ComputeEngineBase : public Initializable<821>
{
    public:
        using Params = Params_;

    virtual void init() = 0;

    virtual lwtensorStatus_t operator()(const Context *ctx,
                const Params &params,
                const void* alpha, const void *A,
                const void* beta,  const void *B,
                const void* gamma, const void *C, void *D,
                void* workspace, uint64_t workspaceSize, lwdaStream_t stream,
                const uint32_t candidateIdx,
                const uint32_t containerIdx) const = 0;

    virtual lwtensorStatus_t getCandidate(const Context *ctx,
                                      const Params& params,
                                      const size_t workspaceSize,
                                      const int32_t kernelRank,
                                      int32_t &candidateIdx,
                                      int32_t &containerIdx) const = 0;

    virtual lwtensorStatus_t getCandidatePtr(const int32_t candidateIdx,
                                             const int32_t containerIdx,
                                             const Candidate<Params> *&candidate) const = 0;
};

#ifdef LWTENSOR_ENABLE_LWTE
const ComputeEngineBase<ContractionDescriptorInternal>* getContractionEngineLwte();
#endif

template<int kNumTypedContainers, typename Params>
class ComputeEngine : public ComputeEngineBase<Params>
{
    public:
        ComputeEngine(const std::array<CandidateContainer<Params>*, kNumTypedContainers> &&typedContainers)
            : typedContainers_(typedContainers)
        {
            this->unsetInitialized();
        }

        virtual void init()
        {
            if (this->isInitialized()) return;
            for(auto& container : typedContainers_)
            {
                container->init();
            }
            this->setInitialized();
        }

        /// execute
        virtual lwtensorStatus_t operator()(const Context *ctx,
                const Params &params,
                const void* alpha, const void *A,
                const void* beta,  const void *B,
                const void* gamma, const void *C, void *D,
                void* workspace, uint64_t workspaceSize, lwdaStream_t stream,
                const uint32_t candidateIdx,
                const uint32_t containerIdx) const
        {
            (void) ctx;
            assert(containerIdx < kNumTypedContainers);
            return (*typedContainers_[containerIdx])(ctx,
                    params,
                    alpha, A,
                    beta,  B,
                    gamma, C, D,
                    workspace, workspaceSize, stream, candidateIdx);
        } 

        lwtensorStatus_t getCandidate(const Context *ctx,
                                      const Params& params,
                                      const size_t workspaceSize,
                                      const int32_t kernelRank,
                                      int32_t &candidateIdx,
                                      int32_t &containerIdx) const
        {
            if (! Contractiolwariant::isDefaultKernel(kernelRank))
            {
                RETURN_STATUS(this->getCandidateFromRank(ctx, params, workspaceSize, kernelRank, candidateIdx, containerIdx));
            }
            else
            {
                /** Otherwise, select from a set of block sizes. */
                RETURN_STATUS(this->getCandidateFromHeuristic(ctx, params, workspaceSize, candidateIdx, containerIdx));
            }
        }

        inline lwtensorStatus_t getCandidatePtr(const int32_t candidateIdx,
                                                const int32_t containerIdx,
                                                const Candidate<Params> *&candidate) const
        {
            if (containerIdx >= 0 && containerIdx < typedContainers_.size())
            {
                return typedContainers_[containerIdx]->getCandidatePtr(candidateIdx, candidate);
            }
            else
            {
                RETURN_STATUS(LWTENSOR_STATUS_INTERNAL_ERROR);
            }
        }

    private:
        /// get specific kernel
        lwtensorStatus_t getCandidateFromRank(const Context *ctx,
                const Params& params,
                const size_t workspaceSize,
                const int32_t kernelRank,
                int32_t &candidateIdx,
                int32_t &containerIdx) const
        {
            containerIdx = 0;
            for (auto &container : typedContainers_)
            {
                if (container->isApplicable(ctx, params) == LWTENSOR_STATUS_SUCCESS)
                {
                    if (container->getCandidateFromRank(ctx, params, workspaceSize, kernelRank, candidateIdx)
                            == LWTENSOR_STATUS_SUCCESS)
                    {
                        return LWTENSOR_STATUS_SUCCESS;
                    }
                    else
                    {
                        // once we found a container that's applicable we won't search for other containers;
                        // otherwise we can run into the situation that --depending on kernelRank-- some
                        // candidates are form a certain container (i.e., cc + datatype combination), while others
                        // are from another container (e.g., from cc61 fallback, which is something that we won't
                        // use for autotuning anyway)
                        return LWTENSOR_STATUS_NOT_SUPPORTED;
                    }
                }
                containerIdx++;
            }
            RETURN_STATUS(LWTENSOR_STATUS_NOT_SUPPORTED);
        }

        /// get kernel selected by heuristic
        lwtensorStatus_t getCandidateFromHeuristic(const Context *ctx,
                                      const Params& params,
                                      size_t workspaceSize,
                                      int32_t &candidateIdx,
                                      int32_t &containerIdx) const
        {
            return getCandidateFromRank(ctx, params, workspaceSize, 0, candidateIdx, containerIdx);
        }

        std::array<CandidateContainer<Params>*, kNumTypedContainers> typedContainers_; ///< collection of all 
};
}
