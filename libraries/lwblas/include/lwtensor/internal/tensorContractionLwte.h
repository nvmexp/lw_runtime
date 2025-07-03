#pragma once

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-function"

#include <gett/kernels/gett_template.hpp>

#include <lwtensor/internal/contractionDescriptor.h>
#include <lwtensor/internal/candidateContainer.h>
#include <lwtensor/internal/typesEx.h>
#include <lwtensor/internal/util.h>
#include <lwtensor/internal/featuresUtils.h>
#include <lwtensor/internal/defines.h>
namespace LWTENSOR_NAMESPACE
{

struct CandidateInfoLwte
{
    static const int kThreadblockSize = 5;

    std::array<int, kThreadblockSize> threadblockM;
    std::array<int, kThreadblockSize> threadblockN;
    std::array<int, kThreadblockSize> threadblockK;
    int numM, numN, numK;

    bool useStreamK;
};

/**
 * Instantiation of a concrete Lwte-based tensor contraction
 */
template<
    typename Traits, // should be ContractionTraits
    int kElementsPerAccessA,
    int kElementsPerAccessB,
    int kElementsPerAccessC,
    typename kThreadblockM, typename kThreadblockN, typename kThreadblockK,
    bool kUseStreamK,
    typename ArchTag
    >
class ContractionLwte : public CandidateTyped<ContractionDescriptorInternal, CandidateInfoLwte>
{
    using Params = ContractionDescriptorInternal;
    public:
        ContractionLwte(){}

        static const bool kUseStreamK_ = kUseStreamK;
        static const int kMaxRankM_ = 5;
        static const int kMaxRankN_ = 5;
        static const int kMaxRankK_ = 3;
        static const int kMaxRankL_ = kUseStreamK_ ? 0 : 2;

        using ElementA_ = typename Traits::ElementA_;
        using ElementB_ = typename Traits::ElementB_;
        using ElementC_ = typename Traits::ElementC_;
        using ElementScalar_ = typename Traits::ElementScalar_;
        using ElementCompute_ = typename Traits::ElementCompute_;

        static const int kElementsPerAccessA_ = kElementsPerAccessA;
        static const int kElementsPerAccessB_ = kElementsPerAccessB;
        static const int kElementsPerAccessC_ = kElementsPerAccessC;

        using kThreadblockM_ = kThreadblockM;
        using kThreadblockN_ = kThreadblockN;
        using kThreadblockK_ = kThreadblockK;

        using ArchTag_ = ArchTag;
        static const int ccTarget_ = Traits::ccTarget_;

        using Contraction = typename lwte::Contraction<
                                 ElementA_, ElementB_, ElementC_, ElementScalar_, ElementScalar_,
                                 kThreadblockM_, kThreadblockN_, kThreadblockK_,
                                 kMaxRankM_, kMaxRankN_, kMaxRankK_, kMaxRankL_, kUseStreamK_>;

        void init()
        {
            // nothing to do for LwTe (at a later stage we might also want to querry the
            // number of regs used)
        }

        lwtensorStatus_t operator()(const Context *ctx,
                const Params &params,
                const void* alpha, const void *A,
                const void* unused, const void *B,
                const void* beta,  const void *C, void *D,
                void* workspace, uint64_t workspaceSize, lwdaStream_t stream) const
        {
            (void)unused;
            const int numCTAs = this->getNumThreadblocks(ctx, params);

            int strideAm[kMaxRankM_];
            int strideCm[kMaxRankM_];
            int strideBn[kMaxRankN_];
            int strideCn[kMaxRankN_];
            int strideAk[kMaxRankK_];
            int strideBk[kMaxRankK_];
            int strideAl[kMaxRankL_];
            int strideBl[kMaxRankL_];
            int strideCl[kMaxRankL_];
            for(int i=0; i < kMaxRankM_; ++i)
            {
                strideAm[i] = params.strideAm[i];
                strideCm[i] = params.strideCm[i];
            }
            for(int i=0; i < kMaxRankN_; ++i)
            {
                strideBn[i] = params.strideBn[i];
                strideCn[i] = params.strideCn[i];
            }
            for(int i=0; i < kMaxRankK_; ++i)
            {
                strideAk[i] = params.strideAk[i];
                strideBk[i] = params.strideBk[i];
            }
            for(int i=0; i < kMaxRankL_; ++i)
            {
                strideAl[i] = params.strideAl[i];
                strideBl[i] = params.strideBl[i];
                strideCl[i] = params.strideCl[i];
            }

            typename Contraction::Params lwteParams(
                    (int*)params.extentM, params.nmodeM, // TODO int cast
                    (int*)params.extentN, params.nmodeN,
                    (int*)params.extentK, params.nmodeK,
                    (int*)params.extentL, params.nmodeL,
                    strideAm, strideAk, strideAl,
                    strideBn, strideBk, strideBl,
                    strideCm, strideCn, strideCl);

            HANDLE_ERROR(instance_(lwteParams,
                          A, B, C, D,
                          alpha, beta,
                          workspace, workspaceSize,
                          numCTAs,
                          stream));

            return LWTENSOR_STATUS_SUCCESS;
        }

        template<typename Tuple, int end, int level>
        struct TupleHelper
        {

            static void getBlocking(std::array<int, CandidateInfoLwte::kThreadblockSize>& arr)
            {
                arr[level] = lwte::get<level>(Tuple{}).value;
                TupleHelper<Tuple, end, level+1>::getBlocking(arr);
            }

            static std::string toString()
            {
                std::string ret("");
                if (level == 0)
                {
                    ret += "(";
                }

                ret += std::to_string(lwte::get<level>(Tuple{}).value);

                if (level + 1 < end)
                {
                    ret += ", " + TupleHelper<Tuple, end, level + 1>::toString();
                }
                else
                {
                    ret += ")";
                }
                return ret;
            }

        };
        template<typename Tuple, int end>
        struct TupleHelper<Tuple, end, end>
        {
            static void getBlocking(std::array<int, CandidateInfoLwte::kThreadblockSize>& arr)
            {
            }

            static std::string toString()
            {
                return "";
            }
        };

        CandidateInfoLwte getCandidateInfo() const
        {
            CandidateInfoLwte candidateInfo;

            TupleHelper<kThreadblockM_, rank(kThreadblockM_{}).value, 0>::getBlocking(candidateInfo.threadblockM);
            TupleHelper<kThreadblockN_, rank(kThreadblockN_{}).value, 0>::getBlocking(candidateInfo.threadblockN);
            TupleHelper<kThreadblockK_, rank(kThreadblockK_{}).value, 0>::getBlocking(candidateInfo.threadblockK);

            candidateInfo.numM = rank(kThreadblockM_{}).value;
            candidateInfo.numN = rank(kThreadblockN_{}).value;
            candidateInfo.numK = rank(kThreadblockK_{}).value;

            candidateInfo.useStreamK = kUseStreamK_;

            return candidateInfo;
        }

        bool isApplicable(const Context *ctx,
                          const Params &params,
                          const size_t workspaceSize) const
        {
            typename Contraction::Params lwteParams( // TODO avoid copy (i.e., use our desc)
                    (int*)params.extentM, params.nmodeM, // TODO int cast
                    (int*)params.extentN, params.nmodeN,
                    (int*)params.extentK, params.nmodeK,
                    (int*)params.extentL, params.nmodeL,
                    (int*)params.strideAm, (int*)params.strideAk, (int*)params.strideAl,
                    (int*)params.strideBn, (int*)params.strideBk, (int*)params.strideBl,
                    (int*)params.strideCm, (int*)params.strideCn, (int*)params.strideCl);

            return instance_.isApplicable(lwteParams, workspaceSize);
        }

        void print() const
        {
            size_t size = 2048;
            char buffer[size];
            info(buffer, size);
            fprintf(stderr, "%s", buffer);
        }

        int info(char* dst, size_t sz) const
        {
            return snprintf(dst, sz, " Ker: %s x %s x %s, %d %d %d largek:%d\n",
                    TupleHelper<kThreadblockM_, rank(kThreadblockM_{}).value, 0>::toString().c_str(),
                    TupleHelper<kThreadblockN_, rank(kThreadblockN_{}).value, 0>::toString().c_str(),
                    TupleHelper<kThreadblockK_, rank(kThreadblockK_{}).value, 0>::toString().c_str(),
                    kElementsPerAccessA_, kElementsPerAccessB_, kElementsPerAccessC_, kUseStreamK_);
        }

    private:
        inline int getNumThreadblocks(const Context *ctx, const Params &params) const noexcept
        {
            // try to reduce semaphore preasure for large-k
            int factor = std::min(params.getTotalExtentM(), params.getTotalExtentN()) * 20 < params.getTotalExtentK() ? 1 : 2;
            return factor * ctx->getDeviceProp()->multiProcessorCount;
        }

    private:
        Contraction instance_;
};


class HeuristicLwte : public Heuristic<ContractionDescriptorInternal, CandidateInfoLwte>
{
    public:
    int numberOfFeatures() const override { return 1; }

    virtual void computeFeatures(const ContractionDescriptorInternal &params,
                                 const CandidateInfoLwte &candidateInfo,
                                 const DeviceProp* deviceProp,
                                 float* features) const
    {
        float penalty = 1.f;
        float usefulFlopRatio = 1.f;

        float k = 1.0f;
        for (int i=0; i < params.nmodeK; ++i)
        {
            k *= static_cast<float>(params.extentK[i]);
        }

        if (candidateInfo.useStreamK)
        {
            penalty = (k < 128.f) ? k / 128.f : 1.0f; // penalize stream-k for small k
        }
        else
        {
            // penalize non-stream-k if not enough threadblocks are available

            int numThreadblocks = getNumThreadblocks<5>(candidateInfo.threadblockM, candidateInfo.numM, params.extentM, params.nmodeM);
            numThreadblocks *= getNumThreadblocks<5>(candidateInfo.threadblockN, candidateInfo.numN, params.extentN, params.nmodeN);

            float maxNumCTAsPerSM = 1.0f; // TODO
            float numWaves = numThreadblocks / (maxNumCTAsPerSM * float(deviceProp->multiProcessorCount)); // This value should be close (from below) to an integer
            // we simply assume that full waves attain full perf, while partial waves only attain partial perf
            penalty = numWaves / ceilf(numWaves);
        }

        usefulFlopRatio = 1.0f;
        usefulFlopRatio *= utilization<5>(candidateInfo.threadblockM, candidateInfo.numM, params.extentM, params.nmodeM);
        usefulFlopRatio *= utilization<5>(candidateInfo.threadblockN, candidateInfo.numN, params.extentN, params.nmodeN);
        usefulFlopRatio *= utilization<5>(candidateInfo.threadblockK, candidateInfo.numK, params.extentK, params.nmodeK);

        features[0] = penalty * usefulFlopRatio;
    }

    virtual void evaluate(int numberOfCandidates,
                          const float* features,
                          float* scores) const
    {
        for (int i = 0; i < numberOfCandidates; ++i)
            scores[i] = features[i];
    }

    private:

        template <int N>
        float utilization(const std::array<int, N>& blocking, int numBlocking,
                          const extent_type *extents, int32_t numModes) const
        {
            float u = 1.f;
            for (int level = 0; level < numBlocking && level < numModes; ++level)
            {
                u *= features::getUtilization(extents[level], blocking[level]);
            }

            return u;
        }

        template <int N>
        int getNumThreadblocks(const std::array<int, N>& blocking, int numBlocking,
                               const extent_type *extents, int32_t numModes) const
        {
            int numThreadblocks = 1;
            int level = 0;
            for (; level < numBlocking; ++level)
            {
                numThreadblocks *= ((extents[level] + blocking[level] - 1)/ blocking[level]);
            }
            for (; level < numModes; ++level)
            {
                numThreadblocks *= extents[level];
            }

            return numThreadblocks;
        }

};

} // end LWTENSOR_NAMESPACE
#pragma GCC diagnostic pop
