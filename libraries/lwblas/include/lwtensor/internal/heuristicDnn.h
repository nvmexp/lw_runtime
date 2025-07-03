#pragma once

#if defined(__x86_64__) || defined(_MSC_VER)

#include <gemm.h>

#include "lwtensor/internal/types.h"
#include "lwtensor/internal/typesEx.h"
#include "lwtensor/internal/context.h"
#include "lwtensor/internal/dnnContractionWeights.h"
#include "lwtensor/internal/featuresUtils.h"

namespace LWTENSOR_NAMESPACE
{

template <typename Weights_>
class HeuristicDnn : public Heuristic<ContractionDescriptorInternal, CandidateInfoLwtlass>
{
public:
    using Weights = Weights_;

    HeuristicDnn() : weights() {}

    std::string getName() const override { return "HeuristicDnn"; }

    virtual int numberOfFeatures() const { return Weights::Shape::kSize0; }

    virtual void computeFeatures(const ContractionDescriptorInternal &params,
                                 const CandidateInfoLwtlass &candidateInfo,
                                 const DeviceProp* deviceProp,
                                 float* features) const
    {
        const int numSMs = deviceProp->multiProcessorCount;

        const float FLOPS_ARCH_ILW = 1.f / (14000.f * 1e9);
        const float BANDWIDTH_ARCH_ILW = 1.f / (700.f * 1e9);
        // This value denotes the minimal required arithmetic intensity
        //  (for given arch) to achive peak performance (5x to have sufficent leeway).
        const float AI_ARCH = 5.f * BANDWIDTH_ARCH_ILW / FLOPS_ARCH_ILW;

        const long long int m = params.getTotalExtentM();
        const long long int n = params.getTotalExtentN();
        const long long int k = params.getTotalExtentK();

        const extent_type extent2Dm = params.extentM[0] * params.extentM[1];
        const extent_type extent2Dn = params.extentN[0] * params.extentN[1];

        const auto mc_ = candidateInfo.threadblockN; // WARNING: This hack is necessary to account for the fact that ContractionLwtlass swaps the notion of m/n
        const auto nc_ = candidateInfo.threadblockM; // WARNING: This hack is necessary to account for the fact that ContractionLwtlass swaps the notion of m/n
        const auto kc_ = candidateInfo.shapeK0; // this is only the first k-blocking

        // compute remainder and reduce peak perf accordingly
        const float utilizationM = features::getUtilization(extent2Dm, mc_);
        const float utilizationN = features::getUtilization(extent2Dn, nc_);
        const float utilizationK = features::getUtilization(params.extentK[0], kc_)
                                 * features::getUtilization(params.extentK[1], candidateInfo.shapeK1);

        // ensure that we launch enough CTAs
        const int nCTAs = (float) features::getNumThreadblocks(params, mc_, nc_);

        // trans
        const float transA = params.transA_ ? 1.0f : 0.0f;
        const float transB = params.transB_ ? 1.0f : 0.0f;

        // epilogue
        const float time_flops = (2.0f * m * n * k) * FLOPS_ARCH_ILW;
        const float time_epilogue = (2.0f * m * n) * BANDWIDTH_ARCH_ILW;
        const float epilogue_fraction = time_epilogue / (time_epilogue + time_flops);

        const int tile_m = candidateInfo.threadblockM;
        const int tile_n = candidateInfo.threadblockN;

        const int warp_m = candidateInfo.warpM;
        const int warp_n = candidateInfo.warpN;

        // Model tail effect
        const int maxNumCTAsPerSM = candidateInfo.maxCTAsPerSM;
        assert(maxNumCTAsPerSM >= 1);
        const float partial_waves = static_cast<float>(nCTAs) / static_cast<float>(((nCTAs + numSMs - 1) / numSMs) * numSMs);

        const int num_warps_per_sm = (tile_m / warp_m) * (tile_n / warp_n);
        const int threads = 32 * num_warps_per_sm * (candidateInfo.threadblockK / candidateInfo.warpK);
        const float num_threads = static_cast<float>(threads) * static_cast<float>(maxNumCTAsPerSM) / 512.f;

        const float ai = 2.f * static_cast<float>(m) * static_cast<float>(n) * static_cast<float>(k) /
            (static_cast<float>(m) * static_cast<float>(n) + static_cast<float>(m) * static_cast<float>(k) + static_cast<float>(n) * static_cast<float>(k));
        const float arithmetic_complexity = std::min(ai, AI_ARCH) / AI_ARCH;

        const float num_fmas = warp_m * warp_n * candidateInfo.warpK;
        const float num_fma_issued_per_cycle = getNumFmaIssuedPerCycle(params.typeA_, params.typeB_, params.typeC_, params.typeCompute_);
        const float num_fma_per_sm_per_cycle = 32.f;
        const float min_wait_schedule = num_fmas / (num_fma_per_sm_per_cycle * num_fma_issued_per_cycle);

        const float optimal_ai_smem = 64.f;
        const int active_ctas_per_sm = std::min(maxNumCTAsPerSM, (nCTAs + numSMs - 1) / numSMs);
        const int active_warps_per_sm = active_ctas_per_sm * num_warps_per_sm;

        const int vecMeasure = candidateInfo.elementsPerAccessA
                             + candidateInfo.elementsPerAccessB
                             + candidateInfo.elementsPerAccessC;

        features[ 0] = utilizationM;
        features[ 1] = utilizationN;
        features[ 2] = utilizationK;
        features[ 3] = partial_waves;
        features[ 4] = arithmetic_complexity;
        features[ 5] = epilogue_fraction;
        features[ 6] = transA;
        features[ 7] = transB;
        features[ 8] = num_threads;
        features[ 9] = static_cast<float>(tile_m) / static_cast<float>(128);
        features[10] = static_cast<float>(tile_n) / static_cast<float>(128);
        features[11] = static_cast<float>(warp_m) / static_cast<float>(64);
        features[12] = static_cast<float>(warp_n) / static_cast<float>(64);
        features[13] = static_cast<float>(maxNumCTAsPerSM) / static_cast<float>(4);
        features[14] = std::min(static_cast<float>(candidateInfo.localMemoryUsage) / 1000.f, 1.f);
        features[15] = std::min(static_cast<float>(candidateInfo.avgLDS) / 300.f, 1.f);
        features[16] = std::min(static_cast<float>(candidateInfo.avgAntidep) / 50.f, 1.f);
        features[17] = std::min(static_cast<float>(num_warps_per_sm) * num_fma_issued_per_cycle, 1.f);
        features[18] = static_cast<float>(tile_m * tile_n / (tile_m + tile_n)) / optimal_ai_smem;
        features[19] = std::min(static_cast<float>(active_warps_per_sm) * num_fma_issued_per_cycle, 1.f);
        features[20] = std::min(static_cast<float>(vecMeasure) / 32.f, 1.f);
        features[21] = candidateInfo.numModesContracted > 2 ? 1.f : 0.f;
        features[22] = candidateInfo.shapeK1 > 1 ? 1.f : 0.f;
        features[23] = params.nmodeK > 1 ? 1.f : 0.f;
    }

    virtual void evaluate(int numberOfCandidates,
                          const float* features,
                          float* scores) const
    {
        const int MAX_NUMBER_OF_CANDIDATES = 10;
        using Shape = typename Weights::Shape;

        /* dnnContraction<typename Weights::Shape>(numberOfCandidates, features, scores, weights); */
        assert(numberOfCandidates <= MAX_NUMBER_OF_CANDIDATES);
        assert(Shape::kSize1 >= Shape::kSize2);
        assert(Shape::kSize1 >= Shape::kSize3);
        assert(Shape::kSize1 >= Shape::kSize4);

        float buffer1[MAX_NUMBER_OF_CANDIDATES * Shape::kSize1];
        float buffer2[MAX_NUMBER_OF_CANDIDATES * Shape::kSize1];

        const int batchSize = numberOfCandidates;

        lwBLASLt::dnnHeuristic::lin_act<Shape::kSize0, Shape::kSize1, 1>(batchSize, features, buffer1, weights.weights1_.data(), weights.bias1_.data());
        lwBLASLt::dnnHeuristic::lin_act<Shape::kSize1, Shape::kSize2, 1>(batchSize,  buffer1, buffer2, weights.weights2_.data(), weights.bias2_.data());
        lwBLASLt::dnnHeuristic::lin_act<Shape::kSize2, Shape::kSize3, 1>(batchSize,  buffer2, buffer1, weights.weights3_.data(), weights.bias3_.data());
        lwBLASLt::dnnHeuristic::lin_act<Shape::kSize3, Shape::kSize4, 0>(batchSize,  buffer1,  scores, weights.weights4_.data(), weights.bias4_.data());
    }

private:
    float getNumFmaIssuedPerCycle(lwdaDataType_t typeA, lwdaDataType_t typeB, lwdaDataType_t typeC, lwtensorComputeType_t typeComp) const
    {
        // dddd
        if (typeA == LWDA_R_64F && typeB == LWDA_R_64F && typeC == LWDA_R_64F && typeComp == LWTENSOR_COMPUTE_64F)
        {
            return 0.25f;
        }

        return 0.5f;
    }

    Weights weights;
};

} // namespace LWTENSOR_NAMESPACE

#endif
