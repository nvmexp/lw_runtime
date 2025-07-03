#include "lwtensor/internal/heuristicsLwtlass.h"
#include "lwtensor/internal/featuresUtils.h"

namespace LWTENSOR_NAMESPACE
{

void HeuristicSimple::computeFeatures(const ContractionDescriptorInternal &params,
                                    const CandidateInfoLwtlass &candidateInfo,
                                    const DeviceProp* deviceProp,
                                    float* features) const
{
    const float peakPerfilw = 1.f / (13000.f * 1e9); // TODO
    const float peakReadBWilw = 1.f / (700.f * 1e9);
    const int numSMs = deviceProp->multiProcessorCount;

    auto typeSizeA = getDataTypeSize(params.typeA_);
    auto typeSizeB = getDataTypeSize(params.typeB_);
    const extent_type m = params.getTotalExtentM();
    const extent_type n = params.getTotalExtentN();
    const extent_type k = params.getTotalExtentK();

    extent_type extentBlockedM = 1;
    for (int i = 0; i < candidateInfo.blockedModesM; i++) extentBlockedM *= params.extentM[i];
    extent_type extentBlockedN = 1;
    for (int i = 0; i < candidateInfo.blockedModesN; i++) extentBlockedN *= params.extentN[i];

    auto mc_ = candidateInfo.threadblockN; // WARNING: This hack is necessary to account for the fact that ContractionLwtlass swaps the notion of m/n
    auto nc_ = candidateInfo.threadblockM; // WARNING: This hack is necessary to account for the fact that ContractionLwtlass swaps the notion of m/n

    /***********************************
     * Performance model
     ***********************************/
    float timeA = (((float) k) * m * typeSizeA) * (n / extentBlockedN) * ((extentBlockedN + nc_ - 1) / nc_) * peakReadBWilw;
    if (mc_ <= 16) timeA *= 1.3f;           // too few data to load => penalize
    if (k * m <= 256 * 256) timeA *= 0.4f; // we assume that this stays in L2
    float timeB = (((float) k) * n * typeSizeB) * (m / extentBlockedM) * ((extentBlockedM + mc_ - 1) / mc_) * peakReadBWilw;
    if (nc_ <= 16) timeB *= 1.3f;           // too few data to load => penalize
    if (k * n <= 256 * 256) timeB *= 0.4f; // we assume that this stays in L2

    // compute remainder and reduce peak perf accordingly
    float utilizationM = features::getUtilization(extentBlockedM, mc_);
    float utilizationN = features::getUtilization(extentBlockedN, nc_);
    float utilizationK = features::getUtilization(params.extentK[0], candidateInfo.shapeK0);
    utilizationK *= features::getUtilization(params.extentK[1], candidateInfo.shapeK1);
    float flopPenalty = utilizationM * utilizationN * utilizationK;

    const float nolwecPenaltiy = (typeSizeA <= 2) ? 0.2 : 0.95; // unpacking 16-bit data is especially expensive
    if (candidateInfo.elementsPerAccessA + candidateInfo.elementsPerAccessB + candidateInfo.elementsPerAccessC <= 3)
    {
        flopPenalty *= nolwecPenaltiy;
    }

    // slightly favor kernels with fewer blocked k-modes
    flopPenalty *= (1.f - 0.005f * candidateInfo.numBlockedModesContracted);
    if (candidateInfo.numModesContracted > 2)
    {
       flopPenalty *= 0.97;
    }

    // ensure that we launch enough CTAs
    float nCTAs = (float) features::getNumThreadblocks(params, mc_, nc_);

    // Model tail effect
    int maxNumCTAsPerSM = candidateInfo.maxCTAsPerSM;
    assert(maxNumCTAsPerSM >= 1);
    // WARNING: the name numWaves is confusing since it doesn't factor in maxNumCTAsPerSM.
    // However, we don't factor maxNumCTAsPerSM on purpose due to this counter example:
    //  - 1) kernel might have: maxNumCTAsPerSM=2, numCTAs = 50
    //  - 2) kernel might have: maxNumCTAsPerSM=1, numCTAs = 25
    // In that case (1) is clearly better (assuming numSMs = 80)
    float numWaves = nCTAs / (float(numSMs)); // This value should be close (from below) to an integer
    // we simply assume that full waves attain full perf, while partial waves only attain partial perf
    // load-balancing among SMs
    flopPenalty *= numWaves / ceilf(numWaves);

    float timeFlops = ((2.f * m) * n * k) * peakPerfilw / flopPenalty; // TODO correct prefactor

    // encourage larger tiles (for both energy efficiency as well as ability to hide latencies via ILP):
    //  - assume that a 256x256 blocking is optimal substract 1% perf whenever one dimension is halfed
    constexpr int optimalTilesize = 256;
    assert(optimalTilesize % mc_ == 0 && optimalTilesize % nc_ == 0);
    float penalty = 1 - 0.01 * ((optimalTilesize / mc_) + (optimalTilesize / nc_));

    // Penalize if we don't launch enough warps to saturate a SM
    constexpr int kNumCoresPerSM = 64;
    constexpr int kNumThreadsPerWarp = 32;
    constexpr int kNumRequiredWarps = kNumCoresPerSM / kNumThreadsPerWarp;
    const int kNumResidentWarps = candidateInfo.numThreads / kNumThreadsPerWarp * std::min(float(maxNumCTAsPerSM), ceilf(numWaves));
    if (kNumResidentWarps < kNumRequiredWarps)
    {
        penalty *= kNumResidentWarps / float(kNumRequiredWarps); // we don't have enough warps to fill-up one SM
    }

    float totalTime = std::max(timeFlops + (timeA + timeB) * 0.1f, 0.1f * timeFlops + timeA + timeB) / penalty; // assume almost-perfect overlap

    features[0] = totalTime;
}


void HeuristicSimple::evaluate(int numberOfCandidates,
                             const float* features,
                             float* scores) const
{
    for (int i = 0; i < numberOfCandidates; ++i)
        scores[i] = features[i];
}

} // namespace LWTENSOR_NAMESPACE
