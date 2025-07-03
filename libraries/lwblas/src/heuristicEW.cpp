#include "lwtensor/internal/heuristicEW.h"
#include <lwtensor/internal/featuresUtils.h>

#include <cassert>

#include <array>


namespace LWTENSOR_NAMESPACE
{


namespace features
{


template<bool initial>
bool toggle(const char* name, const char* value)
{
#ifndef LWTENSOR_EXPOSE_INTERNAL
    return initial;
#endif
    auto elw = getelw(name);
    if (! elw) return initial;
    if (strcmp(elw, value) == 0) return true;
    return false;
}


// TODO: Remove vectors for std::initializer_list or something like that...
template<int N>
float piecewiseLinear(const float x, const std::array<float, N>& xs, const std::array<float, N>& ys)
{
    for (int i = 0; i < xs.size() - 1; i++)
    {
        if (xs[i] <= x && x <= xs[i+1])
        {
            return ys[i] + (x - xs[i]) / (xs[i+1] - xs[i]) * (ys[i+1] - ys[i]);
        }
    }
    if (x < xs[0]) return ys[0];
    if (x > xs.back()) return ys.back();
    assert(0);
    return ys[0]; // this should never be reached
}


/// Computes thread layout given dims, blocking, vectorization, threads if vectorization oclwrs along `dim`
void computeThreadLayout(
        const int ndimTile, const uint32_t* blocking,
        const int vector, int threads, const int dim,
        int* threadLayout)
{
#ifdef __GNUC__
    const int vectorLog = __builtin_ffs(vector);
    int blockingLog[3];
    blockingLog[0] = __builtin_ffs(static_cast<int>(blocking[0]));
    blockingLog[1] = __builtin_ffs(static_cast<int>(blocking[1]));
    blockingLog[2] = __builtin_ffs(static_cast<int>(blocking[2]));
    int threadsLog = __builtin_ffs(threads);
    threadLayout[dim] = std::min(blockingLog[dim] - vectorLog, threadsLog);
    threadsLog -= threadLayout[dim];
    threadLayout[dim] = 1 << threadLayout[dim];
    for (int i = 0; i < ndimTile; i++)
    {
        if (i == dim) continue;
        threadLayout[i] = std::min(blockingLog[i], threadsLog);
        threadsLog -= threadLayout[i];
        threadLayout[i] = 1 << threadLayout[i];
    }
    assert(threadsLog == 0);
#else
    threadLayout[dim] = std::min(static_cast<int>(blocking[dim]) / vector, threads);
    threads /= threadLayout[dim];
    for (int i = 0; i < ndimTile; i++)
    {
        if (i == dim) continue;
        threadLayout[i] = std::min(static_cast<int>(blocking[i]), threads);
        threads /= threadLayout[i];
    }
    assert(threads == 1);
#endif
}


/// Computes what percentage of threads actively partake in this read operation
float computeMaskedThreadRatio(int extent, const int blocking, const int vector, const int threads)
{
    if ((extent & (blocking - 1)) == 0) return 1;
    int numBlocks = (extent + blocking - 1) / blocking;
    extent = extent & (blocking - 1);
    extent = (extent + vector - 1) / vector;
    extent = extent > threads ? threads : extent;
    return static_cast<float>(numBlocks * threads - threads + extent) / (threads * numBlocks);
}


}  // namespace features


void HeuristicEW::computeFeatures(
        const ElementwiseParameters &params,
        const CandidateInfoEW &kernel,
        const DeviceProp* deviceProp,
        float* features) const
{
    if (features::toggle<false>("LWTENSOR_EW_HEUR", "old"))
    {

        float estimatedTime = 1.0f;

        float effectiveOpsRatio = 1.0f;
        // account for blocking overhead (effective ops)
        for (int i=0; i < std::min(params.nmodeC_, kernel.ndimTile); ++i)
        {
            effectiveOpsRatio *= features::getUtilization(params.extent_[i], kernel.blocking[i]);
        }

        estimatedTime /= effectiveOpsRatio;

        constexpr float maxVectorization = 32.f; // we assume that 32 is the maximal vectorization among all candidates (it doesn't have to be perfect)
        if (effectiveOpsRatio < 0.9)
        {
            // favor smaller vectorization if the blocking causes an overhead: in
            // this case the larger vectorization would cause some threads to be idle
            estimatedTime *= (1.0f - 0.002f * ((maxVectorization - kernel.vectorWidth) / maxVectorization ));
        }

        // favor opPacks that match exactly
        if (kernel.opPack == params.opPack_)
        {
            estimatedTime *= 0.7f;
        }

//        const int totalTensors = params.useA_ + params.useB_ + params.useC_ + 1;
//                   const int numVectorizable = (isVectorizedA && params.useA_) + (isVectorizedB  && params.useB_) +
//                                               (isVectorizedC && params.useC_) + (isVectorizedD);

        // favor more threadblocks
        const int numSMs = deviceProp->multiProcessorCount;
        const auto total_tiles = getTotalTiles(params, kernel.ndimTile, kernel.blocking);

        // we assume that we'll reach full perf once each SM has at least one threadblock.
        // we model perf like this:
        // 1.0 -                        x
        //     |                    x   .
        //     |                x       .
        //     |            x           .
        //     |        x               .
        // 0.5 -    x                   .
        //     |   x                    .
        //     |  x                     .
        //     | x                      .
        //     |x                       .
        //   0 -,---,-------------------,-
        //      0   1/4               numSMs
        if (total_tiles < 0.25f * numSMs)
        {
            estimatedTime /= (0.5f * (4.0f * total_tiles / numSMs)); // linear interpolation
        }
        else if (total_tiles < numSMs)
        {
            estimatedTime /= ((0.5f + 0.5f * ((total_tiles - 0.25f * numSMs) / ( 0.75f * numSMs)))); // linear interpolation
        }
        features[0] = estimatedTime;
    }
    else
    {
        float estimatedTimeIlw = 1;

        float effectiveOpsRatio = 1;
        for (int i = 0; i < std::min(params.nmodeC_, kernel.ndimTile); i++)
        {
            effectiveOpsRatio *= features::getUtilization(params.extent_[i], kernel.blocking[i]);
        }

        estimatedTimeIlw *= effectiveOpsRatio;

        // Apply a mild penalty to 3d kernels if they are not dominated by masking
        // and actually have to perform a transpose
        if (effectiveOpsRatio > 0.8f)
        {
            if (kernel.ndimTile == 3 && params.strideA_[0] != 1)
            {
                estimatedTimeIlw *= 0.8f;
            }
        }

        // Strongly penalize fallback kernels
        if (kernel.opPack == params.opPack_)
        {
            estimatedTimeIlw *= 10.f;
        }

        const auto total_tiles = getTotalTiles(params, kernel.ndimTile, kernel.blocking);
        const int numSMs = deviceProp->multiProcessorCount;

        float total_tiles_per_sm = static_cast<float>(total_tiles) / numSMs;

        // prefer kernels that have about 0.25 to 2 tiles per SM
        // with deminishing returns afterwards
        estimatedTimeIlw *= features::piecewiseLinear<4>(total_tiles_per_sm, {0, 0.25, 2, 4}, {0, 1, 1, 0.5});

        // For square kernels, prefer those that are "more square" / penalize skewed ones.
        float squareness = static_cast<float>(std::min(kernel.blocking[0], kernel.blocking[1])) / std::max(kernel.blocking[0], kernel.blocking[1]);

        if (params.nmodeC_ == 2)
        {
            estimatedTimeIlw *= features::piecewiseLinear<3>(squareness, {0, 0.2, 1}, {0.5, 0.9, 1});
        }

        // mildly prefer kernels with large transposes
        float transposeSize = kernel.blocking[0] * kernel.blocking[1];

        estimatedTimeIlw *= features::piecewiseLinear<2>(transposeSize, {16, 1024}, {0.9, 1});

        // mildly prefer kernels with many threads
        estimatedTimeIlw *= features::piecewiseLinear<2>(kernel.numThreads, {32, 512}, {0.9, 1});

        // prefer 1d kernels that contain more than one block in the 1d dimension
        // or where the one block it has is sufficiently full.
        if (kernel.ndimTile == 1)
        {
            if (static_cast<float>(params.extent_[0]) / kernel.blocking[0] > 0.8f)
            {
                estimatedTimeIlw *= 2.0f;
            }
        }

        // maskedThreadRatio is like effectiveOpsRatio, but more precisie:
        // it considers whether threads actually execute the load instruction (albeit masked out)
        // or whether the load is skipped entirely
        float maskedThreadRatio = 1;

        if (kernel.ndimTile == 1)
        {
            maskedThreadRatio *= features::computeMaskedThreadRatio(params.extent_[0], kernel.blocking[0], kernel.vectorWidth, kernel.numThreads);
        }
        else
        {
            int threadLayoutRead[3], threadLayoutWrite[3];
            features::computeThreadLayout(kernel.ndimTile, kernel.blocking, kernel.vectorWidth, kernel.numThreads, 1, threadLayoutRead);
            features::computeThreadLayout(kernel.ndimTile, kernel.blocking, kernel.vectorWidth, kernel.numThreads, 0, threadLayoutWrite);
            float maskedThreadRatioWrite0 = features::computeMaskedThreadRatio(params.extent_[0], kernel.blocking[0], kernel.vectorWidth, threadLayoutWrite[0]);
            float maskedThreadRatioWrite1 = features::computeMaskedThreadRatio(params.extent_[1], kernel.blocking[1], 1, threadLayoutWrite[1]);
            float maskedThreadRatioRead0 = maskedThreadRatioWrite0;
            float maskedThreadRatioRead1 = maskedThreadRatioWrite1;
            if (params.strideA_[0] != 1)
            {
                maskedThreadRatioRead0 = features::computeMaskedThreadRatio(params.extent_[1], kernel.blocking[1], kernel.vectorWidth, threadLayoutRead[1]);
                maskedThreadRatioRead1 = features::computeMaskedThreadRatio(params.extent_[0], kernel.blocking[0], 1, threadLayoutRead[0]);
            }
            maskedThreadRatio *= (maskedThreadRatioRead0 * maskedThreadRatioRead1 + maskedThreadRatioWrite0 * maskedThreadRatioWrite1) / 2;
            if (kernel.ndimTile == 3)
            {
                maskedThreadRatio *= features::computeMaskedThreadRatio(params.extent_[2], kernel.blocking[2], 1, threadLayoutWrite[2]);
            }
        }

        estimatedTimeIlw *= maskedThreadRatio;
        features[0] = 1.0f / estimatedTimeIlw;
    }

}  // namespace LWTENSOR_NAMESPACE


}  // namespace LWTENSOR_NAMESPACE
