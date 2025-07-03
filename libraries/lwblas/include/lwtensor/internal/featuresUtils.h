#pragma once

#include "lwtensor/internal/types.h"
#include "lwtensor/internal/typesEx.h"

namespace LWTENSOR_NAMESPACE {

struct CandidateInfoLwtlass
{
    public:

        int threadblockM, threadblockN, threadblockK;
        int shapeK0, shapeK1;
        int numModesContracted;
        int numBlockedModesContracted;
        int warpM, warpN, warpK;
        int elementsPerAccessA, elementsPerAccessB,elementsPerAccessC;
        int blockedModesM, blockedModesN;
        int numThreads;
        int maxCTAsPerSM;
        int localMemoryUsage, waitSchedule;
        int avgLDS, avgLDG, avgAntidep;
};

inline float variable(const char* name, const float initial)
{
#ifndef LWTENSOR_EXPOSE_INTERNAL
    return initial;
#endif
    auto elw = getelw(name);
    if (! elw) return initial;
    auto end = elw;
    float new_value = strtod(elw, &end);
    if (end == elw) return initial;
    return new_value;
}

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


namespace features
{

static const int kBlockedModesN_ = 2;
static const int kBlockedModesM_ = 2;

static int getNumThreadblocks(const ContractionDescriptorInternal &params, const int MC, const int NC)
{
    int nCTAs = 1;

    // blocked modes
    int m = 1;
    for ( int i = 0; i < kBlockedModesM_ && i < params.nmodeM; i ++ )
    {
        m *= params.extentM[i];
    }
    nCTAs *= (m + MC - 1) / MC;
    // unblocked modes
    for ( int i = kBlockedModesM_; i < params.nmodeM; i ++ )
    {
        nCTAs *= params.extentM[ i ];
    }

    // blocked modes
    int n = 1;
    for ( int i = 0; i < kBlockedModesN_ && i < params.nmodeN; i ++ )
    {
        n *= params.extentN[i];
    }
    nCTAs *= (n + NC - 1) / NC;
    // unblocked modes
    for ( int i = kBlockedModesN_; i < params.nmodeN; i ++ )
    {
        nCTAs *= params.extentN[ i ];
    }

    for ( int i = 0; i < params.nmodeL; ++ i )
    {
        nCTAs *= params.extentL[ i ];
    }

    return nCTAs;
}

static bool isPowerOfTwo(extent_type x)
{
    return (x & (x - 1)) == 0;
}

// extent is blocked using a blocking of 'blocking'
// blocking must be a power of two
static float getUtilization(const extent_type extent, const extent_type blocking)
{
#ifdef DEBUG
    assert(isPowerOfTwo(blocking));
#endif
    auto rounded = ((extent - 1) | (blocking - 1)) + 1;
    return static_cast<float>(extent) / rounded;
//    const float totalwork = std::max(blocking, ((extent + blocking - 1) / blocking) * blocking);
//    return ((float) extent) / totalwork; // usefulwork / totalwork
}

}  // namespace features

}  // namespace LWTENSOR_NAMESPACE
