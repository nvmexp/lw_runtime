#pragma once
#include "dnnHeuristic.hpp"
#include "lwtensor/internal/dnnContractionWeightsSSSS.h"
TEST(HeuristicDnnSSSS, numberOfFeatures)
{
    runNumberOfFeaturesTest<ContractionWeightsSSSS>(21);
}
TEST(HeuristicDnnSSSS, computeFeatures_SM72)
{
    runComputeFeaturesTest<ContractionWeightsSSSS>(72,
        {0.8333333333333334,0.9857142857142858,0.7291666666666666,1.0,0.013123359580052492,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.0,0.4980544747081712,0.44,0.684,1,1,0.5,1},
        LWDA_R_32F, LWDA_R_32F, LWDA_R_32F, LWTENSOR_COMPUTE_32F
    );
}
TEST(HeuristicDnnSSSS, computeFeatures_SM80)
{
    runComputeFeaturesTest<ContractionWeightsSSSS>(80,
        {0.8333333333333334,0.9857142857142858,0.65625,1.0,0.013123359580052492,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.0,0.4980544747081712,0.44,0.684,1,1,0.5,1},
        LWDA_R_32F, LWDA_R_32F, LWDA_R_32F, LWTENSOR_COMPUTE_32F
    );
}
TEST(HeuristicDnnSSSS, evaluate)
{
    runEvaluateTest<ContractionWeightsSSSS>(
        {0.6599481,0.075609736,0.23113544,0.81975496,0.09500346,0.28814867,0.2941163,0.14706127,0.11394672,0.22778389}
    );
}
