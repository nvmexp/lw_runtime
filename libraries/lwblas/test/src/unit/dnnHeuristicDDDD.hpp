#pragma once
#include "dnnHeuristic.hpp"
#include "lwtensor/internal/dnnContractionWeightsDDDD.h"
TEST(HeuristicDnnDDDD, numberOfFeatures)
{
    runNumberOfFeaturesTest<ContractionWeightsDDDD>(21);
}
TEST(HeuristicDnnDDDD, computeFeatures_SM72)
{
    runComputeFeaturesTest<ContractionWeightsDDDD>(72,
        {0.8333333333333334,0.9857142857142858,0.7291666666666666,1.0,0.013123359580052492,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.0,0.9961089494163424,0.44,0.684,1,1.0,0.5,1},
        LWDA_R_64F, LWDA_R_64F, LWDA_R_64F, LWTENSOR_COMPUTE_64F
    );
}
TEST(HeuristicDnnDDDD, computeFeatures_SM80)
{
    runComputeFeaturesTest<ContractionWeightsDDDD>(80,
        {0.8333333333333334,0.9857142857142858,0.65625,1.0,0.013123359580052492,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.0,0.9961089494163424,0.44,0.684,1,1.0,0.5,1},
        LWDA_R_64F, LWDA_R_64F, LWDA_R_64F, LWTENSOR_COMPUTE_64F
    );
}
TEST(HeuristicDnnDDDD, evaluate)
{
    runEvaluateTest<ContractionWeightsDDDD>(
        {5.0643066e-11,6.6932896e-12,1.853019e-10,9.981127e-10,1.8204166e-12,5.219024e-13,3.329294e-12,1.0017468e-11,2.818573e-10,1.6094932e-12}
    );
}
