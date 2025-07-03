#pragma once

#include <cassert>
#include <float.h>
#include <math.h>
#include <stdio.h>

#include <lwtensor/internal/candidateContainer.h>
#include <lwtensor/internal/computeEngine.h>
#include <lwtensor/internal/defines.h>

namespace LWTENSOR_NAMESPACE
{
    class Context;

    ComputeEngineBase<ContractionDescriptorInternal>* getContractionEngineLwtlass();

#ifdef LWTENSOR_ENABLE_LWTE
    CandidateContainer<ContractionDescriptorInternal>* getContractionContainer_lwte_sm70_dddd();
    CandidateContainer<ContractionDescriptorInternal>* getContractionContainer_lwte_sm70_ssss();
#endif

    lwtensorStatus_t tcValidateInput( const Context* ctx,
                                 const TensorDescriptor* descA, const mode_type* modeA,
                                 const TensorDescriptor* descB, const mode_type* modeB,
                                 const TensorDescriptor* descC, const mode_type* modeC,
                                 const TensorDescriptor* descD, const mode_type* modeD,
                                 lwtensorComputeType_t typeCompute, lwtensorAlgo_t algo,
                                 bool isReduction);
}
