#pragma once

#include <lwtensor/internal/typesEx.h>
#include <lwtensor/internal/defines.h>

namespace LWTENSOR_NAMESPACE
{

    using OpPackIdentity= ElementwiseStaticOpPack<
        lwtensorOperator_t::LWTENSOR_OP_IDENTITY,
        lwtensorOperator_t::LWTENSOR_OP_IDENTITY,
        lwtensorOperator_t::LWTENSOR_OP_IDENTITY,
        lwtensorOperator_t::LWTENSOR_OP_ADD,
        lwtensorOperator_t::LWTENSOR_OP_IDENTITY,
        lwtensorOperator_t::LWTENSOR_OP_ADD>;

    using OpPackGeneric = ElementwiseStaticOpPack<
        lwtensorOperator_t::LWTENSOR_OP_UNKNOWN,
        lwtensorOperator_t::LWTENSOR_OP_UNKNOWN,
        lwtensorOperator_t::LWTENSOR_OP_UNKNOWN,
        lwtensorOperator_t::LWTENSOR_OP_UNKNOWN,
        lwtensorOperator_t::LWTENSOR_OP_UNKNOWN,
        lwtensorOperator_t::LWTENSOR_OP_UNKNOWN>;

    class CandidateInfoEW
    {
        public:
            CandidateInfoEW() {}

            uint32_t vectorWidth;
            uint32_t ndimTile;
            uint32_t numThreads;
            uint32_t blocking[3];
            ElementwiseOpPack opPack;
    };

    class HeuristicEW : public Heuristic<ElementwiseParameters, CandidateInfoEW>
    {
        public:
        int numberOfFeatures() const override { return 1; }

        void computeFeatures(const ElementwiseParameters &params,
                             const CandidateInfoEW &candidateInfo,
                             const DeviceProp* deviceProp,
                             float* features) const override;

        void evaluate(int numberOfCandidates,
                      const float* features,
                      float* scores) const override
        {
            for (int i = 0; i < numberOfCandidates; ++i)
                scores[i] = features[i];
        }
    };


}  // namespace LWTENSOR_NAMESPACE
