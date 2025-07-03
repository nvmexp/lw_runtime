#pragma once

#include "lwtensor/internal/types.h"
#include "lwtensor/internal/typesEx.h"
#include "lwtensor/internal/context.h"
#include "lwtensor/internal/featuresUtils.h"
#include "lwtensor/internal/heuristicDnn.h"

#include <memory>

namespace LWTENSOR_NAMESPACE
{

class HeuristicSimple : public Heuristic<ContractionDescriptorInternal, CandidateInfoLwtlass>
{
public:

    std::string getName() const override { return "HeuristicSimple"; }

    virtual int numberOfFeatures() const { return 1; }

    virtual void computeFeatures(const ContractionDescriptorInternal &params,
                                 const CandidateInfoLwtlass &candidateInfo,
                                 const DeviceProp* deviceProp,
                                 float* features) const;

    virtual void evaluate(int numberOfCandidates,
                          const float* features,
                          float* scores) const;
};

#if defined(__x86_64__) || defined(_MSC_VER)

template <typename Weights_>
class HeuristicToggle : public Heuristic<ContractionDescriptorInternal, CandidateInfoLwtlass>
{
public:
    using H = Heuristic<ContractionDescriptorInternal, CandidateInfoLwtlass>;
    using Weights = Weights_;

    HeuristicToggle() : selectedHeuristic_(nullptr)
    {
        if (toggle<false>("LWTENSOR_HEUR", "dnn"))
        {
            selectedHeuristic_ = std::unique_ptr<H>(new HeuristicDnn<Weights>);
        }
        else
        {
            selectedHeuristic_ = std::unique_ptr<H>(new HeuristicSimple);
        }
    }

    std::string getName() const override { return selectedHeuristic_->getName(); }

    virtual int numberOfFeatures() const { return selectedHeuristic_->numberOfFeatures(); }

    virtual void computeFeatures(const ContractionDescriptorInternal &params,
                                 const CandidateInfoLwtlass &candidateInfo,
                                 const DeviceProp* deviceProp,
                                 float* features) const
    {
        selectedHeuristic_->computeFeatures(params, candidateInfo, deviceProp, features);
    }

    virtual void evaluate(int numberOfCandidates,
                          const float* features,
                          float* scores) const
    {
        selectedHeuristic_->evaluate(numberOfCandidates, features, scores);
    }

private:
    std::unique_ptr<H> selectedHeuristic_;
};

#endif

}  // namespace LWTENSOR_NAMESPACE
