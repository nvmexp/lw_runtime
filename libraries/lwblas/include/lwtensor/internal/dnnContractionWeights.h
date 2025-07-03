#pragma once

#include <stdint.h>
#include <array>

#include "lwtensor/internal/defines.h"
#include "lwtensor/internal/types.h"

namespace LWTENSOR_NAMESPACE
{

template <lwdaDataType_t DataTypeA, lwdaDataType_t DataTypeB, lwdaDataType_t DataTypeC, lwdaDataType_t DataTypeComp>
struct DnnShape
{
};

template <typename Shape_>
class ContractionWeights
{
public:
    using Shape = Shape_;

    ContractionWeights(const std::array<float, Shape::kSize0 * Shape::kSize1>& weights1,
                       const std::array<float, Shape::kSize1 * Shape::kSize2>& weights2,
                       const std::array<float, Shape::kSize2 * Shape::kSize3>& weights3,
                       const std::array<float, Shape::kSize3 * Shape::kSize4>& weights4,
                       const std::array<float, Shape::kSize1>& bias1,
                       const std::array<float, Shape::kSize2>& bias2,
                       const std::array<float, Shape::kSize3>& bias3,
                       const std::array<float, Shape::kSize4>& bias4)
        : weights1_(weights1), weights2_(weights2), weights3_(weights3), weights4_(weights4),
          bias1_(bias1), bias2_(bias2), bias3_(bias3), bias4_(bias4)
    {
    }

    std::array<float, Shape::kSize0 * Shape::kSize1> weights1_;
    std::array<float, Shape::kSize1 * Shape::kSize2> weights2_;
    std::array<float, Shape::kSize2 * Shape::kSize3> weights3_;
    std::array<float, Shape::kSize3 * Shape::kSize4> weights4_;

    std::array<float, Shape::kSize1> bias1_;
    std::array<float, Shape::kSize2> bias2_;
    std::array<float, Shape::kSize3> bias3_;
    std::array<float, Shape::kSize4> bias4_;
};

} // namespace LWTENSOR_NAMESPACE
