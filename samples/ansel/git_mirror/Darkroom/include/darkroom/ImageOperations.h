#pragma once

#include <vector>
#include "darkroom/Errors.h"

namespace darkroom
{
    enum class TonemapOperator
    {
        kReinhardSimple,
        kFilmic,
        kFilmicLinear,
        kClamp
    };

    // crop 3-component (RGB/BGR) input and put result into output
    template <typename T>
    Error crop(std::vector<T>& output, const std::vector<T>& input, unsigned int width, unsigned int height,
        unsigned int x, unsigned int y, unsigned int cropWidth, unsigned int cropHeight);

    // append two images vertically (top-bottom)
    template <typename T>
    Error appendVertically(std::vector<T>& output, const std::vector<T>& input1, const std::vector<T>& input2);
    // append two images horizontally (left-right)
    template <typename T>
    Error appendHorizontally(std::vector<T>& output, const std::vector<T>& input1, const std::vector<T>& input2, unsigned int width, unsigned int height);

    // downscale input to output
    template <typename T, typename Q>
    Error downscale(T* output, const Q* input,
        unsigned int inputChannels, unsigned int inputWidth, unsigned int inputHeight,
        unsigned int outputChannels, unsigned int outputWidth, unsigned int outputHeight,
        unsigned int xOffset = 0u, unsigned int yOffset = 0u, unsigned int windowWidth = 0u,
        unsigned int windowHeight = 0u);

    template <typename T, typename S, typename Q>
    Error downscaleAclwmulate(const T* input, S* outputAclwm, Q* outputSum, 
        unsigned int inputChannels, unsigned int inputWidth, unsigned int inputHeight,
        unsigned int outputChannels, unsigned int outputWidth, unsigned int outputHeight,
        unsigned int xOffset = 0u, unsigned int yOffset = 0u, unsigned int windowWidth = 0u,
        unsigned int windowHeight = 0u);
    template <typename T, typename S>
    Error downscaleAverage(T* output, S* outputAclwm, float* outputSum, unsigned int outputChannels, unsigned int outputWidth, unsigned int outputHeight);

    // HDR->SDR tonemapping
    template<TonemapOperator op>
    Error tonemap(float* input, unsigned char* output, unsigned int width, unsigned int height, unsigned int channels);
}
