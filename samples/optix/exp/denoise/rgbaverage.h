//
// Copyright 2020 LWPU Corporation. All rights reserved.
//

#ifndef IRAY_BUILD
#include <exp/context/ErrorHandling.h>
#endif

namespace optix_exp {
OptixResult denoiseRGBAverage( const OptixImage2D* input, float* scale, void* scratch, size_t scratchSize, lwdaStream_t streamId, ErrorDetails& errDetails );
OptixResult denoiseRGBAverageComputeMemoryResources( size_t& sizeInBytes, ErrorDetails& errDetails );
};
