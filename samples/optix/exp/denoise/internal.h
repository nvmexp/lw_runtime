//
// Copyright 2020 LWPU Corporation. All rights reserved.
//

#pragma once

namespace optix_exp {

OptixResult optixDenoiserCreateWithUserModel_internal( OptixDeviceContext contextAPI, const OptixDenoiserOptions * options,
                                                       const void * userData, size_t userDataSizeInBytes, OptixDenoiser* denoiserAPI );

}
