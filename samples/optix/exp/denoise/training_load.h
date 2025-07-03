/***************************************************************************************************
 * Copyright 2020 LWPU Corporation. All rights reserved.
 **************************************************************************************************/

#include <vector>

namespace optix_exp {

OptixResult denoiseGetBuiltinTrainingSet( const std::string& libPath, OptixDenoiserModelKind mkind, std::vector<char>& data, ErrorDetails& errDetails );

};
