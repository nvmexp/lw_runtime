/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "lwphy_context.hpp"
#include "lwphy_internal.h"

namespace lwphy_i
{
    
////////////////////////////////////////////////////////////////////////
// lwphy_i::context::context()
context::context()
{
    //------------------------------------------------------------------
    // Retrieve the device that will be associated with this context
    lwdaError_t e = lwdaGetDevice(&deviceIndex_);
    if(lwdaSuccess != e)
    {
        throw lwda_exception(e);
    }
    //------------------------------------------------------------------
    // Retrieve device properties
    lwdaDeviceProp devProp;
    e = lwdaGetDeviceProperties(&devProp, deviceIndex_);
    if(lwdaSuccess != e)
    {
        throw lwda_exception(e);
    }
    cc_                     = make_cc_uint64(devProp.major, devProp.minor);
    sharedMemPerBlockOptin_ = devProp.sharedMemPerBlockOptin;
    multiProcessorCount_    = devProp.multiProcessorCount;
    
};

} // namespace lwphy_i
