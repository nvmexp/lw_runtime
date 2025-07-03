/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "device.hpp"

////////////////////////////////////////////////////////////////////////
// lwphy_i
namespace lwphy_i // lwphy internal
{
//----------------------------------------------------------------------
// device::device()
device::device(int device_index) :
    index_(device_index)
{
    if(device_index >= get_count())
    {
        throw lwda_exception(lwdaErrorIlwalidDevice);
    }
    lwdaError_t res = lwdaGetDeviceProperties(&properties_, device_index);
    if(lwdaSuccess != res)
    {
        throw lwda_exception(res);
    }
}

//----------------------------------------------------------------------
// device::get_count()
int device::get_count()
{
    int         count = 0;
    lwdaError_t res   = lwdaGetDeviceCount(&count);
    if(lwdaSuccess != res)
    {
        throw lwda_exception(res);
    }
    return count;
}

//----------------------------------------------------------------------
// device::get_lwrrent()
int device::get_lwrrent()
{
    int         device_idx = 0;
    lwdaError_t res        = lwdaGetDevice(&device_idx);
    if(lwdaSuccess != res)
    {
        throw lwda_exception(res);
    }
    return device_idx;
}

} // namespace lwphy_i
