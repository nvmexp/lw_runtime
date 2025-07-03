/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(DEVICE_HPP_INCLUDED_)
#define DEVICE_HPP_INCLUDED_

#include "lwphy_internal.h"

////////////////////////////////////////////////////////////////////////
// lwphy_i
namespace lwphy_i // lwphy internal
{
//----------------------------------------------------------------------
// device
class device //
{
public:
    device(int device_index = 0);
    int        multiProcessorCount()    const { return properties_.multiProcessorCount; }
    size_t     sharedMemPerBlock()      const { return properties_.sharedMemPerBlock; }
    size_t     sharedMemPerBlockOptin() const { return properties_.sharedMemPerBlockOptin; }
    static int get_count();
    static int get_lwrrent();

private:
    int            index_;
    lwdaDeviceProp properties_;
};

} // namespace lwphy_i

#endif // !defined(DEVICE_HPP_INCLUDED_)
