/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#if !defined(LWPHY_CONTEXT_HPP_INCLUDED_)
#define LWPHY_CONTEXT_HPP_INCLUDED_

#include <lwda_runtime.h>
#include <stdint.h>

////////////////////////////////////////////////////////////////////////
// lwphyContext
// Empty base class for internal context class, used by forward
// declaration in public-facing lwphy.h.
struct lwphyContext
{
};

namespace lwphy_i // lwphy internal
{

constexpr uint64_t make_cc_uint64(int major, int minor)
{
    return (static_cast<uint64_t>(major) << 32) + static_cast<uint32_t>(minor);
}
    
////////////////////////////////////////////////////////////////////////
// lwphy_i::context
// lwPHY "context" object, used to (perhaps among other things) cache
// device properties.
class context : public lwphyContext
{
public:
    //------------------------------------------------------------------
    // Constructor
    context();
    //------------------------------------------------------------------
    // device index
    int index() const { return deviceIndex_; }
    //------------------------------------------------------------------
    // compute capability
    uint64_t compute_cap() const { return cc_; }
    //------------------------------------------------------------------
    // maximum shared mem per block (optin)
    int max_shmem_per_block_optin() const { return sharedMemPerBlockOptin_; }
    //------------------------------------------------------------------
    // SM count
    int sm_count() const { return multiProcessorCount_; }
private:
    //------------------------------------------------------------------
    // Data
    int      deviceIndex_;            // index of device associated with context
    uint64_t cc_;                     // compute capability (major << 32) | minor
    int      sharedMemPerBlockOptin_; // maximum shared memory per block usable by option
    int      multiProcessorCount_;    // number of multiprocessors on device
};

}

#endif // !defined(LWPHY_CONTEXT_HPP_INCLUDED_)
