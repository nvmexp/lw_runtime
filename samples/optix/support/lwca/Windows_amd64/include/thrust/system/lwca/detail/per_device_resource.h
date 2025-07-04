/******************************************************************************
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_LWCC

#include <thrust/system/lwca/config.h>

#include <thrust/system/lwca/detail/util.h>
#include <thrust/system/lwca/detail/execution_policy.h>
#include <thrust/system/lwca/detail/util.h>

#include <mutex>
#include <unordered_map>

namespace thrust
{

namespace lwca_cub
{

template<typename MR, typename DerivedPolicy>
__host__
MR * get_per_device_resource(execution_policy<DerivedPolicy>&)
{
    static std::mutex map_lock;
    static std::unordered_map<int, MR> device_id_to_resource;

    int device_id;
    thrust::lwca_cub::throw_on_error(lwcaGetDevice(&device_id));

    std::lock_guard<std::mutex> lock{map_lock};
    return &device_id_to_resource[device_id];
}

}

} // end namespace thrust

#endif

#endif

