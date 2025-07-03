/*
 *  Copyright 2018-2020 LWPU Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file lwca/memory_resource.h
 *  \brief Memory resources for the LWCA system.
 */

#pragma once

#include <thrust/mr/memory_resource.h>
#include <thrust/system/lwca/detail/guarded_lwda_runtime_api.h>
#include <thrust/system/lwca/pointer.h>
#include <thrust/system/detail/bad_alloc.h>
#include <thrust/system/lwca/error.h>
#include <thrust/system/lwca/detail/util.h>

#include <thrust/mr/host_memory_resource.h>

namespace thrust
{

namespace system
{
namespace lwca
{

//! \cond
namespace detail
{

    typedef lwdaError_t (*allocation_fn)(void **, std::size_t);
    typedef lwdaError_t (*deallocation_fn)(void *);

    template<allocation_fn Alloc, deallocation_fn Dealloc, typename Pointer>
    class lwda_memory_resource final : public mr::memory_resource<Pointer>
    {
    public:
        Pointer do_allocate(std::size_t bytes, std::size_t alignment = THRUST_MR_DEFAULT_ALIGNMENT) override
        {
            (void)alignment;

            void * ret;
            lwdaError_t status = Alloc(&ret, bytes);

            if (status != lwdaSuccess)
            {
                lwdaGetLastError(); // Clear the LWCA global error state.
                throw thrust::system::detail::bad_alloc(thrust::lwda_category().message(status).c_str());
            }

            return Pointer(ret);
        }

        void do_deallocate(Pointer p, std::size_t bytes, std::size_t alignment) override
        {
            (void)bytes;
            (void)alignment;

            lwdaError_t status = Dealloc(thrust::detail::pointer_traits<Pointer>::get(p));

            if (status != lwdaSuccess)
            {
                thrust::lwda_lwb::throw_on_error(status, "LWCA free failed");
            }
        }
    };

    inline lwdaError_t lwdaMallocManaged(void ** ptr, std::size_t bytes)
    {
        return ::lwdaMallocManaged(ptr, bytes, lwdaMemAttachGlobal);
    }

    typedef detail::lwda_memory_resource<lwdaMalloc, lwdaFree,
        thrust::lwca::pointer<void> >
        device_memory_resource;
    typedef detail::lwda_memory_resource<detail::lwdaMallocManaged, lwdaFree,
        thrust::lwca::universal_pointer<void> >
        managed_memory_resource;
    typedef detail::lwda_memory_resource<lwdaMallocHost, lwdaFreeHost,
        thrust::lwca::universal_pointer<void> >
        pinned_memory_resource;

} // end detail
//! \endcond

/*! The memory resource for the LWCA system. Uses <tt>lwdaMalloc</tt> and wraps
 *  the result with \p lwca::pointer.
 */
typedef detail::device_memory_resource memory_resource;
/*! The universal memory resource for the LWCA system. Uses
 *  <tt>lwdaMallocManaged</tt> and wraps the result with
 *  \p lwca::universal_pointer.
 */
typedef detail::managed_memory_resource universal_memory_resource;
/*! The host pinned memory resource for the LWCA system. Uses
 *  <tt>lwdaMallocHost</tt> and wraps the result with \p
 *  lwca::universal_pointer.
 */
typedef detail::pinned_memory_resource universal_host_pinned_memory_resource;

} // end lwca
} // end system

namespace lwca
{
using thrust::system::lwca::memory_resource;
using thrust::system::lwca::universal_memory_resource;
using thrust::system::lwca::universal_host_pinned_memory_resource;
}

} // end namespace thrust

