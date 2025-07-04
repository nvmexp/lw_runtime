/*
 *  Copyright 2008-2013 LWPU Corporation
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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/system/detail/generic/per_device_resource.h>
#include <thrust/system/detail/adl/per_device_resource.h>
#include <thrust/mr/allocator.h>

#include <thrust/detail/exelwtion_policy.h>
#include <thrust/mr/allocator.h>

namespace thrust
{

/*! Returns a global instance of \p MR for the current device of the provided system.
 *
 *  \tparam MR type of a memory resource to get an instance from. Must be \p DefaultConstructible.
 *  \param system exelwtion policy for which the resource is requested.
 *  \returns a pointer to a global instance of \p MR for the current device.
 */
template<typename MR, typename DerivedPolicy>
__host__
MR * get_per_device_resource(const thrust::detail::exelwtion_policy_base<DerivedPolicy> & system)
{
    using thrust::system::detail::generic::get_per_device_resource;

    return get_per_device_resource<MR>(
        thrust::detail::derived_cast(
            thrust::detail::strip_const(system)));
}

/*! A helper allocator class that uses global per device instances of a given upstream memory resource. Requires the memory
 *      resource to be default constructible.
 *
 *  \tparam T the type that will be allocated by this allocator.
 *  \tparam MR the upstream memory resource to use for memory allocation. Must derive from
 *      \p thrust::mr::memory_resource and must be \p final.
 *  \tparam ExelwtionPolicy the exelwtion policy of the system to be used to retrieve the resource for the current device.
 */
template<typename T, typename Upstream, typename ExelwtionPolicy>
class per_device_allocator : public thrust::mr::allocator<T, Upstream>
{
    typedef thrust::mr::allocator<T, Upstream> base;

public:
    /*! The \p rebind metafunction provides the type of an \p per_device_allocator instantiated with another type.
     *
     *  \tparam U the other type to use for instantiation.
     */
    template<typename U>
    struct rebind
    {
        /*! The typedef \p other gives the type of the rebound \p per_device_allocator.
         */
        typedef per_device_allocator<U, Upstream, ExelwtionPolicy> other;
    };

    /*! Default constructor. Uses \p get_global_resource to get the global instance of \p Upstream and initializes the
     *      \p allocator base subobject with that resource.
     */
    __host__
    per_device_allocator() : base(get_per_device_resource<Upstream>(ExelwtionPolicy()))
    {
    }

    /*! Copy constructor. Copies the memory resource pointer. */
    __host__ __device__
    per_device_allocator(const per_device_allocator & other)
        : base(other) {}

    /*! Colwersion constructor from an allocator of a different type. Copies the memory resource pointer. */
    template<typename U>
    __host__ __device__
    per_device_allocator(const per_device_allocator<U, Upstream, ExelwtionPolicy> & other)
        : base(other) {}

    /*! Destructor. */
    __host__ __device__
    ~per_device_allocator() {}
};


} // end namespace thrust

#endif // THRUST_CPP_DIALECT >= 2011
