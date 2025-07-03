/*
 *  Copyright 2008-2018 LWPU Corporation
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

/*! \file thrust/system/lwca/memory.h
 *  \brief Managing memory associated with Thrust's LWCA system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/lwca/memory_resource.h>
#include <thrust/memory.h>
#include <thrust/detail/type_traits.h>
#include <thrust/mr/allocator.h>
#include <ostream>

namespace thrust { namespace lwda_lwb
{

/*! Allocates an area of memory available to Thrust's <tt>lwca</tt> system.
 *  \param n Number of bytes to allocate.
 *  \return A <tt>lwca::pointer<void></tt> pointing to the beginning of the newly
 *          allocated memory. A null <tt>lwca::pointer<void></tt> is returned if
 *          an error oclwrs.
 *  \note The <tt>lwca::pointer<void></tt> returned by this function must be
 *        deallocated with \p lwca::free.
 *  \see lwca::free
 *  \see std::malloc
 */
inline __host__ __device__ pointer<void> malloc(std::size_t n);

/*! Allocates a typed area of memory available to Thrust's <tt>lwca</tt> system.
 *  \param n Number of elements to allocate.
 *  \return A <tt>lwca::pointer<T></tt> pointing to the beginning of the newly
 *          allocated elements. A null <tt>lwca::pointer<T></tt> is returned if
 *          an error oclwrs.
 *  \note The <tt>lwca::pointer<T></tt> returned by this function must be
 *        deallocated with \p lwca::free.
 *  \see lwca::free
 *  \see std::malloc
 */
template <typename T>
inline __host__ __device__ pointer<T> malloc(std::size_t n);

/*! Deallocates an area of memory previously allocated by <tt>lwca::malloc</tt>.
 *  \param ptr A <tt>lwca::pointer<void></tt> pointing to the beginning of an area
 *         of memory previously allocated with <tt>lwca::malloc</tt>.
 *  \see lwca::malloc
 *  \see std::free
 */
inline __host__ __device__ void free(pointer<void> ptr);

/*! \p lwca::allocator is the default allocator used by the \p lwca system's
 *  containers such as <tt>lwca::vector</tt> if no user-specified allocator is
 *  provided. \p lwca::allocator allocates (deallocates) storage with \p
 *  lwca::malloc (\p lwca::free).
 */
template<typename T>
using allocator = thrust::mr::stateless_resource_allocator<
  T, thrust::system::lwca::memory_resource
>;

/*! \p lwca::universal_allocator allocates memory that can be used by the \p lwca
 *  system and host systems.
 */
template<typename T>
using universal_allocator = thrust::mr::stateless_resource_allocator<
  T, thrust::system::lwca::universal_memory_resource
>;

} // namespace lwda_lwb

namespace system { namespace lwca
{
using thrust::lwda_lwb::malloc;
using thrust::lwda_lwb::free;
using thrust::lwda_lwb::allocator;
using thrust::lwda_lwb::universal_allocator;
}} // namespace system::lwca

/*! \namespace thrust::lwca
 *  \brief \p thrust::lwca is a top-level alias for \p thrust::system::lwca.
 */
namespace lwca
{
using thrust::lwda_lwb::malloc;
using thrust::lwda_lwb::free;
using thrust::lwda_lwb::allocator;
using thrust::lwda_lwb::universal_allocator;
} // namespace lwca

} // namespace thrust

#include <thrust/system/lwca/detail/memory.inl>

