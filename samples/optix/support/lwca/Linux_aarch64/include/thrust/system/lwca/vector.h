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

/*! \file thrust/system/lwca/vector.h
 *  \brief A dynamically-sizable array of elements which reside in memory available to
 *         Thrust's LWCA system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/system/lwca/memory.h>
#include <thrust/detail/vector_base.h>
#include <vector>

namespace thrust { namespace lwda_lwb
{

/*! \p lwca::vector is a container that supports random access to elements,
 *  constant time removal of elements at the end, and linear time insertion
 *  and removal of elements at the beginning or in the middle. The number of
 *  elements in a \p lwca::vector may vary dynamically; memory management is
 *  automatic. The elements contained in a \p lwca::vector reside in memory
 *  accessible by the \p lwca system.
 *
 *  \tparam T The element type of the \p lwca::vector.
 *  \tparam Allocator The allocator type of the \p lwca::vector.
 *          Defaults to \p lwca::allocator.
 *
 *  \see https://en.cppreference.com/w/cpp/container/vector
 *  \see host_vector For the documentation of the complete interface which is
 *                   shared by \p lwca::vector
 *  \see device_vector
 *  \see universal_vector
 */
template <typename T, typename Allocator = thrust::system::lwca::allocator<T>>
using vector = thrust::detail::vector_base<T, Allocator>;

/*! \p lwca::universal_vector is a container that supports random access to
 *  elements, constant time removal of elements at the end, and linear time
 *  insertion and removal of elements at the beginning or in the middle. The
 *  number of elements in a \p lwca::universal_vector may vary dynamically;
 *  memory management is automatic. The elements contained in a
 *  \p lwca::universal_vector reside in memory accessible by the \p lwca system
 *  and host systems.
 *
 *  \tparam T The element type of the \p lwca::universal_vector.
 *  \tparam Allocator The allocator type of the \p lwca::universal_vector.
 *          Defaults to \p lwca::universal_allocator.
 *
 *  \see https://en.cppreference.com/w/cpp/container/vector
 *  \see host_vector For the documentation of the complete interface which is
 *                   shared by \p lwca::universal_vector
 *  \see device_vector
 *  \see universal_vector
 */
template <typename T, typename Allocator = thrust::system::lwca::universal_allocator<T>>
using universal_vector = thrust::detail::vector_base<T, Allocator>;

} // namespace lwda_lwb

namespace system { namespace lwca
{
using thrust::lwda_lwb::vector;
using thrust::lwda_lwb::universal_vector;
}}

namespace lwca
{
using thrust::lwda_lwb::vector;
using thrust::lwda_lwb::universal_vector;
}

} // namespace thrust

