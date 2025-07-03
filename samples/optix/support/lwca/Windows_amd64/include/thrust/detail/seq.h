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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/allocator_aware_exelwtion_policy.h>
#include <thrust/system/detail/sequential/exelwtion_policy.h>

namespace thrust
{
namespace detail
{


struct seq_t : thrust::system::detail::sequential::exelwtion_policy<seq_t>,
  thrust::detail::allocator_aware_exelwtion_policy<
    thrust::system::detail::sequential::exelwtion_policy>
{
  __host__ __device__
  constexpr seq_t() : thrust::system::detail::sequential::exelwtion_policy<seq_t>() {}

  // allow any exelwtion_policy to colwert to seq_t
  template<typename DerivedPolicy>
  __host__ __device__
  seq_t(const thrust::exelwtion_policy<DerivedPolicy> &)
    : thrust::system::detail::sequential::exelwtion_policy<seq_t>()
  {}
};


} // end detail


THRUST_INLINE_CONSTANT detail::seq_t seq;


} // end thrust


