/*
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
#include <thrust/system/detail/sequential/execution_policy.h>
#include <cstdlib> // for malloc & free
#include <thrust/detail/raw_pointer_cast.h>

namespace thrust
{
namespace system
{
namespace detail
{
namespace sequential
{


template<typename DerivedPolicy>
inline __host__ __device__
void *malloc(execution_policy<DerivedPolicy> &, std::size_t n)
{
  return std::malloc(n);
} // end mallc()


template<typename DerivedPolicy, typename Pointer>
inline __host__ __device__
void free(sequential::execution_policy<DerivedPolicy> &, Pointer ptr)
{
  std::free(thrust::raw_pointer_cast(ptr));
} // end mallc()


} // end sequential
} // end detail
} // end system
} // end thrust

