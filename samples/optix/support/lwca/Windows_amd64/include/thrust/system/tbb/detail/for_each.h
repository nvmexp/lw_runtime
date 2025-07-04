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
#include <thrust/system/tbb/detail/execution_policy.h>

namespace thrust
{
namespace system
{
namespace tbb
{
namespace detail
{

template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename UnaryFunction>
  RandomAccessIterator for_each(execution_policy<DerivedPolicy> &exec,
                                RandomAccessIterator first,
                                RandomAccessIterator last,
                                UnaryFunction f);

template<typename DerivedPolicy,
         typename RandomAccessIterator,
         typename Size,
         typename UnaryFunction>
  RandomAccessIterator for_each_n(execution_policy<DerivedPolicy> &exec,
                                  RandomAccessIterator first,
                                  Size n,
                                  UnaryFunction f);

} // end namespace detail
} // end namespace tbb
} // end namespace system
} // end namespace thrust

#include <thrust/system/tbb/detail/for_each.inl>

