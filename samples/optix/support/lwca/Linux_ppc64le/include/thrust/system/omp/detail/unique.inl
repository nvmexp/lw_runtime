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
#include <thrust/system/omp/detail/unique.h>
#include <thrust/system/detail/generic/unique.h>
#include <thrust/pair.h>

namespace thrust
{
namespace system
{
namespace omp
{
namespace detail
{


template<typename DerivedPolicy,
         typename ForwardIterator,
         typename BinaryPredicate>
  ForwardIterator unique(execution_policy<DerivedPolicy> &exec,
                         ForwardIterator first,
                         ForwardIterator last,
                         BinaryPredicate binary_pred)
{
  // omp prefers generic::unique to cpp::unique
  return thrust::system::detail::generic::unique(exec,first,last,binary_pred);
} // end unique()


template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate>
  OutputIterator unique_copy(execution_policy<DerivedPolicy> &exec,
                             InputIterator first,
                             InputIterator last,
                             OutputIterator output,
                             BinaryPredicate binary_pred)
{
  // omp prefers generic::unique_copy to cpp::unique_copy
  return thrust::system::detail::generic::unique_copy(exec,first,last,output,binary_pred);
} // end unique_copy()


} // end namespace detail
} // end namespace omp 
} // end namespace system
} // end namespace thrust

