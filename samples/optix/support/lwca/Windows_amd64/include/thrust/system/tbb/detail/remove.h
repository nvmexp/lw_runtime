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
#include <thrust/system/omp/detail/execution_policy.h>

namespace thrust
{
namespace system
{
namespace omp
{
namespace detail
{

template<typename ExecutionPolicy,
         typename ForwardIterator,
         typename Predicate>
  ForwardIterator remove_if(execution_policy<ExecutionPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            Predicate pred);


template<typename ExecutionPolicy,
         typename ForwardIterator,
         typename InputIterator,
         typename Predicate>
  ForwardIterator remove_if(execution_policy<ExecutionPolicy> &exec,
                            ForwardIterator first,
                            ForwardIterator last,
                            InputIterator stencil,
                            Predicate pred);


template<typename ExecutionPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(execution_policy<ExecutionPolicy> &exec,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                Predicate pred);


template<typename ExecutionPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputIterator,
         typename Predicate>
  OutputIterator remove_copy_if(execution_policy<ExecutionPolicy> &exec,
                                InputIterator1 first,
                                InputIterator1 last,
                                InputIterator2 stencil,
                                OutputIterator result,
                                Predicate pred);


} // end namespace detail
} // end namespace omp
} // end namespace system
} // end namespace thrust

#include <thrust/system/tbb/detail/remove.inl>

