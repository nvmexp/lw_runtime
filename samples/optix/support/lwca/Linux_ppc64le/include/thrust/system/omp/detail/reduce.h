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


/*! \file reduce.h
 *  \brief OpenMP implementation of reduce algorithms.
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


template<typename DerivedPolicy,
         typename InputIterator, 
         typename OutputType,
         typename BinaryFunction>
  OutputType reduce(execution_policy<DerivedPolicy> &exec,
                    InputIterator first,
                    InputIterator last,
                    OutputType init,
                    BinaryFunction binary_op);


} // end namespace detail
} // end namespace omp
} // end namespace system
} // end namespace thrust

#include <thrust/system/omp/detail/reduce.inl>

