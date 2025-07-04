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


/*! \file scan.h
 *  \brief TBB implementations of scan functions.
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

template<typename InputIterator,
         typename OutputIterator,
         typename BinaryFunction>
  OutputIterator inclusive_scan(tag,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                BinaryFunction binary_op);


template<typename InputIterator,
         typename OutputIterator,
         typename T,
         typename BinaryFunction>
  OutputIterator exclusive_scan(tag,
                                InputIterator first,
                                InputIterator last,
                                OutputIterator result,
                                T init,
                                BinaryFunction binary_op);


} // end namespace detail
} // end namespace tbb
} // end namespace system
} // end namespace thrust

#include <thrust/system/tbb/detail/scan.inl>

