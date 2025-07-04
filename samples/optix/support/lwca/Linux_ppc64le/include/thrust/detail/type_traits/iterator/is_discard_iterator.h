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
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/discard_iterator.h>

namespace thrust
{
namespace detail
{

template <typename Iterator>
struct is_discard_iterator
  : public thrust::detail::false_type
{};

template <typename System>
struct is_discard_iterator< thrust::discard_iterator<System> >
 : public thrust::detail::true_type
{};

} // end namespace detail
} // end namespace thrust

