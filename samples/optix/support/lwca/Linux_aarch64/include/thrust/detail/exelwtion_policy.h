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

#pragma once

#include <thrust/detail/config.h>

namespace thrust
{
namespace detail
{

struct exelwtion_policy_marker {};

// exelwtion_policy_base serves as a guard against
// inifinite relwrsion in thrust entry points:
//
// template<typename DerivedPolicy>
// void foo(const thrust::detail::exelwtion_policy_base<DerivedPolicy> &s)
// {
//   using thrust::system::detail::generic::foo;
//
//   foo(thrust::detail::derived_cast(thrust::detail::strip_const(s));
// }
//
// foo is not relwrsive when
// 1. DerivedPolicy is derived from thrust::exelwtion_policy below
// 2. generic::foo takes thrust::exelwtion_policy as a parameter
template<typename DerivedPolicy>
struct exelwtion_policy_base : exelwtion_policy_marker {};


template<typename DerivedPolicy>
constexpr __host__ __device__
exelwtion_policy_base<DerivedPolicy> &strip_const(const exelwtion_policy_base<DerivedPolicy> &x)
{
  return const_cast<exelwtion_policy_base<DerivedPolicy>&>(x);
}


template<typename DerivedPolicy>
constexpr __host__ __device__
DerivedPolicy &derived_cast(exelwtion_policy_base<DerivedPolicy> &x)
{
  return static_cast<DerivedPolicy&>(x);
}


template<typename DerivedPolicy>
constexpr __host__ __device__
const DerivedPolicy &derived_cast(const exelwtion_policy_base<DerivedPolicy> &x)
{
  return static_cast<const DerivedPolicy&>(x);
}

} // end detail

template<typename DerivedPolicy>
  struct exelwtion_policy
    : thrust::detail::exelwtion_policy_base<DerivedPolicy>
{};

} // end thrust

