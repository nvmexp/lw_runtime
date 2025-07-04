/*
 *  Copyright 2008-2013 LWPU Corporation
 *  Copyright 2013 Filipe RNC Maia
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

#include <thrust/complex.h>
#include <thrust/detail/type_traits.h>

namespace thrust {

template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
pow(const complex<T0>& x, const complex<T1>& y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  return exp(log(complex<T>(x)) * complex<T>(y));
}

template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
pow(const complex<T0>& x, const T1& y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  return exp(log(complex<T>(x)) * T(y));
}

template <typename T0, typename T1>
__host__ __device__
complex<typename detail::promoted_numerical_type<T0, T1>::type>
pow(const T0& x, const complex<T1>& y)
{
  typedef typename detail::promoted_numerical_type<T0, T1>::type T;
  // Find `log` by ADL.
  using std::log;
  return exp(log(T(x)) * complex<T>(y));
}

} // end namespace thrust

