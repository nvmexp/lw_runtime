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

#include <thrust/detail/temporary_array.h>
#include <thrust/distance.h>
#include <thrust/system/detail/generic/select_system.h>
#include <thrust/detail/type_traits.h>


namespace thrust
{

namespace detail
{
namespace temporary_array_detail
{


template<typename T> struct avoid_initialization : thrust::detail::has_trivial_copy_constructor<T> {};


template<typename T, typename TemporaryArray, typename Size>
__host__ __device__
typename thrust::detail::enable_if<
  avoid_initialization<T>::value
>::type
  construct_values(TemporaryArray &,
                   Size)
{
  // avoid the overhead of initialization
} // end construct_values()


template<typename T, typename TemporaryArray, typename Size>
__host__ __device__
typename thrust::detail::disable_if<
  avoid_initialization<T>::value
>::type
  construct_values(TemporaryArray &a,
                   Size n)
{
  a.default_construct_n(a.begin(), n);
} // end construct_values()


} // end temporary_array_detail


template<typename T, typename System>
__host__ __device__
  temporary_array<T,System>
    ::temporary_array(thrust::exelwtion_policy<System> &system)
      :super_t(alloc_type(temporary_allocator<T,System>(system)))
{
} // end temporary_array::temporary_array()


template<typename T, typename System>
__host__ __device__
  temporary_array<T,System>
    ::temporary_array(thrust::exelwtion_policy<System> &system, size_type n)
      :super_t(n, alloc_type(temporary_allocator<T,System>(system)))
{
  temporary_array_detail::construct_values<T>(*this, n);
} // end temporary_array::temporary_array()


template<typename T, typename System>
__host__ __device__
  temporary_array<T,System>
    ::temporary_array(int, thrust::exelwtion_policy<System> &system, size_type n)
      :super_t(n, alloc_type(temporary_allocator<T,System>(system)))
{
  // avoid initialization
  ;
} // end temporary_array::temporary_array()


template<typename T, typename System>
  template<typename InputIterator>
  __host__ __device__
    temporary_array<T,System>
      ::temporary_array(thrust::exelwtion_policy<System> &system,
                        InputIterator first,
                        size_type n)
        : super_t(alloc_type(temporary_allocator<T,System>(system)))
{
  super_t::allocate(n);

  super_t::uninitialized_copy_n(system, first, n, super_t::begin());
} // end temporary_array::temporary_array()


template<typename T, typename System>
  template<typename InputIterator, typename InputSystem>
  __host__ __device__
    temporary_array<T,System>
      ::temporary_array(thrust::exelwtion_policy<System> &system,
                        thrust::exelwtion_policy<InputSystem> &input_system,
                        InputIterator first,
                        size_type n)
        : super_t(alloc_type(temporary_allocator<T,System>(system)))
{
  super_t::allocate(n);

  super_t::uninitialized_copy_n(input_system, first, n, super_t::begin());
} // end temporary_array::temporary_array()


template<typename T, typename System>
  template<typename InputIterator>
  __host__ __device__
    temporary_array<T,System>
      ::temporary_array(thrust::exelwtion_policy<System> &system,
                        InputIterator first,
                        InputIterator last)
        : super_t(alloc_type(temporary_allocator<T,System>(system)))
{
  super_t::allocate(thrust::distance(first,last));

  super_t::uninitialized_copy(system, first, last, super_t::begin());
} // end temporary_array::temporary_array()


template<typename T, typename System>
  template<typename InputSystem, typename InputIterator>
  __host__ __device__
    temporary_array<T,System>
      ::temporary_array(thrust::exelwtion_policy<System> &system,
                        thrust::exelwtion_policy<InputSystem> &input_system,
                        InputIterator first,
                        InputIterator last)
        : super_t(alloc_type(temporary_allocator<T,System>(system)))
{
  super_t::allocate(thrust::distance(first,last));

  super_t::uninitialized_copy(input_system, first, last, super_t::begin());
} // end temporary_array::temporary_array()


template<typename T, typename System>
__host__ __device__
  temporary_array<T,System>
    ::~temporary_array()
{
  // note that super_t::destroy will ignore trivial destructors automatically
  super_t::destroy(super_t::begin(), super_t::end());
} // end temporary_array::~temporary_array()

} // end detail

} // end thrust

