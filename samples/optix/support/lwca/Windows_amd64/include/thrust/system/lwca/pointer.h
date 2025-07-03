/*
 *  Copyright 2008-2020 LWPU Corporation
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

/*! \file thrust/system/lwca/memory.h
 *  \brief Managing memory associated with Thrust's Standard C++ system.
 */

#pragma once

#include <thrust/detail/config.h>
#include <type_traits>
#include <thrust/system/lwca/detail/exelwtion_policy.h>
#include <thrust/detail/pointer.h>
#include <thrust/detail/reference.h>

namespace thrust { namespace lwda_lwb
{

/*! \p lwca::pointer stores a pointer to an object allocated in memory
 *  accessible by the \p lwca system. This type provides type safety when
 *  dispatching algorithms on ranges resident in \p lwca memory.
 *
 *  \p lwca::pointer has pointer semantics: it may be dereferenced and
 *  manipulated with pointer arithmetic.
 *
 *  \p lwca::pointer can be created with the function \p lwca::malloc, or by
 *  explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p lwca::pointer may be obtained by eiter
 *  its <tt>get</tt> member function or the \p raw_pointer_cast function.
 *
 *  \note \p lwca::pointer is not a "smart" pointer; it is the programmer's
 *        responsibility to deallocate memory pointed to by \p lwca::pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see lwca::malloc
 *  \see lwca::free
 *  \see raw_pointer_cast
 */
template <typename T>
using pointer = thrust::pointer<
  T,
  thrust::lwda_lwb::tag,
  thrust::tagged_reference<T, thrust::lwda_lwb::tag>
>;

/*! \p lwca::universal_pointer stores a pointer to an object allocated in
 *  memory accessible by the \p lwca system and host systems.
 *
 *  \p lwca::universal_pointer has pointer semantics: it may be dereferenced
 *  and manipulated with pointer arithmetic.
 *
 *  \p lwca::universal_pointer can be created with \p lwca::universal_allocator
 *  or by explicitly calling its constructor with a raw pointer.
 *
 *  The raw pointer encapsulated by a \p lwca::universal_pointer may be
 *  obtained by eiter its <tt>get</tt> member function or the \p
 *  raw_pointer_cast function.
 *
 *  \note \p lwca::universal_pointer is not a "smart" pointer; it is the
 *        programmer's responsibility to deallocate memory pointed to by
 *        \p lwca::universal_pointer.
 *
 *  \tparam T specifies the type of the pointee.
 *
 *  \see lwca::universal_allocator
 *  \see raw_pointer_cast
 */
template <typename T>
using universal_pointer = thrust::pointer<
  T,
  thrust::lwda_lwb::tag,
  typename std::add_lvalue_reference<T>::type
>;

/*! \p lwca::reference is a wrapped reference to an object stored in memory
 *  accessible by the \p lwca system. \p lwca::reference is the type of the
 *  result of dereferencing a \p lwca::pointer.
 *
 *  \tparam T Specifies the type of the referenced object.
 *
 *  \see lwca::pointer
 */
template <typename T>
using reference = thrust::tagged_reference<T, thrust::lwda_lwb::tag>;

} // namespace lwda_lwb

/*! \addtogroup system_backends Systems
 *  \ingroup system
 *  \{
 */

/*! \namespace thrust::system::lwca
 *  \brief \p thrust::system::lwca is the namespace containing functionality
 *  for allocating, manipulating, and deallocating memory available to Thrust's
 *  LWCA backend system. The identifiers are provided in a separate namespace
 *  underneath <tt>thrust::system</tt> for import colwenience but are also
 *  aliased in the top-level <tt>thrust::lwca</tt> namespace for easy access.
 *
 */
namespace system { namespace lwca
{
using thrust::lwda_lwb::pointer;
using thrust::lwda_lwb::universal_pointer;
using thrust::lwda_lwb::reference;
}} // namespace system::lwca
/*! \}
 */

/*! \namespace thrust::lwca
 *  \brief \p thrust::lwca is a top-level alias for \p thrust::system::lwca.
 */
namespace lwca
{
using thrust::lwda_lwb::pointer;
using thrust::lwda_lwb::universal_pointer;
using thrust::lwda_lwb::reference;
} // namespace lwca

} // namespace thrust

