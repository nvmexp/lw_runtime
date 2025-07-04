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


/*! \file fill.h
 *  \brief Fills a range with a constant value
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/exelwtion_policy.h>

namespace thrust
{


/*! \addtogroup transformations
 *  \addtogroup filling
 *  \ingroup transformations
 *  \{
 */


/*! \p fill assigns the value \p value to every element in
 *  the range <tt>[first, last)</tt>. That is, for every
 *  iterator \c i in <tt>[first, last)</tt>, it performs
 *  the assignment <tt>*i = value</tt>.
 *
 *  The algorithm's exelwtion is parallelized as determined by \p exec.
 *
 *  \param exec The exelwtion policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param value The value to be copied.
 *
 *  \tparam DerivedPolicy The name of the derived exelwtion policy.
 *  \tparam ForwardIterator is a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable.
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and \p T's \c value_type is colwertible to \p ForwardIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p fill to set a thrust::device_vector's
 *  elements to a given value using the \p thrust::device exelwtion policy for parallelization:
 *
 *  \code
 *  #include <thrust/fill.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/exelwtion_policy.h>
 *  ...
 *  thrust::device_vector<int> v(4);
 *  thrust::fill(thrust::device, v.begin(), v.end(), 137);
 *
 *  // v[0] == 137, v[1] == 137, v[2] == 137, v[3] == 137
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/fill.html
 *  \see \c fill_n
 *  \see \c uninitialized_fill
 */
template<typename DerivedPolicy, typename ForwardIterator, typename T>
__host__ __device__
  void fill(const thrust::detail::exelwtion_policy_base<DerivedPolicy> &exec,
            ForwardIterator first,
            ForwardIterator last,
            const T &value);


/*! \p fill assigns the value \p value to every element in
 *  the range <tt>[first, last)</tt>. That is, for every
 *  iterator \c i in <tt>[first, last)</tt>, it performs
 *  the assignment <tt>*i = value</tt>.
 *
 *  \param first The beginning of the sequence.
 *  \param last The end of the sequence.
 *  \param value The value to be copied.
 *
 *  \tparam ForwardIterator is a model of <a href="http://www.sgi.com/tech/stl/ForwardIterator.html">Forward Iterator</a>,
 *          and \p ForwardIterator is mutable.
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and \p T's \c value_type is colwertible to \p ForwardIterator's \c value_type.
 *
 *  The following code snippet demonstrates how to use \p fill to set a thrust::device_vector's
 *  elements to a given value.
 *
 *  \code
 *  #include <thrust/fill.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> v(4);
 *  thrust::fill(v.begin(), v.end(), 137);
 *
 *  // v[0] == 137, v[1] == 137, v[2] == 137, v[3] == 137
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/fill.html
 *  \see \c fill_n
 *  \see \c uninitialized_fill
 */
template<typename ForwardIterator, typename T>
__host__ __device__
  void fill(ForwardIterator first,
            ForwardIterator last,
            const T &value);


/*! \p fill_n assigns the value \p value to every element in
 *  the range <tt>[first, first+n)</tt>. That is, for every
 *  iterator \c i in <tt>[first, first+n)</tt>, it performs
 *  the assignment <tt>*i = value</tt>.
 *
 *  The algorithm's exelwtion is parallelized as determined by \p exec.
 *
 *  \param exec The exelwtion policy to use for parallelization.
 *  \param first The beginning of the sequence.
 *  \param n The size of the sequence.
 *  \param value The value to be copied.
 *  \return <tt>first + n</tt>
 *
 *  \tparam DerivedPolicy The name of the derived exelwtion policy.
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>.
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and \p T's \c value_type is colwertible to a type in \p OutputIterator's set of \c value_type.
 *
 *  The following code snippet demonstrates how to use \p fill to set a thrust::device_vector's
 *  elements to a given value using the \p thrust::device exelwtion policy for parallelization:
 *
 *  \code
 *  #include <thrust/fill.h>
 *  #include <thrust/device_vector.h>
 *  #include <thrust/exelwtion_policy.h>
 *  ...
 *  thrust::device_vector<int> v(4);
 *  thrust::fill_n(thrust::device, v.begin(), v.size(), 137);
 *
 *  // v[0] == 137, v[1] == 137, v[2] == 137, v[3] == 137
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/fill_n.html
 *  \see \c fill
 *  \see \c uninitialized_fill_n
 */
template<typename DerivedPolicy, typename OutputIterator, typename Size, typename T>
__host__ __device__
  OutputIterator fill_n(const thrust::detail::exelwtion_policy_base<DerivedPolicy> &exec,
                        OutputIterator first,
                        Size n,
                        const T &value);


/*! \p fill_n assigns the value \p value to every element in
 *  the range <tt>[first, first+n)</tt>. That is, for every
 *  iterator \c i in <tt>[first, first+n)</tt>, it performs
 *  the assignment <tt>*i = value</tt>.
 *
 *  \param first The beginning of the sequence.
 *  \param n The size of the sequence.
 *  \param value The value to be copied.
 *  \return <tt>first + n</tt>
 *
 *  \tparam OutputIterator is a model of <a href="http://www.sgi.com/tech/stl/OutputIterator.html">Output Iterator</a>.
 *  \tparam T is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and \p T's \c value_type is colwertible to a type in \p OutputIterator's set of \c value_type.
 *
 *  The following code snippet demonstrates how to use \p fill to set a thrust::device_vector's
 *  elements to a given value.
 *
 *  \code
 *  #include <thrust/fill.h>
 *  #include <thrust/device_vector.h>
 *  ...
 *  thrust::device_vector<int> v(4);
 *  thrust::fill_n(v.begin(), v.size(), 137);
 *
 *  // v[0] == 137, v[1] == 137, v[2] == 137, v[3] == 137
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/fill_n.html
 *  \see \c fill
 *  \see \c uninitialized_fill_n
 */
template<typename OutputIterator, typename Size, typename T>
__host__ __device__
  OutputIterator fill_n(OutputIterator first,
                        Size n,
                        const T &value);


/*! \} // end filling
 *  \} // transformations
 */

} // end namespace thrust

#include <thrust/detail/fill.inl>

