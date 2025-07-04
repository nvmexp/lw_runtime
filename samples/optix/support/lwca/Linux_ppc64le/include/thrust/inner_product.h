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


/*! \file inner_product.h
 *  \brief Mathematical inner product between ranges
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/exelwtion_policy.h>

namespace thrust
{


/*! \addtogroup reductions
 *  \{
 *  \addtogroup transformed_reductions Transformed Reductions
 *  \ingroup reductions
 *  \{
 */


/*! \p inner_product callwlates an inner product of the ranges
 *  <tt>[first1, last1)</tt> and <tt>[first2, first2 + (last1 - first1))</tt>.
 *
 *  Specifically, this version of \p inner_product computes the sum
 *  <tt>init + (*first1 * *first2) + (*(first1+1) * *(first2+1)) + ... </tt>
 *
 *  The algorithm's exelwtion is parallelized as determined by \p exec.
 *
 *  \param exec The exelwtion policy to use for parallelization.
 *  \param first1 The beginning of the first sequence.
 *  \param last1 The end of the first sequence.
 *  \param first2 The beginning of the second sequence.
 *  \param init Initial value of the result.
 *  \return The inner product of sequences <tt>[first1, last1)</tt>
 *          and <tt>[first2, last2)</tt> plus \p init.
 *
 *  \tparam DerivedPolicy The name of the derived exelwtion policy.
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *  \tparam OutputType is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x is an object of type \p OutputType, and \c y is an object of \p InputIterator1's \c value_type,
 *          and \c z is an object of \p InputIterator2's \c value_type, then <tt>x + y * z</tt> is defined
 *          and is colwertible to \p OutputType.
 *
 *  The following code demonstrates how to use \p inner_product to
 *  compute the dot product of two vectors using the \p thrust::host exelwtion policy for parallelization.
 *
 *  \code
 *  #include <thrust/inner_product.h>
 *  #include <thrust/exelwtion_policy.h>
 *  ...
 *  float vec1[3] = {1.0f, 2.0f, 5.0f};
 *  float vec2[3] = {4.0f, 1.0f, 5.0f};
 *
 *  float result = thrust::inner_product(thrust::host, vec1, vec1 + 3, vec2, 0.0f);
 *
 *  // result == 31.0f
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/inner_product.html
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputType>
__host__ __device__
OutputType inner_product(const thrust::detail::exelwtion_policy_base<DerivedPolicy> &exec,
                         InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 first2,
                         OutputType init);


/*! \p inner_product callwlates an inner product of the ranges
 *  <tt>[first1, last1)</tt> and <tt>[first2, first2 + (last1 - first1))</tt>.
 *
 *  Specifically, this version of \p inner_product computes the sum
 *  <tt>init + (*first1 * *first2) + (*(first1+1) * *(first2+1)) + ... </tt>
 *
 *  Unlike the C++ Standard Template Library function <tt>std::inner_product</tt>,
 *  this version offers no guarantee on order of exelwtion.
 *
 *  \param first1 The beginning of the first sequence.
 *  \param last1 The end of the first sequence.
 *  \param first2 The beginning of the second sequence.
 *  \param init Initial value of the result.
 *  \return The inner product of sequences <tt>[first1, last1)</tt>
 *          and <tt>[first2, last2)</tt> plus \p init.
 *
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *  \tparam OutputType is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and if \c x is an object of type \p OutputType, and \c y is an object of \p InputIterator1's \c value_type,
 *          and \c z is an object of \p InputIterator2's \c value_type, then <tt>x + y * z</tt> is defined
 *          and is colwertible to \p OutputType.
 *
 *  The following code demonstrates how to use \p inner_product to
 *  compute the dot product of two vectors.
 *
 *  \code
 *  #include <thrust/inner_product.h>
 *  ...
 *  float vec1[3] = {1.0f, 2.0f, 5.0f};
 *  float vec2[3] = {4.0f, 1.0f, 5.0f};
 *
 *  float result = thrust::inner_product(vec1, vec1 + 3, vec2, 0.0f);
 *
 *  // result == 31.0f
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/inner_product.html
 */
template<typename InputIterator1, typename InputIterator2, typename OutputType>
OutputType inner_product(InputIterator1 first1, InputIterator1 last1,
                         InputIterator2 first2, OutputType init);


/*! \p inner_product callwlates an inner product of the ranges
 *  <tt>[first1, last1)</tt> and <tt>[first2, first2 + (last1 - first1))</tt>.
 *
 *  This version of \p inner_product is identical to the first, except that is uses
 *  two user-supplied function objects instead of \c operator+ and \c operator*.
 *
 *  Specifically, this version of \p inner_product computes the sum
 *  <tt>binary_op1( init, binary_op2(*first1, *first2) ), ... </tt>
 *
 *  The algorithm's exelwtion is parallelized as determined by \p exec.
 *
 *  \param exec The exelwtion policy to use for parallelization.
 *  \param first1 The beginning of the first sequence.
 *  \param last1 The end of the first sequence.
 *  \param first2 The beginning of the second sequence.
 *  \param init Initial value of the result.
 *  \param binary_op1 Generalized addition operation.
 *  \param binary_op2 Generalized multiplication operation.
 *  \return The inner product of sequences <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>.
 *
 *  \tparam DerivedPolicy The name of the derived exelwtion policy.
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          and \p InputIterator1's \c value_type is colwertible to \p BinaryFunction2's \c first_argument_type.
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>.
 *          and \p InputIterator2's \c value_type is colwertible to \p BinaryFunction2's \c second_argument_type.
 *  \tparam OutputType is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and \p OutputType is colwertible to \p BinaryFunction1's \c first_argument_type.
 *  \tparam BinaryFunction1 is a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>,
 *          and \p BinaryFunction1's \c return_type is colwertible to \p OutputType.
 *  \tparam BinaryFunction2 is a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>,
 *          and \p BinaryFunction2's \c return_type is colwertible to \p BinaryFunction1's \c second_argument_type.
 * 
 *  \code
 *  #include <thrust/inner_product.h>
 *  #include <thrust/exelwtion_policy.h>
 *  ...
 *  float vec1[3] = {1.0f, 2.0f, 5.0f};
 *  float vec2[3] = {4.0f, 1.0f, 5.0f};
 *
 *  float init = 0.0f;
 *  thrust::plus<float>       binary_op1;
 *  thrust::multiplies<float> binary_op2;
 *
 *  float result = thrust::inner_product(thrust::host, vec1, vec1 + 3, vec2, init, binary_op1, binary_op2);
 *
 *  // result == 31.0f
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/inner_product.html
 */
template<typename DerivedPolicy,
         typename InputIterator1,
         typename InputIterator2,
         typename OutputType,
         typename BinaryFunction1,
         typename BinaryFunction2>
__host__ __device__
OutputType inner_product(const thrust::detail::exelwtion_policy_base<DerivedPolicy> &exec,
                         InputIterator1 first1,
                         InputIterator1 last1,
                         InputIterator2 first2,
                         OutputType init, 
                         BinaryFunction1 binary_op1,
                         BinaryFunction2 binary_op2);


/*! \p inner_product callwlates an inner product of the ranges
 *  <tt>[first1, last1)</tt> and <tt>[first2, first2 + (last1 - first1))</tt>.
 *
 *  This version of \p inner_product is identical to the first, except that is uses
 *  two user-supplied function objects instead of \c operator+ and \c operator*.
 *
 *  Specifically, this version of \p inner_product computes the sum
 *  <tt>binary_op1( init, binary_op2(*first1, *first2) ), ... </tt>
 *
 *  Unlike the C++ Standard Template Library function <tt>std::inner_product</tt>,
 *  this version offers no guarantee on order of exelwtion.
 *
 *  \param first1 The beginning of the first sequence.
 *  \param last1 The end of the first sequence.
 *  \param first2 The beginning of the second sequence.
 *  \param init Initial value of the result.
 *  \param binary_op1 Generalized addition operation.
 *  \param binary_op2 Generalized multiplication operation.
 *  \return The inner product of sequences <tt>[first1, last1)</tt> and <tt>[first2, last2)</tt>.
 *
 *  \tparam InputIterator1 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>,
 *          and \p InputIterator1's \c value_type is colwertible to \p BinaryFunction2's \c first_argument_type.
 *  \tparam InputIterator2 is a model of <a href="http://www.sgi.com/tech/stl/InputIterator.html">Input Iterator</a>.
 *          and \p InputIterator2's \c value_type is colwertible to \p BinaryFunction2's \c second_argument_type.
 *  \tparam OutputType is a model of <a href="http://www.sgi.com/tech/stl/Assignable.html">Assignable</a>,
 *          and \p OutputType is colwertible to \p BinaryFunction1's \c first_argument_type.
 *  \tparam BinaryFunction1 is a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>,
 *          and \p BinaryFunction1's \c return_type is colwertible to \p OutputType.
 *  \tparam BinaryFunction2 is a model of <a href="http://www.sgi.com/tech/stl/BinaryFunction.html">Binary Function</a>,
 *          and \p BinaryFunction2's \c return_type is colwertible to \p BinaryFunction1's \c second_argument_type.
 * 
 *  \code
 *  #include <thrust/inner_product.h>
 *  ...
 *  float vec1[3] = {1.0f, 2.0f, 5.0f};
 *  float vec2[3] = {4.0f, 1.0f, 5.0f};
 *
 *  float init = 0.0f;
 *  thrust::plus<float>       binary_op1;
 *  thrust::multiplies<float> binary_op2;
 *
 *  float result = thrust::inner_product(vec1, vec1 + 3, vec2, init, binary_op1, binary_op2);
 *
 *  // result == 31.0f
 *  \endcode
 *
 *  \see http://www.sgi.com/tech/stl/inner_product.html
 */
template<typename InputIterator1, typename InputIterator2, typename OutputType,
         typename BinaryFunction1, typename BinaryFunction2>
OutputType inner_product(InputIterator1 first1, InputIterator1 last1,
                         InputIterator2 first2, OutputType init, 
                         BinaryFunction1 binary_op1, BinaryFunction2 binary_op2);


/*! \} // end transformed_reductions
 *  \} // end reductions
 */

} // end namespace thrust

#include <thrust/detail/inner_product.inl>

