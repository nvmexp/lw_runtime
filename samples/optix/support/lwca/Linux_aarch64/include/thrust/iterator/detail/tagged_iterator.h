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
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/detail/use_default.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

namespace thrust
{
namespace detail
{

template <typename,typename> class tagged_iterator;

template<typename Iterator, typename Tag>
  struct tagged_iterator_base
{
  typedef thrust::iterator_adaptor<
    tagged_iterator<Iterator,Tag>,
    Iterator,
    typename thrust::iterator_value<Iterator>::type,
    Tag,
    typename thrust::iterator_traversal<Iterator>::type,
    typename thrust::iterator_reference<Iterator>::type,
    typename thrust::iterator_difference<Iterator>::type
  > type;
}; // end tagged_iterator_base

template<typename Iterator, typename Tag>
  class tagged_iterator
    : public tagged_iterator_base<Iterator,Tag>::type
{
  private:
    typedef typename tagged_iterator_base<Iterator,Tag>::type super_t;

  public:
    __host__ __device__
    tagged_iterator() {}

    __host__ __device__
    explicit tagged_iterator(Iterator x)
      : super_t(x) {}
}; // end tagged_iterator

} // end detail

// tagged_iterator is trivial if its base iterator is.
template <typename BaseIterator, typename Tag>
struct proclaim_contiguous_iterator<
  detail::tagged_iterator<BaseIterator, Tag>
> : is_contiguous_iterator<BaseIterator> {};

} // end thrust

