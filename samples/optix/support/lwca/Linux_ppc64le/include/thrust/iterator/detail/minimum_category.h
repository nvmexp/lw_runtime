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

#include <thrust/detail/type_traits/minimum_type.h>

namespace thrust
{

namespace detail
{ 

template<typename T1,
         typename T2  = minimum_type_detail::any_colwersion,
         typename T3  = minimum_type_detail::any_colwersion,
         typename T4  = minimum_type_detail::any_colwersion,
         typename T5  = minimum_type_detail::any_colwersion,
         typename T6  = minimum_type_detail::any_colwersion,
         typename T7  = minimum_type_detail::any_colwersion,
         typename T8  = minimum_type_detail::any_colwersion,
         typename T9  = minimum_type_detail::any_colwersion,
         typename T10 = minimum_type_detail::any_colwersion,
         typename T11 = minimum_type_detail::any_colwersion,
         typename T12 = minimum_type_detail::any_colwersion,
         typename T13 = minimum_type_detail::any_colwersion,
         typename T14 = minimum_type_detail::any_colwersion,
         typename T15 = minimum_type_detail::any_colwersion,
         typename T16 = minimum_type_detail::any_colwersion>
  struct minimum_category
    : minimum_type<T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16>
{
}; // end minimum_category

} // end detail

} // end thrust


