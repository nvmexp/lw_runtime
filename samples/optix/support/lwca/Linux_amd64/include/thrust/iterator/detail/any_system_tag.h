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
#include <thrust/detail/exelwtion_policy.h>

namespace thrust
{

struct any_system_tag
  : thrust::exelwtion_policy<any_system_tag>
{
  // allow any_system_tag to colwert to any type at all
  // XXX make this safer using enable_if<is_tag<T>> upon c++11
  template<typename T> operator T () const {return T();}
};

} // end thrust

