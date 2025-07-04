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

// #include the host system's exelwtion_policy header
#define __THRUST_HOST_SYSTEM_TAG_HEADER <__THRUST_HOST_SYSTEM_ROOT/detail/exelwtion_policy.h>
#include __THRUST_HOST_SYSTEM_TAG_HEADER
#undef __THRUST_HOST_SYSTEM_TAG_HEADER

namespace thrust
{

typedef thrust::system::__THRUST_HOST_SYSTEM_NAMESPACE::tag host_system_tag;

} // end thrust

// TODO remove this in 1.8.0
namespace thrust
{

typedef THRUST_DEPRECATED host_system_tag host_space_tag;

} // end thrust

