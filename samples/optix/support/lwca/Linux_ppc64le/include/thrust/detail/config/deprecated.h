/*
 *  Copyright 2018-2020 LWPU Corporation
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

/*! \file deprecated.h
 *  \brief Defines the THRUST_DEPRECATED macro
 */

#pragma once

#include <thrust/detail/config/compiler.h>

#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_MSVC
#  define THRUST_DEPRECATED __declspec(deprecated)
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
#  define THRUST_DEPRECATED __attribute__((deprecated))
#elif THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC
#  define THRUST_DEPRECATED __attribute__((deprecated))
#else
#  define THRUST_DEPRECATED
#endif
