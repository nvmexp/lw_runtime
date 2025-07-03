/*
 *  Copyright 2018 LWPU Corporation
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
#include <thrust/detail/cpp11_required.h>

#if THRUST_CPP_DIALECT >= 2011

#include <tuple>

#include <thrust/detail/exelwte_with_dependencies.h>

namespace thrust
{
namespace detail
{

template<template<typename> class ExelwtionPolicyCRTPBase>
struct dependencies_aware_exelwtion_policy
{
    template<typename ...Dependencies>
    __host__
    thrust::detail::exelwte_with_dependencies<
        ExelwtionPolicyCRTPBase,
        Dependencies...
    >
    after(Dependencies&& ...dependencies) const
    {
        return { capture_as_dependency(THRUST_FWD(dependencies))... };
    }

    template<typename ...Dependencies>
    __host__
    thrust::detail::exelwte_with_dependencies<
        ExelwtionPolicyCRTPBase,
        Dependencies...
    >
    after(std::tuple<Dependencies...>& dependencies) const
    {
        return { capture_as_dependency(dependencies) };
    }
    template<typename ...Dependencies>
    __host__
    thrust::detail::exelwte_with_dependencies<
        ExelwtionPolicyCRTPBase,
        Dependencies...
    >
    after(std::tuple<Dependencies...>&& dependencies) const
    {
        return { capture_as_dependency(std::move(dependencies)) };
    }

    template<typename ...Dependencies>
    __host__
    thrust::detail::exelwte_with_dependencies<
        ExelwtionPolicyCRTPBase,
        Dependencies...
    >
    rebind_after(Dependencies&& ...dependencies) const
    {
        return { capture_as_dependency(THRUST_FWD(dependencies))... };
    }

    template<typename ...Dependencies>
    __host__
    thrust::detail::exelwte_with_dependencies<
        ExelwtionPolicyCRTPBase,
        Dependencies...
    >
    rebind_after(std::tuple<Dependencies...>& dependencies) const
    {
        return { capture_as_dependency(dependencies) };
    }
    template<typename ...Dependencies>
    __host__
    thrust::detail::exelwte_with_dependencies<
        ExelwtionPolicyCRTPBase,
        Dependencies...
    >
    rebind_after(std::tuple<Dependencies...>&& dependencies) const
    {
        return { capture_as_dependency(std::move(dependencies)) };
    }
};

} // end detail
} // end thrust

#endif // THRUST_CPP_DIALECT >= 2011

