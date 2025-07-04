// Copyright (c) 2018 LWPU Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/cpp14_required.h>

#if THRUST_CPP_DIALECT >= 2014

#include <thrust/system/lwca/pointer.h>
#include <thrust/system/lwca/detail/exelwtion_policy.h>

namespace thrust
{

namespace system { namespace lwca
{

struct ready_event;

template <typename T>
struct ready_future;

struct unique_eager_event;

template <typename T>
struct unique_eager_future;

template <typename... Events>
__host__
unique_eager_event when_all(Events&&... evs);

}} // namespace system::lwca

namespace lwca
{

using thrust::system::lwca::ready_event;

using thrust::system::lwca::ready_future;

using thrust::system::lwca::unique_eager_event;
using event = unique_eager_event;

using thrust::system::lwca::unique_eager_future;
template <typename T> using future = unique_eager_future<T>;

using thrust::system::lwca::when_all;

} // namespace lwca

template <typename DerivedPolicy>
__host__ 
thrust::lwca::unique_eager_event
unique_eager_event_type(
  thrust::lwca::exelwtion_policy<DerivedPolicy> const&
) noexcept;

template <typename T, typename DerivedPolicy>
__host__ 
thrust::lwca::unique_eager_future<T>
unique_eager_future_type(
  thrust::lwca::exelwtion_policy<DerivedPolicy> const&
) noexcept;

} // end namespace thrust

#include <thrust/system/lwca/detail/future.inl>

#endif // C++14

