/******************************************************************************
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_LWCC

#include <thrust/detail/config/exec_check_disable.h>
#include <thrust/detail/cstdint.h>
#include <thrust/detail/type_traits.h>
#include <thrust/distance.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/lwca/config.h>
#include <thrust/system/lwca/detail/dispatch.h>

#include <cub/device/device_scan.cuh>

namespace thrust
{
namespace lwca_cub
{
namespace detail
{

__thrust_exec_check_disable__
template <typename Derived,
          typename InputIt,
          typename Size,
          typename OutputIt,
          typename ScanOp>
__host__ __device__
OutputIt inclusive_scan_n_impl(thrust::lwca_cub::execution_policy<Derived> &policy,
                               InputIt first,
                               Size num_items,
                               OutputIt result,
                               ScanOp scan_op)
{
  using Dispatch32 = cub::DispatchScan<InputIt,
                                       OutputIt,
                                       ScanOp,
                                       cub::NullType,
                                       thrust::detail::int32_t>;
  using Dispatch64 = cub::DispatchScan<InputIt,
                                       OutputIt,
                                       ScanOp,
                                       cub::NullType,
                                       thrust::detail::int64_t>;

  lwcaStream_t stream = thrust::lwca_cub::stream(policy);
  lwcaError_t status;

  // Determine temporary storage requirements:
  size_t tmp_size = 0;
  {
    THRUST_INDEX_TYPE_DISPATCH2(status,
                                Dispatch32::Dispatch,
                                Dispatch64::Dispatch,
                                num_items,
                                (nullptr,
                                 tmp_size,
                                 first,
                                 result,
                                 scan_op,
                                 cub::NullType{},
                                 num_items_fixed,
                                 stream,
                                 THRUST_DEBUG_SYNC_FLAG));
    thrust::lwca_cub::throw_on_error(status,
                                     "after determining tmp storage "
                                     "requirements for inclusive_scan");
  }

  // Run scan:
  {
    // Allocate temporary storage:
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived> tmp{
      policy,
      tmp_size};
    THRUST_INDEX_TYPE_DISPATCH2(status,
                                Dispatch32::Dispatch,
                                Dispatch64::Dispatch,
                                num_items,
                                (tmp.data().get(),
                                 tmp_size,
                                 first,
                                 result,
                                 scan_op,
                                 cub::NullType{},
                                 num_items_fixed,
                                 stream,
                                 THRUST_DEBUG_SYNC_FLAG));
    thrust::lwca_cub::throw_on_error(status,
                                     "after dispatching inclusive_scan kernel");
    thrust::lwca_cub::throw_on_error(thrust::lwca_cub::synchronize(policy),
                                     "inclusive_scan failed to synchronize");
  }

  return result + num_items;
}

__thrust_exec_check_disable__
template <typename Derived,
          typename InputIt,
          typename Size,
          typename OutputIt,
          typename InitValueT,
          typename ScanOp>
__host__ __device__
OutputIt exclusive_scan_n_impl(thrust::lwca_cub::execution_policy<Derived> &policy,
                               InputIt first,
                               Size num_items,
                               OutputIt result,
                               InitValueT init,
                               ScanOp scan_op)
{
  using Dispatch32 = cub::DispatchScan<InputIt,
                                       OutputIt,
                                       ScanOp,
                                       InitValueT,
                                       thrust::detail::int32_t>;
  using Dispatch64 = cub::DispatchScan<InputIt,
                                       OutputIt,
                                       ScanOp,
                                       InitValueT,
                                       thrust::detail::int64_t>;

  lwcaStream_t stream = thrust::lwca_cub::stream(policy);
  lwcaError_t status;

  // Determine temporary storage requirements:
  size_t tmp_size = 0;
  {
    THRUST_INDEX_TYPE_DISPATCH2(status,
                                Dispatch32::Dispatch,
                                Dispatch64::Dispatch,
                                num_items,
                                (nullptr,
                                 tmp_size,
                                 first,
                                 result,
                                 scan_op,
                                 init,
                                 num_items_fixed,
                                 stream,
                                 THRUST_DEBUG_SYNC_FLAG));
    thrust::lwca_cub::throw_on_error(status,
                                     "after determining tmp storage "
                                     "requirements for exclusive_scan");
  }

  // Run scan:
  {
    // Allocate temporary storage:
    thrust::detail::temporary_array<thrust::detail::uint8_t, Derived> tmp{
      policy,
      tmp_size};
    THRUST_INDEX_TYPE_DISPATCH2(status,
                                Dispatch32::Dispatch,
                                Dispatch64::Dispatch,
                                num_items,
                                (tmp.data().get(),
                                 tmp_size,
                                 first,
                                 result,
                                 scan_op,
                                 init,
                                 num_items_fixed,
                                 stream,
                                 THRUST_DEBUG_SYNC_FLAG));
    thrust::lwca_cub::throw_on_error(status,
                                     "after dispatching exclusive_scan kernel");
    thrust::lwca_cub::throw_on_error(thrust::lwca_cub::synchronize(policy),
                                     "exclusive_scan failed to synchronize");
  }

  return result + num_items;
}

} // namespace detail

//-------------------------
// Thrust API entry points
//-------------------------

__thrust_exec_check_disable__
template <typename Derived,
          typename InputIt,
          typename Size,
          typename OutputIt,
          typename ScanOp>
__host__ __device__
OutputIt inclusive_scan_n(thrust::lwca_cub::execution_policy<Derived> &policy,
                          InputIt first,
                          Size num_items,
                          OutputIt result,
                          ScanOp scan_op)
{
  OutputIt ret = result;
  if (__THRUST_HAS_CUDART__)
  {
    ret = thrust::lwca_cub::detail::inclusive_scan_n_impl(policy,
                                                          first,
                                                          num_items,
                                                          result,
                                                          scan_op);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::inclusive_scan(cvt_to_seq(derived_cast(policy)),
                                 first,
                                 first + num_items,
                                 result,
                                 scan_op);
#endif
  }
  return ret;
}

template <typename Derived, typename InputIt, typename OutputIt, typename ScanOp>
__host__ __device__
OutputIt inclusive_scan(thrust::lwca_cub::execution_policy<Derived> &policy,
                        InputIt first,
                        InputIt last,
                        OutputIt result,
                        ScanOp scan_op)
{
  using diff_t = typename thrust::iterator_traits<InputIt>::difference_type;
  diff_t const num_items = thrust::distance(first, last);
  return thrust::lwca_cub::inclusive_scan_n(policy,
                                            first,
                                            num_items,
                                            result,
                                            scan_op);
}

template <typename Derived, typename InputIt, typename OutputIt>
__host__ __device__
OutputIt inclusive_scan(thrust::lwca_cub::execution_policy<Derived> &policy,
                        InputIt first,
                        InputIt last,
                        OutputIt result)
{
  return thrust::lwca_cub::inclusive_scan(policy,
                                          first,
                                          last,
                                          result,
                                          thrust::plus<>{});
}

__thrust_exec_check_disable__
template <typename Derived,
          typename InputIt,
          typename Size,
          typename OutputIt,
          typename T,
          typename ScanOp>
__host__ __device__
OutputIt exclusive_scan_n(thrust::lwca_cub::execution_policy<Derived> &policy,
                          InputIt first,
                          Size num_items,
                          OutputIt result,
                          T init,
                          ScanOp scan_op)
{
  OutputIt ret = result;
  if (__THRUST_HAS_CUDART__)
  {
    ret = thrust::lwca_cub::detail::exclusive_scan_n_impl(policy,
                                                          first,
                                                          num_items,
                                                          result,
                                                          init,
                                                          scan_op);
  }
  else
  {
#if !__THRUST_HAS_CUDART__
    ret = thrust::exclusive_scan(cvt_to_seq(derived_cast(policy)),
                                 first,
                                 first + num_items,
                                 result,
                                 init,
                                 scan_op);
#endif
  }
  return ret;
}

template <typename Derived,
          typename InputIt,
          typename OutputIt,
          typename T,
          typename ScanOp>
__host__ __device__
OutputIt exclusive_scan(thrust::lwca_cub::execution_policy<Derived> &policy,
                        InputIt first,
                        InputIt last,
                        OutputIt result,
                        T init,
                        ScanOp scan_op)
{
  using diff_t = typename thrust::iterator_traits<InputIt>::difference_type;
  diff_t const num_items = thrust::distance(first, last);
  return thrust::lwca_cub::exclusive_scan_n(policy,
                                            first,
                                            num_items,
                                            result,
                                            init,
                                            scan_op);
}

template <typename Derived, typename InputIt, typename OutputIt, typename T>
__host__ __device__
OutputIt exclusive_scan(thrust::lwca_cub::execution_policy<Derived> &policy,
                        InputIt first,
                        InputIt last,
                        OutputIt result,
                        T init)
{
  return thrust::lwca_cub::exclusive_scan(policy,
                                          first,
                                          last,
                                          result,
                                          init,
                                          thrust::plus<>{});
}

template <typename Derived, typename InputIt, typename OutputIt>
__host__ __device__
OutputIt exclusive_scan(thrust::lwca_cub::execution_policy<Derived> &policy,
                        InputIt first,
                        InputIt last,
                        OutputIt result)
{
  using init_type = typename thrust::iterator_traits<InputIt>::value_type;
  return lwca_cub::exclusive_scan(policy, first, last, result, init_type{});
};

} // namespace lwca_cub
} // namespace thrust

#include <thrust/scan.h>

#endif // NVCC
