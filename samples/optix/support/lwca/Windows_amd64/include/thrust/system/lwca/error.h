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


/*! \file thrust/system/lwca/error.h
 *  \brief LWCA-specific error reporting
 */

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/type_traits.h>
#include <thrust/system/error_code.h>
#include <thrust/system/lwca/detail/guarded_driver_types.h>

namespace thrust
{

namespace system
{

namespace lwca
{

// To construct an error_code after a LWCA Runtime error:
//
//   error_code(::lwdaGetLastError(), lwda_category())

// XXX N3000 prefers enum class errc { ... }
/*! Namespace for LWCA Runtime errors.
 */
namespace errc
{

/*! \p errc_t enumerates the kinds of LWCA Runtime errors.
 */
enum errc_t
{
  // from lwca/include/driver_types.h
  // mirror their order
  success                            = lwdaSuccess,
  missing_configuration              = lwdaErrorMissingConfiguration,
  memory_allocation                  = lwdaErrorMemoryAllocation,
  initialization_error               = lwdaErrorInitializationError,
  launch_failure                     = lwdaErrorLaunchFailure,
  prior_launch_failure               = lwdaErrorPriorLaunchFailure,
  launch_timeout                     = lwdaErrorLaunchTimeout,
  launch_out_of_resources            = lwdaErrorLaunchOutOfResources,
  ilwalid_device_function            = lwdaErrorIlwalidDeviceFunction,
  ilwalid_configuration              = lwdaErrorIlwalidConfiguration,
  ilwalid_device                     = lwdaErrorIlwalidDevice,
  ilwalid_value                      = lwdaErrorIlwalidValue,
  ilwalid_pitch_value                = lwdaErrorIlwalidPitchValue,
  ilwalid_symbol                     = lwdaErrorIlwalidSymbol,
  map_buffer_object_failed           = lwdaErrorMapBufferObjectFailed,
  unmap_buffer_object_failed         = lwdaErrorUnmapBufferObjectFailed,
  ilwalid_host_pointer               = lwdaErrorIlwalidHostPointer,
  ilwalid_device_pointer             = lwdaErrorIlwalidDevicePointer,
  ilwalid_texture                    = lwdaErrorIlwalidTexture,
  ilwalid_texture_binding            = lwdaErrorIlwalidTextureBinding,
  ilwalid_channel_descriptor         = lwdaErrorIlwalidChannelDescriptor,
  ilwalid_memcpy_direction           = lwdaErrorIlwalidMemcpyDirection,
  address_of_constant_error          = lwdaErrorAddressOfConstant,
  texture_fetch_failed               = lwdaErrorTextureFetchFailed,
  texture_not_bound                  = lwdaErrorTextureNotBound,
  synchronization_error              = lwdaErrorSynchronizationError,
  ilwalid_filter_setting             = lwdaErrorIlwalidFilterSetting,
  ilwalid_norm_setting               = lwdaErrorIlwalidNormSetting,
  mixed_device_exelwtion             = lwdaErrorMixedDeviceExelwtion,
  lwda_runtime_unloading             = lwdaErrorLwdartUnloading,
  unknown                            = lwdaErrorUnknown,
  not_yet_implemented                = lwdaErrorNotYetImplemented,
  memory_value_too_large             = lwdaErrorMemoryValueTooLarge,
  ilwalid_resource_handle            = lwdaErrorIlwalidResourceHandle,
  not_ready                          = lwdaErrorNotReady,
  insufficient_driver                = lwdaErrorInsufficientDriver,
  set_on_active_process_error        = lwdaErrorSetOnActiveProcess,
  no_device                          = lwdaErrorNoDevice,
  ecc_uncorrectable                  = lwdaErrorECLWncorrectable,

#if LWDART_VERSION >= 4020
  shared_object_symbol_not_found     = lwdaErrorSharedObjectSymbolNotFound,
  shared_object_init_failed          = lwdaErrorSharedObjectInitFailed,
  unsupported_limit                  = lwdaErrorUnsupportedLimit,
  duplicate_variable_name            = lwdaErrorDuplicateVariableName,
  duplicate_texture_name             = lwdaErrorDuplicateTextureName,
  duplicate_surface_name             = lwdaErrorDuplicateSurfaceName,
  devices_unavailable                = lwdaErrorDevicesUnavailable,
  ilwalid_kernel_image               = lwdaErrorIlwalidKernelImage,
  no_kernel_image_for_device         = lwdaErrorNoKernelImageForDevice,
  incompatible_driver_context        = lwdaErrorIncompatibleDriverContext,
  peer_access_already_enabled        = lwdaErrorPeerAccessAlreadyEnabled,
  peer_access_not_enabled            = lwdaErrorPeerAccessNotEnabled,
  device_already_in_use              = lwdaErrorDeviceAlreadyInUse,
  profiler_disabled                  = lwdaErrorProfilerDisabled,
  assert_triggered                   = lwdaErrorAssert,
  too_many_peers                     = lwdaErrorTooManyPeers,
  host_memory_already_registered     = lwdaErrorHostMemoryAlreadyRegistered,
  host_memory_not_registered         = lwdaErrorHostMemoryNotRegistered,
  operating_system_error             = lwdaErrorOperatingSystem,
#endif

#if LWDART_VERSION >= 5000
  peer_access_unsupported            = lwdaErrorPeerAccessUnsupported,
  launch_max_depth_exceeded          = lwdaErrorLaunchMaxDepthExceeded,
  launch_file_scoped_texture_used    = lwdaErrorLaunchFileScopedTex,
  launch_file_scoped_surface_used    = lwdaErrorLaunchFileScopedSurf,
  sync_depth_exceeded                = lwdaErrorSyncDepthExceeded,
  attempted_operation_not_permitted  = lwdaErrorNotPermitted,
  attempted_operation_not_supported  = lwdaErrorNotSupported,
#endif

  startup_failure                    = lwdaErrorStartupFailure
}; // end errc_t


} // end namespace errc

} // end namespace lwda_lwb

/*! \return A reference to an object of a type derived from class \p thrust::error_category.
 *  \note The object's \p equivalent virtual functions shall behave as specified
 *        for the class \p thrust::error_category. The object's \p name virtual function shall
 *        return a pointer to the string <tt>"lwca"</tt>. The object's
 *        \p default_error_condition virtual function shall behave as follows:
 *
 *        If the argument <tt>ev</tt> corresponds to a LWCA error value, the function
 *        shall return <tt>error_condition(ev,lwda_category())</tt>.
 *        Otherwise, the function shall return <tt>system_category.default_error_condition(ev)</tt>.
 */
inline const error_category &lwda_category(void);


// XXX N3000 prefers is_error_code_enum<lwca::errc>

/*! Specialization of \p is_error_code_enum for \p lwca::errc::errc_t
 */
template<> struct is_error_code_enum<lwca::errc::errc_t> : thrust::detail::true_type {};


// XXX replace lwca::errc::errc_t with lwca::errc upon c++0x
/*! \return <tt>error_code(static_cast<int>(e), lwca::error_category())</tt>
 */
inline error_code make_error_code(lwca::errc::errc_t e);


// XXX replace lwca::errc::errc_t with lwca::errc upon c++0x
/*! \return <tt>error_condition(static_cast<int>(e), lwca::error_category())</tt>.
 */
inline error_condition make_error_condition(lwca::errc::errc_t e);

} // end system

namespace lwda_lwb
{
namespace errc = system::lwca::errc;
} // end lwda_lwb

namespace lwca
{
// XXX replace with using system::lwda_errc upon c++0x
namespace errc = system::lwca::errc;
} // end lwca

using system::lwda_category;

} // end namespace thrust

#include <thrust/system/lwca/detail/error.inl>

