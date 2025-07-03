
/*
 * Copyright (c) 2021 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

 /**
 * @file   optix_lwda_interop.h
 * @author LWPU Corporation
 * @brief  OptiX public API declarations LWDAInterop
 *
 * OptiX public API declarations for LWCA interoperability
 */

#ifndef __optix_optix_lwda_interop_h__
#define __optix_optix_lwda_interop_h__

#ifdef OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#include "optix.h"
#else // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD
#include "o6/optix.h"
#endif // OPTIX_OPTIONAL_FEATURE_EXTERNAL_BUILD

#ifdef __cplusplus
extern "C" {
#endif

  /**
  * @brief Creates a new buffer object that will later rely on user-side LWCA allocation
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * Deprecated in OptiX 4.0. Now forwards to @ref rtBufferCreate.
  *
  * @param[in]   context          The context to create the buffer in
  * @param[in]   bufferdesc       Bitwise \a or combination of the \a type and \a flags of the new buffer
  * @param[out]  buffer           The return handle for the buffer object
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ILWALID_CONTEXT
  * - @ref RT_ERROR_ILWALID_VALUE
  * - @ref RT_ERROR_MEMORY_ALLOCATION_FAILED
  *
  * <B>History</B>
  *
  * @ref rtBufferCreateForLWDA was introduced in OptiX 3.0.
  *
  * <B>See also</B>
  * @ref rtBufferCreate,
  * @ref rtBufferSetDevicePointer,
  * @ref rtBufferMarkDirty,
  * @ref rtBufferDestroy
  *
  */
  RTresult RTAPI rtBufferCreateForLWDA (RTcontext context, unsigned int bufferdesc, RTbuffer *buffer);

  /**
  * @brief Gets the pointer to the buffer's data on the given device
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferGetDevicePointer returns the pointer to the data of \a buffer on device \a optix_device_ordinal in **\a device_pointer.
  *
  * If @ref rtBufferGetDevicePointer has been called for a single device for a given buffer,
  * the user can change the buffer's content on that device through the pointer. OptiX must then synchronize the new buffer contents to all devices.
  * These synchronization copies occur at every @ref rtContextLaunch "rtContextLaunch", unless the buffer is created with @ref RT_BUFFER_COPY_ON_DIRTY.
  * In this case, @ref rtBufferMarkDirty can be used to notify OptiX that the buffer has been dirtied and must be synchronized.
  *
  * @param[in]   buffer                          The buffer to be queried for its device pointer
  * @param[in]   optix_device_ordinal            The number assigned by OptiX to the device
  * @param[out]  device_pointer                  The return handle to the buffer's device pointer
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ILWALID_CONTEXT
  * - @ref RT_ERROR_ILWALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtBufferGetDevicePointer was introduced in OptiX 3.0.
  *
  * <B>See also</B>
  * @ref rtBufferMarkDirty,
  * @ref rtBufferSetDevicePointer
  *
  */
  RTresult RTAPI rtBufferGetDevicePointer (RTbuffer buffer, int optix_device_ordinal, void** device_pointer);

  /**
  * @brief Sets a buffer as dirty
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * If @ref rtBufferSetDevicePointer or @ref rtBufferGetDevicePointer have been called for a single device for a given buffer,
  * the user can change the buffer's content on that device through the pointer. OptiX must then synchronize the new buffer contents to all devices.
  * These synchronization copies occur at every @ref rtContextLaunch, unless the buffer is declared with @ref RT_BUFFER_COPY_ON_DIRTY.
  * In this case, @ref rtBufferMarkDirty can be used to notify OptiX that the buffer has been dirtied and must be synchronized.
  *
  * Note that RT_BUFFER_COPY_ON_DIRTY lwrrently only applies to LWCA interop buffers (buffers for which the application has a device pointer).
  *
  * @param[in]   buffer                          The buffer to be marked dirty
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ILWALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtBufferMarkDirty was introduced in OptiX 3.0.
  *
  * <B>See also</B>
  * @ref rtBufferGetDevicePointer,
  * @ref rtBufferSetDevicePointer,
  * @ref RT_BUFFER_COPY_ON_DIRTY
  *
  */
  RTresult RTAPI rtBufferMarkDirty (RTbuffer buffer);

  /**
  * @brief Sets the pointer to the buffer's data on the given device
  *
  * @ingroup Buffer
  *
  * <B>Description</B>
  *
  * @ref rtBufferSetDevicePointer sets the pointer to the data of \a buffer on device \a optix_device_ordinal to \a device_pointer.
  *
  * If @ref rtBufferSetDevicePointer has been called for a single device for a given buffer,
  * the user can change the buffer's content on that device through the pointer. OptiX must then synchronize the new buffer contents to all devices.
  * These synchronization copies occur at every @ref rtContextLaunch "rtContextLaunch", unless the buffer is declared with @ref RT_BUFFER_COPY_ON_DIRTY.
  * In this case, @ref rtBufferMarkDirty can be used to notify OptiX that the buffer has been dirtied and must be synchronized.
  *
  * @param[in]   buffer                          The buffer for which the device pointer is to be set
  * @param[in]   optix_device_ordinal            The number assigned by OptiX to the device
  * @param[in]   device_pointer                  The pointer to the data on the specified device
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ILWALID_VALUE
  * - @ref RT_ERROR_ILWALID_CONTEXT
  *
  * <B>History</B>
  *
  * @ref rtBufferSetDevicePointer was introduced in OptiX 3.0.
  *
  * <B>See also</B>
  * @ref rtBufferMarkDirty,
  * @ref rtBufferGetDevicePointer
  *
  */
  RTresult RTAPI rtBufferSetDevicePointer (RTbuffer buffer, int optix_device_ordinal, void* device_pointer);

  /**
  * @brief Sets a LWCA synchronization stream for the command list
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtCommandListSetLwdaStream sets a LWCA synchronization stream for the command list. The
  * command list guarantees that all work on the synchronization stream finishes before any launches
  * of the command list exelwtes on the GPU. It will also have the synchronization stream wait for
  * those launches to complete using LWCA events. This means lwca interop, such as memory copying
  * or kernel exelwtion, can be done in a safe way both before and after exelwting a command list.
  * If LWCA interop is made using streams other than the synchronization stream then LWCA events
  * must be used to make sure that the synchronization stream waits for all work done by other
  * streams, and also that the other streams wait for the synchronization stream after exelwting
  * the command list.
  *
  * Note that the synchronization stream can be created on any active device, there is no need to
  * have one per device.
  *
  * @param[in]   list                            The command list buffer for which the stream is to be set
  * @param[in]   stream                          The LWCA stream to set
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ILWALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtCommandListSetLwdaStream was introduced in OptiX 6.1.
  *
  * <B>See also</B>
  * @ref rtCommandListExelwte
  * @ref rtCommandListGetLwdaStream
  *
  */
  RTresult RTAPI rtCommandListSetLwdaStream( RTcommandlist list, void* stream );

  /**
  * @brief Gets the LWCA synchronization stream set for the command list
  *
  * @ingroup CommandList
  *
  * <B>Description</B>
  *
  * @ref rtCommandListGetLwdaStream gets the LWCA synchronization stream set for the command list.
  *
  * @param[in]   list                            The command list buffer for which to get the stream
  * @param[out]  stream                          Set to the LWCA stream of the command list
  *
  * <B>Return values</B>
  *
  * Relevant return values:
  * - @ref RT_SUCCESS
  * - @ref RT_ERROR_ILWALID_VALUE
  *
  * <B>History</B>
  *
  * @ref rtCommandListGetLwdaStream was introduced in OptiX 6.1.
  *
  * <B>See also</B>
  * @ref rtCommandListSetCommandList
  *
  */
  RTresult RTAPI rtCommandListGetLwdaStream( RTcommandlist list, void** stream );

#ifdef __cplusplus
}
#endif

#endif /* __optix_optix_lwda_interop_h__ */
