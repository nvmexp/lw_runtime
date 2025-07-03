/* Copyright (c) 2022, LWPU CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of LWPU CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROLWREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef COMMON_DRVAPI_ERROR_STRING_H_
#define COMMON_DRVAPI_ERROR_STRING_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __lwda_lwda_h__  // check to see if LWDA_H is included above

// Error Code string definitions here
typedef struct {
  char const *error_string;
  int error_id;
} s_LwdaErrorStr;

/**
 * Error codes
 */
static s_LwdaErrorStr sLwdaDrvErrorString[] = {
    /**
     * The API call returned with no errors. In the case of query calls, this
     * can also mean that the operation being queried is complete (see
     * ::lwEventQuery() and ::lwStreamQuery()).
     */
    {"LWDA_SUCCESS", 0},

    /**
     * This indicates that one or more of the parameters passed to the API call
     * is not within an acceptable range of values.
     */
    {"LWDA_ERROR_ILWALID_VALUE", 1},

    /**
     * The API call failed because it was unable to allocate enough memory to
     * perform the requested operation.
     */
    {"LWDA_ERROR_OUT_OF_MEMORY", 2},

    /**
     * This indicates that the LWCA driver has not been initialized with
     * ::lwInit() or that initialization has failed.
     */
    {"LWDA_ERROR_NOT_INITIALIZED", 3},

    /**
     * This indicates that the LWCA driver is in the process of shutting down.
     */
    {"LWDA_ERROR_DEINITIALIZED", 4},

    /**
     * This indicates profiling APIs are called while application is running
     * in visual profiler mode.
     */
    {"LWDA_ERROR_PROFILER_DISABLED", 5},
    /**
     * This indicates profiling has not been initialized for this context.
     * Call lwProfilerInitialize() to resolve this.
     */
    {"LWDA_ERROR_PROFILER_NOT_INITIALIZED", 6},
    /**
     * This indicates profiler has already been started and probably
     * lwProfilerStart() is incorrectly called.
     */
    {"LWDA_ERROR_PROFILER_ALREADY_STARTED", 7},
    /**
     * This indicates profiler has already been stopped and probably
     * lwProfilerStop() is incorrectly called.
     */
    {"LWDA_ERROR_PROFILER_ALREADY_STOPPED", 8},
    /**
     * This indicates that no LWCA-capable devices were detected by the
     * installed LWCA driver.
     */
    {"LWDA_ERROR_NO_DEVICE (no LWCA-capable devices were detected)", 100},

    /**
     * This indicates that the device ordinal supplied by the user does not
     * correspond to a valid LWCA device.
     */
    {"LWDA_ERROR_ILWALID_DEVICE (device specified is not a valid LWCA device)",
     101},

    /**
     * This indicates that the device kernel image is invalid. This can also
     * indicate an invalid LWCA module.
     */
    {"LWDA_ERROR_ILWALID_IMAGE", 200},

    /**
     * This most frequently indicates that there is no context bound to the
     * current thread. This can also be returned if the context passed to an
     * API call is not a valid handle (such as a context that has had
     * ::lwCtxDestroy() ilwoked on it). This can also be returned if a user
     * mixes different API versions (i.e. 3010 context with 3020 API calls).
     * See ::lwCtxGetApiVersion() for more details.
     */
    {"LWDA_ERROR_ILWALID_CONTEXT", 201},

    /**
     * This indicated that the context being supplied as a parameter to the
     * API call was already the active context.
     * \deprecated
     * This error return is deprecated as of LWCA 3.2. It is no longer an
     * error to attempt to push the active context via ::lwCtxPushLwrrent().
     */
    {"LWDA_ERROR_CONTEXT_ALREADY_LWRRENT", 202},

    /**
     * This indicates that a map or register operation has failed.
     */
    {"LWDA_ERROR_MAP_FAILED", 205},

    /**
     * This indicates that an unmap or unregister operation has failed.
     */
    {"LWDA_ERROR_UNMAP_FAILED", 206},

    /**
     * This indicates that the specified array is lwrrently mapped and thus
     * cannot be destroyed.
     */
    {"LWDA_ERROR_ARRAY_IS_MAPPED", 207},

    /**
     * This indicates that the resource is already mapped.
     */
    {"LWDA_ERROR_ALREADY_MAPPED", 208},

    /**
     * This indicates that there is no kernel image available that is suitable
     * for the device. This can occur when a user specifies code generation
     * options for a particular LWCA source file that do not include the
     * corresponding device configuration.
     */
    {"LWDA_ERROR_NO_BINARY_FOR_GPU", 209},

    /**
     * This indicates that a resource has already been acquired.
     */
    {"LWDA_ERROR_ALREADY_ACQUIRED", 210},

    /**
     * This indicates that a resource is not mapped.
     */
    {"LWDA_ERROR_NOT_MAPPED", 211},

    /**
     * This indicates that a mapped resource is not available for access as an
     * array.
     */
    {"LWDA_ERROR_NOT_MAPPED_AS_ARRAY", 212},

    /**
     * This indicates that a mapped resource is not available for access as a
     * pointer.
     */
    {"LWDA_ERROR_NOT_MAPPED_AS_POINTER", 213},

    /**
     * This indicates that an uncorrectable ECC error was detected during
     * exelwtion.
     */
    {"LWDA_ERROR_ECC_UNCORRECTABLE", 214},

    /**
     * This indicates that the ::LWlimit passed to the API call is not
     * supported by the active device.
     */
    {"LWDA_ERROR_UNSUPPORTED_LIMIT", 215},

    /**
     * This indicates that the ::LWcontext passed to the API call can
     * only be bound to a single CPU thread at a time but is already
     * bound to a CPU thread.
     */
    {"LWDA_ERROR_CONTEXT_ALREADY_IN_USE", 216},

    /**
     * This indicates that peer access is not supported across the given
     * devices.
     */
    {"LWDA_ERROR_PEER_ACCESS_UNSUPPORTED", 217},

    /**
     * This indicates that a PTX JIT compilation failed.
     */
    {"LWDA_ERROR_ILWALID_PTX", 218},

    /**
     * This indicates an error with OpenGL or DirectX context.
     */
    {"LWDA_ERROR_ILWALID_GRAPHICS_CONTEXT", 219},

    /**
     * This indicates that an uncorrectable LWLink error was detected during the
     * exelwtion.
     */
    {"LWDA_ERROR_LWLINK_UNCORRECTABLE", 220},

    /**
     * This indicates that the PTX JIT compiler library was not found.
     */
    {"LWDA_ERROR_JIT_COMPILER_NOT_FOUND", 221},

    /**
     * This indicates that the device kernel source is invalid.
     */
    {"LWDA_ERROR_ILWALID_SOURCE", 300},

    /**
     * This indicates that the file specified was not found.
     */
    {"LWDA_ERROR_FILE_NOT_FOUND", 301},

    /**
     * This indicates that a link to a shared object failed to resolve.
     */
    {"LWDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND", 302},

    /**
     * This indicates that initialization of a shared object failed.
     */
    {"LWDA_ERROR_SHARED_OBJECT_INIT_FAILED", 303},

    /**
     * This indicates that an OS call failed.
     */
    {"LWDA_ERROR_OPERATING_SYSTEM", 304},

    /**
     * This indicates that a resource handle passed to the API call was not
     * valid. Resource handles are opaque types like ::LWstream and ::LWevent.
     */
    {"LWDA_ERROR_ILWALID_HANDLE", 400},

    /**
     * This indicates that a named symbol was not found. Examples of symbols
     * are global/constant variable names, texture names }, and surface names.
     */
    {"LWDA_ERROR_NOT_FOUND", 500},

    /**
     * This indicates that asynchronous operations issued previously have not
     * completed yet. This result is not actually an error, but must be
     * indicated differently than ::LWDA_SUCCESS (which indicates completion).
     * Calls that may return this value include ::lwEventQuery() and
     * ::lwStreamQuery().
     */
    {"LWDA_ERROR_NOT_READY", 600},

    /**
     * While exelwting a kernel, the device encountered a
     * load or store instruction on an invalid memory address.
     * This leaves the process in an inconsistent state and any further LWCA
     * work will return the same error. To continue using LWCA, the process must
     * be terminated and relaunched.
     */
    {"LWDA_ERROR_ILLEGAL_ADDRESS", 700},

    /**
     * This indicates that a launch did not occur because it did not have
     * appropriate resources. This error usually indicates that the user has
     * attempted to pass too many arguments to the device kernel, or the
     * kernel launch specifies too many threads for the kernel's register
     * count. Passing arguments of the wrong size (i.e. a 64-bit pointer
     * when a 32-bit int is expected) is equivalent to passing too many
     * arguments and can also result in this error.
     */
    {"LWDA_ERROR_LAUNCH_OUT_OF_RESOURCES", 701},

    /**
     * This indicates that the device kernel took too long to execute. This can
     * only occur if timeouts are enabled - see the device attribute
     * ::LW_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT for more information. The
     * context cannot be used (and must be destroyed similar to
     * ::LWDA_ERROR_LAUNCH_FAILED). All existing device memory allocations from
     * this context are invalid and must be reconstructed if the program is to
     * continue using LWCA.
     */
    {"LWDA_ERROR_LAUNCH_TIMEOUT", 702},

    /**
     * This error indicates a kernel launch that uses an incompatible texturing
     * mode.
     */
    {"LWDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING", 703},

    /**
     * This error indicates that a call to ::lwCtxEnablePeerAccess() is
     * trying to re-enable peer access to a context which has already
     * had peer access to it enabled.
     */
    {"LWDA_ERROR_PEER_ACCESS_ALREADY_ENABLED", 704},

    /**
     * This error indicates that ::lwCtxDisablePeerAccess() is
     * trying to disable peer access which has not been enabled yet
     * via ::lwCtxEnablePeerAccess().
     */
    {"LWDA_ERROR_PEER_ACCESS_NOT_ENABLED", 705},

    /**
     * This error indicates that the primary context for the specified device
     * has already been initialized.
     */
    {"LWDA_ERROR_PRIMARY_CONTEXT_ACTIVE", 708},

    /**
     * This error indicates that the context current to the calling thread
     * has been destroyed using ::lwCtxDestroy }, or is a primary context which
     * has not yet been initialized.
     */
    {"LWDA_ERROR_CONTEXT_IS_DESTROYED", 709},

    /**
     * A device-side assert triggered during kernel exelwtion. The context
     * cannot be used anymore, and must be destroyed. All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using LWCA.
     */
    {"LWDA_ERROR_ASSERT", 710},

    /**
     * This error indicates that the hardware resources required to enable
     * peer access have been exhausted for one or more of the devices
     * passed to ::lwCtxEnablePeerAccess().
     */
    {"LWDA_ERROR_TOO_MANY_PEERS", 711},

    /**
     * This error indicates that the memory range passed to
     * ::lwMemHostRegister() has already been registered.
     */
    {"LWDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED", 712},

    /**
     * This error indicates that the pointer passed to ::lwMemHostUnregister()
     * does not correspond to any lwrrently registered memory region.
     */
    {"LWDA_ERROR_HOST_MEMORY_NOT_REGISTERED", 713},

    /**
     * While exelwting a kernel, the device encountered a stack error.
     * This can be due to stack corruption or exceeding the stack size limit.
     * This leaves the process in an inconsistent state and any further LWCA
     * work will return the same error. To continue using LWCA, the process must
     * be terminated and relaunched.
     */
    {"LWDA_ERROR_HARDWARE_STACK_ERROR", 714},

    /**
     * While exelwting a kernel, the device encountered an illegal instruction.
     * This leaves the process in an inconsistent state and any further LWCA
     * work will return the same error. To continue using LWCA, the process must
     * be terminated and relaunched.
     */
    {"LWDA_ERROR_ILLEGAL_INSTRUCTION", 715},

    /**
     * While exelwting a kernel, the device encountered a load or store
     * instruction on a memory address which is not aligned. This leaves the
     * process in an inconsistent state and any further LWCA work will return
     * the same error. To continue using LWCA, the process must be terminated
     * and relaunched.
     */
    {"LWDA_ERROR_MISALIGNED_ADDRESS", 716},

    /**
     * While exelwting a kernel, the device encountered an instruction
     * which can only operate on memory locations in certain address spaces
     * (global, shared, or local), but was supplied a memory address not
     * belonging to an allowed address space.
     * This leaves the process in an inconsistent state and any further LWCA
     * work will return the same error. To continue using LWCA, the process must
     * be terminated and relaunched.
     */
    {"LWDA_ERROR_ILWALID_ADDRESS_SPACE", 717},

    /**
     * While exelwting a kernel, the device program counter wrapped its address
     * space. This leaves the process in an inconsistent state and any further
     * LWCA work will return the same error. To continue using LWCA, the process
     * must be terminated and relaunched.
     */
    {"LWDA_ERROR_ILWALID_PC", 718},

    /**
     * An exception oclwrred on the device while exelwting a kernel. Common
     * causes include dereferencing an invalid device pointer and accessing
     * out of bounds shared memory. The context cannot be used }, so it must
     * be destroyed (and a new one should be created). All existing device
     * memory allocations from this context are invalid and must be
     * reconstructed if the program is to continue using LWCA.
     */
    {"LWDA_ERROR_LAUNCH_FAILED", 719},

    /**
     * This error indicates that the number of blocks launched per grid for a
     * kernel that was launched via either ::lwLaunchCooperativeKernel or
     * ::lwLaunchCooperativeKernelMultiDevice exceeds the maximum number of
     * blocks as allowed by ::lwOclwpancyMaxActiveBlocksPerMultiprocessor or
     * ::lwOclwpancyMaxActiveBlocksPerMultiprocessorWithFlags times the number
     * of multiprocessors as specified by the device attribute
     * ::LW_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
     */
    {"LWDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE", 720},

    /**
     * This error indicates that the attempted operation is not permitted.
     */
    {"LWDA_ERROR_NOT_PERMITTED", 800},

    /**
     * This error indicates that the attempted operation is not supported
     * on the current system or device.
     */
    {"LWDA_ERROR_NOT_SUPPORTED", 801},

    /**
     * This indicates that an unknown internal error has oclwrred.
     */
    {"LWDA_ERROR_UNKNOWN", 999},
    {NULL, -1}};

// This is just a linear search through the array, since the error_id's are not
// always olwrring conselwtively
inline const char *getLwdaDrvErrorString(LWresult error_id) {
  int index = 0;

  while (sLwdaDrvErrorString[index].error_id != error_id &&
         sLwdaDrvErrorString[index].error_id != -1) {
    index++;
  }

  if (sLwdaDrvErrorString[index].error_id == error_id)
    return (const char *)sLwdaDrvErrorString[index].error_string;
  else
    return (const char *)"LWDA_ERROR not found!";
}

#endif  // __lwda_lwda_h__

#endif  //  COMMON_DRVAPI_ERROR_STRING_H_
