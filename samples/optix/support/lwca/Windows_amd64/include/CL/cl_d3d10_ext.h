/**********************************************************************************
 * Copyright (c) 2008-2009 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 **********************************************************************************/

#ifndef __OPENCL_CL_D3D10_EXT_H
#define __OPENCL_CL_D3D10_EXT_H

#include <d3d10.h>
#include <CL/cl.h>
#include <CL/cl_platform.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * cl_lw_d3d10_sharing                                                        */

typedef cl_uint cl_d3d10_device_source_lw;
typedef cl_uint cl_d3d10_device_set_lw;

/******************************************************************************/

// Error Codes
#define CL_ILWALID_D3D10_DEVICE_LW             -1002
#define CL_ILWALID_D3D10_RESOURCE_LW           -1003
#define CL_D3D10_RESOURCE_ALREADY_ACQUIRED_LW  -1004
#define CL_D3D10_RESOURCE_NOT_ACQUIRED_LW      -1005

// cl_d3d10_device_source_lw
#define CL_D3D10_DEVICE_LW                     0x4010
#define CL_D3D10_DXGI_ADAPTER_LW               0x4011

// cl_d3d10_device_set_lw
#define CL_PREFERRED_DEVICES_FOR_D3D10_LW      0x4012
#define CL_ALL_DEVICES_FOR_D3D10_LW            0x4013

// cl_context_info
#define CL_CONTEXT_D3D10_DEVICE_LW             0x4014

// cl_mem_info
#define CL_MEM_D3D10_RESOURCE_LW               0x4015

// cl_image_info
#define CL_IMAGE_D3D10_SUBRESOURCE_LW          0x4016

// cl_command_type
#define CL_COMMAND_ACQUIRE_D3D10_OBJECTS_LW    0x4017
#define CL_COMMAND_RELEASE_D3D10_OBJECTS_LW    0x4018

/******************************************************************************/

typedef CL_API_ENTRY cl_int (CL_API_CALL *clGetDeviceIDsFromD3D10LW_fn)(
    cl_platform_id            platform,
    cl_d3d10_device_source_lw d3d_device_source,
    void *                    d3d_object,
    cl_d3d10_device_set_lw    d3d_device_set,
    cl_uint                   num_entries, 
    cl_device_id *            devices, 
    cl_uint *                 num_devices) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *clCreateFromD3D10BufferLW_fn)(
    cl_context     context,
    cl_mem_flags   flags,
    ID3D10Buffer * resource,
    cl_int *       errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *clCreateFromD3D10Texture2DLW_fn)(
    cl_context        context,
    cl_mem_flags      flags,
    ID3D10Texture2D * resource,
    UINT              subresource,
    cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *clCreateFromD3D10Texture3DLW_fn)(
    cl_context        context,
    cl_mem_flags      flags,
    ID3D10Texture3D * resource,
    UINT              subresource,
    cl_int *          errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clEnqueueAcquireD3D10ObjectsLW_fn)(
    cl_command_queue command_queue,
    cl_uint          num_objects,
    const cl_mem *   mem_objects,
    cl_uint          num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event *       event) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clEnqueueReleaseD3D10ObjectsLW_fn)(
    cl_command_queue command_queue,
    cl_uint          num_objects,
    cl_mem *         mem_objects,
    cl_uint          num_events_in_wait_list,
    const cl_event * event_wait_list,
    cl_event *       event) CL_API_SUFFIX__VERSION_1_0;

#ifdef __cplusplus
}
#endif

#endif  // __OPENCL_CL_D3D10_H

