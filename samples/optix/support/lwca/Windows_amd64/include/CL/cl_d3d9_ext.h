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

#ifndef __OPENCL_CL_D3D9_EXT_H
#define __OPENCL_CL_D3D9_EXT_H

#include <d3d9.h>
#include <CL/cl.h>
#include <CL/cl_platform.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * cl_lw_d3d9_sharing                                                         */

typedef cl_uint cl_d3d9_device_source_lw;
typedef cl_uint cl_d3d9_device_set_lw;

/******************************************************************************/

// Error Codes
#define CL_ILWALID_D3D9_DEVICE_LW              -1010
#define CL_ILWALID_D3D9_RESOURCE_LW            -1011
#define CL_D3D9_RESOURCE_ALREADY_ACQUIRED_LW   -1012
#define CL_D3D9_RESOURCE_NOT_ACQUIRED_LW       -1013

// cl_d3d9_device_source_lw
#define CL_D3D9_DEVICE_LW                      0x4022
#define CL_D3D9_ADAPTER_NAME_LW                0x4023

// cl_d3d9_device_set_lw
#define CL_PREFERRED_DEVICES_FOR_D3D9_LW       0x4024
#define CL_ALL_DEVICES_FOR_D3D9_LW             0x4025

// cl_context_info
#define CL_CONTEXT_D3D9_DEVICE_LW              0x4026

// cl_mem_info
#define CL_MEM_D3D9_RESOURCE_LW                0x4027

// cl_image_info
#define CL_IMAGE_D3D9_FACE_LW                  0x4028
#define CL_IMAGE_D3D9_LEVEL_LW                 0x4029

// cl_command_type
#define CL_COMMAND_ACQUIRE_D3D9_OBJECTS_LW     0x402A
#define CL_COMMAND_RELEASE_D3D9_OBJECTS_LW     0x402B

/******************************************************************************/

typedef CL_API_ENTRY cl_int (CL_API_CALL *clGetDeviceIDsFromD3D9LW_fn)(
    cl_platform_id            platform,
    cl_d3d9_device_source_lw  d3d_device_source,
    void *                    d3d_object,
    cl_d3d9_device_set_lw     d3d_device_set,
    cl_uint                   num_entries, 
    cl_device_id *            devices, 
    cl_uint *                 num_devices) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *clCreateFromD3D9VertexBufferLW_fn)(
    cl_context               context,
    cl_mem_flags             flags,
    IDirect3DVertexBuffer9 * resource,
    cl_int *                 errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *clCreateFromD3D9IndexBufferLW_fn)(
    cl_context              context,
    cl_mem_flags            flags,
    IDirect3DIndexBuffer9 * resource,
    cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *clCreateFromD3D9SurfaceLW_fn)(
    cl_context          context,
    cl_mem_flags        flags,
    IDirect3DSurface9 * resource,
    cl_int *            errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *clCreateFromD3D9TextureLW_fn)(
    cl_context         context,
    cl_mem_flags       flags,
    IDirect3DTexture9 *resource,
    UINT               miplevel,
    cl_int *           errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *clCreateFromD3D9LwbeTextureLW_fn)(
    cl_context              context,
    cl_mem_flags            flags,
    IDirect3DLwbeTexture9 * resource,
    D3DLWBEMAP_FACES        facetype,
    UINT                    miplevel,
    cl_int *                errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_mem (CL_API_CALL *clCreateFromD3D9VolumeTextureLW_fn)(
    cl_context                context,
    cl_mem_flags              flags,
    IDirect3DVolumeTexture9 * resource,
    UINT                      miplevel,
    cl_int *                  errcode_ret) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clEnqueueAcquireD3D9ObjectsLW_fn)(
    cl_command_queue command_queue,
    cl_uint num_objects,
    const cl_mem *mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event) CL_API_SUFFIX__VERSION_1_0;

typedef CL_API_ENTRY cl_int (CL_API_CALL *clEnqueueReleaseD3D9ObjectsLW_fn)(
    cl_command_queue command_queue,
    cl_uint num_objects,
    cl_mem *mem_objects,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event) CL_API_SUFFIX__VERSION_1_0;

#ifdef __cplusplus
}
#endif

#endif  // __OPENCL_CL_D3D9_H

