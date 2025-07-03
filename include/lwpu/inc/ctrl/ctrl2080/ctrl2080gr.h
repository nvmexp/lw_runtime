/*
 * SPDX-FileCopyrightText: Copyright (c) 2006-2022 LWPU CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <lwtypes.h>
#if defined(_MSC_VER)
#pragma warning(disable:4324)
#endif

//
// This file was generated with FINN, an LWPU coding tool.
// Source file: ctrl/ctrl2080/ctrl2080gr.finn
//
#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)
//
// Please edit the original FINN IDL file to create desired edits in this header
// See https://confluence.lwpu.com/display/CORERM/FINN for more info on how to
// edit FINN.
//
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)




#include "ctrl/ctrl2080/ctrl2080base.h"

#include "ctrl/ctrl0080/ctrl0080gr.h"        /* 2080 is partially derivative of 0080 */
/*
 * LW2080_CTRL_GR_ROUTE_INFO
 *
 * This structure specifies the routing information used to
 * disambiguate the target GR engine.
 *
 *   flags
 *     This field decides how the route field is interpreted
 *
 *   route
 *     This field has the data to identify target GR engine
 *
 */
#define LW2080_CTRL_GR_ROUTE_INFO_FLAGS_TYPE                 1:0
#define LW2080_CTRL_GR_ROUTE_INFO_FLAGS_TYPE_NONE    0x0U
#define LW2080_CTRL_GR_ROUTE_INFO_FLAGS_TYPE_ENGID   0x1U
#define LW2080_CTRL_GR_ROUTE_INFO_FLAGS_TYPE_CHANNEL 0x2U

#define LW2080_CTRL_GR_ROUTE_INFO_DATA_CHANNEL_HANDLE      31:0
#define LW2080_CTRL_GR_ROUTE_INFO_DATA_ENGID               31:0

typedef LW0080_CTRL_GR_ROUTE_INFO LW2080_CTRL_GR_ROUTE_INFO;

/* LW20_SUBDEVICE_XX gr control commands and parameters */

/*
 * LW2080_CTRL_GR_INFO
 *
 * This structure represents a single 32bit gr engine value.  Clients
 * request a particular gr engine value by specifying a unique gr
 * information index.
 *
 * Legal gr information index values are:
 *   LW2080_CTRL_GR_INFO_INDEX_BUFFER_ALIGNMENT
 *     This index is used to request the surface buffer alignment (in bytes)
 *     required by the associated subdevice.  The return value is GPU
 *     implementation-dependent.
 *   LW2080_CTRL_GR_INFO_INDEX_SWIZZLE_ALIGNMENT
 *     This index is used to request the required swizzled surface alignment
 *     (in bytes) supported by the associated subdevice.  The return value
 *     is GPU implementation-dependent.  A return value of 0 indicates the GPU
 *     does not support swizzled surfaces.
 *   LW2080_CTRL_GR_INFO_INDEX_VERTEX_CACHE_SIZE
 *     This index is used to request the vertex cache size (in entries)
 *     supported by the associated subdevice.  The return value is GPU
 *     implementation-dependent.  A value of 0 indicates the GPU does
 *     have a vertex cache.
 *   LW2080_CTRL_GR_INFO_INDEX_VPE_COUNT
 *     This index is used to request the number of VPE units supported by the
 *     associated subdevice.  The return value is GPU implementation-dependent.
 *     A return value of 0 indicates the GPU does not contain VPE units.
 *   LW2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_COUNT
 *     This index is used to request the number of shader pipes supported by
 *     the associated subdevice.  The return value is GPU
 *     implementation-dependent.  A return value of 0 indicates the GPU does
 *     not contain dedicated shader units.
 *     For tesla: this value is the number of enabled TPCs
 *   LW2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_SUB_COUNT
 *     This index is used to request the number of sub units per
 *     shader pipes supported by the associated subdevice.  The return
 *     value is GPU implementation-dependent.  A return value of 0 indicates
 *     the GPU does not contain dedicated shader units.
 *     For tesla: this value is the number of enabled SMs (per TPC)
 *   LW2080_CTRL_GR_INFO_INDEX_THREAD_STACK_SCALING_FACTOR
 *     This index is used to request the scaling factor for thread stack
 *     memory.
 *     A value of 0 indicates the GPU does not support this function.
 *   LW2080_CTRL_GR_INFO_INDEX_SM_REG_BANK_COUNT
 *     This index is used to request the number of SM register banks supported.
 *     A value of 0 indicates the GPU does not support this function.
 *   LW2080_CTRL_GR_INFO_INDEX_SM_REG_BANK_REG_COUNT
 *     This index is used to request the number of registers per SM register
 *     bank. A value of 0 indicates the GPU does not support this function.
 *   LW2080_CTRL_GR_INFO_INDEX_SM_VERSION
 *     This index is used to determine the SM version.
 *     A value of 0 indicates the GPU does not support this function.
 *     Otherwise one of LW2080_CTRL_GR_INFO_SM_VERSION_*.
 *   LW2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM
 *     This index is used to determine the maximum number of warps
 *     (thread groups) per SM.
 *     A value of 0 indicates the GPU does not support this function.
 *   LW2080_CTRL_GR_INFO_INDEX_MAX_THREADS_PER_WARP
 *     This index is used to determine the maximum number of threads
 *     in each warp (thread group).
 *     A value of 0 indicates the GPU does not support this function.
 *   LW2080_CTRL_GR_INFO_INDEX_FB_MEMORY_REQUEST_GRANULARITY
 *     This index is used to request the default fb memory read/write request
 *     size in bytes (typically based on the memory configuration/controller).
 *     Smaller memory requests are likely to take as long as a full one.
 *     A value of 0 indicates the GPU does not support this function.
 *   LW2080_CTRL_GR_INFO_INDEX_HOST_MEMORY_REQUEST_GRANULARITY
 *     This index is used to request the default host memory read/write request
 *     size in bytes (typically based on the memory configuration/controller).
 *     Smaller memory requests are likely to take as long as a full one.
 *     A value of 0 indicates the GPU does not support this function.
 *   LW2080_CTRL_GR_INFO_INDEX_MAX_SP_PER_SM
 *     This index is used to request the maximum number of streaming processors
 *     per SM.
 *   LW2080_CTRL_GR_INFO_INDEX_LITTER_*
 *     This index is used to query the various LITTER size information from
 *     the chip.
 *   LW2080_CTRL_GR_INFO_INDEX_TIMESLICE_ENABLED
 *     This index is used to query whether the chip has timeslice mode enabled.
 *   LW2080_CTRL_GR_INFO_INDEX_GPU_CORE_COUNT
 *     This index is used to return the number of "GPU Cores"
 *     supported by the graphics pipeline
 *   LW2080_CTRL_GR_INFO_INDEX_RT_CORE_COUNT
 *     This index is used to return the number of "Ray Tracing Cores"
 *     supported by the graphics pipeline
 *    LW2080_CTRL_GR_INFO_INDEX_TENSOR_CORE_COUNT
 *     This index is used to return the number of "Tensor Cores"
 *     supported by the graphics pipeline
 */
typedef LW0080_CTRL_GR_INFO LW2080_CTRL_GR_INFO;

/*
 * Valid GR info index values
 * These indices are offset from supporting the 0080 version of this call
 */
#define LW2080_CTRL_GR_INFO_INDEX_MAXCLIPS                          LW0080_CTRL_GR_INFO_INDEX_MAXCLIPS
#define LW2080_CTRL_GR_INFO_INDEX_MIN_ATTRS_BUG_261894              LW0080_CTRL_GR_INFO_INDEX_MIN_ATTRS_BUG_261894
#define LW2080_CTRL_GR_INFO_XBUF_MAX_PSETS_PER_BANK                 LW0080_CTRL_GR_INFO_XBUF_MAX_PSETS_PER_BANK
/**
 * This index is used to request the surface buffer alignment (in bytes)
 * required by the associated subdevice.  The return value is GPU
 * implementation-dependent.
 */
#define LW2080_CTRL_GR_INFO_INDEX_BUFFER_ALIGNMENT                  LW0080_CTRL_GR_INFO_INDEX_BUFFER_ALIGNMENT
#define LW2080_CTRL_GR_INFO_INDEX_SWIZZLE_ALIGNMENT                 LW0080_CTRL_GR_INFO_INDEX_SWIZZLE_ALIGNMENT
#define LW2080_CTRL_GR_INFO_INDEX_VERTEX_CACHE_SIZE                 LW0080_CTRL_GR_INFO_INDEX_VERTEX_CACHE_SIZE
/**
 * This index is used to request the number of VPE units supported by the
 * associated subdevice.  The return value is GPU implementation-dependent.
 * A return value of 0 indicates the GPU does not contain VPE units.
 */
#define LW2080_CTRL_GR_INFO_INDEX_VPE_COUNT                         LW0080_CTRL_GR_INFO_INDEX_VPE_COUNT
/**
 * This index is used to request the number of shader pipes supported by
 * the associated subdevice.  The return value is GPU
 * implementation-dependent.  A return value of 0 indicates the GPU does
 * not contain dedicated shader units.
 * For tesla: this value is the number of enabled TPCs
 */
#define LW2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_COUNT                 LW0080_CTRL_GR_INFO_INDEX_SHADER_PIPE_COUNT
/**
 * This index is used to request the scaling factor for thread stack
 * memory.
 * A value of 0 indicates the GPU does not support this function.
 */
#define LW2080_CTRL_GR_INFO_INDEX_THREAD_STACK_SCALING_FACTOR       LW0080_CTRL_GR_INFO_INDEX_THREAD_STACK_SCALING_FACTOR
/**
 * This index is used to request the number of sub units per
 * shader pipes supported by the associated subdevice.  The return
 * value is GPU implementation-dependent.  A return value of 0 indicates
 * the GPU does not contain dedicated shader units.
 * For tesla: this value is the number of enabled SMs (per TPC)
 */
#define LW2080_CTRL_GR_INFO_INDEX_SHADER_PIPE_SUB_COUNT             LW0080_CTRL_GR_INFO_INDEX_SHADER_PIPE_SUB_COUNT
#define LW2080_CTRL_GR_INFO_INDEX_SM_REG_BANK_COUNT                 LW0080_CTRL_GR_INFO_INDEX_SM_REG_BANK_COUNT
#define LW2080_CTRL_GR_INFO_INDEX_SM_REG_BANK_REG_COUNT             LW0080_CTRL_GR_INFO_INDEX_SM_REG_BANK_REG_COUNT
/**
 * This index is used to determine the SM version.
 * A value of 0 indicates the GPU does not support this function.
 * Otherwise one of LW2080_CTRL_GR_INFO_SM_VERSION_*.
 */
#define LW2080_CTRL_GR_INFO_INDEX_SM_VERSION                        LW0080_CTRL_GR_INFO_INDEX_SM_VERSION
/**
 * This index is used to determine the maximum number of warps
 * (thread groups) per SM.
 * A value of 0 indicates the GPU does not support this function.
 */
#define LW2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM                  LW0080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM
/**
 * This index is used to determine the maximum number of threads
 * in each warp (thread group).
 * A value of 0 indicates the GPU does not support this function.
 */
#define LW2080_CTRL_GR_INFO_INDEX_MAX_THREADS_PER_WARP              LW0080_CTRL_GR_INFO_INDEX_MAX_THREADS_PER_WARP
#define LW2080_CTRL_GR_INFO_INDEX_GEOM_GS_OBUF_ENTRIES              LW0080_CTRL_GR_INFO_INDEX_GEOM_GS_OBUF_ENTRIES
#define LW2080_CTRL_GR_INFO_INDEX_GEOM_XBUF_ENTRIES                 LW0080_CTRL_GR_INFO_INDEX_GEOM_XBUF_ENTRIES
#define LW2080_CTRL_GR_INFO_INDEX_FB_MEMORY_REQUEST_GRANULARITY     LW0080_CTRL_GR_INFO_INDEX_FB_MEMORY_REQUEST_GRANULARITY
#define LW2080_CTRL_GR_INFO_INDEX_HOST_MEMORY_REQUEST_GRANULARITY   LW0080_CTRL_GR_INFO_INDEX_HOST_MEMORY_REQUEST_GRANULARITY
#define LW2080_CTRL_GR_INFO_INDEX_MAX_SP_PER_SM                     LW0080_CTRL_GR_INFO_INDEX_MAX_SP_PER_SM
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS                   LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCS
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_FBPS                   LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_FBPS
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_ZLWLL_BANKS            LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_ZLWLL_BANKS
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC            LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPC_PER_GPC
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_MIN_FBPS               LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_MIN_FBPS
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_MXBAR_FBP_PORTS        LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_MXBAR_FBP_PORTS
#define LW2080_CTRL_GR_INFO_INDEX_TIMESLICE_ENABLED                 LW0080_CTRL_GR_INFO_INDEX_TIMESLICE_ENABLED
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_FBPAS                  LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_FBPAS
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_PES_PER_GPC            LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_PES_PER_GPC
#define LW2080_CTRL_GR_INFO_INDEX_GPU_CORE_COUNT                    LW0080_CTRL_GR_INFO_INDEX_GPU_CORE_COUNT
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPCS_PER_PES           LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_TPCS_PER_PES
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_MXBAR_HUB_PORTS        LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_MXBAR_HUB_PORTS
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC             LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_SM_PER_TPC
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_HSHUB_FBP_PORTS        LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_HSHUB_FBP_PORTS
/**
 * This index is used to return the number of "Ray Tracing Cores"
 * supported by the graphics pipeline
 */
#define LW2080_CTRL_GR_INFO_INDEX_RT_CORE_COUNT                     LW0080_CTRL_GR_INFO_INDEX_RT_CORE_COUNT
#define LW2080_CTRL_GR_INFO_INDEX_TENSOR_CORE_COUNT                 LW0080_CTRL_GR_INFO_INDEX_TENSOR_CORE_COUNT
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GRS                    LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_GRS
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_LTCS                   LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_LTCS
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_LTC_SLICES             LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_LTC_SLICES
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCMMU_PER_GPC         LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_GPCMMU_PER_GPC
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_LTC_PER_FBP            LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_LTC_PER_FBP
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_ROP_PER_GPC            LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_ROP_PER_GPC
#define LW2080_CTRL_GR_INFO_INDEX_FAMILY_MAX_TPC_PER_GPC            LW0080_CTRL_GR_INFO_INDEX_FAMILY_MAX_TPC_PER_GPC
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_FBPA_PER_FBP           LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_FBPA_PER_FBP
#define LW2080_CTRL_GR_INFO_INDEX_MAX_SUBCONTEXT_COUNT              LW0080_CTRL_GR_INFO_INDEX_MAX_SUBCONTEXT_COUNT
#define LW2080_CTRL_GR_INFO_INDEX_MAX_LEGACY_SUBCONTEXT_COUNT       LW0080_CTRL_GR_INFO_INDEX_MAX_LEGACY_SUBCONTEXT_COUNT
#define LW2080_CTRL_GR_INFO_INDEX_MAX_PER_ENGINE_SUBCONTEXT_COUNT   LW0080_CTRL_GR_INFO_INDEX_MAX_PER_ENGINE_SUBCONTEXT_COUNT
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_SINGLETON_GPCS         LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_SINGLETON_GPCS
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GFXC_GPCS              LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_GFXC_GPCS
#define LW2080_CTRL_GR_INFO_INDEX_LITTER_NUM_GFXC_TPCS_PER_GFXC_GPC LW0080_CTRL_GR_INFO_INDEX_LITTER_NUM_GFXC_TPCS_PER_GFXC_GPC

/* When adding a new INDEX, please update INDEX_MAX and MAX_SIZE accordingly
 * NOTE: 0080 functionality is merged with 2080 functionality, so this max size
 * reflects that.
 */
#define LW2080_CTRL_GR_INFO_INDEX_MAX                               LW0080_CTRL_GR_INFO_INDEX_MAX
#define LW2080_CTRL_GR_INFO_MAX_SIZE                                LW0080_CTRL_GR_INFO_MAX_SIZE

/* valid SM version return values */

#define LW2080_CTRL_GR_INFO_SM_VERSION_NONE                         (0x00000000U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_1_05                         (0x00000105U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_1_1                          (0x00000110U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_1_2                          (0x00000120U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_1_3                          (0x00000130U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_1_4                          (0x00000140U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_1_5                          (0x00000150U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_2_0                          (0x00000200U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_2_1                          (0x00000210U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_2_2                          (0x00000220U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_3_0                          (0x00000300U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_3_1                          (0x00000310U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_3_2                          (0x00000320U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_3_3                          (0x00000330U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_3_5                          (0x00000350U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_3_6                          (0x00000360U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_3_8                          (0x00000380U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_3_9                          (0x00000390U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_4_0                          (0x00000400U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_5_0                          (0x00000500U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_5_02                         (0x00000502U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_5_03                         (0x00000503U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_6_0                          (0x00000600U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_6_01                         (0x00000601U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_6_02                         (0x00000602U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_7_0                          (0x00000700U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_7_01                         (0x00000701U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_7_02                         (0x00000702U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_7_03                         (0x00000703U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_7_05                         (0x00000705U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_8_02                         (0x00000802U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_8_06                         (0x00000806U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_8_07                         (0x00000807U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_8_08                         (0x00000808U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_8_09                         (0x00000809U)
#define LW2080_CTRL_GR_INFO_SM_VERSION_9_00                         (0x00000900U)

/* compatibility SM versions to match the official names in the ISA (e.g., SM5.2)  */
#define LW2080_CTRL_GR_INFO_SM_VERSION_5_2                          (LW2080_CTRL_GR_INFO_SM_VERSION_5_02)
#define LW2080_CTRL_GR_INFO_SM_VERSION_5_3                          (LW2080_CTRL_GR_INFO_SM_VERSION_5_03)
#define LW2080_CTRL_GR_INFO_SM_VERSION_6_1                          (LW2080_CTRL_GR_INFO_SM_VERSION_6_01)
#define LW2080_CTRL_GR_INFO_SM_VERSION_6_2                          (LW2080_CTRL_GR_INFO_SM_VERSION_6_02)
#define LW2080_CTRL_GR_INFO_SM_VERSION_7_1                          (LW2080_CTRL_GR_INFO_SM_VERSION_7_01)
#define LW2080_CTRL_GR_INFO_SM_VERSION_7_2                          (LW2080_CTRL_GR_INFO_SM_VERSION_7_02)
#define LW2080_CTRL_GR_INFO_SM_VERSION_7_3                          (LW2080_CTRL_GR_INFO_SM_VERSION_7_03)
#define LW2080_CTRL_GR_INFO_SM_VERSION_7_5                          (LW2080_CTRL_GR_INFO_SM_VERSION_7_05)
#define LW2080_CTRL_GR_INFO_SM_VERSION_8_2                          (LW2080_CTRL_GR_INFO_SM_VERSION_8_02)
#define LW2080_CTRL_GR_INFO_SM_VERSION_8_6                          (LW2080_CTRL_GR_INFO_SM_VERSION_8_06)
#define LW2080_CTRL_GR_INFO_SM_VERSION_8_7                          (LW2080_CTRL_GR_INFO_SM_VERSION_8_07)
#define LW2080_CTRL_GR_INFO_SM_VERSION_8_8                          (LW2080_CTRL_GR_INFO_SM_VERSION_8_08)
#define LW2080_CTRL_GR_INFO_SM_VERSION_8_9                          (LW2080_CTRL_GR_INFO_SM_VERSION_8_09)
#define LW2080_CTRL_GR_INFO_SM_VERSION_9_0                          (LW2080_CTRL_GR_INFO_SM_VERSION_9_00)


/**
 * LW2080_CTRL_CMD_GR_GET_INFO
 *
 * This command returns gr engine information for the associated GPU.
 * Requests to retrieve gr information use a list of one or more
 * LW2080_CTRL_GR_INFO structures.
 *
 *   grInfoListSize
 *     This field specifies the number of entries on the caller's
 *     grInfoList.
 *   grInfoList
 *     This field specifies a pointer in the caller's address space
 *     to the buffer into which the gr information is to be returned.
 *     This buffer must be at least as big as grInfoListSize multiplied
 *     by the size of the LW2080_CTRL_GR_INFO structure.
 *   grRouteInfo
 *     This parameter specifies the routing information used to
 *     disambiguate the target GR engine. When MIG is enabled, this
 *     is a mandatory parameter.
 */
#define LW2080_CTRL_CMD_GR_GET_INFO                                 (0x20801201U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_INFO_PARAMS_MESSAGE_ID (0x1U)

typedef struct LW2080_CTRL_GR_GET_INFO_PARAMS {
    LwU32 grInfoListSize;
    LW_DECLARE_ALIGNED(LwP64 grInfoList, 8);
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW2080_CTRL_GR_GET_INFO_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


/*
 * LW2080_CTRL_CMD_GR_GET_TESLA_TPC_INFO
 *
 * This API is deprecated and will always fail.  API definition remains until
 * all references are removed.
 */

#define LW2080_CTRL_CMD_GR_GET_TESLA_TPC_INFO (0x20801202U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x2" */

typedef struct LW2080_CTRL_GR_GET_TESLA_TPC_INFO_PARAMS {
    LwU32 tpcMask;
} LW2080_CTRL_GR_GET_TESLA_TPC_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_GR_GET_TESLA_SM_INFO
 *
 * This API is deprecated and will always fail.  API definition remains until
 * all references are removed.
 */

#define LW2080_CTRL_CMD_GR_GET_TESLA_SM_INFO (0x20801203U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x3" */

typedef struct LW2080_CTRL_GR_GET_TESLA_SM_INFO_PARAMS {
    LwU32 tpcId;
    LwU32 smMask;
} LW2080_CTRL_GR_GET_TESLA_SM_INFO_PARAMS;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW2080_CTRL_CMD_GR_CTXSW_ZLWLL_MODE
 *
 * This command is used to set the zlwll context switch mode for the specified
 * channel. A value of LW_ERR_NOT_SUPPORTED is returned if the
 * target channel does not support zlwll context switch mode changes.
 *
 *   hChannel
 *     This parameter specifies the channel handle of
 *     the channel that is to have it's zlwll context switch mode changed.
 *   hShareClient
 *     Support for sharing zlwll buffers across RM clients is no longer
 *     supported.  To maintain API compatibility, this field must match
 *     the hClient used in the control call.
 *   hShareChannel
 *     This parameter specifies the channel handle of
 *     the channel with which the zlwll context buffer is to be shared.  This
 *     parameter is valid when zlwllMode is set to SEPARATE_BUFFER.  This
 *     parameter should be set to the same value as hChannel if no
 *     sharing is intended.
 *   zlwllMode
 *     This parameter specifies the new zlwll context switch mode.
 *     Legal values for this parameter include:
 *       LW2080_CTRL_GR_SET_CTXSW_ZLWLL_MODE_GLOBAL
 *         This mode is the normal zlwll operation where it is not
 *         context switched and there is one set of globally shared
 *         zlwll memory and tables.  This mode is only supported as
 *         long as all channels use this mode.
 *       LW2080_CTRL_GR_SET_CTXSW_ZLWLL_MODE_NO_CTXSW
 *         This mode causes the zlwll tables to be reset on a context
 *         switch, but the zlwll buffer will not be saved/restored.
 *       LW2080_CTRL_GR_SET_CTXSW_ZLWLL_MODE_SEPARATE_BUFFER
 *         This mode will cause the zlwll buffers and tables to be
 *         saved/restored on context switches.  If a share channel
 *         ID is given (shareChID), then the 2 channels will share
 *         the zlwll context buffers.
 */
#define LW2080_CTRL_CMD_GR_CTXSW_ZLWLL_MODE (0x20801205U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x5" */

typedef struct LW2080_CTRL_GR_CTXSW_ZLWLL_MODE_PARAMS {
    LwHandle hChannel;
    LwHandle hShareClient;
    LwHandle hShareChannel;
    LwU32    zlwllMode;
} LW2080_CTRL_GR_CTXSW_ZLWLL_MODE_PARAMS;
/* valid zlwllMode values */
#define LW2080_CTRL_CTXSW_ZLWLL_MODE_GLOBAL          (0x00000000U)
#define LW2080_CTRL_CTXSW_ZLWLL_MODE_NO_CTXSW        (0x00000001U)
#define LW2080_CTRL_CTXSW_ZLWLL_MODE_SEPARATE_BUFFER (0x00000002U)

/**
 * LW2080_CTRL_CMD_GR_GET_ZLWLL_INFO
 *
 * This command is used to query the RM for zlwll information that the
 * driver will need to allocate and manage the zlwll regions.
 *
 *   widthAlignPixels
 *     This parameter returns the width alignment restrictions in pixels
 *     used to adjust a surface for proper aliquot coverage (typically
 *     #TPC's * 16).
 *
 *   heightAlignPixels
 *     This parameter returns the height alignment restrictions in pixels
 *     used to adjust a surface for proper aliquot coverage (typically 32).
 *
 *   pixelSquaresByAliquots
 *     This parameter returns the pixel area covered by an aliquot
 *     (typically #Zlwll_banks * 16 * 16).
 *
 *   aliquotTotal
 *     This parameter returns the total aliquot pool available in HW.
 *
 *   zlwllRegionByteMultiplier
 *     This parameter returns multiplier used to colwert aliquots in a region
 *     to the number of bytes required to save/restore them.
 *
 *   zlwllRegionHeaderSize
 *     This parameter returns the region header size which is required to be
 *     allocated and accounted for in any save/restore operation on a region.
 *
 *   zlwllSubregionHeaderSize
 *     This parameter returns the subregion header size which is required to be
 *     allocated and accounted for in any save/restore operation on a region.
 *
 *   subregionCount
 *     This parameter returns the subregion count.
 *
 *   subregionWidthAlignPixels
 *     This parameter returns the subregion width alignment restrictions in
 *     pixels used to adjust a surface for proper aliquot coverage
 *     (typically #TPC's * 16).
 *
 *   subregionHeightAlignPixels
 *     This parameter returns the subregion height alignment restrictions in
 *     pixels used to adjust a surface for proper aliquot coverage
 *     (typically 62).
 *
 *   The callee should compute the size of a zlwll region as follows.
 *     (numBytes = aliquots * zlwllRegionByteMultiplier +
 *                 zlwllRegionHeaderSize + zlwllSubregionHeaderSize)
 */
#define LW2080_CTRL_CMD_GR_GET_ZLWLL_INFO            (0x20801206U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_ZLWLL_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_ZLWLL_INFO_PARAMS_SUBREGION_SUPPORTED
#define LW2080_CTRL_GR_GET_ZLWLL_INFO_PARAMS_MESSAGE_ID (0x6U)

typedef struct LW2080_CTRL_GR_GET_ZLWLL_INFO_PARAMS {
    LwU32 widthAlignPixels;
    LwU32 heightAlignPixels;
    LwU32 pixelSquaresByAliquots;
    LwU32 aliquotTotal;
    LwU32 zlwllRegionByteMultiplier;
    LwU32 zlwllRegionHeaderSize;
    LwU32 zlwllSubregionHeaderSize;
    LwU32 subregionCount;
    LwU32 subregionWidthAlignPixels;
    LwU32 subregionHeightAlignPixels;
} LW2080_CTRL_GR_GET_ZLWLL_INFO_PARAMS;

/*
 * LW2080_CTRL_CMD_GR_CTXSW_PM_MODE
 *
 * This command is used to set the pm context switch mode for the specified
 * channel. A value of LW_ERR_NOT_SUPPORTED is returned if the
 * target channel does not support pm context switch mode changes.
 *
 *   hChannel
 *     This parameter specifies the channel handle of
 *     the channel that is to have its pm context switch mode changed.
 *   pmMode
 *     This parameter specifies the new pm context switch mode.
 *     Legal values for this parameter include:
 *       LW2080_CTRL_CTXSW_PM_MODE_NO_CTXSW
 *         This mode says that the pms are not to be context switched.
 *       LW2080_CTRL_CTXSW_PM_MODE_CTXSW
 *         This mode says that the pms in Mode-B are to be context switched.
 *       LW2080_CTRL_CTXSW_PM_MODE_STREAM_OUT_CTXSW
 *         This mode says that the pms in Mode-E (stream out) are to be context switched.
 *   grRouteInfo
 *     This parameter specifies the routing information used to
 *     disambiguate the target GR engine.
 */
#define LW2080_CTRL_CMD_GR_CTXSW_PM_MODE (0x20801207U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x7" */

typedef struct LW2080_CTRL_GR_CTXSW_PM_MODE_PARAMS {
    LwHandle hChannel;
    LwU32    pmMode;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW2080_CTRL_GR_CTXSW_PM_MODE_PARAMS;

/* valid pmMode values */
#define LW2080_CTRL_CTXSW_PM_MODE_NO_CTXSW         (0x00000000U)
#define LW2080_CTRL_CTXSW_PM_MODE_CTXSW            (0x00000001U)
#define LW2080_CTRL_CTXSW_PM_MODE_STREAM_OUT_CTXSW (0x00000002U)

/*
 * LW2080_CTRL_CMD_GR_CTXSW_ZLWLL_BIND
 *
 * This command is used to set the zlwll context switch mode and virtual address
 * for the specified channel. A value of LW_ERR_NOT_SUPPORTED is
 * returned if the target channel does not support zlwll context switch mode
 * changes.
 *
 *   hClient
 *     This parameter specifies the client handle of
 *     that owns the zlwll context buffer.
 *   hChannel
 *     This parameter specifies the channel handle of
 *     the channel that is to have its zlwll context switch mode changed.
 *   vMemPtr
 *     This parameter specifies the 64 bit virtual address
 *     for the allocated zlwll context buffer.
 *   zlwllMode
 *     This parameter specifies the new zlwll context switch mode.
 *     Legal values for this parameter include:
 *       LW2080_CTRL_GR_SET_CTXSW_ZLWLL_MODE_GLOBAL
 *         This mode is the normal zlwll operation where it is not
 *         context switched and there is one set of globally shared
 *         zlwll memory and tables.  This mode is only supported as
 *         long as all channels use this mode.
 *       LW2080_CTRL_GR_SET_CTXSW_ZLWLL_MODE_NO_CTXSW
 *         This mode causes the zlwll tables to be reset on a context
 *         switch, but the zlwll buffer will not be saved/restored.
 *       LW2080_CTRL_GR_SET_CTXSW_ZLWLL_MODE_SEPARATE_BUFFER
 *         This mode will cause the zlwll buffers and tables to be
 *         saved/restored on context switches.  If a share channel
 *         ID is given (shareChID), then the 2 channels will share
 *         the zlwll context buffers.
 */
#define LW2080_CTRL_CMD_GR_CTXSW_ZLWLL_BIND        (0x20801208U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x8" */

typedef struct LW2080_CTRL_GR_CTXSW_ZLWLL_BIND_PARAMS {
    LwHandle hClient;
    LwHandle hChannel;
    LW_DECLARE_ALIGNED(LwU64 vMemPtr, 8);
    LwU32    zlwllMode;
} LW2080_CTRL_GR_CTXSW_ZLWLL_BIND_PARAMS;
/* valid zlwllMode values same as above LW2080_CTRL_CTXSW_ZLWLL_MODE */

/*
 * LW2080_CTRL_CMD_GR_CTXSW_PM_BIND
 *
 * This command is used to set the PM context switch mode and virtual address
 * for the specified channel. A value of LW_ERR_NOT_SUPPORTED is
 * returned if the target channel does not support PM context switch mode
 * changes.
 *
 *   hClient
 *     This parameter specifies the client handle of
 *     that owns the PM context buffer.
 *   hChannel
 *     This parameter specifies the channel handle of
 *     the channel that is to have its PM context switch mode changed.
 *   vMemPtr
 *     This parameter specifies the 64 bit virtual address
 *     for the allocated PM context buffer.
 *   pmMode
 *     This parameter specifies the new PM context switch mode.
 *     Legal values for this parameter include:
 *       LW2080_CTRL_GR_SET_CTXSW_PM_MODE_NO_CTXSW
 *         This mode says that the pms are not to be context switched
 *       LW2080_CTRL_GR_SET_CTXSW_PM_MODE_CTXSW
 *         This mode says that the pms are to be context switched
 *   grRouteInfo
 *     This parameter specifies the routing information used to
 *     disambiguate the target GR engine.
 */
#define LW2080_CTRL_CMD_GR_CTXSW_PM_BIND (0x20801209U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x9" */

typedef struct LW2080_CTRL_GR_CTXSW_PM_BIND_PARAMS {
    LwHandle hClient;
    LwHandle hChannel;
    LW_DECLARE_ALIGNED(LwU64 vMemPtr, 8);
    LwU32    pmMode;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW2080_CTRL_GR_CTXSW_PM_BIND_PARAMS;
/* valid pmMode values same as above LW2080_CTRL_CTXSW_PM_MODE */

/*
 * LW2080_CTRL_CMD_GR_SET_GPC_TILE_MAP
 *
 * Send a list of values used to describe GPC/TPC tile mapping tables.
 *
 *   mapValueCount
 *     This field specifies the number of actual map entries.  This count
 *     should equal the number of TPCs in the system.
 *   mapValues
 *     This field is a pointer to a buffer of LwU08 values representing map
 *     data.
 *   grRouteInfo
 *     This parameter specifies the routing information used to
 *     disambiguate the target GR engine.
 */
#define LW2080_CTRL_GR_SET_GPC_TILE_MAP_MAX_VALUES 128U
#define LW2080_CTRL_CMD_GR_SET_GPC_TILE_MAP        (0x2080120aU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0xA" */

typedef struct LW2080_CTRL_GR_SET_GPC_TILE_MAP_PARAMS {
    LwU32 mapValueCount;
    LwU8  mapValues[LW2080_CTRL_GR_SET_GPC_TILE_MAP_MAX_VALUES];
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW2080_CTRL_GR_SET_GPC_TILE_MAP_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)


/*
 * LW2080_CTRL_CMD_GR_GET_TILE_HEIGHT
 *
 * This API is deprecated and will always fail.  API definition remains until
 * all references are removed.
 */
#define LW2080_CTRL_CMD_GR_GET_TILE_HEIGHT (0x2080120bU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0xB" */

typedef struct LW2080_CTRL_GR_GET_TILE_HEIGHT_PARAMS {
    LwU32 reqWidth;
    LwU32 reqHeight;
    LwU32 optHeight;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW2080_CTRL_GR_GET_TILE_HEIGHT_PARAMS;

/*
 * LW2080_CTRL_CMD_GR_SET_ZLWLL_BIT_WAR
 *
 * This API is deprecated and will always fail.  API definition remains until
 * all references are removed.
 */
#define LW2080_CTRL_CMD_GR_SET_ZLWLL_BIT_WAR (0x2080120lw) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0xC" */

typedef struct LW2080_CTRL_GR_SET_ZLWLL_BIT_WAR_PARAMS {
    LwU32    index;
    LwHandle hClientTarget;
    LwHandle hChannelTarget;
} LW2080_CTRL_GR_SET_ZLWLL_BIT_WAR_PARAMS;

/*
 * LW2080_CTRL_CMD_GR_SET_CHANNEL_VPR
 *
 * This command is used to tell GR that a channel is being used in VPR mode
 *
 *   hClient
 *     This parameter specifies the client handle of
 *     that owns the zlwll context buffer.
 *   hChannel
 *     This parameter specifies the channel handle of
 *     the channel that is to be set into vpr mode.
 *   grRouteInfo
 *     This parameter specifies the routing information used to
 *     disambiguate the target GR engine.
 */
#define LW2080_CTRL_CMD_GR_SET_CHANNEL_VPR (0x2080120dU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_SET_CHANNEL_VPR_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_SET_CHANNEL_VPR_PARAMS_MESSAGE_ID (0xDU)

typedef struct LW2080_CTRL_GR_SET_CHANNEL_VPR_PARAMS {
    LwHandle hClient;
    LwHandle hChannel;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW2080_CTRL_GR_SET_CHANNEL_VPR_PARAMS;

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



/*
 * LW2080_CTRL_CMD_GR_CTXSW_SMPC_MODE
 *
 * This command is used to set the SMPC context switch mode for the specified
 * channel or channel group (TSG). A value of LW_ERR_NOT_SUPPORTED
 * is returned if the target channel/TSG does not support SMPC context switch
 * mode changes.  If a channel is part of a TSG, the user must send in the TSG
 * handle and not an individual channel handle, an error will be returned if a
 * channel handle is used in this case.
 *
 * SMPC = SM Performance Counters
 *
 *   hChannel
 *     This parameter specifies the channel or channel group (TSG) handle
 *     that is to have its SMPC context switch mode changed.
 *     If this parameter is set to 0, then the mode below applies to all current
 *     and future channels (i.e. we will be enabling/disabling global mode)
 *   smpcMode
 *     This parameter specifies the new SMPC context switch mode.
 *     Legal values for this parameter include:
 *       LW2080_CTRL_GR_SET_CTXSW_SMPC_MODE_NO_CTXSW
 *         This mode says that the SMPC data is not to be context switched.
 *       LW2080_CTRL_GR_SET_CTXSW_SMPC_MODE_CTXSW
 *         This mode says that the SMPC data is to be context switched.
 *   grRouteInfo
 *     This parameter specifies the routing information used to
 *     disambiguate the target GR engine.
 */
#define LW2080_CTRL_CMD_GR_CTXSW_SMPC_MODE (0x2080120eU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0xE" */

typedef struct LW2080_CTRL_GR_CTXSW_SMPC_MODE_PARAMS {
    LwHandle hChannel;
    LwU32    smpcMode;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW2080_CTRL_GR_CTXSW_SMPC_MODE_PARAMS;

/* valid smpcMode values */
#define LW2080_CTRL_CTXSW_SMPC_MODE_NO_CTXSW                   (0x00000000U)
#define LW2080_CTRL_CTXSW_SMPC_MODE_CTXSW                      (0x00000001U)

/*
 * LW2080_CTRL_CMD_GR_GET_SM_TO_GPC_TPC_MAPPINGS
 *
 * This command returns an array of the mappings between SMs and GPC/TPCs.
 *
 *   smId
 *     An array of the mappings between SMs and GPC/TPCs.
 *   smCount
 *     Returns the number of valid mappings in the array.
 *   grRouteInfo
 *     This parameter specifies the routing information used to
 *     disambiguate the target GR engine.
 */
#define LW2080_CTRL_CMD_GR_GET_SM_TO_GPC_TPC_MAPPINGS          (0x2080120fU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_SM_TO_GPC_TPC_MAPPINGS_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_SM_TO_GPC_TPC_MAPPINGS_MAX_SM_COUNT 144U
#define LW2080_CTRL_GR_GET_SM_TO_GPC_TPC_MAPPINGS_PARAMS_MESSAGE_ID (0xFU)

typedef struct LW2080_CTRL_GR_GET_SM_TO_GPC_TPC_MAPPINGS_PARAMS {
    struct {
        LwU32 gpcId;
        LwU32 tpcId;
    } smId[LW2080_CTRL_GR_GET_SM_TO_GPC_TPC_MAPPINGS_MAX_SM_COUNT];
    LwU32 smCount;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW2080_CTRL_GR_GET_SM_TO_GPC_TPC_MAPPINGS_PARAMS;

/*
 * LW2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE
 *
 * This command is used to set the preemption context switch mode for the specified
 * channel. A value of LW_ERR_NOT_SUPPORTED is returned if the
 * target channel does not support preemption context switch mode changes.
 *
 *   flags
 *     This field specifies flags for the preemption mode changes.
 *     These flags can tell callee which mode is valid in the call
 *     since we handle graphics and/or compute
 *   hChannel
 *     This parameter specifies the channel handle of the channel
 *     that is to have it's preemption context switch mode set.
 *   gfxpPreemptMode
 *     This parameter specifies the new Graphics preemption context switch
 *     mode. Legal values for this parameter include:
 *       LW2080_CTRL_SET_CTXSW_PREEMPTION_MODE_GFX_WFI
 *         This mode is the normal wait-for-idle context switch mode.
 *       LW2080_CTRL_SET_CTXSW_PREEMPTION_MODE_GFX_GFXP
 *         This mode causes the graphics engine to allow preempting the
 *         channel mid-triangle.
 *       LW2080_CTRL_SET_CTXSW_PREEMPTION_MODE_GFX_GFXP_POOL
 *         This mode causes the graphics engine to use a shared pool of buffers
 *         to support GfxP with lower memory overhead
 *   cilpPreemptMode
 *     This parameter specifies the new Compute preemption context switch
 *     mode. Legal values for this parameter include:
 *       LW2080_CTRL_SET_CTXSW_PREEMPTION_MODE_COMPUTE_WFI
 *         This mode is the normal wait-for-idle context switch mode.
 *       LW2080_CTRL_SET_CTXSW_PREEMPTION_MODE_COMPUTE_CTA
 *         This mode causes the compute engine to allow preempting the channel
 *         at the instruction level.
 *       LW2080_CTRL_SET_CTXSW_PREEMPTION_MODE_COMPUTE_CILP
 *         This mode causes the compute engine to allow preempting the channel
 *         at the instruction level.
 *   grRouteInfo
 *     This parameter specifies the routing information used to
 *     disambiguate the target GR engine.
 */
#define LW2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE (0x20801210U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x10" */

typedef struct LW2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_PARAMS {
    LwU32    flags;
    LwHandle hChannel;
    LwU32    gfxpPreemptMode;
    LwU32    cilpPreemptMode;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_PARAMS;

/* valid preemption flags */
#define LW2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_FLAGS_CILP                0:0
#define LW2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_FLAGS_CILP_IGNORE (0x00000000U)
#define LW2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_FLAGS_CILP_SET    (0x00000001U)
#define LW2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_FLAGS_GFXP                1:1
#define LW2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_FLAGS_GFXP_IGNORE (0x00000000U)
#define LW2080_CTRL_GR_SET_CTXSW_PREEMPTION_MODE_FLAGS_GFXP_SET    (0x00000001U)

/* valid Graphics mode values */
#define LW2080_CTRL_SET_CTXSW_PREEMPTION_MODE_GFX_WFI              (0x00000000U)
#define LW2080_CTRL_SET_CTXSW_PREEMPTION_MODE_GFX_GFXP             (0x00000001U)
#define LW2080_CTRL_SET_CTXSW_PREEMPTION_MODE_GFX_GFXP_POOL        (0x00000002U)

/* valid Compute mode values */
#define LW2080_CTRL_SET_CTXSW_PREEMPTION_MODE_COMPUTE_WFI          (0x00000000U)
#define LW2080_CTRL_SET_CTXSW_PREEMPTION_MODE_COMPUTE_CTA          (0x00000001U)
#define LW2080_CTRL_SET_CTXSW_PREEMPTION_MODE_COMPUTE_CILP         (0x00000002U)

/* valid preemption buffers */
typedef enum LW2080_CTRL_CMD_GR_CTXSW_PREEMPTION_BIND_BUFFERS {
    LW2080_CTRL_CMD_GR_CTXSW_PREEMPTION_BIND_BUFFERS_MAIN = 0,
    LW2080_CTRL_CMD_GR_CTXSW_PREEMPTION_BIND_BUFFERS_SPILL = 1,
    LW2080_CTRL_CMD_GR_CTXSW_PREEMPTION_BIND_BUFFERS_PAGEPOOL = 2,
    LW2080_CTRL_CMD_GR_CTXSW_PREEMPTION_BIND_BUFFERS_BETACB = 3,
    LW2080_CTRL_CMD_GR_CTXSW_PREEMPTION_BIND_BUFFERS_RTV = 4,
    LW2080_CTRL_CMD_GR_CTXSW_PREEMPTION_BIND_BUFFERS_CONTEXT_POOL = 5,
    LW2080_CTRL_CMD_GR_CTXSW_PREEMPTION_BIND_BUFFERS_CONTEXT_POOL_CONTROL = 6,
    LW2080_CTRL_CMD_GR_CTXSW_PREEMPTION_BIND_BUFFERS_CONTEXT_POOL_CONTROL_CPU = 7,
    LW2080_CTRL_CMD_GR_CTXSW_PREEMPTION_BIND_BUFFERS_END = 8,
} LW2080_CTRL_CMD_GR_CTXSW_PREEMPTION_BIND_BUFFERS;

/*
 * LW2080_CTRL_CMD_GR_CTXSW_PREEMPTION_BIND
 *
 * This command is used to set the preemption context switch mode and virtual
 * addresses of the preemption buffers for the specified channel. A value of
 * LW_ERR_NOT_SUPPORTED is returned if the target channel does not
 * support preemption context switch mode changes.
 *
 *   flags
 *     This field specifies flags for the preemption mode changes.
 *     These flags can tell callee which mode is valid in the call
 *     since we handle graphics and/or compute
 *   hClient
 *     This parameter specifies the client handle of
 *     that owns the preemption context buffer.
 *   hChannel
 *     This parameter specifies the channel handle of the channel
 *     that is to have its preemption context switch mode set.
 *   vMemPtr
 *     This parameter specifies the 64 bit virtual address
 *     for the allocated preemption context buffer.
 *   gfxpPreemptMode
 *     This parameter specifies the new Graphics preemption context switch
 *     mode. Legal values for this parameter include:
 *       LW2080_CTRL_CTXSW_PREEMPTION_MODE_GFX_WFI
 *         This mode is the normal wait-for-idle context switch mode.
 *       LW2080_CTRL_CTXSW_PREEMPTION_MODE_GFX_GFXP
 *         This mode causes the graphics engine to allow preempting the
 *         channel mid-triangle.
 *   cilpPreemptMode
 *     This parameter specifies the new Compute preemption context switch
 *     mode. Legal values for this parameter include:
 *       LW2080_CTRL_CTXSW_PREEMPTION_MODE_COMPUTE_WFI
 *         This mode is the normal wait-for-idle context switch mode.
 *       LW2080_CTRL_CTXSW_PREEMPTION_MODE_COMPUTE_CTA
 *         This mode causes the compute engine to allow preempting the channel
 *         at the instruction level.
 *       LW2080_CTRL_CTXSW_PREEMPTION_MODE_COMPUTE_CILP
 *         This mode causes the compute engine to allow preempting the channel
 *         at the instruction level.
 *   grRouteInfo
 *     This parameter specifies the routing information used to
 *     disambiguate the target GR engine.
 */
#define LW2080_CTRL_CMD_GR_CTXSW_PREEMPTION_BIND (0x20801211U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x11" */

typedef struct LW2080_CTRL_GR_CTXSW_PREEMPTION_BIND_PARAMS {
    LwU32    flags;
    LwHandle hClient;
    LwHandle hChannel;
    LW_DECLARE_ALIGNED(LwU64 vMemPtrs[LW2080_CTRL_CMD_GR_CTXSW_PREEMPTION_BIND_BUFFERS_END], 8);
    LwU32    gfxpPreemptMode;
    LwU32    cilpPreemptMode;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW2080_CTRL_GR_CTXSW_PREEMPTION_BIND_PARAMS;
/* valid mode and flag values same as above LW2080_CTRL_SET_CTXSW_PREEMPTION_MODE */

/*
 * LW2080_CTRL_CMD_GR_PC_SAMPLING_MODE
 *
 * This command is used to apply the WAR for PC sampling to avoid hang in
 * multi-ctx scenario.
 *
 *   hChannel
 *     This parameter specifies the channel or channel group (TSG) handle
 *     that is to have its PC Sampling mode changed.
 *   samplingMode
 *     This parameter specifies whether sampling is turned ON or OFF.
 *     Legal values for this parameter include:
 *       LW2080_CTRL_GR_SET_PC_SAMPLING_MODE_DISABLED
 *         This mode says that PC sampling is disabled for current context.
 *       LW2080_CTRL_GR_SET_PC_SAMPLING_MODE_ENABLED
 *         This mode says that PC sampling is disabled for current context.
 *   grRouteInfo
 *     This parameter specifies the routing information used to
 *     disambiguate the target GR engine.
 */
#define LW2080_CTRL_CMD_GR_PC_SAMPLING_MODE (0x20801212U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x12" */

typedef struct LW2080_CTRL_GR_PC_SAMPLING_MODE_PARAMS {
    LwHandle hChannel;
    LwU32    samplingMode;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW2080_CTRL_GR_PC_SAMPLING_MODE_PARAMS;

/* valid samplingMode values */
#define LW2080_CTRL_PC_SAMPLING_MODE_DISABLED (0x00000000U)
#define LW2080_CTRL_PC_SAMPLING_MODE_ENABLED  (0x00000001U)

/*
 * LW2080_CTRL_CMD_GR_GET_ROP_INFO
 *
 * Gets information about ROPs including the ROP unit count and information
 * about ROP operations per clock.
 *
 *   ropUnitCount
 *     The count of active ROP units.
 *   ropOperationsFactor.
 *     The number of ROP operations per clock for a single ROP unit.
 *   ropOperationsCount
 *     The number of ROP operations per clock across all active ROP units.
 */
#define LW2080_CTRL_CMD_GR_GET_ROP_INFO       (0x20801213U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_ROP_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_ROP_INFO_PARAMS_MESSAGE_ID (0x13U)

typedef struct LW2080_CTRL_GR_GET_ROP_INFO_PARAMS {
    LwU32 ropUnitCount;
    LwU32 ropOperationsFactor;
    LwU32 ropOperationsCount;
} LW2080_CTRL_GR_GET_ROP_INFO_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW2080_CTRL_CMD_GR_SET_DELAY_CILP_PREEMPT
 *
 * This command is used to set the delay before a CILP preemption is forced.
 * This timeout defines the duration (in sysclk for rtl/silicon/emulation,
 * lwsim "lwclk" for fmodel) CWD waits after receiving PREEMPT_TYPE=_CILP
 * before sending down PREEMPT_SM to downstream. If all CTAs complete within
 * the cycle count, CWD defaults to PREEMPT_TYPE=_CTA. Otherwise CWD issues
 * PREEMPT_SM and waits for preempt_complete from all GPMs.
 *
 *   hChannel
 *     This parameter specifies the channel or channel group (TSG) handle
 *     that is to have the timeout changed.
 *   timeout
 *     This parameter specifies the timeout value.
 */
#define LW2080_CTRL_CMD_GR_SET_DELAY_CILP_PREEMPT (0x20801214U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x14" */

typedef struct LW2080_CTRL_GR_SET_DELAY_CILP_PREEMPT_PARAMS {
    LwHandle hChannel;
    LwU32    timeout;
} LW2080_CTRL_GR_SET_DELAY_CILP_PREEMPT_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW2080_CTRL_CMD_GR_GET_CTXSW_STATS
 *
 * This command is used to get the context switch statistics.  The user can
 * also add a flag to tell RM to reset the stats counters back to 0.
 *
 *   hChannel
 *     This parameter specifies the channel or channel group (TSG) handle
 *     that is to have the stats returned.  Note, must be the TSG handle if
 *     channel is part of a TSG.
 *   flags
 *     This parameter specifies processing flags. See possible flags below.
 *   saveCnt
 *     This parameter returns the number of saves on the channel.
 *   restoreCnt
 *     This parameter returns the number of restores on the channel.
 *   wfiSaveCnt
 *     This parameter returns the number of WFI saves on the channel.
 *   ctaSaveCnt
 *     This parameter returns the number of CTA saves on the channel.
 *   cilpSaveCnt
 *     This parameter returns the number of CILP saves on the channel.
 *   gfxpSaveCnt
 *     This parameter returns the number of GfxP saves on the channel.
 */
#define LW2080_CTRL_CMD_GR_GET_CTXSW_STATS (0x20801215U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x15" */

typedef struct LW2080_CTRL_GR_GET_CTXSW_STATS_PARAMS {
    LwHandle hChannel;
    LwU32    flags;
    LwU32    saveCnt;
    LwU32    restoreCnt;
    LwU32    wfiSaveCnt;
    LwU32    ctaSaveCnt;
    LwU32    cilpSaveCnt;
    LwU32    gfxpSaveCnt;
} LW2080_CTRL_GR_GET_CTXSW_STATS_PARAMS;
/* valid GET_CTXSW_STATS flags settings */
#define LW2080_CTRL_GR_GET_CTXSW_STATS_FLAGS_RESET                      0:0
#define LW2080_CTRL_GR_GET_CTXSW_STATS_FLAGS_RESET_FALSE (0x00000000U)
#define LW2080_CTRL_GR_GET_CTXSW_STATS_FLAGS_RESET_TRUE  (0x00000001U)

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW2080_CTRL_CMD_GR_SET_GFXP_TIMEOUT
 *
 * This command is used to set the timeout before a GFXP preemption is forced.
 * This timeout defines the duration (in sysclk for rtl/silicon/emulation,
 * lwsim "lwclk" for fmodel) FE will WFI before sending a GFXP request
 * downstream.
 *
 *   hChannel
 *     This parameter specifies the channel or channel group (TSG) handle
 *     that is to have the timeout changed.
 *   timeout
 *     This parameter specifies the timeout value.
 */
#define LW2080_CTRL_CMD_GR_SET_GFXP_TIMEOUT              (0x20801216U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x16" */

typedef struct LW2080_CTRL_GR_SET_GFXP_TIMEOUT_PARAMS {
    LwHandle hChannel;
    LwU32    timeout;
} LW2080_CTRL_GR_SET_GFXP_TIMEOUT_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW2080_CTRL_CMD_GR_GET_CTX_BUFFER_SIZE
 *
 * This command provides the size, alignment of all context buffers including global and
 * local context buffers which has been created & will be mapped on a context
 *
 *   hChannel [IN]
 *     This parameter specifies the channel or channel group (TSG) handle
 *   totalBufferSize [OUT]
 *     This parameter returns the total context buffers size.
 */
#define LW2080_CTRL_CMD_GR_GET_CTX_BUFFER_SIZE (0x20801218U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_CTX_BUFFER_SIZE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_CTX_BUFFER_SIZE_PARAMS_MESSAGE_ID (0x18U)

typedef struct LW2080_CTRL_GR_GET_CTX_BUFFER_SIZE_PARAMS {
    LwHandle hChannel;
    LW_DECLARE_ALIGNED(LwU64 totalBufferSize, 8);
} LW2080_CTRL_GR_GET_CTX_BUFFER_SIZE_PARAMS;

/*
 * LW2080_CTRL_GR_CTX_BUFFER_INFO
 *   alignment
 *     Specifies the alignment requirement for each context buffer
 *   size
 *     Aligned size of context buffer
 *   bufferHandle [deprecated]
 *     Opaque pointer to memdesc. Used by kernel clients for tracking purpose only.
 *   pageCount
 *     allocation size in the form of pageCount
 *   physAddr
 *     Physical address of the buffer first page
 *   bufferType
 *     LW2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID type of this buffer
 *   aperture
 *     allocation aperture. Could be SYSMEM, VIDMEM, UNKNOWN
 *   kind
 *     PTE kind of this allocation.
 *   pageSize
 *     Page size of the buffer.
 *   bIsContigous
 *     States if physical allocation for this buffer is contiguous. PageSize will
 *     have no meaning if this flag is set.
 *   bGlobalBuffer
 *     States if a defined buffer is global as global buffers need to be mapped
 *     only once in TSG.
 *   bLocalBuffer
 *     States if a buffer is local to a channel.
 *   bDeviceDescendant
 *     TRUE if the allocation is a constructed under a Device or Subdevice.
 *   uuid
 *     SHA1 UUID of the Device or Subdevice. Valid when deviceDescendant is TRUE.
 */
typedef struct LW2080_CTRL_GR_CTX_BUFFER_INFO {
    LW_DECLARE_ALIGNED(LwU64 alignment, 8);
    LW_DECLARE_ALIGNED(LwU64 size, 8);
    LW_DECLARE_ALIGNED(LwP64 bufferHandle, 8);
    LW_DECLARE_ALIGNED(LwU64 pageCount, 8);
    LW_DECLARE_ALIGNED(LwU64 physAddr, 8);
    LwU32  bufferType;
    LwU32  aperture;
    LwU32  kind;
    LwU32  pageSize;
    LwBool bIsContigous;
    LwBool bGlobalBuffer;
    LwBool bLocalBuffer;
    LwBool bDeviceDescendant;
    LwU8   uuid[16];
} LW2080_CTRL_GR_CTX_BUFFER_INFO;
typedef struct LW2080_CTRL_GR_CTX_BUFFER_INFO *PLW2080_CTRL_GR_CTX_BUFFER_INFO;

#define LW2080_CTRL_GR_MAX_CTX_BUFFER_COUNT    64U

/*
 * LW2080_CTRL_CMD_GR_GET_CTX_BUFFER_INFO
 *
 * This command provides the size, alignment of all context buffers including global and
 * local context buffers which has been created & will be mapped on a context.
 * If the client ilwoking the command is a kernel client, the buffers are retained.
 *
 *   hUserClient [IN]
 *     This parameter specifies the client handle that owns this channel.
 *   hChannel [IN]
 *     This parameter specifies the channel or channel group (TSG) handle
 *   bufferCount [OUT]
 *     This parameter specifies the number of entries in ctxBufferInfo filled
 *     by the command.
 *   ctxBufferInfo [OUT]
 *     Array of context buffer info containing alignment, size etc.
 */
#define LW2080_CTRL_CMD_GR_GET_CTX_BUFFER_INFO (0x20801219U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_CTX_BUFFER_INFO_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_CTX_BUFFER_INFO_PARAMS_MESSAGE_ID (0x19U)

typedef struct LW2080_CTRL_GR_GET_CTX_BUFFER_INFO_PARAMS {
    LwHandle hUserClient;
    LwHandle hChannel;
    LwU32    bufferCount;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_CTX_BUFFER_INFO ctxBufferInfo[LW2080_CTRL_GR_MAX_CTX_BUFFER_COUNT], 8);
} LW2080_CTRL_GR_GET_CTX_BUFFER_INFO_PARAMS;

// Aperture flags
#define LW2080_CTRL_GR_CTX_BUFFER_INFO_APERTURE_UNKNWON ADDR_UNKNOWN
#define LW2080_CTRL_GR_CTX_BUFFER_INFO_APERTURE_SYSMEM ADDR_SYSMEM
#define LW2080_CTRL_GR_CTX_BUFFER_INFO_APERTURE_FBMEM ADDR_FBMEM

/*
 * LW2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER
 *     This command returns the global logical ordering of SM w.r.t GPCs/TPCs.
 *
 * LW2080_CTRL_GR_GET_GLOBAL_SM_ORDER_PARAMS
 *     This structure holds the TPC/SM ordering info.
 *
 *     gpcId
 *         Logical GPC Id.
 *         This is the ordering of enabled GPCs post floor sweeping.
 *         The GPCs are numbered from 0 to N-1, where N is the enabled GPC count.
 *
 *     localTpcId
 *         Local Logical TPC Id.
 *         This is the ordering of enabled TPCs within a GPC post floor sweeping.
 *         This ID is used in conjunction with the gpcId.
 *         The TPCs are numbered from 0 to N-1, where N is the enabled TPC count for the given GPC.
 *
 *     localSmId
 *         Local Logical SM Id.
 *         This is the ordering of enabled SMs within a TPC post floor sweeping.
 *         This ID is used in conjunction with the localTpcId.
 *         The SMs are numbered from 0 to N-1, where N is the enabled SM count for the given TPC.
 *
 *     globalTpcId
 *         Global Logical TPC Id.
 *         This is the ordering of all enabled TPCs in the GPU post floor sweeping.
 *         The TPCs are numbered from 0 to N-1, where N is the enabled TPC count across all GPCs
 *
 *     globalSmId
 *         Global Logical SM Id array.
 *         This is the global ordering of all enabled SMs in the GPU post floor sweeping.
 *         The SMs are numbered from 0 to N-1, where N is the enabled SM count across all GPCs.
 *
 *     virtualGpcId
 *         Virtual GPC Id.
 *         This is the ordering of enabled GPCs post floor sweeping (ordered in increasing
 *         number of TPC counts) The GPCs are numbered from 0 to N-1, where N is the
 *         enabled GPC count and 8-23 for singleton TPC holders.
 *
 *     migratableTpcId
 *         Migratable TPC Id.
 *         This is the same as the Local Tpc Id for virtual GPC 0-8 (true physical gpcs) and 0 for
 *         virtual gpcs 8-23 that represent singleton tpcs.
 *
 *     numSm
 *         Enabled SM count across all GPCs.
 *         This represent the valid entries in the globalSmId array
 *
 *     numTpc
 *         Enabled TPC count across all GPCs.
 *
 *     grRouteInfo
 *         This parameter specifies the routing information used to
 *         disambiguate the target GR engine.
 *
 */
#define LW2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER              (0x2080121bU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_GLOBAL_SM_ORDER_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER_MAX_SM_COUNT 512U

#define LW2080_CTRL_GR_GET_GLOBAL_SM_ORDER_PARAMS_MESSAGE_ID (0x1BU)

typedef struct LW2080_CTRL_GR_GET_GLOBAL_SM_ORDER_PARAMS {
    struct {
        LwU16 gpcId;
        LwU16 localTpcId;
        LwU16 localSmId;
        LwU16 globalTpcId;
        LwU16 virtualGpcId;
        LwU16 migratableTpcId;
    } globalSmId[LW2080_CTRL_CMD_GR_GET_GLOBAL_SM_ORDER_MAX_SM_COUNT];

    LwU16 numSm;
    LwU16 numTpc;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW2080_CTRL_GR_GET_GLOBAL_SM_ORDER_PARAMS;

/*
* LW2080_CTRL_CMD_GR_GET_LWRRENT_RESIDENT_CHANNEL
*
* This command gives current resident channel on GR engine
*
*   chID [OUT]
*       RM returns current resident channel on GR engine
*   grRouteInfo [IN]
*       This parameter specifies the routing information used to
*       disambiguate the target GR engine.
*/
#define LW2080_CTRL_CMD_GR_GET_LWRRENT_RESIDENT_CHANNEL (0x2080121lw) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x1C" */

typedef struct LW2080_CTRL_CMD_GR_GET_LWRRENT_RESIDENT_CHANNEL_PARAMS {
    LwU32 chID;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
} LW2080_CTRL_CMD_GR_GET_LWRRENT_RESIDENT_CHANNEL_PARAMS;

/*
 * LW2080_CTRL_CMD_GR_GET_VAT_ALARM_DATA
 *
 * This command provides the _VAT_ALARM data i.e. error and warning, counter and
 * timestamps along with max GPC and TPC per GPC count.
 *
 *   smVatAlarm [OUT]
 *     VAT Alarm data array per SM containing per GPC per TPC, counter and
 *      timestamp values for error and warning alarms.
 *   maxGpcCount [OUT]
 *     This parameter returns max GPC count.
 *   maxTpcPerGpcCount [OUT]
 *     This parameter returns the max TPC per GPC count.
 */
#define LW2080_CTRL_CMD_GR_GET_VAT_ALARM_MAX_GPC_COUNT         10U
#define LW2080_CTRL_CMD_GR_GET_VAT_ALARM_MAX_TPC_PER_GPC_COUNT 10U

#define LW2080_CTRL_CMD_GR_GET_VAT_ALARM_DATA                  (0x2080121dU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x1D" */

typedef struct LW2080_CTRL_GR_VAT_ALARM_DATA_PER_TPC {
    LW_DECLARE_ALIGNED(LwU64 errorCounter, 8);
    LW_DECLARE_ALIGNED(LwU64 errorTimestamp, 8);
    LW_DECLARE_ALIGNED(LwU64 warningCounter, 8);
    LW_DECLARE_ALIGNED(LwU64 warningTimestamp, 8);
} LW2080_CTRL_GR_VAT_ALARM_DATA_PER_TPC;

typedef struct LW2080_CTRL_GR_VAT_ALARM_DATA_PER_GPC {
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_VAT_ALARM_DATA_PER_TPC tpc[LW2080_CTRL_CMD_GR_GET_VAT_ALARM_MAX_TPC_PER_GPC_COUNT], 8);
} LW2080_CTRL_GR_VAT_ALARM_DATA_PER_GPC;

typedef struct LW2080_CTRL_GR_VAT_ALARM_DATA {
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_VAT_ALARM_DATA_PER_GPC gpc[LW2080_CTRL_CMD_GR_GET_VAT_ALARM_MAX_GPC_COUNT], 8);
} LW2080_CTRL_GR_VAT_ALARM_DATA;

typedef struct LW2080_CTRL_GR_GET_VAT_ALARM_DATA_PARAMS {
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_VAT_ALARM_DATA smVatAlarm, 8);
    LwU32 maxGpcCount;
    LwU32 maxTpcPerGpcCount;
} LW2080_CTRL_GR_GET_VAT_ALARM_DATA_PARAMS;
typedef struct LW2080_CTRL_GR_GET_VAT_ALARM_DATA_PARAMS *PLW2080_CTRL_GR_GET_VAT_ALARM_DATA_PARAMS;

/*
 * LW2080_CTRL_CMD_GR_GET_ATTRIBUTE_BUFFER_SIZE
 *
 * This command provides the size of GR attribute buffer.
 *
 *   attribBufferSize [OUT]
 *     This parameter returns the attribute buffer size.
 */
#define LW2080_CTRL_CMD_GR_GET_ATTRIBUTE_BUFFER_SIZE (0x2080121eU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_ATTRIBUTE_BUFFER_SIZE_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_ATTRIBUTE_BUFFER_SIZE_PARAMS_MESSAGE_ID (0x1EU)

typedef struct LW2080_CTRL_GR_GET_ATTRIBUTE_BUFFER_SIZE_PARAMS {
    LwU32 attribBufferSize;
} LW2080_CTRL_GR_GET_ATTRIBUTE_BUFFER_SIZE_PARAMS;

/*
 * LW2080_CTRL_CMD_GR_GFX_POOL_QUERY_SIZE
 *
 * This API queries size parameters for a request maximum graphics preemption
 * pool size.  It is only available to kernel callers
 *
 * LW2080_CTRL_GR_GFX_POOL_QUERY_SIZE_PARAMS
 *   struct to return the size parameters
 *
 *   maxSlots
 *     Input specifying the maximum number of slots, RM will callwlate the output
 *     parameters based on this.  Must be non-zero
 *   ctrlStructSize
 *     Output indicating the required size in bytes of the control structure to
 *     support a pool of maxSlots size.
 *   ctrlStructAlign
 *     Output indicating the required alignment of the control structure
 *   poolSize
 *     Output indicating the required size in bytes of the GfxP Pool.
 *   poolAlign
 *     Output indicating the required alignment of the GfxP Pool
 *   slotStride
 *     The number of bytes in each slot, i * slotStride gives the offset from the
 *     base of the pool to a given slot
 */
#define LW2080_CTRL_CMD_GR_GFX_POOL_QUERY_SIZE (0x2080121fU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x1F" */

typedef struct LW2080_CTRL_GR_GFX_POOL_QUERY_SIZE_PARAMS {
    LwU32 maxSlots;
    LwU32 slotStride;
    LW_DECLARE_ALIGNED(LwU64 ctrlStructSize, 8);
    LW_DECLARE_ALIGNED(LwU64 ctrlStructAlign, 8);
    LW_DECLARE_ALIGNED(LwU64 poolSize, 8);
    LW_DECLARE_ALIGNED(LwU64 poolAlign, 8);
} LW2080_CTRL_GR_GFX_POOL_QUERY_SIZE_PARAMS;

/*
 * LW2080_CTRL_CMD_GR_GFX_POOL_INITIALIZE
 *
 * This API takes a CPU pointer to a GFxP Pool Control Structure and does the
 * required onetime initialization.  It should be called once and only once
 * before a pool is used.  It is only accessible to kernel callers.
 *
 * LW2080_CTRL_GR_GFX_POOL_INITIALIZE_PARAMS
 *   struct to hand in the required info to RM
 *
 *   pControlStructure
 *     This input is the kernel CPU pointer to the control structure.
 */
#define LW2080_CTRL_CMD_GR_GFX_POOL_INITIALIZE (0x20801220U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x20" */

typedef struct LW2080_CTRL_GR_GFX_POOL_INITIALIZE_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pControlStructure, 8);
    LwU32 maxSlots;
} LW2080_CTRL_GR_GFX_POOL_INITIALIZE_PARAMS;

#define LW2080_CTRL_GR_GFX_POOL_MAX_SLOTS     64U

/*
 * LW2080_CTRL_CMD_GR_GFX_POOL_ADD_SLOTS
 *
 * This API adds a list of buffer slots to a given control structure.  It can
 * only be called when no channel using the given pool is running or may become
 * running for the duration of this call.  If more slots are added than there
 * is room for in the control structure the behavior is undefined. It is only
 * accessible to kernel callers.
 *
 * LW2080_CTRL_GR_GFX_POOL_ADD_SLOTS_PARAMS
 *
 *   pControlStructure
 *     This input is the kernel CPU pointer to the control structure
 *   numSlots
 *     This input indicates how many slots are being added and are contained in the slots parameter
 *   slots
 *     This input contains an array of the slots to be added to the control structure
 */
#define LW2080_CTRL_CMD_GR_GFX_POOL_ADD_SLOTS (0x20801221U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x21" */

typedef struct LW2080_CTRL_GR_GFX_POOL_ADD_SLOTS_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pControlStructure, 8);
    LwU32 numSlots;
    LwU32 slots[LW2080_CTRL_GR_GFX_POOL_MAX_SLOTS];
} LW2080_CTRL_GR_GFX_POOL_ADD_SLOTS_PARAMS;

/*
 * LW2080_CTRL_CMD_GR_GFX_POOL_REMOVE_SLOTS
 *
 * This API removes buffer slots from a given control structure.  It can
 * only be called when no channel using the given pool is running or may become
 * running for the duration of this call. It can operate in two modes, either
 * it will a specified number of slots, or a specified list of slots.
 *
 * It is only accessible to kernel callers.
 *
 * LW2080_CTRL_CMD_GR_GFX_POOL_REMOVE_SLOTS_PARAMS
 *
 *   pControlStructure
 *     This input is the kernel CPU pointer to the control structure
 *   numSlots
 *     This input indicates how many slots are being removed.  if
 *     bRemoveSpecificSlots is true, then it also indicates how many entries in
 *     the slots array are populated.
 *   slots
 *     This array is either an input or output.  If bRemoveSpecificSlots is true,
 *     then this will contain the list of slots to remove.  If it is false, then
 *     it will be populated by RM with the indexes of the slots that were
 *     removed.
 *   bRemoveSpecificSlots
 *     This input determines which mode the call will run in.  If true the caller
 *     will specify the list of slots they want removed, if any of those slots
 *     are not on the freelist, the call will fail.  If false they only specify
 *     the number of slots they want removed and RM will pick up to that
 *     many.  If there are not enough slots on the freelist to remove the
 *     requested amount, RM will return the number it was able to remove.
 */
#define LW2080_CTRL_CMD_GR_GFX_POOL_REMOVE_SLOTS (0x20801222U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x22" */

typedef struct LW2080_CTRL_GR_GFX_POOL_REMOVE_SLOTS_PARAMS {
    LW_DECLARE_ALIGNED(LwP64 pControlStructure, 8);
    LwU32  numSlots;
    LwU32  slots[LW2080_CTRL_GR_GFX_POOL_MAX_SLOTS];
    LwBool bRemoveSpecificSlots;
} LW2080_CTRL_GR_GFX_POOL_REMOVE_SLOTS_PARAMS;

#if (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)

/*
 * LW2080_CTRL_CMD_GR_REG_ACCESS
 *
 * This command reads/write from/to a PGRAPH register requested by client.
 * This is a mods only call and should never be allowed outside of mods.
 * This API can only be used to access Legacy PGRAPH space, other usage is
 * deprecated and disabled.
 *
 *   regOffset[IN]
 *      - Register offset in PGRAPH requested to be accessed
 *
 *   regVal[IN/OUT]
 *      - Register value read/written
 *
 *   grRouteInfo[IN]
 *      - Deprecated. Must pass as 0x0.
 *
 *   accessFlag[IN]
 *      - READ/WRITE -Its a READ access or WRITE access
 *      - LEGACY - Perform access in legacy PGRAPH space ignoring per MIG space
 *                 even if MIG is set. This flag must be set.
 */
#define LW2080_CTRL_CMD_GR_REG_ACCESS (0x20801226U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x26" */

typedef struct LW2080_CTRL_GR_REG_ACCESS_PARAMS {
    LwU32 regOffset;
    LwU32 regVal;
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
    LwU32 accessFlag;
} LW2080_CTRL_GR_REG_ACCESS_PARAMS;

#define LW2080_CTRL_GR_REG_ACCESS_FLAG              2:0
#define LW2080_CTRL_GR_REG_ACCESS_FLAG_READ         LWBIT(0)
#define LW2080_CTRL_GR_REG_ACCESS_FLAG_WRITE        LWBIT(1)
#define LW2080_CTRL_GR_REG_ACCESS_FLAG_LEGACY       LWBIT(2)

/* LWRM_UNPUBLISHED */
#endif // (!defined(LWRM_UNPUBLISHED) || LWRM_UNPUBLISHED == 1)



#define LW2080_CTRL_CMD_GR_GET_CAPS_V2 (0x20801227U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_CAPS_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_CAPS_V2_PARAMS_MESSAGE_ID (0x27U)

typedef LW0080_CTRL_GR_GET_CAPS_V2_PARAMS LW2080_CTRL_GR_GET_CAPS_V2_PARAMS;

#define LW2080_CTRL_CMD_GR_GET_INFO_V2 (0x20801228U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_INFO_V2_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_INFO_V2_PARAMS_MESSAGE_ID (0x28U)

typedef LW0080_CTRL_GR_GET_INFO_V2_PARAMS LW2080_CTRL_GR_GET_INFO_V2_PARAMS;

#define LW2080_CTRL_CMD_GR_SET_CONTEXT_OVERRIDE (0x20801229U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_CMD_GR_SET_CONTEXT_OVERRIDE_PARAMS_MESSAGE_ID" */

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)
#define LW2080_CTRL_CMD_GR_SET_CONTEXT_OVERRIDE_PARAMS_MESSAGE_ID (0x29U)

typedef LW0080_CTRL_CMD_GR_SET_CONTEXT_OVERRIDE_PARAMS LW2080_CTRL_CMD_GR_SET_CONTEXT_OVERRIDE_PARAMS;
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW2080_CTRL_CMD_GR_GET_GPC_MASK
 *
 * This command returns a mask of enabled GPCs for the associated subdevice.
 *
 *    grRouteInfo[IN]
 *      This parameter specifies the routing information used to
 *      disambiguate the target GR engine.
 *
 *    gpcMask[OUT]
 *      This parameter returns a mask of enabled GPCs. Each GPC has an ID
 *      that's equivalent to the corresponding bit position in the mask.
 */
#define LW2080_CTRL_CMD_GR_GET_GPC_MASK (0x2080122aU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_GPC_MASK_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_GPC_MASK_PARAMS_MESSAGE_ID (0x2AU)

typedef struct LW2080_CTRL_GR_GET_GPC_MASK_PARAMS {
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
    LwU32 gpcMask;
} LW2080_CTRL_GR_GET_GPC_MASK_PARAMS;

/*
 * LW2080_CTRL_CMD_GR_GET_TPC_MASK
 *
 * This command returns a mask of enabled TPCs for a specified GPC.
 *
 *    grRouteInfo[IN]
 *      This parameter specifies the routing information used to
 *      disambiguate the target GR engine.
 *
 *    gpcId[IN]
 *      This parameter specifies the GPC for which TPC information is
 *      to be retrieved. If the GPC with this ID is not enabled this command
 *      will return an tpcMask value of zero.
 *
 *    tpcMask[OUT]
 *      This parameter returns a mask of enabled TPCs for the specified GPC.
 *      Each TPC has an ID that's equivalent to the corresponding bit
 *      position in the mask.
 */
#define LW2080_CTRL_CMD_GR_GET_TPC_MASK (0x2080122bU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_TPC_MASK_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_TPC_MASK_PARAMS_MESSAGE_ID (0x2BU)

typedef struct LW2080_CTRL_GR_GET_TPC_MASK_PARAMS {
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
    LwU32 gpcId;
    LwU32 tpcMask;
} LW2080_CTRL_GR_GET_TPC_MASK_PARAMS;

#define LW2080_CTRL_CMD_GR_SET_TPC_PARTITION_MODE (0x2080122lw) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x2C" */

typedef LW0080_CTRL_GR_TPC_PARTITION_MODE_PARAMS LW2080_CTRL_GR_SET_TPC_PARTITION_MODE_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LW2080_CTRL_CMD_GR_SET_TPC_PARTITION_MODE_FINN_PARAMS_MESSAGE_ID (0x2LW)

typedef struct LW2080_CTRL_CMD_GR_SET_TPC_PARTITION_MODE_FINN_PARAMS {
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_SET_TPC_PARTITION_MODE_PARAMS params, 8);
} LW2080_CTRL_CMD_GR_SET_TPC_PARTITION_MODE_FINN_PARAMS;



/*
 * LW2080_CTRL_CMD_GR_GET_ENGINE_CONTEXT_PROPERTIES
 *
 * This command is used to provide the caller with the alignment and size
 * of the context save region for an engine
 *
 *   grRouteInfo
 *     This parameter specifies the routing information used to
 *     disambiguate the target GR engine.
 *   engineId
 *     This parameter is an input parameter specifying the engineId for which
 *     the alignment/size is requested.
 *   alignment
 *     This parameter is an output parameter which will be filled in with the
 *     minimum alignment requirement.
 *   size
 *     This parameter is an output parameter which will be filled in with the
 *     minimum size of the context save region for the engine.
 *   bInfoPopulated
 *     This parameter will be set if alignment and size are already set with
 *     valid values from a previous call.
 */

#define LW2080_CTRL_CMD_GR_GET_ENGINE_CONTEXT_PROPERTIES (0x2080122dU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_ENGINE_CONTEXT_PROPERTIES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_ENGINE_CONTEXT_PROPERTIES_PARAMS_MESSAGE_ID (0x2DU)

typedef struct LW2080_CTRL_GR_GET_ENGINE_CONTEXT_PROPERTIES_PARAMS {
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
    LwU32  engineId;
    LwU32  alignment;
    LwU32  size;
    LwBool bInfoPopulated;
} LW2080_CTRL_GR_GET_ENGINE_CONTEXT_PROPERTIES_PARAMS;

#if (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)


/*
 * LW2080_CTRL_CMD_GR_SET_CTXSW_TEST_STATE
 *
 * This command provides an interface to set context switch test state for
 * verification.
 *
 *    grRouteInfo[IN]
 *      This parameter specifies the routing information used to
 *      disambiguate the target GR engine.
 *
 *    hChannel[IN]
 *      This parameter specifies the channel handle of the channel
 *      that is to have its context switch test state set.
 *
 *    command[IN]
 *      This parameter specifies the type of operation to perform on the
 *      context switch test state. All valid commands are in the
 *      CTXSW_TEST_STATE enum.
 *
 *    arg[IN]
 *      This parameter specifies the argument required by the test state set
 *      command. All valid arguments can be found in lwcm.h under
 *      LW_CFGEX_SET_CTXSW_TEST_STATE.
 */
#define LW2080_CTRL_CMD_GR_SET_CTXSW_TEST_STATE (0x2080122eU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x2E" */

typedef enum LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_COMMAND {
    LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_COMMAND_SAVE = 0,
    LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_COMMAND_RESTORE = 1,
    LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_COMMAND_RESETMASK = 2,
    LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_COMMAND_DEBUG_CONTROL = 3,
    LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_COMMAND_SWITCH_TYPE = 4,
    LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_COMMAND_MODE = 5,
    LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_COMMAND_BASH = 6,
    LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_COMMAND_CHECK_DUPLICATE = 7,
} LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_COMMAND;

// Arg contains one or more of the following for Debug Control
#define LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_ARG_DEBUG_CONTROL_SINGLE_STEP LWBIT32(0)
#define LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_ARG_DEBUG_CONTROL_LOG         LWBIT32(1)
#define LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_ARG_DEBUG_CONTROL_DEBUG       LWBIT32(2)
#define LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_ARG_DEBUG_CONTROL_SERIAL      LWBIT32(3)
#define LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_ARG_DEBUG_CONTROL_SELF        LWBIT32(4)

// Arg contains one of the following for Switch Type
#define LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_ARG_SWITCH_TYPE_WFI                 (0U)
#define LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_ARG_SWITCH_TYPE_HALT                (1U)
#define LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_ARG_SWITCH_TYPE_HALTONMETHOD        (2U)
#define LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_ARG_SWITCH_TYPE_SPILLREPLAYONLY     (3U)
#define LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_ARG_SWITCH_TYPE_HALT_ON_WFI_TIMEOUT (4U)
#define LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_ARG_SWITCH_TYPE_WFI_SCAN_CHAIN      (5U)
#define LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_ARG_SWITCH_TYPE_WFI_ALL_PLI         (6U)
#define LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_ARG_SWITCH_TYPE_WFI_FLOP_PLI        (7U)
#define LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_ARG_SWITCH_TYPE_HALT_ALL_PLI        (8U)
#define LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_ARG_SWITCH_TYPE_HALT_FLOP_PLI       (9U)

// Arg contains one of the following for Mode
#define LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_ARG_MODE_TYPE_HW                    (1U)
#define LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_ARG_MODE_TYPE_SW                    (2U)

typedef struct LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_PARAMS {
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
    LwHandle                                    hChannel;
    LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_COMMAND command;
    LwU32                                       arg;
} LW2080_CTRL_GR_SET_CTXSW_TEST_STATE_PARAMS;


/*
 * LW2080_CTRL_CMD_GR_PROCESS_PREEMPTION_STATS
 * This command provides interaface to get the process Preemption stats.
 * This can be used only in VERIF and DEBUG modes.
 *
 *    grRouteInfo[IN]
 *      This parameter specifies the routing information used to
 *      disambiguate the target GR engine.
 *    hChannel[IN]
 *      This parameter specifies the channel handle of the channel
 *      that is to have its context switch test state set.
 */

#define LW2080_CTRL_CMD_GR_PROCESS_PREEMPTION_STATS (0x2080122fU) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x2F" */

typedef struct LW2080_CTRL_GR_PROCESS_PREEMPTION_STATS_PARAMS {
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
    LwHandle hChannel;
} LW2080_CTRL_GR_PROCESS_PREEMPTION_STATS_PARAMS;

/* LWRM_PUBLISHED_PENDING_IP_REVIEW */
#endif // (!defined(LWRM_PUBLISHED_PENDING_IP_REVIEW) || LWRM_PUBLISHED_PENDING_IP_REVIEW == 1)



/*
 * LW2080_CTRL_CMD_GR_GET_SM_ISSUE_RATE_MODIFIER
 *
 * This command provides an interface to retrieve the speed select values of
 * various instruction types.
 *
 *    grRouteInfo[IN]
 *      This parameter specifies the routing information used to
 *      disambiguate the target GR engine.
 *
 *    imla0[OUT]
 *       The current speed select for IMLA0.
 *
 *    fmla16[OUT]
 *       The current speed select for FMLA16.
 *
 *    dp[OUT]
 *       The current speed select for DP.
 *
 *    fmla32[OUT]
 *       The current speed select for FMLA32.
 *
 *    ffma[OUT]
 *       The current speed select for FFMA.
 *
 *    imla1[OUT]
 *       The current speed select for IMLA1.
 *
 *    imla2[OUT]
 *       The current speed select for IMLA2.
 *
 *    imla3[OUT]
 *       The current speed select for IMLA3.
 *
 *    imla4[OUT]
 *       The current speed select for IMLA4.
 */
#define LW2080_CTRL_CMD_GR_GET_SM_ISSUE_RATE_MODIFIER                       (0x20801230U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA0_FULL_SPEED          (0x0U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA0_REDUCED_SPEED_1_2   (0x1U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA0_REDUCED_SPEED_1_4   (0x2U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA0_REDUCED_SPEED_1_8   (0x3U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA0_REDUCED_SPEED_1_16  (0x4U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA0_REDUCED_SPEED_1_32  (0x5U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA0_REDUCED_SPEED_1_64  (0x6U)

#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FMLA16_FULL_SPEED         (0x0U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FMLA16_REDUCED_SPEED_1_2  (0x1U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FMLA16_REDUCED_SPEED_1_4  (0x2U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FMLA16_REDUCED_SPEED_1_8  (0x3U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FMLA16_REDUCED_SPEED_1_16 (0x4U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FMLA16_REDUCED_SPEED_1_32 (0x5U)

#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_DP_FULL_SPEED             (0x0U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_DP_REDUCED_SPEED          (0x1U)

#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FMLA32_FULL_SPEED         (0x0U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FMLA32_REDUCED_SPEED_1_2  (0x1U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FMLA32_REDUCED_SPEED_1_4  (0x2U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FMLA32_REDUCED_SPEED_1_8  (0x3U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FMLA32_REDUCED_SPEED_1_16 (0x4U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FMLA32_REDUCED_SPEED_1_32 (0x5U)

#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FFMA_FULL_SPEED           (0x0U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FFMA_REDUCED_SPEED_1_2    (0x1U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FFMA_REDUCED_SPEED_1_4    (0x2U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FFMA_REDUCED_SPEED_1_8    (0x3U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FFMA_REDUCED_SPEED_1_16   (0x4U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_FFMA_REDUCED_SPEED_1_32   (0x5U)

#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA1_FULL_SPEED          (0x0U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA1_REDUCED_SPEED_1_2   (0x1U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA1_REDUCED_SPEED_1_4   (0x2U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA1_REDUCED_SPEED_1_8   (0x3U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA1_REDUCED_SPEED_1_16  (0x4U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA1_REDUCED_SPEED_1_32  (0x5U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA1_REDUCED_SPEED_1_64  (0x6U)

#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA2_FULL_SPEED          (0x0U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA2_REDUCED_SPEED_1_2   (0x1U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA2_REDUCED_SPEED_1_4   (0x2U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA2_REDUCED_SPEED_1_8   (0x3U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA2_REDUCED_SPEED_1_16  (0x4U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA2_REDUCED_SPEED_1_32  (0x5U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA2_REDUCED_SPEED_1_64  (0x6U)

#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA3_FULL_SPEED          (0x0U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA3_REDUCED_SPEED_1_2   (0x1U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA3_REDUCED_SPEED_1_4   (0x2U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA3_REDUCED_SPEED_1_8   (0x3U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA3_REDUCED_SPEED_1_16  (0x4U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA3_REDUCED_SPEED_1_32  (0x5U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA3_REDUCED_SPEED_1_64  (0x6U)

#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA4_FULL_SPEED          (0x0U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA4_REDUCED_SPEED_1_2   (0x1U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA4_REDUCED_SPEED_1_4   (0x2U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA4_REDUCED_SPEED_1_8   (0x3U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA4_REDUCED_SPEED_1_16  (0x4U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA4_REDUCED_SPEED_1_32  (0x5U)
#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_IMLA4_REDUCED_SPEED_1_64  (0x6U)

#define LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_PARAMS_MESSAGE_ID (0x30U)

typedef struct LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_PARAMS {
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
    LwU8 imla0;
    LwU8 fmla16;
    LwU8 dp;
    LwU8 fmla32;
    LwU8 ffma;
    LwU8 imla1;
    LwU8 imla2;
    LwU8 imla3;
    LwU8 imla4;
} LW2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_PARAMS;

/*
 * LW2080_CTRL_CMD_GR_FECS_BIND_EVTBUF_FOR_UID
 *
 * *DEPRECATED* Use LW2080_CTRL_CMD_GR_FECS_BIND_EVTBUF_FOR_UID_V2 instead
 *
 * This command is used to create a FECS bind-point to an event buffer that
 * is filtered by UID.
 *
 *  hEventBuffer[IN]
 *      The event buffer to bind to
 *
 *  recordSize[IN]
 *      The size of the FECS record in bytes
 *
 *  levelOfDetail[IN]
 *      One of LW2080_CTRL_GR_FECS_BIND_EVTBUF_LOD_:
 *          FULL: Report all CtxSw events
 *          SIMPLE: Report ACTIVE_REGION_START and ACTIVE_REGION_END only
 *          COMPAT: Events that KMD is interested in (for backwards compatibility)
 *          CUSTOM: Report events in the eventFilter field
 *      NOTE: RM may override the level-of-detail depending on the caller
 *
 *  eventFilter[IN]
 *      Bitmask of events to report if levelOfDetail is CUSTOM
 *
 *  bAllUsers[IN]
 *     Only report FECS CtxSw data for the current user if false, for all users if true
 */

#define LW2080_CTRL_CMD_GR_FECS_BIND_EVTBUF_FOR_UID (0x20801231U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_FECS_BIND_EVTBUF_FOR_UID_PARAMS_MESSAGE_ID" */

typedef enum LW2080_CTRL_GR_FECS_BIND_EVTBUF_LOD {
    LW2080_CTRL_GR_FECS_BIND_EVTBUF_LOD_FULL = 0,
    LW2080_CTRL_GR_FECS_BIND_EVTBUF_LOD_SIMPLE = 1,
    LW2080_CTRL_GR_FECS_BIND_EVTBUF_LOD_COMPAT = 2,
    LW2080_CTRL_GR_FECS_BIND_EVTBUF_LOD_LWSTOM = 3,
} LW2080_CTRL_GR_FECS_BIND_EVTBUF_LOD;

#define LW2080_CTRL_GR_FECS_BIND_EVTBUF_FOR_UID_PARAMS_MESSAGE_ID (0x31U)

typedef struct LW2080_CTRL_GR_FECS_BIND_EVTBUF_FOR_UID_PARAMS {
    LwHandle                            hEventBuffer;
    LwU32                               recordSize;
    LW2080_CTRL_GR_FECS_BIND_EVTBUF_LOD levelOfDetail;
    LwU32                               eventFilter;
    LwBool                              bAllUsers;
} LW2080_CTRL_GR_FECS_BIND_EVTBUF_FOR_UID_PARAMS;

/*
 * LW2080_CTRL_CMD_GR_GET_PHYS_GPC_MASK
 *
 * This command returns a mask of physical GPC Ids for the associated syspipe
 *
 *    physSyspipeId[IN]
 *      This parameter specifies syspipe for which phys GPC mask is requested
 *
 *    gpcMask[OUT]
 *      This parameter returns a mask of mapped GPCs to provided syspipe.
 *      Each GPC-ID has a corresponding bit position in the mask.
 */
#define LW2080_CTRL_CMD_GR_GET_PHYS_GPC_MASK (0x20801232U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_PHYS_GPC_MASK_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_PHYS_GPC_MASK_PARAMS_MESSAGE_ID (0x32U)

typedef struct LW2080_CTRL_GR_GET_PHYS_GPC_MASK_PARAMS {
    LwU32 physSyspipeId;
    LwU32 gpcMask;
} LW2080_CTRL_GR_GET_PHYS_GPC_MASK_PARAMS;

/*
 * LW2080_CTRL_CMD_GR_GET_PPC_MASK
 *
 * This command returns a mask of enabled PPCs for a specified GPC.
 *
 *    grRouteInfo[IN]
 *      This parameter specifies the routing information used to
 *      disambiguate the target GR engine.
 *
 *    gpcId[IN]
 *      This parameter specifies the GPC for which TPC information is
 *      to be retrieved. If the GPC with this ID is not enabled this command
 *      will return an ppcMask value of zero.
 *
 *    ppcMask[OUT]
 *      This parameter returns a mask of enabled PPCs for the specified GPC.
 *      Each PPC has an ID that's equivalent to the corresponding bit
 *      position in the mask.
 */
#define LW2080_CTRL_CMD_GR_GET_PPC_MASK (0x20801233U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_PPC_MASK_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_PPC_MASK_PARAMS_MESSAGE_ID (0x33U)

typedef struct LW2080_CTRL_GR_GET_PPC_MASK_PARAMS {
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_ROUTE_INFO grRouteInfo, 8);
    LwU32 gpcId;
    LwU32 ppcMask;
} LW2080_CTRL_GR_GET_PPC_MASK_PARAMS;

#define LW2080_CTRL_CMD_GR_GET_NUM_TPCS_FOR_GPC (0x20801234U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_NUM_TPCS_FOR_GPC_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_NUM_TPCS_FOR_GPC_PARAMS_MESSAGE_ID (0x34U)

typedef struct LW2080_CTRL_GR_GET_NUM_TPCS_FOR_GPC_PARAMS {
    LwU32 gpcId;
    LwU32 numTpcs;
} LW2080_CTRL_GR_GET_NUM_TPCS_FOR_GPC_PARAMS;

/*
 * LW2080_CTRL_CMD_GR_GET_CTXSW_MODES
 *
 * This command is used to get context switch modes for the specified
 * channel. A value of LW_ERR_NOT_SUPPORTED is returned if the
 * target channel does not support context switch mode changes.
 *
 *   hChannel
 *     This parameter specifies the channel handle of
 *     the channel that is to have its context switch modes retrieved.
 *   zlwllMode
 *     See LW2080_CTRL_CMD_GR_CTXSW_ZLWLL_MODE for possible return values
 *   pmMode
 *     See LW2080_CTRL_CMD_GR_CTXSW_PM_MODE for possible return values
 *   smpcMode
 *     See LW2080_CTRL_CMD_GR_CTXSW_SMPC_MODE for possible return values
 *   cilpPreemptMode
 *     See LW2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE for possible return values
 *   gfxpPreemptMode
 *     See LW2080_CTRL_CMD_GR_SET_CTXSW_PREEMPTION_MODE for possible return values
 */
#define LW2080_CTRL_CMD_GR_GET_CTXSW_MODES (0x20801235U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_CTXSW_MODES_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_CTXSW_MODES_PARAMS_MESSAGE_ID (0x35U)

typedef struct LW2080_CTRL_GR_GET_CTXSW_MODES_PARAMS {
    LwHandle hChannel;
    LwU32    zlwllMode;
    LwU32    pmMode;
    LwU32    smpcMode;
    LwU32    cilpPreemptMode;
    LwU32    gfxpPreemptMode;
} LW2080_CTRL_GR_GET_CTXSW_MODES_PARAMS;

/*
 * LW2080_CTRL_CMD_GR_GET_GPC_TILE_MAP
 *
 * Get a list of values used to describe GPC/TPC tile mapping tables.
 *
 *   mapValueCount
 *     This field specifies the number of actual map entries.  This count
 *     should equal the number of TPCs in the system.
 *   mapValues
 *     This field is a pointer to a buffer of LwU08 values representing map
 *     data.
 *   grRouteInfo
 *     This parameter specifies the routing information used to
 *     disambiguate the target GR engine.
 */
#define LW2080_CTRL_CMD_GR_GET_GPC_TILE_MAP (0x20801236U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | 0x36" */

typedef LW2080_CTRL_GR_SET_GPC_TILE_MAP_PARAMS LW2080_CTRL_GR_GET_GPC_TILE_MAP_PARAMS;

// FINN PORT: The below type was generated by the FINN port to
// ensure that all API's have a unique structure associated
// with them!
#define LW2080_CTRL_CMD_GR_GET_GPC_TILE_MAP_FINN_PARAMS_MESSAGE_ID (0x36U)

typedef struct LW2080_CTRL_CMD_GR_GET_GPC_TILE_MAP_FINN_PARAMS {
    LW_DECLARE_ALIGNED(LW2080_CTRL_GR_GET_GPC_TILE_MAP_PARAMS params, 8);
} LW2080_CTRL_CMD_GR_GET_GPC_TILE_MAP_FINN_PARAMS;



/*
 * LW2080_CTRL_CMD_GR_GET_ZLWLL_MASK
 *
 * This command returns a mask of enabled ZLWLLs for a specified GPC.
 *
 *    gpcId[IN]
 *      This parameter, physical GPC index, specifies the GPC for which ZLWLL
 *      information is to be retrieved. If the GPC with this ID is not enabled
 *      this command will return a zlwllMask value of zero.
 *
 *    zlwllMask[OUT]
 *      This parameter returns a mask of enabled ZLWLLs for the specified GPC.
 *      Each ZLWLL has an ID that's equivalent to the corresponding bit
 *      position in the mask.
 */

#define LW2080_CTRL_CMD_GR_GET_ZLWLL_MASK (0x20801237U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_GET_ZLWLL_MASK_PARAMS_MESSAGE_ID" */

#define LW2080_CTRL_GR_GET_ZLWLL_MASK_PARAMS_MESSAGE_ID (0x37U)

typedef struct LW2080_CTRL_GR_GET_ZLWLL_MASK_PARAMS {
    LwU32 gpcId;
    LwU32 zlwllMask;
} LW2080_CTRL_GR_GET_ZLWLL_MASK_PARAMS;

/*
 * LW2080_CTRL_CMD_GR_FECS_BIND_EVTBUF_FOR_UID_V2
 *
 * This command is used to create a FECS bind-point to an event buffer that
 * is filtered by UID.
 *
 *  hEventBuffer[IN]
 *      The event buffer to bind to
 *
 *  recordSize[IN]
 *      The size of the FECS record in bytes
 *
 *  levelOfDetail[IN]
 *      One of LW2080_CTRL_GR_FECS_BIND_EVTBUF_LOD_:
 *          FULL: Report all CtxSw events
 *          SIMPLE: Report ACTIVE_REGION_START and ACTIVE_REGION_END only
 *          COMPAT: Events that KMD is interested in (for backwards compatibility)
 *          CUSTOM: Report events in the eventFilter field
 *      NOTE: RM may override the level-of-detail depending on the caller
 *
 *  eventFilter[IN]
 *      Bitmask of events to report if levelOfDetail is CUSTOM
 *
 *  bAllUsers[IN]
 *     Only report FECS CtxSw data for the current user if false, for all users if true
 *
 *  reasonCode [OUT]
 *     Reason for failure
 */
#define LW2080_CTRL_CMD_GR_FECS_BIND_EVTBUF_FOR_UID_V2 (0x20801238U) /* finn: Evaluated from "(FINN_LW20_SUBDEVICE_0_GR_INTERFACE_ID << 8) | LW2080_CTRL_GR_FECS_BIND_EVTBUF_FOR_UID_V2_PARAMS_MESSAGE_ID" */

typedef enum LW2080_CTRL_GR_FECS_BIND_EVTBUF_REASON_CODE {
    LW2080_CTRL_GR_FECS_BIND_REASON_CODE_NONE = 0,
    LW2080_CTRL_GR_FECS_BIND_REASON_CODE_GPU_TOO_OLD = 1,
    LW2080_CTRL_GR_FECS_BIND_REASON_CODE_NOT_ENABLED_GPU = 2,
    LW2080_CTRL_GR_FECS_BIND_REASON_CODE_NOT_ENABLED = 3,
    LW2080_CTRL_GR_FECS_BIND_REASON_CODE_NEED_ADMIN = 4,
    LW2080_CTRL_GR_FECS_BIND_REASON_CODE_NEED_CAPABILITY = 5,
} LW2080_CTRL_GR_FECS_BIND_EVTBUF_REASON_CODE;

#define LW2080_CTRL_GR_FECS_BIND_EVTBUF_FOR_UID_V2_PARAMS_MESSAGE_ID (0x38U)

typedef struct LW2080_CTRL_GR_FECS_BIND_EVTBUF_FOR_UID_V2_PARAMS {
    LwHandle                            hEventBuffer;
    LwU32                               recordSize;
    LW2080_CTRL_GR_FECS_BIND_EVTBUF_LOD levelOfDetail;
    LwU32                               eventFilter;
    LwBool                              bAllUsers;
    LwU32                               reasonCode;
} LW2080_CTRL_GR_FECS_BIND_EVTBUF_FOR_UID_V2_PARAMS;

/* _ctrl2080gr_h_ */
