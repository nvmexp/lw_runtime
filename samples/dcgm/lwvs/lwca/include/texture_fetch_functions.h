/*
 * Copyright 1993-2017 LWPU Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to LWPU intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to LWPU and is being provided under the terms and
 * conditions of a form of LWPU software license agreement by and
 * between LWPU and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of LWPU is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, LWPU MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * LWPU DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL LWPU BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__TEXTURE_FETCH_FUNCTIONS_H__)
#define __TEXTURE_FETCH_FUNCTIONS_H__


#if defined(__cplusplus) && defined(__LWDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "builtin_types.h"
#include "lwda_texture_types.h"
#include "host_defines.h"
#include "texture_types.h"


template <typename T>
struct __lw_tex_rmet_ret { };

template<> struct __lw_tex_rmet_ret<char> { typedef char type; };
template<> struct __lw_tex_rmet_ret<signed char> { typedef signed char type; };
template<> struct __lw_tex_rmet_ret<unsigned char> { typedef unsigned char type; };
template<> struct __lw_tex_rmet_ret<char1> { typedef char1 type; };
template<> struct __lw_tex_rmet_ret<uchar1> { typedef uchar1 type; };
template<> struct __lw_tex_rmet_ret<char2> { typedef char2 type; };
template<> struct __lw_tex_rmet_ret<uchar2> { typedef uchar2 type; };
template<> struct __lw_tex_rmet_ret<char4> { typedef char4 type; };
template<> struct __lw_tex_rmet_ret<uchar4> { typedef uchar4 type; };

template<> struct __lw_tex_rmet_ret<short> { typedef short type; };
template<> struct __lw_tex_rmet_ret<unsigned short> { typedef unsigned short type; };
template<> struct __lw_tex_rmet_ret<short1> { typedef short1 type; };
template<> struct __lw_tex_rmet_ret<ushort1> { typedef ushort1 type; };
template<> struct __lw_tex_rmet_ret<short2> { typedef short2 type; };
template<> struct __lw_tex_rmet_ret<ushort2> { typedef ushort2 type; };
template<> struct __lw_tex_rmet_ret<short4> { typedef short4 type; };
template<> struct __lw_tex_rmet_ret<ushort4> { typedef ushort4 type; };

template<> struct __lw_tex_rmet_ret<int> { typedef int type; };
template<> struct __lw_tex_rmet_ret<unsigned int> { typedef unsigned int type; };
template<> struct __lw_tex_rmet_ret<int1> { typedef int1 type; };
template<> struct __lw_tex_rmet_ret<uint1> { typedef uint1 type; };
template<> struct __lw_tex_rmet_ret<int2> { typedef int2 type; };
template<> struct __lw_tex_rmet_ret<uint2> { typedef uint2 type; };
template<> struct __lw_tex_rmet_ret<int4> { typedef int4 type; };
template<> struct __lw_tex_rmet_ret<uint4> { typedef uint4 type; };

#if !defined(__LP64__)
template<> struct __lw_tex_rmet_ret<long> { typedef long type; };
template<> struct __lw_tex_rmet_ret<unsigned long> { typedef unsigned long type; };
template<> struct __lw_tex_rmet_ret<long1> { typedef long1 type; };
template<> struct __lw_tex_rmet_ret<ulong1> { typedef ulong1 type; };
template<> struct __lw_tex_rmet_ret<long2> { typedef long2 type; };
template<> struct __lw_tex_rmet_ret<ulong2> { typedef ulong2 type; };
template<> struct __lw_tex_rmet_ret<long4> { typedef long4 type; };
template<> struct __lw_tex_rmet_ret<ulong4> { typedef ulong4 type; };
#endif /* !__LP64__ */
template<> struct __lw_tex_rmet_ret<float> { typedef float type; };
template<> struct __lw_tex_rmet_ret<float1> { typedef float1 type; };
template<> struct __lw_tex_rmet_ret<float2> { typedef float2 type; };
template<> struct __lw_tex_rmet_ret<float4> { typedef float4 type; };


template <typename T> struct __lw_tex_rmet_cast { typedef T* type;  };
#if !defined(__LP64__)
template<> struct __lw_tex_rmet_cast<long> { typedef int *type; };
template<> struct __lw_tex_rmet_cast<unsigned long> { typedef unsigned int *type; };
template<> struct __lw_tex_rmet_cast<long1> { typedef int1 *type; };
template<> struct __lw_tex_rmet_cast<ulong1> { typedef uint1 *type; };
template<> struct __lw_tex_rmet_cast<long2> { typedef int2 *type; };
template<> struct __lw_tex_rmet_cast<ulong2> { typedef uint2 *type; };
template<> struct __lw_tex_rmet_cast<long4> { typedef int4 *type; };
template<> struct __lw_tex_rmet_cast<ulong4> { typedef uint4 *type; };
#endif /* !__LP64__ */

template <typename T>
static __forceinline__ __device__  typename __lw_tex_rmet_ret<T>::type tex1Dfetch(texture<T, lwdaTextureType1D, lwdaReadModeElementType> t, int x)
{
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__tex1Dfetch_v2", (typename __lw_tex_rmet_cast<T>::type)&temp, t, x);
  return temp;
#endif
}

template <typename T>
struct __lw_tex_rmnf_ret { };

template <> struct __lw_tex_rmnf_ret<char> { typedef float type; };
template <> struct __lw_tex_rmnf_ret<signed char> { typedef float type; };
template <> struct __lw_tex_rmnf_ret<unsigned char> { typedef float type; };
template <> struct __lw_tex_rmnf_ret<short> { typedef float type; };
template <> struct __lw_tex_rmnf_ret<unsigned short> { typedef float type; };
template <> struct __lw_tex_rmnf_ret<char1> { typedef float1 type; };
template <> struct __lw_tex_rmnf_ret<uchar1> { typedef float1 type; };
template <> struct __lw_tex_rmnf_ret<short1> { typedef float1 type; };
template <> struct __lw_tex_rmnf_ret<ushort1> { typedef float1 type; };
template <> struct __lw_tex_rmnf_ret<char2> { typedef float2 type; };
template <> struct __lw_tex_rmnf_ret<uchar2> { typedef float2 type; };
template <> struct __lw_tex_rmnf_ret<short2> { typedef float2 type; };
template <> struct __lw_tex_rmnf_ret<ushort2> { typedef float2 type; };
template <> struct __lw_tex_rmnf_ret<char4> { typedef float4 type; };
template <> struct __lw_tex_rmnf_ret<uchar4> { typedef float4 type; };
template <> struct __lw_tex_rmnf_ret<short4> { typedef float4 type; };
template <> struct __lw_tex_rmnf_ret<ushort4> { typedef float4 type; };

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type tex1Dfetch(texture<T, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, int x) 
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__tex1Dfetch_rmnf_v2", &type_dummy, &retval, t, x);
  return retval;
#endif /* __LWDA_ARCH__ */  
}

// tex1D
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type tex1D(texture<T, lwdaTextureType1D, lwdaReadModeElementType> t, float x)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__tex1D_v2", (typename __lw_tex_rmet_cast<T>::type) &temp, t, x);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type tex1D(texture<T, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__tex1D_rmnf_v2", &type_dummy, &retval, t, x);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}


//tex2D
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type tex2D(texture<T, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;

  __lw_tex_surf_handler("__tex2D_v2", (typename __lw_tex_rmet_cast<T>::type) &temp, t, x, y);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type tex2D(texture<T, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__tex2D_rmnf_v2", &type_dummy, &retval, t, x, y);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}


//tex1DLayered
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type tex1DLayered(texture<T, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__tex1DLayered_v2", (typename __lw_tex_rmet_cast<T>::type) &temp, t, x, layer);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type tex1DLayered(texture<T, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__tex1DLayered_rmnf_v2", &type_dummy, &retval, t, x, layer);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}


//tex2DLayered
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type tex2DLayered(texture<T, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__tex2DLayered_v2", (typename __lw_tex_rmet_cast<T>::type) &temp, t, x, y, layer);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type tex2DLayered(texture<T, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__tex2DLayered_rmnf_v2", &type_dummy, &retval, t, x, y, layer);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}

// tex3D
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type tex3D(texture<T, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__tex3D_v2", (typename __lw_tex_rmet_cast<T>::type) &temp, t, x, y, z);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type tex3D(texture<T, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__tex3D_rmnf_v2", &type_dummy, &retval, t, x, y, z);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}

// texLwbemap
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type texLwbemap(texture<T, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__texLwbemap_v2", (typename __lw_tex_rmet_cast<T>::type) &temp, t, x, y, z);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type texLwbemap(texture<T, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__texLwbemap_rmnf_v2", &type_dummy, &retval, t, x, y, z);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}


template <typename T>
struct __lw_tex2dgather_ret { };
template <> struct __lw_tex2dgather_ret<char> { typedef char4 type; };
template <> struct __lw_tex2dgather_ret<signed char> { typedef char4 type; };
template <> struct __lw_tex2dgather_ret<char1> { typedef char4 type; };
template <> struct __lw_tex2dgather_ret<char2> { typedef char4 type; };
template <> struct __lw_tex2dgather_ret<char3> { typedef char4 type; };
template <> struct __lw_tex2dgather_ret<char4> { typedef char4 type; };
template <> struct __lw_tex2dgather_ret<unsigned char> { typedef uchar4 type; };
template <> struct __lw_tex2dgather_ret<uchar1> { typedef uchar4 type; };
template <> struct __lw_tex2dgather_ret<uchar2> { typedef uchar4 type; };
template <> struct __lw_tex2dgather_ret<uchar3> { typedef uchar4 type; };
template <> struct __lw_tex2dgather_ret<uchar4> { typedef uchar4 type; };

template <> struct __lw_tex2dgather_ret<short> { typedef short4 type; };
template <> struct __lw_tex2dgather_ret<short1> { typedef short4 type; };
template <> struct __lw_tex2dgather_ret<short2> { typedef short4 type; };
template <> struct __lw_tex2dgather_ret<short3> { typedef short4 type; };
template <> struct __lw_tex2dgather_ret<short4> { typedef short4 type; };
template <> struct __lw_tex2dgather_ret<unsigned short> { typedef ushort4 type; };
template <> struct __lw_tex2dgather_ret<ushort1> { typedef ushort4 type; };
template <> struct __lw_tex2dgather_ret<ushort2> { typedef ushort4 type; };
template <> struct __lw_tex2dgather_ret<ushort3> { typedef ushort4 type; };
template <> struct __lw_tex2dgather_ret<ushort4> { typedef ushort4 type; };

template <> struct __lw_tex2dgather_ret<int> { typedef int4 type; };
template <> struct __lw_tex2dgather_ret<int1> { typedef int4 type; };
template <> struct __lw_tex2dgather_ret<int2> { typedef int4 type; };
template <> struct __lw_tex2dgather_ret<int3> { typedef int4 type; };
template <> struct __lw_tex2dgather_ret<int4> { typedef int4 type; };
template <> struct __lw_tex2dgather_ret<unsigned int> { typedef uint4 type; };
template <> struct __lw_tex2dgather_ret<uint1> { typedef uint4 type; };
template <> struct __lw_tex2dgather_ret<uint2> { typedef uint4 type; };
template <> struct __lw_tex2dgather_ret<uint3> { typedef uint4 type; };
template <> struct __lw_tex2dgather_ret<uint4> { typedef uint4 type; };

template <> struct __lw_tex2dgather_ret<float> { typedef float4 type; };
template <> struct __lw_tex2dgather_ret<float1> { typedef float4 type; };
template <> struct __lw_tex2dgather_ret<float2> { typedef float4 type; };
template <> struct __lw_tex2dgather_ret<float3> { typedef float4 type; };
template <> struct __lw_tex2dgather_ret<float4> { typedef float4 type; };

template <typename T>
static __device__ __forceinline__ typename __lw_tex2dgather_ret<T>::type tex2Dgather(texture<T, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, int comp=0)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex2dgather_ret<T>::type retval;
  __lw_tex_surf_handler("__tex2Dgather_v2", &type_dummy, &retval, t, x, y, comp);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}


template<typename T> struct __lw_tex2dgather_rmnf_ret { };
template<> struct __lw_tex2dgather_rmnf_ret<char> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<signed char> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<unsigned char> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<char1> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<uchar1> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<char2> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<uchar2> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<char3> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<uchar3> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<char4> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<uchar4> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<signed short> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<unsigned short> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<short1> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<ushort1> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<short2> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<ushort2> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<short3> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<ushort3> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<short4> { typedef float4 type; };
template<> struct __lw_tex2dgather_rmnf_ret<ushort4> { typedef float4 type; };

template <typename T>
static __device__ __forceinline__  typename __lw_tex2dgather_rmnf_ret<T>::type tex2Dgather(texture<T, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, int comp = 0)
{  
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex2dgather_rmnf_ret<T>::type  retval;
  __lw_tex_surf_handler("__tex2Dgather_rmnf_v2", &type_dummy, &retval, t, x, y, comp);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}


// tex1DLod
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type tex1DLod(texture<T, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float level)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__tex1DLod_v2", (typename __lw_tex_rmet_cast<T>::type)&temp, t, x, level);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type tex1DLod(texture<T, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float level)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__tex1DLod_rmnf_v2", &type_dummy, &retval, t, x, level);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}

// tex2DLod
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type tex2DLod(texture<T, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float level)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__tex2DLod_v2", (typename __lw_tex_rmet_cast<T>::type)&temp, t, x, y, level);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type tex2DLod(texture<T, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float level)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__tex2DLod_rmnf_v2", &type_dummy, &retval, t, x, y, level);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}

// tex1DLayeredLod
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type tex1DLayeredLod(texture<T, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float level)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__tex1DLayeredLod_v2", (typename __lw_tex_rmet_cast<T>::type)&temp, t, x, layer, level);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type tex1DLayeredLod(texture<T, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float level)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__tex1DLayeredLod_rmnf_v2", &type_dummy, &retval, t, x, layer, level);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}

// tex2DLayeredLod
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type tex2DLayeredLod(texture<T, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float level)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__tex2DLayeredLod_v2", (typename __lw_tex_rmet_cast<T>::type)&temp, t, x, y, layer, level);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type tex2DLayeredLod(texture<T, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float level)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__tex2DLayeredLod_rmnf_v2", &type_dummy, &retval, t, x, y, layer, level);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}

// tex3DLod
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type tex3DLod(texture<T, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float level)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__tex3DLod_v2",(typename __lw_tex_rmet_cast<T>::type)&temp, t, x, y, z, level);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type tex3DLod(texture<T, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__tex3DLod_rmnf_v2", &type_dummy, &retval, t, x, y, z, level);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}

// texLwbemapLod
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type texLwbemapLod(texture<T, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float level)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__texLwbemapLod_v2",(typename __lw_tex_rmet_cast<T>::type)&temp, t, x, y, z, level);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type texLwbemapLod(texture<T, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float level)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__texLwbemapLod_rmnf_v2", &type_dummy, &retval, t, x, y, z, level);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}


// texLwbemapLayered
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type texLwbemapLayered(texture<T, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer)
{
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__texLwbemapLayered_v2",(typename __lw_tex_rmet_cast<T>::type)&temp, t, x, y, z, layer);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type texLwbemapLayered(texture<T, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer)
{
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__texLwbemapLayered_rmnf_v2", &type_dummy, &retval, t, x, y, z, layer);
  return retval;
#endif /* __LWDA_ARCH__ */
}


// texLwbemapLayeredLod
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type texLwbemapLayeredLod(texture<T, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float level)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__texLwbemapLayeredLod_v2", (typename __lw_tex_rmet_cast<T>::type)&temp, t, x, y, z, layer, level);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type texLwbemapLayeredLod(texture<T, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float level)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__texLwbemapLayeredLod_rmnf_v2", &type_dummy, &retval, t, x, y, z, layer, level);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}


// texLwbemapGrad
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type texLwbemapGrad(texture<T, lwdaTextureTypeLwbemap, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__texLwbemapGrad_v2", (typename __lw_tex_rmet_cast<T>::type)&temp, t, x, y, z, &dPdx, &dPdy);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type texLwbemapGrad(texture<T, lwdaTextureTypeLwbemap, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__texLwbemapGrad_rmnf_v2", &type_dummy, &retval, t,  x, y, z, &dPdx, &dPdy);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}


// texLwbemapLayeredGrad
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type texLwbemapLayeredGrad(texture<T, lwdaTextureTypeLwbemapLayered, lwdaReadModeElementType> t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__texLwbemapLayeredGrad_v2", (typename __lw_tex_rmet_cast<T>::type)&temp, t, x, y, z, layer, &dPdx, &dPdy);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type texLwbemapLayeredGrad(texture<T, lwdaTextureTypeLwbemapLayered, lwdaReadModeNormalizedFloat> t, float x, float y, float z, int layer, float4 dPdx, float4 dPdy) 
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__texLwbemapLayeredGrad_rmnf_v2", &type_dummy, &retval,t, x, y, z, layer, &dPdx, &dPdy);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}


// tex1DGrad
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type tex1DGrad(texture<T, lwdaTextureType1D, lwdaReadModeElementType> t, float x, float dPdx, float dPdy)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__tex1DGrad_v2", (typename __lw_tex_rmet_cast<T>::type)&temp, t, x, dPdx, dPdy);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type tex1DGrad(texture<T, lwdaTextureType1D, lwdaReadModeNormalizedFloat> t, float x, float dPdx, float dPdy)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__tex1DGrad_rmnf_v2", &type_dummy, &retval,t, x,dPdx, dPdy);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}


// tex2DGrad
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type tex2DGrad(texture<T, lwdaTextureType2D, lwdaReadModeElementType> t, float x, float y, float2 dPdx, float2 dPdy)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__tex2DGrad_v2", (typename __lw_tex_rmet_cast<T>::type)&temp, t, x, y, &dPdx, &dPdy);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type tex2DGrad(texture<T, lwdaTextureType2D, lwdaReadModeNormalizedFloat> t, float x, float y, float2 dPdx, float2 dPdy)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__tex2DGrad_rmnf_v2", &type_dummy, &retval,t, x, y, &dPdx, &dPdy);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}

// tex1DLayeredGrad
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type tex1DLayeredGrad(texture<T, lwdaTextureType1DLayered, lwdaReadModeElementType> t, float x, int layer, float dPdx, float dPdy)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__tex1DLayeredGrad_v2",(typename __lw_tex_rmet_cast<T>::type)&temp, t, x, layer, dPdx, dPdy);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type tex1DLayeredGrad(texture<T, lwdaTextureType1DLayered, lwdaReadModeNormalizedFloat> t, float x, int layer, float dPdx, float dPdy)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__tex1DLayeredGrad_rmnf_v2", &type_dummy, &retval,t, x, layer, dPdx, dPdy);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}

// tex2DLayeredGrad
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type tex2DLayeredGrad(texture<T, lwdaTextureType2DLayered, lwdaReadModeElementType> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__tex2DLayeredGrad_v2",(typename __lw_tex_rmet_cast<T>::type)&temp, t, x, y, layer, &dPdx, &dPdy);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type tex2DLayeredGrad(texture<T, lwdaTextureType2DLayered, lwdaReadModeNormalizedFloat> t, float x, float y, int layer, float2 dPdx, float2 dPdy)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__tex2DLayeredGrad_rmnf_v2", &type_dummy, &retval,t, x, y, layer, &dPdx, &dPdy);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}

// tex3DGrad
template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmet_ret<T>::type tex3DGrad(texture<T, lwdaTextureType3D, lwdaReadModeElementType> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{ 
#ifdef __LWDA_ARCH__
  typename __lw_tex_rmet_ret<T>::type temp;
  __lw_tex_surf_handler("__tex3DGrad_v2", (typename __lw_tex_rmet_cast<T>::type)&temp, t, x, y, z, &dPdx, &dPdy);
  return temp;
#endif
}

template <typename T>
static __forceinline__ __device__ typename __lw_tex_rmnf_ret<T>::type tex3DGrad(texture<T, lwdaTextureType3D, lwdaReadModeNormalizedFloat> t, float x, float y, float z, float4 dPdx, float4 dPdy)
{ 
#ifdef __LWDA_ARCH__
  T type_dummy;
  typename __lw_tex_rmnf_ret<T>::type retval;
  __lw_tex_surf_handler("__tex3DGrad_rmnf_v2", &type_dummy, &retval,t, x, y, z, &dPdx, &dPdy);
  return retval;
#endif /* __LWDA_ARCH__ */ 
}

#endif /* __cplusplus && __LWDACC__ */

#endif /* !__TEXTURE_FETCH_FUNCTIONS_H__ */
