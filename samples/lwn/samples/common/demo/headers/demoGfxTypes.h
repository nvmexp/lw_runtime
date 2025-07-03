/*
** This file contains copyrighted code for the LWN 3D API provided by a
** partner company.  The contents of this file should not be used for
** any purpose other than LWN API development and testing.
*/

#ifndef _DEMO_GFX_TYPES_H_
#define _DEMO_GFX_TYPES_H_

// Switch for printing performace value
//#define ENABLE_PERF_WIN
#define ENABLE_PERF

#include <types.h>
#include "stdlib.h"
#include "lwn/lwn.h"
#include "lwnutil.h"

#include "lwnUtil/lwnUtil_PoolAllocator.h"

using namespace lwnUtil;

#ifdef ENABLE_PERF_WIN
#include "LWN/lwnfnptrinline_perf_win.h"
#else
#include "lwn/lwn_FuncPtrInline.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup demoGfxTypes
/// @{


//
// Data Types
//

// \brief Struct for attribute data
struct DEMOGfxShaderAttributeData
{
    u32				index;
    LWNformat		type;
    u32				relativeOffset;

    u32				bindingIndex;
    u32				bindingStride;
};

// \brief Struct for vertex data
struct DEMOGfxVertexData
{
    u64					address;
    LWNbuffer			*object;
    void*				pBuffer;
};

// \brief Struct for index data
struct DEMOGfxIndexData
{
    u64					address;
    LWNbuffer			*object;
    void*				pBuffer;
};

// \brief Struct for uniform data
struct DEMOGfxUniformData
{
    u64					address;
    LWNbuffer*			object;
    void*				pBuffer;
};

// \brief Struct for sampler data
struct DEMOGfxSamplerData
{
    LWNsampler			*object;
};

// \brief Struct for texture data
struct DEMOGfxTextureData
{
    u64					address;
    LWNbuffer*          object;
    void*				pBuffer;
};

// \brief Struct for texture
struct DEMOGfxTexture
{
    LWNtextureHandle    handle;
    LWNtexture			*object;
    DEMOGfxSamplerData  sampler;
    DEMOGfxTextureData  data;
};

// \brief Struct for texture
struct DEMOGfxShader
{
    LWNprogram			        *pgm;
    u32                         bits;

    u32                         nAttribs;
    u32                         nStreams;
    LWLwertexAttribState        attribs[16];
    LWLwertexStreamState        streams[16];
};

struct DEMOGfxRenderTarget
{
	LWNtexture                  **colors;
	LWNtexture                  *depth;
	int                         lwrrentIndex;
    int                         numColorTextures;
};

struct DEMOGfxContextState
{
    LWNblendState blendState;
    LWNcolorState colorState;
    LWNdepthStencilState depthState;
    LWNpolygonState polygonState;
    LWNchannelMaskState channelMasks;
    struct StencilInfo {
        int frontRef;
        int frontTestMask;
        int frontWriteMask;
        int backRef;
        int backTestMask;
        int backWriteMask;
    } stencilCtrl;

    DEMOGfxShader shader;
};

/// \brief Struct contains floatx4
///
struct DEMO_F32x4
{
    union
    {
        struct{ f32 x, y, z, w;} v;
        struct{ f32 r, g, b, a;} c;
        f32 F32[4];
    } u;
};

/// \brief Struct contains floatx3
///
struct DEMO_F32x3
{
    union
    {
        struct{ f32 x, y, z;} v;
        struct{ f32 r, g, b;} c;
        f32 F32[3];
    } u;
};

/// \brief Struct contains floatx2
///
struct DEMO_F32x2
{
    union
    {
        struct{ f32 x, y;} v;
        struct{ f32 r, g;} c;
        f32 F32[2];
    } u;
};

/// \brief Struct contains float
///
struct DEMO_F32
{
    union
    {
        struct{ f32 x;} v;
        struct{ f32 r;} c;
        f32 F32;
    } u;
};

/// \brief Struct contains U32x4
///
struct DEMO_U32x4
{
    union
    {
        struct{ u32 x, y, z, w;} v;
        struct{ u32 r, g, b, a;} c;
        u32 U32[4];
    } u;
};

/// \brief Struct contains U32x3
///
struct DEMO_U32x3
{
    union
    {
        struct{ u32 x, y, z;} v;
        struct{ u32 r, g, b;} c;
        u32 U32[3];
    } u;
};

/// \brief Struct contains U32x2
///
struct DEMO_U32x2
{
    union
    {
        struct{ u32 x, y;} v;
        struct{ u32 r, g;} c;
        u32 U32[2];
    } u;
};

/// \brief Struct contains U32
///
struct DEMO_U32
{
    union
    {
        struct{ u32 x;} v;
        struct{ u32 r;} c;
        u32 U32;
    } u;
};

/// \brief Struct contains S32x4
///
struct DEMO_S32x4
{
    union
    {
        struct{ s32 x, y, z, w;} v;
        struct{ s32 r, g, b, a;} c;
        s32 S32[4];
    } u;
};

/// \brief Struct contains S32x3
///
struct DEMO_S32x3
{
    union
    {
        struct{ s32 x, y, z;} v;
        struct{ s32 r, g, b;} c;
        s32 S32[3];
    } u;
};

/// \brief Struct contains S32x2
///
struct DEMO_S32x2
{
    union
    {
        struct{ s32 x, y;} v;
        struct{ s32 r, g;} c;
        s32 S32[2];
    } u;
};

/// \brief Struct contains S32
///
struct DEMO_S32
{
    union
    {
        struct{ s32 x;} v;
        struct{ s32 r;} c;
        s32 S32;
    } u;
};

/// \brief Struct contains U16x4
///
struct DEMO_U16x4
{
    union
    {
        struct{ u16 x, y, z, w;} v;
        struct{ u16 r, g, b, a;} c;
        u16 U16[4];
    } u;
};

/// \brief Struct contains U16x3
///
struct DEMO_U16x3
{
    union
    {
        struct{ u16 x, y, z;} v;
        struct{ u16 r, g, b;} c;
        u16 U16[3];
    } u;
};

/// \brief Struct contains U16x2
///
struct DEMO_U16x2
{
    union
    {
        struct{ u16 x, y;} v;
        struct{ u16 r, g;} c;
        u16 U16[2];
    } u;
};

/// \brief Struct contains U16
///
struct DEMO_U16
{
    union
    {
        struct{ u16 x;} v;
        struct{ u16 r;} c;
        u16 U16;
    } u;
};

/// \brief Struct contains S16x4
///
struct DEMO_S16x4
{
    union
    {
        struct{ s16 x, y, z, w;} v;
        struct{ s16 r, g, b, a;} c;
        s16 S16[4];
    } u;
};

/// \brief Struct contains S16x3
///
struct DEMO_S16x3
{
    union
    {
        struct{ s16 x, y, z;} v;
        struct{ s16 r, g, b;} c;
        s16 S16[3];
    } u;
};

/// \brief Struct contains S16x2
///
struct DEMO_S16x2
{
    union
    {
        struct{ s16 x, y;} v;
        struct{ s16 r, g;} c;
        s16 S16[2];
    } u;
};

/// \brief Struct contains S16
///
struct DEMO_S16
{
    union
    {
        struct{ s16 x;} v;
        struct{ s16 r;} c;
        s16 S16;
    } u;
};

/// \brief Struct contains U8x4
///
struct DEMO_U8x4
{
    union
    {
        struct{ u8 x, y, z, w;} v;
        struct{ u8 r, g, b, a;} c;
        u8 U8[4];
    } u;
};

/// \brief Struct contains U8x3
///
struct DEMO_U8x3
{
    union
    {
        struct{ u8 x, y, z;} v;
        struct{ u8 r, g, b;} c;
        u8 U8[3];
    } u;
};

/// \brief Struct contains U8x2
///
struct DEMO_U8x2
{
    union
    {
        struct{ u8 x, y;} v;
        struct{ u8 r, g;} c;
        u8 U8[2];
    } u;
};

/// \brief Struct contains U8
///
struct DEMO_U8
{
    union
    {
        struct{ u8 x;} v;
        struct{ u8 r;} c;
        u8 U8;
    } u;
};

/// \brief Struct contains S8x4
///
struct DEMO_S8x4
{
    union
    {
        struct{ s8 x, y, z, w;} v;
        struct{ s8 r, g, b, a;} c;
        s8 S8[4];
    } u;
};

/// \brief Struct contains S8x3
///
struct DEMO_S8x3
{
    union
    {
        struct{ s8 x, y, z;} v;
        struct{ s8 r, g, b;} c;
        s8 S8[3];
    } u;
};

/// \brief Struct contains S8x2
///
struct DEMO_S8x2
{
    union
    {
        struct{ s8 x, y;} v;
        struct{ s8 r, g;} c;
        s8 S8[2];
    } u;
};

/// \brief Struct contains S8
///
struct DEMO_S8
{
    union
    {
        struct{ s8 x;} v;
        struct{ s8 r;} c;
        s8 S8;
    } u;
};


//
// Helper types for attributes
//

/// \brief Struct contains f32x4x4
///
struct DEMO_F32x4F32x4F32x4F32x4
{
    DEMO_F32x4 F32x4[4];
};

/// \brief Struct contains f32x4x3
///
struct DEMO_F32x4F32x4F32x4
{
    DEMO_F32x4 F32x4[3];
};

/// \brief Struct contains f32x4x2
///
struct DEMO_F32x4F32x4
{
    DEMO_F32x4 F32x4[2];
};

/// \brief Struct contains f32x3x3
///
struct DEMO_F32x3F32x3F32x3
{
    DEMO_F32x3 F32x3[3];
};

/// \brief Struct contains f32x3x2
///
struct DEMO_F32x3F32x3
{
    DEMO_F32x3 F32x3[2];
};

/// \brief Struct contains f32x3f32x2
///
struct DEMO_F32x3F32x2
{
    DEMO_F32x3 F32x3;
    DEMO_F32x2 F32x2;
};


/// \brief Struct contains f32x4f32x3
///
struct DEMO_F32x4F32x3
{
    DEMO_F32x4 F32x4;
    DEMO_F32x3 F32x3;
};

/// \brief Struct contains f32x3f32x4
///
struct DEMO_F32x3F32x4
{
    DEMO_F32x3 F32x3;
    DEMO_F32x4 F32x4;
};

/// \brief Struct contains f32x3f32x4x2
///
struct DEMO_F32x3F32x4F32x4
{
    DEMO_F32x3 F32x3;
    DEMO_F32x4 F32x4[2];
};


/// \brief Struct contains f32x4x2f32x3
///
struct DEMO_F32x4F32x4F32x3
{
    DEMO_F32x4 F32x4[2];
    DEMO_F32x3 F32x3;
};

/// \brief Struct contains f32x4x3f32x2
///
struct DEMO_F32x4F32x4F32x4F32x2
{
    DEMO_F32x4 F32x4[3];
    DEMO_F32x2 F32x2;
};

/// \brief Struct contains f32x4f32x3x2
///
struct DEMO_F32x4F32x3F32x3
{
    DEMO_F32x4 F32x4;
    DEMO_F32x3 F32x3[2];
};

/// \brief Struct contains f32x4f32x3x2f32x2
///
struct DEMO_F32x4F32x4F32x3F32x2
{
    DEMO_F32x4 F32x4[2];
    DEMO_F32x3 F32x3;
    DEMO_F32x2 F32x2;
};


/// \brief Struct contains f32x4f32x3x2f32x2
///
struct DEMO_F32x4F32x3F32x3F32x2
{
    DEMO_F32x4 F32x4;
    DEMO_F32x3 F32x3[2];
    DEMO_F32x2 F32x2;
};

/// \brief Struct contains f32x4f32x3f32x2
///
struct DEMO_F32x4F32x3F32x2
{
    DEMO_F32x4 F32x4;
    DEMO_F32x3 F32x3;
    DEMO_F32x2 F32x2;
};

/// \brief Struct contains f32x3f32x3f32x2
///
struct DEMO_F32x3F32x3F32x2
{
    DEMO_F32x3 F32x3[2];
    DEMO_F32x2 F32x2;
};

/// \brief Struct contains f32x3f32x3f32x3f32x3f32x2
///
struct DEMO_F32x3F32x3F32x3F32x3F32x2
{
    DEMO_F32x3 F32x3[4];
    DEMO_F32x2 F32x2;
};


//
// tga reader
//

#pragma pack(push,x1) // Byte alignment (8-bit)
#pragma pack(1)

struct DEMOGfxTGAHeader
{
   unsigned char  idSize;
   unsigned char  mapType;
   unsigned char  imageType;
   unsigned short paletteStart;
   unsigned short paletteSize;
   unsigned char  paletteEntryDepth;
   unsigned short x;
   unsigned short y;
   unsigned short dwWidth;
   unsigned short dwHeight;
   unsigned char  colorDepth;
   unsigned char  descriptor;
};

struct DEMOGfxTGAInfo
{
    DEMOGfxTGAHeader header;
#if 0	
    u8         id[256];
    u8         map[256*4]; // always BGRA8 format
    u32        mapSize; // # of entries; typically 256
#endif
};

#pragma pack(pop,x1)
/// @}

#ifdef __cplusplus
}
#endif

#endif // _DEMO_GFX_TYPES_H_
