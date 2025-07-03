
// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#pragma once

#include <o6/optix.h>


struct TexModes
{
    int index;
    int wrap;
    int filter;
};

const int NUMBER_OF_TEX_MODES            = 12;
TexModes  tex_modes[NUMBER_OF_TEX_MODES] = {
    {0, 0, 0},
    {0, 0, 1},
    {0, 1, 0},
    {0, 1, 1},
    {0, 2, 0},
    {0, 2, 1},
    {0, 3, 0}, /*{0, 3, 1},*/
    {1, 0, 0},
    /*{1, 0, 1},*/ {1, 1, 0},
    {1, 1, 1},
    /*{1, 2, 0},*/ {1, 2, 1},
    /*{1, 3, 0},*/ {1, 3, 1},
};

enum TestLookupKind
{
    TestTexture_size = 0,

    // Tex functions
    TestTexture_tex_1d,
    TestTexture_tex_2d,
    TestTexture_tex_3d,
    TestTexture_tex_a1d,
    TestTexture_tex_a2d,
    TestTexture_tex_lwbe,
    TestTexture_tex_alwbe,

    // Mip level
    TestTexture_texlevel_1d,
    TestTexture_texlevel_2d,
    TestTexture_texlevel_3d,
    TestTexture_texlevel_a1d,
    TestTexture_texlevel_a2d,
    TestTexture_texlevel_lwbe,
    TestTexture_texlevel_alwbe,

    // Mip grad
    TestTexture_texgrad_1d,
    TestTexture_texgrad_2d,
    TestTexture_texgrad_3d,
    TestTexture_texgrad_a1d,
    TestTexture_texgrad_a2d,
    //TestTexture_texgrad_lwbe,
    //TestTexture_texgrad_alwbe,

    // TLD4
    TestTexture_tld4r_2d,
    TestTexture_tld4g_2d,
    TestTexture_tld4b_2d,
    TestTexture_tld4a_2d,

    // TLD/"fetch" functions
    TestTexture_texfetch_1d,
    TestTexture_texfetch_2d,
    TestTexture_texfetch_3d,
    //TestTexture_texfetch_a1d,
    //TestTexture_texfetch_a2d,

    NUMBER_OF_LOOKUP_KINDS
};

const char* READ_MODE_STRINGS[4] = {"elem", "norm", "elem_srgb", "norm_srgb"};

const char* LOOKUP_KIND_STRINGS[NUMBER_OF_LOOKUP_KINDS] = {
    "size",
    //
    "1d", "2d", "3d", "a1d", "a2d", "lwbe", "alwbe",
    //
    "level_1d", "level_2d", "level_3d", "level_a1d", "level_a2d", "level_lwbe", "level_alwbe",
    //
    "grad_1d", "grad_2d", "grad_3d", "grad_a1d", "grad_a2d",
    //grad_lwbe",
    //grad_alwbe",
    //
    "tld4r_2d", "tld4g_2d", "tld4b_2d", "tld4a_2d",
    //
    "fetch_1d", "fetch_2d", "fetch_3d",
    //"fetch_a1d",
    //"fetch_a2d",
};

const int LOOKUP_KIND_DIM[NUMBER_OF_LOOKUP_KINDS] = {
    3,  // TestTexture_size= 0,
    //
    1,  // TestTexture_tex_1d,
    2,  // TestTexture_tex_2d,
    3,  // TestTexture_tex_3d,
    2,  // TestTexture_tex_a1d,
    3,  // TestTexture_tex_a2d,
    3,  // TestTexture_tex_lwbe,
    3,  // TestTexture_tex_alwbe,
    //
    1,  // TestTexture_texlevel_1d,
    2,  // TestTexture_texlevel_2d,
    3,  // TestTexture_texlevel_3d,
    2,  // TestTexture_texlevel_a1d,
    3,  // TestTexture_texlevel_a2d,
    3,  // TestTexture_texlevel_lwbe,
    3,  // TestTexture_texlevel_alwbe,
    //
    1,  // TestTexture_texgrad_1d,
    2,  // TestTexture_texgrad_2d,
    3,  // TestTexture_texgrad_3d,
    2,  // TestTexture_texgrad_a1d,
    3,  // TestTexture_texgrad_a2d,
    //3,   // TestTexture_texgrad_lwbe,
    //3,   // TestTexture_texgrad_alwbe,
    //
    2,  // TestTexture_tld4r_2d,
    2,  // TestTexture_tld4g_2d,
    2,  // TestTexture_tld4b_2d,
    2,  // TestTexture_tld4a_2d,
    //
    1,  // TestTexture_texfetch_1d,
    2,  // TestTexture_texfetch_2d,
    3,  // TestTexture_texfetch_3d,
        //2,   // TestTexture_texfetch_a1d,
        //3,   // TestTexture_texfetch_a2d,
};

const int LOOKUP_KIND_FLAGS[NUMBER_OF_LOOKUP_KINDS] = {
    0,                                      // TestTexture_size= 0,
                                            //
    0,                                      // TestTexture_tex_1d,
    0,                                      // TestTexture_tex_2d,
    0,                                      // TestTexture_tex_3d,
    RT_BUFFER_LAYERED,                      // TestTexture_tex_a1d,
    RT_BUFFER_LAYERED,                      // TestTexture_tex_a2d,
    RT_BUFFER_LWBEMAP,                      // TestTexture_tex_lwbe,
    RT_BUFFER_LWBEMAP | RT_BUFFER_LAYERED,  // TestTexture_tex_alwbe,
                                            //
    0,                                      // TestTexture_texlevel_1d,
    0,                                      // TestTexture_texlevel_2d,
    0,                                      // TestTexture_texlevel_3d,
    RT_BUFFER_LAYERED,                      // TestTexture_texlevel_a1d,
    RT_BUFFER_LAYERED,                      // TestTexture_texlevel_a2d,
    RT_BUFFER_LWBEMAP,                      // TestTexture_texlevel_lwbe,
    RT_BUFFER_LWBEMAP | RT_BUFFER_LAYERED,  // TestTexture_texlevel_alwbe,
                                            //
    0,                                      // TestTexture_texgrad_1d,
    0,                                      // TestTexture_texgrad_2d,
    0,                                      // TestTexture_texgrad_3d,
    RT_BUFFER_LAYERED,                      // TestTexture_texgrad_a1d,
    RT_BUFFER_LAYERED,                      // TestTexture_texgrad_a2d,
    //RT_BUFFER_LWBEMAP,                     // TestTexture_texgrad_lwbe,
    //RT_BUFFER_LWBEMAP | RT_BUFFER_LAYERED, // TestTexture_texgrad_alwbe,
    //
    0,  // TestTexture_tld4r_2d,
    0,  // TestTexture_tld4g_2d,
    0,  // TestTexture_tld4b_2d,
    0,  // TestTexture_tld4a_2d,
    //
    0,  // TestTexture_texfetch_1d,
    0,  // TestTexture_texfetch_2d,
    0,  // TestTexture_texfetch_3d,
        //RT_BUFFER_LAYERED,                     // TestTexture_texfetch_a1d,
        //RT_BUFFER_LAYERED,                     // TestTexture_texfetch_a2d,
};

const int NUMBER_OF_FORMATS          = 24;
RTformat  FORMATS[NUMBER_OF_FORMATS] = {
    RT_FORMAT_HALF,   RT_FORMAT_FLOAT,           RT_FORMAT_BYTE,  RT_FORMAT_UNSIGNED_BYTE,
    RT_FORMAT_SHORT,  RT_FORMAT_UNSIGNED_SHORT,  RT_FORMAT_INT,   RT_FORMAT_UNSIGNED_INT,
    RT_FORMAT_HALF2,  RT_FORMAT_FLOAT2,          RT_FORMAT_BYTE2, RT_FORMAT_UNSIGNED_BYTE2,
    RT_FORMAT_SHORT2, RT_FORMAT_UNSIGNED_SHORT2, RT_FORMAT_INT2,  RT_FORMAT_UNSIGNED_INT2,
    RT_FORMAT_HALF4,  RT_FORMAT_FLOAT4,          RT_FORMAT_BYTE4, RT_FORMAT_UNSIGNED_BYTE4,
    RT_FORMAT_SHORT4, RT_FORMAT_UNSIGNED_SHORT4, RT_FORMAT_INT4,  RT_FORMAT_UNSIGNED_INT4};


const int NUMBER_OF_FORMATS_AND_READ_MODES = 48;
struct FormatAndReadMode
{
    RTformat          inputFmt;
    RTtexturereadmode readMode;
} formatsAndReadmodes[NUMBER_OF_FORMATS_AND_READ_MODES] = {
    {RT_FORMAT_UNSIGNED_BYTE4, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_UNSIGNED_BYTE2, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_UNSIGNED_BYTE, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_BYTE4, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_BYTE2, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_BYTE, RT_TEXTURE_READ_ELEMENT_TYPE},

    {RT_FORMAT_UNSIGNED_BYTE4, RT_TEXTURE_READ_ELEMENT_TYPE_SRGB},
    {RT_FORMAT_UNSIGNED_BYTE2, RT_TEXTURE_READ_ELEMENT_TYPE_SRGB},
    {RT_FORMAT_UNSIGNED_BYTE, RT_TEXTURE_READ_ELEMENT_TYPE_SRGB},

    {RT_FORMAT_UNSIGNED_SHORT4, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_UNSIGNED_SHORT2, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_UNSIGNED_SHORT, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_SHORT4, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_SHORT2, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_SHORT, RT_TEXTURE_READ_ELEMENT_TYPE},

    {RT_FORMAT_UNSIGNED_INT4, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_UNSIGNED_INT2, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_UNSIGNED_INT, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_INT4, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_INT2, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_INT, RT_TEXTURE_READ_ELEMENT_TYPE},

    {RT_FORMAT_FLOAT4, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_FLOAT2, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_FLOAT, RT_TEXTURE_READ_ELEMENT_TYPE},

    {RT_FORMAT_UNSIGNED_BYTE4, RT_TEXTURE_READ_NORMALIZED_FLOAT},
    {RT_FORMAT_UNSIGNED_BYTE2, RT_TEXTURE_READ_NORMALIZED_FLOAT},
    {RT_FORMAT_UNSIGNED_BYTE, RT_TEXTURE_READ_NORMALIZED_FLOAT},
    {RT_FORMAT_BYTE4, RT_TEXTURE_READ_NORMALIZED_FLOAT},
    {RT_FORMAT_BYTE2, RT_TEXTURE_READ_NORMALIZED_FLOAT},
    {RT_FORMAT_BYTE, RT_TEXTURE_READ_NORMALIZED_FLOAT},

    {RT_FORMAT_UNSIGNED_BYTE4, RT_TEXTURE_READ_NORMALIZED_FLOAT_SRGB},
    {RT_FORMAT_UNSIGNED_BYTE2, RT_TEXTURE_READ_NORMALIZED_FLOAT_SRGB},
    {RT_FORMAT_UNSIGNED_BYTE, RT_TEXTURE_READ_NORMALIZED_FLOAT_SRGB},

    {RT_FORMAT_UNSIGNED_SHORT4, RT_TEXTURE_READ_NORMALIZED_FLOAT},
    {RT_FORMAT_UNSIGNED_SHORT2, RT_TEXTURE_READ_NORMALIZED_FLOAT},
    {RT_FORMAT_UNSIGNED_SHORT, RT_TEXTURE_READ_NORMALIZED_FLOAT},
    {RT_FORMAT_SHORT4, RT_TEXTURE_READ_NORMALIZED_FLOAT},
    {RT_FORMAT_SHORT2, RT_TEXTURE_READ_NORMALIZED_FLOAT},
    {RT_FORMAT_SHORT, RT_TEXTURE_READ_NORMALIZED_FLOAT},

    {RT_FORMAT_FLOAT4, RT_TEXTURE_READ_NORMALIZED_FLOAT},
    {RT_FORMAT_FLOAT2, RT_TEXTURE_READ_NORMALIZED_FLOAT},
    {RT_FORMAT_FLOAT, RT_TEXTURE_READ_NORMALIZED_FLOAT},

    {RT_FORMAT_HALF4, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_HALF2, RT_TEXTURE_READ_ELEMENT_TYPE},
    {RT_FORMAT_HALF, RT_TEXTURE_READ_ELEMENT_TYPE},

    {RT_FORMAT_HALF4, RT_TEXTURE_READ_NORMALIZED_FLOAT},
    {RT_FORMAT_HALF2, RT_TEXTURE_READ_NORMALIZED_FLOAT},
    {RT_FORMAT_HALF, RT_TEXTURE_READ_NORMALIZED_FLOAT}};
