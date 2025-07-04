/*
* Copyright (c) 2016, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/

#include "lwntest_cpp.h"
#include "lwnTest/lwnTest_Formats.h"
#if defined(LW_TEGRA)
#include "lwn_PrivateFormats.h"
#endif

using namespace lwn;

namespace lwnTest {

// All formats exercised by tests in this file
const FormatDesc ALL_FORMATS[] = {
#define FORMATDESC(fmt, s, t, rb, gb, bb, ab, flags)  { Format::fmt, #fmt, s, COMP_TYPE_ ## t, {rb, gb, bb, ab}, flags }
#define PRIVATEDESC(fmt, s, t, rb, gb, bb, ab, flags) { Format::Enum(LWN_FORMAT_PRIVATE_ ## fmt), #fmt, s, COMP_TYPE_ ## t, {rb, gb, bb, ab}, flags }
    FORMATDESC(R8,                  1 * 1, FLOAT,       8,0,0,0,     FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R8SN,                1 * 1, FLOAT,       8,0,0,0,     FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R8UI,                1 * 1, UNSIGNED,    8,0,0,0,     FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R8I,                 1 * 1, INT,         8,0,0,0,     FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R16F,                1 * 2, FLOAT,       16,0,0,0,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R16,                 1 * 2, FLOAT,       16,0,0,0,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R16SN,               1 * 2, FLOAT,       16,0,0,0,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R16UI,               1 * 2, UNSIGNED,    16,0,0,0,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R16I,                1 * 2, INT,         16,0,0,0,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R32F,                1 * 4, FLOAT,       32,0,0,0,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R32UI,               1 * 4, UNSIGNED,    32,0,0,0,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R32I,                1 * 4, INT,         32,0,0,0,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG8,                 2 * 1, FLOAT,       8,8,0,0,     FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG8SN,               2 * 1, FLOAT,       8,8,0,0,     FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG8UI,               2 * 1, UNSIGNED,    8,8,0,0,     FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG8I,                2 * 1, INT,         8,8,0,0,     FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG16F,               2 * 2, FLOAT,       16,16,0,0,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG16,                2 * 2, FLOAT,       16,16,0,0,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG16SN,              2 * 2, FLOAT,       16,16,0,0,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG16UI,              2 * 2, UNSIGNED,    16,16,0,0,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG16I,               2 * 2, INT,         16,16,0,0,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG32F,               2 * 4, FLOAT,       32,32,0,0,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG32UI,              2 * 4, UNSIGNED,    32,32,0,0,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG32I,               2 * 4, INT,         32,32,0,0,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGB8,                3 * 1, FLOAT,       8,8,8,0,     FLAG_VERTEX),
    FORMATDESC(RGB8SN,              3 * 1, FLOAT,       8,8,8,0,     FLAG_VERTEX),
    FORMATDESC(RGB8UI,              3 * 1, UNSIGNED,    8,8,8,0,     FLAG_VERTEX),
    FORMATDESC(RGB8I,               3 * 1, INT,         8,8,8,0,     FLAG_VERTEX),
    FORMATDESC(RGB16F,              3 * 2, FLOAT,       16,16,16,0,  FLAG_VERTEX),
    FORMATDESC(RGB16,               3 * 2, FLOAT,       16,16,16,0,  FLAG_VERTEX),
    FORMATDESC(RGB16SN,             3 * 2, FLOAT,       16,16,16,0,  FLAG_VERTEX),
    FORMATDESC(RGB16UI,             3 * 2, UNSIGNED,    16,16,16,0,  FLAG_VERTEX),
    FORMATDESC(RGB16I,              3 * 2, INT,         16,16,16,0,  FLAG_VERTEX),
    FORMATDESC(RGB32F,              3 * 4, FLOAT,       32,32,32,0,  FLAG_TEXTURE | FLAG_VERTEX),
    FORMATDESC(RGB32UI,             3 * 4, UNSIGNED,    32,32,32,0,  FLAG_TEXTURE | FLAG_VERTEX),
    FORMATDESC(RGB32I,              3 * 4, INT,         32,32,32,0,  FLAG_TEXTURE | FLAG_VERTEX),
    FORMATDESC(RGBA8,               4 * 1, FLOAT,       8,8,8,8,     FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA32F,             4 * 4, FLOAT,       32,32,32,32, FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA8SN,             4 * 1, FLOAT,       8,8,8,8,     FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA8UI,             4 * 1, UNSIGNED,    8,8,8,8,     FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA8I,              4 * 1, INT,         8,8,8,8,     FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA16F,             4 * 2, FLOAT,       16,16,16,16, FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA16,              4 * 2, FLOAT,       16,16,16,16, FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA16SN,            4 * 2, FLOAT,       16,16,16,16, FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA16UI,            4 * 2, UNSIGNED,    16,16,16,16, FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA16I,             4 * 2, INT,         16,16,16,16, FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA32F,             4 * 4, FLOAT,       32,32,32,32, FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA32UI,            4 * 4, UNSIGNED,    32,32,32,32, FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA32I,             4 * 4, INT,         32,32,32,32, FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    // Depth, stencil, and depth+stencil all swizzle differently. It's not yet
    // known what the final output will look like.
    FORMATDESC(STENCIL8,            0 + 1, FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_STENCIL),
    FORMATDESC(DEPTH16,             2 + 0, FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_DEPTH),
    FORMATDESC(DEPTH24,             4 + 0, FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_DEPTH),
    FORMATDESC(DEPTH32F,            4 + 0, FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_DEPTH),
    FORMATDESC(DEPTH24_STENCIL8,    3 + 1, FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_DEPTH | FLAG_STENCIL),
    FORMATDESC(DEPTH32F_STENCIL8,   4 + 4, FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_DEPTH | FLAG_STENCIL),
    FORMATDESC(RGBX8_SRGB,          4 * 1, FLOAT,       8,8,8,0,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBA8_SRGB,          4 * 1, FLOAT,       8,8,8,0,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBA4,               2,     FLOAT,       4,4,4,4,      FLAG_TEXTURE | FLAG_COPYIMAGE),
    FORMATDESC(RGB5,                2,     FLOAT,       5,5,5,0,      FLAG_TEXTURE | FLAG_COPYIMAGE),
    FORMATDESC(RGB5A1,              2,     FLOAT,       5,5,5,1,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGB565,              2,     FLOAT,       5,6,5,0,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGB10A2,             4,     FLOAT,       10,10,10,2,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGB10A2UI,           4,     UNSIGNED,    10,10,10,2,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R11G11B10F,          4,     FLOAT,       11,11,10,0,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGB9E5F,             4,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE),
    FORMATDESC(RGB_DXT1,            0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGBA_DXT1,           0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGBA_DXT3,           0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGBA_DXT5,           0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGB_DXT1_SRGB,       0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGBA_DXT1_SRGB,      0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGBA_DXT3_SRGB,      0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGBA_DXT5_SRGB,      0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGTC1_UNORM,         0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGTC1_SNORM,         0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGTC2_UNORM,         0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGTC2_SNORM,         0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(BPTC_UNORM,          0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(BPTC_UNORM_SRGB,     0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(BPTC_SFLOAT,         0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(BPTC_UFLOAT,         0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(R8_UI2F,             1 * 1, FLOAT,       8,0,0,0,      FLAG_VERTEX),
    FORMATDESC(R8_I2F,              1 * 1, FLOAT,       8,0,0,0,      FLAG_VERTEX),
    FORMATDESC(R16_UI2F,            1 * 2, FLOAT,       16,0,0,0,     FLAG_VERTEX),
    FORMATDESC(R16_I2F,             1 * 2, FLOAT,       16,0,0,0,     FLAG_VERTEX),
    FORMATDESC(R32_UI2F,            1 * 4, FLOAT,       32,0,0,0,     FLAG_VERTEX),
    FORMATDESC(R32_I2F,             1 * 4, FLOAT,       32,0,0,0,     FLAG_VERTEX),
    FORMATDESC(RG8_UI2F,            2 * 1, FLOAT,       8,8,0,0,      FLAG_VERTEX),
    FORMATDESC(RG8_I2F,             2 * 1, FLOAT,       8,8,0,0,      FLAG_VERTEX),
    FORMATDESC(RG16_UI2F,           2 * 2, FLOAT,       16,16,0,0,    FLAG_VERTEX),
    FORMATDESC(RG16_I2F,            2 * 2, FLOAT,       16,16,0,0,    FLAG_VERTEX),
    FORMATDESC(RG32_UI2F,           2 * 4, FLOAT,       32,32,0,0,    FLAG_VERTEX),
    FORMATDESC(RG32_I2F,            2 * 4, FLOAT,       32,32,0,0,    FLAG_VERTEX),
    FORMATDESC(RGB8_UI2F,           3 * 1, FLOAT,       8,8,8,0,      FLAG_VERTEX),
    FORMATDESC(RGB8_I2F,            3 * 1, FLOAT,       8,8,8,0,      FLAG_VERTEX),
    FORMATDESC(RGB16_UI2F,          3 * 2, FLOAT,       16,16,16,0,   FLAG_VERTEX),
    FORMATDESC(RGB16_I2F,           3 * 2, FLOAT,       16,16,16,0,   FLAG_VERTEX),
    FORMATDESC(RGB32_UI2F,          3 * 4, FLOAT,       32,32,32,0,   FLAG_VERTEX),
    FORMATDESC(RGB32_I2F,           3 * 4, FLOAT,       32,32,32,0,   FLAG_VERTEX),
    FORMATDESC(RGBA8_UI2F,          4 * 1, FLOAT,       8,8,8,8,      FLAG_VERTEX),
    FORMATDESC(RGBA8_I2F,           4 * 1, FLOAT,       8,8,8,8,      FLAG_VERTEX),
    FORMATDESC(RGBA16_UI2F,         4 * 2, FLOAT,       16,16,16,16,  FLAG_VERTEX),
    FORMATDESC(RGBA16_I2F,          4 * 2, FLOAT,       16,16,16,16,  FLAG_VERTEX),
    FORMATDESC(RGBA32_UI2F,         4 * 4, FLOAT,       32,32,32,32,  FLAG_VERTEX),
    FORMATDESC(RGBA32_I2F,          4 * 4, FLOAT,       32,32,32,32,  FLAG_VERTEX),
    FORMATDESC(RGB10A2SN,           4,     FLOAT,       10,10,10,2,   FLAG_VERTEX),
    FORMATDESC(RGB10A2I,            4,     INT,         10,10,10,2,   FLAG_VERTEX),
    FORMATDESC(RGB10A2_UI2F,        4,     FLOAT,       10,10,10,2,   FLAG_VERTEX),
    FORMATDESC(RGB10A2_I2F,         4,     FLOAT,       10,10,10,2,   FLAG_VERTEX),
    FORMATDESC(RGBX8,               4 * 1, FLOAT,       8,8,8,0,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX8SN,             4 * 1, FLOAT,       8,8,8,0,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX8UI,             4 * 1, UNSIGNED,    8,8,8,0,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX8I,              4 * 1, INT,         8,8,8,0,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX16F,             4 * 2, FLOAT,       16,16,16,0,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX16,              4 * 2, FLOAT,       16,16,16,0,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX16SN,            4 * 2, FLOAT,       16,16,16,0,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX16UI,            4 * 2, UNSIGNED,    16,16,16,0,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX16I,             4 * 2, INT,         16,16,16,0,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX32F,             4 * 4, FLOAT,       32,32,32,0,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX32UI,            4 * 4, UNSIGNED,    32,32,32,0,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX32I,             4 * 4, INT,         32,32,32,0,   FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBA_ASTC_4x4,       0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_5x4,       0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_5x5,       0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_6x5,       0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_6x6,       0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_8x5,       0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_8x6,       0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_8x8,       0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_10x5,      0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_10x6,      0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_10x8,      0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_10x10,     0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_12x10,     0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_12x12,     0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_4x4_SRGB,  0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_5x4_SRGB,  0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_5x5_SRGB,  0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_6x5_SRGB,  0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_6x6_SRGB,  0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_8x5_SRGB,  0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_8x6_SRGB,  0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_8x8_SRGB,  0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_10x5_SRGB, 0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_10x6_SRGB, 0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_10x8_SRGB, 0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_10x10_SRGB,0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_12x10_SRGB,0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_12x12_SRGB,0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(BGR5,                2,     FLOAT,       5,5,5,0,      FLAG_TEXTURE | FLAG_COPYIMAGE),
    FORMATDESC(BGR5A1,              2,     FLOAT,       5,5,5,1,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(BGR565,              2,     FLOAT,       5,6,5,0,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(A1BGR5,              2,     FLOAT,       5,5,5,1,      FLAG_TEXTURE | FLAG_COPYIMAGE),
    FORMATDESC(BGRA8,               4 * 1, FLOAT,       8,8,8,8,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(BGRX8,               4 * 1, FLOAT,       8,8,8,0,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(BGRA8_SRGB,          4 * 1, FLOAT,       8,8,8,8,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(BGRX8_SRGB,          4 * 1, FLOAT,       8,8,8,0,      FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
#if defined(LW_TEGRA)
    PRIVATEDESC(RGB_ETC1,           0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_PRIVATE),
    PRIVATEDESC(RGBA_ETC1,          0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_PRIVATE),
    PRIVATEDESC(RGB_ETC1_SRGB,      0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_PRIVATE),
    PRIVATEDESC(RGBA_ETC1_SRGB,     0,     FLOAT,       0,0,0,0,      FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_PRIVATE),
#endif
#undef FORMATDESC
#undef PRIVATEDESC
};

int FormatDesc::numFormats()
{
    return sizeof(ALL_FORMATS) / sizeof(ALL_FORMATS[0]);
}

const FormatDesc *FormatDesc::findByFormat(lwn::Format fmt)
{
    for (int i = 0; i < (int)numFormats(); i++) {
        const FormatDesc *desc = &ALL_FORMATS[i];
        if (desc->format == fmt) {
            return desc;
        }
    }
    assert(!"Unknown LWNformat!");
    return nullptr;
}

const FormatDesc *FormatDesc::findByFormat(LWNformat fmt)
{
    // This is dodgy but works because we know that lwn::Format really just holds an int
    const lwn::Format *tmp = (const lwn::Format *)&fmt;
    return findByFormat(*tmp);
}

} // namespace lwnTest
