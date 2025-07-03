/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#if defined(LW_TEGRA)
#include "lwn_PrivateFormats.h"
#endif

#define DEBUG_MODE 0
#if DEBUG_MODE
    #define DEBUG_PRINT(x) do { \
        printf x; \
        fflush(stdout); \
    } while (0)
#else
    #define DEBUG_PRINT(x)
#endif

using namespace lwn;

#define BLOCK_DIV(N, S) (((N) + (S) - 1) / (S))
#define ROUND_UP(N, S) (BLOCK_DIV(N, S) * (S))
#define ROUND_DN(N, S) ((N) - ((N) % (S)))

static const uint32_t POOL_SIZE = 32 * 1024 * 1024;

// All formats exercised by tests in this file
const FormatDesc ALL_FORMATS[] = {
#define FORMATDESC(fmt, s, t, flags) { Format::fmt, #fmt, s, COMP_TYPE_ ## t, {0,0,0,0}, flags }
#define PRIVATEDESC(fmt, s, t, flags){ Format::Enum(LWN_FORMAT_PRIVATE_ ## fmt), #fmt, s, COMP_TYPE_ ## t, {0,0,0,0}, flags }
    FORMATDESC(R8,                  1 * 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R8SN,                1 * 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R8UI,                1 * 1, UNSIGNED,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R8I,                 1 * 1, INT,         FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R16F,                1 * 2, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R16,                 1 * 2, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R16SN,               1 * 2, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R16UI,               1 * 2, UNSIGNED,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R16I,                1 * 2, INT,         FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R32F,                1 * 4, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R32UI,               1 * 4, UNSIGNED,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R32I,                1 * 4, INT,         FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG8,                 2 * 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG8SN,               2 * 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG8UI,               2 * 1, UNSIGNED,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG8I,                2 * 1, INT,         FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG16F,               2 * 2, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG16,                2 * 2, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG16SN,              2 * 2, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG16UI,              2 * 2, UNSIGNED,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG16I,               2 * 2, INT,         FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG32F,               2 * 4, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG32UI,              2 * 4, UNSIGNED,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RG32I,               2 * 4, INT,         FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGB8,                3 * 1, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGB8SN,              3 * 1, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGB8UI,              3 * 1, UNSIGNED,    FLAG_VERTEX),
    FORMATDESC(RGB8I,               3 * 1, INT,         FLAG_VERTEX),
    FORMATDESC(RGB16F,              3 * 2, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGB16,               3 * 2, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGB16SN,             3 * 2, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGB16UI,             3 * 2, UNSIGNED,    FLAG_VERTEX),
    FORMATDESC(RGB16I,              3 * 2, INT,         FLAG_VERTEX),
    FORMATDESC(RGB32F,              3 * 4, FLOAT,       FLAG_TEXTURE | FLAG_VERTEX),
    FORMATDESC(RGB32UI,             3 * 4, UNSIGNED,    FLAG_TEXTURE | FLAG_VERTEX),
    FORMATDESC(RGB32I,              3 * 4, INT,         FLAG_TEXTURE | FLAG_VERTEX),
    FORMATDESC(RGBA8,               4 * 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA8SN,             4 * 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA8UI,             4 * 1, UNSIGNED,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA8I,              4 * 1, INT,         FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA16F,             4 * 2, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA16,              4 * 2, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA16SN,            4 * 2, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA16UI,            4 * 2, UNSIGNED,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA16I,             4 * 2, INT,         FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA32F,             4 * 4, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA32UI,            4 * 4, UNSIGNED,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGBA32I,             4 * 4, INT,         FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    // Depth, stencil, and depth+stencil all swizzle differently. It's not yet
    // known what the final output will look like.
    FORMATDESC(STENCIL8,            0 + 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_STENCIL),
    FORMATDESC(DEPTH16,             2 + 0, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_DEPTH),
    FORMATDESC(DEPTH24,             4 + 0, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_DEPTH),
    FORMATDESC(DEPTH32F,            4 + 0, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_DEPTH),
    FORMATDESC(DEPTH24_STENCIL8,    3 + 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_DEPTH | FLAG_STENCIL),
    FORMATDESC(DEPTH32F_STENCIL8,   4 + 4, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_DEPTH | FLAG_STENCIL),
    FORMATDESC(RGBX8_SRGB,          4 * 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBA8_SRGB,          4 * 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBA4,               2,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE),
    FORMATDESC(RGB5,                2,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE),
    FORMATDESC(RGB5A1,              2,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE),
    FORMATDESC(RGB565,              2,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE),
    FORMATDESC(RGB10A2,             4,     FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGB10A2UI,           4,     UNSIGNED,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(R11G11B10F,          4,     FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE | FLAG_VERTEX),
    FORMATDESC(RGB9E5F,             4,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE),
    FORMATDESC(RGB_DXT1,            0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGBA_DXT1,           0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGBA_DXT3,           0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGBA_DXT5,           0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGB_DXT1_SRGB,       0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGBA_DXT1_SRGB,      0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGBA_DXT3_SRGB,      0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGBA_DXT5_SRGB,      0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGTC1_UNORM,         0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGTC1_SNORM,         0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGTC2_UNORM,         0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(RGTC2_SNORM,         0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(BPTC_UNORM,          0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(BPTC_UNORM_SRGB,     0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(BPTC_SFLOAT,         0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(BPTC_UFLOAT,         0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    FORMATDESC(R8_UI2F,             1 * 1, FLOAT,       FLAG_VERTEX),
    FORMATDESC(R8_I2F,              1 * 1, FLOAT,       FLAG_VERTEX),
    FORMATDESC(R16_UI2F,            1 * 2, FLOAT,       FLAG_VERTEX),
    FORMATDESC(R16_I2F,             1 * 2, FLOAT,       FLAG_VERTEX),
    FORMATDESC(R32_UI2F,            1 * 4, FLOAT,       FLAG_VERTEX),
    FORMATDESC(R32_I2F,             1 * 4, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RG8_UI2F,            2 * 1, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RG8_I2F,             2 * 1, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RG16_UI2F,           2 * 2, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RG16_I2F,            2 * 2, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RG32_UI2F,           2 * 4, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RG32_I2F,            2 * 4, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGB8_UI2F,           3 * 1, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGB8_I2F,            3 * 1, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGB16_UI2F,          3 * 2, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGB16_I2F,           3 * 2, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGB32_UI2F,          3 * 4, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGB32_I2F,           3 * 4, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGBA8_UI2F,          4 * 1, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGBA8_I2F,           4 * 1, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGBA16_UI2F,         4 * 2, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGBA16_I2F,          4 * 2, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGBA32_UI2F,         4 * 4, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGBA32_I2F,          4 * 4, FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGB10A2SN,           4,     FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGB10A2I,            4,     INT,         FLAG_VERTEX),
    FORMATDESC(RGB10A2_UI2F,        4,     FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGB10A2_I2F,         4,     FLOAT,       FLAG_VERTEX),
    FORMATDESC(RGBX8,               4 * 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX8SN,             4 * 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX8UI,             4 * 1, UNSIGNED,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX8I,              4 * 1, INT,         FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX16F,             4 * 2, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX16,              4 * 2, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX16SN,            4 * 2, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX16UI,            4 * 2, UNSIGNED,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX16I,             4 * 2, INT,         FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX32F,             4 * 4, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX32UI,            4 * 4, UNSIGNED,    FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBX32I,             4 * 4, INT,         FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(RGBA_ASTC_4x4,       0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_5x4,       0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_5x5,       0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_6x5,       0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_6x6,       0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_8x5,       0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_8x6,       0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_8x8,       0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_10x5,      0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_10x6,      0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_10x8,      0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_10x10,     0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_12x10,     0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_12x12,     0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_4x4_SRGB,  0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_5x4_SRGB,  0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_5x5_SRGB,  0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_6x5_SRGB,  0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_6x6_SRGB,  0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_8x5_SRGB,  0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_8x6_SRGB,  0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_8x8_SRGB,  0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_10x5_SRGB, 0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_10x6_SRGB, 0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_10x8_SRGB, 0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_10x10_SRGB,0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_12x10_SRGB,0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(RGBA_ASTC_12x12_SRGB,0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED | FLAG_ASTC),
    FORMATDESC(BGR565,              2,     FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(BGR5,                2,     FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(BGR5A1,              2,     FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(A1BGR5,              2,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE),
    FORMATDESC(BGRX8,               4 * 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(BGRA8,               4 * 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(BGRX8_SRGB,          4 * 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
    FORMATDESC(BGRA8_SRGB,          4 * 1, FLOAT,       FLAG_TEXTURE | FLAG_RENDER | FLAG_COPYIMAGE),
#if defined(LW_TEGRA)
    PRIVATEDESC(RGB_ETC1,           0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    PRIVATEDESC(RGBA_ETC1,          0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    PRIVATEDESC(RGB_ETC1_SRGB,      0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
    PRIVATEDESC(RGBA_ETC1_SRGB,     0,     FLOAT,       FLAG_TEXTURE | FLAG_COPYIMAGE | FLAG_COMPRESSED),
#endif

#undef FORMATDESC
};

static const int NUM_FORMATS = int(__GL_ARRAYSIZE(ALL_FORMATS));

struct TestParameters
{
    struct {
        const FormatDesc *fd;
        TextureTarget   target;
        uint32_t        width, height, depth, levels;
    } tex;
    struct {
        uint32_t        level;
        CopyRegion      region;
        uint32_t        rowStride, imgStride;
    } copy;
};


class LWNCopyCompressibleTest {
private:
    bool use2D;
public:
    LWNCopyCompressibleTest(bool _use2D) : use2D(_use2D) {}
    LWNTEST_CppMethods();
};


int LWNCopyCompressibleTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(52, 24);
}

lwString LWNCopyCompressibleTest::getDescription() const
{
    const char *without = use2D ? "" : "out";
    lwStringBuf sb;
    sb <<   "Tests copying from a buffer to a compressible texture using\n"
            "lwnCommandBufferCopyBufferToTexture with" << without << " the lwn::CopyFlags::ENGINE_2D\n"
            "flag, and then uses lwnCommandBufferCopyTextureToBuffer to read back\n"
            "the results. This test tests 9 different configurations per supported\n"
            "texture format, varying in texture type, texture dimensions, levels and size,\n"
            "as well as the level and region copied.\n";
    return sb.str();
}

static void initStamp(Format fmt, uint32_t elemStride, void *elemStamp)
{
    static int seed = 0;
#define SEED(mask) \
    ((seed++ * 0xbeef7531) >> 3 & mask)

    // When using lwn::CopyFlags::ENGINE_2D, processing is content-aware,
    // and will modify the test data for some formats. (For example, it will
    // zero out the X components in an RGBX format.) In order to simplify the
    // comparison step afterwards, we pick test data that won't be modified.
    // (note: the number of special cases here is probably overkill.)
    uint8_t *as8 = (uint8_t *) elemStamp;
    uint16_t *as16 = (uint16_t *) elemStamp;
    uint32_t *as32 = (uint32_t *) elemStamp;
    switch(fmt) {
        case lwn::Format::RGBA16F:
            as16[3] = 0x3C00 | SEED(0xFF);
            as16[2] = 0x3C00 | SEED(0xFF);
        case lwn::Format::RG16F:
            as16[1] = 0x3C00 | SEED(0xFF);
        case lwn::Format::R16F:
            as16[0] = 0x3C00 | SEED(0xFF);
            break;

        case lwn::Format::RGBX16F:
            as16[3] = 0x0;
            as16[2] = 0x3C00 | SEED(0xFF);
            as16[1] = 0x3C00 | SEED(0xFF);
            as16[0] = 0x3C00 | SEED(0xFF);
            break;

        case lwn::Format::RGBA32F:
            as32[3] = 0x3F800000 | SEED(0xFFFF);
            as32[2] = 0x3F800000 | SEED(0xFFFF);
        case lwn::Format::RG32F:
            as32[1] = 0x3F800000 | SEED(0xFFFF);
        case lwn::Format::R32F:
            as32[0] = 0x3F800000 | SEED(0xFFFF);
            break;

        case lwn::Format::RGBX32F:
            as32[3] = 0x0;
            as32[2] = 0x3F800000 | SEED(0xFFFF);
            as32[1] = 0x3F800000 | SEED(0xFFFF);
            as32[0] = 0x3F800000 | SEED(0xFFFF);
            break;

        case lwn::Format::RGBX8UI:
        case lwn::Format::RGBX8I:
        case lwn::Format::RGBX8:
        case lwn::Format::RGBX8SN:
        case lwn::Format::RGBX8_SRGB:
        case lwn::Format::BGRX8:
        case lwn::Format::BGRX8_SRGB:
            as8[3] = 0x0;
            as8[2] = 0x70 | SEED(0xF);
            as8[1] = 0x70 | SEED(0xF);
            as8[0] = 0x70 | SEED(0xF);
            break;

        case lwn::Format::BGR5:
            as16[0] = 0x7000 | SEED(0xFFF);
            break;

        case lwn::Format::RGBX16UI:
        case lwn::Format::RGBX16I:
        case lwn::Format::RGBX16:
        case lwn::Format::RGBX16SN:
            as16[3] = 0x0;
            as16[2] = 0x7000 | SEED(0xFFF);
            as16[1] = 0x7000 | SEED(0xFFF);
            as16[0] = 0x7000 | SEED(0xFFF);
            break;

        case lwn::Format::RGBX32UI:
        case lwn::Format::RGBX32I:
            as32[3] = 0x0;
            as32[2] = 0x70000000 | SEED(0xFFFFFFF);
            as32[1] = 0x70000000 | SEED(0xFFFFFFF);
            as32[0] = 0x70000000 | SEED(0xFFFFFFF);
            break;

        default:
            // all bits are used for the other formats, and none of the values
            // should be reinterpreted/changed by CopyBufferToTexture.
            memset(elemStamp, 0x70 | SEED(0xF), elemStride);
            break;
    }
}

static void memStamp(void *ptr, void *stamp, uint32_t stride, uint32_t length)
{
    // Stamps 'length' oclwrrences of 'stamp' to 'ptr'. The 'stamp' length is
    // 'stride'.
    uint8_t *dst = (uint8_t *) ptr;
    for (uint32_t i=0; i<length; i++) {
        memcpy(dst, stamp, stride);
        dst += stride;
    }
}

#if DEBUG_MODE
static const char *TargetStr(TextureTarget tgt)
{
    switch(tgt) {
        case TextureTarget::TARGET_1D:
            return "1D";
        case TextureTarget::TARGET_1D_ARRAY:
            return "1D_ARRAY";
        case TextureTarget::TARGET_2D:
            return "2D";
        case TextureTarget::TARGET_2D_ARRAY:
            return "2D_ARRAY";
        case TextureTarget::TARGET_LWBEMAP:
            return "LWBEMAP";
        case TextureTarget::TARGET_LWBEMAP_ARRAY:
            return "LWBEMAP_ARRAY";
        case TextureTarget::TARGET_3D:
            return "3D";
        default:
            assert(false);
    }
    return "(error)";
}
#endif

static bool subtest(struct TestParameters &param, MemoryPoolAllocator &texAllocator,
                    Buffer &upBuffer, Buffer &dnBuffer, bool use2D)
{
    DEBUG_PRINT(("%s: %s %ux%ux%u, %u levels, copying level %u, region %u,%u,%u %ux%ux%u: ",
                param.tex.fd->formatName, TargetStr(param.tex.target),
                param.tex.width, param.tex.height, param.tex.depth,
                param.tex.levels,
                param.copy.level,
                param.copy.region.xoffset, param.copy.region.yoffset, param.copy.region.zoffset,
                param.copy.region.width, param.copy.region.height, param.copy.region.depth));
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();
    uint8_t *upPtr = (uint8_t *) upBuffer.Map();
    uint8_t *dnPtr = (uint8_t *) dnBuffer.Map();
    BufferAddress upAddr = upBuffer.GetAddress();
    BufferAddress dnAddr = dnBuffer.GetAddress();
    bool pass = true;

    uint32_t elemStride = param.tex.fd->stride;

    // offset by elemStride
    upPtr += elemStride;
    upAddr += elemStride;

    // create the texture
    TextureBuilder tb;
    tb.SetDevice(device)
      .SetDefaults()
      .SetFlags(TextureFlags::COMPRESSIBLE)
      .SetTarget(param.tex.target)
      .SetSize3D(param.tex.width, param.tex.height, param.tex.depth)
      .SetFormat(param.tex.fd->format)
      .SetLevels(param.tex.levels);
    assert(POOL_SIZE >= tb.GetStorageSize());
    Texture *tex = texAllocator.allocTexture(&tb);

    // If a texture is compressible, we must clear the texture first, since we
    // might only update a subsection with CopyBufferToTexture, and the
    // contents need to be cleared beforehand in order to be valid. (Since
    // compressible texture formats are a subset of the renderable formats,
    // we can just clear all the renderable formats before use.)
    TextureView texView;
    if (param.tex.fd->flags & FLAG_RENDER)
    {
        uint32_t lwbeMultiplier = (param.tex.target == TextureTarget::TARGET_LWBEMAP) ? 6 : 1;
        uint32_t clearColor[4];
        memset(clearColor, 0, sizeof(clearColor));

        CopyRegion region;
        region.xoffset = 0;
        region.yoffset = 0;
        region.zoffset = 0;
        region.height = param.tex.height;
        region.depth = param.tex.depth * lwbeMultiplier;

        for (uint32_t level=0; level<param.tex.levels; level++) {
            region.width = LW_MAX(1, param.tex.width >> level);
            if (param.tex.target != TextureTarget::TARGET_1D_ARRAY) {
                region.height = LW_MAX(1, param.tex.height >> level);
            }
            if (param.tex.target == TextureTarget::TARGET_3D) {
                region.depth = LW_MAX(1, param.tex.depth >> level);
            }

            // init texture view
            texView.SetDefaults()
                   .SetLevels(level, 1);
            // clear to 0
            queueCB.ClearTextureui(tex, &texView, &region, clearColor, ClearColorMask::RGBA);
            if (queueCB.GetCommandMemoryFree() <= 0x100000) {
                queueCB.submit();
                g_lwnTracker->insertFence(queue);
            }
        }
    }

    uint32_t width = param.copy.region.width;
    uint32_t height = param.copy.region.height;
    uint32_t depth = param.copy.region.depth;
    if (param.tex.target == TextureTarget::TARGET_1D_ARRAY) {
        depth = height;
        height = 1;
    }
    uint32_t rowStride = param.copy.rowStride ? param.copy.rowStride : width * elemStride;
    uint32_t imgStride = param.copy.imgStride ? param.copy.imgStride : height * rowStride;

    // generate test data
    uint8_t elemStamp[128];
    memset(upPtr, 0x88, depth * imgStride);
    uint32_t y, z;
    for (z=0; z<depth; z++) {
        for (y=0; y<height; y++) {
            void *rowAddr = &upPtr[(z * imgStride) + (y * rowStride)];
            initStamp(param.tex.fd->format, elemStride, elemStamp);
            memStamp(rowAddr, elemStamp, elemStride, width);
        }
    }

    // upload to texture
    texView.SetDefaults()
           .SetLevels(param.copy.level, 1);
    queueCB.SetCopyRowStride(param.copy.rowStride);
    queueCB.SetCopyImageStride(param.copy.imgStride);
    queueCB.CopyBufferToTexture(upAddr, tex, &texView, &param.copy.region,
                                use2D ? lwn::CopyFlags::ENGINE_2D : 0);

    // download back to another buffer
    queueCB.SetCopyRowStride(0);
    queueCB.SetCopyImageStride(0);
    queueCB.CopyTextureToBuffer(tex, &texView, &param.copy.region, dnAddr, 0);
    queueCB.submit();
    queue->Finish();

    // compare the buffers
    uint32_t packedRowStride = width * elemStride;
    uint32_t packedImgStride = height * packedRowStride;
    bool fulltest = true;
    for (z=0; z<depth; z++) {
        for (y=0; y<height; y++) {
            uint32_t upOffset = (z * imgStride) + (y * rowStride);
            uint32_t dnOffset = (z * packedImgStride) + (y * packedRowStride);
            if (memcmp(&upPtr[upOffset], &dnPtr[dnOffset], width * elemStride) != 0) {
#if DEBUG_MODE
                if (fulltest) {
                    uint32_t x;
                    for (x=0; x<width; x++) {
                        if (memcmp(&upPtr[upOffset + x*elemStride], &dnPtr[dnOffset + x*elemStride], elemStride) != 0) {
                            DEBUG_PRINT(("x=%lu y=%lu z=%lu, expected: ", x, y, z));
                            for (uint32_t e=0; e<elemStride; e++) {
                                DEBUG_PRINT(("%02x", upPtr[upOffset + x*elemStride + e]));
                            }
                            DEBUG_PRINT((", got: "));
                            for (uint32_t e=0; e<elemStride; e++) {
                                DEBUG_PRINT(("%02x", dnPtr[dnOffset + x*elemStride + e]));
                            }
                            DEBUG_PRINT(("\n"));
                        }
                    }
                }
#endif
                pass = false;
                if (!fulltest)
                    goto fail;
            }
        }
    }
fail:
    DEBUG_PRINT(("%s\n", pass ? "PASSED" : "FAILED"));

    texAllocator.freeTexture(tex);

    return pass;
}




void LWNCopyCompressibleTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    uint32_t pitchAlignment = 32;

    MemoryPoolAllocator texAllocator(device, NULL, POOL_SIZE, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    MemoryPoolAllocator bufAllocator(device, NULL, POOL_SIZE*3, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);

    // Create buffers
    BufferBuilder bb;
    bb.SetDevice(device)
      .SetDefaults();
    Buffer *bufferUp = bufAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, POOL_SIZE);
    Buffer *bufferDn = bufAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, POOL_SIZE);

    bool pass = true;

    for (uint32_t i=0; i<NUM_FORMATS; i++) {
        struct TestParameters param;
        bool formatPassed = true;
        const FormatDesc &fd = ALL_FORMATS[i];
        if (!(fd.flags & FLAG_COPYIMAGE)) {
            continue;
        }
        if (fd.flags & FLAG_COMPRESSED) {
            continue;
        }
        if (fd.flags & (FLAG_DEPTH | FLAG_STENCIL)) {
            continue;
        }

        param.tex.target = TextureTarget::TARGET_2D_ARRAY;
        param.tex.fd = &fd;

        // Test a 1x1x2 texture
        param.tex.width = 1;
        param.tex.height = 1;
        param.tex.depth = 2;
        param.tex.levels = 1;
        param.copy.level = 0;
        param.copy.region.xoffset = 0;
        param.copy.region.yoffset = 0;
        param.copy.region.zoffset = 0;
        param.copy.region.width = 1;
        param.copy.region.height = 1;
        param.copy.region.depth = 2;
        param.copy.rowStride = ROUND_UP(fd.stride, pitchAlignment);
        param.copy.imgStride = ROUND_UP((param.copy.rowStride * param.copy.region.height) + 1, pitchAlignment);
        formatPassed &= subtest(param, texAllocator, *bufferUp, *bufferDn, use2D);

        // Test the smallest possible width that doesn't require us to set
        // pitch and image alignment
        uint32_t elemSize = fd.stride;
        param.tex.width = 1;
        while (elemSize & (pitchAlignment-1)) {
            param.tex.width <<= 1;
            elemSize <<= 1;
        }
        param.tex.height = param.tex.width;
        param.tex.depth = 2;
        param.copy.region.width = param.tex.width;
        param.copy.region.height = param.tex.height;
        param.copy.region.depth = 2;
        param.copy.rowStride = 0;
        param.copy.imgStride = 0;
        formatPassed &= subtest(param, texAllocator, *bufferUp, *bufferDn, use2D);

        TestParameters cases[] =
        {
            { { NULL, TextureTarget::TARGET_1D_ARRAY,     1426, 20, 1, 1 },   { 0, { 4, 5, 0,  1200, 10, 1 }, 0, 0 } },
            { { NULL, TextureTarget::TARGET_1D_ARRAY,     128, 64, 1,  2 },   { 1, { 2, 4, 0,  54, 20, 1   }, 0, 0 } },
            { { NULL, TextureTarget::TARGET_2D,           713, 83, 1,  5 },   { 0, { 1, 1, 0,  711, 81, 1  }, 0, 0 } },
            { { NULL, TextureTarget::TARGET_2D_ARRAY,     320, 200, 2, 3 },   { 2, { 1, 1, 0,  78, 48, 2   }, 0, 0 } },
            { { NULL, TextureTarget::TARGET_3D,           34, 34, 4,   3 },   { 1, { 4, 2, 1,  8, 12, 1    }, 0, 0 } },
            { { NULL, TextureTarget::TARGET_LWBEMAP,      64, 64, 1,   4 },   { 2, { 1, 1, 1,  14, 14, 5   }, 0, 0 } },
            { { NULL, TextureTarget::TARGET_LWBEMAP_ARRAY,64, 64, 12,  4 },   { 1, { 2, 2, 7,  28, 28, 4   }, 0, 0 } },

        };
        for (uint32_t c=0; c<__GL_ARRAYSIZE(cases); c++) {
            param = cases[c];
            param.tex.fd = &fd;
            param.copy.rowStride = ROUND_UP(fd.stride * param.copy.region.width, pitchAlignment);
            param.copy.imgStride = 0;
            formatPassed &= subtest(param, texAllocator, *bufferUp, *bufferDn, use2D);
        }

        pass &= formatPassed;
    }

    queueCB.ClearColor(0, pass ? 0.0 : 1.0, pass ? 1.0 : 0.0, 0.0);
    queueCB.submit();
    queue->Finish();

    bufAllocator.freeBuffer(bufferDn);
    bufAllocator.freeBuffer(bufferUp);
}



OGTEST_CppTest(LWNCopyCompressibleTest, lwn_copy_compressible_on, (true));
OGTEST_CppTest(LWNCopyCompressibleTest, lwn_copy_compressible_off, (false));

