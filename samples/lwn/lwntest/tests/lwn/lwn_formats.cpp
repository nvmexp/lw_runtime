/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

// lwn_formats.cpp
//
// Test usages of all formats rendering with noisy data and checking against expected results.
// Use the data to upload to all texture targets and interpreting as all compatible formats.
// Alternatively, use the data as vertex attributes representing color in all compatible formats.

#include <vector>

#include "lwntest_cpp.h"
#include "float_util.h"
#include "lwn_utils.h"
#include "g_lwn_formats_data.h"

#if defined(LW_TEGRA)
#include "lwn_PrivateFormats.h"
#endif

// Enables debug printfs
#define DEBUG_MODE 0

// if #defined, display the results of rendering with this format.
// #define DEBUG_FORMAT Format::RGBX8

// If defined, generate GL data
//#define GENERATE_GL_DATA 1

#if defined(GENERATE_GL_DATA)
#include "GL/gl.h"
#include "GL/glext.h"
#endif

#if DEBUG_MODE
#define DEBUG_PRINT(x) \
    do { \
        printf x; \
        fflush(stdout); \
    } while (0)
#else
#define DEBUG_PRINT(x)
#endif

#ifndef ROUND_UP
    #define ROUND_UP(N, S) ((((N) + (S) - 1) / (S)) * (S))
#endif

using namespace lwn;
using namespace lwn::dt;

namespace {

static const int DEFAULT_WIDTH = 16;
static const int DEFAULT_HEIGHT = 16;
static const int DEFAULT_DEPTH = 4;

enum TestResult {
    TEST_PASS,
    TEST_FAIL,
    TEST_UNSUPPORTED,
};

static const int MAX_STRIDE = 16;

// All formats exercised by tests in this file
static const Format ALL_FORMATS[] = {
    Format::R8,
    Format::R8SN,
    Format::R8UI,
    Format::R8I,
    Format::R16F,
    Format::R16,
    Format::R16SN,
    Format::R16UI,
    Format::R16I,
    Format::R32F,
    Format::R32UI,
    Format::R32I,
    Format::RG8,
    Format::RG8SN,
    Format::RG8UI,
    Format::RG8I,
    Format::RG16F,
    Format::RG16,
    Format::RG16SN,
    Format::RG16UI,
    Format::RG16I,
    Format::RG32F,
    Format::RG32UI,
    Format::RG32I,
    Format::RGB8,
    Format::RGB8SN,
    Format::RGB8UI,
    Format::RGB8I,
    Format::RGB16F,
    Format::RGB16,
    Format::RGB16SN,
    Format::RGB16UI,
    Format::RGB16I,
    Format::RGB32F,
    Format::RGB32UI,
    Format::RGB32I,
    Format::RGBA8,
    Format::RGBA8SN,
    Format::RGBA8UI,
    Format::RGBA8I,
    Format::RGBA16F,
    Format::RGBA16,
    Format::RGBA16SN,
    Format::RGBA16UI,
    Format::RGBA16I,
    Format::RGBA32F,
    Format::RGBA32UI,
    Format::RGBA32I,
    // Depth, stencil, and depth+stencil all swizzle differently. It's not yet
    // known what the final output will look like.
    Format::STENCIL8,
    Format::DEPTH16,
    Format::DEPTH24,
    Format::DEPTH32F,
    Format::DEPTH24_STENCIL8,
    Format::DEPTH32F_STENCIL8,
    Format::RGBX8_SRGB,
    Format::RGBA8_SRGB,
    Format::RGBA4,
    Format::RGB5,
    Format::RGB5A1,
    Format::RGB565,
    Format::RGB10A2,
    Format::RGB10A2UI,
    Format::R11G11B10F,
    Format::RGB9E5F,
    Format::RGB_DXT1,
    Format::RGBA_DXT1,
    Format::RGBA_DXT3,
    Format::RGBA_DXT5,
    Format::RGB_DXT1_SRGB,
    Format::RGBA_DXT1_SRGB,
    Format::RGBA_DXT3_SRGB,
    Format::RGBA_DXT5_SRGB,
    Format::RGTC1_UNORM,
    Format::RGTC1_SNORM,
    Format::RGTC2_UNORM,
    Format::RGTC2_SNORM,
    Format::BPTC_UNORM,
    Format::BPTC_UNORM_SRGB,
    Format::BPTC_SFLOAT,
    Format::BPTC_UFLOAT,
    Format::R8_UI2F,
    Format::R8_I2F,
    Format::R16_UI2F,
    Format::R16_I2F,
    Format::R32_UI2F,
    Format::R32_I2F,
    Format::RG8_UI2F,
    Format::RG8_I2F,
    Format::RG16_UI2F,
    Format::RG16_I2F,
    Format::RG32_UI2F,
    Format::RG32_I2F,
    Format::RGB8_UI2F,
    Format::RGB8_I2F,
    Format::RGB16_UI2F,
    Format::RGB16_I2F,
    Format::RGB32_UI2F,
    Format::RGB32_I2F,
    Format::RGBA8_UI2F,
    Format::RGBA8_I2F,
    Format::RGBA16_UI2F,
    Format::RGBA16_I2F,
    Format::RGBA32_UI2F,
    Format::RGBA32_I2F,
    Format::RGB10A2SN,
    Format::RGB10A2I,
    Format::RGB10A2_UI2F,
    Format::RGB10A2_I2F,
    Format::RGBX8,
    Format::RGBX8SN,
    Format::RGBX8UI,
    Format::RGBX8I,
    Format::RGBX16F,
    Format::RGBX16,
    Format::RGBX16SN,
    Format::RGBX16UI,
    Format::RGBX16I,
    Format::RGBX32F,
    Format::RGBX32UI,
    Format::RGBX32I,
    Format::RGBA_ASTC_4x4,
    Format::RGBA_ASTC_5x4,
    Format::RGBA_ASTC_5x5,
    Format::RGBA_ASTC_6x5,
    Format::RGBA_ASTC_6x6,
    Format::RGBA_ASTC_8x5,
    Format::RGBA_ASTC_8x6,
    Format::RGBA_ASTC_8x8,
    Format::RGBA_ASTC_10x5,
    Format::RGBA_ASTC_10x6,
    Format::RGBA_ASTC_10x8,
    Format::RGBA_ASTC_10x10,
    Format::RGBA_ASTC_12x10,
    Format::RGBA_ASTC_12x12,
    Format::RGBA_ASTC_4x4_SRGB,
    Format::RGBA_ASTC_5x4_SRGB,
    Format::RGBA_ASTC_5x5_SRGB,
    Format::RGBA_ASTC_6x5_SRGB,
    Format::RGBA_ASTC_6x6_SRGB,
    Format::RGBA_ASTC_8x5_SRGB,
    Format::RGBA_ASTC_8x6_SRGB,
    Format::RGBA_ASTC_8x8_SRGB,
    Format::RGBA_ASTC_10x5_SRGB,
    Format::RGBA_ASTC_10x6_SRGB,
    Format::RGBA_ASTC_10x8_SRGB,
    Format::RGBA_ASTC_10x10_SRGB,
    Format::RGBA_ASTC_12x10_SRGB,
    Format::RGBA_ASTC_12x12_SRGB,
    Format::BGR5,
    Format::BGR5A1,
    Format::BGR565,
    Format::A1BGR5,
    Format::BGRA8,
    Format::BGRX8,
    Format::BGRA8_SRGB,
    Format::BGRX8_SRGB,
#if defined(LW_TEGRA)
    Format::Enum(LWN_FORMAT_PRIVATE_RGB_ETC1),
    Format::Enum(LWN_FORMAT_PRIVATE_RGBA_ETC1),
    Format::Enum(LWN_FORMAT_PRIVATE_RGB_ETC1_SRGB),
    Format::Enum(LWN_FORMAT_PRIVATE_RGBA_ETC1_SRGB),
#endif
};

static const int NUM_FORMATS = int(__GL_ARRAYSIZE(ALL_FORMATS));
static const int NUM_COLUMNS = 11;
static const int NUM_ROWS = (NUM_FORMATS + NUM_COLUMNS - 1) / NUM_COLUMNS;

// Fuzzy equality check
bool eq(float x, float y, float epsilon)
{
    if (x == y) {
        // Trivial equality, but also handles inf, etc.
        return true;
    }
    if (fabs(x - y) <= epsilon) {
        return true;
    }
    if (!(x==x) && !(y==y)) {
        // Both NaN, which is valid for our purposes
        return true;
    }
    if (x * y <= 0.0f) {
        // Different signs, or only one is zero.
        return false;
    }
    if (1.0f - epsilon <= (x / y) && (x / y) <= 1.0f + epsilon) {
        // Don't get too picky if the values are large but reasonably close.
        return true;
    }
    return false;
}

// Generic component colwersion. May take advantage of specialized colwersion operators.
template <typename T>
float colwert(const T* data, int index, const Format& /*format*/, bool /*isStencil*/)
{
    return float(data[index]);
}

// Component colwersions that require special handling
template <>
float colwert(const void* data, int index, const Format& format, bool isStencil)
{
    switch (format) {
    case Format::DEPTH24_STENCIL8:
    case Format::DEPTH24:
        {
            if (isStencil) {
                return (*static_cast<const uint32_t*>(data)) & 0xff;
            }
            uint32_t val = (*static_cast<const uint32_t*>(data)) >> 8;
            return float(val) / 0xffffff;
        }
    case Format::DEPTH32F_STENCIL8:
        {
            assert(isStencil); // Depth component should be handled by generic colwersion.
            return (*(static_cast<const uint32_t*>(data) + 1)) & 0xff;
        }
    case Format::RGBX8_SRGB:
    case Format::BGRX8_SRGB:
        {
            float normalized = (*static_cast<const u8lwec4*>(data))[index];
            return srgbToLinear(normalized);
        }
    case Format::RGBA8_SRGB:
    case Format::BGRA8_SRGB:
        {
            float normalized = (*static_cast<const u8lwec4*>(data))[index];
            return (index == 3) ? normalized : srgbToLinear(normalized);
        }

    case Format::RGBA4:
        return (*static_cast<const vec4_rgba4*>(data))[index];
    case Format::RGB5:
        return (*static_cast<const vec3_rgb5*>(data))[index];
    case Format::RGB5A1:
        return (*static_cast<const vec4_rgb5a1*>(data))[index];
    case Format::RGB565:
        return (*static_cast<const vec3_rgb565*>(data))[index];
    case Format::RGB10A2:
        return (*static_cast<const vec4_rgb10a2*>(data))[index];
    case Format::RGB10A2UI:
    case Format::RGB10A2_UI2F:
        return (*static_cast<const vec4_rgb10a2ui*>(data))[index];
    case Format::R11G11B10F:
        {
            float results[3];
            lwColwertR11fG11fB10fToFloat3(*static_cast<const uint32_t*>(data), results);
            return results[index];
        }
    case Format::RGB9E5F:
        {
            float results[3];
            lwColwertRGB9E5ToFloat3(*static_cast<const uint32_t*>(data), results);
            return results[index];
        }
    case Format::RGB10A2I:
    case Format::RGB10A2_I2F:
        return (*static_cast<const vec4_rgb10a2i*>(data))[index];
    case Format::RGB10A2SN:
        return (*static_cast<const vec4_rgb10a2sn*>(data))[index];
    case Format::BGR5:
        return (*static_cast<const vec3_bgr5*>(data))[index];
    case Format::BGR5A1:
        return (*static_cast<const vec4_bgr5a1*>(data))[index];
    case Format::BGR565:
        return (*static_cast<const vec3_bgr565*>(data))[index];
    case Format::A1BGR5:
        return (*static_cast<const vec4_a1bgr5*>(data))[index];

    default:
        DEBUG_PRINT(("Don't know how to colwert format %x\n.", int(format)));
    }
    return 0.0f;
}

#define C(i) colwert(src, i, format, isStencil)
#define FMT(fmt, type, colwersion)                                              \
        case Format::fmt:                                                       \
            {                                                                   \
                const type* src = reinterpret_cast<const type*>(sourcePtr);     \
                result = vec4 colwersion;                                       \
                break;                                                          \
            }

vec4 translateToFormat(const void* sourcePtr, Format format)
{
    vec4 result(0.0f, 0.0f, 0.0f, 1.0f);
    bool isStencil = false;

    switch (format) {
        FMT(R8,                 unorm8,         (C(0), 0.0f, 0.0f, 1.0f))
        FMT(R8SN,               snorm8,         (C(0), 0.0f, 0.0f, 1.0f))
        FMT(R8UI,               uint8_t,        (C(0), 0.0f, 0.0f, 1.0f))
        FMT(R8I,                int8_t,         (C(0), 0.0f, 0.0f, 1.0f))
        FMT(R16F,               float16,        (C(0), 0.0f, 0.0f, 1.0f))
        FMT(R16,                unorm16,        (C(0), 0.0f, 0.0f, 1.0f))
        FMT(R16SN,              snorm16,        (C(0), 0.0f, 0.0f, 1.0f))
        FMT(R16UI,              uint16_t,       (C(0), 0.0f, 0.0f, 1.0f))
        FMT(R16I,               int16_t,        (C(0), 0.0f, 0.0f, 1.0f))
        FMT(R32F,               float,          (C(0), 0.0f, 0.0f, 1.0f))
        FMT(R32UI,              uint32_t,       (C(0), 0.0f, 0.0f, 1.0f))
        FMT(R32I,               int32_t,        (C(0), 0.0f, 0.0f, 1.0f))
        FMT(RG8,                unorm8,         (C(0), C(1), 0.0f, 1.0f))
        FMT(RG8SN,              snorm8,         (C(0), C(1), 0.0f, 1.0f))
        FMT(RG8UI,              uint8_t,        (C(0), C(1), 0.0f, 1.0f))
        FMT(RG8I,               int8_t,         (C(0), C(1), 0.0f, 1.0f))
        FMT(RG16F,              float16,        (C(0), C(1), 0.0f, 1.0f))
        FMT(RG16,               unorm16,        (C(0), C(1), 0.0f, 1.0f))
        FMT(RG16SN,             snorm16,        (C(0), C(1), 0.0f, 1.0f))
        FMT(RG16UI,             uint16_t,       (C(0), C(1), 0.0f, 1.0f))
        FMT(RG16I,              int16_t,        (C(0), C(1), 0.0f, 1.0f))
        FMT(RG32F,              float,          (C(0), C(1), 0.0f, 1.0f))
        FMT(RG32UI,             uint32_t,       (C(0), C(1), 0.0f, 1.0f))
        FMT(RG32I,              int32_t,        (C(0), C(1), 0.0f, 1.0f))
        FMT(RGB8,               unorm8,         (C(0), C(1), C(2), 1.0f))
        FMT(RGB8SN,             snorm8,         (C(0), C(1), C(2), 1.0f))
        FMT(RGB8UI,             uint8_t,        (C(0), C(1), C(2), 1.0f))
        FMT(RGB8I,              int8_t,         (C(0), C(1), C(2), 1.0f))
        FMT(RGB16F,             float16,        (C(0), C(1), C(2), 1.0f))
        FMT(RGB16,              unorm16,        (C(0), C(1), C(2), 1.0f))
        FMT(RGB16SN,            snorm16,        (C(0), C(1), C(2), 1.0f))
        FMT(RGB16UI,            uint16_t,       (C(0), C(1), C(2), 1.0f))
        FMT(RGB16I,             int16_t,        (C(0), C(1), C(2), 1.0f))
        FMT(RGB32F,             float,          (C(0), C(1), C(2), 1.0f))
        FMT(RGB32UI,            uint32_t,       (C(0), C(1), C(2), 1.0f))
        FMT(RGB32I,             int32_t,        (C(0), C(1), C(2), 1.0f))
        FMT(RGBA8,              unorm8,         (C(0), C(1), C(2), C(3)))
        FMT(RGBA8SN,            snorm8,         (C(0), C(1), C(2), C(3)))
        FMT(RGBA8UI,            uint8_t,        (C(0), C(1), C(2), C(3)))
        FMT(RGBA8I,             int8_t,         (C(0), C(1), C(2), C(3)))
        FMT(RGBA16F,            float16,        (C(0), C(1), C(2), C(3)))
        FMT(RGBA16,             unorm16,        (C(0), C(1), C(2), C(3)))
        FMT(RGBA16SN,           snorm16,        (C(0), C(1), C(2), C(3)))
        FMT(RGBA16UI,           uint16_t,       (C(0), C(1), C(2), C(3)))
        FMT(RGBA16I,            int16_t,        (C(0), C(1), C(2), C(3)))
        FMT(RGBA32F,            float,          (C(0), C(1), C(2), C(3)))
        FMT(RGBA32UI,           uint32_t,       (C(0), C(1), C(2), C(3)))
        FMT(RGBA32I,            int32_t,        (C(0), C(1), C(2), C(3)))
        FMT(DEPTH16,            unorm16,        (C(0), 0.0f, 0.0f, 1.0f))
        FMT(DEPTH24,            void,           (C(0), 0.0f, 0.0f, 1.0f))
        FMT(DEPTH32F,           float,          (C(0), 0.0f, 0.0f, 1.0f))
        FMT(DEPTH24_STENCIL8,   void,           (C(0), 0.0f, 0.0f, 1.0f))
        FMT(DEPTH32F_STENCIL8,  float,          (C(0), 0.0f, 0.0f, 1.0f))
        FMT(RGBX8_SRGB,         void,           (C(0), C(1), C(2), 1.0f))
        FMT(RGBA8_SRGB,         void,           (C(0), C(1), C(2), C(3)))
        FMT(RGBA4,              void,           (C(0), C(1), C(2), C(3)))
        FMT(RGB5,               void,           (C(0), C(1), C(2), 1.0f))
        FMT(RGB5A1,             void,           (C(0), C(1), C(2), C(3)))
        FMT(RGB565,             void,           (C(0), C(1), C(2), 1.0f))
        FMT(RGB10A2,            void,           (C(0), C(1), C(2), C(3)))
        FMT(RGB10A2UI,          void,           (C(0), C(1), C(2), C(3)))
        FMT(R11G11B10F,         void,           (C(0), C(1), C(2), 1.0f))
        FMT(RGB9E5F,            void,           (C(0), C(1), C(2), 1.0f))
        FMT(R8_UI2F,            uint8_t,        (C(0), 0.0f, 0.0f, 1.0f))
        FMT(R8_I2F,             int8_t,         (C(0), 0.0f, 0.0f, 1.0f))
        FMT(R16_UI2F,           uint16_t,       (C(0), 0.0f, 0.0f, 1.0f))
        FMT(R16_I2F,            int16_t,        (C(0), 0.0f, 0.0f, 1.0f))
        FMT(R32_UI2F,           uint32_t,       (C(0), 0.0f, 0.0f, 1.0f))
        FMT(R32_I2F,            int32_t,        (C(0), 0.0f, 0.0f, 1.0f))
        FMT(RG8_UI2F,           uint8_t,        (C(0), C(1), 0.0f, 1.0f))
        FMT(RG8_I2F,            int8_t,         (C(0), C(1), 0.0f, 1.0f))
        FMT(RG16_UI2F,          uint16_t,       (C(0), C(1), 0.0f, 1.0f))
        FMT(RG16_I2F,           int16_t,        (C(0), C(1), 0.0f, 1.0f))
        FMT(RG32_UI2F,          uint32_t,       (C(0), C(1), 0.0f, 1.0f))
        FMT(RG32_I2F,           int32_t,        (C(0), C(1), 0.0f, 1.0f))
        FMT(RGB8_UI2F,          uint8_t,        (C(0), C(1), C(2), 1.0f))
        FMT(RGB8_I2F,           int8_t,         (C(0), C(1), C(2), 1.0f))
        FMT(RGB16_UI2F,         uint16_t,       (C(0), C(1), C(2), 1.0f))
        FMT(RGB16_I2F,          int16_t,        (C(0), C(1), C(2), 1.0f))
        FMT(RGB32_UI2F,         uint32_t,       (C(0), C(1), C(2), 1.0f))
        FMT(RGB32_I2F,          int32_t,        (C(0), C(1), C(2), 1.0f))
        FMT(RGBA8_UI2F,         uint8_t,        (C(0), C(1), C(2), C(3)))
        FMT(RGBA8_I2F,          int8_t,         (C(0), C(1), C(2), C(3)))
        FMT(RGBA16_UI2F,        uint16_t,       (C(0), C(1), C(2), C(3)))
        FMT(RGBA16_I2F,         int16_t,        (C(0), C(1), C(2), C(3)))
        FMT(RGBA32_UI2F,        uint32_t,       (C(0), C(1), C(2), C(3)))
        FMT(RGBA32_I2F,         int32_t,        (C(0), C(1), C(2), C(3)))
        FMT(RGB10A2SN,          void,           (C(0), C(1), C(2), C(3)))
        FMT(RGB10A2I,           void,           (C(0), C(1), C(2), C(3)))
        FMT(RGB10A2_UI2F,       void,           (C(0), C(1), C(2), C(3)))
        FMT(RGB10A2_I2F,        void,           (C(0), C(1), C(2), C(3)))
        FMT(RGBX8,              unorm8,         (C(0), C(1), C(2), 1.0f))
        FMT(RGBX8SN,            snorm8,         (C(0), C(1), C(2), 1.0f))
        FMT(RGBX8UI,            uint8_t,        (C(0), C(1), C(2), 1.0f))
        FMT(RGBX8I,             int8_t,         (C(0), C(1), C(2), 1.0f))
        FMT(RGBX16F,            float16,        (C(0), C(1), C(2), 1.0f))
        FMT(RGBX16,             unorm16,        (C(0), C(1), C(2), 1.0f))
        FMT(RGBX16SN,           snorm16,        (C(0), C(1), C(2), 1.0f))
        FMT(RGBX16UI,           uint16_t,       (C(0), C(1), C(2), 1.0f))
        FMT(RGBX16I,            int16_t,        (C(0), C(1), C(2), 1.0f))
        FMT(RGBX32F,            float,          (C(0), C(1), C(2), 1.0f))
        FMT(RGBX32UI,           uint32_t,       (C(0), C(1), C(2), 1.0f))
        FMT(RGBX32I,            int32_t,        (C(0), C(1), C(2), 1.0f))
        FMT(BGR5,               void,           (C(0), C(1), C(2), 1.0f))
        FMT(BGR5A1,             void,           (C(0), C(1), C(2), C(3)))
        FMT(BGR565,             void,           (C(0), C(1), C(2), 1.0f))
        FMT(A1BGR5,             void,           (C(0), C(1), C(2), C(3)))
        FMT(BGRA8,              unorm8,         (C(2), C(1), C(0), C(3)))
        FMT(BGRX8,              unorm8,         (C(2), C(1), C(0), 1.0f))
        FMT(BGRA8_SRGB,         void,           (C(2), C(1), C(0), C(3)))
        FMT(BGRX8_SRGB,         void,           (C(2), C(1), C(0), 1.0f))

        default:
            DEBUG_PRINT(("Unsupported format: 0x%x\n", int(format)));
            break;
    }
    return result;
}

vec4 translateToStencilFormat(const void* sourcePtr, Format format)
{
    vec4 result(0.0f, 0.0f, 0.0f, 1.0f);
    bool isStencil = true;

    switch (format) {
        FMT(STENCIL8,           uint8_t,        (C(0), 0.0f, 0.0f, 1.0f))
        FMT(DEPTH24_STENCIL8,   void,           (C(0), 0.0f, 0.0f, 1.0f))
        FMT(DEPTH32F_STENCIL8,  void,           (C(0), 0.0f, 0.0f, 1.0f))
        default:
            DEBUG_PRINT(("Unsupported stencil format: 0x%x\n", int(format)));
            break;
    }
    return result;
}

#undef FMT
#undef C

// Base class for all tests
class Test {
public:
    virtual ~Test() { }
    // init function separate from the constructor to allow virtual functions.
    virtual void init(Device *device, QueueCommandBuffer &queueCB) = 0;
    virtual TestResult render(Device *device, Queue *queue, QueueCommandBuffer &queueCB, const FormatDesc& formatDesc,
                              Buffer*& sourceBuffer, bool isStencil) = 0;
    virtual ivec2 fbSize() const = 0;
    virtual bool isFormatCompatible(const FormatDesc& desc) const { return true; }
};

// Common code for testing all texture targets. Harness objects should only exist during doGraphics.
class Harness {
    void setViewportAndScissor(int testIndex);
    TestResult renderAndVerify(const FormatDesc& FormatDesc, bool isStencil);
    TestResult testFormat(const FormatDesc& format);
    void renderResult(TestResult result);

    Test& mTest;
    Device *mDevice;
    Queue *mQueue;
    QueueCommandBuffer &mQueueCB;
    Buffer *mSourceBuffer;
#if defined(DEBUG_FORMAT)
    Program *mDebugProgram;
#endif
    Framebuffer mFbo;
    Buffer *mReadbackBuffer;
    BufferAddress mReadbackBufferAddr;

public:
    explicit Harness(Test& test);
    ~Harness();
    void run();
};

Harness::Harness(Test& test) : mTest(test), mQueueCB(DeviceState::GetActive()->getQueueCB())
{
    DeviceState *deviceState = DeviceState::GetActive();
    mDevice = deviceState->getDevice();
    mQueue = deviceState->getQueue();

    // Set up source buffer
    MemoryPool *sourceBufferPool = mDevice->CreateMemoryPool(NULL, sizeof(lwnFormatsSourceData),
                                                             MemoryPoolType::CPU_COHERENT);
    if (!sourceBufferPool) {
        DEBUG_PRINT(("Could not create source buffer pool\n"));
        LWNFailTest();
        return;
    }
    BufferBuilder bufferBuilder;
    bufferBuilder.SetDevice(mDevice).SetDefaults();
    mSourceBuffer = bufferBuilder.CreateBufferFromPool(sourceBufferPool, 0,
                                                       sizeof(lwnFormatsSourceData));
    memcpy(mSourceBuffer->Map(), lwnFormatsSourceData, sizeof(lwnFormatsSourceData));

#if defined(DEBUG_FORMAT)
    {
        VertexShader vs(440);
        vs <<
            "layout(location=0) in vec2 position;\n"
            "out vec2 vPos;\n"
            "void main() {\n"
            "  gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);\n"
            "  vPos = position;\n"
            "}\n";
        FragmentShader fs(440);
        fs <<
            "in vec2 vPos;\n"
            "layout (binding=0) uniform sampler2D tex;\n"
            "out vec4 fcolor;\n"
            "void main() {\n"
            "  fcolor = vec4(texture(tex, vec2(vPos)));\n"
            "}\n";
        mDebugProgram = mDevice->CreateProgram();
        g_glslcHelper->CompileAndSetShaders(mDebugProgram, vs, fs);
    }
#endif

    mFbo.setSize(mTest.fbSize().x(), mTest.fbSize().y());
    mFbo.setColorFormat(0, Format::RGBA32F);
    mFbo.alloc(mDevice);
    mFbo.bind(mQueueCB);
    mQueueCB.SetViewport(0, 0, mTest.fbSize().x(), mTest.fbSize().y());
    mQueueCB.SetScissor(0, 0, mTest.fbSize().x(), mTest.fbSize().y());

    LWNsizei readbackSize = mTest.fbSize().x() * mTest.fbSize().y() * sizeof(vec4);
    MemoryPool *readbackPool = mDevice->CreateMemoryPool(NULL, readbackSize, MemoryPoolType::CPU_COHERENT);
    if (!readbackPool) {
        DEBUG_PRINT(("Could not create readback pool\n"));
        LWNFailTest();
        return;
    }
    bufferBuilder.SetDefaults();
    mReadbackBuffer = bufferBuilder.CreateBufferFromPool(readbackPool, 0, readbackSize);
    mReadbackBufferAddr = mReadbackBuffer->GetAddress();
}

Harness::~Harness()
{
    mQueue->Finish();
    mFbo.destroy();
}

#if !defined(DEBUG_FORMAT) // gcc complains about an unused function if DEBUG_FORMAT is set
void Harness::setViewportAndScissor(int testIndex)
{
    if (testIndex < 0) {
        mQueueCB.SetViewport(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
        mQueueCB.SetScissor(0, 0, lwrrentWindowWidth, lwrrentWindowHeight);
        return;
    }
    int row = testIndex / NUM_COLUMNS;
    int column = testIndex % NUM_COLUMNS;
    int cellLeft = lwrrentWindowWidth * column / NUM_COLUMNS;
    int cellRight = lwrrentWindowWidth * (column + 1) / NUM_COLUMNS;
    int cellBottom = lwrrentWindowHeight * row / NUM_ROWS;
    int cellTop = lwrrentWindowHeight * (row + 1) / NUM_ROWS;
    mQueueCB.SetViewport(cellLeft + 1, cellBottom + 1,
                         cellRight - cellLeft - 2, cellTop - cellBottom - 2);
    mQueueCB.SetScissor(cellLeft + 1, cellBottom + 1,
                        cellRight - cellLeft - 2, cellTop - cellBottom - 2);
}
#endif

vec4 getCompressedPixelFromPrivate(Format format, int index)
{
#if defined(LW_TEGRA)
    const float* resultArray = 0;
    switch ((int)format) {
    case LWN_FORMAT_PRIVATE_RGB_ETC1:
        resultArray = lwnUncompressedData_RGB_ETC1;
        break;
    case LWN_FORMAT_PRIVATE_RGBA_ETC1:
        resultArray = lwnUncompressedData_RGBA_ETC1;
        break;
    case LWN_FORMAT_PRIVATE_RGB_ETC1_SRGB:
        resultArray = lwnUncompressedData_RGB_ETC1_SRGB;
        break;
    case LWN_FORMAT_PRIVATE_RGBA_ETC1_SRGB:
        resultArray = lwnUncompressedData_RGBA_ETC1_SRGB;
        break;
    default:
        DEBUG_PRINT(("Unrecognized compressed format: 0x%x\n", int(format)));
        return vec4(0.0);
    }

    const float* p = resultArray + index * 4;
    return vec4(p[0], p[1], p[2], p[3]);
#else
    DEBUG_PRINT(("Unrecognized compressed format: 0x%x\n", int(format)));
    return vec4(0.0);
#endif
}

vec4 getCompressedPixel(Format format, int index)
{
    const float* resultArray = 0;
    switch (format) {
#define FMT(fmt) case Format::fmt: resultArray = lwnUncompressedData_ ## fmt; break;
        FMT(RGB_DXT1)
        FMT(RGBA_DXT1)
        FMT(RGBA_DXT3)
        FMT(RGBA_DXT5)
        FMT(RGB_DXT1_SRGB)
        FMT(RGBA_DXT1_SRGB)
        FMT(RGBA_DXT3_SRGB)
        FMT(RGBA_DXT5_SRGB)
        FMT(RGTC1_UNORM)
        FMT(RGTC1_SNORM)
        FMT(RGTC2_UNORM)
        FMT(RGTC2_SNORM)
        FMT(BPTC_UNORM)
        FMT(BPTC_UNORM_SRGB)
        FMT(BPTC_SFLOAT)
        FMT(BPTC_UFLOAT)
        FMT(RGBA_ASTC_4x4)
        FMT(RGBA_ASTC_5x4)
        FMT(RGBA_ASTC_5x5)
        FMT(RGBA_ASTC_6x5)
        FMT(RGBA_ASTC_6x6)
        FMT(RGBA_ASTC_8x5)
        FMT(RGBA_ASTC_8x6)
        FMT(RGBA_ASTC_8x8)
        FMT(RGBA_ASTC_10x5)
        FMT(RGBA_ASTC_10x6)
        FMT(RGBA_ASTC_10x8)
        FMT(RGBA_ASTC_10x10)
        FMT(RGBA_ASTC_12x10)
        FMT(RGBA_ASTC_12x12)
        FMT(RGBA_ASTC_4x4_SRGB)
        FMT(RGBA_ASTC_5x4_SRGB)
        FMT(RGBA_ASTC_5x5_SRGB)
        FMT(RGBA_ASTC_6x5_SRGB)
        FMT(RGBA_ASTC_6x6_SRGB)
        FMT(RGBA_ASTC_8x5_SRGB)
        FMT(RGBA_ASTC_8x6_SRGB)
        FMT(RGBA_ASTC_8x8_SRGB)
        FMT(RGBA_ASTC_10x5_SRGB)
        FMT(RGBA_ASTC_10x6_SRGB)
        FMT(RGBA_ASTC_10x8_SRGB)
        FMT(RGBA_ASTC_10x10_SRGB)
        FMT(RGBA_ASTC_12x10_SRGB)
        FMT(RGBA_ASTC_12x12_SRGB)
#undef FMT
        default:
            DEBUG_PRINT(("Unrecognized compressed format: 0x%x\n", int(format)));
            return vec4(0.0);
    }
    const float* p = resultArray + index * 4;
    return vec4(p[0], p[1], p[2], p[3]);
}

vec4 getExpectedCompressedPixel(const FormatDesc& formatDesc, int index)
{
    if (formatDesc.flags & FLAG_PRIVATE) {
        return getCompressedPixelFromPrivate(formatDesc.format, index);
    }

    return getCompressedPixel(formatDesc.format, index);
}

TestResult Harness::renderAndVerify(const FormatDesc& formatDesc, bool isStencil)
{
    TestResult renderResult = mTest.render(mDevice, mQueue, mQueueCB, formatDesc, mSourceBuffer, isStencil);
    mQueueCB.submit();
#if defined(DEBUG_FORMAT)
    mQueue->Finish();
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    mTest.render(mDevice, mQueue, mQueueCB, formatDesc, mSourceBuffer, isStencil);
    mQueueCB.submit();
    mFbo.bind(mQueueCB);
#endif
    if (renderResult != TEST_PASS) {
        return renderResult;
    }
    mQueue->Finish();

    CopyRegion readbackRegion = { 0, 0, 0, mTest.fbSize().x(), mTest.fbSize().y(), 1 };
    mQueueCB.CopyTextureToBuffer(mFbo.getColorTexture(0), NULL, &readbackRegion,
                                 mReadbackBufferAddr, CopyFlags::NONE);
    mQueueCB.submit();
    mQueue->Finish();

    const uint8_t* sourceData = static_cast<const uint8_t*>(mSourceBuffer->Map());
    const vec4* readback = static_cast<const vec4*>(mReadbackBuffer->Map());
    for (int i = 0; i < mTest.fbSize().x() * mTest.fbSize().y(); ++i) {
        vec4 expected(0.0);
        float sigma = 1.0f / 255.0f;
        if (formatDesc.flags & FLAG_COMPRESSED) {
            // Precision issues for compressed textures can get large.
            sigma = 0.1f;
            expected = getExpectedCompressedPixel(formatDesc, i);
        } else {
            const uint8_t* sourcePixel = &sourceData[formatDesc.stride * i];
            expected = isStencil ? translateToStencilFormat(sourcePixel, formatDesc.format) :
                                   translateToFormat(sourcePixel, formatDesc.format);
        }
        if (!(eq((*readback)[0], expected[0], sigma) &&
              eq((*readback)[1], expected[1], sigma) &&
              eq((*readback)[2], expected[2], sigma) &&
              eq((*readback)[3], expected[3], sigma))) {
#if DEBUG_MODE
            int x = i % mTest.fbSize().x();
            int y = i / mTest.fbSize().x();
            DEBUG_PRINT(("Failure detected for format %s at texel(%d, %d).\n",
                         formatDesc.formatName, x, y));
            DEBUG_PRINT(("Expected (%f, %f, %f, %f), got (%f, %f, %f, %f).\n",
                         expected[0], expected[1], expected[2], expected[3],
                         (*readback)[0], (*readback)[1], (*readback)[2], (*readback)[3]));
            if (!(formatDesc.flags & FLAG_COMPRESSED)) {
                DEBUG_PRINT(("Data: "));
                const uint8_t* sourcePixel = &sourceData[formatDesc.stride * i];
                for (int j = 0; j < formatDesc.stride; ++j) {
                    for (int k = 7; k >= 0; --k) {
                        DEBUG_PRINT(("%d", (sourcePixel[j] >> k) & 0x1));
                    }
                }
                DEBUG_PRINT(("\n"));
            }
#endif
            return TEST_FAIL;
        }
        readback += 1;
    }
    return TEST_PASS;
}

TestResult Harness::testFormat(const FormatDesc& formatDesc)
{
    if (!mTest.isFormatCompatible(formatDesc)) {
        return TEST_UNSUPPORTED;
    }
    if ((formatDesc.flags & FLAG_DEPTH) || !(formatDesc.flags & FLAG_STENCIL)) {
        if (renderAndVerify(formatDesc, false) != TEST_PASS) {
            // There shouldn't be any cases where this returns TEST_UNSUPPORTED.
            return TEST_FAIL;
        }
    }
    if (formatDesc.flags & FLAG_STENCIL) {
        // In the case of STENCIL8 it is possible to pass the isFormatCompatible()
        // test at the top of the function but still get an unsupported result.
        return renderAndVerify(formatDesc, true);
    }
    return TEST_PASS;
}

#if !defined(DEBUG_FORMAT) // gcc complains about an unused function if DEBUG_FORMAT is set
void Harness::renderResult(TestResult result)
{
    switch (result) {
    case TEST_PASS:
        mQueueCB.ClearColor(0, 0.0, 1.0, 0.0, 1.0);
        break;
    case TEST_UNSUPPORTED:
        mQueueCB.ClearColor(0, 0.0, 0.0, 1.0, 1.0);
        break;
    case TEST_FAIL:
    default:
        mQueueCB.ClearColor(0, 1.0, 0.0, 0.0, 1.0);
        break;
    }
}
#endif

void Harness::run()
{
    ct_assert(__GL_ARRAYSIZE(ALL_FORMATS) <= NUM_ROWS * NUM_COLUMNS);

    std::vector<TestResult> results;
    results.reserve(__GL_ARRAYSIZE(ALL_FORMATS));

    for (int i = 0; i < NUM_FORMATS; ++i) {
        const FormatDesc &formatDesc = *FormatDesc::findByFormat(ALL_FORMATS[i]);
#if defined(DEBUG_FORMAT)
        if (formatDesc.format != DEBUG_FORMAT) {
            continue;
        }
#endif
        results.push_back(testFormat(formatDesc));
    }

#if !defined(DEBUG_FORMAT)
    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    mQueueCB.ClearColor(0, 0.0, 0.0, 0.1, 1.0);
    for (int i = 0; i < NUM_FORMATS; ++i) {
        setViewportAndScissor(i);
        renderResult(results[i]);
    }
#endif
    g_lwnWindowFramebuffer.setViewportScissor();
    // Work must complete before any memory pool allocators can be freed.
    mQueueCB.submit();
    mQueue->Finish();
}

// Base class for functionality that varies by texture target
class TextureTest : public Test {
public:
    explicit TextureTest(MemoryPoolType::Enum poolType = MemoryPoolType::GPU_ONLY)
    : _poolType(poolType) {
    }
    ~TextureTest();
    void init(Device *device, QueueCommandBuffer &queueCB);
    TestResult render(Device *device, Queue *queue, QueueCommandBuffer &queueCB, const FormatDesc& formatDesc,
                      Buffer *& sourceBuffer, bool isStencil);
    bool isFormatCompatible(const FormatDesc& formatDesc) const;

protected:
    virtual TextureTarget target() const = 0;
    virtual ivec3 texSize() const = 0;
    virtual const char* samplerSuffix() const = 0;
    virtual const char* fetchFunc() const { return "texture"; }
    virtual lwString shaderFunctions() const { return lwString(); }
    virtual lwString samplerCoords() const = 0;
    virtual MemoryPoolType::Enum poolType() const { return _poolType; }
    static lwString getDescriptionForTarget(const char* target);

private:
    Program *mPrograms[3];
    Sampler *mSampler;
    LWNuint mSamplerID;
    TextureBuilder *mTextureBuilder;
    size_t mTexturePoolSize;
    MemoryPool *mTexturePool;
    MemoryPoolAllocator* mVboAllocator;
    MemoryPoolType::Enum _poolType;
};

TextureTest::~TextureTest()
{
    delete mVboAllocator;
}

void TextureTest::init(Device *device, QueueCommandBuffer &queueCB)
{
    // Set up vertex state and data
    struct Vertex {
        vec2 position;
    };
    VertexStream vertexStream(sizeof(Vertex));
    LWN_VERTEX_STREAM_ADD_MEMBER(vertexStream, Vertex, position);
    VertexArrayState vertexState = vertexStream.CreateVertexArrayState();
    static const Vertex vertexData[] = {
        { vec2(0.0, 0.0) },
        { vec2(1.0, 0.0) },
        { vec2(1.0, 1.0) },
        { vec2(0.0, 1.0) },
    };
    mVboAllocator = new MemoryPoolAllocator(device, NULL, sizeof(vertexData),
                                            LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    Buffer *vbo = vertexStream.AllocateVertexBuffer(device, 4, *mVboAllocator, vertexData);
    queueCB.BindVertexBuffer(0, vbo->GetAddress(), sizeof(Vertex) * 4);
    queueCB.BindVertexArrayState(vertexState);

    const char* samplerPrefixes[] = { "", "i", "u" };
    for (int i = 0; i < 3; ++i) {
        VertexShader vs(440);
        vs <<
            "layout(location=0) in vec2 position;\n"
            "out vec2 vPos;\n"
            "void main() {\n"
            "  gl_Position = vec4(position * 2.0 - 1.0, 0.0, 1.0);\n"
            "  vPos = position;\n"
            "}\n";
        FragmentShader fs(440);
        fs <<
            "in vec2 vPos;\n"
            "layout (binding=0) uniform " <<
                samplerPrefixes[i] << "sampler" << samplerSuffix() << " tex;\n"
            "out vec4 fcolor;\n" <<
            shaderFunctions() <<
            "void main() {\n"
            "  fcolor = vec4(" << fetchFunc() << "(tex, " << samplerCoords() << "));\n"
            "}\n";
        mPrograms[i] = device->CreateProgram();
        if (!g_glslcHelper->CompileAndSetShaders(mPrograms[i], vs, fs)) {
            DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
            LWNFailTest();
            return;
        }
    }
    SamplerBuilder sb;
    sb.SetDevice(device).SetDefaults()
                        .SetMinMagFilter(MinFilter::NEAREST, MagFilter::NEAREST);
    mSampler = sb.CreateSampler();
    mSamplerID = mSampler->GetRegisteredID();

    mTextureBuilder = device->CreateTextureBuilder();
    mTextureBuilder->SetDefaults()
                   .SetTarget(target())
                   .SetSize3D(texSize().x(), texSize().y(), texSize().z());
    mTexturePoolSize = 0;
    for (int i = 0; i < NUM_FORMATS; ++i) {
        const FormatDesc &formatDesc = *FormatDesc::findByFormat(ALL_FORMATS[i]);
        if (isFormatCompatible(formatDesc)) {
            mTextureBuilder->SetFormat(formatDesc.format);
            mTexturePoolSize = std::max(mTexturePoolSize, mTextureBuilder->GetStorageSize());
        }
    }
    mTexturePool = device->CreateMemoryPool(NULL, mTexturePoolSize, poolType());
}

TestResult TextureTest::render(Device *device, Queue *queue, QueueCommandBuffer &queueCB, const FormatDesc& formatDesc,
                               Buffer *& sourceBuffer, bool isStencil)
{
    mTextureBuilder->SetFormat(formatDesc.format)
                    .SetDepthStencilMode(isStencil ? TextureDepthStencilMode::STENCIL :
                                                     TextureDepthStencilMode::DEPTH);
    if (FormatIsDepthStencil((LWNformat)(lwn::Format::Enum) formatDesc.format)) {
        mTextureBuilder->SetFlags(mTextureBuilder->GetFlags() | TextureFlags::COMPRESSIBLE);
    }

    assert(mTextureBuilder->GetStorageSize() <= mTexturePoolSize);
    Texture *texture = mTextureBuilder->CreateTextureFromPool(mTexturePool, 0);
    if (!texture) {
        DEBUG_PRINT(("Could not create texture of type %s.\n", formatDesc.formatName));
        return TEST_FAIL;
    }
    TextureHandle texHandle = device->GetTextureHandle(texture->GetRegisteredTextureID(), mSamplerID);
    CopyRegion copyRegion = { 0, 0, 0, texSize().x(), texSize().y(), texSize().z() };
    queueCB.CopyBufferToTexture(sourceBuffer->GetAddress(), texture, NULL, &copyRegion, CopyFlags::NONE);
    queueCB.BindTexture(ShaderStage::FRAGMENT, 0, texHandle);

    SamplerComponentType componentType;
    if (isStencil) {
        componentType = COMP_TYPE_UNSIGNED;
    } else if (formatDesc.flags & FLAG_DEPTH) {
        componentType = COMP_TYPE_FLOAT;
    } else {
        componentType = formatDesc.samplerComponentType;
    }
    queueCB.BindProgram(mPrograms[componentType], ShaderStageBits::ALL_GRAPHICS_BITS);

    queueCB.DrawArrays(DrawPrimitive::TRIANGLE_FAN, 0, 4);
    return TEST_PASS;
}

bool TextureTest::isFormatCompatible(const FormatDesc& desc) const
{
    if (desc.format == Format::STENCIL8 && !g_lwnDeviceCaps.supportsStencil8) {
        return false;
    }
    if (desc.flags & FLAG_ASTC && !g_lwnDeviceCaps.supportsASTC) {
        return false;
    }
    return Test::isFormatCompatible(desc) && (desc.flags & FLAG_TEXTURE);
}

lwString TextureTest::getDescriptionForTarget(const char* target)
{
    return (lwStringBuf() << "Test for LWN texture formats with " << target << " target").str();
}

// Test for 1D texture target
class Test1D : public TextureTest {
public:
    Test1D(MemoryPoolType::Enum poolType) : TextureTest(poolType) {}
    TextureTarget target() const { return TextureTarget::TARGET_1D; }
    ivec3 texSize() const { return ivec3(256, 1, 1); }
    ivec2 fbSize() const { return ivec2(256, 1); }
    const char* samplerSuffix() const { return "1D"; }
    lwString samplerCoords() const { return "vPos.x"; }
    bool isFormatCompatible(const FormatDesc& formatDesc) const
    {
        return TextureTest::isFormatCompatible(formatDesc) &&
               !(formatDesc.flags & FLAG_COMPRESSED);
    }
    static lwString getDescription() { return getDescriptionForTarget("1D"); }
};

// Test for 2D texture target
class Test2D : public TextureTest {
public:
    Test2D(MemoryPoolType::Enum poolType) : TextureTest(poolType) {}
    TextureTarget target() const { return TextureTarget::TARGET_2D; }
    ivec3 texSize() const { return ivec3(DEFAULT_WIDTH, DEFAULT_HEIGHT, 1); }
    ivec2 fbSize() const { return ivec2(DEFAULT_WIDTH, DEFAULT_HEIGHT); }
    const char* samplerSuffix() const { return "2D"; }
    lwString samplerCoords() const { return "vPos"; }
    static lwString getDescription() { return getDescriptionForTarget("2D"); }
};

// Test for 3D texture target
class Test3D : public TextureTest {
public:
    Test3D(MemoryPoolType::Enum poolType) : TextureTest(poolType) {}
    TextureTarget target() const { return TextureTarget::TARGET_3D; }
    ivec3 texSize() const { return ivec3(DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_DEPTH); }
    ivec2 fbSize() const { return ivec2(DEFAULT_WIDTH, DEFAULT_HEIGHT * DEFAULT_DEPTH); }
    const char* samplerSuffix() const { return "3D"; }
    lwString samplerCoords() const
    {
        return (lwStringBuf() <<
                "vec3(vPos.x, fract(vPos.y * " << float(DEFAULT_DEPTH) << "), vPos.y)").str();
    }
    static lwString getDescription() { return getDescriptionForTarget("3D"); }
};

// Test for 1D array texture target
class Test1DArray : public TextureTest {
public:
    Test1DArray(MemoryPoolType::Enum poolType) : TextureTest(poolType) {}
    TextureTarget target() const { return TextureTarget::TARGET_1D_ARRAY; }
    ivec3 texSize() const { return ivec3(DEFAULT_WIDTH, DEFAULT_HEIGHT, 1); }
    ivec2 fbSize() const { return ivec2(DEFAULT_WIDTH, DEFAULT_HEIGHT); }
    const char* samplerSuffix() const { return "1DArray"; }
    lwString samplerCoords() const
    {
        return (lwStringBuf() <<
                "vPos * vec2(1.0, " << float(DEFAULT_HEIGHT) << ") - vec2(0.0, 0.5)").str();
    }
    bool isFormatCompatible(const FormatDesc& formatDesc) const
    {
        return TextureTest::isFormatCompatible(formatDesc) &&
               !(formatDesc.flags & FLAG_COMPRESSED);
    }
    static lwString getDescription() { return getDescriptionForTarget("1D array"); }
};

// Test for 2D array texture target
class Test2DArray : public TextureTest {
public:
    Test2DArray(MemoryPoolType::Enum poolType) : TextureTest(poolType) {}
    TextureTarget target() const { return TextureTarget::TARGET_2D_ARRAY; }
    ivec3 texSize() const { return ivec3(DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_DEPTH); }
    ivec2 fbSize() const { return ivec2(DEFAULT_WIDTH, DEFAULT_HEIGHT * DEFAULT_DEPTH); }
    const char* samplerSuffix() const { return "2DArray"; }
    lwString samplerCoords() const
    {
        return (lwStringBuf() <<
                "vec3(vPos.x, fract(vPos.y * " << float(DEFAULT_DEPTH) << "), " <<
                     "vPos.y * " << float(DEFAULT_DEPTH) << " - 0.5)").str();
    }
    static lwString getDescription() { return getDescriptionForTarget("2D array"); }
};

// Test for Rectangle texture target
class TestRectangle : public TextureTest {
public:
    TestRectangle(MemoryPoolType::Enum poolType) : TextureTest(poolType) {}
    TextureTarget target() const { return TextureTarget::TARGET_RECTANGLE; }
    ivec3 texSize() const { return ivec3(DEFAULT_WIDTH, DEFAULT_HEIGHT, 1); }
    ivec2 fbSize() const { return ivec2(DEFAULT_WIDTH, DEFAULT_HEIGHT); }
    const char* samplerSuffix() const { return "2DRect"; }
    lwString samplerCoords() const
    {
        return (lwStringBuf() << "vPos * vec2(" << float(DEFAULT_WIDTH) << ", " <<
                                 float(DEFAULT_HEIGHT) << ")").str();
    }
    static lwString getDescription() { return getDescriptionForTarget("rectangle"); }
};

// Maps coordinates in [0, 1)^2 to lwbemap texture coordinates, such that texels appear
// in the same pitch-linear order in which they were specified.
const char* LWBE_COORDS =
    "vec3 lwbeCoords(vec2 vPos) {\n"
    "    vec3 result;\n"
    "    int face = int(vPos.y * 6.0);\n"
    "    vec2 p = fract(vPos * vec2(1.0, 6.0)) * 2.0 - vec2(1.0);\n"
    "    if (face == 0) {\n"
    "        result = vec3(1.0, -p.y, -p.x);\n"
    "    } else if (face == 1) {\n"
    "        result = vec3(-1.0, -p.y, p.x);\n"
    "    } else if (face == 2) {\n"
    "        result = vec3(p.x, 1.0, p.y);\n"
    "    } else if (face == 3) {\n"
    "        result = vec3(p.x, -1.0, -p.y);\n"
    "    } else if (face == 4) {\n"
    "        result = vec3(p.x, -p.y, 1.0);\n"
    "    } else {\n"
    "        result = vec3(-p.x, -p.y, -1.0);\n"
    "    }\n"
    "    return result;\n"
    "}\n";

// Test for lwbe map texture target
class TestLwbe : public TextureTest {
public:
    TestLwbe(MemoryPoolType::Enum poolType) : TextureTest(poolType) {}
    TextureTarget target() const { return TextureTarget::TARGET_LWBEMAP; }
    ivec3 texSize() const { return ivec3(DEFAULT_WIDTH, DEFAULT_HEIGHT, 6); }
    ivec2 fbSize() const { return ivec2(DEFAULT_WIDTH, DEFAULT_HEIGHT * 6); }
    const char* samplerSuffix() const { return "Lwbe"; }
    lwString shaderFunctions() const { return LWBE_COORDS; }
    lwString samplerCoords() const { return "lwbeCoords(vPos)"; }
    static lwString getDescription() { return getDescriptionForTarget("lwbemap"); }
};

// Test for lwbe map texture target
class TestLwbeArray : public TextureTest {
public:
    TestLwbeArray(MemoryPoolType::Enum poolType) : TextureTest(poolType) {}
    TextureTarget target() const { return TextureTarget::TARGET_LWBEMAP_ARRAY; }
    ivec3 texSize() const { return ivec3(DEFAULT_WIDTH, DEFAULT_HEIGHT, 6 * 2); }
    ivec2 fbSize() const { return ivec2(DEFAULT_WIDTH, DEFAULT_HEIGHT * 6 * 2); }
    const char* samplerSuffix() const { return "LwbeArray"; }
    lwString shaderFunctions() const { return LWBE_COORDS; }
    lwString samplerCoords() const
    {
        return "vec4(lwbeCoords(fract(vPos * vec2(1.0, 2.0))), int(vPos.y * 2.0))";
    }
    static lwString getDescription() { return getDescriptionForTarget("lwbemap array"); }
};

// Test for buffer texture target
class TestBuffer : public TextureTest {
public:
    TestBuffer(MemoryPoolType::Enum poolType) : TextureTest(poolType) {}
    TextureTarget target() const { return TextureTarget::TARGET_BUFFER; }
    ivec3 texSize() const { return ivec3(256, 1, 1); }
    ivec2 fbSize() const { return ivec2(256, 1); }
    const char* samplerSuffix() const { return "1D"; }
    const char* fetchFunc() const { return "texelFetch"; }
    lwString samplerCoords() const { return "int(vPos.x * 256.0), 0"; }
    bool isFormatCompatible(const FormatDesc& formatDesc) const
    {
        return TextureTest::isFormatCompatible(formatDesc) &&
               !(formatDesc.flags & (FLAG_COMPRESSED | FLAG_DEPTH | FLAG_STENCIL));
    }
    static lwString getDescription() { return getDescriptionForTarget("Buffer"); }
    MemoryPoolType::Enum poolType() const { return MemoryPoolType::CPU_COHERENT; }
};

// Test for vertex "target".
class VertexTest : public Test {
public:
    VertexTest(MemoryPoolType::Enum poolType) {} // poolType is not actually but satisfies base class constructor
    void init(Device *device, QueueCommandBuffer &queueCB);
    TestResult render(Device *device, Queue *queue, QueueCommandBuffer &queueCB, const FormatDesc& formatDesc,
                      Buffer *& sourceBuffer, bool isStencil);
    bool isFormatCompatible(const FormatDesc& formatDesc) const;
    static lwString getDescription() { return "Test for LWN vertex formats"; }
    ivec2 fbSize() const { return ivec2(DEFAULT_WIDTH, DEFAULT_HEIGHT); }
private:
    Buffer *mPositiolwbo;
    Program *mPrograms[3];
};

void VertexTest::init(Device *device, QueueCommandBuffer &queueCB)
{
    int positionSize = DEFAULT_WIDTH * DEFAULT_HEIGHT * sizeof(vec2);
    MemoryPool *positionPool = device->CreateMemoryPool(NULL, positionSize, MemoryPoolType::CPU_COHERENT);
    BufferBuilder vboBuilder;
    vboBuilder.SetDevice(device).SetDefaults();
    mPositiolwbo = vboBuilder.CreateBufferFromPool(positionPool, 0, positionSize);
    vec2* positionPointer = static_cast<vec2*>(mPositiolwbo->Map());
    for (int row = 0; row < DEFAULT_HEIGHT; ++row) {
        float y = -1.0f + 2.0f * (row + 0.5f) / DEFAULT_HEIGHT;
        for (int col = 0; col < DEFAULT_WIDTH; ++col) {
            float x = -1.0f + 2.0f * (col + 0.5f) / DEFAULT_WIDTH;
            *positionPointer++ = vec2(x, y);
        }
    }

    const char* attribPrefixes[] = { "", "i", "u" };
    for (int i = 0; i < 3; ++i) {
        VertexShader vs(440);
        vs <<
            "layout(location=0) in vec2 position;\n"
            "layout(location=1) in " << attribPrefixes[i] << "vec4 color;\n"
            "flat out vec4 vColor;\n"
            "void main() {\n"
            "  gl_Position = vec4(position, 0.0, 1.0);\n"
            "  vColor = color;\n"
            "}\n";
        FragmentShader fs(440);
        fs <<
            "flat in vec4 vColor;\n"
            "out vec4 fColor;\n"
            "void main() {\n"
            "  fColor = vColor;\n"
            "}\n";
        mPrograms[i] = device->CreateProgram();
        if (!g_glslcHelper->CompileAndSetShaders(mPrograms[i], vs, fs)) {
            DEBUG_PRINT(("Shader compile error. infoLog =\n%s\n", g_glslcHelper->GetInfoLog()));
            LWNFailTest();
            return;
        }
    }
}

TestResult VertexTest::render(Device *device, Queue *queue, QueueCommandBuffer &queueCB, const FormatDesc& formatDesc,
                              Buffer *& sourceBuffer, bool isStencil)
{
    queueCB.BindVertexBuffer(0, mPositiolwbo->GetAddress(),
                             DEFAULT_WIDTH * DEFAULT_HEIGHT * sizeof(vec2));
    queueCB.BindVertexBuffer(1, sourceBuffer->GetAddress(),
                             DEFAULT_WIDTH * DEFAULT_HEIGHT * formatDesc.stride);

    VertexAttribState attribState[2];
    VertexStreamState streamState[2];
    attribState[0].SetDefaults()
                  .SetFormat(Format::RG32F, 0)
                  .SetStreamIndex(0);
    attribState[1].SetDefaults()
                  .SetFormat(formatDesc.format, 0)
                  .SetStreamIndex(1);
    streamState[0].SetDefaults().SetStride(sizeof(vec2));
    streamState[1].SetDefaults().SetStride(formatDesc.stride);
    queueCB.BindVertexAttribState(2, attribState);
    queueCB.BindVertexStreamState(2, streamState);
    queueCB.BindProgram(mPrograms[formatDesc.samplerComponentType],
                        ShaderStageBits::ALL_GRAPHICS_BITS);

    queueCB.DrawArrays(DrawPrimitive::POINTS, 0, DEFAULT_WIDTH * DEFAULT_HEIGHT);
    return TEST_PASS;
}

bool VertexTest::isFormatCompatible(const FormatDesc& desc) const
{
    return Test::isFormatCompatible(desc) && (desc.flags & FLAG_VERTEX);
}

} // namespace

template<class T>
class LWNFormatsTest
{
public:
    LWNFormatsTest<T>(MemoryPoolType::Enum poolType = MemoryPoolType::GPU_ONLY)
    : _poolType(poolType)
    {
    }
    LWNTEST_CppMethods();
private:
    MemoryPoolType::Enum _poolType;
};

template<class T>
lwString LWNFormatsTest<T>::getDescription() const
{
    return T::getDescription();
}

template<class T>
int LWNFormatsTest<T>::isSupported() const
{
#if defined(_WIN32)
    return (MemoryPoolType::GPU_ONLY == _poolType) && lwogCheckLWNAPIVersion(12, 0);
#else
    return lwogCheckLWNAPIVersion(12, 0);
#endif
}

template<class T>
void LWNFormatsTest<T>::doGraphics() const
{
    T test(_poolType);
    test.init(DeviceState::GetActive()->getDevice(), *g_lwnQueueCB);
    Harness(test).run();
}

// Pseudo-test used for generating raw test input data and corresponding decompressed texture data.
// Writes to stdout.
class LWNGenFormatData
{
public:
    OGTEST_CppMethods();
};

void LWNGenFormatData::initGraphics()
{
}

void LWNGenFormatData::doGraphics()
{
#ifdef GENERATE_GL_DATA

#if defined(_WIN32)
    PFNGLCOMPRESSEDTEXIMAGE2DPROC glCompressedTexImage2D = (PFNGLCOMPRESSEDTEXIMAGE2DPROC)wglGetProcAddress("glCompressedTexImage2D");
#endif

    struct FormatSpec {
        LWNformat lwnFormat;
        const char* lwnFormatName;
        GLuint glFormat;

        // Used for DXT / RGTC / GPTC compressed formats.
        int bitsPerPixel;

        // Used for ASTC compressed formats.
        int blockWidth;
        int blockHeight;
        int bitsPerBlock;
    } formatSpecs[] = {
#define FMT(lwnFmt, glFmt, bpp) { LWN_FORMAT_ ## lwnFmt, #lwnFmt, GL_COMPRESSED_ ## glFmt, bpp, 0, 0, 0 }
#define FMT_ASTC(lwnFmt, glFmt, BW, BH, BPB) { LWN_FORMAT_ ## lwnFmt, #lwnFmt, GL_COMPRESSED_ ## glFmt, -1, BW, BH, BPB }
#define FMT_ETC1(lwnFmt, glFmt, BW, BH, BPB) { LWN_FORMAT_PRIVATE_ ## lwnFmt, #lwnFmt, GL_COMPRESSED_ ## glFmt, -1, BW, BH, BPB }
        FMT(RGB_DXT1, RGB_S3TC_DXT1_EXT, 4),
        FMT(RGBA_DXT1, RGBA_S3TC_DXT1_EXT, 4),
        FMT(RGBA_DXT3, RGBA_S3TC_DXT3_EXT, 8),
        FMT(RGBA_DXT5, RGBA_S3TC_DXT5_EXT, 8),
        FMT(RGB_DXT1_SRGB, SRGB_S3TC_DXT1_EXT, 4),
        FMT(RGBA_DXT1_SRGB, SRGB_ALPHA_S3TC_DXT1_EXT, 4),
        FMT(RGBA_DXT3_SRGB, SRGB_ALPHA_S3TC_DXT3_EXT, 8),
        FMT(RGBA_DXT5_SRGB, SRGB_ALPHA_S3TC_DXT5_EXT, 8),
        FMT(RGTC1_UNORM, RED_RGTC1, 4),
        FMT(RGTC1_SNORM, SIGNED_RED_RGTC1, 4),
        FMT(RGTC2_UNORM, RG_RGTC2, 8),
        FMT(RGTC2_SNORM, SIGNED_RG_RGTC2, 8),
        FMT(BPTC_UNORM, RGBA_BPTC_UNORM, 8),
        FMT(BPTC_UNORM_SRGB, SRGB_ALPHA_BPTC_UNORM, 8),
        FMT(BPTC_SFLOAT, RGB_BPTC_SIGNED_FLOAT, 8),
        FMT(BPTC_UFLOAT, RGB_BPTC_UNSIGNED_FLOAT, 8),
        FMT_ASTC(RGBA_ASTC_4x4, RGBA_ASTC_4x4_KHR, 4, 4, 128),
        FMT_ASTC(RGBA_ASTC_5x4, RGBA_ASTC_5x4_KHR, 5, 4, 128),
        FMT_ASTC(RGBA_ASTC_5x5, RGBA_ASTC_5x5_KHR, 5, 5, 128),
        FMT_ASTC(RGBA_ASTC_6x5, RGBA_ASTC_6x5_KHR, 6, 5, 128),
        FMT_ASTC(RGBA_ASTC_6x6, RGBA_ASTC_6x6_KHR, 6, 6, 128),
        FMT_ASTC(RGBA_ASTC_8x5, RGBA_ASTC_8x5_KHR, 8, 5, 128),
        FMT_ASTC(RGBA_ASTC_8x6, RGBA_ASTC_8x6_KHR, 8, 6, 128),
        FMT_ASTC(RGBA_ASTC_8x8, RGBA_ASTC_8x8_KHR, 8, 8, 128),
        FMT_ASTC(RGBA_ASTC_10x5, RGBA_ASTC_10x5_KHR, 10, 5, 128),
        FMT_ASTC(RGBA_ASTC_10x6, RGBA_ASTC_10x6_KHR, 10, 6, 128),
        FMT_ASTC(RGBA_ASTC_10x8, RGBA_ASTC_10x8_KHR, 10, 8, 128),
        FMT_ASTC(RGBA_ASTC_10x10, RGBA_ASTC_10x10_KHR, 10, 10, 128),
        FMT_ASTC(RGBA_ASTC_12x10, RGBA_ASTC_12x10_KHR, 12, 10, 128),
        FMT_ASTC(RGBA_ASTC_12x12, RGBA_ASTC_12x12_KHR, 12, 12, 128),
        FMT_ASTC(RGBA_ASTC_4x4_SRGB, SRGB8_ALPHA8_ASTC_4x4_KHR, 4, 4, 128),
        FMT_ASTC(RGBA_ASTC_5x4_SRGB, SRGB8_ALPHA8_ASTC_5x4_KHR, 5, 4, 128),
        FMT_ASTC(RGBA_ASTC_5x5_SRGB, SRGB8_ALPHA8_ASTC_5x5_KHR, 5, 5, 128),
        FMT_ASTC(RGBA_ASTC_6x5_SRGB, SRGB8_ALPHA8_ASTC_6x5_KHR, 6, 5, 128),
        FMT_ASTC(RGBA_ASTC_6x6_SRGB, SRGB8_ALPHA8_ASTC_6x6_KHR, 6, 6, 128),
        FMT_ASTC(RGBA_ASTC_8x5_SRGB, SRGB8_ALPHA8_ASTC_8x5_KHR, 8, 5, 128),
        FMT_ASTC(RGBA_ASTC_8x6_SRGB, SRGB8_ALPHA8_ASTC_8x6_KHR, 8, 6, 128),
        FMT_ASTC(RGBA_ASTC_8x8_SRGB, SRGB8_ALPHA8_ASTC_8x8_KHR, 8, 8, 128),
        FMT_ASTC(RGBA_ASTC_10x5_SRGB, SRGB8_ALPHA8_ASTC_10x5_KHR, 10, 5, 128),
        FMT_ASTC(RGBA_ASTC_10x6_SRGB, SRGB8_ALPHA8_ASTC_10x6_KHR, 10, 6, 128),
        FMT_ASTC(RGBA_ASTC_10x8_SRGB, SRGB8_ALPHA8_ASTC_10x8_KHR, 10, 8, 128),
        FMT_ASTC(RGBA_ASTC_10x10_SRGB, SRGB8_ALPHA8_ASTC_10x10_KHR, 10, 10, 128),
        FMT_ASTC(RGBA_ASTC_12x10_SRGB, SRGB8_ALPHA8_ASTC_12x10_KHR, 12, 10, 128),
        FMT_ASTC(RGBA_ASTC_12x12_SRGB, SRGB8_ALPHA8_ASTC_12x12_KHR, 12, 12, 128),
        FMT_ETC1(RGB_ETC1, RGB8_ETC2, 4, 4, 64),
        FMT_ETC1(RGBA_ETC1_EAC, RGBA8_ETC2_EAC, 4, 4, 128),
        FMT_ETC1(RGB_ETC1_SRGB, SRGB8_ETC2, 4, 4, 64),
        FMT_ETC1(RGBA_ETC1_SRGB, SRGB8_ALPHA8_ETC2_EAC, 4, 4, 128),
#undef FMT
#undef FMT_ASTC
    };

    // Must support both default depth and a minimal lwbe map array.
    // Account for rounding up to blocksize here.
    static const int NUM_PIXELS = (DEFAULT_WIDTH << 1) * (DEFAULT_HEIGHT << 1) *
                                  std::max(DEFAULT_DEPTH, 6 * 2);
    static const int SOURCE_DATA_SIZE = NUM_PIXELS * MAX_STRIDE;

    std::vector<uint8_t> sourceData;
    sourceData.reserve(SOURCE_DATA_SIZE);
    for (int i = 0; i < SOURCE_DATA_SIZE; ++i) {
        // This can generate invalid ASTC blocks. We hope that our HW handles these in a
        // consistent manner.
        sourceData.push_back(lwIntRand(0, 256));
    }

    printf("#ifndef G_LWN_FORMATS_DATA_H\n"
           "#define G_LWN_FORMATS_DATA_H\n"
           "\n"
           "// Machine-generated source data for lwn_formats_* tests.\n"
           "// Do not edit by hand. Edit and re-run the lwn_formats_gen_data test.\n"
           "\n"
           "static unsigned char lwnFormatsSourceData[] = {");
    for (int i = 0; i < SOURCE_DATA_SIZE; ++i) {
        if (i % 16 == 0) {
            printf("\n    ");
        } else {
            printf(" ");
        }
        printf("0x%2.2x", sourceData[i]);
        if (i < SOURCE_DATA_SIZE - 1) {
            printf(",");
        }
    }
    printf("\n};\n\n");

    static const int UNCOMPRESSED_SIZE = NUM_PIXELS * 4;
    std::vector<GLfloat> uncompressedData;
    uncompressedData.reserve(UNCOMPRESSED_SIZE);
    for (int i = 0; i < UNCOMPRESSED_SIZE; ++i) {
        uncompressedData.push_back(0.0f);
    }
    for (size_t k = 0; k < __GL_ARRAYSIZE(formatSpecs); ++k) {
        const FormatSpec& formatSpec = formatSpecs[k];

        GLuint tex;
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);

        GLsizei compressedSize;
        int totalHeight;
        int pitchStride;
        int sliceStride;

        if (formatSpec.bitsPerPixel > -1) {
            totalHeight = DEFAULT_HEIGHT * std::max(DEFAULT_DEPTH, 6 * 2);
            compressedSize = (DEFAULT_WIDTH * totalHeight * formatSpec.bitsPerPixel + 7) / 8;
            pitchStride = DEFAULT_WIDTH * sizeof(GLfloat);
            sliceStride = pitchStride * DEFAULT_HEIGHT;
        } else {
            // We squash depth here to avoid using CompressedTexImage3D. We can pretend a block of
            // 2D texture is 3D if the texture height size matches the blockHeight and no padding oclwrs.
            // For ASTC formats we unfortunately don't have texture heights that match exactly to a multiple
            // of the block height, because the block height varies per format. The HW will round width & height
            // up and then just ignore the data as padding; we need to replicate that behaviour here.
            totalHeight = ROUND_UP(DEFAULT_HEIGHT, formatSpec.blockHeight) * std::max(DEFAULT_DEPTH, 6 * 2);

            // ATSC has variable bpp but constant bytes per block.
            compressedSize = (ROUND_UP(DEFAULT_WIDTH, formatSpec.blockWidth) / formatSpec.blockWidth) *
                             (ROUND_UP(totalHeight, formatSpec.blockHeight) / formatSpec.blockHeight) *
                             (formatSpec.bitsPerBlock / 8);

            pitchStride = ROUND_UP(DEFAULT_WIDTH, formatSpec.blockWidth) * sizeof(GLfloat);
            sliceStride = pitchStride * ROUND_UP(DEFAULT_HEIGHT, formatSpec.blockHeight);
        }

        glCompressedTexImage2D(GL_TEXTURE_2D, 0, formatSpec.glFormat, pitchStride / sizeof(GLfloat),
                               totalHeight, 0, compressedSize, &sourceData[0]);
        GLuint err = glGetError();
        if (err != GL_NO_ERROR) {
            DEBUG_PRINT(("Unexpected error for format %d\n", int(k)));
            assert(!"stop");
        }

        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, &uncompressedData[0]);
        printf("static float lwnUncompressedData_%s[] = {", formatSpec.lwnFormatName);
        for (int i = 0, z = 0; z < std::max(DEFAULT_DEPTH, 6 * 2); ++z) {
            for (int y = 0; y < DEFAULT_HEIGHT; ++y) {
                for (int x = 0; x < DEFAULT_WIDTH * (int)sizeof(GLfloat); ++x, ++i) {
                    int offset = z * sliceStride + y * pitchStride + x;
                    assert(offset >= 0 && offset < UNCOMPRESSED_SIZE);
                    if (i % 8 == 0) {
                        printf("\n    ");
                    } else {
                        printf(" ");
                    }
                    if (i % 4 != 3) {
                        // glGetTexImage does not colwert sRGB to linear, so do it manually.
                        switch (formatSpec.glFormat) {
                        case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT:
                        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT:
                        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT:
                        case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT:
                        case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:
                        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:
                        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:
                        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:
                        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:
                        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:
                        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:
                        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:
                        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:
                        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:
                        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:
                        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:
                        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:
                        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:
                        case GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:
                        case GL_COMPRESSED_SRGB8_ETC2:
                        case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:
                            uncompressedData[offset] = srgbToLinear(uncompressedData[offset]);
                            break;
                        default:
                            break;
                        }
                    } else {
                        // XXX: reading back RGB DXT1 texture data as RGBA produces alpha 0 for black texels
                        // (behaving like RGBA DXT1).
                        switch (formatSpec.glFormat) {
                        case GL_COMPRESSED_RGB_S3TC_DXT1_EXT:
                        case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT:
                            uncompressedData[offset] = 1.0f;
                            break;
                        default:
                            break;
                        }
                    }
                    // Note that for sRGB formats glGetTexImage does not colwert to linear. This gets
                    // handled in the test itself.
                    printf("%.4f", uncompressedData[offset]);
                    if (i < UNCOMPRESSED_SIZE - 1) {
                        printf(",");
                    }
                }
            }
        }
        printf("\n};\n\n");
        glDeleteTextures(1, &tex);
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    printf("#endif // G_LWN_FORMATS_DATA_H\n");
#else
    (void)MAX_STRIDE;
#endif
}

void LWNGenFormatData::exitGraphics()
{
}

lwString LWNGenFormatData::getDescription()
{
    return "Helper test to generate decompressed texture data for lwn_formats_* tests. "
        "Not intended for driver validation.";
}

int LWNGenFormatData::isSupported()
{
    return 1;
}

#define TEXTURE_TEST(__test_class__, __test_name__)                                        \
    OGTEST_CppTest(__test_class__, __test_name__, );                                        \
    OGTEST_CppTest(__test_class__, __test_name__##_coherent, (MemoryPoolType::CPU_COHERENT));

TEXTURE_TEST(LWNFormatsTest<Test1D>, lwn_formats_tex_1d );
TEXTURE_TEST(LWNFormatsTest<Test2D>, lwn_formats_tex_2d );
TEXTURE_TEST(LWNFormatsTest<Test3D>, lwn_formats_tex_3d );
TEXTURE_TEST(LWNFormatsTest<Test1DArray>, lwn_formats_tex_1d_array );
TEXTURE_TEST(LWNFormatsTest<Test2DArray>, lwn_formats_tex_2d_array );
TEXTURE_TEST(LWNFormatsTest<TestRectangle>, lwn_formats_tex_rectangle );
TEXTURE_TEST(LWNFormatsTest<TestLwbe>, lwn_formats_tex_lwbe );
TEXTURE_TEST(LWNFormatsTest<TestLwbeArray>, lwn_formats_tex_lwbe_array );
TEXTURE_TEST(LWNFormatsTest<TestBuffer>, lwn_formats_tex_buffer );

OGTEST_CppTest(LWNFormatsTest<VertexTest>, lwn_formats_vertex, );

OGTEST_CppTest(LWNGenFormatData, lwn_formats_gen_data, );





// ------------------------------------------------------
// CopyTextureToTexture - Texture colwersion validation
// ------------------------------------------------------

#define C(i) colwert(src, i, format, isStencil)
#define FMT(fmt, type, colwersion)                                              \
        case Format::fmt:                                                       \
            {                                                                   \
                const type* src = reinterpret_cast<const type*>(sourcePtr);     \
                result = vec4 colwersion;                                       \
                break;                                                          \
            }

static vec4 translateToFormatCopyTexture(const void* sourcePtr, Format format)
{
    vec4 result(0.0f, 0.0f, 0.0f, 1.0f);
    bool isStencil = false;

    switch (format) {
        FMT(R8,                 unorm8,         (C(0), C(0), C(0), 1.0f)) // R is replicated to G,B
        FMT(R8SN,               snorm8,         (C(0), C(0), C(0), 1.0f)) // R is replicated to G,B
        FMT(R16F,               float16,        (C(0), C(0), C(0), 1.0f)) // R is replicated to G,B
        FMT(R16,                unorm16,        (C(0), C(0), C(0), 1.0f)) // R is replicated to G,B
        FMT(R16SN,              snorm16,        (C(0), C(0), C(0), 1.0f)) // R is replicated to G,B
        FMT(R32F,               float,          (C(0), C(0), C(0), 1.0f)) // R is replicated to G,B
        FMT(RG8,                unorm8,         (C(0), C(1), 0.0f, 1.0f))
        FMT(RG8SN,              snorm8,         (C(0), C(1), 0.0f, 1.0f))
        FMT(RG16F,              float16,        (C(0), C(1), 0.0f, 1.0f))
        FMT(RG16,               unorm16,        (C(0), C(1), 0.0f, 1.0f))
        FMT(RG16SN,             snorm16,        (C(0), C(1), 0.0f, 1.0f))
        FMT(RG32F,              float,          (C(0), C(1), 0.0f, 1.0f))
        FMT(RGBA8,              unorm8,         (C(0), C(1), C(2), C(3)))
        FMT(RGBA8SN,            snorm8,         (C(0), C(1), C(2), C(3)))
        FMT(RGBA16F,            float16,        (C(0), C(1), C(2), C(3)))
        FMT(RGBA16,             unorm16,        (C(0), C(1), C(2), C(3)))
        FMT(RGBA16SN,           snorm16,        (C(0), C(1), C(2), C(3)))
        FMT(RGBA32F,            float,          (C(0), C(1), C(2), C(3)))
        FMT(RGBX8_SRGB,         void,           (C(0), C(1), C(2), 1.0f))
        FMT(RGBA8_SRGB,         void,           (C(0), C(1), C(2), C(3)))
        FMT(RGB5,               void,           (C(0), C(1), C(2), 1.0f))
        FMT(RGB5A1,             void,           (C(0), C(1), C(2), C(3)))
        FMT(RGB565,             void,           (C(0), C(1), C(2), 1.0f))
        FMT(RGB10A2,            void,           (C(0), C(1), C(2), C(3)))
        FMT(RGBX8,              unorm8,         (C(0), C(1), C(2), 1.0f))
        FMT(RGBX8SN,            snorm8,         (C(0), C(1), C(2), C(3))) // XTOA
        FMT(RGBX16F,            float16,        (C(0), C(1), C(2), 1.0f))
        FMT(RGBX16,             unorm16,        (C(0), C(1), C(2), C(3))) // XTOA
        FMT(RGBX16SN,           snorm16,        (C(0), C(1), C(2), C(3))) // XTOA
        FMT(RGBX32F,            float,          (C(0), C(1), C(2), 1.0f))
        FMT(BGR5,               void,           (C(0), C(1), C(2), 1.0f))
        FMT(BGR5A1,             void,           (C(0), C(1), C(2), C(3)))
        FMT(BGR565,             void,           (C(0), C(1), C(2), 1.0f))
        FMT(BGRA8,              unorm8,         (C(2), C(1), C(0), C(3)))
        FMT(BGRX8,              unorm8,         (C(2), C(1), C(0), 1.0f))
        FMT(BGRA8_SRGB,         void,           (C(2), C(1), C(0), C(3)))
        FMT(BGRX8_SRGB,         void,           (C(2), C(1), C(0), 1.0f))
        default:
            DEBUG_PRINT(("Unsupported format: 0x%x\n", int(format)));
            break;
    }
    return result;
}


#define FLAG_R          0x0001
#define FLAG_G          0x0002
#define FLAG_B          0x0004
#define FLAG_A          0x0008
#define FLAG_RGB        0x0010
#define FLAG_BGR        0x0020
#define FLAG_XTOA       0x0040  // formats that interpret the X bits as alpha
#define FLAG_SRCONLY    0x0080
#define FLAG_BGRSRC     0x0100  // when used with BGR formats, this format can only be used as a src
#define FLAG_UNORM      0x0200
#define FLAG_SNORM      0x0400
#define FLAG_16F        0x0800
#define FLAG_32F        0x1000

#define FLAG_1COMP      (FLAG_R)
#define FLAG_2COMP      (FLAG_R | FLAG_G)
#define FLAG_3COMP      (FLAG_R | FLAG_G | FLAG_B)
#define FLAG_4COMP      (FLAG_R | FLAG_G | FLAG_B | FLAG_A)

#define SIGMA_8     (1.0f / 255.0f)
#define SIGMA_16    (1.0f / 65535.0f)
#define SIGMA_16F   (0.01f)
#define SIGMA_32F   (0.0001f)
#define SIGMA_10    (1.0f / 1023.0f)
#define SIGMA_2     (1.0f / 3.0f)
#define SIGMA_5     (1.0f / 31.0f)
#define SIGMA_6     (1.0f / 63.0f)
#define SIGMA_1     (0.501f)
#define SIGMA_8SN   (2.0f / 255.0f)
#define SIGMA_16SN  (2.0f / 65535.0f)
#define SIGMA_8SRGB (2.0f / 255.0f)

#define SIGMA_1010102   SIGMA_10, SIGMA_10, SIGMA_10, SIGMA_2
#define SIGMA_565       SIGMA_5, SIGMA_6, SIGMA_5, SIGMA_1
#define SIGMA_5551      SIGMA_5, SIGMA_5, SIGMA_5, SIGMA_1
#define SIGMA4x(n)      n, n, n, n

struct FormatCopyDesc {
    Format format;
    const char* formatName;
    int stride;
    float sigmaR;
    float sigmaG;
    float sigmaB;
    float sigmaA;
    int components;
    int flags;
};

const FormatCopyDesc ALL_COPY_FORMATS[] = {
#define FORMATDESC(fmt, s, sigma, c, flags) { Format::fmt, #fmt, s, sigma, c, flags }
    FORMATDESC(R8,          1 * 1,  SIGMA4x(SIGMA_8),       1,  FLAG_UNORM | FLAG_1COMP | FLAG_RGB | FLAG_BGRSRC | FLAG_SRCONLY),
    FORMATDESC(R8SN,        1 * 1,  SIGMA4x(SIGMA_8SN),     1,  FLAG_SNORM | FLAG_1COMP | FLAG_RGB | FLAG_BGRSRC),
    FORMATDESC(R16F,        1 * 2,  SIGMA4x(SIGMA_16F),     1,  FLAG_16F   | FLAG_1COMP | FLAG_RGB | FLAG_BGRSRC),
    FORMATDESC(R16,         1 * 2,  SIGMA4x(SIGMA_16),      1,  FLAG_UNORM | FLAG_1COMP | FLAG_RGB | FLAG_BGRSRC | FLAG_SRCONLY),
    FORMATDESC(R16SN,       1 * 2,  SIGMA4x(SIGMA_16SN),    1,  FLAG_SNORM | FLAG_1COMP | FLAG_RGB | FLAG_BGRSRC),
    FORMATDESC(R32F,        1 * 4,  SIGMA4x(SIGMA_32F),     1,  FLAG_32F   | FLAG_1COMP | FLAG_RGB | FLAG_BGRSRC),
    FORMATDESC(RG8,         2 * 1,  SIGMA4x(SIGMA_8),       2,  FLAG_UNORM | FLAG_2COMP | FLAG_RGB),
    FORMATDESC(RG8SN,       2 * 1,  SIGMA4x(SIGMA_8SN),     2,  FLAG_SNORM | FLAG_2COMP | FLAG_RGB),
    FORMATDESC(RG16F,       2 * 2,  SIGMA4x(SIGMA_16F),     2,  FLAG_16F   | FLAG_2COMP | FLAG_RGB),
    FORMATDESC(RG16,        2 * 2,  SIGMA4x(SIGMA_16),      2,  FLAG_UNORM | FLAG_2COMP | FLAG_RGB),
    FORMATDESC(RG16SN,      2 * 2,  SIGMA4x(SIGMA_16SN),    2,  FLAG_SNORM | FLAG_2COMP | FLAG_RGB),
    FORMATDESC(RG32F,       2 * 4,  SIGMA4x(SIGMA_32F),     2,  FLAG_32F   | FLAG_2COMP | FLAG_RGB),
    FORMATDESC(RGBA8,       4 * 1,  SIGMA4x(SIGMA_8),       4,  FLAG_UNORM | FLAG_4COMP | FLAG_RGB),
    FORMATDESC(RGBA8SN,     4 * 1,  SIGMA4x(SIGMA_8SN),     4,  FLAG_SNORM | FLAG_4COMP | FLAG_RGB),
    FORMATDESC(RGBA16F,     4 * 2,  SIGMA4x(SIGMA_16F),     4,  FLAG_16F   | FLAG_4COMP | FLAG_RGB),
    FORMATDESC(RGBA16,      4 * 2,  SIGMA4x(SIGMA_16),      4,  FLAG_UNORM | FLAG_4COMP | FLAG_RGB),
    FORMATDESC(RGBA16SN,    4 * 2,  SIGMA4x(SIGMA_16SN),    4,  FLAG_SNORM | FLAG_4COMP | FLAG_RGB),
    FORMATDESC(RGBA32F,     4 * 4,  SIGMA4x(SIGMA_32F),     4,  FLAG_32F   | FLAG_4COMP | FLAG_RGB),
    FORMATDESC(RGBX8_SRGB,  4 * 1,  SIGMA4x(SIGMA_8SRGB),   4,  FLAG_UNORM | FLAG_3COMP | FLAG_RGB),
    FORMATDESC(RGBA8_SRGB,  4 * 1,  SIGMA4x(SIGMA_8SRGB),   4,  FLAG_UNORM | FLAG_4COMP | FLAG_RGB),
    FORMATDESC(RGB5,        2,      SIGMA_5551,             0,  FLAG_UNORM | FLAG_3COMP | FLAG_BGR),
    FORMATDESC(RGB5A1,      2,      SIGMA_5551,             0,  FLAG_UNORM | FLAG_4COMP | FLAG_BGR),
    FORMATDESC(RGB565,      2,      SIGMA_565,              0,  FLAG_UNORM | FLAG_3COMP | FLAG_BGR),
    FORMATDESC(RGB10A2,     4,      SIGMA_1010102,          0,  FLAG_UNORM | FLAG_4COMP | FLAG_RGB),
    FORMATDESC(RGBX8,       4 * 1,  SIGMA4x(SIGMA_8),       4,  FLAG_UNORM | FLAG_3COMP | FLAG_RGB),
    FORMATDESC(RGBX8SN,     4 * 1,  SIGMA4x(SIGMA_8SN),     4,  FLAG_SNORM | FLAG_3COMP | FLAG_RGB | FLAG_XTOA),
    FORMATDESC(RGBX16F,     4 * 2,  SIGMA4x(SIGMA_16F),     4,  FLAG_16F   | FLAG_3COMP | FLAG_RGB),
    FORMATDESC(RGBX16,      4 * 2,  SIGMA4x(SIGMA_16),      4,  FLAG_UNORM | FLAG_3COMP | FLAG_RGB | FLAG_XTOA),
    FORMATDESC(RGBX16SN,    4 * 2,  SIGMA4x(SIGMA_16SN),    4,  FLAG_SNORM | FLAG_3COMP | FLAG_RGB | FLAG_XTOA),
    FORMATDESC(RGBX32F,     4 * 4,  SIGMA4x(SIGMA_32F),     4,  FLAG_32F   | FLAG_3COMP | FLAG_RGB),
    FORMATDESC(BGR565,      2,      SIGMA_565,              0,  FLAG_UNORM | FLAG_3COMP | FLAG_RGB),
    FORMATDESC(BGR5,        2,      SIGMA_5551,             0,  FLAG_UNORM | FLAG_3COMP | FLAG_RGB),
    FORMATDESC(BGR5A1,      2,      SIGMA_5551,             0,  FLAG_UNORM | FLAG_4COMP | FLAG_RGB),
    FORMATDESC(BGRA8,       4 * 1,  SIGMA4x(SIGMA_8),       4,  FLAG_UNORM | FLAG_4COMP | FLAG_RGB),
    FORMATDESC(BGRX8,       4 * 1,  SIGMA4x(SIGMA_8),       4,  FLAG_UNORM | FLAG_3COMP | FLAG_RGB),
    FORMATDESC(BGRA8_SRGB,  4 * 1,  SIGMA4x(SIGMA_8SRGB),   4,  FLAG_UNORM | FLAG_4COMP | FLAG_RGB),
    FORMATDESC(BGRX8_SRGB,  4 * 1,  SIGMA4x(SIGMA_8SRGB),   4,  FLAG_UNORM | FLAG_3COMP | FLAG_RGB),
#undef FORMATDESC
};

static const int NUM_COPY_FORMATS = int(__GL_ARRAYSIZE(ALL_COPY_FORMATS));


static void clampFloat(float &val, float min, float max)
{
    if (val < min) {
        val = min;
    } else if (val > max) {
        val = max;
    }
}

static void clampFloatNoInf(float &val, float min, float max)
{
    if (LW_ISINF(val))
        return;
    clampFloat(val, min, max);
}

static void formatClamp(const FormatCopyDesc *srcDesc, const FormatCopyDesc *dstDesc, vec4 &srcRGBA)
{
//    bool isSrlwnorm = (0 != (srcDesc->flags & FLAG_UNORM));
//    bool isSrcSnorm = (0 != (srcDesc->flags & FLAG_SNORM));
    bool isSrcFloat16 = (0 != (srcDesc->flags & FLAG_16F));
    bool isSrcFloat32 = (0 != (srcDesc->flags & FLAG_32F));

    bool isDstUnorm = (0 != (dstDesc->flags & FLAG_UNORM));
    bool isDstSnorm = (0 != (dstDesc->flags & FLAG_SNORM));
    bool isDstFloat16 = (0 != (dstDesc->flags & FLAG_16F));
    bool isDstFloat32 = (0 != (dstDesc->flags & FLAG_32F));

    if ((isSrcFloat16 || isSrcFloat32) && !(isDstFloat16 || isDstFloat32)) {
        // flush nan to 0
        if (LW_ISNAN(srcRGBA[0])) srcRGBA[0] = 0.0f;
        if (LW_ISNAN(srcRGBA[1])) srcRGBA[1] = 0.0f;
        if (LW_ISNAN(srcRGBA[2])) srcRGBA[2] = 0.0f;
        if (LW_ISNAN(srcRGBA[3])) srcRGBA[3] = 0.0f;
    }

    if (isSrcFloat32 && isDstFloat16) {
        // clamp without flushing infs
        clampFloatNoInf(srcRGBA[0], -65504.0, 65504.0);
        clampFloatNoInf(srcRGBA[1], -65504.0, 65504.0);
        clampFloatNoInf(srcRGBA[2], -65504.0, 65504.0);
        clampFloatNoInf(srcRGBA[3], -65504.0, 65504.0);
    }

    if (isDstUnorm) {
        clampFloat(srcRGBA[0], 0.0f, 1.0f);
        clampFloat(srcRGBA[1], 0.0f, 1.0f);
        clampFloat(srcRGBA[2], 0.0f, 1.0f);
        clampFloat(srcRGBA[3], 0.0f, 1.0f);
        return;
    }
    if (isDstSnorm) {
        clampFloat(srcRGBA[0], -1.0f, 1.0f);
        clampFloat(srcRGBA[1], -1.0f, 1.0f);
        clampFloat(srcRGBA[2], -1.0f, 1.0f);
        clampFloat(srcRGBA[3], -1.0f, 1.0f);
        return;
    }
}

template <typename T>
static void appendValue(void **data, T v, int count)
{
    T *d = *((T **) data);
    for (int i = 0; i < count; i++)
    {
        d[i] = v;
    }
    *data = (void *) (d + count);
}

static int fillTestValues(const FormatCopyDesc *fd, void *data)
{
    // Rely exclusively on random data for the 16-bit RGB and BGR formats
    if (fd->components == 0) {
        return 0;
    }

    int bpc = fd->stride / fd->components;
    int c = fd->components;

    switch (fd->flags & (FLAG_UNORM | FLAG_SNORM | FLAG_16F | FLAG_32F))
    {
        case FLAG_UNORM:
            switch (bpc) {
                case 1:
                    appendValue<uint8_t>(&data, 0, c);
                    appendValue<uint8_t>(&data, 2, c);
                    appendValue<uint8_t>(&data, 127, c);
                    appendValue<uint8_t>(&data, 255, c);
                    return 4;
                case 2:
                    appendValue<uint16_t>(&data, 0, c);
                    appendValue<uint16_t>(&data, 1, c);
                    appendValue<uint16_t>(&data, 256, c);
                    appendValue<uint16_t>(&data, 32767, c);
                    appendValue<uint16_t>(&data, 65535, c);
                    return 5;
                default:
                    printf("Unhandled %s bpc=%i\n", fd->formatName, bpc);
                    assert(0); // should not get here
            }
            break;
        case FLAG_SNORM:
            switch (bpc) {
                case 1:
                    appendValue<int8_t>(&data, 0, c);
                    appendValue<int8_t>(&data, 1, c);
                    appendValue<int8_t>(&data, 64, c);
                    appendValue<int8_t>(&data, 127, c);
                    appendValue<int8_t>(&data, -127, c);
                    appendValue<int8_t>(&data, -128, c);
                    return 6;
                case 2:
                    appendValue<int16_t>(&data, 0, c);
                    appendValue<int16_t>(&data, 1, c);
                    appendValue<int16_t>(&data, 256, c);
                    appendValue<int16_t>(&data, 32767, c);
                    appendValue<int16_t>(&data, -32767, c);
                    appendValue<int16_t>(&data, -32768, c);
                    return 6;
                default:
                    printf("Unhandled %s bpc=%i\n", fd->formatName, bpc);
                    assert(0); // should not get here
            }
            break;
        case FLAG_16F:
            appendValue<uint16_t>(&data, 0x0000, c); // 0
            appendValue<uint16_t>(&data, 0x8000, c); // -0
            appendValue<uint16_t>(&data, 0x7c00, c); // inf
            appendValue<uint16_t>(&data, 0xfc00, c); // -inf
            appendValue<uint16_t>(&data, 0x7e00, c); // nan
            appendValue<uint16_t>(&data, 0x3c00, c); // 1
            appendValue<uint16_t>(&data, lwF32toS10E5(127.0), c);
            appendValue<uint16_t>(&data, lwF32toS10E5(-127.0), c);
            appendValue<uint16_t>(&data, lwF32toS10E5(255.0), c);
            appendValue<uint16_t>(&data, lwF32toS10E5(65504.0), c);
            appendValue<uint16_t>(&data, lwF32toS10E5(-65504.0), c);
            return 11;
        case FLAG_32F:
            appendValue<uint32_t>(&data, 0x00000000, c); // 0
            appendValue<uint32_t>(&data, 0x80000000, c); // -0
            appendValue<uint32_t>(&data, 0x7f800000, c); // inf
            appendValue<uint32_t>(&data, 0xff800000, c); // -inf
            appendValue<uint32_t>(&data, 0x7fc00000, c); // nan
            appendValue<float>(&data, 1.0f, c);
            appendValue<float>(&data, 127.0f, c);
            appendValue<float>(&data, -127.0f, c);
            appendValue<float>(&data, 255.0, c);
            appendValue<float>(&data, 65504.0, c);
            appendValue<float>(&data, -65504.0, c);
            appendValue<float>(&data, FLT_MAX, c);
            appendValue<float>(&data, -FLT_MAX, c);
            return 13;
        default:
            printf("Unhandled %s bpc=%i\n", fd->formatName, bpc);
            assert(0); // should not get here
    }
    return 0;
}


class LWNFormatsCopyTexColwert
{
    static const int texSize = 32;

public:
    LWNTEST_CppMethods();
    explicit LWNFormatsCopyTexColwert() {}
};

lwString LWNFormatsCopyTexColwert::getDescription() const
{
    lwStringBuf sb;
    sb << "Test LWN's CopyTextureToTexture. Tests format colwersion for each color format that\n"
          "supports colwersions, by colwerting between every legal format combination:\n"
          " * Creates a " << texSize << "x" << texSize << " texture in the source format, and fills it with test data.\n"
          " * Copies the contents of that texture into a second destination texture using the\n"
          "   CopyTextureToTexture API.\n"
          " * The destination texture's contents are copied back to CPU memory, and both the source\n"
          "   and destination test data is colwerted to float and compared to verify the results,\n"
          "   taking into account any precision or range differences that may occur as a result of\n"
          "   the format colwersion.\n"
          "\n"
          "Results are displayed as a grid, with source texture format being a given row and\n"
          "destination texture format being a given column. Green is pass, red is fail, and blue\n"
          "indicates an unsupported colwersion.\n";
    return sb.str();
}

int LWNFormatsCopyTexColwert::isSupported() const
{
    return lwogCheckLWNAPIVersion(50, 1);
}


void LWNFormatsCopyTexColwert::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    const int cellWidth = NUM_COPY_FORMATS;
    const int cellHeight = NUM_COPY_FORMATS;
    cellTestInit(cellWidth, cellHeight);

    queueCB.ClearColor(0, 0.0, 0.0, 0.0, 0.0);
    queueCB.submit();

    // Pick a nice big number for storage size
    int maxBufferSize = texSize * texSize * 4 * sizeof(float);
    int texStorageSize = maxBufferSize * 3; // (maxBufferSize * 2) + scratch

    MemoryPoolAllocator sysAllocator(device, NULL, texStorageSize, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    MemoryPoolAllocator texAllocator(device, NULL, texStorageSize, LWN_MEMORY_POOL_TYPE_GPU_ONLY);

    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *srcBuf = sysAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_READ_BIT, maxBufferSize);
    Buffer *dstBuf = sysAllocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, maxBufferSize);
    BufferAddress srcBufAddress = srcBuf->GetAddress();
    BufferAddress dstBufAddress = dstBuf->GetAddress();
    char *randBuffer = new char[maxBufferSize];
    char *srcPtr = (char *) srcBuf->Map();
    char *dstPtr = (char *) dstBuf->Map();

    // Fill the source buffer with random data
    for (int i=0; i<maxBufferSize; i++)
    {
        randBuffer[i] = lwIntRand(0, 255);
    }

    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults();
    tb.SetTarget(TextureTarget::TARGET_2D);
    tb.SetSize2D(texSize, texSize);

    for (int srcFmtIndex = 0; srcFmtIndex < NUM_COPY_FORMATS; srcFmtIndex++)
    {
        for (int dstFmtIndex = 0; dstFmtIndex < NUM_COPY_FORMATS; dstFmtIndex++)
        {
            const FormatCopyDesc *srcDesc = &ALL_COPY_FORMATS[srcFmtIndex];
            const FormatCopyDesc *dstDesc = &ALL_COPY_FORMATS[dstFmtIndex];
            int srcFlags = srcDesc->flags;
            int dstFlags = dstDesc->flags;

            // FLAG_BGRSRC indicates that a format can be used as a source for
            // BGR formats, but not a destination. (This effectively means
            // that, when used as a source, it can be used with both BGR and
            // RGB formats.)
            if (srcFlags & FLAG_BGRSRC) {
                srcFlags |= FLAG_BGR;
            }

            // skip unsupported format colwersions
            {
                // Check to make sure we can colwert between the source and
                // destination texture formats. The main stumbling block here
                // is whether the source and destination are RGB vs BGR.
                if (!(srcFlags & dstFlags & (FLAG_BGR | FLAG_RGB))) {
                    SetCellViewportScissorPadded(queueCB, dstFmtIndex, srcFmtIndex, 1);
                    queueCB.ClearColor(0, 0.0, 0.0, 1.0);
                    continue;
                }
                if ((dstFlags & FLAG_SRCONLY) && (dstFmtIndex != srcFmtIndex)) {
                    SetCellViewportScissorPadded(queueCB, dstFmtIndex, srcFmtIndex, 1);
                    queueCB.ClearColor(0, 0.0, 0.0, 1.0);
                    continue;
                }
            }

            bool skipR = !(dstFlags & FLAG_R);
            bool skipG = !(dstFlags & FLAG_G);
            bool skipB = !(dstFlags & FLAG_B);
            bool skipA = !(dstFlags & FLAG_A);

            // Test START!
            const Format srcFmt = srcDesc->format;
            const Format dstFmt = dstDesc->format;
            const LWNuint srcBpp = srcDesc->stride;
            const LWNuint dstBpp = dstDesc->stride;

            float sigmaR = srcDesc->sigmaR + dstDesc->sigmaR;
            float sigmaG = srcDesc->sigmaG + dstDesc->sigmaG;
            float sigmaB = srcDesc->sigmaB + dstDesc->sigmaB;
            float sigmaA = srcDesc->sigmaA + dstDesc->sigmaA;

            // Generate the data for the source texture, and then fill with
            // additional random data
            {
                int filled = fillTestValues(srcDesc, srcPtr);
                filled *= srcDesc->stride;
                memcpy(srcPtr + filled, randBuffer, maxBufferSize - filled);
            }

            // Destination texture
            tb.SetFormat(dstFmt);
            Texture *dstTex = texAllocator.allocTexture(&tb);

            // Create and fill the source texture
            tb.SetFormat(srcFmt);
            Texture *srcTex = texAllocator.allocTexture(&tb);

            CopyRegion b2tRegion = { 0, 0, 0, texSize, texSize, 1 };
            queueCB.CopyBufferToTexture(srcBufAddress, srcTex, NULL, &b2tRegion, CopyFlags::NONE);

            // Copy to destination
            queueCB.CopyTextureToTexture(srcTex, NULL, &b2tRegion, dstTex, NULL, &b2tRegion, CopyFlags::NONE);

            // Copy back to system memory
            queueCB.CopyTextureToBuffer(dstTex, NULL, &b2tRegion, dstBufAddress, CopyFlags::NONE);

            // submit, make sure the GPU is done.
            queueCB.submit();
            queue->Finish();

            // Compare with expected values
            bool passed = true;
            char *srcData = srcPtr;
            char *dstData = dstPtr;
            for (int y = 0; y < texSize && passed; y++) {
                for (int x = 0; x < texSize && passed; x++) {
                    vec4 srcRGBA = translateToFormatCopyTexture(srcData, srcFmt);
                    vec4 dstRGBA = translateToFormatCopyTexture(dstData, dstFmt);
                    formatClamp(srcDesc, dstDesc, srcRGBA);
                    if (!((skipR || eq(srcRGBA[0], dstRGBA[0], sigmaR)) &&
                          (skipG || eq(srcRGBA[1], dstRGBA[1], sigmaG)) &&
                          (skipB || eq(srcRGBA[2], dstRGBA[2], sigmaB)) &&
                          (skipA || eq(srcRGBA[3], dstRGBA[3], sigmaA))))
                    {
#if 0
                        printf("failed %s to %s, coord(%i,%i). src: %1.3f %1.3f %1.3f %1.3f  dst %1.3f %1.3f %1.3f %1.3f\n",
                                srcDesc->formatName,
                                dstDesc->formatName,
                                x, y,
                                srcRGBA[0], srcRGBA[1], srcRGBA[2], srcRGBA[3],
                                dstRGBA[0], dstRGBA[1], dstRGBA[2], dstRGBA[3]);
#endif
                        passed = false;
                    }
                    srcData += srcBpp;
                    dstData += dstBpp;
                }
            }

            SetCellViewportScissorPadded(queueCB, dstFmtIndex, srcFmtIndex, 1);
            if (passed)
            {
                queueCB.ClearColor(0, 0.0, 1.0, 0.0);
            } else {
                queueCB.ClearColor(0, 1.0, 0.0, 0.0);
            }

            texAllocator.freeTexture(srcTex);
            texAllocator.freeTexture(dstTex);
        }
    }

    queueCB.submit();
    queue->Finish();

    sysAllocator.freeBuffer(srcBuf);
    sysAllocator.freeBuffer(dstBuf);

    delete [] randBuffer;
}



OGTEST_CppTest(LWNFormatsCopyTexColwert, lwn_formats_copy_tex_colwert, );

