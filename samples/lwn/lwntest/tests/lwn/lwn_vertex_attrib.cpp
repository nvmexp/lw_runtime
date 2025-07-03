/*
 * Copyright (c) 2015-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"
#include "string.h"
#include "float_util.h"

/**********************************************************************/

using namespace lwn;
using namespace lwn::dt;

/**********************************************************************/
// lwogtest doesn't like random spew to standard output.  We just eat any
// output unless LWN_DEBUG_LOG is set to 1.
#define LWN_DEBUG_LOG 0
#if LWN_DEBUG_LOG
#define LOG(x) do { \
    printf x; \
    fflush(stdout); \
} while (0)
#else
#define LOG(x)
#endif

/**********************************************************************/

#define LWN_MAX_STREAMS 16
#define LWN_MAX_ATTRIBS 16

static const char *attrs[LWN_MAX_ATTRIBS] = {
    "attr0", "attr1", "attr2",  "attr3",  "attr4",  "attr5",  "attr6",  "attr7",
    "attr8", "attr9", "attr10", "attr11", "attr12", "attr13", "attr14", "attr15",
};

static const char *outs[LWN_MAX_ATTRIBS] = {
    "out0", "out1", "out2",  "out3",  "out4",  "out5",  "out6",  "out7",
    "out8", "out9", "out10", "out11", "out12", "out13", "out14", "out15",
};

/**********************************************************************/
// Unaligned value extraction routines. On ARM, we can't just make
// arbitrary casts of transform feedback results, because the values
// may not be aligned, which will trigger an exception.

template <typename T>
static void unpackValues(T *destPtr, const uint8_t *srcPtr, int count)
{
    uint8_t *dest = reinterpret_cast<uint8_t*>(destPtr);
    memcpy(dest, srcPtr, sizeof(T) * count);
}

template <typename T>
static T unpackValue(const uint8_t *value)
{
    T result;
    memcpy(&result, value, sizeof(T));
    return result;
}

// Shorter macro for reinterpret_cast
// RCD  - reinterpret cast defreference
// RCCD - reinterpret cast const dereference
#define RCD(t, s) unpackValue<t>(s)
#define RCCD(t, s) unpackValue<t>(s)

/**********************************************************************/
// Comparison functions to translate and compare formatted input
// with the output from transform feedback
typedef bool (*CompareFunction)(const uint8_t *compare, const uint8_t *result);

template <typename T>
union CompareType {
    CompareType() {}
    CompareType(const LWNfloat &f) : fValue(f) {}
    LWNfloat fValue;
    T         value;
};
typedef CompareType<LWNuint> compareUI;
typedef CompareType<LWNint>  compareI;

static bool compareNONE(const uint8_t *compare, const uint8_t *result) {
    return false;
}

static bool compareR8(const uint8_t *compare, const uint8_t *result) {
    return (compareUI(*compare/255.0f).value == RCCD(LWNuint, result));
}

static bool compareR8SN(const uint8_t *compare, const uint8_t *result) {
    int8_t x = RCCD(int8_t, compare);
    if (x < -127) { x = -127; }
    return (compareUI(x/127.0f).value == RCCD(LWNuint, result));
}

static bool compareR8UI(const uint8_t *compare, const uint8_t *result) {
    return (compare[0] == RCCD(LWNuint, result));
}

static bool compareR8I(const uint8_t *compare, const uint8_t *result) {
    return (RCCD(int8_t, compare) == RCCD(LWNint, result));
}

static bool compareR16F(const uint8_t *compare, const uint8_t *result) {
    return (compareUI(lwS10E5toF32(RCCD(LWs10e5, compare))).value == RCCD(LWNuint, result));
}

static bool compareR16(const uint8_t *compare, const uint8_t *result) {
    return (compareUI(RCCD(uint16_t, compare)/65535.0f).value == RCCD(LWNuint, result));
}

static bool compareR16SN(const uint8_t *compare, const uint8_t *result) {
    int16_t x = RCCD(int16_t, compare);
    if (x < -32767) { x = -32767; }
    return (compareUI(x/32767.0f).value == RCCD(LWNuint, result));
}

static bool compareR16UI(const uint8_t *compare, const uint8_t *result) {
    return (RCCD(uint16_t, compare) == RCCD(LWNuint, result));
}

static bool compareR16I(const uint8_t *compare, const uint8_t *result) {
    return (RCCD(int16_t, compare) == RCCD(LWNint, result));
}

static bool compareR32F(const uint8_t *compare, const uint8_t *result) {
    return (RCCD(LWNuint, compare) == RCCD(LWNuint, result));
}

static bool compareR32UI(const uint8_t *compare, const uint8_t *result) {
    return (RCCD(LWNuint, compare) == RCCD(LWNuint, result));
}

static bool compareR32I(const uint8_t *compare, const uint8_t *result) {
    return (RCCD(LWNint, compare) == RCCD(LWNint, result));
}

static bool compareR8_UI2F(const uint8_t *compare, const uint8_t *result) {
    return (compareUI(LWNfloat(compare[0])).value == RCCD(LWNuint,result));
}

static bool compareR8_I2F(const uint8_t *compare, const uint8_t *result) {
    return (compareUI(LWNfloat(RCCD(int8_t, &compare[0]))).value == RCCD(LWNuint, result));
}

static bool compareR16_UI2F(const uint8_t *compare, const uint8_t *result) {
    return (compareUI(LWNfloat(RCCD(uint16_t, &compare[0]))).value == RCCD(LWNuint, result));
}

static bool compareR16_I2F(const uint8_t *compare, const uint8_t *result) {
    return (compareUI(LWNfloat(RCCD(int16_t, &compare[0]))).value == RCCD(LWNuint, result));
}

static bool compareR32_UI2F(const uint8_t *compare, const uint8_t *result) {
    return (compareUI(LWNfloat(RCCD(LWNuint, &compare[0]))).value == RCCD(LWNuint, result));
}

static bool compareR32_I2F(const uint8_t *compare, const uint8_t *result) {
    return (compareUI(LWNfloat(RCCD(LWNint, &compare[0]))).value == RCCD(LWNuint, result));
}

static bool compareR11G11B10F(const uint8_t *compare, const uint8_t *result) {
    compareUI x[3];
    lwColwertR11fG11fB10fToFloat3(RCCD(LWNuint, compare), &(x[0].fValue));
    LWNuint y[4];
    unpackValues(y, result, 4);
    return (x[0].value == y[0] && x[1].value == y[1] && x[2].value == y[2]);
}

/**********************************************************************/
// Create all derived comparison functions
#define MAKE_COMPARE_FUNC(_a, _b, _c, _d, _e)                                                \
static bool compare##_a(const uint8_t *compare, const uint8_t *result) {                   \
    return (compare##_b(&compare[0], &result[0]) && compare##_c(&compare[_d], &result[_e])); \
}

MAKE_COMPARE_FUNC(RG8,         R8,         R8,        1,  4)
MAKE_COMPARE_FUNC(RG8SN,       R8SN,       R8SN,      1,  4)
MAKE_COMPARE_FUNC(RG8UI,       R8UI,       R8UI,      1,  4)
MAKE_COMPARE_FUNC(RG8I,        R8I,        R8I,       1,  4)
MAKE_COMPARE_FUNC(RG16F,       R16F,       R16F,      2,  4)
MAKE_COMPARE_FUNC(RG16,        R16,        R16,       2,  4)
MAKE_COMPARE_FUNC(RG16SN,      R16SN,      R16SN,     2,  4)
MAKE_COMPARE_FUNC(RG16UI,      R16UI,      R16UI,     2,  4)
MAKE_COMPARE_FUNC(RG16I,       R16I,       R16I,      2,  4)
MAKE_COMPARE_FUNC(RG32F,       R32F,       R32F,      4,  4)
MAKE_COMPARE_FUNC(RG32UI,      R32UI,      R32UI,     4,  4)
MAKE_COMPARE_FUNC(RG32I,       R32I,       R32I,      4,  4)
MAKE_COMPARE_FUNC(RGB8,        RG8,        R8,        2,  8)
MAKE_COMPARE_FUNC(RGB8SN,      RG8SN,      R8SN,      2,  8)
MAKE_COMPARE_FUNC(RGB8UI,      RG8UI,      R8UI,      2,  8)
MAKE_COMPARE_FUNC(RGB8I,       RG8I,       R8I,       2,  8)
MAKE_COMPARE_FUNC(RGB16F,      RG16F,      R16F,      4,  8)
MAKE_COMPARE_FUNC(RGB16,       RG16,       R16,       4,  8)
MAKE_COMPARE_FUNC(RGB16SN,     RG16SN,     R16SN,     4,  8)
MAKE_COMPARE_FUNC(RGB16UI,     RG16UI,     R16UI,     4,  8)
MAKE_COMPARE_FUNC(RGB16I,      RG16I,      R16I,      4,  8)
MAKE_COMPARE_FUNC(RGB32F,      RG32F,      R32F,      8,  8)
MAKE_COMPARE_FUNC(RGB32UI,     RG32UI,     R32UI,     8,  8)
MAKE_COMPARE_FUNC(RGB32I,      RG32I,      R32I,      8,  8)
MAKE_COMPARE_FUNC(RGBA8,       RGB8,       R8,        3, 12)
MAKE_COMPARE_FUNC(RGBA8SN,     RGB8SN,     R8SN,      3, 12)
MAKE_COMPARE_FUNC(RGBA8UI,     RGB8UI,     R8UI,      3, 12)
MAKE_COMPARE_FUNC(RGBA8I,      RGB8I,      R8I,       3, 12)
MAKE_COMPARE_FUNC(RGBA16F,     RGB16F,     R16F,      6, 12)
MAKE_COMPARE_FUNC(RGBA16,      RGB16,      R16,       6, 12)
MAKE_COMPARE_FUNC(RGBA16SN,    RGB16SN,    R16SN,     6, 12)
MAKE_COMPARE_FUNC(RGBA16UI,    RGB16UI,    R16UI,     6, 12)
MAKE_COMPARE_FUNC(RGBA16I,     RGB16I,     R16I,      6, 12)
MAKE_COMPARE_FUNC(RGBA32F,     RGB32F,     R32F,     12, 12)
MAKE_COMPARE_FUNC(RGBA32UI,    RGB32UI,    R32UI,    12, 12)
MAKE_COMPARE_FUNC(RGBA32I,     RGB32I,     R32I,     12, 12)
MAKE_COMPARE_FUNC(RG8_UI2F,    R8_UI2F,    R8_UI2F,   1,  4)
MAKE_COMPARE_FUNC(RG8_I2F,     R8_I2F,     R8_I2F,    1,  4)
MAKE_COMPARE_FUNC(RG16_UI2F,   R16_UI2F,   R16_UI2F,  2,  4)
MAKE_COMPARE_FUNC(RG16_I2F,    R16_I2F,    R16_I2F,   2,  4)
MAKE_COMPARE_FUNC(RG32_UI2F,   R32_UI2F,   R32_UI2F,  4,  4)
MAKE_COMPARE_FUNC(RG32_I2F,    R32_I2F,    R32_I2F,   4,  4)
MAKE_COMPARE_FUNC(RGB8_UI2F,   RG8_UI2F,   R8_UI2F,   2,  8)
MAKE_COMPARE_FUNC(RGB8_I2F,    RG8_I2F,    R8_I2F,    2,  8)
MAKE_COMPARE_FUNC(RGB16_UI2F,  RG16_UI2F,  R16_UI2F,  4,  8)
MAKE_COMPARE_FUNC(RGB16_I2F,   RG16_I2F,   R16_I2F,   4,  8)
MAKE_COMPARE_FUNC(RGB32_UI2F,  RG32_UI2F,  R32_UI2F,  8,  8)
MAKE_COMPARE_FUNC(RGB32_I2F,   RG32_I2F,   R32_I2F,   8,  8)
MAKE_COMPARE_FUNC(RGBA8_UI2F,  RGB8_UI2F,  R8_UI2F,   3, 12)
MAKE_COMPARE_FUNC(RGBA8_I2F,   RGB8_I2F,   R8_I2F,    3, 12)
MAKE_COMPARE_FUNC(RGBA16_UI2F, RGB16_UI2F, R16_UI2F,  6, 12)
MAKE_COMPARE_FUNC(RGBA16_I2F,  RGB16_I2F,  R16_I2F,   6, 12)
MAKE_COMPARE_FUNC(RGBA32_UI2F, RGB32_UI2F, R32_UI2F, 12, 12)
MAKE_COMPARE_FUNC(RGBA32_I2F,  RGB32_I2F,  R32_I2F,  12, 12)

#undef MAKE_COMPARE_FUNC

template <typename T>
union RGB10A2Type {
    RGB10A2Type(const uint8_t *x) : value(RCCD(T, x)) {}
    struct {
        T r : 10;
        T g : 10;
        T b : 10;
        T a : 2 ;
    } bits;
    T value;
};
typedef RGB10A2Type<LWNuint> RGB10A2UI;
typedef RGB10A2Type<LWNint>  RGB10A2I;

static bool compareRGB10A2(const uint8_t *compare, const uint8_t *result) {
    RGB10A2UI rgb10a2(compare);
    compareUI x[4];
    x[0].fValue = rgb10a2.bits.r/LWNfloat((1 << 10) - 1);
    x[1].fValue = rgb10a2.bits.g/LWNfloat((1 << 10) - 1);
    x[2].fValue = rgb10a2.bits.b/LWNfloat((1 << 10) - 1);
    x[3].fValue = rgb10a2.bits.a/LWNfloat((1 <<  2) - 1);
    LWNuint y[4];
    unpackValues(y, result, 4);
    return (x[0].value == y[0] && x[1].value == y[1] && x[2].value == y[2] && x[3].value == y[3]);
}

static bool compareRGB10A2SN(const uint8_t *compare, const uint8_t *result) {
    RGB10A2I rgb10a2(compare);
    if (rgb10a2.bits.r < -((1 << 9) - 1)) { rgb10a2.bits.r = -((1 << 9) - 1); }
    if (rgb10a2.bits.g < -((1 << 9) - 1)) { rgb10a2.bits.g = -((1 << 9) - 1); }
    if (rgb10a2.bits.b < -((1 << 9) - 1)) { rgb10a2.bits.b = -((1 << 9) - 1); }
    if (rgb10a2.bits.a < -((1 << 1) - 1)) { rgb10a2.bits.a = -((1 << 1) - 1); }
    compareI x[4];
    x[0].fValue = rgb10a2.bits.r/LWNfloat((1 << 9) - 1);
    x[1].fValue = rgb10a2.bits.g/LWNfloat((1 << 9) - 1);
    x[2].fValue = rgb10a2.bits.b/LWNfloat((1 << 9) - 1);
    x[3].fValue = rgb10a2.bits.a/LWNfloat((1 << 1) - 1);
    LWNint y[4];
    unpackValues(y, result, 4);
    return (x[0].value == y[0] && x[1].value == y[1] && x[2].value == y[2] && x[3].value == y[3]);
}

static bool compareRGB10A2UI(const uint8_t *compare, const uint8_t *result) {
    RGB10A2UI rgb10a2(compare);
    LWNuint y[4];
    unpackValues(y, result, 4);
    return (rgb10a2.bits.r == y[0] && rgb10a2.bits.g == y[1] && rgb10a2.bits.b == y[2] && rgb10a2.bits.a == y[3]);
}

static bool compareRGB10A2I(const uint8_t *compare, const uint8_t *result) {
    RGB10A2I rgb10a2(compare);
    LWNint y[4];
    unpackValues(y, result, 4);
    return (rgb10a2.bits.r == y[0] && rgb10a2.bits.g == y[1] && rgb10a2.bits.b == y[2] && rgb10a2.bits.a == y[3]);
}

static bool compareRGB10A2_UI2F(const uint8_t *compare, const uint8_t *result) {
    RGB10A2UI rgb10a2(compare);
    compareUI x[4];
    x[0].fValue = LWNfloat(rgb10a2.bits.r);
    x[1].fValue = LWNfloat(rgb10a2.bits.g);
    x[2].fValue = LWNfloat(rgb10a2.bits.b);
    x[3].fValue = LWNfloat(rgb10a2.bits.a);
    LWNuint y[4];
    unpackValues(y, result, 4);
    return (x[0].value == y[0] && x[1].value == y[1] && x[2].value == y[2] && x[3].value == y[3]);
}

static bool compareRGB10A2_I2F(const uint8_t *compare, const uint8_t *result) {
    RGB10A2I rgb10a2(compare);
    compareI x[4];
    x[0].fValue = LWNfloat(rgb10a2.bits.r);
    x[1].fValue = LWNfloat(rgb10a2.bits.g);
    x[2].fValue = LWNfloat(rgb10a2.bits.b);
    x[3].fValue = LWNfloat(rgb10a2.bits.a);
    LWNint y[4];
    unpackValues(y, result, 4);
    return (x[0].value == y[0] && x[1].value == y[1] && x[2].value == y[2] && x[3].value == y[3]);
}

/**********************************************************************/
// Randomization functions for floating point attributes
// Avoiding INF and NaN
union FloatBits {
   LWNfloat f;
   struct {
       LWNuint mant : 23;
       LWNuint expo :  8;
       LWNuint sgn  :  1;
   } u;
   bool nan() const { return (u.expo == 0xff); }
};

typedef void (*RandomFunction)(uint8_t *result);

static void randomR16F(uint8_t *result) {
    FloatBits x;
    LWs10e5   i;
    do {
        i = lwIntRand(0, 65535);
        x.f = lwS10E5toF32(i);
    } while (x.nan());
    memcpy(result, &i, 2);
}

static void randomR32F(uint8_t *result) {
    FloatBits x;
    LWNuint i;
    do {
        i = lwBitRand(32);
        ct_assert(sizeof(i) == sizeof(x.f));
        memcpy(&x.f, &i, sizeof(x.f));
    } while (x.nan());
    memcpy(result, &i, 4);
}

static void randomR32_UI2F(uint8_t *result) {
    LWNuint   i;
    i = lwIntRand(0, 16777215);
    memcpy(result, &i, 4);
}

static void randomR32_I2F(uint8_t *result) {
    LWNint    i;
    i = lwIntRand(-8388607, 8388607);
    memcpy(result, &i, 4);
}

static void randomR11G11B10F(uint8_t *result) {
    LWNuint   i;
    FloatBits x[3];
    do {
        i = lwBitRand(32);
        lwColwertR11fG11fB10fToFloat3(i, &(x[0].f));
    } while (x[0].nan() || x[1].nan() || x[2].nan());
    memcpy(result, &i, 4);
}

#define MAKE_RANDOM_FUNC(_a, _b, _c, _d)   \
static void random##_a(uint8_t *result) { \
    random##_b(&result[ 0]);               \
    random##_c(&result[_d]);               \
}

MAKE_RANDOM_FUNC(RG16F,       R16F,       R16F,      2)
MAKE_RANDOM_FUNC(RG32F,       R32F,       R32F,      4)
MAKE_RANDOM_FUNC(RGB16F,      RG16F,      R16F,      4)
MAKE_RANDOM_FUNC(RGB32F,      RG32F,      R32F,      8)
MAKE_RANDOM_FUNC(RGBA16F,     RGB16F,     R16F,      6)
MAKE_RANDOM_FUNC(RGBA32F,     RGB32F,     R32F,     12)
MAKE_RANDOM_FUNC(RG32_UI2F,   R32_UI2F,   R32_UI2F,  4)
MAKE_RANDOM_FUNC(RG32_I2F,    R32_I2F,    R32_I2F,   4)
MAKE_RANDOM_FUNC(RGB32_UI2F,  RG32_UI2F,  R32_UI2F,  8)
MAKE_RANDOM_FUNC(RGB32_I2F,   RG32_I2F,   R32_I2F,   8)
MAKE_RANDOM_FUNC(RGBA32_UI2F, RGB32_UI2F, R32_UI2F, 12)
MAKE_RANDOM_FUNC(RGBA32_I2F,  RGB32_I2F,  R32_I2F,  12)

#undef MAKE_RANDOM_FUNC

/**********************************************************************/
// Shader type strings
static const char *st[] = {
    "",
    "float",
    "uint",
    "int",
    "vec2",
    "uvec2",
    "ivec2",
    "vec3",
    "uvec3",
    "ivec3",
    "vec4",
    "uvec4",
    "ivec4"
};
#define LWN_SHADER_TYPE_NONE    0
#define LWN_SHADER_TYPE_FLOAT   1
#define LWN_SHADER_TYPE_UINT    2
#define LWN_SHADER_TYPE_INT     3
#define LWN_SHADER_TYPE_VEC2    4
#define LWN_SHADER_TYPE_UVEC2   5
#define LWN_SHADER_TYPE_IVEC2   6
#define LWN_SHADER_TYPE_VEC3    7
#define LWN_SHADER_TYPE_UVEC3   8
#define LWN_SHADER_TYPE_IVEC3   9
#define LWN_SHADER_TYPE_VEC4    10
#define LWN_SHADER_TYPE_UVEC4   11
#define LWN_SHADER_TYPE_IVEC4   12

/**********************************************************************/
// All legal vertex format types
static const struct {
    Format          format;
    uint8_t         size;
    uint8_t         components;
    uint8_t         shaderType;
    CompareFunction compare;
    RandomFunction  random;
} ft[] = {
#define LWN_FORMAT_TYPE(_f, _s, _c, _t, _rf) { Format::_f, _s, _c, LWN_SHADER_TYPE_##_t, compare##_f, _rf }
    LWN_FORMAT_TYPE(NONE,          0, 0, NONE,  NULL),
    LWN_FORMAT_TYPE(R8,            1, 1, FLOAT, NULL),
    LWN_FORMAT_TYPE(R8SN,          1, 1, FLOAT, NULL),
    LWN_FORMAT_TYPE(R8UI,          1, 1, UINT,  NULL),
    LWN_FORMAT_TYPE(R8I,           1, 1, INT,   NULL),
    LWN_FORMAT_TYPE(R16F,          2, 1, FLOAT, randomR16F),
    LWN_FORMAT_TYPE(R16,           2, 1, FLOAT, NULL),
    LWN_FORMAT_TYPE(R16SN,         2, 1, FLOAT, NULL),
    LWN_FORMAT_TYPE(R16UI,         2, 1, UINT,  NULL),
    LWN_FORMAT_TYPE(R16I,          2, 1, INT,   NULL),
    LWN_FORMAT_TYPE(R32F,          4, 1, FLOAT, randomR32F),
    LWN_FORMAT_TYPE(R32UI,         4, 1, UINT,  NULL),
    LWN_FORMAT_TYPE(R32I,          4, 1, INT,   NULL),
    LWN_FORMAT_TYPE(RG8,           2, 2, VEC2,  NULL),
    LWN_FORMAT_TYPE(RG8SN,         2, 2, VEC2,  NULL),
    LWN_FORMAT_TYPE(RG8UI,         2, 2, UVEC2, NULL),
    LWN_FORMAT_TYPE(RG8I,          2, 2, IVEC2, NULL),
    LWN_FORMAT_TYPE(RG16F,         4, 2, VEC2,  randomRG16F),
    LWN_FORMAT_TYPE(RG16,          4, 2, VEC2,  NULL),
    LWN_FORMAT_TYPE(RG16SN,        4, 2, VEC2,  NULL),
    LWN_FORMAT_TYPE(RG16UI,        4, 2, UVEC2, NULL),
    LWN_FORMAT_TYPE(RG16I,         4, 2, IVEC2, NULL),
    LWN_FORMAT_TYPE(RG32F,         8, 2, VEC2,  randomRG32F),
    LWN_FORMAT_TYPE(RG32UI,        8, 2, UVEC2, NULL),
    LWN_FORMAT_TYPE(RG32I,         8, 2, IVEC2, NULL),
    LWN_FORMAT_TYPE(RGB8,          3, 3, VEC3,  NULL),
    LWN_FORMAT_TYPE(RGB8SN,        3, 3, VEC3,  NULL),
    LWN_FORMAT_TYPE(RGB8UI,        3, 3, UVEC3, NULL),
    LWN_FORMAT_TYPE(RGB8I,         3, 3, IVEC3, NULL),
    LWN_FORMAT_TYPE(RGB16F,        6, 3, VEC3,  randomRGB16F),
    LWN_FORMAT_TYPE(RGB16,         6, 3, VEC3,  NULL),
    LWN_FORMAT_TYPE(RGB16SN,       6, 3, VEC3,  NULL),
    LWN_FORMAT_TYPE(RGB16UI,       6, 3, UVEC3, NULL),
    LWN_FORMAT_TYPE(RGB16I,        6, 3, IVEC3, NULL),
    LWN_FORMAT_TYPE(RGB32F,       12, 3, VEC3,  randomRGB32F),
    LWN_FORMAT_TYPE(RGB32UI,      12, 3, UVEC3, NULL),
    LWN_FORMAT_TYPE(RGB32I,       12, 3, IVEC3, NULL),
    LWN_FORMAT_TYPE(RGBA8,         4, 4, VEC4,  NULL),
    LWN_FORMAT_TYPE(RGBA8SN,       4, 4, VEC4,  NULL),
    LWN_FORMAT_TYPE(RGBA8UI,       4, 4, UVEC4, NULL),
    LWN_FORMAT_TYPE(RGBA8I,        4, 4, IVEC4, NULL),
    LWN_FORMAT_TYPE(RGBA16F,       8, 4, VEC4,  randomRGBA16F),
    LWN_FORMAT_TYPE(RGBA16,        8, 4, VEC4,  NULL),
    LWN_FORMAT_TYPE(RGBA16SN,      8, 4, VEC4,  NULL),
    LWN_FORMAT_TYPE(RGBA16UI,      8, 4, UVEC4, NULL),
    LWN_FORMAT_TYPE(RGBA16I,       8, 4, IVEC4, NULL),
    LWN_FORMAT_TYPE(RGBA32F,      16, 4, VEC4,  randomRGBA32F),
    LWN_FORMAT_TYPE(RGBA32UI,     16, 4, UVEC4, NULL),
    LWN_FORMAT_TYPE(RGBA32I,      16, 4, IVEC4, NULL),
    LWN_FORMAT_TYPE(RGB10A2,       4, 4, VEC4,  NULL),
    LWN_FORMAT_TYPE(RGB10A2SN,     4, 4, VEC4,  NULL),
    LWN_FORMAT_TYPE(RGB10A2UI,     4, 4, UVEC4, NULL),
    LWN_FORMAT_TYPE(RGB10A2I,      4, 4, IVEC4, NULL),
    LWN_FORMAT_TYPE(R11G11B10F,    4, 3, VEC3,  randomR11G11B10F),
    LWN_FORMAT_TYPE(R8_UI2F,       1, 1, FLOAT, NULL),
    LWN_FORMAT_TYPE(R8_I2F,        1, 1, FLOAT, NULL),
    LWN_FORMAT_TYPE(R16_UI2F,      2, 1, FLOAT, NULL),
    LWN_FORMAT_TYPE(R16_I2F,       2, 1, FLOAT, NULL),
    LWN_FORMAT_TYPE(R32_UI2F,      4, 1, FLOAT, randomR32_UI2F),
    LWN_FORMAT_TYPE(R32_I2F,       4, 1, FLOAT, randomR32_I2F),
    LWN_FORMAT_TYPE(RG8_UI2F,      2, 2, VEC2,  NULL),
    LWN_FORMAT_TYPE(RG8_I2F,       2, 2, VEC2,  NULL),
    LWN_FORMAT_TYPE(RG16_UI2F,     4, 2, VEC2,  NULL),
    LWN_FORMAT_TYPE(RG16_I2F,      4, 2, VEC2,  NULL),
    LWN_FORMAT_TYPE(RG32_UI2F,     8, 2, VEC2,  randomRG32_UI2F),
    LWN_FORMAT_TYPE(RG32_I2F,      8, 2, VEC2,  randomRG32_I2F),
    LWN_FORMAT_TYPE(RGB8_UI2F,     3, 3, VEC3,  NULL),
    LWN_FORMAT_TYPE(RGB8_I2F,      3, 3, VEC3,  NULL),
    LWN_FORMAT_TYPE(RGB16_UI2F,    6, 3, VEC3,  NULL),
    LWN_FORMAT_TYPE(RGB16_I2F,     6, 3, VEC3,  NULL),
    LWN_FORMAT_TYPE(RGB32_UI2F,   12, 3, VEC3,  randomRGB32_UI2F),
    LWN_FORMAT_TYPE(RGB32_I2F,    12, 3, VEC3,  randomRGB32_I2F),
    LWN_FORMAT_TYPE(RGBA8_UI2F,    4, 4, VEC4,  NULL),
    LWN_FORMAT_TYPE(RGBA8_I2F,     4, 4, VEC4,  NULL),
    LWN_FORMAT_TYPE(RGBA16_UI2F,   8, 4, VEC4,  NULL),
    LWN_FORMAT_TYPE(RGBA16_I2F,    8, 4, VEC4,  NULL),
    LWN_FORMAT_TYPE(RGBA32_UI2F,  16, 4, VEC4,  randomRGBA32_UI2F),
    LWN_FORMAT_TYPE(RGBA32_I2F,   16, 4, VEC4,  randomRGBA32_I2F),
    LWN_FORMAT_TYPE(RGB10A2_UI2F,  4, 4, UVEC4, NULL),
    LWN_FORMAT_TYPE(RGB10A2_I2F,   4, 4, IVEC4, NULL)
#undef LWN_FORMAT_TYPE
};

/**********************************************************************/
// VertexShader randomization class
class VertexShaderRandom {
public:
    uint8_t    numAttribs;                     // Number of attributes
    uint8_t    formats[LWN_MAX_ATTRIBS];       // Format index values

    void randomize() {
        numAttribs = lwIntRand(1, LWN_MAX_ATTRIBS);
        for (uint8_t i = 0; i < numAttribs; i++) {
            formats[i] = lwIntRand(1, __GL_ARRAYSIZE(ft) - 1);
        }
    }

    uint8_t randomFormat(uint8_t attrib) const
    {
        assert(attrib < numAttribs);
        uint8_t shaderFormat = formats[attrib];
        uint8_t newFormat;
        do {
            newFormat = lwIntRand(1, __GL_ARRAYSIZE(ft) - 1);
        } while (ft[newFormat].shaderType != ft[shaderFormat].shaderType);
        return newFormat;
    }
};

// VertexStreamSet randomization class
// Inherits from VertexStreamSet
class VertexStreamSetRandom : public VertexStreamSet {
public:
    VertexStream        *vs;
    uint8_t      numAttribs;                  // Number of attributes
    uint8_t      numStreams;                  // Number of streams
    uint8_t         streams[LWN_MAX_ATTRIBS]; // The stream index of each attribute
    LWNuint       inOffsets[LWN_MAX_ATTRIBS]; // The relative offset for each attribute
    LWNuint      outOffsets[LWN_MAX_ATTRIBS]; // The relative offset for the output attributes
    LWNuint       inStrides[LWN_MAX_STREAMS]; // The vertex stride for each stream
    LWNuint       outStride;                  // The vertex stride of the output buffer
    CompareFunction compare[LWN_MAX_ATTRIBS]; // Pointer to input/output comparison function for each attribute
    RandomFunction   random[LWN_MAX_ATTRIBS]; // Pointer to special case randomization functions

    VertexStreamSetRandom() :
        vs(NULL),
        numAttribs(0),
        numStreams(0),
        outStride(0) {
            for (uint8_t i = 0; i < LWN_MAX_ATTRIBS; i++) {
                streams[i] = 0;
                inOffsets[i] = 0;
                outOffsets[i] = 0;
                compare[i] = NULL;
                random[i] = NULL;
            }
            for (uint8_t i = 0; i < LWN_MAX_STREAMS; i++) {
                inStrides[i] = 0;
            }
        }
    ~VertexStreamSetRandom() {
        if (vs) {
            delete[] vs;
            vs = NULL;
        }
    }

    bool randomize(VertexShaderRandom &shader) {

        if (vs) {
            return false;
        }
        uint8_t formats[LWN_MAX_ATTRIBS];
        uint8_t   sizes[LWN_MAX_ATTRIBS];

        // Pick a random number of streams to hold those attributes
        // Allocate storage for the picked number of vertex streams
        numAttribs = shader.numAttribs;
        numStreams = lwIntRand(1, numAttribs);
        vs = new VertexStream[numStreams];
        if (!vs) {
            LOG(("Failed to allocate %d bytes for VertexStreams.\n", numStreams * sizeof(VertexStream)));
            return false;
        }

        // Pick a random format for each attribute
        // Use at least one attribute from each stream
        // Spread any remaining attributes across streams
        for (uint8_t i = 0; i < numAttribs; i++) {
            formats[i] = shader.randomFormat(i);
            sizes[i] = ft[formats[i]].size;
            if (i < numStreams) {
                streams[i] = 1;
            }
            else {
                streams[lwIntRand(0, numStreams - 1)]++;
            }
        }

        // Get vertex stride per stream
        // Get attribute offsets within stream
        // Possibly randomize the vertex size (with unused padding)
        uint8_t attribIndex = 0;
        for (uint8_t i = 0; i < numStreams; i++) {
            uint8_t firstAttrib = attribIndex;

            inStrides[i] = 0;
            for (uint8_t j = 0; j < streams[i]; j++) {
                inOffsets[attribIndex] = inStrides[i];
                inStrides[i] += sizes[attribIndex];
                attribIndex++;
            }

            // This is an arbitrary number of bytes.
            // Maximum allowed stride is 4095, but that is huge.
            // Pick a number of bytes of random padding
            // Randomly sprinkle the pad bytes around the vertex
            if (inStrides[i] < 64) {
                uint8_t padding = lwIntRand(0, 64 - inStrides[i]);
                inStrides[i] += padding;
                while (padding) {
                    padding--;
                    for (uint8_t j = lwIntRand(0, streams[i]); j < streams[i]; j++) {
                        inOffsets[firstAttrib + j]++;
                    }
                }
            }
        }

        // Add attributes to a new vertex stream
        // Add the new stream to the VertexStateSet
        attribIndex = 0;
        for (uint8_t i = 0; i < numStreams; i++) {
            vs[i].setStride(inStrides[i]);
            for (uint8_t j = 0; j < streams[i]; j++) {
                vs[i].addAttributeExplicit(
                    ft[formats[attribIndex]].format,
                    inOffsets[attribIndex]);
                attribIndex++;
            }
            addStream(vs[i]);
        }

        // Colwert the streams array to the stream index for each attribute
        {   attribIndex = 0;
            uint8_t newStreams[LWN_MAX_ATTRIBS];
            for (uint8_t i = 0; i < numStreams; i++) {
                for (uint8_t j = 0; j < streams[i]; j++) {
                    newStreams[attribIndex++] = i;
                }
            }
            for (uint8_t i = 0; i < LWN_MAX_ATTRIBS; i++) {
                streams[i] = newStreams[i];
            }
        }

        // Fill in result buffer stride and output buffer offsets
        outStride = 0;
        for (uint8_t i = 0; i < numAttribs; i++) {
            outOffsets[i] = outStride;
            outStride += (ft[formats[i]].components * 4);
        }

        // Fill in shader type strings
        // Fill in compare functions
        // Fill in random functions
        for (uint8_t i = 0; i < numAttribs; i++) {
            compare[i] = ft[formats[i]].compare;
            random[i] = ft[formats[i]].random;
        }

        LOG(("  numAttribs:  %d\n", numAttribs));
        LOG(("  numStreams:  %d\n", numStreams));
        LOG(("  streams:     %-3d", streams[0]));
        for (uint8_t i = 1; i < numAttribs; i++) {
            LOG((", %-3d", streams[i]));
        }
        LOG(("\n  inOffsets:   %-3d", inOffsets[0]));
        for (uint8_t i = 1; i < numAttribs; i++) {
            LOG((", %-3d", inOffsets[i]));
        }
        LOG(("\n  outOffsets:  %-3d", outOffsets[0]));
        for (uint8_t i = 1; i < numAttribs; i++) {
            LOG((", %-3d", outOffsets[i]));
        }
        LOG(("\n  inStrides:   %-3d", inStrides[0]));
        for (uint8_t i = 1; i < numStreams; i++) {
            LOG((", %-3d", inStrides[i]));
        }
        LOG(("\n  outStride:   %-3d\n", outStride));
        LOG(("  shaderTypes: %-5s", shaderTypes[0]));
        for (uint8_t i = 1; i < numAttribs; i++) {
            LOG((", %-5s", shaderTypes[i]));
        }
        LOG(("\n\n"));

        return true;
    }

    Buffer *AllocateVertexBuffer(LWNuint streamIndex,
                                 Device *device, int lwertices,
                                 MemoryPoolAllocator& allocator,
                                 const void *data) {
        return (streamIndex < numStreams) ?
            vs[streamIndex].AllocateVertexBuffer(device, lwertices, allocator, data) :
            NULL;
    }
};

/**********************************************************************/
// LWN Vertex attribute test class
class LWLWertexAttribTest
{
    static const LWNint numCells    = 1000;
    static const LWNint cellsPerCompile = 10;
    static const LWNint numVertices = 1000;

    void run(
        Device *device,
        Queue *queue,
        QueueCommandBuffer &cmd,
        MemoryPoolAllocator &allocator,
        VertexStreamSetRandom &vss,
        uint8_t *result, const LWNuint outSize,
        uint8_t *bufferData[LWN_MAX_STREAMS],
        LWNuint  bufferSize[LWN_MAX_STREAMS]) const;
    bool verify(
        Device *device,
        Queue *queue,
        QueueCommandBuffer &cmd,
        MemoryPoolAllocator &allocator,
        VertexShaderRandom &shader) const;
    LWNboolean createProgram(Program *pgm, const VertexShaderRandom &shader) const;

public:
    LWNTEST_CppMethods();
};

static LWNboolean needsFlatQualifier(uint8_t shaderType)
{
    switch (shaderType)
    {
    case LWN_SHADER_TYPE_UINT:
    case LWN_SHADER_TYPE_INT:
    case LWN_SHADER_TYPE_UVEC2:
    case LWN_SHADER_TYPE_IVEC2:
    case LWN_SHADER_TYPE_UVEC3:
    case LWN_SHADER_TYPE_IVEC3:
    case LWN_SHADER_TYPE_UVEC4:
    case LWN_SHADER_TYPE_IVEC4:
        return LWN_TRUE;
    case LWN_SHADER_TYPE_FLOAT:
    case LWN_SHADER_TYPE_VEC2:
    case LWN_SHADER_TYPE_VEC3:
    case LWN_SHADER_TYPE_VEC4:
        return LWN_FALSE;
    default:
        assert(!"Unknown shader type");
    }

    return false;
}

LWNboolean LWLWertexAttribTest::createProgram(Program *pgm, const VertexShaderRandom &shader) const {

    const char *shaderTypes[LWN_MAX_ATTRIBS];
    for (LWNuint j = 0; j < shader.numAttribs; j++) {
        shaderTypes[j] = st[ft[shader.formats[j]].shaderType];
    }

    VertexShader vs(440);
    for (LWNuint j = 0; j < shader.numAttribs; j++) {
        vs << "layout(location=" << j << ") in " << shaderTypes[j] << " " << attrs[j] << ";\n";
    }
    for (LWNuint j = 0; j < shader.numAttribs; j++) {
        vs << "out " << shaderTypes[j] << " " << outs[j] << ";\n";
    }
    vs << "void main() {\n"
          "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n";
    for (LWNuint j = 0; j < shader.numAttribs; j++) {
        vs << "  " << outs[j] << " = " << attrs[j] <<";\n";
    }
    vs << "}\n";

    FragmentShader fs(440);
    for (LWNuint j = 0; j < shader.numAttribs; j++) {
        if (needsFlatQualifier(ft[shader.formats[j]].shaderType)) {
            fs << "flat ";
        }
        fs << "in " << shaderTypes[j] << " " << outs[j] << ";\n";
    }
    fs << "out vec4 fcolor;\n"
          "void main() {\n"
          "  fcolor = vec4(0.0, 0.0, 0.0, 1.0);\n"
          "}\n";

    g_glslcHelper->SetTransformFeedbackVaryings(shader.numAttribs, outs);
    return g_glslcHelper->CompileAndSetShaders(pgm, vs, fs);
}

lwString LWLWertexAttribTest::getDescription() const
{
    return
    "Runs 1000 iterations of the following test case using 1000 vertices drawn as points:\n"
    "  1. Picks a random number of vertex attributes [1-16]\n"
    "  2. Picks a random number of vertex streams [1 - number of vertex attributes chosen]\n"
    "  3. Picks a random number of vertex attributes for each vertex stream, where each\n"
    "     vertex stream gets at least 1 vertex attribute\n"
    "  4. Picks a random vertex attribute format for each vertex attribute\n"
    "  5. Pads the vertex stride for each vertex stream with a random number of unused bytes,\n"
    "     where vertex attribute offsets are shifted randomly within the larger vertex stride\n"
    "  6. Vertex buffers are allocated and populated with random data\n"
    "  7. The vertex buffers and state are bound and drawn with:\n"
    "       * A dynamically generated pass-through vertex -> fragment shader\n"
    "       * A transform feedback output buffer to capture the interleaved stream of vertex attributes\n"
    "  8. The transform feedback buffer is copied back and compared with the random input data\n"
    "  9. Because transform feedback translates input to:\n"
    "       * float\n"
    "       * unsigned int\n"
    "       * int\n"
    "     The comparison is made using a software translation for each input vertex attribute format.\n"
    "\n"
    "If all 1000 iterations correctly compare, the screen is cleared to green.\n"
    "Otherwise, the screen is cleared to red.\n";
}

int LWLWertexAttribTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(38, 1);
}

void LWLWertexAttribTest::run(
    Device *device,
    Queue *queue,
    QueueCommandBuffer &cmd,
    MemoryPoolAllocator &allocator,
    VertexStreamSetRandom &vss,
    uint8_t *result, const LWNuint outSize,
    uint8_t  * bufferData[LWN_MAX_STREAMS],
    LWNuint    bufferSize[LWN_MAX_STREAMS]) const
{
    // Allocate/populate vertex buffers
    Buffer *buffers[LWN_MAX_STREAMS] = {};
    for (uint8_t i = 0; i < vss.numStreams; i++) {
        buffers[i] = vss.AllocateVertexBuffer(i, device, numVertices, allocator, bufferData[i]);
        cmd.BindVertexBuffer(i, buffers[i]->GetAddress(), bufferSize[i]);
    }

    // Bind the Vertex Attribute/Stream state
    vss.CreateVertexArrayState().bind(cmd);


    // Allocate the transform feedback buffers
    BufferAlignBits xfbAlign = BufferAlignBits(BUFFER_ALIGN_TRANSFORM_FEEDBACK_BIT |
                                               BUFFER_ALIGN_COPY_READ_BIT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *xfbResult = allocator.allocBuffer(&bb, xfbAlign, outSize);
    BufferAddress xfbResultAddr = xfbResult->GetAddress();
    Buffer *xfbControl = allocator.allocBuffer(&bb, BUFFER_ALIGN_TRANSFORM_FEEDBACK_CONTROL_BIT, 32);
    BufferAddress xfbControlAddr = xfbControl->GetAddress();

    // Draw with transform feedback
    cmd.BindTransformFeedbackBuffer(0, xfbResult->GetAddress(), outSize);
    cmd.BeginTransformFeedback(xfbControlAddr);
    cmd.DrawArrays(DrawPrimitive::POINTS, 0, numVertices);
    cmd.EndTransformFeedback(xfbControlAddr);
    cmd.submit();
    queue->Finish();

    // Copy back results
    bb.SetDefaults();
    Buffer *readbackBuffer = allocator.allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, outSize);
    BufferAddress readbackBufferAddr = readbackBuffer->GetAddress();
    cmd.CopyBufferToBuffer(xfbResultAddr, readbackBufferAddr, outSize, CopyFlags::NONE);
    cmd.submit();
    g_lwnTracker->insertFence(queue);
    queue->Finish();
    memcpy(result, readbackBuffer->Map(), outSize);

    // Clean up
    allocator.freeBuffer(readbackBuffer);
    allocator.freeBuffer(xfbControl);
    allocator.freeBuffer(xfbResult);
    for (uint8_t i = 0; i < vss.numStreams; i++) {
        allocator.freeBuffer(buffers[i]);
    }
}

bool LWLWertexAttribTest::verify(
    Device *device,
    Queue *queue,
    QueueCommandBuffer &cmd,
    MemoryPoolAllocator &allocator,
    VertexShaderRandom &shader) const
{
    bool      ret = true;
    uint8_t  *result = NULL;
    LWNuint   outSize = 0;
    uint8_t  *bufferData[LWN_MAX_STREAMS] = {};
    LWNuint   bufferSize[LWN_MAX_STREAMS] = {};

    VertexStreamSetRandom vss;
    if (!vss.randomize(shader)) {
        LOG(("Failed to pick a random vertex stream set state.\n"));
        ret = false;
        goto cleanup;
    }

    // Allocate result buffer
    outSize = numVertices * vss.outStride;
    result = (uint8_t *)__LWOG_MALLOC(outSize);
    if (!result) {
        LOG(("Failed to allocate %d bytes for results comparison buffer.\n", outSize));
        ret = false;
        goto cleanup;
    }

    // Allocate and randomize input buffer data
    for (uint8_t i = 0; i < vss.numStreams; i++) {
        bufferSize[i] = numVertices * vss.inStrides[i];
        LWNuint allocWords = (bufferSize[i] + 3) / 4;
        bufferData[i] = (uint8_t *)__LWOG_MALLOC(4 * allocWords);
        if (!bufferData[i]) {
            LOG(("Failed to allocate %d bytes for vertex input stream %d.\n", bufferSize[i], i));
            ret = false;
            goto cleanup;
        }
        LWNuint *bufferDataUI = (LWNuint *) bufferData[i];
        for (LWNuint j = 0; j < allocWords; j++) {
            bufferDataUI[j] = lwBitRand(32);
        }
    }

    // Do any special case randomization
    for (uint8_t i = 0; i < vss.numAttribs; i++) {
        const RandomFunction random = vss.random[i];
        if (random) {
            uint8_t           *p = &bufferData[vss.streams[i]][vss.inOffsets[i]];
            const uint8_t   *end = &bufferData[vss.streams[i]][bufferSize[vss.streams[i]]];
            const LWNuint stride = vss.inStrides[vss.streams[i]];
            while (p < end) {
                random(p);
                p += stride;
            }
        }
    }

    // Run the test
    run(device, queue, cmd, allocator, vss, result, outSize, bufferData, bufferSize);

    // Compare result
    for (uint8_t i = 0; i < vss.numAttribs; i++) {
        const CompareFunction compare = vss.compare[i];
        const uint8_t              *p = &bufferData[vss.streams[i]][vss.inOffsets[i]];
        const uint8_t            *end = &bufferData[vss.streams[i]][bufferSize[vss.streams[i]]];
        const uint8_t              *q = &result[vss.outOffsets[i]];
        const LWNuint        inStride = vss.inStrides[vss.streams[i]];
        const LWNuint       outStride = vss.outStride;
        while (p < end) {
            if (!compare(p, q)) {
                LOG(("Mismatch on attribute %d.\n", i));
                ret = false;
                goto cleanup;
            }
            p +=  inStride;
            q += outStride;
        }
    }

    // Clean up
cleanup:
    for (uint8_t i = 0; i < LWN_MAX_STREAMS; i++) {
        if (bufferData[i]) {
            __LWOG_FREE(bufferData[i]);
            bufferData[i] = NULL;
        }
    }
    if (result) {
        __LWOG_FREE(result);
        result = NULL;
    }

    return ret;
}

void LWLWertexAttribTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &cmd = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();
    VertexShaderRandom shader;

    // Clear to green if pass, red if fail
    LWNfloat clearColor[4] = { 0, 1, 0, 1 };

    MemoryPoolAllocator allocator(device, NULL, 0x1000000UL, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    for (LWNuint i = 0; i < numCells; i++) {

        // Every Nth cell, including the first, we do a full randomization and
        // generate a new program.
        if ((i % cellsPerCompile) == 0) {
            shader.randomize();
            Program *pgm = device->CreateProgram();
            createProgram(pgm, shader);
            cmd.BindProgram(pgm, ShaderStageBits::ALL_GRAPHICS_BITS);
        }

        LOG(("%-5d:\n", i));
        if (!verify(device, queue, cmd, allocator, shader)) {
            LOG(("Failed on test iteration %d.\n", i));
            clearColor[0] = 1;
            clearColor[1] = 0;
            break;
        }
    }

    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    cmd.ClearColor(0, clearColor, LWN_CLEAR_COLOR_MASK_RGBA);
    cmd.submit();
    queue->Finish();
}

OGTEST_CppTest(LWLWertexAttribTest, lwn_vertex_attrib, );
