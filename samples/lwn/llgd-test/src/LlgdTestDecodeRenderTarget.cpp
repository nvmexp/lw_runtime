/*
 * Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <LlgdTest.h>
#include <LlgdTestUtil.h>
#include <LlgdTestUtilLWN.h>

#include <liblwn-llgd.h>

#include <class/cl9097.h>

#define VALIDATE_GET_RENDER_TARGET_SIZE(width, height, memory, format, expectedValue) \
    TEST_EQ_FMT(llgdLwnGetRenderTargetSize(width, height, memory, format), expectedValue, "llgdLwnGetRenderTargetSize(%d, %d, %d, %s) failed", width, height, memory, DescString(format))

static const uint32_t s_testColorTargetFormats[] = {
    LW9097_SET_COLOR_TARGET_FORMAT_V_DISABLED,
    LW9097_SET_COLOR_TARGET_FORMAT_V_RF32_GF32_BF32_AF32,
    LW9097_SET_COLOR_TARGET_FORMAT_V_RS32_GS32_BS32_X32,
    LW9097_SET_COLOR_TARGET_FORMAT_V_R16_G16_B16_A16,
    LW9097_SET_COLOR_TARGET_FORMAT_V_RF32_GF32,
    LW9097_SET_COLOR_TARGET_FORMAT_V_A8R8G8B8,
    LW9097_SET_COLOR_TARGET_FORMAT_V_R16_G16,
    LW9097_SET_COLOR_TARGET_FORMAT_V_BF10GF11RF11,
    LW9097_SET_COLOR_TARGET_FORMAT_V_RS32,
    LW9097_SET_COLOR_TARGET_FORMAT_V_X8RL8GL8BL8,
    LW9097_SET_COLOR_TARGET_FORMAT_V_R5G6B5,
    LW9097_SET_COLOR_TARGET_FORMAT_V_A1R5G5B5,
    LW9097_SET_COLOR_TARGET_FORMAT_V_G8R8,
    LW9097_SET_COLOR_TARGET_FORMAT_V_R16,
    LW9097_SET_COLOR_TARGET_FORMAT_V_RN8,
    LW9097_SET_COLOR_TARGET_FORMAT_V_RU8,
};
static const int s_numOfTestColorTargetFormats = sizeof(s_testColorTargetFormats) / sizeof(uint32_t);

static const uint8_t BytesPerElements(uint32_t format)
{
    switch (format)
    {
    case LW9097_SET_COLOR_TARGET_FORMAT_V_RF32_GF32_BF32_AF32:
    case LW9097_SET_COLOR_TARGET_FORMAT_V_RS32_GS32_BS32_X32:
        return 16;
    case LW9097_SET_COLOR_TARGET_FORMAT_V_R16_G16_B16_A16:
    case LW9097_SET_COLOR_TARGET_FORMAT_V_RF32_GF32:
        return 8;
    case LW9097_SET_COLOR_TARGET_FORMAT_V_A8R8G8B8:
    case LW9097_SET_COLOR_TARGET_FORMAT_V_R16_G16:
    case LW9097_SET_COLOR_TARGET_FORMAT_V_BF10GF11RF11:
    case LW9097_SET_COLOR_TARGET_FORMAT_V_RS32:
    case LW9097_SET_COLOR_TARGET_FORMAT_V_X8RL8GL8BL8:
        return 4;
    case LW9097_SET_COLOR_TARGET_FORMAT_V_R5G6B5:
    case LW9097_SET_COLOR_TARGET_FORMAT_V_A1R5G5B5:
    case LW9097_SET_COLOR_TARGET_FORMAT_V_G8R8:
    case LW9097_SET_COLOR_TARGET_FORMAT_V_R16:
        return 2;
    case LW9097_SET_COLOR_TARGET_FORMAT_V_RN8:
    case LW9097_SET_COLOR_TARGET_FORMAT_V_RU8:
        return 1;
    case LW9097_SET_COLOR_TARGET_FORMAT_V_DISABLED:
    default:
        return 0;
    }
}

static const char* DescString(uint32_t format)
{
    switch (format)
    {
    case LW9097_SET_COLOR_TARGET_FORMAT_V_DISABLED:
        return "NONE";
    case LW9097_SET_COLOR_TARGET_FORMAT_V_RF32_GF32_BF32_AF32:
        return "RF32_GF32_BF32_AF32";
    case LW9097_SET_COLOR_TARGET_FORMAT_V_RS32_GS32_BS32_X32:
        return "RS32_GS32_BS32_X32";
    case LW9097_SET_COLOR_TARGET_FORMAT_V_R16_G16_B16_A16:
        return "R16_G16_B16_A16";
    case LW9097_SET_COLOR_TARGET_FORMAT_V_RF32_GF32:
        return "RF32_GF32";
    case LW9097_SET_COLOR_TARGET_FORMAT_V_A8R8G8B8:
        return "A8R8G8B8";
    case LW9097_SET_COLOR_TARGET_FORMAT_V_R16_G16:
        return "R16_G16";
    case LW9097_SET_COLOR_TARGET_FORMAT_V_BF10GF11RF11:
        return "BF10GF11RF11";
    case LW9097_SET_COLOR_TARGET_FORMAT_V_RS32:
        return "RS32";
    case LW9097_SET_COLOR_TARGET_FORMAT_V_X8RL8GL8BL8:
        return "X8RL8GL8BL8";
    case LW9097_SET_COLOR_TARGET_FORMAT_V_R5G6B5:
        return "R5G6B5";
    case LW9097_SET_COLOR_TARGET_FORMAT_V_A1R5G5B5:
        return "A1R5G5B5";
    case LW9097_SET_COLOR_TARGET_FORMAT_V_G8R8:
        return "G8R8";
    case LW9097_SET_COLOR_TARGET_FORMAT_V_R16:
        return "R16";
    case LW9097_SET_COLOR_TARGET_FORMAT_V_RN8:
        return "RN8";
    case LW9097_SET_COLOR_TARGET_FORMAT_V_RU8:
        return "RU8";
    default:
        return "Not supported in this test";
    }
}


class DecodeRenderTargetValidator {
public:
    bool Test();

private:
    bool TestIsRenderTargetEnabled();
    bool TestGetRenderTargetNumLayers();
    bool TestGetRenderTargetSize();
};

bool DecodeRenderTargetValidator::Test()
{
    if (!TestIsRenderTargetEnabled()) { return false; }
    if (!TestGetRenderTargetNumLayers()) { return false; }
    if (!TestGetRenderTargetSize()) { return false; }
    return true;
}

// Test llgdLwnIsRenderTargetEnabled
bool DecodeRenderTargetValidator::TestIsRenderTargetEnabled()
{
    for (int i = 0; i < s_numOfTestColorTargetFormats; ++i)
    {
        uint32_t format = s_testColorTargetFormats[i];
        if (format != LW9097_SET_COLOR_TARGET_FORMAT_V_DISABLED) {
            TEST_EQ_FMT(llgdLwnIsRenderTargetEnabled(0x1000, format), true, "llgdLwnIsRenderTargetEnabled(%d, %s) failed", 0x1000, DescString(format));
            TEST_EQ_FMT(llgdLwnIsRenderTargetEnabled(0, format), false, "llgdLwnIsRenderTargetEnabled(%d, %s) failed", 0, DescString(format));
        }
        else {
            TEST_EQ_FMT(llgdLwnIsRenderTargetEnabled(0x1000, format), false, "llgdLwnIsRenderTargetEnabled(%d, %s) failed", 0x1000, DescString(format));
            TEST_EQ_FMT(llgdLwnIsRenderTargetEnabled(0, format), false, "llgdLwnIsRenderTargetEnabled(%d, %s) failed", 0, DescString(format));
        }
    }
    return true;
}

// Test llgdLwnGetRenderTargetNumLayers
bool DecodeRenderTargetValidator::TestGetRenderTargetNumLayers()
{
    // if layout == blockLinear
    uint32_t memory =
        LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_WIDTH_ONE_GOB << 0 |
        LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT_ONE_GOB << 4 |
        LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH_ONE_GOB << 8 |
        LW9097_SET_COLOR_TARGET_MEMORY_LAYOUT_BLOCKLINEAR << 12 |
        LW9097_SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_ARRAY_SIZE << 16;

    uint32_t thirdDimension = 32;
    TEST_EQ_FMT(llgdLwnGetRenderTargetNumLayers(memory, thirdDimension), thirdDimension, "llgdLwnGetRenderTargetNumLayers(%d, %d) failed", memory, thirdDimension);

    memory =
        LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_WIDTH_ONE_GOB << 0 |
        LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT_FOUR_GOBS << 4 |
        LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH_TWO_GOBS << 8 |
        LW9097_SET_COLOR_TARGET_MEMORY_LAYOUT_BLOCKLINEAR << 12 |
        LW9097_SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_ARRAY_SIZE << 16;

    TEST_EQ_FMT(llgdLwnGetRenderTargetNumLayers(memory, thirdDimension), thirdDimension, "llgdLwnGetRenderTargetNumLayers(%d, %d) failed", memory, thirdDimension);

    memory =
        LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_WIDTH_ONE_GOB << 0 |
        LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT_FOUR_GOBS << 4 |
        LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH_TWO_GOBS << 8 |
        LW9097_SET_COLOR_TARGET_MEMORY_LAYOUT_BLOCKLINEAR << 12 |
        LW9097_SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_DEPTH_SIZE << 16;

    TEST_EQ_FMT(llgdLwnGetRenderTargetNumLayers(memory, thirdDimension), 1, "llgdLwnGetRenderTargetNumLayers(%d, %d) failed", memory, thirdDimension);

    memory =
        LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_WIDTH_ONE_GOB << 0 |
        LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT_THIRTYTWO_GOBS << 4 |
        LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH_THIRTYTWO_GOBS << 8 |
        LW9097_SET_COLOR_TARGET_MEMORY_LAYOUT_BLOCKLINEAR << 12 |
        LW9097_SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_DEPTH_SIZE << 16;

    TEST_EQ_FMT(llgdLwnGetRenderTargetNumLayers(memory, thirdDimension), 1, "llgdLwnGetRenderTargetNumLayers(%d, %d) failed", memory, thirdDimension);

    // if layout != block linear
    memory =
        (LW9097_SET_COLOR_TARGET_MEMORY_LAYOUT_PITCH << 12) |
        (LW9097_SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_ARRAY_SIZE << 16);

    TEST_EQ_FMT(llgdLwnGetRenderTargetNumLayers(memory, 1), 1, "llgdLwnGetRenderTargetNumLayers(%d, %d) failed", memory, 1);

    return true;
}

// Test llgdLwnGetRenderTargetSize
bool DecodeRenderTargetValidator::TestGetRenderTargetSize()
{
    for (int i = 0; i < s_numOfTestColorTargetFormats; ++i)
    {
        uint32_t format = s_testColorTargetFormats[i];
        if (format != LW9097_SET_COLOR_TARGET_FORMAT_V_DISABLED)
        {
            // if layout == blockLinear
            // Gobs are 512B and 8 rows
            uint32_t smallestAllowedWidthBL = 64 / BytesPerElements(format);

            uint32_t memory =
                LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_WIDTH_ONE_GOB << 0 |
                LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT_ONE_GOB << 4 |
                LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH_ONE_GOB << 8 |
                LW9097_SET_COLOR_TARGET_MEMORY_LAYOUT_BLOCKLINEAR << 12 |
                LW9097_SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_ARRAY_SIZE << 16;

            // Note, dimensions can be non-power-of-two values
            VALIDATE_GET_RENDER_TARGET_SIZE(64, 60, memory, format, 64 * 64 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(64, 16, memory, format, 64 * 16 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(256, 999, memory, format, 256 * 1000 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(2048, 1024, memory, format, 2048 * 1024 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(smallestAllowedWidthBL, 15, memory, format, smallestAllowedWidthBL * 16 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(smallestAllowedWidthBL * 11, 900, memory, format, (smallestAllowedWidthBL * 11) * 904 * BytesPerElements(format));

            memory =
                LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_WIDTH_ONE_GOB << 0 |
                LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT_FOUR_GOBS << 4 |
                LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH_TWO_GOBS << 8 |
                LW9097_SET_COLOR_TARGET_MEMORY_LAYOUT_BLOCKLINEAR << 12 |
                LW9097_SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_ARRAY_SIZE << 16;

            VALIDATE_GET_RENDER_TARGET_SIZE(64, 60, memory, format, 64 * 64 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(64, 16, memory, format, 64 * 32 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(256, 999, memory, format, 256 * 1024 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(2048, 1024, memory, format, 2048 * 1024 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(smallestAllowedWidthBL, 15, memory, format, smallestAllowedWidthBL * 32 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(smallestAllowedWidthBL * 11, 900, memory, format, (smallestAllowedWidthBL * 11) * 928 * BytesPerElements(format));

            memory =
                LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_WIDTH_ONE_GOB << 0 |
                LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT_EIGHT_GOBS << 4 |
                LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH_SIXTEEN_GOBS << 8 |
                LW9097_SET_COLOR_TARGET_MEMORY_LAYOUT_BLOCKLINEAR << 12 |
                LW9097_SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_ARRAY_SIZE << 16;

            VALIDATE_GET_RENDER_TARGET_SIZE(64, 60, memory, format, 64 * 64 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(64, 16, memory, format, 64 * 64 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(256, 999, memory, format, 256 * 1024 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(2048, 1024, memory, format, 2048 * 1024 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(smallestAllowedWidthBL, 15, memory, format, smallestAllowedWidthBL * 64 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(smallestAllowedWidthBL * 11, 900, memory, format, (smallestAllowedWidthBL * 11) * 960 * BytesPerElements(format));

            memory =
                LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_WIDTH_ONE_GOB << 0 |
                LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_HEIGHT_THIRTYTWO_GOBS << 4 |
                LW9097_SET_COLOR_TARGET_MEMORY_BLOCK_DEPTH_THIRTYTWO_GOBS << 8 |
                LW9097_SET_COLOR_TARGET_MEMORY_LAYOUT_BLOCKLINEAR << 12 |
                LW9097_SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_DEPTH_SIZE << 16;

            VALIDATE_GET_RENDER_TARGET_SIZE(64, 60, memory, format, 64 * 256 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(64, 16, memory, format, 64 * 256 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(256, 999, memory, format, 256 * 1024 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(2048, 1024, memory, format, 2048 * 1024 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(smallestAllowedWidthBL, 15, memory, format, smallestAllowedWidthBL * 256 * BytesPerElements(format));
            VALIDATE_GET_RENDER_TARGET_SIZE(smallestAllowedWidthBL * 11, 900, memory, format, (smallestAllowedWidthBL * 11) * 1024 * BytesPerElements(format));

            // if layout != block linear
            memory =
                (LW9097_SET_COLOR_TARGET_MEMORY_LAYOUT_PITCH << 12) |
                (LW9097_SET_COLOR_TARGET_MEMORY_THIRD_DIMENSION_CONTROL_THIRD_DIMENSION_DEFINES_ARRAY_SIZE << 16);

            VALIDATE_GET_RENDER_TARGET_SIZE(64, 60, memory, format, 64 * 60);
            VALIDATE_GET_RENDER_TARGET_SIZE(64, 16, memory, format, 64 * 16);
            VALIDATE_GET_RENDER_TARGET_SIZE(256, 999, memory, format, 256 * 999);
            VALIDATE_GET_RENDER_TARGET_SIZE(2048, 1024, memory, format, 2048 * 1024);
            VALIDATE_GET_RENDER_TARGET_SIZE(8, 15, memory, format, 8 * 15);
            VALIDATE_GET_RENDER_TARGET_SIZE(88, 900, memory, format, 88 * 900);
        }
    }
    return true;
}

LLGD_DEFINE_TEST(DecodeRenderTarget, UNIT,
LwError Execute()
{
    DecodeRenderTargetValidator v;
    if (!v.Test())  { return LwError_IlwalidState; }
    else            { return LwSuccess;            }
}
); // LLGD_DEFINE_TEST
