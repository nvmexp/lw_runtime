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

#define VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(width, height, blockSize, format, expectedValue)       \
    TEST_EQ_FMT(llgdLwnGetDepthStencilTargetSize(width, height, blockSize, format), expectedValue,    \
                "llgdLwnGetDepthStencilTargetSize(%d, %d, %d, %s) failed. smallestAllowedWidthBL=%u", \
                width, height, blockSize, DescString(format), smallestAllowedWidthBL)

static const uint32_t s_allZtFormats[] = {
    LW9097_SET_ZT_FORMAT_V_Z16,
    LW9097_SET_ZT_FORMAT_V_Z24S8,
    LW9097_SET_ZT_FORMAT_V_ZF32,
    LW9097_SET_ZT_FORMAT_V_ZF32_X24S8,
};
static const int s_numOfZtFormats = sizeof(s_allZtFormats) / sizeof(uint32_t);

static const uint8_t BytesPerElements(uint32_t format)
{
    switch (format)
    {
    case LW9097_SET_ZT_FORMAT_V_Z16:
        return 2;
    case LW9097_SET_ZT_FORMAT_V_Z24S8:
    case LW9097_SET_ZT_FORMAT_V_ZF32:
        return 4;
    case LW9097_SET_ZT_FORMAT_V_ZF32_X24S8:
        return 8;
    default:
        return 0;
    }
}

static const char* DescString(uint32_t format)
{
    switch (format)
    {
    case LW9097_SET_ZT_FORMAT_V_Z16:
        return "Z16";
    case LW9097_SET_ZT_FORMAT_V_Z24S8:
        return "Z24S8";
    case LW9097_SET_ZT_FORMAT_V_ZF32:
        return "ZF32";
    case LW9097_SET_ZT_FORMAT_V_ZF32_X24S8:
        return "ZF32_X24S8";
    default:
        return "Invalid";
    }
}


class DecodeDepthStencilTargetValidator {
public:
    bool Test();

private:
    bool TestIsDepthStencilTargetEnabled();
    bool TestGetDepthStencilTargetNumLayers();
    bool TestGetDepthStencilTargetSize();
};

bool DecodeDepthStencilTargetValidator::Test()
{
    if (!TestIsDepthStencilTargetEnabled()) { return false; }
    if (!TestGetDepthStencilTargetNumLayers()) { return false;  }
    if (!TestGetDepthStencilTargetSize()) { return false; }
    return true;
}

// Test llgdLwnIsDepthStencilTargetEnabled
bool DecodeDepthStencilTargetValidator::TestIsDepthStencilTargetEnabled()
{
    for (int i = 0; i < s_numOfZtFormats; ++i)
    {
        uint32_t format = s_allZtFormats[i];
        TEST_EQ_FMT(llgdLwnIsDepthStencilTargetEnabled(0x1000, format, 1), true, "llgdLwnIsDepthStencilTargetEnabled(0x1000, %s, 1) failed", DescString(format));
        TEST_EQ_FMT(llgdLwnIsDepthStencilTargetEnabled(0, format, 1), false, "llgdLwnIsDepthStencilTargetEnabled(0, %s, 1) failed", DescString(format));
        TEST_EQ_FMT(llgdLwnIsDepthStencilTargetEnabled(0x1000, format, 0), false, "llgdLwnIsDepthStencilTargetEnabled(0x1000, %s, 0) failed", DescString(format));
        TEST_EQ_FMT(llgdLwnIsDepthStencilTargetEnabled(0, format, 0), false, "llgdLwnIsDepthStencilTargetEnabled(0, %s, 0) failed", DescString(format));
    }
    TEST_EQ(llgdLwnIsDepthStencilTargetEnabled(0x1000, 0, 1), false);
    TEST_EQ(llgdLwnIsDepthStencilTargetEnabled(0, 0, 1), false);
    TEST_EQ(llgdLwnIsDepthStencilTargetEnabled(0x1000, 0, 0), false);
    TEST_EQ(llgdLwnIsDepthStencilTargetEnabled(0, 0, 0), false);
    return true;
}

// Test llgdLwnGetDepthStencilTargetNumLayers
bool DecodeDepthStencilTargetValidator::TestGetDepthStencilTargetNumLayers()
{
    TEST_EQ(llgdLwnGetDepthStencilTargetNumLayers(1), 1);
    TEST_EQ(llgdLwnGetDepthStencilTargetNumLayers(32), 32);
    return true;
}

// Test llgdLwnGetDepthStencilTargetSize
bool DecodeDepthStencilTargetValidator::TestGetDepthStencilTargetSize()
{
    for (int i = 0; i < s_numOfZtFormats; ++i)
    {
        uint32_t format = s_allZtFormats[i];

        // Gobs are 512B and 8 rows
        uint32_t smallestAllowedWidthBL = 64 / BytesPerElements(format);

        uint32_t blockSize =
            LW9097_SET_ZT_BLOCK_SIZE_WIDTH_ONE_GOB << 0 |
            LW9097_SET_ZT_BLOCK_SIZE_HEIGHT_ONE_GOB << 4 |
            LW9097_SET_ZT_BLOCK_SIZE_DEPTH_ONE_GOB << 8;

        // Note, dimensions can be non-power-of-two values
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(64, 60, blockSize, format, 64 * 64 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(64, 16, blockSize, format, 64 * 16 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(256, 999, blockSize, format, 256 * 1000 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(2048, 1024, blockSize, format, 2048 * 1024 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(smallestAllowedWidthBL, 15, blockSize, format, smallestAllowedWidthBL * 16 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(smallestAllowedWidthBL * 11, 900, blockSize, format, (smallestAllowedWidthBL * 11) * 904 * BytesPerElements(format));

        blockSize =
            LW9097_SET_ZT_BLOCK_SIZE_WIDTH_ONE_GOB << 0 |
            LW9097_SET_ZT_BLOCK_SIZE_HEIGHT_FOUR_GOBS << 4 |
            LW9097_SET_ZT_BLOCK_SIZE_DEPTH_ONE_GOB << 8;

        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(64, 60, blockSize, format, 64 * 64 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(64, 16, blockSize, format, 64 * 32 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(256, 999, blockSize, format, 256 * 1024 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(2048, 1024, blockSize, format, 2048 * 1024 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(smallestAllowedWidthBL, 15, blockSize, format, smallestAllowedWidthBL * 32 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(smallestAllowedWidthBL * 11, 900, blockSize, format, (smallestAllowedWidthBL * 11) * 928 * BytesPerElements(format));

        blockSize =
            LW9097_SET_ZT_BLOCK_SIZE_WIDTH_ONE_GOB << 0 |
            LW9097_SET_ZT_BLOCK_SIZE_HEIGHT_EIGHT_GOBS << 4 |
            LW9097_SET_ZT_BLOCK_SIZE_DEPTH_ONE_GOB << 8;

        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(64, 60, blockSize, format, 64 * 64 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(64, 16, blockSize, format, 64 * 64 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(256, 999, blockSize, format, 256 * 1024 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(2048, 1024, blockSize, format, 2048 * 1024 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(smallestAllowedWidthBL, 15, blockSize, format, smallestAllowedWidthBL * 64 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(smallestAllowedWidthBL * 11, 900, blockSize, format, (smallestAllowedWidthBL * 11) * 960 * BytesPerElements(format));

        blockSize =
            LW9097_SET_ZT_BLOCK_SIZE_WIDTH_ONE_GOB << 0 |
            LW9097_SET_ZT_BLOCK_SIZE_HEIGHT_THIRTYTWO_GOBS << 4 |
            LW9097_SET_ZT_BLOCK_SIZE_DEPTH_ONE_GOB << 8;

        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(64, 60, blockSize, format, 64 * 256 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(64, 16, blockSize, format, 64 * 256 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(256, 999, blockSize, format, 256 * 1024 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(2048, 1024, blockSize, format, 2048 * 1024 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(smallestAllowedWidthBL, 15, blockSize, format, smallestAllowedWidthBL * 256 * BytesPerElements(format));
        VALIDATE_GET_DEPTH_STENCIL_TARGET_SIZE(smallestAllowedWidthBL * 11, 900, blockSize, format, (smallestAllowedWidthBL * 11) * 1024 * BytesPerElements(format));
    }
    return true;
}

LLGD_DEFINE_TEST(DecodeDepthStencilTarget, UNIT,
LwError Execute()
{
    DecodeDepthStencilTargetValidator v;
    if (!v.Test())  { return LwError_IlwalidState; }
    else            { return LwSuccess;            }
}
); // LLGD_DEFINE_TEST
