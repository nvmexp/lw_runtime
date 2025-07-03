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
#include "float_util.h"

using namespace lwn;

class LWNZbcTest
{
    // Basic component type used to represent the data we care about in a
    // particular format.
    enum ComponentType {
        UNORM,
        SNORM,
        UINT,
        SINT,
        FLOAT,
    };

    // Properties of an individual format to test.
    struct FormatInfo {
        Format format;
        bool supportsZBC;               // does hardware actually support ZBC for the format?
        ComponentType componentType;
        int components;                 // number of relevant components (0 for special formats)
        int componentBits;              // bits per component (0 for special formats)
    };

    // We test ZBC formats in batches. Because hardware supports only a limited
    // number of ZBC values and ZBC values can't be recycled without tearing
    // down just about everything, we can't test all possible formats in a
    // single test run because we will run out of indices. Instead, we split
    // the testing up into several numbered batches of sub-tests. Each batch
    // has a sufficiently small number of formats so that we will have enough
    // ZBC slots to register one color/depth value in each format.
    //
    // In a full test run, we will expect only the first batch to run will be
    // guaranteed to successfully register values. We select some of the more
    // common formats in this batch so that a normal regression run will test
    // these formats. To test all formats, indepenedent test runs selecting
    // only one batch at a time will be needed.
    struct FormatBatch {
        const FormatInfo *formats;
        int nFormats;
    };
    static const FormatBatch formatBatches[];
    static const size_t formatBatchCount;
    static uint32_t firstRunBatch;
    uint32_t m_batch;

    // Arrays for individual batches.
    static const FormatInfo formats0[], formats1[], formats2[], formats3[], formats4[];

    // Get the format information for entry <index> in the tested batch.
    const FormatInfo *getFormatInfo(int index) const
    {
        assert(m_batch < formatBatchCount);
        assert(index < formatBatches[m_batch].nFormats);
        return &formatBatches[m_batch].formats[index];
    }

    // Get the total number of formats in the tested batch.
    int getBatchFormatCount() const
    {
        assert(m_batch < formatBatchCount);
        return formatBatches[m_batch].nFormats;
    }

    // Get the aggregate index of the current batch's entry <index> across all
    // batches.
    int getAggregateFormatIndex(int index) const
    {
        assert(m_batch < formatBatchCount);
        assert(index < formatBatches[m_batch].nFormats);
        int result = index;
        for (uint32_t i = 0; i < m_batch; i++) {
            result += formatBatches[i].nFormats;
        }
        return result;
    }

    // Get the total number of formats in all batches.
    static int getTotalFormatCount()
    {
        int result = 0;
        for (uint32_t i = 0; i < formatBatchCount; i++) {
            result += formatBatches[i].nFormats;
        }
        return result;
    }

    // Record the beginning of a run. If this is the first time this test has
    // run, set <firstRunBatch> to record this as the test that should succeed
    // in registering ZBC values.
    void beginBatch() const
    {
        if (firstRunBatch == ~0U) {
            firstRunBatch = m_batch;
        }
    }

    // Returns true if we expect ZBC registration to always succeed. This only
    // happens on HOS (which recycles ZBC indices between app runs) and only on
    // the first batch run for a particular test run.
    bool zbcRegistrationMustPass() const
    {
#if defined(LW_TEGRA)
        return m_batch == firstRunBatch;
#else
        return false;
#endif
    }

    // For CheetAh X1, where we can make render targets with CPU-coherent memory,
    // this function checks if ZBC happened by looking at the first 128 bytes
    // of memory (which are nominally cleared). Returns true if nothing has
    // changed from the "magic" byte pattern (ZBC happened) or false otherwise.
    // Note that we could sometimes get away with checking only the first few
    // bytes, but those bytes may not be changed due to other non-ZBC
    // compression or bit scrambling done for depth/stencil formats.
    static bool checkFBChanges(uint8_t *texCpuAddress, uint8_t magic)
    {
        for (int i = 0; i < 128; i++) {
            if (texCpuAddress[i] != magic) {
                return true;
            }
        }
        return false;
    }

public:
    LWNZbcTest(uint32_t batch) : m_batch(batch) {}

    static const int FBO_WIDTH  = 128;
    static const int FBO_HEIGHT = 128;

    LWNTEST_CppMethods();

private:
    enum ClearTestType {
        CLEAR_ZERO,
        CLEAR_ONE,
        CLEAR_UNREGISTERED,
        CLEAR_REGISTERED
    };

    static const int cellSize = 40;
    static const int cellMargin = 2;
    static const int cellsX = 640 / cellSize;
    static const int cellsY = 480 / cellSize;

#if defined(LW_TEGRA)
    static const bool isCpuCoherentSupportedForRTs = true;
#else
    static const bool isCpuCoherentSupportedForRTs = false;
#endif

    bool apiTest(Device *dev) const;
    bool clearTest(DeviceState *deviceState, uint32_t formatIndex, ClearTestType clearType, bool useCompression) const;

    void drawCell(QueueCommandBuffer &queueCB, int cx, int cy, bool result) const;
    void drawResultCells(QueueCommandBuffer &queueCB, const std::vector<bool>& results) const;

    void setupClearValues(ClearTestType clearType, uint32_t formatIndex,
                          float *clearValues, uint8_t expectedValues[16], int *nBytes) const;
};

// Global holding the batch number of the first lwn_zbc test to be run. "~0U"
// means "I haven't run a test yet."
uint32_t LWNZbcTest::firstRunBatch = ~0U;

#if defined(LW_TEGRA)
#define IS_DEPTH16_ZBC_SUPPORTED true
#else
// There's lwrrently no support for DEPTH16 ZBC in LWN on Windows
#define IS_DEPTH16_ZBC_SUPPORTED false
#endif

//////////////////////////////////////////////////////////////////////////
//
// Batches of formats to be tested.
//

const LWNZbcTest::FormatInfo LWNZbcTest::formats0[] =
{
    { Format::RGBA8, true, UNORM, 4, 8 },
    { Format::RGBA8_SRGB, true, UNORM, 4, 8 },
    { Format::RGBA16F, true, FLOAT, 4, 16 },
    { Format::RGBA32F, true, FLOAT, 4, 32 },
    { Format::R32UI, true, UINT, 1, 32 },
    { Format::DEPTH16, false, UNORM, 0, 0 },
    { Format::DEPTH16, IS_DEPTH16_ZBC_SUPPORTED, UNORM, 0, 0 }, // uses "prefer ZBC" texture flag to enable DEPTH16 ZBC
    { Format::DEPTH24, true, UNORM, 0, 0 },
    { Format::DEPTH32F, true, FLOAT, 0, 0 },
    { Format::DEPTH24_STENCIL8, true, UNORM, 0, 0 },
    { Format::DEPTH32F_STENCIL8, true, FLOAT, 0, 0 },
};

const LWNZbcTest::FormatInfo LWNZbcTest::formats1[] =
{
    // This batch has a lot of small formats (<32 bits per pixel) that don't
    // support ZBC. Only ten total in this set do support ZBC, which fits into
    // the ~13 indices we should have available.
    { Format::R8, false, UNORM, 1, 8 },
    { Format::R8SN, false, SNORM, 1, 8 },
    { Format::R8UI, false, UINT, 1, 8 },
    { Format::R8I, false, SINT, 1, 8 },
    { Format::R16F, false, FLOAT, 1, 16 },
    { Format::R16, false, UNORM, 1, 16 },
    { Format::R16SN, false, SNORM, 1, 16 },
    { Format::R16UI, false, UINT, 1, 16 },
    { Format::R16I, false, SINT, 1, 16 },
    { Format::R32F, true, FLOAT, 1, 32 },
    { Format::R32I, true, SINT, 1, 32 },
    { Format::RG8, false, UNORM, 2, 8 },
    { Format::RG8SN, false, SNORM, 2, 8 },
    { Format::RG8UI, false, UINT, 2, 8 },
    { Format::RG8I, false, SINT, 2, 8 },
    { Format::RG16F, true, FLOAT, 2, 16 },
    { Format::RG16, true, UNORM, 2, 16 },
    { Format::RG16SN, true, SNORM, 2, 16 },
    { Format::RG16UI, true, UINT, 2, 16 },
    { Format::RG16I, true, SINT, 2, 16 },
    { Format::RG32F, true, FLOAT, 2, 32 },
    { Format::RG32UI, true, UINT, 2, 32 },
    { Format::RG32I, true, SINT, 2, 32 },
};

const LWNZbcTest::FormatInfo LWNZbcTest::formats2[] =
{
    { Format::RGBA8SN, true, SNORM, 4, 8 },
    { Format::RGBA8UI, true, UINT, 4, 8 },
    { Format::RGBA8I, true, SINT, 4, 8 },
    { Format::RGBA16, true, UNORM, 4, 16 },
    { Format::RGBA16SN, true, SNORM, 4, 16 },
    { Format::RGBA16UI, true, UINT, 4, 16 },
    { Format::RGBA16I, true, SINT, 4, 16 },
    { Format::RGBA32UI, true, UINT, 4, 32 },
    { Format::RGBA32I, true, SINT, 4, 32 },
    { Format::RGBX8_SRGB, true, UNORM, 3, 8 },
};

const LWNZbcTest::FormatInfo LWNZbcTest::formats3[] =
{
    { Format::RGB10A2, true, UNORM, 0, 0 },
    { Format::RGB10A2UI, true, UINT, 0, 0 },
    { Format::R11G11B10F, true, FLOAT, 0, 0 },
    { Format::RGBX8, true, UNORM, 3, 8 },
    { Format::RGBX8SN, true, SNORM, 3, 8 },
    { Format::RGBX8UI, true, UINT, 3, 8 },
    { Format::RGBX8I, true, SINT, 3, 8 },
    { Format::RGBX16F, true, FLOAT, 3, 16 },
    { Format::RGBX16, true, UNORM, 3, 16 },
    { Format::RGBX16SN, true, SNORM, 3, 16 },
};

const LWNZbcTest::FormatInfo LWNZbcTest::formats4[] =
{
    { Format::RGBX16UI, true, UINT, 3, 16 },
    { Format::RGBX16I, true, SINT, 3, 16 },
    { Format::RGBX32F, true, FLOAT, 3, 32 },
    { Format::RGBX32UI, true, UINT, 3, 32 },
    { Format::RGBX32I, true, SINT, 3, 32 },
    { Format::BGR565, false, UNORM, 0, 0 },
    { Format::BGR5, false, UNORM, 0, 0 },
    { Format::BGR5A1, false, UNORM, 0, 0 },
    { Format::BGRX8, true, UNORM, 3, 8 },
    { Format::BGRA8, true, UNORM, 4, 8 },
    { Format::BGRX8_SRGB, true, UNORM, 3, 8 },
    { Format::BGRA8_SRGB, true, UNORM, 4, 8 },
};

#define BATCH(n) { LWNZbcTest::formats ## n, __GL_ARRAYSIZE(LWNZbcTest::formats ## n) }
const LWNZbcTest::FormatBatch LWNZbcTest::formatBatches[] =
{
    BATCH(0),
    BATCH(1),
    BATCH(2),
    BATCH(3),
    BATCH(4),
};
const size_t LWNZbcTest::formatBatchCount = __GL_ARRAYSIZE(LWNZbcTest::formatBatches);
#undef BATCH


// Set up clear values for a particular test type and format. We compute
// clearValues to pass in the API into <clearValues>. We store the expected
// framebuffer byte pattern into <expectedValues> and store the number of
// framebuffer bytes to test in <nBytes>.
void LWNZbcTest::setupClearValues(ClearTestType clearType, uint32_t formatIndex,
                                  float *clearValues, uint8_t expectedValues[16],
                                  int *nBytes) const
{
    const FormatInfo *formatInfo = getFormatInfo(formatIndex);
    double c[4], scale;

    // Set up a base clear color based on the clear type.
    switch (clearType) {
    case CLEAR_ZERO:
        c[0] = c[1] = c[2] = c[3] = 0.0;
        break;
    case CLEAR_ONE:
        c[0] = c[1] = c[2] = c[3] = 1.0;
        break;
    case CLEAR_REGISTERED:
        c[0] = 0.3146;
        c[1] = 0.9012;
        c[2] = 0.0947;
        c[3] = 0.7777;
        break;
    case CLEAR_UNREGISTERED:
        c[0] = 0.1846;
        c[1] = 0.8712;
        c[2] = 0.3836;
        c[3] = 0.5737;
        break;
    }

    // When generating the color, "rotate" the values by adding in a
    // per-format bias to try and ensure that no two formats register the same
    // clear color.  Some formats alias as far as ZBC is concerned (e.g.,
    // "RGBA8" vs.  "RGBX8") and if values collide, one format may end up
    // getting ZBC because the same color was registered from the other
    // format.
    double formatBias = getAggregateFormatIndex(formatIndex) / getTotalFormatCount();
    if (clearType != CLEAR_ZERO && clearType != CLEAR_ONE) {
        for (int i = 0; i < 4; i++) {
            c[i] += formatBias;
            if (c[i] > 1.0) {
                c[i] -= 1.0;
            }
        }
    }

    // For basic cases (fixed component count and format), use generic code to
    // compute the expected clear values and framebuffer values.
    if (formatInfo->components && formatInfo->componentBits) {
        switch (formatInfo->componentType) {
        case UNORM:
        case UINT:
            // For UNORM/UINT, compute integer values in [0, 2^N-1] that
            // will be the expected framebuffer values.
            scale = pow(2.0, formatInfo->componentBits) - 1.0;
            for (int i = 0; i < 4; i++) {
                uint32_t uval = uint32_t(c[i] * scale);
                if (formatInfo->componentType == UNORM) {
                    clearValues[i] = double(uval) / scale;
                } else {
                    ((uint32_t *)clearValues)[i] = uval;
                }
                memcpy(expectedValues + i * formatInfo->componentBits / 8, &uval, formatInfo->componentBits / 8);
            }
            *nBytes = formatInfo->components * formatInfo->componentBits / 8;
            break;
        case SNORM:
        case SINT:
            // For SNORM/SINT, compute integer values in [-2^(N-1)-1,
            // 2^(N-1)-1] that will be the expected framebuffer values. For the
            // non-zero/one cases, remap our clear values in [0,1] into [-1,+1]
            // to test negative values.
            scale = pow(2.0, formatInfo->componentBits - 1) - 1.0;
            for (int i = 0; i < 4; i++) {
                if (clearType == CLEAR_REGISTERED || clearType == CLEAR_UNREGISTERED) {
                    c[i] = 2.0 * (c[i] - 0.5);
                }
                int32_t ival = int(c[i] * scale);
                if (formatInfo->componentType == SNORM) {
                    clearValues[i] = double(ival) / scale;
                } else {
                    ((int32_t *) clearValues)[i] = ival;
                }
                memcpy(expectedValues + i * formatInfo->componentBits / 8, &ival, formatInfo->componentBits / 8);
            }
            *nBytes = formatInfo->components * formatInfo->componentBits / 8;
            break;
        case FLOAT:
            // For FLOAT, remap our clear values in [0,1] into [-1,+1] and
            // quantize to 10 fraction bits so they can be exactly represented
            // in FP16 and FP32.
            scale = 1024.0;
            for (int i = 0; i < 4; i++) {
                if (clearType == CLEAR_REGISTERED || clearType == CLEAR_UNREGISTERED) {
                    c[i] = 2.0 * (c[i] - 0.5);
                }
                double d = int(c[i] * scale) / scale;
                clearValues[i] = d;
                if (formatInfo->componentBits == 32) {
                    ((float *)expectedValues)[i] = d;
                } else {
                    ((uint16_t *)expectedValues)[i] = lwF32toS10E5(d);
                }
            }
            *nBytes = formatInfo->components * formatInfo->componentBits / 8;
            break;
        default:
            assert(0);
            break;
        }
    }

    // Handle other special depth/stencil and packed formats.
    switch (formatInfo->format) {
    case Format::DEPTH16:
        scale = pow(2.0, 16) - 1;
        c[0] = int(c[0] * scale);
        clearValues[0] = c[0] / scale;
        ((uint32_t *) expectedValues)[0] = uint32_t(c[0]);
        *nBytes = 2;
        break;
    case Format::DEPTH24:
    case Format::DEPTH24_STENCIL8:
        scale = pow(2.0, 24) - 1;
        c[0] = int(c[0] * scale);
        clearValues[0] = c[0] / scale;
        ((uint32_t *)expectedValues)[0] = uint32_t(c[0]);
        *nBytes = 3;
        break;
    case Format::DEPTH32F:
    case Format::DEPTH32F_STENCIL8:
        clearValues[0] = c[0];
        ((float *)expectedValues)[0] = c[0];
        *nBytes = 4;
        break;
    case Format::RGB10A2:
    case Format::RGB10A2UI:
        for (int i = 0; i < 3; i++) {
            scale = pow(2.0, 10) - 1;
            c[i] = int(c[i] * scale);
            clearValues[i] = c[i] / scale;
        }
        scale = pow(2.0, 2) - 1;
        c[3] = int(c[3] * scale);
        clearValues[3] = c[3] / scale;
        ((uint32_t *) expectedValues)[0] = (uint32_t(c[0]) |
                                            uint32_t(c[1]) << 10 |
                                            uint32_t(c[2]) << 20 |
                                            uint32_t(c[3]) << 30);
        if (formatInfo->format == Format::RGB10A2UI) {
            for (int i = 0; i < 4; i++) {
                ((uint32_t *)clearValues)[i] = c[i];
            }
        }
        *nBytes = 4;
        break;
    case Format::R11G11B10F:
        for (int i = 0; i < 3; i++) {
            scale = pow(2.0, 7);
            c[i] = int(c[i] * scale);
            clearValues[i] = c[i] / scale;
        }
        ((uint32_t *) expectedValues)[0] = lwColwertFloat3ToR11fG11fB10f(clearValues);
        *nBytes = 4;
        break;
    case Format::BGR565:
    case Format::BGR5:
    case Format::BGR5A1:
        for (int i = 0; i < 3; i++) {
            scale = pow(2.0, 5) - 1;
            if (formatInfo->format == Format::BGR565 && i == 1) {
                scale = pow(2.0, 6) - 1;
            }
            c[i] = int(c[i] * scale);
            clearValues[i] = c[i] / scale;
        }
        c[3] = (c[3] >= 0.5) ? 1.0 : 0.0;
        clearValues[3] = c[3];
        switch (formatInfo->format) {
        case Format::BGR565:
            ((uint16_t *) expectedValues)[0] = (uint32_t(c[0]) |
                                                uint32_t(c[1]) << 5 |
                                                uint32_t(c[2]) << 11);
            break;
        case Format::BGR5:
            ((uint16_t *) expectedValues)[0] = (uint32_t(c[0]) |
                                                uint32_t(c[1]) << 5 |
                                                uint32_t(c[2]) << 10);
            break;
        case Format::BGR5A1:
            ((uint16_t *) expectedValues)[0] = (uint32_t(c[0]) |
                                                uint32_t(c[1]) << 5 |
                                                uint32_t(c[2]) << 10 |
                                                uint32_t(c[3]) << 15);
            break;
        default:
            assert(0);
            break;
        }
        *nBytes = 2;
        break;
    default:
        break;
    }

    // For BGR formats, treat the input as specifying packed BGR values. Swap
    // the red and blue clear values since hardware always takes an RGB clear
    // color.
    switch (formatInfo->format) {
    case Format::BGRA8:
    case Format::BGRA8_SRGB:
    case Format::BGRX8:
    case Format::BGRX8_SRGB:
    case Format::BGR565:
    case Format::BGR5:
    case Format::BGR5A1:
        c[0] = clearValues[0];
        clearValues[0] = clearValues[2];
        clearValues[2] = c[0];
        break;
    default:
        break;
    }

    // For sRGB formats, treat the input as specifying an sRGB-encoded
    // destination color. Remap the RGB clear colors to the linear color space.
    switch (formatInfo->format) {
    case Format::RGBX8_SRGB:
    case Format::RGBA8_SRGB:
    case Format::BGRX8_SRGB:
    case Format::BGRA8_SRGB:
        for (int i = 0; i < 3; i++) {
            clearValues[i] = srgbToLinear(clearValues[i]);
        }
        break;
    default:
        break;
    }
}

// Display the results of a subtest in cell (cx,cy) in red/green based on
// <result>.
void LWNZbcTest::drawCell(QueueCommandBuffer &queueCB, int cx, int cy, bool result) const
{
    queueCB.SetScissor(cx * cellSize + cellMargin, cy * cellSize + cellMargin,
                       cellSize - 2 * cellMargin, cellSize - 2* cellMargin);
    queueCB.ClearColor(0, result ? 0.0 : 1.0, result ? 1.0 : 0.0, 0.0, 1.0);
}

void LWNZbcTest::drawResultCells(QueueCommandBuffer &queueCB, const std::vector<bool> &results) const
{
    int test = 0;
    for (std::vector<bool>::const_iterator it = results.begin(); it != results.end(); ++it, test++) {
        drawCell(queueCB, test % cellsX, test / cellsX, *it);
    }
}
lwString LWNZbcTest::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Test for proper handling of fast clear color registration in LWN.  "
        "This test is one of several fast clear tests (" << int(m_batch) <<
        " of " << int(formatBatchCount) << "), each testing a different set of "
        "formats.  The test clears a texture for each format in the batch "
        "to different colors and checks for correct values in the framebuffer.  "
        "It also verifies whether fast clear registration should work or fail, "
        "and whether the clear color is written into framebuffer memory (full "
        "fast clears should not consume any memory bandwidth).  The test "
        "tests pre-registered clear colors, clearing to a manually registered "
        "color, and clearing to an unregistered color.  It also tests textures "
        "with and without compression.  Each combination of format, color, and "
        "compression status result in one test rectangle on-screen, drawn in "
        "green if the clear behaves as expected and red otherwise.\n"
        "\n"
        "Because we have more formats than available colors to register, "
        "only the first batch of formats is guaranteed to register "
        "successfully in a full test run.  If you run each sub-test "
        "individually, the test verifies that all supported formats are "
        "registered successfully.\n"
        "\n"
        "On Windows, there is no guarantee that even the first sub-test "
        "run will register successfully and there is also no support "
        "for manually checking framebuffer memory.  The test accounts "
        "for these limitations.";
    return sb.str();
}

int LWNZbcTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(51, 0);
}

bool LWNZbcTest::apiTest(Device *device) const
{
    // Test that simple uses of the ZBC registration API always work.
    // The only "guaranteed to work" cases are registering black and
    // white colors (or zero or one for depth).  These should be
    // pre-registered on all KMDs and should always succeed.

    const float black[4] = { 0.f, 0.f, 0.f, 0.f };
    const uint32_t blackui[4] = { 0, 0, 0, 0 };
    const int blacki[4] = { 0, 0, 0, 0 };

    bool res = device->RegisterFastClearColor(black, Format::RGBA8) == LWN_TRUE;
    if (!res)
        return false;

    // If the first registration succeeded, then we should be able to register
    // the same thing as many times as we want, as the KMD is supposed to remove
    // duplicate entries.
    for (int iter = 0; iter < 128; iter++) {
        if (device->RegisterFastClearColor(black, Format::RGBA8) != LWN_TRUE) {
            return false;
        }
        if (device->RegisterFastClearDepth(0.0f) != LWN_TRUE) {
            return false;
        }
        if (device->RegisterFastClearDepth(1.0f) != LWN_TRUE) {
            return false;
        }
    }

    // Supported integer fast clear formats
    const Format intFormats[] = { Format::RGBA32UI, Format::RGBA32I, Format::R32UI, Format::R32I };
    for (int i = 0; i < (int)(sizeof(intFormats) / sizeof(intFormats[0])); i++) {
        if (device->RegisterFastClearColorui(blackui, intFormats[i]) != LWN_TRUE) {
            return false;
        }
        if (device->RegisterFastClearColori(blacki, intFormats[i]) != LWN_TRUE) {
            return false;
        }
    }
    // Registering int clear values with a color format like RGBA8888 should
    // not succeed.
    if (device->RegisterFastClearColorui(blackui, Format::RGBA8) == LWN_TRUE) {
        return false;
    }

    return true;
}

static void *getTextureCpuAddress(MemoryPoolAllocator *allocator, Texture *tex)
{
    size_t texOffset = allocator->offset(tex);
    MemoryPool* memPool = allocator->pool(tex);
    return (void*)((uintptr_t)memPool->Map() + texOffset);
}

bool LWNZbcTest::clearTest(DeviceState *deviceState, uint32_t formatIndex, ClearTestType clearType, bool useCompression) const
{
    const FormatInfo *formatInfo = getFormatInfo(formatIndex);
    const int magic = 13;
    bool result = true;

    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Check if we're testing a color or depth/stencil format.
    bool isDepthStencil;
    switch (formatInfo->format) {
    case Format::STENCIL8:
    case Format::DEPTH16:
    case Format::DEPTH24:
    case Format::DEPTH24_STENCIL8:
    case Format::DEPTH32F:
    case Format::DEPTH32F_STENCIL8:
        isDepthStencil = true;
        break;
    default:
        isDepthStencil = false;
        break;
    }

    // Set up a texture with the appropriate format and compression status.
    TextureBuilder tb;
    tb.SetDevice(device).SetDefaults().
        SetSize2D(FBO_WIDTH, FBO_HEIGHT).
        SetTarget(TextureTarget::TARGET_2D).
        SetFormat(formatInfo->format);
    if (FormatIsDepthStencil((LWNformat)(lwn::Format::Enum) formatInfo->format)) {
        useCompression = true;
    }
    if (useCompression) {
        TextureFlags flags = TextureFlags::COMPRESSIBLE;

        // DEPTH16 supports either a ZBC-kind or a Z-plane compression kind
        // but not both.  Ask LWN to use a ZBC-kind for this texture if
        // formatInfo claims DEPTH16 supports ZBC.
        if (formatInfo->format == Format::DEPTH16 && formatInfo->supportsZBC) {
            flags |= TextureFlags::DEPTH16_PREFER_FAST_CLEAR;
        }
        tb.SetFlags(flags);
    }
    size_t texSize = tb.GetStorageSize();

    LWNmemoryPoolFlags flags;
    if (isCpuCoherentSupportedForRTs) {
        flags = LWN_MEMORY_POOL_TYPE_CPU_COHERENT;
    } else {
        flags = LWN_MEMORY_POOL_TYPE_GPU_ONLY;
    }
    MemoryPoolAllocator *texAllocator = new MemoryPoolAllocator(device, NULL, 1024*1024, flags);
    Texture *tex = texAllocator->allocTexture(&tb);
    uint8_t *texCpuAddress = NULL;

    // For platforms that support coherent access to the underlying FB memory,
    // fill the memory with a "magic" bit pattern.
    if (isCpuCoherentSupportedForRTs) {
        texCpuAddress = (uint8_t *) getTextureCpuAddress(texAllocator, tex);
        memset(texCpuAddress, magic, texSize);
    }

    // Set up a buffer object for reading back a few pixels in the texture.
    MemoryPoolAllocator *bufAllocator = new MemoryPoolAllocator(device, NULL, 4096, LWN_MEMORY_POOL_TYPE_CPU_COHERENT);
    BufferBuilder bb;
    bb.SetDevice(device).SetDefaults();
    Buffer *buffer = bufAllocator->allocBuffer(&bb, BUFFER_ALIGN_COPY_WRITE_BIT, 1024);
    BufferAddress bufferAddr = buffer->GetAddress();
    uint8_t *bufferMap = (uint8_t *) buffer->Map();

    // Compute appropriate clear values for this subtest.
    float clearValues[4];
    uint8_t expectedValues[16];
    int readbackBytes;
    setupClearValues(clearType, formatIndex, clearValues, expectedValues, &readbackBytes);

    // For the CLEAR_REGISTERED sub-test, attempt to register our fast clear
    // values with LWN.
    if (clearType == CLEAR_REGISTERED) {
        LWNboolean res;
        if (isDepthStencil) {
            res = device->RegisterFastClearDepth(clearValues[0]);
        } else {
            switch (formatInfo->componentType) {
            case UNORM:
            case SNORM:
            case FLOAT:
                res = device->RegisterFastClearColor(clearValues, formatInfo->format);
                break;
            case UINT:
                res = device->RegisterFastClearColorui((uint32_t *) clearValues, formatInfo->format);
                break;
            case SINT:
                res = device->RegisterFastClearColori((int *) clearValues, formatInfo->format);
                break;
            default:
                assert(0);
                res = LWN_FALSE;
                break;
            }
        }

        // If our format doesn't support ZBC, expect registration to fail
        // (except for DEPTH16, which allows the format-free registration to pass.
        if (!formatInfo->supportsZBC && formatInfo->format != Format::DEPTH16) {
            if (res != LWN_FALSE) {
                result = false;
            }
        }
        
        // If our format does support ZBC, and registrations are supposed to
        // pass, expect the registration to pass.
        if (formatInfo->supportsZBC && zbcRegistrationMustPass()) {
            if (res != LWN_TRUE) {
                result = false;
            }
        }
    }

    // Now, clear the framebuffer.
    queueCB.SetScissor(0, 0, FBO_WIDTH, FBO_HEIGHT);
    queueCB.SetViewport(0, 0, FBO_WIDTH, FBO_HEIGHT);
    if (isDepthStencil) {
        queueCB.SetRenderTargets(0, NULL, NULL, tex, NULL);
        queueCB.ClearDepthStencil(clearValues[0], LWN_TRUE, 0, 0);
    } else {
        queueCB.SetRenderTargets(1, &tex, NULL, NULL, NULL);
        switch (formatInfo->componentType) {
        case UNORM:
        case SNORM:
        case FLOAT:
            queueCB.ClearColor(0, clearValues, LWN_CLEAR_COLOR_MASK_RGBA);
            break;
        case UINT:
            queueCB.ClearColorui(0, (uint32_t *) clearValues, LWN_CLEAR_COLOR_MASK_RGBA);
            break;
        case SINT:
            queueCB.ClearColori(0, (int *) clearValues, LWN_CLEAR_COLOR_MASK_RGBA);
            break;
        default:
            assert(0);
            break;
        }
    }

    // Read back a 4x4 region of the framebuffer.
    CopyRegion region = { 0, 0, 0, 4, 4, 1 };
    queueCB.CopyTextureToBuffer(tex, NULL, &region, bufferAddr, CopyFlags::NONE);
    queueCB.submit();

    // Wait for the readback to complete.
    Sync sync;
    sync.Initialize(device);
    queue->FenceSync(&sync, SyncCondition::ALL_GPU_COMMANDS_COMPLETE, SyncFlagBits::FLUSH_FOR_CPU);
    queue->Flush();
    sync.Wait(LWN_WAIT_TIMEOUT_MAXIMUM);
    sync.Finalize();

    // Check that the values read back match what we're expecting. For Z24
    // formats, skip over the stencil value in the first byte.
    uint8_t *bufferCheck = bufferMap;
    if (formatInfo->format == Format::DEPTH24 || formatInfo->format == Format::DEPTH24_STENCIL8) {
        bufferCheck++;
    }
    if (0 != memcmp(bufferCheck, expectedValues, readbackBytes)) {
        result = false;
    }

    // If we support render targets in coherent memory, check if ZBC was
    // actually used by looking at the underlying memory.
    if (isCpuCoherentSupportedForRTs) {
        bool fbChanged = checkFBChanges(texCpuAddress, magic);

        // The framebuffer memory should change if we're using an unregistered
        // clear color, not compressing, or using a format that doesn't
        // support ZBC.
        bool fbShouldChange = (clearType == CLEAR_UNREGISTERED ||
                               !useCompression ||
                               !formatInfo->supportsZBC);

        // Also, the CLEAR_ONE case (all ones) only matches a ZBC entry for
        // unsigned normalized color formats.
        if (clearType == CLEAR_ONE && formatInfo->componentType != UNORM && !isDepthStencil) {
            fbShouldChange = true;
        }

        if (clearType == CLEAR_REGISTERED && !zbcRegistrationMustPass()) {
            // If we're doing a registered clear without a guarantee that
            // registration succeeds, the framebuffer can always change. Only
            // complain if it doesn't when it should.
            if (fbShouldChange && !fbChanged) {
                result = false;
            }
        } else {
            if (fbChanged != fbShouldChange) {
                result = false;
            }
        }
    }

    queue->Finish();

    delete texAllocator;
    delete bufAllocator;

    return result;
}

void LWNZbcTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    // Note that we're starting a new back of formats, which will detect if
    // this is the first batch being run.
    beginBatch();

    std::vector<bool> testResults;

    for (int fi = 0; fi < getBatchFormatCount(); fi++) {
        for (int ti = (int) CLEAR_ZERO; ti <= (int) CLEAR_REGISTERED; ti++) {
            ClearTestType clearType = ClearTestType(ti);
            for (int ci = 0; ci < 2; ci++) {
                bool compress = (ci != 0);
                testResults.push_back(clearTest(deviceState, fi, clearType, compress));
            }
        }
    }

    if (m_batch == 0) {
        testResults.push_back(apiTest(device));
    }

    g_lwnWindowFramebuffer.bind();
    g_lwnWindowFramebuffer.setViewportScissor();
    queueCB.ClearColor(0, 0.3f, 0.3f, 0.3f, 1.f);
    drawResultCells(queueCB, testResults);
    queueCB.submit();
    queue->Finish();
}

OGTEST_CppTest(LWNZbcTest, lwn_zbc, (0));
OGTEST_CppTest(LWNZbcTest, lwn_zbc1, (1));
OGTEST_CppTest(LWNZbcTest, lwn_zbc2, (2));
OGTEST_CppTest(LWNZbcTest, lwn_zbc3, (3));
OGTEST_CppTest(LWNZbcTest, lwn_zbc4, (4));
