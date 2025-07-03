/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "lwntest_cpp.h"
#include "lwn_utils.h"

#define DEBUG_MODE 0

#if (DEBUG_MODE)
#define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINT(...)
#endif

namespace {

using namespace lwn;

class LWLWiewOffsetTest {
public:
    LWNTEST_CppMethods();
};

lwString LWLWiewOffsetTest::getDescription() const
{
    return "Verify that Texture::GetViewOffset() and TextureBuilder::GetViewOffset() return the "
           "same values, since they are implemented in different ways.";
}

int LWLWiewOffsetTest::isSupported() const
{
    return lwogCheckLWNAPIVersion(200, 1);
}

struct TestCase {
    int width;
    int height;
    int depth;
    int levels;
    TextureTarget target;
    Format format;
    int testLayer;
    int testLevel;
};
#define T(width, height, depth, levels, target, format, testLayer, testLevel) \
    { width, height, depth, levels, TextureTarget::TARGET_##target, Format::format, testLayer, testLevel }

const TestCase CASES[] = {
    //W     H       D   Levels  Target                  Format              Test Layer  Test Level
    T(32,   1,      1,  1,      1D,                     R8,                 0,          0),
    T(73,   1,      1,  2,      1D,                     R8,                 0,          1),
    T(57,   1,      1,  1,      1D,                     RGBA8,              0,          0),
    T(64,   3,      1,  1,      1D_ARRAY,               R8,                 0,          0),
    T(73,   5,      1,  4,      1D_ARRAY,               RGBA16,             3,          2),
    T(73,   5,      1,  4,      1D_ARRAY,               RGBA32F,            3,          2),
    T(73,   5,      1,  4,      1D_ARRAY,               RGBA16,             3,          2),
    T(16,   16,     1,  1,      2D,                     RG8,                0,          0),
    T(37,   42,     1,  3,      2D,                     RGB_DXT1,           0,          2),
    T(117,  134,    1,  4,      2D,                     RGBX8,              0,          3),
    T(88,   213,    5,  3,      2D_ARRAY,               RGB_DXT1,           3,          2),
    T(88,   213,    5,  3,      2D_ARRAY,               RGBA_ASTC_6x5_SRGB, 3,          2),
    T(128,  96,     4,  1,      2D_ARRAY,               DEPTH24,            3,          0),
    T(314,  314,    36, 5,      2D_ARRAY,               RGBA8,              31,         4),
    T(211,  19,     6,  4,      2D_ARRAY,               RGB32F,             4,          3),
    T(44,   53,     1,  1,      RECTANGLE,              RGBA8,              0,          0),
    T(58,   58,     1,  4,      LWBEMAP,                RGBA8,              5,          2),
    T(314,  314,    6,  5,      LWBEMAP_ARRAY,          RGBA8,              31,         4),
    T(72,   35,     13, 3,      3D,                     RG16,               0,          2),
    T(277,  403,    7,  1,      2D_MULTISAMPLE_ARRAY,   RG16,               5,          0),
};

const int NUM_CASES = int(sizeof(CASES) / sizeof(CASES[0]));

void LWLWiewOffsetTest::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    int columns = 5;
    int rows = (NUM_CASES + columns - 1) / columns;
    cellTestInit(columns, rows);
    MemoryPoolAllocator allocator(device, nullptr, 0, LWN_MEMORY_POOL_TYPE_GPU_ONLY);
    queueCB.ClearColor(0, 0.0f, 0.0f, 0.0f, 1.0f);

    for (int i = 0; i < NUM_CASES; ++i) {
        SetCellViewportScissorPadded(queueCB, i % columns, i / columns, 1);

        const TestCase& testCase = CASES[i];
        TextureBuilder builder;
        builder.SetDefaults()
               .SetDevice(device)
               .SetSize3D(testCase.width, testCase.height, testCase.depth)
               .SetTarget(testCase.target)
               .SetFormat(testCase.format)
               .SetLevels(testCase.levels);
        if (testCase.target == TextureTarget::TARGET_2D_MULTISAMPLE_ARRAY) {
            builder.SetSamples(8);
        }
        if (FormatIsDepthStencil(testCase.format)) {
            builder.SetFlags(TextureFlags::COMPRESSIBLE);
        }
        TextureView view;
        view.SetDefaults()
            .SetLayers(testCase.testLayer, 1)
            .SetLevels(testCase.testLevel, 1);
        if ((testCase.target == TextureTarget::TARGET_LWBEMAP ||
             testCase.target == TextureTarget::TARGET_LWBEMAP_ARRAY)) {
            // Typical use case would be to select individual faces
            view.SetTarget(TextureTarget::TARGET_2D_ARRAY);
        }
        ptrdiff_t builderOffset = builder.GetViewOffset(&view);
        Texture* texture = allocator.allocTexture(&builder);
        ptrdiff_t textureOffset = texture->GetViewOffset(&view);
        if (textureOffset == builderOffset) {
            queueCB.ClearColor(0, 0.0f, 1.0f, 0.0f, 1.0f);
        } else {
            DEBUG_PRINT("Mismatch for test case %d -- Builder offset: %td; Texture offset: %td\n",
                        i, builderOffset, textureOffset);
            queueCB.ClearColor(0, 1.0f, 0.0f, 0.0f, 1.0f);
        }
        allocator.freeTexture(texture);
    }
    queueCB.submit();
    queue->Finish();
}

} // namespace

OGTEST_CppTest(LWLWiewOffsetTest, lwn_view_offset, );

