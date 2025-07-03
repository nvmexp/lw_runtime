/*
* Copyright(c) 2016 LWPU Corporation.All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU Corporation is strictly prohibited.
*/

#include "lwntest_cpp.h"
#include "lwn_utils.h"

using namespace lwn;

struct TexTestProperties {
    bool            incOffset;
    TextureFlags    flags;
    TextureTarget   target;
    LWNuint         width;
    LWNuint         height;
    LWNuint         depth;
    LWNuint         levels;
    Format          format;
    LWNuint         samples;
    TextureSwizzle  swizzle[4];
};

// List of texture properties that is used to generate the texture objects. The incOffset indicates if the offset into the pool should
// be incremented. To make sure the comparison is not always failing due to the pool offset this is not done for every texture. Since
// the textures are not used for rendering it is no issue using the same storage in the pool.
static const TexTestProperties s_texTestProperties[] = {
    { true,  0,                     TextureTarget::TARGET_2D,             32, 32, 1, 1, Format::RGBA8,   0, { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A } },
    { false, 0,                     TextureTarget::TARGET_2D,             32, 32, 1, 1, Format::RGBA8,   0, { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A } },
    { false, TextureFlags::DISPLAY, TextureTarget::TARGET_2D,             32, 32, 1, 1, Format::RGBA8,   0, { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A } },
    { false, 0,                     TextureTarget::TARGET_2D_ARRAY,       32, 32, 4, 1, Format::RGBA8,   0, { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A } },
    { false, 0,                     TextureTarget::TARGET_RECTANGLE,      32, 32, 1, 1, Format::RGBA8,   0, { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A } },
    { false, 0,                     TextureTarget::TARGET_2D,             64, 32, 1, 1, Format::RGBA8,   0, { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A } },
    { false, 0,                     TextureTarget::TARGET_2D,             32, 64, 1, 1, Format::RGBA8,   0, { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A } },
    { false, 0,                     TextureTarget::TARGET_2D,             32, 32, 1, 2, Format::RGBA8,   0, { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A } },
    { false, 0,                     TextureTarget::TARGET_2D,             32, 32, 1, 1, Format::RGB10A2, 0, { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A } },
    { false, 0,                     TextureTarget::TARGET_2D_MULTISAMPLE, 32, 32, 1, 1, Format::RGBA8,   4, { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A } },
    { false, 0,                     TextureTarget::TARGET_2D,             32, 32, 1, 1, Format::RGBA8,   0, { TextureSwizzle::R, TextureSwizzle::B, TextureSwizzle::G, TextureSwizzle::A } },
    { true,  0,                     TextureTarget::TARGET_2D_MULTISAMPLE, 32, 32, 1, 1, Format::RGBA8,   8, { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A } },
    { true,  0,                     TextureTarget::TARGET_2D,             32, 32, 1, 4, Format::RGBA8,   0, { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A } }
};

struct TexViewTestProperties {
    bool                        hasLevels;
    bool                        hasLayers;
    bool                        hasFormat;
    bool                        hasSwizzle;
    bool                        hasDepthStencilMode;
    bool                        hasTarget;
    uint8_t                     minLevel;
    uint8_t                     numLevels;
    uint16_t                    minLayer;
    uint16_t                    numLayers;
    Format                      format;
    TextureSwizzle              swizzle[4];
    TextureDepthStencilMode     depthStencilMode;
    TextureTarget               target;
};

TexViewTestProperties s_texViewProperties[] = {
    { true,  true,  true,  true , true,  true,  0, 4, 0, 8, Format::RGBA8,   { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A }, TextureDepthStencilMode::DEPTH,   TextureTarget::TARGET_2D },
    { false, true,  true,  true,  true,  true,  0, 4, 0, 8, Format::RGBA8,   { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A }, TextureDepthStencilMode::DEPTH,   TextureTarget::TARGET_2D },
    { true,  false, true,  true,  true,  true,  0, 4, 0, 8, Format::RGBA8,   { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A }, TextureDepthStencilMode::DEPTH,   TextureTarget::TARGET_2D },
    { true,  true,  false, true,  true,  true,  0, 4, 0, 8, Format::RGBA8,   { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A }, TextureDepthStencilMode::DEPTH,   TextureTarget::TARGET_2D },
    { true,  true,  true,  false, true,  true,  0, 4, 0, 8, Format::RGBA8,   { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A }, TextureDepthStencilMode::DEPTH,   TextureTarget::TARGET_2D },
    { true,  true,  true,  true,  false, true,  0, 4, 0, 8, Format::RGBA8,   { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A }, TextureDepthStencilMode::DEPTH,   TextureTarget::TARGET_2D },
    { true,  true,  true,  true,  true,  false, 0, 4, 0, 8, Format::RGBA8,   { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A }, TextureDepthStencilMode::DEPTH,   TextureTarget::TARGET_2D },
    { true,  true,  true,  true,  true,  true,  1, 4, 0, 8, Format::RGBA8,   { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A }, TextureDepthStencilMode::DEPTH,   TextureTarget::TARGET_2D },
    { true,  true,  true,  true,  true,  true,  0, 8, 0, 8, Format::RGBA8,   { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A }, TextureDepthStencilMode::DEPTH,   TextureTarget::TARGET_2D },
    { true,  true,  true,  true,  true,  true,  0, 4, 2, 8, Format::RGBA8,   { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A }, TextureDepthStencilMode::DEPTH,   TextureTarget::TARGET_2D },
    { true,  true,  true,  true,  true,  true,  0, 4, 0, 4, Format::RGBA8,   { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A }, TextureDepthStencilMode::DEPTH,   TextureTarget::TARGET_2D },
    { true,  true,  true,  true,  true,  true,  0, 4, 0, 8, Format::DEPTH24, { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A }, TextureDepthStencilMode::DEPTH,   TextureTarget::TARGET_2D },
    { true,  true,  true,  true,  true,  true,  0, 4, 0, 8, Format::RGBA8,   { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::A, TextureSwizzle::B }, TextureDepthStencilMode::DEPTH,   TextureTarget::TARGET_2D },
    { true,  true,  true,  true,  true,  true,  0, 4, 0, 8, Format::RGBA8,   { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A }, TextureDepthStencilMode::STENCIL, TextureTarget::TARGET_2D },
    { true,  true,  true,  true,  true,  true,  0, 4, 0, 8, Format::RGBA8,   { TextureSwizzle::R, TextureSwizzle::G, TextureSwizzle::B, TextureSwizzle::A }, TextureDepthStencilMode::DEPTH,   TextureTarget::TARGET_1D }
};

struct SamplerTestProperties {
    MinFilter           minFilter;
    MagFilter           magFilter;
    WrapMode            wrapS;
    WrapMode            wrapT;
    WrapMode            wrapR;
    LWNfloat            minLodClamp;
    LWNfloat            maxLodClamp;
    LWNfloat            lodBias;
    CompareMode         depthCompareMode;
    CompareFunc         depthCompareFunc;
    LWNfloat            borderColor[4];
    LWNfloat            maxAnisotropy;
    SamplerReduction    reduction;
};

static const SamplerTestProperties s_samplerTestProperties[] = {
    { MinFilter::NEAREST, MagFilter::NEAREST, WrapMode::CLAMP,           WrapMode::CLAMP,         WrapMode::CLAMP,        0.0f, 1000.0f,  0.0f, CompareMode::NONE,                 CompareFunc::LESS,  { 0.0f, 0.0f, 0.0f, 0.0f },  1.0f, SamplerReduction::AVERAGE },
    { MinFilter::LINEAR,  MagFilter::NEAREST, WrapMode::CLAMP,           WrapMode::CLAMP,         WrapMode::CLAMP,        0.0f, 1000.0f,  0.0f, CompareMode::NONE,                 CompareFunc::LESS,  { 0.0f, 0.0f, 0.0f, 0.0f },  1.0f, SamplerReduction::AVERAGE },
    { MinFilter::NEAREST, MagFilter::LINEAR,  WrapMode::CLAMP,           WrapMode::CLAMP,         WrapMode::CLAMP,        0.0f, 1000.0f,  0.0f, CompareMode::NONE,                 CompareFunc::LESS,  { 0.0f, 0.0f, 0.0f, 0.0f },  1.0f, SamplerReduction::AVERAGE },
    { MinFilter::NEAREST, MagFilter::NEAREST, WrapMode::CLAMP_TO_BORDER, WrapMode::CLAMP,         WrapMode::CLAMP,        0.0f, 1000.0f,  0.0f, CompareMode::NONE,                 CompareFunc::LESS,  { 0.0f, 0.0f, 0.0f, 0.0f },  1.0f, SamplerReduction::AVERAGE },
    { MinFilter::NEAREST, MagFilter::NEAREST, WrapMode::CLAMP,           WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP,        0.0f, 1000.0f,  0.0f, CompareMode::NONE,                 CompareFunc::LESS,  { 0.0f, 0.0f, 0.0f, 0.0f },  1.0f, SamplerReduction::AVERAGE },
    { MinFilter::NEAREST, MagFilter::NEAREST, WrapMode::CLAMP,           WrapMode::CLAMP,         WrapMode::MIRROR_CLAMP, 0.0f, 1000.0f,  0.0f, CompareMode::NONE,                 CompareFunc::LESS,  { 0.0f, 0.0f, 0.0f, 0.0f },  1.0f, SamplerReduction::AVERAGE },
    { MinFilter::NEAREST, MagFilter::NEAREST, WrapMode::CLAMP,           WrapMode::CLAMP,         WrapMode::CLAMP,       10.0f, 1000.0f,  0.0f, CompareMode::NONE,                 CompareFunc::LESS,  { 0.0f, 0.0f, 0.0f, 0.0f },  1.0f, SamplerReduction::AVERAGE },
    { MinFilter::NEAREST, MagFilter::NEAREST, WrapMode::CLAMP,           WrapMode::CLAMP,         WrapMode::CLAMP,        0.0f,  500.0f,  0.0f, CompareMode::NONE,                 CompareFunc::LESS,  { 0.0f, 0.0f, 0.0f, 0.0f },  1.0f, SamplerReduction::AVERAGE },
    { MinFilter::NEAREST, MagFilter::NEAREST, WrapMode::CLAMP,           WrapMode::CLAMP,         WrapMode::CLAMP,        0.0f, 1000.0f, -0.5f, CompareMode::NONE,                 CompareFunc::LESS,  { 0.0f, 0.0f, 0.0f, 0.0f },  1.0f, SamplerReduction::AVERAGE },
    { MinFilter::NEAREST, MagFilter::NEAREST, WrapMode::CLAMP,           WrapMode::CLAMP,         WrapMode::CLAMP,        0.0f, 1000.0f,  0.0f, CompareMode::COMPARE_R_TO_TEXTURE, CompareFunc::LESS,  { 0.0f, 0.0f, 0.0f, 0.0f },  1.0f, SamplerReduction::AVERAGE },
    { MinFilter::NEAREST, MagFilter::NEAREST, WrapMode::CLAMP,           WrapMode::CLAMP,         WrapMode::CLAMP,        0.0f, 1000.0f,  0.0f, CompareMode::NONE,                 CompareFunc::EQUAL, { 0.0f, 0.0f, 0.0f, 0.0f },  1.0f, SamplerReduction::AVERAGE },
    { MinFilter::NEAREST, MagFilter::NEAREST, WrapMode::CLAMP,           WrapMode::CLAMP,         WrapMode::CLAMP,        0.0f, 1000.0f,  0.0f, CompareMode::NONE,                 CompareFunc::LESS,  { 0.0f, 1.0f, 0.0f, 0.0f },  1.0f, SamplerReduction::AVERAGE },
    { MinFilter::NEAREST, MagFilter::NEAREST, WrapMode::CLAMP,           WrapMode::CLAMP,         WrapMode::CLAMP,        0.0f, 1000.0f,  0.0f, CompareMode::NONE,                 CompareFunc::LESS,  { 0.0f, 0.0f, 0.0f, 0.0f }, 10.0f, SamplerReduction::AVERAGE },
    { MinFilter::NEAREST, MagFilter::NEAREST, WrapMode::CLAMP,           WrapMode::CLAMP,         WrapMode::CLAMP,        0.0f, 1000.0f,  0.0f, CompareMode::NONE,                 CompareFunc::LESS,  { 0.0f, 0.0f, 0.0f, 0.0f },  1.0f, SamplerReduction::MAX }
};

class LWNCompareObjects
{
public:
    LWNTEST_CppMethods();

    size_t buildTextureLists(Device *device, MemoryPool *pool, Texture** &texList1, Texture** &texList2) const;
    size_t buildTexViewLists(TextureView** &tvList1, TextureView** &tvList2) const;
    size_t buildSamplerLists(Device *device, Sampler** &samplerList1, Sampler** &samplerList2) const;

    template<class T>
    bool runTest(T **list1, T **list2, size_t numElements) const;

    template<class T>
    void deleteLists(T **l1, T **l2, size_t numElements) const;
};

lwString LWNCompareObjects::getDescription() const
{
    lwStringBuf sb;
    sb <<
        "Test to verify the lwnTextureCompare, lwnTextureViewCompare and lwnSamplerCompare functions. "
        "The test generates two identical lists of different Textures, TextureViews and Samplers. Then "
        "it compares the elements of list one against the elements of list two. "
        "The tests succeeds if the correct element is found and all other comparisons failed. "
        "In this case a green quad is drawn, if the test failed a red quad is drawn.";

    return sb.str();
}

int LWNCompareObjects::isSupported() const
{
    return lwogCheckLWNAPIVersion(51, 3);
}

size_t LWNCompareObjects::buildTextureLists(Device *device, MemoryPool *pool, Texture** &texList1, Texture** &texList2) const
{
    size_t numTextures = __GL_ARRAYSIZE(s_texTestProperties);

    Texture **tl[2];

    tl[0] = new Texture *[numTextures];
    tl[1] = new Texture *[numTextures];

    TextureBuilder tb;
    tb.SetDevice(device);

    for (size_t n = 0; n < 2; ++n) {
        LWNuintptr offset = 0;

        for (size_t i = 0; i < numTextures; ++i) {
            tb.SetDefaults()
              .SetFlags(s_texTestProperties[i].flags)
              .SetTarget(s_texTestProperties[i].target)
              .SetWidth(s_texTestProperties[i].width)
              .SetHeight(s_texTestProperties[i].height)
              .SetDepth(s_texTestProperties[i].depth)
              .SetLevels(s_texTestProperties[i].levels)
              .SetFormat(s_texTestProperties[i].format)
              .SetSamples(s_texTestProperties[i].samples)
              .SetSwizzle(s_texTestProperties[i].swizzle[0], s_texTestProperties[i].swizzle[1],
                          s_texTestProperties[i].swizzle[2], s_texTestProperties[i].swizzle[3]);

            offset = (offset + (tb.GetStorageAlignment() - 1)) & ~(tb.GetStorageAlignment() - 1);

            tl[n][i] = tb.CreateTextureFromPool(pool, offset);

            if (s_texTestProperties[i].incOffset) {
                offset += tb.GetStorageSize();
            }
        }
    }

    texList1 = tl[0];
    texList2 = tl[1];

    return numTextures;
}

size_t LWNCompareObjects::buildTexViewLists(TextureView** &tvList1, TextureView** &tvList2) const
{
    size_t numTexViews = __GL_ARRAYSIZE(s_texViewProperties);

    TextureView **tv[2];

    tv[0] = new TextureView *[numTexViews];
    tv[1] = new TextureView *[numTexViews];

    for (size_t n = 0; n < 2; ++n) {
        for (size_t i = 0; i < numTexViews; ++i) {
            tv[n][i] = TextureView::Create();
            tv[n][i]->SetDefaults();

            if (s_texViewProperties[i].hasLevels) {
                tv[n][i]->SetLevels(s_texViewProperties[i].minLevel, s_texViewProperties[i].numLevels);
            }
            if (s_texViewProperties[i].hasLayers) {
                tv[n][i]->SetLayers(s_texViewProperties[i].minLayer, s_texViewProperties[i].numLayers);
            }
            if (s_texViewProperties[i].hasFormat) {
                tv[n][i]->SetFormat(s_texViewProperties[i].format);
            }
            if (s_texViewProperties[i].hasSwizzle) {
                tv[n][i]->SetSwizzle(s_texViewProperties[i].swizzle[0], s_texViewProperties[i].swizzle[1],
                                    s_texViewProperties[i].swizzle[2], s_texViewProperties[i].swizzle[3]);
            }
            if (s_texViewProperties[i].hasDepthStencilMode) {
                tv[n][i]->SetDepthStencilMode(s_texViewProperties[i].depthStencilMode);
            }
            if (s_texViewProperties[i].hasTarget) {
                tv[n][i]->SetTarget(s_texViewProperties[i].target);
            }
        }
    }

    tvList1 = tv[0];
    tvList2 = tv[1];

    return numTexViews;
}

size_t LWNCompareObjects::buildSamplerLists(Device *device, Sampler** &samplerList1, Sampler** &samplerList2) const
{
    size_t numSampler = __GL_ARRAYSIZE(s_samplerTestProperties);
    size_t numSamplerToTest = 0;

    Sampler **sl[2];

    sl[0] = new Sampler *[numSampler];
    sl[1] = new Sampler *[numSampler];

    SamplerBuilder sb;

    for (size_t i = 0; i < numSampler; ++i) {

        // Skip tests that use a MIN/MAX reduction mode if not supported by the device.
        if (!g_lwnDeviceCaps.supportsMinMaxReduction &&
            s_samplerTestProperties[i].reduction != SamplerReduction::AVERAGE) {
            continue;
        }

        sb.SetDefaults()
            .SetDevice(device)
            .SetMinMagFilter(s_samplerTestProperties[i].minFilter, s_samplerTestProperties[i].magFilter)
            .SetWrapMode(s_samplerTestProperties[i].wrapS, s_samplerTestProperties[i].wrapT, s_samplerTestProperties[i].wrapR)
            .SetLodClamp(s_samplerTestProperties[i].minLodClamp, s_samplerTestProperties[i].maxLodClamp)
            .SetLodBias(s_samplerTestProperties[i].lodBias)
            .SetCompare(s_samplerTestProperties[i].depthCompareMode, s_samplerTestProperties[i].depthCompareFunc)
            .SetBorderColor(s_samplerTestProperties[i].borderColor)
            .SetMaxAnisotropy(s_samplerTestProperties[i].maxAnisotropy);
        if (g_lwnDeviceCaps.supportsMinMaxReduction) {
            sb.SetReductionFilter(s_samplerTestProperties[i].reduction);
        }

        // Create two identical samplers for each set of state.
        sl[0][numSamplerToTest] = sb.CreateSampler();
        sl[1][numSamplerToTest] = sb.CreateSampler();
        numSamplerToTest++;
    }

    samplerList1 = sl[0];
    samplerList2 = sl[1];

    return numSamplerToTest;
}

template<class T>
void LWNCompareObjects::deleteLists(T **l1, T **l2, size_t numElements) const
{
    for (size_t i = 0; i < numElements; ++i) {
        l1[i]->Free();
        l2[i]->Free();
    }

    delete[] l1;
    delete[] l2;
}

template<class T>
bool LWNCompareObjects::runTest(T **list1, T **list2, size_t numElements) const
{
    bool success = true;

    for (size_t i = 0; (i < numElements) && success; ++i) {
        for (size_t j = 0; (j < numElements) && success; ++j) {
            if (list1[i]->Compare(list2[j])) {
                success = success && (i == j);
            } else {
                success = success && (i != j);
            }
        }
    }

    return success;
}

void LWNCompareObjects::doGraphics() const
{
    DeviceState *deviceState = DeviceState::GetActive();
    Device *device = deviceState->getDevice();
    QueueCommandBuffer &queueCB = deviceState->getQueueCB();
    Queue *queue = deviceState->getQueue();

    const size_t gpuPoolSize = 64 * 1024;

    MemoryPool *gpuPool = device->CreateMemoryPool(NULL, gpuPoolSize, MemoryPoolType::GPU_ONLY);

    // Create two lists that contain identical texture objects. The test will search for each
    // object of list 1 the equivalent in list 2 and is supposed to find exactly one at
    // the same index as in list 1.
    Texture **texList[2] = { NULL };
    size_t numTextureTests = buildTextureLists(device, gpuPool, texList[0], texList[1]);

    // Create two lists of identical TextureView to run the same test as for the texture objects.
    TextureView **tvList[2] = { NULL };
    size_t numTexViewTests = buildTexViewLists(tvList[0], tvList[1]);

    // Create two lists of identical samplers to run the same test as for the texture objects.
    Sampler **samplerList[2] = { NULL };
    size_t numSamplerTests = buildSamplerLists(device, samplerList[0], samplerList[1]);

    bool success = runTest<Texture>(texList[0], texList[1], numTextureTests);

    if (success) {
        success = runTest<TextureView>(tvList[0], tvList[1], numTexViewTests);
    }

    if (success) {
        success = runTest<Sampler>(samplerList[0], samplerList[1], numSamplerTests);
    }

    if (success) {
        queueCB.ClearColor(0, 0.0f, 1.0f, 0.0f, 1.0f);
    } else {
        queueCB.ClearColor(0, 1.0f, 0.0f, 0.0f, 1.0f);
    }

    queueCB.submit();

    queue->Finish();

    deleteLists<Texture>(texList[0], texList[1], numTextureTests);
    deleteLists<TextureView>(tvList[0], tvList[1], numTexViewTests);
    deleteLists<Sampler>(samplerList[0], samplerList[1], numSamplerTests);

    gpuPool->Free();
}

OGTEST_CppTest(LWNCompareObjects, lwn_compare_objects, );