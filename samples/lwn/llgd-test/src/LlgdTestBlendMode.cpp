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
#include <lwndevtools_bootstrap.h>

#define ct_assert(x) static_assert(x, #x)
#define __GL_ARRAYSIZE(x) (sizeof(x) / sizeof(*(x)))

const static BlendAdvancedMode allBlendModes[] = {
    BlendAdvancedMode::BLEND_NONE,
    BlendAdvancedMode::BLEND_ZERO,
    BlendAdvancedMode::BLEND_SRC,
    BlendAdvancedMode::BLEND_DST,
    BlendAdvancedMode::BLEND_SRC_OVER,
    BlendAdvancedMode::BLEND_DST_OVER,
    BlendAdvancedMode::BLEND_SRC_IN,
    BlendAdvancedMode::BLEND_DST_IN,
    BlendAdvancedMode::BLEND_SRC_OUT,
    BlendAdvancedMode::BLEND_DST_OUT,
    BlendAdvancedMode::BLEND_SRC_ATOP,
    BlendAdvancedMode::BLEND_DST_ATOP,
    BlendAdvancedMode::BLEND_XOR,
    BlendAdvancedMode::BLEND_PLUS,
    BlendAdvancedMode::BLEND_PLUS_CLAMPED,
    BlendAdvancedMode::BLEND_PLUS_CLAMPED_ALPHA,
    BlendAdvancedMode::BLEND_PLUS_DARKER,
    BlendAdvancedMode::BLEND_MULTIPLY,
    BlendAdvancedMode::BLEND_SCREEN,
    BlendAdvancedMode::BLEND_OVERLAY,
    BlendAdvancedMode::BLEND_DARKEN,
    BlendAdvancedMode::BLEND_LIGHTEN,
    BlendAdvancedMode::BLEND_COLORDODGE,
    BlendAdvancedMode::BLEND_COLORBURN,
    BlendAdvancedMode::BLEND_HARDLIGHT,
    BlendAdvancedMode::BLEND_SOFTLIGHT,
    BlendAdvancedMode::BLEND_DIFFERENCE,
    BlendAdvancedMode::BLEND_MINUS,
    BlendAdvancedMode::BLEND_MINUS_CLAMPED,
    BlendAdvancedMode::BLEND_EXCLUSION,
    BlendAdvancedMode::BLEND_CONTRAST,
    BlendAdvancedMode::BLEND_ILWERT,
    BlendAdvancedMode::BLEND_ILWERT_RGB,
    BlendAdvancedMode::BLEND_ILWERT_OVG,
    BlendAdvancedMode::BLEND_LINEARDODGE,
    BlendAdvancedMode::BLEND_LINEARBURN,
    BlendAdvancedMode::BLEND_VIVIDLIGHT,
    BlendAdvancedMode::BLEND_LINEARLIGHT,
    BlendAdvancedMode::BLEND_PINLIGHT,
    BlendAdvancedMode::BLEND_HARDMIX,
    BlendAdvancedMode::BLEND_RED,
    BlendAdvancedMode::BLEND_GREEN,
    BlendAdvancedMode::BLEND_BLUE,
    BlendAdvancedMode::BLEND_HSL_HUE,
    BlendAdvancedMode::BLEND_HSL_SATURATION,
    BlendAdvancedMode::BLEND_HSL_COLOR,
    BlendAdvancedMode::BLEND_HSL_LUMINOSITY,
};
ct_assert(__GL_ARRAYSIZE(allBlendModes) ==
            (LWN_BLEND_ADVANCED_MODE_HSL_LUMINOSITY - LWN_BLEND_ADVANCED_MODE_ZERO + 2));

const static BlendAdvancedOverlap allBlendOverlaps[] = {
    BlendAdvancedOverlap::UNCORRELATED,
    BlendAdvancedOverlap::DISJOINT,
    BlendAdvancedOverlap::CONJOINT,
};
ct_assert(__GL_ARRAYSIZE(allBlendOverlaps) ==
            (LWN_BLEND_ADVANCED_OVERLAP_CONJOINT - LWN_BLEND_ADVANCED_OVERLAP_UNCORRELATED + 1));

class BlendValidator {
public:
    void Initialize();
    bool Test();

private:
    uint32_t GetShadowBlendNumber();

    void InjectBlend(const BlendAdvancedMode mode, const BlendAdvancedOverlap overlap,
                     const bool premult, const bool clamped);
    bool RecoverBlend(LWNblendAdvancedMode& mode, LWNblendAdvancedOverlap& overlap,
                      bool& premult, bool& clamped);
    bool TestOneSet(const BlendAdvancedMode mode, const BlendAdvancedOverlap overlap,
                    const bool premult, const bool clamped);

    const LWNdevtoolsBootstrapFunctions* devtools;

    llgd_lwn::QueueHolder qh;
};

void BlendValidator::Initialize()
{
    qh.Initialize(g_device);
    devtools = lwnDevtoolsBootstrap();
}

void BlendValidator::InjectBlend(const BlendAdvancedMode mode, const BlendAdvancedOverlap overlap,
                                 const bool premult, const bool clamped)
{
    const static size_t SIZE = 8192;
    const size_t ONE_PAGE = 65536;
    const size_t ALIGNT = LWN_MEMORY_POOL_STORAGE_ALIGNMENT;
    auto storage = LlgdAlignedAllocPodType<uint8_t>(ONE_PAGE, ALIGNT);

    uint8_t ctrl_space[SIZE] __attribute__((aligned(4096)));

    llgd_lwn::MemoryPoolHolder mph;
    {
        MemoryPoolBuilder pool_builder;

        pool_builder.SetDevice(g_device).SetDefaults()
                    .SetFlags(MemoryPoolFlags::CPU_UNCACHED |
                              MemoryPoolFlags::GPU_CACHED |
                              MemoryPoolFlags::COMPRESSIBLE);

        pool_builder.SetStorage(storage.get(), ONE_PAGE);
        CHECK(mph.Initialize(&pool_builder));
    }

    llgd_lwn::CommandBufferHolder cbh;
    CHECK(cbh.Initialize((Device*)g_device));

    // this resets the cb
    cbh->AddCommandMemory(mph, 0, ONE_PAGE);
    cbh->AddControlMemory(ctrl_space, SIZE);

    cbh->BeginRecording();
    {
        BlendState bs;
        ColorState cs;

        bs.SetDefaults();
        bs.SetAdvancedNormalizedDst(clamped);
        bs.SetAdvancedPremultipliedSrc(premult);
        bs.SetAdvancedOverlap(overlap);
        bs.SetAdvancedMode(mode);

        cs.SetDefaults();
        cs.SetBlendEnable(0, true);

        cbh->BindBlendState(&bs);
        cbh->BindColorState(&cs);
    }

    CommandHandle handle = cbh->EndRecording();

    qh->SubmitCommands(1, &handle);
    qh->Finish();
}

uint32_t BlendValidator::GetShadowBlendNumber()
{
    static const size_t BUFFER_SIZE = 80000;
    uint8_t buffer[BUFFER_SIZE];

    size_t size = 0;

    memset(buffer, 0, BUFFER_SIZE);
    devtools->GetGrCtxSizeForQueue(qh, &size);
    devtools->GetGrCtxForQueue(qh, buffer, size);

    size_t baseShadowBlend = 8870 + 0x2D /*LW9097_LWN_MME_SCRATCH_BLEND_MODE*/;
    return (buffer[baseShadowBlend + 2] << 8) + buffer[baseShadowBlend + 1];
}

bool BlendValidator::RecoverBlend(LWNblendAdvancedMode& mode, LWNblendAdvancedOverlap& overlap,
                                  bool& premult, bool& clamped)
{
    uint32_t shadowBlend = GetShadowBlendNumber();
    return llgdLwnDecodeBlendMode(shadowBlend, mode, overlap, premult, clamped);
}

bool BlendValidator::TestOneSet(const BlendAdvancedMode mode, const BlendAdvancedOverlap overlap,
                                const bool premult, const bool clamped)
{
    LWNblendAdvancedMode    recovered_mode;
    LWNblendAdvancedOverlap recovered_overlap;
    bool                    recovered_premult;
    bool                    recovered_clamped;
    bool                    blend_was_valid;

    InjectBlend(mode, overlap, premult, clamped);
    blend_was_valid = RecoverBlend(recovered_mode, recovered_overlap,
                                   recovered_premult, recovered_clamped);

    // The cast to uint32_t is usefull to compare the two "different" enums.
    // They hold the same values.
#define __BLEND_CHECK(v) (((uint32_t)recovered_##v) == ((uint32_t)v))
    return blend_was_valid        &&
           __BLEND_CHECK(mode)    &&
           __BLEND_CHECK(overlap) &&
           __BLEND_CHECK(premult) &&
           __BLEND_CHECK(clamped);
#undef __BLEND_CHECK
}

bool BlendValidator::Test()
{
    // see : void __glGK100LWNDebugTestIteratedBlend(void)
    uint32_t I = __GL_ARRAYSIZE(allBlendModes);
    uint32_t J = __GL_ARRAYSIZE(allBlendOverlaps);

    for (uint32_t i = 0; i <  I; i++) {
    for (uint32_t j = 0; j <  J; j++) {
    for (uint32_t k = 0; k <= 1; k++) {
    for (uint32_t l = 0; l <= 1; l++) {
        if (!TestOneSet(allBlendModes[i], allBlendOverlaps[j], k!=0, l!=0)) { return false; }
    }}}}

    TEST_FMT(devtools->BlendAdvancedMethodsHash() == __LLGD_BLEND_HASH, "blends dont add up, see also LlgdGpuStateEditor.cpp");

    return true;
}

LLGD_DEFINE_TEST(BlendMode, UNIT,
LwError Execute()
{
    BlendValidator v;
    v.Initialize();

    if (!v.Test()) { return LwError_IlwalidState; }
    else           { return LwSuccess;            }
}
);
