/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
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
#include <LlgdTestUtilEditState.h>

#include <liblwn-llgd.h>
#include <lwndevtools_bootstrap.h>
#include "lwnTool/lwnTool_GlslcInterface.h"

#include <array>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include <nn/os.h>

namespace {

#include "EditProgramReplacerShaders/null_vs.h"
#include "EditProgramReplacerShaders/null_vs.cpp"

#include "EditProgramReplacerShaders/grey_fs.h"
#include "EditProgramReplacerShaders/grey_fs.cpp"

#include "EditProgramReplacerShaders/red_fs.h"
#include "EditProgramReplacerShaders/red_fs.cpp"

#include "EditProgramReplacerShaders/zero_cs.h"
#include "EditProgramReplacerShaders/zero_cs.cpp"

#include "EditProgramReplacerShaders/one_cs.h"
#include "EditProgramReplacerShaders/one_cs.cpp"

LlgdProgram ToLlgdProgram(LWNprogram* program)
{
    static const auto& devtools = *lwnDevtoolsBootstrap();

    static_assert(sizeof(LlgdProgram) == sizeof(LWNdevtoolsProgram), "binary compatibility broken");
    static_assert(alignof(LlgdProgram) == alignof(LWNdevtoolsProgram), "binary compatibility broken");
    LlgdProgram result{ 0 };
    devtools.ProgramGetDevtoolsProgram(program, reinterpret_cast<LWNdevtoolsProgram*>(&result));
    return result;
}

class Random {
    std::random_device rd;
    std::mt19937 mt;
    std::uniform_int_distribution<uint64_t> dist;

public:
    Random(uint64_t max) : mt(rd()) , dist(1, max) { }
    uint64_t get() { return dist(mt); }
};

class Validator {
public:
    bool Initialize();
    bool Test();

    static const int PAGE = 65536;
    static const int UCODE = PAGE;
    static const int COLOR = PAGE;
    static const int DEPTH = PAGE;
    static const int RT = COLOR + DEPTH;
    static const int VB = PAGE;
    static const int SCRATCH = 2 * PAGE;

    static const int SIZE = RT + VB + SCRATCH;
    static const int UCODES = 3 * UCODE;

private:
    bool PrepareForColorTesting(llgd_lwn::ProgramHolder& greyProgram, llgd_lwn::ProgramHolder& redProgram, llgd_lwn::TextureHolder& colorTex, llgd_lwn::TextureHolder& depthTex, int padRed, int padGrey);

    bool TestColors(int, int);
    bool TestStopEditAtFirstDraw();
    bool TestCompute();

    uint8_t backing[SIZE] __attribute__((aligned(PAGE)));
    uint8_t ucodes[UCODES] __attribute__((aligned(PAGE)));

    llgd_lwn::QueueHolder qh;
    llgd_lwn::MemoryPoolHolder mph;
    llgd_lwn::MemoryPoolHolder mphUcodes;

    std::unique_ptr<llgd_lwn::CommandHandleEditingHelper> editingHelper;
};

bool Validator::Initialize()
{
    qh.Initialize(g_device);

    editingHelper = std::make_unique<llgd_lwn::CommandHandleEditingHelper>(qh);
    TEST(editingHelper->Initialize());

    {
        using MPF = MemoryPoolFlags;
        lwn::MemoryPoolBuilder builder;
        builder.SetDefaults();
        builder.SetDevice(g_device);

        builder.SetFlags(MPF::CPU_UNCACHED | MPF::GPU_CACHED | MPF::SHADER_CODE);
        builder.SetStorage(ucodes, UCODES);
        TEST(mphUcodes.Initialize(&builder));

        builder.SetFlags(MPF::CPU_UNCACHED | MPF::GPU_CACHED | MPF::COMPRESSIBLE);
        builder.SetStorage(backing, SIZE);
        TEST(mph.Initialize(&builder));
        memset(backing, 0x0, SIZE);
    }

    return true;
}

struct Store {
    std::vector<uint8_t> ctrl{ 100ull, uint8_t{ 0u } };
    int ctrlIndex = 0;
    std::vector<uint8_t> cmd{ 100ull, uint8_t{ 0u } };
    int cmdIndex = 0;
};

void* WriteControl(const void* data, size_t size, void* user)
{
    auto& store = *reinterpret_cast<Store*>(user);
    auto pos = &store.ctrl[store.ctrlIndex];
    memcpy(pos, data, size);
    store.ctrlIndex += size;
    return pos;
}

void* WriteCommand(const void* data, size_t size, void* user)
{
    auto& store = *reinterpret_cast<Store*>(user);
    auto pos = &store.cmd[store.cmdIndex];
    memcpy(pos, data, size);
    store.cmdIndex += size;
    return pos;
}

LlgdProgram Mock(const char* methods)
{
    LlgdProgram result{ 0 };
    result.state = LlgdProgramState::Graphics;
    LlgdProgramGraphics& graphics = result.graphics;
    graphics.methodCount = strlen(methods) / 4;
    memcpy(graphics.methods, methods, strlen(methods));
    return result;
}

bool TestStrCmp(uint64_t increment, const char* original, const char* replacement, const char* cb, const char* expect, size_t len, const char* alt, size_t altLen)
{
    auto replacer = aligned_alloc(llgxCommandSetProgramReplacerAlignof(), llgxCommandSetProgramReplacerSizeof());

    LlgdProgram mockOriginal = Mock(original);
    LlgdProgram mockReplacement = Mock(replacement);

    GpuState unused;

    Store store{};
    llgxCommandSetProgramReplacerEmplace(replacer,
        [](uint32_t, void*){ CHECK(!"never"); return 0u; }, 0, 0,
        unused,
        unused,
        &mockOriginal, &mockReplacement,
        &WriteControl, &WriteCommand, &store);
    llgxCommandSetProgramReplacerActivate(replacer, true);

    Random generator{ increment };
    for (uint32_t i = 0; i < strlen(cb);) {
        const auto step = 4 * generator.get();
        llgxCommandSetProgramReplacerWriteCommandMemory(replacer, const_cast<char*>(cb) + i, step);
        i += step;
    }

    llgxCommandSetProgramReplacerDestruct(replacer);
    free(replacer);

    return !memcmp((char*)&store.cmd[0], expect, len)
        || ((altLen > 0) && (!memcmp((char*)&store.cmd[0], alt, altLen)));
}

bool TestStrCmp(const char* original, const char* replacement, const char* cb, const char* expect, size_t len, const char* alt = nullptr, size_t altLen = 0)
{
    for (uint32_t step = 1; step <= strlen(cb) / 4; ++step) {
        if(!::TestStrCmp(step, original, replacement, cb, expect, len, alt, altLen)) return false;
    }
    return true;
}

ShaderData LoadShader(uint64_t address, uint8_t* map, const uint8_t* shader)
{
    const GLSLCoutput* glslcOutput = (const GLSLCoutput*)shader;
    ShaderData shaderData{ 0 };
    for (uint32_t section = 0; section < glslcOutput->numSections; ++section) {
        const auto type = glslcOutput->headers[section].genericHeader.common.type;
        if (type != GLSLC_SECTION_TYPE_GPU_CODE)
            continue;

        GLSLCgpuCodeHeader gpuCodeHeader = (GLSLCgpuCodeHeader)(glslcOutput->headers[section].gpuCodeHeader);
        const char* data = (const char *)glslcOutput + gpuCodeHeader.common.dataOffset;
        const char* ucode = data + gpuCodeHeader.dataOffset;
        const char* control = data + gpuCodeHeader.controlOffset;
        const int32_t ucodeSize = gpuCodeHeader.dataSize;
        CHECK(ucodeSize < Validator::UCODE);

        memcpy(map, ucode, ucodeSize);
        nn::os::FlushDataCache(map, ucodeSize);

        shaderData.data = address;
        shaderData.control = control;

        break;
    }
    return shaderData;
}

bool Validator::PrepareForColorTesting(llgd_lwn::ProgramHolder& greyProgram, llgd_lwn::ProgramHolder& redProgram, llgd_lwn::TextureHolder& colorTex, llgd_lwn::TextureHolder& depthTex, int padRed, int padGrey)
{
    static const auto& devtools = *lwnDevtoolsBootstrap();

    greyProgram.Initialize((Device*)g_device);
    redProgram.Initialize((Device*)g_device);
    {
        const auto base = mphUcodes->GetBufferAddress();
        const auto nullVs = LoadShader(base, ucodes, nullVsData);
        const auto greyFs = LoadShader(base + UCODE, ucodes + UCODE, greyFsData);
        const auto redFs = LoadShader(base + 2 * UCODE, ucodes + 2 * UCODE, redFsData);

        std::array<ShaderData, 2> shaders;
        shaders[0] = nullVs;

        shaders[1] = greyFs;
        TEST(greyProgram->SetShaders(shaders.size(), &shaders[0]));
        devtools.ProgramNopPadCompiledState(greyProgram, padGrey);

        shaders[1] = redFs;
        TEST(redProgram->SetShaders(shaders.size(), &shaders[0]));
        devtools.ProgramNopPadCompiledState(redProgram, padRed);
    }

    {
        TextureBuilder builder;
        builder.SetDefaults();
        builder.SetDevice(g_device);
        builder.SetSize2D(2, 2);
        builder.SetTarget(TextureTarget::TARGET_2D);
        builder.SetFlags(TextureFlags::COMPRESSIBLE);

        builder.SetFormat(Format::RGBA8);
        builder.SetStorage(mph, 0);
        CHECK(builder.GetStorageSize() <= COLOR);
        CHECK(builder.GetStorageAlignment() <= PAGE);
        colorTex.Initialize(&builder);

        builder.SetFormat(Format::DEPTH24);
        builder.SetStorage(mph, COLOR);
        CHECK(builder.GetStorageSize() <= DEPTH);
        CHECK(builder.GetStorageAlignment() <= PAGE);
        depthTex.Initialize(&builder);
    }

    return true;
}

bool Validator::TestColors(int padRed, int padGrey)
{
    llgd_lwn::ProgramHolder greyProgram;
    llgd_lwn::ProgramHolder redProgram;
    llgd_lwn::TextureHolder colorTex;
    llgd_lwn::TextureHolder depthTex;
    const auto prepared = PrepareForColorTesting(greyProgram, redProgram, colorTex, depthTex, padRed, padGrey);
    TEST(prepared);

    Texture* rtColor = colorTex;
    const static float WHITE[4]{ 1, 1, 1, 1};
    const auto handle = editingHelper->MakeHandle([&] (CommandBuffer* cb) {
        cb->BindProgram(greyProgram, LWN_SHADER_STAGE_ALL_GRAPHICS_BITS);
        cb->BindVertexAttribState(0, nullptr);
        cb->BindVertexStreamState(0, nullptr);
        cb->BindVertexBuffer(0, mph->GetBufferAddress() + RT, VB);
        cb->SetRenderTargets(1, &rtColor, nullptr, depthTex, nullptr);
        cb->SetScissor(0, 0, 2, 2);
        cb->SetViewport(0, 0, 2, 2);
        cb->ClearColor(0, WHITE, ClearColorMask::RGBA);
        cb->SetShaderScratchMemory(mph, RT + VB, SCRATCH);
        cb->DrawElements(DrawPrimitive::POINTS, IndexType::UNSIGNED_SHORT, 1, mph->GetBufferAddress());
    });

    const static uint32_t grey = 0xff7f7f7ful;
    const static uint32_t red = 0xff0000fful;

    const auto state = editingHelper->RunAndExtractGpuState(handle);
    TEST_EQ(*((uint32_t*)backing), grey);

    editingHelper->ResetPointersForEditingCB();

    LlgdProgram redLlgdProgram = ToLlgdProgram(redProgram);
    LlgdProgram greyLlgdProgram = ToLlgdProgram(greyProgram);

    const auto patched = llgdCommandSetReplaceProgram(
        handle,
        [](uint32_t index, void*) { return 10 + index; }, 0, 1000, /* test replace, not edge insert */
        state, state,
        &greyLlgdProgram, &redLlgdProgram,
        editingHelper->WriteControlMemoryForEditing,
        editingHelper->WriteCommandMemoryForEditing,
        editingHelper.get());
    const auto decoded = editingHelper->MakeCommandHandleRunnable(patched);
    editingHelper->Run(decoded);
    TEST_EQ(*((uint32_t*)backing), red);

    return true;
}

bool Validator::TestStopEditAtFirstDraw()
{

    llgd_lwn::ProgramHolder greyProgram;
    llgd_lwn::ProgramHolder redProgram;
    llgd_lwn::ProgramHolder greyProgram2;
    llgd_lwn::TextureHolder colorTex;
    llgd_lwn::TextureHolder depthTex;

    const auto prepared = PrepareForColorTesting(greyProgram, redProgram, colorTex, depthTex, 0, 0);
    TEST(prepared);

    TEST(greyProgram2.Initialize((Device*)g_device));
    {
        const auto base = mphUcodes->GetBufferAddress();
        const auto nullVs = LoadShader(base, ucodes, nullVsData);
        const auto greyFs = LoadShader(base + UCODE, ucodes + UCODE, greyFsData);

        std::array<ShaderData, 2> shaders{ nullVs, greyFs };

        TEST(greyProgram2->SetShaders(shaders.size(), &shaders[0]));
    }

    Texture* rtColor = colorTex;

    const static float WHITE[4]{ 1, 1, 1, 1};
    const auto handle = editingHelper->MakeHandle([&](CommandBuffer* cb) {
/*11*/  cb->BindProgram(greyProgram, LWN_SHADER_STAGE_ALL_GRAPHICS_BITS);
/*11*/  cb->BindVertexAttribState(0, nullptr);
/*11*/  cb->BindVertexStreamState(0, nullptr);
/*11*/  cb->BindVertexBuffer(0, mph->GetBufferAddress() + RT, VB);
/*11*/  cb->SetRenderTargets(1, &rtColor, nullptr, depthTex, nullptr);
/*11*/  cb->SetScissor(0, 0, 2, 2);
/*11*/  cb->SetViewport(0, 0, 2, 2);
/*11*/  cb->ClearColor(0, WHITE, ClearColorMask::RGBA);
/*12*/  cb->SetShaderScratchMemory(mph, RT + VB, SCRATCH);
/*12*/  cb->DrawElements(DrawPrimitive::POINTS, IndexType::UNSIGNED_SHORT, 1, mph->GetBufferAddress());
/*13*/
/*13*/  cb->BindProgram(greyProgram2, LWN_SHADER_STAGE_ALL_GRAPHICS_BITS);
/*13*/  cb->BindVertexAttribState(0, nullptr);
/*13*/  cb->BindVertexStreamState(0, nullptr);
/*13*/  cb->BindVertexBuffer(0, mph->GetBufferAddress() + RT, VB);
/*13*/  cb->SetRenderTargets(1, &rtColor, nullptr, depthTex, nullptr);
/*13*/  cb->SetScissor(0, 0, 2, 2);
/*13*/  cb->SetViewport(0, 0, 2, 2);
/*13*/  cb->ClearColor(0, WHITE, ClearColorMask::RGBA);
/*14*/  cb->SetShaderScratchMemory(mph, RT + VB, SCRATCH);
/*14*/  cb->DrawElements(DrawPrimitive::POINTS, IndexType::UNSIGNED_SHORT, 1, mph->GetBufferAddress());
    });
    const auto state = editingHelper->RunAndExtractGpuState(handle);

    const static uint32_t grey = 0xff7f7f7ful;
    TEST_EQ(*((uint32_t*)backing), grey);

    editingHelper->ResetPointersForEditingCB();
    memset(backing, 0x0, SIZE);

    LlgdProgram redLlgdProgram = ToLlgdProgram(redProgram);
    LlgdProgram greyLlgdProgram = ToLlgdProgram(greyProgram);

    // Stop editing just after 1st draw finished.
    auto patched = llgdCommandSetReplaceProgram(
        handle,
        [](uint32_t index, void*) { return 10 + index; }, 0, 12,
        state, state,
        &greyLlgdProgram, &redLlgdProgram,
        editingHelper->WriteControlMemoryForEditing,
        editingHelper->WriteCommandMemoryForEditing,
        editingHelper.get());
    auto decoded = editingHelper->MakeCommandHandleRunnable(patched);
    editingHelper->Run(decoded);

    // 2nd draw shouldn't be replaced: still grey.
    TEST_EQ(*((uint32_t*)backing), grey);

    return true;
}

bool Validator::TestCompute()
{
    llgd_lwn::ProgramHolder zeroProgram;
    llgd_lwn::ProgramHolder oneProgram;
    zeroProgram.Initialize((Device*)g_device);
    oneProgram.Initialize((Device*)g_device);
    {
        const auto base = mphUcodes->GetBufferAddress();
        const auto zeroCs = LoadShader(base, ucodes, zeroCsData);
        const auto oneCs = LoadShader(base + UCODE, ucodes + UCODE, oneCsData);

        std::array<ShaderData, 1> shaders;

        shaders[0] = zeroCs;
        TEST(zeroProgram->SetShaders(shaders.size(), &shaders[0]));

        shaders[0] = oneCs;
        TEST(oneProgram->SetShaders(shaders.size(), &shaders[0]));
    }

    uint32_t& result = *(uint32_t*)backing;
    result = 0x42;

    const auto handle = editingHelper->MakeHandle([&] (CommandBuffer* cb) {
        cb->BindProgram(zeroProgram, ShaderStageBits::COMPUTE);
        cb->BindStorageBuffer(ShaderStage::COMPUTE, 0, mph->GetBufferAddress(), sizeof(uint32_t));
        cb->DispatchCompute(1, 1, 1);
    });

    const auto state = editingHelper->RunAndExtractGpuState(handle);
    if (result != 0x0) return false;

    LlgdProgram zeroLlgdProgram = ToLlgdProgram(zeroProgram);
    LlgdProgram oneLlgdProgram = ToLlgdProgram(oneProgram);

    editingHelper->ResetPointersForEditingCB();
    const auto patched = llgdCommandSetReplaceProgram(
        handle,
        [](uint32_t index, void*) { return 10 + index; }, 0, 1000,
        state, state,
        &zeroLlgdProgram, &oneLlgdProgram,
        editingHelper->WriteControlMemoryForEditing,
        editingHelper->WriteCommandMemoryForEditing,
        editingHelper.get());
    const auto decoded = editingHelper->MakeCommandHandleRunnable(patched);
    editingHelper->Run(decoded);
    if (result != 0x1) return false;

    return true;
}

bool Validator::Test()
{
    for (int i = 0; i < 100; ++i) {
        TEST(TestStrCmp("OPQR", "1234", "OPQR", "1234", 4))
        TEST(TestStrCmp("OPQR", "12345678", "ABCDefghOPQRwxyz", "ABCDefgh12345678wxyz", 20))
        TEST(TestStrCmp("OPQROPQR", "9012", "ABCDefghOPQROPQRwxyz",
            "ABCDefgh9012wxyz", 16,
            "ABCDefgh9012\x02\x00\x00\x80wxyz", 20))
        TEST(TestStrCmp("OPQR", "1234", "abcdefghXOPQOPQRwxyz", "abcdefghXOPQ1234wxyz", 19))
    }

    TEST(TestColors(0, 0))
    TEST(TestColors(-1, 0))
    TEST(TestColors(+1, 0))
    TEST(TestColors(0, -1))
    TEST(TestColors(0, +1))
    TEST(TestColors(-10, 0))
    TEST(TestColors(+10, 0))
    TEST(TestColors(0, -10))
    TEST(TestColors(0, +10))

    TEST(TestCompute())

    TEST(TestStopEditAtFirstDraw())

    return true;
}

LLGD_DEFINE_TEST(EditProgramReplace, UNIT,
LwError Execute()
{
    auto v = std::make_unique<Validator>();
    return (v->Initialize() && v->Test()) ? LwSuccess : LwError_IlwalidState;
});
}
