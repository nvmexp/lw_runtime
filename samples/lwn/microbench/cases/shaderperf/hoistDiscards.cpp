/*
* Copyright (c) 2017 LWPU Corporation.  All rights reserved.
*
* LWPU Corporation and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from LWPU Corporation is strictly prohibited.
*/

#include "cases/shaderperf.hpp"

// "hoistDiscards" pragma performance test:
//   perf pragma "hoistDiscards" modifies scheduler behavior to schedule KIL instructions as early as possible, in spite of stalls.

//   The shader that benefited from this change had the following features :
//   1. High pixel discard rate
//   2. Discard dependent on high latency instruction(such as TEX)
//   3. IPA bottleneck (the discard was independent of these varyings and could therefore be scheduled earlier)

// For hoistDiscards.glsl:
//   We use the varyings multiple times, and then the discard is done for half the screen.
//   The texture read is done after the discard, so the KIL instruction can be scheduled earlier when hoistDiscards == true

struct SubtestParams
{
    bool hoistDiscards;

    SubtestParams(bool p) : hoistDiscards(p)
    {
    }
};

class HoistDiscardsTest : public ShaderTest
{
private:
    LWNprogram               m_pgm;

public:
    HoistDiscardsTest(ShaderTest::Context* params, SubtestParams subtest);
    virtual ~HoistDiscardsTest();
};

#include "shaders/g_hoistDiscards.h"

static const char *VS_STRING =
"#version 440 core\n"
"#extension GL_LW_gpu_shader5:require\n"
"layout(location = 0) in vec3 position;\n"
"layout(location = 1) in vec2 texCoord;\n"
"out IO {\n"
"   vec4 varying1;\n"
"   float varying2;\n"
"   float varying3;\n"
"   vec4 varying4;\n"
"   float varying5;\n"
"   float varying6;\n"
"   flat int varying7;\n"
"   vec2 vPos;\n"
"   vec2 fTexCoord;\n"
"};\n"
"void main() {\n"
"  gl_Position = vec4(position, 1.0);\n"
"  varying1 = vec4(0.5, 0.5, 0.5, 1.0);\n"
"  varying2 = 0.14;\n"
"  varying3 = 0.75;\n"
"  varying4 = vec4(0.24, 0.21, 0.56, 4.5);\n"
"  varying5 = 0.05;\n"
"  varying6 = 0.5;\n"
"  varying7 = 1;\n"
"  vPos = position.xy;"
"  fTexCoord = texCoord;\n"
"}\n";

HoistDiscardsTest::HoistDiscardsTest(ShaderTest::Context* context, SubtestParams params) :
    ShaderTest(context)
{
    LWNcommandBuffer *cmd = cmdBuf()->cmd();
    // Create programs from the device, provide them shader code and
    // compile/link them
    lwnProgramInitialize(&m_pgm, device());

    // replace defines
    const char *fragShader = shader_hoistDiscards;
    std::string shaderTemp(fragShader);
    // replace the define for HOISTDISCARDS
    if (params.hoistDiscards) {
        shaderTemp.replace(shaderTemp.find("#define HOISTDISCARDS 0"), strlen("#define HOISTDISCARDS 0"), "#define HOISTDISCARDS 1");
    }
    // replace the define for SAMPLERINDEX
    int samplerIndex = 4;
    char samplerIndexStr[25];
    sprintf(samplerIndexStr, "#define SAMPLERINDEX %d", samplerIndex);
    shaderTemp.replace(shaderTemp.find("#define SAMPLERINDEX 0"), strlen("#define SAMPLERINDEX 0"), samplerIndexStr);
    fragShader = shaderTemp.c_str();

    // set shader scratch memory
    // TODO: Callwlate exact size needed
    LwnUtil::setShaderScratchMemory(gpuPool(), DEFAULT_SHADER_SCRATCH_MEMORY_SIZE, cmd);

    LWNshaderStage stages[2] = { LWN_SHADER_STAGE_VERTEX, LWN_SHADER_STAGE_FRAGMENT };
    const char *sources[2] = { VS_STRING, fragShader };

    int32_t nSources = 2;

    if (!LwnUtil::compileAndSetShaders(&m_pgm, stages, nSources, sources))
    {
        assert(0);
        return;
    }

    setupSampler2DTexture(1024, 1024, samplerIndex);

    lwnCommandBufferBindProgram(cmd, &m_pgm, LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);
}

HoistDiscardsTest::~HoistDiscardsTest()
{
    lwnProgramFinalize(&m_pgm);
}

SHADERTEST_CREATE(HoistDiscardsTest, shaderperf_hoistDiscards_hoistDiscards0, SubtestParams(false))
SHADERTEST_CREATE(HoistDiscardsTest, shaderperf_hoistDiscards_hoistDiscards1, SubtestParams(true))