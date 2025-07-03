/*
 * Copyright (c) 2016 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "cases/shaderperf.hpp"

// Shader Specialization performance test
// IPA Scheduling:
//   IPA instruction = Evaluate an attribute for this fragment.
//   Consider a shader that uses a varying at the start of the program and then again much later
//   Assume that shader specialization renders early uses of a varying dead, but the later ones
//   are still live.
//   This results in IPAs and their uses being scheduled too far apart and increases register pressure.

// For ipa_scheduling.glsl:
// In the preprocessed case when COND == false, the if condition will be
// eliminated, thus removing earlier uses of the varyings
// When the bool value specConst is specialized to false, the if condition
// should be eliminated similar to the preprocessed COND == false case

struct SubtestParams
{
    bool preprocessed;
    bool specConst;

    SubtestParams(bool p, bool s) : preprocessed(p), specConst(s)
    {
    }
};

class IpaTest : public ShaderTest
{
private:
    struct Block1
    {
        bool specConst;
        unsigned int loops;
    };

    LwnUtil::UboArr<Block1>  m_ubo;
    LWNprogram               m_pgm;
public:
    IpaTest(ShaderTest::Context* params, SubtestParams subtest);
    virtual ~IpaTest();
};

#include "shaders/g_ipa_scheduling.h"

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
"};\n"
"void main() {\n"
"  gl_Position = vec4(position, 1.0);\n"
"  varying1 = vec4(0.5, 0.5, 0.5, 1.0);\n"
"  varying2 = 1.5;\n"
"  varying3 = 0.75;\n"
"  varying4 = vec4(1.24, 5.6, 0.56, 4.5);\n"
"  varying5 = 3.2;\n"
"  varying6 = 0.5;\n"
"  varying7 = 2;\n"
"}\n";

IpaTest::IpaTest(ShaderTest::Context* context, SubtestParams params) :
    ShaderTest(context),
    m_ubo(device(), coherentPool(), 1)
{
    LWNcommandBuffer *cmd = cmdBuf()->cmd();
    // Create programs from the device, provide them shader code and
    // compile/link them
    lwnProgramInitialize(&m_pgm, device());

    const char *fragShader = shader_ipa_scheduling;
    // replace the defines for preprocessed versions
    std::string shaderTemp(fragShader);
    if (params.preprocessed) {
        shaderTemp.replace(shaderTemp.find("#define UNI_SPEC 0"), strlen("#define UNI_SPEC 0"), "#define UNI_SPEC 1");
        fragShader = shaderTemp.c_str();
    }

    // setup specialized uniforms (setupConst1, setupConst2)
    GLSLCspecializationUniform uniform_specConst;
    LwnUtil::ArrayUnion arrys[1];
    uniform_specConst.values = (void*)(&arrys[0]);
    LwnUtil::setData(&uniform_specConst, "specConst", 1, LwnUtil::ARG_TYPE_INT, 1, 1);
    LwnUtil::addSpecializationUniform(0, &uniform_specConst);

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

    LwnUtil::clearSpecializationUniformArrays();
    lwnCommandBufferBindProgram(cmd, &m_pgm, LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);

    Block1 b;
    b.specConst = params.specConst;
    b.loops = 128;
    m_ubo.set(0, b);
    lwnCommandBufferBindUniformBuffer(cmd, LWN_SHADER_STAGE_FRAGMENT,
                                      0,
                                      m_ubo.address() + m_ubo.offset(0),
                                      sizeof(Block1));
}

IpaTest::~IpaTest()
{
    lwnProgramFinalize(&m_pgm);
}

SHADERTEST_CREATE(IpaTest, shaderperf_ipa_preprocessed0_specConst0, SubtestParams(false, false))
SHADERTEST_CREATE(IpaTest, shaderperf_ipa_preprocessed0_specConst1, SubtestParams(false, true))
SHADERTEST_CREATE(IpaTest, shaderperf_ipa_preprocessed1_specConst0, SubtestParams(true, false))
SHADERTEST_CREATE(IpaTest, shaderperf_ipa_preprocessed1_specConst1, SubtestParams(true, true))
