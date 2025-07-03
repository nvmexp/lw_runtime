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
// Dead Code Elimination (DCE) comparison

// For dce_lmemaccess.glsl:
// When IS_PREPROCESSED == 0, we specialize the uniforms and try to eliminate
// lmemaccess asm.
// Performance should be similar between the preprocessed and specialized
// cases.

struct SubtestParams
{
    bool preprocessed;
    int specConstCount;
    int specConstIndex;
    int specConstBranch;

    SubtestParams(bool p, int count, int index, int branch) : preprocessed(p), specConstCount(count), specConstIndex(index), specConstBranch(branch)
    {
    }
};

class DceTest : public ShaderTest
{
private:
    struct Block1
    {
        int specConstCount;
        int specConstIndex;
        int specConstBranch;
    };

    LwnUtil::UboArr<Block1>  m_ubo;
    LWNprogram               m_pgm;
public:
    DceTest(ShaderTest::Context* params, SubtestParams subtest);
    virtual ~DceTest();
};

#include "shaders/g_dce_lmemaccess.h"

static const char *VS_STRING =
"#version 440 core\n"
"#extension GL_LW_gpu_shader5:require\n"
"layout(location = 0) in vec3 position;\n"
"layout(location = 1) in vec2 texCoord;\n"
"out IO { vec4 vtxColor; };\n"
"void main() {\n"
"  gl_Position = vec4(position, 1.0);\n"
"  vtxColor = vec4(texCoord.x, texCoord.y, 0, 0);\n"
"}\n";

DceTest::DceTest(ShaderTest::Context* context, SubtestParams params) :
    ShaderTest(context),
    m_ubo(device(), coherentPool(), 1)
{
    LWNcommandBuffer *cmd = cmdBuf()->cmd();
    // Create programs from the device, provide them shader code and
    // compile/link them
    lwnProgramInitialize(&m_pgm, device());

    const char *fragShader = shader_dce_lmemaccess;
    // replace the defines for preprocessed versions
    std::string shaderTemp(fragShader);
    if (params.preprocessed) {
        shaderTemp.replace(shaderTemp.find("#define IS_PREPROCESSED 0"), strlen("#define IS_PREPROCESSED 0"), "#define IS_PREPROCESSED 1");

        // replace the uniforms
        char specConst[25];
        sprintf(specConst, "#define UNI_COUNT %d", params.specConstCount);
        shaderTemp.replace(shaderTemp.find("#define UNI_COUNT 0"), strlen("#define UNI_COUNT 0"), specConst);

        sprintf(specConst, "#define UNI_INDEX %d", params.specConstIndex);
        shaderTemp.replace(shaderTemp.find("#define UNI_INDEX 0"), strlen("#define UNI_INDEX 0"), specConst);

        sprintf(specConst, "#define UNI_BRANCH %d", params.specConstBranch);
        shaderTemp.replace(shaderTemp.find("#define UNI_BRANCH 0"), strlen("#define UNI_BRANCH 0"), specConst);

        fragShader = shaderTemp.c_str();
    } else {
        // setup specialized uniforms (setupConstCount, setupConstIndex, setupConstBranch)
        // TODO: Add a function so we don't have to repeat these
        GLSLCspecializationUniform uniform_specConstCount;
        LwnUtil::ArrayUnion arrys[1];
        uniform_specConstCount.values = (void*)(&arrys[0]);
        LwnUtil::setData(&uniform_specConstCount, "specConstCount", 1, LwnUtil::ARG_TYPE_INT, 1, params.specConstCount);
        LwnUtil::addSpecializationUniform(0, &uniform_specConstCount);

        GLSLCspecializationUniform uniform_specConstIndex;
        LwnUtil::ArrayUnion arrys2[1];
        uniform_specConstIndex.values = (void*)(&arrys2[0]);
        LwnUtil::setData(&uniform_specConstIndex, "specConstIndex", 1, LwnUtil::ARG_TYPE_INT, 1, params.specConstIndex);
        LwnUtil::addSpecializationUniform(0, &uniform_specConstIndex);

        GLSLCspecializationUniform uniform_specConstBranch;
        LwnUtil::ArrayUnion arrys3[1];
        uniform_specConstBranch.values = (void*)(&arrys3[0]);
        LwnUtil::setData(&uniform_specConstBranch, "specConstBranch", 1, LwnUtil::ARG_TYPE_INT, 1, params.specConstBranch);
        LwnUtil::addSpecializationUniform(0, &uniform_specConstBranch);
    }

    LWNshaderStage stages[2] = { LWN_SHADER_STAGE_VERTEX, LWN_SHADER_STAGE_FRAGMENT };
    const char *sources[2] = { VS_STRING, fragShader };

    int32_t nSources = 2;

    if (!LwnUtil::compileAndSetShaders(&m_pgm, stages, nSources, sources))
    {
        assert(0);
        return;
    }

    if (!params.preprocessed) {
        LwnUtil::clearSpecializationUniformArrays();
    }
    lwnCommandBufferBindProgram(cmd, &m_pgm, LWN_SHADER_STAGE_VERTEX_BIT | LWN_SHADER_STAGE_FRAGMENT_BIT);

    Block1 b;
    b.specConstCount = params.specConstCount;
    b.specConstIndex = params.specConstIndex;
    b.specConstBranch = params.specConstBranch;
    m_ubo.set(0, b);
    lwnCommandBufferBindUniformBuffer(cmd, LWN_SHADER_STAGE_FRAGMENT,
                                      0,
                                      m_ubo.address() + m_ubo.offset(0),
                                      sizeof(Block1));
}

DceTest::~DceTest()
{
    lwnProgramFinalize(&m_pgm);
}

SHADERTEST_CREATE(DceTest, shaderperf_dce_preprocessed0_count0_index1_branch1, SubtestParams(false, 0, 1, 1))
SHADERTEST_CREATE(DceTest, shaderperf_dce_preprocessed0_count4_index1_branch1, SubtestParams(false, 4, 1, 1))
SHADERTEST_CREATE(DceTest, shaderperf_dce_preprocessed0_count4_index1_branch0, SubtestParams(false, 4, 1, 0))
SHADERTEST_CREATE(DceTest, shaderperf_dce_preprocessed1_count0_index1_branch1, SubtestParams(true,  0, 1, 1))
SHADERTEST_CREATE(DceTest, shaderperf_dce_preprocessed1_count4_index1_branch1, SubtestParams(true,  4, 1, 1))
SHADERTEST_CREATE(DceTest, shaderperf_dce_preprocessed1_count4_index1_branch0, SubtestParams(true,  4, 1, 0))