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
// Algebraic expressions

struct SubtestParams
{
    bool preprocessed;
    float specConst;

    SubtestParams(bool p, float s) : preprocessed(p), specConst(s)
    {
    }
};

class ExprTest : public ShaderTest
{
private:
    struct Block1
    {
        float specConst;
        float val1[4];
        float val2;
        float val3;
    };

    LwnUtil::UboArr<Block1>  m_ubo;
    LWNprogram               m_pgm;
public:
    ExprTest(ShaderTest::Context* params, SubtestParams subtest);
    virtual ~ExprTest();
};

#include "shaders/g_algebraic_expression.h"

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

ExprTest::ExprTest(ShaderTest::Context* context, SubtestParams params) :
    ShaderTest(context),
    m_ubo(device(), coherentPool(), 1)
{
    LWNcommandBuffer *cmd = cmdBuf()->cmd();
    // Create programs from the device, provide them shader code and
    // compile/link them
    lwnProgramInitialize(&m_pgm, device());

    const char *fragShader = shader_algebraic_expression;
    // replace the defines for preprocessed versions
    std::string shaderTemp(fragShader);
    if (params.preprocessed) {
        shaderTemp.replace(shaderTemp.find("#define IS_PREPROCESSED 0"), strlen("#define IS_PREPROCESSED 0"), "#define IS_PREPROCESSED 1");

        char specConst[25];
        sprintf(specConst, "#define UNI_SPEC %.2f", params.specConst);
        shaderTemp.replace(shaderTemp.find("#define UNI_SPEC 0.00"), strlen("#define UNI_SPEC 0.00"), specConst);
        fragShader = shaderTemp.c_str();
    }
    else {
        // setup specialized uniforms (specConst)
        GLSLCspecializationUniform uniform_specConst;
        LwnUtil::ArrayUnion arrys[1];
        uniform_specConst.values = (void*)(&arrys[0]);
        LwnUtil::setData(&uniform_specConst, "specConst", 1, LwnUtil::ARG_TYPE_FLOAT, 1, params.specConst);
        LwnUtil::addSpecializationUniform(0, &uniform_specConst);
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
    b.specConst = params.specConst;
    b.val1[0] = 0.8f; b.val1[1] = 0.9f; b.val1[2] = 0.6f; b.val1[3] = 1.0f;
    b.val2 = 0.8f;
    b.val3 = 0.65f;
    m_ubo.set(0, b);
    lwnCommandBufferBindUniformBuffer(cmd, LWN_SHADER_STAGE_FRAGMENT,
                                      0,
                                      m_ubo.address() + m_ubo.offset(0),
                                      sizeof(Block1));
}

ExprTest::~ExprTest()
{
    lwnProgramFinalize(&m_pgm);
}

SHADERTEST_CREATE(ExprTest, shaderperf_expr_preprocessed0_specConst2, SubtestParams(false, 2))
SHADERTEST_CREATE(ExprTest, shaderperf_expr_preprocessed1_specConst2, SubtestParams(true, 2))