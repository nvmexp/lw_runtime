/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include "glprogram.hpp"

#include <stdio.h>
#include <assert.h>

GlProgram::GlProgram(void) :
    m_setShaders(0),
    m_linked(false)
{
    m_shaderHandles.resize(ShaderMax);
}

GlProgram::~GlProgram(void)
{
    LwnUtil::g_glDeleteProgram(m_program);
}

void GlProgram::shaderSource(ShaderType ty, const std::string& s, const std::vector<std::string>* prepend)
{
    const char* typestr = NULL;

    char compilerSpew[256];
    assert(!m_linked);
    assert(ty >= 0 && ty < ShaderMax);
    // ShaderSource ok only once per shdr type
    assert((m_setShaders & (1<<(int)ty)) == 0);
    m_setShaders |= 1 << (int)ty;

    GLuint glShaderType = 0;

    switch(ty) {
    case VertexShader:
        glShaderType = GL_VERTEX_SHADER;
        typestr = "vertex shader";
        break;
    case FragmentShader:
        glShaderType = GL_FRAGMENT_SHADER;
        typestr = "fragment shader";
        break;
    case TessControlShader:
        glShaderType = GL_TESS_CONTROL_SHADER;
        typestr = "tess control shader";
        break;
    case TessEvaluationShader:
        glShaderType = GL_TESS_EVALUATION_SHADER;
        typestr = "tess evaluation shader";
        break;
    case ComputeShader:
        glShaderType = GL_COMPUTE_SHADER;
        typestr = "compute shader";
        break;
    default: assert(0);
    }

    GLuint sh = LwnUtil::g_glCreateShader(glShaderType);
    m_shaderHandles[(int)ty] = sh;

    //void shaderSource(ShaderType ty, const FW::String& s, const FW::Array<String>* prepend = NULL);
    std::vector<const char*> shaderParts;
    if (prepend) {
        for (size_t i = 0; i < prepend->size(); i++)
            shaderParts.push_back((*prepend)[i].data());
    }
    shaderParts.push_back(s.data());

    LwnUtil::g_glShaderSource(sh, (int)shaderParts.size(), shaderParts.data(), 0);
    LwnUtil::g_glCompileShader(sh);

    GLint compileSuccess;
    LwnUtil::g_glGetShaderiv(sh, GL_COMPILE_STATUS, &compileSuccess);
    LwnUtil::g_glGetShaderInfoLog(sh, sizeof(compilerSpew), 0, compilerSpew);

    if (!compileSuccess)
    {
        ::printf("GlProgramm::shaderSource (%s): compile error: %s\n", typestr, compilerSpew);
        assert(0);
    }
}

void GlProgram::useProgram(void)
{
    char compilerSpew[256];

    if (!m_linked)
    {
        m_program = LwnUtil::g_glCreateProgram();

        for (int i = 0; i < ShaderMax; i++)
        {
            if (m_setShaders & (1<<i))
                LwnUtil::g_glAttachShader(m_program, m_shaderHandles[i]);
        }

        LwnUtil::g_glLinkProgram(m_program);

        GLint linkSuccess;
        LwnUtil::g_glGetProgramiv(m_program, GL_LINK_STATUS, &linkSuccess);
        LwnUtil::g_glGetProgramInfoLog(m_program, sizeof(compilerSpew), 0, compilerSpew);
        if (!linkSuccess)
        {
            ::printf("GlProgramm::useProgram: link error: %s\n", compilerSpew);
            assert(0);
        }

        m_linked = true;
    }

    LwnUtil::g_glUseProgram(m_program);
}

GLint GlProgram::uniformLocation(const char* name) const
{
    return LwnUtil::g_glGetUniformLocation(m_program, name);
}

GLint GlProgram::attribLocation(const char* name) const
{
    return LwnUtil::g_glGetAttribLocation(m_program, name);
}
