/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#pragma once

#include "utils.hpp"
#include <vector>
#include <string>

// Use this class as follows:
//
// GlProgram* prog = new GlProgram();
// prog->shaderSource(GlProgramm::VertexShader, "myshader");
// prog->shaderSource(GlProgramm::FragmentShader, "myshader2");
//
// prog->useProgram();
//
// GlProgramm::useProgram() will on first use attach shaders and link.
// Use shaderSource only once per shader type, and do not use them
// after useProgram.

class GlProgram
{
public:
    enum ShaderType {
        VertexShader,
        FragmentShader,
        TessControlShader,
        TessEvaluationShader,
        ComputeShader,
        ShaderMax,
    };

    GlProgram(void);
    ~GlProgram(void);

    void shaderSource(ShaderType ty, const std::string& s, const std::vector<std::string>* prepend = NULL);
    void useProgram(void);

    GLint uniformLocation(const char* name) const;
    GLint attribLocation(const char* name) const;

    GLuint programHandle(void) const { return m_program; }

private:
    GlProgram(const GlProgram&);
    GlProgram& operator=(const GlProgram&);

    std::vector<int> m_shaderHandles;
    uint32_t         m_setShaders;
    GLuint           m_program;
    bool             m_linked;
};
