/*
 * Copyright (c) 2005 - 2018 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#include <cassert>
#include "error.h"
#include "gl_util.h"

bool CheckGlError(const char* op)
{
    bool result = false;

    while (GLint error = glGetError())
    {
        const char* error_string = "";
        switch (error)
        {
        case GL_NO_ERROR:
            error_string = "GL_NO_ERROR";
            break;

        case GL_ILWALID_ENUM:
            error_string = "GL_ILWALID_ENUM";
            break;

        case GL_ILWALID_VALUE:
            error_string = "GL_ILWALID_VALUE";
            break;

        case GL_ILWALID_OPERATION:
            error_string = "GL_ILWALID_OPERATION";
            break;

        case GL_STACK_OVERFLOW:
            error_string = "GL_STACK_OVERFLOW";
            break;

        case GL_STACK_UNDERFLOW:
            error_string = "GL_STACK_UNDERFLOW";
            break;

        case GL_OUT_OF_MEMORY:
            error_string = "GL_OUT_OF_MEMORY";
            break;

        case GL_TABLE_TOO_LARGE:
            error_string = "GL_TABLE_TOO_LARGE";
            break;

        case GL_ILWALID_FRAMEBUFFER_OPERATION:
            error_string = "GL_ILWALID_FRAMEBUFFER_OPERATION";
            break;

        default:
            assert(!"unknown error");
        }

        DEBUG_MSG("after %s() glError %s\n", op, error_string);
        result = true;
    }
    return result;
}

GLuint LoadShader(GLenum shaderType, GLsizei count, const GLchar **pSource)
{
    GLuint shader = glCreateShader(shaderType);

    if (!shader)
    {
        Error("Cannot create shader");
    }

    glShaderSource(shader, count, pSource, NULL);
    glCompileShader(shader);
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);

    if (!compiled)
    {
        GLint infoLen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);

        if (infoLen)
        {
            char* buf = (char*) malloc(infoLen);
            if (buf)
            {
                glGetShaderInfoLog(shader, infoLen, NULL, buf);
                DEBUG_MSG("Could not compile shader %d:\n%s\n",
                          shaderType, buf);
                free(buf);
            }
        }
        else
        {
            DEBUG_MSG("Guessing at GL_INFO_LOG_LENGTH size\n");
            char* buf = (char*) malloc(0x1000);
            if (buf)
            {
                glGetShaderInfoLog(shader, 0x1000, NULL, buf);
                DEBUG_MSG("Could not compile shader %d:\n%s\n",
                          shaderType, buf);
                free(buf);
            }
        }
        glDeleteShader(shader);
        Error("Shader compilation failed");
    }

    return shader;
}

GLuint CreateProgram(GLuint vertShader, GLuint fragShader)
{
    GLuint program = glCreateProgram();

    if (!program)
    {
        Error("Cannot create program");
    }
    glAttachShader(program, vertShader);
    glAttachShader(program, fragShader);

    glLinkProgram(program);
    GLint linkStatus = GL_FALSE;

    glGetProgramiv(program, GL_LINK_STATUS, &linkStatus);

    if (linkStatus != GL_TRUE)
    {
        GLint bufLength = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &bufLength);
        if (bufLength)
        {
            char* buf = (char*) malloc(bufLength);
            if (buf)
            {
                glGetProgramInfoLog(program, bufLength, NULL, buf);
                DEBUG_MSG("Could not link program:\n%s\n", buf);
                free(buf);
            }
        }
        glDeleteProgram(program);
        program = 0;
        Error("Failed shader linking");
    }
    return program;
}
