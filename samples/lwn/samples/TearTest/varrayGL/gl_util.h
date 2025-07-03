#pragma once

/*
 * Copyright (c) 2005 - 2018 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */


#include <stdio.h>
#include <stdlib.h>
#include "gl_pointers.h"

#define DEBUG_MSG printf

bool CheckGlError(const char* op);
GLuint LoadShader(GLenum shaderType, GLsizei count, const GLchar **pSource);
GLuint CreateProgram(GLuint vertShader, GLuint fragShader);
