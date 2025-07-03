/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include <iostream>

#define NOMINMAX

#include "lwn/lwn.h"
#include "lwn/lwn_Cpp.h"
#include "lwn/lwn_CppFuncPtr.h"
#include "lwn/lwn_CppMethods.h"
#include "lwn/lwn_FuncPtrGlobal.h"

#define LWNUTIL_USE_CPP_INTERFACE
#include "lwnUtil/lwnUtil_Interface.h"
#include "lwnUtil/lwnUtil_AlignedStorage.h"
#include "lwnUtil/lwnUtil_AlignedStorageImpl.h"
#include "lwnUtil/lwnUtil_GlslcHelper.h"
#include "lwnUtil/lwnUtil_GlslcHelperImpl.h"
#include "lwnUtil/lwnUtil_PoolAllocator.h"
#include "lwnUtil/lwnUtil_PoolAllocatorImpl.h"

#include "lwnGlslc.h"

// Make these global instead of class members so that other pieces of code
// are sheltered from the glslc related lwnutil header files.

static lwnUtil::GLSLCLibraryHelper *g_glslcLibraryHelper = NULL;
static lwnUtil::GLSLCHelper *g_glslcHelper = NULL;

void GlslcProgram::LoadGlslcProgramHelper(lwn::Device *device, size_t maxGPUMemory)
{
    g_glslcLibraryHelper = new lwnUtil::GLSLCLibraryHelper;
    g_glslcLibraryHelper->LoadDLL(NULL);
    g_glslcHelper = new lwnUtil::GLSLCHelper(reinterpret_cast<LWNdevice *> (device),
                                             maxGPUMemory, g_glslcLibraryHelper);
}

lwn::Program *GlslcProgram::CreateProgram(lwn::Device *device, const char *vertShader, const char *fragShader)
{
    if (g_glslcHelper == NULL) {
        LoadGlslcProgramHelper(device);
    }
    lwn::Program *pgm = new lwn::Program;
    pgm->Initialize(device);

    lwn::ShaderStage stages[2] = {lwn::ShaderStage::VERTEX, lwn::ShaderStage::FRAGMENT};
    const char *sources[] = {vertShader, fragShader};
    int nSources = 2;

    if (!g_glslcHelper->CompileAndSetShaders(pgm, stages, nSources, sources))
    {
        printf("Shader compile error. infoLog=\n%s\n", g_glslcHelper->GetInfoLog());
        return NULL;
    }

    return pgm;
}

GlslcProgram::~GlslcProgram()
{
    if (g_glslcHelper)
    {
        delete g_glslcHelper;
        delete g_glslcLibraryHelper;

        g_glslcHelper = NULL;
        g_glslcLibraryHelper = NULL;
    }
}
