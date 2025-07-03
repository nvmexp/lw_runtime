#pragma once

/*
 * Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU CORPORATION is strictly prohibited.
 */

#include "lwn/lwn_Cpp.h"

class GlslcProgram
{
public:
    // Called implicitly but can be called explicitly if maxGPUMemory needs to be changed.
    void LoadGlslcProgramHelper(lwn::Device *device, size_t maxGPUMemory = 0x100000UL);
    lwn::Program *CreateProgram(lwn::Device *device, const char *vertShader, const char *fragShader);
    ~GlslcProgram();
};
