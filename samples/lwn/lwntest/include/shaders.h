/*
 * Copyright (c) 2009 - 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __SHADERS_H__
#define __SHADERS_H__

//
// shaders.h:  General definitions used for shader-related utilities.
//

// LWShaderStage:  Enum defining a particular programmable stage that will be
// targeted by a program.  Used, for example, in C++ shader utilities for
// EXT_separate_shader_objects where a (potentially multi-stage) program
// object is designated as active for only a single programmable stage.
typedef enum {
    VERTEX_STAGE,
    FRAGMENT_STAGE,
    GEOMETRY_STAGE,
    TESS_CONTROL_STAGE,
    TESS_EVALUATION_STAGE,
    COMPUTE_STAGE,

    LW_SHADER_STAGE_COUNT,      // number of shader stages supported
    ILWALID_SHADER_STAGE = LW_SHADER_STAGE_COUNT,
} LWShaderStage;

// LWProgramType, LWShaderType:  Enum defining the particular type of a given
// assembly program object or GLSL shader object.  Used to determine the
// particular APIs that should be used to manipulate the object, as well as
// text that shader-related utility APIs may prepend onto programs/shaders of
// that type.
typedef enum LWProgramType {

    VP10,                       // !!VP1.0
    VP11,                       // !!VP1.1
    VP20,                       // !!VP2.0
    ARBvp10,                    // !!ARBvp1.0 (no LW option)
    ARBvpn20,                   // !!ARBvp1.0 + LW_vertex_program2 option
    ARBvpn30,                   // !!ARBvp1.0 + LW_vertex_program3 option
    LWvp40,                     // !!LWvp4.0
    FP10,                       // !!FP1.0
    ARBfp10,                    // !!ARBfp1.0 (no LW option)
    ARBfpn10,                   // !!ARBfp1.0 + LW_fragment_program option
    ARBfpn20,                   // !!ARBfp1.0 + LW_fragment_program2 option
    LWfp40,                     // !!LWfp4.0
    LWgp40,                     // !!LWgp4.0
    LWvp41,                     // !!LWvp4.1
    LWgp41,                     // !!LWgp4.1
    LWfp41,                     // !!LWfp4.1
    LWvp50,                     // !!LWvp5.0
    LWgp50,                     // !!LWvp5.0
    LWfp50,                     // !!LWvp5.0
    LWtcp50,                    // !!LWtcp5.0
    LWtep50,                    // !!LWtep5.0

    LWProgramTypeFirstASM = VP10,
    LWProgramTypeLastASM = LWtep50,

    VSarb,                      // ARB_vertex_shader
    FSarb,                      // ARB_fragment_shader

    LWProgramTypeFirstGLSLarbso = VSarb,
    LWProgramTypeLastGLSLarbso = FSarb,

    VScore,                     // GLSL core vertex (no #version)
    VS120,                      // GLSL vertex (GLSL 1.20)
    VS130,                      // GLSL vertex (GLSL 1.30)
    VS140,                      // GLSL vertex (GLSL 1.40)
    VS150,                      // GLSL vertex (GLSL 1.50)
    VS150comp,                  // GLSL vertex (GLSL 1.50 + compatibility)
    VS330,                      // GLSL vertex (GLSL 3.30)
    VS400,                      // GLSL vertex (GLSL 4.00)
    VS410,                      // GLSL vertex (GLSL 4.10)
    VS420,                      // GLSL vertex (GLSL 4.20)
    VS430,                      // GLSL vertex (GLSL 4.30)
    VS440,                      // GLSL vertex (GLSL 4.40)
    VS450,                      // GLSL vertex (GLSL 4.50)
    VSlw50,                     // GLSL vertex (LW_gpu_shader5 + #version 140)
    VSes2,                      // GLSL vertex (#version 100 which enables GLSL-ES mode under OpenGL 4.1)
    VSes3,                      // GLSL vertex (#version 300 es which enables GLSL-ES mode under OpenGL 4.3)
    VSes31,                     // GLSL vertex (#version 310 es which enables GLSL-ES mode under OpenGL 4.3)
    VSes32,                     // GLSL vertex (#version 320 es which enables GLSL-ES mode)

    FScore,                     // GLSL fragment (no #version)
    FS120,                      // GLSL fragment (GLSL 1.20)
    FS130,                      // GLSL fragment (GLSL 1.30)
    FS140,                      // GLSL fragment (GLSL 1.40)
    FS150,                      // GLSL fragment (GLSL 1.50)
    FS150comp,                  // GLSL fragment (GLSL 1.50 + compatibility)
    FS330,                      // GLSL fragment (GLSL 3.30)
    FS400,                      // GLSL fragment (GLSL 4.00)
    FS410,                      // GLSL fragment (GLSL 4.10)
    FS420,                      // GLSL fragment (GLSL 4.20)
    FS430,                      // GLSL fragment (GLSL 4.30)
    FS440,                      // GLSL fragment (GLSL 4.40)
    FS450,                      // GLSL fragment (GLSL 4.50)
    FSlw50,                     // GLSL fragment (LW_gpu_shader5 + #version 140)
    FSes2,                      // GLSL fragment (#version 100 which enables GLSL-ES mode under OpenGL 4.1)
    FSes3,                      // GLSL fragment (#version 300 es which enables GLSL-ES mode under OpenGL 4.3)
    FSes31,                     // GLSL fragment (#version 310 es which enables GLSL-ES mode under OpenGL 4.3)
    FSes32,                     // GLSL fragment (#version 320 es which enables GLSL-ES mode)

    GSext,                      // GLSL geometry (EXT_geometry_shader + #version 120)
    GSext130,                   // GLSL geometry (EXT_geometry_shader + #version 130)
    GSext140,                   // GLSL geometry (EXT_geometry_shader + #version 140)
    GS150,                      // GLSL geometry (GLSL 1.50)
    GS150comp,                  // GLSL geometry (GLSL 1.50 + compatibility)
    GS330,                      // GLSL geometry (GLSL 3.30)
    GS400,                      // GLSL geometry (GLSL 4.00)
    GS410,                      // GLSL geometry (GLSL 4.10)
    GS420,                      // GLSL geometry (GLSL 4.20)
    GS430,                      // GLSL geometry (GLSL 4.30)
    GS440,                      // GLSL geometry (GLSL 4.40)
    GS450,                      // GLSL geometry (GLSL 4.50)
    GSlw50,                     // GLSL geometry (LW_gpu_shader5 + #version 140)
    GSesext31,                  // GLSL geometry (EXT_geometry_shader)
    GSesoes31,                  // GLSL geometry (OES_geometry_shader)
    GSes32,                     // GLSL geometry (geometry_shader)

    TCSlw50,                    // GLSL tessellation control (LW_gpu_shader5 + ARB_tessellation_shader + #version 150)
    TCS400,                     // GLSL tessellation control (GLSL 4.00)
    TCS410,                     // GLSL tessellation control (GLSL 4.10)
    TCS420,                     // GLSL tessellation control (GLSL 4.20)
    TCS430,                     // GLSL tessellation control (GLSL 4.30)
    TCS440,                     // GLSL tessellation control (GLSL 4.40)
    TCS450,                     // GLSL tessellation control (GLSL 4.50)
    TCSesext31,                 // GLSL tessellation control (EXT_tessellation_shader)
    TCSesoes31,                 // GLSL tessellation control (OES_tessellation_shader)
    TCSes32,                    // GLSL tessellation control (tessellation_shader)

    TESlw50,                    // GLSL tessellation evaluation (LW_gpu_shader5 + ARB_tessellation_shader + #version 150)
    TES400,                     // GLSL tessellation evaluation (GLSL 4.00)
    TES410,                     // GLSL tessellation evaluation (GLSL 4.10)
    TES420,                     // GLSL tessellation evaluation (GLSL 4.20)
    TES430,                     // GLSL tessellation evaluation (GLSL 4.30)
    TES440,                     // GLSL tessellation evaluation (GLSL 4.40)
    TES450,                     // GLSL tessellation evaluation (GLSL 4.50)
    TESesext31,                 // GLSL tessellation evaluation (EXT_tessellation_shader)
    TESesoes31,                 // GLSL tessellation evaluation (OES_tessellation_shader)
    TESes32,                    // GLSL tessellation evaluation (tessellation_shader)

    CS430,                      // GLSL compute (GLSL 4.30)
    CS440,                      // GLSL compute (GLSL 4.40)
    CS450,                      // GLSL compute (GLSL 4.50)
    CSes31,                     // GLSL compute (#version 310 es)
    CSes32,                     // GLSL compute (#version 320 es)

    LWProgramTypeFirstGLSLcore = VScore,
    LWProgramTypeLastGLSLcore = CSes32,

    LWProgramTypeFirstGLSL = LWProgramTypeFirstGLSLarbso,
    LWProgramTypeLastGLSL = LWProgramTypeLastGLSLcore,

    LWProgramTypeEnumCount = LWProgramTypeLastGLSL + 1,

} LWProgramType;

typedef enum LWProgramClass {

    LWProgramClassAssembly,         // Assembly using ARB extension APIs and program types
    LWProgramClassAssemblyLW,       // Assembly using LW extension APIs and program types
    LWProgramClassGLSLCore,         // GLSL using OpenGL 2.0+ core APIs
    LWProgramClassGLSLES,           // GLSL using OpenGL ES APIs and language versions
    LWProgramClassGLSLExt,          // GLSL using ARB_shader_objects APIs
    LWProgramClassGLSLCore_NonGL,   // Desktop GLSL language using non-OpenGL APIs
    LWProgramClassGLSLES_NonGL,     // GLSL ES language using non-OpenGL APIs
    LWProgramClassUnknown,

} LWProgramClass;


#endif // #ifndef __SHADERS_H__
