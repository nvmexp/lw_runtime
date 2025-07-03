/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/
#ifndef __lwnTest_Shader_h__
#define __lwnTest_Shader_h__

#include "lwn/lwn.h"

#define CPPSHADERS_NON_GL_API
#include "cppshaders.h"                 // C++ shader library with OpenGL support disabled

namespace lwnTest {

//////////////////////////////////////////////////////////////////////////
//
//                      C++ SHADER CLASS SUPPORT
//
// We can use the C++ shader classes from cppshaders.h for LWN shaders.
// However, unlike OpenGL, we only use these objects to build GLSL source
// strings.  Assembly isn't supported, LWN doesn't have a notion of "shader
// objects", and we don't use the lwProgram and lwProgramPipeline classes from
// cppshaders.h to build program objects or bind programs.
//

class Shader : public lwShader {
public:
    Shader() {}
    Shader(lwShader s) : lwShader(s) {}

    // Determine the LWN shader stage enum for the shader object.
    LWNshaderStage getStage() const
    {
        switch (targetStage()) {
        case VERTEX_STAGE:
            return LWN_SHADER_STAGE_VERTEX;
        case FRAGMENT_STAGE:
            return LWN_SHADER_STAGE_FRAGMENT;
        case GEOMETRY_STAGE:
            return LWN_SHADER_STAGE_GEOMETRY;
        case TESS_CONTROL_STAGE:
            return LWN_SHADER_STAGE_TESS_CONTROL;
        case TESS_EVALUATION_STAGE:
            return LWN_SHADER_STAGE_TESS_EVALUATION;
        case COMPUTE_STAGE:
            return LWN_SHADER_STAGE_COMPUTE;
        default:
            assert(0);
            return LWN_SHADER_STAGE_VERTEX;
        }
    }
};

typedef lwGLSLVertexShader_NonGL            VertexShader;
typedef lwGLSLTessControlShader_NonGL       TessControlShader;
typedef lwGLSLTessEvaluationShader_NonGL    TessEvaluationShader;
typedef lwGLSLGeometryShader_NonGL          GeometryShader;
typedef lwGLSLFragmentShader_NonGL          FragmentShader;
typedef lwGLSLComputeShader_NonGL           ComputeShader;
typedef lwESVertexShader_NonGL              ESVertexShader;
typedef lwESTessControlShader_NonGL         ESTessControlShader;
typedef lwESTessEvaluationShader_NonGL      ESTessEvaluationShader;
typedef lwESGeometryShader_NonGL            ESGeometryShader;
typedef lwESFragmentShader_NonGL            ESFragmentShader;
typedef lwESComputeShader_NonGL             ESComputeShader;

};

#endif // #ifndef __lwnTest_Shader_h__
