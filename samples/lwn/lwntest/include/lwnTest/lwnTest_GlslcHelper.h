/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/
#ifndef __lwnTest_GlslcHelper_h__
#define __lwnTest_GlslcHelper_h__

#include "lwnUtil/lwnUtil_Interface.h"

#include "lwn/lwn.h"
#include "lwnTest/lwnTest_Shader.h"
#include "lwnUtil/lwnUtil_GlslcHelper.h"

#if defined(SPIRV_ENABLED)
#include "shaderc/shaderc.h"
#endif

// Global GLSLC library helper object.
extern lwnUtil::GLSLCLibraryHelper * g_glslcLibraryHelper;
// Global DXC library helper object.
extern lwnUtil::DXCLibraryHelper * g_dxcLibraryHelper;

void glslcLoggingFunction(const char *format, va_list arg);

// Writes out the GLSLC cache binary to disk, which can subsequently be used with an LWNtest exelwtion with the
// -lwnGlslcInputFile option.
int WriteGlslcOutput();

namespace lwnTest {

// LWNTEST version of GLSLCHelper which includes some additional helper methods specific to lwntest tests.
class GLSLCHelper : public lwnUtil::GLSLCHelper {
    void init(LWNdevice * device, LWNsizeiptr maxGPUMemory);
public:
#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_C
    GLSLCHelper(LWNdevice * device, LWNsizeiptr maxGPUMemory, lwnUtil::GLSLCLibraryHelper * libraryHelper,
        lwnUtil::GLSLCHelperCache * helperCache, lwnUtil::DXCLibraryHelper * dxcLibraryHelper = NULL) :
        lwnUtil::GLSLCHelper(device, maxGPUMemory, libraryHelper, helperCache, dxcLibraryHelper)
    {
        init(device, maxGPUMemory);
    }
#else
    GLSLCHelper(lwn::Device * device, LWNsizeiptr maxGPUMemory, lwnUtil::GLSLCLibraryHelper * libraryHelper,
        lwnUtil::GLSLCHelperCache * helperCache, lwnUtil::DXCLibraryHelper * dxcLibraryHelper = NULL) :
        lwnUtil::GLSLCHelper(reinterpret_cast<LWNdevice *>(device), maxGPUMemory, libraryHelper, helperCache, dxcLibraryHelper)
    {
        LWNdevice *cdevice = reinterpret_cast<LWNdevice *>(device);
        init(cdevice, maxGPUMemory);
    }
#endif

#if defined(SPIRV_ENABLED)

    // Compile the input shaders in <glslSources> to SPIR-V binary modules.  The outputs <spirvBins>
    // correspond to the output SPIR-V module for each GLSL entry in <glslSources>.  <spvParams> will
    // have the sizes for each module filled in.
    // The function will return true upon successful compilation, with <spirvBins>
    // containing the output binaries and <spvParams>'s "sizes" fields filled in.
    // The function will return false upon any glslang compilation error.
    // This function will also increment the global lwntest lwnGlslcSpirvErrorCount counter upon failure.
    bool CompileGlslToSpirv(LWNuint nShaders, const LWNshaderStage * stages, const char ** glslSources,
                            char ** spirvBins, uint32_t * spirvBinSizes);

    // A helper class for managing the interface between the shaderc/glslang compiler and the lwnTest::GLSLCHelper.
    class SpirvCompiler {
    public:
        SpirvCompiler() : m_shadercCompiler(NULL), m_shadercOptionsGL(NULL) {}

        // Call once before compilation to initialize the SpirvCompiler object.
        // A SpirvCompiler object can only be used in a single thread.
        void SpirvInitialize();

        // Call once when finished with the SpirvCompiler to clean shader/glslang.
        void SpirvFinalize();

        // Compile GLSL shader source to SPIR-V binaries.  This requires that SpirvInitialize be
        // called prior to this function call, otherwise the function will return false.
        bool CompileGlslToSpirv(LWNuint nShaders, const LWNshaderStage * stages, const char ** glslStrings,
            char ** spirvBins, uint32_t * spirvBinSizes);

    private:
        shaderc_compiler_t m_shadercCompiler;
        shaderc_compile_options_t m_shadercOptionsGL;

        char * SpirvCompile(const char* glslSourceStr, const char* fileName, int lwrShaderIndex,
                            LWNshaderStage shaderType, int keepFile, uint32_t * spirvSourceSize);
    };

    SpirvCompiler m_spirvCompiler;

    // Whether the last compile was a SPIR-V compile or not.  This is useful for some
    // tests which need to handle the SPIR-V reflection information differently
    // from GLSL reflection information.
    LWNboolean m_wasLastCompileSpirv;
#endif

    LWNboolean WasLastCompileSpirv();

    // Initialize a program object from a collection of <nShaders> shader objects
    // in <shaders>.  We use temporary arrays to hold the shader source code
    // pointers and stage enums.
    LWNboolean CompileAndSetShaders(LWNprogram *program, LWNuint nShaders, Shader *shaders);

    // Utility functions to initialize a program object from individual shader inputs.
    // All functions eventually call into the base class GLSLLwtil's CompileAndSetShaders method.
    LWNboolean CompileAndSetShaders(LWNprogram *, Shader s1, Shader s2 = Shader(), Shader s3 = Shader(), Shader s4 = Shader(), Shader s5 = Shader());

    // Utility functions to perform only shader compilation without interacting with LWN.  All functions eventually call into
    // the base class GLSLLwtil's CompileShaders method.
    LWNboolean CompileShaders(Shader s1, Shader s2 = Shader(), Shader s3 = Shader(), Shader s4 = Shader(), Shader s5 = Shader());
    LWNboolean CompileShaders(LWNuint nShaders, Shader *shaders);
    LWNboolean CompileShaders(const LWNshaderStage * stages, LWNuint count, const char ** shaderSources);

    // Sets global GLSLC options from lwntest.
    void SetGlobalOptions();

    virtual ~GLSLCHelper();

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    //
    // Methods to provide a native C++ interface to the GLSLCHelper class,
    // using reinterpret_cast to colwert between C and C++ object types.
    //
    LWNboolean CompileAndSetShaders(lwn::Program *program, LWNuint nShaders, Shader *shaders)
    {
        LWNprogram *cprogram = reinterpret_cast<LWNprogram *>(program);
        return CompileAndSetShaders(cprogram, nShaders, shaders);
    }
    LWNboolean CompileAndSetShaders(lwn::Program *program, Shader s1, Shader s2 = Shader(), Shader s3 = Shader(), Shader s4 = Shader(), Shader s5 = Shader())
    {
        LWNprogram *cprogram = reinterpret_cast<LWNprogram *>(program);
        return CompileAndSetShaders(cprogram, s1, s2, s3, s4, s5);
    }
    LWNboolean CompileShaders(const lwn::ShaderStage *stages, LWNuint count, const char ** shaderSources)
    {
        const LWNshaderStage *cstages = reinterpret_cast<const LWNshaderStage *>(stages);
        return CompileShaders(cstages, count, shaderSources);
    }
#endif // #if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP

private:

    // Updates global flags for cache misses/compatibility issues.  Used for lwntest reporting.
    void UpdateCacheMismatchCount(LWNboolean compileSuccess);
};

// Utility function initialize the GLSLC helper cache with data from disk.
bool LoadAndSetBinaryCacheFromFile(lwnUtil::GLSLCHelperCache * helperCache, const char *filename);

} // namespace lwnTest

#endif // #ifndef __lwnTest_GlslcHelper_h__
