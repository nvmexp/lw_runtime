/*
* Copyright (c) 2015, Lwpu Corporation.  All rights reserved.
*
* THE INFORMATION CONTAINED HEREIN IS PROPRIETARY AND CONFIDENTIAL TO
* LWPU, CORPORATION.  USE, REPRODUCTION OR DISCLOSURE TO ANY THIRD PARTY
* IS SUBJECT TO WRITTEN PRE-APPROVAL BY LWPU, CORPORATION.
*
*
*/

#include "lwntest_c.h"
#include "lwnTest/lwnTest_GlslcHelper.h"

#include "cmdline.h"
#include "lwn_utils.h"

// GLSLC object to assist with loading the GLSLC DLL and the entry points.
lwnUtil::GLSLCLibraryHelper *g_glslcLibraryHelper = NULL;
// DXC object to assist with loading the DXC DLL and the entry points.
lwnUtil::DXCLibraryHelper *g_dxcLibraryHelper = NULL;

// A function that is passed as a pointer to the GLSLClogger utility class in both
// GLSLCHelper and GLSLCLibraryHelper during initialization to be used during any
// GLSLC utility debug logging.
void glslcLoggingFunction(const char *format, va_list arg) {
    vprintf(format, arg);
}

// 50 MB size limit for GLSLC cached binaries for now.
#define MAX_GLSLC_CACHE_FILE_SIZE 1024*1024*50
// Writes out the GLSLC cache binary to disk, which can subsequently be used with an LWNtest exelwtion with the
// -lwnGlslcInputFile option.
// Use C linkage here since this is used in testloop.c
int WriteGlslcOutput ()
{
    if (lwnGlslcOutputFile) {
        size_t size = 0;
        void * binary = g_glslcHelperCache->CreateSerializedBinary(&size);
        assert(size <= MAX_GLSLC_CACHE_FILE_SIZE);

        FILE * binaryFile = fopen(lwnGlslcOutputFile, "wb");
        if (!binaryFile) {
            printf("Could not open GLSLC binary cache file for writing: %s\n", lwnGlslcOutputFile);
            return 0;
        }


        fwrite(binary, size, 1, binaryFile);
        fclose(binaryFile);

        g_glslcHelperCache->DestroySerializedBinary(binary);
        return 1;
    }
    return 0;
}

bool lwnTest::LoadAndSetBinaryCacheFromFile(lwnUtil::GLSLCHelperCache * helperCache, const char *filename)
{
    assert(helperCache);
    void *binaryData = NULL;
    FILE *binaryFile = fopen(filename, "rb");

    if (!binaryFile) {
        assert(!"Could not open cache file!");
        return false;
    }

    // Gets the file size
    fseek(binaryFile, 0L, SEEK_END);
    size_t size = ftell(binaryFile);
    fseek(binaryFile, 0L, SEEK_SET);

    if (size > 0) {
        binaryData = calloc(1, size);
    }

    size_t elemRead = fread(binaryData, 1, size, binaryFile);

    fclose(binaryFile);

    bool success = false;

    if (elemRead != size) {
        assert(!"Elements read from cache file don't match expected file size!");
    } else {
        success = (helperCache->SetFromSerializedBinary((const char *)binaryData, elemRead) != NULL);
    }

    // SetFromSerializedBinary makes a copy of the data.
    free(binaryData);

    return success;
}

namespace lwnTest {

GLSLCHelper::~GLSLCHelper()
{
#if defined(SPIRV_ENABLED)
    if (useSpirv) {
        m_spirvCompiler.SpirvFinalize();
    }
#endif
}

void GLSLCHelper::init(LWNdevice * device, LWNsizeiptr maxGPUMemory)
{
    // Enable logging in the GLSLCHelper class by default.
    GLSLCHelper::GetLogger()->SetEnable(true);
    GLSLCHelper::GetLogger()->SetLoggerFunction(glslcLoggingFunction);

    // If -lwnGlslcInputFile is requested, make sure to disable writing to the cache
    // since we only want to read the precompiled bits but not add to them.
    // If -lwnGlslcOutputFile is requested, make sure to disable reading from the cache
    // since we explitly want all tests to trigger a compile and not read from the
    // cache we are building.
    GLSLCHelper::SetAllowCacheWrite(false);
    GLSLCHelper::SetAllowCacheRead(false);

    if (lwnGlslcInputFile) {
        GLSLCHelper::SetAllowCacheRead(LWN_TRUE);
    } else if (lwnGlslcOutputFile) {
        GLSLCHelper::SetAllowCacheWrite(LWN_TRUE);
    }

    OverrideGlslangOptions(glslang, glslangFallbackOnError, glslangFallbackOnAbsolute);

#if defined(SPIRV_ENABLED)
    if (useSpirv) {
        m_spirvCompiler.SpirvInitialize();
    }

    m_wasLastCompileSpirv = LWN_FALSE;
#endif
}


LWNboolean GLSLCHelper::CompileShaders(LWNuint nShaders, Shader *shaders)
{
    const char *c_strings[GLSLC_NUM_SHADER_STAGES] = { NULL };
    lwString strings[GLSLC_NUM_SHADER_STAGES];
    LWNshaderStage stages[GLSLC_NUM_SHADER_STAGES];

    for (LWNuint i = 0; i < nShaders; i++) {
        strings[i] = shaders[i].source();
        c_strings[i] = strings[i].c_str();
        stages[i] = shaders[i].getStage();
    }

    return GLSLCHelper::CompileShaders((LWNshaderStage *)stages, nShaders, c_strings);
}

LWNboolean GLSLCHelper::CompileShaders(Shader s1,
                                       Shader s2 /*= Shader()*/,
                                       Shader s3 /*= Shader()*/,
                                       Shader s4 /*= Shader()*/,
                                       Shader s5 /*= Shader()*/)
{
    Shader shaders[5];
    LWNuint nShaders = 0;
    if (s1.exists()) { shaders[nShaders++] = s1; }
    if (s2.exists()) { shaders[nShaders++] = s2; }
    if (s3.exists()) { shaders[nShaders++] = s3; }
    if (s4.exists()) { shaders[nShaders++] = s4; }
    if (s5.exists()) { shaders[nShaders++] = s5; }

    return CompileShaders(nShaders, shaders);
}

void lwnTest::GLSLCHelper::SetGlobalOptions()
{
    assert(lwnGlslcDebugLevel >= -1);
    assert(lwnGlslcDebugLevel <= 2);

    if (lwnGlslcDebugLevel != -1) {
        // A value of <n>, where <n> >= 0, means "GLSLC_DEBUG_LEVEL_G<n>".
        // A value of <n> == -1 maps to "GLSLC_DEBUG_LEVEL_NONE", which is also the default value.
        GLSLCHelper::SetDebugLevel(GLSLCdebugInfoLevelEnum(lwnGlslcDebugLevel + 1));
    }

    assert((lwnGlslcOptLevel == 0) || (lwnGlslcOptLevel == 1));
    GLSLCHelper::SetOptLevel(GLSLCoptLevelEnum(lwnGlslcOptLevel));
}

// Initialize a program object from a collection of <nShaders> shader objects
// in <shaders>.  We use temporary arrays to hold the shader source code
// pointers and stage enums.
LWNboolean lwnTest::GLSLCHelper::CompileAndSetShaders(LWNprogram *program, LWNuint nShaders, Shader *shaders)
{
    const char *c_strings[GLSLC_NUM_SHADER_STAGES] = { NULL };
    lwString strings[GLSLC_NUM_SHADER_STAGES];
    LWNshaderStage stages[GLSLC_NUM_SHADER_STAGES];
    char *spirvBins[GLSLC_NUM_SHADER_STAGES] = { NULL };

    for (LWNuint i = 0; i < nShaders; i++) {
        strings[i] = shaders[i].source();
        c_strings[i] = strings[i].c_str();
        stages[i] = shaders[i].getStage();
    }

    SetGlobalOptions();

    SpirvParams spvParams;
    bool compileAsSpirv = false;

#if defined(SPIRV_ENABLED)
    if (useSpirv) {
        compileAsSpirv = CompileGlslToSpirv(nShaders, stages, c_strings, spirvBins, spvParams.sizes);
    }

    m_wasLastCompileSpirv = LWNboolean(compileAsSpirv);

    if (compileAsSpirv && UsesCache()) {
        // For SPIR-V shaders, we want to cache the compiled binary using the
        // original source shader string and input options (not the SPIR-V
        // that's generated with glslang).  Requiring SPIR-V generation from
        // glslang isn't really required just to compute the cache key since
        // there is a 1:1 correspondence between GLSL source and SPIR-V
        // binaries.  Additionally this lets consumers of the cache not need
        // to recompile to SPIR-V just to get the cache key for pulling the
        // precompiled binary.
        GLSLCinput overrideInput;
        GLSLCoptions overrideOptions;

        InitializeGlslcInputAndOptions(&overrideInput, &overrideOptions, stages,
                nShaders, c_strings, NULL);
        GLSLCHelperCacheKey overrideKey =
            GetHelperCache()->HashShaders(&overrideInput, &overrideOptions);
        SetOverrideCacheKey(&overrideKey);
    }
#endif

    LWNboolean compileSuccess = lwnUtil::GLSLCHelper::CompileAndSetShaders(
        program, (LWNshaderStage *)stages, nShaders,
        (compileAsSpirv ? (const char **)spirvBins : c_strings),
        (compileAsSpirv ? &spvParams : NULL));

    if (compileAsSpirv) {
        if (UsesCache()) {
            SetOverrideCacheKey(NULL);
        }
        // Free the SPIR-V binaries allocated inside SpirvCompiler::CompileGlslToSpirv.
        for (int i = 0; i < GLSLC_NUM_SHADER_STAGES; ++i) {
            __LWOG_FREE(spirvBins[i]);
            spirvBins[i] = NULL;
        }
    }

    UpdateCacheMismatchCount(compileSuccess);

    return compileSuccess;
}

LWNboolean GLSLCHelper::CompileShaders(const LWNshaderStage *stages, LWNuint count, const char ** shaderSources)
{
    SpirvParams spvParams;
    char *spirvBins[GLSLC_NUM_SHADER_STAGES] = { NULL };

    bool compileAsSpirv = false;

    SetGlobalOptions();

#if defined(SPIRV_ENABLED)
    if (useSpirv) {
        compileAsSpirv = CompileGlslToSpirv(count, stages, shaderSources, spirvBins, spvParams.sizes);
    }

    if (compileAsSpirv && UsesCache()) {
        // For SPIR-V shaders, we want to cache the compiled binary using the
        // original source shader string and input options (not the SPIR-V
        // that's generated with glslang).  Requiring SPIR-V generation from
        // glslang isn't really required just to compute the cache key since
        // there is a 1:1 correspondence between GLSL source and SPIR-V
        // binaries.  Additionally this lets consumers of the cache not need
        // to recompile to SPIR-V just to get the cache key for pulling the
        // precompiled binary.
        GLSLCinput overrideInput;
        GLSLCoptions overrideOptions;

        InitializeGlslcInputAndOptions(&overrideInput, &overrideOptions,
                stages, count, shaderSources, NULL);
        GLSLCHelperCacheKey overrideKey =
            GetHelperCache()->HashShaders(&overrideInput, &overrideOptions);
        SetOverrideCacheKey(&overrideKey);
    }
#endif

    LWNboolean compileSuccess = lwnUtil::GLSLCHelper::CompileShaders(stages, count,
                                                                     (compileAsSpirv ? (const char **)spirvBins : shaderSources),
                                                                     (compileAsSpirv ? &spvParams : NULL));

    UpdateCacheMismatchCount(compileSuccess);

    if (compileAsSpirv) {
        if (UsesCache()) {
            SetOverrideCacheKey(NULL);
        }

        // Free the SPIR-V binaries allocated inside SpirvCompiler::CompileGlslToSpirv.
        for (int i = 0; i < GLSLC_NUM_SHADER_STAGES; ++i) {
            __LWOG_FREE(spirvBins[i]);
            spirvBins[i] = NULL;
        }
    }

    return compileSuccess;
}

LWNboolean GLSLCHelper::CompileAndSetShaders(
                                       LWNprogram *program,
                                       Shader s1,
                                       Shader s2 /*= Shader()*/,
                                       Shader s3 /*= Shader()*/,
                                       Shader s4 /*= Shader()*/,
                                       Shader s5 /*= Shader()*/)
{
    Shader shaders[5];
    LWNuint nShaders = 0;
    if (s1.exists()) { shaders[nShaders++] = s1; }
    if (s2.exists()) { shaders[nShaders++] = s2; }
    if (s3.exists()) { shaders[nShaders++] = s3; }
    if (s4.exists()) { shaders[nShaders++] = s4; }
    if (s5.exists()) { shaders[nShaders++] = s5; }
    return CompileAndSetShaders(program, nShaders, shaders);
}

void lwnTest::GLSLCHelper::UpdateCacheMismatchCount(LWNboolean compileSuccess)
{
    // Check if there was a miss, if using glslc input.
    if (GLSLCHelper::UsesCache() && lwnGlslcInputFile) {
        if (!GLSLCHelper::LastCacheEntryHit() && compileSuccess) {
            // We had a cache miss and a successful compile afterwards.
            lwnGlslcBinaryMissCount++;
        } else {
            if (!GLSLCHelper::IsCacheAPIVersionCompatible()) {
                lwnGlslcBinaryCacheApiMismatch++;
            }
            if (!GLSLCHelper::IsCacheGPUVersionCompatible()) {
                lwnGlslcBinaryCacheGpuMismatch++;
            }
        }
    }
}

LWNboolean lwnTest::GLSLCHelper::WasLastCompileSpirv()
{
#if defined(SPIRV_ENABLED)
    return m_wasLastCompileSpirv;
#else
    return LWN_FALSE;
#endif
}

#if defined(SPIRV_ENABLED)

bool GLSLCHelper::CompileGlslToSpirv(LWNuint nShaders, const LWNshaderStage * stages, const char ** glslSources,
                                     char ** spirvBins, uint32_t * spvSizes)
{
    bool spirvSuccess = false;
    spirvSuccess = m_spirvCompiler.CompileGlslToSpirv(nShaders, stages, glslSources, spirvBins, spvSizes);

    if (!spirvSuccess) {
        // Failure to compile indicates we should fallback to GLSL compilation mode.
        // We also set the global error count so lwntest can log this test as having such an
        // error.
        lwnGlslcSpirvErrorCount++;

        if (logSpirvErrors) {
            printf("Couldn't compile GLSL to SPIR-V using glslang, falling back to GLSL compilation...\n");
        }
    }

    return spirvSuccess;
}

// Initialize Shaderc library or for spir-v compilation; 
void GLSLCHelper::SpirvCompiler::SpirvInitialize()
{
    // check if we are already initialized, and if so, we reset the object
    if (m_shadercCompiler != NULL ||
        m_shadercOptionsGL != NULL) {
        SpirvFinalize();
    }

    m_shadercCompiler = shaderc_compiler_initialize();
    if (m_shadercCompiler == NULL) {
        printf("ERROR: Can't enable SPIR-V compilation (shaderc_compiler_initialize)\n");
        Terminate(EXIT_STATUS_NOT_NORMAL);
    }

    m_shadercOptionsGL = shaderc_compile_options_initialize();
    if (m_shadercOptionsGL == NULL) {
        printf("ERROR: Can't enable SPIR-V compilation (shaderc_compile_options_initialize)\n");
        Terminate(EXIT_STATUS_NOT_NORMAL);
    }

    shaderc_compile_options_set_target_elw(m_shadercOptionsGL, shaderc_target_elw_opengl, 0);
    shaderc_compile_options_set_auto_map_locations(m_shadercOptionsGL, true);
}

bool GLSLCHelper::SpirvCompiler::CompileGlslToSpirv(LWNuint nShaders,
                                                    const LWNshaderStage * stages,
                                                    const char ** glslStrings,
                                                    char ** spirvBins,
                                                    uint32_t * spirvBinSizes)
{
    memset(spirvBins, 0, sizeof(const char *) * GLSLC_NUM_SHADER_STAGES);
    memset(spirvBinSizes, 0, sizeof(uint32_t) * GLSLC_NUM_SHADER_STAGES);

    if (!m_shadercCompiler || !m_shadercOptionsGL) {
        assert(!"Spirv not initialized!");
        return false;
    }

    bool spirvSuccess = true;
    for (LWNuint i = 0; i < nShaders; i++) {
        spirvBins[i] = SpirvCompile(glslStrings[i], NULL, 0, stages[i], false, &spirvBinSizes[i]);

        if (!spirvBins[i] || !spirvBinSizes[i]) {
            spirvSuccess = false;
        }
    }

    if (!spirvSuccess) {
        for (unsigned int i = 0; i < nShaders; ++i) {
            if (spirvBins[i]) {
                free(spirvBins[i]);
                spirvBins[i] = NULL;
                spirvBinSizes[i] = 0;
            }
        }
    }

    return spirvSuccess;
}


// De-initialize everything for SPIR-V compilation
void GLSLCHelper::SpirvCompiler::SpirvFinalize()
{
    if (m_shadercOptionsGL != NULL) {
        shaderc_compile_options_release(m_shadercOptionsGL);
        m_shadercOptionsGL = NULL;
    }

    if (m_shadercCompiler != NULL) {
        shaderc_compiler_release(m_shadercCompiler);
        m_shadercCompiler = NULL;
    }
}

char* GLSLCHelper::SpirvCompiler::SpirvCompile(const char* glslSourceStr, const char* fileName, int lwrShaderIndex,
                                               LWNshaderStage shaderStage, int keepFile, uint32_t* spirvBinSize)
{
    assert(m_shadercCompiler);
    assert(m_shadercOptionsGL);

    *spirvBinSize = 0;

    shaderc_shader_kind shaderKind;

    switch (shaderStage) {
    case LWN_SHADER_STAGE_VERTEX:
        shaderKind = shaderc_glsl_vertex_shader;
        break;
    case LWN_SHADER_STAGE_FRAGMENT:
        shaderKind = shaderc_glsl_fragment_shader;
        break;
    case LWN_SHADER_STAGE_GEOMETRY:
        shaderKind = shaderc_glsl_geometry_shader;
        break;
    case LWN_SHADER_STAGE_TESS_CONTROL:
        shaderKind = shaderc_glsl_tess_control_shader;
        break;
    case LWN_SHADER_STAGE_TESS_EVALUATION:
        shaderKind = shaderc_glsl_tess_evaluation_shader;
        break;
    case LWN_SHADER_STAGE_COMPUTE:
        shaderKind = shaderc_glsl_compute_shader;
        break;
    default:
        assert(!"Unsupported shader type");
        return NULL;
    }

    shaderc_compilation_result_t spirvModule =
        shaderc_compile_into_spv(m_shadercCompiler, glslSourceStr, strlen(glslSourceStr),
                                 shaderKind, "test-shader", "main", m_shadercOptionsGL);

    if (!spirvModule) {
        printf("ERROR: shaderc_compile_into_spv encountered a fatal error!\n");
        return NULL;
    }

    if (shaderc_result_get_compilation_status(spirvModule) != shaderc_compilation_status_success) {
        if (logSpirvErrors) {
            unsigned int prodVersion = 0;
            unsigned int prodRevision = 0;
            shaderc_get_spv_version(&prodVersion, &prodRevision);
            printf("\n============================================\n");
            printf("Shaderc SPIR-V version/revision: %d/%d\n", prodVersion, prodRevision);
            printf("--------------------------------------------\n");
            printf("%s\n", shaderc_result_get_error_message(spirvModule));
            printf("--------------------------------------------\n");
            printf("%s\n", glslSourceStr);
            printf("============================================\n");
        }

        shaderc_result_release(spirvModule);
        return NULL;
    }

    size_t spirvLen = shaderc_result_get_length(spirvModule);
    char *result = (char *)__LWOG_MALLOC(spirvLen);
    if (!result) {
        if (logSpirvErrors) {
            printf("ERROR: out of memory at %s:%d\n", __FILE__, __LINE__);
        }

        shaderc_result_release(spirvModule);
        return NULL;
    }

    memcpy(result, shaderc_result_get_bytes(spirvModule), spirvLen);
    shaderc_result_release(spirvModule);
    *spirvBinSize = (uint32_t)spirvLen;
    return result;
}
#endif // defined(SPIRV_ENABLED)
} // namespace lwnTest
