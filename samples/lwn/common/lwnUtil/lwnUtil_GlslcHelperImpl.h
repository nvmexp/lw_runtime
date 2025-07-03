/*
 * Copyright (c) 2015 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __lwnUtil_GlslcHelperImpl_h__
#define __lwnUtil_GlslcHelperImpl_h__

#include "stddef.h"
#include "stdio.h"
#include "assert.h"

#include "lwnUtil/lwnUtil_AlignedStorage.h"
#include "lwnUtil/lwnUtil_GlslcHelper.h"

extern LWNbufferBuilder * lwnDeviceCreateBufferBuilder(LWNdevice *device);

namespace lwnUtil {

void GLSLCLogger::Log(const char *format, ...)
{
    if (
#if defined(GLSLC_LOGGER_FORCE_ENABLE)
        GLSLC_LOGGER_FORCE_ENABLE
#else
        m_enabled
#endif
        ) {
        if (m_loggerFunction == NULL) {
            assert(!"GLSLCLogger enabled but no logging function has been set!");
        } else {
            va_list args;
            va_start(args, format);

#if defined(GLSLC_LOGGER_FORCE_PRINTF) && (GLSLC_LOGGER_FORCE_PRINTF == 1)
            vprintf(format, args);
#else
            m_loggerFunction(format, args);
#endif
            va_end(args);
        }
    }
}

void GLSLCLogger::SetEnable (bool enable)
{
    m_enabled = enable;
}

bool GLSLCLogger::IsEnabled ()
{
    return m_enabled;
}

void GLSLCLogger::SetLoggerFunction(GLSLCLoggerFunction funcPtr)
{
    m_loggerFunction = funcPtr;
}


LWNboolean GLSLCHelper::UsesCache()
{
    return m_cacheParameters.m_dataCache &&
        (m_cacheParameters.m_allowCacheRead || m_cacheParameters.m_allowCacheWrite);
}

DXCLibraryHelper::DXCLibraryHelper() :
#ifdef DXC_ENABLED
    m_pCompiler(NULL), m_pLibrary(NULL),
#endif // DXC_ENABLED
    initialized(false)
{
#ifdef DXC_ENABLED
    if (SUCCEEDED(dllHelper.Initialize()) &&
        SUCCEEDED(dllHelper.CreateInstance(CLSID_DxcCompiler, &m_pCompiler)) &&
        SUCCEEDED(dllHelper.CreateInstance(CLSID_DxcLibrary, &m_pLibrary))) {
        initialized = true;
    }
#endif // DXC_ENABLED
}

DXCLibraryHelper::~DXCLibraryHelper()
{
#ifdef DXC_ENABLED
    if (m_pCompiler) {
        m_pCompiler->Release();
        m_pCompiler = NULL;
    }
    m_pLibrary = NULL;
    initialized = false;
#endif // DXC_ENABLED
}

#ifdef DXC_ENABLED
LPCWSTR DXCLibraryHelper::GetHLSLShaderStage(unsigned int shaderStage, LPCWSTR & shaderKind)
{
    switch (shaderStage) {
    case LWN_SHADER_STAGE_VERTEX:
        shaderKind = L"Vertex Shader";
        return L"vs_6_3";
    case LWN_SHADER_STAGE_FRAGMENT:
        shaderKind = L"Fragment Shader";
        return L"ps_6_3";
    case LWN_SHADER_STAGE_GEOMETRY:
        shaderKind = L"Geometry Shader";
        return L"gs_6_3";
    case LWN_SHADER_STAGE_TESS_CONTROL:
        shaderKind = L"Tess Control Shader";
        return L"hs_6_3";
    case LWN_SHADER_STAGE_TESS_EVALUATION:
        shaderKind = L"Tess Evaluation Shader";
        return L"ds_6_3";
    case LWN_SHADER_STAGE_COMPUTE:
        shaderKind = L"Compute Shader";
        return L"cs_6_3";
    default:
        return L"";
    }
}

char* DXCLibraryHelper::DxcCompile(const char* hlslSourceStr, int lwrShaderIndex,
    LWNshaderStage shaderType, uint32_t * spirvBinSize, GLSLCLogger * logger)
{
    *spirvBinSize = 0;

    // TODO: pass the entry name from the tests if it is not main.
    const char* entryName = "main";

    IDxcBlobEncoding *pSource;
    IDxcOperationResult *pResult;
    HRESULT status;
    LPCWSTR compileArgs[] = { L"-spirv",  L"-fspv-target-elw=vulkan1.2" };

    wchar_t *wEntryName = nullptr;
    LPCWSTR shaderKind = L"Compile as library";
    LPCWSTR targetProfile = entryName == nullptr ? L"lib_6_3" : GetHLSLShaderStage(shaderType, shaderKind);
    // The compileAsLibrary functionality is not tested.
    bool compileAsLibrary = entryName == nullptr;

    if (!compileAsLibrary) {
        wEntryName = (wchar_t *)__LWOG_MALLOC(sizeof(wchar_t) * (strlen(entryName) + 1));
        mbstowcs(wEntryName, entryName, (strlen(entryName) + 1));
    }

    m_pLibrary->CreateBlobWithEncodingFromPinned(hlslSourceStr, (uint32_t)(strlen(hlslSourceStr) + 1),
        0, &pSource);

    m_pCompiler->Compile(
        pSource,                                // Program text
        shaderKind,                             // Debug label for error messages
        compileAsLibrary ? L"" : wEntryName,    // Entry point function
        targetProfile,                          // Target profile
        compileArgs,                            // Compilation arguments
        _countof(compileArgs),                  // Number of compilation arguments
        nullptr, 0,                             // Name/value defines and their count
        nullptr,                                // Handler for #include directives
        &pResult);

    char *spirvBin = NULL;

    pResult->GetStatus(&status);
    if (SUCCEEDED(status)) {
        IDxcBlob *compileResult;
        pResult->GetResult(&compileResult);

        int bufSize = compileResult->GetBufferSize();
        // This will be freed after done with the SetShaders.
        spirvBin = (char *)__LWOG_MALLOC(bufSize);
        if (!spirvBin) {
            logger->Log("ERROR: out of memory at %s:%d\n", __FILE__, __LINE__);
            compileResult->Release();
            return NULL;
        }
        memset(spirvBin, 0, bufSize);
        memcpy(spirvBin, compileResult->GetBufferPointer(), bufSize);
        *spirvBinSize = bufSize;

        compileResult->Release();
    } else {
        IDxcBlobEncoding *errorLog;
        pResult->GetErrorBuffer(&errorLog);
        logger->Log("\nHLSL Compilation Failed : %s", (char *)errorLog->GetBufferPointer());
    }

    if (!compileAsLibrary) {
        __LWOG_FREE(wEntryName);
    }
    pResult->Release();

    return spirvBin;
}
#endif // DXC_ENABLED

GLSLCLibraryHelper::GLSLCLibraryHelper()
{
#if defined(_WIN32) && defined(GLSLC_LIB_DYNAMIC_LOADING)
    glslcHMod = NULL;
#endif

    // Initialize all function pointers to NULL.
    glslcCompile = NULL;
    glslcInitialize = NULL;
    glslcFinalize = NULL;
    glslcGetVersion = NULL;
    glslcCompilePreSpecialized = NULL;
    glslcCompileSpecialized = NULL;
    glslcGetDefaultOptions = NULL;
    glslcCompileSpecializedMT = NULL;
    glslcFreeSpecializedResultsMT = NULL;
    glslcCompareControlSections = NULL;
    glslcGetDebugDataHash = NULL;
    glslcSetDebugDataHash = NULL;

    memset(&libraryVersion, 0, sizeof(GLSLCversion));
}

GLSLCLibraryHelper::~GLSLCLibraryHelper()
{
    if (IsLoaded()) {
#if defined(_WIN32) && defined(GLSLC_LIB_DYNAMIC_LOADING)
        FreeLibrary(glslcHMod);
#endif
        UnloadDLLFunctions();
    }
}

void GLSLCLibraryHelper::UnloadDLLFunctions()
{
    this->glslcCompile = NULL;
    this->glslcInitialize = NULL;
    this->glslcFinalize = NULL;
    this->glslcGetVersion = NULL;
    this->glslcCompilePreSpecialized = NULL;
    this->glslcCompileSpecialized = NULL;
    this->glslcGetDefaultOptions = NULL;
    this->glslcCompileSpecializedMT = NULL;
    this->glslcFreeSpecializedResultsMT = NULL;
    this->glslcCompareControlSections = NULL;
    this->glslcGetDebugDataHash = NULL;
    this->glslcSetDebugDataHash = NULL;
}

bool GLSLCLibraryHelper::LoadDLL( const char * DLLFileName )
{
#if defined(_WIN32) && defined(GLSLC_LIB_DYNAMIC_LOADING)
    HMODULE hMod = LoadLibrary(DLLFileName);

    if (!hMod) {
        logger.Log("GLSLCLibraryHelper: Can not load the GLSLC library named %s\n", DLLFileName);
        return false;
    }

    glslcCompile = (GLSLCCOMPILEFUNCTION) GetProcAddress(hMod, "glslcCompile");
    glslcInitialize = (GLSLCINITIALIZEFUNCTION) GetProcAddress(hMod, "glslcInitialize");
    glslcFinalize = (GLSLCFINALIZEFUNCTION) GetProcAddress(hMod, "glslcFinalize");
    glslcGetVersion = (GLSLCGETVERSIONFUNCTION) GetProcAddress(hMod, "glslcGetVersion");
    glslcCompilePreSpecialized = (GLSLCCOMPILEPRESPECIALIZEDFUNCTION) GetProcAddress(hMod, "glslcCompilePreSpecialized");
    glslcCompileSpecialized = (GLSLCCOMPILESPECIALIZEDFUNCTION) GetProcAddress(hMod, "glslcCompileSpecialized");
    glslcGetDefaultOptions = (GLSLCGETDEFAULTOPTIONSFUNCTION) GetProcAddress(hMod, "glslcGetDefaultOptions");
    glslcCompileSpecializedMT = (GLSLCCOMPILESPECIALIZEDMTFUNCTION) GetProcAddress(hMod, "glslcCompileSpecializedMT");
    glslcFreeSpecializedResultsMT = (GLSLCFREESPECIALIZEDRESULTSMTFUNCTION) GetProcAddress(hMod, "glslcFreeSpecializedResultsMT");
    glslcCompareControlSections = (GLSLCCOMPARECONTROLSECTIONSFUNCTION) GetProcAddress(hMod, "glslcCompareControlSections");
    glslcGetDebugDataHash = (GLSLCGETDEBUGDATAHASHFUNCTION) GetProcAddress(hMod, "glslcGetDebugDataHash");
    glslcSetDebugDataHash = (GLSLCSETDEBUGDATAHASHFUNCTION) GetProcAddress(hMod, "glslcSetDebugDataHash");

    if (!glslcCompile ||
        !glslcInitialize ||
        !glslcFinalize ||
        !glslcGetVersion ||
        !glslcCompilePreSpecialized ||
        !glslcCompileSpecialized ||
        !glslcGetDefaultOptions ||
        !glslcCompileSpecializedMT ||
        !glslcFreeSpecializedResultsMT ||
        !glslcCompareControlSections ||
        !glslcGetDebugDataHash ||
        !glslcSetDebugDataHash) {

        // Unload library since it was loaded but we failed to load all the needed bits.
        FreeLibrary(hMod);
        hMod = NULL;

        UnloadDLLFunctions();

        logger.Log("GLSLCLibraryHelper: Can not load entry points from the GLSLC library named %s\n", DLLFileName);
        return false;
    }
    glslcHMod = hMod;

#else
    // Linking directly to static export lib (Windows) or dynamic libs (HOS and L4T)

#if !defined(GLSLC_LIB_DYNAMIC_LOADING)
    // Obtain the GLSLC functions from a static imports library.
    this->glslcCompile = ::glslcCompile;
    this->glslcInitialize = ::glslcInitialize;
    this->glslcFinalize = ::glslcFinalize;
    this->glslcGetVersion = ::glslcGetVersion;
    this->glslcCompilePreSpecialized = ::glslcCompilePreSpecialized;
    this->glslcCompileSpecialized = ::glslcCompileSpecialized;
    this->glslcGetDefaultOptions = ::glslcGetDefaultOptions;
    this->glslcCompileSpecializedMT = ::glslcCompileSpecializedMT;
    this->glslcFreeSpecializedResultsMT = ::glslcFreeSpecializedResultsMT;
    this->glslcCompareControlSections = ::glslcCompareControlSections;
    this->glslcGetDebugDataHash = ::glslcGetDebugDataHash;
    this->glslcSetDebugDataHash = ::glslcSetDebugDataHash;
#endif

    if (!this->glslcCompile ||
        !this->glslcInitialize ||
        !this->glslcFinalize ||
        !this->glslcGetVersion ||
        !this->glslcCompilePreSpecialized ||
        !this->glslcCompileSpecialized ||
        !this->glslcGetDefaultOptions ||
        !this->glslcCompileSpecializedMT ||
        !this->glslcFreeSpecializedResultsMT ||
        !this->glslcCompareControlSections ||
        !this->glslcGetDebugDataHash ||
        !this->glslcSetDebugDataHash) {

        UnloadDLLFunctions();

        logger.Log("GLSLCLibraryHelper: Can not load entry points from the GLSLC library.\n", DLLFileName);
        return false;
    }
#endif

    libraryVersion = this->glslcGetVersion();

    if (!GLSLCLibraryHelper::GLSLCCheckAPIVersion(GLSLC_API_VERSION_MAJOR, GLSLC_API_VERSION_MINOR,
                                                  libraryVersion.apiMajor, libraryVersion.apiMinor)) {
        logger.Log("GLSLCLibrarHelper: GLSLC DLL reported API version (%d.%d) which is not compatible with the version\n"
                   "compiled into the application (%d.%d).  Please check the correct versioned lwnTool_GlslcInterface.h\n"
                   "is being used by the application.\n", libraryVersion.apiMajor, libraryVersion.apiMinor,
                   GLSLC_API_VERSION_MAJOR, GLSLC_API_VERSION_MINOR);
        UnloadDLLFunctions();

        return false;
    }

    return true;
}

bool GLSLCLibraryHelper::IsLoaded()
{
#if defined(_WIN32) && defined(GLSLC_LIB_DYNAMIC_LOADING)
    return (glslcHMod != NULL);
#else
    return glslcCompile &&
           glslcInitialize &&
           glslcFinalize &&
           glslcGetVersion &&
           glslcCompilePreSpecialized &&
           glslcCompileSpecialized &&
           glslcGetDefaultOptions &&
           glslcCompileSpecializedMT &&
           glslcFreeSpecializedResultsMT &&
           glslcCompareControlSections &&
           glslcGetDebugDataHash &&
           glslcSetDebugDataHash;
#endif
}

GLSLCversion GLSLCLibraryHelper::GetVersion()
{
    return glslcGetVersion();
}

GLSLCLogger * GLSLCLibraryHelper::GetLogger()
{
    return &logger;
}

GLSLCHelper::GLSLCHelper( LWNdevice * device, size_t maxGPUMemory, GLSLCLibraryHelper * libraryHelper,
    GLSLCHelperCache * cache, DXCLibraryHelper * dxcLibraryHelper) :
    m_allocator(device, NULL, maxGPUMemory,
                LWNmemoryPoolFlags(LWN_MEMORY_POOL_FLAGS_CPU_UNCACHED_BIT |
                                   LWN_MEMORY_POOL_FLAGS_GPU_CACHED_BIT |
                                   LWN_MEMORY_POOL_FLAGS_SHADER_CODE_BIT)),
    m_device(device),
    m_dxcLibraryHelper(dxcLibraryHelper),
    m_libraryHelper(libraryHelper),
    m_cacheParameters(cache),
    m_overrideArch(0),
    m_overrideImpl(0),
    m_overrideDoGlslangShim(0),
    m_overrideGlslangFallbackOnError(0),
    m_overrideGlslangFallbackOnAbsolute(0)
{
    memset(&m_compileObject, 0, sizeof(GLSLCcompileObject));
    m_poolSize = maxGPUMemory;

    // Initialize a default set of options for the user options.
    if (m_libraryHelper->IsLoaded()) {  
        m_userOptions = m_libraryHelper->glslcGetDefaultOptions();
    }

    // Check the version and report an error.
    GLSLCversion dllVersion = m_libraryHelper->GetVersion();

    int glslcMaxBilwersionMajor = 0;
    int glslcMinBilwersionMajor = 0;
    int glslcMaxBilwersionMinor = 0;
    int glslcMinBilwersionMinor = 0;
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_GLSLC_MAX_SUPPORTED_GPU_CODE_MAJOR_VERSION, &glslcMaxBilwersionMajor);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_GLSLC_MIN_SUPPORTED_GPU_CODE_MAJOR_VERSION, &glslcMinBilwersionMajor);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_GLSLC_MAX_SUPPORTED_GPU_CODE_MINOR_VERSION, &glslcMaxBilwersionMinor);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_GLSLC_MIN_SUPPORTED_GPU_CODE_MINOR_VERSION, &glslcMinBilwersionMinor);

    if (!GLSLCLibraryHelper::GLSLCCheckGPUCodeVersion(glslcMaxBilwersionMajor, glslcMinBilwersionMajor,
                              glslcMaxBilwersionMinor, glslcMinBilwersionMinor,
                              dllVersion.gpuCodeVersionMajor, dllVersion.gpuCodeVersionMinor)) {
        m_logger.Log("GLSLCHelper: GLSLC binary version not compatibile with this version of LWN:\n"
                     "LWN reports being able to use GLSLC binary version %d.%d through %d.%d.\n"
                     "GLSLC reported being able to produce a binary version %d.%d.\n",
                     glslcMinBilwersionMajor, glslcMinBilwersionMinor,
                     glslcMaxBilwersionMajor, glslcMaxBilwersionMinor,
                     dllVersion.gpuCodeVersionMajor, dllVersion.gpuCodeVersionMinor);
    }
}

void GLSLCHelper::SetAllowCacheRead(LWNboolean flag)
{
    if (m_cacheParameters.m_dataCache == NULL) {
        return;
    }

    m_cacheParameters.m_allowCacheRead = flag;
}

void GLSLCHelper::SetAllowCacheWrite(LWNboolean flag)
{
    if (m_cacheParameters.m_dataCache == NULL) {
        return;
    }

    m_cacheParameters.m_allowCacheWrite = flag;
}

LWNboolean GLSLCHelper::GetAllowCacheRead()
{
    if (m_cacheParameters.m_dataCache == NULL) {
        return LWN_FALSE;
    }

    return m_cacheParameters.m_allowCacheRead;
}

LWNboolean GLSLCHelper::GetAllowCacheWrite()
{
    if (m_cacheParameters.m_dataCache == NULL) {
        return LWN_FALSE;
    }

    return m_cacheParameters.m_allowCacheWrite;
}

void GLSLCHelper::ResetOptions()
{
    m_userOptions = m_libraryHelper->glslcGetDefaultOptions();

    // If there were any include paths, clear the list
    m_includePaths.clear();
}

void GLSLCHelper::Reset()
{
    ResetBuffers();
    ResetOptions();

    if (m_libraryHelper->IsLoaded()) {
        m_libraryHelper->glslcFinalize(&m_compileObject);
    }
}

const void * GLSLCHelper::FindValueInPiqMap ( ReflectionTypeEnum type, const char * name ) const
{
    PiqMapType::const_iterator groupMapIter;
    groupMapIter = m_piqMap.find(type);
    if (groupMapIter != m_piqMap.end()) {
        InterfaceMemberMapType::const_iterator valIter;
        valIter = groupMapIter->second.find(name);
        if (valIter != groupMapIter->second.end()) {
            return valIter->second;
        }
    }

    return NULL;
}


static const GLSLCHelper::ReflectionTypeEnum subroutineStages [GLSLC_NUM_SHADER_STAGES] = {
   GLSLCHelper::REFLECTION_TYPE_SUBROUTINE_VERTEX,
   GLSLCHelper::REFLECTION_TYPE_SUBROUTINE_FRAGMENT,
   GLSLCHelper::REFLECTION_TYPE_SUBROUTINE_GEOMETRY,
   GLSLCHelper::REFLECTION_TYPE_SUBROUTINE_TESS_CONTROL,
   GLSLCHelper::REFLECTION_TYPE_SUBROUTINE_TESS_EVALUATION,
   GLSLCHelper::REFLECTION_TYPE_SUBROUTINE_COMPUTE,
};

static const GLSLCHelper::ReflectionTypeEnum subroutineUniformStages[GLSLC_NUM_SHADER_STAGES] = {
   GLSLCHelper::REFLECTION_TYPE_SUBROUTINE_UNIFORM_VERTEX,
   GLSLCHelper::REFLECTION_TYPE_SUBROUTINE_UNIFORM_FRAGMENT,
   GLSLCHelper::REFLECTION_TYPE_SUBROUTINE_UNIFORM_GEOMETRY,
   GLSLCHelper::REFLECTION_TYPE_SUBROUTINE_UNIFORM_TESS_CONTROL,
   GLSLCHelper::REFLECTION_TYPE_SUBROUTINE_UNIFORM_TESS_EVALUATION,
   GLSLCHelper::REFLECTION_TYPE_SUBROUTINE_UNIFORM_COMPUTE,
};

const GLSLCsubroutineInfo * GLSLCHelper::GetSubroutineInfo(LWNshaderStage stage, const char *name) const
{
    return (const GLSLCsubroutineInfo *)FindValueInPiqMap(subroutineStages[stage], name);
}

const GLSLCsubroutineUniformInfo * GLSLCHelper::GetSubroutineUniformInfo(LWNshaderStage stage, const char *name) const
{
    return (const GLSLCsubroutineUniformInfo *)FindValueInPiqMap(subroutineUniformStages[stage], name);
}

const unsigned int * GLSLCHelper::GetCompatibleSubroutineIndices(LWNshaderStage stage, const char *name, unsigned int *numCompatibleSubroutines) const
{
    const GLSLCsubroutineUniformInfo * subroutineUniformInfo = GetSubroutineUniformInfo(stage, name);
    if (!subroutineUniformInfo) {
        *numCompatibleSubroutines = 0;
        return NULL;
    }

    return GetCompatibleSubroutineIndices(subroutineUniformInfo, numCompatibleSubroutines);
}

const unsigned int * GLSLCHelper::GetCompatibleSubroutineIndices(const GLSLCsubroutineUniformInfo * subroutineUniformInfo, unsigned int *numCompatibleSubroutines) const
{
    const GLSLCoutput *glslcOutput = GetCompiledOutput(0);
    assert(glslcOutput);

    const char * compatibleIndexPoolPtr = NULL;

    // Find the compatible index pool
    for (unsigned int i = 0; i < glslcOutput->numSections; ++i) {
        if (glslcOutput->headers[i].genericHeader.common.type ==
            GLSLC_SECTION_TYPE_REFLECTION) {
            const GLSLCprogramReflectionHeader * reflectionHeader = &glslcOutput->headers[i].programReflectionHeader;
            const char * reflectionData = (const char*)glslcOutput + reflectionHeader->common.dataOffset;

            compatibleIndexPoolPtr = reflectionData + reflectionHeader->subroutineCompatibleIndexPoolOffset;
        }
    }

    *numCompatibleSubroutines = subroutineUniformInfo->numCompatibleSubroutines;
    const unsigned int * compatibleIndicesPtr = (const unsigned int *)(compatibleIndexPoolPtr + subroutineUniformInfo->compatibleSubroutineInfoOffset);

    return compatibleIndicesPtr;
}

LWNsubroutineLinkageMapPtr GLSLCHelper::GetSubroutineLinkageMap(LWNshaderStage stage, unsigned int compiledOutputNdx, int *size) const
{
    // Gets the linkage map for this GLSLC program stage.
    const GLSLCoutput * glslcOutput = GetCompiledOutput(compiledOutputNdx);
    const void * retVal = NULL;

    assert(compiledOutputNdx < GetNumCompiledOutputs());

    for (unsigned int i = 0; i < glslcOutput->numSections; ++i) {
        if (glslcOutput->headers[i].genericHeader.common.type ==
            GLSLC_SECTION_TYPE_GPU_CODE) {

            const GLSLCgpuCodeHeader *gpuCodeHeader =
                (const GLSLCgpuCodeHeader *)(&glslcOutput->headers[i].gpuCodeHeader);

            char * data = (char *)glslcOutput + gpuCodeHeader->common.dataOffset;

            if (gpuCodeHeader->stage != stage) {
                continue;
            }

            assert(retVal == NULL);

            // Get the subroutine information.
            if (gpuCodeHeader->subroutineLinkageMapSize == 0) {
                // No linkage information for this stage
                return NULL;
            }
            retVal = data + gpuCodeHeader->subroutineLinkageMapOffset;
            *size = gpuCodeHeader->subroutineLinkageMapSize;
            break;
        }
    }

    return (const LWNsubroutineLinkageMapPtr)retVal;
}

int32_t GLSLCHelper::ProgramGetResourceLocation(LWNprogram *programLWN, LWNshaderStage stage,
                                                LWNprogramResourceType type, const char *name)
{
    (void) programLWN;      // unused
    const GLSLCprogramInputInfo * programInput = NULL;
    const GLSLLwniformInfo * uniform = NULL;
    const GLSLCbufferVariableInfo * bufferVar = NULL;
    const GLSLLwniformBlockInfo * ubo = NULL;
    const GLSLCssboInfo * ssbo = NULL;
    int retVal = -1;

    switch (type){
    case LWN_PROGRAM_RESOURCE_TYPE_VERTEX_ATTRIB:
        programInput = (const GLSLCprogramInputInfo *) FindValueInPiqMap(REFLECTION_TYPE_VERTEX_ATTRIB, name);
        if (programInput) {
            retVal = programInput->location;
        }
        break;
    case LWN_PROGRAM_RESOURCE_TYPE_SAMPLER:
        uniform = (const GLSLLwniformInfo *) FindValueInPiqMap(REFLECTION_TYPE_SAMPLER, name);
        if (uniform) {
            retVal = uniform->bindings[stage];
        }
        break;
    case LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_BLOCK:
        ubo = (const GLSLLwniformBlockInfo *) FindValueInPiqMap(REFLECTION_TYPE_UNIFORM_BLOCK, name);
        if (ubo) {
            retVal = ubo->bindings[stage];
        }
        break;
    case LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_BLOCK_SIZE:
        ubo = (const GLSLLwniformBlockInfo *) FindValueInPiqMap(REFLECTION_TYPE_UNIFORM_BLOCK, name);
        if (ubo) {
            retVal = ubo->size;
        }
        break;
    case LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_OFFSET:
        uniform = (const GLSLLwniformInfo *) FindValueInPiqMap(REFLECTION_TYPE_UNIFORM, name);
        if (uniform) {
            retVal = uniform->blockOffset;
        }
        break;
    case LWN_PROGRAM_RESOURCE_TYPE_IMAGE:
        uniform = (const GLSLLwniformInfo *) FindValueInPiqMap(REFLECTION_TYPE_IMAGE, name);
        if (uniform) {
            retVal = uniform->bindings[stage];
        }
        break;
    case LWN_PROGRAM_RESOURCE_TYPE_STORAGE_BLOCK:
        ssbo = (const GLSLCssboInfo *) FindValueInPiqMap(REFLECTION_TYPE_STORAGE_BLOCK, name);
        if (ssbo) {
            retVal = ssbo->bindings[stage];
        }
        break;
    case LWN_PROGRAM_RESOURCE_TYPE_STORAGE_BLOCK_SIZE:
        ssbo = (const GLSLCssboInfo *) FindValueInPiqMap(REFLECTION_TYPE_STORAGE_BLOCK, name);
        if (ssbo) {
            retVal = ssbo->size;
        }
        break;
    case LWN_PROGRAM_RESOURCE_TYPE_BUFFER_VARIABLE_OFFSET:
        bufferVar = (const GLSLCbufferVariableInfo *) FindValueInPiqMap(REFLECTION_TYPE_BUFFER_VARIABLE, name);
        if (bufferVar) {
            retVal = bufferVar->blockOffset;
        }
        break;
    case LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_ARRAY_STRIDE:
        uniform = (const GLSLLwniformInfo *) FindValueInPiqMap(REFLECTION_TYPE_UNIFORM, name);
        if (uniform) {
            if (uniform->isArray) {
                retVal = (int32_t)uniform->arrayStride;
            }
        }
        break;
    default:
        assert(0);
        break;
    };

    return retVal;
}

void GLSLCHelper::AddIncludePath( const char * path )
{
    m_includePaths.push_back(path);
}

void GLSLCHelper::AddforceIncludeStdHeader( const char * path )
{
    m_userOptions.forceIncludeStdHeader = path; 
}

void GLSLCHelper::SetSeparable( LWNboolean isSeparable )
{
    m_userOptions.optionFlags.glslSeparable = isSeparable;
}

void GLSLCHelper::SetEnablePiqSupport( LWNboolean enablePiq )
{
    m_userOptions.optionFlags.outputShaderReflection = enablePiq;
}

void GLSLCHelper::EnablePerfStats( LWNboolean outputPerfStats )
{
    m_userOptions.optionFlags.outputPerfStats = outputPerfStats;
}

void GLSLCHelper::EnableThinBinaries( LWNboolean outputThinGpuBinaries )
{
    m_userOptions.optionFlags.outputThinGpuBinaries = outputThinGpuBinaries;
}

void GLSLCHelper::EnableTessellationAndPassthroughGS(LWNboolean enable)
{
    m_userOptions.optionFlags.tessellationAndPassthroughGS = enable ? 1 : 0;
}

void GLSLCHelper::SetDebugLevel(GLSLCdebugInfoLevelEnum debugLevel)
{
    assert(debugLevel <= GLSLC_DEBUG_LEVEL_G2);

    m_userOptions.optionFlags.outputDebugInfo = debugLevel;
}

void GLSLCHelper::SetOptLevel(GLSLCoptLevelEnum optLevel)
{
    assert((optLevel == GLSLC_OPTLEVEL_DEFAULT) || (optLevel == GLSLC_OPTLEVEL_NONE));

    m_userOptions.optionFlags.optLevel = optLevel;
}

void GLSLCHelper::EnableSassDump( LWNboolean enable )
{
    m_userOptions.optionFlags.outputAssembly = enable;
}

void GLSLCHelper::EnableCBF( LWNboolean enable )
{
    m_userOptions.optionFlags.enableCBFOptimization = enable ? 1 : 0;
}

void GLSLCHelper::EnableWarpLwlling( LWNboolean enable )
{
    m_userOptions.optionFlags.enableWarpLwlling = enable ? 1 : 0;
}

void GLSLCHelper::EnableMultithreadCompilation( LWNboolean enable )
{
    m_userOptions.optionFlags.enableMultithreadCompilation = enable ? 1 : 0;
}

const GLSLCoutput * GLSLCHelper::GetGlslcOutput()
{
    if (!m_libraryHelper->IsLoaded()) {
        return NULL;
    }

    return GetCompiledOutput(0);
}

void GLSLCHelper::SetTransformFeedbackVaryings(uint32_t count, const char ** ilwaryings)
{
    // Sets the XFB varyings inside the GLSLC options structure
    m_userOptions.xfbVaryingInfo.numVaryings = count;
    m_userOptions.xfbVaryingInfo.varyings = ilwaryings;
}

void GLSLCHelper::AddSpecializationUniform(uint32_t index, const GLSLCspecializationUniform * uniform) {
    // Adds to our internal specialization uniform vectors.
    m_glslcSpecArrays[index].push_back(*uniform);
}

void GLSLCHelper::ClearSpecializationUniformArray(uint32_t index) {
    m_glslcSpecArrays[index].clear();
}

void GLSLCHelper::ClearSpecializationUniformArrays() {
    for (int i = 0; i < MAX_SPEC_ARRAYS; ++i) {
        ClearSpecializationUniformArray(i);
    }
}

void GLSLCHelper::AddSpirvSpecializationConstant(LWNshaderStage stage, uint32_t constantID, uint32_t data) {
    m_spirvSpecConstArrays[stage].constantIDs.push_back(constantID);
    m_spirvSpecConstArrays[stage].datum.push_back(data);
}

void GLSLCHelper::ClearSpirvSpecializationConstantArray(LWNshaderStage stage) {
    m_spirvSpecConstArrays[stage].constantIDs.clear();
    m_spirvSpecConstArrays[stage].datum.clear();
}

void GLSLCHelper::ClearSpirvSpecializationConstantArrays() {
    for (int i = 0; i < GLSLC_NUM_SHADER_STAGES; ++i) {
        ClearSpirvSpecializationConstantArray(LWNshaderStage(i));
    }
}

uint32_t GLSLCHelper::GetNumCompiledOutputs() const {
    uint32_t count = 0;

    for (unsigned int i = 0; i < MAX_SPEC_ARRAYS; ++i) {
        if (GetCompiledOutput(i) != NULL) {
            count++;
        }
    }

    return count;
}

const GLSLCoutput * GLSLCHelper::GetCompiledOutput(uint32_t index) const
{
    if (index >= MAX_SPEC_ARRAYS) {
        assert(!"Index value is too large.");
        return NULL;
    }

    return m_lastCompiledOutputs[index];
}

void GLSLCHelper::OverrideChip(unsigned int arch, unsigned int impl)
{
    m_overrideArch = arch;
    m_overrideImpl = impl;
}

void GLSLCHelper::OverrideGlslangOptions(uint8_t doGlslangShim, uint8_t glslangFallbackOnError, uint8_t glslangFallbackOnAbsolute)
{
    m_overrideDoGlslangShim = doGlslangShim;
    m_overrideGlslangFallbackOnError = glslangFallbackOnError;
    m_overrideGlslangFallbackOnAbsolute = glslangFallbackOnAbsolute;
}

void GLSLCHelper::InitializeGlslcInputAndOptions(GLSLCinput * input, GLSLCoptions * options,
        const LWNshaderStage * stages, uint32_t count, const char ** shaderSources,
        const SpirvParams * spvParams)
{
    *options = m_libraryHelper->glslcGetDefaultOptions();
    memset(input, 0, sizeof(GLSLCinput));

    // Set the internal options with those set by individual tests.  m_userOptions is initialized with the defaults, so
    // if tests don't set them then options->optionFlags just gets set with the default initialized values.
    options->optionFlags = m_userOptions.optionFlags;

    // For program interface query support, GLSLC needs to report the reflection information.
    // This is enabled by default for tests, and can be disabled using the function SetEnablePiqSupport.
    options->optionFlags.outputShaderReflection = 1;

    // Don't let tests override the output of GPU binaries.
    options->optionFlags.outputGpuBinaries = 1;

    // Always enable performance statistics.  Many toolchains seem to enable this flag by default,
    // so using it in our testing is good coverage.  Generation of perf stats should not interfere
    // with tests.
    options->optionFlags.outputPerfStats = 1;

    options->forceIncludeStdHeader = m_userOptions.forceIncludeStdHeader;

    // Set the include paths based on the settings.
    if (m_includePaths.size() > 0) {
        options->includeInfo.paths = const_cast<char**>(&(m_includePaths[0]));
        options->includeInfo.numPaths = uint32_t (m_includePaths.size());
    }

    // Set the options->xfb
    options->xfbVaryingInfo.numVaryings = m_userOptions.xfbVaryingInfo.numVaryings;
    options->xfbVaryingInfo.varyings = m_userOptions.xfbVaryingInfo.varyings;

    // Set the input to GLSLC
    input->count = (uint8_t) count;
    input->stages = stages;
    input->sources = shaderSources;

    if (spvParams) {
        input->spirvModuleSizes = &spvParams->sizes[0];
        options->optionFlags.language = GLSLC_LANGUAGE_SPIRV;

        // set up the Specialization Constant parameters for the spirv.
        memset(m_pGlslSpirvSpecConstInfo, 0, sizeof(m_pGlslSpirvSpecConstInfo));
        memset(m_glslSpirvSpecConstInfo, 0, sizeof(m_glslSpirvSpecConstInfo));

        int specStageNums = 0;
        for (uint32_t lwrrCount = 0; lwrrCount < count; ++lwrrCount) {
            LWNshaderStage stage = input->stages[lwrrCount];

            if (!m_spirvSpecConstArrays[stage].constantIDs.empty()) {
                assert(m_spirvSpecConstArrays[stage].constantIDs.size() == m_spirvSpecConstArrays[stage].datum.size());

                m_glslSpirvSpecConstInfo[stage].constantIDs = &m_spirvSpecConstArrays[stage].constantIDs[0];
                m_glslSpirvSpecConstInfo[stage].data = &m_spirvSpecConstArrays[stage].datum[0];
                m_glslSpirvSpecConstInfo[stage].numEntries = (uint32_t)m_spirvSpecConstArrays[stage].constantIDs.size();

                m_pGlslSpirvSpecConstInfo[lwrrCount] = &m_glslSpirvSpecConstInfo[stage];
                ++specStageNums;
            }

            if (specStageNums != 0) {
                input->spirvSpecInfo = (const GLSLCspirvSpecializationInfo * const *) &m_pGlslSpirvSpecConstInfo;
            }
        }
    }
}

const GLSLCcompileObject * GLSLCHelper::GetCompileObject()
{
    return &m_compileObject;
}

GLSLCHelperCache * GLSLCHelper::GetHelperCache()
{
    return m_cacheParameters.m_dataCache;
}

void GLSLCHelper::OverrideGLSLCPrivateOptions(GLSLCcompileObject * compileObject)
{
    void ** privateData = (void **)(compileObject->privateData);
    if (privateData) {
        uint32_t * privateOptions = (uint32_t *)(*privateData);
        if (privateOptions) {
            // Chip overrides (only work on special builds of GLSLC, not meant for production)
            // Arch/implementation have constant offsets in GLSLC compile object, but they
            // are hidden so we need to access them manually (will also be ignored if GLSLC
            // is not an internal build)
            privateOptions[4] = m_overrideArch;
            privateOptions[5] = m_overrideImpl;

#if defined(_WIN32)
            // GLSLANG overrides. This is only supported on Windows versions of GLSLC.
            uint32_t glslangOptions = 0;
            glslangOptions |= m_overrideDoGlslangShim;
            glslangOptions |= (m_overrideGlslangFallbackOnError << 8);
            glslangOptions |= (m_overrideGlslangFallbackOnAbsolute << 16);
            privateOptions[6] = glslangOptions;
#endif
        }
    }
}

// Only compile the shaders using glslcCompilePreSpecialized.  glslcCompileSpecialized can be used
// later to continue compiling from specialization parameters.
LWNboolean GLSLCHelper::CompileShadersPreSpecialized(const LWNshaderStage * stages,
                                                     uint32_t count, const char ** shaderSources)
{
    assert(m_libraryHelper->IsLoaded());

    GLSLCcompileObject * compileObject = &m_compileObject;

    // Finalize any previous compile object instance.
    m_libraryHelper->glslcFinalize(compileObject);

    if (!m_libraryHelper->glslcInitialize(compileObject)) {
        m_logger.Log("GLSLCHelper: glslcInitialize failed.  Compile object's "
                     "error code: %d.\n", compileObject->initStatus);
        return LWN_FALSE;
    }

    OverrideGLSLCPrivateOptions(compileObject);

    GLSLCoptions * options = &compileObject->options;
    GLSLCinput * input = &compileObject->input;

    InitializeGlslcInputAndOptions(input, options, stages, count, shaderSources, NULL);

    uint8_t compileSuccess = 0;

    // Call into GLSLC to compile the shaders with the supplied options.
    compileSuccess = CompileGLSLCShaders(input, options, compileObject, NULL, GLSLC_COMPILE_TYPE_PRE_SPECIALIZED);
    if (!compileSuccess) {
        return LWN_FALSE;
    }

    return LWN_TRUE;
}

LWNboolean GLSLCHelper::CompileShaders( const LWNshaderStage * stages,
        uint32_t count, const char ** shaderSources, SpirvParams * spvParams)
{
    assert(m_libraryHelper->IsLoaded());

    GLSLCcompileObject * compileObject = &m_compileObject;
    //
    // Finalize any previous compile object instance.
    m_libraryHelper->glslcFinalize(compileObject);

    if (!m_libraryHelper->glslcInitialize(compileObject)) {
        m_logger.Log("GLSLCHelper: glslcInitialize failed.  Compile object's "
                     "error code: %d.\n",
                compileObject->initStatus);
        return LWN_FALSE;
    }

    OverrideGLSLCPrivateOptions(compileObject);

    GLSLCoptions * options = &compileObject->options;
    GLSLCinput * input = &compileObject->input;

    InitializeGlslcInputAndOptions(input, options, stages, count, shaderSources, spvParams);

    uint8_t compileSuccess = 0;

    int numSpecArrays = 0;
    for (int i = 0; i < MAX_SPEC_ARRAYS; ++i) {
        if (m_glslcSpecArrays[i].size() != 0) {
            numSpecArrays++;
        }
    }

    // Call into GLSLC to compile the shaders with the supplied options.
    compileSuccess = CompileGLSLCShaders(input, options, compileObject, (numSpecArrays > 0) ? m_glslcSpecArrays : NULL);
    if (!compileSuccess) {
        return LWN_FALSE;
    }

    return LWN_TRUE;
}

const GLSLCperfStatsHeader * GLSLCHelper::ExtractPerfStatsSection(const GLSLCoutput * glslcOutput, const GLSLCgpuCodeHeader * gpuSectionHeader)
{
    // No GPU code section found for the given stage.
    if (gpuSectionHeader == NULL) {
        return NULL;
    }

    uint32_t perfStatsSectionIndex = gpuSectionHeader->perfStatsSectionNdx;

    // The perf statistics section will only be 0 if no perf statistics section
    // exists in the output.
    if (perfStatsSectionIndex == 0) {
        return NULL;
    }
    return &(glslcOutput->headers[perfStatsSectionIndex].perfStatsHeader);
}

GLSLCHelper::ReflectionTypeEnum GLSLCHelper::GetUniformKind(const GLSLLwniformInfo * uniform)
{
    GLSLCHelper::ReflectionTypeEnum kind;

    switch(uniform->kind) {
    case GLSLC_PIQ_UNIFORM_KIND_PLAIN:
        kind = REFLECTION_TYPE_UNIFORM;
        break;
    case GLSLC_PIQ_UNIFORM_KIND_SAMPLER:
        kind = REFLECTION_TYPE_SAMPLER;
        break;
    case GLSLC_PIQ_UNIFORM_KIND_IMAGE:
        kind = REFLECTION_TYPE_IMAGE;
        break;
    default:
        kind = REFLECTION_TYPE_ILWALID;
        break;
    };

    return kind;
}

void GLSLCHelper::ParseReflectionInfo(const GLSLCprogramReflectionHeader * reflectionHeader, const char * data)
{
    // Clear the internal reflection map.
    m_piqMap.clear();

    // Go through each reflection type and store in the corresponding map
    const char *stringPool = data + reflectionHeader->stringPoolOffset;

    // Program inputs
    const GLSLCprogramInputInfo * programInput = (const GLSLCprogramInputInfo *)(data + reflectionHeader->programInputsOffset);
    for (unsigned int i = 0; i < reflectionHeader->numProgramInputs; ++i) {
        const char * name = stringPool + programInput->nameInfo.nameOffset;
        m_piqMap[REFLECTION_TYPE_VERTEX_ATTRIB][name] = (const void *)programInput;
        programInput++;
    }

    // Uniforms (uniforms, images, samplers)
    const GLSLLwniformInfo *uniform = (const GLSLLwniformInfo *)(data + reflectionHeader->uniformOffset);
    for (unsigned int i = 0; i < reflectionHeader->numUniforms; ++i) {
        const char * name = stringPool + uniform->nameInfo.nameOffset;
        ReflectionTypeEnum reflectionType = GetUniformKind(uniform);
        m_piqMap[reflectionType][name] = (const void *)uniform;
        uniform++;
    }

    // Uniform blocks
    const GLSLLwniformBlockInfo * ubo = (const GLSLLwniformBlockInfo *)((const char *)data + reflectionHeader->uniformBlockOffset);
    for (unsigned int i = 0; i < reflectionHeader->numUniformBlocks; ++i) {
        const char * name = stringPool + ubo->nameInfo.nameOffset;
        m_piqMap[REFLECTION_TYPE_UNIFORM_BLOCK][name] = (const void *)ubo;
        ubo++;
    }

    // Storage blocks
    const GLSLCssboInfo * ssbo = (const GLSLCssboInfo *)((const char *)data + reflectionHeader->ssboOffset);
    for (unsigned int i = 0; i < reflectionHeader->numSsbo; ++i) {
        const char * name = stringPool + ssbo->nameInfo.nameOffset;
        m_piqMap[REFLECTION_TYPE_STORAGE_BLOCK][name] = (const void *)ssbo;
        ssbo++;
    }

    // Buffer variables
    const GLSLCbufferVariableInfo * bufferVar = (const GLSLCbufferVariableInfo *)((const char *)data + reflectionHeader->bufferVariableOffset);
    for (unsigned int i = 0; i < reflectionHeader->numBufferVariables; ++i) {
        const char * name = stringPool + bufferVar->nameInfo.nameOffset;
        m_piqMap[REFLECTION_TYPE_BUFFER_VARIABLE][name] = (const void *)bufferVar;
        bufferVar++;
    }

    ReflectionTypeEnum subroutineReflectionTypes[GLSLC_NUM_SHADER_STAGES] = {
        REFLECTION_TYPE_SUBROUTINE_VERTEX,
        REFLECTION_TYPE_SUBROUTINE_FRAGMENT,
        REFLECTION_TYPE_SUBROUTINE_GEOMETRY,
        REFLECTION_TYPE_SUBROUTINE_TESS_CONTROL,
        REFLECTION_TYPE_SUBROUTINE_TESS_EVALUATION,
        REFLECTION_TYPE_SUBROUTINE_COMPUTE,
    };

    // Subroutines
    const GLSLCsubroutineInfo * subroutineInfo = (const GLSLCsubroutineInfo *)((const char *)data + reflectionHeader->subroutineOffset);
    for (unsigned int i = 0; i < reflectionHeader->numSubroutines; ++i) {
        const char * name = stringPool + subroutineInfo->nameInfo.nameOffset;
        LWNshaderStage stage = subroutineInfo->stage;

        m_piqMap[subroutineReflectionTypes[stage]][name] = (const void *)subroutineInfo;
        subroutineInfo++;
    }

    ReflectionTypeEnum subroutineUniformReflectionTypes[GLSLC_NUM_SHADER_STAGES] = {
        REFLECTION_TYPE_SUBROUTINE_UNIFORM_VERTEX,
        REFLECTION_TYPE_SUBROUTINE_UNIFORM_FRAGMENT,
        REFLECTION_TYPE_SUBROUTINE_UNIFORM_GEOMETRY,
        REFLECTION_TYPE_SUBROUTINE_UNIFORM_TESS_CONTROL,
        REFLECTION_TYPE_SUBROUTINE_UNIFORM_TESS_EVALUATION,
        REFLECTION_TYPE_SUBROUTINE_UNIFORM_COMPUTE,
    };

    // Subroutine uniforms
    const GLSLCsubroutineUniformInfo * subroutineUniformInfo = (const GLSLCsubroutineUniformInfo *)((const char *)data + reflectionHeader->subroutineUniformOffset);
    for (unsigned int i = 0; i < reflectionHeader->numSubroutineUniforms; ++i) {
        const char * name = stringPool + subroutineUniformInfo->nameInfo.nameOffset;
        LWNshaderStage stage = subroutineUniformInfo->stage;

        m_piqMap[subroutineUniformReflectionTypes[stage]][name] = (const void *)subroutineUniformInfo;
        subroutineUniformInfo++;
    }
}

const GLSLCprogramReflectionHeader * GLSLCHelper::ExtractReflectionSection(const GLSLCoutput * glslcOutput)
{
    for (uint32_t i = 0; i < glslcOutput->numSections; ++i) {
        GLSLCsectionTypeEnum type = glslcOutput->headers[i].genericHeader.common.type;
        if (type == GLSLC_SECTION_TYPE_REFLECTION) {
            // The stage matches, get the corresponding perf stats section.
            return &glslcOutput->headers[i].programReflectionHeader;
        }
    }

    return NULL;
}

const GLSLCgpuCodeHeader * GLSLCHelper::ExtractGpuCodeSection(const GLSLCoutput * glslcOutput, LWNshaderStage stage)
{
    for (uint32_t i = 0; i < glslcOutput->numSections; ++i) {
        GLSLCsectionTypeEnum type = glslcOutput->headers[i].genericHeader.common.type;
        if (type == GLSLC_SECTION_TYPE_GPU_CODE) {

            // check the stage
            LWNshaderStage gpuCodeStage =
                glslcOutput->headers[i].gpuCodeHeader.stage;

            if (gpuCodeStage != stage) {
                continue;
            }

            // The stage matches, get the corresponding perf stats section.
            return &glslcOutput->headers[i].gpuCodeHeader; 
        }
    }

    // No match found for the input stage.
    return NULL;
}

LWNboolean GLSLCHelper::CompileHlslToSpirv(uint32_t nShaders, const LWNshaderStage * stages,
    const char ** glslStrings, char ** spirvBins, uint32_t * spirvBinSizes)
{
    if (!m_dxcLibraryHelper || !m_dxcLibraryHelper->IsLoaded()) {
        // At this point the library needs to be loaded.
        return LWN_FALSE;
    }

    memset(spirvBins, 0, sizeof(const char *) * GLSLC_NUM_SHADER_STAGES);
    memset(spirvBinSizes, 0, sizeof(uint32_t) * GLSLC_NUM_SHADER_STAGES);

    bool spirvSuccess = LWN_TRUE;
    for (uint32_t i = 0; i < nShaders; i++) {
#ifdef DXC_ENABLED
        spirvBins[i] = m_dxcLibraryHelper->DxcCompile(glslStrings[i], 0, stages[i], &spirvBinSizes[i], &m_logger);
#endif // DXC_ENABLED
        if (!spirvBins[i] || !spirvBinSizes[i]) {
            spirvSuccess = LWN_FALSE;
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

LWNboolean GLSLCHelper::CompileAndSetShadersHLSL(LWNprogram *program, const LWNshaderStage *stages,
    uint32_t count, const char ** shaderSources)
{
    // Use the dxc dll to compile the hlsl to spv.
    char *spirvBins[GLSLC_NUM_SHADER_STAGES] = { NULL };
    SpirvParams spvParams;
    if (!CompileHlslToSpirv(count, stages, shaderSources, spirvBins, spvParams.sizes)) {
        return LWN_FALSE;
    }

    // Compile the spirv to exelwtable binary.
    LWNboolean compileSuccess = lwnUtil::GLSLCHelper::CompileAndSetShaders(
        program, stages, count, (const char **)spirvBins, &spvParams);

    // Free the SPIR-V binaries allocated inside DxcCompiler::CompileHlslToSpirv.
    for (int i = 0; i < GLSLC_NUM_SHADER_STAGES; ++i) {
        free(spirvBins[i]);
        spirvBins[i] = NULL;
    }

    return compileSuccess;
}

LWNboolean GLSLCHelper::CompileAndSetShaders (LWNprogram * program, const LWNshaderStage * stages, uint32_t count,
                                              const char ** shaderSources, SpirvParams * spirvParams)
{
    if (!m_libraryHelper->IsLoaded()) {
        // At this point the library needs to be loaded.
        m_logger.Log("GLSLCHelper: GLSLC Library must be loaded and initialized to compile shaders\n");
        return LWN_FALSE;
    }

    if (!CompileShaders(stages, count, shaderSources, spirvParams)) {
        m_logger.Log("GLSLCHelper: Can not compile shaders.\nInfolog:\n%s\n\n", GetInfoLog() );
        for (unsigned int i = 0; i < count; ++i) {
            m_logger.Log("Shader source for stage %d:\n%s\n", stages[i], shaderSources[i]);
        }

        return LWN_FALSE;
    }

    // Call set shaders with the first GLSLCoutput.  If not using specialization, this will
    // be the only output.  If using specialization, this will be the first output.
    return SetShaders(program, GetCompiledOutput(0));
}

void GLSLCHelper::ResetBuffers()
{
    for (unsigned int i = 0; i < m_buffers.size(); ++i) {
        LWNbuffer * buff = m_buffers[i];
        m_allocator.freeBuffer(buff);
    }

    m_buffers.clear();
}

GLSLCHelper::~GLSLCHelper()
{
    Reset();
}

const char * GLSLCHelper::GetInfoLog()
{
    if (m_compileObject.initStatus != GLSLC_INIT_SUCCESS) {
        return "The GLSLC compile object was not initialized properly. "
               "Check the GLSLCcompileObject::initStatus flag.\n";
    }

    GLSLCresults * results = m_compileObject.lastCompiledResults;

    if (results == NULL || results->compilationStatus == NULL) {
        // This could happen if we retrieved the compiled results from the cache
        // instead of from GLSLC.
        return "No infolog available (Note: cache input doesn't provide an infolog).";
    }

    GLSLCcompilationStatus * status = results->compilationStatus;

    if (status->infoLog == NULL) {
        return "No infolog available.";
    }

    return status->infoLog;
}

LWNboolean GLSLCHelper::SetShaders(LWNprogram * program, const GLSLCoutput * glslcOutput)
{
    LWNshaderData shaderData[6];
    memset(&shaderData[0], 0, sizeof(LWNshaderData)*6);

    // Check that the compiled shader's required scratch memory is sufficient to run.
    ScratchMemCheckEnum check = CheckScratchMem(glslcOutput);

    if (check == SCRATCH_MEM_CHECK_INSUFFICIENT) {
        // This is a hard stop.
        return LWN_FALSE;
    }

    assert((check == SCRATCH_MEM_CHECK_SUFFICIENT) || (check == SCRATCH_MEM_CHECK_THROTTLE));

    int count = 0;
    int lwrrIndex = 0;

    for (unsigned int i = 0; i < glslcOutput->numSections; ++i) {
        if (glslcOutput->headers[i].genericHeader.common.type ==
            GLSLC_SECTION_TYPE_GPU_CODE) {

            char * ucode = NULL;
            char * control = NULL;
            int ucodeSize = 0;

            GLSLCgpuCodeHeader gpuCodeHeader =
                (GLSLCgpuCodeHeader)(glslcOutput->headers[i].gpuCodeHeader);

            char * data = (char *)glslcOutput + gpuCodeHeader.common.dataOffset;

            ucode = data + gpuCodeHeader.dataOffset;
            control = data + gpuCodeHeader.controlOffset;
            ucodeSize = gpuCodeHeader.dataSize;

            // Create a new buffer to contain the memory.
            LWNbufferBuilder bufferBuilder;
            lwnBufferBuilderSetDefaults(&bufferBuilder);
            lwnBufferBuilderSetDevice(&bufferBuilder, m_device);

            // Allocate the memory in the pool for the program.
            LWNbuffer * progBuf = m_allocator.allocBuffer(&bufferBuilder, BUFFER_ALIGN_VERTEX_BIT, ucodeSize);

            if (m_allocator.pool(progBuf)) {
                ptrdiff_t bufOffset = m_allocator.offset(progBuf);
                size_t bufSize = m_allocator.size(progBuf);
                lwnMemoryPoolFlushMappedRange(m_allocator.pool(progBuf), bufOffset, bufSize);
            }

            // Copy the memory over.
            void * cpuMap = lwnBufferMap(progBuf);
            memcpy(cpuMap, ucode, ucodeSize);

            shaderData[lwrrIndex].data = lwnBufferGetAddress(progBuf);
            shaderData[lwrrIndex].control = control;
            count++;

            m_buffers.push_back(progBuf);
            lwrrIndex++;
        }
    }

    return lwnProgramSetShaders(program, count, &(shaderData[0]));
}

LWNboolean GLSLCHelper::SetShaderScratchMemory(LWNmemoryPool * memPool, ptrdiff_t offset, size_t size, LWNcommandBuffer * cmdBuf)
{
    // Ensure the input size matches the required granularity

    int granularity = 0;
    lwnDeviceGetInteger(m_device, LWN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_GRANULARITY, &granularity);

    if ((size % granularity) != 0) {
        //assert(false);
        return LWN_FALSE;
    }

    m_scratchMemPool = memPool;
    m_scratchMemPoolOffset = offset;
    m_scratchMemPoolSize = size;

    // If <cmdBuf> is not NULL, use it to program the hardware.  If <cmdBuf>
    // is NULL, assume the allocation has already been registered and this
    // call is just to inform the GLSLCHelper.
    if (cmdBuf) {
        lwnCommandBufferSetShaderScratchMemory(cmdBuf, memPool, offset, size);
    }

    return LWN_TRUE;
}

GLSLCHelper::ScratchMemCheckEnum GLSLCHelper::CheckScratchMem(const GLSLCoutput * glslcOutput)
{
    // This function gets the maximum amount of scratch memory from all of
    // the stages.

    size_t scratchMemoryRequired = GetScratchMemoryPerWarp(glslcOutput);

    bool isCompute = false;

    for (unsigned int i = 0; i < glslcOutput->numSections; ++i) {
        if (glslcOutput->headers[i].genericHeader.common.type ==
            GLSLC_SECTION_TYPE_GPU_CODE) {

            const GLSLCgpuCodeHeader *gpuCodeHeader =
                &(glslcOutput->headers[i].gpuCodeHeader);

            if (gpuCodeHeader->stage == LWN_SHADER_STAGE_COMPUTE) {
                isCompute = true;
            }
        }
    }

    // Compute how much is absolutely required for this system.
    int minScaleFactor = 0;
    int recommendedScaleFactor = 0;
    LWNdeviceInfo scaleMinimumEnum = (isCompute ?
        LWN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_COMPUTE_SCALE_FACTOR_MINIMUM :
        LWN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_SCALE_FACTOR_MINIMUM);

    lwnDeviceGetInteger(m_device, scaleMinimumEnum, &minScaleFactor);
    lwnDeviceGetInteger(m_device, LWN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_SCALE_FACTOR_RECOMMENDED, &recommendedScaleFactor);

    if (scratchMemoryRequired * minScaleFactor > m_scratchMemPoolSize) {
        return SCRATCH_MEM_CHECK_INSUFFICIENT;
    }

    // Check if we have the recommended amount.
    if (scratchMemoryRequired * recommendedScaleFactor > m_scratchMemPoolSize) {
        return SCRATCH_MEM_CHECK_THROTTLE;
    }

    return SCRATCH_MEM_CHECK_SUFFICIENT;

}

size_t GLSLCHelper::GetScratchMemoryPerWarp(const GLSLCoutput * glslcOutput)
{
    size_t maxFound = 0;

    for (unsigned int i = 0; i < glslcOutput->numSections; ++i) {
        if (glslcOutput->headers[i].genericHeader.common.type ==
                GLSLC_SECTION_TYPE_GPU_CODE) {

            GLSLCgpuCodeHeader gpuCodeHeader =
                (GLSLCgpuCodeHeader)(glslcOutput->headers[i].gpuCodeHeader);

            if (gpuCodeHeader.scratchMemBytesPerWarp > maxFound) {
                maxFound = gpuCodeHeader.scratchMemBytesPerWarp;
            }
        }
    }

    return maxFound;
}

size_t GLSLCHelper::GetScratchMemoryRecommended(const LWNdevice *device, const GLSLCoutput * glslcOutput)
{
    (void) device;          // unused parameter

    int scratchScaleFactor, scratchGranularity;

    lwnDeviceGetInteger(m_device, LWN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_SCALE_FACTOR_RECOMMENDED, &scratchScaleFactor);
    lwnDeviceGetInteger(m_device, LWN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_GRANULARITY, &scratchGranularity);

    return AlignSize(GetScratchMemoryPerWarp(glslcOutput) * scratchScaleFactor, scratchGranularity);
}

size_t GLSLCHelper::GetScratchMemoryMinimum(const GLSLCoutput * glslcOutput)
{
    // The minimum scale factors of compute shader and graphic shader are different.
    bool isCompute = false;
    for (unsigned int i = 0; i < glslcOutput->numSections; ++i) {
        if (glslcOutput->headers[i].genericHeader.common.type ==
            GLSLC_SECTION_TYPE_GPU_CODE) {

            const GLSLCgpuCodeHeader *gpuCodeHeader =
                &(glslcOutput->headers[i].gpuCodeHeader);

            if (gpuCodeHeader->stage == LWN_SHADER_STAGE_COMPUTE) {
                isCompute = true;
            }
        }
    }

    int minScaleFactor, scratchGranularity;
    LWNdeviceInfo scaleMinimumEnum = (isCompute ?
        LWN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_COMPUTE_SCALE_FACTOR_MINIMUM :
        LWN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_SCALE_FACTOR_MINIMUM);

    lwnDeviceGetInteger(m_device, scaleMinimumEnum, &minScaleFactor);
    lwnDeviceGetInteger(m_device, LWN_DEVICE_INFO_SHADER_SCRATCH_MEMORY_GRANULARITY, &scratchGranularity);

    return AlignSize(GetScratchMemoryPerWarp(glslcOutput) * minScaleFactor, scratchGranularity);
}
GLSLCLogger * GLSLCHelper::GetLogger() {
    return &m_logger;
}

LWNboolean GLSLCHelper::CompileGLSLCShadersSpecialized(GLSLCinput * input,
                                                       GLSLCoptions * options,
                                                       GLSLCcompileObject * compileObject,
                                                       std::vector<GLSLCspecializationUniform> * inputSpecArrays,
                                                       SpecializedCompileType type)
{
    (void) input;           // unused parameter
    (void) options;         // unused parameter
    GLSLCspecializationBatch specEntries;
    GLSLCspecializationSet entries[MAX_SPEC_ARRAYS];
    memset(entries, 0, sizeof(entries));
    int numSpecEntries = 0;


    // TODO: Enable caching for specialization entries.

    if (!m_libraryHelper->glslcCompilePreSpecialized(compileObject)) {
        GLSLCresults * results = compileObject->lastCompiledResults;
        (void)results;
        assert(results->compilationStatus->success == false);
        return LWN_FALSE;
    }

    if (type == GLSLC_COMPILE_TYPE_PRE_SPECIALIZED) {
        // We only wanted to do precompilation, so end here.
        return LWN_TRUE;
    }

    // From this point forward, we're going all the way with specialization.
    assert(type == GLSLC_COMPILE_TYPE_FULL);

    // Count the number of specialization variables that are used, if any.
    assert(inputSpecArrays);
    for (int i = 0; i < MAX_SPEC_ARRAYS; ++i) {
        if (!inputSpecArrays[i].empty()) {
            entries[numSpecEntries].uniforms = &inputSpecArrays[i][0];
            entries[numSpecEntries].numUniforms = (uint32_t)inputSpecArrays[i].size();
            numSpecEntries++;
        }
    }

    assert(numSpecEntries > 0);

    specEntries.entries = entries;
    specEntries.numEntries = numSpecEntries;

    const GLSLCoutput * const * specResults = m_libraryHelper->glslcCompileSpecialized(compileObject, &specEntries);

    // Backend failed to compile.
    if (!compileObject->lastCompiledResults->compilationStatus->success) {
        return LWN_FALSE;
    }

    assert(specResults);
    uint32_t lwrrOutputIndex = 0;
    for (int i = 0; i < numSpecEntries; ++i) {
        assert(specResults[i]);

        assert(lwrrOutputIndex < MAX_SPEC_ARRAYS);

        // Place the outputs in the same slots as the inputs.  The GLSLC interface expects inputs
        // in sequential order, but the test interface allows tests to add specialization arrays to
        // specific indexes.  The output array is also expected to have outputs at each of the assigned
        // input indexes.
        for (int ii = lwrrOutputIndex; ii < MAX_SPEC_ARRAYS; ++ii) {
            if (inputSpecArrays[ii].size() != 0) {
                // Found the next output slot.
                break;
            }
            lwrrOutputIndex++;
        }
        m_lastCompiledOutputs[lwrrOutputIndex++] = specResults[i];
    }

    // Compilation success.
    return LWN_TRUE;
}

LWNboolean GLSLCHelper::CompileGLSLCShadersNonSpecialized(GLSLCinput * input, GLSLCoptions * options, GLSLCcompileObject * compileObject)
{
    GLSLCHelperCacheKey cacheHash = ~((uint64_t) 0);
    const GLSLCoutput * obtainedGlslcOutput = NULL;
    LWNboolean compileSuccess = LWN_FALSE;

    m_cacheParameters.m_lastCacheHit = LWN_FALSE;

    if (UsesCache()) {
        if (m_cacheParameters.m_doCacheOverride) {
            cacheHash = m_cacheParameters.m_overrideCacheKey;
        } else {
            cacheHash = m_cacheParameters.m_dataCache->HashShaders(input, options);
        }

        // #includes are not supported, and tests should disable caching if #includes
        // are used.  There isn't a good way to determine whether a shader is using #includes
        // or not since #include directives are in the shader strings themselves, and we don't
        // do any pre-parsing of the strings in this helper.  We assert on cases where #include
        // search paths are used, but that may not catch cases of #includes where no search
        // paths are specified (which should be rare).
        if (options->includeInfo.numPaths != 0) {
            assert(!"Trying to use includes with the GLSLC cache is not allowed.  Please disable the cache in the application"
                    "for these shaders");
            return LWN_FALSE;
        }
    }

    if (GetAllowCacheRead()) {
        // Note: This will not work for #includes

        // Take a hash of the shader/data and index into the binary cache to retrieve the precompiled
        // binaries.
        assert(m_cacheParameters.m_dataCache);
        GLSLCHelperCache * dataCache = m_cacheParameters.m_dataCache;
        GLSLCHelperCacheEntry * cacheData = dataCache->Get(cacheHash);

        if (cacheData) {
            bool apiVersionCompat = dataCache->IsAPIVersionCompatible();
            bool gpuVersionCompat = dataCache->IsGPUVersionCompatible();

            if (apiVersionCompat && gpuVersionCompat) {
                obtainedGlslcOutput = cacheData->output;
                m_cacheParameters.m_lastCacheHit = LWN_TRUE;
                compileSuccess = LWN_TRUE;
            }
        }
    }

    // If we didn't obtain anything from the cache, or we are using a cache and the
    // APIversion or GPU version is incompatible with our current application/driver
    // combination, recompile.
    if (!compileSuccess) {
        // Compile.
        compileSuccess = m_libraryHelper->glslcCompile(compileObject);

        // Sanity check that a GLSLC output exists.
        GLSLCresults * results = compileObject->lastCompiledResults;
        (void)results;

        if (compileSuccess == LWN_FALSE) {
            assert(results->compilationStatus->success == false);
            return LWN_FALSE;
        }

        assert(results->glslcOutput);

        obtainedGlslcOutput = compileObject->lastCompiledResults->glslcOutput;

        // Hash the results if we are using the data cache and we are allowing cache writes.
        if (GetAllowCacheWrite()) {
            assert(obtainedGlslcOutput);

            // Create a GLSLCHelperCacheEntry structure
            int totalSize = sizeof(GLSLCHelperCacheEntry) + obtainedGlslcOutput->size;
            GLSLCHelperCacheEntry * cacheKeyData = (GLSLCHelperCacheEntry *)calloc(1, totalSize);
            if (!cacheKeyData) {
                assert(!"Failure to allocate Cache entry");
            }

            // Setup the header.
            GLSLCHelperCacheEntryHeader * header = &cacheKeyData->header;

            header->magic = CACHE_ENTRY_MAGIC_NUMBER;
            header->totalSize = totalSize;
            memcpy(&cacheKeyData->output[0], obtainedGlslcOutput, obtainedGlslcOutput->size);

            header->hashLo = (uint32_t)(cacheHash & 0xFFFFFFFF);
            header->hashHi = (uint32_t)(cacheHash >> 32);

            // Add the entry to the cache.
            assert(m_cacheParameters.m_dataCache);
            m_cacheParameters.m_dataCache->Add(cacheHash, cacheKeyData);


            free(cacheKeyData);
        }
    }

    m_lastCompiledOutputs[0] = obtainedGlslcOutput;

    return compileSuccess;
}

LWNboolean GLSLCHelper::CompileGLSLCShaders(GLSLCinput * input, GLSLCoptions * options, GLSLCcompileObject * compileObject,
    std::vector<GLSLCspecializationUniform> * inputSpecArrays, SpecializedCompileType type)
{
    LWNboolean compileSuccess = LWN_FALSE;
    char * reflectionData = NULL;
    const GLSLCprogramReflectionHeader * reflectionHeader = NULL;

    memset(m_lastCompiledOutputs, 0, sizeof(m_lastCompiledOutputs));

    // If there are any specialization entries, take the specialized compile path.
    // We don't lwrrently implement caching the specialized outputs yet.
    if (inputSpecArrays || type == GLSLC_COMPILE_TYPE_PRE_SPECIALIZED) {
        // For FULL and have specialization entries, or type is PRE_SPECIALIZED, use the specialization path.
        compileSuccess = CompileGLSLCShadersSpecialized(input, options, compileObject, inputSpecArrays, type);

        if (compileSuccess) {
            // For specialization compiles, the reflection information comes from the
            // compile object's reflection section.  This section is common amongst all
            // compile outputs.
            assert(m_compileObject.reflectionSection);
            reflectionHeader = m_compileObject.reflectionSection;
            reflectionData = ((char *)reflectionHeader) + reflectionHeader->common.dataOffset;
        }
    } else {
        compileSuccess = CompileGLSLCShadersNonSpecialized(input, options, compileObject);

        if (compileSuccess) {
            const GLSLCoutput *obtainedGlslcOutput = GetCompiledOutput(0);
            assert(obtainedGlslcOutput);
            reflectionHeader = ExtractReflectionSection(obtainedGlslcOutput);
            reflectionData = ((char*)(obtainedGlslcOutput)) + reflectionHeader->common.dataOffset;
        }
    }

    if (reflectionHeader && reflectionData) {
        ParseReflectionInfo(reflectionHeader, reflectionData);
    }

    return compileSuccess;
}

LWNboolean GLSLCHelper::LastCacheEntryHit()
{
    if (!GetAllowCacheRead()) {
        return LWN_FALSE;
    }

    return m_cacheParameters.m_lastCacheHit;
}

LWNboolean GLSLCHelper::UsesOverrideCacheKey()
{
    return m_cacheParameters.m_doCacheOverride;
}

void GLSLCHelper::SetOverrideCacheKey(const GLSLCHelperCacheKey * cacheKey)
{
    if (cacheKey) {
        m_cacheParameters.m_doCacheOverride = true;
        m_cacheParameters.m_overrideCacheKey = *cacheKey;
    } else {
        m_cacheParameters.m_doCacheOverride = false;
    }
}

GLSLCHelperCache::GLSLCHelperCache( LWNdevice * device )
{
    assert(device != NULL);

    m_cacheHeader.apiVersionMajor = GLSLC_API_VERSION_MAJOR;
    m_cacheHeader.apiVersionMinor = GLSLC_API_VERSION_MINOR;
    m_cacheHeader.gpuCodeVersionMajor = GLSLC_GPU_CODE_VERSION_MAJOR;
    m_cacheHeader.gpuCodeVersionMinor = GLSLC_GPU_CODE_VERSION_MINOR;

    // Set from the driver's max/min binary version numbers.
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_GLSLC_MAX_SUPPORTED_GPU_CODE_MAJOR_VERSION, &m_driverMaxBilwersionMajor);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_GLSLC_MIN_SUPPORTED_GPU_CODE_MAJOR_VERSION, &m_driverMinBilwersionMajor);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_GLSLC_MAX_SUPPORTED_GPU_CODE_MINOR_VERSION, &m_driverMaxBilwersionMinor);
    lwnDeviceGetInteger(device, LWN_DEVICE_INFO_GLSLC_MIN_SUPPORTED_GPU_CODE_MINOR_VERSION, &m_driverMinBilwersionMinor);

    // Initialize the cache binary data to the size of this header (no entries yet).
    m_cacheHeader.totalSize = sizeof(m_cacheHeader);
}

void GLSLCHelperCache::InitVersionCheck()
{
    int gpuCodeVersionMajor;
    int gpuCodeVersionMinor;
    bool versionCheck = true;
    m_isGpuVersionCompatible = false;
    m_isApiVersionCompatible = false;

    // Check if the GPU code version in the cache is compatible.
    gpuCodeVersionMajor = m_cacheHeader.gpuCodeVersionMajor;
    gpuCodeVersionMinor = m_cacheHeader.gpuCodeVersionMinor;

    if (!GLSLCLibraryHelper::GLSLCCheckGPUCodeVersion(m_driverMaxBilwersionMajor, m_driverMinBilwersionMajor,
                                                      m_driverMaxBilwersionMinor, m_driverMinBilwersionMinor,
                                                      gpuCodeVersionMajor, gpuCodeVersionMinor)) {
        versionCheck = false;
    }

    m_isGpuVersionCompatible = versionCheck;

    // Check API version
    versionCheck = true;

    int apiCodeVersionMajor = m_cacheHeader.apiVersionMajor;
    int apiCodeVersionMinor = m_cacheHeader.apiVersionMinor;

    if (apiCodeVersionMajor != GLSLC_API_VERSION_MAJOR ||
        apiCodeVersionMinor < GLSLC_API_VERSION_MINOR) {
        versionCheck = false;
    }

    m_isApiVersionCompatible = versionCheck;
}

GLSLCHelperCacheEntry * GLSLCHelperCache::Add(GLSLCHelperCacheKey cacheKey, const GLSLCHelperCacheEntry *data)
{
    // First see if there is an entry in the cache already using this key, and sanity check to make sure the
    // data is the same.
    GLSLCHelperCacheKeyMapType::iterator it = m_data.find(cacheKey);

    if (it != m_data.end()) {
        bool dataMismatch = (memcmp(data, it->second, data->header.totalSize) != 0);
        if (dataMismatch) {
            // Data should match if the key is already used in the cache.  Failure in this case indicates
            // an error with the hashing functions/cache data, or two shaders legitimately have the same
            // hash value (which should be rare).  In both cases, this assert will alert devs running
            // this path of the issue.
            assert(!"Data mismatch between cache entry and input data using same key.");
            // Remove the data found in the cache with the same hash so that consumers of the cache in
            // the future don't read different binaries (the wrong binaries) with the same hash value.
            m_data.erase(it);
            return NULL;
        }
        return it->second;
    }

    // Data is not in the cache, so add it.
    // Create a new GLSLCoutput and copy over the results since the original GLSLCoutput could be cleaned up
    // during the next compile.
    GLSLCHelperCacheEntry * newOutput = (GLSLCHelperCacheEntry *)calloc(1, data->header.totalSize);
    memcpy(newOutput, data, data->header.totalSize);
    m_data[cacheKey] = newOutput;

    // Add to the total size of the cache header, and increment the number of entries.
    m_cacheHeader.totalSize += data->header.totalSize;
    m_cacheHeader.numEntries++;

    return newOutput;
}

GLSLCHelperCacheEntry * GLSLCHelperCache::Get(GLSLCHelperCacheKey cacheKey)
{
    GLSLCHelperCacheKeyMapType::iterator it = m_data.find(cacheKey);

    if (it != m_data.end()) {
        // Data is already in the cache.
        return it->second;
    }

    // Couldn't find the data in the cache.
    return NULL;
}

// !!! DO NOT CHANGE THIS FUNCTION UNLESS YOU KNOW WHAT YOU ARE DOING !!!
//
// If newer versions require more things to be cached, then those need to be added to the appropriate
// version checks below to ensure shaders are cached according to the same API rules that the binaries
// contained in the cache were hashed.
//
// Any changes to this function has the potential to ilwalidate cache binaries generated by
// previous exelwtion of the applications.
//
// The process to add new entries to the cache is to create a version check below, hash that data,
// then add to the next free slot in the <shaderHashes> array.

// Indicates maximum number of unique values can contribute to each cache entry hash.
// We may want to add to this number for future GLSLC API versions, but don't want to ilwalidate
// the way previous versions are hashes, so this number should not change (since the number itself
// factors into the hash computation).
#define GLSLC_HELPER_CACHE_MAX_HASH_VALUES 32

// Minimum GLSLC major API version to be considered for caching.
#define GLSLC_MIN_API_FOR_CACHING 13

// Index values for where to store each value computation in the hash array used to compute
// the final hash value.
#define GLSLC_SHADER_STRING_HASH_INDEX 0
#define GLSLC_SHADER_OPTIONS_HASH_INDEX 1
#define GLSLC_SHADER_XFB_HASH_INDEX 2
#define GLSLC_SHADER_XFB_MAX_HASHES 32

GLSLCHelperCacheKey GLSLCHelperCache::HashShaders(GLSLCinput * input, GLSLCoptions * options)
{
    const char * shaders[GLSLC_NUM_SHADER_STAGES] = {NULL};
    uint64_t shaderStringHashes[GLSLC_NUM_SHADER_STAGES] = { 0 };
    uint32_t glslcApiVersionMajor = m_cacheHeader.apiVersionMajor;


    // We have 32 entries, where only the first few are used.
    uint64_t shaderHashes[GLSLC_HELPER_CACHE_MAX_HASH_VALUES] = { 0 };

    // Return value.
    uint64_t hash = 0;

    if (glslcApiVersionMajor < GLSLC_MIN_API_FOR_CACHING) {
        assert(!"GLSLC version is not compatible to use with caching.");
        return hash;
    }

    for (int i = 0; i < input->count; ++i) {
        assert(input->sources[i]);

        if (input->sources[i]) {
            shaders[input->stages[i]] = input->sources[i];
            shaderStringHashes[i] = Hash64((const unsigned char *)shaders[input->stages[i]], (uint32_t)(strlen(shaders[input->stages[i]]) + 1));
        }
    }

    shaderHashes[GLSLC_SHADER_STRING_HASH_INDEX] = Hash64((const unsigned char *)(shaderStringHashes), sizeof(uint64_t) * GLSLC_NUM_SHADER_STAGES);

    // Now hash the options bit flags
    shaderHashes[GLSLC_SHADER_OPTIONS_HASH_INDEX] = Hash64((const unsigned char*)(&options->optionFlags), sizeof(GLSLCoptionFlags));

    // Hash the XFB parameters.
    uint64_t xfbHashes[GLSLC_SHADER_XFB_MAX_HASHES] = { 0 };
    assert(options->xfbVaryingInfo.numVaryings <= GLSLC_SHADER_XFB_MAX_HASHES);
    for (uint32_t i = 0; i < options->xfbVaryingInfo.numVaryings; ++i) {
        xfbHashes[i] = Hash64((const unsigned char*)(options->xfbVaryingInfo.varyings[i]), (uint32_t)strlen(options->xfbVaryingInfo.varyings[i]));
    }
    shaderHashes[GLSLC_SHADER_XFB_HASH_INDEX] = Hash64((const unsigned char*)xfbHashes, sizeof(xfbHashes));

    // TODO: Add specialization hashes.

    //
    // All hashing is finished.  Compute the final hash value from the shaderHashes array.
    hash = Hash64((const unsigned char*)shaderHashes, sizeof(uint64_t) * GLSLC_HELPER_CACHE_MAX_HASH_VALUES);

    return hash;
}

bool GLSLCHelperCache::IsGPUVersionCompatible()
{
    return m_isGpuVersionCompatible;
}

bool GLSLCHelperCache::IsAPIVersionCompatible()
{
    return m_isApiVersionCompatible;
}

int GLSLCHelperCache::GetAPIMajorVersion()
{
    return m_cacheHeader.apiVersionMajor;
}

int GLSLCHelperCache::GetAPIMinorVersion()
{
    return m_cacheHeader.apiVersionMinor;
}

int GLSLCHelperCache::GetGPUCodeMajorVersion()
{
    return m_cacheHeader.gpuCodeVersionMajor;
}

int GLSLCHelperCache::GetGPUCodeMinorVersion()
{
    return m_cacheHeader.gpuCodeVersionMinor;
}

void * GLSLCHelperCache::CreateSerializedBinary (size_t * outSize)
{
    *outSize = 0;

    uint32_t totSize = sizeof(GLSLCCacheBinaryHeader);

    // compute total size.
    for (GLSLCHelperCacheKeyMapType::const_iterator cit = m_data.begin();
        cit != m_data.end();
            ++cit) {
        totSize += cit->second->header.totalSize;
    }

    assert(totSize == m_cacheHeader.totalSize);

    // Write an output containining:
    // 1. GLSLCCacheBinaryHeader
    // 2. <n> GLSCHelperCacheEntry sections

    // the cache entries should already have the magic number written as the first entry as part of the
    // entry header set up.
    char * retVal = NULL;
    retVal = (char *)malloc(m_cacheHeader.totalSize);

    char *lwrrOffset = retVal;

    memcpy(lwrrOffset, &m_cacheHeader, sizeof(GLSLCCacheBinaryHeader));

    lwrrOffset += sizeof(GLSLCCacheBinaryHeader);
    for (GLSLCHelperCacheKeyMapType::const_iterator cit = m_data.begin();
            cit != m_data.end();
            ++cit ) {
        GLSLCHelperCacheEntryHeader dataHeader = cit->second->header;
        memcpy(lwrrOffset, (cit->second), dataHeader.totalSize);
        lwrrOffset += cit->second->header.totalSize;
    }

    assert((lwrrOffset - retVal) == ptrdiff_t(m_cacheHeader.totalSize));

    *outSize = m_cacheHeader.totalSize;
    return (void *)(retVal);
}

void GLSLCHelperCache::DestroySerializedBinary(void * binary)
{
    free(binary);
}

const GLSLCCacheBinaryHeader * GLSLCHelperCache::SetFromSerializedBinary (const char * inBin, size_t size)
{
    (void) size;        // unused parameter
    const GLSLCCacheBinaryHeader * header = (const GLSLCCacheBinaryHeader *)inBin;
    m_cacheHeader = *header;

    if (m_data.size() != 0) {
        // Free any previous results
        FreeCache();
    }

    const GLSLCHelperCacheEntry * lwrrBin = (const GLSLCHelperCacheEntry *)(inBin + sizeof(GLSLCCacheBinaryHeader));

    // Insert the entries from the binary into the cache.
    for (uint32_t i = 0; i < header->numEntries; ++i) {
        const GLSLCHelperCacheEntryHeader * lwrrHeader = (const GLSLCHelperCacheEntryHeader *)lwrrBin;

        // Check to make sure this is a real entry by checking the magic number.
        assert(lwrrHeader->magic == CACHE_ENTRY_MAGIC_NUMBER);
        if (lwrrHeader->magic != CACHE_ENTRY_MAGIC_NUMBER) {
            return NULL;
        }

        uint64_t headerHash = 0;
        headerHash |= (uint64_t)(lwrrHeader->hashLo);
        headerHash |= ((uint64_t)(lwrrHeader->hashHi) << 32);

        Add(headerHash, lwrrBin);

        lwrrBin = (GLSLCHelperCacheEntry *)(((char*)lwrrBin) + lwrrHeader->totalSize);
    }

    // Check for API version or GPU version incompatibities, and set the appropriate flags to signal that case.
    InitVersionCheck();

    return header;
}

// Hash function and parameters.
// !!! DO NOT CHANGE THESE !!!
// Changing these would ilwalidate any previously serialized cached binaries.
static inline void mix64(uint64_t &a, uint64_t &b, uint64_t &c)
{
    a -= b; a -= c; a ^= (c >> 43);
    b -= c; b -= a; b ^= (a << 9);
    c -= a; c -= b; c ^= (b >> 8);
    a -= b; a -= c; a ^= (c >> 38);
    b -= c; b -= a; b ^= (a << 23);
    c -= a; c -= b; c ^= (b >> 5);
    a -= b; a -= c; a ^= (c >> 35);
    b -= c; b -= a; b ^= (a << 49);
    c -= a; c -= b; c ^= (b >> 11);
    a -= b; a -= c; a ^= (c >> 12);
    b -= c; b -= a; b ^= (a << 18);
    c -= a; c -= b; c ^= (b >> 22);
}

// A known good hash function.
uint64_t GLSLCHelperCache::Hash64(const unsigned char *k, uint32_t length)
{
    uint64_t a, b, c;
    unsigned int len;

    // Set up the internal state
    len = length;
    a = b = 0;
    c = 0x9e3779b97f4a7c13ull; // the golden ratio; an arbitrary value

    // handle most of the key
    while (len >= 24) {
        a += ((uint64_t)k[ 0]      + ((uint64_t)k[ 1]<< 8) + ((uint64_t)k[ 2]<<16) + ((uint64_t)k[ 3]<<24)
          +  ((uint64_t)k[ 4]<<32) + ((uint64_t)k[ 5]<<40) + ((uint64_t)k[ 6]<<48) + ((uint64_t)k[ 7]<<56));
        b += ((uint64_t)k[ 8]      + ((uint64_t)k[ 9]<< 8) + ((uint64_t)k[10]<<16) + ((uint64_t)k[11]<<24)
          +  ((uint64_t)k[12]<<32) + ((uint64_t)k[13]<<40) + ((uint64_t)k[14]<<48) + ((uint64_t)k[15]<<56));
        c += ((uint64_t)k[16]      + ((uint64_t)k[17]<< 8) + ((uint64_t)k[18]<<16) + ((uint64_t)k[19]<<24)
          +  ((uint64_t)k[20]<<32) + ((uint64_t)k[21]<<40) + ((uint64_t)k[22]<<48) + ((uint64_t)k[23]<<56));

        mix64(a,b,c);
        k += 24; len -= 24;
    }

    // handle the last 23 bytes
    c += length;
    switch (len) {
    case 23: c += ((uint64_t)k[22]<<56);
    case 22: c += ((uint64_t)k[21]<<48);
    case 21: c += ((uint64_t)k[20]<<40);
    case 20: c += ((uint64_t)k[19]<<32);
    case 19: c += ((uint64_t)k[18]<<24);
    case 18: c += ((uint64_t)k[17]<<16);
    case 17: c += ((uint64_t)k[16]<<8);
    // the first byte of c is reserved for the length
    case 16: b += ((uint64_t)k[15]<<56);
    case 15: b += ((uint64_t)k[14]<<48);
    case 14: b += ((uint64_t)k[13]<<40);
    case 13: b += ((uint64_t)k[12]<<32);
    case 12: b += ((uint64_t)k[11]<<24);
    case 11: b += ((uint64_t)k[10]<<16);
    case 10: b += ((uint64_t)k[ 9]<<8);
    case  9: b += ((uint64_t)k[ 8]);
    case  8: a += ((uint64_t)k[ 7]<<56);
    case  7: a += ((uint64_t)k[ 6]<<48);
    case  6: a += ((uint64_t)k[ 5]<<40);
    case  5: a += ((uint64_t)k[ 4]<<32);
    case  4: a += ((uint64_t)k[ 3]<<24);
    case  3: a += ((uint64_t)k[ 2]<<16);
    case  2: a += ((uint64_t)k[ 1]<<8);
    case  1: a += ((uint64_t)k[ 0]);
    // case 0: nothing left to add
    }

    mix64(a,b,c);

    return c;
}

void GLSLCHelperCache::FreeCache()
{
    // These were created during the test run.
    for (GLSLCHelperCacheKeyMapType::iterator it = m_data.begin(); it != m_data.end(); ++it) {
        // Delete the binaries.
        free(it->second);
    }

    m_data.clear();
    m_cacheHeader = GLSLCCacheBinaryHeader();
}

GLSLCHelperCache::~GLSLCHelperCache()
{
    FreeCache();
}

} //namespace lwnUtil

#endif // #ifndef __lwnUtil_GlslcHelperImpl_h__
