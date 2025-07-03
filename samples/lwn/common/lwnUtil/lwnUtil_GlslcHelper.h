/*
 * Copyright (c) 2015-2020 LWPU Corporation.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from LWPU Corporation is strictly prohibited.
 */

#ifndef __lwnUtil_GlslcHelper_h__
#define __lwnUtil_GlslcHelper_h__

#include "lwnUtil_Interface.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#ifdef DXC_ENABLED
#include <unknwn.h>
#include "dxcapi.use.h"
#endif // DXC_ENABLED
#endif

#include "stddef.h"

#include "lwnTool/lwnTool_GlslcInterface.h"

#include <map>
#include <vector>

#include "lwnUtil/lwnUtil_PoolAllocator.h"

// At the beginning of all GLSLCCacheEntry structures in the serialized cache binary.
#define CACHE_ENTRY_MAGIC_NUMBER 0x14842929

// GLSLC function type definitions.
typedef uint8_t (*GLSLCCOMPILEFUNCTION)(GLSLCcompileObject *);
typedef uint8_t (*GLSLCINITIALIZEFUNCTION)(GLSLCcompileObject *);
typedef void (*GLSLCFINALIZEFUNCTION)(GLSLCcompileObject *);
typedef GLSLCversion (*GLSLCGETVERSIONFUNCTION)();
typedef bool(*GLSLCCOMPILEPRESPECIALIZEDFUNCTION)(GLSLCcompileObject *);
typedef const GLSLCoutput * const * (*GLSLCCOMPILESPECIALIZEDFUNCTION)(GLSLCcompileObject *, const GLSLCspecializationBatch *);
typedef GLSLCoptions (*GLSLCGETDEFAULTOPTIONSFUNCTION)();
typedef GLSLCresults const * const * (*GLSLCCOMPILESPECIALIZEDMTFUNCTION) (const GLSLCcompileObject * compileObject, const GLSLCspecializationBatch * specEntries);
typedef void (*GLSLCFREESPECIALIZEDRESULTSMTFUNCTION)(GLSLCresults const * const * specResults);
typedef uint8_t (*GLSLCCOMPARECONTROLSECTIONSFUNCTION)(const void *, const void *);
typedef uint8_t (*GLSLCGETDEBUGDATAHASHFUNCTION) (const void *, GLSLCdebugDataHash *);
typedef uint8_t (*GLSLCSETDEBUGDATAHASHFUNCTION) (void *, const GLSLCdebugDataHash *);

// Platform dependent function to get entry points from the LWN library.
typedef void * (*GLSLCGETLWNPROCADDRESS)(const char *);

// Resource type definition from the online compilation path.
typedef enum LWNprogramResourceType {
    LWN_PROGRAM_RESOURCE_TYPE_VERTEX_ATTRIB = 0,
    LWN_PROGRAM_RESOURCE_TYPE_SAMPLER = 1,
    LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_BLOCK = 2,
    LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_BLOCK_SIZE = 3,
    LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_OFFSET = 4,
    LWN_PROGRAM_RESOURCE_TYPE_IMAGE = 5,
    LWN_PROGRAM_RESOURCE_TYPE_STORAGE_BLOCK = 6,
    LWN_PROGRAM_RESOURCE_TYPE_STORAGE_BLOCK_SIZE = 7,
    LWN_PROGRAM_RESOURCE_TYPE_BUFFER_VARIABLE_OFFSET = 8,
    LWN_PROGRAM_RESOURCE_TYPE_UNIFORM_ARRAY_STRIDE = 9,
    LWN_ENUM_32BIT(PROGRAM_RESOURCE_TYPE),
} LWNprogramResourceType;

namespace lwnUtil {

class ProgramResourceType {
public:
    enum Enum {
        VERTEX_ATTRIB = 0,
        SAMPLER = 1,
        UNIFORM_BLOCK = 2,
        UNIFORM_BLOCK_SIZE = 3,
        UNIFORM_OFFSET = 4,
        IMAGE = 5,
        STORAGE_BLOCK = 6,
    };
private:
    Enum m_value;
public:
    ProgramResourceType(Enum value) : m_value(value) {}
    ProgramResourceType(LWNprogramResourceType value) : m_value(Enum(value)) {}
    operator LWNprogramResourceType() const { return LWNprogramResourceType(m_value); }
};

// Data types for the GLSLC helper binary cache
typedef uint64_t GLSLCHelperCacheKey;
struct GLSLCHelperCacheEntry;
typedef std::map<GLSLCHelperCacheKey, GLSLCHelperCacheEntry *> GLSLCHelperCacheKeyMapType;

// Parameters for using SPIR-V as input into GLSLC.  These parameters correspond to the shader
// sources arrays used in various GLSLChelper compilation routines.
struct SpirvParams {
    SpirvParams() {
        memset(entryPoints, 0, sizeof(entryPoints));
        memset(sizes, 0, sizeof(sizes));
    }

    // List of entry points.  If these are NULL, the default "main" will be used by the GLSLC
    // DLL.
    const char *entryPoints[GLSLC_NUM_SHADER_STAGES];

    // Sizes of each spirv binary module.
    uint32_t sizes[GLSLC_NUM_SHADER_STAGES];
};

// The header data for each cache entry in the GLSLC helper binary cache.
struct GLSLCHelperCacheEntryHeader {
    GLSLCHelperCacheEntryHeader() :
        magic(CACHE_ENTRY_MAGIC_NUMBER), cacheVersion(0), totalSize(0),
                                         hashLo(0), hashHi(0), offsetGlslcOutput(0)
    {
        memset(reserved, 0, sizeof(reserved));
    }

    // A magic number indicating this is a cache entry when reading from a serialized cache binary.
    uint32_t magic;

    // Simple versioning info.  If this structure is extended in the future, this will allow
    // the cache to determine if the cached contents have the same format.
    // TODO: Lwrrently this is unused.
    uint32_t cacheVersion;

    // Total size of the cache entry, including this header.  This header is followed by the actual
    // cache data in memory.
    uint32_t totalSize;

    // Hash belonging to this shader/options combination.
    // 32-bit definitions here as not to affect padding/alignment of this structure.
    uint32_t hashLo;
    uint32_t hashHi;

    // Offset of the GLSLCoutput into the data section, in bytes.
    // TODO: Right now always 0.  In the future, we may also have option entries like shader source, etc.
    uint32_t offsetGlslcOutput;

    // Reserved for future entries.
    // TODO: Future entries will include:
    // * Offset for shader source (optional output).
    // * Offset for options (optional output).
    // * Offset for specialization param (optional output).
    // * etc.
    uint32_t reserved[32];
};

struct GLSLCHelperCacheEntry {
    // Header of the cache entry describing the data.
    GLSLCHelperCacheEntryHeader header;

    // GLSLCoutput.  The actual entry will be in malloc'ed data with the GLSLCoutput data appended
    // to the end of this structure.
    // The header.totalSize member can be used to determine how much data is reserved for this entry.
    GLSLCoutput output[1];
};

// Header for this cache, including the GLSLC versions of the entries it contains.
// This header also will be embedded at the beginning of the GLSLC cache binary.
struct GLSLCCacheBinaryHeader {
    GLSLCCacheBinaryHeader() : apiVersionMajor(0), apiVersionMinor(0), gpuCodeVersionMajor(0),
                               gpuCodeVersionMinor(0), totalSize(0), numEntries(0)
    {
        memset(reserved, 0, sizeof(reserved));
    }

    // These are stored when written by the cache.  When reading the cache binary, these are checked against the
    // versions supported by the driver.
    int apiVersionMajor;
    int apiVersionMinor;
    int gpuCodeVersionMajor;
    int gpuCodeVersionMinor;

    // Total size of all cache entries + the size required for this header.
    uint32_t totalSize;

    // Number of entries in the cache/binary.
    uint32_t numEntries;

    // In case we need to add more entries but don't want to ilwalidate already-generated caches.
    uint32_t reserved[32];
};

// GLSLCHelper cache structure
struct GLSLCHelperCache {
    explicit GLSLCHelperCache( LWNdevice * device );

    // Adds an entry to the cache.  If the entry doesn't exist in the cache yet, will create a new
    // entry in the cache and return that entry.  If the entry already exists in the cache, will
    // just return the cached entry without modifying the cache.
    GLSLCHelperCacheEntry * Add(GLSLCHelperCacheKey cacheKey, const GLSLCHelperCacheEntry *data);

    // Finds and returns the entry in the cache corresponding to <cacheKey>.
    // If the cacheKey doesn't have an entry in the cache, NULL is returned.
    GLSLCHelperCacheEntry * Get(GLSLCHelperCacheKey cacheKey);

    // Create a cache key from the input shaders/options.
    GLSLCHelperCacheKey HashShaders(GLSLCinput * input, GLSLCoptions * options);

    // Return whether the GPU code version in the binary cache is compatible with the driver.
    bool IsGPUVersionCompatible();

    // Return whether the API version which is compiled into the application (specified inside lwnTool_GlslcInterface.h)
    // is compatible with the version being #included in this helper.
    bool IsAPIVersionCompatible();

    // Return the API major version used in the binary cache.
    int GetAPIMajorVersion();

    // Return the API minor version used in the binary cache.
    int GetAPIMinorVersion();

    // Return the GPU major version used in the binary cache.
    int GetGPUCodeMajorVersion();

    // Return the GPU minor version used in the binary cache.
    int GetGPUCodeMinorVersion();

    // Return a serialized version of the GLSLC binary cache.  This can be saved to disk
    // and further loaded up with SetFromSerializedBinary() to initialize the cache.
    // <outSize> will contain the size of the binary.
    void * CreateSerializedBinary (size_t * outSize);

    // Frees the memory backing the binary returned from the CreateSerializedBinary call.
    void DestroySerializedBinary(void * binary);

    // Initializes the cache from a serialized binary obtained from CreateSerializedBinary.
    // Should return NULL if a failure, or !NULL if success.
    const GLSLCCacheBinaryHeader *SetFromSerializedBinary (const char * inBin, size_t size);

    // A known good hash function.
    uint64_t Hash64(const unsigned char *k, uint32_t length);

    ~GLSLCHelperCache();
private:
    // Query and initialize the flags indicating whether the cache is compatible with the driver's
    // GLSLC GPU code version and API version.  When a cache is obtained from disk, this function
    // is used to check whether the API version and the GPU code version in the binary are compatible
    // with the current driver.  This function will set various flags internally to indicate whether
    // the cache can be used based on this compatibility.
    // This method will be called during the GLSLCHelperCache initialization phase.
    void InitVersionCheck();


    // A header for the cache, used to store/read common properties of the cache entries.
    GLSLCCacheBinaryHeader m_cacheHeader;

    // The actual cache key-value pairs.
    GLSLCHelperCacheKeyMapType m_data;

    // Flag indicating whether the cache is compatible with the current GLSLC API or GPU code version.
    bool m_isGpuVersionCompatible;
    bool m_isApiVersionCompatible;

    // Variables from the driver about supported GPU code versions.  These are initialized at construction.
    int m_driverMaxBilwersionMinor;
    int m_driverMaxBilwersionMajor;
    int m_driverMinBilwersionMinor;
    int m_driverMinBilwersionMajor;

    // Free the cache entries and resets the cache.
    void FreeCache();
};

typedef void (*GLSLCLoggerFunction)(const char * format, va_list arg);

class GLSLCLogger {
public:
    // GLSLCLogger initialized with logging disabled.
    GLSLCLogger() : m_loggerFunction(NULL), m_enabled(false) {}

    // Enable or disable logging by the logger.  If disabled, the Log function does nothing.
    void SetEnable(bool enable);

    // Returns if logging is enabled or not.
    bool IsEnabled();

    // Specify the logger function
    void SetLoggerFunction(GLSLCLoggerFunction funcPtr);

    // Log an output message using the logger function provided.
    // Nothing will happen if logging is disabled (m_enabled == false).
    // The global define GLSLC_LOGGER_FORCE_ENABLE can be set to 1 or 0 to override the GLSLCLogger's
    // set enabled flag.
    // The global flag GLSLC_LOGGER_FORCE_PRINTF can be set to use the "printf"
    // function instead of calling the m_loggerFunction pointer.
    void Log(const char *format, ...);
private:
    GLSLCLoggerFunction m_loggerFunction;
    bool m_enabled;
};

// Helper class for comparing GLSLC GPU code versions.
struct GLSLCGpuCodeVersionInfo {
    int gpuCodeVersionMajor;
    int gpuCodeVersionMinor;

    GLSLCGpuCodeVersionInfo() {}
    GLSLCGpuCodeVersionInfo(int major, int minor) :
        gpuCodeVersionMajor(major), gpuCodeVersionMinor(minor) {}

    GLSLCGpuCodeVersionInfo(const GLSLCversion &glslcVersion) :
        gpuCodeVersionMajor(glslcVersion.gpuCodeVersionMajor),
        gpuCodeVersionMinor(glslcVersion.gpuCodeVersionMinor) {}

    inline bool operator < (const GLSLCGpuCodeVersionInfo &other) const
    {
        if (gpuCodeVersionMajor < other.gpuCodeVersionMajor) {
            return true;
        } else if (gpuCodeVersionMajor == other.gpuCodeVersionMajor) {
            return gpuCodeVersionMinor < other.gpuCodeVersionMinor;
        } else {
            return false;
        }
    }
    inline bool operator <= (const GLSLCGpuCodeVersionInfo &other) const
    { return !(other < *this); }
    inline bool operator > (const GLSLCGpuCodeVersionInfo &other) const
    { return other < *this; }
    inline bool operator >= (const GLSLCGpuCodeVersionInfo &other) const
    { return !(*this < other); }
};

// Helper class that loads the DXC library.
class DXCLibraryHelper {
public:
    DXCLibraryHelper();
    ~DXCLibraryHelper();

    bool IsLoaded() { return initialized; }
#ifdef DXC_ENABLED
    LPCWSTR GetHLSLShaderStage(unsigned int shaderStage, LPCWSTR & shaderKind);
    char * DxcCompile(const char* hlslSourceStr, int lwrShaderIndex,
        LWNshaderStage shaderType, uint32_t * spirvBinSize, GLSLCLogger * logger);
#endif // DXC_ENABLED

private:
    bool initialized;
#ifdef DXC_ENABLED
    dxc::DxcDllSupport dllHelper;
    IDxcCompiler *m_pCompiler;
    IDxcLibrary  *m_pLibrary;
#endif // DXC_ENABLED
};

// Helper class that loads the GLSLC library and sets up the internal function
// pointers from the library.
class GLSLCLibraryHelper {
public:

    GLSLCLibraryHelper();
    ~GLSLCLibraryHelper();

    // Load library specified via the DLL file name.
    // If any of the GLSLC DLL could not be loaded, include the function
    // entry points, then the entire operation is unrolled and the library
    // is unloaded.
    bool LoadDLL( const char * DLLFileName );

    // Was the library loaded successfully.
    bool IsLoaded();

    GLSLCversion GetVersion();

    // Return the logger associated with this class.
    GLSLCLogger * GetLogger();

    // Helper function to compare whether the application's API version (specified via the appApi<Major/Minor>Version parameters)
    // is compatible with the API version reported from the GLSLC library (specified via the glslcApi<Major/Minor>Version parameters)
    static bool GLSLCCheckAPIVersion(int appApiMajorVersion, int appApiMinorVersion,
                                     int glslcApiMajorVersion, int glslcApiMinorVersion) {
        // An application is compatible with the GLSLC API only if the major version is the same, and the application's minor version
        // is <= GLSLC's API minor version.
        if (glslcApiMajorVersion != appApiMajorVersion ||
            glslcApiMinorVersion < appApiMinorVersion) {
            return false;
        }

        return true;
    }

    // Helper function to compare whether the GPU version reported by the GLSLC library (specified via the glslcGpu<Major/Minor>
    // parameters) is compatible with the driver's maximum/minimum supported GPU versions (specified as <max/min>SupportedGpu<Major/Minor>
    static bool GLSLCCheckGPUCodeVersion(int maxSupportedGpuMajor, int minSupportedGpuMajor,
                                         int maxSupportedGpuMinor, int minSupportedGpuMinor,
                                         int glslcGpuMajor, int glslcGpuMinor)
    {
        GLSLCGpuCodeVersionInfo glslcGpuCodeVersion(glslcGpuMajor, glslcGpuMinor);

        // If the GLSLC's gpu code binary version is a.b, minimum supported binary version is c.d, and maximum
        // supported version is e.f, then if a.b >= c.d && a.b <= e.f, the version is supported
        return (glslcGpuCodeVersion >= GLSLCGpuCodeVersionInfo(minSupportedGpuMajor, minSupportedGpuMinor) &&
                glslcGpuCodeVersion <= GLSLCGpuCodeVersionInfo(maxSupportedGpuMajor, maxSupportedGpuMinor));
    }


    // Function entries from the GLSLC DLL
    GLSLCCOMPILEFUNCTION glslcCompile;
    GLSLCINITIALIZEFUNCTION glslcInitialize;
    GLSLCFINALIZEFUNCTION glslcFinalize;
    GLSLCGETVERSIONFUNCTION glslcGetVersion;
    GLSLCCOMPILEPRESPECIALIZEDFUNCTION glslcCompilePreSpecialized;
    GLSLCCOMPILESPECIALIZEDFUNCTION glslcCompileSpecialized;
    GLSLCGETDEFAULTOPTIONSFUNCTION glslcGetDefaultOptions;
    GLSLCCOMPILESPECIALIZEDMTFUNCTION glslcCompileSpecializedMT;
    GLSLCFREESPECIALIZEDRESULTSMTFUNCTION glslcFreeSpecializedResultsMT;
    GLSLCCOMPARECONTROLSECTIONSFUNCTION glslcCompareControlSections;
    GLSLCGETDEBUGDATAHASHFUNCTION glslcGetDebugDataHash;
    GLSLCSETDEBUGDATAHASHFUNCTION glslcSetDebugDataHash;

private:
    GLSLCversion libraryVersion;
#if defined(_WIN32) && defined(GLSLC_LIB_DYNAMIC_LOADING)
    HMODULE glslcHMod;
#endif

    GLSLCLogger logger;

    // Unloads the DLL functions by setting them to NULL.  After a call
    // to this function, GLSLCLibraryHelper::IsLoaded() will return false.
    void UnloadDLLFunctions();
};

// This class provides some colwenience methods for parsing the GLSLCoutput structure.
// When CompileShaders or CompileAndSetShaders is called, the last GLSLC
// compile object is retained so that it can be queried.  Once another
// CompileShaders or CompileAndSetShaders is called, the previous compile
// results are overwritten with new ones.
class GLSLCHelper {
public:
    // Used to map names of variables to the program query values.
    struct StringCompare {
       bool operator()(const char * str1, const char * str2) const
       {
           return strcmp(str1, str2) < 0;
       }
    };

    // Used when indexing into the reflection map.
    enum ReflectionTypeEnum {
        REFLECTION_TYPE_UNIFORM = 0,
        REFLECTION_TYPE_SAMPLER,
        REFLECTION_TYPE_IMAGE,
        REFLECTION_TYPE_UNIFORM_BLOCK,
        REFLECTION_TYPE_STORAGE_BLOCK,
        REFLECTION_TYPE_VERTEX_ATTRIB,
        REFLECTION_TYPE_BUFFER_VARIABLE,
        REFLECTION_TYPE_SUBROUTINE_VERTEX,
        REFLECTION_TYPE_SUBROUTINE_FRAGMENT,
        REFLECTION_TYPE_SUBROUTINE_TESS_CONTROL,
        REFLECTION_TYPE_SUBROUTINE_TESS_EVALUATION,
        REFLECTION_TYPE_SUBROUTINE_GEOMETRY,
        REFLECTION_TYPE_SUBROUTINE_COMPUTE,
        REFLECTION_TYPE_SUBROUTINE_UNIFORM_VERTEX,
        REFLECTION_TYPE_SUBROUTINE_UNIFORM_FRAGMENT,
        REFLECTION_TYPE_SUBROUTINE_UNIFORM_TESS_CONTROL,
        REFLECTION_TYPE_SUBROUTINE_UNIFORM_TESS_EVALUATION,
        REFLECTION_TYPE_SUBROUTINE_UNIFORM_GEOMETRY,
        REFLECTION_TYPE_SUBROUTINE_UNIFORM_COMPUTE,
        REFLECTION_TYPE_ILWALID
    };

    // Used when determining if the scratch memory allocation reported to the
    // GLSLCHelper via SetShaderScratchMemory is sufficient to handle the
    // shaders being programmed by the helper.
    enum ScratchMemCheckEnum {
        SCRATCH_MEM_CHECK_SUFFICIENT = 0,
        SCRATCH_MEM_CHECK_THROTTLE = 1,
        SCRATCH_MEM_CHECK_INSUFFICIENT = 2
    };

    typedef std::map<const char *, const void *, StringCompare> InterfaceMemberMapType;
    typedef std::map<ReflectionTypeEnum, InterfaceMemberMapType > PiqMapType;

    // The maximum number of specialization arrays that can be used in a batch specialization
    // compile.
    enum {
        MAX_SPEC_ARRAYS=8
    };

    GLSLCHelper(LWNdevice * device, size_t maxGPUMemory, GLSLCLibraryHelper * libraryHelper,
        GLSLCHelperCache * cache = NULL, DXCLibraryHelper * dxcLibraryHelper = NULL);

#define GLSLC_CHIP_OVERRIDE_ARCH_TURING 352
#define GLSLC_CHIP_OVERRIDE_IMPL_TURING 2

    // Overrides the device that GLSLC compiles code for.  This is an internal-only feature and
    // requires a special build of GLSLC.
    void OverrideChip(unsigned int arch, unsigned int impl);

    // Overrides the Glslang options that GLSLC uses.  This is an internal-only feature.
    void OverrideGlslangOptions(uint8_t doGlslangShim, uint8_t glslangFallbackOnError, uint8_t glslangFallbackOnAbsolute);

    // Returns true if we are either reading or writing with the cache.
    LWNboolean UsesCache();

    // Sets reading from the cache to be enabled.
    void SetAllowCacheRead(LWNboolean);

    // Gets true if we are reading from the cache.
    LWNboolean GetAllowCacheRead();

    // Sets writing to the cache to be enabled.
    void SetAllowCacheWrite(LWNboolean);

    // Gets whether we are writing to the cache.
    LWNboolean GetAllowCacheWrite();

    // Compile the shaders and call lwnProgramSetShaders.  If <spvParams> is non-NULL, <shaderSources>
    // is assumed to point to SPIR-V binary modules.
    LWNboolean CompileAndSetShaders(LWNprogram *, const LWNshaderStage * stages, uint32_t count,
                                    const char ** shaderSources, SpirvParams * spirvParams = NULL);

    // Compile the HLSL shaders and call lwnProgramSetShaders.
    LWNboolean CompileAndSetShadersHLSL(LWNprogram *program, const LWNshaderStage *stages, uint32_t count,
        const char ** shaderSources);
    LWNboolean CompileHlslToSpirv(uint32_t nShaders, const LWNshaderStage * stages, const char ** hlslSources,
        char ** spirvBins, uint32_t * spirvBinSizes);

    // Colwenience functions for setting options.
    void EnablePerfStats(LWNboolean outputPerfStats);
    void SetSeparable(LWNboolean isSeparable);
    void SetTransformFeedbackVaryings(uint32_t count, const char **);
    void AddIncludePath(const char *path);
    void AddforceIncludeStdHeader(const char *path);
    void EnableThinBinaries( LWNboolean outputThinGpuBinaries );
    void EnableTessellationAndPassthroughGS(LWNboolean enable);
    void SetDebugLevel(GLSLCdebugInfoLevelEnum debugLevel);
    void SetOptLevel(GLSLCoptLevelEnum optLevel);
    void EnableSassDump(LWNboolean enable);
    void EnableCBF(LWNboolean enable);
    void EnableWarpLwlling(LWNboolean enable);
    void EnableMultithreadCompilation(LWNboolean enable);

    void Reset();
    void ResetOptions();

    // Retrieves the info log from the GLSLC compile object.
    const char * GetInfoLog();

    // Compile only, useful for testing the GLSLC interface.  Most
    // tests won't use this directly.  If <spvParams> is non-NULL, <shaderSources> is assumed
    // to point to SPIR-V binary modules.
    LWNboolean CompileShaders(const LWNshaderStage *, uint32_t count, const char ** shaderSources,
                              SpirvParams * spvParams = NULL);

    // Compiles only the front-end portion of the shader (using glslcCompilePreSpecialized).  The compile object
    // can then be obtained for use in back-end compiles using glslcCompileSpecialized.
    LWNboolean CompileShadersPreSpecialized(const LWNshaderStage * stages,
                                            uint32_t count, const char ** shaderSources);

    // Useful for validating the GLSLC output structure.  Most tests
    // won't use this.
    const GLSLCoutput * GetGlslcOutput();

    // Returns query information about individual resources in the GLSL shaders.  This information is
    // obtained from the GLSLC reflection information.
    int32_t ProgramGetResourceLocation(LWNprogram * programLWN, LWNshaderStage stage, LWNprogramResourceType type, const char * name);
    
    // Enable Piq support.  This tells GLSLC to output the reflection information during compile, and
    // must be true if using "ExtractReflectionSection" to query variables.
    // This option is set to "true" internally by GLSLCHelper by default.
    void SetEnablePiqSupport( LWNboolean enablePiq );

    // Extract the perf statistics corresponding to stage <stage>.  Most
    // tests won't use this.
    static const GLSLCperfStatsHeader * ExtractPerfStatsSection(const GLSLCoutput * glslcOutput, LWNshaderStage stage);

    // Give the relevant gpu code header section, extract the perf statistics. Most tests won't use this.
    static const GLSLCperfStatsHeader * ExtractPerfStatsSection(const GLSLCoutput * glslcOutput, const GLSLCgpuCodeHeader *);

    // Extract the GPU code section corresponding to stage <stage>
    // Most tests won't use this.
    static const GLSLCgpuCodeHeader * ExtractGpuCodeSection(const GLSLCoutput * glslcOutput, LWNshaderStage stage);

    // Extract the program reflection section from the GLSLCoutput.  This is required if any program
    // interface query support is needed in the individual tests.
    // If no reflection section is present in the GLSLC output, NULL is returned.
    static const GLSLCprogramReflectionHeader * ExtractReflectionSection(const GLSLCoutput * glslcOutput);

    // Registered shader scratch memory with the GLSLC helper.  If <cmdBuf> is non-NULL,
    // this internally calls lwnCommandBufferSetShaderScratchMemory.
    LWNboolean SetShaderScratchMemory(LWNmemoryPool *memPool, ptrdiff_t offset, size_t size, LWNcommandBuffer * cmdBuf);

    // Determines if the registered scratch memory is sufficient to handle the shaders in <glslcOutput>.
    ScratchMemCheckEnum CheckScratchMem(const GLSLCoutput * glslcOutput);

    // Returns the amount of memory required per warp (32 threads) to run the shaders in <glslcOutput>.
    size_t GetScratchMemoryPerWarp(const GLSLCoutput * glslcOutput);

    // Returns the recommeded amount of memory to run the shaders in <glslcOutput> on <device>.
    size_t GetScratchMemoryRecommended(const LWNdevice *device, const GLSLCoutput * glslcOutput);

    // Returns the minimum amount of memory to run the shaders potentially at throttled performance in <glslcOutput>.
    size_t GetScratchMemoryMinimum(const GLSLCoutput * glslcOutput);

    // Add a new GLSLCspecializationUniform to the specialization array <index>.  There
    // can be a total of MAX_SPEC_ARRAYS specialization arrays used for batch specialization compiles,
    // and <index> < MAX_SPEC_ARRAYS.
    void AddSpecializationUniform(uint32_t index, const GLSLCspecializationUniform * uniform);

    // Clear the specialization array denoted by the index <index>.
    void ClearSpecializationUniformArray(uint32_t index);

    // Clear all specialization arrays 0 through MAX_SPEC_ARRAYS.
    void ClearSpecializationUniformArrays();

    // Add a new constantID with data to the spirv specialization constant array for the stage <stage>.
    void AddSpirvSpecializationConstant(LWNshaderStage stage, uint32_t constantID, uint32_t data);

    // Clear the spirv specialization constant array denoted by the stage <stage>.
    void ClearSpirvSpecializationConstantArray(LWNshaderStage stage);

    // Clear all spirv specialization constant arrays 0 through GLSLC_NUM_SHADER_STAGES.
    void ClearSpirvSpecializationConstantArrays();

    // Retrieves the GLSLCoutput from the last list of outputs from the last compile call.
    // If specialization is used, 0 <= <index> < MAX_SPEC_ARRAYS.  If specialization is not used,
    // <index> must be 0.
    // NULL will be returned if <index> >= GetNumCompiledOutputs().
    const GLSLCoutput * GetCompiledOutput(uint32_t index) const;

    // Get the number of compiled outputs from the last compile call.
    uint32_t GetNumCompiledOutputs() const;

    // GLSLC version of LWN's SetShaders.  This will take the compiled programs,
    // load them into GPU memory, and call lwnProgramSetShaders to set the data.
    LWNboolean SetShaders(LWNprogram * program, const GLSLCoutput * glslcOutput);

    GLSLCLogger * GetLogger();

    // Get a GLSLCsubroutineInfo corresponding to the input <stage> and <name>.  If a subroutine of that
    // <name> is not found in the list of subroutines for the <stage>, NULL is returned.
    const GLSLCsubroutineInfo * GetSubroutineInfo(LWNshaderStage stage, const char *name) const;

    // Get a GLSLCsubroutineUniformInfo corresponding to the input <stage> and <name>.  If a subroutine uniform
    // of that <name> is not found in the list of subroutine uniforms for the <stage>, NULL is returned.
    const GLSLCsubroutineUniformInfo * GetSubroutineUniformInfo(LWNshaderStage stage, const char *name) const;

    // Gets a pointer to the linkage map to be used in the lwnCommandBufferSetProgramSubroutines function.
    // compiledOutputNdx - should be 0, unless using specialization in which case it should be whichever specialization
    // batch compile required.
    // outsize - returns the size of the linkage map.
    LWNsubroutineLinkageMapPtr GetSubroutineLinkageMap(LWNshaderStage stage, unsigned int compiledOutputNdx, int *outSize) const;

    const unsigned int * GetCompatibleSubroutineIndices(LWNshaderStage stage, const char *name, unsigned int *numCompatibleSubroutines) const;

    const unsigned int * GetCompatibleSubroutineIndices(const GLSLCsubroutineUniformInfo * subroutineUniformInfo, unsigned int *numCompatibleSubroutines) const;

    bool IsCacheAPIVersionCompatible()
    {
        // Not using a cache, so we're not compatible.
        if (!m_cacheParameters.m_dataCache) {
            return false;
        }

        return m_cacheParameters.m_dataCache->IsAPIVersionCompatible();
    }

    bool IsCacheGPUVersionCompatible()
    {
        // Not using a cache, so we're not compatible.
        if (!m_cacheParameters.m_dataCache) {
            return false;
        }

        return m_cacheParameters.m_dataCache->IsGPUVersionCompatible();
    }

    // Returns true or false depending on whether the last entry came from the disk cache.
    // Always returns false if not using a cache or cache reads are disabled.
    LWNboolean LastCacheEntryHit();

    // Returns true or false depending on whether we are lwrrently overriding the cache key.
    LWNboolean UsesOverrideCacheKey();

    // Set the override cache key.  This can be used to set up the cache key
    // externally by clients of this class rather than letting this class
    // compute the cache key itself.  This is useful when clients of
    // GLSLCHelper want to transform the input somehow but still want to store
    // the binaries based on the key of the original input.
    //
    // If <cacheKey> is non-NULL, <cacheKey> will be used to perform lookups on
    // all future cache operations.  If <cacheKey> is NULL, no cache override
    // will be used and any set cache override will be disabled, and GLSLCHelper
    // will instead internally callwlate the key based on the input shaders and
    // GLSLCoptions using the GLSLCHelperCache::HashShaders function.
    void SetOverrideCacheKey(const GLSLCHelperCacheKey * cacheKey);

    // Initializes the GLSLCinput and GLSLCoptions structures based on any options set
    // through the GLSLCHelper API.
    void InitializeGlslcInputAndOptions(GLSLCinput * input, GLSLCoptions * options,
            const LWNshaderStage * stages, uint32_t count, const char ** shaderSources,
            const SpirvParams * spvParams);

    // Retrieves the GLSLCHelperCache structure attached to this object.
    GLSLCHelperCache * GetHelperCache();

    // Returns the compile object associated with this helper.  Typically this
    // compile object is reconstructed each compile, so applications should not
    // try to use this except in special cirlwmstances.  One such cirlwmstance
    // is if the application calls CompilePreSpecialized and then wants to
    // manually call glslcCompileSpecialized: It requires the compile object
    // previously set up in glslcCompilePreSpecialized
    const GLSLCcompileObject * GetCompileObject();

    virtual ~GLSLCHelper();

#if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP
    //
    // Methods to provide a native C++ interface to the core GLSLCHelper
    // class, using reinterpret_cast to colwert between C and C++ object
    // types.
    //
    LWNboolean CompileAndSetShadersHLSL(lwn::Program *program, const lwn::ShaderStage *stages, uint32_t count, const char ** shaderSources)
    {
        LWNprogram *cprogram = reinterpret_cast<LWNprogram *>(program);
        const LWNshaderStage *cstages = reinterpret_cast<const LWNshaderStage *>(stages);
        return CompileAndSetShadersHLSL(cprogram, cstages, count, shaderSources);
    }
    LWNboolean CompileAndSetShaders(lwn::Program *program, const lwn::ShaderStage * stages, uint32_t count,
                                    const char ** shaderSources, SpirvParams * spirvParams = NULL)
    {
        LWNprogram *cprogram = reinterpret_cast<LWNprogram *>(program);
        const LWNshaderStage *cstages = reinterpret_cast<const LWNshaderStage *>(stages);
        return CompileAndSetShaders(cprogram, cstages, count, shaderSources, spirvParams);
    }
    LWNboolean CompileShaders(const lwn::ShaderStage *stages, uint32_t count, const char ** shaderSources, SpirvParams * spvParams = NULL)
    {
        const LWNshaderStage *cstages = reinterpret_cast<const LWNshaderStage *>(stages);
        return CompileShaders(cstages, count, shaderSources, spvParams);
    }

    LWNboolean CompileShadersPreSpecialized(const lwn::ShaderStage *stages, uint32_t count, const char ** shaderSources)
    {
        const LWNshaderStage *cstages = reinterpret_cast<const LWNshaderStage *>(stages);
        return CompileShadersPreSpecialized(cstages, count, shaderSources);
    }

    int32_t ProgramGetResourceLocation(lwn::Program *program, lwn::ShaderStage stage, LWNprogramResourceType type, const char * name)
    {
        LWNprogram *cprogram = reinterpret_cast<LWNprogram *>(program);
        LWNshaderStage cstage = LWNshaderStage(int(stage));
        return ProgramGetResourceLocation(cprogram, cstage, type, name);
    }
    static const GLSLCperfStatsHeader * ExtractPerfStatsSection(const GLSLCoutput * glslcOutput, lwn::ShaderStage stage)
    {
        LWNshaderStage cstage = LWNshaderStage(int(stage));
        return ExtractPerfStatsSection(glslcOutput, cstage);
    }
    static const GLSLCgpuCodeHeader * ExtractGpuCodeSection(const GLSLCoutput * glslcOutput, lwn::ShaderStage stage)
    {
        LWNshaderStage cstage = LWNshaderStage(int(stage));
        return ExtractGpuCodeSection(glslcOutput, cstage);
    }
    LWNboolean SetShaderScratchMemory(lwn::MemoryPool *memPool, ptrdiff_t offset, size_t size, lwn::CommandBuffer * cmdBuf)
    {
        LWNmemoryPool *cpool = reinterpret_cast<LWNmemoryPool *>(memPool);
        LWNcommandBuffer *ccb = reinterpret_cast<LWNcommandBuffer *>(cmdBuf);
        return SetShaderScratchMemory(cpool, offset, size, ccb);
    }
    size_t GetScratchMemoryRecommended(const lwn::Device *device, const GLSLCoutput * glslcOutput)
    {
        const LWNdevice *cdevice = reinterpret_cast<const LWNdevice *>(device);
        return GetScratchMemoryRecommended(cdevice, glslcOutput);
    }
    LWNboolean SetShaders(lwn::Program *program, const GLSLCoutput * glslcOutput)
    {
        LWNprogram *cprogram = reinterpret_cast<LWNprogram *>(program);
        return SetShaders(cprogram, glslcOutput);
    }
    const GLSLCsubroutineInfo * GetSubroutineInfo(lwn::ShaderStage stage, const char *name) const
    {
        LWNshaderStage cstage = LWNshaderStage(int(stage));
        return GetSubroutineInfo(cstage, name);
    }
    const GLSLCsubroutineUniformInfo * GetSubroutineUniformInfo(lwn::ShaderStage stage, const char *name) const
    {
        LWNshaderStage cstage = LWNshaderStage(int(stage));
        return GetSubroutineUniformInfo(cstage, name);
    }
    LWNsubroutineLinkageMapPtr GetSubroutineLinkageMap(lwn::ShaderStage stage, unsigned int compiledOutputNdx, int *outSize) const
    {
        LWNshaderStage cstage = LWNshaderStage(int(stage));
        return GetSubroutineLinkageMap(cstage, compiledOutputNdx, outSize);
    }
    const unsigned int * GetCompatibleSubroutineIndices(lwn::ShaderStage stage, const char *name, unsigned int *numCompatibleSubroutines) const
    {
        LWNshaderStage cstage = LWNshaderStage(int(stage));
        return GetCompatibleSubroutineIndices(cstage, name, numCompatibleSubroutines);
    }
#endif // #if LWNUTIL_INTERFACE_TYPE == LWNUTIL_INTERFACE_TYPE_CPP

private:

    enum SpecializedCompileType {
        GLSLC_COMPILE_TYPE_FULL,            // Do full specialized compile (glslcCompilePreSpecialized + glslcCompileSpecialized)
        GLSLC_COMPILE_TYPE_PRE_SPECIALIZED, // Do only the glslcCompilePreSpecialized part
    };

    // Either compiles the shaders or gets the shaders from the cache.  This will internally call into
    // the specialized compile or non-specialized compile functions.
    LWNboolean CompileGLSLCShaders(GLSLCinput * input, GLSLCoptions * options,
                              GLSLCcompileObject * compileObject,
                              std::vector<GLSLCspecializationUniform> * inputSpecArrays = NULL,
                              SpecializedCompileType type = GLSLC_COMPILE_TYPE_FULL);


    // Compile shaders using shader specialization.
    LWNboolean CompileGLSLCShadersSpecialized(GLSLCinput * input, GLSLCoptions * options, GLSLCcompileObject * compileObject,
                                              std::vector<GLSLCspecializationUniform> * inputSpecArrays, SpecializedCompileType type);

    // Compile shaders without using shader specialization
    LWNboolean CompileGLSLCShadersNonSpecialized(GLSLCinput * input, GLSLCoptions * options, GLSLCcompileObject * compileObject);

    // Override the private options in GLSLC compile object.
    void OverrideGLSLCPrivateOptions(GLSLCcompileObject * compileObject);

    // Looks up a variable in the reflecion map using the variable's name and the variable type.
    // A pointer to the data section is returned which needs to be cast to GLSLC's basic reflection
    // info types.  If the variable can not be found given the type, NULL is returned.
    const void * FindValueInPiqMap(ReflectionTypeEnum type, const char * name) const;

    // Parses the input reflection data into a double map of variables so they can be efficiently accessed
    // later using program interface query routines, such as ProgramGetResourceLocation
    void ParseReflectionInfo(const GLSLCprogramReflectionHeader * reflectionHeader, const char * data);

    // Determines whether the uniform is an image, sampler, or normal uniform.
    ReflectionTypeEnum GetUniformKind(const GLSLLwniformInfo * uniform);

    // Reset the GPU memory collected when loading the programs into the API.
    void ResetBuffers();

    GLSLCoptions m_userOptions;

    size_t m_poolSize;

    lwnUtil::MemoryPoolAllocator m_allocator;

    // Buffers containing compiled programs for use with lwnProgramSetSource.
    // Note: Since a GLSLCHelper can be used with multiple programs, a single
    // GLSLCHelper could be retaining memory for programs for multiple LWNprogram objects.
    std::vector<LWNbuffer *> m_buffers;

    std::vector<const char *> m_includePaths;

    LWNdevice * m_device;

    // The compile object from the last compilation.
    GLSLCcompileObject m_compileObject;

    // First map is keyed off of type ReflectionTypeEnum, where the inner key is based on the variable's
    // name.  The final data is a pointer to the entry in the GLSLC reflection data block.
    PiqMapType m_piqMap;

    // An array of specialization uniforms to be used as inputs with GLSLC's specialization
    // functionality.
    std::vector<GLSLCspecializationUniform> m_glslcSpecArrays[MAX_SPEC_ARRAYS];

    struct SpirvSpecializationConstantArrays {
        std::vector<uint32_t> constantIDs;
        std::vector<uint32_t> datum;
    };

    // Arrays of spirv specialization constant to be used as inputs with GLSLC's GLSLCinput.
    SpirvSpecializationConstantArrays m_spirvSpecConstArrays[GLSLC_NUM_SHADER_STAGES];
    GLSLCspirvSpecializationInfo m_glslSpirvSpecConstInfo[GLSLC_NUM_SHADER_STAGES];
    const GLSLCspirvSpecializationInfo * m_pGlslSpirvSpecConstInfo[GLSLC_NUM_SHADER_STAGES];

    DXCLibraryHelper * m_dxcLibraryHelper;

    GLSLCLibraryHelper * m_libraryHelper;

    // Scratch memory parameters variables that GLSLC will set during a call to SetShaderScratchMemory.
    // During shader compilation, there are sanity checks to ensure we have enough scratch memory
    // w.r.t. the compiled program binary's scratch memory requirements.
    LWNmemoryPool * m_scratchMemPool;
    ptrdiff_t m_scratchMemPoolOffset;
    size_t m_scratchMemPoolSize;

    // List of compiled outputs from a previous CompileShaders call.
    const GLSLCoutput * m_lastCompiledOutputs[MAX_SPEC_ARRAYS];

    // Logger to be used to log output.
    GLSLCLogger m_logger;


    // Parameters relating to the shader cache.
    struct CacheParameters {
        CacheParameters (GLSLCHelperCache * cache) :
            m_dataCache(cache), m_lastCacheHit(LWN_FALSE),
            m_allowCacheRead(LWN_FALSE), m_allowCacheWrite(LWN_FALSE),
            m_doCacheOverride(LWN_FALSE) {}

        // A shader cache (disabled by default) which holds already-compiled binaries keyed by hash values.
        GLSLCHelperCache * m_dataCache;

        // A flag indicating whether the last entry was obtained from the cache or not.
        LWNboolean m_lastCacheHit;

        // Determines if we want to read from the binary cache (if available).
        LWNboolean m_allowCacheRead;

        // Determines if we want to write to the binary data cache during successful compile (if available).
        LWNboolean m_allowCacheWrite;

        // If <m_doCacheOverride> is true, the cache key <m_overrideCacheKey> will be used when performing lookups and reads
        // from the cache.
        LWNboolean m_doCacheOverride;
        GLSLCHelperCacheKey m_overrideCacheKey;
    } m_cacheParameters;

    unsigned int m_overrideArch;
    unsigned int m_overrideImpl;

    uint8_t m_overrideDoGlslangShim;
    uint8_t m_overrideGlslangFallbackOnError;
    uint8_t m_overrideGlslangFallbackOnAbsolute;
};

} // namespace lwnUtil

#endif // #ifndef __lwnUtil_GlslcHelper_h__
