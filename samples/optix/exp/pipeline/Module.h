/*
 * Copyright (c) 2019, LWPU CORPORATION.  All rights reserved.
 *
 * LWPU Corporation and its licensors retain all intellectual property and proprietary
 * rights in and to this software, related documentation and any modifications thereto.
 * Any use, reproduction, disclosure or distribution of this software and related
 * documentation without an express license agreement from LWPU Corporation is strictly
 * prohibited.
 *
 * TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
 * AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
 * INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS BE LIABLE FOR ANY
 * SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
 * LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
 * BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS) ARISING OUT OF THE USE OF OR
 * INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS BEEN ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGES
 */

#pragma once

#include <exp/context/OpaqueApiObject.h>
#include <exp/context/Task.h>
#include <exp/pipeline/Compile.h>
#include <optix_types.h>
#include <rtcore/interface/types.h>

#include <atomic>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>

namespace llvm {
class LLVMContext;
class Module;
class StringRef;
}

namespace optix {
class PersistentStream;
}

namespace optix_exp {

class DeviceContext;
class ErrorDetails;
struct SplitModuleFunctionInfo;

OptixResult createModule( DeviceContext*                     context,
                          const OptixModuleCompileOptions*   moduleCompileOptions,
                          const OptixPipelineCompileOptions* pipelineCompileOptions,
                          const char*                        PTX,
                          size_t                             PTXsize,
                          OptixModule*                       moduleAPI,
                          bool                               allowUnencryptedIfEncryptionIsEnabled,
                          bool                               isBuiltinModule,
                          bool                               enableLwstomPrimitiveVA,
                          bool                               useD2IR,
                          const std::vector<unsigned int>&   privateCompileTimeConstants,
                          char*                              logString,
                          size_t*                            logStringSize,
                          ErrorDetails&                      errDetails );

struct EntryFunctionSemantics
{
    EntryFunctionSemantics() = default;
    EntryFunctionSemantics( unsigned int payloadTypeMask )
        : m_payloadTypeMask( payloadTypeMask )
    {
    }

    // bitwise combination of supported OptixPayloadTypeID's
    unsigned m_payloadTypeMask = 0;
};

class SubModule
{
public:
    struct EntryFunctionInfo
    {
        EntryFunctionInfo() = default;
        EntryFunctionInfo( size_t traceCallCount, size_t continuationCallableCallCount, size_t directCallableCallCount, size_t basicBlockCount, size_t instructionCount )
            : m_traceCallCount( traceCallCount )
            , m_continuationCallableCallCount( continuationCallableCallCount )
            , m_directCallableCallCount( directCallableCallCount )
            , m_basicBlockCount( basicBlockCount )
            , m_instructionCount( instructionCount )
        {
        }
        size_t m_traceCallCount                = 0;
        size_t m_continuationCallableCallCount = 0;
        size_t m_directCallableCallCount       = 0;
        size_t m_basicBlockCount               = 0;
        size_t m_instructionCount              = 0;

        EntryFunctionInfo& operator+=( const EntryFunctionInfo& other )
        {
            m_traceCallCount += other.m_traceCallCount;
            m_continuationCallableCallCount += other.m_continuationCallableCallCount;
            m_directCallableCallCount += other.m_directCallableCallCount;
            m_basicBlockCount += other.m_basicBlockCount;
            m_instructionCount += other.m_instructionCount;
            return *this;
        }
    };

    struct NonEntryFunctionInfo
    {
        NonEntryFunctionInfo() = default;
        NonEntryFunctionInfo( size_t count, size_t basicBlockCount, size_t instructionCount )
            : m_count( count )
            , m_basicBlockCount( basicBlockCount )
            , m_instructionCount( instructionCount )
        {
        }
        size_t m_count            = 0;
        size_t m_basicBlockCount  = 0;
        size_t m_instructionCount = 0;

        NonEntryFunctionInfo& operator+=( const NonEntryFunctionInfo& other )
        {
            m_count += other.m_count;
            m_basicBlockCount += other.m_basicBlockCount;
            m_instructionCount += other.m_instructionCount;
            return *this;
        }
    };

    enum class ModuleSymbolType
    {
        FUNCTION,
        DATA
    };

    static std::string getSymbolTypeString( ModuleSymbolType type );

    struct ModuleSymbol
    {
        size_t size;
        ModuleSymbolType type;
    };

public:
    OptixResult destroy( ErrorDetails& errDetails );

    void registerEntryFunction( const std::string&            unmangledName,
                                EntryFunctionInfo&&           entryFunctionInfo,
                                const EntryFunctionSemantics& entryFunctionSemantics,
                                SemanticType                  stype );
    void setNonEntryFunctionInfo( NonEntryFunctionInfo&& nonEntryFunctionInfo );

    RtcCompiledModule getRtcCompiledModule() const { return m_rtcModule->m_rtcModule; }
    EntryFunctionInfo getEntryFunctionInfo( const std::string& unmangledName );
    const std::map<std::string, ModuleSymbol>& getExportedSymbols() const { return m_exportedSymbols; }
    const std::map<std::string, ModuleSymbol>& getImportedSymbols() const { return m_importedSymbols; }
    void addExportedDataSymbol( const std::string& symbolName, size_t symbolSize );
    void addImportedDataSymbol( const std::string& symbolName, size_t symbolSize );
    void addExportedFunctionSymbol( const std::string& symbolName, size_t symbolSize );
    void addImportedFunctionSymbol( const std::string& symbolName, size_t symbolSize );

    // Check whether this module uses the texture footprint intrinsic.
    bool usesTextureIntrinsic() const { return m_usesTextureIntrinsic; }

    // Set flag indicating that this module uses the texture footprint intrinsic.
    void setUsesTextureIntrinsic() { m_usesTextureIntrinsic = true; }


    OptixResult saveToStream( optix::PersistentStream* stream, const char* label, ErrorDetails& errDetails );
    OptixResult readFromStream( optix::PersistentStream* stream, const char* label, const std::string& cacheKey, ErrorDetails& errDetails );


    Module* m_parentModule = nullptr;

    // Stores information about all non-entry functions of the module.
    NonEntryFunctionInfo m_nonEntryFunctionInfo;
    // Stores information about the entry function.
    std::map<std::string, EntryFunctionInfo> m_entryFunctionInfo;

    // Keeps track of semantics of registered entry functions of the module (by unmangled name).
    // All supported payload type variations of an entry function are compiled into the same submodel (for now).
    std::map<std::string, EntryFunctionSemantics> m_entryFunctionSemantics;

    // Map of mangled entry function names to program indices in the rtcModule.
    std::map<std::string, Rtlw32> m_mangledEntryFunctionToProgramIndex;

    std::map<std::string, ModuleSymbol> m_exportedSymbols;
    std::map<std::string, ModuleSymbol> m_importedSymbols;

    // True if this module uses the texture footprint intrinsic.
    bool m_usesTextureIntrinsic = false;

    std::unique_ptr<RtcoreModule> m_rtcModule;

    std::unique_ptr<llvm::LLVMContext>                    m_llvmContext;
    llvm::Module*                                         m_llvmModule = nullptr;
    std::shared_ptr<std::string>                          m_serializedModule;
    std::shared_ptr<std::vector<SplitModuleFunctionInfo>> m_splitModuleFunctionInfos;
    unsigned int                                          m_splitModuleBinID = 0;

    int m_moduleIndex = -1;
};

class Module : public OpaqueApiObject
{
  public:
    struct InitialCompileTask : public Task
    {
        InitialCompileTask( Module* parentModule, InternalCompileParameters&& compileParams, bool decryptInput, const char* input, size_t inputSize );

        OptixResult exelwteImpl( OptixTask*    additionalTasksAPI,
                                 unsigned int  maxNumAdditionalTasks,
                                 unsigned int* numAdditionalTasksCreated,
                                 ErrorDetails& errDetails );
        OptixResult execute( OptixTask*    additionalTasksAPI,
                             unsigned int  maxNumAdditionalTasks,
                             unsigned int* numAdditionalTasksCreated,
                             ErrorDetails& errDetails ) override;

        void logErrorDetails( OptixResult result, ErrorDetails&& errDetails ) override;

        Module*                   m_parentModule = nullptr;
        InternalCompileParameters m_compileParams;
        bool                      m_decryptInput = false;
        const char*               m_input        = nullptr;
        size_t                    m_inputSize    = 0;
    };

    struct SubModuleCompileTask : public Task
    {
        SubModuleCompileTask( SubModule* subModule, const RtcCompileOptions& rtcOptions );

        OptixResult exelwteImpl( OptixTask*    additionalTasksAPI,
                                 unsigned int  maxNumAdditionalTasks,
                                 unsigned int* numAdditionalTasksCreated,
                                 ErrorDetails& errDetails );
        OptixResult execute( OptixTask*    additionalTasksAPI,
                             unsigned int  maxNumAdditionalTasks,
                             unsigned int* numAdditionalTasksCreated,
                             ErrorDetails& errDetails ) override;

        void logErrorDetails( OptixResult result, ErrorDetails&& errDetails ) override;

        SubModule*        m_subModule  = nullptr;
        RtcCompileOptions m_rtcOptions = {}; // these are a copy since we could modify them per SubModule
    };

    struct ModuleCompletionData
    {
        bool        m_decryptInput = false;
        std::string m_encryptedCacheKey;
        std::string m_cacheKey;
    };

    static constexpr size_t s_ilwalidPipelineParamsSize = static_cast<size_t>( -1 );

    // For Cache
    Module( DeviceContext* context, std::string&& ptxHash );
    // For user input
    Module( DeviceContext* context, InternalCompileParameters&& compileParams, bool decryptInput, const char* input, size_t inputSize, char* logString, size_t* logStringSize );

    // This is like a move assign operator, but with OptixResult and ErrorDetails. It is
    // only designed to be used with Module objects created from the cache.
    OptixResult moveAssignFromCache( std::unique_ptr<Module>& otherModule, ErrorDetails& errDetails );

    OptixResult destroy( ErrorDetails& errDetails ) { return destroy( true, errDetails ); };

    OptixResult destroyWithoutUnregistration( ErrorDetails& errDetails ) { return destroy( false, errDetails ); };

    DeviceContext* getDeviceContext() const { return m_context; }

    const std::string& getPtxHash() const { return m_ptxHash; }
    void setPtxHash( std::string&& ptxHash ) { m_ptxHash = std::move( ptxHash ); }

    int getModuleId() const { return m_moduleId; }

    const SubModule* getSubModule( const char* mangledName ) const;
    std::vector<const SubModule*> getSubModuleAndDependencies( const char* mangledName ) const;
    const std::vector<SubModule*>& getSubModules() const { return m_subModules; }
    void addSubModule( SubModule* subModule );
    void addSubModuleCompilationTask( Task* task );
    OptixResult subModuleFinished( SubModule* subModule, ErrorDetails& errDetails );
    const std::vector<OptixTask>& getSubModuleCompileTasks() { return m_subModuleCompileTasks; }
    void setModuleCompletionData( bool decryptInput, std::string&& encryptedCacheKey, std::string&& cacheKey );
    void aggregateSubModuleData();

    size_t getPipelineParamsSize() const { return m_pipelineParamsSize; }

    const InternalCompileParameters& getCompileParameters() const { return m_compileParameters; }

    // Indicates whether an entry function of the given unmangled name has been registered.
    bool hasEntryFunction( const std::string& unmangledName ) const;

    // Returns semantics information about an entry function of the given unmangled name (or a default constructed instance if
    // there is no such entry function).
    EntryFunctionSemantics getEntryFunctionSemantics( const std::string& unmangledName ) const;

    // Returns the mangled name for an unmangled name.
    //
    // Name mangling adds an underscore and the PTX hash, unless the PTX hash is the empty string.
    // If the function is of type ST_NOINLINE, no name mangling is done and the input string is returned.
    // If stype is ST_ILWALID then it computes the semantic type.
    std::string getMangledName( const llvm::StringRef& name, unsigned int optixPayloadTypeID, SemanticType stype ) const;

    // Used by ProgramGroup when filling in the SBT header
    OptixResult getRtcCompiledModuleAndProgramIndex( const std::string& mangledName,
                                                     RtcCompiledModule& rtcModule,
                                                     Rtlw32&            programIndex,
                                                     ErrorDetails&      errDetails ) const;

    InitialCompileTask* getInitialTask() { return m_initialTask.get(); }

    // Return the type Id corresponding to the type specification (or OPTIX_PAYLOAD_TYPE_DEFAULT if none is found)
    unsigned int getCompatiblePayloadTypeId( const CompilePayloadType& type, unsigned int optixPayloadTypeMask = ~0u ) const;

    // Return the type specification corresponding to the specified OptixPayloadTypeID
    const CompilePayloadType* getPayloadTypeFromId( unsigned int typeId ) const;

    // Logs the error details to the output string and the logger
    void logTaskErrorDetails( OptixResult taskResult, ErrorDetails&& errDetails );

    bool isBuiltinModule() const { return m_compileParameters.isBuiltinModule; }

    // Sets m_pipelineParamsSize.
    void setPipelineParamsSize( size_t pipelineParamsSize ) { m_pipelineParamsSize = pipelineParamsSize; }

    // Sets m_compileParameters.
    void setCompileParameters( InternalCompileParameters&& compileParams );

    // To be used by DeviceContext only.
    struct DeviceContextIndex_fn
    {
        int& operator()( const Module* module ) { return module->m_deviceContextIndex; }
    };

    static OptixResult saveModuleToDiskCache( DeviceContext*                            context,
                                              const std::string&                        cacheKey,
                                              const optix_exp::Module* module,
                                              ErrorDetails&                             errDetails );
    static OptixResult loadModuleFromDiskCache( DeviceContext*           context,
                                                const std::string&       cacheKey,
                                                std::unique_ptr<Module>& module,
                                                ErrorDetails&            errDetails );

    void setModuleIdentifier( const std::string& moduleId )
    {
        m_moduleIdentifier = moduleId;
    }
    const std::string& getModuleIdentifier() const
    {
        return m_moduleIdentifier;
    }

    void setCompileStateStarted() { m_compileState->store( OPTIX_MODULE_COMPILE_STATE_STARTED ); }
    OptixModuleCompileState getCompileState() { return m_compileState->load(); }

    // Handle debug information state.
    void setHasDebugInformation() { m_hasDebugInformation = true; }
    bool hasDebugInformation() const { return m_hasDebugInformation; }

  private:
    // Use unique ID instead of m_ptxHash when dumping files.
    static std::atomic_int s_serializedModuleId;

    OptixResult destroy( bool doUnregisterModule, ErrorDetails& errDetails );

    DeviceContext*    m_context   = nullptr;

    // Hash of PTX code.
    std::string m_ptxHash;

    int m_moduleId = 0;

    size_t m_pipelineParamsSize = s_ilwalidPipelineParamsSize;

    InternalCompileParameters m_compileParameters;

    std::unique_ptr<std::atomic<OptixModuleCompileState>> m_compileState;

    std::unique_ptr<InitialCompileTask> m_initialTask;
    // needs to be a flat array, so we can pass a pointer to the array out of the API
    std::vector<OptixTask>            m_subModuleCompileTasks;
    std::unique_ptr<std::atomic<int>> m_numSubModulesLeft;
    std::unique_ptr<std::atomic<int>> m_numSubModulesLwrrentlyActive;
    ModuleCompletionData              m_completionData;

    std::unique_ptr<std::mutex>         m_taskLogLock;
    std::vector<ErrorDetails>           m_taskLogsWithErrors;
    std::vector<ErrorDetails>           m_taskLogsWithoutErrors;
    char*                               m_logString     = nullptr;
    size_t*                             m_logStringSize = nullptr;
    // This is the size of the log memory buffer and needs to be separate than
    // m_logStringSize, since that value needs to eventually contain the maximum size
    // required to emit the full log and subsequent log messages need the size of the
    // buffer.
    size_t                              m_logStringMemSize = 0;
#if 0
    using ModuleListType = optix::IndexedVector<CompileFunctionTask*, Module::CompileFunctionTask::TaskIndex_fn>;
    ModuleListType m_tasks;
    std::unique_ptr<std::mutex> m_taskLock;
#endif

    // Keeps track of sub modules keyed on the mangled name.
    std::map<std::string, size_t>                 m_mangledEntryFunctionNameToSubModule;

    // Keeps track of semantics of registered entry functions of the module (by unmangled name).
    std::map<std::string, EntryFunctionSemantics> m_entryFunctionSemantics;

    std::vector<SubModule*> m_subModules;
    SubModule* m_nonEntryFunctionModule = nullptr;

    mutable int m_deviceContextIndex = -1;

    std::string m_moduleIdentifier;

    // Is the module built with debug information?
    std::atomic<bool> m_hasDebugInformation;
};


inline OptixResult implCast( OptixModule moduleAPI, Module*& module )
{
    module = reinterpret_cast<Module*>( moduleAPI );
    // It's OK for moduleAPI to be nullptr
    if( module && module->m_apiType != OpaqueApiObject::ApiType::Module )
    {
        return OPTIX_ERROR_ILWALID_VALUE;
    }
    return OPTIX_SUCCESS;
}

inline OptixModule apiCast( Module* module )
{
    return reinterpret_cast<OptixModule>( module );
}

}  // end namespace optix_exp

namespace optix {
void readOrWrite( PersistentStream* stream, optix_exp::SubModule::ModuleSymbol* symbol, const char* label );
}
