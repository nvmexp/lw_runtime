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

#include <rtcore/interface/types.h>
#include <exp/context/OptixABI.h>

#include <cassert>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

struct OptixModuleCompileOptions;
struct OptixPipelineCompileOptions;
struct OptixPayloadType;

namespace llvm {
class LLVMContext;
class Module;
class StringRef;
}

namespace optix_exp {
class ABIDecisionLogger;
class DeviceContext;
class ErrorDetails;
class Module;
class SubModule;

struct CompileBoundValueEntry
{
    size_t            offset;
    std::vector<char> value;
    std::string       annotation;
};

// Internal compiler payload type (See OptixPayloadType)
struct CompilePayloadType
{
    CompilePayloadType() {};
    CompilePayloadType( const OptixPayloadType& type );

    bool operator== (const CompilePayloadType& t) const
    {
        assert( (semantics == t.semantics) == (mangledName == t.mangledName) );
        return semantics == t.semantics;
    }

    // semantics for all payload values in the type
    std::vector<unsigned int> semantics;

    // a mangled name uniquely identifying the type semantics
    std::string mangledName;
};

struct InternalCompileParameters
{
    // options from the context
    int      maxSmVersion;
    OptixABI abiVersion;
    int      sbtHeaderSize;
    bool     noinlineEnabled;
    bool     validationModeDebugExceptions;

    // options from OptixModuleCompileOptions
    int                                 maxRegisterCount;
    OptixCompileOptimizationLevel       optLevel;
    OptixCompileDebugLevel              debugLevel;
    std::vector<CompileBoundValueEntry> specializedLaunchParam;

    // options from OptixPipelineCompileOptions
    bool                            usesMotionBlur;
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    bool                            allowVisibilityMaps;
    bool                            allowDisplacedMicromeshes;
#endif // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    unsigned int                    traversableGraphFlags;
    int                             numAttributeValues;
    unsigned int                    exceptionFlags;
    unsigned int                    usesPrimitiveTypeFlags;
    std::string                     pipelineLaunchParamsVariableName;
    std::vector<CompilePayloadType> payloadTypes;
    RtcAbiVariant                   abiVariant; // not directly in OptixPipelineCompileOptions, but derived by it

    // options from knobs
    int          callableParamRegCount;
    int          inlineCallLimitHigh;
    int          inlineCallLimitLow;
    int          inlineInstructionLimit;
    bool         removeUnusedNoinlineFunctions;
    std::string  forceInlineSet;
    std::string  disableNoinlineFunc;
    bool         allowIndirectFunctionCalls;
    bool         disableActiveMaskCheck;
    bool         enableLwstomABIProcessing;
    int          numAdditionalABIScratchRegs;
    bool         enableCoroutines;
    bool         enableProfiling;
    bool         useSoftwareTextureFootprint;
    unsigned int splitModuleMinBinSize;
    bool         serializeModuleId;

    // other options
    bool useD2IR;
    bool enableLWPTXFallback;
    bool enableCallableParamCheck;
    int  paramCheckExceptionRegisters;
    bool addBuiltinPrimitiveCheck;
    bool isBuiltinModule;
    bool enableLwstomPrimitiveVA;
    bool elideUserThrow;
    bool hideModule;

    std::vector<unsigned int> privateCompileTimeConstants;
};

OptixResult setRtcCompileOptions( RtcCompileOptions&               compileOptions,
                                  const InternalCompileParameters& compileParams,
                                  ErrorDetails&                    errDetails );

OptixResult createSubModules( Module*                              optixModule,
                              std::unique_ptr<llvm::LLVMContext>&& llvmContext,
                              llvm::Module*                        llvmModule,
                              unsigned int                         maxNumAdditionalTasks,
                              ErrorDetails&                        errDetails );
OptixResult compileSubModule( SubModule* subModule, bool& fellBackToLWPTX, ErrorDetails& errDetails );


RtcAbiVariant getRtcAbiVariant( const InternalCompileParameters& compileParams,
                                const DeviceContext*             context,
                                ABIDecisionLogger*               decisionLog = 0 );

int getExceptionFlags( const OptixPipelineCompileOptions* pipelineCompileOptions, bool enableAll );

OptixResult setInternalCompileOptions( InternalCompileParameters&         compileParams,
                                       const OptixModuleCompileOptions*   moduleCompileOptions,
                                       const OptixPipelineCompileOptions* pipelineCompileOptions,
                                       const DeviceContext*               context,
                                       const bool                         isBuiltinModule,
                                       const bool                         enableLwstomPrimitiveVA,
                                       const bool                         useD2IR,
                                       const std::vector<unsigned int>&   privateCompileTimeConstants,
                                       ErrorDetails&                      errDetails );

enum SemanticType
{
    ST_RAYGEN,
    ST_MISS,
    ST_CLOSESTHIT,
    ST_ANYHIT,
    ST_INTERSECTION,
    ST_EXCEPTION,
    ST_DIRECT_CALLABLE,
    ST_CONTINUATION_CALLABLE,
    ST_NOINLINE,
    ST_ILWALID
};

std::string semanticTypeToString( SemanticType stype );
std::string semanticTypeToAbbreviationString( SemanticType stype );

optix_exp::SemanticType getSemanticTypeForFunctionName( const llvm::StringRef& functionName,
                                                        bool                   noInlineEnabled,
                                                        const std::string&     disableNoinlineFunc );

// Helper function to retrieve information about module's debug state
bool isPtxDebugEnabled( llvm::Module* llvmModule );

}  // end namespace optix_exp
