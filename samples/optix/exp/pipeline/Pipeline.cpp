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

#include <exp/context/DeviceContext.h>
#include <exp/context/ErrorHandling.h>
#include <exp/context/LaunchResourceManager.h>
#include <exp/context/Metrics.h>
#include <exp/context/OptixResultOneShot.h>
#include <exp/context/GpuWarmup.h>
#include <exp/context/WatchdogTimer.h>
#include <exp/functionTable/compileOptionsTranslate.h>
#include <exp/pipeline/Compile.h>
#include <exp/pipeline/DefaultException.h>
#include <exp/pipeline/DefaultException_ptx_bin.h>
#include <exp/pipeline/Module.h>
#include <exp/pipeline/Pipeline.h>

#include <exp/pipeline/O7TextureFootprintWrappersHW_bin.h>
#include <exp/pipeline/O7TextureFootprintWrappersSW_bin.h>

#include <corelib/compiler/LLVMUtil.h>
#include <corelib/math/MathUtil.h>
#include <corelib/misc/ProfileDump.h>
#include <corelib/misc/String.h>
#include <corelib/system/LwdaDriver.h>
#include <corelib/system/Preprocessor.h>
#include <prodlib/misc/LwdaStopwatch.h>
#include <prodlib/exceptions/Exception.h>
#include <prodlib/misc/LWTXProfiler.h>
#include <prodlib/system/Knobs.h>
#include <rtcore/interface/types.h>

#include <lwda_runtime.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

namespace {
// clang-format off
Knob<std::string> k_enableProfiling( RT_DSTRING( "o7.enableProfiling" ), "", RT_DSTRING( "String when non-empty enables profiling and stores the result to the prefix defined here" ) );
Knob<long long>   k_blocksyncThreshold( RT_DSTRING( "o7.blocksync.instructionThreshold" ), 20000, RT_DSTRING( "Pipeline instruction threshold which when exceeded enables blocksync scheduling" ) );
Knob<bool>        k_enableSpecializations( RT_DSTRING( "o7.compile.enableLaunchParamSpecialization" ), true, RT_DSTRING( "Enable the use of specialized launch params." ) );
Knob<bool>        k_enableSpecializationsConsistencyChecks( RT_DSTRING( "o7.enableLaunchParamSpecializationConsistencyChecks" ), true, RT_DSTRING( "Verify specializations are consistent between modules and if validation is enabled against optixLaunch's params" ) );
Knob<bool>        k_forceSpecializationsConsistencyChecksAtLaunch( RT_DSTRING( "o7.forceLaunchParamSpecializationConsistencyChecksAtLaunch" ), false, RT_DSTRING( "Verify specializations are consistent against optixLaunch's params even if validation isn't enabled" ) );
Knob<bool>        k_enablePrelinkSymbolResolving( RT_DSTRING( "o7.enablePrelinkSymbolResolving" ), true, RT_DSTRING( "Verify imported and exported symbols of modules at pipeline creation before linking" ) );
Knob<bool>        k_gpuWarmupEnabled( RT_DSTRING( "rtx.gpuWarmupEnabled" ), true, RT_DSTRING( "Enable the GPU warm-up kernel." ) );
Knob<float>       k_gpuWarmupThreshold( RT_DSTRING( "rtx.gpuWarmupThreshold" ), 2.0, RT_DSTRING( "Time in seconds to wait before relaunching the GPU warm-up kernel" ) );
// clang-format on
}  // namespace

namespace optix_exp {

Pipeline::Pipeline( DeviceContext*                        context,
                    RtcPipeline                           rtcPipeline,
                    RtcCompiledModule                     launchParamModule,
                    Module*                               defaultExceptionModule,
                    OptixPipelineCompileOptions           pipelineCompileOptions,
                    OptixPipelineLinkOptions              pipelineLinkOptions,
                    size_t                                launchBufferSizeInBytes,
                    size_t                                launchBufferAlignment,
                    size_t                                launchParamOffset,
                    size_t                                launchParamSizeInBytes,
                    size_t                                toolsOutputSizeInBytes,
                    const RtcPipelineInfoShaderInfo&      shaderInfo,
                    const RtcPipelineInfoProfileMetadata& profileMetadata,
                    bool                                  ignorePipelineLaunchParamsAtLaunch,
                    LaunchParamSpecialization&&           computedSpecializations,
                    LWevent                               conlwrrentLaunchEvent,
                    std::vector<NamedConstant*>&&         namedConstants,
                    bool                                  hasDebugInformation )
    : OpaqueApiObject( OpaqueApiObject::ApiType::Pipeline )
    , m_context( context )
    , m_rtcPipeline( rtcPipeline )
    , m_launchParamModule( launchParamModule )
    , m_defaultExceptionModule( defaultExceptionModule )
    , m_pipelineCompileOptions( pipelineCompileOptions )
    , m_pipelineLinkOptions( pipelineLinkOptions )
    , m_launchBufferSizeInBytes( launchBufferSizeInBytes )
    , m_launchBufferAlignment( launchBufferAlignment )
    , m_launchParamOffset( launchParamOffset )
    , m_launchParamSizeInBytes( launchParamSizeInBytes )
    , m_toolsOutputSizeInBytes( toolsOutputSizeInBytes )
    , m_shaderInfo( shaderInfo )
    , m_profileMetadata( profileMetadata )
    , m_ignorePipelineLaunchParamsAtLaunch( ignorePipelineLaunchParamsAtLaunch )
    , m_computedSpecializations( std::move( computedSpecializations ) )
    , m_conlwrrentLaunchEvent( conlwrrentLaunchEvent )
    , m_additionalNamedConstants( namedConstants )
    , m_hasDebugInformation( hasDebugInformation )
{
    m_pipelineCompileOptions.pipelineLaunchParamsVariableName =
        duplicate( m_pipelineCompileOptions.pipelineLaunchParamsVariableName );
}

OptixResult Pipeline::destroy( bool doUnregisterPipeline, ErrorDetails& errDetails )
{
    OptixResultOneShot result;

    LwdaContextPushPop lwCtx( m_context );
    result += lwCtx.init( errDetails );

    if( m_launchParamModule )
    {
        if( const RtcResult rtcResult = m_context->getRtcore().compiledModuleDestroy( m_launchParamModule ) )
            result += errDetails.logDetails( rtcResult, "Error destroying rtcore launch params" );
        m_launchParamModule = nullptr;
    }

    for( NamedConstant* namedConstant : m_additionalNamedConstants )
    {
        result += namedConstant->m_destroyFunctor( errDetails, *namedConstant );
        if( const RtcResult rtcResult = m_context->getRtcore().compiledModuleDestroy( namedConstant->m_rtcModule ) )
            result += errDetails.logDetails( rtcResult, "Error destroying rtcore named constant" );
        delete namedConstant;
    }

    if( m_defaultExceptionModule )
        result += m_defaultExceptionModule->destroy( errDetails );
    delete m_defaultExceptionModule;

    result += doUnregisterPipeline ? m_context->unregisterPipeline( this, errDetails ) : OPTIX_SUCCESS;

    if( m_rtcPipeline )
    {
        if( const RtcResult rtcResult = m_context->getRtcore().pipelineDestroy( m_rtcPipeline ) )
            result += errDetails.logDetails( rtcResult, "Error destroying rtcore pipeline" );
        m_rtcPipeline = nullptr;
    }
    result += lwCtx.destroy( errDetails );

    if( m_conlwrrentLaunchEvent )
    {
        // Synchronize the event, so that we know that we are done with the resource
        if( LWresult lwResultEvent = corelib::lwdaDriver().LwEventSynchronize( m_conlwrrentLaunchEvent ) )
            result += errDetails.logDetails( lwResultEvent, "Error synching on OptixPipeline event" );
        if( LWresult lwResultEvent = corelib::lwdaDriver().LwEventDestroy( m_conlwrrentLaunchEvent ) )
            result += errDetails.logDetails( lwResultEvent, "Error destroying OptixPipeline event" );
        m_conlwrrentLaunchEvent = nullptr;
    }

    return result;
}

const char* Pipeline::duplicate( const char* s )
{
    if( !s )
        return nullptr;

    m_strings.emplace_back( s );
    return m_strings.back().c_str();
}

void LaunchParamSpecialization::clear()
{
    data.clear();
    specializedRanges.clear();
}

void LaunchParamSpecialization::addSpecialization( const CompileBoundValueEntry& entry, const std::string& moduleId )
{
    addSpecialization( entry, 0, moduleId );
}

void LaunchParamSpecialization::addSpecialization( const CompileBoundValueEntry& entry, size_t offsetInEntryData, const std::string& moduleId )
{
    specializedRanges.push_back( {entry.offset, entry.value.size(), moduleId, entry.annotation} );
    memcpy( data.data() + entry.offset + offsetInEntryData, entry.value.data() + offsetInEntryData,
            entry.value.size() - offsetInEntryData );
}

static RtcPipelineOptions generateRtcPipelineOptions( const DeviceContext*             context,
                                                      const InternalCompileParameters& compileParams,
                                                      const OptixPipelineLinkOptions*  pipelineLinkOptions,
                                                      size_t                           pipelineInstructionCount )
{
    RtcPipelineOptions pipelineOptions = {};

    pipelineOptions.abiVariant = getRtcAbiVariant( compileParams, context );

    if( pipelineOptions.abiVariant == RTC_ABI_VARIANT_TTU_A )
        pipelineOptions.traversalType = RTC_TRAVERSER_TYPE_TTU_2LH;
    else if( pipelineOptions.abiVariant == RTC_ABI_VARIANT_TTU_B )
        pipelineOptions.traversalType = RTC_TRAVERSER_TYPE_TTU_2LH;
#if LWCFG( GLOBAL_FEATURE_GR1354_MICROMESH )
    else if( pipelineOptions.abiVariant == RTC_ABI_VARIANT_TTU_D )
        pipelineOptions.traversalType = RTC_TRAVERSER_TYPE_TTU_2LH;
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    else if( pipelineOptions.abiVariant == RTC_ABI_VARIANT_MTTU )
        pipelineOptions.traversalType = RTC_TRAVERSER_TYPE_MTTU;
    else
        pipelineOptions.traversalType = RTC_TRAVERSER_TYPE_UNIVERSAL;

    pipelineOptions.maxTraceRelwrsionDepth        = pipelineLinkOptions->maxTraceDepth;
    pipelineOptions.defaultCallableRelwrsionDepth = 2;

    bool blocksync       = pipelineInstructionCount > k_blocksyncThreshold.get();
#ifdef OPTIX_ENABLE_LOGGING
    llog( 20 ) << "blocksync = " << std::boolalpha << blocksync << std::noboolalpha << " = " << pipelineInstructionCount
               << " > " << k_blocksyncThreshold.get() << " ( pipelineInstructionCount > k_blocksyncThreshold.get() )\n";
#endif
    pipelineOptions.type = blocksync ? RTC_PIPELINE_TYPE_BLOCKSYNC : RTC_PIPELINE_TYPE_MEGAKERNEL_SIMPLE;

#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    if( compileParams.allowVisibilityMaps )
        pipelineOptions.pipelineFlags |= RTC_PIPELINE_FLAG_ALLOW_VISIBILITY_MAPS;
    if( compileParams.allowDisplacedMicromeshes )
        pipelineOptions.pipelineFlags |= RTC_PIPELINE_FLAG_ALLOW_DISPLACED_MICROMESHES;
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

    return pipelineOptions;
}

static OptixResult checkPipelineCompileOptions( const OptixPipelineCompileOptions* pipelineCompileOptions,
                                                bool                               enableAllDebugExceptions,
                                                ErrorDetails&                      errDetails )
{
    if( pipelineCompileOptions->usesMotionBlur != 0 && pipelineCompileOptions->usesMotionBlur != 1 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "usesMotionBlur is neither 0 nor 1" );

    unsigned int traversalDepthMask = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
                                      | OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    if( ( pipelineCompileOptions->traversableGraphFlags & ~traversalDepthMask ) != 0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "traversableGraphFlags has invalid bits set" );

    if( pipelineCompileOptions->numPayloadValues > OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "numPayloadValues exceeds OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT (" + std::to_string( OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT ) + ")" );

    if( pipelineCompileOptions->numAttributeValues > 8 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "numAttributeValues exceeds 8" );

    unsigned int exceptionFlagsMask = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH
                                      | OPTIX_EXCEPTION_FLAG_USER | OPTIX_EXCEPTION_FLAG_DEBUG;
    if( ( getExceptionFlags( pipelineCompileOptions, enableAllDebugExceptions ) & ~exceptionFlagsMask ) != 0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "exceptionFlags has invalid bits set" );

#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    if( pipelineCompileOptions->allowVisibilityMaps != 0 && pipelineCompileOptions->allowVisibilityMaps != 1 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "allowVisibilityMaps is neither 0 nor 1" );
    if( pipelineCompileOptions->allowDisplacedMicromeshes != 0 && pipelineCompileOptions->allowDisplacedMicromeshes != 1 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "allowDisplacedMicromeshes is neither 0 nor 1" );
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

    return OPTIX_SUCCESS;
}

static OptixResult checkPipelineLinkOptions( const OptixPipelineLinkOptions* pipelineLinkOptions, ErrorDetails& errDetails )
{
    if( pipelineLinkOptions->maxTraceDepth > 31 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "maxTraceDepth too high" );

    return OPTIX_SUCCESS;
}

static OptixResult checkCommonCompileOptions( const InternalCompileParameters& commonCompileOptions,
                                              size_t&                          commonParamsSize,
                                              const InternalCompileParameters& specificCompileOptions,
                                              size_t                           specificParamsSize,
                                              unsigned int                     index,
                                              const std::string&               kindStr,
                                              ErrorDetails&                    errDetails )
{
#define OPTIX_COMPARE_OPTION( name )                                                                                     \
    if( !( specificCompileOptions.name == commonCompileOptions.name ) )                                                  \
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,                                                         \
                                      "pipeline compile option " #name " for \"programGroups[" + std::to_string( index ) \
                                          + "]." + kindStr + "\" does not match value from pipeline creation" );

    OPTIX_COMPARE_OPTION( usesMotionBlur );
    OPTIX_COMPARE_OPTION( traversableGraphFlags );
    OPTIX_COMPARE_OPTION( numAttributeValues );
    OPTIX_COMPARE_OPTION( exceptionFlags );
    OPTIX_COMPARE_OPTION( usesPrimitiveTypeFlags );
    OPTIX_COMPARE_OPTION( pipelineLaunchParamsVariableName );
#undef OPTIX_COMPARE_OPTION

    // Similar for params size, but the value is initially unknown, we need to keep track of the common value, and the
    // size might not be known for a module if the params are not used by that module.
    if( commonParamsSize != Module::s_ilwalidPipelineParamsSize
        && specificParamsSize != Module::s_ilwalidPipelineParamsSize && specificParamsSize != commonParamsSize )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "size of params variable in \"programGroups[" + std::to_string( index ) + "]." + kindStr
                                          + "\" does not match size of params variable from previous modules" );
    if( commonParamsSize == Module::s_ilwalidPipelineParamsSize )
        commonParamsSize = specificParamsSize;

    return OPTIX_SUCCESS;
}

namespace {

struct FunctionInfoAclwmulator
{
    size_t                          moduleCount        = 0;
    size_t                          entryFunctionCount = 0;
    SubModule::EntryFunctionInfo    entryFunctionInfo;
    SubModule::NonEntryFunctionInfo nonEntryFunctionInfo;
    std::set<std::string>           seenMangledNames;

    void aclwmulateEntryFunctionInfo( const SubModule* subModule, const char* mangledName )
    {
        if( !seenMangledNames.insert( mangledName ).second )
            return;
        const auto& iter = subModule->m_entryFunctionInfo.find( mangledName );
        if( iter == subModule->m_entryFunctionInfo.end() )
            return;
        ++entryFunctionCount;
        entryFunctionInfo += iter->second;
    }
    void aclwmulateNonEntryFunctionInfo( const SubModule* subModule )
    {
        nonEntryFunctionInfo += subModule->m_nonEntryFunctionInfo;
    }
};

}  // namespace


struct ModuleExportSymbol
{
    std::string             exportingModule;
    SubModule::ModuleSymbol symbol;
};

static OptixResult aclwmulateExports( std::map<std::string, ModuleExportSymbol>& exportedSymbols,
                                      const std::vector<const SubModule*>& subModules,
                                      ErrorDetails&                        errDetails )
{
    OptixResultOneShot result;
    for( const SubModule* subModule : subModules )
    {
        for( const auto& exported : subModule->getExportedSymbols() )
        {
            auto inserted = exportedSymbols.insert(
                std::make_pair( exported.first, ModuleExportSymbol{subModule->m_parentModule->getModuleIdentifier(), exported.second} ) );
            if( !inserted.second )
            {
                // multiply defined symbol
                errDetails.m_compilerFeedback << "Error: Symbol '" << exported.first
                                              << "' was defined multiple times. First seen in: '"
                                              << inserted.first->second.exportingModule << "'\n";
                result += OPTIX_ERROR_PIPELINE_LINK_ERROR;
            }
        }
    }
    return result;
}


static OptixResult resolveModuleSymbols( const std::vector<const SubModule*>& subModules, ErrorDetails& errDetails )
{
    if( !k_enablePrelinkSymbolResolving.get() )
        return OPTIX_SUCCESS;

    std::map<std::string, ModuleExportSymbol> exportedSymbols;

    OptixResultOneShot result;
    result += aclwmulateExports( exportedSymbols, subModules, errDetails );

    for( const SubModule* subModule : subModules )
    {
        const std::string& moduleIdentifier = subModule->m_parentModule->getModuleIdentifier();
        for( const auto& import : subModule->getImportedSymbols() )
        {
            auto exported = exportedSymbols.find( import.first );
            if( exported != exportedSymbols.end() )
            {
                const ModuleExportSymbol exportSymbol = exported->second;
                if( exportSymbol.symbol.type != import.second.type )
                {

                    errDetails.m_compilerFeedback
                        << "Error: Symbol '" << exported->first << "' was defined as a "
                        << SubModule::getSymbolTypeString( exportSymbol.symbol.type ) << " in module '" << exportSymbol.exportingModule
                        << "' but imported as a " << SubModule::getSymbolTypeString( import.second.type ) << " in module '"
                        << moduleIdentifier << "'\n";
                    result += OPTIX_ERROR_PIPELINE_LINK_ERROR;
                }
                else if( exportSymbol.symbol.size != import.second.size )
                {
                    bool isSizeMismatch = true;
                    if( exportSymbol.symbol.type == SubModule::ModuleSymbolType::FUNCTION )
                    {
                        // Any return value that is smaller than 32 bit is promoted to 32 bit. Except void.
                        size_t importSize = import.second.size != 0 && import.second.size < 4 ? 4 : import.second.size;
                        size_t exportSize =
                            exportSymbol.symbol.size != 0 && exportSymbol.symbol.size < 4 ? 4 : exportSymbol.symbol.size;
                        isSizeMismatch = importSize != exportSize;
                        if( !isSizeMismatch )
                        {
                            errDetails.m_compilerFeedback
                                << "Warning: Size mismatch in return type of function '" << exported->first
                                << "' in module '" << moduleIdentifier << "', first seen in module '"
                                << exportSymbol.exportingModule << "'\n";
                        }
                    }
                    if( isSizeMismatch )
                    {
                        if( import.second.type == SubModule::ModuleSymbolType::FUNCTION )
                        {
                            errDetails.m_compilerFeedback
                                << "Error: Signature mismatch for function '" << exported->first << "' in module '"
                                << moduleIdentifier << "', first seen in module '"
                                << exportSymbol.exportingModule << "'. Return value size does not match.\n";
                        }
                        else
                        {
                            errDetails.m_compilerFeedback << "Error: Size mismatch for symbol '" << exported->first
                                                          << "' in module '" << moduleIdentifier
                                                          << "', first seen in module '" << exportSymbol.exportingModule
                                                          << "'\n";
                        }
                        result += OPTIX_ERROR_PIPELINE_LINK_ERROR;
                    }
                }
            }
            else
            {
                errDetails.m_compilerFeedback << "Error: Unresolved external symbol '" << import.first
                                              << "' in module '" << moduleIdentifier << "'\n";
                result += OPTIX_ERROR_PIPELINE_LINK_ERROR;
            }
        }
    }
    return result;
}

namespace {
// This wrapper ensures proper memory cleanup of the constructed named constants' resources via a lwstomized
// deleter until the named constant gets finally registered with the pipeline. As we store these named constants
// inside a vector<> we need to provide the appropriate move copy constructor and move assignment operator to
// handle the unique_ptr<> correctly.
struct NamedConstantWrapper
{
    using LocalPtrWithDeleter = std::unique_ptr<Pipeline::NamedConstant, std::function<void( Pipeline::NamedConstant* )>>;
    NamedConstantWrapper( LocalPtrWithDeleter namedConstant )
        : _namedConstant( std::move( namedConstant ) )
    {
    }
    ~NamedConstantWrapper() {}
    NamedConstantWrapper( NamedConstantWrapper&& other ) noexcept : _namedConstant( std::move( other._namedConstant ) )
    {
    }
    NamedConstantWrapper& operator=( NamedConstantWrapper&& other ) noexcept
    {
        _namedConstant = std::move( other._namedConstant );
        return *this;
    }
    LocalPtrWithDeleter _namedConstant;
};
}  // namespace

static OptixResult computePipelineLaunchParamSpecialization( const std::set<Module*>&   optixModulesSet,
                                                             size_t                     launchParamSize,
                                                             LaunchParamSpecialization& computedSpecializations,
                                                             ErrorDetails&              errDetails )
{
    if( k_forceSpecializationsConsistencyChecksAtLaunch.get() )
    {
        if( !k_enableSpecializations.get() )
            return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR,
                                          RT_DSTRING( "Forced via knob launch time pipeline specialization consistency "
                                                      "check, but specializations are not enabled" ) );
        else if( !k_enableSpecializationsConsistencyChecks.get() )
            return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR,
                                          RT_DSTRING( "Forced via knob launch time pipeline specialization consistency "
                                                      "check, but specialization consistency between modules are not "
                                                      "enabled" ) );
    }
    if( !k_enableSpecializations.get() || !k_enableSpecializationsConsistencyChecks.get() )
        return OPTIX_SUCCESS;

    std::vector<std::pair<const CompileBoundValueEntry*, std::string>> specializedLaunchParam;
    for( Module* module : optixModulesSet )
    {
        const InternalCompileParameters& compileParams = module->getCompileParameters();
        for( unsigned int i = 0, e = compileParams.specializedLaunchParam.size(); i < e; ++i )
            specializedLaunchParam.push_back(
                std::make_pair( &compileParams.specializedLaunchParam[i], module->getModuleIdentifier() ) );
    }
    if( specializedLaunchParam.empty() )
        return OPTIX_SUCCESS;

    std::sort( specializedLaunchParam.begin(), specializedLaunchParam.end(),
               []( const std::pair<const CompileBoundValueEntry*, std::string>& a,
                   const std::pair<const CompileBoundValueEntry*, std::string>& b ) {
                   if( a.first->offset != b.first->offset )
                       return a.first->offset < b.first->offset;
                   return a.first->value.size() < b.first->value.size();
               } );

    computedSpecializations.data.resize( launchParamSize, 0 );

    std::pair<const CompileBoundValueEntry*, std::string>& previous = specializedLaunchParam[0];
    computedSpecializations.addSpecialization( *previous.first, previous.second );

    // Compute overlapping specialization regions and make sure that their values match.
    for( size_t i = 1; i < specializedLaunchParam.size(); ++i )
    {
        std::pair<const CompileBoundValueEntry*, std::string>& lwrrentPair = specializedLaunchParam[i];
        const CompileBoundValueEntry* current = lwrrentPair.first;

        // Compute the end of the previous specialization entry.
        size_t previousEnd = previous.first->offset + previous.first->value.size();
        if( previousEnd >= current->offset )
        {
            // The two specializations overlap
            // <----...     previous
            //   <--...     current
            //
            // Check overlapping data for equality
            const CompileBoundValueEntry* previousEntry = previous.first;
            // start of the overlap in previous
            size_t offsetInPrevious = current->offset - previousEntry->offset;
            // fullOverlap is true if previous completely spans current (or beyond).
            bool fullOverlap = previousEnd >= current->offset + current->value.size();
            // size of the overlap
            size_t overlapSize = fullOverlap ? current->value.size() : previousEnd - current->offset;
            if( memcmp( previousEntry->value.data() + offsetInPrevious, current->value.data(), overlapSize ) )
            {
                errDetails.m_compilerFeedback << "Error: Specialization values mismatch in modules.\n"
                                              << "\tModule: " << previous.second << "\n"
                                              << "\t\tannotation     : " << previousEntry->annotation << "\n"
                                              << "\t\toffset         : " << previousEntry->offset << "\n"
                                              << "\t\tsize           : " << previousEntry->value.size() << "\n"
                                              << "\tModule: " << lwrrentPair.second << "\n"
                                              << "\t\tannotation     : " << current->annotation << "\n"
                                              << "\t\toffset         : " << current->offset << "\n"
                                              << "\t\tsize           : " << current->value.size() << "\n";
                // JB: I think we should look for more inconsistencies.
                return OPTIX_ERROR_ILWALID_VALUE;
            }

            if( !fullOverlap )
            {
                // The end of the current specialization is beyond the end of the previous one.
                // This increases the size of the entry in the computedSpecializations.
                // <---->       previous
                //   <---->     current
                size_t rangeIndex   = computedSpecializations.specializedRanges.size() - 1;
                size_t newDataStart = previousEnd - current->offset;
                computedSpecializations.addSpecialization( *current, newDataStart, lwrrentPair.second );
                previous = lwrrentPair;
            }
            // else current is "swallowed" by previous. No need to adjust previousEnd in that case:
            // <-------->   previous
            //    <--->     current
            //      or
            //    <----->   current
        }
        else
        {
            // Specializations are not overlapping.
            computedSpecializations.addSpecialization( *current, lwrrentPair.second );
            previous = lwrrentPair;
        }
    }
    return OPTIX_SUCCESS;
}

static void collectSubModulesAndInfo( Module* module, std::set<const SubModule*>& subModuleSet, FunctionInfoAclwmulator& functionInfo, const char* entryFunctionName )
{
    std::vector<const SubModule*> subModules = module->getSubModuleAndDependencies( entryFunctionName );
    subModuleSet.insert( subModules.begin(), subModules.end() );
    functionInfo.aclwmulateEntryFunctionInfo( module->getSubModule( entryFunctionName ), entryFunctionName );

}

static OptixResult createPipeline( DeviceContext*                     context,
                                   const OptixPipelineCompileOptions* pipelineCompileOptions,
                                   const OptixPipelineLinkOptions*    pipelineLinkOptions,
                                   const OptixProgramGroup*           programGroups,
                                   unsigned int                       programGroupCount,
                                   OptixPipeline*&                    pipelineAPI,
                                   ErrorDetails&                      errDetails )
{
    OptixResultOneShot result;

    result += checkPipelineCompileOptions( pipelineCompileOptions, context->hasValidationModeDebugExceptions(), errDetails );

    result += checkPipelineLinkOptions( pipelineLinkOptions, errDetails );

    std::set<const SubModule*> subModuleSet;
    size_t                     pipelineParamsSize = Module::s_ilwalidPipelineParamsSize;
    FunctionInfoAclwmulator    functionInfo;
    bool                       foundProgramGroupKindRaygen = false;
    bool                       hasDebugInformation         = false;

    InternalCompileParameters compileParams;
    result += setInternalCompileOptions( compileParams, nullptr, pipelineCompileOptions, context,
                                         /*isBuiltinModule*/ false, /*enableLwstomPrimitiveVA*/ false,
                                         context->isD2IREnabled(), /*privateCompileTimeConstants*/ {}, errDetails );

    for( unsigned int i = 0; i < programGroupCount; ++i )
    {
        OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2( ProgramGroup, programGroup, programGroups[i],
                                                  "programGroups[" + std::to_string( i ) + "]", "optixPipelineCreate" );
        // shared common module handling functionality
        auto moduleHandler =
            [&]( OptixModule moduleAPI, const char* kindStr, const char* entryFunctionName, unsigned int i ) {
#define OPTIX_CHECK_COMMON_COMPILE_OPTIONS( module, kindStr )                                                          \
    result += checkCommonCompileOptions( compileParams, pipelineParamsSize, module->getCompileParameters(),            \
                                         module->getPipelineParamsSize(), i, kindStr, errDetails )
                Module* module;
                implCast( moduleAPI, module );
                OPTIX_CHECK_COMMON_COMPILE_OPTIONS( module, kindStr );
                collectSubModulesAndInfo( module, subModuleSet, functionInfo, entryFunctionName );
                if( module->hasDebugInformation() )
                    hasDebugInformation = true;
        };

        const auto& programGroupImpl = programGroup->getImpl();

        switch( programGroupImpl.kind )
        {
            case OPTIX_PROGRAM_GROUP_KIND_RAYGEN: {
                moduleHandler( programGroupImpl.raygen.module, "raygen.module", programGroupImpl.raygen.entryFunctionName, i );
                foundProgramGroupKindRaygen = true;
                break;
            }

            case OPTIX_PROGRAM_GROUP_KIND_MISS: {
                if( !programGroupImpl.miss.module )
                    break;
                moduleHandler( programGroupImpl.miss.module, "miss.module", programGroupImpl.miss.entryFunctionName, i );
                break;
            }

            case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION: {
                if( !programGroupImpl.exception.module )
                    break;
                moduleHandler( programGroupImpl.exception.module, "exception.module",
                               programGroupImpl.exception.entryFunctionName, i );
                break;
            }

            case OPTIX_PROGRAM_GROUP_KIND_HITGROUP: {
                if( programGroupImpl.hitgroup.moduleCH )
                    moduleHandler( programGroupImpl.hitgroup.moduleCH, "hitgroup.moduleCH",
                                   programGroupImpl.hitgroup.entryFunctionNameCH, i );
                if( programGroupImpl.hitgroup.moduleAH )
                    moduleHandler( programGroupImpl.hitgroup.moduleAH, "hitgroup.moduleAH",
                                   programGroupImpl.hitgroup.entryFunctionNameAH, i );
                if( programGroupImpl.hitgroup.moduleIS )
                    moduleHandler( programGroupImpl.hitgroup.moduleIS, "hitgroup.moduleIS",
                                   programGroupImpl.hitgroup.entryFunctionNameIS, i );
                break;
            }

            case OPTIX_PROGRAM_GROUP_KIND_CALLABLES: {
                if( programGroupImpl.callables.moduleDC )
                    moduleHandler( programGroupImpl.callables.moduleDC, "callables.moduleDC",
                                   programGroupImpl.callables.entryFunctionNameDC, i );
                if( programGroupImpl.callables.moduleCC )
                    moduleHandler( programGroupImpl.callables.moduleCC, "callables.moduleCC",
                                   programGroupImpl.callables.entryFunctionNameCC, i );
                break;
            }
        }

#undef OPTIX_CHECK_COMMON_COMPILE_OPTIONS
#undef OPTIX_CHECK_COMMON_LINK_OPTIONS
    }

    std::set<Module*>             optixModulesSet;
    std::set<RtcCompiledModule>   rtcoreModulesSet;
    std::vector<const SubModule*> deduplicatedSubModules;

    for( const SubModule* subModule : subModuleSet )
    {
        if( rtcoreModulesSet.insert( subModule->getRtcCompiledModule() ).second )
            deduplicatedSubModules.push_back( subModule );
        optixModulesSet.insert( subModule->m_parentModule );
        functionInfo.aclwmulateNonEntryFunctionInfo( subModule );
    }
    functionInfo.moduleCount = optixModulesSet.size();

    LaunchParamSpecialization computedSpecializations;
    if( OptixResult res = computePipelineLaunchParamSpecialization( optixModulesSet, pipelineParamsSize,
                                                                    computedSpecializations, errDetails ) )
        return res;

    bool consistencyChecksEnabled = context->hasValidationModeSpecializationConsistency();
    if( !( k_forceSpecializationsConsistencyChecksAtLaunch.get()
           || ( consistencyChecksEnabled && k_enableSpecializationsConsistencyChecks.get() ) ) )
    {
        computedSpecializations.clear();
    }

    // clang-format off
    errDetails.m_compilerFeedback
        << "Info: Pipeline has " << functionInfo.moduleCount << " module(s), "
        << functionInfo.entryFunctionCount << " entry function(s), "
        << functionInfo.entryFunctionInfo.m_traceCallCount << " trace call(s), "
        << functionInfo.entryFunctionInfo.m_continuationCallableCallCount << " continuation callable call(s), "
        << functionInfo.entryFunctionInfo.m_directCallableCallCount << " direct callable call(s), "
        << functionInfo.entryFunctionInfo.m_basicBlockCount << " basic block(s) in entry functions, "
        << functionInfo.entryFunctionInfo.m_instructionCount << " instruction(s) in entry functions, "
        << functionInfo.nonEntryFunctionInfo.m_count << " non-entry function(s), "
        << functionInfo.nonEntryFunctionInfo.m_basicBlockCount << " basic block(s) in non-entry functions, "
        << functionInfo.nonEntryFunctionInfo.m_instructionCount << " instruction(s) in non-entry functions, "
        << (!hasDebugInformation ? "no" : "has") << " debug information\n";
    // clang-format on

    if( result )
        return result;

    if( !foundProgramGroupKindRaygen )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "Pipeline does not contain any program group of kind raygen" );

    LwdaContextPushPop lwCtx( context );
    if( OptixResult res = lwCtx.init( errDetails ) )
    {
        return res;
    }

    // additionally named constants
    std::vector<NamedConstantWrapper> namedConstants;
    if( context->hasValidationModeDebugExceptions() )
    {
        // this deleter is just locally active to avoid resource leaks - when assigning to pipeline pointer is released
        auto localDeleter = [context, &errDetails]( Pipeline::NamedConstant* ptr ) {
            ptr->m_destroyFunctor( errDetails, *ptr );
            corelib::lwdaDriver().LwMemFreeHost( ptr->m_hostPtr );
            // as the named constant holds (later) the module pointer as well, it gets handled here too
            // Alternative: keep allocated modules separately and handle them outside of the named constant
            if( ptr->m_rtcModule )
                if( const RtcResult rtcResult = context->getRtcore().compiledModuleDestroy( ptr->m_rtcModule ) )
                    errDetails.logDetails( rtcResult, "Error destroying rtcore named constant" );
            delete ptr;
        };
        NamedConstantWrapper::LocalPtrWithDeleter namedConstant(
            new Pipeline::NamedConstant( "__optixValidationModeExceptionCode", sizeof( LWdeviceptr ) ), localDeleter );
        namedConstant->m_destroyFunctor = []( ErrorDetails& errDetails, Pipeline::NamedConstant& namedConstant ) -> OptixResult {
            OptixResultOneShot result;
            if( LWresult error = corelib::lwdaDriver().LwMemFreeHost( namedConstant.m_hostPtr ) )
                result += errDetails.logDetails( error, "Error releasing namedConstant's internal resources" );
            return result;
        };
        // now allocating the pinned memory
        if( LWresult error = corelib::lwdaDriver().LwMemAllocHost( (void**)&namedConstant->m_hostPtr, sizeof( int ) ) )
            return errDetails.logDetails( OPTIX_ERROR_LWDA_ERROR,
                                          "Failed to allocate internal memory for validation mode" );
        // default initializing final value, ie the returned exceptionCode
        *(int*)namedConstant->m_hostPtr = 0;

        namedConstants.push_back( move( namedConstant ) );
    }

    size_t pipelineInstructionCount =
        functionInfo.entryFunctionInfo.m_instructionCount + functionInfo.nonEntryFunctionInfo.m_instructionCount;
    RtcPipelineOptions rtcPipelineOptions =
        generateRtcPipelineOptions( context, compileParams, pipelineLinkOptions, pipelineInstructionCount );

    std::vector<RtcCompiledModule> rtcoreModulesVector( rtcoreModulesSet.begin(), rtcoreModulesSet.end() );

    LWevent conlwrrentLaunchEvent = nullptr;
    {
        unsigned int Flags = LW_EVENT_DISABLE_TIMING;
        if( LWresult lwResult = corelib::lwdaDriver().LwEventCreate( &conlwrrentLaunchEvent, Flags ) )
        {
            return errDetails.logDetails( lwResult, "Failed to allocate event for OptixPipeline" );
        }
    }

    // deleter functor for modules
    auto rtcoreModuleDestroyLambda = [context]( RtcCompiledModule rtcModule ) {
        if( context->getRtcore().compiledModuleDestroy( rtcModule ) )
        {
            lerr << "Error destroying module.\n";
        }
    };
    auto moduleDestroyLambda = [&errDetails]( Module* module ) {
        module->destroy( errDetails );
        delete module;
    };

    std::unique_ptr<RtcCompiledModule_t, decltype( rtcoreModuleDestroyLambda )> launchParamModule( nullptr, rtcoreModuleDestroyLambda );
    if( pipelineCompileOptions->pipelineLaunchParamsVariableName )
    {
        if( pipelineParamsSize == 0 )
        {
            lwCtx.destroy( errDetails );
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "size of params variable is 0" );
        }

        // The launch params might have been optimized away, so we do not produce and error here, but add a warning.
        if( pipelineParamsSize == Module::s_ilwalidPipelineParamsSize )
        {
            std::string msg = std::string( "params variable \"" ) + pipelineCompileOptions->pipelineLaunchParamsVariableName
                              + "\" not found in any module. It might have been optimized away.";
            context->getLogger().callback( DeviceContextLogger::LOG_LEVEL::Warning, "PIPELINE CREATE", msg.c_str() );
        }
        else
        {
            RtcCompiledModule module;
            if( const RtcResult rtcResult = context->getRtcore().compileNamedConstant(
                    context->getRtcDeviceContext(), pipelineCompileOptions->pipelineLaunchParamsVariableName,
                    static_cast<int>( pipelineParamsSize ), &module ) )
                return errDetails.logDetails( rtcResult, "failed to create module for params variable" );

            launchParamModule.reset( module );
            rtcoreModulesVector.push_back( module );
        }
    }

    // do the same work for additional named constants
    for( NamedConstantWrapper& namedConstantWrapper : namedConstants )
    {
        Pipeline::NamedConstant* namedConstant = namedConstantWrapper._namedConstant.get();
        RtcCompiledModule        module;
        if( const RtcResult rtcResult = context->getRtcore().compileNamedConstant(
                context->getRtcDeviceContext(), namedConstant->m_name.c_str(), namedConstant->m_size, &module ) )
        {
            std::string str( "failed to create module for " );
            str += namedConstant->m_name;
            str += " variable";
            return errDetails.logDetails( rtcResult, str );
        }

        namedConstant->m_rtcModule = module;
        rtcoreModulesVector.push_back( module );
    }

    // the default exception handler is added if exceptions are enabled
    std::unique_ptr<Module, decltype( moduleDestroyLambda )> defaultExceptionModule( nullptr, moduleDestroyLambda );
    if( getExceptionFlags( pipelineCompileOptions, context->hasValidationModeDebugExceptions() ) )
    {
        ErrorDetails errDetailsFlags;

        OptixModule               defaultExceptionOptixModule;
        OptixModuleCompileOptions moduleCompileOptionsIn = {};
        OptixModuleCompileOptions moduleCompileOptions;
        OptixResult res = translateABI_OptixModuleCompileOptions( &moduleCompileOptionsIn, pipelineCompileOptions, context->getAbiVersion(),
                                                                  &moduleCompileOptions, errDetails );

        if( !res )
            res = createModule( context, &moduleCompileOptions, pipelineCompileOptions,
                                optix::data::getDefaultExceptionSources()[1],
                                optix::data::getDefaultExceptionSourceSizes()[0], &defaultExceptionOptixModule,
                                /*allowUnencryptedIfEncryptionIsEnabled*/ true, /*isBuiltinModule=*/true,
                                /*enableLwstomPrimitiveVA=*/true, context->isD2IREnabled(), /*privateCompileTimeConstants*/ {},
                                /*logString*/ nullptr, /*logStringSize*/ nullptr, errDetailsFlags );

        if( res )
        {
            std::swap( errDetailsFlags.m_compilerFeedback, errDetails.m_compilerFeedback );
            std::swap( errDetailsFlags.m_description, errDetails.m_description );
            return errDetails.logDetails( res, "failed to create module for default exception handler" );
        }

        Module* module = nullptr;
        implCast( defaultExceptionOptixModule, module );

        for( const SubModule* subModule : module->getSubModules() )
            rtcoreModulesVector.push_back( subModule->getRtcCompiledModule() );

        defaultExceptionModule.reset( module );
    }

    std::unique_ptr<Module, decltype( moduleDestroyLambda )> sparseTextureModule( nullptr, moduleDestroyLambda );

    // Check whether any modules use texture footprint intrinsic, in which case we load the sparse texture module.
    bool footprintUsed = false;
    for( const SubModule* subModule : subModuleSet )
    {
        if( subModule->usesTextureIntrinsic() )
        {
            footprintUsed = true;
            break;
        }
    }
    if( footprintUsed )
    {
        ErrorDetails errDetailsFlags;

        OptixModule               textureFootprintWrappersModule;
        OptixModuleCompileOptions moduleCompileOptionsIn{};
        moduleCompileOptionsIn.maxRegisterCount = 128;
        moduleCompileOptionsIn.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptionsIn.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT;
        OptixModuleCompileOptions moduleCompileOptions;
        OptixResult res = translateABI_OptixModuleCompileOptions( &moduleCompileOptionsIn, pipelineCompileOptions, context->getAbiVersion(),
                                                                  &moduleCompileOptions, errDetails );

        const char* bitcode;
        size_t      bitcodeLen;
        if( context->getComputeCapability() >= 75 && !compileParams.useSoftwareTextureFootprint )
        {
            bitcode    = optix::data::getO7TextureFootprintWrappersHWData();
            bitcodeLen = optix::data::getO7TextureFootprintWrappersHWDataLength();
        }
        else
        {
            bitcode    = optix::data::getO7TextureFootprintWrappersSWData();
            bitcodeLen = optix::data::getO7TextureFootprintWrappersSWDataLength();
        }

        // do custom abi lowering

        if( !res )
            res = createModule( context, &moduleCompileOptions, pipelineCompileOptions, bitcode, bitcodeLen, &textureFootprintWrappersModule,
                                /*allowUnencryptedIfEncryptionIsEnabled*/ true, /*isBuiltinModule*/ true,
                                /*enableLwstomPrimitiveVA*/ true,
                                /*useD2IR*/ true, /*privateCompileTimeConstants*/ {}, /*logString*/ nullptr,
                                /*logStringSize*/ nullptr, errDetailsFlags );

        if( res )
        {
            std::swap( errDetailsFlags.m_compilerFeedback, errDetails.m_compilerFeedback );
            std::swap( errDetailsFlags.m_description, errDetails.m_description );
            return errDetails.logDetails( res, "failed to create module for sparse textures" );
        }

        Module* module = nullptr;
        implCast( textureFootprintWrappersModule, module );

        for( const SubModule* subModule : module->getSubModules() )
            rtcoreModulesVector.push_back( subModule->getRtcCompiledModule() );

        sparseTextureModule.reset( module );
    }

    RtcCompileOptions rtcCompileOptions = {};
    if( setRtcCompileOptions( rtcCompileOptions, compileParams, errDetails ) )
    {
        lwCtx.destroy( errDetails );
        return result;
    }

    RtcPipeline rtcPipeline;
    if( const RtcResult rtcResult = context->getRtcore().pipelineCreate(
            context->getRtcDeviceContext(), &rtcPipelineOptions, &rtcCompileOptions, rtcoreModulesVector.data(),
            static_cast<int>( rtcoreModulesVector.size() ), &rtcPipeline ) )
    {
        lwCtx.destroy( errDetails );

        // If pipeline creation failed, we check if we can identify potential symbol resolve problems.
        if( OptixResult res = resolveModuleSymbols( deduplicatedSubModules, errDetails ) )
            return res;

        return errDetails.logDetails( rtcResult, "failed to create pipeline" );
    }

    // This rtcore call on the pipeline (and the subsequent ones below) are not locked since this thread just created
    // the pipeline and no other thread has access to it yet. (Some of the rtcore calls might also be thread-safe even
    // if it is not guaranteed.)
    Rtlw64 launchBufferSizeInBytes = 0;
    Rtlw64 launchBufferAlignment   = 0;
    if( const RtcResult rtcResult =
            context->getRtcore().pipelineGetLaunchBufferInfo( rtcPipeline, &launchBufferSizeInBytes, &launchBufferAlignment ) )
    {
        result += errDetails.logDetails( rtcResult, "failed to get launch buffer info" );
    }

    Rtlw64 launchParamOffset = 0, launchParamSizeInBytes = 0;
    if( pipelineCompileOptions->pipelineLaunchParamsVariableName && pipelineParamsSize != Module::s_ilwalidPipelineParamsSize )
    {
        RtcResult rtcResultGetLaunchParamOffset = context->getRtcore().pipelineGetNamedConstantInfo(
            rtcPipeline, pipelineCompileOptions->pipelineLaunchParamsVariableName, &launchParamOffset, &launchParamSizeInBytes );
        if( rtcResultGetLaunchParamOffset )
        {
            result += errDetails.logDetails( rtcResultGetLaunchParamOffset, "failed to get launch param offset" );
        }
    }

    for( NamedConstantWrapper& namedConstantWrapper : namedConstants )
    {
        Pipeline::NamedConstant* namedConstant                  = namedConstantWrapper._namedConstant.get();
        RtcResult                rtcResultGetPinnedMemoryOffset = context->getRtcore().pipelineGetNamedConstantInfo(
            rtcPipeline, namedConstant->m_name.c_str(), &namedConstant->m_memoryOffset, &namedConstant->m_memorySizeInBytes );
        if( rtcResultGetPinnedMemoryOffset )
        {
            std::string msg( "failed to get " );
            msg += namedConstant->m_name;
            msg += " offset";
            result += errDetails.logDetails( rtcResultGetPinnedMemoryOffset, msg );
        }
    }

    Rtlw64                         toolsOutputSizeInBytes = 0;
    RtcPipelineInfoShaderInfo      shaderInfo             = {};
    RtcPipelineInfoProfileMetadata profileMetadata        = {};
    if( rtcCompileOptions.enabledTools & RTC_TOOLS_FLAG_PROFILING )
    {
        if( k_enableProfiling.get().empty() )
            return errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR,
                                          "Profiling enabled outside of knob.  Enable knob for profiling." );

        if( const RtcResult rtcResult = context->getRtcore().pipelineGetInfo( rtcPipeline, RTC_PIPELINE_INFO_TYPE_SHADER_INFO,
                                                                              sizeof( RtcPipelineInfoShaderInfo ), &shaderInfo ) )
            return errDetails.logDetails( rtcResult, "Unable to get shader info for profiling" );

        if( const RtcResult rtcResult = context->getRtcore().pipelineGetInfo( rtcPipeline, RTC_PIPELINE_INFO_TYPE_PROFILING_METADATA,
                                                                              sizeof( profileMetadata ), &profileMetadata ) )
            return errDetails.logDetails( rtcResult, "Unable to get pipeline info for profiling" );
        toolsOutputSizeInBytes = profileMetadata.outputDataSizeInBytes;
    }

    if( result )
    {
        if( const RtcResult rtcResult = context->getRtcore().pipelineDestroy( rtcPipeline ) )
            errDetails.logDetails( rtcResult, "Error destroying RTX pipeline" );
        lwCtx.destroy( errDetails );
        return result;
    }

    if( OptixResult res = lwCtx.destroy( errDetails ) )
    {
        if( const RtcResult rtcResult = context->getRtcore().pipelineDestroy( rtcPipeline ) )
            errDetails.logDetails( rtcResult, "Error destroying RTX pipeline" );
        return res;
    }

    // transfer namedConstants from array of wrappers to pipeline via this temporary(!) array
    std::vector<Pipeline::NamedConstant*> namedConstantsArray;
    for( NamedConstantWrapper& namedConstantWrapper : namedConstants )
        namedConstantsArray.push_back( namedConstantWrapper._namedConstant.release() );
    std::unique_ptr<Pipeline> pipeline( new Pipeline(
        context, rtcPipeline, launchParamModule.release(), defaultExceptionModule.release(), *pipelineCompileOptions,
        *pipelineLinkOptions, launchBufferSizeInBytes, launchBufferAlignment, launchParamOffset, launchParamSizeInBytes,
        toolsOutputSizeInBytes, shaderInfo, profileMetadata, pipelineParamsSize == Module::s_ilwalidPipelineParamsSize,
        std::move( computedSpecializations ), conlwrrentLaunchEvent, std::move( namedConstantsArray ), hasDebugInformation ) );
    context->registerPipeline( pipeline.get(), errDetails );
    *pipelineAPI = apiCast( pipeline.release() );

    return OPTIX_SUCCESS;
}
}  // namespace optix_exp

extern "C" OptixResult optixPipelineCreate( OptixDeviceContext                 contextAPI,
                                            const OptixPipelineCompileOptions* pipelineCompileOptions,
                                            const OptixPipelineLinkOptions*    pipelineLinkOptions,
                                            const OptixProgramGroup*           programGroups,
                                            unsigned int                       numProgramGroups,
                                            char*                              logString,
                                            size_t*                            logStringSize,
                                            OptixPipeline*                     pipelineAPI )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT_W_LOG_STRING();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::PIPELINE_CREATE );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT_W_LOG_STRING( pipelineCompileOptions );
    OPTIX_CHECK_NULL_ARGUMENT_W_LOG_STRING( pipelineLinkOptions );
    OPTIX_CHECK_NULL_ARGUMENT_W_LOG_STRING( pipelineAPI );
    OPTIX_CHECK_NULL_ARGUMENT_W_LOG_STRING( programGroups );
    OPTIX_CHECK_ZERO_ARGUMENT_W_LOG_STRING( numProgramGroups );

    *pipelineAPI = nullptr;

    try
    {
        optix_exp::ErrorDetails     errDetails;
        OptixPipelineCompileOptions translatedPipelineCompileOptions;
        if( OptixResult result = optix_exp::translateABI_PipelineCompileOptions(
                pipelineCompileOptions, context->getAbiVersion(), &translatedPipelineCompileOptions, errDetails ) )
        {
            clog.sendError( errDetails );
            optix_exp::copyCompileDetails( errDetails.m_description, logString, logStringSize );
            return result;
        }
        OptixPipelineLinkOptions translatedPipelineLinkOptions;
        if( OptixResult result = optix_exp::translateABI_PipelineLinkOptions( pipelineLinkOptions, context->getAbiVersion(),
                                                                              &translatedPipelineLinkOptions, errDetails ) )
        {
            clog.sendError( errDetails );
            optix_exp::copyCompileDetails( errDetails.m_description, logString, logStringSize );
            return result;
        }
        OptixResult result = createPipeline( context, &translatedPipelineCompileOptions, &translatedPipelineLinkOptions,
                                             programGroups, numProgramGroups, pipelineAPI, errDetails );

        optix_exp::DeviceContextLogger::LOG_LEVEL level = optix_exp::DeviceContextLogger::LOG_LEVEL::Print;
        if( result )
        {
            level = optix_exp::DeviceContextLogger::LOG_LEVEL::Error;
            std::ostringstream compileFeedback2;
            // putting error first in the logString to avoid it falling off if the buffer is too small
            compileFeedback2 << "COMPILE ERROR: " << errDetails.m_description << "\n";
            compileFeedback2 << errDetails.m_compilerFeedback.str();
            std::swap( compileFeedback2, errDetails.m_compilerFeedback );
        }
        if( errDetails.m_compilerFeedback.str().length() > 0 )
            clog.callback( level, "COMPILE FEEDBACK", errDetails.m_compilerFeedback.str().c_str() );
        optix_exp::copyCompileDetails( errDetails.m_compilerFeedback, logString, logStringSize );

        return result;
    }
    OPTIX_API_EXCEPTION_CHECK_W_LOG_STRING;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixPipelineDestroy( OptixPipeline pipelineAPI )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( Pipeline, pipeline, "OptixPipeline" );
    SCOPED_LWTX_RANGE( pipeline->getDeviceContext()->getLWTXProfiler(), LWTXRegisteredString::PIPELINE_DESTROY );
    optix_exp::DeviceContextLogger& clog = pipeline->getDeviceContext()->getLogger();

    try
    {
        optix_exp::ErrorDetails errDetails;
        OptixResult             result = pipeline->destroy( errDetails );
        if( result )
            clog.sendError( errDetails );
        delete pipeline;
        return result;
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixPipelineSetStackSize( OptixPipeline pipelineAPI,
                                                  unsigned int  directCallableStackSizeFromTraversal,
                                                  unsigned int  directCallableStackSizeFromState,
                                                  unsigned int  continuationStackSize,
                                                  unsigned int  maxTraversableGraphDepth )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( Pipeline, pipeline, "OptixPipeline" );
    SCOPED_LWTX_RANGE( pipeline->getDeviceContext()->getLWTXProfiler(), LWTXRegisteredString::PIPELINE_SET_STACK_SIZE );
    optix_exp::DeviceContextLogger& clog = pipeline->getDeviceContext()->getLogger();

    try
    {
        optix_exp::ErrorDetails errDetails;
        OptixResult             optixResult = OPTIX_SUCCESS;

        {
            if( pipeline->getDeviceContext()->getMaxSceneGraphDepth() < maxTraversableGraphDepth )
                optixResult = errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                     "\"maxTraversableGraphDepth\" exceeds the maximum scene graph "
                                                     "depth" );
            else if( maxTraversableGraphDepth == 0 )
                optixResult =
                    errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "\"maxTraversableGraphDepth\" equals 0" );
            else if( pipeline->getPipelineCompileOptions().traversableGraphFlags != OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY )
            {
                if( maxTraversableGraphDepth > 2 )
                    optixResult = errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                         "Multi-level graphs are disabled but "
                                                         "\"maxTraversableGraphDepth\" is larger than 2" );
                else if( pipeline->getPipelineCompileOptions().traversableGraphFlags == OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
                         && maxTraversableGraphDepth != 1 )
                    optixResult = errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                         "Only single gas graphs are enabled but "
                                                         "\"maxTraversableGraphDepth\" is not 1" );
                else if( pipeline->getPipelineCompileOptions().traversableGraphFlags == OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING
                         && maxTraversableGraphDepth != 2 )
                    optixResult = errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                         "Only single ias graphs are enabled but "
                                                         "\"maxTraversableGraphDepth\" is not 2" );
            }

            if( optixResult == OPTIX_SUCCESS )
            {
                RtcResult                   rtcResult = RTC_SUCCESS;
                std::lock_guard<std::mutex> lock( pipeline->getRtcoreMutex() );
                rtcResult = pipeline->getDeviceContext()->getRtcore().pipelineSetStackSize(
                    pipeline->getRtcPipeline(), directCallableStackSizeFromTraversal, directCallableStackSizeFromState,
                    continuationStackSize, maxTraversableGraphDepth );

                if( rtcResult )
                {
                    std::string msg = "failed to set pipeline stack size";
                    if( rtcResult == RTC_ERROR_ILWALID_STACK_SIZE && pipeline->hasDebugInformation() )
                    {
                        msg +=
                            ". This failure might be due to input being compiled with full debug info (-G), which "
                            "increases stack size requirements. Consider lineinfo generation (-lineinfo) only.";
                    }
                    optixResult = errDetails.logDetails( rtcResult, msg );
                }
            }
        }

        if( optixResult != OPTIX_SUCCESS )
            clog.sendError( errDetails );

        return optixResult;
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

namespace optix_exp {
static OptixResult checkPipelineLaunchParams( Pipeline*                      pipeline,
                                              LWdeviceptr                    pipelineParams,
                                              size_t                         pipelineParamsSize,
                                              const OptixShaderBindingTable* sbt,
                                              ErrorDetails&                  errDetails )
{
    if( !pipeline->getIgnorePipelineLaunchParamsAtLaunch() )
    {
        if( pipelineParams && pipelineParamsSize == 0 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          "\"pipelineParams\" specified with zero \"pipelineParamsSize\"" );
        if( !pipelineParams && pipelineParamsSize > 0 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          "\"pipelineParamsSize\" specified with null \"pipelineParams\"" );

        if( pipelineParams && pipeline->getPipelineCompileOptions().pipelineLaunchParamsVariableName == nullptr )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          "no variable name to bind \"pipelineParams\" to specified in pipeline "
                                          "compile options" );

        if( !pipelineParams && pipeline->getPipelineCompileOptions().pipelineLaunchParamsVariableName != nullptr )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          "\"pipelineParams\" is null, but pipeline compile options specify a variable "
                                          "name to bind \"pipelineParams\" to" );

        if( pipelineParams && pipeline->getLaunchParamSizeInBytes() < pipelineParamsSize )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          corelib::stringf( "pipeline launch param size configured by pipeline link "
                                                            "options (%zu) is smaller than pipelineParamsSize (%zu)",
                                                            pipeline->getLaunchParamSizeInBytes(), pipelineParamsSize ) );
    }

    if( sbt->raygenRecord == 0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "\"sbt->raygenRecord\" is null" );

    if( sbt->raygenRecord % OPTIX_SBT_RECORD_ALIGNMENT != 0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "\"sbt->raygenRecord\" points to a memory area which is not correctly aligned" );


    if( sbt->exceptionRecord % OPTIX_SBT_RECORD_ALIGNMENT != 0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "\"sbt->exceptionRecord\" points to a memory area which is not correctly "
                                      "aligned" );


    if( sbt->missRecordBase == 0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "\"sbt->missSbtRecord\" is null" );

    if( sbt->missRecordBase % OPTIX_SBT_RECORD_ALIGNMENT != 0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "\"sbt->missSbtRecord\" points to a memory area which is not correctly aligned" );

    if( sbt->missRecordCount == 0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "\"sbt->missRecordCount\" is zero" );

    if( sbt->missRecordStrideInBytes == 0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "\"sbt->missRecordStrideInBytes\" is zero" );

    if( sbt->missRecordStrideInBytes % OPTIX_SBT_RECORD_ALIGNMENT != 0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "\"sbt->missRecordStrideInBytes\" is not a multiple of "
                                      "OPTIX_SBT_RECORD_ALIGNMENT" );

    if( sbt->hitgroupRecordBase == 0 )
    {
        if( sbt->hitgroupRecordCount > 0 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          "\"sbt->hitgroupRecordBase\" is null, but "
                                          "\"sbt->hitgroupRecordCount\" is non-zero" );
    }
    else
    {
        if( sbt->hitgroupRecordBase % OPTIX_SBT_RECORD_ALIGNMENT != 0 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          "\"sbt->hitgroupRecordBase\" points to a memory area which is not correctly "
                                          "aligned" );

        if( sbt->hitgroupRecordCount == 0 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "\"sbt->hitgroupRecordCount\" is zero" );

        if( sbt->hitgroupRecordStrideInBytes == 0 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "\"sbt->hitgroupRecordStrideInBytes\" is zero" );

        if( sbt->hitgroupRecordStrideInBytes % OPTIX_SBT_RECORD_ALIGNMENT != 0 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          "\"sbt->hitgroupRecordStrideInBytes\" is not a multiple of "
                                          "OPTIX_SBT_RECORD_ALIGNMENT" );
    }

    if( sbt->callablesRecordBase == 0 )
    {
        if( sbt->callablesRecordCount > 0 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          "\"sbt->callablesRecordBase\" is null, but "
                                          "\"sbt->callablesRecordCount\" is non-zero" );
    }
    else
    {
        if( sbt->callablesRecordBase % OPTIX_SBT_RECORD_ALIGNMENT != 0 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          "\"sbt->callablesRecordBase\" points to a memory area which is not correctly "
                                          "aligned" );

        if( sbt->callablesRecordCount == 0 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          "\"sbt->callablesRecordBase\" is non-null, but "
                                          "\"sbt->callablesRecordCount\" is zero" );

        if( sbt->callablesRecordStrideInBytes == 0 )
            return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                          "\"sbt->callablesRecordBase\" is non-null, but "
                                          "\"sbt->callablesRecordStrideInBytes\" is zero" );
    }

    if( sbt->callablesRecordStrideInBytes % OPTIX_SBT_RECORD_ALIGNMENT != 0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "\"sbt->callablesRecordStrideInBytes\" is not a multiple of "
                                      "OPTIX_SBT_RECORD_ALIGNMENT" );

    return OPTIX_SUCCESS;
}

// TODO: Use the version from optix_7_enum_printers.h - but this causes compile issues right now
// Raven_bfm\apps\optix\exp\pipeline\Pipeline.cpp(1469): error C2733: 'optixLaunch': second C linkage of overloaded function not allowed
namespace {
// Exact copy from optix_7_enum_printers.h
std::string toString( OptixExceptionCodes value )
{
    switch( value )
    {
        case OPTIX_EXCEPTION_CODE_STACK_OVERFLOW:
            return "OPTIX_EXCEPTION_CODE_STACK_OVERFLOW";
        case OPTIX_EXCEPTION_CODE_TRACE_DEPTH_EXCEEDED:
            return "OPTIX_EXCEPTION_CODE_TRACE_DEPTH_EXCEEDED";
        case OPTIX_EXCEPTION_CODE_TRAVERSAL_DEPTH_EXCEEDED:
            return "OPTIX_EXCEPTION_CODE_TRAVERSAL_DEPTH_EXCEEDED";
        case OPTIX_EXCEPTION_CODE_TRAVERSAL_ILWALID_TRAVERSABLE:
            return "OPTIX_EXCEPTION_CODE_TRAVERSAL_ILWALID_TRAVERSABLE";
        case OPTIX_EXCEPTION_CODE_TRAVERSAL_ILWALID_MISS_SBT:
            return "OPTIX_EXCEPTION_CODE_TRAVERSAL_ILWALID_MISS_SBT";
        case OPTIX_EXCEPTION_CODE_TRAVERSAL_ILWALID_HIT_SBT:
            return "OPTIX_EXCEPTION_CODE_TRAVERSAL_ILWALID_HIT_SBT";
        case OPTIX_EXCEPTION_CODE_UNSUPPORTED_PRIMITIVE_TYPE:
            return "OPTIX_EXCEPTION_CODE_UNSUPPORTED_PRIMITIVE_TYPE";
        case OPTIX_EXCEPTION_CODE_ILWALID_RAY:
            return "OPTIX_EXCEPTION_CODE_ILWALID_RAY";
        case OPTIX_EXCEPTION_CODE_CALLABLE_PARAMETER_MISMATCH:
            return "OPTIX_EXCEPTION_CODE_CALLABLE_PARAMETER_MISMATCH";
        case OPTIX_EXCEPTION_CODE_BUILTIN_IS_MISMATCH:
            return "OPTIX_EXCEPTION_CODE_BUILTIN_IS_MISMATCH";
        case OPTIX_EXCEPTION_CODE_CALLABLE_ILWALID_SBT:
            return "OPTIX_EXCEPTION_CODE_CALLABLE_ILWALID_SBT";
        case OPTIX_EXCEPTION_CODE_CALLABLE_NO_DC_SBT_RECORD:
            return "OPTIX_EXCEPTION_CODE_CALLABLE_NO_DC_SBT_RECORD";
        case OPTIX_EXCEPTION_CODE_CALLABLE_NO_CC_SBT_RECORD:
            return "OPTIX_EXCEPTION_CODE_CALLABLE_NO_CC_SBT_RECORD";
        case OPTIX_EXCEPTION_CODE_UNSUPPORTED_SINGLE_LEVEL_GAS:
            return "OPTIX_EXCEPTION_CODE_UNSUPPORTED_SINGLE_LEVEL_GAS";
        case OPTIX_EXCEPTION_CODE_ILWALID_VALUE_ARGUMENT_0:
            return "OPTIX_EXCEPTION_CODE_ILWALID_VALUE_ARGUMENT_0";
        case OPTIX_EXCEPTION_CODE_ILWALID_VALUE_ARGUMENT_1:
            return "OPTIX_EXCEPTION_CODE_ILWALID_VALUE_ARGUMENT_1";
        case OPTIX_EXCEPTION_CODE_ILWALID_VALUE_ARGUMENT_2:
            return "OPTIX_EXCEPTION_CODE_ILWALID_VALUE_ARGUMENT_2";
        case OPTIX_EXCEPTION_CODE_UNSUPPORTED_DATA_ACCESS:
            return "OPTIX_EXCEPTION_CODE_UNSUPPORTED_DATA_ACCESS";
        case OPTIX_EXCEPTION_CODE_PAYLOAD_TYPE_MISMATCH:
            return "OPTIX_EXCEPTION_CODE_PAYLOAD_TYPE_MISMATCH";
    }
    return std::to_string( static_cast<unsigned long>( value ) );
}

}  // namespace

static OptixResult launchPipeline( Pipeline*                      pipeline,
                                   LWstream                       stream,
                                   LWdeviceptr                    pipelineParams,
                                   size_t                         pipelineParamsSize,
                                   const OptixShaderBindingTable* sbt,
                                   unsigned int                   width,
                                   unsigned int                   height,
                                   unsigned int                   depth,
                                   ErrorDetails&                  errDetails )
{
    if( const OptixResult result = checkPipelineLaunchParams( pipeline, pipelineParams, pipelineParamsSize, sbt, errDetails ) )
        return result;
    OPTIX_CHECK_VALIDATION_MODE_LWRRENT_LWDA_CONTEXT( pipeline->getDeviceContext() );
    OPTIX_CHECK_VALIDATION_MODE_STREAM_STATE( pipeline->getDeviceContext(), stream );

    Rtlw64    scratchBufferSizeInBytesMin, scratchBufferSizeInBytes, scratchBufferAlignment;
    RtcResult resultGetScratchBufferInfo = RTC_SUCCESS;
    {
        std::lock_guard<std::mutex> lock( pipeline->getRtcoreMutex() );
        resultGetScratchBufferInfo = pipeline->getDeviceContext()->getRtcore().pipelineGetScratchBufferInfo3D(
            pipeline->getRtcPipeline(), width, height, depth, &scratchBufferSizeInBytesMin, &scratchBufferSizeInBytes,
            &scratchBufferAlignment );
    }
    if( resultGetScratchBufferInfo )
        return errDetails.logDetails( resultGetScratchBufferInfo, "Unable to get launch resource size" );

    // RTcore requires a command list.  Because it is a lightweight object, we will
    // allocate and deallocate it every launch.  The ultimate fix (TODO) is to add an API
    // to rtcore to create an stack allocated command list.
    ScopedCommandList commandList( pipeline->getDeviceContext() );
    if( const OptixResult result = commandList.init( stream, errDetails ) )
        return result;

    // Use the default exception program when the user did not provide an exception program
    const bool useDefaultException = ( sbt->exceptionRecord == 0 ) && ( pipeline->getDefaultExceptionModule() != nullptr );

    // Compute total size of launch state buffer. Round up section sizes such that the sections have the required
    // alignment provided the base addresses has the maximum alignment.
    size_t launchBufferAlignment = pipeline->getLaunchBufferAlignment();
    if( scratchBufferAlignment == 0 )
        scratchBufferAlignment  = 1;
    size_t toolsBufferAlignment = 16;
    size_t maxAlignment = std::max( std::max( launchBufferAlignment, (size_t)scratchBufferAlignment ), toolsBufferAlignment );

    size_t launchBufferSizeInBytes          = pipeline->getLaunchBufferSizeInBytes();
    size_t launchBufferSizeInBytesRoundedUp = corelib::roundUp( launchBufferSizeInBytes, (size_t)scratchBufferAlignment );

    size_t scratchBufferSizeInBytesRoundedUp = corelib::roundUp( (size_t)scratchBufferSizeInBytes, toolsBufferAlignment );

    size_t toolsOutputSizeInBytesRoundedUp =
        corelib::roundUp( pipeline->getToolsOutputSizeInBytes(), (size_t)OPTIX_SBT_RECORD_ALIGNMENT );

    // When the default exception program is enabled, an extra exception sbt record is concatenate at the end of the launch buffer.
    const size_t sbtRecordSize =
        corelib::roundUp( OPTIX_SBT_RECORD_HEADER_SIZE + sizeof( ExceptionSbtRecordData ), OPTIX_SBT_RECORD_HEADER_SIZE );
    size_t defaultExceptionRecordSizeInBytes = useDefaultException ? sbtRecordSize : 0;

    // Round up final size to maxAlignment to ensure that subsequent request are aligned to (at least) this value.
    size_t launchStateSize = corelib::roundUp( launchBufferSizeInBytesRoundedUp + scratchBufferSizeInBytesRoundedUp
                                                   + toolsOutputSizeInBytesRoundedUp + defaultExceptionRecordSizeInBytes,
                                               (size_t)maxAlignment );

    // Round up final size to 128 (required by LaunchResources, but maxAlignment is probably higher).
    launchStateSize = corelib::roundUp( launchStateSize, (size_t)128 );

    // For Turing, we need to prime the TTU when returning from an idle state by opening a dummy TTU
    // transaction. Otherwise, the TTU can return false misses. This has been fixed in SM86+.
    // See: Bug 3147856, Turing HW Bug 2648362.
    if( k_gpuWarmupEnabled.get() && pipeline->getDeviceContext()->getComputeCapability() == 75
        && pipeline->getDeviceContext()->hasTTU() )
    {
        // Kicking the watchdog returns the elapsed time since the last kick and resets the internal timer.
        const float kickThreshold = std::max( k_gpuWarmupThreshold.get(), 0.0f );
        if( pipeline->getDeviceContext()->getTTUWatchdog().kick() >= kickThreshold )
        {
            // The warm-up kernel requires the current LWCA context to match the DeviceContext's.
            // It's not required for optixLaunch in general because it uses the driver API with
            // asynchronous calls.
            LwdaContextPushPop lwCtx( pipeline->getDeviceContext() );
            if( OptixResult result = lwCtx.init( errDetails ) )
                return result;

            const unsigned int smCount = pipeline->getDeviceContext()->getMultiProcessorCount();
            if( OptixResult result = pipeline->getDeviceContext()->getGpuWarmup().launch( smCount, errDetails ) )
                return result;

            if( OptixResult result = lwCtx.destroy( errDetails ) )
                return result;
        }
    }

    // Between acquireResource/releaseResource is a critical section where it blocks out
    // all threads.  Make this section as minimal as possible.
    LWdeviceptr launchState;
    if( const OptixResult result =
            pipeline->getDeviceContext()->getLaunchResources().acquireResource( stream, launchStateSize, launchState, errDetails ) )
    {
        return errDetails.logDetails( result, "Failed to allocate internal launch buffer" );
    }

    OptixResultOneShot result;

    {
        RtcGpuVA launchBufferVA           = launchState;
        RtcGpuVA scratchBufferVA          = launchState + launchBufferSizeInBytesRoundedUp;
        RtcGpuVA toolsOutputVA            = scratchBufferVA + scratchBufferSizeInBytesRoundedUp;
        RtcGpuVA defaultExceptionRecordVA = toolsOutputVA + toolsOutputSizeInBytesRoundedUp;

        if( launchBufferVA % maxAlignment != 0 )
        {
            result +=
                errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, "Launch state buffer has incorrect alignment" );
            goto launch_cleanup;
        }

        if( scratchBufferVA % scratchBufferAlignment != 0 )
        {
            result += errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, "Scratch buffer has incorrect alignment" );
            goto launch_cleanup;
        }

        if( toolsOutputVA % toolsBufferAlignment != 0 )
        {
            result += errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, "Tools buffer has incorrect alignment" );
            goto launch_cleanup;
        }

        if( defaultExceptionRecordVA % OPTIX_SBT_RECORD_ALIGNMENT != 0 )
        {
            result +=
                errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR, "Default exception record has incorrect alignment" );
            goto launch_cleanup;
        }

        RtcGpuVA exceptionRecordVA = sbt->exceptionRecord;

        if( useDefaultException )
        {
            // upload the default exception sbt record
            std::vector<char> defaultExceptionSbtRecord( sbtRecordSize, 0 );

            Module* module = pipeline->getDefaultExceptionModule();
            std::string programName = module->getMangledName( "__exception__default", OPTIX_PAYLOAD_TYPE_DEFAULT, ST_EXCEPTION );
            RtcCompiledModule rtcModule = nullptr;

            ExceptionSbtRecordData* record =
                reinterpret_cast<ExceptionSbtRecordData*>( defaultExceptionSbtRecord.data() + OPTIX_SBT_RECORD_HEADER_SIZE );
            record->exceptionCounter      = 0;
            record->exceptionMessageLimit = 5;

            Rtlw32 programIndex = ~0;
            if( OptixResult result = module->getRtcCompiledModuleAndProgramIndex( programName, rtcModule, programIndex, errDetails ) )
                goto launch_cleanup;

            if( const RtcResult rtcResult = pipeline->getDeviceContext()->getRtcore().packSbtRecordHeader(
                    pipeline->getDeviceContext()->getRtcDeviceContext(), rtcModule, programIndex, nullptr, ~0, nullptr,
                    ~0, defaultExceptionSbtRecord.data() ) )
            {
                result += errDetails.logDetails( rtcResult, "Unable to pack default exception record header" );
                goto launch_cleanup;
            }

            if( LWresult lwdaErr = corelib::lwdaDriver().LwMemcpyAsync( (LWdeviceptr)defaultExceptionRecordVA,
                                                                        (LWdeviceptr)defaultExceptionSbtRecord.data(),
                                                                        sbtRecordSize, stream ) )
            {
                result += errDetails.logDetails( lwdaErr, "Failed to copy default exception record" );
                goto launch_cleanup;
            }

            exceptionRecordVA = defaultExceptionRecordVA;
        }

        if( pipelineParams && !pipeline->getIgnorePipelineLaunchParamsAtLaunch() )
        {
            if( LWresult lwdaErr =
                    corelib::lwdaDriver().LwMemcpyDtoDAsync( ( LWdeviceptr )( launchState + pipeline->getLaunchParamOffset() ),
                                                             (LWdeviceptr)pipelineParams, pipelineParamsSize, stream ) )
            {
                result += errDetails.logDetails( lwdaErr, "Failed to copy internal launch buffer" );
                goto launch_cleanup;
            }
        }

        for( const Pipeline::NamedConstant* namedConstant : pipeline->getAdditionalNamedConstants() )
        {
            if( LWresult lwdaErr = corelib::lwdaDriver().LwMemcpyHtoDAsync(
                    ( LWdeviceptr )( launchState + namedConstant->m_memoryOffset ), (void*)&namedConstant->m_hostPtr,
                    namedConstant->m_memorySizeInBytes, stream ) )
            {
                std::string msg( "Failed to copy named constant " );
                msg += namedConstant->m_name;
                result += errDetails.logDetails( lwdaErr, msg );
                goto launch_cleanup;
            }
        }

        // WAR for conlwrrent launches.  Wait on the previous pipeline launch to finish before
        // starting a new launch.
        if( LWresult lwResult = corelib::lwdaDriver().LwStreamWaitEvent( stream, pipeline->getConlwrrentLaunchEvent(), 0 ) )
        {
            result += errDetails.logDetails( lwResult,
                                             "Error synchronizing on OptixPipeline's previous launch to prevent "
                                             "conlwrrent launches" );
            goto launch_cleanup;
        }

        if( pipeline->getDeviceContext()->isLwptiProfilingEnabled() )
        {
            if( const OptixResult res = pipeline->getDeviceContext()->getLwptiProfiler().beginProfile( errDetails ) )
            {
                result += res;
                goto launch_cleanup;
            }
        }

        RtcResult launchResult = RTC_SUCCESS;
        {
            std::lock_guard<std::mutex> lock( pipeline->getRtcoreMutex() );
            launchResult = pipeline->getDeviceContext()->getRtcore().launch3D(
                commandList.get(), pipeline->getRtcPipeline(), launchBufferVA, scratchBufferVA, sbt->raygenRecord,
                exceptionRecordVA, sbt->missRecordBase, sbt->missRecordStrideInBytes, sbt->missRecordCount,
                sbt->hitgroupRecordBase, sbt->hitgroupRecordStrideInBytes, sbt->hitgroupRecordCount,
                sbt->callablesRecordBase, sbt->callablesRecordStrideInBytes, sbt->callablesRecordCount, toolsOutputVA,
                pipeline->getToolsOutputSizeInBytes(), scratchBufferSizeInBytes, width, height, depth );
        }

        int exceptionCode = 0;
        if( pipeline->getDeviceContext()->hasValidationModeDebugExceptions() )
        {
            // synchronize such that we can access memory potentially written to by device
            // note that we ignore errors deliberately as that might be due to an explicit trap() call
            corelib::lwdaDriver().LwStreamSynchronize( stream );
            for( const Pipeline::NamedConstant* namedConstant : pipeline->getAdditionalNamedConstants() )
            {
                if( namedConstant->m_name == "__optixValidationModeExceptionCode" )
                    exceptionCode = *static_cast<int*>( namedConstant->m_hostPtr );
                // resetting to exceptionCode = 0
                *static_cast<int*>( namedConstant->m_hostPtr ) = 0;
            }

            if( exceptionCode < 0 )
            {
                std::ostringstream errorMsg;
                errorMsg << "Validation mode caught builtin exception "
                         << toString( static_cast<OptixExceptionCodes>( exceptionCode ) );
                result += errDetails.logDetails( OPTIX_ERROR_VALIDATION_FAILURE, errorMsg.str() );
                goto launch_cleanup;
            }
        }

        if( launchResult )
        {
            result += errDetails.logDetails( launchResult, "Error launching work to RTX" );
            goto launch_cleanup;
        }

        if( LWresult lwResult = corelib::lwdaDriver().LwEventRecord( pipeline->getConlwrrentLaunchEvent(), stream ) )
        {
            result += errDetails.logDetails( lwResult,
                                             "Error recording event to prevent conlwrrent launches on the same "
                                             "OptixPipeline" );
            goto launch_cleanup;
        }

        // Copy back the launch params and compare to computedSpecializations.
        const LaunchParamSpecialization& computedSpecializations = pipeline->getComputedSpecializations();
        if( !computedSpecializations.data.empty() )
        {
            if( computedSpecializations.data.size() != pipelineParamsSize )
            {
                result += errDetails.logDetails( OPTIX_ERROR_INTERNAL_ERROR,
                                                 "pipelineParamsSize " + std::to_string( computedSpecializations.data.size() )
                                                     + " and specialization sizes ( "
                                                     + std::to_string( pipelineParamsSize ) + " don't match" );
                // Do we stop processing or just move on?
                goto launch_cleanup;
            }
            std::vector<char> deviceParams( pipelineParamsSize );
            if( LWresult lwdaErr = corelib::lwdaDriver().LwMemcpyDtoH( deviceParams.data(), (LWdeviceptr)pipelineParams, pipelineParamsSize ) )
            {
                result += errDetails.logDetails( lwdaErr, "Failed to copy launch params back to host" );
                // Do we stop processing or just move on?
                goto launch_cleanup;
            }
            for( const EntryInfo& range : computedSpecializations.specializedRanges )
            {
                if( memcmp( deviceParams.data() + range.offset, computedSpecializations.data.data() + range.offset,
                            range.size ) )
                {
                    std::ostringstream out;
                    out << "Error: Specialization values mismatch between pipeline and pipelineParams.\n"
                        << "\tModule: " << range.moduleId << "\n"
                        << "\t\tannotation     : " << range.annotation << "\n"
                        << "\t\toffset         : " << range.offset << "\n"
                        << "\t\tsize           : " << range.size << "\n";
                    result += errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, out.str() );
                }
            }
        }

        if( pipeline->getDeviceContext()->isLwptiProfilingEnabled() )
        {
            optix_exp::ErrorDetails lwptiErrDetails;
            if( const OptixResult res = pipeline->getDeviceContext()->getLwptiProfiler().endProfile( errDetails ) )
            {
                result += res;
                goto launch_cleanup;
            }
        }

        if( !k_enableProfiling.get().empty() )
        {
            std::string filename = k_enableProfiling.get() + "_profile.txt";
            FILE*       dumpfile = fopen( filename.c_str(), "a+" );
            if( !dumpfile )
                goto launch_cleanup;

            std::vector<char> profileData( pipeline->getToolsOutputSizeInBytes() );

            if( LWresult lwdaErr = corelib::lwdaDriver().LwMemcpyDtoH( profileData.data(), (LWdeviceptr)toolsOutputVA,
                                                                       pipeline->getToolsOutputSizeInBytes() ) )
            {
                result += errDetails.logDetails( lwdaErr, "Failed to copy profile data back to host" );
                fclose( dumpfile );
                goto launch_cleanup;
            }
            corelib::dumpRTcoreProfileData( dumpfile, &pipeline->getProfileMetadata(), &pipeline->getShaderInfo(),
                                            profileData.data() );
            fclose( dumpfile );
        }
    }

launch_cleanup:
    // Release launch request, regardless of errors
    result += pipeline->getDeviceContext()->getLaunchResources().releaseResource( stream, launchStateSize, errDetails );
    result += commandList.destroy( errDetails );

    return result;
}
}  // end namespace optix_exp

extern "C" OptixResult optixLaunch( OptixPipeline            pipelineAPI,
                                    LWstream                 stream,
                                    LWdeviceptr              pipelineParams,
                                    size_t                   pipelineParamsSize,
                                    OptixShaderBindingTable* sbt,
                                    unsigned int             width,
                                    unsigned int             height,
                                    unsigned int             depth )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( Pipeline, pipeline, "OptixPipeline" );
    SCOPED_LWTX_RANGE( pipeline->getDeviceContext()->getLWTXProfiler(), LWTXRegisteredString::LAUNCH );
    optix_exp::DeviceContextLogger& clog = pipeline->getDeviceContext()->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( sbt );
    OPTIX_CHECK_ZERO_ARGUMENT( width );
    OPTIX_CHECK_ZERO_ARGUMENT( height );
    OPTIX_CHECK_ZERO_ARGUMENT( depth );

    try
    {
        optix_exp::DeviceContext*      deviceContext = pipeline->getDeviceContext();
        optix_exp::StreamMetricTimer scopeGuard( deviceContext, stream, "launch_time_ms" );

        optix_exp::ErrorDetails errDetails;
        if( const OptixResult result = launchPipeline( pipeline, stream, pipelineParams, pipelineParamsSize, sbt, width, height, depth, errDetails ) )
        {
            clog.sendError( errDetails );
            return result;
        }
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}
