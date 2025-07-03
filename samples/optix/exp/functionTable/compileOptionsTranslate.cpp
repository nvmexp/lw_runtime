/*
 * Copyright (c) 2020, LWPU CORPORATION.  All rights reserved.
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

#include <exp/context/OptixResultOneShot.h>
#include <exp/functionTable/compileOptionsTranslate.h>

#include <exp/context/ErrorHandling.h>

#include <prodlib/system/Knobs.h>

#define ABI_VERSION_CHANGE_V1_TO_V2 OptixABI::ABI_34
namespace {
// clang-format off
Knob<bool> k_forceDebugMode( RT_DSTRING( "o7.forceDebugMode" ), false, RT_DSTRING( "Force OptixModuleCompileOptions::debugLevel to OPTIX_COMPILE_DEBUG_LEVEL_FULL and OptixModuleCompileOptions::optLevel to OPTIX_COMPILE_OPTIMIZATION_LEVEL_0." ) );
// clang-format on
}  // namespace

namespace ABI_v1 {
/// Optimization levels
///
/// \see #OptixModuleCompileOptions::optLevel
typedef enum OptixCompileOptimizationLevel
{
    /// No optimizations
    OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 = 0,
    /// Some optimizations
    OPTIX_COMPILE_OPTIMIZATION_LEVEL_1 = 1,
    /// Most optimizations
    OPTIX_COMPILE_OPTIMIZATION_LEVEL_2 = 2,
    /// All optimizations
    OPTIX_COMPILE_OPTIMIZATION_LEVEL_3 = 3,
    OPTIX_COMPILE_OPTIMIZATION_DEFAULT = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3
} OptixCompileOptimizationLevel;

/// Debug levels
///
/// \see #OptixModuleCompileOptions::debugLevel
typedef enum OptixCompileDebugLevel
{
    /// No debug information
    OPTIX_COMPILE_DEBUG_LEVEL_NONE = 0,
    /// Generate lineinfo only
    OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL = 1,
    /// Generate dwarf debug information.
    OPTIX_COMPILE_DEBUG_LEVEL_FULL = 2,
} OptixCompileDebugLevel;
} // end namespace ABI_v1

namespace ABI_v47 {

typedef struct OptixPipelineCompileOptions
{
    /// Boolean value indicating whether motion blur could be used
    int usesMotionBlur;

    /// Traversable graph bitfield. See OptixTraversableGraphFlags
    unsigned int traversableGraphFlags;

    /// How much storage, in 32b words, to make available for the payload, [0..32]
    /// Must be zero if numPayloadTypes is not zero.
    int numPayloadValues;

    /// How much storage, in 32b words, to make available for the attributes. The
    /// minimum number is 2. Values below that will automatically be changed to 2. [2..8]
    int numAttributeValues;

    /// A bitmask of OptixExceptionFlags indicating which exceptions are enabled.
    unsigned int exceptionFlags;

    /// The name of the pipeline parameter variable.  If 0, no pipeline parameter
    /// will be available. This will be ignored if the launch param variable was
    /// optimized out or was not found in the modules linked to the pipeline.
    const char* pipelineLaunchParamsVariableName;

    /// Bit field enabling primitive types. See OptixPrimitiveTypeFlags.
    /// Setting to zero corresponds to enabling OPTIX_PRIMITIVE_TYPE_FLAGS_LWSTOM and OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE.
    unsigned int usesPrimitiveTypeFlags;

    /// The number of different payload types available for compilation.
    /// Must be zero if numPayloadValues is not zero.
    unsigned int numPayloadTypes;

    /// Points to host array of payload type definitions, size must match numPayloadValues
    OptixPayloadType *payloadTypes;

} OptixPipelineCompileOptions;

typedef struct OptixProgramGroupOptions
{
    /// Specifies the payload type of this program group corresponding to the selected ID.
    /// All programs in the group must support the payload type
    /// (Program support for a type is specifed by calling
    /// \see #optixSetPayloadTypes or otherwise all types specified in
    /// \see #OptixPipelineCompileOptions are supported).
    /// If a program is not available for the requested payload type,
    /// optixProgramGroupCreate returns OPTIX_ERROR_PAYLOAD_TYPE_MISMATCH.
    /// If OptixPayloadTypeID is OPTIX_PAYLOAD_TYPE_DEFAULT, a unique type is deduced.
    /// The payload type can be uniquely deduced if there is exactly one payload type
    /// for which all programs in the group are available.
    /// If the payload type could not be deduced uniquely
    /// optixProgramGroupCreate returns OPTIX_ERROR_PAYLOAD_TYPE_RESOLUTION_FAILED.
    OptixPayloadTypeID payloadType;
} OptixProgramGroupOptions;

} // end namespace ABI_v47

namespace optix_exp {

OptixResult validateCompileDebugLevel( OptixCompileDebugLevel debugLevel, OptixABI abi, ErrorDetails& errDetails )
{
    if( abi >= ABI_VERSION_CHANGE_V1_TO_V2 )
    {
        switch( debugLevel )
        {
        case OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT:
        case OPTIX_COMPILE_DEBUG_LEVEL_NONE:
        case OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL:
        case OPTIX_COMPILE_DEBUG_LEVEL_FULL:
            return OPTIX_SUCCESS;
        case OPTIX_COMPILE_DEBUG_LEVEL_MODERATE:
            if( abi >= OptixABI::ABI_55 )
                return OPTIX_SUCCESS;
        }
    }
    else
    {
        switch( static_cast<ABI_v1::OptixCompileDebugLevel>( debugLevel ) )
        {
        case ABI_v1::OPTIX_COMPILE_DEBUG_LEVEL_NONE:
        case ABI_v1::OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL:
        case ABI_v1::OPTIX_COMPILE_DEBUG_LEVEL_FULL:
            return OPTIX_SUCCESS;
        }
    }
    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "Unknown OptixCompileDebugLevel: " + std::to_string( debugLevel ) );
}

OptixResult translateCompileDebugLevel( OptixCompileDebugLevel& newDebugLevel, OptixCompileDebugLevel debugLevel, OptixABI abi, ErrorDetails& errDetails )
{
    if( abi >= ABI_VERSION_CHANGE_V1_TO_V2 )
    {
        newDebugLevel = debugLevel;
        if( abi < OptixABI::ABI_55 && newDebugLevel == OPTIX_COMPILE_DEBUG_LEVEL_MODERATE )
            newDebugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
    }
    else
    {
        switch ( static_cast<ABI_v1::OptixCompileDebugLevel>( debugLevel ) )
        {
        case ABI_v1::OPTIX_COMPILE_DEBUG_LEVEL_NONE:
            newDebugLevel =  OPTIX_COMPILE_DEBUG_LEVEL_NONE;
            break;
        case ABI_v1::OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL:
            newDebugLevel =  OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
            break;
        case ABI_v1::OPTIX_COMPILE_DEBUG_LEVEL_FULL:
            newDebugLevel =  OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            break;
        }
    }
    return OPTIX_SUCCESS;
}

OptixResult validateCompileOptimizationLevel( OptixCompileOptimizationLevel optLevel, OptixABI abi, ErrorDetails& errDetails )
{
    if( abi >= ABI_VERSION_CHANGE_V1_TO_V2 )
    {
        switch ( optLevel )
        {
        case OPTIX_COMPILE_OPTIMIZATION_DEFAULT:
        case OPTIX_COMPILE_OPTIMIZATION_LEVEL_0:
        case OPTIX_COMPILE_OPTIMIZATION_LEVEL_1:
        case OPTIX_COMPILE_OPTIMIZATION_LEVEL_2:
        case OPTIX_COMPILE_OPTIMIZATION_LEVEL_3:
            return OPTIX_SUCCESS;
        }
    }
    else
    {
        switch ( static_cast<ABI_v1::OptixCompileOptimizationLevel>( optLevel ) )
        {
        case ABI_v1::OPTIX_COMPILE_OPTIMIZATION_LEVEL_0:
        case ABI_v1::OPTIX_COMPILE_OPTIMIZATION_LEVEL_1:
        case ABI_v1::OPTIX_COMPILE_OPTIMIZATION_LEVEL_2:
        case ABI_v1::OPTIX_COMPILE_OPTIMIZATION_LEVEL_3:
            return OPTIX_SUCCESS;
        }
    }
    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "Unknown OptixCompileOptimizationLevel: " + std::to_string( optLevel ) );
}

OptixResult translateCompileOptimizationLevel( OptixCompileOptimizationLevel& newOptLevel, OptixCompileOptimizationLevel optLevel, OptixABI abi, ErrorDetails& errDetails )
{
    if( abi >= ABI_VERSION_CHANGE_V1_TO_V2 )
    {
        newOptLevel = optLevel;
    }
    else
    {
        switch ( static_cast<ABI_v1::OptixCompileOptimizationLevel>( optLevel ) )
        {
        case ABI_v1::OPTIX_COMPILE_OPTIMIZATION_LEVEL_0:
            newOptLevel =  OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            break;
        case ABI_v1::OPTIX_COMPILE_OPTIMIZATION_LEVEL_1:
            newOptLevel =  OPTIX_COMPILE_OPTIMIZATION_LEVEL_1;
            break;
        case ABI_v1::OPTIX_COMPILE_OPTIMIZATION_LEVEL_2:
            newOptLevel =  OPTIX_COMPILE_OPTIMIZATION_LEVEL_2;
            break;
        case ABI_v1::OPTIX_COMPILE_OPTIMIZATION_LEVEL_3:
            newOptLevel =  OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
            break;
        }
    }
    return OPTIX_SUCCESS;
}

OptixResult validateOptDebugLevelBeforeTranslation( OptixCompileDebugLevel debugLevel, OptixCompileOptimizationLevel optLevel, OptixABI abi, ErrorDetails& errDetails )
{
    // ptxas does not support this. Review once we switched to D2IR.
    if( debugLevel == OPTIX_COMPILE_DEBUG_LEVEL_FULL && optLevel != OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                      "Debug level OPTIX_COMPILE_DEBUG_LEVEL_FULL requires optimization level "
                                      "OPTIX_COMPILE_OPTIMIZATION_LEVEL_0." );
    return OPTIX_SUCCESS;
}
OptixResult translateDefaultModuleCompileOptions( OptixModuleCompileOptions* moduleCompileOptions, ErrorDetails& errDetails )
{
    if( moduleCompileOptions->debugLevel == OPTIX_COMPILE_DEBUG_LEVEL_DEFAULT )
        moduleCompileOptions->debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    if( moduleCompileOptions->optLevel == OPTIX_COMPILE_OPTIMIZATION_DEFAULT )
    {
        if( moduleCompileOptions->debugLevel == OPTIX_COMPILE_DEBUG_LEVEL_FULL )
            moduleCompileOptions->optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        else
            moduleCompileOptions->optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    }
    return OPTIX_SUCCESS;
}

OptixResult translateABI_OptixModuleCompileOptions( const OptixModuleCompileOptions*   inOptions,
                                                    const OptixPipelineCompileOptions* inPipelineOptions,
                                                    OptixABI                           inAbi,
                                                    OptixModuleCompileOptions*         outOptions,
                                                    ErrorDetails&                      errDetails )
{
    OptixResultOneShot result;
    // Validate input options
    result += validateCompileOptimizationLevel( inOptions->optLevel, inAbi, errDetails );
    result += validateCompileDebugLevel( inOptions->debugLevel, inAbi, errDetails );
    result += validateOptDebugLevelBeforeTranslation( inOptions->debugLevel, inOptions->optLevel, inAbi, errDetails );

    *outOptions                = {};
    const int maxRegisterCount = inOptions->maxRegisterCount;
    if( maxRegisterCount < 0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "maxRegisterCount must be >= 0. Value is: " + std::to_string( maxRegisterCount ) );

    outOptions->maxRegisterCount = maxRegisterCount;
    outOptions->optLevel         = inOptions->optLevel;
    outOptions->debugLevel       = inOptions->debugLevel;
    if( inAbi >= OptixABI::ABI_37 )
    {
        outOptions->boundValues      = inOptions->boundValues;
        outOptions->numBoundValues   = inOptions->numBoundValues;
    }

    if( inAbi >= OptixABI::ABI_48 )
    {
        outOptions->numPayloadTypes = inOptions->numPayloadTypes;
        outOptions->payloadTypes    = inOptions->payloadTypes;
    }
    else if( inAbi == OptixABI::ABI_47 )
    {
        // OptiX7.3 (ABI_47) contained two 'reserved' members in OptixPipelineCompileOptions.
        // Those are expected to be zero. Internal tests with ABI_47 could already use the payloads feature.
        // However, there is no need of supporting internal tests with ABI_47 and payload types feature enabled.
        // Simply assume that the feature is disabled for all ABI_47 applications.
        outOptions->numPayloadTypes = 0u;
        outOptions->payloadTypes    = nullptr;
    }

    // Patch up the OptixModuleCompileOptions based on the ABI
    result += translateCompileOptimizationLevel( outOptions->optLevel, outOptions->optLevel, inAbi, errDetails );

    result += translateCompileDebugLevel( outOptions->debugLevel, outOptions->debugLevel, inAbi, errDetails );

    // Translate the default values into their useful value
    result += translateDefaultModuleCompileOptions( outOptions, errDetails );

    return result;
}

OptixResult translateABI_PipelineCompileOptions( const OptixPipelineCompileOptions* inOptions,
                                                 OptixABI                           inAbi,
                                                 OptixPipelineCompileOptions*       outOptions,
                                                 ErrorDetails&                      errDetails )
{
    *outOptions                                  = {};
    outOptions->usesMotionBlur                   = inOptions->usesMotionBlur;
    outOptions->traversableGraphFlags            = inOptions->traversableGraphFlags;
    outOptions->numPayloadValues                 = inOptions->numPayloadValues;
    outOptions->numAttributeValues               = inOptions->numAttributeValues;
    outOptions->exceptionFlags                   = inOptions->exceptionFlags;
    outOptions->pipelineLaunchParamsVariableName = inOptions->pipelineLaunchParamsVariableName;
    if( inAbi >= OptixABI::ABI_27 )
        outOptions->usesPrimitiveTypeFlags = inOptions->usesPrimitiveTypeFlags;

#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    // FIXME: need to specify proper ABI version when MM becomes part of the SDK
    if( inAbi >= OptixABI::ABI_LWRRENT )
    {
        outOptions->allowVisibilityMaps       = inOptions->allowVisibilityMaps;
        outOptions->allowDisplacedMicromeshes = inOptions->allowDisplacedMicromeshes;
    }
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

    return OPTIX_SUCCESS;
}

OptixResult translateABI_PipelineLinkOptions( const OptixPipelineLinkOptions* inOptions,
                                              OptixABI                        inAbi,
                                              OptixPipelineLinkOptions*       outOptions,
                                              ErrorDetails&                   errDetails )
{
    *outOptions               = {};
    outOptions->maxTraceDepth = inOptions->maxTraceDepth;
    if( k_forceDebugMode.get() )
        outOptions->debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    else
        outOptions->debugLevel = inOptions->debugLevel;
    // ignore old field overrideUsesMotionBlur

#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
    // FIXME: need to specify proper ABI version when MM becomes part of the SDK
    if( inAbi >= OptixABI::ABI_LWRRENT )
    {
        outOptions->enableVisibilityMaps = inOptions->enableVisibilityMaps;
    }
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)

    return OPTIX_SUCCESS;
}

OptixResult translateABI_ProgramGroupPayloadTypeID( const OptixProgramGroupOptions* inOptions,
                                                    OptixABI                        inAbi,
                                                    OptixPayloadTypeID*             outPayloadTypeID,
                                                    ErrorDetails&                   errDetails )
{
    *outPayloadTypeID = OPTIX_PAYLOAD_TYPE_DEFAULT;
    if( inAbi == OptixABI::ABI_47 )
        *outPayloadTypeID = reinterpret_cast<const ABI_v47::OptixProgramGroupOptions*>( inOptions )->payloadType;
    return OPTIX_SUCCESS;
}

OptixResult validatePayloadType( const OptixPayloadType& payloadType, ErrorDetails& errDetails )
{
    if( payloadType.numPayloadValues != 0 && payloadType.payloadSemantics == 0 )
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE, "OptixPayloadType::payloadSemantics must not be zero when OptixPayloadType::numPayloadValues is non-zero" );

    if( payloadType.numPayloadValues > OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT )
    {
        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
            "OptixPayloadType::numPayloadValues must be less than or equal to "
            + std::to_string( OPTIX_COMPILE_DEFAULT_MAX_PAYLOAD_VALUE_COUNT )
            + ": " + std::to_string( payloadType.numPayloadValues ) );
    }
    return OPTIX_SUCCESS;
}

} // end namespace optix_exp
