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
#include <exp/pipeline/ProgramGroup.h>
#include <exp/functionTable/compileOptionsTranslate.h>

#include <corelib/misc/String.h>
#include <prodlib/exceptions/Exception.h>
#include <prodlib/misc/LWTXProfiler.h>
#include <prodlib/exceptions/Assert.h>

#include <llvm/ADT/StringRef.h>

#include <cstring>
#include <memory>

namespace optix_exp {

ProgramGroup::ProgramGroup( DeviceContext* context, const OptixProgramGroupDesc& impl, const CompilePayloadType* payloadType )
    : OpaqueApiObject( OpaqueApiObject::ApiType::ProgramGroup )
    , m_context( context )
    , m_impl( impl )
{
    switch( m_impl.kind )
    {
        case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
        {
            m_impl.raygen.entryFunctionName = mangleAndDuplicate( impl.raygen.entryFunctionName, impl.raygen.module, payloadType );
            break;
        }

        case OPTIX_PROGRAM_GROUP_KIND_MISS:
        {
            m_impl.miss.entryFunctionName = mangleAndDuplicate( impl.miss.entryFunctionName, impl.miss.module, payloadType );
            break;
        }

        case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
        {
            m_impl.exception.entryFunctionName = mangleAndDuplicate( impl.exception.entryFunctionName, impl.exception.module, payloadType );
            break;
        }

        case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
        {
            m_impl.hitgroup.entryFunctionNameCH = mangleAndDuplicate( impl.hitgroup.entryFunctionNameCH, impl.hitgroup.moduleCH, payloadType );
            m_impl.hitgroup.entryFunctionNameAH = mangleAndDuplicate( impl.hitgroup.entryFunctionNameAH, impl.hitgroup.moduleAH, payloadType );

            Module* moduleIS;
            implCast( impl.hitgroup.moduleIS, moduleIS );

            // if the IS module is a builtin module, set the entry name to the intersection function name used by all builtin modules.
            const char* entryFunctionNameIS = impl.hitgroup.entryFunctionNameIS;
            if( impl.hitgroup.moduleIS && moduleIS->isBuiltinModule() )
                entryFunctionNameIS = "__intersection__is";

            m_impl.hitgroup.entryFunctionNameIS = mangleAndDuplicate( entryFunctionNameIS, impl.hitgroup.moduleIS, payloadType );
            break;
        }

        case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
        {
            m_impl.callables.entryFunctionNameDC =
                mangleAndDuplicate( impl.callables.entryFunctionNameDC, impl.callables.moduleDC, payloadType );
            m_impl.callables.entryFunctionNameCC =
                mangleAndDuplicate( impl.callables.entryFunctionNameCC, impl.callables.moduleCC, payloadType );
            break;
        }
    }
}

OptixResult ProgramGroup::destroy( bool doUnregisterProgramGroup, ErrorDetails& errDetails )
{
    return doUnregisterProgramGroup ? m_context->unregisterProgramGroup( this, errDetails ) : OPTIX_SUCCESS;
}

const char* ProgramGroup::mangleAndDuplicate( const char* s, OptixModule moduleAPI, const CompilePayloadType* payloadType )
{
    if( !s )
        return nullptr;

    Module* module;
    implCast( moduleAPI, module );
    OptixPayloadTypeID payloadTypeId = OPTIX_PAYLOAD_TYPE_DEFAULT;
    if( payloadType )
        payloadTypeId = (OptixPayloadTypeID)module->getCompatiblePayloadTypeId( *payloadType );
    std::string mangledName = module->getMangledName(  llvm::StringRef( s ), payloadTypeId, ST_ILWALID );
    m_strings.push_back( mangledName );
    return m_strings.back().c_str();
}

static bool isPowerOfTwo( unsigned int i )
{
    return (i != 0) && ( ( i & ( i - 1 ) ) == 0 );
}

static OptixResult createProgramGroups( DeviceContext*                  context,
                                        const OptixProgramGroupDesc*    programDescriptions,
                                        unsigned int                    numProgramGroups,
                                        const OptixProgramGroupOptions* options,
                                        OptixProgramGroup*              programGroups,
                                        ErrorDetails&                   errDetails )
{
    CompilePayloadType userCompilePayloadType;
    const CompilePayloadType *compilePayloadType = nullptr;

    // Per module payload types where introduced at ABI 48
    if( context->getAbiVersion() >= OptixABI::ABI_48 )
    {
        if( options->payloadType )
        {
            const OptixResult result = validatePayloadType( *options->payloadType, errDetails );
            if( result )
                return result;

            userCompilePayloadType = CompilePayloadType( *options->payloadType );
            compilePayloadType = &userCompilePayloadType;
        }
    }
    else if( context->getAbiVersion() == OptixABI::ABI_47 )
    {
        // Fallback for ABI 47. This was never exposed but is used for testing the payload feature for DXR.
        // TODO: remove when we stop testing r465-SDK.

        OptixPayloadTypeID payloadTypeID = OPTIX_PAYLOAD_TYPE_DEFAULT;
        const OptixResult result = translateABI_ProgramGroupPayloadTypeID( options, context->getAbiVersion(), &payloadTypeID, errDetails );
        if( result )
            return result;

        if( payloadTypeID != OPTIX_PAYLOAD_TYPE_DEFAULT )
        {
            // Map the ID to a payload type using any of the modules. They should all agree on the Id to payload mapping.
            for( unsigned int i = 0; i < numProgramGroups && compilePayloadType == nullptr; ++i )
            {
                const std::string iStr( std::to_string( i ) );

                if( programDescriptions[i].kind == OPTIX_PROGRAM_GROUP_KIND_MISS )
                {
                    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2_NULL_OK( Module, module, programDescriptions[i].miss.module,
                                                                    "programDescriptions[" + iStr + "].miss.module",
                                                                    "optixProgramGroupCreate" );
                    if( module )
                        compilePayloadType = module->getPayloadTypeFromId( payloadTypeID );
                }
                else if( programDescriptions[i].kind == OPTIX_PROGRAM_GROUP_KIND_HITGROUP )
                {
                    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2_NULL_OK( Module, moduleCH, programDescriptions[i].hitgroup.moduleCH,
                                                                    "programDescriptions[" + iStr + "].hitgroup.moduleCH",
                                                                    "optixProgramGroupCreate" );
                    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2_NULL_OK( Module, moduleAH, programDescriptions[i].hitgroup.moduleAH,
                                                                    "programDescriptions[" + iStr + "].hitgroup.moduleAH",
                                                                    "optixProgramGroupCreate" );
                    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2_NULL_OK( Module, moduleIS, programDescriptions[i].hitgroup.moduleIS,
                                                                    "programDescriptions[" + iStr + "].hitgroup.moduleIS",
                                                                    "optixProgramGroupCreate" );

                    if( moduleCH )
                        compilePayloadType = moduleCH->getPayloadTypeFromId( payloadTypeID );
                    else if ( moduleAH )
                        compilePayloadType = moduleAH->getPayloadTypeFromId( payloadTypeID );
                    else if ( moduleIS )
                        compilePayloadType = moduleIS->getPayloadTypeFromId( payloadTypeID );
                }
            }
        }
    }

    std::vector<const CompilePayloadType*> payloadTypes;
    for( unsigned int i = 0; i < numProgramGroups; ++i )
    {
        const std::string iStr( std::to_string( i ) );

        const CompilePayloadType* payloadType = nullptr;
        switch( programDescriptions[i].kind )
        {
            case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
            {
                OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2( Module, module, programDescriptions[i].raygen.module,
                                                          "programDescriptions[" + iStr + "].raygen.module",
                                                          "optixProgramGroupCreate" );

                OPTIX_CHECK_SAME_CONTEXT( context, module->getDeviceContext(),
                                          "programDescriptions[" + iStr + "].raygen.module" );

                const char* name = programDescriptions[i].raygen.entryFunctionName;

                if( !name )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "programDescriptions[" + iStr
                                                      + "].raygen.entryFunctionName is null" );

                const std::string prefix = "__raygen__";

                if( !corelib::stringBeginsWith( name, prefix ) )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "programDescriptions[" + iStr
                                                      + "].raygen.entryFunctionName does not start with \"" + prefix
                                                      + "\"" );
                if( !module->hasEntryFunction( name ) )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "\"" + std::string( name ) + "\" not found in programDescriptions["
                                                      + iStr + "].raygen.module" );

                break;
            }

            case OPTIX_PROGRAM_GROUP_KIND_MISS:
            {
                OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2_NULL_OK( Module, module, programDescriptions[i].miss.module,
                                                                  "programDescriptions[" + iStr + "].miss.module",
                                                                  "optixProgramGroupCreate" );

                if( module )
                    OPTIX_CHECK_SAME_CONTEXT( context, module->getDeviceContext(),
                                              "programDescriptions[" + iStr + "].miss.module" );

                const char* name = programDescriptions[i].miss.entryFunctionName;

                if( !!module ^ !!name )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "exactly one of programDescriptions[" + iStr
                                                      + "].miss.module and programDescriptions[" + iStr
                                                      + "].miss.entryFunctionName is null " );

                const std::string prefix = "__miss__";

                if( name && !corelib::stringBeginsWith( name, prefix ) )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "programDescriptions[" + iStr
                                                      + "].miss.entryFunctionName does not start with \"" + prefix
                                                      + "\"" );
                if( name && !module->hasEntryFunction( name ) )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "\"" + std::string( name ) + "\" not found in programDescriptions["
                                                       + iStr + "].miss.module" );
                if( name )
                {
                    const EntryFunctionSemantics& moduleEntry = module->getEntryFunctionSemantics( name );

                    if( compilePayloadType )
                    {
                        OptixPayloadTypeID payloadTypeId = (OptixPayloadTypeID)module->getCompatiblePayloadTypeId( *compilePayloadType, moduleEntry.m_payloadTypeMask );

                        // Either there is a miss function with this name for the specified payload type
                        if( ( moduleEntry.m_payloadTypeMask & payloadTypeId ) == 0 )
                        {
                            return errDetails.logDetails( OPTIX_ERROR_PAYLOAD_TYPE_MISMATCH,
                                                  "\"" + std::string( name ) + "\" in programDescriptions["
                                                       + iStr + "].miss.module does not support the specified payloadType" );
                        }

                        payloadType = compilePayloadType;
                    }
                    else
                    {
                       // Or no payload type is specified and the miss function only supports one unique payload type
                       if( !isPowerOfTwo( moduleEntry.m_payloadTypeMask ) )
                       {
                           return errDetails.logDetails( OPTIX_ERROR_PAYLOAD_TYPE_RESOLUTION_FAILED,
                                                  "\"" + std::string( name ) + "\" in programDescriptions["
                                                       + iStr + "].miss.module could not be resolved to a unique payloadType" );
                       }

                       payloadType = module->getPayloadTypeFromId( (OptixPayloadTypeID)moduleEntry.m_payloadTypeMask );
                    }
                }

                break;
            }

            case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
            {
                OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2( Module, module, programDescriptions[i].exception.module,
                                                          "programDescriptions[" + iStr + "].exception.module",
                                                          "optixProgramGroupCreate" );

                OPTIX_CHECK_SAME_CONTEXT( context, module->getDeviceContext(),
                                          "programDescriptions[" + iStr + "].exception.module" );

                const char* name = programDescriptions[i].exception.entryFunctionName;

                if( !name )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "programDescriptions[" + iStr
                                                      + "].exception.entryFunctionName is null" );

                const std::string prefix = "__exception__";

                if( !corelib::stringBeginsWith( name, "__exception__" ) )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "programDescriptions[" + iStr
                                                      + "].exception.entryFunctionName does not start with \"" + prefix
                                                      + "\"" );
                if( !module->hasEntryFunction( name ) )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "\"" + std::string( name ) + "\" not found in programDescriptions["
                                                      + iStr + "].exception.module" );

                break;
            }

            case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
            {
                OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2_NULL_OK( Module, moduleCH, programDescriptions[i].hitgroup.moduleCH,
                                                                  "programDescriptions[" + iStr + "].hitgroup.moduleCH",
                                                                  "optixProgramGroupCreate" );
                OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2_NULL_OK( Module, moduleAH, programDescriptions[i].hitgroup.moduleAH,
                                                                  "programDescriptions[" + iStr + "].hitgroup.moduleAH",
                                                                  "optixProgramGroupCreate" );
                OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2_NULL_OK( Module, moduleIS, programDescriptions[i].hitgroup.moduleIS,
                                                                  "programDescriptions[" + iStr + "].hitgroup.moduleIS",
                                                                  "optixProgramGroupCreate" );

                if( moduleCH )
                    OPTIX_CHECK_SAME_CONTEXT( context, moduleCH->getDeviceContext(),
                                              "programDescriptions[" + iStr + "].hitgroup.moduleCH" );
                if( moduleAH )
                    OPTIX_CHECK_SAME_CONTEXT( context, moduleAH->getDeviceContext(),
                                              "programDescriptions[" + iStr + "].hitgroup.moduleAH" );
                if( moduleIS )
                    OPTIX_CHECK_SAME_CONTEXT( context, moduleIS->getDeviceContext(),
                                              "programDescriptions[" + iStr + "].hitgroup.moduleIS" );

                const char* nameCH = programDescriptions[i].hitgroup.entryFunctionNameCH;
                const char* nameAH = programDescriptions[i].hitgroup.entryFunctionNameAH;
                const char* nameIS = programDescriptions[i].hitgroup.entryFunctionNameIS;

                const bool builtinIS = !!moduleIS && moduleIS->isBuiltinModule();

                if( !!moduleCH ^ !!nameCH )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "exactly one of programDescriptions[" + iStr
                                                      + "].hitgroup.moduleCH and programDescriptions[" + iStr
                                                      + "].hitgroup.entryFunctionNameCH is null " );
                if( !!moduleAH ^ !!nameAH )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "exactly one of programDescriptions[" + iStr
                                                      + "].hitgroup.moduleAH and programDescriptions[" + iStr
                                                      + "].hitgroup.entryFunctionNameAH is null " );
                if( !builtinIS )
                {
                    if( !!moduleIS ^ !!nameIS )
                        return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                      "exactly one of programDescriptions[" + iStr
                                                          + "].hitgroup.moduleIS and programDescriptions[" + iStr
                                                          + "].hitgroup.entryFunctionNameIS is null " );
                }
                else
                {
                    if( builtinIS && !!nameIS )
                        return errDetails.logDetails(
                            OPTIX_ERROR_ILWALID_VALUE,
                            "programDescriptions[" + iStr
                                + "].hitgroup.moduleIS is a builtin module but programDescriptions[" + iStr
                                + "].hitgroup.entryFunctionNameIS is not null " );
                    nameIS = "__intersection__is"; // TODO, figure out what the name should be when we have more than one
                }

                const std::string prefixCH = "__closesthit__";
                const std::string prefixAH = "__anyhit__";
                const std::string prefixIS = "__intersection__";

                if( nameCH && !corelib::stringBeginsWith( nameCH, prefixCH ) )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "programDescriptions[" + iStr
                                                      + "].hitgroup.entryFunctionNameCH does not start with \""
                                                      + prefixCH + "\"" );

                if( nameAH && !corelib::stringBeginsWith( nameAH, prefixAH ) )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "programDescriptions[" + iStr
                                                      + "].hitgroup.entryFunctionNameAH does not start with \""
                                                      + prefixAH + "\"" );

                if( nameIS && !corelib::stringBeginsWith( nameIS, prefixIS ) )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "programDescriptions[" + iStr
                                                      + "].hitgroup.entryFunctionNameIS does not start with \""
                                                      + prefixIS + "\"" );

                if( nameCH && !moduleCH->hasEntryFunction( nameCH ) )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "\"" + std::string( nameCH ) + "\" not found in programDescriptions["
                                                      + iStr + "].hitgroup.moduleCH" );

                if( nameAH && !moduleAH->hasEntryFunction( nameAH ) )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "\"" + std::string( nameAH ) + "\" not found in programDescriptions["
                                                      + iStr + "].hitgroup.moduleAH" );

                if( nameIS && !moduleIS->hasEntryFunction( nameIS ) )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "\"" + std::string( nameIS ) + "\" not found in programDescriptions["
                                                      + iStr + "].hitgroup.moduleIS" );

                if( compilePayloadType != nullptr )
                {
                    // Either there is a CH, AH and IS function with for the specified payload type

                    if( nameCH )
                    {
                        const EntryFunctionSemantics& moduleEntry = moduleCH->getEntryFunctionSemantics( nameCH );
                        OptixPayloadTypeID payloadType = ( OptixPayloadTypeID )moduleCH->getCompatiblePayloadTypeId( *compilePayloadType, moduleEntry.m_payloadTypeMask );
                        if( ( moduleEntry.m_payloadTypeMask & payloadType ) == 0 )
                        {
                            return errDetails.logDetails(
                                OPTIX_ERROR_PAYLOAD_TYPE_MISMATCH,
                                "\"" + std::string( nameCH ) + "\" in programDescriptions[" + iStr
                                    + "].hitgroup.moduleCH does not support the specified payloadType" );
                        }
                    }

                    if( nameAH )
                    {
                        const EntryFunctionSemantics& moduleEntry = moduleAH->getEntryFunctionSemantics( nameAH );
                        OptixPayloadTypeID payloadType = ( OptixPayloadTypeID )moduleAH->getCompatiblePayloadTypeId( *compilePayloadType, moduleEntry.m_payloadTypeMask );
                        if( ( moduleEntry.m_payloadTypeMask & payloadType ) == 0 )
                        {
                            return errDetails.logDetails(
                                OPTIX_ERROR_PAYLOAD_TYPE_MISMATCH,
                                "\"" + std::string( nameAH ) + "\" in programDescriptions[" + iStr
                                    + "].hitgroup.moduleAH does not support the specified payloadType" );
                        }
                    }

                    if( nameIS )
                    {
                        const EntryFunctionSemantics& moduleEntry = moduleIS->getEntryFunctionSemantics( nameIS );
                        OptixPayloadTypeID payloadType = (OptixPayloadTypeID)moduleIS->getCompatiblePayloadTypeId( *compilePayloadType, moduleEntry.m_payloadTypeMask );
                        if( ( moduleEntry.m_payloadTypeMask & payloadType ) == 0 )
                        {
                            return errDetails.logDetails(
                                OPTIX_ERROR_PAYLOAD_TYPE_MISMATCH,
                                "\"" + std::string( nameIS ) + "\" in programDescriptions[" + iStr
                                    + "].hitgroup.moduleIS does not support the specified payloadType" );
                        }
                    }

                    payloadType = compilePayloadType;
                }
                else if( nameCH || nameAH || nameIS )
                {
                    // Or there is only one payload type for which all functions have support

                    const unsigned int payloadTypeMaskCH = nameCH ? moduleCH->getEntryFunctionSemantics( nameCH ).m_payloadTypeMask : 0;
                    const unsigned int payloadTypeMaskAH = nameAH ? moduleAH->getEntryFunctionSemantics( nameAH ).m_payloadTypeMask : 0;
                    const unsigned int payloadTypeMaskIS = nameIS ? moduleIS->getEntryFunctionSemantics( nameIS ).m_payloadTypeMask : 0;

                    // Pick any valid entry function
                    const Module* module = (nameCH ? moduleCH : (nameAH ? moduleAH : moduleIS ) );
                    const char*   name   = (nameCH ? nameCH   : (nameAH ? nameAH   : nameIS   ) );

                    // Iterate over all payload types supported by the entry function
                    const unsigned int payloadTypeMask = module->getEntryFunctionSemantics( name ).m_payloadTypeMask;
                    bool foundMatch = false;
                    for( int payloadTypeId = 1; payloadTypeId <= payloadTypeMask; payloadTypeId <<= 1 )
                    {
                        if( payloadTypeMask & payloadTypeId )
                        {
                            // Check if all specified entry functions support this type
                            const CompilePayloadType* supportedPayloadType = module->getPayloadTypeFromId( payloadTypeId );
                            if( supportedPayloadType == nullptr )
                                return OPTIX_ERROR_INTERNAL_COMPILER_ERROR;

                            if( nameCH )
                            {
                                OptixPayloadTypeID payloadTypeIdCH = ( OptixPayloadTypeID )moduleCH->getCompatiblePayloadTypeId( *supportedPayloadType, payloadTypeMaskCH );
                                if( payloadTypeIdCH == OPTIX_PAYLOAD_TYPE_DEFAULT )
                                    continue;
                            }

                            if( nameAH )
                            {
                                OptixPayloadTypeID payloadTypeIdAH = ( OptixPayloadTypeID )moduleAH->getCompatiblePayloadTypeId( *supportedPayloadType, payloadTypeMaskAH );
                                if( payloadTypeIdAH == OPTIX_PAYLOAD_TYPE_DEFAULT )
                                    continue;
                            }

                            if( nameIS )
                            {
                                OptixPayloadTypeID payloadTypeIdIS = ( OptixPayloadTypeID )moduleIS->getCompatiblePayloadTypeId( *supportedPayloadType, payloadTypeMaskIS );
                                if( payloadTypeIdIS == OPTIX_PAYLOAD_TYPE_DEFAULT )
                                    continue;
                            }

                            if( foundMatch )
                            {
                                return errDetails.logDetails( OPTIX_ERROR_PAYLOAD_TYPE_RESOLUTION_FAILED,
                                                  "\"" + std::string( nameCH ? nameCH : "null" ) + "\" in programDescriptions["
                                                       + iStr + "].miss.moduleCH, "
                                                    "\"" + std::string( nameAH ? nameAH : "null" ) + "\" in programDescriptions["
                                                       + iStr + "].miss.moduleAH and "
                                                    "\"" + std::string( nameIS ? nameIS : "null" ) + "\" in programDescriptions["
                                                       + iStr + "].miss.moduleIS could not be resolved to a unique payloadType" );
                            }

                            foundMatch = true;
                            payloadType = supportedPayloadType;
                        }
                    }

                    if( !foundMatch )
                    {
                        return errDetails.logDetails( OPTIX_ERROR_PAYLOAD_TYPE_RESOLUTION_FAILED,
                                                  "\"" + std::string( nameCH ? nameCH : "null" ) + "\" in programDescriptions["
                                                       + iStr + "].miss.moduleCH, "
                                                    "\"" + std::string( nameAH ? nameAH : "null" ) + "\" in programDescriptions["
                                                       + iStr + "].miss.moduleAH and "
                                                    "\"" + std::string( nameIS ? nameIS : "null" ) + "\" in programDescriptions["
                                                       + iStr + "].miss.moduleIS could not be resolved to a common payloadType" );
                    }
                }
                else
                {
                    // the hitgroup only has null shaders
                }

                break;
            }

            case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
            {
                OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2_NULL_OK(
                    Module, moduleDC, programDescriptions[i].callables.moduleDC,
                    "programDescriptions[" + iStr + "].callables.moduleDC", "optixProgramGroupCreate" );
                OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT2_NULL_OK(
                    Module, moduleCC, programDescriptions[i].callables.moduleCC,
                    "programDescriptions[" + iStr + "].callables.moduleCC", "optixProgramGroupCreate" );

                if( moduleDC )
                    OPTIX_CHECK_SAME_CONTEXT( context, moduleDC->getDeviceContext(),
                                              "programDescriptions[" + iStr + "].callables.moduleDC" );
                if( moduleCC )
                    OPTIX_CHECK_SAME_CONTEXT( context, moduleCC->getDeviceContext(),
                                              "programDescriptions[" + iStr + "].callables.moduleCC" );


                const char* nameDC = programDescriptions[i].callables.entryFunctionNameDC;
                const char* nameCC = programDescriptions[i].callables.entryFunctionNameCC;

                if( !!moduleDC ^ !!nameDC )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "exactly one of programDescriptions[" + iStr
                                                      + "].callables.moduleDC and programDescriptions[" + iStr
                                                      + "].callables.entryFunctionNameDC is null " );
                if( !!moduleCC ^ !!nameCC )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "exactly one of programDescriptions[" + iStr
                                                      + "].callables.moduleCC and programDescriptions[" + iStr
                                                      + "].callables.entryFunctionNameCC is null " );

                if( !moduleDC && !moduleCC )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "no DC nor CC program set in programDescriptions[" + iStr
                                                      + "].callables" );


                const std::string prefixDC = "__direct_callable__";
                const std::string prefixCC = "__continuation_callable__";

                if( nameDC && !corelib::stringBeginsWith( nameDC, prefixDC ) )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "programDescriptions[" + iStr
                                                      + "].callables.entryFunctionNameDC does not start with \""
                                                      + prefixDC + "\"" );

                if( nameCC && !corelib::stringBeginsWith( nameCC, prefixCC ) )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "programDescriptions[" + iStr
                                                      + "].callables.entryFunctionNameCC does not start with \""
                                                      + prefixCC + "\"" );

                if( nameDC && !moduleDC->hasEntryFunction( nameDC ) )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "\"" + std::string( nameDC ) + "\" not found in programDescriptions["
                                                      + iStr + "].callables.moduleDC" );

                if( nameCC && !moduleCC->hasEntryFunction( nameCC ) )
                    return errDetails.logDetails( OPTIX_ERROR_ILWALID_VALUE,
                                                  "\"" + std::string( nameCC ) + "\" not found in programDescriptions["
                                                      + iStr + "].callables.moduleCC" );

                break;
            }
        }
        payloadTypes.push_back( payloadType );
    }

    for( unsigned int i = 0; i < numProgramGroups; ++i )
    {
        std::unique_ptr<ProgramGroup> tmp( new ProgramGroup( context, programDescriptions[i], payloadTypes[i] ) );
        context->registerProgramGroup( tmp.get(), errDetails );
        programGroups[i] = apiCast( tmp.release() );
    }

    return OPTIX_SUCCESS;
}

static OptixResult packSbtRecordHeader( ProgramGroup* programGroup, void* sbtRecordHeaderHostPointer, ErrorDetails& errDetails )
{
    DeviceContext* context = programGroup->getDeviceContext();
    switch( programGroup->getImpl().kind )
    {
        case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
        {
            Module* module;
            implCast( programGroup->getImpl().raygen.module, module );
            const char* programName = programGroup->getImpl().raygen.entryFunctionName;

            RtcCompiledModule rtcModule = nullptr;

            Rtlw32 programIndex = ~0;
            if( OptixResult result = module->getRtcCompiledModuleAndProgramIndex( programName, rtcModule, programIndex, errDetails ) )
                return result;

            if( const RtcResult rtcResult =
                    context->getRtcore().packSbtRecordHeader( context->getRtcDeviceContext(), rtcModule, programIndex,
                                                              nullptr, ~0, nullptr, ~0, sbtRecordHeaderHostPointer ) )
                return errDetails.logDetails( rtcResult, "Failed to pack SBT header" );
            break;
        }

        case OPTIX_PROGRAM_GROUP_KIND_MISS:
        {
            Module* module;
            implCast( programGroup->getImpl().miss.module, module );
            const char* programName = programGroup->getImpl().miss.entryFunctionName;

            RtcCompiledModule rtcModule = nullptr;

            Rtlw32 programIndex = ~0;
            if( module )
                if( OptixResult result = module->getRtcCompiledModuleAndProgramIndex( programName, rtcModule, programIndex, errDetails ) )
                    return result;

            if( const RtcResult rtcResult =
                    context->getRtcore().packSbtRecordHeader( context->getRtcDeviceContext(), rtcModule, programIndex,
                                                              nullptr, ~0, nullptr, ~0, sbtRecordHeaderHostPointer ) )
                return errDetails.logDetails( rtcResult, "Failed to pack SBT header" );
            break;
        }

        case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
        {
            Module* module;
            implCast( programGroup->getImpl().exception.module, module );
            const char* programName = programGroup->getImpl().exception.entryFunctionName;

            RtcCompiledModule rtcModule = nullptr;

            Rtlw32 programIndex = ~0;
            if( module )
                if( OptixResult result = module->getRtcCompiledModuleAndProgramIndex( programName, rtcModule, programIndex, errDetails ) )
                    return result;

            if( const RtcResult rtcResult =
                    context->getRtcore().packSbtRecordHeader( context->getRtcDeviceContext(), rtcModule, programIndex, nullptr,
                                                              ~0, nullptr, ~0, sbtRecordHeaderHostPointer ) )
                return errDetails.logDetails( rtcResult, "Failed to pack SBT header" );
            break;
        }

        case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
        {
            Module* moduleCH;
            Module* moduleAH;
            Module* moduleIS;

            implCast( programGroup->getImpl().hitgroup.moduleCH, moduleCH );
            implCast( programGroup->getImpl().hitgroup.moduleAH, moduleAH );
            implCast( programGroup->getImpl().hitgroup.moduleIS, moduleIS );

            const char* programNameCH = programGroup->getImpl().hitgroup.entryFunctionNameCH;
            const char* programNameAH = programGroup->getImpl().hitgroup.entryFunctionNameAH;
            const char* programNameIS = programGroup->getImpl().hitgroup.entryFunctionNameIS;

            RtcCompiledModule rtcModuleCH = nullptr;
            RtcCompiledModule rtcModuleAH = nullptr;
            RtcCompiledModule rtcModuleIS = nullptr;

            Rtlw32 programIndexCH = ~0;
            if( moduleCH )
                if( OptixResult result = moduleCH->getRtcCompiledModuleAndProgramIndex( programNameCH, rtcModuleCH, programIndexCH, errDetails ) )
                    return result;
            Rtlw32 programIndexAH = ~0;
            if( moduleAH )
                if( OptixResult result = moduleAH->getRtcCompiledModuleAndProgramIndex( programNameAH, rtcModuleAH, programIndexAH, errDetails ) )
                    return result;
            Rtlw32 programIndexIS = ~0;
            if( moduleIS )
                if( OptixResult result = moduleIS->getRtcCompiledModuleAndProgramIndex( programNameIS, rtcModuleIS, programIndexIS, errDetails ) )
                    return result;

            if( const RtcResult rtcResult = context->getRtcore().packSbtRecordHeader(
                    context->getRtcDeviceContext(), rtcModuleCH, programIndexCH, rtcModuleAH, programIndexAH,
                    rtcModuleIS, programIndexIS, sbtRecordHeaderHostPointer ) )
                return errDetails.logDetails( rtcResult, "Failed to pack SBT header" );
            break;
        }

        case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
        {
            Module* moduleDC;
            Module* moduleCC;

            implCast( programGroup->getImpl().callables.moduleDC, moduleDC );
            implCast( programGroup->getImpl().callables.moduleCC, moduleCC );

            const char* programNameDC = programGroup->getImpl().callables.entryFunctionNameDC;
            const char* programNameCC = programGroup->getImpl().callables.entryFunctionNameCC;

            RtcCompiledModule rtcModuleDC = nullptr;
            RtcCompiledModule rtcModuleCC = nullptr;

            Rtlw32 programIndexDC = ~0;
            if( programNameDC )
                if( OptixResult result = moduleDC->getRtcCompiledModuleAndProgramIndex( programNameDC, rtcModuleDC, programIndexDC, errDetails ) )
                    return result;
            Rtlw32 programIndexCC = ~0;
            if( programNameCC )
                if( OptixResult result = moduleCC->getRtcCompiledModuleAndProgramIndex( programNameCC, rtcModuleCC, programIndexCC, errDetails ) )
                    return result;

            if( const RtcResult rtcResult =
                    context->getRtcore().packSbtRecordHeader( context->getRtcDeviceContext(), rtcModuleCC, programIndexCC, rtcModuleDC,
                                                              programIndexDC, nullptr, ~0, sbtRecordHeaderHostPointer ) )
                return errDetails.logDetails( rtcResult, "Failed to pack SBT header" );
            break;
        }
    }
    return OPTIX_SUCCESS;
}
}  // end namespace optix_exp

extern "C" OptixResult optixProgramGroupCreate( OptixDeviceContext              contextAPI,
                                                const OptixProgramGroupDesc*    programDescriptions,
                                                unsigned int                    numProgramGroups,
                                                const OptixProgramGroupOptions* options,
                                                char*                           logString,
                                                size_t*                         logStringSize,
                                                OptixProgramGroup*              programGroups )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_CONTEXT_W_LOG_STRING();
    SCOPED_LWTX_RANGE( context->getLWTXProfiler(), LWTXRegisteredString::PROGRAM_GROUP_CREATE );
    optix_exp::DeviceContextLogger& clog = context->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT_W_LOG_STRING( programDescriptions );
    OPTIX_CHECK_NULL_ARGUMENT_W_LOG_STRING( programGroups );
    OPTIX_CHECK_ZERO_ARGUMENT_W_LOG_STRING( numProgramGroups );

    *programGroups = nullptr;

    try
    {
        optix_exp::ErrorDetails errDetails;

        OptixResult result =
            createProgramGroups( context, programDescriptions, numProgramGroups, options, programGroups, errDetails );

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


extern "C" OptixResult optixProgramGroupDestroy( OptixProgramGroup programGroupAPI )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( ProgramGroup, programGroup, "OptixProgramGroup" );
    SCOPED_LWTX_RANGE( programGroup->getDeviceContext()->getLWTXProfiler(), LWTXRegisteredString::PROGRAM_GROUP_DESTROY );
    optix_exp::DeviceContextLogger& clog = programGroup->getDeviceContext()->getLogger();

    try
    {
        optix_exp::ErrorDetails errDetails;
        OptixResult             result = programGroup->destroy( errDetails );
        if( result )
            clog.sendError( errDetails );
        delete programGroup;
        return result;
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

extern "C" OptixResult optixSbtRecordPackHeader( OptixProgramGroup programGroupAPI, void* sbtRecordHeaderHostPointer )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( ProgramGroup, programGroup, "OptixProgramGroup" );
    SCOPED_LWTX_RANGE( programGroup->getDeviceContext()->getLWTXProfiler(), LWTXRegisteredString::SBT_RECORD_PACK_HEADER );
    optix_exp::DeviceContextLogger& clog = programGroup->getDeviceContext()->getLogger();

    OPTIX_CHECK_NULL_ARGUMENT( sbtRecordHeaderHostPointer );

    try
    {
        optix_exp::ErrorDetails errDetails;
        OptixResult             result = packSbtRecordHeader( programGroup, sbtRecordHeaderHostPointer, errDetails );
        if( result )
            clog.sendError( errDetails );
        return result;
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}

namespace optix_exp {
OptixResult getProgramGroupStackSize( ProgramGroup* programGroup, OptixStackSizes* stackSizes, ErrorDetails& errDetails )
{
    DeviceContext*       context = programGroup->getDeviceContext();
    DeviceContextLogger& clog    = context->getLogger();
    OPTIX_CHECK_NULL_ARGUMENT( stackSizes );

    memset( stackSizes, 0, sizeof( OptixStackSizes ) );

    switch( programGroup->getImpl().kind )
    {
        case OPTIX_PROGRAM_GROUP_KIND_RAYGEN:
        {
            Module* module;
            implCast( programGroup->getImpl().raygen.module, module );
            const char* programName = programGroup->getImpl().raygen.entryFunctionName;

            Rtlw32 dss = 0, css = 0;
            RtcCompiledModule rtcModule    = nullptr;
            Rtlw32            programIndex = ~0;
            if( OptixResult result = module->getRtcCompiledModuleAndProgramIndex( programName, rtcModule, programIndex, errDetails ) )
                return result;

            if( const RtcResult rtcResult =
                    context->getRtcore().compiledModuleGetStackSize( rtcModule, programIndex, &dss, &css ) )
                return errDetails.logDetails( rtcResult, "Failed to get stack size" );
            stackSizes->cssRG = css;
            break;
        }

        case OPTIX_PROGRAM_GROUP_KIND_MISS:
        {
            Module* module;
            implCast( programGroup->getImpl().miss.module, module );
            if( !module )
                break;

            const char* programName = programGroup->getImpl().miss.entryFunctionName;

            Rtlw32 dss = 0, css = 0;
            RtcCompiledModule rtcModule    = nullptr;
            Rtlw32            programIndex = ~0;
            if( OptixResult result = module->getRtcCompiledModuleAndProgramIndex( programName, rtcModule, programIndex, errDetails ) )
                return result;

            if( const RtcResult rtcResult =
                    context->getRtcore().compiledModuleGetStackSize( rtcModule, programIndex, &dss, &css ) )
                return errDetails.logDetails( rtcResult, "Failed to get stack size" );
            stackSizes->cssMS = css;
            break;
        }

        case OPTIX_PROGRAM_GROUP_KIND_EXCEPTION:
        {
            // nothing to do
            break;
        }

        case OPTIX_PROGRAM_GROUP_KIND_HITGROUP:
        {
            Module* moduleCH;
            Module* moduleAH;
            Module* moduleIS;

            implCast( programGroup->getImpl().hitgroup.moduleCH, moduleCH );
            implCast( programGroup->getImpl().hitgroup.moduleAH, moduleAH );
            implCast( programGroup->getImpl().hitgroup.moduleIS, moduleIS );

            const char* programNameCH = programGroup->getImpl().hitgroup.entryFunctionNameCH;
            const char* programNameAH = programGroup->getImpl().hitgroup.entryFunctionNameAH;
            const char* programNameIS = programGroup->getImpl().hitgroup.entryFunctionNameIS;

            Rtlw32 dssCH = 0, cssCH = 0;
            Rtlw32 dssAH = 0, cssAH = 0;
            Rtlw32 dssIS = 0, cssIS = 0;

            if( moduleCH )
            {
                RtcCompiledModule rtcModule    = nullptr;
                Rtlw32            programIndex = ~0;
                if( OptixResult result = moduleCH->getRtcCompiledModuleAndProgramIndex( programNameCH, rtcModule, programIndex, errDetails ) )
                    return result;

                if( const RtcResult rtcResult = context->getRtcore().compiledModuleGetStackSize(
                        rtcModule, programIndex, &dssCH, &cssCH ) )
                    return errDetails.logDetails( rtcResult, "Failed to get stack size" );
            }
            if( moduleAH )
            {
                RtcCompiledModule rtcModule    = nullptr;
                Rtlw32            programIndex = ~0;
                if( OptixResult result = moduleAH->getRtcCompiledModuleAndProgramIndex( programNameAH, rtcModule, programIndex, errDetails ) )
                    return result;

                if( const RtcResult rtcResult = context->getRtcore().compiledModuleGetStackSize(
                        rtcModule, programIndex, &dssAH, &cssAH ) )
                    return errDetails.logDetails( rtcResult, "Failed to get stack size" );
            }
            if( moduleIS )
            {
                RtcCompiledModule rtcModule    = nullptr;
                Rtlw32            programIndex = ~0;
                if( OptixResult result = moduleIS->getRtcCompiledModuleAndProgramIndex( programNameIS, rtcModule, programIndex, errDetails ) )
                    return result;

                if( const RtcResult rtcResult = context->getRtcore().compiledModuleGetStackSize(
                        rtcModule, programIndex, &dssIS, &cssIS ) )
                    return errDetails.logDetails( rtcResult, "Failed to get stack size" );
            }

            stackSizes->cssCH = cssCH;
            stackSizes->cssAH = cssAH;
            stackSizes->cssIS = cssIS;
            break;
        }

        case OPTIX_PROGRAM_GROUP_KIND_CALLABLES:
        {
            Module* moduleDC;
            Module* moduleCC;

            implCast( programGroup->getImpl().callables.moduleDC, moduleDC );
            implCast( programGroup->getImpl().callables.moduleCC, moduleCC );

            const char* programNameDC = programGroup->getImpl().callables.entryFunctionNameDC;
            const char* programNameCC = programGroup->getImpl().callables.entryFunctionNameCC;

            Rtlw32 dssDC = 0, cssDC = 0;
            Rtlw32 dssCC = 0, cssCC = 0;

            if( moduleDC )
            {
            RtcCompiledModule rtcModule    = nullptr;
            Rtlw32            programIndex = ~0;
            if( OptixResult result = moduleDC->getRtcCompiledModuleAndProgramIndex( programNameDC, rtcModule, programIndex, errDetails ) )
                return result;

                if( const RtcResult rtcResult = context->getRtcore().compiledModuleGetStackSize(
                        rtcModule, programIndex, &dssDC, &cssDC ) )
                    return errDetails.logDetails( rtcResult, "Failed to get stack size" );
                // TODO maybe a warning is sufficient?
                if( cssDC != 0 )
                    return errDetails.logDetails( RTC_ERROR_UNKNOWN, "internal error (cssDC != 0)" );
            }
            if( moduleCC )
            {
            RtcCompiledModule rtcModule    = nullptr;
            Rtlw32            programIndex = ~0;
            if( OptixResult result = moduleCC->getRtcCompiledModuleAndProgramIndex( programNameCC, rtcModule, programIndex, errDetails ) )
                return result;

                if( const RtcResult rtcResult = context->getRtcore().compiledModuleGetStackSize(
                        rtcModule, programIndex, &dssCC, &cssCC ) )
                    return errDetails.logDetails( rtcResult, "Failed to get stack size" );
            }

            stackSizes->dssDC = dssDC;
            stackSizes->cssCC = cssCC;
            break;
        }
    }
    return OPTIX_SUCCESS;
}
}
extern "C" OptixResult optixProgramGroupGetStackSize( OptixProgramGroup programGroupAPI, OptixStackSizes* stackSizes )
{
    OPTIX_CHECK_AND_COLWERT_OPAQUE_ARGUMENT_NO_CLOG( ProgramGroup, programGroup, "OptixProgramGroup" );
    SCOPED_LWTX_RANGE( programGroup->getDeviceContext()->getLWTXProfiler(), LWTXRegisteredString::PROGRAM_GROUP_GET_STACK_SIZE );
    optix_exp::DeviceContextLogger& clog = programGroup->getDeviceContext()->getLogger();

    try
    {
        optix_exp::ErrorDetails errDetails;
        OptixResult             result = getProgramGroupStackSize( programGroup, stackSizes, errDetails );
        if( result )
            clog.sendError( errDetails );
        return result;
    }
    OPTIX_API_EXCEPTION_CHECK;

    return OPTIX_SUCCESS;
}
