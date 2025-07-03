// Copyright LWPU Corporation 2019
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <prodlib/misc/LWTXProfiler.h>

using namespace optix_exp;

LWTXProfiler::LWTXProfiler( const char* domainName )
    : m_domain( lwtxDomainCreateA( domainName ) )
{
}

LWTXProfiler::~LWTXProfiler()
{
    lwtxDomainDestroy( m_domain );
}

const char* LWTXProfiler::getMessageString( LWTXRegisteredString messageIdentifier )
{
    switch( messageIdentifier )
    {
        case LWTXRegisteredString::PTX_ENCRYPTION_GET_OPTIX_SALT:
            return "optixExtPtxEncryptionGetOptixSalt";
        case LWTXRegisteredString::PTX_ENCRYPTION_SET_OPTIX_SALT:
            return "optixExtPtxEncryptionSetOptixSalt";
        case LWTXRegisteredString::PTX_ENCRYPTION_SET_VENDOR_SALT:
            return "optixExtPtxEncryptionSetVendorSalt";
        case LWTXRegisteredString::PTX_ENCRYPTION_SET_PUBLIC_VENDOR_KEY:
            return "optixExtPtxEncryptionSetPublicVendorKey";
        case LWTXRegisteredString::DEVICE_CONTEXT_GET_PROPERTY:
            return "optixDeviceContextGetProperty";
        case LWTXRegisteredString::DEVICE_CONTEXT_SET_LOG_CALLBACK:
            return "optixDeviceContextSetLogCallback";
        case LWTXRegisteredString::DEVICE_CONTEXT_SET_CACHE_ENABLED:
            return "optixDeviceContextSetCacheEnabled";
        case LWTXRegisteredString::DEVICE_CONTEXT_SET_CACHE_LOCATION:
            return "optixDeviceContextSetCacheLocation";
        case LWTXRegisteredString::DEVICE_CONTEXT_GET_CACHE_ENABLED:
            return "optixDeviceContextGetCacheEnabled";
        case LWTXRegisteredString::DEVICE_CONTEXT_GET_CACHE_LOCATION:
            return "optixDeviceContextGetCacheLocation";
        case LWTXRegisteredString::DEVICE_CONTEXT_GET_CACHE_DATABASE_SIZES:
            return "optixDeviceContextGetCacheDatabaseSizes";
        case LWTXRegisteredString::DEVICE_CONTEXT_SET_CACHE_DATABASE_SIZES:
            return "optixDeviceContextSetCacheDatabaseSizes";
        case LWTXRegisteredString::DENOISER_CREATE:
            return "optixDenoiserCreate";
        case LWTXRegisteredString::DENOISER_DESTROY:
            return "optixDenoiserDestroy";
        case LWTXRegisteredString::DENOISER_ILWOKE:
            return "optixDenoiserIlwoke";
        case LWTXRegisteredString::DENOISER_COMPUTE_MEMORY_RESOURCES:
            return "optixDenoiserComputeMemoryResources";
        case LWTXRegisteredString::DENOISER_SETUP:
            return "optixDenoiserSetup";
        case LWTXRegisteredString::DENOISER_SET_MODEL:
            return "optixDenoiserSetModel";
        case LWTXRegisteredString::DENOISER_COMPUTE_INTENSITY:
            return "optixDenoiserComputeIntensity";
        case LWTXRegisteredString::PROGRAM_GROUP_CREATE:
            return "optixProgramGroupCreate";
        case LWTXRegisteredString::PROGRAM_GROUP_DESTROY:
            return "optixProgramGroupDestroy";
        case LWTXRegisteredString::SBT_RECORD_PACK_HEADER:
            return "optixSbtRecordPackHeader";
        case LWTXRegisteredString::PROGRAM_GROUP_GET_STACK_SIZE:
            return "optixProgramGroupGetStackSize";
        case LWTXRegisteredString::PIPELINE_CREATE:
            return "optixPipelineCreate";
        case LWTXRegisteredString::PIPELINE_DESTROY:
            return "optixPipelineDestroy";
        case LWTXRegisteredString::PIPELINE_SET_STACK_SIZE:
            return "optixPipelineSetStackSize";
        case LWTXRegisteredString::LAUNCH:
            return "optixLaunch";
        case LWTXRegisteredString::MODULE_CREATE_FROM_PTX:
            return "optixModuleCreateFromPTX";
        case LWTXRegisteredString::MODULE_CREATE_FROM_PTX_WITH_TASKS:
            return "optixModuleCreateFromPTXWithTasks";
        case LWTXRegisteredString::MODULE_DESTROY:
            return "optixModuleDestroy";
        case LWTXRegisteredString::MODULE_GET_COMPILATION_STATE:
            return "optixModuleGetCompilationState";
        case LWTXRegisteredString::TASK_EXELWTE:
            return "optixTaskExelwte";
        case LWTXRegisteredString::BUILTIN_IS_MODULE_GET:
            return "optixBuiltinISModuleGet";
        case LWTXRegisteredString::ACCEL_COMPUTE_MEMORY_USAGE:
            return "optixAccelComputeMemoryUsage";
        case LWTXRegisteredString::ACCEL_BUILD:
            return "optixAccelBuild";
        case LWTXRegisteredString::ACCEL_GET_RELOCATION_INFO:
            return "optixAccelGetRelocationInfo";
        case LWTXRegisteredString::ACCEL_CHECK_RELOCATION_COMPATIBILITY:
            return "optixAccelCheckRelocationCompatibility";
        case LWTXRegisteredString::ACCEL_RELOCATE:
            return "optixAccelRelocate";
        case LWTXRegisteredString::ACCEL_COMPACT:
            return "optixAccelCompact";
        case LWTXRegisteredString::ACCEL_INSTANCE_AABBS_COMPUTE_MEMORY_USAGE:
            return "optixAccelInstanceAabbsComputeMemoryUsage";
        case LWTXRegisteredString::ACCEL_INSTANCE_AABBS_COMPUTE:
            return "optixAccelInstanceAabbsCompute";
        case LWTXRegisteredString::COLWERT_POINTER_TO_TRAVERSABLE_HANDLE:
            return "optixColwertPointerToTraversableHandle";
#if LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
        case LWTXRegisteredString::VISIBILITY_MAP_ARRAY_COMPUTE_MEMORY_USAGE:
            return "optixVisibilityMapArrayComputeMemoryUsage";
        case LWTXRegisteredString::VISIBILITY_MAP_ARRAY_BUILD:
            return "optixVisibilityMapArrayBuild";
        case LWTXRegisteredString::DISPLACED_MICROMESH_ARRAY_COMPUTE_MEMORY_USAGE:
            return "optixDisplacedMicromeshArrayComputeMemoryUsage";
        case LWTXRegisteredString::DISPLACED_MICROMESH_ARRAY_BUILD:
            return "optixDisplacedMicromeshArrayBuild";
#endif  // LWCFG(GLOBAL_FEATURE_GR1354_MICROMESH)
        case LWTXRegisteredString::MESSAGES_END:
            return "optixLWTXStringNotFound";
    }

    return "";
}

lwtxStringHandle_t LWTXProfiler::getMessage( LWTXRegisteredString messageIdentifier )
{
    const int messageIndex = static_cast<int>( messageIdentifier );
    if( m_messages[messageIndex] == NULL )
    {
        m_messages[messageIndex] = lwtxDomainRegisterStringA( m_domain, getMessageString( messageIdentifier ) );
    }
    return m_messages[messageIndex];
}

// ARGB color for Lwpu green.
static const int LWIDIA_GREEN = 0xFF76B900;

void LWTXProfiler::pushRange( LWTXRegisteredString profileString )
{
    lwtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version               = LWTX_VERSION;
    eventAttrib.size                  = LWTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType             = LWTX_COLOR_ARGB;
    eventAttrib.color                 = LWIDIA_GREEN;
    eventAttrib.messageType           = LWTX_MESSAGE_TYPE_REGISTERED;
    eventAttrib.message.registered    = getMessage( profileString );

    lwtxDomainRangePushEx( m_domain, &eventAttrib );
}

void LWTXProfiler::popRange()
{
    lwtxDomainRangePop( m_domain );
}
