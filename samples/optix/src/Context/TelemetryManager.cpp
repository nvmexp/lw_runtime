// Copyright (c) 2017, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <Context/LwTelemetryClient.h>
#include <Context/TelemetryManager.h>
#include <iostream>
#include <prodlib/system/Logger.h>
#ifdef _WIN32
#include <Rpc.h>  // UuidCreate
#endif

using namespace optix;

#ifdef _WIN32
static std::string uuid_generate()
{
    UUID uuid;
    UuidCreate( &uuid );

    unsigned char* str;
    UuidToStringA( &uuid, &str );

    std::string s( (char*)str );

    RpcStringFreeA( &str );

    return s;
}
#endif

TelemetryManager::TelemetryManager( Context* context )
    : m_context( context )
    , m_is_initialized( false )
{
#ifdef _WIN32
    m_contextUUID = uuid_generate();

    if( m_telemetry_on )
    {
        HRESULT res;
        if( ( res = LwTelemetry::optix_lwbackend::Init() ) == E_FAIL )
            lwarn << "LwTelemetry client library could not be loaded." << std::endl;
        else if( res != S_OK )
            lwarn << "LwTelemetry client could not be initialized." << std::endl;
        else
            m_is_initialized = true;
    }
#endif
}

TelemetryManager::~TelemetryManager()
{
#ifdef _WIN32
    HRESULT res;
    if( m_telemetry_on )
        if( ( res = LwTelemetry::optix_lwbackend::DeInit() ) != S_OK )
            lwarn << "Error uninitializing LwTelemetry client." << std::endl;
#endif
}

void TelemetryManager::setContextCreateData( const char* name, const char* value )
{
    m_ContextCreateStrData[name] = value;
}

// Add data to the collection
void TelemetryManager::setContextCreateData( const char* name, int value )
{
    m_ContextCreateNumData[name] = value;
}

void TelemetryManager::setContextTeardownData( const char* name, const char* value )
{
    m_ContextTeardownStrData[name] = value;
}

// Add data to the collection
void TelemetryManager::setContextTeardownData( const char* name, int value )
{
    m_ContextTeardownNumData[name] = value;
}


bool TelemetryManager::uploadContextCreateData()
{
#ifdef _WIN32
    if( m_telemetry_on )
    {
        // Create telemetry event and send it to GFE servers.
        if( m_is_initialized )
        {
            HRESULT res;
            res = LwTelemetry::optix_lwbackend::Send_ContextCreate_Event(
                m_ContextCreateNumData["lwda_device"], m_ContextCreateNumData["gpu_memory"],
                m_ContextCreateStrData["gpu_name"], m_ContextCreateNumData["sm_arc"],
                m_ContextCreateNumData["sm_clock"], m_ContextCreateNumData["sm_count"],
                m_ContextCreateNumData["tcc_driver"], m_ContextCreateStrData["display_driver"],
                m_ContextCreateStrData["compatible_devices"], m_ContextCreateStrData["optix_build"], m_contextUUID, m_clientUUID,
                std::string( "0.2" ),       // client version
                std::string( "undefined" )  // user id
                );
            if( res != S_OK )
            {
                lwarn << "ContextCreate event data could not be uploaded." << std::endl;
                return false;
            }
        }
        else
        {
            lwarn << "LwTelemetry client library not loaded." << std::endl;
            return false;
        }

        return true;
    }
#endif
    return false;
}

bool TelemetryManager::uploadContextTeardownData()
{
#ifdef _WIN32
    if( m_telemetry_on )
    {
        // Create telemetry event and send it to GFE servers.
        if( m_is_initialized )
        {
            HRESULT res;
            res = LwTelemetry::optix_lwbackend::Send_ContextTearDown_Event(
                m_ContextTeardownNumData["context_lifetime"], m_context->getDenoiserLaunchCount(),
                m_context->getKernelLaunchCount(), m_context->getDenoiserTimeSpent(), m_contextUUID, m_clientUUID,
                std::string( "0.2" ),       // client version
                std::string( "undefined" )  // user id
                );
            if( res != S_OK )
            {
                lwarn << "ContextTearDown event data could not be uploaded." << std::endl;
                return false;
            }
        }
        else
        {
            lwarn << "LwTelemetry client library not loaded." << std::endl;
            return false;
        }

        return true;
    }
#endif
    return false;
}
