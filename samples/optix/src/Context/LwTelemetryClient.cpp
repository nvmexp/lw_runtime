
//
// Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.
//

#ifdef _WIN32

#include <Context/LwTelemetryClient.h>

#include <LwTelemetryEvents.h>

#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#pragma warning( push, 0 )
#include <shlobj.h>
#pragma warning( pop )

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace LwTelemetry {
namespace optix_lwbackend {
static std::wstring GetShellDirectory( int csidl )
{
    wchar_t buf[MAX_PATH + 1] = {0};

    HRESULT hr = SHGetFolderPathW( NULL, csidl, NULL, SHGFP_TYPE_LWRRENT, buf );

    if( FAILED( hr ) )
    {
        throw std::runtime_error( "GetShellDirectory failed" );
    }

    return buf;
}

std::wstring dllPath()
{
    std::wstring bitness;
#if _WIN64
    bitness = L"64";
#else
    bitness = L"32";
#endif

    std::wstring res = GetShellDirectory( CSIDL_PROGRAM_FILES );
    res += L"\\LWPU Corporation\\LwTelemetry\\LwTelemetryAPI";
    res += bitness;
    res += L".dll";
    return res;
}

const std::string gs_clientId       = "56653497851724908";
const std::string gs_eventSchemaVer = "0.3";

static auto gs_freeDll = []( HMODULE h ) {
    if( h )
    {
        FreeLibrary( h );
    }
};
static std::unique_ptr<std::remove_pointer<HMODULE>::type, decltype( gs_freeDll )> gs_dll( nullptr, gs_freeDll );
static HRESULT ( *pSend )( const char* );
static HRESULT ( *pInit )();
static HRESULT ( *pDeInit )();

HRESULT Init()
{
    if( !gs_dll || !pInit )
    {
        gs_dll.reset( LoadLibraryW( dllPath().c_str() ) );
        if( !gs_dll )
        {
            return E_FAIL;
        }

        pInit = reinterpret_cast<decltype( pInit )>( GetProcAddress( gs_dll.get(), "Init" ) );
        if( !pInit )
        {
            //probably old API version without Init
            return S_OK;
        }
    }

    return pInit();
}

HRESULT DeInit()
{
    if( !gs_dll || !pDeInit )
    {
        gs_dll.reset( LoadLibraryW( dllPath().c_str() ) );
        if( !gs_dll )
        {
            return S_OK;
        }

        pDeInit = reinterpret_cast<decltype( pDeInit )>( GetProcAddress( gs_dll.get(), "DeInit" ) );
        if( !pDeInit )
        {
            // DeInit is optional and not supported before 1.2.0.0
            return S_OK;
        }
    }

    return pDeInit();
}

// Throwing version of rapidjson::CrtAllocator
class ThrowingCrtAllocator
{
  public:
    static const bool kNeedFree = true;
    void* Malloc( size_t size )
    {
        auto ptr = m_allocator.Malloc( size );
        if( !ptr )
        {
            throw std::bad_alloc();
        }
        return ptr;
    }

    void* Realloc( void* originalPtr, size_t originalSize, size_t newSize )
    {
        auto ptr = m_allocator.Realloc( originalPtr, originalSize, newSize );
        if( !ptr )
        {
            throw std::bad_alloc();
        }
        return ptr;
    }

    static void Free( void* ptr ) { m_allocator.Free( ptr ); }

  private:
    // CrtAllocator is stateless and thread-safe
    static rapidjson::CrtAllocator m_allocator;
};

rapidjson::CrtAllocator ThrowingCrtAllocator::m_allocator;

using RapidjsonDolwment =
    rapidjson::GenericDolwment<rapidjson::UTF8<>, rapidjson::MemoryPoolAllocator<ThrowingCrtAllocator>, ThrowingCrtAllocator>;

using Rapidjsolwalue = rapidjson::GenericValue<rapidjson::UTF8<>, rapidjson::MemoryPoolAllocator<ThrowingCrtAllocator>>;


HRESULT Send_ContextCreate_Event( DeviceNumAttribute    lwdaDevice,
                                  DeviceNumAttribute    gpuMemory,
                                  DeviceStringAttribute gpuName,
                                  DeviceNumAttribute    smArc,
                                  DeviceNumAttribute    smClock,
                                  DeviceNumAttribute    smCount,
                                  DeviceNumAttribute    tccDriver,
                                  DeviceStringAttribute displayDriver,
                                  DeviceStringAttribute compatibleDevices,
                                  BuildString           optixBuild,
                                  UuidString            contextUUID,
                                  UuidString            clientUUID,
                                  const std::string&    clientVer,
                                  const std::string&    userId )
{
    try
    {
        if( !gs_dll || !pSend )
        {
            gs_dll.reset( LoadLibraryW( dllPath().c_str() ) );
            if( !gs_dll )
            {
                throw std::runtime_error( "Failed to load LwTelemetry API DLL" );
            }

            pSend = reinterpret_cast<decltype( pSend )>( GetProcAddress( gs_dll.get(), "LwTelemetrySendEvent" ) );
            if( !pSend )
            {
                throw std::runtime_error( "Could not find method LwTelemetrySendEvent in LwTelemetry API DLL" );
            }
        }

        RapidjsonDolwment d;
        d.SetObject();
        auto& a = d.GetAllocator();

        d.AddMember( "clientId", gs_clientId, a );
        d.AddMember( "clientVer", clientVer, a );
        d.AddMember( "userId", userId, a );
        d.AddMember( "eventSchemaVer", gs_eventSchemaVer, a );
        d.AddMember( "event", Rapidjsolwalue( rapidjson::kObjectType ), a );
        d["event"].AddMember( "name", "ContextCreate", a );
        d["event"].AddMember( "parameters", Rapidjsolwalue( rapidjson::kObjectType ), a );
        d["event"]["parameters"].AddMember( "lwdaDevice", lwdaDevice, a );
        d["event"]["parameters"].AddMember( "gpuMemory", gpuMemory, a );
        d["event"]["parameters"].AddMember( "gpuName", gpuName.substr( 0, 16 ), a );
        d["event"]["parameters"].AddMember( "smArc", smArc, a );
        d["event"]["parameters"].AddMember( "smClock", smClock, a );
        d["event"]["parameters"].AddMember( "smCount", smCount, a );
        d["event"]["parameters"].AddMember( "tccDriver", tccDriver, a );
        d["event"]["parameters"].AddMember( "displayDriver", displayDriver.substr( 0, 16 ), a );
        d["event"]["parameters"].AddMember( "compatible_devices", compatibleDevices, a );
        d["event"]["parameters"].AddMember( "optixBuild", optixBuild, a );
        d["event"]["parameters"].AddMember( "contextUUID", contextUUID, a );
        d["event"]["parameters"].AddMember( "clientUUID", clientUUID, a );

        rapidjson::StringBuffer                    buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer( buffer );
        d.Accept( writer );

        return pSend( buffer.GetString() );
    }
    catch( const std::bad_alloc& )
    {
        return E_OUTOFMEMORY;
    }
    catch( const std::ilwalid_argument& )
    {
        return E_ILWALIDARG;
    }
    catch( const std::exception& )
    {
        return E_FAIL;
    }
}

HRESULT Send_ContextTearDown_Event( NumTimer           contextLifetime,
                                    NumCounter         countDenoiserLaunches,
                                    NumCounter         countKernelLaunches,
                                    NumTimer           sumDenoiserTimeSpent,
                                    UuidString         contextUUID,
                                    UuidString         clientUUID,
                                    const std::string& clientVer,
                                    const std::string& userId )
{
    try
    {
        if( !gs_dll || !pSend )
        {
            gs_dll.reset( LoadLibraryW( dllPath().c_str() ) );
            if( !gs_dll )
            {
                throw std::runtime_error( "Failed to load LwTelemetry API DLL" );
            }

            pSend = reinterpret_cast<decltype( pSend )>( GetProcAddress( gs_dll.get(), "LwTelemetrySendEvent" ) );
            if( !pSend )
            {
                throw std::runtime_error( "Could not find method LwTelemetrySendEvent in LwTelemetry API DLL" );
            }
        }

        RapidjsonDolwment d;
        d.SetObject();
        auto& a = d.GetAllocator();

        d.AddMember( "clientId", gs_clientId, a );
        d.AddMember( "clientVer", clientVer, a );
        d.AddMember( "userId", userId, a );
        d.AddMember( "eventSchemaVer", gs_eventSchemaVer, a );
        d.AddMember( "event", Rapidjsolwalue( rapidjson::kObjectType ), a );
        d["event"].AddMember( "name", "ContextTearDown", a );
        d["event"].AddMember( "parameters", Rapidjsolwalue( rapidjson::kObjectType ), a );

        d["event"]["parameters"].AddMember( "contextLifetime", contextLifetime, a );

        d["event"]["parameters"].AddMember( "countDenoiserLaunches", countDenoiserLaunches, a );

        d["event"]["parameters"].AddMember( "countKernelLaunches", countKernelLaunches, a );

        d["event"]["parameters"].AddMember( "sumDenoiserTimeSpent", sumDenoiserTimeSpent, a );

        d["event"]["parameters"].AddMember( "contextUUID", contextUUID, a );

        d["event"]["parameters"].AddMember( "clientUUID", clientUUID, a );

        rapidjson::StringBuffer                    buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer( buffer );
        d.Accept( writer );

        return pSend( buffer.GetString() );
    }
    catch( const std::bad_alloc& )
    {
        return E_OUTOFMEMORY;
    }
    catch( const std::ilwalid_argument& )
    {
        return E_ILWALIDARG;
    }
    catch( const std::exception& )
    {
        return E_FAIL;
    }
}
}
}
#endif
