/*
 * Copyright (c) 2021 LWPU CORPORATION.  All rights reserved.
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

#include <Util/LWMLWrapper.h>

#include <corelib/misc/String.h>
#include <corelib/system/ExelwtableModule.h>
#include <corelib/system/SystemError.h>

#include <support/lwml/include/lwml.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Shlobj.h>
#include <comutil.h>
#include <windows.h>
#endif

#include <cctype>
#include <memory>
#include <string>

using namespace corelib;

namespace optix {
namespace LWML {

// Helper for casting function prototypes from an ExelwtableModule.
template <typename Fn>
void getFunction( Fn*& ptr, ExelwtableModule& module, const char* name )
{
    ptr = reinterpret_cast<Fn*>( module.getFunction( name ) );
}

void Wrapper::load()
{
#if defined( _WIN32 )
    const std::string lwmlLibName = "lwml.dll";
#elif defined( __linux__ )
    const std::string lwmlLibName = "liblwidia-ml.so.1";
#else
    // LWML is not supported on mac
    const std::string lwmlLibName = "dummy";
#endif
    std::unique_ptr<corelib::ExelwtableModule> lwmlLib( new ExelwtableModule( lwmlLibName.c_str() ) );

    if( !lwmlLib->initLwLoadSystemLibrary() )
    {
        if( !lwmlLib->init() )
        {
            throw std::runtime_error( "LWML loading error, Cannot open shared library: " + lwmlLibName + "\nERROR: "
                                      + getSystemErrorString() );
        }
    }

    getFunction( m_init, *lwmlLib, "lwmlInit" );
    getFunction( m_shutdown, *lwmlLib, "lwmlShutdown" );
    getFunction( m_deviceGetCount, *lwmlLib, "lwmlDeviceGetCount" );
    getFunction( m_deviceGetName, *lwmlLib, "lwmlDeviceGetName" );
	getFunction( m_deviceGetUUID, *lwmlLib, "lwmlDeviceGetUUID" );
    getFunction( m_deviceGetHandleByIndex, *lwmlLib, "lwmlDeviceGetHandleByIndex" );
    getFunction( m_deviceGetHandleByPciBusId, *lwmlLib, "lwmlDeviceGetHandleByPciBusId" );
	getFunction( m_deviceGetHandleByUUID, *lwmlLib, "lwmlDeviceGetHandleByUUID" );
    getFunction( m_deviceGetMemoryInfo, *lwmlLib, "lwmlDeviceGetMemoryInfo" );
    getFunction( m_deviceGetComputeMode, *lwmlLib, "lwmlDeviceGetComputeMode" );
    getFunction( m_deviceGetLwLinkCapability, *lwmlLib, "lwmlDeviceGetLwLinkCapability" );
    getFunction( m_deviceGetLwLinkState, *lwmlLib, "lwmlDeviceGetLwLinkState" );
    getFunction( m_deviceGetLwLinkRemotePciInfo, *lwmlLib, "lwmlDeviceGetLwLinkRemotePciInfo" );
    getFunction( m_deviceSetComputeMode, *lwmlLib, "lwmlDeviceSetComputeMode" );
    getFunction( m_systemGetDriverVersion, *lwmlLib, "lwmlSystemGetDriverVersion" );
    getFunction( m_errorString, *lwmlLib, "lwmlErrorString" );
    getFunction( m_deviceGetPciInfo, *lwmlLib, "lwmlDeviceGetPciInfo" );

    // If lwmlInit not found then lwml library is not available
    if( !m_init )
    {
        throw std::runtime_error( "lwmlInit not found in LWML library" );
    }
    if( const lwmlReturn_t res = m_init() )
    {
        throw std::runtime_error( "lwmlInit() call failed" );
    }

    m_lwmlLib       = std::move( lwmlLib );
    m_lwmlAvailable = true;
}

void Wrapper::unload()
{
    if( m_lwmlAvailable && m_shutdown )
    {
        if( const lwmlReturn_t res = m_shutdown() )
        {
            throw std::runtime_error( "lwmlShutdown failed." );
        }
    }
}

bool Wrapper::available() const
{
    return m_lwmlAvailable;
}

////////////////////////////////
// LWML functions

lwmlReturn_t Wrapper::deviceGetCount( unsigned int* deviceCount ) const
{
    return m_deviceGetCount( deviceCount );
}

lwmlReturn_t Wrapper::deviceGetName( lwmlDevice_t device, char* name, unsigned int length ) const
{
    return m_deviceGetName( device, name, length );
}

lwmlReturn_t Wrapper::deviceGetUUID( lwmlDevice_t device, char* uuid, unsigned int length ) const
{
	return m_deviceGetUUID( device, uuid, length );
}

lwmlReturn_t Wrapper::deviceGetHandleByIndex( unsigned int index, lwmlDevice_t* device ) const
{
    return m_deviceGetHandleByIndex( index, device );
}

lwmlReturn_t Wrapper::deviceGetHandleByPciBusId( const char* pciBusId, lwmlDevice_t* device ) const
{
    return m_deviceGetHandleByPciBusId( pciBusId, device );
}

lwmlReturn_t Wrapper::deviceGetHandleByUUID( const char* uuid, lwmlDevice_t* device ) const
{
	return m_deviceGetHandleByUUID( uuid, device );
}

lwmlReturn_t Wrapper::deviceGetMemoryInfo( lwmlDevice_t device, lwmlMemory_t* memory ) const
{
    return m_deviceGetMemoryInfo( device, memory );
}

lwmlReturn_t Wrapper::deviceGetLwLinkCapability( lwmlDevice_t           device,
                                                 unsigned int           link,
                                                 lwmlLwLinkCapability_t capability,
                                                 unsigned int*          capResult ) const
{
    return m_deviceGetLwLinkCapability( device, link, capability, capResult );
}

lwmlReturn_t Wrapper::deviceGetLwLinkState( lwmlDevice_t device, unsigned int link, lwmlEnableState_t* isActive ) const
{
    return m_deviceGetLwLinkState( device, link, isActive );
}

lwmlReturn_t Wrapper::deviceGetLwLinkRemotePciInfo( lwmlDevice_t device, unsigned int link, lwmlPciInfo_t* pci ) const
{
    return m_deviceGetLwLinkRemotePciInfo( device, link, pci );
}

lwmlReturn_t Wrapper::deviceGetPciInfo( lwmlDevice_t device, lwmlPciInfo_t* pci ) const
{
    return m_deviceGetPciInfo( device, pci );
}

lwmlReturn_t Wrapper::systemGetDriverVersion( char* version, unsigned int length ) const
{
    return m_systemGetDriverVersion( version, length );
}

}  // namespace LWML
}  // namespace optix
