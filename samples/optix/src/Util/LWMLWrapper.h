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

#pragma once

#include <support/lwml/include/lwml.h>

#include <corelib/system/ExelwtableModule.h>

#include <memory>

namespace optix {
namespace LWML {

class Wrapper
{
  public:
    Wrapper()  = default;
    ~Wrapper() = default;

    void load();
    void unload();

    bool         available() const;
    lwmlReturn_t deviceGetCount( unsigned int* deviceCount ) const;
    lwmlReturn_t deviceGetName( lwmlDevice_t device, char* name, unsigned int length ) const;
	lwmlReturn_t deviceGetUUID( lwmlDevice_t device, char* uuid, unsigned int length ) const;
    lwmlReturn_t deviceGetHandleByIndex( unsigned int index, lwmlDevice_t* device ) const;
    lwmlReturn_t deviceGetHandleByPciBusId( const char* pciBusId, lwmlDevice_t* device ) const;
	lwmlReturn_t deviceGetHandleByUUID( const char* uuid, lwmlDevice_t* device ) const;
    lwmlReturn_t deviceGetMemoryInfo( lwmlDevice_t device, lwmlMemory_t* memory ) const;
    lwmlReturn_t deviceGetLwLinkCapability( lwmlDevice_t device, unsigned int link, lwmlLwLinkCapability_t capability, unsigned int* capResult ) const;
    lwmlReturn_t deviceGetLwLinkState( lwmlDevice_t device, unsigned int link, lwmlEnableState_t* isActive ) const;
    lwmlReturn_t deviceGetLwLinkRemotePciInfo( lwmlDevice_t device, unsigned int link, lwmlPciInfo_t* pci ) const;
    lwmlReturn_t deviceGetPciInfo( lwmlDevice_t device, lwmlPciInfo_t* pci ) const;
    lwmlReturn_t systemGetDriverVersion( char* version, unsigned int length ) const;

  private:
    // lwml.dll handle
    std::unique_ptr<corelib::ExelwtableModule> m_lwmlLib;

    // DLL capabilities
    bool m_lwmlAvailable = false;

    using lwmlInit_t                      = lwmlReturn_t();
    using lwmlShutdown_t                  = lwmlReturn_t();
    using lwmlDeviceGetCount_t            = lwmlReturn_t( unsigned int* deviceCount );
    using lwmlDeviceGetName_t             = lwmlReturn_t( lwmlDevice_t device, char* name, unsigned int length );
	using lwmlDeviceGetUUID_t             = lwmlReturn_t( lwmlDevice_t device, char* uuid, unsigned int length );
    using lwmlDeviceGetHandleByIndex_t    = lwmlReturn_t( unsigned int index, lwmlDevice_t* device );
    using lwmlDeviceGetHandleByPciBusId_t = lwmlReturn_t( const char* pciBusId, lwmlDevice_t* device );
	using lwmlDeviceGetHandleByUUID_t     = lwmlReturn_t( const char* uuid, lwmlDevice_t* device );
    using lwmlDeviceGetMemoryInfo_t       = lwmlReturn_t( lwmlDevice_t device, lwmlMemory_t* memory );
    using lwmlDeviceGetComputeMode_t      = lwmlReturn_t( lwmlDevice_t device, lwmlComputeMode_t* mode );
    using lwmlDeviceGetLwLinkCapability_t = lwmlReturn_t( lwmlDevice_t           device,
                                                          unsigned int           link,
                                                          lwmlLwLinkCapability_t capability,
                                                          unsigned int*          capResult );
    using lwmlDeviceGetPciInfo       = lwmlReturn_t( lwmlDevice_t device, lwmlPciInfo_t* pci );
    using lwmlDeviceGetLwLinkState_t = lwmlReturn_t( lwmlDevice_t device, unsigned int link, lwmlEnableState_t* isActive );
    using lwmlDeviceGetLwLinkRemotePciInfo_t = lwmlReturn_t( lwmlDevice_t device, unsigned int link, lwmlPciInfo_t* pci );
    using lwmlDeviceGetPciInfo_t             = lwmlReturn_t( lwmlDevice_t device, lwmlPciInfo_t* pci );
    using lwmlDeviceSetComputeMode_t         = lwmlReturn_t( lwmlDevice_t device, lwmlComputeMode_t mode );
    using lwmlSystemGetDriverVersion_t       = lwmlReturn_t( char* version, unsigned int length );
    using lwmlErrorString_t                  = const char*( lwmlReturn_t result );

    lwmlInit_t*                         m_init                         = nullptr;
    lwmlShutdown_t*                     m_shutdown                     = nullptr;
    lwmlDeviceGetCount_t*               m_deviceGetCount               = nullptr;
	lwmlDeviceGetName_t*                m_deviceGetName                = nullptr;
	lwmlDeviceGetUUID_t*                m_deviceGetUUID                = nullptr;
    lwmlDeviceGetHandleByIndex_t*       m_deviceGetHandleByIndex       = nullptr;
    lwmlDeviceGetHandleByPciBusId_t*    m_deviceGetHandleByPciBusId    = nullptr;
	lwmlDeviceGetHandleByUUID_t*        m_deviceGetHandleByUUID        = nullptr;
    lwmlDeviceGetMemoryInfo_t*          m_deviceGetMemoryInfo          = nullptr;
    lwmlDeviceGetComputeMode_t*         m_deviceGetComputeMode         = nullptr;
    lwmlDeviceGetLwLinkCapability_t*    m_deviceGetLwLinkCapability    = nullptr;
    lwmlDeviceGetLwLinkState_t*         m_deviceGetLwLinkState         = nullptr;
    lwmlDeviceGetLwLinkRemotePciInfo_t* m_deviceGetLwLinkRemotePciInfo = nullptr;
    lwmlDeviceGetPciInfo_t*             m_deviceGetPciInfo             = nullptr;
    lwmlDeviceSetComputeMode_t*         m_deviceSetComputeMode         = nullptr;
    lwmlSystemGetDriverVersion_t*       m_systemGetDriverVersion       = nullptr;
    lwmlErrorString_t*                  m_errorString                  = nullptr;
};


}  // namespace LWML
}  // namespace optix
