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

#include <Util/LWML.h>
#include <Util/LWMLWrapper.h>

#include <corelib/misc/String.h>
#include <corelib/system/ExelwtableModule.h>
#include <corelib/system/SystemError.h>

#include <prodlib/system/Knobs.h>
#include <prodlib/system/Logger.h>

#include <support/lwml/include/lwml.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Shlobj.h>
#include <comutil.h>
#include <windows.h>
#endif

#include <cctype>
#include <memory>
#include <stdexcept>
#include <string>

using namespace corelib;

namespace {
// clang-format off
Knob<bool> k_enableLWML(   RT_DSTRING("lwml.enable"),   true, RT_DSTRING( "Use LWPU Management Library") );
Knob<int>  k_lwmlLogLevel( RT_DSTRING("lwml.logLevel"), 50,   RT_DSTRING( "Log level for LWML status messages") );
// clang-format on
}  // namespace

namespace optix {
namespace LWML {
namespace {

class LWMLImpl
{
  public:
    LWMLImpl();
    ~LWMLImpl();

    static std::string  driverVersion();
    static unsigned int deviceCount();
    static std::string deviceName( unsigned int devIndex );
    static bool canAccessPeerViaLwLink( const std::string& pciBusId0, const std::string& pciBusId1 );
    static MemoryInfo getMemoryInfo( const std::string& pciBusId );

  private:
    bool available() const { return m_lwmlAvailable; }

    // lwml.dll wrapper
    Wrapper m_wrapper;

    // DLL capabilities
    bool m_lwmlAvailable = false;
};

// instance access
LWMLImpl& lwml()
{
    static LWMLImpl funcs;
    return funcs;
}

// Helper for casting function prototypes from an ExelwtableModule.
template <typename Fn>
void getFunction( Fn*& ptr, ExelwtableModule& module, const char* name )
{
    ptr = reinterpret_cast<Fn*>( module.getFunction( name ) );
}

LWMLImpl::LWMLImpl()
{
    if( !k_enableLWML.get() )
        return;

    try
    {
        m_wrapper.load();
    }
    catch( const std::runtime_error& e )
    {
        lwarn << e.what();
        return;
    }

    m_lwmlAvailable = true;
}

LWMLImpl::~LWMLImpl()
{
    if( m_lwmlAvailable )
    {
        try
        {
            m_wrapper.unload();
        }
        catch( const std::runtime_error& e)
        {
            lwarn << e.what();
        }
    }
}

std::string LWMLImpl::driverVersion()
{
    if( lwml().available() )
    {
        char         version[LWML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE];
        lwmlReturn_t res = lwml().m_wrapper.systemGetDriverVersion( version, LWML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE );
        if( res == LWML_SUCCESS )
            return version;
    }
    return "unknown";
}

unsigned int LWMLImpl::deviceCount()
{
    if( lwml().available() )
    {
        unsigned int count;
        lwmlReturn_t res = lwml().m_wrapper.deviceGetCount( &count );
        if( res == LWML_SUCCESS )
            return count;
    }
    return 0;
}

std::string LWMLImpl::deviceName( unsigned int devIndex )
{
    if( lwml().available() )
    {
        lwmlDevice_t device;
        char         name[LWML_DEVICE_NAME_BUFFER_SIZE];
        if( lwml().m_wrapper.deviceGetHandleByIndex( devIndex, &device ) == LWML_SUCCESS
            && lwml().m_wrapper.deviceGetName( device, name, LWML_DEVICE_NAME_BUFFER_SIZE ) == LWML_SUCCESS )
        {
            return name;
        }
    }
    return "Unknown";
}

bool LWMLImpl::canAccessPeerViaLwLink( const std::string& pciBusId0, const std::string& pciBusId1 )
{
    if( !lwml().available() )
    {
        return false;
    }

    lwmlReturn_t res     = LWML_SUCCESS;
    lwmlDevice_t device0 = nullptr;
    res                  = lwml().m_wrapper.deviceGetHandleByPciBusId( pciBusId0.c_str(), &device0 );
    if( res != LWML_SUCCESS )
    {
        return false;
    }
    llog( k_lwmlLogLevel.get() ) << "LWML: checking lwlinks on device with pci bus id: " << pciBusId0 << "\n";
    llog( k_lwmlLogLevel.get() ) << "LWML: searching for peer id: " << pciBusId1 << "\n";

    // Iterate over links on the first device, see if any of them is connected to the second device
    for( unsigned int link = 0; link < LWML_LWLINK_MAX_LINKS; ++link )
    {
        llog( k_lwmlLogLevel.get() ) << "LWML: link: " << link << "\n";

        // Check if P2P is supported on this link
        unsigned int capResult = 0;
        res = lwml().m_wrapper.deviceGetLwLinkCapability( device0, link, LWML_LWLINK_CAP_P2P_SUPPORTED, &capResult );
        if( res != LWML_SUCCESS || capResult == 0 )
        {
            llog( k_lwmlLogLevel.get() ) << "LWML: \tP2P not supported\n";
            continue;
        }

        // Check if LWLINK is active on this link
        lwmlEnableState_t isActive = LWML_FEATURE_DISABLED;
        res                        = lwml().m_wrapper.deviceGetLwLinkState( device0, link, &isActive );
        if( res != LWML_SUCCESS || isActive != LWML_FEATURE_ENABLED )
        {
            llog( k_lwmlLogLevel.get() ) << "LWML: \tlink not active\n";
            continue;
        }

        // Check if we're connected to the second device
        lwmlPciInfo_t pci{};
        res = lwml().m_wrapper.deviceGetLwLinkRemotePciInfo( device0, link, &pci );
        if( res != LWML_SUCCESS )
        {
            llog( k_lwmlLogLevel.get() ) << "LWML: \tcould not get remote PCI info\n";
            continue;
        }

        llog( k_lwmlLogLevel.get() ) << "LWML: \tremote PCI bus id: " << pci.busId << "\n";

        // If the bus ID matches, assume the link is up and we don't have to
        // do the same checks as above on the second device.
        if( caseInsensitiveEquals( std::string( pci.busId ), pciBusId1 ) )
        {
            llog( k_lwmlLogLevel.get() ) << "LWML: \tfound match\n";
            return true;
        }
    }

    return false;
}

MemoryInfo LWMLImpl::getMemoryInfo( const std::string& pciBusId )
{
    MemoryInfo meminfo{};

    if( !lwml().available() )
		return meminfo;
	lwmlDevice_t device = nullptr;
	lwmlReturn_t res    = lwml().m_wrapper.deviceGetHandleByPciBusId( pciBusId.c_str(), &device );

	if( res == LWML_SUCCESS )
	{
		lwmlMemory_t m{};
		res = lwml().m_wrapper.deviceGetMemoryInfo( device, &m );
		if( res == LWML_SUCCESS )
		{
			meminfo.total = m.total;
			meminfo.free  = m.free;
			meminfo.used  = m.used;
		}
	}
    return meminfo;
}

}  // namespace

// Exposed LWML API functions

std::string driverVersion()
{
    return LWMLImpl::driverVersion();
}

unsigned int deviceCount()
{
    return LWMLImpl::deviceCount();
}

std::string deviceName( unsigned int devIndex )
{
    return LWMLImpl::deviceName( devIndex );
}

bool canAccessPeerViaLwLink( const std::string& pciBusId0, const std::string& pciBusId1 )
{
    return LWMLImpl::canAccessPeerViaLwLink( pciBusId0, pciBusId1 );
}

MemoryInfo getMemoryInfo( const std::string& pciBusId )
{
    return LWMLImpl::getMemoryInfo( pciBusId );
}

}  // namespace LWML
}  // namespace optix
