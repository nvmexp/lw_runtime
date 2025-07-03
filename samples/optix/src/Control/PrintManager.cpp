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

#include <Control/PrintManager.h>

#include <Context/Context.h>
#include <Context/UpdateManager.h>
#include <Device/LWDADevice.h>
#include <Device/DeviceManager.h>
#include <corelib/misc/String.h>
#include <prodlib/exceptions/IlwalidValue.h>
#include <prodlib/system/Knobs.h>

// clang-format off
namespace {
  Knob<int>         k_enableOverride( RT_DSTRING( "print.enableOverride" ), -1, RT_DSTRING( "Override the print enabled flag. 0-force off, 1-force on. (Default: -1)" ) );
  Knob<std::string> k_indexOverride(  RT_DSTRING( "print.indexOverride" ),  "", RT_DSTRING( "Override the print index (comma separated)" ) );
}
// clang-format on

#include <optixu/optixu_math.h>  // make_int3

using namespace optix;
using namespace corelib;
using namespace prodlib;


PrintManager::PrintManager( Context* context )
    : m_context( context )
    , m_printLaunchIndex( make_int3( -1, -1, -1 ) )
{
}

PrintManager::~PrintManager()
{
}

void PrintManager::setPrintEnabled( bool enabled )
{
    if( k_enableOverride.get() != -1 )
        enabled = k_enableOverride.get() != 0;

    if( m_printEnabled == enabled )  // no change
        return;

    // Set the new flag and mark context for recompile.
    m_context->getUpdateManager()->eventContextSetPrinting( m_printEnabled, m_printBufferSize, m_printLaunchIndex,
                                                            enabled, m_printBufferSize, m_printLaunchIndex );
    m_printEnabled = enabled;
}

bool PrintManager::getPrintEnabled() const
{
    return m_printEnabled;
}

void PrintManager::setPrintBufferSize( size_t bufsize )
{
    DeviceManager* deviceManager = m_context->getDeviceManager();

    // Go over active GPU devices and force the new size of the buffer.
    for( LWDADevice* lwdaDevice : LWDADeviceArrayView( deviceManager->activeDevices() ) )
    {
        if( lwdaDevice->isEnabled() )
            lwdaDevice->setPrintBufferSize( bufsize );
    }

    // We store the buffer size requested by the use, not the buffer size actually set onto the device.
    m_printBufferSize = bufsize;
}

size_t PrintManager::getPrintBufferSize() const
{
    // We return the value that was user asked for, not the actual value set onto the device.
    return m_printBufferSize;
}

void PrintManager::setPrintLaunchIndex( int x, int y, int z )
{
    int3 newLaunchIndex = make_int3( x, y, z );
    if( m_printLaunchIndex == newLaunchIndex )
        return;

    m_context->getUpdateManager()->eventContextSetPrinting( m_printEnabled, m_printBufferSize, m_printLaunchIndex,
                                                            m_printEnabled, m_printBufferSize, newLaunchIndex );
    m_printLaunchIndex = newLaunchIndex;
}

int3 PrintManager::getPrintLaunchIndex() const
{
    int3 printLaunchIndex = m_printLaunchIndex;
    if( !k_indexOverride.get().empty() )
    {
        std::vector<std::string> tokens = tokenize( k_indexOverride.get(), " ," );
        if( tokens.empty() || tokens.size() > 3 )
            throw IlwalidValue( RT_EXCEPTION_INFO, RT_DSTRING( "Invalid format for print index override" ) );
        printLaunchIndex.x = ( 0 < tokens.size() ) ? from_string<int>( tokens[0] ) : -1;
        printLaunchIndex.y = ( 1 < tokens.size() ) ? from_string<int>( tokens[1] ) : -1;
        printLaunchIndex.z = ( 2 < tokens.size() ) ? from_string<int>( tokens[2] ) : -1;
    }

    return printLaunchIndex;
}
