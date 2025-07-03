// Copyright (c) 2021, LWPU CORPORATION.
// TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED
// *AS IS* AND LWPU AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS
// OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY
// AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL LWPU OR ITS SUPPLIERS
// BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES
// WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS,
// BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PELWNIARY LOSS)
// ARISING OUT OF THE USE OF OR INABILITY TO USE THIS SOFTWARE, EVEN IF LWPU HAS
// BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES

#include <Util/SystemInfo.h>

#include <Util/LWML.h>
#include <corelib/system/System.h>
#include <prodlib/system/System.h>
#include <prodlib/system/Knobs.h>

#include <private/optix_version_string.h>

using namespace prodlib;

namespace optix {

SystemInfo getSystemInfo()
{
    SystemInfo info{};
    info.timestamp = corelib::getTimestamp();

    info.platform        = getPlatform();
    info.hostName        = getHostName();
    info.cpuName         = getCPUName();
    info.numCpuCores     = getNumberOfCPUCores();
    info.availableMemory = getAvailableSystemMemoryInBytes();

    info.driverVersion = LWML::driverVersion();
    info.gpuDescriptions.clear();
    unsigned int count = LWML::deviceCount();
    for( unsigned int i = 0; i < count; ++i )
        info.gpuDescriptions.push_back( LWML::deviceName( i ) );

    info.buildDescription = OPTIX_BUILD_DESCRIPTION;
    info.nondefaultKnobs  = knobRegistry().getNonDefaultKnobs();
    return info;
}

}  // namespace optix
