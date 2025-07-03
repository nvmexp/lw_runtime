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

#include <Device/APIDeviceAttributes.h>

#include <Util/ContainerAlgorithm.h>

#include <prodlib/exceptions/IlwalidValue.h>

#include <cstring>
#include <sstream>
#include <vector>

using namespace optix;
using namespace prodlib;

template <typename T>
static void copyAttribute( RTsize size, void* p, const char* name, T value )
{
    if( size != sizeof( T ) )
    {
        std::ostringstream err;
        err << "Invalid copy of attribute \"" << name << "\" of size: " << sizeof( T ) << " to output of size: " << size;
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, err.str() );
    }
    *static_cast<T*>( p ) = value;
}

template <typename T>
static void copyAttribute( RTsize size, void* p, const char* name, const std::vector<T>& values )
{
    const size_t valueSize = sizeof( T ) * values.size();
    if( size < valueSize )
    {
        std::ostringstream err;
        err << "Invalid copy of attribute \"" << name << "\" of size: " << valueSize << " to output of size: " << size;
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, err.str() );
    }
    T* dest = static_cast<T*>( p );
    algorithm::copy( values, dest );
}

inline static void copyAttribute( RTsize size, void* p, const char* name, const std::string& value )
{
    const size_t valueSize = value.length() + 1;
    if( size < valueSize )
    {
        std::ostringstream err;
        err << "Invalid copy of attribute \"" << name << "\" of size: " << valueSize << " to output of size: " << size;
        throw prodlib::IlwalidValue( RT_EXCEPTION_INFO, err.str() );
    }
    char* destination = static_cast<char*>( p );
    std::strncpy( destination, value.c_str(), valueSize );
    destination[valueSize - 1] = 0;
}


void APIDeviceAttributes::getAttribute( RTdeviceattribute attrib, RTsize size, void* p ) const
{
    switch( attrib )
    {
        case RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
            copyAttribute<int>( size, p, "RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK", maxThreadsPerBlock );
            return;

        case RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT:
            copyAttribute<int>( size, p, "RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT", multiprocessorCount );
            return;

        case RT_DEVICE_ATTRIBUTE_CLOCK_RATE:
            copyAttribute<int>( size, p, "RT_DEVICE_ATTRIBUTE_CLOCK_RATE", clockRate );
            return;

        case RT_DEVICE_ATTRIBUTE_EXELWTION_TIMEOUT_ENABLED:
            copyAttribute<int>( size, p, "RT_DEVICE_ATTRIBUTE_EXELWTION_TIMEOUT_ENABLED", exelwtionTimeoutEnabled );
            return;

        case RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT:
            copyAttribute<int>( size, p, "RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT", maxHardwareTextureCount );
            return;

        case RT_DEVICE_ATTRIBUTE_NAME:
            copyAttribute( size, p, "RT_DEVICE_ATTRIBUTE_NAME", name );
            return;

        case RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY:
            copyAttribute<int2>( size, p, "RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY", computeCapability );
            return;

        case RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY:
            copyAttribute<RTsize>( size, p, "RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY", totalMemory );
            return;

        case RT_DEVICE_ATTRIBUTE_TCC_DRIVER:
            copyAttribute<int>( size, p, "RT_DEVICE_ATTRIBUTE_TCC_DRIVER", tccDriver );
            return;

        case RT_DEVICE_ATTRIBUTE_LWDA_DEVICE_ORDINAL:
            copyAttribute<int>( size, p, "RT_DEVICE_ATTRIBUTE_LWDA_DEVICE_ORDINAL", lwdaDeviceOrdinal );
            return;

        case RT_DEVICE_ATTRIBUTE_PCI_BUS_ID:
            copyAttribute( size, p, "RT_DEVICE_ATTRIBUTE_PCI_BUS_ID", pciBusId );
            return;

        case RT_DEVICE_ATTRIBUTE_COMPATIBLE_DEVICES:
            copyAttribute( size, p, "RT_DEVICE_ATTRIBUTE_COMPATIBLE_DEVICES", compatibleDevices );
            return;

        case RT_DEVICE_ATTRIBUTE_RTCORE_VERSION:
            copyAttribute( size, p, "RT_DEVICE_ATTRIBUTE_RTCORE_VERSION", rtcoreVersion );
            return;
    }

    throw IlwalidValue( RT_EXCEPTION_INFO, "Invalid attribute\n" );
}
