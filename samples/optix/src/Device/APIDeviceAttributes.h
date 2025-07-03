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

#pragma once

#include <o6/optix.h>
#include <optixu/optixu_math.h>

#include <string>
#include <vector>

namespace optix {

struct APIDeviceAttributes
{
    // RT_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,      /*!< Max Threads per Block sizeof(int) */
    int maxThreadsPerBlock;
    // RT_DEVICE_ATTRIBUTE_CLOCK_RATE,                 /*!< Clock rate sizeof(int) */
    int clockRate;
    // RT_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,       /*!< Multiprocessor count sizeof(int) */
    int multiprocessorCount;
    // RT_DEVICE_ATTRIBUTE_EXELWTION_TIMEOUT_ENABLED,  /*!< Exelwtion timeout enabled sizeof(int) */
    int exelwtionTimeoutEnabled;
    // RT_DEVICE_ATTRIBUTE_MAX_HARDWARE_TEXTURE_COUNT, /*!< Hardware Texture count sizeof(int) */
    int maxHardwareTextureCount;
    // RT_DEVICE_ATTRIBUTE_NAME,                       /*!< Attribute Name */
    std::string name;
    // RT_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY,         /*!< Compute Capabilities sizeof(int2) */
    int2 computeCapability;
    // RT_DEVICE_ATTRIBUTE_TOTAL_MEMORY,               /*!< Total Memory sizeof(RTsize) */
    RTsize totalMemory;
    // RT_DEVICE_ATTRIBUTE_TCC_DRIVER,                 /*!< TCC driver sizeof(int) */
    int tccDriver;
    // RT_DEVICE_ATTRIBUTE_LWDA_DEVICE_ORDINAL,        /*!< LWCA device ordinal sizeof(int) */
    int lwdaDeviceOrdinal;
    // RT_DEVICE_ATTRIBUTE_PCI_BUS_ID,                 /*!< PCI Bus Id */
    std::string pciBusId;
    // RT_DEVICE_ATTRIBUTE_COMPATIBLE_DEVICES          /*!< Ordinals of compatible devices sizeof(int=N) + N*sizeof(int) */
    std::vector<int> compatibleDevices;
    // RT_DEVICE_ATTRIBUTE_RTCORE_VERSION              /*!< RT core (in the sense of TTU, not rtcore library) version (0 for no support, 10 for version 1.0) sizeof(int) */
    int rtcoreVersion;


    void getAttribute( RTdeviceattribute attrib, RTsize size, void* p ) const;
};


}  // namespace optix
