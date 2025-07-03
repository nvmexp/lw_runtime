
//
// Copyright (c) 2017, LWPU CORPORATION.  All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.
//

#pragma once

#ifdef _WIN32
#include <windows.h>
#endif

#include <cstdint>
#include <string>

namespace LwTelemetry {
namespace optix_lwbackend {

typedef int64_t DeviceNumAttribute;

typedef int64_t DeviceNumInfo;

typedef const std::string& DeviceStringAttribute;

typedef const std::string& BuildString;

typedef const std::string& UuidString;

typedef int64_t NumTimer;

typedef int64_t NumCounter;

#ifdef _WIN32

HRESULT Init();
HRESULT DeInit();

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
                                  const std::string&    userId );

HRESULT Send_ContextTearDown_Event( NumTimer           contextLifetime,
                                    NumCounter         countDenoiserLaunches,
                                    NumCounter         countKernelLaunches,
                                    NumTimer           sumDenoiserTimeSpent,
                                    UuidString         contextUUID,
                                    UuidString         clientUUID,
                                    const std::string& clientVer,
                                    const std::string& userId );
#endif
}
}
