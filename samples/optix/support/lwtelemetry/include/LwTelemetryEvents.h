
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

#include <windows.h>
#include <string>
#include <cstdint>

#define RAPIDJSON_HAS_STDSTRING 1

namespace LwTelemetry
{
  namespace gfe_lwbackend
  {
    
    typedef int64_t Counter;
    
    enum class ProductComponent : uint8_t
    {
      GFE_UI,
      OSC_UI,
      Downloader,
      LwBackend
    };
    
    typedef const std::string& URLString;
    
    typedef uint16_t HTTPStatusCode;
    
    typedef const std::string& Version;
    
    HRESULT Init();
    HRESULT DeInit();
    
    HRESULT Send_HTTPSuccess_Event(
      ProductComponent sourceComponent,
      URLString url,
      HTTPStatusCode httpStatus,
      Counter durationMs,
      Version GFEVersion,
      const std::string& clientVer,
      const std::string& userId
    );
  
  }
}

