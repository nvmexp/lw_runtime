
//
// Copyright (c) 2018, LWPU CORPORATION.  All rights reserved.
//
// LWPU CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from LWPU CORPORATION is strictly prohibited.

///////////////////////////////////////////////////////////////////////////////
//                                                                           
//        THIS FILE IS GENERATED FROM EVENT SCHEMA v0.7, DO NOT MODIFY IT
//        Please use https://sms.gfe.lwpu.com/ to update the schema
//        and generate the code.
//
///////////////////////////////////////////////////////////////////////////////

#include "AnselTelemetryGeneratedCode.h"
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <memory>
#pragma warning(push, 0)
#include <shlobj.h>
#pragma warning(pop)
#include <unordered_set>
#include <iomanip>
#include <sstream>
#include <lwSelwreLoadLibrary.h>

#define DO_EXPAND(VAL)  VAL ## 1
#define EXPAND(VAL)     DO_EXPAND(VAL)

#if !defined(RAPIDJSON_HAS_STDSTRING) || (EXPAND(RAPIDJSON_HAS_STDSTRING) != 11)
#error "Please define RAPIDJSON_HAS_STDSTRING=1 in your build configuration"
#endif

namespace LwTelemetry
{
  namespace Ansel
  {
    static HRESULT GetShellDirectory(int csidl, std::wstring& path)
    {
      wchar_t buf[MAX_PATH + 1] = { 0 };

      HRESULT hr = SHGetFolderPathW(
        NULL,
        csidl,
        NULL,
        SHGFP_TYPE_LWRRENT,
        buf);

      if(FAILED(hr))
      {
        return hr;
      }

      try
      {
        path = buf;
      }
      catch(const std::bad_alloc&)
      {
        return E_OUTOFMEMORY;
      }
        
      return S_OK;
    }
    
    static std::string FormatTimeUTC(uint64_t timeFileTime)
    {
      const char* errorValue = "1970-01-01T00:00:00.000Z";
      std::stringstream s;
      FILETIME fileTime{};
      fileTime.dwHighDateTime = static_cast<DWORD>(timeFileTime >> 32);
      fileTime.dwLowDateTime = static_cast<DWORD>(timeFileTime);
      SYSTEMTIME sysTime{};
      if(!FileTimeToSystemTime(&fileTime, &sysTime))
      {
        return errorValue;
      }

      char dateBuffer[64]{};
      if(!GetDateFormatA(LOCALE_ILWARIANT, 0, &sysTime, "yyyy-MM-dd", dateBuffer, _countof(dateBuffer)))
      {
        return errorValue;
      }
      s << dateBuffer << 'T';

      char timeBuffer[64]{};
      if(!GetTimeFormatA(LOCALE_ILWARIANT, TIME_FORCE24HOURFORMAT, &sysTime, "HH:mm:ss", timeBuffer, _countof(timeBuffer)))
      {
        return errorValue;
      }
      s << timeBuffer << '.' << std::setw(3) << std::setfill('0') << sysTime.wMilliseconds << "Z";

      return s.str();
    }
        
    HRESULT dllPath(std::wstring& path)
    {
      std::wstring bitness;
#if _WIN64
      bitness = L"64";
#else
      bitness = L"32";
#endif

      HRESULT hr = GetShellDirectory(CSIDL_PROGRAM_FILES, path);
      if(FAILED(hr))
      {
        return hr;
      }
      
      try
      {
        path += L"\\LWPU Corporation\\LwTelemetry\\LwTelemetryAPI";
        path += bitness;
        path += L".dll";
      }
      catch(const std::bad_alloc&)
      {
        return E_OUTOFMEMORY;
      }
      return S_OK;
    }

    const std::string gs_clientId = "44951455890061760";
    const std::string gs_schemaVer = "0.7";
    
    std::string StyleTransferStatusEnumToString(StyleTransferStatusEnum value)
    {
      switch(value)
      {
        case StyleTransferStatusEnum::FORWARD_SUCCESS:
          return "FORWARD_SUCCESS";
        case StyleTransferStatusEnum::FORWARD_FAILED:
          return "FORWARD_FAILED";
        case StyleTransferStatusEnum::FORWARD_FAILED_NOT_ENOUGH_VRAM:
          return "FORWARD_FAILED_NOT_ENOUGH_VRAM";
        case StyleTransferStatusEnum::FORWARD_HDR_SUCCESS:
          return "FORWARD_HDR_SUCCESS";
        case StyleTransferStatusEnum::FORWARD_HDR_FAILED:
          return "FORWARD_HDR_FAILED";
        case StyleTransferStatusEnum::FORWARD_HDR_FAILED_NOT_ENOUGH_VRAM:
          return "FORWARD_HDR_FAILED_NOT_ENOUGH_VRAM";
        case StyleTransferStatusEnum::HDR_COLWERT_FAILED:
          return "HDR_COLWERT_FAILED";
        case StyleTransferStatusEnum::HDR_STORAGE_COLWERT_FAILED:
          return "HDR_STORAGE_COLWERT_FAILED";
        case StyleTransferStatusEnum::INSTALLATION_FAILED:
          return "INSTALLATION_FAILED";
        case StyleTransferStatusEnum::DOWNLOADING_FAILED:
          return "DOWNLOADING_FAILED";
        case StyleTransferStatusEnum::EXCEPTION_OCLWRED:
          return "EXCEPTION_OCLWRED";
        case StyleTransferStatusEnum::OPERATION_FAILED:
          return "OPERATION_FAILED";
        case StyleTransferStatusEnum::OPERATION_TIMEOUT:
          return "OPERATION_TIMEOUT";
        case StyleTransferStatusEnum::STARTUP_FAILURE:
          return "STARTUP_FAILURE";
        case StyleTransferStatusEnum::INSTALLATION_SUCCESS:
          return "INSTALLATION_SUCCESS";
        case StyleTransferStatusEnum::DOWNLOAD_STARTUP_FAILED:
          return "DOWNLOAD_STARTUP_FAILED";
        case StyleTransferStatusEnum::COMPUTE_CAPABILITY_TO_OLD:
          return "COMPUTE_CAPABILITY_TO_OLD";
        case StyleTransferStatusEnum::LIBRESTYLE_NOT_FOUND:
          return "LIBRESTYLE_NOT_FOUND";
        case StyleTransferStatusEnum::MODEL_NOT_FOUND:
          return "MODEL_NOT_FOUND";
        case StyleTransferStatusEnum::NOT_ENOUGH_VRAM:
          return "NOT_ENOUGH_VRAM";
        case StyleTransferStatusEnum::INITIALIZATION_FAILED:
          return "INITIALIZATION_FAILED";
        case StyleTransferStatusEnum::LOADING_STYLE_FAILED:
          return "LOADING_STYLE_FAILED";
        case StyleTransferStatusEnum::DECLINED_INSTALLATION:
          return "DECLINED_INSTALLATION";
        case StyleTransferStatusEnum::ACCEPTED_INSTALLATION:
          return "ACCEPTED_INSTALLATION";
        default:
          throw std::ilwalid_argument("Invalid StyleTransferStatusEnum value passed: " + std::to_string(static_cast<int>(value)));
      }
    }
  
    std::string KindSliderEnumToString(KindSliderEnum value)
    {
      switch(value)
      {
        case KindSliderEnum::REGULAR:
          return "REGULAR";
        case KindSliderEnum::MONO_360:
          return "MONO_360";
        case KindSliderEnum::HIGHRES:
          return "HIGHRES";
        case KindSliderEnum::STEREO:
          return "STEREO";
        case KindSliderEnum::STEREO_360:
          return "STEREO_360";
        case KindSliderEnum::NONE:
          return "NONE";
        case KindSliderEnum::REGULAR_UI:
          return "REGULAR_UI";
        default:
          throw std::ilwalid_argument("Invalid KindSliderEnum value passed: " + std::to_string(static_cast<int>(value)));
      }
    }
  
    std::string SpecialEffectsModeEnumToString(SpecialEffectsModeEnum value)
    {
      switch(value)
      {
        case SpecialEffectsModeEnum::NONE:
          return "NONE";
        case SpecialEffectsModeEnum::YAML:
          return "YAML";
        default:
          throw std::ilwalid_argument("Invalid SpecialEffectsModeEnum value passed: " + std::to_string(static_cast<int>(value)));
      }
    }
  
    std::string UserConstantTypeEnumToString(UserConstantTypeEnum value)
    {
      switch(value)
      {
        case UserConstantTypeEnum::BOOL:
          return "BOOL";
        case UserConstantTypeEnum::INT:
          return "INT";
        case UserConstantTypeEnum::UINT:
          return "UINT";
        case UserConstantTypeEnum::FLOAT:
          return "FLOAT";
        default:
          throw std::ilwalid_argument("Invalid UserConstantTypeEnum value passed: " + std::to_string(static_cast<int>(value)));
      }
    }
  
    std::string UserConstantSliderStateEnumToString(UserConstantSliderStateEnum value)
    {
      switch(value)
      {
        case UserConstantSliderStateEnum::NOT_CREATED:
          return "NOT_CREATED";
        case UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED:
          return "CREATED_VISIBLE_ENABLED";
        default:
          throw std::ilwalid_argument("Invalid UserConstantSliderStateEnum value passed: " + std::to_string(static_cast<int>(value)));
      }
    }
  
    std::string AnselCaptureStateToString(AnselCaptureState value)
    {
      switch(value)
      {
        case AnselCaptureState::CAPTURE_STATE_NOT_STARTED:
          return "CAPTURE_STATE_NOT_STARTED";
        case AnselCaptureState::CAPTURE_STATE_STARTED:
          return "CAPTURE_STATE_STARTED";
        case AnselCaptureState::CAPTURE_STATE_ABORT:
          return "CAPTURE_STATE_ABORT";
        case AnselCaptureState::CAPTURE_STATE_REGULAR:
          return "CAPTURE_STATE_REGULAR";
        case AnselCaptureState::CAPTURE_STATE_REGULARSTEREO:
          return "CAPTURE_STATE_REGULARSTEREO";
        case AnselCaptureState::CAPTURE_STATE_HIGHRES:
          return "CAPTURE_STATE_HIGHRES";
        case AnselCaptureState::CAPTURE_STATE_360:
          return "CAPTURE_STATE_360";
        case AnselCaptureState::CAPTURE_STATE_360STEREO:
          return "CAPTURE_STATE_360STEREO";
        default:
          throw std::ilwalid_argument("Invalid AnselCaptureState value passed: " + std::to_string(static_cast<int>(value)));
      }
    }
  
    std::string ErrorTypeToString(ErrorType value)
    {
      switch(value)
      {
        case ErrorType::HANDLE_FAILURE_FATAL_ERROR:
          return "HANDLE_FAILURE_FATAL_ERROR";
        case ErrorType::EFFECT_COMPILATION_ERROR:
          return "EFFECT_COMPILATION_ERROR";
        case ErrorType::NON_FATAL_ERROR:
          return "NON_FATAL_ERROR";
        default:
          throw std::ilwalid_argument("Invalid ErrorType value passed: " + std::to_string(static_cast<int>(value)));
      }
    }
  
    std::string ColorRangeTypeToString(ColorRangeType value)
    {
      switch(value)
      {
        case ColorRangeType::RGB:
          return "RGB";
        case ColorRangeType::EXR:
          return "EXR";
        default:
          throw std::ilwalid_argument("Invalid ColorRangeType value passed: " + std::to_string(static_cast<int>(value)));
      }
    }
  
    std::string UIModeTypeToString(UIModeType value)
    {
      switch(value)
      {
        case UIModeType::STANDALONE_UI:
          return "STANDALONE_UI";
        case UIModeType::IPC_UI:
          return "IPC_UI";
        default:
          throw std::ilwalid_argument("Invalid UIModeType value passed: " + std::to_string(static_cast<int>(value)));
      }
    }
  
    std::string GamepadMappingTypeToString(GamepadMappingType value)
    {
      switch(value)
      {
        case GamepadMappingType::UNKNOWN:
          return "UNKNOWN";
        case GamepadMappingType::SHIELD:
          return "SHIELD";
        case GamepadMappingType::XBOX360:
          return "XBOX360";
        case GamepadMappingType::XBOXONE:
          return "XBOXONE";
        case GamepadMappingType::DUALSHOCK4:
          return "DUALSHOCK4";
        default:
          throw std::ilwalid_argument("Invalid GamepadMappingType value passed: " + std::to_string(static_cast<int>(value)));
      }
    }
  
    static auto gs_freeDll = [](HMODULE h) { if (h) { FreeLibrary(h); }};
    static std::unique_ptr<std::remove_pointer<HMODULE>::type, decltype(gs_freeDll)> gs_dll(nullptr, gs_freeDll);
    static HRESULT(__cdecl *pSendEvent)(const char*);
    static HRESULT(__cdecl *pSendFeedback)(const char*, const wchar_t*[], const char*[], const uint64_t);
    static HRESULT(__cdecl *pDeInit)();
    static HRESULT(__cdecl *pGetUserTelemetryConsent)(const char* userId, uint32_t* consentFlags);
    static HRESULT(__cdecl *pGetDeviceTelemetryConsent)(const char* clientId, uint32_t* consentFlags);
    static HRESULT(__cdecl *pSetUserTelemetryConsent)(const char* userId, uint32_t consentFlags);
    static HRESULT(__cdecl *pSetDeviceTelemetryConsent)(const char* clientId, uint32_t consentFlags);


    HRESULT Init()
    {
      if (!gs_dll)
      {
        std::wstring path;
        HRESULT hr = dllPath(path);
        if(FAILED(hr))
        {
          return hr;
        }
        gs_dll.reset(lwLoadSignedLibraryW(path.c_str(), TRUE));
        if (!gs_dll)
        {
          return HRESULT_FROM_WIN32(GetLastError());
        }
      }
      
      auto pInit = reinterpret_cast<HRESULT(__cdecl *)()>(GetProcAddress(gs_dll.get(), "Init"));     
      pDeInit = reinterpret_cast<decltype(pDeInit)>(GetProcAddress(gs_dll.get(), "DeInit"));
      pGetUserTelemetryConsent = reinterpret_cast<decltype(pGetUserTelemetryConsent)>(GetProcAddress(gs_dll.get(), "GetUserTelemetryConsent"));
      pGetDeviceTelemetryConsent = reinterpret_cast<decltype(pGetDeviceTelemetryConsent)>(GetProcAddress(gs_dll.get(), "GetDeviceTelemetryConsent"));
      pSetUserTelemetryConsent = reinterpret_cast<decltype(pSetUserTelemetryConsent)>(GetProcAddress(gs_dll.get(), "SetUserTelemetryConsent"));
      pSetDeviceTelemetryConsent = reinterpret_cast<decltype(pSetDeviceTelemetryConsent)>(GetProcAddress(gs_dll.get(), "SetDeviceTelemetryConsent"));
    
      pSendEvent = reinterpret_cast<decltype(pSendEvent)>(GetProcAddress(gs_dll.get(), "LwTelemetrySendEvent"));
      
      if(!pInit || !pDeInit || (!pSendEvent && !pSendFeedback) || !pGetUserTelemetryConsent || !pGetDeviceTelemetryConsent ||
        !pSetUserTelemetryConsent || !pSetDeviceTelemetryConsent)
      {
        const auto le = GetLastError();
        pInit = nullptr;
        pDeInit = nullptr;
        pSendEvent = nullptr;
        pSendFeedback = nullptr;
        pGetUserTelemetryConsent = nullptr;
        pGetDeviceTelemetryConsent = nullptr;
        pSetUserTelemetryConsent = nullptr;
        pSetDeviceTelemetryConsent = nullptr;
        gs_dll.reset();
        return HRESULT_FROM_WIN32(le);
      }

      HRESULT hr = pInit();
      if(FAILED(hr))
      {
        pInit = nullptr;
        pDeInit = nullptr;
        pSendEvent = nullptr;
        pSendFeedback = nullptr;
        pGetUserTelemetryConsent = nullptr;
        pGetDeviceTelemetryConsent = nullptr;
        pSetUserTelemetryConsent = nullptr;
        pSetDeviceTelemetryConsent = nullptr;
        gs_dll.reset();
      }
      return hr;
    }
    
    HRESULT DeInit()
    {
      if (!gs_dll)
      {
        return E_NOT_VALID_STATE;
      }

      HRESULT hr = pDeInit();
      
      gs_dll.reset();
      return hr;
    }
    
    static HRESULT HasConsentForTelemetryCategory(const std::string& userId, uint32_t consentCategory)
    {
      if (!gs_dll)
      {
        return E_NOT_VALID_STATE;
      }
      uint32_t consentFlags = 0;
      HRESULT hr = S_OK;
      if (!userId.empty() && userId != "undefined")
      {
        hr = pGetUserTelemetryConsent(userId.c_str(), &consentFlags);
      }
      else
      {
        hr = pGetDeviceTelemetryConsent(gs_clientId.c_str(), &consentFlags);
      }
      if(FAILED(hr))
      {
        return hr;
      }
      return (consentFlags & consentCategory) != 0 ? S_OK : S_FALSE;
    }

    HRESULT SetUserConsent(const std::string& userId, uint32_t consentFlags)
    {
      if (!gs_dll)
      {
        return E_NOT_VALID_STATE;
      }
      return pSetUserTelemetryConsent(userId.c_str(), consentFlags);
    }
    
    HRESULT SetDeviceConsent(uint32_t consentFlags)
    {
      return SetDeviceConsent(gs_clientId, consentFlags);
    }
    
    HRESULT SetDeviceConsent(const std::string& clientId, uint32_t consentFlags)
    {
      if (!gs_dll)
      {
        return E_NOT_VALID_STATE;
      }
      return pSetDeviceTelemetryConsent(clientId.c_str(), consentFlags);
    }

    HRESULT HasFunctionalConsent(const std::string& userId)
    {
      const uint32_t FunctionalConsentCategory = 1;
      return HasConsentForTelemetryCategory(userId, FunctionalConsentCategory);
    }

    HRESULT HasTechnicalConsent(const std::string& userId)
    {
      const uint32_t TechnicalConsentCategory = 2;
      return HasConsentForTelemetryCategory(userId, TechnicalConsentCategory);
    }

    HRESULT HasBehavioralConsent(const std::string& userId)
    {
      const uint32_t BehavioralConsentCategory = 4;
      return HasConsentForTelemetryCategory(userId, BehavioralConsentCategory);
    }

    // Throwing version of rapidjson::CrtAllocator
    class ThrowingCrtAllocator
    {
    public:
      static const bool kNeedFree = true;
      void* Malloc(size_t size)
      {
        auto ptr = m_allocator.Malloc(size);
        if(!ptr)
        {
          throw std::bad_alloc();
        }
        return ptr;
      }

      void* Realloc(void* originalPtr, size_t originalSize, size_t newSize)
      {
        auto ptr = m_allocator.Realloc(originalPtr, originalSize, newSize);
        if(!ptr)
        {
          throw std::bad_alloc();
        }
        return ptr;
      }

      static void Free(void *ptr)
      {
        m_allocator.Free(ptr);
      }

    private:
      // CrtAllocator is stateless and thread-safe
      static rapidjson::CrtAllocator m_allocator;
    };

    rapidjson::CrtAllocator ThrowingCrtAllocator::m_allocator;

    using RapidjsonDolwment = rapidjson::GenericDolwment<
      rapidjson::UTF8<>,
      rapidjson::MemoryPoolAllocator<ThrowingCrtAllocator>,
      ThrowingCrtAllocator>;

    using Rapidjsolwalue = rapidjson::GenericValue<rapidjson::UTF8<>, rapidjson::MemoryPoolAllocator<ThrowingCrtAllocator>>;

    
    HRESULT Send_StyleTransferDownloadStarted_Event(
      StringType schemaDefinedArg_url,
      StringType schemaDefinedArg_version,
      UintType schemaDefinedArg_computeCapMajor,
      UintType schemaDefinedArg_computeCapMinor,
      const std::string& clientVersion,
      const std::string& userId,
      const uint64_t* timestampFileTimeFormatOptional
      )
    {
      try
      {
        if (!gs_dll)
        {
          return E_NOT_VALID_STATE;
        }

        RapidjsonDolwment d;
        d.SetObject();
        auto& a = d.GetAllocator();

        d.AddMember("clientId", gs_clientId, a);
        d.AddMember("clientVer", clientVersion, a);
        d.AddMember("userId", userId, a);
        
        
        if (timestampFileTimeFormatOptional)
        {
          d.AddMember("ts", FormatTimeUTC(*timestampFileTimeFormatOptional), a);
        }
        
        d.AddMember("eventSchemaVer", gs_schemaVer, a);
        
        d.AddMember("event", Rapidjsolwalue(rapidjson::kObjectType), a);
        Rapidjsolwalue& v = d["event"];
        
        v.AddMember("name", "StyleTransferDownloadStarted", a);
        
        v.AddMember("GDPRCategory", "technical", a);
          
        
        v.AddMember("parameters", Rapidjsolwalue(rapidjson::kObjectType), a);
        
        v["parameters"].AddMember("url", schemaDefinedArg_url, a);
            
        v["parameters"].AddMember("version", schemaDefinedArg_version, a);
            
        v["parameters"].AddMember("computeCapMajor", schemaDefinedArg_computeCapMajor, a);
            
        v["parameters"].AddMember("computeCapMinor", schemaDefinedArg_computeCapMinor, a);
            
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        d.Accept(writer);
        
        
        return pSendEvent(buffer.GetString());
        
      }
      catch(const std::bad_alloc&)
      {
        return E_OUTOFMEMORY;
      }
      catch(const std::ilwalid_argument&)
      {
        return E_ILWALIDARG;
      }
      catch(const std::exception&)
      {
        return E_FAIL;
      }
    }

    HRESULT Send_StyleTransferDownloadFinished_Event(
      UintType schemaDefinedArg_secondsSpent,
      IntType schemaDefinedArg_status,
      const std::string& clientVersion,
      const std::string& userId,
      const uint64_t* timestampFileTimeFormatOptional
      )
    {
      try
      {
        if (!gs_dll)
        {
          return E_NOT_VALID_STATE;
        }

        RapidjsonDolwment d;
        d.SetObject();
        auto& a = d.GetAllocator();

        d.AddMember("clientId", gs_clientId, a);
        d.AddMember("clientVer", clientVersion, a);
        d.AddMember("userId", userId, a);
        
        
        if (timestampFileTimeFormatOptional)
        {
          d.AddMember("ts", FormatTimeUTC(*timestampFileTimeFormatOptional), a);
        }
        
        d.AddMember("eventSchemaVer", gs_schemaVer, a);
        
        d.AddMember("event", Rapidjsolwalue(rapidjson::kObjectType), a);
        Rapidjsolwalue& v = d["event"];
        
        v.AddMember("name", "StyleTransferDownloadFinished", a);
        
        v.AddMember("GDPRCategory", "technical", a);
          
        
        v.AddMember("parameters", Rapidjsolwalue(rapidjson::kObjectType), a);
        
        v["parameters"].AddMember("secondsSpent", schemaDefinedArg_secondsSpent, a);
            
        v["parameters"].AddMember("status", schemaDefinedArg_status, a);
            
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        d.Accept(writer);
        
        
        return pSendEvent(buffer.GetString());
        
      }
      catch(const std::bad_alloc&)
      {
        return E_OUTOFMEMORY;
      }
      catch(const std::ilwalid_argument&)
      {
        return E_ILWALIDARG;
      }
      catch(const std::exception&)
      {
        return E_FAIL;
      }
    }

    HRESULT Send_StyleTransferStatus_Event(
      StyleTransferStatusEnum schemaDefinedArg_status,
      StringType schemaDefinedArg_comment,
      const std::string& clientVersion,
      const std::string& userId,
      const uint64_t* timestampFileTimeFormatOptional
      )
    {
      try
      {
        if (!gs_dll)
        {
          return E_NOT_VALID_STATE;
        }

        RapidjsonDolwment d;
        d.SetObject();
        auto& a = d.GetAllocator();

        d.AddMember("clientId", gs_clientId, a);
        d.AddMember("clientVer", clientVersion, a);
        d.AddMember("userId", userId, a);
        
        
        if (timestampFileTimeFormatOptional)
        {
          d.AddMember("ts", FormatTimeUTC(*timestampFileTimeFormatOptional), a);
        }
        
        d.AddMember("eventSchemaVer", gs_schemaVer, a);
        
        d.AddMember("event", Rapidjsolwalue(rapidjson::kObjectType), a);
        Rapidjsolwalue& v = d["event"];
        
        v.AddMember("name", "StyleTransferStatus", a);
        
        v.AddMember("GDPRCategory", "technical", a);
          
        
        v.AddMember("parameters", Rapidjsolwalue(rapidjson::kObjectType), a);
        
        v["parameters"].AddMember("status", StyleTransferStatusEnumToString(schemaDefinedArg_status), a);
            
        v["parameters"].AddMember("comment", schemaDefinedArg_comment, a);
            
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        d.Accept(writer);
        
        
        return pSendEvent(buffer.GetString());
        
      }
      catch(const std::bad_alloc&)
      {
        return E_OUTOFMEMORY;
      }
      catch(const std::ilwalid_argument&)
      {
        return E_ILWALIDARG;
      }
      catch(const std::exception&)
      {
        return E_FAIL;
      }
    }

    HRESULT Send_CaptureStarted_Event(
      StringType schemaDefinedArg_appExeName,
      StringType schemaDefinedArg_drsProfileName,
      StringType schemaDefinedArg_drsAppName,
      UintType schemaDefinedArg_screenResolutionX,
      UintType schemaDefinedArg_screenResolutionY,
      DxgiFormat schemaDefinedArg_colorBufferFormat,
      DxgiFormat schemaDefinedArg_depthBufferFormat,
      KindSliderEnum schemaDefinedArg_kindOfShot,
      UintType schemaDefinedArg_highresMultiplier,
      FloatType schemaDefinedArg_quality360Fov,
      FloatType schemaDefinedArg_fov,
      FloatType schemaDefinedArg_roll,
      FloatType schemaDefinedArg_lwrrentCameraPosX,
      FloatType schemaDefinedArg_lwrrentCameraPosY,
      FloatType schemaDefinedArg_lwrrentCameraPosZ,
      FloatType schemaDefinedArg_lwrrentCameraRotX,
      FloatType schemaDefinedArg_lwrrentCameraRotY,
      FloatType schemaDefinedArg_lwrrentCameraRotZ,
      FloatType schemaDefinedArg_lwrrentCameraRotW,
      FloatType schemaDefinedArg_originalCameraPosX,
      FloatType schemaDefinedArg_originalCameraPosY,
      FloatType schemaDefinedArg_originalCameraPosZ,
      FloatType schemaDefinedArg_originalCameraRotX,
      FloatType schemaDefinedArg_originalCameraRotY,
      FloatType schemaDefinedArg_originalCameraRotZ,
      FloatType schemaDefinedArg_originalCameraRotW,
      SpecialEffectsModeEnum schemaDefinedArg_specialEffectsMode,
      StringType schemaDefinedArg_effectName,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant0_sliderState,
      StringType schemaDefinedArg_userConstant0_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant0_type,
      StringType schemaDefinedArg_userConstant0_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant1_sliderState,
      StringType schemaDefinedArg_userConstant1_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant1_type,
      StringType schemaDefinedArg_userConstant1_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant2_sliderState,
      StringType schemaDefinedArg_userConstant2_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant2_type,
      StringType schemaDefinedArg_userConstant2_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant3_sliderState,
      StringType schemaDefinedArg_userConstant3_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant3_type,
      StringType schemaDefinedArg_userConstant3_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant4_sliderState,
      StringType schemaDefinedArg_userConstant4_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant4_type,
      StringType schemaDefinedArg_userConstant4_value,
      PackedAllUserConstants schemaDefinedArg_allUserConstants,
      UIModeType schemaDefinedArg_uiMode,
      ColorRangeType schemaDefinedArg_colorRange,
      Int64Type schemaDefinedArg_quality360resolution,
      GamepadMappingType schemaDefinedArg_gamepadMapping,
      UintType schemaDefinedArg_gamepadProductId,
      UintType schemaDefinedArg_gamepadVendorId,
      UintType schemaDefinedArg_gamepadVersionNumber,
      BoolType schemaDefinedArg_gamepadUsedForCameraOperationDuringThisSession,
      BoolType schemaDefinedArg_gamepadUsedForUIInteractionDuringThisSession,
      DoubleType schemaDefinedArg_timeElapsedSinceSessionStarted,
      BoolType schemaDefinedArg_highresEnhancementEnabled,
      UintType schemaDefinedArg_anselSDKMajor,
      UintType schemaDefinedArg_anselSDKMinor,
      UintType schemaDefinedArg_anselSDKCommit,
      const std::string& clientVersion,
      const std::string& userId,
      const uint64_t* timestampFileTimeFormatOptional
      )
    {
      try
      {
        if (!gs_dll)
        {
          return E_NOT_VALID_STATE;
        }

        RapidjsonDolwment d;
        d.SetObject();
        auto& a = d.GetAllocator();

        d.AddMember("clientId", gs_clientId, a);
        d.AddMember("clientVer", clientVersion, a);
        d.AddMember("userId", userId, a);
        
        
        if (timestampFileTimeFormatOptional)
        {
          d.AddMember("ts", FormatTimeUTC(*timestampFileTimeFormatOptional), a);
        }
        
        d.AddMember("eventSchemaVer", gs_schemaVer, a);
        
        d.AddMember("event", Rapidjsolwalue(rapidjson::kObjectType), a);
        Rapidjsolwalue& v = d["event"];
        
        v.AddMember("name", "CaptureStarted", a);
        
        v.AddMember("GDPRCategory", "technical", a);
          
        
        v.AddMember("parameters", Rapidjsolwalue(rapidjson::kObjectType), a);
        
        v["parameters"].AddMember("appExeName", schemaDefinedArg_appExeName, a);
            
        v["parameters"].AddMember("drsProfileName", schemaDefinedArg_drsProfileName, a);
            
        v["parameters"].AddMember("drsAppName", schemaDefinedArg_drsAppName, a);
            
        v["parameters"].AddMember("screenResolutionX", schemaDefinedArg_screenResolutionX, a);
            
        v["parameters"].AddMember("screenResolutionY", schemaDefinedArg_screenResolutionY, a);
            
        v["parameters"].AddMember("colorBufferFormat", schemaDefinedArg_colorBufferFormat, a);
            
        v["parameters"].AddMember("depthBufferFormat", schemaDefinedArg_depthBufferFormat, a);
            
        v["parameters"].AddMember("kindOfShot", KindSliderEnumToString(schemaDefinedArg_kindOfShot), a);
            
        v["parameters"].AddMember("highresMultiplier", schemaDefinedArg_highresMultiplier, a);
            
        v["parameters"].AddMember("quality360Fov", schemaDefinedArg_quality360Fov, a);
            
        v["parameters"].AddMember("fov", schemaDefinedArg_fov, a);
            
        v["parameters"].AddMember("roll", schemaDefinedArg_roll, a);
            
        v["parameters"].AddMember("lwrrentCameraPosX", schemaDefinedArg_lwrrentCameraPosX, a);
            
        v["parameters"].AddMember("lwrrentCameraPosY", schemaDefinedArg_lwrrentCameraPosY, a);
            
        v["parameters"].AddMember("lwrrentCameraPosZ", schemaDefinedArg_lwrrentCameraPosZ, a);
            
        v["parameters"].AddMember("lwrrentCameraRotX", schemaDefinedArg_lwrrentCameraRotX, a);
            
        v["parameters"].AddMember("lwrrentCameraRotY", schemaDefinedArg_lwrrentCameraRotY, a);
            
        v["parameters"].AddMember("lwrrentCameraRotZ", schemaDefinedArg_lwrrentCameraRotZ, a);
            
        v["parameters"].AddMember("lwrrentCameraRotW", schemaDefinedArg_lwrrentCameraRotW, a);
            
        v["parameters"].AddMember("originalCameraPosX", schemaDefinedArg_originalCameraPosX, a);
            
        v["parameters"].AddMember("originalCameraPosY", schemaDefinedArg_originalCameraPosY, a);
            
        v["parameters"].AddMember("originalCameraPosZ", schemaDefinedArg_originalCameraPosZ, a);
            
        v["parameters"].AddMember("originalCameraRotX", schemaDefinedArg_originalCameraRotX, a);
            
        v["parameters"].AddMember("originalCameraRotY", schemaDefinedArg_originalCameraRotY, a);
            
        v["parameters"].AddMember("originalCameraRotZ", schemaDefinedArg_originalCameraRotZ, a);
            
        v["parameters"].AddMember("originalCameraRotW", schemaDefinedArg_originalCameraRotW, a);
            
        v["parameters"].AddMember("specialEffectsMode", SpecialEffectsModeEnumToString(schemaDefinedArg_specialEffectsMode), a);
            
        v["parameters"].AddMember("effectName", schemaDefinedArg_effectName, a);
            
        v["parameters"].AddMember("userConstant0_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant0_sliderState), a);
            
        v["parameters"].AddMember("userConstant0_name", schemaDefinedArg_userConstant0_name, a);
            
        v["parameters"].AddMember("userConstant0_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant0_type), a);
            
        v["parameters"].AddMember("userConstant0_value", schemaDefinedArg_userConstant0_value, a);
            
        v["parameters"].AddMember("userConstant1_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant1_sliderState), a);
            
        v["parameters"].AddMember("userConstant1_name", schemaDefinedArg_userConstant1_name, a);
            
        v["parameters"].AddMember("userConstant1_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant1_type), a);
            
        v["parameters"].AddMember("userConstant1_value", schemaDefinedArg_userConstant1_value, a);
            
        v["parameters"].AddMember("userConstant2_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant2_sliderState), a);
            
        v["parameters"].AddMember("userConstant2_name", schemaDefinedArg_userConstant2_name, a);
            
        v["parameters"].AddMember("userConstant2_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant2_type), a);
            
        v["parameters"].AddMember("userConstant2_value", schemaDefinedArg_userConstant2_value, a);
            
        v["parameters"].AddMember("userConstant3_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant3_sliderState), a);
            
        v["parameters"].AddMember("userConstant3_name", schemaDefinedArg_userConstant3_name, a);
            
        v["parameters"].AddMember("userConstant3_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant3_type), a);
            
        v["parameters"].AddMember("userConstant3_value", schemaDefinedArg_userConstant3_value, a);
            
        v["parameters"].AddMember("userConstant4_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant4_sliderState), a);
            
        v["parameters"].AddMember("userConstant4_name", schemaDefinedArg_userConstant4_name, a);
            
        v["parameters"].AddMember("userConstant4_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant4_type), a);
            
        v["parameters"].AddMember("userConstant4_value", schemaDefinedArg_userConstant4_value, a);
            
        v["parameters"].AddMember("allUserConstants", schemaDefinedArg_allUserConstants, a);
            
        v["parameters"].AddMember("uiMode", UIModeTypeToString(schemaDefinedArg_uiMode), a);
            
        v["parameters"].AddMember("colorRange", ColorRangeTypeToString(schemaDefinedArg_colorRange), a);
            
        v["parameters"].AddMember("quality360resolution", schemaDefinedArg_quality360resolution, a);
            
        v["parameters"].AddMember("gamepadMapping", GamepadMappingTypeToString(schemaDefinedArg_gamepadMapping), a);
            
        v["parameters"].AddMember("gamepadProductId", schemaDefinedArg_gamepadProductId, a);
            
        v["parameters"].AddMember("gamepadVendorId", schemaDefinedArg_gamepadVendorId, a);
            
        v["parameters"].AddMember("gamepadVersionNumber", schemaDefinedArg_gamepadVersionNumber, a);
            
        v["parameters"].AddMember("gamepadUsedForCameraOperationDuringThisSession", schemaDefinedArg_gamepadUsedForCameraOperationDuringThisSession, a);
            
        v["parameters"].AddMember("gamepadUsedForUIInteractionDuringThisSession", schemaDefinedArg_gamepadUsedForUIInteractionDuringThisSession, a);
            
        v["parameters"].AddMember("timeElapsedSinceSessionStarted", schemaDefinedArg_timeElapsedSinceSessionStarted, a);
            
        v["parameters"].AddMember("highresEnhancementEnabled", schemaDefinedArg_highresEnhancementEnabled, a);
            
        v["parameters"].AddMember("anselSDKMajor", schemaDefinedArg_anselSDKMajor, a);
            
        v["parameters"].AddMember("anselSDKMinor", schemaDefinedArg_anselSDKMinor, a);
            
        v["parameters"].AddMember("anselSDKCommit", schemaDefinedArg_anselSDKCommit, a);
            
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        d.Accept(writer);
        
        
        return pSendEvent(buffer.GetString());
        
      }
      catch(const std::bad_alloc&)
      {
        return E_OUTOFMEMORY;
      }
      catch(const std::ilwalid_argument&)
      {
        return E_ILWALIDARG;
      }
      catch(const std::exception&)
      {
        return E_FAIL;
      }
    }

    HRESULT Send_CaptureAborted_Event(
      StringType schemaDefinedArg_appExeName,
      StringType schemaDefinedArg_drsProfileName,
      StringType schemaDefinedArg_drsAppName,
      UintType schemaDefinedArg_screenResolutionX,
      UintType schemaDefinedArg_screenResolutionY,
      DxgiFormat schemaDefinedArg_colorBufferFormat,
      DxgiFormat schemaDefinedArg_depthBufferFormat,
      KindSliderEnum schemaDefinedArg_kindOfShot,
      UintType schemaDefinedArg_highresMultiplier,
      FloatType schemaDefinedArg_quality360Fov,
      FloatType schemaDefinedArg_fov,
      FloatType schemaDefinedArg_roll,
      FloatType schemaDefinedArg_lwrrentCameraPosX,
      FloatType schemaDefinedArg_lwrrentCameraPosY,
      FloatType schemaDefinedArg_lwrrentCameraPosZ,
      FloatType schemaDefinedArg_lwrrentCameraRotX,
      FloatType schemaDefinedArg_lwrrentCameraRotY,
      FloatType schemaDefinedArg_lwrrentCameraRotZ,
      FloatType schemaDefinedArg_lwrrentCameraRotW,
      FloatType schemaDefinedArg_originalCameraPosX,
      FloatType schemaDefinedArg_originalCameraPosY,
      FloatType schemaDefinedArg_originalCameraPosZ,
      FloatType schemaDefinedArg_originalCameraRotX,
      FloatType schemaDefinedArg_originalCameraRotY,
      FloatType schemaDefinedArg_originalCameraRotZ,
      FloatType schemaDefinedArg_originalCameraRotW,
      SpecialEffectsModeEnum schemaDefinedArg_specialEffectsMode,
      StringType schemaDefinedArg_effectName,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant0_sliderState,
      StringType schemaDefinedArg_userConstant0_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant0_type,
      StringType schemaDefinedArg_userConstant0_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant1_sliderState,
      StringType schemaDefinedArg_userConstant1_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant1_type,
      StringType schemaDefinedArg_userConstant1_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant2_sliderState,
      StringType schemaDefinedArg_userConstant2_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant2_type,
      StringType schemaDefinedArg_userConstant2_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant3_sliderState,
      StringType schemaDefinedArg_userConstant3_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant3_type,
      StringType schemaDefinedArg_userConstant3_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant4_sliderState,
      StringType schemaDefinedArg_userConstant4_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant4_type,
      StringType schemaDefinedArg_userConstant4_value,
      PackedAllUserConstants schemaDefinedArg_allUserConstants,
      UIModeType schemaDefinedArg_uiMode,
      ColorRangeType schemaDefinedArg_colorRange,
      Int64Type schemaDefinedArg_quality360resolution,
      DoubleType schemaDefinedArg_timeElapsedSinceCaptureStarted,
      BoolType schemaDefinedArg_highresEnhancementEnabled,
      const std::string& clientVersion,
      const std::string& userId,
      const uint64_t* timestampFileTimeFormatOptional
      )
    {
      try
      {
        if (!gs_dll)
        {
          return E_NOT_VALID_STATE;
        }

        RapidjsonDolwment d;
        d.SetObject();
        auto& a = d.GetAllocator();

        d.AddMember("clientId", gs_clientId, a);
        d.AddMember("clientVer", clientVersion, a);
        d.AddMember("userId", userId, a);
        
        
        if (timestampFileTimeFormatOptional)
        {
          d.AddMember("ts", FormatTimeUTC(*timestampFileTimeFormatOptional), a);
        }
        
        d.AddMember("eventSchemaVer", gs_schemaVer, a);
        
        d.AddMember("event", Rapidjsolwalue(rapidjson::kObjectType), a);
        Rapidjsolwalue& v = d["event"];
        
        v.AddMember("name", "CaptureAborted", a);
        
        v.AddMember("GDPRCategory", "technical", a);
          
        
        v.AddMember("parameters", Rapidjsolwalue(rapidjson::kObjectType), a);
        
        v["parameters"].AddMember("appExeName", schemaDefinedArg_appExeName, a);
            
        v["parameters"].AddMember("drsProfileName", schemaDefinedArg_drsProfileName, a);
            
        v["parameters"].AddMember("drsAppName", schemaDefinedArg_drsAppName, a);
            
        v["parameters"].AddMember("screenResolutionX", schemaDefinedArg_screenResolutionX, a);
            
        v["parameters"].AddMember("screenResolutionY", schemaDefinedArg_screenResolutionY, a);
            
        v["parameters"].AddMember("colorBufferFormat", schemaDefinedArg_colorBufferFormat, a);
            
        v["parameters"].AddMember("depthBufferFormat", schemaDefinedArg_depthBufferFormat, a);
            
        v["parameters"].AddMember("kindOfShot", KindSliderEnumToString(schemaDefinedArg_kindOfShot), a);
            
        v["parameters"].AddMember("highresMultiplier", schemaDefinedArg_highresMultiplier, a);
            
        v["parameters"].AddMember("quality360Fov", schemaDefinedArg_quality360Fov, a);
            
        v["parameters"].AddMember("fov", schemaDefinedArg_fov, a);
            
        v["parameters"].AddMember("roll", schemaDefinedArg_roll, a);
            
        v["parameters"].AddMember("lwrrentCameraPosX", schemaDefinedArg_lwrrentCameraPosX, a);
            
        v["parameters"].AddMember("lwrrentCameraPosY", schemaDefinedArg_lwrrentCameraPosY, a);
            
        v["parameters"].AddMember("lwrrentCameraPosZ", schemaDefinedArg_lwrrentCameraPosZ, a);
            
        v["parameters"].AddMember("lwrrentCameraRotX", schemaDefinedArg_lwrrentCameraRotX, a);
            
        v["parameters"].AddMember("lwrrentCameraRotY", schemaDefinedArg_lwrrentCameraRotY, a);
            
        v["parameters"].AddMember("lwrrentCameraRotZ", schemaDefinedArg_lwrrentCameraRotZ, a);
            
        v["parameters"].AddMember("lwrrentCameraRotW", schemaDefinedArg_lwrrentCameraRotW, a);
            
        v["parameters"].AddMember("originalCameraPosX", schemaDefinedArg_originalCameraPosX, a);
            
        v["parameters"].AddMember("originalCameraPosY", schemaDefinedArg_originalCameraPosY, a);
            
        v["parameters"].AddMember("originalCameraPosZ", schemaDefinedArg_originalCameraPosZ, a);
            
        v["parameters"].AddMember("originalCameraRotX", schemaDefinedArg_originalCameraRotX, a);
            
        v["parameters"].AddMember("originalCameraRotY", schemaDefinedArg_originalCameraRotY, a);
            
        v["parameters"].AddMember("originalCameraRotZ", schemaDefinedArg_originalCameraRotZ, a);
            
        v["parameters"].AddMember("originalCameraRotW", schemaDefinedArg_originalCameraRotW, a);
            
        v["parameters"].AddMember("specialEffectsMode", SpecialEffectsModeEnumToString(schemaDefinedArg_specialEffectsMode), a);
            
        v["parameters"].AddMember("effectName", schemaDefinedArg_effectName, a);
            
        v["parameters"].AddMember("userConstant0_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant0_sliderState), a);
            
        v["parameters"].AddMember("userConstant0_name", schemaDefinedArg_userConstant0_name, a);
            
        v["parameters"].AddMember("userConstant0_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant0_type), a);
            
        v["parameters"].AddMember("userConstant0_value", schemaDefinedArg_userConstant0_value, a);
            
        v["parameters"].AddMember("userConstant1_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant1_sliderState), a);
            
        v["parameters"].AddMember("userConstant1_name", schemaDefinedArg_userConstant1_name, a);
            
        v["parameters"].AddMember("userConstant1_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant1_type), a);
            
        v["parameters"].AddMember("userConstant1_value", schemaDefinedArg_userConstant1_value, a);
            
        v["parameters"].AddMember("userConstant2_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant2_sliderState), a);
            
        v["parameters"].AddMember("userConstant2_name", schemaDefinedArg_userConstant2_name, a);
            
        v["parameters"].AddMember("userConstant2_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant2_type), a);
            
        v["parameters"].AddMember("userConstant2_value", schemaDefinedArg_userConstant2_value, a);
            
        v["parameters"].AddMember("userConstant3_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant3_sliderState), a);
            
        v["parameters"].AddMember("userConstant3_name", schemaDefinedArg_userConstant3_name, a);
            
        v["parameters"].AddMember("userConstant3_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant3_type), a);
            
        v["parameters"].AddMember("userConstant3_value", schemaDefinedArg_userConstant3_value, a);
            
        v["parameters"].AddMember("userConstant4_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant4_sliderState), a);
            
        v["parameters"].AddMember("userConstant4_name", schemaDefinedArg_userConstant4_name, a);
            
        v["parameters"].AddMember("userConstant4_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant4_type), a);
            
        v["parameters"].AddMember("userConstant4_value", schemaDefinedArg_userConstant4_value, a);
            
        v["parameters"].AddMember("allUserConstants", schemaDefinedArg_allUserConstants, a);
            
        v["parameters"].AddMember("uiMode", UIModeTypeToString(schemaDefinedArg_uiMode), a);
            
        v["parameters"].AddMember("colorRange", ColorRangeTypeToString(schemaDefinedArg_colorRange), a);
            
        v["parameters"].AddMember("quality360resolution", schemaDefinedArg_quality360resolution, a);
            
        v["parameters"].AddMember("timeElapsedSinceCaptureStarted", schemaDefinedArg_timeElapsedSinceCaptureStarted, a);
            
        v["parameters"].AddMember("highresEnhancementEnabled", schemaDefinedArg_highresEnhancementEnabled, a);
            
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        d.Accept(writer);
        
        
        return pSendEvent(buffer.GetString());
        
      }
      catch(const std::bad_alloc&)
      {
        return E_OUTOFMEMORY;
      }
      catch(const std::ilwalid_argument&)
      {
        return E_ILWALIDARG;
      }
      catch(const std::exception&)
      {
        return E_FAIL;
      }
    }

    HRESULT Send_AnselUIClosed_Event(
      StringType schemaDefinedArg_appExeName,
      StringType schemaDefinedArg_drsProfileName,
      StringType schemaDefinedArg_drsAppName,
      UintType schemaDefinedArg_screenResolutionX,
      UintType schemaDefinedArg_screenResolutionY,
      DxgiFormat schemaDefinedArg_colorBufferFormat,
      DxgiFormat schemaDefinedArg_depthBufferFormat,
      KindSliderEnum schemaDefinedArg_kindOfShot,
      UintType schemaDefinedArg_highresMultiplier,
      FloatType schemaDefinedArg_quality360Fov,
      FloatType schemaDefinedArg_fov,
      FloatType schemaDefinedArg_roll,
      FloatType schemaDefinedArg_lwrrentCameraPosX,
      FloatType schemaDefinedArg_lwrrentCameraPosY,
      FloatType schemaDefinedArg_lwrrentCameraPosZ,
      FloatType schemaDefinedArg_lwrrentCameraRotX,
      FloatType schemaDefinedArg_lwrrentCameraRotY,
      FloatType schemaDefinedArg_lwrrentCameraRotZ,
      FloatType schemaDefinedArg_lwrrentCameraRotW,
      FloatType schemaDefinedArg_originalCameraPosX,
      FloatType schemaDefinedArg_originalCameraPosY,
      FloatType schemaDefinedArg_originalCameraPosZ,
      FloatType schemaDefinedArg_originalCameraRotX,
      FloatType schemaDefinedArg_originalCameraRotY,
      FloatType schemaDefinedArg_originalCameraRotZ,
      FloatType schemaDefinedArg_originalCameraRotW,
      SpecialEffectsModeEnum schemaDefinedArg_specialEffectsMode,
      StringType schemaDefinedArg_effectName,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant0_sliderState,
      StringType schemaDefinedArg_userConstant0_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant0_type,
      StringType schemaDefinedArg_userConstant0_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant1_sliderState,
      StringType schemaDefinedArg_userConstant1_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant1_type,
      StringType schemaDefinedArg_userConstant1_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant2_sliderState,
      StringType schemaDefinedArg_userConstant2_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant2_type,
      StringType schemaDefinedArg_userConstant2_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant3_sliderState,
      StringType schemaDefinedArg_userConstant3_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant3_type,
      StringType schemaDefinedArg_userConstant3_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant4_sliderState,
      StringType schemaDefinedArg_userConstant4_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant4_type,
      StringType schemaDefinedArg_userConstant4_value,
      PackedAllUserConstants schemaDefinedArg_allUserConstants,
      UIModeType schemaDefinedArg_uiMode,
      ColorRangeType schemaDefinedArg_colorRange,
      Int64Type schemaDefinedArg_quality360resolution,
      GamepadMappingType schemaDefinedArg_gamepadMapping,
      UintType schemaDefinedArg_gamepadProductId,
      UintType schemaDefinedArg_gamepadVendorId,
      UintType schemaDefinedArg_gamepadVersionNumber,
      BoolType schemaDefinedArg_gamepadUsedForCameraOperationDuringThisSession,
      BoolType schemaDefinedArg_gamepadUsedForUIInteractionDuringThisSession,
      DoubleType schemaDefinedArg_timeElapsedSinceSessionStarted,
      BoolType schemaDefinedArg_highresEnhancementEnabled,
      UintType schemaDefinedArg_anselSDKMajor,
      UintType schemaDefinedArg_anselSDKMinor,
      UintType schemaDefinedArg_anselSDKCommit,
      const std::string& clientVersion,
      const std::string& userId,
      const uint64_t* timestampFileTimeFormatOptional
      )
    {
      try
      {
        if (!gs_dll)
        {
          return E_NOT_VALID_STATE;
        }

        RapidjsonDolwment d;
        d.SetObject();
        auto& a = d.GetAllocator();

        d.AddMember("clientId", gs_clientId, a);
        d.AddMember("clientVer", clientVersion, a);
        d.AddMember("userId", userId, a);
        
        
        if (timestampFileTimeFormatOptional)
        {
          d.AddMember("ts", FormatTimeUTC(*timestampFileTimeFormatOptional), a);
        }
        
        d.AddMember("eventSchemaVer", gs_schemaVer, a);
        
        d.AddMember("event", Rapidjsolwalue(rapidjson::kObjectType), a);
        Rapidjsolwalue& v = d["event"];
        
        v.AddMember("name", "AnselUIClosed", a);
        
        v.AddMember("GDPRCategory", "technical", a);
          
        
        v.AddMember("parameters", Rapidjsolwalue(rapidjson::kObjectType), a);
        
        v["parameters"].AddMember("appExeName", schemaDefinedArg_appExeName, a);
            
        v["parameters"].AddMember("drsProfileName", schemaDefinedArg_drsProfileName, a);
            
        v["parameters"].AddMember("drsAppName", schemaDefinedArg_drsAppName, a);
            
        v["parameters"].AddMember("screenResolutionX", schemaDefinedArg_screenResolutionX, a);
            
        v["parameters"].AddMember("screenResolutionY", schemaDefinedArg_screenResolutionY, a);
            
        v["parameters"].AddMember("colorBufferFormat", schemaDefinedArg_colorBufferFormat, a);
            
        v["parameters"].AddMember("depthBufferFormat", schemaDefinedArg_depthBufferFormat, a);
            
        v["parameters"].AddMember("kindOfShot", KindSliderEnumToString(schemaDefinedArg_kindOfShot), a);
            
        v["parameters"].AddMember("highresMultiplier", schemaDefinedArg_highresMultiplier, a);
            
        v["parameters"].AddMember("quality360Fov", schemaDefinedArg_quality360Fov, a);
            
        v["parameters"].AddMember("fov", schemaDefinedArg_fov, a);
            
        v["parameters"].AddMember("roll", schemaDefinedArg_roll, a);
            
        v["parameters"].AddMember("lwrrentCameraPosX", schemaDefinedArg_lwrrentCameraPosX, a);
            
        v["parameters"].AddMember("lwrrentCameraPosY", schemaDefinedArg_lwrrentCameraPosY, a);
            
        v["parameters"].AddMember("lwrrentCameraPosZ", schemaDefinedArg_lwrrentCameraPosZ, a);
            
        v["parameters"].AddMember("lwrrentCameraRotX", schemaDefinedArg_lwrrentCameraRotX, a);
            
        v["parameters"].AddMember("lwrrentCameraRotY", schemaDefinedArg_lwrrentCameraRotY, a);
            
        v["parameters"].AddMember("lwrrentCameraRotZ", schemaDefinedArg_lwrrentCameraRotZ, a);
            
        v["parameters"].AddMember("lwrrentCameraRotW", schemaDefinedArg_lwrrentCameraRotW, a);
            
        v["parameters"].AddMember("originalCameraPosX", schemaDefinedArg_originalCameraPosX, a);
            
        v["parameters"].AddMember("originalCameraPosY", schemaDefinedArg_originalCameraPosY, a);
            
        v["parameters"].AddMember("originalCameraPosZ", schemaDefinedArg_originalCameraPosZ, a);
            
        v["parameters"].AddMember("originalCameraRotX", schemaDefinedArg_originalCameraRotX, a);
            
        v["parameters"].AddMember("originalCameraRotY", schemaDefinedArg_originalCameraRotY, a);
            
        v["parameters"].AddMember("originalCameraRotZ", schemaDefinedArg_originalCameraRotZ, a);
            
        v["parameters"].AddMember("originalCameraRotW", schemaDefinedArg_originalCameraRotW, a);
            
        v["parameters"].AddMember("specialEffectsMode", SpecialEffectsModeEnumToString(schemaDefinedArg_specialEffectsMode), a);
            
        v["parameters"].AddMember("effectName", schemaDefinedArg_effectName, a);
            
        v["parameters"].AddMember("userConstant0_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant0_sliderState), a);
            
        v["parameters"].AddMember("userConstant0_name", schemaDefinedArg_userConstant0_name, a);
            
        v["parameters"].AddMember("userConstant0_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant0_type), a);
            
        v["parameters"].AddMember("userConstant0_value", schemaDefinedArg_userConstant0_value, a);
            
        v["parameters"].AddMember("userConstant1_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant1_sliderState), a);
            
        v["parameters"].AddMember("userConstant1_name", schemaDefinedArg_userConstant1_name, a);
            
        v["parameters"].AddMember("userConstant1_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant1_type), a);
            
        v["parameters"].AddMember("userConstant1_value", schemaDefinedArg_userConstant1_value, a);
            
        v["parameters"].AddMember("userConstant2_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant2_sliderState), a);
            
        v["parameters"].AddMember("userConstant2_name", schemaDefinedArg_userConstant2_name, a);
            
        v["parameters"].AddMember("userConstant2_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant2_type), a);
            
        v["parameters"].AddMember("userConstant2_value", schemaDefinedArg_userConstant2_value, a);
            
        v["parameters"].AddMember("userConstant3_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant3_sliderState), a);
            
        v["parameters"].AddMember("userConstant3_name", schemaDefinedArg_userConstant3_name, a);
            
        v["parameters"].AddMember("userConstant3_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant3_type), a);
            
        v["parameters"].AddMember("userConstant3_value", schemaDefinedArg_userConstant3_value, a);
            
        v["parameters"].AddMember("userConstant4_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant4_sliderState), a);
            
        v["parameters"].AddMember("userConstant4_name", schemaDefinedArg_userConstant4_name, a);
            
        v["parameters"].AddMember("userConstant4_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant4_type), a);
            
        v["parameters"].AddMember("userConstant4_value", schemaDefinedArg_userConstant4_value, a);
            
        v["parameters"].AddMember("allUserConstants", schemaDefinedArg_allUserConstants, a);
            
        v["parameters"].AddMember("uiMode", UIModeTypeToString(schemaDefinedArg_uiMode), a);
            
        v["parameters"].AddMember("colorRange", ColorRangeTypeToString(schemaDefinedArg_colorRange), a);
            
        v["parameters"].AddMember("quality360resolution", schemaDefinedArg_quality360resolution, a);
            
        v["parameters"].AddMember("gamepadMapping", GamepadMappingTypeToString(schemaDefinedArg_gamepadMapping), a);
            
        v["parameters"].AddMember("gamepadProductId", schemaDefinedArg_gamepadProductId, a);
            
        v["parameters"].AddMember("gamepadVendorId", schemaDefinedArg_gamepadVendorId, a);
            
        v["parameters"].AddMember("gamepadVersionNumber", schemaDefinedArg_gamepadVersionNumber, a);
            
        v["parameters"].AddMember("gamepadUsedForCameraOperationDuringThisSession", schemaDefinedArg_gamepadUsedForCameraOperationDuringThisSession, a);
            
        v["parameters"].AddMember("gamepadUsedForUIInteractionDuringThisSession", schemaDefinedArg_gamepadUsedForUIInteractionDuringThisSession, a);
            
        v["parameters"].AddMember("timeElapsedSinceSessionStarted", schemaDefinedArg_timeElapsedSinceSessionStarted, a);
            
        v["parameters"].AddMember("highresEnhancementEnabled", schemaDefinedArg_highresEnhancementEnabled, a);
            
        v["parameters"].AddMember("anselSDKMajor", schemaDefinedArg_anselSDKMajor, a);
            
        v["parameters"].AddMember("anselSDKMinor", schemaDefinedArg_anselSDKMinor, a);
            
        v["parameters"].AddMember("anselSDKCommit", schemaDefinedArg_anselSDKCommit, a);
            
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        d.Accept(writer);
        
        
        return pSendEvent(buffer.GetString());
        
      }
      catch(const std::bad_alloc&)
      {
        return E_OUTOFMEMORY;
      }
      catch(const std::ilwalid_argument&)
      {
        return E_ILWALIDARG;
      }
      catch(const std::exception&)
      {
        return E_FAIL;
      }
    }

    HRESULT Send_AnselErrorOclwredFull_Event(
      StringType schemaDefinedArg_appExeName,
      StringType schemaDefinedArg_drsProfileName,
      StringType schemaDefinedArg_drsAppName,
      UintType schemaDefinedArg_screenResolutionX,
      UintType schemaDefinedArg_screenResolutionY,
      DxgiFormat schemaDefinedArg_colorBufferFormat,
      DxgiFormat schemaDefinedArg_depthBufferFormat,
      KindSliderEnum schemaDefinedArg_kindOfShot,
      UintType schemaDefinedArg_highresMultiplier,
      FloatType schemaDefinedArg_quality360Fov,
      FloatType schemaDefinedArg_fov,
      FloatType schemaDefinedArg_roll,
      FloatType schemaDefinedArg_lwrrentCameraPosX,
      FloatType schemaDefinedArg_lwrrentCameraPosY,
      FloatType schemaDefinedArg_lwrrentCameraPosZ,
      FloatType schemaDefinedArg_lwrrentCameraRotX,
      FloatType schemaDefinedArg_lwrrentCameraRotY,
      FloatType schemaDefinedArg_lwrrentCameraRotZ,
      FloatType schemaDefinedArg_lwrrentCameraRotW,
      FloatType schemaDefinedArg_originalCameraPosX,
      FloatType schemaDefinedArg_originalCameraPosY,
      FloatType schemaDefinedArg_originalCameraPosZ,
      FloatType schemaDefinedArg_originalCameraRotX,
      FloatType schemaDefinedArg_originalCameraRotY,
      FloatType schemaDefinedArg_originalCameraRotZ,
      FloatType schemaDefinedArg_originalCameraRotW,
      SpecialEffectsModeEnum schemaDefinedArg_specialEffectsMode,
      StringType schemaDefinedArg_effectName,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant0_sliderState,
      StringType schemaDefinedArg_userConstant0_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant0_type,
      StringType schemaDefinedArg_userConstant0_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant1_sliderState,
      StringType schemaDefinedArg_userConstant1_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant1_type,
      StringType schemaDefinedArg_userConstant1_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant2_sliderState,
      StringType schemaDefinedArg_userConstant2_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant2_type,
      StringType schemaDefinedArg_userConstant2_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant3_sliderState,
      StringType schemaDefinedArg_userConstant3_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant3_type,
      StringType schemaDefinedArg_userConstant3_value,
      UserConstantSliderStateEnum schemaDefinedArg_userConstant4_sliderState,
      StringType schemaDefinedArg_userConstant4_name,
      UserConstantTypeEnum schemaDefinedArg_userConstant4_type,
      StringType schemaDefinedArg_userConstant4_value,
      PackedAllUserConstants schemaDefinedArg_allUserConstants,
      UIModeType schemaDefinedArg_uiMode,
      ColorRangeType schemaDefinedArg_colorRange,
      Int64Type schemaDefinedArg_quality360resolution,
      GamepadMappingType schemaDefinedArg_gamepadMapping,
      UintType schemaDefinedArg_gamepadProductId,
      UintType schemaDefinedArg_gamepadVendorId,
      UintType schemaDefinedArg_gamepadVersionNumber,
      BoolType schemaDefinedArg_gamepadUsedForCameraOperationDuringThisSession,
      BoolType schemaDefinedArg_gamepadUsedForUIInteractionDuringThisSession,
      DoubleType schemaDefinedArg_timeElapsedSinceSessionStarted,
      DoubleType schemaDefinedArg_timeElapsedSinceCaptureStarted,
      AnselCaptureState schemaDefinedArg_captureStateOnError,
      ErrorType schemaDefinedArg_errorType,
      StringType schemaDefinedArg_sourceFilename,
      UintType schemaDefinedArg_sourceLine,
      StringType schemaDefinedArg_errorString,
      UintType schemaDefinedArg_errorCode,
      BoolType schemaDefinedArg_highresEnhancementEnabled,
      UintType schemaDefinedArg_anselSDKMajor,
      UintType schemaDefinedArg_anselSDKMinor,
      UintType schemaDefinedArg_anselSDKCommit,
      const std::string& clientVersion,
      const std::string& userId,
      const uint64_t* timestampFileTimeFormatOptional
      )
    {
      try
      {
        if (!gs_dll)
        {
          return E_NOT_VALID_STATE;
        }

        RapidjsonDolwment d;
        d.SetObject();
        auto& a = d.GetAllocator();

        d.AddMember("clientId", gs_clientId, a);
        d.AddMember("clientVer", clientVersion, a);
        d.AddMember("userId", userId, a);
        
        
        if (timestampFileTimeFormatOptional)
        {
          d.AddMember("ts", FormatTimeUTC(*timestampFileTimeFormatOptional), a);
        }
        
        d.AddMember("eventSchemaVer", gs_schemaVer, a);
        
        d.AddMember("event", Rapidjsolwalue(rapidjson::kObjectType), a);
        Rapidjsolwalue& v = d["event"];
        
        v.AddMember("name", "AnselErrorOclwredFull", a);
        
        v.AddMember("GDPRCategory", "technical", a);
          
        
        v.AddMember("parameters", Rapidjsolwalue(rapidjson::kObjectType), a);
        
        v["parameters"].AddMember("appExeName", schemaDefinedArg_appExeName, a);
            
        v["parameters"].AddMember("drsProfileName", schemaDefinedArg_drsProfileName, a);
            
        v["parameters"].AddMember("drsAppName", schemaDefinedArg_drsAppName, a);
            
        v["parameters"].AddMember("screenResolutionX", schemaDefinedArg_screenResolutionX, a);
            
        v["parameters"].AddMember("screenResolutionY", schemaDefinedArg_screenResolutionY, a);
            
        v["parameters"].AddMember("colorBufferFormat", schemaDefinedArg_colorBufferFormat, a);
            
        v["parameters"].AddMember("depthBufferFormat", schemaDefinedArg_depthBufferFormat, a);
            
        v["parameters"].AddMember("kindOfShot", KindSliderEnumToString(schemaDefinedArg_kindOfShot), a);
            
        v["parameters"].AddMember("highresMultiplier", schemaDefinedArg_highresMultiplier, a);
            
        v["parameters"].AddMember("quality360Fov", schemaDefinedArg_quality360Fov, a);
            
        v["parameters"].AddMember("fov", schemaDefinedArg_fov, a);
            
        v["parameters"].AddMember("roll", schemaDefinedArg_roll, a);
            
        v["parameters"].AddMember("lwrrentCameraPosX", schemaDefinedArg_lwrrentCameraPosX, a);
            
        v["parameters"].AddMember("lwrrentCameraPosY", schemaDefinedArg_lwrrentCameraPosY, a);
            
        v["parameters"].AddMember("lwrrentCameraPosZ", schemaDefinedArg_lwrrentCameraPosZ, a);
            
        v["parameters"].AddMember("lwrrentCameraRotX", schemaDefinedArg_lwrrentCameraRotX, a);
            
        v["parameters"].AddMember("lwrrentCameraRotY", schemaDefinedArg_lwrrentCameraRotY, a);
            
        v["parameters"].AddMember("lwrrentCameraRotZ", schemaDefinedArg_lwrrentCameraRotZ, a);
            
        v["parameters"].AddMember("lwrrentCameraRotW", schemaDefinedArg_lwrrentCameraRotW, a);
            
        v["parameters"].AddMember("originalCameraPosX", schemaDefinedArg_originalCameraPosX, a);
            
        v["parameters"].AddMember("originalCameraPosY", schemaDefinedArg_originalCameraPosY, a);
            
        v["parameters"].AddMember("originalCameraPosZ", schemaDefinedArg_originalCameraPosZ, a);
            
        v["parameters"].AddMember("originalCameraRotX", schemaDefinedArg_originalCameraRotX, a);
            
        v["parameters"].AddMember("originalCameraRotY", schemaDefinedArg_originalCameraRotY, a);
            
        v["parameters"].AddMember("originalCameraRotZ", schemaDefinedArg_originalCameraRotZ, a);
            
        v["parameters"].AddMember("originalCameraRotW", schemaDefinedArg_originalCameraRotW, a);
            
        v["parameters"].AddMember("specialEffectsMode", SpecialEffectsModeEnumToString(schemaDefinedArg_specialEffectsMode), a);
            
        v["parameters"].AddMember("effectName", schemaDefinedArg_effectName, a);
            
        v["parameters"].AddMember("userConstant0_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant0_sliderState), a);
            
        v["parameters"].AddMember("userConstant0_name", schemaDefinedArg_userConstant0_name, a);
            
        v["parameters"].AddMember("userConstant0_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant0_type), a);
            
        v["parameters"].AddMember("userConstant0_value", schemaDefinedArg_userConstant0_value, a);
            
        v["parameters"].AddMember("userConstant1_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant1_sliderState), a);
            
        v["parameters"].AddMember("userConstant1_name", schemaDefinedArg_userConstant1_name, a);
            
        v["parameters"].AddMember("userConstant1_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant1_type), a);
            
        v["parameters"].AddMember("userConstant1_value", schemaDefinedArg_userConstant1_value, a);
            
        v["parameters"].AddMember("userConstant2_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant2_sliderState), a);
            
        v["parameters"].AddMember("userConstant2_name", schemaDefinedArg_userConstant2_name, a);
            
        v["parameters"].AddMember("userConstant2_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant2_type), a);
            
        v["parameters"].AddMember("userConstant2_value", schemaDefinedArg_userConstant2_value, a);
            
        v["parameters"].AddMember("userConstant3_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant3_sliderState), a);
            
        v["parameters"].AddMember("userConstant3_name", schemaDefinedArg_userConstant3_name, a);
            
        v["parameters"].AddMember("userConstant3_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant3_type), a);
            
        v["parameters"].AddMember("userConstant3_value", schemaDefinedArg_userConstant3_value, a);
            
        v["parameters"].AddMember("userConstant4_sliderState", UserConstantSliderStateEnumToString(schemaDefinedArg_userConstant4_sliderState), a);
            
        v["parameters"].AddMember("userConstant4_name", schemaDefinedArg_userConstant4_name, a);
            
        v["parameters"].AddMember("userConstant4_type", UserConstantTypeEnumToString(schemaDefinedArg_userConstant4_type), a);
            
        v["parameters"].AddMember("userConstant4_value", schemaDefinedArg_userConstant4_value, a);
            
        v["parameters"].AddMember("allUserConstants", schemaDefinedArg_allUserConstants, a);
            
        v["parameters"].AddMember("uiMode", UIModeTypeToString(schemaDefinedArg_uiMode), a);
            
        v["parameters"].AddMember("colorRange", ColorRangeTypeToString(schemaDefinedArg_colorRange), a);
            
        v["parameters"].AddMember("quality360resolution", schemaDefinedArg_quality360resolution, a);
            
        v["parameters"].AddMember("gamepadMapping", GamepadMappingTypeToString(schemaDefinedArg_gamepadMapping), a);
            
        v["parameters"].AddMember("gamepadProductId", schemaDefinedArg_gamepadProductId, a);
            
        v["parameters"].AddMember("gamepadVendorId", schemaDefinedArg_gamepadVendorId, a);
            
        v["parameters"].AddMember("gamepadVersionNumber", schemaDefinedArg_gamepadVersionNumber, a);
            
        v["parameters"].AddMember("gamepadUsedForCameraOperationDuringThisSession", schemaDefinedArg_gamepadUsedForCameraOperationDuringThisSession, a);
            
        v["parameters"].AddMember("gamepadUsedForUIInteractionDuringThisSession", schemaDefinedArg_gamepadUsedForUIInteractionDuringThisSession, a);
            
        v["parameters"].AddMember("timeElapsedSinceSessionStarted", schemaDefinedArg_timeElapsedSinceSessionStarted, a);
            
        v["parameters"].AddMember("timeElapsedSinceCaptureStarted", schemaDefinedArg_timeElapsedSinceCaptureStarted, a);
            
        v["parameters"].AddMember("captureStateOnError", AnselCaptureStateToString(schemaDefinedArg_captureStateOnError), a);
            
        v["parameters"].AddMember("errorType", ErrorTypeToString(schemaDefinedArg_errorType), a);
            
        v["parameters"].AddMember("sourceFilename", schemaDefinedArg_sourceFilename, a);
            
        v["parameters"].AddMember("sourceLine", schemaDefinedArg_sourceLine, a);
            
        v["parameters"].AddMember("errorString", schemaDefinedArg_errorString, a);
            
        v["parameters"].AddMember("errorCode", schemaDefinedArg_errorCode, a);
            
        v["parameters"].AddMember("highresEnhancementEnabled", schemaDefinedArg_highresEnhancementEnabled, a);
            
        v["parameters"].AddMember("anselSDKMajor", schemaDefinedArg_anselSDKMajor, a);
            
        v["parameters"].AddMember("anselSDKMinor", schemaDefinedArg_anselSDKMinor, a);
            
        v["parameters"].AddMember("anselSDKCommit", schemaDefinedArg_anselSDKCommit, a);
            
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        d.Accept(writer);
        
        
        return pSendEvent(buffer.GetString());
        
      }
      catch(const std::bad_alloc&)
      {
        return E_OUTOFMEMORY;
      }
      catch(const std::ilwalid_argument&)
      {
        return E_ILWALIDARG;
      }
      catch(const std::exception&)
      {
        return E_FAIL;
      }
    }

    HRESULT Send_AnselErrorOclwredShort_Event(
      ErrorType schemaDefinedArg_errorType,
      StringType schemaDefinedArg_sourceFilename,
      UintType schemaDefinedArg_sourceLine,
      StringType schemaDefinedArg_errorString,
      UintType schemaDefinedArg_errorCode,
      const std::string& clientVersion,
      const std::string& userId,
      const uint64_t* timestampFileTimeFormatOptional
      )
    {
      try
      {
        if (!gs_dll)
        {
          return E_NOT_VALID_STATE;
        }

        RapidjsonDolwment d;
        d.SetObject();
        auto& a = d.GetAllocator();

        d.AddMember("clientId", gs_clientId, a);
        d.AddMember("clientVer", clientVersion, a);
        d.AddMember("userId", userId, a);
        
        
        if (timestampFileTimeFormatOptional)
        {
          d.AddMember("ts", FormatTimeUTC(*timestampFileTimeFormatOptional), a);
        }
        
        d.AddMember("eventSchemaVer", gs_schemaVer, a);
        
        d.AddMember("event", Rapidjsolwalue(rapidjson::kObjectType), a);
        Rapidjsolwalue& v = d["event"];
        
        v.AddMember("name", "AnselErrorOclwredShort", a);
        
        v.AddMember("GDPRCategory", "technical", a);
          
        
        v.AddMember("parameters", Rapidjsolwalue(rapidjson::kObjectType), a);
        
        v["parameters"].AddMember("errorType", ErrorTypeToString(schemaDefinedArg_errorType), a);
            
        v["parameters"].AddMember("sourceFilename", schemaDefinedArg_sourceFilename, a);
            
        v["parameters"].AddMember("sourceLine", schemaDefinedArg_sourceLine, a);
            
        v["parameters"].AddMember("errorString", schemaDefinedArg_errorString, a);
            
        v["parameters"].AddMember("errorCode", schemaDefinedArg_errorCode, a);
            
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        d.Accept(writer);
        
        
        return pSendEvent(buffer.GetString());
        
      }
      catch(const std::bad_alloc&)
      {
        return E_OUTOFMEMORY;
      }
      catch(const std::ilwalid_argument&)
      {
        return E_ILWALIDARG;
      }
      catch(const std::exception&)
      {
        return E_FAIL;
      }
    }

  }
}
