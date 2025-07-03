#pragma once

#include "ansel/Camera.h"
#include "ansel/Configuration.h"
#include "CommonStructs.h"

#include <sstream>
#include <string>
#include <vector>

#ifdef _DEBUG
#define HandleFailure() reportFatalError(__FILE__, __LINE__, FatalErrorCode::kGeneric, "D3D error status=%i\n", status); __debugbreak(); return status;
#define HandleFailureWMessage(message, ...) reportFatalError(__FILE__, __LINE__, FatalErrorCode::kGeneric, std::string("D3D error status=").append(std::to_string(status)).append(" message=").append(message).c_str(), __VA_ARGS__); __debugbreak(); return status;
#define ReportFailure() reportFatalError(__FILE__, __LINE__, FatalErrorCode::kGeneric, ""); __debugbreak(); 
#define ReportFailureWCode(code) reportFatalError(__FILE__, __LINE__, code, ""); __debugbreak(); 
#define ReportFailureWMessage(message, ...) reportFatalError(__FILE__, __LINE__, FatalErrorCode::kGeneric, message, __VA_ARGS__); __debugbreak(); 
#define ReportFailureWCodeAndMessage(code, message, ...) reportFatalError(__FILE__, __LINE__, code, message, __VA_ARGS__); __debugbreak(); 
#define ReportNonFatalFailure() reportNonFatalError(__FILE__, __LINE__, FatalErrorCode::kGeneric, ""); __debugbreak(); 
#define ReportNonFatalFailureWCode(code) reportNonFatalError(__FILE__, __LINE__, code, ""); __debugbreak(); 
#define ReportNonFatalFailureWMessage(message, ...) reportNonFatalError(__FILE__, __LINE__, FatalErrorCode::kGeneric, message, __VA_ARGS__); __debugbreak(); 
#define ReportNonFatalFailureWCodeAndMessage(code, message, ...) reportNonFatalError(__FILE__, __LINE__, code, message, __VA_ARGS__); __debugbreak(); 

#else
#define HandleFailure() reportFatalError(__FILE__, __LINE__, FatalErrorCode::kGeneric, "D3D error status=%i\n", status); return status;
#define HandleFailureWMessage(message, ...) reportFatalError(__FILE__, __LINE__, FatalErrorCode::kGeneric, std::string("D3D error status=").append(std::to_string(status)).append(" message=").append(message).c_str(), __VA_ARGS__); return status;
#define ReportFailure() reportFatalError(__FILE__, __LINE__, FatalErrorCode::kGeneric, "");
#define ReportFailureWCode(code) reportFatalError(__FILE__, __LINE__, code, "");  
#define ReportFailureWMessage(message, ...) reportFatalError(__FILE__, __LINE__, FatalErrorCode::kGeneric, message, __VA_ARGS__);
#define ReportFailureWCodeAndMessage(code, message, ...) reportFatalError(__FILE__, __LINE__, code, message, __VA_ARGS__); 
#define ReportNonFatalFailure() reportNonFatalError(__FILE__, __LINE__, FatalErrorCode::kGeneric, "");
#define ReportNonFatalFailureWCode(code) reportNonFatalError(__FILE__, __LINE__, code, "");  
#define ReportNonFatalFailureWMessage(message, ...) reportNonFatalError(__FILE__, __LINE__, FatalErrorCode::kGeneric, message, __VA_ARGS__);
#define ReportNonFatalFailureWCodeAndMessage(code, message, ...) reportNonFatalError(__FILE__, __LINE__, code, message, __VA_ARGS__); 

#endif

namespace lwanselutils
{
    float colwertToHorizontalFov(const ansel::Camera& cam, const ansel::Configuration& cfg, uint32_t viewportWidth, uint32_t viewportHeight);
    std::string appendTimeA(const char * inString_pre, const char * inString_post = nullptr);
    std::wstring appendTimeW(const wchar_t * inString_pre, const wchar_t * inString_post = nullptr);
    void buildSplitStringFromNumber(uint64_t number, wchar_t * buf, size_t bufSize);
    bool CreateDirectoryRelwrsively(const wchar_t *path);
    std::wstring getAppNameFromProcess();

    // from https://www.fluentcpp.com/2017/04/21/how-to-split-a-string-in-c/
    static std::vector<std::wstring> StrSplit(const std::wstring& s, wchar_t delimiter)
    {
        std::vector<std::wstring> tokens;
        std::wstring token;
        std::wistringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter))
        {
            tokens.push_back(token);
        }
        return tokens;
    }

    // feature flags
    //const bool isHybridCameraEnabled = false;

    std::string GetDxgiFormatName(UINT format);
#define DxgiFormat_cstr(format) lwanselutils::GetDxgiFormatName(format).c_str()

    DXGI_FORMAT colwertFromTypelessIfNeeded(DXGI_FORMAT inFormat, bool checkColw = true);
    DXGI_FORMAT colwertToTypeless(DXGI_FORMAT inFormat, bool checkColw = true);

    DXGI_FORMAT getSRVFormatDepth(DXGI_FORMAT inFormat, bool checkColw = true);
    DXGI_FORMAT getSRVFormatStencil(DXGI_FORMAT inFormat, bool checkColw = true);

#if DO_DEBUG_PRINTS
    void DebugPrint(const char * format, ...)
    {
        char orig[1000];

        va_list argptr;
        va_start(argptr, format);
        vsprintf_s(orig, format, argptr);
        va_end(argptr);

        // Colwert to a wchar_t*
        size_t origsize = strlen(orig) + 1;
        const size_t newsize = 1000;
        size_t colwertedChars = 0;
        wchar_t wcstring[newsize];
        mbstowcs_s(&colwertedChars, wcstring, origsize, orig, _TRUNCATE);
        OutputDebugString(wcstring);
    }
#else
#define DebugPrint(...)
#endif
}
