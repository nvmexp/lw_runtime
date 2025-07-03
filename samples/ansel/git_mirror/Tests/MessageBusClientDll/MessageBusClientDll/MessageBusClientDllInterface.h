#pragma once

#ifdef MESSAGEBUS_CLIENT_DLL_EXPORTS
#define MESSAGEBUS_CLIENT_DLL_API extern "C" __declspec(dllexport)
#else
#define MESSAGEBUS_CLIENT_DLL_API
#endif

#include <cstdint>

typedef void(__cdecl *PFNONMESSAGEBUSCALLBACK)(const char* msg, uint32_t size);

MESSAGEBUS_CLIENT_DLL_API bool __cdecl joinMessageBus(const char* system, const char* module, PFNONMESSAGEBUSCALLBACK clbk);
MESSAGEBUS_CLIENT_DLL_API bool __cdecl postMessage(const char* msg, uint32_t size);
MESSAGEBUS_CLIENT_DLL_API void __cdecl leaveMessageBus();
