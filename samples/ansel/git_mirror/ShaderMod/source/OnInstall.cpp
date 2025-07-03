#define ANSEL_DLL_EXPORTS
#include <Windows.h>
#include "OnInstall.h"
#define IPC_ENABLED 1
#include "ipc/AnselIPC.h"

ANSEL_DLL_API void __cdecl OnInstall()
{
    ipc::onInstall();
}
