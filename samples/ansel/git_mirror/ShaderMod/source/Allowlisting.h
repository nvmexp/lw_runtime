#pragma once

#include <lwapi.h>
#include <LwApiDriverSettings.h>
#include "drs/LwDrsWrapper.h"
#include "drs/LwDrsDefines.h"
#include "CommonStructs.h"
#include "DenylistParser.h"

#include "Log.h"

#if _DEBUG
#define IS_GOLD_DB_CHECK_FORCED false
#else
#define IS_GOLD_DB_CHECK_FORCED true
#endif

static uint32_t getFreeStyleMode(DWORD pid, AnselDenylist* denylist)
{
    uint32_t freeStyleModeValue;
    // Read GOLD DRS value of the mode key
    LOG_DEBUG("Reading FreeStyle Mode...");
    const bool isFreestyleModeKeySet = drs::getKeyValue(ANSEL_FREESTYLE_MODE_ID, freeStyleModeValue, IS_GOLD_DB_CHECK_FORCED);
    if (isFreestyleModeKeySet)
    {
        // TEMPORARY WARKAROUND FOR DRS/LWAPI SECURITY FLAW - only allow APPROVED_ONLY and ENABLED for now
        if (freeStyleModeValue != ANSEL_FREESTYLE_MODE_APPROVED_ONLY
            && freeStyleModeValue != ANSEL_FREESTYLE_MODE_ENABLED)
        {
            freeStyleModeValue = ANSEL_FREESTYLE_MODE_DISABLED;
        }
    }
    else
    {
        freeStyleModeValue = ANSEL_FREESTYLE_MODE_DEFAULT;
    }

    // If we're getting any flag other than "disable" from DRS, check if developers opted-out manually
    //  via the named mutex
    if (freeStyleModeValue != ANSEL_FREESTYLE_MODE_DISABLED)
    {
        char fsMutexName[MAX_PATH];
        sprintf_s(fsMutexName, "LWPU/FreeStyleDisallow/%d", pid);
        HANDLE fsDisallowMutex = OpenMutexA(SYNCHRONIZE, false, fsMutexName);

        bool isFreeStyleDisallowed = fsDisallowMutex ? true : false;
        if (isFreeStyleDisallowed)
        {
            LOG_INFO("FreeStyle disallowed by developer");
            CloseHandle(fsDisallowMutex);
            freeStyleModeValue = ANSEL_FREESTYLE_MODE_DISABLED;
        }
    }

    if (ANSEL_FREESTYLE_MODE_DISABLED != freeStyleModeValue)
    {
        // Only spend time initializing and checking the denylist when freestyle is enabled.
        AnselDenylist denylist_local;
        if (!denylist)
        {
            denylist = &denylist_local;
        }
        denylist->CheckToInitializeWithDRS();
        if (denylist->ActiveBuildIDIsDenylisted())
        {
            LOG_DEBUG("Current build is denylisted for Freestyle. Freestyle disabled...");
            freeStyleModeValue = ANSEL_FREESTYLE_MODE_DISABLED;
        }
        else
        {
            LOG_DEBUG("Current build is *NOT* denylisted for Freestyle.");
        }
    }

    switch (freeStyleModeValue)
    {
        case ANSEL_FREESTYLE_MODE_ENABLED:
            LOG_INFO("Unrestricted Freestyle is enabled for this game.");
            break;
        case ANSEL_FREESTYLE_MODE_APPROVED_ONLY:
            LOG_INFO("Restricted Freestyle is enabled for this game.");
            break;
        default:
            LOG_INFO("Freestyle is *NOT* enabled for this game.");
            break;
    }

    return freeStyleModeValue;
}

static uint32_t getBuffersDisabled()
{
    uint32_t buffersDisabledValue;
    // Read GOLD DRS value
    LOG_DEBUG("Reading Denylisted Buffers...");
    const bool isBuffersDisabledKeySet = drs::getKeyValue(ANSEL_BUFFERS_DISABLED_ID, buffersDisabledValue, IS_GOLD_DB_CHECK_FORCED);
    if (!isBuffersDisabledKeySet)
    {
        buffersDisabledValue = ANSEL_BUFFERS_DISABLED_DEFAULT;
    }

    return buffersDisabledValue;
}

static uint32_t getAllowOffline()
{
    uint32_t allowOfflineValue;
    // Read GOLD DRS value
    LOG_DEBUG("Reading if allowed offline...");
    const bool isAllowOfflineKeySet = drs::getKeyValue(ANSEL_ALLOW_OFFLINE_ID, allowOfflineValue, IS_GOLD_DB_CHECK_FORCED);
    if (!isAllowOfflineKeySet)
    {
        allowOfflineValue = ANSEL_ALLOW_OFFLINE_DEFAULT;
    }

    return allowOfflineValue;
}

