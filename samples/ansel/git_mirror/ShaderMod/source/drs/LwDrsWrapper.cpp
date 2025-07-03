#include "LwDrsWrapper.h"
#include <LwDrsLow.h>
#include <algorithm>
#include <sstream>

#ifndef LWAPI_DISABLE_LOGGING
#   include <Log.h>
#   define LOG_LWAPI_ERROR(message, status) LOG_ERROR("LwApi: " message " - error code %d", status)
#else
#   define LOG_LWAPI_ERROR(message, status)
#endif

#ifndef LWAPI_DRS_ADDITIONAL_LOGGING
#   define LOG_LWDRS_DEBUG(...)
#else
#   define LOG_LWDRS_DEBUG(...) LOG_DEBUG(__VA_ARGS__)
#endif


#include "darkroom/StringColwersion.h"

namespace
{
    bool getProfileNameHelper(std::wstring& profileName)
    {
        bool ret = false;
        LwU32 profileType = 0, bytesNeeded = 0;
        const wchar_t *wcProfileName = nullptr, *wcAppName = nullptr;
        LwDrsSession session = { 0 };
        LwDrsSessionOptions sessionOptions = { 0 };
        sessionOptions.appendPath = 1;
        sessionOptions.appendEXE = 1;
        LwDrsStatus status = LwDrsSessionCreate(&session);
        if (status == DRS_STATUS_OK)
        {
            status = LwDrsSessionSetOptions(&session, sessionOptions);
            if (status == DRS_STATUS_OK)
            {
                status = LwDrsSessionComputePublicSettingsCacheSize(&session, &bytesNeeded);
                if (status == DRS_STATUS_OK)
                {
                    status = LwDrsSessionGetProfileName(&session, &wcProfileName, &profileType);
                    if (status == DRS_STATUS_OK)
                    {
                        // only return true if we have profile name and it's not base profile
                        if (profileType != DRS_FLAG_PROFILE_TYPE_GLOBAL && wcProfileName)
                        {
                            profileName = wcProfileName;
                            ret = true;
                        }
                    }
                    else
                        LOG_LWAPI_ERROR("LwDrsSessionGetProfileName failed (%d)", status);
                }
                else
                    LOG_LWAPI_ERROR("LwDrsSessionComputePublicSettingsCacheSize failed (%d)", status);
            }
            else
                LOG_LWAPI_ERROR("LwDrsSessionSetOptions failed (%d)", status);

            LwDrsSessionDestroy(&session);
        }
        else
            LOG_LWAPI_ERROR("LwDrsSessionCreate failed (%d)", status);

        return ret;
    }

    bool getKeyValueHelper(uint32_t keyID, LwU32& keyValueSizeInBytes, uint32_t& keyValue, std::wstring* keyValueString, bool readGold, bool readString)
    {
        bool ret = false;
        LwU32 profileType = 0, bytesNeeded = 0, numSettings = 0;
        LwDrsSession session;
        LwDrsSessionOptions sessionOptions = { 0 };
        sessionOptions.appendPath = 1;
        sessionOptions.appendEXE = 1;
        LwDrsStatus status = LwDrsSessionCreate(&session);
        if (status == DRS_STATUS_OK)
        {
            if (readGold)
            {
                status = LwDrsSessionSpecifyReadGoldDB(&session);
                LOG_DEBUG("  Reading Gold DB...");
            }
            else
            {
                LOG_DEBUG("  Reading User DB...");
            }

            if (status == DRS_STATUS_OK)
            {
                status = LwDrsSessionSetOptions(&session, sessionOptions);
            }
            else
            {
                LOG_LWAPI_ERROR("  Couldn't switch DRS mode to GOLD", status);
                LOG_LWDRS_DEBUG("  Reading User DB Instead...");
            }

            if (status == DRS_STATUS_OK)
            {
                //LOG_LWDRS_DEBUG("  DRS session created.");
                status = LwDrsSessionComputePublicSettingsCacheSize(&session, &bytesNeeded);
                if (status == DRS_STATUS_OK)
                {
                    LwDrsPublicSetting *pPubSet = static_cast<LwDrsPublicSetting*>(malloc(bytesNeeded));
                    if (pPubSet != NULL)
                    {
                        status = LwDrsSessionLoadLwrrentProcessSettingsOnUserCache(&session, (LwU32 *)pPubSet, bytesNeeded, &numSettings);
                        if (status == DRS_STATUS_OK)
                        {
                            //LOG_LWDRS_DEBUG("  Loaded Settings (%d in total)", numSettings);
                            std::stringstream settingsStream;
                            auto i = 0u;
                            for (i; i < numSettings; i++)
                            {
                                settingsStream << "0x" << std::hex << uint32_t(pPubSet[i].id);
                                if (i != numSettings - 1) { settingsStream << ", "; }
                                if (pPubSet[i].id == keyID)
                                {
                                    //LOG_LWDRS_DEBUG("  Setting found!");
                                    keyValueSizeInBytes = pPubSet[i].sizeInBytes;
                                    if (!readString && keyValueSizeInBytes == sizeof(LwU32))
                                    {
                                        if (keyValueString)
                                        {
                                            (*keyValueString) = L"";
                                        }
                                        //LOG_LWDRS_DEBUG("  Reading value...");
                                        keyValue = pPubSet[i].u.value;
                                        LOG_DEBUG("  Value: 0x%08x", keyValue);
                                        ret = true;
                                    }
                                    else if (readString)
                                    {
                                        keyValue = 0;
                                        if (keyValueString)
                                        {
                                            (*keyValueString) = reinterpret_cast<wchar_t*>(pPubSet[i].u.ptr);
                                            LOG_DEBUG("  StringValue: %S", keyValueString->c_str());
                                        }
                                        LOG_DEBUG("  SizeInBytes: %d", keyValueSizeInBytes);
                                        ret = true;
                                    }
                                    break;
                                }
                            }
                            if (i == numSettings)
                            {
                                LOG_DEBUG("  **NOT FOUND** Did not find 0x%08x in loaded settings: (%s)", keyID, settingsStream.str().c_str());
                            }
                        }
                        else
                            LOG_LWAPI_ERROR("  LwDrsSessionLoadLwrrentProcessSettingsOnUserCache failed (%d)", status);

                        free(pPubSet);
                    }
                    else
                        LOG_LWAPI_ERROR("  Couldn't allocate memory to store settings", 0);
                }
                else
                    LOG_LWAPI_ERROR("  LwDrsSessionComputePublicSettingsCacheSize failed (%d)", status);

                LwDrsSessionDestroy(&session);
            }
            else
                LOG_LWAPI_ERROR("  LwDrsSessionSetOptions failed (%d)", status);
        }
        else
            LOG_LWAPI_ERROR("  LwDrsSessionCreate failed (%d)", status);

        return ret;
    }
}

namespace drs
{
    bool getProfileName(std::wstring& profileName)
    {
        return getProfileNameHelper(profileName);
    }

    bool getKeyValue(uint32_t keyID, uint32_t& keyValue, bool readGold)
    {
        LwU32 keyValueSizeInBytes;
        return getKeyValueHelper(keyID, keyValueSizeInBytes, keyValue, NULL, readGold, false);
    }

    bool getKeyValueString(uint32_t keyID, std::wstring& keyValueString, bool readGold)
    {
        LwU32 keyValueSizeInBytes; uint32_t keyValue;
        return getKeyValueHelper(keyID, keyValueSizeInBytes, keyValue, &keyValueString, readGold, true);
    }
}

