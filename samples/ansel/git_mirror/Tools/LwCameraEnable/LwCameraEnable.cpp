// LwCameraEnable.cpp : Defines the entry point for the console application.
//

#include <lwapi.h>
#include <LwApiDriverSettings.h>

#include <stdio.h>
#include <tchar.h>
#include <string.h>
#include <string>
#include <codecvt>

static bool s_isAnselEnabled = false;
static bool s_isAnselAllowlisting = false;
static bool s_generalError = false;

//Was getting link errors until I did this!
// http://stackoverflow.com/questions/31053670/unresolved-external-symbol-vsnprintf-in-dxerr-lib
int(__cdecl * __vsnwprintf)(wchar_t *, size_t, const wchar_t*, va_list) = _vsnwprintf;

void ReportLwAPIError(LwAPI_Status status)
{
    LwAPI_ShortString szDesc = { 0 };
    LwAPI_GetErrorMessage(status, szDesc);
    printf("LWAPI Error: %s\n", szDesc);
    s_generalError = true;
}

bool SaveDRS(LwU32 keyID, bool & returlwalue)
{
    bool foundValue = false;

    LwAPI_Initialize();
    LwAPI_Status ret;
    LwDRSSessionHandle hSession;
    LwDRSProfileHandle hProfile;

    ret = LwAPI_DRS_CreateSession(&hSession);
    ret = LwAPI_DRS_LoadSettings(hSession);
    ret = LwAPI_DRS_GetBaseProfile(hSession, &hProfile);
    if (ret != LWAPI_OK) ReportLwAPIError(ret);

    LWDRS_SETTING lwApiSetting = { 0 };
    {
        lwApiSetting.version = LWDRS_SETTING_VER;
        lwApiSetting.settingId = keyID;
        lwApiSetting.settingType = LWDRS_DWORD_TYPE;

        ret = LwAPI_DRS_GetSetting(hSession, hProfile, keyID, &lwApiSetting);
        if (ret != LWAPI_OK && ret != LWAPI_SETTING_NOT_FOUND) ReportLwAPIError(ret);

        foundValue = ret != LWAPI_SETTING_NOT_FOUND;
        if (foundValue)
        {
            returlwalue = (lwApiSetting.u32LwrrentValue ? true : false);
        }
    }

    ret = LwAPI_DRS_DestroySession(hSession);
    if (ret != LWAPI_OK) ReportLwAPIError(ret);

    return foundValue;
}

void ApplyDRS(LwU32 keyID, LwU32 value, bool reportErrors = false)
{
    LwAPI_Initialize();
    LwAPI_Status ret;
    LwDRSSessionHandle hSession;
    LwDRSProfileHandle hProfile;

    ret = LwAPI_DRS_CreateSession(&hSession);
    ret = LwAPI_DRS_LoadSettings(hSession);
    ret = LwAPI_DRS_GetBaseProfile(hSession, &hProfile);
    if (ret != LWAPI_OK && reportErrors)
    {
        printf("LwAPI_DRS_GetBaseProfile %d\n", ret);
    }

    {
        //ANSELENABLE
        LWDRS_SETTING lwApiSetting = { 0 };
        lwApiSetting.version = LWDRS_SETTING_VER;
        lwApiSetting.settingId = keyID;
        lwApiSetting.settingType = LWDRS_DWORD_TYPE;
        lwApiSetting.u32LwrrentValue = value;

        ret = LwAPI_DRS_SetSetting(hSession, hProfile, &lwApiSetting);
        if (ret != LWAPI_OK && reportErrors)
        {
            printf("LwAPI_DRS_SetSetting %d\n", ret);
        }
    }

    ret = LwAPI_DRS_SaveSettings(hSession);
    if (ret != LWAPI_OK && reportErrors)
    {
        printf("LwAPI_DRS_SaveSettings %d\n", ret);
    }
    ret = LwAPI_DRS_DestroySession(hSession);
    if (ret != LWAPI_OK && reportErrors)
    {
        printf("LwAPI_DRS_DestroySession %d\n", ret);
    }
}

void RestoreDRS(LwU32 keyID)
{
    LwAPI_Initialize();
    LwAPI_Status ret;
    LwDRSSessionHandle hSession;
    LwDRSProfileHandle hProfile;

    ret = LwAPI_DRS_CreateSession(&hSession);
    ret = LwAPI_DRS_LoadSettings(hSession);
    ret = LwAPI_DRS_GetBaseProfile(hSession, &hProfile);
    if (ret != LWAPI_OK) ReportLwAPIError(ret);

    ret = LwAPI_DRS_DeleteProfileSetting(hSession, hProfile, keyID);
    // Setting may already have been removed and we should not report that as an error:
    if (ret != LWAPI_OK && ret != LWAPI_SETTING_NOT_FOUND) ReportLwAPIError(ret);

    ret = LwAPI_DRS_SaveSettings(hSession);
    if (ret != LWAPI_OK) ReportLwAPIError(ret);
    ret = LwAPI_DRS_DestroySession(hSession);
    if (ret != LWAPI_OK) ReportLwAPIError(ret);
}

void DeleteKeyFromProfileInApplicationPath(const wchar_t * appPath, LwU32 keyID)
{
    LwAPI_Initialize();
    {
        LwDRSSessionHandle session;
        if (LwAPI_DRS_CreateSession(&session) == LWAPI_OK)
        {
            if (LwAPI_DRS_LoadSettings(session) == LWAPI_OK)
            {
                LwDRSProfileHandle profile;

                LwAPI_UnicodeString appName = { 0 };
                LWDRS_APPLICATION app = { 0 };
                app.version = LWDRS_APPLICATION_VER;

                wcsncpy_s((wchar_t*)appName, LWAPI_UNICODE_STRING_MAX, appPath, wcslen(appPath));
                if (LwAPI_DRS_FindApplicationByName(session, appName, &profile, &app) == LWAPI_OK)
                {
                    LwAPI_Status ret;

                    ret = LwAPI_DRS_DeleteProfileSetting(session, profile, keyID);
                    ret = LwAPI_DRS_SaveSettings(session);
                }
                else
                    printf("ERROR: No profile found for application. No key to remove.\n");
            }
            LwAPI_DRS_DestroySession(session);
        }
    }
    LwAPI_Unload();
}

void SetKeyInProfileInApplicationPath(const wchar_t * appPath, LwU32 keyID, LwU32 value)
{
    LwAPI_Initialize();
    {
        LwDRSSessionHandle session;
        if (LwAPI_DRS_CreateSession(&session) == LWAPI_OK)
        {
            if (LwAPI_DRS_LoadSettings(session) == LWAPI_OK)
            {
                LwDRSProfileHandle profile;

                LwAPI_UnicodeString appName = { 0 };
                LWDRS_APPLICATION app = { 0 };
                app.version = LWDRS_APPLICATION_VER;

                wcsncpy_s((wchar_t*)appName, LWAPI_UNICODE_STRING_MAX, appPath, wcslen(appPath));
                if (LwAPI_DRS_FindApplicationByName(session, appName, &profile, &app) == LWAPI_OK)
                {
                    LWDRS_SETTING lwApiSetting = { 0 };
                    lwApiSetting.version = LWDRS_SETTING_VER;
                    lwApiSetting.settingId = keyID;
                    lwApiSetting.settingType = LWDRS_DWORD_TYPE;
                    lwApiSetting.u32LwrrentValue = value;

                    LwAPI_Status ret = LwAPI_DRS_SetSetting(session, profile, &lwApiSetting);

                    ret = LwAPI_DRS_SaveSettings(session);
                }
                else
                    printf("ERROR: No profile found for application. Key cannot be set.\n");
            }
            LwAPI_DRS_DestroySession(session);
        }
    }
    LwAPI_Unload();
}

// Widestring -> UTF8
void getUtf8FromWstrInternal(const wchar_t * in, std::string & out)
{
    static std::wstring_colwert<std::codecvt_utf8_utf16<wchar_t>> colwerter;
    out = colwerter.to_bytes(in);
}

void ReportProfilesCompatibleWithAnsel()
{
    LwAPI_Initialize();
    {
        LwDRSSessionHandle session;
        if (LwAPI_DRS_CreateSession(&session) == LWAPI_OK)
        {
            if (LwAPI_DRS_LoadSettings(session) == LWAPI_OK)
            {
                LwU32 profileCount = 0;
                if (LwAPI_DRS_GetNumProfiles(session, &profileCount) == LWAPI_OK)
                {
                    for (LwU32 i = 0; i < profileCount; ++i)
                    {
                        LwDRSProfileHandle profile;
                        if (LwAPI_DRS_EnumProfiles(session, i, &profile) == LWAPI_OK)
                        {
                            LWDRS_SETTING lwApiSetting = { 0 };
                            {
                                LwU32 keyID = ANSEL_ALLOWLISTED_ID;
                                lwApiSetting.version = LWDRS_SETTING_VER;
                                lwApiSetting.settingId = keyID;
                                lwApiSetting.settingType = LWDRS_DWORD_TYPE;

                                if (LwAPI_DRS_GetSetting(session, profile, keyID, &lwApiSetting) == LWAPI_OK)
                                {
                                    if (lwApiSetting.u32LwrrentValue == 1)
                                    {
                                        LWDRS_PROFILE profileInfo = {};
                                        profileInfo.version = LWDRS_PROFILE_VER;
                                        if (LwAPI_DRS_GetProfileInfo(session, profile, &profileInfo) == LWAPI_OK)
                                        {
                                            // Colwert to utf8:
                                            std::string profileNameUtf8;
                                            getUtf8FromWstrInternal((const wchar_t*)profileInfo.profileName, profileNameUtf8);
                                            printf("Profile: %s\n", profileNameUtf8.c_str());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            LwAPI_DRS_DestroySession(session);
        }
    }
    LwAPI_Unload();
}





#define ENABLE_BY_DELETING_KEY	0

int main(int argc, char** argv)
{
    const LwU32 anselEnableID = ANSEL_ENABLE_ID;
    const LwU32 anselEnableIDValue = 1;
    const LwU32 anselDisableIDValue = 0;

    const LwU32 anselAllowlistingID = ANSEL_ALLOWLISTED_ID;
    const LwU32 anselAllowlistingIDValue = 1;
    const LwU32 anselAllowlistingAllowValue = 1;
    const LwU32 anselAllowlistingDisallowValue = 0;

    if (argc == 1)
    {
        SaveDRS(anselEnableID, s_isAnselEnabled);
        printf("%d\n", (s_isAnselEnabled ? 1 : 0));
    }
    else if (_stricmp(argv[1], "allowlisting") == 0)
    {
        SaveDRS(anselAllowlistingID, s_isAnselAllowlisting);
        printf(s_isAnselAllowlisting ? "allowlisting-everything" : "allowlisting-default");
    }
    else if (_stricmp(argv[1], "ion") == 0)
    {
#if (ENABLE_BY_DELETING_KEY == 1)
        // TODO: enable instrumentation here
        RestoreDRS(anselEnableID);
#else
        ApplyDRS(anselEnableID, anselEnableIDValue, true);
#endif
    }
    else if (_stricmp(argv[1], "on") == 0)
    {
#if (ENABLE_BY_DELETING_KEY == 1)
        RestoreDRS(anselEnableID);
#else
        ApplyDRS(anselEnableID, anselEnableIDValue);
#endif
    }
    else if (_stricmp(argv[1], "off") == 0)
    {
        ApplyDRS(anselEnableID, anselDisableIDValue);
    }
    else if (_stricmp(argv[1], "allowlisting-ieverything") == 0)
    {
        ApplyDRS(anselAllowlistingID, anselAllowlistingIDValue, true);
    }
    else if (_stricmp(argv[1], "allowlisting-everything") == 0 || _stricmp(argv[1], "we") == 0)
    {
        ApplyDRS(anselAllowlistingID, anselAllowlistingIDValue);
    }
    else if (_stricmp(argv[1], "allowlisting-default") == 0 || _stricmp(argv[1], "wd") == 0)
    {
        RestoreDRS(anselAllowlistingID);
    }
    else if (_stricmp(argv[1], "enable") == 0)
    {
#if (ENABLE_BY_DELETING_KEY == 1)
        RestoreDRS(anselEnableID);
#else
        ApplyDRS(anselEnableID, anselEnableIDValue);
#endif
        printf("%d\n", (s_generalError ? 0 : 1));
    }
    else if (_stricmp(argv[1], "disable") == 0)
    {
        ApplyDRS(anselEnableID, anselDisableIDValue);
        printf("%d\n", (s_generalError ? 0 : 1));
    }
    else if (_stricmp(argv[1], "allowlisting-remove") == 0 && argc == 3 )
    {
        //colwert from ascii to unicode:
        std::wstring_colwert<std::codecvt_utf8_utf16<wchar_t>> colwerter;

        std::wstring appPath = colwerter.from_bytes(argv[2]);
        DeleteKeyFromProfileInApplicationPath(appPath.c_str(), anselAllowlistingID);
    }
    else if (_stricmp(argv[1], "allowlisting-allow") == 0 && argc == 3)
    {
        //colwert from ascii to unicode:
        std::wstring_colwert<std::codecvt_utf8_utf16<wchar_t>> colwerter;

        std::wstring appPath = colwerter.from_bytes(argv[2]);
        SetKeyInProfileInApplicationPath(appPath.c_str(), anselAllowlistingID, anselAllowlistingAllowValue);
    }
    else if (_stricmp(argv[1], "allowlisting-disallow") == 0 && argc == 3)
    {
        //colwert from ascii to unicode:
        std::wstring_colwert<std::codecvt_utf8_utf16<wchar_t>> colwerter;

        std::wstring appPath = colwerter.from_bytes(argv[2]);
        SetKeyInProfileInApplicationPath(appPath.c_str(), anselAllowlistingID, anselAllowlistingDisallowValue);
    }
    else if (_stricmp(argv[1], "allowlisted-profiles") == 0)
    {
        ReportProfilesCompatibleWithAnsel();
    }
    else
    {
        printf("Usage: LwCameraEnable {|on|off|allowlisting|allowlisting-default|allowlisting-everything|allowlisted-profiles}\n");
        printf("       LwCameraEnable {allowlisting-remove|allowlisting-allow|allowlisting-disallow} <path-to-exelwtable-in-utf8-format>\n");
        s_generalError = true;
    }

    return s_generalError ? 1 : 0;
}

