#include <RegistrySettings.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <Windows.h>
#include "Log.h"

RegistrySettings::RegistrySettings() :
    m_registryPathAnsel(HKEY_LWRRENT_USER, L"SOFTWARE\\LWPU Corporation\\Ansel"),
    m_registryPathAnselWrite(HKEY_LWRRENT_USER, L"SOFTWARE\\LWPU Corporation\\Ansel", true),
    m_registryPathShadowPlay(HKEY_LWRRENT_USER, L"SOFTWARE\\LWPU Corporation\\Global\\ShadowPlay\\LWSPCAPS"),
    m_registryPathDrsPath(HKEY_LOCAL_MACHINE, L"SOFTWARE\\LWPU Corporation\\Global\\DrsPath"),
    m_registryPathAnselBackup(HKEY_LOCAL_MACHINE, L"SOFTWARE\\LWPU Corporation\\Ansel"),
    m_registryPathAnselWriteBackup(HKEY_LOCAL_MACHINE, L"SOFTWARE\\LWPU Corporation\\Ansel", true)
{
}

const RegistrySettings::RegKeyHolder& RegistrySettings::registryPathAnsel() const { return m_registryPathAnsel; }
const RegistrySettings::RegKeyHolder& RegistrySettings::registryPathAnselWrite() const { return m_registryPathAnselWrite; }
const RegistrySettings::RegKeyHolder& RegistrySettings::registryPathShadowPlay() const { return m_registryPathShadowPlay; }
const RegistrySettings::RegKeyHolder& RegistrySettings::registryPathDrsPath() const { return m_registryPathDrsPath; }
const RegistrySettings::RegKeyHolder& RegistrySettings::registryPathAnselBackup() const { return m_registryPathAnselBackup; }
const RegistrySettings::RegKeyHolder& RegistrySettings::registryPathAnselWriteBackup() const { return m_registryPathAnselWriteBackup; }

DWORD RegistrySettings::WriteRegistryStringW(const HKEY key, const wchar_t* queryString, const std::wstring& inString)
{
    DWORD dwType = REG_SZ;

    const auto retcode = RegSetValueExW(key, queryString, 0, dwType, (const BYTE *)inString.c_str(), (DWORD)(inString.length() * sizeof(wchar_t)));
    if (retcode == ERROR_SUCCESS)
    {
        return 0;
    }

    return -1;
}

bool RegistrySettings::WriteRegistryBoolW(const HKEY key, const wchar_t* queryString, bool value)
{
    std::wstring sValue = value ? L"True" : L"False";
    DWORD status = WriteRegistryStringW(key, queryString, sValue);
    return (status == 0);
}

bool RegistrySettings::WriteRegistryFloatW(const HKEY key, const wchar_t* queryString, float value)
{
    DWORD status = WriteRegistryStringW(key, queryString, std::to_wstring(value));
    return (status == 0);
}

bool RegistrySettings::WriteRegistryIntW(const HKEY key, const wchar_t* queryString, int32_t value)
{
    DWORD status = WriteRegistryStringW(key, queryString, std::to_wstring(value));
    return (status == 0);
}

DWORD RegistrySettings::ReadRegistryStringW(const HKEY key, const wchar_t* queryString, std::wstring& outString)
{
    const DWORD bufferLen = 1024;
    std::vector<wchar_t> bufferW;
    bufferW.resize(bufferLen);

    DWORD dwType = REG_SZ;
    DWORD inoutLen = bufferLen;

    const auto retcode = RegQueryValueExW(key, queryString, NULL, &dwType, (LPBYTE)&bufferW[0], &inoutLen);
    if (retcode == ERROR_SUCCESS)
    {
        // it is theoretically possible for the string to not be null terminated
        // so we assume it is not (all normal editing tools, including regedit will null terminate it)
        // A user would have to call the set value programmatically to accomplish this,
        // but we assume a hacker is doing it.
        // http://stackoverflow.com/questions/32191165/how-can-a-string-value-stored-in-the-registry-not-be-null-terminated

        while (inoutLen > 0 && bufferW[inoutLen - 1] == 0)
            --inoutLen;

        if (inoutLen > 0)
        {
            outString.assign(&bufferW[0], inoutLen);
            return inoutLen;
        }
    }

    return 0;
}

bool RegistrySettings::ReadRegistryFloatW(const HKEY key, const wchar_t* queryString, float& value, float min, float max, float default)
{
    std::wstring sValue;
    if (ReadRegistryStringW(key, queryString, sValue))
    {
        value = (float)_wtof(sValue.c_str());
        if (value >= min && value <= max)
            return true;
    }

    value = default;
    return false;
}

bool RegistrySettings::ReadRegistryIntW(const HKEY key, const wchar_t* queryString, int32_t& value, int32_t min, int32_t max, int32_t default)
{
    std::wstring sValue;
    if (ReadRegistryStringW(key, queryString, sValue))
    {
        value = _wtoi(sValue.c_str());
        if (value >= min && value <= max)
            return true;
    }

    value = default;
    return false;
}

bool RegistrySettings::ReadRegistryDWORD(const HKEY key, const wchar_t* queryString, int32_t& value, int32_t min, int32_t max, int32_t default)
{
    std::wstring sValue;
    if (ReadRegistryStringW(key, queryString, sValue))
    {
        value = _wtoi(sValue.c_str());
        if (value >= min && value <= max)
            return true;
    }

    value = default;
    return false;
}

bool RegistrySettings::ReadRegistryBoolW(const HKEY key, const wchar_t* queryString, bool& value, bool default)
{
    std::wstring sValue;
    if (ReadRegistryStringW(key, queryString, sValue))
    {
        int temp = -1;
        if (sValue == L"True")
            temp = 1;
        else if (sValue == L"False")
            temp = 0;

        if (temp == 1 || temp == 0)
        {
            value = (temp == 1) ? true : false;
            return true;
        }
    }

    value = default;
    return false;
}

// generic settings
bool RegistrySettings::keyExists(const RegKeyHolder& keyPath, const wchar_t* keyName)
{
    if (keyPath)
    {
        RegistrySettings::RegKeyHolder holder(HKEY_LWRRENT_USER, keyPath.path + std::wstring(L"\\") + keyName);
        DWORD dwType = REG_SZ;
        HKEY key = 0;
        auto ret = RegOpenKeyEx(holder.key, keyName, NULL, KEY_QUERY_VALUE, &key);
        if (ret == ERROR_SUCCESS)
        {
            RegCloseKey(key);
            return true;
        }
    }
    return false;
}

const float RegistrySettings::getValue(const RegKeyHolder& keyPath, const std::wstring& keyName, const float& min, const float& max, const float& defaultValue) const
{
    return getValue(keyPath, keyName.c_str(), min, max, defaultValue);
}

const int32_t RegistrySettings::getValue(const RegKeyHolder& keyPath, const std::wstring& keyName, const int32_t& min, const int32_t& max, const int32_t& defaultValue) const
{
    return getValue(keyPath, keyName.c_str(), min, max, defaultValue);
}

const DWORD RegistrySettings::getValue(const RegKeyHolder& keyPath, const std::wstring& keyName, const DWORD& min, const DWORD& max, const DWORD& defaultValue) const
{
    return getValue(keyPath, keyName.c_str(), min, max, defaultValue);
}

const bool RegistrySettings::getValue(const RegKeyHolder& keyPath, const std::wstring& keyName, const bool& defaultValue) const
{
    return getValue(keyPath, keyName.c_str(), defaultValue);
}

const std::wstring RegistrySettings::getValue(const RegKeyHolder& keyPath, const std::wstring& keyName, const wchar_t* defaultValue) const
{
    return getValue(keyPath, keyName.c_str(), defaultValue);
}

const float RegistrySettings::getValue(const RegKeyHolder& keyPath, const wchar_t* keyName, const float& min, const float& max, const float& defaultValue) const
{
    float result = defaultValue;
    if (keyPath)
        ReadRegistryFloatW(keyPath, keyName, result, min, max, defaultValue);
    return result;
}

const int32_t RegistrySettings::getValue(const RegKeyHolder& keyPath, const wchar_t* keyName, const int32_t& min, const int32_t& max, const int32_t& defaultValue) const
{
    int32_t result = defaultValue;
    if (keyPath)
        ReadRegistryIntW(keyPath, keyName, result, min, max, defaultValue);
    return result;
}

const DWORD RegistrySettings::getValue(const RegKeyHolder& keyPath, const wchar_t* keyName, const DWORD& min, const DWORD& max, const DWORD& defaultValue) const
{
    int32_t result = defaultValue;
    if (keyPath)
        ReadRegistryDWORD(keyPath, keyName, result, min, max, defaultValue);
    return static_cast<DWORD>(result);
}

const bool RegistrySettings::getValue(const RegKeyHolder& keyPath, const wchar_t* keyName, const bool& defaultValue) const
{
    bool result = defaultValue;
    if (keyPath)
        ReadRegistryBoolW(keyPath, keyName, result, defaultValue);
    return result;
}

const std::wstring RegistrySettings::getValue(const RegKeyHolder& keyPath, const wchar_t* keyName, const wchar_t* defaultValue) const
{
    std::wstring result = defaultValue;
    if (keyPath)
        ReadRegistryStringW(keyPath, keyName, result);
    return result;
}

void RegistrySettings::setValue(const RegKeyHolder& keyPath, const wchar_t* keyName, bool value)
{
    if (keyPath)
        WriteRegistryBoolW(keyPath, keyName, value);
}

void RegistrySettings::setValue(const RegKeyHolder& keyPath, const wchar_t* keyName, const std::wstring & value)
{
    if (keyPath)
        WriteRegistryStringW(keyPath, keyName, value);
}

// title settings
bool RegistrySettings::valueExists(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName)
{
    if (keyPath)
    {
        RegistrySettings::RegKeyHolder holder(HKEY_LWRRENT_USER, keyPath.path + std::wstring(L"\\") + titleName);
        DWORD dwType = REG_SZ;
        return RegQueryValueExW(holder.key, keyName, NULL, &dwType, NULL, NULL) == ERROR_SUCCESS;
    }
    return false;
}

const float RegistrySettings::getValue(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName, const float& min, const float& max, const float& defaultValue) const
{
    float result = defaultValue;
    if (keyPath)
    {
        RegistrySettings::RegKeyHolder holder(HKEY_LWRRENT_USER, keyPath.path + std::wstring(L"\\") + titleName);
        ReadRegistryFloatW(holder.key, keyName, result, min, max, defaultValue);
    }
    return result;
}

const int32_t RegistrySettings::getValue(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName, const int32_t& min, const int32_t& max, const int32_t& defaultValue) const
{
    int32_t result = defaultValue;
    if (keyPath)
    {
        RegistrySettings::RegKeyHolder holder(HKEY_LWRRENT_USER, keyPath.path + std::wstring(L"\\") + titleName);
        ReadRegistryIntW(holder.key, keyName, result, min, max, defaultValue);
    }
    return result;
}

const bool RegistrySettings::getValue(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName, const bool& defaultValue) const
{
    bool result = defaultValue;
    if (keyPath)
    {
        RegistrySettings::RegKeyHolder holder(HKEY_LWRRENT_USER, keyPath.path + std::wstring(L"\\") + titleName);
        ReadRegistryBoolW(holder.key, keyName, result, defaultValue);
    }
    return result;
}

const std::wstring RegistrySettings::getValue(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName, const wchar_t* defaultValue) const
{
    std::wstring result = defaultValue;
    if (keyPath)
    {
        RegistrySettings::RegKeyHolder holder(HKEY_LWRRENT_USER, keyPath.path + std::wstring(L"\\") + titleName);
        ReadRegistryStringW(holder.key, keyName, result);
    }
    return result;
}

void RegistrySettings::setValue(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName, bool value)
{
    if (keyPath)
    {
        RegistrySettings::RegKeyHolder holder(HKEY_LWRRENT_USER, keyPath.path + std::wstring(L"\\") + titleName, true);
        WriteRegistryBoolW(holder.key, keyName, value);
    }
}

void RegistrySettings::setValue(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName, float value)
{
    if (keyPath)
    {
        RegistrySettings::RegKeyHolder holder(HKEY_LWRRENT_USER, keyPath.path + std::wstring(L"\\") + titleName, true);
        WriteRegistryFloatW(holder.key, keyName, value);
    }
}

void RegistrySettings::setValue(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName, int32_t value)
{
    if (keyPath)
    {
        RegistrySettings::RegKeyHolder holder(HKEY_LWRRENT_USER, keyPath.path + std::wstring(L"\\") + titleName, true);
        WriteRegistryIntW(holder.key, keyName, value);
    }
}

void RegistrySettings::setValue(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName, const std::wstring& value)
{
    if (keyPath)
    {
        RegistrySettings::RegKeyHolder holder(HKEY_LWRRENT_USER, keyPath.path + std::wstring(L"\\") + titleName, true);
        WriteRegistryStringW(holder.key, keyName, value);
    }
}

bool RegistrySettings::createKey(const RegKeyHolder& keyPath, const wchar_t* keyName, DWORD& disposition)
{
    if (keyPath.key)
    {
        HKEY key = NULL;
        const auto ret = RegCreateKeyExW(keyPath.key, keyName, 0, NULL, REG_OPTION_NON_VOLATILE, KEY_SET_VALUE, NULL, &key, &disposition);
        if (ret != ERROR_SUCCESS)
        {
            LOG_ERROR("Can't create registry key '%s'", keyName);
            return false;
        }
        else
        {
            if (key)
            {
                RegCloseKey(key);
            }
            return true;
        }
    }
    return false;
}

bool RegistrySettings::onRegistryKeyChanged(const std::function<void()>& func)
{
    if (m_notifyEvent == 0)
    {
        if ((m_notifyEvent = CreateEvent(NULL, TRUE, FALSE, NULL)) == NULL)
        {
            return false;
        }

        if (RegNotifyChangeKeyValue(m_registryPathAnsel,
            TRUE,
            REG_NOTIFY_CHANGE_NAME | REG_NOTIFY_CHANGE_LAST_SET | REG_NOTIFY_CHANGE_ATTRIBUTES,
            m_notifyEvent,
            TRUE) != ERROR_SUCCESS)
        {
            return false;
        }
    }

    m_callback = func;
    return true;
}

void RegistrySettings::tick()
{
    if (!m_callback)
    {
        return;
    }

    DWORD retcode = WaitForSingleObject(m_notifyEvent, 0);
    if (retcode == WAIT_OBJECT_0)
    {
        m_callback();
        if (RegNotifyChangeKeyValue(m_registryPathAnsel,
            TRUE,
            REG_NOTIFY_CHANGE_NAME | REG_NOTIFY_CHANGE_LAST_SET | REG_NOTIFY_CHANGE_ATTRIBUTES,
            m_notifyEvent,
            TRUE) != ERROR_SUCCESS)
        {
            // TODO: handle this somehow
        }
    }
}

RegistrySettings::~RegistrySettings()
{
    if (m_notifyEvent != 0)
    {
        CloseHandle(m_notifyEvent);
        m_notifyEvent = 0;
    }

    m_callback = nullptr;
}

RegistrySettings::RegKeyHolder::RegKeyHolder(HKEY category, const std::wstring& _path, bool forWriting)
    : path(_path)
{
    REGSAM samDesired = forWriting ? KEY_WRITE : KEY_READ;
    LONG result = RegOpenKeyExW(category, path.c_str(), 0, samDesired, &key);
    if (result != ERROR_SUCCESS && isLogInitialized())
    {
        LOG_WARN("Unable to open registry key, error code %d", result);
    }
}
RegistrySettings::RegKeyHolder::~RegKeyHolder()
{
    RegCloseKey(key);
}

RegistrySettings::RegKeyHolder::operator HKEY() const
{
    return key;
}

RegistrySettings::RegKeyHolder::operator bool() const
{
    return key != 0;
}

