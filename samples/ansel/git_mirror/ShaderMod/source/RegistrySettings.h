#pragma once
#include <string>
#include <vector>
#include <functional>
#include <cstdint>
#include <Windows.h>

namespace Settings
{
    constexpr const char* SnapshotDir = "SnapshotsDir";
    constexpr const char* SnapshotShadowplayDir = "DefaultPathW";
    constexpr const char* StylesDir = "UserStylesDir";
    constexpr const char* IntermediateDir = "IntermediateShotsDir";
    constexpr const char* MaxHighRes = "MaxHighRes";
    constexpr const char* MaxSphereRes = "MaxSphereRes";
    constexpr const char* StereoEyeSeparation = "StereoEyeSeparation";
    constexpr const char* CameraSpeedMult = "CameraSpeedMult";
    constexpr const char* IPCenabled = "IPCenabled";
    constexpr const char* RemoveBlackTint = "RemoveBlackTint";
    constexpr const char* KeepIntermediateShots = "KeepIntermediateShots";
    constexpr const char* EnableStyleTransfer = "EnableStyleTransfer";
    constexpr const char* AllowStyleTransferWhileMoving = "AllowStyleTransferWhileMoving";
    constexpr const char* RenderDebugInformation = "RenderDebugInformation";
    constexpr const char* LosslessOutput = "LosslessOutput";
    constexpr const char* LosslessOutput360 = "LosslessOutput360";
    constexpr const char* AllowNotifications = "AllowNotifications";
    constexpr const char* EnableEnhancedHighres = "EnableEnhancedHighres";
    constexpr const char* EnhancedHighresCoeff = "EnhancedHighresCoeff";
    constexpr const char* UseHybridController = "UseHybridController";
    constexpr const char* AllowTelemetry = "AllowTelemetry";
    constexpr const char* RequireSDK = "RequireSDK";
    constexpr const char* FiltersInGame = "FiltersInGame";
    constexpr const char* StandaloneModding = "StandaloneModding";
    constexpr const char* DynamicFilterStacking = "DynamicFilterStacking";
    constexpr const char* ForceLang = "ForceLang";
    constexpr const char* ToggleHotkeyModCtrl = "ToggleHotkeyModCtrl";
    constexpr const char* ToggleHotkeyModShift = "ToggleHotkeyModShift";
    constexpr const char* ToggleHotkeyModAlt = "ToggleHotkeyModAlt";
    constexpr const char* ToggleHotkey = "ToggleHotkey";
    constexpr const char* ModsEnabledCheck = "ModsEnabledCheck";
    constexpr const char* CheckTraficLocal = "CheckTraficLocal";
    constexpr const char* EffectFolders = "EffectFolders";
    constexpr const char* FreestyleEnabled = "FreestyleEnabled";
    constexpr const char* OutputPresets = "SavePresetWithShot";
    constexpr const char* AllowBufferOptionsFilter = "AllowBufferOptionsFilter";
    constexpr const char* SaveCaptureAsPhotoShop = "SaveCaptureAsPhotoShop";
}

// this class is intended to work from a single thread
class RegistrySettings
{
private:
    struct RegKeyHolder
    {
        HKEY key = 0;
        std::wstring path;
        RegKeyHolder(HKEY category, const std::wstring& path, bool forWriting = false);
        RegKeyHolder(const RegKeyHolder& other) = delete;
        ~RegKeyHolder();
        operator HKEY() const;
        operator bool() const;
    };

    RegKeyHolder m_registryPathAnsel, m_registryPathShadowPlay, m_registryPathDrsPath;
    RegKeyHolder m_registryPathAnselWrite;
    RegKeyHolder m_registryPathAnselBackup, m_registryPathAnselWriteBackup;

    HANDLE m_notifyEvent = 0;
    std::function<void()> m_callback;

    bool m_isDirty = false;

    static DWORD WriteRegistryStringW(const HKEY key, const wchar_t* queryString, const std::wstring& inString);
    static bool WriteRegistryBoolW(const HKEY key, const wchar_t* queryString, bool value);
    static bool WriteRegistryIntW(const HKEY key, const wchar_t* queryString, int32_t inString);
    static bool WriteRegistryFloatW(const HKEY key, const wchar_t* queryString, float value);

    static DWORD ReadRegistryStringW(const HKEY key, const wchar_t* queryString, std::wstring& outString);
    static bool ReadRegistryFloatW(const HKEY key, const wchar_t* queryString, float& value, float min, float max, float default);
    static bool ReadRegistryIntW(const HKEY key, const wchar_t* queryString, int32_t & value, int32_t min, int32_t max, int32_t default);
    static bool ReadRegistryDWORD(const HKEY key, const wchar_t* queryString, int32_t & value, int32_t min, int32_t max, int32_t default);
    static bool ReadRegistryBoolW(const HKEY key, const wchar_t* queryString, bool& value, bool default);

public:
    RegistrySettings();
    ~RegistrySettings();

    bool onRegistryKeyChanged(const std::function<void()>& func);

    const RegKeyHolder& registryPathAnsel() const;
    const RegKeyHolder& registryPathAnselWrite() const;
    const RegKeyHolder& registryPathShadowPlay() const;
    const RegKeyHolder& registryPathDrsPath() const;
    const RegKeyHolder& registryPathAnselBackup() const;
    const RegKeyHolder& registryPathAnselWriteBackup() const;

    void markDirty() { m_isDirty = true; }
    void clearDirty() { m_isDirty = false; }
    bool isDirty() const { return m_isDirty; }

    void tick();

    // generic settings
    bool createKey(const RegKeyHolder& keyPath, const wchar_t* keyName, DWORD& disposition);
    bool keyExists(const RegKeyHolder& keyPath, const wchar_t* keyName);
    const float getValue(const RegKeyHolder& keyPath, const wchar_t* keyName, const float& min, const float& max, const float& defaultValue) const;
    const int32_t getValue(const RegKeyHolder& keyPath, const wchar_t* keyName, const int32_t& min, const int32_t& max, const int32_t& defaultValue) const;
    const DWORD getValue(const RegKeyHolder& keyPath, const wchar_t* keyName, const DWORD& min, const DWORD& max, const DWORD& defaultValue) const;
    const bool getValue(const RegKeyHolder& keyPath, const wchar_t* keyName, const bool& defaultValue) const;
    const std::wstring getValue(const RegKeyHolder& keyPath, const wchar_t* keyName, const wchar_t* defaultValue) const;
    const float getValue(const RegKeyHolder& keyPath, const std::wstring& keyName, const float& min, const float& max, const float& defaultValue) const;
    const int32_t getValue(const RegKeyHolder& keyPath, const std::wstring& keyName, const int32_t& min, const int32_t& max, const int32_t& defaultValue) const;
    const DWORD getValue(const RegKeyHolder& keyPath, const std::wstring& keyName, const DWORD& min, const DWORD& max, const DWORD& defaultValue) const;
    const bool getValue(const RegKeyHolder& keyPath, const std::wstring& keyName, const bool& defaultValue) const;
    const std::wstring getValue(const RegKeyHolder& keyPath, const std::wstring& keyName, const wchar_t* defaultValue) const;
    void setValue(const RegKeyHolder& keyPath, const wchar_t* keyName, bool value);
    void setValue(const RegKeyHolder& keyPath, const wchar_t* keyName, const std::wstring& value);
    // title settings
    bool valueExists(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName);

    const float getValue(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName, const float& min, const float& max, const float& defaultValue) const;
    const int32_t getValue(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName, const int32_t& min, const int32_t& max, const int32_t& defaultValue) const;
    const bool getValue(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName, const bool& defaultValue) const;
    const std::wstring getValue(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName, const wchar_t* defaultValue) const;

    void setValue(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName, bool value);
    void setValue(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName, float value);
    void setValue(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName, int32_t value);
    void setValue(const RegKeyHolder& keyPath, const wchar_t* titleName, const wchar_t* keyName, const std::wstring& value);
};
