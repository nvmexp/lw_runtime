#define ANSEL_SDK_EXPORTS
#include <ansel/UserControls.h>
#include <array>
#include <mutex>
#include <vector>
#include <string>
#include <codecvt>

#include <Windows.h>

namespace ansel
{
    namespace
    {
        struct LocalizedLabel
        {
            std::string lang; //'en-US', 'ru-RU', etc
            std::string labelUtf8;
        };

        union UserControlValue
        {
            float floatValue;
            bool boolValue;
        };

        struct UserControlDescInternal
        {
            UserControlInfo info;
            std::vector<LocalizedLabel> labels;
            UserControlCallback callback = nullptr;
            UserControlValue value;
        };

        const std::array<char, 3> s_disallowedLabelCharacters = { '\n', '\t', '\r' };
        const std::array<UserControlType, 2> s_supportedControlTypes = { kUserControlSlider, kUserControlBoolean };
        bool s_userControlDescriptionListIsDirty = false;
        const size_t s_labelLengthLimit = 20u;

        // dummy user control to return a reference to in case needed desc is not found
        UserControlDescInternal s_dummy;
        std::mutex s_cachedUserControlDescriptionsMutex;
        std::vector<UserControlDescInternal> s_cachedUserControlDescriptions;
        std::vector<const char*> s_localizedNamePointers;

        bool isIdCached(uint32_t id)
        {
            for (const auto& desc : s_cachedUserControlDescriptions)
                if (desc.info.userControlId == id)
                    return true;
            return false;
        }

        UserControlDescInternal& getUserControlDescById(uint32_t id)
        {
            for (auto& desc : s_cachedUserControlDescriptions)
                if (desc.info.userControlId == id)
                    return desc;
            return s_dummy;
        }

        bool isLabelValid(const char* labelUtf8)
        {
            if (!labelUtf8)
                return false;

            const auto labelLength = strlen(labelUtf8);
            if (labelLength == 0 || labelLength > s_labelLengthLimit)
                return false;

            // scan for disallowed characters
            for (size_t i = 0u; i < labelLength; ++i)
                for (auto c : s_disallowedLabelCharacters)
                    if (labelUtf8[i] == c)
                        return false;

            return true;
        }
    }

    UserControlStatus addUserControl(const UserControlDesc& desc)
    {
        // Add high quality game setting toggle.
        // We will remove it later if the changeQualityCallback remains undefined.
        if (s_cachedUserControlDescriptions.empty() && desc.info.userControlId != HIGH_QUALITY_CONTROL_ID)
        {
            UserControlDesc highQualityToggleDesc;
            highQualityToggleDesc.labelUtf8 = "High Quality";
            highQualityToggleDesc.info.userControlId = HIGH_QUALITY_CONTROL_ID;
            highQualityToggleDesc.info.userControlType = ansel::kUserControlBoolean;
            static bool s_highQualityEnabled;
            highQualityToggleDesc.info.value = &s_highQualityEnabled;
            highQualityToggleDesc.callback = [](const ansel::UserControlInfo& info) {
                s_highQualityEnabled = *reinterpret_cast<const float*>(info.value);
            };
            addUserControl(highQualityToggleDesc);
        }

        if (!desc.labelUtf8)
            return kUserControlIlwalidLabel;

        if (std::find(s_supportedControlTypes.cbegin(), s_supportedControlTypes.cend(), desc.info.userControlType) == s_supportedControlTypes.cend())
            return kUserControlIlwalidType;

        if (!isLabelValid(desc.labelUtf8))
            return kUserControlIlwalidLabel;

        if (isIdCached(desc.info.userControlId))
            return kUserControlIdAlreadyExists;

        if (!desc.info.value)
            return kUserControlIlwalidValue;

        UserControlDescInternal descInternal;

        if (desc.info.userControlType == kUserControlSlider)
        {
            descInternal.value.floatValue = *reinterpret_cast<const float*>(desc.info.value);
            if (!(std::isfinite(descInternal.value.floatValue) && 
                descInternal.value.floatValue >= 0.0f && 
                descInternal.value.floatValue <= 1.0f))
                return kUserControlIlwalidValue;
        }
        else if (desc.info.userControlType == kUserControlBoolean)
            descInternal.value.boolValue = *reinterpret_cast<const bool*>(desc.info.value);

        descInternal.info = desc.info;
        descInternal.callback = desc.callback;
        descInternal.labels = { { "en-US", desc.labelUtf8 } };

        {
            std::lock_guard<std::mutex> lock(s_cachedUserControlDescriptionsMutex);
            s_cachedUserControlDescriptions.push_back(descInternal);
        }
        s_userControlDescriptionListIsDirty = true;
        return kUserControlOk;
    }

    UserControlStatus removeUserControl(uint32_t userControlId)
    {
        std::lock_guard<std::mutex> lock(s_cachedUserControlDescriptionsMutex);

        for (auto& it = s_cachedUserControlDescriptions.begin(); it != s_cachedUserControlDescriptions.end(); ++it)
        {
            if (it->info.userControlId == userControlId)
            {
                s_cachedUserControlDescriptions.erase(it);
                s_userControlDescriptionListIsDirty = true;
                return kUserControlOk;
            }
        }

        return kUserControlIlwalidId;
    }

    UserControlStatus setUserControlLabelLocalization(uint32_t userControlId, const char* lang, const char* labelUtf8)
    {
        auto& desc = getUserControlDescById(userControlId);
        if (!desc.callback)
            return kUserControlIlwalidId;
        if (!lang)
            return kUserControlIlwalidLocale;
        if (!labelUtf8)
            return kUserControlIlwalidLabel;

        // verify label
        if(!isLabelValid(labelUtf8))
            return kUserControlIlwalidLabel;

        // verify lang
        std::wstring langid;
        std::wstring_colwert<std::codecvt_utf8_utf16<wchar_t>> colwerter;

        try
        {
            langid = colwerter.from_bytes(lang);
        }
        catch (...)
        {
            return kUserControlIlwalidLocale;
        }

        if (IsValidLocaleName(langid.c_str()))
        {
            std::lock_guard<std::mutex> lock(s_cachedUserControlDescriptionsMutex);
            desc.labels.push_back({ lang, labelUtf8 });
        }
        else
            return kUserControlIlwalidLocale;

        s_userControlDescriptionListIsDirty = true;
        return kUserControlOk;
    }

    UserControlStatus getUserControlValue(uint32_t userControlId, void* value)
    {
        for (auto& desc : s_cachedUserControlDescriptions)
            if (desc.info.userControlId == userControlId)
            {
                if (desc.info.userControlType == UserControlType::kUserControlBoolean)
                {
                    *static_cast<bool*>(value) = desc.value.boolValue;
                    return kUserControlOk;
                }
                else if (desc.info.userControlType == UserControlType::kUserControlSlider)
                {
                    *static_cast<float*>(value) = desc.value.floatValue;
                    return kUserControlOk;
                }
                else
                    return kUserControlIlwalidType;
            }
        return kUserControlIlwalidId;
    }

    UserControlStatus setUserControlValue(uint32_t userControlId, const void* value)
    {
        for (auto& desc : s_cachedUserControlDescriptions)
            if (desc.info.userControlId == userControlId)
            {
                if (desc.info.userControlType == UserControlType::kUserControlBoolean)
                {
                    desc.value.boolValue = *static_cast<const bool*>(value);
                    s_userControlDescriptionListIsDirty = true;
                    return kUserControlOk;
                }
                else if (desc.info.userControlType == UserControlType::kUserControlSlider)
                {
                    desc.value.floatValue = *static_cast<const float*>(value);
                    s_userControlDescriptionListIsDirty = true;
                    return kUserControlOk;
                }
                else
                    return kUserControlIlwalidType;
            }
        return kUserControlIlwalidId;
    }

    // These functions gets called by the driver.

    ANSEL_SDK_INTERNAL_API uint32_t getUserControlDescriptionsSize()
    {
        std::lock_guard<std::mutex> lock(s_cachedUserControlDescriptionsMutex);
        return uint32_t(s_cachedUserControlDescriptions.size());
    }

    ANSEL_SDK_INTERNAL_API void lockUserControlDescriptions()
    {
        s_cachedUserControlDescriptionsMutex.lock();
    }

    ANSEL_SDK_INTERNAL_API void unlockUserControlDescriptions()
    {
        s_cachedUserControlDescriptionsMutex.unlock();
    }

    ANSEL_SDK_INTERNAL_API void getUserControlDescription(uint32_t index,
        UserControlType& controlType,
        const char**& labels,
        uint32_t& labelsCount,
        void*& userControlValue,
        uint32_t& userControlId)
    {
        if (index < s_cachedUserControlDescriptions.size())
        {
            auto& desc = s_cachedUserControlDescriptions[index];
            controlType = desc.info.userControlType;
            userControlId = desc.info.userControlId;
            userControlValue = &desc.value;

            s_localizedNamePointers = decltype(s_localizedNamePointers)();
            for (const auto& label : desc.labels)
            {
                s_localizedNamePointers.push_back(label.lang.c_str());
                s_localizedNamePointers.push_back(label.labelUtf8.c_str());
            }
            labels = s_localizedNamePointers.data();
            labelsCount = uint32_t(s_localizedNamePointers.size());
        }
    }

    ANSEL_SDK_INTERNAL_API bool isUserControlDescListDirty()
    {
        return s_userControlDescriptionListIsDirty;
    }

    ANSEL_SDK_INTERNAL_API void clearUserControlDescListDirtyFlag()
    {
        s_userControlDescriptionListIsDirty = false;
        s_localizedNamePointers = decltype(s_localizedNamePointers)();
    }

    ANSEL_SDK_INTERNAL_API void userControlValueChanged(uint32_t id, const void* value)
    {
        if (value)
        {
            for (auto& desc : s_cachedUserControlDescriptions)
                if (desc.info.userControlId == id)
                {
                    if (desc.info.userControlType == kUserControlBoolean)
                        desc.value.boolValue = *reinterpret_cast<const bool*>(value);
                    else if (desc.info.userControlType == kUserControlSlider)
                        desc.value.floatValue = *reinterpret_cast<const float*>(value);
                    desc.info.value = &desc.value;
                    if (desc.callback)
                        desc.callback(desc.info);
                }
        }
    }

    ANSEL_SDK_INTERNAL_API UserControlStatus addUserControl_Internal(const UserControlDesc& desc)
    {
        return addUserControl(desc);
    }

    ANSEL_SDK_INTERNAL_API UserControlStatus setUserControlLabelLocalization_Internal(uint32_t userControlId, const char* lang, const char* labelUtf8)
    {
        return setUserControlLabelLocalization(userControlId, lang, labelUtf8);
    }

    ANSEL_SDK_INTERNAL_API UserControlStatus removeUserControl_Internal(uint32_t userControlId)
    {
        return removeUserControl(userControlId);
    }

    ANSEL_SDK_INTERNAL_API UserControlStatus getUserControlValue_Internal(uint32_t userControlId, void* value)
    {
        return getUserControlValue(userControlId, value);
    }

    ANSEL_SDK_INTERNAL_API UserControlStatus setUserControlValue_Internal(uint32_t userControlId, const void* value)
    {
        return setUserControlValue(userControlId, value);
    }
}
