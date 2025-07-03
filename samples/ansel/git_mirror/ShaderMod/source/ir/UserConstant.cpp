#include <sstream>
#include <iostream>

#include "ir/UserConstant.h"

namespace shadermod
{
namespace ir
{
    namespace std = ::std;

    // TODO: implement each function for higher-dimensional data

    std::wstring wstringify(const userConstTypes::Bool& val)
    {
        std::wstringbuf buf;
        std::wostream strstream(&buf);
        strstream << (val == userConstTypes::Bool::kTrue ? L"True" : L"False");

        return buf.str();
    }

    std::wstring wstringify(const userConstTypes::Int& val)
    {
        std::wstringbuf buf;
        std::wostream strstream(&buf);
        strstream << val;

        return buf.str();
    }

    std::wstring wstringify(const userConstTypes::UInt& val)
    {
        std::wstringbuf buf;
        std::wostream strstream(&buf);
        strstream << val;

        return buf.str();
    }

    std::wstring wstringify(const userConstTypes::Float& val)
    {
        std::wstringbuf buf;
        std::wostream strstream(&buf);
        strstream << val;

        return buf.str();
    }

    std::string stringify(const bool& val)
    {
        std::stringbuf buf;
        std::ostream strstream(&buf);
        strstream << (val ? "True" : "False");

        return buf.str();
    }

    std::string stringify(const userConstTypes::Bool& val)
    {
        std::stringbuf buf;
        std::ostream strstream(&buf);
        strstream << (val == userConstTypes::Bool::kTrue ? "True" : "False");

        return buf.str();
    }

    std::string stringify(const userConstTypes::Int& val)
    {
        std::stringbuf buf;
        std::ostream strstream(&buf);
        strstream << val;

        return buf.str();
    }

    std::string stringify(const userConstTypes::UInt& val)
    {
        std::stringbuf buf;
        std::ostream strstream(&buf);
        strstream << val;

        return buf.str();
    }

    std::string stringify(const userConstTypes::Float& val)
    {
        std::stringbuf buf;
        std::ostream strstream(&buf);
        strstream << val;

        return buf.str();
    }

#define STRINGIFY_ARRAY(varName, length, val) \
    std::string s = #varName "("; \
    for (unsigned int i = 0; i < length; i++) \
    { \
        if (i != 0) s += ", "; \
        s += ir::stringify(val[i]); \
    } \
    s += ")"; \
    return s;

    std::string stringify(unsigned int length, const userConstTypes::Int * val)
    {
        STRINGIFY_ARRAY(int, length, val);
    }

    std::string stringify(unsigned int length, const userConstTypes::UInt * val)
    {
        STRINGIFY_ARRAY(UINT, length, val);
    }

    std::string stringify(unsigned int length, const userConstTypes::Float * val)
    {
        STRINGIFY_ARRAY(float, length, val);
    }

    std::string stringify(unsigned int length, const userConstTypes::Bool * val)
    {
        STRINGIFY_ARRAY(bool, length, val);
    }

    std::string stringify(unsigned int length, const bool * val)
    {
        STRINGIFY_ARRAY(nativeBool, length, val);
    }


    const std::string& UserConstant::StringWithLocalization::getLocalized(unsigned short langid, bool* found) const
    {
        auto it = m_localization.find(langid);
        if (it != m_localization.end())
        {
            if (it->second.length() > 0)
            {
                if (found) *found = true;
                return it->second;
            }
        }

        if (found) *found = false;
        return getDefault();
    }

    UserConstant::UserConstant(const std::string& name,
        UserConstDataType type, UiControlType uiControl,
        const TypelessVariable& defaultValue,
        const StringWithLocalization& uiLabel,
        const TypelessVariable& minimumValue, const TypelessVariable& maximumValue,
        const TypelessVariable& uiValueStep, const StringWithLocalization& uiHint,
        const StringWithLocalization& uiValueUnit, float stickyValue, float stickyRegion,
        const TypelessVariable& uiValueMin, const TypelessVariable& uiValueMax, const ListOptions& listOptions,
        const std::vector<StringWithLocalization>& valueDisplayName):
        m_name(name), m_type(type), m_uiControl(uiControl), m_defaultValue(defaultValue), m_uiLabel(uiLabel),
        m_minimumValue(minimumValue), m_maximumValue(maximumValue), m_uiValueStep(uiValueStep), m_uiHint(uiHint),
        m_uiValueUnit(uiValueUnit), m_stickyValue(stickyValue), m_stickyRegion(stickyRegion),
        m_uiValueMin(uiValueMin), m_uiValueMax(uiValueMax), m_listOptions(listOptions),
        m_position(0xFFffFFff), m_uid(0xFFffFFff)
    {
        assert(valueDisplayName.size() <= MAX_GROUPED_VARIABLE_DIMENSION);
        for (unsigned int i = 0; i < valueDisplayName.size(); i++)
        {
            m_valueDisplayName[i] = valueDisplayName[i];
        }

        setValue(m_defaultValue);
    }
    
    const std::string& UserConstant::getUiLabelLocalized(unsigned short langid, bool* found) const
    {
        return m_uiLabel.getLocalized(langid, found);
    }
    
    const std::string& UserConstant::getUiHintLocalized(unsigned short langid, bool* found) const
    {
        return m_uiHint.getLocalized(langid, found);
    }
        
    const std::string& UserConstant::getUiValueUnitLocalized(unsigned short langid, bool* found) const
    {
        return m_uiValueUnit.getLocalized(langid, found);
    }
    
    const TypelessVariable& UserConstant::getListOption(unsigned int idx) const
    {
        assert(idx < m_listOptions.m_options.size());
        return m_listOptions.m_options[idx].m_value;
    }

    const UserConstant::StringWithLocalization & UserConstant::getListOptionLocalization(unsigned int idx) const
    {
        assert(idx < m_listOptions.m_options.size());

        return m_listOptions.m_options[idx].m_name;
    }

    const std::string& UserConstant::getListOptionName(unsigned int idx) const
    {
        assert(idx < m_listOptions.m_options.size());

        return m_listOptions.m_options[idx].m_name.m_string;
    }

    const std::string& UserConstant::getListOptionNameLocalized(unsigned int idx, unsigned short langid, bool* found) const
    {
        assert(idx < m_listOptions.m_options.size());
        return m_listOptions.m_options[idx].m_name.getLocalized(langid, found);
    }

    int UserConstant::getDefaultListOption() const
    {
        return m_listOptions.m_defaultOption;
    }

    void UserConstant::setValue(const TypelessVariable& ref)
    {
        m_value = ref;
        m_isDirty = true;
    }

    bool UserConstant::getValue(void* buf, size_t bufSz) const
    {
        size_t sz = sizeOfUserConstType(m_type);

        if (sz > bufSz)
            return false;

        switch (m_type)
        {
        case UserConstDataType::kBool:
            m_value.get(reinterpret_cast<userConstTypes::Bool*>(buf), m_defaultValue.getDimensionality());
            break;
        case UserConstDataType::kFloat:
        {
            m_value.get(reinterpret_cast<userConstTypes::Float*>(buf), m_defaultValue.getDimensionality());
            break;
        }
        case UserConstDataType::kUInt:
            m_value.get(reinterpret_cast<userConstTypes::UInt*>(buf), m_defaultValue.getDimensionality());
            break;
        case UserConstDataType::kInt:
            m_value.get(reinterpret_cast<userConstTypes::Int*>(buf), m_defaultValue.getDimensionality());
            break;
        default:
            assert(false && "Unknown user constant data type!");
        }

        return true;
    }
}
}

