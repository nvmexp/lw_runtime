#pragma once

#include <climits>
#include <cfloat>
#include <vector>
#include <map>
#include <string>
#include <assert.h>

#include "ir/TypeEnums.h"
#include "ir/TypelessVariable.h"

namespace shadermod
{
    namespace ir
    {
        namespace std = ::std;

        template<UserConstDataType uct>
        struct UserConstType
        {
        };

        template<>
        struct UserConstType<UserConstDataType::kBool>
        {
            typedef userConstTypes::Bool type;

            static type getZero() { return userConstTypes::Bool::kFalse; }
            static type getOne() { return userConstTypes::Bool::kTrue; }
            static type getSmallestVal() { return userConstTypes::Bool::kTrue; }
            static type getMilwal() { return userConstTypes::Bool::kFalse; }
            static type getMaxVal() { return userConstTypes::Bool::kTrue; }
        };

        template<>
        struct UserConstType<UserConstDataType::kInt>
        {
            typedef userConstTypes::Int type;

            static type getZero() { return 0; }
            static type getOne() { return 1; }
            static type getSmallestVal() { return  1; }
            static type getMilwal() { return INT_MIN; }
            static type getMaxVal() { return INT_MAX; }
        };

        template<>
        struct UserConstType<UserConstDataType::kUInt>
        {
            typedef userConstTypes::UInt type;

            static type getZero() { return 0; }
            static type getOne() { return 1; }
            static type getSmallestVal() { return 1; }
            static type getMilwal() { return 0; }
            static type getMaxVal() { return UINT_MAX; }
        };

        template<>
        struct UserConstType<UserConstDataType::kFloat>
        {
            typedef userConstTypes::Float type;

            static type getZero() { return 0.0f; }
            static type getOne() { return 1.0f; }
            static type getSmallestVal() { return FLT_MIN; }
            static type getMilwal() { return  -FLT_MAX; }
            static type getMaxVal() { return  FLT_MAX; }
        };

        std::string stringify(const bool& val);
        std::string stringify(const userConstTypes::Bool& val);
        std::string stringify(const userConstTypes::Int& val);
        std::string stringify(const userConstTypes::UInt& val);
        std::string stringify(const userConstTypes::Float& val);
        std::string stringify(unsigned int length, const bool * val);
        std::string stringify(unsigned int length, const userConstTypes::Int * val);
        std::string stringify(unsigned int length, const userConstTypes::UInt * val);
        std::string stringify(unsigned int length, const userConstTypes::Float * val);
        std::string stringify(unsigned int length, const userConstTypes::Bool * val);

        static unsigned int sizeOfUserConstType(UserConstDataType type)
        {
            switch (type)
            {
            case UserConstDataType::kBool:
                return sizeof(userConstTypes::Bool);
            case UserConstDataType::kInt:
                return sizeof(userConstTypes::Int);
            case UserConstDataType::kUInt:
                return sizeof(userConstTypes::UInt);
            case UserConstDataType::kFloat:
                return sizeof(userConstTypes::Float);
            default:
                return 0;
            }
        }

        class UserConstant
        {
        public:
            
            using LocalizedString = std::map<unsigned short, std::string>; //utf8

            struct StringWithLocalization
            {
                StringWithLocalization() = default;
                StringWithLocalization(const std::string& str, const LocalizedString& loc) : m_string(str), m_localization(loc) {}
                const std::string& getDefault() const { return m_string; }
                const std::string& getLocalized(unsigned short langid, bool* found = nullptr) const;

                std::string              m_string; //utf8
                LocalizedString            m_localization;
            };

            struct ListOption
            {
                ListOption() = default;

                ListOption(const TypelessVariable& val, const StringWithLocalization& name):
                    m_value(val),
                    m_name(name)
                {}

                TypelessVariable            m_value;
                StringWithLocalization          m_name;
            };

            struct ListOptions
            {
                ListOptions() = default;

                ListOptions(const std::vector<ListOption>& opts, int defOpt) :
                    m_options(opts), m_defaultOption(defOpt)
                {};

                std::vector<ListOption>    m_options;
                int              m_defaultOption;
            };


            struct IlwalidDataTypeException {};
            struct ValueOutOfBoundException {};
            struct TypesMismatchException {};
            struct IncompatibleOptionsException {};

            UserConstant(
                const std::string & name,
                UserConstDataType type, UiControlType uiControl,
                const TypelessVariable& defaultValue,
                const StringWithLocalization& uiLabel,
                const TypelessVariable& minimumValue, const TypelessVariable& maximumValue,
                const TypelessVariable& uiValueStep, const StringWithLocalization& uiHint,
                const StringWithLocalization& uiValueUnit, float stickyValue, float stickyRegion,
                const TypelessVariable& uiValueMin, const TypelessVariable& uiValueMax, const ListOptions& listOptions,
                const std::vector<StringWithLocalization>& valueDisplayName
                );

            const std::string& getName() const
            {
                return m_name;
            }

            UserConstDataType getType() const
            {
                return m_type;
            }

            unsigned int getTypeDimensions() const
            {
                return m_value.m_dimensions;
            }

            UiControlType getControlType() const
            {
                return m_uiControl;
            }

            const TypelessVariable& getDefaultValue() const
            {
                return m_defaultValue;
            }

            template<typename R> void getDefaultValue(R& ret) const
            {
                m_defaultValue.get(&ret, 1);
            }

            template<typename R> void getDefaultValue(R * ret, unsigned int maxDims) const
            {
                m_defaultValue.get(ret, maxDims);
            }

            const TypelessVariable& getMinimumValue() const
            {
                return m_minimumValue;
            }

            template<typename R> void getMinimumValue(R& ret) const
            {
                m_minimumValue.get(ret);
            }

            const TypelessVariable& getMaximumValue() const
            {
                return m_maximumValue;
            }

            template<typename R> void getMaximumValue(R& ret) const
            {
                m_maximumValue.get(ret);
            }

            const TypelessVariable& getUiValueStep() const
            {
                return m_uiValueStep;
            }

            template<typename R> void getUiValueStep(R& ret) const
            {
                m_uiValueStep.get(&ret, 1);
            }

            template<typename R> void getUiValueStep(R * ret, unsigned int maxDims) const
            {
                m_uiValueStep.get(ret, maxDims);
            }

            const std::string& getUiLabel() const
            {
                return m_uiLabel.m_string;
            }

            const std::string& getUiLabelLocalized(unsigned short langid, bool* found = nullptr) const;
            
            const std::string& getUiHint() const
            {
                return m_uiHint.m_string;
            }

            const std::string& getUiHintLocalized(unsigned short langid, bool* found = nullptr) const;
        
            const std::string& getUiValueUnit() const
            {
                return m_uiValueUnit.m_string;
            }
            
            const std::string& getUiValueUnitLocalized(unsigned short langid, bool* found = nullptr) const;

            float getStickyValue() const
            {
                return m_stickyValue;
            }
            
            float getStickyRegion() const
            {
                return m_stickyRegion;
            }

            const TypelessVariable& getUiValueMin() const
            {
                return m_uiValueMin;
            }

            template<typename R> void getUiValueMin(R& ret) const
            {
                m_uiValueMin.get(&ret, 1);
            }

            template<typename R> void getUiValueMin(R * ret, unsigned int maxDims) const
            {
                m_uiValueMin.get(ret, maxDims);
            }

            const TypelessVariable& getUiValueMax() const
            {
                return m_uiValueMax;
            }

            template<typename R> void getUiValueMax(R& ret) const
            {
                m_uiValueMax.get(&ret, 1);
            }

            template<typename R> void getUiValueMax(R * ret, unsigned int maxDims) const
            {
                m_uiValueMax.get(ret, maxDims);
            }

            const StringWithLocalization& getValueDisplayName(unsigned int index) const
            {
                assert(index < MAX_GROUPED_VARIABLE_DIMENSION);
                return m_valueDisplayName[index];
            }

            unsigned int UserConstant::getNumListOptions() const
            {
                return (unsigned int)m_listOptions.m_options.size();
            }
            
            const TypelessVariable& getListOption(unsigned int idx) const;

            template<typename R> void getListOption(unsigned int idx, R& ret) const
            {
                assert(idx < m_listOptions.m_options.size());
                m_listOptions.m_options[idx].m_value.get(ret);
            }

            const StringWithLocalization & getListOptionLocalization(unsigned int idx) const;
            const std::string & getListOptionName(unsigned int idx) const;
            const std::string & getListOptionNameLocalized(unsigned int idx, unsigned short langid, bool* found = nullptr) const;

            static const int NoDefaultOption = -1;

            int getDefaultListOption() const;

            void setValue(const TypelessVariable& ref);

            template<typename T> void setValue(const T& ref)
            {
                m_value.set(ref);
                m_isDirty = true;
            }

            template<typename T> void setValue(const T * ref, unsigned int dims)
            {
                m_value.set(ref, dims);
                m_isDirty = true;
            }

            const TypelessVariable& getValue() const
            {
                return m_value;
            }

            template<typename R> void getValue(R& ret) const
            {
                m_value.get(&ret, 1);
            }

            template<typename R> void getValue(R * ret, unsigned int maxDims) const
            {
                m_value.get(ret, maxDims);
            }

            unsigned int getPosition() const
            {
                return m_position;
            }

            unsigned int getUid() const
            {
                return m_uid;
            }

            StringWithLocalization      m_uiLabel;
            StringWithLocalization      m_uiHint;
            StringWithLocalization      m_uiValueUnit;

            ListOptions            m_listOptions;

        protected:
            friend class UserConstantManager;

            bool isUpdated() const
            {
                return m_isDirty;
            }

            void markClean()
            {
                m_isDirty = false;
            }

            bool getValue(void* buf, size_t bufSz) const;

            void setPosition(unsigned int idx)
            {
                m_position = idx;
            }

            void setUid(unsigned int idx)
            {
                m_uid = idx;
            }

        public:
            std::string            m_name;
            UserConstDataType        m_type;
            UiControlType          m_uiControl;
            TypelessVariable        m_defaultValue;
            TypelessVariable        m_minimumValue;
            TypelessVariable        m_maximumValue;
            TypelessVariable        m_uiValueStep;
            float              m_stickyValue;
            float              m_stickyRegion;
            TypelessVariable        m_uiValueMin;
            TypelessVariable        m_uiValueMax;
            StringWithLocalization      m_valueDisplayName[MAX_GROUPED_VARIABLE_DIMENSION];

        protected:
            //dynamic fields:
            TypelessVariable        m_value;
            bool              m_isDirty;

            //for the UC manager:
            unsigned int          m_position;
            unsigned int          m_uid;
        };
    }
}


