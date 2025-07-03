#pragma once

#include <string>

namespace YAML
{
    struct Mark;
}

namespace shadermod
{
    enum struct MultipassConfigParserErrorEnum
    {
        eOK = 0,
        eYAMLError,
        eMainNotFound,
        eBadImportDirective,
        eParentNotFound,
        eIlwalidProperty,
        eRelwrsiveImport,
        eNodeNotMap,
        eUnexpectedProperty,
        eIndexOutOfRange,
        eNodeNotScalar,
        eIlwalidPositiveInt,
        eIlwalidEnumValue,
        eRequiredPropertyMissing,
        eUnknownConstant,
        eIllegalShaderPath,
        eEmptyMapNotAllowed,
        eInternalError,
        eShaderCompilationError,
        eDuplicateBindingSlot,
        eReflectionFailed,
        eFileNotfound,
        eTabInYamlDetected,
        eBadUserConstant,
        eIllegalParent,
        eNodeNotSequence,
        eIlwalidBool,
        eIlwalidInt,
        eIlwalidUInt,
        eIlwalidFloat,
        eUserConstantValueNotOneOfOptions,
        eValueOutOfRange,
        eIlwalidValue,
        eCreateUserConstantFailed,
        eDuplicateNameNotAllowed,
        NUM_VALUES
    };

    extern const char* MultipassConfigParserErrorMessage[];

    struct MultipassConfigParserError
    {
        MultipassConfigParserError(MultipassConfigParserErrorEnum code, const std::string& msg = std::string()) : m_errCode(code), m_errMsg(msg) {}
        MultipassConfigParserError(MultipassConfigParserErrorEnum code, const YAML::Mark& mark);
        MultipassConfigParserError(MultipassConfigParserErrorEnum code, const std::string& msg, const YAML::Mark& mark);

        std::string getFullErrorMessage() const
        {
            return std::string(MultipassConfigParserErrorMessage[(int)m_errCode]) + m_errMsg;
        }

        operator bool()
        {
            return m_errCode != MultipassConfigParserErrorEnum::eOK;
        }

        MultipassConfigParserErrorEnum m_errCode;
        std::string m_errMsg;
    };
}
