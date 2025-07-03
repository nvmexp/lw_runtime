#include <iostream>
#include "yaml-cpp/yaml.h"
#include "MultipassConfigParserError.h"

namespace shadermod
{
    const char* MultipassConfigParserErrorMessage[] =
    {
        "Sucess",
        "Yaml parser failed! Parser error: ",
        "Can't find the entry point (main node)",
        "Import sequence contains non-string entry! Node: ",
        "Can't find imported node: ",
        "Property name should be a string: ",
        "Relwrsive import forbidden. Relwrsion detected while importing node ",
        "The node is expected to be a map! Node at: ",
        "The property is unexpected in this context: ",
        "The index is out of range: ",
        "The node is expected to be a scalar! Node at: ",
        "The property is expected to be a strictly positive integer! Node at: ",
        "The property is expected to be a valid enumeration value! Node at: ",
        "A required property is missing! Property: ",
        "This constant is unknown (neither system-provided nor user-tweakable)! Constant: ",
        "Illegal shader filename! Shader filename should be raltive to the config and, \"..\\ \" is not allowed! Filename:",
        "The node should have non-zero content! Node: \n",
        "Internal error: ",
        "Shader compiler returned error: ",
        "Attempting to bind a variable to a slot that is already in use: ",
        "Failed to associate a variable refrenced in the config with the HLSL code: ",
        "Can't open file for reading: ",
        "Illegal character (TAB) found in YAML config at: ",
        "Undeclared user constant referenced: ",
        "Importing this node is not allowed: ", 
        "The node is expected to be a sequence! Node at: ",
        "Invalid boolean value: ",
        "Invalid integer: ",
        "Invalid unsigned integer: ",
        "Invalid floating-point value: ",
        "The value has to be one of the options provided: ",
        "The value is out of the allowed range: ",
        "The value is not valid: ",
        "Creating user consant failed (duplicate name?): ",
        "Duplicate name not allowed: ",
    };

    static_assert(sizeof(MultipassConfigParserErrorMessage) / sizeof(const char*) == (int)MultipassConfigParserErrorEnum::NUM_VALUES, "MultipassConfigParserErrorMessage doesn't have all error messgaes!");

    MultipassConfigParserError::MultipassConfigParserError(MultipassConfigParserErrorEnum code, const YAML::Mark& mark) : m_errCode(code)
    {
        std::stringbuf buf;
        std::ostream strstream(&buf);
        strstream << "position " << mark.pos << " line " << mark.line << " column " << mark.column;

        m_errMsg = buf.str();
    }

    MultipassConfigParserError::MultipassConfigParserError(MultipassConfigParserErrorEnum code, const std::string& msg, const YAML::Mark& mark) : m_errCode(code)
    {
        std::stringbuf buf;
        std::ostream strstream(&buf);
        strstream << msg << " at position " << mark.pos << " line " << mark.line << " column " << mark.column;

        m_errMsg = buf.str();
    }
}
