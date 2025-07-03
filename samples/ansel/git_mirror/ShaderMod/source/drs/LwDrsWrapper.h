#pragma once
#include <string>

namespace drs
{
    bool getProfileName(std::wstring& profileName);
    bool getKeyValue(uint32_t keyID, uint32_t& keyValue, bool readGold = false);
    bool getKeyValueString(uint32_t keyID, std::wstring& keyValueString, bool readGold = false);
}

