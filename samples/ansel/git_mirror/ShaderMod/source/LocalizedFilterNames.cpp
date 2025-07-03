#pragma once

#include <iostream>
#include <fstream>

#include <toml.h>

#include <darkroom\StringColwersion.h>

#include "LocalizedFilterNames.h"

namespace shadermod
{

    void LocalizedEffectNamesParser::parseInternal(const wchar_t * filename, LocalizedConfigFileParsedData * configFileParsedData)
    {
        if (!filename || !configFileParsedData)
            return;

        LocalizedConfigFileParsedData & cfgData = *configFileParsedData;

        std::ifstream ifs(filename);
        if (!ifs)
        {
            return;
        }

        toml::ParseResult pr = toml::parse(ifs);

        if (!pr.valid())
        {
            //std::cout << pr.errorReason << std::endl;
            return;
        }

        toml::Value * v = &pr.value;
        const toml::Table & topLevelTable = v->as<toml::Table>();

        size_t filterIdx = 0;
        for (auto it = topLevelTable.begin(); it != topLevelTable.end(); ++it)
        {
            if (it->second.type() != toml::Value::TABLE_TYPE)
                continue;

            const std::wstring filterName = darkroom::getWstrFromUtf8(it->first);

            cfgData.m_filenamesToFilterIds.insert(std::make_pair(filterName, (unsigned int)filterIdx));

            cfgData.m_filterNamesLocalized.push_back(std::map<WORD, std::wstring>());
            std::map<WORD, std::wstring> & filterLocalizationMap = cfgData.m_filterNamesLocalized[filterIdx];

            const toml::Table & filterLocalizationTable = it->second.as<toml::Table>();
            for (auto locIt = filterLocalizationTable.begin(); locIt != filterLocalizationTable.end(); ++locIt)
            {
                if (locIt->second.type() != toml::Value::STRING_TYPE)
                    continue;

                std::wstring localeName(darkroom::getWstrFromUtf8(locIt->first));
                std::wstring localizedName(darkroom::getWstrFromUtf8(locIt->second.as<std::string>()));

                darkroom::tolowerInplace(localeName);

                if (localeName == L"default")
                {
                    // In case "default" will be present several times
                    if (cfgData.m_filterNames.size() <= filterIdx)
                        cfgData.m_filterNames.push_back(localizedName);

                    continue;
                }

                // colwert deprecated zh-chs and zh-cht to zh-cn and zh-tw respectively
                if (localeName == L"zh-chs")
                    localeName = L"zh-cn";
                else if (localeName == L"zh-cht")
                    localeName = L"zh-tw";

                LCID localeId = LocaleNameToLCID(localeName.c_str(), 0);
                if (localeId == 0)
                    continue;

                WORD langId = LANGIDFROMLCID(localeId);

                filterLocalizationMap.insert(std::make_pair(langId, localizedName));
            }

            ++filterIdx;
        }
    }

    void LocalizedEffectNamesParser::reset()
    {
        for (auto cfgPathToParsedData : m_cfgPathToParsedData)
        {
            cfgPathToParsedData.second.reset();
        }
        m_cfgPathToParsedData.clear();
    }


    void LocalizedEffectNamesParser::parseSingleFileInternal(const wchar_t * path, const wchar_t * filename)
    {
        std::wstring filepath = std::wstring(path) + filename;
        auto cfgParsedDataPair = m_cfgPathToParsedData.insert(std::make_pair(path, LocalizedConfigFileParsedData()));
        // cfgParsedDataPair.first contains iterator to the inserted (or existing) pair
        parseInternal(filepath.c_str(), &cfgParsedDataPair.first->second);
    }
    void LocalizedEffectNamesParser::parseSingleFile(const wchar_t * path, const wchar_t * filename)
    {
        __try
        {
            parseSingleFileInternal(path, filename);
        }
        __except (EXCEPTION_EXELWTE_HANDLER)
        {
            return;
        }
    }

    bool LocalizedEffectNamesParser::getFilterName(const std::wstring & path, const std::wstring & filename, std::wstring & name) const
    {
        auto itParsedData = m_cfgPathToParsedData.find(path);

        if (itParsedData == m_cfgPathToParsedData.end())
            return false; // config file in this path didn't exist

        const LocalizedConfigFileParsedData & cfgData = itParsedData->second;

        auto it = cfgData.m_filenamesToFilterIds.find(filename);

        if (it == cfgData.m_filenamesToFilterIds.end())
            return false; //this filename isn't on the list

        unsigned int idx = it->second;

        if (idx >= cfgData.m_filterNames.size())
        {
            assert(false); //should never be a valid state of the object

            return false;
        }

        name = cfgData.m_filterNames[idx]; //fallback to default language

        return true;
    }

    bool LocalizedEffectNamesParser::getFilterNameLocalized(const std::wstring & path, const std::wstring & filename, std::wstring & name, WORD langID) const
    {
        auto itParsedData = m_cfgPathToParsedData.find(path);

        if (itParsedData == m_cfgPathToParsedData.end())
            return false; // config file in this path didn't exist

        const LocalizedConfigFileParsedData & cfgData = itParsedData->second;

        assert(cfgData.m_filterNames.size() == cfgData.m_filterNamesLocalized.size());

        auto it = cfgData.m_filenamesToFilterIds.find(filename);

        if (it == cfgData.m_filenamesToFilterIds.end())
            return false; //this filename isn't on the list

        unsigned int idx = it->second;

        if (idx >= cfgData.m_filterNamesLocalized.size())
        {
            assert(false); //should never be a valid state of the object

            return false;
        }

        const auto& langMap = cfgData.m_filterNamesLocalized[idx];
        auto lit = langMap.find(langID);

        if (lit != langMap.end())
        {
            name = lit->second;

            if (name.length() > 0)
            {
                return true;
            }
        }

        name = cfgData.m_filterNames[idx]; //fallback to default language

        return true;
    }

}
