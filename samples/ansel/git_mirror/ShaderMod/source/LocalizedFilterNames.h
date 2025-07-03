#pragma once

#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <Windows.h>
#include <assert.h>

#include "darkroom\StringColwersion.h"

namespace shadermod
{

    class LocalizedEffectNamesParser
    {
    public:

        class LocalizedConfigFileParsedData
        {
        public:

            std::vector<std::map<WORD, std::wstring>> m_filterNamesLocalized;
            std::vector<std::wstring> m_filterNames;
            std::map<std::wstring, unsigned int> m_filenamesToFilterIds;

            void reset()
            {
                m_filterNamesLocalized.clear();
                m_filterNames.clear();
                m_filenamesToFilterIds.clear();
            }
        };

        std::map<std::wstring, LocalizedConfigFileParsedData> m_cfgPathToParsedData;

        void reset();

        void parseSingleFile(const wchar_t * path, const wchar_t * filename);

        bool getFilterName(const std::wstring & path, const std::wstring & filename, std::wstring & name) const;
        bool getFilterNameLocalized(const std::wstring & path, const std::wstring & filename, std::wstring & name, WORD langID) const;

    protected:

        void LocalizedEffectNamesParser::parseSingleFileInternal(const wchar_t * path, const wchar_t * filename);
        void LocalizedEffectNamesParser::parseInternal(const wchar_t * filename, LocalizedConfigFileParsedData * configFileParsedData);
    };
}
