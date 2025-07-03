#pragma once

#include "Config.h"
#include "ipc/AnselIPC.h"
#include <string>
#include <vector>
#include <map>

#if ANSEL_SIDE_PRESETS

class AnselFilter
{
public:
    std::wstring filterID;
    std::vector<std::pair<std::wstring, std::vector<float>>> attributes;

    void clear()
    {
        filterID = L"";
        attributes.clear();
    }
};

struct AnselPreset
{
    std::vector<AnselFilter> filters;
    size_t stackIdx;
    std::wstring presetID;
};

AnselIpc::Status ParseAnselPreset(std::wstring filename, std::vector < std::pair<std::wstring, std::wstring>> filters, AnselPreset* preset, std::vector<std::wstring>& unappliedFilters);
AnselIpc::Status ExportAnselPreset(std::wstring filename, const AnselPreset& preset, std::vector<std::wstring>& duplicateFilters);

#endif
