#include "AnselPreset.h"
#include "Utils.h"

#include <assert.h>
#include <fstream>
#include <algorithm>
#include <string>

#if ANSEL_SIDE_PRESETS

enum PresetParserState {
    HEAD,
    PARSING_FILE,
    FILE_COMPLETE
};

bool startswith(std::wstring line, std::wstring prefix)
{
    return line.length() >= prefix.length() && std::equal(prefix.begin(), prefix.end(), line.begin());
}
bool endswith(std::wstring line, std::wstring suffix)
{
    return line.length() >= suffix.length() && std::equal(suffix.rbegin(), suffix.rend(), line.rbegin());
}

std::wstring getFilterNameFromFile(std::wstring filename)
{
    std::wstring yaml_suffix = L".yaml";

    if (filename == L"")
        return L"";
    else if (std::equal(yaml_suffix.rbegin(), yaml_suffix.rend(), filename.rbegin()))
    {
        // In the YAML case, the technique is the filename without the ".yaml"

        size_t lastSlash = filename.rfind('\\');

        return filename.substr(lastSlash+1, filename.length() - lastSlash - 1 - yaml_suffix.length());
    }

    std::wifstream ifs(filename);
    std::wstring line;

    while (std::getline(ifs, line) && !startswith(line, L"technique"));

    if (startswith(line, L"technique"))
    {
        // format: technique <techniqueName> ?{?
        // Get the word after the first space, possibly ended by a '{'
        size_t firstSpace = line.find(' ');
        size_t bracket = line.find('{');
        if (bracket != std::string::npos)
        {
            return line.substr(firstSpace+1, bracket - firstSpace - 1);
        }
        else
        {
            return line.substr(firstSpace+1);
        }
    }
    return L"";
}

AnselIpc::Status ParseAnselPreset(std::wstring filename, std::vector<std::pair<std::wstring, std::wstring>> filters, AnselPreset* preset, std::vector<std::wstring>& unappliedFilters)
{
    assert(preset != nullptr);
    preset->filters.clear();
    preset->presetID = filename;

    // Parser State Variables
    PresetParserState parserState = HEAD;
    AnselFilter lwrrentFilter;
    int lwrrentFilterIdx = -1;
    std::map<std::wstring, std::wstring> fileToTechnique;
    std::vector<std::wstring> techniques;
    bool foundTechniques = false;

    std::wifstream ifs(filename);
    std::wstring line;


    while (std::getline(ifs, line))
    {
        if (startswith(line, L"Techniques="))
        {
            if (foundTechniques)
            {
                // Multiple techniques lines
                return AnselIpc::Status::kErrorParsingFile;
            }
            techniques = lwanselutils::StrSplit(line.substr(std::string("Techniques=").length()), ',');
            for (size_t i = 0; i < techniques.size(); i++)
            {
                //"\t\n\v\f\r " represents all whitespace
                size_t first = techniques[i].find_first_not_of(L"\t\n\v\f\r ");
                size_t last = techniques[i].find_last_not_of(L"\t\n\v\f\r ");
                if (first != std::string::npos && last != std::string::npos)
                {
                    techniques[i] = techniques[i].substr(first, last - first + 1);
                }
                else
                {
                    techniques[i] = L"";
                }
            }

            // Don't allow any duplicate filters
            for (size_t i = 0; i < techniques.size(); i++)
            {
                techniques.erase(std::remove_if(techniques.begin() + i + 1, techniques.end(), [&](auto other) { return other == techniques[i]; }), techniques.end());
            }

            preset->filters.resize(techniques.size());

            foundTechniques = true;
        }
        else if (line[0] == '[')
        {
            // Close out previous FX file, if necessary
            if (parserState == PARSING_FILE)
            {
                preset->filters[lwrrentFilterIdx] = lwrrentFilter;
                lwrrentFilterIdx = -1;
                parserState = FILE_COMPLETE;
            }
            else if (!foundTechniques)
            {
                // No techniques line before filter list
                return AnselIpc::Status::kErrorParsingFile;
            }

            std::wstring fullFilterID = L"";
            // find the file in the list of filters provided
            for (std::pair<std::wstring, std::wstring> filter : filters)
            {
                if (endswith(filter.first, line.substr(1, line.size() - 2)))
                {
                    fullFilterID = filter.first;
                    break;
                }
            }

            // Determine the filter name in this file
            std::wstring techniqueInFile = L"";
            if (fullFilterID != L"")
            {
                techniqueInFile = getFilterNameFromFile(fullFilterID);
            }


            if (techniqueInFile != L"")
            {
                // Determine if this technique is used
                for (size_t i = 0; i < techniques.size(); i++)
                {
                    if (techniqueInFile.compare(techniques[i]) == 0)
                    {
                        // found this technique
                        parserState = PARSING_FILE;
                        lwrrentFilter.clear();
                        lwrrentFilter.filterID = fullFilterID;
                        lwrrentFilterIdx = (int) i;

                        break;
                    }
                }
            }
        }
        else if (parserState == PARSING_FILE && (std::find(line.begin(), line.end(), L'=') != line.end()))
        {
            // must be a variable definition
            std::vector<std::wstring> splitline = lwanselutils::StrSplit(line, '=');
            if (splitline.size() != 2)
            {
                return AnselIpc::Status::kErrorParsingFile;
            }
            std::wstring variable = splitline[0], value = splitline[1];
            
            // We treat all parsed attributes as multidimensional floats
            // At filter-apply-time, we'll colwert them into the correct format
            std::vector<std::wstring> values = lwanselutils::StrSplit(value, ',');
            std::vector<float> floatValues;
            for (std::wstring stringValue : values)
            {
                wchar_t* pEnd;
                float floatValue = wcstof(stringValue.c_str(), &pEnd);
                if (*pEnd)
                {
                    // error parsing number
                    return AnselIpc::Status::kErrorParsingFile;
                }
                floatValues.push_back(floatValue);
            }
            lwrrentFilter.attributes.push_back(std::make_pair(variable, floatValues));
        }
    }

    // If we reach the end of a file during parsing
    if (parserState == PARSING_FILE)
    {
        preset->filters[lwrrentFilterIdx] = lwrrentFilter;
    }


    // Prune any techniques that we didnt't find during the parsing process
    for (size_t i = 0; i < preset->filters.size(); i++)
    {
        if (preset->filters[i].filterID == L"")
        {
            unappliedFilters.push_back(techniques[i]);
            preset->filters.erase(preset->filters.begin() + i);
            techniques.erase(techniques.begin() + i);
            i--;
        }
    }

    return preset->filters.size() > 0 ? AnselIpc::Status::kOk : AnselIpc::Status::kErrorParsingFile;
}

AnselIpc::Status ExportAnselPreset(std::wstring filename, const AnselPreset &preset, std::vector<std::wstring>& duplicateFilters)
{
    std::wofstream ofs(filename);

    std::wstring filterList = L"";
    std::vector<AnselFilter> filterListCopy = preset.filters;

    for (std::vector<AnselFilter>::iterator p = filterListCopy.begin(); p != filterListCopy.end(); ++p) {

        std::wstring filterName = getFilterNameFromFile(p->filterID);

        // We will remove the second copy of any filters we find, and add them to a list of duplicate filters.
        // find_if is used instead of find to implement a custom comparator.
        std::vector<AnselFilter>::iterator find = std::find_if(p + 1, filterListCopy.end(), 
            [&p](const AnselFilter & other){
                return other.filterID == p->filterID;
            });
        if (find != filterListCopy.end())
        {
            duplicateFilters.push_back(filterName);
            filterListCopy.erase(find);
        }

        filterList += filterName;
        if (p != filterListCopy.end() - 1)
            filterList += L',';
    }

    ofs << "Techniques=" << filterList << std::endl;
    //ReShade provides a list of all techniques active, in order, for sorting purposes. We don't use it (and only output active filters), but we might as well add this line.
    ofs << "TechniquesSorting=" << filterList << std::endl;
    
    ofs << std::endl;

    // Then, we just output a list of filenames and settings, stripping off all directory structure
    for (AnselFilter filter: preset.filters)
    {
        std::wstring filterFilename = filter.filterID.substr(filter.filterID.rfind(L'\\')+1);
        ofs << "[" << filterFilename << "]" << std::endl;

        for (std::pair<std::wstring, std::vector<float>> attribute : filter.attributes)
        {
            std::wstring attributeList = L"";
            for (std::vector<float>::const_iterator p = attribute.second.begin(); p != attribute.second.end(); ++p) {
                attributeList += std::to_wstring(*p);
                if (p != attribute.second.end() - 1)
                    attributeList += L',';
            }

            ofs << attribute.first << "=" << attributeList << std::endl;
        }

        ofs << std::endl;
    }

    return AnselIpc::Status::kOk;
}

#endif
