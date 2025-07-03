#include <limits>
#include <wchar.h>

#include "DenylistParser.h"
#ifndef DENYLIST_STRING_VALIDATOR
#include "Log.h"
#include "Allowlisting.h"
#include "AnselVersionInfo.h"
#else
#include "AnselDefines.h"
#endif
#include "darkroom/StringColwersion.h"

//#define PRINT_DENYLISTING_DEBUG_LOGS

#ifdef PRINT_DENYLISTING_DEBUG_LOGS
#define LOG_DEBUG_DENYLISTING(...) LOG_DEBUG(__VA_ARGS__)
#else
#define LOG_DEBUG_DENYLISTING(...)
#endif

#define LOG_ERROR_DENYLISTING_PARSE_WITH_SOURCE(errorToSet, srcStr, ...) \
    LOG_ERROR(__VA_ARGS__); \
    if (L"" != srcStr) \
    { \
        LOG_ERROR("  Source String:"); \
        LOG_ERROR("    %S", std::wstring(srcStr).c_str()); \
    } \
    errorToSet = true;

#define LOG_ERROR_DENYLISTING_PARSE(errorToSet, ...) \
    LOG_ERROR_DENYLISTING_PARSE_WITH_SOURCE(errorToSet, L"", __VA_ARGS__);

namespace DenylistParsingTools
{

    const std::map<std::wstring, DenylistSetType>& GetPrefixToSetTypeMap()
    {
        static const std::map<std::wstring, DenylistSetType> prefixToSetTypeMap =
        {
            {L"{", DenylistSetType::kChild},
            {L"builds", DenylistSetType::kBuilds},
            {L"buffers", DenylistSetType::kBuffers},
            {L"filters", DenylistSetType::kFilters}
        };
        return prefixToSetTypeMap;
    }

    const std::wstring& GetCombinedListOfValidPrefixes()
    {
        static std::wstring combinedListOfValidPrefixes = L"";
        if (combinedListOfValidPrefixes.empty())
        {
            combinedListOfValidPrefixes = L"(";
            const auto prefixToSetTypeMap = GetPrefixToSetTypeMap();

            // Build combinedListOfValidPrefixes
            for (auto validPrefixItr = prefixToSetTypeMap.begin(); validPrefixItr != prefixToSetTypeMap.end(); validPrefixItr++)
            {
                if (validPrefixItr != prefixToSetTypeMap.begin())
                {
                    combinedListOfValidPrefixes += L", ";
                }
                combinedListOfValidPrefixes += (validPrefixItr->first);
            }
            combinedListOfValidPrefixes += L")";
        }
        return combinedListOfValidPrefixes;
    }

    std::map<std::wstring, ansel::BufferType>& GetBufferStringToTypeMap()
    {
        static std::map<std::wstring, ansel::BufferType> bufferStringToTypeMap =
        {
            {L"hdr", ansel::BufferType::kBufferTypeHDR},
            {L"depth", ansel::BufferType::kBufferTypeDepth},
            {L"hudless", ansel::BufferType::kBufferTypeHUDless},
            {L"finalcolor", ansel::BufferType::kBufferTypeFinalColor}
        };
        return bufferStringToTypeMap;
    }

    const std::wstring& GetCombinedListOfValidBufferStrings()
    {
        static std::wstring combinedListOfValidBufferStrings = L"";
        if (combinedListOfValidBufferStrings.empty())
        {
            combinedListOfValidBufferStrings = L"(";
            auto bufferStringToTypeMap = GetBufferStringToTypeMap();

            // Build combinedListOfValidPrefixes
            for (auto validBufferItr = bufferStringToTypeMap.begin(); validBufferItr != bufferStringToTypeMap.end(); validBufferItr++)
            {
                if (validBufferItr != bufferStringToTypeMap.begin())
                {
                    combinedListOfValidBufferStrings += L", ";
                }
                combinedListOfValidBufferStrings += (validBufferItr->first);
            }
            combinedListOfValidBufferStrings += L")";
        }
        return combinedListOfValidBufferStrings;
    }

    static void CheckToPushLwrrentParsedTopLevelElement(std::list<std::wstring>& splitElements, const std::wstring& s, size_t& lwrStart, size_t lwrPos)
    {
        if (lwrStart < lwrPos)
        {
            splitElements.push_back(darkroom::trimWhitespaceFromEnds(s.substr(lwrStart, (lwrPos - lwrStart))));
        }
        lwrStart = lwrPos + 1;
    }
    // This checks for {} and () subsets, and ignores them.
    static void CheckSubsetBounds(const std::wstring& s, const size_t lwrPos, const wchar_t leftBound, const wchar_t rightBound, UINT& boundCount, const std::wstring& boundTypeName, bool& errorToSet)
    {
        const wchar_t lwrChar = s[lwrPos];
        if (lwrChar == leftBound)
        {
            boundCount++;
        }
        else if (lwrChar == rightBound)
        {
            if (boundCount == 0)
            {
                LOG_ERROR_DENYLISTING_PARSE(errorToSet, "Unmatched right %S at position %d in string:\n %S", boundTypeName.c_str(), static_cast<int>(lwrPos), s.c_str());
            }
            else
            {
                boundCount--;
            }
        }
    }

    void SplitTopLevelElements(const std::wstring& s, wchar_t delimiter, std::list<std::wstring>& splitElements, bool& errorToSet)
    {
        // This just splits along top level delimiters
        size_t lwrPos = 0;
        size_t lwrStart = 0;
        UINT leftLwrlyBracketCount = 0;
        UINT leftParenCount = 0;

        while (lwrPos < s.size())
        {
            CheckSubsetBounds(s, lwrPos, L'{', L'}', leftLwrlyBracketCount, L"lwrley brace", errorToSet);
            CheckSubsetBounds(s, lwrPos, L'(', L')', leftLwrlyBracketCount, L"parentheses", errorToSet);

            if (s[lwrPos] == delimiter && leftLwrlyBracketCount == 0 && leftParenCount == 0) // Only split on top level delimiters
            {
                CheckToPushLwrrentParsedTopLevelElement(splitElements, s, lwrStart, lwrPos);
            }

            lwrPos++;
        }
        CheckToPushLwrrentParsedTopLevelElement(splitElements, s, lwrStart, lwrPos);

    #ifdef PRINT_DENYLISTING_DEBUG_LOGS
        LOG_DEBUG("");
        LOG_DEBUG("Delimiter \"%lc\" splitting complete:", delimiter);
        LOG_DEBUG("  \"%S\":", s.c_str());
        for (auto splitStringItr = splitElements.begin(); splitStringItr != splitElements.end(); splitStringItr++)
        {
            LOG_DEBUG("      \"%S\"", splitStringItr->c_str());
        }
        LOG_DEBUG("");
    #endif // PRINT_DENYLISTING_DEBUG_LOGS
    }

    bool GetProcessedBuildID(const std::wstring& buildIDRaw, int& buildID_out, int& buildIDBranch_out, bool& errorToSet)
    {
        buildID_out = -1;
        buildIDBranch_out = -1;

        if (buildIDRaw.empty()) return true;

        // First validate the build number
        // Expected xxx.xxx.xxx.xxx
        // We will accept up to 4 sets of numbers each separated by dots L'.'
        UINT dotCount = 0;
        int lastDotPos = -1;
        int secondToLastDotPos = -1;
        for (UINT i = 0; i < buildIDRaw.size(); i++)
        {
            if (buildIDRaw[i] == L'.')
            {
                dotCount++;
                secondToLastDotPos = lastDotPos;
                lastDotPos = static_cast<int>(i);
            }
            else if (!iswdigit(buildIDRaw[i]))
            {
                LOG_ERROR_DENYLISTING_PARSE(errorToSet, "\"%S\" is not a valid build ID. Build ID must be only dot(.) separated numbers.", buildIDRaw.c_str());
                return false;
            }
        }
        if (dotCount > 3)
        {
            LOG_ERROR_DENYLISTING_PARSE(errorToSet, "\"%S\" is not a valid build ID. Build ID cannot have more than 4 sets of dot(.) separated numbers.", buildIDRaw.c_str());
            return false;
        }

        // Then read the last 2 numbers.
        if (lastDotPos == -1)
        {
            // If only 1 number, use that.
            // If no dots, branch is not specified, and thus we will give an "unknown branch", "-1"
            buildIDBranch_out = -1;
            buildID_out = _wtoi(buildIDRaw.c_str());
        }
        else
        {
            // If 2 or more numbers, then use the last number for the build ID if it is not 0, otherwise use the second to last number for the build ID

            size_t lastNumberStartPos = lastDotPos + 1;
            size_t lastNumberLen = buildIDRaw.size() - lastNumberStartPos;

            size_t secondToLastNumberStartPos = secondToLastDotPos + 1;
            size_t secondToLastNumberLen = lastDotPos - secondToLastNumberStartPos;

            std::wstring lastNumber_str = buildIDRaw.substr(lastNumberStartPos, lastNumberLen);
            std::wstring secondToLastNumber_str = buildIDRaw.substr(secondToLastNumberStartPos, secondToLastNumberLen);

            int lastNumber = -1;
            if (!lastNumber_str.empty())
            {
                lastNumber = _wtoi(lastNumber_str.c_str());
            }
            int secondToLastNumber = -1;
            if (!secondToLastNumber_str.empty())
            {
                secondToLastNumber = _wtoi(secondToLastNumber_str.c_str());
            }

            if (0 == lastNumber)
            {
                buildID_out = secondToLastNumber;
                buildIDBranch_out = 0;
            }
            else
            {
                buildID_out = lastNumber;
                buildIDBranch_out = secondToLastNumber;
            }
        }

        LOG_DEBUG_DENYLISTING("  Raw build ID: \"%S\" read as build \"%d\" off of branch \"%d\"", buildIDRaw.c_str(), buildID_out, buildIDBranch_out);

        return true;
    }

    bool RangeIncludesID(int buildIDToCheck, int buildBranchToCheck, const std::pair<std::wstring, std::wstring>& range, bool& errorToSet)
    {
        LOG_DEBUG_DENYLISTING("Checking to see if build \"%d\" in branch \"%d\" is within range: \"%S-%S\"", buildIDToCheck, buildBranchToCheck, range.first.c_str(), range.second.c_str());

        int buildID_Range1, buildBranch_Range1, buildID_Range2, buildBranch_Range2;
        if (!GetProcessedBuildID(range.first, buildID_Range1, buildBranch_Range1, errorToSet)
            || !GetProcessedBuildID(range.second, buildID_Range2, buildBranch_Range2, errorToSet))
        {
            LOG_ERROR_DENYLISTING_PARSE(errorToSet, "Invalid build specified in range: \"%S-%S\"", range.first.c_str(), range.second.c_str());
            return false;
        }

        {
            // If only one of the builds specifices a branch, force both to match the specified branch.
            if (-1 == buildBranch_Range1)
            {
                buildBranch_Range1 = buildBranch_Range2;
            }
            if (-1 == buildBranch_Range2)
            {
                buildBranch_Range2 = buildBranch_Range1;
            }
        }

        if (buildBranch_Range1 != buildBranch_Range2)
        {
            LOG_ERROR_DENYLISTING_PARSE(errorToSet, "Range: \"%S-%S\" Branch \"%d\" does not match branch \"%d\". A build range must be across a single branch.", range.first.c_str(), range.second.c_str(), buildBranch_Range1, buildBranch_Range2);
            return false;
        }

        if (-1 != buildBranchToCheck && -1 != buildBranch_Range1 && buildBranchToCheck != buildBranch_Range1)
        {
            // If both the range and build in question specify a branch, and they do not match, then the build is from a different branch, and is not included in the range.
            return false;
        }

        // If the second part of the range does not specify an end build, then there is no limit, and it should be set to the max possible build ID.
        if (-1 == buildID_Range2)
        {
            buildID_Range2 = INT_MAX;
        }

        if (buildID_Range1 > buildID_Range2)
        {
            // Make sure the first build in the range is smaller.
            std::swap(buildID_Range1, buildID_Range2);
        }

        LOG_DEBUG_DENYLISTING("    Range: \"%S-%S\" processed to check builds between \"%d\" and \"%d\" on branch \"%d\"", range.first.c_str(), range.second.c_str(), buildID_Range1, buildID_Range2, buildBranch_Range1);

        if (buildIDToCheck >= buildID_Range1 && buildIDToCheck <= buildID_Range2)
        {
            // Build is between the range of builds specified for the same branch.
            LOG_DEBUG_DENYLISTING("    Build \"%d\" on branch \"%d\" is within range: \"%S-%S\"", buildIDToCheck, buildBranchToCheck, range.first.c_str(), range.second.c_str());
            return true;
        }

        // Build is from the same branch as the specified range, but it is not within the specified range.
        return false;
    }

} // namespace DenylistParsingTools

ElementParseData::ElementParseData(const std::wstring& sourceElement, bool allowChildren, bool& errorToSet)
{
    m_sourceElement = sourceElement;

    if (m_sourceElement.empty())
    {
        LOG_ERROR_DENYLISTING_PARSE(errorToSet, "Empty source element provided.");
        return;
    }

    // Parse prefix and value
    std::wstring prefix = L"";
    if (m_sourceElement[0] == L'{')
    {
        if (!allowChildren)
        {
            LOG_ERROR_DENYLISTING_PARSE(errorToSet, "Invalid denylisting entry. Cannot have nested lwrly bracketed subsets:\n   \"%S\"", m_sourceElement.c_str());
            return;
        }

        prefix = L"{";
        if (m_sourceElement[m_sourceElement.size() - 1] != L'}')
        {
            LOG_ERROR_DENYLISTING_PARSE(errorToSet, "Invalid denylisting entry. Denylisting sets must start with '{' and end with '}':\n    \"%S\"", m_sourceElement.c_str());
            return;
        }

        // Do not include L'{' and L'}' on the ends.
        m_valueStartPos = 1;
        m_valueLen = m_sourceElement.size() - 2;
    }
    else
    {
        size_t posOfFirstEqual = m_sourceElement.find(L'=');
        if (posOfFirstEqual == std::wstring::npos)
        {
            LOG_ERROR_DENYLISTING_PARSE(errorToSet, "Invalid denylisting entry. No '=' found. Must define as <TYPE>=<VALUE>:\n  \"%S\"", m_sourceElement.c_str());
            return;
        }

        // Check that the value starts with L'(' and ends with L')' in this case.
        if (m_sourceElement[posOfFirstEqual + 1] != L'(' || m_sourceElement[m_sourceElement.size() - 1] != L')')
        {
            LOG_ERROR_DENYLISTING_PARSE(errorToSet, "Invalid denylisting entry. Denylisting sets must start with '(' and end with ')':\n    \"%S\"", m_sourceElement.c_str());
            return;
        }

        prefix = m_sourceElement.substr(0, posOfFirstEqual);
        darkroom::tolowerInplace(prefix); // Make prefixes not case sensitive

        // Do not include L'(' and L')' on the ends.
        m_valueStartPos = posOfFirstEqual + 2;
        m_valueLen = ((m_sourceElement.size() - 1) - m_valueStartPos);
    }

    if (0 == m_valueLen)
    {
        LOG_ERROR_DENYLISTING_PARSE(errorToSet, "Invalid denylist element: \"%S\". This element has no value.", sourceElement.c_str());
        return;
    }

    prefix = darkroom::trimWhitespaceFromEnds(prefix);

    // Check validity of set
    const auto prefixToSetTypeMap = DenylistParsingTools::GetPrefixToSetTypeMap();
    if (prefixToSetTypeMap.find(prefix) == prefixToSetTypeMap.end())
    {
        LOG_ERROR_DENYLISTING_PARSE(errorToSet, "\"%S\": Invalid denylist set name: \"%S\". Denylisting sets must start with one of:\n  %S", sourceElement.c_str(), prefix.c_str(), DenylistParsingTools::GetCombinedListOfValidPrefixes().c_str());
        return;
    }

    m_type = prefixToSetTypeMap.at(prefix);
}

std::wstring& ElementParseData::GetRawValue()
{
    CheckToSetValueRaw();
    return m_value_Raw;
}

void ElementParseData::AddValues(std::set<std::wstring>& setToAddTo, bool& errorToSet)
{
    CheckToSetValueRaw();

    if (m_splitValueElements.empty())
    {
        DenylistParsingTools::SplitTopLevelElements(m_value_Raw, L',', m_splitValueElements, errorToSet);
        for (auto valueElemItr = m_splitValueElements.begin(); valueElemItr != m_splitValueElements.end(); valueElemItr++)
        {
            darkroom::tolowerInplace(*valueElemItr); // Make all values not case sensitive.
        }
    }

    for (auto valueElemItr = m_splitValueElements.begin(); valueElemItr != m_splitValueElements.end(); valueElemItr++)
    {
        setToAddTo.insert(*valueElemItr);
    }
}

void ElementParseData::CheckToSetValueRaw()
{
    if (m_value_Raw.empty())
    {
        m_value_Raw = m_sourceElement.substr(m_valueStartPos, m_valueLen);
        m_value_Raw = darkroom::trimWhitespaceFromEnds(m_value_Raw);
    }
}

bool AnselFeatureSet::BufferExists(ansel::BufferType bufferType) const
{
    return (m_buffers.find(bufferType) != m_buffers.end());
}

bool AnselFeatureSet::FilterExists(std::wstring filter) const
{
    darkroom::tolowerInplace(filter);
    return (m_filters.find(filter) != m_filters.end());
}

void ParsedDenylist::Initialize(const std::wstring& rawDenylistString, bool isAChild, bool& errorToSet)
{
    if (rawDenylistString.size() > 512)
    {
        LOG_ERROR_DENYLISTING_PARSE_WITH_SOURCE(errorToSet, rawDenylistString, "Denylist length is too long. DRS will fail to read a string longer than 512!");
        return;
    }

    DenylistParsingTools::SplitTopLevelElements(rawDenylistString, L',', m_splitElements, errorToSet);
    for (auto elementItr = m_splitElements.begin(); elementItr != m_splitElements.end(); elementItr++)
    {
        m_splitElementsParsed.emplace_back(ElementParseData(*elementItr, !isAChild, errorToSet));
        m_splitElementsParsedByType[m_splitElementsParsed.back().m_type].push_back(&m_splitElementsParsed.back());
    }
}

bool ParsedDenylist::IncludesBuild(int buildID, int buildBranch, bool& errorToSet)
{
    CheckToInitializeBuildsAndRanges(errorToSet);

    if (m_buildIDs.find(buildID) != m_buildIDs.end()) return true;
    for (auto rangeItr = m_buildIDRanges.begin(); rangeItr != m_buildIDRanges.end(); rangeItr++)
    {
        if (DenylistParsingTools::RangeIncludesID(buildID, buildBranch, *rangeItr, errorToSet)) return true;
    }

    return false;
}

void ParsedDenylist::AddToFeatureSet(AnselFeatureSet& setToAddTo, bool& errorToSet)
{
    CheckToInitializeFeatureSet(errorToSet);

    setToAddTo.m_filters.insert(m_featureSet.m_filters.begin(), m_featureSet.m_filters.end());
    setToAddTo.m_buffers.insert(m_featureSet.m_buffers.begin(), m_featureSet.m_buffers.end());
}

bool ParsedDenylist::Validate(bool& errorToSet)
{
    ValidateAllBuildsAndRanges(errorToSet);
    CheckToInitializeFeatureSet(errorToSet);
    return true;
}

std::list<ParsedDenylist>& ParsedDenylist::GetParsedChildren(bool& errorToSet)
{
    auto& splitElements_Child = m_splitElementsParsedByType[DenylistSetType::kChild];
    if (m_parsedChildren.size() != splitElements_Child.size())
    {
        for (auto childItr = splitElements_Child.begin(); childItr != splitElements_Child.end(); childItr++)
        {
            m_parsedChildren.push_back(ParsedDenylist());
            m_parsedChildren.back().Initialize((*childItr)->GetRawValue(), true, errorToSet);
        }
    }

    return m_parsedChildren;
}

void ParsedDenylist::AddAllValuesOfTypeToSet(DenylistSetType type, std::set<std::wstring>& setToAddTo, bool& errorToSet)
{
    auto& splitElementsOfType = m_splitElementsParsedByType[type];
    for (auto splitElementItr = splitElementsOfType.begin(); splitElementItr != splitElementsOfType.end(); splitElementItr++)
    {
        (*splitElementItr)->AddValues(setToAddTo, errorToSet);
    }
}

bool ParsedDenylist::CheckToInitializeBuildsAndRanges(bool& errorToSet)
{
    if (!m_initializedBuildIDs)
    {
        AddAllValuesOfTypeToSet(DenylistSetType::kBuilds, m_buildIDValues_Raw, errorToSet);

        for (auto buildIDItr = m_buildIDValues_Raw.begin(); buildIDItr != m_buildIDValues_Raw.end(); buildIDItr++)
        {
            size_t posOfFirstDash = buildIDItr->find(L'-');
            if (posOfFirstDash == std::wstring::npos)
            {
                int lwrBuildID, lwrBuildBranch;
                DenylistParsingTools::GetProcessedBuildID((*buildIDItr), lwrBuildID, lwrBuildBranch, errorToSet);
                m_buildIDs.insert(lwrBuildID);
            }
            else
            {
                const size_t value1Len = posOfFirstDash;
                const size_t value2StartPos = posOfFirstDash + 1;
                const size_t value2Len = ((buildIDItr->size() - 1) - value2StartPos) + 1;
                std::pair<std::wstring, std::wstring> lwrPair;
                lwrPair.first = buildIDItr->substr(0, value1Len);
                lwrPair.second = buildIDItr->substr(value2StartPos, value2Len);
                m_buildIDRanges.insert(lwrPair);
            }
        }

        m_initializedBuildIDs = true;
    }
    return true;
}

bool ParsedDenylist::ValidateAllBuildsAndRanges(bool& errorToSet)
{
    CheckToInitializeBuildsAndRanges(errorToSet);
    for (auto rangeItr = m_buildIDRanges.begin(); rangeItr != m_buildIDRanges.end(); rangeItr++)
    {
        DenylistParsingTools::RangeIncludesID(-1, -1, *rangeItr, errorToSet);
    }
    return true;
}

bool ParsedDenylist::CheckToInitializeFeatureSet(bool& errorToSet)
{
    if (!m_initializedFeatureSet)
    {
        AddAllValuesOfTypeToSet(DenylistSetType::kFilters, m_featureSet.m_filters, errorToSet);

        AddAllValuesOfTypeToSet(DenylistSetType::kBuffers, m_bufferValues_Raw, errorToSet);
        for (auto bufferItr = m_bufferValues_Raw.begin(); bufferItr != m_bufferValues_Raw.end(); bufferItr++)
        {
            // Check validity of buffer
            auto bufferStringToTypeMap = DenylistParsingTools::GetBufferStringToTypeMap();
            if (bufferStringToTypeMap.find(*bufferItr) == bufferStringToTypeMap.end())
            {
                LOG_ERROR_DENYLISTING_PARSE(errorToSet, "Invalid buffer name: \"%S\". Buffers must be one of:\n %S", bufferItr->c_str(), DenylistParsingTools::GetCombinedListOfValidBufferStrings().c_str());
                return false;
            }

            m_featureSet.m_buffers.insert(bufferStringToTypeMap[*bufferItr]);
        }

        m_initializedFeatureSet = true;
    }
    return true;
}

#ifndef DENYLIST_STRING_VALIDATOR
static void CheckToAddDenylistFromDRS(std::unordered_set<std::wstring>& denylistStrings, std::wstring& lwrDenylistString, uint32_t keyID, std::wstring defaultValue, bool readGold)
{
    if (gLogSeverityLevel == LogSeverity::kDebug)
    {
        lwrDenylistString = defaultValue;
    }
    const bool isKeySet = drs::getKeyValueString(keyID, lwrDenylistString, readGold);
    if (isKeySet)
    {
        if (L"" != lwrDenylistString)
        {
            denylistStrings.insert(lwrDenylistString);
        }
    }
    else if (L"" != defaultValue)
    {
        denylistStrings.insert(defaultValue);
    }
}
#endif

void AnselDenylist::CheckToInitializeWithDRS()
{
    if (!m_initialized)
    {
        InitializeWithDRS();
    }
}

void AnselDenylist::InitializeWithDRS()
{
#ifndef DENYLIST_STRING_VALIDATOR
    LOG_DEBUG("Initializing denylist with DRS:");

    // Initialize Denylist - check both global and per game UserDB and GoldDB values
    std::wstring lwrDenylistString;
    std::unordered_set<std::wstring> denylistStrings; // unordered_set ensures no duplicate strings and thus no duplicate parsing work.
    // Per Game Denylist from the GoldDB
    CheckToAddDenylistFromDRS(denylistStrings, lwrDenylistString, ANSEL_DENYLIST_PER_GAME_ID, ANSEL_DENYLIST_PER_GAME_DEFAULT, true);
    LOG_DEBUG("    Denylist PerGame Gold: \"%S\"", lwrDenylistString.c_str());
    // Global Denylist from the GoldDB
    CheckToAddDenylistFromDRS(denylistStrings, lwrDenylistString, ANSEL_DENYLIST_ALL_PROFILED_ID, ANSEL_DENYLIST_ALL_PROFILED_DEFAULT, true);
    LOG_DEBUG("    Denylist Global Gold: \"%S\"", lwrDenylistString.c_str());
    // Per Game Denylist from the UserDB
    CheckToAddDenylistFromDRS(denylistStrings, lwrDenylistString, ANSEL_DENYLIST_PER_GAME_ID, ANSEL_DENYLIST_PER_GAME_DEFAULT, false);
    LOG_DEBUG("    Denylist PerGame User: \"%S\"", lwrDenylistString.c_str());
    // Global Denylist from the UserDB
    CheckToAddDenylistFromDRS(denylistStrings, lwrDenylistString, ANSEL_DENYLIST_ALL_PROFILED_ID, ANSEL_DENYLIST_ALL_PROFILED_DEFAULT, false);
    LOG_DEBUG("    Denylist Global User: \"%S\"", lwrDenylistString.c_str());

    std::wstring anselFileVersionString = darkroom::getWstrFromUtf8(ANSEL_FILEVERSION_STRING);

    Initialize(anselFileVersionString, denylistStrings);
#else
    LOG_ERROR_DENYLISTING_PARSE(m_denylistParsingErrorOclwrred, "InitializeWithDRS() should not be called in the Denylist String Validator.");
#endif
}

void AnselDenylist::Initialize(const std::wstring& activeBuildID_String, const std::unordered_set<std::wstring>& denylistStrings)
{
    if (m_initialized)
    {
        LOG_WARN("Denylist has already been initialized! Resetting and re-initializing...");
        // Reset this structure to a freshly created one.
        *this = AnselDenylist();
    }

    LOG_DEBUG("Initializing denylist under build ID: \"%S\" with the following strings:", activeBuildID_String.c_str());
    if (denylistStrings.empty())
    {
        LOG_DEBUG("  No strings provided!");
    }

    // An unordered_set input ensures no duplicates.
    for (auto rawStringItr = denylistStrings.begin(); rawStringItr != denylistStrings.end(); rawStringItr++)
    {
        LOG_DEBUG("  \"%S\"", rawStringItr->c_str());
        m_parsedDenylists.emplace_back(ParsedDenylist());
        m_parsedDenylists.back().Initialize(*rawStringItr, false, m_denylistParsingErrorOclwrred);
    }

    ChangeActiveBuildID(activeBuildID_String);

    m_initialized = true;
}

bool AnselDenylist::CheckThatAllDenylistsAreValid()
{
    if (!m_initialized)
    {
        LOG_ERROR_DENYLISTING_PARSE(m_denylistParsingErrorOclwrred, "CheckThatAllDenylistsAreValid(): Denylist not initialized yet!");
        return false;
    }

    // Iterate through all our denylists
    for (auto parsedDenylistItr = m_parsedDenylists.begin(); parsedDenylistItr != m_parsedDenylists.end(); parsedDenylistItr++)
    {
        // Validate the global list.
        parsedDenylistItr->Validate(m_denylistParsingErrorOclwrred);
        for (auto parsedChildItr = parsedDenylistItr->GetParsedChildren(m_denylistParsingErrorOclwrred).begin(); parsedChildItr != parsedDenylistItr->GetParsedChildren(m_denylistParsingErrorOclwrred).end(); parsedChildItr++)
        {
            // Validate the subset children.
            parsedChildItr->Validate(m_denylistParsingErrorOclwrred);
        }
    }

    if (m_denylistParsingErrorOclwrred)
    {
        LOG_ERROR_DENYLISTING_PARSE(m_denylistParsingErrorOclwrred, "");
        LOG_ERROR_DENYLISTING_PARSE(m_denylistParsingErrorOclwrred, "Parsing error encountered. Provided denylists are not valid!");
        LOG_ERROR_DENYLISTING_PARSE(m_denylistParsingErrorOclwrred, "");
    }
    else
    {
        LOG_DEBUG("");
        LOG_DEBUG("Passed denylist validation. All denylists are valid!");
        LOG_DEBUG("");
    }
    return !m_denylistParsingErrorOclwrred;
}

void AnselDenylist::ChangeActiveBuildID(const std::wstring& activeBuildID_Raw)
{
    m_activeBuildID_Raw = activeBuildID_Raw;
    if (!DenylistParsingTools::GetProcessedBuildID(m_activeBuildID_Raw, m_activeBuildID_Processed, m_activeBuildID_Branch, m_denylistParsingErrorOclwrred))
    {
        LOG_ERROR_DENYLISTING_PARSE_WITH_SOURCE(m_denylistParsingErrorOclwrred, activeBuildID_Raw, "ChangeActiveBuildID(): BuildID not valid!");
        return;
    }
    m_activeDenylist = NULL; // This will automatically be reset when it is requested.
}

bool AnselDenylist::ActiveBuildIDIsDenylisted()
{
    if (!m_initialized)
    {
        LOG_ERROR_DENYLISTING_PARSE(m_denylistParsingErrorOclwrred, "BuildIDDenylisted(): Denylist not initialized yet!");
        return true; // In the event of an error, default to denylisted.
    }

    if (m_denylistParsingErrorOclwrred)
    {
        return true; // In the event of an error, default to denylisted.
    }

    for (auto parsedDenylistItr = m_parsedDenylists.begin(); parsedDenylistItr != m_parsedDenylists.end(); parsedDenylistItr++)
    {
        if (parsedDenylistItr->IncludesBuild(m_activeBuildID_Processed, m_activeBuildID_Branch, m_denylistParsingErrorOclwrred))
        {
            return true;
        }
    }

    return false;
}
bool AnselDenylist::BufferDenylisted(ansel::BufferType bufferType)
{
    if (!m_initialized)
    {
        LOG_ERROR_DENYLISTING_PARSE(m_denylistParsingErrorOclwrred, "BufferDenylisted(): Denylist not initialized yet!");
    }

    if (m_denylistParsingErrorOclwrred)
    {
        return true; // In the event of an error, default to denylisted.
    }

    return GetActiveDenylist().BufferExists(bufferType);
}
bool AnselDenylist::FilterDenylisted(const std::wstring& filter)
{
    if (!m_initialized)
    {
        LOG_ERROR_DENYLISTING_PARSE(m_denylistParsingErrorOclwrred, "FilterDenylisted(): Denylist not initialized yet!");
        return true; // In the event of an error, default to denylisted.
    }

    if (m_denylistParsingErrorOclwrred)
    {
        return true; // In the event of an error, default to denylisted.
    }

    return GetActiveDenylist().FilterExists(filter);
}

const AnselFeatureSet& AnselDenylist::GetDenylist(const std::wstring& buildID_Raw, const int& buildID, const int& buildBranch)
{
    if (!m_initialized)
    {
        LOG_ERROR_DENYLISTING_PARSE(m_denylistParsingErrorOclwrred, "GetDenylist(): Denylist not initialized yet!");
    }

    if (m_denylists.find(buildID_Raw) == m_denylists.end())
    {
        FillInDenylistedFeatureSetForBuildID(buildID, buildBranch, m_denylists[buildID_Raw]);
    }
    return (m_denylists[buildID_Raw]);
}

const AnselFeatureSet& AnselDenylist::GetActiveDenylist()
{
    if (!m_activeDenylist)
    {
        m_activeDenylist = &(GetDenylist(m_activeBuildID_Raw, m_activeBuildID_Processed, m_activeBuildID_Branch));
    }

    return *m_activeDenylist;
}

void AnselDenylist::FillInDenylistedFeatureSetForBuildID(const int& buildID, const int& buildBranch, AnselFeatureSet& featureSetToFillIn)
{
    // Iterate through all our denylists
    for (auto parsedDenylistItr = m_parsedDenylists.begin(); parsedDenylistItr != m_parsedDenylists.end(); parsedDenylistItr++)
    {
        // Always add the globally denylisted feature set.
        parsedDenylistItr->AddToFeatureSet(featureSetToFillIn, m_denylistParsingErrorOclwrred);
        for (auto parsedChildItr = parsedDenylistItr->GetParsedChildren(m_denylistParsingErrorOclwrred).begin(); parsedChildItr != parsedDenylistItr->GetParsedChildren(m_denylistParsingErrorOclwrred).end(); parsedChildItr++)
        {
            // For each child set, add the denylisted feature set if the build is within the ranges specified for that set.
            if (parsedChildItr->IncludesBuild(buildID, buildBranch, m_denylistParsingErrorOclwrred))
            {
                parsedChildItr->AddToFeatureSet(featureSetToFillIn, m_denylistParsingErrorOclwrred);
            }
        }
    }
}

