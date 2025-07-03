////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  This provides the ability to specify advanced denylisting options in DRS.
//  You have the ability to denylist 3 different things:
//  1. Denylist Freestyle: This will remove Freestyle and Ansel-Lite altogether from the game, and prevent the LwCamera dll from loading.
//  2. Denylist Specific Filters: This will remove these filters from the list of approved filters in Freestyle and Ansel-Lite
//  3. Denylist Specific Buffers: This will prevent LwCamera from automatically capturing specific buffers in Freestyle and Ansel-Lite.
//
//  The format of this denylist is a comma separated list of elements. Aside from the "Child" element, there are 3 other elements you can specify. Each of these are specified with the following format:
//  <Element_Type_Name>=(<Comma_Separated_List>)
//
//  1. Builds: The element type name is "Builds". In order to specify a build in the comma separated list of builds, there are 2 important parts: the build ID and the branch ID. Both of these must be numbers. The standard LwCamera version is 4 dot separated numbers, such as: "7.0.558.575" or "7.0.600.0". In both of these examples, the build ID and branch ID are the last 2 numbers. HOWEVER. There is a special rule that determines which number is which. The Build ID is always the last number, and the branch ID is always the second to last number EXCEPT when the last number is 0. When the last number is 0, then they are swapped, and the build ID is the second to last number, and 0 is the branch ID. For example:
//      "7.0.558.575" = (BuildID=575, BranchID=558)
//      "7.0.600.0"   = (BuildID=600, BranchID=0)
//      When specifying a build, you may specify either just the build ID, or both the build ID and branch ID. When you do not specify the branch ID, then it will match any build that matches the build ID regardless of what the branch is. Additionally, you may specify either a single build or a range of builds. For example, the following is a valid build element:
//      Builds=(120, 130-140, 150-, 160.0, 170.180, 190.200-190.210, 220.0-230.0)
//      Note that you may have one side of the range be blank in order to specify all builds that come before or after the build specified on the other side.
//      Note: when you specify a range of builds, the branch ID must match for the range to be valid. For example, the following are ALL INVALID:
//      Builds=(140.130-150.130, 140.130-150.160, 140.0-150.160)
//      None of these ranges have matching branch IDs. Although, also note, that you may optionally ony specify the branch ID in one side of the range, and then the branch ID on the other side will be assumed to be the same. For example, of these are valid:
//      Builds=(140-150.130, 140.130-160, 140.0-150)
//
//  2. Filters: The element type name is "Filters". In order to specify a filter, you just write the entire file name, including the extension. For example:
//      Filters=(Letterbox.yaml, BlacknWhite.yaml, ASCII.fx)
//
//  3. Buffers: The element type name is "Buffers". In order to specify a buffer, you just write the name of the buffer. The following are all the valid buffer names:
//          1. HDR
//          2. Depth
//          3. HUDless
//          4. FinalColor
//      For example:
//          Buffers=(HDR, Depth)
//
//  With this string based denylisting, there are 2 ways that you can specify a denylisting rule:
//      1. Add a "Top-Level" global rule: Specifying builds like this will denylist Freestyle for those builds. Specifying buffers or filters like this will globally denylist these filters and buffers in Freestyle and Ansel-Lite for ALL builds.
//          Top Level elements can be any of the 4 types: Builds, Filters, Buffers or ChildElement"{}"
//      2. Add a "Child" rule: This is an element surrounded in lwrly brackets "{}". Specifying a "Child" rule will give you the ability to denylist specific buffers and filters in only the desired builds. For example, you could denylist the Greescreen.yaml filter only in builds 598 and before.
//          Elements within a child element can only be: Builds, Filters, or Buffers. You cannot have a relwrsive ChildElement"{}" within another child element.
//
//  Note: In DRS, a string is limited to a maximum length of 512.
//  Note: No part of this denylist is ever case sensitive.
//
//  Finally, here are 2 examples of a fully fleshed out, complex denylist, as you would define it within a DRS prd file:
//      SettingString ANSEL_DENYLIST_PER_GAME="Builds=(20000-22000, 23000, 25000-, 5000.8000, 8500.0), Filters=(Letterbox.yaml, BlacknWhite.yaml), Buffers=(HUDless), {Filters=(NightMode.yaml, TiltShift.yaml), Buffers=(Depth), Builds=(12000, 13000-14000, 15000-)}"
//      SettingString ANSEL_DENYLIST_PER_GAME="Builds=(10000, 10001-10003), Builds=(10004, 10005.0-10007), Builds=(10008, 10009-10011), Builds=(10012, 10013-10015), Builds=(10016, 10017-10019), Builds=(10020, 10021-10023), Buffers=(Depth), Buffers=(Depth), Buffers=(Depth), Buffers=(Depth), Buffers=(Depth), Buffers=(Depth), Buffers=(Depth), Buffers=(Depth), Buffers=(Depth), Buffers=(Depth), {Filters=(NightMode.yaml, TiltShift.yaml), Builds=(12000, 13000-14000, 15000-)}, {Filters=(NightMode.yaml, TiltShift.yaml), Builds=(12000, 13000-14000, 15000-)}"
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef DENYLIST_PARSER_H
#define DENYLIST_PARSER_H

#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>
#include <list>
#include <vector>

#include "AnselSDK.h"

enum class DenylistSetType
{
    kBuilds = 0,
    kFilters,
    kBuffers,
    kChild,
    kDenylistSetTypeCount
};

namespace DenylistParsingTools
{
    const std::map<std::wstring, DenylistSetType>& GetPrefixToSetTypeMap();
    const std::wstring& GetCombinedListOfValidPrefixes(); // Builds a comma separated list of all prefixes. This is only needed when there is an error.
    std::map<std::wstring, ansel::BufferType>& GetBufferStringToTypeMap();
    const std::wstring& GetCombinedListOfValidBufferStrings();
    void SplitTopLevelElements(const std::wstring& s, wchar_t delimiter, std::list<std::wstring>& splitElements, bool& errorToSet);
    bool GetProcessedBuildID(const std::wstring& buildIDRaw, int& buildID_out, int& buildIDBranch_out, bool& errorToSet);
    bool RangeIncludesID(int buildIDToCheck, int buildBranchToCheck, const std::pair<std::wstring, std::wstring>& range, bool& errorToSet);
} // namespace DenylistParsingTools

struct ElementParseData
{
public:
    ElementParseData(const std::wstring& sourceElement, bool allowChildren, bool& errorToSet);
    std::wstring& GetRawValue();
    void AddValues(std::set<std::wstring>& setToAddTo, bool& errorToSet);

    DenylistSetType m_type = DenylistSetType::kDenylistSetTypeCount;
    std::wstring m_sourceElement = L"";
    size_t m_valueStartPos = 0;
    size_t m_valueLen = 0;
    std::wstring m_value_Raw = L"";
    std::list<std::wstring> m_splitValueElements;

private:
    void CheckToSetValueRaw();
};

struct AnselFeatureSet
{
    // Defining as an ordered set because performance is not a concern.
    std::set<ansel::BufferType> m_buffers;
    std::set<std::wstring> m_filters;

    bool BufferExists(ansel::BufferType bufferType) const;
    bool FilterExists(std::wstring filter) const;
};

class ParsedDenylist
{
public:
    void Initialize(const std::wstring& rawDenylistString, bool isAChild, bool& errorToSet);
    bool IncludesBuild(int buildID, int buildBranch, bool& errorToSet);
    void AddToFeatureSet(AnselFeatureSet& setToAddTo, bool& errorToSet);
    bool Validate(bool& errorToSet);

    std::list<ParsedDenylist>& GetParsedChildren(bool& errorToSet);
private:
    std::list<std::wstring> m_splitElements;
    std::list<ElementParseData> m_splitElementsParsed;
    std::unordered_map<DenylistSetType, std::list<ElementParseData*> > m_splitElementsParsedByType;
    std::set<std::wstring> m_buildIDValues_Raw;
    std::set<std::wstring> m_bufferValues_Raw;
    std::list<ParsedDenylist> m_parsedChildren;

    AnselFeatureSet m_featureSet;
    bool m_initializedFeatureSet = false;
    std::set<int> m_buildIDs;
    std::set<std::pair<std::wstring, std::wstring> > m_buildIDRanges;
    bool m_initializedBuildIDs = false;

    void AddAllValuesOfTypeToSet(DenylistSetType type, std::set<std::wstring>& setToAddTo, bool& errorToSet);
    bool CheckToInitializeBuildsAndRanges(bool& errorToSet);
    bool ValidateAllBuildsAndRanges(bool& errorToSet);
    bool CheckToInitializeFeatureSet(bool& errorToSet);
};

class AnselDenylist
{
public:
    void CheckToInitializeWithDRS();
    void InitializeWithDRS();
    void Initialize(const std::wstring& activeBuildID_String, const std::unordered_set<std::wstring>& denylistStrings);
    bool CheckThatAllDenylistsAreValid();
    void ChangeActiveBuildID(const std::wstring& activeBuildID_String);
    bool ActiveBuildIDIsDenylisted();
    bool BufferDenylisted(ansel::BufferType bufferType);
    bool FilterDenylisted(const std::wstring& filter);

private:
    std::list<ParsedDenylist> m_parsedDenylists; // Each of these are created from an element in m_rawDenylistStrings.
    std::map<std::wstring, AnselFeatureSet> m_denylists; // Maps build IDs to denylisted feature sets. Defining as an ordered map because performance is not a concern. Entries to this map are only added when they are asked for.
    const AnselFeatureSet* m_activeDenylist = NULL;
    std::wstring m_activeBuildID_Raw;
    int m_activeBuildID_Processed = 0;
    int m_activeBuildID_Branch = 0;
    bool m_initialized = false;
    bool m_denylistParsingErrorOclwrred = false;

    const AnselFeatureSet& GetDenylist(const std::wstring& buildID_Raw, const int& buildID, const int& buildBranch);
    const AnselFeatureSet& GetActiveDenylist();
    void FillInDenylistedFeatureSetForBuildID(const int& buildID, const int& buildBranch, AnselFeatureSet& featureSetToFillIn);
};

#endif // DENYLIST_PARSER_H

