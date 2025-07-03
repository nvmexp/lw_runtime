#pragma once
#include <string>
#include <vector>
#include <unordered_map>

namespace darkroom
{
    class ImageMetadata
    {
    public:
        std::string tagMake;
        std::string tagModel;
        std::string tagType;
        std::string tagSoftware;
        std::string tagDescription;
        std::string tagDrsName;
        std::string tagDrsProfileName;
        std::string tagAppTitleName;
        std::string tagAppCMSID;
        std::string tagAppShortName;
        std::string tagActiveFilters;
        std::vector<char> xmpPacket;
    };
}
