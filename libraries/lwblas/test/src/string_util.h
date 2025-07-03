#pragma once

#include <string>
#include <vector>

namespace string_util
{

std::vector<std::string> split(const std::string& src, const std::string& sep)
{
    std::vector<std::string> result;
    size_t from = 0;
    while (from < src.size())
    {
        size_t to = src.find(sep, from);
        if (to == std::string::npos)
        {
            std::string val = src.substr(from);
            if (! val.empty())
            {
                result.push_back(val);
            }
            break;
        }
        std::string val = src.substr(from, to - from);
        if (! val.empty())
        {
            result.push_back(val);
        }
        from = to + sep.size();
    }
    return result;
}

bool contains(const std::string& haystack, const std::string& needle)
{
    return haystack.find(needle) != std::string::npos;
}

bool startswith(const std::string& haystack, const std::string& needle)
{
    return haystack.find(needle) == 0;
}

std::string strip(const std::string& src)
{
    std::string result = src;
    result.erase(result.begin(), std::find_if_not(result.begin(), result.end(), [](int ch) {
        return ch == ' ';
    }));
    result.erase(std::find_if_not(result.rbegin(), result.rend(), [](int ch) {
        return ch == ' ';
    }).base(), result.end());
    return result;
}

std::string with_default(const char* value, std::string alternative)
{
    if (value) return value;
    return alternative;
}

}  // namespace string_util
