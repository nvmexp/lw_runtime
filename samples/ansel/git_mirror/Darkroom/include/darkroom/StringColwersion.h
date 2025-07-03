#pragma once

#include <guiddef.h>

#include <string>

namespace darkroom
{
    enum class StringColwersionStatus
    {
        kSUCCESS = 0,
        kFAIL,
        kFALLBACK,

        kNUM_ENTRIES
    };

    void tolowerInplace(std::string & str);
    void tolowerInplace(std::wstring & str);

    std::wstring trimWhitespaceFromEnds(const std::wstring& s);

    // Two-step colwersion functions
    /////////////////////////////////////////////////////////////////////////////////////

    // Widestring -> UTF8
    StringColwersionStatus getUtf8FromWstr(const wchar_t * in, std::string & out);
    std::string getUtf8FromWstr(const wchar_t * in);

    StringColwersionStatus getUtf8FromWstr(const std::wstring & in, std::string & out);
    std::string getUtf8FromWstr(const std::wstring & in);

    // UTF8 -> Widestring
    StringColwersionStatus getWstrFromUtf8(const char * in, std::wstring & out);
    std::wstring getWstrFromUtf8(const char * in);
    StringColwersionStatus getWstrFromUtf8(const std::string & in, std::wstring & out);
    std::wstring getWstrFromUtf8(const std::string & in);

    // Three-step colwersion functions
    //  First they try to colwert utf8Main to Wstr
    //  Then, if failed, try to colwert utf8Fallback to Wstr
    //  Finally, if previous failed too, set output string to an empty string
    /////////////////////////////////////////////////////////////////////////////////////

    // UTF8 -> Widestring
    StringColwersionStatus getWstrFromUtf8Fallback(const char * utf8Main, const char * utf8Fallback, std::wstring & out);
    std::wstring getWstrFromUtf8Fallback(const char * utf8Main, const char * utf8Fallback);

    StringColwersionStatus getWstrFromUtf8Fallback(const std::string & utf8Main, const std::string & utf8Fallback, std::wstring & out);
    std::wstring getWstrFromUtf8Fallback(const std::string & utf8Main, const std::string & utf8Fallback);

    // B64 Colwersion Encoding
    std::string base64_encode(unsigned char const*, unsigned int len);
    std::string base64_encode(std::string const& s);
    std::wstring base64_encode(std::wstring const& ws);

    std::string base64_decode(std::string const& s);
    std::wstring base64_decode(std::wstring const& ws);

    const std::string GuidToString(const GUID& guid);
}
