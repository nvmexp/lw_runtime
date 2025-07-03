#include "darkroom\StringColwersion.h"

#include <excpt.h>

#include <algorithm>
#include <codecvt>

namespace darkroom
{
    void tolowerInplace(std::string & str)
    {
        std::transform(str.begin(), str.end(), str.begin(), [](const char & ch) { return (char)::tolower(ch); });
    }
    void tolowerInplace(std::wstring & str)
    {
        std::transform(str.begin(), str.end(), str.begin(), [](const wchar_t & ch) { return (wchar_t)::towlower(ch); });
    }

    std::wstring trimWhitespaceFromEnds(const std::wstring& s)
    {
        static const std::wstring WHITESPACE = L" \n\r\t\f\v";
        const size_t firstNonWhitespacePos = s.find_first_not_of(WHITESPACE);
        const size_t lastNonWhitespacePos = s.find_last_not_of(WHITESPACE);

        if (firstNonWhitespacePos == std::string::npos || lastNonWhitespacePos == std::string::npos) return L"";
        return s.substr(firstNonWhitespacePos, (lastNonWhitespacePos - firstNonWhitespacePos) + 1);
    }

    // Two-step
    /////////////////////////////////////////////////////////////////////////////////////
    // Widestring -> UTF8
    void getUtf8FromWstrInternal(const wchar_t * in, std::string & out)
    {
        static std::wstring_colwert<std::codecvt_utf8_utf16<wchar_t>> colwerter;
        out = colwerter.to_bytes(in);
    }
    StringColwersionStatus getUtf8FromWstr(const wchar_t * in, std::string & out)
    {
        __try
        {
            getUtf8FromWstrInternal(in, out);
            return StringColwersionStatus::kSUCCESS;
        }
        __except (EXCEPTION_EXELWTE_HANDLER)
        {
            out = "";
            return StringColwersionStatus::kFAIL;
        }
    }
    std::string getUtf8FromWstr(const wchar_t * in)
    {
        std::string out;
        getUtf8FromWstr(in, out);
        return out;
    }
    StringColwersionStatus getUtf8FromWstr(const std::wstring & in, std::string & out)
    {
        __try
        {
            getUtf8FromWstrInternal(in.c_str(), out);
            return StringColwersionStatus::kSUCCESS;
        }
        __except (EXCEPTION_EXELWTE_HANDLER)
        {
            out = "";
            return StringColwersionStatus::kFAIL;
        }
    }
    std::string getUtf8FromWstr(const std::wstring & in)
    {
        std::string out;
        getUtf8FromWstrInternal(in.c_str(), out);
        return out;
    }

    // Two-step
    /////////////////////////////////////////////////////////////////////////////////////
    // UTF8 -> Widestring
    void getWstrFromUtf8Internal(const char * in, std::wstring & out)
    {
        static std::wstring_colwert<std::codecvt_utf8_utf16<wchar_t>> colwerter;
        out = colwerter.from_bytes(in);
    }

    StringColwersionStatus getWstrFromUtf8(const char * in, std::wstring & out)
    {
        __try
        {
            getWstrFromUtf8Internal(in, out);
            return StringColwersionStatus::kSUCCESS;
        }
        __except (EXCEPTION_EXELWTE_HANDLER)
        {
            out = L"";
            return StringColwersionStatus::kFAIL;
        }
    }
    std::wstring getWstrFromUtf8(const char * in)
    {
        std::wstring out;
        getWstrFromUtf8(in, out);
        return out;
    }
    StringColwersionStatus getWstrFromUtf8(const std::string & in, std::wstring & out)
    {
        __try
        {
            getWstrFromUtf8Internal(in.c_str(), out);
            return StringColwersionStatus::kSUCCESS;
        }
        __except (EXCEPTION_EXELWTE_HANDLER)
        {
            out = L"";
            return StringColwersionStatus::kFAIL;
        }
    }
    std::wstring getWstrFromUtf8(const std::string & in)
    {
        std::wstring out;
        getWstrFromUtf8(in, out);
        return out;
    }

    // Three-step
    /////////////////////////////////////////////////////////////////////////////////////
    // UTF8 -> Widestring
    StringColwersionStatus getWstrFromUtf8Fallback(const char * utf8Main, const char * utf8Fallback, std::wstring & out)
    {
        __try
        {
            getWstrFromUtf8Internal(utf8Main, out);
            return StringColwersionStatus::kSUCCESS;
        }
        __except (EXCEPTION_EXELWTE_HANDLER)
        {
            __try
            {
                getWstrFromUtf8Internal(utf8Fallback, out);
                return StringColwersionStatus::kFALLBACK;
            }
            __except (EXCEPTION_EXELWTE_HANDLER)
            {
                out = L"";
                return StringColwersionStatus::kFAIL;
            }
        }
    }
    std::wstring getWstrFromUtf8Fallback(const char * utf8Main, const char * utf8Fallback)
    {
        std::wstring out;
        getWstrFromUtf8Fallback(utf8Main, utf8Fallback, out);
        return out;
    }

    StringColwersionStatus getWstrFromUtf8Fallback(const std::string & utf8Main, const std::string & utf8Fallback, std::wstring & out)
    {
        __try
        {
            getWstrFromUtf8Internal(utf8Main.c_str(), out);
            return StringColwersionStatus::kSUCCESS;
        }
        __except (EXCEPTION_EXELWTE_HANDLER)
        {
            __try
            {
                getWstrFromUtf8Internal(utf8Fallback.c_str(), out);
                return StringColwersionStatus::kFALLBACK;
            }
            __except (EXCEPTION_EXELWTE_HANDLER)
            {
                out = L"";
                return StringColwersionStatus::kFAIL;
            }
        }
    }
    std::wstring getWstrFromUtf8Fallback(const std::string & utf8Main, const std::string & utf8Fallback)
    {
        std::wstring out;
        getWstrFromUtf8Fallback(utf8Main, utf8Fallback, out);
        return out;
    }

    // Base64 Encode/Decode Implementation taken from https://renenyffenegger.ch/notes/development/Base64/Encoding-and-decoding-base-64-with-cpp
    // Unrestricted license is specified.
    static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

    static inline bool is_base64(unsigned char c)
    {
        return (isalnum(c) || (c == '+') || (c == '/'));
    }

    std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len)
    {
        std::string ret;
        int i = 0;
        int j = 0;
        unsigned char char_array_3[3];
        unsigned char char_array_4[4];

        while (in_len--)
        {
            char_array_3[i++] = *(bytes_to_encode++);
            if (i == 3)
            {
                char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
                char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
                char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
                char_array_4[3] = char_array_3[2] & 0x3f;

                for (i = 0; (i < 4); i++)
                {
                    ret += base64_chars[char_array_4[i]];
                }
                i = 0;
            }
        }

        if (i)
        {
            for (j = i; j < 3; j++)
            {
                char_array_3[j] = '\0';
            }

            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

            for (j = 0; (j < i + 1); j++)
            {
                ret += base64_chars[char_array_4[j]];
            }

            while ((i++ < 3))
            {
                ret += '=';
            }
        }

        return ret;
    }

    std::string base64_encode(std::string const& s)
    {
        return std::string(base64_encode((const unsigned char*)(s.c_str()), static_cast<unsigned int>(s.size())));
    }
    std::wstring base64_encode(std::wstring const& ws)
    {
        std::string s = darkroom::getUtf8FromWstr(ws);
        return darkroom::getWstrFromUtf8(base64_encode(s));
    }

    std::string base64_decode(std::string const& encoded_string)
    {
        int in_len = static_cast<int>(encoded_string.size());
        int i = 0;
        int j = 0;
        int in_ = 0;
        unsigned char char_array_4[4], char_array_3[3];
        std::string ret;

        while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_]))
        {
            char_array_4[i++] = encoded_string[in_]; in_++;
            if (i == 4)
            {
                for (i = 0; i < 4; i++)
                {
                    char_array_4[i] = static_cast<unsigned char>(base64_chars.find(char_array_4[i]));
                }

                char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
                char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
                char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

                for (i = 0; (i < 3); i++)
                {
                    ret += char_array_3[i];
                }
                i = 0;
            }
        }

        if (i)
        {
            for (j = 0; j < i; j++)
            {
                char_array_4[j] = static_cast<unsigned char>(base64_chars.find(char_array_4[j]));
            }

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);

            for (j = 0; (j < i - 1); j++)
            {
                ret += char_array_3[j];
            }
        }

        return ret;
    }

    std::wstring base64_decode(std::wstring const& ws)
    {
        std::string s = darkroom::getUtf8FromWstr(ws);
        return darkroom::getWstrFromUtf8(base64_decode(s));
    }

    const std::string GuidToString(const GUID& guid)
    {
        char str[37] = { 0 }; // Ex: AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE
        snprintf( str, sizeof(str),
            "%08lX-%04hX-%04hX-%02hhX%02hhX-%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX",
            guid.Data1, guid.Data2, guid.Data3,
            guid.Data4[0], guid.Data4[1], guid.Data4[2],
            guid.Data4[3], guid.Data4[4], guid.Data4[5],
            guid.Data4[6], guid.Data4[7]);

        return str;
    }
}
