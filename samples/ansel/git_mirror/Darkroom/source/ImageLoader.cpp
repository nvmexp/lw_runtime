#include "darkroom/ImageLoader.h"
#include <algorithm>
#pragma warning(push)
#pragma warning(disable:4668)
#define WINDOWS_LEAN_AND_MEAN
#include <Windows.h>
#include <Shlwapi.h>
#pragma warning(pop)

#include "darkroom/StringColwersion.h"
#include "darkroom/Bmp.h"
#include "darkroom/Jpeg.h"
#include "darkroom/Png.h"
#include "darkroom/Tga.h"
#include "darkroom/Dds.h"
#include "ImageLoaderWIC.h"

namespace
{
    std::wstring findExtension(const std::wstring& filename)
    {
        const wchar_t* extPtr = PathFindExtension(filename.c_str());
        if (*extPtr != '\0')
            return std::wstring(extPtr);
        else
            return std::wstring();
    }

    std::string findExtension(const std::string& filename)
    {
        const char* extPtr = PathFindExtensionA(filename.c_str());
        if (*extPtr != '\0')
            return std::string(extPtr);
        else
            return std::string();
    }

    template <typename T> class Extensions;
    template <> struct Extensions<std::string>
    {
        static const char* kBmp;
        static const char* kPng;
        static const char* kTga;
        static const char* kJpg;
        static const char* kJpeg;
        static const char* kJp2;
        static const char* kDds;
    };
    template <> struct Extensions<std::wstring>
    {
        static const wchar_t* kBmp;
        static const wchar_t* kPng;
        static const wchar_t* kTga;
        static const wchar_t* kJpg;
        static const wchar_t* kJpeg;
        static const wchar_t* kJp2;
        static const wchar_t* kDds;
    };

    const char* Extensions<std::string>::kBmp = ".bmp";
    const char* Extensions<std::string>::kPng = ".png";
    const char* Extensions<std::string>::kTga = ".tga";
    const char* Extensions<std::string>::kJpg = ".jpg";
    const char* Extensions<std::string>::kJpeg = ".jpeg";
    const char* Extensions<std::string>::kJp2 = ".jp2";
    const char* Extensions<std::string>::kDds = ".dds";
    const wchar_t* Extensions<std::wstring>::kBmp = L".bmp";
    const wchar_t* Extensions<std::wstring>::kPng = L".png";
    const wchar_t* Extensions<std::wstring>::kTga = L".tga";
    const wchar_t* Extensions<std::wstring>::kJpg = L".jpg";
    const wchar_t* Extensions<std::wstring>::kJpeg = L".jpeg";
    const wchar_t* Extensions<std::wstring>::kJp2 = L".jp2";
    const wchar_t* Extensions<std::wstring>::kDds = L".dds";


    // Common
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    template<typename T>
    std::vector<unsigned char> loadImageInternal(const T& filename, unsigned int & w, unsigned int & h, darkroom::BufferFormat format)
    {
        auto ext = findExtension(filename);
        darkroom::tolowerInplace(ext);

        if (ext.empty())
            return std::vector<unsigned char>();

        if (ext == Extensions<T>::kBmp)
            return darkroom::loadBmp(filename, w, h, format);
        else if (ext == Extensions<T>::kTga)
            return darkroom::loadTga(filename, w, h, format);
        else if (ext == Extensions<T>::kJpg || ext == Extensions<T>::kJpeg)
        {
            std::unordered_map<uint16_t, std::string> tags;
            std::vector<char> xmpPacket;
            return darkroom::loadJpeg(filename, w, h, format, tags, xmpPacket);
        }
        else if (ext == Extensions<T>::kPng)
        {
            std::unordered_map<std::string, std::string> tags;
            return darkroom::loadPng(filename, w, h, format, tags);
        }
        else if (ext == Extensions<T>::kDds)
        {
            return darkroom::loadDds(filename, w, h, format);
        }
        return std::vector<unsigned char>();
    }
}

namespace darkroom
{
    std::vector<unsigned char> loadImage(const std::string& filename, unsigned int & w, unsigned int & h, BufferFormat format)
    {
        return loadImageInternal(filename, w, h, format);
    }

    std::vector<unsigned char> loadImage(const std::wstring& filename, unsigned int & w, unsigned int & h, BufferFormat format)
    {
        return loadImageInternal(filename, w, h, format);
    }

    Error getImageSize(const std::string& filename, unsigned int & w, unsigned int & h)
    {
        return getImageSizeWinApiInternal(filename, w, h);
    }

    Error getImageSize(const std::wstring& filename, unsigned int & w, unsigned int & h)
    {
        return getImageSizeWinApiInternal(filename, w, h);
    }
}
