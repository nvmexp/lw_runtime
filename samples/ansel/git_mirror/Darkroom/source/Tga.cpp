#include <array>
#pragma warning(push)
#pragma warning(disable:4668)
#define WINDOWS_LEAN_AND_MEAN
#include <Windows.h>
#pragma warning(pop)
#include <Shlwapi.h>
#include <assert.h>
#include "darkroom/Tga.h"

namespace
{
    using darkroom::BufferFormat;

    bool getFileTime(const wchar_t* filepath, uint64_t& dateTime)
    {
        HANDLE hFile = CreateFile(
            filepath,
            GENERIC_READ,
            0,
            NULL,
            OPEN_EXISTING,
            0,
            NULL
        );

        if (hFile != ILWALID_HANDLE_VALUE)
        {
            FILETIME lastwritetime;
            bool result = (GetFileTime(hFile, 0, 0, &lastwritetime) != 0);
            CloseHandle(hFile);
            dateTime = static_cast<uint64_t>(lastwritetime.dwLowDateTime) | (static_cast<uint64_t>(lastwritetime.dwHighDateTime) << 32);

            return result;
        }
        return false;
    }

    bool directoryExists(const wchar_t *path)
    {
        if (PathIsRoot(path))
            return true;

        if (PathIsDirectory(path))
            return true;

        return false;
    }

    bool createDirectoryRelwrsively(const wchar_t *path)
    {
        if (directoryExists(path))
            return true;

        wchar_t folder[MAX_PATH];
        ZeroMemory(folder, MAX_PATH * sizeof(wchar_t));

        const wchar_t* endBackslash = wcschr(path, L'\\'), *endFrontslash = wcschr(path, L'/');
        const wchar_t* endFirst = endBackslash < endFrontslash ? endBackslash : endFrontslash;
        const wchar_t* endLast = endBackslash >= endFrontslash ? endBackslash : endFrontslash;
        const wchar_t* end = endFirst ? endFirst : endLast; //if the pointer is zero, try the other one

        bool success = true;
        while (end != NULL)
        {
            wcsncpy_s(folder, path, end - path + size_t(1));

            if (!directoryExists(folder))
            {
                if (!CreateDirectory(folder, NULL))
                {
                    success = false;
                    break;
                }
            }

            end += 1;
            endBackslash = wcschr(end, L'\\');
            endFrontslash = wcschr(end, L'/');
            endFirst = endBackslash < endFrontslash ? endBackslash : endFrontslash;
            endLast = endBackslash >= endFrontslash ? endBackslash : endFrontslash;
            end = endFirst ? endFirst : endLast; //if the pointer is zero, try the other one
        }
        return success;
    }

    void getLwrrentPath(wchar_t * path, size_t maxSize)
    {
        GetModuleFileName(NULL, path, (DWORD)maxSize);
        // TODO[error]: add check if funciton failed
        for (ptrdiff_t i = ptrdiff_t(wcslen(path) - 1); i >= 0; --i)
        {
            if (path[i] == L'\\' || path[i] == L'/')
            {
                path[i] = L'\0';
                break;
            }
        }
    }

    class MappedFile
    {
    private:
        HANDLE m_fileHandle;
        HANDLE m_mappingHandle;
        unsigned char* m_mapAddr;
        unsigned char* m_endMapaddr;

        template <typename T>
        HANDLE openFile(const T* filename) { return HANDLE(); }
        template <>
        HANDLE openFile(const wchar_t* filename)
        {
            return CreateFileW(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        }

        template <>
        HANDLE openFile(const char* filename)
        {
            return CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        }

    public:
        MappedFile() :
            m_fileHandle(ILWALID_HANDLE_VALUE),
            m_mappingHandle(NULL),
            m_mapAddr(nullptr),
            m_endMapaddr(nullptr)
        {}

        template<typename T>
        MappedFile(const T* filename) :
            m_fileHandle(ILWALID_HANDLE_VALUE),
            m_mappingHandle(NULL),
            m_mapAddr(nullptr),
            m_endMapaddr(nullptr)
        {
            open(filename);
        }


        ~MappedFile()
        {
            close();
        }

        const unsigned char* begin() const
        {
            return m_mapAddr;
        }

        const unsigned char* end() const
        {
            return m_endMapaddr;
        }

        bool isValid() const
        {
            assert(m_endMapaddr >= m_mapAddr);

            return m_mapAddr != nullptr;
        }

        template <typename T>
        bool open(const T* filename)
        {
            if (m_mapAddr)
                return false; //already open

            m_fileHandle = openFile(filename);

            if (m_fileHandle == ILWALID_HANDLE_VALUE)
            {
                return false;
            }

            unsigned long fileSize = GetFileSize(m_fileHandle, NULL);

            if (!fileSize)
            {
                CloseHandle(m_fileHandle);
                m_fileHandle = ILWALID_HANDLE_VALUE;

                return false;
            }

            m_mappingHandle = CreateFileMapping(m_fileHandle, NULL, PAGE_READONLY, 0, 0, NULL);

            if (m_mappingHandle == NULL)
            {
                CloseHandle(m_fileHandle);
                m_fileHandle = ILWALID_HANDLE_VALUE;

                return false;
            }

            // Map the view and test the results.

            m_mapAddr = (unsigned char*)MapViewOfFile(m_mappingHandle, FILE_MAP_READ, 0, 0, 0);

            if (m_mapAddr == NULL)
            {
                CloseHandle(m_mappingHandle);
                m_mappingHandle = NULL;
                CloseHandle(m_fileHandle);
                m_fileHandle = ILWALID_HANDLE_VALUE;

                return false;
            }

            m_endMapaddr = m_mapAddr + fileSize;

            return true;
        }

        void close()
        {
            if (m_mapAddr)
                UnmapViewOfFile(m_mapAddr);

            m_mapAddr = m_endMapaddr = nullptr;

            if (m_mappingHandle != NULL)
                CloseHandle(m_mappingHandle);

            m_mappingHandle = NULL;

            if (m_fileHandle != ILWALID_HANDLE_VALUE)
                CloseHandle(m_fileHandle);

            m_fileHandle = ILWALID_HANDLE_VALUE;
        }
    };

    bool loadTGA_Uncompressed(unsigned int w, unsigned int h, unsigned char alpha, darkroom::BufferFormat format, unsigned char * imageData, unsigned int bytesPerPixel, unsigned int outputBpp, const unsigned char* ptr, const unsigned char* endptr)
    {
        unsigned char colorBuffer[4];   // Max 4 bytes per pixel
        const int imageByteSize = int(bytesPerPixel * w * h);
        for (int y = 0; y < int(h); ++y)
        {
            int yOffset = int(h - y - 1) * int(w);
            for (int x = 0; x < int(w); ++x)
            {
                if (ptr + bytesPerPixel * sizeof(unsigned char) <= endptr)
                {
                    memcpy(colorBuffer, ptr, bytesPerPixel * sizeof(unsigned char));
                    ptr += bytesPerPixel * sizeof(unsigned char);
                }
                else
                    return false;

                const auto offset = (x + yOffset) * outputBpp;
                if (format == darkroom::BufferFormat::RGBA8)
                {
                    imageData[offset] = colorBuffer[0];
                    imageData[offset + 1] = colorBuffer[1];
                    imageData[offset + 2] = colorBuffer[2];
                    imageData[offset + 3] = (bytesPerPixel == 4) ? colorBuffer[3] : alpha;
                }
                else if (format == darkroom::BufferFormat::BGRA8)
                {
                    imageData[offset] = colorBuffer[2];
                    imageData[offset + 1] = colorBuffer[1];
                    imageData[offset + 2] = colorBuffer[0];
                    imageData[offset + 3] = (bytesPerPixel == 4) ? colorBuffer[3] : alpha;
                }
                else if (format == darkroom::BufferFormat::RGB8)
                {
                    imageData[offset] = colorBuffer[0];
                    imageData[offset + 1] = colorBuffer[1];
                    imageData[offset + 2] = colorBuffer[2];
                }
                else if (format == darkroom::BufferFormat::BGR8)
                {
                    imageData[offset] = colorBuffer[2];
                    imageData[offset + 1] = colorBuffer[1];
                    imageData[offset + 2] = colorBuffer[0];
                }
            }
        }
        return true;
    }

    bool loadTGA_Compressed(unsigned int w, unsigned int h, unsigned char alpha, darkroom::BufferFormat format, unsigned char * imageData, unsigned int bytesPerPixel, unsigned int outputBpp, const unsigned char* ptr, const unsigned char* endptr)
    {
        unsigned int byteCount = 4 * w * h;
        unsigned int lwrrentByte = 0;
        unsigned char colorBuffer[4];   // Max 4 bytes per pixel

        do
        {
            if (ptr <= endptr)
            {
                unsigned char chunkheader = ptr[0];
                ptr += 1;

                if (chunkheader < 128)
                {
                    chunkheader++;
                    for (int counter = 0; counter < chunkheader; ++counter)
                    {
                        if (ptr + bytesPerPixel * sizeof(unsigned char) <= endptr)
                        {
                            memcpy(colorBuffer, ptr, bytesPerPixel * sizeof(unsigned char));
                            ptr += bytesPerPixel * sizeof(unsigned char);
                        }
                        else return false;

                        const auto lwrrentPixel = (lwrrentByte >> 2);
                        const auto x = lwrrentPixel % w;
                        const auto y = lwrrentPixel / w;
                        const auto offset = (x + (h - y - 1)*w) * outputBpp;
                        if (format == darkroom::BufferFormat::RGBA8)
                        {
                            imageData[offset] = colorBuffer[0];
                            imageData[offset + 1] = colorBuffer[1];
                            imageData[offset + 2] = colorBuffer[2];
                            imageData[offset + 3] = (bytesPerPixel == 4) ? colorBuffer[3] : alpha;
                        }
                        else if (format == darkroom::BufferFormat::BGRA8)
                        {
                            imageData[offset] = colorBuffer[2];
                            imageData[offset + 1] = colorBuffer[1];
                            imageData[offset + 2] = colorBuffer[0];
                            imageData[offset + 3] = (bytesPerPixel == 4) ? colorBuffer[3] : alpha;
                        }
                        else if (format == darkroom::BufferFormat::RGB8)
                        {
                            imageData[offset] = colorBuffer[0];
                            imageData[offset + 1] = colorBuffer[1];
                            imageData[offset + 2] = colorBuffer[2];
                        }
                        else if (format == darkroom::BufferFormat::BGR8)
                        {
                            imageData[offset] = colorBuffer[2];
                            imageData[offset + 1] = colorBuffer[1];
                            imageData[offset + 2] = colorBuffer[0];
                        }
                        lwrrentByte += 4;

                        // Too many pixels read, something went wrong
                        if (lwrrentByte > byteCount)
                            return false;
                    }
                }
                else
                {
                    chunkheader -= 127;

                    if (ptr + bytesPerPixel * sizeof(unsigned char) <= endptr)
                    {
                        memcpy(colorBuffer, ptr, bytesPerPixel * sizeof(unsigned char));
                        ptr += bytesPerPixel * sizeof(unsigned char);
                    }
                    else return false;

                    for (int counter = 0; counter < chunkheader; ++counter)
                    {
                        const auto lwrrentPixel = (lwrrentByte >> 2);
                        const auto x = lwrrentPixel % w;
                        const auto y = lwrrentPixel / w;
                        const auto offset = (x + (h - y - 1)*w) << 2;
                        if (format == darkroom::BufferFormat::RGBA8)
                        {
                            imageData[offset] = colorBuffer[0];
                            imageData[offset + 1] = colorBuffer[1];
                            imageData[offset + 2] = colorBuffer[2];
                            imageData[offset + 3] = (bytesPerPixel == 4) ? colorBuffer[3] : alpha;
                        }
                        else if (format == darkroom::BufferFormat::BGRA8)
                        {
                            imageData[offset] = colorBuffer[2];
                            imageData[offset + 1] = colorBuffer[1];
                            imageData[offset + 2] = colorBuffer[0];
                            imageData[offset + 3] = (bytesPerPixel == 4) ? colorBuffer[3] : alpha;
                        }
                        lwrrentByte += 4;

                        // Too many pixels read, something went wrong
                        if (lwrrentByte > byteCount)
                            return false;
                    }
                }

            }
            // Could not read RLE header
            else return false;

        } while (lwrrentByte < byteCount);

        return true;
    }

    template<typename T>
    std::vector<unsigned char> loadTGA(const T* filename, unsigned int & w, unsigned int & h, darkroom::BufferFormat format)
    {
        w = 0;
        h = 0;

        const std::array<BufferFormat, 4> supportedFormats = { BufferFormat::BGR8, BufferFormat::RGB8, BufferFormat::BGRA8, BufferFormat::RGBA8 };
        // in case the format is not supported return empty result
        if (std::find(supportedFormats.cbegin(), supportedFormats.cend(), format) == supportedFormats.cend())
            return std::vector<unsigned char>();

        const auto outputBpp = (format == BufferFormat::BGR8 || format == BufferFormat::RGB8) ? 3u : 4u;

        MappedFile in(filename);

        if (!in.isValid())
            return std::vector<unsigned char>();

        const size_t tgaReadHeaderSize = 12;
        unsigned char tgaReadHeader[tgaReadHeaderSize];

        unsigned char uTGAcompare[tgaReadHeaderSize] = { 0,0, 2,0,0,0,0,0,0,0,0,0 };    // Uncompressed TGA Header
        unsigned char cTGAcompare[tgaReadHeaderSize] = { 0,0,10,0,0,0,0,0,0,0,0,0 };    // Compressed TGA Header

        const unsigned char* mapAddr = in.begin(), *endMapaddr = in.end();

        if (mapAddr + tgaReadHeaderSize * sizeof(unsigned char) <= endMapaddr)
        {
            memcpy(reinterpret_cast<unsigned char *>(tgaReadHeader), mapAddr, tgaReadHeaderSize * sizeof(unsigned char));
            mapAddr += tgaReadHeaderSize * sizeof(unsigned char);
        }
        else
        {
            return std::vector<unsigned char>();
        }

        const size_t tgaImgDataHeaderSize = 6;
        unsigned char tgaImgDataHeader[tgaImgDataHeaderSize];

        if (mapAddr + tgaImgDataHeaderSize * sizeof(unsigned char) <= endMapaddr)
        {
            memcpy((char *)tgaImgDataHeader, mapAddr, tgaImgDataHeaderSize * sizeof(unsigned char));
            mapAddr += tgaImgDataHeaderSize * sizeof(unsigned char);
        }
        else
        {
            // Could not read info header
            return std::vector<unsigned char>();
        }

        int width, height, bitDepth;
        width = tgaImgDataHeader[1] * 256 + tgaImgDataHeader[0];
        height = tgaImgDataHeader[3] * 256 + tgaImgDataHeader[2];
        bitDepth = tgaImgDataHeader[4];

        if ((width <= 0) || (height <= 0) || ((bitDepth != 24) && (bitDepth != 32)))
        {
            // Invalid texture information
            return std::vector<unsigned char>();
        }

        w = static_cast<unsigned int>(width);
        h = static_cast<unsigned int>(height);

        unsigned int bytesPerPixel = static_cast<unsigned int>(bitDepth / 8);
        const size_t imageSize = (outputBpp * w * h);
        std::vector<unsigned char> image(imageSize);

        if (image.empty())
            return std::vector<unsigned char>();

        // for 3bpp formats we set alpha to 255 in case 4bpp format was requested
        const auto alpha = 255;

        if (memcmp(uTGAcompare, tgaReadHeader, tgaReadHeaderSize * sizeof(unsigned char)) == 0)
        {
            if (loadTGA_Uncompressed(w, h, alpha, format, &image[0], bytesPerPixel, outputBpp, mapAddr, endMapaddr))
                return image;
        }
        else if (memcmp(cTGAcompare, tgaReadHeader, tgaReadHeaderSize * sizeof(unsigned char)) == 0)
        {
            if (loadTGA_Compressed(w, h, alpha, format, &image[0], bytesPerPixel, outputBpp, mapAddr, endMapaddr))
                return image;
        }
        else
        {
            // TGA file be type 2 or type 10
            return std::vector<unsigned char>();
        }

        return std::vector<unsigned char>();
    }
}

namespace darkroom
{
    std::vector<unsigned char> loadTga(const std::wstring& filename, unsigned int& w, unsigned int& h, BufferFormat format)
    {
        return loadTGA(filename.c_str(), w, h, format);
    }

    std::vector<unsigned char> loadTga(const std::string& filename, unsigned int& w, unsigned int& h, BufferFormat format)
    {
        return loadTGA(filename.c_str(), w, h, format);
    }
}
