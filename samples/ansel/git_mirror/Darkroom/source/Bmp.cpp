#pragma warning(disable: 4514 4711 4710)

#pragma warning(push)
#pragma warning(disable: 4668 4917)
#define WINDOWS_LEAN_AND_MEAN
#include <Windows.h>
#include <Wincodec.h>
#pragma warning(pop)
#include <fstream>
#include "ImageLoaderWIC.h"
#include "darkroom/Bmp.h"
#include "darkroom/StringColwersion.h"

#pragma comment(lib, "windowscodecs.lib")

namespace darkroom
{
    std::vector<unsigned char> generateBmpHeader(unsigned int w, unsigned int h)
    {
        BITMAPINFOHEADER bmpheader = { 0 };
        bmpheader.biSize = 40;
        bmpheader.biWidth = static_cast<int>(w);
        bmpheader.biHeight = -static_cast<int>(h);
        bmpheader.biPlanes = 1;
        bmpheader.biBitCount = 24;

        std::vector<unsigned char> result{ 'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0 };
        result.resize(54);
        unsigned char* bmphdr = reinterpret_cast<unsigned char*>(&bmpheader);
        std::copy(bmphdr, bmphdr + sizeof(bmpheader), result.begin() + 14);
        return result;
    }
}

namespace
{
    using darkroom::BufferFormat;

    std::vector<unsigned char> encodeBmp(const unsigned char* image, unsigned int w, unsigned int h, BufferFormat format)
    {
        auto bmpheader = darkroom::generateBmpHeader(w, h);
        const auto rowSize = static_cast<size_t>(3 * w);
        auto bytesPerRow = rowSize;
        bytesPerRow = bytesPerRow % 4 == 0 ? bytesPerRow : bytesPerRow + (4 - bytesPerRow % 4);

        std::vector<unsigned char> data;
        data.resize(bytesPerRow * h);

        size_t i = 0;
        if (format == BufferFormat::BGR8)
        {
            for (size_t y = 0; y < h; ++y)
            {
                for (size_t x = 0; x < w; ++x)
                {
                    data[i++] = image[3 * (w * y + x) + 0];
                    data[i++] = image[3 * (w * y + x) + 1];
                    data[i++] = image[3 * (w * y + x) + 2];
                }
                for (size_t x = 0; x < bytesPerRow - rowSize; ++x)
                    data[i++] = 0;
            }
        }
        else if (format == BufferFormat::RGB8)
        {
            for (size_t y = 0; y < h; ++y)
            {
                for (size_t x = 0; x < w; ++x)
                {
                    data[i++] = image[3 * (w * y + x) + 2];
                    data[i++] = image[3 * (w * y + x) + 1];
                    data[i++] = image[3 * (w * y + x) + 0];
                }
                for (size_t x = 0; x < bytesPerRow - rowSize; ++x)
                    data[i++] = 0;
            }
        }
        else if (format == BufferFormat::BGRA8)
        {
            for (size_t y = 0; y < h; ++y)
            {
                for (size_t x = 0; x < w; ++x)
                {
                    data[i++] = image[4 * (w * y + x) + 0];
                    data[i++] = image[4 * (w * y + x) + 1];
                    data[i++] = image[4 * (w * y + x) + 2];
                }
                for (size_t x = 0; x < bytesPerRow - rowSize; ++x)
                    data[i++] = 0;
            }
        }
        else if (format == BufferFormat::RGBA8)
        {
            for (size_t y = 0; y < h; ++y)
            {
                for (size_t x = 0; x < w; ++x)
                {
                    data[i++] = image[4 * (w * y + x) + 2];
                    data[i++] = image[4 * (w * y + x) + 1];
                    data[i++] = image[4 * (w * y + x) + 0];
                }
                for (size_t x = 0; x < bytesPerRow - rowSize; ++x)
                    data[i++] = 0;
            }
        }
        else if (format == BufferFormat::ABGR8)
        {
            for (size_t y = 0; y < h; ++y)
            {
                for (size_t x = 0; x < w; ++x)
                {
                    data[i++] = image[4 * (w * y + x) + 1];
                    data[i++] = image[4 * (w * y + x) + 2];
                    data[i++] = image[4 * (w * y + x) + 3];
                }
                for (size_t x = 0; x < bytesPerRow - rowSize; ++x)
                    data[i++] = 0;
            }
        }
        else if (format == BufferFormat::ARGB8)
        {
            for (size_t y = 0; y < h; ++y)
            {
                for (size_t x = 0; x < w; ++x)
                {
                    data[i++] = image[4 * (w * y + x) + 3];
                    data[i++] = image[4 * (w * y + x) + 2];
                    data[i++] = image[4 * (w * y + x) + 1];
                }
                for (size_t x = 0; x < bytesPerRow - rowSize; ++x)
                    data[i++] = 0;
            }
        }
        std::vector<unsigned char> result;
        result.reserve(bmpheader.size() + data.size());
        result.insert(result.end(), bmpheader.begin(), bmpheader.end());
        result.insert(result.end(), data.begin(), data.end());

        result[2] = result.size() % 256;
        result[3] = (result.size() / 256) % 256;
        result[4] = (result.size() / 65536) % 256;
        result[5] = static_cast<unsigned char>(result.size() / 16777216);
        return result;
    }
}

namespace darkroom
{
    Error saveBmp(const unsigned char* bgr, const std::wstring& filename, unsigned int w, unsigned int h, BufferFormat format)
    {
        std::ofstream out(filename, std::ofstream::binary);
        if (out)
        {
            std::vector<unsigned char> bmp = encodeBmp(bgr, w, h, format);
            out.write(reinterpret_cast<char*>(bmp.data()), bmp.size());
            return Error::kSuccess;
        }
        else
            return Error::kCouldntCreateFile;
    }

    Error saveBmp(const unsigned char* bgr, const std::string& filename, unsigned int w, unsigned int h, BufferFormat format)
    {
        std::ofstream out(filename, std::ofstream::binary);
        if (out)
        {
            std::vector<unsigned char> bmp = encodeBmp(bgr, w, h, format);
            out.write(reinterpret_cast<char*>(bmp.data()), bmp.size());
            return Error::kSuccess;
        }
        else
            return Error::kCouldntCreateFile;
    }

    std::vector<unsigned char> loadBmp(const std::wstring& filename, unsigned int& w, unsigned int& h, BufferFormat format)
    {
        return loadImageWinApiInternal(filename, w, h, format);
    }

    std::vector<unsigned char> loadBmp(const std::string& filename, unsigned int& w, unsigned int& h, BufferFormat format)
    {
        return loadImageWinApiInternal(filename, w, h, format);
    }

    Error loadBmpStreaming(const std::wstring& filename, ProcessDataCallback<unsigned char> callback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format)
    {
        return loadImageStreamingWinApiInternal(filename, callback, maxBytes, w, h, format);
    }

    Error loadBmpStreaming(const std::string& filename, ProcessDataCallback<unsigned char> callback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format)
    {
        return loadImageStreamingWinApiInternal(filename, callback, maxBytes, w, h, format);
    }
}
