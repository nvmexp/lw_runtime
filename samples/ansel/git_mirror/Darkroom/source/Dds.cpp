#pragma warning(push)
#pragma warning(disable: 4668 4917)
#define WINDOWS_LEAN_AND_MEAN
#include <Windows.h>
#include <Wincodec.h>
#pragma warning(pop)

#include "darkroom/PixelFormat.h"
#include "darkroom/Errors.h"
#include "darkroom/StringColwersion.h"
#include "darkroom/Dds.h"

#include "lw_dds.h"

#include "ImageLoaderWIC.h"

namespace darkroom
{
    std::vector<unsigned char> loadDds(const std::wstring& filename, unsigned int& w, unsigned int& h, BufferFormat format)
    {
        // First try WIC - it is great with compressed DDS, but has some troubles with uncompressed DDS
        std::vector<unsigned char> result = loadImageWinApiInternal(filename, w, h, format);
        if (!result.empty())
            return result;

        // If WIC didn't work, use LWDDS; LWDDS however is hard to use with compression (we need uncompressed output)
        uint32_t channels = 3u;

        if (format == BufferFormat::BGR8 || format == BufferFormat::RGB8)
        {
            channels = 3u;
        }
        if (format == BufferFormat::BGRA8 || format == BufferFormat::RGBA8)
        {
            channels = 4u;
        }
        else
            return std::vector<unsigned char>();

        // According to lw_dds.cpp:
        // "Image is flipped when its loaded as DX images are stored with different coordinate system"
        // we don't need it flipped
        lw_dds::CDDSImage image;
        bool isOK = image.load(filename.c_str(), false);

        if (!isOK)
            return std::vector<unsigned char>();

        w = image.get_width();
        h = image.get_height();

        lw_dds::TextureFormat sourceFormat = image.get_format();
        unsigned int components = image.get_components();

        unsigned char * imgDataDDS = image;

        result.resize(size_t(uint64_t(w) * uint64_t(h) * uint64_t(channels)));

        bool forceColwersion = false;
        switch (sourceFormat)
        {
        case lw_dds::TextureFormat::TextureBGR:
            {
                if (format != BufferFormat::BGR8)
                    forceColwersion = true;
                break;
            }
        case lw_dds::TextureFormat::TextureBGRA:
            {
                if (format != BufferFormat::BGRA8)
                    forceColwersion = true;
                break;
            }
        case lw_dds::TextureFormat::TextureDXT1:
        case lw_dds::TextureFormat::TextureDXT3:
        case lw_dds::TextureFormat::TextureDXT5:
            {
                assert(false && "Compressed DDS should be processed with WIC!");
                break;
            }
        case lw_dds::TextureFormat::TextureUnknown:
        case lw_dds::TextureFormat::TextureLuminance:
        default:
            {
                forceColwersion = true;
                break;
            }
        }

        if (!forceColwersion)
            memcpy(result.data(), imgDataDDS, w*h*channels*sizeof(unsigned char));
        else
        {
            unsigned char * imgDataResult = result.data();
            for (uint32_t j = 0, jEnd = h; j < jEnd; ++j)
            {
                for (uint32_t i = 0, iEnd = w; i < iEnd; ++i)
                {
                    size_t offsetDDS = (i + j*w)*components;
                    int r = 0, g = 0, b = 0, a = 255;

                    // If image is not full RGB, propagate last known color channel (except alpha)
                    b = imgDataDDS[offsetDDS+0];
                    if (components > 1)
                    {
                        g = imgDataDDS[offsetDDS+1];
                    }
                    else
                    {
                        g = b;
                    }
                    if (components > 2)
                    {
                        r = imgDataDDS[offsetDDS+2];
                    }
                    else
                    {
                        r = g;
                    }
                    if (components > 3)
                    {
                        a = imgDataDDS[offsetDDS+3];
                    }

                    size_t offsetTarget = (i + j*w)*channels;

                    // We only support these 4 ATM
                    switch (format)
                    {
                    case BufferFormat::BGR8:
                        {
                            imgDataResult[offsetTarget+0] = (unsigned char)b;
                            imgDataResult[offsetTarget+1] = (unsigned char)g;
                            imgDataResult[offsetTarget+2] = (unsigned char)r;
                            break;
                        }
                    case BufferFormat::BGRA8:
                        {
                            imgDataResult[offsetTarget+0] = (unsigned char)b;
                            imgDataResult[offsetTarget+1] = (unsigned char)g;
                            imgDataResult[offsetTarget+2] = (unsigned char)r;
                            imgDataResult[offsetTarget+3] = (unsigned char)a;
                            break;
                        }
                    case BufferFormat::RGB8:
                        {
                            imgDataResult[offsetTarget+0] = (unsigned char)r;
                            imgDataResult[offsetTarget+1] = (unsigned char)g;
                            imgDataResult[offsetTarget+2] = (unsigned char)b;
                            break;
                        }
                    case BufferFormat::RGBA8:
                        {
                            imgDataResult[offsetTarget+0] = (unsigned char)r;
                            imgDataResult[offsetTarget+1] = (unsigned char)g;
                            imgDataResult[offsetTarget+2] = (unsigned char)b;
                            imgDataResult[offsetTarget+3] = (unsigned char)a;
                            break;
                        }
                    case BufferFormat::ARGB8:
                    case BufferFormat::ABGR8:
                    case BufferFormat::RGB16:
                    case BufferFormat::BGR16:
                    case BufferFormat::RGBA16:
                    case BufferFormat::BGRA16:
                    case BufferFormat::RGB32:
                    case BufferFormat::BGR32:
                    case BufferFormat::RGBA32:
                    case BufferFormat::BGRA32:
                    case BufferFormat::R32:
                    default:
                        {
                            assert(false && "Unsupported target format in DDS texture load!");
                        }
                    }
                }
            }
        }

        return result;
    }

    std::vector<unsigned char> loadDds(const std::string& filename, unsigned int& w, unsigned int& h, BufferFormat format)
    {
        return loadDds(darkroom::getWstrFromUtf8(filename), w, h, format);
    }

    Error loadDdsStreaming(const std::wstring& filename, ProcessDataCallback<unsigned char> callback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format)
    {
        // Streamed DDS loading is made via WIC - i.e. will have troubles with certain uncompressed DDS
        return loadImageStreamingWinApiInternal(filename, callback, maxBytes, w, h, format);
    }

    Error loadDdsStreaming(const std::string& filename, ProcessDataCallback<unsigned char> callback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format)
    {
        // Streamed DDS loading is made via WIC - i.e. will have troubles with certain uncompressed DDS
        return loadImageStreamingWinApiInternal(filename, callback, maxBytes, w, h, format);
    }
}
