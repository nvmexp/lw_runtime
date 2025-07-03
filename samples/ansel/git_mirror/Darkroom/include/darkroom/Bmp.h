#pragma once
#include <vector>
#include <array>
#include "darkroom/PixelFormat.h"
#include "darkroom/Errors.h"
#include "darkroom/ImageLoader.h"

namespace darkroom
{
    // generates BMPv3 header (54 bytes)
    std::vector<unsigned char> generateBmpHeader(unsigned int w, unsigned int h);
    // load BMP file, supports RGB8/BGR8/RGBA8/BGRA8
    std::vector<unsigned char> loadBmp(const std::wstring& filename, unsigned int& w, unsigned int& h, BufferFormat format = BufferFormat::BGR8);
    std::vector<unsigned char> loadBmp(const std::string& filename, unsigned int& w, unsigned int& h, BufferFormat format = BufferFormat::BGR8);
    // load BMP file in streaming mode, supports RGB8/BGR8/RGBA8/BGRA8
    Error loadBmpStreaming(const std::wstring& filename, ProcessDataCallback<unsigned char> callback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format = BufferFormat::BGR8);
    Error loadBmpStreaming(const std::string& filename, ProcessDataCallback<unsigned char> callback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format = BufferFormat::BGR8);
    // save BMP specifying pixel format (see BufferFormat)
    Error saveBmp(const unsigned char* pixels, const std::wstring& filename, unsigned int w, unsigned int h, BufferFormat format = BufferFormat::BGR8);
    Error saveBmp(const unsigned char* pixels, const std::string& filename, unsigned int w, unsigned int h, BufferFormat format = BufferFormat::BGR8);
}
