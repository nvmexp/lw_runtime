#pragma once
#include <darkroom/PixelFormat.h>
#include <darkroom/Errors.h>

#include <stdint.h>
#include <vector>
#include <string>

namespace darkroom
{
    // load tga file, RGB8/BGR8/RGBA8/BGRA8 target formats are supported
    std::vector<unsigned char> loadTga(const std::string& filename, unsigned int& w, unsigned int& h, BufferFormat format);
    // wide string version
    std::vector<unsigned char> loadTga(const std::wstring& filename, unsigned int& w, unsigned int& h, BufferFormat format);
}
