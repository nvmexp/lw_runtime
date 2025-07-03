#pragma once
#include <string>
#include <vector>

#include "darkroom/PixelFormat.h"
#include "darkroom/Errors.h"

namespace darkroom
{
    template<typename T>
    using ProcessDataCallback = Error(*)(const typename T* data, uint32_t rowsRead);

    // load BMPv4, PNG, TGA or JPEG, specifying target buffer color format
    std::vector<unsigned char> loadImage(const std::string& filename, unsigned int & w, unsigned int & h, BufferFormat format);
    // wide string version
    std::vector<unsigned char> loadImage(const std::wstring& filename, unsigned int & w, unsigned int & h, BufferFormat format);

    // return just the image size, handy when reading very large images
    Error getImageSize(const std::string& filename, unsigned int & w, unsigned int & h);
    Error getImageSize(const std::wstring& filename, unsigned int & w, unsigned int & h);
}
