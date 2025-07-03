#pragma once
#include <string>
#include <vector>

#include "darkroom/PixelFormat.h"
#include "darkroom/Errors.h"
#include "darkroom/ImageLoader.h"

namespace darkroom
{
    std::vector<unsigned char> loadImageWinApiInternal(const std::wstring& filename, unsigned int& w, unsigned int& h, BufferFormat format = BufferFormat::BGR8);
    std::vector<unsigned char> loadImageWinApiInternal(const std::string& filename, unsigned int& w, unsigned int& h, BufferFormat format = BufferFormat::BGR8);

    Error loadImageStreamingWinApiInternal(const std::wstring& filename, ProcessDataCallback<unsigned char> callback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format = BufferFormat::BGR8);
    Error loadImageStreamingWinApiInternal(const std::string& filename, ProcessDataCallback<unsigned char> callback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format = BufferFormat::BGR8);

    // return just the image size, handy when reading very large images
    Error getImageSizeWinApiInternal(const std::string& filename, unsigned int & w, unsigned int & h);
    Error getImageSizeWinApiInternal(const std::wstring& filename, unsigned int & w, unsigned int & h);
}
