#pragma once

#include <vector>
#include <string>

#include "darkroom/Errors.h"
#include "darkroom/ImageLoader.h"

namespace darkroom
{
    std::vector<unsigned char> loadDds(const std::wstring& filename, unsigned int& w, unsigned int& h, BufferFormat format);
    std::vector<unsigned char> loadDds(const std::string& filename, unsigned int& w, unsigned int& h, BufferFormat format);

    Error loadDdsStreaming(const std::wstring& filename, ProcessDataCallback<unsigned char> callback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format);
    Error loadDdsStreaming(const std::string& filename, ProcessDataCallback<unsigned char> callback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format);
}