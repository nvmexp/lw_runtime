#pragma once
#include <vector>
#include <unordered_map>
#include "darkroom/PixelFormat.h"
#include "darkroom/ImageLoader.h"
#include "darkroom/Errors.h"

namespace darkroom
{
    using RemapDataCallback = const void*(*)(unsigned int& rowsMapped);

    // loads EXR file and fills in width 'w' and height 'h'
    std::vector<float> loadExr(const std::string& filename, unsigned int& w, unsigned int& h, std::unordered_map<std::string, std::string>& tags, darkroom::BufferFormat fmt = darkroom::BufferFormat::RGB32);
    // wide string version
    std::vector<float> loadExr(const std::wstring& filename, unsigned int& w, unsigned int& h, std::unordered_map<std::string, std::string>& tags, darkroom::BufferFormat fmt = darkroom::BufferFormat::RGB32);

    // loads EXR file and fills in width 'w' and height 'h'
    Error loadExrStreaming(const std::string& filename, ProcessDataCallback<float> callback, size_t maxBytes, unsigned int& w, unsigned int& h, std::unordered_map<std::string, std::string>& tags, darkroom::BufferFormat fmt = darkroom::BufferFormat::RGB32);
    // wide string version
    Error loadExrStreaming(const std::wstring& filename, ProcessDataCallback<float> callback, size_t maxBytes, unsigned int& w, unsigned int& h, std::unordered_map<std::string, std::string>& tags, darkroom::BufferFormat fmt = darkroom::BufferFormat::RGB32);

    // return just the image size, handy when reading very large images
    Error getExrImageSize(const std::string& filename, unsigned int & w, unsigned int & h);
    Error getExrImageSize(const std::wstring& filename, unsigned int & w, unsigned int & h);

    // save color buffer as EXR file
    // format - only RGBA16 (fp16 for each channel) is supported for now (RGBA32 and other variations will come later)
    // threadCount - all available if 0 or as specified
    // Exactly one of the two arguments - 'pixels' or 'remapCallback' should be not nullptr.
    // In case remapCallback is not nullptr, memory conservative mode is used where the function calls remapCallback to perform memory mapping of the next portion of file
    // The callback should return an address aligned to the row boundary and fill rowsMapped accordingly. The callback can be called multiple times, so it should slide the mapping
    // window accordingly.
    // in case remapCallback is nullptr, simple mode is used, where it's assumed all data is available through 'rgb' pointer
    Error saveExr(const void* pixels,
        RemapDataCallback remapCallback,
        const std::string& filename,
        unsigned int width,
        unsigned int height,
        BufferFormat format,
        const std::unordered_map<std::string, std::string>& tags);
    // wide string version
    Error saveExr(const void* pixels,
        RemapDataCallback remapCallback,
        const std::wstring& filename,
        unsigned int width,
        unsigned int height,
        BufferFormat format,
        const std::unordered_map<std::string, std::string>& tags);
}
