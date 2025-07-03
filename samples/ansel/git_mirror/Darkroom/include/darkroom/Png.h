#pragma once
#include <vector>
#include <unordered_map>
#include "darkroom/PixelFormat.h"
#include "darkroom/Errors.h"
#include "darkroom/Metadata.h"
#include "darkroom/ImageLoader.h"

namespace darkroom
{
    using RemapDataCallback = const void*(*)(unsigned int& rowsMapped);

    // load PNG file, supports RGB8/BGR8/RGBA8/BGRA8
    std::vector<unsigned char> loadPng(const std::string& filename, unsigned int& w, unsigned int& h, BufferFormat format, std::unordered_map<std::string, std::string>& tags);
    std::vector<unsigned char> loadPng(const std::wstring& filename, unsigned int& w, unsigned int& h, BufferFormat format, std::unordered_map<std::string, std::string>& tags);

    // load PNG file, supports RGB8/BGR8/RGBA8/BGRA8
    Error loadPngStreaming(const std::string& filename, ProcessDataCallback<unsigned char> callback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format);
    Error loadPngStreaming(const std::wstring& filename, ProcessDataCallback<unsigned char> callback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format);

    // save color buffer as PNG file
    // compression - 0 - no compression
    //               1 - best speed
    //               9 - best compression (but slower)
    // format - RGB8/BGR8/RGBA8/BGRA8 are supported
    // threadCount - all available if 0 or as specified
    // Exactly one of the two arguments - 'pixels' or 'remapCallback' should be not nullptr.
    // In case remapCallback is not nullptr, memory conservative mode is used where the function calls remapCallback to perform memory mapping of the next portion of file
    // The callback should return an address aligned to the row boundary and fill rowsMapped accordingly. The callback can be called multiple times, so it should slide the mapping
    // window accordingly.
    // in case remapCallback is nullptr, simple mode is used, where it's assumed all data is available through 'rgb' pointer
    Error savePng(const unsigned char* pixels, 
        RemapDataCallback remapCallback, 
        const std::string& filename, 
        unsigned int width, 
        unsigned int height, 
        BufferFormat format,
        const std::unordered_map<std::string, std::string>& tags = std::unordered_map<std::string, std::string>(),
        int compression = 2, 
        unsigned int threadCount = 0u);
    // wide string version
    Error savePng(const unsigned char* pixels, 
        RemapDataCallback remapCallback, 
        const std::wstring& filename, 
        unsigned int width, 
        unsigned int height, 
        BufferFormat format, 
        const std::unordered_map<std::string, std::string>& tags = std::unordered_map<std::string, std::string>(),
        int compression = 2,
        unsigned int threadCount = 0u);

    // Metadata manipulation
    void colwertToImageMetadata(const std::unordered_map<std::string, std::string> & tags, ImageMetadata & metadata);
    void colwertFromImageMetadata(const ImageMetadata & metadata, std::unordered_map<std::string, std::string> & tags);
}
