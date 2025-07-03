#pragma once
#include <stdint.h>
#include <vector>
#include <unordered_map>
#include <array>
#include "darkroom/PixelFormat.h"
#include "darkroom/Errors.h"
#include "darkroom/Metadata.h"
#include "darkroom/ImageLoader.h"

namespace darkroom
{
    const uint16_t gJPEG_TAG_DESCRIPTION = 270;         // Exif.Image.ImageDescription
    const uint16_t gJPEG_TAG_SOURCE = 271;              // Exif.Image.Make
    const uint16_t gJPEG_TAG_MODEL_1 = 272;             // Exif.Image.Model
    const uint16_t gJPEG_TAG_MODEL_2 = 50708;           // Exif.Image.UniqueCameraModel
    const uint16_t gJPEG_TAG_SOFTWARE = 305;            // Exif.Image.Software
    const uint16_t gJPEG_TAG_TYPE = 37500;              // Exif.Photo.MakerNote
    const uint16_t gJPEG_TAG_DRSNAME = 11;              // Exif.Image.ProcessingSoftware
    const uint16_t gJPEG_TAG_DRSPROFILENAME = 50934;    // Exif.Image.AsShotProfileName
    const uint16_t gJPEG_TAG_APPTITLENAME = 42032;      // Exif.Photo.CameraOwnerName
    const uint16_t gJPEG_TAG_APPCMSID = 50735;          // Exif.Image.CameraSerialNumber
    const uint16_t gJPEG_TAG_APPSHORTNAME = 51105;      // Exif.Image.CameraLabel
    const uint16_t gJPEG_TAG_ACTIVEFILTERS = 37395;     // Exif.Image.ImageHistory

    enum class JpegStreamingQuality
    {
        kLow,
        kHigh
    };

    enum class JpegStreamingScale
    {
        k1_1,       // 1x
        k1_2,       // 0.5x
        k1_3,       // 0.3333x
        k1_4,       // 0.25x
        k1_5,       // 0.2x
        k1_6,       // 0.16666x
        k1_7,       // 0.142857x
        k1_8        // 0.125x
    };

    using RemapDataCallback = const void*(*)(unsigned int& rowsMapped);

    // save JPEG specifying quality and pixel format. Supports RGB8/BGR8/RGBA8/BGRA8 formats
    Error saveJpeg(const unsigned char* rgb,
        RemapDataCallback remapCallback,
        const std::string& filename, 
        unsigned int width, 
        unsigned int height, 
        BufferFormat format, 
        const std::unordered_map<uint16_t, std::string>& tags = std::unordered_map<uint16_t, std::string>(),
        const std::vector<char>& xmpPacket = std::vector<char>(),
        unsigned int jpegQuality = 100);
    // wide string version
    Error saveJpeg(const unsigned char* rgb,
        RemapDataCallback remapCallback,
        const std::wstring& filename,
        unsigned int width,
        unsigned int height,
        BufferFormat format,
        const std::unordered_map<uint16_t, std::string>& tags = std::unordered_map<uint16_t, std::string>(),
        const std::vector<char>& xmpPacket = std::vector<char>(),
        unsigned int jpegQuality = 100);

    // load jpeg file, RGB8/BGR8/RGBA8/BGRA8 target formats are supported
    std::vector<unsigned char> loadJpeg(const std::string& filename, unsigned int& w, unsigned int& h, BufferFormat format, std::unordered_map<uint16_t, std::string>& tags,
        std::vector<char>& xmpPacket);
    // wide string version
    std::vector<unsigned char> loadJpeg(const std::wstring& filename, unsigned int& w, unsigned int& h, BufferFormat format, std::unordered_map<uint16_t, std::string>& tags,
        std::vector<char>& xmpPacket);

    // streaming version of load jpeg file, RGB8/BGR8/RGBA8/BGRA8 target formats are supported
    Error loadJpegStreaming(const std::string& filename, ProcessDataCallback<unsigned char> mapCallback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format, std::unordered_map<uint16_t, std::string>& tags,
        std::vector<char>& xmpPacket, JpegStreamingQuality quality = JpegStreamingQuality::kHigh, JpegStreamingScale scale = JpegStreamingScale::k1_1);

    // wide string version
    Error loadJpegStreaming(const std::wstring& filename, ProcessDataCallback<unsigned char> mapCallback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format, std::unordered_map<uint16_t, std::string>& tags,
        std::vector<char>& xmpPacket, JpegStreamingQuality quality = JpegStreamingQuality::kHigh, JpegStreamingScale scale = JpegStreamingScale::k1_1);

    // streaming version of load jpeg file, RGB8/BGR8/RGBA8/BGRA8 target formats are supported
    Error getJpegSize(const std::string& filename, unsigned int& w, unsigned int& h);
    // wide string version
    Error getJpegSize(const std::wstring& filename, unsigned int& w, unsigned int& h);

    // Metadata manipulation
    void colwertToImageMetadata(const std::unordered_map<uint16_t, std::string> & tags, const std::vector<char> & xmpPacket, ImageMetadata & metadata);
    void colwertFromImageMetadata(const ImageMetadata & metadata, std::unordered_map<uint16_t, std::string> & tags, std::vector<char>& xmpPacket);
}
