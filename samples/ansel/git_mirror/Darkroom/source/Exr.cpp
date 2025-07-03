#include "darkroom/Exr.h"
#include "darkroom/StringColwersion.h"

#include <array>
#include <algorithm>

#pragma warning(disable: 4996 4244 4365)
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfRgbaFile.h>
#include <OpenEXR/ImfNamespace.h>
#include <OpenEXR/ImfStringAttribute.h>

namespace IMF = OPENEXR_IMF_NAMESPACE;

namespace
{
    class WideOrNarrowStringOstream : public IMF::OStream
    {
    private:
        std::wstring mFileNameWide;
        std::string mFileNameNarrow;
        std::ofstream mOut;
    public:
        WideOrNarrowStringOstream(const std::wstring& path) : mFileNameWide(path), mOut(mFileNameWide, std::ofstream::binary), IMF::OStream(darkroom::getUtf8FromWstr(path).c_str()) {}
        WideOrNarrowStringOstream(const std::string& path) : mFileNameNarrow(path), mOut(mFileNameNarrow, std::ofstream::binary), IMF::OStream(mFileNameNarrow.c_str()) {}
        void write(const char c[/*n*/], int n) { mOut.write(c, n); }
        IMF::Int64 tellp() { return mOut.tellp(); }
        void seekp(IMF::Int64 pos) { mOut.seekp(pos); }
    };

    class WideOrNarrowStringIstream : public IMF::IStream
    {
    private:
        std::wstring mFileNameWide;
        std::string mFileNameNarrow;
        std::ifstream mIn;
    public:
        WideOrNarrowStringIstream(const std::wstring& path) : mFileNameWide(path), mIn(mFileNameWide, std::ifstream::binary), IMF::IStream(darkroom::getUtf8FromWstr(path).c_str()) {}
        WideOrNarrowStringIstream(const std::string& path) : mFileNameNarrow(path), mIn(mFileNameNarrow, std::ifstream::binary), IMF::IStream(mFileNameNarrow.c_str()) {}
        bool read(char c[/*n*/], int n) { mIn.read(c, n); return bool(mIn); }
        IMF::Int64 tellg() { return mIn.tellg(); }
        void seekg(IMF::Int64 pos) { mIn.seekg(pos); }
    };

    using darkroom::RemapDataCallback;
    using darkroom::BufferFormat;
    using darkroom::Error;

    template<typename T>
    Error saveExrInternal(const void* pixels,
        RemapDataCallback remapCallback,
        const T& filename,
        unsigned int width,
        unsigned int height,
        BufferFormat format,
        const std::unordered_map<std::string, std::string>& tags)
    {
        using namespace IMF;

        // check if the input format is supported
        const std::array<BufferFormat, 4> supportedFormats16bit = { BufferFormat::RGB16, BufferFormat::BGR16, BufferFormat::RGBA16, BufferFormat::BGRA16 };
        const std::array<BufferFormat, 5> supportedFormats32bit = { BufferFormat::RGB32, BufferFormat::BGR32, BufferFormat::RGBA32, BufferFormat::BGRA32, BufferFormat::R32 };

        const bool isFormat16Bit = std::find(supportedFormats16bit.cbegin(), supportedFormats16bit.cend(), format) != supportedFormats16bit.cend();
        const bool isFormat32Bit = std::find(supportedFormats32bit.cbegin(), supportedFormats32bit.cend(), format) != supportedFormats32bit.cend();

        if (!isFormat16Bit && !isFormat32Bit)
            return Error::kIlwalidArgument;

        const PixelType pixelType = isFormat16Bit ? HALF : FLOAT;
        const unsigned int channelWidthInBytes = isFormat16Bit ? 2u : 4u;
        unsigned int inputChannelCount = 4;

        if (format == BufferFormat::RGB16 || format == BufferFormat::BGR16 || format == BufferFormat::RGB32 || format == BufferFormat::BGR32)
            inputChannelCount = 3;
        else if (format == BufferFormat::R32)
            inputChannelCount = 1;

        // create EXR header defining three channels (no alpha), no compression
        Header header(width, height, 1, IMATH_NAMESPACE::V2f(0, 0), 1, INCREASING_Y, NO_COMPRESSION);
        // write tags
        for (const auto& pair : tags)
            if (!pair.first.empty())
                header.insert(pair.first, StringAttribute(pair.second));
        // create just R channel by default
        header.channels().insert("R", Channel(pixelType));
        // in case there is more than one channel, create all three RGB channels
        if (inputChannelCount != 1)
        {
            header.channels().insert("G", Channel(pixelType));
            header.channels().insert("B", Channel(pixelType));
        }
        // create output file
        WideOrNarrowStringOstream out(filename);
        OutputFile file(out, header);

        // write data
        if (remapCallback == nullptr)
        {
            // setup framebuffer
            FrameBuffer frameBuffer;

            char* redChannelBasePointer = static_cast<char*>(const_cast<void*>(pixels));
            if (format == BufferFormat::BGR16 || format == BufferFormat::BGR32 || format == BufferFormat::BGRA16 || format == BufferFormat::BGRA32)
                redChannelBasePointer += channelWidthInBytes * 2;

            frameBuffer.insert("R", Slice(pixelType, redChannelBasePointer, channelWidthInBytes * inputChannelCount, channelWidthInBytes * inputChannelCount * width));

            if (inputChannelCount != 1)
            {
                char* greenChannelBasePointer = static_cast<char*>(const_cast<void*>(pixels)) + channelWidthInBytes; // G is always in the middle
                char* blueChannelBasePointer = static_cast<char*>(const_cast<void*>(pixels));
                if (format == BufferFormat::RGB16 || format == BufferFormat::RGB32 || format == BufferFormat::RGBA16 || format == BufferFormat::RGBA32)
                    blueChannelBasePointer += channelWidthInBytes * 2;

                frameBuffer.insert("G", Slice(pixelType, greenChannelBasePointer, channelWidthInBytes * inputChannelCount, channelWidthInBytes * inputChannelCount * width));
                frameBuffer.insert("B", Slice(pixelType, blueChannelBasePointer, channelWidthInBytes * inputChannelCount, channelWidthInBytes * inputChannelCount * width));
            }


            file.setFrameBuffer(frameBuffer);
            file.writePixels(height);
        }
        else
        {
            unsigned int rowsMapped = 0;
            size_t rowsWritten = 0;
#pragma warning(disable : 4127)
            while (true)
            {
                const void* rgb = static_cast<const float*>(remapCallback(rowsMapped));
                if (rowsMapped == 0 || rowsWritten == height)
                    break;
                // setup framebuffer
                FrameBuffer frameBuffer;

                char* redChannelBasePointer = static_cast<char*>(const_cast<void*>(rgb));
                if (format == BufferFormat::BGR16 || format == BufferFormat::BGR32 || format == BufferFormat::BGRA16 || format == BufferFormat::BGRA32)
                    redChannelBasePointer += channelWidthInBytes * 2;

                char* greenChannelBasePointer = static_cast<char*>(const_cast<void*>(rgb)) + channelWidthInBytes; // G is always in the middle

                char* blueChannelBasePointer = static_cast<char*>(const_cast<void*>(rgb));
                if (format == BufferFormat::RGB16 || format == BufferFormat::RGB32 || format == BufferFormat::RGBA16 || format == BufferFormat::RGBA32)
                    blueChannelBasePointer += channelWidthInBytes * 2;

                const auto pitch = width * channelWidthInBytes * inputChannelCount;
                frameBuffer.insert("R", Slice(pixelType, redChannelBasePointer - rowsWritten * pitch, channelWidthInBytes * inputChannelCount, channelWidthInBytes * inputChannelCount * width));
                frameBuffer.insert("G", Slice(pixelType, greenChannelBasePointer - rowsWritten * pitch, channelWidthInBytes * inputChannelCount, channelWidthInBytes * inputChannelCount * width));
                frameBuffer.insert("B", Slice(pixelType, blueChannelBasePointer - rowsWritten * pitch, channelWidthInBytes * inputChannelCount, channelWidthInBytes * inputChannelCount * width));

                file.setFrameBuffer(frameBuffer);
                file.writePixels(rowsMapped);
                rowsWritten += rowsMapped;
            }
#pragma warning(default : 4127)
        }

        return Error::kSuccess;
    }

    template<typename T>
    std::vector<float> loadExrInternal(const T& filename, unsigned int& w, unsigned int& h, std::unordered_map<std::string, std::string>& tags, darkroom::BufferFormat fmt)
    {
        WideOrNarrowStringIstream inputStream(filename);
        using namespace IMF;
        InputFile in(inputStream);
        const auto box = in.header().dataWindow();
        // load tags
        for (auto it = in.header().begin(); it != in.header().end(); ++it)
        {
            if (it.name())
            {
                const auto nameString = std::string(it.name());
                if (!nameString.empty())
                {
                    const StringAttribute* attr = in.header().findTypedAttribute <StringAttribute>(it.name());
                    if (attr)
                        tags[nameString] = attr->value();
                }
            }
        }
        // Count channels.
        const Imf::ChannelList& channels = in.header().channels();
        uint32_t inputChannelCount = 0;
        for (Imf::ChannelList::ConstIterator it = channels.begin(); it != channels.end(); ++it)
            inputChannelCount++;

        w = box.max.x - box.min.x + 1;
        h = box.max.y - box.min.y + 1;
        std::vector<float> result(w * h * inputChannelCount);
        FrameBuffer frameBuffer;
        frameBuffer.insert("R", Slice(FLOAT, reinterpret_cast<char*>(&result[0]), sizeof(float) * inputChannelCount, sizeof(float) * inputChannelCount * w));
        if (inputChannelCount != 1)
        {
            frameBuffer.insert("G", Slice(FLOAT, reinterpret_cast<char*>(&result[0] + 1), sizeof(float) * inputChannelCount, sizeof(float) * inputChannelCount * w));
            frameBuffer.insert("B", Slice(FLOAT, reinterpret_cast<char*>(&result[0] + 2), sizeof(float) * inputChannelCount, sizeof(float) * inputChannelCount * w));
            if (fmt == darkroom::BufferFormat::RGBA32)
                frameBuffer.insert("A", Slice(FLOAT, reinterpret_cast<char*>(&result[0] + 3), sizeof(float) * inputChannelCount, sizeof(float) * inputChannelCount * w));
        }

        in.setFrameBuffer(frameBuffer);
        in.readPixels(box.min.y, box.max.y);
        return result;
    }

    template<typename T>
    Error loadExrStreamingInternal(const T& filename, darkroom::ProcessDataCallback<float> callback, size_t maxBytes,
        unsigned int& w, unsigned int& h, std::unordered_map<std::string, std::string>& tags, darkroom::BufferFormat fmt)
    {
        WideOrNarrowStringIstream inputStream(filename);
        using namespace IMF;
        InputFile in(inputStream);
        const auto box = in.header().dataWindow();
        // load tags
        for (auto it = in.header().begin(); it != in.header().end(); ++it)
        {
            if (it.name())
            {
                const auto nameString = std::string(it.name());
                if (!nameString.empty())
                {
                    const StringAttribute* attr = in.header().findTypedAttribute <StringAttribute>(it.name());
                    if (attr)
                        tags[nameString] = attr->value();
                }
            }
        }

        // Count channels.
        const Imf::ChannelList& channels = in.header().channels();
        uint32_t inputChannelCount = 0;
        for (Imf::ChannelList::ConstIterator it = channels.begin(); it != channels.end(); ++it)
            inputChannelCount++;

        w = box.max.x - box.min.x + 1;
        h = box.max.y - box.min.y + 1;

        std::vector<float> data(maxBytes / sizeof(float));
        std::vector<std::string> channelNames = { "R" };
        FrameBuffer frameBuffer;
        frameBuffer.insert("R", Slice(FLOAT, reinterpret_cast<char*>(&data[0]), sizeof(float) * inputChannelCount, sizeof(float) * inputChannelCount * w));
        if (inputChannelCount != 1)
        {
            frameBuffer.insert("G", Slice(FLOAT, reinterpret_cast<char*>(&data[0] + 1), sizeof(float) * inputChannelCount, sizeof(float) * inputChannelCount * w));
            frameBuffer.insert("B", Slice(FLOAT, reinterpret_cast<char*>(&data[0] + 2), sizeof(float) * inputChannelCount, sizeof(float) * inputChannelCount * w));
            channelNames.push_back("G");
            channelNames.push_back("B");
            if (fmt == darkroom::BufferFormat::RGBA32)
            {
                frameBuffer.insert("A", Slice(FLOAT, reinterpret_cast<char*>(&data[0] + 3), sizeof(float) * inputChannelCount, sizeof(float) * inputChannelCount * w));
                channelNames.push_back("A");
            }
        }

        in.setFrameBuffer(frameBuffer);

        const uint32_t maxLines = uint32_t((maxBytes / (inputChannelCount * sizeof(float))) / w);

        if (maxLines == 0u)
            return Error::kOperationFailed;

        uint32_t lwrrentLine = 0u;
        while (lwrrentLine < h)
        {
            const auto linesToRead = std::min(maxLines, h - lwrrentLine);
            const auto endLine = lwrrentLine + linesToRead - 1;
            
            in.readPixels(lwrrentLine, endLine);
            const auto err = callback(data.data(), uint32_t(linesToRead));
            if (err != Error::kSuccess)
                break;
            lwrrentLine += linesToRead;

            for (const auto& channelName : channelNames)
                frameBuffer[channelName].base -= linesToRead * w * inputChannelCount * sizeof(float);
            in.setFrameBuffer(frameBuffer);
        }

        return darkroom::Error::kSuccess;
    }

    template<typename T>
    Error getExrImageSizeInternal(const T& filename, unsigned int & w, unsigned int & h)
    {
        WideOrNarrowStringIstream inputStream(filename);
        using namespace IMF;
        InputFile in(inputStream);
        const auto box = in.header().dataWindow();
        w = box.max.x - box.min.x + 1;
        h = box.max.y - box.min.y + 1;
        return Error::kSuccess;
    }
}

namespace darkroom
{
    Error saveExr(const void* pixels,
        RemapDataCallback remapCallback,
        const std::string& filename,
        unsigned int width,
        unsigned int height,
        BufferFormat format,
        const std::unordered_map<std::string, std::string>& tags)
    {
        return saveExrInternal(pixels, remapCallback, filename, width, height, format, tags);
    }

    Error saveExr(const void* pixels,
        RemapDataCallback remapCallback,
        const std::wstring& filename,
        unsigned int width,
        unsigned int height,
        BufferFormat format,
        const std::unordered_map<std::string, std::string>& tags)
    {
        return saveExrInternal(pixels, remapCallback, filename, width, height, format, tags);
    }

    std::vector<float> loadExr(const std::string& filename, unsigned int& w, unsigned int& h, std::unordered_map<std::string, std::string>& tags, darkroom::BufferFormat fmt)
    {
        return loadExrInternal(filename, w, h, tags, fmt);
    }

    std::vector<float> loadExr(const std::wstring& filename, unsigned int& w, unsigned int& h, std::unordered_map<std::string, std::string>& tags, darkroom::BufferFormat fmt)
    {
        return loadExrInternal(filename, w, h, tags, fmt);
    }

    Error loadExrStreaming(const std::string& filename, ProcessDataCallback<float> callback, size_t maxBytes, 
        unsigned int& w, unsigned int& h, std::unordered_map<std::string, std::string>& tags, darkroom::BufferFormat fmt)
    {
        return loadExrStreamingInternal(filename, callback, maxBytes, w, h, tags, fmt);
    }

    Error loadExrStreaming(const std::wstring& filename, ProcessDataCallback<float> callback, size_t maxBytes, 
        unsigned int& w, unsigned int& h, std::unordered_map<std::string, std::string>& tags, darkroom::BufferFormat fmt)
    {
        return loadExrStreamingInternal(filename, callback, maxBytes, w, h, tags, fmt);
    }

    Error getExrImageSize(const std::string& filename, unsigned int & w, unsigned int & h)
    {
        return getExrImageSizeInternal(filename, w, h);
    }

    Error getExrImageSize(const std::wstring& filename, unsigned int & w, unsigned int & h)
    {
        return getExrImageSizeInternal(filename, w, h);
    }
}
