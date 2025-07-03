#include <stdint.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <array>
#include <emmintrin.h>
#pragma warning(disable: 4668 4917)
#define WINDOWS_LEAN_AND_MEAN
#include <Wincodec.h>
#include "zlib.h"
#include "ImageLoaderWIC.h"
#include "darkroom/Png.h"
#include "darkroom/StringColwersion.h"

#pragma comment(lib, "windowscodecs.lib")

#pragma warning(disable:4127) // condition is always constant
#pragma warning(disable:4571) // Informational: catch(...) semantics changed since Visual C++ 7.1; structured exceptions (SEH) are no longer caught
#pragma warning(disable:4265) // class has virtual functions, but destructor is not virtual - lambdas cause that

#include <thread>
const bool gUsePngUpFilter = false;

namespace
{
    template<typename T> T clamp(T x, T min, T max) { return x > max ? max : x < min ? min : x; }

    /* Table of CRCs of all 8-bit messages. */
    unsigned long crc_table[256];

    const auto kColorType = 2;
    const auto kCompressionMethod = 0;
    const auto kFilterMethod = 0;
    const auto kInterlaceMethod = 0;

    struct Ihdr
    {
        uint8_t width[4], height[4];
        uint8_t depth, colorType, compressionMethod, filterMethod, interlaceMethod;
    };

    struct PerStreamAdler
    {
        uint32_t adler;
        long len;
    };

    /* Flag: has the table been computed? Initially false. */
    int crc_table_computed = 0;

    /* Make the table for a fast CRC. */
    void make_crc_table(void)
    {
        unsigned long c;
        int n, k;

        for (n = 0; n < 256; n++) {
            c = (unsigned long)n;
            for (k = 0; k < 8; k++) {
                if (c & 1)
                    c = 0xedb88320L ^ (c >> 1);
                else
                    c = c >> 1;
            }
            crc_table[n] = c;
        }
        crc_table_computed = 1;
    }

    /* Update a running CRC with the bytes buf[0..len-1]--the CRC
    should be initialized to all 1's, and the transmitted value
    is the 1's complement of the final running CRC (see the
    crc() routine below)). */

    unsigned long update_crc(unsigned long crc, const unsigned char *buf,
        int len)
    {
        unsigned long c = crc;
        int n;

        if (!crc_table_computed)
            make_crc_table();
        for (n = 0; n < len; n++) {
            c = crc_table[(c ^ buf[n]) & 0xff] ^ (c >> 8);
        }
        return c;
    }

    /* Return the CRC of the bytes buf[0..len-1]. */
    uint32_t gencrc(const unsigned char *buf, int len)
    {
        return update_crc(0xffffffffL, buf, len) ^ 0xffffffffL;
    }

    std::vector<unsigned char> generatePngHeader()
    {
        const std::vector<unsigned char> header = { 137, 80, 78, 71, 13, 10, 26, 10 };
        return header;
    }

    std::vector<unsigned char> generateChunk(const std::string& name,
        const std::vector<unsigned char>::const_iterator& payloadStart,
        const std::vector<unsigned char>::const_iterator& payloadEnd)
    {
        struct ChunkStart
        {
            uint8_t length[4];
            uint8_t name[4];
        };
        const auto payloadSize = payloadEnd - payloadStart;
        std::vector<unsigned char> result(size_t(4 + 4 + 4 + payloadSize));
        std::copy(payloadStart, payloadEnd, result.begin() + 4 + 4);
        ChunkStart cs;
        cs.length[0] = uint8_t((payloadSize >> 24) & 0xFF);
        cs.length[1] = uint8_t((payloadSize >> 16) & 0xFF);
        cs.length[2] = uint8_t((payloadSize >> 8) & 0xFF);
        cs.length[3] = uint8_t(payloadSize & 0xFF);
        for (auto i = 0u; i < 4u; ++i)
            if (i < name.size())
                cs.name[i] = uint8_t(name[i]);
            else
                cs.name[i] = '?';
        std::memcpy(&result[0], &cs, sizeof(cs));
        uint32_t crc = gencrc(result.data() + 4, int32_t(result.size()) - 8);
        result[result.size() - 4] = (crc >> 24) & 0xFF;
        result[result.size() - 3] = (crc >> 16) & 0xFF;
        result[result.size() - 2] = (crc >> 8) & 0xFF;
        result[result.size() - 1] = crc & 0xFF;
        return result;
    }

    std::vector<unsigned char> generateChunk(const std::string& name, const std::vector<unsigned char>& payload)
    {
        return generateChunk(name, payload.cbegin(), payload.cend());
    }

    std::vector<unsigned char> generateTextChunk(const std::string& keyword, const std::string& value)
    {
        std::vector<unsigned char> payload;
        payload.reserve(keyword.size() + value.size() + 1);
        payload.insert(payload.end(), keyword.cbegin(), keyword.cend());
        payload.push_back('\0');
        payload.insert(payload.end(), value.cbegin(), value.cend());
        return generateChunk("tEXt", payload);
    }

    std::vector<unsigned char> generateIhdrChunk(unsigned int width, unsigned int height, uint8_t bitdepth)
    {
        const Ihdr hdr = { { (width >> 24) & 0xFF, (width >> 16) & 0xFF, (width >> 8) & 0xFF, width & 0xFF },
        { (height >> 24) & 0xFF, (height >> 16) & 0xFF, (height >> 8) & 0xFF, height & 0xFF },
            bitdepth, kColorType, kCompressionMethod, kFilterMethod, kInterlaceMethod };

        std::vector<unsigned char> result(sizeof(hdr));
        std::memcpy(&result[0], &hdr, sizeof(hdr));
        return generateChunk("IHDR", result);
    }

    std::vector<unsigned char> generateIendChunk()
    {
        return generateChunk("IEND", std::vector<unsigned char>());
    }

    std::vector<unsigned char> compressPixels(const std::vector<unsigned char>& data, unsigned int compression, std::vector<PerStreamAdler>& adlerCrc, uint32_t threadNo, bool finishStream)
    {
        // pessimistically allocate twice as much for compressed data than uncompressed
        std::vector<unsigned char> compressed(data.size() * 2);
        z_stream c_stream;
        c_stream.zalloc = nullptr;
        c_stream.zfree = nullptr;
        c_stream.opaque = (void *)0;

        deflateInit(&c_stream, int(compression));

        c_stream.next_in = data.data();
        c_stream.next_out = compressed.data();
        c_stream.avail_in = uint32_t(data.size());
        c_stream.avail_out = uint32_t(compressed.size());

        const int flush = finishStream ? Z_FINISH : Z_FULL_FLUSH;

        deflate(&c_stream, flush);

        compressed.resize(c_stream.total_out);
        adlerCrc[threadNo] = { c_stream.adler, long(data.size()) };
        deflateEnd(&c_stream);
        return compressed;
    }

    std::unordered_map<std::string, std::vector<std::vector<unsigned char>>> readChunks(const std::vector<char>& input, size_t offset = 0u)
    {
        std::unordered_map<std::string, std::vector<std::vector<unsigned char>>> chunks;

        for (size_t i = offset; i < input.size(); ++i)
        {
            // in case we don't have enough bytes left to read a chunk length, break and return all the chunks collected so far
            if (i + sizeof(uint32_t) > input.size())
                break;
            const uint32_t chunkDataSize = _byteswap_ulong(*reinterpret_cast<const uint32_t*>(&input[i]));
            // in case we don't have enough bytes to read a chunk type and its payload, break and return all the chunks collected so far
            if (i + 2 * sizeof(uint32_t) + chunkDataSize > input.size())
                break;
            const auto chunkTypeStart = input.cbegin() + ptrdiff_t(i + sizeof(uint32_t));
            const auto chunkTypeEnd = input.cbegin() + ptrdiff_t(i + 2 * sizeof(uint32_t));
            const std::string chunkType(chunkTypeStart, chunkTypeEnd);
            const auto chunkCrcOffset = i + 2 * sizeof(uint32_t) + chunkDataSize;
            const auto payloadStartIterator = input.cbegin() + ptrdiff_t(i + 2 * sizeof(uint32_t));
            const auto payloadEndIterator = input.cbegin() + ptrdiff_t(chunkCrcOffset);
            std::vector<unsigned char> payload(payloadStartIterator, payloadEndIterator);
            // in case chunk crc doesn't checkout - skip it
            const uint32_t chunkCrc = _byteswap_ulong(*reinterpret_cast<const uint32_t*>(&input[chunkCrcOffset]));
            if (gencrc(reinterpret_cast<const unsigned char*>(&*chunkTypeStart), int(payload.size() + sizeof(uint32_t))) == chunkCrc)
                chunks[chunkType].push_back(payload);
            // advance index by chunk length, type, crc and payload size
            i += 3 * sizeof(uint32_t) + chunkDataSize - 1;
        }

        return chunks;
    }

    using darkroom::Error;
    using darkroom::BufferFormat;
    using darkroom::RemapDataCallback;

    Error savePngInternal(const unsigned char* pixels, 
        RemapDataCallback remapCallback, 
        std::ofstream& out, 
        unsigned int width, 
        unsigned int height, 
        BufferFormat format, 
        const std::unordered_map<std::string, std::string>& tags,
        int compression, 
        unsigned int threadCount)
    {
        if (out)
        {
            uint32_t bytesPerPixel = 0;
            const uint8_t bitdepth = 8;

            if (format == BufferFormat::RGB8 || format == BufferFormat::BGR8)
                bytesPerPixel = 3;
            else if (format == BufferFormat::RGBA8 || format == BufferFormat::BGRA8)
                bytesPerPixel = 4;

            if (bytesPerPixel == 0)
                return Error::kIlwalidArgument;

            compression = clamp(compression, Z_NO_COMPRESSION, Z_BEST_COMPRESSION);

            unsigned int workerCount = threadCount == 0 ? std::thread::hardware_conlwrrency() : threadCount;
            // very rare case, but possible according to the documentation (std::thread::hardware_conlwrrency() might return 0)
            if (workerCount == 0)
                workerCount = 1;

            const uint64_t pitch = uint64_t(width) * bytesPerPixel;
            const uint64_t pngPitch = uint64_t(width) * 3;
            // generate PNG chunks that we can generate now - signature, IHDR and IEND chunks
            const auto pngHeader = generatePngHeader();
            const auto pngIhdr = generateIhdrChunk(width, height, bitdepth);
            const auto pngIend = generateIendChunk();

            uint32_t adler = 0u;
            std::vector<PerStreamAdler> adlerCrc;
            adlerCrc.resize(workerCount);

            // used to generate deflate streams (IDAT contents) in parallel
            const auto processImageStrip = [&](const unsigned char* rgb, uint64_t start, uint64_t end, std::vector<unsigned char>* deflateStream, uint32_t threadNo)
            {
                std::vector<unsigned char> scanlines(size_t((end - start) * (pngPitch + 1)));
                for (auto i = start; i < end; ++i)
                {
                    // for now we don't filter the image (as suggested by PNG specification)
                    // each scanline is prepended with byte 0, which means pixels go as is, without filtering
                    scanlines[size_t((i - start) * (pngPitch + 1))] = 0;
                    if (i > start && format == BufferFormat::RGB8)
                    {
                        if (gUsePngUpFilter)
                        {
                            const size_t offset = size_t((i - start) * (pngPitch + 1) + 1);
                            scanlines[offset - 1] = 2u;

                            const unsigned char* inputUpper = rgb + (i - 1) * pitch;
                            const unsigned char* inputLwrrent = rgb + i * pitch;
                            unsigned char* output = &scanlines[offset];
                            for (uint64_t k = 0; k < pitch; k += 16)
                            {
                                const __m128i upperRow = _mm_load_si128((__m128i*)(inputUpper + k));
                                const __m128i lwrrentRow = _mm_load_si128((__m128i*)(inputLwrrent + k));
                                _mm_storeu_si128((__m128i*)(&output[k]), _mm_sub_epi8(lwrrentRow, upperRow));
                            }
                        }
                        std::copy(rgb + i * pitch, rgb + (i + 1) * pitch, scanlines.begin() + off_t((i - start) * (pngPitch + 1) + 1));
                    }
                    else if (i == start && format == BufferFormat::RGB8)
                    {
                        std::copy(rgb + i * pitch, rgb + (i + 1) * pitch, scanlines.begin() + off_t((i - start) * (pngPitch + 1) + 1));
                    }
                    else if (format == BufferFormat::RGBA8)
                    {
                        unsigned char* output = &scanlines[size_t((i - start) * (pngPitch + 1))];
                        const unsigned char* input = &rgb[i * pitch];
                        for (auto j = 0u; j < width; ++j)
                        {
                            output[j * 3 + 1] = input[j * bytesPerPixel + 0];
                            output[j * 3 + 2] = input[j * bytesPerPixel + 1];
                            output[j * 3 + 3] = input[j * bytesPerPixel + 2];
                        }
                    }
                    else if (format == BufferFormat::BGR8 || format == BufferFormat::BGRA8)
                    {
                        unsigned char* output = &scanlines[size_t((i - start) * (pngPitch + 1))];
                        const unsigned char* input = &rgb[i * pitch];
                        for (auto j = 0u; j < width; ++j)
                        {
                            output[j * 3 + 1] = input[j * bytesPerPixel + 2];
                            output[j * 3 + 2] = input[j * bytesPerPixel + 1];
                            output[j * 3 + 3] = input[j * bytesPerPixel + 0];
                        }
                    }
                }
                *deflateStream = compressPixels(scanlines, static_cast<unsigned int>(compression), adlerCrc, threadNo, end == height);
            };

            std::vector<std::thread> workers;
            std::vector<std::vector<unsigned char>> deflateStreams(workerCount);

            // write png signature
            out.write(reinterpret_cast<const char*>(pngHeader.data()), pngHeader.size());
            // write IHDR chunk
            out.write(reinterpret_cast<const char*>(pngIhdr.data()), pngIhdr.size());
            // write tags if there are any into a series of 'tEXt' chunks
            for (const auto& pair : tags)
            {
                const auto& textChunk = generateTextChunk(pair.first, pair.second);
                out.write(reinterpret_cast<const char*>(textChunk.data()), textChunk.size());
            }

            unsigned int rowsProcessed = 0u;

            while (rowsProcessed < height)
            {
                unsigned int rowsMapped = 0u;
                const unsigned char* rgb = remapCallback ? static_cast<const unsigned char*>(remapCallback(rowsMapped)) : pixels;
                if (!remapCallback)
                    rowsMapped = height;
                auto rowsPerThread = rowsMapped / workerCount;
                if (rowsPerThread * workerCount < rowsMapped)
                    rowsPerThread += 1;

                const auto workersUsed = 1u + (rowsMapped - 1) / rowsPerThread;

                // do not spawn a single thread, instead do the work in the current thread
                if (workersUsed > 1)
                {
                    // run deflate streams generation in parallel
                    for (auto i = 0u; i < workersUsed; ++i)
                        workers.push_back(std::thread(processImageStrip, rgb, i * rowsPerThread, min((i + 1) * rowsPerThread, rowsMapped), &deflateStreams[i], i));
                    for (auto& w : workers)
                        w.join();
                }
                else
                    processImageStrip(rgb, 0, rowsMapped, &deflateStreams[0], 0);

                workers.clear();

                if (workersUsed > 1)
                {
                    auto& lastDeflateStream = deflateStreams[workersUsed - 1];
                    // first stream ever initialized adler checksum of the whole deflate stream
                    if (rowsProcessed == 0)
                        adler = adlerCrc[0].adler;
                    // sort streams by their sequency numbers
                    for (auto i = rowsProcessed == 0 ? 1u : 0u; i < workersUsed; ++i)
                        adler = adler32_combine(adler, adlerCrc[i].adler, adlerCrc[i].len);

                    // last deflate stream in the whole image should have Adler checksum fixed:
                    // this mean we need to substitute per-stream checksum with the whole stream checksum
                    if (rowsProcessed + rowsMapped >= height)
                    {
                        adler = _byteswap_ulong(adler);
                        std::memcpy(&lastDeflateStream[lastDeflateStream.size() - sizeof(adler)], &adler, sizeof(adler));
                    }
                }

                // generate & write all IDAT chunks
                for (auto i = 0u; i < workersUsed; ++i)
                {
                    std::vector<unsigned char> pngIdat;
                    // strip zlib header for all but the first streams
                    if (rowsProcessed == 0 && i == 0)
                        pngIdat = std::move(generateChunk("IDAT", deflateStreams[i]));
                    else
                        pngIdat = std::move(generateChunk("IDAT", deflateStreams[i].cbegin() + 2, deflateStreams[i].cend()));
                    out.write(reinterpret_cast<const char*>(pngIdat.data()), pngIdat.size());
                }

                rowsProcessed += rowsMapped;
            }

            // write IEND chunk
            out.write(reinterpret_cast<const char*>(pngIend.data()), pngIend.size());
        }
        else
            return Error::kCouldntCreateFile;
        return Error::kSuccess;
    }

    bool loadPngTagsInternal(std::ifstream& in, unsigned int& w, unsigned int& h, std::unordered_map<std::string, std::string>& tags, BufferFormat format)
    {
        if (in.fail())
            return false;

        const std::array<BufferFormat, 4> supportedFormats = { BufferFormat::BGR8, BufferFormat::RGB8, BufferFormat::BGRA8, BufferFormat::RGBA8 };

        // in case the format is not supported return empty result
        if (std::find(supportedFormats.cbegin(), supportedFormats.cend(), format) == supportedFormats.cend())
            return false;

        // read entire file into a buffer
        in.seekg(0, std::ios_base::end);
        std::vector<char> contents(static_cast<size_t>(in.tellg()));
        in.seekg(0, std::ios_base::beg);
        in.read(&contents[0], contents.size());
        // check PNG signature
        const auto pngSignatureUchar = generatePngHeader();
        const std::vector<char> pngSignature(pngSignatureUchar.cbegin(), pngSignatureUchar.cend());
        if (contents.size() < pngSignature.size() || pngSignature != decltype(contents)(contents.cbegin(), contents.cbegin() + ptrdiff_t(pngSignature.size())))
            return false;
        // break it into chunks
        const auto chunks = readChunks(contents, pngSignature.size());
        // free memory for the file contents, now that we have all the chunks
        contents = decltype(contents)();
        // examine all the chunks we have and check for basic PNG structure sanity:
        // each PNG should have IHDR and IEND chunks as well as at least one IDAT chunk
        // this forms a minimal valid PNG file
        const auto pngIhdr = chunks.find("IHDR");
        const auto pngIdat = chunks.find("IDAT");
        const auto pngIend = chunks.find("IEND");
        if (pngIhdr == chunks.cend() ||         // there is at least one IHDR chunk
            pngIhdr->second.size() != 1 ||      // it's a single IHDR chunk
            pngIdat == chunks.cend() ||         // there is at least one IDAT chunk
            pngIdat->second.size() == 0 ||      // at least one
            pngIend == chunks.cend() ||         // there is at least one IEND chunk
            pngIend->second.size() != 1)        // it's a single IEND chunk
            return false;

        // examine IHDR chunk first
        Ihdr ihdr;
        memcpy(&ihdr, &pngIhdr->second[0][0], sizeof(ihdr));

        w = _byteswap_ulong(*reinterpret_cast<const uint32_t*>(&ihdr));
        h = _byteswap_ulong(*(reinterpret_cast<const uint32_t*>(&ihdr) + 1));

        // examine all tEXt chunks and fill tags
        const auto textChunks = chunks.find("tEXt");
        if (textChunks != chunks.cend())
        {
            for (const auto& chunk : textChunks->second)
            {
                const auto separator = std::find(chunk.cbegin(), chunk.cend(), '\0');
                const std::string key(chunk.cbegin(), separator);
                const std::string value(separator + 1u, chunk.cend());
                if (!key.empty() && !value.empty())
                    tags[key] = value;
            }
        }
        return true;
    }
}

namespace darkroom
{
    Error savePng(const unsigned char* rgb, 
        RemapDataCallback remapCallback, 
        const std::string& filename, 
        unsigned int width, 
        unsigned int height, 
        BufferFormat format, 
        const std::unordered_map<std::string, std::string>& tags, 
        int compression, 
        unsigned int threadCount)
    {
        std::ofstream out(filename.c_str(), std::ofstream::binary);
        return savePngInternal(rgb, remapCallback, out, width, height, format, tags, compression, threadCount);
    }

    Error savePng(const unsigned char* rgb,
        RemapDataCallback remapCallback,
        const std::wstring& filename,
        unsigned int width,
        unsigned int height,
        BufferFormat format,
        const std::unordered_map<std::string, std::string>& tags,
        int compression,
        unsigned int threadCount)
    {
        std::ofstream out(filename.c_str(), std::ofstream::binary);
        return savePngInternal(rgb, remapCallback, out, width, height, format, tags, compression, threadCount);
    }

    std::vector<unsigned char> loadPng(const std::string& filename, unsigned int& w, unsigned int& h, BufferFormat format, std::unordered_map<std::string, std::string>& tags)
    {
        bool canLoadTags = false;
        {
            std::ifstream in(filename.c_str(), std::ofstream::binary);
            canLoadTags = loadPngTagsInternal(in, w, h, tags, format);
        }

        if (canLoadTags)
            return loadImageWinApiInternal(filename, w, h, format);
        return std::vector<unsigned char>();
    }

    std::vector<unsigned char> loadPng(const std::wstring& filename, unsigned int& w, unsigned int& h, BufferFormat format, std::unordered_map<std::string, std::string>& tags)
    {
        bool canLoadTags = false;
        {
            std::ifstream in(filename.c_str(), std::ofstream::binary);
            canLoadTags = loadPngTagsInternal(in, w, h, tags, format);
        }

        if (canLoadTags)
            return loadImageWinApiInternal(filename, w, h, format);
        return std::vector<unsigned char>();
    }

    Error loadPngStreaming(const std::string& filename, ProcessDataCallback<unsigned char> callback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format)
    {
        return loadImageStreamingWinApiInternal(filename, callback, maxBytes, w, h, format);
    }

    Error loadPngStreaming(const std::wstring& filename, ProcessDataCallback<unsigned char> callback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format)
    {
        return loadImageStreamingWinApiInternal(filename, callback, maxBytes, w, h, format);
    }

    void colwertToImageMetadata(const std::unordered_map<std::string, std::string> & tags, ImageMetadata & metadata)
    {
        metadata = ImageMetadata();

        auto iterator = tags.find("Source");
        if (iterator != tags.end())
        {
            metadata.tagMake = iterator->second;
        }

        iterator = tags.find("Description");
        if (iterator != tags.end())
        {
            metadata.tagDescription = iterator->second;
        }

        iterator = tags.find("Model");
        if (iterator != tags.end())
        {
            metadata.tagModel = iterator->second;
        }

        iterator = tags.find("Software");
        if (iterator != tags.end())
        {
            metadata.tagSoftware = iterator->second;
        }

        iterator = tags.find("Comment");
        if (iterator != tags.end())
        {
            metadata.tagType = iterator->second;
        }

        iterator = tags.find("DRSName");
        if (iterator != tags.end())
        {
            metadata.tagDrsName = iterator->second;
        }

        iterator = tags.find("DRSProfileName");
        if (iterator != tags.end())
        {
            metadata.tagDrsProfileName = iterator->second;
        }

        iterator = tags.find("AppTitleName");
        if (iterator != tags.end())
        {
            metadata.tagAppTitleName = iterator->second;
        }

        iterator = tags.find("AppCMSID");
        if (iterator != tags.end())
        {
            metadata.tagAppCMSID = iterator->second;
        }

        iterator = tags.find("AppShortName");
        if (iterator != tags.end())
        {
            metadata.tagAppShortName = iterator->second;
        }

        iterator = tags.find("ActiveFilters");
        if (iterator != tags.end())
        {
            metadata.tagActiveFilters = iterator->second;
        }

        if (metadata.tagType.empty())
        {
            iterator = tags.find("MakerNote");
            if (iterator != tags.end())
            {
                metadata.tagType = iterator->second;
            }
        }
    }

    void colwertFromImageMetadata(const ImageMetadata & metadata, std::unordered_map<std::string, std::string> & tags)
    {
        tags["Description"] = metadata.tagDescription;
        tags["Source"] = metadata.tagMake;
        tags["Model"] = metadata.tagModel;
        tags["Software"] = metadata.tagSoftware;
        tags["Comment"] = metadata.tagType;
        tags["MakerNote"] = metadata.tagType;
        if (!metadata.tagDrsName.empty())
            tags["DRSName"] = metadata.tagDrsName;
        if (!metadata.tagDrsProfileName.empty())
            tags["DRSProfileName"] = metadata.tagDrsProfileName;
        if (!metadata.tagAppTitleName.empty())
            tags["AppTitleName"] = metadata.tagAppTitleName;
        if (!metadata.tagAppCMSID.empty())
            tags["AppCMSID"] = metadata.tagAppCMSID;
        if (!metadata.tagAppShortName.empty())
            tags["AppShortName"] = metadata.tagAppShortName;
        if (!metadata.tagActiveFilters.empty())
            tags["ActiveFilters"] = metadata.tagActiveFilters;
    }
}
