#include <iostream>
#include <algorithm>
#include <fstream>
#include "darkroom/Jpeg.h"

#include "jpeglib.h"
#include "turbojpeg.h"

namespace
{
    class ApplicationMarker
    {
    public:
        ApplicationMarker()
        {
            app1 = { 0xFF, 0xE1, // app1 marker
                0x00, 0x00, // size (filled later)
                'E', 'x', 'i', 'f', 0x00, 0x00, // exif marker
                'M', 'M', 0, 42, // 'MM' stands for Motorola byte order. 42 - TIFF "version number". 
                                 // This version number is never changed and the value (42) was choosen for its deep philosophical value.
                0x00, 0x00, 0x00, 0x08, // offset to first IFD
                0x00, 0x00 // number of tags (filled later)
            };
            // now we're ready to write tags
        }

        void writeTag(const uint16_t marker, const std::string& value) { tags.push_back({ marker, value }); }

        void finalize()
        {
            if (tags.empty())
                return;
            // write each tag
            auto tagsTemp = tags;
            const uint16_t tagCount = uint16_t(tags.size());
            uint16_t aclwmulatedOffset = 0;
            while (!tagsTemp.empty())
            {
                const auto marker = std::get<0>(tagsTemp.back());
                const auto value = std::get<1>(tagsTemp.back());
                tagsTemp.pop_back();

                app1.push_back(uint8_t(marker >> 8)); app1.push_back(uint8_t(marker & 0xFF)); // marker ID
                app1.push_back(uint8_t(0x00)); app1.push_back(uint8_t(0x02)); // marker type - 2 - 'ascii string'
                
                // write tag value length
                const auto len = value.size() + 1; // including term
                app1.push_back((len >> 24) & 0xFF);
                app1.push_back((len >> 16) & 0xFF);
                app1.push_back((len >> 8) & 0xFF);
                app1.push_back(len & 0xFF);
                // write tag value offset
                const auto offset = tags.size() * 12 + 10 + 4 + aclwmulatedOffset;
                aclwmulatedOffset += uint16_t(len);
                app1.push_back((offset >> 24) & 0xFF);
                app1.push_back((offset >> 16) & 0xFF);
                app1.push_back((offset >> 8) & 0xFF);
                app1.push_back(offset & 0xFF);
            }

            // write offset to next IFD - zero means that's all
            app1.insert(app1.end(), { 0x00, 0x00, 0x00, 0x00 });
            // IFD0 data
            while (!tags.empty())
            {
                const auto value = std::get<1>(tags.back());
                tags.pop_back();
                app1.insert(app1.end(), value.cbegin(), value.cend());
                app1.push_back(0x00);
            }
            // fill in sizes now
            const uint16_t app1size = uint16_t(app1.size() - 2u);
            app1[2] = uint8_t((app1size >> 8) & 0xFF);
            app1[3] = uint8_t(app1size & 0xFF);
            app1[18] = uint8_t((tagCount >> 8) & 0xFF);
            app1[19] = uint8_t(tagCount & 0xFF);
        }

        const std::vector<unsigned char>& data() const { return app1; }
    private:
        std::vector<unsigned char> app1;
        std::vector<std::pair<uint16_t, std::string>> tags;
    };

    template<typename T>
    std::vector<unsigned char> loadFile(const T& filename)
    {
        std::ifstream in(filename, std::ifstream::binary);
        if (in)
        {
            in.seekg(0, in.end);
            std::vector<char> bmp(static_cast<size_t>(in.tellg()));
            in.seekg(0, in.beg);
            in.read(&bmp[0], bmp.size());
            return std::vector<unsigned char>(bmp.cbegin(), bmp.cend());
        }
        return std::vector<unsigned char>();
    }

    FILE* openFile(const std::wstring& filename)
    {
        FILE* in = nullptr;
        _wfopen_s(&in, filename.c_str(), L"rb");
        return in;
    }

    FILE* openFile(const std::string& filename)
    {
        FILE* in = nullptr;
        fopen_s(&in, filename.c_str(), "rb");
        return in;
    }

    using darkroom::Error;
    using darkroom::BufferFormat;
    using darkroom::RemapDataCallback;

    Error saveJpegInternal(const unsigned char* rgb, 
        RemapDataCallback remapCallback,
        FILE* out,
        unsigned int width, 
        unsigned int height, 
        BufferFormat format, 
        const std::unordered_map<uint16_t, std::string>& tags, 
        const std::vector<char>& xmpPacket,
        unsigned int jpegQuality)
    {
        jpeg_compress_struct cinfo;
        jpeg_error_mgr jerr;
        cinfo.err = jpeg_std_error(&jerr);
        jpeg_create_compress(&cinfo);
        cinfo.image_width = width;
        cinfo.image_height = height;
        if (format == BufferFormat::RGB8)
        {
            cinfo.in_color_space = JCS_EXT_RGB;
            cinfo.input_components = 3;
        }
        else if (format == BufferFormat::BGR8)
        {
            cinfo.in_color_space = JCS_EXT_BGR;
            cinfo.input_components = 3;
        }
        else if (format == BufferFormat::RGBA8)
        {
            cinfo.in_color_space = JCS_EXT_RGBX;
            cinfo.input_components = 4;
        }
        else if (format == BufferFormat::BGRA8)
        {
            cinfo.in_color_space = JCS_EXT_BGRX;
            cinfo.input_components = 4;
        }
        jpeg_set_defaults(&cinfo);
        cinfo.dct_method = JDCT_FLOAT;
        cinfo.num_components = 3;

        jpeg_stdio_dest(&cinfo, out);
        jpeg_set_quality(&cinfo, int(jpegQuality), TRUE);
        jpeg_start_compress(&cinfo, TRUE);

        // write tags if there are any into a FFE1
        if (!tags.empty())
        {
            ApplicationMarker app1;
            for (const auto& pair : tags)
                app1.writeTag(pair.first, pair.second);
            app1.finalize();
            jpeg_write_marker(&cinfo, (JPEG_APP0 + 1), app1.data().data() + 4, static_cast<unsigned int>(app1.data().size()) - 4);
        }
        if (!xmpPacket.empty())
            jpeg_write_marker(&cinfo, (JPEG_APP0 + 1), reinterpret_cast<const JOCTET*>(xmpPacket.data()), static_cast<unsigned int>(xmpPacket.size()));

        if (remapCallback == nullptr)
        {
            while (cinfo.next_scanline < cinfo.image_height)
            {
                JSAMPROW samples[16];
                for (int i = 0; i < 16; ++i)
                    samples[i] = reinterpret_cast<JSAMPLE*>(const_cast<unsigned char*>(&rgb[(cinfo.next_scanline + i) * cinfo.image_width * cinfo.input_components]));
                jpeg_write_scanlines(&cinfo, samples, 16);
            }
        }
        else
        {
            while (cinfo.next_scanline < cinfo.image_height)
            {
                unsigned int rowsMapped = 0;
                
                const unsigned char* rgb = static_cast<const unsigned char*>(remapCallback(rowsMapped));
                JSAMPROW* samples = new JSAMPROW[rowsMapped];
                for (auto i = 0u; i < rowsMapped; ++i)
                    samples[i] = reinterpret_cast<JSAMPLE*>(const_cast<unsigned char*>(&rgb[i * cinfo.image_width * cinfo.input_components]));
                jpeg_write_scanlines(&cinfo, samples, rowsMapped);
                delete[] samples;
            }
        }

        jpeg_finish_compress(&cinfo);
        jpeg_destroy_compress(&cinfo);
        return Error::kSuccess;
    }

    template<typename T>
    Error loadJpegTagsInternal(const T& filename, std::unordered_map<uint16_t, std::string>& tags, std::vector<char>& xmpPacket)
    {
        Error ret = Error::kSuccess;
        if (auto file = openFile(filename))
        {
            jpeg_decompress_struct cinfo;
            struct jpeg_error_mgr jerr;
            cinfo.err = jpeg_std_error(&jerr);
            jpeg_create_decompress(&cinfo);
            jpeg_stdio_src(&cinfo, file);
            jpeg_save_markers(&cinfo, JPEG_APP0 + 1, 0xffff);
            jpeg_read_header(&cinfo, TRUE);
            jpeg_start_decompress(&cinfo);
            jpeg_saved_marker_ptr mptr = cinfo.marker_list;
            while (mptr != NULL)
            {
                if (mptr->marker == 0xe1)
                {
                    std::vector<unsigned char> markerData(mptr->data_length);
                    std::copy(mptr->data, mptr->data + mptr->data_length, markerData.begin());
                    if (markerData.size() > 4)
                    {
                        const auto first4bytes = decltype(markerData)(markerData.cbegin(), markerData.cbegin() + 4);
                        if (first4bytes == decltype(markerData){'E', 'x', 'i', 'f'} && markerData.size() >= 16)
                        {
                            // parse exif data into 'tags'
                            const bool motorolaOrder = markerData[5] == 'M' && markerData[6] == 'M';
                            const auto ifd0offset = _byteswap_ulong(*reinterpret_cast<const uint32_t*>(&markerData[10]));
                            const auto tagCount = _byteswap_ushort(*reinterpret_cast<const uint16_t*>(&markerData[14]));
                            // break in case we have corrupted exif block
                            const auto ifdEntrySize = 12u;
                            if (ifdEntrySize * tagCount + 16 >= markerData.size())
                                break;
                            for (auto i = 0u; i < tagCount; ++i)
                            {
                                const auto tagOffset = 16 + i * 12;
                                // check against invalid tagCount (and going out of markerData blob)
                                if (tagOffset >= markerData.size())
                                {
                                    fclose(file);
                                    return Error::kIlwalidData;
                                }
                                const auto tagNo = _byteswap_ushort(*reinterpret_cast<const uint16_t*>(&markerData[tagOffset]));
                                const auto tagType = _byteswap_ushort(*reinterpret_cast<const uint16_t*>(&markerData[tagOffset + 2]));
                                if (tagType != 2) // 2 is ascii string type of the ifd entry. We only support ascii string type here
                                    continue;
                                const auto tagValueLength = _byteswap_ulong(*reinterpret_cast<const uint32_t*>(&markerData[tagOffset + 4]));
                                const auto tagValueOffset = _byteswap_ulong(*reinterpret_cast<const uint32_t*>(&markerData[tagOffset + 8]));

                                const auto startOffset = ptrdiff_t(6u + tagValueOffset);
                                const auto startToEndOffset = ptrdiff_t(tagValueLength - 1);
                                if (size_t(startOffset) > markerData.size() || size_t(startToEndOffset) > markerData.size())
                                    continue;

                                const auto tagValueStartIt = markerData.cbegin() + startOffset;
                                const auto tagValueEndIt = tagValueStartIt + startToEndOffset;
                                
                                // check both iterators are within the bounds of the markerData blob
                                if (std::distance(markerData.cbegin(), tagValueStartIt) < ptrdiff_t(markerData.size()) &&
                                    std::distance(markerData.cbegin(), tagValueEndIt) < ptrdiff_t(markerData.size()))
                                {
                                    const auto tagValue = std::string(tagValueStartIt, tagValueEndIt);
                                    tags[tagNo] = tagValue;
                                }
                                else
                                {
                                    // return kIlwalidData retcode, but continue trying other tags
                                    // (in the best case we will skip invalid tag)
                                    ret = Error::kIlwalidData;
                                }
                            }
                        }
                        else if (first4bytes == decltype(markerData){'h', 't', 't', 'p'})
                            xmpPacket = std::vector<char>(mptr->data, mptr->data + mptr->data_length);
                    }
                }
                mptr = mptr->next;
            }
            jpeg_abort_decompress(&cinfo);
            jpeg_destroy_decompress(&cinfo);
            fclose(file);
        }
        return ret;
    }

    template<typename T>
    std::vector<unsigned char> loadJpegInternal(const T& filename, unsigned int& width, unsigned int& height, 
        BufferFormat format, std::unordered_map<uint16_t, std::string>& tags, std::vector<char>& xmpPacket)
    {
        int subsamp = 0, colorSpace = 0;
        int w = 0, h = 0;

        const auto tagLoadingError = loadJpegTagsInternal(filename, tags, xmpPacket);

        const auto tjHandle = tjInitDecompress();
        const auto file = loadFile(filename);
        if (file.empty())
            return std::vector<unsigned char>();

        int ret = tjDecompressHeader3(tjHandle, &file[0], static_cast<unsigned long>(file.size()), &w, &h, &subsamp, &colorSpace);
        width = static_cast<unsigned int>(w);
        height = static_cast<unsigned int>(h);
        if (ret == 0)
        {
            size_t bytesPerPixel = 3;
            int pixelFormat = -1;
            if (format == BufferFormat::RGB8)
                pixelFormat = TJPF_RGB;
            else if (format == BufferFormat::BGR8)
                pixelFormat = TJPF_BGR;
            else if (format == BufferFormat::RGBA8)
            {
                pixelFormat = TJPF_RGBX;
                bytesPerPixel = 4;
            }
            else if (format == BufferFormat::BGRA8)
            {
                pixelFormat = TJPF_BGRX;
                bytesPerPixel = 4;
            }

            const auto imageSizeInBytes = uint32_t(uint64_t(w) * uint64_t(h) * uint64_t(bytesPerPixel));
            std::vector<unsigned char> bgr(imageSizeInBytes);
            ret = tjDecompress2(tjHandle, &file[0], static_cast<unsigned long>(file.size()), &bgr[0], w, 0, h, pixelFormat, 0);
            if (ret == 0)
            {
                tjDestroy(tjHandle);
                return bgr;
            }
            else
                return std::vector<unsigned char>();
        }
        else
            return std::vector<unsigned char>();
    }

    template<typename T>
    Error loadJpegStreamingInternal(const T& filename, darkroom::ProcessDataCallback<unsigned char> processCallback, size_t maxBytes, unsigned int& w, unsigned int& h,
        BufferFormat format, std::unordered_map<uint16_t, std::string>& tags, std::vector<char>& xmpPacket,
        darkroom::JpegStreamingQuality quality, darkroom::JpegStreamingScale scale)
    {
        Error ret = Error::kSuccess;

        const auto tagLoadingError = loadJpegTagsInternal(filename, tags, xmpPacket);

        if (auto file = openFile(filename))
        {
            jpeg_decompress_struct cinfo;
            struct jpeg_error_mgr jerr;
            cinfo.err = jpeg_std_error(&jerr);
            jpeg_create_decompress(&cinfo);
            jpeg_stdio_src(&cinfo, file);
            jpeg_read_header(&cinfo, TRUE);
            cinfo.dct_method = JDCT_FASTEST;
            if (quality == darkroom::JpegStreamingQuality::kLow)
            {
                cinfo.do_block_smoothing = FALSE;
                cinfo.do_fancy_upsampling = FALSE;
                cinfo.enable_2pass_quant = FALSE;
            }
            cinfo.scale_num = 1;
            if (scale == darkroom::JpegStreamingScale::k1_2)
                cinfo.scale_denom = 2;
            if (scale == darkroom::JpegStreamingScale::k1_3)
                cinfo.scale_denom = 3;
            if (scale == darkroom::JpegStreamingScale::k1_4)
                cinfo.scale_denom = 4;
            if (scale == darkroom::JpegStreamingScale::k1_5)
                cinfo.scale_denom = 5;
            if (scale == darkroom::JpegStreamingScale::k1_6)
                cinfo.scale_denom = 6;
            if (scale == darkroom::JpegStreamingScale::k1_7)
                cinfo.scale_denom = 7;
            if (scale == darkroom::JpegStreamingScale::k1_8)
                cinfo.scale_denom = 8;
            size_t bytesPerPixel = 3;
            if (format == BufferFormat::RGB8)
                cinfo.out_color_space = JCS_RGB;
            else if (format == BufferFormat::BGR8)
                cinfo.out_color_space = JCS_EXT_BGR;
            else if (format == BufferFormat::RGBA8)
            {
                cinfo.out_color_space = JCS_EXT_RGBX;
                bytesPerPixel = 4;
            }
            else if (format == BufferFormat::BGRA8)
            {
                cinfo.out_color_space = JCS_EXT_BGRX;
                bytesPerPixel = 4;
            }

            jpeg_calc_output_dimensions(&cinfo);
            w = cinfo.output_width;
            h = cinfo.output_height;
            jpeg_start_decompress(&cinfo);

            const size_t maxLines = size_t((maxBytes / bytesPerPixel) / w);
            size_t lwrrentLine = 0u, rowIndex = 0u;

            if (maxLines == 0u)
                return Error::kOperationFailed;

            std::vector<unsigned char> data(maxBytes);
            std::vector<unsigned char*> rowPtrs(maxLines);

            for (auto& rowPtr : rowPtrs)
                rowPtr = data.data() + w * cinfo.num_components * rowIndex++;

            while (lwrrentLine < h)
            {
                uint32_t linesRead = 0u;
                const auto linesToRead = std::min(maxLines, h - lwrrentLine);
                while (linesRead < linesToRead)
                {
                    const auto lines = jpeg_read_scanlines(&cinfo, &rowPtrs[linesRead], JDIMENSION(linesToRead - linesRead));
                    if (lines == 0u)
                        break;
                    linesRead += lines;
                }
                if (linesRead == 0u)
                {
                    ret = processCallback(nullptr, 0u);
                    break;
                }
                lwrrentLine += linesRead;
                ret = processCallback(data.data(), linesRead);
                if (ret != Error::kSuccess)
                    break;
            }

            jpeg_abort_decompress(&cinfo);
            jpeg_destroy_decompress(&cinfo);
            fclose(file);
        }
        return ret;
    }

    template<typename T>
    Error getJpegSizeInternal(const T& filename, unsigned int& w, unsigned int& h)
    {
        Error ret = Error::kIlwalidData;

        if (auto file = openFile(filename))
        {
            jpeg_decompress_struct cinfo;
            struct jpeg_error_mgr jerr;
            cinfo.err = jpeg_std_error(&jerr);
            jpeg_create_decompress(&cinfo);
            jpeg_stdio_src(&cinfo, file);
            jpeg_read_header(&cinfo, TRUE);
            jpeg_start_decompress(&cinfo);

            w = cinfo.image_width;
            h = cinfo.image_height;

            jpeg_abort_decompress(&cinfo);
            jpeg_destroy_decompress(&cinfo);
            fclose(file);
            ret = darkroom::Error::kSuccess;
        }
        return ret;
    }
}

namespace darkroom
{
    Error saveJpeg(const unsigned char* rgb,
        RemapDataCallback remapCallback,
        const std::string& filename,
        unsigned int width,
        unsigned int height,
        BufferFormat format,
        const std::unordered_map<uint16_t, std::string>& tags,
        const std::vector<char>& xmpPacket,
        unsigned int jpegQuality)
    {
        Error err = Error::kSuccess;
        FILE* out = nullptr;
        fopen_s(&out, filename.c_str(), "wb");
        if (out)
        {
            err = saveJpegInternal(rgb, remapCallback, out, width, height, format, tags, xmpPacket, jpegQuality);
            fclose(out);
        }
        else
            return Error::kCouldntCreateFile;
        
        return err;
    }

    Error saveJpeg(const unsigned char* rgb,
        RemapDataCallback remapCallback,
        const std::wstring& filename,
        unsigned int width,
        unsigned int height,
        BufferFormat format,
        const std::unordered_map<uint16_t, std::string>& tags,
        const std::vector<char>& xmpPacket,
        unsigned int jpegQuality)
    {
        Error err = Error::kSuccess;
        FILE* out = nullptr;
        _wfopen_s(&out, filename.c_str(), L"wb");
        if (out)
        {
            err = saveJpegInternal(rgb, remapCallback, out, width, height, format, tags, xmpPacket, jpegQuality);
            fclose(out);
        }
        else
            return Error::kCouldntCreateFile;

        return err;
    }

    std::vector<unsigned char> loadJpeg(const std::string& filename, unsigned int& width, unsigned int& height, 
        BufferFormat format, std::unordered_map<uint16_t, std::string>& tags, std::vector<char>& xmpPacket)
    {
        return loadJpegInternal(filename, width, height, format, tags, xmpPacket);
    }

    std::vector<unsigned char> loadJpeg(const std::wstring& filename, unsigned int& width, unsigned int& height, 
        BufferFormat format, std::unordered_map<uint16_t, std::string>& tags, std::vector<char>& xmpPacket)
    {
        return loadJpegInternal(filename, width, height, format, tags, xmpPacket);
    }

    Error loadJpegStreaming(const std::string& filename, ProcessDataCallback<unsigned char> mapCallback, size_t maxBytes, 
        unsigned int& w, unsigned int& h, BufferFormat format, std::unordered_map<uint16_t, std::string>& tags, std::vector<char>& xmpPacket,
        JpegStreamingQuality quality, JpegStreamingScale scale)
    {
        return loadJpegStreamingInternal(filename, mapCallback, maxBytes, w, h, format, tags, xmpPacket, quality, scale);
    }

    Error loadJpegStreaming(const std::wstring& filename, ProcessDataCallback<unsigned char> mapCallback, size_t maxBytes, 
        unsigned int& w, unsigned int& h, BufferFormat format, std::unordered_map<uint16_t, std::string>& tags, std::vector<char>& xmpPacket,
        JpegStreamingQuality quality, JpegStreamingScale scale)
    {
        return loadJpegStreamingInternal(filename, mapCallback, maxBytes, w, h, format, tags, xmpPacket, quality, scale);
    }

    Error getJpegSize(const std::string& filename, unsigned int& w, unsigned int& h)
    {
        return getJpegSizeInternal(filename, w, h);
    }

    Error getJpegSize(const std::wstring& filename, unsigned int& w, unsigned int& h)
    {
        return getJpegSizeInternal(filename, w, h);
    }

    void colwertToImageMetadata(const std::unordered_map<uint16_t, std::string> & tags, const std::vector<char> & xmpPacket, ImageMetadata & metadata)
    {
        metadata = ImageMetadata();

        metadata.xmpPacket = xmpPacket;

        auto iterator = tags.find(gJPEG_TAG_SOURCE);
        if (iterator != tags.end())
        {
            metadata.tagMake = iterator->second;
        }

        iterator = tags.find(gJPEG_TAG_DESCRIPTION);
        if (iterator != tags.end())
        {
            metadata.tagDescription = iterator->second;
        }

        iterator = tags.find(gJPEG_TAG_MODEL_1);
        if (iterator != tags.end())
        {
            metadata.tagModel = iterator->second;
        }

        iterator = tags.find(gJPEG_TAG_MODEL_2);
        if (iterator != tags.end())
        {
            metadata.tagModel = iterator->second;
        }

        iterator = tags.find(gJPEG_TAG_SOFTWARE);
        if (iterator != tags.end())
        {
            metadata.tagSoftware = iterator->second;
        }

        iterator = tags.find(gJPEG_TAG_TYPE);
        if (iterator != tags.end())
        {
            metadata.tagType = iterator->second;
        }

        iterator = tags.find(gJPEG_TAG_DRSNAME);
        if (iterator != tags.end())
        {
            metadata.tagDrsName = iterator->second;
        }

        iterator = tags.find(gJPEG_TAG_DRSPROFILENAME);
        if (iterator != tags.end())
        {
            metadata.tagDrsProfileName = iterator->second;
        }

        iterator = tags.find(gJPEG_TAG_APPTITLENAME);
        if (iterator != tags.end())
        {
            metadata.tagAppTitleName = iterator->second;
        }

        iterator = tags.find(gJPEG_TAG_APPCMSID);
        if (iterator != tags.end())
        {
            metadata.tagAppCMSID = iterator->second;
        }

        iterator = tags.find(gJPEG_TAG_APPSHORTNAME);
        if (iterator != tags.end())
        {
            metadata.tagAppShortName = iterator->second;
        }

        iterator = tags.find(gJPEG_TAG_ACTIVEFILTERS);
        if (iterator != tags.end())
        {
            metadata.tagActiveFilters = iterator->second;
        }
    }

    void colwertFromImageMetadata(const ImageMetadata & metadata, std::unordered_map<uint16_t, std::string> & tags, std::vector<char>& xmpPacket)
    {
        // Description
        tags[gJPEG_TAG_DESCRIPTION] = metadata.tagDescription;
        // Source
        tags[gJPEG_TAG_SOURCE] = metadata.tagMake;

        // Model
        tags[gJPEG_TAG_MODEL_1] = metadata.tagModel;
        tags[gJPEG_TAG_MODEL_2] = metadata.tagModel;

        // Software
        tags[gJPEG_TAG_SOFTWARE] = metadata.tagSoftware;

        // Type
        tags[gJPEG_TAG_TYPE] = metadata.tagType;

        // DRS tags
        if (!metadata.tagDrsName.empty())
            tags[gJPEG_TAG_DRSNAME] = metadata.tagDrsName;
        if (!metadata.tagDrsProfileName.empty())
            tags[gJPEG_TAG_DRSPROFILENAME] = metadata.tagDrsProfileName;

        if (!metadata.tagAppTitleName.empty())
            tags[gJPEG_TAG_APPTITLENAME] = metadata.tagAppTitleName;
        if (!metadata.tagAppCMSID.empty())
            tags[gJPEG_TAG_APPCMSID] = metadata.tagAppCMSID;
        if (!metadata.tagAppShortName.empty())
            tags[gJPEG_TAG_APPSHORTNAME] = metadata.tagAppShortName;

        if (!metadata.tagActiveFilters.empty())
            tags[gJPEG_TAG_ACTIVEFILTERS] = metadata.tagActiveFilters;

        xmpPacket = metadata.xmpPacket;
    }
}
