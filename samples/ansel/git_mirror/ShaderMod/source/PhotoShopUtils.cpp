#define NOMINMAX

#include "PhotoShopUtils.h"
#include "Psd/PsdDolwment.h"
#include "Psd/PsdLayerMaskSection.h"
#include "Psd/PsdParseLayerMaskSection.h"
#include "Psd/PsdChannel.h"
#include "Psd/PsdChannelType.h"
#include "Psd/PsdLayerCalwasCopy.h"
#include "Log.h"
#include "darkroom/StringColwersion.h"

#include <stdio.h>
#include <tchar.h>
#include <fstream>
#include <iterator>
#include <limits>

static const unsigned int CHANNEL_NOT_FOUND = UINT_MAX;
const std::wstring PhotoShopUtils::s_dstPathExt = L"psd";
const std::wstring PhotoShopUtils::s_watermarkFileName = L"ShotWithGeforce518x32.rgba";

// Caches watermark data into a private variable
void PhotoShopUtils::init(std::vector<std::wstring> possibleWatermarkFolders)
{
    // Search through all folders where the watermark file could be until we find it
    for (std::wstring folder : possibleWatermarkFolders)
    {
        const std::wstring watermarkFile = folder + s_watermarkFileName;
        const int numWaterMarkBytes = s_watermarkHeight * s_watermarkWidth * 4;	// 4 bytes per pixel RGBA

        std::ifstream ifs(watermarkFile, std::ios::binary);
        if (ifs.is_open())
        {
            m_watermarkData.reserve(numWaterMarkBytes);
            m_watermarkData.insert(m_watermarkData.begin(), std::istream_iterator<unsigned char>(ifs), std::istream_iterator<unsigned char>());
            LOG_INFO("Found watermark file \"%s\" in \"%s\"", darkroom::getUtf8FromWstr(s_watermarkFileName).c_str(), darkroom::getUtf8FromWstr(folder).c_str());
            ifs.close();
            return;
        }
        else
        {
            LOG_WARN("Could not find watermark file \"%s\" in \"%s\"", darkroom::getUtf8FromWstr(s_watermarkFileName).c_str(), darkroom::getUtf8FromWstr(folder).c_str());
        }
    }
    LOG_ERROR("Could not find watermark file \"%s\" in app folder, Custom folder, reg key folders, installation path, or Ansel program files directory", darkroom::getUtf8FromWstr(s_watermarkFileName).c_str());
}

void PhotoShopUtils::SetPsdExportEnable(bool psdExportEnable)
{
    m_isPsdExportEnabled = psdExportEnable;
}

const bool PhotoShopUtils::GetPsdExportEnable() const
{
    return m_isPsdExportEnabled;
}

template<typename T>
void PhotoShopUtils::ExportCaptureAsPsd(const std::vector<unsigned char> &presentData, const std::vector<unsigned char> &colorData, const std::vector<unsigned char> &depthData, const std::vector<unsigned char> &hudlessData, const unsigned int width, const unsigned int height, const bool presentIsHdr, const std::wstring& shotName, const ChannelWidth channelWidthMode)
{
    psd::MallocAllocator allocator;
    psd::NativeFile file(&allocator);

    const std::wstring dstPath = shotName.substr(0, shotName.length() - s_dstPathExt.length()) + s_dstPathExt;

    // Try opening the file for PSD export. If it fails, bail out.
    if (!file.OpenWrite(dstPath.c_str()))
    {
        LOG_ERROR("Cannot open file for PSD export. There was an issue allocating psd::NativeFile using psd::MallocAllocator.\n");
        return;
    }

    psd::ExportDolwment* exDoc = CreateExportDolwment(&allocator, width, height, static_cast<unsigned int>(channelWidthMode), psd::exportColorMode::RGB);

    // Adding a buffer is like pushing to a layer stack, the order of layers shown in PhotoShop will have "Watermark Layer" at the top
    if (!depthData.empty())
    {
        UpdateExportDocWithBuffer<T>(exDoc, allocator, depthData, width, height, true, false, "Depth Layer", channelWidthMode);
    }
    if (!hudlessData.empty())
    {
        UpdateExportDocWithBuffer<T>(exDoc, allocator, hudlessData, width, height, false, false, "Hudless Layer", channelWidthMode);
    }
    UpdateExportDocWithBuffer<T>(exDoc, allocator, colorData, width, height, false, false, "Color Layer", channelWidthMode);
    UpdateExportDocWithPresentBuffer<T>(exDoc, allocator, presentData, width, height, presentIsHdr, "Present Layer", channelWidthMode);
    UpdateExportDocWithBuffer<T>(exDoc, allocator, m_watermarkData, width, height, false, true, "Watermark Layer", channelWidthMode);

    // Write the PSD
    psd::WriteDolwment(exDoc, &allocator, &file);

    // Clean up
    psd::DestroyExportDolwment(exDoc, &allocator);
    
    file.Close();
}

template<typename T>
void PhotoShopUtils::separateDataIntoRGBAChannels(RawBufferChannels<T> &bufferChannels, const std::vector<unsigned char> &src, const unsigned int width, const unsigned int height, const bool isDepthBuffer, const ChannelWidth channelWidthMode)
{
    bufferChannels.resize(width * height * 4, isDepthBuffer);

    // Separate incoming depth data from stencil data (only care about the MSByte for the depth info, this colwerts 24 bit to 8 bit)
    if (isDepthBuffer)
    {
        switch (channelWidthMode)
        {
            case ChannelWidth::kWidth08:
            {
                for (UINT i = 0; i < width * height * 4; i += 4)
                {
                    bufferChannels.stencil[i >> 2] = src[i];
                    bufferChannels.Z[i >> 2] = src[i + 1];
                }
                break;
            }
            case ChannelWidth::kWidth16:
            {
                for (UINT i = 0; i < width * height * 4; i += 4)
                {
                    // src is bitshifted left 8 to scale to 16 bits
                    bufferChannels.stencil[i >> 2] = static_cast<T>(src[i] << 8);

                    // Here we take the most significant 16 bits of depth and drop the least significant 8 bits
                    bufferChannels.Z[i >> 2] = static_cast<T>((src[i + 1] << 8) | src[i + 2]);
                }
                break;
            }
            case ChannelWidth::kWidth32:
            {
                for (UINT i = 0; i < width * height * 4; i += 4)
                {
                    // src is divided by max 8 bit value to produce float between 0.0-1.0
                    bufferChannels.stencil[i >> 2] = static_cast<T>(src[i] / 255.f);

                    unsigned int tempDepthBytes = 0;

                    // Since depth is 24 bits we must shift the incoming ilwidual bytes.
                    // We then divide the 24 bit depth by the max 24 bit number to produce a float between 0.0-1.0
                    tempDepthBytes = (src[i + 1] << 16) | (src[i + 2] << 8) | (src[i + 3]);
                    bufferChannels.Z[i >> 2] = static_cast<T>(tempDepthBytes / 16777215.f);
                }
                break;
            }
        }
    }
    else
    {
        switch (channelWidthMode)
        {
            case ChannelWidth::kWidth08:
            {
                for (UINT i = 0; i < width * height * 4; i += 4)
                {
                    bufferChannels.R[i >> 2] = src[i];
                    bufferChannels.G[i >> 2] = src[i + 1];
                    bufferChannels.B[i >> 2] = src[i + 2];
                    bufferChannels.A[i >> 2] = src[i + 3];
                }
                break;
            }
            case ChannelWidth::kWidth16:
            {
                // src is bitshifted left 8 to scale to 16 bits
                for (UINT i = 0; i < width * height * 4; i += 4)
                {
                    bufferChannels.R[i >> 2] = static_cast<T>(src[i] << 8);
                    bufferChannels.G[i >> 2] = static_cast<T>(src[i + 1] << 8);
                    bufferChannels.B[i >> 2] = static_cast<T>(src[i + 2] << 8);
                    bufferChannels.A[i >> 2] = static_cast<T>(src[i + 3] << 8);
                }
                break;
            }
            case ChannelWidth::kWidth32:
            {
                // src is divided by max 8 bit value to produce float between 0.0-1.0
                for (UINT i = 0; i < width * height * 4; i += 4)
                {
                    bufferChannels.R[i >> 2] = static_cast<T>(src[i] / 255.f);
                    bufferChannels.G[i >> 2] = static_cast<T>(src[i + 1] / 255.f);
                    bufferChannels.B[i >> 2] = static_cast<T>(src[i + 2] / 255.f);
                    bufferChannels.A[i >> 2] = static_cast<T>(src[i + 3] / 255.f);
                }
                break;
            }
        }
    }
}

template<typename T>
void PhotoShopUtils::separateHDRDataIntoRGBAChannels(RawBufferChannels<T> &bufferChannels, const std::vector<unsigned char> &src, const unsigned int width, const unsigned int height, const ChannelWidth channelWidthMode)
{
    // If the present buffer is HDR, 32 bit export should be the only option
    // I'll leave in switch statement in case we decide to allow 16 and 8 bit HDR export for psd in the future
    assert(channelWidthMode == PhotoShopUtils::ChannelWidth::kWidth32);

    const unsigned int bufferSize = width * height * 4;

    // Colwert float vector stored as unsigned chars (src) into float vector stored as floats
    std::vector<T> floatSrc;
    floatSrc.resize(bufferSize);
    memcpy(floatSrc.data(), src.data(), src.size());

    bufferChannels.resize(bufferSize, false);

    switch (channelWidthMode)
    {
        case PhotoShopUtils::ChannelWidth::kWidth32:
        {
            // We colwert groups of 4 bytes into their corresponding float value to store into bufferChannels.X
            // Unsure if this is the correct way to get float values from unsigned char vector (each float is a span of 4 bytes)
            for (UINT i = 0; i < width * height * 4; i += 4)
            {
                bufferChannels.R[i >> 2] = static_cast<T>(floatSrc[i]);
                bufferChannels.G[i >> 2] = static_cast<T>(floatSrc[i + 1]);
                bufferChannels.B[i >> 2] = static_cast<T>(floatSrc[i + 2]);
                bufferChannels.A[i >> 2] = static_cast<T>(floatSrc[i + 3]);
            }
            break;
        }
    }
}

template<typename T>
void PhotoShopUtils::createPsdLayer(psd::ExportDolwment* exDoc, psd::MallocAllocator &allocator, const unsigned int layer, T *R, T *G, T *B, T *A, const unsigned int width, const unsigned int height, const bool isWatermarkLayer)
{
    int leftIndex = isWatermarkLayer ? width - s_watermarkWidth : 0;
    int topIndex = isWatermarkLayer ? height - s_watermarkHeight : 0;

    UpdateLayer(exDoc, &allocator, layer, psd::exportChannel::RED, leftIndex, topIndex, width, height, R, psd::compressionType::RAW);
    UpdateLayer(exDoc, &allocator, layer, psd::exportChannel::GREEN, leftIndex, topIndex, width, height, G, psd::compressionType::RAW);
    UpdateLayer(exDoc, &allocator, layer, psd::exportChannel::BLUE, leftIndex, topIndex, width, height, B, psd::compressionType::RAW);
    UpdateLayer(exDoc, &allocator, layer, psd::exportChannel::ALPHA, leftIndex, topIndex, width, height, A, psd::compressionType::RAW);
}

template<typename T>
void PhotoShopUtils::createPsdLayer(psd::ExportDolwment* exDoc, psd::MallocAllocator &allocator, const unsigned int layer, T *grayChannel, T *alphaChannel, const unsigned int width, const unsigned int height, const bool isWatermarkLayer)
{
    createPsdLayer<T>(exDoc, allocator, layer, grayChannel, grayChannel, grayChannel, alphaChannel, width, height, isWatermarkLayer);
}

template <typename T>
void PhotoShopUtils::UpdateExportDocWithBuffer(psd::ExportDolwment *exDoc, psd::MallocAllocator &allocator, const std::vector<unsigned char> &bufferData, const unsigned int width, const unsigned int height, const bool isDepthBuffer, const bool isWatermark, const std::string &layerName, const ChannelWidth channelWidthMode)
{
    const unsigned int layer = psd::AddLayer(exDoc, &allocator, const_cast<char*>(layerName.c_str()));

    T fillVal = (channelWidthMode == ChannelWidth::kWidth32 ? static_cast<T>(1.0f) : std::numeric_limits<T>::max());
    static const std::vector<T> fullAlpha(width * height, fillVal);
    
    RawBufferChannels<T> channels;
    if (isWatermark)
    {
        // Separate incoming RGBA data from the buffer into individual channels
        separateDataIntoRGBAChannels<T>(channels, bufferData, s_watermarkWidth, s_watermarkHeight, false, channelWidthMode);

        createPsdLayer<T>(exDoc, allocator, layer, channels.R.data(), channels.G.data(), channels.B.data(), channels.A.data(), width, height, true);
    }
    else
    {
        // Separate incoming RGBA data from the buffer into individual channels
        separateDataIntoRGBAChannels<T>(channels, bufferData, width, height, isDepthBuffer, channelWidthMode);

        // Add buffer data to a layer of the psd file
        if (isDepthBuffer)
        {
            createPsdLayer<T>(exDoc, allocator, layer, channels.Z.data(), const_cast<T*>(fullAlpha.data()), width, height, false);

            const unsigned int layer2 = psd::AddLayer(exDoc, &allocator, "Stencil Layer");
            createPsdLayer<T>(exDoc, allocator, layer2, channels.stencil.data(), const_cast<T*>(fullAlpha.data()), width, height, false);
        }
        else
        {
            createPsdLayer<T>(exDoc, allocator, layer, channels.R.data(), channels.G.data(), channels.B.data(), const_cast<T*>(fullAlpha.data()), width, height, false);
        }
    }
}

template<typename T>
void PhotoShopUtils::UpdateExportDocWithPresentBuffer(psd::ExportDolwment *exDoc, psd::MallocAllocator &allocator, const std::vector<unsigned char> &bufferData, const unsigned int width, const unsigned int height, const bool presentIsHdr, const std::string &layerName, const ChannelWidth channelWidthMode)
{
    const unsigned int layer = psd::AddLayer(exDoc, &allocator, const_cast<char*>(layerName.c_str()));

    T fillVal = (channelWidthMode == ChannelWidth::kWidth32 ? static_cast<T>(1.0f) : std::numeric_limits<T>::max());
    static const std::vector<T> fullAlpha(width * height, fillVal);

    RawBufferChannels<T> channels;

    // Separate incoming RGBA data from the buffer into individual channels
    if (!presentIsHdr)
    {
        separateDataIntoRGBAChannels<T>(channels, bufferData, width, height, false, channelWidthMode);
    }
    else
    {
        separateHDRDataIntoRGBAChannels<T>(channels, bufferData, width, height, channelWidthMode);
    }

    // Add buffer data to a layer of the psd file
    createPsdLayer<T>(exDoc, allocator, layer, channels.R.data(), channels.G.data(), channels.B.data(), const_cast<T*>(fullAlpha.data()), width, height, false);
}

template<typename T>
void PhotoShopUtils::GetRawChannelData(const PhotoShopImportDoc *psDoc, RawBufferChannels<T> &channels, const unsigned int layerIndex)
{
    assert(psDoc);

    // Function fills 'channels' with values from the imported psd document and the specified layer
    psd::LayerMaskSection *layerMaskSection = psDoc->GetLayerMaskSection();
    psd::Document *document = psDoc->GetDolwment();
    psd::MallocAllocator allocator = psDoc->GetMallocAllocator();
    psd::NativeFile file = psDoc->GetNativeFile();

    int numElements = document->width * document->height;

    if (layerMaskSection)
    {
        psd::Layer* layer = &layerMaskSection->layers[layerIndex];
        ExtractLayer(document, &file, &allocator, layer);

        // Check availability of R, G, B, and A channels
        // We need to determine the indices of channels individually, because there is no guarantee that R is the first channel,
        // G is the second, B is the third, and so on
        const unsigned int indexR = FindChannel(layer, psd::channelType::R);
        const unsigned int indexG = FindChannel(layer, psd::channelType::G);
        const unsigned int indexB = FindChannel(layer, psd::channelType::B);
        const unsigned int indexA = FindChannel(layer, psd::channelType::TRANSPARENCY_MASK);

        // Note that channel data is only as big as the layer it belongs to, e.g. it can be smaller or bigger than the calwas,
        // depending on where it is positioned. Therefore, we use the provided utility functions to expand/shrink the channel data
        // to the calwas size. Of course, you can work with the channel data directly if you need to
        void* calwasData[4] = {};
        if ((indexR != CHANNEL_NOT_FOUND) && (indexG != CHANNEL_NOT_FOUND) && (indexB != CHANNEL_NOT_FOUND))
        {
            // RGB channels were found
            calwasData[0] = ExpandChannelToCalwas(document, &allocator, layer, &layer->channels[indexR]);
            calwasData[1] = ExpandChannelToCalwas(document, &allocator, layer, &layer->channels[indexG]);
            calwasData[2] = ExpandChannelToCalwas(document, &allocator, layer, &layer->channels[indexB]);
            channels.R = std::vector<T> (static_cast<T*>(calwasData[0]), static_cast<T*>(calwasData[0]) + numElements);
            channels.G = std::vector<T> (static_cast<T*>(calwasData[1]), static_cast<T*>(calwasData[1]) + numElements);
            channels.B = std::vector<T> (static_cast<T*>(calwasData[2]), static_cast<T*>(calwasData[2]) + numElements);

            if (indexA != CHANNEL_NOT_FOUND)
            {
                // A channel was also found
                calwasData[3] = ExpandChannelToCalwas(document, &allocator, layer, &layer->channels[indexA]);
                channels.A = std::vector<T> (static_cast<T*>(calwasData[3]), static_cast<T*>(calwasData[3]) + numElements);
            }
        }
    }
}

unsigned int PhotoShopUtils::FindChannel(psd::Layer* layer, int16_t channelType)
{
    for (unsigned int i = 0; i < layer->channelCount; ++i)
    {
        psd::Channel* channel = &layer->channels[i];
        if (channel->data && channel->type == channelType)
        {
            return i;
        }
    }

    return CHANNEL_NOT_FOUND;
}

template <typename T>
void* PhotoShopUtils::ExpandChannelToCalwas(psd::Allocator* allocator, const psd::Layer* layer, const void* data, unsigned int calwasWidth, unsigned int calwasHeight)
{
    const unsigned int numBytes = sizeof(T) * calwasWidth * calwasHeight;
    T* calwasData = static_cast<T*>(allocator->Allocate(numBytes, 16u));
    memset(calwasData, 0u, numBytes);

    psd::imageUtil::CopyLayerData(static_cast<const T*>(data), calwasData, layer->left, layer->top, layer->right, layer->bottom, calwasWidth, calwasHeight);

    return calwasData;
}

void* PhotoShopUtils::ExpandChannelToCalwas(const psd::Document* document, psd::Allocator* allocator, psd::Layer* layer, psd::Channel* channel)
{
    if (document->bitsPerChannel == 8)
    {
        return ExpandChannelToCalwas<uint8_t>(allocator, layer, channel->data, document->width, document->height);
    }
    else if (document->bitsPerChannel == 16)
    {
        return ExpandChannelToCalwas<uint16_t>(allocator, layer, channel->data, document->width, document->height);
    }
    else if (document->bitsPerChannel == 32)
    {
        return ExpandChannelToCalwas<float32_t>(allocator, layer, channel->data, document->width, document->height);
    }

    return nullptr;
}

template void PhotoShopUtils::separateDataIntoRGBAChannels(RawBufferChannels<unsigned char> &bufferChannels, const std::vector<unsigned char> &src, const unsigned int width, const unsigned int height, const bool isDepthBuffer, const ChannelWidth channelWidthMode);
template void PhotoShopUtils::separateDataIntoRGBAChannels(RawBufferChannels<uint16_t> &bufferChannels, const std::vector<unsigned char> &src, const unsigned int width, const unsigned int height, const bool isDepthBuffer, const ChannelWidth channelWidthMode);
template void PhotoShopUtils::separateDataIntoRGBAChannels(RawBufferChannels<float32_t> &bufferChannels, const std::vector<unsigned char> &src, const unsigned int width, const unsigned int height, const bool isDepthBuffer, const ChannelWidth channelWidthMode);

template void PhotoShopUtils::createPsdLayer(psd::ExportDolwment* exDoc, psd::MallocAllocator &allocator, const unsigned int layer, unsigned char *R, unsigned char *G, unsigned char *B, unsigned char *A, const unsigned int width, const unsigned int height, const bool isWatermarkLayer);
template void PhotoShopUtils::createPsdLayer(psd::ExportDolwment* exDoc, psd::MallocAllocator &allocator, const unsigned int layer, unsigned char *grayChannel, unsigned char *alphaChannel, const unsigned int width, const unsigned int height, const bool isWatermarkLayer);

template void PhotoShopUtils::createPsdLayer(psd::ExportDolwment* exDoc, psd::MallocAllocator &allocator, const unsigned int layer, uint16_t *R, uint16_t *G, uint16_t *B, uint16_t *A, const unsigned int width, const unsigned int height, const bool isWatermarkLayer);
template void PhotoShopUtils::createPsdLayer(psd::ExportDolwment* exDoc, psd::MallocAllocator &allocator, const unsigned int layer, uint16_t *grayChannel, uint16_t *alphaChannel, const unsigned int width, const unsigned int height, const bool isWatermarkLayer);

template void PhotoShopUtils::createPsdLayer(psd::ExportDolwment* exDoc, psd::MallocAllocator &allocator, const unsigned int layer, float32_t *R, float32_t *G, float32_t *B, float32_t *A, const unsigned int width, const unsigned int height, const bool isWatermarkLayer);
template void PhotoShopUtils::createPsdLayer(psd::ExportDolwment* exDoc, psd::MallocAllocator &allocator, const unsigned int layer, float32_t *grayChannel, float32_t *alphaChannel, const unsigned int width, const unsigned int height, const bool isWatermarkLayer);

template void PhotoShopUtils::separateHDRDataIntoRGBAChannels(RawBufferChannels<float32_t> &bufferChannels, const std::vector<unsigned char> &src, const unsigned int width, const unsigned int height, const ChannelWidth channelWidthMode);

template void PhotoShopUtils::UpdateExportDocWithBuffer<unsigned char>(psd::ExportDolwment *exDoc, psd::MallocAllocator &allocator, const std::vector<unsigned char> &bufferData, const unsigned int width, const unsigned int height, const bool isDepthBuffer, const bool isWatermark, const std::string &layerName, const ChannelWidth channelWidthMode);
template void PhotoShopUtils::UpdateExportDocWithBuffer<uint16_t>(psd::ExportDolwment *exDoc, psd::MallocAllocator &allocator, const std::vector<unsigned char> &bufferData, const unsigned int width, const unsigned int height, const bool isDepthBuffer, const bool isWatermark, const std::string &layerName, const ChannelWidth channelWidthMode);
template void PhotoShopUtils::UpdateExportDocWithBuffer<float32_t>(psd::ExportDolwment *exDoc, psd::MallocAllocator &allocator, const std::vector<unsigned char> &bufferData, const unsigned int width, const unsigned int height, const bool isDepthBuffer, const bool isWatermark, const std::string &layerName, const ChannelWidth channelWidthMode);

template void PhotoShopUtils::ExportCaptureAsPsd<unsigned char>(const std::vector<unsigned char> &presentData, const std::vector<unsigned char> &colorData, const std::vector<unsigned char> &depthData, const std::vector<unsigned char> &hudlessData, const unsigned int width, const unsigned int height, const bool presentIsHdr, const std::wstring& shotName, const ChannelWidth channelWidthMode);
template void PhotoShopUtils::ExportCaptureAsPsd<uint16_t>(const std::vector<unsigned char> &presentData, const std::vector<unsigned char> &colorData, const std::vector<unsigned char> &depthData, const std::vector<unsigned char> &hudlessData, const unsigned int width, const unsigned int height, const bool presentIsHdr, const std::wstring& shotName, const ChannelWidth channelWidthMode);
template void PhotoShopUtils::ExportCaptureAsPsd<float32_t>(const std::vector<unsigned char> &presentData, const std::vector<unsigned char> &colorData, const std::vector<unsigned char> &depthData, const std::vector<unsigned char> &hudlessData, const unsigned int width, const unsigned int height, const bool presentIsHdr, const std::wstring& shotName, const ChannelWidth channelWidthMode);

template void PhotoShopUtils::UpdateExportDocWithPresentBuffer<unsigned char>(psd::ExportDolwment *exDoc, psd::MallocAllocator &allocator, const std::vector<unsigned char> &bufferData, const unsigned int width, const unsigned int height, const bool presentIsHdr, const std::string &layerName, const ChannelWidth channelWidthMode);
template void PhotoShopUtils::UpdateExportDocWithPresentBuffer<uint16_t>(psd::ExportDolwment *exDoc, psd::MallocAllocator &allocator, const std::vector<unsigned char> &bufferData, const unsigned int width, const unsigned int height, const bool presentIsHdr, const std::string &layerName, const ChannelWidth channelWidthMode);
template void PhotoShopUtils::UpdateExportDocWithPresentBuffer<float32_t>(psd::ExportDolwment *exDoc, psd::MallocAllocator &allocator, const std::vector<unsigned char> &bufferData, const unsigned int width, const unsigned int height, const bool presentIsHdr, const std::string &layerName, const ChannelWidth channelWidthMode);

template void PhotoShopUtils::GetRawChannelData<unsigned char>(const PhotoShopImportDoc *psDoc, RawBufferChannels<unsigned char> &channels, const unsigned int layerIndex);
template void PhotoShopUtils::GetRawChannelData<uint16_t>(const PhotoShopImportDoc *psDoc, RawBufferChannels<uint16_t> &channels, const unsigned int layerIndex);
template void PhotoShopUtils::GetRawChannelData<float32_t>(const PhotoShopImportDoc *psDoc, RawBufferChannels<float32_t> &channels, const unsigned int layerIndex);
