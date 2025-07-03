#pragma once

#include "PhotoShopImportDoc.h"
#include "Psd/PsdExport.h"
#include "Psd/PsdLayer.h"

#include <d3d11.h>

#include <string>
#include <vector>


class PhotoShopUtils {
public:
    void init(std::vector<std::wstring> possibleWatermarkFolders);
    void SetPsdExportEnable(bool exportAsPsd);
    const bool GetPsdExportEnable() const;

    enum class ChannelWidth
    {
        kWidth08 = 8,
        kWidth16 = 16,
        kWidth32 = 32
    };

    template <typename T>
    struct RawBufferChannels
    {
        std::vector<T> R;
        std::vector<T> G;
        std::vector<T> B;
        std::vector<T> A;

        std::vector<T> Z;       // To be used with the depth buffer only
        std::vector<T> stencil; // To be used with the depth buffer only

        void resize(const unsigned int numElements, const bool isDepthBuffer)
        {
            if (isDepthBuffer)
            {
                Z.resize(numElements);
                stencil.resize(numElements);
            }
            else
            {
                R.resize(numElements);
                G.resize(numElements);
                B.resize(numElements);
                A.resize(numElements);
            }
        }
    };

    template<typename T>
    void ExportCaptureAsPsd(const std::vector<unsigned char> &presentData, const std::vector<unsigned char> &colorData, const std::vector<unsigned char> &depthData, const std::vector<unsigned char> &hudlessData, const unsigned int width, const unsigned int height, const bool presentIsHdr, const std::wstring& shotName, const ChannelWidth channelWidthMode);

    template<typename T>
    void GetRawChannelData(const PhotoShopImportDoc *psDoc, RawBufferChannels<T> &channels, const unsigned int layerIndex);

private:

    bool m_isPsdExportEnabled = false;
    std::vector<unsigned char> m_watermarkData;

    static const unsigned int s_watermarkWidth = 518;
    static const unsigned int s_watermarkHeight = 32;

    static const std::wstring s_dstPathExt;
    static const std::wstring s_watermarkFileName;

    template<typename T>
    static void UpdateExportDocWithBuffer(psd::ExportDolwment *exDoc, psd::MallocAllocator &allocator, const std::vector<unsigned char> &bufferData, const unsigned int width, const unsigned int height, const bool isDepthBuffer, const bool isWatermark, const std::string &layerName, const ChannelWidth channelWidthMode);

    template<typename T>
    static void createPsdLayer(psd::ExportDolwment* exDoc, psd::MallocAllocator &allocator, const unsigned int layer, T *R, T *G, T *B, T *A, const unsigned int width, const unsigned int height, const bool isWatermarkLayer);
    template<typename T>
    static void createPsdLayer(psd::ExportDolwment* exDoc, psd::MallocAllocator &allocator, const unsigned int layer, T *grayChannel, T *alphaChannel, const unsigned int width, const unsigned int height, const bool isWatermarkLayer);

    template<typename T>
    static void separateDataIntoRGBAChannels(RawBufferChannels<T> &bufferChannels, const std::vector<unsigned char> &src, const unsigned int width, const unsigned int height, const bool isDepthBuffer, const ChannelWidth channelWidthMode);

    // HDR buffer functions
    template<typename T>
    static void separateHDRDataIntoRGBAChannels(RawBufferChannels<T> &bufferChannels, const std::vector<unsigned char> &src, const unsigned int width, const unsigned int height, const ChannelWidth channelWidthMode);

    template<typename T>
    static void UpdateExportDocWithPresentBuffer(psd::ExportDolwment *exDoc, psd::MallocAllocator &allocator, const std::vector<unsigned char> &bufferData, const unsigned int width, const unsigned int height, const bool presentIsHdr, const std::string &layerName, const ChannelWidth channelWidthMode);

    // Import helper functions
    unsigned int PhotoShopUtils::FindChannel(psd::Layer* layer, int16_t channelType);
    static void* ExpandChannelToCalwas(const psd::Document* document, psd::Allocator* allocator, psd::Layer* layer, psd::Channel* channel);

    template <typename T>
    static void* ExpandChannelToCalwas(psd::Allocator* allocator, const psd::Layer* layer, const void* data, unsigned int calwasWidth, unsigned int calwasHeight);
};
