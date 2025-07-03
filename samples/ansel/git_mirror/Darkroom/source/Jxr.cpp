#include <wrl/client.h>

#include "darkroom/StringColwersion.h"
#include "../../ShaderMod/source/CommonTools.h"
#include "../../ShaderMod/source/Log.h"

#include "darkroom/Jxr.h"

#include <system_error>

using Microsoft::WRL::ComPtr;

namespace
{
    // If needed, we'll update the input data to something that's easier to work with. Also setup various
    // parameters about the data that will be used later during WIC file creation
    const std::vector<BYTE> update_raw_pixel_data(const std::vector<BYTE>& hdrData, const unsigned int width,
        const unsigned int height, const bool forceFloatDst, WICPixelFormatGUID* pSrcFormatGUID, UINT *pSrcBitsPerPixel,
        UINT *pSrcStride, WICPixelFormatGUID *pDstFormatGUID)
    {
        assert(pSrcFormatGUID);
        assert(pSrcBitsPerPixel);
        assert(pSrcStride);
        assert(pDstFormatGUID);

        std::vector<BYTE> updatedHdrData;

        if (*pSrcFormatGUID == GUID_WICPixelFormat128bppRGBAFloat)
        {
            static const int numChannels = 4;
            const unsigned int numChannelValues = width * height * numChannels;

            // Make a copy of the data that we'll soon update
            updatedHdrData = hdrData;

            // Force alpha to 1.0f on float-type buffers
            float *pAlpha = reinterpret_cast<float*>(updatedHdrData.data());
            for (UINT i = 0; i < numChannelValues; i += numChannels)
            {
                pAlpha[i + (numChannels - 1)] = 1.0f;
            }

        }
        else if (*pSrcFormatGUID == GUID_WICPixelFormat32bppRGBA1010102)
        {
            // If we want the output to be colwerted to float HDR, then colwert the source first.
            if (forceFloatDst)
            {
                // Colwert RGB10A2 to float16 with appropriate color-space colwersions
                updatedHdrData = shadermod::Tools::ColwertHDR10ToCCCS(hdrData);

                *pSrcFormatGUID = GUID_WICPixelFormat128bppRGBAFloat;
            }
            else
            {
                // Otherwise, leave input as-is, since WIC will perform a colwersion on the destination
                updatedHdrData = hdrData;
            }
        }
        else
        {
            LOG_ERROR("saveJxr lwrrently only supports GUID_WICPixelFormat32bppRGBA1010102/"
                "GUID_WICPixelFormat128bppRGBAFloat. Can't create JXR with format: %s",
                darkroom::GuidToString(*pSrcFormatGUID).c_str());
            return updatedHdrData;
        }

        // Update output values after having changed the input format
        if (*pSrcFormatGUID == GUID_WICPixelFormat128bppRGBAFloat)
        {
            *pSrcBitsPerPixel = 128;
            *pDstFormatGUID = GUID_WICPixelFormat128bppRGBAFloat;
        }
        else
        {
            *pSrcBitsPerPixel = 32;
            *pDstFormatGUID = GUID_WICPixelFormat32bppBGR101010;
        }
        *pSrcStride = shadermod::Tools::ceilingBitsToBytes(*pSrcBitsPerPixel * width);

        return updatedHdrData;
    }

    HRESULT colwert_hdrdata_into_bitmap_frame(ComPtr<IWICImagingFactory> pFactory, const unsigned int width,
        const unsigned int height, const std::vector<BYTE>& hdrData, const WICPixelFormatGUID srcFormatGUID,
        const WICPixelFormatGUID dstFormatGUID, const UINT srcStride, ComPtr<IWICBitmapFrameEncode> pBitmapFrame)
    {
        HRESULT hr = S_OK;

        ComPtr<IWICBitmap> pSrcBitmap;
        hr =  pFactory->CreateBitmapFromMemory(
            width,
            height,
            srcFormatGUID,
            srcStride,
            static_cast<UINT>(hdrData.size()),
            const_cast<BYTE*>(hdrData.data()),
            &pSrcBitmap);
        if (!SUCCEEDED(hr))
        {
            LOG_ERROR("CreateBitmapFromMemory failed with error: %s", std::system_category().message(hr).c_str());
            return hr;
        }

        // Create a format colwerter that will populate the BitmapFrame with desired dst hdr data
        ComPtr<IWICFormatColwerter> pFormatColwerter;
        hr = pFactory->CreateFormatColwerter(&pFormatColwerter);
        if (!SUCCEEDED(hr))
        {
            LOG_ERROR("CreateFormatColwerter failed with error: %s", std::system_category().message(hr).c_str());
            return hr;
        }

        // Validate that we can indeed colwert between src and dst formats
        BOOL canColwert = false;
        hr = pFormatColwerter->CanColwert(srcFormatGUID, dstFormatGUID, &canColwert);
        if (!SUCCEEDED(hr) || !canColwert)
        {
            LOG_ERROR("FormatColwerter->CanColwert returned %d with error: %s",
                canColwert, std::system_category().message(hr).c_str());
            return hr;
        }

        // Initialize the format colwerter.
        hr = pFormatColwerter->Initialize(
            pSrcBitmap.Get(),               // Input source to colwert
            dstFormatGUID,                  // Destination pixel format
            WICBitmapDitherTypeNone,        // Specified dither pattern
            NULL,                           // Specify a particular palette
            0.f,                            // Alpha threshold
            WICBitmapPaletteTypeLwstom      // Palette translation type
        );
        if (!SUCCEEDED(hr))
        {
            LOG_ERROR("FormatColwerter->Initialize failed with error: %s", std::system_category().message(hr).c_str());
            return hr;
        }

        ComPtr<IWICMetadataQueryWriter> pMetawriter;
        if (SUCCEEDED(pBitmapFrame->GetMetadataQueryWriter(&pMetawriter)))
        {
            PROPVARIANT value;
            PropVariantInit(&value);

            value.vt = VT_LPSTR;
            value.pszVal = const_cast<char*>("LwCamera"); // TODO(david): Put in a proper tag

            // Set Software name
            hr = pMetawriter->SetMetadataByName(L"System.ApplicationName", &value);
            if (!SUCCEEDED(hr))
            {
                LOG_ERROR("Metawriter->SetMetadataByName(ApplicationName) failed with error: %s",
                    std::system_category().message(hr).c_str());
                return hr;
            }

            // HACK: This indicates the screenshot is an "Xbox" screenshot, which some tools then know to interpret the pixel data
            // as HDR10. For example, the "HDR + WCG Viewer" created by Micrsoft employees has a check for it:
            // https://github.com/13thsymphony/HDRImageViewer/blob/master/HDRImageViewer/ImageLoader.cpp#L654
            hr = pMetawriter->SetMetadataByName(L"/ifd/xmp/{wstr=http://ns.microsoft.com/gamedvr/1.0/}:Extended",
                &value);
            if (!SUCCEEDED(hr))
            {
                LOG_ERROR("Metawriter->SetMetadataByName(Xbox) failed with error: %s",
                    std::system_category().message(hr).c_str());
                return hr;
            }
        }

        return pBitmapFrame->WriteSource(pFormatColwerter.Get(), NULL);
    }
}

namespace darkroom
{
    // This function is used to create a .jxr file of the game's Present buffer (screenshot) when GFE detects a game running
    // with HDR
    Error saveJxr(const std::vector<BYTE>& inputHdrData, const unsigned int width, const unsigned int height,
        const WICPixelFormatGUID inputSrcFormatGUID, const std::wstring& shotName, bool forceFloatDst, bool doLossless)
    {
        LOG_DEBUG("saveJxr start");

        UINT srcBitsPerPixel = 0;
        UINT srcStride = 0;
        WICPixelFormatGUID srcFormatGUID = inputSrcFormatGUID;
        WICPixelFormatGUID dstFormatGUID = GUID_WICPixelFormat128bppRGBAFloat;

        const std::vector<BYTE> hdrData = update_raw_pixel_data(inputHdrData, width, height, forceFloatDst,
            &srcFormatGUID, &srcBitsPerPixel, &srcStride, &dstFormatGUID);
        if (hdrData.empty())
        {
            return Error::kCouldntCreateFile;
        }

        // From: https://docs.microsoft.com/en-us/windows/win32/wic/-wic-creating-encoder
        ComPtr<IWICImagingFactory> pFactory;
        ComPtr<IWICBitmapEncoder> pEncoder;
        ComPtr<IWICBitmapFrameEncode> pBitmapFrame;
        ComPtr<IPropertyBag2> pPropertybag;
        ComPtr<IWICStream> pStream;

        HRESULT hr = CoCreateInstance(
            CLSID_WICImagingFactory,
            NULL,
            CLSCTX_INPROC_SERVER,
            IID_IWICImagingFactory,
            (LPVOID*)&pFactory);
        if (!SUCCEEDED(hr))
        {
            LOG_ERROR("CoCreateInstance failed with error: %s", std::system_category().message(hr).c_str());
            return Error::kCouldntCreateFile;
        }

        hr = pFactory->CreateStream(&pStream);
        if (!SUCCEEDED(hr))
        {
            LOG_ERROR("CreateStream failed with error: %s", std::system_category().message(hr).c_str());
            return Error::kCouldntCreateFile;
        }

        hr = pStream->InitializeFromFilename(shotName.c_str(), GENERIC_WRITE);
        if (!SUCCEEDED(hr))
        {
            LOG_ERROR("Stream->InitializeFromFilename failed with error: %s",
                std::system_category().message(hr).c_str());
            return Error::kCouldntCreateFile;
        }

        hr = pFactory->CreateEncoder(GUID_ContainerFormatWmp, NULL, &pEncoder);
        if (!SUCCEEDED(hr))
        {
            LOG_ERROR("CreateEncoder failed with error: %s", std::system_category().message(hr).c_str());
            return Error::kCouldntCreateFile;
        }

        hr = pEncoder->Initialize(pStream.Get(), WICBitmapEncoderNoCache);
        if (!SUCCEEDED(hr))
        {
            LOG_ERROR("Encoder->Initialize failed with error: %s", std::system_category().message(hr).c_str());
            return Error::kCouldntCreateFile;
        }

        hr = pEncoder->CreateNewFrame(&pBitmapFrame, &pPropertybag);
        if (!SUCCEEDED(hr))
        {
            LOG_ERROR("Encoder->CreateNewFrame failed with error: %s", std::system_category().message(hr).c_str());
            return Error::kCouldntCreateFile;
        }

        // If requested, disable compression to create a lossless JXR file
        if (doLossless)
        {
            PROPBAG2 option = { 0 };
            option.pstrName = L"Lossless";
            VARIANT varValue;
            VariantInit(&varValue);
            varValue.vt = VT_BOOL;
            varValue.boolVal = VARIANT_TRUE;

            hr = pPropertybag->Write(1, &option, &varValue);
            if (!SUCCEEDED(hr))
            {
                LOG_ERROR("Propertybag->Write failed with error: %s", std::system_category().message(hr).c_str());
                return Error::kCouldntCreateFile;
            }
        }

        hr = pBitmapFrame->Initialize(pPropertybag.Get());
        if (!SUCCEEDED(hr))
        {
            LOG_ERROR("BitmapFrame->Initialize failed with error: %s", std::system_category().message(hr).c_str());
            return Error::kCouldntCreateFile;
        }

        hr = pBitmapFrame->SetSize(width, height);
        if (!SUCCEEDED(hr))
        {
            LOG_ERROR("BitmapFrame->SetSize failed with error: %s", std::system_category().message(hr).c_str());
            return Error::kCouldntCreateFile;
        }

        hr = pBitmapFrame->SetPixelFormat(&dstFormatGUID);
        if (!SUCCEEDED(hr))
        {
            LOG_ERROR("BitmapFrame->SetPixelFormat failed with error: %s", std::system_category().message(hr).c_str());
            return Error::kCouldntCreateFile;
        }

        // Sanity check that we're able to set the desired desitnation format
        if (dstFormatGUID != GUID_WICPixelFormat128bppRGBAFloat &&
            dstFormatGUID != GUID_WICPixelFormat32bppBGR101010)
        {
            LOG_ERROR("Unexpected dstFormatGUID: %s", darkroom::GuidToString(dstFormatGUID).c_str());
            return Error::kCouldntCreateFile;
        }

        // If our src and dst formats aren't the same, we need to create a bitmap that will be used to
        // colwert between the formats
        if (srcFormatGUID != dstFormatGUID)
        {
            LOG_DEBUG("saveJxr performing format colwersion between %s and %s",
                darkroom::GuidToString(srcFormatGUID).c_str(),
                darkroom::GuidToString(dstFormatGUID).c_str());

            hr = colwert_hdrdata_into_bitmap_frame(pFactory, width, height, hdrData, srcFormatGUID, dstFormatGUID,
                srcStride, pBitmapFrame);
        }
        // Otherwise, use the hdrData directly since it's already in 128bppRGBAFloat
        else
        {
            const UINT cbBufferSize = height * srcStride;
            hr = pBitmapFrame->WritePixels(height, srcStride, cbBufferSize, const_cast<BYTE *>(hdrData.data()));
        }

        if (!SUCCEEDED(hr))
        {
            LOG_ERROR("BitmapFrame->Write failed with error: %s", std::system_category().message(hr).c_str());
            return Error::kIlwalidArgument;
        }

        // Commit the output data and finalize the file
        hr = pBitmapFrame->Commit();
        if (!SUCCEEDED(hr))
        {
            LOG_ERROR("BitmapFrame->Commit failed with error: %s", std::system_category().message(hr).c_str());
            return Error::kCouldntCreateFile;
        }

        hr = pEncoder->Commit();
        if (!SUCCEEDED(hr))
        {
            LOG_ERROR("Encoder->Commit failed with error: %s", std::system_category().message(hr).c_str());
            return Error::kCouldntCreateFile;
        }

        LOG_DEBUG("saveJxr finished successfully");

        return Error::kSuccess;
    }
}
