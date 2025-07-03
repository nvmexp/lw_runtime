#define NOMINMAX
#include "ImageLoaderWIC.h"
#include <algorithm>
#pragma warning(push)
#pragma warning(disable:4668 4917)
#define WINDOWS_LEAN_AND_MEAN
#include <Windows.h>
#include <Wincodec.h>
#pragma warning(pop)
#include <atlbase.h>

#include "darkroom/StringColwersion.h"

namespace
{
    class ComContext
    {
    public:
        ComContext() { m_initialized = SUCCEEDED(CoInitializeEx(NULL, COINIT_MULTITHREADED)); }
        ~ComContext() { if (m_initialized) { CoUninitialize(); } }
    protected:
        bool m_initialized;
    };
}

namespace darkroom
{
    Error getImageSizeWinApiInternal(const std::wstring& filename, unsigned int& w, unsigned int& h)
    {
        CComPtr<IWICImagingFactory> pFactory;
        CComPtr<IWICBitmapDecoder> pDecoder;
        CComPtr<IWICBitmapSource> pSource;
        CComPtr<IWICBitmapFrameDecode> pDecoderFrame;

        ComContext ctx;

        HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pFactory));

        if (FAILED(hr))
            return Error::kOperationFailed;

        hr = pFactory->CreateDecoderFromFilename(filename.c_str(), NULL, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &pDecoder);

        if (FAILED(hr))
            return Error::kOperationFailed;

        hr = pDecoder->GetFrame(0, &pDecoderFrame);

        if (FAILED(hr))
            return Error::kOperationFailed;

        hr = pDecoderFrame->GetSize(&w, &h);
        if (FAILED(hr))
            return Error::kOperationFailed;

        return Error::kSuccess;
    }

    std::vector<unsigned char> loadImageWinApiInternal(const std::wstring& filename, unsigned int& w, unsigned int& h, BufferFormat format)
    {
        std::vector<unsigned char> result;

        CComPtr<IWICImagingFactory> pFactory;
        CComPtr<IWICBitmapDecoder> pDecoder;
        CComPtr<IWICBitmapSource> pSource;

        CComPtr<IWICBitmapFrameDecode> pDecoderFrame;
        CComPtr<IWICBitmapSource> pColwerter;

        ComContext ctx;

        WICPixelFormatGUID targetPixelFormat;

        uint32_t channels = 3u;

        if (format == BufferFormat::BGR8)
            targetPixelFormat = GUID_WICPixelFormat24bppBGR;
        else if (format == BufferFormat::RGB8)
            targetPixelFormat = GUID_WICPixelFormat24bppRGB;
        else if (format == BufferFormat::BGRA8)
        {
            targetPixelFormat = GUID_WICPixelFormat32bppBGRA;
            channels = 4u;
        }
        else if (format == BufferFormat::RGBA8)
        {
            targetPixelFormat = GUID_WICPixelFormat32bppRGBA;
            channels = 4u;
        }
        else
            return result;

        HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pFactory));

        if (FAILED(hr))
            return result;

        hr = pFactory->CreateDecoderFromFilename(filename.c_str(), NULL, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &pDecoder);

        if (FAILED(hr))
            return result;

        hr = pDecoder->GetFrame(0, &pDecoderFrame);

        if (FAILED(hr))
            return result;

        hr = pDecoderFrame->GetSize(&w, &h);
        if (FAILED(hr))
            return result;

        WICPixelFormatGUID sourcePixelFormat;
        hr = pDecoderFrame->GetPixelFormat(&sourcePixelFormat);

        pSource = pDecoderFrame;
        if (!IsEqualGUID(sourcePixelFormat, targetPixelFormat))
        {
            hr = WICColwertBitmapSource(targetPixelFormat, pSource, &pColwerter);

            if (FAILED(hr))
                return result;

            pSource = pColwerter;
        }

        result.resize(size_t(uint64_t(w) * uint64_t(h) * uint64_t(channels)));
        hr = pSource->CopyPixels(nullptr, UINT(channels * w), UINT(result.size()), result.data());

        if (FAILED(hr))
            return std::vector<unsigned char>();

        return result;
    }

    Error loadImageStreamingWinApiInternal(const std::wstring& filename, ProcessDataCallback<unsigned char> callback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format)
    {
        auto err = darkroom::Error::kSuccess;
        CComPtr<IWICImagingFactory> pFactory;
        CComPtr<IWICBitmapDecoder> pDecoder;
        CComPtr<IWICBitmapSource> pSource;
        CComPtr<IWICBitmapFrameDecode> pDecoderFrame;
        CComPtr<IWICBitmapSource> pColwerter;

        ComContext ctx;

        WICPixelFormatGUID targetPixelFormat;

        uint32_t channels = 3u;

        if (format == BufferFormat::BGR8)
            targetPixelFormat = GUID_WICPixelFormat24bppBGR;
        else if (format == BufferFormat::RGB8)
            targetPixelFormat = GUID_WICPixelFormat24bppRGB;
        else if (format == BufferFormat::BGRA8)
        {
            targetPixelFormat = GUID_WICPixelFormat32bppBGRA;
            channels = 4u;
        }
        else if (format == BufferFormat::RGBA8)
        {
            targetPixelFormat = GUID_WICPixelFormat32bppRGBA;
            channels = 4u;
        }
        else
            return Error::kOperationFailed;

        HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pFactory));

        if (FAILED(hr))
            return Error::kOperationFailed;

        hr = pFactory->CreateDecoderFromFilename(filename.c_str(), NULL, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &pDecoder);

        if (FAILED(hr))
            return Error::kOperationFailed;

        hr = pDecoder->GetFrame(0, &pDecoderFrame);

        if (FAILED(hr))
            return Error::kOperationFailed;

        hr = pDecoderFrame->GetSize(&w, &h);
        if (FAILED(hr))
            return Error::kOperationFailed;

        WICPixelFormatGUID sourcePixelFormat;
        hr = pDecoderFrame->GetPixelFormat(&sourcePixelFormat);

        pSource = pDecoderFrame;
        if (!IsEqualGUID(sourcePixelFormat, targetPixelFormat))
        {
            hr = WICColwertBitmapSource(targetPixelFormat, pSource, &pColwerter);

            if (FAILED(hr))
                return Error::kOperationFailed;

            pSource = pColwerter;   // let's treat the 128bppPABGR colwerter as the source
        }
        std::vector<unsigned char> data(maxBytes);
        const uint32_t maxLines = uint32_t((maxBytes / channels) / w);

        if (maxLines == 0u)
            return Error::kOperationFailed;

        WICRect rect;
        rect.X = 0;
        rect.Y = 0;
        rect.Width = INT(w);

        while (uint32_t(rect.Y) < h)
        {
            rect.Height = INT(std::min(maxLines, h - rect.Y));
            hr = pSource->CopyPixels(&rect, UINT(channels * w), UINT(data.size()), data.data());

            if (FAILED(hr))
                return Error::kOperationFailed;

            err = callback(data.data(), uint32_t(rect.Height));
            if (err != Error::kSuccess)
                break;

            rect.Y += rect.Height;
        }

        return err;
    }

    Error loadImageStreamingWinApiInternal(const std::string& filename, ProcessDataCallback<unsigned char> callback, size_t maxBytes, unsigned int& w, unsigned int& h, BufferFormat format)
    {
        return loadImageStreamingWinApiInternal(darkroom::getWstrFromUtf8(filename), callback, maxBytes, w, h, format);
    }

    std::vector<unsigned char> loadImageWinApiInternal(const std::string& filename, unsigned int& w, unsigned int& h, BufferFormat format)
    {
        return loadImageWinApiInternal(darkroom::getWstrFromUtf8(filename), w, h, format);
    }

    Error getImageSizeWinApiInternal(const std::string& filename, unsigned int & w, unsigned int & h)
    {
        return getImageSizeWinApiInternal(darkroom::getWstrFromUtf8(filename), w, h);
    }
}
