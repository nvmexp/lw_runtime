#pragma once

#include <d3d11.h>
#include <string>
#include <vector>

namespace shadermod
{
namespace Tools
{
    static void ClearD3D11TexDesc(D3D11_TEXTURE2D_DESC & textureDesc)
    {
        textureDesc.Width = 0;
        textureDesc.Height = 0;
        textureDesc.MipLevels = 1;
        textureDesc.ArraySize = 1;
        textureDesc.Format = DXGI_FORMAT_UNKNOWN;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.SampleDesc.Quality = 0;
        textureDesc.Usage = D3D11_USAGE_DEFAULT;
        textureDesc.BindFlags = 0;
        textureDesc.CPUAccessFlags = 0;
        textureDesc.MiscFlags = 0;
    }

    static const std::wstring wstrNone = L"None";

    template<typename T>
    void ExtractDataFromBufferTexture(ID3D11Device* device, ID3D11DeviceContext* immediateContext, ID3D11Texture2D* texture, std::vector<T> &buffer);

    HRESULT CreateTexture2D(ID3D11Device* d3dDevice, const D3D11_TEXTURE2D_DESC* pDesc, const D3D11_SUBRESOURCE_DATA* pInitialData, ID3D11Texture2D** ppTexture2D);

    UINT ceilingBitsToBytes(UINT numBits);

    std::vector<BYTE> ColwertHDR10ToCCCS(const std::vector<BYTE>& hdr);
}
}
