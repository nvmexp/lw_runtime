#include "CommonTools.h"
#include "Log.h"

#include <assert.h>
#include <d3d11.h>
#include <lwapi.h>

namespace
{
    // This file is compiled by projects which don't yet compile with C++17, and utilize packman packages which contain C++
    // features that have been removed in C++17. Therefore, we cannot use C++17's clamp function and define it here locally
    template<typename T> T clamp(T x, T min, T max) { return x > max ? max : x < min ? min : x; }

    // Generate standard PQ (SMPTE2084) EOTF value
    double pqeotf_from_colour(double colorValue)
    {
        const double m1 = 0.1593017578125;
        const double m2 = 78.84375;
        const double c1 = 0.8359375;
        const double c2 = 18.8515625;
        const double c3 = 18.6875;

        // colwersion from PQ encoded values to linear values
        const double N = colorValue;
        const double L = pow(std::max<double>((pow(N, 1.0/m2) - c1), 0.0) / (c2 - c3*pow(N, 1.0/m2)), 1.0/m1);
        return clamp(L, 0.0, 1.0);
    }

    void colwert_hdr10_to_cccs(const uint32_t hdr10, float* pR, float* pG, float* pB)
    {
        assert(pR);
        assert(pG);
        assert(pB);

        // Linearize color by normalizing the value to 1.0
        double rLin = pqeotf_from_colour(((hdr10 >>  0) & 0x3FF) / 1023.0);
        double gLin = pqeotf_from_colour(((hdr10 >> 10) & 0x3FF) / 1023.0);
        double bLin = pqeotf_from_colour(((hdr10 >> 20) & 0x3FF) / 1023.0);

        // Apply luminance scaling
        rLin *= 125.0;
        gLin *= 125.0;
        bLin *= 125.0;

        // Apply Rec2020 to sRGB color space colwersion matrix
        static const double matRec2020ToSRGB[3][3] = {
            { 1.6603034854, -0.5875701425, -0.0728900602},
            {-0.1243755953,  1.1328344814, -0.0083597372},
            {-0.0181122800, -0.1005836085,  1.1187703262}
        };

        *pR = static_cast<float>(matRec2020ToSRGB[0][0] * rLin + matRec2020ToSRGB[0][1] * gLin + matRec2020ToSRGB[0][2] * bLin);
        *pG = static_cast<float>(matRec2020ToSRGB[1][0] * rLin + matRec2020ToSRGB[1][1] * gLin + matRec2020ToSRGB[1][2] * bLin);
        *pB = static_cast<float>(matRec2020ToSRGB[2][0] * rLin + matRec2020ToSRGB[2][1] * gLin + matRec2020ToSRGB[2][2] * bLin);
    }
}

namespace shadermod
{
namespace Tools
{
    template<typename T>
    void ExtractDataFromBufferTexture(ID3D11Device* device, ID3D11DeviceContext* immediateContext, ID3D11Texture2D* texture, std::vector<T> &buffer)
    {
        HRESULT status = S_OK;
        if (texture)
        {
            D3D11_MAPPED_SUBRESOURCE msr;
            D3D11_TEXTURE2D_DESC oldDesc;
            texture->GetDesc(&oldDesc);

            D3D11_TEXTURE2D_DESC newDesc;
            ID3D11Texture2D *newTexture;

            // Initialize the render target texture description.
            ZeroMemory(&newDesc, sizeof(newDesc));

            newDesc.MipLevels = oldDesc.MipLevels;
            newDesc.ArraySize = oldDesc.ArraySize;
            newDesc.SampleDesc.Count = oldDesc.SampleDesc.Count;
            newDesc.SampleDesc.Quality = oldDesc.SampleDesc.Quality;
            newDesc.MiscFlags = 0;

            newDesc.Usage = D3D11_USAGE_STAGING;
            newDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

            newDesc.Width = oldDesc.Width;
            newDesc.Height = oldDesc.Height;
            newDesc.Format = oldDesc.Format;
            newDesc.BindFlags = 0;

            if (!SUCCEEDED(status = CreateTexture2D(device, &newDesc, NULL, &newTexture)))
            {
                LOG_ERROR("Save shot failed: failed to create readback texture");
                return;
            }

            immediateContext->CopyResource(newTexture, texture);

            if (!SUCCEEDED(status = immediateContext->Map(newTexture, 0, D3D11_MAP_READ, 0, &msr)))
            {
                LOG_ERROR("Save shot failed: failed to map readback texture");
                return;
            }

            buffer.resize(newDesc.Width * newDesc.Height * 4); // Four channels per pixel

            for (unsigned int i = 0, fbheight = newDesc.Height, fbwidth = newDesc.Width; i < newDesc.Height; ++i)
            {
                memcpy(buffer.data() + 4 * fbwidth * i, (T*)msr.pData + i * (msr.RowPitch / sizeof(T)), sizeof(T) * 4 * fbwidth);
            }
        }
    }

    HRESULT shadermod::Tools::CreateTexture2D(ID3D11Device* d3dDevice, const D3D11_TEXTURE2D_DESC* pDesc, const D3D11_SUBRESOURCE_DATA* pInitialData, ID3D11Texture2D** ppTexture2D)
    {
        assert(d3dDevice != NULL);
        assert(pDesc != NULL);

        HRESULT status = S_OK;
        if (!SUCCEEDED(status = d3dDevice->CreateTexture2D(pDesc, pInitialData, ppTexture2D)))
        {
            LOG_ERROR("error: CreateTexture2D failed (%d)", status);
            __debugbreak();
            return status;
        }

        if (ppTexture2D == NULL)
        {
            // NULL is a valid value for ppTexture2D in CreateTexture2D. However this does not result in a texture being created, thus no resource hint should be set.
            return status;
        }

        IDXGIResource* pDXGIResource(NULL);
        if (!SUCCEEDED(status = (*ppTexture2D)->QueryInterface(__uuidof(IDXGIResource), (LPVOID*)&pDXGIResource)))
        {
            LOG_ERROR("error: CreateTexture2D: interface queries failed (%d)", status);
            __debugbreak();
            return status;
        }

        LWDX_ObjectHandle lwhResource = NULL;
        if (LwAPI_D3D_GetObjectHandleForResource(d3dDevice, pDXGIResource, &lwhResource) == LWAPI_OK)
        {
            LwU32 appControlledP2P = 1;
            LwAPI_D3D_SetResourceHint(d3dDevice, lwhResource, LWAPI_D3D_SRH_CATEGORY_SLI,
                LWAPI_D3D_SRH_SLI_APP_CONTROLLED_INTERFRAME_CONTENT_SYNC, &appControlledP2P);
        }

        return status;
    }

    UINT ceilingBitsToBytes(UINT numBits)
    {
        // Function rounds up to the first multiple of 8
        return (numBits + 7) / 8;
    }

    template void ExtractDataFromBufferTexture(ID3D11Device* device, ID3D11DeviceContext* immediateContext, ID3D11Texture2D* texture, std::vector<unsigned char> &buffer);
    template void ExtractDataFromBufferTexture(ID3D11Device* device, ID3D11DeviceContext* immediateContext, ID3D11Texture2D* texture, std::vector<uint16_t> &buffer);

    // Infromation HDR10 and CCCS color-spaces and why colwersions are needed are dolwmented in:
    // https://confluence.lwpu.com/display/NGX/HDR+Snapshots
    std::vector<BYTE> ColwertHDR10ToCCCS(const std::vector<BYTE>& hdr)
    {
        // We're going to split out each channel from hdr into separate float channels
        std::vector<byte> cccsData(hdr.size() * 4);

        // Use raw pointers to avoid needing to repeatedly reinterpret cast
        const uint32_t* pHdrDataUint = reinterpret_cast<const uint32_t*>(hdr.data());
        float* pCCCSDataFloat = reinterpret_cast<float*>(cccsData.data());

        for(size_t i = 0; i < hdr.size()/4; i++)
        {
            const uint32_t hdr32BitVal = *pHdrDataUint++;

            float* pR = pCCCSDataFloat++;
            float* pG = pCCCSDataFloat++;
            float* pB = pCCCSDataFloat++;
            float* pA = pCCCSDataFloat++;

            colwert_hdr10_to_cccs(hdr32BitVal, pR, pG, pB);
            *pA = 1.0f; // Discard alpha
        }

        return cccsData;
    }
}
}

