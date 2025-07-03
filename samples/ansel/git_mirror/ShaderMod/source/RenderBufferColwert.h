#pragma once

#include "Ansel.h"
#include "RenderBuffer.h"
#include "Utils.h"
#include "D3DCompilerHandler.h"

class AnselBufferColwerter
{
public:

    ID3D11Device *          m_d3dDevice = nullptr;
    ID3D11DeviceContext *   m_d3dImmediateContext = nullptr;
    D3DCompilerHandler *    m_d3dCompiler = nullptr;

    void setRenderer(ID3D11Device * d3dDevice, ID3D11DeviceContext * d3dImmediateContext, D3DCompilerHandler * d3dCompiler);

    struct HelperD3DStates
    {
        ID3D11InputLayout *         pInputLayout = nullptr;
        ID3D11VertexShader *        pVS = nullptr;
        ID3D11RasterizerState *     pRasterizerState = nullptr;
        ID3D11SamplerState *        pLinearSamplerState = nullptr;
        ID3D11SamplerState *        pPointSamplerState = nullptr;
        ID3D11BlendState *          pBlendState = nullptr;
        ID3D11DepthStencilState *   pDepthStencilState = nullptr;
        ID3D11Buffer *              pConstBuf = nullptr;
        ID3D11Buffer *              pRectVertexBuf = nullptr;
        ID3D11Buffer *              pRectIndexBuf = nullptr;
        ID3DBlob *                  pVSBlob = nullptr,
                 *                  pVSBlobErrors = nullptr;

        UINT                        vbStride = 0;
    } helperStatesGeneral;
    void constructHelperStates();
    void releaseHelperStates();
    bool areHelperStatesValid(const HelperD3DStates & helperStates);
    HRESULT createHelperStates_noPS();

    class HelperD3DResources
    {
        ID3D11Device *              m_d3dDevice = nullptr;
        bool                        m_copyBackSetup = false;

    public:

        bool isCopyBackSetup() const { return m_copyBackSetup; }

        ID3D11ShaderResourceView *  sourceTextureSRV = nullptr;
        ID3D11RenderTargetView *    sourceTextureRTV = nullptr;
        ID3D11Texture2D *           sourceTexture = nullptr;
        DXGI_FORMAT                 sourceFormat = DXGI_FORMAT_UNKNOWN;

        ID3D11ShaderResourceView *  targetTextureSRV = nullptr;
        ID3D11RenderTargetView *    targetTextureRTV = nullptr;
        ID3D11Texture2D *           targetTexture = nullptr;
        DXGI_FORMAT                 targetFormat = DXGI_FORMAT_UNKNOWN;
        unsigned int                targetWidth = 0;
        unsigned int                targetHeight = 0;
        
        // Multisampling is not supported lwrrently in Helpers
        const unsigned int          targetSampleCount = 1;
        const unsigned int          targetSampleQuality = 0;

        ~HelperD3DResources();

        void setD3DDevice(ID3D11Device * d3dDevice)
        {
            m_d3dDevice = d3dDevice;
        }
        ID3D11Device * getD3DDevice() const
        {
            return m_d3dDevice;
        }

        bool needCreateTextureRTV(unsigned int w, unsigned int h, DXGI_FORMAT tf, unsigned int samCnt = 1, unsigned int samQlt = 0)
        {
            return !targetFormat || !targetTextureRTV  ||
                    (targetWidth != w) || (targetHeight != h) ||
                    (targetFormat != tf) || (targetSampleCount != samCnt) ||
                    (targetSampleQuality != samQlt);
        }

        bool needCreateSRV(ID3D11Texture2D * inTex, DXGI_FORMAT inFormat)
        {
            return (sourceFormat != inFormat) || (sourceTexture != inTex);
        }

        HRESULT createResourcesIfNeeded(const AnselResourceData & inputData, DXGI_FORMAT requestedTargetFormat, bool copyBackNeeded, bool isSourceMultiSampled = false, const DXGI_FORMAT overrideSRVFormat = DXGI_FORMAT_UNKNOWN);

        void deinit();
    };

    void init();
    void deinit();

    HRESULT resolveDepthTexture(const AnselResourceData & inputData);

    HelperD3DResources          m_rgba8ColwersionHelperResources;

    char *                      m_depthResolvePSText = nullptr;
    ID3D11PixelShader *         m_depthResolvePS = nullptr;

    HelperD3DResources          m_depthResolveD3DResources;
    unsigned int                m_depthResolveSampleCount = 0;

    // This function checks if depth texture needs to be resolved, and if it is the case, does the resolve and returns proper D3D objects
    HRESULT getResolvedDepthTexture(const AnselResourceData & inputData, ID3D11Texture2D ** resolvedDepthTexture, DXGI_FORMAT * resolvedDepthFormat);

    ID3D11PixelShader *         m_colwert4ChPS = nullptr;
    // Colwerts between 4-channel formats
    HRESULT colwertTexture4Channel(ID3D11ShaderResourceView * source, ID3D11RenderTargetView * dest, float width, float height, bool usePointSampling = false);
    HRESULT getColwertedTexture4Ch(const AnselResourceData & inputData, float width, float height, DXGI_FORMAT targetFormat, HelperD3DResources * helperResources, ID3D11Texture2D ** outTexture, bool usePointSampling = false);

    ID3D11PixelShader *         m_colwert1ChPS = nullptr;
    // Colwerts (arbitrary channel?) to 1-channel formats
    HRESULT colwertTexture1Channel(ID3D11ShaderResourceView * source, ID3D11RenderTargetView * dest, float width, float height, bool usePointSampling = false);
    HRESULT getColwertedTexture1Ch(const AnselResourceData & inputData, float width, float height, DXGI_FORMAT targetFormat, HelperD3DResources * helperResources, ID3D11Texture2D ** outTexture, bool usePointSampling = false);

    HelperD3DResources          m_hdr32FD3DResources;
    HRESULT getHDR32FTexture(const AnselResourceData & inputData, ID3D11Texture2D ** hdr32FTexture);

    HelperD3DResources          m_hdrLwstomD3DResources;
    HRESULT getHDRLwstomTexture(const AnselResourceData & inputData, bool copyBackNeeded, ID3D11Texture2D ** hdr32FTexture, DXGI_FORMAT targetFormat);
    HRESULT copybackHDRLwstomTexture(DWORD width, DWORD height);

    void setPSConstBufDataViewport(float width, float height) {
        m_psConstBufData.viewportWidth = width;
        m_psConstBufData.viewportHeight = height;
    }

    bool didViewportChange() const {
        return memcmp(&m_psConstBufData, &m_prevPSConstBufData, sizeof(PSConstBufData)) != 0;
    }

    void resetViewport() {
        m_prevPSConstBufData.viewportWidth = 0.0f;
        m_prevPSConstBufData.viewportHeight = 0.0f;
    }

private:
    struct PSConstBufData
    {
        float viewportWidth = 0.0f;
        float viewportHeight = 0.0f;
    };

    PSConstBufData m_psConstBufData;
    PSConstBufData m_prevPSConstBufData;

    void performCopyingDraw(ID3D11PixelShader* pPixelShader, ID3D11ShaderResourceView* source, ID3D11RenderTargetView* dest, float width, float height, bool usePointSampling = false);
};
