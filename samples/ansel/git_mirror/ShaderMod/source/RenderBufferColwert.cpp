#include <map>
#include <vector>
#include <Windows.h>
#include <d3d11.h>

#include "RenderBufferColwert.h"
#include "CommonTools.h"

#define SAFE_RELEASE(x) if (x) (x)->Release(), (x) = nullptr;
#ifdef _DEBUG
#define AnselBufferHandleFailure() __debugbreak(); return status;
#else
#define AnselBufferHandleFailure() return status;
#endif

void AnselBufferColwerter::performCopyingDraw(ID3D11PixelShader *   pPixelShader, ID3D11ShaderResourceView * source, ID3D11RenderTargetView * dest, float width, float height, bool usePointSampling)
{
    // draw call
    const HelperD3DStates & helperStates = helperStatesGeneral;
    m_d3dImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    m_d3dImmediateContext->IASetInputLayout(helperStates.pInputLayout);
    m_d3dImmediateContext->VSSetShader(helperStates.pVS, NULL, 0);

    ID3D11ShaderResourceView* pSRVs[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    ID3D11RenderTargetView * pRTVs[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    ID3D11DepthStencilView * pDSV = NULL;
    m_d3dImmediateContext->OMSetRenderTargets(8, pRTVs, pDSV);
    m_d3dImmediateContext->PSSetShaderResources(0, 16, pSRVs);

    D3D11_VIEWPORT viewPort;
    viewPort.MinDepth = 0.0f;
    viewPort.MaxDepth = 1.0f;
    viewPort.TopLeftX = 0;
    viewPort.TopLeftY = 0;
    viewPort.Width = width;
    viewPort.Height = height;
    m_d3dImmediateContext->RSSetViewports(1, &viewPort);
    m_d3dImmediateContext->RSSetState(helperStates.pRasterizerState);
    m_d3dImmediateContext->PSSetShader(pPixelShader, NULL, 0);
    m_d3dImmediateContext->PSSetShaderResources(0, 1, &source);
    if (usePointSampling)
        m_d3dImmediateContext->PSSetSamplers(0, 1, &helperStates.pPointSamplerState);
    else
        m_d3dImmediateContext->PSSetSamplers(0, 1, &helperStates.pLinearSamplerState);
    m_d3dImmediateContext->OMSetRenderTargets(1, &dest, NULL);
    m_d3dImmediateContext->OMSetDepthStencilState(helperStates.pDepthStencilState, 0xFFFFFFFF);
    m_d3dImmediateContext->OMSetBlendState(helperStates.pBlendState, NULL, 0xffffffff);

    m_d3dImmediateContext->IASetIndexBuffer(helperStates.pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
    const UINT offset = 0;
    m_d3dImmediateContext->IASetVertexBuffers(0, 1, &helperStates.pRectVertexBuf, &helperStates.vbStride, &offset);
    m_d3dImmediateContext->DrawIndexed(6, 0, 0);

    m_d3dImmediateContext->VSSetShader(0, NULL, 0);
    m_d3dImmediateContext->PSSetShader(0, NULL, 0);
}

void AnselBufferColwerter::setRenderer(ID3D11Device *   d3dDevice, ID3D11DeviceContext * d3dImmediateContext, D3DCompilerHandler * d3dCompiler)
{
    m_d3dDevice = d3dDevice;
    m_d3dImmediateContext = d3dImmediateContext;
    m_d3dCompiler = d3dCompiler;

    m_depthResolveD3DResources.setD3DDevice(d3dDevice);
    m_hdr32FD3DResources.setD3DDevice(d3dDevice);
    m_hdrLwstomD3DResources.setD3DDevice(d3dDevice);
    m_rgba8ColwersionHelperResources.setD3DDevice(d3dDevice);
}

void AnselBufferColwerter::constructHelperStates()
{
    helperStatesGeneral.pInputLayout = nullptr;
    helperStatesGeneral.pVS = nullptr;
    helperStatesGeneral.pRasterizerState = nullptr;
    helperStatesGeneral.pLinearSamplerState = nullptr;
    helperStatesGeneral.pPointSamplerState = nullptr;
    helperStatesGeneral.pBlendState = nullptr;
    helperStatesGeneral.pDepthStencilState = nullptr;
    helperStatesGeneral.pConstBuf = nullptr;
    helperStatesGeneral.pRectVertexBuf = nullptr;
    helperStatesGeneral.pRectIndexBuf = nullptr;
    helperStatesGeneral.pVSBlob = nullptr;
    helperStatesGeneral.pVSBlobErrors = nullptr;
    helperStatesGeneral.vbStride = 0;
}
void AnselBufferColwerter::releaseHelperStates()
{
    SAFE_RELEASE(helperStatesGeneral.pInputLayout);
    SAFE_RELEASE(helperStatesGeneral.pVS);
    SAFE_RELEASE(helperStatesGeneral.pRasterizerState);
    SAFE_RELEASE(helperStatesGeneral.pLinearSamplerState);
    SAFE_RELEASE(helperStatesGeneral.pPointSamplerState);
    SAFE_RELEASE(helperStatesGeneral.pBlendState);
    SAFE_RELEASE(helperStatesGeneral.pDepthStencilState);
    SAFE_RELEASE(helperStatesGeneral.pConstBuf);
    SAFE_RELEASE(helperStatesGeneral.pRectVertexBuf);
    SAFE_RELEASE(helperStatesGeneral.pRectIndexBuf);
    SAFE_RELEASE(helperStatesGeneral.pVSBlob);
    SAFE_RELEASE(helperStatesGeneral.pVSBlobErrors);
}
bool AnselBufferColwerter::areHelperStatesValid(const HelperD3DStates & helperStates)
{
    // TODO avoroshilov: make helperStates a class, and make this a member function
    return (helperStates.pInputLayout &&
            helperStates.pVS &&
            helperStates.pRasterizerState &&
            (helperStates.pLinearSamplerState || helperStatesGeneral.pPointSamplerState) &&
            helperStates.pBlendState &&
            helperStates.pDepthStencilState &&
            helperStates.pConstBuf &&
            helperStates.pRectVertexBuf &&
            helperStates.pRectIndexBuf &&
            helperStates.pVSBlob);
}
HRESULT AnselBufferColwerter::createHelperStates_noPS()
{
    HRESULT status = S_OK;

    releaseHelperStates();

    D3D11_INPUT_ELEMENT_DESC inputLayoutDesc[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };

    D3D11_RASTERIZER_DESC rastStateDesc =
    {
        D3D11_FILL_SOLID,          //FillMode;
        D3D11_LWLL_NONE,           //LwllMode;
        FALSE,                     //FrontCounterClockwise;
        0,                         //DepthBias;
        0.0f,                      //DepthBiasClamp;
        0.0f,                      //SlopeScaledDepthBias;
        TRUE,                      //DepthClipEnable;
        FALSE,                     //ScissorEnable;
        FALSE,                     //MultisampleEnable;
        FALSE                      //AntialiasedLineEnable;
    };

    D3D11_BLEND_DESC blendStateDesc;
    memset(&blendStateDesc, 0, sizeof(blendStateDesc));
    blendStateDesc.AlphaToCoverageEnable = FALSE;
    blendStateDesc.IndependentBlendEnable = FALSE;
    blendStateDesc.RenderTarget[0].BlendEnable = FALSE;
    blendStateDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_COLOR;
    blendStateDesc.RenderTarget[0].DestBlend = D3D11_BLEND_DEST_COLOR;
    blendStateDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blendStateDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_SRC_ALPHA;
    blendStateDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ILW_SRC_ALPHA;
    blendStateDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blendStateDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

    D3D11_DEPTH_STENCIL_DESC dsStateDesc;
    memset(&dsStateDesc, 0, sizeof(dsStateDesc));
    dsStateDesc.DepthEnable = FALSE;
    dsStateDesc.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ALL;
    dsStateDesc.DepthFunc = D3D11_COMPARISON_ALWAYS;
    dsStateDesc.StencilEnable = FALSE;
    dsStateDesc.StencilReadMask = 0xFF;
    dsStateDesc.StencilWriteMask = 0xFF;
    dsStateDesc.FrontFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.FrontFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.FrontFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.FrontFace.StencilFunc = D3D11_COMPARISON_ALWAYS;
    dsStateDesc.BackFace.StencilFailOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.BackFace.StencilDepthFailOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.BackFace.StencilPassOp = D3D11_STENCIL_OP_KEEP;
    dsStateDesc.BackFace.StencilFunc = D3D11_COMPARISON_ALWAYS;


    const unsigned int vertsPerQuad = 4;
    const unsigned int indsPerQuad = 6;
    unsigned int inds[indsPerQuad] = { 0, 1, 2, 2, 3, 0 };

    D3D11_BUFFER_DESC vertexBufferDesc, indexBufferDesc, constBufDesc;
    D3D11_SUBRESOURCE_DATA vertexData, indexData, constBufData;

    // Indexes
    indexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
    indexBufferDesc.ByteWidth = sizeof(unsigned int) * indsPerQuad;
    indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
    indexBufferDesc.CPUAccessFlags = 0;
    indexBufferDesc.MiscFlags = 0;
    indexBufferDesc.StructureByteStride = 0;

    indexData.pSysMem = inds;
    indexData.SysMemPitch = 0;
    indexData.SysMemSlicePitch = 0;

    // Verts
    struct VSInput
    {
        float position[4];
        float texcoord[2];
    };

    vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
    vertexBufferDesc.ByteWidth = sizeof(VSInput) * vertsPerQuad;
    vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
    vertexBufferDesc.CPUAccessFlags = 0;
    vertexBufferDesc.MiscFlags = 0;
    vertexBufferDesc.StructureByteStride = 0;

    // TODO avoroshilov: replace this shader with the vertex-only one FS triangle, probably
    VSInput verts[vertsPerQuad] =
    {
        { { -1.0f, -1.0f, 0.0f, 1.0f },{ 0.0f, 1.0f } },
        { {  1.0f, -1.0f, 0.0f, 1.0f },{ 1.0f, 1.0f } },
        { {  1.0f,  1.0f, 0.0f, 1.0f },{ 1.0f, 0.0f } },
        { { -1.0f,  1.0f, 0.0f, 1.0f },{ 0.0f, 0.0f } }
    };

    // Ordinary
    vertexData.pSysMem = verts;
    vertexData.SysMemPitch = 0;
    vertexData.SysMemSlicePitch = 0;

    // create new render target here, do not use readback as it's staging

    // Constant Buffers
    ZeroMemory(&constBufDesc, sizeof(constBufDesc));
    constBufDesc.Usage = D3D11_USAGE_DYNAMIC;
    constBufDesc.ByteWidth = (sizeof(PSConstBufData) + 15) & ~15; // Constant Buffers must be aligned to multiples of 16
    constBufDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    constBufDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    constBufDesc.MiscFlags = 0;
    constBufDesc.StructureByteStride = 0;
    
    constBufData.pSysMem = &m_psConstBufData;
    constBufData.SysMemPitch = 0;
    constBufData.SysMemSlicePitch = 0;

    // Vertex Shader
    const char* vsText =
        "struct VSIn                                                                                    \n"
        "{                                                                                              \n"
        "   float4 position : POSITION;                                                                 \n"
        "   float2 texcoord : TEXCOORD;                                                                 \n"
        "};                                                                                             \n"
        "                                                                                               \n"
        "struct VSOut                                                                                   \n"
        "{                                                                                              \n"
        "   float4 position : SV_POSITION;                                                              \n"
        "   float2 texcoord : TEXCOORD;                                                                 \n"
        "};                                                                                             \n"
        "                                                                                               \n"
        "VSOut VS( VSIn vertex )                                                                        \n"
        "{                                                                                              \n"
        "   VSOut output;                                                                               \n"
        "                                                                                               \n"
        "   output.position = vertex.position;                                                          \n"
        "   output.texcoord = vertex.texcoord;                                                          \n"
        "                                                                                               \n"
        "   return output;                                                                              \n"
        "}                                                                                              \n";

    HelperD3DStates & helperStates = helperStatesGeneral;

    if (!SUCCEEDED(status = m_d3dDevice->CreateBuffer(&constBufDesc, &constBufData, &helperStates.pConstBuf)))
    {
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Helper state const.buffer creation failed");
        AnselBufferHandleFailure();
    }
    if (!SUCCEEDED(status = m_d3dDevice->CreateBuffer(&vertexBufferDesc, &vertexData, &helperStates.pRectVertexBuf)))
    {
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Helper state v.buffer creation failed");
        AnselBufferHandleFailure();
    }
    if (!SUCCEEDED(status = m_d3dDevice->CreateBuffer(&indexBufferDesc, &indexData, &helperStates.pRectIndexBuf)))
    {
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Helper state i.buffer creation failed");
        AnselBufferHandleFailure();
    }

    auto d3dCompileFunc = m_d3dCompiler->getD3DCompileFunc();
    if (!SUCCEEDED(status = d3dCompileFunc(vsText, strlen(vsText), NULL, NULL, NULL, "VS", "vs_4_0", 0, 0, &helperStates.pVSBlob, &helperStates.pVSBlobErrors)))
    {
        char * error = (char *)helperStates.pVSBlobErrors->GetBufferPointer();
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Helper state v.shader compilation failed: %s", error);
        AnselBufferHandleFailure();
    }
    if (!SUCCEEDED(status = m_d3dDevice->CreateVertexShader(helperStates.pVSBlob->GetBufferPointer(), helperStates.pVSBlob->GetBufferSize(), NULL, &helperStates.pVS)))
    {
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Helper state v.shader creation failed");
        AnselBufferHandleFailure();
    }
    if (!SUCCEEDED(status = m_d3dDevice->CreateInputLayout(inputLayoutDesc, 2, helperStates.pVSBlob->GetBufferPointer(), helperStates.pVSBlob->GetBufferSize(), &helperStates.pInputLayout)))
    {
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Helper state input layout creation failed");
        AnselBufferHandleFailure();
    }
    if (!SUCCEEDED(status = m_d3dDevice->CreateRasterizerState(&rastStateDesc, &helperStates.pRasterizerState)))
    {
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Helper state rasterizer state initialization failed");
        AnselBufferHandleFailure();
    }
    if (!SUCCEEDED(status = m_d3dDevice->CreateBlendState(&blendStateDesc, &helperStates.pBlendState)))
    {
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Helper state blend state initialization failed");
        AnselBufferHandleFailure();
    }
    if (!SUCCEEDED(status = m_d3dDevice->CreateDepthStencilState(&dsStateDesc, &helperStates.pDepthStencilState)))
    {
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Helper state depth/stencil state initialization failed");
        AnselBufferHandleFailure();
    }

    D3D11_SAMPLER_DESC samplerState;
    memset(&samplerState, 0, sizeof(samplerState));
    samplerState.Filter = D3D11_FILTER_MIN_MAG_POINT_MIP_LINEAR;
    samplerState.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerState.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerState.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    samplerState.MipLODBias = 0;
    samplerState.MaxAnisotropy = 1;
    samplerState.ComparisonFunc = D3D11_COMPARISON_EQUAL;
    samplerState.BorderColor[0] = 0.0f;
    samplerState.BorderColor[1] = 0.0f;
    samplerState.BorderColor[2] = 0.0f;
    samplerState.BorderColor[3] = 0.0f;
    samplerState.MinLOD = 0;
    samplerState.MaxLOD = 0;

    if (!SUCCEEDED(status = m_d3dDevice->CreateSamplerState(&samplerState, &helperStates.pPointSamplerState)))
    {
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Helper state point sampler state initialization failed");
        AnselBufferHandleFailure();
    }

    samplerState.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;

    if (!SUCCEEDED(status = m_d3dDevice->CreateSamplerState(&samplerState, &helperStates.pLinearSamplerState)))
    {
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Helper state linear sampler state initialization failed");
        AnselBufferHandleFailure();
    }

    helperStates.vbStride = sizeof(VSInput);

    return status;
}

AnselBufferColwerter::HelperD3DResources::~HelperD3DResources()
{
    deinit();
}

HRESULT AnselBufferColwerter::HelperD3DResources::createResourcesIfNeeded(const AnselResourceData & inputData, DXGI_FORMAT requestedTargetFormat, bool copyBackNeeded, bool isSourceMultiSampled, const DXGI_FORMAT overrideSRVFormat)
{
    HRESULT status = S_OK;
    bool texCreationNeeded = needCreateTextureRTV(inputData.width, inputData.height, requestedTargetFormat);

    if (texCreationNeeded)
    {
        SAFE_RELEASE(targetTextureSRV);
        SAFE_RELEASE(targetTextureRTV);
        SAFE_RELEASE(targetTexture);

        m_copyBackSetup = false;
    }

    if (needCreateSRV(inputData.pTexture2D, inputData.format))
    {
        SAFE_RELEASE(sourceTextureSRV);
        SAFE_RELEASE(sourceTextureRTV);

        m_copyBackSetup = false;
    }

    if (texCreationNeeded)
    {
        D3D11_TEXTURE2D_DESC textureDesc;
        // Initialize the render target texture description.
        ZeroMemory(&textureDesc, sizeof(textureDesc));

        textureDesc.MipLevels = 1;
        textureDesc.ArraySize = 1;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.SampleDesc.Quality = 0;
        textureDesc.MiscFlags = 0;

        textureDesc.Usage = D3D11_USAGE_DEFAULT;
        textureDesc.CPUAccessFlags = 0;

        textureDesc.Width = inputData.width;
        textureDesc.Height = inputData.height;
        textureDesc.Format = requestedTargetFormat;
        textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET | D3D11_BIND_UNORDERED_ACCESS;

        if (!SUCCEEDED(status = shadermod::Tools::CreateTexture2D(m_d3dDevice, &textureDesc, NULL, &targetTexture)))
        {
            LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Failed to create D3D colwerter helper resource texture, status: %d", HRESULT_CODE(status));
            // This is not a fatal error
            AnselBufferHandleFailure();
        }

        targetWidth = textureDesc.Width;
        targetHeight = textureDesc.Height;
        targetFormat = textureDesc.Format;

        D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;
        memset(&rtvDesc, 0, sizeof(rtvDesc));
        rtvDesc.Format = textureDesc.Format;
        rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        rtvDesc.Texture2D.MipSlice = 0;

        if (!SUCCEEDED(status = m_d3dDevice->CreateRenderTargetView(targetTexture, &rtvDesc, &targetTextureRTV)))
        {
            LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Failed to create D3D colwerter helper texture render target");
            // This is not a fatal error
            AnselBufferHandleFailure();
        }

        if (copyBackNeeded)
        {
            D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
            memset(&srvDesc, 0, sizeof(srvDesc));
            srvDesc.Format = targetFormat;

            // see above, textureDesc.SampleDesc.Count = 1;
            srvDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            if (!isSourceMultiSampled)
            {
                srvDesc.Texture2D.MostDetailedMip = 0;
                srvDesc.Texture2D.MipLevels = 1;
            }

            if (!SUCCEEDED(status = m_d3dDevice->CreateShaderResourceView(targetTexture, &srvDesc, &targetTextureSRV)))
            {
                LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "D3D colwerter helper target SRV creation failed");
                // This is not a fatal error
                AnselBufferHandleFailure();
            }
        }
    }

    if (!sourceTextureSRV)
    {
        sourceFormat = inputData.format;

        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc;
        memset(&srvDesc, 0, sizeof(srvDesc));
        srvDesc.Format = sourceFormat;

        if (overrideSRVFormat != DXGI_FORMAT_UNKNOWN)
        {
            srvDesc.Format = overrideSRVFormat;
        }

        srvDesc.ViewDimension = isSourceMultiSampled ? D3D11_SRV_DIMENSION_TEXTURE2DMS : D3D11_SRV_DIMENSION_TEXTURE2D;
        if (!isSourceMultiSampled)
        {
            srvDesc.Texture2D.MostDetailedMip = 0;
            srvDesc.Texture2D.MipLevels = 1;
        }

        if (!SUCCEEDED(status = m_d3dDevice->CreateShaderResourceView(inputData.pTexture2D, &srvDesc, &sourceTextureSRV)))
        {
            LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "D3D colwerter helper SRV creation failed");
            // This is not a fatal error
            AnselBufferHandleFailure();
        }

        sourceTexture = inputData.pTexture2D;

        if (copyBackNeeded)
        {
            D3D11_RENDER_TARGET_VIEW_DESC rtvDesc;
            memset(&rtvDesc, 0, sizeof(rtvDesc));
            rtvDesc.Format = srvDesc.Format;
            rtvDesc.ViewDimension = isSourceMultiSampled ? D3D11_RTV_DIMENSION_TEXTURE2DMS : D3D11_RTV_DIMENSION_TEXTURE2D;
            rtvDesc.Texture2D.MipSlice = 0;

            if (!SUCCEEDED(status = m_d3dDevice->CreateRenderTargetView(sourceTexture, &rtvDesc, &sourceTextureRTV)))
            {
                LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Failed to create D3D colwerter helper source texture render target");
                // This is not a fatal error
                AnselBufferHandleFailure();
            }
        }
    }

    if (copyBackNeeded)
        m_copyBackSetup = true;

    return S_OK;
}

void AnselBufferColwerter::HelperD3DResources::deinit()
{
    SAFE_RELEASE(targetTextureSRV);
    SAFE_RELEASE(targetTextureRTV);
    SAFE_RELEASE(targetTexture);
    SAFE_RELEASE(sourceTextureSRV);
    SAFE_RELEASE(sourceTextureRTV);
}

void AnselBufferColwerter::init()
{
    constructHelperStates();
}
void AnselBufferColwerter::deinit()
{
    releaseHelperStates();

    m_depthResolveD3DResources.deinit();
    m_rgba8ColwersionHelperResources.deinit();

    SAFE_RELEASE(m_colwert4ChPS);
    SAFE_RELEASE(m_colwert1ChPS);

    m_hdr32FD3DResources.deinit();
    m_hdrLwstomD3DResources.deinit();
}

HRESULT AnselBufferColwerter::resolveDepthTexture(const AnselResourceData & inputData)
{
    HRESULT status = S_OK;

    if (!areHelperStatesValid(helperStatesGeneral))
    {
        createHelperStates_noPS();
    }

    // Unfortunately, multi-sampled texture in HLSL shader requires explicit sample count
    //  so we need to recreate pixel shader each time the amount of samples has changed
    static const char* psTextTemplate =     // Pixel Shader
        "struct VSOut\n"
        "{\n"
        "   float4 position : SV_POSITION;\n"
        "   float2 texcoord : TEXCOORD;\n"
        "};\n"
        \
        "cbuffer ControlBuf : register(b0)\n"
        "{\n"
        "   float2 g_Viewport: packoffset(c0);\n"
        "}\n"
        \
        "Texture2DMS<float2, %d> depthMS : register( t0 );\n"
        \
        "float PS( VSOut frag ): SV_Target\n"
        "{\n"
        "   uint sizeX, sizeY, numSamples;\n"
        "   depthMS.GetDimensions(sizeX, sizeY, numSamples);\n"
        "   return depthMS.Load(frag.texcoord * g_Viewport, 0).r;\n"
        "}\n";
    static const size_t psTextTemplateLen = strlen(psTextTemplate);
    static const size_t psTextLen = psTextTemplateLen + 10;

    if (!m_depthResolvePSText)
    {
        m_depthResolvePSText = new char [psTextLen];
    }

    if (inputData.sampleCount != m_depthResolveSampleCount)
    {
        m_depthResolveSampleCount = inputData.sampleCount;
        sprintf_s(m_depthResolvePSText, psTextLen * sizeof(char), psTextTemplate, m_depthResolveSampleCount);

        SAFE_RELEASE(m_depthResolvePS);
    }

    auto d3dCompileFunc = m_d3dCompiler->getD3DCompileFunc();
    if (!m_depthResolvePS)
    {
        ID3DBlob *pPSBlob = nullptr, *pPSBlobErrors = nullptr;

        if (!SUCCEEDED(status = d3dCompileFunc(m_depthResolvePSText, strlen(m_depthResolvePSText), NULL, NULL, NULL, "PS", "ps_4_0", 0, 0, &pPSBlob, &pPSBlobErrors)))
        {
            char * error = (char *)pPSBlobErrors->GetBufferPointer();
            OutputDebugStringA(error);
            LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Depth resolve p.shader compilation failed [sample count %d]: %s", m_depthResolveSampleCount, error);
            AnselBufferHandleFailure();
        }
        if (!SUCCEEDED(status = m_d3dDevice->CreatePixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, &m_depthResolvePS)))
        {
            LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Depth resolve p.shader creation failed [sample count %d]", m_depthResolveSampleCount);
            AnselBufferHandleFailure();
        }

        SAFE_RELEASE(pPSBlob);
        SAFE_RELEASE(pPSBlobErrors);
    }

    UINT viewPortsNum = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
    D3D11_VIEWPORT viewPorts[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
    m_d3dImmediateContext->RSGetViewports(&viewPortsNum, viewPorts);

    D3D11_VIEWPORT depthViewPort;
    memcpy(&depthViewPort, viewPorts, sizeof(D3D11_VIEWPORT));
    depthViewPort.Width = static_cast<float>(inputData.width);
    depthViewPort.Height = static_cast<float>(inputData.height);

    m_d3dImmediateContext->RSSetViewports(1, &depthViewPort);

    const HelperD3DStates& helperStates = helperStatesGeneral;

    // In case the constant buffer viewport data has not be set, use the entire texture size as the viewport
    if (m_psConstBufData.viewportWidth == 0 || m_psConstBufData.viewportHeight == 0)
    {
        m_psConstBufData.viewportWidth = depthViewPort.Width;
        m_psConstBufData.viewportHeight = depthViewPort.Height;
    }

    // In case of Dynamic Resolution Scaling, it's possible that the depth was only written to a portion of the
    // texture buffer. Therefore we need to tell the shader how much of the depth texture buffer should be
    // used for rendering
    if (didViewportChange())
    {
        D3D11_MAPPED_SUBRESOURCE subResource;
        m_d3dImmediateContext->Map(helperStates.pConstBuf, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
        memcpy(subResource.pData, &m_psConstBufData, sizeof(PSConstBufData));
        m_d3dImmediateContext->Unmap(helperStates.pConstBuf, 0);
        m_d3dImmediateContext->VSSetConstantBuffers(0, 1, &helperStates.pConstBuf);

        m_prevPSConstBufData = m_psConstBufData;
    }

    // draw call
    m_d3dImmediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    m_d3dImmediateContext->IASetInputLayout(helperStates.pInputLayout);
    m_d3dImmediateContext->VSSetShader(helperStates.pVS, NULL, 0);
    m_d3dImmediateContext->RSSetState(helperStates.pRasterizerState);
    m_d3dImmediateContext->PSSetShader(m_depthResolvePS, NULL, 0);
    m_d3dImmediateContext->PSSetShaderResources(0, 1, &m_depthResolveD3DResources.sourceTextureSRV);
    m_d3dImmediateContext->PSSetConstantBuffers(0, 1, &helperStates.pConstBuf);
    m_d3dImmediateContext->OMSetRenderTargets(1, &m_depthResolveD3DResources.targetTextureRTV, NULL);
    m_d3dImmediateContext->OMSetDepthStencilState(helperStates.pDepthStencilState, 0xFFFFFFFF);
    m_d3dImmediateContext->OMSetBlendState(helperStates.pBlendState, NULL, 0xffffffff);

    m_d3dImmediateContext->IASetIndexBuffer(helperStates.pRectIndexBuf, DXGI_FORMAT_R32_UINT, 0);
    const UINT offset = 0;
    m_d3dImmediateContext->IASetVertexBuffers(0, 1, &helperStates.pRectVertexBuf, &helperStates.vbStride, &offset);
    m_d3dImmediateContext->DrawIndexed(6, 0, 0);

    m_d3dImmediateContext->VSSetShader(0, NULL, 0);
    m_d3dImmediateContext->PSSetShader(0, NULL, 0);

    m_d3dImmediateContext->RSSetViewports(viewPortsNum, viewPorts);

    return status;
}


HRESULT AnselBufferColwerter::getResolvedDepthTexture(const AnselResourceData & inputData, ID3D11Texture2D ** resolvedDepthTexture, DXGI_FORMAT * resolvedDepthFormat)
{
    HRESULT status = S_OK;

    if (!resolvedDepthTexture || !resolvedDepthFormat)
    {
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "NULL pointers for resolvedDepthTexture or resolvedDepthFormat");
        return S_OK;
    }

    // If depth resolve failed, we cannot return anything
    *resolvedDepthTexture = nullptr;
    *resolvedDepthFormat = DXGI_FORMAT_UNKNOWN;

    // Perform depth resolve, if needed
    //  1. If app uses MSAA and doesn't do custom resolve, we get MSAA-depth, which cannot be directly used in the shader
    //  (requires special treatment of Texture2DMS etc.) - so we need to create an intermediate texture where we "resolve"
    //  the MSAA-depth - basically just getting sub-sample 0 from each fragment
    // 2. If the app is rendering depth to only a viewport within the frame (ex. due to Dynamic Resolutino Scaling)
    // 3. The viewport data of this frame is different than the last time depth was resolved

    if (inputData.sampleCount > 1 ||
        inputData.width != m_psConstBufData.viewportWidth || inputData.height != m_psConstBufData.viewportHeight ||
        didViewportChange())
    {
        DXGI_FORMAT targetFormat = inputData.format;
        if (targetFormat == DXGI_FORMAT_D24_UNORM_S8_UINT || targetFormat == DXGI_FORMAT_R32_TYPELESS ||
            targetFormat == DXGI_FORMAT_D32_FLOAT || targetFormat == DXGI_FORMAT_D32_FLOAT_S8X24_UINT)
        {
            targetFormat = DXGI_FORMAT_R32_FLOAT;
        }
        else if (targetFormat == DXGI_FORMAT_D16_UNORM)
        {
            targetFormat = DXGI_FORMAT_R16_FLOAT;
        }

        DXGI_FORMAT overrideSRVFormat = inputData.format;
        switch (inputData.format)
        {
            case DXGI_FORMAT_R32_TYPELESS:
            case DXGI_FORMAT_R32_FLOAT:
            case DXGI_FORMAT_D32_FLOAT:
                overrideSRVFormat = DXGI_FORMAT_R32_FLOAT;
                break;

            case DXGI_FORMAT_D16_UNORM:
            case DXGI_FORMAT_R16_FLOAT:
                overrideSRVFormat = DXGI_FORMAT_R16_FLOAT;
                break;

            case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:
                overrideSRVFormat = DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
                break;

            case DXGI_FORMAT_D24_UNORM_S8_UINT:
            default:
                overrideSRVFormat = DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
                break;
        };

        if (!SUCCEEDED( status = m_depthResolveD3DResources.createResourcesIfNeeded(inputData, targetFormat, false, true, overrideSRVFormat) ))
        {
            LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Failed to create depth resolve helper resources");
            // This is not a fatal error, we can continue functioning without depth buffer
            return S_OK;
        }

        if (!SUCCEEDED( status = resolveDepthTexture(inputData) ))
        {
            LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Failed to resolve depth buffer");
            // This is not a fatal error, we can continue functioning without depth buffer
            return S_OK;
        }

        // Resolve was successful, we can return the resolved texture
        *resolvedDepthTexture = m_depthResolveD3DResources.targetTexture;
        *resolvedDepthFormat = m_depthResolveD3DResources.targetFormat;
    }
    else
    {
        // No resolve needed
        *resolvedDepthTexture = inputData.pTexture2D;
        *resolvedDepthFormat = inputData.format;
    }

    return S_OK;
}

HRESULT AnselBufferColwerter::colwertTexture4Channel(ID3D11ShaderResourceView * source, ID3D11RenderTargetView * dest, float width, float height, bool usePointSampling)
{
    HRESULT status = S_OK;

    if (!areHelperStatesValid(helperStatesGeneral))
    {
        createHelperStates_noPS();
    }

    if (!m_colwert4ChPS)
    {
        ID3DBlob *pPSBlob = nullptr, *pPSBlobErrors = nullptr;

        const char* psText =    // Pixel Shader
            "struct VSOut\n"
            "{\n"
            "   float4 position : SV_POSITION;\n"
            "   float2 texcoord : TEXCOORD;\n"
            "};\n"
            "\n"
            "Texture2D input : register( t0 );\n"
            "SamplerState inSampler : register( s0 );\n"
            "\n"
            "float4 PS( VSOut frag ): SV_Target\n"
            "{\n"
            "   return input.Sample(inSampler, frag.texcoord);\n"
            "}\n";

        auto d3dCompileFunc = m_d3dCompiler->getD3DCompileFunc();
        if (!SUCCEEDED(status = d3dCompileFunc(psText, strlen(psText), NULL, NULL, NULL, "PS", "ps_4_0", 0, 0, &pPSBlob, &pPSBlobErrors)))
        {
            char * error = (char *)pPSBlobErrors->GetBufferPointer();
            LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Tex4ch colwersion p.shader compilation failed: %s", error);
            AnselBufferHandleFailure();
        }
        if (!SUCCEEDED(status = m_d3dDevice->CreatePixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, &m_colwert4ChPS)))
        {
            LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Tex4ch colwersion p.shader creation failed");
            AnselBufferHandleFailure();
        }

        SAFE_RELEASE(pPSBlob);
        SAFE_RELEASE(pPSBlobErrors);
    }

    performCopyingDraw(m_colwert4ChPS, source, dest, width, height, usePointSampling);

    ID3D11ShaderResourceView* pSRVs[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    ID3D11RenderTargetView * pRTVs[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    ID3D11DepthStencilView * pDSV = NULL;
    m_d3dImmediateContext->OMSetRenderTargets(8, pRTVs, pDSV);
    m_d3dImmediateContext->PSSetShaderResources(0, 16, pSRVs);

    return status;
}
// TODO avorshilov: probably unify with 1Ch, and decide which route to take by the format
HRESULT AnselBufferColwerter::getColwertedTexture4Ch(const AnselResourceData & inputData, float width, float height, DXGI_FORMAT targetFormat, HelperD3DResources * helperResources, ID3D11Texture2D ** outTexture, bool usePointSampling)
{
    HRESULT status = S_OK;
    *outTexture = nullptr;

    // TODO avoroshilov: probably create local helkper resources and release, if needed
    if (!helperResources)
        return E_ILWALIDARG;

    if (helperResources->getD3DDevice() != m_d3dDevice)
    {
        helperResources->deinit();
        helperResources->setD3DDevice(m_d3dDevice);
    }
    if (!SUCCEEDED( status = helperResources->createResourcesIfNeeded(inputData, targetFormat, false) ))
    {
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Failed to create texture colwersion helper resources");
        // This is not a fatal error
        AnselBufferHandleFailure();
    }

    if (!SUCCEEDED( status = colwertTexture4Channel(helperResources->sourceTextureSRV, helperResources->targetTextureRTV, (float)inputData.width, (float)inputData.height, usePointSampling) ))
    {
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Texture colwersion failed");
        // This is not a fatal error
        AnselBufferHandleFailure();
    }

    *outTexture = helperResources->targetTexture;

    return status;
}

HRESULT AnselBufferColwerter::colwertTexture1Channel(ID3D11ShaderResourceView * source, ID3D11RenderTargetView * dest, float width, float height, bool usePointSampling)
{
    HRESULT status = S_OK;

    if (!areHelperStatesValid(helperStatesGeneral))
    {
        createHelperStates_noPS();
    }

    if (!m_colwert1ChPS)
    {
        ID3DBlob *pPSBlob = nullptr, *pPSBlobErrors = nullptr;

        const char* psText =    // Pixel Shader
            "struct VSOut\n"
            "{\n"
            "   float4 position : SV_POSITION;\n"
            "   float2 texcoord : TEXCOORD;\n"
            "};\n"
            "\n"
            "Texture2D input : register( t0 );\n"
            "SamplerState inSampler : register( s0 );\n"
            "\n"
            "float PS( VSOut frag ): SV_Target\n"
            "{\n"
            "   return input.Sample(inSampler, frag.texcoord);\n"
            "}\n";

        auto d3dCompileFunc = m_d3dCompiler->getD3DCompileFunc();
        if (!SUCCEEDED(status = d3dCompileFunc(psText, strlen(psText), NULL, NULL, NULL, "PS", "ps_4_0", 0, 0, &pPSBlob, &pPSBlobErrors)))
        {
            char * error = (char *)pPSBlobErrors->GetBufferPointer();
            LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Tex1ch colwersion p.shader compilation failed: %s", error);
            AnselBufferHandleFailure();
        }
        if (!SUCCEEDED(status = m_d3dDevice->CreatePixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, &m_colwert1ChPS)))
        {
            LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Tex1ch colwersion p.shader creation failed");
            AnselBufferHandleFailure();
        }

        SAFE_RELEASE(pPSBlob);
        SAFE_RELEASE(pPSBlobErrors);
    }

    performCopyingDraw(m_colwert1ChPS, source, dest, width, height, usePointSampling);

    return status;
}
HRESULT AnselBufferColwerter::getColwertedTexture1Ch(const AnselResourceData & inputData, float width, float height, DXGI_FORMAT targetFormat, HelperD3DResources * helperResources, ID3D11Texture2D ** outTexture, bool usePointSampling)
{
    HRESULT status = S_OK;
    *outTexture = nullptr;

    if (!helperResources)
        return E_ILWALIDARG;

    if (!SUCCEEDED( status = helperResources->createResourcesIfNeeded(inputData, targetFormat, false) ))
    {
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Failed to create texture colwersion helper resources");
        // This is not a fatal error
        AnselBufferHandleFailure();
    }

    if (!SUCCEEDED( status = colwertTexture1Channel(helperResources->sourceTextureSRV, helperResources->targetTextureRTV, (float)inputData.width, (float)inputData.height, usePointSampling) ))
    {
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Texture colwersion failed");
        // This is not a fatal error
        AnselBufferHandleFailure();
    }

    *outTexture = helperResources->targetTexture;

    return status;
}

HRESULT AnselBufferColwerter::getHDR32FTexture(const AnselResourceData & inputData, ID3D11Texture2D ** hdr32FTexture)
{
    HRESULT status = S_OK;
    *hdr32FTexture = nullptr;

    const DXGI_FORMAT targetFormat = DXGI_FORMAT_R32G32B32A32_FLOAT;
    if (inputData.format != targetFormat)
    {
        if (!SUCCEEDED( status = m_hdr32FD3DResources.createResourcesIfNeeded(inputData, targetFormat, false) ))
        {
            LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Failed to create HDR32F colwersion helper resources");
            // This is not a fatal error
            AnselBufferHandleFailure();
        }

        if (!SUCCEEDED( status = colwertTexture4Channel(m_hdr32FD3DResources.sourceTextureSRV, m_hdr32FD3DResources.targetTextureRTV, (float)inputData.width, (float)inputData.height) ))
        {
            LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "HDR colwersion failed");
            // This is not a fatal error
            AnselBufferHandleFailure();
        }
        *hdr32FTexture = m_hdr32FD3DResources.targetTexture;
    }
    else
    {
        *hdr32FTexture = inputData.pTexture2D;
    }

    return status;
}

HRESULT AnselBufferColwerter::getHDRLwstomTexture(const AnselResourceData & inputData, bool copyBackNeeded, ID3D11Texture2D ** hdrTexture, DXGI_FORMAT targetFormat)
{
    HRESULT status = S_OK;
    *hdrTexture = nullptr;

    if (inputData.format != targetFormat)
    {
        if (!SUCCEEDED(status = m_hdrLwstomD3DResources.createResourcesIfNeeded(inputData, targetFormat, copyBackNeeded)))
        {
            LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "Failed to create HDRLwstom colwersion helper resources");
            // This is not a fatal error
            AnselBufferHandleFailure();
        }

        if (!SUCCEEDED(status = colwertTexture4Channel(m_hdrLwstomD3DResources.sourceTextureSRV, m_hdrLwstomD3DResources.targetTextureRTV, (float)inputData.width, (float)inputData.height)))
        {
            LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "HDR colwersion failed");
            // This is not a fatal error
            AnselBufferHandleFailure();
        }
        *hdrTexture = m_hdrLwstomD3DResources.targetTexture;
    }
    else
    {
        *hdrTexture = inputData.pTexture2D;
    }

    return status;
}

HRESULT AnselBufferColwerter::copybackHDRLwstomTexture(DWORD width, DWORD height)
{
    HRESULT status = S_OK;

    if (!m_hdrLwstomD3DResources.isCopyBackSetup())
    {
        status = E_FAIL;
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "HDR copyback was not setup properly");
        // This is not a fatal error
        AnselBufferHandleFailure();
    }

    ID3D11ShaderResourceView* pSRVs[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    ID3D11RenderTargetView * pRTVs[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    ID3D11DepthStencilView * pDSV = NULL;
    m_d3dImmediateContext->OMSetRenderTargets(8, pRTVs, pDSV);
    m_d3dImmediateContext->PSSetShaderResources(0, 16, pSRVs);

    if (!SUCCEEDED(status = colwertTexture4Channel(m_hdrLwstomD3DResources.targetTextureSRV, m_hdrLwstomD3DResources.sourceTextureRTV, (float)width, (float)height)))
    {
        LOG_ERROR(LogChannel::kRenderBuffer_Colwerter, "HDR colwersion failed");
        // This is not a fatal error
        AnselBufferHandleFailure();
    }

    return status;
}

#undef SAFE_RELEASE
#undef AnselBufferHandleFailure
