#include "DebugRenderer.h"
#include "CommonTools.h"

#include <algorithm>

#ifdef _DEBUG
#define HandleFailure() __debugbreak(); return status;
#else
#define HandleFailure() return status;
#endif

#define SAFE_RELEASE(x) if (x) x->Release(), x = nullptr;

HRESULT DebugRenderer::init(ID3D11Device * d3dDevice, ID3D11DeviceContext * immediateContext, PFND3DCOMPILEFUNC d3dCompileFunc)
{
    HRESULT status = S_OK;

    m_d3dDevice = d3dDevice;
    m_immediateContext = immediateContext;
    m_d3dCompileFunc = d3dCompileFunc;

    D3D11_BLEND_DESC blendStateDesc;
    memset(&blendStateDesc, 0, sizeof(blendStateDesc));
    blendStateDesc.AlphaToCoverageEnable = FALSE;
    blendStateDesc.IndependentBlendEnable = FALSE;
    blendStateDesc.RenderTarget[0].BlendEnable = TRUE;
    blendStateDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_ALPHA;
    blendStateDesc.RenderTarget[0].DestBlend = D3D11_BLEND_ILW_SRC_ALPHA;
    blendStateDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blendStateDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_SRC_ALPHA;
    blendStateDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ILW_SRC_ALPHA;
    blendStateDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blendStateDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

    if (!SUCCEEDED(status = m_d3dDevice->CreateBlendState(&blendStateDesc, &m_blendState)))
    {
        HandleFailure();
    }

    ID3DBlob                *pVSBlob = NULL;
    ID3DBlob                *pVSBlobErrors = NULL;

    char * vsText = NULL;

    ID3DBlob                *pPSBlob = NULL;
    ID3DBlob                *pPSBlobErrors = NULL;

    char * psText = NULL;

    // Vertex Shader
    vsText =
        "struct VSIn                                                                                    \n"
        "{                                                                                              \n"
        "   float4 position : POSITION;                                                                 \n"
        "   float2 texcoord : TEXCOORD;                                                                 \n"
        "};                                                                                             \n"
        "                                                                                               \n"
        "cbuffer ControlBuf : register(b0)                                                              \n"
        "{                                                                                              \n"
        "   float4 g_Color: packoffset(c0);                                                             \n"
        "   float g_OffsetX: packoffset(c1);                                                            \n"
        "   float g_OffsetY: packoffset(c1.y);                                                          \n"
        "   float g_ScaleX: packoffset(c1.z);                                                           \n"
        "   float g_ScaleY: packoffset(c1.w);                                                           \n"
        "}                                                                                              \n"
        "                                                                                               \n"
        "struct VSOut                                                                                   \n"
        "{                                                                                              \n"
        "   float4 position : SV_POSITION;                                                              \n"
        "   float2 texcoord : TEXCOORD;                                                                 \n"
        "   float4 color : COLOR;                                                                       \n"
        "};                                                                                             \n"
        "                                                                                               \n"
        "VSOut Main( VSIn vertex )                                                                      \n"
        "{                                                                                              \n"
        "   VSOut output;                                                                               \n"
        "                                                                                               \n"
        "   output.position = float4(g_ScaleX, g_ScaleY, 1.0, 1.0) * vertex.position + float4(g_OffsetX, g_OffsetY, 0.0, 0.0);\n"
        "   output.texcoord = vertex.texcoord;                                                          \n"
        "   output.color = g_Color;                                                                     \n"
        "                                                                                               \n"
        "   return output;                                                                              \n"
        "}                                                                                              \n";


    if (!SUCCEEDED(status = m_d3dCompileFunc(vsText, strlen(vsText), NULL, NULL, NULL, "Main", "vs_4_0", 0, 0, &pVSBlob, &pVSBlobErrors)))
    {
        char * error = (char *)pVSBlobErrors->GetBufferPointer();
        HandleFailure();
    }

    if (!SUCCEEDED(status = m_d3dDevice->CreateVertexShader(pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), NULL, &m_vs)))
    {
        HandleFailure();
    }

    D3D11_INPUT_ELEMENT_DESC inputLayoutDesc[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 16, D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };

    if (!SUCCEEDED(status = m_d3dDevice->CreateInputLayout(inputLayoutDesc, 2, pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), &m_inputLayout)))
    {
        HandleFailure();
    }

    if (pVSBlob) pVSBlob->Release();
    if (pVSBlobErrors) pVSBlobErrors->Release();

    // Pixel Shader
    psText =
        "struct VSOut                                                                                   \n"
        "{                                                                                              \n"
        "   float4 position : SV_POSITION;                                                              \n"
        "   float2 texcoord : TEXCOORD;                                                                 \n"
        "   float4 color : COLOR;                                                                       \n"
        "};                                                                                             \n"
        "                                                                                               \n"
        "Texture2D txDiffuse : register( t0 );                                                          \n"
        "SamplerState samLinear : register( s0 );                                                       \n"
        "                                                                                               \n"
        "float4 PS( VSOut frag ): SV_Target                                                             \n"
        "{                                                                                              \n"
        "    float4 clr = frag.color * txDiffuse.Sample(samLinear, frag.texcoord);                      \n"
        "                                                                                               \n"
        "    return clr;                                                                                \n"
        "}                                                                                              \n";

    if (!SUCCEEDED(status = m_d3dCompileFunc(psText, strlen(psText), NULL, NULL, NULL, "PS", "ps_4_0", 0, 0, &pPSBlob, &pPSBlobErrors)))
    {
        char * error = (char *)pPSBlobErrors->GetBufferPointer();
        HandleFailure();
    }
    if (!SUCCEEDED(status = m_d3dDevice->CreatePixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, &m_ps)))
    {
        HandleFailure();
    }

    if (pPSBlob) pPSBlob->Release();
    if (pPSBlobErrors) pPSBlobErrors->Release();

    D3D11_TEXTURE2D_DESC textureDesc;
    // Init
    {
        ZeroMemory(&textureDesc, sizeof(textureDesc));
        textureDesc.MipLevels = 1;
        textureDesc.ArraySize = 1;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.SampleDesc.Quality = 0;
        textureDesc.Usage = D3D11_USAGE_DEFAULT;
        textureDesc.CPUAccessFlags = 0;
        textureDesc.MiscFlags = 0;
    }

    D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
    // Init
    {
        ZeroMemory(&shaderResourceViewDesc, sizeof(shaderResourceViewDesc));
        shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        shaderResourceViewDesc.Texture2D.MostDetailedMip = 0;
        shaderResourceViewDesc.Texture2D.MipLevels = 1;
    }

    D3D11_BUFFER_DESC constBufDesc;
    // Init
    {
        ZeroMemory(&constBufDesc, sizeof(constBufDesc));
        constBufDesc.Usage = D3D11_USAGE_DYNAMIC;
        constBufDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        constBufDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        constBufDesc.MiscFlags = 0;
        constBufDesc.StructureByteStride = 0;
    }

    // One character 9x15
    unsigned int w = 256, h = 32;
    unsigned char * numbersImageData = (unsigned char *)malloc(w*h*4*sizeof(unsigned char));

    memset(numbersImageData, 0, w*h*4*sizeof(unsigned char));

    unsigned char numSections[10] =
    {
        //87654321
        0b01110111, // 0
        0b00100100, // 1
        0b01011101, // 2
        0b01101101, // 3
        0b00101110, // 4
        0b01101011, // 5
        0b01111011, // 6
        0b00100101, // 7
        0b01111111, // 8
        0b01101111  // 9
                    //87654321
    };

    // Original character design ix 3x5
    const int charScale = m_charScale;

    const int scaledCharSizeX = charScale * m_baseCharSizeX;
    const int scaledCharSizeY = charScale * m_baseCharSizeY;

    auto plotPixel = [charScale, numbersImageData, w, h](int offsetX, int offsetY, int pixelX, int pixelY)
    {
        for (int i = 0; i < charScale; ++i)
        {
            for (int j = 0; j < charScale; ++j)
            {
                int xCoord = offsetX + pixelX*charScale + i;
                int yCoord = offsetY + pixelY*charScale + j;
                int channelCoord = (xCoord + yCoord*w)*4;
                numbersImageData[channelCoord  ] = 255;
                numbersImageData[channelCoord+1] = 255;
                numbersImageData[channelCoord+2] = 255;
                numbersImageData[channelCoord+3] = 255;
            }
        }
    };

    // Create font texture
    for (int num = 0; num < 10; ++num)
    {
        int offsetX = m_charOutline + num * (scaledCharSizeX + 2*m_charOutline);
        int offsetY = m_charOutline;
        unsigned char numSegm = numSections[num];

        if (numSegm & 0b00000001)
        {
            plotPixel(offsetX, offsetY, 0, 0);
            plotPixel(offsetX, offsetY, 1, 0);
            plotPixel(offsetX, offsetY, 2, 0);
        }
        if (numSegm & 0b00000010)
        {
            plotPixel(offsetX, offsetY, 0, 0);
            plotPixel(offsetX, offsetY, 0, 1);
            plotPixel(offsetX, offsetY, 0, 2);
        }
        if (numSegm & 0b00000100)
        {
            plotPixel(offsetX, offsetY, 2, 0);
            plotPixel(offsetX, offsetY, 2, 1);
            plotPixel(offsetX, offsetY, 2, 2);
        }
        if (numSegm & 0b00001000)
        {
            plotPixel(offsetX, offsetY, 0, 2);
            plotPixel(offsetX, offsetY, 1, 2);
            plotPixel(offsetX, offsetY, 2, 2);
        }
        if (numSegm & 0b00010000)
        {
            plotPixel(offsetX, offsetY, 0, 2);
            plotPixel(offsetX, offsetY, 0, 3);
            plotPixel(offsetX, offsetY, 0, 4);
        }
        if (numSegm & 0b00100000)
        {
            plotPixel(offsetX, offsetY, 2, 2);
            plotPixel(offsetX, offsetY, 2, 3);
            plotPixel(offsetX, offsetY, 2, 4);
        }
        if (numSegm & 0b01000000)
        {
            plotPixel(offsetX, offsetY, 0, 4);
            plotPixel(offsetX, offsetY, 1, 4);
            plotPixel(offsetX, offsetY, 2, 4);
        }
    }

    // Add outline
    for (int x = 0; x < (int)w; ++x)
    {
        for (int y = 0; y < (int)h; ++y)
        {
            if (numbersImageData[(x + y*w)*4+3] != 0)
                continue;

            bool neighbourFound = false;
            for (int xo = -m_charOutline; xo <= m_charOutline; ++xo)
            {
                for (int yo = -m_charOutline; yo <= m_charOutline; ++yo)
                {
                    int x1 = std::clamp(x + xo, 0, (int)(w - 1));
                    int y1 = std::clamp(y + yo, 0, (int)(h - 1));

                    if (numbersImageData[(x1 + y1*w)*4] != 0)
                    {
                        neighbourFound = true;
                        break;
                    }
                }
            }

            if (neighbourFound)
            {
                // Black outline
                numbersImageData[(x + y*w)*4  ] = 0;
                numbersImageData[(x + y*w)*4+1] = 0;
                numbersImageData[(x + y*w)*4+2] = 0;
                numbersImageData[(x + y*w)*4+3] = 255;
            }
        }
    }

    //darkroom::saveBmp(numbersImageData, "d:\\fps_font.bmp", w, h, darkroom::BufferFormat::RGBA8);

    textureDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    textureDesc.Width = w;
    textureDesc.Height = h;
    textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    D3D11_SUBRESOURCE_DATA initialData;
    initialData.pSysMem = numbersImageData;
    initialData.SysMemPitch = w * 4 * sizeof(unsigned char);
    initialData.SysMemSlicePitch = 0;

    if (!SUCCEEDED(status = shadermod::Tools::CreateTexture2D(m_d3dDevice, &textureDesc, &initialData, &m_numbersTexture)))
    {
        HandleFailure();
    }

    free(numbersImageData);
    numbersImageData = nullptr;

    shaderResourceViewDesc.Format = textureDesc.Format;
    if (!SUCCEEDED(status = m_d3dDevice->CreateShaderResourceView(m_numbersTexture, &shaderResourceViewDesc, &m_numbersTextureSRV)))
    {
        HandleFailure();
    }

    constBufDesc.ByteWidth = (sizeof(ShaderConstBuf) + 15) & ~15;
    if (!SUCCEEDED(status = d3dDevice->CreateBuffer(&constBufDesc, 0, &m_variableOffsetsBuffer)))
    {
        HandleFailure();
    }

    {
        const unsigned int indsPerNumber = 6;
        unsigned int inds[indsPerNumber] = { 0, 1, 2, 2, 3, 0 };

        D3D11_BUFFER_DESC indexBufferDesc;
        D3D11_SUBRESOURCE_DATA indexData;

        indexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        indexBufferDesc.ByteWidth = sizeof(unsigned int) * indsPerNumber;
        indexBufferDesc.BindFlags = D3D11_BIND_INDEX_BUFFER;
        indexBufferDesc.CPUAccessFlags = 0;
        indexBufferDesc.MiscFlags = 0;
        indexBufferDesc.StructureByteStride = 0;

        indexData.pSysMem = inds;
        indexData.SysMemPitch = 0;
        indexData.SysMemSlicePitch = 0;

        if (!SUCCEEDED(status = m_d3dDevice->CreateBuffer(&indexBufferDesc, &indexData, &m_numbersIndexBuf)))
        {
            HandleFailure();
        }
    }

    m_vertexStride = sizeof(VSInput);
    for (int num = 0; num < 10; ++num)
    {
        const unsigned int vertsPerButton = 4;
        D3D11_BUFFER_DESC vertexBufferDesc;
        D3D11_SUBRESOURCE_DATA vertexData;

        vertexBufferDesc.Usage = D3D11_USAGE_DEFAULT;
        vertexBufferDesc.ByteWidth = sizeof(VSInput) * vertsPerButton;
        vertexBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER;
        vertexBufferDesc.CPUAccessFlags = 0;
        vertexBufferDesc.MiscFlags = 0;
        vertexBufferDesc.StructureByteStride = 0;

        const int charSizeX = scaledCharSizeX + 2*m_charOutline;
        const int charSizeY = scaledCharSizeY + 2*m_charOutline;
        const int charOffsetX = num * charSizeX;
        const int charOffsetY = 0;

        const float lwrCharTCX0 = charOffsetX / (float)w;
        const float lwrCharTCX1 = (charOffsetX + charSizeX) / (float)w;
        const float lwrCharTCY0 = charOffsetY / (float)h;
        const float lwrCharTCY1 = (charOffsetY + charSizeY) / (float)h;

        VSInput verts[vertsPerButton] =
        {
            // 0
            {
                { 0.0f, 0.0f, 0.0f, 1.0f },{ lwrCharTCX0, lwrCharTCY0 }
            },

            // 1
            {
                { 1.0f, 0.0f, 0.0f, 1.0f },{ lwrCharTCX1, lwrCharTCY0 }
            },

            // 2
            {
                { 1.0f, 1.0f, 0.0f, 1.0f },{ lwrCharTCX1, lwrCharTCY1 }
            },

            // 3
            {
                { 0.0f, 1.0f, 0.0f, 1.0f },{ lwrCharTCX0, lwrCharTCY1 }
            }
        };

        // Ordinary
        vertexData.pSysMem = verts;
        vertexData.SysMemPitch = 0;
        vertexData.SysMemSlicePitch = 0;

        if (!SUCCEEDED(status = m_d3dDevice->CreateBuffer(&vertexBufferDesc, &vertexData, &m_numbersVertexBufs[num])))
        {
            HandleFailure();
        }
    }

    m_isInitialized = true;

    return S_OK;
}

void DebugRenderer::deinit()
{
    m_isInitialized = false;

    SAFE_RELEASE(m_blendState);
    SAFE_RELEASE(m_inputLayout);
    SAFE_RELEASE(m_vs);
    SAFE_RELEASE(m_ps);

    SAFE_RELEASE(m_variableOffsetsBuffer);

    SAFE_RELEASE(m_numbersTexture);
    SAFE_RELEASE(m_numbersTextureSRV);

    SAFE_RELEASE(m_numbersIndexBuf);
    for (int num = 0; num < 10; ++num)
    {
        SAFE_RELEASE(m_numbersVertexBufs[num]);
    }
}

HRESULT DebugRenderer::renderFPS(double dt, FPSCounterPos fpsCounterPos,
            unsigned int renderBufWidth, unsigned int renderBufHeight,
            ID3D11RasterizerState * extRasterizerState, ID3D11SamplerState * extSamplerState, ID3D11DepthStencilState * extDepthStencilState,
            ID3D11RenderTargetView * extRTV
            )
{
    m_immediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    m_immediateContext->IASetInputLayout(m_inputLayout);
    m_immediateContext->VSSetShader(m_vs, NULL, 0);
    m_immediateContext->RSSetState(extRasterizerState);
    m_immediateContext->PSSetShader(m_ps, NULL, 0);
    m_immediateContext->PSSetShaderResources(0, 1, &m_numbersTextureSRV);
    m_immediateContext->PSSetSamplers(0, 1, &extSamplerState);
    m_immediateContext->OMSetRenderTargets(1, &extRTV, NULL);
    m_immediateContext->OMSetDepthStencilState(extDepthStencilState, 0xFFFFFFFF);
    m_immediateContext->OMSetBlendState(m_blendState, NULL, 0xffffffff);

#define DBG_FPS_OFF                 0
#define DBG_FPS_TOPRIGHT            1
#define DBG_FPS_BOTRIGHT            2
#define DBG_FPS_BOTLEFT             3
#define DBG_FPS_TOPLEFT             4

    const int scaledCharSizeX = m_charScale * m_baseCharSizeX;
    const int scaledCharSizeY = m_charScale * m_baseCharSizeY;
    int charSizeX = scaledCharSizeX + 2*m_charOutline;
    int charSizeY = scaledCharSizeY + 2*m_charOutline;

    double dtAclwmAlpha = 0.2;
    static double dtAclwm = 0.0;
    dtAclwm = dt * dtAclwmAlpha + dtAclwm * (1.0 - dtAclwmAlpha);

    static double timeSinceUpdate = 0.0;
    static int fps = 0;

    const double updateInterval = 500.0;
    if (timeSinceUpdate > updateInterval)
    {
        fps = (int)(1000.0 / dtAclwm);
        timeSinceUpdate = 0.0;
    }
    else
    {
        timeSinceUpdate += dt;
    }

#define DBG_FONT_HALIGN_LEFT    0
#define DBG_FONT_HALIGN_CENTER  1
#define DBG_FONT_HALIGN_RIGHT   2
#define DBG_FONT_VALIGN_BOT     0
#define DBG_FONT_VALIGN_CENTER  1
#define DBG_FONT_VALIGN_TOP     2

    int h_align = DBG_FONT_HALIGN_RIGHT;
    int v_align = DBG_FONT_VALIGN_CENTER;
    float startOffsetXNormalized = 1.0f;
    float startOffsetYNormalized = 1.0f;

    if (fpsCounterPos == FPSCounterPos::kTOP_RIGHT)
    {
        h_align = DBG_FONT_HALIGN_RIGHT;
        v_align = DBG_FONT_VALIGN_TOP;
        startOffsetXNormalized =  1.0f;
        startOffsetYNormalized =  1.0f;
    }
    else if (fpsCounterPos == FPSCounterPos::kBOTTOM_RIGHT)
    {
        h_align = DBG_FONT_HALIGN_RIGHT;
        v_align = DBG_FONT_VALIGN_BOT;
        startOffsetXNormalized =  1.0f;
        startOffsetYNormalized = -1.0f;
    }
    else if (fpsCounterPos == FPSCounterPos::kBOTTOM_LEFT)
    {
        h_align = DBG_FONT_HALIGN_LEFT;
        v_align = DBG_FONT_VALIGN_BOT;
        startOffsetXNormalized = -1.0f;
        startOffsetYNormalized = -1.0f;
    }
    else if (fpsCounterPos == FPSCounterPos::kTOP_LEFT)
    {
        h_align = DBG_FONT_HALIGN_LEFT;
        v_align = DBG_FONT_VALIGN_TOP;
        startOffsetXNormalized = -1.0f;
        startOffsetYNormalized =  1.0f;
    }

    char fps_str[16];
    sprintf_s(fps_str, 16, "%d", fps);

    auto renderFPSNumber = [&](int num, float offsetX, float offsetY, float sizeX, float sizeY)
    {
        ShaderConstBuf controlData_Left =
        {
            1.0f, 1.0f, 1.0f, 1.0f, // Color
            offsetX, offsetY, sizeX, sizeY
        };

        D3D11_MAPPED_SUBRESOURCE subResource;

        m_immediateContext->Map(m_variableOffsetsBuffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &subResource);
        memcpy(subResource.pData, &controlData_Left, sizeof(ShaderConstBuf));
        m_immediateContext->Unmap(m_variableOffsetsBuffer, 0);
        m_immediateContext->VSSetConstantBuffers(0, 1, &m_variableOffsetsBuffer);

        UINT vbStride = m_vertexStride;
        UINT offset = 0;

        m_immediateContext->IASetIndexBuffer(m_numbersIndexBuf, DXGI_FORMAT_R32_UINT, 0);
        m_immediateContext->IASetVertexBuffers(0, 1, &m_numbersVertexBufs[num], &vbStride, &offset);
        m_immediateContext->DrawIndexed(6, 0, 0);
    };

    const float charSizeXNormalized = charSizeX / (float)renderBufWidth * 2.0f;
    const float charSizeYNormalized = charSizeY / (float)renderBufHeight * 2.0f;

    float offsetX = startOffsetXNormalized;
    float offsetY = (v_align == DBG_FONT_VALIGN_TOP) ? startOffsetYNormalized :
        (v_align == DBG_FONT_VALIGN_BOT) ? startOffsetYNormalized + charSizeYNormalized :
        startOffsetYNormalized + charSizeYNormalized * 0.5f;
    const float charPaddingXNormalized = 2.0f / (float)renderBufWidth;
    if (h_align == DBG_FONT_HALIGN_LEFT)
    {
        for (int fpsNum = 0, fpsNumEnd = (int)strlen(fps_str); fpsNum < fpsNumEnd; ++fpsNum)
        {
            int num = std::clamp(fps_str[fpsNum] - '0', 0, 9);

            renderFPSNumber(num, offsetX, offsetY, charSizeXNormalized, -charSizeYNormalized);
            offsetX += charSizeXNormalized + charPaddingXNormalized;
        }
    }
    else if (h_align == DBG_FONT_HALIGN_CENTER)
    {
        int strlenFPS = (int)strlen(fps_str);
        offsetX -= (charSizeXNormalized * strlenFPS + charPaddingXNormalized * (strlenFPS - 1)) * 0.5f;
        for (int fpsNum = 0, fpsNumEnd = strlenFPS; fpsNum < fpsNumEnd; ++fpsNum)
        {
            int num = std::clamp(fps_str[fpsNum] - '0', 0, 9);

            renderFPSNumber(num, offsetX, offsetY, charSizeXNormalized, -charSizeYNormalized);
            offsetX += charSizeXNormalized + charPaddingXNormalized;
        }
    }
    else if (h_align == DBG_FONT_HALIGN_RIGHT)
    {
        for (int fpsNum = (int)strlen(fps_str) - 1, fpsNumEnd = 0; fpsNum >= fpsNumEnd; --fpsNum)
        {
            int num = std::clamp(fps_str[fpsNum] - '0', 0, 9);

            renderFPSNumber(num, offsetX - charSizeXNormalized, offsetY, charSizeXNormalized, -charSizeYNormalized);
            offsetX -= charSizeXNormalized + charPaddingXNormalized;
        }
    }

#undef DBG_FONT_HALIGN_LEFT
#undef DBG_FONT_HALIGN_CENTER
#undef DBG_FONT_HALIGN_RIGHT
#undef DBG_FONT_VALIGN_BOT
#undef DBG_FONT_VALIGN_CENTER
#undef DBG_FONT_VALIGN_TOP

    return S_OK;
}
