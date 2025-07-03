#pragma once

#include <d3d11.h>

#include "D3DCompilerHandler.h"
#include "CommonStructs.h"

class DebugRenderer 
{
protected:

    ID3D11Device *          m_d3dDevice = nullptr;
    ID3D11DeviceContext *   m_immediateContext = nullptr;
    PFND3DCOMPILEFUNC       m_d3dCompileFunc = nullptr;

public:

    struct VSInput
    {
        float position[4];
        float texcoord[2];
    };

    struct ShaderConstBuf
    {
        float cr, cg, cb, ca;
        float posX, posY;
        float sizeX, sizeY;
    };

    bool                        m_isInitialized = false;

    // Rendering settings
    ID3D11BlendState *          m_blendState = nullptr;
    ID3D11InputLayout *         m_inputLayout = nullptr;
    ID3D11VertexShader *        m_vs = nullptr;
    ID3D11PixelShader *         m_ps = nullptr;

    ID3D11Buffer *              m_variableOffsetsBuffer = nullptr;

    ID3D11Texture2D *           m_numbersTexture = nullptr;
    ID3D11ShaderResourceView *  m_numbersTextureSRV = nullptr;

    unsigned int        m_vertexStride = 0;
    ID3D11Buffer *      m_numbersIndexBuf = nullptr;
    ID3D11Buffer *      m_numbersVertexBufs[10] = { nullptr };

    // Debug font settings
    int m_baseCharSizeX = 3;
    int m_baseCharSizeY = 5;
    int m_charOutline = 1;
    int m_charScale = 3;

    HRESULT init(ID3D11Device * d3dDevice, ID3D11DeviceContext * immediateContext, PFND3DCOMPILEFUNC d3dCompileFunc);
    void deinit();

    enum class FPSCounterPos
    {
        kTOP_RIGHT = 1,
        kBOTTOM_RIGHT,
        kBOTTOM_LEFT,
        kTOP_LEFT,
    };
    HRESULT renderFPS(double dt, FPSCounterPos fpsCounterPos,
                unsigned int renderBufWidth, unsigned int renderBufHeight,
                ID3D11RasterizerState * extRasterizerState, ID3D11SamplerState * extSamplerState, ID3D11DepthStencilState * extDepthStencilState,
                ID3D11RenderTargetView * extRTV
                );
};
