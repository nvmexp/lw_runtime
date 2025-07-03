#pragma once

#include "CommonTools.h"
#include "D3DCompilerHandler.h"

#include <vector>

namespace shadermod
{
#define SAFE_RELEASE(x) if (x) { x->Release(); x = nullptr; }

    class ResourceManager
    {
    public:

        ResourceManager(ID3D11Device* d3dDevice, D3DCompilerHandler* d3dCompiler)
        {
            init(d3dDevice, d3dCompiler);
        }

        virtual ~ResourceManager()
        {
            destroy();
        }

        ID3D11Device*                           m_d3dDevice     = nullptr;
        D3DCompilerHandler*                     m_d3dCompiler   = nullptr;

        std::vector<ID3D11Texture2D *>          m_textures;
        std::vector<ID3D11SamplerState *>       m_samplers;
        std::vector<ID3D11Buffer *>             m_constantBuffers;
        std::vector<ID3D11ShaderResourceView *> m_shaderResources;
        std::vector<ID3D11RenderTargetView *>   m_renderTargets;

        std::vector<ID3D11VertexShader *>       m_vertexShaders;
        std::vector<ID3D11PixelShader *>        m_pixelShaders;

        std::vector<ID3D11RasterizerState *>    m_rasterizerStates;
        std::vector<ID3D11DepthStencilState *>  m_depthStencilStates;
        std::vector<ID3D11BlendState *>         m_alphaBlendStates;

        HRESULT createTexture(const D3D11_TEXTURE2D_DESC & desc, const D3D11_SUBRESOURCE_DATA * pInitialData, ID3D11Texture2D ** ppTexture)
        {
            HRESULT hr = shadermod::Tools::CreateTexture2D(m_d3dDevice, &desc, pInitialData, ppTexture);

            if (!FAILED(hr))
            {
                m_textures.push_back(*ppTexture);
            }
            return hr;
        }

        HRESULT createSampler(const D3D11_SAMPLER_DESC & desc, ID3D11SamplerState ** ppSampler)
        {
            HRESULT hr = m_d3dDevice->CreateSamplerState(&desc, ppSampler);

            if (!FAILED(hr))
            {
                m_samplers.push_back(*ppSampler);
            }
            return hr;
        }

        HRESULT createConstantBuffer(const D3D11_BUFFER_DESC & desc, D3D11_SUBRESOURCE_DATA * pInitialData, ID3D11Buffer ** ppBuffer)
        {
            HRESULT hr = m_d3dDevice->CreateBuffer(&desc, pInitialData, ppBuffer);

            if (!FAILED(hr))
            {
                m_constantBuffers.push_back(*ppBuffer);
            }
            return hr;
        }

        void destroyShaderResource(ID3D11ShaderResourceView *pSRView)
        {
            for (size_t i = 0, iend = m_shaderResources.size(); i < iend; ++i)
            {
                if (m_shaderResources[i] == pSRView)
                {
                    m_shaderResources.erase(m_shaderResources.begin() + i);
                    SAFE_RELEASE(pSRView);

                    return;
                }
            }

        }
        HRESULT createShaderResource(ID3D11Resource * pResource, const D3D11_SHADER_RESOURCE_VIEW_DESC & desc, ID3D11ShaderResourceView **ppSRView)
        {
            HRESULT hr = m_d3dDevice->CreateShaderResourceView(pResource, &desc, ppSRView);

            if (!FAILED(hr))
            {
                m_shaderResources.push_back(*ppSRView);
            }
            return hr;
        }

        void destroyRenderTarget(ID3D11RenderTargetView *pRTView)
        {
            for (size_t i = 0, iend = m_renderTargets.size(); i < iend; ++i)
            {
                if (m_renderTargets[i] == pRTView)
                {
                    m_renderTargets.erase(m_renderTargets.begin() + i);
                    SAFE_RELEASE(pRTView);

                    return;
                }
            }

        }
        HRESULT createRenderTarget(ID3D11Resource * pResource, const D3D11_RENDER_TARGET_VIEW_DESC & desc, ID3D11RenderTargetView **ppRTView)
        {
            HRESULT hr = m_d3dDevice->CreateRenderTargetView(pResource, &desc, ppRTView);

            if (!FAILED(hr))
            {
                m_renderTargets.push_back(*ppRTView);
            }
            return hr;
        }

        HRESULT createVertexShader(const void * pShaderBytecode, size_t bytecodeLen, ID3D11ClassLinkage * pClassLinkage, ID3D11VertexShader **ppVertexShader)
        {
            HRESULT hr = m_d3dDevice->CreateVertexShader(pShaderBytecode, bytecodeLen, pClassLinkage, ppVertexShader);

            if (!FAILED(hr))
            {
                m_vertexShaders.push_back(*ppVertexShader);
            }
            return hr;
        }

        HRESULT createPixelShader(const void * pShaderBytecode, size_t bytecodeLen, ID3D11ClassLinkage * pClassLinkage, ID3D11PixelShader **ppPixelShader)
        {
            HRESULT hr = m_d3dDevice->CreatePixelShader(pShaderBytecode, bytecodeLen, pClassLinkage, ppPixelShader);

            if (!FAILED(hr))
            {
                m_pixelShaders.push_back(*ppPixelShader);
            }
            return hr;
        }

        HRESULT createRasterizerState(const D3D11_RASTERIZER_DESC & rasterizerStateDesc, ID3D11RasterizerState ** ppRasterizerState)
        {
            HRESULT hr = m_d3dDevice->CreateRasterizerState(&rasterizerStateDesc, ppRasterizerState);

            if (!FAILED(hr))
            {
                m_rasterizerStates.push_back(*ppRasterizerState);
            }
            return hr;
        }
        HRESULT createDepthStencilState(const D3D11_DEPTH_STENCIL_DESC & depthStencilStateDesc, ID3D11DepthStencilState ** ppDepthStencilState)
        {
            HRESULT hr = m_d3dDevice->CreateDepthStencilState(&depthStencilStateDesc, ppDepthStencilState);

            if (!FAILED(hr))
            {
                m_depthStencilStates.push_back(*ppDepthStencilState);
            }
            return hr;
        }
        HRESULT createAlphaBlendState(const D3D11_BLEND_DESC & alphaBlendStateDesc, ID3D11BlendState ** ppAlphaBlendState)
        {
            HRESULT hr = m_d3dDevice->CreateBlendState(&alphaBlendStateDesc, ppAlphaBlendState);

            if (!FAILED(hr))
            {
                m_alphaBlendStates.push_back(*ppAlphaBlendState);
            }
            return hr;
        }

        void init(ID3D11Device* d3dDevice, D3DCompilerHandler* d3dCompiler)
        {
            m_d3dDevice = d3dDevice;
            m_d3dCompiler = d3dCompiler;
        }

        void deleteAllResources()
        {
            for (size_t i = 0, iend = m_textures.size(); i < iend; ++i)
            {
                SAFE_RELEASE(m_textures[i]);
            }
            m_textures.resize(0);
            for (size_t i = 0, iend = m_samplers.size(); i < iend; ++i)
            {
                SAFE_RELEASE(m_samplers[i]);
            }
            m_samplers.resize(0);
            for (size_t i = 0, iend = m_constantBuffers.size(); i < iend; ++i)
            {
                SAFE_RELEASE(m_constantBuffers[i]);
            }
            m_constantBuffers.resize(0);
            for (size_t i = 0, iend = m_shaderResources.size(); i < iend; ++i)
            {
                SAFE_RELEASE(m_shaderResources[i]);
            }
            m_shaderResources.resize(0);
            for (size_t i = 0, iend = m_renderTargets.size(); i < iend; ++i)
            {
                SAFE_RELEASE(m_renderTargets[i]);
            }
            m_renderTargets.resize(0);
            for (size_t i = 0, iend = m_pixelShaders.size(); i < iend; ++i)
            {
                SAFE_RELEASE(m_pixelShaders[i]);
            }
            m_pixelShaders.resize(0);
            for (size_t i = 0, iend = m_vertexShaders.size(); i < iend; ++i)
            {
                SAFE_RELEASE(m_vertexShaders[i]);
            }
            m_vertexShaders.resize(0);

            for (size_t i = 0, iend = m_rasterizerStates.size(); i < iend; ++i)
            {
                SAFE_RELEASE(m_rasterizerStates[i]);
            }
            m_rasterizerStates.resize(0);
            for (size_t i = 0, iend = m_depthStencilStates.size(); i < iend; ++i)
            {
                SAFE_RELEASE(m_depthStencilStates[i]);
            }
            m_depthStencilStates.resize(0);
            for (size_t i = 0, iend = m_alphaBlendStates.size(); i < iend; ++i)
            {
                SAFE_RELEASE(m_alphaBlendStates[i]);
            }
            m_alphaBlendStates.resize(0);
        }

        void destroy()
        {
            deleteAllResources();
        }

    protected:

    };

#undef SAFE_RELEASE
}
