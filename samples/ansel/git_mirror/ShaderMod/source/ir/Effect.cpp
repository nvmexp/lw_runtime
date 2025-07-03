#include <d3d11_1.h>
#include <d3dcompiler.h>

#include "Log.h"
#include "Utils.h"

#include "sha2.hpp"

#include "CommonTools.h"
#include "ResourceManager.h"
#include "D3D11CommandProcessor.h"
#include "D3D11CommandProcessorColwersions.h"
#include "MultipassConfigParserError.h"

#include "ir/Defines.h"
#include "ir/UserConstantManager.h"
#include "ir/TypeEnums.h"
#include "ir/TypeColwersions.h"
#include "ir/SpecializedPool.h"
#include "ir/Sampler.h"
#include "ir/Constant.h"
#include "ir/DataSource.h"
#include "ir/Texture.h"
#include "ir/ShaderHelpers.h"
#include "ir/VertexShader.h"
#include "ir/PixelShader.h"
#include "ir/PipelineStates.h"
#include "ir/Pass.h"
#include "ir/Effect.h"

#include "darkroom/ImageLoader.h"
#include "darkroom/StringColwersion.h"

#include <vector>
#include <list>
#include <algorithm>
#include <sstream>
#include <assert.h>
#include <stdint.h>
#include <string>

namespace shadermod
{
namespace ir
{
    void updateTextureDescFromTexture(Texture * pTexture, D3D11_TEXTURE2D_DESC & textureDesc)
    {
        Tools::ClearD3D11TexDesc(textureDesc);
        textureDesc.Format = ircolwert::formatToDXGIFormat(pTexture->m_format);
        textureDesc.Width = pTexture->m_width;
        textureDesc.Height = pTexture->m_height;
        textureDesc.MipLevels = pTexture->m_levels;
        if (textureDesc.MipLevels != 1)
        {
            textureDesc.MiscFlags = D3D11_RESOURCE_MISC_GENERATE_MIPS;
        }

#if 0
        textureDesc.BindFlags = (pTexture->m_needsRTV ? D3D11_BIND_RENDER_TARGET : 0) |
            (pTexture->m_needsSRV ? D3D11_BIND_SHADER_RESOURCE : 0);
#else
        // TODO: handle texture binding flags properly
        // at the moment they are always for both SRV and RTV binding due to potential reuse in random place,
        // but probably it can be better
        textureDesc.BindFlags = (pTexture->isReuseAllowed() ? D3D11_BIND_RENDER_TARGET : 0) | D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
#endif
    }

    Effect::Effect(CmdProcEffect * effect, ResourceManager * resourceManager):
        m_effect(effect),
        m_resourceManager(resourceManager)
    {
    }

    Effect::~Effect()
    {
        m_userConstantManager.destroyAllUserConstants();

        for (size_t i = 0, iend = m_samplers.size(); i < iend; ++i)
            m_samplers[i]->~Sampler();
        m_samplers.clear();

        for (size_t i = 0, iend = m_constants.size(); i < iend; ++i)
            m_constants[i]->~Constant();
        m_constants.clear();

        for (size_t i = 0, iend = m_constantBufs.size(); i < iend; ++i)
            m_constantBufs[i]->~ConstantBuf();
        m_constantBufs.clear();

        for (size_t i = 0, iend = m_textures.size(); i < iend; ++i)
            m_textures[i]->~Texture();
        m_textures.clear();

        for (size_t i = 0, iend = m_pixelShaders.size(); i < iend; ++i)
            m_pixelShaders[i]->~PixelShader();
        m_pixelShaders.clear();

        for (size_t i = 0, iend = m_vertexShaders.size(); i < iend; ++i)
            m_vertexShaders[i]->~VertexShader();
        m_vertexShaders.clear();

        for (size_t i = 0, iend = m_passes.size(); i < iend; ++i)
            m_passes[i]->~Pass();
        m_passes.clear();

        m_includeFilesPool.destroy();

        m_samplersPool.destroy();
        m_constantsPool.destroy();
        m_constantBufsPool.destroy();
        m_texturesPool.destroy();
        m_pixelShadersPool.destroy();
        m_vertexShadersPool.destroy();
        m_passesPool.destroy();
        
        m_namesPool.destroy();

        m_freeTexturesList.clear();

        m_resourceManager->deleteAllResources();

        m_effect->destroy();
    }

    void Effect::reset()
    {
        m_userConstantManager.destroyAllUserConstants();

        for (size_t i = 0, iend = m_samplers.size(); i < iend; ++i)
        {
            m_samplersPool.deleteElement(m_samplers[i]);
        }
        m_samplers.clear();

        for (size_t i = 0, iend = m_constants.size(); i < iend; ++i)
        {
            m_constantsPool.deleteElement(m_constants[i]);
        }
        m_constants.clear();

        for (size_t i = 0, iend = m_constantBufs.size(); i < iend; ++i)
        {
            m_constantBufsPool.deleteElement(m_constantBufs[i]);
        }
        m_constantBufs.clear();

        for (size_t i = 0, iend = m_textures.size(); i < iend; ++i)
        {
            m_texturesPool.deleteElement(m_textures[i]);
        }
        m_textures.clear();

        for (size_t i = 0, iend = m_pixelShaders.size(); i < iend; ++i)
        {
            m_pixelShadersPool.deleteElement(m_pixelShaders[i]);
        }
        m_pixelShaders.clear();

        for (size_t i = 0, iend = m_vertexShaders.size(); i < iend; ++i)
        {
            m_vertexShadersPool.deleteElement(m_vertexShaders[i]);
        }
        m_vertexShaders.clear();

        for (size_t i = 0, iend = m_passes.size(); i < iend; ++i)
        {
            m_passesPool.deleteElement(m_passes[i]);
        }
        m_passes.clear();

        // Simply re-initialize resource names pool
        m_namesPool.destroy();
        m_namesPool.preallocate();

        m_freeTexturesList.clear();

        m_resourceManager->deleteAllResources();

        m_effect->destroy();
    }

    void Effect::reserve()
    {
        // TODO: decide how much to reserve
        m_samplers.reserve(2);
        m_constants.reserve(5);
        m_constantBufs.reserve(2);
        m_textures.reserve(10);
        m_pixelShaders.reserve(10);
        m_vertexShaders.reserve(10);
        m_passes.reserve(10);
    }

    FragmentFormat Effect::getInputColorFormat() const { return m_inputColorFormat; }
    FragmentFormat Effect::getInputDepthFormat() const { return m_inputDepthFormat; }
    FragmentFormat Effect::getInputHUDlessFormat() const { return m_inputHUDlessFormat; }
    FragmentFormat Effect::getInputHDRFormat() const { return m_inputHDRFormat; }
    FragmentFormat Effect::getInputColorBaseFormat() const { return m_inputColorBaseFormat; }

    int Effect::getInputWidth() const { return m_inputWidth; }
    int Effect::getInputHeight() const { return m_inputHeight; }
    int Effect::getInputDepthWidth() const { return m_inputDepthWidth; }
    int Effect::getInputDepthHeight() const { return m_inputDepthHeight; }
    int Effect::getInputHUDlessWidth() const { return m_inputHUDlessWidth; }
    int Effect::getInputHUDlessHeight() const { return m_inputHUDlessHeight; }
    int Effect::getInputHDRWidth() const { return m_inputHDRWidth; }
    int Effect::getInputHDRHeight() const { return m_inputHDRHeight; }
    int Effect::getInputColorBaseWidth() const { return m_inputColorBaseWidth; }
    int Effect::getInputColorBaseHeight() const { return m_inputColorBaseHeight; }

    ID3D11Texture2D* Effect::getInputColorTexture() const { return m_D3DColorTexture; }
    ID3D11Texture2D* Effect::getInputDepthTexture() const { return m_D3DDepthTexture; }
    ID3D11Texture2D* Effect::getInputHUDlessTexture() const { return m_D3DHUDlessTexture; }
    ID3D11Texture2D* Effect::getInputHDRTexture() const { return m_D3DHDRTexture; }
    ID3D11Texture2D* Effect::getInputColorBaseTexture() const { return m_D3DColorBaseTexture; }
    bool Effect::isInputColorTextureSet() const { return m_D3DColorTexture != nullptr; }
    bool Effect::isInputDepthTextureSet() const { return m_D3DDepthTexture != nullptr; }
    bool Effect::isInputHUDlessTextureSet() const { return m_D3DHUDlessTexture != nullptr; }
    bool Effect::isInputHDRTextureSet() const { return m_D3DHDRTexture != nullptr; }
    bool Effect::isInputColorBaseTextureSet() const { return m_D3DColorBaseTexture != nullptr; }

    void Effect::setInputs(const InputData & colorInput, const InputData & depthInput, const InputData & hudlessInput, const InputData & hdrInput, const InputData & colorBaseInput)
    {
        m_inputWidth = colorInput.width;
        m_inputHeight = colorInput.height;
        m_inputColorFormat = colorInput.format;
        m_D3DColorTexture = colorInput.texture;

        m_inputDepthWidth = depthInput.width;
        m_inputDepthHeight = depthInput.height;
        m_inputDepthFormat = depthInput.format;
        m_D3DDepthTexture = depthInput.texture;

        m_inputHUDlessWidth = hudlessInput.width;
        m_inputHUDlessHeight = hudlessInput.height;
        m_inputHUDlessFormat = hudlessInput.format;
        m_D3DHUDlessTexture = hudlessInput.texture;

        m_inputHDRWidth = hdrInput.width;
        m_inputHDRHeight = hdrInput.height;
        m_inputHDRFormat = hdrInput.format;
        m_D3DHDRTexture = hdrInput.texture;

        m_inputColorBaseWidth = colorBaseInput.width;
        m_inputColorBaseHeight = colorBaseInput.height;
        m_inputColorBaseFormat = colorBaseInput.format;
        m_D3DColorBaseTexture = colorBaseInput.texture;
    }

    void Effect::fixInputs()
    {
        D3D11_TEXTURE2D_DESC textureDesc;
        // Init
        Tools::ClearD3D11TexDesc(textureDesc);

        D3D11_RENDER_TARGET_VIEW_DESC renderTargetViewDesc;
        // Init
        {
            ZeroMemory(&renderTargetViewDesc, sizeof(renderTargetViewDesc));
            renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
            renderTargetViewDesc.Texture2D.MipSlice = 0;
        }

        D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
        // Init
        {
            ZeroMemory(&shaderResourceViewDesc, sizeof(shaderResourceViewDesc));
            shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
            shaderResourceViewDesc.Texture2D.MostDetailedMip = 0;
            shaderResourceViewDesc.Texture2D.MipLevels = 1;
        }

        for (size_t ti = 0, tiEnd = m_textures.size(); ti < tiEnd; ++ti)
        {
            if (m_textures[ti]->m_type == Texture::TextureType::kInputColor || m_textures[ti]->m_type == Texture::TextureType::kInputDepth ||
                m_textures[ti]->m_type == Texture::TextureType::kInputHUDless || m_textures[ti]->m_type == Texture::TextureType::kInputHDR ||
                m_textures[ti]->m_type == Texture::TextureType::kInputColorBase)
            {
                Texture * pTexture = m_textures[ti];

                if (pTexture->m_type == Texture::TextureType::kInputColor)
                    pTexture->m_D3DTexture = m_D3DColorTexture;
                if (pTexture->m_type == Texture::TextureType::kInputDepth)
                    pTexture->m_D3DTexture = m_D3DDepthTexture;
                if (pTexture->m_type == Texture::TextureType::kInputHUDless)
                    pTexture->m_D3DTexture = m_D3DHUDlessTexture;
                if (pTexture->m_type == Texture::TextureType::kInputHDR)
                    pTexture->m_D3DTexture = m_D3DHDRTexture;
                if (pTexture->m_type == Texture::TextureType::kInputColorBase)
                    pTexture->m_D3DTexture = m_D3DColorBaseTexture;

                ID3D11ShaderResourceView * oldSRV = pTexture->m_D3DSRV;
                if (pTexture->m_D3DSRV)
                {
                    m_resourceManager->destroyShaderResource(pTexture->m_D3DSRV);
                    pTexture->m_D3DSRV = nullptr;
                }
                ID3D11RenderTargetView * oldRTV = pTexture->m_D3DRTV;
                if (pTexture->m_D3DRTV)
                {
                    m_resourceManager->destroyRenderTarget(pTexture->m_D3DRTV);
                    pTexture->m_D3DRTV = nullptr;
                }

                updateTextureDescFromTexture(pTexture, textureDesc);
                initTextureViews(pTexture, textureDesc, renderTargetViewDesc, shaderResourceViewDesc);

                // Fixup CmdProc resource references
                for (size_t pi = 0, piEnd = m_effect->m_passes.size(); pi < piEnd; ++pi)
                {
                    CmdProcPass & pass = m_effect->m_passes[pi];
                    for (size_t srvi = 0, srviEnd = pass.m_shaderResourceDescs.size(); srvi < srviEnd; ++srvi)
                    {
                        if ((oldSRV != nullptr) && (pass.m_shaderResourceDescs[srvi].pResource == oldSRV))
                        {
                            pass.m_shaderResourceDescs[srvi].pResource = pTexture->m_D3DSRV;
                        }
                        else if (oldSRV == nullptr)
                        {
                            Texture::TextureType cmdTexType = colwertResourceKindToTextureType(pass.m_shaderResourceDescs[srvi].kind);
                            if (cmdTexType == pTexture->m_type)
                            {
                                pass.m_shaderResourceDescs[srvi].pResource = pTexture->m_D3DSRV;
                            }
                        }
                    }
                    for (size_t rtvi = 0, rtviEnd = pass.m_renderTargets.size(); rtvi < rtviEnd && (oldRTV != nullptr); ++rtvi)
                    {
                        if (pass.m_renderTargets[rtvi] == oldRTV)
                        {
                            pass.m_renderTargets[rtvi] = pTexture->m_D3DRTV;
                        }
                    }
                }
            }
        }
    }

    Sampler * Effect::createSampler(AddressType addrU, AddressType addrV, FilterType filterMin, FilterType filterMag, FilterType filterMip)
    {
        Sampler * sampler = m_samplersPool.getElement();
        new (sampler) Sampler(addrU, addrV, filterMin, filterMag, filterMip);
        m_samplers.push_back(sampler);

        return sampler;
    }


    Constant* Effect::createConstant(ConstType type, int packOffsetInComponents)
    {
        Constant * constant = m_constantsPool.getElement();
        new (constant)Constant(type, packOffsetInComponents);
        m_constants.push_back(constant);

        return constant;
    }

    Constant* Effect::createConstant(ConstType type, const char* bindName)
    {
        Constant * constant = m_constantsPool.getElement();
        new (constant)Constant(type, bindName);
        m_constants.push_back(constant);

        return constant;
    }

    Constant* Effect::createConstant(const char* userConstName, int packOffsetInComponents)
    {
        Constant * constant = m_constantsPool.getElement();
        new (constant)Constant(userConstName, packOffsetInComponents);
        m_constants.push_back(constant);

        return constant;
    }

    Constant* Effect::createConstant(const char* userConstName, const char* bindName)
    {
        Constant * constant = m_constantsPool.getElement();
        new (constant)Constant(userConstName, bindName);
        m_constants.push_back(constant);

        return constant;
    }
    
    ConstantBuf * Effect::createConstantBuffer()
    {
        ConstantBuf * constantBuf = m_constantBufsPool.getElement();
        new (constantBuf) ConstantBuf();
        m_constantBufs.push_back(constantBuf);

        return constantBuf;
    }

    Texture * Effect::createTextureFromMemory(int width, int height, FragmentFormat format, const void * mem, bool skipLoading)
    {
        Texture * texture = m_texturesPool.getElement();
        new (texture) Texture();
        m_textures.push_back(texture);

        texture->fromMemory(width, height, format);

        // Load Texture here
        {
            if (texture->m_initData)
            {
                free(texture->m_initData);
                texture->m_initData = nullptr;
            }

            if (!skipLoading)
            {
                size_t bytesPerPixel = ircolwert::formatBitsPerPixel(ircolwert::formatToDXGIFormat(format)) / 8;

                texture->m_initData = malloc(texture->m_width * texture->m_height * bytesPerPixel*sizeof(unsigned char));
                memcpy(texture->m_initData, mem, texture->m_width * texture->m_height * bytesPerPixel*sizeof(unsigned char));
            }
        }

        return texture;
    }

    Texture * Effect::createTextureFromFile(int width, int height, FragmentFormat & formatCanChange, const wchar_t * path, bool skipLoading, bool excludeHash)
    {
        Texture * texture = nullptr;

        if (skipLoading)
        {
            texture = createTextureFromMemory(0, 0, formatCanChange, nullptr, skipLoading);
            texture->m_filepath = path;
            texture->m_excludeHash = excludeHash;

            if (width == Texture::SetAsInputFileSize)
            {
                texture->m_widthBase = Texture::TextureSizeBase::kTextureWidth;
                texture->m_widthMul = 1.0f;
                texture->m_width = 0;
            }
            if (height == Texture::SetAsInputFileSize)
            {
                texture->m_heightBase = Texture::TextureSizeBase::kTextureHeight;
                texture->m_heightMul = 1.0f;
                texture->m_height = 0;
            }
        }
        else
        {
            unsigned int imgWidth, imgHeight;
            std::vector<unsigned char> imageData = darkroom::loadImage(path, imgWidth, imgHeight, darkroom::BufferFormat::RGBA8);

            if (imageData.empty())
                return nullptr;

            // TODO: if we will support fancy formats with BPP not divisible by 8, fix this
            size_t bytesPerPixel = ircolwert::formatBitsPerPixel(ircolwert::formatToDXGIFormat(formatCanChange)) / 8;

            if (bytesPerPixel != 4)
            {
                // We do not support non-4bpp formats right now, but we can relatively safely just cast internally the R8/RG8 formats
                if (formatCanChange == FragmentFormat::kR8_uint || formatCanChange == FragmentFormat::kRG8_uint)
                {
                    formatCanChange = FragmentFormat::kRGBA8_uint;
                    bytesPerPixel = 4;
                }
            }

            // We don't support non-4bpp textures
            if (bytesPerPixel != 4)
            {
                return nullptr;
            }

            int finalWidth = (width == Texture::SetAsInputFileSize) ? imgWidth : width;
            int finalHeight = (height == Texture::SetAsInputFileSize) ? imgHeight : height;

            if (finalWidth < 1 || finalHeight < 1)
            {
                return nullptr;
            }
            
            // TODO: better rescaling
            if ((imgWidth != finalWidth) || (imgHeight != finalHeight))
            {
                std::vector<unsigned char> imageDataRescaled;
                imageDataRescaled.resize(finalWidth * finalHeight * bytesPerPixel);

                uint32_t imageDataSize = (uint32_t)imageData.size();
                unsigned char * imgDataRaw = &imageData[0];
                unsigned char * imgDataRescaledRaw = &imageDataRescaled[0];

                // Perform the colwersion
                for (int x = 0; x < finalWidth; ++x)
                {
                    for (int y = 0; y < finalHeight; ++y)
                    {
                        float xImgSpace = 0.0f;
                        if (finalWidth > 1)
                            xImgSpace = x / (float)(finalWidth - 1);
                        float yImgSpace = 0.0f;
                        if (finalHeight > 1)
                            yImgSpace = y / (float)(finalHeight - 1);

                        int xOrig = (int)(xImgSpace * (imgWidth - 1));
                        int yOrig = (int)(yImgSpace * (imgHeight - 1));

                        int pixelOffset = (int)((x+y*finalWidth)*bytesPerPixel);
                        uint32_t pixelOffsetOrig = (int)((xOrig+yOrig*imgWidth)*bytesPerPixel);
                        if (pixelOffsetOrig+3 >= imageDataSize)
                        {
                            // Sizes didn't match for some reason (corruption or garbage data)
                            return nullptr;
                        }
                        imgDataRescaledRaw[pixelOffset  ] = imgDataRaw[pixelOffsetOrig  ];
                        imgDataRescaledRaw[pixelOffset+1] = imgDataRaw[pixelOffsetOrig+1];
                        imgDataRescaledRaw[pixelOffset+2] = imgDataRaw[pixelOffsetOrig+2];
                        imgDataRescaledRaw[pixelOffset+3] = imgDataRaw[pixelOffsetOrig+3];
                    }
                }

                imageData = imageDataRescaled;

                imgWidth = finalWidth;
                imgHeight = finalHeight;
            }

            // TODO: perform format colwersion: imageData is in RGBA (4Bpp), colwert to match 'format'
            if (!imageData.empty() && (imgWidth == finalWidth) && (imgHeight == finalHeight) && (bytesPerPixel == 4))
            {
                texture = createTextureFromMemory(finalWidth, finalHeight, formatCanChange, imageData.data());
                texture->m_filepath = path;
                texture->m_excludeHash = excludeHash;

                if (width == Texture::SetAsInputFileSize)
                {
                    texture->m_widthBase = Texture::TextureSizeBase::kTextureWidth;
                    texture->m_widthMul = 1.0f;
                    texture->m_width = finalWidth;
                }
                if (height == Texture::SetAsInputFileSize)
                {
                    texture->m_heightBase = Texture::TextureSizeBase::kTextureHeight;
                    texture->m_heightMul = 1.0f;
                    texture->m_height = finalHeight;
                }
            }
        }

        return texture;
    }

    Texture * Effect::createNoiseTexture(int width, int height, FragmentFormat format)
    {
        int texW = (width == Pass::SetAsEffectInputSize) ? m_inputWidth : width,
            texH = (height == Pass::SetAsEffectInputSize) ? m_inputHeight : height;

        Texture * texture = m_texturesPool.getElement();
        new (texture) Texture();
        m_textures.push_back(texture);

        texture->genNoise(width, height, format);

        // Generate Noise Texture
        {
            if (texture->m_initData)
            {
                free(texture->m_initData);
                texture->m_initData = nullptr;
            }
            
            size_t bytesPerPixel = ircolwert::formatBitsPerPixel(ircolwert::formatToDXGIFormat(format)) / 8;
            texture->m_initData = malloc(texture->m_width * texture->m_height * bytesPerPixel*sizeof(unsigned char));

            // TODO[error]: add ptr check and throw 'out of mem' if needed
            unsigned char * ptr = (unsigned char *)texture->m_initData;
            int yOffset = 0;
            for (int nsy = 0; nsy < texture->m_height; ++nsy)
            {
                for (int nsx = 0; nsx < texture->m_width; ++nsx)
                {
                    int offset = (nsx+yOffset) * (int)bytesPerPixel;
                    for (size_t bpp = 0; bpp < bytesPerPixel; ++bpp)
                    {
                        ptr[offset+bpp] = rand() & 255;
                    }
                }
                yOffset += texture->m_width;
            }
        }

        return texture;
    }

    Texture * Effect::createRTTexture(int width, int height, FragmentFormat format)
    {
        Texture * texture = m_texturesPool.getElement();
        new (texture) Texture();
        m_textures.push_back(texture);

        texture->makeRenderTarget(width, height, format);

        return texture;
    }

    Texture * Effect::createInputColor()
    {
        Texture * texture = m_texturesPool.getElement();
        new (texture) Texture();
        m_textures.push_back(texture);

        texture->makeInputColor(m_inputWidth, m_inputHeight, m_inputColorFormat);
    
        return texture;
    }
    Texture * Effect::createInputDepth()
    {
        Texture * texture = m_texturesPool.getElement();
        new (texture) Texture();
        m_textures.push_back(texture);

        texture->makeInputDepth(m_inputDepthWidth, m_inputDepthHeight, m_inputDepthFormat);

        return texture;
    }
    Texture * Effect::createInputHUDless()
    {
        Texture * texture = m_texturesPool.getElement();
        new (texture) Texture();
        m_textures.push_back(texture);

        texture->makeInputHUDless(m_inputHUDlessWidth, m_inputHUDlessHeight, m_inputHUDlessFormat);

        return texture;
    }
    Texture * Effect::createInputHDR()
    {
        Texture * texture = m_texturesPool.getElement();
        new (texture) Texture();
        m_textures.push_back(texture);

        texture->makeInputHDR(m_inputHDRWidth, m_inputHDRHeight, m_inputHDRFormat);

        return texture;
    }
    Texture * Effect::createInputColorBase()
    {
        Texture * texture = m_texturesPool.getElement();
        new (texture) Texture();
        m_textures.push_back(texture);

        texture->makeInputColorBase(m_inputColorBaseWidth, m_inputColorBaseHeight, m_inputColorBaseFormat);

        return texture;
    }

    PixelShader * Effect::createPixelShader(const wchar_t * fileName, const char * entryPoint)
    {
        PixelShader * pixelShader = m_pixelShadersPool.getElement();
        new (pixelShader) PixelShader(fileName, entryPoint);
        m_pixelShaders.push_back(pixelShader);
        pixelShader->setTraverseStackPtr(&m_pixelShadersTraverseStack);

        return pixelShader;
    }

    VertexShader * Effect::createVertexShader(const wchar_t * fileName, const char * entryPoint)
    {
        VertexShader * vertexShader = m_vertexShadersPool.getElement();
        new (vertexShader) PixelShader(fileName, entryPoint);
        m_vertexShaders.push_back(vertexShader);
        vertexShader->setTraverseStackPtr(&m_vertexShadersTraverseStack);

        return vertexShader;
    }

    Pass * Effect::createPass(VertexShader * vertexShader, PixelShader * pixelShader, int width, int height, FragmentFormat * mrtFormats, int numMRTChannels)
    {
        Pass * pass = m_passesPool.getElement();
        new (pass) Pass(&m_namesPool, vertexShader, pixelShader, width, height, mrtFormats, numMRTChannels);
        return pass;
    }

    Pass * Effect::addPass(Pass * pass)
    {
        m_passes.push_back(pass);
        return pass;
    }

    HRESULT Effect::initTextureViews(Texture * pTexture, D3D11_TEXTURE2D_DESC & textureDesc, D3D11_RENDER_TARGET_VIEW_DESC & renderTargetViewDesc, D3D11_SHADER_RESOURCE_VIEW_DESC & shaderResourceViewDesc)
    {
        HRESULT hr = S_OK;

        // Create RTV and/or SRV if they are required for the new resource, but wasn't craeted for the old one
        if (pTexture->m_needsRTV && pTexture->m_D3DRTV == nullptr)
        {
            renderTargetViewDesc.Format = lwanselutils::colwertFromTypelessIfNeeded(lwanselutils::colwertToTypeless(textureDesc.Format));
            hr = m_resourceManager->createRenderTarget(pTexture->m_D3DTexture, renderTargetViewDesc, &pTexture->m_D3DRTV);
            if (FAILED(hr))
                return hr;
        }

        if (pTexture->m_needsSRV && pTexture->m_D3DSRV == nullptr && pTexture->m_D3DTexture)
        {
            if (pTexture->m_type == Texture::TextureType::kInputDepth)
                shaderResourceViewDesc.Format = ircolwert::formatToDXGIFormat_DepthSRV(pTexture->m_format);
            else
                shaderResourceViewDesc.Format = lwanselutils::colwertFromTypelessIfNeeded(lwanselutils::colwertToTypeless(textureDesc.Format));
            shaderResourceViewDesc.Texture2D.MipLevels = pTexture->m_levels;
            hr = m_resourceManager->createShaderResource(pTexture->m_D3DTexture, shaderResourceViewDesc, &pTexture->m_D3DSRV);
            if (FAILED(hr))
                return hr;
        }

        return hr;
    }
    HRESULT Effect::initTextureResource(Texture * pTexture, D3D11_TEXTURE2D_DESC & textureDesc, D3D11_RENDER_TARGET_VIEW_DESC & renderTargetViewDesc, D3D11_SHADER_RESOURCE_VIEW_DESC & shaderResourceViewDesc)
    {
        HRESULT hr = S_OK;

        if (m_allowReuse && pTexture->isReuseAllowed())
        {
            // Try to search texture resource with same properties in the list of free textures
            for (auto poolI = m_freeTexturesList.begin(); poolI != m_freeTexturesList.end(); poolI++)
            {
                Texture * pSrcTexture = *poolI;
                if (pTexture->compare(pSrcTexture))
                {
                    // Copy resources from the source texture to the target
                    pTexture->copyResourcesFrom(pSrcTexture);

                    // Source object is no longer useful
                    //pSrcTexture->ilwalidateResources();   // Cannot ilwalidate resources, as they actually can be useful in the wrapper
                    // TODO: add m_resourcesValid instead
                    m_freeTexturesList.erase(poolI);

                    initTextureViews(pTexture, textureDesc, renderTargetViewDesc, shaderResourceViewDesc);

                    return S_OK;
                }
            }
        }

        // No luck in finding already created resource with required properties, create new one
        {
            if (pTexture->m_type == Texture::TextureType::kInputColor)
            {
                pTexture->m_D3DTexture = m_D3DColorTexture;
            }
            else if (pTexture->m_type == Texture::TextureType::kInputDepth)
            {
                pTexture->m_D3DTexture = m_D3DDepthTexture;
            }
            else if (pTexture->m_type == Texture::TextureType::kInputHUDless)
            {
                pTexture->m_D3DTexture = m_D3DHUDlessTexture;
            }
            else if (pTexture->m_type == Texture::TextureType::kInputHDR)
            {
                pTexture->m_D3DTexture = m_D3DHDRTexture;
            }
            else if (pTexture->m_type == Texture::TextureType::kInputColorBase)
            {
                pTexture->m_D3DTexture = m_D3DColorBaseTexture;
            }
            else
            {
                updateTextureDescFromTexture(pTexture, textureDesc);
                textureDesc.Format = lwanselutils::colwertToTypeless(textureDesc.Format);

                D3D11_SUBRESOURCE_DATA initialData;
                D3D11_SUBRESOURCE_DATA * pInitialData = NULL;
                if (pTexture->m_initData)
                {
                    initialData.pSysMem = pTexture->m_initData;
                    initialData.SysMemPitch = pTexture->m_width * 4*sizeof(unsigned char);
                    initialData.SysMemSlicePitch = 0;

                    pInitialData = &initialData;
                }

                hr = m_resourceManager->createTexture(textureDesc, pInitialData, &pTexture->m_D3DTexture);
                if (FAILED(hr))
                    return hr;
            }

            if (pTexture->m_needsRTV)
            {
                renderTargetViewDesc.Format = lwanselutils::colwertFromTypelessIfNeeded(lwanselutils::colwertToTypeless(ircolwert::formatToDXGIFormat(pTexture->m_format)));
                hr = m_resourceManager->createRenderTarget(pTexture->m_D3DTexture, renderTargetViewDesc, &pTexture->m_D3DRTV);
                if (FAILED(hr))
                    return hr;
            }

            if (pTexture->m_needsSRV)
            {
                if (pTexture->m_D3DTexture)
                {
                    if (pTexture->m_type == Texture::TextureType::kInputDepth)
                        shaderResourceViewDesc.Format = ircolwert::formatToDXGIFormat_DepthSRV(pTexture->m_format);
                    else
                        shaderResourceViewDesc.Format = lwanselutils::colwertFromTypelessIfNeeded(lwanselutils::colwertToTypeless(ircolwert::formatToDXGIFormat(pTexture->m_format)));
                
                    shaderResourceViewDesc.Texture2D.MipLevels = pTexture->m_levels;
                    hr = m_resourceManager->createShaderResource(pTexture->m_D3DTexture, shaderResourceViewDesc, &pTexture->m_D3DSRV);
                    if (FAILED(hr))
                        return hr;
                }
                else
                {
                    pTexture->m_D3DSRV = nullptr;
                }
            }
        }

        return hr;
    }

    MultipassConfigParserError Effect::resolveReferences()
    {
        // Dry run:
        //  1. Create texture outputs for passes
        //  2. Set SRV/RTV flags
        //  3. Callwlate resource lifetime - for reuse 
        for (size_t i = 0, iend = m_passes.size(); i < iend; ++i)
        {
            Pass & lwrPass = *m_passes[i];
            lwrPass.deriveSize(m_inputWidth, m_inputHeight);
            for (size_t j = 0, jend = lwrPass.getDataSrcNum(); j < jend; ++j)
            {
                if (lwrPass.getDataSrc(j)->getDataType() == DataSource::DataType::kPass)
                {
                    Pass * pSourcePass = static_cast<Pass *>(lwrPass.getDataSrc(j));
                    int srcPassMRTChannel = lwrPass.getMRTChannelSrc(j);

                    bool outputFound = false;

                    // Search if the source pass already have texture created for that MRT channel
                    for (size_t jn = 0, jnend = pSourcePass->getDataOutNum(); jn < jnend; ++jn)
                    {
                        if (pSourcePass->getMRTChannelOut(jn) == srcPassMRTChannel)
                        {
                            Texture * pLwrTexture = pSourcePass->getDataOut(jn);

                            // For the CURRENT pass, replace inputs from pass to its output texture
                            lwrPass.replaceDataSrc(j, pLwrTexture);

                            pLwrTexture->m_lastPassReadIdx = (int)i;

                            outputFound = true;
                            break;
                        }
                    }

                    // Source pass doesn't have the required texture, create new
                    if (!outputFound)
                    {
                        Texture * newRTTexture = createRTTexture(
                            pSourcePass->m_width,
                            pSourcePass->m_height,
                            pSourcePass->getMRTChannelFormat(srcPassMRTChannel)
                            );

                        // Derive size for texture
                        if (pSourcePass->m_baseWidth == Pass::SetAsEffectInputSize)
                        {
                            newRTTexture->m_widthBase = Texture::TextureSizeBase::kColorBufferWidth;
                            newRTTexture->m_widthMul = pSourcePass->m_scaleWidth;
                        }
                        if (pSourcePass->m_baseHeight == Pass::SetAsEffectInputSize)
                        {
                            newRTTexture->m_heightBase = Texture::TextureSizeBase::kColorBufferHeight;
                            newRTTexture->m_heightMul = pSourcePass->m_scaleHeight;
                        }

                        newRTTexture->deriveSize(m_inputWidth, m_inputHeight, m_inputDepthWidth, m_inputDepthHeight);

                        newRTTexture->m_needsSRV = true;
                        pSourcePass->addDataOut(newRTTexture, srcPassMRTChannel);

                        // For the CURRENT pass, replace inputs from pass to its output texture
                        lwrPass.replaceDataSrc(j, newRTTexture);

                        newRTTexture->m_firstPassReadIdx = (int)i;
                        newRTTexture->m_lastPassReadIdx = (int)i;
                    }
                }
                else if (lwrPass.getDataSrc(j)->getDataType() == DataSource::DataType::kTexture)
                {
                    Texture * pLwrTexture = static_cast<Texture *>(lwrPass.getDataSrc(j));
                    pLwrTexture->m_needsSRV = true;

                    if (pLwrTexture->m_firstPassReadIdx == -1)
                        pLwrTexture->m_firstPassReadIdx = (int)i;
                    pLwrTexture->m_lastPassReadIdx = (int)i;
                }
            }

            // We need to set RTV needed flag for last pass (its outputs don't get processed, since they are no inputs)
            Pass & lastPass = *m_passes[m_passes.size()-1];
            for (size_t j = 0, jend = lwrPass.getDataOutNum(); j < jend; ++j)
            {
                Texture * pLwrOutTexture = lwrPass.getDataOut(j);

                if (pLwrOutTexture->m_firstPassWriteIdx == -1)
                    pLwrOutTexture->m_firstPassWriteIdx = (int)i;
                pLwrOutTexture->m_lastPassWriteIdx = (int)i;

                pLwrOutTexture->m_needsRTV = true;
            }
        }

        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
    }

    MultipassConfigParserError Effect::finalize(Texture*& tex, const wchar_t* effectsFolderPath, const wchar_t*  tempFolderPath, bool calcHashes)
    {
        if (resolveReferences() != MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK))
            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eInternalError, "Failed to resolve references!");

        m_hashesValid = false;

        bool isOk = true;

        sha256_ctx hashContext;

        HRESULT hr;

        D3D11_SAMPLER_DESC sampDesc;
        // Init
        {
            ZeroMemory(&sampDesc, sizeof(sampDesc));
            sampDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
            sampDesc.MinLOD = 0;
            sampDesc.MaxLOD = D3D11_FLOAT32_MAX;
        }

        D3D11_TEXTURE2D_DESC textureDesc;
        // Init
        Tools::ClearD3D11TexDesc(textureDesc);

        D3D11_RENDER_TARGET_VIEW_DESC renderTargetViewDesc;
        // Init
        {
            ZeroMemory(&renderTargetViewDesc, sizeof(renderTargetViewDesc));
            renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
            renderTargetViewDesc.Texture2D.MipSlice = 0;
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

        // Create Samplers
        for (size_t i = 0, iend = m_samplers.size(); i < iend; ++i)
        {
            Sampler & lwrSampler = *m_samplers[i];

            sampDesc.Filter = ircolwert::filterToDXGIFilter(lwrSampler.m_filterMin, lwrSampler.m_filterMag, lwrSampler.m_filterMip);
            sampDesc.AddressU = ircolwert::addressToDXGIAddress(lwrSampler.m_addrU);
            sampDesc.AddressV = ircolwert::addressToDXGIAddress(lwrSampler.m_addrV);
            sampDesc.AddressW = ircolwert::addressToDXGIAddress(lwrSampler.m_addrW);

            hr = m_resourceManager->createSampler(sampDesc, &lwrSampler.m_D3DSampler);

            if (FAILED(hr))
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eInternalError, "Creating D3D sampler failed!");
        }

        // TODO: move that out to members?
        shaderhelpers::IncludeHandler incHandler(effectsFolderPath, &m_includeFilesPool);
        incHandler.setSystemBasePath(effectsFolderPath);

        // D3D11 compilers produce shader bytecode with the following structure:
        //  0-3    "DXBC"  - string "DXBC", always
        //  4-19  *    - HLSL compiler produced checksum, callwlated using proprietary algorithm
        //  20-23  1    - number 1, always
        //  24-27  *    - total size, in bytes, of the compiled shader, including the header
        //  28-31  *    - chunk count
        // But we're possibly compiling from file, so the offset of DXBC struct is not 0,
        //  it varies based on the source filename, so we need to search for it
        auto eraseShaderChecksum = [](uint8_t * shaderBuf, size_t shaderBufSize)
        {
            size_t parsedBytes = 0;
            uint8_t * dxbcPos = shaderBuf;
            bool checksumFound = true;
            while (parsedBytes < shaderBufSize - 21)
            {
                // Check that first four bytes are "DXBC" and that enclosing sequence is 1, 0
                if ((*(dxbcPos  ) == 'D') && (*(dxbcPos+1) == 'X') &&
                    (*(dxbcPos+2) == 'B') && (*(dxbcPos+3) == 'C') &&
                    (*(dxbcPos+20) == 1) && (*(dxbcPos+21) == 0))
                {
                    checksumFound = true;
                    break;
                }
                ++dxbcPos;
                ++parsedBytes;
            }

            // DXBC not found - could it even happen with valid bytecode?
            if (!checksumFound)
                return;

            const uint32_t checksumOffset = 4;
            const uint32_t checksumLastByte = 19;
            memset(dxbcPos+checksumOffset, 0, (checksumLastByte - checksumOffset + 1)*sizeof(uint8_t));
        };
        std::vector<uint8_t> bytecodeToHash;

        // Create VertexShaders
        ID3DBlob* pVSBlob = nullptr;
        if (calcHashes)
        {
            sha256_init(&hashContext);
        }
        for (size_t i = 0, iend = m_vertexShaders.size(); i < iend; ++i)
        {
            VertexShader & lwrVertexShader = *m_vertexShaders[i];

            // Setup initial shader file description ("root header file")
            swprintf_s(lwrVertexShader.m_rootFile.m_filenameFull, IR_FILENAME_MAX, L"%s", lwrVertexShader.m_fileName);
            lwrVertexShader.m_rootFile.computePath();
            incHandler.setRootIncludeFile(&lwrVertexShader.m_rootFile);

            // TODO: add shader compile from memory as well
            // TODO: add optional disabling of shader reflection in the Effect settings
            std::string errorString;
            hr = compileEffectShaderFromFileOrCache(shaderhelpers::ShaderType::kVertex, lwrVertexShader.m_fileName, lwrVertexShader.m_entryPoint,
                        effectsFolderPath, tempFolderPath,
                        &pVSBlob, &incHandler, *m_resourceManager->m_d3dCompiler, &lwrVertexShader.m_D3DReflection, &errorString);

            if (FAILED(hr))
            {
                if (pVSBlob)
                {
                    pVSBlob->Release();
                    pVSBlob = nullptr;
                }

                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eShaderCompilationError, errorString);
            }

            if (calcHashes)
            {
                const size_t bytecodeSize = pVSBlob->GetBufferSize();

                ID3DBlob * disasmBlob;
                PFND3DDISASSEMBLEFUNC pfnD3DDisassembleBlob = m_resourceManager->m_d3dCompiler->getD3DDisassembleFunc();
                pfnD3DDisassembleBlob(pVSBlob->GetBufferPointer(), bytecodeSize, D3D_DISASM_INSTRUCTION_ONLY | D3D_DISASM_DISABLE_DEBUG_INFO, nullptr, &disasmBlob);

                const char * disasmText = (const char *)disasmBlob->GetBufferPointer();
                sha256_update(&hashContext, (uint8_t *)disasmText, (uint32_t)disasmBlob->GetBufferSize());
                disasmBlob->Release();
            }

            hr = m_resourceManager->createVertexShader(pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), nullptr, &lwrVertexShader.m_D3DVertexShader);

            if (pVSBlob)
            {
                pVSBlob->Release();
                pVSBlob = nullptr;
            }

            if (FAILED(hr))
            {
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eInternalError, "Creating D3D vertex shader failed!");
            }
        }
        if (calcHashes)
        {
            sha256_final(&hashContext, m_vsBlobsHash);
        }

        // Create PixelShaders
        ID3DBlob* pPSBlob = nullptr;
        if (calcHashes)
        {
            sha256_init(&hashContext);
        }
        for (size_t i = 0, iend = m_pixelShaders.size(); i < iend; ++i)
        {
            PixelShader & lwrPixelShader = *m_pixelShaders[i];

            // Setup initial shader file description ("root header file")
            swprintf_s(lwrPixelShader.m_rootFile.m_filenameFull, IR_FILENAME_MAX, L"%s", lwrPixelShader.m_fileName);
            lwrPixelShader.m_rootFile.computePath();
            incHandler.setRootIncludeFile(&lwrPixelShader.m_rootFile);

            // TODO: add shader compile from memory as well
            // TODO: add optional disabling of shader reflection in the Effect settings
            std::string errorString;
            hr = compileEffectShaderFromFileOrCache(shaderhelpers::ShaderType::kPixel, lwrPixelShader.m_fileName, lwrPixelShader.m_entryPoint,
                        effectsFolderPath, tempFolderPath,
                        &pPSBlob, &incHandler, *m_resourceManager->m_d3dCompiler, &lwrPixelShader.m_D3DReflection, &errorString);

            if (pPSBlob && calcHashes)
            {
                const size_t bytecodeSize = pPSBlob->GetBufferSize();

                ID3DBlob * disasmBlob;
                PFND3DDISASSEMBLEFUNC pfnD3DDisassembleBlob = m_resourceManager->m_d3dCompiler->getD3DDisassembleFunc();
                pfnD3DDisassembleBlob(pPSBlob->GetBufferPointer(), bytecodeSize, D3D_DISASM_INSTRUCTION_ONLY | D3D_DISASM_DISABLE_DEBUG_INFO, nullptr, &disasmBlob);

                const char * disasmText = (const char *)disasmBlob->GetBufferPointer();
                sha256_update(&hashContext, (uint8_t *)disasmText, (uint32_t)disasmBlob->GetBufferSize());
                disasmBlob->Release();
            }

            if (FAILED(hr))
            {
                if (pPSBlob)
                {
                    pPSBlob->Release();
                    pPSBlob = nullptr;
                }

                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eShaderCompilationError, errorString);
            }

            hr = m_resourceManager->createPixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), nullptr, &lwrPixelShader.m_D3DPixelShader);

            if (pPSBlob)
            {
                pPSBlob->Release();
                pPSBlob = nullptr;
            }

            if (FAILED(hr))
            {
                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eInternalError, "Creating D3D pixel shader failed!");
            }
        }
        if (calcHashes)
        {
            sha256_final(&hashContext, m_psBlobsHash);
        }

        Texture * outputTexture = nullptr;
        Texture * finalTexture = nullptr;  // Pointer to a texture created for the output buffer, in case pass has no outputs specified (typically this is output)

        struct RasterizerStatePair
        {
            D3D11_RASTERIZER_DESC desc;
            ID3D11RasterizerState * statePtr;
        };
        struct DepthStencilStatePair
        {
            D3D11_DEPTH_STENCIL_DESC desc;
            ID3D11DepthStencilState * statePtr;
        };
        struct AlphaBlendStatePair
        {
            D3D11_BLEND_DESC desc;
            ID3D11BlendState * statePtr;
        };
        std::vector<RasterizerStatePair> rasterizerStatePairs;
        std::vector<DepthStencilStatePair> depthStencilStatePairs;
        std::vector<AlphaBlendStatePair> alphaBlendStatePairs;

        for (int bufIdx = 0; bufIdx < (int)Texture::TextureType::kNUM_ENTRIES; ++bufIdx)
        {
            m_inputBufferAvailable[bufIdx] = false;
        }

        // Allocate resources
        m_effect->m_passes.reserve(m_passes.size());
        sha256_ctx texturesHashContext;
        if (calcHashes)
        {
            sha256_init(&texturesHashContext);
        }
        for (size_t passIdx = 0, passesTotal = m_passes.size(); passIdx < passesTotal; ++passIdx)
        {
            Pass & lwrPass = *m_passes[passIdx];

            m_effect->m_passes.push_back(CmdProcPass());
            CmdProcPass & pass = m_effect->m_passes[m_effect->m_passes.size()-1];

            bool lastPass = (passIdx == passesTotal-1);
            bool noOutTex = false;
            if (lwrPass.getDataOutNum() == 0)
            {
                // Special case - pass can have no predefined outputs
                //  typically is an ouput pass, such pass will draw into the special output texture
                noOutTex = true;
            }

            if (lwrPass.m_vertexShader)
            {
                pass.m_vertexShader = lwrPass.m_vertexShader->m_D3DVertexShader;
            }
            else
            {
                // CmdProc will resolve it into default VS
                pass.m_vertexShader = nullptr;
            }
            pass.m_pixelShader = lwrPass.m_pixelShader->m_D3DPixelShader;
            pass.m_width = (float)lwrPass.m_width;
            pass.m_height = (float)lwrPass.m_height;

            // States
            int stateFoundIdx;

            // Try to find matching rasterizer state
            stateFoundIdx = -1;
            D3D11_RASTERIZER_DESC lwrRasterizerStateDesc = CmdProcColwertRasterizerStateD3D11(lwrPass.m_rasterizerState);
            for (int statePairIdx = 0, statePairIdxEnd = (int)rasterizerStatePairs.size(); statePairIdx < statePairIdxEnd; ++statePairIdx)
            {
                const RasterizerStatePair & statePair = rasterizerStatePairs[statePairIdx];
                if (memcmp(&statePair.desc, &lwrRasterizerStateDesc, sizeof(D3D11_RASTERIZER_DESC)) == 0)
                {
                    stateFoundIdx = statePairIdx;
                    break;
                }
            }

            if (stateFoundIdx >= 0)
            {
                // Similar state found, reuse
                pass.m_rasterizerState = rasterizerStatePairs[stateFoundIdx].statePtr;
            }
            else
            {
                // State not found, create new
                if (!SUCCEEDED(hr = m_resourceManager->createRasterizerState(lwrRasterizerStateDesc, &pass.m_rasterizerState)))
                {
                    LOG_FATAL("Effect rasterizer state initialization failed");
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eInternalError, "Creating D3D rasterizer state failed!");
                }
                rasterizerStatePairs.push_back({ lwrRasterizerStateDesc, pass.m_rasterizerState });
            }

            // Try to find matching depth stencil state
            stateFoundIdx = -1;
            D3D11_DEPTH_STENCIL_DESC lwrDepthStencilStateDesc = CmdProcColwertDepthStencilStateD3D11(lwrPass.m_depthStencilState);
            for (int statePairIdx = 0, statePairIdxEnd = (int)depthStencilStatePairs.size(); statePairIdx < statePairIdxEnd; ++statePairIdx)
            {
                const DepthStencilStatePair & statePair = depthStencilStatePairs[statePairIdx];
                if (memcmp(&statePair.desc, &lwrDepthStencilStateDesc, sizeof(D3D11_DEPTH_STENCIL_DESC)) == 0)
                {
                    stateFoundIdx = statePairIdx;
                    break;
                }
            }

            if (stateFoundIdx >= 0)
            {
                // Similar state found, reuse
                pass.m_depthStencilState = depthStencilStatePairs[stateFoundIdx].statePtr;
            }
            else
            {
                // State not found, create new
                if (!SUCCEEDED(hr = m_resourceManager->createDepthStencilState(lwrDepthStencilStateDesc, &pass.m_depthStencilState)))
                {
                    LOG_FATAL("Effect depth stencil state initialization failed");
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eInternalError, "Creating D3D depth stencil state failed!");
                }
                depthStencilStatePairs.push_back({ lwrDepthStencilStateDesc, pass.m_depthStencilState });
            }

            // Try to find matching alpha blend state
            stateFoundIdx = -1;
            D3D11_BLEND_DESC lwrAlphaBlendStateDesc = CmdProcColwertAlphaBlendStateD3D11(lwrPass.m_alphaBlendState);
            for (int statePairIdx = 0, statePairIdxEnd = (int)alphaBlendStatePairs.size(); statePairIdx < statePairIdxEnd; ++statePairIdx)
            {
                const AlphaBlendStatePair & statePair = alphaBlendStatePairs[statePairIdx];
                if (memcmp(&statePair.desc, &lwrAlphaBlendStateDesc, sizeof(D3D11_BLEND_DESC)) == 0)
                {
                    stateFoundIdx = statePairIdx;
                    break;
                }
            }

            if (stateFoundIdx >= 0)
            {
                // Similar state found, reuse
                pass.m_alphaBlendState = alphaBlendStatePairs[stateFoundIdx].statePtr;
            }
            else
            {
                // State not found, create new
                if (!SUCCEEDED(hr = m_resourceManager->createAlphaBlendState(lwrAlphaBlendStateDesc, &pass.m_alphaBlendState)))
                {
                    LOG_FATAL("Effect alpha blend state initialization failed");
                    return MultipassConfigParserError(MultipassConfigParserErrorEnum::eInternalError, "Creating D3D alpha blend state failed!");
                }
                alphaBlendStatePairs.push_back({ lwrAlphaBlendStateDesc, pass.m_alphaBlendState });
            }

            ID3D11ShaderReflection * pixelShaderReflection = lwrPass.m_pixelShader->m_D3DReflection;

            D3D11_SHADER_DESC pixelShaderDesc;
            pixelShaderReflection->GetDesc(&pixelShaderDesc);

            pass.m_samplerDescs.reserve(lwrPass.getSamplersNum());
            for (size_t j = 0, jend = lwrPass.getSamplersNum(); j < jend; ++j)
            {
                int slotNum = lwrPass.getSamplerSlot(j);
                if (slotNum == Pass::BindByName)
                {
                    D3D11_SHADER_INPUT_BIND_DESC bindingDesc;
                    hr = pixelShaderReflection->GetResourceBindingDescByName(lwrPass.getSamplerName(j), &bindingDesc);

                    if (FAILED(hr))
                    {
                        std::wstring wsFilename(lwrPass.m_pixelShader->m_fileName);
                        std::string strFilename(darkroom::getUtf8FromWstr(lwrPass.m_pixelShader->m_fileName));
                        std::stringbuf buf;
                        std::ostream strstream(&buf);
                        strstream << lwrPass.getSamplerName(j) << " from file " << strFilename << " with entry point " << lwrPass.m_pixelShader->m_entryPoint;
                        
                        slotNum = IR_REFLECTION_NOT_FOUND;
                        // TODO: replace with warnings interface
                        //return MultipassConfigParserError(MultipassConfigParserErrorEnum::eReflectionFailed, buf.str());

                        // TODO[error]: throw an error here and think what to do in this case (likely texture get optimized out or misspelling - suggest to user)
                        //    we shouldn't stop there, as it could hapen even in valid effect
                    }
                    else
                    {
                        slotNum = bindingDesc.BindPoint;
                    }
                }
                pass.m_samplerDescs.push_back({lwrPass.getSampler(j)->m_D3DSampler, slotNum});
            }

            pass.m_constantBufPSDescs.reserve(lwrPass.getConstBufsPSNum());
            for (size_t j = 0, jend = lwrPass.getConstBufsPSNum(); j < jend; ++j)
            {
                ConstantBuf * pLwrConstantBuf = lwrPass.getConstBufPS(j);

                int slotNum = lwrPass.getConstBufPSSlot(j);

                if (slotNum == Pass::BindByName)
                {
                    D3D11_SHADER_INPUT_BIND_DESC bindingDesc;
                    hr = pixelShaderReflection->GetResourceBindingDescByName(lwrPass.getConstBufPSName(j), &bindingDesc);

                    if (FAILED(hr))
                    {
                        std::wstring wsFilename(lwrPass.m_pixelShader->m_fileName);
                        std::string strFilename(darkroom::getUtf8FromWstr(lwrPass.m_pixelShader->m_fileName));
                        std::stringbuf buf;
                        std::ostream strstream(&buf);
                        strstream << lwrPass.getConstBufPSName(j) << " from file [PS] " << strFilename << " with entry point " << lwrPass.m_pixelShader->m_entryPoint;

                        slotNum = IR_REFLECTION_NOT_FOUND;
                        // TODO: replace with warnings interface
                        //return MultipassConfigParserError(MultipassConfigParserErrorEnum::eReflectionFailed, buf.str());

                        // TODO[error]: throw an error here and think what to do in this case (likely texture get optimized out or misspelling - suggest to user)
                        //    we shouldn't stop there, as it could hapen even in valid effect.     feodob - why?
                    }
                    else
                    {
                        slotNum = bindingDesc.BindPoint;
                    }
                }

                // At this point, pLwrConstantBuf->m_D3DBuffer could be nullptr, but this is fine - it would be fixed later
                pass.m_constantBufPSDescs.push_back(CmdProcConstantBufDesc(nullptr, slotNum));

                if (slotNum != IR_REFLECTION_NOT_FOUND)
                {
                    ID3D11ShaderReflection * shaderReflection = lwrPass.m_pixelShader->m_D3DReflection;
                    ID3D11ShaderReflectionConstantBuffer * cbufferReflection = nullptr;

                    CmdProcConstantBufDesc & cmdConstantBuf = pass.m_constantBufPSDescs[pass.m_constantBufPSDescs.size() - 1];

                    size_t maxOffset = 0, maxOffset_Size = 0;

                    cmdConstantBuf.m_constants.reserve(pLwrConstantBuf->m_constants.size());
                    for (size_t k = 0, kend = pLwrConstantBuf->m_constants.size(); k < kend; ++k)
                    {
                        CmdProcConstHandle constHandle;

                        const Constant * pLwrConstant = pLwrConstantBuf->m_constants[k];

                        CmdProcConstDataType constDataType = CmdProcConstDataType::kNUM_ENTRIES;

                        unsigned int numElements = 0;
                        if (pLwrConstant->m_type == ConstType::kNUM_ENTRIES)
                        {
                            const UserConstant* uc = m_userConstantManager.findByName(pLwrConstant->m_userConstName);

                            if (!uc)
                            {
                                std::wstring wsFilename(lwrPass.m_pixelShader->m_fileName);
                                std::string strFilename(darkroom::getUtf8FromWstr(lwrPass.m_pixelShader->m_fileName));
                                std::stringbuf buf;
                                std::ostream strstream(&buf);
                                strstream << pLwrConstant->m_userConstName << " bound to " << pLwrConstant->m_constBindName << " from file [PS] " << strFilename << " with entry point " << lwrPass.m_pixelShader->m_entryPoint;

                                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eBadUserConstant, buf.str());
                            }

                            UserConstDataType uct = uc->getType();
                            constDataType = ircolwert::userConstTypeToCmdProcConstElementDataType(uct);
                            numElements = uc->getDefaultValue().getDimensionality();
                            constHandle = makeCmdProcConstHandle(uc->getUid());
                        }
                        else
                        {
                            CmdProcSystemConst sc = ircolwert::constTypeToCmdProcSystemConst(pLwrConstant->m_type);
                            constDataType = getCmdProcSystemConstElementDataType(sc);
                            numElements = getCmdProcSystemConstDimensions(sc);
                            constHandle = makeCmdProcConstHandle(sc);
                        }

                        size_t constSize = numElements*getCmdProcConstDataElementTypeSize(constDataType);
                        int constantOffsetInComponents = pLwrConstant->m_constantOffsetInComponents;

                        if (constantOffsetInComponents == Constant::OffsetAuto)
                        {
                            bool reflectionFailed = true;

                            if (cbufferReflection == nullptr)
                                cbufferReflection = shaderReflection->GetConstantBufferByName(lwrPass.getConstBufPSName(j));

                            if (cbufferReflection)
                            {
                                ID3D11ShaderReflectiolwariable * constantReflection = cbufferReflection->GetVariableByName(pLwrConstant->m_constBindName);
                                if (constantReflection)
                                {
                                    D3D11_SHADER_VARIABLE_DESC constantDesc;
                                    hr = constantReflection->GetDesc(&constantDesc);

                                    if (SUCCEEDED(hr))
                                    {
                                        constantOffsetInComponents = constantDesc.StartOffset / sizeof(float);  // This offset should be in vector components (float) rather than bytes
                                        reflectionFailed = false;
                                    }
                                }
                            }

                            if (reflectionFailed)
                            {
                                std::wstring wsFilename(lwrPass.m_pixelShader->m_fileName);
                                std::string strFilename(darkroom::getUtf8FromWstr(lwrPass.m_pixelShader->m_fileName));
                                std::stringbuf buf;
                                std::ostream strstream(&buf);
                                strstream << pLwrConstant->m_constBindName << " from file [PS] " << strFilename << " with entry point " << lwrPass.m_pixelShader->m_entryPoint;

                                // TODO: constants shouldn't actually be lacking from the constant buffer, so if that happens - it is definitely an error?
                                //    actually there could be a situation when reflection failed for one constant within the constant buffer
                                //    that probably shouold result in different layout creation
                                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eReflectionFailed, buf.str());
                            }
                        }

                        if (maxOffset + maxOffset_Size <= constantOffsetInComponents * sizeof(float)+constSize)
                        {
                            maxOffset = constantOffsetInComponents * sizeof(float);
                            maxOffset_Size = constSize;
                        }

                        // TODO: put this CmdProc type inside ir::Constant
                        cmdConstantBuf.m_constants.push_back({ constHandle, constantOffsetInComponents });
                    }

                    ID3D11Buffer* d3dbuf = nullptr;

                    for (auto layout : pLwrConstantBuf->m_layouts)
                    {
                        if (std::equal(layout.m_constantsAndOffsets.cbegin(), layout.m_constantsAndOffsets.cend(), cmdConstantBuf.m_constants.cbegin(),
                                        [](const ConstantBuf::Layout::ConstantDesc & a, const CmdProcConstantDesc & b)
                                        {
                                            return a.m_constHandle == b.constHandle && a.m_offsetInComponents == b.offsetInComponents;
                                        }
                            ))
                        {
                            d3dbuf = layout.m_D3DBuf;
                            break;
                        }
                    }

                    // Create constant buffer if needed
                    if (d3dbuf == nullptr)
                    {
                        // Should be in multiples of 16
                        constBufDesc.ByteWidth = ((maxOffset + maxOffset_Size) + 15) & ~15;

                        hr = m_resourceManager->createConstantBuffer(constBufDesc, NULL, &d3dbuf);

                        if (FAILED(hr))
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eInternalError, "Creating D3D consant buffer failed!");

                        pLwrConstantBuf->m_layouts.push_back(ConstantBuf::Layout());
                        ConstantBuf::Layout& newLayout = pLwrConstantBuf->m_layouts.back();
                        newLayout.m_constantsAndOffsets.reserve(cmdConstantBuf.m_constants.size());

                        for (auto& c : cmdConstantBuf.m_constants)
                            newLayout.m_constantsAndOffsets.push_back({ c.constHandle, c.offsetInComponents });

                        newLayout.m_D3DBuf = d3dbuf;
                    }

                    cmdConstantBuf.m_pBuffer = d3dbuf;
                }
            }

            pass.m_constantBufVSDescs.reserve(lwrPass.getConstBufsVSNum());
            for (size_t j = 0, jend = lwrPass.getConstBufsVSNum(); j < jend; ++j)
            {
                ConstantBuf * pLwrConstantBuf = lwrPass.getConstBufVS(j);

                int slotNum = lwrPass.getConstBufVSSlot(j);

                if (slotNum == Pass::BindByName)
                {
                    ID3D11ShaderReflection * vertexShaderReflection = lwrPass.m_vertexShader->m_D3DReflection;

                    D3D11_SHADER_INPUT_BIND_DESC bindingDesc;
                    hr = vertexShaderReflection->GetResourceBindingDescByName(lwrPass.getConstBufVSName(j), &bindingDesc);

                    if (FAILED(hr))
                    {
                        std::wstring wsFilename(lwrPass.m_vertexShader->m_fileName);
                        std::string strFilename(darkroom::getUtf8FromWstr(lwrPass.m_vertexShader->m_fileName));
                        std::stringbuf buf;
                        std::ostream strstream(&buf);
                        strstream << lwrPass.getConstBufVSName(j) << " from file [VS] " << strFilename << " with entry point " << lwrPass.m_vertexShader->m_entryPoint;

                        slotNum = IR_REFLECTION_NOT_FOUND;
                        // TODO: replace with warnings interface
                        //return MultipassConfigParserError(MultipassConfigParserErrorEnum::eReflectionFailed, buf.str());

                        // TODO[error]: throw an error here and think what to do in this case (likely texture get optimized out or misspelling - suggest to user)
                        //    we shouldn't stop there, as it could hapen even in valid effect.     feodob - why?
                    }
                    else
                    {
                        slotNum = bindingDesc.BindPoint;
                    }
                }

                // At this point, pLwrConstantBuf->m_D3DBuffer could be nullptr, but this is fine - it would be fixed later
                pass.m_constantBufVSDescs.push_back(CmdProcConstantBufDesc(nullptr, slotNum));

                if (slotNum != IR_REFLECTION_NOT_FOUND)
                {
                    ID3D11ShaderReflection * shaderReflection = lwrPass.m_vertexShader->m_D3DReflection;
                    ID3D11ShaderReflectionConstantBuffer * cbufferReflection = nullptr;

                    CmdProcConstantBufDesc & cmdConstantBuf = pass.m_constantBufVSDescs[pass.m_constantBufVSDescs.size() - 1];

                    size_t maxOffset = 0, maxOffset_Size = 0;

                    cmdConstantBuf.m_constants.reserve(pLwrConstantBuf->m_constants.size());
                    for (size_t k = 0, kend = pLwrConstantBuf->m_constants.size(); k < kend; ++k)
                    {
                        CmdProcConstHandle constHandle;

                        const Constant * pLwrConstant = pLwrConstantBuf->m_constants[k];

                        CmdProcConstDataType constDataType = CmdProcConstDataType::kNUM_ENTRIES;

                        unsigned int numElements = 0;
                        if (pLwrConstant->m_type == ConstType::kNUM_ENTRIES)
                        {
                            const UserConstant* uc = m_userConstantManager.findByName(pLwrConstant->m_userConstName);

                            if (!uc)
                            {
                                std::wstring wsFilename(lwrPass.m_vertexShader->m_fileName);
                                std::string strFilename(darkroom::getUtf8FromWstr(lwrPass.m_vertexShader->m_fileName));
                                std::stringbuf buf;
                                std::ostream strstream(&buf);
                                strstream << pLwrConstant->m_userConstName << " bound to " << pLwrConstant->m_constBindName << " from file " << strFilename << " with entry point " << lwrPass.m_vertexShader->m_entryPoint;

                                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eBadUserConstant, buf.str());
                            }

                            UserConstDataType uct = uc->getType();
                            constDataType = ircolwert::userConstTypeToCmdProcConstElementDataType(uct);
                            numElements = uc->getDefaultValue().getDimensionality();
                            constHandle = makeCmdProcConstHandle(uc->getUid());
                        }
                        else
                        {
                            CmdProcSystemConst sc = ircolwert::constTypeToCmdProcSystemConst(pLwrConstant->m_type);
                            constDataType = getCmdProcSystemConstElementDataType(sc);
                            numElements = getCmdProcSystemConstDimensions(sc);
                            constHandle = makeCmdProcConstHandle(sc);
                        }

                        size_t constSize = numElements*getCmdProcConstDataElementTypeSize(constDataType);
                        int constantOffsetInComponents = pLwrConstant->m_constantOffsetInComponents;

                        if (constantOffsetInComponents == Constant::OffsetAuto)
                        {
                            bool reflectionFailed = true;

                            if (cbufferReflection == nullptr)
                                cbufferReflection = shaderReflection->GetConstantBufferByName(lwrPass.getConstBufVSName(j));

                            if (cbufferReflection)
                            {
                                ID3D11ShaderReflectiolwariable * constantReflection = cbufferReflection->GetVariableByName(pLwrConstant->m_constBindName);
                                if (constantReflection)
                                {
                                    D3D11_SHADER_VARIABLE_DESC constantDesc;
                                    hr = constantReflection->GetDesc(&constantDesc);

                                    if (SUCCEEDED(hr))
                                    {
                                        constantOffsetInComponents = constantDesc.StartOffset / sizeof(float);  // This offset should be in vector components (float) rather than bytes
                                        reflectionFailed = false;
                                    }
                                }
                            }

                            if (reflectionFailed)
                            {
                                std::wstring wsFilename(lwrPass.m_vertexShader->m_fileName);
                                std::string strFilename(darkroom::getUtf8FromWstr(lwrPass.m_vertexShader->m_fileName));
                                std::stringbuf buf;
                                std::ostream strstream(&buf);
                                strstream << pLwrConstant->m_constBindName << " from file " << strFilename << " with entry point " << lwrPass.m_vertexShader->m_entryPoint;

                                // TODO: constants shouldn't actually be lacking from the constant buffer, so if that happens - it is definitely an error?
                                //    actually there could be a situation when reflection failed for one constant within the constant buffer
                                //    that probably shouold result in different layout creation
                                return MultipassConfigParserError(MultipassConfigParserErrorEnum::eReflectionFailed, buf.str());
                            }
                        }

                        if (maxOffset + maxOffset_Size <= constantOffsetInComponents * sizeof(float)+constSize)
                        {
                            maxOffset = constantOffsetInComponents * sizeof(float);
                            maxOffset_Size = constSize;
                        }

                        // TODO: put this CmdProc type inside ir::Constant
                        cmdConstantBuf.m_constants.push_back({ constHandle, constantOffsetInComponents });
                    }

                    ID3D11Buffer* d3dbuf = nullptr;

                    for (auto layout : pLwrConstantBuf->m_layouts)
                    {
                        if (std::equal(layout.m_constantsAndOffsets.cbegin(), layout.m_constantsAndOffsets.cend(), cmdConstantBuf.m_constants.cbegin(),
                                        [](const ConstantBuf::Layout::ConstantDesc & a, const CmdProcConstantDesc & b)
                                        {
                                            return a.m_constHandle == b.constHandle && a.m_offsetInComponents == b.offsetInComponents;
                                        }
                            ))
                        {
                            d3dbuf = layout.m_D3DBuf;
                            break;
                        }
                    }

                    // Create constant buffer if needed
                    if (d3dbuf == nullptr)
                    {
                        // Should be in multiples of 16
                        constBufDesc.ByteWidth = ((maxOffset + maxOffset_Size) + 15) & ~15;

                        hr = m_resourceManager->createConstantBuffer(constBufDesc, NULL, &d3dbuf);

                        if (FAILED(hr))
                            return MultipassConfigParserError(MultipassConfigParserErrorEnum::eInternalError, "Creating D3D consant buffer failed!");

                        pLwrConstantBuf->m_layouts.push_back(ConstantBuf::Layout());
                        ConstantBuf::Layout& newLayout = pLwrConstantBuf->m_layouts.back();
                        newLayout.m_constantsAndOffsets.reserve(cmdConstantBuf.m_constants.size());

                        for (auto& c : cmdConstantBuf.m_constants)
                            newLayout.m_constantsAndOffsets.push_back({ c.constHandle, c.offsetInComponents });

                        newLayout.m_D3DBuf = d3dbuf;
                    }

                    cmdConstantBuf.m_pBuffer = d3dbuf;
                }
            }

            // TODO: go through dataOut RTVs and see if there are dangling RTVs (link to dummy one created specifically for these purposes)
            // Binding Render Targets in-order
            pass.m_renderTargets.reserve(lwrPass.getMRTChannelsTotal());
            for (size_t j = 0, jend = lwrPass.getMRTChannelsTotal(); j < jend; ++j)
            {
                // TODO: presort+pad and change from searching to O(1) indexing

                if (!noOutTex)
                {
                    bool mrtChannelFound = false;
                    for (size_t jn = 0, jnend = lwrPass.getDataOutNum(); jn < jnend; ++jn)
                    {
                        if (lwrPass.getMRTChannelOut(jn) == j)
                        {
                            Texture * pLwrTexture = lwrPass.getDataOut(jn);

                            // Initialize texture resource if needed
                            if (pLwrTexture->m_D3DTexture == nullptr)
                            {
                                if (pLwrTexture->m_initData && calcHashes)
                                {
                                    size_t bytesPerPixel = ircolwert::formatBitsPerPixel(ircolwert::formatToDXGIFormat(pLwrTexture->m_format)) / 8;
                                    size_t initDataSize = pLwrTexture->m_width * pLwrTexture->m_height * bytesPerPixel*sizeof(unsigned char);
                                    sha256_update(&texturesHashContext, reinterpret_cast<uint8_t*>(pLwrTexture->m_initData), (uint32_t)initDataSize);
                                }
                                hr = initTextureResource(pLwrTexture, textureDesc, renderTargetViewDesc, shaderResourceViewDesc);
                                // TODO[error]: throw an error here in case of failure
                            }

                            // In case it's last pass - fill in the output tex
                            outputTexture = pLwrTexture;

                            // Bind texture RTV to the current pass if needed
                            if (pLwrTexture->m_needsRTV)
                            {
                                pass.m_width = (float)pLwrTexture->m_width;
                                pass.m_height = (float)pLwrTexture->m_height;
                                pass.m_renderTargets.push_back(pLwrTexture->m_D3DRTV);
                            }
                            else
                            {
                                // TODO[error]: throw an error here
                                isOk = false;
                                break;
                            }

                            mrtChannelFound = true;
                            break;
                        }
                    }

                    if (!mrtChannelFound)
                    {
                        // MRT channel not referenced - throw an error or warning
                        // TODO: prevent FAILURE(?) due to not referenced MRT channels
                        pass.m_renderTargets.push_back(nullptr);
                    }
                }
                else
                {
                    // Pass, with no out texture specified
                    // Typically, output pass; however there still could be work done after the output pass, e.g. for persistent textures
                    if (finalTexture == nullptr)
                    {
                        Texture * newTexture = createRTTexture(lwrPass.m_width, lwrPass.m_height, lwrPass.getMRTChannelFormat(j));
                        newTexture->m_needsRTV = true;
                        // No need to hash pass RenderTarget texture
                        hr = initTextureResource(newTexture, textureDesc, renderTargetViewDesc, shaderResourceViewDesc);
                        finalTexture = newTexture;
                    }
                    pass.m_renderTargets.push_back(finalTexture->m_D3DRTV);

                    outputTexture = finalTexture;
                }
            }

            // Create input resourcess for the pass, if needed
            pass.m_shaderResourceDescs.reserve(lwrPass.getDataSrcNum());
            for (size_t j = 0, jend = lwrPass.getDataSrcNum(); j < jend; ++j)
            {
                if (lwrPass.getDataSrc(j)->getDataType() == DataSource::DataType::kPass)
                {
                    // TODO[error]: throw an error here
                    isOk = false;
                    break;
                } else if (lwrPass.getDataSrc(j)->getDataType() == DataSource::DataType::kTexture)
                {
                    Texture * pLwrTexture = static_cast<Texture *>(lwrPass.getDataSrc(j));

                    // Initialize texture resource if needed
                    if (pLwrTexture->m_D3DTexture == nullptr)
                    {
                        if (pLwrTexture->m_initData && calcHashes && !pLwrTexture->m_excludeHash)
                        {
                            size_t bytesPerPixel = ircolwert::formatBitsPerPixel(ircolwert::formatToDXGIFormat(pLwrTexture->m_format)) / 8;
                            size_t initDataSize = pLwrTexture->m_width * pLwrTexture->m_height * bytesPerPixel*sizeof(unsigned char);
                            sha256_update(&texturesHashContext, reinterpret_cast<uint8_t*>(pLwrTexture->m_initData), (uint32_t)initDataSize);
                        }
                        hr = initTextureResource(pLwrTexture, textureDesc, renderTargetViewDesc, shaderResourceViewDesc);
                    }

                    // Bind texture SRV to the current pass if needed
                    if (pLwrTexture->m_needsSRV)
                    {
                        int slotNum = lwrPass.getSlotSrc(j);
                        if (slotNum == Pass::BindByName)
                        {
                            //D3D11_SHADER_DESC shaderDesc;
                            //pixelShaderReflection->GetDesc(&shaderDesc);
                            D3D11_SHADER_INPUT_BIND_DESC bindingDesc;
                            hr = pixelShaderReflection->GetResourceBindingDescByName(lwrPass.getNameSrc(j), &bindingDesc);

                            if (FAILED(hr))
                            {
                                std::wstring wsFilename(lwrPass.m_pixelShader->m_fileName);
                                std::string strFilename(darkroom::getUtf8FromWstr(lwrPass.m_pixelShader->m_fileName));
                                std::stringbuf buf;
                                std::ostream strstream(&buf);
                                strstream << lwrPass.getNameSrc(j) << " from file " << strFilename << " with entry point " << lwrPass.m_pixelShader->m_entryPoint;

                                slotNum = IR_REFLECTION_NOT_FOUND;
                                // TODO: replace with warnings interface
                                //return MultipassConfigParserError(MultipassConfigParserErrorEnum::eReflectionFailed, buf.str());
                                // TODO[error]: throw an error here and think what to do in this case (likely texture get optimized out or misspelling - suggest to user)
                                //    we shouldn't stop there, as it could hapen even in valid effect -- why?
                            }
                            else
                            {
                                Texture::TextureType texType = pLwrTexture->getTextureType();
                                int texTypeInt = (int)texType;
                                if ((texTypeInt >= (int)Texture::TextureType::kFIRST_INPUT) &&
                                    (texTypeInt <= (int)Texture::TextureType::kLAST_INPUT))
                                {
                                    m_inputBufferAvailable[texTypeInt] = true;
                                }

                                slotNum = bindingDesc.BindPoint;
                            }
                        }
                        else
                        {
                            // Check if texture resource is really present
                            // could limit that to only internal buffers to avoid overhead
                            // but at the moment used for each and every buffer

                            D3D11_SHADER_INPUT_BIND_DESC bindingDesc;

                            bool resourceFound = false;
                            for (uint32_t i = 0; i < pixelShaderDesc.BoundResources; ++i)
                            {
                                hr = pixelShaderReflection->GetResourceBindingDesc(i, &bindingDesc);
                                if (!FAILED(hr) && (bindingDesc.BindPoint == slotNum) && (bindingDesc.Type == D3D_SIT_TEXTURE))
                                {
                                    resourceFound = true;
                                    break;
                                }
                            }

                            if (!resourceFound)
                            {
                                // Resouirce is not really used in the shader
                                slotNum = IR_REFLECTION_NOT_FOUND;
                            }
                            else
                            {
                                Texture::TextureType texType = pLwrTexture->getTextureType();
                                int texTypeInt = (int)texType;
                                if ((texTypeInt >= (int)Texture::TextureType::kFIRST_INPUT) &&
                                    (texTypeInt <= (int)Texture::TextureType::kLAST_INPUT))
                                {
                                    m_inputBufferAvailable[texTypeInt] = true;
                                }
                            }
                        }

                        pass.m_shaderResourceDescs.push_back({pLwrTexture->m_D3DSRV, slotNum, colwertTextureTypeToResourceKind(pLwrTexture->getTextureType())});
                    }
                    else
                    {
                        // TODO[error]: throw an error here
                        isOk = false;
                        break;
                    }
                }
            }

            // Put textures that are no longer useful into the free textures pool
            // TODO: potentially this could be unified with the previous loop, but needs additional guarantees that textures
            //    on the same pass will not be re-used
            if (m_allowReuse)
            {
                for (size_t j = 0, jend = lwrPass.getDataSrcNum(); j < jend; ++j)
                {
                    if (lwrPass.getDataSrc(j)->getDataType() == DataSource::DataType::kTexture)
                    {
                        Texture * pLwrTexture = static_cast<Texture *>(lwrPass.getDataSrc(j));
                        bool isPersistent = (pLwrTexture->m_firstPassReadIdx <= pLwrTexture->m_firstPassWriteIdx);
                        if (!isPersistent && pLwrTexture->isReuseAllowed() && pLwrTexture->m_lastPassReadIdx == passIdx)
                            m_freeTexturesList.push_front(pLwrTexture);
                    }
                }
            }
        }
        if (calcHashes)
        {
            sha256_final(&texturesHashContext, m_texturesHash);
        }

        tex = finalTexture ? finalTexture : outputTexture;

        if (calcHashes)
        {
            m_hashesValid = true;
        }

        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
    }
    
    void Effect::setReusePolicy(bool allowReuse)
    {
        m_allowReuse = allowReuse;
    }

}
}
