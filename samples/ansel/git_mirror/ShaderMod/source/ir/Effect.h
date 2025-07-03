#pragma once

#include "Constant.h"
#include "Pass.h"
#include "PixelShader.h"
#include "Sampler.h"
#include "ShaderHelpers.h"
#include "SpecializedPool.h"
#include "Texture.h"
#include "UserConstantManager.h"
#include "VertexShader.h"
#include "D3D11CommandProcessor.h"
#include "MultipassConfigParserError.h"

#include <list>
#include <vector>

class CmdProcEffect;
class ResourceManager;

namespace shadermod
{
namespace ir
{
    class UserConstantManager;
    
    class Effect
    {
    public:

        struct InputData
        {
            unsigned int width, height;
            ir::FragmentFormat format;
            ID3D11Texture2D * texture;
        };

        bool m_inputBufferAvailable[(int)Texture::TextureType::kNUM_ENTRIES];
        // NOTE: values for anything other than kInput* are undefined
        bool isBufferAvailable(Texture::TextureType bufInputType)
        {
            int bufTypeInt = (int)bufInputType;
            if ((bufTypeInt >= (int)Texture::TextureType::kFIRST_INPUT) &&
                (bufTypeInt <= (int)Texture::TextureType::kLAST_INPUT))
            {
                return m_inputBufferAvailable[bufTypeInt];
            }

            return false;
        }

        Effect::Effect(CmdProcEffect * effect, ResourceManager * resourceManager);
        virtual ~Effect();

        void Effect::reset();
        void Effect::reserve();

        ShaderResourceKind colwertTextureTypeToResourceKind(const Texture::TextureType & texType)
        {
            switch (texType)
            {
            case Texture::TextureType::kInputColor:
                return ShaderResourceKind::kColor;
            case Texture::TextureType::kInputDepth:
                return ShaderResourceKind::kDepth;
            case Texture::TextureType::kInputHDR:
                return ShaderResourceKind::kHDR;
            case Texture::TextureType::kInputHUDless:
                return ShaderResourceKind::kHUDless;
            case Texture::TextureType::kInputColorBase:
                return ShaderResourceKind::kColorBase;
            default:
                return (ShaderResourceKind)0;
            };
        }
        Texture::TextureType colwertResourceKindToTextureType(const ShaderResourceKind & resourceKind)
        {
            switch (resourceKind)
            {
            case ShaderResourceKind::kColor:
                return Texture::TextureType::kInputColor;
            case ShaderResourceKind::kDepth:
                return Texture::TextureType::kInputDepth;
            case ShaderResourceKind::kHDR:
                return Texture::TextureType::kInputHDR;
            case ShaderResourceKind::kHUDless:
                return Texture::TextureType::kInputHUDless;
            case ShaderResourceKind::kColorBase:
                return Texture::TextureType::kInputColorBase;
            default:
                return Texture::TextureType::kNUM_ENTRIES;
            };
        }

        FragmentFormat getInputColorFormat() const;
        FragmentFormat getInputDepthFormat() const;
        FragmentFormat getInputHUDlessFormat() const;
        FragmentFormat getInputHDRFormat() const;
        FragmentFormat getInputColorBaseFormat() const;

        int getInputWidth() const;
        int getInputHeight() const;
        int getInputDepthWidth() const;
        int getInputDepthHeight() const;
        int getInputHUDlessWidth() const;
        int getInputHUDlessHeight() const;
        int getInputHDRWidth() const;
        int getInputHDRHeight() const;
        int getInputColorBaseWidth() const;
        int getInputColorBaseHeight() const;

        ID3D11Texture2D* getInputColorTexture() const;
        ID3D11Texture2D* getInputDepthTexture() const;
        ID3D11Texture2D* getInputHUDlessTexture() const;
        ID3D11Texture2D* getInputHDRTexture() const;
        ID3D11Texture2D* getInputColorBaseTexture() const;
        bool isInputColorTextureSet() const;
        bool isInputDepthTextureSet() const;
        bool isInputHUDlessTextureSet() const;
        bool isInputHDRTextureSet() const;
        bool isInputColorBaseTextureSet() const;

        void Effect::setInputs(const InputData & colorInput, const InputData & depthInput, const InputData & hudlessInput, const InputData & hdrInput, const InputData & colorBaseInput);

        // WARNING! This method should only be called if the only thing that changed is ID3D11Texture2D resource
        //  if anything else changed (size, format) - re-finalization is required
        void fixInputs();

        Sampler * createSampler(AddressType addrU, AddressType addrV, FilterType filterMin, FilterType filterMag, FilterType filterMip);

        Constant * createConstant(ConstType type, int packOffsetInComponents);
        Constant * createConstant(ConstType type, const char* bindName);
        Constant * createConstant(const char* userConstName, int packOffsetInComponents);
        Constant * createConstant(const char* userConstName, const char* bindName);

        ConstantBuf * createConstantBuffer();

        Texture * createTextureFromMemory(int width, int height, FragmentFormat format, const void * mem, bool skipLoading = false);
        Texture * createTextureFromFile(int width, int height, FragmentFormat & formatCanChange, const wchar_t * path, bool skipLoading = false, bool excludeHash = false);
        Texture * createNoiseTexture(int width, int height, FragmentFormat format);
        Texture * createRTTexture(int width, int height, FragmentFormat format);
        Texture * createInputColor();
        Texture * createInputDepth();
        Texture * createInputHUDless();
        Texture * createInputHDR();
        Texture * createInputColorBase();

        VertexShader * createVertexShader(const wchar_t * fileName, const char * entryPoint);
        PixelShader * createPixelShader(const wchar_t * fileName, const char * entryPoint);

        Pass * createPass(VertexShader * vertexShader, PixelShader * pixelShader, int width, int height, FragmentFormat * mrtFormats, int numMRTChannels = 1);
        Pass * addPass(Pass * pass);

        HRESULT Effect::initTextureViews(Texture * pTexture, D3D11_TEXTURE2D_DESC & textureDesc, D3D11_RENDER_TARGET_VIEW_DESC & renderTargetViewDesc, D3D11_SHADER_RESOURCE_VIEW_DESC & shaderResourceViewDesc);
        HRESULT initTextureResource(Texture * pTexture, D3D11_TEXTURE2D_DESC & textureDesc, D3D11_RENDER_TARGET_VIEW_DESC & renderTargetViewDesc, D3D11_SHADER_RESOURCE_VIEW_DESC & shaderResourceViewDesc);

        MultipassConfigParserError resolveReferences();

        static const size_t c_hashKeySizeBytes = 32;
        bool m_hashesValid = false;
        uint8_t m_vsBlobsHash[c_hashKeySizeBytes];
        uint8_t m_psBlobsHash[c_hashKeySizeBytes];
        uint8_t m_texturesHash[c_hashKeySizeBytes];

        MultipassConfigParserError finalize(Texture*&, const wchar_t* effectsFolderPath, const wchar_t*  tempFolderPath, bool calcHashes);

        void setReusePolicy(bool allowReuse);

        const UserConstantManager& getUserConstantManager() const { return m_userConstantManager; } 
        UserConstantManager& getUserConstantManager() { return m_userConstantManager; }
        
        std::vector<Sampler *>      m_samplers;
        std::vector<Constant *>     m_constants;
        std::vector<ConstantBuf *>  m_constantBufs;
        std::vector<Texture *>      m_textures;
        std::vector<VertexShader *> m_vertexShaders;
        std::vector<PixelShader *>  m_pixelShaders;
        std::vector<Pass *>         m_passes;

        std::list<Texture *>        m_freeTexturesList;

        CmdProcEffect *             m_effect;
        ResourceManager *           m_resourceManager;

        int                         m_inputWidth = -1;
        int                         m_inputHeight = -1;
        int                         m_inputDepthWidth = -1;
        int                         m_inputDepthHeight = -1;
        int                         m_inputHUDlessWidth = -1;
        int                         m_inputHUDlessHeight = -1;
        int                         m_inputHDRWidth = -1;
        int                         m_inputHDRHeight = -1;
        int                         m_inputColorBaseWidth = -1;
        int                         m_inputColorBaseHeight = -1;
        ID3D11Texture2D *           m_D3DColorTexture = nullptr;
        ID3D11Texture2D *           m_D3DDepthTexture = nullptr;
        ID3D11Texture2D *           m_D3DHUDlessTexture = nullptr;
        ID3D11Texture2D *           m_D3DHDRTexture = nullptr;
        ID3D11Texture2D *           m_D3DColorBaseTexture = nullptr;

    protected:

        bool                        m_allowReuse = true;

        std::vector<int>            m_vertexShadersTraverseStack;
        std::vector<int>            m_pixelShadersTraverseStack;

        Pool<ResourceName>          m_namesPool;

        Pool<shaderhelpers::IncludeFileDesc> m_includeFilesPool;

        Pool<Sampler>               m_samplersPool;
        Pool<Constant>              m_constantsPool;
        Pool<ConstantBuf>           m_constantBufsPool;
        Pool<Texture>               m_texturesPool;
        Pool<VertexShader>          m_vertexShadersPool;
        Pool<PixelShader>           m_pixelShadersPool;
        Pool<Pass>                  m_passesPool;

        FragmentFormat              m_inputColorFormat = FragmentFormat::kBGRA8_uint;
        FragmentFormat              m_inputDepthFormat = FragmentFormat::kD24S8;
        FragmentFormat              m_inputHUDlessFormat = FragmentFormat::kBGRA8_uint;
        FragmentFormat              m_inputHDRFormat = FragmentFormat::kRGBA32_fp;
        FragmentFormat              m_inputColorBaseFormat = FragmentFormat::kBGRA8_uint;

        UserConstantManager         m_userConstantManager;

    };

}
}
