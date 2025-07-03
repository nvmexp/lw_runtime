#pragma once

#include "DataSource.h"
#include "TypeEnums.h"

#include <string>

namespace shadermod
{
namespace ir
{

    // TODO: add proper const get/setters
    class Texture : public DataSource
    {
    public:

        static const int SetAsInputFileSize = -1;

        virtual DataType getDataType() const override { return DataType::kTexture; }

        enum class TextureType
        {
            kRenderTarget,
            kFromFile,
            kInputColor,
            kInputDepth,
            kInputHUDless,
            kInputHDR,
            kInputColorBase,
            kNoise,

            kNUM_ENTRIES,
            kFIRST_INPUT = kInputColor,
            kLAST_INPUT = kInputColorBase
        };

        enum class TextureSizeBase
        {
            kOne,
            kColorBufferWidth,
            kColorBufferHeight,
            kDepthBufferWidth,
            kDepthBufferHeight,
            kTextureWidth,
            kTextureHeight,

            kNUM_ENTRIES
        };

        enum class TextureFormatBase
        {
            kOwn,
            kColorBufferFormat,
            kDepthBufferFormat,

            kNUM_ENTRIES
        };

        Texture::Texture();
        Texture::~Texture();

        void fromMemory(int width, int height, FragmentFormat format);
        void genNoise(int width, int height, FragmentFormat format);
        void makeRenderTarget(int width, int height, FragmentFormat format);
        void makeInputColor(int width, int height, FragmentFormat format);
        void makeInputDepth(int width, int height, FragmentFormat format);
        void makeInputHUDless(int width, int height, FragmentFormat format);
        void makeInputHDR(int width, int height, FragmentFormat format);
        void makeInputColorBase(int width, int height, FragmentFormat format);

        static void deriveSizeDimension(int inColorWidth, int inColorHeight, int inDepthWidth, int inDepthHeight, TextureSizeBase valueBase, float valueMul, int * value)
        {
            if (!value)
                return;

            switch (valueBase)
            {
            case TextureSizeBase::kOne:
                *value = (int)valueMul;
                break;
            case TextureSizeBase::kColorBufferWidth:
                *value = (int)(inColorWidth * valueMul);
                break;
            case TextureSizeBase::kColorBufferHeight:
                *value = (int)(inColorHeight * valueMul);
                break;
            case TextureSizeBase::kDepthBufferWidth:
                *value = (int)(inDepthWidth * valueMul);
                break;
            case TextureSizeBase::kDepthBufferHeight:
                *value = (int)(inDepthHeight * valueMul);
                break;
            case TextureSizeBase::kTextureWidth:
            case TextureSizeBase::kTextureHeight:
                // In case Texture Size dependency, do nothing, we should have the value in width/height
                break;
            };
        }

        void deriveSize(int inColorWidth, int inColorHeight, int inDepthWidth, int inDepthHeight)
        {
            deriveSizeDimension(inColorWidth, inColorHeight, inDepthWidth, inDepthHeight, m_widthBase, m_widthMul, &m_width);
            deriveSizeDimension(inColorWidth, inColorHeight, inDepthWidth, inDepthHeight, m_heightBase, m_heightMul, &m_height);
        }

        bool isReuseAllowed() const;
        bool compare(const Texture * other) const;

        void copyResourcesFrom(const Texture * other);

        void ilwalidateResources();

        TextureType  getTextureType() const { return m_type; }

        TextureType         m_type;

        bool                m_needsSRV = false;
        bool                m_needsRTV = false;

        int                 m_width;
        int                 m_height;
        FragmentFormat      m_format;

        unsigned int        m_levels = 1;

        TextureSizeBase         m_widthBase = TextureSizeBase::kColorBufferWidth;
        float                   m_widthMul = 1.0f;
        TextureSizeBase         m_heightBase = TextureSizeBase::kColorBufferHeight;
        float                   m_heightMul = 1.0f;
        TextureFormatBase       m_formatBase = TextureFormatBase::kColorBufferFormat;

        int          m_firstPassReadIdx = -1;
        int          m_lastPassReadIdx = -1;

        int          m_firstPassWriteIdx = -1;
        int          m_lastPassWriteIdx = -1;

        void *              m_initData;    // Used by pre-generated textures;

        std::wstring        m_filepath;    // Used by textures loaded from file

        bool        m_excludeHash = false;

        // GAPI-specific
        ID3D11Texture2D *      m_D3DTexture = nullptr;
        ID3D11ShaderResourceView *  m_D3DSRV = nullptr;
        ID3D11RenderTargetView *  m_D3DRTV = nullptr;

    protected:

        bool        m_allowReuse = true;
    };

}
}
