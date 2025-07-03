#include <d3d11_1.h>

#include "ir/TypeEnums.h"
#include "ir/Sampler.h"
#include "ir/DataSource.h"
#include "ir/Texture.h"

namespace shadermod
{
namespace ir
{

    Texture::Texture():
        m_initData(nullptr)
    {
    }

    Texture::~Texture()
    {
        if (m_initData)
            free(m_initData);
    }

    void Texture::fromMemory(int width, int height, FragmentFormat format)
    {
        m_widthBase = Texture::TextureSizeBase::kOne;
        m_widthMul = (float)width;
        m_heightBase = Texture::TextureSizeBase::kOne;
        m_heightMul = (float)height;

        m_width = width;
        m_height = height;
        m_format = format;
        m_type = TextureType::kFromFile;
        m_allowReuse = false;
    }

    void Texture::genNoise(int width, int height, FragmentFormat format)
    {
        m_widthBase = Texture::TextureSizeBase::kOne;
        m_widthMul = (float)width;
        m_heightBase = Texture::TextureSizeBase::kOne;
        m_heightMul = (float)height;

        m_width = width;
        m_height = height;
        m_format = format;
        m_type = TextureType::kNoise;
        m_allowReuse = false;
    }

    void Texture::makeRenderTarget(int width, int height, FragmentFormat format)
    {
        m_widthBase = Texture::TextureSizeBase::kOne;
        m_widthMul = (float)width;
        m_heightBase = Texture::TextureSizeBase::kOne;
        m_heightMul = (float)height;

        m_width = width;
        m_height = height;
        m_format = format;
        m_type = TextureType::kRenderTarget;
        m_allowReuse = true;
    }

    void Texture::makeInputColor(int width, int height, FragmentFormat format)
    {
        m_widthBase = Texture::TextureSizeBase::kColorBufferWidth;
        m_widthMul = 1.0f;
        m_heightBase = Texture::TextureSizeBase::kColorBufferHeight;
        m_heightMul = 1.0f;

        m_width = width;
        m_height = height;
        m_format = format;
        // TODO: think about making it re-usable
        m_type = TextureType::kInputColor;
        m_allowReuse = false;
    }

    void Texture::makeInputDepth(int width, int height, FragmentFormat format)
    {
        m_widthBase = Texture::TextureSizeBase::kDepthBufferWidth;
        m_widthMul = 1.0f;
        m_heightBase = Texture::TextureSizeBase::kDepthBufferHeight;
        m_heightMul = 1.0f;

        m_width = width;
        m_height = height;
        m_format = format;
        m_type = TextureType::kInputDepth;
        m_allowReuse = false;
    }
    
    void Texture::makeInputHUDless(int width, int height, FragmentFormat format)
    {
        m_widthBase = Texture::TextureSizeBase::kOne;
        m_widthMul = (float)width;
        m_heightBase = Texture::TextureSizeBase::kOne;
        m_heightMul = (float)height;

        m_width = width;
        m_height = height;
        m_format = format;
        m_type = TextureType::kInputHUDless;
        m_allowReuse = false;
    }

    void Texture::makeInputHDR(int width, int height, FragmentFormat format)
    {
        m_widthBase = Texture::TextureSizeBase::kOne;
        m_widthMul = (float)width;
        m_heightBase = Texture::TextureSizeBase::kOne;
        m_heightMul = (float)height;

        m_width = width;
        m_height = height;
        m_format = format;
        m_type = TextureType::kInputHDR;
        m_allowReuse = false;
    }

    void Texture::makeInputColorBase(int width, int height, FragmentFormat format)
    {
        m_widthBase = Texture::TextureSizeBase::kColorBufferWidth;
        m_widthMul = 1.0f;
        m_heightBase = Texture::TextureSizeBase::kColorBufferHeight;
        m_heightMul = 1.0f;

        m_width = width;
        m_height = height;
        m_format = format;
        // TODO: think about making it re-usable
        m_type = TextureType::kInputColorBase;
        m_allowReuse = false;
    }

    bool Texture::isReuseAllowed() const
    {
        return m_allowReuse;
    }

    bool Texture::compare(const Texture * other) const
    {
        return (m_width == other->m_width &&
                m_height == other->m_height &&
                m_format == other->m_format);
    }

    void Texture::copyResourcesFrom(const Texture * other)
    {
        m_D3DTexture = other->m_D3DTexture;
        m_D3DSRV = other->m_D3DSRV;
        m_D3DRTV = other->m_D3DRTV;
    }

    void Texture::ilwalidateResources()
    {
        m_D3DTexture = nullptr;
        m_D3DSRV = nullptr;
        m_D3DRTV = nullptr;
    }

}
}
