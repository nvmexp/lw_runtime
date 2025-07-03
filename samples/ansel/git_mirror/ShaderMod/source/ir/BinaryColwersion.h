#pragma once

#include "Log.h"
#include "darkroom/StringColwersion.h"
#include "ir/UserConstantManager.h"
#include "acef.h"

namespace shadermod
{
namespace ir
{

    int validateAbsPath(const wchar_t * cfgPath, size_t cfgPathLen, const wchar_t * path)
    {
        // Validation: see if config_path actually matches the beginning of the shader path (otherwise they need to be moved in the future)
        for (size_t valIdx = 0, valIdxEnd = cfgPathLen; valIdx < valIdxEnd; ++valIdx)
        {
            if (cfgPath[valIdx] != path[valIdx])
            {
                return -1;
            }
        }
        return (int)cfgPathLen;
    }

    void colwertIRToBinary(const wchar_t * effectPath, const wchar_t * effectTempPath, const Effect & m_irEffect, acef::EffectStorage * acefEffectStorage, uint64_t timestamp)
    {
        if (acefEffectStorage == nullptr)
            return;

        const wchar_t * tempFolderPlaceholder = L"\\/temp\\/";
        const size_t tempFolderPlaceholderLen = wcslen(tempFolderPlaceholder);

        size_t effectPathLen = wcslen(effectPath);

        // IR->ACEF structures colwersion
        /////////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////////////

        struct IRColwersionMappings
        {
            std::unordered_map<shadermod::ir::VertexShader *, uint32_t> vertexShaderPtrToIndex;
            std::unordered_map<shadermod::ir::PixelShader *, uint32_t> pixelShaderPtrToIndex;

            std::unordered_map<shadermod::ir::ConstantBuf *, uint32_t> constBufPtrToIndex;
            std::unordered_map<shadermod::ir::Texture *, uint32_t> readBufPtrToIndex;
            std::unordered_map<shadermod::ir::Texture *, uint32_t> writeBufPtrToIndex;
            std::unordered_map<shadermod::ir::Sampler *, uint32_t> samplerPtrToIndex;
        };

        IRColwersionMappings irColwersionMappings;

        const shadermod::ir::UserConstantManager & ucManager = m_irEffect.getUserConstantManager();

        auto colwertWcharToUTF8Char = [&acefEffectStorage](const wchar_t * wcharStr) -> char *
        {
            const std::string utf8String = darkroom::getUtf8FromWstr(wcharStr);
            const size_t utf8StringByteSize = utf8String.length() * sizeof(char);

            char * buffer = (char *)acefEffectStorage->allocateMem(utf8StringByteSize);
            memcpy(buffer, utf8String.c_str(), utf8StringByteSize);

            return buffer;
        };

        auto copyChar = [&acefEffectStorage](const char * charStr) -> char *
        {
            const size_t charStrLen = strlen(charStr);
            char * buffer = (char *)acefEffectStorage->allocateMem(charStrLen * sizeof(char));

            memcpy(buffer, charStr, charStrLen * sizeof(char));

            return buffer;
        };


        // TODO avoroshilov ACEF: add function that automatically fills in the main header
        // to avoid version track issues etc
        acefEffectStorage->header.binStorage.magicWord = compilerMagicWordAndVersion;
        char compiler[9] = " YAML/IR";
        char hash[9] = "FFffFFff";
        memcpy(acefEffectStorage->header.binStorage.compiler, compiler, 8*sizeof(char));
        memcpy(acefEffectStorage->header.binStorage.hash, hash, 8*sizeof(char));
        acefEffectStorage->header.binStorage.version = 0;
        acefEffectStorage->header.binStorage.timestamp = timestamp;

        // For now, single IR user is YAML compiler, and it doesn't allow for dependencies
        acefEffectStorage->header.binStorage.dependenciesNum = 0;
        acefEffectStorage->header.fileTimestamps = nullptr;
        acefEffectStorage->header.filePathLens = nullptr;
        acefEffectStorage->header.filePathOffsets = nullptr;
        acefEffectStorage->header.filePathsUtf8 = nullptr;


        // Resource Chunk
        /////////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////////////

        acefEffectStorage->resourceHeader = { 0 };

        for (int i = 0, iend = (int)m_irEffect.m_textures.size(); i < iend; ++i)
        {
            shadermod::ir::Texture * lwrTex = m_irEffect.m_textures[i];

            if (lwrTex->m_needsSRV)
                ++acefEffectStorage->resourceHeader.binStorage.readBuffersNum;

            if (lwrTex->m_needsRTV)
                ++acefEffectStorage->resourceHeader.binStorage.writeBuffersNum;
        }

        // Read buffers offset

        auto checkIfTextureParamterized = [](shadermod::ir::Texture * texture) -> bool
        {
            if (texture->m_type == shadermod::ir::Texture::TextureType::kNoise)
                return true;
            return false;
        };
        auto checkIfTextureIntermediate = [](shadermod::ir::Texture * texture) -> bool
        {
            if (texture->m_type == shadermod::ir::Texture::TextureType::kRenderTarget)
                return true;
            return false;
        };
        auto checkIfTextureFromFile = [](shadermod::ir::Texture * texture) -> bool
        {
            if (texture->m_type == shadermod::ir::Texture::TextureType::kFromFile)
                return true;
            return false;
        };
        auto checkIfTextureSystem = [](shadermod::ir::Texture * texture) -> bool
        {
            if (texture->m_type == shadermod::ir::Texture::TextureType::kInputColor ||
                texture->m_type == shadermod::ir::Texture::TextureType::kInputDepth ||
                texture->m_type == shadermod::ir::Texture::TextureType::kInputHUDless ||
                texture->m_type == shadermod::ir::Texture::TextureType::kInputHDR ||
                texture->m_type == shadermod::ir::Texture::TextureType::kInputColorBase)
                return true;
            return false;
        };

        // Fill in the read/write buffers arrays
        acefEffectStorage->resourceHeader.readBufferTextureHandles = (uint32_t *)acefEffectStorage->allocateMem(acefEffectStorage->resourceHeader.binStorage.readBuffersNum * sizeof(uint32_t));
        acefEffectStorage->resourceHeader.writeBufferTextureHandles = (uint32_t *)acefEffectStorage->allocateMem(acefEffectStorage->resourceHeader.binStorage.writeBuffersNum * sizeof(uint32_t));

        const uint32_t ilwalidBuffer = 0xFFffFFff;

        uint32_t inputColorReadBufferIdx = ilwalidBuffer;
        uint32_t inputColorWriteBufferIdx = ilwalidBuffer;
        uint32_t inputDepthReadBufferIdx = ilwalidBuffer;
        uint32_t inputDepthWriteBufferIdx = ilwalidBuffer;
        uint32_t inputHUDlessReadBufferIdx = ilwalidBuffer;
        uint32_t inputHUDlessWriteBufferIdx = ilwalidBuffer;
        uint32_t inputHDRReadBufferIdx = ilwalidBuffer;
        uint32_t inputHDRWriteBufferIdx = ilwalidBuffer;
        uint32_t inputColorBaseReadBufferIdx = ilwalidBuffer;
        uint32_t inputColorBaseWriteBufferIdx = ilwalidBuffer;

        {
            uint32_t * lwrrentReadBufHandle = acefEffectStorage->resourceHeader.readBufferTextureHandles;
            uint32_t * lwrrentWriteBufHandle = acefEffectStorage->resourceHeader.writeBufferTextureHandles;

            // Pass over the system textures
            for (int i = 0, iend = (int)m_irEffect.m_textures.size(); i < iend; ++i)
            {
                shadermod::ir::Texture * lwrTex = m_irEffect.m_textures[i];

                if (!checkIfTextureSystem(lwrTex))
                    continue;

                ptrdiff_t lwrrentReadBufHandleIdx = lwrrentReadBufHandle - acefEffectStorage->resourceHeader.readBufferTextureHandles;
                ptrdiff_t lwrrentWriteBufHandleIdx = lwrrentWriteBufHandle - acefEffectStorage->resourceHeader.writeBufferTextureHandles;

                if (lwrTex->m_type == shadermod::ir::Texture::TextureType::kInputColor)
                {
                    if (lwrTex->m_needsSRV)
                    {
                        inputColorReadBufferIdx = (uint32_t)lwrrentReadBufHandleIdx;
                        *lwrrentReadBufHandle++ = (uint32_t)acef::SystemTexture::kInputColor;
                    }
                    if (lwrTex->m_needsRTV)
                    {
                        // Theoretically, that's an error
                        inputColorWriteBufferIdx = (uint32_t)lwrrentWriteBufHandleIdx;
                        *lwrrentWriteBufHandle++ = (uint32_t)acef::SystemTexture::kInputColor;
                    }
                }
                else if (lwrTex->m_type == shadermod::ir::Texture::TextureType::kInputDepth)
                {
                    if (lwrTex->m_needsSRV)
                    {
                        inputDepthReadBufferIdx = (uint32_t)lwrrentReadBufHandleIdx;
                        *lwrrentReadBufHandle++ = (uint32_t)acef::SystemTexture::kInputDepth;
                    }
                    if (lwrTex->m_needsRTV)
                    {
                        // Theoretically, that's an error
                        inputDepthWriteBufferIdx = (uint32_t)lwrrentWriteBufHandleIdx;
                        *lwrrentWriteBufHandle++ = (uint32_t)acef::SystemTexture::kInputDepth;
                    }
                }
                else if (lwrTex->m_type == shadermod::ir::Texture::TextureType::kInputHUDless)
                {
                    if (lwrTex->m_needsSRV)
                    {
                        inputHUDlessReadBufferIdx = (uint32_t)lwrrentReadBufHandleIdx;
                        *lwrrentReadBufHandle++ = (uint32_t)acef::SystemTexture::kInputHUDless;
                    }
                    if (lwrTex->m_needsRTV)
                    {
                        // Theoretically, that's an error
                        inputHUDlessWriteBufferIdx = (uint32_t)lwrrentWriteBufHandleIdx;
                        *lwrrentWriteBufHandle++ = (uint32_t)acef::SystemTexture::kInputHUDless;
                    }
                }
                else if (lwrTex->m_type == shadermod::ir::Texture::TextureType::kInputHDR)
                {
                    if (lwrTex->m_needsSRV)
                    {
                        inputHDRReadBufferIdx = (uint32_t)lwrrentReadBufHandleIdx;
                        *lwrrentReadBufHandle++ = (uint32_t)acef::SystemTexture::kInputHDR;
                    }
                    if (lwrTex->m_needsRTV)
                    {
                        // Theoretically, that's an error
                        inputHDRWriteBufferIdx = (uint32_t)lwrrentWriteBufHandleIdx;
                        *lwrrentWriteBufHandle++ = (uint32_t)acef::SystemTexture::kInputHDR;
                    }
                }
                else if (lwrTex->m_type == shadermod::ir::Texture::TextureType::kInputColorBase)
                {
                    if (lwrTex->m_needsSRV)
                    {
                        inputColorBaseReadBufferIdx = (uint32_t)lwrrentReadBufHandleIdx;
                        *lwrrentReadBufHandle++ = (uint32_t)acef::SystemTexture::kInputColorBase;
                    }
                    if (lwrTex->m_needsRTV)
                    {
                        // Theoretically, that's an error
                        inputColorBaseWriteBufferIdx = (uint32_t)lwrrentWriteBufHandleIdx;
                        *lwrrentWriteBufHandle++ = (uint32_t)acef::SystemTexture::kInputColorBase;
                    }
                }
            }

            uint32_t textureIdx = 0;

            // Pass over the parametrized textures
            for (int i = 0, iend = (int)m_irEffect.m_textures.size(); i < iend; ++i)
            {
                shadermod::ir::Texture * lwrTex = m_irEffect.m_textures[i];

                if (!checkIfTextureParamterized(lwrTex))
                    continue;

                ptrdiff_t lwrrentReadBufHandleIdx = lwrrentReadBufHandle - acefEffectStorage->resourceHeader.readBufferTextureHandles;
                ptrdiff_t lwrrentWriteBufHandleIdx = lwrrentWriteBufHandle - acefEffectStorage->resourceHeader.writeBufferTextureHandles;
                if (lwrTex->m_needsSRV)
                {
                    irColwersionMappings.readBufPtrToIndex[lwrTex] = (uint32_t)lwrrentReadBufHandleIdx;
                    *lwrrentReadBufHandle++ = textureIdx;
                }
                if (lwrTex->m_needsRTV)
                {
                    irColwersionMappings.writeBufPtrToIndex[lwrTex] = (uint32_t)lwrrentWriteBufHandleIdx;
                    *lwrrentWriteBufHandle++ = textureIdx;
                }

                ++textureIdx;
            }

            // Pass over intermediate textures
            for (int i = 0, iend = (int)m_irEffect.m_textures.size(); i < iend; ++i)
            {
                shadermod::ir::Texture * lwrTex = m_irEffect.m_textures[i];

                if (!checkIfTextureIntermediate(lwrTex))
                    continue;

                ptrdiff_t lwrrentReadBufHandleIdx = lwrrentReadBufHandle - acefEffectStorage->resourceHeader.readBufferTextureHandles;
                ptrdiff_t lwrrentWriteBufHandleIdx = lwrrentWriteBufHandle - acefEffectStorage->resourceHeader.writeBufferTextureHandles;
                if (lwrTex->m_needsSRV)
                {
                    irColwersionMappings.readBufPtrToIndex[lwrTex] = (uint32_t)lwrrentReadBufHandleIdx;
                    *lwrrentReadBufHandle++ = textureIdx;
                }
                if (lwrTex->m_needsRTV)
                {
                    irColwersionMappings.writeBufPtrToIndex[lwrTex] = (uint32_t)lwrrentWriteBufHandleIdx;
                    *lwrrentWriteBufHandle++ = textureIdx;
                }

                ++textureIdx;
            }

            // Pass over textures from file
            for (int i = 0, iend = (int)m_irEffect.m_textures.size(); i < iend; ++i)
            {
                shadermod::ir::Texture * lwrTex = m_irEffect.m_textures[i];

                if (!checkIfTextureFromFile(lwrTex))
                    continue;

                ptrdiff_t lwrrentReadBufHandleIdx = lwrrentReadBufHandle - acefEffectStorage->resourceHeader.readBufferTextureHandles;
                ptrdiff_t lwrrentWriteBufHandleIdx = lwrrentWriteBufHandle - acefEffectStorage->resourceHeader.writeBufferTextureHandles;
                if (lwrTex->m_needsSRV)
                {
                    irColwersionMappings.readBufPtrToIndex[lwrTex] = (uint32_t)lwrrentReadBufHandleIdx;
                    *lwrrentReadBufHandle++ = textureIdx;
                }
                if (lwrTex->m_needsRTV)
                {
                    irColwersionMappings.writeBufPtrToIndex[lwrTex] = (uint32_t)lwrrentWriteBufHandleIdx;
                    *lwrrentWriteBufHandle++ = textureIdx;
                }

                ++textureIdx;
            }
        }

        // Vertex shaders
        //////////////////////////////////////////////////////////////
        {
            acefEffectStorage->resourceHeader.binStorage.vertexShadersNum = 0;
            for (int i = 0, iend = (int)m_irEffect.m_vertexShaders.size(); i < iend; ++i)
            {
                shadermod::ir::VertexShader * lwrVS = m_irEffect.m_vertexShaders[i];

                irColwersionMappings.vertexShaderPtrToIndex[lwrVS] = (uint32_t)acefEffectStorage->vertexShaders.size();
                ++acefEffectStorage->resourceHeader.binStorage.vertexShadersNum;

                // Validation: see if config_path actually matches the beginning of the shader path (otherwise they need to be moved in the future)
                if (validateAbsPath(effectPath, effectPathLen, lwrVS->m_fileName) < 0)
                {
                    assert(false);
                    LOG_WARN("Shader path doesn't match config path");
                    break;
                }

                // Size/offset
                int filenameLen = (int)wcslen(lwrVS->m_fileName+effectPathLen);
                int entryFunctionLen = (int)strlen(lwrVS->m_entryPoint);

                acef::ResourceVertexShader vertexShader;

                vertexShader.binStorage.filePathLen = filenameLen;
                vertexShader.binStorage.entryFunctionLen = entryFunctionLen;

                vertexShader.filePathUtf8 = colwertWcharToUTF8Char(lwrVS->m_fileName+effectPathLen);
                vertexShader.entryFunctionAscii = copyChar(lwrVS->m_entryPoint);

                acefEffectStorage->vertexShaders.push_back(vertexShader);
            }

            assert(acefEffectStorage->vertexShaders.size() == acefEffectStorage->resourceHeader.binStorage.vertexShadersNum);
        }

        // Pixel shaders
        //////////////////////////////////////////////////////////////
        {
            acefEffectStorage->resourceHeader.binStorage.pixelShadersNum = 0;
            for (int i = 0, iend = (int)m_irEffect.m_pixelShaders.size(); i < iend; ++i)
            {
                shadermod::ir::PixelShader * lwrPS = m_irEffect.m_pixelShaders[i];

                irColwersionMappings.pixelShaderPtrToIndex[lwrPS] = (uint32_t)acefEffectStorage->pixelShaders.size();
                ++acefEffectStorage->resourceHeader.binStorage.pixelShadersNum;

                // Validation: see if config_path actually matches the beginning of the shader path (otherwise they need to be moved in the future)
                if (validateAbsPath(effectPath, effectPathLen, lwrPS->m_fileName) < 0)
                {
                    assert(false);
                    LOG_WARN("Shader path doesn't match config path");
                    break;
                }

                // Size/offset
                int filenameLen = (int)wcslen(lwrPS->m_fileName+effectPathLen);
                int entryFunctionLen = (int)strlen(lwrPS->m_entryPoint);

                acef::ResourcePixelShader pixelShader;

                pixelShader.binStorage.filePathLen = filenameLen;
                pixelShader.binStorage.entryFunctionLen = entryFunctionLen;

                pixelShader.filePathUtf8 = colwertWcharToUTF8Char(lwrPS->m_fileName+effectPathLen);
                pixelShader.entryFunctionAscii = copyChar(lwrPS->m_entryPoint);

                acefEffectStorage->pixelShaders.push_back(pixelShader);
            }

            assert(acefEffectStorage->pixelShaders.size() == acefEffectStorage->resourceHeader.binStorage.pixelShadersNum);
        }


        // Textures
        //////////////////////////////////////////////////////////////

        auto colwertFragmentFormat = [](shadermod::ir::FragmentFormat irFragmentFormat) -> acef::FragmentFormat
        {
            switch (irFragmentFormat)
            {
            case shadermod::ir::FragmentFormat::kRGBA8_uint:
                return acef::FragmentFormat::kRGBA8_uint;
            case shadermod::ir::FragmentFormat::kBGRA8_uint:
                return acef::FragmentFormat::kBGRA8_uint;

            case shadermod::ir::FragmentFormat::kRGBA16_uint:
                return acef::FragmentFormat::kRGBA16_uint;
            case shadermod::ir::FragmentFormat::kRGBA16_fp:
                return acef::FragmentFormat::kRGBA16_fp;
            case shadermod::ir::FragmentFormat::kRGBA32_fp:
                return acef::FragmentFormat::kRGBA32_fp;

            case shadermod::ir::FragmentFormat::kSBGRA8_uint:
                return acef::FragmentFormat::kSBGRA8_uint;
            case shadermod::ir::FragmentFormat::kSRGBA8_uint:
                return acef::FragmentFormat::kSRGBA8_uint;

            case shadermod::ir::FragmentFormat::kR10G10B10A2_uint:
                return acef::FragmentFormat::kR10G10B10A2_uint;

            case shadermod::ir::FragmentFormat::kR11G11B10_float:
                return acef::FragmentFormat::kR11G11B10_float;

            case shadermod::ir::FragmentFormat::kRG8_uint:
                return acef::FragmentFormat::kRG8_uint;
            case shadermod::ir::FragmentFormat::kRG16_uint:
                return acef::FragmentFormat::kRG16_uint;
            case shadermod::ir::FragmentFormat::kRG16_fp:
                return acef::FragmentFormat::kRG16_fp;
            case shadermod::ir::FragmentFormat::kRG32_fp:
                return acef::FragmentFormat::kRG32_fp;
            case shadermod::ir::FragmentFormat::kRG32_uint:
                return acef::FragmentFormat::kRG32_uint;

            case shadermod::ir::FragmentFormat::kR8_uint:
                return acef::FragmentFormat::kR8_uint;
            case shadermod::ir::FragmentFormat::kR16_uint:
                return acef::FragmentFormat::kR16_uint;
            case shadermod::ir::FragmentFormat::kR16_fp:
                return acef::FragmentFormat::kR16_fp;
            case shadermod::ir::FragmentFormat::kR32_fp:
                return acef::FragmentFormat::kR32_fp;
            case shadermod::ir::FragmentFormat::kR32_uint:
                return acef::FragmentFormat::kR32_uint;

            case shadermod::ir::FragmentFormat::kD24S8:
                return acef::FragmentFormat::kD24S8;
            case shadermod::ir::FragmentFormat::kD32_fp:
                return acef::FragmentFormat::kD32_fp;
            case shadermod::ir::FragmentFormat::kD32_fp_S8X24_uint:
                return acef::FragmentFormat::kD32_fp_S8X24_uint;

            default:
                LOG_WARN("ACEF compiler - unknown type colwersion %d", (int)irFragmentFormat);
                return acef::FragmentFormat::kNUM_ENTRIES;
            }
        };

        auto colwertTexParametrizedType = [](shadermod::ir::Texture::TextureType texType)
        {
            if (texType == shadermod::ir::Texture::TextureType::kNoise)
                return acef::ResourceTextureParametrizedType::kNOISE;
            else
            {
                LOG_WARN("ACEF compiler - unknown texture parametrized type %d", (int)texType);
                return acef::ResourceTextureParametrizedType::kNOISE;
            }
        };

        auto colwertTexSizeBase = [](shadermod::ir::Texture::TextureSizeBase texSizeBase)
        {
            switch (texSizeBase)
            {
            case shadermod::ir::Texture::TextureSizeBase::kOne:
                return acef::TextureSizeBase::kOne;
            case shadermod::ir::Texture::TextureSizeBase::kColorBufferWidth:
                return acef::TextureSizeBase::kColorBufferWidth;
            case shadermod::ir::Texture::TextureSizeBase::kColorBufferHeight:
                return acef::TextureSizeBase::kColorBufferHeight;
            case shadermod::ir::Texture::TextureSizeBase::kDepthBufferWidth:
                return acef::TextureSizeBase::kDepthBufferWidth;
            case shadermod::ir::Texture::TextureSizeBase::kDepthBufferHeight:
                return acef::TextureSizeBase::kDepthBufferHeight;
            case shadermod::ir::Texture::TextureSizeBase::kTextureWidth:
                return acef::TextureSizeBase::kTextureWidth;
            case shadermod::ir::Texture::TextureSizeBase::kTextureHeight:
                return acef::TextureSizeBase::kTextureHeight;
            default:
                LOG_WARN("ACEF compiler - unknown texture size base colwersion %d", (int)texSizeBase);
                return acef::TextureSizeBase::kOne;
            }
        };

        static_assert((int)acef::FragmentFormat::kNUM_ENTRIES == (int)shadermod::ir::FragmentFormat::kNUM_ENTRIES, "acef::FragmentFormat doesn't cover all supported formats");

        // Parametrized
        {
            acefEffectStorage->resourceHeader.binStorage.texturesParametrizedNum = 0;
            for (int i = 0, iend = (int)m_irEffect.m_textures.size(); i < iend; ++i)
            {
                shadermod::ir::Texture * lwrTex = m_irEffect.m_textures[i];

                if (!checkIfTextureParamterized(lwrTex))
                    continue;

                ++acefEffectStorage->resourceHeader.binStorage.texturesParametrizedNum;

                // Fill in info
                acef::ResourceTextureParametrized texParametrized;

                acef::TextureSizeStorage texSize;

                texSize.binStorage.mul = lwrTex->m_widthMul;
                texSize.binStorage.texSizeBase = colwertTexSizeBase(lwrTex->m_widthBase);
                texParametrized.binStorage.width = texSize;

                if (texSize.binStorage.texSizeBase == acef::TextureSizeBase::kTextureWidth ||
                    texSize.binStorage.texSizeBase == acef::TextureSizeBase::kTextureHeight)
                {
                    LOG_ERROR("IR->ACEF colwerter - parametrized texture wants to have width of some file");
                }

                texSize.binStorage.mul = lwrTex->m_heightMul;
                texSize.binStorage.texSizeBase = colwertTexSizeBase(lwrTex->m_heightBase);
                texParametrized.binStorage.height = texSize;

                if (texSize.binStorage.texSizeBase == acef::TextureSizeBase::kTextureWidth ||
                    texSize.binStorage.texSizeBase == acef::TextureSizeBase::kTextureHeight)
                {
                    LOG_ERROR("IR->ACEF colwerter - parametrized texture wants to have height of some file");
                }

                texParametrized.binStorage.format = colwertFragmentFormat(lwrTex->m_format);

                memset(texParametrized.binStorage.parameters, 0, acef::textureParametrizedNumParameters*sizeof(float));

                texParametrized.binStorage.type = colwertTexParametrizedType(lwrTex->m_type);

                acefEffectStorage->texturesParametrized.push_back(texParametrized);
            }

            assert(acefEffectStorage->texturesParametrized.size() == acefEffectStorage->resourceHeader.binStorage.texturesParametrizedNum);
        }

        // Intermediate
        {
            acefEffectStorage->resourceHeader.binStorage.texturesIntermediateNum = 0;
            for (int i = 0, iend = (int)m_irEffect.m_textures.size(); i < iend; ++i)
            {
                shadermod::ir::Texture * lwrTex = m_irEffect.m_textures[i];

                if (!checkIfTextureIntermediate(lwrTex))
                    continue;

                ++acefEffectStorage->resourceHeader.binStorage.texturesIntermediateNum;

                // Fill in info
                acef::ResourceTextureIntermediate texIntermediate;

                acef::TextureSizeStorage texSize;

                texSize.binStorage.mul = lwrTex->m_widthMul;
                texSize.binStorage.texSizeBase = colwertTexSizeBase(lwrTex->m_widthBase);
                texIntermediate.binStorage.width = texSize;

                if (texSize.binStorage.texSizeBase == acef::TextureSizeBase::kTextureWidth ||
                    texSize.binStorage.texSizeBase == acef::TextureSizeBase::kTextureHeight)
                {
                    LOG_ERROR("IR->ACEF colwerter - intermediate texture wants to have width of some file");
                }

                texSize.binStorage.mul = lwrTex->m_heightMul;
                texSize.binStorage.texSizeBase = colwertTexSizeBase(lwrTex->m_heightBase);
                texIntermediate.binStorage.height = texSize;

                if (texSize.binStorage.texSizeBase == acef::TextureSizeBase::kTextureWidth ||
                    texSize.binStorage.texSizeBase == acef::TextureSizeBase::kTextureHeight)
                {
                    LOG_ERROR("IR->ACEF colwerter - intermediate texture wants to have height of some file");
                }

                texIntermediate.binStorage.format = colwertFragmentFormat(lwrTex->m_format);

                texIntermediate.binStorage.levels = lwrTex->m_levels;

                acefEffectStorage->texturesIntermediate.push_back(texIntermediate);
            }

            assert(acefEffectStorage->texturesIntermediate.size() == acefEffectStorage->resourceHeader.binStorage.texturesIntermediateNum);
        }

        // FromFile
        {
            acefEffectStorage->resourceHeader.binStorage.texturesFromFileNum = 0;
            for (int i = 0, iend = (int)m_irEffect.m_textures.size(); i < iend; ++i)
            {
                shadermod::ir::Texture * lwrTex = m_irEffect.m_textures[i];

                if (!checkIfTextureFromFile(lwrTex))
                    continue;

                ++acefEffectStorage->resourceHeader.binStorage.texturesFromFileNum;

                // Validation: see if config_path actually matches the beginning of the shader path (otherwise they need to be moved in the future)
                if (validateAbsPath(effectPath, effectPathLen, lwrTex->m_filepath.c_str()) < 0)
                {
                    assert(false);
                    LOG_WARN("Texture from file path doesn't match config path");
                    break;
                }

                // Size/offset
                int pathLen = (int)wcslen(lwrTex->m_filepath.c_str() + effectPathLen);

                // Fill in info
                acef::ResourceTextureFromFile texFromFile;

                acef::TextureSizeStorage texSize;

                texFromFile.binStorage.pathLen = pathLen;

                texSize.binStorage.mul = lwrTex->m_widthMul;
                texSize.binStorage.texSizeBase = colwertTexSizeBase(lwrTex->m_widthBase);
                texFromFile.binStorage.width = texSize;

                texSize.binStorage.mul = lwrTex->m_heightMul;
                texSize.binStorage.texSizeBase = colwertTexSizeBase(lwrTex->m_heightBase);
                texFromFile.binStorage.height = texSize;

                texFromFile.binStorage.format = colwertFragmentFormat(lwrTex->m_format);

                texFromFile.pathUtf8 = colwertWcharToUTF8Char(lwrTex->m_filepath.c_str() + effectPathLen);
                texFromFile.binStorage.excludeHash = lwrTex->m_excludeHash;

                acefEffectStorage->texturesFromFile.push_back(texFromFile);
            }

            assert(acefEffectStorage->texturesFromFile.size() == acefEffectStorage->resourceHeader.binStorage.texturesFromFileNum);
        }

        auto colwertSamplerAddressType = [](shadermod::ir::AddressType addrType)
        {
            switch (addrType)
            {
            case shadermod::ir::AddressType::kWrap:
                return acef::ResourceSamplerAddressType::kWrap;
            case shadermod::ir::AddressType::kClamp:
                return acef::ResourceSamplerAddressType::kClamp;
            case shadermod::ir::AddressType::kMirror:
                return acef::ResourceSamplerAddressType::kMirror;
            case shadermod::ir::AddressType::kBorder:
                return acef::ResourceSamplerAddressType::kBorder;
            default:
            {
                LOG_WARN("ACEF compiler - unknown sampler address type %d", (int)addrType);
                return acef::ResourceSamplerAddressType::kClamp;
            }
            };
        };

        auto colwertSamplerFilterType = [](shadermod::ir::FilterType filterType)
        {
            switch (filterType)
            {
            case shadermod::ir::FilterType::kPoint:
                return acef::ResourceSamplerFilterType::kPoint;
            case shadermod::ir::FilterType::kLinear:
                return acef::ResourceSamplerFilterType::kLinear;
            default:
            {
                LOG_WARN("ACEF compiler - unknown sampler filter type %d", (int)filterType);
                return acef::ResourceSamplerFilterType::kLinear;
            }
            };
        };

        // Samplers
        //////////////////////////////////////////////////////////////
        {
            acefEffectStorage->resourceHeader.binStorage.samplersNum = 0;
            for (int i = 0, iend = (int)m_irEffect.m_samplers.size(); i < iend; ++i)
            {
                shadermod::ir::Sampler * lwrSampler = m_irEffect.m_samplers[i];

                irColwersionMappings.samplerPtrToIndex[lwrSampler] = (uint32_t)acefEffectStorage->samplers.size();
                ++acefEffectStorage->resourceHeader.binStorage.samplersNum;

                // Fill in info
                acef::ResourceSampler sampler;

                sampler.binStorage.addrU = colwertSamplerAddressType(lwrSampler->m_addrU);
                sampler.binStorage.addrV = colwertSamplerAddressType(lwrSampler->m_addrV);
                sampler.binStorage.addrW = colwertSamplerAddressType(lwrSampler->m_addrW);
                sampler.binStorage.filterMag = colwertSamplerFilterType(lwrSampler->m_filterMag);
                sampler.binStorage.filterMin = colwertSamplerFilterType(lwrSampler->m_filterMin);
                sampler.binStorage.filterMip = colwertSamplerFilterType(lwrSampler->m_filterMip);

                acefEffectStorage->samplers.push_back(sampler);
            }

            assert(acefEffectStorage->samplers.size() == acefEffectStorage->resourceHeader.binStorage.samplersNum);
        }

        // Constant buffers
        //////////////////////////////////////////////////////////////

        auto isSystemConst = [](shadermod::ir::Constant * constant)
        {
            return (int)constant->m_type < (int)shadermod::ir::ConstType::kUserConstantBase;
        };

        auto callwlateConstantHandle = [&isSystemConst, &ucManager](shadermod::ir::Constant * constant) -> uint32_t
        {
            if (isSystemConst(constant))
            {
                switch (constant->m_type)
                {
                case shadermod::ir::ConstType::kCaptureState:
                    return (uint32_t)acef::SystemConstant::kCaptureState;
                case shadermod::ir::ConstType::kDT:
                    return (uint32_t)acef::SystemConstant::kDT;
                case shadermod::ir::ConstType::kElapsedTime:
                    return (uint32_t)acef::SystemConstant::kElapsedTime;
                case shadermod::ir::ConstType::kFrame:
                    return (uint32_t)acef::SystemConstant::kFrame;
                case shadermod::ir::ConstType::kScreenSize:
                    return (uint32_t)acef::SystemConstant::kScreenSize;
                case shadermod::ir::ConstType::kTileUV:
                    return (uint32_t)acef::SystemConstant::kTileUV;
                case shadermod::ir::ConstType::kDepthAvailable:
                    return (uint32_t)acef::SystemConstant::kDepthAvailable;
                case shadermod::ir::ConstType::kHDRAvailable:
                    return (uint32_t)acef::SystemConstant::kHDRAvailable;
                case shadermod::ir::ConstType::kHUDlessAvailable:
                    return (uint32_t)acef::SystemConstant::kHUDlessAvailable;
                default:
                {
                    LOG_WARN("ACEF compiler - unknown constant type %d", (int)constant->m_type);
                    return (uint32_t)acef::SystemConstant::kCaptureState;
                }
                }
            }
            else
            {
                return (uint32_t)ucManager.findByName(constant->m_userConstName)->getUid();
            }
        };

        size_t totalCharacters;

        {
            acefEffectStorage->resourceHeader.binStorage.constantBuffersNum = 0;
            for (int i = 0, iend = (int)m_irEffect.m_constantBufs.size(); i < iend; ++i)
            {
                shadermod::ir::ConstantBuf * lwrConstBuf = m_irEffect.m_constantBufs[i];

                irColwersionMappings.constBufPtrToIndex[lwrConstBuf] = (uint32_t)acefEffectStorage->constantBuffers.size();
                ++acefEffectStorage->resourceHeader.binStorage.constantBuffersNum;

                // Size/offset
                int buffersByteSize = (int)(lwrConstBuf->m_constants.size() * sizeof(uint32_t) + lwrConstBuf->m_constants.size() * sizeof(uint32_t));
                int totalByteSize = (int)acef::ResourceConstantBuffer::BinaryStorage::storageByteSize();

                // Fill in info
                acef::ResourceConstantBuffer constantBuffer;

                constantBuffer.binStorage.constantsNum = (uint32_t)lwrConstBuf->m_constants.size();

                constantBuffer.constantHandle = (uint32_t *)acefEffectStorage->allocateMem(constantBuffer.binStorage.constantsNum * sizeof(uint32_t));
                constantBuffer.constantOffsetInComponents = (uint32_t *)acefEffectStorage->allocateMem(constantBuffer.binStorage.constantsNum * sizeof(uint32_t));
                constantBuffer.constantNameLens = (uint16_t *)acefEffectStorage->allocateMem(constantBuffer.binStorage.constantsNum * sizeof(uint16_t));
                constantBuffer.constantNameOffsets = (uint32_t *)acefEffectStorage->allocateMem(constantBuffer.binStorage.constantsNum * sizeof(uint32_t));

                totalCharacters = 0;
                for (uint32_t constIdx = 0, constIdxEnd = constantBuffer.binStorage.constantsNum; constIdx < constIdxEnd; ++constIdx)
                {
                    shadermod::ir::Constant * lwrConstant = lwrConstBuf->m_constants[constIdx];
                    constantBuffer.constantHandle[constIdx] = callwlateConstantHandle(lwrConstant);
                    constantBuffer.constantOffsetInComponents[constIdx] = lwrConstant->m_constantOffsetInComponents;

                    constantBuffer.constantNameOffsets[constIdx] = (uint32_t)totalCharacters;
                    if (constantBuffer.constantOffsetInComponents[constIdx] == shadermod::ir::Constant::OffsetAuto)
                    {
                        const char * name = lwrConstant->m_constBindName;
                        size_t nameLen = strlen(name);
                        constantBuffer.constantNameLens[constIdx] = (uint16_t)nameLen;
                        totalCharacters += nameLen;
                    }
                    else
                    {
                        constantBuffer.constantNameLens[constIdx] = 0;
                    }
                }

                constantBuffer.constantNames = (char *)acefEffectStorage->allocateMem(totalCharacters * sizeof(char));
                totalCharacters = 0;
                for (uint32_t constIdx = 0, constIdxEnd = constantBuffer.binStorage.constantsNum; constIdx < constIdxEnd; ++constIdx)
                {
                    if (constantBuffer.constantOffsetInComponents[constIdx] == shadermod::ir::Constant::OffsetAuto)
                    {
                        shadermod::ir::Constant * lwrConstant = lwrConstBuf->m_constants[constIdx];
                        const char * name = lwrConstant->m_constBindName;
                        size_t nameLen = strlen(name);
                        memcpy(constantBuffer.constantNames+totalCharacters, name, nameLen * sizeof(char));
                        totalCharacters += nameLen;
                    }
                }

                acefEffectStorage->constantBuffers.push_back(constantBuffer);
            }

            assert(acefEffectStorage->constantBuffers.size() == acefEffectStorage->resourceHeader.binStorage.constantBuffersNum);
        }

        // UI Controls Chunk
        /////////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////////////

        acefEffectStorage->uiControlsHeader = { 0 };

        auto colwertUIDataType = [](shadermod::ir::UserConstDataType dataType)
        {
            switch (dataType)
            {
            case shadermod::ir::UserConstDataType::kInt:
                return acef::UserConstDataType::kInt;
            case shadermod::ir::UserConstDataType::kUInt:
                return acef::UserConstDataType::kUInt;
            case shadermod::ir::UserConstDataType::kFloat:
                return acef::UserConstDataType::kFloat;
            case shadermod::ir::UserConstDataType::kBool:
                return acef::UserConstDataType::kBool;
            default:
                {
                    LOG_WARN("ACEF compiler - unknown user constant data type %d", (int)dataType);
                    return acef::UserConstDataType::kFloat;
                }
            }
        };

        auto colwertUIControlType = [](shadermod::ir::UiControlType controlType)
        {
            switch (controlType)
            {
            case shadermod::ir::UiControlType::kCheckbox:
                return acef::UIControlType::kCheckbox;
            case shadermod::ir::UiControlType::kEditbox:
                return acef::UIControlType::kEditbox;
            case shadermod::ir::UiControlType::kFlyout:
                return acef::UIControlType::kFlyout;
            case shadermod::ir::UiControlType::kSlider:
                return acef::UIControlType::kSlider;
            case shadermod::ir::UiControlType::kColorPicker:
                return acef::UIControlType::kColorPicker;
            case shadermod::ir::UiControlType::kRadioButton:
                return acef::UIControlType::kRadioButton;
            default:
                {
                    LOG_WARN("ACEF compiler - unknown user constant control type %d", (int)controlType);
                    return acef::UIControlType::kSlider;
                }
            }
        };

        auto colwertTypelessVariable = [](shadermod::ir::TypelessVariable variable)
        {
            acef::TypelessVariableStorage acefVariable;

            // Only copy acef::TypelessVariableStorage::numBytes bytes ATM
            memcpy(acefVariable.binStorage.data, variable.getRawMemory(), acef::TypelessVariableStorage::numBytes);

            return acefVariable;
        };

        acefEffectStorage->uiControlsHeader.binStorage.userConstantsNum = 0;
        shadermod::ir::UserConstantManager::ConstRange ucRange = ucManager.getPointersToAllUserConstants();
        for (; ucRange.begin < ucRange.end; ++ucRange.begin)
        {
            const shadermod::ir::UserConstant * lwrUC = *(ucRange.begin);

            ++acefEffectStorage->uiControlsHeader.binStorage.userConstantsNum;

            // Fill in info
            acef::UserConstant userConstant;

            auto copyBuffersStringWithLocalization = [&copyChar, &acefEffectStorage](const shadermod::ir::UserConstant::StringWithLocalization & strWithLocalization, acef::UILocalizedStringBuffers * acefStringBuffers)
            {
                const uint32_t localizationsNum = (uint32_t)strWithLocalization.m_localization.size();
                acefStringBuffers->defaultStringAscii = copyChar(strWithLocalization.m_string.c_str());
                acefStringBuffers->localizedStrings = (acef::UILocalizedStringBuffers::LocalizedString *)acefEffectStorage->allocateMem(localizationsNum * sizeof(acef::UILocalizedStringBuffers::LocalizedString));

                uint32_t locIdx = 0;
                auto localizationIter = strWithLocalization.m_localization.begin();
                for (; localizationIter != strWithLocalization.m_localization.end(); ++localizationIter, ++locIdx)
                {
                    acef::UILocalizedStringBuffers::LocalizedString * lwrLocalization = acefStringBuffers->localizedStrings + locIdx;
                    lwrLocalization->binStorage.langid = (uint16_t)localizationIter->first;
                    lwrLocalization->binStorage.strLen = (uint16_t)localizationIter->second.size();

                    lwrLocalization->stringUtf8 = copyChar(localizationIter->second.c_str());
                }
            };

            // Storageable data
            userConstant.binStorage.controlNameLen = (uint32_t)lwrUC->getName().size();
            userConstant.binStorage.dataType = colwertUIDataType(lwrUC->getType());
            userConstant.binStorage.dataDimensionality = lwrUC->getTypeDimensions();
            userConstant.binStorage.uiControlType = colwertUIControlType(lwrUC->getControlType());

            userConstant.binStorage.defaultValue = colwertTypelessVariable(lwrUC->getDefaultValue());
            userConstant.binStorage.minimumValue = colwertTypelessVariable(lwrUC->getMinimumValue());
            userConstant.binStorage.maximumValue = colwertTypelessVariable(lwrUC->getMaximumValue());

            userConstant.binStorage.uiMinimumValue = colwertTypelessVariable(lwrUC->getUiValueMin());
            userConstant.binStorage.uiMaximumValue = colwertTypelessVariable(lwrUC->getUiValueMax());
            userConstant.binStorage.uiValueStep = colwertTypelessVariable(lwrUC->getUiValueStep());

            userConstant.binStorage.stickyValue = lwrUC->getStickyValue();
            userConstant.binStorage.stickyRegion = lwrUC->getStickyRegion();

            const uint32_t optionsNum = lwrUC->getNumListOptions();
            userConstant.binStorage.optionsNum = optionsNum;
            userConstant.binStorage.optionDefault = lwrUC->getDefaultListOption();

            userConstant.binStorage.label.binStorage.defaultStringLen = (uint16_t)lwrUC->m_uiLabel.m_string.size();
            userConstant.binStorage.label.binStorage.localizationsNum = (uint16_t)lwrUC->m_uiLabel.m_localization.size();

            userConstant.binStorage.hint.binStorage.defaultStringLen = (uint16_t)lwrUC->m_uiHint.m_string.size();
            userConstant.binStorage.hint.binStorage.localizationsNum = (uint16_t)lwrUC->m_uiHint.m_localization.size();

            userConstant.binStorage.uiValueUnit.binStorage.defaultStringLen = (uint16_t)lwrUC->m_uiValueUnit.m_string.size();
            userConstant.binStorage.uiValueUnit.binStorage.localizationsNum = (uint16_t)lwrUC->m_uiValueUnit.m_localization.size();

            // Metadata
            userConstant.controlNameAscii = copyChar(lwrUC->getName().c_str());

            // Dropdown/Flyout/Radio Options
            userConstant.optiolwalues = (acef::TypelessVariableStorage *)acefEffectStorage->allocateMem(optionsNum * sizeof(acef::TypelessVariableStorage));
            userConstant.optionNames = (acef::UILocalizedStringStorage *)acefEffectStorage->allocateMem(optionsNum * sizeof(acef::UILocalizedStringStorage));
            userConstant.optionNamesBuffers = (acef::UILocalizedStringBuffers *)acefEffectStorage->allocateMem(optionsNum * sizeof(acef::UILocalizedStringBuffers));
            for (uint32_t optIdx = 0; optIdx < optionsNum; ++optIdx)
            {
                const shadermod::ir::TypelessVariable & irLwrOptiolwalue = lwrUC->getListOption(optIdx);
                const shadermod::ir::UserConstant::StringWithLocalization & irLwrOptionLocalization = lwrUC->getListOptionLocalization(optIdx);
                acef::UILocalizedStringStorage * lwrOptionNameStorage = userConstant.optionNames + optIdx;
                acef::UILocalizedStringBuffers * lwrOptionNameBuffer = userConstant.optionNamesBuffers + optIdx;

                userConstant.optiolwalues[optIdx] = colwertTypelessVariable(irLwrOptiolwalue);

                lwrOptionNameStorage->binStorage.defaultStringLen = (uint16_t)irLwrOptionLocalization.m_string.size();
                const uint32_t localizationsNum = (uint32_t)irLwrOptionLocalization.m_localization.size();
                lwrOptionNameStorage->binStorage.localizationsNum = (uint16_t)localizationsNum;

                copyBuffersStringWithLocalization(irLwrOptionLocalization, lwrOptionNameBuffer);
            }

            // Grouped Variable Names
            const uint32_t dimensionality = lwrUC->getTypeDimensions();
            userConstant.variableNames = (acef::UILocalizedStringStorage *)acefEffectStorage->allocateMem(dimensionality * sizeof(acef::UILocalizedStringStorage));
            userConstant.variableNamesBuffers = (acef::UILocalizedStringBuffers *)acefEffectStorage->allocateMem(dimensionality * sizeof(acef::UILocalizedStringBuffers));
            for (uint32_t dimIdx = 0; dimIdx < dimensionality; ++dimIdx)
            {
                const shadermod::ir::UserConstant::StringWithLocalization & irLwrVariableLocalization = lwrUC->getValueDisplayName(dimIdx);
                acef::UILocalizedStringStorage * lwrVariableNameStorage = userConstant.variableNames + dimIdx;
                acef::UILocalizedStringBuffers * lwrVariableNameBuffer = userConstant.variableNamesBuffers + dimIdx;

                lwrVariableNameStorage->binStorage.defaultStringLen = (uint16_t)irLwrVariableLocalization.m_string.size();
                const uint32_t localizationsNum = (uint32_t)irLwrVariableLocalization.m_localization.size();
                lwrVariableNameStorage->binStorage.localizationsNum = (uint16_t)localizationsNum;

                copyBuffersStringWithLocalization(irLwrVariableLocalization, lwrVariableNameBuffer);
            }

            copyBuffersStringWithLocalization(lwrUC->m_uiLabel, &userConstant.labelBuffers);
            copyBuffersStringWithLocalization(lwrUC->m_uiHint, &userConstant.hintBuffers);
            copyBuffersStringWithLocalization(lwrUC->m_uiValueUnit, &userConstant.uiValueUnitBuffers);

            acefEffectStorage->userConstants.push_back(userConstant);
        }
        assert(acefEffectStorage->userConstants.size() == acefEffectStorage->uiControlsHeader.binStorage.userConstantsNum);

        // Passes Chunk
        /////////////////////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////////////

        acefEffectStorage->passesHeader = { 0 };

        {
            auto colwertRasterFillModeIRtoACEF = [](const RasterizerFillMode & irRasterFillMode) -> acef::RasterizerFillMode
            {
                switch (irRasterFillMode)
                {
                case RasterizerFillMode::kSolid:
                    return acef::RasterizerFillMode::kSolid;
                case RasterizerFillMode::kWireframe:
                    return acef::RasterizerFillMode::kWireframe;
                default:
                    {
                        LOG_WARN("IR->ACEF colwersion - unknown RasterizerFillMode");
                        return acef::RasterizerFillMode::kSolid;
                    }
                }
            };

            auto colwertRasterLwllModeIRtoACEF = [](const RasterizerLwllMode & irRasterLwllMode) -> acef::RasterizerLwllMode
            {
                switch (irRasterLwllMode)
                {
                case RasterizerLwllMode::kBack:
                    return acef::RasterizerLwllMode::kBack;
                case RasterizerLwllMode::kFront:
                    return acef::RasterizerLwllMode::kFront;
                case RasterizerLwllMode::kNone:
                    return acef::RasterizerLwllMode::kNone;
                default:
                    {
                        LOG_WARN("IR->ACEF colwersion - unknown RasterizerLwllMode");
                        return acef::RasterizerLwllMode::kNone;
                    }
                }
            };

            auto colwertRasterizerStateIRtoACEF = [&colwertRasterFillModeIRtoACEF, &colwertRasterLwllModeIRtoACEF](const RasterizerState & irRasterState) -> acef::RasterizerStateStorage
            {
                acef::RasterizerStateStorage acefRasterState;

                acefRasterState.binStorage.fillMode = colwertRasterFillModeIRtoACEF(irRasterState.fillMode);
                acefRasterState.binStorage.lwllMode = colwertRasterLwllModeIRtoACEF(irRasterState.lwllMode);

                acefRasterState.binStorage.depthBias = irRasterState.depthBias;
                acefRasterState.binStorage.depthBiasClamp = irRasterState.depthBiasClamp;
                acefRasterState.binStorage.slopeScaledDepthBias = irRasterState.slopeScaledDepthBias;

                acefRasterState.binStorage.frontCounterClockwise = irRasterState.frontCounterClockwise;
                acefRasterState.binStorage.depthClipEnable = irRasterState.depthClipEnable;
                acefRasterState.binStorage.scissorEnable = irRasterState.scissorEnable;
                acefRasterState.binStorage.multisampleEnable = irRasterState.multisampleEnable;
                acefRasterState.binStorage.antialiasedLineEnable = irRasterState.antialiasedLineEnable;

                return acefRasterState;
            };

            auto colwertComparisonFuncIRtoACEF = [](const ComparisonFunc & irComparisonFunc) -> acef::ComparisonFunc
            {
                switch (irComparisonFunc)
                {
                case ComparisonFunc::kAlways:
                    return acef::ComparisonFunc::kAlways;
                case ComparisonFunc::kEqual:
                    return acef::ComparisonFunc::kEqual;
                case ComparisonFunc::kGreater:
                    return acef::ComparisonFunc::kGreater;
                case ComparisonFunc::kGreaterEqual:
                    return acef::ComparisonFunc::kGreaterEqual;
                case ComparisonFunc::kLess:
                    return acef::ComparisonFunc::kLess;
                case ComparisonFunc::kLessEqual:
                    return acef::ComparisonFunc::kLessEqual;
                case ComparisonFunc::kNever:
                    return acef::ComparisonFunc::kNever;
                case ComparisonFunc::kNotEqual:
                    return acef::ComparisonFunc::kNotEqual;
                default:
                    {
                        LOG_WARN("IR->ACEF colwersion - unknown ComparisonFunc");
                        return acef::ComparisonFunc::kAlways;
                    }
                }
            };

            auto colwertStencilOpIRtoACEF = [](const StencilOp & irStencilOp) -> acef::StencilOp
            {
                switch (irStencilOp)
                {
                case StencilOp::kDecr:
                    return acef::StencilOp::kDecr;
                case StencilOp::kDecrSat:
                    return acef::StencilOp::kDecrSat;
                case StencilOp::kIncr:
                    return acef::StencilOp::kIncr;
                case StencilOp::kIncrSat:
                    return acef::StencilOp::kIncrSat;
                case StencilOp::kIlwert:
                    return acef::StencilOp::kIlwert;
                case StencilOp::kKeep:
                    return acef::StencilOp::kKeep;
                case StencilOp::kReplace:
                    return acef::StencilOp::kReplace;
                case StencilOp::kZero:
                    return acef::StencilOp::kZero;
                default:
                    {
                        LOG_WARN("IR->ACEF colwersion - unknown StencilOp");
                        return acef::StencilOp::kKeep;
                    }
                }
            };

            auto colwertDepthStencilOpIRtoACEF = [&colwertStencilOpIRtoACEF, &colwertComparisonFuncIRtoACEF](const DepthStencilOp & irDepthStencilOp) -> acef::DepthStencilOpStorage
            {
                acef::DepthStencilOpStorage acefDepthStencilOp;

                acefDepthStencilOp.binStorage.failOp = colwertStencilOpIRtoACEF(irDepthStencilOp.failOp);
                acefDepthStencilOp.binStorage.depthFailOp = colwertStencilOpIRtoACEF(irDepthStencilOp.depthFailOp);
                acefDepthStencilOp.binStorage.passOp = colwertStencilOpIRtoACEF(irDepthStencilOp.depthFailOp);
                acefDepthStencilOp.binStorage.func = colwertComparisonFuncIRtoACEF(irDepthStencilOp.func);

                return acefDepthStencilOp;
            };

            auto colwertDepthWriteMaskIRtoACEF = [](const DepthWriteMask & irDepthWriteMask) -> acef::DepthWriteMask
            {
                switch (irDepthWriteMask)
                {
                case DepthWriteMask::kAll:
                    return acef::DepthWriteMask::kAll;
                case DepthWriteMask::kZero:
                    return acef::DepthWriteMask::kZero;
                default:
                    {
                        LOG_WARN("IR->ACEF colwersion - unknown DepthWriteMask");
                        return acef::DepthWriteMask::kAll;
                    }
                }
            };

            auto colwertDepthStencilStateIRtoACEF = [&colwertDepthStencilOpIRtoACEF, &colwertComparisonFuncIRtoACEF, &colwertDepthWriteMaskIRtoACEF](const DepthStencilState & irDepthStencilState) -> acef::DepthStencilStateStorage
            {
                acef::DepthStencilStateStorage acefDepthStencilState;

                acefDepthStencilState.binStorage.frontFace = colwertDepthStencilOpIRtoACEF(irDepthStencilState.frontFace);
                acefDepthStencilState.binStorage.backFace = colwertDepthStencilOpIRtoACEF(irDepthStencilState.backFace);

                acefDepthStencilState.binStorage.depthWriteMask = colwertDepthWriteMaskIRtoACEF(irDepthStencilState.depthWriteMask);
                acefDepthStencilState.binStorage.depthFunc = colwertComparisonFuncIRtoACEF(irDepthStencilState.depthFunc);

                acefDepthStencilState.binStorage.stencilReadMask = irDepthStencilState.stencilReadMask;
                acefDepthStencilState.binStorage.stencilWriteMask = irDepthStencilState.stencilWriteMask;
                acefDepthStencilState.binStorage.isDepthEnabled = irDepthStencilState.isDepthEnabled;
                acefDepthStencilState.binStorage.isStencilEnabled = irDepthStencilState.isStencilEnabled;

                return acefDepthStencilState;
            };

            auto colwertBlendCoefIRtoACEF = [](const BlendCoef & irBlendCoef) -> acef::BlendCoef
            {
                switch (irBlendCoef)
                {
                case BlendCoef::kZero:
                    return acef::BlendCoef::kZero;
                case BlendCoef::kOne:
                    return acef::BlendCoef::kOne;
                case BlendCoef::kSrcColor:
                    return acef::BlendCoef::kSrcColor;
                case BlendCoef::kIlwSrcColor:
                    return acef::BlendCoef::kIlwSrcColor;
                case BlendCoef::kSrcAlpha:
                    return acef::BlendCoef::kSrcAlpha;
                case BlendCoef::kIlwSrcAlpha:
                    return acef::BlendCoef::kIlwSrcAlpha;
                case BlendCoef::kDstAlpha:
                    return acef::BlendCoef::kDstAlpha;
                case BlendCoef::kIlwDstAlpha:
                    return acef::BlendCoef::kIlwDstAlpha;
                case BlendCoef::kDstColor:
                    return acef::BlendCoef::kDstColor;
                case BlendCoef::kIlwDstColor:
                    return acef::BlendCoef::kIlwDstColor;
                case BlendCoef::kSrcAlphaSat:
                    return acef::BlendCoef::kSrcAlphaSat;
                case BlendCoef::kBlendFactor:
                    return acef::BlendCoef::kBlendFactor;
                case BlendCoef::kIlwBlendFactor:
                    return acef::BlendCoef::kIlwBlendFactor;
                case BlendCoef::kSrc1Color:
                    return acef::BlendCoef::kSrc1Color;
                case BlendCoef::kIlwSrc1Color:
                    return acef::BlendCoef::kIlwSrc1Color;
                case BlendCoef::kSrc1Alpha:
                    return acef::BlendCoef::kSrc1Alpha;
                case BlendCoef::kIlwSrc1Alpha:
                    return acef::BlendCoef::kIlwSrc1Alpha;
                default:
                    {
                        LOG_WARN("IR->ACEF colwersion - unknown BlendCoef");
                        return acef::BlendCoef::kOne;
                    }
                }
            };

            auto colwertBlendOpIRtoACEF = [](const BlendOp & irBlendOp) -> acef::BlendOp
            {
                switch (irBlendOp)
                {
                case BlendOp::kAdd:
                    return acef::BlendOp::kAdd;
                case BlendOp::kSub:
                    return acef::BlendOp::kSub;
                case BlendOp::kRevSub:
                    return acef::BlendOp::kRevSub;
                case BlendOp::kMin:
                    return acef::BlendOp::kMin;
                case BlendOp::kMax:
                    return acef::BlendOp::kMax;
                default:
                    {
                        LOG_WARN("IR->ACEF colwersion - unknown BlendOp");
                        return acef::BlendOp::kAdd;
                    }
                }
            };

            auto colwertColorWriteEnableBitsIRtoACEF = [](const ColorWriteEnableBits & irColorWriteEnable) -> acef::ColorWriteEnableBits
            {
                switch (irColorWriteEnable)
                {
                case ColorWriteEnableBits::kRed:
                    return acef::ColorWriteEnableBits::kRed;
                case ColorWriteEnableBits::kGreen:
                    return acef::ColorWriteEnableBits::kGreen;
                case ColorWriteEnableBits::kBlue:
                    return acef::ColorWriteEnableBits::kBlue;
                case ColorWriteEnableBits::kAlpha:
                    return acef::ColorWriteEnableBits::kAlpha;
                case ColorWriteEnableBits::kAll:
                    return acef::ColorWriteEnableBits::kAll;
                default:
                    {
                        LOG_WARN("IR->ACEF colwersion - unknown ColorWriteEnableBits");
                        return acef::ColorWriteEnableBits::kAll;
                    }
                }
            };

            auto colwertAlphaBlendRenderTargetStateIRtoACEF = [&colwertBlendCoefIRtoACEF, &colwertBlendOpIRtoACEF, &colwertColorWriteEnableBitsIRtoACEF](const AlphaBlendRenderTargetState & irAlphaBlendRenderTargetState) -> acef::AlphaBlendRenderTargetStateStorage
            {
                acef::AlphaBlendRenderTargetStateStorage acefAlphaBlendRenderTargetState;

                acefAlphaBlendRenderTargetState.binStorage.src = colwertBlendCoefIRtoACEF(irAlphaBlendRenderTargetState.src);
                acefAlphaBlendRenderTargetState.binStorage.dst = colwertBlendCoefIRtoACEF(irAlphaBlendRenderTargetState.dst);
                acefAlphaBlendRenderTargetState.binStorage.op = colwertBlendOpIRtoACEF(irAlphaBlendRenderTargetState.op);

                acefAlphaBlendRenderTargetState.binStorage.srcAlpha = colwertBlendCoefIRtoACEF(irAlphaBlendRenderTargetState.srcAlpha);
                acefAlphaBlendRenderTargetState.binStorage.dstAlpha = colwertBlendCoefIRtoACEF(irAlphaBlendRenderTargetState.dstAlpha);
                acefAlphaBlendRenderTargetState.binStorage.opAlpha = colwertBlendOpIRtoACEF(irAlphaBlendRenderTargetState.opAlpha);

                acefAlphaBlendRenderTargetState.binStorage.renderTargetWriteMask = colwertColorWriteEnableBitsIRtoACEF(irAlphaBlendRenderTargetState.renderTargetWriteMask);
                acefAlphaBlendRenderTargetState.binStorage.isEnabled = irAlphaBlendRenderTargetState.isEnabled;

                return acefAlphaBlendRenderTargetState;
            };

            auto colwertAlphaBlendStateIRtoACEF = [&colwertAlphaBlendRenderTargetStateIRtoACEF](const AlphaBlendState & irAlphaBlendState) -> acef::AlphaBlendStateStorage
            {
                acef::AlphaBlendStateStorage acefAlphaBlendState;

                for (uint32_t idx = 0; idx < acef::AlphaBlendStateStorage::renderTargetsNum; ++idx)
                {
                    acefAlphaBlendState.binStorage.renderTargetState[idx] = colwertAlphaBlendRenderTargetStateIRtoACEF(irAlphaBlendState.renderTargetState[idx]);
                }

                acefAlphaBlendState.binStorage.alphaToCoverageEnable = irAlphaBlendState.alphaToCoverageEnable;
                acefAlphaBlendState.binStorage.independentBlendEnable = irAlphaBlendState.independentBlendEnable;

                return acefAlphaBlendState;
            };

            acefEffectStorage->passesHeader.binStorage.passesNum = 0;
            for (int i = 0, iend = (int)m_irEffect.m_passes.size(); i < iend; ++i)
            {
                shadermod::ir::Pass * lwrPass = m_irEffect.m_passes[i];

                ++acefEffectStorage->passesHeader.binStorage.passesNum;

                // Size/offset
                uint32_t readBuffersNum = (uint32_t)lwrPass->m_dataSources.size();
                uint32_t writeBuffersNum = (uint32_t)lwrPass->m_dataOut.size();
                uint32_t constantBuffersVSNum = (uint32_t)lwrPass->m_constBufsVS.size();
                uint32_t constantBuffersPSNum = (uint32_t)lwrPass->m_constBufsPS.size();
                uint32_t samplersNum = (uint32_t)lwrPass->m_samplers.size();

                int buffersByteSize = 0;

                // Fill in info
                acef::Pass pass;

                pass.binStorage.rasterizerState = colwertRasterizerStateIRtoACEF(lwrPass->m_rasterizerState);
                pass.binStorage.depthStencilState = colwertDepthStencilStateIRtoACEF(lwrPass->m_depthStencilState);
                pass.binStorage.alphaBlendState = colwertAlphaBlendStateIRtoACEF(lwrPass->m_alphaBlendState);

                pass.binStorage.constantBuffersVSNum = constantBuffersVSNum;
                pass.binStorage.constantBuffersPSNum = constantBuffersPSNum;
                pass.binStorage.readBuffersNum = readBuffersNum;
                pass.binStorage.writeBuffersNum = writeBuffersNum;
                pass.binStorage.samplersNum = samplersNum;

                if (lwrPass->m_vertexShader != nullptr)
                {
                    auto vsPtrToIdxIter = irColwersionMappings.vertexShaderPtrToIndex.find(lwrPass->m_vertexShader);
                    if (vsPtrToIdxIter != irColwersionMappings.vertexShaderPtrToIndex.end())
                        pass.binStorage.vertexShaderIndex = vsPtrToIdxIter->second;
                    else
                    {
                        LOG_ERROR("ACEF compiler - VS mapping failure for pass %d", (int)i);
                        return;
                    }
                }
                else
                {
                    pass.binStorage.vertexShaderIndex = acef::Pass::vertexShaderDefault;
                }

                auto psPtrToIdxIter = irColwersionMappings.pixelShaderPtrToIndex.find(lwrPass->m_pixelShader);
                if (psPtrToIdxIter != irColwersionMappings.pixelShaderPtrToIndex.end())
                    pass.binStorage.pixelShaderIndex = psPtrToIdxIter->second;
                else
                {
                    LOG_ERROR("ACEF compiler - PS mapping failure for pass %d", (int)i);
                    return;
                }

                // TODO avoroshilov ACEF: textureSizes callwlation - this should be deprecated once we'll have output pass writeBuffers instead of just sizes
                // TODO avoroshilov ACEF: remove pass size base/mul
                acef::TextureSizeStorage texSize;

                texSize.binStorage.mul = 1.0f;
                texSize.binStorage.texSizeBase = acef::TextureSizeBase::kOne;
                pass.binStorage.width = texSize;

                texSize.binStorage.mul = 1.0f;
                texSize.binStorage.texSizeBase = acef::TextureSizeBase::kOne;
                pass.binStorage.height = texSize;

                pass.readBuffersSlots = (uint32_t *)acefEffectStorage->allocateMem(readBuffersNum*sizeof(uint32_t));
                pass.readBuffersNameLens = (uint16_t *)acefEffectStorage->allocateMem(readBuffersNum*sizeof(uint16_t));
                pass.readBuffersNameOffsets = (uint32_t *)acefEffectStorage->allocateMem(readBuffersNum*sizeof(uint32_t));
                pass.readBuffersIndices = (uint32_t *)acefEffectStorage->allocateMem(readBuffersNum*sizeof(uint32_t));

                totalCharacters = 0;
                for (uint32_t idx = 0; idx < readBuffersNum; ++idx)
                {
                    pass.readBuffersSlots[idx] = lwrPass->getSlotSrc(idx);
                    // Bound-by-name
                    pass.readBuffersNameOffsets[idx] = (uint32_t)totalCharacters;
                    if (pass.readBuffersSlots[idx] == shadermod::ir::Pass::BindByName)
                    {
                        const char * name = lwrPass->getNameSrc(idx);
                        size_t nameLen = strlen(name);
                        pass.readBuffersNameLens[idx] = (uint16_t)nameLen;
                        totalCharacters += nameLen;
                    }
                    else
                    {
                        pass.readBuffersNameLens[idx] = 0;
                    }

                    if (lwrPass->getDataSrc(idx)->getDataType() != shadermod::ir::DataSource::DataType::kTexture)
                    {
                        LOG_ERROR("ACEF compiler - pass %d contains reference to pass - not supported", (int)i);
                        return;
                    }

                    shadermod::ir::Texture * dataSrcTex = static_cast<shadermod::ir::Texture *>(lwrPass->getDataSrc(idx));
                    if (checkIfTextureSystem(dataSrcTex))
                    {
                        if (dataSrcTex->m_type == shadermod::ir::Texture::TextureType::kInputColor)
                        {
                            pass.readBuffersIndices[idx] = inputColorReadBufferIdx;
                        }
                        else if (dataSrcTex->m_type == shadermod::ir::Texture::TextureType::kInputDepth)
                        {
                            pass.readBuffersIndices[idx] = inputDepthReadBufferIdx;
                        }
                        else if (dataSrcTex->m_type == shadermod::ir::Texture::TextureType::kInputHUDless)
                        {
                            pass.readBuffersIndices[idx] = inputHUDlessReadBufferIdx;
                        }
                        else if (dataSrcTex->m_type == shadermod::ir::Texture::TextureType::kInputHDR)
                        {
                            pass.readBuffersIndices[idx] = inputHDRReadBufferIdx;
                        }
                        else if (dataSrcTex->m_type == shadermod::ir::Texture::TextureType::kInputColorBase)
                        {
                            pass.readBuffersIndices[idx] = inputColorBaseReadBufferIdx;
                        }
                        else
                        {
                            LOG_ERROR("ACEF compiler - WB system mapping failure for pass %d", (int)i);
                            return;
                        }
                    }
                    else
                    {
                        auto rbPtrToIdxIter = irColwersionMappings.readBufPtrToIndex.find(dataSrcTex);
                        if (rbPtrToIdxIter != irColwersionMappings.readBufPtrToIndex.end())
                            pass.readBuffersIndices[idx] = rbPtrToIdxIter->second;
                        else
                        {
                            LOG_ERROR("ACEF compiler - RB mapping failure for pass %d", (int)i);
                            return;
                        }
                    }
                }

                // Copy over names
                pass.readBuffersNames = (char *)acefEffectStorage->allocateMem(totalCharacters*sizeof(char));
                totalCharacters = 0;
                for (uint32_t idx = 0; idx < readBuffersNum; ++idx)
                {
                    // Bound-by-name
                    if (pass.readBuffersSlots[idx] == shadermod::ir::Pass::BindByName)
                    {
                        const char * name = lwrPass->getNameSrc(idx);
                        size_t nameLen = strlen(name);
                        memcpy(pass.readBuffersNames+totalCharacters, name, nameLen * sizeof(char));
                        totalCharacters += nameLen;
                    }
                }


                pass.writeBuffersSlots = (uint32_t *)acefEffectStorage->allocateMem(writeBuffersNum*sizeof(uint32_t));
                pass.writeBuffersNameLens = (uint16_t *)acefEffectStorage->allocateMem(writeBuffersNum*sizeof(uint16_t));
                pass.writeBuffersNameOffsets = (uint32_t *)acefEffectStorage->allocateMem(writeBuffersNum*sizeof(uint32_t));
                pass.writeBuffersIndices = (uint32_t *)acefEffectStorage->allocateMem(writeBuffersNum*sizeof(uint32_t));

                totalCharacters = 0;
                for (uint32_t idx = 0; idx < writeBuffersNum; ++idx)
                {
                    pass.writeBuffersSlots[idx] = lwrPass->getMRTChannelOut(idx);
                    // Bound-by-name is not applicable for the write buffers
                    pass.writeBuffersNameLens[idx] = 0;
                    pass.writeBuffersNameOffsets[idx] = (uint32_t)totalCharacters;

                    if (lwrPass->getDataSrc(idx)->getDataType() != shadermod::ir::DataSource::DataType::kTexture)
                    {
                        LOG_ERROR("ACEF compiler - pass %d contains reference to pass - not supported", (int)i);
                        return;
                    }

                    shadermod::ir::Texture * dataOutTex = lwrPass->getDataOut(idx);
                    if (checkIfTextureSystem(dataOutTex))
                    {
                        if (dataOutTex->m_type == shadermod::ir::Texture::TextureType::kInputColor)
                        {
                            pass.writeBuffersIndices[idx] = inputColorWriteBufferIdx;
                        }
                        else if (dataOutTex->m_type == shadermod::ir::Texture::TextureType::kInputDepth)
                        {
                            pass.writeBuffersIndices[idx] = inputDepthWriteBufferIdx;
                        }
                        else if (dataOutTex->m_type == shadermod::ir::Texture::TextureType::kInputHUDless)
                        {
                            pass.writeBuffersIndices[idx] = inputHUDlessWriteBufferIdx;
                        }
                        else if (dataOutTex->m_type == shadermod::ir::Texture::TextureType::kInputHDR)
                        {
                            pass.writeBuffersIndices[idx] = inputHDRWriteBufferIdx;
                        }
                        else if (dataOutTex->m_type == shadermod::ir::Texture::TextureType::kInputColorBase)
                        {
                            pass.writeBuffersIndices[idx] = inputColorBaseWriteBufferIdx;
                        }
                        else
                        {
                            LOG_ERROR("ACEF compiler - WB system mapping failure for pass %d", (int)i);
                            return;
                        }
                    }
                    else
                    {
                        auto wbPtrToIdxIter = irColwersionMappings.writeBufPtrToIndex.find(dataOutTex);
                        if (wbPtrToIdxIter != irColwersionMappings.writeBufPtrToIndex.end())
                            pass.writeBuffersIndices[idx] = wbPtrToIdxIter->second;
                        else
                        {
                            LOG_ERROR("ACEF compiler - WB mapping failure for pass %d", (int)i);
                            return;
                        }
                    }
                }

                /*
                // NOT REQUIRED, writeBuffers do not support bind by names (that's basically MRT channels, they only have indices)
                // Copy over names
                pass.writeBuffersNames = (char *)acefEffectStorage->allocateMem(totalCharacters*sizeof(char));
                totalCharacters = 0;
                for (uint32_t idx = 0; idx < writeBuffersNum; ++idx)
                {
                    // Bound-by-name
                    if (pass.writeBuffersSlots[idx] == shadermod::ir::Pass::BindByName)
                    {
                        //const char * name = lwrPass->getNameSrc(idx);
                        size_t nameLen = strlen(name);
                        memcpy(pass.writeBuffersNames+totalCharacters, name, nameLen * sizeof(char));
                        totalCharacters += nameLen;
                    }
                }
                */

                pass.constantBuffersVSSlots = (uint32_t *)acefEffectStorage->allocateMem(constantBuffersVSNum*sizeof(uint32_t));
                pass.constantBuffersVSNameLens = (uint16_t *)acefEffectStorage->allocateMem(constantBuffersVSNum*sizeof(uint16_t));
                pass.constantBuffersVSNameOffsets = (uint32_t *)acefEffectStorage->allocateMem(constantBuffersVSNum*sizeof(uint32_t));
                pass.constantBuffersVSIndices = (uint32_t *)acefEffectStorage->allocateMem(constantBuffersVSNum*sizeof(uint32_t));

                totalCharacters = 0;
                for (uint32_t idx = 0; idx < constantBuffersVSNum; ++idx)
                {
                    pass.constantBuffersVSSlots[idx] = lwrPass->m_constBufVSSlots[idx];
                    // Bound-by-name
                    pass.constantBuffersVSNameOffsets[idx] = (uint32_t)totalCharacters;
                    if (pass.constantBuffersVSSlots[idx] == shadermod::ir::Pass::BindByName)
                    {
                        const char * name = lwrPass->m_constBufVSNames[idx]->name;
                        size_t nameLen = strlen(name);
                        pass.constantBuffersVSNameLens[idx] = (uint16_t)nameLen;
                        totalCharacters += nameLen;
                    }
                    else
                    {
                        pass.constantBuffersVSNameLens[idx] = 0;
                    }

                    auto cbPtrToIdxIter = irColwersionMappings.constBufPtrToIndex.find(lwrPass->m_constBufsVS[idx]);
                    if (cbPtrToIdxIter != irColwersionMappings.constBufPtrToIndex.end())
                        pass.constantBuffersVSIndices[idx] = cbPtrToIdxIter->second;
                    else
                    {
                        LOG_ERROR("ACEF compiler - CB mapping failure for pass %d", (int)i);
                        return;
                    }
                }

                // Copy over names
                pass.constantBuffersVSNames = (char *)acefEffectStorage->allocateMem(totalCharacters*sizeof(char));
                totalCharacters = 0;
                for (uint32_t idx = 0; idx < constantBuffersVSNum; ++idx)
                {
                    // Bound-by-name
                    if (pass.constantBuffersVSSlots[idx] == shadermod::ir::Pass::BindByName)
                    {
                        const char * name = lwrPass->m_constBufVSNames[idx]->name;
                        size_t nameLen = strlen(name);
                        memcpy(pass.constantBuffersVSNames+totalCharacters, name, nameLen * sizeof(char));
                        totalCharacters += nameLen;
                    }
                }

                pass.constantBuffersPSSlots = (uint32_t *)acefEffectStorage->allocateMem(constantBuffersPSNum*sizeof(uint32_t));
                pass.constantBuffersPSNameLens = (uint16_t *)acefEffectStorage->allocateMem(constantBuffersPSNum*sizeof(uint16_t));
                pass.constantBuffersPSNameOffsets = (uint32_t *)acefEffectStorage->allocateMem(constantBuffersPSNum*sizeof(uint32_t));
                pass.constantBuffersPSIndices = (uint32_t *)acefEffectStorage->allocateMem(constantBuffersPSNum*sizeof(uint32_t));

                totalCharacters = 0;
                for (uint32_t idx = 0; idx < constantBuffersPSNum; ++idx)
                {
                    pass.constantBuffersPSSlots[idx] = lwrPass->m_constBufPSSlots[idx];
                    // Bound-by-name
                    pass.constantBuffersPSNameOffsets[idx] = (uint32_t)totalCharacters;
                    if (pass.constantBuffersPSSlots[idx] == shadermod::ir::Pass::BindByName)
                    {
                        const char * name = lwrPass->m_constBufPSNames[idx]->name;
                        size_t nameLen = strlen(name);
                        pass.constantBuffersPSNameLens[idx] = (uint16_t)nameLen;
                        totalCharacters += nameLen;
                    }
                    else
                    {
                        pass.constantBuffersPSNameLens[idx] = 0;
                    }

                    auto cbPtrToIdxIter = irColwersionMappings.constBufPtrToIndex.find(lwrPass->m_constBufsPS[idx]);
                    if (cbPtrToIdxIter != irColwersionMappings.constBufPtrToIndex.end())
                        pass.constantBuffersPSIndices[idx] = cbPtrToIdxIter->second;
                    else
                    {
                        LOG_ERROR("ACEF compiler - CB mapping failure for pass %d", (int)i);
                        return;
                    }
                }

                // Copy over names
                pass.constantBuffersPSNames = (char *)acefEffectStorage->allocateMem(totalCharacters*sizeof(char));
                totalCharacters = 0;
                for (uint32_t idx = 0; idx < constantBuffersPSNum; ++idx)
                {
                    // Bound-by-name
                    if (pass.constantBuffersPSSlots[idx] == shadermod::ir::Pass::BindByName)
                    {
                        const char * name = lwrPass->m_constBufPSNames[idx]->name;
                        size_t nameLen = strlen(name);
                        memcpy(pass.constantBuffersPSNames+totalCharacters, name, nameLen * sizeof(char));
                        totalCharacters += nameLen;
                    }
                }

                pass.samplersSlots = (uint32_t *)acefEffectStorage->allocateMem(samplersNum*sizeof(uint32_t));
                pass.samplersNameLens = (uint16_t *)acefEffectStorage->allocateMem(samplersNum*sizeof(uint16_t));
                pass.samplersNameOffsets = (uint32_t *)acefEffectStorage->allocateMem(samplersNum*sizeof(uint32_t));
                pass.samplersIndices = (uint32_t *)acefEffectStorage->allocateMem(samplersNum*sizeof(uint32_t));

                totalCharacters = 0;
                for (uint32_t idx = 0; idx < samplersNum; ++idx)
                {
                    pass.samplersSlots[idx] = lwrPass->getSamplerSlot(idx);
                    // Bound-by-name
                    pass.samplersNameOffsets[idx] = (uint32_t)totalCharacters;
                    if (pass.samplersSlots[idx] == shadermod::ir::Pass::BindByName)
                    {
                        const char * name = lwrPass->getSamplerName(idx);
                        size_t nameLen = strlen(name);
                        pass.samplersNameLens[idx] = (uint16_t)nameLen;
                        totalCharacters += nameLen;
                    }
                    else
                    {
                        pass.samplersNameLens[idx] = 0;
                    }

                    auto sPtrToIdxIter = irColwersionMappings.samplerPtrToIndex.find(lwrPass->getSampler(idx));
                    if (sPtrToIdxIter != irColwersionMappings.samplerPtrToIndex.end())
                        pass.samplersIndices[idx] = sPtrToIdxIter->second;
                    else
                    {
                        LOG_ERROR("ACEF compiler - Sampler mapping failure for pass %d", (int)i);
                        return;
                    }
                }

                // Copy over names
                pass.samplersNames = (char *)acefEffectStorage->allocateMem(totalCharacters*sizeof(char));
                totalCharacters = 0;
                for (uint32_t idx = 0; idx < samplersNum; ++idx)
                {
                    // Bound-by-name
                    if (pass.samplersSlots[idx] == shadermod::ir::Pass::BindByName)
                    {
                        const char * name = lwrPass->getSamplerName(idx);
                        size_t nameLen = strlen(name);
                        memcpy(pass.samplersNames+totalCharacters, name, nameLen * sizeof(char));
                        totalCharacters += nameLen;
                    }
                }

                acefEffectStorage->passes.push_back(pass);
            }
        }
        assert(acefEffectStorage->passes.size() == acefEffectStorage->passesHeader.binStorage.passesNum);


        // DBG DBG DBG
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        const bool DBGaddOptionToControl = false;
        if (DBGaddOptionToControl && acefEffectStorage->userConstants.size() > 0)
        {
            // DBG: adding fake option nums if there aren't any
            char optionNameDefault[] = "optionNameDefault";
            const int localizationsNum = 2;
            char optionNameLoc0[] = "optionNameLoc0_ruRU";
            char optionNameLoc1[] = "optionNameLoc1_frFR";
            char *optionNameLocalizations[] = { optionNameLoc0, optionNameLoc1 };

            acef::UserConstant & uc = acefEffectStorage->userConstants[0];
            uint32_t optIdx = uc.binStorage.optionsNum;
            if (optIdx == 0)
            {
                uint32_t optionsNum = ++uc.binStorage.optionsNum;
                uc.optiolwalues = (acef::TypelessVariableStorage *)acefEffectStorage->allocateMem(optionsNum * sizeof(acef::TypelessVariableStorage));
                uc.optionNames = (acef::UILocalizedStringStorage *)acefEffectStorage->allocateMem(optionsNum * sizeof(acef::UILocalizedStringStorage));
                uc.optionNamesBuffers = (acef::UILocalizedStringBuffers *)acefEffectStorage->allocateMem(optionsNum * sizeof(acef::UILocalizedStringBuffers));
                for (uint32_t optIdx = 0; optIdx < optionsNum; ++optIdx)
                {
                    acef::UILocalizedStringStorage * lwrOptionNameStorage = uc.optionNames + optIdx;
                    acef::UILocalizedStringBuffers * lwrOptionNameBuffer = uc.optionNamesBuffers + optIdx;

                    memset(uc.optiolwalues[optIdx].binStorage.data, 0, sizeof(uc.optiolwalues[optIdx].binStorage.data));

                    lwrOptionNameStorage->binStorage.defaultStringLen = (uint16_t)strlen(optionNameDefault);
                    lwrOptionNameStorage->binStorage.localizationsNum = (uint16_t)localizationsNum;

                    lwrOptionNameBuffer->defaultStringAscii = copyChar(optionNameDefault);
                    lwrOptionNameBuffer->localizedStrings = (acef::UILocalizedStringBuffers::LocalizedString *)acefEffectStorage->allocateMem(localizationsNum * sizeof(acef::UILocalizedStringBuffers::LocalizedString));

                    for (uint32_t locIdx = 0, locIdxEnd = localizationsNum; locIdx < locIdxEnd; ++locIdx)
                    {
                        acef::UILocalizedStringBuffers::LocalizedString * lwrLocalization = lwrOptionNameBuffer->localizedStrings + locIdx;
                        lwrLocalization->binStorage.langid = 113;
                        lwrLocalization->binStorage.strLen = (uint16_t)strlen(optionNameLocalizations[locIdx]);

                        lwrLocalization->stringUtf8 = copyChar(optionNameLocalizations[locIdx]);
                    }
                }
            }
        }
    }


    // TODO: make textuer desc structure w/ ID3D11Texture2D*, ir::FragmentFormat, int, int
    MultipassConfigParserError colwertBinaryToIR(
        const wchar_t * rootDir, const wchar_t * tempsDir,

        const Effect::InputData & finalColorInput, const Effect::InputData & depthInput,
        const Effect::InputData & hudlessInput, const Effect::InputData & hdrInput,
        const Effect::InputData & colorBaseInput,

        const wchar_t * effectPath, const wchar_t * effectTempPath, const acef::EffectStorage & acefEffectStorageLoaded,
        Effect * irEffect,
        bool * doesEffectRequireColor, bool * doesEffectRequireDepth,
        bool * doesEffectRequireHUDless, bool * doesEffectRequireHDR,
        bool * doesEffectRequireColorBase,
        ir::Texture ** outTex,
        bool calcHashes
        )
    {
        if (irEffect == nullptr || outTex == nullptr)
            return MultipassConfigParserError(
                MultipassConfigParserErrorEnum::eInternalError,
                "Binary representation load: invalid arguments provided"
                );

        size_t effectPathLen = wcslen(effectPath);

        bool texturesRelativeToTempPath = false;
        bool shadersRelativeToTempPath = false;
        const wchar_t * tempFolderPlaceholder = L"\\/temp\\/";
        const size_t tempFolderPlaceholderLen = wcslen(tempFolderPlaceholder);

        auto isTempPlaceholderUsed = [&tempFolderPlaceholder, &tempFolderPlaceholderLen](const wchar_t * inStr)
        {
            if (inStr == nullptr)
                return false;

            for (size_t idx = 0; idx < tempFolderPlaceholderLen; ++idx)
            {
                if (tempFolderPlaceholder[idx] != inStr[idx])
                    return false;
            }

            return true;
        };
        auto resolveTempFolderInPlace = [&tempFolderPlaceholder, &tempFolderPlaceholderLen, &isTempPlaceholderUsed](const wchar_t * tempsDir, std::wstring * in)
        {
            if (isTempPlaceholderUsed(in->c_str()))
            {
                in->replace(0, tempFolderPlaceholderLen, tempsDir);
            }
        };
        auto resolveResourceFilename = [&tempFolderPlaceholder, &tempFolderPlaceholderLen, &isTempPlaceholderUsed, &effectTempPath, &effectPath](const wchar_t * widePath, wchar_t * outBuf, size_t bufLen)
        {
            if (isTempPlaceholderUsed(widePath))
            {
                swprintf(outBuf, bufLen, L"%s%s", effectTempPath, widePath + tempFolderPlaceholderLen);
                return true;
            }
            else
            {
                if (effectPath[0] != L'\0')\
                {
                    swprintf(outBuf, bufLen, L"%s\\%s", effectPath, widePath);
                }
                else
                {
                    swprintf(outBuf, bufLen, L"%s", widePath);
                }
                return false;
            }
        };

        irEffect->setInputs(finalColorInput, depthInput, hudlessInput, hdrInput, colorBaseInput);

        auto colwertFragmentFormatACEFtoIR = [](acef::FragmentFormat acefFragmentFormat) -> shadermod::ir::FragmentFormat
        {
            switch (acefFragmentFormat)
            {
            case acef::FragmentFormat::kRGBA8_uint:
                return shadermod::ir::FragmentFormat::kRGBA8_uint;
            case acef::FragmentFormat::kBGRA8_uint:
                return shadermod::ir::FragmentFormat::kBGRA8_uint;

            case acef::FragmentFormat::kRGBA16_uint:
                return shadermod::ir::FragmentFormat::kRGBA16_uint;
            case acef::FragmentFormat::kRGBA16_fp:
                return shadermod::ir::FragmentFormat::kRGBA16_fp;
            case acef::FragmentFormat::kRGBA32_fp:
                return shadermod::ir::FragmentFormat::kRGBA32_fp;

            case acef::FragmentFormat::kSBGRA8_uint:
                return shadermod::ir::FragmentFormat::kSBGRA8_uint;
            case acef::FragmentFormat::kSRGBA8_uint:
                return shadermod::ir::FragmentFormat::kSRGBA8_uint;

            case acef::FragmentFormat::kR10G10B10A2_uint:
                return shadermod::ir::FragmentFormat::kR10G10B10A2_uint;

            case acef::FragmentFormat::kR11G11B10_float:
                return shadermod::ir::FragmentFormat::kR11G11B10_float;

            case acef::FragmentFormat::kRG8_uint:
                return shadermod::ir::FragmentFormat::kRG8_uint;
            case acef::FragmentFormat::kRG16_uint:
                return shadermod::ir::FragmentFormat::kRG16_uint;
            case acef::FragmentFormat::kRG32_uint:
                return shadermod::ir::FragmentFormat::kRG32_uint;
            case acef::FragmentFormat::kRG16_fp:
                return shadermod::ir::FragmentFormat::kRG16_fp;
            case acef::FragmentFormat::kRG32_fp:
                return shadermod::ir::FragmentFormat::kRG32_fp;

            case acef::FragmentFormat::kR8_uint:
                return shadermod::ir::FragmentFormat::kR8_uint;
            case acef::FragmentFormat::kR16_uint:
                return shadermod::ir::FragmentFormat::kR16_uint;
            case acef::FragmentFormat::kR16_fp:
                return shadermod::ir::FragmentFormat::kR16_fp;
            case acef::FragmentFormat::kR32_uint:
                return shadermod::ir::FragmentFormat::kR32_uint;
            case acef::FragmentFormat::kR32_fp:
                return shadermod::ir::FragmentFormat::kR32_fp;


            case acef::FragmentFormat::kD24S8:
                return shadermod::ir::FragmentFormat::kD24S8;
            case acef::FragmentFormat::kD32_fp:
                return shadermod::ir::FragmentFormat::kD32_fp;
            case acef::FragmentFormat::kD32_fp_S8X24_uint:
                return shadermod::ir::FragmentFormat::kD32_fp_S8X24_uint;


            default:
                LOG_WARN("ACEF->IR colwerter - unknown type colwersion %d", (int)acefFragmentFormat);
                return shadermod::ir::FragmentFormat::kNUM_ENTRIES;
            }
        };

        auto colwertTexParametrizedTypeACEFtoIR = [](acef::ResourceTextureParametrizedType texType)
        {
            if (texType == acef::ResourceTextureParametrizedType::kNOISE)
                return shadermod::ir::Texture::TextureType::kNoise;
            else
            {
                LOG_WARN("ACEF->IR colwerter - unknown texture parametrized type %d", (int)texType);
                return shadermod::ir::Texture::TextureType::kNoise;
            }
        };

        auto colwertSamplerAddressTypeACEFtoIR = [](acef::ResourceSamplerAddressType addrType)
        {
            switch (addrType)
            {
            case acef::ResourceSamplerAddressType::kWrap:
                return shadermod::ir::AddressType::kWrap;
            case acef::ResourceSamplerAddressType::kClamp:
                return shadermod::ir::AddressType::kClamp;
            case acef::ResourceSamplerAddressType::kMirror:
                return shadermod::ir::AddressType::kMirror;
            case acef::ResourceSamplerAddressType::kBorder:
                return shadermod::ir::AddressType::kBorder;
            default:
            {
                LOG_WARN("ACEF->IR colwerter - unknown sampler address type %d", (int)addrType);
                return shadermod::ir::AddressType::kClamp;
            }
            };
        };

        auto colwertSamplerFilterTypeACEFtoIR = [](acef::ResourceSamplerFilterType filterType)
        {
            switch (filterType)
            {
            case acef::ResourceSamplerFilterType::kPoint:
                return shadermod::ir::FilterType::kPoint;
            case acef::ResourceSamplerFilterType::kLinear:
                return shadermod::ir::FilterType::kLinear;
            default:
            {
                LOG_WARN("ACEF->IR colwerter - unknown sampler filter type %d", (int)filterType);
                return shadermod::ir::FilterType::kLinear;
            }
            };
        };

        auto colwertTexSizeBaseACEFtoIR = [](acef::TextureSizeBase texSizeBase)
        {
            switch (texSizeBase)
            {
            case acef::TextureSizeBase::kOne:
                return shadermod::ir::Texture::TextureSizeBase::kOne;
            case acef::TextureSizeBase::kColorBufferWidth:
                return shadermod::ir::Texture::TextureSizeBase::kColorBufferWidth;
            case acef::TextureSizeBase::kColorBufferHeight:
                return shadermod::ir::Texture::TextureSizeBase::kColorBufferHeight;
            case acef::TextureSizeBase::kDepthBufferWidth:
                return shadermod::ir::Texture::TextureSizeBase::kDepthBufferWidth;
            case acef::TextureSizeBase::kDepthBufferHeight:
                return shadermod::ir::Texture::TextureSizeBase::kDepthBufferHeight;
            case acef::TextureSizeBase::kTextureWidth:
                return shadermod::ir::Texture::TextureSizeBase::kTextureWidth;
            case acef::TextureSizeBase::kTextureHeight:
                return shadermod::ir::Texture::TextureSizeBase::kTextureHeight;
            default:
                LOG_WARN("ACEF->IR colwerter - unknown texture size base colwersion %d", (int)texSizeBase);
                return shadermod::ir::Texture::TextureSizeBase::kOne;
            }
        };
        const acef::ResourceHeader & resourceHeader = acefEffectStorageLoaded.resourceHeader;
        const acef::UIControlsHeader & uiControlsHeader = acefEffectStorageLoaded.uiControlsHeader;
        const acef::PassesHeader & passesHeader = acefEffectStorageLoaded.passesHeader;

        const uint32_t texIntermediateNum = resourceHeader.binStorage.texturesIntermediateNum;
        const uint32_t texParametrizedNum = resourceHeader.binStorage.texturesParametrizedNum;
        const uint32_t texFromFileNum = resourceHeader.binStorage.texturesFromFileNum;

        std::vector<ir::Texture *> irTexturesParametrized;
        irTexturesParametrized.reserve(texParametrizedNum);

        for (uint32_t texIdx = 0; texIdx < texParametrizedNum; ++texIdx)
        {
            const acef::ResourceTextureParametrized & acefTex = acefEffectStorageLoaded.texturesParametrized[texIdx];
            ir::Texture * irTex = nullptr;

            int texWidth, texHeight;

            Texture::TextureSizeBase widthBase = colwertTexSizeBaseACEFtoIR(acefTex.binStorage.width.binStorage.texSizeBase);
            float widthMul = acefTex.binStorage.width.binStorage.mul;

            if (widthBase == Texture::TextureSizeBase::kTextureWidth ||
                widthBase == Texture::TextureSizeBase::kTextureHeight)
            {
                LOG_ERROR("ACEF->IR colwerter - parametrized texture wants to have width of some file");
            }

            Texture::TextureSizeBase heightBase = colwertTexSizeBaseACEFtoIR(acefTex.binStorage.height.binStorage.texSizeBase);
            float heightMul = acefTex.binStorage.height.binStorage.mul;

            if (heightBase == Texture::TextureSizeBase::kTextureWidth ||
                heightBase == Texture::TextureSizeBase::kTextureHeight)
            {
                LOG_ERROR("ACEF->IR colwerter - parametrized texture wants to have height of some file");
            }

            // TODO: we're not adding HDR/HUDless to the size derivation procedure yet (considered not needed)
            Texture::deriveSizeDimension(finalColorInput.width, finalColorInput.height, depthInput.width, depthInput.height, widthBase, widthMul, &texWidth);
            Texture::deriveSizeDimension(finalColorInput.width, finalColorInput.height, depthInput.width, depthInput.height, heightBase, heightMul, &texHeight);

            switch (acefTex.binStorage.type)
            {
            case acef::ResourceTextureParametrizedType::kNOISE:
            {
                ir::FragmentFormat fmt = colwertFragmentFormatACEFtoIR(acefTex.binStorage.format);

                irTex = irEffect->createNoiseTexture(texWidth, texHeight, fmt);
                break;
            }
            default:
                assert("Unexpected procedural texture type!" && false);
                break;
            }

            irTex->m_widthBase = widthBase;
            irTex->m_widthMul = widthMul;

            irTex->m_heightBase = heightBase;
            irTex->m_heightMul = heightMul;

            irTex->deriveSize(finalColorInput.width, finalColorInput.height, depthInput.width, depthInput.height);

            irTexturesParametrized.push_back(irTex);
        }


        std::vector<ir::Texture *> irTexturesIntermediate;
        irTexturesIntermediate.reserve(texIntermediateNum);

        for (uint32_t texIdx = 0; texIdx < texIntermediateNum; ++texIdx)
        {
            const acef::ResourceTextureIntermediate & acefTex = acefEffectStorageLoaded.texturesIntermediate[texIdx];
            ir::Texture * irTex = nullptr;

            int texWidth, texHeight;

            Texture::TextureSizeBase widthBase = colwertTexSizeBaseACEFtoIR(acefTex.binStorage.width.binStorage.texSizeBase);
            float widthMul = acefTex.binStorage.width.binStorage.mul;

            if (widthBase == Texture::TextureSizeBase::kTextureWidth ||
                widthBase == Texture::TextureSizeBase::kTextureHeight)
            {
                LOG_ERROR("ACEF->IR colwerter - intermediate texture wants to have width of some file");
            }

            Texture::TextureSizeBase heightBase = colwertTexSizeBaseACEFtoIR(acefTex.binStorage.height.binStorage.texSizeBase);
            float heightMul = acefTex.binStorage.height.binStorage.mul;

            if (heightBase == Texture::TextureSizeBase::kTextureWidth ||
                heightBase == Texture::TextureSizeBase::kTextureHeight)
            {
                LOG_ERROR("ACEF->IR colwerter - intermediate texture wants to have height of some file");
            }

            Texture::deriveSizeDimension(finalColorInput.width, finalColorInput.height, depthInput.width, depthInput.height, widthBase, widthMul, &texWidth);
            Texture::deriveSizeDimension(finalColorInput.width, finalColorInput.height, depthInput.width, depthInput.height, heightBase, heightMul, &texHeight);

            ir::FragmentFormat fmt = colwertFragmentFormatACEFtoIR(acefTex.binStorage.format);

            irTex = irEffect->createRTTexture(texWidth, texHeight, fmt);

            irTex->m_widthBase = widthBase;
            irTex->m_widthMul = widthMul;

            irTex->m_heightBase = heightBase;
            irTex->m_heightMul = heightMul;

            if (acefTex.binStorage.levels == 0)
            {
                LOG_ERROR("ACEF->IR colwerter - intermediate texture does not specify number of mipmap levels");
            }

            irTex->m_levels = acefTex.binStorage.levels;

            irTex->deriveSize(finalColorInput.width, finalColorInput.height, depthInput.width, depthInput.height);

            irTexturesIntermediate.push_back(irTex);
        }


        std::vector<ir::Texture *> irTexturesFromFile;
        irTexturesFromFile.reserve(texFromFileNum);

        for (uint32_t texIdx = 0; texIdx < texFromFileNum; ++texIdx)
        {
            const acef::ResourceTextureFromFile & acefTex = acefEffectStorageLoaded.texturesFromFile[texIdx];

            std::string narrowPath = std::string(acefTex.pathUtf8, acefTex.binStorage.pathLen);
            std::wstring widePath = darkroom::getWstrFromUtf8(narrowPath);

            wchar_t wc_filename_texture[FILENAME_MAX];

            bool relativeToTempPath = resolveResourceFilename(widePath.c_str(), wc_filename_texture, FILENAME_MAX);
            if (relativeToTempPath)
                texturesRelativeToTempPath = true;

            int texWidth, texHeight;

            Texture::TextureSizeBase widthBase = colwertTexSizeBaseACEFtoIR(acefTex.binStorage.width.binStorage.texSizeBase);
            float widthMul = acefTex.binStorage.width.binStorage.mul;

            Texture::TextureSizeBase heightBase = colwertTexSizeBaseACEFtoIR(acefTex.binStorage.height.binStorage.texSizeBase);
            float heightMul = acefTex.binStorage.height.binStorage.mul;

            Texture::deriveSizeDimension(finalColorInput.width, finalColorInput.height, depthInput.width, depthInput.height, widthBase, widthMul, &texWidth);
            Texture::deriveSizeDimension(finalColorInput.width, finalColorInput.height, depthInput.width, depthInput.height, heightBase, heightMul, &texHeight);

            // Both Width and Height dependance result in width inheritance in this case
            // TODO: probably add fine-grained width/height dependance
            if (widthBase == Texture::TextureSizeBase::kTextureWidth ||
                widthBase == Texture::TextureSizeBase::kTextureHeight)
            {
                texWidth = Texture::SetAsInputFileSize;
            }

            // Both Width and Height dependance result in height inheritance in this case
            // TODO: probably add fine-grained width/height dependance
            if (heightBase == Texture::TextureSizeBase::kTextureWidth ||
                heightBase == Texture::TextureSizeBase::kTextureHeight)
            {
                texHeight = Texture::SetAsInputFileSize;
            }

            ir::FragmentFormat fmt = colwertFragmentFormatACEFtoIR(acefTex.binStorage.format);

            ir::Texture* irTex = irEffect->createTextureFromFile(texWidth, texHeight, fmt, wc_filename_texture);

            if (!irTex)
            {
                LOG_ERROR("ACEF->IR coolwerter - texture file not found: %s", narrowPath.c_str());
                throw MultipassConfigParserError(MultipassConfigParserErrorEnum::eFileNotfound, darkroom::getUtf8FromWstr(effectPath) + std::string("\\") + narrowPath.c_str());
            }

            // In case it is set to automatic, createTextureFromFile derived these sizes
            if (texWidth != Texture::SetAsInputFileSize)
            {
                irTex->m_widthBase = widthBase;
                irTex->m_widthMul = widthMul;
            }
            if (texHeight != Texture::SetAsInputFileSize)
            {
                irTex->m_heightBase = heightBase;
                irTex->m_heightMul = heightMul;
            }

            irTex->deriveSize(finalColorInput.width, finalColorInput.height, depthInput.width, depthInput.height);

            irTex->m_excludeHash = acefTex.binStorage.excludeHash;

            irTexturesFromFile.push_back(irTex);
        }

        enum class SystemTextureIDs
        {
            kColor = 0,
            kDepth = 1,
            kHUDless = 2,
            kHDR = 3,
            kColorBase = 4,
            kNUM_ENTRIES
        };

        std::vector<ir::Texture *> irSystemDatasources;
        irSystemDatasources.resize((int)SystemTextureIDs::kNUM_ENTRIES, nullptr);

        // Read buffers
        for (uint32_t texHandleIdx = 0, texHandleIdxEnd = resourceHeader.binStorage.readBuffersNum; texHandleIdx < texHandleIdxEnd; ++texHandleIdx)
        {
            if (resourceHeader.readBufferTextureHandles[texHandleIdx] == (uint32_t)acef::SystemTexture::kInputColor)
            {
                irSystemDatasources[(int)SystemTextureIDs::kColor] = irEffect->createInputColor();
            }
            else if (resourceHeader.readBufferTextureHandles[texHandleIdx] == (uint32_t)acef::SystemTexture::kInputDepth)
            {
                irSystemDatasources[(int)SystemTextureIDs::kDepth] = irEffect->createInputDepth();
            }
            else if (resourceHeader.readBufferTextureHandles[texHandleIdx] == (uint32_t)acef::SystemTexture::kInputHUDless)
            {
                irSystemDatasources[(int)SystemTextureIDs::kHUDless] = irEffect->createInputHUDless();
            }
            else if (resourceHeader.readBufferTextureHandles[texHandleIdx] == (uint32_t)acef::SystemTexture::kInputHDR)
            {
                irSystemDatasources[(int)SystemTextureIDs::kHDR] = irEffect->createInputHDR();
            }
            else if (resourceHeader.readBufferTextureHandles[texHandleIdx] == (uint32_t)acef::SystemTexture::kInputColorBase)
            {
                irSystemDatasources[(int)SystemTextureIDs::kColorBase] = irEffect->createInputColorBase();
            }
        }

        // Write buffers
        //  At the moment, it is considered incorrect to write to the depth buffer, at least when it wasn't requested initially as read buffers
        /*
        for (uint32_t texHandleIdx = 0, texHandleIdxEnd = resourceHeader.binStorage.writeBuffersNum; texHandleIdx < texHandleIdxEnd; ++texHandleIdx)
        {
            if (resourceHeader.writeBufferTextureHandles[texHandleIdx] == (uint32_t)acef::SystemTexture::kInputColor)
            {
                if (irSystemDatasources[(int)SystemTextureIDs::kColor] == nullptr)
                {
                    irSystemDatasources[(int)SystemTextureIDs::kColor] = irEffect->createInputColor();
                }
            }
            else if (resourceHeader.writeBufferTextureHandles[texHandleIdx] == (uint32_t)acef::SystemTexture::kInputDepth)
            {
                if (irSystemDatasources[(int)SystemTextureIDs::kDepth] == nullptr)
                {
                    irSystemDatasources[(int)SystemTextureIDs::kDepth] = irEffect->createInputDepth();
                }
            }
        }
        */

        std::vector<ir::ConstantBuf *> irCbuffers;
        irCbuffers.reserve(resourceHeader.binStorage.constantBuffersNum);

        auto colwertSystemConstACEFtoIR = [](uint32_t constHandle) -> shadermod::ir::ConstType
        {
            switch (constHandle)
            {
            case (uint32_t)acef::SystemConstant::kCaptureState:
                return shadermod::ir::ConstType::kCaptureState;
            case (uint32_t)acef::SystemConstant::kDT:
                return shadermod::ir::ConstType::kDT;
            case (uint32_t)acef::SystemConstant::kElapsedTime:
                return shadermod::ir::ConstType::kElapsedTime;
            case (uint32_t)acef::SystemConstant::kFrame:
                return shadermod::ir::ConstType::kFrame;
            case (uint32_t)acef::SystemConstant::kScreenSize:
                return shadermod::ir::ConstType::kScreenSize;
            case (uint32_t)acef::SystemConstant::kTileUV:
                return shadermod::ir::ConstType::kTileUV;
            case (uint32_t)acef::SystemConstant::kDepthAvailable:
                return shadermod::ir::ConstType::kDepthAvailable;
            case (uint32_t)acef::SystemConstant::kHDRAvailable:
                return shadermod::ir::ConstType::kHDRAvailable;
            case (uint32_t)acef::SystemConstant::kHUDlessAvailable:
                return shadermod::ir::ConstType::kHUDlessAvailable;
            default:
                {
                    LOG_WARN("ACEF->IR coolwerter - unknown constant type %d", (int)constHandle);
                    return shadermod::ir::ConstType::kCaptureState;
                }
            }
        };

        auto colwertUIDataTypeACEFtoIR = [](acef::UserConstDataType dataType) -> shadermod::ir::UserConstDataType
        {
            switch (dataType)
            {
            case acef::UserConstDataType::kInt:
                return shadermod::ir::UserConstDataType::kInt;
            case acef::UserConstDataType::kUInt:
                return shadermod::ir::UserConstDataType::kUInt;
            case acef::UserConstDataType::kFloat:
                return shadermod::ir::UserConstDataType::kFloat;
            case acef::UserConstDataType::kBool:
                return shadermod::ir::UserConstDataType::kBool;
            default:
                {
                    LOG_WARN("ACEF->IR colwerter - unknown user constant data type %d", (int)dataType);
                    return shadermod::ir::UserConstDataType::kFloat;
                }
            }
        };

        for (uint32_t idx = 0, idxEnd = resourceHeader.binStorage.constantBuffersNum; idx < idxEnd; ++idx)
        {
            const acef::ResourceConstantBuffer & acefConstBuf = acefEffectStorageLoaded.constantBuffers[idx];
            ir::ConstantBuf * irConstBuf = irEffect->createConstantBuffer();

            for (uint32_t constIdx = 0, constIdxEnd = acefConstBuf.binStorage.constantsNum; constIdx < constIdxEnd; ++constIdx)
            {
                uint32_t acefConstHandle = acefConstBuf.constantHandle[constIdx];

                std::string constantName;
                if (acefConstBuf.constantNameLens[constIdx] > 0)
                {
                    constantName = std::string(acefConstBuf.constantNames+acefConstBuf.constantNameOffsets[constIdx], acefConstBuf.constantNameLens[constIdx]);
                }
                uint32_t constantOffset = acefConstBuf.constantOffsetInComponents[constIdx];

                if (acef::isConstantSystem(acefConstHandle))
                {
                    ir::ConstType cType = colwertSystemConstACEFtoIR(acefConstHandle);
                    ir::Constant * irConst = nullptr;
                    if (constantName.length() > 0)
                    {
                        irConst = irEffect->createConstant(cType, constantName.c_str());
                    }
                    else
                    {
                        irConst = irEffect->createConstant(cType, constantOffset);
                    }
                    irConstBuf->addConstant(irConst);
                }
                else
                {
                    std::string controlNameAscii(acefEffectStorageLoaded.userConstants[acefConstHandle].controlNameAscii, acefEffectStorageLoaded.userConstants[acefConstHandle].binStorage.controlNameLen);
                    ir::Constant * irConst = nullptr;
                    if (constantName.length() > 0)
                    {
                        irConst = irEffect->createConstant(controlNameAscii.c_str(), constantName.c_str());
                    }
                    else
                    {
                        irConst = irEffect->createConstant(controlNameAscii.c_str(), constantOffset);
                    }
                    irConstBuf->addConstant(irConst);
                }
            }

            irCbuffers.push_back(irConstBuf);
        }

        std::vector<ir::Sampler *> irSamplers;
        irSamplers.reserve(resourceHeader.binStorage.samplersNum);

        for (uint32_t idx = 0, idxEnd = resourceHeader.binStorage.samplersNum; idx < idxEnd; ++idx)
        {
            const acef::ResourceSampler & sampler = acefEffectStorageLoaded.samplers[idx];

            ir::AddressType addTypeU = colwertSamplerAddressTypeACEFtoIR(sampler.binStorage.addrU);
            ir::AddressType addTypeV = colwertSamplerAddressTypeACEFtoIR(sampler.binStorage.addrV);
            ir::AddressType addTypeW = colwertSamplerAddressTypeACEFtoIR(sampler.binStorage.addrW);
            ir::FilterType fltMin = colwertSamplerFilterTypeACEFtoIR(sampler.binStorage.filterMin);
            ir::FilterType fltMag = colwertSamplerFilterTypeACEFtoIR(sampler.binStorage.filterMag);
            ir::FilterType fltMip = colwertSamplerFilterTypeACEFtoIR(sampler.binStorage.filterMip);

            ir::Sampler * irSampler = irEffect->createSampler(addTypeU, addTypeV, fltMin, fltMag, fltMip);
            irSamplers.push_back(irSampler);
        }

        ir::UserConstantManager & constMan = irEffect->getUserConstantManager();
        for (uint32_t idx = 0, idxEnd = uiControlsHeader.binStorage.userConstantsNum; idx < idxEnd; ++idx)
        {
            const acef::UserConstant & acefUserConstant = acefEffectStorageLoaded.userConstants[idx];

            unsigned int dimensions = acefUserConstant.binStorage.dataDimensionality;
            
            auto colwertTypelessVariableACEFtoIR = [](const acef::TypelessVariableStorage & data, acef::UserConstDataType type, unsigned int dims) -> shadermod::ir::TypelessVariable
            {
                if (type == acef::UserConstDataType::kBool)
                {
                    return ir::TypelessVariable(reinterpret_cast<const bool *>(data.binStorage.data), dims);
                }
                else if (type == acef::UserConstDataType::kFloat)
                {
                    return ir::TypelessVariable(reinterpret_cast<const float *>(data.binStorage.data), dims);
                }
                else if (type == acef::UserConstDataType::kInt)
                {
                    return ir::TypelessVariable(reinterpret_cast<const int *>(data.binStorage.data), dims);
                }
                else if (type == acef::UserConstDataType::kUInt)
                {
                    return ir::TypelessVariable(reinterpret_cast<const unsigned int *>(data.binStorage.data), dims);
                }
                else
                {
                    LOG_WARN("ACEF->IR colwerter - failed to colwert user constant data: unknown type %d", (int)type);
                    return ir::TypelessVariable();
                }
            };

            auto colwertUIControlTypeACEFtoIR = [](acef::UIControlType controlType) -> shadermod::ir::UiControlType
            {
                switch (controlType)
                {
                case acef::UIControlType::kCheckbox:
                    return shadermod::ir::UiControlType::kCheckbox;
                case acef::UIControlType::kEditbox:
                    return shadermod::ir::UiControlType::kEditbox;
                case acef::UIControlType::kFlyout:
                    return shadermod::ir::UiControlType::kFlyout;
                case acef::UIControlType::kSlider:
                    return shadermod::ir::UiControlType::kSlider;
                case acef::UIControlType::kColorPicker:
                    return shadermod::ir::UiControlType::kColorPicker;
                case acef::UIControlType::kRadioButton:
                    return shadermod::ir::UiControlType::kRadioButton;
                default:
                    {
                        LOG_WARN("ACEF->IR colwerter - unknown user constant control type %d", (int)controlType);
                        return shadermod::ir::UiControlType::kSlider;
                    }
                }
            };

            auto buildLocalizationMapACEFtoIR = [](const acef::UILocalizedStringStorage & strStorage, const acef::UILocalizedStringBuffers & strBuffers) -> std::map<unsigned short, std::string>
            {
                std::map<unsigned short, std::string> result;

                // Default string is processed separately
                const uint32_t localizationsNum = strStorage.binStorage.localizationsNum;
                for (uint32_t locIdx = 0; locIdx < localizationsNum; ++locIdx)
                {
                    const acef::UILocalizedStringBuffers::LocalizedString & lwrLocStr = strBuffers.localizedStrings[locIdx];
                    auto it = result.insert(std::make_pair( (unsigned short)lwrLocStr.binStorage.langid, std::string(lwrLocStr.stringUtf8, lwrLocStr.binStorage.strLen) ));
                    assert(it.second);
                }

                return result;
            };

            auto buildOptionsListACEFtoIR = [&colwertTypelessVariableACEFtoIR, &buildLocalizationMapACEFtoIR](
                uint16_t optionsNum,
                uint16_t optionDefault,
                acef::UserConstDataType dataType,
                unsigned int dimensions,
                acef::TypelessVariableStorage * optiolwalues,
                acef::UILocalizedStringStorage * optionNameStorages,
                acef::UILocalizedStringBuffers * optionNamesBuffers

                ) -> ir::UserConstant::ListOptions
            {
                std::vector<ir::UserConstant::ListOption> options;
                options.reserve(optionsNum);

                for (uint32_t optIdx = 0; optIdx < (uint32_t)optionsNum; ++optIdx)
                {
                    options.push_back(
                        ir::UserConstant::ListOption(colwertTypelessVariableACEFtoIR(optiolwalues[optIdx], dataType, dimensions),
                            ir::UserConstant::StringWithLocalization(
                                std::string(optionNamesBuffers[optIdx].defaultStringAscii, optionNameStorages[optIdx].binStorage.defaultStringLen),
                                buildLocalizationMapACEFtoIR(optionNameStorages[optIdx], optionNamesBuffers[optIdx])
                            )
                        )
                    );

                }


                return ir::UserConstant::ListOptions(options, optionDefault);
            };

            // valueDisplayName contains the localized names of individual elements within a multi-dimensional variable. For example, a 3D RGB color will have individual element names: "Red", "Green", "Blue"
            std::vector<UserConstant::StringWithLocalization> valueDisplayName;
            for (uint8_t dimIdx = 0; dimIdx < acefUserConstant.binStorage.dataDimensionality; dimIdx++)
            {
                valueDisplayName.push_back(ir::UserConstant::StringWithLocalization(
                    std::string(acefUserConstant.variableNamesBuffers[dimIdx].defaultStringAscii, acefUserConstant.variableNames[dimIdx].binStorage.defaultStringLen),
                    buildLocalizationMapACEFtoIR(acefUserConstant.variableNames[dimIdx], acefUserConstant.variableNamesBuffers[dimIdx])));
            }

            // TODO avoroshilov: this is NOT ok, remove variable argument c-tor
            ir::UserConstant * uc = constMan.pushBackUserConstant(
                std::string(acefUserConstant.controlNameAscii, acefUserConstant.binStorage.controlNameLen),
                colwertUIDataTypeACEFtoIR(acefUserConstant.binStorage.dataType),
                //acefUserConstant.binStorage.dataDimensionality,
                colwertUIControlTypeACEFtoIR(acefUserConstant.binStorage.uiControlType),
                colwertTypelessVariableACEFtoIR(acefUserConstant.binStorage.defaultValue, acefUserConstant.binStorage.dataType, dimensions),
                ir::UserConstant::StringWithLocalization(
                    std::string(acefUserConstant.labelBuffers.defaultStringAscii, acefUserConstant.binStorage.label.binStorage.defaultStringLen),
                    buildLocalizationMapACEFtoIR(acefUserConstant.binStorage.label, acefUserConstant.labelBuffers)
                ),
                colwertTypelessVariableACEFtoIR(acefUserConstant.binStorage.minimumValue, acefUserConstant.binStorage.dataType, dimensions),
                colwertTypelessVariableACEFtoIR(acefUserConstant.binStorage.maximumValue, acefUserConstant.binStorage.dataType, dimensions),
                colwertTypelessVariableACEFtoIR(acefUserConstant.binStorage.uiValueStep, acefUserConstant.binStorage.dataType, dimensions),
                ir::UserConstant::StringWithLocalization(
                    std::string(acefUserConstant.hintBuffers.defaultStringAscii, acefUserConstant.binStorage.hint.binStorage.defaultStringLen),
                    buildLocalizationMapACEFtoIR(acefUserConstant.binStorage.hint, acefUserConstant.hintBuffers)
                ),
                ir::UserConstant::StringWithLocalization(
                    std::string(acefUserConstant.uiValueUnitBuffers.defaultStringAscii, acefUserConstant.binStorage.uiValueUnit.binStorage.defaultStringLen),
                    buildLocalizationMapACEFtoIR(acefUserConstant.binStorage.uiValueUnit, acefUserConstant.uiValueUnitBuffers)
                ),
                acefUserConstant.binStorage.stickyValue,
                acefUserConstant.binStorage.stickyRegion,
                colwertTypelessVariableACEFtoIR(acefUserConstant.binStorage.uiMinimumValue, acefUserConstant.binStorage.dataType, dimensions),
                colwertTypelessVariableACEFtoIR(acefUserConstant.binStorage.uiMaximumValue, acefUserConstant.binStorage.dataType, dimensions),
                buildOptionsListACEFtoIR(
                    acefUserConstant.binStorage.optionsNum,
                    acefUserConstant.binStorage.optionDefault,
                    acefUserConstant.binStorage.dataType,
                    dimensions,
                    acefUserConstant.optiolwalues,
                    acefUserConstant.optionNames,
                    acefUserConstant.optionNamesBuffers
                ),
                valueDisplayName
            );

            if (!uc)
            {
                LOG_ERROR("ACEF->IR colwerter - failed to create user constant #%d", (int)idx);

                throw MultipassConfigParserError(MultipassConfigParserErrorEnum::eCreateUserConstantFailed,
                    std::string("user constant name ") + std::string(acefUserConstant.controlNameAscii, acefUserConstant.binStorage.controlNameLen));
            }
        }

        auto colwertRasterFillModeACEFtoIR = [](const acef::RasterizerFillMode & acefRasterFillMode) -> RasterizerFillMode
        {
            switch (acefRasterFillMode)
            {
            case acef::RasterizerFillMode::kSolid:
                return RasterizerFillMode::kSolid;
            case acef::RasterizerFillMode::kWireframe:
                return RasterizerFillMode::kWireframe;
            default:
                {
                    LOG_WARN("ACEF->IR colwersion - unknown RasterizerFillMode");
                    return RasterizerFillMode::kSolid;
                }
            }
        };

        auto colwertRasterLwllModeACEFtoIR = [](const acef::RasterizerLwllMode & acefRasterLwllMode) -> RasterizerLwllMode
        {
            switch (acefRasterLwllMode)
            {
            case acef::RasterizerLwllMode::kBack:
                return RasterizerLwllMode::kBack;
            case acef::RasterizerLwllMode::kFront:
                return RasterizerLwllMode::kFront;
            case acef::RasterizerLwllMode::kNone:
                return RasterizerLwllMode::kNone;
            default:
                {
                    LOG_WARN("ACEF->IR colwersion - unknown RasterizerLwllMode");
                    return RasterizerLwllMode::kNone;
                }
            }
        };

        auto colwertRasterizerStateACEFtoIR = [&colwertRasterFillModeACEFtoIR, &colwertRasterLwllModeACEFtoIR](const acef::RasterizerStateStorage & acefRasterState) -> RasterizerState
        {
            RasterizerState irRasterState;

            irRasterState.fillMode = colwertRasterFillModeACEFtoIR(acefRasterState.binStorage.fillMode);
            irRasterState.lwllMode = colwertRasterLwllModeACEFtoIR(acefRasterState.binStorage.lwllMode);

            irRasterState.depthBias = acefRasterState.binStorage.depthBias;
            irRasterState.depthBiasClamp = acefRasterState.binStorage.depthBiasClamp;
            irRasterState.slopeScaledDepthBias = acefRasterState.binStorage.slopeScaledDepthBias;

            irRasterState.frontCounterClockwise = acefRasterState.binStorage.frontCounterClockwise;
            irRasterState.depthClipEnable = acefRasterState.binStorage.depthClipEnable;
            irRasterState.scissorEnable = acefRasterState.binStorage.scissorEnable;
            irRasterState.multisampleEnable = acefRasterState.binStorage.multisampleEnable;
            irRasterState.antialiasedLineEnable = acefRasterState.binStorage.antialiasedLineEnable;

            return irRasterState;
        };

        auto colwertComparisonFuncACEFtoIR = [](const acef::ComparisonFunc & acefComparisonFunc) -> ComparisonFunc
        {
            switch (acefComparisonFunc)
            {
            case acef::ComparisonFunc::kAlways:
                return ComparisonFunc::kAlways;
            case acef::ComparisonFunc::kEqual:
                return ComparisonFunc::kEqual;
            case acef::ComparisonFunc::kGreater:
                return ComparisonFunc::kGreater;
            case acef::ComparisonFunc::kGreaterEqual:
                return ComparisonFunc::kGreaterEqual;
            case acef::ComparisonFunc::kLess:
                return ComparisonFunc::kLess;
            case acef::ComparisonFunc::kLessEqual:
                return ComparisonFunc::kLessEqual;
            case acef::ComparisonFunc::kNever:
                return ComparisonFunc::kNever;
            case acef::ComparisonFunc::kNotEqual:
                return ComparisonFunc::kNotEqual;
            default:
                {
                    LOG_WARN("ACEF->IR colwersion - unknown ComparisonFunc");
                    return ComparisonFunc::kAlways;
                }
            }
        };

        auto colwertStencilOpACEFtoIR = [](const acef::StencilOp & acefStencilOp) -> StencilOp
        {
            switch (acefStencilOp)
            {
            case acef::StencilOp::kDecr:
                return StencilOp::kDecr;
            case acef::StencilOp::kDecrSat:
                return StencilOp::kDecrSat;
            case acef::StencilOp::kIncr:
                return StencilOp::kIncr;
            case acef::StencilOp::kIncrSat:
                return StencilOp::kIncrSat;
            case acef::StencilOp::kIlwert:
                return StencilOp::kIlwert;
            case acef::StencilOp::kKeep:
                return StencilOp::kKeep;
            case acef::StencilOp::kReplace:
                return StencilOp::kReplace;
            case acef::StencilOp::kZero:
                return StencilOp::kZero;
            default:
                {
                    LOG_WARN("ACEF->IR colwersion - unknown StencilOp");
                    return StencilOp::kKeep;
                }
            }
        };

        auto colwertDepthStencilOpACEFtoIR = [&colwertStencilOpACEFtoIR, &colwertComparisonFuncACEFtoIR](const acef::DepthStencilOpStorage & acefDepthStencilOp) -> DepthStencilOp
        {
            DepthStencilOp irDepthStencilOp;

            irDepthStencilOp.failOp = colwertStencilOpACEFtoIR(acefDepthStencilOp.binStorage.failOp);
            irDepthStencilOp.depthFailOp = colwertStencilOpACEFtoIR(acefDepthStencilOp.binStorage.depthFailOp);
            irDepthStencilOp.passOp = colwertStencilOpACEFtoIR(acefDepthStencilOp.binStorage.depthFailOp);
            irDepthStencilOp.func = colwertComparisonFuncACEFtoIR(acefDepthStencilOp.binStorage.func);

            return irDepthStencilOp;
        };

        auto colwertDepthWriteMaskACEFtoIR = [](const acef::DepthWriteMask & acefDepthWriteMask) -> DepthWriteMask
        {
            switch (acefDepthWriteMask)
            {
            case acef::DepthWriteMask::kAll:
                return DepthWriteMask::kAll;
            case acef::DepthWriteMask::kZero:
                return DepthWriteMask::kZero;
            default:
                {
                    LOG_WARN("ACEF->IR colwersion - unknown DepthWriteMask");
                    return DepthWriteMask::kAll;
                }
            }
        };

        auto colwertDepthStencilStateACEFtoIR = [&colwertDepthStencilOpACEFtoIR, &colwertComparisonFuncACEFtoIR, &colwertDepthWriteMaskACEFtoIR](const acef::DepthStencilStateStorage & acefDepthStencilState) -> DepthStencilState
        {
            DepthStencilState irDepthStencilState;

            irDepthStencilState.frontFace = colwertDepthStencilOpACEFtoIR(acefDepthStencilState.binStorage.frontFace);
            irDepthStencilState.backFace = colwertDepthStencilOpACEFtoIR(acefDepthStencilState.binStorage.backFace);

            irDepthStencilState.depthWriteMask = colwertDepthWriteMaskACEFtoIR(acefDepthStencilState.binStorage.depthWriteMask);
            irDepthStencilState.depthFunc = colwertComparisonFuncACEFtoIR(acefDepthStencilState.binStorage.depthFunc);

            irDepthStencilState.stencilReadMask = acefDepthStencilState.binStorage.stencilReadMask;
            irDepthStencilState.stencilWriteMask = acefDepthStencilState.binStorage.stencilWriteMask;
            irDepthStencilState.isDepthEnabled = acefDepthStencilState.binStorage.isDepthEnabled;
            irDepthStencilState.isStencilEnabled = acefDepthStencilState.binStorage.isStencilEnabled;

            return irDepthStencilState;
        };

        auto colwertBlendCoefACEFtoIR = [](const acef::BlendCoef & acefBlendCoef, bool isBufferActive) -> BlendCoef
        {
            switch (acefBlendCoef)
            {
            case acef::BlendCoef::kZero:
                return BlendCoef::kZero;
            case acef::BlendCoef::kOne:
                return BlendCoef::kOne;
            case acef::BlendCoef::kSrcColor:
                return BlendCoef::kSrcColor;
            case acef::BlendCoef::kIlwSrcColor:
                return BlendCoef::kIlwSrcColor;
            case acef::BlendCoef::kSrcAlpha:
                return BlendCoef::kSrcAlpha;
            case acef::BlendCoef::kIlwSrcAlpha:
                return BlendCoef::kIlwSrcAlpha;
            case acef::BlendCoef::kDstAlpha:
                return BlendCoef::kDstAlpha;
            case acef::BlendCoef::kIlwDstAlpha:
                return BlendCoef::kIlwDstAlpha;
            case acef::BlendCoef::kDstColor:
                return BlendCoef::kDstColor;
            case acef::BlendCoef::kIlwDstColor:
                return BlendCoef::kIlwDstColor;
            case acef::BlendCoef::kSrcAlphaSat:
                return BlendCoef::kSrcAlphaSat;
            case acef::BlendCoef::kBlendFactor:
                return BlendCoef::kBlendFactor;
            case acef::BlendCoef::kIlwBlendFactor:
                return BlendCoef::kIlwBlendFactor;
            case acef::BlendCoef::kSrc1Color:
                return BlendCoef::kSrc1Color;
            case acef::BlendCoef::kIlwSrc1Color:
                return BlendCoef::kIlwSrc1Color;
            case acef::BlendCoef::kSrc1Alpha:
                return BlendCoef::kSrc1Alpha;
            case acef::BlendCoef::kIlwSrc1Alpha:
                return BlendCoef::kIlwSrc1Alpha;
            default:
                {
                    if (isBufferActive)
                        LOG_WARN("ACEF->IR colwersion - unknown BlendCoef: %d", (int)acefBlendCoef);
                    else
                        LOG_DEBUG("ACEF->IR colwersion - unknown BlendCoef for inactive buffer: %d", (int)acefBlendCoef);

                    return BlendCoef::kOne;
                }
            }
        };

        auto colwertBlendOpACEFtoIR = [](const acef::BlendOp & acefBlendOp, bool isBufferActive) -> BlendOp
        {
            switch (acefBlendOp)
            {
            case acef::BlendOp::kAdd:
                return BlendOp::kAdd;
            case acef::BlendOp::kSub:
                return BlendOp::kSub;
            case acef::BlendOp::kRevSub:
                return BlendOp::kRevSub;
            case acef::BlendOp::kMin:
                return BlendOp::kMin;
            case acef::BlendOp::kMax:
                return BlendOp::kMax;
            default:
                {
                    if (isBufferActive)
                        LOG_WARN("ACEF->IR colwersion - unknown BlendOp: %d", (int)acefBlendOp);
                    else
                        LOG_DEBUG("ACEF->IR colwersion - unknown BlendOp for inactive buffer: %d", (int)acefBlendOp);

                    return BlendOp::kAdd;
                }
            }
        };

        auto colwertColorWriteEnableBitsACEFtoIR = [](const acef::ColorWriteEnableBits & acefColorWriteEnable, bool isBufferActive) -> ColorWriteEnableBits
        {
            switch (acefColorWriteEnable)
            {
            case acef::ColorWriteEnableBits::kRed:
                return ColorWriteEnableBits::kRed;
            case acef::ColorWriteEnableBits::kGreen:
                return ColorWriteEnableBits::kGreen;
            case acef::ColorWriteEnableBits::kBlue:
                return ColorWriteEnableBits::kBlue;
            case acef::ColorWriteEnableBits::kAlpha:
                return ColorWriteEnableBits::kAlpha;
            case acef::ColorWriteEnableBits::kAll:
                return ColorWriteEnableBits::kAll;
            default:
                {
                    if (isBufferActive)
                        LOG_WARN("ACEF->IR colwersion - unknown ColorWriteEnableBits: %d", (int)acefColorWriteEnable);
                    else
                        LOG_DEBUG("ACEF->IR colwersion - unknown ColorWriteEnableBits for inactive buffer: %d", (int)acefColorWriteEnable);

                    return ColorWriteEnableBits::kAll;
                }
            }
        };

        auto colwertAlphaBlendRenderTargetStateACEFtoIR = [&colwertBlendCoefACEFtoIR, &colwertBlendOpACEFtoIR, &colwertColorWriteEnableBitsACEFtoIR](const acef::AlphaBlendRenderTargetStateStorage & acefAlphaBlendRenderTargetState, bool isBufferActive) -> AlphaBlendRenderTargetState
        {
            AlphaBlendRenderTargetState irAlphaBlendRenderTargetState;

            irAlphaBlendRenderTargetState.src = colwertBlendCoefACEFtoIR(acefAlphaBlendRenderTargetState.binStorage.src, isBufferActive);
            irAlphaBlendRenderTargetState.dst = colwertBlendCoefACEFtoIR(acefAlphaBlendRenderTargetState.binStorage.dst, isBufferActive);
            irAlphaBlendRenderTargetState.op = colwertBlendOpACEFtoIR(acefAlphaBlendRenderTargetState.binStorage.op, isBufferActive);

            irAlphaBlendRenderTargetState.srcAlpha = colwertBlendCoefACEFtoIR(acefAlphaBlendRenderTargetState.binStorage.srcAlpha, isBufferActive);
            irAlphaBlendRenderTargetState.dstAlpha = colwertBlendCoefACEFtoIR(acefAlphaBlendRenderTargetState.binStorage.dstAlpha, isBufferActive);
            irAlphaBlendRenderTargetState.opAlpha = colwertBlendOpACEFtoIR(acefAlphaBlendRenderTargetState.binStorage.opAlpha, isBufferActive);

            irAlphaBlendRenderTargetState.renderTargetWriteMask = colwertColorWriteEnableBitsACEFtoIR(acefAlphaBlendRenderTargetState.binStorage.renderTargetWriteMask, isBufferActive);
            irAlphaBlendRenderTargetState.isEnabled = acefAlphaBlendRenderTargetState.binStorage.isEnabled;

            return irAlphaBlendRenderTargetState;
        };

        auto colwertAlphaBlendStateACEFtoIR = [&colwertAlphaBlendRenderTargetStateACEFtoIR](const acef::AlphaBlendStateStorage & acefAlphaBlendState, unsigned int numRTs) -> AlphaBlendState
        {
            AlphaBlendState irAlphaBlendState;

            for (uint32_t idx = 0; idx < acef::AlphaBlendStateStorage::renderTargetsNum; ++idx)
            {
                irAlphaBlendState.renderTargetState[idx] = colwertAlphaBlendRenderTargetStateACEFtoIR(acefAlphaBlendState.binStorage.renderTargetState[idx], (idx < numRTs));
            }

            irAlphaBlendState.alphaToCoverageEnable = acefAlphaBlendState.binStorage.alphaToCoverageEnable;
            irAlphaBlendState.independentBlendEnable = acefAlphaBlendState.binStorage.independentBlendEnable;

            return irAlphaBlendState;
        };


        //auto& shaderPasses = parser.getShaderPassDatasources();
        std::vector<ir::Pass *> irPasses;
        irPasses.reserve(passesHeader.binStorage.passesNum);

        for (uint32_t passIdx = 0; passIdx < passesHeader.binStorage.passesNum; ++passIdx)
        {
            const acef::Pass & acefPass = acefEffectStorageLoaded.passes[passIdx];

            wchar_t wc_filename_shader[FILENAME_MAX];
            std::string entryPointName;

            // VS
            ir::VertexShader * vs = nullptr;
             if (acefPass.binStorage.vertexShaderIndex != acef::Pass::vertexShaderDefault)
            {
                const acef::ResourceVertexShader & acefVS = acefEffectStorageLoaded.vertexShaders[acefPass.binStorage.vertexShaderIndex];
                std::wstring vs_widePath = darkroom::getWstrFromUtf8(std::string(acefVS.filePathUtf8, acefVS.binStorage.filePathLen));

                bool relativeToTempPath = resolveResourceFilename(vs_widePath.c_str(), wc_filename_shader, FILENAME_MAX);
                if (relativeToTempPath)
                    shadersRelativeToTempPath = true;

                entryPointName = std::string(acefVS.entryFunctionAscii, acefVS.binStorage.entryFunctionLen);
                vs = irEffect->createVertexShader(wc_filename_shader, entryPointName.c_str());
            }

            // PS
            const acef::ResourcePixelShader & acefPS = acefEffectStorageLoaded.pixelShaders[acefPass.binStorage.pixelShaderIndex];
            std::wstring ps_widePath = darkroom::getWstrFromUtf8(std::string(acefPS.filePathUtf8, acefPS.binStorage.filePathLen));

            bool relativeToTempPath = resolveResourceFilename(ps_widePath.c_str(), wc_filename_shader, FILENAME_MAX);
            if (relativeToTempPath)
                shadersRelativeToTempPath = true;

            entryPointName = std::string(acefPS.entryFunctionAscii, acefPS.binStorage.entryFunctionLen);
            ir::PixelShader * ps = irEffect->createPixelShader(wc_filename_shader, entryPointName.c_str());

            std::vector<ir::FragmentFormat> fragmentFormats;
            fragmentFormats.reserve(acefPass.binStorage.writeBuffersNum);

            for (uint32_t wbIdx = 0, wbIdxEnd = acefPass.binStorage.writeBuffersNum; wbIdx < wbIdxEnd; ++wbIdx)
            {
                uint32_t wbHandleIdx = acefPass.writeBuffersIndices[wbIdx];
                uint32_t wbHandle = resourceHeader.writeBufferTextureHandles[wbHandleIdx];

                if (wbHandle == (uint32_t)acef::SystemTexture::kInputColor)
                {
                    fragmentFormats.push_back(finalColorInput.format);
                }
                else if (wbHandle == (uint32_t)acef::SystemTexture::kInputDepth)
                {
                    fragmentFormats.push_back(depthInput.format);
                }
                else if (wbHandle == (uint32_t)acef::SystemTexture::kInputHUDless)
                {
                    fragmentFormats.push_back(hudlessInput.format);
                }
                else if (wbHandle == (uint32_t)acef::SystemTexture::kInputHDR)
                {
                    fragmentFormats.push_back(hdrInput.format);
                }
                else if (wbHandle == (uint32_t)acef::SystemTexture::kInputColorBase)
                {
                    fragmentFormats.push_back(colorBaseInput.format);
                }
                else
                {
                    acef::FragmentFormat acefFmt;

                    // TODO avoroshilov ACEF: implement "inherit format from Color/Depth buffers"
                    if (wbHandle < resourceHeader.binStorage.texturesParametrizedNum)
                    {
                        uint32_t texIdx = wbHandle;
                        acefFmt = acefEffectStorageLoaded.texturesParametrized[texIdx].binStorage.format;
                    }
                    else if (wbHandle < resourceHeader.binStorage.texturesParametrizedNum + resourceHeader.binStorage.texturesIntermediateNum)
                    {
                        uint32_t texIdx = wbHandle - resourceHeader.binStorage.texturesParametrizedNum;
                        acefFmt = acefEffectStorageLoaded.texturesIntermediate[texIdx].binStorage.format;
                    }
                    else if (wbHandle < resourceHeader.binStorage.texturesParametrizedNum + resourceHeader.binStorage.texturesIntermediateNum + resourceHeader.binStorage.texturesFromFileNum)
                    {
                        uint32_t texIdx = wbHandle - (resourceHeader.binStorage.texturesParametrizedNum + resourceHeader.binStorage.texturesIntermediateNum);
                        acefFmt = acefEffectStorageLoaded.texturesFromFile[texIdx].binStorage.format;
                    }
                    fragmentFormats.push_back(colwertFragmentFormatACEFtoIR(acefFmt));
                }
            }

            ir::Pass * pass = irEffect->createPass(
                vs,
                ps,
                ir::Pass::SetAsEffectInputSize,
                ir::Pass::SetAsEffectInputSize,
                fragmentFormats.data(),
                (int)fragmentFormats.size()
                );

            pass->setSizeScale(1.0f, 1.0f);

            pass->m_rasterizerState = colwertRasterizerStateACEFtoIR(acefPass.binStorage.rasterizerState);
            pass->m_depthStencilState = colwertDepthStencilStateACEFtoIR(acefPass.binStorage.depthStencilState);
            pass->m_alphaBlendState = colwertAlphaBlendStateACEFtoIR(acefPass.binStorage.alphaBlendState, acefPass.binStorage.writeBuffersNum);


            for (uint32_t smpIdx = 0, smpIdxEnd = acefPass.binStorage.samplersNum; smpIdx < smpIdxEnd; ++smpIdx)
            {
                if (acefPass.samplersNameLens[smpIdx] > 0)
                {
                    std::string samplerName(acefPass.samplersNames+acefPass.samplersNameOffsets[smpIdx], acefPass.samplersNameLens[smpIdx]);
                    pass->addSampler(irSamplers[acefPass.samplersIndices[smpIdx]], samplerName.c_str());
                }
                else
                {
                    pass->addSampler(irSamplers[acefPass.samplersIndices[smpIdx]], acefPass.samplersSlots[smpIdx]);
                }
            }

            for (uint32_t cbufIdx = 0, cbufIdxEnd = acefPass.binStorage.constantBuffersVSNum; cbufIdx < cbufIdxEnd; ++cbufIdx)
            {
                if (acefPass.constantBuffersVSNameLens[cbufIdx] > 0)
                {
                    std::string cbufName(acefPass.constantBuffersVSNames+acefPass.constantBuffersVSNameOffsets[cbufIdx], acefPass.constantBuffersVSNameLens[cbufIdx]);
                    pass->addConstantBufferVS(irCbuffers[acefPass.constantBuffersVSIndices[cbufIdx]], cbufName.c_str());
                }
                else
                {
                    pass->addConstantBufferVS(irCbuffers[acefPass.constantBuffersVSIndices[cbufIdx]], acefPass.constantBuffersVSSlots[cbufIdx]);
                }
            }

            for (uint32_t cbufIdx = 0, cbufIdxEnd = acefPass.binStorage.constantBuffersPSNum; cbufIdx < cbufIdxEnd; ++cbufIdx)
            {
                if (acefPass.constantBuffersPSNameLens[cbufIdx] > 0)
                {
                    std::string cbufName(acefPass.constantBuffersPSNames+acefPass.constantBuffersPSNameOffsets[cbufIdx], acefPass.constantBuffersPSNameLens[cbufIdx]);
                    pass->addConstantBufferPS(irCbuffers[acefPass.constantBuffersPSIndices[cbufIdx]], cbufName.c_str());
                }
                else
                {
                    pass->addConstantBufferPS(irCbuffers[acefPass.constantBuffersPSIndices[cbufIdx]], acefPass.constantBuffersPSSlots[cbufIdx]);
                }
            }

            for (uint32_t srcIdx = 0, srcIdxEnd = acefPass.binStorage.readBuffersNum; srcIdx < srcIdxEnd; ++srcIdx)
            {
                std::string readBufName;
                if (acefPass.readBuffersNameLens[srcIdx] > 0)
                {
                    readBufName = std::string(acefPass.readBuffersNames+acefPass.readBuffersNameOffsets[srcIdx], acefPass.readBuffersNameLens[srcIdx]);
                }
                uint32_t readBufSlot = acefPass.readBuffersSlots[srcIdx];

                uint32_t readBufIdx = acefPass.readBuffersIndices[srcIdx];
                uint32_t readBufHandle = resourceHeader.readBufferTextureHandles[readBufIdx];

                ir::Texture * irTex = nullptr;

                if (readBufHandle == (uint32_t)acef::SystemTexture::kInputColor)
                {
                    irTex = irSystemDatasources[(int)SystemTextureIDs::kColor];
                }
                else if (readBufHandle == (uint32_t)acef::SystemTexture::kInputDepth)
                {
                    irTex = irSystemDatasources[(int)SystemTextureIDs::kDepth];
                }
                else if (readBufHandle == (uint32_t)acef::SystemTexture::kInputHUDless)
                {
                    irTex = irSystemDatasources[(int)SystemTextureIDs::kHUDless];
                }
                else if (readBufHandle == (uint32_t)acef::SystemTexture::kInputHDR)
                {
                    irTex = irSystemDatasources[(int)SystemTextureIDs::kHDR];
                }
                else if (readBufHandle == (uint32_t)acef::SystemTexture::kInputColorBase)
                {
                    irTex = irSystemDatasources[(int)SystemTextureIDs::kColorBase];
                }
                else
                {
                    if (readBufHandle < resourceHeader.binStorage.texturesParametrizedNum)
                    {
                        uint32_t texIdx = readBufHandle;
                        irTex = irTexturesParametrized[texIdx];
                    }
                    else if (readBufHandle < resourceHeader.binStorage.texturesParametrizedNum + resourceHeader.binStorage.texturesIntermediateNum)
                    {
                        uint32_t texIdx = readBufHandle - resourceHeader.binStorage.texturesParametrizedNum;
                        irTex = irTexturesIntermediate[texIdx];
                    }
                    else if (readBufHandle < resourceHeader.binStorage.texturesParametrizedNum + resourceHeader.binStorage.texturesIntermediateNum + resourceHeader.binStorage.texturesFromFileNum)
                    {
                        uint32_t texIdx = readBufHandle - (resourceHeader.binStorage.texturesParametrizedNum + resourceHeader.binStorage.texturesIntermediateNum);
                        irTex = irTexturesFromFile[texIdx];
                    }
                }

                if (readBufName.length() > 0)
                {
                    pass->addDataSource(irTex, readBufName.c_str());
                }
                else
                {
                    pass->addDataSource(irTex, readBufSlot);
                }

                if (irTex->m_firstPassReadIdx == -1 || irTex->m_firstPassReadIdx > (int)passIdx)
                    irTex->m_firstPassReadIdx = (int)passIdx;
                if (irTex->m_lastPassReadIdx < (int)passIdx)
                    irTex->m_lastPassReadIdx = (int)passIdx;

                // TODO avoroshilov ACEF: move to the addDataSource?
                irTex->m_needsSRV = true;
            }

            for (uint32_t dstIdx = 0, dstIdxEnd = acefPass.binStorage.writeBuffersNum; dstIdx < dstIdxEnd; ++dstIdx)
            {
                /*
                // TODO avoroshilov ACEF: remove names for write bufs?

                std::string writeBufName;
                if (acefPass.writeBuffersNameLens[dstIdx] > 0)
                {
                writeBufName = std::string(acefPass.writeBuffersNames+acefPass.writeBuffersNameOffsets[dstIdx], acefPass.writeBuffersNameLens[dstIdx]);
                }
                */
                uint32_t writeBufSlot = acefPass.writeBuffersSlots[dstIdx];

                uint32_t writeBufIdx = acefPass.writeBuffersIndices[dstIdx];
                uint32_t writeBufHandle = resourceHeader.writeBufferTextureHandles[writeBufIdx];

                ir::Texture * irTex = nullptr;
                if (writeBufHandle == (uint32_t)acef::SystemTexture::kInputColor)
                {
                    irTex = irSystemDatasources[(int)SystemTextureIDs::kColor];
                }
                else if (writeBufHandle == (uint32_t)acef::SystemTexture::kInputDepth)
                {
                    irTex = irSystemDatasources[(int)SystemTextureIDs::kDepth];
                }
                else if (writeBufHandle == (uint32_t)acef::SystemTexture::kInputHUDless)
                {
                    irTex = irSystemDatasources[(int)SystemTextureIDs::kHUDless];
                }
                else if (writeBufHandle == (uint32_t)acef::SystemTexture::kInputHDR)
                {
                    irTex = irSystemDatasources[(int)SystemTextureIDs::kHDR];
                }
                else if (writeBufHandle == (uint32_t)acef::SystemTexture::kInputColorBase)
                {
                    irTex = irSystemDatasources[(int)SystemTextureIDs::kColorBase];
                }
                else
                {
                    if (writeBufHandle < resourceHeader.binStorage.texturesParametrizedNum)
                    {
                        uint32_t texIdx = writeBufHandle;
                        irTex = irTexturesParametrized[texIdx];
                    }
                    else if (writeBufHandle < resourceHeader.binStorage.texturesParametrizedNum + resourceHeader.binStorage.texturesIntermediateNum)
                    {
                        uint32_t texIdx = writeBufHandle - resourceHeader.binStorage.texturesParametrizedNum;
                        irTex = irTexturesIntermediate[texIdx];
                    }
                    else if (writeBufHandle < resourceHeader.binStorage.texturesParametrizedNum + resourceHeader.binStorage.texturesIntermediateNum + resourceHeader.binStorage.texturesFromFileNum)
                    {
                        uint32_t texIdx = writeBufHandle - (resourceHeader.binStorage.texturesParametrizedNum + resourceHeader.binStorage.texturesIntermediateNum);
                        irTex = irTexturesFromFile[texIdx];
                    }
                }

                pass->addDataOut(irTex, writeBufSlot);

                if (irTex->m_firstPassWriteIdx == -1 || irTex->m_firstPassWriteIdx > (int)passIdx)
                    irTex->m_firstPassWriteIdx = (int)passIdx;
                if (irTex->m_lastPassWriteIdx < (int)passIdx)
                    irTex->m_lastPassWriteIdx = (int)passIdx;

                // TODO avoroshilov ACEF: move m_needsSRV = true to the addDataSource?
                irTex->m_needsSRV = true;
            }

            // Pass could have dataOut missing
            //  this means that this pass - is output pass
            if (acefPass.binStorage.writeBuffersNum == 0)
            {
                pass->m_dataOutMRTTotal = 1;
                pass->m_mrtChannelFormats.push_back(finalColorInput.format);
            }

            irEffect->addPass(pass);
            irPasses.push_back(pass);
        }

        const wchar_t * effectBasePath = shadersRelativeToTempPath ? effectTempPath : rootDir;
        MultipassConfigParserError finalizeErr = irEffect->finalize(*outTex, effectBasePath, tempsDir, calcHashes);

        if (finalizeErr)
        {
            LOG_ERROR("ACEF->IR colwerter - failed to finalize effect");
            assert(false && "ACEF->IR finalize error");
            return finalizeErr;
        }

        if (!outTex)
        {
            // TODO: this check duplicates the one that is already present above?
            LOG_ERROR("ACEF->IR colwerter - failed to finalize effect [outTex empty]");
            assert(false && "ACEF->IR finalize error [outTex empty]");
            return MultipassConfigParserError(
                MultipassConfigParserErrorEnum::eInternalError,
                "Binary representation load: failed to finalize effect [outTex empty]"
                );
        }

        if (doesEffectRequireColor != nullptr)
        {
            *doesEffectRequireColor = irEffect->isBufferAvailable(ir::Texture::TextureType::kInputColor);
        }

        if (doesEffectRequireDepth != nullptr)
        {
            *doesEffectRequireDepth = irEffect->isBufferAvailable(ir::Texture::TextureType::kInputDepth);
        }

        if (doesEffectRequireHUDless != nullptr)
        {
            *doesEffectRequireHUDless = irEffect->isBufferAvailable(ir::Texture::TextureType::kInputHUDless);
        }

        if (doesEffectRequireHDR != nullptr)
        {
            *doesEffectRequireHDR = irEffect->isBufferAvailable(ir::Texture::TextureType::kInputHDR);
        }

        if (doesEffectRequireColorBase != nullptr)
        {
            *doesEffectRequireColorBase = irEffect->isBufferAvailable(ir::Texture::TextureType::kInputColorBase);
        }

        return MultipassConfigParserError(MultipassConfigParserErrorEnum::eOK);
    }

}
}