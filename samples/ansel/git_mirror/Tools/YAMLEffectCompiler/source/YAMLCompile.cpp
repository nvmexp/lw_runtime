#include <stdio.h>
#include <stdlib.h>
#include <locale>
#include <string>
#define _USE_MATH_DEFINES
#include <math.h>
#include <thread>
#include <assert.h>
#include <Shlwapi.h>
#include <d3d11.h>

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

#include "acef.h"

// Darkroom
#include "darkroom/Bmp.h"
#include "darkroom/Exr.h"
#include "darkroom/Png.h"
#include "darkroom/Jpeg.h"
#include "darkroom/Metadata.h"
#include "darkroom/StringColwersion.h"
#include "darkroom/CmdLineParser.hpp"

#include "D3D11CommandProcessor.h"
#include "D3D11CommandProcessorColwersions.h"
#include "MultipassConfigParserError.h"
#include "ResourceManager.h"
#include "ir/IRCPPHeaders.h"
#include "ir/BinaryColwersion.h"
#include "ir/FileHelpers.h"
#include "EffectParser.h"

#include "StubsJPEG.h"
#include "StubsPNG.h"

#define SAFE_DELETE(x)	if (x) { delete x; (x) = nullptr; }


shadermod::CmdProcEffect	g_effect;			// ACEF: shouldn't be required
shadermod::ResourceManager	g_resourceManager(nullptr, nullptr);
shadermod::ir::Effect		g_irEffect(&g_effect, &g_resourceManager);

shadermod::MultipassConfigParserError initEffectFromYaml(
    const wchar_t * rootDir, const wchar_t * tempsDir, const wchar_t * filename_yaml
    )
{
    wchar_t config_path[FILENAME_MAX];
    size_t pathEndingPos;

    bool onlyFilename = false;
    if (rootDir[0] != L'\0')
    {
        swprintf_s(config_path, FILENAME_MAX, L"%s\\%s", rootDir, filename_yaml);
        onlyFilename = false;
    }
    else
    {
        swprintf_s(config_path, FILENAME_MAX, L"%s", filename_yaml);
        onlyFilename = true;
    }

    shadermod::effectParser::EffectParser parser;
    shadermod::MultipassConfigParserError err = parser.parse(config_path);

    if (err)
        return err;

    if (onlyFilename)
    {
        config_path[0] = 0;
    }
    else
    {
        for (pathEndingPos = wcslen(config_path) - 1; pathEndingPos > 0 && config_path[pathEndingPos] != L'\\' && config_path[pathEndingPos] != L'/'; --pathEndingPos);
        if (pathEndingPos != 0)
            config_path[pathEndingPos] = 0;
    }

    auto& samplers = parser.getSamplers();
    std::vector<shadermod::ir::Sampler *> irSamplers;
    irSamplers.reserve(samplers.size());

    for (auto it = samplers.begin(), end = samplers.end(); it != end; ++it)
    {
        shadermod::ir::AddressType addtypeU = shadermod::effectParser::colwertSamplerAddressTypeToIrSamplerAddressType(it->m_adressU);
        shadermod::ir::AddressType addtypeV = shadermod::effectParser::colwertSamplerAddressTypeToIrSamplerAddressType(it->m_adressV);
        shadermod::ir::FilterType minflt, magflt, mipflt;
        shadermod::effectParser::colwertSamplerFilterTypeToIrFilterType(it->m_filter, minflt, magflt, mipflt);
        shadermod::ir::Sampler * linearSampler = g_irEffect.createSampler(addtypeU, addtypeV, minflt, magflt, mipflt);
        irSamplers.push_back(linearSampler);
    }

    auto& cbuffers = parser.getCBuffers();
    std::vector<shadermod::ir::ConstantBuf *> irCbuffers;
    irCbuffers.reserve(cbuffers.size());

    for (auto it = cbuffers.cbegin(), end = cbuffers.cend(); it != end; ++it)
    {
        shadermod::ir::ConstantBuf * constBuf = g_irEffect.createConstantBuffer();

        for (auto it2 = it->m_variableBindings.cbegin(), end2 = it->m_variableBindings.cend(); it2 != end2; ++it2)
        {
            auto key = it2->first;

            if (it2->second.m_variableIdx == shadermod::effectParser::SystemConstantsType::NUM_ENTRIES)
            {
                if (key.isBoundByName())
                {
                    shadermod::ir::Constant* bufConst = g_irEffect.createConstant(it2->second.m_userConstName.c_str(), key.m_name.c_str());
                    constBuf->addConstant(bufConst);
                }
                else
                {
                    unsigned int dwordOffset = key.m_componentIdx * 4 + (unsigned int)key.m_subcomponentIdx;

                    shadermod::ir::Constant* bufConst = g_irEffect.createConstant(it2->second.m_userConstName.c_str(), dwordOffset);
                    constBuf->addConstant(bufConst);
                }
            }
            else
            {
                shadermod::ir::ConstType ctype = shadermod::effectParser::colwertSystemConstantsTypeToIrConstType(it2->second.m_variableIdx);

                if (key.isBoundByName())
                {
                    shadermod::ir::Constant* bufConst = g_irEffect.createConstant(ctype, key.m_name.c_str());
                    constBuf->addConstant(bufConst);
                }
                else
                {
                    unsigned int dwordOffset = key.m_componentIdx * 4 + (unsigned int)key.m_subcomponentIdx;

                    shadermod::ir::Constant* bufConst = g_irEffect.createConstant(ctype, dwordOffset);
                    constBuf->addConstant(bufConst);
                }
            }
        }

        irCbuffers.push_back(constBuf);
    }

    auto& userconsts = parser.getUserConstants();

    shadermod::ir::UserConstantManager& constMan = g_irEffect.getUserConstantManager();

    for (auto it = userconsts.cbegin(), end = userconsts.cend(); it != end; ++it)
    {
        using shadermod::ir::TypelessVariable;

        auto flexibleTypeToTyplessVariableFunc = [](const shadermod::effectParser::FlexibleDataType& data, const shadermod::effectParser::UserConstDataType& t) ->TypelessVariable
        {
            if (t == shadermod::effectParser::UserConstDataType::kBool)
            {
                return TypelessVariable(data.boolValue);
            }
            else if (t == shadermod::effectParser::UserConstDataType::kFloat)
            {
                return TypelessVariable(data.floatValue);
            }
            else if (t == shadermod::effectParser::UserConstDataType::kInt)
            {
                return TypelessVariable(data.intValue);
            }
            else if (t == shadermod::effectParser::UserConstDataType::kUInt)
            {
                return TypelessVariable(data.uintValue);
            }
            else
            {
                assert(false && "Unknown data type!");
                return TypelessVariable();
            }
        };

        auto makeLocalizationMapFunc = [] (const shadermod::effectParser::UserConstLocalizedString& locstr) -> std::map<unsigned short, std::string>
        {
            std::map<unsigned short, std::string> ret;

            for (int lang = 0; lang < (int)shadermod::effectParser::UserConstOptionalLanguageType::NUM_ENTRIES; ++lang)
            {
                if (locstr.str[lang].length() == 0)
                    continue;

                auto it = ret.insert(std::make_pair(shadermod::effectParser::colwertUserConstOptionalLanguageTypeToLANGID((shadermod::effectParser::UserConstOptionalLanguageType) lang), locstr.str[lang]));
                assert(it.second);
            }

            return ret;
        };

        auto colwertOptionFunc = [flexibleTypeToTyplessVariableFunc, makeLocalizationMapFunc]
        (const shadermod::effectParser::UserConstDesc::ListOption& option, const shadermod::effectParser::UserConstDataType& t) 
            -> shadermod::ir::UserConstant::ListOption
        {
            return shadermod::ir::UserConstant::ListOption(flexibleTypeToTyplessVariableFunc(option.value, t),
                shadermod::ir::UserConstant::StringWithLocalization(option.name, makeLocalizationMapFunc(option.nameLocalized)));
        };

        auto colwertOptionsFunc = [colwertOptionFunc]
        (const std::vector<shadermod::effectParser::UserConstDesc::ListOption>& options, int defaultOption, const shadermod::effectParser::UserConstDataType& t)
            -> shadermod::ir::UserConstant::ListOptions
        {
            std::vector<shadermod::ir::UserConstant::ListOption> opts;
            opts.reserve(options.size());

            for (const auto& o : options)
                opts.push_back(colwertOptionFunc(o, t));

            return shadermod::ir::UserConstant::ListOptions(opts, defaultOption);
        };

        auto colwertUserConstUiControlTypeFunc = [] (shadermod::effectParser::UserConstUiControlType uiControl) -> shadermod::ir::UiControlType
        {
            shadermod::ir::UiControlType ret;

            switch (uiControl)
            {
            case shadermod::effectParser::UserConstUiControlType::kCheckbox:
                ret = shadermod::ir::UiControlType::kCheckbox;
                break;
            case shadermod::effectParser::UserConstUiControlType::kEditbox:
                ret = shadermod::ir::UiControlType::kEditbox;
                break;
            case shadermod::effectParser::UserConstUiControlType::kFlyout:
                ret = shadermod::ir::UiControlType::kFlyout;
                break;
            case shadermod::effectParser::UserConstUiControlType::kSlider:
                ret = shadermod::ir::UiControlType::kSlider;
                break;
            default:
                ret = shadermod::ir::UiControlType::kSlider;
                assert(false && "Unknown ui control type!");
            }

            return ret;
        };

        std::vector<shadermod::ir::UserConstant::StringWithLocalization> valueDisplayName; // TODO: Add YAML support for defining display names for specific values.
        shadermod::ir::UserConstant* uc = constMan.pushBackUserConstant(
            it->name, shadermod::effectParser::colwertUserConstDataTypeToIrUserConstType(it->type), colwertUserConstUiControlTypeFunc(it->uiControlType),
            flexibleTypeToTyplessVariableFunc(it->defaultValue, it->type),
            shadermod::ir::UserConstant::StringWithLocalization(it->uiLabel, makeLocalizationMapFunc(it->uiLabelLocalized)),
            flexibleTypeToTyplessVariableFunc(it->minimumValue, it->type), flexibleTypeToTyplessVariableFunc(it->maximumValue, it->type),
            flexibleTypeToTyplessVariableFunc(it->uiValueStep, it->type),
            shadermod::ir::UserConstant::StringWithLocalization(it->uiHint, makeLocalizationMapFunc(it->uiHintLocalized)),
            shadermod::ir::UserConstant::StringWithLocalization(it->uiValueUnit, makeLocalizationMapFunc(it->uiValueUnitLocalized)),
            it->stickyValue, it->stickyRegion,
            flexibleTypeToTyplessVariableFunc(it->uiValueMin, it->type), flexibleTypeToTyplessVariableFunc(it->uiValueMax, it->type),
            colwertOptionsFunc(it->options, it->defaultOption, it->type), valueDisplayName);

        if (!uc)
        {
            return shadermod::MultipassConfigParserError(shadermod::MultipassConfigParserErrorEnum::eCreateUserConstantFailed,
                std::string("user constant name ") + it->name);
        }
    }

    auto& textureDatasources = parser.getInputTextureDatasources();

    std::vector<shadermod::ir::Texture *> irTextureDatasources;
    irTextureDatasources.reserve(textureDatasources.size());

    for (auto it = textureDatasources.cbegin(), end = textureDatasources.cend(); it != end; ++it)
    {
        std::wstring widePath = darkroom::getWstrFromUtf8(it->m_filename);

        // We do no allow absolute include paths
        if (!PathIsRelativeW(widePath.c_str()))
        {
            //TODO: support wide chars or utf8 in the error reporter
            return shadermod::MultipassConfigParserError(shadermod::MultipassConfigParserErrorEnum::eIllegalShaderPath, it->m_filename);
        }

        // We do not allow '..\' in the include paths
        if (strstr(it->m_filename.c_str(), "..\\") != 0 || strstr(it->m_filename.c_str(), "../") != 0)
        {
            return shadermod::MultipassConfigParserError(shadermod::MultipassConfigParserErrorEnum::eIllegalShaderPath, it->m_filename);
        }

        wchar_t	wc_filename_texture[FILENAME_MAX];
        if (config_path[0] != L'\0')
        {
            swprintf(wc_filename_texture, FILENAME_MAX, L"%s\\%s", config_path, widePath.c_str());
        }
        else
        {
            swprintf(wc_filename_texture, FILENAME_MAX, L"%s", widePath.c_str());
        }

        shadermod::ir::FragmentFormat fmt = it->m_type != shadermod::effectParser::SurfaceType::kUndefined ?
            // ACEF: Replaced (colorSourceTextureFormat, depthSourceTextureFormat) with (FragmentFormat::kNUM_ENTRIES, FragmentFormat::kNUM_ENTRIES)
            shadermod::effectParser::colwertSurfaceTypeToIrFragmentFormat(it->m_type, shadermod::ir::FragmentFormat::kNUM_ENTRIES, shadermod::ir::FragmentFormat::kNUM_ENTRIES) :
            shadermod::ir::FragmentFormat::kRGBA8_uint;

        shadermod::ir::Texture* tex = g_irEffect.createTextureFromFile(
            it->m_width ? it->m_width : shadermod::ir::Texture::SetAsInputFileSize,
            it->m_height ? it->m_height : shadermod::ir::Texture::SetAsInputFileSize,
            
            fmt,

            wc_filename_texture,
            true,
            it->m_excludeHash);

        if (!tex)
            return shadermod::MultipassConfigParserError(shadermod::MultipassConfigParserErrorEnum::eFileNotfound, darkroom::getUtf8FromWstr(config_path) + std::string("\\") + std::string(it->m_filename.c_str()));

        irTextureDatasources.push_back(tex);
    }

    auto& procTexDatasources = parser.getProceduralTextureDatasource();

    std::vector<shadermod::ir::Texture *> irProcTexDatasources;
    irProcTexDatasources.reserve(procTexDatasources.size());

    for (auto it = procTexDatasources.cbegin(), end = procTexDatasources.cend(); it != end; ++it)
    {
        shadermod::ir::Texture* tex = nullptr;

        switch (it->m_procedureType)
        {
        case shadermod::effectParser::ProceduralTextureType::eNoise:
            tex = g_irEffect.createNoiseTexture(it->m_width ? it->m_width : shadermod::ir::Pass::SetAsEffectInputSize,
                it->m_height ? it->m_height : shadermod::ir::Pass::SetAsEffectInputSize,
                it->m_surfaceType != shadermod::effectParser::SurfaceType::kUndefined ?
                // ACEF: Replaced (colorSourceTextureFormat, depthSourceTextureFormat) with (FragmentFormat::kNUM_ENTRIES, FragmentFormat::kNUM_ENTRIES)
                shadermod::effectParser::colwertSurfaceTypeToIrFragmentFormat(it->m_surfaceType, shadermod::ir::FragmentFormat::kNUM_ENTRIES, shadermod::ir::FragmentFormat::kNUM_ENTRIES) :
                shadermod::ir::FragmentFormat::kRGBA8_uint
            );
            break;
        default:
            assert("Unexpected procedural texture type!" && false);
            break;
        }


        irProcTexDatasources.push_back(tex);
    }

    auto& sysdatasource = parser.getSystemDatasource();
    std::vector<shadermod::ir::Texture *> irSystemDatasources;
    irSystemDatasources.resize((int)shadermod::effectParser::SystemChannelType::NUM_ENTRIES, nullptr);

    if (sysdatasource.isChannelReferenced(shadermod::effectParser::SystemChannelType::kPIPE_INPUTS_COLOR))
    {
        irSystemDatasources[(int)shadermod::effectParser::SystemChannelType::kPIPE_INPUTS_COLOR] = g_irEffect.createInputColor();
    }

    if (sysdatasource.isChannelReferenced(shadermod::effectParser::SystemChannelType::kPIPE_INPUTS_DEPTH))
    {
        irSystemDatasources[(int)shadermod::effectParser::SystemChannelType::kPIPE_INPUTS_DEPTH] = g_irEffect.createInputDepth();
    }
    if (sysdatasource.isChannelReferenced(shadermod::effectParser::SystemChannelType::kPIPE_INPUTS_HUDLESS))
    {
        irSystemDatasources[(int)shadermod::effectParser::SystemChannelType::kPIPE_INPUTS_HUDLESS] = g_irEffect.createInputHUDless();
    }
    if (sysdatasource.isChannelReferenced(shadermod::effectParser::SystemChannelType::kPIPE_INPUTS_HDR))
    {
        irSystemDatasources[(int)shadermod::effectParser::SystemChannelType::kPIPE_INPUTS_HDR] = g_irEffect.createInputHDR();
    }
    if (sysdatasource.isChannelReferenced(shadermod::effectParser::SystemChannelType::kPIPE_INPUTS_COLOR_BASE))
    {
        irSystemDatasources[(int)shadermod::effectParser::SystemChannelType::kPIPE_INPUTS_COLOR_BASE] = g_irEffect.createInputColorBase();
    }

    auto& shaderPasses = parser.getShaderPassDatasources();
    std::vector<shadermod::ir::Pass *> irPasses;
    irPasses.reserve(shaderPasses.size());

    for (auto it = shaderPasses.cbegin(), end = shaderPasses.cend(); it != end; ++it)
    {
        std::wstring widePath = darkroom::getWstrFromUtf8(it->m_filename);

        // We do no allow absolute include paths
        if (!PathIsRelativeW(widePath.c_str()))
        {
            return shadermod::MultipassConfigParserError(shadermod::MultipassConfigParserErrorEnum::eIllegalShaderPath, it->m_filename);
        }

        // We do not allow '..\' in the include paths
        if (strstr(it->m_filename.c_str(), "..\\") != 0 || strstr(it->m_filename.c_str(), "../") != 0)
        {
            return shadermod::MultipassConfigParserError(shadermod::MultipassConfigParserErrorEnum::eIllegalShaderPath, it->m_filename);
        }

        wchar_t	wc_filename_shader[FILENAME_MAX];
        if (config_path[0] != L'\0')
        {
            swprintf(wc_filename_shader, FILENAME_MAX, L"%s\\%s", config_path, widePath.c_str());
        }
        else
        {
            swprintf(wc_filename_shader, FILENAME_MAX, L"%s", widePath.c_str());
        }

        std::vector<shadermod::ir::FragmentFormat> fragmentFormats;
        fragmentFormats.reserve(it->m_typesReferenced.size());

        for (auto it2 = it->m_typesReferenced.cbegin(), end2 = it->m_typesReferenced.cend(); it2 != end2; ++it2)
        {
            //Filling unused entries with shadermod::ir::FragmentFormat::kRGBA8_uint. TODO: revise that
            // ACEF: Replaced (colorSourceTextureFormat, depthSourceTextureFormat) with (FragmentFormat::kNUM_ENTRIES, FragmentFormat::kNUM_ENTRIES)
            fragmentFormats.push_back(*it2 != shadermod::effectParser::SurfaceType::kUndefined ? shadermod::effectParser::colwertSurfaceTypeToIrFragmentFormat(*it2, shadermod::ir::FragmentFormat::kNUM_ENTRIES, shadermod::ir::FragmentFormat::kNUM_ENTRIES)
                : shadermod::ir::FragmentFormat::kRGBA8_uint);
        }

        shadermod::ir::PixelShader *ps = g_irEffect.createPixelShader(wc_filename_shader, it->m_shadername.c_str());
        shadermod::ir::Pass * pass = g_irEffect.createPass(nullptr, ps, it->m_width ? it->m_width : shadermod::ir::Pass::SetAsEffectInputSize,
            it->m_height ? it->m_height : shadermod::ir::Pass::SetAsEffectInputSize,
            fragmentFormats.data(), int(fragmentFormats.size()));

        pass->setSizeScale(it->m_scaleWidth, it->m_scaleHeight);

        shadermod::ir::initRasterizerState(&pass->m_rasterizerState);
        shadermod::ir::initDepthStencilState(&pass->m_depthStencilState);
        shadermod::ir::initAlphaBlendState(&pass->m_alphaBlendState);

        for (auto it2 = it->m_samplers.cbegin(), end2 = it->m_samplers.cend(); it2 != end2; ++it2)
        {
            if (it2->first.isBoundByName())
                pass->addSampler(irSamplers[it2->second], it2->first.m_name.c_str());
            else
                pass->addSampler(irSamplers[it2->second], it2->first.m_slotIdx);
        }

        for (auto it2 = it->m_cbuffers.cbegin(), end2 = it->m_cbuffers.cend(); it2 != end2; ++it2)
        {
            if (it2->first.isBoundByName())
                pass->addConstantBufferPS(irCbuffers[it2->second], it2->first.m_name.c_str());
            else
                pass->addConstantBufferPS(irCbuffers[it2->second], it2->first.m_slotIdx);
        }

        irPasses.push_back(pass);
    }

    std::vector<unsigned int> passesNotBlocked;
    passesNotBlocked.reserve(shaderPasses.size());

    struct PassBlocker
    {
        unsigned int	whichOneBlocked;
        unsigned int	byWhichOneBlocked;
    };

    std::vector<PassBlocker> blockers;
    blockers.reserve(shaderPasses.size() * 3); //3 is a wild guess

    struct BlockerPass
    {
        BlockerPass(): firstBlocker(0), numBlockers(0){}
        BlockerPass(unsigned int first, unsigned int num): firstBlocker(first), numBlockers(num){}

        unsigned int firstBlocker;
        unsigned int numBlockers;
    };

    std::vector<BlockerPass> blockerPasses;
    blockerPasses.resize(shaderPasses.size());

    std::vector<unsigned int> numBlockers;
    numBlockers.resize(shaderPasses.size(), 0);

    auto irPassesIt = irPasses.begin();
    unsigned int passCtr = 0;
    for (auto it = shaderPasses.cbegin(), end = shaderPasses.cend(); it != end; ++it, ++irPassesIt, ++passCtr)
    {
        shadermod::ir::Pass* pass = *irPassesIt;

        unsigned int numDependencies = 0;

        for (auto it2 = it->m_textures.cbegin(), end2 = it->m_textures.cend(); it2 != end2; ++it2)
        {
            switch (it2->second.getType())
            {
            case shadermod::effectParser::ShaderPassDatasource::TextureInput::TextureInputType::eShaderPassInput:
            {
                auto shpi = it2->second.getShaderPassInput();

                if (it2->first.isBoundByName())
                    pass->addDataSource(irPasses[shpi->m_shaderPassDatasource], it2->first.m_name.c_str(), (int)shpi->m_channel);
                else
                    pass->addDataSource(irPasses[shpi->m_shaderPassDatasource], it2->first.m_slotIdx, (int)shpi->m_channel);

                blockers.push_back({ passCtr, (unsigned int)shpi->m_shaderPassDatasource });
                ++numDependencies;

                break;
            }
            case shadermod::effectParser::ShaderPassDatasource::TextureInput::TextureInputType::eUserTextureInput:
            {
                auto uti = it2->second.getUserTextureInput();

                if (it2->first.isBoundByName())
                    pass->addDataSource(irTextureDatasources[uti->m_textureDatasource], it2->first.m_name.c_str());
                else
                    pass->addDataSource(irTextureDatasources[uti->m_textureDatasource], it2->first.m_slotIdx);

                break;
            }
            case shadermod::effectParser::ShaderPassDatasource::TextureInput::TextureInputType::eProceduralTextureInput:
            {
                auto pti = it2->second.getProceduralTextureInput();

                if (it2->first.isBoundByName())
                    pass->addDataSource(irProcTexDatasources[pti->m_procTexDatasource], it2->first.m_name.c_str());
                else
                    pass->addDataSource(irProcTexDatasources[pti->m_procTexDatasource], it2->first.m_slotIdx);

                break;
            }
            case shadermod::effectParser::ShaderPassDatasource::TextureInput::TextureInputType::eSystemInput:
            {
                auto si = it2->second.getSystemInput();

                if (it2->first.isBoundByName())
                    pass->addDataSource(irSystemDatasources[(int)si->m_channel], it2->first.m_name.c_str());
                else
                    pass->addDataSource(irSystemDatasources[(int)si->m_channel], it2->first.m_slotIdx);

                break;
            }
            }
        }

        if (!numDependencies)
            passesNotBlocked.push_back(passCtr);

        numBlockers[passCtr] = numDependencies;
    }

    std::sort(blockers.begin(), blockers.end(), [](const PassBlocker& l, const PassBlocker& r)
    {return l.byWhichOneBlocked < r.byWhichOneBlocked ? true : r.byWhichOneBlocked < l.byWhichOneBlocked ? false : l.whichOneBlocked < r.whichOneBlocked; });

    if (blockers.begin() != blockers.end())
    {
        blockerPasses[blockers.begin()->byWhichOneBlocked].firstBlocker = 0;
    }

    for (auto lwrrBlocker = blockers.begin(), endBlocker = blockers.end(); lwrrBlocker != endBlocker;)
    {
        auto nextBlocker = lwrrBlocker;
        ++nextBlocker;

        if (nextBlocker == endBlocker || nextBlocker->byWhichOneBlocked != lwrrBlocker->byWhichOneBlocked)
        {
            size_t nextblockerIdx = (lwrrBlocker - blockers.begin()) + 1;
            blockerPasses[lwrrBlocker->byWhichOneBlocked].numBlockers = (unsigned int)nextblockerIdx - blockerPasses[lwrrBlocker->byWhichOneBlocked].firstBlocker;

            if (nextBlocker != blockers.end())
            {
                blockerPasses[nextBlocker->byWhichOneBlocked].firstBlocker = (unsigned int)nextblockerIdx;
            }
        }

        lwrrBlocker = nextBlocker;
    }

    size_t passesToSchedule = shaderPasses.size();

    for (unsigned int shedPassIdx = 0; shedPassIdx != passesNotBlocked.size(); ++shedPassIdx)
    {
        unsigned int schedPass = passesNotBlocked[shedPassIdx];

        g_irEffect.addPass(irPasses[schedPass]);
        --passesToSchedule;

        for (unsigned int blkrId = blockerPasses[schedPass].firstBlocker, endBlckrId = blockerPasses[schedPass].firstBlocker + blockerPasses[schedPass].numBlockers;
            blkrId != endBlckrId; ++blkrId)
        {
            PassBlocker& passBlocker = blockers[blkrId];

            if (--numBlockers[passBlocker.whichOneBlocked] == 0)
            {
                passesNotBlocked.push_back(passBlocker.whichOneBlocked);
            }
        }
    }

    assert(passesToSchedule == 0);

    shadermod::ir::Texture * outTex = nullptr;
    shadermod::MultipassConfigParserError finalizeErr = g_irEffect.resolveReferences();

    if (finalizeErr)
        return finalizeErr;

    return shadermod::MultipassConfigParserError(shadermod::MultipassConfigParserErrorEnum::eOK);
}

struct CmdLineOptions
{
    std::wstring inFilepath;
    std::wstring outFilepath;
    std::wstring errorFilepath;
} g_cmdLineOptions;

void processCmdLineArgs(int argc, wchar_t * argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        std::wstring lwrArg = argv[i];
        darkroom::tolowerInplace(lwrArg);

        wchar_t * nextArg = (i < argc-1) ? argv[i+1] : nullptr;

        if (lwrArg == L"--in" || lwrArg == L"--input-path")
        {
            if (nextArg)
            {
                g_cmdLineOptions.inFilepath = nextArg;
                wprintf(L"Input file path: %s\n", nextArg);
            }
            else
            {
                wprintf(L"Error in cmd line argument \"--in\" - no file specified\n");
            }
        }
        if (lwrArg == L"--out" || lwrArg == L"--output-path")
        {
            if (nextArg)
            {
                g_cmdLineOptions.outFilepath = nextArg;
                wprintf(L"Output file path: %s\n", nextArg);
            }
            else
            {
                wprintf(L"Error in cmd line argument \"--out\" - no file specified\n");
            }
        }
        if (lwrArg == L"--elog" || lwrArg == L"--error-file")
        {
            if (nextArg)
            {
                g_cmdLineOptions.errorFilepath = nextArg;
                wprintf(L"Error file path: %s\n", nextArg);
            }
            else
            {
                wprintf(L"Error in cmd line argument \"--elog\" - no file specified\n");
            }
        }
        else if (lwrArg == L"-?" || lwrArg == L"-h" || lwrArg == L"-help")
        {
            printf("Command line arguments:\n");
            printf("--in or --input-path\n");
            printf("--out or --output-path\n");
            printf("--elog or --error-file\n");
        }
    }
}

int wmain(int argc, wchar_t ** argv, wchar_t ** elwp)
{
    processCmdLineArgs(argc, argv);

    const std::wstring yamlFilepath = g_cmdLineOptions.inFilepath;
    const std::wstring tempFilepath = g_cmdLineOptions.outFilepath;
    const std::wstring errorFilepath = g_cmdLineOptions.errorFilepath;

    shadermod::MultipassConfigParserError err(shadermod::MultipassConfigParserErrorEnum::eOK);

    auto splitFilepathToPathFilename = [](const std::string & filepath, std::string * path, std::string * filename)
    {
        const size_t slashPos = filepath.rfind("\\");
        if (slashPos != std::string::npos)
        {
            *path = std::string(filepath.c_str(), slashPos + 1);
            *filename = std::string(filepath.c_str() + slashPos + 1);
        }
        else
        {
            *path = "";
            *filename = filepath;
        }

    };

    auto splitFilepathToPathFilenameW = [](const std::wstring & filepath, std::wstring * path, std::wstring * filename)
    {
        const size_t slashPos = filepath.rfind(L"\\");
        if (slashPos != std::wstring::npos)
        {
            *path = std::wstring(filepath.c_str(), slashPos + 1);
            *filename = std::wstring(filepath.c_str() + slashPos + 1);
        }
        else
        {
            *path = L"";
            *filename = filepath;
        }

    };

    std::wstring yamlPathW;
    std::wstring yamlFilenameW;
    splitFilepathToPathFilenameW(yamlFilepath, &yamlPathW, &yamlFilenameW);

    std::wstring tempPathW;
    std::wstring tempFilenameW;
    splitFilepathToPathFilenameW(tempFilepath, &tempPathW, &tempFilenameW);

    //std::wstring yamlPathW = darkroom::getWstrFromUtf8(yamlPath);
    //std::wstring yamlFilenameW = darkroom::getWstrFromUtf8(yamlFilename);
    //std::wstring tempPathW = darkroom::getWstrFromUtf8(tempPath);

    if (!tempPathW.empty() && tempPathW[tempPathW.length()-1] != L'\\')
    {
        tempPathW += L"\\";
    }


    wchar_t lwrDir[1024];
    GetLwrrentDirectoryW(1024, lwrDir);

    err = initEffectFromYaml(yamlPathW.c_str(), tempPathW.c_str(), yamlFilenameW.c_str());
    if (err)
    {
        std::wstring errorPathW;
        std::wstring errorFilenameW;
        splitFilepathToPathFilenameW(errorFilepath, &errorPathW, &errorFilenameW);

        shadermod::ir::filehelpers::createDirectoryRelwrsively(errorPathW.c_str());

        FILE * fp_err = nullptr;
        _wfopen_s(&fp_err, errorFilepath.c_str(), L"w");

        fprintf(fp_err, "%s", err.getFullErrorMessage().c_str());

        fclose(fp_err);

        return -1;
    }

    acef::EffectStorage acefEffectStorage;

    std::wstring binaryPathW = tempPathW;
    std::wstring acefFilenameW = tempFilenameW;

    shadermod::ir::filehelpers::createDirectoryRelwrsively(binaryPathW.c_str());

    uint64_t timestamp = 0;
    shadermod::ir::filehelpers::getFileTime((yamlPathW + yamlFilenameW).c_str(), timestamp);

    shadermod::ir::colwertIRToBinary(
        // in
        yamlPathW.c_str(), binaryPathW.c_str(), g_irEffect,
        // out
        &acefEffectStorage,
        timestamp
        );
    // Callwlating offsets
    acef::calcByteOffsets(&acefEffectStorage);

    // SAVING
    acef::save((binaryPathW+acefFilenameW).c_str(), acefEffectStorage);


    printf("\nFINISHED!\n");
    return 0;
}

