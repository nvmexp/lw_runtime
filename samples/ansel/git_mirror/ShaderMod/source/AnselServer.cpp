// Windows / C includes
#include <tchar.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#define NOMINMAX
#include <windows.h>
#include <shlobj.h>
#include <Shlwapi.h>
#include <assert.h>
#include <d3d11.h>
#include <string.h>
#define _USE_MATH_DEFINES // to get M_PI
#include <math.h>

// STL includes
#include <map>
#include <set>
#include <list>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <ctime>

#include <lwapi.h>
#include <LwApiDriverSettings.h>
#include <dxgi1_2.h>
#include "drs/LwDrsWrapper.h"
#include "drs/LwDrsDefines.h"

// AnselSDK includes
#include "Log.h"
#include "Profiling.h"
#include "AnselSDK.h"
#include "anselutils/Utils.h"

#define ANSEL_DLL_EXPORTS
#include "ErrorReporting.h"
#include "Ansel.h"
#include "AnselServer.h"
#include "Allowlisting.h"
#include "CommonStructs.h"
#include "CommonTools.h"
#include "AnselSDKState.h"
#include "Config.h"
#include "UIBase.h"
#include "UI.h"
#include "AnselInput.h"
#include "ir/FileHelpers.h"
#include "Ansel.h"
#define DO_DEBUG_PRINTS 0
#include "Utils.h"
#include "ui/classes.h"
#include "ui/elements.h"
#include "darkroom/Bmp.h"
#include "darkroom/Png.h"
#include "darkroom/Jpeg.h"
#include "darkroom/Exr.h"
#include "darkroom/Jxr.h"
#include "darkroom/ImageLoader.h"
#include "darkroom/ImageOperations.h"
#include "darkroom/StringColwersion.h"
#include "AnselTelemetryGeneratedCode.h"
#include "AnselVersionInfo.h"
#include "ir/UserConstantManager.h"
#include "LocalizedFilterNames.h"
#include "toml.h"
#include "HardcodedFX.h"

#include "i18n/LocalizedStringHelper.h"
#include "i18n/text.en-US.h"

// IPC
#include "ipc/AnselIPC.h"
#include "ipc/UIIPC.h"

#ifdef ENABLE_STYLETRANSFER
#include "lwselwreloadlibrary/lwSelwreLoadLibrary.h"
#endif

// This define lets the exception flow through Ansel without being caught and processed
// so that they can be caught by a programmer during debug
#ifdef _DEBUG
#   define DBG_EXCEPTIONS_PASSTHROUGH   1
#else
#   define DBG_EXCEPTIONS_PASSTHROUGH   0
#endif

#if IPC_ENABLED == 1

// Link import libraries here (instead of project properties)
#pragma comment(lib, "MessageBus.lib")
#pragma comment(lib, "delayimp")

#endif

#if defined(_WIN64) || defined(_WIN32)
#   ifdef _WIN64
#       define MODULE_NAME_A "LwCamera64.dll"
#       define MODULE_NAME_W L"LwCamera64.dll"
#   else
#       define MODULE_NAME_A "LwCamera32.dll"
#       define MODULE_NAME_W L"LwCamera32.dll"
#   endif
#else
#   error "Unknown machine arch"
#endif

//#define SAFE_RELEASE(x) if (x) x->Release(), x = nullptr;
#define SAFE_DELETE(x) if (x) { delete x; x = nullptr; }

#define HIGH_QUALITY_CONTROL_ID 189237

// delay loading hook is needed to resolve correct style transfer library
namespace
{
    template<typename T>
    T findExtension(const T& filename)
    {
        const T::value_type* extPtr = PathFindExtension(filename.c_str());
        if (*extPtr != '\0')
            return T(extPtr);
        else
            return T();
    }

    void getUserTempDirectory(std::wstring& outString)
    {
        const DWORD bufferLen = 1024;
        std::vector<wchar_t> bufferW;
        bufferW.resize(bufferLen);
        GetTempPathW(bufferLen, &bufferW[0]);
        outString = &bufferW[0];
    }

    void sessionStartFunc(void* userData)
    {
        AnselServer* server = (AnselServer*)userData;
        server->m_bNextFrameForceEnableAnsel = true;
        LOG_DEBUG("Ansel start requested by the game");
    }

    void sessionStopFunc(void* userData)
    {
        AnselServer* server = (AnselServer*)userData;
        server->m_bNextFrameForceDisableAnsel = true;
        LOG_DEBUG("Ansel stop requested by the game");
    }

    void bufferFinishedFunc(void * userData, ansel::BufferType bufferType, uint64_t threadId)
    {
        AnselServer* server = (AnselServer*)userData;
        server->bufferFinished(bufferType, threadId);
    }

#ifdef ENABLE_STYLETRANSFER
    // this is supposed to be in AppData\Local
    std::wstring modelRoot;
    std::wstring modelRootRelative(L"\\LWPU Corporation\\Ansel\\Models\\");
    const std::wstring binariesRoot(L"\\LWPU Corporation\\Ansel\\Bin");

    const std::map<std::wstring, std::pair<std::wstring, std::wstring>> networks =
    {
        { L"8", { L"encoder_vgg_8.t7", L"decoder_vgg_8.t7" } },
        { L"64", { L"encoder_vgg_64.t7", L"decoder_vgg_64.t7" } }
    };
#endif
}

static OSVERSIONINFOEXW osVersionInfo = { 0 };
static bool osLessThanOrEqualToWin7 = false;
bool OSWin7OrLess()
{
    // Check forQuery OS version info (if not done yet)
    if (!osVersionInfo.dwOSVersionInfoSize)
    {
        DWORDLONG const dwlConditionMask = VerSetConditionMask(
                                            VerSetConditionMask(
                                                VerSetConditionMask(0, VER_MAJORVERSION, VER_GREATER_EQUAL),
                                                VER_MINORVERSION, VER_GREATER_EQUAL),
                                            VER_SERVICEPACKMAJOR, VER_GREATER_EQUAL);

        osVersionInfo.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEXW);
        osVersionInfo.dwMajorVersion = HIBYTE(_WIN32_WINNT_WIN8);
        osVersionInfo.dwMinorVersion = LOBYTE(_WIN32_WINNT_WIN8);
        osVersionInfo.wServicePackMajor = 0;

        osLessThanOrEqualToWin7 = VerifyVersionInfoW(&osVersionInfo, VER_MAJORVERSION | VER_MINORVERSION | VER_SERVICEPACKMAJOR, dwlConditionMask) ? false : true;
        LOG_DEBUG("OS is %s.", osLessThanOrEqualToWin7 ? "less than or equal to Win7" : "greater than Win7");
    }
    return osLessThanOrEqualToWin7;
}

AnselServer::AnselServer(HINSTANCE hDLLInstance)
    :
    MultiPassProcessor(false),
    m_hDLLInstance(hDLLInstance),
    m_bInitialized(false),
    m_maxHighResResolution(63),
    m_maxSphericalResolution(8),
    m_eyeSeparation(6.3f),
    m_cameraSpeedMultiplier(4.0f),
    m_removeBlackTint(false),
    m_keepIntermediateShots(false),
    m_renderDebugInfo(false),
    m_hClient(NULL),
    m_pClientFunctionTable(NULL),
    m_enableDepthExtractionChecks(true),
    m_depthBufferUsed(false),
    m_bRenderDepthAsRGB(false),
    m_bRunShaderMod(false),
#if DBG_HARDCODED_EFFECT_BW_ENABLED
    m_bEnableBlackAndWhite(false),
    m_bNextFrameEnableBlackAndWhite(m_bEnableBlackAndWhite),
#endif
    m_bNextFrameRenderDepthAsRGB(m_bRenderDepthAsRGB),
    m_bNextFrameRunShaderMod(m_bRunShaderMod),
    m_bNextFrameNextEffect(false),
    m_bNextFramePrevEffect(false),
    m_bNextFrameRebuildYAML(false),
    m_bNextFrameRefreshEffectStack(false),
    m_sessionStartTime(0.0),
    m_captureStartTime(0.0),
    m_psUtil(),
    m_bufTestingOptionsFilter(m_effectsInfo, m_renderBufferColwerter, m_bufDB.Depth(), m_bufDB.Hudless())
{
    m_effectsInfo.m_selectedEffect = 0;
}

AnselServer::~AnselServer()
{}

const std::map<std::string, AnselServer::PrepackagedEffects> AnselServer::s_effectStringToEnum =
{
    // Yaml Filters
    {"Adjustments.yaml", PrepackagedEffects::kAdjustments_yaml},
    {"BlacknWhite.yaml", PrepackagedEffects::kBlacknWhite_yaml},
    {"Color.yaml", PrepackagedEffects::kColor_yaml},
    {"Colorblind.yaml", PrepackagedEffects::kColorblind_yaml},
    {"DOF.yaml", PrepackagedEffects::kDOF_yaml},
    {"Details.yaml", PrepackagedEffects::kDetails_yaml},
    {"Letterbox.yaml", PrepackagedEffects::kLetterbox_yaml},
    {"NightMode.yaml", PrepackagedEffects::kNightMode_yaml},
    {"OldFilm.yaml", PrepackagedEffects::kOldFilm_yaml},
    {"Painterly.yaml", PrepackagedEffects::kPainterly_yaml},
    {"RemoveHud.yaml", PrepackagedEffects::kRemoveHud_yaml},
    {"Sharpen.yaml", PrepackagedEffects::kSharpen_yaml},
    {"SpecialFX.yaml", PrepackagedEffects::kSpecialFX_yaml},
    {"Splitscreen.yaml", PrepackagedEffects::kSplitscreen_yaml},
    {"TiltShift.yaml", PrepackagedEffects::kTiltShift_yaml},
    {"Vignette.yaml", PrepackagedEffects::kVignette_yaml},
    {"Watercolor.yaml", PrepackagedEffects::kWatercolor_yaml},

    // ReShade Filters
    {"3DFX.fx", PrepackagedEffects::k3DFX_fx},
    {"ASCII.fx", PrepackagedEffects::kASCII_fx},
    {"AdaptiveSharpen.fx", PrepackagedEffects::kAdaptiveSharpen_fx},
    {"AmbientLight.fx", PrepackagedEffects::kAmbientLight_fx},
    {"Bloom.fx", PrepackagedEffects::kBloom_fx},
    {"Border.fx", PrepackagedEffects::kBorder_fx},
    {"CRT.fx", PrepackagedEffects::kCRT_fx},
    {"Cartoon.fx", PrepackagedEffects::kCartoon_fx},
    {"ChromaticAberration.fx", PrepackagedEffects::kChromaticAberration_fx},
    {"Clarity.fx", PrepackagedEffects::kClarity_fx},
    {"Colourfulness.fx", PrepackagedEffects::kColourfulness_fx},
    {"Lwrves.fx", PrepackagedEffects::kLwrves_fx},
    {"Daltonize.fx", PrepackagedEffects::kDaltonize_fx},
    {"Deband.fx", PrepackagedEffects::kDeband_fx},
    {"Denoise.fx", PrepackagedEffects::kDenoise_fx},
    {"DenoiseKNN.fx", PrepackagedEffects::kDenoiseKNN_fx},
    {"DenoiseNLM.fx", PrepackagedEffects::kDenoiseNLM_fx},
    {"FXAA.fx", PrepackagedEffects::kFXAA_fx},
    {"FakeMotionBlur.fx", PrepackagedEffects::kFakeMotionBlur_fx},
    {"GaussianBlur.fx", PrepackagedEffects::kGaussianBlur_fx},
    {"Glitch.fx", PrepackagedEffects::kGlitch_fx},
    {"HQ4X.fx", PrepackagedEffects::kHQ4X_fx},
    {"Monochrome.fx", PrepackagedEffects::kMonochrome_fx},
    {"NightVision.fx", PrepackagedEffects::kNightVision_fx},
    {"PPFX_Godrays.fx", PrepackagedEffects::kPPFX_Godrays_fx},
    {"Prism.fx", PrepackagedEffects::kPrism_fx},
    {"SMAA.fx", PrepackagedEffects::kSMAA_fx},
    {"Test_Reshade_Hashed.fx", PrepackagedEffects::kTest_Reshade_Hashed_fx},
    {"TiltShift.fx", PrepackagedEffects::kTiltShift_fx},
    {"TriDither.fx", PrepackagedEffects::kTriDither_fx},
    {"Vignette.fx", PrepackagedEffects::kVignette_fx},

    // Special shader, used in the system tests
    {"Testing.yaml", PrepackagedEffects::kTesting_yaml}
};

void AnselServer::effectFilterIdAndName(const std::wstring & effectPath, const std::wstring & effectFilename, LANGID langId, std::wstring & filterId, std::wstring & filterDisplayName) const
{
    filterId = effectPath + effectFilename;

    bool found = m_localizedEffectNamesParser.getFilterNameLocalized(effectPath, effectFilename, filterDisplayName, langId);
    if (!found)
    {
        filterDisplayName = effectFilename.substr(0, effectFilename.find_last_of(L'.'));
    }
}

void AnselServer::getAllFileNamesWithinFolders(const std::vector<std::wstring> & exts, std::vector<std::wstring> * outFilenames, std::vector<std::wstring> * outFolders) const
{
    if (!outFilenames)
        return;

    std::unordered_set<std::wstring> uniqueFilenames;

    for (auto& folder : m_effectFoldersAdditional)
    {
        std::wstring searchPath;

        searchPath = folder + L"*.*";

        WIN32_FIND_DATA fd;
        HANDLE hFind = FindFirstFile(searchPath.c_str(), &fd);

        if (hFind != ILWALID_HANDLE_VALUE)
        {
            do
            {
                // read all (real) files in current folder
                // , delete '!' read other 2 default folder . and ..
                if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
                {
                    std::wstring filename(fd.cFileName);
                    const size_t filename_length = filename.length();

                    // If the exts list is empty, return the full file list
                    bool validExtension = (exts.size() == 0);
                    for (size_t extIdx = 0, extIdxEnd = exts.size(); extIdx < extIdxEnd; ++extIdx)
                    {
                        const size_t lastof = filename.rfind(exts[extIdx]);
                        const size_t exts_length = exts[extIdx].length();
                        if (lastof == filename_length - exts_length)
                        {
                            validExtension = true;
                            break;
                        }
                    }

                    if (validExtension)
                    {
                        // The same filter may exist in multiple locations (for example: program files folder as well as application folder as well as the driver store folder in case of DCH)
                        // In such cases, we want to override the filters and not duplicate them.
                        // The priority order for overriding filters is:
                        // 1. Application folder\LwCamera
                        // 2. Custom folder: Program Files\LWPU Corporation\Ansel\Custom
                        // 3. Any other folders specified through regkey
                        // 4. DriverStore\LwCamera (only for DCH)
                        // 5. Program Files\LWPU Corporation\Ansel
                        // Note: By this point in the LwCamera flow, the filters should already be appearing in that order.
                        // We simply need to ensure that there are no duplicates that show up at this point.
                        if (uniqueFilenames.count(fd.cFileName) == 0)
                        {
                            outFilenames->push_back(fd.cFileName);
                            if (outFolders)
                                outFolders->push_back(folder);
                            uniqueFilenames.insert(fd.cFileName);
                        }
                    }
                }
            } while (::FindNextFile(hFind, &fd));

            FindClose(hFind);
        }
    }
}

std::vector<std::wstring> getAllFileNamesWithinFolder(const std::wstring& folder, const std::vector<std::wstring> & exts)
{
    std::vector<std::wstring> names;
    std::wstring searchPath;

    searchPath = folder + L"*.*";

    WIN32_FIND_DATA fd;
    HANDLE hFind = FindFirstFile(searchPath.c_str(), &fd);

    if (hFind != ILWALID_HANDLE_VALUE)
    {
        do
        {
            // read all (real) files in current folder
            // , delete '!' read other 2 default folder . and ..
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
            {
                std::wstring filename(fd.cFileName);
                const size_t filename_length = filename.length();

                // If the exts list is empty, return the full file list
                bool validExtension = (exts.size() == 0);
                for (size_t extIdx = 0, extIdxEnd = exts.size(); extIdx < extIdxEnd; ++extIdx)
                {
                    const size_t lastof = filename.rfind(exts[extIdx]);
                    const size_t exts_length = exts[extIdx].length();
                    if (lastof == filename_length - exts_length)
                    {
                        validExtension = true;
                        break;
                    }
                }

                if (validExtension)
                    names.push_back(fd.cFileName);
            }
        } while (::FindNextFile(hFind, &fd));

        FindClose(hFind);
    }

    return names;
}

void AnselServer::populateEffectsList(uint32_t restrictedSetID)
{
    // First, we need to build a list of what files current effects in the stack actually refer to
    m_storageEffectFilenames.resize(m_effectsInfo.m_effectSelected.size());
    m_storageEffectRootFolders.resize(m_effectsInfo.m_effectSelected.size());

    for (int effectIdx = 0, effectIdxEnd = (int)m_effectsInfo.m_effectSelected.size(); effectIdx < effectIdxEnd; ++effectIdx)
    {
        int selectedEffect = m_effectsInfo.m_effectSelected[effectIdx];
        if (selectedEffect > 0)
        {
            m_storageEffectFilenames[effectIdx] = m_effectsInfo.m_effectFilesList[selectedEffect - 1];
            m_storageEffectRootFolders[effectIdx] = m_effectsInfo.m_effectRootFoldersList[selectedEffect - 1];
        }
        else
        {
            m_storageEffectFilenames[effectIdx].clear();
            m_storageEffectRootFolders[effectIdx].clear();
        }
    }

    // Then, repopulate the files list
    m_effectsInfo.m_effectFilesList.resize(0);
    m_effectsInfo.m_effectRootFoldersList.resize(0);

    getAllFileNamesWithinFolders(m_fxExtensions, &m_effectsInfo.m_effectFilesList, &m_effectsInfo.m_effectRootFoldersList);
    AddHardcodedEffectsAndAlphabetize(m_effectsInfo.m_effectFilesList, m_effectsInfo.m_effectRootFoldersList, m_effectInstallationFolderPath);

    // Filter out files that need to be filtered
    if (restrictedSetID != (uint32_t)ModdingRestrictedSetID::kEmpty)
    {
        for (int effectFileIdx = 0; effectFileIdx < (int)m_effectsInfo.m_effectFilesList.size(); ++effectFileIdx)
        {
#if ANSEL_SIDE_PRESETS
            std::wstring ini_suffix = L".ini";
            if (std::equal(ini_suffix.rbegin(), ini_suffix.rend(), m_effectsInfo.m_effectFilesList[effectFileIdx].rbegin()))
            {
                continue;
            }
#endif
            Hash::Data effNameKey;
            sha256_ctx effNameHashContext;
            sha256_init(&effNameHashContext);
            sha256_update(&effNameHashContext, (uint8_t *)m_effectsInfo.m_effectFilesList[effectFileIdx].c_str(), (uint32_t)(m_effectsInfo.m_effectFilesList[effectFileIdx].length() * sizeof(wchar_t)));
            sha256_final(&effNameHashContext, effNameKey.data());

            auto effHashIdx = m_moddingRestrictedSets[restrictedSetID].find(effNameKey);
            if (effHashIdx == m_moddingRestrictedSets[restrictedSetID].end()
                || m_denylist.FilterDenylisted(m_effectsInfo.m_effectFilesList[effectFileIdx]))
            {
                m_effectsInfo.m_effectFilesList.erase(m_effectsInfo.m_effectFilesList.begin() + effectFileIdx);
                m_effectsInfo.m_effectRootFoldersList.erase(m_effectsInfo.m_effectRootFoldersList.begin() + effectFileIdx);
                --effectFileIdx;
            }
        }
    }


    // Recallwlate the selected file index based on previously stored helper data
    for (int effectIdx = 0, effectIdxEnd = (int)m_effectsInfo.m_effectSelected.size(); effectIdx < effectIdxEnd; ++effectIdx)
    {
        // Effect was set to "None", no need to re-match
        if (m_storageEffectFilenames[effectIdx].empty())
        {
            continue;
        }

        bool matchFound = false;
        for (size_t effectFileIdx = 0, effectFileIdxEnd = m_effectsInfo.m_effectFilesList.size(); effectFileIdx < effectFileIdxEnd; ++effectFileIdx)
        {
            if ((m_storageEffectFilenames[effectIdx] == m_effectsInfo.m_effectFilesList[effectFileIdx]) &&
                (m_storageEffectRootFolders[effectIdx] == m_effectsInfo.m_effectRootFoldersList[effectFileIdx]))
            {
                m_effectsInfo.m_effectSelected[effectIdx] = (int)effectFileIdx + 1;
                matchFound = true;
                break;
            }
        }
        if (!matchFound)
        {
            // Ilwalidate effect
            m_effectsInfo.m_effectSelected[effectIdx] = 0;
            m_effectsInfo.m_effectRebuildRequired[effectIdx] = true;
        }
    }

    // Rebuild localization configurations
    m_localizedEffectNamesParser.reset();
    for (auto searchPath : m_effectFoldersAdditional)
    {
        m_localizedEffectNamesParser.parseSingleFile(searchPath.c_str(), FILTER_NAMES_CONFIG_FILENAME);
    }

    // Build FilterIDs list (root folder + filename)
    m_effectsInfo.m_effectFilterIds.resize(0);
    for (size_t i = 0, iend = m_effectsInfo.m_effectFilesList.size(); i < iend; ++i)
    {
        m_effectsInfo.m_effectFilterIds.push_back(m_effectsInfo.m_effectRootFoldersList[i] + m_effectsInfo.m_effectFilesList[i]);
    }
}

#ifdef ENABLE_STYLETRANSFER
void AnselServer::populateStylesList()
{
    const std::vector<std::wstring> extensions = { L"jpg", L"png" };

    const auto defaultStylePath = m_installationFolderPath + L"Styles\\";

    // it's possible to add more paths containing styles here
    const auto paths = { defaultStylePath, m_userStylesFolderPath };

    m_stylesFilesList.resize(0);
    m_stylesFilesListTrimmed.resize(0);
    m_stylesFilesPaths.resize(0);

    bool atLeastOnePathIsNotEmpty = false;
    for (const auto& path : paths)
    {
        const auto filesList = getAllFileNamesWithinFolder(path, extensions);
        if (!filesList.empty())
        {
            atLeastOnePathIsNotEmpty = true;
            m_stylesFilesList.insert(m_stylesFilesList.end(), filesList.begin(), filesList.end());
            for (const auto& fileName : filesList)
            {
                m_stylesFilesPaths.push_back(path + fileName);
                m_stylesFilesListTrimmed.push_back(fileName.substr(0, fileName.find_last_of(L'.')));
            }
        }
    }
}

void AnselServer::populateStyleNetworksList()
{
    m_styleNetworksIds.clear();
    m_styleNetworksLabels.clear();

    m_styleNetworksIds.push_back(L"64");
    m_styleNetworksLabels.push_back(i18n::getLocalizedString(IDS_STYLE_NETHIGH, m_UI->getLangId()));
    m_styleNetworksIds.push_back(L"8");
    m_styleNetworksLabels.push_back(i18n::getLocalizedString(IDS_STYLE_NETLOW, m_UI->getLangId()));
}

int AnselServer::getStyleIndexByName(const std::wstring & styleName) const
{
    if (styleName == shadermod::Tools::wstrNone)
        return 0;

    const auto it = std::find(m_stylesFilesListTrimmed.begin(), m_stylesFilesListTrimmed.end(), styleName);
    if (it == m_stylesFilesListTrimmed.end())
        return -1;

    return (int)(it - m_stylesFilesListTrimmed.begin() + 1);
}
#endif

int AnselServer::getFilterIndexById(const std::wstring & filterId) const
{
    if (filterId == shadermod::Tools::wstrNone)
        return 0;

    const auto it = std::find(m_effectsInfo.m_effectFilterIds.begin(), m_effectsInfo.m_effectFilterIds.end(), filterId);
    if (it == m_effectsInfo.m_effectFilterIds.end())
        return -1;

    return (int)(it - m_effectsInfo.m_effectFilterIds.begin() + 1);
}

void SetLwrrentDefaultMinMaxDisplayNameAndStickyValues(AnselUIBase::EffectPropertiesDescription::EffectAttributes& attrib, const shadermod::ir::UserConstant * lwrUC, LANGID langID)
{
    shadermod::ir::UserConstDataType ucdt = lwrUC->getType();
    if (ucdt != shadermod::ir::UserConstDataType::kFloat
        && ucdt != shadermod::ir::UserConstDataType::kInt
        && ucdt != shadermod::ir::UserConstDataType::kUInt
        && ucdt != shadermod::ir::UserConstDataType::kBool)
    {
        LOG_ERROR("Unsupported user constant type: %d", (int)ucdt);
        return;
    }
    attrib.lwrrentValue = lwrUC->getValue();
    attrib.defaultValue = lwrUC->getDefaultValue();
    attrib.milwalue = lwrUC->getMinimumValue();
    attrib.maxValue = lwrUC->getMaximumValue();

    for (UINT i = 0; i < MAX_GROUPED_VARIABLE_DIMENSION; i++)
    {
        attrib.valueDisplayName[i] = darkroom::getWstrFromUtf8(lwrUC->getValueDisplayName(i).getLocalized(langID).c_str());
    }

    attrib.stickyValue = lwrUC->getStickyValue();
    attrib.stickyRegion = lwrUC->getStickyRegion();
}

void SetStepSizeMinAndMaxForUI(AnselUIBase::EffectPropertiesDescription::EffectAttributes& attrib, const shadermod::ir::UserConstant * lwrUC)
{
    shadermod::ir::UserConstDataType ucdt = lwrUC->getType();
    if (ucdt != shadermod::ir::UserConstDataType::kFloat
        && ucdt != shadermod::ir::UserConstDataType::kInt
        && ucdt != shadermod::ir::UserConstDataType::kUInt)
    {
        LOG_ERROR("Invalid user constant type for UI values: %d", (int)ucdt);
        return;
    }

    attrib.stepSizeUI = lwrUC->getUiValueStep();
    attrib.uiMilwalue = lwrUC->getUiValueMin();
    attrib.uiMaxValue = lwrUC->getUiValueMax();
}

bool AnselServer::getEffectDescription(const shadermod::MultiPassEffect* eff, LANGID langID, AnselUIBase::EffectPropertiesDescription * effectDesc) const
{
    if (!eff)
    {
        effectDesc->filterId = shadermod::Tools::wstrNone;

        return false;
    }

    WORD langID_English = MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US);

    shadermod::ir::UserConstantManager::ConstRange ucrange = eff->getUserConstantManager().getPointersToAllUserConstants();

    const float editAsSliderHackMin = -100.0f;
    const float editAsSliderHackMax =  100.0f;

    for (; ucrange.begin < ucrange.end; ++ucrange.begin)
    {
        const shadermod::ir::UserConstant * lwrUC = *(ucrange.begin);

        AnselUIBase::EffectPropertiesDescription::EffectAttributes attrib;

        attrib.userConstant = lwrUC;

        attrib.displayNameEnglish = darkroom::getWstrFromUtf8Fallback(lwrUC->getUiLabelLocalized(langID_English).c_str(), lwrUC->getUiLabel().c_str());
        attrib.displayName = darkroom::getWstrFromUtf8Fallback(lwrUC->getUiLabelLocalized(langID).c_str(), lwrUC->getUiLabel().c_str());
        attrib.uiMeasurementUnit = darkroom::getWstrFromUtf8Fallback(lwrUC->getUiValueUnitLocalized(langID).c_str(), lwrUC->getUiValueUnit().c_str());

        attrib.controlId = lwrUC->getUid();

        shadermod::ir::UserConstDataType ucdt = lwrUC->getType();
        shadermod::ir::UiControlType ucct = lwrUC->getControlType();

        if (ucdt == shadermod::ir::UserConstDataType::kFloat)
        {
            attrib.dataType = AnselUIBase::DataType::kFloat;
            SetLwrrentDefaultMinMaxDisplayNameAndStickyValues(attrib, lwrUC, langID);
            SetStepSizeMinAndMaxForUI(attrib, lwrUC);
        }
        else if (ucdt == shadermod::ir::UserConstDataType::kBool)
        {
            attrib.dataType = AnselUIBase::DataType::kBool;
            SetLwrrentDefaultMinMaxDisplayNameAndStickyValues(attrib, lwrUC, langID);
        }
        else if (ucdt == shadermod::ir::UserConstDataType::kInt ||
            ucdt == shadermod::ir::UserConstDataType::kUInt)
        {
            attrib.dataType = AnselUIBase::DataType::kInt;
            SetLwrrentDefaultMinMaxDisplayNameAndStickyValues(attrib, lwrUC, langID);
            SetStepSizeMinAndMaxForUI(attrib, lwrUC);
        }
        else if (ucct != shadermod::ir::UiControlType::kFlyout && ucct != shadermod::ir::UiControlType::kRadioButton)
        {
            LOG_ERROR("Unsupported user constant type %d", (int) ucdt);
            continue;
        }

        if (ucct == shadermod::ir::UiControlType::kSlider)
        {
            attrib.controlType = AnselUIBase::ControlType::kSlider;

            if (attrib.dataType != AnselUIBase::DataType::kFloat
                && attrib.dataType != AnselUIBase::DataType::kInt
                && attrib.dataType != AnselUIBase::DataType::kBool)
            {
                assert(false);
            }
        }
        else if (ucct == shadermod::ir::UiControlType::kEditbox)
        {
            attrib.controlType = AnselUIBase::ControlType::kEditbox;
        }
        else if (ucct == shadermod::ir::UiControlType::kCheckbox)
        {
            attrib.controlType = AnselUIBase::ControlType::kCheckbox;
        }
        else if (ucct == shadermod::ir::UiControlType::kColorPicker)
        {
            attrib.controlType = AnselUIBase::ControlType::kColorPicker;
        }
        else if (ucct == shadermod::ir::UiControlType::kFlyout)
        {
            attrib.controlType = AnselUIBase::ControlType::kFlyout;
        }
        else if (ucct == shadermod::ir::UiControlType::kRadioButton)
        {
            attrib.controlType = AnselUIBase::ControlType::kRadioButton;
        }
        else
        {
            LOG_ERROR("Unsupported control type %d", (int) ucct);
            continue;
        }

        effectDesc->attributes.push_back(attrib);
    }

    const std::wstring & filterName = eff->getFxFilename();
    effectFilterIdAndName(eff->getRootDir(), eff->getFxFilename(), langID_English, effectDesc->filterId, effectDesc->filterDisplayNameEnglish);
    effectFilterIdAndName(eff->getRootDir(), eff->getFxFilename(), langID, effectDesc->filterId, effectDesc->filterDisplayName);

    return true;
}

void AnselServer::unloadEffectUIClient(size_t effIdx)
{
    m_activeControlClient->updateEffectControls(effIdx, nullptr);
}

void AnselServer::unloadEffectInternal(size_t effIdx)
{
    m_effectsInfo.m_effectsStack[effIdx] = nullptr;
    m_effectsInfo.m_effectSelected[effIdx] = 0;
    m_effectsInfo.m_effectsStackMapping[effIdx] = -1;
    // Do not cause unnecessary buffer checks
    m_effectsInfo.m_bufferCheckRequired[effIdx] = false;

    unloadEffectUIClient(effIdx);
}

void AnselServer::destroyEffectInternal(size_t effIdx)
{
    uint32_t internalIdx = m_effectsInfo.m_effectsStackMapping[effIdx];
    removeSingleEffect(internalIdx);
    for (size_t effIdx = 0, effIdxEnd = m_effectsInfo.m_effectsStackMapping.size(); effIdx < effIdxEnd; ++effIdx)
    {
        if (m_effectsInfo.m_effectsStackMapping[effIdx] > (int)internalIdx)
        {
            m_effectsInfo.m_effectsStackMapping[effIdx] -= 1;
        }
    }

    unloadEffectInternal(effIdx);
}

void AnselServer::applyEffectChange(AnselUIBase::EffectChange& effectChange, shadermod::MultiPassEffect* eff)
{
    if (!eff)
        return;

    if (effectChange.filterId != eff->getFxFileFullPath())
        return;

    if (!effectChange.filterId.empty() && (effectChange.filterId != eff->getFxFileFullPath()))
        return;

    shadermod::ir::UserConstant* uc = nullptr;
    if (effectChange.controlId == AnselUIBase::ControlIDUnknown)
    {
        uc = eff->getUserConstantManager().findByName(effectChange.controlName);
    }
    else
    {
        uc = eff->getUserConstantManager().findByUid(effectChange.controlId);
    }

    if (!uc)
        return;

#define APPLY_EFFECT_CHANGE(varType) \
    { \
        varType varType##tmp[4]; \
        effectChange.value.get(varType##tmp, effectChange.value.getDimensionality()); \
        uc->setValue(varType##tmp, effectChange.value.getDimensionality()); \
        LOG_DEBUG("  effectChange(filterId: %ls, stackIdx: %i, controlId: %i, " #varType "Value: %s)", effectChange.filterId.c_str(), effectChange.stackIdx, effectChange.controlId, effectChange.value.stringify().c_str()); \
    }

    switch (uc->getType())
    {
    case shadermod::ir::UserConstDataType::kBool:
        APPLY_EFFECT_CHANGE(bool);
        break;
    case shadermod::ir::UserConstDataType::kInt:
        APPLY_EFFECT_CHANGE(int);
        break;
    case shadermod::ir::UserConstDataType::kUInt:
        APPLY_EFFECT_CHANGE(UINT);
        break;
    case shadermod::ir::UserConstDataType::kFloat:
        APPLY_EFFECT_CHANGE(float);
        break;
    }

    // Effect change applied. Remove ID.
    effectChange.filterId = L"";
}

bool AnselServer::applyEffectChanges()
{
    static double timeSinceLastControlChange = 0.0;
    bool effectControlChanged = false;
    std::vector<AnselUIBase::EffectChange> unappliedEffectChanges;
    std::vector<AnselUIBase::EffectChange>& controlChangeQueue = m_activeControlClient->getEffectChanges();
    if (controlChangeQueue.size() > 0)
    {
        LOG_DEBUG("Applying effect/attribute Changes...");
        effectControlChanged = true;
        for (size_t ccqIdx = 0, ccqEnd = controlChangeQueue.size(); ccqIdx < ccqEnd; ++ccqIdx)
        {
            AnselUIBase::EffectChange& effectChange = controlChangeQueue[ccqIdx];

            if (effectChange.stackIdx == AnselUIBase::GameSpecificStackIdx)
            {
                // TODO: callbacks to the external wiring here
                bool isValueCorrect = true;

                // this needs to be updated in conjuction with AnselSDK
                union
                {
                    float floatValue;
                    bool boolValue;
                } val;

                if (effectChange.value.getType() == shadermod::ir::UserConstDataType::kBool)
                {
                    effectChange.value.get(&val.boolValue, 1);
                }
                else if (effectChange.value.getType() == shadermod::ir::UserConstDataType::kFloat)
                {
                    effectChange.value.get(&val.floatValue, 1);
                }
                else
                {
                    isValueCorrect = false;
                    LOG_WARN("Unsupported value type for game-specific control");
                }
                if (isValueCorrect)
                {
                    if (HIGH_QUALITY_CONTROL_ID == effectChange.controlId)
                    {
                        m_activeControlClient->setHighQualityEnabled(val.boolValue);
                    }
                    // We still need to call userControlChanged for HIGH_QUALITY_CONTROL_ID so that GFE properly saves the new value for the next Ansel session.
                    m_anselSDK.userControlChanged(effectChange.controlId, &val);
                }
            }
            else if (effectChange.stackIdx < static_cast<uint32_t>(m_effectsInfo.m_effectsStack.size())
                && m_effectsInfo.m_effectsStack[effectChange.stackIdx]
                && m_effectsInfo.m_effectsStack[effectChange.stackIdx]->getFxFileFullPath() == effectChange.filterId)
            {
                shadermod::MultiPassEffect* eff = m_effectsInfo.m_effectsStack[effectChange.stackIdx];
                applyEffectChange(effectChange, eff);
            }
            else if (effectChange.filterId != L"")
            {
                // This oclwres when the filter and attribute are set at the same time, and it is checking to set the attribute before the filter has taken effect.
                unappliedEffectChanges.push_back(effectChange);
            }
        }

        controlChangeQueue = unappliedEffectChanges;
    }

    if (!effectControlChanged)
    {
        timeSinceLastControlChange += m_globalPerfCounters.dt;
        if (timeSinceLastControlChange < 1000.0)
        {
            effectControlChanged = true;
        }
    }
    else
    {
        timeSinceLastControlChange = 0.0;
    }

    return effectControlChanged;
}

shadermod::MultipassConfigParserError AnselServer::loadEffectOnStack(size_t effIdx, size_t internalEffIdx, bool replace, const std::set<Hash::Effects> * pExpectedHashSet, bool compareHashes)
{
    shadermod::MultipassConfigParserError err(shadermod::MultipassConfigParserErrorEnum::eOK);

    int selectedEffect = m_effectsInfo.m_effectSelected[effIdx];
    if (selectedEffect > 0)
    {
        shadermod::MultiPassEffect * effectPtr;
        const std::wstring & effectRootDirectory = m_effectsInfo.m_effectRootFoldersList[selectedEffect - 1];
        if (replace)
        {
            err = replaceEffect(
                    m_installationFolderPath.c_str(), effectRootDirectory.c_str(), m_intermediateFolderPath.c_str(),
                    m_fxExtensionToolMap,
                    m_effectsInfo.m_effectFilesList[selectedEffect - 1].c_str(), &effectPtr, (int)internalEffIdx,
                    pExpectedHashSet,
                    compareHashes
                    );
        }
        else
        {
            err = pushBackEffect(
                    m_installationFolderPath.c_str(), effectRootDirectory.c_str(), m_intermediateFolderPath.c_str(),
                    m_fxExtensionToolMap,
                    m_effectsInfo.m_effectFilesList[selectedEffect - 1].c_str(), &effectPtr,
                    pExpectedHashSet,
                    compareHashes
                    );
        }
        if (!err)
        {
            m_effectsInfo.m_effectsStack[effIdx] = effectPtr;
            m_effectsInfo.m_effectsStackMapping[effIdx] = (int)internalEffIdx;
        }
        else
        {
            if (replace)
                removeSingleEffect((unsigned int)internalEffIdx, true);

            unloadEffectInternal(effIdx);

            // L"There was an error parsing the file \""
            const std::wstring wideErr = darkroom::getWstrFromUtf8(err.getFullErrorMessage().data());
            m_displayMessageStorage.resize(0);
            m_displayMessageStorage.push_back(m_effectsInfo.m_effectFilterIds[selectedEffect - 1].c_str());
            m_displayMessageStorage.push_back(wideErr.c_str());
            m_activeControlClient->displayMessage(AnselUIBase::MessageType::kErrorParsingFile, m_displayMessageStorage);

            LOG_ERROR("Effect compilation error: %s", err.getFullErrorMessage().c_str());
            OutputDebugStringA(err.getFullErrorMessage().c_str());

            //telemetry
            {
                ErrorDescForTelemetry desc;

                desc.errorCode = 0;
                desc.lineNumber = __LINE__;
                desc.errorMessage = err.getFullErrorMessage();
                desc.filename = __FILE__;

                AnselStateForTelemetry state;
                HRESULT telemetryStatus = makeStateSnapshotforTelemetry(state);
                if (telemetryStatus == S_OK)
                    sendTelemetryEffectCompilationErrorEvent(state, m_anselSDK.getCaptureState(), desc);
            }
        }

        // Reset error
        err = shadermod::MultipassConfigParserErrorEnum::eOK;
    }
    else
    {
        if (replace)
        {
            removeSingleEffect((unsigned int)internalEffIdx, false);
            for (size_t effIdx = 0, effIdxEnd = m_effectsInfo.m_effectsStackMapping.size(); effIdx < effIdxEnd; ++effIdx)
            {
                if (m_effectsInfo.m_effectsStackMapping[effIdx] > (int)internalEffIdx)
                {
                    m_effectsInfo.m_effectsStackMapping[effIdx] -= 1;
                }
            }
        }


        m_effectsInfo.m_effectsStack[effIdx] = nullptr;
        m_effectsInfo.m_effectsStackMapping[effIdx] = -1;
    }

    shadermod::MultiPassEffect* eff = m_effectsInfo.m_effectsStack[effIdx];
    if (eff)
    {
        LOG_DEBUG("Set Filter: %ls", eff->getFxFileFullPath().c_str());

        std::vector<AnselUIBase::EffectChange>& controlChangeQueue = m_activeControlClient->getEffectChanges();
        // Check to apply any pending attribute changes
        for (size_t ccqIdx = 0, ccqEnd = controlChangeQueue.size(); ccqIdx < ccqEnd; ++ccqIdx)
        {
            applyEffectChange(controlChangeQueue[ccqIdx], eff);
        }
    }

    return err;
}

void AnselServer::changeEffectState(bool enableShaderMod)
{
    m_bNextFrameRunShaderMod = enableShaderMod;
}

std::wstring AnselServer::getD3dCompilerFullPath() const
{
    std::wstring d3dCompiler = L"d3dcompiler_47_64.dll";

#if defined(_M_IX86)
    d3dCompiler = L"d3dcompiler_47_32.dll";
#endif

    std::wstring fullPath = m_toolsFolderPath + d3dCompiler;
    return fullPath;
}

uint32_t RemoveFilenameFromPath(wchar_t* pathW, const uint32_t maxLength)
{
    uint32_t appNameLen = 0;
    wchar_t* appNameW = nullptr;
    uint32_t pathLen = 0;

    if (pathW)
    {
        pathLen = 0;
        wchar_t* filePath = pathW;
        while ((*filePath != 0) && (pathLen < maxLength))
        {
            filePath++;
            pathLen++;
        }
    }

    if ((pathLen > 0) && (pathLen < MAX_PATH))
    {
        appNameW = pathW + pathLen;
        while (appNameW > pathW)
        {
            if (*appNameW == '\\' || *appNameW == '/')
            {
                appNameW++;
                break;
            }
            appNameW--;
            appNameLen++;
        }
        uint32_t directoryPathLen = pathLen - appNameLen;
        pathW[directoryPathLen] = 0;
        return directoryPathLen;
    }

    return -1;
}

void AnselServer::initRegistryDependentPathsAndOptions()
{
    m_areMouseButtonsSwapped = (GetSystemMetrics(SM_SWAPBUTTON) != 0);
    const auto readRegistrySettings = [&]()
    {
        LOG_INFO("Settings read from registry:");
        std::wstring modulePath(MAX_PATH, ' ');
        GetModuleFileName(m_hDLLInstance, &modulePath[0], DWORD(modulePath.size()));

        uint32_t directoryPathLen = RemoveFilenameFromPath(&modulePath[0], MAX_PATH);
        if (directoryPathLen != -1)
        {
            m_installationFolderPath = modulePath.c_str();      // Directory where the LwCameraXX.dll is loaded from
            m_installationFolderPath.append(L"\\");
            m_toolsFolderPath = m_installationFolderPath;
            m_effectInstallationFolderPath = m_installationFolderPath;
        }
        else
        {
            LOG_FATAL("Unable to find directory where LwCamera DLL is located");
        }

        m_effectFoldersAdditional.resize(0);

        // The priority order for loading effects is supposed to be:
        // 1. Application folder\LwCamera
        // 2. Custom Folder: Program Files\LWPU Corporation\Ansel\Custom
        // 3. Any other folders specified through regkey
        // 4. DriverStore\LwCamera (only for DCH) (comes from m_installationFolderPath...added in populateEffectsList)
        // 5. Program Files\LWPU Corporation\Ansel
        // Note: the search order depends on the order of insertion of paths into m_effectFoldersAdditional
        HMODULE hModule = GetModuleHandle(NULL);
        if (hModule)
        {
            wchar_t appPath[MAX_PATH];
            DWORD dw = GetModuleFileNameW(hModule, appPath, MAX_PATH);
            if (dw)
            {
                directoryPathLen = RemoveFilenameFromPath(appPath, MAX_PATH);
                if (directoryPathLen != -1)
                {
                    errno_t err = wcsncat_s(appPath, L"\\LwCamera\\", 10);          // Application folder\LwCamera
                    if (err != 0)
                    {
                        LOG_WARN("Unable to load filters from the application folder");
                    }
                    else
                    {
                        m_effectFoldersAdditional.push_back(appPath);
                    }
                }
                else
                {
                    LOG_WARN("Unable to find the application folder");
                }
            }
        }


        // Temporary solution for an issue loading Custom filters from the wrong program files directory. Ideally, we wouldn't like to rely on a system regkey
        // since it may change or we may lose access to it in the future
        // If read registry needed in CV Guest OS, we may require to change implementation to read host registry (use pfnQAI2 for host registry).
        wchar_t programFilesPath[MAX_PATH] = { 0 };

        const wchar_t* KeyName = L"ProgramFilesDir";
        HKEY hKey;
        HRESULT getFolderPathHR = RegOpenKeyEx(HKEY_LOCAL_MACHINE,
            _T("SOFTWARE\\Microsoft\\Windows\\LwrrentVersion"),
            NULL,
            KEY_WOW64_64KEY | KEY_READ,
            &hKey);
        if (SUCCEEDED(getFolderPathHR))
        {
            DWORD Type;
            DWORD cbData = MAX_PATH;
            getFolderPathHR = RegQueryValueExW(hKey, KeyName, NULL, &Type, (BYTE*)programFilesPath, &cbData);
            RegCloseKey(hKey);


            if (SUCCEEDED(getFolderPathHR))
            {
                wchar_t lwstomFolderPath[MAX_PATH];
                wcscpy_s(lwstomFolderPath, programFilesPath);
                errno_t err = wcsncat_s(lwstomFolderPath, L"\\LWPU Corporation\\Ansel\\Custom\\", 33);        // Custom folder: Program Files\LWPU Corporation\Ansel\Custom
                if (err != 0)
                {
                    LOG_WARN("Unable to load filters from the custom folder");
                }
                else
                {
                    m_effectFoldersAdditional.push_back(lwstomFolderPath);
                }
            }
        }
        else
        {
            LOG_WARN("Unable to query Program Files path");
        }



        m_registrySettings.markDirty();

        // Semicolon-separated list of folders
        std::wstring effectFoldersAdditional;
        effectFoldersAdditional = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::EffectFolders), L"");

        std::wistringstream effectFoldersStream(effectFoldersAdditional);
        if (!effectFoldersAdditional.empty())
        {
            std::wstring effectFolder;
            while (std::getline(effectFoldersStream, effectFolder, L';'))
            {
                const size_t effectFolderLen = effectFolder.length();

                // Effect folder is expected to have (back)slash at the end
                const wchar_t lastCharacter = effectFolder[effectFolderLen-1];
                if (lastCharacter != L'\\' && lastCharacter != L'/')
                {
                    effectFolder.append(L"\\");
                }

                // Basic validation
                if (!PathIsDirectory(effectFolder.c_str()))
                {
                    continue;
                }

                m_effectFoldersAdditional.push_back(effectFolder);          // Any other folders specified through regkey
            }
        }
        m_effectFoldersAdditional.push_back(m_installationFolderPath);      // DriverStore\LwCamera (in case of DCH)... Program Files\LWPU Corporation\Ansel in case of non-DCH

        if (SUCCEEDED(getFolderPathHR))
        {
            errno_t err = wcsncat_s(programFilesPath, L"\\LWPU Corporation\\Ansel\\", 26);
            if (err != 0)
            {
                LOG_WARN("Unable to load filters from the application folder");
            }
            else if (m_installationFolderPath.compare(programFilesPath))
            {
                m_effectFoldersAdditional.push_back(programFilesPath);          // Program Files\LWPU Corporation\Ansel... only add it in case of DCH (non-DCH already has it)
            }
        }

        // Remove duplicates from m_effectFoldersAdditional
        {
            std::unordered_set<std::wstring> uniqueFolderSet;
            for (int i = 0; i < static_cast<int>(m_effectFoldersAdditional.size()); i++)
            {
                if (uniqueFolderSet.find(m_effectFoldersAdditional[i]) != uniqueFolderSet.end())
                {
                    m_effectFoldersAdditional.erase(m_effectFoldersAdditional.begin() + i);
                    i--;
                    continue;
                }
                uniqueFolderSet.insert(m_effectFoldersAdditional[i]);
            }
        }

        // Initialize PhotoShopUtils tool by providing it with possible folders where the ShotByGeforce watermark may be found
        m_psUtil.init(m_effectFoldersAdditional);

        m_snapshotsFolderPath = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::SnapshotDir), L"");
        if (m_snapshotsFolderPath.empty())
        {
            LOG_INFO("Reading from ShadowPlay key");
            m_snapshotsFolderPath = m_registrySettings.getValue(m_registrySettings.registryPathShadowPlay(), darkroom::getWstrFromUtf8(Settings::SnapshotShadowplayDir), L"");
        }
        m_settingsAsStrings[Settings::SnapshotDir] = m_snapshotsFolderPath;
        LOG_INFO(">   Snapshots path = '%s'", darkroom::getUtf8FromWstr(m_snapshotsFolderPath).c_str());
        const auto defaultUserStylesPath = m_snapshotsFolderPath + L"Ansel\\Styles";
        m_userStylesFolderPath = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::StylesDir), defaultUserStylesPath.c_str());
        m_settingsAsStrings[Settings::StylesDir] = m_userStylesFolderPath;
        LOG_INFO(">   User styles path = '%s'", darkroom::getUtf8FromWstr(m_userStylesFolderPath).c_str());
        m_intermediateFolderPath = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::IntermediateDir), L"");
        m_settingsAsStrings[Settings::IntermediateDir] = m_intermediateFolderPath;
        LOG_INFO(">   Intermediates path = '%s'", darkroom::getUtf8FromWstr(m_intermediateFolderPath).c_str());
        m_maxHighResResolution = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::MaxHighRes), 4, 128, 63);
        m_settingsAsStrings[Settings::MaxHighRes] = std::to_wstring(m_maxHighResResolution);
        LOG_INFO(">   MaxHighResResolution = %d", m_maxHighResResolution);
        m_maxSphericalResolution = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::MaxSphereRes), 4, 63, 8);
        m_settingsAsStrings[Settings::MaxSphereRes] = std::to_wstring(m_maxSphericalResolution);
        LOG_INFO(">   MaxSphericalResolution = %d", m_maxSphericalResolution);
        m_eyeSeparation = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::StereoEyeSeparation), 1.0f, 10.0f, 6.3f);
        m_settingsAsStrings[Settings::StereoEyeSeparation] = std::to_wstring(m_eyeSeparation);
        LOG_INFO(">   EyeSeparation = %f", m_eyeSeparation);
        m_cameraSpeedMultiplier = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::CameraSpeedMult), 1.0f, 10.0f, 4.0f);
        m_settingsAsStrings[Settings::CameraSpeedMult] = std::to_wstring(m_cameraSpeedMultiplier);
        LOG_INFO(">   CameraSpeedMultiplier = %f", m_cameraSpeedMultiplier);

        // we don't support standalone <-> ipc switch at runtime, so only initialize it once
        if (!m_ipcValueInitialized)
        {
            if (m_registrySettings.valueExists(m_registrySettings.registryPathAnsel(), L"", darkroom::getWstrFromUtf8(Settings::IPCenabled).c_str()))
            {
                m_WAR_IPCmodeEnabled = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::IPCenabled), -100, 100, 0);
            }
            else
            {
                m_WAR_IPCmodeEnabled = m_registrySettings.getValue(m_registrySettings.registryPathAnselBackup(), darkroom::getWstrFromUtf8(Settings::IPCenabled), -100, 100, 0);
            }
            m_settingsAsStrings[Settings::IPCenabled] = std::to_wstring(m_WAR_IPCmodeEnabled);
            LOG_INFO(">   IPCMode = %d", m_WAR_IPCmodeEnabled);
            m_ipcValueInitialized = true;
        }

        m_removeBlackTint = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::RemoveBlackTint), false);
        m_settingsAsStrings[Settings::RemoveBlackTint] = std::to_wstring(m_removeBlackTint);
        LOG_INFO(">   RemoveBlackTint = %d", m_removeBlackTint);
        m_keepIntermediateShots = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::KeepIntermediateShots), false);
        m_settingsAsStrings[Settings::KeepIntermediateShots] = std::to_wstring(m_keepIntermediateShots);
        LOG_INFO(">   KeepIntermediates = %d", m_keepIntermediateShots);
#ifdef ENABLE_STYLETRANSFER
        if (!m_bInitialized)
        {
            // Only read this key once at startup
            m_isStyleTransferEnabled = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::EnableStyleTransfer), true);
            m_settingsAsStrings[Settings::EnableStyleTransfer] = std::to_wstring(m_isStyleTransferEnabled);
        }
        LOG_INFO(">   EnableStyleTransfer = %d", (int)m_isStyleTransferEnabled);
        m_allowStyleTransferWhileMovingCamera = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::AllowStyleTransferWhileMoving), false);
        m_settingsAsStrings[Settings::AllowStyleTransferWhileMoving] = std::to_wstring(m_allowStyleTransferWhileMovingCamera);
        LOG_INFO(">   AllowStyleTransferWhileMoving = %d", m_allowStyleTransferWhileMovingCamera);
#endif
        m_renderDebugInfo = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::RenderDebugInformation), false);
        m_settingsAsStrings[Settings::RenderDebugInformation] = std::to_wstring(m_renderDebugInfo);
        LOG_INFO(">   RenderDebugInfo = %d", m_renderDebugInfo);
        m_losslessOutputSuperRes = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::LosslessOutput), false);
        m_settingsAsStrings[Settings::LosslessOutput] = std::to_wstring(m_losslessOutputSuperRes);
        LOG_INFO(">   LosslessSuperResolution = %d", m_losslessOutputSuperRes);
        m_losslessOutput360 = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::LosslessOutput360), false);
        m_settingsAsStrings[Settings::LosslessOutput360] = std::to_wstring(m_losslessOutput360);
        LOG_INFO(">   Lossless360 = %d", m_losslessOutput360);
        m_allowNotifications = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::AllowNotifications), true);
        m_settingsAsStrings[Settings::AllowNotifications] = std::to_wstring(m_allowNotifications);
        LOG_INFO(">   Notifications are %s", m_allowNotifications ? "allowed" : "not allowed");
        m_enableEnhancedHighres = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::EnableEnhancedHighres), false);
        m_settingsAsStrings[Settings::EnableEnhancedHighres] = std::to_wstring(m_enableEnhancedHighres);
        LOG_INFO(">   Enhancing highres captures are %s", m_enableEnhancedHighres ? "enabled" : "disabled");
        float enhancedHighresCoeffUser = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::EnhancedHighresCoeff), 0.1f, 1.0f, 0.35f);
        m_settingsAsStrings[Settings::EnhancedHighresCoeff] = std::to_wstring(enhancedHighresCoeffUser);
        LOG_INFO(">   Enhancing highres coefficient = %.0f%%", enhancedHighresCoeffUser * 100.0f);
        m_enhancedHighresCoeff = m_enhancedHighresCoeffMinimal + enhancedHighresCoeffUser * (m_enhancedHighresCoeffAggressive - m_enhancedHighresCoeffMinimal);

        // Registry key for whether to allow the BufferTestingOptions.yaml filter to be used
        bool allowBufferOptionsFilter = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::AllowBufferOptionsFilter), false);
        m_settingsAsStrings[Settings::AllowBufferOptionsFilter] = std::to_wstring(allowBufferOptionsFilter);
        LOG_INFO(">   Allow buffer options filter to be used: %s", allowBufferOptionsFilter ? "yes" : "no");
        m_bufTestingOptionsFilter.toggleAllow(allowBufferOptionsFilter);

        // Registry key for whether to enable saving capture as a PhotShop (.psd) file
        bool isPsdExportEnabled = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::SaveCaptureAsPhotoShop), false);
        m_settingsAsStrings[Settings::SaveCaptureAsPhotoShop] = std::to_wstring(isPsdExportEnabled);
        LOG_INFO(">   Save capture as PhotoShop (.psd) file: %s", isPsdExportEnabled ? "yes" : "no");
        m_psUtil.SetPsdExportEnable(isPsdExportEnabled);

        if (!m_bInitialized)
        {
            // We can only switch controllers on Ansel init
            //  we can also do it per-session theoretically, but we don't have such logic yet
            m_useHybridController = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::UseHybridController), false);
            m_settingsAsStrings[Settings::UseHybridController] = std::to_wstring(m_useHybridController);
        }

        if (m_useHybridController)
        {
            LOG_INFO(">   Using Hybrid camera controller");
        }

        m_allowTelemetry = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::AllowTelemetry), true);
        LOG_INFO(">   Telemetry collection is %s", m_allowTelemetry ? "enabled" : "disabled");
        m_settingsAsStrings[Settings::AllowTelemetry] = std::to_wstring(m_allowTelemetry);
        m_requireAnselSDK = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::RequireSDK), true);
        m_settingsAsStrings[Settings::RequireSDK] = std::to_wstring(m_requireAnselSDK);
        LOG_INFO(">   Ansel SDK is %s", m_requireAnselSDK ? "required" : "not required");

        m_standaloneModding = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::StandaloneModding), false);
        m_settingsAsStrings[Settings::StandaloneModding] = std::to_wstring(m_standaloneModding);

        m_allowFiltersInGame = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::FiltersInGame), false);
        m_settingsAsStrings[Settings::FiltersInGame] = std::to_wstring(m_allowFiltersInGame);
        m_allowDynamicFilterStacking = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::DynamicFilterStacking), true);
        m_settingsAsStrings[Settings::DynamicFilterStacking] = std::to_wstring(m_allowDynamicFilterStacking);

        m_modEnableFeatureCheck = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::ModsEnabledCheck), true);
        m_settingsAsStrings[Settings::ModsEnabledCheck] = std::to_wstring(m_modEnableFeatureCheck);
        m_checkTraficLocal = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::CheckTraficLocal), true);
        m_settingsAsStrings[Settings::CheckTraficLocal] = std::to_wstring(m_checkTraficLocal);

        //LOG_INFO(">   Ansel SDK is %s", m_allowFiltersInGame ? "required" : "not required");

        if (m_WAR_IPCmodeEnabled)
        {
            m_requireAnselSDK = false;
            m_allowFiltersInGame = m_moddingSinglePlayer.mode != ModdingMode::kDisabled;
        }
        else if (!m_bInitialized)
        {
            std::wstring localeName = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::ForceLang), L"");
            if (!localeName.empty())
            {
                m_forcedLocale = LocaleNameToLCID(localeName.c_str(), 0);
                if (m_forcedLocale != 0)
                {
                    LOG_INFO(">   ForceLang = %s", darkroom::getUtf8FromWstr(localeName).c_str());
                }
            }
            m_settingsAsStrings[Settings::ForceLang] = std::to_wstring(m_forcedLocale);
        }

        m_toggleHotkeyModCtrl = static_cast<USHORT>(m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::ToggleHotkeyModCtrl), DWORD(0), DWORD(0xFF), DWORD(0)));
        m_toggleHotkeyModShift = static_cast<USHORT>(m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::ToggleHotkeyModShift), DWORD(0), DWORD(0xFF), DWORD(0)));
        m_toggleHotkeyModAlt = static_cast<USHORT>(m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::ToggleHotkeyModAlt), DWORD(0), DWORD(0xFF), DWORD(1)));
        m_toggleAnselHotkey = static_cast<USHORT>(m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::ToggleHotkey), DWORD(0), DWORD(0xFF), DWORD(VK_F2)));

        m_settingsAsStrings[Settings::ToggleHotkeyModCtrl] = std::to_wstring(m_toggleHotkeyModCtrl);
        m_settingsAsStrings[Settings::ToggleHotkeyModShift] = std::to_wstring(m_toggleHotkeyModShift);
        m_settingsAsStrings[Settings::ToggleHotkeyModAlt] = std::to_wstring(m_toggleHotkeyModAlt);
        m_settingsAsStrings[Settings::ToggleHotkey] = std::to_wstring(m_toggleAnselHotkey);

        {
            const size_t vkNameMaxSize = 24;
            wchar_t vkName[vkNameMaxSize];

            UINT scanCode = MapVirtualKeyW(m_toggleAnselHotkey, MAPVK_VK_TO_VSC);
            LONG lParamValue = (scanCode << 16);

            int result = GetKeyNameTextW(lParamValue, vkName, vkNameMaxSize);

            m_toggleHotkeyComboText.resize(0);
            if (m_toggleHotkeyModCtrl)
                m_toggleHotkeyComboText += L"Ctrl + ";
            if (m_toggleHotkeyModShift)
                m_toggleHotkeyComboText += L"Shift + ";
            if (m_toggleHotkeyModAlt)
                m_toggleHotkeyComboText += L"Alt + ";

            m_toggleHotkeyComboText += vkName;

            LOG_INFO(">   ToggleHotkeyCombo = %s", darkroom::getUtf8FromWstr(m_toggleHotkeyComboText).c_str());
        }

        if (m_intermediateFolderPath.empty())
        {
            // we should use windows temp if user doesn't specify location
            getUserTempDirectory(m_intermediateFolderPath);
            m_intermediateFolderPath.append(L"LwCamera\\");
        }

        if (m_snapshotsFolderPath.empty())
        {
            // we should use user's Videos folder if user doesn't specify another location
            wchar_t* videosPath = nullptr;
            if (SHGetKnownFolderPath(FOLDERID_Videos, KF_FLAG_CREATE, NULL, &videosPath) == S_OK)
            {
                m_snapshotsFolderPath = videosPath;
            }
        }

        // Append trailing backslash if missing:
        if (!m_snapshotsFolderPath.empty())
        {
            if (m_snapshotsFolderPath.back() != L'\\')
            {
                m_snapshotsFolderPath.push_back(L'\\');
            }
        }
    };

    readRegistrySettings();
    m_registrySettings.onRegistryKeyChanged(readRegistrySettings);

    const std::wstring welcomeNotificationCounterValueName = L"WelcomeNotificationCounter";
    std::wstring titleRegistryKeyName;

    // First try to get a profile name:
    wchar_t appPath[APP_PATH_MAXLEN];
    appPath[0] = '\0'; // null terminating buffer at start
    DWORD length = GetModuleFileNameW(NULL, appPath, APP_PATH_MAXLEN);

    // Buffer should always be large enough but we might as well check so that the code
    // is robust
    if (length < APP_PATH_MAXLEN)
    {
        if (!drs::getProfileName(titleRegistryKeyName))
        {
            size_t pos = 0;
            std::wstring path(appPath);
            const std::wstring delim(L"\\");
            std::vector<std::wstring> pathComponents;
            while ((pos = path.find(delim)) != std::string::npos)
            {
                pathComponents.push_back(path.substr(0, pos));
                path.erase(0, pos + delim.size());
            }

            pathComponents.push_back(path);

            const ptrdiff_t maxComponents = 4;
            if (pathComponents.size() > maxComponents)
                pathComponents = decltype(pathComponents)(pathComponents.cend() - maxComponents, pathComponents.cend());

            for (size_t i = 0u; i < pathComponents.size() - 1; ++i)
                titleRegistryKeyName += pathComponents[i] + L"_";
            titleRegistryKeyName += pathComponents.back();

            std::transform(titleRegistryKeyName.cbegin(), titleRegistryKeyName.cend(), titleRegistryKeyName.begin(), [](const wchar_t& c)
            {
                if (c == L'\\' || c == L':')
                    return L'_';
                return c;
            });
        }
    }

    if (!titleRegistryKeyName.empty())
    {
        LOG_INFO("DRS Profile name: \"%ls\"", titleRegistryKeyName.c_str());
        // check application key exists
        DWORD disposition = 0;
        int32_t welcomeNotificationCounter = WELCOME_NOTIFICATION_DEFAULT;
        m_registrySettings.createKey(m_registrySettings.registryPathAnsel(), titleRegistryKeyName.c_str(), disposition);
        // check welcome notification setting exists
        if (m_registrySettings.valueExists(m_registrySettings.registryPathAnsel(),
            titleRegistryKeyName.c_str(),
            welcomeNotificationCounterValueName.c_str()))
        {
            welcomeNotificationCounter = m_registrySettings.getValue(m_registrySettings.registryPathAnsel(),
                titleRegistryKeyName.c_str(), welcomeNotificationCounterValueName.c_str(),
                0u, std::numeric_limits<int32_t>::max(), WELCOME_NOTIFICATION_DEFAULT);
            // decrement welcome notification counter
            if (welcomeNotificationCounter > 0)
            {
                m_registrySettings.setValue(m_registrySettings.registryPathAnsel(),
                    titleRegistryKeyName.c_str(), welcomeNotificationCounterValueName.c_str(), welcomeNotificationCounter - 1);
                m_enableWelcomeNotification = true;
            }
            LOG_DEBUG("Notifications are %s, counter is %d", m_enableWelcomeNotification ? "enabled" : "disabled", welcomeNotificationCounter);
        }
        else
        {
            LOG_INFO("No DRS profile found!");
            // if it doesn't exist, create a key with decremented value
            m_registrySettings.setValue(m_registrySettings.registryPathAnsel(),
                titleRegistryKeyName.c_str(), welcomeNotificationCounterValueName.c_str(), welcomeNotificationCounter - 1);
            m_enableWelcomeNotification = true;
            LOG_DEBUG("Notifications are enabled (not set), counter is %d", welcomeNotificationCounter);
        }
    }
    else
        LOG_ERROR("Couldn't generate valid title registry key name");

    if (m_UI)
        m_UI->updateSettings(m_settingsAsStrings);
}

template <typename T1, typename T2>
void buildMapIdentifiersList(std::map<T1, T2> srcMap, std::vector<T1> * outList)
{
    if (outList == nullptr)
        return;

    outList->clear();
    for (auto it = srcMap.begin(); it != srcMap.end(); ++it)
    {
        outList->push_back(it->first);
    }
}

HRESULT AnselServer::parseEffectCompilersList(std::wstring filename)
{
    m_fxExtensionToolMap.clear();

    std::ifstream ifs(filename);
    if (!ifs)
    {
        return E_FAIL;
    }

    toml::ParseResult pr = toml::parse(ifs);

    if (!pr.valid())
    {
        return E_FAIL;
    }

    toml::Value * v = &pr.value;
    const toml::Table & topLevelTable = v->as<toml::Table>();

    size_t filterIdx = 0;
    for (auto it = topLevelTable.begin(); it != topLevelTable.end(); ++it)
    {
        if (it->second.type() != toml::Value::TABLE_TYPE)
            continue;

        std::wstring extension = darkroom::getWstrFromUtf8(it->first);
        darkroom::tolowerInplace(extension);

        std::wstring toolName;
        std::wstring toolExt;

        const toml::Table & extPropertiesTable = it->second.as<toml::Table>();
        for (auto locIt = extPropertiesTable.begin(); locIt != extPropertiesTable.end(); ++locIt)
        {
            std::wstring fieldName(darkroom::getWstrFromUtf8(locIt->first));
            darkroom::tolowerInplace(fieldName);

            if ((fieldName == L"tool") && (locIt->second.type() == toml::Value::STRING_TYPE))
            {
                toolName = darkroom::getWstrFromUtf8(locIt->second.as<std::string>());
            }
            else if ((fieldName == L"ext") && (locIt->second.type() == toml::Value::STRING_TYPE))
            {
                toolExt = darkroom::getWstrFromUtf8(locIt->second.as<std::string>());
            }
        }
        static const std::wstring s_platformArch =
#if _M_AMD64
            L"64";
#else
            L"32";
#endif

        m_fxExtensionToolMap.insert(std::make_pair(extension, toolName + s_platformArch + L"." + toolExt));
    }

    return S_OK;
}

HRESULT AnselServer::dumpFilterStack()
{
    const size_t maxIndexLen = 16;
    char formattedIndex[maxIndexLen];

    std::wstring filterDumpFolder = m_intermediateFolderPath + L"_filterDumps\\";
    if (!lwanselutils::CreateDirectoryRelwrsively(filterDumpFolder.c_str()))
        return E_FAIL;

    std::wstring filterDumpFilename = lwanselutils::appendTimeW(m_appName.c_str(), L".txt");
    std::ofstream ofs((filterDumpFolder + filterDumpFilename).c_str());
    if (!ofs.is_open())
        return E_FAIL;

    std::wstring filterName;
    std::string formattedName;
    std::string filterNameUTF8;

    std::vector<toml::Value> tomlArray;
    toml::Value filterStackDesc;

    for (size_t effIdx = 0, effIdxEnd = m_effectsInfo.m_effectsStack.size(); effIdx < effIdxEnd; ++effIdx)
    {
        shadermod::MultiPassEffect* eff = m_effectsInfo.m_effectsStack[effIdx];

        filterName = shadermod::Tools::wstrNone;
        toml::Value filterDesc = toml::Table();
        if (eff)
        {
            filterName = eff->getFxFilename();

            shadermod::ir::UserConstantManager::Range ucrange = eff->getUserConstantManager().getPointersToAllUserConstants();
            int constantIdx = 0;
            for (; ucrange.begin < ucrange.end; ++ucrange.begin)
            {
                toml::Value filterConstantDesc;

                const shadermod::ir::UserConstant * lwrUC = *(ucrange.begin);

                const shadermod::ir::TypelessVariable & value = lwrUC->getValue();
                switch (value.getType())
                {
                case shadermod::ir::UserConstDataType::kBool:
                {
                    uint32_t dims = value.getDimensionality();

                    bool boolValue[4];
                    value.get(boolValue, 4);

                    if (dims > 1)
                    {
                        tomlArray.resize(0);
                        for (uint32_t di = 0; di < dims; ++di)
                            tomlArray.push_back(boolValue[di]);
                        filterConstantDesc = toml::Value(tomlArray);
                    }
                    else
                    {
                        filterConstantDesc = toml::Value(boolValue[0]);
                    }

                    break;
                }
                case shadermod::ir::UserConstDataType::kFloat:
                {
                    uint32_t dims = value.getDimensionality();

                    float floatValue[4];
                    value.get(floatValue, 4);

                    if (dims > 1)
                    {
                        tomlArray.resize(0);
                        for (uint32_t di = 0; di < dims; ++di)
                            tomlArray.push_back(floatValue[di]);
                        filterConstantDesc = toml::Value(tomlArray);
                    }
                    else
                    {
                        filterConstantDesc = toml::Value(floatValue[0]);
                    }

                    break;
                }
                case shadermod::ir::UserConstDataType::kInt:
                {
                    uint32_t dims = value.getDimensionality();

                    int intValue[4];
                    value.get(intValue, 4);

                    if (dims > 1)
                    {
                        tomlArray.resize(0);
                        for (uint32_t di = 0; di < dims; ++di)
                            tomlArray.push_back(intValue[di]);
                        filterConstantDesc = toml::Value(tomlArray);
                    }
                    else
                    {
                        filterConstantDesc = toml::Value(intValue[0]);
                    }

                    break;
                }
                case shadermod::ir::UserConstDataType::kUInt:
                {
                    uint32_t dims = value.getDimensionality();

                    uint32_t uintValue[4];
                    value.get(uintValue, 4);

                    if (dims > 1)
                    {
                        tomlArray.resize(0);
                        for (uint32_t di = 0; di < dims; ++di)
                            tomlArray.push_back((int)uintValue[di]);
                        filterConstantDesc = toml::Value(tomlArray);
                    }
                    else
                    {
                        filterConstantDesc = toml::Value((int)uintValue[0]);
                    }

                    break;
                }
                }

                sprintf_s(formattedIndex, maxIndexLen, "%02d:", constantIdx);
                formattedName = formattedIndex + lwrUC->getName();
                filterDesc[formattedName] = filterConstantDesc;

                ++constantIdx;
            }
        }

        sprintf_s(formattedIndex, maxIndexLen, "%02d:", (int)effIdx);
        filterNameUTF8 = formattedIndex + darkroom::getUtf8FromWstr(filterName);
        filterStackDesc[filterNameUTF8] = filterDesc;
    }

    filterStackDesc.write(&ofs);

    return S_OK;
}

std::wstring AnselServer::generateActiveEffectsTag() const
{
    std::wstring effectList(L"");
    for (size_t effIdx = 0, effIdxEnd = m_effectsInfo.m_effectsStack.size(); effIdx < effIdxEnd; ++effIdx)
    {
        shadermod::MultiPassEffect* eff = m_effectsInfo.m_effectsStack[effIdx];
        if (eff)
        {
            effectList += eff->getFxFilename() + L",";
        }
    }
    effectList = effectList.substr(0, std::max(0, int(effectList.size()) - 1)); // cut off last comma
    return effectList;
}

std::wstring AnselServer::generateSoftwareTag() const
{
    std::wstring softwareTag(L"");
    if (getIPCModeEnabled())
    {
        softwareTag.append(L"VdChip Experience - Ansel");
    }
    else
    {
        softwareTag.append(L"Ansel");
    }

    if (!m_anselSDK.isDetected())
    {
        softwareTag.append(L" NoSDK");
    }
    return softwareTag;
}

HRESULT AnselServer::initCanFail(HANSELCLIENT hClient, ClientFunctionTable * pFunctionTable, LARGE_INTEGER adapterLUID)
{
    m_hClient = hClient;
    m_pClientFunctionTable = pFunctionTable;


    ANSEL_PROFILE_INIT("Ansel Profiling");

#if defined(_M_IX86)
    LOG_DEBUG("32-bit mode");
#else
    LOG_DEBUG("64-bit mode");
#endif

    WCHAR appPath[APP_PATH_MAXLEN];
    GetModuleFileName(NULL, appPath, APP_PATH_MAXLEN); // TODO: Check return value.
    const std::wstring appPathString(appPath);

    m_denylist.CheckToInitializeWithDRS();

    // Read what buffers have been disabled
    uint32_t buffersDisabledValue = getBuffersDisabled();
    {
        if (m_denylist.BufferDenylisted(ansel::BufferType::kBufferTypeDepth)        || buffersDisabledValue & ANSEL_BUFFERS_DISABLED_DEPTH)         { DenylistBufferExtractionType(ansel::BufferType::kBufferTypeDepth); }
        if (m_denylist.BufferDenylisted(ansel::BufferType::kBufferTypeHDR)          || buffersDisabledValue & ANSEL_BUFFERS_DISABLED_HDR)           { DenylistBufferExtractionType(ansel::BufferType::kBufferTypeHDR); }
        if (m_denylist.BufferDenylisted(ansel::BufferType::kBufferTypeHUDless)      || buffersDisabledValue & ANSEL_BUFFERS_DISABLED_HUDLESS)       { DenylistBufferExtractionType(ansel::BufferType::kBufferTypeHUDless); }
        if (m_denylist.BufferDenylisted(ansel::BufferType::kBufferTypeFinalColor)   || buffersDisabledValue & ANSEL_BUFFERS_DISABLED_FINAL_COLOR)   { DenylistBufferExtractionType(ansel::BufferType::kBufferTypeFinalColor); }
    }

    uint32_t freeStyleModeValue = ANSEL_FREESTYLE_MODE_DEFAULT;
    {
        DWORD pid = GetLwrrentProcessId();
        freeStyleModeValue = getFreeStyleMode(pid, &m_denylist);

#define DBG_HARDCODE_ALLOWLIST_TESTAPPS 1
#if (DBG_HARDCODE_ALLOWLIST_TESTAPPS == 1)
        std::wstring appFilename;
        size_t backslashPos = appPathString.rfind(L'\\');
        if (backslashPos > 0)
            appFilename = appPathString.substr(backslashPos + 1);
        else
            appFilename = appPathString;
        darkroom::tolowerInplace(appFilename);

        bool needToCheckMutex = false;
        if (appFilename == L"anselintegrationtestapp.exe")
        {
            // D3D11 test app
            needToCheckMutex = true;
        }
        else if (appFilename == L"anselintegrationtestapp9.exe")
        {
            // D3D9 test app
            needToCheckMutex = true;
        }
        else if (appFilename == L"vkballsansel.exe")
        {
            // Vulkan test app
            needToCheckMutex = true;
        }
        else if (appFilename == L"anselintegrationtestapp12.exe")
        {
            // DX12 test app
            needToCheckMutex = true;
        }

        // All of our test apps should use latest SDK that feature explicit SDK mutex creation
        if (needToCheckMutex)
        {
            char name[MAX_PATH];
            sprintf_s(name, "LWPU/Ansel/%d", pid);
            HANDLE mutex = OpenMutexA(SYNCHRONIZE, false, name);

            if (mutex)
            {
                CloseHandle(mutex);

                // In case of the hardcoded set of testing apps, allow reading non-gold DRS key
                uint32_t freeStyleModeValueNonGold = ANSEL_FREESTYLE_MODE_DEFAULT;
                const bool isFreestyleModeKeySetNonGold = drs::getKeyValue(ANSEL_FREESTYLE_MODE_ID, freeStyleModeValueNonGold, false);
                if (isFreestyleModeKeySetNonGold)
                {
                    // only allow ENABLED or APPROVED_ONLY for test apps
                    if (freeStyleModeValueNonGold == ANSEL_FREESTYLE_MODE_ENABLED
                        || freeStyleModeValueNonGold == ANSEL_FREESTYLE_MODE_APPROVED_ONLY)
                    {
                        freeStyleModeValue = freeStyleModeValueNonGold;
                    }
                }
            }
        }
#endif
    }

    handleDrsGameSettings();

//#define FORCE_UNRESTRICTED_FREESTYLE
//#define FORCE_RESTRICTED_FREESTYLE // FORCE_UNRESTRICTED_FREESTYLE will override this
#if defined(FORCE_UNRESTRICTED_FREESTYLE)
    // This is to be used for development purposes only, so that developers can dynamically edit filters and test them in Freestyle and Non-Ansel Integrated games.
    // Printing this as an error multiple times to increase visability.
    for (int i = 0; i < 10; i++) { LOG_ERROR("THIS CODE IS NOT FOR PRODUCTION!!! Forcing all filters to be available in Freestyle, regardless of hash."); }
    freeStyleModeValue = ANSEL_FREESTYLE_MODE_ENABLED;
#elif defined(FORCE_RESTRICTED_FREESTYLE)
    // This is to be used for development purposes only, in order to test restricted freestyle on new games.
    // Printing this as an error multiple times to increase visability.
    for (int i = 0; i < 10; i++) { LOG_ERROR("THIS CODE IS NOT FOR PRODUCTION!!! Forcing Restricted Freestyle."); }
    freeStyleModeValue = ANSEL_FREESTYLE_MODE_APPROVED_ONLY;
#endif

    // Parse the freestyle mode into internal modding states
    parseModdingModes(freeStyleModeValue, &m_moddingSinglePlayer, &m_moddingMultiPlayer);
    m_freeStyleModeValueVerif = freeStyleModeValue;

    // Fill restricted effects sets
    {
        sha256_ctx effSetContext;
        Hash::Data effKey;
        std::wstring effName;
        m_moddingRestrictedSets.resize(static_cast<size_t>(ModdingRestrictedSetID::kNUM_ENTRIES));

        auto & prepackagedEffectsMap = m_moddingRestrictedSets[static_cast<size_t>(ModdingRestrictedSetID::kPrepackaged)];
        auto & mpApprovedEffectsMap = m_moddingRestrictedSets[static_cast<size_t>(ModdingRestrictedSetID::kMPApproved)];

        {
            effName = L"Adjustments.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kAdjustments_yaml);
            prepackagedEffectsMap.insert(effPair);
            //mpApprovedEffectsMap.insert(effPair);
        }

        {
            effName = L"BlacknWhite.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kBlacknWhite_yaml);
            prepackagedEffectsMap.insert(effPair);
            mpApprovedEffectsMap.insert(effPair);
        }

        {
            effName = L"Color.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kColor_yaml);
            prepackagedEffectsMap.insert(effPair);
            mpApprovedEffectsMap.insert(effPair);
        }

        {
            effName = L"Colorblind.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kColorblind_yaml);
            prepackagedEffectsMap.insert(effPair);
            mpApprovedEffectsMap.insert(effPair);
        }

        {
            effName = L"DOF.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kDOF_yaml);
            prepackagedEffectsMap.insert(effPair);
            //mpApprovedEffectsMap.insert(effPair);
        }

        {
            effName = L"Details.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kDetails_yaml);
            prepackagedEffectsMap.insert(effPair);
            //mpApprovedEffectsMap.insert(effPair);
        }

        {
            effName = L"Letterbox.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kLetterbox_yaml);
            prepackagedEffectsMap.insert(effPair);
            mpApprovedEffectsMap.insert(effPair);
        }

        {
            effName = L"NightMode.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kNightMode_yaml);
            prepackagedEffectsMap.insert(effPair);
            mpApprovedEffectsMap.insert(effPair);
        }

        {
            effName = L"OldFilm.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kOldFilm_yaml);
            prepackagedEffectsMap.insert(effPair);
            mpApprovedEffectsMap.insert(effPair);
        }

        {
            effName = L"Painterly.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kPainterly_yaml);
            prepackagedEffectsMap.insert(effPair);
            mpApprovedEffectsMap.insert(effPair);
        }

        {
            effName = L"RemoveHud.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kRemoveHud_yaml);
            prepackagedEffectsMap.insert(effPair);
            mpApprovedEffectsMap.insert(effPair);
        }

        {
            effName = L"Sharpen.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kSharpen_yaml);
            prepackagedEffectsMap.insert(effPair);
            mpApprovedEffectsMap.insert(effPair);
        }

        {
            effName = L"SpecialFX.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kSpecialFX_yaml);
            prepackagedEffectsMap.insert(effPair);
            //mpApprovedEffectsMap.insert(effPair);
        }

        {
            effName = L"Splitscreen.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kSplitscreen_yaml);
            prepackagedEffectsMap.insert(effPair);
            mpApprovedEffectsMap.insert(effPair);
        }

        {
            effName = L"TiltShift.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kTiltShift_yaml);
            prepackagedEffectsMap.insert(effPair);
            mpApprovedEffectsMap.insert(effPair);
        }

        {
            effName = L"Vignette.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kVignette_yaml);
            prepackagedEffectsMap.insert(effPair);
            mpApprovedEffectsMap.insert(effPair);
        }

        {
            effName = L"Watercolor.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kWatercolor_yaml);
            prepackagedEffectsMap.insert(effPair);
            mpApprovedEffectsMap.insert(effPair);
        }

        /*
        // This is being temporarily disabled in Freestyle due to an exploit in Warface.
        // Bug: http://lwbugs/2699079 [FreeStyle][Battlefield V] GreenScreen should no longer be allowlisted for Freestyle.
        {
            effName = L"Stickers.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kStickers_yaml);
            prepackagedEffectsMap.insert(effPair);
            mpApprovedEffectsMap.insert(effPair);
        }

        // This is being temporarily disabled in Freestyle due to an exploit in BFV.
        // Bug: http://lwbugs/2699079 [FreeStyle][Battlefield V] GreenScreen should no longer be allowlisted for Freestyle.
        {
            effName = L"GreenScreen.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kGreenScreen_yaml);
            prepackagedEffectsMap.insert(effPair);
            mpApprovedEffectsMap.insert(effPair);
        }
        */

        {
            effName = L"Testing.yaml";
            sha256_init(&effSetContext);
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t)));
            sha256_final(&effSetContext, effKey.data());

            auto effPair = std::make_pair(effKey, PrepackagedEffects::kTesting_yaml);
            prepackagedEffectsMap.insert(effPair);
            mpApprovedEffectsMap.insert(effPair);
        }

        // ReShade Filters
#define AddReShadeFilterToRestrictedSet(baseName) \
        { \
            effName = L#baseName L".fx"; \
            sha256_init(&effSetContext); \
            sha256_update(&effSetContext, (uint8_t *)effName.c_str(), (uint32_t)(effName.length() * sizeof(wchar_t))); \
            sha256_final(&effSetContext, effKey.data()); \
            \
            auto effPair = std::make_pair(effKey, PrepackagedEffects::k##baseName##_fx); \
            prepackagedEffectsMap.insert(effPair); \
            mpApprovedEffectsMap.insert(effPair); \
        }
        AddReShadeFilterToRestrictedSet(3DFX);
        AddReShadeFilterToRestrictedSet(ASCII);
        AddReShadeFilterToRestrictedSet(AdaptiveSharpen);
        AddReShadeFilterToRestrictedSet(AmbientLight);
        AddReShadeFilterToRestrictedSet(Bloom);
        AddReShadeFilterToRestrictedSet(Border);
        AddReShadeFilterToRestrictedSet(CRT);
        AddReShadeFilterToRestrictedSet(Cartoon);
        AddReShadeFilterToRestrictedSet(ChromaticAberration);
        AddReShadeFilterToRestrictedSet(Clarity);
        AddReShadeFilterToRestrictedSet(Colourfulness);
        AddReShadeFilterToRestrictedSet(Lwrves);
        AddReShadeFilterToRestrictedSet(Daltonize);
        AddReShadeFilterToRestrictedSet(Deband);
        AddReShadeFilterToRestrictedSet(Denoise);
        AddReShadeFilterToRestrictedSet(DenoiseKNN);
        AddReShadeFilterToRestrictedSet(DenoiseNLM);
        AddReShadeFilterToRestrictedSet(FXAA);
        AddReShadeFilterToRestrictedSet(FakeMotionBlur);
        AddReShadeFilterToRestrictedSet(GaussianBlur);
        AddReShadeFilterToRestrictedSet(Glitch);
        AddReShadeFilterToRestrictedSet(HQ4X);
        AddReShadeFilterToRestrictedSet(Monochrome);
        AddReShadeFilterToRestrictedSet(NightVision);
        AddReShadeFilterToRestrictedSet(PPFX_Godrays);
        AddReShadeFilterToRestrictedSet(Prism);
        AddReShadeFilterToRestrictedSet(SMAA);
        AddReShadeFilterToRestrictedSet(Test_Reshade_Hashed);
        AddReShadeFilterToRestrictedSet(TiltShift);
        AddReShadeFilterToRestrictedSet(TriDither);
        AddReShadeFilterToRestrictedSet(Vignette);
    }

    if (pFunctionTable->CopyClientResource12)
    {
        // Initalize these critical sections for DX12.
        InitializeCriticalSection(&m_csExec);
    }

    m_appName = lwanselutils::getAppNameFromProcess();

    HRESULT status = S_OK;

    initRegistryDependentPathsAndOptions();

    LOG_VERBOSE("Device init %x", m_hClient);

    setD3DCompilerPath(getD3dCompilerFullPath().c_str());
    m_D3DCompileFunc = m_d3dCompiler.getD3DCompileFunc();

    PFND3DCOMPILEFROMFILEFUNC pfnD3DCompileFromFile = m_d3dCompiler.getD3DCompileFromFileFunc();
    PFND3DREFLECTFUNC pfnD3DReflect = m_d3dCompiler.getD3DReflectFunc();

    if (!m_D3DCompileFunc || !pfnD3DCompileFromFile || !pfnD3DReflect)
    {
        status = E_FAIL;
        LOG_FATAL("D3D Compiler initialization failed");
        HandleFailureWMessage("D3D Compiler initialization failed. m_D3DCompileFunc = %p, pfnD3DCompileFromFile = %p, pfnD3DReflect = %p, D3D compiler path = %s",
            (void *) m_D3DCompileFunc, (void *) pfnD3DCompileFromFile, (void *) pfnD3DReflect, darkroom::getUtf8FromWstr(m_d3dCompiler.getD3DCompilerPath()).c_str());
    }

#if 0
    if (!SUCCEEDED(status = CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)&pFactory)))
    {
        HandleFailure();
    }

    if (!SUCCEEDED(status = pFactory->EnumAdapters(0, &pAdapter)))
    {
        HandleFailure();
    }

    static const D3D_FEATURE_LEVEL featureLevels[] =
    {
        D3D_FEATURE_LEVEL_11_0,
    };

    // The call to D3D11CreateDevice *must* be done in this function and cannot
    // be deferred to another function or deferred to another thread.  The
    // client sets a thread local flag to avoid initializing another Ansel shim
    // while this function is being called. Deferring the device creation would
    // lead to an infinite loop.
    D3D_FEATURE_LEVEL featureLevel;
#ifdef _DEBUG
    //    UINT flags = D3D11_CREATE_DEVICE_DEBUG;
#else
    UINT flags = 0;
#endif
    UINT numFeatureLevels = sizeof(featureLevels) / sizeof(D3D_FEATURE_LEVEL);
    if (!SUCCEEDED(status = D3D11CreateDevice(pAdapter,
        D3D_DRIVER_TYPE_UNKNOWN,
        NULL,
        flags,
        featureLevels,
        numFeatureLevels,
        D3D11_SDK_VERSION,
        &pDevice,
        &featureLevel,
        &pContext)))
    {
        HandleFailure();
    }
#endif

    if (!initFramework(adapterLUID))
    {
        LOG_FATAL("Error initializing effect framework");
        HandleFailure();
    }

    // We don't need to populate effects list here, as it is done on each session start
    //populateEffectsList();
    // Initialize selected effect to 1 so that ShaderMod could load [needs cleanup]
    m_effectsInfo.m_selectedEffect = 1;

    if (!SUCCEEDED(status = createPassthroughEffect(&m_passthroughEffect)))
    {
        LOG_FATAL("Error creating passthrough effect");
        HandleFailure();
    }

    if (!SUCCEEDED(status = createGridEffect(&m_gridEffect)))
    {
        LOG_FATAL("Error creating passthrough effect");
        HandleFailure();
    }

#if DBG_HARDCODED_EFFECT_BW_ENABLED

    if (!SUCCEEDED(status = CreateBlackAndWhiteEffect(&m_blackAndWhiteEffect)))
    {
        HandleFailure();
    }

#endif

    // get adapter name in pretty form
    DXGI_ADAPTER_DESC temp_desc;

    // determine current device index (we want to run style transfer on the same device)
    IDXGIAdapter* dxgiAdapter = nullptr;
    IDXGIAdapter* lwrrentAdapter = nullptr;
    IDXGIDevice* dxgiDevice = nullptr;
    m_d3dDevice->QueryInterface(__uuidof(IDXGIDevice), (void **)&dxgiDevice);
    dxgiDevice->GetAdapter(&dxgiAdapter);
    dxgiAdapter->GetDesc(&temp_desc);
    dxgiAdapter->Release();
    dxgiDevice->Release();

    m_deviceName = temp_desc.Description;

#ifdef ENABLE_STYLETRANSFER
    const auto lwrrentAdapterLuid = temp_desc.AdapterLuid;

    IDXGIFactory * pFactory = NULL;
    CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&pFactory);
    UINT deviceIndex = 0;
    for (deviceIndex = 0; pFactory->EnumAdapters(deviceIndex, &dxgiAdapter) != DXGI_ERROR_NOT_FOUND; ++deviceIndex)
    {
        dxgiAdapter->GetDesc(&temp_desc);
        if (temp_desc.AdapterLuid.LowPart == lwrrentAdapterLuid.LowPart &&
            temp_desc.AdapterLuid.HighPart == lwrrentAdapterLuid.HighPart)
            break;
    }

    SAFE_RELEASE(pFactory);

    // now that we know deviceIndex, query it's properties
    const lwdaError_t error = lwdaGetDeviceProperties(&m_lwdaDeviceProps, deviceIndex);
    if (error != lwdaSuccess)
    {
        LOG_ERROR("Can't determine device compute capability for style transfer feature");
        m_lwdaDeviceProps.major = 0;
        m_lwdaDeviceProps.minor = 0;
    }

    if (m_lwdaDeviceProps.major < 3)
    {
        LOG_ERROR("Device too old for the style transfer feature. Disabling it.");
        m_isStyleTransferEnabled = false;
    }

    // we only support all consumer hardware up to CC 7.5
    if (m_lwdaDeviceProps.major > 7 || (m_lwdaDeviceProps.major == 7 && m_lwdaDeviceProps.minor > 5))
    {
        LOG_ERROR("Style transfer feature is not supported on this device. Disabling it.");
        m_isStyleTransferEnabled = false;
    }
#endif

    if (getIPCModeEnabled() != 0)
    {
#if IPC_ENABLED == 1
        m_UI = new UIIPC;
        UIIPC* UIipc = static_cast<UIIPC*>(m_UI);
        LOG_INFO("IPC client loaded");

#ifdef ENABLE_STYLETRANSFER
        m_UI->mainSwitchStyleTransfer(m_isStyleTransferEnabled);
#endif
        //feodorb: In my view, a class should ideally subscribe itself for events on creation, and unsubscribe on destruction
        //however, AnselSDKState IS a field of AnselServer, so it is created and destroyed with the AnselServer
        //which, in turn, does the delayed initialization for purely performance reasons. The input handler is UI-dependent,
        //and it is not yet created when the AnselServer is. This is ugly, and this requires subscribing the SDK for the events
        //at the strange places.
        //Q: Can an AnselServer exist without any UI? MY guess is that it can't. So the UI IS in fact a part of the AnselServer,
        //so I guess we should rather have two AnselServers for the two UIs, both inheriting the BaseAnselServer.
        //In order to avoid perf loss when the driver creates an AnselServer many times, we'd have an AnselServerDummy that would
        //only do as much work as needed to create an AnselServerImpl.lServerDummy, that would be created by the driver. keep it as clean as possible.
        //But so far, here's a kludge:
        UIipc->getInputHandler().addEventConsumer(&m_anselSDK);
#else
        m_UI = nullptr;
        LOG_FATAL("IPC is disabled, but IPC codepath is selected");
        assert(false && "IPC is disabled, but IPC codepath is selected!");
#endif
    }
    else
    {
        m_UI = new AnselUI;
        AnselUI * UI_standalone = static_cast<AnselUI *>(m_UI);
        UI_standalone->setForcedLocale(m_forcedLocale);
        UI_standalone->setUseHybridController(m_useHybridController);
        UI_standalone->setStandaloneModding(m_standaloneModding);

#ifdef ENABLE_STYLETRANSFER
        m_UI->mainSwitchStyleTransfer(m_isStyleTransferEnabled);
#endif

        // We cannot handle disabling dynamic filter stacking on the fly, once it was enabled
        //  since it will require some additional logic of cleaning up memory and UI elements
        //  thus, once it was enabled, it will remain enabled for the given process
        if (m_allowDynamicFilterStacking)
        {
            UI_standalone->setDynamicFilterStackingState(m_allowDynamicFilterStacking);
        }

        if (!SUCCEEDED(status = UI_standalone->init(m_hDLLInstance, m_d3dDevice, this, m_installationFolderPath)))
        {
            LOG_FATAL("There was an error in Standalone client initialization");
            HandleFailure();
        }
        LOG_INFO("Standalone client loaded");
        UI_standalone->getInputHandler().addEventConsumer(&m_anselSDK);
    }

    m_UI->setDefaultEffectPath(m_effectInstallationFolderPath.c_str());

    m_UI->updateSettings(m_settingsAsStrings);

    m_globalPerfCounters.timer.Start();
    m_globalPerfCounters.dt = 0.0;
    m_globalPerfCounters.elapsedTime = 0.0;

    m_errorManager.init(20);

    initBufferInterfaces();

    setupColwerter();

    AnselSDKState::DetectionStatus initializationStatus = m_anselSDK.detectAndInitializeAnselSDK(m_installationFolderPath, m_intermediateFolderPath,
        m_snapshotsFolderPath, sessionStartFunc, sessionStopFunc, bufferFinishedFunc, this);
    // Need to initialize SDK here so that games can start sessions:
    if (initializationStatus != AnselSDKState::DetectionStatus::kDRIVER_API_MISMATCH  && !m_lightweightShimMode)
    {
        // We want to enter lightweight shim mode immediately
        // If there are any active filters, lightweight mode will be exited when they are applied,
        // so we don't need to worry about whether or not there are active filters at initialization time
        enterLightweightMode();
    }

#if (ENABLE_DEBUG_RENDERER != 0)

    if (!SUCCEEDED(m_debugRenderer.init(m_d3dDevice, m_immediateContext, m_D3DCompileFunc)))
    {
        m_debugRenderer.deinit();
    }

#endif

#ifdef ENABLE_STYLETRANSFER
    if (m_isStyleTransferEnabled)
    {
        wchar_t path[MAX_PATH] = {0};
        if (SHGetFolderPath(NULL, CSIDL_LOCAL_APPDATA, NULL, SHGFP_TYPE_LWRRENT, path) == S_OK)
            modelRoot = std::wstring(path) + modelRootRelative;
        else
        {
            LOG_ERROR("Failed to resolve local data path");
            m_UI->setStyleTransferStatus(false);
            m_UI->displayMessage(AnselUIBase::MessageType::kStyle_NoModelFound);
        }

        populateStylesList();
        populateStyleNetworksList();
    }
    m_styleSelected = 0;
#endif

    m_specialClientActive = true;
    m_activeControlClient = m_UI;
    m_anselClientsList.push_back(m_UI);

#if (ENABLE_CONTROL_SDK == 1)
    m_anselControlSDK.detectAndInitializeAnselControlSDK();
    m_anselClientsList.push_back(&m_anselControlSDK);
    m_anselControlSDKDetectAndInitializeAttempts = 1;
#endif
    m_modsAvailableQueryTimepoint = std::chrono::steady_clock::now();
    m_sendAnselReady = true;
    m_bInitialized = true;

    return S_OK;
}

HRESULT AnselServer::init(HANSELCLIENT hClient, ClientFunctionTable * pFunctionTable, LARGE_INTEGER adapterLUID)
{
    static bool kaboomHappenned = false;

    if (kaboomHappenned)
        return S_OK;

    const auto exception_filter = [&](LPEXCEPTION_POINTERS exceptionInfo)
    {
        const void* exceptionAddress = exceptionInfo->ExceptionRecord->ExceptionAddress;
        const auto exceptionCode = exceptionInfo->ExceptionRecord->ExceptionCode;
        std::array<void*, 32> frames = { 0 };
        CaptureStackBackTrace(0, DWORD(frames.size()), &frames[0], 0);
        const HMODULE baseAddr = GetModuleHandle(MODULE_NAME_W);
        // Changes to this error string should also be reflected in externals/AnselErrorStringParser
        reportFatalError(__FILE__, __LINE__, FatalErrorCode::kInitFail, "init: top level exception handler exelwted (eCode=%d, eAddr=%p, baseAddr=%p, stacktrace=%p,%p,%p,%p,%p,%p,%p,%p,%p,%p,%p,%p,%p,%p,%p,%p",
            exceptionCode, exceptionAddress, baseAddr,
            frames[0], frames[1], frames[2], frames[3], frames[4], frames[5], frames[6], frames[7],
            frames[8], frames[9], frames[10], frames[11], frames[12], frames[13], frames[14], frames[15]);
        return EXCEPTION_EXELWTE_HANDLER;
    };

#if (DBG_EXCEPTIONS_PASSTHROUGH == 1)
#else
    __try
    {
#endif

        return initCanFail(hClient, pFunctionTable, adapterLUID);

#if (DBG_EXCEPTIONS_PASSTHROUGH == 1)
#else
    }
    __except (exception_filter(GetExceptionInformation()))
    {
        reportFatalError(__FILE__, __LINE__, FatalErrorCode::kInitFail, "init: top level exception handler exelwted");
        kaboomHappenned = true;
        deactivateAnsel(false);
        notifyFrameComplete(true);
    }
#endif



    return S_OK;
}

void AnselServer::handleDrsGameSettings()
{
    // Depth Buffer Settings
    uint32_t depthSettings = ANSEL_BUFFERS_DEPTH_SETTINGS_DEFAULT;
    drs::getKeyValue(ANSEL_BUFFERS_DEPTH_SETTINGS_ID, depthSettings, true);

    std::wstring depthWeights = ANSEL_BUFFERS_DEPTH_WEIGHTS_DEFAULT;
    drs::getKeyValueString(ANSEL_BUFFERS_DEPTH_WEIGHTS_ID, depthWeights, true);

    AnselBufferDepth& depthBuf = m_bufDB.Depth();
    depthBuf.setStatsEn((depthSettings & ANSEL_BUFFERS_DEPTH_SETTINGS_USE_STATS) != 0);
    depthBuf.setViewportChecksEn((depthSettings & ANSEL_BUFFERS_DEPTH_SETTINGS_USE_VIEWPORT) != 0);
    depthBuf.setViewportScalingEn((depthSettings & ANSEL_BUFFERS_DEPTH_SETTINGS_VIEWPORT_SCALING) != 0);
    depthBuf.setWeights(depthWeights);

    // Hudless Buffer Settings
    uint32_t hudlessSettings = ANSEL_BUFFERS_HUDLESS_SETTINGS_DEFAULT;
    drs::getKeyValue(ANSEL_BUFFERS_HUDLESS_SETTINGS_ID, hudlessSettings, true);

    uint32_t hudlessDrawCall = ANSEL_BUFFERS_HUDLESS_DRAWCALL_DEFAULT;
    drs::getKeyValue(ANSEL_BUFFERS_HUDLESS_DRAWCALL_ID, hudlessDrawCall, true);

    std::wstring hudlessWeights = ANSEL_BUFFERS_HUDLESS_WEIGHTS_DEFAULT;
    drs::getKeyValueString(ANSEL_BUFFERS_HUDLESS_WEIGHTS_ID, hudlessWeights, true);

    AnselBufferHudless& hudlessBuf = m_bufDB.Hudless();
    hudlessBuf.setStatsEn((hudlessSettings & ANSEL_BUFFERS_HUDLESS_SETTINGS_USE_STATS) != 0);
    hudlessBuf.setSingleRTV((hudlessSettings & ANSEL_BUFFERS_HUDLESS_SETTINGS_ONLY_SINGLE_RTV_BINDS) != 0);
    hudlessBuf.setRestrictFormats((hudlessSettings & ANSEL_BUFFERS_HUDLESS_SETTINGS_RESTRICT_FORMATS) != 0);
    hudlessBuf.setCompareDrawNum(hudlessDrawCall);
    hudlessBuf.setWeights(hudlessWeights);

    LOG_INFO("DRS Game Setting being applied:");
    LOG_INFO(" - Depth Settings: 0x%x", depthSettings);
    LOG_INFO(" - Hudless Settings: 0x%x", hudlessSettings);
    LOG_INFO(" - Hudless Draw Call: %u", hudlessDrawCall);
}

std::vector<std::wstring> AnselServer::getFilterList() const { return m_effectsInfo.m_effectFilesList; }
uint32_t AnselServer::getScreenWidth() const { return getWidth(); }
uint32_t AnselServer::getScreenHeight() const { return getHeight(); }
uint32_t AnselServer::getMaximumHighresResolution() const { return m_maxHighResResolution; }

namespace passthrough_shaders_ps40
{
#include "shaders/include/passthrough.ps_40.h"
}
namespace passthrough_shaders_vs40
{
#include "shaders/include/passthrough.vs_40.h"
}

HRESULT AnselServer::createPassthroughEffect(AnselEffectState * pOut)
{
    HRESULT status = S_OK;

    D3D11_RASTERIZER_DESC rastStateDesc =
    {
        D3D11_FILL_SOLID,           //FillMode;
        D3D11_LWLL_BACK,            //LwllMode;
        FALSE,                      //FrontCounterClockwise;
        0,                          //DepthBias;
        0.0f,                       //DepthBiasClamp;
        0.0f,                       //SlopeScaledDepthBias;
        TRUE,                       //DepthClipEnable;
        FALSE,                      //ScissorEnable;
        FALSE,                      //MultisampleEnable;
        FALSE                       //AntialiasedLineEnable;
    };

    ID3D11RasterizerState * pRasterizerState;
    if (!SUCCEEDED(status = m_d3dDevice->CreateRasterizerState(&rastStateDesc, &pRasterizerState)))
    {
        LOG_FATAL("Passthrough rasterizer state initialization failed");
        HandleFailure();
    }

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

    ID3D11DepthStencilState * pDepthStencilState;
    if (!SUCCEEDED(status = m_d3dDevice->CreateDepthStencilState(&dsStateDesc, &pDepthStencilState)))
    {
        LOG_FATAL("Passthrough depth/stencil state initialization failed");
        HandleFailure();
    }

    PFND3DCREATEBLOBFUNC pfnD3DCreateBlob = m_d3dCompiler.getD3DCreateBlobFunc();

    // Vertex Shader
    ID3D11VertexShader *pVS = NULL;
    const BYTE * vsByteCode = passthrough_shaders_vs40::g_main;
    size_t vsByteCodeSize = sizeof(passthrough_shaders_vs40::g_main)/sizeof(BYTE);
    if (!SUCCEEDED(status = m_d3dDevice->CreateVertexShader(vsByteCode, vsByteCodeSize, NULL, &pVS)))
    {
        LOG_FATAL("Passthrough v.shader initialization failed");
        HandleFailure();
    }

    // Pixel Shader
    ID3D11PixelShader *pPS = NULL;
    const BYTE * psByteCode = passthrough_shaders_ps40::g_main;
    size_t psByteCodeSize = sizeof(passthrough_shaders_ps40::g_main)/sizeof(BYTE);
    if (!SUCCEEDED(status = m_d3dDevice->CreatePixelShader(psByteCode, psByteCodeSize, NULL, &pPS)))
    {
        LOG_FATAL("Passthrough p.shader initialization failed");
        HandleFailure();
    }

    ID3D11SamplerState * pSamplerState = NULL;
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

    if (!SUCCEEDED(status = m_d3dDevice->CreateSamplerState(&samplerState, &pSamplerState)))
    {
        LOG_FATAL("Passthrough sampler state initialization failed");
        HandleFailure();
    }

    pOut->pVertexShader = pVS;
    pOut->pPixelShader = pPS;
    pOut->pRasterizerState = pRasterizerState;
    pOut->pDepthStencilState = pDepthStencilState;
    pOut->pSamplerState = pSamplerState;

    pOut->pBlendState = NULL;
    return S_OK;
}

namespace grideffect_shaders_ps40
{
#include "shaders/include/grideffect.ps_40.h"
}

HRESULT AnselServer::createGridEffect(AnselEffectState * pOut)
{
    HRESULT status = S_OK;

    // Supposedly shares these states with passthrough effect
    pOut->pRasterizerState = nullptr;
    pOut->pDepthStencilState = nullptr;
    pOut->pVertexShader = nullptr;
    pOut->pSamplerState = nullptr;
    pOut->pBlendState = nullptr;

    PFND3DCREATEBLOBFUNC pfnD3DCreateBlob = m_d3dCompiler.getD3DCreateBlobFunc();

    // Pixel Shader
    ID3D11PixelShader   *pPS = NULL;

    const BYTE * psByteCode = grideffect_shaders_ps40::g_main;
    size_t psByteCodeSize = sizeof(grideffect_shaders_ps40::g_main)/sizeof(BYTE);
    if (!SUCCEEDED(status = m_d3dDevice->CreatePixelShader(psByteCode, psByteCodeSize, NULL, &pPS)))
    {
        LOG_FATAL("Grid effect p.shader initialization failed");
        HandleFailure();
    }

    pOut->pPixelShader = pPS;

    return S_OK;
}

HRESULT AnselServer::createBlackAndWhiteEffect(AnselEffectState * pOut)
{
    D3D11_RASTERIZER_DESC rastStateDesc =
    {
        D3D11_FILL_SOLID,          //FillMode;
        D3D11_LWLL_BACK,           //LwllMode;
        FALSE,                     //FrontCounterClockwise;
        0,                         //DepthBias;
        0.0f,                      //DepthBiasClamp;
        0.0f,                      //SlopeScaledDepthBias;
        TRUE,                      //DepthClipEnable;
        FALSE,                     //ScissorEnable;
        FALSE,                     //MultisampleEnable;
        FALSE                      //AntialiasedLineEnable;
    };

    ID3D11RasterizerState * pRasterizerState;
    HRESULT status = S_OK;
    if (!SUCCEEDED(status = m_d3dDevice->CreateRasterizerState(&rastStateDesc, &pRasterizerState)))
    {
        LOG_FATAL("B&W rasterizer state initialization failed");
        HandleFailure();
    }

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

    ID3D11DepthStencilState * pDepthStencilState;
    if (!SUCCEEDED(status = m_d3dDevice->CreateDepthStencilState(&dsStateDesc, &pDepthStencilState)))
    {
        LOG_FATAL("B&W depth/stencil state initialization failed");
        HandleFailure();
    }

    // Vertex Shader
    ID3D11VertexShader          *pVS = NULL;
    ID3D10Blob                  *pVSBlob = NULL;
    ID3D10Blob                  *pVSBlobErrors = NULL;

    const char vsText[] =
        "struct Output                                                                                  \n"
        "{                                                                                              \n"
        "   float4 position_cs : SV_POSITION;                                                           \n"
        "   float2 texcoord : TEXCOORD;                                                                 \n"
        "};                                                                                             \n"
        "                                                                                               \n"
        "Output Main(uint id: SV_VertexID)                                                              \n"
        "{                                                                                              \n"
        "   Output output;                                                                              \n"
        "                                                                                               \n"
        "   output.texcoord = float2((id << 1) & 2, id & 2);                                            \n"
        "   output.position_cs = float4(output.texcoord * float2(2, -2) + float2(-1, 1), 0, 1);         \n"
        "                                                                                               \n"
        "   return output;                                                                              \n"
        "}                                                                                              \n";

    if (!SUCCEEDED(status = m_D3DCompileFunc(vsText, sizeof(vsText) - 1, NULL, NULL, NULL, "Main", "vs_4_0", 0, 0, &pVSBlob, NULL)))
    {
        char * error = (char *)pVSBlobErrors->GetBufferPointer();
        LOG_FATAL("B&W v.shader compilation failed: %s", error);
        HandleFailure();
    }

    if (!SUCCEEDED(status = m_d3dDevice->CreateVertexShader(pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), NULL, &pVS)))
    {
        LOG_FATAL("B&W v.shader initialization failed");
        HandleFailure();
    }

    if (pVSBlob) pVSBlob->Release();
    if (pVSBlobErrors) pVSBlobErrors->Release();

    // Pixel Shader
    ID3D11PixelShader   *pPS = NULL;
    ID3D10Blob          *pPSBlob = NULL;
    ID3D10Blob          *pPSBlobErrors = NULL;
#if (ENABLE_SEPIA_FOR_BW == 0)
    const char psText[] =
        "struct VSOut                                                                     \n"
        "{                                                                                \n"
        "    float4 position : SV_Position;                                               \n"
        "    float2 texcoord: TexCoord;                                                   \n"
        "};                                                                               \n"
        "                                                                                 \n"
        "Texture2D txDiffuse : register( t0 );                                            \n"
        "SamplerState samLinear : register( s0 );                                         \n"
        "                                                                                 \n"
        "float4 PS( VSOut frag ): SV_Target                                               \n"
        "{                                                                                \n"
        "    float4 clr = txDiffuse.Sample(samLinear, frag.texcoord);                     \n"
        "                                                                                 \n"
        "    const float4 lumFilter = { 0.2126, 0.7152, 0.0722, 0.0 };                    \n"
        "                                                                                 \n"
        "    return float4( dot(clr, lumFilter).xxxx );                                   \n"
        "}                                                                                \n";
#else
    const char psText[] =
        "struct VSOut                                                                     \n"
        "{                                                                                \n"
        "    float4 position : SV_Position;                                               \n"
        "    float2 texcoord: TexCoord;                                                   \n"
        "};                                                                               \n"
        "                                                                                 \n"
        "Texture2D txDiffuse : register( t0 );                                            \n"
        "SamplerState samLinear : register( s0 );                                         \n"
        "                                                                                 \n"
        "float4 PS( VSOut frag ): SV_Target                                               \n"
        "{                                                                                \n"
        "    float4 clr = txDiffuse.Sample(samLinear, frag.texcoord);                     \n"
        "                                                                                 \n"
        "    const float4 sepiaLight = { 1.2, 1.0, 0.8, 1.0 };                            \n"
        "    const float4 lumFilter = { 0.2126, 0.7152, 0.0722, 0.0 };                    \n"
        "                                                                                 \n"
        "    return float4( dot(clr, lumFilter).xxxx * sepiaLight );                      \n"
        "}                                                                                \n";
#endif

    if (!SUCCEEDED(status = m_D3DCompileFunc(psText, sizeof(psText) - 1, NULL, NULL, NULL, "PS", "ps_4_0", 0, 0, &pPSBlob, &pPSBlobErrors)))
    {
        char * error = (char *)pPSBlobErrors->GetBufferPointer();
        LOG_FATAL("B&W p.shader compilation failed: %s", error);
        HandleFailure();
    }
    if (!SUCCEEDED(status = m_d3dDevice->CreatePixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, &pPS)))
    {
        LOG_FATAL("B&W p.shader initialization failed");
        HandleFailure();
    }

    if (pPSBlob) pPSBlob->Release();
    if (pPSBlobErrors) pPSBlobErrors->Release();

    ID3D11SamplerState * pSamplerState = NULL;
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

    if (!SUCCEEDED(status = m_d3dDevice->CreateSamplerState(&samplerState, &pSamplerState)))
    {
        LOG_FATAL("Passthrough sampler state initialization failed");
        HandleFailure();
    }

    pOut->pVertexShader = pVS;
    pOut->pPixelShader = pPS;
    pOut->pRasterizerState = pRasterizerState;
    pOut->pDepthStencilState = pDepthStencilState;
    pOut->pSamplerState = pSamplerState;

    pOut->pBlendState = NULL;
    return S_OK;
}

HRESULT AnselServer::createDepthRenderEffect(AnselEffectState * pOut)
{
    D3D11_RASTERIZER_DESC rastStateDesc =
    {
        D3D11_FILL_SOLID,          //FillMode;
        D3D11_LWLL_BACK,           //LwllMode;
        FALSE,                     //FrontCounterClockwise;
        0,                         //DepthBias;
        0.0f,                      //DepthBiasClamp;
        0.0f,                      //SlopeScaledDepthBias;
        TRUE,                      //DepthClipEnable;
        FALSE,                     //ScissorEnable;
        FALSE,                     //MultisampleEnable;
        FALSE                      //AntialiasedLineEnable;
    };

    ID3D11RasterizerState * pRasterizerState;
    HRESULT status = S_OK;
    if (!SUCCEEDED(status = m_d3dDevice->CreateRasterizerState(&rastStateDesc, &pRasterizerState)))
    {
        HandleFailure();
    }

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

    ID3D11DepthStencilState * pDepthStencilState;
    if (!SUCCEEDED(status = m_d3dDevice->CreateDepthStencilState(&dsStateDesc, &pDepthStencilState)))
    {
        HandleFailure();
    }

    // Vertex Shader
    ID3D11VertexShader          *pVS = NULL;
    ID3D10Blob                  *pVSBlob = NULL;
    ID3D10Blob                  *pVSBlobErrors = NULL;

    const char vsText[] =
        "struct Output                                                                                  \n"
        "{                                                                                              \n"
        "   float4 position_cs : SV_POSITION;                                                           \n"
        "   float2 texcoord : TEXCOORD;                                                                 \n"
        "};                                                                                             \n"
        "                                                                                               \n"
        "Output Main(uint id: SV_VertexID)                                                              \n"
        "{                                                                                              \n"
        "   Output output;                                                                              \n"
        "                                                                                               \n"
        "   output.texcoord = float2((id << 1) & 2, id & 2);                                            \n"
        "   output.position_cs = float4(output.texcoord * float2(2, -2) + float2(-1, 1), 0, 1);         \n"
        "                                                                                               \n"
        "   return output;                                                                              \n"
        "}                                                                                              \n";

    if (!SUCCEEDED(status = m_D3DCompileFunc(vsText, sizeof(vsText) - 1, NULL, NULL, NULL, "Main", "vs_4_0", 0, 0, &pVSBlob, NULL)))
    {
        char * error = (char *)pVSBlobErrors->GetBufferPointer();
        HandleFailure();
    }

    if (!SUCCEEDED(status = m_d3dDevice->CreateVertexShader(pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), NULL, &pVS)))
    {
        HandleFailure();
    }

    if (pVSBlob) pVSBlob->Release();
    if (pVSBlobErrors) pVSBlobErrors->Release();

    // Pixel Shader
    ID3D11PixelShader   *pPS = NULL;
    ID3D10Blob          *pPSBlob = NULL;
    ID3D10Blob          *pPSBlobErrors = NULL;

    char * psText = NULL;

    psText =
        "struct VSOut                                                                     \n"
        "{                                                                                \n"
        "    float4 position : SV_Position;                                               \n"
        "    float2 texcoord: TexCoord;                                                   \n"
        "};                                                                               \n"
        "                                                                                 \n"
        "Texture2D txDepth : register( t0 );                                              \n"
        "SamplerState samLinear : register( s0 );                                         \n"
        "                                                                                 \n"
        "float4 PS( VSOut frag ): SV_Target                                               \n"
        "{                                                                                \n"
        "    float depth = txDepth.Sample(samLinear, frag.texcoord).r;                    \n"
        "    depth = (depth - 0.9f) * 10.0f;                                              \n"
        "    depth = 1.0f - depth;                                                        \n"
        "    float4 clr = float4(0.0f, depth, 0.0f, 0.5f);                                \n"
        "                                                                                 \n"
        "    return clr;                                                                  \n"
        "}                                                                                \n";

    if (!SUCCEEDED(status = m_D3DCompileFunc(psText, strlen(psText), NULL, NULL, NULL, "PS", "ps_4_0", 0, 0, &pPSBlob, &pPSBlobErrors)))
    {
        char * error = (char *)pPSBlobErrors->GetBufferPointer();
        HandleFailure();
    }
    if (!SUCCEEDED(status = m_d3dDevice->CreatePixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, &pPS)))
    {
        HandleFailure();
    }

    if (pPSBlob) pPSBlob->Release();
    if (pPSBlobErrors) pPSBlobErrors->Release();

    ID3D11SamplerState * pSamplerState = NULL;
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

    if (!SUCCEEDED(status = m_d3dDevice->CreateSamplerState(&samplerState, &pSamplerState)))
    {
        HandleFailure();
    }

    D3D11_BLEND_DESC blendStateDesc;
    memset(&blendStateDesc, 0, sizeof(blendStateDesc));
    blendStateDesc.AlphaToCoverageEnable = FALSE;
    blendStateDesc.IndependentBlendEnable = FALSE;
    blendStateDesc.RenderTarget[0].BlendEnable = TRUE;
    blendStateDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_COLOR;
    blendStateDesc.RenderTarget[0].DestBlend = D3D11_BLEND_DEST_COLOR;
    blendStateDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blendStateDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_SRC_ALPHA;
    blendStateDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ILW_SRC_ALPHA;
    blendStateDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blendStateDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

    ID3D11BlendState * pBlendState = NULL;
    if (!SUCCEEDED(status = m_d3dDevice->CreateBlendState(&blendStateDesc, &pBlendState)))
    {
        HandleFailure();
    }

    pOut->pVertexShader = pVS;
    pOut->pPixelShader = pPS;
    pOut->pRasterizerState = pRasterizerState;
    pOut->pDepthStencilState = pDepthStencilState;
    pOut->pSamplerState = pSamplerState;

    pOut->pBlendState = pBlendState;
    return S_OK;
}

HRESULT AnselServer::createDepthRenderRGBEffect(AnselEffectState * pOut)
{
    D3D11_RASTERIZER_DESC rastStateDesc =
    {
        D3D11_FILL_SOLID,          //FillMode;
        D3D11_LWLL_BACK,           //LwllMode;
        FALSE,                     //FrontCounterClockwise;
        0,                         //DepthBias;
        0.0f,                      //DepthBiasClamp;
        0.0f,                      //SlopeScaledDepthBias;
        TRUE,                      //DepthClipEnable;
        FALSE,                     //ScissorEnable;
        FALSE,                     //MultisampleEnable;
        FALSE                      //AntialiasedLineEnable;
    };

    ID3D11RasterizerState * pRasterizerState;
    HRESULT status = S_OK;
    if (!SUCCEEDED(status = m_d3dDevice->CreateRasterizerState(&rastStateDesc, &pRasterizerState)))
    {
        HandleFailure();
    }

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

    ID3D11DepthStencilState * pDepthStencilState;
    if (!SUCCEEDED(status = m_d3dDevice->CreateDepthStencilState(&dsStateDesc, &pDepthStencilState)))
    {
        HandleFailure();
    }

    // Vertex Shader
    ID3D11VertexShader          *pVS = NULL;
    ID3D10Blob                  *pVSBlob = NULL;
    ID3D10Blob                  *pVSBlobErrors = NULL;

    const char vsText[] =
        "struct Output                                                                                  \n"
        "{                                                                                              \n"
        "   float4 position_cs : SV_POSITION;                                                           \n"
        "   float2 texcoord : TEXCOORD;                                                                 \n"
        "};                                                                                             \n"
        "                                                                                               \n"
        "Output Main(uint id: SV_VertexID)                                                              \n"
        "{                                                                                              \n"
        "   Output output;                                                                              \n"
        "                                                                                               \n"
        "   output.texcoord = float2((id << 1) & 2, id & 2);                                            \n"
        "   output.position_cs = float4(output.texcoord * float2(2, -2) + float2(-1, 1), 0, 1);         \n"
        "                                                                                               \n"
        "   return output;                                                                              \n"
        "}                                                                                              \n";

    if (!SUCCEEDED(status = m_D3DCompileFunc(vsText, sizeof(vsText) - 1, NULL, NULL, NULL, "Main", "vs_4_0", 0, 0, &pVSBlob, NULL)))
    {
        char * error = (char *)pVSBlobErrors->GetBufferPointer();
        HandleFailure();
    }

    if (!SUCCEEDED(status = m_d3dDevice->CreateVertexShader(pVSBlob->GetBufferPointer(), pVSBlob->GetBufferSize(), NULL, &pVS)))
    {
        HandleFailure();
    }

    if (pVSBlob) pVSBlob->Release();
    if (pVSBlobErrors) pVSBlobErrors->Release();

    // Pixel Shader
    ID3D11PixelShader   *pPS = NULL;
    ID3D10Blob          *pPSBlob = NULL;
    ID3D10Blob          *pPSBlobErrors = NULL;

    char * psText = NULL;

    psText =
        "struct VSOut                                                                     \n"
        "{                                                                                \n"
        "    float4 position : SV_Position;                                               \n"
        "    float2 texcoord: TexCoord;                                                   \n"
        "};                                                                               \n"
        "                                                                                 \n"
        "Texture2D txDepth : register( t0 );                                              \n"
        "SamplerState samLinear : register( s0 );                                         \n"
        "                                                                                 \n"
        "float4 PS( VSOut frag ): SV_Target                                               \n"
        "{                                                                                \n"
        "    float4 depth = txDepth.Sample(samLinear, frag.texcoord);                     \n"
        "    float4 clr = float4(1.0f - depth.r, 1.0f - depth.g, 1.0f - depth.b, 0.5f);   \n"
        "                                                                                 \n"
        "    return clr;                                                                  \n"
        "}                                                                                \n";

    if (!SUCCEEDED(status = m_D3DCompileFunc(psText, strlen(psText), NULL, NULL, NULL, "PS", "ps_4_0", 0, 0, &pPSBlob, &pPSBlobErrors)))
    {
        char * error = (char *)pPSBlobErrors->GetBufferPointer();
        HandleFailure();
    }
    if (!SUCCEEDED(status = m_d3dDevice->CreatePixelShader(pPSBlob->GetBufferPointer(), pPSBlob->GetBufferSize(), NULL, &pPS)))
    {
        HandleFailure();
    }

    if (pPSBlob) pPSBlob->Release();
    if (pPSBlobErrors) pPSBlobErrors->Release();

    ID3D11SamplerState * pSamplerState = NULL;
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

    if (!SUCCEEDED(status = m_d3dDevice->CreateSamplerState(&samplerState, &pSamplerState)))
    {
        HandleFailure();
    }

    D3D11_BLEND_DESC blendStateDesc;
    memset(&blendStateDesc, 0, sizeof(blendStateDesc));
    blendStateDesc.AlphaToCoverageEnable = FALSE;
    blendStateDesc.IndependentBlendEnable = FALSE;
    blendStateDesc.RenderTarget[0].BlendEnable = TRUE;
    blendStateDesc.RenderTarget[0].SrcBlend = D3D11_BLEND_SRC_COLOR;
    blendStateDesc.RenderTarget[0].DestBlend = D3D11_BLEND_DEST_COLOR;
    blendStateDesc.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
    blendStateDesc.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_SRC_ALPHA;
    blendStateDesc.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_ILW_SRC_ALPHA;
    blendStateDesc.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
    blendStateDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;

    ID3D11BlendState * pBlendState = NULL;
    if (!SUCCEEDED(status = m_d3dDevice->CreateBlendState(&blendStateDesc, &pBlendState)))
    {
        HandleFailure();
    }

    pOut->pVertexShader = pVS;
    pOut->pPixelShader = pPS;
    pOut->pRasterizerState = pRasterizerState;
    pOut->pDepthStencilState = pDepthStencilState;
    pOut->pSamplerState = pSamplerState;

    pOut->pBlendState = pBlendState;
    return S_OK;
}

HRESULT AnselServer::destroy()
{
    LOG_VERBOSE("Shutdown Lwbin API");
    ShutdownLwbin();

    LOG_VERBOSE("Device destroy %x", m_hClient);

#ifdef ENABLE_STYLETRANSFER
    SAFE_RELEASE(m_styleTransferOutputBuffer.tex);
    SAFE_RELEASE(m_styleTransferOutputHDRStorageBuffer.tex);

    if (restyleDeinitializeFunc)
        restyleDeinitializeFunc();
#endif
#if (ENABLE_DEBUG_RENDERER != 0)
    m_debugRenderer.deinit();
#endif

    // In case we haven't had the opportunity
    // to close the session on finalize frame.
    m_anselSDK.stopSession();

    m_effectsInfo.m_effectFilesList.clear();
    m_effectsInfo.m_effectRootFoldersList.clear();
    m_localizedEffectNamesParser.reset();

    for (std::map<HCLIENTRESOURCE, AnselResource*>::iterator it = m_handleToAnselResource.begin(); it != m_handleToAnselResource.end(); ++it)
    {
        AnselResource * pAnselResource = it->second;

        if (pAnselResource)
        {
            releaseAnselResource(pAnselResource);
            delete pAnselResource;
        }
    }
    m_handleToAnselResource.clear();

    m_bufDB.destroy();

    m_renderBufferColwerter.deinit();

    if (m_UI)
    {
        m_UI->getInputHandler().removeEventConsumer(&m_anselSDK);
        SAFE_DELETE(m_UI);
    }

#if DBG_HARDCODED_EFFECT_BW_ENABLED
    releaseEffectState(&m_blackAndWhiteEffect);
#endif

    releaseEffectState(&m_passthroughEffect);
    releaseEffectState(&m_gridEffect);

    SAFE_RELEASE(m_readbackTexture);
    SAFE_RELEASE(m_resolvedPrereadbackTexture);

    if (m_readbackData)
    {
        delete[] m_readbackData;
        m_readbackData = nullptr;
    }

    destroyAllEffectsInStack();
    destroyDevice();

    if (m_pClientFunctionTable->CopyClientResource12)
    {
        DeleteCriticalSection(&m_csExec);
    }

    destroyTelemetry();

    ANSEL_PROFILE_DEINIT();

    m_bInitialized = false;

    LOG_DEBUG("Device destroy complete.");

    return S_OK;
}

HRESULT AnselServer::notifyDraw()
{
    if (m_lightweightShimMode)
    {
        LOG_ERROR("Draw notification in LW mode");
        return S_OK;
    }

    if (!m_lwrrentBufferInterface)
    {
        setBufferInterfaceD3D11();
    }

    HRESULT status = S_OK;

    if (!SUCCEEDED(status = m_pClientFunctionTable->GetDeviceStates(m_hClient, &m_deviceStates)))
    {
        HandleFailure();
    }

    // Run checks to see if the buffer that is lwrrently bound is believed to be a depth buffer.
    // Only perform these checks if there is a filter enabled that requires depth
    AnselBufferDepth& depthBuf = m_bufDB.Depth();
    if (m_enableDepthExtractionChecks && m_depthBufferUsed && !depthBuf.isForced() && depthBuf.getClientResource() == nullptr && m_d3d11Interface.m_lwrrentDepthBuffer != nullptr)
    {
        bool setViewport = false;
        if (depthBuf.useStats())
        {
            // Check if we've already selected a buffer as part of a resolve in a previous frame, and select it so
            // that any subsequent notifyDepthStencilClear calls and finalizeFrame will correctly copy this depth buffer
            if (depthBuf.compareAgainstSelBuf(m_d3d11Interface.m_lwrrentDepthBuffer))
            {
                depthBuf.selectBuffer(m_d3d11Interface.m_lwrrentDepthBuffer);
                setViewport = depthBuf.useViewportScaling();
            }

            // If we haven't yet selected a depth buffer, addStats will capture device state on this draw
            // call which will be used to determine if this buffer is a depth buffer
            depthBuf.addStats(m_d3d11Interface.m_lwrrentDepthBuffer, m_deviceStates);
        }
        else
        {
            bool depthBufferMatch;
            depthBuf.checkBuffer(m_d3d11Interface.m_lwrrentDepthBuffer, m_deviceStates, getDepthWidth(), getDepthHeight(), &depthBufferMatch);
            if (depthBufferMatch)
            {
                depthBuf.selectBuffer(m_d3d11Interface.m_lwrrentDepthBuffer);
                setViewport = depthBuf.useViewportScaling();
            }
        }

        if (setViewport)
        {
            m_renderBufferColwerter.setPSConstBufDataViewport(m_deviceStates.ViewportWidth, m_deviceStates.ViewportHeight);
        }
    }

    AnselBufferHudless& hudlessBuf = m_bufDB.Hudless();
    if (m_enableHUDlessExtractionChecks && m_hudlessBufferUsed)
    {
        if (hudlessBuf.useStats())
        {
            // Track stats for every buffer we've associated as hudless
            hudlessBuf.addStats(m_d3d11Interface.m_lwrrentHUDless, m_deviceStates);

            // Check if this draw call is the one where we should grab a copy of the buffer prior to HUD being
            // drawn, then perform the copy now
            hudlessBuf.copyHudless(m_d3d11Interface.m_lwrrentHUDless);
        }
        else
        {
            // We're detecting the first HUD element here, if found - that means we need to copy our current HUDless right now
            if (AnselBuffer::isHUDlessColor(m_deviceStates))
            {
                if (!hudlessBuf.isForced())
                {
                    bool hudlessBufferMatch;
                    hudlessBuf.checkBuffer(m_d3d11Interface.m_lwrrentHUDless, m_deviceStates, &hudlessBufferMatch);

                    // TODO: for hints, we need to figure out an automatic way of determining when to copy HUDless, or
                    //  do this strictly only when the finalization was hinted. In case of automatic way - there could possibly
                    //  be 2 copies, one automatic, and then another triggered by the app. Need a way to resolve that (e.g.
                    //  disable auto when the double-copying happened, or finalize hint encountered).

                    // We're copying the HUDless just before the first HUD element will be rendered
                    if (hudlessBufferMatch)
                    {
                        hudlessBuf.selectBuffer(m_d3d11Interface.m_lwrrentHUDless);
                        hudlessBuf.copyResource(0);
                    }
                }
                else
                {
                    hudlessBuf.copyResource(0);
                }
            }
        }
    }

    return status;
}

HRESULT AnselServer::notifyDepthStencilCreate(HCLIENTRESOURCE hClientResource)
{
    if (m_lightweightShimMode)
    {
        LOG_ERROR("DepthStencilCreate notification in LW mode");
        return S_OK;
    }

    if (!m_lwrrentBufferInterface)
    {
        setBufferInterfaceD3D11();
    }

    LOG_DEBUG("notify CreateDepthStencilView for 0x%0x", hClientResource);

    //// In reference to http://lwbugs/2515076
    //// There was a memory leak that oclwrred when CreateDepthStencilView was called each frame. This reset the server's
    //// hClientResource to Ansel Resource mapping for that resource, causing LwCamera to generate a new Ansel resource for
    //// it every single frame without cleaning up the old one. Skipping this mapping reset WARs the memory leak. However,
    //// releasing the Ansel resource here triggers creation of a new Ansel resource at the next finalize frame call, and
    //// triggers the memory leak again, and it is unknown what is leaking in this case. So we are leaving this to just skip
    //// the mapping reset until further ilwestigation is required.
    // eraseClientResourceHandleIfFound(hClientResource, m_pBuffers);

    return S_OK;
}

HRESULT AnselServer::notifyDepthStencilBind(HCLIENTRESOURCE hClientResource)
{
    if (!m_lwrrentBufferInterface)
    {
        setBufferInterfaceD3D11();
    }

    uint64_t threadId = 0;
    ansel::HintType hintType = ansel::kHintTypePreBind;
    if (m_anselSDK.isBufferBindHintActive(ansel::kBufferTypeDepth, threadId, hintType) && m_depthBufferUsed)
    {
        if (threadId == UINT64_MAX || threadId == GetLwrrentThreadId())
        {
            HCLIENTRESOURCE depthResourceToUse = hintType == ansel::kHintTypePreBind ? hClientResource : m_lastBoundDepthResource;
            m_bufDB.Depth().setForced(depthResourceToUse);
            m_anselSDK.clearBufferBindHint(ansel::kBufferTypeDepth);
            m_wasDepthHintUsed = true;
        }
    }
    m_lastBoundDepthResource = hClientResource;

    if (m_lightweightShimMode)
    {
        return S_OK;
    }

    if (!m_bufDB.Depth().isForced())
    {
        m_d3d11Interface.m_lwrrentDepthBuffer = hClientResource;

        // Mark this buffer as bound in the depth buffer tracker
        m_bufDB.Depth().bufBound(hClientResource);
    }
    return S_OK;
}

HRESULT AnselServer::bufferFinished(ansel::BufferType bufferType, uint64_t threadId)
{
    if (threadId == UINT64_MAX || threadId == GetLwrrentThreadId())
    {
        // This callback could be called early, before we even initialized anything
        // in this case, we just ignore this callback
        if (!m_lwrrentBufferInterface)
        {
            return S_OK;
        }

        switch (bufferType)
        {
        case ansel::BufferType::kBufferTypeDepth:
            {
                // wasBufferForced is checked to work around an issue with bufferFinished being called by a DX12 app, which does not support Buffer Hints yet.
                if (m_enableDepthExtractionChecks && m_bufDB.Depth().isForced())
                {
                    m_bufDB.Depth().copyResource(0);
                }
                break;
            }
        case ansel::BufferType::kBufferTypeHDR:
            {
                if (m_enableHDRExtractionChecks && m_bufDB.HDR().isForced())
                {
                    m_bufDB.HDR().copyResource(0);
                }
                break;
            }
        case ansel::BufferType::kBufferTypeHUDless:
            {
                if (m_enableHUDlessExtractionChecks && m_bufDB.Hudless().isForced())
                {
                    m_bufDB.Hudless().copyResource(0);
                }
                break;
            }
        case ansel::BufferType::kBufferTypeFinalColor:
            {
                if (m_enableFinalColorExtractionChecks && m_bufDB.Final().isForced())
                {
                    m_bufDB.Final().copyResource(0);
                }
                break;
            }
        default:
            {
                return E_ILWALIDARG;
            }
        }
    }
    else
    {
        return E_ACCESSDENIED;
    }

    return S_OK;
}

HRESULT AnselServer::checkHDRHints(HCLIENTRESOURCE hdrResource, const AnselClientResourceInfo & resourceInfo)
{
    // this is just a straight hint-based case
    uint64_t threadId = 0;
    ansel::HintType hintType = ansel::kHintTypePreBind;
    if (m_anselSDK.isBufferBindHintActive(ansel::kBufferTypeHDR, threadId, hintType) && isHDRBufferUsed())
    {
        if (threadId == UINT64_MAX || threadId == GetLwrrentThreadId())
        {
            DXGI_FORMAT colwertedFormat = lwanselutils::colwertFromTypelessIfNeeded(DXGI_FORMAT(resourceInfo.Format));
            if (!isHdrFormatSupported( colwertedFormat ))
            {
                // In case of hints, we can also tolerate RGBA10_UNORM
                if (colwertedFormat != DXGI_FORMAT_R10G10B10A2_UNORM)
                {
                    // WARNING: there is a risk that this log will be too excessive - e.g. each frame
                    LOG_VERBOSE("HDR hint was specified, but the render target has incompatible format (%d)", resourceInfo.Format);
                }
            }

            HCLIENTRESOURCE hdrResourceToUse = hintType == ansel::kHintTypePreBind ? hdrResource : m_lastBoundColorResource;

            m_bufDB.HDR().setForced(hdrResourceToUse);

            m_anselSDK.clearBufferBindHint(ansel::kBufferTypeHDR);
            m_wasHDRHintUsed = true;
        }
    }

    return S_OK;
}

HRESULT AnselServer::checkHUDlessHints(HCLIENTRESOURCE hudlessResource, const AnselClientResourceInfo & resourceInfo)
{
    // this is just a straight hint-based case
    uint64_t threadId = 0;
    ansel::HintType hintType = ansel::kHintTypePreBind;
    if (m_anselSDK.isBufferBindHintActive(ansel::kBufferTypeHUDless, threadId, hintType) && m_hudlessBufferUsed)
    {
        if (threadId == UINT64_MAX || threadId == GetLwrrentThreadId())
        {
            HCLIENTRESOURCE hudlessResourceToUse = hintType == ansel::kHintTypePreBind ? hudlessResource : m_lastBoundColorResource;

            m_bufDB.Hudless().setForced(hudlessResourceToUse);

            m_anselSDK.clearBufferBindHint(ansel::kBufferTypeHUDless);
            m_wasHUDlessHintUsed = true;
        }
    }

    return S_OK;
}

HRESULT AnselServer::checkFinalColorHints(HCLIENTRESOURCE colorResource, const AnselClientResourceInfo & resourceInfo)
{
    // this is just a straight hint-based case
    uint64_t threadId = 0;
    ansel::HintType hintType = ansel::kHintTypePreBind;
    if (m_anselSDK.isBufferBindHintActive(ansel::kBufferTypeFinalColor, threadId, hintType))
    {
        if (threadId == UINT64_MAX || threadId == GetLwrrentThreadId())
        {
            HCLIENTRESOURCE colorResourceToUse = hintType == ansel::kHintTypePreBind ? colorResource : m_lastBoundColorResource;

            m_bufDB.Final().setForced(colorResourceToUse);

            m_anselSDK.clearBufferBindHint(ansel::kBufferTypeFinalColor);
            m_wasFinalColorHintUsed = true;
        }
    }

    return S_OK;
}

HRESULT AnselServer::notifyRenderTargetBind(HCLIENTRESOURCE* phClientResource, DWORD dwNumRTs)
{
    if (!m_lwrrentBufferInterface)
    {
        setBufferInterfaceD3D11();
    }

    HRESULT status = S_OK;
    if (dwNumRTs == 0)
    {
        if (m_bufDB.Hudless().useStats())
        {
            m_d3d11Interface.m_lwrrentHUDless = nullptr;
        }
        return status;
    }

    AnselClientResourceInfo resourceInfo;
    HCLIENTRESOURCE rtResource = phClientResource[0];
    if (!SUCCEEDED(status = m_pClientFunctionTable->GetClientResourceInfo(m_hClient, rtResource, &resourceInfo)))
    {
        if (m_bufDB.Hudless().useStats())
        {
            m_d3d11Interface.m_lwrrentHUDless = nullptr;
        }
        return S_OK;
    }

    checkHDRHints(rtResource, resourceInfo);
    checkHUDlessHints(rtResource, resourceInfo);
    checkFinalColorHints(rtResource, resourceInfo);
    m_lastBoundColorResource = rtResource;

    if (m_lightweightShimMode)
    {
        // We went through the hinting, and if we're in the LW mode, we are not allowed to go further (i.e. apply any heuristic)
        return S_OK;
    }

    AnselBufferHDR& hdrBuf = m_bufDB.HDR();
    AnselBufferHudless& hudlessBuf = m_bufDB.Hudless();

    bool wasHDRBufSelected = false;
    if (m_enableHDRExtractionChecks)
    {
        // Also checks whether the HDR buffer was forced
        hdrBuf.checkBuffer(rtResource, getWidth(), getHeight(), &wasHDRBufSelected);
    }

    if (wasHDRBufSelected)
    {
        m_hdrBufferAvailable = true;
        if (isHDRBufferUsed())
        {
            hdrBuf.selectBuffer(rtResource);
        }
    }
    if (!hudlessBuf.isForced())
    {
        // Old mechanism
        if (!hudlessBuf.useStats())
        {
            m_d3d11Interface.m_lwrrentHUDless = rtResource;
            return status;
        }

        bool setHudlessBindBuffer = true;
        if (hudlessBuf.useSingleRTV() && dwNumRTs != 1)
        {
            setHudlessBindBuffer = false;
        }
        if (hudlessBuf.useRestrictFormats() && !hudlessBuf.isFormatSupported(resourceInfo.Format))
        {
            setHudlessBindBuffer = false;
        }

        if (setHudlessBindBuffer)
        {
            m_d3d11Interface.m_lwrrentHUDless = rtResource;
            hudlessBuf.bufBound(rtResource);
        }
        else
        {
            m_d3d11Interface.m_lwrrentHUDless = nullptr;
        }
    }

    return status;
}

HRESULT AnselServer::notifyUnorderedAccessBind(DWORD startOffset, DWORD count, HCLIENTRESOURCE* phClientResource)
{
    if (m_lightweightShimMode)
    {
        LOG_ERROR("UnorderedAccessBind notification in LW mode");
        return S_OK;
    }

    if (!m_lwrrentBufferInterface)
    {
        setBufferInterfaceD3D11();
    }

    if (!m_bufDB.Hudless().useStats() && startOffset == 0 && count)
    {
        m_d3d11Interface.m_lwrrentHUDless = nullptr;
    }

    return S_OK;
}

HRESULT AnselServer::notifyDepthStencilDestroy(HCLIENTRESOURCE hClientResource)
{
    if (!m_lwrrentBufferInterface)
    {
        setBufferInterfaceD3D11();
    }

    // The driver never actually calls this to notify the Ansel server of Depth Stencil Destruction.
    LOG_DEBUG("notify DepthStencilDestroy for 0x%0x", hClientResource);

    m_bufDB.Depth().removeBuf(hClientResource);

    return S_OK;
}

void AnselServer::releaseAnselResource(AnselResource * pAnselResource)
{
    SAFE_RELEASE(pAnselResource->toServerRes.pTexture2D);
    SAFE_RELEASE(pAnselResource->toServerRes.pTexture2DMutex);
    SAFE_RELEASE(pAnselResource->toServerRes.pSRV);
    SAFE_RELEASE(pAnselResource->toServerRes.pRTV);
    SAFE_RELEASE(pAnselResource->toServerRes.pDSV);
    SAFE_RELEASE(pAnselResource->toServerRes.pUAV);
    SAFE_RELEASE(pAnselResource->toClientRes.pTexture2D);
    SAFE_RELEASE(pAnselResource->toClientRes.pTexture2DMutex);
    SAFE_RELEASE(pAnselResource->toClientRes.pSRV);
    SAFE_RELEASE(pAnselResource->toClientRes.pRTV);
    SAFE_RELEASE(pAnselResource->toClientRes.pDSV);
    SAFE_RELEASE(pAnselResource->toClientRes.pUAV);
}

void AnselServer::releaseEffectState(AnselEffectState * pEffectState)
{
    SAFE_RELEASE(pEffectState->pVertexShader);
    SAFE_RELEASE(pEffectState->pPixelShader);
    SAFE_RELEASE(pEffectState->pRasterizerState);
    SAFE_RELEASE(pEffectState->pDepthStencilState);
    SAFE_RELEASE(pEffectState->pSamplerState);
    SAFE_RELEASE(pEffectState->pBlendState);
}

// TODO avoroshilov: consider moving this into D3D interface
HRESULT AnselServer::eraseClientResourceHandleIfFound(HCLIENTRESOURCE hClientResource)
{
    std::map<HCLIENTRESOURCE, AnselResource *>::iterator it = m_handleToAnselResource.find(hClientResource);
    std::lock_guard<std::mutex> lock(m_handleToAnselResourceLock);
    if (it != m_handleToAnselResource.end())
    {
        AnselResource * pAnselResource = it->second;
        if (pAnselResource)
        {
            m_bufDB.clearAnselResource(pAnselResource);
            releaseAnselResource(pAnselResource);
            delete pAnselResource;
        }

        // Erase dangling pointer in selected buffer after deleting the resource
        // Without this, it is possible for a selected buffer to contain a pointer to a deleted resource
        m_handleToAnselResource.erase(it);
    }

    return S_OK;
}

HRESULT AnselServer::notifyClientResourceDestroy(HCLIENTRESOURCE hClientResource)
{
    if (!m_lwrrentBufferInterface)
    {
        setBufferInterfaceD3D11();
    }

    eraseClientResourceHandleIfFound(hClientResource);

    if (hClientResource == m_d3d11Interface.m_lwrrentDepthBuffer)
    {
        m_d3d11Interface.m_lwrrentDepthBuffer = nullptr;
    }
    if (hClientResource == m_d3d11Interface.m_lwrrentHUDless)
    {
        m_d3d11Interface.m_lwrrentHUDless = nullptr;
    }
    if (hClientResource == m_d3d11Interface.m_lwrrentHDR)
    {
        m_d3d11Interface.m_lwrrentHDR = nullptr;
    }
    m_bufDB.clearClientResource(hClientResource);

    return S_OK;
}

HRESULT AnselServer::notifyDepthStencilClear(HCLIENTRESOURCE hClientResource)
{
    if (m_lightweightShimMode)
    {
        LOG_ERROR("DepthStencilClear notification in LW mode");
        return S_OK;
    }

    if (!m_lwrrentBufferInterface)
    {
        setBufferInterfaceD3D11();
    }

    HRESULT status = S_OK;
    if (m_enableDepthExtractionChecks && m_depthBufferUsed && hClientResource != nullptr)
    {
        AnselBufferDepth& depthBuf = m_bufDB.Depth();

        // When Stats mechanism is enabled, only copy when the client resource AND instance match
        // When Stats mechanism is disabled, only check if the client resource is a match
        if ((!depthBuf.useStats() && hClientResource == depthBuf.getClientResource()) ||
            depthBuf.useStats() && depthBuf.compareAgainstSelBuf(hClientResource))
        {
            depthBuf.copyResource(0);
        }

        // Mark this buffer as cleared in the depth buffer tracker
        depthBuf.bufCleared(hClientResource);
    }
    return status;
}

HRESULT AnselServer::notifyRenderTargetClear(HCLIENTRESOURCE hClientResource)
{
    if (m_lightweightShimMode)
    {
        LOG_ERROR("RenderTargetClear notification in LW mode");
        return S_OK;
    }

    if (!m_lwrrentBufferInterface)
    {
        setBufferInterfaceD3D11();
    }

    AnselBufferHudless& hudlessBuf = m_bufDB.Hudless();
    if (!hudlessBuf.useStats())
    {
        return S_OK;
    }

    if (m_enableHUDlessExtractionChecks && m_hudlessBufferUsed && hClientResource != nullptr)
    {
        // Mark this buffer as cleared in the hudless buffer tracker
        hudlessBuf.bufCleared(hClientResource);
    }

    return S_OK;
}

HRESULT AnselServer::notifyClientMode(DWORD clientMode)
{
    // We care only about the LW mode switch actually
    if ((clientMode & ANSEL_CLIENT_MODE_LIGHTWEIGHT) != 0 && !m_lightweightShimMode)
    {
        LOG_DEBUG("Entering lightweight mode");

        m_lightweightShimMode = true;

        // If we added resource references prior to that notification, and we were in the heavy mode
        //  we need to empty all of them out, and start from scratch

        HCLIENTRESOURCE hSelectedDepthBuffer = m_bufDB.Depth().getClientResource();
        if (hSelectedDepthBuffer)
        {
            m_bufDB.Depth().resetStatus();
            eraseClientResourceHandleIfFound(hSelectedDepthBuffer);
        }
        if (m_d3d11Interface.m_lwrrentDepthBuffer)
        {
            eraseClientResourceHandleIfFound(m_d3d11Interface.m_lwrrentDepthBuffer);
            m_d3d11Interface.m_lwrrentDepthBuffer = nullptr;
        }

        HCLIENTRESOURCE hSelectedHUDlessBuffer = m_bufDB.Hudless().getClientResource();
        if (hSelectedHUDlessBuffer)
        {
            m_bufDB.Hudless().resetStatus();
            eraseClientResourceHandleIfFound(hSelectedHUDlessBuffer);
        }
        if (m_d3d11Interface.m_lwrrentHUDless)
        {
            eraseClientResourceHandleIfFound(m_d3d11Interface.m_lwrrentHUDless);
            m_d3d11Interface.m_lwrrentHUDless = nullptr;
        }

        HCLIENTRESOURCE hSelectedHDRBuffer = m_bufDB.HDR().getClientResource();
        if (hSelectedHDRBuffer)
        {
            m_bufDB.HDR().resetStatus();
            eraseClientResourceHandleIfFound(hSelectedHDRBuffer);
        }
        if (m_d3d11Interface.m_lwrrentHDR)
        {
            eraseClientResourceHandleIfFound(m_d3d11Interface.m_lwrrentHDR);
            m_d3d11Interface.m_lwrrentHDR = nullptr;
        }
    }
    else if ((clientMode & ANSEL_CLIENT_MODE_LIGHTWEIGHT) == 0)
    {
        if (m_lightweightShimMode)
        {
            LOG_DEBUG("Exiting lightweight mode");
        }

        m_lightweightShimMode = false;
    }

    return S_OK;
}

AnselUIBase::Status AnselServer::startSession(bool isModdingAllowed, uint32_t effectsRestrictedSetIDPhoto, uint32_t effectsRestrictedSetIDModding)
{
    HRESULT status = S_OK;
    AnselUIBase::Status ret = AnselUIBase::Status::kUnknown;
    bool shotHDRAllowed = false;
    bool shotTypeEnabled[(int)ShotType::kNumEntries];

    shotTypeEnabled[(int)ShotType::kRegular] = true;
    shotTypeEnabled[(int)ShotType::kRegularUI] = true;
    shotTypeEnabled[(int)ShotType::k360] = false;
    shotTypeEnabled[(int)ShotType::k360Stereo] = false;
    shotTypeEnabled[(int)ShotType::kHighRes] = false;
    shotTypeEnabled[(int)ShotType::kStereo] = false;

    bool enhancedSessionStarted = false;
    if (m_activeControlClient->isAnselSDKSessionRequested())
    {
        if (m_anselSDK.isDetected())
        {
            bool enhancedSessionAllowed = m_anselSDK.startSession(getWidth(), getHeight(), m_useHybridController);

            m_anselSDK.initTitleAndDrsNames();

            // We need this check in case game didn't yet set up its sessionConfig
            if (enhancedSessionAllowed)
            {
                // TODO avoroshilov UIA
                //  determine how to set up elements in this place properly, esp. setting FOV limits

                // Set UI configuration

                // Engine limitations
                const auto& config = m_anselSDK.getConfiguration();
                auto sessionConfig = m_anselSDK.getSessionConfiguration();

                // Temporary workaround for a bug in the Obduction integration:
                if (m_appName.find(L"Obduction") != std::string::npos)
                {
                    sessionConfig.isFovChangeAllowed = true;
                }

                shotTypeEnabled[(int)ShotType::k360] = config.isCameraRotationSupported && config.isCameraFovSupported;
                shotTypeEnabled[(int)ShotType::k360Stereo] = config.isCameraRotationSupported && config.isCameraFovSupported && config.isCameraTranslationSupported;
                shotTypeEnabled[(int)ShotType::kHighRes] = config.isCameraOffcenteredProjectionSupported && config.isCameraFovSupported;
                shotTypeEnabled[(int)ShotType::kStereo] = config.isCameraTranslationSupported;

                // Session configuration
                shotTypeEnabled[(int)ShotType::k360] = shotTypeEnabled[(int)ShotType::k360] &&
                    (sessionConfig.is360MonoAllowed &&
                        sessionConfig.isRotationAllowed &&
                        sessionConfig.isFovChangeAllowed &&
                        sessionConfig.isPauseAllowed);
                shotTypeEnabled[(int)ShotType::k360Stereo] = shotTypeEnabled[(int)ShotType::k360Stereo] &&
                    (sessionConfig.is360StereoAllowed &&
                        sessionConfig.isRotationAllowed &&
                        sessionConfig.isTranslationAllowed &&
                        sessionConfig.isFovChangeAllowed &&
                        sessionConfig.isPauseAllowed);
                shotTypeEnabled[(int)ShotType::kHighRes] = shotTypeEnabled[(int)ShotType::kHighRes] &&
                    (sessionConfig.isHighresAllowed &&
                        sessionConfig.isFovChangeAllowed &&
                        sessionConfig.isPauseAllowed);
                // C/C++ doesn't have a &&= but using &= (bitwise and with assignment) is perfectly safe for booleans
                shotTypeEnabled[(int)ShotType::kStereo] &= sessionConfig.isPauseAllowed;

                shotHDRAllowed = sessionConfig.isRawAllowed;

                //m_activeControlClient->m_shotTypeEnabled[(int)ShotType::kStereo] = m_activeControlClient->m_shotTypeEnabled[(int)ShotType::kStereo] && sessionConfiguration->isRegularStereoAllowed;
                m_activeControlClient->setFovControlEnabled(config.isCameraFovSupported && sessionConfig.isFovChangeAllowed);
                m_activeControlClient->set360WidthLimits(4096, m_maxSphericalResolution * 1024);
                // clamp maximum FoV value to [140, 179] inclusive range
                const float maximumFov = std::min(std::max(sessionConfig.maximumFovInDegrees, 140.0f), 179.0f);
                m_activeControlClient->setFOVLimitsDegrees(10.0, maximumFov);
                m_activeControlClient->setRollLimitsDegrees(-180.0, 180.0);
                ret = AnselUIBase::Status::kOkAnsel;

                enhancedSessionStarted = true;
            }
            else  // Temporary fix to not enable ansel UI when the game says no!
            {
                LOG_INFO("Enhanced session is not allowed");

                if (!isModdingAllowed)
                {
                    LOG_INFO("Enhanced session is required to launch Ansel");
                    return AnselUIBase::Status::kDeclined;
                }
                else
                {
                    LOG_INFO("Ansel will be launched in Filter-only mode");
                    ret = AnselUIBase::Status::kOkFiltersOnly;
                }
            }
        }
        else
        {
            // We require SDK to use Ansel in any form lwrrently
            LOG_INFO("SDK is not detected");
            if (!isModdingAllowed)
            {
                LOG_INFO("SDK integration is required to launch Ansel");
                return AnselUIBase::Status::kDeclined;
            }
            else
            {
                LOG_INFO("Ansel will be launched in Filter-only mode");
                ret = AnselUIBase::Status::kOkFiltersOnly;
            }
        }
    }
    else
    {
        if (!isModdingAllowed)
        {
            LOG_INFO("Modding is not allowed");
            return AnselUIBase::Status::kDeclined;
        }
        else
        {
            LOG_INFO("Ansel will be launched in Filter-only mode");
            ret = AnselUIBase::Status::kOkFiltersOnly;
        }
    }

    // TODO avoroshilov UIA:
    //  move this call into pre-start, as isDetected is callwlated there anyways (but through pre-start API)
    m_activeControlClient->setAnselSDKDetected(m_anselSDK.isDetected());
    m_activeControlClient->setShotTypePermissions(shotHDRAllowed, shotTypeEnabled, (int)ShotType::kNumEntries);

    m_activeControlClient->onSessionStarted(m_anselSDK.isSessionActive());
    LOG_VERBOSE("Ansel session started %f", m_globalPerfCounters.elapsedTime);

    // Getting list of supported FX extensions and tools
    status = parseEffectCompilersList(m_installationFolderPath + L"fxtools.cfg");
    if (status != S_OK)
    {
        LOG_WARN("Failed to parse fxtools.cfg.. reverting to the integrated list");

#if _M_AMD64
        m_fxExtensionToolMap.insert(std::make_pair(L"yaml", L"YAMLFXC64.exe"));
        m_fxExtensionToolMap.insert(std::make_pair(L"fx", L"ReShadeFXC64.exe"));
#else
        m_fxExtensionToolMap.insert(std::make_pair(L"yaml", L"YAMLFXC32.exe"));
        m_fxExtensionToolMap.insert(std::make_pair(L"fx", L"ReShadeFXC32.exe"));
#endif
    }

#if ANSEL_SIDE_PRESETS
    if (m_specialClientActive)
    {
        m_fxExtensionToolMap.insert(std::make_pair(L"ini", L""));
    }
#endif

    buildMapIdentifiersList(m_fxExtensionToolMap, &m_fxExtensions);

    // Adding dot to the extension to avoid partial matches
    const size_t dotExtSize = 32;
    wchar_t dotExt[dotExtSize];
    for (size_t cnt = 0, cntEnd = m_fxExtensions.size(); cnt < cntEnd; ++cnt)
    {
        swprintf_s(dotExt, dotExtSize, L".%s", m_fxExtensions[cnt].c_str());
        m_fxExtensions[cnt] = dotExt;
    }

    // Repopulate effects list [requires extensions list to be ready]
    populateEffectsList(enhancedSessionStarted ? effectsRestrictedSetIDPhoto : effectsRestrictedSetIDModding);
    if (m_effectsInfo.m_effectFilesList.size() == 0)
        m_effectsInfo.m_selectedEffect = 0;

#ifdef ENABLE_STYLETRANSFER
    if (m_isStyleTransferEnabled)
    {
        populateStylesList();
        populateStyleNetworksList();
    }
#endif

    m_activeControlClient->setScreenSize(getWidth(), getHeight());
    m_prevWidth = getWidth();
    m_prevHeight = getHeight();
    m_sessionStartTime = m_globalPerfCounters.elapsedTime; //reset the session duration

    m_sessionFrameCount = 0;

    return ret;
}

void AnselServer::stopSession()
{
    //was enabled and got disabled
    {
        //telemetry
        AnselStateForTelemetry state;
        HRESULT telemetryStatus = makeStateSnapshotforTelemetry(state);
        if (telemetryStatus == S_OK)
            sendTelemetryCloseUIEvent(state);
    }

    m_anselSDK.stopSession();
    LOG_VERBOSE("Ansel session stopped %f", m_globalPerfCounters.elapsedTime);

    m_activeControlClient->onSessionStopped();
#ifdef ENABLE_STYLETRANSFER
    m_styleSelected = -1;
    m_activeControlClient->setStyleTransferStatus(false);

    if (restyleDeinitializeFunc)
        restyleDeinitializeFunc();

    SAFE_RELEASE(m_styleTransferOutputBuffer.tex);
    SAFE_RELEASE(m_styleTransferOutputHDRStorageBuffer.tex);
    m_refreshStyleTransferAfterCapture = false;
    m_refreshStyleTransfer = true;
#endif

    // Since DLL notifications are serialized, and the order in which notifications execute is undefined, we cannot rely on destroying telemetry within the DLL_PROCESS_DETACH notification.
    // Otherwise, if we destroy telemetry during DLL_PROCESS_DETACH, LwTelemetry might try to be unloaded before LwCamera, causing a deadlock where LwTelemetry waits for LwCamera to destroy its telemetry.
    // Instead we destroy telemetry when a session ends.
    destroyTelemetry();
}

#define ANSEL_FEATURE_ENABLE                0xFF
#define ANSEL_FEATURE_FADEIN                0xFE
#define ANSEL_FEATURE_TEST_MODE             0xFD

HRESULT AnselServer::setConfig(AnselConfig * pConfig)
{
    // TODO avoroshilov: review this piece - seems obsolete
    m_toggleHotkeyModCtrl = false;
    m_toggleHotkeyModShift = false;
    m_toggleHotkeyModAlt = false;
    switch (pConfig->hotkeyModifier)
    {
    case ANSEL_HOTKEY_MODIFIER_CTRL:
        m_toggleHotkeyModCtrl = true;
        break;
    case ANSEL_HOTKEY_MODIFIER_SHIFT:
        m_toggleHotkeyModShift = true;
        break;
    case ANSEL_HOTKEY_MODIFIER_ALT:
        m_toggleHotkeyModAlt = true;
        break;
    default:
        break;
    };

    m_toggleAnselHotkey = (unsigned short)pConfig->keyEnable;

    for (unsigned int i = 0; i < pConfig->numAnselFeatures; ++i)
    {
        if (pConfig->pAnselFeatures[i].featureId == ANSEL_FEATURE_ENABLE)
        {
            if (pConfig->pAnselFeatures[i].featureState != ANSEL_FEATURE_STATE_UNKNOWN)
            {
                m_bNextFrameForceEnableAnsel = pConfig->pAnselFeatures[i].featureState == ANSEL_FEATURE_STATE_ENABLE;
                m_bNextFrameForceDisableAnsel = pConfig->pAnselFeatures[i].featureState == ANSEL_FEATURE_STATE_DISABLE;
            }
        }
        if (pConfig->pAnselFeatures[i].featureId == ANSEL_FEATURE_FADEIN)
        {
            if (pConfig->pAnselFeatures[i].featureState != ANSEL_FEATURE_STATE_UNKNOWN)
            {
                m_bNextFrameEnableFade = pConfig->pAnselFeatures[i].featureState == ANSEL_FEATURE_STATE_ENABLE;
                m_bNextFrameDisableFade = pConfig->pAnselFeatures[i].featureState == ANSEL_FEATURE_STATE_DISABLE;
            }
        }
        if (pConfig->pAnselFeatures[i].featureId == ANSEL_FEATURE_TEST_MODE)
        {
            if (pConfig->pAnselFeatures[i].featureState != ANSEL_FEATURE_STATE_UNKNOWN)
            {
                if (pConfig->pAnselFeatures[i].featureState == ANSEL_FEATURE_STATE_ENABLE)
                {
                    m_showMouseWhileDefolwsed = true;
                }
                else if (pConfig->pAnselFeatures[i].featureState == ANSEL_FEATURE_STATE_DISABLE)
                {
                    m_showMouseWhileDefolwsed = false;
                }
            }
        }
    }

    //feodorb: this functionality has to be revisited on the interface level. Disabled so far, as we can't make any meaningful use out of it
#if 0
    for (unsigned int i = 0; i < pConfig->numAnselFeatures; ++i)
    {
        if (pConfig->pAnselFeatures[i].featureId == ANSEL_FEATURE_BLACK_AND_WHITE)
        {
            if (pConfig->pAnselFeatures[i].hotkey)
            {
                m_HotkeyEnableBlackAndWhite = pConfig->pAnselFeatures[i].hotkey;
            }
            if (pConfig->pAnselFeatures[i].featureState != ANSEL_FEATURE_STATE_UNKNOWN)
            {
                m_bNextFrameEnableBlackAndWhite = pConfig->pAnselFeatures[i].featureState == ANSEL_FEATURE_STATE_ENABLE;
            }
        }
        if (pConfig->pAnselFeatures[i].featureId == 0x3) // Not sure we want to expose these publically yet?
        {
            if (pConfig->pAnselFeatures[i].hotkey)
            {
                m_HotkeyEnableDepthRender = pConfig->pAnselFeatures[i].hotkey;
            }
            if (pConfig->pAnselFeatures[i].featureState != ANSEL_FEATURE_STATE_UNKNOWN)
            {
                m_bNextFrameDepthBufferUsed = pConfig->pAnselFeatures[i].featureState == ANSEL_FEATURE_STATE_ENABLE;
            }
        }
        if (pConfig->pAnselFeatures[i].featureId == 0x4) // Not sure we want to expose these publically yet?
        {
            if (pConfig->pAnselFeatures[i].hotkey)
            {
                m_HotkeyRenderDepthAsRGB = pConfig->pAnselFeatures[i].hotkey;
            }
            if (pConfig->pAnselFeatures[i].featureState != ANSEL_FEATURE_STATE_UNKNOWN)
            {
                m_bNextFrameRenderDepthAsRGB = pConfig->pAnselFeatures[i].featureState == ANSEL_FEATURE_STATE_ENABLE;
            }
        }
    }
#endif
    return S_OK;
}

//FDTODO: AnselInput remove this
HRESULT AnselServer::notifyHotkey(DWORD vkey)
{
    HRESULT status = S_OK;
    return status;
}

HRESULT AnselServer::updateGPUMask(DWORD activeGPUMask)
{
    LwAPI_Status ret = LWAPI_OK;

    LWAPI_SLI_UPDATE_MASK_STRUCT sliUpdateMask = { LWAPI_SLI_UPDATE_MASK_STRUCT_VER1 };
    sliUpdateMask.appActiveMask = activeGPUMask;
    ret = LwAPI_D3D_UpdateSLIMask(m_d3dDevice, &sliUpdateMask);
    if (ret != LWAPI_OK)
    {
        LwAPI_ShortString szDesc = { 0 };
        LwAPI_GetErrorMessage(ret, szDesc);
        printf("%s\n", szDesc);
        return E_FAIL;
    }

    return S_OK;
}

ID3D11Texture2D* AnselServer::createFullscreenTexture(DXGI_FORMAT fmt, D3D11_USAGE usage, UINT bindFlags, UINT cpuAccess)
{
    ID3D11Texture2D* output = nullptr;
    D3D11_TEXTURE2D_DESC textureDesc;
    // Initialize the render target texture description.
    ZeroMemory(&textureDesc, sizeof(textureDesc));

    textureDesc.MipLevels = 1;
    textureDesc.ArraySize = 1;
    textureDesc.SampleDesc.Count = 1;
    textureDesc.SampleDesc.Quality = 0;
    textureDesc.MiscFlags = 0;

    textureDesc.Usage = usage;
    textureDesc.CPUAccessFlags = cpuAccess;

    textureDesc.Width = getWidth();
    textureDesc.Height = getHeight();
    textureDesc.Format = fmt;
    textureDesc.BindFlags = bindFlags;

    HRESULT status = shadermod::Tools::CreateTexture2D(m_d3dDevice, &textureDesc, NULL, &output);
    return output;
}

std::wstring GenerateB64MetadataDescription(
    const std::wstring& tagDRSAppName,
    const std::wstring& tagDRSProfileName,
    const std::wstring& tagShortName,
    const std::wstring& tagCmsId)
{
    // This description is needed by GFE for thier A8 UI implementation.
    std::wstringstream ss;
    ss << L"{";
    ss << "\"DRSAppName\" : \"" << tagDRSAppName << "\"";
    ss << L", ";
    ss << "\"DRSProfileName\" : \"" << tagDRSProfileName << "\"";
    ss << L", ";
    ss << "\"ShortName\" : \"" << tagShortName << "\"";
    ss << L", ";
    ss << "\"CmsId\" : \"" << tagCmsId << "\"";
    ss << L"}";
    return darkroom::base64_encode(ss.str());
}

HRESULT AnselServer::saveShot(const AnselSDKCaptureParameters &captureParams, bool forceRGBA8Colwersion, const std::wstring& shotName, bool useAdditionalResource)
{
    HRESULT status = S_OK;

#if _DEBUG && DBG_EMIT_HASHES

    // Save hashes generated for all the lwrrently available restricted filters to the log file
    SaveHashes();

#endif

    const AnselResourceData * pPresentResourceData = useAdditionalResource ? captureParams.pPresentResourceDataAdditional : captureParams.pPresentResourceData;
    if (!pPresentResourceData)
    {
        LOG_ERROR("Save shot failed: buffer was empty");
        reportNonFatalError(__FILE__, __LINE__, 0, "Save shot failed: buffer was empty");
        return E_FAIL;
    }

    const std::string ext = darkroom::getUtf8FromWstr(findExtension(shotName));
    const bool isExtensionEXR = (ext == ".exr");
    const bool isExtensionJXR = (ext == ".jxr");

    darkroom::BufferFormat fmt;
    DXGI_FORMAT dxgifmt = DXGI_FORMAT_UNKNOWN;
    WICPixelFormatGUID jxrfmt = GUID_WICPixelFormatUndefined;
    unsigned int pixelByteSize = 0;
    bool isHdr = false;

    ID3D11Texture2D* sourceTexture = pPresentResourceData->pTexture2D;
    DWORD sourceSampleCount = pPresentResourceData->sampleCount;
    UINT sourceWidth = pPresentResourceData->width;
    UINT sourceHeight = pPresentResourceData->height;

    const DXGI_FORMAT sourceFormat = lwanselutils::colwertFromTypelessIfNeeded(pPresentResourceData->format);
    switch (sourceFormat)
    {
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            dxgifmt = DXGI_FORMAT_R8G8B8A8_UNORM;
            fmt = darkroom::BufferFormat::RGBA8;
            jxrfmt = GUID_WICPixelFormat32bppRGBA;
            pixelByteSize = 4;
            break;

        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            dxgifmt = DXGI_FORMAT_B8G8R8A8_UNORM;
            fmt = darkroom::BufferFormat::BGRA8;
            jxrfmt = GUID_WICPixelFormat32bppBGRA;
            pixelByteSize = 4;
            break;

        case DXGI_FORMAT_R10G10B10A2_UNORM:
            dxgifmt = DXGI_FORMAT_R10G10B10A2_UNORM;
            fmt = darkroom::BufferFormat::RGBA8;
            pixelByteSize = 4;
            jxrfmt = GUID_WICPixelFormat32bppRGBA1010102;
            break;

        case DXGI_FORMAT_R11G11B10_FLOAT:
        case DXGI_FORMAT_R16G16B16A16_FLOAT:
        case DXGI_FORMAT_R32G32B32_FLOAT:
        case DXGI_FORMAT_R32G32B32A32_FLOAT:
            dxgifmt = DXGI_FORMAT_R32G32B32A32_FLOAT;
            jxrfmt = GUID_WICPixelFormat128bppRGBAFloat;
            fmt = darkroom::BufferFormat::RGBA32;
            pixelByteSize = 16;
            isHdr = true;
            break;
            //no alfa-first formats in DXGI; no 3-byte formats in DXGI
    }

    // If we're requested to save EXR, so be it
    if (isExtensionEXR)
    {
        dxgifmt = DXGI_FORMAT_R32G32B32A32_FLOAT;
        fmt = darkroom::BufferFormat::RGBA32;
        pixelByteSize = 16;
        isHdr = true;
    }

    // if RT format is not supported, we can't save it
    if (dxgifmt == DXGI_FORMAT_UNKNOWN)
    {
        reportNonFatalError(__FILE__, __LINE__, 0, "saveShot couldn't save the surface - unsupported format (%d)", dxgifmt);
        return E_FAIL;
    }

    if (!pixelByteSize)
    {
        return E_FAIL;
    }

    if (m_readbackTexture && (m_readbackWidth != getWidth() || m_readbackHeight != getHeight() || m_readbackFormat != dxgifmt))
    {
        SAFE_RELEASE(m_readbackTexture);
    }

    if (m_resolvedPrereadbackTexture && (m_readbackWidth != getWidth() || m_readbackHeight != getHeight()
        || m_resolvedPrereadbackFormat != sourceFormat))
    {
        SAFE_RELEASE(m_resolvedPrereadbackTexture);
    }

    // TODO: also check if resolution changed - feodorb: ???
    if (!m_readbackTexture && pixelByteSize > 0)
    {
        // Special treatment: readback texture
        D3D11_TEXTURE2D_DESC textureDesc;
        // Initialize the render target texture description.
        ZeroMemory(&textureDesc, sizeof(textureDesc));

        textureDesc.MipLevels = 1;
        textureDesc.ArraySize = 1;
        textureDesc.SampleDesc.Count = 1;
        textureDesc.SampleDesc.Quality = 0;
        textureDesc.MiscFlags = 0;

        textureDesc.Usage = D3D11_USAGE_STAGING;
        textureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;

        m_readbackWidth = getWidth();
        m_readbackHeight = getHeight();
        m_readbackFormat = dxgifmt;
        textureDesc.Width = getWidth();
        textureDesc.Height = getHeight();
        textureDesc.Format = dxgifmt;
        textureDesc.BindFlags = 0;

        if (m_readbackData)
        {
            delete[] m_readbackData;
            m_readbackData = nullptr;
        }

        m_readbackData = new unsigned char[getWidth() * getHeight() * pixelByteSize];

        if (!m_readbackData || !SUCCEEDED(status = shadermod::Tools::CreateTexture2D(m_d3dDevice, &textureDesc, NULL, &m_readbackTexture)))
        {
            if (m_readbackData)
                delete[] m_readbackData;

            reportFatalError(__FILE__, __LINE__, FatalErrorCode::kReadbackCreateFail, "Failed to create readback texture (%d)", status);
            LOG_ERROR("Save shot failed: failed to create readback texture");
            HandleFailure();
        }
    }

    if (!m_resolvedPrereadbackTexture && sourceSampleCount > 1)
    {
        // Special treatment: readback texture
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

        textureDesc.Width = sourceWidth;
        textureDesc.Height = sourceHeight;
        textureDesc.Format = m_resolvedPrereadbackFormat = sourceFormat;
        textureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

        if (!SUCCEEDED(status = shadermod::Tools::CreateTexture2D(m_d3dDevice, &textureDesc, NULL, &m_resolvedPrereadbackTexture)))
        {
            LOG_ERROR("Save shot failed: failed to create resolved pre-readback texture");
            HandleFailure();
        }
    }

    if (sourceTexture)
    {
        ID3D11Texture2D * resolvedTexture = sourceTexture;
        if (sourceSampleCount != 1)
        {
            m_immediateContext->ResolveSubresource(m_resolvedPrereadbackTexture, 0, sourceTexture, 0, sourceFormat);
            resolvedTexture = m_resolvedPrereadbackTexture;
        }

        // TODO (david): Why is this needed??? Can we skip at least for JXR?
        if (isHdr && sourceFormat != DXGI_FORMAT_R32G32B32A32_FLOAT)
        {
            // TODO avoroshilov: collapse the code even further
            ID3D11Texture2D * hdrPrereadbackTexture;
            AnselResourceData hdrPrereadbackData = { };
            hdrPrereadbackData.format = sourceFormat;
            hdrPrereadbackData.pTexture2D = resolvedTexture;
            hdrPrereadbackData.width = getWidth();
            hdrPrereadbackData.height = getHeight();
            hdrPrereadbackData.sampleCount = 1;
            hdrPrereadbackData.sampleQuality = 0;

            if (!SUCCEEDED( status = m_renderBufferColwerter.getHDR32FTexture(hdrPrereadbackData, &hdrPrereadbackTexture) ))
            {
                LOG_ERROR("Save shot failed: failed to colwert HDR texture (%d)", (int)sourceFormat);
                HandleFailure();
            }

            m_immediateContext->CopySubresourceRegion(m_readbackTexture, 0, 0, 0, 0, hdrPrereadbackTexture, 0, 0);
        }
        else
        {
            m_immediateContext->CopySubresourceRegion(m_readbackTexture, 0, 0, 0, 0, resolvedTexture, 0, 0);
        }
    }

    D3D11_MAPPED_SUBRESOURCE msr;
    if (!SUCCEEDED(status = m_immediateContext->Map(m_readbackTexture, 0, D3D11_MAP_READ, 0, &msr)))
    {
        LOG_ERROR("Save shot failed: failed to map readback texture");
        HandleFailure();
    }

    for (unsigned int i = 0, fbheight = getHeight(), fbwidth = getWidth(); i < captureParams.height; ++i)
    {
        memcpy(m_readbackData + sizeof(unsigned char) * pixelByteSize * fbwidth * i, (unsigned char *)msr.pData + i * msr.RowPitch, sizeof(unsigned char) * pixelByteSize * fbwidth);

        if (dxgifmt == DXGI_FORMAT_R10G10B10A2_UNORM && !isExtensionJXR) //special treatment
        {
            for (size_t x = 0; x < fbwidth; ++x)
            {
                //rrrr rrrr    gggg ggrr  bbbb  gggg   aabb bbbb
                unsigned char rhi = m_readbackData[4 * (captureParams.width * i + x) + 0]; //rrrr rrrr
                unsigned char rlo_ghi = m_readbackData[4 * (captureParams.width * i + x) + 1];//gggg ggrr
                unsigned char glo_bhi = m_readbackData[4 * (captureParams.width * i + x) + 2];//bbbb  gggg
                unsigned char blo_a = m_readbackData[4 * (captureParams.width * i + x) + 3];//aabb bbbb

                unsigned short r = static_cast<unsigned short>(rhi) | (static_cast<unsigned short>(rlo_ghi) & 0x3) << 8;
                unsigned short g = (static_cast<unsigned short>(rlo_ghi)& ~0x3) >> 2 | (static_cast<unsigned short>(glo_bhi) & 0xf) << 6;
                unsigned short b = (static_cast<unsigned short>(glo_bhi)& ~0xf) >> 4 | (static_cast<unsigned short>(blo_a) & 0x3f) << 4;
                unsigned short a = (static_cast<unsigned short>(blo_a)& ~0x3f) >> 6;

                //TODO: here we clamp the color to 8 bit, we should eventually overcome this limitation
                m_readbackData[4 * (fbwidth * i + x) + 0] = (unsigned char)(r >> 2);
                m_readbackData[4 * (fbwidth * i + x) + 1] = (unsigned char)(g >> 2);
                m_readbackData[4 * (fbwidth * i + x) + 2] = (unsigned char)(b >> 2);
                m_readbackData[4 * (fbwidth * i + x) + 3] = (unsigned char)(a << 6);
            }
        }
    }

    unsigned char * dataToSave = m_readbackData;

    if (isHdr && (forceRGBA8Colwersion || (!isExtensionEXR && !isExtensionJXR)))
    {
        LOG_DEBUG("Forcing RGBA8 colwersion...");
        if (m_readbackWidth != m_readbackTonemappedWidth || m_readbackHeight != m_readbackTonemappedHeight)
        {
            delete [] m_readbackTonemappedData;
            m_readbackTonemappedData = nullptr;
        }

        if (m_readbackTonemappedData == nullptr)
        {
            m_readbackTonemappedWidth = m_readbackWidth;
            m_readbackTonemappedHeight = m_readbackHeight;
            m_readbackTonemappedData = new unsigned char [m_readbackTonemappedWidth * m_readbackTonemappedHeight * 4];
        }

        // It is game dependent which tonemapping method gives the best results for colwerting tonemapped HDR buffers to SDR.
        // With some games, Reinhard tonemapping is best for this, and with other games, Filmic tonemapping is best.
        // Filmic Linear tonemapping, (also referred to as filmic-raw in some cases) appears to be better for colwerting raw HDR buffers to SDR.
        // However, in all cases where we ever need to colwert HDR to SDR, the HDR buffer is always the presentable buffer which is always already tonemapped.
        // This is because if we ever need an SDR thumbnail for a raw HDR buffer capture, we will just capture the presentable buffer for the thumbnail,
        // and the presentable buffer will always either be SDR already, or it will be an HDR buffer that has already been tonemapped.
        darkroom::tonemap<darkroom::TonemapOperator::kFilmic>(reinterpret_cast<float *>(m_readbackData), m_readbackTonemappedData, m_readbackTonemappedWidth, m_readbackTonemappedHeight, 4);
        dataToSave = m_readbackTonemappedData;
        fmt = darkroom::BufferFormat::RGBA8;
        pixelByteSize = 4;
    }

    if (captureParams.snapshotPath.c_str())
    {
        if (!lwanselutils::CreateDirectoryRelwrsively(captureParams.snapshotPath.c_str()))
        {
            LOG_ERROR("Save shot failed: failed to create directories");
            reportNonFatalError(__FILE__, __LINE__, 0, "Save shot failed: failed to create directories");

            return E_FAIL;
        }
    }

    using darkroom::Error;
    Error retcode;

    std::unordered_map<std::string, std::string> pngTags;

    auto tagDescription = darkroom::getUtf8FromWstr(GenerateB64MetadataDescription(m_anselSDK.getDrsAppName(), m_anselSDK.getDrsProfileName(), m_activeControlClient->getAppShortName(), m_activeControlClient->getAppCMSID()));
    auto drsName = darkroom::getUtf8FromWstr(m_anselSDK.getDrsAppName());
    auto drsProfileName = darkroom::getUtf8FromWstr(m_anselSDK.getDrsProfileName());
    if (drsName.empty())
        drsName = "None";
    if (drsProfileName.empty())
        drsProfileName = "None";

    if (ext == ".png" || isExtensionEXR || isExtensionJXR)
    {
        pngTags = {
            { "Model", darkroom::getUtf8FromWstr(m_deviceName) },
            { "Source", "LWPU" },
            { "Make", "LWPU" },
            { "Software", darkroom::getUtf8FromWstr(generateSoftwareTag()) },
            { "Description", tagDescription },
            { "Comment", "Regular" },
            { "MakerNote", "Regular" },
            { "DRSName", drsName },
            { "DRSProfileName", drsProfileName },
            { "AppTitleName", m_anselSDK.getTitleForTagging() },
            { "AppCMSID", darkroom::getUtf8FromWstr(m_activeControlClient->getAppCMSID()) },
            { "AppShortName", darkroom::getUtf8FromWstr(m_activeControlClient->getAppShortName()) },
            { "ActiveFilters", darkroom::getUtf8FromWstr(generateActiveEffectsTag()) }
        };
    }

    /////////////////////////PSD////////////////////////////////////
    if (m_psUtil.GetPsdExportEnable() && shotName.find(L"thumbnail") == std::wstring::npos)
    {
        std::vector<unsigned char> presentData(dataToSave, dataToSave + (captureParams.width * captureParams.height * pixelByteSize));
        std::vector<unsigned char> depthData, hudlessData, colorData;
        shadermod::Tools::ExtractDataFromBufferTexture<unsigned char>(m_d3dDevice, m_immediateContext, captureParams.depthTexture, depthData);
        shadermod::Tools::ExtractDataFromBufferTexture<unsigned char>(m_d3dDevice, m_immediateContext, captureParams.hudlessTexture, hudlessData);
        shadermod::Tools::ExtractDataFromBufferTexture<unsigned char>(m_d3dDevice, m_immediateContext, captureParams.colorTexture, colorData);

        // We would like to export with the smallest channel width as possible to decrease the size of the output file.
        // If we have a depth buffer or an hdr buffer we must export with the largest channel width (32 bits).
        // If we have neither a depth nor hdr buffer we are ok to export with 8 bit channel width.
        // 16 bit channel width functionality is supported, but does not have a use yet (perhaps a future buffer will use it)
        if (!depthData.empty() || pixelByteSize == 16)
        {
            m_psUtil.ExportCaptureAsPsd<float>(presentData, colorData, depthData, hudlessData, captureParams.width, captureParams.height, pixelByteSize == 16, shotName, PhotoShopUtils::ChannelWidth::kWidth32);
        }
        else
        {
            m_psUtil.ExportCaptureAsPsd<unsigned char>(presentData, colorData, depthData, hudlessData, captureParams.width, captureParams.height, false, shotName, PhotoShopUtils::ChannelWidth::kWidth08);
        }
    }
    //////////////////////////////////////////////////////////////////////////////

    if (ext == ".png")
    {
        retcode = darkroom::savePng(dataToSave, nullptr, shotName, captureParams.width, captureParams.height, fmt, pngTags, 1 /* best speed*/, 0 /* multithreaded */);
    }
    else if (ext == ".bmp")
    {
        retcode = darkroom::saveBmp(dataToSave, shotName, captureParams.width, captureParams.height, fmt);
    }
    else if (ext == ".jpg" || ext == ".jpeg")
    {
        const std::unordered_map<uint16_t, std::string> jpegTags = {
            { darkroom::gJPEG_TAG_SOURCE, "LWPU" }, // Make
            { darkroom::gJPEG_TAG_MODEL_1, darkroom::getUtf8FromWstr(m_deviceName) }, // Model
            { darkroom::gJPEG_TAG_MODEL_2, "Ansel" }, // UniqueCameraModel
            { darkroom::gJPEG_TAG_DESCRIPTION, tagDescription }, // ImageDescription
            { darkroom::gJPEG_TAG_SOFTWARE, darkroom::getUtf8FromWstr(generateSoftwareTag()) }, // Software
            { darkroom::gJPEG_TAG_TYPE, "Regular" }, // MakerNoteUnknownText
            { darkroom::gJPEG_TAG_DRSNAME, drsName },
            { darkroom::gJPEG_TAG_DRSPROFILENAME, drsProfileName },
            { darkroom::gJPEG_TAG_APPTITLENAME, m_anselSDK.getTitleForTagging() },
            { darkroom::gJPEG_TAG_APPCMSID, darkroom::getUtf8FromWstr(m_activeControlClient->getAppCMSID()) },
            { darkroom::gJPEG_TAG_APPSHORTNAME, darkroom::getUtf8FromWstr(m_activeControlClient->getAppShortName()) },
            { darkroom::gJPEG_TAG_ACTIVEFILTERS, darkroom::getUtf8FromWstr(generateActiveEffectsTag()) }
        };
        retcode = darkroom::saveJpeg(dataToSave, nullptr, shotName, captureParams.width, captureParams.height, fmt, jpegTags);
    }
    else if (isExtensionEXR)
    {
        retcode = darkroom::saveExr(dataToSave, nullptr, shotName, captureParams.width, captureParams.height, fmt, pngTags);
    }
    else if (isExtensionJXR)
    {
        // Record HDR data in vector to fill alpha channel to pass into jxr export function
        const unsigned int numBytes = getHeight() * getWidth() * pixelByteSize;
        std::vector<BYTE> hdrData(dataToSave, dataToSave + numBytes);

        // Saving a JXR file on a 4K monitor takes about 0.5s, which is too long to stall a frame. The saving of the file
        // can be done as a background thread, but without a lot of code rework, it's not easy to capture the Success/Fail
        // of the detached thread. For now, we'll assume the file creation worked, since JXR is only happening in conjuction
        // with PNG creation, so if JXR were to fail, it's likely the PNG would fail too, whose return code will be captured.
        std::thread t(darkroom::saveJxr, std::move(hdrData), captureParams.width, captureParams.height, jxrfmt, std::move(shotName), true, false);
        t.detach();

        LOG_WARN("JXR file creation spawned in detached thread; assuming success");
        retcode = Error::kSuccess;
    }

    // give a message box, abort capture
    if (retcode == Error::kCouldntCreateFile || (retcode == Error::kIlwalidArgument))
    {
        if (retcode == Error::kCouldntCreateFile)
        {
            m_activeControlClient->displayMessage(AnselUIBase::MessageType::kCouldntCreateFileNotEnoughSpaceOrPermissions);
            LOG_ERROR("Save shot failed: couldn't create a file");
            reportNonFatalError(__FILE__, __LINE__, 0, "Save shot failed: couldn't create a file");
        }
        else if (retcode == Error::kIlwalidArgument)
        {
            m_activeControlClient->displayMessage(AnselUIBase::MessageType::kCouldntSaveFile);
            LOG_ERROR("Save shot failed: couldn't save a file");
            reportNonFatalError(__FILE__, __LINE__, 0, "Save shot failed: couldn't save a file");
        }

        abortCapture();
        LOG_ERROR("Save shot failed: file saving error");
        status = E_FAIL;
    }

    m_immediateContext->Unmap(m_readbackTexture, 0);

    return status;
}

HRESULT AnselServer::createSharedResource(
    DWORD width,
    DWORD height,
    DWORD sampleCount,
    DWORD sampleQuality,
    DWORD format,
    HANDLE * pHandle,
    void * pServerPrivateData
    )
{
    HRESULT status = S_OK;

    LOG_VERBOSE("Creating shared resource with %s.", DxgiFormat_cstr(format));

    CREATESHAREDRESOURCEDATA * pCreateSharedResourceData = static_cast<CREATESHAREDRESOURCEDATA *>(pServerPrivateData);
    AnselSharedResourceData * pSharedResourceData = pCreateSharedResourceData->pSharedResourceData;

    D3D11_TEXTURE2D_DESC descDepth;
    descDepth.Width = width;
    descDepth.Height = height;
    descDepth.MipLevels = 1;
    descDepth.ArraySize = 1;
    DXGI_FORMAT dxgiFormat = (DXGI_FORMAT)format;
    dxgiFormat = lwanselutils::colwertToTypeless(dxgiFormat);
    if (pCreateSharedResourceData->overrideFormat != DXGI_FORMAT_UNKNOWN)
    {
        dxgiFormat = (DXGI_FORMAT)pCreateSharedResourceData->overrideFormat;
    }
    descDepth.Format = dxgiFormat;
    descDepth.SampleDesc.Count = sampleCount;
    if (pCreateSharedResourceData->overrideSampleCount)
    {
        descDepth.SampleDesc.Count = pCreateSharedResourceData->overrideSampleCount;
    }

    descDepth.SampleDesc.Quality = sampleQuality;
    descDepth.Usage = D3D11_USAGE_DEFAULT;
    descDepth.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    if (pCreateSharedResourceData->overrideBindFlags)
    {
        descDepth.BindFlags = pCreateSharedResourceData->overrideBindFlags;
    }
    descDepth.CPUAccessFlags = 0;
    descDepth.MiscFlags = D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;

    pSharedResourceData->width = width;
    pSharedResourceData->height = height;
    pSharedResourceData->sampleCount = sampleCount;
    pSharedResourceData->sampleQuality = sampleQuality;
    pSharedResourceData->format = (DXGI_FORMAT)format;

    LOG_DEBUG("CreateTexture2D() Called...");
    if (!SUCCEEDED(status = shadermod::Tools::CreateTexture2D(m_d3dDevice, &descDepth, NULL, &pSharedResourceData->pTexture2D)))
    {
        LOG_ERROR("Shared resource creation: texture creation failed");
        if (descDepth.Width == 0 || descDepth.Height == 0)
        {
            return status;
        }
        else
        {
            HandleFailure();
        }
    }

    IDXGIResource* pDXGIResource(NULL);
    if (!SUCCEEDED(status = pSharedResourceData->pTexture2D->QueryInterface(__uuidof(IDXGIResource), (LPVOID*)&pDXGIResource)))
    {
        LOG_ERROR("Shared resource creation: interface queries failed (0)");
        HandleFailure();
    }

    HANDLE sharedHandle;
    pDXGIResource->GetSharedHandle(&sharedHandle);

    if (!SUCCEEDED(status = pSharedResourceData->pTexture2D->QueryInterface(_uuidof(IDXGIKeyedMutex), (void **)&pSharedResourceData->pTexture2DMutex)))
    {
        LOG_ERROR("Shared resource creation: interface queries failed (1)");
        HandleFailure();
    }

    SAFE_RELEASE(pDXGIResource);

    *pHandle = sharedHandle;
    pSharedResourceData->sharedHandle = sharedHandle;

    return S_OK;
}

AnselResource * AnselServer::lookupAnselResource(HCLIENTRESOURCE hClientResource)
{
    AnselResource * rtn = NULL;
    std::lock_guard<std::mutex> lock(m_handleToAnselResourceLock);
    std::map<HCLIENTRESOURCE, AnselResource *>::iterator it = m_handleToAnselResource.find(hClientResource);
    if (it != m_handleToAnselResource.end())
    {
        AnselResource * pResViewData = it->second;
        if (pResViewData)
        {
            rtn = pResViewData;
        }
    }
    return rtn;
}

void AnselServer::debugRenderFrameNumber()
{
#if 0
    // TODO avoroshilov UIA
    //  enable this by providing pointer to the FW object(s)
    const size_t ui_textBufSize = 256;
    wchar_t ui_text[ui_textBufSize];

    static int frameNo = 0;
    swprintf_s(ui_text, ui_textBufSize, L"Ansel frame #%d", frameNo);
    ++frameNo;

    m_UI->FW1.pRenderStatesSergoeUI->SetStates(m_immediateContext, 0);
    m_immediateContext->PSSetShader(m_UI->FW1.pPSOutline, NULL, 0);

    // Draw some strings (Y goes top to bottom)
    m_UI->FW1.pFontWrapperSergoeUI->DrawString(
        m_immediateContext,
        ui_text,// String
        16.0f,// Font size
        (200.f / 1920.f) * getWidth(),// X offset
        (48.f / 1080.f) * getHeight(),// Y offset
        0xFFffFFff,// Text color, 0xAaBbGgRr
        FW1_LEFT | FW1_TOP | FW1_STATEPREPARED// Flags
    );

    //m_immediateContext->VSSetShader(0, NULL, 0);
    m_immediateContext->HSSetShader(0, NULL, 0);
    m_immediateContext->DSSetShader(0, NULL, 0);
    m_immediateContext->GSSetShader(0, NULL, 0);
    //m_immediateContext->PSSetShader(0, NULL, 0);
#endif
}

void AnselServer::abortCapture()
{
    //telemetry
    {
        AnselStateForTelemetry state;
        HRESULT telemetryStatus = makeStateSnapshotforTelemetry(state);
        if (telemetryStatus == S_OK)
            sendTelemetryAbortCaptureEvent(state);
    }

    m_anselSDK.abortCapture();
}

#ifdef ENABLE_STYLETRANSFER
std::pair<std::vector<float>, std::vector<float>> AnselServer::loadStyleTransferCache(const std::wstring& styleFilename, const std::wstring& stylePath)
{
    const auto styleCachePath = m_intermediateFolderPath + std::wstring(L"_styles\\");
    const auto memoryCacheName = styleFilename + m_activeControlClient->getLwrrentStyleNetwork();

    if (!m_styleStatisticsCache.count(styleFilename))
    {
        // if second level cache is a hit, load first level cache with the data

        // TODO: for now there is a cache collision for the similar named files in different style locations
        const std::wstring fileCachePath = styleCachePath + std::wstring(styleFilename) + m_activeControlClient->getLwrrentStyleNetwork() + L".bin";
        bool callwlateStats = true;
        // check if we can open this file
        if (PathFileExists(fileCachePath.c_str()))
        {
            // if we can, check it's timestamp is later than the original style image file timestamp
            uint64_t cacheDate = 0u, fileDate = 0u;
            if (shadermod::ir::filehelpers::getFileTime(fileCachePath.c_str(), cacheDate) &&
                shadermod::ir::filehelpers::getFileTime(stylePath.c_str(), fileDate) &&
                cacheDate > fileDate)
            {
                std::ifstream in(fileCachePath, std::wifstream::binary);

                if (in.good())
                {
                    auto& stats = m_styleStatisticsCache[memoryCacheName];
                    in.seekg(0, in.end);
                    const auto cacheSize = static_cast<size_t>(in.tellg());
                    in.seekg(0, in.beg);
                    const auto elementCount = (cacheSize / 2) / sizeof(float);
                    stats.first.resize(elementCount);
                    stats.second.resize(elementCount);
                    in.read(reinterpret_cast<char*>(stats.first.data()), elementCount * sizeof(float));
                    in.read(reinterpret_cast<char*>(stats.second.data()), elementCount * sizeof(float));
                    callwlateStats = false;
                }
            }
        }

        if (callwlateStats)
        {
            // otherwise, callwlate statistics
            uint32_t ws = 0u, hs = 0u;
            const auto style = darkroom::loadImage(stylePath, ws, hs, darkroom::BufferFormat::RGB8);
            if (!style.empty())
            {
                std::pair<std::vector<float>, std::vector<float>> stats;
                const auto network = m_activeControlClient->getLwrrentStyleNetwork();
                if (networks.find(network) == networks.cend())
                {
                    m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_NoModelFound);
                }
                else
                {
                    const auto net = networks.at(network);
                    const auto encoderPath = modelRoot + net.first;
                    const auto encoderPathW = darkroom::getUtf8FromWstr(encoderPath);
                    // stats.first.data(), stats.second.data(), stats.first.size()
                    std::vector<float> mean(1024), var(1024);
                    size_t statisticsSize = 0u;
                    Status status = Status::kFailed;
                    try
                    {
                        status = restyleCalcAdainMomentsFunc(encoderPathW.c_str(), style.data(), style.size(), hs, ws, &mean[0], &var[0], &statisticsSize);
                    }
                    catch (const std::bad_alloc&)
                    {
                        status = Status::kFailedNotEnoughMemory;
                    }
                    mean.resize(statisticsSize);
                    var.resize(statisticsSize);
                    stats.first = var;
                    stats.second = mean;

                    if (status == Status::kFailedNotEnoughMemory)
                    {
                        m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_NotEnoughVRAM);
                        m_activeControlClient->setStyleTransferStatus(false);
                        m_styleSelected = -1;
                    }
                    else if (status == Status::kFailed)
                    {
                        m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_FailedToInitalizeStyleTransfer);
                        m_activeControlClient->setStyleTransferStatus(false);
                        m_styleSelected = -1;
                    }

                    if (status == Status::kOk)
                    {
                        // in memory cache will grow, but given that 1 style statistics occupy 4kb of RAM, it's going to be really a lot of styles until
                        // this will matter
                        m_styleStatisticsCache[memoryCacheName] = stats;
                        // when writing to the first level cache, we should also write to second level cache
                        shadermod::ir::filehelpers::createDirectoryRelwrsively(styleCachePath.c_str());
                        const std::wstring fileCachePath = styleCachePath + std::wstring(styleFilename) + m_activeControlClient->getLwrrentStyleNetwork() + L".bin";
                        std::ofstream out(fileCachePath, std::wofstream::binary);
                        out.write(reinterpret_cast<char*>(stats.first.data()), stats.first.size() * sizeof(decltype(stats.first[0])));
                        out.write(reinterpret_cast<char*>(stats.second.data()), stats.second.size() * sizeof(decltype(stats.second[0])));
                    }
                }
            }
            else
            {
                m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_FailedToLoadStyle);
            }
        }
    }
    return m_styleStatisticsCache[memoryCacheName];
}

std::wstring AnselServer::generateRestyleLibraryName(
    const std::wstring& path,
    const std::wstring& baseName,
    const uint32_t major, const uint32_t minor,
    const std::wstring& versionString,
    bool debug)
{
    std::wstring librestyleName = baseName;
    if (major > 0u)
        librestyleName += std::to_wstring(major) + std::to_wstring(minor) + std::wstring(L".");
    librestyleName += versionString + (debug ? L"d" : L"") + std::wstring(L".dll");
    const auto librestylePath = std::wstring(path) + binariesRoot + L"\\";
    return librestylePath + librestyleName;
}
#endif

void AnselServer::DenylistBufferExtractionType(ansel::BufferType bufferType)
{
    m_denylistedBufferExtractionTypes.insert(bufferType);
    LOG_INFO("Denylisting \"%s\" buffer extraction.", ansel::GetBufferTypeName(bufferType).c_str());
}

void AnselServer::parseModdingModes(uint32_t modeFlags, ModdingSettings * moddingSP, ModdingSettings * moddingMP)
{
    moddingSP->mode = ModdingMode::kDisabled;
    moddingMP->mode = ModdingMode::kDisabled;
    moddingSP->bufferUse = ModdingBuffersUse::kDisallowAllExtra;
    moddingMP->bufferUse = ModdingBuffersUse::kDisallowAllExtra;
    moddingSP->restrictedEffectSetId = (uint32_t)ModdingRestrictedSetID::kEmpty;
    moddingMP->restrictedEffectSetId = (uint32_t)ModdingRestrictedSetID::kEmpty;
    switch (modeFlags)
    {
    case ANSEL_FREESTYLE_MODE_ENABLED:
        {
            moddingSP->mode = ModdingMode::kEnabled;
            moddingMP->mode = ModdingMode::kEnabled;
            moddingSP->bufferUse = ModdingBuffersUse::kAllowAll;
            moddingMP->bufferUse = ModdingBuffersUse::kAllowAll;
            break;
        }
    case ANSEL_FREESTYLE_MODE_DISABLED:
        {
            moddingSP->mode = ModdingMode::kDisabled;
            moddingMP->mode = ModdingMode::kDisabled;
            break;
        }
    case ANSEL_FREESTYLE_MODE_APPROVED_ONLY:
        {
            moddingSP->mode = ModdingMode::kRestrictedEffects;
            moddingMP->mode = ModdingMode::kRestrictedEffects;
            // The "DisallowDepth" here is by explicit request from marketing. It will basically mean that we have a prepackaged filters mode
            //  with some of the standard filters not working properly (i.e. DoF, GreenScreen, etc.) - and this was communicated quite
            //  explicitly to the marketing team.
            // As well as the fact that we have many restricted sets.
            // But that was their precise and explicit requirement to restrict exactly this mode to *all* prepackaged effects with depth buffers
            //  disabled. So here it is.
            moddingSP->bufferUse = ModdingBuffersUse::kAllowAll;//ModdingBuffersUse::kDisallowDepth;
            moddingMP->bufferUse = ModdingBuffersUse::kAllowAll;//ModdingBuffersUse::kDisallowDepth;
            moddingSP->restrictedEffectSetId = (uint32_t)ModdingRestrictedSetID::kPrepackaged;
            moddingMP->restrictedEffectSetId = (uint32_t)ModdingRestrictedSetID::kPrepackaged;
            break;
        }
    case ANSEL_FREESTYLE_MODE_MULTIPLAYER_APPROVED_ONLY:
        {
            moddingSP->mode = ModdingMode::kRestrictedEffects;
            moddingMP->mode = ModdingMode::kRestrictedEffects;
            moddingSP->bufferUse = ModdingBuffersUse::kAllowAll;
            moddingMP->bufferUse = ModdingBuffersUse::kAllowAll;
            moddingSP->restrictedEffectSetId = (uint32_t)ModdingRestrictedSetID::kMPApproved;
            moddingMP->restrictedEffectSetId = (uint32_t)ModdingRestrictedSetID::kMPApproved;
            break;
        }
    case ANSEL_FREESTYLE_MODE_MULTIPLAYER_DISABLED:
        {
            moddingSP->mode = ModdingMode::kEnabled;
            moddingMP->mode = ModdingMode::kDisabled;
            moddingSP->bufferUse = ModdingBuffersUse::kAllowAll;
            moddingMP->bufferUse = ModdingBuffersUse::kDisallowAllExtra;
            // SP effect set id is not important
            moddingMP->restrictedEffectSetId = (uint32_t)ModdingRestrictedSetID::kEmpty;
            break;
        }
    case ANSEL_FREESTYLE_MODE_MULTIPLAYER_DISABLE_DEPTH:
        {
            moddingSP->mode = ModdingMode::kEnabled;
            moddingMP->mode = ModdingMode::kEnabled;
            moddingSP->bufferUse = ModdingBuffersUse::kAllowAll;
            moddingMP->bufferUse = ModdingBuffersUse::kDisallowDepth;
            break;
        }
    case ANSEL_FREESTYLE_MODE_MULTIPLAYER_DISABLE_EXTRA_BUFFERS:
        {
            moddingSP->mode = ModdingMode::kEnabled;
            moddingMP->mode = ModdingMode::kEnabled;
            moddingSP->bufferUse = ModdingBuffersUse::kDisallowAllExtra;
            moddingMP->bufferUse = ModdingBuffersUse::kDisallowAllExtra;
            break;
        }
    }
}

void AnselServer::resizeEffectsInfo(size_t newEffectsNum)
{
    const size_t prevEffectsNum = m_effectsInfo.m_effectSelected.size();

    m_effectsInfo.m_effectSelected.resize(newEffectsNum);
    m_effectsInfo.m_effectRebuildRequired.resize(newEffectsNum);
    m_effectsInfo.m_bufferCheckRequired.resize(newEffectsNum);
    m_effectsInfo.m_bufferCheckMessages.resize(newEffectsNum);
    m_effectsInfo.m_effectsHashedName.resize(newEffectsNum);

    // m_effectsStack and m_effectsStackMapping will be resized/filled later on stack rebuild

    for (size_t effIdx = prevEffectsNum; effIdx < newEffectsNum; ++effIdx)
    {
        m_effectsInfo.m_effectSelected[effIdx] = 0;
        m_effectsInfo.m_effectRebuildRequired[effIdx] = false;
        m_effectsInfo.m_bufferCheckRequired[effIdx] = false;
        m_effectsInfo.m_bufferCheckMessages[effIdx] = (uint32_t)EffectsInfo::BufferToCheck::kNONE;
    }
}

void LOG_DEBUG_AnselBufferDetails(AnselBuffer* anselBuffer)
{
    if (anselBuffer && getLogSeverity() <= LogSeverity::kDebug)
    {
        std::string anselResourceLabel = anselBuffer->getInternalName();
        AnselResource* anselResource = anselBuffer->getAnselResource();
        if (anselResource)
        {
            DXGI_FORMAT savedFormat = anselResource->toServerRes.format;
            std::string savedFormatName = lwanselutils::GetDxgiFormatName(savedFormat);
            DXGI_FORMAT colwertedFormat = lwanselutils::colwertFromTypelessIfNeeded(savedFormat);
            bool isBufferHDR = isHdrFormatSupported(colwertedFormat);
            std::string isBufferHDRName = isBufferHDR ? "HDR" : "SDR";
            if (savedFormat == colwertedFormat)
            {
                LOG_DEBUG("%s: [%s] [0x%lx] [%s]", anselResourceLabel.c_str(), isBufferHDRName.c_str(), anselResource, savedFormatName.c_str());
            }
            else
            {
                std::string colwertedFormatName = lwanselutils::GetDxgiFormatName(colwertedFormat);
                LOG_DEBUG("%s: [%s] [0x%lx] [%s -> %s]", anselResourceLabel.c_str(), isBufferHDRName.c_str(), anselResource, savedFormatName.c_str(), colwertedFormatName.c_str());
            }
        }
        else
        {
            LOG_DEBUG("%s: [NULL] [0x%lx]", anselResourceLabel.c_str(), anselResource);
        }
    }
}

// Some apps (partilwlarly, holodeck) load the AnselControlSDK late.
// We check for the SDK at most once per ANSEL_CONTROL_SDK_MILLISECONDS_PER_DETECT_AND_INITIALIZE milliseconds,
// until ANSEL_CONTROL_SDK_MILLISECONDS_UNTIL_LAST_DETECT_AND_INTIALIZE milliseconds have passed
// We do not need to check if the SDK has already been detected, as the function will return early in that case.
// This is not an exact process - we wait at least 100 milliseconds after each check, so finishing all of the checks can take more than 30s
void AnselServer::enableControlSdk()
{
#if (ENABLE_CONTROL_SDK == 1)
    static std::chrono::steady_clock::time_point last_sdk_check = std::chrono::steady_clock::now();
    static constexpr int detectAndInitializeMaxAttempts = ANSEL_CONTROL_SDK_MILLISECONDS_UNTIL_LAST_DETECT_AND_INITIALIZE / ANSEL_CONTROL_SDK_MILLISECONDS_PER_DETECT_AND_INTIALIZE;

    if (m_anselControlSDKDetectAndInitializeAttempts <= detectAndInitializeMaxAttempts)
    {
        std::chrono::steady_clock::time_point time_now = std::chrono::steady_clock::now();
        uint32_t delta_time = uint32_t(std::chrono::duration_cast<std::chrono::milliseconds>(time_now - last_sdk_check).count());
        if (delta_time >= ANSEL_CONTROL_SDK_MILLISECONDS_PER_DETECT_AND_INTIALIZE)
        {
            m_anselControlSDK.detectAndInitializeAnselControlSDK();
            ++m_anselControlSDKDetectAndInitializeAttempts;
            last_sdk_check = time_now;
        }
    }

    m_anselControlSDK.updateControlState();
#endif
}

// DBG Debug force HUDless
//  !!! Be VERY careful as this copy is performed each frame even when Ansel is inactive
void AnselServer::debugForceRenderHudless(HCLIENTRESOURCE hPresentResource, DWORD subResIndex)
{
#ifdef _DEBUG
    static bool enableDebug = false;
    if (enableDebug)
    {
        AnselBufferHudless& hudlessBuf = m_bufDB.Hudless();
        AnselBufferPresent& presetBuf = m_bufDB.Present();
        if (!hudlessBuf.useStats() || hudlessBuf.getSelectedResource() != nullptr)
        {
            m_hudlessBufferUsed = true;

            presetBuf.setClientResource(hPresentResource);
            hudlessBuf.acquire(0);
            presetBuf.acquire(subResIndex);

            presetBuf.m_needCopyToClient = true;

            AnselResource* presentResourceData = m_d3d11Interface.lookupAnselResource(hPresentResource);
            AnselResource* hudlessResourceData = m_d3d11Interface.lookupAnselResource(hudlessBuf.getClientResource());
            if (hudlessResourceData)
            {
                m_immediateContext->CopySubresourceRegion(presentResourceData->toClientRes.pTexture2D, 0, 0, 0, 0, hudlessBuf.getValidResourceData()->pTexture2D, 0, 0);
            }
            else
            {
                m_immediateContext->CopySubresourceRegion(presentResourceData->toClientRes.pTexture2D, 0, 0, 0, 0, presentResourceData->toServerRes.pTexture2D, 0, 0);
            }

            presetBuf.release();
            hudlessBuf.release();
        }
    }
#endif
}

HRESULT AnselServer::finalizeFrame(HCLIENTRESOURCE hPresentResource, DWORD subResIndex)
{
    if (!m_bInitialized)
        return S_OK;

    ANSEL_PROFILE_ZONE(finalizeFrame, "Finalize Frame");

    // update session frame count
    m_sessionFrameCount++;

    // update registry
    m_registrySettings.tick();
    // update global perf counters
    m_globalPerfCounters.update();
#if ENABLE_NETWORK_HOOKS == 1
    // update network activity detector
    m_networkDetector.tick();
#endif

    enableControlSdk();

    // Resolve the stats buffers in case we were using stats system in any of them
    m_bufDB.Depth().resolveStats();
    m_bufDB.Hudless().resolveStats();

    debugForceRenderHudless(hPresentResource, subResIndex);

    bool forceNotSkipRestFrame = false;

    ANSEL_PROFILE_START(ff01_processAnselStatus, "processAnselStatus");

    // Piece of code where UIs actually detect that they want control
    if (getIPCModeEnabled() != 0)
    {
        // at this point (the first Present) Ansel SDK should either be loaded or not
        if (m_UiUpdateSdkStatusNeeded)
        {
            AnselSDKState::DetectionStatus status = m_anselSDK.detectAndInitializeAnselSDK(m_installationFolderPath, m_intermediateFolderPath,
                m_snapshotsFolderPath, sessionStartFunc, sessionStopFunc, bufferFinishedFunc, this);

            if (status == AnselSDKState::DetectionStatus::kSUCCESS)
            {
                m_UI->setAnselSDKDetected(m_anselSDK.isDetected());
            }

            m_UiUpdateSdkStatusNeeded = false;
        }
        // If the SDK is loaded after the first frame, then we check whenever we recieve a FeatureSetRequest
        if (m_activeControlClient->isAnselFeatureSetRequested())
        {
            AnselSDKState::DetectionStatus status = m_anselSDK.detectAndInitializeAnselSDK(m_installationFolderPath, m_intermediateFolderPath,
                m_snapshotsFolderPath, sessionStartFunc, sessionStopFunc, bufferFinishedFunc, this);

            if (status == AnselSDKState::DetectionStatus::kSUCCESS)
            {
                m_UI->setAnselSDKDetected(m_anselSDK.isDetected());
            }
            m_activeControlClient->anselFeatureSetRequestDone();
        }

#if IPC_ENABLED == 1
        UIIPC* ui = static_cast<UIIPC *>(m_UI);
        if (getIPCModeEnabled() != 0)
        {
            ui->checkInitObserver();
        }
        ui->exelwteBusMessages();
        // If high quality mode was changed by an IPC message, we have to update the UI to reflect this.
        bool highQuality = false;
        if (ansel::kUserControlIlwalidCallback != m_anselSDK.getUserControlValue(HIGH_QUALITY_CONTROL_ID, &highQuality)
            && highQuality != m_activeControlClient->isHighQualityEnabled())
        {
            bool isControlClientHighQualityEnabled = m_activeControlClient->isHighQualityEnabled();
            m_anselSDK.setUserControlValue(HIGH_QUALITY_CONTROL_ID, &isControlClientHighQualityEnabled);
        }

        // in IPC mode we want to unload LwCamera in case FreeStyle is disabled in
        // GFE settings.
        if (!m_anselSDK.isDetected() && m_WAR_IPCmodeEnabled && m_modEnableFeatureCheck && !m_UI->isModdingEnabled())
        {
            LOG_DEBUG("Deactivating LwCamera because mods are disabled");
            reportFatalError(__FILE__, __LINE__, FatalErrorCode::kModsDisabledFail, "Deactivating LwCamera because mods are disabled");
            deactivateAnsel();
            return S_FALSE;
        }
#endif
    }
    else
    {
        // 'anselSessionStateChangeRequest' and 'needToSkipRestFrame' is a special case scenario - Ansel deactivation
        //  we need it as it is right now, before we fix this in a right way

        // TODO avoroshilov UIA
        //  fix deactivation in a right way

        bool needToSkipRestFrame = false;
        int anselSessionStateChangeRequest = -1;
        AnselUI * UI_standalone = static_cast<AnselUI *>(m_UI);

        UI_standalone->processAnselStatus(m_anselSDK, &needToSkipRestFrame, &anselSessionStateChangeRequest, &forceNotSkipRestFrame);

        if (anselSessionStateChangeRequest == 0)
            stopSession();
    }

    if (m_sendAnselReady & (getIPCModeEnabled() != 0))
    {
        UIIPC* ui = static_cast<UIIPC *>(m_UI);
        if (ui->needToSendAnselReady())
            m_sendAnselReady = !ui->sendAnselReady();
    }

    // If there is no client active, check if any client requests access
    if (!m_isClientActive)
    {
        bool controlRequestsPresent = false;
        for (size_t uiIdx = 0, uiIdxEnd = m_anselClientsList.size(); uiIdx < uiIdxEnd; ++uiIdx)
        {
            AnselUIBase * lwrClient = m_anselClientsList[uiIdx];
            if (lwrClient->isRequestingControl())
            {
                controlRequestsPresent = true;
#if (ENABLE_CONTROL_SDK == 1)
                if (!m_anselControlSDK.isExlusive() || (lwrClient == static_cast<AnselUIBase *>(&m_anselControlSDK)))
                {
                    m_specialClientActive = (lwrClient == static_cast<AnselUIBase *>(m_UI));
                    m_activeControlClient = lwrClient;
                }
                else
                {
                    // Client requested control, but Ansel is in Ansel Control SDK exclusive mode
                    // The client will be rejected down the road
                }
#else
                m_specialClientActive = (lwrClient == static_cast<AnselUIBase *>(m_UI));
                m_activeControlClient = lwrClient;
#endif
            }
        }
        // Always set active control client to internal if not exclusive
        if (!controlRequestsPresent)
        {
#if (ENABLE_CONTROL_SDK == 1)
            if (!m_anselControlSDK.isExlusive())
            {
                m_specialClientActive = true;
                m_activeControlClient = m_UI;
            }
            else
            {
                m_specialClientActive = false;
                m_activeControlClient = &m_anselControlSDK;
            }
#else
            m_specialClientActive = true;
            m_activeControlClient = m_UI;
#endif
        }
    }

    for (size_t uiIdx = 0, uiIdxEnd = m_anselClientsList.size(); uiIdx < uiIdxEnd; ++uiIdx)
    {
        if (m_anselClientsList[uiIdx] != m_activeControlClient && m_anselClientsList[uiIdx]->isRequestingControl())
        {
            m_anselClientsList[uiIdx]->rejectControlRequest();
        }
    }

    if (m_allowFiltersInGame != m_wasModdingAllowedGlobally)
    {
        // Main switch to disable modding, not affected by the local networking changes
        if (!m_allowFiltersInGame)
            LOG_DEBUG("Disabling modding because (network status is %d)", m_networkActivityDetected);

        m_activeControlClient->setModdingStatus(m_allowFiltersInGame);
    }
    m_wasModdingAllowedGlobally = m_allowFiltersInGame;

    bool isAnselSessionActive = m_anselSDK.isSDKDetectedAndSessionActive();

    bool moddingMultiplayerMode = false;
    bool filtersLwrrentlyAllowed = m_allowFiltersInGame;
    if (filtersLwrrentlyAllowed)
    {
        if (m_networkActivityDetected)
        {
            // Multiplayer allowlisting logic
            const double networkActivityModdingCheckTime = 5000.0;
            m_nextNetworkActivityModdingCheck = networkActivityModdingCheckTime;

            moddingMultiplayerMode = true;
            if (m_moddingMultiPlayer.mode == ModdingMode::kDisabled)
            {
                filtersLwrrentlyAllowed = false;

                m_moddingDepthBufferAllowed = false;
                m_moddingHDRBufferAllowed = false;
                m_moddingHUDlessBufferAllowed = false;
            }
            else
            {
                m_moddingDepthBufferAllowed = true;
                m_moddingHDRBufferAllowed = true;
                m_moddingHUDlessBufferAllowed = true;
                if ((uint32_t)m_moddingMultiPlayer.bufferUse & (uint32_t)ModdingBuffersUse::kDisallowDepth)
                {
                    m_moddingDepthBufferAllowed = false;
                }
                if ((uint32_t)m_moddingMultiPlayer.bufferUse & (uint32_t)ModdingBuffersUse::kDisallowHDR)
                {
                    m_moddingHDRBufferAllowed = false;
                }
                if ((uint32_t)m_moddingMultiPlayer.bufferUse & (uint32_t)ModdingBuffersUse::kDisallowHUDless)
                {
                    m_moddingHUDlessBufferAllowed = false;
                }
            }
        }
        else
        {
            m_nextNetworkActivityModdingCheck -= m_globalPerfCounters.dt;
        }

        if (!m_networkActivityDetected && m_nextNetworkActivityModdingCheck < 0.0)
        {
            // Singleplayer allowlisting logic
            moddingMultiplayerMode = false;
            if (m_moddingSinglePlayer.mode == ModdingMode::kDisabled)
            {
                filtersLwrrentlyAllowed = false;

                m_moddingDepthBufferAllowed = false;
                m_moddingHDRBufferAllowed = false;
                m_moddingHUDlessBufferAllowed = false;
            }
            else
            {
                m_moddingDepthBufferAllowed = true;
                m_moddingHDRBufferAllowed = true;
                m_moddingHUDlessBufferAllowed = true;
                if ((uint32_t)m_moddingSinglePlayer.bufferUse & (uint32_t)ModdingBuffersUse::kDisallowDepth)
                {
                    m_moddingDepthBufferAllowed = false;
                }
                if ((uint32_t)m_moddingSinglePlayer.bufferUse & (uint32_t)ModdingBuffersUse::kDisallowHDR)
                {
                    m_moddingHDRBufferAllowed = false;
                }
                if ((uint32_t)m_moddingSinglePlayer.bufferUse & (uint32_t)ModdingBuffersUse::kDisallowHUDless)
                {
                    m_moddingHUDlessBufferAllowed = false;
                }
            }
        }
    }

    if (filtersLwrrentlyAllowed)
    {
        m_enableAnselModding = m_activeControlClient->isModdingAllowed();
        // Reset flag to show message if network activity will be detected again
        m_shownNetworkActivityMsg = false;
    }
    else
        m_enableAnselModding = false;

    ANSEL_PROFILE_STOP(ff01_processAnselStatus);

    bool forceToggleAnsel = false;
    if (m_specialClientActive)
    {
        if (m_bNextFrameForceEnableAnsel && !m_isClientEnabled)
        {
            m_UI->forceEnableUI();
            forceToggleAnsel = true;
            m_bNextFrameForceEnableAnsel = false;
        }
        if (m_bNextFrameForceDisableAnsel && m_isClientEnabled)
        {
            m_UI->forceDisableUI();
            forceToggleAnsel = true;
            m_bNextFrameForceDisableAnsel = false;
        }
        m_bNextFrameForceEnableAnsel = false;
        m_bNextFrameForceDisableAnsel = false;

        if (m_bNextFrameEnableFade)
        {
            m_bNextFrameEnableFade = false;
            m_UI->setFadeState(true);
        }
        if (m_bNextFrameDisableFade)
        {
            m_bNextFrameDisableFade = false;
            m_UI->setFadeState(false);
        }
    }

    bool applyingFilters = false;

    // If we only disable depth when network activity detected, we need to go the full path every time m_allowFiltersInGame is true
    // otherwise (if we disallow all the filtering when network actrivity detected) - we only go the full path if there is no network activity
    if (filtersLwrrentlyAllowed)
    {
        forceNotSkipRestFrame = forceToggleAnsel || forceNotSkipRestFrame || (m_activeControlClient->isAnselPrestartRequested() || m_activeControlClient->isAnselStartRequested()) || m_enableAnselModding;
        applyingFilters = m_isClientEnabled || m_enableAnselModding;
    }
    else
    {
        forceNotSkipRestFrame = forceToggleAnsel || forceNotSkipRestFrame || (m_activeControlClient->isAnselPrestartRequested() || m_activeControlClient->isAnselStartRequested());
        applyingFilters = m_isClientEnabled;
    }

    // If nothing needs to be done on our side - skip the shared resource code completely
    // The logic should be as follow: if Ansel is not used as either Camera or Modding(Filtering)
    //      then we can safely ignore everything that is past this condition
    bool skipAnselProcessing =
        !forceNotSkipRestFrame && !m_isClientActive && (!applyingFilters || !m_bRunShaderMod || !m_effectsInfo.m_selectedEffect);

    ANSEL_PROFILE_START(ff02_notifications, "Notifications");

    if (m_specialClientActive && m_allowNotifications)
    {
        if (getIPCModeEnabled() != 0)
        {
        }
        else
        {
            // Add greeting notification, only need to do this once
            if (m_enableWelcomeNotification)
            {
                static int frameCount = 0;
                if (frameCount >= 0)
                {
                    ++frameCount;
                    if (frameCount > 100)
                    {
                        HMODULE hAnselSDKModule = m_anselSDK.findAnselSDKModule();
                        if (hAnselSDKModule)
                        {
                            AnselUI * UI_standalone = static_cast<AnselUI *>(m_UI);

                            ErrorManager::ErrorEntry anselNotification;
                            anselNotification.lifeTime = 5.0f;
                            anselNotification.message = m_toggleHotkeyComboText;
                            UI_standalone->addGameplayOverlayNotification(AnselUIBase::NotificationType::kWelcome, anselNotification, true);
                        }
                        else
                        {
                            if (!m_welcomeNotificationLogged)
                            {
                                LOG_DEBUG("Notifications are not shown because AnselSDK integration is not detected");
                                m_welcomeNotificationLogged = true;
                            }
                        }
                        frameCount = -1;
                    }
                }
            }

            AnselUI * UI_standalone = static_cast<AnselUI *>(m_UI);

            const bool allowGameplayOverlay = UI_standalone->isGameplayOverlayRequired() && (skipAnselProcessing || m_bWasAnselDeactivated);

            if (allowGameplayOverlay)
            {
                m_bufDB.Present().setClientResource(hPresentResource);

                HRESULT status = S_OK;
                // There are two ways to report buffer state from the acquire
                //  First is direct value return from the function, it should only has errors which should prevent Ansel
                //  from further operations from ANY buffer, e.g. if mutex acquire failed - this means something
                //  really bad happened and we better stop Ansel
                //  Second - check for "m_isReadyToUse", this allows to distinguish between vital and collateral buffers
                if (!SUCCEEDED( status = m_bufDB.Present().acquire(subResIndex) ))
                {
                    return status;
                }

                // If presentable buffer manipulations failed, we cannot recover (vital buffer)
                if (!m_bufDB.Present().isBufferReadyToUse())
                {
                    std::string errStr = "Presentable buffer preparation failed on gameplay overlay";
                    HRESULT status = E_FAIL;

                    if (m_bufDB.Present().m_bIlwalidResSize)
                    {
                        LOG_WARN(errStr.c_str());
                        // Ansel doesn't need deactivation due to invalid resource size which may be passed down by the app in case of Alt-Tab scenarios
                        // to switch out of the game as one would want to continue using Ansel when switching back to the game.
                        status = S_OK;
                    }
                    else
                    {
                        LOG_FATAL(errStr.c_str());
                        status = E_FAIL;
                    }

                    return status;
                }

                // We always need to copy presentable back to the client
                //  unless we're doing some sort of a passthrough
                m_bufDB.Present().m_needCopyToClient = true;

                AnselResource * presentResourceData = m_d3d11Interface.lookupAnselResource(hPresentResource);
                AnselResource * pPresentResourceData = m_bufDB.Present().getAnselResource();

                AnselRenderBufferReleaseHelper persentableReleaseHelper(&m_bufDB.Present());

                {
                    // Copy over the old data
                    AnselEffectState* pColorEffect = &m_passthroughEffect;
                    static FLOAT clearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

                    D3D11_VIEWPORT viewPort;
                    viewPort.Width = float(pPresentResourceData->toServerRes.width);
                    viewPort.Height = float(pPresentResourceData->toServerRes.height);
                    viewPort.MinDepth = 0.0f;
                    viewPort.MaxDepth = 1.0f;
                    viewPort.TopLeftX = 0;
                    viewPort.TopLeftY = 0;

                    m_immediateContext->RSSetViewports(1, &viewPort);
                    m_immediateContext->ClearRenderTargetView(pPresentResourceData->toClientRes.pRTV, clearColor);

                    UINT ZERO = 0;
                    UINT vertexStride = 20;
                    m_immediateContext->VSSetShader(pColorEffect->pVertexShader, NULL, 0);
                    m_immediateContext->PSSetShader(pColorEffect->pPixelShader, NULL, 0);

                    m_immediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                    m_immediateContext->PSSetShaderResources(0, 1, &pPresentResourceData->toServerRes.pSRV);

                    m_immediateContext->IASetInputLayout(0);
                    m_immediateContext->PSSetSamplers(0, 1, &pColorEffect->pSamplerState);
                    m_immediateContext->RSSetState(pColorEffect->pRasterizerState);
                    m_immediateContext->OMSetDepthStencilState(pColorEffect->pDepthStencilState, 0xFFFFFFFF);
                    m_immediateContext->OMSetRenderTargets(1, &pPresentResourceData->toClientRes.pRTV, NULL);
                    m_immediateContext->OMSetBlendState(pColorEffect->pBlendState, NULL, 0xffffffff);
                    m_immediateContext->Draw(3, 0);
                }

                UI_standalone->renderGameplayOverlay(
                    m_globalPerfCounters.dt,
                    m_immediateContext,
                    presentResourceData,
                    &m_passthroughEffect
                    );

                m_bufDB.Present().release();
            }
        }
    }

#if ANSEL_SIDE_PRESETS
#if IPC_ENABLED == 1
    if (m_UI)
    {
        if (getIPCModeEnabled() != 0)
        {
            UIIPC * UI_IPC = static_cast<UIIPC *>(m_UI);

            if (UI_IPC->m_state.filters.presetErrors.size() > 0)
            {
                for (auto error : UI_IPC->m_state.filters.presetErrors)
                {
                    m_displayMessageStorage.resize(0);
                    m_displayMessageStorage.push_back(error.filterId);
                    m_displayMessageStorage.push_back(error.message);

                    AnselUIBase::MessageType messageType;
                    switch (error.status)
                    {
                    case AnselIpc::Status::kAlreadySet:
                        //kAlreadySet is treated as a parsing error, to ensure that the file is removed.
                    case AnselIpc::Status::kErrorParsingFile:
                        messageType = AnselUIBase::MessageType::kErrorParsingFile;
                        break;
                    case AnselIpc::Status::kEffectRequiresDepth:
                        messageType = AnselUIBase::MessageType::kEffectRequiresDepth;
                        break;
                    default:
                        LOG_ERROR("Invalid status for Preset Error: %d", error.status);
                        messageType = AnselUIBase::MessageType::kNone;
                        break;
                    }

                    if (messageType != AnselUIBase::MessageType::kNone)
                    {
                        m_activeControlClient->displayMessage(messageType, m_displayMessageStorage, true);
                    }
                }

                UI_IPC->m_state.filters.presetErrors.clear();
            }
        }
    }
#endif
#endif

    ANSEL_PROFILE_STOP(ff02_notifications);

    ///////////////////////////

    if (skipAnselProcessing || !m_isAnselActive)
    {
        if (!m_bWasAnselDeactivated)
        {
            // Between the end of the last frame with Ansel active and the start of this one, buffer handles will have been selected.
            // However, by the time Ansel next runs, these buffers could be invalid (if the location of the buffer changes,
            // for example, after a resolution change). As such, they should be released, so they will not be used for that frame.
            // This only needs to happen on the first frame after disabling, since after that, extraction checks are disabled.
            m_bufDB.Final().release(true /* forceNoCopy */);
            m_bufDB.HDR().release(true);
            m_bufDB.Depth().release(true);
            m_bufDB.Hudless().release(true);
        }
        m_bWasAnselDeactivated = true;
        m_enableDepthExtractionChecks = false;
        m_enableHDRExtractionChecks = false;
        m_enableHUDlessExtractionChecks = false;
        m_enableFinalColorExtractionChecks = false;
        return S_OK;
    }
    else
    {
        m_enableDepthExtractionChecks = true;
        m_enableHDRExtractionChecks = true;
        m_enableHUDlessExtractionChecks = true;
        m_enableFinalColorExtractionChecks = true;
    }

    // Check to force extraction check disablement. We can only allow extraction of a buffer type if that type is not found in the m_denylistedBufferExtractionTypes set.
    // Only prevent extraction if we are not in a fully Ansel Integrated Photo Mode Session (no-SDK session)
    if (!m_anselSDK.isDetected() || !m_anselSDK.isSessionActive())
    {
#define CheckBufferExtractionDenylist(bufferType, enablementBool, denylist) enablementBool &= (denylist.find(bufferType) == denylist.end())
        CheckBufferExtractionDenylist(ansel::BufferType::kBufferTypeDepth, m_enableDepthExtractionChecks, m_denylistedBufferExtractionTypes);
        CheckBufferExtractionDenylist(ansel::BufferType::kBufferTypeHDR, m_enableHDRExtractionChecks, m_denylistedBufferExtractionTypes);
        CheckBufferExtractionDenylist(ansel::BufferType::kBufferTypeHUDless, m_enableHUDlessExtractionChecks, m_denylistedBufferExtractionTypes);
        CheckBufferExtractionDenylist(ansel::BufferType::kBufferTypeFinalColor, m_enableFinalColorExtractionChecks, m_denylistedBufferExtractionTypes);
    }

    if (m_moddingHashVerif0PrepStep == static_cast<uint32_t>(PrepackagedEffects::kNUM_ENTRIES)*3+1)
    {
        ModdingSettings moddingSinglePlayer;
        ModdingSettings moddingMultiPlayer;
        parseModdingModes(m_freeStyleModeValueVerif, &moddingSinglePlayer, &moddingMultiPlayer);

        bool spMatch =
            (moddingSinglePlayer.mode == m_moddingSinglePlayer.mode) &&
            (moddingSinglePlayer.bufferUse == m_moddingSinglePlayer.bufferUse) &&
            (moddingSinglePlayer.restrictedEffectSetId == m_moddingSinglePlayer.restrictedEffectSetId);

        bool mpMatch =
            (moddingMultiPlayer.mode == m_moddingMultiPlayer.mode) &&
            (moddingMultiPlayer.bufferUse == m_moddingMultiPlayer.bufferUse) &&
            (moddingMultiPlayer.restrictedEffectSetId == m_moddingMultiPlayer.restrictedEffectSetId);

        if (!spMatch || !mpMatch)
        {
            // Something tampered with the hash, need to permanently disable Ansel
            if (m_anselSDK.isSDKDetectedAndSessionActive())
            {
                m_anselSDK.stopClientSession();
                m_activeControlClient->emergencyAbort();
            }
            reportFatalError(__FILE__, __LINE__, FatalErrorCode::kModsSettingFail, "Deactivating LwCamera due to modding settings mismatch");
            deactivateAnsel();
            return S_OK;
        }

        // Restart the verification cycle
        m_moddingHashVerif0PrepStep = 0;
    }
    else if (m_moddingHashVerif0PrepStep == static_cast<uint32_t>(PrepackagedEffects::kNUM_ENTRIES)*3)
    {
        const Hash::Data verifHash = m_moddingEffectHashDB.GetGeneratedMainHash();
        if (!m_moddingEffectHashDB.CompareMainHash(verifHash))
        {
            // Something tampered with the hash, need to permanently disable Ansel
            if (m_anselSDK.isSDKDetectedAndSessionActive())
            {
                m_anselSDK.stopClientSession();
                m_activeControlClient->emergencyAbort();
            }
            reportFatalError(__FILE__, __LINE__, FatalErrorCode::kHashFail, "Deactivating LwCamera due to hash mismatch");
            deactivateAnsel();
            return S_OK;
        }
        ++m_moddingHashVerif0PrepStep;
    }
    else
    {
        ++m_moddingHashVerif0PrepStep;
    }

    // initialize network detection at the first Present. It's ok to call this every frame, because it's going to early exit
    // if it's already initialized
#if ENABLE_NETWORK_HOOKS == 1
    m_networkDetector.initialize(m_checkTraficLocal);
    // usage: m_networkDetector.isActivityDetected();
    m_networkActivityDetected = m_networkDetector.isActivityDetected();
#endif
    m_anselSDK.setSessionData(sessionStartFunc, sessionStopFunc, this);

    ANSEL_PROFILE_START(ff03_dormantUpdate, "Dormant Update");

    if (m_specialClientActive)
    {
        if (getIPCModeEnabled() != 0)
        {
        }
        else
        {
            // We need this needToAbortCapture, since it doesn't fall into typical Ansel routine.
            //  Dormant mode is Standalone-specific, and is a special case of frame skipping, so
            //  we won't get to the pipelined abortCapture check

            bool needToAbortCapture = false, needToSkipRestFrame = false;
            AnselUI * UI_standalone = static_cast<AnselUI *>(m_UI);
            UI_standalone->dormantUpdate(m_anselSDK, &needToAbortCapture, &needToSkipRestFrame);

            if (needToAbortCapture)
                abortCapture();

            if (needToSkipRestFrame)
                return S_OK;
        }
    }

    ANSEL_PROFILE_STOP(ff03_dormantUpdate);

    ANSEL_PROFILE_START(ff04_checkToggle, "Check Toggle CamWorks");

    if (m_specialClientActive)
    {
        if (getIPCModeEnabled() != 0)
        {
        }
        else
        {
            AnselUI * UI_standalone = static_cast<AnselUI *>(m_UI);
            UI_standalone->checkToggleCamWorks();
            UI_standalone->setShowMouseWhileDefolwsed(m_showMouseWhileDefolwsed);
        }
    }

    ANSEL_PROFILE_STOP(ff04_checkToggle);

    ANSEL_PROFILE_START(ff05_UI_update, "UI Update");

    m_activeControlClient->update(m_globalPerfCounters.dt);

    ANSEL_PROFILE_STOP(ff05_UI_update);

    using lwanselutils::appendTimeW;

    ANSEL_PROFILE_START(ff06_UI_processInput, "UI Process Input");

    if (m_specialClientActive)
    {
        if (getIPCModeEnabled() != 0)
        {
        }
        else
        {
            AnselUI * UI_standalone = static_cast<AnselUI *>(m_UI);
            UI_standalone->setAreMouseButtonsSwapped(m_areMouseButtonsSwapped);
            UI_standalone->processInputState(m_anselSDK, (float)m_globalPerfCounters.dt);
        }
    }

    ANSEL_PROFILE_STOP(ff06_UI_processInput);

    ANSEL_PROFILE_START(ff07_effectRebuild, "effectRebuild");

    bool effectChanged = false;
    bool styleChanged = false;

    //const int prevSelectedEffect = m_selectedEffect;
    const size_t prevEffectsNum = m_effectsInfo.m_effectSelected.size();
    const size_t effectsNum = m_activeControlClient->getLwrrentFilterNum();
    bool filterInfoQueried = false;

    int numPrevActiveEffects = 0;
    for (size_t effIdx = 0; effIdx < prevEffectsNum; ++effIdx)
    {
        const int prevSelectedEffect = m_effectsInfo.m_effectSelected[effIdx];
        if (prevSelectedEffect != 0)
        {
            ++numPrevActiveEffects;
        }
    }

    resizeEffectsInfo(effectsNum);

    // In case we still require effect stack rebuilding, we don't know if it requires depth buffer or not
    //  so we will have to copy it just in case
    if (m_stackRebuildRequired)
    {
        m_depthBufferUsed = true;
        m_hudlessBufferUsed = true;
        m_hdrBufferUsedByFilter = true;
    }

    int numActiveEffects = 0;
    for (size_t effIdx = 0; effIdx < effectsNum; ++effIdx)
    {
        const int prevSelectedEffect = (effIdx < prevEffectsNum) ? m_effectsInfo.m_effectSelected[effIdx] : 0;
        const int effectSelected = int(getFilterIndexById(m_activeControlClient->getLwrrentFilter(effIdx)));
        filterInfoQueried = filterInfoQueried || m_activeControlClient->getLwrrentFilterInfoQuery(effIdx);
        m_effectsInfo.m_effectSelected[effIdx] = effectSelected;

        if (effectSelected != 0)
        {
            ++numActiveEffects;
        }
        if ((effectSelected != prevSelectedEffect) || (effIdx != m_activeControlClient->getLwrrentFilterOldStackIdx(effIdx)))
        {
            m_effectsInfo.m_effectRebuildRequired[effIdx] = true;
            effectChanged = true;
            m_bNextFrameRebuildYAML = true;

            // In case effect was changed, we don't know if it requires depth buffer or not
            //  so we will have to copy it just in case
            m_depthBufferUsed = true;
            m_hudlessBufferUsed = true;
            m_hdrBufferUsedByFilter = true;
        }

        shadermod::MultiPassEffect* eff = (effIdx < m_effectsInfo.m_effectsStack.size()) ? m_effectsInfo.m_effectsStack[effIdx] : nullptr;
        if (m_activeControlClient->getLwrrentFilterResetValues(effIdx))
        {
            if (eff)
            {
                shadermod::ir::UserConstantManager::Range ucrange = eff->getUserConstantManager().getPointersToAllUserConstants();

                for (; ucrange.begin < ucrange.end; ucrange.begin++)
                {
                    shadermod::ir::UserConstant * lwrUC = *ucrange.begin;
                    lwrUC->setValue(lwrUC->getDefaultValue());
                }
            }
            m_activeControlClient->lwrrentFilterResetValuesDone(effIdx);
        }
    }

    if (numActiveEffects != numPrevActiveEffects)
    {
        effectChanged = true;
        m_bNextFrameRebuildYAML = true;
        m_depthBufferUsed = true;
        m_hudlessBufferUsed = true;
        m_hdrBufferUsedByFilter = true;
    }

    if (effectChanged)
    {
        userConstantStorage.oldStackIdx.resize(0);
        for (size_t effIdx = 0; effIdx < effectsNum; ++effIdx)
        {
            int oldStackIdx = m_activeControlClient->getLwrrentFilterOldStackIdx(effIdx);
            if (oldStackIdx == AnselUIBase::oldStackIndexCreated)
                oldStackIdx = -1;
            userConstantStorage.oldStackIdx.push_back(oldStackIdx);
        }
        m_activeControlClient->updateLwrrentFilterStackIndices();

        if (numActiveEffects != 0 &&  // There are active effects, and
            m_lightweightShimMode)// the shim is lwrrently in lightweight mode.
        {
            exitLightweightMode();
        }
        else if (numActiveEffects == 0 &&   // There are no active effects, and
            !m_isClientActive &&            // the UI is not active, and
            !m_lightweightShimMode)         // the shim is not lwrrently in lightweight mode.
        {
            enterLightweightMode();
        }

        m_hadNoActiveEffects = (numActiveEffects == 0);
    }

    ANSEL_PROFILE_STOP(ff07_effectRebuild);

    ANSEL_PROFILE_START(ff08_buffersExtraction, "Buffers Extraction");

    HRESULT status = S_OK;

    m_bufDB.Present().setClientResource(hPresentResource);

    // There are two ways to report buffer state from the acquire
    //  First is direct value return from the function, it should only has errors which should prevent Ansel
    //  from further operations from ANY buffer, e.g. if mutex acquire failed - this means something
    //  really bad happened and we better stop Ansel
    //  Second - check for "m_isReadyToUse", this allows to distinguish between vital and collateral buffers
    LOG_DEBUG("Acquiring Present Buffer...");
    if (!SUCCEEDED( status = m_bufDB.Present().acquire(subResIndex) ))
    {
        LOG_DEBUG("Failed to acquire Presentable buffer");
        return status;
    }

    // If presentable buffer manipulations failed, we cannot recover (vital buffer)
    if (!m_bufDB.Present().isBufferReadyToUse())
    {
        std::string errStr = "Presentable buffer preparation failed on gameplay overlay";
        HRESULT status = E_FAIL;

        if (m_bufDB.Present().m_bIlwalidResSize)
        {
            LOG_WARN(errStr.c_str());
            // Ansel doesn't need deactivation due to invalid resource size which may be passed down by the app in case of Alt-Tab scenarios
            // to switch out of the game as one would want to continue using Ansel when switching back to the game.
            status = S_OK;
        }
        else
        {
            LOG_FATAL(errStr.c_str());
            status = E_FAIL;
        }

        return status;
    }

    // We always need to copy presentable back to the client
    //  unless we're doing some sort of a passthrough
    m_bufDB.Present().m_needCopyToClient = true;

    // For these buffers we're OK if some operations with the buffers failed (collateral buffers)
    //  however, if acquire returned error, this means we absolutely need to disable Ansel
    LOG_DEBUG("Acquiring Final Color buffer...");
    if (!SUCCEEDED( status = m_bufDB.Final().acquire(0) ))
    {
        LOG_DEBUG("Failed to acquire Final Color buffer");
        return status;
    }
    LOG_DEBUG("Acquiring Depth Buffer...");
    if (!SUCCEEDED( status = m_bufDB.Depth().acquire(0) ))
    {
        LOG_DEBUG("Failed to acquire Depth buffer");
        return status;
    }
    LOG_DEBUG("Acquiring HUDless Buffer...");
    if (!SUCCEEDED( status = m_bufDB.Hudless().acquire(0) ))
    {
        LOG_DEBUG("Failed to acquire HUDless buffer");
        return status;
    }
    LOG_DEBUG("Acquiring HDR Buffer...");
    if (!SUCCEEDED( status = m_bufDB.HDR().acquire(0) ))
    {
        LOG_DEBUG("Failed to acquire HDR buffer");
        return status;
    }

    const DXGI_FORMAT bufferPresentableFormat = m_bufDB.Present().getAnselResource()->toServerRes.format;
    const DXGI_FORMAT bufferPresentablecolwertedFormat = lwanselutils::colwertFromTypelessIfNeeded(bufferPresentableFormat);
    const bool isBufferPresentableHDR = isHdrFormatSupported(bufferPresentablecolwertedFormat);
    LOG_DEBUG_AnselBufferDetails(&m_bufDB.Present());
    LOG_DEBUG_AnselBufferDetails(&m_bufDB.Final());
    LOG_DEBUG_AnselBufferDetails(&m_bufDB.Depth());
    LOG_DEBUG_AnselBufferDetails(&m_bufDB.Hudless());
    LOG_DEBUG_AnselBufferDetails(&m_bufDB.HDR());

    bool isRawHDRBufferAvailable = m_hdrBufferAvailable;
    const bool isRawHDRBufferCaptured = m_bufDB.HDR().getAnselResource() ? true : false;

    AnselResource * pPresentResource = m_bufDB.Present().getAnselResource();
    AnselResource * pFinalColorResource = m_bufDB.Final().getAnselResource();

    // Update UI client if HDR buffers are available or not
    if (m_anselSDK.isDetected())
    {
        if (m_anselSDK.isSessionActive())
        {
            // If the SDK integrated game does not allow raw HDR capture, then we will not capture any raw HDR buffers, even if they are available.
            if (!m_anselSDK.getSessionConfiguration().isRawAllowed)
            {
                isRawHDRBufferAvailable = false;
            }

            if (m_sessionFrameCount == 2) // We need to wait at least 1 frame in order to process all the buffers, and know what buffers we have to work with (eg raw HDR buffers).
            {
                bool isHDRCaptureAvailable = isBufferPresentableHDR || isRawHDRBufferAvailable;
                m_activeControlClient->setShotTypePermissions(isHDRCaptureAvailable, NULL, -1);

                // If an SDK Integrated app is allowing raw HDR capture as part of their integration, but we are not actually capturing any raw HDR buffers, log this error.
                if (m_anselSDK.getSessionConfiguration().isRawAllowed && !m_bufDB.HDR().getAnselResource())
                {
                    LOG_WARN("Warning: SDK integrated app has set HDR isRawAllowed to true, but HDR buffers are not available. Disabling HDR capture.");
                }
            }
        }
    }
    else
    {
        // In the case of ansel-lite, aka no SDK integration, we have to update shot permissions to reflect if we have access to HDR buffers or not.
        bool isHDRBufferAvailableForCapture = isBufferPresentableHDR || isRawHDRBufferAvailable;
        m_activeControlClient->setShotTypePermissions(isHDRBufferAvailableForCapture, NULL, -1);
    }

    // This will point to a resource which we want to use to save captures, strictly
    //  it could be different from the presentable resource
    AnselResource * pSelectedCaptureResource = pPresentResource;
    if (m_bufDB.Final().getClientResource() != nullptr)
    {
        pSelectedCaptureResource = pFinalColorResource;
    }

    // This will point to the resource which will be used for filter processing
    //  e.g. if capture resource is different from presentable resource, we might want to
    //  process presentable buffer when capture is not in process, and capture resource otherwise
    // This resource will be selected below
    AnselResource * pSelectedFilteringResource = nullptr;

    AnselResource * pDepthResource = m_bufDB.Depth().getAnselResource();
    AnselResource * pHDRResource = m_bufDB.HDR().getAnselResource();
    const AnselResourceData * pHUDlessResourceData = m_bufDB.Hudless().getValidResourceData();

    // These helpers will call 'release' for tracked buffers upon destruction
    //  typically this will not do anything, since in a common scenario release
    //  will already happen in manual calls at the end of the finalizeFrame;
    //  however this wouldn't be the case when Ansel does early exit from the function
    //  this is where those helpers will do the job
    AnselRenderBufferReleaseHelper persentableReleaseHelper(&m_bufDB.Present());
    AnselRenderBufferReleaseHelper finalColorReleaseHelper(&m_bufDB.Final());
    AnselRenderBufferReleaseHelper depthReleaseHelper(&m_bufDB.Depth());
    AnselRenderBufferReleaseHelper hdrReleaseHelper(&m_bufDB.HDR());
    AnselRenderBufferReleaseHelper hudlessReleaseHelper(&m_bufDB.Hudless());

    ANSEL_PROFILE_STOP(ff08_buffersExtraction);

    if (getNumEffects() > 0 || m_bNextFrameRebuildYAML)//(effectsNum > 0)
    {
        m_bRunShaderMod = true;
    }
    else
    {
        m_bRunShaderMod = false;
    }

    if (!m_bShaderModInitialized)
    {
        //OutputDebugStringA("Loaded effects:\n");
        if (0)
        for (size_t i = 0u, iend = m_effectsInfo.m_effectFilesList.size(); i < iend; ++i)
        {
            char effect[256];
            sprintf_s(effect, 256, "%zd. %ws\n", i, m_effectsInfo.m_effectFilesList[i].c_str());
            //OutputDebugStringA(effect);
        }
        m_bShaderModInitialized = true;
    }

    int newWidth = pPresentResource->toServerRes.width;
    int newHeight = pPresentResource->toServerRes.height;
    if (newWidth != m_prevWidth || newHeight != m_prevHeight)
    {
        m_activeControlClient->setScreenSize(newWidth, newHeight);
        m_prevWidth = newWidth;
        m_prevHeight = newHeight;
    }
    if (m_activeControlClient->isHighResolutionRecalcRequested())
    {
        // Callwlate the multiplier
        unsigned int maxSize = (newWidth > newHeight) ? newWidth : newHeight;
        int32_t maxMultiplier = (m_maxHighResResolution * 1024) / maxSize;

        if (maxMultiplier > s_maxHighresMultiplier)
            maxMultiplier = s_maxHighresMultiplier;
        else if (maxMultiplier < 4)
        {
            // IMPORTANT: we can NOT use s_minHighresMultiplier here
            //  since modifying it to be (x4) will cause incorrect work of the CameraWorks parts
            //  and using as-is value (x2) will not leave any notches on the slider, that would be edge case
            //      one way of avoiding this edge case - is to limit the min highres multiplier to 4
            maxMultiplier = 4;
        }

        // Multiplier of 1 doesn't count
        if (maxMultiplier)
            --maxMultiplier;

        uint32_t width = newWidth;
        uint32_t height = newHeight;

        darkroom::ShotDescription desc;
        desc.type = darkroom::ShotDescription::EShotType::HIGHRES;
        desc.bmpWidth = width & ~1u; // a bit of cheating for odd tile height (for now);
        desc.bmpHeight = height;
        desc.panoWidth = 0;
        desc.horizontalFov = 90.f;

        std::vector<AnselUIBase::HighResolutionEntry> highResEntries;
        for (int32_t resCnt = 0; resCnt < maxMultiplier; ++resCnt)
        {
            AnselUIBase::HighResolutionEntry hrEntry;

            uint64_t highResWidth = (resCnt + 2) * (width & ~1u); // a bit of cheating for odd tile height (for now);
            uint64_t highResHeight = (resCnt + 2) * height;

            desc.highresMultiplier = (int)resCnt + 2;
            darkroom::CaptureTaskEstimates estimates;
            estimates = darkroom::CameraDirector::estimateCaptureTask(desc);

            hrEntry.width = highResWidth;
            hrEntry.height = highResHeight;
            hrEntry.byteSize = estimates.inputDatasetSizeTotalInBytes + estimates.outputSizeInBytes;

            highResEntries.push_back(hrEntry);
        }

        m_activeControlClient->highResolutionRecalcDone(highResEntries);
    }


    AnselUIBase::ShotDesc shotDesc = m_activeControlClient->shotCaptureRequested();
    m_shotToTake = shotDesc.shotType;

    bool wasShotRegular = false;
    if (m_shotToTake != ShotType::kNone)
    {
        m_makeScreenshotHDR = m_activeControlClient->isShotEXR();
        m_makeScreenshotHDRJXR = false;
        if (m_shotToTake == ShotType::kRegularUI)
        {
            m_makeScreenshot = false;
            m_makeScreenshotWithUI = true;
            wasShotRegular = true;
            m_shotToTake = ShotType::kNone;
        }
        else if (m_shotToTake == ShotType::kRegular)
        {
            m_makeScreenshot = true;
            m_makeScreenshotWithUI = false;
            wasShotRegular = true;
            m_shotToTake = ShotType::kNone;

            // We export an hdr image in JXR format whenever an hdr buffer is available, but not when we want to export EXR.
            m_makeScreenshotHDRJXR = (!m_makeScreenshotHDR &&
                                      m_activeControlClient->isShotJXR() &&
                                      (isRawHDRBufferAvailable || isBufferPresentableHDR));
        }

        if (wasShotRegular)
        {
            // TODO probably unify this for each shot type, lwrrently for multipart shots it is done in handleCaptureTaskStartup
            //  and for regulars - here, since that part in handle capture startup only happens when director has task
            m_anselSDK.setProgressLwrrent(0);
        }
    }

    // Remap capture states
    //  it is OK to check SDK capture state here for multi-part shots, since the frame we launched multi-part, the file couldn't actually be saved, so
    //  we can process it in the shader as we want, it won't get to the final image.
    //  So this will actually happen one frame AFTER the multi-part shot was triggered, and this agrees with the logic behind multi-part shots.
    //  However, singular shots (e.g. regular) need to have additional processing here, since they are not properly reflected in the SDK capture state.
    bool isMultiPartCapture = false;
    int captureStateShader = SHADER_CAPTURE_NOT_STARTED;
    {
        switch (m_anselSDK.getCaptureState())
        {
        case CAPTURE_REGULAR:
        {
            captureStateShader = SHADER_CAPTURE_REGULAR;
            break;
        }
        case CAPTURE_REGULARSTEREO:
        {
            captureStateShader = SHADER_CAPTURE_REGULARSTEREO;
            isMultiPartCapture = true;
            break;
        }
        case CAPTURE_HIGHRES:
        {
            captureStateShader = SHADER_CAPTURE_HIGHRES;
            isMultiPartCapture = true;
            break;
        }
        case CAPTURE_360:
        {
            captureStateShader = SHADER_CAPTURE_360;
            isMultiPartCapture = true;
            break;
        }
        case CAPTURE_360STEREO:
        {
            captureStateShader = SHADER_CAPTURE_360STEREO;
            isMultiPartCapture = true;
            break;
        }
        default:
        {
            captureStateShader = SHADER_CAPTURE_NOT_STARTED;
            break;
        }
        }

        // Those two don't get delivered via the SDK capture state
        if (wasShotRegular)
            captureStateShader = SHADER_CAPTURE_REGULAR;
        if (m_shotToTake == ShotType::kStereo)
        {
            captureStateShader = SHADER_CAPTURE_REGULARSTEREO;
            isMultiPartCapture = true;
        }

        setCaptureState(captureStateShader);
    }

    // By default, we want to process buffer we're looking at (presentable)
    pSelectedFilteringResource = pPresentResource;
    if (captureStateShader != SHADER_CAPTURE_NOT_STARTED)
    {
        // If capture is in progress, we want to process capture buffer if it is present
        if (m_bufDB.Final().getClientResource() != nullptr)
            pSelectedFilteringResource = pFinalColorResource;
    }

    shadermod::MultipassConfigParserError compileErr(shadermod::MultipassConfigParserErrorEnum::eOK);

    DXGI_FORMAT inputColorFormat = pSelectedFilteringResource->toServerRes.format;
    ID3D11Texture2D * inputColorTexture = pSelectedFilteringResource->toServerRes.pTexture2D;

    //TODO: set the depth width and height separately if needed ever
    // We need to keep the rebuild state in case Ansel filtering is not yet working

    ID3D11Texture2D * resolvedDepthTexture = nullptr;
    DXGI_FORMAT resolvedDepthFormat = DXGI_FORMAT_UNKNOWN;

    // Set of variables that set buffer state for server (general buffer availability)
    //  it is not passed directly to effects/shaders though, since server and effect framework treat certain cases differently
    bool depthAvailable = false, hdrAvailable = false, hudlessAvailable = false;

    if (m_depthBufferUsed && pDepthResource)
    {
        if (!SUCCEEDED( status = m_renderBufferColwerter.getResolvedDepthTexture(pDepthResource->toServerRes, &resolvedDepthTexture, &resolvedDepthFormat) ))
        {
            LOG_ERROR("Depth resolve failed while depth resource is present (format: %d, s/q: %d %d)", pDepthResource->toServerRes.format, pDepthResource->toServerRes.sampleCount, pDepthResource->toServerRes.sampleQuality);
        }
    }
    else
    {
        // Ensure Render Buffer Colwerter's Viewport data is reset so depth buffer gets properly loaded next time we need it
        m_renderBufferColwerter.resetViewport();
    }

    ID3D11Texture2D * inputDepthTexture = resolvedDepthTexture;
    DXGI_FORMAT inputDepthFormat = resolvedDepthFormat;

    ANSEL_PROFILE_START(ff09_effectPreparation, "Effect Preparation");

    // We don't need to save HUDless/HDR buffer formats, as we don't send it as telemetry now
    saveBufferFormats(inputColorFormat, (m_depthBufferUsed && pDepthResource) ? pDepthResource->toServerRes.format : DXGI_FORMAT_UNKNOWN);

    shadermod::ir::Effect::InputData finalColorInput, depthInput, hudlessInput, hdrInput;

    finalColorInput.width = pSelectedFilteringResource->toServerRes.width;
    finalColorInput.height = pSelectedFilteringResource->toServerRes.height;
    finalColorInput.format = shadermod::ir::ircolwert::DXGIformatToFormat(inputColorFormat);
    finalColorInput.texture = inputColorTexture;

    // Grab information from the BufferTestingOptions filter, if it's on the stack
    bool bufferOptionsChanged = m_bufTestingOptionsFilter.checkFilter(m_depthBufferUsed, m_hudlessBufferUsed);

    bool ignoreZeroTextureSwitch = false;
    if (m_depthBufferUsed && pDepthResource && !bufferOptionsChanged)
    {
        depthInput.width = pDepthResource->toServerRes.width;
        depthInput.height = pDepthResource->toServerRes.height;
        depthInput.format = shadermod::ir::ircolwert::DXGIformatToFormat(inputDepthFormat);
        if (!isAnselSessionActive && !m_moddingDepthBufferAllowed)
        {
            LOG_DEBUG("Depth buffer requested, but not set due to active restrictions");

            depthInput.texture = nullptr;
            ignoreZeroTextureSwitch = true;
        }
        else
        {
            depthInput.texture = inputDepthTexture;
        }
        // Although depth texture is not set, trick server shader framework into thinking that
        //  depth is available, to avoid depth-related errors and effect unloads
        depthAvailable = true;
    }
    else
    {
        // If we want depth buffer, but it is not present (pDepthResource == 0),
        //  then we want to set initial sizes to something meaningful, in order for depth detection to work
        if (m_depthBufferUsed)
        {
            LOG_DEBUG("Depth buffer requested, but not available");

            depthInput.width = pPresentResource->toServerRes.width;
            depthInput.height = pPresentResource->toServerRes.height;
        }
        else
        {
            depthInput.width = -1;
            depthInput.height = -1;
        }
        depthInput.format = shadermod::ir::FragmentFormat::kNUM_ENTRIES;
        depthInput.texture = nullptr;

        depthAvailable = false;
    }

    if (m_hudlessBufferUsed)
    {
        if (pHUDlessResourceData)
        {
            hudlessInput.width = pHUDlessResourceData->width;
            hudlessInput.height = pHUDlessResourceData->height;
            hudlessInput.format = shadermod::ir::ircolwert::DXGIformatToFormat(pHUDlessResourceData->format);

            if (!isAnselSessionActive && !m_moddingHUDlessBufferAllowed)
            {
                LOG_DEBUG("HUDless buffer requested, but not set due to active restrictions");

                hudlessInput.texture = nullptr;
                ignoreZeroTextureSwitch = true;
            }
            else
            {
                hudlessInput.texture = pHUDlessResourceData->pTexture2D;
            }

            hudlessAvailable = true;
        }
        else
        {
            LOG_DEBUG("HUDless buffer requested, but not available - set to the final color");

            // Special code, if HUDless is unavailable, feed final color buffer
            //  this ensures diff being zero => masking efefct won't affect the picture at all
            hudlessInput.width = pPresentResource->toServerRes.width;
            hudlessInput.height = pPresentResource->toServerRes.height;
            hudlessInput.format = shadermod::ir::ircolwert::DXGIformatToFormat(inputColorFormat);
            hudlessInput.texture = inputColorTexture;

            hudlessAvailable = false;
        }
    }
    else
    {
        hudlessInput.width = -1;
        hudlessInput.height = -1;
        hudlessInput.format = shadermod::ir::FragmentFormat::kNUM_ENTRIES;
        hudlessInput.texture = nullptr;

        hudlessAvailable = false;
    }

    if (pHDRResource)
    {
        hdrInput.width = pHDRResource->toServerRes.width;
        hdrInput.height = pHDRResource->toServerRes.height;
        hdrInput.format = shadermod::ir::ircolwert::DXGIformatToFormat(pHDRResource->toServerRes.format);

        if (!isAnselSessionActive && !m_moddingHDRBufferAllowed)
        {
            LOG_DEBUG("HDR buffer requested, but not set due to active restrictions");

            hdrInput.texture = nullptr;
            ignoreZeroTextureSwitch = true;
        }
        else
        {
            hdrInput.texture = pHDRResource->toServerRes.pTexture2D;
        }

        hdrAvailable = true;
    }
    else
    {
        hdrInput.width = -1;
        hdrInput.height = -1;
        hdrInput.format = shadermod::ir::FragmentFormat::kNUM_ENTRIES;
        hdrInput.texture = nullptr;

        hdrAvailable = false;
    }

    setShaderBuffersAvailability(
        depthAvailable && (depthInput.texture != nullptr),
        hdrAvailable,
        hudlessAvailable
        );

    m_stackRebuildRequired |= (setInputs(
        finalColorInput, depthInput, hudlessInput, hdrInput,
        compileErr
        )
        && numActiveEffects != 0); // if numActiveEffects == 0, there's no stack to rebuild.

    ANSEL_PROFILE_STOP(ff09_effectPreparation);

    if (compileErr)
    {
        LOG_ERROR("Effect compilation error: %s", compileErr.getFullErrorMessage().c_str());
        OutputDebugStringA(compileErr.getFullErrorMessage().c_str());
    }

    ANSEL_PROFILE_START(ff10_UI_requests, "UI requests processing");

#if (ENABLE_CONTROL_SDK == 1)
    m_anselControlSDK.m_captureCameraInitialized = m_anselSDK.isCameraInitialized();
#endif

    if (m_activeControlClient->isAnselPrestartRequested())
    {
        // We call this function here in case game loads AnselSDK DLL late (after D3D device creation)
        AnselSDKState::DetectionStatus status = m_anselSDK.detectAndInitializeAnselSDK(m_installationFolderPath, m_intermediateFolderPath,
            m_snapshotsFolderPath, sessionStartFunc, sessionStopFunc, bufferFinishedFunc, this);

        if (status == AnselSDKState::DetectionStatus::kDRIVER_API_MISMATCH)
        {
            if (m_specialClientActive && m_allowNotifications)
            {
                if (getIPCModeEnabled() != 0)
                {
                }
                else
                {
                    AnselUI * UI_standalone = static_cast<AnselUI *>(m_UI);

                    ErrorManager::ErrorEntry anselNotification;
                    anselNotification.lifeTime = 5.0f;
                    UI_standalone->addGameplayOverlayNotification(AnselUIBase::NotificationType::kDrvUpdate, anselNotification, false);
                }
            }
        }

        m_isClientActive = true;
        if (m_lightweightShimMode)// the shim is lwrrently in lightweight mode.
        {
            exitLightweightMode(); // The UI is activating, so we exit lightweight mode.
        }

        LOG_INFO("Ansel::Prestart requested, SDK is %s", m_anselSDK.isDetected() ? "detected" : "not detected");

        m_activeControlClient->anselPrestartDone(AnselUIBase::Status::kOk, m_anselSDK.isDetected(), m_requireAnselSDK);
    }
    if (m_activeControlClient->isAnselStartRequested())
    {
        LOG_INFO("Ansel::Start requested");

        initRegistryDependentPathsAndOptions();
        auto status = startSession(filtersLwrrentlyAllowed, (uint32_t)ModdingRestrictedSetID::kEmpty, moddingMultiplayerMode ? (uint32_t)m_moddingMultiPlayer.restrictedEffectSetId : (uint32_t)m_moddingSinglePlayer.restrictedEffectSetId);

        if (status != AnselUIBase::Status::kDeclined)
        {
            m_isClientEnabled = true;
        }
        else
        {
            if (m_specialClientActive && m_allowNotifications)
            {
                if (getIPCModeEnabled() != 0)
                {
                }
                else
                {
                    AnselUI * UI_standalone = static_cast<AnselUI *>(m_UI);

                    ErrorManager::ErrorEntry anselNotification;
                    anselNotification.lifeTime = 5.0f;
                    UI_standalone->addGameplayOverlayNotification(AnselUIBase::NotificationType::kSessDeclined, anselNotification, false);
                }
            }
        }

        LOG_INFO("Ansel::Start status is ");
        if (status == AnselUIBase::Status::kDeclined)
            LOG_INFO("\tkDeclined");
        else if (status == AnselUIBase::Status::kOk)
            LOG_INFO("\tkOk");
        else if (status == AnselUIBase::Status::kOkAnsel)
            LOG_INFO("\tkOkAnsel");
        else if (status == AnselUIBase::Status::kOkFiltersOnly)
            LOG_INFO("\tkOkFiltersOnly");

        m_activeControlClient->anselStartDone(status);
    }
    if (m_activeControlClient->isAnselStopRequested())
    {
        LOG_INFO("Ansel::Stop requested");
        if (m_isClientEnabled)
        {
            if (m_anselSDK.isDetected())
            {
                // If some multipart capture is in progress, we want to abort the capture
                //  then let one frame flow so that capture was aborted properly, and integration SDK
                //  updated the camera properly (this we delay toggling by 1 frame)
                if (m_anselSDK.getCaptureState() != CAPTURE_NOT_STARTED)
                {
                    abortCapture();
                    m_toggleDelay = 1;
                }
            }
            if (m_toggleDelay == 0)
            {
                m_isClientEnabled = false;

                stopSession();
                m_activeControlClient->anselStopDone(AnselUIBase::Status::kOk);
            }
            else
            {
                --m_toggleDelay;
            }
        }
        else
        {
            // This is a shortlwt teardown, if Ansel wasn't really enabled
            LOG_DEBUG("Ansel Shortlwt Teardown");
            m_activeControlClient->anselStopDone(AnselUIBase::Status::kOk);
        }
    }
    if (m_activeControlClient->isAnselPoststopRequested())
    {
        m_isClientActive = false;
        LOG_INFO("Ansel::Poststop requested");

        // Show the style transfer folder reminder once a session
#ifdef ENABLE_STYLETRANSFER
        m_wasStyleTransferFolderShown = false;
#endif

        if ((m_hadNoActiveEffects || !m_enableAnselModding) && // There are no active effects or Ansel is not being used for modding, and
            !m_lightweightShimMode)         // the shim is not lwrrently in lightweight mode.
        {
            enterLightweightMode();
        }
        m_activeControlClient->anselPoststopDone(AnselUIBase::Status::kOk);
    }

    isAnselSessionActive = m_anselSDK.isSDKDetectedAndSessionActive();
    // Until client is fully enabled, Ansel is in transitioning state, so we might want to postpone checks
    //  because Ansel session won't activate until after StartRequested processing
    bool isTransitioningState = !m_isClientEnabled;
    if (!isTransitioningState && !isAnselSessionActive)
    {
        if (!filtersLwrrentlyAllowed)
        {
            bool disabledDueToNetworkUse = moddingMultiplayerMode ? m_networkActivityDetected : false;
            // Modding is not allowed, and Ansel session is not active
            if (!m_shownNetworkActivityMsg && disabledDueToNetworkUse)
            {
                // Modding was likely disabled due to network activity, show message
                m_UI->displayMessage(AnselUIBase::MessageType::kModdingNetworkActivity);
                m_shownNetworkActivityMsg = true;
            }

            LOG_DEBUG("Disabling modding (in transition)");
            m_activeControlClient->setModdingStatus(false);
            if (m_specialClientActive && m_isClientActive)
                m_bNextFrameForceDisableAnsel = true;
        }
    }
    m_wasModdingAllowed = filtersLwrrentlyAllowed;

    if (m_activeControlClient->isSDKCaptureAbortRequested())
    {
        if (m_anselSDK.getCaptureState() != CAPTURE_NOT_STARTED)
        {
            LOG_DEBUG("Capture Aborted");
            abortCapture();
            m_activeControlClient->sdkCaptureAbortDone(0);
        }
        else
        {
            m_activeControlClient->sdkCaptureAbortDone(-1); // NO CAPTURE TO ABORT
        }
    }

    // we will only send game specific control if an Ansel session was created
    if (isAnselSessionActive && (m_anselSDK.isUserControlUpdated() || m_UI->isUpdateGameSpecificControlsRequired()))
    {
        AnselUIBase::EffectPropertiesDescription effectDesc;

        effectDesc.filterId = L"GameSpecific";
        effectDesc.filterDisplayName = L"GameSpecific";

        const int localeNameSize = 32;
        wchar_t localeNameW[localeNameSize];
        LCIDToLocaleName(m_activeControlClient->getLangId(), localeNameW, localeNameSize, 0);

        std::string localeName = darkroom::getUtf8FromWstr(localeNameW);

        std::vector<AnselUserControlDesc> gameSpecificControls = m_anselSDK.getUserControlDescriptions();
        for (int cnt = 0, cneEnd = (int)gameSpecificControls.size(); cnt < cneEnd; ++cnt)
        {
            AnselUserControlDesc & controlDesc = gameSpecificControls[cnt];
            AnselUIBase::EffectPropertiesDescription::EffectAttributes attribDesc;

            bool attribValid = true;

            attribDesc.controlId = (uint32_t)controlDesc.info.userControlId;

            // Search for matching label
            std::string * enLabel = nullptr;
            std::string * matchingLabel = nullptr;
            for (int lblCnt = 0, lblCntEnd = (int)controlDesc.labels.size(); lblCnt < lblCntEnd; ++lblCnt)
            {
                if (controlDesc.labels[lblCnt].lang == localeName)
                {
                    matchingLabel = &controlDesc.labels[lblCnt].labelUtf8;
                    break;
                }
                else if (controlDesc.labels[lblCnt].lang == "en-US")
                {
                    enLabel = &controlDesc.labels[lblCnt].labelUtf8;
                }
            }
            attribDesc.displayName = darkroom::getWstrFromUtf8(matchingLabel ? *matchingLabel : (enLabel ? *enLabel : std::string("Control")));
            attribDesc.displayNameEnglish = darkroom::getWstrFromUtf8( enLabel ? *enLabel : std::string("Control") );
            attribDesc.uiMeasurementUnit = L"";

            switch (controlDesc.info.userControlType)
            {
            case ansel::kUserControlSlider:
            {
                attribDesc.controlType = AnselUIBase::ControlType::kSlider;
                attribDesc.dataType = AnselUIBase::DataType::kFloat;

                attribDesc.defaultValue.reinitialize(*reinterpret_cast<const float*>(controlDesc.info.value));
                attribDesc.lwrrentValue.reinitialize(*reinterpret_cast<const float*>(controlDesc.info.value));

                attribDesc.milwalue.reinitialize(0.0f);
                attribDesc.maxValue.reinitialize(1.0f);
                attribDesc.uiMilwalue.reinitialize(0.0f);
                attribDesc.uiMaxValue.reinitialize(1.0f);
                attribDesc.stepSizeUI.reinitialize(0.0f);
                attribDesc.valueDisplayName[0] = L"";

                attribDesc.stickyValue = 0.5f;
                attribDesc.stickyRegion = 0.01f;

                break;
            }
            case ansel::kUserControlBoolean:
            {
                attribDesc.controlType = AnselUIBase::ControlType::kCheckbox;
                attribDesc.dataType = AnselUIBase::DataType::kBool;

                attribDesc.defaultValue.reinitialize((*reinterpret_cast<const bool*>(controlDesc.info.value)) ? 1.0f : 0.0f);
                attribDesc.lwrrentValue.reinitialize((*reinterpret_cast<const bool*>(controlDesc.info.value)) ? 1.0f : 0.0f);

                attribDesc.milwalue.reinitialize(0.0f);
                attribDesc.maxValue.reinitialize(1.0f);
                attribDesc.uiMilwalue.reinitialize(0.0f);
                attribDesc.uiMaxValue.reinitialize(1.0f);
                attribDesc.stepSizeUI.reinitialize(0.0f);
                attribDesc.valueDisplayName[0] = L"";

                attribDesc.stickyValue = 1.0f;
                attribDesc.stickyRegion = 0.0f;

                break;
            }
            default:
            {
                attribValid = false;
                LOG_ERROR("Unknown game-specific user control type");
                break;
            }
            }
            attribDesc.userConstant = nullptr;

            if (attribValid)
                effectDesc.attributes.push_back(attribDesc);
        }

        m_activeControlClient->updateGameSpecificControls(&effectDesc);

        m_anselSDK.clearUserControlUpdated();
    }

    if (m_activeControlClient->isEffectListRequested())
    {
        std::vector<std::wstring> effectsIds, effectsNames;

        int meEffectNumber = -1;
        // TODO avoroshilov: we no longer need compatibility with custom - better adderess this
        for (size_t resCnt = 0, resCntEnd = m_effectsInfo.m_effectFilesList.size(); resCnt < resCntEnd; ++resCnt)
        {
            if (wcscmp(L"custom.yaml", m_effectsInfo.m_effectFilesList[resCnt].c_str()) == 0)
            {
                meEffectNumber = (int)resCnt;
            }
        }

        if (meEffectNumber >= 0)
        {
            effectsNames.push_back(L"Custom");
            effectsIds.push_back(L"custom");
        }

        std::wstring lwrName, lwrId;
        for (size_t resCnt = 0, resCntEnd = m_effectsInfo.m_effectFilesList.size(); resCnt < resCntEnd; ++resCnt)
        {
            if ((int)resCnt == meEffectNumber)
                continue;

            // Hacky skip Adjustments and SpecialFX since they are always present in fixed stacking

            effectFilterIdAndName(m_effectsInfo.m_effectRootFoldersList[resCnt], m_effectsInfo.m_effectFilesList[resCnt], m_activeControlClient->getLangId(), lwrId, lwrName);
            effectsIds.push_back(lwrId);
            effectsNames.push_back(lwrName);
        }

        m_activeControlClient->repopulateEffectsList(effectsIds, effectsNames);
    }

#ifdef ENABLE_STYLETRANSFER
    if (m_activeControlClient->isStylesListRequested())
        m_activeControlClient->repopulateStylesList(m_stylesFilesListTrimmed, m_stylesFilesList, m_stylesFilesPaths);

    if (m_activeControlClient->isStyleNetworksListRequested())
        m_activeControlClient->repopulateStyleNetworksList(m_styleNetworksIds, m_styleNetworksLabels);
#endif

    bool effectControlChanged = applyEffectChanges();

    ANSEL_PROFILE_STOP(ff10_UI_requests);


    m_anselSDK.updateTileUV(&getTileInfo().m_tileTLU, &getTileInfo().m_tileTLV, &getTileInfo().m_tileBRU, &getTileInfo().m_tileBRV);


    ANSEL_PROFILE_START(ff11_effectApply, "Effect Apply");

    // We are running pass-through shader (instead of full effect processing pipeline), in case if
    //  filtering is completely disallowed, effect processing framework could not be run, or effect is not selected
    //  also, if Ansel was just activated, we need to skip processing, since resources (e.g. Depth buffers) might not be ready
    if (m_bWasAnselDeactivated || !applyingFilters || !m_bRunShaderMod || !m_effectsInfo.m_selectedEffect)
    {
        AnselEffectState* pColorEffect = &m_passthroughEffect;
        static FLOAT clearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };

        D3D11_VIEWPORT viewPort;
        viewPort.Width = float(pPresentResource->toServerRes.width);
        viewPort.Height = float(pPresentResource->toServerRes.height);
        viewPort.MinDepth = 0.0f;
        viewPort.MaxDepth = 1.0f;
        viewPort.TopLeftX = 0;
        viewPort.TopLeftY = 0;

        m_immediateContext->RSSetViewports(1, &viewPort);
        m_immediateContext->ClearRenderTargetView(pPresentResource->toClientRes.pRTV, clearColor);

        UINT ZERO = 0;
        UINT vertexStride = 20;
        m_immediateContext->VSSetShader(pColorEffect->pVertexShader, NULL, 0);
        m_immediateContext->PSSetShader(pColorEffect->pPixelShader, NULL, 0);

        m_immediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        m_immediateContext->PSSetShaderResources(0, 1, &pPresentResource->toServerRes.pSRV);

        m_immediateContext->IASetInputLayout(0);
        m_immediateContext->PSSetSamplers(0, 1, &pColorEffect->pSamplerState);
        m_immediateContext->RSSetState(pColorEffect->pRasterizerState);
        m_immediateContext->OMSetDepthStencilState(pColorEffect->pDepthStencilState, 0xFFFFFFFF);
        m_immediateContext->OMSetRenderTargets(1, &pPresentResource->toClientRes.pRTV, NULL);
        m_immediateContext->OMSetBlendState(pColorEffect->pBlendState, NULL, 0xffffffff);
        m_immediateContext->Draw(3, 0);

        m_depthBufferUsed = m_psUtil.GetPsdExportEnable();
        m_hudlessBufferUsed = m_psUtil.GetPsdExportEnable();
    }
    else
    {
        shadermod::MultipassConfigParserError err(shadermod::MultipassConfigParserErrorEnum::eOK);

        // There was checking buffers availability for effects
        // The decision then was to do the follwoing instead of removing effect from the stack in case buffer is not available:
        //  related buffer availability system constant is set to false, and effect should process this fact on its side;
        //  for example, DoF will become effectively pass-through, untill depth buffer is available
        for (size_t effIdx = 0, effIdxEnd = m_effectsInfo.m_effectSelected.size(); effIdx < effIdxEnd; ++effIdx)
        {
            if (m_effectsInfo.m_bufferCheckRequired[effIdx])
            {
                shadermod::MultiPassEffect* eff = m_effectsInfo.m_effectsStack[effIdx];

                if (!eff)
                    continue;

                if (eff->isDepthRequired())
                {
                    if (!depthAvailable)
                    {
                        // Only show message once per unavailable buffer timespan
                        if (m_effectsInfo.m_bufferCheckMessages[effIdx] & (uint32_t)EffectsInfo::BufferToCheck::kDepth)
                        {
                            m_displayMessageStorage.resize(0);
                            m_displayMessageStorage.push_back(eff->getFxFileFullPath().c_str());
                            m_activeControlClient->displayMessage(AnselUIBase::MessageType::kEffectRequiresDepth, m_displayMessageStorage);
                            // Do not show the message until bufer is available again
                            m_effectsInfo.m_bufferCheckMessages[effIdx] &= (~(uint32_t)EffectsInfo::BufferToCheck::kDepth);
                        }
                    }
                    else
                    {
                        // If buffer becomes available, raise the messaging flag so that next time it disappears,
                        //  user can be notified
                        m_effectsInfo.m_bufferCheckMessages[effIdx] |= (uint32_t)EffectsInfo::BufferToCheck::kDepth;
                    }
                }

                if (eff->isHDRRequired())
                {
                    if (!hdrAvailable)
                    {
                        if (m_effectsInfo.m_bufferCheckMessages[effIdx] & (uint32_t)EffectsInfo::BufferToCheck::kHDR)
                        {
                            m_displayMessageStorage.resize(0);
                            m_displayMessageStorage.push_back(eff->getFxFileFullPath().c_str());
                            m_activeControlClient->displayMessage(AnselUIBase::MessageType::kEffectRequiresHDR, m_displayMessageStorage);
                            m_effectsInfo.m_bufferCheckMessages[effIdx] &= (~(uint32_t)EffectsInfo::BufferToCheck::kHDR);
                        }
                    }
                    else
                    {
                        m_effectsInfo.m_bufferCheckMessages[effIdx] |= (uint32_t)EffectsInfo::BufferToCheck::kHDR;
                    }
                }

                if (eff->isHUDlessRequired())
                {
                    // HUDless is replaced by the final color buffer
                }
            }
        }

        auto fillEffectsHashes = [this](
            const Hash::Data & effectFilenameHashed, uint32_t restrictedSetID,
            std::set<Hash::Effects> &expectedHashes
            ) -> bool
        {
            PrepackagedEffects effIdx = PrepackagedEffects::kNUM_ENTRIES;

            const auto& itEff = m_moddingRestrictedSets[restrictedSetID].find(effectFilenameHashed);
            if (itEff != m_moddingRestrictedSets[restrictedSetID].end())
            {
                effIdx = itEff->second;
            }

            return m_moddingEffectHashDB.GetGeneratedShiftedHashSet(effIdx, expectedHashes);
        };

        bool needToFetchHash = moddingMultiplayerMode ? m_moddingMultiPlayer.mode == ModdingMode::kRestrictedEffects : m_moddingSinglePlayer.mode == ModdingMode::kRestrictedEffects;
        uint32_t restrictedEffectSetId = moddingMultiplayerMode ? m_moddingMultiPlayer.restrictedEffectSetId : m_moddingSinglePlayer.restrictedEffectSetId;

        if (!isTransitioningState && !isAnselSessionActive)
        {
            bool failedValidation = false;
            bool fetchHash = false;
            for (size_t effIdx = 0, effIdxEnd = m_effectsInfo.m_effectsStack.size(); effIdx < effIdxEnd; ++effIdx)
            {
                if (effIdx >= m_effectsInfo.m_effectSelected.size())
                    continue;

                uint32_t selectedEffect = m_effectsInfo.m_effectSelected[effIdx];
                if (selectedEffect == 0 || selectedEffect == -1)
                    continue;

                shadermod::MultiPassEffect * eff = m_effectsInfo.m_effectsStack[effIdx];

                // Effect selected but not yet created
                if (eff == nullptr || m_effectsInfo.m_effectRebuildRequired[effIdx])
                    continue;

                fetchHash = needToFetchHash;
                uint32_t lwrRestrictedEffectSetId = restrictedEffectSetId;
                if (m_SharpenLwbinData.m_HWSupport && eff->getFxFilename() == L"Sharpen.yaml")
                {
                    fetchHash = true;
                    lwrRestrictedEffectSetId = uint32_t(ModdingRestrictedSetID::kPrepackaged);
                }

                if (fetchHash)
                {
                    std::set<Hash::Effects> targetHashSet;
                    bool effectFoundInSet = fillEffectsHashes(
                        m_effectsInfo.m_effectsHashedName[effIdx],
                        lwrRestrictedEffectSetId,
                        targetHashSet
                    );

                    bool hashCorrect = false;
                    if (effectFoundInSet)
                    {
                        if (targetHashSet.find(eff->getCallwlatedHashes()) != targetHashSet.end())
                        {
                            hashCorrect = true;
                        }
                    }

                    if (!hashCorrect)
                    {
                        destroyEffectInternal(effIdx);
                        m_effectsInfo.m_effectSelected[effIdx] = 0;
                        m_effectsInfo.m_effectRebuildRequired[effIdx] = true;
                        failedValidation = true;
                    }
                }
            }

            if (failedValidation)
            {
                m_stackRebuildRequired = true;
                effectChanged = true;
            }
        }

        std::set<Hash::Effects> expectedHashSet;
        const std::set<Hash::Effects> *pExpectedHashSet = nullptr;

        // We need to callwlate hashes in case there is even a hint of restricted mode, otherwise validation will unconditionally fail
        //  when the mode that requires restriction will be activated
        bool compareHashes = false;
        if (needToFetchHash)
        {
            pExpectedHashSet = &expectedHashSet;
        }

        bool rebuiltEffects = m_stackRebuildRequired;
        if (m_stackRebuildRequired || m_bNextFrameRefreshEffectStack || effectChanged || !m_bShaderModInitialized)
        {
            // We need to save all the values prior to the any stack modifications
            userConstantStorage.values.resize(0);
            userConstantStorage.names.resize(0);
            userConstantStorage.offsets.resize(0);
            userConstantStorage.effectStack.resize(0);
            int offset = 0;
            for (size_t effIdx = 0, effIdxEnd = m_effectsInfo.m_effectsStack.size(); effIdx < effIdxEnd; ++effIdx)
            {
                userConstantStorage.offsets.push_back(offset);
                shadermod::MultiPassEffect* eff = m_effectsInfo.m_effectsStack[effIdx];

                if (eff)
                {
                    userConstantStorage.effectStack.push_back(eff->getFxFilename());
                    shadermod::ir::UserConstantManager::Range ucrange = eff->getUserConstantManager().getPointersToAllUserConstants();

                    for (; ucrange.begin < ucrange.end; ++ucrange.begin)
                    {
                        const shadermod::ir::UserConstant * lwrUC = *(ucrange.begin);
                        userConstantStorage.values.push_back(lwrUC->getValue());
                        userConstantStorage.names.push_back(lwrUC->getName());
                        ++offset;
                    }
                }
                else
                {
                    userConstantStorage.effectStack.push_back(L"");
                }
            }

            bool fullStackRebuild = true;
            if (m_bNextFrameRefreshEffectStack || m_effectsInfo.m_effectsStack.size() != m_effectsInfo.m_effectSelected.size() || m_effectsInfo.m_effectsStack.size() != getNumEffects())
            {
                // We need full rebuild
            }
            else
            {
                // Attempt to partially replace effects
                fullStackRebuild = false;

                int multiPassProcStackIdx = 0;
                bool fetchHash = false;
                for (size_t effIdx = 0, effIdxEnd = m_effectsInfo.m_effectSelected.size(); effIdx < effIdxEnd; ++effIdx)
                {
                    if (m_effectsInfo.m_effectRebuildRequired[effIdx])
                    {
                        compareHashes = !isAnselSessionActive && needToFetchHash;
                        int selectedEffect = m_effectsInfo.m_effectSelected[effIdx];
                        if (selectedEffect > 0)
                        {
                            fetchHash = needToFetchHash;
                            uint32_t lwrRestrictedEffectSetId = restrictedEffectSetId;
                            if (m_SharpenLwbinData.m_HWSupport && m_effectsInfo.m_effectFilesList[selectedEffect - 1] == L"Sharpen.yaml")
                            {
                                // Ensure hash-check for Sharpen lwbin happens in all cases : full - Ansel and Freestyle, Ansel - Integrated mode only as well as Ansel - Lite mode.
                                fetchHash = compareHashes = true;
                                pExpectedHashSet = &expectedHashSet;
                                lwrRestrictedEffectSetId = uint32_t(ModdingRestrictedSetID::kPrepackaged);
                            }

                            if (fetchHash)
                            {
                                Hash::Data effNameKey;
                                sha256_ctx effNameHashContext;
                                sha256_init(&effNameHashContext);
                                sha256_update(&effNameHashContext, (uint8_t *)m_effectsInfo.m_effectFilesList[selectedEffect - 1].c_str(), (uint32_t)(m_effectsInfo.m_effectFilesList[selectedEffect - 1].length() * sizeof(wchar_t)));
                                sha256_final(&effNameHashContext, effNameKey.data());

                                fillEffectsHashes(
                                    effNameKey,
                                    lwrRestrictedEffectSetId,
                                    expectedHashSet
                                );

                                m_effectsInfo.m_effectsHashedName[effIdx] = effNameKey;
                            }
                        }

                        loadEffectOnStack(effIdx, multiPassProcStackIdx, true, pExpectedHashSet, compareHashes);
                        m_effectsInfo.m_effectRebuildRequired[effIdx] = false;
                    }

                    if (m_effectsInfo.m_effectsStack[effIdx] != nullptr)
                    {
                        ++multiPassProcStackIdx;
                    }
                }

                // If this returns false/error - there were mismatching surface parameters, trigger full rebuild
                err = relinkEffects(ignoreZeroTextureSwitch);
                if (err)
                {
                    fullStackRebuild = true;
                }
            }

            if (fullStackRebuild)
            {
                // Trigger full rebuild

                // Remove old effects
                for (size_t effInternalIdx = 0, effInternalIdxEnd = getNumEffects(); effInternalIdx < effInternalIdxEnd; ++effInternalIdx)
                {
                    popBackEffect();
                }
                m_effectsInfo.m_effectsStack.resize(m_effectsInfo.m_effectSelected.size());
                memset(m_effectsInfo.m_effectsStack.data(), 0, sizeof(shadermod::MultiPassEffect *) * m_effectsInfo.m_effectSelected.size());
                m_effectsInfo.m_effectsStackMapping.resize(m_effectsInfo.m_effectSelected.size());

                int multiPassProcStackIdx = 0;
                bool fetchHash = false;
                for (size_t effIdx = 0, effIdxEnd = m_effectsInfo.m_effectSelected.size(); effIdx < effIdxEnd; ++effIdx)
                {
                    compareHashes = !isAnselSessionActive && needToFetchHash;
                    int selectedEffect = m_effectsInfo.m_effectSelected[effIdx];
                    if (selectedEffect > 0)
                    {
                        fetchHash = needToFetchHash;
                        uint32_t lwrRestrictedEffectSetId = restrictedEffectSetId;
                        if (m_SharpenLwbinData.m_HWSupport && m_effectsInfo.m_effectFilesList[selectedEffect - 1] == L"Sharpen.yaml")
                        {
                            // Ensure hash-check for Sharpen lwbin happens in all cases : full - Ansel and Freestyle, Ansel - Integrated mode only as well as Ansel - Lite mode.
                            fetchHash = compareHashes = true;
                            pExpectedHashSet = &expectedHashSet;
                            lwrRestrictedEffectSetId = uint32_t(ModdingRestrictedSetID::kPrepackaged);
                        }

                        if (fetchHash)
                        {
                            Hash::Data effNameKey;
                            sha256_ctx effNameHashContext;
                            sha256_init(&effNameHashContext);
                            sha256_update(&effNameHashContext, (uint8_t *)m_effectsInfo.m_effectFilesList[selectedEffect - 1].c_str(), (uint32_t)(m_effectsInfo.m_effectFilesList[selectedEffect - 1].length() * sizeof(wchar_t)));
                            sha256_final(&effNameHashContext, effNameKey.data());

                            fillEffectsHashes(
                                effNameKey,
                                lwrRestrictedEffectSetId,
                                expectedHashSet
                            );

                            m_effectsInfo.m_effectsHashedName[effIdx] = effNameKey;
                        }
                    }

                    loadEffectOnStack(effIdx, multiPassProcStackIdx, false, pExpectedHashSet, compareHashes);
                    m_effectsInfo.m_effectRebuildRequired[effIdx] = false;

                    shadermod::MultiPassEffect* eff = m_effectsInfo.m_effectsStack[effIdx];

                    if (eff != nullptr)
                    {
                        ++multiPassProcStackIdx;
                    }
                }
            }

            // Attempt to restore saved values
            for (size_t effIdx = 0, effIdxEnd = m_effectsInfo.m_effectSelected.size(); effIdx < effIdxEnd; ++effIdx)
            {
                shadermod::MultiPassEffect * eff = m_effectsInfo.m_effectsStack[effIdx];
                if (eff == nullptr)
                    continue;

                int effOldIdx = -1;
                if (effIdx < userConstantStorage.oldStackIdx.size())
                {
                    effOldIdx = userConstantStorage.oldStackIdx[effIdx];
                }
                else if (userConstantStorage.oldStackIdx.size() == 0)
                {
                    // Not a UI-triggered update, so stack indices remain untouched
                    effOldIdx = (int)effIdx;
                }

                if ((effOldIdx >= 0) && (eff != nullptr) && (effOldIdx < (int)userConstantStorage.effectStack.size()) && (eff->getFxFilename() == userConstantStorage.effectStack[effOldIdx]))
                {
                    int userConstantOffset = userConstantStorage.offsets[effOldIdx];
                    unsigned int numConstantsSaved = (effOldIdx < (int)userConstantStorage.effectStack.size() - 1) ? static_cast<unsigned int>(userConstantStorage.offsets[effOldIdx + 1]) :
                        static_cast<unsigned int>(userConstantStorage.values.size());
                    numConstantsSaved -= userConstantOffset;

                    for (unsigned int ucsaved = 0; ucsaved < numConstantsSaved; ++ucsaved)
                    {
                        shadermod::ir::UserConstant * lwrUC = eff->getUserConstantManager().findByName(userConstantStorage.names[userConstantOffset + ucsaved]);

                        if (lwrUC && lwrUC->getType() == userConstantStorage.values[userConstantOffset + ucsaved].getType())
                        {
                            lwrUC->setValue(userConstantStorage.values[userConstantOffset + ucsaved]);
                        }
                    }
                }
            }

            // We have all the effects in place, so we can go through them to see if we really need the depth buffer
            int multiPassProcStackIdx = 0;
            for (size_t effIdx = 0, effIdxEnd = m_effectsInfo.m_effectsStack.size(); effIdx < effIdxEnd; ++effIdx)
            {
                shadermod::MultiPassEffect* eff = m_effectsInfo.m_effectsStack[effIdx];
                if (eff)
                {
                    if (eff->isDepthRequired() || eff->isHDRRequired() || eff->isHUDlessRequired())
                    {
                        // We cannot check for buffers availability right now,
                        //  as the buffer extraction could simply be disabled if previous effects didn't require
                        //  those buffers;
                        // Hense, 1 frame lag is introduced for checks - enable buffer extraction first and let it run for a frame
                        m_effectsInfo.m_bufferCheckRequired[effIdx] = true;
                        m_effectsInfo.m_bufferCheckMessages[effIdx] = (uint32_t)EffectsInfo::BufferToCheck::kALL;
                    }

                    if (eff->isDepthRequired())
                    {
                        m_depthBufferUsed = true;
                    }
                    if (eff->isHDRRequired())
                    {
                        m_hdrBufferUsedByFilter = true;
                    }
                    if (eff->isHUDlessRequired())
                    {
                        m_hudlessBufferUsed = true;
                    }

                    ++multiPassProcStackIdx;
                }
            }

            m_stackRebuildRequired = false;     // Stack rebiulding is done
            rebuiltEffects = true;              // Via this variable we signal that rebuilding was done - this is used later down the pipe
            m_bShaderModInitialized = true;

            userConstantStorage.oldStackIdx.resize(0);
        }

        if (!isDeviceValid() || !areInputsValid())
        {
            LOG_FATAL("Effects framework device or outputs are invalid");
            HandleFailure();
        }

        // TODO avoroshilov stacking:
        //  fix this

        // This happens when effect failed for some reason when rebuild is triggered
        //  e.g. when game changes resolution or something; during this times, effect can be rebuilt from source, and if source changed
        //  this could cause an issue. We can ciorlwmvent that by not reading file again and rebuilding what we already have (this should be quicker too)
        //  and re-read file only when explicitly asked, thus limiting the places where it could fail
        if (isDefunctEffectOnStack())
        {
            //telemetry
            {
                ErrorDescForTelemetry desc;

                desc.errorCode = 0;
                desc.lineNumber = __LINE__;
                desc.errorMessage = err.getFullErrorMessage();
                desc.filename = __FILE__;

                AnselStateForTelemetry state;
                HRESULT telemetryStatus = makeStateSnapshotforTelemetry(state);
                if (telemetryStatus == S_OK)
                    sendTelemetryEffectCompilationErrorEvent(state, m_anselSDK.getCaptureState(), desc);
            }

            // Fall back to "None"
            const std::wstring wideErr = darkroom::getWstrFromUtf8(err.getFullErrorMessage().data());
            m_displayMessageStorage.resize(0);
            m_displayMessageStorage.push_back(m_effectsInfo.m_effectFilterIds[m_effectsInfo.m_selectedEffect - 1].c_str());
            m_displayMessageStorage.push_back(wideErr.c_str());
            m_activeControlClient->displayMessage(AnselUIBase::MessageType::kErrorParsingFile, m_displayMessageStorage);

            // Report error
            m_activeControlClient->updateEffectControls(0, nullptr);

            AnselUIBase::EffectPropertiesDescription effectDesc;
            effectDesc.filterId = shadermod::Tools::wstrNone;
            m_activeControlClient->updateEffectControls(TEMP_SELECTABLE_FILTER_ID, &effectDesc);

            m_effectsInfo.m_selectedEffect = 0;
            m_bNextFrameRunShaderMod = false;
        }

        if (rebuiltEffects)
        {
            for (size_t effIdx = 0, effIdxEnd = m_effectsInfo.m_effectSelected.size(); effIdx < effIdxEnd; ++effIdx)
            {
                shadermod::MultiPassEffect* eff = m_effectsInfo.m_effectsStack[effIdx];
                int effectsStackMapping = m_effectsInfo.m_effectsStackMapping[effIdx];

                AnselUIBase::EffectPropertiesDescription effectDesc;
                getEffectDescription(eff, m_activeControlClient->getLangId(), &effectDesc);
                m_activeControlClient->updateEffectControls(effIdx, &effectDesc);
            }
        }

        for (size_t effIdx = 0, effIdxEnd = m_effectsInfo.m_effectSelected.size(); (effIdx < effIdxEnd) && filterInfoQueried; ++effIdx)
        {
            if (!m_activeControlClient->getLwrrentFilterInfoQuery(effIdx))
                continue;

            shadermod::MultiPassEffect* eff = nullptr;

            if (m_effectsInfo.m_effectsStack.size() > effIdx)
            {
                eff = m_effectsInfo.m_effectsStack[effIdx];
            }
            AnselUIBase::EffectPropertiesDescription effectDesc;
            getEffectDescription(eff, m_activeControlClient->getLangId(), &effectDesc);
            m_activeControlClient->updateEffectControlsInfo(effIdx, &effectDesc);
        }

        // This shouldn't happen; this check is an emergency matter trying to avoid crashes due to access violations if it happens
        //  in some wrong state. After the current frame, Ansel should fall into pass-through mode in case this happened, and
        //  the picture will be back to normal
        if (getNumEffects() > 0)
        {
            // this function changes GAPI states, but it doesn't matter since all the subsequent calls set their own states
            processData(pSelectedFilteringResource->toClientRes.pRTV);
        }
    }

    if (m_activeControlClient->isStackFiltersListRequested())
    {
        std::vector<std::wstring> stackFiltersList;
        stackFiltersList.resize(0);
        for (size_t effIdx = 0, effIdxEnd = m_effectsInfo.m_effectSelected.size(); effIdx < effIdxEnd; ++effIdx)
        {
            int selectedEffect = m_effectsInfo.m_effectSelected[effIdx];
            if (selectedEffect > 0)
            {
                stackFiltersList.push_back(m_effectsInfo.m_effectFilterIds[selectedEffect - 1].c_str());
            }
            else
            {
                stackFiltersList.push_back(shadermod::Tools::wstrNone);
            }
        }
        m_activeControlClient->stackFiltersListDone(stackFiltersList);
    }

    ANSEL_PROFILE_STOP(ff11_effectApply);

    m_errorManager.diminishLifeTime(m_globalPerfCounters.dt * 0.001);

    if (0)
        debugRenderFrameNumber();

#ifdef ENABLE_STYLETRANSFER
    if (m_anselSDK.isSessionActive())
    {
        ANSEL_PROFILE_START(ff12_styleTransfer, "Style Transfer Initialization");

        #define STRINGIZE_WITH_PREFIX_(pre, x) pre#x
        #define STRINGIZE_L(x) STRINGIZE_WITH_PREFIX_(L, x)

        const auto versionString = std::wstring(STRINGIZE_L(RESTYLE_LIB_VERSION_FROM_PROPS));
        // download and install package if needed
        if (m_activeControlClient->getRestyleDownloadConfirmationStatus() == AnselUIBase::RestyleDownloadStatus::kConfirmed)
        {
            m_activeControlClient->setStyleTransferStatus(false);
            const std::wstring packageVersion(L"1.2.3.0");
            // e.g. https://international-gfe.download.lwpu.com/Ansel/InstPckgs/1.2.3.0/InstallStyleTransfer.30_v1.2.3.0.exe
            const std::wstring librestyleUrl = std::wstring(L"https://international-gfe.download.lwpu.com/Ansel/InstPckgs/") +
                packageVersion +
                L"/InstallStyleTransfer." +
                std::to_wstring(m_lwdaDeviceProps.major) +
                std::to_wstring(m_lwdaDeviceProps.minor) +
                L"_v" +
                packageVersion +
                L".exe";

            sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::ACCEPTED_INSTALLATION));
            const auto error = m_sideloadHandler.downloadAndInstall(librestyleUrl, m_intermediateFolderPath + L"//lwstyletransferinstaller.exe", { L"/S" }, INFINITE);
            // send telemetry: StyleTransferDownloadStarted event
            if (error != darkroom::Error::kSuccess)
            {
                m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_CantStartDownload);
                sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::DOWNLOAD_STARTUP_FAILED));
                m_styleSelected = -1;
            }
            else
            {
                sendTelemetryStyleTransferDownloadStartedEvent(darkroom::getUtf8FromWstr(librestyleUrl), darkroom::getUtf8FromWstr(versionString), m_lwdaDeviceProps.major, m_lwdaDeviceProps.minor);
                m_downloadStarttime = std::chrono::steady_clock::now();

                m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_Downloading);
                m_activeControlClient->setRestyleProgressState(AnselUIBase::RestyleProgressState::kDownloadProgress);
                m_activeControlClient->toggleRestyleProgressBar(true);
            }

            m_activeControlClient->clearRestyleDownloadConfirmationStatus();
        }
        else if (m_activeControlClient->getRestyleDownloadConfirmationStatus() == AnselUIBase::RestyleDownloadStatus::kRejected)
        {
            m_activeControlClient->clearRestyleDownloadConfirmationStatus();
            sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::DECLINED_INSTALLATION));
        }

        auto& future = m_sideloadHandler.getFuture();
        static bool restyleProgressStateInstallSet = false;
        if (future.valid() && future.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
        {
            const auto err = future.get();
            if (err == darkroom::Error::kSuccess)
            {
                LOG_INFO("Style transfer package installed");
                m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_Install_Success);
                const auto duration = std::chrono::steady_clock::now() - m_downloadStarttime;
                sendTelemetryStyleTransferDownloadFinishedEvent(uint32_t(std::chrono::duration_cast<std::chrono::seconds>(duration).count()),
                    int32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::INSTALLATION_SUCCESS));
                sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::INSTALLATION_SUCCESS));
                populateStylesList();
                m_activeControlClient->repopulateStylesList(m_stylesFilesListTrimmed, m_stylesFilesList, m_stylesFilesPaths);
                m_activeControlClient->setStyleTransferStatus(true);
            }
            else
            {
                if (err == darkroom::Error::kInstallFailed)
                {
                    sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::INSTALLATION_FAILED));
                    LOG_ERROR("Style transfer package installation failed");
                }
                else if (err == darkroom::Error::kDownloadFailed)
                {
                    sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::DOWNLOADING_FAILED));
                    LOG_ERROR("Style transfer package download failed");
                }
                else if (err == darkroom::Error::kExceptionOclwred)
                {
                    sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::EXCEPTION_OCLWRED));
                    LOG_ERROR("Style transfer exception oclwrred");
                }
                else if (err == darkroom::Error::kOperationFailed)
                {
                    sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::OPERATION_FAILED));
                    LOG_ERROR("Style transfer operation failed");
                }
                else if (err == darkroom::Error::kOperationTimeout)
                {
                    sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::OPERATION_TIMEOUT));
                    LOG_ERROR("Style transfer operation timeout");
                }
                else if (err == darkroom::Error::kCouldntStartupTheProcess)
                {
                    sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::STARTUP_FAILURE));
                    LOG_ERROR("Style transfer startup failure");
                }
                m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_Install_Failed);
            }
            m_activeControlClient->toggleRestyleProgressBar(false);
            restyleProgressStateInstallSet = false;
        }

        if (m_sideloadHandler.getSideloadStatus() != sideload::SideloadStatus::kNone)
        {
            // read progress from pipe
            if (m_sideloadHandler.getSideloadStatus() == sideload::SideloadStatus::kInstalling)
            {
                if (!restyleProgressStateInstallSet)
                {
                    m_activeControlClient->setRestyleProgressState(AnselUIBase::RestyleProgressState::kInstalling);
                    restyleProgressStateInstallSet = true;
                }
            }
            else
                m_activeControlClient->setRestyleProgressBarValue(float(m_sideloadHandler.getDownloadProgress()) / 100.0f);
        }

        static auto previousNetwork = m_activeControlClient->getLwrrentStyleNetwork();

        bool reinitializeNetwork = previousNetwork != m_activeControlClient->getLwrrentStyleNetwork();

        const int prevSelectedStyle = m_styleSelected;
        int styleSelected = m_styleSelected;

        bool styleTransferEnabled = m_activeControlClient->isStyleTransferEnabled();
        if (styleTransferEnabled)
        {
            styleSelected = int(getStyleIndexByName(m_activeControlClient->getLwrrentStyle()));

            if (!m_wasStyleTransferFolderShown)
            {
                m_displayMessageStorage.resize(0);
                m_displayMessageStorage.push_back(m_userStylesFolderPath.c_str());
                m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_UserFolder, m_displayMessageStorage);
                m_wasStyleTransferFolderShown = true;
            }
        }
        m_styleSelected = styleSelected;

        static bool loadedLibraries = false;

        bool styleTransferTextureRecreationRequired = false;

        auto checkStyleTransferTexture = [](bool isHDR, const StyleTransferTextureBuffer & stStagingBuf, const StyleTransferTextureBuffer & stStorageBuf, DXGI_FORMAT targetFmt, size_t targetW, size_t targetH) -> bool
        {
            if (isHDR)
            {
                // if the buffer is HDR buffer - we need to check GPU storage texture
                if ((stStorageBuf.tex == nullptr) || (stStorageBuf.fmt != targetFmt) || (stStorageBuf.w != targetW) || (stStorageBuf.h != targetH))
                {
                    return false;
                }
            }
            else
            {
                // if the buffer is not HDR, we need to check staging texture
                if ((stStagingBuf.tex == nullptr) || (stStagingBuf.fmt != targetFmt) || (stStagingBuf.w != targetW) || (stStagingBuf.h != targetH))
                {
                    return false;
                }
            }

            return true;
        };

        auto isStyleTransferHDR = [](DXGI_FORMAT targetFmt) -> bool
        {
            const std::array<DXGI_FORMAT, 12u> supportedFormats =
            {
                DXGI_FORMAT_R8G8B8A8_TYPELESS,
                DXGI_FORMAT_R8G8B8A8_UNORM,
                DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
                DXGI_FORMAT_R8G8B8A8_UINT,
                DXGI_FORMAT_R8G8B8A8_SNORM,
                DXGI_FORMAT_R8G8B8A8_SINT,
                DXGI_FORMAT_B8G8R8A8_UNORM,
                DXGI_FORMAT_B8G8R8X8_UNORM,
                DXGI_FORMAT_B8G8R8A8_TYPELESS,
                DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
                DXGI_FORMAT_B8G8R8X8_TYPELESS,
                DXGI_FORMAT_B8G8R8X8_UNORM_SRGB
            };

            // For now, we only support 8 bit formats
            if (std::find(supportedFormats.cbegin(), supportedFormats.cend(), targetFmt) == supportedFormats.cend())
            {
                return true;
            }

            return false;
        };

        if (!reinitializeNetwork &&
            m_styleSelected > 0 &&
            styleTransferEnabled &&
            loadedLibraries)
        {
            // Style transfer network configuration depends on the "content" buffer format and sizes
            //  so it needs to be reinitialized once those changed
            if (!checkStyleTransferTexture(m_styleTransferHDR, m_styleTransferOutputBuffer, m_styleTransferOutputHDRStorageBuffer, inputColorFormat, getWidth(), getHeight()))
            {
                reinitializeNetwork = true;
            }
        }

        if ((m_styleSelected > 0 || m_styleTransferStyles.size() == 0) &&
            styleTransferEnabled &&
            (!loadedLibraries ||
            ((reinitializeNetwork ||
            (styleSelected != prevSelectedStyle)))))
        {
            styleChanged = true;

            m_refreshStyleTransfer = true;
            if (!loadedLibraries)
            {
                wchar_t path[MAX_PATH] = { 0 };
                if (SHGetFolderPath(NULL, CSIDL_LOCAL_APPDATA, NULL, SHGFP_TYPE_LWRRENT, path) == S_OK)
                {
                    if (m_lwdaDeviceProps.major < 3)
                    {
                        sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::COMPUTE_CAPABILITY_TO_OLD));
                        m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_ComputeCapTooLow);
                        loadedLibraries = false;
                        m_activeControlClient->setStyleTransferStatus(false);
                    }
                    else
                    {
                        // Lwrrently we have librestyle builds for lwca 3.0, 3.5, 5.0, 6.1, & 7.0
                        switch (m_lwdaDeviceProps.major)
                        {
                            case 3:
                                if (m_lwdaDeviceProps.minor >= 5)
                                {
                                    m_lwdaDeviceProps.minor = 5; // 3.5+ -> 3.5
                                }
                                else
                                {
                                    m_lwdaDeviceProps.minor = 0; // 3.0-3.4 -> 3.0
                                }
                                break;
                            case 5:
                                m_lwdaDeviceProps.minor = 0; // 5.0+ -> 5.0
                                break;
                            case 6:
                                if (m_lwdaDeviceProps.minor >= 1)
                                {
                                    m_lwdaDeviceProps.minor = 1; // 6.1+ -> 6.1
                                }
                                else
                                {
                                    LOG_ERROR("Unsupported librestyle Lwca version: 6.0");
                                }
                                break;
                            case 7:
                                m_lwdaDeviceProps.minor = 0; // 7.0+ -> 7.0
                                break;
                            default:
                                LOG_ERROR("Unsupported librestyle Lwca version: %d.%d", m_lwdaDeviceProps.major, m_lwdaDeviceProps.minor);
                                break;
                        }

                        const std::wstring baseName(L"librestyle64.");
                        auto styleTransferPath = generateRestyleLibraryName(path,
                            baseName,
                            m_lwdaDeviceProps.major,
                            m_lwdaDeviceProps.minor,
                            versionString);
                        HMODULE handle = lwLoadSignedLibrary(styleTransferPath.c_str(), FALSE);

                        // if a specific package is not found, try 'all families' package
                        if (handle == NULL)
                        {
                            auto styleTransferPath = generateRestyleLibraryName(path, baseName, 0u, 0u, versionString);
                            handle = lwLoadSignedLibrary(styleTransferPath.c_str(), FALSE);
                        }

                        if (handle == NULL)
                        {
                            m_activeControlClient->showRestyleDownloadConfirmation();
                            loadedLibraries = false;
                            m_activeControlClient->setStyleTransferStatus(false);
                            m_styleSelected = -1;
                        }
                        else
                        {
                            // resolve functions here
                            // first resolve restyleGetVersion function to check version
                            bool loadedDllCheckFailed = false;
                            restyleGetVersionFunc = reinterpret_cast<decltype(restyleGetVersionFunc)>(GetProcAddress(handle, "restyleGetVersion"));

                            if (restyleGetVersionFunc != nullptr)
                            {
                                // check library version, only accept expected version
                                const auto actualVersion = restyleGetVersionFunc();
                                const std::wstring actualVersionAsString = std::to_wstring((actualVersion & 0xFFFF0000) >> 16) +
                                    L"." + std::to_wstring(actualVersion & 0x0000FFFF);
                                // if the version is fine continue resolving the rest of the functions and check if they are non null
                                if (actualVersionAsString == versionString)
                                {
                                    restyleUpdateStyleFunc = reinterpret_cast<decltype(restyleUpdateStyleFunc)>(GetProcAddress(handle, "restyleUpdateStyle"));
                                    restyleIsInitializedFunc = reinterpret_cast<decltype(restyleIsInitializedFunc)>(GetProcAddress(handle, "restyleIsInitialized"));
                                    restyleEstimateVRamUsageFunc = reinterpret_cast<decltype(restyleEstimateVRamUsageFunc)>(GetProcAddress(handle, "restyleEstimateVRamUsage"));
                                    restyleInitializeWithStyleFunc = reinterpret_cast<decltype(restyleInitializeWithStyleFunc)>(GetProcAddress(handle, "restyleInitializeWithStyle"));
                                    restyleInitializeWithStyleStatisticsFunc = reinterpret_cast<decltype(restyleInitializeWithStyleStatisticsFunc)>(GetProcAddress(handle, "restyleInitializeWithStyleStatistics"));
                                    restyleCalcAdainMomentsFunc = reinterpret_cast<decltype(restyleCalcAdainMomentsFunc)>(GetProcAddress(handle, "restyleCalcAdainMoments"));
                                    restyleForwardFunc = reinterpret_cast<decltype(restyleForwardFunc)>(GetProcAddress(handle, "restyleForward"));
                                    restyleForwardHDRFunc = reinterpret_cast<decltype(restyleForwardHDRFunc)>(GetProcAddress(handle, "restyleForwardHDR"));
                                    restyleDeinitializeFunc = reinterpret_cast<decltype(restyleDeinitializeFunc)>(GetProcAddress(handle, "restyleDeinitialize"));
                                    restyleResizeUsedFunc = reinterpret_cast<decltype(restyleResizeUsedFunc)>(GetProcAddress(handle, "restyleResizeUsed"));

                                    if (restyleUpdateStyleFunc != nullptr &&
                                        restyleIsInitializedFunc != nullptr &&
                                        restyleEstimateVRamUsageFunc != nullptr &&
                                        restyleInitializeWithStyleFunc != nullptr &&
                                        restyleInitializeWithStyleStatisticsFunc != nullptr &&
                                        restyleCalcAdainMomentsFunc != nullptr &&
                                        restyleForwardFunc != nullptr &&
                                        restyleForwardHDRFunc != nullptr &&
                                        restyleDeinitializeFunc != nullptr &&
                                        restyleResizeUsedFunc != nullptr)
                                    {
                                        loadedLibraries = true;
                                    }
                                    else
                                        loadedDllCheckFailed = true;
                                }
                                else
                                    loadedDllCheckFailed = true;
                            }
                            else
                                loadedDllCheckFailed = true;

                            // in case:
                            // a) restyleGetVersion was not found
                            // b) the version is not as expected
                            // c) not all of the needed functions were resolved
                            // unload this version of librestyle,
                            // disable style transfer and ask user if the
                            // download process should start
                            if (loadedDllCheckFailed)
                            {
                                FreeLibrary(handle);
                                m_activeControlClient->showRestyleDownloadConfirmation();
                                loadedLibraries = false;
                                m_activeControlClient->setStyleTransferStatus(false);
                                m_styleSelected = -1;
                                restyleGetVersionFunc = nullptr;
                                restyleUpdateStyleFunc = nullptr;
                                restyleIsInitializedFunc = nullptr;
                                restyleEstimateVRamUsageFunc = nullptr;
                                restyleInitializeWithStyleFunc = nullptr;
                                restyleInitializeWithStyleStatisticsFunc = nullptr;
                                restyleCalcAdainMomentsFunc = nullptr;
                                restyleForwardFunc = nullptr;
                                restyleForwardHDRFunc = nullptr;
                                restyleDeinitializeFunc = nullptr;
                                restyleResizeUsedFunc = nullptr;
                            }
                        }
                    }
                }
                else
                {
                    sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::LIBRESTYLE_NOT_FOUND));
                    m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_CouldntLoadLibrestyle);
                    loadedLibraries = false;
                    m_activeControlClient->setStyleTransferStatus(false);
                }
            }

            if (loadedLibraries && m_styleSelected > 0)
            {
                if (styleTransferEnabled && !checkStyleTransferTexture(m_styleTransferHDR, m_styleTransferOutputBuffer, m_styleTransferOutputHDRStorageBuffer, inputColorFormat, getWidth(), getHeight()))
                {
                    styleTransferTextureRecreationRequired = true;
                    // We need to determine how to create Style Transfer network, so need to update status early
                    m_styleTransferHDR = isStyleTransferHDR(inputColorFormat);
                }

                const auto status = restyleIsInitializedFunc();
                if ((status != Status::kOk) || reinitializeNetwork)
                {
                    if (reinitializeNetwork && (restyleIsInitializedFunc() == Status::kOk))
                        restyleDeinitializeFunc();

                    const auto stats = loadStyleTransferCache(m_stylesFilesList[m_styleSelected - 1], m_activeControlClient->getLwrrentStylePath());
                    if (!stats.first.empty() && !stats.second.empty())
                    {
                        const auto network = m_activeControlClient->getLwrrentStyleNetwork();
                        if (networks.find(network) == networks.cend())
                        {
                            sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::MODEL_NOT_FOUND));
                            m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_NoModelFound);
                        }
                        else
                        {
                            const auto net = networks.at(network);
                            const auto encoderPath = modelRoot + net.first;
                            const auto decoderPath = modelRoot + net.second;
                            const auto encoderPathN = darkroom::getUtf8FromWstr(encoderPath);
                            const auto decoderPathN = darkroom::getUtf8FromWstr(decoderPath);

                            size_t model_vram = 0;
                            restyleEstimateVRamUsageFunc(encoderPathN.c_str(), decoderPathN.c_str(), 32, 32, getHeight(), getWidth(), model_vram, forward_vram_estimation, false);
                            Status status;
                            try
                            {
                                status = restyleInitializeWithStyleStatisticsFunc(encoderPathN.c_str(),
                                    decoderPathN.c_str(), stats.first.data(), stats.second.data(),
                                    stats.first.size(), m_prevHeight, m_prevWidth, m_styleTransferHDR, false);
                                previousNetwork = network;
                            }
                            catch (const std::bad_alloc&)
                            {
                                status = Status::kFailedNotEnoughMemory;
                            }
                            if (status == Status::kFailedNotEnoughMemory)
                            {
                                sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::NOT_ENOUGH_VRAM));
                                m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_NotEnoughVRAM);
                                m_activeControlClient->setStyleTransferStatus(false);
                                m_styleSelected = -1;
                            }
                            else if (status == Status::kFailed)
                            {
                                sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::INITIALIZATION_FAILED));
                                m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_FailedToInitalizeStyleTransfer);
                                m_activeControlClient->setStyleTransferStatus(false);
                                m_styleSelected = -1;
                            }
                        }
                    }
                    else
                    {
                        sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::LOADING_STYLE_FAILED));
                        m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_FailedToLoadStyle);
                    }
                }
                else
                {
                    // in memory cache is the first level cache, file cache is the second level cache
                    // if first level cache check was a miss, check second level cache
                    const auto& stats = loadStyleTransferCache(m_stylesFilesList[m_styleSelected - 1], m_activeControlClient->getLwrrentStylePath());
                    if (!stats.first.empty() && !stats.second.empty())
                        restyleUpdateStyleFunc(stats.first.data(), stats.second.data(), stats.first.size());
                    else
                    {
                        sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::LOADING_STYLE_FAILED));
                        m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_FailedToLoadStyle);
                    }
                }
            }
        }
        else if (m_styleSelected > 0 &&
                styleTransferEnabled &&
                loadedLibraries &&
                    (!checkStyleTransferTexture(m_styleTransferHDR, m_styleTransferOutputBuffer, m_styleTransferOutputHDRStorageBuffer, inputColorFormat, getWidth(), getHeight()))
                )
        {
            styleTransferTextureRecreationRequired = true;
        }

        if (styleTransferTextureRecreationRequired)
        {
            m_styleTransferOutputBuffer.fmt = inputColorFormat;
            SAFE_RELEASE(m_styleTransferOutputBuffer.tex);

            // For now, we only support 8 bit formats
            if (isStyleTransferHDR(inputColorFormat))
            {
                m_styleTransferHDR = true;

                // initialize style transfer texture here
                m_styleTransferOutputBuffer.tex = createFullscreenTexture(DXGI_FORMAT_R32G32B32A32_FLOAT, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE);
                m_styleTransferOutputBuffer.w = getWidth();
                m_styleTransferOutputBuffer.h = getHeight();
                if (m_styleTransferOutputBuffer.fmt != DXGI_FORMAT_R32G32B32A32_FLOAT)
                {
                    SAFE_RELEASE(m_styleTransferOutputHDRStorageBuffer.tex);

                    m_styleTransferOutputHDRStorageBuffer.fmt = m_styleTransferOutputBuffer.fmt;
                    m_styleTransferOutputHDRStorageBuffer.tex = createFullscreenTexture(m_styleTransferOutputBuffer.fmt, D3D11_USAGE_DEFAULT, 0, 0);
                    m_styleTransferOutputHDRStorageBuffer.w = getWidth();
                    m_styleTransferOutputHDRStorageBuffer.h = getHeight();
                    m_styleTransferOutputBuffer.fmt = DXGI_FORMAT_R32G32B32A32_FLOAT;
                }
            }
            else
            {
                m_styleTransferHDR = false;

                // initialize style transfer texture here
                m_styleTransferOutputBuffer.tex = createFullscreenTexture(m_styleTransferOutputBuffer.fmt, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE);
                m_styleTransferOutputBuffer.w = getWidth();
                m_styleTransferOutputBuffer.h = getHeight();
            }
        }

        // in case we're unable to initalize the new network, reset select of the network control to the previous value
        if ((previousNetwork != m_activeControlClient->getLwrrentStyleNetwork()) && styleTransferEnabled)
            m_activeControlClient->setLwrrentStyleNetwork(previousNetwork);

        ANSEL_PROFILE_STOP(ff12_styleTransfer);
    }

    const bool transferStyleWhileMoving = m_anselSDK.getCaptureState() != CAPTURE_NOT_STARTED || m_allowStyleTransferWhileMovingCamera;

    const bool uiInteractionActive = effectControlChanged || m_activeControlClient->isUIInteractionActive();
    const auto isSessionActive = m_anselSDK.isSessionActive();
    const auto isStyleTransferEnabled = m_activeControlClient->isStyleTransferEnabled();
    const auto frameNotChanged = (transferStyleWhileMoving || !m_anselSDK.isCameraChanged());
    Status restyleInitStatus = Status::kFailed;
    if (restyleIsInitializedFunc)
        restyleInitStatus = restyleIsInitializedFunc();
    if (isSessionActive &&
        !uiInteractionActive &&
        isStyleTransferEnabled &&
        frameNotChanged &&
        m_styleSelected > 0 &&
        (restyleInitStatus == Status::kOk))
    {
        ANSEL_PROFILE_START(ff13_styleTransfer, "Style Transfer");
        // if camera was changed:
        static bool firstTimeStyleInitialized = false;
        if ((!m_activeControlClient->getCameraDragActive() || m_allowStyleTransferWhileMovingCamera) &&
            (m_anselSDK.isCameraChanged() ||
            (m_anselSDK.getCaptureState() != CAPTURE_NOT_STARTED && m_anselSDK.isLwrrentShotCapture()) ||
            !firstTimeStyleInitialized || m_refreshStyleTransfer))
        {
            if (m_anselSDK.getCaptureState() != CAPTURE_NOT_STARTED)
            {
                m_refreshStyleTransferAfterCapture = true;
            }

            if (m_styleTransferHDR)
            {
                ID3D11Texture2D * hdrPrereadbackTexture = nullptr;
                bool colwersionRequired = false;
                if (inputColorFormat != m_styleTransferOutputBuffer.fmt)
                {
                    colwersionRequired = true;

                    AnselResourceData hdrPrereadbackData = {};
                    hdrPrereadbackData.format = pPresentResource->toClientRes.format;
                    hdrPrereadbackData.pTexture2D = pPresentResource->toClientRes.pTexture2D;
                    hdrPrereadbackData.width = getWidth();
                    hdrPrereadbackData.height = getHeight();
                    hdrPrereadbackData.sampleCount = 1;
                    hdrPrereadbackData.sampleQuality = 0;

                    if (!SUCCEEDED(status = m_renderBufferColwerter.getHDR32FTexture(hdrPrereadbackData, &hdrPrereadbackTexture)))
                    {
                        sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::HDR_COLWERT_FAILED));
                        LOG_ERROR("Save shot failed: failed to colwert HDR texture (%d)", (int)inputColorFormat);
                        HandleFailure();
                    }

                    m_immediateContext->CopySubresourceRegion(m_styleTransferOutputBuffer.tex, 0, 0, 0, 0, hdrPrereadbackTexture, 0, 0);
                }
                else
                {
                    m_immediateContext->CopySubresourceRegion(m_styleTransferOutputBuffer.tex, 0, 0, 0, 0, pPresentResource->toClientRes.pTexture2D, 0, 0);
                }

                const auto width = getWidth();
                const auto height = getHeight();
                D3D11_MAPPED_SUBRESOURCE msr;
                m_immediateContext->Map(m_styleTransferOutputBuffer.tex, 0, D3D11_MAP_READ_WRITE, 0, &msr);
                float * srcImageHDR = static_cast<float *>(msr.pData);
                // run style transfer in place
                {
                    size_t free_byte = 0u, total_byte = 0u;
                    lwdaError_t err = lwdaMemGetInfo(&free_byte, &total_byte);

                    Status forwardStatus = kOk;

                    if (lwdaSuccess != err)
                    {
                        m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_FailedToInitalizeStyleTransfer);
                        m_activeControlClient->setStyleTransferStatus(false);
                        m_styleSelected = -1;
                        forwardStatus = kFailed;
                    }

                    if (free_byte < forward_vram_estimation)
                    {
                        m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_NotEnoughVRAM);
                        m_activeControlClient->setStyleTransferStatus(false);
                        m_styleSelected = -1;
                        status = kFailedNotEnoughMemory;
                    }
                    else
                    {
                        status = restyleForwardHDRFunc(srcImageHDR, msr.RowPitch);
                    }

                    const auto styleIndex = m_styleSelected - 1;
                    if (styleIndex >= 0 && styleIndex < m_stylesFilesList.size() && m_oldStyleTelemetry != m_stylesFilesList[styleIndex])
                    {
                        m_oldStyleTelemetry = m_stylesFilesList[m_styleSelected - 1];
                        const auto net = darkroom::getUtf8FromWstr(m_activeControlClient->getLwrrentStyleNetwork()) + "/" + darkroom::getUtf8FromWstr(m_stylesFilesList[m_styleSelected - 1]);
                        if (forwardStatus == kOk)
                        {
                            sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::FORWARD_HDR_SUCCESS), net);
                        }
                        else if (forwardStatus == kFailed)
                        {
                            sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::FORWARD_HDR_FAILED), net);
                        }
                        else if (forwardStatus == kFailedNotEnoughMemory)
                        {
                            sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::FORWARD_HDR_FAILED_NOT_ENOUGH_VRAM), net);
                        }
                    }
                }

                // unmap m_styleTransferOutputTexture
                m_immediateContext->Unmap(m_styleTransferOutputBuffer.tex, 0);

                if (colwersionRequired)
                {
                    m_immediateContext->CopySubresourceRegion(hdrPrereadbackTexture, 0, 0, 0, 0, m_styleTransferOutputBuffer.tex, 0, 0);

                    ID3D11Texture2D * hdrStorageTexture = nullptr;

                    AnselResourceData hdrStorageData = {};
                    hdrStorageData.format = m_styleTransferOutputBuffer.fmt;
                    hdrStorageData.pTexture2D = hdrPrereadbackTexture;
                    hdrStorageData.width = getWidth();
                    hdrStorageData.height = getHeight();
                    hdrStorageData.sampleCount = 1;
                    hdrStorageData.sampleQuality = 0;

                    if (!SUCCEEDED(status = m_renderBufferColwerter.getHDRLwstomTexture(hdrStorageData, false, &hdrStorageTexture, m_styleTransferOutputHDRStorageBuffer.fmt)))
                    {
                        sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::HDR_STORAGE_COLWERT_FAILED));
                        LOG_ERROR("Failed to colwert ST HDR storage texture (%d)", (int)inputColorFormat);
                        HandleFailure();
                    }

                    m_immediateContext->CopySubresourceRegion(m_styleTransferOutputHDRStorageBuffer.tex, 0, 0, 0, 0, hdrStorageTexture, 0, 0);
                    m_immediateContext->CopySubresourceRegion(pPresentResource->toClientRes.pTexture2D, 0, 0, 0, 0, m_styleTransferOutputHDRStorageBuffer.tex, 0, 0);
                }
                else
                {
                    m_immediateContext->CopySubresourceRegion(m_styleTransferOutputHDRStorageBuffer.tex, 0, 0, 0, 0, m_styleTransferOutputBuffer.tex, 0, 0);
                    m_immediateContext->CopySubresourceRegion(pPresentResource->toClientRes.pTexture2D, 0, 0, 0, 0, m_styleTransferOutputHDRStorageBuffer.tex, 0, 0);
                }

                firstTimeStyleInitialized = true;
                if (m_refreshStyleTransfer)
                    m_refreshStyleTransfer = false;
            }
            else
            {
                // copy frame buffer into m_styleTransferOutputTexture staging texture
                m_immediateContext->CopySubresourceRegion(m_styleTransferOutputBuffer.tex, 0, 0, 0, 0, pPresentResource->toClientRes.pTexture2D, 0, 0);
                // map m_styleTransferOutputTexture
                const auto width = getWidth();
                const auto height = getHeight();
                D3D11_MAPPED_SUBRESOURCE msr;
                m_immediateContext->Map(m_styleTransferOutputBuffer.tex, 0, D3D11_MAP_READ_WRITE, 0, &msr);
                unsigned char* srcImage = static_cast<unsigned char*>(msr.pData);
                // run style transfer in place

                size_t free_byte = 0u, total_byte = 0u;
                lwdaError_t err = lwdaMemGetInfo(&free_byte, &total_byte);

                if (lwdaSuccess != err)
                {
                    m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_FailedToInitalizeStyleTransfer);
                    m_activeControlClient->setStyleTransferStatus(false);
                    m_styleSelected = -1;
                }

                Status status = Status::kOk;
                if (free_byte < forward_vram_estimation)
                {
                    m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_NotEnoughVRAM);
                    m_activeControlClient->setStyleTransferStatus(false);
                    m_styleSelected = -1;
                }
                else
                {
                    Status status = restyleForwardFunc(srcImage, msr.RowPitch);
                    if (status != Status::kOk)
                    {
                        m_activeControlClient->displayMessage(AnselUIBase::MessageType::kStyle_FailedToInitalizeStyleTransfer);
                        m_activeControlClient->setStyleTransferStatus(false);
                        m_styleSelected = -1;
                    }
                }

                const auto styleIndex = m_styleSelected - 1;
                if (styleIndex >= 0 && styleIndex < m_stylesFilesList.size() && m_oldStyleTelemetry != m_stylesFilesList[styleIndex])
                {
                    m_oldStyleTelemetry = m_stylesFilesList[m_styleSelected - 1];
                    const auto net = darkroom::getUtf8FromWstr(m_activeControlClient->getLwrrentStyleNetwork()) + "/" + darkroom::getUtf8FromWstr(m_stylesFilesList[m_styleSelected - 1]);;
                    if (status == kOk)
                    {
                        sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::FORWARD_SUCCESS), net);
                    }
                    else if (status == kFailed)
                    {
                        sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::FORWARD_FAILED), net);
                    }
                    else if (status == kFailedNotEnoughMemory)
                    {
                        sendTelemetryStyleTransferStatusEvent(uint32_t(LwTelemetry::Ansel::StyleTransferStatusEnum::FORWARD_FAILED_NOT_ENOUGH_VRAM), net);
                    }
                }
                // unmap m_styleTransferOutputTexture
                m_immediateContext->Unmap(m_styleTransferOutputBuffer.tex, 0);
                m_immediateContext->CopySubresourceRegion(pPresentResource->toClientRes.pTexture2D, 0, 0, 0, 0, m_styleTransferOutputBuffer.tex, 0, 0);
                firstTimeStyleInitialized = true;
                if (m_refreshStyleTransfer)
                    m_refreshStyleTransfer = false;
            }
        }
        else
        {
            if (m_refreshStyleTransferAfterCapture && (m_anselSDK.getCaptureState() == CAPTURE_NOT_STARTED))
            {
                m_refreshStyleTransfer = true;
                m_refreshStyleTransferAfterCapture = false;
            }

            // copy cached style transfer result staging texture into our framebuffer
            if (m_styleTransferHDR)
            {
                m_immediateContext->CopySubresourceRegion(pPresentResource->toClientRes.pTexture2D, 0, 0, 0, 0, m_styleTransferOutputHDRStorageBuffer.tex, 0, 0);
            }
            else
            {
                m_immediateContext->CopySubresourceRegion(pPresentResource->toClientRes.pTexture2D, 0, 0, 0, 0, m_styleTransferOutputBuffer.tex, 0, 0);
            }
        }
        ANSEL_PROFILE_STOP(ff13_styleTransfer);
    }
    else
        m_refreshStyleTransfer = true;
#endif

    // Download data from GPU (save screenshot)
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ANSEL_PROFILE_START(ff14_sdkPrepare, "SDK Prepare");

    if (m_registrySettings.isDirty())
    {
        m_activeControlClient->setHighresEnhance(m_enableEnhancedHighres);
    }
    m_enableEnhancedHighres = m_activeControlClient->isHighresEnhance();

    // We delay the initialization of titleName until it's needed.  The init is slow because is
    // requires loading of VdChip profiles (we could look at threading it out in the future):
    if (m_makeScreenshot || m_makeScreenshotWithUI)
    {
        m_anselSDK.initTitleAndDrsNames();
    }

    const auto& titleName = m_anselSDK.getTitleForFileNaming();

    AnselSDKCaptureParameters captureParameters;
    captureParameters.appName = m_appName.c_str();
    captureParameters.tagModel = m_deviceName;
    captureParameters.tagSoftware = generateSoftwareTag();
    captureParameters.tagDrsName = m_anselSDK.getDrsAppName();
    captureParameters.tagDrsProfileName = m_anselSDK.getDrsProfileName();
    captureParameters.tagAppTitleName = darkroom::getWstrFromUtf8(m_anselSDK.getTitleForTagging());
    captureParameters.tagAppCMSID = m_activeControlClient->getAppCMSID();
    captureParameters.tagDescription = GenerateB64MetadataDescription(m_anselSDK.getDrsAppName(), m_anselSDK.getDrsProfileName(), m_activeControlClient->getAppShortName(), m_activeControlClient->getAppCMSID());
    captureParameters.tagAppShortName = m_activeControlClient->getAppShortName();
    captureParameters.tagActiveFilters = generateActiveEffectsTag();
    captureParameters.makeScreenshot = m_makeScreenshot;
    captureParameters.shotToTake = m_shotToTake;
    captureParameters.depthTexture = inputDepthTexture;
    captureParameters.hudlessTexture = hudlessInput.texture;
    captureParameters.colorTexture = inputColorTexture;
    // In compliance with GFE we put every shot into a folder with the name 'titleName':
    captureParameters.snapshotPath = m_snapshotsFolderPath + titleName + L'\\';
    captureParameters.intermediateFolderPath = m_intermediateFolderPath;
    captureParameters.keepIntermediateShots = m_keepIntermediateShots;

    // With JXR, force Present Buffer to be used for screenshot if it's in a supported HDR format
    if (m_makeScreenshotHDRJXR)
    {
        captureParameters.isShotHDR = false;
        captureParameters.isShotHDRJXR = true;
        captureParameters.isShotRawHDR = false;
        captureParameters.isShotDisplayHDR = true;
    }
    else if (m_makeScreenshotHDR || m_makeScreenshotHDRJXR)
    {
        captureParameters.isShotHDR = m_makeScreenshotHDR;          // Ensures EXR export
        captureParameters.isShotHDRJXR = m_makeScreenshotHDRJXR;    // Ensures JXR export

        bool isRawHDRCaptureAllowedByIntegratedApp = !m_anselSDK.isDetected() || !m_anselSDK.isSessionActive() || m_anselSDK.getSessionConfiguration().isRawAllowed; // Also allowed if there is no integration.
        if (isRawHDRCaptureAllowedByIntegratedApp && isRawHDRBufferAvailable)
        {
            // If a raw HDR buffer is allowed and available, prefer to use that.
            captureParameters.isShotRawHDR = true;
            captureParameters.isShotDisplayHDR = false;
        }
        else if (isBufferPresentableHDR)
        {
            // Otherwise use the presentable HDR buffer.
            captureParameters.isShotRawHDR = false;
            captureParameters.isShotDisplayHDR = true;
        }
        else
        {
            // This should never be reached because the option to "Save as HDR" is supposed to be greyed out when no HDR buffers are available.
            LOG_ERROR("Save as HDR requested, but no HDR buffers available!");
            captureParameters.isShotHDR = false;
            captureParameters.isShotHDRJXR = false;
            captureParameters.isShotRawHDR = false;
            captureParameters.isShotDisplayHDR = false;
        }
    }
    else
    {
        captureParameters.isShotHDR = false;
        captureParameters.isShotHDRJXR = false;
        captureParameters.isShotRawHDR = false;
        captureParameters.isShotDisplayHDR = isBufferPresentableHDR;
    }

    captureParameters.isEnhancedHighresEnabled = m_enableEnhancedHighres;
    captureParameters.enhancedHighresCoefficient = m_enhancedHighresCoeff;
    captureParameters.generateThumbnail = (shotDesc.thumbnailRequired || m_makeScreenshotHDRJXR);

    captureParameters.pPresentResourceData = &pSelectedCaptureResource->toClientRes;
    if (captureParameters.isShotRawHDR)
    {
        captureParameters.pPresentResourceData = &pHDRResource->toServerRes;
    }
    else
    {
        DXGI_FORMAT colwertedFormat = lwanselutils::colwertFromTypelessIfNeeded(pSelectedCaptureResource->toServerRes.format);
        // Probably this should be a different function, since 'isHdrFormatSupported' is actually related to
        // scene-referred HDR, and we're looking at output-referred HDR at the moment
        if (isHdrFormatSupported( colwertedFormat ))
        {
            captureParameters.isShotDisplayHDR = true;
        }

        // This is hardcoded for now, but iun the future this should be per-title setting
        // The reason for that is e.g. Obduction - it displays simple sRGB as RGBA10 for some reason
        if (colwertedFormat == DXGI_FORMAT_R10G10B10A2_UNORM)
        {
            // In case game presents sRGBA in RGBA10, we just need to clamp it in SaveShot, it's not real HDR
            captureParameters.isShotDisplayHDR = false;
        }
    }

    captureParameters.isShotPreviewRequired = m_activeControlClient->isShotPreviewRequired();
    if (m_makeScreenshotHDR)
    {
        captureParameters.pPresentResourceDataAdditional = shotDesc.thumbnailRequired ? &pPresentResource->toClientRes : nullptr;
    }
    else if (m_makeScreenshotHDRJXR)
    {
        captureParameters.pPresentResourceDataAdditional = &pPresentResource->toClientRes;
    }
    else
    {
        captureParameters.pPresentResourceDataAdditional = m_activeControlClient->isShotPreviewRequired() ? (&pSelectedCaptureResource->toClientRes) : nullptr;
    }

    captureParameters.width = getWidth();
    captureParameters.height = getHeight();
    captureParameters.forceLosslessSuperRes = m_losslessOutputSuperRes;
    captureParameters.forceLossless360 = m_losslessOutput360;
    captureParameters.restoreRoll = m_activeControlClient->isResetRollNeeded();

    AnselSDKUIParameters inputParameters;
    inputParameters.isRollChanged = m_activeControlClient->processRollChange();
    inputParameters.rollDegrees = (float)m_activeControlClient->getRollDegrees();
    if (std::abs(inputParameters.rollDegrees) < 1e-2)
        inputParameters.rollDegrees = 0.0f;
    inputParameters.isCameraDragActive = m_activeControlClient->getCameraDragActive();
    inputParameters.isFOVChanged = m_activeControlClient->processFOVChange();
    inputParameters.fovSliderValue = (float)m_activeControlClient->getFOVDegrees();
    inputParameters.cameraSpeedMultiplier = m_cameraSpeedMultiplier;
    inputParameters.eyeSeparation = m_eyeSeparation;
    inputParameters.highresMultiplier = shotDesc.highResMult;
    inputParameters.sphericalQualityFov = (float)darkroom::CameraDirector::estimateTileHorizontalFovSpherical(uint32_t(shotDesc.resolution360 * 2), getWidth());
    inputParameters.isHighQualityEnabled = m_activeControlClient->isHighQualityEnabled();
    inputParameters.uiInterface = m_activeControlClient;

    AnselSDKMiscParameters miscParameters;
    miscParameters.dtSeconds = m_globalPerfCounters.dt * 0.001;
    miscParameters.useHybridController = m_useHybridController;

    ANSEL_PROFILE_STOP(ff14_sdkPrepare);

    ANSEL_PROFILE_START(ff15_sdkUpdate, "SDK Update");

    if ((captureParameters.shotToTake != ShotType::kNone || captureParameters.makeScreenshot) &&
        (m_makeScreenshotHDR || m_makeScreenshotHDRJXR) &&
        (!isRawHDRBufferCaptured && isRawHDRBufferAvailable))
    {
        // delay a frame to capture the raw HDR buffer.
        m_makeScreenshotNextFrame = true;

        m_makeScreenshotNextFrameValue = captureParameters.makeScreenshot;
        captureParameters.makeScreenshot = false;
        m_shotToTakeNextFrame = captureParameters.shotToTake;
        captureParameters.shotToTake = ShotType::kNone;

        m_hdrBufferUsedByShot = true;
    }
    else if (m_makeScreenshotNextFrame)
    {
        captureParameters.makeScreenshot = m_makeScreenshotNextFrameValue;
        captureParameters.shotToTake = m_shotToTakeNextFrame;

        m_shotToTakeNextFrame = ShotType::kNone;
        m_makeScreenshotNextFrameValue = false;
        m_makeScreenshotNextFrame = false;
    }
    else if ((!captureParameters.makeScreenshot && captureParameters.shotToTake == ShotType::kNone) || (!m_makeScreenshotHDR && !m_makeScreenshotHDRJXR))
    {
        m_hdrBufferUsedByShot = false;
    }
    // update Ansel SDK - give input parameters and get output parameters
    const AnselSDKUpdateParameters outputParameters = m_anselSDK.update(&m_activeControlClient->getInputHandler(), m_activeControlClient->isCameraInteractive(), m_errorManager, captureParameters, inputParameters, miscParameters, this);

    m_activeControlClient->resetRollDone();
    m_activeControlClient->setResetRollStatus(outputParameters.isResetRollAvailable);

    ANSEL_PROFILE_STOP(ff15_sdkUpdate);

    ANSEL_PROFILE_START(ff16_sdkPost, "SDK Post-actions");

    // Done propagating reg changes, need to clear dirty flag
    m_registrySettings.clearDirty();

    m_makeScreenshot = outputParameters.makeScreenshot;
    m_shotToTake = outputParameters.shotToTake;

    if (outputParameters.needsFOVupdate)
    {
        m_activeControlClient->setFOVDegrees(outputParameters.lwrrentHorizontalFov);
    }
    m_activeControlClient->setRollDegrees(outputParameters.roll);

    // Disable "Snap" button
    if (outputParameters.needToDisableSnapButton || wasShotRegular)
    {
        int numShotsTotal = m_anselSDK.getProgressTotal();
        m_activeControlClient->onCaptureStarted(numShotsTotal);
        m_captureStartTime = m_globalPerfCounters.elapsedTime;
        LOG_INFO("onCaptureStarted (%d) event issued", numShotsTotal);
    }

    if (outputParameters.screenshotTaken)
    {
        m_activeControlClient->onCaptureTaken(m_anselSDK.getProgressLwrrent());
        LOG_INFO("onCaptureTaken (%d) event issued", m_anselSDK.getProgressLwrrent());

#if ANSEL_SIDE_PRESETS
        if (m_registrySettings.getValue(m_registrySettings.registryPathAnsel(), darkroom::getWstrFromUtf8(Settings::OutputPresets).c_str(), false))
        {
            // First, generate the preset name
            size_t filenamePosition = outputParameters.processedAbsPath.rfind(L'.');
            std::wstring presetFilename = outputParameters.processedAbsPath.substr(0, filenamePosition) + L".ini";

            // Generate AnselPreset from current filters
            AnselPreset preset;

            assert(m_effectsInfo.m_effectSelected.size() == m_effectsInfo.m_effectsStack.size());
            for (size_t i = 0; i < m_effectsInfo.m_effectsStack.size(); i++)
            {
                shadermod::MultiPassEffect* effect = m_effectsInfo.m_effectsStack[i];
                if (effect)
                {
                    AnselFilter filter;

                    long effectSelected = m_effectsInfo.m_effectSelected[i] - 1; // effectSelected is 1-based, effectFilterIds is 0-based
                    assert(effectSelected >= 0);
                    filter.filterID = m_effectsInfo.m_effectFilterIds[effectSelected];

                    for (auto ucIter = effect->getUserConstantManager().getPointersToAllUserConstants().begin; ucIter < effect->getUserConstantManager().getPointersToAllUserConstants().end; ucIter++)
                    {
                        std::pair <std::wstring, std::vector<float>> attribute;

                        std::string attributeName = (*ucIter)->getName();
                        unsigned int dimensionality = (*ucIter)->getValue().getDimensionality();

                        attribute.first = std::wstring(attributeName.begin(), attributeName.end());
                        attribute.second.resize(dimensionality);
                        (*ucIter)->getValue().get(attribute.second.data(), dimensionality);

                        filter.attributes.push_back(attribute);
                    }

                    preset.filters.push_back(filter);
                }
            }

            if (preset.filters.size() > 0)
            {
                std::vector<std::wstring> duplicateFilters;
                ExportAnselPreset(presetFilename, preset, duplicateFilters);

                if (duplicateFilters.size() != 0)
                {
                    // Some filters have not been applied successfully
                    std::wstring fullError = L"Warning: ";

                    // Join based on https://www.oreilly.com/library/view/c-cookbook/0596007612/ch04s09.html
                    for (std::vector<std::wstring>::const_iterator p = duplicateFilters.begin(); p != duplicateFilters.end(); ++p)
                    {
                        LOG_WARN("Duplicate filter %s during Preset Export", darkroom::getUtf8FromWstr(*p).c_str());
                        fullError += *p;
                        if (p != duplicateFilters.end() - 1)
                            fullError += L", ";
                    }

                    fullError += L" also exists in preset. Only the first copy ";
                    if (duplicateFilters.size() > 1)
                    {
                        fullError += L"of each ";
                    }
                    fullError += L"will be saved.";

                    m_displayMessageStorage.resize(0);
                    m_displayMessageStorage.push_back(m_effectsInfo.m_effectFilterIds[m_effectsInfo.m_effectSelected[0] - 1]);
                    m_displayMessageStorage.push_back(fullError);
                    m_activeControlClient->displayMessage(AnselUIBase::MessageType::kEffectRequiresDepth, m_displayMessageStorage, true);
                }
            }
        }
#endif
    }

    if (outputParameters.processingCompleted)
    {
        m_activeControlClient->onCaptureProcessingDone(0, outputParameters.processedAbsPath);
        LOG_INFO("onCaptureProcessingDone event issued");
    }

    // TODO avoroshilov UI
    //  check if 'outputParameters.needToFinalizeDirector' is set when capture is aborted?
    //  seems so at the first glance
    if (outputParameters.needToFinalizeDirector || wasShotRegular)
    {
        m_activeControlClient->onCaptureStopped(outputParameters.captureStatus);
        LOG_INFO("onCaptureStopped event issued");
    }

    ANSEL_PROFILE_STOP(ff16_sdkPost);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // UI ///

    ANSEL_PROFILE_START(ff17_UI_render, "UI Render");

    UIProgressInfo progressinfo;
    progressinfo.removeBlackTint = m_removeBlackTint;

    UIDebugPrintInfo debugPrintInfo;
    debugPrintInfo.dt = static_cast<float>(m_globalPerfCounters.dt);
    debugPrintInfo.renderDebugInfo = m_renderDebugInfo;
    debugPrintInfo.shotCaptureLatency = m_anselSDK.getCaptureLatency();
    debugPrintInfo.shotSettleLatency = m_anselSDK.getSettleLatency();
    debugPrintInfo.networkBytesTransferred = uint32_t(m_networkDetector.bytesTransferred());

#if DEBUG_GAMEPAD

    debugPrintInfo.gamepadDebugInfo.a = m_activeControlClient->getInputHandler().m_gamepadState.isButtonDown(input::EGamepadButton::kA);
    debugPrintInfo.gamepadDebugInfo.b = m_activeControlClient->getInputHandler().m_gamepadState.isButtonDown(input::EGamepadButton::kB);
    debugPrintInfo.gamepadDebugInfo.x = m_activeControlClient->getInputHandler().m_gamepadState.isButtonDown(input::EGamepadButton::kX);
    debugPrintInfo.gamepadDebugInfo.y = m_activeControlClient->getInputHandler().m_gamepadState.isButtonDown(input::EGamepadButton::kY);
    debugPrintInfo.gamepadDebugInfo.lcap = m_activeControlClient->getInputHandler().m_gamepadState.isButtonDown(input::EGamepadButton::kLeftStickPress);
    debugPrintInfo.gamepadDebugInfo.rcap = m_activeControlClient->getInputHandler().m_gamepadState.isButtonDown(input::EGamepadButton::kRightStickPress);
    debugPrintInfo.gamepadDebugInfo.lshoulder = m_activeControlClient->getInputHandler().m_gamepadState.isButtonDown(input::EGamepadButton::kLeftShoulder);
    debugPrintInfo.gamepadDebugInfo.rshoulder = m_activeControlClient->getInputHandler().m_gamepadState.isButtonDown(input::EGamepadButton::kRightShoulder);
    debugPrintInfo.gamepadDebugInfo.dpad = m_activeControlClient->getInputHandler().m_gamepadState.getDpadDirection();
    debugPrintInfo.gamepadDebugInfo.fz = input::GamepadState::axisToFloat(m_activeControlClient->getInputHandler().m_gamepadState.getAxisZ());
    debugPrintInfo.gamepadDebugInfo.lx = m_activeControlClient->getInputHandler().m_gamepadState.getAxisLX();
    debugPrintInfo.gamepadDebugInfo.rx = m_activeControlClient->getInputHandler().m_gamepadState.getAxisRX();
    debugPrintInfo.gamepadDebugInfo.ly = m_activeControlClient->getInputHandler().m_gamepadState.getAxisLY();
    debugPrintInfo.gamepadDebugInfo.ry = m_activeControlClient->getInputHandler().m_gamepadState.getAxisRY();
    debugPrintInfo.gamepadDebugInfo.z = m_activeControlClient->getInputHandler().m_gamepadState.getAxisZ();

#endif

    if (m_activeControlClient->isGridOfThirdsEnabled() && !isMultiPartCapture)
    {
        D3D11_VIEWPORT viewPort;
        viewPort.Width = float(pPresentResource->toServerRes.width);
        viewPort.Height = float(pPresentResource->toServerRes.height);
        viewPort.MinDepth = 0.0f;
        viewPort.MaxDepth = 1.0f;
        viewPort.TopLeftX = 0;
        viewPort.TopLeftY = 0;

        m_immediateContext->RSSetViewports(1, &viewPort);

        m_immediateContext->VSSetShader(m_passthroughEffect.pVertexShader, NULL, 0);
        m_immediateContext->PSSetShader(m_gridEffect.pPixelShader, NULL, 0);

        m_immediateContext->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        // We only need texture dimensions in the shader, not the contents => toServerRes should be fine
        m_immediateContext->PSSetShaderResources(0, 1, &pPresentResource->toServerRes.pSRV);

        m_immediateContext->IASetInputLayout(0);
        m_immediateContext->PSSetSamplers(0, 1, &m_passthroughEffect.pSamplerState);
        m_immediateContext->RSSetState(m_passthroughEffect.pRasterizerState);
        m_immediateContext->OMSetDepthStencilState(m_passthroughEffect.pDepthStencilState, 0xFFFFFFFF);
        m_immediateContext->OMSetRenderTargets(1, &pPresentResource->toClientRes.pRTV, NULL);
        m_immediateContext->OMSetBlendState(m_passthroughEffect.pBlendState, NULL, 0xffffffff);
        m_immediateContext->Draw(3, 0);
    }

    if (m_specialClientActive)
    {
        if (getIPCModeEnabled() != 0)
        {
        }
        else
        {
            AnselUI * UI_standalone = static_cast<AnselUI *>(m_UI);
            UI_standalone->render(
                m_immediateContext,
                pPresentResource,
                &m_passthroughEffect,
                m_UI->getInputHandler().hasFolws() && m_UI->getInputHandler().isMouseInClientArea(),
                isHdrFormatSupported( inputColorFormat ),
                m_errorManager,
                progressinfo,
                debugPrintInfo,
                m_anselSDK.isDetected(),
                m_anselSDK.getDisplayCamera()
                );
        }
    }

    ANSEL_PROFILE_STOP(ff17_UI_render);

#if (ENABLE_DEBUG_RENDERER != 0)
    m_debugRenderer.renderFPS(
                    m_globalPerfCounters.dt, DebugRenderer::FPSCounterPos::kTOP_RIGHT,
                    getWidth(), getHeight(),
                    m_passthroughEffect.pRasterizerState, m_passthroughEffect.pSamplerState, m_passthroughEffect.pDepthStencilState,
                    pPresentResourceData->toClientRes.pRTV
                    );
#endif

    ANSEL_PROFILE_START(ff18_UI_post, "UI Post-actions");

    // Download data from GPU (save screenshot)
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    if (m_makeScreenshotWithUI)
    {
        m_makeScreenshotWithUI = false;

        // Get file name similar to how Regular screenshots are named
        const auto now = darkroom::generateTimestamp();
        std::wstring filenameName = darkroom::generateFileName(darkroom::JobType::REGULAR, now, m_anselSDK.getTitleForFileNaming(), L".png");

        // Make it "Screenshot-UI"
        size_t screenshotSubstrPos = filenameName.rfind(L"Screenshot", std::wstring::npos);
        filenameName = filenameName.substr(0, screenshotSubstrPos + 10) + L"-UI" + filenameName.substr(screenshotSubstrPos + 10);

        const std::wstring screenshotFullFilename = captureParameters.snapshotPath + std::wstring(filenameName);

        captureParameters.pPresentResourceData = static_cast<AnselResourceData *>(&pPresentResource->toClientRes);
        captureParameters.width = getWidth();
        captureParameters.height = getHeight();

        HRESULT shotStatus = saveShot(captureParameters, false, screenshotFullFilename);

        AnselUIBase::MessageType uiShotMsgType;
        if (shotStatus == S_OK)
            //  Shot with UI saved: %ls
            uiShotMsgType = AnselUIBase::MessageType::kShotWithUISaved;
        else if (shotStatus == E_FAIL)
            //  Unable to save shot with UI: %ls
            uiShotMsgType = AnselUIBase::MessageType::kUnableToSaveShotWithUI;
        m_displayMessageStorage.resize(0);
        m_displayMessageStorage.push_back(screenshotFullFilename.c_str());
        m_activeControlClient->displayMessage(uiShotMsgType, m_displayMessageStorage);
    }

    m_activeControlClient->shotCaptureDone(AnselUIBase::Status::kOk);

    ANSEL_PROFILE_STOP(ff18_UI_post);

    //feodorb TODO: wtf?
    // TODO: move that into UI mouse state reset function
    //memcpy(m_inputstate.gamepadState().button_wasPressed, m_inputstate.gamepadState().button_pressed, sizeof(bool)*UI_GAMEPAD_MAX_BUTTONS);
    //m_inputstate.mouseState().lmb_wasDown = m_inputstate.mouseState().lmb_down;
    //m_inputstate.mouseState().mmb_wasDown = m_inputstate.mouseState().mmb_down;
    //m_inputstate.mouseState().rmb_wasDown = m_inputstate.mouseState().rmb_down;
    //m_inputstate.mouseState().coordsAclwmX = 0;
    //m_inputstate.mouseState().coordsAclwmY = 0;

    ANSEL_PROFILE_START(ff19_buffersRelease, "Buffers Release");

    m_bufDB.release();

    ANSEL_PROFILE_STOP(ff19_buffersRelease);

#if DBG_HARDCODED_EFFECT_BW_ENABLED
    m_bEnableBlackAndWhite = m_bNextFrameEnableBlackAndWhite;
#endif
    m_bRenderDepthAsRGB = m_bNextFrameRenderDepthAsRGB;
    m_bRunShaderMod = m_bNextFrameRunShaderMod;

    m_bNextFramePrevEffect = false;
    m_bNextFrameNextEffect = false;
    m_bNextFrameRebuildYAML = false;
    m_bNextFrameRefreshEffectStack = false;

    m_bWasAnselDeactivated = false;

    ANSEL_PROFILE_ENDFRAME();

    return status;
}

HRESULT AnselServer::exelwtePostProcessing(HCLIENTRESOURCE hPresentResource, DWORD subResIndex)
{
    static bool kaboomHappenned = false;

    if (kaboomHappenned)
        return S_OK;

    const auto exception_filter = [&](LPEXCEPTION_POINTERS exceptionInfo)
    {
        const void* exceptionAddress = exceptionInfo->ExceptionRecord->ExceptionAddress;
        const auto exceptionCode = exceptionInfo->ExceptionRecord->ExceptionCode;
        std::array<void*, 32> frames = { 0 };
        CaptureStackBackTrace(0, DWORD(frames.size()), &frames[0], 0);
        const HMODULE baseAddr = GetModuleHandle(MODULE_NAME_W);
        // Changes to this error string should also be reflected in externals/AnselErrorStringParser
        reportFatalError(__FILE__, __LINE__, FatalErrorCode::kPostProcessFail, "exelwtePostProcessing: top level exception handler exelwted (eCode=%d, eAddr=%p, baseAddr=%p, stacktrace=%p,%p,%p,%p,%p,%p,%p,%p,%p,%p,%p,%p,%p,%p,%p,%p",
            exceptionCode, exceptionAddress, baseAddr,
            frames[0], frames[1], frames[2], frames[3], frames[4], frames[5], frames[6], frames[7],
            frames[8], frames[9], frames[10], frames[11], frames[12], frames[13], frames[14], frames[15]);
        return EXCEPTION_EXELWTE_HANDLER;
    };


#if (DBG_EXCEPTIONS_PASSTHROUGH == 1)
#else
    __try
    {
#endif

        HRESULT status = S_OK;
        setBufferInterfaceD3D11();
        if (!SUCCEEDED(status = finalizeFrame(hPresentResource, subResIndex)))
        {
            LOG_ERROR("Frame processing error");
            // We're telling the game that the session should be ended to unpause it if needed
            if (m_anselSDK.isSDKDetectedAndSessionActive())
            {
                m_anselSDK.stopClientSession();
                m_activeControlClient->emergencyAbort();
            }
            reportFatalError(__FILE__, __LINE__, FatalErrorCode::kFrameProcessFail, "Deactivating LwCamera due to frame processing error");
            deactivateAnsel();
        }


        if (!m_lightweightShimMode || m_bNotifyRMAnselStop)
        {
            notifyFrameComplete(m_bNotifyRMAnselStop);
            m_bNotifyRMAnselStop = false;
        }

        return S_OK;

#if (DBG_EXCEPTIONS_PASSTHROUGH == 1)
#else
    }
    __except (exception_filter(GetExceptionInformation()))
    {
        kaboomHappenned = true;
        // We're telling the game that the session should be ended to unpause it if needed
        if (m_anselSDK.isSDKDetectedAndSessionActive())
        {
            m_anselSDK.stopClientSession();
            m_activeControlClient->emergencyAbort();
        }
        // We're telling teh UMD to put Ansel shim in "passthrough" mode (no Ansel-related activity)
        reportFatalError(__FILE__, __LINE__, FatalErrorCode::kOtherException, "Deactivating LwCamera due to generic exception");
        deactivateAnsel();
    }
#endif

    return S_OK;
}

AnselServer::AnselStateForTelemetry AnselServer::makeStateSnapshotforTelemetryBody() const
{
    AnselStateForTelemetry ret;

    ret.exeName = darkroom::getUtf8FromWstr(m_appName);
    darkroom::tolowerInplace(ret.exeName);
    // It should be noted here that in older versions of the code base the rawProfileName was
    // assigned the exe name as a fall back - this means that it contained the name used for tagging
    // pictures. To stay consistent with older data we use that identifier here:
    ret.rawProfileName = m_anselSDK.getTitleForTagging();
    ret.drsAppName = darkroom::getUtf8FromWstr(m_anselSDK.getDrsAppName());

    ret.width = getWidth();
    ret.height = getHeight();
    ret.colorBufferFormat = m_colorBufferFormat;
    ret.depthBufferFormat = m_depthBufferFormat;

    ansel::Camera dispCam = m_anselSDK.getDisplayCamera();
    ansel::Camera origCam = m_anselSDK.getOriginalCamera();

    ret.dispCam_position_x = dispCam.position.x;
    ret.dispCam_position_y = dispCam.position.y;
    ret.dispCam_position_z = dispCam.position.z;
    ret.dispCam_rotation_x = dispCam.rotation.x;
    ret.dispCam_rotation_y = dispCam.rotation.y;
    ret.dispCam_rotation_z = dispCam.rotation.z;
    ret.dispCam_rotation_w = dispCam.rotation.w;
    ret.origCam_position_x = origCam.position.x;
    ret.origCam_position_y = origCam.position.y;
    ret.origCam_position_z = origCam.position.z;
    ret.origCam_rotation_x = origCam.rotation.x;
    ret.origCam_rotation_y = origCam.rotation.y;
    ret.origCam_rotation_z = origCam.rotation.z;
    ret.origCam_rotation_w = origCam.rotation.w;
    if (m_anselSDK.isDetected())
    {
        ret.anselSDKMajor = m_anselSDK.getAnselSdkVersionMajor();
        ret.anselSDKMinor = m_anselSDK.getAnselSdkVersionMinor();
        ret.anselSDKCommit = m_anselSDK.getAnselSdkVersionCommit();
    }

    if (m_activeControlClient)
    {
        m_activeControlClient->getTelemetryData(ret.uiSpecificData);
        ret.fov360 = (float)darkroom::CameraDirector::estimateTileHorizontalFovSpherical(uint32_t(ret.uiSpecificData.resolution360 * 2), uint32_t(ret.width));

        input::GamepadDevice::GamepadStats stats;
        m_activeControlClient->getInputHandler().getGamepadStats(stats);
        ret.gamepadProductId = stats.dwProductId;
        ret.gamepadVendorId = stats.dwVendorId;
        ret.gamepadVersionNumber = stats.dwVersionNumber;
        ret.gamepadDeviceType = stats.type;

        size_t numFilters = m_effectsInfo.m_effectSelected.size();

        if (numFilters != m_effectsInfo.m_effectsStack.size() || numFilters != m_effectsInfo.m_effectRebuildRequired.size())
            numFilters = 0; //state borked

        bool isAtLeastOneEffectOnStack = false;

        std::stringbuf packedUserConstants;
        std::ostream strstreamUC(&packedUserConstants);

        std::stringbuf effectNames;
        std::ostream strstreamNames(&effectNames);

        for (size_t i = 0; i < numFilters; ++i)
        {
            shadermod::MultiPassEffect* eff = nullptr;

            int effectNameIdx = m_effectsInfo.m_effectSelected[i];
            std::string name = "None";

            if (effectNameIdx > 0)
            {
                if ((size_t) effectNameIdx > m_effectsInfo.m_effectFilterIds.size())
                {
                    numFilters = 0; //state borked
                    break;
                }

                isAtLeastOneEffectOnStack = true;

                const std::wstring& wname = m_effectsInfo.m_effectFilterIds[effectNameIdx - 1];
                name = darkroom::getUtf8FromWstr(wname);
            }

            bool isValid = !m_effectsInfo.m_effectRebuildRequired[i];

            if (isValid)
                eff = m_effectsInfo.m_effectsStack[i];

            strstreamNames << name << "\n";

            if (eff)
            {
                AnselUIBase::EffectPropertiesDescription effectDesc;
                getEffectDescription(eff, m_activeControlClient->getLangId(), &effectDesc);

                unsigned long numuserconstants = (unsigned long)effectDesc.attributes.size();
                strstreamUC << numuserconstants << "\n";

                for (unsigned long j = 0; j < numuserconstants; ++j)
                {
                    const shadermod::ir::UserConstant* uc = effectDesc.attributes[j].userConstant;

                    assert(uc);

                    if (!uc)
                    {
                        strstreamUC << "INVALID\nbool\nFalse\n";
                        continue;
                    }

                    if (i == 0 && j < 5)
                    {
                        AnselStateForTelemetry::UserVarData& var = ret.top5Var[j];
                        var.type = uc->getType();

                        shadermod::ir::userConstTypes::Bool boolVal = shadermod::ir::userConstTypes::Bool::kFalse;
                        shadermod::ir::userConstTypes::Int intVal = 0;
                        shadermod::ir::userConstTypes::Int uintVal = 0;
                        shadermod::ir::userConstTypes::Float floatVal = 0.0f;

                        switch (uc->getType())
                        {
                        case shadermod::ir::UserConstDataType::kBool:
                            uc->getValue(boolVal);
                            var.value = shadermod::ir::stringify(boolVal);
                            break;
                        case shadermod::ir::UserConstDataType::kInt:
                            uc->getValue(intVal);
                            var.value = shadermod::ir::stringify(intVal);
                            break;
                        case shadermod::ir::UserConstDataType::kUInt:
                            uc->getValue(uintVal);
                            var.value = shadermod::ir::stringify(uintVal);
                            break;
                        case shadermod::ir::UserConstDataType::kFloat:
                            uc->getValue(floatVal);
                            var.value = shadermod::ir::stringify(floatVal);
                            break;
                        }

                        var.name = uc->getName();
                    }

                    strstreamUC << uc->getName();

                    shadermod::ir::userConstTypes::Bool boolVal = shadermod::ir::userConstTypes::Bool::kFalse;
                    shadermod::ir::userConstTypes::Int intVal = 0;
                    shadermod::ir::userConstTypes::Int uintVal = 0;
                    shadermod::ir::userConstTypes::Float floatVal = 0.0f;

                    switch (uc->getType())
                    {
                    case shadermod::ir::UserConstDataType::kBool:
                        uc->getValue(boolVal);
                        strstreamUC << "\nbool";
                        strstreamUC << (boolVal == shadermod::ir::userConstTypes::Bool::kTrue ? "\nTrue\n" : "\nFalse\n");
                        break;
                    case shadermod::ir::UserConstDataType::kInt:
                        uc->getValue(intVal);
                        strstreamUC << "\nint\n" << intVal << "\n";
                        break;
                    case shadermod::ir::UserConstDataType::kUInt:
                        uc->getValue(uintVal);
                        strstreamUC << "\nuint\n" << uintVal << "\n";
                        break;
                    case shadermod::ir::UserConstDataType::kFloat:
                        uc->getValue(floatVal);
                        strstreamUC << "\nfloat\n" << floatVal << "\n";
                        break;
                    }
                }
            }
            else
            {
                strstreamUC << 0 << "\n";
            }
        }

        if (numFilters)
        {
            ret.packedUserConstants = packedUserConstants.str();
            ret.effectNames = effectNames.str();
            ret.specialFxEnabled = isAtLeastOneEffectOnStack;
        }
    }

    ret.isInIPCMode = getIPCModeEnabled() != 0;
    ret.sessionDuration = getSessionDuration();
    ret.captureDuration = getCaptureDuration();
    ret.highresEnhancementEnabled = m_enableEnhancedHighres;

    AnselSDKStateTelemetry telem;
    m_anselSDK.getTelemetryData(telem);
    ret.usedGamepadForCameraDuringTheSession = telem.usedGamepadForCameraDuringTheSession;

    return ret;
}

HRESULT AnselServer::makeStateSnapshotforTelemetry(AnselStateForTelemetry & state) const
{
    try
    {
        state = makeStateSnapshotforTelemetryBody();
        return S_OK;
    }
    catch (...)
    {
        // Write to log, but otherwise do nothing
        LOG_WARN("Unhandled exception in snapshot generation");
        return E_FAIL;
    }
}

double AnselServer::getSessionDuration() const
{
    return m_globalPerfCounters.elapsedTime - m_sessionStartTime;
}

double AnselServer::getCaptureDuration() const
{
    return m_globalPerfCounters.elapsedTime - m_captureStartTime;
}

bool AnselServer::isTelemetryAvailable() const
{
    if (!m_allowTelemetry)
        return false;

    if (m_telemetryRetryAttempts >= TELEMETRY_INIT_RETRY_COUNT)
        return false;

    if (!m_telemetryInitialized)
    {
        LOG_DEBUG("Attempting to initialize telemetry...");
        HRESULT ok = LwTelemetry::Ansel::Init();

        if (SUCCEEDED(ok))
        {
            LOG_INFO("Telemetry initialized!");
            m_telemetryInitialized = true;
        }
        else
        {
            LOG_WARN("Telemetry initialization failed...");
            m_telemetryRetryAttempts = TELEMETRY_INIT_RETRY_COUNT; //if we fail to init, don't try to send telemetry

            return false;
        }
    }

    return true;
}

void AnselServer::destroyTelemetry() const
{
    LOG_DEBUG("Checking to destroy telemetry...");
    if (m_telemetryInitialized)
    {
        LOG_INFO("Destroying telemetry...");
        HRESULT ok = LwTelemetry::Ansel::DeInit();

        assert(SUCCEEDED(ok));

        m_telemetryInitialized = false;
        m_telemetryRetryAttempts = 0;
        LOG_INFO("Telemetry destroyed.");
    }
}


static LwTelemetry::Ansel::UserConstantTypeEnum colwertIrUserConstTypeToTelemetryType(const shadermod::ir::UserConstDataType& v)
{
    LwTelemetry::Ansel::UserConstantTypeEnum ret = LwTelemetry::Ansel::UserConstantTypeEnum::BOOL;

    switch (v)
    {
    case shadermod::ir::UserConstDataType::kBool:
        ret = LwTelemetry::Ansel::UserConstantTypeEnum::BOOL;
        break;
    case shadermod::ir::UserConstDataType::kInt:
        ret = LwTelemetry::Ansel::UserConstantTypeEnum::INT;
        break;
    case shadermod::ir::UserConstDataType::kUInt:
        ret = LwTelemetry::Ansel::UserConstantTypeEnum::UINT;
        break;
    case shadermod::ir::UserConstDataType::kFloat:
        ret = LwTelemetry::Ansel::UserConstantTypeEnum::FLOAT;
        break;
    case shadermod::ir::UserConstDataType::NUM_ENTRIES:
        ret = LwTelemetry::Ansel::UserConstantTypeEnum::INT;
        break;
    default:
        assert(false && "unsupported type colwersion!");
    }

    return ret;
}

static LwTelemetry::Ansel::KindSliderEnum colwertSliderKindToTemetryType(ShotType kindAnsel)
{
    LwTelemetry::Ansel::KindSliderEnum kind = LwTelemetry::Ansel::KindSliderEnum::REGULAR;

    switch (kindAnsel)
    {
    case ShotType::kNone:
        kind = LwTelemetry::Ansel::KindSliderEnum::NONE;
        break;
    case ShotType::kRegular:
        kind = LwTelemetry::Ansel::KindSliderEnum::REGULAR;
        break;
    case ShotType::kRegularUI:
        kind = LwTelemetry::Ansel::KindSliderEnum::REGULAR_UI;
        break;
    case ShotType::kHighRes:
        kind = LwTelemetry::Ansel::KindSliderEnum::HIGHRES;
        break;
    case ShotType::k360:
        kind = LwTelemetry::Ansel::KindSliderEnum::MONO_360;
        break;
    case ShotType::kStereo:
        kind = LwTelemetry::Ansel::KindSliderEnum::STEREO;
        break;
    case ShotType::k360Stereo:
        kind = LwTelemetry::Ansel::KindSliderEnum::STEREO_360;
        break;
    default:
        kind = LwTelemetry::Ansel::KindSliderEnum::REGULAR;
        assert(false && "Unknown shot type colwersion!");
    }

    return kind;
}

static LwTelemetry::Ansel::ColorRangeType colwertSliderKindToTelemetryColorRangeType(bool isShotHDR)
{
    LwTelemetry::Ansel::ColorRangeType ret = LwTelemetry::Ansel::ColorRangeType::RGB;

    if (isShotHDR)
        ret = LwTelemetry::Ansel::ColorRangeType::EXR;

    return ret;
}

static LwTelemetry::Ansel::AnselCaptureState colwertCaptureStateToTelemetryType(int captureState)
{
    using namespace LwTelemetry::Ansel;

    LwTelemetry::Ansel::AnselCaptureState state = AnselCaptureState::CAPTURE_STATE_NOT_STARTED;

    switch (captureState)
    {
    case CAPTURE_NOT_STARTED:
        state = AnselCaptureState::CAPTURE_STATE_NOT_STARTED;
        break;
    case CAPTURE_ABORT:
        state = AnselCaptureState::CAPTURE_STATE_ABORT;
        break;
    case CAPTURE_STARTED:
        state = AnselCaptureState::CAPTURE_STATE_STARTED;
        break;
    case CAPTURE_HIGHRES:
        state = AnselCaptureState::CAPTURE_STATE_HIGHRES;
        break;
    case CAPTURE_REGULAR:
        state = AnselCaptureState::CAPTURE_STATE_REGULAR;
        break;
    case CAPTURE_REGULARSTEREO:
        state = AnselCaptureState::CAPTURE_STATE_REGULARSTEREO;
        break;
    case CAPTURE_360:
        state = AnselCaptureState::CAPTURE_STATE_360;
        break;
    case CAPTURE_360STEREO:
        state = AnselCaptureState::CAPTURE_STATE_360STEREO;
        break;
    default:
        assert(false && "Unknown shot type colwersion!");
    }

    return state;
}

static LwTelemetry::Ansel::GamepadMappingType colwertGamepadDeviceTypeToTelemetryType(input::GamepadDevice::EGamepadDevice devType)
{
    using namespace LwTelemetry::Ansel;
    using namespace input;

    LwTelemetry::Ansel::GamepadMappingType ret = GamepadMappingType::UNKNOWN;

    switch (devType)
    {
    case GamepadDevice::EGamepadDevice::kShield:
        ret = GamepadMappingType::SHIELD;
        break;
    case GamepadDevice::EGamepadDevice::kDualShock4:
        ret = GamepadMappingType::DUALSHOCK4;
        break;
    case GamepadDevice::EGamepadDevice::kXbox360:
        ret = GamepadMappingType::XBOX360;
        break;
    case GamepadDevice::EGamepadDevice::kXboxOne:
        ret = GamepadMappingType::XBOXONE;
        break;
    case GamepadDevice::EGamepadDevice::kUnknown:
        ret = GamepadMappingType::UNKNOWN;
        break;
    default:
        assert(false && "Unknown gamepad device type colwersion!");
    }

    return ret;
}

void AnselServer::sendTelemetryMakeSnapshotEvent(const AnselStateForTelemetry& state) const
{
    if (!isTelemetryAvailable())
        return;

    using namespace shadermod;
    using namespace LwTelemetry;

    HRESULT ok = E_FAIL;
    LOG_DEBUG("Sending Telemetry: Make Snapshot Event...");
    try
    {
        ok = Ansel::Send_CaptureStarted_Event(
            state.exeName,
            state.rawProfileName,
            state.drsAppName,
            state.width,
            state.height,
            state.colorBufferFormat,
            state.depthBufferFormat,
            colwertSliderKindToTemetryType(state.uiSpecificData.kindOfShot),
            state.uiSpecificData.highresMult,
            state.fov360,
            state.uiSpecificData.fov,
            state.uiSpecificData.roll,
            state.dispCam_position_x,
            state.dispCam_position_y,
            state.dispCam_position_z,
            state.dispCam_rotation_x,
            state.dispCam_rotation_y,
            state.dispCam_rotation_z,
            state.dispCam_rotation_w,
            state.origCam_position_x,
            state.origCam_position_y,
            state.origCam_position_z,
            state.origCam_rotation_x,
            state.origCam_rotation_y,
            state.origCam_rotation_z,
            state.origCam_rotation_w,
            state.specialFxEnabled ? Ansel::SpecialEffectsModeEnum::YAML : Ansel::SpecialEffectsModeEnum::NONE,
            state.effectNames,
            (state.top5Var[0].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[0].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[0].type),
            state.top5Var[0].value,
            (state.top5Var[1].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[1].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[1].type),
            state.top5Var[1].value,
            (state.top5Var[2].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[2].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[2].type),
            state.top5Var[2].value,
            (state.top5Var[3].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[3].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[3].type),
            state.top5Var[3].value,
            (state.top5Var[4].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[4].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[4].type),
            state.top5Var[4].value,
            state.packedUserConstants,
            state.isInIPCMode ? Ansel::UIModeType::IPC_UI : Ansel::UIModeType::STANDALONE_UI,
            colwertSliderKindToTelemetryColorRangeType(state.uiSpecificData.isShotHDR),
            state.uiSpecificData.resolution360,
            colwertGamepadDeviceTypeToTelemetryType(state.gamepadDeviceType),
            state.gamepadProductId,
            state.gamepadVendorId,
            state.gamepadVersionNumber,
            state.usedGamepadForCameraDuringTheSession,
            state.uiSpecificData.usedGamepadForUIDuringTheSession,
            state.sessionDuration,
            state.highresEnhancementEnabled,
            state.anselSDKMajor,
            state.anselSDKMinor,
            state.anselSDKCommit,
            ANSEL_FILEVERSION_STRING,
            TELEMETRY_USER_ID
        );
    }
    catch(...)
    {
        ok = E_FAIL;
    }

    if (FAILED(ok))
    {
        LOG_DEBUG("Sending Telemetry FAILED: Make Snapshot Event");
        m_telemetryRetryAttempts++;
    }

    assert(SUCCEEDED(ok));
}

void AnselServer::sendTelemetryAbortCaptureEvent(const AnselStateForTelemetry& state) const
{
    if (!isTelemetryAvailable())
        return;

    using namespace shadermod;
    using namespace LwTelemetry;

    HRESULT ok = E_FAIL;

    LOG_DEBUG("Sending Telemetry: Abort Capture Event...");
    try
    {
        ok = Ansel::Send_CaptureAborted_Event(
            state.exeName,
            state.rawProfileName,
            state.drsAppName,
            state.width,
            state.height,
            state.colorBufferFormat,
            state.depthBufferFormat,
            colwertSliderKindToTemetryType(state.uiSpecificData.kindOfShot),
            state.uiSpecificData.highresMult,
            state.fov360,
            state.uiSpecificData.fov,
            state.uiSpecificData.roll,
            state.dispCam_position_x,
            state.dispCam_position_y,
            state.dispCam_position_z,
            state.dispCam_rotation_x,
            state.dispCam_rotation_y,
            state.dispCam_rotation_z,
            state.dispCam_rotation_w,
            state.origCam_position_x,
            state.origCam_position_y,
            state.origCam_position_z,
            state.origCam_rotation_x,
            state.origCam_rotation_y,
            state.origCam_rotation_z,
            state.origCam_rotation_w,
            state.specialFxEnabled ? Ansel::SpecialEffectsModeEnum::YAML : Ansel::SpecialEffectsModeEnum::NONE,
            state.effectNames,
            (state.top5Var[0].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[0].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[0].type),
            state.top5Var[0].value,
            (state.top5Var[1].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[1].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[1].type),
            state.top5Var[1].value,
            (state.top5Var[2].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[2].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[2].type),
            state.top5Var[2].value,
            (state.top5Var[3].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[3].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[3].type),
            state.top5Var[3].value,
            (state.top5Var[4].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[4].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[4].type),
            state.top5Var[4].value,
            state.packedUserConstants,
            state.isInIPCMode ? Ansel::UIModeType::IPC_UI : Ansel::UIModeType::STANDALONE_UI,
            colwertSliderKindToTelemetryColorRangeType(state.uiSpecificData.isShotHDR),
            state.uiSpecificData.resolution360,
            state.captureDuration,
            state.highresEnhancementEnabled,
            ANSEL_FILEVERSION_STRING,
            TELEMETRY_USER_ID);
    }
    catch (...)
    {
        ok = E_FAIL;
    }

    if (FAILED(ok))
    {
        LOG_DEBUG("Sending Telemetry FAILED: Abort Capture Event");
        m_telemetryRetryAttempts++;
    }

    assert(SUCCEEDED(ok));
}

void AnselServer::sendTelemetryStyleTransferDownloadStartedEvent(
    const std::string& url,
    const std::string& version,
    uint32_t computeCapMajor,
    uint32_t computeCapMinor) const
{
    if (!isTelemetryAvailable())
        return;

    using namespace shadermod;
    using namespace LwTelemetry;

    HRESULT ok = E_FAIL;
    LOG_DEBUG("Sending Telemetry: Style Transfer Download Started Event...");
    try
    {
        ok = Ansel::Send_StyleTransferDownloadStarted_Event(url, version, computeCapMajor, computeCapMinor, ANSEL_FILEVERSION_STRING, TELEMETRY_USER_ID);
    }
    catch (...)
    {
        ok = E_FAIL;
    }

    if (FAILED(ok))
    {
        LOG_DEBUG("Sending Telemetry FAILED: Style Transfer Download Started Event");
        m_telemetryRetryAttempts++;
    }

    assert(SUCCEEDED(ok));
}

void AnselServer::sendTelemetryStyleTransferDownloadFinishedEvent(uint32_t secondsSpent, int32_t status) const
{
    if (!isTelemetryAvailable())
        return;

    using namespace shadermod;
    using namespace LwTelemetry;

    HRESULT ok = E_FAIL;
    LOG_DEBUG("Sending Telemetry: Style Transfer Download Finished Event...");
    try
    {
        ok = Ansel::Send_StyleTransferDownloadFinished_Event(secondsSpent, status, ANSEL_FILEVERSION_STRING, TELEMETRY_USER_ID);
    }
    catch (...)
    {
        ok = E_FAIL;
    }

    if (FAILED(ok))
    {
        LOG_DEBUG("Sending Telemetry FAILED: Style Transfer Download Finished Event");
        m_telemetryRetryAttempts++;
    }

    assert(SUCCEEDED(ok));
}

void AnselServer::sendTelemetryStyleTransferStatusEvent(const uint32_t reason, const std::string& comment) const
{
    if (!isTelemetryAvailable())
        return;

    using namespace shadermod;
    using namespace LwTelemetry;

    HRESULT ok = E_FAIL;
    LOG_DEBUG("Sending Telemetry: Style Transfer Status Event...");
    try
    {
        ok = Ansel::Send_StyleTransferStatus_Event(Ansel::StyleTransferStatusEnum(reason), comment, ANSEL_FILEVERSION_STRING, TELEMETRY_USER_ID);
    }
    catch (...)
    {
        ok = E_FAIL;
    }

    if (FAILED(ok))
    {
        LOG_DEBUG("Sending Telemetry FAILED: Style Transfer Status Event");
        m_telemetryRetryAttempts++;
    }

    assert(SUCCEEDED(ok));
}

void AnselServer::sendTelemetryCloseUIEvent(const AnselStateForTelemetry& state) const
{
    if (!isTelemetryAvailable())
        return;

    using namespace shadermod;
    using namespace LwTelemetry;

    HRESULT ok = E_FAIL;
    LOG_DEBUG("Sending Telemetry:  Close UI Event...");
    try
    {
        ok = Ansel::Send_AnselUIClosed_Event(
            state.exeName,
            state.rawProfileName,
            state.drsAppName,
            state.width,
            state.height,
            state.colorBufferFormat,
            state.depthBufferFormat,
            colwertSliderKindToTemetryType(state.uiSpecificData.kindOfShot),
            state.uiSpecificData.highresMult,
            state.fov360,
            state.uiSpecificData.fov,
            state.uiSpecificData.roll,
            state.dispCam_position_x,
            state.dispCam_position_y,
            state.dispCam_position_z,
            state.dispCam_rotation_x,
            state.dispCam_rotation_y,
            state.dispCam_rotation_z,
            state.dispCam_rotation_w,
            state.origCam_position_x,
            state.origCam_position_y,
            state.origCam_position_z,
            state.origCam_rotation_x,
            state.origCam_rotation_y,
            state.origCam_rotation_z,
            state.origCam_rotation_w,
            state.specialFxEnabled ? Ansel::SpecialEffectsModeEnum::YAML : Ansel::SpecialEffectsModeEnum::NONE,
            state.effectNames,
            (state.top5Var[0].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[0].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[0].type),
            state.top5Var[0].value,
            (state.top5Var[1].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[1].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[1].type),
            state.top5Var[1].value,
            (state.top5Var[2].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[2].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[2].type),
            state.top5Var[2].value,
            (state.top5Var[3].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[3].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[3].type),
            state.top5Var[3].value,
            (state.top5Var[4].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[4].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[4].type),
            state.top5Var[4].value,
            state.packedUserConstants,
            state.isInIPCMode ? Ansel::UIModeType::IPC_UI : Ansel::UIModeType::STANDALONE_UI,
            colwertSliderKindToTelemetryColorRangeType(state.uiSpecificData.isShotHDR),
            state.uiSpecificData.resolution360,
            colwertGamepadDeviceTypeToTelemetryType(state.gamepadDeviceType),
            state.gamepadProductId,
            state.gamepadVendorId,
            state.gamepadVersionNumber,
            state.usedGamepadForCameraDuringTheSession,
            state.uiSpecificData.usedGamepadForUIDuringTheSession,
            state.sessionDuration,
            state.highresEnhancementEnabled,
            state.anselSDKMajor,
            state.anselSDKMinor,
            state.anselSDKCommit,
            ANSEL_FILEVERSION_STRING,
            TELEMETRY_USER_ID
        );
    }
    catch (...)
    {
        ok = E_FAIL;
    }

    if (FAILED(ok))
    {
        LOG_DEBUG("Sending Telemetry FAILED:  Close UI Event");
        m_telemetryRetryAttempts++;
    }

    assert(SUCCEEDED(ok));
}

void AnselServer::sendTelemetryLwCameraErrorEvent(const AnselStateForTelemetry& state, int captureState, const ErrorDescForTelemetry& errDesc, bool isFatal) const
{
    if (!isTelemetryAvailable())
        return;

    using namespace shadermod;
    using namespace LwTelemetry;

    HRESULT ok = E_FAIL;
    LOG_DEBUG("Sending Telemetry:  LwCamera Error Event...");
    try
    {
        ok = Ansel::Send_AnselErrorOclwredFull_Event(
            state.exeName,
            state.rawProfileName,
            state.drsAppName,
            state.width,
            state.height,
            state.colorBufferFormat,
            state.depthBufferFormat,
            colwertSliderKindToTemetryType(state.uiSpecificData.kindOfShot),
            state.uiSpecificData.highresMult,
            state.fov360,
            state.uiSpecificData.fov,
            state.uiSpecificData.roll,
            state.dispCam_position_x,
            state.dispCam_position_y,
            state.dispCam_position_z,
            state.dispCam_rotation_x,
            state.dispCam_rotation_y,
            state.dispCam_rotation_z,
            state.dispCam_rotation_w,
            state.origCam_position_x,
            state.origCam_position_y,
            state.origCam_position_z,
            state.origCam_rotation_x,
            state.origCam_rotation_y,
            state.origCam_rotation_z,
            state.origCam_rotation_w,
            state.specialFxEnabled ? Ansel::SpecialEffectsModeEnum::YAML : Ansel::SpecialEffectsModeEnum::NONE,
            state.effectNames,
            (state.top5Var[0].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[0].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[0].type),
            state.top5Var[0].value,
            (state.top5Var[1].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[1].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[1].type),
            state.top5Var[1].value,
            (state.top5Var[2].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[2].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[2].type),
            state.top5Var[2].value,
            (state.top5Var[3].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[3].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[3].type),
            state.top5Var[3].value,
            (state.top5Var[4].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[4].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[4].type),
            state.top5Var[4].value,
            state.packedUserConstants,
            state.isInIPCMode ? Ansel::UIModeType::IPC_UI : Ansel::UIModeType::STANDALONE_UI,
            colwertSliderKindToTelemetryColorRangeType(state.uiSpecificData.isShotHDR),
            state.uiSpecificData.resolution360,
            colwertGamepadDeviceTypeToTelemetryType(state.gamepadDeviceType),
            state.gamepadProductId,
            state.gamepadVendorId,
            state.gamepadVersionNumber,
            state.usedGamepadForCameraDuringTheSession,
            state.uiSpecificData.usedGamepadForUIDuringTheSession,
            state.sessionDuration,
            state.captureDuration,
            colwertCaptureStateToTelemetryType(captureState),
            isFatal ? Ansel::ErrorType::HANDLE_FAILURE_FATAL_ERROR : Ansel::ErrorType::NON_FATAL_ERROR,
            errDesc.filename,
            errDesc.lineNumber,
            errDesc.errorMessage,
            errDesc.errorCode,
            state.highresEnhancementEnabled,
            state.anselSDKMajor,
            state.anselSDKMinor,
            state.anselSDKCommit,
            ANSEL_FILEVERSION_STRING,
            TELEMETRY_USER_ID
        );
    }
    catch (...)
    {
        ok = E_FAIL;
    }

    if (FAILED(ok))
    {
        LOG_DEBUG("Sending Telemetry FAILED:  LwCamera Error Event");
        m_telemetryRetryAttempts++;
    }

    assert(SUCCEEDED(ok));

    // Errors can potentially oclwrr at any time and we need to make sure telemetry is destroyed before the program exits.
    destroyTelemetry();
}

void AnselServer::sendTelemetryLwCameraErrorShortEvent(const ErrorDescForTelemetry& errDesc, bool isFatal) const
{
    if (!isTelemetryAvailable())
        return;

    using namespace shadermod;
    using namespace LwTelemetry;

    HRESULT ok = E_FAIL;
    LOG_DEBUG("Sending Telemetry: Error Short Event...");
    try
    {
        ok = Ansel::Send_AnselErrorOclwredShort_Event(
            isFatal ? Ansel::ErrorType::HANDLE_FAILURE_FATAL_ERROR : Ansel::ErrorType::NON_FATAL_ERROR,
            errDesc.filename,
            errDesc.lineNumber,
            errDesc.errorMessage,
            errDesc.errorCode,
            ANSEL_FILEVERSION_STRING,
            TELEMETRY_USER_ID
        );
    }
    catch (...)
    {
        ok = E_FAIL;
    }

    if (FAILED(ok))
    {
        LOG_DEBUG("Sending Telemetry FAILED: Error Short Event");
        m_telemetryRetryAttempts++;
    }

    assert(SUCCEEDED(ok));

    // Errors can potentially oclwrr at any time and we need to make sure telemetry is destroyed before the program exits.
    destroyTelemetry();
}

void AnselServer::sendTelemetryEffectCompilationErrorEvent(const AnselStateForTelemetry& state, int captureState, const ErrorDescForTelemetry& errDesc) const
{
    if (!isTelemetryAvailable())
        return;

    using namespace shadermod;
    using namespace LwTelemetry;

    HRESULT ok = E_FAIL;
    LOG_DEBUG("Sending Telemetry: Effect Compilation Error Event...");
    try
    {
        ok = Ansel::Send_AnselErrorOclwredFull_Event(
            state.exeName,
            state.rawProfileName,
            state.drsAppName,
            state.width,
            state.height,
            state.colorBufferFormat,
            state.depthBufferFormat,
            colwertSliderKindToTemetryType(state.uiSpecificData.kindOfShot),
            state.uiSpecificData.highresMult,
            state.fov360,
            state.uiSpecificData.fov,
            state.uiSpecificData.roll,
            state.dispCam_position_x,
            state.dispCam_position_y,
            state.dispCam_position_z,
            state.dispCam_rotation_x,
            state.dispCam_rotation_y,
            state.dispCam_rotation_z,
            state.dispCam_rotation_w,
            state.origCam_position_x,
            state.origCam_position_y,
            state.origCam_position_z,
            state.origCam_rotation_x,
            state.origCam_rotation_y,
            state.origCam_rotation_z,
            state.origCam_rotation_w,
            state.specialFxEnabled ? Ansel::SpecialEffectsModeEnum::YAML : Ansel::SpecialEffectsModeEnum::NONE,
            state.effectNames,
            (state.top5Var[0].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[0].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[0].type),
            state.top5Var[0].value,
            (state.top5Var[1].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[1].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[1].type),
            state.top5Var[1].value,
            (state.top5Var[2].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[2].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[2].type),
            state.top5Var[2].value,
            (state.top5Var[3].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[3].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[3].type),
            state.top5Var[3].value,
            (state.top5Var[4].type != ir::UserConstDataType::NUM_ENTRIES ? Ansel::UserConstantSliderStateEnum::CREATED_VISIBLE_ENABLED :
                Ansel::UserConstantSliderStateEnum::NOT_CREATED),
            state.top5Var[4].name,
            colwertIrUserConstTypeToTelemetryType(state.top5Var[4].type),
            state.top5Var[4].value,
            state.packedUserConstants,
            state.isInIPCMode ? Ansel::UIModeType::IPC_UI : Ansel::UIModeType::STANDALONE_UI,
            colwertSliderKindToTelemetryColorRangeType(state.uiSpecificData.isShotHDR),
            state.uiSpecificData.resolution360,
            colwertGamepadDeviceTypeToTelemetryType(state.gamepadDeviceType),
            state.gamepadProductId,
            state.gamepadVendorId,
            state.gamepadVersionNumber,
            state.usedGamepadForCameraDuringTheSession,
            state.uiSpecificData.usedGamepadForUIDuringTheSession,
            state.sessionDuration,
            state.captureDuration,
            colwertCaptureStateToTelemetryType(captureState),
            Ansel::ErrorType::EFFECT_COMPILATION_ERROR,
            errDesc.filename,
            errDesc.lineNumber,
            errDesc.errorMessage,
            errDesc.errorCode,
            state.highresEnhancementEnabled,
            state.anselSDKMajor,
            state.anselSDKMinor,
            state.anselSDKCommit,
            ANSEL_FILEVERSION_STRING,
            TELEMETRY_USER_ID
        );
    }
    catch (...)
    {
        ok = E_FAIL;
    }

    if (FAILED(ok))
    {
        LOG_DEBUG("Sending Telemetry FAILED: Effect Compilation Error Event");
        m_telemetryRetryAttempts++;
    }

    assert(SUCCEEDED(ok));

    // Errors can potentially oclwrr at any time and we need to make sure telemetry is destroyed before the program exits.
    destroyTelemetry();
}

void AnselServer::reportFatalError(const char* filename, int lineNumber, FatalErrorCode code, const char* format, va_list args) const
{
    ErrorDescForTelemetry desc;

    desc.errorCode = unsigned int(code);
    desc.lineNumber = lineNumber;
    desc.filename = filename;

    int numChars = _vscprintf(format, args);

    if (numChars >= 0) //-1 is returnd if parsing failed
    {
        std::vector<char> formattedText; //we use a vector because c functions add zero character at the end of the string
        formattedText.resize(numChars + 1);
        vsnprintf_s(formattedText.data(), formattedText.size(), _TRUNCATE, format, args);

        LOG_ERROR(formattedText.data());

        desc.errorMessage = formattedText.data();
        if (m_activeControlClient)
            m_activeControlClient->reportFatalError(code, desc.filename, lineNumber, desc.errorMessage);
    }

    //in case the large event preparaton crashes us, at least we send the quick event..
    sendTelemetryLwCameraErrorShortEvent(desc);

    AnselStateForTelemetry state;
    HRESULT telemetryStatus = makeStateSnapshotforTelemetry(state);
    if (telemetryStatus == S_OK)
        sendTelemetryLwCameraErrorEvent(state, m_anselSDK.getCaptureState(), desc);
}

void AnselServer::reportNonFatalError(const char* filename, int lineNumber, unsigned int code, const char* format, va_list args) const
{
    ErrorDescForTelemetry desc;

    desc.errorCode = code;
    desc.lineNumber = lineNumber;
    desc.filename = filename;

    int numChars = _vscprintf(format, args);

    if (numChars >= 0) //-1 is returnd if parsing failed
    {
        std::vector<char> formattedText; //we use a vector because c functions add zero character at the end of the string
        formattedText.resize(numChars + 1);
        vsnprintf_s(formattedText.data(), formattedText.size(), _TRUNCATE, format, args);

        LOG_WARN(formattedText.data());

        desc.errorMessage = formattedText.data();
        m_activeControlClient->reportNonFatalError(code, desc.filename, lineNumber, desc.errorMessage);
    }

    //in case the large event preparaton crashes us, at least we send the quick event..
    sendTelemetryLwCameraErrorShortEvent(desc, false);

    AnselStateForTelemetry state;
    HRESULT telemetryStatus = makeStateSnapshotforTelemetry(state);
    if (telemetryStatus == S_OK)
        sendTelemetryLwCameraErrorEvent(state, m_anselSDK.getCaptureState(), desc, false);
}


void AnselServer::reportFatalError(const char* filename, int lineNumber, FatalErrorCode code, const char* format, ...) const
{
    va_list args;
    va_start(args, format);

    reportFatalError(filename, lineNumber, code, format, args);

    va_end(args);
}

void AnselServer::reportNonFatalError(const char* filename, int lineNumber, unsigned int code, const char* format, ...) const
{
    va_list args;
    va_start(args, format);

    reportNonFatalError(filename, lineNumber, code, format, args);

    va_end(args);
}


//****************************************************************************
// DX12 Interfaces
//****************************************************************************
HRESULT AnselServer::notifyPresentBake12(HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList, void ** ppServerData)
{
    // Return non-NULL value in ppServerData to get a call to
    // ExelwtePostProcessing12.  Just set a non-NULL value for now, but in the
    // future will probably want to pass more complicated messages to
    // ExelwtePostProcessing12, such as the current device state, which is only
    // readable during a *Bake12 function.
    *ppServerData = (void *)0x1;
    return S_OK;
}

HRESULT AnselServer::exelwtePostProcessing12CanFail(ANSEL_EXEC_DATA *pAnselExecData, HCLIENTRESOURCE hPresentResource, DWORD subResIndex)
{
    AnselAutoCS autoCS(&m_csExec);
    // Set ExecData pointer as it is used in AcquireSharedResources12/ReleaseSharedResources12
    m_pLwrrentExecData = pAnselExecData;
    HRESULT status = S_OK;
    setBufferInterfaceD3D12();
    m_d3d12Interface.setLwrrentExecDataPointer(pAnselExecData);
    if (!SUCCEEDED(status = finalizeFrame(hPresentResource, subResIndex)))
    {
        LOG_ERROR("Frame processing error 12");
    }
    return status;
}

HRESULT AnselServer::exelwtePostProcessing12(ANSEL_EXEC_DATA *pAnselExecData, HCLIENTRESOURCE hPresentResource, DWORD subResIndex)
{
    static bool kaboomHappenned = false;

    if (kaboomHappenned)
        return S_OK;


#if (DBG_EXCEPTIONS_PASSTHROUGH == 1)
#else
    __try
    {
#endif

        HRESULT status = S_OK;
        if (!SUCCEEDED(status = exelwtePostProcessing12CanFail(pAnselExecData, hPresentResource, subResIndex)))
        {
            reportFatalError(__FILE__, __LINE__, FatalErrorCode::kFrameProcessFail, "Deactivating LwCamera due to frame processing error (D3D12)");
            deactivateAnsel();
        }

        if (!m_lightweightShimMode || m_bNotifyRMAnselStop)
        {
            notifyFrameComplete(m_bNotifyRMAnselStop);
            m_bNotifyRMAnselStop = false;
        }

        return S_OK;


#if (DBG_EXCEPTIONS_PASSTHROUGH == 1)
#else
    }
    __except (EXCEPTION_EXELWTE_HANDLER)
    {
        kaboomHappenned = true;

        // We're telling the game that the session should be ended to unpause it if needed
        if (m_anselSDK.isSDKDetectedAndSessionActive())
            m_anselSDK.stopClientSession();
        // We're telling teh UMD to put Ansel shim in "passthrough" mode (no Ansel-related activity)
        reportFatalError(__FILE__, __LINE__, FatalErrorCode::kOtherException, "Deactivating LwCamera due to generic exception (D3D12)");
        deactivateAnsel(false);
        notifyFrameComplete(true);
    }
#endif

    return S_OK;
}

HRESULT AnselServer::notifyCmdListCreate12(HCMDLIST hCmdList, HANSELCMDLIST *hAnselCmdList)
{
    *hAnselCmdList = new AnselCommandList();
    return S_OK;
}

HRESULT AnselServer::notifyCmdListDestroy12(HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList)
{
    AnselCommandList *anselCommandList = static_cast<AnselCommandList*>(hAnselCmdList);
    delete anselCommandList;
    hAnselCmdList = NULL;
    return S_OK;
}

HRESULT AnselServer::notifyCmdListReset12(HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList)
{
    if (hAnselCmdList)
    {
        AnselCommandList *anselCommandList = static_cast<AnselCommandList*>(hAnselCmdList);
        anselCommandList->Reset();
    }
    return S_OK;
}

HRESULT AnselServer::notifyCmdListClose12(HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList)
{
    return S_OK;
}

HRESULT AnselServer::notifySetRenderTargetBake12(HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList, void ** ppServerData)
{
    *ppServerData = NULL;

    HRESULT status = S_OK;
    AnselDeviceStates deviceStates;
    if (!SUCCEEDED(status = m_pClientFunctionTable->GetDeviceStates12(m_hClient, hCmdList, &deviceStates)))
    {
        HandleFailure();
    }
    return notifySetRenderTargetBake12Common(deviceStates, hAnselCmdList, ppServerData);
}

HRESULT AnselServer::notifySetRenderTargetBakeWithDeviceStates12(const AnselDeviceStates deviceStates, HANSELCMDLIST hAnselCmdList, void ** ppServerData)
{
    *ppServerData = NULL;
    return notifySetRenderTargetBake12Common(deviceStates, hAnselCmdList, ppServerData);
}

HRESULT AnselServer::notifySetRenderTargetBake12Common(const AnselDeviceStates& deviceStates, HANSELCMDLIST hAnselCmdList, void ** ppServerData)
{
    if (!m_lwrrentBufferInterface)
    {
        setBufferInterfaceD3D12();
    }

    *ppServerData = NULL;

    HRESULT status = S_OK;

    // We won't get bind notifications anymore in DX12. Must read current RT and
    // depth buffer from deviceStates.
    HCLIENTRESOURCE hLwrrentDSBuffer = NULL;
    HCLIENTRESOURCE hSelectedHUDless = NULL;
    HCLIENTRESOURCE hSelectedHDR = NULL;

    if (m_enableDepthExtractionChecks)
    {
        bool depthBufferMatches;
        m_bufDB.Depth().checkBuffer(deviceStates.hLwrrentDSBuffer, deviceStates, getDepthWidth(), getDepthHeight(), &depthBufferMatches);
        if (depthBufferMatches)
        {
            hLwrrentDSBuffer = deviceStates.hLwrrentDSBuffer;
        }
    }

    if (m_enableHDRExtractionChecks && deviceStates.hLwrrentRTZero)
    {
        bool hdrBufferMatches;
        m_bufDB.HDR().checkBuffer(deviceStates.hLwrrentRTZero, getWidth(), getHeight(), &hdrBufferMatches);
        if (hdrBufferMatches)
        {
            hSelectedHDR = deviceStates.hLwrrentRTZero;
        }
    }
    else if (m_enableHUDlessExtractionChecks)
    {
        bool hudlessBufferMatches;
        m_bufDB.Hudless().checkBuffer(deviceStates.hLwrrentRTZero, deviceStates, &hudlessBufferMatches);
        if (hudlessBufferMatches)
        {
            hSelectedHUDless = deviceStates.hLwrrentRTZero;
        }
    }
    if (hLwrrentDSBuffer || hSelectedHUDless || hSelectedHDR)
    {
        // Defer depth buffer selection or HUDless buffer copying to NotifyDrawExec12
        AnselBufferSnapshot * pBufferSnapshot = new AnselBufferSnapshot();
        if (!pBufferSnapshot)
        {
            return E_OUTOFMEMORY;
        }
        pBufferSnapshot->hLwrrentDSBuffer = hLwrrentDSBuffer;
        pBufferSnapshot->hSelectedHUDless = hSelectedHUDless;
        pBufferSnapshot->hSelectedHDR = hSelectedHDR;
        *ppServerData = pBufferSnapshot;

        AnselCommandList *anselCommandList = static_cast<AnselCommandList*>(hAnselCmdList);
        anselCommandList->AddSnapshot(pBufferSnapshot);
    }

    return S_OK;
}

HRESULT AnselServer::notifySetRenderTargetExec12(ANSEL_EXEC_DATA *pAnselExecData)
{
    if (!m_lwrrentBufferInterface)
    {
        setBufferInterfaceD3D12();
    }

    HRESULT status = S_OK;

    AnselBufferSnapshot * pBufferSnapshot = static_cast<AnselBufferSnapshot *>(pAnselExecData->pServerData);

    AnselAutoCS autoCS(&m_csExec);

    m_d3d12Interface.setLwrrentExecDataPointer(pAnselExecData);

    if (m_depthBufferUsed)
    {
        m_bufDB.Depth().selectBuffer(pBufferSnapshot->hLwrrentDSBuffer);
    }

    if (pBufferSnapshot->hSelectedHUDless)
    {
        if (m_hudlessBufferUsed)
        {
            m_bufDB.Hudless().selectBuffer(pBufferSnapshot->hSelectedHUDless);
        }
    }
    else
    {
        // Copy the HUDless buffer immediately after another, non-HUDless render target is being set instead of it.
        m_bufDB.Hudless().copyResource(0);
    }
    if (pBufferSnapshot->hSelectedHDR)
    {
        m_hdrBufferAvailable = true;
        if (isHDRBufferUsed())
        {
            m_bufDB.HDR().selectBuffer(pBufferSnapshot->hSelectedHDR);
        }
    }
    return status;
}

HRESULT AnselServer::notifyDepthStencilClearBake12(HCMDLIST hCmdList, HANSELCMDLIST hAnselCmdList, HCLIENTRESOURCE hDepthStencil, void ** pServerData)
{
    if (!m_lwrrentBufferInterface)
    {
        setBufferInterfaceD3D12();
    }

    if (m_enableDepthExtractionChecks)
    {
        *pServerData = (void *)hDepthStencil;
    }
    return S_OK;
}
HRESULT AnselServer::notifyDepthStencilClearExec12(ANSEL_EXEC_DATA *pAnselExecData)
{
    if (!m_lwrrentBufferInterface)
    {
        setBufferInterfaceD3D12();
    }

    HRESULT status = S_OK;
    HCLIENTRESOURCE hClientResource = static_cast<HCLIENTRESOURCE>(pAnselExecData->pServerData);
    AnselAutoCS autoCS(&m_csExec);
    if (m_enableDepthExtractionChecks)
    {
        m_d3d12Interface.setLwrrentExecDataPointer(pAnselExecData);

        if (hClientResource == m_bufDB.Depth().getClientResource())
        {
            m_bufDB.Depth().copyResource(0);
        }
    }
    return S_OK;
}

void AnselServer::deactivateAnsel(bool doCleanup)
{
    LOG_DEBUG("Deactivating Ansel, and %scleaning up...", doCleanup ? "" : "*NOT* ");
    m_pClientFunctionTable->DisableClient(m_hClient);
    if (doCleanup)
        destroy();
    else
        m_bNotifyRMAnselStop = true;
}

void AnselServer::enterLightweightMode()
{
    m_pClientFunctionTable->EnterLightweightMode(m_hClient);
    m_bNotifyRMAnselStop = true;
}
void AnselServer::exitLightweightMode()
{
    m_pClientFunctionTable->ExitLightweightMode(m_hClient);
}

HRESULT AnselServer::notifyFrameComplete(bool deactivatingAnsel)
{
    // Start with latest supported version, and track which one worked. 0 disables attempts to notify OPTP
    static UINT32 version = 2;

    LwAPI_Status ret = LWAPI_ERROR;

    switch (version)
    {
        // OPTP is not supported on this driver; do nothing
        case 0:
            return S_OK;

        // Try latest version in case the driver supports it. Lwrrently, this is explicitly functional for version 2;
        // if newer versions are introduced that need parameters modified, a new case-statement will need
        // to be added. Otherwise, future versions should continue to work.
        case 2:
        {
            LW_FRAME_PRESENT_NOTIFY_PARAMS params = { 0 };
            params.version = LW_FRAME_PRESENT_NOTIFY_PARAMS_VER;
            params.bIsAnsel = true;
            params.bStop = deactivatingAnsel;

            ret = LwAPI_D3D_FramePresentNotify(m_d3dDevice, &params);
            if (ret != LWAPI_INCOMPATIBLE_STRUCT_VERSION)
            {
                break;
            }

            // If v2 is incompatible, intentionally fall-through and try v1
            version = 1;
        }

        // Try version 1 in case the driver supports it
        case 1:
        {
            LW_FRAME_PRESENT_NOTIFY_PARAMS params = { 0 };
            params.version = LW_FRAME_PRESENT_NOTIFY_PARAMS_VER1;
            params.bStop = deactivatingAnsel;

            ret = LwAPI_D3D_FramePresentNotify(m_d3dDevice, &params);
            break;
        }

        // Unsupported version; there is a bug if this happens
        default:
            assert(!"AnselServer::notifyFrameComplete unknown version!");
            version = 0;
            return E_FAIL;
    }

    // This is true in drivers before R440
    if (ret == LWAPI_NO_IMPLEMENTATION)
    {
        version = 0;
        return S_OK;
    }
    else if (ret != LWAPI_OK)
    {
        LwAPI_ShortString szDesc = { 0 };
        LwAPI_GetErrorMessage(ret, szDesc);
        LOG_ERROR("LWAPI Error: %s", szDesc);
        return E_FAIL;
    }

    return S_OK;
}

void AnselServer::disconnectIpc()
{
#if IPC_ENABLED == 1
    if (m_UI)
    {
        if (getIPCModeEnabled() != 0)
        {
            UIIPC * UI_IPC = static_cast<UIIPC *>(m_UI);
            UI_IPC->disconnectIpc();
        }
    }
#endif
}

void AnselServer::SaveHashes()
{
#if _DEBUG && DBG_EMIT_HASHES
    // Create dummy values
    shadermod::ir::Effect::InputData finalColorInput, depthInput, hudlessInput, hdrInput, colorBaseInput;

    finalColorInput.width = getWidth();
    finalColorInput.height = getHeight();
    finalColorInput.format = shadermod::ir::FragmentFormat::kNUM_ENTRIES;
    finalColorInput.texture = nullptr;

    depthInput.width = getDepthWidth();
    depthInput.height = getDepthHeight();
    depthInput.format = shadermod::ir::FragmentFormat::kNUM_ENTRIES;
    depthInput.texture = nullptr;

    hudlessInput.width = getWidth();
    hudlessInput.height = getHeight();
    hudlessInput.format = shadermod::ir::FragmentFormat::kNUM_ENTRIES;
    hudlessInput.texture = nullptr;

    hdrInput.width = getWidth();
    hdrInput.height = getHeight();
    hdrInput.format = shadermod::ir::FragmentFormat::kNUM_ENTRIES;
    hdrInput.texture = nullptr;

    colorBaseInput.width = getWidth();
    colorBaseInput.height = getHeight();
    colorBaseInput.format = shadermod::ir::FragmentFormat::kNUM_ENTRIES;
    colorBaseInput.texture = nullptr;

    // Needed to indicate that hashes should be created, although they won't be compared
    const std::set<Hash::Effects> dummyHash;

    ModdingEffectHashDB::EffectsMap refEffects;

    // Loop through all effects and append their hashes to the new HashedFiles.cpp
    for (size_t effIdx = 0; effIdx < m_effectsInfo.m_effectFilesList.size(); ++effIdx)
    {
        const std::wstring& effectRootDirectory = m_effectsInfo.m_effectRootFoldersList[effIdx];
        const std::wstring& effNameW = m_effectsInfo.m_effectFilesList[effIdx];
        std::string effName(darkroom::getUtf8FromWstr(effNameW));
        //AnselServer::PrepackagedEffects effEnum = EffectStringToEnum(effName);
        const auto& effEnum = s_effectStringToEnum.find(effName);

        // Some effects do not need hash callwlations, these effects will not be found in the map
        if (effEnum == s_effectStringToEnum.end())
            continue;

        // TODO: It would be better if we could just perform the hashing rather than actually
        //       initializing each effect
        shadermod::MultiPassEffect eff(
            m_installationFolderPath.c_str(), effectRootDirectory.c_str(), m_intermediateFolderPath.c_str(),
            effNameW.c_str(),
            m_fxExtensionToolMap,
            finalColorInput, depthInput, hudlessInput, hdrInput, colorBaseInput,
            m_d3dDevice, &m_d3dCompiler,
            &dummyHash, false);


        //Build up refEffects to hold the effects to build the main hash later
        std::set<Hash::Effects> effHash;
        Hash::Data acefHash = Hash::UnshiftHash(eff.getCallwlatedHashes().GetHash(Hash::Type::Acef));
        Hash::Data resHash = Hash::UnshiftHash(eff.getCallwlatedHashes().GetHash(Hash::Type::Resource));
        Hash::Data shaderHash = Hash::UnshiftHash(eff.getCallwlatedHashes().GetHash(Hash::Type::Shader));

        effHash.emplace(acefHash, resHash, shaderHash, effName);
        refEffects.emplace(effEnum->second, effHash);
    }

    // We create a new HashedFiles.cpp that overwrites the old one to update the hashes
    std::stringstream newHashedFilesCpp;
    newHashedFilesCpp << "#include \"AnselServer.h\"\n\n" <<
                         "const std::map<AnselServer::PrepackagedEffects, std::set<Hash::Effects>> AnselServer::ModdingEffectHashDB::m_effHashes =\n{\n";

    uint32_t effCounter = 0;
    for (const auto& itEffHashes : refEffects)
    {
        effCounter++;
        for (const auto& itHash : itEffHashes.second)
        {
            std::string effName_WithUnderscores = itHash.GetEffName();
            for (UINT idx = 0; idx < effName_WithUnderscores.length(); idx++)
            {
                if (effName_WithUnderscores[idx] == '.')
                {
                    effName_WithUnderscores[idx] = '_';
                }
            }

            newHashedFilesCpp << "\t{ // " << itHash.GetEffName() <<
                                 "\n\t\tAnselServer::PrepackagedEffects::k" << effName_WithUnderscores <<
                                 ",\n\t\t{\n\t\t\t{ // v0" <<
                                 "\n\t\t\t/* ACEF     */" << Hash::FormatHash(itHash.GetHash(Hash::Type::Acef)) <<
                                 ",\n\t\t\t/* Resource */" << Hash::FormatHash(itHash.GetHash(Hash::Type::Resource)) <<
                                 ",\n\t\t\t/* Shader   */" << Hash::FormatHash(itHash.GetHash(Hash::Type::Shader)) <<
                                 ",\n\t\t\t/* EffName  */\"" << itHash.GetEffName() << "\"" <<
                                 "\n\t\t\t}\n\t\t}\n\t}";

            // Last effect
            if (effCounter == refEffects.size())
                newHashedFilesCpp << "\n";
            else
                newHashedFilesCpp << ",\n";
        }
    }

    // Write the rest of the contents of HashedFiles.cpp to the new HashedFiles.cpp, including the new main hash
    newHashedFilesCpp << "};\n\nconst Hash::Data AnselServer::ModdingEffectHashDB::m_mainHash = " << Hash::FormatHash(m_moddingEffectHashDB.GenerateMainHash(refEffects))
                        << ";\n\n// These need to be defined here so that their static intialization oclwrs after the hash intialization above\n" <<
                        "const Hash::Data AnselServer::ModdingEffectHashDB::m_generatedMainHash = GenerateMainHash(m_effHashes);\n" <<
                        "const AnselServer::ModdingEffectHashDB::EffectsMap AnselServer::ModdingEffectHashDB::m_effShiftedHashes = GenerateShiftedHashSet();\n";

    std::ofstream out("HashedFiles.cpp");
    out << newHashedFilesCpp.str();
    out.close();
#endif // _DEBUG && DBG_EMIT_HASHES
}

bool AnselServer::ModdingEffectHashDB::CompareMainHash(const Hash::Data& hash)
{
    return m_mainHash == hash;
}

const Hash::Data AnselServer::ModdingEffectHashDB::GenerateMainHash(const ModdingEffectHashDB::EffectsMap &refEffects)
{
    Hash::Data verifHash;

    sha256_ctx verifHashState;
    sha256_init(&verifHashState);

    for (const auto& itEffHashes : refEffects)
    {
        for (const auto& itHash : itEffHashes.second)
        {
            sha256_update(&verifHashState, itHash.GetHash(Hash::Type::Acef).data(), SHA256_DIGEST_SIZE);
            sha256_update(&verifHashState, itHash.GetHash(Hash::Type::Resource).data(), SHA256_DIGEST_SIZE);
            sha256_update(&verifHashState, itHash.GetHash(Hash::Type::Shader).data(), SHA256_DIGEST_SIZE);
        }
    }
    sha256_final(&verifHashState, verifHash.data());

    return verifHash;
}

const Hash::Data AnselServer::ModdingEffectHashDB::GetGeneratedMainHash()
{
    return m_generatedMainHash;
}

bool AnselServer::ModdingEffectHashDB::GetGeneratedShiftedHashSet(PrepackagedEffects effIdx, std::set<Hash::Effects>& retHash)
{
    retHash.clear();

    // kNUM_ENTRIES indicates the filter is unrestricted and we should return an empty shifted hash
    if (effIdx == PrepackagedEffects::kNUM_ENTRIES)
    {
        retHash.insert(Hash::s_emptyShiftedEffects);
        return false;
    }

    assert(effIdx < PrepackagedEffects::kNUM_ENTRIES);

    const auto& itHashSet = m_effShiftedHashes.find(effIdx);
    if (itHashSet == m_effShiftedHashes.end())
    {
        assert(!"Unknown PrepackagedEffects!");
        return false;
    }

    retHash = itHashSet->second;

    return true;
}

const AnselServer::ModdingEffectHashDB::EffectsMap AnselServer::ModdingEffectHashDB::GenerateShiftedHashSet()
{
    EffectsMap hashMap;

    for (const auto& itEffHashes : m_effHashes)
    {
        // Loop through each hash in the set, adding the shift to each one, then inserting them
        // into hashMap
        for(const auto& itHash : itEffHashes.second)
        {
            Hash::Effects newHash(Hash::s_emptyShiftedEffects);
            newHash.Add(itHash);

            const auto& itInserted = hashMap[itEffHashes.first].insert(newHash);

            // The inserted hashes should always be unique; assert if not
            assert(itInserted.second);
        }
    }

    return hashMap;
}

